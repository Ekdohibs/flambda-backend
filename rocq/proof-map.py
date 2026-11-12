#!/usr/bin/env python3
"""proof-map.py [options]

Draws a map of the Rocq development as one PDF (rocq/proof-map.pdf): what
is specification, what is implementation, what is proof, how the three are
kept apart, which theorems join them, and what a reviewer has to read.

It reads the .v sources only (no build is needed), so it also works on a
tree that is half built. Where .glob files of a build are present and newer
than their sources, `--check-globs` compares what this script takes each
definition to refer to with what Rocq recorded.

How it reads a development
--------------------------
Every file is cut into its vernacular commands, and each command that
names something (Definition, Fixpoint, Inductive, Record, Theorem, Lemma,
Axiom, ...) becomes an item. An item has two sets of references:

  * what its *statement* mentions: the type of a theorem; the type and the
    body of a definition (a definition is transparent, so its body is part
    of what it means);
  * what its *proof* mentions (between Proof and Qed).

Only the first matters to what a theorem says. A reviewer who trusts Rocq
reads a theorem's statement and everything reached from it through
statements, its cone, and never a proof.

A claim is a `Definition X_statement : Prop` with the `Theorem X :
X_statement` that proves it. A claim whose proof is used (through any
chain of lemmas) by the proof of another claim is a lemma of that claim;
the claims that no other claim uses are the results. Reading the cones of
the results is enough.

The cone of a result is itself cut in two. A function that the statement
names directly is its subject: the statement holds of whatever the
function computes (`check x = true -> good (f x)` is true whatever check
and f are), so the function, and what is reached from it through functions
only, need not be read for the theorem to mean what it says. The rest is
the meaning of the statement, and has to be read: the propositions and
the types it names, and all that they are defined with. The last page of
the PDF says where this cut can be wrong.

Requires python3, graphviz (`dot`) and `lualatex`.
"""

import argparse
import bisect
import collections
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

LIBRARIES = [
    # (directory, library name, short title)
    ("rocq", "OxCamlStdlib", "OxCaml's own types"),
    ("backend/rocq", "OxCamlBackend", "Backend-wide"),
    ("backend/ssa/rocq", "OxCamlSsa", "SSA graph"),
    ("backend/ssa/loop/rocq", "OxCamlLoop", "SSA loop passes"),
    ("middle_end/flambda2/rocq", "OxCamlFlambda2", "Flambda 2 terms"),
    ("middle_end/flambda2/types/rocq", "OxCamlFlambda2Types",
     "Flambda 2 types"),
]

DEFAULT_FOCUS = ["OxCamlFlambda2", "OxCamlFlambda2Types"]

# --------------------------------------------------------------------------
# Reading the sources
# --------------------------------------------------------------------------

THEOREM_WORDS = {
    "Theorem", "Lemma", "Corollary", "Proposition", "Fact", "Remark",
    "Example",
}
DEF_WORDS = {
    "Definition", "Fixpoint", "CoFixpoint", "Function", "Instance", "Let",
    "Abbreviation",
}
IND_WORDS = {
    "Inductive", "CoInductive", "Variant", "Record", "Structure", "Class",
}
AXIOM_WORDS = {"Axiom", "Parameter", "Conjecture", "Axioms", "Parameters"}
HYP_WORDS = {"Hypothesis", "Hypotheses", "Variable", "Variables", "Context"}

HEAD_RE = re.compile(
    r"\s*(?:(?:Local|Global|Polymorphic|Monomorphic|Program|Equations"
    r"|#\[[^\]]*\])\s+)*"
    r"([A-Z][A-Za-z]*)\b\s*([A-Za-z_][A-Za-z0-9_']*)?")
IDENT_RE = re.compile(
    r"[A-Za-z_][A-Za-z0-9_']*(?:\.[A-Za-z_][A-Za-z0-9_']*)*")
SENTENCE_END_RE = re.compile(r"\.(?=\s|$)")
PROOF_END = {"Qed", "Defined", "Admitted", "Abort", "Save"}

KEYWORDS = set("""
forall fun exists exists2 match with end let in if then else return as fix
cofix struct Prop Set Type SProp at using where for of
""".split())


def blank_comments(text):
    """Replace comments and strings by spaces, keeping every offset."""
    out = list(text)
    i, n, depth = 0, len(text), 0
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == '"':
                in_string = False
            elif c != "\n":
                out[i] = " "
            i += 1
        elif depth == 0 and c == '"':
            in_string = True
            i += 1
        elif c == "(" and i + 1 < n and text[i + 1] == "*":
            depth += 1
            out[i] = out[i + 1] = " "
            i += 2
        elif depth and c == "*" and i + 1 < n and text[i + 1] == ")":
            depth -= 1
            out[i] = out[i + 1] = " "
            i += 2
        else:
            if depth and c != "\n":
                out[i] = " "
            i += 1
    return "".join(out)


def split_top(text, sep):
    """Offsets of [sep] in [text] outside brackets."""
    found, depth, i, n = [], 0, 0, len(text)
    while i < n:
        c = text[i]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif depth == 0 and text.startswith(sep, i):
            found.append(i)
            i += len(sep)
            continue
        i += 1
    return found


BINDER_RES = [
    re.compile(r"[({]\s*((?:[A-Za-z_][\w']*\s+)*[A-Za-z_][\w']*)\s*:[^=]"),
    re.compile(r"(?:\bforall|\bfun|\bexists2?|\xe2\x88\x80|\xe2\x88\x83"
               r"|\xce\xbb)\s+((?:[A-Za-z_][\w']*\s+)*[A-Za-z_][\w']*)\s*"
               r"(?:,|=>|:)"),
    re.compile(r"\bintros\s+((?:[A-Za-z_][\w']*\s*)+)"),
    re.compile(r"\b(?:let|set|pose|remember)\s*\(?\s*([A-Za-z_][\w']*)\s*:="),
]


def binders(text):
    """Names that [text] binds locally (roughly)."""
    out = set()
    for r in BINDER_RES:
        for m in r.finditer(text):
            out.update(m.group(1).split())
    return out


class Item:
    __slots__ = (
        "id", "file", "name", "qual", "word", "kind", "is_prop", "start",
        "stmt_end", "end", "line", "stmt_lines", "proof_lines", "names",
        "sdeps", "pdeps", "hyps", "module", "stmt_text", "opaque",
        "admitted", "extracted", "local", "is_type")

    def __repr__(self):
        return "<%s %s.%s>" % (self.kind, self.file.module, self.qual)


class File:
    def __init__(self, path, library, libtitle):
        self.path = path
        self.rel = os.path.relpath(path, ROOT)
        self.library = library
        self.libtitle = libtitle
        self.module = os.path.basename(path)[:-2]
        m = self.module
        if m.endswith("Spec"):
            self.role, self.stem = "spec", m[:-4]
        elif m.endswith("Proofs"):
            self.role, self.stem = "proofs", m[:-6]
        elif m.endswith("Model"):
            self.role, self.stem = "model", m
        elif m.endswith("Extract"):
            self.role, self.stem = "extract", m
        else:
            self.role, self.stem = "impl", m
        self.is_examples = "Example" in m
        with open(path, encoding="latin-1") as f:
            self.raw = f.read()
        self.text = blank_comments(self.raw)
        self.lines = self.raw.count("\n") + 1
        self.line_starts = [0] + [
            m.end() for m in re.finditer("\n", self.raw)]
        self.requires = []      # module names, in order
        self.imports = set()    # modules opened with Import
        self.items = []
        self.hyps = []          # (start, end, Item) of section hypotheses
        self.extract_roots = []  # names given to Extraction commands
        self.closure = None
        self.code_lines = sum(
            1 for l in self.text.split("\n") if l.strip())
        self.header = self._header()

    def _header(self):
        """The first sentence of the comment that follows the licence."""
        m = re.search(r"\*{20,}\)\s*\(\*(.*?)\*\)", self.raw, re.S)
        if not m:
            return ""
        s = " ".join(m.group(1).split())
        s = s.encode("latin-1").decode("utf-8", "replace")
        cut = re.search(r"\.(\s|$)", s)
        return s[:cut.start() + 1] if cut else s

    def qual_index(self):
        if not hasattr(self, "_qual"):
            self._qual = {}
            for it in self.items:
                if it.kind == "hyp":
                    continue
                prefix = it.module + "." if it.module else ""
                for nm in it.names:
                    self._qual.setdefault(prefix + nm, it)
        return self._qual

    def line_of(self, off):
        return bisect.bisect_right(self.line_starts, off)


def count_lines(text):
    return sum(1 for l in text.split("\n") if l.strip())


def parse_file(f):
    text = f.text
    sentences = []
    pos = 0
    for m in SENTENCE_END_RE.finditer(text):
        sentences.append((pos, m.end()))
        pos = m.end()
    modules = []       # stack of (name, is_module)
    sections = []      # stack of (name, start offset, [hyp items])
    i = 0
    n = len(sentences)

    def new_item(word, name, start, stmt_end, end):
        it = Item()
        it.file, it.name, it.word = f, name, word
        it.module = ".".join(m for m in modules)
        it.qual = (it.module + "." if it.module else "") + name
        it.start, it.stmt_end, it.end = start, stmt_end, end
        it.line = f.line_of(start + len(text[start:]) -
                            len(text[start:].lstrip()))
        it.stmt_text = text[start:stmt_end]
        it.stmt_lines = count_lines(it.stmt_text)
        it.proof_lines = count_lines(text[stmt_end:end])
        it.names = [name]
        it.sdeps, it.pdeps, it.hyps = set(), set(), []
        it.is_prop = False
        it.opaque = False
        it.admitted = False
        it.extracted = False
        it.local = False
        it.is_type = False
        it.kind = "def"
        return it

    while i < n:
        s, e = sentences[i]
        sent = text[s:e]
        stripped = sent.lstrip()
        m = HEAD_RE.match(sent)
        i += 1
        if not m:
            continue
        word, name = m.group(1), m.group(2)
        if word in ("Require", "From"):
            mm = re.search(r"Require\s+(?:Import\s+|Export\s+)?(.*)\.$",
                           stripped, re.S)
            if mm:
                for tok in mm.group(1).split():
                    f.requires.append(tok.split(".")[-1])
            continue
        if word == "Module" and name and name != "Type" \
                and ":=" not in sent:
            modules.append(name)
            continue
        if word == "Section" and name:
            sections.append((name, s, []))
            continue
        if word in ("Import", "Export"):
            f.imports.update(
                t.split(".")[-1] for t in stripped[:-1].split()[1:])
            continue
        if word == "End" and name:
            if sections and sections[-1][0] == name:
                _, start, hyps = sections.pop()
                for h in hyps:
                    f.hyps.append((start, e, h))
            elif modules and modules[-1] == name:
                modules.pop()
            continue
        if word in ("Extraction", "Separate", "Recursive") and \
                f.role == "extract":
            body = re.sub(r"^\s*(Separate|Recursive)?\s*Extraction"
                          r"(\s+Library)?", "", sent)
            if not re.match(r"\s*(Language|Inline|NoInline|Blacklist"
                            r"|Implicit|TestCompile)", body):
                f.extract_roots += IDENT_RE.findall(body[:-1])
            continue
        if not name:
            continue
        if word in HYP_WORDS:
            # Variable a b : T.  Hypothesis H : P.  Context (x : T) ...
            head = sent
            colon = split_top(head, ":")
            names = []
            if word == "Context":
                names = re.findall(r"[({]\s*([A-Za-z_][\w' ]*?)\s*:", head)
                names = [x for grp in names for x in grp.split()]
            elif colon:
                names = IDENT_RE.findall(head[m.start(2):colon[0]])
            for nm in names or [name]:
                it = new_item(word, nm, s, e, e)
                it.kind = "hyp"
                it.is_prop = word.startswith("Hypothes")
                it.local = True
                if sections:
                    sections[-1][2].append(it)
                    f.items.append(it)
            continue
        if word in AXIOM_WORDS:
            it = new_item(word, name, s, e, e)
            it.kind = "axiom"
            f.items.append(it)
            continue
        is_thm = word in THEOREM_WORDS
        if not (is_thm or word in DEF_WORDS or word in IND_WORDS):
            continue
        # Is there a proof script after the head?
        has_body = bool(split_top(sent, ":="))
        end = e
        proof = False
        if i < n:
            nxt = text[sentences[i][0]:sentences[i][1]].strip()
            if nxt.startswith("Proof"):
                proof = True
            elif not has_body and (is_thm or word in DEF_WORDS):
                proof = True
        admitted = False
        opaque = False
        if proof:
            j = i
            while j < n:
                t = text[sentences[j][0]:sentences[j][1]].strip()
                w = t[:-1].split(None, 1)[0] if t[:-1].strip() else ""
                if w in PROOF_END:
                    end = sentences[j][1]
                    opaque = w in ("Qed", "Admitted", "Save")
                    admitted = w == "Admitted"
                    break
                if w == "Proof" and t[:-1].strip() != "Proof" and \
                        not t.startswith("Proof with") and \
                        not t.startswith("Proof using"):
                    # Proof term.
                    end = sentences[j][1]
                    opaque = True
                    break
                j += 1
            else:
                end = sentences[-1][1]
            i = j + 1
        it = new_item(word, name, s, e, end)
        it.opaque = opaque
        it.admitted = admitted
        head = sent[:split_top(sent, ":=")[0]] if has_body else sent[:-1]
        it.is_prop = bool(re.search(
            r":\s*(.*(->|\xe2\x86\x92)\s*)?(Prop|relation\s+[\w'.]+)\s*$",
            head, re.S))
        it.is_type = bool(re.search(
            r":\s*(.*(->|\xe2\x86\x92)\s*)?(Type|Set)\s*$", head, re.S))
        if word in IND_WORDS:
            it.kind = "inductive"
            names = re.findall(r"\|\s*([A-Za-z_][\w']*)", sent)
            names += re.findall(r":=\s*([A-Za-z_][\w']*)", sent)
            if word in ("Record", "Structure", "Class"):
                names += re.findall(r"[{;]\s*([A-Za-z_][\w']*)\s*[:(]",
                                    sent)
            names += re.findall(r"^\s*with\s+([A-Za-z_][\w']*)", sent,
                                re.M)
            it.names += [x for x in names if x not in KEYWORDS]
            for suffix in ("_ind", "_rect", "_rec", "_sind"):
                it.names.append(name + suffix)
        elif (is_thm and (proof and opaque or not has_body)) or \
                (proof and opaque and word in DEF_WORDS):
            it.kind = "theorem"
            it.is_prop = True
        else:
            it.kind = "def"
            if name.endswith("_statement"):
                it.kind = "statement"
                it.is_prop = True
            it.names += re.findall(r"^\s*with\s+([A-Za-z_][\w']*)", sent,
                                   re.M)
            if word == "Function":
                it.names += [name + s2 for s2 in
                             ("_ind", "_equation", "_rect", "_rec")]
        if word == "Let" and sections:
            it.local = True
            sections[-1][2].append(it)
        f.items.append(it)


# --------------------------------------------------------------------------
# The development: files, items, references
# --------------------------------------------------------------------------

class Development:
    def __init__(self, focus):
        self.files = []
        self.by_module = {}
        for d, lib, title in LIBRARIES:
            full = os.path.join(ROOT, d)
            if not os.path.isdir(full):
                continue
            for fn in sorted(os.listdir(full)):
                if fn.endswith(".v"):
                    f = File(os.path.join(full, fn), lib, title)
                    self.files.append(f)
                    self.by_module[f.module] = f
        for f in self.files:
            parse_file(f)
        self.items = []
        for f in self.files:
            for it in f.items:
                it.id = len(self.items)
                self.items.append(it)
        self.focus = set(focus)
        self._closures()
        self._index()
        self._resolve_all()

    def _closures(self):
        for f in self.files:
            f.direct = [self.by_module[m] for m in f.requires
                        if m in self.by_module and m != f.module]

        def closure(f, seen):
            if f.closure is not None:
                return f.closure
            if f in seen:
                return set()
            seen.add(f)
            c = set()
            for g in f.direct:
                c.add(g)
                c |= closure(g, seen)
            f.closure = c
            return c
        sys.setrecursionlimit(10000)
        for f in self.files:
            closure(f, set())

    def _index(self):
        self.by_name = collections.defaultdict(list)
        for it in self.items:
            if it.local:
                continue
            for nm in it.names:
                self.by_name[nm].append(it)

    def resolve(self, f, token, pos):
        """The items that [token], met at offset [pos] of [f], may name."""
        parts = token.split(".")
        short = parts[-1]
        out = []
        if len(parts) == 1:
            for (s, e, h) in f.hyps:
                if s <= pos < e and short in h.names and h.start < pos:
                    out.append(h)
            if out:
                return out[-1:]
        cands = self.by_name.get(short)
        if not cands:
            return out
        if len(parts) > 1:
            prefix = parts[:-1]
            keep = []
            for it in cands:
                path = [it.file.module] + (
                    it.module.split(".") if it.module else [])
                if path[-len(prefix):] == prefix or \
                        path[1:][-len(prefix):] == prefix:
                    keep.append(it)
            cands = keep
            if not cands:
                return out
        if len(parts) == 1:
            # A name inside a module is seen unqualified from inside that
            # module, or where the module is imported.
            here = None
            keep = []
            for it in cands:
                if not it.module:
                    keep.append(it)
                elif it.module.split(".")[-1] in f.imports:
                    keep.append(it)
                elif it.file is f:
                    if here is None:
                        here = self.module_at(f, pos)
                    if (here + ".").startswith(it.module + "."):
                        keep.append(it)
            cands = keep
        same = [it for it in cands if it.file is f and it.start <= pos]
        if same:
            # the nearest one above, or the item itself (recursion)
            return [max(same, key=lambda it: it.start)]
        vis = [it for it in cands if it.file in f.closure]
        direct = [it for it in vis if it.file in f.direct]
        if len(direct) > 1:
            # Rocq takes the one required last.
            order = {g: k for k, g in enumerate(f.direct)}
            last = max(order[it.file] for it in direct)
            direct = [it for it in direct if order[it.file] == last]
        return direct or vis

    def module_at(self, f, pos):
        best = ""
        for it in f.items:
            if it.start > pos:
                break
            best = it.module
        return best

    def refs(self, f, start, end, bound=None):
        found = set()
        bound = set() if bound is None else bound
        for m in IDENT_RE.finditer(f.text, start, end):
            tok = m.group(0)
            if tok in KEYWORDS or tok in bound:
                continue
            for it in self.resolve(f, tok, m.start()):
                found.add(it)
        return found

    def _resolve_all(self):
        for f in self.files:
            for it in f.items:
                bound = binders(f.text[it.start:it.end])
                bound.difference_update(it.names)
                s = self.refs(f, it.start, it.stmt_end, bound)
                p = self.refs(f, it.stmt_end, it.end, bound)
                s.discard(it)
                p.discard(it)
                if it.kind == "hyp":
                    it.sdeps = s
                    continue
                # A section hypothesis that the proof uses is a premise.
                hyps = {x for x in p if x.local}
                it.sdeps = s | hyps
                it.pdeps = p - hyps - it.sdeps
                if not it.opaque and it.end > it.stmt_end:
                    # Defined: the proof is the body.
                    it.sdeps |= it.pdeps
                    it.pdeps = set()

    def check_globs(self):
        """Compare the references read from the sources with Rocq's."""
        libs = {lib for _, lib, _ in LIBRARIES}
        tot = collections.Counter()
        missed = collections.Counter()
        extra = collections.Counter()
        nfiles = 0
        for f in self.files:
            g = f.path[:-2] + ".glob"
            if not os.path.exists(g) or \
                    os.path.getmtime(g) < os.path.getmtime(f.path):
                continue
            nfiles += 1
            starts = [it.start for it in f.items]
            by_qual = {}
            glob_s = collections.defaultdict(set)
            glob_p = collections.defaultdict(set)
            for line in open(g, encoding="latin-1"):
                if not line.startswith("R"):
                    continue
                parts = line.split()
                if len(parts) != 5 or parts[4] in (
                        "lib", "var", "not", "binder", "mod", "sec",
                        "modtype", "abbrev"):
                    if not (len(parts) == 5 and parts[4] == "abbrev"):
                        continue
                lib, _, mod = parts[1].rpartition(".")
                if lib not in libs or mod not in self.by_module:
                    continue
                off = int(parts[0][1:].split(":")[0])
                k = bisect.bisect_right(starts, off) - 1
                if k < 0:
                    continue
                src = f.items[k]
                if off >= src.end or src.kind == "hyp":
                    continue
                tf = self.by_module[mod]
                qual = ("" if parts[2] == "<>" else parts[2] + ".") \
                    + parts[3]
                tgt = tf.qual_index().get(qual)
                if tgt is None:
                    tot["unknown target"] += 1
                    continue
                if tgt is src:
                    continue
                if off < src.stmt_end or not src.opaque:
                    glob_s[src].add(tgt)
                else:
                    glob_p[src].add(tgt)
            for it in f.items:
                if it.kind == "hyp":
                    continue
                mine_s = {x for x in it.sdeps if x.kind != "hyp"}
                mine_p = {x for x in it.pdeps if x.kind != "hyp"}
                gs = glob_s[it]
                gp = glob_p[it] - gs
                for kind, mine, theirs in (("statement", mine_s, gs),
                                           ("proof", mine_p, gp)):
                    tot[kind, "rocq"] += len(theirs)
                    tot[kind, "script"] += len(mine)
                    tot[kind, "both"] += len(mine & theirs)
                    for x in theirs - mine:
                        missed[kind, it.file.module, it.qual, x.qual] += 1
                    for x in mine - theirs:
                        extra[kind, it.file.module, it.qual, x.qual] += 1
        lines = ["%d files with a .glob newer than the source" % nfiles]
        note = []
        for kind in ("statement", "proof"):
            r, s, b = (tot[kind, k] for k in ("rocq", "script", "both"))
            lines.append(
                "%-9s references: Rocq %d, this script %d, both %d "
                "(found %.1f%%, right %.1f%%)"
                % (kind, r, s, b, 100.0 * b / max(r, 1),
                   100.0 * b / max(s, 1)))
            note.append(
                "of the %s references from %ss that Rocq recorded, %.1f"
                "\\%% were found here, and %.1f\\%% of those found here "
                "are Rocq's" % ("{:,}".format(r).replace(",", "\\,"),
                                kind, 100.0 * b / max(r, 1),
                                100.0 * b / max(s, 1)))
        self.glob_summary = "\n".join(lines)
        if nfiles:
            self.glob_note = (
                "For the %d files that have a \\texttt{.glob} file of a "
                "build newer than the source, the references were "
                "compared with those that Rocq wrote there: %s. A "
                "reference found here and not by Rocq is mostly a bound "
                "variable with the name of a definition; it makes a cone "
                "larger, never smaller. Run with \\texttt{--check-globs "
                "-v} for the differences." % (nfiles, "; ".join(note)))
        else:
            self.glob_note = (
                "No \\texttt{.glob} file of a build was found, so the "
                "references could not be compared with Rocq's. Build "
                "with \\texttt{make -C rocq all} and run with "
                "\\texttt{--check-globs}.")
        return missed, extra

    def cone(self, roots):
        """Everything reached from [roots] through statements."""
        seen = set()
        todo = list(roots)
        while todo:
            it = todo.pop()
            if it in seen:
                continue
            seen.add(it)
            todo.extend(it.sdeps - seen)
        return seen


# --------------------------------------------------------------------------
# What is claimed, and what has to be read
# --------------------------------------------------------------------------

class Claim:
    def __init__(self, statement):
        self.statement = statement
        self.proof = None        # the theorem that proves it
        self.uses = set()        # claims its proof goes through
        self.used_by = set()
        self.must = set()        # items to read
        self.tests = set()       # items reached from tests only
        self.parts = []
        self.is_result = False

    @property
    def name(self):
        return self.statement.name[:-len("_statement")]


class Analysis:
    def __init__(self, dev):
        self.dev = dev
        self.focus = [f for f in dev.files if f.library in dev.focus]
        self.items = [it for f in self.focus for it in f.items]
        self._claims()
        self._results()
        self._cones()
        self._extracted()
        self._audit()

    # -- claims ----------------------------------------------------------

    def _claims(self):
        dev = self.dev
        self.claims = {}
        for it in self.items:
            if it.kind == "statement":
                self.claims[it] = Claim(it)
        self.prover = {}     # theorem -> claim
        by_name = collections.defaultdict(list)
        for it in self.items:
            if it.kind == "theorem":
                by_name[it.name].append(it)
        for c in self.claims.values():
            st = c.statement
            cands = [t for t in by_name.get(c.name, ())
                     if st.file in t.file.closure or t.file is st.file]
            named = [t for t in cands if st in t.sdeps]
            best = named or cands
            if best:
                best.sort(key=lambda t: (t.file.stem != st.file.stem,
                                         t.file.module))
                c.proof = best[0]
                self.prover[c.proof] = c
        # which claims the proof of a claim goes through
        memo = {}

        def claims_below(t, stack):
            """Claims met going down from theorem [t] through proofs,
            stopping at each claim."""
            if t in memo:
                return memo[t]
            memo[t] = set()
            out = set()
            for d in t.pdeps:
                if d.kind != "theorem" or d in stack:
                    continue
                c = self.prover.get(d)
                if c:
                    out.add(c)
                else:
                    out |= claims_below(d, stack | {d})
            memo[t] = out
            return out
        for c in self.claims.values():
            if c.proof:
                c.uses = claims_below(c.proof, {c.proof}) - {c}
        for c in self.claims.values():
            for u in c.uses:
                u.used_by.add(c)

    def _results(self):
        for c in self.claims.values():
            c.is_result = not c.used_by
        self.results = [c for c in self.claims.values() if c.is_result]
        # everything a claim rests on, transitively
        for c in self.claims.values():
            c.below = None

        def below(c, stack=()):
            if c.below is not None:
                return c.below
            c.below = set()
            out = set()
            for u in c.uses:
                out.add(u)
                out |= below(u)
            c.below = out
            return out
        for c in sorted(self.claims.values(),
                        key=lambda c: c.statement.id):
            below(c)

    # -- cones -----------------------------------------------------------

    def split_cone(self, roots):
        """The cone of [roots], cut in two: (meaning, subject).

        A function that a statement names directly is the subject of the
        statement: the statement holds of whatever the function computes,
        so its body adds nothing to what the statement means. What it
        calls is subject too. Everything else gives the statement its
        meaning: the propositions and the types it names, and all that
        they are defined with, functions included."""
        meaning, subject = set(), set()
        todo = [(it, True) for r in roots for it in r.sdeps]
        while todo:
            it, direct = todo.pop()
            if it in meaning:
                continue
            if direct and it.kind == "def" and not it.is_prop \
                    and not it.is_type:
                if it in subject:
                    continue
                subject.add(it)
                # an alias stands for what it abbreviates
                todo.extend((d, True) for d in it.sdeps)
            else:
                meaning.add(it)
                todo.extend((d, False) for d in it.sdeps)
        return meaning, subject - meaning

    def _cones(self):
        for c in self.claims.values():
            c.must, c.tests = self.split_cone([c.statement])
            c.direct = set(c.statement.sdeps)
            # A statement that names no proposition is made of equations
            # between functions and of nothing else: there the functions
            # are the meaning, all of them.
            c.equational = not any(
                it.is_prop and it.kind != "statement" for it in c.must)
            if c.equational:
                c.must |= c.tests
                c.tests = set()
        self.must = set()
        self.tests = set()
        self.reach = collections.Counter()   # item -> results reading it
        for c in self.results:
            self.must |= c.must
            self.tests |= c.tests
            for it in c.must:
                self.reach[it] += 1
        self.tests -= self.must
        self.lemma_cone = set()
        for c in self.claims.values():
            if not c.is_result:
                self.lemma_cone |= c.must | c.tests
        self.lemma_cone -= self.must | self.tests

    def status(self, it):
        if it in self.must:
            return "must"
        if it in self.tests:
            return "tests"
        if it.kind == "statement":
            return "result" if self.claims[it].is_result else "lemma"
        if it in self.lemma_cone:
            return "lemma"
        return "none"

    # -- extraction ------------------------------------------------------

    def _extracted(self):
        dev = self.dev
        roots = set()
        self.extract_files = [f for f in dev.files if f.role == "extract"]
        for f in self.extract_files:
            for tok in f.extract_roots:
                roots.update(dev.resolve(f, tok, len(f.text)))
        self.extract_roots = roots
        self.extracted = {it for it in dev.cone(roots)
                          if not it.is_prop and it.kind != "theorem"}
        for it in self.extracted:
            it.extracted = True

    # -- separation ------------------------------------------------------

    def _audit(self):
        self.require_matrix = collections.Counter()
        self.ref_matrix = collections.Counter()
        self.proof_matrix = collections.Counter()
        self.bad_requires = []
        self.bad_refs = []
        for f in self.focus:
            for g in f.direct:
                self.require_matrix[f.role, g.role] += 1
                if f.role != "proofs" and g.role == "proofs":
                    self.bad_requires.append((f, g))
            for it in f.items:
                for d in it.sdeps:
                    if d.file is f:
                        continue
                    self.ref_matrix[f.role, d.file.role] += 1
                    if f.role != "proofs" and d.file.role == "proofs":
                        self.bad_refs.append((it, d))
                for d in it.pdeps:
                    if d.file is not f:
                        self.proof_matrix[f.role, d.file.role] += 1


def lines_of(items):
    return sum(it.stmt_lines for it in items)


def dominators(root, succ):
    """Immediate dominators of what [root] reaches (Cooper, Harvey and
    Kennedy's iteration). [succ] gives the successors of a node."""
    order, seen = [], {root}
    stack = [(root, iter(succ(root)))]
    while stack:
        node, it = stack[-1]
        for nxt in it:
            if nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(succ(nxt))))
                break
        else:
            order.append(node)
            stack.pop()
    order.reverse()
    index = {n: i for i, n in enumerate(order)}
    preds = collections.defaultdict(list)
    for n in order:
        for m in succ(n):
            if m in index:
                preds[m].append(n)
    idom = {root: root}

    def meet(a, b):
        while a is not b:
            while index[a] > index[b]:
                a = idom[a]
            while index[b] > index[a]:
                b = idom[b]
        return a
    changed = True
    while changed:
        changed = False
        for n in order[1:]:
            new = None
            for q in preds[n]:
                if q in idom:
                    new = q if new is None else meet(q, new)
            if idom.get(n) is not new:
                idom[n] = new
                changed = True
    return idom


def greedy_cover(an, pool, limit):
    """Results in the order in which each covers the most claims that the
    ones before it leave uncovered."""
    covered, picks = set(), []
    pool = list(pool)
    while pool and len(picks) < limit:
        best = max(pool, key=lambda c: (len((c.below | {c}) - covered),
                                        -c.statement.id))
        gain = len((best.below | {best}) - covered)
        if gain <= 1 and picks:
            break
        covered |= best.below | {best}
        picks.append((best, gain, len(covered)))
        pool.remove(best)
    return picks


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------

PAGE_W, PAGE_H = 39.0, 26.7     # the drawing area of a page, in cm

COLOURS = {
    "ink": "1F2933", "soft": "52606D", "faint": "9AA5B1", "rule": "D9DEE4",
    "paper": "FFFFFF", "wash": "F5F7FA",
    "meaning": "2B5F9E", "meaningL": "D6E3F3",
    "subject": "D4881C", "subjectL": "F9E7C8",
    "proof": "A9B2BC", "proofL": "ECEFF2",
    "lemma": "8FB8D8", "lemmaL": "E4EEF6",
    "result": "1E8A6A", "resultL": "D3EDE4",
    "trust": "7B4EA3", "trustL": "E9DFF2",
    "alert": "C23B31", "alertL": "F8DCD9",
}

TEX_SPECIAL = {
    "\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "_": r"\_",
    "^": r"\^{}", "~": r"\~{}", "#": r"\#", "$": r"\$", "%": r"\%",
    "&": r"\&",
}


def esc(s):
    return "".join(TEX_SPECIAL.get(c, c) for c in str(s))


def code(s):
    """[s] escaped for the monospaced font, breakable at underscores."""
    return "".join(TEX_SPECIAL.get(c, c) + (r"\allowbreak{}" if c in "_."
                                            else "") for c in str(s))


def num(n):
    return "{:,}".format(int(round(n))).replace(",", r"\,")


def short(module):
    return module[7:] if module.startswith("Flambda") and \
        len(module) > 7 else module


class Page:
    """One page: a picture with its origin at the bottom left."""

    def __init__(self, title, subtitle=""):
        self.out = []
        self.text(0, PAGE_H, title, size=22, weight="bold",
                  anchor="north west")
        if subtitle:
            self.text(0, PAGE_H - 1.15, subtitle, size=10.5, colour="soft",
                      anchor="north west", width=PAGE_W)
        self.top = PAGE_H - (2.6 if subtitle else 1.6)

    def raw(self, s):
        self.out.append(s)

    def text(self, x, y, s, size=9, colour="ink", anchor="west",
             weight="", width=None, mono=False, align="left", lead=1.28,
             raw=False):
        font = r"\fontsize{%.2f}{%.2f}\selectfont" % (size, size * lead)
        if mono:
            font += r"\ttfamily"
        if weight == "bold":
            font += r"\bfseries"
        elif weight == "light":
            font += r"\lightfont"
        opts = "anchor=%s, inner sep=0, text=%s, font=%s" % (
            anchor, colour, font)
        if width:
            opts += ", text width=%.3fcm, align=%s" % (width, align)
        self.out.append(r"\node[%s] at (%.3f,%.3f) {%s};" % (
            opts, x, y, s if raw else s))

    def rect(self, x, y, w, h, fill=None, draw=None, r=0.0, lw=0.4,
             opacity=None, dash=False):
        if w <= 0 or h <= 0:
            return
        opts = []
        if fill:
            opts.append("fill=" + fill)
        if draw:
            opts.append("draw=%s, line width=%.2fpt" % (draw, lw))
        if r:
            opts.append("rounded corners=%.3fcm" % min(r, w / 2, h / 2))
        if opacity is not None:
            opts.append("fill opacity=%.2f" % opacity)
        if dash:
            opts.append("dash pattern=on 2pt off 1.5pt")
        self.out.append(r"\path[%s] (%.3f,%.3f) rectangle (%.3f,%.3f);" % (
            ", ".join(opts), x, y, x + w, y + h))

    def line(self, pts, colour="rule", lw=0.5, arrow=False, dash=False,
             bend=None):
        opts = ["draw=" + colour, "line width=%.2fpt" % lw]
        if arrow:
            opts.append("-{Stealth[length=5pt,width=4pt]}")
        if dash:
            opts.append("dash pattern=on 3pt off 2pt")
        if bend is not None and len(pts) == 2:
            path = "(%.3f,%.3f) to[bend left=%d] (%.3f,%.3f)" % (
                pts[0][0], pts[0][1], bend, pts[1][0], pts[1][1])
        else:
            path = " -- ".join("(%.3f,%.3f)" % p for p in pts)
        self.out.append(r"\path[%s] %s;" % (", ".join(opts), path))

    def heading(self, x, y, s, width=None):
        self.text(x, y, s.upper(), size=8, colour="soft", weight="bold",
                  anchor="north west", width=width,
                  raw=True)
        return y - 0.62

    def stacked(self, x, y, w, h, parts, total=None, r=0.06):
        """A stacked bar; parts are (value, colour)."""
        total = total or sum(v for v, _ in parts) or 1
        pos = x
        for v, colour in parts:
            ww = w * v / total
            self.rect(pos, y, ww, h, fill=colour)
            pos += ww
        return pos

    def render(self):
        return ("\\begin{tikzpicture}[x=1cm,y=1cm]\n"
                "\\path[use as bounding box] (0,0) rectangle (%.2f,%.2f);\n"
                % (PAGE_W, PAGE_H)
                + "\n".join(self.out) + "\n\\end{tikzpicture}\n\\newpage\n")


def legend(page, x, y, entries, gap=0.25):
    for colour, label in entries:
        page.rect(x, y - 0.14, 0.42, 0.28, fill=colour, r=0.05)
        page.text(x + 0.58, y, label, size=8.5, colour="soft")
        x += 0.58 + 0.165 * len(re.sub(r"\\[a-z]+|[{}]", "", label)) \
            + 0.45 + gap
    return x


# -- graphs laid out by dot, drawn here ------------------------------------

def dot_layout(nodes, edges, rankdir="LR", ranksep=0.9, nodesep=0.12,
               clusters=None):
    """nodes: {key: (width, height)} in cm; edges: [(a, b)]. Gives
    (width, height, {key: (x, y)}, [(a, b, [points])]) in cm."""
    keys = {k: "n%d" % i for i, k in enumerate(nodes)}
    back = {v: k for k, v in keys.items()}
    lines = ["digraph G {",
             "rankdir=%s; ranksep=%.2f; nodesep=%.2f; splines=true;"
             % (rankdir, ranksep / 2.54, nodesep / 2.54),
             "node [shape=box, fixedsize=true, label=\"\"];",
             "edge [arrowsize=0.5];"]
    for k, (w, h) in nodes.items():
        lines.append("%s [width=%.3f, height=%.3f];"
                     % (keys[k], w / 2.54, h / 2.54))
    for i, (name, members) in enumerate((clusters or {}).items()):
        lines.append("subgraph cluster_%d { %s }" % (
            i, " ".join(keys[m] + ";" for m in members if m in keys)))
    for a, b in edges:
        lines.append("%s -> %s;" % (keys[a], keys[b]))
    lines.append("}")
    try:
        res = subprocess.run(["dot", "-Tplain"], input="\n".join(lines),
                             capture_output=True, text=True, check=True)
    except FileNotFoundError:
        sys.exit("proof-map: graphviz's `dot` is needed")
    w = h = 0.0
    pos, splines = {}, []
    for line in res.stdout.split("\n"):
        t = line.split()
        if not t:
            continue
        if t[0] == "graph":
            w, h = float(t[2]) * 2.54, float(t[3]) * 2.54
        elif t[0] == "node":
            pos[back[t[1]]] = (float(t[2]) * 2.54, float(t[3]) * 2.54)
        elif t[0] == "edge":
            n = int(t[3])
            pts = [(float(t[4 + 2 * i]) * 2.54, float(t[5 + 2 * i]) * 2.54)
                   for i in range(n)]
            splines.append((back[t[1]], back[t[2]], pts))
    return w, h, pos, splines


def reduce_edges(edges):
    """The transitive reduction, near enough (cycles are left alone)."""
    succ = collections.defaultdict(set)
    for a, b in edges:
        if a != b:
            succ[a].add(b)
    memo = {}

    def reach(n):
        if n in memo:
            return memo[n]
        memo[n] = set()
        out = set()
        for m in succ[n]:
            out.add(m)
            out |= reach(m)
        memo[n] = out
        return out
    keep = []
    for a in list(succ):
        for b in succ[a]:
            if not any(w != b and b in reach(w) and w not in reach(b)
                       for w in succ[a]):
                keep.append((a, b))
    return keep


def draw_graph(page, x0, y0, w0, h0, nodes, edges, draw_node,
               edge_colour="faint", max_scale=1.0, rankdirs=("RL", "TB"),
               **kw):
    """Lay the graph out, whichever way fills the box given best, and
    draw it there. [draw_node] is called with a list to add TikZ to, the
    key and the centre."""
    best = None
    for rankdir in rankdirs:
        w, h, pos, splines = dot_layout(nodes, edges, rankdir=rankdir, **kw)
        if not w or not h:
            return 1.0
        scale = min(w0 / w, h0 / h, max_scale)
        if best is None or scale > best[0] * 1.02:
            best = (scale, w, h, pos, splines)
    scale, w, h, pos, splines = best
    ox = x0 + (w0 - w * scale) / 2
    oy = y0 + (h0 - h * scale) / 2
    out = [r"\begin{scope}[shift={(%.3f,%.3f)}, scale=%.4f, "
           r"transform shape]" % (ox, oy, scale)]
    for a, b, pts in splines:
        path = "(%.3f,%.3f)" % pts[0]
        for i in range(1, len(pts) - 2, 3):
            path += " .. controls (%.3f,%.3f) and (%.3f,%.3f) .. " \
                "(%.3f,%.3f)" % (pts[i] + pts[i + 1] + pts[i + 2])
        colour = edge_colour(a, b) if callable(edge_colour) else edge_colour
        out.append(r"\path[draw=%s, line width=0.45pt, "
                   r"-{Stealth[length=3.5pt,width=3pt]}] %s;"
                   % (colour, path))
    for k, (cx, cy) in pos.items():
        draw_node(out, k, cx, cy)
    out.append(r"\end{scope}")
    page.raw("\n".join(out))
    return scale


def box_node(out, cx, cy, w, h, fill, draw, label, sub=None, lw=0.5,
             size=7.0, text="ink", bold=True):
    out.append(r"\path[fill=%s, draw=%s, line width=%.2fpt, rounded "
               r"corners=0.1cm] (%.3f,%.3f) rectangle (%.3f,%.3f);"
               % (fill, draw, lw, cx - w / 2, cy - h / 2, cx + w / 2,
                  cy + h / 2))
    dy = 0.14 if sub else 0
    out.append(r"\node[anchor=center, inner sep=0, text=%s, font="
               r"\fontsize{%.1f}{%.1f}\selectfont%s] at (%.3f,%.3f) {%s};"
               % (text, size, size * 1.2, r"\bfseries" if bold else "",
                  cx, cy + dy, label))
    if sub:
        out.append(r"\node[anchor=center, inner sep=0, text=soft, font="
                   r"\fontsize{5.5}{6.5}\selectfont] at (%.3f,%.3f) {%s};"
                   % (cx, cy - 0.17, sub))


# --------------------------------------------------------------------------
# The pages
# --------------------------------------------------------------------------

class Report:
    def __init__(self, an, args):
        self.an = an
        self.dev = an.dev
        self.args = args
        self.pages = []
        an_results = [c for c in an.results
                      if not c.statement.file.is_examples
                      and not c.equational]
        self.cover = greedy_cover(an, an_results, 60)
        wanted = []
        for name in args.result or []:
            found = [c for c in an.claims.values() if c.name == name]
            if not found:
                sys.exit("proof-map: no claim named %s" % name)
            wanted += found
        self.featured = wanted or [c for c, _, _ in
                                   self.cover[:args.results]]
        self.chosen = bool(wanted)
        self.f_must, self.f_tests = set(), set()
        for c in self.featured:
            self.f_must |= c.must
            self.f_tests |= c.tests
        self.f_tests -= self.f_must
        self.f_below = set(self.featured)
        for c in self.featured:
            self.f_below |= c.below

    def add(self, page):
        self.pages.append(page)

    # -- totals ----------------------------------------------------------

    def totals(self):
        an = self.an
        t = {}
        t["all"] = sum(f.code_lines for f in an.focus)
        for role in ("spec", "impl", "proofs", "model", "extract"):
            t[role] = sum(f.code_lines for f in an.focus if f.role == role)
        t["by_name"] = t["spec"] + t["model"]
        results = [c.statement for c in an.results]
        t["result_statements"] = lines_of(results)
        t["cones"] = lines_of(an.must | an.tests) + t["result_statements"]
        t["meaning"] = lines_of(an.must) + t["result_statements"]
        t["featured"] = lines_of(self.f_must) + lines_of(
            c.statement for c in self.featured)
        return t

    # -- page: overview --------------------------------------------------

    def overview(self):
        an = self.an
        t = self.totals()
        p = Page("Proof map",
                 "The Rocq development of %s, as read from its sources: "
                 "what is specification, what is implementation, what is "
                 "proof, and what a reviewer has to read."
                 % " and ".join(sorted(self.dev.focus)))
        p.text(PAGE_W, PAGE_H - 0.1, esc(self.stamp()), size=8,
               colour="faint", anchor="north east")
        y = p.top
        # figures
        claims = an.claims.values()
        figures = [
            (len(an.focus), "files"),
            (t["all"], "lines of Rocq"),
            (len(claims), "claims"),
            (len(an.results), "results"),
            (sum(1 for it in an.items if it.kind == "axiom"), "axioms"),
            (sum(1 for it in an.items if it.admitted), "admitted"),
        ]
        x = 0
        for value, label in figures:
            p.text(x, y, num(value), size=30, weight="light",
                   anchor="north west")
            p.text(x, y - 1.35, label, size=9.5, colour="soft",
                   anchor="north west")
            x += 6.5
        y -= 2.7
        p.line([(0, y), (PAGE_W, y)])
        y -= 0.6
        # left: how to read
        lw = 15.5
        yy = p.heading(0, y, "How this map reads the development")
        paras = [
            r"Every \textbf{item} (a definition, a type, a theorem) has two "
            r"sets of references: what its \emph{statement} mentions and "
            r"what its \emph{proof} mentions. Rocq checks the proofs; only "
            r"statements can say the wrong thing. A definition is all "
            r"statement: its body is what it means.",
            r"A \textbf{claim} is a \texttt{Definition X\_statement : Prop} "
            r"together with the theorem that proves it. A claim that the "
            r"proof of another claim goes through is a \emph{lemma} of it. "
            r"A claim that no other claim uses is a \textbf{result}.",
            r"The \textbf{cone} of a result is everything reached from its "
            r"statement through statements. It is cut in two. A function "
            r"that the statement names directly is its "
            r"\textcolor{subject}{\textbf{subject}}: the statement holds "
            r"of whatever that function computes, so the function need not "
            r"be read for the theorem to mean what it says. The rest gives "
            r"the statement its "
            r"\textcolor{meaning}{\textbf{meaning}}: the propositions and "
            r"types it names and all they are defined with. Only the "
            r"meaning has to be reviewed.",
            r"A statement that names no proposition at all (an equation "
            r"between functions) has no such cut: there the functions are "
            r"the meaning.",
        ]
        for para in paras:
            p.text(0, yy, para, size=9.5, anchor="north west", width=lw,
                   lead=1.42)
            yy -= 0.52 * (len(re.sub(r"\\[a-z]+|[{}]", "", para)) // 92 + 1) \
                + 0.42
        # right: where the lines are
        rx = 17.5
        rw = PAGE_W - rx
        yy = p.heading(rx, y, "Where the lines are, by the name of the file")
        roles = [("proofs", "proof", "Proofs"), ("spec", "meaning", "Spec"),
                 ("impl", "subject", "implementation"),
                 ("model", "trust", "Model"), ("extract", "trust", "")]
        p.stacked(rx, yy - 0.75, rw, 0.75,
                  [(t[r], c) for r, c, _ in roles], total=t["all"])
        pos = rx
        small = []
        for r, c, label in roles:
            ww = rw * t[r] / t["all"]
            if label and ww > 2.5:
                p.text(pos + 0.2, yy - 0.375,
                       r"\textbf{%s}\quad %s" % (label, num(t[r])), size=8.5,
                       colour="paper" if c != "proof" else "ink")
            elif label:
                small.append(r"\textcolor{%s}{\textbf{%s}} %s"
                             % (c, label, num(t[r])))
            pos += ww
        p.text(rx + rw, yy - 1.05, r"\quad ".join(small), size=8,
               colour="soft", anchor="east")
        yy -= 2.0
        # the funnel
        yy = p.heading(rx, yy, "What has to be read")
        steps = [
            (t["all"], "proof", "Everything",
             "all the Rocq, comments and blank lines left out"),
            (t["all"] - t["proofs"], "lemma", "Not the proofs",
             "Rocq checks them"),
            (t["by_name"], "lemma", "By file name",
             r"every \texttt{*Spec.v} and \texttt{*Model.v}, as the "
             "headers say"),
            (t["cones"], "lemma", "The cones of the results",
             "statements of the %s results and all they reach"
             % num(len(an.results))),
            (t["meaning"], "meaning", "Their meaning only",
             "subjects left out"),
            (t["featured"], "result",
             "%d chosen results" % len(self.featured)
             if self.chosen else
             "The %d results that carry the most" % len(self.featured),
             "their meaning; see the pages on results"),
        ]
        bar_w = rw - 9.2
        for value, colour, label, note in steps:
            ww = max(bar_w * value / t["all"], 0.04)
            p.rect(rx + 9.2, yy - 0.62, ww, 0.62, fill=colour, r=0.05)
            p.text(rx, yy - 0.2, r"\textbf{%s}" % label, size=9.5)
            p.text(rx, yy - 0.58, note, size=7.5, colour="soft")
            inside = ww > 2.2
            p.text(rx + 9.2 + (ww - 0.2 if inside else ww + 0.2), yy - 0.31,
                   num(value), size=9.5, weight="bold",
                   colour="paper" if inside and colour != "lemma"
                   else "ink", anchor="east" if inside else "west")
            yy -= 1.08
        p.text(rx, yy - 0.05,
               r"Lines are lines of Rocq without comments and blank "
               r"lines; for an item, the lines of its statement. Proof "
               r"scripts are counted only in the first two bars.",
               size=7.5, colour="faint", anchor="north west", width=rw)
        self.schematic(p, 1.9, 9.4)
        # legend
        p.line([(0, 1.25), (PAGE_W, 1.25)])
        p.text(0, 0.78, "COLOURS, ON EVERY PAGE", size=7.5, colour="soft",
               weight="bold")
        legend(p, 0, 0.25, self.legend_entries())
        self.add(p)

    def schematic(self, p, y0, y1):
        """How a theorem joins a function to what is said of it."""
        an = self.an
        p.line([(0, y1 + 0.5), (PAGE_W, y1 + 0.5)])
        p.heading(0, y1 + 0.15, "How a theorem is kept honest")
        h = 2.3
        top, bot = y1 - 0.75 - h, y0
        linking = sum(1 for c in an.claims.values()
                      if c.tests and not c.equational)

        def box(x, y, w, colour, title, lines, light=False):
            p.rect(x, y, w, h, fill=colour + ("L" if light else ""),
                   draw=colour if light else None, r=0.15, lw=0.8)
            ink = "ink" if light else "paper"
            p.text(x + 0.35, y + h - 0.45, r"\textbf{%s}" % title,
                   size=11, colour=ink)
            p.text(x + 0.35, y + h - 0.85, lines, size=8, colour=ink,
                   anchor="north west", width=w - 0.7, lead=1.35)
        bw = 9.5
        xm, xs, xj = 0, (PAGE_W - bw) / 2, PAGE_W - bw
        box(xm, top, bw, "meaning", "Meaning",
            "The propositions and types that statements are written "
            "with, and all they are defined with: the machine, "
            "refinement, what it is for a value to have a type. "
            r"\textbf{Reviewed by reading.}")
        box(xs, top, bw, "result", "Statements",
            r"%s claims, each a \texttt{Definition X\_statement : Prop}. "
            r"%s are results; the others are lemmas of a result. "
            r"\textbf{The results are reviewed by reading.}"
            % (num(len(an.claims)), num(len(an.results))))
        box(xj, top, bw, "subject", "Subject",
            "The functions that statements are about: the validation "
            "and its rewrites, the meet and the join. A statement "
            "holds of whatever they compute. "
            r"\textbf{Not reviewed: run.}")
        box(xm, bot, bw, "trust", "Trusted outside Rocq",
            "That the model is the compiler: the conversion of terms, "
            r"the extraction directives. In \texttt{rocq/TRUST-LEDGER.md}; "
            "not on this map.", light=True)
        box(xs, bot, bw, "proof", "Proofs",
            r"%s lines in %d files. Each claim has a theorem of its "
            r"name. \textbf{Checked by Rocq}; nothing outside a Proofs "
            r"file refers to one."
            % (num(sum(f.code_lines for f in an.focus
                       if f.role == "proofs")),
               sum(1 for f in an.focus if f.role == "proofs")),
            light=True)
        box(xj, bot, bw, "ink", "OCaml",
            "%s definitions extracted from %s names, for the tests "
            "that run the validation on what Simplify did."
            % (num(len(an.extracted)), num(len(an.extract_roots))))
        mid = top + h / 2
        p.line([(xs, mid), (xm + bw + 0.1, mid)], colour="ink", lw=1.0,
               arrow=True)
        p.text((xm + bw + xs) / 2, mid + 0.3, "is written with", size=8.5,
               anchor="center")
        p.line([(xs + bw, mid), (xj - 0.1, mid)], colour="ink", lw=1.0,
               arrow=True)
        p.text((xs + bw + xj) / 2, mid + 0.3, "is about", size=8.5,
               anchor="center")
        p.text((xs + bw + xj) / 2, mid - 0.32, "%s claims" % num(linking),
               size=7.5, colour="soft", anchor="center")
        cx = xs + bw / 2
        p.line([(cx, bot + h), (cx, top - 0.1)], colour="ink", lw=1.0,
               arrow=True)
        p.text(cx + 0.25, (bot + h + top) / 2, "prove", size=8.5)
        cx = xj + bw / 2
        p.line([(cx, top), (cx, bot + h + 0.1)], colour="ink", lw=1.0,
               arrow=True)
        p.text(cx + 0.25, (bot + h + top) / 2, "extraction", size=8.5)
        bad = len(an.bad_refs) + len(an.bad_requires)
        for x in (xm + bw + 0.1, xj - 0.1):
            p.line([(x, top + 0.25), (xs if x < xs else xs + bw,
                                      bot + h - 0.25)],
                   colour="alert" if bad else "faint", lw=0.9, dash=True)
        p.text((xm + bw + xs) / 2, bot + h / 2 + 0.55,
               "never refers to" if not bad else "refers to", size=8,
               colour="alert" if bad else "soft", anchor="center")
        p.text((xm + bw + xs) / 2, bot + h / 2 + 0.17,
               r"0 references \checkmark" if not bad else "%d times" % bad,
               size=7.5, colour="alert" if bad else "result",
               anchor="center")

    def legend_entries(self):
        return [("meaning", "meaning: to be read"),
                ("subject", "subject: run, not read"),
                ("lemma", "needed by lemmas or other results only"),
                ("proof", "proof: checked by Rocq"),
                ("result", "a result"),
                ("trust", "trusted base (Model, Extract)"),
                ("alert", "a breach")]

    def stamp(self):
        def git(*a):
            try:
                return subprocess.run(
                    ("git", "-C", ROOT) + a, capture_output=True, text=True
                ).stdout.strip()
            except OSError:
                return ""
        import datetime
        return "%s at %s  ·  %s" % (
            git("rev-parse", "--abbrev-ref", "HEAD"),
            git("rev-parse", "--short=10", "HEAD"),
            datetime.date.today().isoformat())

    # -- page: separation --------------------------------------------------

    def nature(self, it):
        if it.kind == "statement":
            return "statement"
        if it.kind == "theorem":
            return "theorem"
        if it.kind in ("hyp", "axiom"):
            return "assumption"
        if it.is_prop:
            return "proposition"
        if it.kind == "inductive" or it.is_type:
            return "type"
        return "function"

    def separation(self):
        an = self.an
        p = Page("Specification, implementation and proof",
                 "Whether the three are kept apart, and which theorems "
                 "join them. A proof is vacuous when what it proves is "
                 "defined with what it is about, or when a statement can "
                 "be changed from a file that nobody reads.")
        y = p.top
        # ---- left: the matrix of what lives where
        lw = 17.0
        yy = p.heading(0, y, "What each kind of file holds (lines of "
                       "statements)")
        natures = [("proposition", "propositions", "meaning"),
                   ("type", "types", "meaning"),
                   ("function", "functions", "subject"),
                   ("statement", "statements", "result"),
                   ("assumption", "assumptions", "alert"),
                   ("theorem", "theorems", "proof")]
        roles = [("spec", "Spec"), ("impl", "implementation"),
                 ("model", "Model"), ("proofs", "Proofs")]
        cell = collections.Counter()
        count = collections.Counter()
        extracted = collections.Counter()
        for it in an.items:
            if it.local and it.kind != "hyp":
                continue
            k = (it.file.role, self.nature(it))
            cell[k] += it.stmt_lines
            count[k] += 1
            if it.extracted:
                extracted[k] += it.stmt_lines
        biggest = max(cell.values())
        cw, ch = 2.3, 1.75
        gx, gy = 3.1, yy - 1.0
        for j, (_, label, _) in enumerate(natures):
            p.text(gx + j * cw + cw / 2, gy + 0.3, label, size=8.5,
                   colour="soft", anchor="south")
        for i, (role, rlabel) in enumerate(roles):
            cy = gy - i * ch - ch / 2
            p.text(0, cy + 0.15, r"\textbf{%s}" % rlabel, size=9.5)
            nf = sum(1 for f in an.focus if f.role == role)
            p.text(0, cy - 0.27, "%d files" % nf, size=7.5, colour="soft")
            p.line([(gx, cy - ch / 2), (gx + len(natures) * cw,
                                        cy - ch / 2)], lw=0.3)
            for j, (nat, _, colour) in enumerate(natures):
                v = cell[role, nat]
                if not v:
                    p.text(gx + j * cw + cw / 2, cy, r"\,·\,", size=8,
                           colour="faint", anchor="center")
                    continue
                side = max(0.12, 1.5 * (v / biggest) ** 0.5)
                cx = gx + j * cw + cw / 2
                p.rect(cx - side / 2, cy + 0.12 - side / 2, side, side,
                       fill=colour, r=0.05)
                e = extracted[role, nat]
                if e:
                    es = max(0.08, 1.5 * (e / biggest) ** 0.5)
                    p.rect(cx - es / 2, cy + 0.12 - es / 2, es, es,
                           draw="ink", lw=0.7, r=0.03)
                p.text(cx, cy - 0.72, num(v), size=7.5, colour="soft",
                       anchor="center")
        my = gy - len(roles) * ch - 0.35
        p.rect(gx, my - 0.14, 0.28, 0.28, draw="ink", lw=0.7)
        p.text(gx + 0.45, my,
               "the part that is extracted to OCaml (%s functions and "
               "types, from %s names given to the extraction)"
               % (num(len(an.extracted)), num(len(an.extract_roots))),
               size=7.5, colour="soft")
        my -= 0.75
        p.text(0, my,
               r"The suffix of a file says who may \emph{depend} on it, "
               r"not what is in it: the Spec files hold most of the "
               r"functions of the development, because a statement about "
               r"a rewrite sits next to the rewrite. What separates a "
               r"function from its specification is therefore not the "
               r"file but the position of the function in the statement, "
               r"which the pages on results show.",
               size=8.5, colour="soft", anchor="north west", width=lw,
               lead=1.4)
        # ---- left bottom: three kinds of claim
        my -= 2.6
        yy2 = p.heading(0, my, "What the claims join")
        kinds = collections.Counter()
        for c in an.claims.values():
            if c.equational:
                kinds["equational"] += 1
            elif c.tests:
                kinds["linking"] += 1
            else:
                kinds["theory"] += 1
        total = len(an.claims) or 1
        rows = [("linking", "result",
                 r"\textbf{Function meets proposition.} The statement "
                 r"names a function and says that a proposition holds of "
                 r"it. These join implementation to specification."),
                ("theory", "meaning",
                 r"\textbf{Proposition to proposition.} No function is "
                 r"the subject: facts about the specification itself."),
                ("equational", "subject",
                 r"\textbf{Function to function.} Equations only, with "
                 r"no proposition to say what they mean; every function "
                 r"named is to be read.")]
        for key, colour, textv in rows:
            v = kinds[key]
            p.text(2.2, yy2 - 0.1, num(v), size=20, weight="light",
                   anchor="north east", colour=colour)
            p.rect(2.6, yy2 - 0.32, (lw - 2.6) * v / total, 0.2,
                   fill=colour, r=0.04)
            p.text(2.6, yy2 - 0.5, textv, size=8.5, anchor="north west",
                   width=lw - 2.6, lead=1.35)
            yy2 -= 1.75
        # ---- right: the checks
        rx = 19.0
        rw = PAGE_W - rx
        yy = p.heading(rx, y, "Checks")
        unproved = [c for c in an.claims.values() if not c.proof]
        axioms = [it for it in an.items if it.kind == "axiom"]
        admitted = [it for it in an.items if it.admitted]
        leak = [(it, d) for it, d in an.bad_refs
                if it.kind in ("statement", "def", "inductive")]
        ext_in_proofs = [it for it in an.extracted
                         if it.file.role == "proofs"]
        impl_meaning = collections.Counter()
        for it in an.must:
            if it.file.role == "impl" and self.nature(it) == "function":
                impl_meaning[it.file.module] += it.stmt_lines

        def names(items, n=4):
            s = ", ".join(r"\texttt{%s}" % code(x) for x in items[:n])
            if len(items) > n:
                s += " and %d more" % (len(items) - n)
            return s
        checks = [
            (not an.bad_requires,
             "No Spec, Model or implementation file requires a Proofs "
             "file",
             "so nothing in a Proofs file can change what a statement "
             "says or what is extracted."
             if not an.bad_requires else
             names(["%s requires %s" % (a.module, b.module)
                    for a, b in an.bad_requires])),
            (not leak,
             "No statement or definition outside the Proofs files "
             "mentions anything defined in one",
             "%s references from statements were followed."
             % num(sum(an.ref_matrix.values()))
             if not leak else
             names(["%s uses %s" % (a.qual, b.qual) for a, b in leak])),
            (not unproved,
             "Every claim has its theorem (%s of %s)"
             % (num(len(an.claims) - len(unproved)), num(len(an.claims))),
             "a theorem of the claim's name, in a file that sees the "
             "statement." if not unproved else
             "Not found: " + names([c.name for c in unproved])),
            (not admitted, "Nothing is admitted",
             "no proof ends with \\texttt{Admitted}." if not admitted
             else names([it.qual for it in admitted])),
            (not axioms, "No axiom and no parameter",
             "in the libraries mapped; those of the libraries they "
             "require are in \\texttt{rocq/TRUST-LEDGER.md}."
             if not axioms else names([it.qual for it in axioms])),
            (not ext_in_proofs,
             "Nothing that is extracted is defined in a Proofs file",
             "%s extracted definitions, all in Spec, Model and "
             "implementation files." % num(len(an.extracted))
             if not ext_in_proofs else
             names([it.qual for it in ext_in_proofs])),
            (not impl_meaning,
             "No result is specified with functions of an "
             "implementation file",
             "the meaning of every result stays within Spec and Model "
             "files." if not impl_meaning else
             r"%s lines of functions in implementation files are part "
             r"of what some result \emph{means}: a proposition is "
             r"defined with them, or an equation is stated of them. "
             r"They are read as specification. Most are in %s."
             % (num(sum(impl_meaning.values())), ", ".join(
                 r"\texttt{%s}" % esc(short(m)) for m, _ in
                 impl_meaning.most_common(4)))),
        ]
        for ok, title, note in checks:
            colour = "result" if ok else "alert"
            p.raw(r"\path[fill=%s] (%.3f,%.3f) circle (0.26);"
                  % (colour, rx + 0.26, yy - 0.3))
            p.text(rx + 0.26, yy - 0.3,
                   r"\textbf{\checkmark}" if ok else r"\textbf{!}", size=9,
                   colour="paper", anchor="center")
            p.text(rx + 0.85, yy - 0.08, r"\textbf{%s}" % title, size=9.5,
                   anchor="north west", width=rw - 0.85)
            extra = 0.42 if len(title) > 78 else 0
            p.text(rx + 0.85, yy - 0.55 - extra, note, size=8,
                   colour="soft", anchor="north west", width=rw - 0.85,
                   lead=1.35)
            yy -= 1.25 + extra + 0.36 * (
                len(re.sub(r"\\[a-z]+|[{}]", "", note)) // 108)
        # ---- right bottom: who refers to whom
        yy -= 0.3
        yy = p.heading(rx, yy, "Who refers to whom, across files "
                       "(references from statements; from proofs)")
        order = [("spec", "Spec"), ("impl", "impl."), ("model", "Model"),
                 ("proofs", "Proofs")]
        cw, ch = 2.9, 0.95
        gx = rx + 3.0
        p.text(gx + 2 * cw, yy - 0.05, "refers to", size=7.5,
               colour="faint", anchor="center")
        for j, (_, label) in enumerate(order):
            p.text(gx + j * cw + cw / 2, yy - 0.55, label, size=8.5,
                   colour="soft", anchor="center")
        big = max(list(an.ref_matrix.values()) + [1])
        for i, (ra, la) in enumerate(order):
            cy = yy - 0.9 - i * ch - ch / 2
            p.text(rx, cy, r"\textbf{%s}" % la, size=8.5)
            for j, (rb, _) in enumerate(order):
                v = an.ref_matrix[ra, rb]
                w = an.proof_matrix[ra, rb]
                forbidden = ra != "proofs" and rb == "proofs"
                if forbidden:
                    fill = "alertL" if v or w else "resultL"
                else:
                    fill = "meaning!%d!wash" % int(
                        75 * (v / big) ** 0.5) if v else "wash"
                p.rect(gx + j * cw + 0.05, cy - ch / 2 + 0.05, cw - 0.1,
                       ch - 0.1, fill=fill, r=0.06)
                dark = not forbidden and v / big > 0.35
                if forbidden and not (v or w):
                    label = r"none \checkmark"
                else:
                    label = num(v) + (r" \textcolor{%s}{; %s}" % (
                        "paper" if dark else "soft", num(w)) if w else "")
                p.text(gx + j * cw + cw / 2, cy, label, size=8,
                       colour="paper" if dark else "ink", anchor="center")
        self.add(p)

    # -- pages: the maps ---------------------------------------------------

    def model_map(self, library, title):
        an = self.an
        files = [f for f in an.focus if f.library == library]
        models = collections.defaultdict(list)
        for f in files:
            models[f.stem].append(f)
        model_of = {f: f.stem for f in files}
        info = {}
        for stem, fs in models.items():
            d = {"spec": 0, "impl": 0, "proofs": 0, "model": 0,
                 "extract": 0}
            for f in fs:
                d[f.role] += f.code_lines
            items = [it for f in fs for it in f.items]
            d["must"] = lines_of(it for it in items if it in self.f_must)
            d["subject"] = lines_of(it for it in items
                                    if it in self.f_tests)
            d["results"] = sum(1 for it in items if it.kind == "statement"
                               and an.claims[it].is_result)
            d["featured"] = [c for c in self.featured
                             if c.statement.file in fs]
            d["defs"] = d["spec"] + d["impl"] + d["model"]
            info[stem] = d
        shown = [s for s, d in info.items() if d["defs"]]
        proof_only = [s for s, d in info.items() if not d["defs"]]
        edges = set()
        outside = collections.Counter()
        for f in files:
            if f.role == "proofs" or f.stem not in shown:
                continue
            for it in f.items:
                for dd in it.sdeps:
                    g = dd.file
                    if g.role == "proofs" or g is f:
                        continue
                    if g in model_of:
                        if model_of[g] != f.stem:
                            edges.add((f.stem, model_of[g]))
                    elif g.library in self.dev.focus:
                        outside[f.stem] += 1
        edges = reduce_edges(edges)
        nw, nh = 3.3, 1.0
        p = Page(title,
                 "Each box is a model: its Spec, implementation and "
                 "Proofs files together. An arrow goes from a model to "
                 "one whose definitions its own definitions are written "
                 "with (proofs left out; arrows implied by others are "
                 "not drawn), so what everything stands on is where the "
                 "arrows end. The deeper the blue, the more of the model "
                 "is meaning of %s."
                 % ("the chosen results" if self.chosen else
                    "the %d results that carry the most" %
                    len(self.featured)))
        biggest = max(d["defs"] + d["proofs"] for d in info.values())

        def draw(out, k, cx, cy):
            d = info[k]
            share = d["must"] / d["defs"] if d["defs"] else 0
            fill = "meaning!%d!paper" % int(8 + 62 * min(1, share)) \
                if d["must"] else "paper"
            dark = share > 0.6
            feat = bool(d["featured"])
            out.append(
                r"\path[fill=%s, draw=%s, line width=%.2fpt, rounded "
                r"corners=0.09cm] (%.3f,%.3f) rectangle (%.3f,%.3f);"
                % (fill, "result" if feat else
                   ("meaning" if d["must"] else "faint"),
                   1.4 if feat else 0.5, cx - nw / 2, cy - nh / 2,
                   cx + nw / 2, cy + nh / 2))
            name = short(k)
            size = 7.0 if len(name) < 22 else 7.0 * 22 / len(name)
            out.append(
                r"\node[anchor=west, inner sep=0, text=%s, font="
                r"\fontsize{%.2f}{8}\selectfont\bfseries] at (%.3f,%.3f) "
                r"{%s};" % ("paper" if dark else "ink", size,
                            cx - nw / 2 + 0.14, cy + 0.2, esc(name)))
            note = []
            if d["must"]:
                note.append("read %s" % num(d["must"]))
            if d["results"]:
                note.append("%d result%s" % (
                    d["results"], "" if d["results"] == 1 else "s"))
            out.append(
                r"\node[anchor=west, inner sep=0, text=%s, font="
                r"\fontsize{5.2}{6}\selectfont] at (%.3f,%.3f) {%s};"
                % ("paper" if dark else "soft", cx - nw / 2 + 0.14,
                   cy - 0.1, r" \,·\, ".join(note)))
            # the bar: Spec, implementation, Proofs
            total = d["defs"] + d["proofs"]
            bw = (nw - 0.28) * (total / biggest) ** 0.5
            bx = cx - nw / 2 + 0.14
            for v, colour in ((d["spec"] + d["model"], "meaning"),
                              (d["impl"], "subject"),
                              (d["proofs"], "proof")):
                ww = bw * v / total
                if ww > 0:
                    out.append(
                        r"\path[fill=%s] (%.3f,%.3f) rectangle "
                        r"(%.3f,%.3f);" % (colour, bx, cy - 0.38, bx + ww,
                                           cy - 0.28))
                bx += ww
            if dark:
                out.append(
                    r"\path[draw=paper, line width=0.2pt] (%.3f,%.3f) "
                    r"rectangle (%.3f,%.3f);"
                    % (cx - nw / 2 + 0.14, cy - 0.38,
                       cx - nw / 2 + 0.14 + bw, cy - 0.28))
        nodes = {k: (nw, nh) for k in shown}
        draw_graph(p, 0, 1.3, PAGE_W, p.top - 1.5, nodes, edges, draw,
                   ranksep=0.55, nodesep=0.08)
        x = legend(p, 0, 0.3, [
            ("meaning", "Spec and Model lines"),
            ("subject", "implementation lines"),
            ("proof", "Proofs lines (bar length: square root of the "
             "model's size)")])
        p.rect(x, 0.16, 0.42, 0.28, draw="result", lw=1.4, r=0.05)
        p.text(x + 0.58, 0.3, "holds one of the results featured", size=8.5,
               colour="soft")
        if proof_only:
            p.text(0, 0.85, "Proof-only modules, not drawn: "
                   + ", ".join(esc(short(s)) for s in sorted(proof_only)),
                   size=7, colour="faint", width=PAGE_W,
                   anchor="west")
        self.add(p)

    # -- page: the results ---------------------------------------------------

    def results_page(self):
        an = self.an
        p = Page("The results, and how much each carries",
                 "A result is a claim that no other claim's proof uses. "
                 "The results are put in the order in which each covers "
                 "the most claims not yet covered: a claim is covered "
                 "when it is the result or one of the claims that the "
                 "proof of the result goes through. Examples and "
                 "statements of equations only are left out.")
        y = p.top
        total = len(an.claims)
        # curve
        cw, chh = 12.5, 8.0
        yy = p.heading(0, y, "Claims covered by the first n results")
        ox, oy = 1.1, yy - chh - 0.3
        pts = [(0, 0)] + [(i + 1, cov) for i, (_, _, cov)
                          in enumerate(self.cover)]
        n = len(self.cover)
        for frac in (0.25, 0.5, 0.75, 1.0):
            gy = oy + chh * frac
            p.line([(ox, gy), (ox + cw, gy)], lw=0.3)
            p.text(ox - 0.15, gy, "%d\\%%" % (frac * 100), size=7.5,
                   colour="soft", anchor="east")
        for k in range(0, n + 1, 10):
            p.text(ox + cw * k / n, oy - 0.3, str(k), size=7.5,
                   colour="soft", anchor="center")
        path = " -- ".join("(%.3f,%.3f)" % (ox + cw * i / n,
                                            oy + chh * c / total)
                           for i, c in pts)
        p.raw(r"\path[fill=resultL] (%.3f,%.3f) -- %s -- (%.3f,%.3f) -- "
              r"cycle;" % (ox, oy, path, ox + cw, oy))
        p.raw(r"\path[draw=result, line width=1.4pt, line join=round] %s;"
              % path)
        p.line([(ox, oy), (ox + cw, oy)], colour="soft", lw=0.6)
        k = len(self.featured) if not self.chosen else 0
        if k and k <= n:
            cx, cyy = ox + cw * k / n, oy + chh * pts[k][1] / total
            p.raw(r"\path[fill=result] (%.3f,%.3f) circle (0.1);"
                  % (cx, cyy))
            p.text(cx + 0.25, cyy - 0.35,
                   r"\textbf{%d results: %d\%% of the claims}"
                   % (k, round(100 * pts[k][1] / total)), size=8.5,
                   colour="result")
        final = pts[-1][1]
        p.text(ox + cw, oy + chh * final / total + 0.3,
               "%d results: %d\\%%" % (n, round(100 * final / total)),
               size=8, colour="soft", anchor="east")
        # the distribution of results
        yy2 = oy - 1.4
        yy2 = p.heading(0, yy2, "The %s results, by how many claims "
                        "each rests on" % num(len(an.results)))
        buckets = [(0, 0, "none"), (1, 4, "1 to 4"), (5, 19, "5 to 19"),
                   (20, 99, "20 to 99"), (100, 10 ** 9, "100 or more")]
        counts = [sum(1 for c in an.results if lo <= len(c.below) <= hi)
                  for lo, hi, _ in buckets]
        mx = max(counts)
        for (lo, hi, label), v in zip(buckets, counts):
            p.text(2.4, yy2 - 0.25, label, size=8.5, anchor="east",
                   colour="soft")
            ww = 9.0 * v / mx
            p.rect(2.7, yy2 - 0.45, ww, 0.4, fill="result", r=0.04)
            p.text(2.7 + ww + 0.2, yy2 - 0.25, num(v), size=8.5)
            yy2 -= 0.62
        p.text(0, yy2 - 0.2,
               r"A result that rests on no other claim is proved from "
               r"lemmas of its Proofs file alone. Most are small facts "
               r"that fix the reading of a definition; nothing depends "
               r"on them, and their statements are what there is to "
               r"review.", size=8, colour="soft", anchor="north west",
               width=13.5, lead=1.4)
        # table
        tx = 15.5
        tw = PAGE_W - tx
        yy = p.heading(tx, y, "The first %d, with what each asks a "
                       "reviewer to read" % min(30, n))
        cols = [(0, "result"), (11.3, "rests on"), (13.3, "new"),
                (15.3, "meaning and subject, in lines")]
        for cx, label in cols:
            p.text(tx + cx, yy - 0.1, label, size=7.5, colour="faint")
        yy -= 0.5
        rows = self.cover[:30]
        mxl = max(lines_of(c.must) + lines_of(c.tests) for c, _, _ in rows)
        bar = tw - 15.3 - 1.6
        for i, (c, gain, cov) in enumerate(rows):
            feat = c in self.featured
            if feat:
                p.rect(tx - 0.2, yy - 0.34, tw + 0.2, 0.68, fill="wash",
                       r=0.05)
            p.text(tx, yy + 0.1, r"\texttt{%s}" % code(
                c.name if len(c.name) < 62 else c.name[:60] + "…"),
                size=7.6, weight="bold" if feat else "")
            p.text(tx, yy - 0.17, esc(short(c.statement.file.module)),
                   size=6.3, colour="soft")
            p.text(tx + 12.6, yy, num(len(c.below)), size=8, anchor="east")
            p.text(tx + 14.4, yy, "+" + num(gain), size=8, anchor="east",
                   colour="result")
            m, s = lines_of(c.must), lines_of(c.tests)
            end = p.stacked(tx + 15.3, yy - 0.16, bar * (m + s) / mxl, 0.32,
                            [(m, "meaning"), (s, "subject")])
            p.text(end + 0.15, yy, r"%s \textcolor{soft}{+ %s}"
                   % (num(m), num(s)), size=7)
            yy -= 0.7
        legend(p, tx, 0.3, [("meaning", "meaning: to be read"),
                            ("subject", "subject: run, not read")])
        self.add(p)

    # -- page: what to read ----------------------------------------------

    def reading_page(self):
        an = self.an
        t = self.totals()
        what = "the chosen results" if self.chosen else \
            "the %d results that carry the most" % len(self.featured)
        p = Page("What to read",
                 "Every file that holds something a result means. The "
                 "dark part of a bar is meaning of %s; the lighter blue "
                 "is meaning of some other result only; amber is the "
                 "subject of every result that reaches it; grey is the "
                 "rest of the file's statements (lemma statements, and "
                 "what only they use)." % what)
        y = p.top
        rows = []
        for f in an.focus:
            if f.role == "proofs":
                continue
            d = collections.Counter()
            for it in f.items:
                if it.kind == "theorem" or (it.local and it.kind != "hyp"):
                    continue
                if it in self.f_must:
                    d["a"] += it.stmt_lines
                elif it in an.must:
                    d["b"] += it.stmt_lines
                elif it in an.tests or it in self.f_tests:
                    d["c"] += it.stmt_lines
                else:
                    d["d"] += it.stmt_lines
            rows.append((f, d))
        rows.sort(key=lambda r: (-r[1]["a"], -r[1]["b"]))
        per_col = 46
        shown = rows[:per_col * 3]
        mx = max(sum(d.values()) for _, d in shown)
        colw = (PAGE_W - 2 * 0.9) / 3
        bar = colw - 5.6
        for i, (f, d) in enumerate(shown):
            cx = (i // per_col) * (colw + 0.9)
            cy = y - 0.3 - (i % per_col) * 0.49
            role = {"spec": "meaning", "impl": "subject",
                    "model": "trust", "extract": "trust"}[f.role]
            p.rect(cx, cy - 0.13, 0.1, 0.26, fill=role)
            name = short(f.module)
            p.text(cx + 0.22, cy, esc(name if len(name) < 31
                                      else name[:29] + "…"), size=7.3)
            p.text(cx + 5.35, cy, num(d["a"]) if d["a"] else "", size=7.3,
                   anchor="east", weight="bold", colour="meaning")
            p.stacked(cx + 5.5, cy - 0.15, bar * sum(d.values()) / mx, 0.3,
                      [(d["a"], "meaning"), (d["b"], "lemma"),
                       (d["c"], "subject"), (d["d"], "proofL")])
        rest = rows[len(shown):]
        x = legend(p, 0, 0.3, [
            ("meaning", "meaning of %s (the number)" % what),
            ("lemma", "meaning of other results only"),
            ("subject", "subject only"),
            ("proofL", "other statements")])
        p.text(PAGE_W, 0.3,
               ("%d more files not shown.  " % len(rest) if rest else "")
               + r"Total to read for %s: \textbf{%s lines} in %d files."
               % (what, num(lines_of(self.f_must)),
                  sum(1 for _, d in rows if d["a"])),
               size=8.5, colour="ink", anchor="east")
        self.add(p)

    # -- page: load-bearing definitions --------------------------------------

    def bearing_page(self):
        an = self.an
        p = Page("The definitions that bear the most",
                 "For each definition, the number of results whose "
                 "meaning includes it. A mistake in one of these changes "
                 "what that many theorems say, so these are where a "
                 "reviewer's attention is worth the most. Review them "
                 "first, and against the compiler.")
        y = p.top
        n = len(an.results)
        ranked = sorted(
            (it for it in an.reach
             if it.kind in ("def", "inductive") and not it.local),
            key=lambda it: (-an.reach[it], it.id))
        per_col = 44
        shown = ranked[:per_col * 2]
        colw = (PAGE_W - 1.2) / 2
        for i, it in enumerate(shown):
            cx = (i // per_col) * (colw + 1.2)
            cy = y - 0.3 - (i % per_col) * 0.52
            nat = self.nature(it)
            colour = {"proposition": "meaning", "type": "trust",
                      "function": "subject"}.get(nat, "meaning")
            p.text(cx, cy, r"\texttt{%s}" % code(it.qual), size=7.6)
            p.text(cx + 7.4, cy, esc(short(it.file.module)), size=6.5,
                   colour="soft")
            p.text(cx + 12.6, cy, "%s, %d l." % (nat, it.stmt_lines),
                   size=6.5, colour="soft", anchor="east")
            ww = (colw - 14.2) * an.reach[it] / n
            p.rect(cx + 12.8, cy - 0.15, ww, 0.3, fill=colour, r=0.03)
            p.text(cx + 12.8 + ww + 0.12, cy, num(an.reach[it]), size=7)
        legend(p, 0, 0.3, [("trust", "a type"),
                           ("meaning", "a proposition"),
                           ("subject", "a function that propositions are "
                            "defined with")])
        p.text(PAGE_W, 0.3, "bar: results whose meaning includes the "
               "definition, of %s" % num(n), size=8.5, colour="soft",
               anchor="east")
        self.add(p)

    # -- pages: one result ---------------------------------------------------

    def result_page(self, c):
        an = self.an
        st = c.statement
        below_files = {b.statement.file for b in c.below}
        p = Page(r"\texttt{%s}" % code(c.name),
                 "%s, line %d. %s Its proof goes through %s other claims "
                 "of %d Spec files."
                 % (esc(st.file.rel), st.line,
                    ("Proved in %s." % esc(c.proof.file.module))
                    if c.proof else r"\textcolor{alert}{No proof found.}",
                    num(len(c.below)), len(below_files)))
        y = p.top
        lw = 17.5
        # statement
        yy = p.heading(0, y, "The statement")
        text = st.file.raw[st.start:st.stmt_end]
        text = text.encode("latin-1").decode("utf-8", "replace")
        text = re.sub(r"\(\*.*?\*\)", "", text, flags=re.S)
        lines = [l.rstrip() for l in text.split("\n") if l.strip()]
        max_lines = 17
        if len(lines) > max_lines:
            lines = lines[:max_lines - 1] + ["  … (%d more lines)"
                                             % (len(lines) - max_lines + 1)]
        widest = max(len(l) for l in lines)
        size = min(8.0, 8.0 * 76 / max(widest, 76))
        boxh = len(lines) * size * 0.047 + 0.5
        p.rect(0, yy - boxh, lw, boxh, fill="wash", r=0.12)
        ty = yy - 0.3
        subjects = {d.name for d in c.direct if d in c.tests}
        for l in lines:
            body = esc(l).replace(" ", r"\ ")
            for nm in sorted(subjects, key=len, reverse=True):
                body = re.sub(
                    r"(?<![A-Za-z0-9_'\\])%s(?!\\_|[A-Za-z0-9'])"
                    % re.escape(esc(nm)),
                    lambda m: r"\textcolor{subject}{\textbf{%s}}"
                    % m.group(0), body)
            p.text(0.3, ty, body, size=size, mono=True, anchor="north west")
            ty -= size * 0.047
        yy -= boxh + 0.55
        # what each name costs: the dominator tree of the cone
        cone = c.must | c.tests
        idom = dominators(st, lambda it: [d for d in it.sdeps
                                          if d in cone or it is st])
        owner = {}

        def top(it):
            chain = []
            while it not in owner:
                parent = idom.get(it)
                if parent is None or parent is st:
                    owner[it] = it
                    break
                chain.append(it)
                it = parent
            for x in chain:
                owner[x] = owner[it]
            return owner[it]
        only = collections.Counter()
        for it in cone:
            if it in idom:
                only[top(it)] += it.stmt_lines
        direct = sorted((d for d in c.direct if d in cone),
                        key=lambda d: -only[d])
        shared = lines_of(cone) - sum(only[d] for d in direct)
        m_lines, s_lines = lines_of(c.must), lines_of(c.tests)
        yy = p.heading(0, yy, "What each name in the statement brings "
                       "with it (lines that are read for it alone)")
        rows = direct[:11]
        mx = max([only[d] for d in rows] + [shared, 1])
        bar = lw - 9.3
        for d in rows:
            subj = d in c.tests
            colour = "subject" if subj else "meaning"
            p.text(0, yy - 0.2, r"\texttt{%s}" % esc(
                d.qual if len(d.qual) < 40 else d.qual[:38] + "…"),
                size=8)
            p.text(7.9, yy - 0.2,
                   "subject" if subj else self.nature(d), size=7,
                   colour=colour, anchor="east")
            ww = bar * only[d] / mx
            p.rect(8.1, yy - 0.36, max(ww, 0.03), 0.32, fill=colour,
                   r=0.03)
            p.text(8.1 + ww + 0.15, yy - 0.2, num(only[d]), size=7.5)
            yy -= 0.5
        if len(direct) > len(rows):
            p.text(0, yy - 0.2, "and %d more names" % (
                len(direct) - len(rows)), size=7.5, colour="faint")
            yy -= 0.5
        ww = bar * shared / mx
        p.text(0, yy - 0.2, "shared by several of them", size=8,
               colour="soft")
        p.rect(8.1, yy - 0.36, max(ww, 0.03), 0.32, fill="lemma", r=0.03)
        p.text(8.1 + ww + 0.15, yy - 0.2, num(shared), size=7.5)
        yy -= 0.95
        # the reading list
        yy = p.heading(0, yy, "To review this theorem, read %s lines; "
                       "%s more are its subject" % (num(m_lines),
                                                     num(s_lines)))
        by_file = collections.Counter()
        sub_file = collections.Counter()
        for it in c.must:
            by_file[it.file] += it.stmt_lines
        for it in c.tests:
            sub_file[it.file] += it.stmt_lines
        room = int((yy - 0.6) / 0.46)
        files = sorted(set(by_file) | set(sub_file),
                       key=lambda f: (-by_file[f], -sub_file[f]))
        rows = files[:max(room, 3)]
        mx = max(by_file[f] + sub_file[f] for f in rows) if rows else 1
        for f in rows:
            p.text(0, yy - 0.2, esc(short(f.module)), size=8)
            p.text(7.9, yy - 0.2, num(by_file[f]) if by_file[f] else "",
                   size=7.5, anchor="east", colour="meaning",
                   weight="bold")
            end = p.stacked(8.1, yy - 0.36,
                            bar * (by_file[f] + sub_file[f]) / mx, 0.32,
                            [(by_file[f], "meaning"),
                             (sub_file[f], "subject")])
            if sub_file[f]:
                p.text(end + 0.15, yy - 0.2, "+ " + num(sub_file[f]),
                       size=7, colour="subject")
            yy -= 0.46
        if len(files) > len(rows):
            p.text(0, yy - 0.2, "and %d more files, with %s lines to read"
                   % (len(files) - len(rows),
                      num(sum(by_file[f] for f in files[len(rows):]))),
                   size=7.5, colour="faint")
        # the claims below
        rx = 19.0
        rw = PAGE_W - rx
        yy = p.heading(rx, y, "The claims that its proof goes through")
        below = c.below | {c}
        if len(below) <= 46:
            nodes, labels = {}, {}
            for b in below:
                name = b.name if len(b.name) <= 34 else b.name[:32] + "…"
                nodes[b] = (0.3 + 0.118 * max(len(name), 12), 0.72)
                labels[b] = (name, short(b.statement.file.module))
            edges = reduce_edges([(a, b) for a in below for b in a.uses
                                  if b in below])

            def draw(out, k, cx, cy):
                root = k is c
                box_node(out, cx, cy, nodes[k][0], nodes[k][1],
                         "result" if root else "paper",
                         "result" if root else "meaning",
                         r"\ttfamily " + esc(labels[k][0]), size=5.8,
                         sub=None if root else esc(labels[k][1]),
                         text="paper" if root else "ink", bold=root)
            caption = "one box for each claim; an arrow goes to a claim " \
                "that the proof uses"
        else:
            groups = collections.defaultdict(set)
            for b in below:
                groups[b.statement.file].add(b)
            nodes, labels = {}, {}
            for f, bs in groups.items():
                name = short(f.module)
                nodes[f] = (0.4 + 0.15 * max(len(name), 10), 0.8)
                labels[f] = (name, len(bs))
            edges = reduce_edges(
                [(a.statement.file, b.statement.file) for a in below
                 for b in a.uses if b in below
                 and a.statement.file is not b.statement.file])

            def draw(out, k, cx, cy):
                root = k is st.file
                box_node(out, cx, cy, nodes[k][0], nodes[k][1],
                         "result" if root else "paper",
                         "result" if root else "meaning",
                         esc(labels[k][0]), size=7,
                         sub=None if root else "%d claim%s" % (
                             labels[k][1], "" if labels[k][1] == 1
                             else "s"),
                         text="paper" if root else "ink")
            caption = "one box for each Spec file, with the number of " \
                "its claims that the proof goes through"
        draw_graph(p, rx, 0.9, rw, yy - 1.1, nodes, edges, draw,
                   ranksep=0.7, nodesep=0.14, max_scale=1.5)
        p.text(rx, 0.3, caption + ". None of them has to be read: they "
               "are lemmas here.", size=8, colour="soft")
        self.add(p)

    # -- pages: all the results ------------------------------------------

    def all_results(self):
        an = self.an
        rows = sorted(an.results, key=lambda c: (
            c.statement.file.library, c.statement.file.module,
            c.statement.line))
        per_col, cols = 70, 3
        colw = (PAGE_W - (cols - 1) * 0.8) / cols
        mx = max(lines_of(c.must) + lines_of(c.tests) for c in rows)
        # with a line for each file
        entries = []
        last = None
        for c in rows:
            if c.statement.file is not last:
                last = c.statement.file
                entries.append(("file", last))
            entries.append(("claim", c))
        per_page = per_col * cols
        chunks = [entries[i:i + per_page]
                  for i in range(0, len(entries), per_page)]
        for n, chunk in enumerate(chunks):
            p = Page("Every result" + (" (%d of %d)" % (n + 1, len(chunks))
                                       if len(chunks) > 1 else ""),
                     "By file. For each: the claims its proof goes "
                     "through, and the lines of its meaning and of its "
                     "subject. \\textbf{=} marks a statement of "
                     "equations only.")
            y = p.top
            for i, (kind, v) in enumerate(chunk):
                cx = (i // per_col) * (colw + 0.8)
                cy = y - 0.15 - (i % per_col) * 0.322
                if kind == "file":
                    p.text(cx, cy, r"\textbf{%s}" % esc(short(v.module)),
                           size=6.6, colour="meaning")
                    p.line([(cx + 0.15 * len(short(v.module)) + 0.3, cy),
                            (cx + colw, cy)], lw=0.3)
                    continue
                c = v
                name = c.name if len(c.name) < 50 else c.name[:48] + "…"
                p.text(cx + 0.15, cy, r"\texttt{%s}" % esc(name), size=5.6)
                p.text(cx + 7.6, cy, num(len(c.below)) if c.below else "",
                       size=5.6, colour="result", anchor="east")
                m, s = lines_of(c.must), lines_of(c.tests)
                end = p.stacked(cx + 7.8, cy - 0.09,
                                (colw - 9.6) * (m + s) / mx, 0.18,
                                [(m, "meaning"), (s, "subject")])
                p.text(end + 0.1, cy, num(m) + (r"\,\textbf{=}"
                                                if c.equational else ""),
                       size=5.2, colour="soft")
            legend(p, 0, 0.3, [("meaning", "meaning"),
                               ("subject", "subject")])
            p.text(PAGE_W, 0.3, "green: claims the proof goes through; "
                   "grey: lines of meaning", size=8, colour="soft",
                   anchor="east")
            self.add(p)

    # -- page: method ------------------------------------------------------

    def method_page(self, check):
        p = Page("How this was worked out, and where it can be wrong")
        y = p.top
        colw = (PAGE_W - 1.6) / 2
        left = [
            ("Source", r"The \texttt{.v} files are read as text, with "
             r"no build: comments are removed, the text is cut into "
             r"commands, and each command that names something is an "
             r"item. A reference is an identifier that names an item "
             r"visible at that point (the same file above it, or a file "
             r"it requires, directly or not; a qualified name, the item "
             r"of that module). Notations made of symbols are not "
             r"followed."),
            ("Checked against Rocq", check),
            ("Statement and proof", r"The statement of a theorem is its "
             r"type; of a definition, a type or a function, the whole "
             r"of it, since Rocq unfolds it. A proof that ends with "
             r"\texttt{Defined} is counted as statement. A section "
             r"hypothesis that a proof uses is a premise of the "
             r"theorem, so it is counted in the statement."),
            ("Claims", r"A claim is a definition whose name ends in "
             r"\texttt{\_statement}, with the theorem of the same name "
             r"less the suffix, in a file that sees it. Claim $A$ uses "
             r"claim $B$ when the proof of $A$ mentions the theorem of "
             r"$B$, or a lemma that is not a claim and whose proof does, "
             r"and so on down. A result is a claim that no claim uses."),
        ]
        right = [
            ("Meaning and subject", r"Within the cone of a statement, "
             r"a function that the statement names itself, and what is "
             r"reached from it through functions only, is subject; an "
             r"abbreviation counts as what it abbreviates. Whatever is "
             r"also reached through a proposition or a type is "
             r"meaning. If the statement names no proposition, the "
             r"whole cone is meaning."),
            ("Why a subject need not be read", r"The theorem "
             r"\texttt{check x = true -> good (f x)} is true of "
             r"whatever \texttt{check} and \texttt{f} compute; a "
             r"reader who knows \texttt{good} knows what is promised. "
             r"What reading cannot be skipped for is \emph{usefulness}: "
             r"a \texttt{check} that never answers \texttt{true} makes "
             r"the theorem empty. That is a question for the tests that "
             r"run the extracted code and for the examples, not for the "
             r"review of the statement."),
            ("Where the cut is too generous", r"A function of the "
             r"\emph{semantics} named in a statement beside a "
             r"proposition (\texttt{run n s = Exit k vs -> P vs}) is "
             r"taken for a subject, though it is the machine and has to "
             r"be read. It is read all the same wherever a proposition "
             r"of the same cone is defined with it, as refinement is; "
             r"each page on a result lists its subjects by name, so "
             r"that one that is not can be seen."),
            ("What is not looked at", r"That the model says what the "
             r"compiler does: the conversion of terms, the extraction "
             r"directives and the trusted OCaml are in "
             r"\texttt{rocq/TRUST-LEDGER.md}. Whether a hypothesis is "
             r"reasonable. Whether a proposition, read, says what its "
             r"name does."),
            ("Running it", r"\texttt{rocq/proof-map.py} writes "
             r"\texttt{rocq/proof-map.pdf}; it needs python3, graphviz "
             r"and lualatex, and no build. \texttt{--result NAME} "
             r"(repeated) features the claims named in place of those "
             r"that carry the most, and colours the maps and the "
             r"reading list by them; \texttt{--results N} features the "
             r"first N. \texttt{--library NAME} (repeated) maps other "
             r"libraries of \texttt{rocq/Makefile}. "
             r"\texttt{--check-globs -v} lists where the references "
             r"read here differ from Rocq's."),
        ]
        for x, col in ((0, left), (colw + 1.6, right)):
            yy = y
            for title, body in col:
                yy = p.heading(x, yy, title)
                p.text(x, yy, body, size=9.5, anchor="north west",
                       width=colw, lead=1.45)
                yy -= 0.56 * (len(re.sub(r"\\[a-z]+|[{}]", "", body))
                              // 108 + 1) + 0.9
        self.add(p)

    # -- the document ------------------------------------------------------

    def build(self, check):
        self.overview()
        self.separation()
        for d, lib, title in LIBRARIES:
            if lib in self.dev.focus:
                self.model_map(lib, "The models of %s" % lib)
        self.results_page()
        self.reading_page()
        self.bearing_page()
        for c in self.featured:
            self.result_page(c)
        self.all_results()
        self.method_page(check)
        colours = "\n".join(r"\definecolor{%s}{HTML}{%s}" % kv
                            for kv in COLOURS.items())
        return PREAMBLE.replace("@COLOURS@", colours) + "".join(
            pg.render() for pg in self.pages) + "\\end{document}\n"


PREAMBLE = r"""\documentclass{article}
\usepackage[a3paper,landscape,margin=1.5cm]{geometry}
\usepackage{fontspec}
\usepackage{amssymb}
\usepackage{tikz}
\usetikzlibrary{arrows.meta}
\IfFontExistsTF{Avenir Next}{%
  \setmainfont{Avenir Next}[Ligatures=TeX]%
  \newfontfamily\lightfont{Avenir Next Ultra Light}}{%
\IfFontExistsTF{Inter}{%
  \setmainfont{Inter}[Ligatures=TeX]\newfontfamily\lightfont{Inter Light}}{%
\IfFontExistsTF{Helvetica Neue}{%
  \setmainfont{Helvetica Neue}[Ligatures=TeX]%
  \newfontfamily\lightfont{Helvetica Neue Light}}{%
  \setmainfont{Latin Modern Sans}[Ligatures=TeX]%
  \newfontfamily\lightfont{Latin Modern Sans}}}}
\IfFontExistsTF{Menlo}{\setmonofont{Menlo}[Scale=0.88]}{%
\IfFontExistsTF{DejaVu Sans Mono}{\setmonofont{DejaVu Sans Mono}[Scale=0.88]}{%
  \setmonofont{Latin Modern Mono}}}
@COLOURS@
\pagestyle{empty}
\setlength\parindent{0pt}
\hyphenpenalty=10000 \exhyphenpenalty=10000
\begin{document}
\color{ink}
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output",
                    default=os.path.join(ROOT, "rocq", "proof-map.pdf"))
    ap.add_argument("--library", action="append",
                    help="a library to map (default: the Flambda 2 ones)")
    ap.add_argument("--result", action="append", metavar="NAME",
                    help="a claim to feature, without _statement (may be "
                    "repeated; default: those that carry the most)")
    ap.add_argument("--results", type=int, default=6, metavar="N",
                    help="how many results to feature when none is named")
    ap.add_argument("--check-globs", action="store_true",
                    help="compare the references with Rocq's .glob files "
                    "and stop")
    ap.add_argument("--keep-tex", action="store_true",
                    help="leave the LaTeX source next to the PDF")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    dev = Development(args.library or DEFAULT_FOCUS)
    if args.check_globs:
        missed, extra = dev.check_globs()
        print(dev.glob_summary)
        if args.verbose:
            for title, c in (("missed", missed), ("extra", extra)):
                print("--", title)
                for k, _ in c.most_common(60):
                    print("  ", *k)
        return
    an = Analysis(dev)
    dev.check_globs()
    report = Report(an, args)
    tex = report.build(dev.glob_note)
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="proof-map-")
    try:
        with open(os.path.join(tmp, "proof-map.tex"), "w",
                  encoding="utf-8") as f:
            f.write(tex)
        if args.keep_tex:
            shutil.copy(os.path.join(tmp, "proof-map.tex"),
                        args.output[:-4] + ".tex")
        try:
            res = subprocess.run(
                ["lualatex", "-interaction=nonstopmode", "-halt-on-error",
                 "proof-map.tex"], cwd=tmp, capture_output=True, text=True)
        except FileNotFoundError:
            sys.exit("proof-map: `lualatex` is needed")
        if res.returncode != 0:
            sys.stderr.write(res.stdout[-3000:])
            sys.exit("proof-map: lualatex failed")
        shutil.copy(os.path.join(tmp, "proof-map.pdf"), args.output)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("%s: %d pages" % (os.path.relpath(args.output), len(report.pages)))


if __name__ == "__main__":
    main()
