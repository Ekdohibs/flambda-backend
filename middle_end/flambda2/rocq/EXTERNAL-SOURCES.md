# External sources read for this work

This file records the sources outside this repository that those working on the Flambda 2
Rocq development have read for it, as far as they have been reported or recovered: papers,
books, slides, web pages, other projects' code or documentation, issue trackers (OxCaml's
included), the sources of upstream OCaml and of Rocq as installed, and searches made. It
has gaps, and says where. Files of this repository are not listed.

Words. "Upstream" is said only of the released OCaml 5.4.0 that this work compares OxCaml
with: its sources as installed and its behaviour. An issue or pull request of the OCaml
repository is "OCaml issue #N" or "OCaml PR #N", as one of OxCaml's is "OxCaml issue #N" or
"OxCaml PR #N"; a state of OCaml's development branch is "OCaml trunk". Dates are of 2026 and
times are UTC. A name in backquotes is a name of this tree; the names of files and
identifiers of outside sources are written plain. An entry has a key in square brackets,
such as [P4] or [K2]; a key is not the label of a session (P1, C1, R1 and N1 are sessions
too). `NAMES.md` and the paths that begin with `nominal/`, `terms/`, `rewrites/`,
`semantics/`, `heap/`, `requests/` or `status/` are under `middle_end/flambda2/rocq/`.

## The rule

- Whoever reads an outside source for the work writes its entry here when the reading
  happens, in the same series of commits as the work that uses it, or before.
- An entry says: what the source is (full citation or address; the commit or version for
  code); its licence if it is code; who read it and when; how much (whole, parts, abstract
  only, named only); what for; which files of the tree it bears on; what is owed to
  it (an idea credited, an interface used, nothing); and whether text or code was taken, and
  if so where and under what notice.
- One entry for a source, with all its readers under it. A source that is named in the work
  and is not recorded as read gets a line in section 7.
- The IP auditor reads each range of commits against this file. C0's gate fails a range whose
  accounts of provenance name an outside source that is not here.

## What is not certain

Said once here; it holds for every entry below.

1. **The entries are rebuilt, not remembered.** The work is done by sessions of a language
   model, named by their labels (N1, T5, ...). Each session's context has been summarised
   many times since the work began. Except where an entry says otherwise, what is dated before
   16 November is rebuilt from the session's notes, files and commit messages. For most
   sessions the notes of the first days (11 to 13 November) are thin. A reading not recorded
   at the time may not be recoverable now: "none by its records" means that, and not "never".
   Many of the accounts note that their session has tools that search the web and fetch
   pages.
2. **The first day has no account.** The 21 commits of 10 November were made before the team
   was formed, and no session answers for them. The files as they stood that evening cite no
   outside work (searched); two of them speak of "nominal sets".
3. **What the model knows cannot be traced.** Sessions can draw on pre-existing model
   knowledge whose training sources cannot be reconstructed. The places where the work leans
   on such knowledge with no source opened, by the sessions' own accounts:
   - the first files on names (10 to 12 November: `nominal/FlambdaNominalSpec.v`,
     `nominal/FlambdaAlphaSpec.v`, `nominal/FlambdaNameAbstractionSpec.v`, `NAMES.md`), which
     speak of nominal sets and cite nothing;
   - C4's design of the heap (`requests/C4-heap.md`): a heap of locations with fresh
     allocation, two runs related by a partial bijection on locations, a bisimulation for
     cyclic data, exits that nothing later tells apart. No source was read for these notions
     and C4 can name none to credit;
   - C1's and P1's arithmetic of the machine in `semantics/FlambdaPrimitiveSpec.v` (two's
     complement at a width, division that truncates, shifts, the swap of bytes; that x86-64
     shifts by the amount modulo the size of the register), and what `requests/P1-floats.md`
     and `status/P1.md` say of IEEE 754, of NaNs, of x86-64 and of arm64;
   - N2's canonical choice of bound names (`nominal/FlambdaRenameApartCanonical*`): common
     knowledge of the theory of binding, no document recorded;
   - T1's reading of the compiler's own pattern-matching code, helped by general knowledge of
     how OCaml compiles matches; what T1 and T5 say "the manual" of OCaml says;
   - Rocq's tactics and tactic languages (rewrite_strat, autorewrite, Ltac2): used from
     what the model knows and from trying them. Section 5 has the only recorded readings of
     Rocq's sources and manual;
   - test programs. The units under `middle_end/flambda2/tests/semantics/` were written in
     this work; none is recorded as copied from outside, and their shapes are common ones.
     Not in the tree: N2's 81 units of "textbook algorithms", T5's "small classic
     algorithms", and X5's programs after a public collection of benchmarks (section 7,
     [N10]).
4. **What the prover printed.** The accounts of the sessions that prove report that Rocq was
   asked for the statements of library lemmas (Search, Check, Print, About). Those queries are
   not recorded one by one. Most of these accounts say that they cannot rule out that an
   installed library file was once opened to read a lemma; none has a record of taking a
   proof or a definition from one.
5. **"Whole page"** of a tracker means what the fetch tool returned. Long pages may have come
   back cut, and code came back with its lines broken.
6. **Licences.** Each entry says on what its licence information rests (a licence file, a
   package's licence field, or a project page) and says where it was not checked. Licences of
   papers were not looked at.
7. **Helpers.** C4 had one helper session, R2 two and R4 three. Their accounts here are taken
   from their final reports, which name no outside source; the helpers were not asked.
8. **Instructions of the tooling.** The tool that the sessions run in loads pages of its own
   instructions (how its sandbox works, how to run Python, how to review code). They are not
   sources of the work and are not listed.

## Where this first version comes from

Each session wrote its own list on 16 November between 15:08 and 15:25: the orchestrator; C0,
C1 with C5, C2, C3, C4; N1, N2, N3; P1; R1 to R6; T1 to T7 and T5b (T6 is also X3); V2, V3, V4;
X1, X2, X4, X5; the session that prepared the reports of compiler bugs ("the lead of the
bug reports" below) and its helpers A and B. The IP auditor gave the list of what its reports
hold, and of what it fetched itself, and read the draft against its records. Every session
has answered. The reports of compiler
bugs, the testers' notes and the reviewers' scripts that the entries speak of are not in
this tree.

## 1. Papers and books

### 1.1 Read

| | Paper | Address that R3 fetched it from |
|---|---|---|
| [P1] | A. M. Pitts. Alpha-structural recursion and induction. Journal of the ACM 53(3), 2006 | https://www.cl.cam.ac.uk/~amp12/papers/alpsri/alpsri.pdf |
| [P2] | B. Aydemir, A. Bohannon, S. Weirich. Nominal reasoning techniques in Coq (extended abstract). LFMTP 2006; ENTCS 174(5), 2007 | https://www.seas.upenn.edu/~sweirich/papers/nominal-coq/lfmtp06.pdf |
| [P3] | C. Urban, S. Berghofer, M. Norrish. Barendregt's variable convention in rule inductions. CADE-21, 2007 | https://urbanchr.github.io/Publications/cade07.pdf |
| [P4] | B. Aydemir, A. Charguéraud, B. C. Pierce, R. Pollack, S. Weirich. Engineering formal metatheory. POPL 2008 | https://www.chargueraud.org/research/2007/binders/binders_popl_08.pdf |
| [P5] | E. Copello, Á. Tasistro, N. Szasz, A. Bove, M. Fernández. Alpha-structural induction and recursion for the lambda calculus in constructive type theory. ENTCS 323, 2016 | https://kclpure.kcl.ac.uk/ws/portalfiles/portal/62014231/Alpha_Structural_Induction_COPELLO_Published11July2016_GOLD_VoR.pdf |
| [P6] | F. S. Paranhos, D. Ventura. Nominal sets in Rocq. LSFA 2025; EPTCS 430 | https://arxiv.org/pdf/2509.25883 (revision not recorded) |

- **Read by** N1, 13 November: whole, by the word of its report; how N1 obtained each is
  not recorded. And by R3, 13 November before 18:05, to check N1's report: fetched from the
  addresses above, with these passages checked: [P1] Example 3.4, Theorems 3.5, 4.1, 4.5, 5.1,
  5.2 and section 7.1; [P2] sections 3, 5, 6; [P3] Definition 5 and Theorem 1; [P4] sections 4.2
  and 4.6 and Figure 6; [P5] section 3; [P6] sections 2 to 4. R3's record does not claim that
  every page was read.
- **For** N1's report on recursion and induction up to alpha-equivalence, which the human
  asked for, and R3's check of it. Neither the report nor R3's assessment is in the tree.
- **Bears on** the files that the report came before: `nominal/FlambdaEquivarianceTactic.v`,
  `terms/FlambdaTermInductionProofs.v`, `rewrites/common/FlambdaScopeTactic.v`,
  `rewrites/common/FlambdaSpineWalk{Spec,Proofs,NamesSpec,NamesProofs}.v`.
- **Owed: an idea, credited.** To keep a database of laws of renaming, with a tactic that
  proves the law of a new function by recursion on its argument and rewriting, is how Nominal
  Isabelle has it, as [P4] describes (section 4.6: the package "automatically derives
  equivariance lemmas"). No source of Nominal Isabelle was read.
- **Taken into the tree (this ledger included):** no implementation, statement or proof
  script was taken; this ledger contains the attributed short quotation above. N1's report
  and R3's assessment, which are not in the tree, quote and summarise the papers with their
  source.

### 1.2 Cited by N2 in the tree; whether N2 read them then is not recorded

Cited by N2 on 11 November, in the first hours of the work on the freshness quantifier.

| | Work | Cited in | What is owed |
|---|---|---|---|
| [P7] | M. J. Gabbay, A. M. Pitts. A new approach to abstract syntax with variable binding. 2002 | `nominal/FlambdaFreshQuantifierSpec.v` | the freshness quantifier ("for all but finitely many names") and its laws: an idea credited |
| [P8] | A. M. Pitts. Nominal Sets: Names and Symmetry in Computer Science. 2013; section 3.2 | the same | the same; and that in the papers every predicate has a finite support |
| [P4] | Engineering formal metatheory (above) | the same | a comparison: the quantifier is the cofinite quantification of locally nameless developments |
| [P9] | C. Urban, C. Kaliszyk. General bindings and alpha-equivalence in Nominal Isabelle. 2012 | `nominal/FlambdaFreshAbstractionSpec.v`, `requests/N2-fresh-quantifier.md` | a remark: work on binders of several names also defines the equality of abstractions with a permutation |

N2's working files hold no copy, note, address or search for any of the four. Whether a page
was opened that evening, or each was cited from what the model knows, cannot be told; what is
owed is stated as if they had been read. No text or code was taken, by the record: the files
are Rocq written against the tree's own names. `NAMES.md`, `status/N2.md` and other comments
say "the papers on nominal sets" and mean these. R1 reviewed these files and has no record of
reading the works.

### 1.3 Read in part

For the same survey and report of N1 as 1.1 and section 2; which files of the tree these
bear on is not separately recorded, and the bibliographic details that are missing here are
not in the record.

- Popescu, Nominal recursors as epi-recursors, 2024: the abstract only (N1, 13 November).
- Pagano and Solsona, 2022 (on Agda), and Annenkov's slides of 2017 (HIPERFIT): in N1's
  survey notes of 13 November under "related-work sections only". The titles are not
  recorded, nor how much of each was read. Nothing owed, nothing taken.

## 2. Other projects' code and documentation

**[C1] fasapa/nominal** (F. S. Paranhos; the development of [P6]):
https://github.com/fasapa/nominal at commit 78f41a6c7ca1b55814b7b7c7d734ad4decaac523.
Licence: MIT, "Copyright (c) 2026 Fabrício S. Paranhos" (its LICENSE file, read).
- **Read by** N1, 13 November, about 17:57 to 18:15: eleven of its twenty .v files
  (Prelude, PermT, Nominal, Fresh, Permutation, Name, NameAbstraction,
  FreshnessTheorem, Instances/SupportedFunctions, Example/Lambda,
  Example/Substitution), the LICENSE, the README, \_CoqProject and the listing of the
  tree; \_CoqProject again on 16 November about 12:30. Not built, not downloaded into any
  worktree. And by the IP auditor and its three readers, 16 November: the same eleven files
  and the LICENSE, whole, fetched at that commit, to compare them with the tactic's files.
- **For** N1's survey: could an existing development of nominal sets in Rocq be used as it is.
- **Bears on** the same files as [P1] to [P6]. The IP audit compared it with those files and with
  `rewrites/common/FlambdaScopedInductionProofs.v`, which is N3's; N3 reports no outside
  reading for it.
- **Owed:** the standard words and notions of the subject, and nothing more that anyone
  found. The audit observed one short resemblance of a conventional kind: our `Renaming_of`
  (in `nominal/FlambdaEquivarianceTactic.v`) and its PermAction each package the action of
  a renaming in a class of one field. It has no tactic for equivariance; ours uses no type
  class of permutations, no support and no setoid.
- **Taken into the tree (this ledger included):** nothing was taken but the attributed short
  quotation above, the copyright line of its licence. The audit found no concrete borrowed
  identifier, statement, proof or tactic; it says that this is a bounded observation and not
  a proof that nothing was.

**[C2] arthuraa/coq-utils** (A. Azevedo de Amorim): https://github.com/arthuraa/coq-utils at
commit 81eaf5b6c2ed58d230de0fe3ab1fc19e0c99e297. MIT by the project page; the text of the
licence was not read. N1, 13 November: the README and the listing of the tree only;
theories/nominal.v was not read. Its README names the paper "Binding operators for
nominal sets" (MFPS 2016), which was not read. For the same survey. Nothing owed, nothing
taken.

**[C3] Project pages only**, N1, 13 November, the same survey; no source file read, no licence
shown on the pages; nothing owed, nothing taken:
https://github.com/lewer/nominal-coq (its last commit of 23 September 2015);
https://github.com/annenkov/stlcnorm (its last commit of 17 July 2021);
https://www.seas.upenn.edu/~sweirich/papers/nominal-coq/ (the page of [P2]; its tarball was not
fetched); caipora-mat/nominal (Lean 4; seen in a search result only).

## 3. Issue trackers

Nothing in the tree bears on any of this reading. It bears on the reports of compiler bugs
that this work prepared: only the numbers of issues and pull requests were taken into them,
each with a few words of our own on how it relates.

**[K1] The OxCaml tracker**, https://github.com/oxcaml/oxcaml (issues and pull requests), read
on 16 November to learn which of the compiler bugs found by this work are already reported.

| Reader | Pages read whole | Also |
|---|---|---|
| the orchestrator | issues #7705, #7724, #7763, #7892, #7919, #7932, #7936 (came back cut), #7977, #8030, #8056; PR #8019 (the description, not the diff). Earlier the same day, not recorded page by page: issues #7888, #7889, #7998, #8002; PRs #7715, #7938, #7945, #8000, #8003, #8004, #8009, #8014, #8027. Later the same day, to check reference fields of the bug reports: issues #4797 and #7918; PRs #2663 and #2745 (the conversation page of each, not the diff) | the lists of open and of recently merged pull requests |
| the lead of the bug reports | issues #3219, #4803, #7884, #8056; PRs #7450, #7498, #7562, #7588 | 7 list pages and 38 searches (appendix) |
| helper A | issues #1390, #1792, #4797, #6374, #6751, #6825, #7576, #7724, #7763, #7977, #8030; PRs #1870, #2663, #3832, #6396, #6502, #6558, #6753, #6974, #7371, #7715, #7810 | 46 queries, in 50 fetches of pages of results (appendix) |
| helper B | issues #5438, #7724, #7929, #7930, #7931; PRs #6037, #7642, #7875, #8011, #8034 | 60 searches (appendix) |

- **Taken:** the numbers. Three programs shown on issue pages were also typed out, to run
  them: the first program of #4803 (the lead), and the programs of #7763 and #7977 (helper
  A; the orchestrator, whose copies T1 also read and ran on 16 November about 14:10 without
  opening the tracker). None of the three is in the tree or in the reports of compiler bugs.
  Their licence was not checked.
- The titles in the result lists of the searches were read too. Two references of the
  reports (#2745 and #7918) first rested on such a title or on a mention on another page;
  the orchestrator then opened both pages, on 16 November, and the wording of the references
  agrees with them.

**[K2] OCaml issue #15171, OCaml PR #15184, and OxCaml issue #7884.**
- **Read by** T1, 13 November between about 11:20 and 12:20. How they reached T1 (a web tool
  or another session's message) and how much was read is not recorded: at least the report
  with its program, where the repair is, and what its test is; #7884 may only have been
  named. And by a reader of the IP auditor, 16 November, 15:30 to 15:31:
  https://github.com/ocaml/ocaml/issues/15171, https://github.com/ocaml/ocaml/pull/15184 and
  its page of changed files (the four hunks shown; the test at head
  dd7da143c23f18c825fc59b1f66e9d469a4cbc2a), as the fetch tool returned them (the numbers in
  the body of the issue were lost in the rendering).
- **For** T1: to say that a stop of the compiler met by generated test programs ("impossible
  empty matrix in a Total match") is known, and repaired by that pull request, and so not a
  finding. For the auditor: to compare the pages with one test program of T1's, of four
  lines, which is not in the tree and not in the reports of compiler bugs.
- **What the comparison observed.** T1's program re-expresses the trigger of the issue: a
  case that takes everything, followed by a guarded or-pattern that cannot be reached. No
  complete line of code and no complete program of the two shown in the issue, or of the
  regression test of the pull request, matches it. Its header names the issue. Whether T1
  wrote it by editing the report's program or afresh is not recorded.
- **Owed:** the knowledge that the stop is known and repaired, and the shape of the trigger,
  which the program's header credits. **Taken into the tree (this ledger included):** nothing
  was taken but the attributed short quotation above, a message of the compiler, which this
  repository's own compiler has too. No program and no other text of the pages is in the
  tree. The auditor's reader copied short excerpts of the pages for the comparison; they are
  not in the tree. No licence is stated on the pages for the programs shown.

**[K3] OxCaml issue #7934 and OxCaml PR #7969.**
- **Named** in X5's records of 14 November; whether X5 opened either page, which parts and
  which version, is not recorded. **Read by** a reader of the IP auditor, 16 November, about
  15:29 to 15:36:
  https://github.com/oxcaml/oxcaml/issues/7934, https://github.com/oxcaml/oxcaml/pull/7969
  and its page of changed files (the five diffs shown, at head
  7b2fca1b8c3fd5603bc52b3cdabd27ddd6b5756b), as the fetch tool returned them (the marks of
  added and removed lines were lost, and the comments did not load).
- **For** X5: to say that a fault found in the audit of the runtime (a hexadecimal float
  string that is rounded twice) was already reported. For the auditor: to compare the pages
  with two test programs of X5's, which are not in the tree and not in the reports of
  compiler bugs.
- **What the comparison observed.** None of the three hexadecimal strings of X5's programs
  (0xDFFFFFFFFFFFFFFp-1132, 0xDFFFFFFFp-179, 0x1.BFFFFFFFFFFFFFEp-1073) is on the
  pages. The programs and the pages have the same ordinary alias of the Float32 module and
  the same uses of its interface, and the same ideas of the test ("midpoint/double-spelling"
  and "literal-versus-string", in the audit's words). These observations do not settle
  whether X5 used the pages as models; the origin remains uncertain.
- **Owed:** the knowledge that the fault is known. **Taken:** nothing of the pages is in the
  tree. Whether anything of the pages is in X5's two programs is not settled by the
  comparison. The auditor's reader copied short excerpts of the pages for the comparison;
  they are not in the tree. No licence is stated on the pages, for the pages or for the test
  shown.

## 4. Upstream OCaml as installed

**[U1] OCaml 5.4.0** in the opam switch 5.4.0 of the machine: the installed library
(lib/ocaml) and the source tree that opam keeps (ocaml-compiler.5.4.0). Licence:
LGPL-2.1-or-later with the OCaml linking exception (the package's licence field and the
first lines of its LICENSE, read on 16 November by T3, T5b and X5).

| Reader, when | What was read | What for |
|---|---|---|
| X5, 13 and 14 November | runtime/: the whole directory compared with this tree's by diff and by lists of function names; read in it memory.c (alloc_shr), floats.c (caml_ldexp_float and its unboxed form, with their comment), bigarray.c (the lines that store a double as float16); which other files, not recorded. stdlib/stdlib.ml or .mli (raise). lambda/lambda.ml (make_key) | to tell which C code of the runtime is OxCaml's own, and to say of a fault whether upstream has it too |
| T3, 14 November | lib/ocaml/format.ml lines 1355 to 1392 (format_text), compared by diff with this tree's `stdlib/format.ml`: the same text; lib/ocaml/obj.ml line 24 | the same question for four faults of tail calls |
| T5, 13 and 14 November | lib/ocaml/effect.ml (match_with and try_with); the val and external lines of the library's .mli files, compared with this tree's; the interfaces of compiler-libs, against which one probe tool is built (which interface files were read is not recorded) | the same question for one fault of effect handlers; to list the names that this tree's library adds |

- **Bears on** nothing in the tree: the testers' notes on their findings and, through them,
  what the reports of compiler bugs say of upstream.
- **Owed:** facts about what upstream does. **Taken:** short quotations, with their source
  named: T5 quotes one line of effect.ml ("| Some f -> f k") in a note, and the words "where
  OCaml 5.4.0 has f k" stand in one report of a compiler bug; X5 quotes, in a note, one
  comment that upstream's runtime/floats.c and this tree's both have. The notes are not in
  the tree. Into the tree nothing was taken but the attributed short quotation above, the
  line of effect.ml.
- **Run and not read:** the 5.4.0 compilers were run as a reference by T1 (also those of
  OCaml 4.14.0 and 4.12.0), T3, T5, T5b, T7, X5, the orchestrator, and the lead of the bug
  reports with its helpers.

**[U2] Sentences about upstream's sources whose origin is not recorded.** Each may rest on the
installed source tree, on this tree's own files of the same name, or on what the model knows.
T7 (14 November): that upstream has the same passes in lambda/tmc.ml, lambda/simplif.ml
and lambda/translcore.ml. T3: two sentences (two tests are upstream's too; what
lambda/simplif.ml does upstream). T5: three (where rewrite_double_underscore_paths
starts, a rule of translprim.ml, how upstream's fibers grow). T5b: two (what alloc_shr
calls upstream; that upstream has no Iarray.init_local). T4: one, on the order of two
passes of lambda/simplif.ml, written from general knowledge and not verified. T2 says of
three external lines of a test program that they have the text of upstream's
stdlib/stdlib.ml, from what the model knows; no copy was opened. All of these are in the
testers' notes or in the reports of compiler bugs; none is in the tree.

**[U3] OCaml 5.4.1** of the opam switch that builds Rocq and the tools: run by everybody; no
source or interface file of it is recorded as read by anyone.

## 5. Rocq and other tools as installed

**[R1] The Rocq Prover 9.3.0** (packages rocq-core and rocq-runtime 9.3.0, rocq-stdlib
9.2.0, with Ltac2). Licence: LGPL-2.1-only (the packages' licence fields and the LICENSE of
rocq-core, read on 16 November by several sessions). Its reference manual: Open Publication
License v1.0 or later, with neither option elected (its doc/LICENSE, read by R3 and R6).
Importing a module of the standard library, or naming one of its lemmas, does not by itself
establish that its source was read, and is not an entry. What was read of its sources and
documents:

| Reader, when | What was read | What for; bears on | Owed; taken |
|---|---|---|---|
| P1, 11 or 12 November; and 16 November | Floats/SpecFloat.v, PrimFloat.v, FloatAxioms.v of rocq-core. For the first reading the record does not say how much, nor whether the files were opened or Rocq was asked to print their contents. On 16 November, for this file: the first 12 lines of SpecFloat.v, the lines that define four names, and a count of the axioms of FloatAxioms.v | where a model of floats could come from; `requests/P1-floats.md`, `status/P1.md`. No .v file of the tree names them | nothing; four names of definitions and two of modules stand in the prose |
| R3 and R6, 15 November, about 05:22 to 05:30 | in the source package of rocq-core 9.3.0, parts of toplevel/coqc.ml, toplevel/vernac.ml, lib/flags.ml, lib/cErrors.ml, vernac/vernacControl.ml, vernac/vernacentries.ml, vernac/topfmt.ml, sysinit/coqargs.ml, sysinit/coqinit.ml, stm/stmargs.ml, test-suite/Makefile, and of the manual doc/sphinx/proof-engine/vernacular-commands.rst (the command Fail); R6 also vernac/himsg.ml | why "Fail Check" prints nothing in a batch run, and the flag -test-mode for the reviewers' scripts; nothing in the tree | the flag and the wording of the messages are Rocq's. Not in the tree, with the source named and no licence notice added: R3's note quotes short fragments of the sources (a condition; the command of the test suite), and R6's review scripts match short messages of Rocq word for word (such as "The command has indeed failed with message:"). Into the tree nothing was taken but that attributed short quotation |
| N1, 16 November: about 14:10 to 14:50 for the three files named; for anything further, the time is not recorded | of the Ltac2 library as installed: parts of Control.v and Init.v, the declarations of Ltac1.v, and the listing of its directory. The tactic also calls Env.expand, Std.VarRef, Constr.Unsafe.iter and functions of Message, Ref, List, Int and Ident by name; N1 has no record either way of opening the files that define them. So for the rest of the installed Ltac2 directory: parts may have been read on 16 November, by a search for a name; which files, how much and when is not recorded. Whether N1 also looked at Rocq's reference manual is likewise not recorded either way | to write the tactic and its bound (a limit of time, stopping a call, the number of goals); `nominal/FlambdaEquivarianceTactic.v` | an interface used (functions called by name); no text |
| N2, 11 November | Utf8_core is named in `nominal/FlambdaFreshQuantifierSpec.v` as the model for making a letter a keyword; whether the installed file was opened is not recorded | that comment | the remark |
| C0 and C2, 16 November | the help text that "rocq compile" prints (C2 also rocq and "rocq native-precompile") | timing and rebuilding in their own working trees; nothing in the tree | nothing |
| N2 and C0, 18 November, between 11:49 and 12:05 | no source text of the library read. By N2: the names of the source files of the library as installed and of the packages of the installation, by listings; what Rocq prints of its version; Rocq's answers when asked to locate 23 names of definitions and 7 of modules (it prints full names and no body); and the size and the sha256 of Floats/SpecFloat.v, computed by a tool over its bytes, to check a reader's report of it. By C0, for its own account of the same files: the names of the compiled files of the twelve modules, of the directories of the installation and of its packages, by listings; the first five lines of the file META of rocq-stdlib as installed (a file of the package's installation, not of the library's source), after a search of the META files of rocq-core, rocq-stdlib and rocq-runtime for the line of the version; what Rocq prints of its version and of where its library is; and Rocq's answers when asked to locate the twelve names of modules and twelve of definitions | which module of the library each generated file of `middle_end/flambda2/tests/semantics/extracted_rocq_library/` is made from: the two paragraphs after this table | nothing; no text or code taken |
| C0, 18 November, from 12:23 | of the packages as installed. By hand, between 12:23 and about 12:40. The file doc/rocq-core/LICENSE of rocq-core 9.3.0 (24381 bytes, 458 lines, sha256 d4594b82f4d50840df6a7e9d14132a8c0a3cc05d0ac46d15310a264a1f75447e): its first 40 lines printed, its headings and the words copyright, inria, rocq and coq searched for, its size and hash computed; the same hash computed of the LICENSE of rocq-runtime, of rocq-devtools and of the source of rocq-stdlib 9.2.0 as opam keeps it. The fields license, version, authors and maintainer of the opam files of rocq-core 9.3.0 and rocq-stdlib 9.2.0. Of the source file of each of the twelve modules of the paragraphs after this table (under lib/coq of the installation: theories/ for rocq-core, user-contrib/Stdlib/ for rocq-stdlib), the header and little more: the hash of its nine lines; the line before them where there is one; a search of its first 25 lines for a year, an author or a copyright; the nine lines of theories/Init/Datatypes.v printed; lines 10 to 14 of Arith/PeanoNat.v printed, one of which is a line of source that is not of the header (a line that begins "From Stdlib Require Import"); and, later, a search of ZArith/BinInt.v for the words "Initial author", which printed its line 21. The names of the files of the packages' directories, by listings; a file CREDITS of the source of rocq-core was seen in a listing and not opened. No definition, lemma or proof of the library printed or read. After those readings C0's scripts hashed, compared and copied the same files | the files `LICENSE` and `README.md` of `middle_end/flambda2/tests/semantics/extracted_rocq_library/` | COPIED into the tree, with the source of each named in that `README.md`. The whole of doc/rocq-core/LICENSE, byte for byte, as `LICENSE`: the text of the GNU Lesser General Public License, version 2.1. Lines 1 to 9 of theories/Init/Datatypes.v (sha256 of the nine lines b4a9db4ebd8900837ba85450347bba5b605d515252c193f409b29ad1e619444e): the header of Rocq's source files, which states the copyright and the licence, and whose third line is the copyright line. And, quoted in the same file: the first line of BinInt.v, BinPos.v and PosDef.v, the words of line 10 of Arith/PeanoNat.v and line 21 of ZArith/BinInt.v, the last two of which name an author |
| N2, 18 November, between 12:24 and 12:50 | no installed source file of the library opened; what was printed is text that Rocq generated, and lines of the sources as C0 quotes them. Between 12:24 and 12:30: Rocq's extraction run in scratch directories outside the tree, on two files written for the purpose: one that asks for SFcompare of Floats/SpecFloat alone, where the two .mli files that Rocq wrote for SpecFloat and IntDef were printed (a type of four constructors and two declarations) and the lines of its files counted; and one with a definition that calls 17 functions of the library, each put onto a function of OCaml's library by a directive, where the files that Rocq wrote were printed (those of the library's modules empty, but for one short function, bool_dec, in that of Bool) and compiled by OCaml, the first time with an error and the second without. The names of the compiled files of the installed directory of Rocq's extraction modules, by a listing. Of another package as installed, zarith 1.14: the field license of its opam file. Between 12:47 and 12:50: the header of Rocq's source files and the three other lines, read as C0's `README.md` of the directory quotes them | the probes: a page for the human on what not generating these files would take, which is not in the tree. The quoted lines: the paragraphs after this table | nothing. That page names two of Rocq's extraction modules beside the one that `validation/checks/FlambdaExtract.v` takes, and gives zarith's licence field; what it says that those two do is from the model's knowledge (point 3 of "What is not certain"), not from a reading |

Files generated from Rocq's library are in the tree, in a directory of their own.
`rocq/Makefile` compiles `middle_end/flambda2/rocq/validation/checks/FlambdaExtract.v` in
`middle_end/flambda2/tests/semantics/extracted/`, where Rocq then writes OCaml files, and
passes those files through ocamlformat. That .v file names definitions of this tree only
(its command Separate Extraction), and Rocq writes an .ml and an .mli for each module that
they use. The same rule of the Makefile then moves to
`middle_end/flambda2/tests/semantics/extracted_rocq_library/` each of those files whose
module is not a .v file that the Makefile compiles; it keeps no list of their names. No line
of the generated files of either directory is written by hand. Before landing, C0's gate is
to delete them, make the extraction again and check that each is written with the bytes of
the commit. Twenty-four files are in the second directory: an .ml and an .mli for each of
twelve modules of Rocq's library, named BinInt, BinNums, BinPos, Bool, Datatypes, List,
ListDef, Nat, NatDef, PeanoNat, PosDef and Specif (the two files of Specif are empty). Nine
of the twelve came into the tree with commits of 10 November (point 2 of "What is not
certain"), and Bool, Nat and NatDef with commits of 11 November. Until 18 November all were
in the first directory, among the generated files of the tree's own modules, and no file of
notices was beside them. The files have no header: the extraction writes none. Beside them
the second directory has a file `LICENSE`, which is the text of the GNU Lesser General
Public License, version 2.1, as rocq-core 9.3.0 installs it, and a file `README.md`, which
has a table of the twelve modules and the header of Rocq's source files, where Rocq states
the copyright and the licence. The first is a copy, whole; the second is written for the
tree and quotes that header and three other lines of Rocq's source files. What was copied is
entered in C0's row of 18 November in the table. The directory has nothing else but its dune
file.

What is known of where the files come from is of one day. As the library is installed on 18
November (rocq-core 9.3.0, rocq-stdlib 9.2.0), Rocq has modules of those names as follows.
In rocq-core alone: Datatypes and Specif (of Corelib.Init). In rocq-stdlib alone: BinInt (of
Stdlib.ZArith), BinPos (of Stdlib.PArith), Bool (of Stdlib.Bool), List (of Stdlib.Lists) and
PeanoNat (of Stdlib.Arith). In both: BinNums, ListDef, Nat, NatDef and PosDef, and the
definitions of those five that were located are of rocq-core's modules
(Corelib.Numbers.BinNums, Corelib.Lists.ListDef, Corelib.Init.Nat, Corelib.BinNums.NatDef,
Corelib.BinNums.PosDef). That is by the names of the modules and of one to three definitions
of each file, which Rocq was asked to locate (the row of N2 and C0 in the table), not by
every definition; the table of that `README.md` gives the same, with the source file of each
module. On 18 November each of the twelve source files has the same header of nine lines,
which C0 compared by a hash and copied once (its row). What is not known: which versions of
Rocq and of ocamlformat wrote the files in the commits of 10 to 13 November that added or
changed them, and what notice the sources carried then. None of the accounts cited here
records a reading of a definition in the original Rocq source files of these twelve modules.
Besides the header and other notice lines, C0 printed one line of source code: the line of
Arith/PeanoNat.v beginning "From Stdlib Require Import" (C0's row). Readings of the 24
generated files already in this repository are outside this inventory's scope. The licence
of the library is as said above.

**[R2] The other tools, run and not read:** OCaml 5.4.1, dune 3.24.2, menhir 20231231,
ocamlformat 0.29.0, Python 3.6.8, git, GNU Make, perl, bash, the GNU coreutils, gcc and ld.
No manual or source of any is recorded as read. Python's own integers and its struct and
fractions modules gave the expected answers of some tests (`requests/C1-int-ops.py`; and
test programs of X5 that are not in the tree).

## 6. Web searches

- N1, 13 November, for its survey ([C1] to [C3]). The queries: "nominal sets Coq Rocq library
  formalization github"; "arthuraa coq-nominal Binding operators for nominal sets Coq"; "Coq
  library equivariance tactic nominal finitely supported some/any stdpp"; "opam coq- rocq-
  package nominal sets"; "Coq Rocq equivariance tactic derive permutation action inductive
  type nominal datatype plugin". The pages of results were not kept.
- X5, 14 November: one search for an existing report that ldexp uses only the low 32 bits of
  its exponent. It found none. The query and the pages shown are not recorded.
- The searches of the OxCaml tracker of 16 November are in [K1] and in the appendix.
- No other session has a record of a web search.

## 7. Named in the work; not recorded as read

| | What | Where it is named, and how |
|---|---|---|
| [N1] | Pitts, Nominal Sets (2013); Norrish, Recursive function definition for types with binders (2004); Urban and Berghofer, A recursion combinator for nominal datatypes (2006); Gordon and Melham, Five axioms of alpha-conversion (1996); Charguéraud, The locally nameless representation (2012); Sato and Pollack (2010); Pollack, Sato and Ricciotti (2012); Swan (2017); Choudhury (2015) | the bibliography of N1's report (not in the tree): cited from the papers of 1.1, from an abstract or from memory |
| [N2] | Pottier, Visitors unchained (2017), with alphaLib; Allais, Atkey, Chapman, McBride and McKinna, A type and scope safe universe of syntaxes with binding (2018) | N1's report. **Owed: an idea, credited from sources not read:** a generic walk of a term whose law of renaming is proved once is of the kind of alphaLib's visitors and of Allais and others. The walk of `rewrites/common/FlambdaSpineWalkSpec.v` is of that kind; its text is the recursion of a function that the tree already had |
| [N3] | Nominal Isabelle (Nominal2); paramcoq; the derive of coq-elpi; LNgen with Metalib and Ott; Autosubst 2; stdpp, MathComp, coq-deriving, extructures, Equations, the Categories library | N1's report and survey notes: named in what was read, or from memory; none fetched, no licence looked at |
| [N4] | IEEE 754; Flocq; CompCert; the manuals of the x86-64 and arm64 processors | `requests/P1-floats.md`, `status/P1.md`, `requests/C1-primitives.md`: named from what the model knows. The request says itself that its rule for NaN was not read against a manual. X5's entries name IEEE 754 in the same way |
| [N5] | The reference manual of OCaml | T1's and T5's notes on expected answers of tests. A copy of the manual is in this repository; no copy from the web is recorded |
| [N6] | Numbers of OCaml and OxCaml issues and pull requests that come from this repository itself (subjects of its commits, its `Changes` file, comments of its sources and tests) | in the testers' notes and in X1's; no tracker page was opened for them |
| [N7] | Addresses in comments of the compiler's own sources (a PDF of the x86-64 ABI, the wiki of AddressSanitizer, a Wikipedia page) | seen by R2 and its helpers in files of this repository; not followed |
| [N8] | OxCaml PR #7450 and OxCaml issues #7763 and #7977 | named in the orchestrator's rulings to every session; read only by the readers of [K1] |
| [N9] | The option `-dcanonical-ids` of this repository's `typing/ident.ml`, which its `Changes` gives to OCaml PR #13787 | X1 read the in-tree file and follows its idea in a change to `backend/ssa/ssa.ml` that is not in this tree; no line copied |
| [N10] | The Computer Language Benchmarks Game: six of its problems (n-body, spectral-norm, fannkuch-redux, binary-trees, mandelbrot, k-nucleotide) | X5, 13 November: ten test programs, none in the tree. **The record cannot say whether a page or a program of the collection was opened.** Data of the collection is in X5's files (the initial conditions of n-body, the constants of the generator and the five words of k-nucleotide, the output line of fannkuch, the published results used as checks). No record that program text was copied; no licence notice in the files |

## 8. Sessions with nothing else to enter

The accounts supplied identify no paper, book, web page, tracker page or other project's
source read, no web search made, and no source file of upstream OCaml read, beyond the
entries above, for: C0, C1 with C5, C2, C3, C4, N3, R1, R2, R4, R5, T2, T4, T6 (X3), V2, V3,
V4, X1, X2, X4; and the same for the helper of C4, the two of R2 and the three of R4, by their
final reports and not by lists of their own (point 7 above). These sessions used Rocq and
the other tools as installed (section 5). No account says more than "none by my records":
each has its own section on what its records cannot settle, and point 1 above stands for
all of them.

## Appendix. The searches of the OxCaml tracker, 16 November

Only the lists of results were read, but for the pages named in [K1]. One query to a line; a
note in brackets after a query is ours.

The lead, list pages:

    is:issue is:open
    is:issue created:>2026-10-01   [two pages]
    is:issue is:closed closed:>2026-09-30
    is:issue closed:>2026-10-01 created:<2026-10-02
    is:issue unboxed
    is:issue unboxed record match

The lead, searches:

    or-pattern
    exclave region
    unsafe_to_string
    to_cmm load
    safe-subst
    parmatch
    lub unboxed
    johnyob unboxed record pattern
    nan sign
    binding operator external
    binding operator assertion
    heavy-invariants
    dir32
    "Cannot rebind"
    rebind
    join "Stack overflow"
    inconsistent assumptions
    Consistbl
    fno-avx
    internal-assembler
    result-types
    static exception
    "Unbound static exception"
    letop
    string load bytes coeffect
    extra args exception handler
    End_try_region
    End_region flush
    float identities simplify
    sign_bit
    stack allocated read after region
    try reference raise wrong value
    translate_apply
    nan float multiplication
    meet_and_join cycle
    canonical concrete type
    concrete-types-only-on-canonicals
    is:pr is:open to_cmm

Helper A, with is:issue:

    tail call
    created:>2026-10-01   [three fetches: two pages of results, then no more issues]
    "stack overflow"
    tailcall
    region
    unboxable
    match-in-match
    n-way
    meet-strategy

Helper A, without:

    tail_mod_cons
    letop
    "let-operator" OR "let*" OR "binding operator"
    "tail call"   [three pages]
    tailcall region   [the page came back cut, with no results shown]
    optional argument default
    split_default_wrapper
    simplif beta
    over-application OR over_application OR overapplication
    over_app_region
    "Fun.id" tail
    identity tail position
    End_region
    "Local allocation stack overflow"
    join_env
    bfs meet
    layout_poly
    poly_ local closure
    Static_const compare
    Immutable_float_block
    sharing constants float
    simplify-stubs OR simplify_stubs
    "indirect tailcall" OR "indirect call" zero_alloc
    "warning 55" OR inlining-impossible
    newer_version_of OR "dead code" functor
    lxor OR xor simplify
    create_switch OR "switch arms" OR "same arms" debuginfo
    Apply_cont Debuginfo compare
    match-in-match region
    nontail OR "not a tail call" OR "constant stack"
    "constant stack"
    "x + 0" OR "lxor 0" OR "identity element" OR "neutral element"
    "unreachable switch case" OR "inverse relation" OR recover_inverse_relations
    join-algorithm wrong
    Rc_close_at_apply OR applytail
    unboxed_param_region OR "unboxable" local tail
    Lkindinstantiate OR "layout polymorphic" local region

Helper B, issues:

    backtrace
    created:>2026-09-01   [three pages]
    makearray_dynamic
    float32 nan
    "mixed block"
    Out_of_memory
    "out of memory"
    ldexp
    float16
    reshape
    effect handler
    Dynarray
    make_matrix
    arrayblit
    comprehension
    oclassic attribute
    flambda_oclassic
    poll
    unboxable
    tailcall
    tailcall warning
    inlined warning
    method order
    "evaluation order"
    debuginfo location
    "local stack"
    compare unboxed array
    "stack overflow"
    Bigarray
    Marshal
    region local allocation
    "polymorphic compar"

Helper B, pull requests:

    backtrace
    reraise
    poll error
    comprehension region
    makearray_dynamic
    float32 compare
    ldexp
    float16
    reshape
    Dynarray
    effect.ml nontail
    Out_of_memory
    arrayblit
    tailcall
    split_default_wrapper
    unboxable
    oclassic attribute
    make_key
    Lsend order
    compare mixed block array
    opt_level attribute
    try_with
    emit_tail_infos
    poll insertion
    compare_unordered
    comprehension local
    bigarray created:>2026-09-01
    flambda_oclassic
