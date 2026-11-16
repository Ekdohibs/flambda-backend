# Scaling the Flambda 2 Rocq work to many agents

This is a plan for taking the Rocq semantics of Flambda 2 from one agent
working serially to ten or twenty working in parallel. It says what
exists, what the work is, how to cut it into pieces that do not collide,
and the rules the agents follow so that they do not.

Read first: `rocq/HACKING.md` ("The semantics of Flambda 2"),
`ROCQ-REVIEW-GUIDE.md`, and the header comment of each `*Spec.v` file in
the folders of this directory (`README.md` has the tree).

## 1. Where things stand

About 14,000 lines of Rocq, 82 proved statements, no admitted proofs.

- **Terms and machine.** `FlambdaModel.v` is a fragment of Flambda 2
  (`Let`, `Let_static`, both `Let_cont`s, `Apply`, `Apply_cont`,
  `Switch`, `Invalid`; closures with value slots; static constants).
  `FlambdaSemanticsSpec.v` is a deterministic abstract machine, with
  equivalence and refinement of states and of expressions.
- **Reasoning tools.** Contexts and congruence (`FlambdaContext*`),
  renaming and alpha-equivalence (`FlambdaNominal*`, `FlambdaAlpha*`),
  inlining (`FlambdaApply*`), static bindings and well-formed units
  (`FlambdaStatic*`, `FlambdaWellFormed*`).
- **Validation.** `FlambdaValidateSpec.validate_unit` is one extracted
  function: it normalises the term before and the term after Simplify by
  proved rewrites (`FlambdaNormalise*`, `FlambdaUnusedStatics*`,
  `types/naive/FlambdaAliases*`), and compares them
  (`FlambdaCompare*`). `validate_unit_sound` says that what it accepts is
  a correct simplification of the unit run from empty environments.
- **Tests.** `../tests/semantics/` holds `.fl` examples;
  `../tests/tools/flsemantics.ml` runs the real Simplify on each and
  calls `validate_unit`. Fifteen of eighteen are validated; the other
  three are deliberately not checkable.
- **Types.** `types/naive/` is a first, naive model: four type
  constructors, a typing environment as a list, alias following.

What is missing, roughly in order of how much it limits us:

1. Primitives are a parameter of the model (`Parameter prim`,
   `eval_prim`), realised by hand-written OCaml in the test tool, with
   three axioms about them.
2. No mutable state, no effects, no exception handlers with trap
   actions, sets of closures with one function only, no kinds on
   parameters.
3. The types model covers almost nothing of `../types/`.
4. The normaliser knows a handful of rewrites. Simplify knows hundreds.
5. Trusted OCaml: the converter from compiler terms to the model, and the
   aliasing of Simplify's new code ids to the old ones.

## 2. The two aims, and the method

**Aim A: validate more.** Widen the model and the normaliser until
`validate_unit` accepts what Simplify does to real programs, not only to
hand-written examples.

**Aim B: verified components.** Write parts of Flambda 2 in Rocq, prove
them, extract them, and show that the extracted code agrees with the
existing OCaml. The typing library (`../types/`) is the first candidate:
it is self-contained, mostly pure, and everything in Simplify rests on
it.

The two meet: a verified `meet`, `join` or prover is exactly what the
normaliser needs to follow Simplify's reasoning.

**The method for every piece**, using the existing OCaml as the guide:

1. *Transcribe.* Take one OCaml module or function. Write the Rocq
   definition next to it in meaning and in names. Where the OCaml uses
   mutation for speed, write the pure function it computes. Where it uses
   maps and sets, use the shared finite-map module (section 5). Say in
   the header comment which OCaml file and function it models and every
   deliberate difference.
2. *State.* Write `Definition foo_statement : Prop` in the `Spec` file:
   what the function guarantees in terms of the semantics (for a type
   operation, in terms of which values satisfy a type). Statements are
   what a human reviews, so they must be short and readable.
3. *Prove.* In the `Proofs` file. No `Admitted`, no new axioms.
4. *Extract and compare.* Extract the function. Add a differential test
   that runs it and the original OCaml on the same inputs (generated, and
   taken from real compilations) and compares the results through the
   converter. A disagreement is a bug in the model or in the compiler;
   both are worth knowing.
5. *Use.* Wire it into `normalise_before` / `normalise_after`, or into
   the tool, and add `.fl` examples that need it.

Step 4 is what keeps a transcription honest. A proof says the Rocq
function is sound; only the comparison says it is the function the
compiler runs.

**Out of scope for now:** match-in-match and continuation
specialisation, mutable unboxing (`simplify/flow/mutable_unboxing.ml`),
the reaper (workstream S7 below is kept for later and is not to be
started), anything about cost or inlining *decisions* (a decision is an
untrusted input; only its effect is verified).

## 3. Why naive parallelism fails here

Almost every proof depends on three things: the type `expr`
(`FlambdaModel.v`), the function `step` and the type `state`
(`FlambdaSemanticsSpec.v`). A change to any of them breaks proofs in
every file. Recent experience: adding one `if` to `step` for recursive
continuations needed fixes in eleven files.

So the work divides into two kinds, and they must be scheduled
differently:

- **Leaf work** adds new files that depend on the core but that nothing
  else depends on: a new rewrite and its proof, a type operation, an
  example. Any number of agents can do leaf work at once.
- **Core work** changes `expr`, `state`, `step`, or a relation every
  proof uses. Only one core change can be in flight at a time, and
  everyone else must rebase after it.

The plan below is built around that: one agent owns the core and
serialises changes to it; the others do leaf work in files they own.

## 4. Phase 0: make the repository safe for parallel work

One agent, before any fan-out. A few days. Nothing else should start
until this is merged, because every item removes a place where parallel
branches would conflict.

1. **Step lemmas.** Give `step` a lemma for each constructor
   (`step_let`, `step_let_static`, `step_apply_cont`, ...) stating what a
   state steps to under which conditions, and rewrite the existing proofs
   to use them instead of `simpl` on `step`. After this, a core change
   updates the lemmas, and most proofs do not notice.
2. **State as a record.** Make `state` a record with named fields and
   projections, and stop destructing it by position. Adding a field (a
   heap, a trap stack) then breaks nothing that does not mention it.
3. **Finite maps.** Choose one representation for environments (today
   association lists with `lookup`) and put its lemmas in one file. The
   types work needs maps and sets heavily; it must not grow its own.
4. **One expected file per example.** Replace
   `tests/semantics/semantics.expected` by `X.expected` beside each
   `X.fl`, and generate the dune rules from the directory listing. Two
   agents adding examples then never touch the same file.
5. **Makefile and extraction lists.** One file name per line, sorted, in
   `rocq/Makefile`; the same for the `Separate Extraction` list in
   `FlambdaExtract.v`. Additions then merge cleanly.
6. **Extracted files.** Decide that only the integrator commits
   `tests/semantics/extracted/`. Workstream branches regenerate them
   locally and do not commit them. (They are generated; committing them
   on every branch guarantees conflicts.)
7. **A trust ledger.** A file listing every axiom and every piece of
   trusted OCaml, with the reason. `make assumptions` should check the
   axioms against it. No workstream may add to it without sign-off.
8. **CODEOWNERS-style table.** The ownership table of section 6, checked
   in, kept current.

## 5. Shared conventions

These already hold; with many agents they must be enforced.

- Each topic is a pair `FooSpec.v` (definitions and
  `Definition x_statement : Prop`) and `FooProofs.v`
  (`Theorem x : x_statement`). `make -C rocq all` checks the proofs,
  checks that every statement is proved, and regenerates the extraction.
- A `Spec` header says what is to be reviewed and against which OCaml.
- No `Admitted`. No axiom outside the trust ledger.
- Lines under 80 columns. Say "Rocq". Write `[ |` with the space.
- Vocabulary: a `Let_cont` has a *handler*, and the *body* is what
  applies it; never "handler body". Never "object": say *statically
  allocated value*, with `Static_` prefixes.
- A statement that holds only vacuously is a bug. For every refinement
  theorem, add an example where the source is defined.
- Prefer equivalences. Use refinement only where the source may be
  undefined and the target not. Rewrites of the term after Simplify must
  be equivalences.
- A rewrite that needs to know nothing of its context is stated as
  `forall e, expr_refines e (f e)` and reaches code bodies through
  `rewrite_code`. One that needs facts about the static environment uses
  the `_where` lemmas of `FlambdaContextProofs.v`.

## 6. Workstreams

Each has one owning agent, a set of files only that agent edits, and a
definition of done. "Guide" is the OCaml to transcribe from. Paths are
relative to `middle_end/flambda2/`.

### Core (serialised)

**C0. Core owner and integrator.** Owns `FlambdaModel.v`,
`FlambdaSemantics*`, `FlambdaContext*`, `FlambdaWellFormed*`,
`rocq/Makefile`, `FlambdaExtract.v`, `rocq/HACKING.md`, the extracted
directory, and the integration branch. Applies core change requests
(section 7), fixes or fans out the fallout, merges workstream branches,
runs the full build before every merge. Does no leaf work.

The core changes, in the order they should land. Each is designed by its
workstream owner and applied by C0.

**C1. Primitives as data.** Replace `Parameter prim` by an inductive type
of the primitives modelled, with `eval_prim` defined in Rocq. Start with
what the examples use (tagging and untagging, integer arithmetic and
comparison, `Is_int`, `Get_tag`, `Opaque_identity`), then boxing and
unboxing of numbers, immutable block creation and load. Removes the
hand-written evaluator, `prim_eqb_sound`, `prim_eqb_refl` and
`eval_prim_avoids` from what is trusted. Guide:
`terms/flambda_primitive.ml`, and the `simplify_*_primitive.ml` files for
what Simplify assumes of each.

**C2. Kinds and well-formed terms.** Kinds on parameters and on
`Let`-bound variables; a checker for "names in scope, kinds agree,
arities agree" as a boolean function with a preservation theorem. Many
later statements want "for well-formed terms". Guide: `kinds/`,
`terms/`.

**C3. Sets of closures.** Several function slots, projection of function
slots, mutual recursion through the set, `my_closure` per function.
Guide: `terms/` (set of closures, function declarations),
`simplify/simplify_set_of_closures.ml`.

**C4. Heap.** A store in the state; mutable blocks, arrays and
references; allocation, load and store primitives; what is observed at
exit (values up to heap isomorphism). This is the largest core change:
equivalence must be restated up to a relation on heaps. Design it on
paper first and have it reviewed by a human.

**C5. Exceptions.** Trap actions on `Apply_cont`, exception
continuations with extra arguments, `raise`. Guide: `terms/`
(trap actions, exn continuations), `simplify/simplify_apply_cont_expr.ml`.

C1 to C3 are independent of each other in design and can be designed in
parallel; they still land one at a time. C4 and C5 should follow.

### Types (mostly independent of the core)

The types library needs the *values* of the machine and little else, so
it can run in parallel with the core changes. It lives in
`types/` here, library `OxCamlFlambda2Types`. The aim is a full,
standalone formalisation: grammar, meaning, environment, meet, join,
provers, each extracted and compared with the OCaml.

**T1. Grammar and meaning.** The type grammar (`types/grammar/
type_grammar.ml`, `type_descr.ml`): per kind, `Unknown`, `Bottom`,
aliases, variants (blocks and immediates), closures, boxed numbers,
naked numbers as sets. The denotation `type_satisfied : env -> value ->
type -> Prop`. Everything else in T is stated against this, so it lands
first and its statements get the most review.

**T2. Typing environment.** `types/env/typing_env.ml`,
`typing_env_level.ml`, `typing_env_extension.ml`, `binding_time.ml`:
the environment, its levels, adding definitions and equations, and its
invariants. `env_satisfies`.

**T3. Aliases.** `types/env/aliases.ml`: canonical elements, the order
on names by binding time, adding an alias. Statement: aliased names have
equal values in any satisfying environment; the canonical element is
the earliest. Replaces today's `FlambdaAliases*`.

**T4. Meet.** `types/meet.ml`, `meet_and_n_way_join.ml`,
`types/env/meet_env.ml`. Statement: a value that satisfies both types
satisfies the meet, under the extended environment; `Bottom` only if no
value satisfies both.

**T5. Join.** `types/join_levels.ml`, `types/env/join_env.ml`.
Statement: a value that satisfies either satisfies the join. Then the
join of the environments at the uses of a continuation as the
environment of its handler: this is the general form of what
`thread_switches` does for one case.

**T6. Provers and reification.** `types/provers.ml`, `reify.ml`,
`expand_head.ml`: each prover (`prove_is_int`,
`prove_equals_tagged_immediates`, the closure provers, ...) with the
statement that a `Proved` answer holds of every satisfying value.

**T7. Differential testing of types.** A generator of types and
environments, a converter between the compiler's types and the model's,
and a test that the extracted meet, join and provers agree with the
OCaml ones. Owns the converter for types. Starts as soon as T1 has a
grammar, with whatever operations exist.

T1 first; then T2, T3 and T7 in parallel; then T4, T5, T6 in parallel.

### Simplifier rewrites (leaf work on top of core and types)

Each is a pass with a proof, in its own file pair, wired into the
validator by V1. Each owner also writes the `.fl` examples.

**S1. Primitives through types.** Simplification of a primitive from
the types of its arguments (not only from constants, as
`fold_constants` does today). Needs C1, T6. Guide:
`simplify/simplify_unary_primitive.ml`, `simplify_binary_primitive.ml`,
`simplify_ternary_primitive.ml`, `simplify_variadic_primitive.ml`.

**S2. Switch.** Arm pruning from the type of the scrutinee, what each
arm teaches about it, switches that become boolean primitives. Needs T4,
T5. Guide: `simplify/simplify_switch_expr.ml`. Subsumes
`collapse_switches` and `thread_switches`.

**S3. Continuations.** Unused parameters, aliases of continuations
(`simplify/continuation_shortcut.ml`), inlining of a non-recursive
continuation used once wherever the use is, rewriting of uses
(`simplify/apply_cont_rewrite.ml`), join points
(`simplify/join_points.ml`). Subsumes `inline_tail_conts`.

**S4. Calls.** Indirect to direct from closure types, inlining with the
decision as an input, over- and partial application, removal of unused
code. Guide: `simplify/simplify_apply_expr.ml`, `simplify/inlining/`.
Includes the code-id relation: Simplify gives new code ids to simplified
code, and the comparison must relate them instead of the tool aliasing
them (the CR in `FlambdaCompareSpec.v`; guide:
`types/env/code_age_relation.ml`).

**S5. Lets, CSE and lifting.** `simplify/simplify_let_expr.ml`,
`common_subexpression_elimination.ml`, and lifting of constants to
static bindings (`simplify/lifting/`). Lifting is where the toplevel
condition of `FlambdaWellFormedSpec.v` earns its keep.

**S6. Unboxing of continuation parameters.** `simplify/unboxing/`,
immutable cases only. Later; needs C1, T5, S3.

**S7. The reaper.** Out of scope for now; not to be started. When it
is: run `validate_unit` on the reaper's output. Removal of unused fields
and parameters as refinements. The analysis itself is not verified; its
result is checked.

### Validation and infrastructure

**V1. Validator.** Owns `FlambdaValidate*`, `FlambdaCompare*`,
`FlambdaNormalise*` (as existing passes are superseded, deleting them),
`FlambdaUnusedStatics*`. Integrates each new pass into
`normalise_before` / `normalise_after` and keeps `validate_unit_sound`
proved. The only agent besides C0 who edits those files.

**V2. Corpus and harness.** Owns `tests/tools/flsemantics.ml` and
`tests/semantics/`. Makes the tool run on real compilations: Flambda
taken from compiling the standard library and the testsuite, not only
`.fl` files. Publishes, every day, a table of what was validated and the
reasons for the rest (unsupported construct, unsupported primitive, not
validated). That table is the priority queue for everyone else.

**V3. Converter.** Owns the OCaml that converts compiler terms to the
model. As the model grows this is the main trusted component, so it gets
its own owner: a printer from the model back to fexpr, round-trip tests,
and an independent reading of every case against `terms/`.

**N1. Nominal loose ends.** Equivariance of execution for continuations
(the CR in `FlambdaAlphaSpec.v`). Small; a good first task.

### Review

**R1, R2. Reviewers.** They write no proofs. For every `Spec` file that
is about to merge they read the statements against the OCaml named in
the header, and try to break them: a statement that is vacuous, a
premise that is never satisfiable, a definition that differs from the
compiler in a way the header does not admit, an axiom that crept in.
They own `ROCQ-REVIEW-GUIDE.md` and the trust ledger. A merge needs a
reviewer's sign-off on the `Spec` diff; `Proofs` files need none.

## 7. Coordination rules

**Branches.** The integration branch is
`mshinwell/flambda2-rocq-semantics`. Each workstream has one branch off
it, in its own worktree, named `flambda2-rocq/<stream>` (for example
`flambda2-rocq/T4-meet`). Rebase on the integration branch at least
daily. Merge small and often: a branch that lives a week will not merge.

**Commits.** Commit to your own branch without asking, whenever the work
is in a state worth keeping. Only C0 commits to the integration branch.
Nobody pushes. Never pass `--no-verify`: if a hook refuses a commit,
reword it. If something cannot be done from your own worktree, report it
and take another task; do not ask for wider permissions.

**Ownership.** A file has one owner. Only the owner edits it. If you
need a change in a file you do not own, ask its owner; do not make it.
New work goes in new files, which you then own. The table is checked in
and updated in the same commit that creates a file.

**Shared files.** `rocq/Makefile`, `FlambdaExtract.v`, the extracted
directory, `rocq/HACKING.md`, and `FlambdaValidateSpec.v` are edited only
by C0 or V1. A workstream that adds files says so in its merge request;
the integrator adds the lines.

**Core change requests.** To change `expr`, `state`, `step`, a value
type or a relation of `FlambdaSemanticsSpec.v`, write a short request:
the diff to the `Spec`, why, and which statements change meaning. C0
batches requests into *epochs*. In an epoch C0 applies one change,
repairs the core's own lemmas, and posts the list of files that no
longer build; each owner repairs their own. No leaf merges during an
epoch. Between epochs the core is frozen and leaf work merges freely.
Aim for one epoch a week, announced in advance.

**Design a core change to be conservative.** Add a constructor rather
than change one. Add a record field rather than a positional argument.
Make a new behaviour undefined-by-default so that existing refinements
still hold. If a change would falsify an existing statement, say so in
the request.

**Decisions that are a human's.** What is observable, what is undefined
behaviour, how the heap is modelled, and anything that changes what
`validate_unit_sound` says. Write the options and a recommendation; do
not pick silently.

**Definition of done for a merge.**

1. `make -C rocq all` passes (proofs, claims, extraction).
2. The semantics tests pass:
   `dune build @middle_end/flambda2/tests/semantics/runtest` in the
   project's workspace.
3. No `Admitted`; the axioms are those of the trust ledger.
4. At least one `.fl` example, or one differential test, needs the new
   work and passes.
5. The `Spec` header names the OCaml it models and its differences.
6. A reviewer has signed off the `Spec` diff.
7. Lines under 80 columns; `rocq/HACKING.md` updated by the integrator.

**Status.** Each agent keeps a short status file in its branch (what is
done, what is blocked, on whom). The integrator keeps a board built from
them. An agent blocked on another stream for more than a day says so
there and takes a task from the corpus table instead of waiting.

**Do not** rewrite another stream's proofs to make yours pass, weaken a
statement to get it through, or delete a failing example. Report it.

**Outside sources.** Whoever reads a source outside this repository for
the work (a paper, a book, a web page, another project's code or
documentation, an issue tracker, the sources of an installed OCaml or
Rocq, a search) writes its entry in `EXTERNAL-SOURCES.md` when the
reading happens, in the same series of commits as the work that uses
it, or before. A merge whose accounts of provenance name an outside
source that the file lacks is not done.

## 8. Dependencies and order

```
Phase 0 (one agent)
   |
   +-- N1, V2, V3, R1                      start at once
   +-- T1 --> T2, T3, T7 --> T4, T5, T6    types
   +-- C1 design, C2 design, C3 design     in parallel, on paper
         |
   epochs: C1 --> C2 --> C3 --> C5 --> C4  one at a time
         |
   S1 (after C1, T6)   S2 (after T4, T5)   S3 (any time)
   S4 (after C3)       S5 (after C1)       S6 (after S3, T5)
   S7 (out of scope for now)
```

S3 and N1 need nothing that does not exist and can start on day one.

## 9. Suggested allocation

**Ten agents.**

| Agent | Stream |
|-------|--------|
| 1 | C0 core owner and integrator (Phase 0 first) |
| 2 | C1 primitives, then S1 |
| 3 | C3 sets of closures, then S4 |
| 4 | T1 grammar and meaning, then T4 meet |
| 5 | T2 environment and T3 aliases, then T5 join |
| 6 | T7 differential testing, then T6 provers |
| 7 | S3 continuations (N1 as a warm-up) |
| 8 | V1 validator |
| 9 | V2 corpus and V3 converter |
| 10 | R1 reviewer |

C2, C4, C5, S2, S5, S6 wait, or are picked up as agents free up.

**Twenty agents.** As above, and split what was doubled up:

| Agent | Stream |
|-------|--------|
| 11 | C2 kinds and well-formedness |
| 12 | C5 exceptions (design, then epoch) |
| 13 | C4 heap (design with a human, then epoch) |
| 14 | T3 aliases on its own |
| 15 | T6 provers on its own |
| 16 | S2 switch |
| 17 | S5 lets, CSE, lifting |
| 18 | V3 converter on its own |
| 19 | unassigned (S7, the reaper, is out of scope for now) |
| 20 | R2 second reviewer |

Beyond about twenty, the limit is the core owner and the reviewers, not
the number of tasks. Add reviewers before adding provers.

## 10. Measures of progress

- The share of functions of the standard library whose simplification
  `validate_unit` accepts (from V2's table). This is the headline.
- The trust ledger: axioms and lines of trusted OCaml. It should shrink.
- For types: the share of `types/` with a Rocq counterpart that is
  proved and agrees with the OCaml under differential test.
- Statements proved, and how many a reviewer has signed off.
