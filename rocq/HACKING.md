# Modelling compiler passes in Rocq

Some analyses in the compiler are written as Rocq models: the analysis is
defined in Rocq, its intended meaning is stated alongside, the theorems
connecting the two are proved, and the OCaml that the compiler builds is
*extracted* from the Rocq definitions. The extracted `.ml`/`.mli` files are
committed and are the implementation; the compiler's build does not run
Rocq. This page is for anyone (a person or a coding agent) adding to or
changing such a model. It records how the models are laid out, what is
trusted, and what was learned writing the first ones.

The first models are the loop analyses behind `-ssa-delete-empty-loops`
and `-ssa-bounds-check-elim` (`backend/ssa/loop/rocq/`); read them as
worked examples. `DeleteEmptyLoops.v`, `DeleteEmptyLoopsProofs.v` and
`DeleteEmptyLoopsSpec.v` show the three files of a model.

Start review of empty-loop deletion at `LoopEligibilitySpec.v`: read
`eligible_entry` and `find_deletions_eligible_statement`, then their
imported semantics and contracts. The statement says, for every header/exit pair
the pass reports, that the execution from the header first arrives at
the exit after finitely many steps (`SsaSemanticsSpec.steps_until`) and
agrees with the first header execution on every register read before
its next definition. Its premises are two records, `finished_graph`
(`SsaProgressSpec.v`: what the SSA builder guarantees) and
`reached_header` (`SsaContextSpec.v`: bound inputs and an observed prefix
to the header). It assumes no recurrence, invariant bound, guard outcome,
progress or finite exit, and the discovered loop does not appear in the
statement. The mechanically rewritten graph, renaming and cleanup are
deliberately outside this proof's scope.

Start review of bounds-check elimination at `BoundsCheckExecutionSpec.v`:
`find_eliminations_sound_statement` says, and `BoundsCheckExecutionProofs.v`
proves, for every block/target pair the pass reports, that every run of
the observation semantics (`SsaContextSpec.context_steps`) from the entry
to the end of the block steps to the target, so the block's two-way branch
on an unsigned comparison may become a jump. The decision is
`BoundsCheckElim.v`, over the guard facts of `AffineSsa.v` (with the index
linearized modulo 2^64, `linearize_goal`) and the bounds of loop counters
certified in `InductionRange.v`, whose meaning is `InductionRangeProofs.v`
and whose operational meaning is
`InductionRangeExecutionProofs.counter_bounded`. Its premises are
`finished_graph`, bound inputs and the observed run; nothing is skipped,
so no liveness or finite-exit argument is involved.

The loop whitelist remains integer constants, additions, subtractions and
comparisons, with Goto/Switch control flow. `SsaSemanticsSpec.v` defines
exact execution; its evaluator additionally supports immediate shifts and
or for computations before the loop, and the target-specific operations
the target itself reads as linear combinations of their arguments
(`Operation.specific_linear`, trusted), which re-tag the counters of
`for` loops. `SsaContextSpec.v` overapproximates word observations in the
unchanged prefix: opaque instructions can produce
arbitrary bounded outputs, and opaque terminators can take CFG successors
with arbitrary parameter values. Memory effects are unconstrained, not
no-ops. Trap markers step without touching a register, and the exception
handler of a Call or Raise is one of its successors, so a prefix through
a `try` is represented like any other. The extracted `graph_supported`
gate rejects any graph containing a modeled operation family with other
than one result, the one instruction shape with no rule; it checks every
block, so such code even after or beside the loop prevents deletion.
Opaque loads and calls are allowed. This avoids a full memory/call/vector
semantics; the correspondence of word observations to compiler operations
remains a review boundary. The simulation argument from `eligible_entry`
to the rewritten function is written out in ROCQ-REVIEW-GUIDE.md (section
4.7), which also says how to review the development as a whole.

`SsaExecutionProofs.v` supplies definition-site and frame facts;
`SsaContextProofs.v` proves available-value equations and execution
invariants once for both exact and opaque steps. `SsaLivenessSpec.v`
defines reads before redefinition; `SsaProgressProofs.v` derives progress
from bound operands. `LoopExecutionProofs.v` uses accessibility to construct
finite traversals to the next header or exit.
`LoopInvariantExecutionProofs.v` proves bound constancy on finite
segments; `LoopCounterExecutionProofs.v` iterates up to the arithmetic
exit witness. `LoopValueSnapshotProofs.v`, `LoopGuardExecutionProofs.v`
and `LoopRangeExecutionProofs.v` connect affine entailment to the actual
prefix, including duplicate Switch edges. `LoopLiveOutProofs.v` derives
live-out agreement. `InductionRangeExecutionProofs.v` carries a counter's
certified bounds along every observed step of its loop;
`BoundsCheckExecutionProofs.v` evaluates a removed check under the snapshot
valuation.

The public theorem's non-algorithmic premises are the fields of
`finished_graph` (structural SSA/dominance metadata, edge multiplicities,
Goto/parameter layout, the definition table) and of `reached_header`
(bounded live function inputs, the observed prefix, and defined Switch
indices from the reached header onward). Out-of-range Switch indices are
undefined behavior in SSA.
The extracted gate enforces at most $2^{32}$ graph blocks with a bounded
list walk. `find_deletions_supported` proves `graph_size_supported` and
prefix instruction coverage, so graph size is not a public theorem premise.
Finite exit is constructed directly; definedness uses classical case
splits, not new semantic axioms.

`find_deletions_spec` and `find_eliminations_spec` are the
analysis-certificate theorems.
`well_formed_ssa` formally assumes that finished graph blocks are
structurally reachable, as the OCaml builder's reachability pass ensures;
it does not assume dynamic feasibility.

## The directories

Each directory is a Rocq library (`rocq/Makefile` maps them):

| Directory                | Library        | Contents                                      |
|--------------------------|----------------|-----------------------------------------------|
| `rocq/`                  | `OxCamlStdlib` | Models of OxCaml's own types (`int`, `Int64`, lists, exceptions as options) and generic arithmetic (`IntArith.v`), each a file of definitions and axioms with its lemmas in a `*Proofs.v`; proof helpers (`ZTactics.v`); and the extraction directives mapping the models onto OxCaml's functions (`OxCamlExtract.v`) |
| `backend/rocq/`          | `OxCamlBackend`| Backend-wide definitions: the semantics of `Cmm.integer_comparison` (with `IntegerComparisonProofs.v`), the model's copy of `Operation.t` |
| `backend/ssa/rocq/`      | `OxCamlSsa`    | The SSA graph as the models see it, each model split into Spec, implementation and Proofs files (see "The layout of a model"): `SsaModel.v` (the abstract interface), `SsaGraph*` (terminators, edges and uses), `SsaDominance*` (dominance, paths and program points), `SsaSemantics*` (execution of the restricted integer fragment), `SsaExecution*` (well-formedness and available-value equations), `SsaLiveness*` (reads before redefinition), `SsaProgress*` (the `finished_graph` contract, definedness and progress), `SsaContext*` (opaque surrounding computations and the `reached_header` contract) |
| `backend/ssa/loop/rocq/` | `OxCamlLoop`   | The loop analyses, each an implementation file and a `*Proofs.v`; the operational proofs in `LoopExecutionProofs.v`, `LoopInvariantExecutionProofs.v`, `LoopCounterExecutionProofs.v`, `LoopValueSnapshotProofs.v`, `LoopGuardExecutionProofs.v`, `LoopRangeExecutionProofs.v`, `LoopLiveOutProofs.v` and `InductionRangeExecutionProofs.v`; the statements to review in `LoopEligibilitySpec.v`, `DeleteEmptyLoopsSpec.v` and `BoundsCheckExecutionSpec.v`, proved in `LoopEligibilityProofs.v`, `DeleteEmptyLoopsProofs.v` and `BoundsCheckExecutionProofs.v`; and `Extract.v`, which extracts the implementation |
| `middle_end/flambda2/rocq/` | `OxCamlFlambda2` | The beginning of a semantics of Flambda 2, extracted for its tests only: `FlambdaMap*` (finite maps and sets keyed by names, the one representation of environments), `FlambdaKindModel.v` (the kinds), `FlambdaPrimitiveModel.v` (the primitives of the fragment covered so far), `FlambdaModel.v` (the terms of that fragment, which carry kinds and arities), `FlambdaValue*` (the values of the machine, and how an allocated value is read), `FlambdaPrimitive*` (what each primitive computes, and the laws that all of them obey), `FlambdaSemantics*` (execution, behaviours, equivalence and refinement), `FlambdaWellFormed*` (the well-formedness of a unit, given with its two continuations: static bindings at toplevel only, each static name bound once, variables and continuations in scope, the names of each binder distinct, code closed, whole sets of closures, numbers in range, exception handlers kept apart from the other continuations and trap actions in order; and that what a static name is bound to never changes in a run), `FlambdaTrap*` (a second machine, with a stack of exception handlers, and that it agrees with the first on a unit that follows the discipline of trap actions), `FlambdaHeap*` (the prototype of the heap: a machine with a heap beside the one that exists, and the rewrites that Simplify needs, proved for it; `FlambdaHeapEquivariance*` is the action of renamings on it), `FlambdaBinders*` (the free variables and the free continuations of a term, and the three conditions on names), `FlambdaScope*` (that the run of a unit that meets them reads no variable that has no value and leaves by one of its two continuations), `FlambdaWellKinded*` (a check of the kinds that a term carries, and what the machine makes undefined on account of kinds), `FlambdaClosures*` (sets of closures of several functions: the derived forms that make one, the projections, and what a well-formed unit keeps true of its sets), `FlambdaPartialApplication*` (what a partial application and an over-application come to, as steps of the machine), `FlambdaTupled*` (what a call that gives tupled code a tuple comes to), `FlambdaRange*` (that the run of a unit whose numbers are in range holds no number out of range), `FlambdaNoArrayValue*` (what the immutable array as a value leaves as it was: a state that holds none and may run nothing that makes one stays so, and the relations that came to relate arrays relate, between memories that hold none, what they related), `FlambdaControlFlow*` (rewrites of control flow stated and proved against the semantics), `FlambdaContext*` (contexts, and that refinement and equivalence are preserved under them; `FlambdaContextWhere*` is the same for a hole that refines only from the states that have a property which a step keeps), `FlambdaNominal*` (renamings, free names, and binders as name abstractions), `FlambdaAlpha*` (the group action, equivariance, and alpha-equivalence with its soundness; `middle_end/flambda2/rocq/NAMES.md` lists the other files on names), `FlambdaApply*` (function calls and inlining), `FlambdaStatic*` (first facts about `Let_static`), `FlambdaCompare*` (a decision that two expressions are the same up to bound names, proved sound), `FlambdaNormalise*` (normalising a term by proved rewrites: inlining direct calls of known code, removing unused `Let`s, folding constants), `FlambdaBlockLoad*`, `FlambdaUnusedConts*` and `FlambdaUnbox*` (three more normalisations: the load of a field of a block that is known, removing a continuation that is not used, unboxing a number known to be boxed), `FlambdaUnusedStatics*` (removing unused static bindings, under environments in which their names do not occur), `FlambdaLift*` and `FlambdaMyClosure*` (making a `Let` of a closure, a block or a boxed number a static binding of a symbol, as Simplify does), `FlambdaUnusedSlots*` (removing the value slots of static sets of closures that nothing projects), `FlambdaKnownSymbols*` (folding with what the unit binds its symbols to), `FlambdaKnownSlots*` (reading, where it is projected, a value slot that every set of closures of the unit gives the same symbol or constant), `FlambdaSharedReads*` (making the read of a place of a static constant the variable of an earlier read, where one group of static bindings gave the two places one variable), `FlambdaStubCalls*` (making indirect the direct calls in the code of a stub, as Simplify does where it simplifies that code), `FlambdaCopyCode*` (giving static sets of closures a copy of their code under new code ids, where Simplify has given such a set new code and kept the old for another closure), `FlambdaClosureCalls*` (making direct a call of a closure that a `Let` in sight makes), `FlambdaSetsOfCode*` (what every set of closures of a unit that has a given code at a function slot has, and that a run keeps it) and `FlambdaClosureOfCode*` (using, in the body of code, what `my_closure` is known to be from its code: a call of it or of another function of its set made direct, a value slot that every such set gives one symbol or constant read), `FlambdaIdentities*`, `FlambdaCse*`, `FlambdaStaticsOrder*` and `FlambdaRenameApart*` (passes that are proved and not yet among the normalisations of the validation: the identities of integer arithmetic that Simplify applies, a primitive already computed not computed again, the bindings of a static group in the order of another term, every binder given a name of its own), `FlambdaValidate*` (the whole check of a unit before and after Simplify, and that what it accepts is correct for the unit run on its own), `FlambdaValidatorEquivariance*` (that the rewrites and the checks of the validation commute with renaming); see "The semantics of Flambda 2" below |
| `middle_end/flambda2/types/rocq/` | `OxCamlFlambda2Types` | A model of the compiler's types library; the tests of types extract the grammar and the functions that decide what a type means. `FlambdaTypeGrammar*` (the types, after `type_grammar.ml`), `FlambdaTypeMeaning*` and `FlambdaTypeMeaningBool*` (what it is for a value to have a type, and a function that decides it), `FlambdaTypeWellFormed*`, `FlambdaTypeEqual*`, `FlambdaTypeAccess*`, `FlambdaTypeCreators*`, `FlambdaTypeProject*`, `FlambdaTypeRename*` and `FlambdaTypeNominal*` (the well-formedness of types, their equality, the functions that read, make, project and rename them), `FlambdaTypingEnv*` and `FlambdaEnvAliases*` (typing environments with their levels and extensions, and the aliases; what it is for the values of the names to satisfy one), `FlambdaEnvRename*` (renaming a typing environment), `FlambdaExpandHead*` and `FlambdaMeet*` (the meet, transcribed, and its contract), `FlambdaJoin*` (the n-way join of environments and of types), `FlambdaProvers*` and `FlambdaReify*` (the provers and reification), `FlambdaSimplify*` (the first passes of Simplify that go by types: the rewrites of primitives, the function that assembles them, the arms of a `Switch` that a type excludes), and the first, naive model that came before the grammar, `FlambdaTypes*` and `FlambdaAliases*`. `middle_end/flambda2/rocq/OWNERS.md` has every file with its owner, and the status files say what is stated, what is proved and under which hypotheses |

Put a definition or lemma where its OxCaml counterpart would live. A fact
about lists goes in `OxCamlListProofs.v`, about machine integers in
`OxCamlIntProofs.v`, `OxCamlInt64Proofs.v` or `IntArithProofs.v` (a
definition, in the file without the suffix), about terminators or values
of any SSA graph in the `SsaGraph` files, about dominance in the
`SsaDominance` files; only what is specific to a pass stays in its files.
The same question one asks of an OxCaml module ("is this really about
loops?") applies, and then the question of "The layout of a model" below
("does a statement to be reviewed need it?") picks the file.

## The workflow

With Rocq and the repository's `ocamlformat` on `PATH`:

1. Edit the `.v` files.
2. `make -C rocq all` checks every proof and regenerates the extracted
   files (about ten seconds from scratch). It fails if a proof breaks, and
   it refuses an extraction that produces a module it does not expect
   (see "Adding a module" below).
3. Review `git diff` of the extracted `.ml`/`.mli`: it should show exactly
   the change you meant. Never edit those files by hand; change the `.v`
   and regenerate.
4. `make -s boot-compiler`, then the tests of the pass (for the loop
   passes, the `expect.opt` tests in `testsuite/tests/backend/ssa/loop/`,
   whose `[%%expect_ssa]` blocks show the SSA each function ends up with
   and how many loops were deleted or checks removed from it, and the
   unit test alias
   `dune build @oxcaml/tests/backend/ssa/runtest`), then `make -s fmt`.
5. Commit the `.v` files and the regenerated `.ml`/`.mli` together. (Not
   so for the Flambda 2 semantics on a workstream branch: see "Who commits
   the extracted files" under "Testing Simplify against the semantics".)

After renaming or moving `.v` files, stale `.vo` files can shadow the new
ones: `touch` the sources (or `make -C rocq clean`) before rebuilding.

### Printing the review obligations

Run `make -s -C rocq assumptions > assumptions.txt` to build and audit every
module listed in this Makefile. The report has two parts:

1. `rocq check -o` rechecks the compiled modules and lists global axioms and
   abstract constants, including those in imported libraries. It also
   reports unsafe typing assumptions. This is a whole-context inventory,
   not the minimal dependencies of one theorem.
2. `Print Namespace` prints constant types for each project module,
   including nested modules, theorem premises generalized from `Section`s,
   and the projections of contract records. Unlike `Search`, it does not
   hide names through a search blacklist. This part includes proved
   theorems too; it is not another list of axioms.

Between the two parts the target checks the axioms against the trust
ledger, `rocq/TRUST-LEDGER.md`, and fails if they differ. Every axiom of
the first part must be listed there, with its reason; every axiom listed
there must be in the first part; and the statements of the Flambda 2
libraries must depend only on the axioms of the ledger's section
"Flambda 2" (so that, for one, none comes to use classical logic). An
axiom or a `Parameter` added to a model is therefore added to the ledger
in the same change. For the Flambda 2 work that needs a reviewer's
sign-off, and the aim is for the ledger to shrink.

For a particular theorem, `Check @find_deletions_eligible.` (or
`Check @find_eliminations_sound.`) shows its full premises and
`Print Assumptions find_deletions_eligible.` gives its global axiom
dependencies. Named predicates still need inspection in the source:
declaration types do not print definition bodies, and an assumption audit
cannot check that the semantics or extraction match the compiler.

The target derives its module list from the build lists, so newly added
modules are included automatically. It prints to standard output and does
not leave generated audit sources in the workspace.

### The review document

`make -C rocq review` typesets what a reviewer reads as one PDF,
`rocq/review.pdf`, in parts: the shared models (the sections of
`ROCQ-REVIEW-GUIDE.md` about every pass, then every reviewed `.v` file of
`rocq/`, `backend/rocq/` and `backend/ssa/rocq/`: the Spec, Model and
definition files; implementation and Proofs files are left out), then one
part for what the loop passes share and one per pass, each a top-level
section of the guide promoted to a part, with the pass's Spec files at
the end, then the extraction directives and the axioms each statement
rests on. Which guide sections make a part, or open the first one, and
which Spec files belong to a pass is set at the top of
`rocq/review-pdf.pl`; a new Spec file or top-level guide section must be
assigned to a part there, or the script stops. The files of a library are
shown in the order of its list in `rocq/Makefile`, which is a dependency
order; the Flambda 2 lists are in alphabetical order instead (see "Adding
a module"), so for them the script works out a dependency order itself,
preferring the reading order written at the top of the script (`order`),
and a new file needs no entry there. The sources are typeset by
`rocq doc` with every identifier linked to its definition, so the target
needs the `.glob` files of a complete build, and LuaLaTeX with `fontspec`
(Libertinus and DejaVu Sans Mono are used when installed).
The layout of a file carries over: its header comment opens its section, a
banner comment (`(* --- Title --- *)`, or a title between two `(* === *)`
lines) becomes a heading, and `[ident]` in a comment is typeset as code
and linked, as are a file name and a `Module.ident` written without
brackets. The script is `rocq/review-pdf.pl`.

### Adding a module

A new `.v` file goes in the library's list in `rocq/Makefile` (`STDLIB`,
`BACKEND`, `SSA` or `LOOP`); the Spec, implementation and Proofs files of
a model are three such entries, and `make claims` finds the Spec files in
the `SSA` and `LOOP` lists by their suffix. If a file is extracted (it
defines functions the compiler runs, as opposed to only lemmas), also add
its `.ml`/`.mli` to `SSA_MODULES` or `LOOP_EXTRACTED` there, add the
module to the list in the root `dune` file next to the other extracted
modules, and mention it in the header of `Extract.v`. Extraction writes
every file into `backend/ssa/loop/rocq/`; the Makefile moves the
`OxCamlSsa` ones next to their sources. A module whose every definition is
mapped onto OxCaml's own functions extracts to an empty file, which the
Makefile deletes.

The Flambda 2 libraries are worked on by several people and agents at
once, and their lists are laid out so that two branches that each add
something merge without a conflict: `FLAMBDA2` and `FLAMBDA2_TYPES` in
`rocq/Makefile`, and the `Separate Extraction` list of `FlambdaExtract.v`,
have one name to a line, in alphabetical order. Put a new name in its
place in that order. The order means nothing: make takes the dependencies
between the files from `rocq dep`, the review document orders the files
itself, and what is extracted does not depend on the order of the names.
`make claims` finds the Spec files of these lists by their suffix too.

These lists are the integrator's (`middle_end/flambda2/rocq/OWNERS.md`),
but a workstream cannot build a new file without its line, nor can whoever
reviews the branch. So a workstream asks for its lines with a commit:

- The lines go in one commit of their own, the last on the branch, with
  the subject `Flambda 2 Rocq: <stream>: lines for the shared lists`. It
  touches nothing but these lists, and adds only lines for that stream's
  own files, each in its place in the order. No other commit of the
  branch touches the lists. (The one commit that may follow it is that
  of the test tool and the examples: see "Who commits the extracted
  files" under "Testing Simplify against the semantics".)
- To add a line later, amend that commit if it is still the last. If
  other commits have followed it, move it back to the end before asking
  for a merge: `git rebase --onto <it>^ <it>` takes it out of the branch
  and `git cherry-pick <it>` puts it on top. Neither runs the commit
  hooks, so reread the messages.
- The integrator checks the commit at the merge, and refuses a branch
  whose lists differ from the integration branch's in any other way.

## The layout of a model

A model is split over up to three files, so that a reviewer knows from a
file's name whether to read it:

- **`XSpec.v`: what is claimed.** Everything a reviewer has to read about
  the model and nothing else: the definitions its statements are made with
  (a semantics such as `SsaSemanticsSpec.step`, a contract such as
  `finished_graph`, the meaning of a decision such as `eligible_entry`),
  and the statements of its theorems, each bound as a proposition
  (`Definition step_deterministic_statement : Prop := ...`). A Spec file
  imports Model, Spec and implementation files only, so nothing in a Proofs
  file can change what it says; it mentions an implementation function by
  name (`find_deletions` in a statement), never its body. The header says
  what is trusted and what to check the definitions against.
- **`X.v`: the implementation.** The analysis itself, written to mirror the
  shape of the OxCaml it replaces: same decomposition into functions, same
  names where possible, booleans for checks. This is what is extracted
  (`X.ml`). It needs no review: the Spec says what its answers mean and the
  Proofs show that they do.
- **`XProofs.v`: the proofs.** Each theorem proves a statement of the Spec
  under the statement's name without its suffix
  (`Theorem step_deterministic : step_deterministic_statement`), with the
  stepping-stone lemmas and the intermediate specifications they go through
  (the meaning of a helper check such as `uses_only`, an invariant of
  execution such as `valid_position`). Nothing here needs review, and
  `make -C rocq claims` (run by `make -C rocq all`) checks that every
  statement a Spec binds is proved. It finds a statement by the line
  `Definition X_statement : Prop`, which is to be written on one line
  however long the name: the target fails on a Spec file in which a
  definition of such a name is written otherwise, since it would not
  be checked.

A model with nothing to extract has no `X.v`; one whose meaning appears in
no reviewed statement has no `XSpec.v`. A Spec whose definitions the
implementation calls is extracted too (`SsaGraphSpec.ml`: the two readings
of a terminator shared by the semantics and the checks). `*Model.v` and
`*Extract.v` stand apart: they are the trusted base (below), reviewed by
inspection.

A definition goes in a Spec only if a statement that has to be read uses
it, and the test is mechanical: start from the theorems ROCQ-REVIEW-GUIDE.md
names and follow their definitions; what is reached is Spec, and what is
not, however meaningful, is a Proofs definition (`available_value` is one:
the loop proofs use it and no reviewed statement does). A lemma statement
goes in a Spec only when the reader needs it to interpret a theorem
(`step_deterministic_statement` is what makes `steps_until` speak of *the*
execution from a state). Each conjunct of a specification is a named
predicate (`exits_only_to`, `params_are_dead_bivs`, ...), so that it reads
as a sentence and needs no comment explaining its parts. Local
compiler-data contracts are `Record`s of `Prop`s or `Hypothesis` in a
`Section` (`control_edges`, `parameter_layout`). Execution belongs in an
independent semantics such as `SsaSemanticsSpec.step`; the facts an
analysis needs from execution must be proved, not packaged as a trusted run
record.

A statement bound as a proposition is used by `apply`, `eapply`, `rewrite`,
`destruct` and `exact` as if it were the theorem's type; only
`rewrite ... at` and `setoid_rewrite` need it unfolded first.
Analysis-certificate theorems go one way: a boolean check answering `true`
implies its meaning. Nothing is claimed about `false` answers, so the
analyses may be conservative wherever that is convenient (size limits,
overflow, unrecognised shapes).

The models of OxCaml's own types and the backend definitions (`rocq/`,
`backend/rocq/`) are the trusted base rather than models with a Spec:
`OxCamlInt.v`, `OxCamlInt64.v`, `OxCamlList.v`, `IntArith.v`,
`IntegerComparison.v` and `Operation.v` hold definitions and axioms only,
read by inspection against their `*Extract.v` like `SsaModel.v`, and their
lemmas are in `*Proofs.v` files. A proof-only module, with nothing
extracted and no reviewed statement, is a single `XProofs.v`
(`LoopExecutionProofs.v`). In the loop library only two statements are
reviewed: `LoopEligibilitySpec.find_deletions_eligible_statement`, the
result, and the two facts of `DeleteEmptyLoopsSpec.v` that the arguments
outside Rocq rely on; the certificates of the analyses
(`InductionVarProofs.biv_spec`, `DeleteEmptyLoopsProofs.deletion_certificate`)
are reached by no reviewed statement and are Proofs definitions.

## Modelling OxCaml

- **`int`** is Rocq's primitive signed 63-bit integer (`Sint63`), which is
  OxCaml's `int` exactly; `to_Z` reads it as a `Z`. Literals are named
  (`zero`, `one`, `minus_one`, `two` in `OxCamlInt.v`) so that extraction
  prints OxCaml literals; a new literal needs an
  `Extract Inlined Constant ... => "..."` line in the `*Extract.v` file,
  and the two must be kept in step (see `FourierMotzkinExtract.v`).
  Arithmetic wraps: `Sint63.add_spec` and friends give the result modulo
  `2^63` as `cmod`; `OxCamlIntProofs.v` has the lemmas that let `lia` work
  on such sums.
- **`Int64`** is a `Parameter` type with axioms (`OxCamlInt64.v`), each the
  documented behaviour of one OxCaml operation. Do not put `Definition`s
  inside `Module Int64` there: they would be extracted into a module
  shadowing OxCaml's `Int64`.
- **Exceptions** become `option`: a function that may raise returns `None`
  instead, and `let* x := m in k` (`OxCamlOption.v`) propagates it.
  `map_opt` and `filter_map_opt` are `List.map` and `List.filter_map`
  with such callbacks.
- **Arrays** are lists, indexed with `nth_int` (`None` where OxCaml would
  raise), compared in length with `same_length`. `OxCamlList.fold_left`
  has OxCaml's argument order.
- **The compiler's data**: blocks are a `Parameter` type in
  `SsaModel.v`, with the few observations the models make of them
  (`terminator`, `params`, `predecessors`, ...) also as `Parameter`s,
  each realised at extraction by a function of `Ssa`, at most wrapped
  with `Array.to_list`. Values, instructions and terminators are
  `Inductive` types there, copying the compiler's with the fields the
  models read, and `Extract Inductive` maps each back onto the
  compiler's type with a match function (its last argument) that reads
  the compiler's data; so the models match on the compiler's values
  directly, and the operation defining a value is `Operation.v`'s copy of
  `Operation.t` (`backend/rocq/`), kept in step with
  `backend/operation.mli`. To observe something new about a block: add
  the `Parameter` and its `Extract` directive in `SsaModelExtract.v`.
  `SsaModel.v` proves nothing; what the proofs rely on about an
  observation is a hypothesis of the theorem using it.
- **Extracted inductives standing for compiler types** must list their
  constructors in the same order as the OxCaml type
  (`Extract Inductive ... [ ... ]`), and a comment must say which type to
  compare with.
- **Recursion** that Rocq cannot see terminates is written either with
  `Function ... {measure ...}` and a decreasing measure (`gcd64`), or with
  an explicit "fuel" list and a completeness theorem showing that the
  fuel the caller passes is enough (`NaturalLoop.walk_predecessors`).
- **Mutable state** is threaded explicitly: a context is passed in and
  returned, so that each function is a pure function of its arguments
  (`AffineSsa.ctx`).

### What is trusted

The trusted base is small and must stay listed in the headers, and in
`rocq/TRUST-LEDGER.md`, which has every axiom and, for the Flambda 2
semantics, every piece of trusted OCaml, each with its reason
(`make -C rocq assumptions` checks the axioms against it):

- the axioms about `int`, `Int64` and the list functions mapped onto
  OxCaml's (`OxCamlInt64.v`, `OxCamlList.v`);
- the `Parameter`s with axioms standing for a few lines of OxCaml that
  are easier to review than to model (`IntArith.v`'s overflow tests);
- the target's reading of its specific operations as linear combinations
  of their arguments (`Operation.specific_linear`, realised by
  `Arch.linear_specific_operation`), the one thing the models take on
  trust about a target's semantics;
- the extraction directives, which say which OxCaml value realises each
  model definition (`*Extract.v`), including the match functions written
  in them;
- the hypotheses the theorems take about the graph.

Everything else is proved. Never `Admitted`. When something cannot be
proved, state it as a hypothesis where it is used and say so in the
header.

## The semantics of Flambda 2

`middle_end/flambda2/rocq/` is separate from the SSA models: no pass
depends on it, and what it extracts is for its tests only. Its aim is a
semantics of Flambda 2 against which the simplifier's transformations can
be proved correct, built a fragment at a time. The files follow the same Spec/Proofs split, and
`make claims` checks its statements too.

- The work is planned for many people and agents at once:
  `middle_end/flambda2/rocq/PARALLEL-PLAN.md` is the plan and its rules,
  and `OWNERS.md` beside it says who owns which file. Edit only files you
  own.
- Each stream keeps a status file under
  `middle_end/flambda2/rocq/status/`, and a change to the core has a
  note under `requests/` beside it. Such a document, like every
  committed document and comment, cites only commits that are on the
  integration branch and only files that are in the tree, and has no
  local path, host name or session id in it. The commits of a branch
  that is not merged yet are not to be cited: they get other hashes
  when the branch is rebased or applied. What is elsewhere is said in
  words.
- `FlambdaModel.v` is the model's copy of the terms (`flambda.mli`), for
  the fragment covered; its header lists what is left out and how the
  presentation differs from the compiler's (plain names in place of name
  abstractions, in particular). Extend the fragment there first.
- `FlambdaSemanticsSpec.v` defines execution as a step *function* on states
  of an abstract machine (an expression, an environment of values and one
  of continuation closures), so determinism holds by construction and
  a term can be run (by `vm_compute`: `step` is opaque to `simpl` once
  `FlambdaSemanticsProofs.v` is loaded, see below). A state's behaviour
  is to exit through an unbound continuation with values, to go wrong, or
  to diverge. A transformation is correct when its result `refines` its
  source (anything may be done to a term that goes wrong); `equivalent`
  is the symmetric notion. `refines` is stated of the behaviours that
  end, so that no proof needs the excluded middle: to prove an
  equivalence or a refinement, divergence never has to be considered
  (`equivalent_of_terminating`).
- A function call has no stack (`call_function`): the body of the code
  runs with the code's return and exception continuations standing for
  the caller's, so returning is applying a continuation. A continuation
  name stands for a handler closure or, unbound, for a way out of the
  term (`resolve`).
- An exception handler is a continuation, and a raise is an
  `Apply_cont` of it with the exception as first argument. A handler
  says whether it is an exception handler, an `Apply_cont` carries its
  trap action, and an `Apply` the extra arguments of its exception
  continuation. The machine has no stack of handlers and does not
  execute a trap action: where an exception goes is said by the term,
  as for any continuation. A call gives its callee, as the
  continuation to raise to, the caller's handler with the values of
  the extra arguments kept (`Extra_args`, `with_extra_args`). The
  compiled program goes by a stack of handlers instead.
  `traps_well_formed` is the discipline that the front end follows, as
  a check of the term; nothing in Flambda 2 checks it (discrepancy
  C5-1). `FlambdaTrapSpec.v` has a second machine of the model, which
  goes by such a stack, and the statement that the two machines have
  the same behaviours on a unit whose binders are distinct and that
  passes the check (`stack_agrees_statement`); without either
  condition they can differ (`stack_differs_statement`,
  `stack_needs_distinct_binders_statement`). That the compiled program
  behaves as the second machine is not stated or proved. It also has
  the statement that, in a unit that passes
  `unit_exn_handlers_well_formed`, the `Apply_cont`s that `to_cmm`
  takes for raises are those that the term says are raises
  (`raises_as_marked_statement`), and that the second machine commutes
  with renaming. Both
  conditions on exceptions (`unit_exn_handlers_well_formed`,
  `traps_well_formed`) are conjuncts of `well_formed_unit`, and both
  are stricter than the compiler; that they are asked at all was
  decided while the owner of the branch was away. Inlining a call is
  stated for a call with no extra argument, and a continuation applied
  with a trap action is not inlined.
- The terms carry the compiler's kinds. A `Let` gives the kind of its
  variable, a handler the kinds of its parameters, code the arity of
  its parameters and that of its results, and an `Apply` the two
  arities it is made at; an arity is `Flambda_arity.t` without
  subkinds, unboxed products included. Execution tests the kinds
  that a term carries where a variable is bound: at a call with as
  many arguments as the code of its callee has parameters (the
  arguments, against the call's arity, and the two arities of the
  call against those of the code), at a `Let` (the value of the
  defining expression, against the kind of the `Let`) and where a
  handler is applied (the values, against the kinds of its
  parameters): `FlambdaSemanticsSpec.call_code`, `step`,
  `apply_cont_value`. A step that fails a test is undefined. So the
  step that binds a variable binds a value of the kind of its binder,
  whatever the term (`FlambdaWellKindedSpec.let_binds_at_kind`,
  `handler_binds_at_kinds`, `call_binds_at_kinds`, and
  `called_closure_at_kind` for `my_closure`), and a proof gets
  the fact from `let_refines_step`, `let_equivalent_step` and
  `handler_refines_at`, `handler_refines_where`,
  `handler_refines_from`, which hand it to the hypothesis about the
  body or the handler. The tests at a `Let` and at a handler are
  decision D10, with a review comment before `step`. For a rewrite:
  one that keeps a `Let` at its kind with the same value needs
  nothing; one that removes a `Let` or a handler with parameters, or
  moves a value past one, is a refinement, and an equivalence only
  where the kind is known on the term; a `Let` that a rewrite makes
  must give the kind of the value it binds. The machine with a heap
  (`FlambdaHeapMachineSpec.heap_step`) has the same two tests. A call that
  fails its test is one that Simplify replaces by `Invalid`, and
  well-typed programs give rise to them in branches that never run,
  so it has to be undefined; so has a primitive on an argument of
  another kind than it takes (`eval_prim_kinds`).
- A call with fewer or more arguments than the code of its callee has
  parameters is a partial application or an over-application, whatever
  the kind of the call (`shape_of_closure_call`, which has the tests of
  arities that the compiler makes of one, and a CR on what the backend
  makes of a direct call that is not full). A partial application is a
  value (`Value_partial_application`: the closure, the arguments given
  and their arity), which the call gives to its return continuation in
  the same step; an indirect call of it calls the closure with those
  arguments and the new ones. An over-application runs the code on the
  first arguments with, for return continuation, what is left of the
  application (`Over_application`), which calls the value returned with
  the arguments that remain; both calls raise to the exception
  continuation of the call, with its extra arguments.
  `FlambdaPartialApplicationSpec.v` has what the two come to, as steps,
  and examples;
  `middle_end/flambda2/rocq/requests/C3-partial-application.md` has the
  choices made.
- Code says whether it is tupled (`cm_is_tupled`, `code_is_tupled`).
  Every call gives tupled code one argument, a tuple, followed by any
  others. The machine reads the fields of the tuple as Simplify's
  projections do (`tuple_fields`: an immutable block of tag 0 with one
  field of kind `Value` for each parameter of the code) and goes on as
  for a call of the fields followed by the other arguments
  (`arguments_for_code`, then `shape_of_code_call`). So a proof about a
  call needs one more fact, about the fields of a tuple: the traversal
  of values takes it as `P_fields`, and a relation between two runs
  has it from `arguments_for_code_related` or
  `arguments_for_code_mono` (`FlambdaSemanticsProofs.v`). A call of
  tupled code never makes a partial application. The compiler's direct
  call of tupled code, which gives the code the fields, is not in the
  model yet; `inlinable` and `FlambdaLoopifySpec.loopable` ask that
  the code is not tupled. `FlambdaTupledSpec.v` has what such a call
  comes to, and examples;
  `middle_end/flambda2/rocq/requests/C3-tupled-functions.md` has the
  choices made.
- `FlambdaWellKindedSpec.well_kinded_unit` checks that the kinds a unit
  carries agree with one another. A unit that passes it starts in a
  well-kinded state, a step keeps a state well-kinded, and a
  well-kinded state has none of the faults of `kind_fault` (it reads
  no unbound variable, applies no handler to the wrong number of
  arguments, its `Switch` is on a naked immediate, it binds no value
  of another kind than its `Let` gives and applies no handler to
  values of other kinds than its parameters': `wrong_kind_bound`,
  `wrong_kinds_of_arguments`); a call that
  does not agree with its code and a primitive on an argument of
  another kind are not among those, and stay undefined. The check is a
  sufficient condition and a diagnostic, which the test tool prints.
  The compiler accepts terms that fail it, as it checks kinds only
  where Simplify visits, and that the units it builds pass it is known
  by experiment only; so it is not a condition of `well_formed_unit`
  and no statement about the compiler's terms may take it as a
  premise. The header of the file says more.
- Mutable memory is in the model: a form of value, `Value_location`,
  which none of the five observations of `FlambdaValueSpec.v` reads (a
  location is read with a heap, by `cell_of`), and four primitives of
  the heap under one constructor of `prim` (`Heap`: the allocation of
  a block that has an identity, `Block_set`, the allocation of a
  mutable array of values, and `Array_set`), on which `eval_prim` has
  no value. The load from such a block and the comparison of two are
  `Block_load`, with its mark of mutability, and `Phys_equal`. The
  length and the load of an array are `Array_length` and `Array_load`.
  A cell of the heap is a block or an array (`FlambdaValueSpec.cell`).
  The state carries a heap, and `step` runs a `Let` on it:
  `FlambdaSemanticsSpec.exec_named` gives the value of the defining
  expression and the heap after, by `FlambdaHeapSpec.exec_prim` where
  it is a primitive. `exec_prim` says what each primitive does on a
  heap, where it has a value: a primitive of the heap makes a cell or
  writes to one; a unary primitive of a location, and a binary
  primitive other than `Phys_equal` whose first argument is a
  location, read its cell; `Phys_equal` with a location among its two
  arguments is a comparison of its own (`eval_phys_equal_location`),
  which reads no cell; and every other primitive has the value that
  `eval_prim` gives it. It has no value where an argument is a symbol
  or a location that is bound to nothing. The state also carries a
  read record (`FlambdaValueSpec.read_record`), which nothing adds to
  and nothing reads yet (decisions D49 and D52). The conversion of the
  test tool makes the four primitives, so a run of the tool can have
  cells; the trust ledger says which forms it converts and which it
  refuses ("The conversion of terms"). `FlambdaHeapMachineSpec.v` is
  the prototype that the design
  (`middle_end/flambda2/rocq/requests/C4-heap.md`) rested on: a
  machine of its own beside `step`, with a heap carried along and an
  exit that carries the memory, which ran these primitives before
  `step` did. A state of the prototype whose next step is not a `Let`
  of a primitive steps as the state of `step` that it projects to
  (`heap_step_agrees_statement`). What that file still has and
  `FlambdaSemanticsSpec.v` has not is the relation on what is observed
  at exit up to a correspondence between the locations of two runs,
  stated of that machine and not of `step` (decision D40), and the
  rewrites that Simplify needs, proved for that machine in that
  relation. A proof by cases on values or on primitives has the case
  of a location and that of `Heap`, and so has OCaml code that matches
  on the extracted `value` or `prim` without a wildcard.
- A unit is well-formed with respect to its interface
  (`FlambdaWellFormedSpec.unit_interface`: the continuation it returns
  to and its exception continuation). `well_formed_unit` is to be the
  one definition of a well-formed unit:
  `middle_end/flambda2/rocq/requests/C0-well-formed.md` has the table
  of its conditions and of those to come. A condition on terms that
  the compiler's units meet goes there, and not into a premise of the
  statements that need it.
- "Let symbol" is `Let_static`: a group of bindings of code ids to code
  and of symbols to static constants. It extends the static environment,
  a fourth component of the machine state that is global (closures do not
  keep it) and only grows. A symbol is a value by itself, a pointer
  (`Value_symbol`), and what it points to is looked up where it is used
  (`closure_of`), so groups may be recursive and static data cyclic.
  Renaming does not touch symbols, code ids or code.
- A set of closures of several functions has no constructor of `expr`.
  `Closure_of_set slot function_decls value_slots` makes the closure at
  one function slot of a new set, `Project_function_slot` goes from a
  closure to another of its set, and `Project_value_slot_from` reads a
  value slot. A `Let` of several closures is the derived form
  `FlambdaModel.let_closures` (the first closure is made, the others are
  projected from it), and a static set is `bind_closures` (one
  `Bind_symbol` for each symbol, each repeating the set). Read a value
  as a closure through `closure_of` (its code and value slots) or
  `closure_in_set` (its slot and its set). The older forms
  (`Set_of_closures`, `Project_value_slot`,
  `Static_const_set_of_closures`) are a set of one function whose slot
  has no name; they are to be deleted, so write new statements with the
  named forms. The terms can write sets that the compiler cannot make.
  `FlambdaWellFormedSpec.sets_well_formed` checks the function slots of
  a set, that its functions have each their own code, and that a static
  set is bound whole, with no code in common between the sets of one
  group; it does not yet ask that the value slots are distinct.
  `FlambdaClosuresSpec.v` has what the run of a well-formed unit keeps
  true of its sets.
- The body of code bound by a `Let_static` may be rewritten: a context may
  have its hole in code (`Let_static_code`), and `plug_refines` covers it
  (`FlambdaStaticSpec.v`: `let_static_code_refines`). The proof relates
  the static environments of the two runs (`statics_related`: the same
  values, related code); values hold code ids and not code, so they stay
  equal. The renaming bisimulation still keeps the static environments
  equal, as renaming does not reach into code.
- Inlining a direct call is a refinement (`FlambdaApplySpec.v`:
  `inline`), given that the code is closed and that binding the
  parameters captures nothing. `inline_identity`
  (`middle_end/flambda2/types/rocq/`) is a worked example of chaining
  inlining, rewriting under a context, following an alias and removing
  unused bindings.
- Environments, and whatever else maps names to things or collects
  names, are the finite maps and sets of `FlambdaMapSpec.v`: association
  lists in which the first binding of a name counts (`lookup`), and lists
  of names (`among`). Its header says why lists, and what was rejected.
  To bind is to cons. For the rest use its operations (`is_bound`,
  `unbind`, `map_values`, `domain`, `compact`, `filter_bindings`, `merge`,
  `union_with`, `inter_with`; `all_among`, `without`, `inter`, `distinct`,
  `dedup`), each with a statement of what it does to `lookup` or `among`,
  and say what a new function over maps does in the same way. Two maps
  are the same when `map_equal`, not when equal. Do not write another
  lookup, membership test or merge; a lemma about these that is not about
  the terms goes in `FlambdaMapProofs.v`, so that there is one place to
  look.
- Maps and sets keyed by something that is not a name (tags and numbers,
  which are integers; the names of the types) are the same definitions
  again, in the module `Keyed` of `FlambdaMapSpec.v`. Those that compare
  keys take the equality test of the keys as their first argument:
  `Keyed.lookup Z.eqb tag cases`, `Keyed.inter Z.eqb`, `Keyed.merge Z.eqb`.
  Those that compare none take none (`Keyed.finite_map`,
  `Keyed.map_values`, `Keyed.tabulate`, `Keyed.is_empty`). Use one test
  for each type of key, always the same: `Pos.eqb` for names, `Z.eqb` for
  integers, and for another type of key one defined beside the type with
  its proof of `decides_equality`.
  Their lemmas are the `keyed_...` of `FlambdaMapProofs.v`; one that
  depends on the test takes a proof that it decides equality (`Z.eqb_eq`,
  `Pos.eqb_eq`). The definitions on names are those of `Keyed` at
  `Pos.eqb` by computation, so a `keyed_...` lemma applies to them as it
  stands. A new operation goes in `Keyed` only, with a statement for any
  key. A lemma is proved once, on keys: the lemmas on names, in the last
  part of the file, are each `exact` of the one on keys at `Pos.eqb`. To
  add one on names, prove it on keys first.
- To define something by recursion on a type that holds lists or maps of
  itself (the meaning of a type, a check of one), go through them with
  `every`, `Keyed.for_all_bindings` and `Keyed.at_binding`, or the tests
  `every_b`, `Keyed.for_all_bindings_b` and `Keyed.at_binding_b`; on
  names the last four are written `for_all_bindings`, `at_binding`,
  `for_all_bindings_b` and `at_binding_b`. Rocq refuses a recursive call
  on what `lookup` returns, and accepts one made by the predicate or the
  test given to these.
- A state is a record with the fields `state_expr`, `state_vars`,
  `state_conts` and `state_statics`. Say what a property is about with
  the projections, and never destruct a state by position: a field added
  later must break nothing that does not mention it. For a proof by cases
  on any state `s`, `rewrite (state_eta s)` and destruct `state_expr s`;
  `simpl_state` reduces the projections of a state given by its
  components (both are in `FlambdaSemanticsProofs.v`).
- A proof never unfolds or simplifies `step`. `FlambdaSemanticsProofs.v`
  has one lemma for each construct (`step_let`, `step_let_static`,
  `step_let_cont_non_recursive`, `step_let_cont_recursive`, `step_apply`,
  `step_apply_cont`, `step_switch`, `step_invalid`), an equation saying
  what a state at that construct steps to and under which conditions, and
  `step` is opaque after them. Rewrite with the lemma of the construct at
  hand; in a proof by cases on the construct, `autorewrite with step`
  rewrites every case. `step_apply_cont_handler`, `step_apply_cont_exit`
  and `step_switch_cases` say the same of the ways in which an
  `Apply_cont` and a `Switch` do not go wrong.
- To go under a binder that both sides have, use the lemmas ending in
  `_equivalent_step` and `_refines_step` (`let_refines_step`,
  `let_static_equivalent_step`, `let_cont_non_recursive_refines_step`,
  `let_cont_recursive_equivalent_step`, ...): the two states take the
  same step, and what is left is to relate the bodies. A proof written
  with them does not depend on what the step of the binder is.
- A change to `step` is then a change to these lemmas, and to the proofs
  that use the lemma of the case that changed. To check that nothing else
  has come to depend on the definition, replace it for one build by a
  `Parameter step` with an axiom `step s = step_impl s`, `step_impl` being
  the definition, and prove the eight lemmas by `rewrite` with the axiom:
  everything else must still build. (Do not commit that.)
- To prove a local rewrite, step both sides until they meet:
  `step_equivalent` for a step of one side, `equivalent_step` and
  `refines_step` for one of each, `step_eq_equivalent` where the two
  first steps come to the same thing, `step_undefined_refines` where the
  source is undefined. `FlambdaControlFlowProofs.v` has the worked
  examples.
- Primitives are data: `FlambdaPrimitiveModel.v` has their types, a
  fragment of `flambda_primitive.mli` with the compiler's names, and
  `FlambdaPrimitiveSpec.eval_prim` says what each computes on the values
  of its arguments, given the static environment. Where it has no value
  the term is undefined. There is no axiom about them.
  - Nothing outside the three `FlambdaPrimitive*` files looks inside a
    primitive, and `eval_prim` is opaque once `FlambdaPrimitiveProofs.v`
    is loaded, as `step` is. Elsewhere use the laws that every primitive
    obeys, stated in `FlambdaPrimitiveSpec.v`: its value depends on the
    static environment only through what the symbols among its arguments
    are bound to (`eval_prim_static_values`, with the corollaries
    `eval_prim_same_static_values` and `eval_prim_consts`); if it has a
    value its arguments had the kinds it takes and the value has its
    result kind (`eval_prim_kinds`); it makes its value out of its
    arguments (`eval_prim_preserves`); it keeps its value when the
    static environment grows (`eval_prim_statics_extend`); if it has
    a value, no argument of it was a symbol bound to nothing
    (`eval_prim_symbols_bound`; this is about the arguments themselves,
    not about symbols inside their fields); and it brings no number out
    of range (`eval_prim_in_range`: given values whose constants are in
    the range of their kind and whose blocks have tags in range, it
    returns such a value. It has two more premises: the numbers the
    primitive carries itself are in range, and each symbol among the
    arguments is bound to something in range, since a primitive reads
    through a symbol: unboxing a symbol bound to a boxed number out of
    range would give that number); and it does not tell apart values
    that the observations read alike (`eval_prim_related`: where it has
    a value on some arguments it has a related one on arguments related
    to them, for a relation that `respects_observations`, such as the
    one between a block and a symbol bound to an equal block. The law
    goes one way. It is why no primitive may have a value that depends
    on which allocation an immutable value is). To reason about one
    primitive,
    rewrite with the equations `eval_prim_unary`, `eval_prim_binary`
    and `eval_prim_variadic`.
  - Adding a primitive is leaf work in those three files: a constructor,
    a case of the evaluation, of the kinds and of `prim_in_range` (the
    numbers the primitive carries: `true` if it has none), and a case in
    each proof of `FlambdaPrimitiveProofs.v` that goes by cases. Then
    teach the conversion in `flsemantics.ml` and add an example.
  - The numbers of the model are integers, where the compiler's are of
    types that hold a range. `FlambdaPrimitiveSpec.v` (the section
    "Numbers in range") says which integers the compiler's are;
    `FlambdaWellFormedSpec.numbers_in_range` is the check of a term, and
    `FlambdaRangeSpec.v` has what it gives: every state of the run of a
    unit that passes holds numbers in range only. A new place in a term
    for a number (a constant, a tag, an index) gets its case in the
    check, and a new form of value in `value_in_range`.
  - The signature of `eval_prim` and the statements of the laws are what
    the rest of the model is built on: a change to one of them is a core
    change, though the file is not a core file.
- The values of the machine are in `FlambdaValueSpec.v`. To ask what a
  value is, use an observation (`closure_of`, `block_of`,
  `boxed_number_of`, `string_of`, `array_of`, and `kind_of_value`): it
  takes the static environment, and answers the same for a value made
  as the term runs and for a symbol bound to the same thing. Do not
  match on the
  constructors of `value` for that. Where a definition by cases on
  `value` treats what is left alike, end it with a wildcard, so that a
  new form of value does not stop it compiling; a proof by cases names
  them all.
- A form of value that one primitive makes need not disturb what was
  proved before it. `FlambdaNoArrayValueSpec.v` says so of the
  immutable array as a value: a state that holds none and may run
  nothing that makes one stays so, for both machines, and between
  memories that hold none the two relations that came to relate arrays
  relate what they related. Statements of that kind are owed when a
  new form of value makes a relation of a landed statement relate
  more pairs. A primitive more is also a term more that a handler may
  be written with: a relation with a universal over such terms inside
  (`FlambdaObserversSpec.worlds_observed`, `refines_under_observers`)
  then asks more of what it relates, and a statement that has it in a
  premise, or on both sides, is owed for a class of them
  (`worlds_observed_by`, `refines_observed_by`) or for those that
  there were. `FlambdaNoArrayValueObserversSpec.v` has those of the
  primitive that makes an array.

- A rewrite between expressions applies inside any term
  (`FlambdaContextSpec.v`: `plug_refines`, `plug_equivalent`). The proof
  (`FlambdaContextProofs.transfer`) is the pattern for any argument where
  two runs differ in what their continuation closures hold: relate the
  environments, and go by induction on the length of one run.

- Names follow the compiler's `nominal/` directory, as a partial theory of
  nominal sets (`FlambdaNominalSpec.v`). Terms keep their named binders,
  which shadow, and there is no type of alpha-equivalence classes: a
  renaming is a pair of permutations given as lists of swaps, it acts on
  terms without capture (`rename_expr`), and `expr_equivalent` stands in
  for equality of classes. The statements say that a renaming fixing the
  free names of a binder with its scope may be applied to it (what
  `Name_abstraction.pattern_match` relies on), and that an unused binding
  may be removed. A fresh name is any name not free where it matters;
  nothing generates one.
- An expression stands for its alpha-equivalence class in this sense
  (`FlambdaAlphaSpec.v`): `alpha_equivalent` is the congruence generated
  by renaming bound names; it is equivariant and preserves the free names;
  and alpha-equivalent expressions are `expr_equivalent`
  (`alpha_equivalent_sound`), so everything said with `expr_equivalent`
  or `expr_refines` is said of classes. To carry a rule about a renaming
  `s` across another renaming `r`, use the conjugate
  (`FlambdaAlphaProofs.conjugate`).
- Two runs that stay in step are related by a bisimulation
  (`FlambdaSemanticsProofs.bisimulation_equivalent`), as a state and its
  renaming are (`FlambdaNominalProofs.renamed_equivalent`, from which
  `coincide_equivalent`: behaviour depends only on the free names). In
  it a closure of a recursive group is related to its renaming with a
  renaming chosen per handler (`closure_renamed`, `handler_agrees`), so
  that the handlers of a group can be renamed one at a time. Two
  runs of different lengths need the induction of
  `FlambdaContextProofs.transfer`.

- A rewrite that depends on what is known of the variables in scope is
  stated relative to a typing environment
  (`middle_end/flambda2/types/rocq/FlambdaTypesSpec.v`: `expr_refines_in`,
  over `env_satisfies`). The model of types there is naive on purpose (a
  type is unknown, bottom, an alias to a `Simple`, or that of the closures
  of given code with given value slots); a richer grammar is to add cases
  to `type_satisfied`. `follow_aliases` uses the closure types to make a
  call of a known closure direct and to replace the projection of a value
  slot from one by what the slot holds. A binder gone under must not occur in
  the typing environment, which is the simplifier's freshness invariant.
  There is no substitution: as in the simplifier, a variable is replaced
  at its uses by following its alias, and its `Let` is then unused.

  Into a handler, what is known at the `Let_cont` stays known and the
  parameters are added (`let_cont_handler_refines_in`); underneath,
  `FlambdaContextProofs.handler_refines_at` says that a handler refines
  another where it runs, under the variables of its closure.

Not yet there: the types of a continuation's parameters, joined over its
uses; types beyond aliases, with meet and join; that the generated alpha-equivalence is contained in the
one defined by recursion on two expressions (`alpha_structural`; the other
containment is proved); equivariance of execution in the continuations
(renaming closures and observed exits; there is a CR in
`FlambdaAlphaSpec.v`); the simplifier's
invariant that bound names are distinct from those in scope;
removing an unused static binding (which needs the free symbols and code
ids of terms, code and values); rewriting the body of code by what is
known of the static environment; the compiler's direct call of tupled
code, which gives the code the fields of the tuple; mutable memory in
the state of the machine (the prototype has it in a machine of its
own), and any link to the simplifier's code.

### Changes to the compiler made on this branch

The work on the semantics does not change the compiler: where the model
and the compiler part ways, the difference is recorded and the model
follows the compiler. The exceptions are listed here, each made at the
request of the branch's owner, each a commit of its own with a subject
in the compiler's style, so that it can be proposed upstream by itself.

- `middle_end/flambda2/types/meet_and_n_way_join.ml`,
  `n_way_join_closures_entry`: the n-way join keeps the function type of
  a function slot only if every input has one, as the binary join does
  (discrepancies T1-4 and R3-7). Before, with the n-way join, a correct
  program could stop on an `Invalid` put in place of a live call, and
  the `zero_alloc` check could accept a function that allocates. Tests:
  `testsuite/tests/flambda2/n_way_join_function_types*.ml`.
- `middle_end/flambda2/simplify/simplify_unary_primitive.ml`,
  `simplify_get_tag`: `Get_tag` of an argument known to be a tagged
  immediate (a constant, or a name whose type has no blocks) becomes
  `Invalid`; nothing changes when the argument may be a block. The model
  gives `Get_tag` of an immediate no value (decision D5). Before, the
  primitive was left in the term in branches that are never executed,
  which code using GADTs has. Provisional, with a `CR` at the place; the
  `Get_tag` relation that the types record is not changed. Tests:
  `testsuite/tests/flambda2/simplify/get_tag_of_immediate.ml` and
  `testsuite/tests/flambda2/get_tag_of_immediate_run.ml`.
- A flag, `-flambda2-join-sharing {physical|structural|none}`
  (`Oxcaml_flags.Flambda2.join_sharing`, `Flambda_features.join_sharing`;
  `flambda2-join-sharing=structural` in `OCAMLPARAM`). The join of types
  keeps a type as it is, and does not join it, where the joined
  environments have it physically equal, in four places
  (`Join_env.n_way_join_env_extension`,
  `Join_env.import_or_delay_n_way_join_of_concrete_type`,
  `Meet_and_n_way_join.n_way_join_int_indexed_product` and
  `Meet_and_join.join_int_indexed_product`), so its result depends on
  how types are shared in memory. The flag chooses the test there:
  physical equality (the default: nothing changes without the flag), a
  structural equality on types that is new (`Type_grammar.equal`,
  `Type_descr.equal`), or none. The model ignores physical equality
  (decision D30) and is compared with the compiler under `structural`.
  Test: `middle_end/flambda2/tests/api_tests/join_sharing.ml`; its
  expected file prints the stamps of variables, as `meet_test.expected`
  does, and is to be regenerated when the joins create another number
  of variables.
- Three commits that let a test see the types that Simplify computes
  and the names that a term is stored with (decision D33). None
  changes what the compiler does, and each new entry point of the
  observer and of the openings has a `CR` at its place saying that it
  is for tests. The test of the openings registers an observer, so to
  revert the observer, revert the openings first.
  - `Type_grammar.Block_size.of_target_ocaml_int` and
    `to_target_ocaml_int`: the type stays abstract; the two are the
    identity and check nothing.
  - An observer of typing environments.
    `Typing_env.For_testing.set_observer`, in the internal interface
    of the library of types, registers a function, and Simplify calls
    `Typing_env.notify_test_observer` at the start of each continuation
    handler that it simplifies and, with the final environment, at the
    return continuation of the unit. With no observer, which is always
    so in the compiler, the call reads a reference and returns. Test:
    `middle_end/flambda2/tests/api_tests/typing_env_observer.ml`.
  - `pattern_match_for_printing` of `Let_expr`, `Continuation_handler`,
    `Non_recursive_let_cont_handler`, `Recursive_let_cont_handlers` and
    `Function_params_and_body`: the opening of a binder that does not
    freshen the names it binds (unless `-dfreshen` is given), which the
    printer had and nothing exported. The environments reported to the
    observer are keyed by those names. Test:
    `middle_end/flambda2/tests/api_tests/binder_openings.ml`.

### Testing Simplify against the semantics

`middle_end/flambda2/tests/semantics/` checks the real simplifier against
the model on small examples written in the fexpr language. For each `.fl`
file, `flsemantics` (`middle_end/flambda2/tests/tools/flsemantics.ml`)
parses it to a Flambda unit, runs Simplify, converts the unit before and
the unit after to terms of the model, runs both on the machine extracted
from `FlambdaSemanticsSpec.v`, and compares what the two runs come to: the
continuation exited through and its arguments, followed through symbols
into static blocks. A closure is not looked into. The check passes if the
outcomes are the same, or if the run before is undefined. This is a test,
not a proof.

- Beside the test of one run, the tool reports whether the pair is
  *validated*: whether `FlambdaValidateSpec.validate_unit`, extracted,
  accepts it. That is one function with one theorem
  (`validate_unit_sound`): the term after refines the term before as a
  compilation unit is run, on its own from empty environments. The tool
  does not put the rewrites together itself; to change which are applied,
  or their order, change `normalise_before` and `normalise_after` there.
  The rest of this item is what `validate_unit` does.
  The decision is `FlambdaCompareSpec.expr_eqb`, equality up to bound
  names, after the term before has been normalised by rewrites proved to
  refine it: `inline_calls` (direct calls of code bound by an enclosing
  `Let_static`, in handlers and in the bodies of code too), `fold_constants`, `collapse_switches` (a `Switch` whose
  arms are all the same), `inline_tail_conts` (a continuation applied
  once at the end of its scope), `follow_aliases`, `remove_unused_lets`
  and `drop_deleted_code`, gone round until nothing changes as
  they enable one another. Only equivalences may be applied to the term
  after: `follow_aliases`, `inline_known_calls` (inlining where the callee
  is a symbol bound to a closure of the code called, which makes it an
  equivalence), `fold_constants`, `remove_unused_defined_lets` (an unused
  `Let` whose defining expression has a value given the variables in
  scope), `drop_deleted_code` and `merge_statics` (one group for
  `Let_static`s that follow one another, code before symbols).
  `fold_constants` asks `eval_prim` for the value of a primitive on
  constants, under the empty static environment: on constants a
  primitive does not look at it (`eval_prim_consts`). Which primitives
  are folded, and removed when unused, is `prim_may_fold`: all but
  `Opaque_identity`, which Simplify leaves alone, so that an example can
  have values Simplify knows nothing of.
- A `Let_static` is permitted at the toplevel of a unit only
  (`FlambdaWellFormedSpec.well_formed_unit`, after Simplify's
  `at_unit_toplevel`). The tool checks the term before, and does not give
  one that fails to Simplify (`static_in_loop.fl`); the term after failing
  is an error. The machine enforces what keeps a static binding from
  running twice: calling code whose body binds something static is
  undefined, as is a `Let_cont_recursive` one of whose handlers does
  (`binds_statics`), and so is binding a static name twice; so static
  environments only grow (`run_extends_statics`). A rewrite of the body
  of code or of a recursive handler must therefore keep whether it binds
  something static (the premise of `plug_refines` for such a context),
  and code is inlined only if its body binds nothing static
  (`FlambdaApplySpec.inlinable`). A well-formed unit run on its own never
  meets one of these three (`well_formed_never_faults`). The invariant of
  the proof is about the names a state may still bind: they are distinct,
  none is bound, and those of a handler are not among those of the
  continuations its closure holds. A rewrite inside a handler may then
  rely on what was bound where the handler is defined: the lemmas for it
  are the ones ending in `_where` in `FlambdaContextProofs.v`, which take
  a property of the static environment that every extension keeps
  (`inline_in_handler.fl` needs them). A rewrite that needs more of
  where it runs than the static environment (that the state is in
  range, say) uses the statements ending in `_from` of
  `FlambdaContextWhereSpec.v`, which take a predicate on states that a
  step keeps; the one for the body of code also gives the kinds of the
  values that the call passes.
- Only the two inlining passes go into the bodies of code by
  themselves, as they need what is known there. The others are applied
  to the bodies by `rewrite_code`, which takes any rewrite and is proved
  to refine, or to be an equivalence, when the rewrite does under all
  environments (`inline_in_code.fl` needs both).
- `showcase.fl` is the example to read first: direct and indirect calls,
  a closure with a value slot, nested switches and a join, all inlined by
  Simplify and validated. In `switch_nested.fl` Simplify uses what a
  `Switch` arm tells it of the scrutinee in the continuation the arm goes
  to. The normalisation has one case of that, `thread_switches`: an arm
  that goes to a handler which is a `Switch` on the same scrutinee goes
  where that one's arm does. What is known at the uses of a continuation
  does not reach its handler in general.
- Removing a static binding that nothing uses is not sound under every
  environment (a value already there may be a closure of the code), only
  under those in which the names removed do not occur
  (`FlambdaUnusedStaticsSpec.drop_statics`, `envs_free_of`); a term run
  on its own is a case (`program_refines`). The tool guesses which names
  are unused and the extracted function checks the guess. An example
  that needs it is validated like any other, for the unit run on its
  own; the two other ways of being validated hold under every
  environment as well (`validate_unit_sound_everywhere`).
  Its proof is a simulation that lets the second state do anything once
  the first is undefined (`FlambdaSemanticsProofs.simulation_refines`),
  under the invariant that nothing reachable refers to a name removed;
  that needs the primitives to introduce no symbols or code ids, which is
  a lemma (`eval_prim_avoids`, an instance of `eval_prim_preserves`).
- Simplify makes a closure, an immutable block or a boxed number whose
  parts are known a static binding of a symbol, and in the code of a
  closure so lifted `my_closure` is the symbol. The validation does
  the same to the term before (`FlambdaLiftSpec.lift_unit`, with
  `FlambdaMyClosureSpec.bind_my_closures`), where a guess says to:
  which `Let`s, to which symbols, and which code is that of which
  symbol. The tool reads the guess off the term after, and the
  extracted function checks it; a wrong guess is a unit that is not
  validated. This is the fourth way of being validated
  (`Same_once_lifted`), and like the third it holds for the unit run
  on its own. Lifting changes the number of allocations (discrepancy
  V3-2), which the model does not observe.
- Setting `FLSEMANTICS_PRINT_TERMS` makes the tool print the two
  normalised terms of an example that is not validated, which is how to
  find the rewrite that is missing.
- Code ids must be the same on the two sides, and Simplify gives a new
  one to each new version of code. The tool gives the new id the number
  of the old where the old is bound as deleted and not otherwise used;
  this is trusted, and there are CRs in `flsemantics.ml` and
  `FlambdaCompareSpec.v` on relating code ids in the comparison itself.
  An example that is the same but not validated shows what normalisation
  is missing. To add a rewrite: write it as a function in a Spec file,
  prove that its result refines (or is equivalent to) its argument, add
  it to `FlambdaExtract.v`, and chain it in `flsemantics.ml`.
- The machine is extracted by `FlambdaExtract.v` into
  `tests/semantics/extracted/` (`make -C rocq all` regenerates it; the
  files are committed, and none is written by hand). No part of the
  compiler is built from it.
- Who commits the extracted files. Only the integrator commits
  `tests/semantics/extracted/`, on the integration branch. They are
  generated, one file from many sources, so two workstream branches that
  each committed them would conflict at every merge, in files nobody
  wrote. On a workstream branch, run `make -C rocq all` before the tests,
  which regenerates the files where the tests find them, and leave them
  out of your commits: stage the files you changed by name, not with
  `git add -A`. `git status` will go on showing them as modified;
  `git checkout -- middle_end/flambda2/tests/semantics/extracted` puts
  them back, which a rebase needs first. When a branch is merged, the
  integrator runs `make -C rocq all` on the result and commits what it
  regenerates, on top of the branch.
- An extracted file that is no longer written. The extraction writes a
  file for each module that something extracted uses, and leaves every
  other file of the directory as it is. So when what is extracted no
  longer uses a module, its `.ml` and `.mli` stay committed as they
  were last written and are still compiled into the library of the
  tests, and `git diff` after `make -C rocq all` does not show them. To
  see them: delete every `.ml` and `.mli` of `tests/semantics/extracted/`
  (not its `dune` file, which is written by hand) and
  `middle_end/flambda2/rocq/FlambdaExtract.vo`, and run
  `make -C rocq all`. Look at `git status` of the directory only when
  that build has succeeded: a build that stops before the extraction
  leaves every file deleted. A file then shown as deleted is one that
  the extraction no longer writes. That alone does not show that nothing
  uses it: a test tool is written by hand and may still name the module.
  Search the test tools and the extracted files that remain for the
  module. If one names it, what it uses of the module is to be named in
  `FlambdaExtract.v` again, which makes the extraction write the file;
  if none does, the integrator removes the file.
- The test tool and the examples come last. The compiler must build at
  every commit of the integration branch, and `make -s boot-compiler`
  type-checks `flsemantics.ml` against the extracted files that are
  committed. So whatever needs the regenerated files, to build or to
  pass (a change to `flsemantics.ml` or to another test tool, a new
  `.fl` example and its expected file), goes at the end of the branch,
  after the commit with the lines for the shared lists if there is one:
  one commit, or a short series in which each builds on the one before.
  Every commit before that must pass `make -C rocq all` and
  `make -C rocq assumptions`. The integrator does not fast-forward
  those last commits: it applies them in order, and the first of them
  as one commit with the regenerated files.
- Trusted and unverified: the conversion from Flambda terms to model
  terms, primitives and constants included. It must send each to the one
  of the same name, and refuse what the model has not got; read it in
  `flsemantics.ml`. What a primitive computes is not trusted OCaml: it is
  `eval_prim`, a definition to be reviewed. The whole list is the section
  "Flambda 2" of `rocq/TRUST-LEDGER.md`, which has no axiom.
- A construct or primitive the conversion does not know makes the example
  "not checked"; it is never run as if undefined, which would pass for
  nothing. When the model gains a construct, teach the conversion and add
  an example.
- To run:
  `dune build --root=. --workspace=duneconf/main.ws
  @middle_end/flambda2/tests/semantics/runtest` (after
  `make -s boot-compiler`). The expected output of `X.fl` is
  `X.expected`, beside it.
- To add an example, add `X.fl`, whose module symbol must match its name,
  and `X.expected`; `dune promote` writes the latter from what the tool
  printed. Nothing else is to be touched: there is no list of examples.
  The dune rules are generated from the `.fl` files that are there
  (`gen_rules.sh`, included by `run/dune`; the `dune` file says why it
  takes two directories), so two branches that each add an example do not
  meet in any file.

### Discrepancies with the compiler

Record every place where the compiler and the model part ways; some of
them will be bugs or misfeatures of the compiler. That is, whenever:

- the compiler (Simplify, the types library, the terms, the tests or
  their tools) does, assumes or permits something that the semantics,
  read plainly, does not justify: it rewrites to `Invalid` something the
  machine can run, or relies on an invariant that nothing checks or
  documents;
- two parts of the compiler disagree with each other;
- the model had to make something undefined, take a premise or make a
  special case in order to follow the compiler;
- an extracted function and the OCaml it models disagree in a test.

An entry says what the compiler does and where (file and lines), what
the model says, the example or statement that shows it and how to run
it, and which it seems to be: a candidate bug, an unchecked assumption,
an inconsistency in the compiler, intended and documented, or not
known. Do not change the compiler to remove a discrepancy, and do not
weaken a statement to hide one: model what the compiler does where that
is sound, say so in the header of the `Spec` file, and record the
entry. An entry has a label, the stream that recorded it and a number
(`C3-1`), and comments and documents cite it by the label. What each
label cited in the tree stands for, and each decision cited as `D<n>`,
is said in a line or two in `middle_end/flambda2/rocq/DECISIONS.md`. A
mistake in the model that is then fixed is not an entry, unless it is
not clear which side is wrong.

## Extraction pitfalls

- `&&` is extracted to OxCaml's short-circuiting `( && )`, but a `let` is
  evaluated eagerly: bind an expensive check inside the `if` or `&&`
  that needs it, not in a `let` before it (`find_deletions` runs the
  termination analysis last for this reason).
- An extracted module may depend on compiler modules, but a compiler
  module that the extraction maps onto (such as `Ssa`) must not depend
  on an extracted one, or dune reports a dependency cycle.
- Extraction puts the same `open` lines at the top of a `.ml` and its
  `.mli`; the Makefile strips the unused ones from the `.mli`, where only
  types can be referred to.
- `Separate Extraction` extracts only what the listed entry points use;
  an unused definition silently disappears from the `.ml`. Keep the entry
  points in `Extract.v` to what the compiler calls.
- Everything extracted is run through `ocamlformat` by the Makefile, so
  `make fmt` passes without touching the generated files.

## Writing the statements

- Say what the booleans mean with `Prop`s over the observations of
  `SsaModel.v`, not over the compiler's data.
- Use the notations `SsaModel.v` defines (`a ▷ b` for `a`
  dominating `b`) in Spec and Proofs files, where a statement is read by
  eye. Implementation files are code and call the primitives as the
  compiler does.
- Use `∀` and `∃` for quantifiers and `∧` and `∨` for conjunction and
  disjunction, importing `OxCamlLogic.v`. Keep the default `->` and `<->`
  arrows, and `exists` for the tactic.
- Use `∈` and `∉` for list membership and non-membership; `OxCamlList.v`
  defines these notations for `List.In` and its negation.
- Prefer a per-element statement to a counting one: "every parameter at
  position `k` has a dead induction variable with `param_index = k`" is
  checkable; "there are as many as parameters" is not what the pass
  relies on.
- The existential witnesses in a specification should be things the
  analysis actually computes (`∃ args, goto_args ... = Some args`),
  so that the proof is a reading-back of the code.
- Define machine execution once (`SsaSemanticsSpec.v`). Derive operation
  equations only for available, canonical definitions
  (`SsaExecutionProofs.v` and `SsaContextProofs.v`). An abstract
  arithmetic valuation such as
  `AffineSsa.value_words` is constrained by `SsaSemanticsSpec.eval_operation`
  (`AffineSsaProofs.op_semantics`), not by a second statement of the
  arithmetic, and needs a proved connection to execution
  (`LoopValueSnapshotProofs.v`) before it can justify optimization.
- Keep analysis preconditions in named propositions, not in executable
  types. For example, the counter proof needs `biv_spec`, which
  `InductionVarProofs.analyze_spec` establishes for the discovered variables.
  Composing the analyses must discharge these preconditions, rather than
  turn them into trusted assumptions about the program.
- Execution must not depend on the analysis being verified. For example,
  `LoopExecutionProofs.region_steps` is a sequence of instruction steps,
  not a record assuming counter recurrence or invariant bounds. A
  property of an analysis result is stated as what it means
  (`loop_invariant_value`), tied to the function by a read-back theorem
  (`is_loop_invariant_spec`), and then justified operationally
  (`LoopInvariantExecutionProofs.loop_invariant_value_same`).

## Proof style

- Statements and proofs stay under 80 columns. Give explicit types to
  statement binders and use `insn` for instruction variables.
- Case splits that the reader does not need to follow are written inline
  with `[ ... | ... ]`; those that carry an argument use bullets
  (`-`, `+`, `*`), with a comment when the case is not evident from the
  pattern.
- To read a boolean check back: `unfold` it, `destruct (f x) eqn:H;
  [ | discriminate ]` on each test, `apply andb_true_iff in H` on each
  `&&`, `rewrite forallb_forall in H` on each `forallb`, `cbv zeta` and
  `cbv iota` to step a `let` or a `match` on a known constructor,
  `injection H as <-` to read a `Some`.
- Equations on `int` go through `to_Z` and `lia`; `case_int_tests`
  (`OxCamlInt.v`) and `case_zcmp` (`ZTactics.v`) split on the boolean
  comparisons in a goal first.
- `Opaque helper` keeps a helper folded while the function calling it is
  unfolded; `Transparent` it afterwards.
- `functional induction` proves properties of a `Function`; the
  decreasing measure is proved once, at the definition.
- A `Section` with `Variable`s and `Hypothesis`es shares the setting of
  many lemmas (the graph in `LoopCounterExecutionProofs.v`); the lemmas are
  generalised over them at `End`.
- Name hypotheses after what they say (`Hexit`, `Hwf`), not `H0`, `H1`.
- Do not comment what a named predicate already says; do comment every
  non-trivial lemma, and every case of a long proof that is not evident.

## Checklist before pushing

- `make -C rocq all` passes and the extracted diff is the intended one.
- `make -s boot-compiler`, the pass's tests and `make -s fmt` pass.
- New definitions sit in the most generic file that fits, and in a Spec
  only if a reviewed statement needs them; new modules are registered in
  `rocq/Makefile`, `dune` and `Extract.v`.
- Every non-trivial lemma and theorem has a comment; no comment restates
  a named predicate.
- The headers list what is trusted, and nothing is `Admitted`;
  `make -s -C rocq assumptions` passes, so there is no axiom that
  `rocq/TRUST-LEDGER.md` does not list.
- For the Flambda 2 semantics: every discrepancy with the compiler met
  on the way is recorded ("Discrepancies with the compiler"), what
  needs the regenerated extracted files is in the last commit, and no
  document or comment cites a commit that is not on the integration
  branch or a file that is not in the tree.
- A change to the body of a definition of a Spec file, or of
  `FlambdaModel.v`, `FlambdaPrimitiveModel.v` or `FlambdaKindModel.v`,
  comes with the list of the statements already in the tree that the
  definition reaches, directly or through other definitions, and with what
  happens to each: a premise is strengthened, a conclusion is
  weakened, or neither. `python3 rocq/check-changed-defs.py BASE TIP`
  lists, for the commits from BASE to TIP, the definitions whose body
  changes and the statements that each reaches, named where there
  are at most 40 and counted otherwise;
  `python3 rocq/check-changed-defs.py --reach REV NAME` does the same
  for one definition at one commit. Both are run from the root of
  the tree, and the record is counted on the tree that it is
  committed on. It is a search over names: what happens to each
  statement is read, not computed. Do not write that a function
  "validates more" as if it were a statement: say what the function
  now does, and give what was measured as a measure.
