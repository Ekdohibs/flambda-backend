# Who owns which file of the Flambda 2 Rocq work

This is the ownership table of `PARALLEL-PLAN.md` (sections 6 and 7): for
each workstream, its owner and the files only that owner edits. A file
has one owner. If you need a change in a file you do not own, ask its
owner; do not make it.

How to read it and keep it current:

- A stream is owned by the agent working on it, named by the stream
  (`C0`, `V1`, ...). "Unassigned" means no agent has the stream yet; the
  orchestrator says when one does, and the integrator records it here,
  with the stream's branch.
- A pattern such as `FlambdaNormalise*` covers the files that match it
  now and later, in the directory given.
- A new file is its creator's. Add it to your own stream's section in the
  commit that creates it, unless a pattern there covers it already. Edit
  no section but your own: two streams then never change the same lines
  of this file.
- `middle_end/flambda2/rocq/status/<stream>.md` is its stream's, and
  needs no entry.
- A file that no section covers is C0's.
- C0 keeps the layout of this file; the sections are their streams'.

Paths are from the root of the repository.

## C0: core owner and integrator

Owner: C0. Phase 0 was done as C0, and what it created is C0's.

- `middle_end/flambda2/rocq/FlambdaModel.v`
- `middle_end/flambda2/rocq/FlambdaKindModel.v` (the kinds, moved out of
  `FlambdaModel.v` by the change that made the primitives data)
- `middle_end/flambda2/rocq/FlambdaValue*` (the values of the machine
  and their observations, moved out of `FlambdaSemanticsSpec.v` by the
  same change)
- `middle_end/flambda2/rocq/FlambdaSemantics*`
- the signature of `eval_prim` and the statements of the laws of
  `middle_end/flambda2/rocq/FlambdaPrimitiveSpec.v`. The file is C1's,
  and a primitive is added there without asking; a change to the
  signature or to a law is a core change
- `middle_end/flambda2/rocq/FlambdaContext*`
- `middle_end/flambda2/rocq/FlambdaWellFormed*`
- `middle_end/flambda2/rocq/FlambdaMap*` (Phase 0: finite maps and sets
  of names)
- `middle_end/flambda2/rocq/FlambdaExtract.v` (a stream asks for names
  in its list with a commit of its own: `rocq/HACKING.md`, "Adding a
  module")
- `middle_end/flambda2/rocq/OWNERS.md` (Phase 0: its layout; see above)
- `middle_end/flambda2/rocq/DECISIONS.md` (the index of the labels of
  decisions and discrepancies that the tree cites; the integrator adds
  a label's line when a merge brings the label)
- `middle_end/flambda2/rocq/requests/C0-well-formed.md` and
  `requests/C0-toplevel-try-body.md` (requests)
- `middle_end/flambda2/rocq/status/phase0.md` (Phase 0)
- `middle_end/flambda2/tests/semantics/extracted/` (generated; only the
  integrator commits it, see `rocq/HACKING.md`)
- `middle_end/flambda2/tests/semantics/dune`,
  `middle_end/flambda2/tests/semantics/gen_rules.sh` and
  `middle_end/flambda2/tests/semantics/run/dune` (Phase 0: the rules that
  run the examples; the examples themselves are V2's)
- `rocq/Makefile` (a stream asks for lines in its lists in the same way
  as for `FlambdaExtract.v`)
- `rocq/HACKING.md`
- `rocq/review-pdf.pl`
- `rocq/check-changed-defs.py` (the search for the statements that a
  changed definition reaches: `rocq/HACKING.md`, "Checklist before
  pushing")
- the integration branch

Files that exist and that the plan gives to no stream. They are C0's
until the stream in brackets, whose subject they are, has an owner; then
they are that stream's, and the integrator moves the line to its
section:

- `middle_end/flambda2/rocq/FlambdaControlFlow*` (S3, with S2 for the
  switches)
- `middle_end/flambda2/rocq/FlambdaApply*` (S4)
- `middle_end/flambda2/rocq/FlambdaStatic*` (S5)

## C1 to C5: core changes

Each is designed by its owner and applied by C0, one at a time
(`PARALLEL-PLAN.md`, section 7, "Core change requests"). The owner edits
its design note, under `middle_end/flambda2/rocq/requests/`, and the new
files it creates, and lists them here; the changes to C0's files are
made by C0. On its own branch an owner may change C0's files to try the
design out; that is the request, not the change.

- C1, primitives as data. Owner: C1, on `flambda2-rocq/C1-primitives`.
  - `middle_end/flambda2/rocq/requests/C1-primitives.md` (the request)
    and `requests/C1-int-ops.py` beside it (the generator of the
    examples that compare the integer primitives with a reference)
  - `middle_end/flambda2/rocq/FlambdaPrimitive*` (the primitives, what
    they compute and the laws they obey; adding a primitive changes
    these files and no other `.v` file; see C0's section for what of
    them is the core's)
  - the conversion of primitives and of constants in
    `middle_end/flambda2/tests/tools/flsemantics.ml`, until V3 has an
    owner
  - `middle_end/flambda2/rocq/FlambdaRange*` (what a unit whose numbers
    are in range gains; the check itself is in C0's
    `FlambdaWellFormedSpec.v`)
- C2, kinds and well-formed terms. Owner: C2, on
  `flambda2-rocq/C2-epoch` (the series, merged),
  `flambda2-rocq/C2-binders` (the second series) and
  `flambda2-rocq/C2-kinds` (the prototype).
  - `middle_end/flambda2/rocq/requests/C2-kinds.md` (the request)
  - `middle_end/flambda2/rocq/FlambdaWellKinded*` (the check of kinds
    and its statements)
  - `middle_end/flambda2/rocq/FlambdaStaticsInScope*` (the check that
    the static names of a unit are in scope where they are mentioned,
    row 4 of the conditions of a well-formed unit, and what it gives
    of a run; the conjunct of `well_formed_unit` is C0's to wire)
  - `middle_end/flambda2/tests/semantics/kinds_*.fl` and their
    `.expected` files
  - `middle_end/flambda2/rocq/FlambdaRegions*` (a machine with
    regions beside that of `FlambdaSemanticsSpec.v`, which says where
    a value allocated in a region is used after the region is gone,
    and its statements; on `flambda2-rocq/C2-regions`)
  - `middle_end/flambda2/tests/semantics/region_*.fl` and their
    `.expected` files (the controls of the check of regions that the
    test tool makes with that machine)
  - `middle_end/flambda2/tests/semantics/regions/` (controls of the
    check of regions that are run with its counts printed, some with
    the options of the compiler that make Simplify lift a handler, and
    the rules that run them)
  - `middle_end/flambda2/tests/tools/flmodel_to_fexpr.ml` and
    `middle_end/flambda2/tests/tools/flroundtrip.ml` (a term of the
    model printed back as fexpr, and the round trip that checks the
    conversion with it; on `flambda2-rocq/C2-roundtrip`; the plan's
    converter stream's when it has an owner). The lines of
    `flsemantics.ml` that give the round trip the conversion are V2's
  - `middle_end/flambda2/tests/semantics/roundtrip/` (the rules that
    take every example through the round trip, and the answers that
    are not "the same", with their reasons)
- C3, sets of closures. Owner: C3, on `flambda2-rocq/C3-closures`.
  - `middle_end/flambda2/rocq/requests/C3-closures.md`
  - `middle_end/flambda2/rocq/requests/C3-partial-application.md`
  - `middle_end/flambda2/rocq/requests/C3-tupled-functions.md`
  - `middle_end/flambda2/rocq/FlambdaClosures*` (first facts about sets
    of closures; S4's once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaPartialApplication*` (what a call
    with fewer or more arguments than parameters comes to)
  - `middle_end/flambda2/rocq/FlambdaTupled*` (what a call that gives
    tupled code a tuple comes to)
  - `middle_end/flambda2/rocq/FlambdaKnownSlots*` (a value slot that
    every set of closures of the unit gives the same symbol or
    constant is read as that, in code too, for the validator; S4's
    once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaStubCalls*` (in the code of a
    stub, the direct calls of the unit's other code are made indirect,
    for the validator; S4's once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaCopyCode*` (code of the unit
    copied under new code ids for static sets of closures that
    Simplify gave new code, for the validator; S4's once it is
    staffed)
  - `middle_end/flambda2/rocq/FlambdaClosureCalls*` (a call of a
    closure that a `Let` in sight makes is made direct, for the
    validator; S4's once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaSetsOfCode*` (what every set of
    closures of a unit that has a code at a function slot has, kept
    by a run) and `middle_end/flambda2/rocq/FlambdaClosureOfCode*`
    (in the body of code, what `my_closure` is known to be from its
    code is used, for the validator; S4's once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaRegionsErasedProofs.v` and
    `middle_end/flambda2/rocq/FlambdaRegionsPlainStepProofs.v` (for
    C2's erasure of regions: the relation between a state of the
    machine of regions and the state that runs the erased term, and
    the steps at the expressions where that machine does what `step`
    does; the other `FlambdaRegions*` files are C2's, but for
    `FlambdaRegionsCalls*`, which are C5's)
  - `middle_end/flambda2/rocq/status/C3.md`
  - the examples `middle_end/flambda2/tests/semantics/set_*.fl`,
    `apply_partial*.fl`, `apply_over*.fl`, `tupled_apply*.fl` and
    `stub_by_hand*.fl` there, with their `.expected` files
- C4, heap. Owner: C4, on `flambda2-rocq/C4-heap`.
  - `middle_end/flambda2/rocq/requests/C4-heap.md` (the request) and
    `requests/C4-heap-fallout.md` (what each step costs the other
    owners)
  - `middle_end/flambda2/rocq/FlambdaHeapSpec.v` and
    `middle_end/flambda2/rocq/FlambdaHeapProofs.v` (the primitives of
    the heap, their effects and coeffects, closed memories; they come
    before the semantics)
  - `middle_end/flambda2/rocq/FlambdaHeapMachineSpec.v` and
    `middle_end/flambda2/rocq/FlambdaHeapMachineProofs.v` (the
    prototype: a machine that runs those primitives, beside the machine
    that exists; `FlambdaHeapEquivariance*` are N1's)
  - `middle_end/flambda2/rocq/FlambdaExitWorldSpec.v` and
    `middle_end/flambda2/rocq/FlambdaExitWorldProofs.v` (what a run
    leaves at an exit, with the static environment of its last state,
    and refinement with two such worlds related)
  - `middle_end/flambda2/rocq/FlambdaRunUnderSpec.v` and
    `middle_end/flambda2/rocq/FlambdaRunUnderProofs.v` (a unit run
    under a continuation environment is the unit run on its own and
    then what the environment binds the continuation of its exit to)
  - `middle_end/flambda2/rocq/FlambdaObserversSpec.v` and
    `middle_end/flambda2/rocq/FlambdaObserversProofs.v` (what may
    follow a unit, and two exits that nothing that may follow tells
    apart)
  - `middle_end/flambda2/rocq/FlambdaValidateRelatedSpec.v` and
    `middle_end/flambda2/rocq/FlambdaValidateRelatedProofs.v` (a second
    check of a unit against the unit it came from, for the units whose
    exits a lifting changes: nothing that may follow a unit tells the
    two exits apart; the test tool asks it only of a unit that it
    reports as not validated, and prints what it accepts on a line of
    its own, never as validated)
  - `middle_end/flambda2/rocq/status/C4.md`
- C5, exceptions. Owner: C5, on `flambda2-rocq/C5-epoch`.
  - `middle_end/flambda2/rocq/requests/C5-exceptions.md` (the request)
  - `middle_end/flambda2/rocq/requests/C5-imported-symbols.md`
  - `middle_end/flambda2/rocq/FlambdaTrap*` (examples of the checks of
    exception handlers and of trap actions, which are in
    `FlambdaWellFormedSpec.v`; later, what is proved of them)
  - `middle_end/flambda2/rocq/FlambdaImports*` (the static environment
    of a unit that imports symbols, and what a primitive does with an
    imported symbol; a prototype)
  - `middle_end/flambda2/rocq/FlambdaValidateImports*` (the validation
    of a unit run from its imports: which verdicts hold of it)
  - `middle_end/flambda2/rocq/FlambdaDemote*` (the demotion of an
    exception handler that no call can raise to, as a rewrite, with
    what it keeps)
  - `middle_end/flambda2/rocq/requests/C5-never-returns.md` (a
    request: calls that never return)
  - `middle_end/flambda2/rocq/FlambdaNeverReturns*` (the rewrite of
    the term before that gives a call of code that never returns a
    return continuation whose handler is `Invalid`)
  - `middle_end/flambda2/rocq/requests/C5-imports-with-contents.md`
    (a request: a unit run after the units it depends on)
  - `middle_end/flambda2/rocq/FlambdaRegionsCalls*` (calls on C2's
    machine of regions: code cannot end a region of its caller, and
    what follows of the values that a run exits with; C5's, though
    the names match the pattern of C2's files)
  - `middle_end/flambda2/rocq/status/C5.md`
  - the examples `middle_end/flambda2/tests/semantics/`
    `exn_extra_args.fl`, `push_unit_return.fl`, `raise_in_call.fl`,
    `raise_over_handler.fl`, `try_in_code.fl`, `try_raise.fl` and
    `uncaught.fl`, with their `.expected` files

## T1 to T7: types

All in `middle_end/flambda2/types/rocq/`, library `OxCamlFlambda2Types`.

- T1, grammar and meaning. Owner: T1, on `flambda2-rocq/T1-grammar`.
  - `middle_end/flambda2/types/rocq/FlambdaTypes*` (today's naive model,
    which T1 replaces)
  - `middle_end/flambda2/types/rocq/FlambdaTypeGrammar*` (the grammar
    of types)
  - `middle_end/flambda2/types/rocq/FlambdaTypeMeaning*` (what a type
    means: `type_satisfied`; `FlambdaTypeMeaningBool*` is the meaning
    as a test)
  - `middle_end/flambda2/types/rocq/FlambdaTypeWellFormed*` (when a
    type is well formed)
  - `middle_end/flambda2/types/rocq/FlambdaTypeEqual*` (when two types
    are the same type, after `Type_grammar.equal`)
  - `middle_end/flambda2/types/rocq/FlambdaTypeAliasKinds*` (the
    aliases of a type with their kinds)
  - `middle_end/flambda2/types/rocq/FlambdaTypeRename*` (renaming the
    variables of a type, and the names that a type mentions)
  - `middle_end/flambda2/types/rocq/FlambdaTypeNominal*` (types as
    things that contain names)
  - `middle_end/flambda2/types/rocq/FlambdaTypeAccess*` (what the
    provers and reification read of a type)
  - `middle_end/flambda2/types/rocq/FlambdaTypeCreators*` (what the
    other creation functions of types make)
  - `middle_end/flambda2/types/rocq/FlambdaMeetTermination*` (the fuel
    of the meet: more fuel changes no answer, and what it is for the
    fuel to be enough; T1's by agreement with T4, though the names
    match T4's pattern)
  - `middle_end/flambda2/types/rocq/FlambdaMeetDepth*` (the fuel of
    the meet of types with no name inside: their depth is enough;
    T1's by the same agreement)
  - `middle_end/flambda2/types/rocq/FlambdaMeetRelationFuel*` (the
    fuel of the meet for an arm of a `Switch` whose scrutinee is the
    result of a relation; T1's by the same agreement)
- T2, typing environment. Owner: T2, on `flambda2-rocq/T2-env`.
  - `middle_end/flambda2/types/rocq/FlambdaBindingTime.v` (binding times
    and name modes, after `types/env/binding_time.ml` and
    `nominal/name_mode.ml`; written first on T3's branch, which needs it)
  - `middle_end/flambda2/types/rocq/FlambdaTypingEnv*` (the typing
    environment, after `types/env/typing_env.ml`, `typing_env_level.ml`,
    `cached_level.ml` and `typing_env_extension.ml`; what satisfies one)
  - `middle_end/flambda2/rocq/requests/T2-join-call.md` (a request:
    the call of the join in the rewrites from the types)
- T3, aliases. Owner: T3, on `flambda2-rocq/T3-aliases`.
  - `middle_end/flambda2/types/rocq/FlambdaAliases*` (today's, which T3
    replaces; `follow_aliases` is used by the validator, so a change to
    it is agreed with V1)
  - `middle_end/flambda2/types/rocq/FlambdaEnvAliases*` (the aliases of
    the typing environment, after `types/env/aliases.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaTypingEnvExtension*` (the
    extensions of a typing environment, after
    `types/env/typing_env_extension.ml`; T3's by agreement with T2,
    though the names match T2's pattern)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyValueSlots*` (the
    projection of a value slot replaced by the `Simple` that a prover
    answers, a slice of the rewrites through types, by agreement with
    T2, who puts the rewrites together)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyStatics*` (the
    types that Simplify gives the symbols of a `Let_static`, after
    `simplify/simplify_static_const.ml`; the same agreement)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyCodeEnv*` (the
    environment in which the body of code is rewritten, with the
    types of the symbols, after `Typing_env.closure_env`; the same
    agreement)
  - `middle_end/flambda2/rocq/FlambdaValidateKnowing*` (the validation
    asked again with the new `follow_aliases` in front of each round;
    T3's by agreement with the owner of the validator, whose files it
    leaves as they are)
- T4, meet. Owner: T4, on `flambda2-rocq/T4-meet`.
  - `middle_end/flambda2/types/rocq/FlambdaExpandHead*` (expanded
    types and `expand_head`, after `types/expand_head.ml`, and what
    expanding keeps)
  - `middle_end/flambda2/types/rocq/FlambdaMeet*` (the environment of a
    meet, after `types/env/meet_env.ml`; the contract of meet; the meet,
    after `types/meet_and_n_way_join.ml` and `types/meet.ml`)
  - `middle_end/flambda2/rocq/requests/T4-meet.md`
- T5, the n-way join. Owner: T5, on `flambda2-rocq/T5-join`.
  - `middle_end/flambda2/types/rocq/FlambdaJoin*` (the n-way join of
    typing environments and of types, after `types/env/join_env.ml`,
    `types/join_levels.ml` and the join functions of
    `types/meet_and_n_way_join.ml`; what a join guarantees).
    `FlambdaJoinTypes*` are T5b's, on `flambda2-rocq/T5b-join-types`.
  - `middle_end/flambda2/types/rocq/FlambdaTypeProject.v`
    (`Type_grammar.project_variables_out`)
  - `middle_end/flambda2/rocq/requests/T5-join.md`
- T6, provers and reification. Owner: T6, on `flambda2-rocq/T6-provers`.
  - `middle_end/flambda2/rocq/requests/T6-provers.md` (the plan: the
    provers, the shape of their statements, the order)
  - `middle_end/flambda2/types/rocq/FlambdaProvers*` (the provers,
    after `types/provers.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaReify*` (reification, after
    `types/reify.ml`)
  - `middle_end/flambda2/rocq/requests/T6-unboxing.md`,
    `middle_end/flambda2/rocq/FlambdaUnboxParams*`,
    `middle_end/flambda2/rocq/FlambdaUnboxVariants*`,
    `middle_end/flambda2/rocq/FlambdaUnboxInScope*`,
    `middle_end/flambda2/rocq/FlambdaUnboxLoops*` and
    `middle_end/flambda2/rocq/FlambdaUnboxTogether*` (S6, the unboxing
    of continuation parameters on the validator's side, after
    `simplify/unboxing/`, its use in code and in loops with the
    static blocks in scope known, the parameters of a loop that
    is given boxes made in scope, and the boxed numbers that
    continuations pass to one another; not `FlambdaUnboxSpec.v` and
    `FlambdaUnboxProofs.v`, which are the validator's)
- T7, differential testing of types, and the converter between the
  compiler's types and the model's. Owner: T7, on
  `flambda2-rocq/T7-difftest`.
  - `middle_end/flambda2/tests/semantics/types/` (the converter between
    `Type_grammar.t` and the model's `flambda_type`, which is trusted;
    the generator of types; the differential tests and their dune
    rules). What it links of the model is extracted by
    `FlambdaExtract.v`, which is C0's, into
    `middle_end/flambda2/tests/semantics/extracted/`.

## S1 to S7: simplifier rewrites

Each is a pass with its proof in a file pair of its own, which its owner
creates and lists here, and the `.fl` examples that need it. V1 wires the
pass into the validator.

- S1, primitives through types. Owner of the first slice: T2, on
  `flambda2-rocq/T2-env`; the rest unassigned.
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyPrimitives*` (a
    primitive replaced by a constant, or the term by `Invalid`, on the
    answer of a prover about the types of its arguments; after
    `simplify/simplify_primitive.ml`, `simplify_unary_primitive.ml` and
    `simplify_binary_primitive.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyProjections*` (a
    load from a block or a projection from a closure replaced by the
    `Simple` that a prover answers, and the type of a block that is
    made; after `simplify/simplify_unary_primitive.ml` and
    `simplify_variadic_primitive.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyBoxing*` (the types
    of a number that is tagged or boxed, and the unboxing primitive
    replaced by the `Simple` that a prover answers for the number
    inside; after `simplify/simplify_unary_primitive.ml` and
    `simplify/unboxing/unboxing_epa.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyArith*` (the
    arithmetic of tagged immediates replaced by a constant, by one of
    its arguments or by `Invalid`, and the types of its result; after
    `simplify/simplify_binary_primitive.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyPass*` (one function
    on terms that carries a typing environment down a term and applies
    the rewrites of the files above, with the proof that its answer
    refines the term; for the validator to run)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyPrimitivesRenaming*`
    and `middle_end/flambda2/types/rocq/FlambdaSimplifyPassRenaming*`
    (that function and its parts commute with renaming)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyHandlers*` (the
    environment under which the handler of a continuation is
    rewritten, at the kinds that the handler gives its parameters;
    after `simplify/simplify_let_cont_expr.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyUses*` (the handler
    of a continuation rewritten under the environment of its one
    application, and under any environment that has the guarantees of
    the join of the environments of its applications; after
    `simplify/join_points.ml`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyPruning*` (a second
    function, which also removes the arms of a `Switch` that the type
    of the scrutinee excludes; its theorem is of the states of the run
    of a unit whose sets of closures are well formed; and that it
    commutes with renaming, `FlambdaSimplifyPruningRenaming*`)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyCarried*` (the
    first function without the test of the environment at every
    `Let`: it carries what is known of the environment, as the second
    does, and is proved equal to the first; the validation calls it)
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyArms*` (the arms of
    a `Switch`, each rewritten under what its arm knows)
- S2, switch. Owner of the first three slices: T1, on
  `flambda2-rocq/T1-grammar`; the rest unassigned.
  - `middle_end/flambda2/types/rocq/FlambdaSimplifySwitch*` (the arms
    of a `Switch` that the type of its scrutinee excludes are removed,
    those that are kept are rewritten in the environment that the
    meet answers for each, and arms that all apply one continuation
    are merged into that application, with arguments computed from
    the scrutinee; after `simplify/simplify_switch_expr.ml`)
  - `middle_end/flambda2/rocq/FlambdaMergeSwitches*` (that merging as
    a rewrite of a whole term, for the validator: the arguments that
    differ from arm to arm are the number of the arm tagged, or that
    negated, and the variable of the scrutinee is bound again)
  - `middle_end/flambda2/rocq/FlambdaMergeTables*` (the same where
    the argument is loaded from a static table that is in scope, and
    that it keeps scope; T1's)
- S3, continuations. Owner: unassigned.
- S4, calls. Owner of the first slice: T4, on
  `flambda2-rocq/T4-meet-rename`; the rest unassigned.
  - `middle_end/flambda2/types/rocq/FlambdaSimplifyApply*` (an indirect
    call of a callee whose type a prover shows to be a closure of known
    code is made the direct call of that code; after
    `simplify/simplify_apply_expr.ml`)
- S5, lets, CSE and lifting. Owner: unassigned.
  - `middle_end/flambda2/rocq/FlambdaLift*` (lifting a `Let` of a
    closure, a block or a boxed number to a static binding, for the
    validator; written on `flambda2-rocq/V3-lifting` by the stream that
    the roster calls V3, which is not the converter of the section V3
    below; S5's once it is staffed)
  - `middle_end/flambda2/rocq/FlambdaMyClosure*` (in the code of a
    closure that is a symbol, `my_closure` is the symbol; same stream)
  - `middle_end/flambda2/rocq/FlambdaUnusedSlots*` (the value slots of
    a static set of closures that nothing projects are removed; same
    stream)
  - `middle_end/flambda2/rocq/FlambdaKnownSymbols*` (a primitive or a
    projection applied to symbols that the unit binds to constants is
    replaced by its value; same stream)
  - `middle_end/flambda2/rocq/FlambdaKnownClosures*` (a value slot of a
    static closure whose binding is in sight is read, and a call of
    one is made direct; same stream)
  - `middle_end/flambda2/rocq/FlambdaFloatCode*` (the bindings of code
    moved up to the head of what they are in; same stream)
  - `middle_end/flambda2/rocq/FlambdaGroupsUp*` (a group of static
    bindings at the head of the body of a continuation binding bound
    before it; same stream)
  - `middle_end/flambda2/rocq/FlambdaHoistCode*` (all the code of a unit
    bound first, for a unit run on its own; same stream)
  - `middle_end/flambda2/rocq/FlambdaLiftFromCode*` (a closure or a block
    that mentions no variable lifted out of the code of a function; same
    stream)
  - `middle_end/flambda2/rocq/FlambdaLiftFromArms*` (a closure or a block
    that mentions no variable lifted out of the arm of a toplevel test,
    to the head of the unit; same stream)
  - `middle_end/flambda2/rocq/FlambdaHoistStatics*` (the static bindings
    that need nothing from around them bound first, for a unit run on
    its own; same stream)
  - `middle_end/flambda2/rocq/FlambdaSharedReads*` (in code, a second
    read, from symbols, of a variable that one group of static bindings
    put in two places of its constants is the variable of the first
    read; same stream)
  - `middle_end/flambda2/rocq/FlambdaSimulationProofs.v` (simulations in
    which one step is answered by one or more; same stream, and used by
    the loops of V4)
  - `middle_end/flambda2/rocq/FlambdaUnitFromProofs.v` (the two facts
    of `FlambdaWellFormedSpec.v` about the run of a well-formed unit,
    for a run from a static environment that holds no code and binds
    no name that the unit binds; same stream)
  - `middle_end/flambda2/rocq/status/V3.md`
- S6, unboxing of continuation parameters. Owner: unassigned.
- S7, the reaper. Out of scope for now, unassigned.

## V1: validator

Owner: unassigned. The only stream besides C0 that edits these.

- `middle_end/flambda2/rocq/FlambdaValidate*`
- `middle_end/flambda2/rocq/FlambdaCompare*`
- `middle_end/flambda2/rocq/FlambdaNormalise*`
- `middle_end/flambda2/rocq/FlambdaUnusedStatics*`
- `middle_end/flambda2/rocq/FlambdaUnusedHeadStatics*`
- `middle_end/flambda2/rocq/FlambdaSameComparison*`

## V2: corpus and harness

Owner: V2, on `flambda2-rocq/V2-examples`, for the corpus; the tool
stays C0's for now.

- `middle_end/flambda2/tests/tools/flsemantics.ml`
- `middle_end/flambda2/tests/semantics/*.fl` and their `.expected`
  files, but for an example that a stream adds for its own work, which
  is that stream's
- `middle_end/flambda2/tests/semantics/ocaml/` (units made from OCaml
  sources: the sources, the terms, the expected files, the script that
  makes a term and the dune rules)
- `middle_end/flambda2/rocq/FlambdaBlockLoad*` (a normalisation for the
  validator, written for the corpus: the load of a field of a block
  that is known; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaUnusedConts*` (another: removing a
  continuation that is not used; S3's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaUnboxSpec.v` and
  `middle_end/flambda2/rocq/FlambdaUnboxProofs.v` (another: unboxing a
  number that is known to be boxed, untagging one that is known to be
  tagged; S1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaCse*` (another: a primitive already
  computed is not computed again; S5's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaShareProjections*` (another: a
  value slot already projected is not projected again; S5's once it
  is staffed)
- `middle_end/flambda2/rocq/FlambdaRecoverComparison*` (another: a
  test against 0 of the result of a three-way comparison becomes the
  comparison of its operands; T5b's, on `flambda2-rocq/T5b-compare`;
  reviewer R5; S1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaStaticsOrder*` (another: the
  bindings of a group of static bindings put in the order in which
  another term has them; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaRenameApart*` (another: every binder
  of a term given a name of its own; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaAliasLets*` (another: in the term
  after, a `Let` of a variable to the variable that the `Let` around it
  has just bound is made one `Let` with it)
- `middle_end/flambda2/rocq/FlambdaInlineSiblings*` (another: in the
  body of code, the direct calls of the other code of its group are
  inlined, for the term before; a refinement)
- `middle_end/flambda2/rocq/FlambdaKnownSiblings*` (another: the same
  calls, where the callee is a symbol that the group binds to a closure
  of the code called, inlined as an equivalence, for the term after)
- `middle_end/flambda2/rocq/FlambdaUnusedProjections*` (another: in the
  term after, the `Let` of the projection of a function slot that
  nothing uses, from a closure that a `Let` around it has made, is
  removed)
- `middle_end/flambda2/rocq/requests/V2-missing-constructs.md`

## V3: converter

Owner: unassigned.

- the conversion from the compiler's terms to the model's. It is in
  `middle_end/flambda2/tests/tools/flsemantics.ml`, which is V2's. It
  is moved to a file of its own when V2 and V3 both have owners, and
  not before; from then until it has moved, a change to it is agreed
  between the two. Until V3 has an owner, the part of it that converts
  primitives and constants is C1's.

## P1: primitives that OCaml programs need

Owner: P1, on `flambda2-rocq/P1-prims` and the branches that follow
it. Leaf work in C1's files
(`FlambdaPrimitive*` and the conversion of primitives in the test
tool), delegated by the orchestrator, and:

- `middle_end/flambda2/rocq/FlambdaIdentities*` (a normalisation for
  the validator: the identities of integer arithmetic that Simplify
  applies; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaArrayLoad*` (another: the length
  of an immutable array that is known from the term, and a load from
  it at a constant index; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaValidateReading*` other than
  `FlambdaValidateReadingRename*` (the validation asked of the term
  before with its known arrays read; V1's once it is staffed)
- `middle_end/flambda2/rocq/FlambdaReadsAfter*` (two normalisations
  that are equivalences, for the term after Simplify: the read of a
  known block where the read is sure to have a value, and the removal
  of an allocation of constants that nothing uses; V1's once it is
  staffed)
- `middle_end/flambda2/rocq/FlambdaExtcall*` (what some C functions
  of the runtime compute, as a table that nothing uses; whoever owns
  calls of C functions once they have a meaning)
- `middle_end/flambda2/rocq/FlambdaNoArrayValue*` (what the immutable
  array as a value leaves as it was: the states that hold none stay
  so, and the relations of the heap and of the lifting relate, where
  none is held, what they related; and, in
  `FlambdaNoArrayValueObservers*`, the laws of what may follow a unit
  for the observers that make no array)
- `middle_end/flambda2/rocq/status/P1.md`
- `middle_end/flambda2/rocq/requests/P1-floats.md`,
  `requests/P1-c-calls.md`, `requests/P1-arrays.md`,
  `requests/P1-immutable-arrays.md`,
  `requests/P1-immutable-array-values.md` and
  `requests/P1-array-lifting.md`
- the section "Arrays" of `FlambdaPrimitiveSpec.v` and of its proofs
  (what the primitives of arrays compute on the elements of an
  array); the cases of arrays in `FlambdaHeapSpec.v` and
  `FlambdaHeapProofs.v` are the heap's
- `middle_end/flambda2/rocq/requests/P1-num-conv.py`,
  `requests/P1-int-shift.py`, `requests/P1-int-bswap.py`,
  `requests/P1-reinterpret.py` (the generators of the examples that
  compare conversions, shifts, swaps and reinterpretations with a
  reference) and `requests/P1-extcall.py` (the check of the table of
  C functions against the runtime)
- the examples of `middle_end/flambda2/tests/semantics/` for its
  primitives (`phys_equal*`, `num_conv*`, `boolean_not*`,
  `int_shift*`, `string_length*`, `int_not`, `int_xor_minus_one`,
  `string_load*`, `block_load_mut*`, `int_bswap`,
  `reinterpret_int63_int64`), with their `.expected` files

## V4: loops

Owner: V4, on `flambda2-rocq/V4-loops`: what the validator needs for
units with loops (`while`, `for`, tail-recursive functions).

- `middle_end/flambda2/rocq/FlambdaRecHandlers*` (applying a rewrite to
  the handlers of recursive continuations)
- `middle_end/flambda2/rocq/FlambdaContAlikeProofs.v` (a tool of proof:
  continuations that apply alike may stand for each other)
- `middle_end/flambda2/rocq/FlambdaSinkConts*` (moving a continuation to
  the handler that alone uses it)
- `middle_end/flambda2/rocq/FlambdaShortcut*` (an arm of a switch that
  applies a continuation which only applies another)
- `middle_end/flambda2/rocq/FlambdaSameHandlers*` (a continuation
  whose handler is that of a continuation bound around it applies that
  one)
- `middle_end/flambda2/rocq/FlambdaUnusedParams*` (removing the
  parameters of a continuation that its handler does not use)
- `middle_end/flambda2/rocq/FlambdaLoopAliases*` (following, inside a
  loop, the aliases and constants bound outside it)
- `middle_end/flambda2/rocq/FlambdaLoopify*` (loopify: the self tail
  calls of a function become jumps to a recursive continuation)
- `middle_end/flambda2/rocq/FlambdaLoopLets*` (removing, inside a loop
  too, the unused `Let`s whose defining expression has a value)
- `middle_end/flambda2/rocq/FlambdaParamAliases*` (binding, at the start
  of a handler, a parameter that every application gives the same
  thing)
- `middle_end/flambda2/rocq/FlambdaForward*` (an application of a
  continuation whose handler only applies another;
  `FlambdaForwardJumps*` is the case of no parameters, which is an
  equivalence, and `FlambdaForwardKinded*` that of arguments known to
  have the kinds of the parameters, an equivalence too;
  `FlambdaForwardArm*` is an application that gives a number to the
  parameter that the handler tests, which goes to the arm of that
  number)
- `middle_end/flambda2/rocq/FlambdaRoundsProofs.v` (a tool of proof:
  simulations by rounds of one or more steps of each run)
- `middle_end/flambda2/rocq/FlambdaInlineTail*` (that
  `inline_tail_conts` is an equivalence)
- `middle_end/flambda2/rocq/FlambdaDeadParams*` (removing the
  parameters of a loop that are only passed round it)
- `middle_end/flambda2/rocq/FlambdaCodeLets*` (removing, in the body
  of code, the unused `Let`s of its parameters)
- `middle_end/flambda2/rocq/FlambdaSinkUses*` (moving a continuation
  down past the bindings under it)

## N1: nominal loose ends

Owner: N1.

- `middle_end/flambda2/rocq/FlambdaNominal*` and
  `middle_end/flambda2/rocq/FlambdaAlpha*` (C0's until the step that
  retired one bisimulation)
- `middle_end/flambda2/rocq/FlambdaCompareAlpha*`
- `middle_end/flambda2/rocq/FlambdaBinders*` (the free names of a term,
  and the checks of rows 3, 5 and 6 of well-formedness; that they are
  conjuncts of `well_formed_unit` is C0's to write)
- `middle_end/flambda2/rocq/FlambdaScope*`
- `middle_end/flambda2/rocq/FlambdaFreeNames*`
- `middle_end/flambda2/rocq/FlambdaNameAbstraction*`
- `middle_end/flambda2/rocq/FlambdaEquivariance*`
- `middle_end/flambda2/rocq/FlambdaNominalInstances*`
- `middle_end/flambda2/rocq/FlambdaCanonicalFreeNames*`
- `middle_end/flambda2/rocq/FlambdaStateSupport*`
- `middle_end/flambda2/rocq/FlambdaHeapEquivariance*` (N1's, though the
  name matches the pattern of the heap's files: the renamings of
  variables and continuations on the machine with a heap)
- `middle_end/flambda2/rocq/FlambdaLocations*` (the locations of the
  heap as a sort of names: the action of their permutations, and the
  primitives and the machine under it; by agreement with the heap's
  owner)
- `middle_end/flambda2/rocq/FlambdaExitWorldActions*` (the renamings
  and the permutations of locations on what a run leaves at an exit;
  the files of the exit itself are not N1's)
- `middle_end/flambda2/rocq/FlambdaObserversActions*` (the renamings
  and the permutations of locations on what may follow a unit; the
  files of what may follow a unit are not N1's)
- `middle_end/flambda2/types/rocq/FlambdaEnvRenameJoinTypesProofs.v`
  (N1's, though it is in the types library and its name matches the
  pattern of N3's files on the join under renaming: lemmas only, the
  traversal of types of the join on renamed arguments, which N3's
  proofs of the join of environments import)
- `middle_end/flambda2/rocq/NAMES.md`

## N2: the freshness quantifier

Owner: N2.

- `middle_end/flambda2/rocq/FlambdaFreshQuantifier*`
- `middle_end/flambda2/rocq/FlambdaFreshAbstraction*`
- `middle_end/flambda2/rocq/FlambdaFreshBinders*`
- `middle_end/flambda2/rocq/FlambdaCompareConverse*` (N2's, though it
  matches the pattern of V1's section: the converse of the comparison,
  which N1 left to N2)
- `middle_end/flambda2/rocq/FlambdaAlphaFresh*` and
  `middle_end/flambda2/rocq/FlambdaAlphaGroup*` (N2's, though they
  match the pattern of N1's section: alpha-equivalence at each binder
  by the quantifier, N1 agreeing)
- `middle_end/flambda2/rocq/FlambdaInlineFresh*`
- `middle_end/flambda2/rocq/FlambdaContextWhere*` (N2's, though it
  matches the pattern of C0's section: rewriting in a handler or in the
  body of code from the states that have a predicate that a step
  keeps. Its lemmas are in C0's `FlambdaContextProofs.v`, which N2
  changed for it by C0's leave; that file stays C0's)
- `middle_end/flambda2/rocq/FlambdaGuardScope*` (N2's, though it is
  run by the validation: a rewrite made to keep scope by a test)
- `middle_end/flambda2/rocq/FlambdaKeepsScope*` (N2's, though it is
  about V1's and V2's rewrites: each keeps scope, which the statements
  of N3's `FlambdaValidatorEquivarianceSpec.v` about rewrites that make
  names ask of a rewrite)
- `middle_end/flambda2/rocq/FlambdaLoopsScope*` (N2's, though it is
  about V4's rewrites for loops: each keeps scope)
- `middle_end/flambda2/rocq/FlambdaScopedProofs.v` (N2's, though it
  matches the pattern of N1's section, N1 agreeing: lemmas only, about
  a term that keeps the scope of another, for the proofs that a rewrite
  of the validation keeps scope)
- `middle_end/flambda2/rocq/FlambdaValidateScope*` (N2's, though it is
  about the rounds of `FlambdaValidateSpec.v`: each keeps scope)
- `middle_end/flambda2/rocq/FlambdaLiftConts*` (N2's, though it is a
  normalisation written for the validation, which does not apply it:
  the continuations that a handler binds on its way to its `Switch`,
  bound outside the handler)
- `middle_end/flambda2/rocq/requests/N2-fresh-quantifier.md`

## N3: what commutes with renaming

Owner: N3, on `flambda2-rocq/N3-equivariance` for the validator and on
`flambda2-rocq/N3-types` for the types. N1 leads the work on names.

- `middle_end/flambda2/rocq/FlambdaValidatorEquivariance*` (each
  rewrite and each check of the validator commutes with renaming:
  what `NAMES.md`, section 7, owes for the files of V1 and V2 and for
  the checks of `FlambdaWellFormedSpec.v`)
- `middle_end/flambda2/rocq/FlambdaStaticsScope*` (the rewrites that
  the check of the validation applies to the term before, other than
  those of a round, keep scope: what the statement about
  `validate_unit` in the files above rests on, with
  `FlambdaKeepsScope*`)
- `middle_end/flambda2/rocq/FlambdaUnaliasParams*` (a parameter of a
  continuation that every application gives one variable is that
  variable: an equivalence, written for the validation to apply to
  either of its terms; and the lemmas for its laws of renaming and of
  scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaBindConstParams*` (a parameter of
  a continuation that every application gives one symbol or one
  constant is that symbol or that constant: an equivalence, written
  for the validation to apply to either of its terms; and the lemmas
  for its laws of renaming and of scope, which are stated in the
  files above)
- `middle_end/flambda2/rocq/FlambdaUnaliasRecParams*` (a parameter of
  a loop that every turn passes on, and that the loop is given one
  variable for, is that variable: an equivalence, written for the
  validation to apply to either of its terms; and the lemmas for its
  laws of renaming and of scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaWrapReturns*` (a parameter that
  its handler does not use, of a continuation that a call returns to,
  is removed, and the call returns to a wrapper, as Simplify makes
  it: a refinement, for the term before; and the lemmas for its laws
  of renaming and of scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaUntagParams*` (parameters of
  continuations that hold a tagged immediate and are only untagged or
  passed on, as a flag kept in a reference and set in a loop is, are
  untagged where they are given, for a set of positions at once, as
  Simplify has them: a refinement, for the term before; and the
  lemmas for its laws of renaming and of scope, which are stated in
  the files above)
- `middle_end/flambda2/rocq/FlambdaGiveScrutinee*` (an argument of a
  continuation that is the number of the arm that it is in, of a
  `Switch` on a variable, is made that variable where the parameter
  can then be taken away, as for the flag of a `while` loop: an
  equivalence, for both terms; and the lemmas for its laws of renaming
  and of scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaArmNumbers*` (the scrutinee of a
  `Switch` is made the number of the arm where the arm, and the
  handler that only that arm applies, read it, and the variable that
  it is the untagging of the tagged number, as Simplify has them for
  a flag that a loop tests before it sets it: an equivalence, for
  both terms; and the lemmas for its laws of renaming and of scope,
  which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaShadowParams*` (the parameters of
  continuations that hold a tagged immediate and are untagged are
  bound again to themselves at the head of their handlers, which
  gives the rewrite that makes names a binder to name: an
  equivalence, for the term before; and the lemmas for its laws of
  renaming and of scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaTwinParams*` (a parameter of a
  continuation that holds a tagged immediate, and that the head of
  its handler binds again, gets a twin that holds the number, as
  Simplify has it for a flag that a loop both tests and keeps: a
  refinement, for the term before; and the lemmas for its laws of
  renaming and of scope, which are stated in the files above)
- `middle_end/flambda2/rocq/FlambdaLaggingProofs.v` (a way of proving
  a refinement in which the first run takes steps that the second
  does not answer, with a number that bounds them; proofs only)
- `middle_end/flambda2/rocq/FlambdaValidateImportsRename*` (the tests
  by which a verdict of the validation holds of a unit run from its
  imports answer of renamed terms what they answer of the terms;
  the tests are those of `FlambdaValidateImportsSpec.v`, which stays
  its owner's)
- `middle_end/flambda2/rocq/FlambdaValidateRelatedRename*` (the
  second check of a unit, of `FlambdaValidateRelatedSpec.v`, answers
  of renamed units what it answers of the units, and its liftings
  made in turn keep scope and commute with renaming up to the names
  that the rounds make; the check and its files stay their owner's)
- `middle_end/flambda2/rocq/FlambdaCseEquivalent*` (the sharing of
  primitives of `FlambdaCseSpec.v` gives a term that is equivalent to
  the one it is given, which is what a rewrite of the term after
  Simplify owes; `FlambdaCseSpec.v` and its proofs stay V2's)
- `middle_end/flambda2/rocq/FlambdaValidateReadingRename*` (the
  validation asked of the term before with its known arrays read, of
  `FlambdaValidateReadingSpec.v`, answers of a renamed pair what it
  answers of the pair, and what it is made of keeps scope and commutes
  with renaming, up to the names made where the rounds make names; the
  functions and their files stay their owner's)
- `middle_end/flambda2/types/rocq/FlambdaEnvRename*` (the action of a
  renaming on a typing environment and on its levels, their free
  variables, and that the functions of `FlambdaTypingEnv.v` commute
  with it; the file of the environment and its statements stay T2's)
- `middle_end/flambda2/types/rocq/FlambdaJoinMeetRename*` (the join
  that Simplify calls and the meet of the model with its own join, on
  renamed arguments with the source of new names renamed; and an
  example in which the join with the one source of the model does not
  answer the renamed answer)
- `middle_end/flambda2/types/rocq/FlambdaTypeProjectRenameProofs.v`
  (lemmas only: the removal of variables from a type under renaming)

## R1, R2: reviewers

Owner: R1, on `flambda2-rocq/R1-review`. R2 is unassigned. They write
no proofs.

- `ROCQ-REVIEW-GUIDE.md`
- `rocq/TRUST-LEDGER.md` (Phase 0 wrote the first version; no stream
  may add to it without a reviewer's sign-off)

## The human

- `middle_end/flambda2/rocq/PARALLEL-PLAN.md`
- the decisions listed in its section 7 ("Decisions that are a human's")
