# T5b, the type traversal of the n-way join: status

## A leaf among the rewrites of the validation (2026-11-12)

Not part of the join. `FlambdaRecoverComparison*`, in
`middle_end/flambda2/rocq/`: `recover_known_comparisons` makes a
test against the tagged immediate 0 of the tagged result of a
three-way `Int_comp` the comparison of its two operands, as
`Simplify_binary_primitive.recover_comparison_primitive` does
(`compare a b < 0` becomes `a < b`), at every kind of integer and
with the compiler's table for the six comparisons. Four files: the
definitions and the statements (`FlambdaRecoverComparisonSpec.v`),
their proofs, the statement that the pass keeps scope, and its
proof.

- Proved: the result refines the term
  (`recover_known_comparisons_refines`); the pass commutes with
  renaming (`recover_comparisons_knowing_equivariant`,
  `recover_known_comparisons_equivariant`); it keeps scope
  (`recover_known_comparisons_keeps_scope`).
- The validation applies it, on the term before only: innermost of
  the rewrites that follow the values in `normalise_expr_before`,
  under `share_known_primitives` (the place agreed with the owner
  of `FlambdaValidateSpec.v`). The proofs of the round use the
  three theorems above.
- Measured before the proofs, with the definitions applied in that
  place in a build that is not in this tree: of 1356 units (the 370
  of the corpus, 840 generated ones, and 146 written to probe the
  validation, which are not in this tree), seven more are
  validated, each a unit that tests the result of `compare`
  against zero; none is lost, no output moves, and no expected
  file of the corpus changes.
- It reads what `FlambdaCseSpec.share_primitives_knowing` keeps
  (`computed`) and changes nothing there. It goes through every
  entry for a variable, so the entry that the untagging of a
  variable adds for that variable (`tagged_again`) hides nothing
  from it. One lemma of the scope proof
  (`computed_under_let_operands_within`) goes by the definition of
  `computed_under_let`, with a case for that entry, and needs a
  case more if the definition gains another.
- At the stage of the heap (2026-11-14: the machine runs a `Let` of
  a primitive on the heap, and a state has a record of the reads),
  `recover_comparisons_knowing_refines` is proved on that machine,
  in `FlambdaRecoverComparisonProofs.v`, with an invariant of its
  own, `comparisons_hold`, in the place of `all_computed` of
  `FlambdaCseSpec.v`, whose text that stage changed. Of what is
  known the pass reads the entries of two forms only, a three-way
  comparison and the tagging of a variable; neither reads a cell
  (`read_form_reads_no_cell`), so the invariant says nothing of the
  heap or of the record of the reads, and the lemmas that keep it
  (`comparisons_hold_forget`, `comparisons_hold_bind_params`,
  `comparisons_hold_extend`, `comparisons_hold_under_let`) are that
  file's own. No statement of this leaf's Spec files changed its
  text; one comment of `FlambdaRecoverComparisonSpec.v` was
  reworded.
- Reviewer: R5.

The series is merged into the integration branch (the type
traversal and the assembled conditional theorem; the reviewer's
scope is those, not the join of environments end to end). What
followed that merge is merged too. T4's two functions are
arguments of the traversal, and `FlambdaJoinMeet.v` gives them
("Decided overnight"). (My work on slices of the
meet, for T4, has a status file of its own,
`status/T5b-meet.md`.) Files, all in
`middle_end/flambda2/types/rocq/`:

- `FlambdaJoinTypes.v`: the join functions of
  `types/meet_and_n_way_join.ml` and their helpers. Definitions only.
- `FlambdaJoinTypesSpec.v`: what the type traversal is proved to
  meet, and what it assumes of the other streams' functions.
- `FlambdaJoinTypesProofs.v`: the proofs.
- `FlambdaJoinTypesKeepsSpec.v`, `FlambdaJoinTypesKeepsProofs.v`:
  what the join of types does to the state, with no premise on the
  state or the types.
- `FlambdaJoinTypesWithEnvSpec.v`, `FlambdaJoinTypesWithEnvProofs.v`:
  the join of types with the functions of the join of environments,
  which is what the join of environments is given for its rounds.
- `FlambdaJoinTypesExamplesSpec.v`,
  `FlambdaJoinTypesExamplesProofs.v`: what the transcription answers
  on small types, by computation.
- `FlambdaJoinWellFormedSpec.v`, `FlambdaJoinWellFormedProofs.v`:
  the types of the equations that the join of environments holds
  and adds are well formed.
- `FlambdaJoinEquationNamesSpec.v`,
  `FlambdaJoinEquationNamesProofs.v`: the names of the equations
  inside those types are of their kinds.
- `FlambdaJoinMeetFitsSpec.v`, `FlambdaJoinMeetFitsProofs.v`: so
  the types that the join put together with the meet adds fit the
  environment of the meet, which was a premise of T5's statements
  about that join: the join that the meet calls, and the join that
  Simplify calls.

The last three pairs are about T5's join of environments
(`FlambdaJoinEnv.v`) and stand on T5's files; T5 handed the two
open points over (the orchestrator's item of the night). They are
merged. What is about the join that
Simplify calls, and the join and the meet together, stands on that
series, on T5's later commits for that join and on T4's series of
what a meet keeps, and is merged too.

The interface is T5's `FlambdaJoinSpec.v`, section "The join of
types, inside a join of environments". I do not change that file.

## Done

- The transcription: the thirty join functions and their helpers
  (the tables below).
- `n_way_join_sound`: for every number that bounds its depth, the
  join of types meets `n_way_join_type_sound`, the contract that the
  join of environments assumes of it, under the six named premises
  of the Spec ("What is assumed" below).
- `n_way_join_bound_is_met`: its corollary with the witnesses of
  valid states: where the join answers a type, a value of a type
  given has it under some valuation of the target that extends the
  input's.
- `flat_heads_sound`: the eight conjuncts of the first slice.
- `n_way_join_keeps`: a reflexive and transitive relation on states
  that the three functions of the environment side keep is kept by
  the join of types, at every state and for every list of types.
  `n_way_join_keeps_definitions`: its instance at T5's
  `keeps_definitions`, with the three functions of `Join_env`; it
  is `join_of_types_keeps_definitions` (`FlambdaJoinEnvSpec.v`) of
  the model's join of types, which T5's statements about the rounds
  of the join of environments suppose of the join of types that
  they are given. T5 asked for it.
- `n_way_join_of_types_sound` (`FlambdaJoinTypesWithEnvSpec.v`):
  `n_way_join_of_types`, the traversal with the three functions of
  `Join_env` and T1's `type_eqb` and `block_shape_eqb`, meets
  `n_way_join_type_sound` at T2's `typing_env_invariant` and T5's
  `state_valid`, `state_kinded` and `kind_in_target_env`. Premises:
  mine of `expand_head` and of the reduction of naked immediates
  (T4's functions, still arguments), and those of T5's theorems
  (`fresh_avoids p`, `env_extension_eqb_sound p`). No premise about
  the constants of
  the environments joined, nor about the tests of equality.
  `n_way_join_of_types_keeps_definitions`: the same function keeps
  the definitions, with no premise. T5 asked for both.
- `n_way_join_of_types_true` (after the merge): the same contract at
  T5's `state_true source`, the invariant that says what the state
  means, for every `source`; one premise more,
  `parameters_type_eqb_sound p`. It is the hypothesis of T5's
  statements about the rounds (`first_phases_true`).
- `n_way_join_equations_kinded` (`FlambdaJoinTypesSpec.v`) and
  `n_way_join_of_types_equations_kinded`
  (`FlambdaJoinTypesWithEnvSpec.v`), after the merge: the join of
  types meets T5's contract for the names that have equations inside
  the type answered (`n_way_join_type_equations_kinded`): they are at
  the kinds of the types of their equations in the target, if those
  of the types given are in the environments of their indices and
  those of the types of the environments joined are. The first is
  for any state and any functions of the environment side that meet
  their two contracts for those names; the second is
  `n_way_join_of_types` with T5's functions, which T5 has proved to
  meet them. One premise more than the soundness has,
  `expand_head_equations_kinded` ("What is assumed"). T5 asked for
  it: the target of a join is to be an environment in which a meet
  may be made. How it is proved: "The shape of the proof".
- `n_way_join_well_formed` (`FlambdaJoinTypesSpec.v`) and
  `n_way_join_of_types_well_formed`
  (`FlambdaJoinTypesWithEnvSpec.v`), after the merge: the type that
  the join of types answers is well formed (T1's
  `type_well_formed`) if the types given are, the environments
  joined have well-formed types and the state is kinded
  (`n_way_join_type_well_formed`, a contract of mine in
  `FlambdaJoinTypesSpec.v`). The first is for any state and any
  functions of the environment side whose join of extensions
  answers a well-formed extension for well-formed ones; the second
  is `n_way_join_of_types`, whose join of extensions T5 has proved
  to answer an extension that it was given or the empty one. One
  premise more than the soundness has, `expand_head_well_formed`
  ("What is assumed"). T5 asked for it. No condition of the
  well-formedness of types was found to fail of an answer. How it
  is proved: "The shape of the proof".
- After the merge (`FlambdaJoinWellFormedSpec.v`): the types of
  the equations that
  the state of the join of environments holds are well formed
  through its first phases (`first_phases_well_formed`), and so are
  the types of the two extensions that its last phase adds
  (`last_phase_adds_well_formed`). The join of types keeps it with
  no premise (`n_way_join_keeps_well_formed`, by
  `n_way_join_keeps`), and recovering the relations of a
  well-formed type answers a well-formed type
  (`recover_inverse_relations_well_formed`). The premises beside
  those of T5's `first_phases_true`: the joined environments have
  well-formed types, the type of a constant is well formed, the
  join of types answers a well-formed type
  (`n_way_join_type_well_formed`). Of what the join is given in
  `FlambdaJoinMeet.v`: `join_parameters_well_formed`,
  `expand_head_for_join_well_formed` (by T4's theorem).
- The same for the names of the equations inside those types
  (`FlambdaJoinEquationNamesSpec.v`): they are of their kinds "in
  reach" through the first phases
  (`first_phases_equation_names`), in the target where every
  variable of an equation is in the target
  (`scoped_equation_names`), and so are those of the types that
  the last phase adds (`last_phase_adds_equation_names`). It
  follows T5's statements for the aliases
  (`FlambdaJoinKindsSpec.v`). The join of types keeps it
  (`n_way_join_keeps_equation_names`), and each name of an
  equation inside the type that `recover_inverse_relations` answers
  is one of the type given, or is reached through aliases between
  naked immediates from a variable that the type given has an alias
  to at that kind, so is a naked immediate
  (`recover_equation_names`; the statement says where a name may
  come from, not that one only is new).
- So (`FlambdaJoinMeetFitsSpec.v`) the names of the equations
  inside the types added are of their kinds in the environment of
  the meet in which the variables of the target are defined
  (`last_phase_equation_names_known`); the join that the meet
  calls adds types that fit otherwise
  (`model_join_adds_types_that_fit_otherwise`: T5's premise
  `join_adds_types_that_fit_otherwise`, for every choice, depth
  and fuel); and it has the conclusions of the meet's contract for
  its join that are not about values with no premise left about
  the join itself (`model_join_scoped_envs_keeps_outright`). The
  same of T5's conclusion about values on the runs that have its
  conditions (`model_join_scoped_envs_satisfied_outright`).
- The same of the join that Simplify calls (`model_join`), in a
  section of `FlambdaJoinMeetFitsSpec.v`, on T5's three
  statements about it (`model_join_keeps_statement`,
  `model_join_describes_uses_statement`,
  `model_join_with_admits_use_statement`, `FlambdaJoinMeetSpec.v`),
  which have the premise `types_fit_otherwise` for the run that
  they name. `model_join_run_adds_types_that_fit_otherwise`: the
  premise follows from theirs about the inputs and one more,
  `uses_fit_otherwise` (the types of the uses that are joined are
  well formed and have the names of their equations at their
  kinds). `model_join_keeps_outright`: a meet may be made in the
  target, the target keeps the source, and its new variables are
  as the contract says (four of the six parts of
  `join_guarantees`), with no premise about a run; it is T5's
  theorem with the run found from the answer of the function
  (`model_join_run_found`; where no use is cut there is no run,
  and T5's argument for that case is used as it is).
  `model_join_describes_uses_outright`: the same for what the
  analysis speaks of (`analysis_describes_uses`, a fifth part).
  What stays a premise of the two: that the meet of expanded
  types of the model has `meet_expanded_head_keeps`.
  `model_join_with_admits_use_outright`: T5's statement about
  valuations and the definitions of the analysis (the conclusion
  of `join_admits_use`) for that join made with any meet of
  expanded types that has the two contracts
  (`model_join_with_admits_use_statement`), word for word, with
  `uses_fit_otherwise` in the place of its premise about the
  types added. It is not stated at the meet of the model, which
  is not shown to have the contract about values and may not
  have it (T5's remark). The proof of the two joins is one lemma
  (`phases_add_types_that_fit`), given what the first phases
  suppose of the environments joined. T5 asked for it.
- The join and the meet of the model together (the section after
  that one). `model_meet_expanded_head_keeps`: the meet of
  expanded types of the model has `meet_expanded_head_keeps`, for
  every choice of the join, depth and fuel, and for choices of
  the meet that have the three conditions of the meet's contract
  (`repaired_choices` has them; `compiler_choices` has not). It
  is T4's `meet_expanded_head_fuel_keeps`
  (`FlambdaMeetKeepsSpec.v`: the meet keeps at every fuel, from a
  join that has the conclusions of its contract that are not
  about values) at `model_join_scoped_envs_keeps_outright`.
  `model_join_guarantees_on_inputs`: so the join that Simplify
  calls has five of the six parts of `join_guarantees` (a meet
  may be made in the target, which gives the first two; the
  target keeps the source; its new variables are as the contract
  says; the analysis speaks of the uses) under premises about its
  inputs and those three conditions only; nothing is supposed of
  a function of the model. T5 asked that it be stated here. The
  sixth part (`join_admits_use`) is had on the runs that have
  T5's conditions, of the join made with a meet that has both
  contracts.
- `model_meet_keeps_terms` (the same section): the meet of the
  model (`model_meet`), with such choices, adds no variable that
  may occur in terms (`meet_keeps_terms`,
  `FlambdaMeetTermsSpec.v`), which is what T1's statements about
  the arms of a `Switch` suppose of the meet that they are given.
  With it in `FlambdaMeetTermsSpec.v`:
  `meet_to_prove_keeps_terms_of_keeps`, the conclusion of that
  file's statement for a join that has `join_scoped_envs_keeps`
  (another hypothesis than the contract of the join: neither
  implies the other). T4 proposed it.
- Validation of these two sections, at the tip that was merged:
  `make -C rocq -j8 all`, `make -C rocq assumptions`.
- A variant of the join of types with a bound on its expansions
  (`n_way_join_bounded`, `FlambdaJoinTypes.v`; it is not the
  compiler's function, and nothing else in the model uses it).
  Where a call would expand an alias to join it with a type that
  is not an alias, a count is taken down; at 0 it answers the
  unknown type of the kind of the types, in the state given.
  `n_way_join_bounded_sound` (`FlambdaJoinTypesSpec.v`): the
  variant meets `n_way_join_type_sound` for every bound, under the
  premises of `n_way_join_sound`. That is a statement about this
  variant of the model, and the contract is the one about values
  that the join of environments supposes; the change proposed
  for the compiler ("Decided") is not transcribed, and that it
  computes what the variant computes is not checked. Not
  established of the variant: the two other contracts (the names
  of equations, the well-formedness of the type answered), for
  which proofs like those of `n_way_join` are expected; and that
  it ends: the first bound is still there for the definition to
  be accepted, no statement says that some value of it is
  enough, and what is in "Decided" on that is an argument, not a
  proof.
- `Expanded_type` is T4's module: `FlambdaJoinTypes.v` names it
  (`Module Expanded_type := FlambdaExpandHead.Expanded_type`) where
  it had a copy. Nothing else changed, in my files or in T5's: the
  two were the same definitions.
- Twenty-nine examples, each an equation proved by computation:
  unions of sets of numbers and of strings, constants and other
  aliases, unknown and bottom inputs, heads of two forms, blocks of
  one tag (fields joined, or kept where they are the same; two
  sizes; two shapes), of two tags (a variable for `Get_tag`), a
  constant constructor and a block (a variable for `Is_int`), two
  variants, closures (one code, two codes, one input with no
  function type: T1-4), and the answers that are `None` (no type,
  two kinds, the bound reached, a closure type of another kind,
  T5b-3). The environment side of the
  examples is a stub defined in the file, not the join of
  environments; the two tests of equality are T1's. No answer was
  read off a run of the compiler. T7 has since run them through the
  compiler's join of environments: no unexpected answer in the cases
  compared (the header of the file says how they were run, what the
  compiler does where the stub makes a variable, and which three
  were not run; T7 has since run the one of closure types of another
  kind, where the compiler stops at the fatal error that the
  transcription has as `None`). The example of the two mutable
  blocks was not compared
  as it is written (`Heap` and `Local`): `Heap` and `Heap_or_local`
  were compared in its place.
  T7 has since run the join of the model itself (`model_join`,
  extracted, with the compiler's choices) on the inputs of the
  examples beside the compiler's n-way join, both through the whole
  join of environments: of thirty-two inputs, twenty-nine give the
  same target (the bottom mark, the type and the canonical element
  of each name, the types of the variables made, up to their
  names); on the two where the compiler stops the model answers
  `None`; one was not run (mode `Local`). And five thousand
  generated joins over six shapes: all agree.
- Proved on the way, each stated with its proof in
  `FlambdaJoinTypesProofs.v`, in the form of `join_step_sound` or
  what it becomes for a function that takes several lists or may
  answer bottom: `n_way_join_generic_product`,
  `n_way_join_int_indexed_product`, `n_way_join_closures_entry`,
  `generic_merge_map_known`, `n_way_join_index`, `n_way_join_case`,
  `n_way_join_knowns`, `n_way_join_row_like_for_blocks`,
  `n_way_join_row_like_for_closures`, `n_way_join_variant`,
  `n_way_join_head_of_kind_value_non_null`,
  `n_way_join_head_of_kind_value`, `n_way_join_expanded_head`,
  `n_way_join_body`.
- Validation at the tip: `make -C rocq -j8 all`,
  `make -C rocq assumptions` (the statements of the Flambda 2
  libraries depend on no axiom). No OCaml or extracted file changes;
  the semantics tests were not run.
- Discrepancies T5b-1, T5b-2 and T5b-3.

## Reviews

Reviewer R4. Signed, and merged: the functions; the five slices of
proofs; the two statements of `FlambdaJoinTypesKeepsSpec.v`; the
definition and the first two statements of
`FlambdaJoinTypesWithEnvSpec.v`; the examples.

- R4 found a fatal error of the compiler that the transcription
  lacked ("Found so far"); two examples on inputs that the
  compiler's creation functions refuse (a record of floats at tag
  0; closure types that are not of kind value); and one on an input
  that no closure has (a known index without the function slot of
  its case: the raw creation function accepts it, and the
  well-formedness of types does not). Repaired before the merge.
- R4 asked that the premise `joined_envs_constants_of_kind` stay in
  the statements that are for any well-formedness of environments,
  and go only where the invariant gives that aliases are
  consistent. Done so.
- R4 asked where the kind of an expansion comes from, and that no
  condition on kinds be hidden in the well-formedness of
  environments: see "Decided overnight".
- Signed after the merge, and merged: `n_way_join_of_types_true`,
  and the paragraph of the header of the examples about the run.
- Signed after the merge, and merged:
  `n_way_join_equations_kinded_statement` with the premise
  `expand_head_equations_kinded`, and
  `n_way_join_of_types_equations_kinded_statement`.
- To be asked: the sentence of the header of the examples about the
  example of closure types of another kind, which T7 has run.
- Merged since: the two premises, the contract and the statement of
  `FlambdaJoinTypesSpec.v` for well-formedness
  (`expand_head_well_formed`,
  `n_way_join_env_extension_well_formed`,
  `joined_envs_types_well_formed`, `n_way_join_type_well_formed`,
  `n_way_join_well_formed_statement`), and
  `n_way_join_of_types_well_formed_statement`.

- Signed by R4, and merged: the three Specs of the last three
  pairs of files. No formal finding; one repair of prose asked and
  made (what `recover_equation_names_statement` says, against what
  the function does). With them, the change of `Expanded_type` (R4
  read that each operation that was copied is the one of T4's
  module) and the statement about values
  (`model_join_scoped_envs_satisfied_outright_statement`).
- Signed by R4, and merged: the last two sections of
  `FlambdaJoinMeetFitsSpec.v` (`uses_fit_otherwise` and four
  statements about the join that Simplify calls; three statements
  about the join and the meet together), and the statement added
  to `FlambdaMeetTermsSpec.v`. No formal finding; one repair of
  prose asked and made (`join_scoped_envs_keeps` is another
  hypothesis than the contract of the join, not a weaker one). R4
  checked that the premises can be had together: on an ordinary
  environment with a use that is not bottom, and the repaired
  choices of the meet, the join of the model answers and
  `model_join_guarantees_on_inputs` applies with no premise left.
  R4's limit, which the statements have: nothing about values is
  concluded of the meet of the model.

## Stand-ins carried

None. The copy of `Expanded_type` is gone (above).

`expand_head` and `reduce_head_of_kind_naked_immediate` are
parameters of the traversal and of `n_way_join_of_types`, with what
is assumed of each a named proposition of the Spec; T5's
`FlambdaJoinMeet.v` gives them T4's functions, and T5's statements
and mine there discharge the premises with T4's theorems ("Decided
overnight"). `type_eqb` and `block_shape_eqb` are parameters of the
traversal too; `n_way_join_of_types` gives it T1's. The fold over a
map that threads a state is C0's (`Keyed.filter_map_fold`); my copy
is gone.

## Agreed with other streams

- T5: the plan; `joined_env = None` is `None`; the first input is the
  one of least index; in B3 the test is `type_eqb other first`. A
  symbol answered by the join of simples at kind naked immediate
  cannot happen under the contract's premises
  (`n_way_join_simples_sound` says a symbol is answered only at kind
  value); the two branches of the compiler for it are transcribed as
  they are. The proofs need nothing of the state beyond the four
  contracts of `FlambdaJoinSpec.v`.
- T4: the statements I take as premises are T4's of `expand_head`
  (values kept, scope, kinds, the kind of the expansion) and of
  `reduce_head_of_kind_naked_immediate` (no value lost), which T4
  has proved under T1's law of the relations of constants after D26
  was amended. The kind of the expansion is T4's statement word for
  word: it needs `constants_of_kind` of the environment (T4-7). My
  `n_way_join_expanded_head` goes by the head, not by the kind.
- T2 and T5: `constants_of_kind` of the environments joined is to
  follow from the kinds premise of the contract, when T2's
  `typing_env_kinded` (the types kinded, the aliases of one kind,
  the symbols of kind value) replaces T5's under the same name; T5
  tells me when. Done: `n_way_join_of_types_sound` has no such
  premise. The statements that are for any well-formedness keep it
  (R4).
- T5: no second form of `n_way_join_keeps` is needed for the kinds
  of the variables of the state: `n_way_join_sound` gives
  `state_grows` at valid states, which is where T5 uses it.
- T4: its proofs of the contract of the meet are made general in
  what is supposed of a static environment, by a script, so that
  the meet keeps what it keeps from a join that has only the
  conclusions that are not about values. I agreed that T4 commit
  the scripted change to my three proof files of the meet
  (products, heads of kind value, row-like structures); no proof
  script changed, and each theorem is the general one at
  validity.
- T5 and T4: `join_scoped_envs_keeps` is defined twice, word for
  word (`FlambdaJoinMeetSpec.v`, `FlambdaMeetKeepsSpec.v`), until
  T5 deletes its copy; `model_meet_expanded_head_keeps` is proved
  by the conversion between the two.
- T1: `type_eqb` identifies types with the same values;
  `block_shape_eqb` decides equality. `type_satisfied_coincide` is
  what the proofs use where a type is imported as it stands (no
  monotonicity is needed). `block_shape_element_kind` is mine to
  define (T1 will take it).
- C0: the fold, as above; I prove what I need from its equation.
- The orchestrator: D26 amended on my report: `Is_null` relates
  poison to 0; only `Is_int` and `Get_tag` are never true of a
  scrutinee that holds poison.

## What is transcribed

The compiler is the integration branch's with the change of
`n_way_join_closures_entry` (a function type is kept only if every
entry has one). Lines are those of `types/meet_and_n_way_join.ml`
before that change, which moves nothing above line 2920.

Helpers:

| Function | Lines | In the model |
|----------|-------|--------------|
| `map_join_result`, `let>>+` | 49-51 | in full |
| `join_alloc_mode` | 768-773 | in full (`unknown ()` is `Heap_or_local`, as in the grammar) |
| `n_way_join_unknown` | 790-801 | in full |
| `n_way_join_head_of_kind_naked_number` | 803-805 | in full |
| `generic_merge_map_known` | 807-865 | in full, on the maps of the model, with the fold that threads the state |
| `deduce_is_int`, `deduce_is_int_simple` | 867-878 | in full |
| `deduce_get_tag`, `deduce_get_tag_simple` | 880-901 | in full |
| `n_way_join_relation_simples` | 903-915 | in full |
| `join_array_element_kinds` | 936-950 | left out: the grammar has no arrays |
| `reduce_head_of_kind_naked_immediate` | 952-1006 | T4's; a parameter here |

The thirty join functions:

| Function | Lines | In the model |
|----------|-------|--------------|
| `n_way_join` | 1885-1913 | in full; fuel is spent here |
| `n_way_join_expanded_head` | 1915-2106 | in full; the eleven cases of the naked numbers other than immediates are one |
| `n_way_join_head_of_kind_value` | 2108-2173 | in full |
| `n_way_join_head_of_kind_value_non_null` | 2175-2493 | `Variant`, `Mutable_block`, the nine boxed numbers as one, `Closures`, `String`; `Array` left out |
| `n_way_join_array_contents` | 2495-2548 | left out with arrays |
| `n_way_join_variant` | 2550-2586 | in full |
| `n_way_join_head_of_kind_naked_immediate` | 2588-2617 | in full |
| `n_way_join_head_of_kind_naked_float32` to `..._naked_mask` (eleven) | 2619-2661 | one function, the kind an argument |
| `n_way_join_head_of_kind_rec_info` | 2663-2666 | the head is `unit` in the grammar: always known |
| `n_way_join_head_of_kind_region` | 2668-2670 | in full |
| `n_way_join_row_like` | 2678-2804 | in full |
| `n_way_join_row_like_for_blocks` | 2806-2849 | in full |
| `n_way_join_row_like_for_closures` | 2851-2893 | in full |
| `n_way_join_closures_entry` | 2895-2953 | in full (the function types are there already) |
| `n_way_join_generic_product` | 2955-3008 | in full |
| `n_way_join_function_slot_indexed_product`, `n_way_join_value_slot_indexed_product` | 3010-3034 | in full; the first with the test of kinds of the creation function of its product |
| `n_way_join_int_indexed_product` | 3036-3096 | in full, with the test of `-flambda2-join-sharing structural` in place of `!=` |
| `n_way_join_function_type` | 3098-3131 | there already; no `rec_info` in the grammar |
| `n_way_join_env_extension` | 3133-3139 | in full: the environment's function, bottom made the empty extension |

A fatal error of the compiler, an `assert false` and an exception
that nothing catches are `None`: the join of zero types, the mismatch
of kinds that `n_way_join` tests, a head of another kind in
`n_way_join_expanded_head`, an environment that has no such index, an
empty set given to `create_naked_immediates_non_empty` (R1-16), a
symbol answered by the join of the `is_null` simples, the two fatal
errors of `n_way_join_row_like_for_closures`, a field index outside a
mixed shape, a joined closure type that is not of kind value (the
test of `Product.Function_slot_indexed.create`, which R4 found
missing from the transcription). No other `None` but fuel.

## The shape of the recursion

Every cycle among the thirty goes through `n_way_join`. So:

- every function but `n_way_join` is an ordinary definition, in a
  `Section` whose variables are the abstract state with
  `joined_env`, the three functions of the environment side
  (`n_way_join_simples`, `import_type`, `n_way_join_env_extension`),
  `expand_head` and `reduce_head_of_kind_naked_immediate`, `type_eqb`
  and `block_shape_eqb`, and `n_way_join_rec`, the join of types that
  the function calls on the parts of its arguments
  (`n_way_join_expanded_head` is also a recursion on its list);
- `n_way_join_body n_way_join_rec` is the compiler's `n_way_join`;
- `n_way_join fuel` is `fun _ _ => None` at 0 and
  `n_way_join_body (n_way_join fuel)` at a successor.

Fuel is needed, not only convenient: `Expand_head` goes from an alias
to a type of the environment, which is no part of the type joined, so
no measure on the types given decreases (T5b-1).

## The shape of the proof

`n_way_join_type_sound` of `n_way_join fuel`, for every fuel, by
induction on the fuel: at 0 the function answers nothing, and at a
successor it is the body applied to a function that is sound by the
induction hypothesis. So what is proved is: if `n_way_join_rec` meets
the contract, so does `n_way_join_body n_way_join_rec`
(`n_way_join_body_is_sound`).

One predicate says what it is for a function of the traversal to be
sound, for any type of argument and of answer (`join_step_sound`):
the invariant is kept, the state grows, the answer is in the target,
the answer holds under every valuation that extends an input's
wherever that input's argument holds, and it is kinded in the target
if the state is kinded. Each function has it with its own "holds"
(`type_satisfied`, `head_of_kind_value_satisfied`,
`row_like_for_blocks_satisfied`, ...), and the proofs compose by
three facts about the interface: `state_grows` is transitive,
`extends_input` at a later state gives it at an earlier one, and
scope and kindedness of arguments do not depend on the state beyond
`joined_env`.

The list of `Simple`s with kinds at which the things given and
the answers are kinded is a parameter of the proofs (`kinds_with`):
the aliases of a type (T1's `aliases_with_kinds`), to which two
things are added or not. The first is the names that have equations
inside the type (`equation_names_with_kinds`), each at the kind of
the type of its equation. The second is, for each test of the
well-formedness of types that the type fails, a `Simple` at a kind
of which it is not (`ill_formed`, a constant at a kind that is not
its own): a type is kinded at such a list, in an environment or in
the target, only if it passes every test, so it is kinded there
when its aliases are and it is well formed (`kinds_checked`). The
lists are defined by one recursion; with nothing added it is
`aliases_with_kinds` by computation. Every function that calls the
join of types is proved sound once, for any of the lists. What is
supposed of the three places where a type or an extension comes
from outside the traversal (the import of a type, the join of
extensions, the expansion of a head) is supposed at the list: with
nothing added it is in the contracts of the environment side and
in `expand_head_sound`; with the names, in those and in the two
contracts of the environment side for the names and
`expand_head_equations_kinded`; with the checks, in those and in
`n_way_join_env_extension_well_formed` and
`expand_head_well_formed` (a type imported is the type as it
stood). The contract for the names needs that the environments
joined have the names of the equations inside their types at their
kinds, and the well-formedness of the answer that they have
well-formed types; each is given to the proofs as part of the
well-formedness of environments, which is a parameter of the
interface.

With the checks, each function that builds a structure is shown to
build one that passes the tests about that structure, from the
tests that the structures given pass: the sizes and the shapes of
the cases of blocks (the fields answered are as many as the
shortest product given, against the least size); the slots of the
entries of closures against the indexes (a slot is kept by the
join of entries only if every entry has it, and the index answered
has the slots that every index has; an exact index is answered
only for exact indexes with the same slots); the maps that bind a
key once (the folds go through a list of distinct keys); the kind
of the closure types (the creation function tests it); the sets
that are not empty (a union, or the test of the creation function).
The types in the places that ask for a kind (the immediates of a
variant, the contents of a boxed number) are of it because the
join of types answers a type of the kind of the types given.

Where the compiler's own structure is generic, so is the proof: the
fold over a map that threads the state, `generic_merge_map_known`,
and the join of cases and of their indices are proved once, for any
index, shape and thing mapped to, and used for blocks and for
closures; the five sorts of head of `n_way_join_expanded_head` go by
one lemma.

Three things that the proofs rest on and that a reader may want to
check first:

- a key for which cases were gathered is never dropped by
  `generic_merge_map_known`, because the join of the cases of one key
  is not bottom (there is at least one case, and the join of the
  shapes of some cases is not bottom);
- the variable joined for `Get_tag` is given no `Simple` by a variant
  with no block, and is asked for the tag of no block of it;
- `n_way_join_variant` answers the state of before the extensions
  were joined when both joined extensions are empty: the answer then
  mentions nothing made since.

## What is assumed, and of whom

Hypotheses of the statements (no axiom, nothing admitted). Each is a
named proposition in `FlambdaJoinTypesSpec.v`. Those about the two
functions of the meet are discharged where the join is given T4's
functions (`FlambdaJoinMeet.v`), by T5's statements and mine there.

| Assumed | Where it stands |
|---------|-----------------|
| `environment_side_sound`: `n_way_join_simples_sound`, `import_type_sound`, `n_way_join_env_extension_sound`, `valid_states_have_witnesses` (`FlambdaJoinSpec.v`); the last is used by `n_way_join_bound_is_met` only | T5 has proved the four of the functions of `Join_env`, under `fresh_avoids p` and `env_extension_eqb_sound p`; those two are the premises of `n_way_join_of_types_sound` in its place |
| `expand_head_sound`: a value of a type has the expanded type under a valuation that satisfies the environment strictly; the expansion is in scope and kinded where the type and the environment are; it is of the kind of the type where the type is kinded and the environment has `constants_of_kind` | T4; a premise of `n_way_join_of_types_sound`, discharged for T4's function in `FlambdaJoinMeetProofs.v` (`expand_head_for_join_sound`) |
| `expand_head_equations_kinded`: in an environment that has the names of the equations inside its types at their kinds, a type that has its own at their kinds expands to one that has | T4 (an expanded type is the type, an unknown type, the type of a constant or a type of the environment); a premise of the two statements about the names of equations only, discharged for T4's function in `FlambdaJoinMeetFitsProofs.v` (`expand_head_for_join_equations_kinded`) |
| `expand_head_well_formed`: in an environment whose types are well formed, a type that is well formed expands to one that is | T4 has proved it of its function, as it is written here; a premise of the two statements about well-formedness only, discharged for T4's function in `FlambdaJoinWellFormedProofs.v` (`expand_head_for_join_well_formed`) |
| `n_way_join_env_extension_well_formed`: the extension that the join of extensions answers is well formed if those given are | discharged in `n_way_join_of_types_well_formed` with T5's `n_way_join_env_extension_answers_given` (the function answers an extension that it was given, or the empty one) |
| `import_type_equations_kinded`, `n_way_join_env_extension_equations_kinded` (`FlambdaJoinSpec.v`) | T5 has proved both of the functions of `Join_env`, under `fresh_avoids p`; premises of `n_way_join_equations_kinded`, discharged in `n_way_join_of_types_equations_kinded` |
| `joined_envs_constants_of_kind`: in each environment joined, a constant that is the canonical element of a `Simple` of a kind is of that kind | discharged in `n_way_join_of_types_sound`, from the contract's kinds premise and the invariant of typing environments (T2's `typing_env_kinded_constants_of_kind`) |
| `reduce_head_of_kind_naked_immediate_sound`: a number that has a head, under a valuation that satisfies the environment strictly, has the reduced head, which is then not bottom | T4, who has proved it with no premise (`FlambdaMeetStepsProofs.v`); a premise of `n_way_join_of_types_sound`, discharged with that theorem where the join is given T4's function |
| `type_eqb_sound`: types that the test identifies have the same values | discharged with T1's `type_eqb` (`type_equal_satisfied`) |
| `block_shape_eqb_sound`: shapes that the test identifies are one shape | discharged with T1's `block_shape_eqb` |

## Decided overnight

- Where the kind of what `Expand_head` answers comes from. An alias
  whose canonical element is a constant expands to the type of the
  constant, whatever the kind of the alias (T4-7), so the join needs
  `constants_of_kind` of each environment joined to answer a type of
  the kind of the types given. The contract of the join of types has
  no premise that gives it. Options: (a) a condition of structure in
  the well-formedness of environments (the type of a name that is
  not canonical is the alias type of its canonical element): the
  compiler's environments do not have it (T2; T4-10); (b) a premise
  of my statements, about kinds only, beside the contract's
  `joined_envs_kinded`; (c) the contract's kinds premise asks the
  aliases of each use to be of one kind (`aliases_kinded`), from
  which it follows. Chosen: (b) now, as `joined_envs_constants_of_kind`
  with a `CR mshinwell` at its definition, and (c) asked of T5 and
  T2, who have done it. Why: kinds stay explicit premises and out of
  well-formedness (D10, D12; R4's reading). Where it stands: the
  statements for any well-formedness of environments keep (b); the
  statement at the invariant of typing environments
  (`n_way_join_of_types_sound`) has it from (c) and asks nothing
  more. What stays for the human: (c) makes "the aliases of each
  environment joined are of one kind" a premise of the contract;
  whether ill-kinded code that the compiler accepts can break it is
  not known (T5b-3).

- What is assumed of `Expand_head` for the names of the equations.
  The contract for those names needs that a type whose equations
  are kinded expands to one whose equations are. Options: (a) wait
  for T4 to state it of its function; (b) a named premise of my two
  statements, `expand_head_equations_kinded`, in the form of the
  other premises about `expand_head`. Chosen: (b). Why: the
  statement did not wait, and the premise is what T4 would prove.
  Where it stands: T4 has since proved it of its function. Cost to
  change: none; the premise goes when the traversal takes T4's
  function.

- The well-formedness of the type answered: three choices.
  (1) The state is asked to be kinded
  (`n_way_join_type_well_formed`). The proof rides on the kinds
  clause of the contract of the join of types, which is under that
  premise; whether well-formedness needs it is not known. Options:
  a proof of its own through every function (as long again as the
  soundness), or the premise. Chosen: the premise; the join of
  environments has it wherever it has the kinds of an answer. Cost
  to change: a second traversal of the proofs.
  (2) What is asked of the join of extensions is that it answers a
  well-formed extension for well-formed ones
  (`n_way_join_env_extension_well_formed`), not that it answers one
  of those given: it is what the proof uses, and T5's statement
  about its function gives it.
  (3) What is asked of `Expand_head` is a new named premise,
  `expand_head_well_formed`, with the well-formedness of the types
  of the environment as its only premise on the environment. T4
  has since proved it of its function as it is written.

- What the examples run on. Options: (a) the join of environments
  itself, on typing environments built for each example: the
  answers would show the types given to the variables made, and the
  terms would be large; (b) a stub of the environment side, defined
  in the examples file: a counter for the state, empty
  environments, the join of `Simple`s answering the `Simple` if all
  are the same and a variable made otherwise. Chosen: (b), with a
  header that says so and says that no answer was read off a run of
  the compiler. Why: the examples are there to show what the
  traversal computes, and the orchestrator asked for a small file.
  Cost to change: the equations are stated again over (a); nothing
  else depends on them.

- A symbol in the target of a join (2026-11-12). T5 needed, of
  every alias to a symbol inside a type that the join adds, that
  the target knows the symbol; the contract of the join of types
  said of a symbol only that it stands at kind value
  (`simple_of_kind_in_target`, `FlambdaJoinSpec.v`). Options: (a)
  a contract beside the main one, as for the names of equations;
  (b) the definition says more of a symbol: of kind value, and
  known to an environment joined. Chosen: (b), on my advice, and
  T5 has made the change (the definition is T5's and has a dated
  `CR mshinwell`; my two files were edited by T5 with my
  agreement: one argument more to the reading, one case more in
  the lemma that carries it to a later state). Why: (a) would need
  a second traversal of the thirty functions, because my theorem
  about them is general in the list read off a type and not in
  the two predicates; with (b) nothing of the traversal changes,
  since it never looks inside the case of a symbol. Cost to
  change: T5's chain through the phases, and the argument in my
  files. It is a change of meaning of a signed definition, for
  the reviewer of the join to read.

- The two functions of the meet stay arguments of the join of
  types (2026-11-12). Options: (a) `n_way_join_of_types` takes
  T4's `expand_head` and `reduce_head_of_kind_naked_immediate`
  itself: two arguments fewer, and the premises about them gone
  from its five statements, which are signed; (b) they stay
  arguments, and the file that puts the join and the meet together
  (T5's `FlambdaJoinMeet.v`) gives them, as it does. Chosen: (b).
  Why: that file exists and its statements discharge the premises;
  the statement that the join commutes with a renaming of
  variables (N3 has begun it) is simpler of a traversal that takes
  those functions as arguments; (a) would change signed statements
  for no new fact. What was done: the copy of `Expanded_type` went,
  by a module that names T4's, so T5's conversion between the two
  (`join_expanded`) converts a type to itself and can go when T5
  wishes. Cost to change to (a): mechanical.

- What is supposed of the uses of the join that Simplify calls
  (2026-11-12). The types that its last phase adds fit the
  environment of the meet only if the types of the environments
  joined are well formed and have the names of their equations at
  their kinds; the contract of the join supposes neither (nor
  that a meet may be made in the source, which T5's statements
  suppose). Options: (a) a premise that a meet may be made in each
  use (`typing_env_fits`); (b) a premise of those two things only,
  of the uses that are not bottom, which are the ones joined
  (`uses_fit_otherwise`); (c) wait until the contract says which
  environments Simplify gives the join. Chosen: (b), with a
  `CR mshinwell` at the definition. Why: it is the least that the
  proof uses, and with what the contract and `uses_kinded` give
  it is (a) for those uses. What stays for the human: whether
  every environment that Simplify gives the join has it is not
  known. Cost to change: the premise of five statements.

- Where the last three pairs of files stand. They are about T5's
  functions and are in files of mine, by agreement with T5 and the
  orchestrator; T5 owns the functions and the statements that
  mine discharge the premise of. The last statement
  (`model_join_scoped_envs_keeps_outright`) is T5's
  `model_join_scoped_envs_keeps` with its premise discharged; it
  is in my file because T5's proofs file cannot import mine
  without a cycle. T5 may move it.

## Decided

- The repair proposed for T5b-1 (2026-11-12; a candidate change of
  the compiler, not in this tree, for the human to judge). The
  question: how the two joins of types are to end where an alias
  is joined with a type that is not one. Options: (a) a bound on
  the depth of such expansions; (b) a test for a cycle (the names
  expanded on the way are remembered, and one met again stops the
  join); (c) both. Chosen: (a), with a count of its own in each
  join and the bound that the joins have already (the join depth,
  5 unless a flag says otherwise). In the binary join the count
  is beside the depth that is counted where both types are
  aliases; in the n-way join it is in the join environment, taken
  up before the heads are joined and down after, beside the
  rounds. At the bound the join answers the unknown type, as it
  does today at the bound for two aliases. Where all the types
  are aliases, or none is, nothing changes. Why (a): it is the
  smaller change; in the model, the variant with such a bound
  keeps the contract about values (`n_way_join_bounded_sound`);
  the join is expected to end with it (argued, not proved: a call
  either takes a count down or is on types that are parts of the
  types that it was given); and (b) alone does not bound the
  work, since a way through the names may go through each of them
  once. What it can lose: where a join goes today, and ends,
  through more such expansions than the bound, one inside
  another, the part below the bound becomes unknown. Types that
  are not aliases, one inside another, are needed on one side for
  that. Cost to change to (c): a set of names in each join
  environment and one test.
  Why a count of its own, in the binary join too: the first
  candidate counted these expansions in the depth that the binary
  join has for two aliases. A measurement on generated programs
  by the stream that reproduces compiler issues (not in this
  tree) found a program that does not overflow and loses a direct
  call with it: the way to a closure went through five joins of
  two aliases, which is the whole depth, and one join of an alias
  with a type that is not one. With a count of its own that
  program compiles as it does today. The variant of the model is
  of the n-way join, whose count was its own from the start.

## Found so far

- T5b-1: `n_way_join` has no bound on its depth and no test for a
  cycle (the binary join has one); on two inputs that are an alias
  and a type in turn it does not terminate. Shown by T7 with a test
  of the types library on two such environments (the stack
  overflows, in the n-way and in the binary join), and since from
  OCaml source for both joins: the compiler's stack overflows at
  `-O3` on one file with the default join, and with the n-way join
  on a unit compiled against one that the default join compiled.
  A repair is proposed ("Decided").
- T5b-2: the join gives an `is_null` variable to a join one of whose
  inputs is `Not_null`, which says `Is_null` is 0 of a poison value
  of that input. An unchecked assumption; it is why D26 was amended.
- T5b-3: the join trusts the kinds of what `Expand_head` answers.
  An alias of one kind whose canonical element is a constant of
  another expands to a head of the constant's kind, and
  `n_way_join_expanded_head` then meets its `assert false`, or
  answers a type of another kind than the types given. An unchecked
  assumption (that aliases are of one kind); the example
  `example_join_alias_of_another_kind` shows the first on the stub.
  T7 ran the test in the compiler: `Meet_env.add_equation` accepts
  the alias, and the n-way join stops at the `assert false`.
- A mistake in the transcription of the join of environments,
  found by the well-formedness of what it stores:
  `Join_env.add_or_replace_equation` put a second equation on a
  name where the compiler's (`Name.Map.add`) replaces the one that
  the name had. The type that `recover_inverse_relations` answers
  for a variant whose extension for blocks already had an equation
  on the variable of its tag was then not well formed (an
  extension binds a name once). Reported to T5, who has repaired
  the function; no statement of T5's changed.
- Met again here: R1-16 (the empty set given to
  `create_naked_immediates_non_empty`: `None` in the model), X3-5
  (the joined relations are dropped when the join of the variant is
  unknown: transcribed as it is), T4-7 (the kind of an expansion:
  "Decided overnight").
- A mistake of mine that the reviewer found: the creation function of
  a product of function slots stops on a component that is not of
  kind value, and the transcription did not. Fixed; every creation
  function that the join functions call was then read for such a
  test, and that one is the only one (the empty set of R1-16 was
  there already).
- A mistake of mine that the proof found: a pattern in
  `n_way_join_head_of_kind_naked_immediate` that kept the tail of
  the set of numbers. Fixed before the first request.

## What is left

- Arrays, when the grammar has them (`Array`,
  `n_way_join_array_contents`, `join_array_element_kinds`).
- Code age and `rec_info`, with the function types.

- The conclusion about values of the meet's contract for its join
  (`join_scoped_envs_sound`) for the model's join: T5's, with the
  meet's own statements; the conclusions that are not about values
  are had.
- The half about values of the meet of the model
  (`meet_expanded_head_sound` of `model_meet_expanded_head`): not
  shown, and it may not hold, since the join that it calls has
  the conclusion about values only on some runs (the item
  before). So the sixth part of `join_guarantees` is stated of
  the join made with a meet that has both contracts, not with
  the model's. I was asked to tie the two for that half as for
  the other, once the join has the value clause of its contract
  with no condition on runs. T5 has since found, by computing an
  instance, that the join of the model has not that clause with
  either of its choices (a variable brought in from one
  environment joined with a type that no value has on the run),
  and T4 that no premise that the meet can carry gives it. So
  there is nothing to tie for that half with the join as it is,
  and I have written nothing for it.
- The same for the choices of the compiler (`compiler_choices`):
  T4's theorems are for the choices that have the three
  conditions.

## Asked of me next

Nothing that is not under "What is left".

## Open questions

None for the human.
