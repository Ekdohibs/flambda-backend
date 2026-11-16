# T5: the contract of the n-way join, and how to divide the work

Branch `flambda2-rocq/T5-join`. For the reviewer, the orchestrator
and the human. The contract is `rocq/types/join/FlambdaJoinSpec.v`; this
note explains it, says what is not settled, and proposes how to split
the transcription. Paths are from `middle_end/flambda2/`.

## 1. The contract in one page

The join is given a **source** environment (in Simplify: the handler's
environment before anything is known of the uses, parameters defined;
in the meet: the environment of the meet), a scope, and a list of
**use** environments with the caller's identifiers. It answers the
**target** (the source with one more level) and an **analysis**.

`join_guarantees`, for inputs that are well formed and kinded:

1. **Upper bound under one extension** (`join_admits_use`). For each
   use and each valuation that satisfies that use and the source
   strictly, some valuation satisfies the target strictly, gives every
   variable that the use or the source knows exactly what it had (the
   same value, or none in both), and gives every variable that has a
   definition in the analysis the value of that definition at that
   use, where it has one. One valuation for all the equations of the
   target. The agreement is exact so that the witness satisfies the
   source of the next join or meet (agreed with T4, whose meet has the
   same agreement): a caller can then supply "the valuation satisfies
   the source and the use" from one step to the next. For it the
   inputs must have no name of type bottom (`no_bottom_types`, below).
2. **The target is the source extended** (`join_keeps_source`): same
   earlier levels, scope, symbols; every name keeps its binding time
   and name mode.
3. **Names** (`join_introduces`): a variable the target knows and the
   source does not is in types only, and is either known to a use that
   is not bottom (imported under its own name) or outside the list of
   names to avoid (made by the join). Membership only; which name is
   made is not specified (agreed with N1: the join takes `avoid`, and
   a `fresh` function is an argument of the implementation).
4. **The analysis** (`analysis_describes_uses`): its identifiers are
   those of uses that are not bottom; a definition is of a variable of
   the target of the kind given, by a `Simple` of that kind at the
   use; one at the normal name mode may occur in terms at that use.
5. The target is well formed (the typing environment's invariant, a
   parameter until that file names it).

Not in `join_guarantees`: that the target is kinded. It needs more of
the inputs than the upper bound does, so it is a statement of its own,
`join_keeps_kinded`: if the source is kinded too and the uses have the
names of the source at the source's kinds
(`uses_agree_with_source_on_kinds`), the target is kinded. With it a
later join gets `uses_kinded`, and a meet the kinds it asks (T4 needs
this for a meet that follows a disjunction). To be proved of the
transcription.

**The compiler's join and names with no value (T5-10).** The contract
is strict: `join_admits_use` speaks of every valuation that satisfies
the use and the source strictly, and one may give no value to a
variable the use knows (the target of an n-way join is satisfied on
some runs only by such a valuation: the variables imported from the
other uses, T5-5). That is what the meet assumes of its join. The
compiler's join does not have it: it reads "the environment has a
type for `x`" as "`x` has a value". So what is to be proved of the
transcription is `sound_join_where_values_are_read`: the same
contract for the runs that satisfy three named conditions:

- `scrutinees_have_values`: for every relation the target has on a
  variable (put back by `recover_inverse_relations` whenever every
  joined environment has a type for the scrutinee), the scrutinee has
  a value in every run in which the result has one. Nothing in the
  compiler establishes it; a `CR mshinwell` is there.
- `joined_variables_have_values`: a variable of the source that every
  live use has an equation on has a value in the run. This is what
  "an equation is added on a name where the name has a value" gives a
  caller, and the meet can keep it as an invariant of a scope.
- `imported_variables_have_values`: a variable that the use knows and
  the join brings into the target has a value in the run.

Without the last two the type joiner may leave out the use at which
such a variable has no value (it passes over a type that it finds to
have no value: one that is bottom, or a head of naked immediates that
its relations reduce to nothing; R1's case, through T4), and the
target then asks of the variable what the other uses say (a type with
one constant for only value makes the variable an alias of it), or is
bottom, on a run that exists.

A third thing the compiler's join needs is in the contract itself, as
a condition on the inputs of every join and a guarantee of its target
(`no_bottom_types`, with a `CR mshinwell`: the model is stricter than
the compiler here): no name has the type bottom in an environment
that is not bottom. The type joiner drops a bottom type; without the
condition the target could force a value on a name that has none at a
use, and the exact agreement above would fail.

A join that has the strict contract has this one
(`sound_join_where_values_are_read_of_sound`, proved). That the
transcription lacks the strict contract without the first condition
is proved on an instance (`transcribed_join_not_strict_statement` in
`FlambdaJoinEnvSpec.v`, section 5 below). And it is no longer only
the model's concern: X2 has a program that the compiler, with the
n-way join and `-O3`, compiles wrongly because of it (the arm of a
switch that the run takes is deleted).
A weaker conclusion (target satisfied but for the results of such
relations) was rejected with T4 and the reviewer: the meet that
follows reduces the scrutinee's type from such a result and can
answer bottom on a run that exists.

In the machine (`join_guarantees_in_machine_statement` and
`join_admits_use_in_machine_statement`, proved): if a variable that
may occur in terms in the source may occur in terms at each use, a
state that satisfies a use satisfies the target, and the witness gives
a variable with a normal-mode definition the value the machine has for
that `Simple`. That is what entitles `join_points.ml` to add a
parameter for such a variable and pass the `Simple` at each use.

Inside the join, the functions on types are specified against a
reading of the join's state (`n_way_join_type_sound`,
`n_way_join_simples_sound`): a value of an input's type has the joined
type under any valuation of the target that extends that input's.

Proved now (sixteen statements, no axioms; eleven here and five about
the instance of section 5): the three in-the-machine
corollaries; what a constant in the analysis means, and that the two
lists Simplify folds over hold what the vectors have at their indices;
that the contract is met by the join that answers the source and is
not met by one that answers bottom (so it is neither contradictory nor
empty); and the two about closures below (the function as it was is
not an upper bound; as changed it is).

**What it does not yet cover of Simplify's joins.** The contract is a
foundation for runs in which one valuation satisfies a use and the
source strictly, through environments that are kinded. Outside it:

- a use made by variant unboxing with a poison argument (D26, T5-7;
  the unboxing workstream owes what holds at a join after an unboxed
  continuation);
- a source that holds the types of constants lifted in other branches
  (T5-4; the continuations workstream owes that the valuation of a use
  satisfies the source, once T2 has said where the type of a symbol
  may be used);
- a use environment that is not kinded: an alias at a kind that is not
  the kind the environment has for its target (T5-9). Ill-kinded code
  may make these; the compiler's join does not test for them when it
  makes a variable of a kind for simples, and with the depth limit it
  can answer without ever testing. `uses_kinded` is a premise, not an
  invariant claimed of the compiler's inputs.

## 2. What Simplify may read of the analysis

- `fold_variables_created_at_join` (`join_points.ml`): for each
  variable with a normal-mode definition, the `Simple` at each use. It
  may pass that `Simple` as an extra argument at that use for a new
  parameter that it declares equal to the variable. Despite the name,
  such a variable can be one of the source (a parameter chosen as the
  representative of an existential): the caller already skips those
  that have a normal-mode canonical.
- `simple_refined_at_join` with `fold_values_at_uses`
  (`simplify_switch_expr.ml`, `specialization_cost.ml`): `Known c` at
  a use means the variable is `c` there
  (`known_value_at_use_statement`). `Unknown` and
  `Not_refined_at_join` mean nothing (the source has a comment saying
  the latter is not accurate). Both callers use it for a decision
  (specialise or not, cost), which is out of scope; no rewrite rests
  on it.

## 3. Decisions in the contract, and what is not promised

- **Partial valuations, strict satisfaction.** Found while agreeing
  the interface with T4 and T2. The n-way join imports a variable that
  only some uses define, with the join of its types over those uses,
  and makes existentials for vectors that cover only some uses. At the
  other uses no value need have that type. The binary join gave such
  names bottom; the n-way one leaves them, reachable only through
  cases that are dead at those uses. So a valuation may leave a name
  without a value; a type that reaches it by an alias then has no
  value. The typing environment's Spec now has this
  (`typing_env_satisfied_strictly`, T2); T4 states the meet the same
  way.
- **Poison (D26).** The contract needs strict satisfaction of the
  use: a name that holds poison must have a type poison has. With the
  exemption of names that hold poison the statement is false, for the
  join and for anything that expands an alias (example in the comment
  at `join_admits_use`). So nothing is promised of a use made by
  variant unboxing with poison arguments. A relation (`Is_int`,
  `Get_tag`, `Is_null`) is never true of a scrutinee that holds poison
  (D26 as revised at 18:30 UTC). T1 is removing the poison clauses of
  `relation_holds`; until that reaches this branch the definition it
  builds on still has them, and nothing proved here depends on which.
  Aliases and definitions are exact.
- **Not promised:** leastness; equality with the binary join; that
  the join answers (no fatal error, enough fuel); anything about code
  age, allocation modes, symbol projections.
- **Precision promised:** one case, `join_keeps_constants` (a source
  variable with an equation to the same constant in every live use's
  level is that constant in the target). Stated, not yet proved (it
  needs the functions). Aliases between source variables are a
  candidate second case, under `binding_times_coherent`; not promised.
- **Physical equality (D30).** Not in the model. Three places in my
  files change the result with sharing. The human has chosen
  `structural`: the model tests an equality of types there (T1's
  `type_eqb`, a parameter until it exists), as the compiler does under
  `-flambda2-join-sharing structural` (X3). The upper bound holds of
  either branch at each place. What the test gives in precision (an
  extension, or a field, equal in every input is kept as it is) will
  be stated separately, with the functions.

## 4. The hypotheses of the header of `join_env.ml`

- **"Any name defined in the source environment is also defined in
  all the joined environments."** Checked by the compiler: no.
  `check_join_inputs` says it is false (in-types variables made for
  lifted constants), and the header's next paragraph says that symbols
  of other branches are missing. In the contract: not required;
  replaced by the premise, for each valuation, that it satisfies the
  source as well as the use (that a use entails the source is false
  even of the environments the meet makes: R1's example, T7-4), and,
  only for the caller to conclude of the machine, by
  `terms_in_scope_at_uses`. Discrepancy T5-1.
- **"Any symbol defined in one of the joined environments is also
  defined in the source environment."** Checked: only of the set of
  symbols that Simplify passes, not of the environments, and not for
  the meet's calls. In the contract: `symbols_of_uses_in_source`, for
  the well-formedness of the target only. Discrepancy T5-2.
- **"The relative order of variables defined in the source
  environment is preserved across all the joined environments."**
  Checked: no. In the contract: `binding_times_coherent`, defined and
  not required for the guarantee (the compiler checks its candidate
  again with `equal_in_all_joined_envs`); a premise of precision
  only. Discrepancy T5-3.

Also required and not in the header: every environment well formed
and with no name of type bottom unless it is bottom; `cut` defined;
the names to avoid include every known variable.

## 5. The known failure, and what else was found

- **Closures (T1-4, R3-7): found, shown, and repaired.**
  `n_way_join_closures_entry` kept a function type that only some
  entries have (the cases `None, Some val2` and `Some _, None` of its
  merge). `n_way_join_function_types_before_fix` in
  `FlambdaJoinTypes.v` is that part as it was, and
  `n_way_join_function_types_before_fix_unsound_statement`
  (proved) gives two closure types that differ only in function types
  and a value that has the first and not the type with the joined
  function types. The contract was not weakened. X2 showed the failure
  end to end (a correct program aborts with `-O3` and the n-way join)
  and made the change that repairs it: the merge answers `None` unless
  both sides have a function type (as `n_way_join_generic_product`
  and the binary `join_closures_entry` do). The human has taken that
  change for the integration branch, so it is the compiler that is
  transcribed: `n_way_join_function_types`, with
  `n_way_join_function_types_sound_statement` (proved). T1-4 is
  now a discrepancy that was demonstrated and fixed.
- **Recovered relations (T5-10): found, shown in Rocq, and shown
  from source.** `recover_inverse_relations` gives a variable `r` the
  type "the result of a relation on `x`" when every joined environment
  has a type for `x`; that is true only where `x` has a value, and a
  variable that an earlier n-way join imported from some of its uses
  has none on the other runs. In Rocq
  (`FlambdaJoinEnvSpec.v`, module `Witness`): two uses in which an
  in-types `x` has a type naming `r` for its `Is_null`; a run in which
  `x` has no value and `r` holds 0; the transcribed join, run with
  stand-ins for the meet's environment and the type joiner that do on
  this input what the compiler's functions do, answers a target that
  no valuation agreeing with the run satisfies
  (`witness_use_not_admitted_statement`,
  `transcribed_join_not_strict_statement`, with
  `witness_inputs_well_formed_statement`,
  `witness_run_satisfies_statement` and
  `witness_join_of_types_upper_bound_statement`). From source (X2): a
  variable imported from one branch whose type names an older tag
  variable for its `Get_tag`; a later join whose every use has a type
  for it puts the relation on the tag variable; a switch on the tag
  variable then loses the arm for a tag that only the runs without
  the imported variable have. A wrong result at run time.
  A repaired test is in the model beside the compiler's
  (`Join_env.repaired_choices`; `repaired_join` in
  `FlambdaJoinEnvSpec.v`): a relation is recovered for `x` only if,
  besides, what `x` stands for in every joined environment has a
  canonical element at the normal name mode there. Proved: with that
  test every relation that the rounds record is on a variable that has
  a value at every use, in a run that binds the variables of terms
  (`repaired_rounds_scrutinees_have_values_statement`); and the
  repaired join admits both uses of the instance
  (`witness_repaired_join_admits_statement`). It is a candidate for
  the change to the compiler; it does not recover a relation on a
  variable that exists only in types and has no alias in terms, even
  one that has a value on every run.
- **Equations on names with no value (T5-5), where the meet meets
  them.** The level that the join adds has equations on variables
  that it makes for a vector of simples, and on variables brought in
  from some of the uses. On a run through a use where the vector has
  no simple with a value, or that does not know the variable, the
  name has no value. As an environment the target is satisfied (a
  name with no value is asked nothing). The meet makes environment
  extensions of levels, and an extension asks that each of its names
  has a value; so the meet's contract for its join asks that no
  equation is added on a name with no value. The compiler's join has
  that only on the runs of a fourth condition,
  `added_equations_have_values`, with which
  `sound_join_where_equations_have_values` is the contract to prove
  for the join that the meet calls. Nothing in the compiler
  establishes the condition.
- **R1-16** (an empty set given to
  `create_naked_immediates_non_empty` when every head reduces to
  bottom): not looked at further; still traced, not shown reachable.
- **Lifted constants (T5-4).** The source holds the types of constants
  lifted in other branches. If such a constant had a field computed
  in one branch, its type would be false on a path through another,
  and no valuation would satisfy the source there. Looked for from
  source and not found: a constant with a field computed at run time
  is bound only at the top level of the unit, where every later
  instruction passes it, and what a branch lifts is static
  (`simplify/simplify_let_expr.ml`, `at_unit_toplevel`;
  `simplify/simplify_let_cont_expr.ml`). So the premise that the
  valuation satisfies the source is expected to hold on this count;
  the continuations workstream still owes the statement.
- **Kinds (T5-9).** See the third exclusion above.
- **Imported variables defined twice (T4-1)**, in the meet's callers
  of the join; T4 has it.
- **Order of names (T5-6):** the representative of an existential's
  aliases, the order of definition of imported and existential
  variables, the order of the rounds.

## 6. Dividing the work

Two halves, with `FlambdaJoinSpec.v` between them.

**E, the environment side (I keep it).** `types/env/join_env.ml`
(2031 lines) and the n-way entry of `types/join_levels.ml`.
- `FlambdaJoinEnv.v` (transcribed to the end): `Source_env`
  (`candidate_canonical_in_source_env`), `Bindings_in_target_env`,
  `Joined_envs`, the state `Join_env.t`,
  `n_way_join_canonical_simples`, the imports
  (`import_var_transitive` and the three with it),
  `join_aliases_into_bindings` (the leapfrog iterators become the
  intersection of the levels' domains, which is what they compute),
  `recover_inverse_relations`, the rounds,
  `define_or_eliminate_variables_and_add_equations`,
  `cut_and_n_way_join`, `Analysis.create`, and the three functions the
  type join calls (`n_way_join_simples`, `import_type`,
  `n_way_join_env_extension`).
- `FlambdaTypeProject.v` (transcribed): `Type_grammar.project_variables_out`
  (substituting types for existentials).
- `FlambdaJoinEnvSpec.v`, `FlambdaJoinEnvProofs.v`: the invariant and
  the meaning of the state, the statements of the three functions, and
  `sound_join_of_environments` for the transcription, with the meet
  and the type join as hypotheses.

**T, the type traversal (a second agent).** The 30 join functions of
`types/meet_and_n_way_join.ml` and their helpers
(`generic_merge_map_known`, `n_way_join_unknown`, `deduce_is_int`,
`deduce_get_tag`, `n_way_join_relation_simples`).
- `FlambdaJoinTypes.v` (handed over; it has the function-types part
  today, as changed and as it was), `FlambdaJoinTypesSpec.v`,
  `FlambdaJoinTypesProofs.v`:
  `n_way_join_type_sound`, by T1's `types_induction`, from the
  statements of the three environment functions.

**The interface.** Neither file imports the other. The type join is a
function of the state type and the three environment functions (a
section); the environment join takes `n_way_join_type` and the meet's
`meet_expanded_head` as arguments, as in the compiler. A third small
file ties the knot with fuel, with T4's meet. The statements each side
assumes of the other are in `FlambdaJoinSpec.v`, section "The join of
types, inside a join of environments", and are fixed there:

- the state is abstract, with three readings (`joined_env`,
  `definitions_of_existentials`, `kind_in_target_env`, from which
  `exists_in_target_env`) and two predicates that are the environment
  side's: the invariant `state_valid`, and `state_kinded`;
- four contracts, each with the premises that the environments joined
  are well formed and kinded (they do not change during a join, so
  kindedness is a premise and not a part of `state_valid`), and each
  "the invariant is kept, the state grows, the answer is in the scope
  of the target, the upper bound holds under every valuation that
  `extends_input`, and kinds": the type joined has the kind of each
  type given; a simple joined at a kind other than value is not a
  symbol; and at a state that is kinded (`state_kinded`, a second
  predicate of the environment side) the answer is kinded in the
  target (`type_kinded_in_target`, over the reading
  `kind_in_target_env`). The four are:
  `n_way_join_type_sound` (T proves it, E assumes it), and
  `n_way_join_simples_sound`, `import_type_sound`,
  `n_way_join_env_extension_sound` (E proves them, T assumes them).
  `import_type_sound` says nothing of values: a value that has the
  type at an input keeps it in the target because variables keep
  their values and the meaning of a type grows with the valuation
  (T1's `type_satisfied_monotone_all`); the type may gain values;
- every argument at an index mentions only variables that the
  environment of that index knows (`join_args_in_scope`), each index
  is given once (`join_args_distinct`), and the types given are kinded
  in their environment (`type_kinded`, defined with
  `aliases_with_kinds`), so that the simples given to the join of
  simples are of the kind given;
- `valid_states_have_witnesses`, which E proves of its invariant: at a
  valid state every valuation of an input extends to one of the
  target. Without it the upper bounds could be vacuous (the reviewer's
  point).

T also needs from T4 `expand_head` and
`reduce_head_of_kind_naked_immediate` with their statements, from T1
`type_eqb`, and from C0 `Keyed.filter_map_fold_on`.

**With T4 (agreed by message).** The meet's parameter is
`join_scoped_envs meet_expanded_head cut_after initial_env
initial_tenv scoped_envs : option Meet_env.t`. Fuel is inside. Rely:
T4's `meet_expanded_head_sound`. Guarantee: T4's (J1) to (J4), which
are items 1, 2, 3 and 5 above with no analysis. Both of us ask, for
each valuation, that it satisfy the initial environment as well as the
scoped one. `Meet_env.t` carries the names to avoid. T4
defines `Meet_env.t`, `Expanded_type.t` and the functions on them that
I call (`add_variable_definition`, `add_env_extension`,
`make_bottom`).

## 7. What I need, by name

- **T2**: `typing_env_satisfied_strictly`, `env_satisfies_strictly`
  (partial valuations; in progress), `typing_env_well_formed` with
  `cut_defined` and `cut_equations_found`,
  `typing_env_satisfied_coincide`,
  `add_variable_definition_satisfied` (a variable defined in types
  may be given any value of its kind, or none),
  `increment_scope_satisfied`, renaming of an environment. Already
  there and used: `cut`, `mem`, `find`, `binding_time_and_mode`,
  `get_canonical_simple_ignoring_name_mode`,
  `get_canonical_simple_exn`, `add_variable_definition`.
- **T3**: there now: `simple_equal_eq`,
  `canonical_element_idempotent`, `canonical_element_mentioned`,
  `respected_aliased`, `respected_canonical_at_mode`. Later: renaming
  of the aliases.
- **T1**: `type_well_formed`, `rename_type`, a boolean equality of
  types if X3's `structural` is chosen, and a word on who writes
  `project_variables_out`.
- **T4**: the definitions file of `Meet_env` and `Expanded_type`;
  `expand_head`, `reduce_head_of_kind_naked_immediate`;
  `add_equation_sound`.
- **N1**: `fresh_name` and its statement; the form of equivariance
  (agreed: up to an explicit permutation of the names made).
- **X3**: the list of uses of `==` and the behaviour under the flag.

## 8. Questions for the human

1. **Closures.** Answered: the human has taken the change to the
   compiler, and the model follows the changed function. No exception
   in the soundness theorem.
2. **Poison at joins.** Nothing is promised of a use satisfied only
   with the D26 exemption. Either that is accepted (the proof of
   unboxing must then treat a join after an unboxed continuation
   itself), or the environment records which names may hold poison so
   that the join can leave their types alone. Recommendation: accept,
   and revisit with the unboxing workstream.
3. **Lifted constants in the source (T5-4).** For T2 and the
   continuations workstream; the join only needs the premise. A
   search from source found no program in which the premise fails.
4. **Kinds (T5-9).** The contract has the premise that the use
   environments are kinded. Either the compiler's environments always
   are (then whoever states the kinds of environments proves it, and
   the join's unchecked path is unreachable), or the join should test
   the kinds of the simples when it makes a variable for them (one
   test in `existential_for_these_canonicals`). Recommendation: find
   out which with a reproducer (X2 or X5), and keep the premise
   meanwhile.
5. **Variables imported by a join from some of its uses (T5-5,
   T5-10).** After such a join the environment has variables that
   mean nothing on some runs, and the compiler goes on treating every
   variable of an environment as having a value. X2 has a program
   that the n-way join compiles wrongly for this reason. The model
   proves the compiler's join only for runs on which each recovered
   scrutinee has a value where the result of its relation has one.
   For the compiler: either a relation is not recovered on a
   scrutinee that was imported (or that stands, at a use, for a
   variable that was), or the environment records which variables may
   mean nothing on a run. For the model the conditions stay until the
   compiler is changed.
