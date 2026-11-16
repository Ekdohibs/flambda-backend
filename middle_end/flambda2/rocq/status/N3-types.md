# N3, on the side of the types: status

Owner: N3. The status of the validator's side of the same stream is
`status/N3.md`. N1 leads the work on names.

What this file describes is merged.

The task: what the work on names asks of everything that holds names,
for the typing environment: its renaming, its free names, and that its
functions commute with renaming. The owner of the environment had none
of it and left it to this stream.

Files (new, mine): `middle_end/flambda2/rocq/types/env/FlambdaEnvRenameSpec.v`
and `FlambdaEnvRenameProofs.v`; for the meet, four files of lemmas,
`FlambdaEnvRenameMeetProofs.v`, `FlambdaEnvRenameMeetPartsProofs.v`,
`FlambdaEnvRenameMeetProductsProofs.v` and
`FlambdaEnvRenameMeetValuesProofs.v` (below); for the join,
`FlambdaEnvRenameJoinSpec.v` and `FlambdaEnvRenameJoinProofs.v`,
`FlambdaTypeProjectRenameProofs.v`, `FlambdaJoinMeetRenameSpec.v`
and `FlambdaJoinMeetRenameProofs.v`, and an example,
`FlambdaJoinMeetRenameExampleSpec.v` and
`FlambdaJoinMeetRenameExampleProofs.v` (below). No file of another
owner is edited.

## Done

Seventy-nine statements.

- **The action** on a level, on what the environment knows of its
  names, and on an environment (`rename_level`, `rename_cached_level`,
  `rename_typing_env`); the free variables; the laws (identity,
  composition, inverse, two renamings that agree on the free variables,
  one that fixes them, the free variables of what is renamed); the two
  instances of the interface for things that contain names, with its
  laws (fourteen).
- **The functions of a level** and of what the environment knows of its
  names (sixteen): `defined_variables`, `find_kind`, `is_empty`,
  `add_definition`, `add_or_replace_equation`, `concat`,
  `as_extension_without_bindings`, `variables_in_order`,
  `fold_on_defined_vars`, and those of `Cached_level`.
- **What an environment is asked** (eighteen): `find` and the two
  functions beside it, `binding_time_and_mode`, `mem`,
  `binding_time_resolver`, `alias_is_bound_strictly_earlier`,
  `get_canonical_simple_exn` and the two beside it,
  `type_simple_in_term_exn`.
- **What is done to an environment** (fifteen): `add_variable_definition`,
  `add_definition`, `add_definitions_of_params`, the definitions of
  symbols, `replace_equation`, `aliases_add`, `add_alias`, `cut`,
  `cut_as_extension`, and the changes of scope.
- **What satisfies a renamed environment** (seven): the renamed
  valuation satisfies the renamed environment exactly when the
  valuation satisfies the environment, in each sense of
  `FlambdaTypingEnvSpec.v` (`typing_env_satisfied_but` with the exempt
  names renamed, `typing_env_satisfied_strictly`,
  `typing_env_satisfied`, `env_satisfies`, `env_satisfies_strictly`),
  with the same of `exempt_by_poison` and `in_scope_in_terms`. The
  owner of the environment asked for these.
- **The invariant and the kinds of a renamed environment** (nine): the
  test of the invariant answers the same of the renamed environment
  (`typing_env_well_formed`); `levels_agree`, `typing_env_invariant`,
  `types_well_formed` and `typing_env_kinded` hold of the renamed
  environment exactly when they hold of the environment, and the test
  of the types being well formed (`types_well_formed_b`) answers the
  same of the two (the owner of the pass that rewrites primitives from
  the types asked for it: the pass runs the test); a renamed
  `Simple` is of a kind in the renamed environment exactly when the
  `Simple` is in the environment (`simple_of_kind`); the same of the
  second condition on kinds, of the names that have an equation in an
  extension inside a type (`type_equations_kinded`,
  `types_equations_kinded`). The owner of the environment asked for
  these too. Each condition of the invariant is
  shown to be kept by a renaming, one lemma a condition (seventeen
  conditions, with the one that a variable that has a type is defined
  by a level); the converse is the same fact of the inverse renaming.

Each statement about a function is an equation with no premise on the
environment; one has a premise on an argument that is a function
(`fold_on_defined_vars`).

## The meet: lemmas for its owner

The owner of the meet states that the meet commutes with renaming
(`FlambdaMeetRenameSpec.v`, with the action on an environment of the
meet and on what a meet answers). Four files of mine hold lemmas
that its proof needs. They have no
statement: each lemma is named after a function, `<function>_renamed`,
and says that on the renamed environment and renamed arguments the
function gives the renamed answer.

- **`FlambdaEnvRenameMeetProofs.v`** (fifty-six lemmas). Every function
  of the module `Meet_env` of `FlambdaMeetEnv.v`, with `map_result`
  and `extract_value`; `kind_of_simple`; `expand_head0`. A function
  that is given the meet of expanded types commutes if that meet does
  (`meet_expanded_head_equivariant`, a hypothesis of a section). Then,
  for a join, choices and a fuel at which the meet of expanded types
  commutes: `Meet.meet` (which is `meet_function_equivariant` of it)
  and `Meet.meet_shape`, and the functions that add to a typing
  environment (`Flambda2_types.add_equation`,
  `add_equations_on_params`, `add_env_extension`, `add_relation` and
  the three relations).
- **`FlambdaEnvRenameMeetPartsProofs.v`** (seventeen lemmas). The
  functions of `FlambdaMeet.v` that put a meet together from the meets
  of parts (`map_return_value`, `combine_meet_return_values`,
  `sequence_return_value`, `set_meet`, `meet_mapping`, `map_meet`,
  `combine_results2`, `combine_results3`, `meet_code_id`,
  `meet_alloc_mode`, `meet_unknown`, `meet_or_unknown_or_bottom`) and
  those of the relations of naked immediates
  (`head_of_kind_naked_immediate_from_descr` and the one beside it,
  `reduce_head_of_kind_naked_immediate`, `meet_inverse_relations`).
  A function that is given the meet of a part is given, on the renamed
  side, a meet that answers the renamed answer of renamed parts. The
  keys of a map and the numbers of a set are not renamed.
- **`FlambdaEnvRenameMeetProductsProofs.v`** (nineteen lemmas). For a
  meet of less fuel that commutes: `meet_relation`,
  `meet_function_type`, `meet_array_of_types`,
  `meet_int_indexed_product`, the two products indexed by slots and
  `meet_closures_entry`; then `reduce_inverse_relations` and
  `meet_head_of_kind_naked_immediate`, and the meets of naked numbers,
  of `Rec_info` and of regions. Where the function is a meet of two
  parts the lemma is `meet_of_parts_equivariant` of it, which is what
  the owner's lemma for a row-like structure takes. The two for naked
  immediates need that a renaming of variables fixes the type that the
  choices give a name of which a number is the tag; that is a
  hypothesis, and it is shown of the compiler's choices and of the
  repaired ones.
- **`FlambdaEnvRenameMeetValuesProofs.v`** (seven lemmas). The meets
  of the heads of values: `meet_variant`,
  `meet_head_of_kind_value_non_null` and `meet_head_of_kind_value`,
  with the four meets of alternatives that they give
  `meet_disjunction`. They take a join and a meet of less fuel that
  commute, and the meets of the two row-like structures, which are
  the owner's to prove; the law of `meet_disjunction` is the owner's
  too (`FlambdaMeetRenameScopesProofs.v`).

What is left to the owner of the meet: the join inside
`meet_disjunction` and `meet_row_like`, the two rows, the meet of
expanded heads, the induction on the fuel, and the statements.

Six definitions are in these files, for the statements of the
lemmas: the renaming of a list of equations and of a list of
parameters, of the state of two folds, and of what `meet_variant`
answers. If a statement to review comes to need one, it moves to the
`Spec` file of that statement.

## The join of typing environments under renaming

The orchestrator's order: every function of the model that reads
names commutes with renaming, and the meet is stated for a join that a
renaming relates to another. This says that the model's join is such
a join.

- **The source of new names.** The join makes variables with a source
  that is one of its parameters. No function from the names to avoid
  to a name commutes with every renaming (`no_source_commutes`),
  which is a fact about the source alone: the equation of the join
  cannot be proved with one source on both of its sides, function by
  function. That the model's join with its one source does not answer
  the renamed answer is shown by an example (below). What is stated:
  a renaming acts on
  the source too (`rename_fresh`: the renaming undone on the names to
  avoid, the source asked, its answer renamed), and the join with the
  renamed source, of renamed arguments, answers the renamed answer.
  It is said as a relation between two sources (`fresh_related`), and
  the same for the parameters of the join (`parameters_related`), the
  choices (`choices_related`), the join of types
  (`n_way_join_type_related`) and the functions of the environment of
  the meet (`meet_env_functions_related`).
- **`FlambdaEnvRenameJoinSpec.v`** (fifteen statements, all proved):
  the five about the source; the model's parameters with the renamed
  source are related to the model's; the one function that makes a
  variable, and the function of the state that calls it; the two
  choices are related to themselves; the model's join of types is
  related to itself where the parameters are; the join that the meet
  calls (`cut_and_n_way_join_related`), the join that Simplify calls
  and the function of `join_levels.ml` over it, each for any related
  functions of the environment of the meet; and what Simplify reads of
  an analysis. The action on the state of a join, on the names of the
  target, on the joined environments, on the aliases of the variables
  made and on an analysis is defined there, to be reviewed.
- **`FlambdaEnvRenameJoinProofs.v`** (about 150 lemmas): every
  function of `FlambdaJoinEnv.v` has its lemma, named
  `<function>_renamed`. The traversal of the types
  (`FlambdaJoinTypes.v`) is N1's file,
  `FlambdaEnvRenameJoinTypesProofs.v`, for any state and any action on
  it; its seven hypotheses are discharged here at the state of the
  join.
- **`FlambdaTypeProjectRenameProofs.v`**: the removal of variables
  from a type (`project_variables_out`) on renamed arguments, by the
  induction on types.
- **`FlambdaJoinMeetRenameSpec.v`** (eight statements, all proved):
  the model's functions of the environment of the meet are related to
  themselves; two joins with related parameters and choices are
  related in the sense of the meet's file
  (`join_scoped_envs_related`); so the model's join and the model's
  join with the renamed source are, and the model's meet with the
  renamed source, of a renamed environment and renamed types, answers
  the renamed answer (`model_meet_related`); the same of the join
  that Simplify calls (`model_join_related`).
- **`FlambdaJoinMeetRenameExampleSpec.v`** (three statements, each a
  computation): a join of the model that makes a variable; with the
  model's source, its answer of the inputs renamed by an exchange of
  that variable with one that is nowhere in them is not the renamed
  answer; with the source renamed it is. Nothing is proved of another
  source.

Not stated: that the join or the meet with ONE source answers, of
renamed arguments, the answer renamed up to the variables made. It
needs one more fact, that two sources give answers that differ by a
renaming of the variables made.

## To do

- The fact above, if the up-to form is wanted.
- The extensions, when their owner asks.

## Decided overnight

1. **The free variables of an environment are every variable written
   in it, and its equality is that of Rocq.** Options: (a) that; (b)
   the variables that its levels define are bound, and environments
   are equal up to their renaming. Chosen: (a). An environment is the
   context of a term; the variables it defines are free in the term,
   and it records the identity of those that no term may mention too
   (type-only variables, variables out of scope). With (b) the
   functions that look a name up (`find`, `mem`, the canonical element
   of a name) are not functions of the class of an environment: they
   do not descend to the quotient as they are. Where a level is cut
   out and its variables made anew (the levels that a join is given),
   the level with what it scopes over is a name abstraction: that is
   for the stream of the join, with the interface. Cost of changing:
   the two instances and the header.
2. **The files are named `FlambdaEnvRename*`**, not
   `FlambdaTypesEquivariance*` as first suggested: `FlambdaTypes*` and
   `FlambdaTypingEnv*` are patterns of other owners in `OWNERS.md`.
3. **The base was at first the series with the renaming of types**,
   not the integration tip of that time: the action on an environment
   needs that renaming. That series is merged.
4. **Statements are written with the renaming of a name of the types'
   file** (`FlambdaTypeRenameSpec.rename_name`). The file of the
   aliases had the same function under the same name; its owner has
   removed it, and there is one renaming of names. The same of
   `variables_renaming`, which the types' file of instances and N1's
   file of instances both define: N1's stays.

## Found

- **No source of new names commutes with every renaming**
  (`no_source_commutes`: the empty list of names to avoid is fixed by
  every renaming, and no variable is). So the equation of the join
  cannot be proved with one source on both of its sides in the way
  used here, function by function down to the call of the source.
  That the join as a whole, with one source, fails to commute with
  some renaming does not follow from that: it takes an example of a
  join that calls the source. There is one for the model's join and
  the model's source (`FlambdaJoinMeetRenameExampleSpec.v`, three
  statements, each a computation: a join that makes a variable; with
  the model's source, its answer of the inputs renamed by an exchange
  of that variable with one that is nowhere in them is not the
  renamed answer; with the source renamed it is). The reviewer found
  the example; nothing is proved of another source. Not a discrepancy
  with the compiler, whose join is never given renamed environments.
  Nothing else in the join goes by the number or the
  order of a name: variables are compared by equality and by binding
  time, and the one call of the source is in
  `existential_for_these_simples`.
- Nothing that is not equivariant. No function of the environment makes
  a name or goes by the order of names: the order of binding is read
  from binding times, which a name keeps, and a tie is settled by the
  order of a list of the model. The same of the environment of the
  meet and of the functions that put a meet together.
- The functions of the aliases take the environment's map from names to
  types and read only its binding times; their statements under
  renaming keep the types of that map as they are. The action here
  renames those types, so the proofs here first show that the functions
  of the aliases do not read them (`add_alike`,
  `get_canonical_element_exn_alike` in the `Proofs` file). The owner of
  the aliases has since stated them in its own file, with the proofs
  from here, and the proofs here use its statements.
- The check of claims (`make claims`) reads a statement only where
  `Definition ..._statement : Prop` is on one line. A statement whose
  name is long enough for `: Prop` to go to the next line is not
  checked, and nothing says so. Two statements here were so; they are
  on one line now. The integrator is told.

## Review

The reviewer of the types read the seventy statements and the
definitions and asked for two repairs of the header (it cited a file
that is not in the tree of this base; it said that the variables of
the levels are those in scope in the term, which leaves out the
type-only variables and those out of scope) and one of this file (the
reason against binding the variables of the levels was put too
broadly). They are folded into the commits they belong to. The owner
of the environment read the statements against its definitions and
had no objection.

The reviewer then read all seventy-eight statements of that time and
signed them, and asked for two repairs of comments (the file cited for
`variables_renaming`; what the header says the model has of the
compiler's two structures with a renaming). They are folded into the
commits they belong to. The statement about `types_well_formed_b` was
added after that and is to be read.

The reviewer read and signed the twenty-three statements of the join
under renaming, after one repair of the header (what the theorem on
sources of new names does and does not say of the join as a whole),
and found the example of the join with one source, which it then
read and signed as three statements.

## Obstacles

None.
