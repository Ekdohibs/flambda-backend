# T5b on the meet: slices of the proof of the contract of meet

The slices are merged. T4 owns the meet, its contract and its
statements, and leads; I prove slices of the traversal that T4 has
cut loose, in files of my own, to statements that T4 fixes. Reviewer
of the meet: R1. (My own stream, the type traversal of the n-way
join, is `status/T5b.md`.)

Read this first (2026-11-18, at the landed tip of stage 131). This
file is of 11 and 12 November 2026 and is not kept current: what it
says in the present was so then. One thing is later: the folder
that it names for the files was brought up to date on 16 November,
in N2's commit of the folders.

Files, in `middle_end/flambda2/rocq/`:

- `FlambdaMeetProductsSpec.v`, `FlambdaMeetProductsProofs.v`: slice 1.
- `FlambdaMeetHeadsSpec.v`, `FlambdaMeetHeadsProofs.v`: slice 3.
- `FlambdaMeetRowLikeSpec.v`, `FlambdaMeetRowLikeProofs.v`: slice 2.
- `FlambdaMeetTermsSpec.v`, `FlambdaMeetTermsProofs.v`: a meet adds
  no variable that may occur in terms (after the slices; below).

## The cut, as agreed with T4

1. Products (`FlambdaMeetProducts*`): `combine_results3`, `map_meet`,
   `meet_int_indexed_product` (with `meet_array_of_types`),
   `meet_function_type` (with `meet_code_id`), the two products
   indexed by slots, `meet_closures_entry`. Done.
2. Row-like structures (`FlambdaMeetRowLike*`): `meet_row_like` and
   its instances for blocks and for closures. `meet_row_like` and the
   instance for blocks are done; the instance for closures is T4's,
   on top of the general statement (agreed with T4).
3. Heads (`FlambdaMeetHeads*`): `meet_relation`, `meet_variant`,
   `meet_head_of_kind_value_non_null`, `meet_head_of_kind_value`, to
   T4's target: for every value `v`, `part_meet_ok` of
   `meet_head_of_kind_value` at "the type made of the head fits" and
   "`v` has the type made of the head".

T4 keeps: the scopes and `meet_disjunction`; the head of naked
immediates; `meet_expanded_head0`, the induction on fuel, `Meet.meet`
and the contracts of `Typing_env`; the instance for the compiler's
join.

What I may suppose, as premises under these names: `join` with
`join_scoped_envs_sound join`; `choices` with `define_again choices =
false` and `lift_variables_of_single_case choices = true`; `mh` with
`meet_expanded_head_keeps mh` and `meet_expanded_head_sound mh`.
Nothing else without telling T4.

## Slice 1: done

Ten statements, proved; `make -C rocq -j8 all` and
`make -C rocq assumptions` pass with them.

- `combine_results3_ok`: as T4's `combine_results2_ok`, for three
  meets. (`combine_results3` is `combine_results2` of the first meet
  and of `combine_results2` of the other two.)
- `map_meet_ok`: the meet of two maps is right if the meet of what
  they bind is. A map fits if its keys are distinct and what it binds
  fits; it is true if what it binds each key to is true for that key.
- `map_meet_keys`: an answer of the meet of two maps binds the keys
  that either input binds.
- `meet_int_indexed_product_ok`, `meet_int_indexed_product_length`:
  the fields of blocks; an answer has as many types as the longer
  input.
- `product_satisfied_fields`: what "true" is for the types of the
  fields, against T1's `fields_satisfied`.
- `meet_function_type_ok`.
- `meet_closures_entry_ok`, `meet_closures_entry_slots`: an answer
  has a type for a slot if one of the inputs has.
- `extract_value_answer`: `extract_value` is an answer.

Three choices in it, which T4 has agreed:

- "True" for a product is T1's predicate and that the value has every
  position that the product has a type for. T1's meaning asks nothing
  of the type of a field past the end of the block or of a slot that
  the set of closures has not, and the meet of two products meets the
  types of such a position all the same; with T1's predicate alone
  the statements are false. The index of the case and conditions 6
  and 8 of the well-formedness of types give the rest, in slice 2,
  for the fields of blocks, the function types and the value slots.
  For the closure types of an entry they do not (found by R1): see
  "Settled with T4 and T1" below.
- An entry of closures fits if it is well formed under some index:
  the two entries met are under two indices and the answer under a
  third. `meet_closures_entry_slots` and
  `meet_int_indexed_product_length` are what `meet_row_like` needs to
  put the answer under the index answered.
- The statements about keys, lengths and slots are about any answer
  (`meet_answer`): an input that the meet names, either if it names
  both, or the new thing. A meet of several parts that names its
  second input names it whole, though a part may have named both.

The statements follow T4's contract as it is now: the first half of
`meet_call_ok` has the new names (`new_names_avoided`), `type_fits`
has the kinds of the names of equations, and an entry of closures
that fits has them too. The kind of the type that
`Meet_env.meet_type` answers is in T4's `meet_type_keeps`; the
premise that stood for it here is gone.

## Slice 3: done, given the rows of blocks and of closures

In `FlambdaMeetHeadsSpec.v`, each proved:

- `meet_relation_ok`: the meet of two variables that stand for the
  result of one primitive on the value is right. Such a variable fits
  an environment that knows it and has it at the kind of naked
  immediates (`relation_variable_fits`), and is true where it holds
  the number that the primitive answers on the value
  (`relation_variable_satisfied`, T1's `variable_is`); the number may
  depend on the static environment. From T4's `add_alias_keeps` and
  `meet_env_add_alias_sound`.
- `meet_disjunction_answer`: what `meet_disjunction` answers for each
  of its two parts is the part of an input, the bottom of that side,
  or what the meet of that side answered (`side_answer`). It supposes
  nothing of environments or values.
- `meet_head_of_kind_value_non_null_ok`: the meet of two heads that
  are not null is right, for any value, if the meet of two variants
  (`meet_variant_parts`) and the meet of the rows of two sets of
  closures are. Both are premises in the form `type_part_meet_ok`: a
  part fits, and is true, as the type that it makes. Two heads of
  different constructors have no value in common (T1's
  `heads_compatible_satisfied`), but for a variant and a mutable
  block; no value has the type of a mutable block in T1's meaning,
  so every case with one is right whatever is answered, and what is
  answered fits.
- `meet_variant_ok`: the meet of two variants is right, if the meet
  of the rows of two sets of blocks is (a premise in the same form,
  at `blocks_type`: the variants that are blocks among them). By
  T4's `meet_disjunction_ok`, the alternatives being a tagged
  immediate and a block; the variable for `Is_int` holds 1 on a
  tagged immediate and 0 on a block, by `meet_relation_ok` at the
  number that reads which the value is. In the proofs: a variant
  fits exactly when its parts do (the lists of free variables, of
  aliases and of names of equations of a variant are those of its
  parts; an extension inside a type fits in T4's sense, by lookups,
  exactly when its lists do, its keys being distinct and the
  environment having its invariant).
- `meet_head_of_kind_value_ok`, T4's target: the meet of two heads
  that may be null is right, if the meet of two heads that are not
  null is. By `meet_disjunction_ok`, the alternatives being a value
  that is not null and null. T1's meaning says, of a value that is
  not null, that the variable for `Is_null` holds 0, which is about
  the other side: that is had from `meet_disjunction_answer` (the
  meet of that side answers the variable of an input, which the
  environment of the meet knew and which holds what it held).
- `meet_value_heads_ok`: the three together. The meet of two heads
  of kind value is right if the meets of the rows of blocks
  (`meet_row_like_for_blocks`) and of the rows of closures
  (`meet_row_like_for_closures`) are. Those two are slice 2, and
  what is left of the heads of kind value.

## Slice 2: `meet_row_like` and the rows of blocks done

In `FlambdaMeetRowLikeSpec.v`:

- `meet_row_like_for_blocks_ok_statement`, proved
  (`meet_row_like_for_blocks_ok`): the meet of the rows of two sets
  of blocks is right, for any value, in the form that
  `meet_value_heads_ok` takes as its first premise. It supposes the
  join's contract, `define_again choices = false`,
  `lift_variables_of_single_case choices = true` and the two facts
  about `mh`.
- `meet_row_like_for_closures_right`: the same of the rows of
  closures, the second premise. Not proved here and not named as a
  statement: T4 proves it, in a file of T4's, from the general
  statement below.

In `FlambdaMeetRowLikeProofs.v`, the general statement,
`row_like_body_ok`: the body of `meet_row_like`
(`meet_row_like_unfold`) is right whatever its cases are of, and
whatever is taken for `need_join`. What it asks of an instance:

- of the products: what it is for one to fit (stable, and by the
  kinds of names only), `part_meet_ok` of their meet at "holds"
  (true, and the value has every position), and "true" as the meaning
  has it, stable;
- of the indexes: `index_ok` (index and product well formed
  together, for the case of a tag or of the other tags), under which
  "true" and a true index give "holds" on a valid static environment;
  that the index answered and the product answered are `index_ok`;
  that two indexes true of the value meet, to an index true of it;
- of the map of the cases: `find`, `known_ok`, that an empty map
  binds nothing, and that the merge goes through the tags of both
  maps once each with the state (said with an invariant of the state,
  an order that the steps follow, and a fact about each tag).

A row fits (`row_fits`) if its map is well formed and each case
fits: index and product well formed together, the product fits, the
extension fits (T4's `extension_fits`). It is true of the value in
hand, whose tag is given (`row_true`): the case that the tag selects
is unknown, or its index, its product and its extension are true.

How it goes. The cases are met one pair after the other with a
state: the environments of the pairs that stay, the extra variables
and the names to avoid. Kept of the state (`row_inv`): each
environment kept is the environment of the meet one scope further,
extended, and fits; two of them share only what the environment of
the meet knows; the variables new in one are to be avoided; an extra
variable is defined by the level of an environment kept, at its
kind. Kept of a pair: the case answered fits the environment of the
meet, or an environment kept whose level the extra variables cover;
where the two cases are true of the value, an environment kept has a
valuation under which the case answered is true. At the end the
environments kept are joined and the extra variables defined: by the
join's contract the environment answered knows what each kept one
knows, and the valuation of the pair of the tag of the value extends
to it.

The instance for blocks: `index_ok` is conditions 6 and 7 of the
well-formedness of types; an index true of the block and condition 6
say that the block has a field for each type of the product; two
canonical shapes that a block both has are one (T1's
`block_shapes_disjoint`); the product answered is as long as the
longer input (`meet_int_indexed_product_length`). The block in hand
is the one that the value is, read in the static environment.

With it `meet_value_heads_ok` has one premise left, the rows of
closures.

## Settled with T4 and T1: a valid static environment

The closure types of an entry of closures, on a static environment
that is not valid (found by R1 in review). A closure bound to a
symbol is projected to another function slot g of its set only if
the static set names a symbol for g (`project_function_slot`), though
g is among the functions of the set (`closure_in_set`). T1's
`closures_entry_satisfied` asks nothing of the closure type at g
where the projection has no value. So a value has two entries whose
closure types at g have no value in common, and
`meet_closures_entry` answers bottom: on such a static environment
the meet loses a value. `meet_closures_entry_ok` was not touched (its
"true" asks that the projections have values), but the statement of
the rows of closures, in slice 2, cannot get that from the index and
the well-formedness alone. C3's `statics_valid` gives it
(`project_function_slot_valid_statement`).

T4 and T1 chose: the meaning stays, and every statement that says
that no value is lost supposes `statics_valid statics` (T4's
`meet_call_ok`, second half, and the contracts above it). The
statements here keep their form; the proofs take the hypothesis and
pass it on. T1 adds `well_formed_closures_case_closure_type`: with a
valid static environment, a well-formed case that is satisfied gives
the projection at each slot that has a closure type, and its type.
That is the bridge for slice 2.

## After the slices: a meet adds no variable that may occur in terms

Asked by T1, for the second slice of the rewrite of the `Switch`,
through the orchestrator. Merged; R1 signed it.

- `meet_keeps_terms meet` (`FlambdaMeetTermsSpec.v`): a variable that
  may occur in terms under the environment that a meet answers
  (`in_scope_in_terms`) may occur in terms under the environment
  given. T1's definition, moved to a file of the meet at T1's
  request.
- `meet_to_prove_keeps_terms` (statement
  `meet_to_prove_keeps_terms_statement`), proved:
  `meet_keeps_terms (meet_to_prove join fuel)` for every fuel and
  every `join` with `join_scoped_envs_sound join`. The join that
  forgets has that contract (`forgetful_join_sound`, proved again),
  so there is an instance with no premise.
- After that merge, and merged since:
  `meet_to_prove_keeps_terms_of_keeps`, the same conclusion for
  every `join` with `join_scoped_envs_keeps join`
  (`FlambdaMeetKeepsSpec.v`: the conclusions of the contract that
  are not about values, with every meet that keeps; another
  hypothesis than the contract, which neither implies nor is
  implied by it). T4 proposed it: the proof used only
  `meet_expanded_head_keeps` of the meet of expanded types, which
  T4's theorem gives from such a join. Its instance for the join
  of the model is `model_meet_keeps_terms`
  (`FlambdaJoinMeetFitsSpec.v`).

It did not follow from the contract as it was: for the steps of a
meet that define no variable it follows from what T4 proves (the
names known keep their name modes), but nothing said the name mode
of a variable that is new, and a new variable comes from
`define_in_types` or from the join. So the contract has two clauses
more, in T4's `FlambdaMeetSpec.v`, with T4's agreement:

- `new_names_avoided`, a clause of `meet_expanded_head_keeps`, has a
  third conjunct: a variable that the environment answered knows and
  the first does not may not occur in terms there.
- `join_scoped_envs_sound` has a conclusion more (J6 in T4's request
  document): a variable of the result that the first environment
  does not know may not occur in terms, whether or not an
  environment joined knows it. It is a clause of its own, and not a
  conjunct of the one about the variables that no environment given
  knows, because the join brings into its result variables of the
  environments joined, and their name mode there does not carry
  over (T5 pointed it out).

What was proved again: `new_names_avoided_trans` (with
`keeps_names_mode`: a known name keeps its name mode where the least
binding time is the same); in `FlambdaMeetScopesProofs.v`, the two
places where variables are defined after an alternative or a join
(`direct_return_keeps`, `join_return_env_keeps`), with
`add_defined_vars_not_in_terms` (defining keeps a variable out of
terms, whatever `define_again` is); in my `FlambdaMeetRowLikeProofs.v`,
the join of the environments kept and the definition of the extra
variables (`row_join_all_spec`); `forgetful_join_sound`. No proof
leans on `define_again = false` for the new clause.

Checked against the compiler (`types/`): the meet defines variables
in `add_defined_vars` and in the fold over the extra variables after
the join (`meet_and_n_way_join.ml`), the join in
`define_or_eliminate_variables_and_add_equations` (`env/join_env.ml`),
and `Meet_env` in `add_env_extension_with_extra_variables` and
`add_env_extension_from_level` (`env/meet_env.ml`): all with
`Name_mode.in_types`. No place was found where a meet or a join
defines a variable with the normal name mode.

In the same series, asked by T5 through T4: `scoped_env_of` (what
the meet gives the join of each environment joined) has a fourth
conjunct, that the environment joined knows no symbol that the
first does not. A meet defines no symbol (`meet_env_extends` keeps
the defined symbols), so it costs one line (`scoped_scoped_env_of`).

Validation at the tip: `make -C rocq -j8 all`,
`make -C rocq assumptions`.

## Decided overnight

- A change to a signed contract, to have `meet_keeps_terms`. The
  question: T1's rewrite needs that a meet adds no variable that may
  occur in terms, and the contract of the meet did not give it.
  Options: (a) a named premise on the join beside the contract, and
  a second traversal of every function of the meet to carry the name
  modes (days: `meet_disjunction`, `meet_row_like`, the folds,
  `Meet_env`); (b) one conjunct more in `new_names_avoided` and one
  conclusion more in `join_scoped_envs_sound`, so that the traversal
  that exists carries it. Chosen: (b), with T4 (owner), T5 (whose
  join is to have the contract) and the orchestrator; R1 to review.
  Why: it is where the contract speaks of the new variables, the
  proofs change in five places, and both clauses are true of the
  compiler's functions as read. What it costs: the join's contract
  asks one thing more (T5 has not proved the contract of its join
  yet, and says the clause holds of it); `transcription_sound` has
  that stronger premise. `meet_contract`, `meet_sound`,
  `meet_keeps_well_formed` and `transcription_sound` are word for
  word as they were. Cost to change back: the two clauses and the
  five proofs, and T1's hypothesis returns.
- The fourth conjunct of `scoped_env_of`: T5's request, T4's
  decision; made here because the files were open.

## Found (2026-11-11)

Nothing of the compiler yet.
