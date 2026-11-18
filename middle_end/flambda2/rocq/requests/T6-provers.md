# T6: the provers and reification

A note of 18 November 2026. This is the plan as it was written on
11 November 2026, and it is not kept current: where it says "today",
or that something is owed, missing, open, on a branch or not under
this branch, that was so on that day. What was done is in
`middle_end/flambda2/rocq/status/T6.md`. On 18 November 2026 these
were brought to what is so on that day, each with its date: four
rows of the column "State" of the table of section 4, and what is
said under that table of a stand-in; and, in section 5, item 7 and
what is owed of renaming. Nothing else was changed.

What workstream T6 transcribes and states, on what it rests, and in what
order. The guides are `middle_end/flambda2/types/provers.ml` (1624
lines, 67 exported functions) and `middle_end/flambda2/types/reify.ml`
(793 lines, one exported function). T1 wrote the first notes for this
stream; this is their rewrite by the owner. Reviewer: R3.

Simplify asks a prover at every primitive, switch and call, so the
statements here are the interface between the types and the rewrites
(the S streams): a rewrite is justified by the statement of the prover
it called.

## 1. The answers, and the one shape of statement

`provers.ml` has three types of answer.

- `proof_of_property`: `Proved x` or `Unknown` (the `prove_*`
  functions). `Proved x` is about every value of the type.
- `meet_shortcut`: `Known_result x`, `Need_meet` or `Invalid` (the
  `meet_*` functions). The answer stands for the meet of the type with
  the type of the values on which the operation that asks is defined,
  its domain. So it is about the values of the type that are in the
  domain: `Known_result x` says `x` of each of them, and `Invalid` that
  there is none.
- `generic_proof`: `Proved x`, `Unknown` or `Invalid`. Internal: one
  function on a head (`prove_*_generic_value`) serves both forms,
  through `gen_value_to_proof` (which answers `Unknown` on a head that
  may be null) and `gen_value_to_meet` (which looks at the part that is
  not null, so its domain excludes `Null`).

A prover stops the compiler (a fatal error) where `expand_head` does: a
name that is not bound, a canonical element whose type is an alias. The
model's functions answer an `option`, `None` being that stop, as T4's
`expand_head` does.

Two generic predicates, each instantiated once per prover. `holds` and
`domain` are given by the statement of each prover; they read the value
through the observations of the machine (`block_of`, `boxed_number_of`,
`closure_in_set`, `string_of`, `is_immediate`), not through
`eval_prim`.

    Definition prover_sound {A : Type}
        (prover : Typing_env.t -> flambda_type
                  -> option (Proof_of_property.t A))
        (input : Typing_env.t -> flambda_type -> Prop)
        (holds : static_env -> env -> A -> value -> Prop) : Prop :=
      ∀ statics valuation tenv t x v,
        typing_env_satisfied_strictly statics valuation tenv ->
        input tenv t ->
        type_satisfied statics valuation v t ->
        prover tenv t = Some (Proof_of_property.Proved x) ->
        holds statics valuation x v.

    Definition meet_shortcut_sound {A : Type}
        (prover : Typing_env.t -> flambda_type
                  -> option (Meet_shortcut.t A))
        (input : Typing_env.t -> flambda_type -> Prop)
        (domain : static_env -> value -> Prop)
        (holds : static_env -> env -> A -> value -> Prop) : Prop :=
      ∀ statics valuation tenv t v,
        typing_env_satisfied_strictly statics valuation tenv ->
        input tenv t ->
        type_satisfied statics valuation v t ->
        domain statics v ->
        match prover tenv t with
        | Some (Meet_shortcut.Known_result x) => holds statics valuation x v
        | Some Meet_shortcut.Invalid => False
        | Some Meet_shortcut.Need_meet | None => True
        end.

`input` is a named condition on what a prover is asked, for the
provers that rely on more than the meaning of a type gives (the
reviewer's request): none (`any_input`), or that the type and the
types of the environment are well formed (`well_formed_input`, with
T1's `type_well_formed`). The second is needed where the model's
grammar has a type that the compiler's has not (a type of naked
numbers at the kind of the immediates), and where a prover reads a
function type that is missing under an exact index as no closure at
all.

A third form, in the proofs only, is the second one for the internal
functions on heads; two lemmas take it through `gen_value_to_proof` and
`gen_value_to_meet`, so each prover is proved once, on heads. A prover
with more arguments (a kind, a field index, a name mode) is stated for
every value of them. One statement per primitive then ties `domain` to
the primitive: the primitive has a value only on a value of the
domain, and its value is what `holds` says; not the converse. Those of
the scrutinee questions are with slice 1; the others are for the S
streams.

Nothing is claimed of `Unknown`, `Need_meet` or `None`. Completeness (a
prover answers whenever the type decides the question) is not stated.
That the provers do not stop on the environments Simplify builds is a
later statement, from T2's invariant and T4's for `expand_head`.

## 2. Poison

The statements take `typing_env_satisfied_strictly`: every name that has
a value has its type, poison or not (decision D26 as amended). Under it:

- A value that is poison has only the types that say no more than its
  kind, that it is the value of a name, or that it is not `Null`. On
  such a type every prover answers `Unknown` or `Need_meet`, but for
  three: `meet_is_null` answers `Known_result false` on a type that is
  not null (poison is not `Null`, as the `Is_null` relation has it; the
  primitive has no value on poison); `prove_equals_to_simple_of_kind`
  and the provers that answer a `Simple` answer the canonical element,
  which has the same value, poison included (T3's classes are exact);
  and `prove_physical_equality` (below). This is stated once, as
  `prover_silent_on_poison`, for the provers of each slice.
- A name read inside a type (the immediates of a variant that are an
  alias, the name of an inverse relation in `meet_naked_immediates`)
  has its type, so a head found for it is a fact about its value. With
  T1's `relation_holds` as amended (`Is_int` and `Get_tag` hold of no
  poison) the three relations are sound in `meet_naked_immediates`,
  and `Get_tag` there does not make the mistake of meet's
  `reduce_inverse_relations` (T1-1): it answers only when the
  immediates of the name's type are bottom.

Under the weaker notion (`typing_env_satisfied`, the names that hold
poison exempt), which is what holds after unboxing of variants, a
prover that expands a name that holds poison answers what the
environment says of the name, not what is true of the value. One case
is known to be wrong there: a name `n` that holds poison and has the
type of `Null`, and a naked immediate with the relation `(Is_null, n)`:
the relation gives 0 and `meet_naked_immediates` answers `{1}`. It is
recorded as a discrepancy (not demonstrated: no program is known that
gives such a name that type), and the statements for the weaker notion,
with the premise that the names expanded hold no poison, come after the
strict ones.

## 3. The functions

Lines are of `provers.ml` on the integration branch. "Slice" is the
order of work (section 5). `D` is the domain of a meet shortcut.

### Expressible in the model's grammar today

| Function | Lines | Slice | What `Proved x` or `Known_result x` says of a value `v` of the type |
|----------|-------|-------|------|
| `as_meet_shortcut`, `as_property`, `gen_value_to_gen`, `gen_value_to_proof`, `gen_value_to_meet` | 55-103 | 1 | the two lemmas of section 1 |
| `prove_is_int` (`prove_is_int_generic_value`, `prove_is_int_generic`) | 131-157 | 1 | `x = true` and `v` is a tagged immediate, or `x = false` and it is not one |
| `meet_is_int_variant_only` | 159 | 1 | the same; `D`: a tagged immediate or a block |
| `prove_is_not_a_pointer` | 162-179 | 1 | `x = true`: a tagged immediate or `Null`; `x = false`: neither |
| `prove_get_tag` (`prove_get_tag_generic_value`, `prove_get_tag_generic`) | 186-217 | 1 | `v` is a block whose tag is in `x` |
| `meet_is_null` (`prove_is_null_generic`) | 219-233 | 1 | `x = true` and `v` is `Null`, or `x = false` and it is not; `D`: kind value |
| `meet_naked_immediates` (`prove_naked_immediates_generic`) | 235-303 | 1 | the number of `v` is in `x`; `D`: a naked immediate |
| `prove_equals_tagged_immediates` | 308-338 | 1 | `v` is a tagged immediate whose number is in `x` |
| `meet_equals_tagged_immediates`, `meet_equals_single_tagged_immediate` | 340-369 | 1 | the number of `v` is in `x` (is `x`); `D`: a tagged immediate |
| `meet_naked_float32s` ... `meet_naked_masks` (eleven, one `meet_naked_number`) | 371-520 | 1 | the number of `v` is in `x`; `D`: a number of the kind |
| `prove_variant_like`, `meet_variant_like` | 522-580 | 2 | a tagged immediate in `const_ctors` if known, or a block whose tag is in the map, with the size and shape there; `D`: a tagged immediate or a block |
| `prove_unique_tag_and_size` | 697-729 | 2 | a block with that tag, shape and size |
| `prove_unique_fully_constructed_immutable_heap_block` | 731-752 | 2 | a block with that tag, shape and size, whose fields are the values of the `Simple`s |
| `meet_block_field_simple` | 1112-1157 | 2 | the field at the index is the value of the `Simple`; `D`: the load has a value (a block with that field, of the kind asked) |
| `prove_is_a_boxed_or_tagged_number`, `prove_is_a_tagged_immediate`, the nine `prove_is_a_boxed_*`, `prove_nothing` | 582-695 | 3 | a tagged immediate; or a boxed number of the kind whose contents have the type answered |
| `prove_is_or_is_not_a_boxed_float` | 648-653 | 3 | `x = true`: a boxed float; `x = false`: not one (right because the observations of the machine are disjoint) |
| the nine `meet_boxed_*_containing_simple` | 992-1110 | 3 | the contents of the box are the value of the `Simple`; `D`: a boxed number of the kind |
| `prove_single_closures_entry`, `meet_single_closures_entry` | 835-867 | 4 | a closure at the slot answered, of a set of which the entry holds; needs T1's `type_well_formed` |
| `prove_code_ids`, `meet_code_ids` | 869-901 | 4 | the code of the closure is one of the code ids (exactly: the model has no code age) |
| `meet_project_function_slot_simple`, `meet_project_value_slot_simple` | 1159-1211 | 4 | the projection is the value of the `Simple`; `D`: the projection has a value |
| `prove_strings`, `meet_strings` | 922-934 | 5 | a string that is one of those answered |
| `prove_equals_to_simple_of_kind` | 105-127 | 6 | `v` is the value of the `Simple` |
| `prove_tagging_of_simple`, `meet_tagging_of_simple` | 936-991 | 6 | `v` is the tagged immediate of the number that is the value of the `Simple` |
| `prove_physical_equality` | 1297-1624 | 7 | of two types: `Phys_equal` of two values of them, where it has a value, is `x` |

### Transcribed, with nothing stated

| Function | Lines | Why |
|----------|-------|-----|
| `prove_alloc_mode_of_boxed_number` | 1239-1256 | allocation modes have no meaning in the denotation |
| `never_holds_locally_allocated_values` | 1258-1295 | the same |

The allocation mode in the answers of `prove_unique_tag_and_size_value`
and `prove_is_a_boxed_or_tagged_number`, and the test on it in
`prove_unique_fully_constructed_immutable_heap_block`, are transcribed
and nothing is stated of them.

### Not expressible today

| Function | Lines | What is missing |
|----------|-------|-----------------|
| `meet_is_flat_float_array`, `meet_is_non_empty_naked_number_array`, `prove_is_immediates_array`, `meet_is_immutable_array`, `prove_is_immutable_array` | 754-833, 903-920 | the grammar has no `Array`, the machine no arrays |
| `meet_rec_info` | 1213-1237 | the grammar has the types of kind `Rec_info`, but their head is `unit`: the expression that the prover answers is left out |
| the `Array` case of every prover on heads | | the same; a head has no such case |
| coercions on the `Simple`s answered | | a `Simple` of the model has no coercion |

### Reification

`reify` (`reify.ml` 222-793, with `try_to_reify_fields` 66-220), after
the provers. Three answers to state:

- `Simple s`: the value of `s` is `v`.
- `Invalid`: the type has no value.
- `Lift (Immutable_block ...)`, `Lift (Boxed_* ...)`: `v` is a block
  with that tag whose fields are the values given, or a box of that
  number. Not "the lifted symbol is `v`": a block and a symbol bound to
  an equal block are different values of the model (decision D6), so
  the soundness of the replacement belongs to the rewrite (S5).

The conditions under which the compiler lifts (`var_allowed`:
allocation modes, defined at toplevel, symbol projections) have no
meaning in the denotation; they are transcribed and nothing is stated
of them, with a CR. The array cases are left out.

## 4. What it rests on

| Needed | Of | State |
|--------|----|-------|
| `type_satisfied` and its equations; `relation_holds` as amended; `is_obviously_bottom_satisfied_statement`, `row_like_for_blocks_is_bottom_satisfied_statement`, `relation_is_int_statement`, `relation_get_tag_statement`, `relation_is_null_statement`, `type_satisfied_kind_statement`, `alias_satisfied_statement` | T1 | on 11 November, on T1's branch, which this branch was cut from; on 18 November 2026, in the tree (`middle_end/flambda2/rocq/types/grammar/FlambdaTypeMeaningSpec.v`) |
| `row_like_for_blocks_all_tags` with `all_tags_satisfied_statement` (slice 1); `all_tags_and_sizes`, `get_singleton`, `get_field` (slice 2); `get_single_tag`, `get_closure`, `get_env_var`, `find_function_type` (slice 4); `type_well_formed` (slices 1 and 4) | T1 | on 11 November, on T1's branch; on 18 November 2026, in the tree (`FlambdaTypeAccessSpec.v`, `FlambdaTypeWellFormedSpec.v`, in `middle_end/flambda2/rocq/types/grammar/`) |
| `typing_env_satisfied_strictly`; `get_canonical_simple_satisfied_statement`, `get_alias_then_canonical_simple_satisfied_statement` (slices 2, 4, 6); `find_satisfied_strictly_statement` | T2 | merged |
| the aliases are exact (through T2's statements) | T3 | merged |
| `Expanded_type`, `expand_head`, `expand_head_sound_statement` | T4 | on 11 November, on T4's branch, with a stand-in here (below); on 18 November 2026, in the tree (`FlambdaExpandHead.v`, `FlambdaExpandHeadSpec.v`, in `middle_end/flambda2/rocq/types/meet/`), with no stand-in |
| `Phys_equal` on allocated values and symbols (slice 7) | the core | on 11 November, open (decisions D19, D6); on 18 November 2026 the primitive has a value on some imported symbols and none on two allocated values (section 5, item 7) |

On 18 November 2026 there is no stand-in: the provers read the
`expand_head` of `FlambdaExpandHead.v`, and no statement has its
soundness as a premise. The next paragraph is as it was written.

The stand-in for T4: a copy of T4's `Expanded_type` and `expand_head`
under T4's names, in a file of its own that goes when T4's is under
this branch; and the soundness of `expand_head` as a premise of each
statement, named `expand_head_sound_assumed`, with T4's text. T6 adds
`descr_oub`, which T4 left out because meet does not use it.

## 5. The order

By use. `simplify_unary_primitive.ml` calls `prove_is_int`,
`meet_equals_tagged_immediates`, `meet_block_field_simple`,
`prove_unique_fully_constructed_immutable_heap_block`,
`prove_is_a_boxed_or_tagged_number`, `prove_is_or_is_not_a_boxed_float`,
three of the `meet_naked_*s`, the two projections and the provers of
strings; `simplify_binary_primitive.ml` calls
`prove_equals_tagged_immediates`, `meet_naked_immediates`, two of the
`meet_naked_*s` and `prove_physical_equality`;
`simplify_switch_expr.ml` calls none directly.

1. The scrutinee questions and the numbers (slice 1 of the table).
   They rest on `expand_head` and on T1's statements for a `Variant`
   only.
2. Blocks.
3. Boxed numbers.
4. Closures.
5. Strings.
6. The provers that answer an alias.
7. `prove_physical_equality`, last: every `Proved` it gives on symbols
   and allocated values is a claim about `Phys_equal` there, which the
   machine did not define on 11 November 2026. P1-1 (two poison
   constants fold to "different") is its known disagreement. On
   18 November 2026 the primitive has a value on two tagged
   immediates; on a tagged immediate against an allocated value or
   an imported symbol of the class of identities or of data; and on
   an identity (an imported symbol of the class `Import_identity`)
   against another identity, an allocated value or an imported
   symbol of the class of data. It has none on two allocated values,
   on null or on poison (decision D19; `eval_phys_equal` of
   `middle_end/flambda2/rocq/semantics/FlambdaPrimitiveSpec.v`).
8. `reify`.

Each slice is one review request: the functions, the statements, the
proofs. Owed for every prover, and not in the slices yet: that it
commutes with a renaming of names. The statement needs the renaming
of typing environments, which is T2's and is not under this branch.
On 18 November 2026 the statements of the provers under renaming
are in the tree, with their proofs (`FlambdaProversRenameSpec.v`,
`FlambdaProversRenameProofs.v`, in
`middle_end/flambda2/rocq/types/provers/`).
T7 can run each extracted prover against the compiler's on its
generated types, as each is a function of an environment and a type.

## 6. How the transcription differs from the OCaml

- Sets of numbers and of tags are lists (T1's grammar); an answer is to
  be compared as a set.
- `Target_ocaml_int.bool` is 1 or 0 and `Tag.to_targetint_31_63` is the
  number of the tag: the model has no machine width.
- `wrong_kind` answers its default.
- The fifteen constructors of `descr_oub` are five, as those of T1's
  grammar are; the eleven `meet_naked_*s` are one function that takes
  the kind.
- `inverse_relations` is a list of pairs of a relation and a name;
  `prove_naked_immediates_generic` folds over it in the order of the
  list. The answer does not depend on the order, as a set.
