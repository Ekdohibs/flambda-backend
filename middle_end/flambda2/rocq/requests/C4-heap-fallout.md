# C4, the heap: what the prototype costs on the integration branch

The prototype was written on the integration branch at 0a2933e7c9. It
was merged on fcd6e9c15c, after C1's law that no primitive has a value
on a symbol bound to nothing and its numbers in range, C3's sets of
closures and valid values, C2's kinds on terms, C5's exceptions, P1's
primitives, V3's lifting, C0's well-formedness, N1's names, V2's
normalisations, and the types, T1's to T6's. This note says what
applying its edits to the model cost each owner there, and what
changed in the prototype's own two files. Paths are in
`middle_end/flambda2/` unless they say otherwise. The counts are on
fcd6e9c15c. Of what came to the integration branch after C3's epoch,
eleven files needed an edit (two of C2's, one of T6's, one of N3's,
one of P1's, three of V3's, one of V2's, one of T7's and one more
function of C3's, below); the rest did not (T2's typing environments,
T3's aliases, T4's meet, T5's joins, N1's renamings, N2's binders,
N3's rewrites up to renaming, C5's exceptions and its second machine,
more of C1's laws, V3's known symbols).

## The edits to existing files, by owner

The prototype's edits are: a value `Value_location` and the types
`location`, `cell` and `heap` in `FlambdaValueSpec.v`; the two
primitives that only the heap has, under one new constructor `Heap` of
`prim` in `FlambdaPrimitiveModel.v`; and the kind of their results and
the test of their numbers in `FlambdaPrimitiveSpec.v`. Everything else
in the table is a repair that they force: a case where a match or a
proof by cases had none for the new constructor. "+a -b" counts lines
added and removed.

| File | Owner | Lines | What changes |
|------|-------|-------|--------------|
| `rocq/semantics/FlambdaValueSpec.v` | C0 | +29 -1 | `location`; `Value_location` in `value`; `cell`, `heap`, `empty_heap`, `cell_of`. No observation changes: each has a wildcard, so a location is no closure, block, boxed number or string, and is not allocated. |
| `rocq/terms/FlambdaPrimitiveModel.v` | C1 (P1 adds to it) | +30 -1 | `cell_mutability`, `init_or_assign`, `heap_primitive` (two constructors, `Make_block_with_identity` and `Block_set`), and `Heap` in `prim`. `mutability`, `equality_comparison`, `Block_load` with its mutability and `Phys_equal` are P1's, and are used as they are. |
| `rocq/semantics/FlambdaPrimitiveSpec.v` | C1 (the signature of `eval_prim` and the laws are C0's; neither changes) | +27 -1 | `result_kind_of_heap_primitive`, and a case of `prim_result_kind`. `access_in_range` and `heap_primitive_in_range`, and a case of `prim_in_range`: a primitive of the heap carries the numbers of the `Make_block` or the access that it stands for. `value_in_range`: a location is in range (it holds no number). `eval_prim` has a wildcard already: it has no value on `Heap _`, and treats a location as any value that no observation reads. |
| `rocq/semantics/FlambdaPrimitiveProofs.v` | C1 | +8 -6 | `eval_prim_arity`: a fourth branch (`Heap`: no value). `eval_prim_static_values`, `eval_prim_statics_extend`: a fourth pattern in the destruct on `prim`. `eval_is_null_statics_extend`: a seventh pattern in the destruct on `value`. |
| `rocq/semantics/FlambdaSemanticsProofs.v` | C0 | +3 -3 | `closure_in_set_of`, `closure_in_set_slot`, `project_function_slot_defined`: one more name in the pattern of the destruct on `value`; the existing catch-all branches take it. |
| `rocq/semantics/FlambdaClosuresSpec.v` | C3 | +5 -2 | `value_valid`: a location is valid (there is nothing in it). `value_sets_are`: a location holds no closure, so the property of the closures in a value holds of it. |
| `rocq/semantics/FlambdaClosuresProofs.v` | C3 | +6 -6 | Six destructs on `value` (among them `closure_in_set_valid`): one more name in the pattern; the existing catch-all branches take it. |
| `rocq/validation/names/FlambdaValidatorEquivarianceProofs.v` | N3 | +1 -1 | `block_made_equivariant`: a fourth pattern in the destruct on `prim`. |
| `rocq/rewrites/lets/FlambdaBlockLoadProofs.v` | V2 | +2 -2 | `block_made_spec`, `unary_applied_spec`: a fourth pattern in the destruct on `prim`. |
| `rocq/rewrites/lets/FlambdaIdentitiesProofs.v` | P1 | +2 -2 | `binary_applied_spec`, `identity_named_equivariant`: a fourth pattern in the destruct on `prim`. |
| `rocq/rewrites/lifting/FlambdaLiftProofs.v` | V3 | +4 -4 | `lift_named_lifted`, `lift_named_no_code`, `lift_named_rename`: a fourth pattern in the destruct on `prim`; in the first, one more pattern in a destruct on `value`. |
| `rocq/rewrites/closures/FlambdaMyClosureProofs.v` | V3 | +5 -4 | `value_free` (a definition of the proofs: a value free of a code id): a location is free of it, as a constant is. `closure_of_free`, `closure_in_set_free`, `project_function_slot_free`: one more name in the pattern of the destruct on `value`. |
| `rocq/rewrites/closures/FlambdaUnusedSlotsProofs.v` | V3 | +4 -2 | `closure_of_drop`, `closure_in_set_drop`: one more name in the pattern of the destruct on `value`. |
| `rocq/rewrites/statics/FlambdaStaticsOrderProofs.v` | V2 | +2 -1 | `value_avoids_nothing`: one more name in the pattern of the destruct on `value`. C0 made this repair when it merged the series: the file came to the integration branch after the series was cut. |
| `rocq/rewrites/statics/FlambdaUnusedStaticsSpec.v` | V1 (unassigned) | +1 | `value_avoids`: a location avoids every name. |
| `rocq/rewrites/statics/FlambdaUnusedStaticsProofs.v` | V1 (unassigned) | +6 -3 | `closure_of_erased`, `closure_in_set_erased`, `project_function_slot_erased`: one more name in the pattern, and one more branch, that of a boxed number (no closure, nothing erased). |
| `rocq/types/grammar/FlambdaTypeMeaningBoolSpec.v` | T1 | +1 | `value_eqb`: two locations are equal when they are the same location. Without it `value_eqb_decides_equality_statement` would be false: a location would not be equal to itself. |
| `rocq/types/grammar/FlambdaTypeMeaningBoolProofs.v` | T1 | +11 -7 | `value_induction`: a hypothesis and a case for a location. `value_eqb_eq`: a branch for two locations. Three destructs on `value` (`is_poison_b_eq`, `relation_holds_b_eq`, `naked_numbers_satisfied_b_eq`): one more pattern. |
| `rocq/types/grammar/FlambdaTypeMeaningProofs.v` | T1 | +13 -13 | Ten destructs on `value`: one more pattern, and one more empty branch where the destruct lists the branches. |
| `rocq/wellformed/FlambdaWellKindedSpec.v` | C2 | +3 -1 | `value_well_kinded`: a location is well kinded (there is nothing in it; the fields of its cell are for the heap's own condition, when the heap is in `state`). |
| `rocq/wellformed/FlambdaWellKindedProofs.v` | C2 | +3 -3 | Three destructs on `value`: one more name in the pattern; the existing catch-all branches take it. |
| `rocq/types/provers/FlambdaProversProofs.v` | T6 | +3 -3 | `tagged_immediate_of_value`, `eval_is_null_cases`: one more pattern in three destructs on `value`. |
| `rocq/Makefile` | C0 | +2 | `FlambdaHeapProofs.v` and `FlambdaHeapSpec.v` in `FLAMBDA2` (the commit of lines for the shared lists). |
| `tests/tools/flsemantics.ml` | C0 (the tool stays C0's for now) | +6 | An observation `Location`, printed `<location n>`; a case of `print_prim` for `Heap`, which the converter never makes. No expected output changes. |
| `tests/semantics/types/types_difftest.ml` | T7 | +5 -3 | Four matches on a value of the model. A location is no tagged immediate and not the contents of a boxed number, so the generator of values for a type makes none from it; it has no parts; it is printed `<location n>`. The tests make no location, and no expected output changes. |

Nothing else matches on `value` or `prim` without a wildcard. T1's
meaning of types needed no case beyond `value_eqb`: it reads a value
through the observations and its kind, so a location satisfies what any
value of kind `Value` that is not null satisfies (an unknown head), and
no shape of variant, block, boxed number, closure or string.

## Statements of existing `Spec` files whose text changed

None.

## What changed in `FlambdaHeapSpec.v`

The machine (`heap_step`, `exec_named`, `heap_call_function`,
`heap_apply_cont_value`) is unchanged: `step`, `call_function` and
`apply_cont_value` are as they were on the old base, and the new forms
of `named` reach the heap machine through `eval_named`, which
`exec_named` calls on everything that is not a primitive.
`heap_step_agrees` is proved as before. On C2's epoch the machine
follows `step` again: `heap_step` ignores the kind of a `Let`,
`heap_apply_cont_value` the metadata of a handler, and
`heap_call_function` takes the two arities of the call and is undefined
unless the call agrees with the code called, as `call_function` is.
Two related lists of arguments have the same kinds, so the relation on
states is kept. Each statement about a `Let` has the kind of its
variable as one more variable. `exec_prim` was not changed by
that: its test `forallb (argument_bound statics h) vs` is as it was,
and `eval_prim` takes the same arguments.

On C5's epoch the machine follows `step` once more. `heap_step`
evaluates the extra arguments of the exception continuation of a call
and gives the callee that continuation with them
(`FlambdaSemanticsSpec.with_extra_args`); `heap_apply_cont_value` is a
`Fixpoint`, as `apply_cont_value` is, and applies a continuation with
extra arguments to the arguments followed by them. A trap action is
not executed, as in `step`. `cont_value_closed`, `cont_related` and
`cont_code_free_of` have a case for a continuation with extra
arguments: its extra arguments are closed, and related to those of the
continuation it is related to; the handlers it holds are those of the
continuation inside. No statement changes text. In the proofs, what
was said of the application of a continuation by cases is now three
lemmas by induction on the continuation (the heap is left as it is;
the state after is closed; its handlers name no symbol left out), and
the simulation has the case.

On P1's primitives the prototype loses what P1 defines, and keeps what
it says of a location. Gone: its `mutability` and
`equality_comparison`; `Phys_equal` and `Block_load_mutable` under
`Heap`; `immediate_or_null` and its own `eval_phys_equal`. Two
primitives are left under `Heap`, `Make_block_with_identity` and
`Block_set`. `exec_bound_prim` has three cases before `eval_prim`: a
primitive under `Heap`; a unary primitive of a location, which reads
the cell (`exec_unary_on_cell`); and `Phys_equal` with a location
among its two arguments (`eval_phys_equal_location`). What changed in
what the machine does:

- A `Block_load` of a location reads the field where its mark allows
  (`load_allowed`): a load marked `Mutable` reads any cell, and one
  marked `Immutable` or `Immutable_unique` a cell made
  `Immutable_unique`. That is what the two constructors of before
  did; the load marked `Immutable_unique` is new, and is treated as
  the one marked `Immutable`. A `Block_load` of a value that is not a
  location is P1's, whatever its mark.
- `exec_unary_on_cell` ends in a wildcard: P1's unary primitives on
  numbers and on strings have no value on a location.
- `Phys_equal` without a location is P1's `eval_phys_equal`, which has
  no value on null. The prototype's own had one: two nulls were
  equal, and null differed from an allocated value and from a
  location. Now a location against null has no value either. This
  narrows what is defined, and follows the CR that is in the tree
  (the request, section 2.6 and decision 3).
- `coeffects_of_prim`: the load with coeffects is `Block_load` marked
  `Mutable`. `effects_of_prim` has no effects on every unary primitive
  but `Opaque_identity` and `Box_number`, and on every binary one,
  which is what the compiler says of P1's (`String_or_bigstring_load`
  here is a load from a string, which is immutable).

Statements: `load_known_field_statement` is now of a load with any
mark, where it was of the load marked `Immutable`; a load from an
immutable block does not look at its mark. No other statement changed
text. In the proofs, what was proved of each primitive of `eval_prim`
by cases now comes from its laws: `eval_prim_related` (V3's: no
primitive tells apart values that the observations read alike) for
related arguments, and `eval_prim_preserves` for closedness. So no
proof lists the unary or the binary primitives, and one that P1 adds
needs no case in the heap's files unless it reads a cell.

Definitions:

- `named_at_most_generative_effects`, `named_has_no_effects`: the list
  of what is not a primitive gains `Closure_of_set`,
  `Project_value_slot_from` and `Project_function_slot` (true, as for
  everything that is not a primitive).
- `locations_of_value`: a `Value_closure_of_set` has the locations of
  its value slots, as a `Value_closure` has.
- `values_of_static`: a `Static_closure_of_set` holds the values of its
  value slots, as a `Static_closure` does.
- `value_step`, the rule for closures. Old:

      Step_closure v v' code slots slots' :
        closure_of statics v = Some (code, slots) ->
        closure_of statics' v' = Some (code, slots') ->
        map fst slots = map fst slots' ->
        Forall2 R (map snd slots) (map snd slots') ->
        value_step R v v'

  New:

      Step_closure v v' slot function_decls slots slots' :
        closure_in_set statics v = Some (slot, function_decls, slots) ->
        closure_in_set statics' v' = Some (slot, function_decls, slots') ->
        map fst slots = map fst slots' ->
        Forall2 R (map snd slots) (map snd slots') ->
        (set_bound statics v ∧ set_bound statics' v')
        ∨ ((∀ move_to w,
             project_function_slot statics slot move_to v = Some w ->
             ∃ w', project_function_slot statics' slot move_to v' = Some w'
                   ∧ R w w')
           ∧ (∀ move_to,
               project_function_slot statics slot move_to v = None ->
               project_function_slot statics' slot move_to v' = None)) ->
        value_step R v v'

  The first four premises are the rule asked for: closures at the same
  function slot of sets with the same function declarations, whose
  value slots have the same names and related values. The fifth is
  new; see below. `set_bound statics v` says that the set of `v` is
  known in full: `v` was made by a `Let`, or is a symbol bound to the
  closure of one function, or is a symbol of a static set that names
  a symbol for each of its functions, each bound to the closure of the
  set at its slot (`FlambdaClosuresSpec.static_set_bound`).
- `value_step` gains a tenth rule, `Step_same_closure_of_set`: a
  `Value_closure_of_set` at a function slot that its set has not got is
  itself. No observation reads it, as none reads the boxed number of
  `Step_same_boxed_number`; without the rule a closed outcome that holds
  one would not be related to itself.

Statements: no text changed in the adaptation to the three epochs (one
changed on P1's primitives, above). The 25 statements that the
prototype proved on the old base are proved with the same text, and
so are the three that C4 proved since (`heap_step_closed_statement`,
`exchange_lets_statement`, `cse_adjacent_statement`), which needed only
extra cases. Since the rebase `lift_block` is proved, with three
statements for the relation that it needs (`requests/C4-heap.md`,
section 4.2), and its statement lost a premise that the proof does not
use; `lift_closure`, `lift_closure_of_set`, `related_projections` and
`outcome_equivalent_symmetric` are stated and proved. Thirty-six
statements are proved, and there is no conjecture.

## What surprised me

- A projection of a function slot breaks a relation of closures by
  their contents. From a symbol, `project_function_slot` gives the
  symbol that its static set names for the slot, whatever that symbol
  is bound to (the CR before `project_function_slot`, decision D4); from
  a closure made by a `Let` it gives the closure of the same set at the
  slot. So two closures with the same slot, the same declarations and
  related value slots can have unrelated closures at another slot, and
  the machine, which runs `Project_function_slot`, would not respect
  the relation that the four premises alone define. Hence the fifth
  premise: the closures at the other slots are related. For two sets
  that are known in full this follows from the first four, and nothing
  more is asked; a closure made by a `Let` is then related to a symbol
  of a static set with the same functions and related value slots,
  which is what lifting a set of closures needs. For a static set of
  which nothing is known, the projections themselves are asked to be
  related, from both closures or from neither. A first form of the
  premise compared the two projections by whether they were symbols.
  It was wrong twice over: it related two symbols of a static set whose
  other symbols were bound to different things, and it did not relate a
  closure to its lifted symbol.
- The prototype had not met C1's law: `Opaque_identity` and `Make_block`
  now test `bound_if_symbol`. Two lemmas of the proofs cover it: a
  value related to one that passes the test passes it.
- `Is_int` and `Is_null` of `eval_prim` have no value on a location,
  which `is_allocated` does not count as allocated, as on a symbol bound
  to nothing. The heap machine never asks: `exec_bound_prim` reads a
  unary primitive of a location from its cell (`exec_unary_on_cell`).
  T1's relation `Is_null` answers 0 of a location, as of any value of
  kind `Value` that is not null. The two do not meet today; they will
  when the heap is in `state` and `eval_prim` is asked of a location.
- `FlambdaSemanticsSpec.state_values_all` says where a state holds
  values, and that a new place for them, such as a heap, is added
  there. The prototype's machine is a copy beside `state`, so it does
  not add its heap there; its own invariant (`state_closed`) walks the
  heap and the static environment. Putting the heap in `state` adds the
  cells to `state_values_all`.
- The cost to the other owners is small: twenty-five files, most of
  them a name in a pattern. The work is in the prototype's own proofs,
  which gain the lemmas on closures of a set (a closure of a set is a
  closure; what a closure that is not a symbol projects; a projection
  does not change as the static environment grows) and a case in each
  proof by cases on the relation.

## The heap in `state`: the rehearsals of the sweep

The step after the first is the sweep: `state` gains the heap, and
nothing else changes. It has been rehearsed to the end on 106a2d501a
(a rehearsal, not in this tree): after the script, the rules and the
repairs by hand every proof passes, and `make -C rocq all` and
`make -C rocq assumptions` pass, with no axiom under the statements
of the Flambda 2 libraries. It was then run again on f24bfe1689,
eleven commits later, taking from the rehearsal what it had repaired
("The run of the day", below). The real sweep is run in that way on
the tip of its day, over C0's commit that names the initial state of
a unit. The script is `requests/C4-heap-sweep.py`; the checker with
the rules is `requests/C4-heap-sweep-check.py`; each says at its head
how it is run.

### The change

In `rocq/semantics/FlambdaSemanticsSpec.v`: `state` has a fifth field,
`state_heap : heap`, last. `Exit k vs hp` and `Exits k vs hp` carry
the heap, last. `call_code`, `run_call`, `apply_cont_value`,
`apply_continuation` and `call_function` take it as their last
argument, and `step` gives every state it builds the heap of the state
it was given. `behaves s (Exits k vs hp)` is `∃ fuel, run fuel s =
Exit k vs hp`, so `refines` and `equivalent` compare the heap at an
exit, exactly. No term makes a cell, so the heap of a run is the heap
it started with. `state_values_all` is not changed.

### What was decided where nothing said

1. The heap is named `hp` in every statement. `h` is a handler in
   some four hundred places of the proofs.
2. A statement gains one binder, `(hp : heap)`, after the binder of
   the static environment beside which the heap is first used, and
   the two states of a statement share it, also where their static
   environments differ. With a heap for each side the statement would
   be false: exits are compared by their heaps.
3. `Exits` carries the heap, as `Exit` does.
4. A state that an example builds has `empty_heap`, and so has its
   exit. C0's definition of the initial state will carry it.
5. The functions of `Proofs` files that compare two results (`ended`,
   `result_related` and `result_simulated` of
   `FlambdaSemanticsProofs.v`, `related` of `FlambdaTrapProofs.v`)
   compare the heaps of two exits, as they compare their values.
6. C5's machine with a stack follows: `Stack_exit` carries the heap,
   and `unwind`, `stack_enter`, `stack_call`, `stack_run_call` and
   `stack_apply_cont` take it.
7. N1's `rename_state` builds the state with the heap it was given: a
   cell holds values, and values hold no names.
8. The heap's own machine stays beside `step` until `step` runs the
   primitives of the heap. `forget_heap` keeps the heap, and
   `heap_step_agrees_statement` changes by one pattern.
9. A section whose statements build states from a `Variable` that is
   a static environment gains a variable for the heap.
10. `state_values_all` gains the cells with the primitives, not here:
    no cell exists before, and C0 asked for a commit that adds the
    component and nothing else.
11. A quantifier inside a statement that binds a static environment
    of its own, under a statement that has the heap, binds a heap of
    its own, `hp'` (two statements of `FlambdaContextWhereSpec.v`;
    N2 and C0 agreed).
12. In a branch of a match on an exit, where the branch compares
    another result with that exit, the exit matched has a heap of its
    own, `hp'`, and the other side keeps the heap of the statement:
    the branch then says that the other run exits with the same
    heap. Three statements of `Proofs` files
    (`fold_step_components`, `apply_cont_aliased`,
    `drop_step_components`; V3 and V4 agreed).
13. A lemma that says that a call or an application of a continuation
    does not exit says it of an exit with any heap (V3's
    `call_code_no_exit`, `run_call_no_exit` and
    `apply_cont_value_closed` of `FlambdaLiftProofs.v`; V3 agreed).
14. A run from the empty heap stays in it. Where a statement builds a
    state on `empty_static_env`, which has `empty_heap`, and an
    existential binds the static environment of the other states that
    it builds, those have `empty_heap` too and the existential binds
    no heap (`head_bound` of `FlambdaLiftFromArmsProofs.v`). An
    existential heap would say less, and the proofs that use the
    lemma could not name it.

### What a script does

The script edits the text of the `.v` files, and compiles nothing. On
106a2d501a it changes 89 files:

- 1,265 applications of `State`, `Exit`, `Exits`, `Stack_exit` and of
  the functions that take the heap gain it: `hp`; `_` in the 44
  patterns that do not use it; `empty_heap` beside
  `empty_static_env` and in the 28 exits and states that examples
  write out;
- 454 statements, lemmas and constructors of relations gain the
  binder, 85 of them in 25 `Spec` files (below);
- 43 first `intros` of a proof gain the name, where the binder went
  into the leading quantifier of the statement;
- 191 explicit applications of a lemma that gained the binder gain
  `_` at its position;
- the eleven definitions that take the heap gain the parameter: the
  five of `FlambdaSemanticsSpec.v`, the five of C5's machine, and
  `rest_valid` of `rocq/simplify/FlambdaSimplifyPruningProofs.v`;
- the three constructors, the field, four comparisons of two exits,
  four patterns that take a state apart, the three exits of decision
  12, and four variables of sections.

After it every statement and every definition of the 378 files
compiles (checked with `rocq compile -vos`, which runs no proof
outside a section). The `Spec` files whose statements gain a binder,
with the number of binders: `FlambdaContextWhereSpec.v` 17,
`FlambdaSemanticsSpec.v` 9, `FlambdaPartialApplicationSpec.v` 9,
`FlambdaWellKindedSpec.v` 6, `FlambdaClosuresSpec.v` 5,
`FlambdaLiftSpec.v` 5, `FlambdaTrapSpec.v` 5,
`FlambdaControlFlowSpec.v` 3, `FlambdaApplySpec.v`,
`FlambdaEquivarianceSpec.v`, `FlambdaInlineFreshSpec.v` and
`FlambdaUnusedStaticsSpec.v` 2 each, `FlambdaAlphaSpec.v`,
`FlambdaCanonicalFreeNamesSpec.v`, `FlambdaKnownClosuresSpec.v`,
`FlambdaKnownSymbolsSpec.v`, `FlambdaRangeSpec.v` and
`FlambdaStaticsInScopeSpec.v` 1 each; in the types library
`FlambdaSimplifyPruningSpec.v`, `FlambdaSimplifySwitchSpec.v`,
`FlambdaSimplifySwitchMergeSpec.v`, `FlambdaTypesSpec.v` and
`FlambdaTypingEnvRefinementSpec.v` 2 each,
`FlambdaEnvAliasesFollowSpec.v` and
`FlambdaSimplifySwitchArmsSpec.v` 1 each. No statement changes
meaning: each says of one more component, which nothing reads, what
it said.

The rehearsals corrected the script six times:

- an application, outside its section, of a lemma of a section that
  gained the variable is left to the rules (16 places): the script
  does not know at which position the lemma takes the heap once the
  section is closed;
- decision 11;
- the definitions that gain the heap as a parameter because they
  build a state from their arguments are found by running the first
  phase again until none is new (`rest_valid`, a definition of a
  `Proofs` file, was not seen before);
- decision 12;
- a sentence that declares several groups of variables of a section,
  one of them the static environment, gains `(hp : heap)` after that
  group. Before, the state that `merging_inv` of
  `rocq/simplify/FlambdaSimplifySwitchMergeProofs.v` builds was given
  `empty_heap`, which made two proofs fail for a good reason;
- decision 14. Before, the existential of `head_bound` bound a heap,
  and its proof and the proof that uses it failed.

The script and the rules leave 131 lines longer than 80 columns. A
tool that is not in this tree breaks each at a space, and the tree is
built after it.

### The edits that are choices

Seven edits are not the script's, because each is a choice. The first
six are exact replacements made after the script; the seventh is among
the repairs by hand. Their owners have agreed to them.

- C0: the tactic `simpl_state` of `rocq/semantics/FlambdaSemanticsProofs.v`
  reduces `state_heap` as it reduces the other four projections.
- C4: four lemmas of `rocq/heap/FlambdaHeapProofs.v` that say that the
  heap's own machine agrees with `step` say it of a state whose heap
  is the heap of that machine.
- C5: the invariant between the machine that goes by names and the
  machine with a stack (`inv` of `rocq/wellformed/FlambdaTrapProofs.v`) says
  that the two states have the same heap.
- V3: decision 13, with the two proofs that use the three lemmas.
- C0 (V1's file): the tactic `rewrite_code_cases` of
  `rocq/rewrites/common/FlambdaNormaliseProofs.v` introduces the heaps of
  `expr_refines` and of `handler_refines_at`. A tactic is not a
  proof: neither the script nor the rules read it.
- C0: `empty_heap` is in the list of what `rocq/validation/checks/FlambdaExtract.v`
  extracts, before `empty_static_env`: the test tool builds the
  initial state of a run with it. The OCaml checks on the tree after
  the sweep showed that it was missing.
- N1: `forget_with_heap` of `rocq/nominal/FlambdaHeapEquivarianceProofs.v`
  takes the premise that the heap given is the heap of the state. A
  state has a heap now, so the lemma is false without it; its two
  uses discharge the premise by computation.

### The proofs

There are 9,242 proofs in the 378 files. After the script and the
first five edits above 364 of them fail, in 73 files. Each proof is
checked on its own (`rocq compile -vok`, with the proofs that fail
admitted in a copy), so that one that fails hides none; a proof of a
section that is not run is admitted with the variables that Rocq says
it uses on the tree before the sweep, so that the lemma is the same
lemma after the section. Rules driven by the error message then edit
a proof that fails and check it again, keeping an edit when the proof
passes, or else when its first error moves beyond what the old error
covered, or else forward. They repair 339 proofs, and
make 562 edits. What the rules are:

- the name of the heap after the name of a static environment, or of
  the values of an exit, in an `intros` or in a pattern that takes a
  relation on states or an exit apart (342 edits);
- the heap, or `_`, as an argument that an explicit application of a
  lemma or of a hypothesis is short of (171);
- `hp`, `empty_heap` or `(state_heap s)` for a `_` that cannot be
  inferred, and `_` for an `hp` that is not in scope (23);
- a third part in a pattern that takes the comparison of two exits
  apart, and the heap of an exit that a proof names (18);
- a binder for the heap in a statement that a proof asserts (8).

25 proofs are left. Two of them (`bind_prim_sound` and
`merge_arms_refines` of
`rocq/simplify/FlambdaSimplifySwitchMergeProofs.v`) failed by the
script's fault and pass since its fifth correction. The other 23 are
repaired by hand. None changes a statement of a `Spec` file. By owner
and file:

- C0, 3. `rocq/semantics/FlambdaContextProofs.v`:
  `related_states_refines_from`, `related_states_refines`,
  `let_static_code_refines_where`. The heap at its place in
  applications by position of lemmas of a section, and in the binders
  of two functions given as arguments.
- C2, 2. `rocq/wellformed/FlambdaWellKindedProofs.v`: `run_exit`,
  `exit_well_kinded_here`. The heap of an exit named in the patterns;
  `(state_heap s)` where a state is read by its projections.
- C3, 1. `rocq/semantics/FlambdaClosuresProofs.v`:
  `project_function_slot_let_closures`. The heap at its place in the
  applications of two lemmas.
- C5, 2. `rocq/wellformed/FlambdaTrapProofs.v`: `step_related` (an assertion is
  stated at the heap of the state, where no variable for it is in
  scope), `unwind_equivariant` (an exit carries its own heap and
  `unwind` goes on with it, so the induction is general in the heap).
- N1, 3. `rocq/nominal/FlambdaHeapEquivarianceProofs.v`: `forget_with_heap`
  (above), `heap_state_of_renamed_expr`.
  `rocq/nominal/FlambdaStateSupportProofs.v`: `state_finitely_supported` (the
  combinator of four arguments is given the constructor of a state
  with its heap fixed). N1 wrote them.
- T6, 4. `rocq/rewrites/unboxing/FlambdaUnboxParamsProofs.v`: `ahead_exit`,
  `ahead_ends`, `lockstep_answers`.
  `rocq/rewrites/unboxing/FlambdaUnboxVariantsProofs.v`: `certain_step_refines`. The
  heap of an exit named in the patterns and given to the witnesses;
  the heap at its place in an `intros`.
- V3 (S5's files), 3. `rocq/rewrites/statics/FlambdaKnownSymbolsProofs.v`:
  `fold_step_components` (the heap at its place in an application of
  a lemma of a section). `rocq/rewrites/lifting/FlambdaLiftProofs.v`: `lifted_step`
  (the heap named where the relation on states is taken apart),
  `lift_simulates` (the heap of an exit named in a pattern and given
  to a witness).
- V4, 5. `rocq/semantics/FlambdaContAlikeProofs.v`: `applies_alike_carries`.
  `rocq/rewrites/loops/FlambdaLoopifyProofs.v`: `loops_step`.
  `rocq/rewrites/params/FlambdaUnusedParamsProofs.v`: `drops_step`,
  `drops_round_expr`, `drops_round`. The heap named in patterns and
  given in applications; a third part where two exits are compared;
  one more component in an existential; in `drops_round` the relation
  is taken apart, since the two states that it relates have one heap.

One more since, on b859e1bdef: T2's `let_refines_value` of
`rocq/simplify/FlambdaSimplifyPassProofs.v`, which is new there. Its
statement has no state, and its proof runs a probe, which now starts
from `empty_heap`; T2 has been told.

The script and the rules as they are now were run once more with
nothing taken from a rehearsal, on 8550ac6ab5 (407 files, 9,826
proofs): the rules make 621 edits in 377 proofs and leave 22 proofs
for a hand, all of them among those above. Two that were by hand are
the rules' now: C3's `project_function_slot_let_closures` and V4's
`applies_alike_carries`.

### The run of the day

The run is made so that what a rehearsal has repaired is not repaired
again:

1. The statements of the tip are compiled, and Rocq is asked which
   variables of its section each proof uses.
2. The script, and the six edits that are choices.
3. A file that the script leaves as it left it in the rehearsal takes
   the text that the rehearsal ended with. In a file that differs, a
   lemma whose text after the script is what it was after the script
   in the rehearsal takes, from its first line to the end of its
   proof, the text that the rehearsal ended with.
4. The statements are compiled, and the rules run on the files that
   differ: there they find the lemmas that are new or changed since
   the rehearsal.
5. In those files, a proof that a rehearsal repaired by hand is given
   its repaired text, if its lemma has not changed since.
6. What still fails is for a hand.
7. The lines are wrapped, and the tree is built: `make -C rocq all`,
   `make -C rocq assumptions`, then the OCaml checks, with C0's two
   edits to the test tool and the extracted files generated again.

The tools of steps 3 and 5 are not in this tree; they compare texts
and copy them. Each run is the rehearsal of the next.

Measured, on a machine of 64 cores:

- On 106a2d501a, without step 3: 77 minutes before the repairs by
  hand (the statements twice, 200 s and 178 s; the variables, 54 s;
  the script and the edits, 55 s; the rules, 69 minutes), then 23
  repairs by hand, eight of them new in that rehearsal.
- On f24bfe1689, with step 3 by file only: 25 of the 378 files had
  changed since the rehearsal. Steps 1 to 6 took 16 minutes 35
  seconds and left nothing for a hand.
- On 23daac59dc, after C3's tupled functions, with step 3 by file
  only: 42 of the 382 files had changed, large files of the core
  among them. Steps 1 to 6 took 43 minutes 49 seconds, 31 of them
  the rules going again over every proof of those files, and left
  nothing for a hand. Step 3 by lemma was made for this case.
- On 8a5eb534ec, with step 3 as above, on a machine that nothing
  else was using: 11 of the 386 files had changed. Steps 1 to 6 took
  7 minutes 28 seconds (the statements twice, 176 s and 124 s; the
  variables, 53 s; the script and the edits, 40 s; step 3, 15 s; the
  rules on the 11 files, 31 s; the last check of them, 7 s) and left
  nothing for a hand. `make -C rocq all` then took 199 s and
  `make -C rocq assumptions` 234 s; both pass. With C0's two edits to
  the test tool, `make -s boot-compiler`, the tests of
  `tests/semantics/` and `make -s fmt` pass on that tree, and no
  expected output changes.
- On b859e1bdef: 22 of the 394 files had changed. Steps 1 to 6 took
  4 minutes 31 seconds and left one proof for a hand (T2's, above).
  `make -C rocq all` then took 293 s and `make -C rocq assumptions`
  252 s; both pass.
- On 60a8f698b6: 16 of the 398 files had changed. Steps 1 to 6 took
  5 minutes and left two proofs of a new file, which failed by the
  script's fault and pass since its sixth correction.

The builds are slower when other work uses the machine: on
f24bfe1689 and 23daac59dc `make -C rocq all` took about four minutes
and `make -C rocq assumptions` about twelve.

So the cost of the day is about a quarter of an hour, on a machine
that is free, from a clean tip to a tree of Rocq that builds, then
the OCaml checks, and a hand for what the lemmas that are new since
the last rehearsal leave. The rehearsal is run again as the tip
moves, so that those are few.

### The test tool and the extracted files

`rocq/validation/checks/FlambdaExtract.v` extracts `empty_heap` (the edits that are
choices). `tests/tools/flsemantics.ml` changes in two places: the
pattern of an exit has a third component, and the record of the
initial state has `state_heap = Values.empty_heap`. The extracted
files are generated again: `FlambdaSemanticsSpec` and
`FlambdaValueSpec`, each an `.ml` and an `.mli`. No expected output
changes: the heap of every run is empty.
