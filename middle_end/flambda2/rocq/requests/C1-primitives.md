# C1, primitives as data: core change request

Stream C1 of `PARALLEL-PLAN.md` (section 6). This is the request of
section 7 ("Core change requests"): the diff to the `Spec` files, why,
which statements change meaning, the fallout, and the plan for the epoch.

The prototype was made on the integration branch at fbcbbade2c. It was
whole: every `.v` file built, `make -C rocq all` and `make -C rocq
assumptions` passed, and the semantics tests passed. The change is on
the integration branch, from 2c6b618546 to 5ed48f2c6a: the epoch, the
fourth law and the second and third stages. Read the `Spec` diff there:

    git diff 4dfea64957 5ed48f2c6a -- \
      'middle_end/flambda2/rocq/*Spec.v' 'middle_end/flambda2/rocq/*Model.v'

Paths below are in `middle_end/flambda2/rocq/` unless they say otherwise.

## Summary

`Parameter prim` becomes an inductive type shaped like the compiler's
`Flambda_primitive.Without_args.t`, and `Parameter eval_prim` a
definition. The seven axioms of the ledger's section "Flambda 2" go, and
with them the hand-written evaluator of the test tool: the statements of
the Flambda 2 libraries then depend on no axiom at all.

`step` does not change. The change to the machine is one call, in
`eval_named`: `eval_prim p vs` becomes `eval_prim statics p vs`, as a
primitive that reads an allocated value has to look through a symbol.
`value` gains two constructors.

The rest of the model does not look inside a primitive. It uses three
laws that hold of every primitive, and `eval_prim` is opaque outside its
own files, as `step` is. So a primitive added later is a constructor, a
case of the evaluation and of the kinds, and a case of the proofs of the
three laws, all in three files that only this stream owns. The second
and third stages below were done that way, as a check: no other `.v`
file changed.

## 1. The change to the `Spec` files

### New files

- `FlambdaKindModel.v`: `naked_number_kind` and `flambda_kind`, moved as
  they are from `FlambdaModel.v`; `kind_eq_dec`, moved as it is from
  `FlambdaCompareSpec.v`; and, new: `kind_eqb`, `standard_int`
  (`Flambda_kind.Standard_int.t`) with `kind_of_standard_int`, and
  `boxable_number` (`Flambda_kind.Boxable_number.t`) with `unboxed_kind`
  and its converse `boxable_number_of_kind`.
- `FlambdaPrimitiveModel.v`: the types of the primitives (section 2).
  It must come before `named`, hence before `FlambdaModel.v`, and it
  needs the kinds, hence the file above.
- `FlambdaValueSpec.v`: `value` (with its two new constructors),
  `statically_allocated_value`, `static_env`, `empty_static_env` and
  `closure_of`, moved from `FlambdaSemanticsSpec.v`; and, new:
  `kind_of_value`, `block_of`, `boxed_number_of`, `string_of`,
  `is_allocated`, `boxable_number_of_const` (section 3), and the
  statement `observations_disjoint_statement`.
- `FlambdaValueProofs.v`: the proof of that statement. Not a `Spec`.
- `FlambdaPrimitiveSpec.v`: `eval_prim` and what it is made of, the
  kinds of the primitives, `prim_eq_dec` and `prim_eqb`, and the three
  laws. It must come after the values and before `eval_named`, hence
  the file above.
- `FlambdaPrimitiveProofs.v`: the proofs. Not a `Spec`.

`FlambdaModel.v` exports the first two and `FlambdaSemanticsSpec.v` the
other two `Spec` files (`Require Export`), and `FlambdaSemanticsProofs.v`
exports `FlambdaValueProofs.v` and `FlambdaPrimitiveProofs.v`. So no
file that imports them needs
a new import, and no definition that moved changes its unqualified name.

### The diff to existing definitions

`FlambdaModel.v`:

    -Parameter prim : Type.

(and the two kind types leave for `FlambdaKindModel.v`).

`FlambdaSemanticsSpec.v`, besides what moves out:

    -Parameter eval_prim : prim -> list value -> option value.
     ...
       | Prim p args =>
         let* vs := eval_simples vars args in
    -    eval_prim p vs
    +    eval_prim statics p vs

`value`, now in `FlambdaValueSpec.v`:

     Inductive value : Type :=
       | Value_const (c : const)
       | Value_closure (code : code_id)
           (value_slots : list (value_slot * value))
    -  | Value_symbol (s : symbol).
    +  | Value_symbol (s : symbol)
    +  | Value_block (tag : Z) (fields : list value)
    +  | Value_boxed_number (contents : const).

`closure_of`: its last case, `| Value_const _ => None`, becomes
`| _ => None`, so that it answers `None` on the new constructors.

`FlambdaCompareSpec.v`:

    -Parameter prim_eqb : prim -> prim -> bool.
    -Axiom prim_eqb_sound : ∀ p p' : prim, prim_eqb p p' = true -> p = p'.
    -Axiom prim_eqb_refl : ∀ p : prim, prim_eqb p p = true.

`prim_eqb` is now `if prim_eq_dec p p' then true else false`, in
`FlambdaPrimitiveSpec.v`; the two axioms are lemmas of the same names in
`FlambdaPrimitiveProofs.v`.

`FlambdaNormaliseSpec.v`:

    -Parameter prim_may_fold : prim -> bool.
    +Definition prim_may_fold (p : prim) : bool :=
    +  match p with
    +  | Unary (Opaque_identity _) => false
    +  | _ => true
    +  end.
     ...
    -      match eval_prim p (map Value_const cs) with
    +      match eval_prim empty_static_env p (map Value_const cs) with
           | Some (Value_const c) => Simple (Const c)
    -      | Some (Value_closure _ _ | Value_symbol _) | None => d
    +      | _ => d
           end

`FlambdaUnusedStaticsSpec.v`:

       | Value_closure code slots =>
         code_kept D code && forallb (fun '(_, v) => value_avoids D v) slots
    +  | Value_block _ fields => forallb (value_avoids D) fields
    +  | Value_boxed_number _ => true
       end.
    -Axiom eval_prim_avoids : ...

`eval_prim_avoids` is a lemma of `FlambdaUnusedStaticsProofs.v`, an
instance of the third law below.

No other `Spec` file changes, in either library.

### The three laws

In `FlambdaPrimitiveSpec.v`, proved in `FlambdaPrimitiveProofs.v`.

    Definition eval_prim_static_values_statement : Prop :=
      ∀ (statics statics' : static_env) (p : prim) (vs : list value),
        (∀ s : symbol,
          Value_symbol s ∈ vs ->
          lookup s (static_values statics)
          = lookup s (static_values statics')) ->
        eval_prim statics p vs = eval_prim statics' p vs.

    Definition eval_prim_kinds_statement : Prop :=
      ∀ (statics : static_env) (p : prim) (vs : list value) (v : value),
        eval_prim statics p vs = Some v ->
        prim_args_well_kinded p (map kind_of_value vs) = true
        ∧ kind_of_value v = prim_result_kind p.

    Definition eval_prim_preserves_statement : Prop :=
      ∀ (P : value -> Prop) (statics : static_env) (p : prim)
        (vs : list value) (v : value),
        holds_of_numbers P ->
        holds_of_blocks P ->
        holds_in_static_blocks statics P ->
        (∀ arg : value, arg ∈ vs -> P arg) ->
        eval_prim statics p vs = Some v ->
        P v.

`holds_of_numbers P`: `P` holds of every `Value_const c` and
`Value_boxed_number c`. `holds_of_blocks P`: `P (Value_block tag fields)`
exactly when `P` holds of every field. `holds_in_static_blocks statics
P`: if `P (Value_symbol s)` and `s` is bound to `Static_block tag
fields`, `P` holds of every field.

What each is for:

- The first is what a proof needs where two runs have different static
  environments: related code (`FlambdaContextProofs`), code compared by
  `code_eqb` (`FlambdaCompareProofs`), bindings erased
  (`FlambdaUnusedStaticsProofs`), and none at all (`fold_named`, on
  constants).
- The second is for C2's preservation of well-kindedness, and says that
  a primitive applied to arguments of the wrong kinds or in the wrong
  number has no value, which Simplify assumes
  (`Simplify_primitive.arg_kind_mismatch` gives `Invalid`).
- The third says that a primitive makes its value out of its arguments.
  `value_avoids D` is such a `P`, which gives `eval_prim_avoids`; C2's
  well-kindedness of values is to be another.

A fourth law, added after C0 read this request, for the meaning of
types (T1): a primitive that has a value keeps it when the static
environment grows.

    Definition eval_prim_statics_extend_statement : Prop :=
      ∀ (statics statics' : static_env) (p : prim) (vs : list value)
        (v : value),
        (∀ (s : symbol) (allocated : statically_allocated_value),
          Value_symbol s ∈ vs ->
          lookup s (static_values statics) = Some allocated ->
          lookup s (static_values statics') = Some allocated) ->
        eval_prim statics p vs = Some v ->
        eval_prim statics' p vs = Some v.

The first law does not give it, as it asks of a symbol that is not
bound that it stay so. It holds because no primitive has a value by a
symbol's being bound to nothing; the option recommended in decision 2
keeps it. The premise is that of `FlambdaWellFormedSpec.statics_extend`,
asked of the symbols among the arguments only. It is not part of the
epoch: a leaf commit after it (47f3819f23).

### How adding a primitive is leaf work

1. The types and the evaluation are in files of their own, which C0's
   files import and never change for a primitive.
2. Nothing outside those files matches on `prim`, or unfolds
   `eval_prim`: it is opaque after `FlambdaPrimitiveProofs.v`. The only
   uses elsewhere are the three laws, `prim_eqb` (derived by `decide
   equality`, so it follows the type) and `prim_may_fold` (a wildcard:
   a new primitive may be folded and removed when unused, which is sound
   whatever it computes).
3. Each proof in `FlambdaPrimitiveProofs.v` that goes by cases on the
   primitives says so. A new primitive adds a case to each or the file
   does not build; nothing else can break.

Checked: the second stage (1301aa5104, a9cb7aaac8) and the third
(c8c81254fd, 5ed48f2c6a, which adds a third arity) change
`FlambdaPrimitiveModel.v`, `FlambdaPrimitiveSpec.v`,
`FlambdaPrimitiveProofs.v`, the converter and examples (with the
extracted files regenerated), and nothing else.

## 2. The primitives, by stage

A primitive application stays `Prim (p : prim) (args : list simple)`.

    Inductive prim : Type :=
      | Unary (p : unary_primitive)
      | Binary (p : binary_primitive)
      | Variadic (p : variadic_primitive).

The constructors of the three types have the compiler's names. The
result of every primitive that answers a boolean or a tag is a naked
immediate. "No value" means `eval_prim` is `None`, so the term is
undefined. Every primitive has no value on the wrong number of
arguments, and on an argument that is not of its argument kind; the
lists give what else.

**Stage 1** (what the examples use; removes the axioms and the
evaluator). All unary but the last two.

- `Tag_immediate`, naked immediate to value: the tagged immediate of the
  same integer. No value on an integer out of 63 bits.
- `Untag_immediate`, value to naked immediate: the converse. No value on
  what is not a tagged immediate in range.
- `Is_int variant_only`, value to naked immediate: 1 on a tagged
  immediate, 0 on a block and, without `variant_only`, 0 on anything
  else allocated. No value: with `variant_only`, on closures, boxed
  numbers and strings; always, on null, poison and an unbound symbol.
- `Is_null`, value to naked immediate: 1 on null, 0 on a tagged
  immediate and on anything allocated. No value on poison and on an
  unbound symbol.
- `Get_tag`, value to naked immediate: the tag of a block. No value on
  what is not a block.
- `Opaque_identity kind`, `kind` to `kind`: its argument. No value on a
  value of another kind.
- `Int_arith kind op`, two integers of `kind` to one: `op` is `Add`,
  `Sub`, `Mul`, `Div`, `Mod` (the last two signed or unsigned), `And`,
  `Or` or `Xor`, and the result is wrapped to the width of `kind`. No
  value on an integer out of range, or on a zero divisor.
- `Int_comp kind behaviour`, two integers of `kind` to naked immediate:
  with `Yielding_bool`, 1 or 0 for the compiler's `Eq`, `Neq`, `Lt`,
  `Gt`, `Le`, `Ge` (the last four signed or unsigned; here `Int_eq` and
  so on, as `Eq`, `Lt` and `Gt` are Rocq's); with
  `Yielding_int_like_compare_functions`, -1, 0 or 1. No value on an
  integer out of range.

`kind` is a `standard_int`: tagged immediate and naked immediate (63
bits), naked int8, int16, int32, int64 and nativeint (8, 16, 32, 64 and
64 bits). The model is of a 64-bit target. Division truncates towards
zero, `Mod` has the sign of the dividend, the least integer divided by
minus one wraps to itself, and the unsigned operations are those of the
unsigned readings. This is `Number_adjuncts` and `One_bit_fewer`, read
case by case.

**Stage 2.** Both unary; `kind` is a `boxable_number`. Neither looks at
the number.

- `Box_number kind`, naked number of `kind` to value:
  `Value_boxed_number` of the constant. No value on poison, or on a
  number of another kind.
- `Unbox_number kind`, value to naked number of `kind`: the constant
  inside a boxed number of `kind`, made by `Box_number` or bound to a
  symbol. No value on anything else, a boxed poison included (section 3).

**Stage 3.**

- `Make_block (Block_kind_values tag)`, variadic, values to value:
  `Value_block tag` of the arguments. No value if an argument is not of
  kind `Value`. The compiler's block kind also has a kind for each
  field, with a subkind, and Simplify takes an argument that is a
  variable to be of it, giving `Invalid` where it cannot be
  (`Simplify_variadic_primitive`). The model has no subkinds, so the
  converter takes only the list that says nothing, `any_value` for every
  field, and reports any other as not in the model. The first version
  dropped the list, so that a term Simplify may make `Invalid` could be
  defined in the model; R1 found it (its B1). fexpr builds `any_value`
  only, so no example changes. The tag is any integer, and the block may
  have no fields (decisions 12 and 13).
- `Block_load (Block_access_values tag size field_kind) field`, unary,
  value to value: field `field` of a block, made by `Make_block` or
  bound to a symbol; `tag` and `size` are options, the compiler's
  `Or_unknown.t`. No value on what is not a block; on a block whose tag
  or number of fields is not the one given, where one is given; on a
  field out of bounds; on a field not of kind `Value`; with
  `field_kind = Immediate`, on a field that is not a tagged immediate.

What the model leaves out of these, which the converter must refuse and
not approximate: allocation modes other than the heap, mutability other
than `Immutable`, block kinds and access kinds other than `Values`,
and kinds of fields of a block other than `any_value`. `Opaque_identity`
drops `middle_end_only`.

(Since then the converter refuses fewer of these. For the validation
it no longer refuses an allocation because its mode is `Local`: it
converts it as it converts the same allocation on the heap (decision
D37). Where it converts a term once more with its regions kept, for
the machine of regions, it says that such an allocation is local by a
`Local_allocation` after it. A `Block_load` keeps its mutability. And
a `Make_block` is no longer refused because its mutability is
`Mutable` or `Immutable_unique`. Where such a block has a field, and
passes what is asked of an `Immutable` one (a block of values, with as
many kinds of fields as arguments, every one `any_value`), it is
converted: not to `Make_block`, which is still made of an `Immutable`
block only, but to `Make_block_with_identity`, which makes a cell of
the heap of `FlambdaHeapSpec.v`. The header of
`FlambdaPrimitiveModel.v` says what the model leaves out as it
stands.)

Later, as leaf work, in the order the corpus asks for: `Int_shift`, the
unary `Int_arith`, `Num_conv`, `Boolean_not`, `Phys_equal`,
`String_length`, `Reinterpret_64_bit_word`; floats need a model of IEEE
arithmetic first.

## 3. Values

The machine gains two forms, and no existing one changes:

- `Value_block (tag : Z) (fields : list value)`: an immutable block of
  values, made by `Make_block`.
- `Value_boxed_number (contents : const)`: made by `Box_number`. It
  carries no kind: the kind of the constant gives it.

Tagged immediates, naked numbers of every kind and null are constants,
as today. `statically_allocated_value` does not change. Both forms are
part of the epoch though the primitives that make them come in stages 2
and 3: a new form of value is not leaf work, so they land once.

An allocated value is its contents: two are the same value when their
contents are, nothing tells two allocations apart, and there is no
physical equality.

**Reading.** An allocated value is read through an observation that
takes the static environment, as `closure_of` does: `block_of statics v
: option (Z * list value)`, `boxed_number_of statics v : option const`,
`string_of statics v : option (list Z)`. Each answers the same for a
value made by a primitive and for a symbol bound to the same thing, and
a value is at most one of a closure, a block, a boxed number and a
string: that is `observations_disjoint_statement`, T1's statement word
for word, proved in `FlambdaValueProofs.v`. The primitives use the
observations, and so will the types.

`boxed_number_of` recognises a boxed number by the kind of its contents,
one of the nine naked number kinds that have a boxed form
(`boxable_number_of_kind`), as T1's observation does on the machine of
today. So a symbol bound to a boxed poison of such a kind is a boxed
number, and allocated. The first version went by the constructor of the
constant and answered `None` on that value, which is a value of today's
machine: the move changed an answer, and with it what a type means. R1
found it (its B2); T1 and C1 agreed to keep T1's answer. The primitives
have their own guard, by the constructor: `Unbox_number` has no value on
a boxed poison, nor `Box_number` on a poison constant (decision 11).

**At exit.** Nothing changes: `Exits k vs` holds the values as they are,
and `equivalent` and `refines` compare them with `=`. So a block made by
`Make_block` and a symbol bound to a static block with the same tag and
fields are different values at exit, though no primitive tells them
apart. This is conservative, and it means that lifting a block or a
boxed number to a symbol, which Simplify does, cannot be stated as an
equivalence yet (`block_lifted.fl`). Looking through symbols at exit is
a change to what is observable: decision 1 of section 7.

**Agreed with the other streams** (recorded in `status/C1.md`).

- T1: the two forms above; the observations, with these names and
  types, defined here and used by `type_satisfied`, T1 defining copies
  over today's machine until the epoch and deleting them then;
  `kind_of_value` defined here. The relations that a type records
  (`Is_int`, `Get_tag`, `Is_null` of a name) stay definitions of T1's:
  R1 found that the compiler records `Get_tag` of a value that may be
  an immediate, so a relation is wider than the run of its primitive.
  What ties the two is one direction, which S1 proves from the lemmas
  here: if the primitive has a value, the relation holds of it. That is
  so for the three as defined here, `Is_int` at `variant_only`.
- C3: `Project_value_slot` stays a constructor of `named`, and C3's
  projection of a function slot goes in `named` too. Nothing here
  matches on `Value_closure`: closures are read through `closure_of`
  only. Tried with a sixth constructor of `value`: the primitives' files
  build as they are; what needs a case is `value_avoids` and
  `FlambdaUnusedStaticsProofs.closure_of_erased`. `Phys_equal`, when
  added, has no value on anything allocated.
- C2: the kind of a primitive's result is `prim_result_kind p`, its
  check on the arguments is `prim_args_well_kinded p kinds`, and the
  second law is what preservation uses. `kind_of_value` and `kind_eqb`
  are defined here. Kinds are in `FlambdaKindModel.v`, where C2 adds
  arities in its epoch. No subkinds. C2's invariant on values is an
  instance of the third law.

## 4. Statements that change meaning

No statement is falsified: every proof on the branch is the proof of
the statement it was, and the only statements whose text changes are
the ones listed under 1.

What changes in meaning, for every statement at once:

1. **Which primitives.** Before, a statement held for every
   interpretation of `prim` and `eval_prim` (with `eval_prim_avoids`
   assumed where used). After, it is about the primitives of
   `FlambdaPrimitiveModel.v`. A statement about a term with a primitive
   used to say nothing of what the primitive computes; now it does.
2. **Which values.** Statements quantify over environments, so now also
   over those that hold blocks and boxed numbers. They are stronger for
   it, and they are still proved.
3. **`eval_named` on a primitive reads the static environment.** Before,
   the value of `Prim p args` did not depend on it. Three proofs relied
   on that, and now use the first law: `FlambdaContextProofs.`
   `eval_named_related` (under `plug_refines`),
   `FlambdaCompareProofs.eval_named_match` (under `expr_eqb_sound`),
   `FlambdaUnusedStaticsProofs.eval_named_erased` (under
   `drop_statics_refines`).

Statement by statement, where there is something to say:

- `expr_eqb_sound`. Before: rests on `prim_eqb_sound` and
  `prim_eqb_refl`, axioms about a test that the tool realises. After:
  no axiom; `prim_eqb` is decided equality.
- `drop_statics_refines`, `drop_statics_program_refines`. Before: rest
  on the axiom `eval_prim_avoids`. After: no axiom; the premise
  `envs_free_of` also asks that the fields of blocks refer to no name
  removed. `eval_prim_avoids` is now a lemma of
  `FlambdaUnusedStaticsProofs.v`, and it is not the axiom with a proof
  attached: it has one more premise, `statics_erased statics statics'`,
  which it needs now that a load reaches a field through a symbol (the
  fields of a static block that is kept must refer to nothing removed).
  The two theorems are not weakened by it: their conditions on the
  environments give the premise.
- `fold_constants_equivalent`. Before: folds by whatever `eval_prim` the
  tool installed. After: by the model's, asked under `empty_static_env`.
- `remove_unused_lets_refines`. Before: `prim_may_fold` a parameter.
  After: all but `Opaque_identity`. The proof never depended on the
  choice.
- `validate_unit_sound`, `validate_unit_sound_everywhere`. Before: rest
  on the three axioms above. After: no axiom.

`step`, `run`, `behaves`, `equivalent`, `refines` and the step lemmas
are unchanged.

A check against vacuity: `FlambdaPrimitiveProofs.v` ends with an
example, by computation, of each primitive having a value and of each
way of having none.

## 5. Fallout

Found by building. Sizes are lines changed on the branch for stage 1.

Rocq, existing files. C0's:

- `FlambdaModel.v`: 34 lines out (the kinds, `Parameter prim`), an
  export in.
- `FlambdaSemanticsSpec.v`: 53 lines out (the values, `Parameter
  eval_prim`, `closure_of`), an export in, one call changed.
- `FlambdaSemanticsProofs.v`: 1 line, the export of the proofs about
  values and primitives.
- `FlambdaContextProofs.v`: 4 lines, in `eval_named_related`.
- `FlambdaExtract.v`: four `Extract Constant` out, `Set Extraction
  KeepSingleton` in (so that a type of the model with one constructor of
  one argument is not unboxed in the OCaml).

V1's, which has no owner yet:

- `FlambdaCompareSpec.v`: 13 lines out.
- `FlambdaCompareProofs.v`: 2 lines, in `eval_named_match`.
- `FlambdaNormaliseSpec.v`: about 12 lines (`prim_may_fold`,
  `fold_named`).
- `FlambdaNormaliseProofs.v`: 3 lines, in `fold_named_sound`.
- `FlambdaUnusedStaticsSpec.v`: 2 cases in, the axiom out.
- `FlambdaUnusedStaticsProofs.v`: about 45 lines: `eval_prim_avoids` and
  `eval_prim_erased` as lemmas, 2 branches of `closure_of_erased`.

Nothing else changes: `FlambdaMap*`, `FlambdaWellFormed*`,
`FlambdaControlFlow*`, `FlambdaNominal*`, `FlambdaAlpha*`,
`FlambdaApply*`, `FlambdaStatic*`, `FlambdaValidate*`, and the four
files of `types/rocq/` build as they are.

Rocq, new files, without their licence headers: `FlambdaKindModel.v`
(115 lines, 30 of them moved), `FlambdaValueSpec.v` (175, 40 moved),
`FlambdaValueProofs.v` (20), `FlambdaPrimitiveModel.v` (90 at stage 1,
125 at stage 3), `FlambdaPrimitiveSpec.v` (420, then 530 with the
fourth law), `FlambdaPrimitiveProofs.v` (640, then 1040).

OCaml and the rest:

- `tests/tools/flsemantics.ml` (V2 and V3, no owner yet): the evaluator,
  `check_supported` and the three hooks out, about 110 lines; the
  conversion of primitives in, about 110; constants of the naked
  integer kinds; the observation of blocks and boxed numbers; a printer
  of primitives; `z_of_int` through `int64` (below).
- `tests/semantics/extracted/prim_hook.ml` (C0): deleted. It is dead on
  the branch, which does not commit in that directory.
- `tests/semantics/extracted/` (C0): regenerated. New modules:
  `FlambdaKindModel`, `FlambdaPrimitiveModel`, `FlambdaPrimitiveSpec`,
  `FlambdaValueSpec`, `OxCamlOption`, `NatDef`, and `Nat` at stage 3.
  Its `dune` lists no modules and does not change.
- `rocq/Makefile` (C0): 6 lines in `FLAMBDA2`, and the comment that
  names `prim_hook.ml`.
- `rocq/HACKING.md` (C0): six places ("Primitives are abstract",
  `fold_constants` with `prim_may_fold`, `eval_prim_avoids`,
  `prim_hook.ml` twice, "Trusted and unverified") and the table of
  directories.
- `rocq/TRUST-LEDGER.md` (R1): section 6; done on the branch as a
  proposal.
- `rocq/review-pdf.pl` (C0): nothing needed; its reading order could
  name the new files.
- The 18 `.expected` files (V2): none changes,
  `unsupported_primitive.expected` included, as `Is_flat_float_array`
  is still not modelled.

A bug found on the way, in trusted OCaml that exists today: `z_of_int`
negated its argument as an `int`, so the least tagged immediate was
converted to -1. The branch fixes it. The comparison of section 8 found
it, and R1 reproduced it: the least immediate and -1 both became -1,
in the conversion of terms and, in the old evaluator, of results. What
the tool reported before the repair is not made good by it: a
validation of a term with that constant could have rested on the
collision. None of the eighteen examples has the constant.

Two places where the converter approximated, against what its header
and the ledger say of it, both repaired on the branch after R1's
review: the kinds of the fields of `Make_block`, which it dropped
(section 2), and the shape of a static block, which it dropped before
this branch, so that a block with a mixed shape was read as a block of
values. Both are now refused.

## 6. Trust

Leaves the ledger: all seven axioms of the section "Flambda 2" (`prim`,
`eval_prim`, `prim_eqb`, `prim_eqb_sound`, `prim_eqb_refl`,
`prim_may_fold`, `eval_prim_avoids`), and of the trusted OCaml "The
meaning of the primitives" and "The equality of primitives". "The
extraction" no longer has a directive that realises a definition by
hand-written OCaml.

Enters: no axiom and no `Parameter`. One item of trusted OCaml in place
of the two that go: "The conversion of primitives" (about 145 lines at
stage 3: the unary, binary and variadic conversions and those of what a
primitive carries), a renaming of constructors that must send a
primitive to the one of the same name and refuse what the model has not
got, a kind of a field of a block that says more than `any_value`
included. What a primitive computes is no longer trusted OCaml; it is
`eval_prim`, a reviewed definition.

The ledger also gets an item for something that was trusted all along
without being named: "The conversion of numeric constants" (about 75
lines at stage 1, 105 at stage 3), trusted to convert each number to
itself at every width and at its bounds, which `z_of_int` did not. The
tool refuses constants of the vector and mask kinds, and poison: the
model has boxing and unboxing at those kinds, and the tool does not
exercise them.

Stays: the conversion of terms, the numbering and identification of
names, what the tool reports.

After the change `make -C rocq assumptions` reports 167 axioms (174
before), and "the statements of the Flambda 2 libraries depend on 0 of
the 0 axioms of the ledger's section". The integers are computed on `Z`
with an explicit wrap, so that no statement comes to depend on Rocq's
primitive integers or on the axioms of `Int64`; lemmas of
`FlambdaPrimitiveProofs.v`, outside every statement, tie them to the
models of `rocq/`: at 64 bits `wrap`, `unsigned` and `in_range` are
`wrap64`, `unsigned64` and `fits_int64` by definition, and at 63 bits
`+`, `-` and `*` of `Sint63` are the wrapped ones.

What is newly to be reviewed by inspection: `eval_prim` against
`flambda_primitive.mli` and `simplify_*_primitive.ml`. It is the
definition every statement about a term with a primitive now rests on.

## 7. Decisions for the human

What is observable and what is undefined behaviour are the human's
(plan, section 7). In each case the prototype takes the choice marked
"now", so none blocks the epoch. Where the compiler settles the
question the note says how.

Most of the choices marked "now" are "no value". That is conservative
in one sense only: the model never invents a result. It is not
conservative about what is proved of the compiler. A term with no value
is undefined, a refinement has nothing to show of an undefined source,
and so each "no value" takes sources out of what the theorems and the
tool check. Where the machine would compute something there, and
Simplify does not assume the case away, a "no value" hides a part of
the compiler's correctness instead of establishing it.

Each place where the model follows the compiler by making something
undefined, or parts from it, is recorded as a discrepancy
(`rocq/HACKING.md`, "Discrepancies with the compiler"); the numbers C1-1
to C1-13 below are those entries, and
R1-2 to R1-12 the reviewer's, which cite the compiler files and give
terms that show each.

1. **Allocated values at exit.** Now: compared as they are, so a block
   made by `Make_block` differs from a symbol bound to an equal static
   block. Option: observation at exit looks through symbols, values
   being compared up to what symbols are bound to. Recommendation: the
   option, decided with the heap (C4), which has the same question for
   heap addresses. Until then lifting is not validated (C1-8).
2. **`Is_int` and `Is_null` on null, poison and unbound symbols.** Now:
   no value on poison and on a symbol bound to nothing; `Is_int` has no
   value on null. Settled by the compiler, for `Is_int` with
   `variant_only` (C1-4): Simplify takes its argument to be a variant
   that is not null. `prove_is_int_generic_value` answers `Invalid` on
   closures, boxed numbers, strings and arrays, and after such an
   `Is_int` of a variable Simplify folds `Is_null` of it to 0. The
   machine would compute the low bit of anything, so the model has to
   give no value there, null included. Without `variant_only` the
   prover answers `Proved false` on those, and the model follows. Open:
   - A symbol bound to nothing (one of another unit, in a term run on
     its own). Option: 0 for `Is_null` and for `Is_int` without
     `variant_only`, as a symbol is a pointer whatever it is bound to.
     Recommendation: that option, which is T1's too; the types are sound
     either way.
   - Poison (C1-5, and R1's R1-2). Simplify folds `Is_null` of a poison
     constant to 0 and `Is_int` of one to 1, and the backend compiles a
     poison of kind value to a tagged immediate, on which the machine
     answers the same. Options: no value; or a poison of kind value is
     an unspecified tagged immediate, which follows the compiler but
     changes what a constant is for the whole model. Recommendation: no
     value, as now. It is sound for what Simplify does, and leaves out
     only the terms that run a primitive on poison. A relation of a
     type that holds of poison and 0 can stand beside an evaluation
     that gives no value there, as long as the relation contains the
     graph of the primitive: T1 and this request need that inclusion,
     not equal domains. It cannot be used to say that running the
     primitive on poison is the same as 0.
   - `Is_int` without `variant_only` on null. Options: 0 (the machine
     word is even; Simplify claims nothing there) or no value.
     Recommendation: no value until `or_null` is modelled in the types.
3. **`Get_tag` off blocks.** Now: no value on anything but a block.
   It is not that Simplify merely learns nothing off blocks, as the
   first version said: `prove_get_tag_generic_value` answers `Unknown`
   on boxed numbers, closures and strings, but the relation that the
   simplifier of the primitive records makes its argument a variant
   that is not null, and with it `Get_tag` of null or of a known boxed
   number becomes `Invalid` (C1-6, R1-5), where the backend would read
   the tag in the header of whatever it is given. On a tagged immediate
   Simplify leaves the primitive (R1-1 says why the types tolerate it).
   So the model has to give no value on null and on boxed numbers; on
   the rest it is a choice. Recommendation: no value off blocks, which
   T1 recommends too.
4. **Integers out of the range of their kind.** A constant of the model
   holds any integer, which no constant of the compiler does (C1-12).
   Now: arithmetic, comparison, tagging and untagging have no value on
   one out of range. Options: wrap it; or make a term with such a
   constant ill-formed. Recommendation: keep "no value", and let C2's
   well-formedness rule such constants out later. The converter never
   makes one.
5. **Division.** By zero: no value; settled, as Simplify takes it to be
   unreachable and gives `Invalid` (C1-1). The least integer by minus
   one: wraps; settled, as Simplify folds it so and the backend forces
   it (`make_safe_divmod` in `backend/cmm_helpers.ml`, and
   `dividend_cannot_be_min_int` in `To_cmm_primitive`; C1-2). If the
   human would rather not rest on the backend, one line makes it
   undefined.
6. **`Block_load`.** No value on a field out of bounds: settled,
   Simplify gives `Invalid`. Now, beyond that: no value on a block of
   another tag than the access kind gives, or of another size where it
   gives one, and when `field_kind` is `Immediate` and the field is not
   a tagged immediate. Simplify takes the block to have that tag only
   where it does not already know the field; where it does, it reads
   the field whatever the tag. It never uses the size, and it reads an
   `Immediate` field that holds a block (C1-7, R1-4; the first version
   of this note said the tag was settled, which was right of one of the
   two paths). The backend tests neither the tag nor the size. So the
   three checks are a contract that the model puts on the term, not
   facts that the compiler checks. Option: not to look at the tag, the
   size or the field kind, which makes more terms defined; each use
   Simplify makes of them is then to be checked against that.
   Recommendation: as now.
7. **Kinds.** A primitive applied to an argument that is not of the
   kind it takes has no value. Simplify gives `Invalid` when the type
   of an argument has another kind (`arg_kind_mismatch`), and stops
   with an error instead under `-flambda2-kind-checks` (C1-3). The
   human: "In general we cannot use -flambda2-kind-checks, because
   oxcaml allows GADTs that can cause kind mismatches in flambda2 (e.g.
   using "any")." So terms of legitimate programs have ill-kinded
   primitives, in code that the typing of the source makes unreachable.
   - Nothing in this design assumes that a term is well-kinded
     statically. `eval_prim` looks at the values it is given; the law
     about kinds is about those values; `prim_args_well_kinded` takes
     the kinds of values; and no statement of section 1, and no repair
     of section 5, has a premise about the kinds of a term. An
     ill-kinded primitive that is never run changes nothing. One that
     is run makes the term undefined, which is what makes Simplify's
     `Invalid` a refinement.
   - What does rest on kinds is that the two tests agree. The model's
     is on the values at run time, Simplify's on the kinds of types.
     They agree where each variable holds values of the kind of its
     type, which is C2's well-kindedness, or where the code is not run.
     With neither, a variable whose type has the wrong kind may hold a
     value of the right one: the model then gives the primitive a
     value, Simplify gives `Invalid`, and the tool would report the
     two runs as different. Whether a legitimate program can run such
     code is what C2 is finding out.
   - `Opaque_identity` checks its argument against the kind it
     carries, by the same rule.
8. **The target.** Now: 64 bits, as the test tool asks of Simplify.
   Option: a machine width as a parameter of the model. Recommendation:
   64 bits until something needs the other.
9. **Shifts**, for when `Int_shift` is added: Simplify gives 0 for an
   amount out of range and says the case is undefined. Recommendation:
   no value outside `[0, width)`; to settle then, with the bound, which
   `Number_adjuncts` puts at 64 for tagged immediates and at 63 for
   naked ones: an arithmetic shift right of -1 by 63 folds to -1 for
   the first and to 0 for the second, where the machine gives -1 for
   both (C1-9).
10. **Identity of allocated values**, for when `Phys_equal` is added:
    no value on anything allocated, until a human says what identity is
    (C3 has the same question for closures).
11. **A boxed poison.** A symbol bound to a boxed number whose contents
    are a poison of a naked number kind is a value of today's machine.
    Now: it is a boxed number for the observation, by the kind of its
    contents, as T1's types read it; so it is allocated, and `Is_null`
    answers 0 on it. `Unbox_number` has no value on it, and
    `Box_number` none on a poison constant. Simplify gives the result
    of a boxing the type "boxed, with contents of the type of the
    argument", with no test for poison, whose type is the unknown one of
    its kind; that does not say what boxing poison does when it is run.
    Options: as now; or a boxed poison is no boxed number, which changes
    what a type means and is to be reviewed as that; or boxing and
    unboxing are defined on poison, which follows the types that
    Simplify gives but defines a run on poison (decision 2).
    Recommendation: as now (R1-11).
12. **Tags out of range.** `Make_block`, a static block and an access
    kind carry any integer as a tag, where the compiler's
    `Tag.Scannable.t` is at least 0 and below 251. The converter makes
    only tags in range. Now: nothing refuses another; `Make_block` with
    the tag -1 makes a block. Options: no value for `Make_block` out of
    range, and the like for static blocks; or leave the integers as
    they are and let C2's well-formedness rule such terms out, as for
    integer constants (decision 4). Recommendation: the second, as these
    are terms the compiler cannot build (R1-8).
13. **Blocks of no fields.** Now: `Make_block` of no arguments makes an
    empty block. The compiler's backend refuses to allocate one (it has
    to be static) and relies on Simplify's having lifted it to a symbol.
    Recommendation: keep it defined. That an empty allocation does not
    reach the backend is an obligation on the compiler, to be stated
    with lifting (decision 1), and no reason to make the source
    undefined (R1-7).
14. **What the kind of a field promises.** The compiler's `Make_block`
    says of each field what subkind it has, from the types of the
    source. Simplify uses it for an argument that is a variable, not
    for a constant or a symbol, and the backend ignores it (R1-3,
    C1-13). The model has no subkinds anywhere, as agreed with C2. Now:
    the converter refuses a block kind that says more than `any_value`,
    so such a term is not checked. Options: keep refusing; or read the
    subkinds the model can (not null, tagged immediate, the boxed
    numbers) as "no value otherwise", which is leaf work in the
    primitives' files but makes a contract of what the compiler only
    assumes, and which variants and arrays do not fit yet.
    Recommendation: refuse for now, and settle what a subkind promises
    together with kinds (decision 7): terms from real programs have
    subkinds on most blocks, so the refusal will cost coverage as soon
    as the corpus is not written in fexpr.

## 8. The plan for the epoch

The work exists on the branch; the epoch is a replay with review, not
new proof.

Order (C0's, after it built the branch and accepted the request):

1. (a) The two moves and nothing else: `FlambdaKindModel.v` and
   `FlambdaValueSpec.v` with what `FlambdaModel.v` and
   `FlambdaSemanticsSpec.v` had, and their two lines of `rocq/Makefile`.
   No meaning changes, and nothing else needs touching.
2. (b) Primitives as data: the new constructors of `value`, the
   observations, the three files of the primitives at stage 1 with
   their lines of `rocq/Makefile`, the `Spec` diffs of section 1, the
   repairs of section 5 to C0's files and to V1's (which has no owner),
   `FlambdaExtract.v`, and the ledger as R1 signs it off.
3. (c) This note, C1's status file and its section of `OWNERS.md`.
4. (d) Last, applied by C0 in one commit with the regenerated extracted
   files and without `prim_hook.ml`: the test tool (V2 and V3 have no
   owner), the examples of stage 1 with their expected files, and the
   generator. The tool compiles only against the extracted files of its
   own model, and only C0 commits those; until (d) the tool and the
   committed extracted files are both the old ones.
5. C0: `rocq/HACKING.md`.
6. Validation. Each of (a) to (c): `make -C rocq all` and `make -C rocq
   assumptions`. After (d): the semantics tests and `make -s test`
   (OCaml changed).
7. After the epoch, leaf merges from C1: the fourth law; stage 2;
   stage 3. In each stage the commit that needs regenerated extracted
   files (the converter and the examples) is the last. Each is a `Spec`
   diff of two files for R1.

The series was applied in that order: 2c6b618546, 6e814332fc,
ffeb4f81d8 and, for (d), 986d7bed8f. The prototype had the commits in
the order they were made.

Step lemmas: none changes, as `step` does not. New, in
`FlambdaPrimitiveProofs.v`: `eval_prim_unary`, `eval_prim_binary`,
`eval_prim_variadic` (the equations by arity), `eval_prim_arity`, and
the corollaries `eval_prim_same_static_values` and `eval_prim_consts`.

Other streams: T1's files do not build over the new `value` (its copy
of `kind_of_value` has no case for a block), so its merges land after
the epoch, rebased, without its copies of the observations and of
`kind_of_value`; C2 and C3 rebase their prototypes; nobody else has
files that break.

Examples (all on the branch, each with its `.expected`):

- `div_mod.fl` (stage 1): validated; not checkable before (`Div`,
  `Mod`, a comparison like `compare`).
- `int32_wraps.fl` (1): validated; not checkable before (naked int32,
  wrapping, an unsigned comparison).
- `int_ops_tagged.fl`, `int_ops_int8.fl` (1): validated; 198 checks
  each, of every operation and comparison at the bounds of the kind.
- `div_by_zero.fl` (1): the source is undefined, and Simplify gives
  `Invalid`.
- `get_tag.fl` (1): `Get_tag` and `Is_int` of a static block come to
  the same before and after. Not validated: the normaliser does not
  fold a primitive applied to a symbol.
- `box_unbox.fl` (2): validated; not checkable before; unboxes a static
  boxed number and one made by `Box_number`.
- `block_load.fl` (3): validated; not checkable before; a block made,
  loaded from and its tag read, in a loop.
- `block_lifted.fl` (3): the same before and after. Not validated:
  Simplify lifts the block to a symbol, and learns from the load that
  the value is a block.

In each validated one the source is defined. The change adds no
refinement theorem.

The integers against a reference. `requests/C1-int-ops.py` generates an
example that computes each operation and comparison of a kind on pairs
of numbers and compares the result with a reference on unbounded
integers. Run once in full: 4256 checks for each of the seven kinds,
29792 in all. On every one the model agrees with the reference (the run
before Simplify), Simplify agrees with it (the run after), and
`validate_unit` validates the pair, which says that `fold_constants`
with the model's `eval_prim` gives the term that Simplify gives. The
epoch should turn this into a test in OCaml against `Number_adjuncts`
(plan, section 2, step 4), which the generator stands in for.

What this run is: finite evidence that the model and Simplify's own
folding agree, on fourteen numbers of each kind. It is not a proof, and
it tests no generated code: the backend is not run. The reference is
written from the same reading as the model and is close to it, so the
two could share a mistake; it leaves out zero divisors (covered by
`div_by_zero.fl`) and arguments that are not integers of the kind; and
the numbers beyond 63 bits are built with a multiplication and an
addition of the kind under test, so a fault there could hide itself.
R1 read the generator and did not run it again.

What could go wrong:

- R1 or C0 refuse the two moves. Then `prim` and its types go in
  `FlambdaModel.v` and the evaluation in `FlambdaSemanticsSpec.v`; the
  definitions, the laws and every proof are the same, but each new
  primitive is an edit of two of C0's files.
- A reviewer finds a primitive whose value differs from the compiler's.
  The fix is local to `FlambdaPrimitiveSpec.v` and the proofs of the
  three laws; nothing else depends on it.
- The decisions of section 7 go another way. Each is a change of a few
  lines in one case of the evaluation, except the first, which changes
  what is observable and is not part of this request.
- Simplify lifts blocks and boxed numbers made of constants to symbols,
  so an example with one is not validated until decision 1 and S5. The
  examples here avoid it with `Opaque_identity`.
- fexpr reads every integer literal through OCaml's `int`, so a literal
  beyond 63 bits cannot be written; the examples compute such numbers.
- A static block with a mixed shape was accepted by the converter, its
  shape dropped. It is refused on the branch now (section 5).

Estimate: half a day for C0 to replay, regenerate and validate, plus
R1's reading of about 770 lines of `Spec` and `Model` in the four new
files (70 of them moved) and of the diff of section 1.

## What the prototype proves, and what it leaves

Proved on the branch, with no `Admitted`, no axiom and no `Parameter`:
the three laws at all three stages, and the fourth; `prim_eqb_sound`,
`prim_eqb_refl` and `eval_prim_avoids` as lemmas; and every theorem of
both libraries, as the whole tree builds. The ones most at risk, which
is why they were looked at first: `drop_statics_refines` (it rested on
the axiom, and its simulation runs the two sides under different static
environments, which primitives now read), `expr_eqb_sound` (it rested
on the two axioms about `prim_eqb`, and compares runs whose static
environments hold different code), `plug_refines` (its transfer relates static
environments with related code), `fold_constants_equivalent` (it calls
`eval_prim` itself) and `validate_unit_sound` (it rests on all of
them).

Left:

- `rocq/HACKING.md` is not updated, and `prim_hook.ml` is still there,
  dead: both are C0's.
- The extracted directory is not committed, by the rule.
- `make -s fmt` has not been run on the branch; `flsemantics.ml` was
  formatted with `ocamlformat`. C0 ran `make -s test` on the branch:
  only the failure that the base has.
- `get_tag.fl` and `block_lifted.fl` are not validated. Both need
  normalisation that is S1's and S5's.
- The comparison with `Number_adjuncts` is a script beside this note,
  not a test of the tree.
