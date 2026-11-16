(******************************************************************************
 *                                  OxCaml                                    *
 * -------------------------------------------------------------------------- *
 *                               MIT License                                  *
 *                                                                            *
 * Copyright (c) 2026 Jane Street Group LLC                                   *
 * opensource-contacts@janestreet.com                                         *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included    *
 * in all copies or substantial portions of the Software.                     *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER *
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING    *
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER        *
 * DEALINGS IN THE SOFTWARE.                                                  *
 ******************************************************************************)



(* Examples of what the provers answer.

   This file is to be reviewed with FlambdaProversSpec.v: it is there
   to show what the provers of FlambdaProvers.v compute on small types,
   so that the transcription can be read against
   middle_end/flambda2/types/provers.ml with answers in hand, and to
   show by an example why the statements of FlambdaProversSpec.v are
   for a valuation that satisfies the environment strictly. The proofs
   are in FlambdaProversExamplesProofs.v, which needs no review.

   The first examples are in the empty environment, of types made with
   the creation functions of FlambdaTypeGrammar.v. The others are in an
   environment with two variables, [n] of kind value, whose type is
   that of null alone, and [x], a naked immediate whose type is the
   relation [Is_null] to [n]: what
   [Simplify_unary_primitive.simplify_is_null] gives the result of
   [Is_null] of [n]. Then a closure whose value slot holds a naked
   float, of which the prover of the projection of that slot, asked
   at the kind value, answers [Invalid]. The last are the answers of
   the provers of blocks, of closures and of physical equality, in the
   empty environment. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaExpandHead FlambdaProvers FlambdaProversSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* In the empty environment                                                 *)
(* ======================================================================== *)

(* A boxed float of which nothing more is known. *)
Definition example_boxed_float : flambda_type :=
  box_number Boxable_naked_float (unknown (Naked_number Kind_naked_float))
    Alloc_mode.Heap.

(* [Is_int]: yes of the tagged immediates, no of the blocks and of a
   boxed float; nothing of a value of which nothing is known. With
   [variant_only] a boxed float is not in the domain. *)
Definition example_is_int_statement : Prop :=
  Provers.prove_is_int Typing_env.create any_tagged_immediate
  = Some (Proof_of_property.Proved true)
  ∧ Provers.prove_is_int Typing_env.create any_block
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_is_int Typing_env.create example_boxed_float
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_is_int Typing_env.create (unknown Value)
    = Some Proof_of_property.Unknown
  ∧ Provers.meet_is_int_variant_only Typing_env.create any_block
    = Some (Meet_shortcut.Known_result false)
  ∧ Provers.meet_is_int_variant_only Typing_env.create example_boxed_float
    = Some Meet_shortcut.Invalid.

(* The tags of blocks, the numbers of tagged immediates and of naked
   floats. *)
Definition example_numbers_statement : Prop :=
  Provers.prove_get_tag Typing_env.create
    (blocks_with_these_tags [0; 1] Alloc_mode.Heap)
  = Some (Proof_of_property.Proved [0; 1])
  ∧ Provers.prove_get_tag Typing_env.create any_block
    = Some Proof_of_property.Unknown
  ∧ Provers.prove_equals_tagged_immediates Typing_env.create
      (these_tagged_immediates [3; 4])
    = Some (Proof_of_property.Proved [3; 4])
  ∧ Provers.meet_equals_single_tagged_immediate Typing_env.create
      (these_tagged_immediates [3; 4])
    = Some Meet_shortcut.Need_meet
  ∧ Provers.meet_equals_single_tagged_immediate Typing_env.create
      (this_tagged_immediate 3)
    = Some (Meet_shortcut.Known_result 3)
  ∧ Provers.meet_equals_tagged_immediates Typing_env.create any_block
    = Some Meet_shortcut.Invalid
  ∧ Provers.meet_naked_floats Typing_env.create
      (Type_naked_number Kind_naked_float (Ok (No_alias [5; 6])))
    = Some (Meet_shortcut.Known_result [5; 6])
  ∧ Provers.meet_naked_floats Typing_env.create (this_naked_immediate 5)
    = Some Meet_shortcut.Invalid.

(* A prover stops where the compiler does: on an alias to a variable
   that the environment does not know. *)
Definition example_stop_statement : Prop :=
  Provers.prove_is_int Typing_env.create (alias_type_of Value (Var 1%positive))
  = None.

(* ======================================================================== *)
(* A relation, and a name that holds poison                                 *)
(* ======================================================================== *)

Definition example_n : variable := 1%positive.
Definition example_x : variable := 2%positive.

(* The type of [x]: 0 or 1, the result of [Is_null] of [n]. *)
Definition example_x_type : flambda_type :=
  is_null_for_scrutinee (Name_var example_n).

Definition example_null_env : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_n Value Name_mode.normal
  in
  let tenv := Typing_env.replace_equation tenv (Name_var example_n) null in
  let tenv :=
    Typing_env.add_variable_definition tenv example_x
      (Naked_number Kind_naked_immediate) Name_mode.normal
  in
  Typing_env.replace_equation tenv (Name_var example_x) example_x_type.

(* The provers read the type of [n] through the alias and through the
   relation: [n] is null, so it is not a pointer, and [x] is 1. *)
Definition example_null_answers_statement : Prop :=
  Provers.meet_is_null example_null_env
    (alias_type_of Value (Var example_n))
  = Some (Meet_shortcut.Known_result true)
  ∧ Provers.prove_is_not_a_pointer example_null_env
      (alias_type_of Value (Var example_n))
    = Some (Proof_of_property.Proved true)
  ∧ Provers.prove_is_int example_null_env
      (alias_type_of Value (Var example_n))
    = Some Proof_of_property.Unknown
  ∧ Provers.meet_naked_immediates example_null_env example_x_type
    = Some (Meet_shortcut.Known_result [1])
  ∧ Provers.meet_naked_immediates example_null_env
      (alias_type_of (Naked_number Kind_naked_immediate) (Var example_x))
    = Some (Meet_shortcut.Known_result [1]).

(* Where [n] holds null and [x] holds 1 the environment is satisfied
   strictly, and the answers are true. *)
Definition example_null_env_strict_statement : Prop :=
  typing_env_satisfied_strictly empty_static_env
    [(example_x, Value_const (Naked_immediate 1));
     (example_n, Value_const Null)]
    example_null_env.

(* Where [n] holds poison and [x] holds 0 the environment is satisfied
   with the names that hold poison exempt, and not strictly. The type
   of [x] is true of 0 there, since [Is_null] relates poison to 0; the
   type of [x] and the types that the environment has for its names
   are well formed ([well_formed_input], which is the condition of the
   prover's statement and not an invariant of the environment); and
   the prover's answer for that type, that the number is 1, is false
   of the value 0. So
   [meet_naked_immediates_sound_statement] does not hold with
   [typing_env_satisfied] in place of strict satisfaction. *)
Definition example_null_env_poison_statement : Prop :=
  let valuation : env :=
    [(example_x, Value_const (Naked_immediate 0));
     (example_n, Value_const (Poison Value))]
  in
  typing_env_satisfied empty_static_env valuation example_null_env
  ∧ ~ typing_env_satisfied_strictly empty_static_env valuation
        example_null_env
  ∧ well_formed_input example_null_env example_x_type
  ∧ type_satisfied empty_static_env valuation
      (Value_const (Naked_immediate 0)) example_x_type
  ∧ ~ naked_immediate_among [1] (Value_const (Naked_immediate 0)).

(* ======================================================================== *)
(* A value slot whose contents are not of the kind asked                    *)
(* ======================================================================== *)

Definition example_value_slot : value_slot := 1%positive.

(* A closure of one function, whose value slot holds a naked float. *)
Definition example_closure : value :=
  Value_closure 1%positive
    [(example_value_slot, Value_const (Naked_float 0))].

(* Its type: the closure of a set of that one function and that value
   slot, whose contents are a naked float. *)
Definition example_closure_type : flambda_type :=
  exactly_this_closure unnamed_function_slot
    [(unnamed_function_slot, Or_unknown.Unknown)] []
    [(example_value_slot, unknown (Naked_number Kind_naked_float))]
    Alloc_mode.Heap.

(* The closure has the type, and the projection of the slot has a
   value, the naked float. Asked at the kind value,
   [meet_project_value_slot_simple] answers [Invalid]; asked at the
   kind of the contents it does not. The closure is not in the domain
   of [meet_project_value_slot_simple_sound_statement] at the kind
   value, so that statement stands; and it is why the statement does
   not, alone, make [Invalid] right of the machine (discrepancy
   T6-4). *)
Definition example_value_slot_kind_statement : Prop :=
  type_satisfied empty_static_env [] example_closure example_closure_type
  ∧ project_value_slot empty_static_env unnamed_function_slot
      example_value_slot example_closure
    = Some (Value_const (Naked_float 0))
  ∧ Provers.meet_project_value_slot_simple Typing_env.create
      Name_mode.normal example_closure_type example_value_slot Value
    = Some Meet_shortcut.Invalid
  ∧ Provers.meet_project_value_slot_simple Typing_env.create
      Name_mode.normal example_closure_type example_value_slot
      (Naked_number Kind_naked_float)
    = Some Meet_shortcut.Need_meet.

(* ======================================================================== *)
(* Blocks, closures and physical equality                                   *)
(* ======================================================================== *)

(* A block of tag 0 whose two fields are the constants 1 and 2; and
   one whose fields are a tagged immediate and a block of which
   nothing more is known. *)
Definition example_pair : flambda_type :=
  immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
    [this_tagged_immediate 1; this_tagged_immediate 2].

Definition example_opaque_pair : flambda_type :=
  immutable_block false 0 (Scannable Value_only) Alloc_mode.Heap
    [any_tagged_immediate; any_block].

(* The constructors of a block type, its tag and size, its fields as
   [Simple]s where they are known, and one field: the field that the
   block has not, and a field asked at another kind than its type has,
   are [Invalid]. *)
Definition example_blocks_statement : Prop :=
  Provers.prove_variant_like Typing_env.create example_opaque_pair
  = Some
      (Proof_of_property.Proved
         (Variant_like_proof (Or_unknown.Known [])
            [(0, (2%nat, Scannable Value_only))]))
  ∧ Provers.prove_unique_tag_and_size Typing_env.create example_opaque_pair
    = Some (Proof_of_property.Proved (0, Scannable Value_only, 2%nat))
  ∧ Provers.prove_unique_fully_constructed_immutable_heap_block
      Typing_env.create example_pair
    = Some
        (Proof_of_property.Proved
           (0, Scannable Value_only, 2%nat,
            [Const (Tagged_immediate 1); Const (Tagged_immediate 2)]))
  ∧ Provers.prove_unique_fully_constructed_immutable_heap_block
      Typing_env.create example_opaque_pair
    = Some Proof_of_property.Unknown
  ∧ Provers.meet_block_field_simple Typing_env.create Name_mode.normal Value
      example_pair 1%nat
    = Some (Meet_shortcut.Known_result (Const (Tagged_immediate 2)))
  ∧ Provers.meet_block_field_simple Typing_env.create Name_mode.normal Value
      example_pair 2%nat
    = Some Meet_shortcut.Invalid
  ∧ Provers.meet_block_field_simple Typing_env.create Name_mode.normal
      (Naked_number Kind_naked_float) example_pair 1%nat
    = Some Meet_shortcut.Invalid
  ∧ Provers.meet_block_field_simple Typing_env.create Name_mode.normal Value
      example_opaque_pair 1%nat
    = Some Meet_shortcut.Need_meet.

Definition example_function_slot : function_slot := 5%positive.
Definition example_code : code_id := 7%positive.

(* The closure of a set of one function, of known code, with one value
   slot that holds the constant 7. *)
Definition example_closure_with_code_type : flambda_type :=
  exactly_this_closure example_function_slot
    [(example_function_slot,
      Or_unknown.Known (Function_type example_code))]
    []
    [(example_value_slot, this_tagged_immediate 7)] Alloc_mode.Heap.

(* The entry and the code of a closure type, and the two projections.
   A block type has no closure. *)
Definition example_closures_statement : Prop :=
  Provers.prove_single_closures_entry Typing_env.create
    example_closure_with_code_type
  = Some
      (Proof_of_property.Proved
         (example_function_slot, Alloc_mode.Heap,
          Closures_entry
            [(example_function_slot,
              Or_unknown.Known (Function_type example_code))]
            []
            [(example_value_slot, this_tagged_immediate 7)],
          Function_type example_code))
  ∧ Provers.prove_code_ids Typing_env.create example_closure_with_code_type
    = Some (Proof_of_property.Proved [example_code])
  ∧ Provers.meet_code_ids Typing_env.create any_block
    = Some Meet_shortcut.Invalid
  ∧ Provers.meet_project_value_slot_simple Typing_env.create
      Name_mode.normal example_closure_with_code_type example_value_slot
      Value
    = Some (Meet_shortcut.Known_result (Const (Tagged_immediate 7)))
  ∧ Provers.meet_project_function_slot_simple Typing_env.create
      Name_mode.normal example_closure_with_code_type example_function_slot
    = Some Meet_shortcut.Need_meet
  ∧ Provers.meet_project_function_slot_simple Typing_env.create
      Name_mode.normal any_block example_function_slot
    = Some Meet_shortcut.Invalid.

(* A boxed float of one number. *)
Definition example_boxed_float_of (bits : Z) : flambda_type :=
  box_number Boxable_naked_float
    (Type_naked_number Kind_naked_float (Ok (No_alias [bits])))
    Alloc_mode.Heap.

(* Physical equality: one constant and two; a block and a tagged
   immediate; two blocks of which nothing is known; a block and a
   boxed float; two boxed floats of two numbers; null and null. *)
Definition example_physical_equality_statement : Prop :=
  Provers.prove_physical_equality Typing_env.create (this_tagged_immediate 1)
    (this_tagged_immediate 1)
  = Some (Proof_of_property.Proved true)
  ∧ Provers.prove_physical_equality Typing_env.create
      (this_tagged_immediate 1) (this_tagged_immediate 2)
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_physical_equality Typing_env.create any_block
      any_tagged_immediate
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_physical_equality Typing_env.create any_block any_block
    = Some Proof_of_property.Unknown
  ∧ Provers.prove_physical_equality Typing_env.create any_block
      example_boxed_float
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_physical_equality Typing_env.create
      (example_boxed_float_of 5) (example_boxed_float_of 6)
    = Some (Proof_of_property.Proved false)
  ∧ Provers.prove_physical_equality Typing_env.create null null
    = Some (Proof_of_property.Proved true).
