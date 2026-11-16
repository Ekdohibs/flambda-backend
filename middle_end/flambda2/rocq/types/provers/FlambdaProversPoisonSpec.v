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


(* The provers where the names that hold poison are exempt.

   This file is to be reviewed: the two forms ([prover_sound_exempt],
   [meet_shortcut_sound_exempt]) and the statements. It follows
   FlambdaProversSpec.v, which has the provers' statements for a
   valuation that satisfies the typing environment strictly, and which
   says what each answer means; nothing of that is said again here.

   Why. After the unboxing of the parameters of a continuation the
   typing environments that Simplify builds are satisfied only in the
   sense that exempts the names that hold poison
   ([typing_env_satisfied] of FlambdaTypingEnvSpec.v, decision D26):
   a parameter that stands for a field of a case of a variant that is
   not the case of the value holds poison, and has the type of the
   field all the same. The statements of FlambdaProversSpec.v say
   nothing of such an environment. Those here do.

   What is stated. Each statement is the statement of the same name,
   without [_exempt], of FlambdaProversSpec.v, with [prover_sound]
   changed for [prover_sound_exempt] and [meet_shortcut_sound] for
   [meet_shortcut_sound_exempt]: the same prover, the same condition
   on what it is asked, the same domain, the same reading of the
   answer. The two forms differ from those two in that the valuation
   satisfies the environment with the names that hold poison exempt,
   and in that the value is not poison. So an answer may be used of a
   value that is not poison, in an environment in which other names
   hold poison. Of a value that is poison nothing is said: it is the
   value of a name that is exempt, and a prover answers what the
   environment says of the name, which may be wrong of poison. So a
   rewrite that follows an answer owes its own case for a value that
   is poison, without the answer. The two statements at the end are
   two such cases: [Is_int] and [Get_tag] have no value on poison
   (FlambdaPrimitiveSpec.v), so they need no premise on the value.
   That is not so of every reader of a value: a primitive may have a
   value on poison ([Opaque_identity] has), and what makes a block or
   a closure may put poison in it.

   Which provers. Those that read no relation of a type of naked
   immediates: the scrutinee questions ([prove_is_int],
   [meet_is_int_variant_only], [prove_is_not_a_pointer],
   [prove_get_tag], [meet_is_null]), [meet_naked_number],
   [prove_equals_to_simple_of_kind], the provers of boxed numbers and
   of strings, [prove_unique_tag_and_size],
   [prove_unique_fully_constructed_immutable_heap_block],
   [meet_block_field_simple], and the provers of closures and of
   projections. What makes them right: such a prover expands the type
   asked about, whose value is not poison, so that the name it is an
   alias of is not exempt; and beyond that it only asks whether the
   immediates of a variant have no value, which it concludes from a
   type that expands to bottom, and a name whose type is bottom and
   which holds a number is not exempt either.

   Which are not here, and why. [meet_naked_immediates],
   [prove_equals_tagged_immediates], [meet_equals_tagged_immediates],
   [meet_equals_single_tagged_immediate], [prove_tagging_of_simple],
   [meet_tagging_of_simple], [prove_variant_like], [meet_variant_like],
   [prove_physical_equality] and reification (FlambdaReifySpec.v) read
   the numbers of a type of naked immediates, with its relations: for
   each relation ([Is_int], [Get_tag], [Is_null]) they ask the prover
   of that question about the name that the relation is on, and take
   the answer for the number. Where that name holds poison the answer
   is what the environment says of a name that is exempt, and it is
   wrong (discrepancy T6-1; the example is in
   FlambdaProversExamplesSpec.v). A name whose own type is an
   [Is_int] or a [Get_tag] relation on a name that holds poison is
   exempt as well ([Is_null] is not among those: it relates a poison
   to 0, FlambdaTypingEnvSpec.v). For those provers the statement that
   there is, is the one where no name holds poison
   ([sound_where_no_poison_statement] of FlambdaProversSpec.v). What
   each would need is less: that the names that the relations read
   are on hold no poison. That statement is not made here. One part
   of two of them is: [prove_tagging_of_simple] and
   [meet_tagging_of_simple] answer a [Simple] that is a name only from
   the alias that the immediates of the variant are, with no relation
   read, and of such an answer the statement is made (the section
   "Tagging").

   The compiler has no such statements. It relies on no answer about
   a name that holds poison being used on a path that runs, and
   nothing checks that; decision D26 says how the model reads it.

   Nothing is trusted here. The proofs are in
   FlambdaProversPoisonProofs.v, which needs no review; they are the
   proofs of FlambdaProversProofs.v, which are made once for the two
   notions of satisfaction. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec FlambdaExpandHead
  FlambdaProvers FlambdaProversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The two forms                                                            *)
(* ======================================================================== *)

(* [prover_sound] of FlambdaProversSpec.v, for a valuation that
   satisfies the environment with the names that hold poison exempt,
   and of a value that is not poison. *)
Definition prover_sound_exempt {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Proof_of_property.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (x : A) (v : value),
    typing_env_satisfied statics valuation tenv ->
    ~ is_poison v ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    prover tenv t = Some (Proof_of_property.Proved x) ->
    holds statics valuation x v.

(* [meet_shortcut_sound], in the same way. *)
Definition meet_shortcut_sound_exempt {A : Type}
    (prover : Typing_env.t -> flambda_type -> option (Meet_shortcut.t A))
    (input : Typing_env.t -> flambda_type -> Prop)
    (domain : static_env -> value -> Prop)
    (holds : static_env -> env -> A -> value -> Prop) : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied statics valuation tenv ->
    ~ is_poison v ->
    input tenv t ->
    type_satisfied statics valuation v t ->
    domain statics v ->
    match prover tenv t with
    | Some (Meet_shortcut.Known_result x) => holds statics valuation x v
    | Some Meet_shortcut.Invalid => False
    | Some Meet_shortcut.Need_meet | None => True
    end.

(* ======================================================================== *)
(* The scrutinee questions and the naked numbers                            *)
(* ======================================================================== *)

Definition prove_is_int_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_is_int any_input (fun _ _ => is_int_is).

Definition meet_is_int_variant_only_sound_exempt_statement : Prop :=
  meet_shortcut_sound_exempt Provers.meet_is_int_variant_only any_input
    variants (fun _ _ => is_int_is).

Definition prove_is_not_a_pointer_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_is_not_a_pointer any_input
    (fun _ _ => is_not_a_pointer_is).

Definition prove_get_tag_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_get_tag any_input
    (fun statics _ => block_with_tag_among statics).

Definition meet_is_null_sound_exempt_statement : Prop :=
  meet_shortcut_sound_exempt Provers.meet_is_null any_input
    (values_of_kind Value) (fun _ _ => is_null_is).

(* At the eleven kinds that are not that of the immediates, as in
   FlambdaProversSpec.v. *)
Definition meet_naked_number_sound_exempt_statement : Prop :=
  ∀ kind : naked_number_kind,
    kind <> Kind_naked_immediate ->
    meet_shortcut_sound_exempt (Provers.meet_naked_number kind) any_input
      (values_of_kind (Naked_number kind))
      (fun _ _ numbers v => naked_numbers_satisfied kind v numbers).

(* ======================================================================== *)
(* Aliases, boxed numbers and strings                                       *)
(* ======================================================================== *)

Definition prove_equals_to_simple_of_kind_sound_exempt_statement : Prop :=
  ∀ kind : flambda_kind,
    prover_sound_exempt
      (fun tenv t => Provers.prove_equals_to_simple_of_kind tenv t kind)
      any_input
      (fun _ valuation s v => eval_simple valuation s = Some v).

Definition prove_is_a_boxed_or_tagged_number_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_is_a_boxed_or_tagged_number any_input
    boxed_or_tagged_number_is.

Definition prove_is_a_tagged_immediate_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_is_a_tagged_immediate any_input
    (fun _ _ _ v => is_immediate v).

Definition prove_is_a_boxed_number_sound_exempt_statement : Prop :=
  ∀ kind : boxable_number,
    prover_sound_exempt (Provers.prove_is_a_boxed_number kind) any_input
      (fun statics _ _ v => boxed_numbers kind statics v).

Definition prove_is_or_is_not_a_boxed_float_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_is_or_is_not_a_boxed_float any_input
    (fun statics _ x v =>
       if x
       then boxed_numbers Boxable_naked_float statics v
       else ~ boxed_numbers Boxable_naked_float statics v).

Definition meet_boxed_number_containing_simple_sound_exempt_statement : Prop :=
  ∀ (kind : boxable_number) (min_name_mode : Name_mode.t),
    meet_shortcut_sound_exempt
      (fun tenv t =>
         Provers.meet_boxed_number_containing_simple kind tenv min_name_mode
           t)
      any_input (boxed_numbers kind)
      (fun statics valuation s v =>
         ∃ c : const,
           boxed_number_of statics v = Some c
           ∧ eval_simple valuation s = Some (Value_const c)).

Definition prove_strings_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_strings any_input
    (fun statics _ => string_among statics).

Definition meet_strings_sound_exempt_statement : Prop :=
  meet_shortcut_sound_exempt Provers.meet_strings any_input strings
    (fun statics _ => string_among statics).

(* ======================================================================== *)
(* Tagging                                                                  *)
(* ======================================================================== *)

(* [Provers.prove_tagging_of_simple] and
   [Provers.meet_tagging_of_simple], where the [Simple] answered is not
   a constant: it is then the canonical element of the alias that the
   immediates of the variant are, and the value is the tagged
   immediate of its number ([tagging_of] of FlambdaProversSpec.v). A
   constant is answered from that alias too, or else from the numbers
   and the relations of the immediates, of which nothing is said
   here. Nothing is asked of the type: the numbers are not read. *)
Definition prove_tagging_of_simple_name_exempt_statement : Prop :=
  ∀ min_name_mode : Name_mode.t,
    prover_sound_exempt
      (fun tenv t => Provers.prove_tagging_of_simple tenv min_name_mode t)
      any_input
      (fun _ valuation s v =>
         (∀ c : const, s <> Const c) -> tagging_of valuation s v).

(* The answer [Known_result] of the meet shortcut, of a tagged
   immediate. Its answer [Invalid] is from the numbers. *)
Definition meet_tagging_of_simple_name_exempt_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (min_name_mode : Name_mode.t) (t : flambda_type) (s : simple)
    (v : value),
    typing_env_satisfied statics valuation tenv ->
    type_satisfied statics valuation v t ->
    is_immediate v ->
    Provers.meet_tagging_of_simple tenv min_name_mode t
    = Some (Meet_shortcut.Known_result s) ->
    (∀ c : const, s <> Const c) ->
    tagging_of valuation s v.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Definition prove_unique_tag_and_size_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_unique_tag_and_size any_input
    (fun statics _ => block_of_tag_shape_and_size statics).

Definition prove_unique_immutable_heap_block_sound_exempt_statement : Prop :=
  prover_sound_exempt
    Provers.prove_unique_fully_constructed_immutable_heap_block any_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ ∀ (i : nat) (s : simple) (field : value),
             nth_error simples i = Some s ->
             nth_error fields i = Some field ->
             eval_simple valuation s = Some field).

Definition prove_unique_immutable_heap_block_fields_exempt_statement : Prop :=
  prover_sound_exempt
    Provers.prove_unique_fully_constructed_immutable_heap_block
    well_formed_input
    (fun statics valuation x v =>
       let '(tag, shape, size, simples) := x in
       ∃ fields : list value,
         block_of statics v = Some (tag, fields)
         ∧ length fields = size
         ∧ block_has_shape shape tag fields
         ∧ length simples = size
         ∧ eval_simples valuation simples = Some fields).

Definition meet_block_field_simple_sound_exempt_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
    (field_index : nat),
    meet_shortcut_sound_exempt
      (fun tenv t =>
         Provers.meet_block_field_simple tenv min_name_mode field_kind t
           field_index)
      any_input (block_field_loads field_kind field_index)
      (fun statics valuation s v =>
         ∃ (tag : Z) (fields : list value) (field : value),
           block_of statics v = Some (tag, fields)
           ∧ nth_error fields field_index = Some field
           ∧ eval_simple valuation s = Some field).

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

Definition prove_single_closures_entry_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_single_closures_entry any_input
    single_closures_entry_holds.

Definition meet_single_closures_entry_sound_exempt_statement : Prop :=
  meet_shortcut_sound_exempt Provers.meet_single_closures_entry
    well_formed_input closures single_closures_entry_holds.

Definition prove_code_ids_sound_exempt_statement : Prop :=
  prover_sound_exempt Provers.prove_code_ids any_input
    (fun statics _ => closure_code_among statics).

Definition meet_code_ids_sound_exempt_statement : Prop :=
  meet_shortcut_sound_exempt Provers.meet_code_ids any_input closures
    (fun statics _ => closure_code_among statics).

Definition meet_project_function_slot_simple_sound_exempt_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (move_to : function_slot),
    meet_shortcut_sound_exempt
      (fun tenv t =>
         Provers.meet_project_function_slot_simple tenv min_name_mode t
           move_to)
      any_input (function_slot_projects move_to)
      (fun statics valuation s v =>
         ∀ (move_from : function_slot) (closure : value),
           project_function_slot statics move_from move_to v = Some closure ->
           eval_simple valuation s = Some closure).

Definition meet_project_value_slot_simple_sound_exempt_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind),
    meet_shortcut_sound_exempt
      (fun tenv t =>
         Provers.meet_project_value_slot_simple tenv min_name_mode t slot
           value_slot_kind)
      any_input (value_slot_projects value_slot_kind slot)
      (fun statics valuation s v =>
         ∀ (project_from : function_slot) (contents : value),
           project_value_slot statics project_from slot v = Some contents ->
           eval_simple valuation s = Some contents).

(* [meet_project_value_slot_simple_known_statement], in the same
   way. *)
Definition meet_project_value_slot_simple_known_exempt_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (min_name_mode : Name_mode.t) (t : flambda_type) (slot : value_slot)
    (value_slot_kind : flambda_kind) (s : simple) (v : value),
    typing_env_satisfied statics valuation tenv ->
    ~ is_poison v ->
    type_satisfied statics valuation v t ->
    Provers.meet_project_value_slot_simple tenv min_name_mode t slot
      value_slot_kind
    = Some (Meet_shortcut.Known_result s) ->
    ∀ (project_from : function_slot) (contents : value),
      project_value_slot statics project_from slot v = Some contents ->
      eval_simple valuation s = Some contents.

(* ======================================================================== *)
(* What Simplify does with [Provers.prove_is_int]                           *)
(* ======================================================================== *)

(* [prove_is_int_is_int_statement] and [prove_is_int_get_tag_statement]
   of FlambdaProversSpec.v, with the names that hold poison exempt and
   no premise on the value: [Is_int] has no value on poison, and
   neither has [Get_tag]. So the two rewrites of Simplify that follow
   an answer of [prove_is_int] are right in an environment in which
   names hold poison, of whatever value the argument has. *)
Definition prove_is_int_is_int_exempt_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (b : bool) (v r : value),
    typing_env_satisfied statics valuation tenv ->
    type_satisfied statics valuation v t ->
    Provers.prove_is_int tenv t = Some (Proof_of_property.Proved b) ->
    eval_prim statics (Unary (Is_int false)) [v] = Some r ->
    r = naked_bool b.

Definition prove_is_int_get_tag_exempt_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (t : flambda_type) (v : value),
    typing_env_satisfied statics valuation tenv ->
    type_satisfied statics valuation v t ->
    Provers.prove_is_int tenv t = Some (Proof_of_property.Proved true) ->
    eval_prim statics (Unary Get_tag) [v] = None.
