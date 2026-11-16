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


(* The provers and reification commute with renaming.

   This file is to be reviewed: the action of a renaming on what the
   expansion of a head answers ([rename_expanded_head],
   [rename_expanded], [rename_descr_oub]) and on the answers of the
   provers and of reification (the section "The answers, renamed"),
   the two forms of statement ([prover_invariant],
   [prover_equivariant]), and the statements. The provers are
   FlambdaProvers.v and reification is FlambdaReify.v; the action on
   types is FlambdaTypeRenameSpec.v and that on typing environments
   FlambdaEnvRenameSpec.v.

   A renaming (FlambdaNominalSpec.renaming) permutes variables. A
   prover is asked about a type in a typing environment, and each
   statement says: asked about the renamed type in the renamed
   environment, it answers what it answers about the type in the
   environment, renamed. Where the one stops ([None]) the other does.
   - An answer that holds no variable is the same answer
     ([prover_invariant]): a boolean, a set of numbers, of tags, of
     strings or of code ids, a tag with a shape and a size, the
     constructors of a variant, an allocation mode.
   - An answer that holds a [Simple] or a type is renamed
     ([prover_equivariant]): the [Simple] of an alias, of a field or
     of a projection; the contents of a boxed number; the entry of a
     closure.
   The other arguments of a prover (a kind, a name mode, an index, a
   slot) hold no variable and are the same on both sides. The variable
   that [never_holds_locally_allocated_values] is asked about is
   renamed.

   Each statement is an equation and holds of every environment and
   every type: nothing is asked of them, well-formedness included. No
   prover makes a name, and none goes by the order of names: where a
   canonical element is chosen, it is by binding times, which names
   keep (FlambdaEnvRenameSpec.v). A renaming is a permutation, so two
   variables are equal exactly when their renamings are, which is
   what [prove_physical_equality] reads of two variables.

   The compiler has no such statements; its provers are not given
   renamed environments. They are here because every function that
   reads names is to commute with renaming, and because the rewrites
   that follow a prover's answer are to commute with it too: those
   take these equations.

   The expansion of a head ([expand_head], FlambdaExpandHead.v) is the
   meet's; that it commutes with renaming is stated here
   ([expand_head_rename_statement]), since the provers are its first
   readers under a renaming and the renaming of environments came
   after the meet's files. The view of an expanded type that the
   provers take ([Provers.expand_head], [Provers.is_bottom]) has its
   statements beside it.

   Reification ([Reify.reify]) takes, beside the environment and the
   type, an environment in which the variables answered must be
   defined and two tests of variables. The first is renamed; the
   tests on the two sides are any two that agree through the renaming
   (the premises of [reify_rename_statement]).

   Nothing is trusted here. The proofs are in
   FlambdaProversRenameProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeAccessSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaTypeRenameSpec FlambdaEnvRenameSpec
  FlambdaExpandHead FlambdaProvers FlambdaReify.
Import ListNotations.

(* ======================================================================== *)
(* The expansion of a head                                                  *)
(* ======================================================================== *)

(* A head of values or of naked immediates holds names; the others do
   not. *)
Definition rename_expanded_head (r : renaming) (head : Expanded_type.head)
    : Expanded_type.head :=
  match head with
  | Expanded_type.Head_value head =>
    Expanded_type.Head_value (rename_head_of_kind_value r head)
  | Expanded_type.Head_naked_immediate head =>
    Expanded_type.Head_naked_immediate
      (rename_head_of_kind_naked_immediate r head)
  | Expanded_type.Head_naked_number k head =>
    Expanded_type.Head_naked_number k head
  | Expanded_type.Head_rec_info head => Expanded_type.Head_rec_info head
  | Expanded_type.Head_region head => Expanded_type.Head_region head
  end.

(* An expanded type: the kind is kept. *)
Definition rename_expanded (r : renaming) (expanded : Expanded_type.t)
    : Expanded_type.t :=
  Expanded_type.create (Expanded_type.kind expanded)
    (or_unknown_or_bottom_map (rename_expanded_head r)
       (Expanded_type.descr expanded)).

(* The view that the provers take of an expanded type. *)
Definition rename_descr_oub (r : renaming) (descr : descr_oub) : descr_oub :=
  match descr with
  | Oub_value head =>
    Oub_value (or_unknown_or_bottom_map (rename_head_of_kind_value r) head)
  | Oub_naked_immediate head =>
    Oub_naked_immediate
      (or_unknown_or_bottom_map (rename_head_of_kind_naked_immediate r) head)
  | Oub_naked_number k head => Oub_naked_number k head
  | Oub_rec_info head => Oub_rec_info head
  | Oub_region head => Oub_region head
  end.

(* [Expand_head.expand_head]: the head of the renamed type in the
   renamed environment is the renamed head. *)
Definition expand_head_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (t : flambda_type),
    expand_head (rename_typing_env r tenv) (rename_type r t)
    = option_map (rename_expanded r) (expand_head tenv t).

(* ======================================================================== *)
(* The answers, renamed                                                     *)
(* ======================================================================== *)

(* An answer of each of the three types of answers, with what it holds
   renamed by a function. *)
Definition generic_proof_map {A B : Type} (f : A -> B)
    (proof : Generic_proof.t A) : Generic_proof.t B :=
  match proof with
  | Generic_proof.Proved a => Generic_proof.Proved (f a)
  | Generic_proof.Unknown => Generic_proof.Unknown
  | Generic_proof.Invalid => Generic_proof.Invalid
  end.

Definition proof_of_property_map {A B : Type} (f : A -> B)
    (proof : Proof_of_property.t A) : Proof_of_property.t B :=
  match proof with
  | Proof_of_property.Proved a => Proof_of_property.Proved (f a)
  | Proof_of_property.Unknown => Proof_of_property.Unknown
  end.

Definition meet_shortcut_map {A B : Type} (f : A -> B)
    (shortcut : Meet_shortcut.t A) : Meet_shortcut.t B :=
  match shortcut with
  | Meet_shortcut.Known_result a => Meet_shortcut.Known_result (f a)
  | Meet_shortcut.Need_meet => Meet_shortcut.Need_meet
  | Meet_shortcut.Invalid => Meet_shortcut.Invalid
  end.

(* A boxed number: the type of its contents is renamed. *)
Definition rename_boxed_or_tagged_number (r : renaming)
    (number : Boxed_or_tagged_number.t) : Boxed_or_tagged_number.t :=
  match number with
  | Boxed_or_tagged_number.Boxed alloc_mode k contents =>
    Boxed_or_tagged_number.Boxed alloc_mode k (rename_type r contents)
  | Boxed_or_tagged_number.Tagged_immediate =>
    Boxed_or_tagged_number.Tagged_immediate
  end.

(* What [prove_unique_fully_constructed_immutable_heap_block] answers:
   the [Simple]s of the fields are renamed. *)
Definition rename_block_with_fields (r : renaming)
    (block : Z * block_shape * nat * list simple)
    : Z * block_shape * nat * list simple :=
  match block with
  | (tag, shape, size, fields) =>
    (tag, shape, size, map (rename_simple r) fields)
  end.

(* What the provers of a single closures entry answer: the entry is
   renamed. A function type holds no variable. *)
Definition rename_single_closures_entry (r : renaming)
    (single : function_slot * Alloc_mode.t * closures_entry * function_type)
    : function_slot * Alloc_mode.t * closures_entry * function_type :=
  match single with
  | (slot, alloc_mode, entry, function_type) =>
    (slot, alloc_mode, rename_closures_entry r entry, function_type)
  end.

(* What reification answers: the fields of a block that is lifted, and
   a [Simple]. *)
Definition rename_to_lift (r : renaming) (to_lift : To_lift.t) : To_lift.t :=
  match to_lift with
  | To_lift.Immutable_block tag is_unique shape fields =>
    To_lift.Immutable_block tag is_unique shape
      (map (rename_simple r) fields)
  | To_lift.Boxed_number k n => To_lift.Boxed_number k n
  end.

Definition rename_reification_result (r : renaming)
    (result : Reification_result.t) : Reification_result.t :=
  match result with
  | Reification_result.Lift to_lift =>
    Reification_result.Lift (rename_to_lift r to_lift)
  | Reification_result.Simple simple =>
    Reification_result.Simple (rename_simple r simple)
  | Reification_result.Cannot_reify => Reification_result.Cannot_reify
  | Reification_result.Invalid => Reification_result.Invalid
  end.

(* ======================================================================== *)
(* The two forms                                                            *)
(* ======================================================================== *)

(* A prover whose answer holds no variable: it answers the same of the
   renamed type in the renamed environment. *)
Definition prover_invariant {A : Type}
    (prover : Typing_env.t -> flambda_type -> option A) : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (t : flambda_type),
    prover (rename_typing_env r tenv) (rename_type r t) = prover tenv t.

(* A prover whose answer holds variables: it answers the renamed
   answer. *)
Definition prover_equivariant {A : Type} (rename_answer : renaming -> A -> A)
    (prover : Typing_env.t -> flambda_type -> option A) : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (t : flambda_type),
    prover (rename_typing_env r tenv) (rename_type r t)
    = option_map (rename_answer r) (prover tenv t).

(* ======================================================================== *)
(* The view of an expanded type                                             *)
(* ======================================================================== *)

Definition is_bottom_rename_statement : Prop :=
  prover_invariant Provers.is_bottom.

Definition provers_expand_head_rename_statement : Prop :=
  prover_equivariant rename_descr_oub Provers.expand_head.

(* ======================================================================== *)
(* The scrutinee questions and the numbers                                  *)
(* ======================================================================== *)

Definition prove_is_int_rename_statement : Prop :=
  prover_invariant Provers.prove_is_int.

Definition meet_is_int_variant_only_rename_statement : Prop :=
  prover_invariant Provers.meet_is_int_variant_only.

Definition prove_is_not_a_pointer_rename_statement : Prop :=
  prover_invariant Provers.prove_is_not_a_pointer.

Definition prove_get_tag_rename_statement : Prop :=
  prover_invariant Provers.prove_get_tag.

Definition meet_is_null_rename_statement : Prop :=
  prover_invariant Provers.meet_is_null.

Definition meet_naked_immediates_rename_statement : Prop :=
  prover_invariant Provers.meet_naked_immediates.

Definition prove_equals_tagged_immediates_rename_statement : Prop :=
  prover_invariant Provers.prove_equals_tagged_immediates.

Definition meet_equals_tagged_immediates_rename_statement : Prop :=
  prover_invariant Provers.meet_equals_tagged_immediates.

Definition meet_equals_single_tagged_immediate_rename_statement : Prop :=
  prover_invariant Provers.meet_equals_single_tagged_immediate.

(* The eleven [meet_naked_float32s], ..., [meet_naked_masks]. *)
Definition meet_naked_number_rename_statement : Prop :=
  ∀ kind : naked_number_kind,
    prover_invariant (Provers.meet_naked_number kind).

(* ======================================================================== *)
(* Aliases, boxed numbers and strings                                       *)
(* ======================================================================== *)

Definition prove_equals_to_simple_of_kind_rename_statement : Prop :=
  ∀ kind : flambda_kind,
    prover_equivariant (fun r => proof_of_property_map (rename_simple r))
      (fun tenv t => Provers.prove_equals_to_simple_of_kind tenv t kind).

Definition prove_tagging_of_simple_rename_statement : Prop :=
  ∀ min_name_mode : Name_mode.t,
    prover_equivariant (fun r => proof_of_property_map (rename_simple r))
      (fun tenv t => Provers.prove_tagging_of_simple tenv min_name_mode t).

Definition meet_tagging_of_simple_rename_statement : Prop :=
  ∀ min_name_mode : Name_mode.t,
    prover_equivariant (fun r => meet_shortcut_map (rename_simple r))
      (fun tenv t => Provers.meet_tagging_of_simple tenv min_name_mode t).

Definition prove_is_a_boxed_or_tagged_number_rename_statement : Prop :=
  prover_equivariant
    (fun r => proof_of_property_map (rename_boxed_or_tagged_number r))
    Provers.prove_is_a_boxed_or_tagged_number.

Definition prove_is_a_tagged_immediate_rename_statement : Prop :=
  prover_invariant Provers.prove_is_a_tagged_immediate.

(* The nine [prove_is_a_boxed_float32], ..., [prove_is_a_boxed_mask]. *)
Definition prove_is_a_boxed_number_rename_statement : Prop :=
  ∀ kind : boxable_number,
    prover_invariant (Provers.prove_is_a_boxed_number kind).

Definition prove_is_or_is_not_a_boxed_float_rename_statement : Prop :=
  prover_invariant Provers.prove_is_or_is_not_a_boxed_float.

(* The nine [meet_boxed_float32_containing_simple], ...,
   [meet_boxed_mask_containing_simple]. *)
Definition meet_boxed_number_containing_simple_rename_statement : Prop :=
  ∀ (kind : boxable_number) (min_name_mode : Name_mode.t),
    prover_equivariant (fun r => meet_shortcut_map (rename_simple r))
      (fun tenv t =>
         Provers.meet_boxed_number_containing_simple kind tenv min_name_mode
           t).

(* [prove_nothing], which answers [Unknown] of everything. *)
Definition prove_nothing_rename_statement : Prop :=
  ∀ A : Type, prover_invariant (@Provers.prove_nothing A).

Definition prove_strings_rename_statement : Prop :=
  prover_invariant Provers.prove_strings.

Definition meet_strings_rename_statement : Prop :=
  prover_invariant Provers.meet_strings.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Definition prove_variant_like_rename_statement : Prop :=
  prover_invariant Provers.prove_variant_like.

Definition meet_variant_like_rename_statement : Prop :=
  prover_invariant Provers.meet_variant_like.

Definition prove_unique_tag_and_size_rename_statement : Prop :=
  prover_invariant Provers.prove_unique_tag_and_size.

Definition prove_unique_immutable_heap_block_rename_statement : Prop :=
  prover_equivariant
    (fun r => proof_of_property_map (rename_block_with_fields r))
    Provers.prove_unique_fully_constructed_immutable_heap_block.

Definition meet_block_field_simple_rename_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (field_kind : flambda_kind)
    (field_index : nat),
    prover_equivariant (fun r => meet_shortcut_map (rename_simple r))
      (fun tenv t =>
         Provers.meet_block_field_simple tenv min_name_mode field_kind t
           field_index).

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

Definition prove_single_closures_entry_rename_statement : Prop :=
  prover_equivariant
    (fun r => proof_of_property_map (rename_single_closures_entry r))
    Provers.prove_single_closures_entry.

Definition meet_single_closures_entry_rename_statement : Prop :=
  prover_equivariant
    (fun r => meet_shortcut_map (rename_single_closures_entry r))
    Provers.meet_single_closures_entry.

Definition prove_code_ids_rename_statement : Prop :=
  prover_invariant Provers.prove_code_ids.

Definition meet_code_ids_rename_statement : Prop :=
  prover_invariant Provers.meet_code_ids.

Definition meet_project_function_slot_simple_rename_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (slot : function_slot),
    prover_equivariant (fun r => meet_shortcut_map (rename_simple r))
      (fun tenv t =>
         Provers.meet_project_function_slot_simple tenv min_name_mode t slot).

Definition meet_project_value_slot_simple_rename_statement : Prop :=
  ∀ (min_name_mode : Name_mode.t) (slot : value_slot)
    (value_slot_kind : flambda_kind),
    prover_equivariant (fun r => meet_shortcut_map (rename_simple r))
      (fun tenv t =>
         Provers.meet_project_value_slot_simple tenv min_name_mode t slot
           value_slot_kind).

(* ======================================================================== *)
(* Physical equality and allocation modes                                   *)
(* ======================================================================== *)

(* Two types, both renamed. *)
Definition prove_physical_equality_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (t1 t2 : flambda_type),
    Provers.prove_physical_equality (rename_typing_env r tenv)
      (rename_type r t1) (rename_type r t2)
    = Provers.prove_physical_equality tenv t1 t2.

Definition prove_alloc_mode_of_boxed_number_rename_statement : Prop :=
  prover_invariant Provers.prove_alloc_mode_of_boxed_number.

(* The prover that is asked about a variable: the variable is
   renamed. *)
Definition never_holds_locally_allocated_values_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (var : variable),
    Provers.never_holds_locally_allocated_values (rename_typing_env r tenv)
      (rename_variable r var)
    = Provers.never_holds_locally_allocated_values tenv var.

(* ======================================================================== *)
(* Reification                                                              *)
(* ======================================================================== *)

(* The environment in which the variables answered must be defined is
   renamed, as the environment of the type is. The two tests of
   variables are functions that the caller gives: those of the renamed
   side answer of a renamed variable what those of the other side
   answer of the variable. *)
Definition reify_rename_statement : Prop :=
  ∀ (r : renaming) (allowed_if_free_vars_defined_in : Typing_env.t)
    (var_is_defined_at_toplevel var_is_symbol_projection
     renamed_var_is_defined_at_toplevel renamed_var_is_symbol_projection
     : variable -> bool)
    (tenv : Typing_env.t) (t : flambda_type),
    (∀ var : variable,
      renamed_var_is_defined_at_toplevel (rename_variable r var)
      = var_is_defined_at_toplevel var) ->
    (∀ var : variable,
      renamed_var_is_symbol_projection (rename_variable r var)
      = var_is_symbol_projection var) ->
    Reify.reify (rename_typing_env r allowed_if_free_vars_defined_in)
      renamed_var_is_defined_at_toplevel renamed_var_is_symbol_projection
      (rename_typing_env r tenv) (rename_type r t)
    = option_map (rename_reification_result r)
        (Reify.reify allowed_if_free_vars_defined_in
           var_is_defined_at_toplevel var_is_symbol_projection tenv t).
