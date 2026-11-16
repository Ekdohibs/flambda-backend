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


(* The equality test of the names of types decides equality, and
   induction on the grammar of types. Nothing here needs review.

   [flambda_type] is defined together with seven other types, and holds
   lists and maps of itself, so Rocq gives no induction principle that is
   of use. [types_induction] is one: to prove something of every type,
   and with it of every head, row-like structure, entry of closures and
   environment extension, prove it of each constructor from what holds of
   its parts. For the parts held in a list or a map, what holds is given
   for every element of the list ([a ∈ l]), which is more than for every
   binding that [lookup] finds.

   Every proof by induction on types goes through this, so that none has
   to be written as a recursive definition that Rocq must check. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar.
Import ListNotations.

(* [name_eqb] tells whether two names are the same: the proof to give
   with it wherever [Keyed] asks for one. *)
Lemma name_eqb_eq : decides_equality name_eqb.
Proof.
  intros [x | s] [x' | s']; simpl;
    try (split; [ discriminate | discriminate ]).
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
Qed.

(* [f] proves [P] of every element of a list; it is used with the
   induction being defined for [f], and is accepted because [f] is only
   applied to elements of the list. *)
Definition list_all {A : Type} (P : A -> Prop) (f : ∀ a : A, P a)
    : ∀ (l : list A) (a : A), a ∈ l -> P a :=
  fix list_all (l : list A) : ∀ a : A, a ∈ l -> P a :=
    match l with
    | [] => fun a Hin => match Hin with end
    | a' :: l =>
      fun a Hin =>
        match Hin with
        | or_introl Heq => eq_ind a' P (f a') a Heq
        | or_intror Hin => list_all l a Hin
        end
    end.

Definition bindings_all {K A : Type} (P : A -> Prop) (f : ∀ a : A, P a)
    : ∀ (l : list (K * A)) (k : K) (a : A), (k, a) ∈ l -> P a :=
  fix bindings_all (l : list (K * A)) : ∀ (k : K) (a : A), (k, a) ∈ l -> P a :=
    match l with
    | [] => fun k a Hin => match Hin with end
    | (k', a') :: l =>
      fun k a Hin =>
        match Hin with
        | or_introl Heq => eq_ind a' P (f a') a (f_equal snd Heq)
        | or_intror Hin => bindings_all l k a Hin
        end
    end.

Section Induction.

Variable P_type : flambda_type -> Prop.
Variable P_head : head_of_kind_value -> Prop.
Variable P_non_null : head_of_kind_value_non_null -> Prop.
Variable P_blocks : row_like_for_blocks -> Prop.
Variable P_closures : row_like_for_closures -> Prop.
Variable P_entry : closures_entry -> Prop.
Variable P_extension : env_extension -> Prop.

Definition P_block_case (case : row_like_block_case) : Prop :=
  (∀ ty : flambda_type, ty ∈ case_maps_to case -> P_type ty)
  ∧ P_extension (case_env_extension case).

Definition P_closures_case (case : row_like_closures_case) : Prop :=
  P_entry (case_maps_to case) ∧ P_extension (case_env_extension case).

Hypothesis H_value :
  ∀ ty : type_descr head_of_kind_value,
    match ty with
    | Ok (No_alias head) => P_head head
    | _ => True
    end ->
    P_type (Type_value ty).
Hypothesis H_naked_immediate :
  ∀ ty : type_descr head_of_kind_naked_immediate,
    P_type (Type_naked_immediate ty).
Hypothesis H_naked_number :
  ∀ (k : naked_number_kind) (ty : type_descr (list Z)),
    P_type (Type_naked_number k ty).
Hypothesis H_rec_info : ∀ ty : type_descr unit, P_type (Type_rec_info ty).
Hypothesis H_region : ∀ ty : type_descr unit, P_type (Type_region ty).
Hypothesis H_head :
  ∀ (non_null : or_unknown_or_bottom head_of_kind_value_non_null)
    (is_null : is_null),
    match non_null with
    | Ok head => P_non_null head
    | _ => True
    end ->
    P_head (Head_of_kind_value non_null is_null).
Hypothesis H_variant :
  ∀ (is_int : option variable) (immediates : Or_unknown.t flambda_type)
    (get_tag : option variable) (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) (is_unique : bool),
    match immediates with
    | Or_unknown.Known ty => P_type ty
    | Or_unknown.Unknown => True
    end ->
    match blocks with
    | Or_unknown.Known blocks => P_blocks blocks
    | Or_unknown.Unknown => True
    end ->
    match extensions with
    | No_extensions => True
    | Ext when_immediate when_block =>
      P_extension when_immediate ∧ P_extension when_block
    end ->
    P_non_null (Variant is_int immediates get_tag blocks extensions is_unique).
Hypothesis H_mutable_block :
  ∀ alloc_mode : Alloc_mode.t, P_non_null (Mutable_block alloc_mode).
Hypothesis H_boxed_number :
  ∀ (k : boxable_number) (contents : flambda_type)
    (alloc_mode : Alloc_mode.t),
    P_type contents -> P_non_null (Boxed_number k contents alloc_mode).
Hypothesis H_closures :
  ∀ (by_function_slot : row_like_for_closures) (alloc_mode : Alloc_mode.t),
    P_closures by_function_slot ->
    P_non_null (Closures by_function_slot alloc_mode).
Hypothesis H_string : ∀ strs : list (list Z), P_non_null (String strs).
Hypothesis H_blocks :
  ∀ (known_tags : Keyed.finite_map Z (Or_unknown.t row_like_block_case))
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode : Alloc_mode.t),
    (∀ (tag : Z) (case : row_like_block_case),
      (tag, Or_unknown.Known case) ∈ known_tags -> P_block_case case) ->
    match other_tags with
    | Or_bottom.Ok case => P_block_case case
    | Or_bottom.Bottom => True
    end ->
    P_blocks (Row_like_for_blocks known_tags other_tags alloc_mode).
Hypothesis H_row_closures :
  ∀ known_closures : finite_map row_like_closures_case,
    (∀ (slot : function_slot) (case : row_like_closures_case),
      (slot, case) ∈ known_closures -> P_closures_case case) ->
    P_closures (Row_like_for_closures known_closures).
Hypothesis H_entry :
  ∀ (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type),
    (∀ (g : function_slot) (ty : flambda_type),
      (g, ty) ∈ closure_types -> P_type ty) ->
    (∀ (w : value_slot) (ty : flambda_type),
      (w, ty) ∈ value_slot_types -> P_type ty) ->
    P_entry (Closures_entry function_types closure_types value_slot_types).
Hypothesis H_extension :
  ∀ equations : Keyed.finite_map name flambda_type,
    (∀ (n : name) (ty : flambda_type), (n, ty) ∈ equations -> P_type ty) ->
    P_extension (Env_extension equations).

Fixpoint type_induction (t : flambda_type) : P_type t :=
  match t with
  | Type_value ty =>
    H_value ty
      match ty with
      | Ok (No_alias head) => head_induction head
      | Unknown | Bottom | Ok (Equals _) => I
      end
  | Type_naked_immediate ty => H_naked_immediate ty
  | Type_naked_number k ty => H_naked_number k ty
  | Type_rec_info ty => H_rec_info ty
  | Type_region ty => H_region ty
  end

with head_induction (head : head_of_kind_value) : P_head head :=
  match head with
  | Head_of_kind_value non_null is_null =>
    H_head non_null is_null
      match non_null with
      | Ok head => non_null_induction head
      | Unknown | Bottom => I
      end
  end

with non_null_induction (head : head_of_kind_value_non_null)
    : P_non_null head :=
  match head with
  | Variant is_int immediates get_tag blocks extensions is_unique =>
    H_variant is_int immediates get_tag blocks extensions is_unique
      match immediates with
      | Or_unknown.Known ty => type_induction ty
      | Or_unknown.Unknown => I
      end
      match blocks with
      | Or_unknown.Known blocks => blocks_induction blocks
      | Or_unknown.Unknown => I
      end
      match extensions with
      | No_extensions => I
      | Ext when_immediate when_block =>
        conj (extension_induction when_immediate)
          (extension_induction when_block)
      end
  | Mutable_block alloc_mode => H_mutable_block alloc_mode
  | Boxed_number k contents alloc_mode =>
    H_boxed_number k contents alloc_mode (type_induction contents)
  | Closures by_function_slot alloc_mode =>
    H_closures by_function_slot alloc_mode
      (closures_induction by_function_slot)
  | String strs => H_string strs
  end

with blocks_induction (blocks : row_like_for_blocks) : P_blocks blocks :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags alloc_mode =>
    let block_case (case : row_like_block_case) : P_block_case case :=
      conj (list_all P_type type_induction (case_maps_to case))
        (extension_induction (case_env_extension case))
    in
    H_blocks known_tags other_tags alloc_mode
      (fun tag case Hin =>
         bindings_all
           (fun case =>
              match case with
              | Or_unknown.Known case => P_block_case case
              | Or_unknown.Unknown => True
              end)
           (fun case =>
              match case with
              | Or_unknown.Known case => block_case case
              | Or_unknown.Unknown => I
              end)
           known_tags tag (Or_unknown.Known case) Hin)
      match other_tags with
      | Or_bottom.Ok case => block_case case
      | Or_bottom.Bottom => I
      end
  end

with closures_induction (closures : row_like_for_closures)
    : P_closures closures :=
  match closures with
  | Row_like_for_closures known_closures =>
    H_row_closures known_closures
      (bindings_all P_closures_case
         (fun case =>
            conj (entry_induction (case_maps_to case))
              (extension_induction (case_env_extension case)))
         known_closures)
  end

with entry_induction (entry : closures_entry) : P_entry entry :=
  match entry with
  | Closures_entry function_types closure_types value_slot_types =>
    H_entry function_types closure_types value_slot_types
      (bindings_all P_type type_induction closure_types)
      (bindings_all P_type type_induction value_slot_types)
  end

with extension_induction (extension : env_extension)
    : P_extension extension :=
  match extension with
  | Env_extension equations =>
    H_extension equations (bindings_all P_type type_induction equations)
  end.

(* All seven at once. *)
Theorem types_induction :
  (∀ t : flambda_type, P_type t)
  ∧ (∀ head : head_of_kind_value, P_head head)
  ∧ (∀ head : head_of_kind_value_non_null, P_non_null head)
  ∧ (∀ blocks : row_like_for_blocks, P_blocks blocks)
  ∧ (∀ closures : row_like_for_closures, P_closures closures)
  ∧ (∀ entry : closures_entry, P_entry entry)
  ∧ (∀ extension : env_extension, P_extension extension).
Proof.
  repeat split;
    [ exact type_induction | exact head_induction | exact non_null_induction
    | exact blocks_induction | exact closures_induction
    | exact entry_induction | exact extension_induction ].
Qed.

End Induction.
