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


(* Proofs of the statements of FlambdaTypeRenameSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeMeaningBoolSpec FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaTypeRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists and the maps over what may be unknown or bottom                    *)
(* ======================================================================== *)

Lemma map_fixes {A : Type} (f : A -> A) (l : list A) :
  (∀ a : A, a ∈ l -> f a = a) -> map f l = l.
Proof.
  intros Hfixes. rewrite <- (map_id l) at 2. apply map_ext_in. exact Hfixes.
Qed.

Lemma map_factors {A B C : Type} (f : A -> C) (g : B -> C) (h : A -> B)
    (l : list A) :
  (∀ a : A, a ∈ l -> f a = g (h a)) -> map f l = map g (map h l).
Proof. intros Hfactors. rewrite map_map. apply map_ext_in. exact Hfactors. Qed.

Lemma flat_map_ext_in' {A B : Type} (f g : A -> list B) (l : list A) :
  (∀ a : A, a ∈ l -> f a = g a) -> flat_map f l = flat_map g l.
Proof.
  induction l as [ | a l IH]; intros Hext; simpl; [ reflexivity | ].
  rewrite (Hext a (or_introl eq_refl)), IH; [ reflexivity | ].
  intros a' Hin. exact (Hext a' (or_intror Hin)).
Qed.

Lemma flat_map_map' {A B C : Type} (f : B -> list C) (g : A -> B)
    (l : list A) :
  flat_map f (map g l) = flat_map (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

Lemma map_flat_map' {A B C : Type} (h : B -> C) (f : A -> list B)
    (l : list A) :
  map h (flat_map f l) = flat_map (fun a => map h (f a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite map_app, IH. reflexivity.
Qed.

Lemma flat_map_flat_map' {A B C : Type} (g : B -> list C) (f : A -> list B)
    (l : list A) :
  flat_map g (flat_map f l) = flat_map (fun a => flat_map g (f a)) l.
Proof.
  induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite flat_map_app, IH. reflexivity.
Qed.

Section Maps.

Context {A B C : Type}.

Lemma or_unknown_map_fixes (f : A -> A) (a : Or_unknown.t A) :
  (∀ x : A, a = Or_unknown.Known x -> f x = x) -> or_unknown_map f a = a.
Proof.
  destruct a as [x | ]; simpl; [ | reflexivity ].
  intros Hfixes. rewrite (Hfixes x eq_refl). reflexivity.
Qed.

Lemma or_bottom_map_fixes (f : A -> A) (a : Or_bottom.t A) :
  (∀ x : A, a = Or_bottom.Ok x -> f x = x) -> or_bottom_map f a = a.
Proof.
  destruct a as [x | ]; simpl; [ | reflexivity ].
  intros Hfixes. rewrite (Hfixes x eq_refl). reflexivity.
Qed.

Lemma or_unknown_or_bottom_map_fixes (f : A -> A)
    (a : or_unknown_or_bottom A) :
  (∀ x : A, a = Ok x -> f x = x) -> or_unknown_or_bottom_map f a = a.
Proof.
  destruct a as [ | x | ]; simpl; [ reflexivity | | reflexivity ].
  intros Hfixes. rewrite (Hfixes x eq_refl). reflexivity.
Qed.

Lemma or_unknown_map_factors (f : A -> C) (g : B -> C) (h : A -> B)
    (a : Or_unknown.t A) :
  (∀ x : A, a = Or_unknown.Known x -> f x = g (h x)) ->
  or_unknown_map f a = or_unknown_map g (or_unknown_map h a).
Proof.
  destruct a as [x | ]; simpl; [ | reflexivity ].
  intros Hfactors. rewrite (Hfactors x eq_refl). reflexivity.
Qed.

Lemma or_bottom_map_factors (f : A -> C) (g : B -> C) (h : A -> B)
    (a : Or_bottom.t A) :
  (∀ x : A, a = Or_bottom.Ok x -> f x = g (h x)) ->
  or_bottom_map f a = or_bottom_map g (or_bottom_map h a).
Proof.
  destruct a as [x | ]; simpl; [ | reflexivity ].
  intros Hfactors. rewrite (Hfactors x eq_refl). reflexivity.
Qed.

Lemma or_unknown_or_bottom_map_factors (f : A -> C) (g : B -> C)
    (h : A -> B) (a : or_unknown_or_bottom A) :
  (∀ x : A, a = Ok x -> f x = g (h x)) ->
  or_unknown_or_bottom_map f a
  = or_unknown_or_bottom_map g (or_unknown_or_bottom_map h a).
Proof.
  destruct a as [ | x | ]; simpl; [ reflexivity | | reflexivity ].
  intros Hfactors. rewrite (Hfactors x eq_refl). reflexivity.
Qed.

Lemma keyed_map_values_fixes {K : Type} (f : A -> A)
    (m : Keyed.finite_map K A) :
  (∀ (key : K) (a : A), (key, a) ∈ m -> f a = a) -> Keyed.map_values f m = m.
Proof.
  intros Hfixes. apply map_fixes. intros [key a] Hin. simpl.
  rewrite (Hfixes key a Hin). reflexivity.
Qed.

Lemma keyed_map_values_factors {K : Type} (f : A -> C) (g : B -> C)
    (h : A -> B) (m : Keyed.finite_map K A) :
  (∀ (key : K) (a : A), (key, a) ∈ m -> f a = g (h a)) ->
  Keyed.map_values f m = Keyed.map_values g (Keyed.map_values h m).
Proof.
  intros Hfactors. unfold Keyed.map_values. rewrite map_map.
  apply map_ext_in. intros [key a] Hin. simpl.
  rewrite (Hfactors key a Hin). reflexivity.
Qed.

(* The same of the maps keyed by slots. *)
Lemma map_values_fixes (f : A -> A) (m : finite_map A) :
  (∀ (slot : positive) (a : A), (slot, a) ∈ m -> f a = a) ->
  map_values f m = m.
Proof. exact (keyed_map_values_fixes f m). Qed.

Lemma map_values_factors (f : A -> C) (g : B -> C) (h : A -> B)
    (m : finite_map A) :
  (∀ (slot : positive) (a : A), (slot, a) ∈ m -> f a = g (h a)) ->
  map_values f m = map_values g (map_values h m).
Proof. exact (keyed_map_values_factors f g h m). Qed.

End Maps.

(* ======================================================================== *)
(* The identity                                                             *)
(* ======================================================================== *)

Lemma rename_name_identity (n : name) : rename_name identity_renaming n = n.
Proof. destruct n; reflexivity. Qed.

Lemma rename_simple_identity (s : simple) :
  rename_simple identity_renaming s = s.
Proof. destruct s; reflexivity. Qed.

Lemma rename_option_identity (x : option variable) :
  option_map (rename_variable identity_renaming) x = x.
Proof. destruct x; reflexivity. Qed.

Lemma rename_descr_identity {head : Type} (rename_head : head -> head)
    (ty : type_descr head) :
  (∀ h : head, ty = Ok (No_alias h) -> rename_head h = h) ->
  rename_descr identity_renaming rename_head ty = ty.
Proof.
  intros Hhead. apply or_unknown_or_bottom_map_fixes. intros [h | s] Hty.
  - rewrite (Hhead h Hty). reflexivity.
  - rewrite rename_simple_identity. reflexivity.
Qed.

Lemma rename_block_case_identity (case : row_like_block_case) :
  P_block_case (fun t => rename_type identity_renaming t = t)
    (fun extension =>
       rename_env_extension identity_renaming extension = extension)
    case ->
  rename_block_case (rename_type identity_renaming)
    (rename_env_extension identity_renaming) case
  = case.
Proof.
  destruct case as [maps_to index extension]. unfold rename_block_case.
  simpl. intros [Htypes Hextension]. simpl in Htypes, Hextension.
  rewrite (map_fixes _ _ Htypes), Hextension. reflexivity.
Qed.

Lemma rename_type_identity_all :
  (∀ t : flambda_type, rename_type identity_renaming t = t)
  ∧ (∀ head : head_of_kind_value,
      rename_head_of_kind_value identity_renaming head = head)
  ∧ (∀ head : head_of_kind_value_non_null,
      rename_head_of_kind_value_non_null identity_renaming head = head)
  ∧ (∀ blocks : row_like_for_blocks,
      rename_row_like_for_blocks identity_renaming blocks = blocks)
  ∧ (∀ closures : row_like_for_closures,
      rename_row_like_for_closures identity_renaming closures = closures)
  ∧ (∀ entry : closures_entry,
      rename_closures_entry identity_renaming entry = entry)
  ∧ (∀ extension : env_extension,
      rename_env_extension identity_renaming extension = extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. simpl. rewrite rename_descr_identity; [ reflexivity | ].
    intros h ->. exact Hhead.
  - (* Type_naked_immediate *)
    intros ty. simpl. rewrite rename_descr_identity; [ reflexivity | ].
    intros [numbers relations] _.
    unfold rename_head_of_kind_naked_immediate. simpl.
    rewrite map_fixes; [ reflexivity | ].
    intros [relation n] _. simpl. rewrite rename_name_identity. reflexivity.
  - intros k ty. simpl. rewrite rename_descr_identity; reflexivity.
  - intros ty. simpl. rewrite rename_descr_identity; reflexivity.
  - intros ty. simpl. rewrite rename_descr_identity; reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null. simpl.
    rewrite or_unknown_or_bottom_map_fixes.
    + destruct is_null as [ | x]; simpl;
        [ | rewrite rename_option_identity ]; reflexivity.
    + intros h ->. exact Hnon_null.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions. simpl.
    rewrite !rename_option_identity,
      (or_unknown_map_fixes (rename_type identity_renaming) immediates),
      (or_unknown_map_fixes (rename_row_like_for_blocks identity_renaming)
         blocks).
    + destruct extensions as [ | when_immediate when_block];
        [ reflexivity | ].
      destruct Hextensions as [-> ->]. reflexivity.
    + intros row ->. exact Hblocks.
    + intros ty ->. exact Himmediates.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. simpl. rewrite Hcontents.
    reflexivity.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. simpl. rewrite Hclosures.
    reflexivity.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    rewrite keyed_map_values_fixes, or_bottom_map_fixes; [ reflexivity | | ].
    + intros case ->. exact (rename_block_case_identity case Hother).
    + intros tag case Hin. apply or_unknown_map_fixes. intros case' ->.
      exact (rename_block_case_identity case' (Hknown tag case' Hin)).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl.
    rewrite map_values_fixes; [ reflexivity | ].
    intros slot [entry index extension] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    simpl in Hentry, Hextension. unfold rename_closures_case. simpl.
    rewrite Hentry, Hextension. reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl.
    rewrite (map_values_fixes _ closure_types Hclosures),
      (map_values_fixes _ value_slot_types Hslots).
    reflexivity.
  - (* Env_extension *)
    intros equations Htypes. simpl. rewrite map_fixes; [ reflexivity | ].
    intros [n ty] Hin. simpl.
    rewrite rename_name_identity, (Htypes n ty Hin). reflexivity.
Qed.

Theorem rename_type_identity : rename_type_identity_statement.
Proof.
  destruct rename_type_identity_all as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  split; assumption.
Qed.

(* ======================================================================== *)
(* One renaming that does what two do, one after the other                  *)
(* ======================================================================== *)

(* Composition and extensionality are two cases of one fact: if
   [both] renames the free variables of a type as [first] and then
   [second] do, it renames the type as they do. *)
Section Factors.

Variables first second both : renaming.

Definition factors_on (xs : list variable) : Prop :=
  ∀ x : variable,
    x ∈ xs ->
    rename_variable both x = rename_variable second (rename_variable first x).

Lemma factors_on_app (xs ys : list variable) :
  factors_on (xs ++ ys) <-> factors_on xs ∧ factors_on ys.
Proof.
  unfold factors_on. split.
  - intros Hon. split; intros x Hin; apply Hon, in_or_app; auto.
  - intros [Hxs Hys] x Hin. apply in_app_or in Hin as [Hin | Hin]; auto.
Qed.

Lemma factors_on_flat_map {A : Type} (f : A -> list variable) (l : list A)
    (a : A) :
  factors_on (flat_map f l) -> a ∈ l -> factors_on (f a).
Proof. intros Hon Hin x Hx. apply Hon, in_flat_map. exists a. auto. Qed.

Lemma rename_simple_factors (s : simple) :
  factors_on (free_variables_of_simple s) ->
  rename_simple both s = rename_simple second (rename_simple first s).
Proof.
  destruct s as [x | sym | c]; simpl; [ | reflexivity | reflexivity ].
  intros Hon. rewrite (Hon x (or_introl eq_refl)). reflexivity.
Qed.

Lemma rename_name_factors (n : name) :
  factors_on (free_variables_of_name n) ->
  rename_name both n = rename_name second (rename_name first n).
Proof.
  destruct n as [x | sym]; simpl; [ | reflexivity ].
  intros Hon. rewrite (Hon x (or_introl eq_refl)). reflexivity.
Qed.

Lemma rename_option_factors (x : option variable) :
  factors_on (free_variables_of_option x) ->
  option_map (rename_variable both) x
  = option_map (rename_variable second) (option_map (rename_variable first) x).
Proof.
  destruct x as [x | ]; simpl; [ | reflexivity ].
  intros Hon. rewrite (Hon x (or_introl eq_refl)). reflexivity.
Qed.

Lemma rename_descr_factors {head : Type}
    (free_variables_of_head : head -> list variable)
    (f g h : head -> head) (ty : type_descr head) :
  factors_on (free_variables_of_descr free_variables_of_head ty) ->
  (∀ x : head, ty = Ok (No_alias x) -> f x = g (h x)) ->
  rename_descr both f ty = rename_descr second g (rename_descr first h ty).
Proof.
  destruct ty as [ | [x | s] | ]; simpl; intros Hon Hhead;
    [ reflexivity | | | reflexivity ].
  - rewrite (Hhead x eq_refl). reflexivity.
  - rewrite (rename_simple_factors s Hon). reflexivity.
Qed.

Let type_factors (t : flambda_type) : Prop :=
  factors_on (free_variables t) ->
  rename_type both t = rename_type second (rename_type first t).

Let extension_factors (extension : env_extension) : Prop :=
  factors_on (free_variables_of_env_extension extension) ->
  rename_env_extension both extension
  = rename_env_extension second (rename_env_extension first extension).

Lemma rename_block_case_factors (case : row_like_block_case) :
  P_block_case type_factors extension_factors case ->
  factors_on
    (free_variables_of_block_case free_variables
       free_variables_of_env_extension case) ->
  rename_block_case (rename_type both) (rename_env_extension both) case
  = rename_block_case (rename_type second) (rename_env_extension second)
      (rename_block_case (rename_type first) (rename_env_extension first)
         case).
Proof.
  destruct case as [maps_to index extension].
  unfold rename_block_case, free_variables_of_block_case. simpl.
  intros [Htypes Hextension] Hon. simpl in Htypes, Hextension.
  apply factors_on_app in Hon as [Hon_fields Hon_extension].
  rewrite (map_factors _ (rename_type second) (rename_type first) maps_to),
    (Hextension Hon_extension); [ reflexivity | ].
  intros ty Hin.
  exact (Htypes ty Hin (factors_on_flat_map _ _ ty Hon_fields Hin)).
Qed.

Lemma rename_type_factors_all :
  (∀ t : flambda_type, type_factors t)
  ∧ (∀ head : head_of_kind_value,
      factors_on (free_variables_of_head_of_kind_value head) ->
      rename_head_of_kind_value both head
      = rename_head_of_kind_value second
          (rename_head_of_kind_value first head))
  ∧ (∀ head : head_of_kind_value_non_null,
      factors_on (free_variables_of_head_of_kind_value_non_null head) ->
      rename_head_of_kind_value_non_null both head
      = rename_head_of_kind_value_non_null second
          (rename_head_of_kind_value_non_null first head))
  ∧ (∀ blocks : row_like_for_blocks,
      factors_on (free_variables_of_row_like_for_blocks blocks) ->
      rename_row_like_for_blocks both blocks
      = rename_row_like_for_blocks second
          (rename_row_like_for_blocks first blocks))
  ∧ (∀ closures : row_like_for_closures,
      factors_on (free_variables_of_row_like_for_closures closures) ->
      rename_row_like_for_closures both closures
      = rename_row_like_for_closures second
          (rename_row_like_for_closures first closures))
  ∧ (∀ entry : closures_entry,
      factors_on (free_variables_of_closures_entry entry) ->
      rename_closures_entry both entry
      = rename_closures_entry second (rename_closures_entry first entry))
  ∧ (∀ extension : env_extension, extension_factors extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead Hon. simpl.
    rewrite
      (rename_descr_factors free_variables_of_head_of_kind_value _
         (rename_head_of_kind_value second) (rename_head_of_kind_value first)
         ty Hon);
      [ reflexivity | ].
    intros h ->. exact (Hhead Hon).
  - (* Type_naked_immediate *)
    intros ty Hon. simpl.
    rewrite
      (rename_descr_factors free_variables_of_naked_immediate _
         (rename_head_of_kind_naked_immediate second)
         (rename_head_of_kind_naked_immediate first) ty Hon);
      [ reflexivity | ].
    intros [numbers relations] ->. simpl in Hon.
    unfold rename_head_of_kind_naked_immediate. simpl. f_equal.
    apply map_factors. intros [relation n] Hin. simpl.
    rewrite
      (rename_name_factors n
         (factors_on_flat_map _ _ (relation, n) Hon Hin)).
    reflexivity.
  - (* Type_naked_number *)
    intros k ty Hon. simpl.
    rewrite
      (rename_descr_factors (fun _ => []) _ (fun numbers => numbers)
         (fun numbers => numbers) ty Hon);
      reflexivity.
  - (* Type_rec_info *)
    intros ty Hon. simpl.
    rewrite
      (rename_descr_factors (fun _ => []) _ (fun head => head)
         (fun head => head) ty Hon);
      reflexivity.
  - (* Type_region *)
    intros ty Hon. simpl.
    rewrite
      (rename_descr_factors (fun _ => []) _ (fun head => head)
         (fun head => head) ty Hon);
      reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null Hon. simpl in Hon.
    apply factors_on_app in Hon as [Hon_head Hon_null]. simpl.
    rewrite
      (or_unknown_or_bottom_map_factors _
         (rename_head_of_kind_value_non_null second)
         (rename_head_of_kind_value_non_null first) non_null).
    + destruct is_null as [ | x]; simpl; [ reflexivity | ].
      rewrite (rename_option_factors x Hon_null). reflexivity.
    + intros h ->. exact (Hnon_null Hon_head).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions Hon. simpl in Hon.
    apply factors_on_app in Hon as [Hon_is_int Hon].
    apply factors_on_app in Hon as [Hon_immediates Hon].
    apply factors_on_app in Hon as [Hon_get_tag Hon].
    apply factors_on_app in Hon as [Hon_blocks Hon_extensions]. simpl.
    rewrite (rename_option_factors is_int Hon_is_int),
      (rename_option_factors get_tag Hon_get_tag),
      (or_unknown_map_factors _ (rename_type second) (rename_type first)
         immediates),
      (or_unknown_map_factors _ (rename_row_like_for_blocks second)
         (rename_row_like_for_blocks first) blocks).
    + destruct extensions as [ | when_immediate when_block];
        [ reflexivity | ].
      apply factors_on_app in Hon_extensions as [Hon_immediate Hon_block].
      destruct Hextensions as [IHimmediate IHblock].
      rewrite (IHimmediate Hon_immediate), (IHblock Hon_block). reflexivity.
    + intros row ->. exact (Hblocks Hon_blocks).
    + intros ty ->. exact (Himmediates Hon_immediates).
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents Hon. simpl.
    rewrite (Hcontents Hon). reflexivity.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures Hon. simpl.
    rewrite (Hclosures Hon). reflexivity.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother Hon.
    simpl in Hon. apply factors_on_app in Hon as [Hon_known Hon_other].
    simpl.
    rewrite
      (keyed_map_values_factors _
         (or_unknown_map
            (rename_block_case (rename_type second)
               (rename_env_extension second)))
         (or_unknown_map
            (rename_block_case (rename_type first)
               (rename_env_extension first)))
         known_tags),
      (or_bottom_map_factors _
         (rename_block_case (rename_type second)
            (rename_env_extension second))
         (rename_block_case (rename_type first) (rename_env_extension first))
         other_tags);
      [ reflexivity | | ].
    + intros case ->. exact (rename_block_case_factors case Hother Hon_other).
    + intros tag case Hin. apply or_unknown_map_factors. intros case' ->.
      apply (rename_block_case_factors case' (Hknown tag case' Hin)).
      exact (factors_on_flat_map _ _ _ Hon_known Hin).
  - (* Row_like_for_closures *)
    intros known_closures Hknown Hon. simpl.
    rewrite
      (map_values_factors _
         (rename_closures_case (rename_closures_entry second)
            (rename_env_extension second))
         (rename_closures_case (rename_closures_entry first)
            (rename_env_extension first))
         known_closures);
      [ reflexivity | ].
    intros slot [entry index extension] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    simpl in Hentry, Hextension.
    pose proof (factors_on_flat_map _ _ _ Hon Hin) as Hon_case.
    simpl in Hon_case.
    apply factors_on_app in Hon_case as [Hon_entry Hon_extension].
    unfold rename_closures_case. simpl.
    rewrite (Hentry Hon_entry), (Hextension Hon_extension). reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      Hon.
    simpl in Hon. apply factors_on_app in Hon as [Hon_closures Hon_slots].
    simpl.
    rewrite
      (map_values_factors _ (rename_type second) (rename_type first)
         closure_types),
      (map_values_factors _ (rename_type second) (rename_type first)
         value_slot_types);
      [ reflexivity | | ].
    + intros w ty Hin.
      exact (Hslots w ty Hin (factors_on_flat_map _ _ _ Hon_slots Hin)).
    + intros g ty Hin.
      exact (Hclosures g ty Hin (factors_on_flat_map _ _ _ Hon_closures Hin)).
  - (* Env_extension *)
    intros equations Htypes Hon. simpl in Hon. simpl. f_equal.
    apply map_factors. intros [n ty] Hin. simpl.
    pose proof (factors_on_flat_map _ _ _ Hon Hin) as Hon_equation.
    simpl in Hon_equation.
    apply factors_on_app in Hon_equation as [Hon_name Hon_type].
    rewrite (rename_name_factors n Hon_name), (Htypes n ty Hin Hon_type).
    reflexivity.
Qed.

End Factors.

Theorem rename_type_compose : rename_type_compose_statement.
Proof.
  split.
  - intros second first t.
    apply (proj1 (rename_type_factors_all first second _)).
    intros x _. apply rename_variable_compose.
  - intros second first extension.
    destruct (rename_type_factors_all first second
                (compose_renaming second first))
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    apply Hextension. intros x _. apply rename_variable_compose.
Qed.

Theorem rename_type_extensional : rename_type_extensional_statement.
Proof.
  split.
  - intros r r' t Hsame.
    rewrite <- (proj1 rename_type_identity t) at 2.
    apply (proj1 (rename_type_factors_all identity_renaming r' r)).
    exact Hsame.
  - intros r r' extension Hsame.
    rewrite <- (proj2 rename_type_identity extension) at 2.
    destruct (rename_type_factors_all identity_renaming r' r)
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    apply Hextension. exact Hsame.
Qed.

Theorem rename_type_fixes : rename_type_fixes_statement.
Proof.
  split.
  - intros r t Hfixes.
    rewrite (proj1 rename_type_extensional r identity_renaming t Hfixes).
    apply rename_type_identity.
  - intros r extension Hfixes.
    rewrite
      (proj2 rename_type_extensional r identity_renaming extension Hfixes).
    apply rename_type_identity.
Qed.

Theorem rename_type_inverse : rename_type_inverse_statement.
Proof.
  split.
  - intros r t. rewrite <- (proj1 rename_type_compose).
    apply rename_type_fixes. intros x _.
    rewrite rename_variable_compose. apply rename_variable_inverse_l.
  - intros r extension. rewrite <- (proj2 rename_type_compose).
    apply rename_type_fixes. intros x _.
    rewrite rename_variable_compose. apply rename_variable_inverse_l.
Qed.

(* ======================================================================== *)
(* The free names of a renamed type                                         *)
(* ======================================================================== *)

Section Renamed_names.

Variable r : renaming.

Lemma free_names_of_simple_renamed (s : simple) :
  free_names_of_simple (rename_simple r s)
  = map (rename_name r) (free_names_of_simple s).
Proof. destruct s; reflexivity. Qed.

Lemma free_names_of_option_renamed (x : option variable) :
  free_names_of_option (option_map (rename_variable r) x)
  = map (rename_name r) (free_names_of_option x).
Proof. destruct x; reflexivity. Qed.

Lemma free_names_of_descr_renamed {head : Type}
    (free_names_of_head : head -> list name) (rename_head : head -> head)
    (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) ->
    free_names_of_head (rename_head h)
    = map (rename_name r) (free_names_of_head h)) ->
  free_names_of_descr free_names_of_head (rename_descr r rename_head ty)
  = map (rename_name r) (free_names_of_descr free_names_of_head ty).
Proof.
  destruct ty as [ | [h | s] | ]; simpl; intros Hhead;
    [ reflexivity | exact (Hhead h eq_refl)
    | apply free_names_of_simple_renamed | reflexivity ].
Qed.

Let type_renamed (t : flambda_type) : Prop :=
  free_names (rename_type r t) = map (rename_name r) (free_names t).

Let extension_renamed (extension : env_extension) : Prop :=
  free_names_of_env_extension (rename_env_extension r extension)
  = map (rename_name r) (free_names_of_env_extension extension).

Lemma free_names_of_block_case_renamed (case : row_like_block_case) :
  P_block_case type_renamed extension_renamed case ->
  free_names_of_block_case free_names free_names_of_env_extension
    (rename_block_case (rename_type r) (rename_env_extension r) case)
  = map (rename_name r)
      (free_names_of_block_case free_names free_names_of_env_extension case).
Proof.
  destruct case as [maps_to index extension].
  unfold rename_block_case, free_names_of_block_case. simpl.
  intros [Htypes Hextension]. simpl in Htypes, Hextension.
  rewrite map_app, flat_map_map', map_flat_map', Hextension. f_equal.
  apply flat_map_ext_in'. exact Htypes.
Qed.

Lemma free_names_renamed_all :
  (∀ t : flambda_type, type_renamed t)
  ∧ (∀ head : head_of_kind_value,
      free_names_of_head_of_kind_value (rename_head_of_kind_value r head)
      = map (rename_name r) (free_names_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      free_names_of_head_of_kind_value_non_null
        (rename_head_of_kind_value_non_null r head)
      = map (rename_name r) (free_names_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      free_names_of_row_like_for_blocks (rename_row_like_for_blocks r blocks)
      = map (rename_name r) (free_names_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      free_names_of_row_like_for_closures
        (rename_row_like_for_closures r closures)
      = map (rename_name r) (free_names_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      free_names_of_closures_entry (rename_closures_entry r entry)
      = map (rename_name r) (free_names_of_closures_entry entry))
  ∧ (∀ extension : env_extension, extension_renamed extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. unfold type_renamed. simpl.
    apply free_names_of_descr_renamed. intros h ->. exact Hhead.
  - (* Type_naked_immediate *)
    intros ty. unfold type_renamed. simpl.
    apply free_names_of_descr_renamed. intros [numbers relations] _.
    unfold free_names_of_naked_immediate. simpl. rewrite !map_map.
    reflexivity.
  - intros k ty. unfold type_renamed. simpl.
    apply free_names_of_descr_renamed. reflexivity.
  - intros ty. unfold type_renamed. simpl.
    apply free_names_of_descr_renamed. reflexivity.
  - intros ty. unfold type_renamed. simpl.
    apply free_names_of_descr_renamed. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct non_null as [ | head | ], is_null as [ | x]; simpl;
      rewrite ?map_app, ?Hnon_null, ?free_names_of_option_renamed;
      reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    destruct extensions as [ | when_immediate when_block].
    + destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, !free_names_of_option_renamed, ?Himmediates,
          ?Hblocks;
        reflexivity.
    + destruct Hextensions as [Himmediate Hblock].
      destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, !free_names_of_option_renamed, ?Himmediates,
          ?Hblocks, Himmediate, Hblock;
        reflexivity.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    unfold Keyed.map_values.
    rewrite map_app, flat_map_map', map_flat_map'. f_equal.
    + apply flat_map_ext_in'. intros [tag [case | ]] Hin; simpl;
        [ | reflexivity ].
      exact (free_names_of_block_case_renamed case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; simpl; [ | reflexivity ].
      exact (free_names_of_block_case_renamed case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. unfold map_values.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [slot [entry index extension]] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    simpl in Hentry, Hextension. simpl.
    rewrite map_app, Hentry, Hextension. reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. unfold map_values.
    rewrite !flat_map_map', map_app, !map_flat_map'. f_equal;
      apply flat_map_ext_in'; intros [slot ty] Hin; simpl;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. unfold extension_renamed. simpl.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [n ty] Hin. simpl. rewrite (Htypes n ty Hin). reflexivity.
Qed.

End Renamed_names.

Theorem free_names_rename_type : free_names_rename_type_statement.
Proof.
  split.
  - intros r t. exact (proj1 (free_names_renamed_all r) t).
  - intros r extension.
    destruct (free_names_renamed_all r)
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* The variables among the free names                                       *)
(* ======================================================================== *)

Lemma free_variables_of_simple_names (s : simple) :
  free_variables_of_simple s
  = flat_map free_variables_of_name (free_names_of_simple s).
Proof. destruct s; reflexivity. Qed.

Lemma free_variables_of_option_names (x : option variable) :
  free_variables_of_option x
  = flat_map free_variables_of_name (free_names_of_option x).
Proof. destruct x; reflexivity. Qed.

Lemma free_variables_of_descr_names {head : Type}
    (free_variables_of_head : head -> list variable)
    (free_names_of_head : head -> list name) (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) ->
    free_variables_of_head h
    = flat_map free_variables_of_name (free_names_of_head h)) ->
  free_variables_of_descr free_variables_of_head ty
  = flat_map free_variables_of_name
      (free_names_of_descr free_names_of_head ty).
Proof.
  destruct ty as [ | [h | s] | ]; simpl; intros Hhead;
    [ reflexivity | exact (Hhead h eq_refl)
    | apply free_variables_of_simple_names | reflexivity ].
Qed.

Lemma free_variables_of_block_case_names (case : row_like_block_case) :
  P_block_case
    (fun t =>
       free_variables t = flat_map free_variables_of_name (free_names t))
    (fun extension =>
       free_variables_of_env_extension extension
       = flat_map free_variables_of_name
           (free_names_of_env_extension extension))
    case ->
  free_variables_of_block_case free_variables free_variables_of_env_extension
    case
  = flat_map free_variables_of_name
      (free_names_of_block_case free_names free_names_of_env_extension case).
Proof.
  unfold free_variables_of_block_case, free_names_of_block_case.
  intros [Htypes Hextension].
  rewrite flat_map_app, flat_map_flat_map', Hextension. f_equal.
  apply flat_map_ext_in'. exact Htypes.
Qed.

Lemma free_names_variables_all :
  (∀ t : flambda_type,
    free_variables t = flat_map free_variables_of_name (free_names t))
  ∧ (∀ head : head_of_kind_value,
      free_variables_of_head_of_kind_value head
      = flat_map free_variables_of_name
          (free_names_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      free_variables_of_head_of_kind_value_non_null head
      = flat_map free_variables_of_name
          (free_names_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      free_variables_of_row_like_for_blocks blocks
      = flat_map free_variables_of_name
          (free_names_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      free_variables_of_row_like_for_closures closures
      = flat_map free_variables_of_name
          (free_names_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      free_variables_of_closures_entry entry
      = flat_map free_variables_of_name (free_names_of_closures_entry entry))
  ∧ (∀ extension : env_extension,
      free_variables_of_env_extension extension
      = flat_map free_variables_of_name
          (free_names_of_env_extension extension)).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. simpl. apply free_variables_of_descr_names.
    intros h ->. exact Hhead.
  - (* Type_naked_immediate *)
    intros ty. simpl. apply free_variables_of_descr_names.
    intros head _. unfold free_variables_of_naked_immediate,
      free_names_of_naked_immediate.
    rewrite flat_map_map'. reflexivity.
  - intros k ty. simpl. apply free_variables_of_descr_names. reflexivity.
  - intros ty. simpl. apply free_variables_of_descr_names. reflexivity.
  - intros ty. simpl. apply free_variables_of_descr_names. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct non_null as [ | head | ], is_null as [ | x]; simpl;
      rewrite ?flat_map_app, ?Hnon_null, <- ?free_variables_of_option_names;
      reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    destruct extensions as [ | when_immediate when_block].
    + destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?flat_map_app, <- !free_variables_of_option_names,
          ?Himmediates, ?Hblocks;
        reflexivity.
    + destruct Hextensions as [Himmediate Hblock].
      destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?flat_map_app, <- !free_variables_of_option_names,
          ?Himmediates, ?Hblocks, Himmediate, Hblock;
        reflexivity.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    rewrite flat_map_app, flat_map_flat_map'. f_equal.
    + apply flat_map_ext_in'. intros [tag [case | ]] Hin; simpl;
        [ | reflexivity ].
      exact (free_variables_of_block_case_names case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; simpl; [ | reflexivity ].
      exact (free_variables_of_block_case_names case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl.
    rewrite flat_map_flat_map'. apply flat_map_ext_in'.
    intros [slot case] Hin. destruct (Hknown slot case Hin)
      as [Hentry Hextension].
    simpl. rewrite flat_map_app, Hentry, Hextension. reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. rewrite flat_map_app, !flat_map_flat_map'. f_equal;
      apply flat_map_ext_in'; intros [slot ty] Hin; simpl;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. simpl.
    rewrite flat_map_flat_map'. apply flat_map_ext_in'.
    intros [n ty] Hin. simpl. rewrite (Htypes n ty Hin). reflexivity.
Qed.

Theorem free_names_variables : free_names_variables_statement.
Proof.
  destruct free_names_variables_all
    as [Htype [_ [_ [_ [_ [_ Hextension]]]]]].
  split; assumption.
Qed.

(* The free variables of a renamed type, from its free names. *)
Lemma free_variables_of_name_renamed (r : renaming) (n : name) :
  free_variables_of_name (rename_name r n)
  = map (rename_variable r) (free_variables_of_name n).
Proof. destruct n; reflexivity. Qed.

Lemma flat_map_variables_renamed (r : renaming) (names : list name) :
  flat_map free_variables_of_name (map (rename_name r) names)
  = map (rename_variable r) (flat_map free_variables_of_name names).
Proof.
  rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
  intros n _. apply free_variables_of_name_renamed.
Qed.

Theorem free_variables_rename_type : free_variables_rename_type_statement.
Proof.
  split.
  - intros r t.
    rewrite !(proj1 free_names_variables), (proj1 free_names_rename_type).
    apply flat_map_variables_renamed.
  - intros r extension.
    rewrite !(proj2 free_names_variables), (proj2 free_names_rename_type).
    apply flat_map_variables_renamed.
Qed.

Theorem kind_rename_type : kind_rename_type_statement.
Proof. intros r [ty | ty | k ty | ty | ty]; reflexivity. Qed.

Lemma row_like_for_blocks_is_bottom_renamed (r : renaming)
    (blocks : row_like_for_blocks) :
  row_like_for_blocks_is_bottom (rename_row_like_for_blocks r blocks)
  = row_like_for_blocks_is_bottom blocks.
Proof.
  destruct blocks as [[ | binding known_tags] [other | ] alloc_mode];
    reflexivity.
Qed.

Lemma simple_must_be_const_renamed (r : renaming) (s : simple) :
  simple_must_be_const (rename_simple r s) = simple_must_be_const s.
Proof. destruct s; reflexivity. Qed.

Theorem must_be_singleton_rename_type :
  must_be_singleton_rename_type_statement.
Proof.
  intros r.
  pose (P_type :=
          fun t : flambda_type =>
            must_be_singleton (rename_type r t) = must_be_singleton t).
  pose (P_non_null :=
          fun head : head_of_kind_value_non_null =>
            match head with
            | Variant _ (Or_unknown.Known immediates) _ _ _ _ =>
              P_type immediates
            | _ => True
            end).
  pose (P_head :=
          fun head : head_of_kind_value =>
            match head with
            | Head_of_kind_value (Ok head) _ => P_non_null head
            | _ => True
            end).
  apply
    (types_induction P_type P_head P_non_null (fun _ => True) (fun _ => True)
       (fun _ => True) (fun _ => True));
    subst P_type P_head P_non_null; try (intros; exact I).
  - (* Type_value *)
    intros [ | [[[ | head | ] null] | s] | ] Hhead; simpl; try reflexivity;
      [ | rewrite simple_must_be_const_renamed; reflexivity ].
    destruct head
      as [is_int [immediates | ] get_tag [blocks | ] extensions is_unique
         | alloc_mode | k contents alloc_mode | closures alloc_mode | strs];
      destruct null as [ | null]; simpl; try reflexivity;
      rewrite row_like_for_blocks_is_bottom_renamed;
      destruct (row_like_for_blocks_is_bottom blocks); simpl;
      try reflexivity.
    simpl in Hhead. rewrite Hhead. reflexivity.
  - (* Type_naked_immediate *)
    intros [ | [head | s] | ]; simpl; try reflexivity.
    rewrite simple_must_be_const_renamed. reflexivity.
  - (* Type_naked_number *)
    intros k [ | [numbers | s] | ]; simpl; try reflexivity.
    rewrite simple_must_be_const_renamed. reflexivity.
  - (* Type_rec_info *)
    intros ty. reflexivity.
  - (* Type_region *)
    intros ty. reflexivity.
  - (* Head_of_kind_value *)
    intros [ | head | ] null Hhead; simpl; auto.
  - (* Variant *)
    intros is_int [immediates | ] get_tag blocks extensions is_unique
      Himmediates _ _; simpl; auto.
Qed.

(* ======================================================================== *)
(* The aliases of a renamed type                                            *)
(* ======================================================================== *)

Section Renamed_aliases.

Variable r : renaming.

Let rename_alias (alias : simple * flambda_kind) : simple * flambda_kind :=
  (rename_simple r (fst alias), snd alias).

Lemma aliases_of_relation_renamed (x : option variable) :
  aliases_of_relation (option_map (rename_variable r) x)
  = map rename_alias (aliases_of_relation x).
Proof. destruct x; reflexivity. Qed.

Lemma aliases_of_descr_renamed {head : Type} (k : flambda_kind)
    (aliases_of_head : head -> list (simple * flambda_kind))
    (rename_head : head -> head) (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) ->
    aliases_of_head (rename_head h) = map rename_alias (aliases_of_head h)) ->
  aliases_of_descr k aliases_of_head (rename_descr r rename_head ty)
  = map rename_alias (aliases_of_descr k aliases_of_head ty).
Proof.
  destruct ty as [ | [h | s] | ]; simpl; intros Hhead;
    [ reflexivity | exact (Hhead h eq_refl) | reflexivity | reflexivity ].
Qed.

Let type_renamed (t : flambda_type) : Prop :=
  aliases_with_kinds (rename_type r t)
  = map rename_alias (aliases_with_kinds t).

Let extension_renamed (extension : env_extension) : Prop :=
  aliases_of_env_extension (rename_env_extension r extension)
  = map rename_alias (aliases_of_env_extension extension).

Lemma aliases_of_block_case_renamed (case : row_like_block_case) :
  P_block_case type_renamed extension_renamed case ->
  aliases_of_block_case aliases_with_kinds aliases_of_env_extension
    (rename_block_case (rename_type r) (rename_env_extension r) case)
  = map rename_alias
      (aliases_of_block_case aliases_with_kinds aliases_of_env_extension
         case).
Proof.
  destruct case as [maps_to index extension].
  unfold rename_block_case, aliases_of_block_case. simpl.
  intros [Htypes Hextension]. simpl in Htypes, Hextension.
  rewrite map_app, flat_map_map', map_flat_map', Hextension. f_equal.
  apply flat_map_ext_in'. exact Htypes.
Qed.

Lemma aliases_renamed_all :
  (∀ t : flambda_type, type_renamed t)
  ∧ (∀ head : head_of_kind_value,
      aliases_of_head_of_kind_value (rename_head_of_kind_value r head)
      = map rename_alias (aliases_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      aliases_of_head_of_kind_value_non_null
        (rename_head_of_kind_value_non_null r head)
      = map rename_alias (aliases_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      aliases_of_row_like_for_blocks (rename_row_like_for_blocks r blocks)
      = map rename_alias (aliases_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      aliases_of_row_like_for_closures
        (rename_row_like_for_closures r closures)
      = map rename_alias (aliases_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      aliases_of_closures_entry (rename_closures_entry r entry)
      = map rename_alias (aliases_of_closures_entry entry))
  ∧ (∀ extension : env_extension, extension_renamed extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. unfold type_renamed. simpl.
    apply aliases_of_descr_renamed. intros h ->. exact Hhead.
  - intros ty. unfold type_renamed. simpl.
    apply aliases_of_descr_renamed. reflexivity.
  - intros k ty. unfold type_renamed. simpl.
    apply aliases_of_descr_renamed. reflexivity.
  - intros ty. unfold type_renamed. simpl.
    apply aliases_of_descr_renamed. reflexivity.
  - intros ty. unfold type_renamed. simpl.
    apply aliases_of_descr_renamed. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct non_null as [ | head | ], is_null as [ | x]; simpl;
      rewrite ?map_app, ?Hnon_null, ?aliases_of_relation_renamed;
      reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    destruct extensions as [ | when_immediate when_block].
    + destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, !aliases_of_relation_renamed, ?Himmediates,
          ?Hblocks;
        reflexivity.
    + destruct Hextensions as [Himmediate Hblock].
      destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, !aliases_of_relation_renamed, ?Himmediates,
          ?Hblocks, Himmediate, Hblock;
        reflexivity.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    unfold Keyed.map_values.
    rewrite map_app, flat_map_map', map_flat_map'. f_equal.
    + apply flat_map_ext_in'. intros [tag [case | ]] Hin; simpl;
        [ | reflexivity ].
      exact (aliases_of_block_case_renamed case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; simpl; [ | reflexivity ].
      exact (aliases_of_block_case_renamed case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. unfold map_values.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [slot [entry index extension]] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    simpl in Hentry, Hextension. simpl.
    rewrite map_app, Hentry, Hextension. reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. unfold map_values.
    rewrite !flat_map_map', map_app, !map_flat_map'. f_equal;
      apply flat_map_ext_in'; intros [slot ty] Hin; simpl;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. unfold extension_renamed. simpl.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [n ty] Hin. simpl. exact (Htypes n ty Hin).
Qed.

End Renamed_aliases.

Theorem aliases_with_kinds_rename_type :
  aliases_with_kinds_rename_type_statement.
Proof.
  split.
  - intros r t. exact (proj1 (aliases_renamed_all r) t).
  - intros r extension.
    destruct (aliases_renamed_all r) as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* The meaning of a renamed type                                            *)
(* ======================================================================== *)

(* One direction, for any two environments of which the second gives
   to a renamed variable the value that the first gives to the
   variable. The other direction is this one for the inverse. *)
Section Renamed_meaning.

Variable r : renaming.
Variable statics : static_env.
Variables vars vars' : env.

Hypothesis Hmoved :
  ∀ (x : variable) (w : value),
    lookup x vars = Some w -> lookup (rename_variable r x) vars' = Some w.

Lemma eval_simple_moved (s : simple) (v : value) :
  eval_simple vars s = Some v ->
  eval_simple vars' (rename_simple r s) = Some v.
Proof. destruct s as [x | sym | c]; simpl; auto. Qed.

Lemma value_of_name_moved (n : name) (v : value) :
  value_of_name vars n = Some v ->
  value_of_name vars' (rename_name r n) = Some v.
Proof. destruct n as [x | sym]; unfold value_of_name; simpl; auto. Qed.

Lemma variable_is_moved (x : option variable) (n : Z) :
  variable_is vars x n ->
  variable_is vars' (option_map (rename_variable r) x) n.
Proof. destruct x as [x | ]; simpl; auto. Qed.

Lemma descr_moved {head : Type} (k : flambda_kind)
    (head_satisfied head_satisfied' : head -> Prop)
    (rename_head : head -> head) (v : value) (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) -> head_satisfied h ->
    head_satisfied' (rename_head h)) ->
  descr_satisfied vars k head_satisfied v ty ->
  descr_satisfied vars' k head_satisfied' v (rename_descr r rename_head ty).
Proof.
  destruct ty as [ | [h | s] | ]; simpl; [ auto | | | auto ].
  - intros Hhead. exact (Hhead h eq_refl).
  - intros _ [Hs Hkind].
    split; [ exact (eval_simple_moved s v Hs) | exact Hkind ].
Qed.

Lemma fields_satisfied_moved (P Q : value -> flambda_type -> Prop)
    (f : flambda_type -> flambda_type) (tys : list flambda_type)
    (fields : list value) :
  (∀ ty : flambda_type, ty ∈ tys -> ∀ v : value, P v ty -> Q v (f ty)) ->
  fields_satisfied P tys fields -> fields_satisfied Q (map f tys) fields.
Proof.
  revert fields. induction tys as [ | ty tys IH]; intros fields Himpl.
  - auto.
  - destruct fields as [ | field fields]; simpl; [ auto | ].
    intros [Hfield Hfields]. split.
    + exact (Himpl ty (or_introl eq_refl) field Hfield).
    + apply IH; [ | exact Hfields ].
      intros ty' Hin. exact (Himpl ty' (or_intror Hin)).
Qed.

(* An equation of the renamed extension is one of the extension,
   renamed: no two names are renamed to one. *)
Lemma lookup_renamed_equations (n : name) (ty' : flambda_type)
    (equations : Keyed.finite_map name flambda_type) :
  Keyed.lookup name_eqb n
    (map
       (fun equation =>
          (rename_name r (fst equation), rename_type r (snd equation)))
       equations)
  = Some ty' ->
  ∃ (n0 : name) (ty : flambda_type),
    n = rename_name r n0
    ∧ ty' = rename_type r ty
    ∧ Keyed.lookup name_eqb n0 equations = Some ty.
Proof.
  induction equations as [ | [n1 ty1] equations IH]; simpl;
    [ discriminate | ].
  destruct (name_eqb n (rename_name r n1)) eqn:Hsame.
  - intros [= <-]. apply name_eqb_eq in Hsame. exists n1, ty1.
    split; [ exact Hsame | ]. split; [ reflexivity | ].
    rewrite (proj2 (name_eqb_eq n1 n1) eq_refl). reflexivity.
  - intros Hlookup. destruct (IH Hlookup) as [n0 [ty [Hn [Hty Hfound]]]].
    exists n0, ty. split; [ exact Hn | ]. split; [ exact Hty | ].
    destruct (name_eqb n0 n1) eqn:Hsame'; [ | exact Hfound ].
    apply name_eqb_eq in Hsame'. subst n0 n.
    rewrite (proj2 (name_eqb_eq _ _) eq_refl) in Hsame. discriminate Hsame.
Qed.

Lemma type_satisfied_moved_all :
  (∀ (t : flambda_type) (v : value),
    type_satisfied statics vars v t ->
    type_satisfied statics vars' v (rename_type r t))
  ∧ (∀ (head : head_of_kind_value) (v : value),
      head_of_kind_value_satisfied statics vars v head ->
      head_of_kind_value_satisfied statics vars' v
        (rename_head_of_kind_value r head))
  ∧ (∀ (head : head_of_kind_value_non_null) (v : value),
      head_of_kind_value_non_null_satisfied statics vars v head ->
      head_of_kind_value_non_null_satisfied statics vars' v
        (rename_head_of_kind_value_non_null r head))
  ∧ (∀ (blocks : row_like_for_blocks) (tag : Z) (fields : list value),
      row_like_for_blocks_satisfied statics vars tag fields blocks ->
      row_like_for_blocks_satisfied statics vars' tag fields
        (rename_row_like_for_blocks r blocks))
  ∧ (∀ (closures : row_like_for_closures) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      row_like_for_closures_satisfied statics vars v slot functions
        value_slots closures ->
      row_like_for_closures_satisfied statics vars' v slot functions
        value_slots (rename_row_like_for_closures r closures))
  ∧ (∀ (entry : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closures_entry_satisfied statics vars v slot functions value_slots
        entry ->
      closures_entry_satisfied statics vars' v slot functions value_slots
        (rename_closures_entry r entry))
  ∧ (∀ extension : env_extension,
      env_extension_satisfied statics vars extension ->
      env_extension_satisfied statics vars'
        (rename_env_extension r extension)).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead v. simpl rename_type. autorewrite with type_satisfied.
    apply descr_moved. intros h ->. exact (Hhead v).
  - (* Type_naked_immediate *)
    intros ty v. simpl rename_type. autorewrite with type_satisfied.
    apply descr_moved. intros head _.
    intros [i [Hv [Hnumbers Hrelations]]]. exists i.
    split; [ exact Hv | ]. split; [ exact Hnumbers | ].
    intros relation n Hin. simpl in Hin.
    apply in_map_iff in Hin as [[relation0 n0] [Hrenamed Hin]].
    simpl in Hrenamed. injection Hrenamed as <- <-.
    destruct (Hrelations relation0 n0 Hin) as [w [Hw H]].
    exists w. split; [ exact (value_of_name_moved n0 w Hw) | exact H ].
  - intros k ty v. simpl rename_type. autorewrite with type_satisfied.
    apply descr_moved. auto.
  - intros ty v. simpl rename_type. autorewrite with type_satisfied.
    apply descr_moved. auto.
  - intros ty v. simpl rename_type. autorewrite with type_satisfied.
    apply descr_moved. auto.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null v. simpl rename_head_of_kind_value.
    autorewrite with type_satisfied.
    intros [[Hnull His_null] | [Hnot_null [Hhead His_null]]].
    + left. split; [ exact Hnull | ].
      destruct is_null as [ | is_null]; [ exact His_null | ].
      exact (variable_is_moved _ _ His_null).
    + right. split; [ exact Hnot_null | ]. split.
      * destruct non_null as [ | head | ]; simpl; auto.
      * destruct is_null as [ | is_null]; [ exact His_null | ].
        exact (variable_is_moved _ _ His_null).
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions v.
    simpl rename_head_of_kind_value_non_null.
    autorewrite with type_satisfied.
    intros [[i [Hv [His_int [Himmediate Hextension]]]]
           | [tag [fields [Hblock [His_int [Hget_tag [Hrow Hextension]]]]]]].
    + left. exists i. split; [ exact Hv | ].
      split; [ exact (variable_is_moved _ _ His_int) | ]. split.
      * destruct immediates as [ty | ]; [ | exact I ].
        exact (Himmediates _ Himmediate).
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        exact (proj1 Hextensions Hextension).
    + right. exists tag, fields. split; [ exact Hblock | ].
      split; [ exact (variable_is_moved _ _ His_int) | ].
      split; [ exact (variable_is_moved _ _ Hget_tag) | ]. split.
      * destruct blocks as [blocks | ]; [ | exact I ].
        exact (Hblocks tag fields Hrow).
      * destruct extensions as [ | when_immediate when_block]; [ exact I | ].
        exact (proj2 Hextensions Hextension).
  - (* Mutable_block *)
    intros alloc_mode v. simpl rename_head_of_kind_value_non_null.
    autorewrite with type_satisfied. auto.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents v.
    simpl rename_head_of_kind_value_non_null.
    autorewrite with type_satisfied.
    intros [c [Hboxed [Hkind Hc]]]. exists c.
    split; [ exact Hboxed | ]. split; [ exact Hkind | ].
    exact (Hcontents _ Hc).
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures v.
    simpl rename_head_of_kind_value_non_null.
    autorewrite with type_satisfied.
    intros [slot [functions [value_slots [Hclosure Hrow]]]].
    exists slot, functions, value_slots. split; [ exact Hclosure | ].
    exact (Hclosures v slot functions value_slots Hrow).
  - (* String *)
    intros strs v. simpl rename_head_of_kind_value_non_null.
    autorewrite with type_satisfied. auto.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother tag fields.
    simpl rename_row_like_for_blocks.
    rewrite !row_like_for_blocks_find_iff, keyed_lookup_map_values.
    assert (Hcase :
      ∀ case : row_like_block_case,
        P_block_case
          (fun t =>
             ∀ v : value,
               type_satisfied statics vars v t ->
               type_satisfied statics vars' v (rename_type r t))
          (fun extension =>
             env_extension_satisfied statics vars extension ->
             env_extension_satisfied statics vars'
               (rename_env_extension r extension))
          case ->
        row_like_block_case_satisfied statics vars tag fields case ->
        row_like_block_case_satisfied statics vars' tag fields
          (rename_block_case (rename_type r) (rename_env_extension r) case)).
    { intros case [Htypes Hextension] [Hindex [Hfields Hext]].
      split; [ exact Hindex | ].
      split; [ | exact (Hextension Hext) ].
      exact (fields_satisfied_moved _ _ _ _ _ Htypes Hfields). }
    destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ] eqn:Hfind;
      simpl.
    + apply (keyed_lookup_in Z.eqb Z.eqb_eq) in Hfind.
      exact (Hcase case (Hknown tag case Hfind)).
    + auto.
    + destruct other_tags as [case | ]; simpl; [ | auto ].
      exact (Hcase case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown v slot functions value_slots.
    simpl rename_row_like_for_closures.
    rewrite !row_like_for_closures_lookup_iff.
    intros [case [Hlookup [Hindex [Hentry Hextension]]]].
    exists (rename_closures_case (rename_closures_entry r)
              (rename_env_extension r) case).
    rewrite lookup_map_values, Hlookup. split; [ reflexivity | ].
    apply lookup_in in Hlookup.
    destruct (Hknown slot case Hlookup) as [IHentry IHextension].
    split; [ exact Hindex | ].
    split; [ | exact (IHextension Hextension) ].
    exact (IHentry v slot functions value_slots Hentry).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      v slot functions value_slots.
    simpl rename_closures_entry.
    rewrite !closures_entry_lookup_iff.
    intros [Hfunctions [Hclosure_types Hvalue_slots]].
    split; [ exact Hfunctions | ]. split.
    + intros g ty' closure Hlookup Hproject.
      rewrite lookup_map_values in Hlookup.
      destruct (lookup g closure_types) as [ty | ] eqn:Hfound;
        [ | discriminate Hlookup ].
      injection Hlookup as <-.
      apply (Hclosures g ty (lookup_in _ _ _ Hfound)).
      exact (Hclosure_types g ty closure Hfound Hproject).
    + intros w ty' contents Hlookup Hcontents.
      rewrite lookup_map_values in Hlookup.
      destruct (lookup w value_slot_types) as [ty | ] eqn:Hfound;
        [ | discriminate Hlookup ].
      injection Hlookup as <-.
      apply (Hslots w ty (lookup_in _ _ _ Hfound)).
      exact (Hvalue_slots w ty contents Hfound Hcontents).
  - (* Env_extension *)
    intros equations Htypes. simpl rename_env_extension.
    rewrite !env_extension_lookup_iff.
    intros Hequations n ty' Hlookup.
    destruct (lookup_renamed_equations n ty' equations Hlookup)
      as [n0 [ty [-> [-> Hfound]]]].
    destruct (Hequations n0 ty Hfound) as [v [Hv Hty]]. exists v.
    split; [ exact (value_of_name_moved n0 v Hv) | ].
    exact (Htypes n0 ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hfound)
             v Hty).
Qed.

End Renamed_meaning.

(* The renamed environment gives to a renamed variable what the
   environment gives to the variable, and the environment gives to a
   variable renamed back what the renamed environment gives to it. *)
Lemma moved_by_rename_env (r : renaming) (vars : env) (x : variable)
    (w : value) :
  lookup x vars = Some w ->
  lookup (rename_variable r x) (rename_env r vars) = Some w.
Proof. rewrite lookup_rename_env. auto. Qed.

Lemma moved_back (r : renaming) (vars : env) (x : variable) (w : value) :
  lookup x (rename_env r vars) = Some w ->
  lookup (rename_variable (inverse_renaming r) x) vars = Some w.
Proof.
  rewrite <- (rename_variable_inverse_r r x) at 1.
  rewrite lookup_rename_env. auto.
Qed.

Theorem type_satisfied_rename : type_satisfied_rename_statement.
Proof.
  intros r statics vars v t. split.
  - intros Hrenamed. rewrite <- (proj1 rename_type_inverse r t).
    exact (proj1
             (type_satisfied_moved_all (inverse_renaming r) statics
                (rename_env r vars) vars (moved_back r vars))
             _ v Hrenamed).
  - exact (proj1
             (type_satisfied_moved_all r statics vars (rename_env r vars)
                (moved_by_rename_env r vars))
             t v).
Qed.

Theorem env_extension_satisfied_rename
  : env_extension_satisfied_rename_statement.
Proof.
  intros r statics vars extension. split.
  - intros Hrenamed. rewrite <- (proj2 rename_type_inverse r extension).
    destruct (type_satisfied_moved_all (inverse_renaming r) statics
                (rename_env r vars) vars (moved_back r vars))
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    exact (Hextension _ Hrenamed).
  - destruct (type_satisfied_moved_all r statics vars (rename_env r vars)
                (moved_by_rename_env r vars))
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* A renamed type is well formed when the type is                           *)
(* ======================================================================== *)

Lemma name_eqb_renamed (r : renaming) (n n' : name) :
  name_eqb (rename_name r n) (rename_name r n') = name_eqb n n'.
Proof.
  destruct n, n'; simpl; [ apply rename_variable_eqb | reflexivity .. ].
Qed.

Lemma among_renamed (r : renaming) (n : name) (names : list name) :
  Keyed.among name_eqb (rename_name r n) (map (rename_name r) names)
  = Keyed.among name_eqb n names.
Proof.
  unfold Keyed.among.
  induction names as [ | n' names IH]; simpl; [ reflexivity | ].
  rewrite name_eqb_renamed, IH. reflexivity.
Qed.

Lemma distinct_renamed (r : renaming) (names : list name) :
  Keyed.distinct name_eqb (map (rename_name r) names)
  = Keyed.distinct name_eqb names.
Proof.
  induction names as [ | n names IH]; simpl; [ reflexivity | ].
  rewrite among_renamed, IH. reflexivity.
Qed.

Lemma keys_map_values {K A B : Type} (f : A -> B)
    (m : Keyed.finite_map K A) :
  keys (Keyed.map_values f m) = keys m.
Proof. unfold keys, Keyed.map_values. rewrite map_map. reflexivity. Qed.

Lemma domain_map_values {A B : Type} (f : A -> B) (m : finite_map A) :
  domain (map_values f m) = domain m.
Proof. unfold domain, map_values. rewrite map_map. reflexivity. Qed.

Section Renamed_well_formed.

Variable r : renaming.

Lemma descr_well_formed_renamed {head : Type}
    (head_well_formed : head -> bool) (rename_head : head -> head)
    (ty : type_descr head) :
  (∀ h : head,
    ty = Ok (No_alias h) -> head_well_formed h = true ->
    head_well_formed (rename_head h) = true) ->
  descr_well_formed head_well_formed ty = true ->
  descr_well_formed head_well_formed (rename_descr r rename_head ty) = true.
Proof. destruct ty as [ | [h | s] | ]; simpl; auto. Qed.

Let type_kept (t : flambda_type) : Prop :=
  type_well_formed t = true -> type_well_formed (rename_type r t) = true.

Let extension_kept (extension : env_extension) : Prop :=
  env_extension_well_formed extension = true ->
  env_extension_well_formed (rename_env_extension r extension) = true.

Lemma block_case_well_formed_renamed (case : row_like_block_case) :
  P_block_case type_kept extension_kept case ->
  row_like_block_case_well_formed case = true ->
  row_like_block_case_well_formed
    (rename_block_case (rename_type r) (rename_env_extension r) case)
  = true.
Proof.
  intros [Htypes Hextension].
  rewrite !(row_like_block_case_well_formed_iff _).
  unfold rename_block_case. simpl.
  intros [Hfits [Hshape [Hwf Hext]]].
  split; [ | split; [ exact Hshape | split ] ].
  - unfold block_product_fits in *. simpl. rewrite length_map. exact Hfits.
  - intros ty' Hin. apply in_map_iff in Hin as [ty [<- Hin]].
    exact (Htypes ty Hin (Hwf ty Hin)).
  - exact (Hextension Hext).
Qed.

Lemma type_well_formed_renamed_all :
  (∀ t : flambda_type, type_kept t)
  ∧ (∀ head : head_of_kind_value,
      head_of_kind_value_well_formed head = true ->
      head_of_kind_value_well_formed (rename_head_of_kind_value r head)
      = true)
  ∧ (∀ head : head_of_kind_value_non_null,
      head_of_kind_value_non_null_well_formed head = true ->
      head_of_kind_value_non_null_well_formed
        (rename_head_of_kind_value_non_null r head)
      = true)
  ∧ (∀ blocks : row_like_for_blocks,
      row_like_for_blocks_well_formed blocks = true ->
      row_like_for_blocks_well_formed (rename_row_like_for_blocks r blocks)
      = true)
  ∧ (∀ closures : row_like_for_closures,
      row_like_for_closures_well_formed closures = true ->
      row_like_for_closures_well_formed
        (rename_row_like_for_closures r closures)
      = true)
  ∧ (∀ (entry : closures_entry)
       (index : row_like_index set_of_closures_contents unit),
      closures_entry_well_formed index entry = true ->
      closures_entry_well_formed index (rename_closures_entry r entry)
      = true)
  ∧ (∀ extension : env_extension, extension_kept extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. unfold type_kept. simpl.
    apply descr_well_formed_renamed. intros h ->. exact Hhead.
  - (* Type_naked_immediate *)
    intros ty. unfold type_kept. simpl.
    apply descr_well_formed_renamed. intros [numbers relations] _ Hhead.
    exact Hhead.
  - intros k ty. unfold type_kept. simpl. rewrite !andb_true_iff.
    intros [Hk Hty]. split; [ exact Hk | ]. revert Hty.
    apply descr_well_formed_renamed. auto.
  - intros ty. unfold type_kept. simpl. auto.
  - intros ty. unfold type_kept. simpl. auto.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null. simpl.
    destruct non_null as [ | head | ]; simpl; auto.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions. simpl.
    rewrite !andb_true_iff. intros [[Himmediate Hrow] Hextension].
    split; [ split | ].
    + destruct immediates as [ty | ]; simpl; [ | reflexivity ].
      apply andb_true_iff in Himmediate as [Hkind Hty].
      rewrite kind_rename_type, Hkind. exact (Himmediates Hty).
    + destruct blocks as [row | ]; simpl; [ | reflexivity ].
      exact (Hblocks Hrow).
    + destruct extensions as [ | when_immediate when_block]; simpl;
        [ reflexivity | ].
      apply andb_true_iff in Hextension as [Himmediate_ext Hblock_ext].
      rewrite (proj1 Hextensions Himmediate_ext),
        (proj2 Hextensions Hblock_ext).
      reflexivity.
  - (* Mutable_block *)
    intros alloc_mode. simpl. auto.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. simpl.
    rewrite !andb_true_iff, kind_rename_type. intros [Hkind Hty].
    split; [ exact Hkind | exact (Hcontents Hty) ].
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    intros strs. simpl. auto.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother.
    simpl rename_row_like_for_blocks.
    rewrite !(row_like_for_blocks_well_formed_iff _ _ _).
    intros [Hdistinct [Hcases Hother_wf]].
    split; [ rewrite keys_map_values; exact Hdistinct | ]. split.
    + intros tag case' Hlookup. rewrite keyed_lookup_map_values in Hlookup.
      destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ]
        eqn:Hfound;
        simpl in Hlookup; try discriminate Hlookup.
      injection Hlookup as <-.
      apply block_case_well_formed_renamed.
      * exact (Hknown tag case (keyed_lookup_in Z.eqb Z.eqb_eq _ _ _ Hfound)).
      * exact (Hcases tag case Hfound).
    + destruct other_tags as [case | ]; simpl; [ | exact I ].
      exact (block_case_well_formed_renamed case Hother Hother_wf).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl rename_row_like_for_closures.
    rewrite !(row_like_for_closures_well_formed_iff _).
    intros [Hdistinct Hcases].
    split; [ rewrite (keys_map_values _ known_closures); exact Hdistinct | ].
    intros slot case' Hlookup. rewrite lookup_map_values in Hlookup.
    destruct (lookup slot known_closures) as [case | ] eqn:Hfound;
      [ | discriminate Hlookup ].
    injection Hlookup as <-.
    destruct (Hknown slot case (lookup_in _ _ _ Hfound))
      as [Hentry Hextension].
    pose proof (Hcases slot case Hfound) as Hcase.
    unfold row_like_closures_case_well_formed in *. simpl.
    rewrite !andb_true_iff in *. destruct Hcase as [[Hslot Hentry_wf] Hext].
    split; [ split; [ exact Hslot | exact (Hentry _ Hentry_wf) ] | ].
    exact (Hextension Hext).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      index.
    simpl rename_closures_entry.
    rewrite !(closures_entry_well_formed_iff _ _ _ _).
    intros [Hfit [[Hfunctions [Hclosure_keys Hslot_keys]]
                    [Hclosure_wf Hslot_wf]]].
    split; [ rewrite !domain_map_values; exact Hfit | ].
    split.
    { split; [ exact Hfunctions | ].
      rewrite (keys_map_values _ closure_types),
        (keys_map_values _ value_slot_types).
      split; assumption. }
    split.
    + intros g ty' Hlookup. rewrite lookup_map_values in Hlookup.
      destruct (lookup g closure_types) as [ty | ] eqn:Hfound;
        [ | discriminate Hlookup ].
      injection Hlookup as <-.
      destruct (Hclosure_wf g ty Hfound) as [Hkind Hty].
      split; [ rewrite kind_rename_type; exact Hkind | ].
      exact (Hclosures g ty (lookup_in _ _ _ Hfound) Hty).
    + intros w ty' Hlookup. rewrite lookup_map_values in Hlookup.
      destruct (lookup w value_slot_types) as [ty | ] eqn:Hfound;
        [ | discriminate Hlookup ].
      injection Hlookup as <-.
      exact (Hslots w ty (lookup_in _ _ _ Hfound) (Hslot_wf w ty Hfound)).
  - (* Env_extension *)
    intros equations Htypes. unfold extension_kept.
    simpl rename_env_extension.
    rewrite !(env_extension_well_formed_iff _).
    intros [Hdistinct Hwf]. split.
    + unfold keys. rewrite map_map. simpl.
      rewrite <- (map_map fst (rename_name r)), distinct_renamed.
      exact Hdistinct.
    + intros n ty' Hlookup.
      destruct (lookup_renamed_equations r n ty' equations Hlookup)
        as [n0 [ty [-> [-> Hfound]]]].
      exact (Htypes n0 ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hfound)
               (Hwf n0 ty Hfound)).
Qed.

End Renamed_well_formed.

Theorem type_well_formed_rename : type_well_formed_rename_statement.
Proof.
  split.
  - intros r t. apply eq_true_iff_eq. split.
    + intros Hrenamed. rewrite <- (proj1 rename_type_inverse r t).
      exact (proj1 (type_well_formed_renamed_all (inverse_renaming r)) _
               Hrenamed).
    + exact (proj1 (type_well_formed_renamed_all r) t).
  - intros r extension. apply eq_true_iff_eq. split.
    + intros Hrenamed. rewrite <- (proj2 rename_type_inverse r extension).
      destruct (type_well_formed_renamed_all (inverse_renaming r))
        as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
      exact (Hextension _ Hrenamed).
    + destruct (type_well_formed_renamed_all r)
        as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
      exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* Two renamed types are the same type when the types are                   *)
(* ======================================================================== *)

Lemma rename_name_inverse_r (r : renaming) (n : name) :
  rename_name r (rename_name (inverse_renaming r) n) = n.
Proof.
  destruct n as [x | sym]; simpl; [ | reflexivity ].
  rewrite rename_variable_inverse_r. reflexivity.
Qed.

Lemma simple_eqb_renamed (r : renaming) (s s' : simple) :
  simple_eqb (rename_simple r s) (rename_simple r s') = simple_eqb s s'.
Proof.
  destruct s, s'; simpl; try reflexivity. apply rename_variable_eqb.
Qed.

Lemma option_variable_eqb_renamed (r : renaming) (x x' : option variable) :
  option_variable_eqb (option_map (rename_variable r) x)
    (option_map (rename_variable r) x')
  = option_variable_eqb x x'.
Proof.
  destruct x, x'; simpl; try reflexivity. apply rename_variable_eqb.
Qed.

Lemma is_null_eqb_renamed (r : renaming) (null null' : is_null) :
  is_null_eqb (rename_is_null r null) (rename_is_null r null')
  = is_null_eqb null null'.
Proof.
  destruct null, null'; simpl; try reflexivity.
  apply option_variable_eqb_renamed.
Qed.

(* Sets of keys, under a function that keeps the keys apart. *)
Section Keys_mapped.

Context {K : Type} (eqb : K -> K -> bool) (f : K -> K).

Hypothesis Hf : ∀ a b : K, eqb (f a) (f b) = eqb a b.

Lemma keyed_among_map_keys (key : K) (keys : list K) :
  Keyed.among eqb (f key) (map f keys) = Keyed.among eqb key keys.
Proof.
  unfold Keyed.among.
  induction keys as [ | key' keys IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH. reflexivity.
Qed.

Lemma keyed_all_among_map (keys allowed : list K) :
  Keyed.all_among eqb (map f keys) (map f allowed)
  = Keyed.all_among eqb keys allowed.
Proof.
  unfold Keyed.all_among.
  induction keys as [ | key keys IH]; simpl; [ reflexivity | ].
  rewrite keyed_among_map_keys, IH. reflexivity.
Qed.

Lemma keyed_same_keys_b_map (keys keys' : list K) :
  Keyed.same_keys_b eqb (map f keys) (map f keys')
  = Keyed.same_keys_b eqb keys keys'.
Proof.
  unfold Keyed.same_keys_b. rewrite !keyed_all_among_map. reflexivity.
Qed.

End Keys_mapped.

Lemma lists_eqb_map {A : Type} (eqb : A -> A -> bool) (f : A -> A)
    (l l' : list A) :
  (∀ a a' : A, a ∈ l -> eqb a a' = true -> eqb (f a) (f a') = true) ->
  lists_eqb eqb l l' = true -> lists_eqb eqb (map f l) (map f l') = true.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | a' l'] Hf; simpl;
    try discriminate; [ reflexivity | ].
  rewrite !andb_true_iff. intros [Ha Hl]. split.
  - exact (Hf a a' (or_introl eq_refl) Ha).
  - apply IH; [ | exact Hl ]. intros b b' Hin. exact (Hf b b' (or_intror Hin)).
Qed.

Lemma map_eqb_map_values {K A : Type} (key_eqb : K -> K -> bool)
    (Hkey : decides_equality key_eqb) (eqb : A -> A -> bool) (f : A -> A)
    (m m' : Keyed.finite_map K A) :
  (∀ (key : K) (a a' : A),
    (key, a) ∈ m -> eqb a a' = true -> eqb (f a) (f a') = true) ->
  map_eqb key_eqb eqb m m' = true ->
  map_eqb key_eqb eqb (Keyed.map_values f m) (Keyed.map_values f m') = true.
Proof.
  intros Hf. rewrite !(map_eqb_iff K key_eqb Hkey). intros Hsame key.
  rewrite !keyed_lookup_map_values. specialize (Hsame key).
  destruct (Keyed.lookup key_eqb key m) as [a | ] eqn:Hfound,
    (Keyed.lookup key_eqb key m') as [a' | ];
    simpl; auto.
  apply (Hf key a a'); [ | exact Hsame ].
  exact (keyed_lookup_in key_eqb Hkey _ _ _ Hfound).
Qed.

Section Renamed_equal.

Variable r : renaming.

(* The equation of the renamed extension on a name is that of the
   extension on the name renamed back. *)
Lemma lookup_renamed_equations_eq (n : name)
    (equations : Keyed.finite_map name flambda_type) :
  Keyed.lookup name_eqb n
    (map
       (fun equation =>
          (rename_name r (fst equation), rename_type r (snd equation)))
       equations)
  = option_map (rename_type r)
      (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n) equations).
Proof.
  induction equations as [ | [n1 ty1] equations IH]; simpl; [ reflexivity | ].
  replace (name_eqb n (rename_name r n1))
    with (name_eqb (rename_name (inverse_renaming r) n) n1).
  - destruct (name_eqb (rename_name (inverse_renaming r) n) n1);
      [ reflexivity | exact IH ].
  - rewrite <- (name_eqb_renamed r), rename_name_inverse_r. reflexivity.
Qed.

Lemma descr_eqb_renamed {head : Type} (head_eqb : head -> head -> bool)
    (rename_head : head -> head) (ty ty' : type_descr head) :
  (∀ h h' : head,
    ty = Ok (No_alias h) -> head_eqb h h' = true ->
    head_eqb (rename_head h) (rename_head h') = true) ->
  descr_eqb head_eqb ty ty' = true ->
  descr_eqb head_eqb (rename_descr r rename_head ty)
    (rename_descr r rename_head ty')
  = true.
Proof.
  destruct ty as [ | [h | s] | ], ty' as [ | [h' | s'] | ]; simpl;
    try discriminate; auto.
  intros _. rewrite simple_eqb_renamed. auto.
Qed.

Let type_kept (t : flambda_type) : Prop :=
  ∀ t' : flambda_type,
    type_eqb t t' = true ->
    type_eqb (rename_type r t) (rename_type r t') = true.

Let extension_kept (extension : env_extension) : Prop :=
  ∀ extension' : env_extension,
    env_extension_eqb extension extension' = true ->
    env_extension_eqb (rename_env_extension r extension)
      (rename_env_extension r extension')
    = true.

Lemma block_case_eqb_renamed (case case' : row_like_block_case) :
  P_block_case type_kept extension_kept case ->
  block_case_eqb type_eqb env_extension_eqb case case' = true ->
  block_case_eqb type_eqb env_extension_eqb
    (rename_block_case (rename_type r) (rename_env_extension r) case)
    (rename_block_case (rename_type r) (rename_env_extension r) case')
  = true.
Proof.
  intros [Htypes Hextension]. unfold block_case_eqb, rename_block_case.
  simpl. rewrite !andb_true_iff. intros [[Hindex Hfields] Hext].
  split; [ split; [ exact Hindex | ] | exact (Hextension _ Hext) ].
  apply lists_eqb_map; [ | exact Hfields ].
  intros ty ty' Hin. exact (Htypes ty Hin ty').
Qed.

Lemma type_eqb_renamed_all :
  (∀ t : flambda_type, type_kept t)
  ∧ (∀ head head' : head_of_kind_value,
      head_of_kind_value_eqb head head' = true ->
      head_of_kind_value_eqb (rename_head_of_kind_value r head)
        (rename_head_of_kind_value r head')
      = true)
  ∧ (∀ head head' : head_of_kind_value_non_null,
      head_of_kind_value_non_null_eqb head head' = true ->
      head_of_kind_value_non_null_eqb
        (rename_head_of_kind_value_non_null r head)
        (rename_head_of_kind_value_non_null r head')
      = true)
  ∧ (∀ blocks blocks' : row_like_for_blocks,
      row_like_for_blocks_eqb blocks blocks' = true ->
      row_like_for_blocks_eqb (rename_row_like_for_blocks r blocks)
        (rename_row_like_for_blocks r blocks')
      = true)
  ∧ (∀ closures closures' : row_like_for_closures,
      row_like_for_closures_eqb closures closures' = true ->
      row_like_for_closures_eqb (rename_row_like_for_closures r closures)
        (rename_row_like_for_closures r closures')
      = true)
  ∧ (∀ entry entry' : closures_entry,
      closures_entry_eqb entry entry' = true ->
      closures_entry_eqb (rename_closures_entry r entry)
        (rename_closures_entry r entry')
      = true)
  ∧ (∀ extension : env_extension, extension_kept extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead [ty' | ty' | k' ty' | ty' | ty']; simpl;
      try discriminate.
    apply descr_eqb_renamed. intros h h' ->. exact (Hhead h').
  - (* Type_naked_immediate *)
    intros ty [ty' | ty' | k' ty' | ty' | ty']; simpl; try discriminate.
    apply descr_eqb_renamed.
    intros [numbers relations] [numbers' relations'] _.
    unfold head_of_kind_naked_immediate_eqb. simpl.
    rewrite !andb_true_iff. intros [Hnumbers Hrelations].
    split; [ exact Hnumbers | ].
    rewrite keyed_same_keys_b_map; [ exact Hrelations | ].
    intros [relation n] [relation' n']. simpl.
    rewrite name_eqb_renamed. reflexivity.
  - (* Type_naked_number *)
    intros k ty [ty' | ty' | k' ty' | ty' | ty']; simpl; try discriminate.
    rewrite !andb_true_iff. intros [Hk Hty]. split; [ exact Hk | ].
    revert Hty. apply descr_eqb_renamed. auto.
  - intros ty [ty' | ty' | k' ty' | ty' | ty']; simpl; try discriminate.
    apply descr_eqb_renamed. auto.
  - intros ty [ty' | ty' | k' ty' | ty' | ty']; simpl; try discriminate.
    apply descr_eqb_renamed. auto.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null [non_null' is_null']. simpl.
    rewrite !andb_true_iff, is_null_eqb_renamed. intros [Hhead Hnull].
    split; [ | exact Hnull ].
    destruct non_null as [ | head | ], non_null' as [ | head' | ];
      simpl in *; try discriminate; auto.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions head'.
    destruct head'
      as [is_int' immediates' get_tag' blocks' extensions' is_unique'
         | alloc_mode' | k' contents' alloc_mode'
         | by_function_slot' alloc_mode' | strs'];
      simpl; try discriminate.
    rewrite !andb_true_iff, !option_variable_eqb_renamed.
    intros [[[[[His_int Himmediate] Hget_tag] Hrow] Hextension] Hunique].
    repeat split; try assumption.
    + destruct immediates as [ty | ], immediates' as [ty' | ]; simpl in *;
        try discriminate; auto.
    + destruct blocks as [row | ], blocks' as [row' | ]; simpl in *;
        try discriminate; auto.
    + destruct extensions as [ | when_immediate when_block],
        extensions' as [ | when_immediate' when_block'];
        simpl in *; try discriminate; auto.
      apply andb_true_iff in Hextension as [Himmediate_ext Hblock_ext].
      rewrite (proj1 Hextensions _ Himmediate_ext),
        (proj2 Hextensions _ Hblock_ext).
      reflexivity.
  - (* Mutable_block *)
    intros alloc_mode head'. destruct head'; simpl; try discriminate; auto.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents head'.
    destruct head'; simpl; try discriminate.
    rewrite !andb_true_iff. intros [[Hk Hty] Hmode].
    split; [ split; [ exact Hk | exact (Hcontents _ Hty) ] | exact Hmode ].
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures head'.
    destruct head'; simpl; try discriminate.
    rewrite !andb_true_iff. intros [Hrow Hmode].
    split; [ exact (Hclosures _ Hrow) | exact Hmode ].
  - (* String *)
    intros strs head'. destruct head'; simpl; try discriminate; auto.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother
      [known_tags' other_tags' alloc_mode'].
    simpl. rewrite !andb_true_iff. intros [[Hmaps Hothers] Hmode].
    split; [ split | exact Hmode ].
    + apply (map_eqb_map_values Z.eqb Z.eqb_eq); [ | exact Hmaps ].
      intros tag [case | ] [case' | ] Hin; simpl; try discriminate; auto.
      exact (block_case_eqb_renamed case case' (Hknown tag case Hin)).
    + destruct other_tags as [case | ], other_tags' as [case' | ];
        simpl in *; try discriminate; auto.
      exact (block_case_eqb_renamed case case' Hother Hothers).
  - (* Row_like_for_closures *)
    intros known_closures Hknown [known_closures']. simpl.
    apply (map_eqb_map_values Pos.eqb Pos.eqb_eq).
    intros slot case case' Hin.
    destruct (Hknown slot case Hin) as [Hentry Hextension].
    unfold closures_case_eqb, rename_closures_case. simpl.
    rewrite !andb_true_iff. intros [[Hindex Hsame] Hext].
    split; [ split; [ exact Hindex | exact (Hentry _ Hsame) ] | ].
    exact (Hextension _ Hext).
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      [function_types' closure_types' value_slot_types'].
    simpl. rewrite !andb_true_iff. intros [[Hfunctions Hclosure] Hslot].
    split; [ split; [ exact Hfunctions | ] | ].
    + apply (map_eqb_map_values Pos.eqb Pos.eqb_eq); [ | exact Hclosure ].
      intros g ty ty' Hin. exact (Hclosures g ty Hin ty').
    + apply (map_eqb_map_values Pos.eqb Pos.eqb_eq); [ | exact Hslot ].
      intros w ty ty' Hin. exact (Hslots w ty Hin ty').
  - (* Env_extension *)
    intros equations Htypes [equations']. simpl.
    rewrite !(map_eqb_iff name name_eqb name_eqb_eq). intros Hsame n.
    rewrite !lookup_renamed_equations_eq.
    specialize (Hsame (rename_name (inverse_renaming r) n)).
    destruct
      (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n) equations)
      as [ty | ] eqn:Hfound,
      (Keyed.lookup name_eqb (rename_name (inverse_renaming r) n) equations')
      as [ty' | ];
      simpl; auto.
    exact (Htypes _ ty (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hfound)
             ty' Hsame).
Qed.

End Renamed_equal.

Theorem type_eqb_rename : type_eqb_rename_statement.
Proof.
  split.
  - intros r t t'. apply eq_true_iff_eq. split.
    + intros Hrenamed.
      rewrite <- (proj1 rename_type_inverse r t),
        <- (proj1 rename_type_inverse r t').
      exact (proj1 (type_eqb_renamed_all (inverse_renaming r)) _ _ Hrenamed).
    + exact (proj1 (type_eqb_renamed_all r) t t').
  - intros r extension extension'. apply eq_true_iff_eq. split.
    + intros Hrenamed.
      rewrite <- (proj2 rename_type_inverse r extension),
        <- (proj2 rename_type_inverse r extension').
      destruct (type_eqb_renamed_all (inverse_renaming r))
        as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
      exact (Hextension _ _ Hrenamed).
    + destruct (type_eqb_renamed_all r)
        as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
      exact (Hextension extension extension').
Qed.

(* ======================================================================== *)
(* The names of the equations of a renamed type                             *)
(* ======================================================================== *)

Section Renamed_equation_names.

Variable r : renaming.

Let rename_equation (equation : name * flambda_kind) : name * flambda_kind :=
  (rename_name r (fst equation), snd equation).

Let type_renamed (t : flambda_type) : Prop :=
  equation_names_with_kinds (rename_type r t)
  = map rename_equation (equation_names_with_kinds t).

Let extension_renamed (extension : env_extension) : Prop :=
  equation_names_of_env_extension (rename_env_extension r extension)
  = map rename_equation (equation_names_of_env_extension extension).

Lemma equation_names_of_block_case_renamed (case : row_like_block_case) :
  P_block_case type_renamed extension_renamed case ->
  equation_names_of_block_case equation_names_with_kinds
    equation_names_of_env_extension
    (rename_block_case (rename_type r) (rename_env_extension r) case)
  = map rename_equation
      (equation_names_of_block_case equation_names_with_kinds
         equation_names_of_env_extension case).
Proof.
  destruct case as [maps_to index extension].
  unfold rename_block_case, equation_names_of_block_case. simpl.
  intros [Htypes Hextension]. simpl in Htypes, Hextension.
  rewrite map_app, flat_map_map', map_flat_map', Hextension. f_equal.
  apply flat_map_ext_in'. exact Htypes.
Qed.

Lemma equation_names_renamed_all :
  (∀ t : flambda_type, type_renamed t)
  ∧ (∀ head : head_of_kind_value,
      equation_names_of_head_of_kind_value
        (rename_head_of_kind_value r head)
      = map rename_equation (equation_names_of_head_of_kind_value head))
  ∧ (∀ head : head_of_kind_value_non_null,
      equation_names_of_head_of_kind_value_non_null
        (rename_head_of_kind_value_non_null r head)
      = map rename_equation
          (equation_names_of_head_of_kind_value_non_null head))
  ∧ (∀ blocks : row_like_for_blocks,
      equation_names_of_row_like_for_blocks
        (rename_row_like_for_blocks r blocks)
      = map rename_equation (equation_names_of_row_like_for_blocks blocks))
  ∧ (∀ closures : row_like_for_closures,
      equation_names_of_row_like_for_closures
        (rename_row_like_for_closures r closures)
      = map rename_equation
          (equation_names_of_row_like_for_closures closures))
  ∧ (∀ entry : closures_entry,
      equation_names_of_closures_entry (rename_closures_entry r entry)
      = map rename_equation (equation_names_of_closures_entry entry))
  ∧ (∀ extension : env_extension, extension_renamed extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead. unfold type_renamed.
    destruct ty as [ | [head | s] | ]; simpl; try reflexivity.
    exact Hhead.
  - intros ty. unfold type_renamed. reflexivity.
  - intros k ty. unfold type_renamed. reflexivity.
  - intros ty. unfold type_renamed. reflexivity.
  - intros ty. unfold type_renamed. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null.
    destruct non_null as [ | head | ]; simpl; try reflexivity.
    exact Hnon_null.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    destruct extensions as [ | when_immediate when_block].
    + destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, ?Himmediates, ?Hblocks;
        reflexivity.
    + destruct Hextensions as [Himmediate Hblock].
      destruct immediates as [ty | ], blocks as [row | ]; simpl;
        rewrite ?map_app, ?Himmediates, ?Hblocks, Himmediate, Hblock;
        reflexivity.
  - (* Mutable_block *)
    reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents. exact Hcontents.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures. exact Hclosures.
  - (* String *)
    reflexivity.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother. simpl.
    unfold Keyed.map_values.
    rewrite map_app, flat_map_map', map_flat_map'. f_equal.
    + apply flat_map_ext_in'. intros [tag [case | ]] Hin; simpl;
        [ | reflexivity ].
      exact (equation_names_of_block_case_renamed case (Hknown tag case Hin)).
    + destruct other_tags as [case | ]; simpl; [ | reflexivity ].
      exact (equation_names_of_block_case_renamed case Hother).
  - (* Row_like_for_closures *)
    intros known_closures Hknown. simpl. unfold map_values.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [slot [entry index extension]] Hin.
    destruct (Hknown slot _ Hin) as [Hentry Hextension].
    simpl in Hentry, Hextension. simpl.
    rewrite map_app, Hentry, Hextension. reflexivity.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    simpl. unfold map_values.
    rewrite !flat_map_map', map_app, !map_flat_map'. f_equal;
      apply flat_map_ext_in'; intros [slot ty] Hin; simpl;
      [ exact (Hclosures slot ty Hin) | exact (Hslots slot ty Hin) ].
  - (* Env_extension *)
    intros equations Htypes. unfold extension_renamed. simpl.
    rewrite flat_map_map', map_flat_map'. apply flat_map_ext_in'.
    intros [n ty] Hin. simpl.
    rewrite (Htypes n ty Hin), kind_rename_type. reflexivity.
Qed.

End Renamed_equation_names.

Theorem equation_names_rename_type : equation_names_rename_type_statement.
Proof.
  split.
  - intros r t. exact (proj1 (equation_names_renamed_all r) t).
  - intros r extension.
    destruct (equation_names_renamed_all r)
      as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
    exact (Hextension extension).
Qed.

(* ======================================================================== *)
(* The creation functions and the renaming                                  *)
(* ======================================================================== *)

Lemma rename_type_unknown (r : renaming) (k : flambda_kind) :
  rename_type r (unknown k) = unknown k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma rename_type_bottom (r : renaming) (k : flambda_kind) :
  rename_type r (bottom k) = bottom k.
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma rename_type_alias_type_of (r : renaming) (k : flambda_kind)
    (s : simple) :
  rename_type r (alias_type_of k s) = alias_type_of k (rename_simple r s).
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Lemma rename_type_this_const (r : renaming) (c : const) :
  rename_type r (this_const c) = this_const c.
Proof. unfold this_const. rewrite rename_type_alias_type_of. reflexivity. Qed.

Lemma rename_type_type_for_const (r : renaming) (c : const) :
  rename_type r (type_for_const c) = type_for_const c.
Proof.
  destruct c; try apply rename_type_this_const; simpl;
    [ reflexivity | apply rename_type_unknown ].
Qed.

Lemma rename_type_these_naked_immediates (r : renaming) (imms : list Z) :
  rename_type r (these_naked_immediates imms) = these_naked_immediates imms.
Proof. destruct imms as [ | i [ | i' imms]]; reflexivity. Qed.

Lemma rename_type_these_tagged_immediates (r : renaming) (imms : list Z) :
  rename_type r (these_tagged_immediates imms) = these_tagged_immediates imms.
Proof. destruct imms as [ | i [ | i' imms]]; reflexivity. Qed.

Lemma rename_type_blocks_with_these_tags (r : renaming) (tags : list Z)
    (alloc_mode : Alloc_mode.t) :
  rename_type r (blocks_with_these_tags tags alloc_mode)
  = blocks_with_these_tags tags alloc_mode.
Proof.
  unfold blocks_with_these_tags. simpl. unfold create_variant, non_null_value.
  do 7 f_equal. unfold Keyed.map_values. rewrite map_map. reflexivity.
Qed.

Theorem rename_type_closed : rename_type_closed_statement.
Proof.
  intros r.
  repeat split; intros;
    auto using rename_type_unknown, rename_type_bottom,
      rename_type_this_const, rename_type_type_for_const,
      rename_type_these_naked_immediates,
      rename_type_these_tagged_immediates,
      rename_type_blocks_with_these_tags.
Qed.

Lemma get_alias_opt_renamed (r : renaming) (t : flambda_type) :
  get_alias_opt (rename_type r t)
  = option_map (rename_simple r) (get_alias_opt t).
Proof.
  destruct t as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; reflexivity.
Qed.

Theorem rename_type_alias : rename_type_alias_statement.
Proof.
  intros r.
  split;
    [ exact (rename_type_alias_type_of r) | exact (get_alias_opt_renamed r) ].
Qed.

Lemma is_obviously_bottom_renamed (r : renaming) (t : flambda_type) :
  is_obviously_bottom (rename_type r t) = is_obviously_bottom t.
Proof.
  destruct t as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; reflexivity.
Qed.

Lemma is_obviously_unknown_renamed (r : renaming) (t : flambda_type) :
  is_obviously_unknown (rename_type r t) = is_obviously_unknown t.
Proof.
  destruct t as [ty | ty | k ty | ty | ty];
    destruct ty as [ | [head | s] | ]; reflexivity.
Qed.

Theorem is_obviously_rename_type : is_obviously_rename_type_statement.
Proof.
  intros r.
  split; [ exact (is_obviously_bottom_renamed r) | ].
  split; [ exact (is_obviously_unknown_renamed r) | ].
  exact (row_like_for_blocks_is_bottom_renamed r).
Qed.

Lemma rename_type_non_null_value (r : renaming)
    (head : head_of_kind_value_non_null) :
  rename_type r (non_null_value head)
  = non_null_value (rename_head_of_kind_value_non_null r head).
Proof. reflexivity. Qed.

Lemma rename_type_create_variant (r : renaming) (is_unique : bool)
    (immediates : Or_unknown.t flambda_type)
    (blocks : Or_unknown.t row_like_for_blocks)
    (extensions : variant_extensions) :
  rename_type r (create_variant is_unique immediates blocks extensions)
  = create_variant is_unique (or_unknown_map (rename_type r) immediates)
      (or_unknown_map (rename_row_like_for_blocks r) blocks)
      (rename_variant_extensions r extensions).
Proof. reflexivity. Qed.

Lemma rename_type_create_closures (r : renaming) (alloc_mode : Alloc_mode.t)
    (closures : row_like_for_closures) :
  rename_type r (create_closures alloc_mode closures)
  = create_closures alloc_mode (rename_row_like_for_closures r closures).
Proof. reflexivity. Qed.

Lemma rename_type_box_number (r : renaming) (k : boxable_number)
    (contents : flambda_type) (alloc_mode : Alloc_mode.t) :
  rename_type r (box_number k contents alloc_mode)
  = box_number k (rename_type r contents) alloc_mode.
Proof. reflexivity. Qed.

Lemma rename_type_tag_immediate (r : renaming) (t : flambda_type) :
  rename_type r (tag_immediate t) = tag_immediate (rename_type r t).
Proof. reflexivity. Qed.

Theorem rename_type_of_head : rename_type_of_head_statement.
Proof. intros r. repeat split. Qed.

Lemma rename_type_immutable_block (r : renaming) (is_unique : bool) (tag : Z)
    (shape : block_shape) (alloc_mode : Alloc_mode.t)
    (fields : list flambda_type) :
  rename_type r (immutable_block is_unique tag shape alloc_mode fields)
  = immutable_block is_unique tag shape alloc_mode
      (map (rename_type r) fields).
Proof. unfold immutable_block. rewrite length_map. reflexivity. Qed.

Theorem rename_type_of_blocks : rename_type_of_blocks_statement.
Proof.
  intros r tag size shape maps_to alloc_mode.
  repeat split. intros is_unique. apply rename_type_immutable_block.
Qed.

Theorem rename_type_of_relation : rename_type_of_relation_statement.
Proof. intros r n. repeat split. Qed.

Lemma rename_type_exactly_this_closure (r : renaming) (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t) :
  rename_type r
    (exactly_this_closure slot function_types closure_types value_slot_types
       alloc_mode)
  = exactly_this_closure slot function_types
      (map_values (rename_type r) closure_types)
      (map_values (rename_type r) value_slot_types) alloc_mode.
Proof.
  unfold exactly_this_closure. rewrite domain_map_values. reflexivity.
Qed.

Lemma rename_type_at_least_this_closure (r : renaming) (slot : function_slot)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type)
    (alloc_mode : Alloc_mode.t) :
  rename_type r
    (at_least_this_closure slot function_types closure_types value_slot_types
       alloc_mode)
  = at_least_this_closure slot function_types
      (map_values (rename_type r) closure_types)
      (map_values (rename_type r) value_slot_types) alloc_mode.
Proof.
  unfold at_least_this_closure. rewrite domain_map_values. reflexivity.
Qed.

Theorem rename_type_of_closure : rename_type_of_closure_statement.
Proof.
  intros r slot function_types closure_types value_slot_types alloc_mode.
  split;
    [ apply rename_type_exactly_this_closure
    | apply rename_type_at_least_this_closure ].
Qed.
