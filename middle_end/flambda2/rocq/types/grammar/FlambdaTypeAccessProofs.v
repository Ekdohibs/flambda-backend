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


(* Proofs of the statements of FlambdaTypeAccessSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs FlambdaTypeRenameSpec
  FlambdaTypeAccessSpec.
Import ListNotations.

(* ======================================================================== *)
(* Blocks                                                                   *)
(* ======================================================================== *)

Theorem all_tags_satisfied : all_tags_satisfied_statement.
Proof.
  intros statics vars tag fields [known_tags other_tags alloc_mode] tags.
  unfold row_like_for_blocks_all_tags.
  destruct other_tags as [case | ]; [ discriminate | ]. intros [= <-].
  rewrite row_like_for_blocks_find_iff. unfold row_like_block_case in *.
  destruct (Keyed.lookup Z.eqb tag known_tags) as [case | ] eqn:Hfound;
    [ intros _ | intros [] ].
  apply (keyed_among_in Z Z.eqb Z.eqb_eq).
  rewrite (keyed_among_domain Z Z.eqb Z.eqb_eq). apply keyed_is_bound_lookup.
  exists case. exact Hfound.
Qed.

Theorem all_tags_and_sizes_satisfied : all_tags_and_sizes_satisfied_statement.
Proof.
  intros statics vars tag fields [known_tags other_tags alloc_mode] sizes.
  unfold row_like_for_blocks_all_tags_and_sizes.
  destruct other_tags as [case | ]; [ discriminate | ].
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test eqn:Hall
  end; [ | discriminate ].
  intros [= <-].
  rewrite row_like_for_blocks_find_iff, keyed_lookup_map_values.
  rewrite (keyed_for_all_bindings_b Z Z.eqb Z.eqb_eq),
    (keyed_for_all_bindings_lookup Z Z.eqb Z.eqb_eq) in Hall.
  unfold row_like_block_case in *.
  destruct (Keyed.lookup Z.eqb tag known_tags) as [case | ] eqn:Hfound;
    [ | intros [] ].
  specialize (Hall tag case Hfound). cbv beta in Hall.
  destruct case as [case | ]; [ | discriminate Hall ].
  unfold block_case_size_and_shape in *.
  destruct (index_domain (case_index case)) as [size | size] eqn:Hdomain;
    [ | discriminate Hall ].
  intros [Hindex _]. unfold block_index_satisfied in Hindex.
  rewrite Hdomain in Hindex.
  exists size, (index_shape (case_index case)).
  split; [ simpl; rewrite Hdomain; reflexivity | exact Hindex ].
Qed.

(* What [get_singleton] answers is the one case of the structure. *)
Lemma get_singleton_case (known_tags : Keyed.finite_map Z _)
    (other_tags : Or_bottom.t row_like_block_case)
    (alloc_mode alloc_mode' : Alloc_mode.t) (tag : Z) (shape : block_shape)
    (size : nat) (tys : list flambda_type) :
  row_like_for_blocks_get_singleton
    (Row_like_for_blocks known_tags other_tags alloc_mode)
  = Some (tag, shape, size, tys, alloc_mode') ->
  other_tags = Or_bottom.Bottom
  ∧ ∃ case : row_like_block_case,
      (∀ tag' : Z,
        Keyed.lookup Z.eqb tag' known_tags
        = if Z.eqb tag' tag then Some (Or_unknown.Known case) else None)
      ∧ index_domain (case_index case) = Row_like_index_domain.Known size
      ∧ index_shape (case_index case) = shape
      ∧ case_maps_to case = tys.
Proof.
  unfold row_like_for_blocks_get_singleton.
  destruct other_tags as [other | ]; [ discriminate | ].
  destruct (Keyed.get_singleton_binding Z.eqb known_tags)
    as [[tag0 [case | ]] | ] eqn:Hsingleton;
    try discriminate.
  destruct (index_domain (case_index case)) as [size0 | size0] eqn:Hdomain;
    [ | discriminate ].
  intros [= <- <- <- <- _]. split; [ reflexivity | ]. exists case.
  split; [ | auto ].
  exact (proj1 (keyed_get_singleton_binding Z Z.eqb Z.eqb_eq _ _ _ _)
           Hsingleton).
Qed.

Theorem get_singleton_satisfied : get_singleton_satisfied_statement.
Proof.
  intros statics vars tag' fields [known_tags other_tags alloc_mode0] tag
    shape size tys alloc_mode Hsingleton.
  apply get_singleton_case in Hsingleton
    as [-> [case [Hlookup [Hdomain [<- <-]]]]].
  rewrite row_like_for_blocks_find_iff. unfold row_like_block_case in *.
  rewrite (Hlookup tag').
  destruct (Z.eqb tag' tag) eqn:Htag; [ | intros [] ].
  apply Z.eqb_eq in Htag. subst tag'.
  intros [Hindex [Hfields _]]. unfold block_index_satisfied in Hindex.
  rewrite Hdomain in Hindex. destruct Hindex as [Hsize Hshape].
  split; [ reflexivity | ]. split; [ exact Hsize | ].
  split; [ exact Hshape | exact Hfields ].
Qed.

Theorem get_field_satisfied : get_field_satisfied_statement.
Proof.
  intros statics vars tag fields blocks index Hrow.
  unfold row_like_for_blocks_get_field.
  destruct (row_like_for_blocks_get_singleton blocks)
    as [[[[[tag0 shape] size] tys] alloc_mode] | ] eqn:Hsingleton;
    [ | exact I ].
  destruct (get_singleton_satisfied statics vars tag fields blocks tag0 shape
              size tys alloc_mode Hsingleton Hrow)
    as [_ [Hsize [_ Hfields]]].
  destruct (Nat.leb size index) eqn:Hindex.
  - apply Nat.leb_le in Hindex. apply nth_error_None. lia.
  - apply Nat.leb_gt in Hindex.
    destruct (nth_error tys index) as [ty | ] eqn:Hty; [ | exact I ].
    destruct (nth_error fields index) as [field | ] eqn:Hfield.
    + exists field. split; [ reflexivity | ].
      exact (proj1 (fields_satisfied_nth_iff _ _ _) Hfields index ty field
               Hty Hfield).
    + apply nth_error_None in Hfield. lia.
Qed.

(* ======================================================================== *)
(* Closures                                                                 *)
(* ======================================================================== *)

Theorem find_function_type_satisfied : find_function_type_satisfied_statement.
Proof.
  intros statics vars v slot functions value_slots
    [function_types closure_types value_slot_types] exact g function_type
    code Hentry.
  apply closures_entry_lookup_iff in Hentry as [Hfunctions _].
  unfold closures_entry_find_function_type.
  destruct (lookup g function_types) as [[function_type' | ] | ] eqn:Hfound;
    try discriminate; [ | destruct exact; discriminate ].
  intros [= <-] Hcode.
  exact (Hfunctions g function_type' code Hfound Hcode).
Qed.

Theorem find_function_type_exact : find_function_type_exact_statement.
Proof.
  intros slot [function_types closure_types value_slot_types] contents
    extension Hcase.
  pose proof (well_formed_exact_case_function_type slot function_types
                closure_types value_slot_types contents extension Hcase)
    as Hbound.
  apply is_bound_lookup in Hbound as [function_type Hfound].
  unfold closures_entry_find_function_type. rewrite Hfound.
  destruct function_type; discriminate.
Qed.

(* What [get_single_tag] answers is the entry of the one case of the
   structure. *)
Lemma get_single_tag_case
    (known_closures : finite_map row_like_closures_case)
    (slot : function_slot) (entry : closures_entry) :
  single_closures_entry
    (row_like_for_closures_get_single_tag
       (Row_like_for_closures known_closures))
  = Some (slot, entry) ->
  ∃ case : row_like_closures_case,
    (∀ slot' : function_slot,
      lookup slot' known_closures
      = if Pos.eqb slot' slot then Some case else None)
    ∧ case_maps_to case = entry.
Proof.
  unfold row_like_for_closures_get_single_tag, row_like_closures_case in *.
  destruct (Keyed.get_singleton_binding Pos.eqb known_closures)
    as [[slot0 case] | ] eqn:Hsingleton;
    [ | discriminate ].
  pose proof (proj1 (keyed_get_singleton_binding positive Pos.eqb Pos.eqb_eq
                       _ _ _ _)
                Hsingleton)
    as Hlookup.
  destruct (index_domain (case_index case)) as [contents | contents];
    destruct (among slot0 (contents_closures contents)); simpl;
    try discriminate;
    intros [= <- <-]; exists case; split; [ exact Hlookup | reflexivity
                                          | exact Hlookup | reflexivity ].
Qed.

Theorem get_single_tag_satisfied : get_single_tag_satisfied_statement.
Proof.
  intros statics vars v slot' functions value_slots [known_closures] slot
    entry Hsingle.
  apply get_single_tag_case in Hsingle as [case [Hlookup <-]].
  rewrite row_like_for_closures_lookup_iff.
  intros [case' [Hfound [_ [Hentry _]]]].
  rewrite (Hlookup slot') in Hfound.
  destruct (Pos.eqb slot' slot) eqn:Hslot; [ | discriminate Hfound ].
  apply Pos.eqb_eq in Hslot. subst slot'. injection Hfound as <-.
  split; [ reflexivity | exact Hentry ].
Qed.

Theorem get_single_tag_exact : get_single_tag_exact_statement.
Proof.
  intros known_closures slot entry.
  unfold row_like_for_closures_get_single_tag, row_like_closures_case in *.
  destruct (Keyed.get_singleton_binding Pos.eqb known_closures)
    as [[slot0 [maps_to [domain []] extension]] | ] eqn:Hsingleton;
    [ | discriminate ].
  pose proof (proj1 (keyed_get_singleton_binding positive Pos.eqb Pos.eqb_eq
                       _ _ _ _)
                Hsingleton)
    as Hlookup.
  simpl. destruct domain as [contents | contents];
    destruct (among slot0 (contents_closures contents)) eqn:Hamong;
    try discriminate.
  intros [= <- <-]. exists contents, extension.
  split; [ exact Hlookup | exact Hamong ].
Qed.

Theorem get_single_tag_well_formed : get_single_tag_well_formed_statement.
Proof.
  intros [known_closures].
  rewrite (row_like_for_closures_well_formed_iff _). intros [_ Hcases].
  unfold row_like_for_closures_get_single_tag, row_like_closures_case in *.
  destruct (Keyed.get_singleton_binding Pos.eqb known_closures)
    as [[slot case] | ] eqn:Hsingleton;
    [ | discriminate ].
  pose proof (proj1 (keyed_get_singleton_binding positive Pos.eqb Pos.eqb_eq
                       _ _ _ _)
                Hsingleton slot)
    as Hlookup.
  rewrite Pos.eqb_refl in Hlookup.
  pose proof (Hcases slot case Hlookup) as Hcase.
  unfold row_like_closures_case_well_formed, closures_index_has_slot in Hcase.
  rewrite !andb_true_iff in Hcase. destruct Hcase as [[Hslot _] _].
  destruct (index_domain (case_index case)) as [contents | contents].
  - rewrite Hslot. discriminate.
  - destruct (among slot (contents_closures contents)); discriminate.
Qed.

Theorem get_closure_satisfied : get_closure_satisfied_statement.
Proof.
  intros statics vars v slot functions value_slots closures g ty closure
    Hrow.
  unfold row_like_for_closures_get_closure.
  destruct (single_closures_entry
              (row_like_for_closures_get_single_tag closures))
    as [[slot0 entry] | ] eqn:Hsingle.
  - destruct (get_single_tag_satisfied statics vars v slot functions
                value_slots closures slot0 entry Hsingle Hrow)
      as [<- Hentry].
    destruct entry as [function_types closure_types value_slot_types].
    apply closures_entry_lookup_iff in Hentry as [_ [Hclosures _]].
    destruct (row_like_for_closures_get_single_tag closures)
      as [[ | slot1 entry1 | slot1 entry1] | ];
      simpl in Hsingle; try discriminate Hsingle;
      injection Hsingle as -> ->;
      (destruct (lookup g closure_types) as [ty' | ] eqn:Hfound;
       [ | discriminate ]);
      intros [= <-] Hproject;
      exact (Hclosures g ty' closure Hfound Hproject).
  - destruct (row_like_for_closures_get_single_tag closures)
      as [[ | slot1 entry1 | slot1 entry1] | ];
      simpl in Hsingle; try discriminate Hsingle; discriminate.
Qed.

Theorem get_env_var_satisfied : get_env_var_satisfied_statement.
Proof.
  intros statics vars v slot functions value_slots closures w ty contents
    Hrow.
  unfold row_like_for_closures_get_env_var.
  destruct (single_closures_entry
              (row_like_for_closures_get_single_tag closures))
    as [[slot0 entry] | ] eqn:Hsingle.
  - destruct (get_single_tag_satisfied statics vars v slot functions
                value_slots closures slot0 entry Hsingle Hrow)
      as [<- Hentry].
    destruct entry as [function_types closure_types value_slot_types].
    apply closures_entry_lookup_iff in Hentry as [_ [_ Hslots]].
    destruct (row_like_for_closures_get_single_tag closures)
      as [[ | slot1 entry1 | slot1 entry1] | ];
      simpl in Hsingle; try discriminate Hsingle;
      injection Hsingle as -> ->;
      (destruct (lookup w value_slot_types) as [ty' | ] eqn:Hfound;
       [ | discriminate ]);
      intros [= <-] Hcontents;
      exact (Hslots w ty' contents Hfound Hcontents).
  - destruct (row_like_for_closures_get_single_tag closures)
      as [[ | slot1 entry1 | slot1 entry1] | ];
      simpl in Hsingle; try discriminate Hsingle; discriminate.
Qed.

(* ======================================================================== *)
(* The answers on a renamed type                                            *)
(* ======================================================================== *)

(* The one binding of a map, and a test of its bindings, under a
   function on what it binds. *)
Lemma keyed_get_singleton_binding_map_values {K A B : Type}
    (eqb : K -> K -> bool) (Heqb : decides_equality eqb) (f : A -> B)
    (m : Keyed.finite_map K A) :
  Keyed.get_singleton_binding eqb (Keyed.map_values f m)
  = option_map (fun binding => (fst binding, f (snd binding)))
      (Keyed.get_singleton_binding eqb m).
Proof.
  destruct (Keyed.get_singleton_binding eqb m) as [[key a] | ] eqn:Hm; simpl.
  - apply (keyed_get_singleton_binding K eqb Heqb). intros key'.
    rewrite keyed_lookup_map_values,
      (proj1 (keyed_get_singleton_binding K eqb Heqb A key a m) Hm key').
    destruct (eqb key' key); reflexivity.
  - destruct (Keyed.get_singleton_binding eqb (Keyed.map_values f m))
      as [[key b] | ] eqn:Hmapped;
      [ | reflexivity ].
    pose proof (proj1 (keyed_get_singleton_binding K eqb Heqb B key b _)
                  Hmapped)
      as Hlookup.
    pose proof (Hlookup key) as Hkey.
    rewrite keyed_lookup_map_values, (proj2 (Heqb key key) eq_refl) in Hkey.
    destruct (Keyed.lookup eqb key m) as [a | ] eqn:Hfound;
      [ | discriminate Hkey ].
    assert (Hsingle : Keyed.get_singleton_binding eqb m = Some (key, a)).
    { apply (keyed_get_singleton_binding K eqb Heqb). intros key'.
      specialize (Hlookup key'). rewrite keyed_lookup_map_values in Hlookup.
      destruct (eqb key' key) eqn:Hsame.
      - apply Heqb in Hsame. subst key'. exact Hfound.
      - destruct (Keyed.lookup eqb key' m); [ discriminate Hlookup | ].
        reflexivity. }
    rewrite Hsingle in Hm. discriminate Hm.
Qed.

Lemma keyed_for_all_bindings_b_map_values {K A B : Type}
    (eqb : K -> K -> bool) (Heqb : decides_equality eqb) (f : A -> B)
    (p : K -> B -> bool) (q : K -> A -> bool) (m : Keyed.finite_map K A) :
  (∀ (key : K) (a : A), p key (f a) = q key a) ->
  Keyed.for_all_bindings_b eqb p (Keyed.map_values f m)
  = Keyed.for_all_bindings_b eqb q m.
Proof.
  intros Hpq. apply eq_true_iff_eq.
  rewrite !(keyed_for_all_bindings_b K eqb Heqb),
    !(keyed_for_all_bindings_lookup K eqb Heqb).
  split.
  - intros Hall key a Hlookup. rewrite <- Hpq. apply Hall.
    rewrite keyed_lookup_map_values, Hlookup. reflexivity.
  - intros Hall key b Hlookup. rewrite keyed_lookup_map_values in Hlookup.
    destruct (Keyed.lookup eqb key m) as [a | ] eqn:Hfound;
      [ | discriminate Hlookup ].
    injection Hlookup as <-. rewrite Hpq. exact (Hall key a Hfound).
Qed.

Lemma block_case_size_and_shape_renamed (r : renaming)
    (case : Or_unknown.t row_like_block_case) :
  block_case_size_and_shape
    (or_unknown_map
       (rename_block_case (rename_type r) (rename_env_extension r)) case)
  = block_case_size_and_shape case.
Proof. destruct case as [case | ]; reflexivity. Qed.

Theorem all_tags_rename : all_tags_rename_statement.
Proof.
  intros r [known_tags [other | ] alloc_mode]; simpl;
    [ split; reflexivity | ].
  split.
  - unfold Keyed.domain, Keyed.map_values. rewrite map_map. reflexivity.
  - rewrite (keyed_for_all_bindings_b_map_values Z.eqb Z.eqb_eq _ _
               (fun _ case =>
                  match block_case_size_and_shape case with
                  | Some _ => true
                  | None => false
                  end));
      [ | intros key case; rewrite block_case_size_and_shape_renamed;
          reflexivity ].
    match goal with
    | |- (if ?test then _ else _) = _ => destruct test
    end; [ | reflexivity ].
    f_equal. unfold Keyed.map_values. rewrite map_map. apply map_ext.
    intros [tag case]. simpl. rewrite block_case_size_and_shape_renamed.
    reflexivity.
Qed.

Theorem get_singleton_rename : get_singleton_rename_statement.
Proof.
  intros r [known_tags [other | ] alloc_mode]; simpl; [ reflexivity | ].
  rewrite (keyed_get_singleton_binding_map_values Z.eqb Z.eqb_eq).
  unfold row_like_block_case in *.
  destruct (Keyed.get_singleton_binding Z.eqb known_tags)
    as [[tag [case | ]] | ]; simpl; try reflexivity.
  destruct (index_domain (case_index case)); reflexivity.
Qed.

Theorem get_field_rename : get_field_rename_statement.
Proof.
  intros r blocks index. unfold row_like_for_blocks_get_field.
  rewrite get_singleton_rename.
  destruct (row_like_for_blocks_get_singleton blocks)
    as [[[[[tag shape] size] tys] alloc_mode] | ]; simpl; [ | reflexivity ].
  destruct (Nat.leb size index); [ reflexivity | ].
  rewrite nth_error_map. destruct (nth_error tys index); reflexivity.
Qed.

Theorem find_function_type_rename : find_function_type_rename_statement.
Proof.
  intros r [function_types closure_types value_slot_types] exact slot.
  reflexivity.
Qed.

Theorem get_single_tag_rename : get_single_tag_rename_statement.
Proof.
  intros r [known_closures].
  assert (Hresult :
            row_like_for_closures_get_single_tag
              (rename_row_like_for_closures r
                 (Row_like_for_closures known_closures))
            = option_map (rename_get_single_tag_result r)
                (row_like_for_closures_get_single_tag
                   (Row_like_for_closures known_closures))).
  { simpl.
    rewrite (keyed_get_singleton_binding_map_values Pos.eqb Pos.eqb_eq).
    unfold row_like_closures_case in *.
    destruct (Keyed.get_singleton_binding Pos.eqb known_closures)
      as [[slot case] | ]; simpl; [ | reflexivity ].
    destruct (index_domain (case_index case)) as [contents | contents];
      destruct (among slot (contents_closures contents)); reflexivity. }
  split; [ exact Hresult | ]. rewrite Hresult.
  destruct (row_like_for_closures_get_single_tag
              (Row_like_for_closures known_closures))
    as [[ | slot entry | slot entry] | ]; reflexivity.
Qed.

Theorem get_closure_rename : get_closure_rename_statement.
Proof.
  intros r closures.
  unfold row_like_for_closures_get_closure,
    row_like_for_closures_get_env_var.
  rewrite (proj1 (get_single_tag_rename r closures)).
  split; intros slot;
    destruct (row_like_for_closures_get_single_tag closures)
      as [[ | slot0 [function_types closure_types value_slot_types]
           | slot0 [function_types closure_types value_slot_types]] | ];
    simpl; try reflexivity; rewrite lookup_map_values;
    match goal with
    | |- context [lookup slot ?m] => destruct (lookup slot m)
    end; reflexivity.
Qed.
