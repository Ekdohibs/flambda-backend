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


(* Proofs of the statements of FlambdaTypeEqualSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaPrimitiveProofs
  FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeMeaningBoolSpec FlambdaTypeMeaningBoolProofs
  FlambdaTypeEqualSpec.
Import ListNotations.

(* ======================================================================== *)
(* What types are made of                                                   *)
(* ======================================================================== *)

Theorem map_eqb_iff : map_eqb_iff_statement.
Proof.
  intros K key_eqb Hkey A eqb m m'. unfold map_eqb.
  rewrite andb_true_iff, !(keyed_for_all_bindings_b K key_eqb Hkey),
    !(keyed_for_all_bindings_lookup K key_eqb Hkey).
  split.
  - intros [Hleft Hright] key.
    destruct (Keyed.lookup key_eqb key m) as [a | ] eqn:Hlookup.
    + specialize (Hleft key a Hlookup). cbv beta in Hleft.
      destruct (Keyed.lookup key_eqb key m') as [a' | ];
        [ exact Hleft | discriminate Hleft ].
    + destruct (Keyed.lookup key_eqb key m') as [a' | ] eqn:Hlookup';
        [ | exact I ].
      specialize (Hright key a' Hlookup'). cbv beta in Hright.
      unfold Keyed.is_bound in Hright. rewrite Hlookup in Hright.
      discriminate Hright.
  - intros Hboth. split.
    + intros key a Hlookup. specialize (Hboth key). rewrite Hlookup in Hboth.
      cbv beta. destruct (Keyed.lookup key_eqb key m') as [a' | ];
        [ exact Hboth | destruct Hboth ].
    + intros key a' Hlookup'. specialize (Hboth key).
      rewrite Hlookup' in Hboth. cbv beta. unfold Keyed.is_bound.
      destruct (Keyed.lookup key_eqb key m) as [a | ];
        [ reflexivity | destruct Hboth ].
Qed.

(* The tests of the small types tell whether two are the same. *)

Lemma simple_eqb_eq : decides_equality simple_eqb.
Proof.
  intros [x | sym | c] [x' | sym' | c']; simpl;
    try (split; discriminate).
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - rewrite (const_eqb_eq c c'). split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
Qed.

Lemma option_variable_eqb_eq : decides_equality option_variable_eqb.
Proof.
  intros [x | ] [x' | ]; simpl; try (split; discriminate).
  - rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | ].
    intros [= ->]. reflexivity.
  - split; reflexivity.
Qed.

Lemma alloc_mode_eqb_eq : decides_equality alloc_mode_eqb.
Proof. intros [] []; simpl; split; (reflexivity || discriminate). Qed.

Lemma relation_eqb_eq : decides_equality relation_eqb.
Proof. intros [] []; simpl; split; (reflexivity || discriminate). Qed.

Lemma naked_number_kind_eqb_eq : decides_equality naked_number_kind_eqb.
Proof.
  intros k k'. unfold naked_number_kind_eqb. rewrite (kind_eqb_eq _ _).
  split; [ intros [= ->]; reflexivity | intros ->; reflexivity ].
Qed.

Lemma boxable_number_eqb_eq : decides_equality boxable_number_eqb.
Proof.
  intros k k'. unfold boxable_number_eqb. rewrite (kind_eqb_eq _ _).
  split; [ apply unboxed_kind_injective | intros ->; reflexivity ].
Qed.

Lemma is_null_eqb_eq : decides_equality is_null_eqb.
Proof.
  intros [ | x] [ | x']; simpl; try (split; discriminate).
  - split; reflexivity.
  - rewrite (option_variable_eqb_eq x x').
    split; [ intros ->; reflexivity | intros [= ->]; reflexivity ].
Qed.

Lemma relation_and_name_eqb_eq :
  decides_equality
    (fun relation relation' : Relation.t * name =>
       relation_eqb (fst relation) (fst relation')
       && name_eqb (snd relation) (snd relation')).
Proof.
  intros [relation n] [relation' n']. simpl.
  rewrite andb_true_iff, (relation_eqb_eq _ _), (name_eqb_eq _ _).
  split; [ intros [-> ->]; reflexivity | intros [= -> ->]; auto ].
Qed.

Lemma naked_number_kinds_eqb_eq :
  decides_equality (lists_eqb naked_number_kind_eqb).
Proof.
  intros l l'. apply lists_eqb_eq. intros k k' _.
  apply naked_number_kind_eqb_eq.
Qed.

Lemma block_shape_eqb_eq : decides_equality block_shape_eqb.
Proof.
  intros [[ | prefix suffix] | ] [[ | prefix' suffix'] | ]; simpl;
    try (split; (reflexivity || discriminate)).
  rewrite andb_true_iff, Nat.eqb_eq, (naked_number_kinds_eqb_eq _ _).
  split; [ intros [-> ->]; reflexivity | intros [= -> ->]; auto ].
Qed.

Lemma block_index_eqb_eq : decides_equality block_index_eqb.
Proof.
  intros [[size | size] shape] [[size' | size'] shape'];
    unfold block_index_eqb; simpl;
    rewrite ?andb_true_iff, ?Nat.eqb_eq, ?(block_shape_eqb_eq _ _).
  - split; [ intros [-> ->]; reflexivity | intros [= -> ->]; auto ].
  - split; discriminate.
  - split; discriminate.
  - split; [ intros [-> ->]; reflexivity | intros [= -> ->]; auto ].
Qed.

Lemma function_type_eqb_eq : decides_equality function_type_eqb.
Proof.
  intros [code] [code']. unfold function_type_eqb. simpl.
  rewrite Pos.eqb_eq.
  split; [ intros ->; reflexivity | intros [= ->]; reflexivity ].
Qed.

(* Two lists of numbers, of strings or of slots with the same members. *)
Lemma same_members {K : Type} (eqb : K -> K -> bool) (keys keys' : list K) :
  decides_equality eqb ->
  Keyed.same_keys_b eqb keys keys' = true ->
  ∀ key : K, key ∈ keys <-> key ∈ keys'.
Proof.
  intros Heqb Hsame key.
  apply (keyed_same_keys_b K eqb Heqb) in Hsame.
  rewrite <- !(keyed_among_in K eqb Heqb), (Hsame key). reflexivity.
Qed.

Theorem env_extension_equal_lookup : env_extension_equal_lookup_statement.
Proof.
  intros equations equations'.
  exact (map_eqb_iff name name_eqb name_eqb_eq _ _ _ _).
Qed.

Theorem type_equal_kind : type_equal_kind_statement.
Proof.
  intros t t'. unfold type_equal.
  destruct t, t'; simpl; try discriminate; try reflexivity.
  rewrite andb_true_iff, (naked_number_kind_eqb_eq _ _).
  intros [-> _]. reflexivity.
Qed.

(* ======================================================================== *)
(* Maps that are the same                                                   *)
(* ======================================================================== *)

(* What [map_eqb] gives at a key, with the bindings as pairs of the
   lists, which is what an induction on types asks for. *)
Lemma map_eqb_lookup {K A : Type} (key_eqb : K -> K -> bool)
    (Hkey : decides_equality key_eqb) (eqb : A -> A -> bool)
    (m m' : Keyed.finite_map K A) (key : K) :
  map_eqb key_eqb eqb m m' = true ->
  match Keyed.lookup key_eqb key m, Keyed.lookup key_eqb key m' with
  | Some a, Some a' => (key, a) ∈ m ∧ (key, a') ∈ m' ∧ eqb a a' = true
  | None, None => True
  | Some _, None | None, Some _ => False
  end.
Proof.
  intros Hmap.
  pose proof (proj1 (map_eqb_iff K key_eqb Hkey A eqb m m') Hmap key) as Hat.
  destruct (Keyed.lookup key_eqb key m) as [a | ] eqn:Hlookup,
    (Keyed.lookup key_eqb key m') as [a' | ] eqn:Hlookup';
    try exact Hat.
  split; [ exact (keyed_lookup_in key_eqb Hkey _ _ _ Hlookup) | ].
  split; [ exact (keyed_lookup_in key_eqb Hkey _ _ _ Hlookup') | exact Hat ].
Qed.

(* What holds of every binding of one of two maps that are the same
   holds of every binding of the other, if it holds alike of things
   that are the same. *)
Lemma map_eqb_for_all {K A : Type} (key_eqb : K -> K -> bool)
    (Hkey : decides_equality key_eqb) (eqb : A -> A -> bool)
    (P : K -> A -> Prop) (m m' : Keyed.finite_map K A) :
  map_eqb key_eqb eqb m m' = true ->
  (∀ (key : K) (a a' : A),
    (key, a) ∈ m -> eqb a a' = true -> (P key a <-> P key a')) ->
  ((∀ (key : K) (a : A), Keyed.lookup key_eqb key m = Some a -> P key a)
   <-> (∀ (key : K) (a' : A),
         Keyed.lookup key_eqb key m' = Some a' -> P key a')).
Proof.
  intros Hmap HP. split.
  - intros Hall key a' Hlookup'.
    pose proof (map_eqb_lookup key_eqb Hkey eqb m m' key Hmap) as Hat.
    rewrite Hlookup' in Hat.
    destruct (Keyed.lookup key_eqb key m) as [a | ] eqn:Hlookup;
      [ | destruct Hat ].
    destruct Hat as [Hin [_ Heq]].
    apply (HP key a a' Hin Heq). exact (Hall key a Hlookup).
  - intros Hall key a Hlookup.
    pose proof (map_eqb_lookup key_eqb Hkey eqb m m' key Hmap) as Hat.
    rewrite Hlookup in Hat.
    destruct (Keyed.lookup key_eqb key m') as [a' | ] eqn:Hlookup';
      [ | destruct Hat ].
    destruct Hat as [Hin [_ Heq]].
    apply (HP key a a' Hin Heq). exact (Hall key a' Hlookup').
Qed.

(* Two maps that are the same, of things that the test tells apart,
   bind every key alike. *)
Lemma map_eqb_same_lookup {K A : Type} (key_eqb : K -> K -> bool)
    (Hkey : decides_equality key_eqb) (eqb : A -> A -> bool)
    (Heqb : decides_equality eqb) (m m' : Keyed.finite_map K A) (key : K) :
  map_eqb key_eqb eqb m m' = true ->
  Keyed.lookup key_eqb key m = Keyed.lookup key_eqb key m'.
Proof.
  intros Hmap.
  pose proof (proj1 (map_eqb_iff K key_eqb Hkey A eqb m m') Hmap key) as Hat.
  destruct (Keyed.lookup key_eqb key m) as [a | ],
    (Keyed.lookup key_eqb key m') as [a' | ];
    [ apply Heqb in Hat; rewrite Hat; reflexivity | destruct Hat
    | destruct Hat | reflexivity ].
Qed.

Lemma or_unknown_eqb_eq {A : Type} (eqb : A -> A -> bool) :
  decides_equality eqb -> decides_equality (or_unknown_eqb eqb).
Proof.
  intros Heqb [a | ] [a' | ]; simpl; try (split; discriminate).
  - rewrite (Heqb a a').
    split; [ intros ->; reflexivity | intros [= ->]; reflexivity ].
  - split; reflexivity.
Qed.

(* ======================================================================== *)
(* An equivalence                                                           *)
(* ======================================================================== *)

(* What it is for a test to be an equivalence, said of one thing: it is
   the same as itself; what it is the same as is the same as it; and
   what that is the same as, it is the same as. An induction on types
   proves the three together of each structure. *)
Definition equivalence_at {A : Type} (eqb : A -> A -> bool) (a : A) : Prop :=
  eqb a a = true
  ∧ (∀ a' : A, eqb a a' = true -> eqb a' a = true)
  ∧ (∀ a' a'' : A,
      eqb a a' = true -> eqb a' a'' = true -> eqb a a'' = true).

Lemma decides_equality_equivalence_at {A : Type} (eqb : A -> A -> bool)
    (a : A) :
  decides_equality eqb -> equivalence_at eqb a.
Proof.
  intros Heqb. split; [ apply Heqb; reflexivity | ]. split.
  - intros a' Heq. apply Heqb in Heq. subst a'. apply Heqb. reflexivity.
  - intros a' a'' Heq Heq'. apply Heqb in Heq. subst a'. exact Heq'.
Qed.

Lemma always_equivalence_at {A : Type} (a : A) :
  equivalence_at (fun _ _ : A => true) a.
Proof. split; [ reflexivity | ]. split; intros; reflexivity. Qed.

Lemma same_keys_equivalence_at {K : Type} (eqb : K -> K -> bool)
    (keys : list K) :
  decides_equality eqb -> equivalence_at (Keyed.same_keys_b eqb) keys.
Proof.
  intros Heqb. split; [ | split ].
  - apply (keyed_same_keys_b K eqb Heqb), keyed_same_keys_refl.
  - intros keys' Hsame. apply (keyed_same_keys_b K eqb Heqb) in Hsame.
    apply (keyed_same_keys_b K eqb Heqb), keyed_same_keys_sym, Hsame.
  - intros keys' keys'' Hsame Hsame'.
    apply (keyed_same_keys_b K eqb Heqb) in Hsame, Hsame'.
    apply (keyed_same_keys_b K eqb Heqb).
    exact (keyed_same_keys_trans eqb _ _ _ Hsame Hsame').
Qed.

Lemma or_unknown_equivalence_at {A : Type} (eqb : A -> A -> bool)
    (a : Or_unknown.t A) :
  (∀ x : A, a = Or_unknown.Known x -> equivalence_at eqb x) ->
  equivalence_at (or_unknown_eqb eqb) a.
Proof.
  intros Hx. destruct a as [x | ].
  - destruct (Hx x eq_refl) as [Hrefl [Hsym Htrans]].
    split; [ exact Hrefl | ]. split.
    + intros [x' | ]; simpl; [ apply Hsym | discriminate ].
    + intros [x' | ] [x'' | ]; simpl;
        [ apply Htrans | intros _ H; exact H | discriminate | discriminate ].
  - split; [ reflexivity | ]. split.
    + intros [x' | ]; simpl; [ discriminate | reflexivity ].
    + intros [x' | ] [x'' | ]; simpl;
        [ discriminate | discriminate | intros _ H; exact H | reflexivity ].
Qed.

Lemma or_bottom_equivalence_at {A : Type} (eqb : A -> A -> bool)
    (a : Or_bottom.t A) :
  (∀ x : A, a = Or_bottom.Ok x -> equivalence_at eqb x) ->
  equivalence_at (or_bottom_eqb eqb) a.
Proof.
  intros Hx. destruct a as [x | ].
  - destruct (Hx x eq_refl) as [Hrefl [Hsym Htrans]].
    split; [ exact Hrefl | ]. split.
    + intros [x' | ]; simpl; [ apply Hsym | discriminate ].
    + intros [x' | ] [x'' | ]; simpl;
        [ apply Htrans | intros _ H; exact H | discriminate | discriminate ].
  - split; [ reflexivity | ]. split.
    + intros [x' | ]; simpl; [ discriminate | reflexivity ].
    + intros [x' | ] [x'' | ]; simpl;
        [ discriminate | discriminate | intros _ H; exact H | reflexivity ].
Qed.

Lemma or_unknown_or_bottom_equivalence_at {A : Type} (eqb : A -> A -> bool)
    (a : or_unknown_or_bottom A) :
  (∀ x : A, a = Ok x -> equivalence_at eqb x) ->
  equivalence_at (or_unknown_or_bottom_eqb eqb) a.
Proof.
  intros Hx. destruct a as [ | x | ].
  - split; [ reflexivity | ]. split.
    + intros [ | x' | ]; simpl; (reflexivity || discriminate).
    + intros [ | x' | ] [ | x'' | ]; simpl; try discriminate; reflexivity.
  - destruct (Hx x eq_refl) as [Hrefl [Hsym Htrans]].
    split; [ exact Hrefl | ]. split.
    + intros [ | x' | ]; simpl; [ discriminate | apply Hsym | discriminate ].
    + intros [ | x' | ] [ | x'' | ]; simpl; try discriminate; apply Htrans.
  - split; [ reflexivity | ]. split.
    + intros [ | x' | ]; simpl; (reflexivity || discriminate).
    + intros [ | x' | ] [ | x'' | ]; simpl; try discriminate; reflexivity.
Qed.

Lemma descr_equivalence_at {head : Type} (head_eqb : head -> head -> bool)
    (ty : type_descr head) :
  (∀ h : head, ty = Ok (No_alias h) -> equivalence_at head_eqb h) ->
  equivalence_at (descr_eqb head_eqb) ty.
Proof.
  intros Hhead. apply or_unknown_or_bottom_equivalence_at.
  intros [h | s] Hty.
  - destruct (Hhead h Hty) as [Hrefl [Hsym Htrans]].
    split; [ exact Hrefl | ]. split.
    + intros [h' | s']; [ apply Hsym | discriminate ].
    + intros [h' | s'] [h'' | s''];
        [ apply Htrans | intros _ H; exact H | discriminate | discriminate ].
  - destruct (decides_equality_equivalence_at simple_eqb s simple_eqb_eq)
      as [Hrefl [Hsym Htrans]].
    split; [ exact Hrefl | ]. split.
    + intros [h' | s']; [ discriminate | apply Hsym ].
    + intros [h' | s'] [h'' | s''];
        [ discriminate | discriminate | intros _ H; exact H | apply Htrans ].
Qed.

Lemma lists_equivalence_at {A : Type} (eqb : A -> A -> bool) (l : list A) :
  (∀ a : A, a ∈ l -> equivalence_at eqb a) ->
  equivalence_at (lists_eqb eqb) l.
Proof.
  induction l as [ | a l IH]; intros Hall.
  - split; [ reflexivity | ]. split.
    + intros [ | a' l']; simpl; [ reflexivity | discriminate ].
    + intros [ | a' l'] l''; simpl; [ intros _ H; exact H | discriminate ].
  - destruct (Hall a (or_introl eq_refl)) as [Hrefl [Hsym Htrans]].
    destruct (IH (fun a' Hin => Hall a' (or_intror Hin)))
      as [IHrefl [IHsym IHtrans]].
    split; [ simpl; rewrite Hrefl, IHrefl; reflexivity | ]. split.
    + intros [ | a' l']; simpl; [ discriminate | ].
      rewrite !andb_true_iff. intros [Ha Hl].
      split; [ apply Hsym, Ha | apply IHsym, Hl ].
    + intros [ | a' l'] [ | a'' l'']; simpl;
        [ discriminate | discriminate | intros _ H; exact H | ].
      rewrite !andb_true_iff. intros [Ha Hl] [Ha' Hl'].
      split; [ exact (Htrans _ _ Ha Ha') | exact (IHtrans _ _ Hl Hl') ].
Qed.

Lemma map_equivalence_at {K A : Type} (key_eqb : K -> K -> bool)
    (Hkey : decides_equality key_eqb) (eqb : A -> A -> bool)
    (m : Keyed.finite_map K A) :
  (∀ (key : K) (a : A), (key, a) ∈ m -> equivalence_at eqb a) ->
  equivalence_at (map_eqb key_eqb eqb) m.
Proof.
  intros Hall. split; [ | split ].
  - apply (map_eqb_iff K key_eqb Hkey). intros key.
    destruct (Keyed.lookup key_eqb key m) as [a | ] eqn:Hlookup;
      [ | exact I ].
    exact (proj1 (Hall key a (keyed_lookup_in key_eqb Hkey _ _ _ Hlookup))).
  - intros m' Hmap. apply (map_eqb_iff K key_eqb Hkey). intros key.
    pose proof (map_eqb_lookup key_eqb Hkey eqb m m' key Hmap) as Hat.
    destruct (Keyed.lookup key_eqb key m) as [a | ],
      (Keyed.lookup key_eqb key m') as [a' | ]; try exact Hat.
    destruct Hat as [Hin [_ Heq]].
    exact (proj1 (proj2 (Hall key a Hin)) a' Heq).
  - intros m' m'' Hmap Hmap'. apply (map_eqb_iff K key_eqb Hkey).
    intros key.
    pose proof (map_eqb_lookup key_eqb Hkey eqb m m' key Hmap) as Hat.
    pose proof (proj1 (map_eqb_iff K key_eqb Hkey A eqb m' m'') Hmap' key)
      as Hat'.
    destruct (Keyed.lookup key_eqb key m) as [a | ],
      (Keyed.lookup key_eqb key m') as [a' | ],
      (Keyed.lookup key_eqb key m'') as [a'' | ];
      try exact Hat'; try (destruct Hat; fail).
    destruct Hat as [Hin [_ Heq]].
    exact (proj2 (proj2 (Hall key a Hin)) a' a'' Heq Hat').
Qed.

(* The test of a structure is that of its parts, one after the other.
   With each part an equivalence at the part, in the context, these
   prove the three properties of the whole. *)
Ltac equivalence_parts :=
  rewrite ?andb_true_iff; intros;
  repeat match goal with H : _ ∧ _ |- _ => destruct H end;
  repeat match goal with |- _ ∧ _ => split end;
  match goal with
  | |- true = true => reflexivity
  | H : equivalence_at ?e ?x |- ?e ?x ?x = true => exact (proj1 H)
  | H : equivalence_at ?e ?x, E : ?e ?x ?y = true |- ?e ?y ?x = true =>
    exact (proj1 (proj2 H) y E)
  | H : equivalence_at ?e ?x, E : ?e ?x ?y = true, E' : ?e ?y ?z = true
    |- ?e ?x ?z = true =>
    exact (proj2 (proj2 H) y z E E')
  end.

Ltac absurd_parts :=
  try (intros; match goal with H : false = true |- _ => discriminate H end).

(* For a structure with constructors: the other structures are taken
   apart, and those of another constructor are not the same. *)
Ltac constructor_equivalence :=
  split; [ simpl; equivalence_parts | split ];
  [ intros a'; destruct a'; simpl; absurd_parts; equivalence_parts
  | intros a' a''; destruct a', a''; simpl; absurd_parts; equivalence_parts ].

(* For a record, read by its fields. *)
Ltac record_equivalence :=
  split; [ equivalence_parts | split ];
  [ intros a'; equivalence_parts | intros a' a''; equivalence_parts ].

Lemma block_case_equivalence_at (case : row_like_block_case) :
  P_block_case (equivalence_at type_eqb) (equivalence_at env_extension_eqb)
    case ->
  equivalence_at (block_case_eqb type_eqb env_extension_eqb) case.
Proof.
  intros [Htypes Hextension].
  pose proof
    (decides_equality_equivalence_at block_index_eqb (case_index case)
       block_index_eqb_eq)
    as Hindex.
  pose proof (lists_equivalence_at type_eqb (case_maps_to case) Htypes)
    as Hfields.
  unfold block_case_eqb. record_equivalence.
Qed.

Lemma closures_index_equivalence_at
    (index : row_like_index set_of_closures_contents unit) :
  equivalence_at closures_index_eqb index.
Proof.
  assert (Hslots :
    ∀ slots : list positive,
      equivalence_at (Keyed.same_keys_b Pos.eqb) slots).
  { intros slots. exact (same_keys_equivalence_at Pos.eqb slots Pos.eqb_eq). }
  unfold closures_index_eqb. split; [ | split ].
  - destruct (index_domain index) as [contents | contents];
      pose proof (Hslots (contents_closures contents));
      pose proof (Hslots (contents_value_slots contents)); equivalence_parts.
  - intros index'.
    destruct (index_domain index) as [contents | contents],
      (index_domain index') as [contents' | contents']; try discriminate;
      pose proof (Hslots (contents_closures contents));
      pose proof (Hslots (contents_value_slots contents)); equivalence_parts.
  - intros index' index''.
    destruct (index_domain index) as [contents | contents],
      (index_domain index') as [contents' | contents'],
      (index_domain index'') as [contents'' | contents''];
      absurd_parts;
      pose proof (Hslots (contents_closures contents));
      pose proof (Hslots (contents_value_slots contents)); equivalence_parts.
Qed.

Lemma closures_case_equivalence_at (case : row_like_closures_case) :
  P_closures_case (equivalence_at closures_entry_eqb)
    (equivalence_at env_extension_eqb) case ->
  equivalence_at (closures_case_eqb closures_entry_eqb env_extension_eqb)
    case.
Proof.
  intros [Hentry Hextension].
  pose proof (closures_index_equivalence_at (case_index case)) as Hindex.
  unfold closures_case_eqb. record_equivalence.
Qed.

Lemma head_of_kind_naked_immediate_equivalence_at
    (head : head_of_kind_naked_immediate) :
  equivalence_at head_of_kind_naked_immediate_eqb head.
Proof.
  assert (Hnumbers :
    equivalence_at (or_unknown_eqb (Keyed.same_keys_b Z.eqb))
      (naked_immediates head)).
  { apply or_unknown_equivalence_at. intros numbers _.
    exact (same_keys_equivalence_at Z.eqb numbers Z.eqb_eq). }
  pose proof
    (same_keys_equivalence_at _ (inverse_relations head)
       relation_and_name_eqb_eq)
    as Hrelations.
  unfold head_of_kind_naked_immediate_eqb. record_equivalence.
Qed.

Lemma variant_extensions_equivalence_at (extensions : variant_extensions) :
  match extensions with
  | No_extensions => True
  | Ext when_immediate when_block =>
    equivalence_at env_extension_eqb when_immediate
    ∧ equivalence_at env_extension_eqb when_block
  end ->
  equivalence_at (variant_extensions_eqb env_extension_eqb) extensions.
Proof.
  destruct extensions as [ | when_immediate when_block].
  - intros _. constructor_equivalence.
  - intros [Himmediate Hblock]. constructor_equivalence.
Qed.

Lemma type_eqb_equivalence_all :
  (∀ t : flambda_type, equivalence_at type_eqb t)
  ∧ (∀ head : head_of_kind_value, equivalence_at head_of_kind_value_eqb head)
  ∧ (∀ head : head_of_kind_value_non_null,
      equivalence_at head_of_kind_value_non_null_eqb head)
  ∧ (∀ blocks : row_like_for_blocks,
      equivalence_at row_like_for_blocks_eqb blocks)
  ∧ (∀ closures : row_like_for_closures,
      equivalence_at row_like_for_closures_eqb closures)
  ∧ (∀ entry : closures_entry, equivalence_at closures_entry_eqb entry)
  ∧ (∀ extension : env_extension,
      equivalence_at env_extension_eqb extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead.
    assert (Hty : equivalence_at (descr_eqb head_of_kind_value_eqb) ty).
    { apply descr_equivalence_at. intros h ->. exact Hhead. }
    constructor_equivalence.
  - (* Type_naked_immediate *)
    intros ty.
    assert (Hty :
      equivalence_at (descr_eqb head_of_kind_naked_immediate_eqb) ty).
    { apply descr_equivalence_at. intros h _.
      apply head_of_kind_naked_immediate_equivalence_at. }
    constructor_equivalence.
  - (* Type_naked_number *)
    intros k ty.
    pose proof
      (decides_equality_equivalence_at naked_number_kind_eqb k
         naked_number_kind_eqb_eq)
      as Hk.
    assert (Hty : equivalence_at (descr_eqb (Keyed.same_keys_b Z.eqb)) ty).
    { apply descr_equivalence_at. intros h _.
      exact (same_keys_equivalence_at Z.eqb h Z.eqb_eq). }
    constructor_equivalence.
  - (* Type_rec_info *)
    intros ty.
    assert (Hty : equivalence_at (descr_eqb (fun _ _ : unit => true)) ty).
    { apply descr_equivalence_at. intros h _. apply always_equivalence_at. }
    constructor_equivalence.
  - (* Type_region *)
    intros ty.
    assert (Hty : equivalence_at (descr_eqb (fun _ _ : unit => true)) ty).
    { apply descr_equivalence_at. intros h _. apply always_equivalence_at. }
    constructor_equivalence.
  - (* Head_of_kind_value *)
    intros non_null is_null Hhead.
    assert (Hnon_null :
      equivalence_at
        (or_unknown_or_bottom_eqb head_of_kind_value_non_null_eqb) non_null).
    { apply or_unknown_or_bottom_equivalence_at. intros h ->. exact Hhead. }
    pose proof
      (decides_equality_equivalence_at is_null_eqb is_null is_null_eqb_eq)
      as His_null.
    constructor_equivalence.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions.
    pose proof
      (decides_equality_equivalence_at option_variable_eqb is_int
         option_variable_eqb_eq)
      as His_int.
    assert (Himm : equivalence_at (or_unknown_eqb type_eqb) immediates).
    { apply or_unknown_equivalence_at. intros ty ->. exact Himmediates. }
    pose proof
      (decides_equality_equivalence_at option_variable_eqb get_tag
         option_variable_eqb_eq)
      as Hget_tag.
    assert (Hrow :
      equivalence_at (or_unknown_eqb row_like_for_blocks_eqb) blocks).
    { apply or_unknown_equivalence_at. intros row ->. exact Hblocks. }
    pose proof (variant_extensions_equivalence_at extensions Hextensions)
      as Hext.
    pose proof
      (decides_equality_equivalence_at Bool.eqb is_unique eqb_true_iff)
      as His_unique.
    constructor_equivalence.
  - (* Mutable_block *)
    intros alloc_mode.
    pose proof
      (decides_equality_equivalence_at alloc_mode_eqb alloc_mode
         alloc_mode_eqb_eq)
      as Hmode.
    constructor_equivalence.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents.
    pose proof
      (decides_equality_equivalence_at boxable_number_eqb k
         boxable_number_eqb_eq)
      as Hk.
    pose proof
      (decides_equality_equivalence_at alloc_mode_eqb alloc_mode
         alloc_mode_eqb_eq)
      as Hmode.
    constructor_equivalence.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures.
    pose proof
      (decides_equality_equivalence_at alloc_mode_eqb alloc_mode
         alloc_mode_eqb_eq)
      as Hmode.
    constructor_equivalence.
  - (* String *)
    intros strs.
    pose proof (same_keys_equivalence_at _ strs bytes_eqb_eq) as Hstrs.
    constructor_equivalence.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother.
    assert (Hknown_tags :
      equivalence_at
        (map_eqb Z.eqb
           (or_unknown_eqb (block_case_eqb type_eqb env_extension_eqb)))
        known_tags).
    { apply (map_equivalence_at Z.eqb Z.eqb_eq). intros tag case Hin.
      apply or_unknown_equivalence_at. intros case' ->.
      apply block_case_equivalence_at. exact (Hknown tag case' Hin). }
    assert (Hother_tags :
      equivalence_at
        (or_bottom_eqb (block_case_eqb type_eqb env_extension_eqb))
        other_tags).
    { apply or_bottom_equivalence_at. intros case ->.
      apply block_case_equivalence_at. exact Hother. }
    pose proof
      (decides_equality_equivalence_at alloc_mode_eqb alloc_mode
         alloc_mode_eqb_eq)
      as Hmode.
    constructor_equivalence.
  - (* Row_like_for_closures *)
    intros known_closures Hknown.
    assert (Hknown_closures :
      equivalence_at
        (map_eqb Pos.eqb
           (closures_case_eqb closures_entry_eqb env_extension_eqb))
        known_closures).
    { apply (map_equivalence_at Pos.eqb Pos.eqb_eq). intros slot case Hin.
      apply closures_case_equivalence_at. exact (Hknown slot case Hin). }
    constructor_equivalence.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots.
    assert (Hfunction_types :
      equivalence_at (map_eqb Pos.eqb (or_unknown_eqb function_type_eqb))
        function_types).
    { apply (map_equivalence_at Pos.eqb Pos.eqb_eq). intros g ty _.
      apply decides_equality_equivalence_at, or_unknown_eqb_eq,
        function_type_eqb_eq. }
    pose proof
      (map_equivalence_at Pos.eqb Pos.eqb_eq type_eqb closure_types Hclosures)
      as Hclosure_types.
    pose proof
      (map_equivalence_at Pos.eqb Pos.eqb_eq type_eqb value_slot_types Hslots)
      as Hvalue_slot_types.
    constructor_equivalence.
  - (* Env_extension *)
    intros equations Htypes.
    pose proof
      (map_equivalence_at name_eqb name_eqb_eq type_eqb equations Htypes)
      as Hequations.
    constructor_equivalence.
Qed.

Theorem type_equal_equivalence : type_equal_equivalence_statement.
Proof.
  destruct type_eqb_equivalence_all as [Htype _]. unfold type_equal.
  split; [ intros t; exact (proj1 (Htype t)) | ]. split.
  - intros t t'. exact (proj1 (proj2 (Htype t)) t').
  - intros t t' t''. exact (proj2 (proj2 (Htype t)) t' t'').
Qed.

Theorem env_extension_equal_equivalence
  : env_extension_equal_equivalence_statement.
Proof.
  destruct type_eqb_equivalence_all as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
  unfold env_extension_equal.
  split; [ intros extension; exact (proj1 (Hextension extension)) | ]. split.
  - intros extension extension'.
    exact (proj1 (proj2 (Hextension extension)) extension').
  - intros extension extension' extension''.
    exact (proj2 (proj2 (Hextension extension)) extension' extension'').
Qed.

(* ======================================================================== *)
(* The same meaning                                                         *)
(* ======================================================================== *)

Section Same_meaning.

Variable statics : static_env.
Variable vars : env.

Lemma descr_eqb_satisfied {head : Type} (k : flambda_kind)
    (head_eqb : head -> head -> bool) (head_satisfied : head -> Prop)
    (v : value) (ty ty' : type_descr head) :
  (∀ h h' : head,
    ty = Ok (No_alias h) -> head_eqb h h' = true ->
    (head_satisfied h <-> head_satisfied h')) ->
  descr_eqb head_eqb ty ty' = true ->
  (descr_satisfied vars k head_satisfied v ty
   <-> descr_satisfied vars k head_satisfied v ty').
Proof.
  intros Hhead.
  destruct ty as [ | [h | s] | ], ty' as [ | [h' | s'] | ]; simpl;
    try discriminate; try (intros _; reflexivity).
  - intros Heq. exact (Hhead h h' eq_refl Heq).
  - intros Heq. apply simple_eqb_eq in Heq. subst s'. reflexivity.
Qed.

Lemma head_of_kind_naked_immediate_eqb_satisfied (v : value)
    (head head' : head_of_kind_naked_immediate) :
  head_of_kind_naked_immediate_eqb head head' = true ->
  (head_of_kind_naked_immediate_satisfied statics vars v head
   <-> head_of_kind_naked_immediate_satisfied statics vars v head').
Proof.
  unfold head_of_kind_naked_immediate_eqb,
    head_of_kind_naked_immediate_satisfied.
  rewrite andb_true_iff. intros [Hnumbers Hrelations].
  pose proof (same_members _ _ _ relation_and_name_eqb_eq Hrelations)
    as Hsame.
  assert (Hnumber :
    ∀ i : Z,
      match naked_immediates head with
      | Or_unknown.Unknown => True
      | Or_unknown.Known numbers => i ∈ numbers
      end
      <-> match naked_immediates head' with
          | Or_unknown.Unknown => True
          | Or_unknown.Known numbers => i ∈ numbers
          end).
  { intros i.
    destruct (naked_immediates head) as [numbers | ],
      (naked_immediates head') as [numbers' | ]; simpl in Hnumbers;
      try discriminate Hnumbers; [ | reflexivity ].
    exact (same_members Z.eqb _ _ Z.eqb_eq Hnumbers i). }
  split; intros [i [Hv [Hi Hrelation]]]; exists i;
    (split; [ exact Hv | ]); (split; [ apply (Hnumber i); exact Hi | ]);
    intros relation n Hin; apply Hrelation; apply (Hsame (relation, n));
    exact Hin.
Qed.

Lemma naked_numbers_eqb_satisfied (k : naked_number_kind) (v : value)
    (numbers numbers' : list Z) :
  Keyed.same_keys_b Z.eqb numbers numbers' = true ->
  (naked_numbers_satisfied k v numbers
   <-> naked_numbers_satisfied k v numbers').
Proof.
  intros Hsame. pose proof (same_members Z.eqb _ _ Z.eqb_eq Hsame) as Hin.
  unfold naked_numbers_satisfied.
  split; intros [c [n [Hv [Hkind [Hnumber Hn]]]]]; exists c, n;
    (split; [ exact Hv | ]); (split; [ exact Hkind | ]);
    (split; [ exact Hnumber | ]); apply (Hin n); exact Hn.
Qed.

Lemma all_among_same (names names' allowed : list positive) :
  same_names names names' ->
  (FlambdaMapSpec.all_among names allowed = true
   <-> FlambdaMapSpec.all_among names' allowed = true).
Proof.
  intros Hsame.
  split; intros Hall; apply FlambdaMapProofs.all_among; intros name Hname;
    apply (proj1 (FlambdaMapProofs.all_among _ _) Hall);
    [ rewrite (Hsame name) | rewrite <- (Hsame name) ]; exact Hname.
Qed.

Lemma same_names_same (names names' other : list positive) :
  same_names names names' ->
  (same_names names other <-> same_names names' other).
Proof.
  intros Hsame.
  split; intros Hother name;
    [ rewrite <- (Hsame name) | rewrite (Hsame name) ]; apply Hother.
Qed.

Lemma closures_index_eqb_satisfied
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (index index' : row_like_index set_of_closures_contents unit) :
  closures_index_eqb index index' = true ->
  (closures_index_satisfied functions value_slots index
   <-> closures_index_satisfied functions value_slots index').
Proof.
  unfold closures_index_eqb, closures_index_satisfied.
  destruct (index_domain index) as [contents | contents],
    (index_domain index') as [contents' | contents']; try discriminate;
    rewrite andb_true_iff,
      !(keyed_same_keys_b positive Pos.eqb Pos.eqb_eq);
    intros [Hclosures Hslots].
  - rewrite (same_names_same _ _ _ Hclosures),
      (same_names_same _ _ _ Hslots).
    reflexivity.
  - rewrite (all_among_same _ _ _ Hclosures), (all_among_same _ _ _ Hslots).
    reflexivity.
Qed.

(* What an induction on types knows of a type, of an environment
   extension, and of the two together in a case of blocks. *)
Let same_type (t : flambda_type) : Prop :=
  ∀ (t' : flambda_type) (v : value),
    type_eqb t t' = true ->
    (type_satisfied statics vars v t <-> type_satisfied statics vars v t').

Let same_extension (extension : env_extension) : Prop :=
  ∀ extension' : env_extension,
    env_extension_eqb extension extension' = true ->
    (env_extension_satisfied statics vars extension
     <-> env_extension_satisfied statics vars extension').

Lemma fields_eqb_satisfied (tys tys' : list flambda_type)
    (fields : list value) :
  (∀ ty : flambda_type, ty ∈ tys -> same_type ty) ->
  lists_eqb type_eqb tys tys' = true ->
  (fields_satisfied (type_satisfied statics vars) tys fields
   <-> fields_satisfied (type_satisfied statics vars) tys' fields).
Proof.
  revert tys' fields.
  induction tys as [ | ty tys IH]; intros [ | ty' tys'] fields Htypes;
    simpl; try discriminate; [ intros _; reflexivity | ].
  rewrite andb_true_iff. intros [Hty Htys].
  destruct fields as [ | field fields]; [ reflexivity | ].
  rewrite (Htypes ty (or_introl eq_refl) ty' field Hty),
    (IH tys' fields (fun ty Hin => Htypes ty (or_intror Hin)) Htys).
  reflexivity.
Qed.

Lemma block_case_eqb_satisfied (tag : Z) (fields : list value)
    (case case' : row_like_block_case) :
  P_block_case same_type same_extension case ->
  block_case_eqb type_eqb env_extension_eqb case case' = true ->
  (row_like_block_case_satisfied statics vars tag fields case
   <-> row_like_block_case_satisfied statics vars tag fields case').
Proof.
  intros [Htypes Hextension]. unfold block_case_eqb.
  rewrite !andb_true_iff. intros [[Hindex Hfields] Hext].
  apply block_index_eqb_eq in Hindex.
  unfold row_like_block_case_satisfied, block_case_satisfied.
  rewrite <- Hindex, (fields_eqb_satisfied _ _ fields Htypes Hfields),
    (Hextension _ Hext).
  reflexivity.
Qed.

Lemma type_equal_satisfied_all :
  (∀ t : flambda_type, same_type t)
  ∧ (∀ (head head' : head_of_kind_value) (v : value),
      head_of_kind_value_eqb head head' = true ->
      (head_of_kind_value_satisfied statics vars v head
       <-> head_of_kind_value_satisfied statics vars v head'))
  ∧ (∀ (head head' : head_of_kind_value_non_null) (v : value),
      head_of_kind_value_non_null_eqb head head' = true ->
      (head_of_kind_value_non_null_satisfied statics vars v head
       <-> head_of_kind_value_non_null_satisfied statics vars v head'))
  ∧ (∀ (blocks blocks' : row_like_for_blocks) (tag : Z)
       (fields : list value),
      row_like_for_blocks_eqb blocks blocks' = true ->
      (row_like_for_blocks_satisfied statics vars tag fields blocks
       <-> row_like_for_blocks_satisfied statics vars tag fields blocks'))
  ∧ (∀ (closures closures' : row_like_for_closures) (v : value)
       (slot : function_slot) (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      row_like_for_closures_eqb closures closures' = true ->
      (row_like_for_closures_satisfied statics vars v slot functions
         value_slots closures
       <-> row_like_for_closures_satisfied statics vars v slot functions
             value_slots closures'))
  ∧ (∀ (entry entry' : closures_entry) (v : value) (slot : function_slot)
       (functions : list (function_slot * code_id))
       (value_slots : list (value_slot * value)),
      closures_entry_eqb entry entry' = true ->
      (closures_entry_satisfied statics vars v slot functions value_slots
         entry
       <-> closures_entry_satisfied statics vars v slot functions
             value_slots entry'))
  ∧ (∀ extension : env_extension, same_extension extension).
Proof.
  apply types_induction.
  - (* Type_value *)
    intros ty Hhead [ty' | ty' | k' ty' | ty' | ty'] v; simpl;
      try discriminate.
    autorewrite with type_satisfied. apply descr_eqb_satisfied.
    intros h h' -> Heq. exact (Hhead h' v Heq).
  - (* Type_naked_immediate *)
    intros ty [ty' | ty' | k' ty' | ty' | ty'] v; simpl; try discriminate.
    autorewrite with type_satisfied. apply descr_eqb_satisfied.
    intros h h' _. apply head_of_kind_naked_immediate_eqb_satisfied.
  - (* Type_naked_number *)
    intros k ty [ty' | ty' | k' ty' | ty' | ty'] v; simpl; try discriminate.
    rewrite andb_true_iff, (naked_number_kind_eqb_eq _ _).
    intros [<- Heq]. autorewrite with type_satisfied.
    revert Heq. apply descr_eqb_satisfied.
    intros h h' _. apply naked_numbers_eqb_satisfied.
  - (* Type_rec_info *)
    intros ty [ty' | ty' | k' ty' | ty' | ty'] v; simpl; try discriminate.
    autorewrite with type_satisfied. apply descr_eqb_satisfied.
    intros h h' _ _. reflexivity.
  - (* Type_region *)
    intros ty [ty' | ty' | k' ty' | ty' | ty'] v; simpl; try discriminate.
    autorewrite with type_satisfied. apply descr_eqb_satisfied.
    intros h h' _ _. reflexivity.
  - (* Head_of_kind_value *)
    intros non_null is_null Hnon_null [non_null' is_null'] v. simpl.
    rewrite andb_true_iff, (is_null_eqb_eq _ _). intros [Heq <-].
    autorewrite with type_satisfied.
    destruct non_null as [ | head | ], non_null' as [ | head' | ];
      try discriminate Heq; [ reflexivity | | reflexivity ].
    rewrite (Hnon_null head' v Heq). reflexivity.
  - (* Variant *)
    intros is_int immediates get_tag blocks extensions is_unique Himmediates
      Hblocks Hextensions
      [is_int' immediates' get_tag' blocks' extensions' is_unique'
      | alloc_mode' | k' contents' alloc_mode' | by_function_slot' alloc_mode'
      | strs'] v; simpl; try discriminate.
    rewrite !andb_true_iff, !(option_variable_eqb_eq _ _).
    intros [[[[[<- Himm] <-] Hrow] Hext] _].
    assert (Himmediate :
      ∀ i : Z,
        match immediates with
        | Or_unknown.Unknown => True
        | Or_unknown.Known ty =>
          type_satisfied statics vars (Value_const (Naked_immediate i)) ty
        end
        <-> match immediates' with
            | Or_unknown.Unknown => True
            | Or_unknown.Known ty =>
              type_satisfied statics vars (Value_const (Naked_immediate i))
                ty
            end).
    { intros i.
      destruct immediates as [ty | ], immediates' as [ty' | ];
        try discriminate Himm; [ | reflexivity ].
      exact (Himmediates ty' _ Himm). }
    assert (Hblock :
      ∀ (tag : Z) (fields : list value),
        match blocks with
        | Or_unknown.Unknown => True
        | Or_unknown.Known blocks =>
          row_like_for_blocks_satisfied statics vars tag fields blocks
        end
        <-> match blocks' with
            | Or_unknown.Unknown => True
            | Or_unknown.Known blocks =>
              row_like_for_blocks_satisfied statics vars tag fields blocks
            end).
    { intros tag fields.
      destruct blocks as [row | ], blocks' as [row' | ];
        try discriminate Hrow; [ | reflexivity ].
      exact (Hblocks row' tag fields Hrow). }
    assert (Hwhen :
      (match extensions with
       | No_extensions => True
       | Ext when_immediate _ =>
         env_extension_satisfied statics vars when_immediate
       end
       <-> match extensions' with
           | No_extensions => True
           | Ext when_immediate _ =>
             env_extension_satisfied statics vars when_immediate
           end)
      ∧ (match extensions with
         | No_extensions => True
         | Ext _ when_block => env_extension_satisfied statics vars when_block
         end
         <-> match extensions' with
             | No_extensions => True
             | Ext _ when_block =>
               env_extension_satisfied statics vars when_block
             end)).
    { destruct extensions as [ | when_immediate when_block],
        extensions' as [ | when_immediate' when_block'];
        try discriminate Hext; [ split; reflexivity | ].
      apply andb_true_iff in Hext as [Hext_immediate Hext_block].
      destruct Hextensions as [IHimmediate IHblock].
      split; [ exact (IHimmediate _ Hext_immediate) | ].
      exact (IHblock _ Hext_block). }
    destruct Hwhen as [Hwhen_immediate Hwhen_block].
    rewrite !variant_satisfied.
    split;
      (intros [[i [Hv [His_int [Hi Hextension]]]]
              | [tag [fields [Hv [His_int [Hget_tag [Hfields Hextension]]]]]]];
       [ left; exists i; split; [ exact Hv | ]; split; [ exact His_int | ];
         split; [ apply (Himmediate i); exact Hi | ];
         apply Hwhen_immediate; exact Hextension
       | right; exists tag, fields; split; [ exact Hv | ];
         split; [ exact His_int | ]; split; [ exact Hget_tag | ];
         split; [ apply (Hblock tag fields); exact Hfields | ];
         apply Hwhen_block; exact Hextension ]).
  - (* Mutable_block *)
    intros alloc_mode
      [is_int' immediates' get_tag' blocks' extensions' is_unique'
      | alloc_mode' | k' contents' alloc_mode' | by_function_slot' alloc_mode'
      | strs'] v; simpl; try discriminate.
    intros _. autorewrite with type_satisfied. reflexivity.
  - (* Boxed_number *)
    intros k contents alloc_mode Hcontents
      [is_int' immediates' get_tag' blocks' extensions' is_unique'
      | alloc_mode' | k' contents' alloc_mode' | by_function_slot' alloc_mode'
      | strs'] v; simpl; try discriminate.
    rewrite !andb_true_iff, (boxable_number_eqb_eq _ _).
    intros [[<- Heq] _]. rewrite !boxed_number_satisfied.
    split; intros [c [Hboxed [Hkind Hc]]]; exists c;
      (split; [ exact Hboxed | ]); (split; [ exact Hkind | ]);
      apply (Hcontents contents' (Value_const c) Heq); exact Hc.
  - (* Closures *)
    intros by_function_slot alloc_mode Hclosures
      [is_int' immediates' get_tag' blocks' extensions' is_unique'
      | alloc_mode' | k' contents' alloc_mode' | by_function_slot' alloc_mode'
      | strs'] v; simpl; try discriminate.
    rewrite andb_true_iff. intros [Heq _]. rewrite !closures_satisfied.
    split; intros [slot [functions [value_slots [Hclosure Hrow]]]];
      exists slot, functions, value_slots; (split; [ exact Hclosure | ]);
      apply (Hclosures by_function_slot' v slot functions value_slots Heq);
      exact Hrow.
  - (* String *)
    intros strs
      [is_int' immediates' get_tag' blocks' extensions' is_unique'
      | alloc_mode' | k' contents' alloc_mode' | by_function_slot' alloc_mode'
      | strs'] v; simpl; try discriminate.
    intros Hsame. rewrite !string_satisfied.
    pose proof (same_members _ _ _ bytes_eqb_eq Hsame) as Hin.
    split; intros [bytes [Hstring Hbytes]]; exists bytes;
      (split; [ exact Hstring | ]); apply (Hin bytes); exact Hbytes.
  - (* Row_like_for_blocks *)
    intros known_tags other_tags alloc_mode Hknown Hother
      [known_tags' other_tags' alloc_mode'] tag fields. simpl.
    rewrite !andb_true_iff. intros [[Hmap Hother_tags] _].
    rewrite !row_like_for_blocks_find_iff.
    pose proof (map_eqb_lookup Z.eqb Z.eqb_eq _ _ _ tag Hmap) as Hat.
    unfold row_like_block_case in *.
    destruct (Keyed.lookup Z.eqb tag known_tags) as [[case | ] | ],
      (Keyed.lookup Z.eqb tag known_tags') as [[case' | ] | ];
      [ destruct Hat as [Hin [_ Heq]]
      | destruct Hat as [_ [_ Heq]]; discriminate Heq
      | destruct Hat
      | destruct Hat as [_ [_ Heq]]; discriminate Heq
      | reflexivity
      | destruct Hat
      | destruct Hat
      | destruct Hat
      | ].
    + exact (block_case_eqb_satisfied tag fields _ _ (Hknown tag case Hin) Heq).
    + destruct other_tags as [case | ], other_tags' as [case' | ];
        try discriminate Hother_tags; [ | reflexivity ].
      exact (block_case_eqb_satisfied tag fields _ _ Hother Hother_tags).
  - (* Row_like_for_closures *)
    intros known_closures Hknown [known_closures'] v slot functions
      value_slots. simpl. intros Hmap.
    rewrite !row_like_for_closures_lookup_iff.
    pose proof (map_eqb_lookup Pos.eqb Pos.eqb_eq _ _ _ slot Hmap) as Hat.
    change (Keyed.lookup Pos.eqb slot known_closures)
      with (lookup slot known_closures) in Hat.
    change (Keyed.lookup Pos.eqb slot known_closures')
      with (lookup slot known_closures') in Hat.
    unfold row_like_closures_case in *.
    destruct (lookup slot known_closures) as [case | ],
      (lookup slot known_closures') as [case' | ];
      try (destruct Hat; fail).
    + destruct Hat as [Hin [_ Heq]].
      apply andb_true_iff in Heq as [Heq Hext].
      apply andb_true_iff in Heq as [Hindex Hentry].
      destruct (Hknown slot case Hin) as [IHentry IHextension].
      assert (Hcase :
        row_like_closures_case_satisfied statics vars v slot functions
          value_slots case
        <-> row_like_closures_case_satisfied statics vars v slot functions
              value_slots case').
      { unfold row_like_closures_case_satisfied, closures_case_satisfied.
        rewrite (closures_index_eqb_satisfied _ _ _ _ Hindex),
          (IHentry _ v slot functions value_slots Hentry),
          (IHextension _ Hext).
        reflexivity. }
      split; intros [found [[= <-] Hfound]];
        [ exists case' | exists case ];
        (split; [ reflexivity | ]); apply Hcase; exact Hfound.
    + split; intros [found [Hfound _]]; discriminate Hfound.
  - (* Closures_entry *)
    intros function_types closure_types value_slot_types Hclosures Hslots
      [function_types' closure_types' value_slot_types'] v slot functions
      value_slots. simpl. rewrite !andb_true_iff.
    intros [[Hfunction_types Hclosure_types] Hvalue_slot_types].
    rewrite !closures_entry_lookup_iff.
    pose proof
      (fun g =>
         map_eqb_same_lookup Pos.eqb Pos.eqb_eq _
           (or_unknown_eqb_eq _ function_type_eqb_eq) _ _ g Hfunction_types)
      as Hfunctions.
    pose proof
      (map_eqb_for_all Pos.eqb Pos.eqb_eq type_eqb
         (fun g ty =>
            ∀ closure : value,
              project_function_slot statics slot g v = Some closure ->
              type_satisfied statics vars closure ty)
         _ _ Hclosure_types
         (fun g ty ty' Hin Heq =>
            conj
              (fun H closure Hproject =>
                 proj1 (Hclosures g ty Hin ty' closure Heq)
                   (H closure Hproject))
              (fun H closure Hproject =>
                 proj2 (Hclosures g ty Hin ty' closure Heq)
                   (H closure Hproject))))
      as Hclosures_iff.
    pose proof
      (map_eqb_for_all Pos.eqb Pos.eqb_eq type_eqb
         (fun w ty =>
            ∀ contents : value,
              lookup w value_slots = Some contents ->
              type_satisfied statics vars contents ty)
         _ _ Hvalue_slot_types
         (fun w ty ty' Hin Heq =>
            conj
              (fun H contents Hcontents =>
                 proj1 (Hslots w ty Hin ty' contents Heq)
                   (H contents Hcontents))
              (fun H contents Hcontents =>
                 proj2 (Hslots w ty Hin ty' contents Heq)
                   (H contents Hcontents))))
      as Hslots_iff.
    split; intros [Hcode [Hclosure Hvalue_slot]].
    + split; [ | split ].
      * intros g function_type code Hlookup.
        apply Hcode. change (lookup g function_types) with
          (Keyed.lookup Pos.eqb g function_types).
        rewrite (Hfunctions g). exact Hlookup.
      * intros g ty closure Hlookup. revert closure.
        exact
          (proj1 Hclosures_iff
             (fun g ty Hlookup closure => Hclosure g ty closure Hlookup) g ty
             Hlookup).
      * intros w ty contents Hlookup. revert contents.
        exact
          (proj1 Hslots_iff
             (fun w ty Hlookup contents => Hvalue_slot w ty contents Hlookup)
             w ty Hlookup).
    + split; [ | split ].
      * intros g function_type code Hlookup.
        apply Hcode. change (lookup g function_types') with
          (Keyed.lookup Pos.eqb g function_types').
        rewrite <- (Hfunctions g). exact Hlookup.
      * intros g ty closure Hlookup. revert closure.
        exact
          (proj2 Hclosures_iff
             (fun g ty Hlookup closure => Hclosure g ty closure Hlookup) g ty
             Hlookup).
      * intros w ty contents Hlookup. revert contents.
        exact
          (proj2 Hslots_iff
             (fun w ty Hlookup contents => Hvalue_slot w ty contents Hlookup)
             w ty Hlookup).
  - (* Env_extension *)
    intros equations Htypes [equations']. simpl. intros Hmap.
    rewrite !env_extension_lookup_iff.
    apply
      (map_eqb_for_all name_eqb name_eqb_eq type_eqb
         (fun n ty =>
            ∃ w : value,
              value_of_name vars n = Some w
              ∧ type_satisfied statics vars w ty)
         _ _ Hmap).
    intros n ty ty' Hin Heq.
    split; intros [w [Hw Hty]]; exists w; (split; [ exact Hw | ]);
      apply (Htypes n ty Hin ty' w Heq); exact Hty.
Qed.

End Same_meaning.

Theorem type_equal_satisfied : type_equal_satisfied_statement.
Proof.
  intros statics vars v t t' Hequal.
  exact (proj1 (type_equal_satisfied_all statics vars) t t' v Hequal).
Qed.

Theorem env_extension_equal_satisfied
  : env_extension_equal_satisfied_statement.
Proof.
  intros statics vars extension extension' Hequal.
  destruct (type_equal_satisfied_all statics vars)
    as [_ [_ [_ [_ [_ [_ Hextension]]]]]].
  exact (Hextension extension extension' Hequal).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* What the test finds the same and what it does not. They hold by
   computation. *)
Module Equal_examples.

Definition x : name := Name_var 1%positive.
Definition y : name := Name_var 2%positive.

(* A map is compared as a map: the order of its bindings is not looked
   at, and a binding that is shadowed is not either. *)
Example the_order_of_bindings :
  env_extension_equal
    (Env_extension [(x, any_tagged_immediate); (y, any_block)])
    (Env_extension [(y, any_block); (x, any_tagged_immediate)]).
Proof. reflexivity. Qed.

Example a_shadowed_binding :
  env_extension_equal
    (Env_extension [(x, any_tagged_immediate); (x, any_block)])
    (Env_extension [(x, any_tagged_immediate)]).
Proof. reflexivity. Qed.

Example one_equation_more :
  ~ env_extension_equal
      (Env_extension [(x, any_tagged_immediate); (y, any_block)])
      (Env_extension [(x, any_tagged_immediate)]).
Proof. discriminate. Qed.

(* A set is compared as a set. *)
Example the_same_numbers :
  type_equal (these_naked_immediates [1; 2; 2]%Z)
    (these_naked_immediates [2; 1]%Z).
Proof. reflexivity. Qed.

(* An alias is the same as an alias to the same simple only: not as a
   type that has the same values, nor as an alias to another name. *)
Example an_alias_and_what_it_means :
  ~ type_equal (this_tagged_immediate 3)
      (tag_immediate
         (Type_naked_immediate
            (Ok
               (No_alias
                  (Head_of_kind_naked_immediate (Or_unknown.Known [3%Z])
                     []))))).
Proof. discriminate. Qed.

Example two_aliases :
  ~ type_equal (alias_type_of Value (simple_of_name x))
      (alias_type_of Value (simple_of_name y)).
Proof. discriminate. Qed.

(* No extensions and two empty extensions are told apart, as are two
   allocation modes. *)
Example no_extensions_and_empty_extensions :
  ~ type_equal
      (create_variant false Or_unknown.Unknown Or_unknown.Unknown
         No_extensions)
      (create_variant false Or_unknown.Unknown Or_unknown.Unknown
         (Ext empty_env_extension empty_env_extension)).
Proof. discriminate. Qed.

Example two_allocation_modes :
  ~ type_equal (mutable_block Alloc_mode.Heap)
      (mutable_block Alloc_mode.Heap_or_local).
Proof. discriminate. Qed.

End Equal_examples.
