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

(* Proofs of the statements of FlambdaMeetClosuresSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia Permutation Setoid.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresSpec FlambdaClosuresProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetScopesSpec FlambdaMeetScopesProofs
  FlambdaMeetProductsSpec FlambdaMeetProductsProofs FlambdaMeetHeadsSpec
  FlambdaMeetHeadsProofs FlambdaMeetRowLikeSpec FlambdaMeetRowLikeProofs
  FlambdaMeetImmediateSpec FlambdaMeetImmediateProofs FlambdaMeetFuelSpec
  FlambdaMeetFuelProofs FlambdaMeetClosuresSpec.
Import ListNotations.

(* The proofs below, as far as the end of the section, are for any
   predicate on static environments that implies validity
   (FlambdaMeetValidity.v): in them the names of the four contracts
   that suppose a valid static environment stand for the same with
   that predicate. The theorems of the signed statements are after the
   section, at validity itself. *)
From OxCamlFlambda2Types Require Import FlambdaMeetValidity.

Section Validity.

Context {V : validity}.

Abbreviation statics_valid := valid.
Abbreviation meet_expanded_head_sound := meet_expanded_head_sound_for.
Abbreviation join_scoped_envs_sound := join_scoped_envs_sound_for.
Abbreviation meet_call_ok := meet_call_ok_for.
Abbreviation part_meet_ok := part_meet_ok_for.
Abbreviation meet_disjunction_right := meet_disjunction_right_for.
Abbreviation type_part_meet_ok := type_part_meet_ok_for.
Abbreviation meet_row_like_for_closures_right :=
  meet_row_like_for_closures_right_for.
Abbreviation value_heads_right := value_heads_right_for.

(* ======================================================================== *)
(* What the indexes say that a set of closures has                          *)
(* ======================================================================== *)

(* The closures and the value slots of one index are among those of
   another. *)
Definition contents_within (a b : set_of_closures_contents) : Prop :=
  (∀ g : function_slot,
     among g (contents_closures a) = true ->
     among g (contents_closures b) = true)
  ∧ ∀ w : value_slot,
      among w (contents_value_slots a) = true ->
      among w (contents_value_slots b) = true.

Lemma contents_within_refl (a : set_of_closures_contents) :
  contents_within a a.
Proof. split; auto. Qed.

Lemma contents_within_trans (a b c : set_of_closures_contents) :
  contents_within a b -> contents_within b c -> contents_within a c.
Proof. intros [H1 H2] [H3 H4]. split; auto. Qed.

Lemma contents_subset_within (a b : set_of_closures_contents) :
  set_of_closures_contents_subset a b = true <-> contents_within a b.
Proof.
  unfold set_of_closures_contents_subset, contents_within.
  rewrite andb_true_iff, !(FlambdaMapProofs.all_among _ _). reflexivity.
Qed.

Lemma contents_equal_within (a b : set_of_closures_contents) :
  set_of_closures_contents_equal a b = true
  <-> contents_within a b ∧ contents_within b a.
Proof.
  unfold set_of_closures_contents_equal.
  rewrite andb_true_iff,
    !(keyed_same_keys_b positive Pos.eqb pos_eqb_decides_equality).
  unfold Keyed.same_keys, contents_within.
  change (Keyed.among Pos.eqb) with among. split.
  - intros [Hc Hv]. repeat split; intros x Hx.
    + rewrite <- Hc. exact Hx.
    + rewrite <- Hv. exact Hx.
    + rewrite Hc. exact Hx.
    + rewrite Hv. exact Hx.
  - intros [[Hc1 Hv1] [Hc2 Hv2]].
    split; intros x; apply eq_true_iff_eq; split; auto.
Qed.

Lemma contents_within_union (a b : set_of_closures_contents) :
  contents_within a (set_of_closures_contents_union a b)
  ∧ contents_within b (set_of_closures_contents_union a b).
Proof.
  unfold contents_within, set_of_closures_contents_union.
  cbn [contents_closures contents_value_slots].
  repeat split; intros x Hx; rewrite among_dedup, among_app, Hx;
    [ reflexivity | reflexivity | apply orb_true_r | apply orb_true_r ].
Qed.

Lemma contents_union_within (a b c : set_of_closures_contents) :
  contents_within a c -> contents_within b c ->
  contents_within (set_of_closures_contents_union a b) c.
Proof.
  unfold contents_within, set_of_closures_contents_union.
  cbn [contents_closures contents_value_slots].
  intros [H1 H2] [H3 H4].
  split; intros x Hx; rewrite among_dedup, among_app in Hx;
    apply orb_true_iff in Hx as [ Hx | Hx ]; auto.
Qed.

(* The slots of an entry against an index: conditions 8a and 8c of the
   well-formedness of types. *)
Definition entry_slots_fit
    (i : row_like_index set_of_closures_contents unit) (m : closures_entry)
    : bool :=
  match m with
  | Closures_entry function_types closure_types value_slot_types =>
    closures_entry_slots_fit i (domain function_types) (domain closure_types)
      (domain value_slot_types)
  end.

Lemma entry_slots_fit_iff
    (i : row_like_index set_of_closures_contents unit)
    (function_types : finite_map (Or_unknown.t function_type))
    (closure_types value_slot_types : finite_map flambda_type) :
  entry_slots_fit i
    (Closures_entry function_types closure_types value_slot_types)
  = true
  <-> (∀ g : function_slot,
         is_bound g function_types = true ->
         among g (contents_closures (closures_index_contents i)) = true)
      ∧ (∀ g : function_slot,
           is_bound g closure_types = true ->
           among g (contents_closures (closures_index_contents i)) = true)
      ∧ (∀ w : value_slot,
           is_bound w value_slot_types = true ->
           among w (contents_value_slots (closures_index_contents i)) = true)
      ∧ match index_domain i with
        | Row_like_index_domain.Known contents =>
          ∀ g : function_slot,
            among g (contents_closures contents) = true ->
            is_bound g function_types = true
        | Row_like_index_domain.At_least _ => True
        end.
Proof.
  assert (Hdomain :
    ∀ (A : Type) (name : positive) (bindings : finite_map A),
      among name (domain bindings) = is_bound name bindings)
    by exact among_domain.
  unfold entry_slots_fit, closures_entry_slots_fit.
  rewrite !andb_true_iff, !(FlambdaMapProofs.all_among _ _).
  destruct (index_domain i) as [ contents | contents ].
  - rewrite (FlambdaMapProofs.all_among _ _). setoid_rewrite Hdomain. tauto.
  - setoid_rewrite Hdomain. tauto.
Qed.

(* What an index says of a set with the functions and the value slots
   given. *)
Lemma index_satisfied_within (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (i : row_like_index set_of_closures_contents unit) :
  let actual :=
    Set_of_closures_contents (domain functions) (domain value_slots)
  in
  closures_index_satisfied functions value_slots i
  <-> match index_domain i with
      | Row_like_index_domain.Known contents =>
        contents_within contents actual ∧ contents_within actual contents
      | Row_like_index_domain.At_least contents =>
        contents_within contents actual
      end.
Proof.
  unfold closures_index_satisfied, contents_within.
  cbn [contents_closures contents_value_slots].
  destruct (index_domain i) as [ contents | contents ].
  - unfold same_names. split.
    + intros [Hc Hv]. repeat split; intros x Hx.
      * rewrite <- Hc. exact Hx.
      * rewrite <- Hv. exact Hx.
      * rewrite Hc. exact Hx.
      * rewrite Hv. exact Hx.
    + intros [[Hc1 Hv1] [Hc2 Hv2]].
      split; intros x; apply eq_true_iff_eq; split; auto.
  - rewrite !(FlambdaMapProofs.all_among _ _). reflexivity.
Qed.

Lemma index_satisfied_contents (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    (i : row_like_index set_of_closures_contents unit) :
  closures_index_satisfied functions value_slots i ->
  contents_within (closures_index_contents i)
    (Set_of_closures_contents (domain functions) (domain value_slots)).
Proof.
  intros Hindex. apply index_satisfied_within in Hindex.
  unfold closures_index_contents.
  destruct (index_domain i) as [ contents | contents ];
    [ exact (proj1 Hindex) | exact Hindex ].
Qed.

(* ======================================================================== *)
(* What is taken for the parts of the general statement                     *)
(* ======================================================================== *)

(* What [meet_row_like_for_closures] gives [meet_row_like]. *)
Definition closures_meet_shape (_ _ : unit) : Or_bottom.t unit :=
  Or_bottom.Ok tt.

(* The index of a case and its entry are well formed together, for the
   case of a function slot: the index has the slot (condition 8b), and
   the slots of the entry are as the index allows (8a and 8c). There
   is no case for the other slots. *)
Definition closures_index_ok (o : option function_slot)
    (i : row_like_index set_of_closures_contents unit) (m : closures_entry)
    : Prop :=
  match o with
  | Some slot =>
    closures_index_has_slot slot i = true ∧ entry_slots_fit i m = true
  | None => False
  end.

Definition closures_find (slot : function_slot)
    (known : finite_map row_like_closures_case)
    : option (Or_unknown.t row_like_closures_case) :=
  option_map Or_unknown.Known (lookup slot known).

Definition closures_known_ok (known : finite_map row_like_closures_case)
    : Prop :=
  distinct (keys known) = true.

(* What the meet of two indexes answers. *)
Lemma closures_meet_index_cases (menv0 menv1 : Meet_env.t)
    (i1 i2 : row_like_index set_of_closures_contents unit)
    (ri : meet_return_value (row_like_index set_of_closures_contents unit)) :
  row_meet_index set_of_closures_contents_equal
    set_of_closures_contents_subset set_of_closures_contents_union
    closures_meet_shape menv0 i1 i2
  = Meet_result.Ok ri menv1 ->
  let answered := extract_value ri i1 i2 in
  contents_within (closures_index_contents i1)
    (closures_index_contents answered)
  ∧ contents_within (closures_index_contents i2)
      (closures_index_contents answered)
  ∧ (answered = i1 ∨ answered = i2
     ∨ ∃ contents : set_of_closures_contents,
         index_domain answered = Row_like_index_domain.At_least contents).
Proof.
  unfold row_meet_index, closures_meet_shape.
  destruct i1 as [d1 s1], i2 as [d2 s2]. cbn [index_domain index_shape].
  destruct d1 as [ a | a ], d2 as [ b | b ]; intros Hindex.
  all: repeat match type of Hindex with
       | context [if ?test then _ else _] =>
         destruct test eqn:?; try discriminate Hindex
       end.
  all: injection Hindex as <- _.
  all: cbn [extract_value closures_index_contents index_domain].
  all: repeat match goal with
       | H : set_of_closures_contents_equal _ _ = true |- _ =>
         apply contents_equal_within in H; destruct H
       | H : set_of_closures_contents_subset _ _ = true |- _ =>
         apply contents_subset_within in H
       end.
  all: try (split; [ apply contents_within_refl | ];
            split; [ assumption | auto ]; fail).
  all: try (split; [ assumption | ];
            split; [ apply contents_within_refl | auto ]; fail).
  split; [ exact (proj1 (contents_within_union a b)) | ].
  split; [ exact (proj2 (contents_within_union a b)) | ].
  right. right. eexists. reflexivity.
Qed.

(* The index answered and the entry answered are well formed
   together. *)
Lemma closures_index_ok_meet (mh : meet_expanded_head)
    (o : option function_slot) (menv0 menv1 menv2 : Meet_env.t)
    (i1 i2 : row_like_index set_of_closures_contents unit)
    (m1 m2 : closures_entry)
    (ri : meet_return_value (row_like_index set_of_closures_contents unit))
    (rm : meet_return_value closures_entry) :
  closures_index_ok o i1 m1 ->
  closures_index_ok o i2 m2 ->
  row_meet_index set_of_closures_contents_equal
    set_of_closures_contents_subset set_of_closures_contents_union
    closures_meet_shape menv0 i1 i2
  = Meet_result.Ok ri menv1 ->
  meet_closures_entry mh menv1 m1 m2 = Some (Meet_result.Ok rm menv2) ->
  closures_index_ok o (extract_value ri i1 i2) (extract_value rm m1 m2).
Proof.
  destruct o as [ slot | ]; [ | intros [] ].
  intros [Hslot1 Hfit1] [Hslot2 Hfit2] Hindex Hmeet.
  destruct (closures_meet_index_cases menv0 menv1 i1 i2 ri Hindex)
    as (Hw1 & Hw2 & Hwhich).
  assert (Hslots :=
    meet_closures_entry_slots_for mh menv1 menv2 m1 m2 (extract_value rm m1 m2)
      rm
      Hmeet (extract_value_answer_for _ rm m1 m2)).
  remember (extract_value ri i1 i2) as answered eqn:Hanswered.
  remember (extract_value rm m1 m2) as entry eqn:Hentry.
  clear Hanswered Hentry Hindex Hmeet.
  destruct m1 as [ft1 ct1 vst1], m2 as [ft2 ct2 vst2], entry as [ft ct vst].
  destruct Hslots as (Hf & Hc & Hv).
  apply entry_slots_fit_iff in Hfit1 as (Hf1 & Hc1 & Hv1 & He1).
  apply entry_slots_fit_iff in Hfit2 as (Hf2 & Hc2 & Hv2 & He2).
  split.
  - destruct Hwhich as [ -> | [ -> | [contents Hdomain] ] ];
      [ exact Hslot1 | exact Hslot2 | ].
    unfold closures_index_has_slot. rewrite Hdomain. reflexivity.
  - apply entry_slots_fit_iff. split; [ | split; [ | split ] ].
    + intros g Hg. rewrite Hf in Hg. apply orb_true_iff in Hg as [ Hg | Hg ];
        [ exact (proj1 Hw1 g (Hf1 g Hg)) | exact (proj1 Hw2 g (Hf2 g Hg)) ].
    + intros g Hg. rewrite Hc in Hg. apply orb_true_iff in Hg as [ Hg | Hg ];
        [ exact (proj1 Hw1 g (Hc1 g Hg)) | exact (proj1 Hw2 g (Hc2 g Hg)) ].
    + intros w Hw. rewrite Hv in Hw. apply orb_true_iff in Hw as [ Hw | Hw ];
        [ exact (proj2 Hw1 w (Hv1 w Hw)) | exact (proj2 Hw2 w (Hv2 w Hw)) ].
    + destruct Hwhich as [ -> | [ -> | [contents Hdomain] ] ].
      * destruct (index_domain i1) as [ contents | contents ]; [ | exact I ].
        intros g Hg. rewrite Hf, (He1 g Hg). reflexivity.
      * destruct (index_domain i2) as [ contents | contents ]; [ | exact I ].
        intros g Hg. rewrite Hf, (He2 g Hg). apply orb_true_r.
      * rewrite Hdomain. exact I.
Qed.

(* Two indexes that a set of closures both has meet, to an index that
   it has. *)
Lemma closures_index_true_meet (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (menv0 : Meet_env.t)
    (i1 i2 : row_like_index set_of_closures_contents unit) :
  closures_index_satisfied functions value_slots i1 ->
  closures_index_satisfied functions value_slots i2 ->
  ∃ ri : meet_return_value (row_like_index set_of_closures_contents unit),
    row_meet_index set_of_closures_contents_equal
      set_of_closures_contents_subset set_of_closures_contents_union
      closures_meet_shape menv0 i1 i2
    = Meet_result.Ok ri menv0
    ∧ closures_index_satisfied functions value_slots (extract_value ri i1 i2).
Proof.
  intros H1 H2. assert (H1' := H1). assert (H2' := H2).
  apply index_satisfied_within in H1', H2'.
  unfold row_meet_index, closures_meet_shape.
  destruct i1 as [d1 []], i2 as [d2 []]. cbn [index_domain index_shape] in *.
  destruct d1 as [ a | a ], d2 as [ b | b ].
  - destruct H1' as [Ha1 Ha2], H2' as [Hb1 Hb2].
    assert (Heq : set_of_closures_contents_equal a b = true).
    { apply contents_equal_within.
      split; eapply contents_within_trans; eassumption. }
    rewrite Heq. eexists. split; [ reflexivity | exact H1 ].
  - destruct H1' as [Ha1 Ha2].
    assert (Hsub : set_of_closures_contents_subset b a = true).
    { apply contents_subset_within.
      eapply contents_within_trans; eassumption. }
    rewrite Hsub. eexists. split; [ reflexivity | exact H1 ].
  - destruct H2' as [Hb1 Hb2].
    assert (Hsub : set_of_closures_contents_subset a b = true).
    { apply contents_subset_within.
      eapply contents_within_trans; eassumption. }
    rewrite Hsub. eexists. split; [ reflexivity | exact H2 ].
  - destruct (set_of_closures_contents_subset a b),
      (set_of_closures_contents_subset b a);
      (eexists; split; [ reflexivity | ]); cbn [extract_value];
      try assumption.
    apply index_satisfied_within. cbn [index_domain].
    exact (contents_union_within a b _ H1' H2').
Qed.

(* ======================================================================== *)
(* The closure in hand                                                      *)
(* ======================================================================== *)

Section Closure.

Variable v : value.
Variable slot : function_slot.
Variable functions : list (function_slot * code_id).
Variable value_slots : list (value_slot * value).

(* The static environment says that the value is the closure at the
   slot of a set with the functions and the value slots. *)
Definition closure_fact (statics : static_env) : Prop :=
  closure_in_set statics v = Some (slot, functions, value_slots).

Definition entry_true (statics : static_env) (valuation : env)
    (m : closures_entry) : Prop :=
  closure_fact statics
  ∧ closures_entry_satisfied statics valuation v slot functions value_slots m.

Definition entry_holds (statics : static_env) (valuation : env)
    (m : closures_entry) : Prop :=
  closure_fact statics
  ∧ closures_entry_holds v slot functions value_slots statics valuation m.

(* The index of a case says that the set has every slot that the
   entry has a type for; and the projection of a function slot that
   the set has has a value, where the static environment is valid. *)
Lemma closures_entry_hold (statics : static_env) (valuation : env)
    (o : option function_slot)
    (i : row_like_index set_of_closures_contents unit) (m : closures_entry) :
  statics_valid statics ->
  closures_index_ok o i m ->
  closures_index_satisfied functions value_slots i ->
  entry_true statics valuation m ->
  entry_holds statics valuation m.
Proof.
  assert (Hdomain :
    ∀ (A : Type) (name : positive) (bindings : finite_map A),
      among name (domain bindings) = is_bound name bindings)
    by exact among_domain.
  intros Hstatics Hok Hindex [Hfact Hs]. split; [ exact Hfact | ].
  split; [ exact Hs | ].
  destruct o as [ slot' | ]; [ | destruct Hok ]. destruct Hok as [_ Hfit].
  destruct m as [ft ct vst].
  apply entry_slots_fit_iff in Hfit as (Hf & Hc & Hv & _).
  destruct (index_satisfied_contents functions value_slots i Hindex)
    as [Hclosures Hvalue_slots].
  cbn [contents_closures contents_value_slots] in Hclosures, Hvalue_slots.
  split; [ | split ].
  - intros g Hg. rewrite <- Hdomain. exact (Hclosures g (Hf g Hg)).
  - intros g Hg.
    apply (project_function_slot_valid statics slot g v functions value_slots
             (valid_statics _ Hstatics) Hfact).
    rewrite <- Hdomain. exact (Hclosures g (Hc g Hg)).
  - intros w Hw. rewrite <- Hdomain. exact (Hvalue_slots w (Hv w Hw)).
Qed.

Lemma closures_entry_fits_by_kinds : fits_by_kinds closures_entry_fits.
Proof.
  intros tenv tenv' m [Hmem Hkind] (Hwf & Hv & Ha & He).
  split; [ exact Hwf | ].
  split; [ intros x Hx; exact (Hmem (Name_var x) (Hv x Hx)) | ].
  split; [ intros s k Hs; exact (Hkind s k (Ha s k Hs)) | ].
  intros n k Hn. exact (Hkind _ k (He n k Hn)).
Qed.

Lemma entry_satisfied_extended (statics : static_env) (tenv : Typing_env.t)
    (valuation valuation' : env) (m : closures_entry) :
  closures_entry_fits tenv m ->
  extended_from tenv valuation valuation' ->
  closures_entry_satisfied statics valuation v slot functions value_slots m ->
  closures_entry_satisfied statics valuation' v slot functions value_slots m.
Proof.
  destruct m as [ft ct vst]. intros Hfits Hextended Hs.
  apply closures_entry_fits_maps in Hfits as (_ & [_ Hct] & [_ Hvst]).
  apply closures_entry_lookup in Hs as (Hf & Hc & Hv).
  apply closures_entry_lookup. split; [ exact Hf | ]. split.
  - intros g ty closure Hlookup Hproject.
    exact (type_satisfied_extended statics tenv valuation valuation' closure
             ty (proj1 (Hct g ty Hlookup)) Hextended
             (Hc g ty closure Hlookup Hproject)).
  - intros w ty contents Hlookup Hcontents.
    exact (type_satisfied_extended statics tenv valuation valuation' contents
             ty (Hvst w ty Hlookup) Hextended
             (Hv w ty contents Hlookup Hcontents)).
Qed.

Lemma entry_true_stable (statics : static_env) (tenv : Typing_env.t)
    (valuation valuation' : env) (m : closures_entry) :
  closures_entry_fits tenv m ->
  extended_from tenv valuation valuation' ->
  entry_true statics valuation m ->
  entry_true statics valuation' m.
Proof.
  intros Hfits Hextended [Hfact Hs]. split; [ exact Hfact | ].
  exact (entry_satisfied_extended statics tenv valuation valuation' m Hfits
           Hextended Hs).
Qed.

Lemma entry_part_stable : part_stable closures_entry_fits entry_holds.
Proof.
  split.
  - intros menv menv' m Hextends Hfits.
    exact (closures_entry_fits_by_kinds _ _ m
             (meet_env_extends_knows menv menv' Hextends) Hfits).
  - intros statics tenv valuation valuation' m Hfits Hextended
      [Hfact [Hs Hpresent]].
    split; [ exact Hfact | ]. split; [ | exact Hpresent ].
    exact (entry_satisfied_extended statics tenv valuation valuation' m Hfits
             Hextended Hs).
Qed.

End Closure.

(* ======================================================================== *)
(* The map of the cases                                                     *)
(* ======================================================================== *)

(* The merge that [meet_row_like_for_closures] gives [meet_row_like]. *)
Definition closures_merge_map_known
    (merge_case :
      function_slot -> option (Or_unknown.t row_like_closures_case) ->
      option (Or_unknown.t row_like_closures_case) -> row_like_state ->
      option (option (Or_unknown.t row_like_closures_case) * row_like_state))
    (known1 known2 : finite_map row_like_closures_case)
    (state : row_like_state) :=
  merge_fold_failing Pos.eqb
    (Keyed.keys_of_both Pos.eqb known1 known2)
    (fun fslot case1 case2 state =>
       let case1 := option_map Or_unknown.Known case1 in
       let case2 := option_map Or_unknown.Known case2 in
       let* '(case, state) := merge_case fslot case1 case2 state in
       match case with
       | None => Some (None, state)
       | Some (Or_unknown.Known case) => Some (Some case, state)
       | Some Or_unknown.Unknown => None
       end)
    known1 known2 state.

Lemma closures_merge_map_known_spec
    (merge_case :
      function_slot -> option (Or_unknown.t row_like_closures_case) ->
      option (Or_unknown.t row_like_closures_case) -> row_like_state ->
      option (option (Or_unknown.t row_like_closures_case) * row_like_state))
    (known1 known2 cases : finite_map row_like_closures_case)
    (state0 staten : row_like_state) (Inv : row_like_state -> Prop)
    (le : row_like_state -> row_like_state -> Prop)
    (P : function_slot -> option (Or_unknown.t row_like_closures_case) ->
         row_like_state -> Prop) :
  (∀ s : row_like_state, le s s) ->
  (∀ s1 s2 s3 : row_like_state, le s1 s2 -> le s2 s3 -> le s1 s3) ->
  Inv state0 ->
  (∀ (tag : function_slot) (s s' : row_like_state)
     (c : option (Or_unknown.t row_like_closures_case)),
     Inv s ->
     merge_case tag (closures_find tag known1) (closures_find tag known2) s
     = Some (c, s') ->
     Inv s' ∧ le s s' ∧ P tag c s') ->
  closures_merge_map_known merge_case known1 known2 state0
  = Some (cases, staten) ->
  closures_known_ok cases
  ∧ Inv staten
  ∧ le state0 staten
  ∧ (∀ tag : function_slot,
       closures_find tag known1 <> None ∨ closures_find tag known2 <> None ->
       ∃ st : row_like_state,
         le st staten ∧ P tag (closures_find tag cases) st)
  ∧ ∀ tag : function_slot,
      closures_find tag known1 = None -> closures_find tag known2 = None ->
      closures_find tag cases = None.
Proof.
  intros Hrefl Htrans Hinv Hstep Hmerge.
  unfold closures_merge_map_known in Hmerge.
  set (keys := Keyed.keys_of_both Pos.eqb known1 known2) in *.
  assert (Hnodup : NoDup keys).
  { apply (keyed_distinct_NoDup Pos.eqb pos_eqb_decides_equality).
    apply (keyed_keys_of_both_distinct positive Pos.eqb
             pos_eqb_decides_equality). }
  assert (Hin :
    ∀ tag : function_slot,
      tag ∈ keys
      <-> Keyed.is_bound Pos.eqb tag known1
          || Keyed.is_bound Pos.eqb tag known2
          = true).
  { intros tag.
    rewrite <- (keyed_among_keys_of_both positive Pos.eqb
                  pos_eqb_decides_equality),
      (keyed_among_in positive Pos.eqb pos_eqb_decides_equality).
    reflexivity. }
  match type of Hmerge with
  | merge_fold_failing _ _ ?g _ _ _ = _ => set (f := g) in Hmerge
  end.
  assert (Hstep' :
    ∀ (key : function_slot) (s s' : row_like_state)
      (c : option row_like_closures_case),
      Inv s ->
      f key (Keyed.lookup Pos.eqb key known1) (Keyed.lookup Pos.eqb key known2)
        s
      = Some (c, s') ->
      Inv s' ∧ le s s' ∧ P key (option_map Or_unknown.Known c) s').
  { intros key s s' c Hinv_s Hf. unfold f in Hf. cbv beta zeta in Hf.
    destruct (merge_case key
                (option_map Or_unknown.Known (Keyed.lookup Pos.eqb key known1))
                (option_map Or_unknown.Known (Keyed.lookup Pos.eqb key known2))
                s) as [ [c0 s0] | ] eqn:Hcase; [ | discriminate Hf ].
    cbv beta iota in Hf.
    destruct c0 as [ [ case | ] | ]; try discriminate Hf;
      injection Hf as <- <-; exact (Hstep key s s0 _ Hinv_s Hcase). }
  destruct (merge_fold_failing_spec Pos.eqb pos_eqb_decides_equality f known1
              known2 Inv le
              (fun key c s => P key (option_map Or_unknown.Known c) s)
              Hrefl Htrans Hstep' keys state0 staten cases Hnodup Hinv Hmerge)
    as (Hout & Hinv_n & Hle & Hkeys & Hrest).
  split.
  { exact (proj2 (keyed_distinct_NoDup Pos.eqb pos_eqb_decides_equality _)
             Hout). }
  split; [ exact Hinv_n | ]. split; [ exact Hle | ]. split.
  - intros tag Hbound. apply Hkeys, Hin.
    unfold Keyed.is_bound, closures_find in *.
    change (lookup tag known1) with (Keyed.lookup Pos.eqb tag known1)
      in Hbound.
    change (lookup tag known2) with (Keyed.lookup Pos.eqb tag known2)
      in Hbound.
    destruct (Keyed.lookup Pos.eqb tag known1); [ reflexivity | ].
    destruct (Keyed.lookup Pos.eqb tag known2); [ reflexivity | ].
    destruct Hbound as [ H | H ]; destruct H; reflexivity.
  - intros tag H1 H2. unfold closures_find in *.
    change (lookup tag known1) with (Keyed.lookup Pos.eqb tag known1) in H1.
    change (lookup tag known2) with (Keyed.lookup Pos.eqb tag known2) in H2.
    change (lookup tag cases) with (Keyed.lookup Pos.eqb tag cases).
    rewrite Hrest; [ reflexivity | ]. intros Hmem. apply Hin in Hmem.
    unfold Keyed.is_bound in Hmem.
    destruct (Keyed.lookup Pos.eqb tag known1); [ discriminate H1 | ].
    destruct (Keyed.lookup Pos.eqb tag known2); [ discriminate H2 | ].
    discriminate Hmem.
Qed.

(* ======================================================================== *)
(* A row of closures, and the type that it makes                            *)
(* ======================================================================== *)

(* A row of closures fits. *)
Definition closures_row_fits
    : Typing_env.t ->
      finite_map row_like_closures_case * Or_bottom.t row_like_closures_case ->
      Prop :=
  row_fits closures_index_ok closures_entry_fits closures_find
    closures_known_ok.

(* It is true of the closure in hand. *)
Definition closures_row_true (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value))
    : static_env -> env ->
      finite_map row_like_closures_case * Or_bottom.t row_like_closures_case ->
      Prop :=
  row_true (closures_index_satisfied functions value_slots)
    (entry_true v slot functions value_slots) closures_find slot.

Lemma closures_type_lists_fit (tenv : Typing_env.t)
    (closures : row_like_for_closures) :
  type_fits tenv (create_closures Alloc_mode.Heap closures)
  <-> lists_fit tenv (row_like_for_closures_well_formed closures)
        (free_variables_of_row_like_for_closures closures)
        (aliases_of_row_like_for_closures closures)
        (equation_names_of_row_like_for_closures closures).
Proof. rewrite type_fits_lists. cbn. rewrite !app_nil_r. reflexivity. Qed.

(* The lists of a case of closures. *)
Definition case_lists_fit (tenv : Typing_env.t) (slot : function_slot)
    (c : row_like_closures_case) : Prop :=
  lists_fit tenv (row_like_closures_case_well_formed slot c)
    (free_variables_of_closures_entry (case_maps_to c)
     ++ free_variables_of_env_extension (case_env_extension c))
    (aliases_of_closures_entry (case_maps_to c)
     ++ aliases_of_env_extension (case_env_extension c))
    (equation_names_of_closures_entry (case_maps_to c)
     ++ equation_names_of_env_extension (case_env_extension c)).

Lemma closures_case_lists (tenv : Typing_env.t) (slot : function_slot)
    (c : row_like_closures_case) :
  case_lists_fit tenv slot c
  <-> closures_index_ok (Some slot) (case_index c) (case_maps_to c)
      ∧ closures_entry_fits tenv (case_maps_to c)
      ∧ lists_fit tenv (env_extension_well_formed (case_env_extension c))
          (free_variables_of_env_extension (case_env_extension c))
          (aliases_of_env_extension (case_env_extension c))
          (equation_names_of_env_extension (case_env_extension c)).
Proof.
  unfold case_lists_fit, row_like_closures_case_well_formed.
  rewrite lists_fit_app, lists_fit_andb.
  unfold lists_fit, closures_index_ok, closures_entry_fits.
  destruct (case_maps_to c) as [ft ct vst]. cbn [entry_slots_fit]. split.
  - intros [(Hslot & Hwf & Hv & Ha & He) Hext].
    split.
    { split; [ exact Hslot | ].
      exact (proj1 (proj1 (closures_entry_well_formed_iff _ _ _ _) Hwf)). }
    split; [ | exact Hext ].
    split; [ exists (case_index c); exact Hwf | auto ].
  - intros ([Hslot Hfit] & ((index & Hwf) & Hv & Ha & He) & Hext).
    split; [ | exact Hext ]. split; [ exact Hslot | ]. split; [ | auto ].
    apply closures_entry_well_formed_iff.
    apply closures_entry_well_formed_iff in Hwf as (_ & Hrest).
    split; [ exact Hfit | exact Hrest ].
Qed.

(* A row of closures fits as the type that it makes does. *)
Lemma closures_type_fits_row (tenv : Typing_env.t)
    (known : finite_map row_like_closures_case) :
  typing_env_invariant tenv ->
  type_fits tenv (create_closures Alloc_mode.Heap (Row_like_for_closures known))
  ->
  closures_row_fits tenv (known, Or_bottom.Bottom).
Proof.
  intros Hinvariant Hfits. apply closures_type_lists_fit in Hfits.
  destruct Hfits as (Hwf & Hv & Ha & He).
  apply row_like_for_closures_well_formed_iff in Hwf as [Hdistinct Hcases].
  split; [ exact Hdistinct | ].
  split; [ | intros c Hc; discriminate Hc ].
  intros slot c Hc. cbn [fst] in Hc. unfold closures_find in Hc.
  destruct (lookup slot known) as [ c' | ] eqn:Hlookup;
    [ | discriminate Hc ].
  injection Hc as <-.
  assert (Hin := lookup_in slot c' known Hlookup).
  assert (Hlists : case_lists_fit tenv slot c').
  { split; [ exact (Hcases slot c' Hlookup) | ]. split; [ | split ].
    - intros x Hx. apply Hv. cbn. apply in_flat_map. exists (slot, c').
      split; [ exact Hin | exact Hx ].
    - intros s k Hs. apply Ha. cbn. apply in_flat_map. exists (slot, c').
      split; [ exact Hin | exact Hs ].
    - intros n k Hn. apply He. cbn. apply in_flat_map. exists (slot, c').
      split; [ exact Hin | exact Hn ]. }
  apply closures_case_lists in Hlists as (Hindex & Hentry & Hext).
  split; [ exact Hindex | ]. split; [ exact Hentry | ].
  exact (lists_fit_extension tenv _ Hinvariant Hext).
Qed.

Lemma closures_row_fits_type (tenv : Typing_env.t)
    (known : finite_map row_like_closures_case)
    (other : Or_bottom.t row_like_closures_case) :
  closures_row_fits tenv (known, other) ->
  type_fits tenv
    (create_closures Alloc_mode.Heap (Row_like_for_closures known)).
Proof.
  intros (Hok & Hknown & _). cbn [fst] in Hok, Hknown.
  apply closures_type_lists_fit.
  assert (Hcase :
    ∀ (slot : function_slot) (c : row_like_closures_case),
      (slot, c) ∈ known -> case_lists_fit tenv slot c).
  { intros slot c Hin.
    assert (Hlookup : lookup slot known = Some c)
      by exact (in_lookup_distinct Pos.eqb pos_eqb_decides_equality known Hok
                  slot c Hin).
    apply closures_case_lists.
    destruct (Hknown slot c) as (Hindex & Hentry & Hext).
    { unfold closures_find. rewrite Hlookup. reflexivity. }
    split; [ exact Hindex | ]. split; [ exact Hentry | ].
    exact (extension_lists_fit tenv _ Hext). }
  split; [ | split; [ | split ] ].
  - apply row_like_for_closures_well_formed_iff. split; [ exact Hok | ].
    intros slot c Hlookup.
    exact (proj1 (Hcase slot c (lookup_in slot c known Hlookup))).
  - intros x Hx. cbn in Hx. apply in_flat_map in Hx as ([slot c] & Hin & Hx).
    exact (proj1 (proj2 (Hcase slot c Hin)) x Hx).
  - intros s k Hs. cbn in Hs. apply in_flat_map in Hs as ([slot c] & Hin & Hs).
    exact (proj1 (proj2 (proj2 (Hcase slot c Hin))) s k Hs).
  - intros n k Hn. cbn in Hn. apply in_flat_map in Hn as ([slot c] & Hin & Hn).
    exact (proj2 (proj2 (proj2 (Hcase slot c Hin))) n k Hn).
Qed.

(* A row of closures is true of a closure as the meaning of types
   says. *)
Lemma closures_row_true_satisfied (v : value) (slot : function_slot)
    (functions : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (statics : static_env)
    (valuation : env) (known : finite_map row_like_closures_case) :
  closure_fact v slot functions value_slots statics
  ∧ row_like_for_closures_satisfied statics valuation v slot functions
      value_slots (Row_like_for_closures known)
  <-> closures_row_true v slot functions value_slots statics valuation
        (known, Or_bottom.Bottom).
Proof.
  rewrite (row_like_for_closures_lookup statics valuation v slot functions
             value_slots known).
  unfold closures_row_true, row_true, closures_find. cbn [fst snd].
  destruct (lookup slot known) as [ c | ]; cbn.
  - unfold case_true, entry_true, row_like_closures_case_satisfied,
      closures_case_satisfied. split.
    + intros [Hfact (case & [= <-] & Hi & He & Hx)]. repeat split; assumption.
    + intros (Hi & [Hfact He] & Hx). split; [ exact Hfact | ]. exists c.
      repeat split; assumption.
  - split; [ intros [_ (case & H & _)]; discriminate H | intros [] ].
Qed.

(* ======================================================================== *)
(* The meet of two rows of closures                                         *)
(* ======================================================================== *)

Section Closures.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.

Hypothesis Hjoin : join_scoped_envs_sound join.
Hypothesis Hdefine : define_again choices = false.
Hypothesis Hlift : lift_variables_of_single_case choices = true.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.

(* The closure in hand. *)
Variable v : value.
Variable slot : function_slot.
Variable functions : list (function_slot * code_id).
Variable value_slots : list (value_slot * value).

(* [meet_row_like], as [meet_row_like_for_closures] calls it, is right
   for the closure in hand. *)
Lemma meet_closure_rows_ok :
  part_meet_ok closures_row_fits
    (closures_row_true v slot functions value_slots)
    (fun (env : Meet_env.t)
         (left right :
           finite_map row_like_closures_case
           * Or_bottom.t row_like_closures_case) =>
       meet_row_like join choices mh (meet_closures_entry mh)
         set_of_closures_contents_equal set_of_closures_contents_subset
         set_of_closures_contents_union closures_meet_shape Keyed.is_empty
         (Keyed.get_singleton_binding Pos.eqb) closures_merge_map_known env
         (fst left) (fst right) (snd left) (snd right)).
Proof.
  intros menv [known1 other1] [known2 other2] result Henv H1 H2 Hmeet.
  cbn [fst snd] in Hmeet. rewrite meet_row_like_unfold in Hmeet.
  refine (row_like_body_ok join choices mh (meet_closures_entry mh)
            set_of_closures_contents_equal set_of_closures_contents_subset
            set_of_closures_contents_union closures_meet_shape Hjoin Hdefine
            Hlift Hkeeps Hsound closures_index_ok closures_entry_fits
            (closures_index_satisfied functions value_slots)
            (entry_true v slot functions value_slots)
            (entry_holds v slot functions value_slots)
            (entry_part_stable v slot functions value_slots)
            closures_entry_fits_by_kinds
            (entry_true_stable v slot functions value_slots)
            (part_meet_ok_fact closures_entry_fits
               (closures_entry_holds v slot functions value_slots)
               (closure_fact v slot functions value_slots)
               (meet_closures_entry mh)
               (meet_closures_entry_ok_for mh v slot functions value_slots
                  Hkeeps
                  Hsound))
            _ _ _ _ _ menv Henv _ Keyed.is_empty closures_merge_map_known
            closures_find closures_known_ok _ closures_merge_map_known_spec
            slot known1 known2 other1 other2 result H1 H2 Hmeet).
  - intros statics valuation o i m Hstatics Hindex Htrue Hm.
    exact (closures_entry_hold v slot functions value_slots statics valuation
             o i m Hstatics Hindex Htrue Hm).
  - intros statics valuation m [Hfact [Hs _]]. split; assumption.
  - intros tag i m [].
  - intros o menv0 menv1 menv2 i1 i2 m1 m2 ri rm.
    apply closures_index_ok_meet.
  - intros o menv0 i1 i2 m1 m2 _ _. apply closures_index_true_meet.
  - intros cases Hempty tag.
    destruct cases; [ reflexivity | discriminate Hempty ].
Qed.

(* [meet_row_like_for_closures] is right for the closure in hand. *)
Lemma meet_row_like_for_closures_at_ok :
  part_meet_ok (fits_as_type (create_closures Alloc_mode.Heap))
    (fun (statics : static_env) (valuation : env)
         (closures : row_like_for_closures) =>
       closure_fact v slot functions value_slots statics
       ∧ row_like_for_closures_satisfied statics valuation v slot functions
           value_slots closures)
    (meet_row_like_for_closures join choices mh).
Proof.
  intros menv [known1] [known2] result Henv H1 H2 Hmeet.
  assert (Hinvariant := proj1 Henv).
  assert (Hr1 := closures_type_fits_row _ known1 Hinvariant H1).
  assert (Hr2 := closures_type_fits_row _ known2 Hinvariant H2).
  unfold meet_row_like_for_closures in Hmeet. cbv zeta in Hmeet.
  destruct (meet_row_like _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
    as [ res | ] eqn:Hrow in Hmeet; [ | discriminate Hmeet ].
  cbv beta iota in Hmeet.
  destruct (meet_closure_rows_ok menv (known1, Or_bottom.Bottom)
              (known2, Or_bottom.Bottom) res Henv Hr1 Hr2 Hrow)
    as [Hkept Hlost].
  split.
  - intros r menv' Hresult. subst result.
    destruct res as [ r0 | r0 menv0 ]; [ discriminate Hmeet | ].
    destruct (Hkept r0 menv0 eq_refl) as (Hextends & Henv' & Hfit & Hnew).
    destruct r0 as [ | | | [known other] ];
      [ | | | destruct other as [ c | ] ];
      try discriminate Hmeet; injection Hmeet as <- <-;
      (split; [ exact Hextends | ]); (split; [ exact Henv' | ]);
      (split; [ | exact Hnew ]);
      cbn [map_return_value extract_value fst] in *; unfold fits_as_type;
      exact (closures_row_fits_type _ _ _ Hfit).
  - intros statics valuation Hstatics Hsatisfied Hs1 Hs2. cbv beta in Hs1, Hs2.
    apply closures_row_true_satisfied in Hs1, Hs2.
    destruct (Hlost statics valuation Hstatics Hsatisfied Hs1 Hs2)
      as (r0 & menv' & valuation' & -> & Hextended & Hsatisfied' & Htrue
          & Hvalued).
    destruct r0 as [ | | | [known other] ];
      [ | | | destruct other as [ c | ] ];
      try discriminate Hmeet; injection Hmeet as <-;
      eexists _, menv', valuation'; (split; [ reflexivity | ]);
      (split; [ exact Hextended | ]); (split; [ exact Hsatisfied' | ]);
      (split; [ | exact Hvalued ]);
      cbn [map_return_value extract_value fst] in *;
      apply closures_row_true_satisfied; exact Htrue.
Qed.

End Closures.

(* [meet_row_like_for_closures] is right, for any value: the closure
   in hand is the one that the value is, which the static environment
   says. *)
Lemma meet_row_like_for_closures_ok_for :
  for_validity meet_row_like_for_closures_ok_statement.
Proof.
  intros join choices mh v Hjoin Hdefine Hlift Hkeeps Hsound.
  set (choose :=
    fun (statics : static_env)
        (closure :
          function_slot * list (function_slot * code_id)
          * list (value_slot * value)) =>
      closure_in_set statics v = Some closure).
  set (satisfied :=
    fun (closure :
          function_slot * list (function_slot * code_id)
          * list (value_slot * value))
        (statics : static_env) (valuation : env)
        (closures : row_like_for_closures) =>
      closure_fact v (fst (fst closure)) (snd (fst closure)) (snd closure)
        statics
      ∧ row_like_for_closures_satisfied statics valuation v
          (fst (fst closure)) (snd (fst closure)) (snd closure) closures).
  assert (Hfunctional :
    ∀ (statics : static_env)
      (closure closure' :
        function_slot * list (function_slot * code_id)
        * list (value_slot * value)),
      choose statics closure -> choose statics closure' -> closure = closure').
  { intros statics closure closure' H H'. unfold choose in H, H'. congruence. }
  assert (Hiff :
    ∀ (statics : static_env) (valuation : env)
      (closures : row_like_for_closures),
      satisfied_as_type (create_closures Alloc_mode.Heap) v statics valuation
        closures
      <-> chosen_satisfied choose satisfied statics valuation closures).
  { intros statics valuation closures.
    unfold satisfied_as_type, chosen_satisfied, choose, satisfied,
      create_closures, closure_fact.
    rewrite FlambdaTypeMeaningProofs.non_null_value_satisfied,
      closures_satisfied.
    split.
    - intros (slot & functions & value_slots & Hclosure & Hrow).
      exists (slot, functions, value_slots). cbn [fst snd]. auto.
    - intros ([[slot functions] value_slots] & Hclosure & _ & Hrow).
      exists slot, functions, value_slots. split; assumption. }
  assert (Hchosen :=
    part_meet_ok_chosen (fits_as_type (create_closures Alloc_mode.Heap))
      choose satisfied (meet_row_like_for_closures join choices mh)
      (1%positive, [], []) Hfunctional
      (fun closure =>
         meet_row_like_for_closures_at_ok join choices mh Hjoin Hdefine Hlift
           Hkeeps Hsound v (fst (fst closure)) (snd (fst closure))
           (snd closure))).
  intros menv closures1 closures2 result Henv H1 H2 Hmeet.
  apply (meet_call_ok_change _ _ _ _ menv closures1 closures2 closures1
           closures2 result
           (Hchosen menv closures1 closures2 result Henv H1 H2 Hmeet)).
  - intros r menv' _ H. exact H.
  - intros statics valuation. apply Hiff.
  - intros statics valuation. apply Hiff.
  - intros r statics valuation. apply Hiff.
Qed.

(* ======================================================================== *)
(* The contract of the meet                                                 *)
(* ======================================================================== *)

(* With a join that has its contract, the meet of two heads of kind
   value is right for every meet of expanded types that has its
   own. *)
Lemma value_heads_of_join_for (join : join_scoped_envs)
    (mh : meet_expanded_head) :
  join_scoped_envs_sound join ->
  meet_expanded_head_keeps mh ->
  meet_expanded_head_sound mh ->
  value_heads_right join repaired_choices mh.
Proof.
  intros Hjoin Hkeeps Hsound v.
  exact (meet_value_heads_ok_for join repaired_choices mh v Hjoin eq_refl Hkeeps
           Hsound
           (meet_row_like_for_blocks_ok_for join repaired_choices mh v Hjoin
              eq_refl eq_refl Hkeeps Hsound)
           (meet_row_like_for_closures_ok_for join repaired_choices mh v Hjoin
              eq_refl eq_refl Hkeeps Hsound)).
Qed.

End Validity.

Theorem meet_row_like_for_closures_ok :
  meet_row_like_for_closures_ok_statement.
Proof.
  first
    [ exact (@meet_row_like_for_closures_ok_for standard_validity)
    | exact meet_row_like_for_closures_ok_for ].
Qed.

(* [value_heads_of_join_for], at validity itself. *)
Definition value_heads_of_join :=
  @value_heads_of_join_for standard_validity.

Theorem meet_to_prove_sound : meet_to_prove_sound_statement.
Proof.
  intros join fuel Hjoin. apply meet_to_prove_contract.
  intros mh Hkeeps Hsound _.
  exact (value_heads_of_join join mh Hjoin Hkeeps Hsound).
Qed.

Theorem typing_env_functions_of_meet_to_prove
  : typing_env_functions_of_meet_to_prove_statement.
Proof.
  intros join fuel Hjoin.
  destruct (meet_expanded_head_fuel_ok join repaired_choices eq_refl
              (fun mh Hkeeps Hsound _ =>
                 value_heads_of_join join mh Hjoin Hkeeps Hsound)
              fuel) as [Hkeeps Hsound].
  exact (typing_env_functions_sound join repaired_choices fuel Hkeeps Hsound).
Qed.

Theorem meet_to_prove_expanded_heads
  : meet_to_prove_expanded_heads_statement.
Proof.
  intros join fuel Hjoin.
  exact (meet_expanded_head_fuel_ok join repaired_choices eq_refl
           (fun mh Hkeeps Hsound _ =>
              value_heads_of_join join mh Hjoin Hkeeps Hsound)
           fuel).
Qed.
