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

(* Proofs of the statements of FlambdaMeetFuelSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaTypeWellFormedSpec FlambdaTypeWellFormedProofs
  FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaEnvAliasesProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvInvariantProofs FlambdaTypingEnvOperationsSpec
  FlambdaTypingEnvOperationsProofs FlambdaTypingEnvKindedSpec
  FlambdaTypingEnvKindedProofs FlambdaMeetGrammar FlambdaExpandHead
  FlambdaExpandHeadSpec FlambdaExpandHeadProofs
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs
  FlambdaMeetStepsSpec FlambdaMeetStepsProofs FlambdaMeetEnvSpec
  FlambdaMeetEnvProofs FlambdaMeetImmediateSpec FlambdaMeetImmediateProofs
  FlambdaMeetFuelSpec.
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
(* An expanded type as a part                                               *)
(* ======================================================================== *)

(* An expanded type fits, and is true of the value in hand, as its
   type does. *)
Definition expanded_fits (tenv : Typing_env.t) (expanded : Expanded_type.t)
    : Prop :=
  type_fits tenv (Expanded_type.to_type expanded).

Definition expanded_satisfied (v : value) (statics : static_env)
    (valuation : env) (expanded : Expanded_type.t) : Prop :=
  type_satisfied statics valuation v (Expanded_type.to_type expanded).

(* What a call answered is right for two expanded types if it is for
   two that have the same types. *)
Lemma meet_call_ok_same_types (v : value) (menv : Meet_env.t)
    (a1 a2 b1 b2 : Expanded_type.t) (result : Meet_result.t Expanded_type.t) :
  Expanded_type.to_type a1 = Expanded_type.to_type b1 ->
  Expanded_type.to_type a2 = Expanded_type.to_type b2 ->
  meet_call_ok expanded_fits (expanded_satisfied v) menv a1 a2 result ->
  meet_call_ok expanded_fits (expanded_satisfied v) menv b1 b2 result.
Proof.
  intros H1 H2 [Hkeeps Hsound].
  assert (Hextract :
    ∀ r : meet_return_value Expanded_type.t,
      Expanded_type.to_type (extract_value r a1 a2)
      = Expanded_type.to_type (extract_value r b1 b2)).
  { intros r. destruct r; cbn [extract_value]; auto. }
  unfold expanded_fits, expanded_satisfied in *. split.
  - intros r menv' Hresult.
    destruct (Hkeeps r menv' Hresult) as (Hextends & Henv & Ha & Hnames).
    rewrite <- Hextract. auto.
  - intros statics valuation Hstatics Henv Hb1 Hb2.
    rewrite <- H1 in Hb1. rewrite <- H2 in Hb2.
    destruct (Hsound statics valuation Hstatics Henv Hb1 Hb2)
      as (r & menv' & valuation' & -> & Hextended & Henv' & Ha & Hvalued).
    exists r, menv', valuation'. rewrite <- Hextract. auto.
Qed.

(* ======================================================================== *)
(* One step                                                                 *)
(* ======================================================================== *)

Section Step.

Variable join : join_scoped_envs.
Variable choices : meet_choices.
Variable mh : meet_expanded_head.
Hypothesis Hchoice :
  type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags.
Hypothesis Hkeeps : meet_expanded_head_keeps mh.
Hypothesis Hsound : meet_expanded_head_sound mh.
Hypothesis Hother : meet_expanded_head_other_kinds mh.
Hypothesis Hvalue : value_heads_right join choices mh.

(* The meet of two heads, as the meet of the expanded types that they
   make. *)
Lemma meet_expanded_head0_ok (v : value) (menv : Meet_env.t)
    (head1 head2 : Expanded_type.head) (k1 k2 : flambda_kind)
    (result : Meet_result.t Expanded_type.t) :
  let tenv := Meet_env.typing_env menv in
  typing_env_fits tenv ->
  expanded_fits tenv (Expanded_type.create k1 (Ok head1)) ->
  expanded_fits tenv (Expanded_type.create k2 (Ok head2)) ->
  meet_expanded_head0 join choices mh menv head1 head2 = Some result ->
  meet_call_ok expanded_fits (expanded_satisfied v) menv
    (Expanded_type.create k1 (Ok head1)) (Expanded_type.create k2 (Ok head2))
    result.
Proof.
  intros tenv Hfits Hfits1 Hfits2 Hmeet.
  destruct head1 as [ head1 | head1 | n1 head1 | head1 | head1 ],
    head2 as [ head2 | head2 | n2 head2 | head2 | head2 ];
    cbn in Hmeet; try discriminate Hmeet.
  - (* Values. *)
    destruct (meet_head_of_kind_value join choices mh menv head1 head2)
      as [ result0 | ] eqn:Hheads; [ | discriminate Hmeet ].
    injection Hmeet as <-.
    apply (meet_call_ok_same_types v menv
             (Expanded_type.create_value head1)
             (Expanded_type.create_value head2));
      [ reflexivity | reflexivity | ].
    apply (map_result_ok_for _ _
             (fun (tenv : Typing_env.t) (head : head_of_kind_value) =>
                type_fits tenv (Type_value (Ok (No_alias head))))
             (fun (statics : static_env) (valuation : env)
                  (head : head_of_kind_value) =>
                type_satisfied statics valuation v
                  (Type_value (Ok (No_alias head))))
             expanded_fits (expanded_satisfied v)
             Expanded_type.create_value).
    + intros tenv0 head Hhead. exact Hhead.
    + intros statics valuation head. reflexivity.
    + exact (Hvalue v menv head1 head2 result0 Hfits Hfits1 Hfits2 Hheads).
  - (* Naked immediates. *)
    destruct (meet_head_of_kind_naked_immediate choices mh menv head1 head2)
      as [ result0 | ] eqn:Hheads; [ | discriminate Hmeet ].
    injection Hmeet as <-.
    apply (meet_call_ok_same_types v menv
             (Expanded_type.create_naked_immediate head1)
             (Expanded_type.create_naked_immediate head2));
      [ reflexivity | reflexivity | ].
    apply (map_result_ok_for _ _
             (fun (tenv : Typing_env.t)
                  (head : head_of_kind_naked_immediate) =>
                type_fits tenv (naked_immediate_head_type head))
             (fun (statics : static_env) (valuation : env)
                  (head : head_of_kind_naked_immediate) =>
                type_satisfied statics valuation v
                  (naked_immediate_head_type head))
             expanded_fits (expanded_satisfied v)
             Expanded_type.create_naked_immediate).
    + intros tenv0 head Hhead. exact Hhead.
    + intros statics valuation head. reflexivity.
    + exact (meet_head_of_kind_naked_immediate_ok_for choices mh v Hchoice
               Hkeeps
               Hsound Hother menv head1 head2 result0 Hfits Hfits1 Hfits2
               Hheads).
  - (* Naked numbers of another kind. *)
    destruct (naked_number_kind_eqb n1 n2) eqn:Hkinds;
      [ | discriminate Hmeet ].
    apply naked_number_kind_eqb_eq in Hkinds. subst n2.
    injection Hmeet as <-.
    assert (Hnumbers :
      ∀ numbers : list Z,
        expanded_fits tenv
          (Expanded_type.create (Naked_number n1)
             (Ok (Expanded_type.Head_naked_number n1 numbers))) ->
        has_naked_number_types n1 = true ∧ numbers <> []).
    { intros numbers (Hwf & _). cbn in Hwf.
      apply andb_true_iff in Hwf. destruct Hwf as [Hhas Hnon_empty].
      split; [ exact Hhas | ]. intros ->. discriminate Hnon_empty. }
    destruct (Hnumbers head1 Hfits1) as [Hhas Hhead1].
    destruct (Hnumbers head2 Hfits2) as [_ Hhead2].
    apply (meet_call_ok_same_types v menv
             (Expanded_type.create_naked_number n1 head1)
             (Expanded_type.create_naked_number n1 head2));
      [ reflexivity | reflexivity | ].
    apply (map_result_ok_for _ _
             (fun (_ : Typing_env.t) (numbers : list Z) => numbers <> [])
             (fun (_ : static_env) (_ : env) (numbers : list Z) =>
                naked_numbers_satisfied n1 v numbers)
             expanded_fits (expanded_satisfied v)
             (Expanded_type.create_naked_number n1)).
    + intros tenv0 numbers Hnon_empty. apply closed_type_fits;
        try reflexivity.
      cbn. rewrite Hhas. destruct numbers; [ destruct (Hnon_empty eq_refl) | ].
      reflexivity.
    + intros statics valuation numbers. reflexivity.
    + exact (naked_number_meet_ok_for n1 v menv head1 head2 _ Hfits Hhead1
               Hhead2
               eq_refl).
  - (* Rec_info. *)
    injection Hmeet as <-. cbn.
    apply (meet_call_ok_same_env expanded_fits (expanded_satisfied v) menv _ _
             Both_inputs); [ exact Hfits | exact Hfits1 | ].
    intros statics valuation _ H1 _. exact H1.
  - (* Region. *)
    injection Hmeet as <-. cbn.
    apply (meet_call_ok_same_env expanded_fits (expanded_satisfied v) menv _ _
             Both_inputs); [ exact Hfits | exact Hfits1 | ].
    intros statics valuation _ H1 _. exact H1.
Qed.

(* An expanded type whose description is bottom is true of nothing. *)
Lemma expanded_bottom_not_satisfied (v : value) (statics : static_env)
    (valuation : env) (expanded : Expanded_type.t) :
  Expanded_type.descr expanded = Bottom ->
  ~ expanded_satisfied v statics valuation expanded.
Proof.
  intros Hdescr. unfold expanded_satisfied, Expanded_type.to_type.
  rewrite Hdescr.
  apply (is_obviously_bottom_satisfied statics valuation v).
  destruct (Expanded_type.kind expanded) as [ | [] | | ]; reflexivity.
Qed.

(* One call of the step is right. *)
Lemma step_call_ok (v : value) (menv : Meet_env.t)
    (expanded1 expanded2 : Expanded_type.t)
    (result : Meet_result.t Expanded_type.t) :
  let tenv := Meet_env.typing_env menv in
  typing_env_fits tenv ->
  expanded_fits tenv expanded1 ->
  expanded_fits tenv expanded2 ->
  meet_expanded_head_step join choices mh menv expanded1 expanded2
  = Some result ->
  meet_call_ok expanded_fits (expanded_satisfied v) menv expanded1 expanded2
    result.
Proof.
  intros tenv Hfits Hfits1 Hfits2 Hmeet.
  unfold meet_expanded_head_step, meet_or_unknown_or_bottom in Hmeet.
  destruct (Expanded_type.descr expanded1) as [ | head1 | ] eqn:Hdescr1,
    (Expanded_type.descr expanded2) as [ | head2 | ] eqn:Hdescr2;
    try (injection Hmeet as <-).
  all: try (apply meet_call_ok_bottom;
            intros statics valuation _ _ H1 H2;
            first
              [ exact (expanded_bottom_not_satisfied v statics valuation _
                         Hdescr1 H1)
              | exact (expanded_bottom_not_satisfied v statics valuation _
                         Hdescr2 H2) ]).
  all: try (apply meet_call_ok_same_env;
            [ exact Hfits
            | first [ exact Hfits1 | exact Hfits2 ]
            | intros statics valuation _ H1 H2;
              first [ exact H1 | exact H2 ] ]).
  destruct expanded1 as [k1 descr1], expanded2 as [k2 descr2].
  cbn [Expanded_type.descr] in Hdescr1, Hdescr2. subst descr1 descr2.
  exact (meet_expanded_head0_ok v menv head1 head2 k1 k2 result Hfits Hfits1
           Hfits2 Hmeet).
Qed.

(* The type that the step answers: its kind, and that it is neither
   bottom nor, unless it is the second input, unknown. *)
Lemma step_answer_shape (menv menv' : Meet_env.t)
    (expanded1 expanded2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t) :
  kind (Expanded_type.to_type expanded1)
  = kind (Expanded_type.to_type expanded2) ->
  meet_expanded_head_step join choices mh menv expanded1 expanded2
  = Some (Meet_result.Ok r menv') ->
  let answered :=
    Expanded_type.to_type (extract_value r expanded1 expanded2)
  in
  kind answered = kind (Expanded_type.to_type expanded1)
  ∧ is_obviously_bottom answered = false
  ∧ (is_obviously_unknown answered = true ->
     r = Right_input ∨ r = Both_inputs).
Proof.
  intros Hkinds Hmeet.
  unfold meet_expanded_head_step, meet_or_unknown_or_bottom in Hmeet.
  destruct expanded1 as [k1 descr1], expanded2 as [k2 descr2].
  cbn [Expanded_type.descr] in Hmeet.
  assert (Hhead :
    ∀ (k : flambda_kind) (head : Expanded_type.head),
      is_obviously_bottom
        (Expanded_type.to_type (Expanded_type.create k (Ok head)))
      = false
      ∧ is_obviously_unknown
          (Expanded_type.to_type (Expanded_type.create k (Ok head)))
        = false).
  { intros k head. destruct head; split; reflexivity. }
  assert (Hinput1 :
    ∀ head1 : Expanded_type.head,
      descr1 = Ok head1 ->
      let answered :=
        Expanded_type.to_type (Expanded_type.create k1 descr1)
      in
      kind answered
      = kind (Expanded_type.to_type (Expanded_type.create k1 descr1))
      ∧ is_obviously_bottom answered = false
      ∧ (is_obviously_unknown answered = true ->
         @Left_input Expanded_type.t = Right_input
         ∨ @Left_input Expanded_type.t = Both_inputs)).
  { intros head1 ->. destruct (Hhead k1 head1) as [Hb Hu].
    split; [ reflexivity | ]. split; [ exact Hb | ].
    intros H. rewrite Hu in H. discriminate H. }
  destruct descr1 as [ | head1 | ], descr2 as [ | head2 | ];
    try discriminate Hmeet.
  - injection Hmeet as <- _. cbn [extract_value].
    split; [ reflexivity | ].
    split; [ apply is_obviously_bottom_unknown | ]. intros _. right.
    reflexivity.
  - injection Hmeet as <- _. cbn [extract_value].
    split; [ symmetry; exact Hkinds | ].
    split; [ exact (proj1 (Hhead k2 head2)) | ]. intros _. left. reflexivity.
  - injection Hmeet as <- _. exact (Hinput1 head1 eq_refl).
  - assert (Hinput2 :
      let answered :=
        Expanded_type.to_type (Expanded_type.create k2 (Ok head2))
      in
      kind answered
      = kind (Expanded_type.to_type (Expanded_type.create k1 (Ok head1)))
      ∧ is_obviously_bottom answered = false).
    { split; [ symmetry; exact Hkinds | exact (proj1 (Hhead k2 head2)) ]. }
    assert (Hnew :
      ∀ expanded : Expanded_type.t,
        (∃ head : Expanded_type.head,
           Expanded_type.to_type expanded
           = Expanded_type.to_type (Expanded_type.create k1 (Ok head))
           ∧ kind (Expanded_type.to_type expanded)
             = kind (Expanded_type.to_type
                       (Expanded_type.create k1 (Ok head1)))) ->
        r = New_result expanded ->
        let answered :=
          Expanded_type.to_type
            (extract_value r (Expanded_type.create k1 (Ok head1))
               (Expanded_type.create k2 (Ok head2)))
        in
        kind answered
        = kind (Expanded_type.to_type (Expanded_type.create k1 (Ok head1)))
        ∧ is_obviously_bottom answered = false
        ∧ (is_obviously_unknown answered = true ->
           r = Right_input ∨ r = Both_inputs)).
    { intros expanded (head & Htype & Hkind) ->. cbn [extract_value].
      split; [ exact Hkind | ]. rewrite Htype.
      destruct (Hhead k1 head) as [Hb Hu]. split; [ exact Hb | ].
      intros H. rewrite Hu in H. discriminate H. }
    assert (Hcases :
      ∀ (A : Type) (make : A -> Expanded_type.t)
        (inner : option (Meet_result.t A)),
        option_map (map_result make) inner = Some (Meet_result.Ok r menv') ->
        r = Left_input ∨ r = Right_input ∨ r = Both_inputs
        ∨ ∃ a : A, r = New_result (make a)).
    { intros A make [ [ r0 | r0 menv0 ] | ] H; try discriminate H.
      cbn in H. destruct r0; cbn in H; injection H as <- _; eauto. }
    assert (Hinputs :
      r = Left_input ∨ r = Right_input ∨ r = Both_inputs ->
      let answered :=
        Expanded_type.to_type
          (extract_value r (Expanded_type.create k1 (Ok head1))
             (Expanded_type.create k2 (Ok head2)))
      in
      kind answered
      = kind (Expanded_type.to_type (Expanded_type.create k1 (Ok head1)))
      ∧ is_obviously_bottom answered = false
      ∧ (is_obviously_unknown answered = true ->
         r = Right_input ∨ r = Both_inputs)).
    { intros [ -> | [ -> | -> ] ]; cbn [extract_value].
      - exact (Hinput1 head1 eq_refl).
      - destruct Hinput2 as [Hk Hb]. split; [ exact Hk | ].
        split; [ exact Hb | ]. intros _. left. reflexivity.
      - split; [ reflexivity | ].
        split; [ exact (proj1 (Hhead k1 head1)) | ]. intros _. right.
        reflexivity. }
    destruct head1 as [ head1 | head1 | n1 head1 | head1 | head1 ],
      head2 as [ head2 | head2 | n2 head2 | head2 | head2 ];
      cbn [meet_expanded_head0] in Hmeet; try discriminate Hmeet.
    + destruct (Hcases _ _ _ Hmeet) as [ H | [ H | [ H | [a H] ] ] ];
        try (apply Hinputs; auto; fail).
      apply (Hnew (Expanded_type.create_value a)
               (ex_intro _ (Expanded_type.Head_value a)
                  (conj eq_refl eq_refl)) H).
    + destruct (Hcases _ _ _ Hmeet) as [ H | [ H | [ H | [a H] ] ] ];
        try (apply Hinputs; auto; fail).
      apply (Hnew (Expanded_type.create_naked_immediate a)
               (ex_intro _ (Expanded_type.Head_naked_immediate a)
                  (conj eq_refl eq_refl)) H).
    + destruct (naked_number_kind_eqb n1 n2); [ | discriminate Hmeet ].
      destruct (Hcases _ _ (Some _) Hmeet) as [ H | [ H | [ H | [a H] ] ] ];
        try (apply Hinputs; auto; fail).
      apply (Hnew (Expanded_type.create_naked_number n1 a)
               (ex_intro _ (Expanded_type.Head_naked_number n1 a)
                  (conj eq_refl eq_refl)) H).
    + destruct (Hcases _ _ (Some _) Hmeet) as [ H | [ H | [ H | [a H] ] ] ];
        try (apply Hinputs; auto; fail).
      apply (Hnew (Expanded_type.create_rec_info a)
               (ex_intro _ (Expanded_type.Head_rec_info a)
                  (conj eq_refl eq_refl)) H).
    + destruct (Hcases _ _ (Some _) Hmeet) as [ H | [ H | [ H | [a H] ] ] ];
        try (apply Hinputs; auto; fail).
      apply (Hnew (Expanded_type.create_region a)
               (ex_intro _ (Expanded_type.Head_region a)
                  (conj eq_refl eq_refl)) H).
Qed.

Lemma step_keeps
    : meet_expanded_head_keeps (meet_expanded_head_step join choices mh).
Proof.
  intros menv menv' expanded1 expanded2 r tenv answered
    (Hfits & Hfits1 & Hfits2 & Hkinds) Hmeet.
  destruct (step_call_ok (Value_const Null) menv expanded1 expanded2 _ Hfits
              Hfits1 Hfits2 Hmeet) as [Hkept _].
  destruct (Hkept r menv' eq_refl) as (Hextends & Henv & Hanswered & Hnames).
  destruct (step_answer_shape menv menv' expanded1 expanded2 r Hkinds Hmeet)
    as (Hkind & Hbottom & Hunknown).
  split; [ exact Hextends | ]. split; [ exact Henv | ].
  split; [ exact Hanswered | ]. split; [ exact Hkind | ].
  split; [ exact Hbottom | ]. split; [ exact Hunknown | exact Hnames ].
Qed.

Lemma step_sound
    : meet_expanded_head_sound (meet_expanded_head_step join choices mh).
Proof.
  intros statics valuation menv expanded1 expanded2 v result tenv
    (Hfits & Hfits1 & Hfits2 & _) Hstatics Hsatisfied H1 H2 Hmeet.
  exact (proj2 (step_call_ok v menv expanded1 expanded2 result Hfits Hfits1
                  Hfits2 Hmeet)
           statics valuation Hstatics Hsatisfied H1 H2).
Qed.

End Step.

Lemma meet_expanded_head_step_ok_for :
  for_validity meet_expanded_head_step_ok_statement.
Proof.
  intros join choices mh Hchoice Hkeeps Hsound Hother Hvalue. split.
  - apply step_keeps; assumption.
  - apply step_sound; assumption.
Qed.

(* ======================================================================== *)
(* Every fuel                                                               *)
(* ======================================================================== *)

Lemma meet_expanded_head_fuel_ok_for :
  for_validity meet_expanded_head_fuel_ok_statement.
Proof.
  intros join choices Hchoice Hvalue fuel.
  induction fuel as [ | fuel [Hkeeps Hsound] ].
  - split.
    + intros menv menv' expanded1 expanded2 r tenv answered _ Hmeet.
      discriminate Hmeet.
    + intros statics valuation menv expanded1 expanded2 v result tenv _ _ _ _
        _ Hmeet.
      discriminate Hmeet.
  - cbn [Meet.meet_expanded_head].
    assert (Hother :=
      meet_expanded_head_fuel_other_kinds_for join choices fuel).
    exact (meet_expanded_head_step_ok_for join choices _ Hchoice Hkeeps Hsound
             Hother (Hvalue _ Hkeeps Hsound Hother)).
Qed.

(* ======================================================================== *)
(* The meet of types                                                        *)
(* ======================================================================== *)

End Validity.

Theorem meet_expanded_head_step_ok :
  meet_expanded_head_step_ok_statement.
Proof.
  first
    [ exact (@meet_expanded_head_step_ok_for standard_validity)
    | exact meet_expanded_head_step_ok_for ].
Qed.

Theorem meet_expanded_head_fuel_ok :
  meet_expanded_head_fuel_ok_statement.
Proof.
  first
    [ exact (@meet_expanded_head_fuel_ok_for standard_validity)
    | exact meet_expanded_head_fuel_ok_for ].
Qed.

Theorem meet_of_fuel_contract : meet_of_fuel_contract_statement.
Proof.
  intros join choices fuel Hkeeps Hsound.
  set (mh := Meet.meet_expanded_head join choices fuel) in *.
  split; [ | split; [ | split ] ].
  - intros statics valuation tenv t1 t2 v result (Hfits & Ht1 & Ht2 & Hkinds)
      Hstatics Hsatisfied H1 H2 Hmeet.
    unfold Meet.meet in Hmeet. fold mh in Hmeet.
    rewrite (satisfied_not_bottom statics valuation tenv Hsatisfied) in Hmeet.
    destruct (Meet_env.meet_type (Meet_env.create tenv) t1 t2 mh)
      as [ result0 | ] eqn:Htype; [ | discriminate Hmeet ].
    cbv beta iota in Hmeet.
    destruct (meet_type_ok mh v Hkeeps Hsound (Meet_env.create tenv) t1 t2
                result0 Hfits Ht1 Ht2 Htype) as [_ Hok].
    destruct (Hok statics valuation Hstatics Hsatisfied H1 H2)
      as (r & menv' & valuation' & -> & Hextended & Hsatisfied' & Hanswer
          & Hvalued).
    assert (Hnot_bottom : is_obviously_bottom (extract_value r t1 t2) = false).
    { destruct (is_obviously_bottom (extract_value r t1 t2)) eqn:Hb;
        [ | reflexivity ].
      destruct (is_obviously_bottom_satisfied statics valuation' v _ Hb
                  Hanswer). }
    cbv zeta in Hmeet. rewrite Hnot_bottom in Hmeet. cbn in Hmeet.
    injection Hmeet as <-.
    exists (extract_value r t1 t2), (Meet_env.typing_env menv'), valuation'.
    auto.
  - intros tenv tenv' t1 t2 t (Hfits & Ht1 & Ht2 & Hkinds) Hmeet.
    unfold Meet.meet in Hmeet. fold mh in Hmeet.
    destruct (Typing_env.is_bottom tenv); [ discriminate Hmeet | ].
    destruct (Meet_env.meet_type (Meet_env.create tenv) t1 t2 mh)
      as [ [ r0 | r menv' ] | ] eqn:Htype; try discriminate Hmeet.
    cbv beta iota zeta in Hmeet.
    destruct (is_obviously_bottom (extract_value r t1 t2));
      [ discriminate Hmeet | ].
    cbn in Hmeet. injection Hmeet as <- <-.
    destruct (meet_type_keeps mh (Meet_env.create tenv) menv' t1 t2 r Hkeeps
                Hfits Ht1 Ht2 Htype)
      as ([[Hextends Hfits'] _] & Hanswer & Hkind).
    split; [ exact Hfits' | ]. split; [ exact Hanswer | ].
    split; [ exact Hkind | ].
    intros n Hmem.
    exact (tenv_extends_mem _ _ n (meet_env_extends_tenv _ _ Hextends) Hmem).
  - intros tenv t1 t2 Hnot_bottom Hkinds. unfold Meet.meet.
    rewrite Hnot_bottom. unfold Meet_env.meet_type, Meet_env.meet.
    assert (Hneq : kind_eqb (kind t1) (kind t2) = false).
    { destruct (kind_eqb (kind t1) (kind t2)) eqn:H; [ | reflexivity ].
      apply kind_eqb_eq in H. contradiction. }
    rewrite Hneq. reflexivity.
  - intros tenv t1 t2 Hbottom. unfold Meet.meet. rewrite Hbottom.
    reflexivity.
Qed.

Theorem meet_to_prove_contract : meet_to_prove_contract_statement.
Proof.
  intros join Hvalue fuel. unfold meet_to_prove.
  destruct (meet_expanded_head_fuel_ok join repaired_choices eq_refl Hvalue
              fuel) as [Hkeeps Hsound].
  exact (meet_of_fuel_contract join repaired_choices fuel Hkeeps Hsound).
Qed.

(* ======================================================================== *)
(* A name that the environment does not know                                *)
(* ======================================================================== *)

Lemma mem_same (tenv tenv' : Typing_env.t) (n : name) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
  = Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) ->
  Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv ->
  Typing_env.mem None tenv' n = Typing_env.mem None tenv n.
Proof.
  intros Hlookup Hsymbols. unfold Typing_env.mem. destruct n as [ x | sym ].
  - rewrite Hlookup.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv))
      as [ [ty binding_time_and_mode] | ]; reflexivity.
  - rewrite Hsymbols. reflexivity.
Qed.

(* The type of a name that an environment does not know cannot be
   read, there or in an environment that has the same for the name. *)
Lemma find_unknown (tenv tenv' : Typing_env.t) (n : name) (k : flambda_kind) :
  typing_env_invariant tenv ->
  Typing_env.mem None tenv n = false ->
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
  = Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) ->
  Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv ->
  Typing_env.find tenv' n (Some k) = None.
Proof.
  intros Hinvariant Hunknown Hlookup Hsymbols.
  destruct (Typing_env.find tenv' n (Some k)) as [ ty | ] eqn:Hfind;
    [ | reflexivity ].
  exfalso.
  assert (Hkind := simple_of_kind_of_find tenv' n k ty Hfind).
  apply simple_of_kind_name in Hkind.
  destruct Hkind as [ (ty0 & binding_time_and_mode & Hfound & _)
                    | (_ & Hmem & _) ].
  - rewrite Hlookup in Hfound.
    assert (Hkind0 : simple_of_kind tenv (simple_of_name n) (kind ty0)).
    { apply simple_of_kind_name. left. exists ty0, binding_time_and_mode.
      split; [ exact Hfound | reflexivity ]. }
    rewrite (simple_of_kind_mem tenv n _ Hinvariant Hkind0) in Hunknown.
    discriminate Hunknown.
  - rewrite (mem_same tenv tenv' n Hlookup Hsymbols) in Hmem.
    rewrite Hmem in Hunknown. discriminate Hunknown.
Qed.

(* A name that the environment does not know is canonical. *)
Lemma unknown_canonical (tenv : Typing_env.t) (n : name) :
  typing_env_invariant tenv ->
  Typing_env.mem None tenv n = false ->
  is_canonical (Typing_env.aliases tenv) (simple_of_name n).
Proof.
  intros Hinvariant Hunknown.
  destruct (canonical_cases (Typing_env.aliases tenv) (simple_of_name n))
    as [ [H _] | (n' & canonical_element & Heq & Hdemoted & _) ];
    [ exact H | ].
  apply simple_of_name_inj in Heq. subst n'.
  rewrite (proj1 (invariant_aliases_known tenv Hinvariant n canonical_element
                    Hdemoted)) in Hunknown.
  discriminate Hunknown.
Qed.

Section Unknown.

Variable mh : meet_expanded_head.

(* No equation is added to a canonical name that the environment does
   not know: the type of the name cannot be read. *)
Lemma add_or_delay_on_unknown (tenv : Typing_env.t) (menv : Meet_env.t)
    (n : name) (expanded : Expanded_type.t) :
  typing_env_invariant tenv ->
  Typing_env.mem None tenv n = false ->
  Keyed.among name_eqb n (Meet_env.adding_equations_for_names menv) = false ->
  Keyed.lookup name_eqb n
    (Typing_env.names_to_types (Meet_env.typing_env menv))
  = Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) ->
  Typing_env.defined_symbols (Meet_env.typing_env menv)
  = Typing_env.defined_symbols tenv ->
  Meet_env.add_or_delay_concrete_equation_on_canonical true menv
    (simple_of_name n) expanded mh
  = None.
Proof.
  intros Hinvariant Hunknown Hnot_adding Hlookup Hsymbols.
  assert (Hname :
    Meet_env.add_or_delay_concrete_equation_on_canonical_name true menv n
      expanded mh
    = None).
  { unfold Meet_env.add_or_delay_concrete_equation_on_canonical_name.
    rewrite Hnot_adding.
    unfold Meet_env.adding_equation_for_name,
      Meet_env.add_concrete_equation_on_canonical_name.
    cbn [Meet_env.typing_env].
    rewrite (find_unknown tenv _ n _ Hinvariant Hunknown Hlookup Hsymbols).
    reflexivity. }
  destruct n as [ x | sym ]; exact Hname.
Qed.

(* Making a name that the environment does not know an alias of a
   canonical element: nothing is done, or the two are different
   constants, which they are not. *)
Lemma add_alias_between_canonicals_unknown (menv : Meet_env.t)
    (k : flambda_kind) (n : name) (c2 : simple)
    (result : Or_bottom.t Meet_env.t) :
  let tenv := Meet_env.typing_env menv in
  typing_env_invariant tenv ->
  Typing_env.mem None tenv n = false ->
  Keyed.among name_eqb n (Meet_env.adding_equations_for_names menv) = false ->
  is_canonical (Typing_env.aliases tenv) c2 ->
  Meet_env.add_alias_between_canonicals true menv k (simple_of_name n) c2 mh
  = Some result ->
  result = Or_bottom.Ok menv
  ∨ (result = Or_bottom.Bottom
     ∧ Typing_env.add_alias tenv (simple_of_name n) c2 = Some Bottom).
Proof.
  intros tenv Hinvariant Hunknown Hnot_adding Hc2 Hadd.
  unfold Meet_env.add_alias_between_canonicals in Hadd. fold tenv in Hadd.
  destruct (simple_equal (simple_of_name n) c2) eqn:Hsame.
  { left. injection Hadd as <-. reflexivity. }
  destruct (Typing_env.add_alias tenv (simple_of_name n) c2)
    as [ [ | result0 | ] | ] eqn:Halias; [ | | | discriminate Hadd ].
  - left. injection Hadd as <-. reflexivity.
  - exfalso. cbv beta iota in Hadd.
    assert (Hc1 := unknown_canonical tenv n Hinvariant Hunknown).
    assert (Hdifferent : simple_of_name n <> c2).
    { intros Heq. apply simple_equal_eq in Heq. rewrite Heq in Hsame.
      discriminate Hsame. }
    destruct (add_alias_demoted tenv _ _ result0 Halias Hinvariant Hc1 Hc2
                Hdifferent) as [Hwhich _].
    destruct (add_alias_frame tenv _ _ result0 Halias)
      as (_ & Hsymbols & _ & _ & Hnames & _).
    set (tenv1 := Typing_env.add_alias_result_t result0) in *.
    unfold Meet_env.record_demotion in Hadd.
    cbn [Meet_env.typing_env Meet_env.with_typing_env] in Hadd.
    destruct Hwhich as [ [Hcanonical Hdemoted] | [Hcanonical Hdemoted] ].
    + destruct (Typing_env.find tenv1 (Typing_env.demoted_name result0)
                  (Some k)) as [ ty_of_demoted | ];
        [ | discriminate Hadd ].
      cbv beta iota zeta in Hadd.
      destruct (expand_head0 _ ty_of_demoted _) as [ expanded | ];
        [ | discriminate Hadd ].
      cbv beta iota in Hadd. rewrite Hcanonical in Hadd.
      rewrite (add_or_delay_on_unknown tenv _ n expanded Hinvariant Hunknown)
        in Hadd; [ discriminate Hadd | exact Hnot_adding | | ].
      * cbn [Meet_env.typing_env Meet_env.map_typing_env
               Meet_env.with_typing_env].
        rewrite replace_equation_lookup_other;
          [ rewrite Hnames; reflexivity | ].
        intros Heq. apply Hdifferent. rewrite Heq. exact Hdemoted.
      * cbn [Meet_env.typing_env Meet_env.map_typing_env
               Meet_env.with_typing_env].
        rewrite (proj1 (replace_equation_frame tenv1 _ _)). exact Hsymbols.
    + apply simple_of_name_inj in Hdemoted. rewrite Hdemoted in Hadd.
      rewrite (find_unknown tenv tenv1 n k Hinvariant Hunknown
                 (f_equal (Keyed.lookup name_eqb n) Hnames) Hsymbols)
        in Hadd.
      discriminate Hadd.
  - right. injection Hadd as <-. split; reflexivity.
Qed.

(* Adding an equation on a name that the environment does not know:
   nothing is done, or the type is an alias and the name and what it
   is an alias of are different constants. *)
Lemma add_equation0_unknown (menv : Meet_env.t) (n : name)
    (ty : flambda_type) (result : Or_bottom.t Meet_env.t) :
  let tenv := Meet_env.typing_env menv in
  meet_expanded_head_other_kinds mh ->
  typing_env_fits tenv ->
  type_fits tenv ty ->
  Typing_env.mem None tenv n = false ->
  Keyed.among name_eqb n (Meet_env.adding_equations_for_names menv) = false ->
  Meet_env.add_equation0 true menv n ty mh = Some result ->
  result = Or_bottom.Ok menv
  ∨ (result = Or_bottom.Bottom
     ∧ ∃ alias : simple,
         get_alias_opt ty = Some alias
         ∧ Typing_env.add_alias tenv (simple_of_name n)
             (Typing_env.get_canonical_simple_ignoring_name_mode tenv alias)
           = Some Bottom).
Proof.
  intros tenv Hother Hfits Hty Hunknown Hnot_adding Hadd.
  assert (Hinvariant := proj1 Hfits).
  assert (Hcanonical :
    Typing_env.get_canonical_simple_ignoring_name_mode tenv (simple_of_name n)
    = simple_of_name n).
  { rewrite get_canonical_simple_ignoring_name_mode_eq.
    apply canonical_element_of_canonical.
    exact (unknown_canonical tenv n Hinvariant Hunknown). }
  destruct (get_alias_opt ty) as [ alias | ] eqn:Halias.
  - unfold Meet_env.add_equation0, Meet_env.add_equation_on_simple0 in Hadd.
    rewrite Halias in Hadd. fold tenv in Hadd. rewrite Hcanonical in Hadd.
    destruct (alias_canonical_of_kind tenv ty alias Hfits Hty Halias)
      as [Hc2 _].
    destruct (add_alias_between_canonicals_unknown menv (kind ty) n _ result
                Hinvariant Hunknown Hnot_adding Hc2 Hadd)
      as [ -> | [-> Hbottom] ].
    + left. reflexivity.
    + right. split; [ reflexivity | ]. exists alias.
      split; [ reflexivity | exact Hbottom ].
  - destruct (add_equation0_kind_cases mh Hother menv n ty result Hfits Halias
                Hadd) as [ Hkind | [ -> | (_ & c & Hcan & _) ] ].
    + rewrite (simple_of_kind_mem tenv n _ Hinvariant Hkind) in Hunknown.
      discriminate Hunknown.
    + left. reflexivity.
    + fold tenv in Hcan. rewrite Hcanonical in Hcan.
      destruct n; discriminate Hcan.
Qed.

End Unknown.

(* ======================================================================== *)
(* Adding to a typing environment                                           *)
(* ======================================================================== *)

(* A name that the environment knows and that has a value of a type is
   of the kind of the type, where the environment is satisfied. *)
Lemma known_name_of_kind (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (n : name) (ty : flambda_type) (v : value) :
  typing_env_satisfied_strictly statics valuation tenv ->
  Typing_env.mem None tenv n = true ->
  value_of_name valuation n = Some v ->
  type_satisfied statics valuation v ty ->
  simple_of_kind tenv (simple_of_name n) (kind ty).
Proof.
  intros (_ & Htypes & _) Hmem Hvalue Hv.
  assert (Hkind := type_satisfied_kind statics valuation v ty Hv).
  apply simple_of_kind_name.
  destruct (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
    as [ [ty0 binding_time_and_mode] | ] eqn:Hlookup.
  - left. exists ty0, binding_time_and_mode. split; [ reflexivity | ].
    destruct (Htypes n ty0 binding_time_and_mode v Hlookup Hvalue)
      as [ [[] _] | Hv0 ].
    rewrite <- Hkind.
    symmetry. exact (type_satisfied_kind statics valuation v ty0 Hv0).
  - right. split; [ reflexivity | ]. split; [ exact Hmem | ].
    destruct n as [ x | sym ].
    + apply mem_var_lookup in Hmem.
      destruct Hmem as (ty0 & binding_time_and_mode & Hfound).
      rewrite Hfound in Hlookup. discriminate Hlookup.
    + cbn in Hvalue. injection Hvalue as <-. symmetry. exact Hkind.
Qed.

Theorem typing_env_functions_sound : typing_env_functions_sound_statement.
Proof.
  intros join choices fuel Hkeeps Hsound.
  set (mh := Meet.meet_expanded_head join choices fuel) in *.
  split; [ | split ].
  - intros statics valuation tenv tenv' n ty v Hfits Hty Hstatics Hsatisfied
      Hvalue Hv Hadd.
    unfold Flambda2_types.add_equation, Meet_env.use_meet_env in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_equation (Meet_env.create tenv) n ty mh)
      as [ menv' | ] eqn:Hadd'; [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    destruct (Typing_env.mem None tenv n) eqn:Hmem.
    + assert (Hkind :=
        known_name_of_kind statics valuation tenv n ty v Hsatisfied Hmem
          Hvalue Hv).
      exact (meet_env_add_equation_sound mh statics valuation
               (Meet_env.create tenv) menv' n ty v Hkeeps Hsound Hfits Hty
               Hkind Hstatics Hsatisfied Hvalue Hv Hadd').
    + unfold Meet_env.add_equation in Hadd'.
      destruct (Meet_env.add_equation0 true (Meet_env.create tenv) n ty mh)
        as [ result | ] eqn:Hadd0; [ | discriminate Hadd' ].
      destruct (add_equation0_unknown mh (Meet_env.create tenv) n ty result
                  (meet_expanded_head_fuel_other_kinds join choices fuel)
                  Hfits Hty Hmem eq_refl Hadd0)
        as [ -> | (-> & alias & Halias & Hbottom) ].
      * cbn in Hadd'. injection Hadd' as <-.
        exact (meet_env_added_refl statics valuation (Meet_env.create tenv)
                 Hsatisfied).
      * exfalso. apply (add_alias_bottom valuation tenv _ _ Hbottom).
        transitivity (Some v); [ exact Hvalue | ]. symmetry.
        exact (alias_canonical_value statics tenv ty alias valuation v
                 Hsatisfied Halias Hv).
  - intros statics valuation tenv tenv' extension Hfits Hextension Hstatics
      Hsatisfied Htrue Hadd.
    unfold Flambda2_types.add_env_extension, Meet_env.use_meet_env in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_env_extension (Meet_env.create tenv) extension mh)
      as [ menv' | ] eqn:Hadd'; [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    apply (meet_env_add_env_extension_sound mh statics valuation
             (Meet_env.create tenv) menv' extension Hkeeps Hsound Hfits
             Hextension); try assumption.
    destruct extension as [equations]. intros n ty Hlookup.
    destruct (Hextension n ty Hlookup) as [Hmem _].
    destruct (proj1 (env_extension_lookup statics valuation equations) Htrue n
                ty Hlookup) as (v & Hvalue & Hv).
    exact (known_name_of_kind statics valuation tenv n ty v Hsatisfied Hmem
             Hvalue Hv).
  - intros statics valuation tenv tenv' params param_types Hfits Htypes
      Hknown Hstatics Hsatisfied Htrue Hadd.
    unfold Flambda2_types.add_equations_on_params, Meet_env.use_meet_env
      in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_equations_on_params (Meet_env.create tenv) params
                param_types mh) as [ menv' | ] eqn:Hadd';
      [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    apply (meet_env_add_equations_on_params_sound mh statics valuation
             (Meet_env.create tenv) menv' params param_types Hkeeps Hsound
             Hfits); try assumption.
    clear Hadd'. induction Htrue as [ | param ty params param_types
                                        (v & Hvalue & Hv) Hrest IH ];
      [ constructor | ].
    constructor.
    + split; [ apply Htypes; left; reflexivity | ].
      exact (known_name_of_kind statics valuation tenv
               (Name_var (fst param)) ty v Hsatisfied
               (Hknown param (or_introl eq_refl)) Hvalue Hv).
    + apply IH.
      * intros ty' Hin. apply Htypes. right. exact Hin.
      * intros param' Hin. apply Hknown. right. exact Hin.
Qed.
