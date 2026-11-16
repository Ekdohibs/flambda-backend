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



(* Proofs of the statements of FlambdaMeetStepsSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetStepsSpec.
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
(* The relations of a naked immediate, reduced                              *)
(* ======================================================================== *)

Lemma relation_eqb_eq : decides_equality relation_eqb.
Proof.
  intros r r'. destruct r, r'; cbn; split; intros H; try reflexivity;
    discriminate.
Qed.

Lemma inverse_relation_eqb_eq : decides_equality inverse_relation_eqb.
Proof.
  intros [r n] [r' n']. unfold inverse_relation_eqb. cbn [fst snd].
  rewrite andb_true_iff, (relation_eqb_eq r r'), (name_eqb_eq n n').
  split.
  - intros [-> ->]. reflexivity.
  - intros H. inversion H. split; reflexivity.
Qed.

(* What the fold of [reduce_head_of_kind_naked_immediate] keeps: the
   number is among the numbers, and each relation kept relates the
   value of its name to the number. *)
Definition reduce_acc_holds (statics : static_env) (valuation : env) (i : Z)
    (acc : reduce_acc) : Prop :=
  match reduce_imms acc with
  | Or_unknown.Unknown => True
  | Or_unknown.Known numbers => i ∈ numbers
  end
  ∧ ∀ (relation : Relation.t) (n : name),
      (relation, n) ∈ reduce_relations acc ->
      ∃ w : value,
        value_of_name valuation n = Some w
        ∧ relation_holds statics relation w i.

Lemma eval_simple_const (valuation : env) (c : const) :
  eval_simple valuation (Const c) = Some (Value_const c).
Proof. reflexivity. Qed.

Lemma reduce_head_of_kind_naked_immediate_sound_for :
  for_validity reduce_head_of_kind_naked_immediate_sound_statement.
Proof.
  intros statics valuation tenv head v Henv (i & Hv & Himms & Hrels).
  unfold reduce_head_of_kind_naked_immediate.
  set (step := fun (state : Or_bottom.t reduce_acc)
                   (inverse : Relation.t * name) => _).
  assert (Hfold :
    ∀ (relations : list (Relation.t * name)) (acc : reduce_acc),
      (∀ (relation : Relation.t) (n : name),
         (relation, n) ∈ relations ->
         ∃ w : value,
           value_of_name valuation n = Some w
           ∧ relation_holds statics relation w i) ->
      reduce_acc_holds statics valuation i acc ->
      ∃ acc' : reduce_acc,
        List.fold_left step relations (Or_bottom.Ok acc) = Or_bottom.Ok acc'
        ∧ reduce_acc_holds statics valuation i acc').
  { induction relations as [ | [relation n] relations IH ];
      intros acc Hall Hacc.
    - exists acc. cbn. auto.
    - cbn [List.fold_left].
      destruct (Hall relation n (or_introl eq_refl)) as (w & Hw & Hholds).
      assert (Hrest :
        ∀ (relation : Relation.t) (n : name),
          (relation, n) ∈ relations ->
          ∃ w : value,
            value_of_name valuation n = Some w
            ∧ relation_holds statics relation w i).
      { intros relation' n' Hin. apply Hall. right. exact Hin. }
      assert (Hcanonical :=
        canonical_ignoring_name_mode_satisfied (fun _ => False) statics
          valuation tenv (simple_of_name n) Henv).
      unfold value_of_name in Hw. rewrite Hw in Hcanonical.
      destruct Hacc as [Hnumbers Hkept].
      assert (Hstep :
        ∃ acc1 : reduce_acc,
          step (Or_bottom.Ok acc) (relation, n) = Or_bottom.Ok acc1
          ∧ reduce_acc_holds statics valuation i acc1).
      { unfold step. cbn [fst snd].
        destruct (Typing_env.get_canonical_simple_ignoring_name_mode tenv
                    (simple_of_name n)) as [ x | s | c ] eqn:Hcan.
        - eexists. split; [ reflexivity | ].
          split; [ exact Hnumbers | ]. cbn [reduce_relations].
          intros relation' n' [ Heq | Hin ]; [ | exact (Hkept _ _ Hin) ].
          inversion Heq; subst. exists w. split; [ exact Hcanonical | ].
          exact Hholds.
        - eexists. split; [ reflexivity | ].
          split; [ exact Hnumbers | ]. cbn [reduce_relations].
          intros relation' n' [ Heq | Hin ]; [ | exact (Hkept _ _ Hin) ].
          inversion Heq; subst. exists w. split; [ exact Hcanonical | ].
          exact Hholds.
        - rewrite eval_simple_const in Hcanonical.
          injection Hcanonical as Hc. subst w.
          destruct (Relation.of_const relation c) as [ imm | ] eqn:Hconst.
          + assert (Himm : i = imm)
              by exact (relation_of_const_sound _ _ _ _ _ Hholds Hconst).
            subst imm.
            destruct (reduce_imms acc) as [ numbers | ] eqn:Hreduce.
            * destruct (Keyed.same_keys_b Z.eqb [i] numbers).
              -- exists acc. split; [ reflexivity | ].
                 split; [ rewrite Hreduce; exact Hnumbers | exact Hkept ].
              -- assert (Hamong : Keyed.among Z.eqb i numbers = true).
                 { apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
                   exact Hnumbers. }
                 rewrite Hamong. cbn [negb].
                 eexists. split; [ reflexivity | ].
                 split; [ cbn; left; reflexivity | exact Hkept ].
            * eexists. split; [ reflexivity | ].
              split; [ cbn; left; reflexivity | exact Hkept ].
          + eexists. split; [ reflexivity | ].
            split; [ exact Hnumbers | exact Hkept ]. }
      destruct Hstep as (acc1 & Hstep & Hacc1).
      rewrite Hstep. exact (IH acc1 Hrest Hacc1). }
  destruct (Hfold (inverse_relations head)
              {| reduce_imms := naked_immediates head;
                 reduce_relations := [];
                 reduce_changed := false |}
              Hrels)
    as (acc & Hfolded & Hnumbers & Hkept).
  { split; [ exact Himms | ]. intros relation n []. }
  rewrite Hfolded.
  destruct (reduce_changed acc); cbn [negb].
  - unfold head_of_kind_naked_immediate_from_descr.
    assert (Hsatisfied :
      head_of_kind_naked_immediate_satisfied statics valuation v
        (Head_of_kind_naked_immediate (reduce_imms acc)
           (Keyed.dedup inverse_relation_eqb (rev (reduce_relations acc))))).
    { exists i. split; [ exact Hv | ]. split; [ exact Hnumbers | ].
      cbn [inverse_relations]. intros relation n Hin. apply Hkept.
      apply in_rev.
      apply (keyed_among_in _ inverse_relation_eqb inverse_relation_eqb_eq).
      rewrite <- (keyed_among_dedup _ inverse_relation_eqb
                    inverse_relation_eqb_eq).
      apply (keyed_among_in _ inverse_relation_eqb inverse_relation_eqb_eq).
      exact Hin. }
    destruct (reduce_imms acc) as [ [ | number numbers ] | ] eqn:Hreduce.
    + destruct Hnumbers.
    + eexists. split; [ reflexivity | exact Hsatisfied ].
    + eexists. split; [ reflexivity | exact Hsatisfied ].
  - exists head. split; [ reflexivity | ].
    exists i. split; [ exact Hv | ]. split; [ exact Himms | exact Hrels ].
Qed.

(* ======================================================================== *)
(* Extending an environment, and a valuation                                *)
(* ======================================================================== *)

Lemma keeps_names_refl (tenv : Typing_env.t) : keeps_names tenv tenv.
Proof.
  intros n ty binding_time_and_mode Hlookup. exists ty.
  split; [ exact Hlookup | reflexivity ].
Qed.

Lemma keeps_names_trans (tenv1 tenv2 tenv3 : Typing_env.t) :
  keeps_names tenv1 tenv2 -> keeps_names tenv2 tenv3 ->
  keeps_names tenv1 tenv3.
Proof.
  intros H12 H23 n ty binding_time_and_mode Hlookup.
  destruct (H12 n ty binding_time_and_mode Hlookup) as (ty2 & Hlookup2 & Hk2).
  destruct (H23 n ty2 binding_time_and_mode Hlookup2)
    as (ty3 & Hlookup3 & Hk3).
  exists ty3. split; [ exact Hlookup3 | congruence ].
Qed.

Lemma meet_env_extends_refl (menv : Meet_env.t) : meet_env_extends menv menv.
Proof.
  repeat split; try reflexivity.
  - intros s kind Hkind. exact Hkind.
  - intros x Hamong. exact Hamong.
  - apply keeps_names_refl.
Qed.

Lemma meet_env_extends_trans (menv1 menv2 menv3 : Meet_env.t) :
  meet_env_extends menv1 menv2 -> meet_env_extends menv2 menv3 ->
  meet_env_extends menv1 menv3.
Proof.
  intros (Hs12 & Hp12 & Hm12 & Hd12 & Hkind12 & Ha12 & Hv12 & Hk12)
    (Hs23 & Hp23 & Hm23 & Hd23 & Hkind23 & Ha23 & Hv23 & Hk23).
  repeat split; try congruence.
  - intros s kind Hkind. exact (Hkind23 s kind (Hkind12 s kind Hkind)).
  - intros x Hamong. apply Hv23, Hv12, Hamong.
  - exact (keeps_names_trans _ _ _ Hk12 Hk23).
Qed.

(* A variable that an environment knows is known to one that keeps its
   names. *)
Lemma keeps_names_mem (tenv tenv' : Typing_env.t) (x : variable) :
  keeps_names tenv tenv' ->
  Typing_env.mem None tenv (Name_var x) = true ->
  Typing_env.mem None tenv' (Name_var x) = true.
Proof.
  intros Hkeeps. unfold Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv))
    as [[ty binding_time_and_mode] | ] eqn:Hlookup; [ | discriminate ].
  intros _.
  destruct (Hkeeps _ _ _ Hlookup) as (ty' & Hlookup' & _).
  rewrite Hlookup'. reflexivity.
Qed.

Lemma extended_from_refl (tenv : Typing_env.t) (valuation : env) :
  extended_from tenv valuation valuation.
Proof. intros x _. reflexivity. Qed.

Lemma extended_from_trans (tenv tenv' : Typing_env.t)
    (valuation valuation' valuation'' : env) :
  keeps_names tenv tenv' ->
  extended_from tenv valuation valuation' ->
  extended_from tenv' valuation' valuation'' ->
  extended_from tenv valuation valuation''.
Proof.
  intros Hkeeps H1 H2 x Hmem.
  rewrite (H2 x (keeps_names_mem _ _ _ Hkeeps Hmem)). exact (H1 x Hmem).
Qed.

(* The variables that a meet defines are new: nothing is defined where
   the environment is the same, and what two meets define, one after
   the other, is new. *)
Lemma new_names_avoided_refl (menv : Meet_env.t) :
  new_names_avoided menv menv.
Proof.
  intros x Hknown Hunknown. rewrite Hknown in Hunknown.
  discriminate Hunknown.
Qed.

(* A variable that the first environment knows has in the second the
   name mode that it had there, where the least binding time is the
   same. *)
Lemma keeps_names_mode (tenv tenv' : Typing_env.t)
    (mode : option Name_mode.t) (x : variable) :
  keeps_names tenv tenv' ->
  Typing_env.min_binding_time tenv' = Typing_env.min_binding_time tenv ->
  Typing_env.mem None tenv (Name_var x) = true ->
  Typing_env.mem mode tenv' (Name_var x)
  = Typing_env.mem mode tenv (Name_var x).
Proof.
  intros Hkeeps Hmin Hknown. unfold Typing_env.mem in *.
  destruct
    (Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv))
    as [[ty binding_time_and_mode] | ] eqn:Hlookup; [ | discriminate Hknown ].
  destruct (Hkeeps _ _ _ Hlookup) as (ty' & Hlookup' & _).
  rewrite Hlookup', Hmin. reflexivity.
Qed.

Lemma meet_env_extends_mode (menv menv' : Meet_env.t)
    (mode : option Name_mode.t) (x : variable) :
  meet_env_extends menv menv' ->
  Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = true ->
  Typing_env.mem mode (Meet_env.typing_env menv') (Name_var x)
  = Typing_env.mem mode (Meet_env.typing_env menv) (Name_var x).
Proof.
  intros (_ & _ & Hmin & _ & _ & _ & _ & Hkeeps).
  exact (keeps_names_mode _ _ mode x Hkeeps Hmin).
Qed.

Lemma new_names_avoided_trans (menv1 menv2 menv3 : Meet_env.t) :
  meet_env_extends menv1 menv2 ->
  meet_env_extends menv2 menv3 ->
  new_names_avoided menv1 menv2 ->
  new_names_avoided menv2 menv3 ->
  new_names_avoided menv1 menv3.
Proof.
  intros (_ & _ & _ & _ & _ & _ & Havoid12 & _) Hextends23 Hnames12 Hnames23
    x Hknown3 Hunknown1.
  pose proof Hextends23 as (_ & _ & _ & _ & _ & _ & Havoid23 & _).
  destruct (Typing_env.mem None (Meet_env.typing_env menv2) (Name_var x))
    eqn:Hknown2.
  - destruct (Hnames12 x Hknown2 Hunknown1) as (Hnot & Hin & Hterms).
    split; [ exact Hnot | ]. split; [ exact (Havoid23 x Hin) | ].
    rewrite (meet_env_extends_mode _ _ _ x Hextends23 Hknown2). exact Hterms.
  - destruct (Hnames23 x Hknown3 Hknown2) as (Hnot & Hin & Hterms).
    split; [ | split; [ exact Hin | exact Hterms ] ].
    destruct (among x (Meet_env.names_to_avoid menv1)) eqn:Havoided;
      [ | reflexivity ].
    rewrite (Havoid12 x Havoided) in Hnot. discriminate Hnot.
Qed.

Lemma meet_env_extends_keeps_names (menv menv' : Meet_env.t) :
  meet_env_extends menv menv' ->
  keeps_names (Meet_env.typing_env menv) (Meet_env.typing_env menv').
Proof. intros (_ & _ & _ & _ & _ & _ & _ & Hkeeps). exact Hkeeps. Qed.

(* ======================================================================== *)
(* The meet of a part of a type                                             *)
(* ======================================================================== *)

Lemma extract_value_map {A B : Type} (f : A -> B) (r : meet_return_value A)
    (a1 a2 : A) :
  extract_value (map_return_value f r) (f a1) (f a2)
  = f (extract_value r a1 a2).
Proof. destruct r; reflexivity. Qed.

Lemma map_result_ok_cases {A B : Type} (f : A -> B)
    (result : Meet_result.t A) (r' : meet_return_value B)
    (menv' : Meet_env.t) :
  map_result f result = Meet_result.Ok r' menv' ->
  ∃ r : meet_return_value A,
    result = Meet_result.Ok r menv' ∧ r' = map_return_value f r.
Proof.
  destruct result as [ r0 | r env ]; [ discriminate | ].
  destruct r; cbn; intros [= <- <-]; eexists; split; reflexivity.
Qed.

Lemma map_result_ok_eq {A B : Type} (f : A -> B) (r : meet_return_value A)
    (menv : Meet_env.t) :
  map_result f (Meet_result.Ok r menv)
  = Meet_result.Ok (map_return_value f r) menv.
Proof. destruct r; reflexivity. Qed.

Lemma map_result_ok_for :
  for_validity map_result_ok_statement.
Proof.
  intros A B fits_a satisfied_a fits_b satisfied_b f menv a1
    a2 result Hfits Hsatisfied [Hkeeps Hsound].
  split.
  - intros r' menv' Hresult.
    destruct (map_result_ok_cases f result r' menv' Hresult)
      as (r & -> & ->).
    destruct (Hkeeps r menv' eq_refl) as (Hextends & Henv & Ha & Hnames).
    split; [ exact Hextends | ]. split; [ exact Henv | ].
    split; [ | exact Hnames ].
    rewrite extract_value_map. apply Hfits, Ha.
  - intros statics valuation Hstatics Henv H1 H2.
    apply Hsatisfied in H1. apply Hsatisfied in H2.
    destruct (Hsound statics valuation Hstatics Henv H1 H2)
      as (r & menv' & valuation' & -> & Hextended & Henv' & Ha & Hvalued).
    exists (map_return_value f r), menv', valuation'.
    split; [ apply map_result_ok_eq | ]. split; [ exact Hextended | ].
    split; [ exact Henv' | ]. split; [ | exact Hvalued ].
    rewrite extract_value_map. apply Hsatisfied, Ha.
Qed.

(* A call that answers bottom is right if nothing has both inputs. *)
Lemma meet_call_ok_bottom {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) (menv : Meet_env.t)
    (a1 a2 : A) (r : meet_return_value unit) :
  (∀ (statics : static_env) (valuation : env),
     statics_valid statics ->
     typing_env_satisfied_strictly statics valuation
       (Meet_env.typing_env menv) ->
     satisfied statics valuation a1 -> satisfied statics valuation a2 ->
     False) ->
  meet_call_ok fits satisfied menv a1 a2
    (Meet_result.Bottom r).
Proof.
  intros Hnone. split.
  - intros r' menv' Hresult. discriminate Hresult.
  - intros statics valuation Hstatics Henv H1 H2.
    destruct (Hnone statics valuation Hstatics Henv H1 H2).
Qed.

(* A call that answers in the environment that it was given is right if
   what it answers fits and is true where both inputs are. *)
Lemma meet_call_ok_same_env {A : Type} (fits : Typing_env.t -> A -> Prop)
    (satisfied : static_env -> env -> A -> Prop) (menv : Meet_env.t)
    (a1 a2 : A) (r : meet_return_value A) :
  typing_env_fits (Meet_env.typing_env menv) ->
  fits (Meet_env.typing_env menv) (extract_value r a1 a2) ->
  (∀ (statics : static_env) (valuation : env),
     statics_valid statics ->
     satisfied statics valuation a1 -> satisfied statics valuation a2 ->
     satisfied statics valuation (extract_value r a1 a2)) ->
  meet_call_ok fits satisfied menv a1 a2
    (Meet_result.Ok r menv).
Proof.
  intros Henv Hfits Hsatisfied. split.
  - intros r' menv' [= <- <-].
    split; [ apply meet_env_extends_refl | ]. split; [ assumption | ].
    split; [ assumption | apply new_names_avoided_refl ].
  - intros statics valuation Hstatics Hvaluation H1 H2.
    exists r, menv, valuation. split; [ reflexivity | ].
    split; [ apply extended_from_refl | ]. split; [ exact Hvaluation | ].
    split; [ exact (Hsatisfied statics valuation Hstatics H1 H2) | ].
    intros cut_after Hvalued. exact Hvalued.
Qed.

Lemma meet_unknown_ok_for :
  for_validity meet_unknown_ok_statement.
Proof.
  intros A fits satisfied meet_contents contents_is_bottom
    Hbottom Hok menv x1 x2 result Henv Hfits1 Hfits2 Hmeet.
  unfold meet_unknown in Hmeet.
  destruct x1 as [ c1 | ], x2 as [ c2 | ].
  - destruct (contents_is_bottom c1) eqn:Hb1;
      destruct (contents_is_bottom c2) eqn:Hb2; cbn in Hmeet.
    + injection Hmeet as <-. apply meet_call_ok_bottom.
      intros statics valuation _ _ H1 _. exact (Hbottom _ _ _ Hb1 H1).
    + injection Hmeet as <-. apply meet_call_ok_bottom.
      intros statics valuation _ _ H1 _. exact (Hbottom _ _ _ Hb1 H1).
    + injection Hmeet as <-. apply meet_call_ok_bottom.
      intros statics valuation _ _ _ H2. exact (Hbottom _ _ _ Hb2 H2).
    + destruct (meet_contents menv c1 c2) as [ result0 | ] eqn:Hcontents;
        [ | discriminate ].
      injection Hmeet as <-.
      apply (map_result_ok_for A (Or_unknown.t A) fits satisfied
               (or_unknown_fits fits) (or_unknown_satisfied satisfied)
               Or_unknown.Known menv c1 c2 result0).
      * intros tenv a Ha. exact Ha.
      * intros statics valuation a. reflexivity.
      * exact (Hok menv c1 c2 result0 Henv Hfits1 Hfits2 Hcontents).
  - destruct (contents_is_bottom c1) eqn:Hb1; injection Hmeet as <-.
    + apply meet_call_ok_bottom.
      intros statics valuation _ _ H1 _. exact (Hbottom _ _ _ Hb1 H1).
    + apply meet_call_ok_same_env; [ exact Henv | exact Hfits1 | ].
      intros statics valuation _ H1 _. exact H1.
  - destruct (contents_is_bottom c2) eqn:Hb2; injection Hmeet as <-.
    + apply meet_call_ok_bottom.
      intros statics valuation _ _ _ H2. exact (Hbottom _ _ _ Hb2 H2).
    + apply meet_call_ok_same_env; [ exact Henv | exact Hfits2 | ].
      intros statics valuation _ _ H2. exact H2.
  - injection Hmeet as <-.
    apply meet_call_ok_same_env; [ exact Henv | exact I | ].
    intros statics valuation _ _ _. exact I.
Qed.

(* What [combine_meet_return_values] answers for a pair: each part of
   it is a part of an input or what the meet of that part answered. *)
Lemma combine_extract {A B C : Type} (rebuild : A -> B -> C)
    (res_a : meet_return_value A) (res_b : meet_return_value B)
    (left_a right_a : A) (left_b right_b : B) :
  ∃ (a : A) (b : B),
    extract_value
      (combine_meet_return_values res_a res_b
         (fun _ =>
            rebuild (extract_value res_a left_a right_a)
              (extract_value res_b left_b right_b)))
      (rebuild left_a left_b) (rebuild right_a right_b)
    = rebuild a b
    ∧ (a = left_a ∨ a = right_a ∨ a = extract_value res_a left_a right_a)
    ∧ (b = left_b ∨ b = right_b ∨ b = extract_value res_b left_b right_b).
Proof.
  destruct res_a, res_b; cbn; eexists; eexists;
    (split; [ reflexivity | ]); split; auto.
Qed.

Lemma combine_results2_ok_for :
  for_validity combine_results2_ok_statement.
Proof.
  intros A B C fits_a satisfied_a fits_b satisfied_b fits_c
    satisfied_c meet_a meet_b rebuild menv left_a right_a left_b right_b
    result tenv [Hstable_a1 Hstable_a2] [Hstable_b1 Hstable_b2] Hok_a Hok_b
    Hfits_c Hsatisfied_c Henv Hla Hra Hlb Hrb Hmeet.
  unfold combine_results2 in Hmeet.
  destruct (meet_a menv left_a right_a) as [ result_a | ] eqn:Hmeet_a;
    [ | discriminate ].
  destruct (Hok_a menv left_a right_a result_a Henv Hla Hra Hmeet_a)
    as [Hkeeps_a Hsound_a].
  cbn in Hmeet.
  destruct result_a as [ r_a | res_a menv1 ].
  { injection Hmeet as <-. apply meet_call_ok_bottom.
    intros statics valuation Hstatics Hvaluation H1 H2.
    apply Hsatisfied_c in H1. apply Hsatisfied_c in H2.
    destruct (Hsound_a statics valuation Hstatics Hvaluation (proj1 H1)
                (proj1 H2))
      as (r & menv' & valuation' & Hresult & _).
    discriminate Hresult. }
  destruct (Hkeeps_a res_a menv1 eq_refl)
    as (Hextends1 & Henv1 & Hanswer_a & Hnames1).
  destruct (meet_b menv1 left_b right_b) as [ result_b | ] eqn:Hmeet_b;
    [ | discriminate ].
  assert (Hlb1 := Hstable_b1 menv menv1 left_b Hextends1 Hlb).
  assert (Hrb1 := Hstable_b1 menv menv1 right_b Hextends1 Hrb).
  destruct (Hok_b menv1 left_b right_b result_b Henv1 Hlb1 Hrb1 Hmeet_b)
    as [Hkeeps_b Hsound_b].
  cbn in Hmeet.
  destruct result_b as [ r_b | res_b menv2 ].
  { injection Hmeet as <-. apply meet_call_ok_bottom.
    intros statics valuation Hstatics Hvaluation H1 H2.
    apply Hsatisfied_c in H1. apply Hsatisfied_c in H2.
    destruct (Hsound_a statics valuation Hstatics Hvaluation (proj1 H1)
                (proj1 H2))
      as (r & menv' & valuation1 & Hresult & Hextended1 & Hvaluation1 & _).
    injection Hresult as <- <-.
    destruct (Hsound_b statics valuation1 Hstatics Hvaluation1
                (Hstable_b2 statics _ valuation valuation1 left_b Hlb
                   Hextended1 (proj2 H1))
                (Hstable_b2 statics _ valuation valuation1 right_b Hrb
                   Hextended1 (proj2 H2)))
      as (r & menv' & valuation' & Hresult & _).
    discriminate Hresult. }
  injection Hmeet as <-.
  destruct (Hkeeps_b res_b menv2 eq_refl)
    as (Hextends2 & Henv2 & Hanswer_b & Hnames2).
  assert (Hextends := meet_env_extends_trans _ _ _ Hextends1 Hextends2).
  destruct (combine_extract rebuild res_a res_b left_a right_a left_b right_b)
    as (a & b & Hextract & Ha & Hb).
  split.
  - intros r menv' [= <- <-].
    split; [ exact Hextends | ]. split; [ exact Henv2 | ].
    split.
    2: exact (new_names_avoided_trans _ _ _ Hextends1 Hextends2 Hnames1
                Hnames2).
    rewrite Hextract. apply Hfits_c.
    + destruct Ha as [ -> | [ -> | -> ] ].
      * exact (Hstable_a1 menv menv2 left_a Hextends Hla).
      * exact (Hstable_a1 menv menv2 right_a Hextends Hra).
      * exact (Hstable_a1 menv1 menv2 _ Hextends2 Hanswer_a).
    + destruct Hb as [ -> | [ -> | -> ] ].
      * exact (Hstable_b1 menv menv2 left_b Hextends Hlb).
      * exact (Hstable_b1 menv menv2 right_b Hextends Hrb).
      * exact Hanswer_b.
  - intros statics valuation Hstatics Hvaluation H1 H2.
    apply Hsatisfied_c in H1. apply Hsatisfied_c in H2.
    destruct H1 as [Hsla Hslb], H2 as [Hsra Hsrb].
    destruct (Hsound_a statics valuation Hstatics Hvaluation Hsla Hsra)
      as (r & menv' & valuation1 & Hresult & Hextended1 & Hvaluation1
          & Hsanswer_a & Hvalued1).
    injection Hresult as <- <-.
    destruct (Hsound_b statics valuation1 Hstatics Hvaluation1
                (Hstable_b2 statics _ valuation valuation1 left_b Hlb
                   Hextended1 Hslb)
                (Hstable_b2 statics _ valuation valuation1 right_b Hrb
                   Hextended1 Hsrb))
      as (r & menv' & valuation2 & Hresult & Hextended2 & Hvaluation2
          & Hsanswer_b & Hvalued2).
    injection Hresult as <- <-.
    assert (Hextended :
      extended_from (Meet_env.typing_env menv) valuation valuation2).
    { exact (extended_from_trans _ _ _ _ _
               (meet_env_extends_keeps_names _ _ Hextends1) Hextended1
               Hextended2). }
    eexists. exists menv2, valuation2. split; [ reflexivity | ].
    split; [ exact Hextended | ]. split; [ exact Hvaluation2 | ].
    split; [ | intros cut_after Hvalued;
               exact (Hvalued2 cut_after (Hvalued1 cut_after Hvalued)) ].
    rewrite Hextract. apply Hsatisfied_c. split.
    + destruct Ha as [ -> | [ -> | -> ] ].
      * exact (Hstable_a2 statics _ valuation valuation2 left_a Hla Hextended
                 Hsla).
      * exact (Hstable_a2 statics _ valuation valuation2 right_a Hra
                 Hextended Hsra).
      * exact (Hstable_a2 statics _ valuation1 valuation2 _ Hanswer_a
                 Hextended2 Hsanswer_a).
    + destruct Hb as [ -> | [ -> | -> ] ].
      * exact (Hstable_b2 statics _ valuation valuation2 left_b Hlb Hextended
                 Hslb).
      * exact (Hstable_b2 statics _ valuation valuation2 right_b Hrb
                 Hextended Hsrb).
      * exact Hsanswer_b.
Qed.

(* ======================================================================== *)
(* Sets                                                                     *)
(* ======================================================================== *)

Lemma set_meet_cases_for :
  for_validity set_meet_cases_statement.
Proof.
  intros K B eqb of_set env s1 s2 Heqb. unfold set_meet.
  assert (Hin := keyed_among_in K eqb Heqb).
  destruct (Keyed.all_among eqb s1 s2) eqn:H12;
    destruct (Keyed.all_among eqb s2 s1) eqn:H21.
  - split; [ reflexivity | ]. exists s1. split; [ reflexivity | ].
    split; [ | intros Hne _; exact Hne ].
    intros key. split; [ | tauto ]. intros Hkey. split; [ exact Hkey | ].
    apply Hin. apply (proj1 (keyed_all_among K eqb Heqb s1 s2) H12).
    apply Hin, Hkey.
  - split; [ reflexivity | ]. exists s1. split; [ reflexivity | ].
    split; [ | intros Hne _; exact Hne ].
    intros key. split; [ | tauto ]. intros Hkey. split; [ exact Hkey | ].
    apply Hin. apply (proj1 (keyed_all_among K eqb Heqb s1 s2) H12).
    apply Hin, Hkey.
  - split; [ reflexivity | ]. exists s2. split; [ reflexivity | ].
    split; [ | intros _ Hne; exact Hne ].
    intros key. split; [ | tauto ]. intros Hkey. split; [ | exact Hkey ].
    apply Hin. apply (proj1 (keyed_all_among K eqb Heqb s2 s1) H21).
    apply Hin, Hkey.
  - assert (Hinter :
      ∀ key : K, key ∈ Keyed.inter eqb s1 s2 <-> key ∈ s1 ∧ key ∈ s2).
    { intros key. rewrite <- !Hin, (keyed_among_inter K eqb Heqb).
      apply andb_true_iff. }
    destruct (Keyed.inter eqb s1 s2) as [ | first rest ] eqn:Hs; cbn.
    + intros key Hkey. apply Hinter in Hkey. destruct Hkey.
    + split; [ reflexivity | ]. exists (first :: rest).
      split; [ reflexivity | ]. split; [ exact Hinter | ].
      intros _ _. discriminate.
Qed.

Lemma naked_number_meet_ok_for :
  for_validity naked_number_meet_ok_statement.
Proof.
  intros k v menv t1 t2 result Henv Hf1 Hf2 [= <-].
  unfold meet_head_of_kind_naked_number.
  assert (Hcases :=
    set_meet_cases_for Z (list Z) Z.eqb (fun s => s) menv t1 t2
      z_eqb_decides_equality).
  assert (Hboth :
    naked_numbers_satisfied k v t1 -> naked_numbers_satisfied k v t2 ->
    ∃ (c : const) (n : Z),
      v = Value_const c
      ∧ kind_of_const c = Naked_number k
      ∧ number_of_const c = Some n
      ∧ n ∈ t1 ∧ n ∈ t2).
  { intros (c & n & -> & Hk & Hn & Hin1) (c' & n' & [= <-] & _ & Hn' & Hin2).
    rewrite Hn in Hn'. injection Hn' as <-. exists c, n. auto. }
  destruct (set_meet Z.eqb (fun s : list Z => s) menv t1 t2)
    as [ r | r env' ].
  - apply meet_call_ok_bottom. intros statics valuation _ _ H1 H2.
    destruct (Hboth H1 H2) as (c & n & _ & _ & _ & Hin1 & Hin2).
    exact (Hcases n (conj Hin1 Hin2)).
  - destruct Hcases as (-> & s & Hextract & Hmem & Hne).
    apply meet_call_ok_same_env; [ exact Henv | | ].
    + rewrite Hextract. exact (Hne Hf1 Hf2).
    + intros statics valuation _ H1 H2. rewrite Hextract.
      destruct (Hboth H1 H2) as (c & n & Hv & Hk & Hn & Hin1 & Hin2).
      exists c, n. split; [ exact Hv | ]. split; [ exact Hk | ].
      split; [ exact Hn | ]. apply Hmem. split; assumption.
Qed.

(* ======================================================================== *)
(* A thing that may be unknown or bottom                                    *)
(* ======================================================================== *)

Lemma meet_or_unknown_or_bottom_ok_for :
  for_validity meet_or_unknown_or_bottom_ok_statement.
Proof.
  intros A B fits satisfied inj meet_elt menv input1 input2
    result tenv Helt Hbottom Henv Hfits1 Hfits2 Hmeet.
  unfold meet_or_unknown_or_bottom in Hmeet.
  destruct input1 as [ | a1 | ], input2 as [ | a2 | ];
    try (injection Hmeet as <-).
  all: try (apply meet_call_ok_bottom;
            intros statics valuation _ _ H1 H2;
            first [ exact (Hbottom _ _ H1) | exact (Hbottom _ _ H2) ]).
  all: try (apply meet_call_ok_same_env;
            [ exact Henv
            | first [ exact Hfits1 | exact Hfits2 ]
            | intros statics valuation _ H1 H2;
              first [ exact H1 | exact H2 ] ]).
  exact (Helt a1 a2 result Hfits1 Hfits2 Hmeet).
Qed.

(* ======================================================================== *)
(* Products                                                                 *)
(* ======================================================================== *)

(* The step of [meet_mapping], as a function of its own. *)
Definition mapping_step {K A : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A)) (key : K)
    (data_left_opt data_right_opt : option A) (state : mapping_state K A)
    : mapping_state K A :=
  match state with
  | None => None
  | Some Or_bottom.Bottom => Some Or_bottom.Bottom
  | Some (Or_bottom.Ok acc) =>
    match data_left_opt, data_right_opt with
    | None, None => None
    | Some data_left, None =>
      Some
        (Or_bottom.Ok
           {| all_left := all_left acc;
              all_right := false;
              mapping := (key, data_left) :: mapping acc;
              mapping_env := mapping_env acc |})
    | None, Some data_right =>
      Some
        (Or_bottom.Ok
           {| all_left := false;
              all_right := all_right acc;
              mapping := (key, data_right) :: mapping acc;
              mapping_env := mapping_env acc |})
    | Some data_left, Some data_right =>
      match meet_data (mapping_env acc) data_left data_right with
      | None => None
      | Some (Meet_result.Bottom _) => Some Or_bottom.Bottom
      | Some (Meet_result.Ok res env) =>
        let result (all_left all_right : bool) (data : A)
            : mapping_state K A :=
          Some
            (Or_bottom.Ok
               {| all_left := all_left;
                  all_right := all_right;
                  mapping := (key, data) :: mapping acc;
                  mapping_env := env |})
        in
        match res with
        | Both_inputs => result (all_left acc) (all_right acc) data_left
        | Left_input => result (all_left acc) false data_left
        | Right_input => result false (all_right acc) data_right
        | New_result data => result false false data
        end
      end
    end
  end.

Lemma meet_mapping_unfold {K A M : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (fold2 :
      (K -> option A -> option A -> mapping_state K A -> mapping_state K A) ->
      M -> M -> mapping_state K A -> mapping_state K A)
    (env : Meet_env.t) (left right : M) (rebuild : list (K * A) -> M) :
  meet_mapping meet_data fold2 env left right rebuild
  = match
      fold2 (mapping_step meet_data) left right
        (Some
           (Or_bottom.Ok
              {| all_left := true;
                 all_right := true;
                 mapping := [];
                 mapping_env := env |}))
    with
    | None => None
    | Some Or_bottom.Bottom => Some (Meet_result.Bottom (New_result tt))
    | Some (Or_bottom.Ok res) =>
      let result :=
        match all_left res, all_right res with
        | true, true => Both_inputs
        | true, false => Left_input
        | false, true => Right_input
        | false, false => New_result (rebuild (mapping res))
        end
      in
      Some (Meet_result.Ok result (mapping_env res))
    end.
Proof. reflexivity. Qed.

(* What a step that answers makes of the state: one component more,
   which is that of the one input that has it, or what the meet of the
   two answered. *)
Lemma mapping_step_ok {K A : Type}
    (meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A)) (key : K)
    (data_left_opt data_right_opt : option A) (acc acc' : mapping_acc K A) :
  mapping_step meet_data key data_left_opt data_right_opt
    (Some (Or_bottom.Ok acc))
  = Some (Or_bottom.Ok acc') ->
  (∃ a : A,
     ((data_left_opt = Some a ∧ data_right_opt = None)
      ∨ (data_left_opt = None ∧ data_right_opt = Some a))
     ∧ mapping acc' = (key, a) :: mapping acc
     ∧ mapping_env acc' = mapping_env acc)
  ∨ ∃ (a b : A) (res : meet_return_value A),
      data_left_opt = Some a
      ∧ data_right_opt = Some b
      ∧ meet_data (mapping_env acc) a b
        = Some (Meet_result.Ok res (mapping_env acc'))
      ∧ mapping acc' = (key, extract_value res a b) :: mapping acc.
Proof.
  unfold mapping_step.
  destruct data_left_opt as [ a | ], data_right_opt as [ b | ].
  - destruct (meet_data (mapping_env acc) a b) as [ [ r | res env ] | ]
      eqn:Hmeet; [ discriminate | | discriminate ].
    intros Hstep. right. exists a, b, res.
    destruct res; injection Hstep as <-; cbn; repeat split; exact Hmeet.
  - intros [= <-]. left. exists a. cbn. auto.
  - intros [= <-]. left. exists b. cbn. auto.
  - discriminate.
Qed.

Lemma mapping_fold_none {K A : Type} (step : mapping_state K A -> K -> _)
    (keys : list K) :
  (∀ key : K, step None key = None) ->
  List.fold_left step keys None = None.
Proof.
  intros Hstep. induction keys as [ | key keys IH ]; [ reflexivity | ].
  cbn. rewrite Hstep. exact IH.
Qed.

Lemma mapping_fold_bottom {K A : Type}
    (step : mapping_state K A -> K -> mapping_state K A) (keys : list K) :
  (∀ key : K, step (Some Or_bottom.Bottom) key = Some Or_bottom.Bottom) ->
  List.fold_left step keys (Some Or_bottom.Bottom) = Some Or_bottom.Bottom.
Proof.
  intros Hstep. induction keys as [ | key keys IH ]; [ reflexivity | ].
  cbn. rewrite Hstep. exact IH.
Qed.

Lemma keeps_equations_valued_refl (tenv : Typing_env.t) (valuation : env) :
  keeps_equations_valued tenv tenv valuation valuation.
Proof. intros cut_after Hvalued. exact Hvalued. Qed.

Lemma keeps_equations_valued_trans (tenv1 tenv2 tenv3 : Typing_env.t)
    (valuation1 valuation2 valuation3 : env) :
  keeps_equations_valued tenv1 tenv2 valuation1 valuation2 ->
  keeps_equations_valued tenv2 tenv3 valuation2 valuation3 ->
  keeps_equations_valued tenv1 tenv3 valuation1 valuation3.
Proof.
  intros H12 H23 cut_after Hvalued. exact (H23 _ (H12 _ Hvalued)).
Qed.

Section Mapping.

Context {K A : Type}.
Variable fits : Typing_env.t -> A -> Prop.
Variable satisfied : K -> static_env -> env -> A -> Prop.
Variable meet_data : Meet_env.t -> A -> A -> option (Meet_result.t A).
Variable get_left get_right : K -> option A.
Variable menv : Meet_env.t.

Hypothesis Hstable : ∀ key : K, part_stable fits (satisfied key).
Hypothesis Hok :
  ∀ key : K, part_meet_ok fits (satisfied key) meet_data.
Hypothesis Hleft :
  ∀ (key : K) (a : A),
    get_left key = Some a -> fits (Meet_env.typing_env menv) a.
Hypothesis Hright :
  ∀ (key : K) (a : A),
    get_right key = Some a -> fits (Meet_env.typing_env menv) a.

Let step (state : mapping_state K A) (key : K) : mapping_state K A :=
  mapping_step meet_data key (get_left key) (get_right key) state.

(* What the fold keeps, whatever the values. *)
Let keeps (done : list K) (acc : mapping_acc K A) : Prop :=
  meet_env_extends menv (mapping_env acc)
  ∧ typing_env_fits (Meet_env.typing_env (mapping_env acc))
  ∧ new_names_avoided menv (mapping_env acc)
  ∧ map fst (mapping acc) = rev done
  ∧ ∀ (key : K) (a : A),
      (key, a) ∈ mapping acc ->
      fits (Meet_env.typing_env (mapping_env acc)) a.

(* What it keeps of a valuation [valuation0] that satisfied the first
   environment and made the components of both inputs true. *)
Let sound (statics : static_env) (valuation0 valuation : env)
    (acc : mapping_acc K A) : Prop :=
  extended_from (Meet_env.typing_env menv) valuation0 valuation
  ∧ typing_env_satisfied_strictly statics valuation
      (Meet_env.typing_env (mapping_env acc))
  ∧ keeps_equations_valued (Meet_env.typing_env menv)
      (Meet_env.typing_env (mapping_env acc)) valuation0 valuation
  ∧ ∀ (key : K) (a : A),
      (key, a) ∈ mapping acc -> satisfied key statics valuation a.

Lemma mapping_fold_ok (todo : list K) :
  ∀ (done : list K) (acc : mapping_acc K A) (final : mapping_state K A),
    keeps done acc ->
    List.fold_left step todo (Some (Or_bottom.Ok acc)) = final ->
    (∀ acc' : mapping_acc K A,
       final = Some (Or_bottom.Ok acc') -> keeps (done ++ todo) acc')
    ∧ ∀ (statics : static_env) (valuation0 valuation : env),
        statics_valid statics ->
        (∀ (key : K) (a : A),
           get_left key = Some a -> satisfied key statics valuation0 a) ->
        (∀ (key : K) (a : A),
           get_right key = Some a -> satisfied key statics valuation0 a) ->
        sound statics valuation0 valuation acc ->
        final <> None ->
        ∃ (acc' : mapping_acc K A) (valuation' : env),
          final = Some (Or_bottom.Ok acc')
          ∧ sound statics valuation0 valuation' acc'.
Proof.
  induction todo as [ | key todo IH ];
    intros done acc final Hkeeps Hfold.
  - cbn in Hfold. subst final. rewrite app_nil_r. split.
    + intros acc' [= <-]. exact Hkeeps.
    + intros statics valuation0 valuation Hstatics _ _ Hsound _.
      exists acc, valuation. split; [ reflexivity | exact Hsound ].
  - cbn [List.fold_left] in Hfold.
    destruct Hkeeps as (Hextends & Henv & Hnames & Hkeys & Hfits).
    destruct (step (Some (Or_bottom.Ok acc)) key)
      as [ [ acc1 | ] | ] eqn:Hstep.
    + (* The step answered. *)
      assert (Hkeeps1 : keeps (done ++ [key]) acc1
        ∧ ∀ (statics : static_env) (valuation0 valuation : env),
            statics_valid statics ->
            (∀ (key : K) (a : A),
               get_left key = Some a -> satisfied key statics valuation0 a) ->
            (∀ (key : K) (a : A),
               get_right key = Some a -> satisfied key statics valuation0 a)
            ->
            sound statics valuation0 valuation acc ->
            ∃ valuation1 : env, sound statics valuation0 valuation1 acc1).
      { destruct (mapping_step_ok meet_data key _ _ acc acc1 Hstep)
          as [ (a & Hside & Hmapping & Henv1)
             | (a & b & res & Hl & Hr & Hmeet & Hmapping) ].
        - (* One input has the component. *)
          assert (Ha : fits (Meet_env.typing_env menv) a).
          { destruct Hside as [ [Hl _] | [_ Hr] ];
              [ exact (Hleft key a Hl) | exact (Hright key a Hr) ]. }
          split.
          + unfold keeps. rewrite Henv1, Hmapping. cbn [map fst].
            rewrite rev_app_distr. cbn [rev app].
            split; [ exact Hextends | ]. split; [ exact Henv | ].
            split; [ exact Hnames | ].
            split; [ rewrite Hkeys; reflexivity | ].
            intros key' a' [ [= <- <-] | Hin ]; [ | exact (Hfits _ _ Hin) ].
            exact (proj1 (Hstable key) menv (mapping_env acc) a Hextends Ha).
          + intros statics valuation0 valuation Hstatics Hsl Hsr
              (Hextended & Hvaluation & Hvalued & Hsat).
            exists valuation. unfold sound. rewrite Henv1, Hmapping.
            split; [ exact Hextended | ]. split; [ exact Hvaluation | ].
            split; [ exact Hvalued | ].
            intros key' a' [ [= <- <-] | Hin ]; [ | exact (Hsat _ _ Hin) ].
            apply (proj2 (Hstable key) statics _ valuation0 valuation a Ha
                     Hextended).
            destruct Hside as [ [Hl _] | [_ Hr] ];
              [ exact (Hsl key a Hl) | exact (Hsr key a Hr) ].
        - (* Both have it: the meet of the two. *)
          assert (Ha : fits (Meet_env.typing_env (mapping_env acc)) a).
          { exact (proj1 (Hstable key) menv _ a Hextends (Hleft key a Hl)). }
          assert (Hb : fits (Meet_env.typing_env (mapping_env acc)) b).
          { exact (proj1 (Hstable key) menv _ b Hextends (Hright key b Hr)). }
          destruct (Hok key (mapping_env acc) a b _ Henv Ha Hb Hmeet)
            as [Hkeeps_data Hsound_data].
          destruct (Hkeeps_data res (mapping_env acc1) eq_refl)
            as (Hextends1 & Henv1 & Hanswer & Hnames1).
          split.
          + unfold keeps. rewrite Hmapping. cbn [map fst].
            rewrite rev_app_distr. cbn [rev app].
            split.
            { exact (meet_env_extends_trans _ _ _ Hextends Hextends1). }
            split; [ exact Henv1 | ].
            split.
            { exact (new_names_avoided_trans _ _ _ Hextends Hextends1 Hnames
                       Hnames1). }
            split; [ rewrite Hkeys; reflexivity | ].
            intros key' a' [ [= <- <-] | Hin ]; [ exact Hanswer | ].
            exact (proj1 (Hstable key') _ _ a' Hextends1 (Hfits _ _ Hin)).
          + intros statics valuation0 valuation Hstatics Hsl Hsr
              (Hextended & Hvaluation & Hvalued & Hsat).
            assert (Hsa : satisfied key statics valuation a).
            { exact (proj2 (Hstable key) statics _ valuation0 valuation a
                       (Hleft key a Hl) Hextended (Hsl key a Hl)). }
            assert (Hsb : satisfied key statics valuation b).
            { exact (proj2 (Hstable key) statics _ valuation0 valuation b
                       (Hright key b Hr) Hextended (Hsr key b Hr)). }
            destruct (Hsound_data statics valuation Hstatics Hvaluation Hsa Hsb)
              as (r & menv' & valuation1 & Hresult & Hextended1 & Hvaluation1
                  & Hsanswer & Hvalued1).
            injection Hresult as <- <-.
            exists valuation1. unfold sound. rewrite Hmapping.
            split.
            { exact (extended_from_trans _ _ _ _ _
                       (meet_env_extends_keeps_names _ _ Hextends) Hextended
                       Hextended1). }
            split; [ exact Hvaluation1 | ].
            split.
            { exact (keeps_equations_valued_trans _ _ _ _ _ _ Hvalued
                       Hvalued1). }
            intros key' a' [ [= <- <-] | Hin ]; [ exact Hsanswer | ].
            exact (proj2 (Hstable key') statics _ valuation valuation1 a'
                     (Hfits _ _ Hin) Hextended1 (Hsat _ _ Hin)). }
      destruct Hkeeps1 as [Hkeeps1 Hsound1].
      destruct (IH (done ++ [key]) acc1 final Hkeeps1 Hfold)
        as [IHkeeps IHsound].
      rewrite <- app_assoc in IHkeeps. cbn [app] in IHkeeps.
      split; [ exact IHkeeps | ].
      intros statics valuation0 valuation Hstatics Hsl Hsr Hsound Hfinal.
      destruct (Hsound1 statics valuation0 valuation Hstatics Hsl Hsr Hsound)
        as (valuation1 & Hsound_acc1).
      exact (IHsound statics valuation0 valuation1 Hstatics Hsl Hsr
               Hsound_acc1 Hfinal).
    + (* The step answered bottom: the meet of a component did. *)
      rewrite (mapping_fold_bottom step todo (fun _ => eq_refl)) in Hfold.
      subst final. split; [ intros acc' Hfinal; discriminate Hfinal | ].
      intros statics valuation0 valuation Hstatics Hsl Hsr
        (Hextended & Hvaluation & Hvalued & Hsat) _.
      exfalso. unfold step, mapping_step in Hstep.
      destruct (get_left key) as [ a | ] eqn:Hl;
        destruct (get_right key) as [ b | ] eqn:Hr; try discriminate Hstep.
      destruct (meet_data (mapping_env acc) a b) as [ [ r | res env ] | ]
        eqn:Hmeet; [ | destruct res; discriminate Hstep | discriminate Hstep ].
      assert (Ha : fits (Meet_env.typing_env (mapping_env acc)) a).
      { exact (proj1 (Hstable key) menv _ a Hextends (Hleft key a Hl)). }
      assert (Hb : fits (Meet_env.typing_env (mapping_env acc)) b).
      { exact (proj1 (Hstable key) menv _ b Hextends (Hright key b Hr)). }
      destruct (Hok key (mapping_env acc) a b _ Henv Ha Hb Hmeet)
        as [_ Hsound_data].
      destruct (Hsound_data statics valuation Hstatics Hvaluation
                  (proj2 (Hstable key) statics _ valuation0 valuation a
                     (Hleft key a Hl) Hextended (Hsl key a Hl))
                  (proj2 (Hstable key) statics _ valuation0 valuation b
                     (Hright key b Hr) Hextended (Hsr key b Hr)))
        as (r' & menv' & valuation1 & Hresult & _).
      discriminate Hresult.
    + (* The step stopped. *)
      rewrite (mapping_fold_none step todo (fun _ => eq_refl)) in Hfold.
      subst final. split; [ intros acc' Hfinal; discriminate Hfinal | ].
      intros statics valuation0 valuation Hstatics _ _ _ Hfinal.
      destruct Hfinal. reflexivity.
Qed.

End Mapping.

Lemma meet_mapping_ok_for :
  for_validity meet_mapping_ok_statement.
Proof.
  intros K A M fits satisfied meet_data fold2 keys get_left
    get_right menv left right rebuild result tenv Hfold2 Hstable Hok Henv
    Hleft Hright Hmeet.
  rewrite meet_mapping_unfold, Hfold2 in Hmeet.
  set (init :=
    {| all_left := true;
       all_right := true;
       mapping := @nil (K * A);
       mapping_env := menv |}) in Hmeet.
  assert (Hinit :
    meet_env_extends menv (mapping_env init)
    ∧ typing_env_fits (Meet_env.typing_env (mapping_env init))
    ∧ new_names_avoided menv (mapping_env init)
    ∧ map fst (mapping init) = rev []
    ∧ ∀ (key : K) (a : A),
        (key, a) ∈ mapping init ->
        fits (Meet_env.typing_env (mapping_env init)) a).
  { split; [ apply meet_env_extends_refl | ]. split; [ exact Henv | ].
    split; [ apply new_names_avoided_refl | ].
    split; [ reflexivity | ]. intros key a []. }
  destruct (mapping_fold_ok fits satisfied meet_data get_left
              get_right menv Hstable Hok Hleft Hright keys [] init _ Hinit
              eq_refl)
    as [Hkeeps Hsound].
  cbn [app] in Hkeeps.
  destruct
    (List.fold_left
       (fun state key =>
          mapping_step meet_data key (get_left key) (get_right key) state)
       keys (Some (Or_bottom.Ok init)))
    as [ [ res | ] | ] eqn:Hfinal; [ | | discriminate Hmeet ].
  - injection Hmeet as <-.
    destruct (Hkeeps res eq_refl)
      as (Hextends & Henv' & Hnames & Hkeys & Hfits).
    assert (Hnew :
      ∀ m : M,
        match all_left res, all_right res with
        | true, true => Both_inputs
        | true, false => Left_input
        | false, true => Right_input
        | false, false => New_result (rebuild (mapping res))
        end
        = New_result m ->
        m = rebuild (mapping res)).
    { intros m. destruct (all_left res), (all_right res); intros [= <-].
      reflexivity. }
    split.
    + intros r menv' [= <- <-].
      split; [ exact Hextends | ]. split; [ exact Henv' | ].
      split; [ exact Hnames | ].
      intros m Hm. exists (mapping res).
      split; [ exact (Hnew m Hm) | ]. split; [ exact Hkeys | exact Hfits ].
    + intros statics valuation Hstatics Hvaluation Hsl Hsr.
      destruct (Hsound statics valuation valuation Hstatics Hsl Hsr) as
        (acc' & valuation' & Hacc' & Hextended & Hvaluation' & Hvalued
         & Hsat).
      { split; [ apply extended_from_refl | ]. split; [ exact Hvaluation | ].
        split; [ apply keeps_equations_valued_refl | ]. intros key a []. }
      { discriminate. }
      injection Hacc' as <-.
      eexists. exists (mapping_env res), valuation'.
      split; [ reflexivity | ]. split; [ exact Hextended | ].
      split; [ exact Hvaluation' | ]. split; [ exact Hvalued | ].
      intros m Hm. exists (mapping res).
      split; [ exact (Hnew m Hm) | ]. split; [ exact Hkeys | ].
      intros key a Hin. split; [ exact (Hfits key a Hin) | ].
      exact (Hsat key a Hin).
  - injection Hmeet as <-. split.
    + intros r menv' Hresult. discriminate Hresult.
    + intros statics valuation Hstatics Hvaluation Hsl Hsr.
      destruct (Hsound statics valuation valuation Hstatics Hsl Hsr) as
        (acc' & valuation' & Hacc' & _).
      { split; [ apply extended_from_refl | ]. split; [ exact Hvaluation | ].
        split; [ apply keeps_equations_valued_refl | ]. intros key a []. }
      { discriminate. }
      discriminate Hacc'.
Qed.

End Validity.

Theorem reduce_head_of_kind_naked_immediate_sound :
  reduce_head_of_kind_naked_immediate_sound_statement.
Proof.
  first
    [ exact (@reduce_head_of_kind_naked_immediate_sound_for standard_validity)
    | exact reduce_head_of_kind_naked_immediate_sound_for ].
Qed.

Theorem map_result_ok :
  map_result_ok_statement.
Proof.
  first
    [ exact (@map_result_ok_for standard_validity)
    | exact map_result_ok_for ].
Qed.

Theorem meet_unknown_ok :
  meet_unknown_ok_statement.
Proof.
  first
    [ exact (@meet_unknown_ok_for standard_validity)
    | exact meet_unknown_ok_for ].
Qed.

Theorem combine_results2_ok :
  combine_results2_ok_statement.
Proof.
  first
    [ exact (@combine_results2_ok_for standard_validity)
    | exact combine_results2_ok_for ].
Qed.

Theorem set_meet_cases :
  set_meet_cases_statement.
Proof.
  first
    [ exact (@set_meet_cases_for standard_validity)
    | exact set_meet_cases_for ].
Qed.

Theorem naked_number_meet_ok :
  naked_number_meet_ok_statement.
Proof.
  first
    [ exact (@naked_number_meet_ok_for standard_validity)
    | exact naked_number_meet_ok_for ].
Qed.

Theorem meet_or_unknown_or_bottom_ok :
  meet_or_unknown_or_bottom_ok_statement.
Proof.
  first
    [ exact (@meet_or_unknown_or_bottom_ok_for standard_validity)
    | exact meet_or_unknown_or_bottom_ok_for ].
Qed.

Theorem meet_mapping_ok :
  meet_mapping_ok_statement.
Proof.
  first
    [ exact (@meet_mapping_ok_for standard_validity)
    | exact meet_mapping_ok_for ].
Qed.
