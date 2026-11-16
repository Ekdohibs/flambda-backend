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


(* Proofs of the statements of FlambdaMeetKeepsSpec.v. Nothing here
   needs review. The proofs of the files of the contract are read at
   the predicate that no static environment has
   (FlambdaMeetValidity.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec FlambdaPrimitiveProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvProofs FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetProofs FlambdaMeetStepsSpec
  FlambdaMeetEnvSpec FlambdaMeetEnvProofs FlambdaMeetHeadsSpec
  FlambdaMeetRowLikeSpec FlambdaMeetClosuresSpec FlambdaMeetFuelSpec
  FlambdaMeetExamplesSpec FlambdaMeetValidity FlambdaMeetHeadsProofs
  FlambdaMeetRowLikeProofs FlambdaMeetFuelProofs FlambdaMeetClosuresProofs
  FlambdaMeetExamplesProofs FlambdaMeetKeepsSpec.
Import ListNotations.

(* With no valid static environment, a join that has the half of its
   contract that is not about values has its contract. *)
Lemma join_keeps_sound_at_none (join : join_scoped_envs) :
  join_scoped_envs_keeps join ->
  join_scoped_envs_sound_for (V := no_validity) join.
Proof.
  intros Hjoin meet_head initial_env result scoped_envs. cbv zeta.
  intros _ Hkeeps Hfits Hscoped Hagree Hcall.
  split.
  - intros scoped_env statics valuation _ Hvalid. destruct Hvalid.
  - exact (Hjoin meet_head initial_env result scoped_envs Hkeeps Hfits
             Hscoped Hagree Hcall).
Qed.

Theorem meet_expanded_head_fuel_keeps :
  meet_expanded_head_fuel_keeps_statement.
Proof.
  intros join choices fuel Hjoin Hchoice Hdefine Hlift.
  pose proof (join_keeps_sound_at_none join Hjoin) as Hjoin'.
  exact
    (proj1
       (@meet_expanded_head_fuel_ok_for no_validity join choices Hchoice
          (fun mh Hkeeps Hsound _ v =>
             @meet_value_heads_ok_for no_validity join choices mh v Hjoin'
               Hdefine Hkeeps Hsound
               (@meet_row_like_for_blocks_ok_for no_validity join choices mh
                  v Hjoin' Hdefine Hlift Hkeeps Hsound)
               (@meet_row_like_for_closures_ok_for no_validity join choices
                  mh v Hjoin' Hdefine Hlift Hkeeps Hsound))
          fuel)).
Qed.

Theorem meet_expanded_head_repaired_keeps :
  meet_expanded_head_repaired_keeps_statement.
Proof.
  intros join fuel Hjoin.
  exact (meet_expanded_head_fuel_keeps join repaired_choices fuel Hjoin
           eq_refl eq_refl eq_refl).
Qed.

(* The join that forgets does not look at the meet that it is given:
   its contract is read at a meet that never answers, which has both
   contracts of a meet. *)
Theorem forgetful_join_keeps : forgetful_join_keeps_statement.
Proof.
  intros meet_head initial_env result scoped_envs. cbv zeta.
  intros _ Hfits Hscoped Hagree Hcall.
  assert (Hsound : meet_expanded_head_sound (fun _ _ _ => None)).
  { intros statics valuation menv expanded1 expanded2 v result' tenv _ _ _ _
      _ Hmeet.
    discriminate Hmeet. }
  assert (Hkeeps : meet_expanded_head_keeps (fun _ _ _ => None)).
  { intros menv menv' expanded1 expanded2 r tenv answered _ Hmeet.
    discriminate Hmeet. }
  exact (proj2
           (forgetful_join_sound (fun _ _ _ => None) initial_env result
              scoped_envs Hsound Hkeeps Hfits Hscoped Hagree Hcall)).
Qed.

Theorem meet_with_forgetful_join_keeps :
  meet_with_forgetful_join_keeps_statement.
Proof.
  intros fuel.
  exact (meet_expanded_head_repaired_keeps forgetful_join fuel
           forgetful_join_keeps).
Qed.

(* The two proofs below are the parts of the proof of
   [meet_of_fuel_contract] (FlambdaMeetFuelProofs.v) that are not
   about values. *)
Theorem meet_of_types_keeps : meet_of_types_keeps_statement.
Proof.
  intros join choices fuel Hjoin Hchoice Hdefine Hlift.
  pose proof
    (meet_expanded_head_fuel_keeps join choices fuel Hjoin Hchoice Hdefine
       Hlift)
    as Hkeeps.
  set (mh := Meet.meet_expanded_head join choices fuel) in *.
  intros tenv tenv' t1 t2 t (Hfits & Ht1 & Ht2 & Hkinds) Hmeet.
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
Qed.

Theorem meet_of_types_kinds : meet_of_types_kinds_statement.
Proof.
  intros join choices fuel. split.
  - intros tenv t1 t2 Hnot_bottom Hkinds. unfold Meet.meet.
    rewrite Hnot_bottom. unfold Meet_env.meet_type, Meet_env.meet.
    assert (Hneq : kind_eqb (kind t1) (kind t2) = false).
    { destruct (kind_eqb (kind t1) (kind t2)) eqn:H; [ | reflexivity ].
      apply kind_eqb_eq in H. contradiction. }
    rewrite Hneq. reflexivity.
  - intros tenv t1 t2 Hbottom. unfold Meet.meet. rewrite Hbottom.
    reflexivity.
Qed.

(* What a function of [Meet_env] keeps, read of the typing
   environments. *)
Lemma typing_env_kept_of_meet_env (tenv : Typing_env.t)
    (menv' : Meet_env.t) :
  meet_env_kept_by_meet (Meet_env.create tenv) menv' ->
  typing_env_kept tenv (Meet_env.final_typing_env menv').
Proof.
  intros [[Hextends Hfits'] _].
  assert (Htenv := meet_env_extends_tenv _ _ Hextends).
  change (Meet_env.typing_env (Meet_env.create tenv)) with tenv in Htenv.
  split; [ exact Hfits' | ]. split; [ | split ].
  - intros ty Hty. exact (tenv_extends_type_fits _ _ ty Htenv Hty).
  - exact (proj1 (proj2 (proj2 (proj2 (proj2 Htenv))))).
  - intros n Hmem. exact (tenv_extends_mem _ _ n Htenv Hmem).
Qed.

Theorem typing_env_functions_keep : typing_env_functions_keep_statement.
Proof.
  intros join choices fuel Hkeeps.
  set (mh := Meet.meet_expanded_head join choices fuel) in *.
  split; [ | split ].
  - intros tenv tenv' n ty Hfits Hty Hkind Hadd.
    unfold Flambda2_types.add_equation, Meet_env.use_meet_env in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_equation (Meet_env.create tenv) n ty mh)
      as [ menv' | ] eqn:Hadd'; [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    apply typing_env_kept_of_meet_env.
    exact (add_equation_keeps mh (Meet_env.create tenv) menv' n ty Hkeeps
             Hfits Hty Hkind Hadd').
  - intros tenv tenv' extension Hfits Hextension Hkinded Hadd.
    unfold Flambda2_types.add_env_extension, Meet_env.use_meet_env in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_env_extension (Meet_env.create tenv) extension mh)
      as [ menv' | ] eqn:Hadd'; [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    apply typing_env_kept_of_meet_env.
    exact (add_env_extension_keeps mh (Meet_env.create tenv) menv' extension
             Hkeeps Hfits Hextension Hkinded Hadd').
  - intros tenv tenv' params param_types Hfits Hparams Hadd.
    unfold Flambda2_types.add_equations_on_params, Meet_env.use_meet_env
      in Hadd.
    fold mh in Hadd.
    destruct (Meet_env.add_equations_on_params (Meet_env.create tenv) params
                param_types mh) as [ menv' | ] eqn:Hadd';
      [ | discriminate Hadd ].
    cbn in Hadd. injection Hadd as <-.
    apply typing_env_kept_of_meet_env.
    exact (add_equations_on_params_keeps mh (Meet_env.create tenv) menv'
             params param_types Hkeeps Hfits Hparams Hadd').
Qed.
