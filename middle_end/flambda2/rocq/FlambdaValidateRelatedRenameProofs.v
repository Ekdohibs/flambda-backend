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

(* Proofs of the statements of FlambdaValidateRelatedRenameSpec.v. This
   file needs no review.

   The liftings made in turn without the test of the exits are those
   of FlambdaValidateSpec.lifted_in_turn with another first step, and
   the proofs are those of FlambdaValidatorEquivarianceProofs.v for
   that function with the laws of that step. The check is then as
   FlambdaValidatorEquivarianceProofs.validate_unit_renamed_alike for
   the two comparisons of lifted terms, with the tests on static names
   by FlambdaValidateImportsRenameProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaAlphaProofs FlambdaWellFormedSpec FlambdaCompareSpec
  FlambdaStaticsOrderSpec FlambdaStaticsOrderProofs FlambdaUnusedStaticsSpec
  FlambdaUnusedHeadStaticsSpec FlambdaLiftSpec FlambdaLiftObservedSpec
  FlambdaKnownSlotsSpec FlambdaKnownSlotsProofs FlambdaHoistStaticsSpec
  FlambdaFloatCodeSpec FlambdaLoopifySpec FlambdaStubCallsSpec
  FlambdaStubCallsProofs FlambdaNeverReturnsSpec FlambdaNeverReturnsProofs
  FlambdaNeverReturnsScopeProofs FlambdaLoopsScopeProofs
  FlambdaStaticsScopeProofs FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceProofs
  FlambdaImportsSpec FlambdaValidateImportsSpec
  FlambdaValidateImportsRenameSpec FlambdaValidateImportsRenameProofs
  FlambdaValidateRelatedSpec FlambdaValidateRelatedRenameSpec.
Import ListNotations.

(* ======================================================================== *)
(* The liftings made in turn without the test of the exits                  *)
(* ======================================================================== *)

Lemma lifted_in_turn_observed_scope (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting) :
  keeps_scope (lifted_in_turn_observed rounds avoid Ls kept).
Proof.
  induction Ls as [ | L Ls IH]; intros e Hclosed;
    cbn [lifted_in_turn_observed].
  - destruct (read_known_slots_anywhere_scope e Hclosed)
      as (Hclosed1 & Hv1 & Hk1).
    destruct (demote_and_tidy_known_scope rounds _ Hclosed1)
      as (Hclosed2 & Hv2 & Hk2).
    split; [ exact Hclosed2 | split ].
    + intros x Hx. apply Hv1, Hv2, Hx.
    + intros k Hk. apply Hk1, Hk2, Hk.
  - destruct
      (FlambdaStaticsScopeProofs.with_lifting_observed_keeps_scope L e
         Hclosed)
      as (Hclosed1 & Hv1 & Hk1).
    destruct
      (normalise_lifted_calling_scope (made_closures_called L) rounds avoid
         kept (with_lifting_observed L e) Hclosed1)
      as (Hclosed2 & Hv2 & Hk2).
    destruct (IH _ Hclosed2) as (Hclosed3 & Hv3 & Hk3).
    split; [ exact Hclosed3 | split ].
    + intros x Hx. apply Hv1, Hv2, Hv3, Hx.
    + intros k Hk. apply Hk1, Hk2, Hk3, Hk.
Qed.

Theorem lifted_in_turn_observed_keeps_scope :
  lifted_in_turn_observed_keeps_scope_statement.
Proof. exact lifted_in_turn_observed_scope. Qed.

Theorem lifted_in_turn_observed_equivariant_up_to :
  lifted_in_turn_observed_equivariant_up_to_statement.
Proof.
  intros r rounds avoid kept Ls.
  induction Ls as [ | L Ls IH]; intros Hunits r1 e Hclosed Halike Havoid;
    cbn [map lifted_in_turn_observed].
  - exists r1. split; [ | split; [ exact Havoid | ] ].
    + destruct
        (renames_alike_kept read_known_slots_anywhere r r1 e
           read_known_slots_anywhere_scope Hclosed Halike)
        as (Hclosed1 & Halike1).
      exact
        (proj2
           (renames_alike_kept (demote_and_tidy_known rounds) r r1 _
              (demote_and_tidy_known_scope rounds) Hclosed1 Halike1)).
    + rewrite FlambdaKnownSlotsProofs.read_known_slots_anywhere_equivariant.
      apply demote_and_tidy_known_equivariant.
  - destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
    rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
      with_lifting_observed_renamed.
    destruct
      (renames_alike_kept (with_lifting_observed L) r r1 e
         (FlambdaStaticsScopeProofs.with_lifting_observed_keeps_scope L)
         Hclosed Halike)
      as (Hclosed1 & Halike1).
    change (made_closures_called (rename_lifting r1 L))
      with (made_closures_called L).
    destruct
      (normalise_lifted_calling_up_to (made_closures_called L) rounds kept r1
         avoid (with_lifting_observed L e) Hclosed1)
      as (r2 & Halike2 & Havoid2 & Heq).
    cbv beta in Heq, Halike2.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
    rewrite Heq.
    destruct
      (normalise_lifted_calling_scope (made_closures_called L) rounds avoid
         kept (with_lifting_observed L e) Hclosed1)
      as (Hclosed2 & _ & _).
    apply (IH (fun L' HL' => Hunits L' (or_intror HL')) r2).
    + exact Hclosed2.
    + exact
        (renames_alike_then
           (normalise_lifted_calling (made_closures_called L) rounds avoid
              kept)
           r r1 r2 (with_lifting_observed L e)
           (normalise_lifted_calling_scope (made_closures_called L) rounds
              avoid kept)
           Hclosed1 Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

(* ======================================================================== *)
(* The test of one lifting                                                  *)
(* ======================================================================== *)

Lemma to_head_observed_renamed (r : renaming) (L : lifting) (e : expr) :
  to_head_observed (rename_lifting r L) (rename_expr r e)
  = rename_expr r (to_head_observed L e).
Proof.
  unfold to_head_observed.
  change (lifting_unit (rename_lifting r L))
    with (rename_unit_interface r (lifting_unit L)).
  change (lifted_to_head (rename_lifting r L)) with (lifted_to_head L).
  rewrite lift_to_head_unit_observed_renamed.
  destruct
    (lift_to_head_unit_observed (lifting_unit L) (lifted_to_head L) e);
    reflexivity.
Qed.

Lemma out_of_code_observed_renamed (r : renaming) (L : lifting) (e : expr) :
  out_of_code_observed (rename_lifting r L) (rename_expr r e)
  = rename_expr r (out_of_code_observed L e).
Proof.
  unfold out_of_code_observed.
  change (lifting_unit (rename_lifting r L))
    with (rename_unit_interface r (lifting_unit L)).
  change (lifted_out_of_code (rename_lifting r L))
    with (lifted_out_of_code L).
  rewrite lift_from_code_unit_observed_renamed.
  destruct
    (lift_from_code_unit_observed (lifting_unit L) (lifted_out_of_code L) e);
    reflexivity.
Qed.

Lemma lets_observed_renamed (r : renaming) (L : lifting) (e : expr) :
  lets_observed (rename_lifting r L) (rename_expr r e)
  = rename_expr r (lets_observed L e).
Proof.
  unfold lets_observed. rewrite lift_unit_observed_renamed.
  destruct (lift_unit_observed L e); reflexivity.
Qed.

Lemma lifting_observed_free_in_renamed (r : renaming) (statics : static_env)
    (L : lifting) (e : expr) :
  lifting_observed_free_in statics (rename_lifting r L) (rename_expr r e)
  = lifting_observed_free_in statics L e.
Proof.
  unfold lifting_observed_free_in. cbv zeta.
  rewrite to_head_observed_renamed, out_of_code_observed_renamed,
    lets_observed_renamed, with_lifting_observed_renamed,
    !static_names_free_in_equivariant.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The test of the liftings made in turn                                    *)
(* ======================================================================== *)

Theorem lifted_observed_free_in_renamed_alike :
  lifted_observed_free_in_renamed_alike_statement.
Proof.
  intros r r1 statics rounds avoid Ls kept. revert r1.
  induction Ls as [ | L Ls IH]; intros r1 e Hunits Hclosed Havoid;
    cbn [map lifted_observed_free_in].
  - rewrite read_known_slots_anywhere_equivariant.
    apply demote_and_tidy_known_free_in_equivariant.
  - destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
    rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
      lifting_observed_free_in_renamed, with_lifting_observed_renamed.
    destruct
      (FlambdaStaticsScopeProofs.with_lifting_observed_keeps_scope L e
         Hclosed)
      as (Hclosed1 & _ & _).
    change (made_closures_called (rename_lifting r1 L))
      with (made_closures_called L).
    pose proof
      (normalise_lifted_calling_free_in_equivariant r1 statics
         (made_closures_called L) rounds avoid kept
         (with_lifting_observed L e) Hclosed1)
      as Htest.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Htest.
    rewrite Htest.
    destruct
      (normalise_lifted_calling_up_to (made_closures_called L) rounds kept r1
         avoid (with_lifting_observed L e) Hclosed1)
      as (r2 & _ & Havoid2 & Heq).
    cbv beta in Heq.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
    rewrite Heq.
    destruct
      (normalise_lifted_calling_scope (made_closures_called L) rounds avoid
         kept (with_lifting_observed L e) Hclosed1)
      as (Hclosed2 & _ & _).
    rewrite (IH r2 _ (fun L' HL' => Hunits L' (or_intror HL')) Hclosed2);
      [ reflexivity | ].
    intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Theorem validate_unit_related_renamed_alike :
  validate_unit_related_renamed_alike_statement.
Proof.
  intros r r1 imports rounds unused slots G Ls kept before after Halike1
    Hheld.
  unfold validate_unit_related. cbv zeta.
  rewrite !code_closed_equivariant.
  destruct (code_closed before) eqn:Hclosed; cbn [andb]; [ | reflexivity ].
  destruct (code_closed after); cbn [andb]; [ | reflexivity ].
  set (statics := imports_statics imports).
  assert (Hunits : rename_conts_alike r r1 (unit_continuations Ls)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. reflexivity. }
  assert (Hloops : rename_conts_alike r r1 (map snd G)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. apply orb_true_r. }
  rewrite FlambdaStubCallsProofs.demote_stub_calls_equivariant,
    FlambdaNeverReturnsProofs.cut_returns_unit_equivariant,
    (rename_loopified_alike r r1 G Hloops), loopify_unit_equivariant.
  destruct
    (renames_alike_kept demote_stub_calls r r1 before
       FlambdaStaticsScopeProofs.demote_stub_calls_keeps_scope Hclosed
       Halike1)
    as (Hclosed0 & Halike0).
  destruct
    (renames_alike_kept (cut_returns []) r r1 (demote_stub_calls before)
       (FlambdaNeverReturnsScopeProofs.cut_returns_keeps_scope [])
       Hclosed0 Halike0)
    as (Hclosed00 & Halike00).
  destruct
    (renames_alike_kept (loopify_unit G) r r1
       (cut_returns [] (demote_stub_calls before))
       (FlambdaLoopsScopeProofs.loopify_unit_keeps_scope G) Hclosed00
       Halike00)
    as (Hclosed' & Halike1').
  clear Hclosed Halike1 Hheld Hloops Hclosed0 Halike0 Hclosed00 Halike00.
  revert Hclosed' Halike1'.
  generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
  clear before.
  intros before Hclosed Halike1.
  destruct
    (lifted_in_turn_observed_equivariant_up_to r rounds
       (unit_continuations Ls) kept Ls (unit_continuations_has Ls) r1 before
       Hclosed Halike1 Hunits)
    as (r2 & Halike2 & Havoid2 & Heq2).
  destruct
    (lifted_in_turn_observed_scope rounds (unit_continuations Ls) kept Ls
       before Hclosed)
    as (Hclosed_l0 & _ & _).
  rewrite unit_continuations_renamed.
  rewrite Heq2, normalise_after_equivariant, tidied_after_equivariant,
    keeps_a_return_equivariant, with_returns_equivariant,
    tidied_before_equivariant, without_head_statics_equivariant,
    without_slots_equivariant, without_statics_equivariant,
    with_code_first_equivariant, !float_code_rewrite_equivariant,
    (with_statics_first_renamed_alike r r Ls _ (fun k _ => eq_refl)),
    (with_statics_first_renamed_alike r r2 Ls _ Havoid2),
    !order_statics_as_equivariant, head_removal_free_in_equivariant,
    (lifted_observed_free_in_renamed_alike r r1 statics rounds
       (unit_continuations Ls) Ls kept before (unit_continuations_has Ls)
       Hclosed Hunits),
    (statics_first_free_in_renamed_alike r r statics Ls _
       (fun k _ => eq_refl)),
    (statics_first_free_in_renamed_alike r r2 statics Ls _ Havoid2).
  set (a := tidied_after rounds (normalise_after rounds kept after)).
  generalize (negb (keeps_a_return a)). intros taken.
  destruct
    (renames_alike_kept (with_returns taken) r r2
       (lifted_in_turn_observed rounds (unit_continuations Ls) Ls kept
          before)
       (with_returns_keeps_scope taken) Hclosed_l0 Halike2)
    as (Hclosed_lt & Halike_lt).
  destruct
    (renames_alike_kept (tidied_before rounds) r r2
       (with_returns taken
          (lifted_in_turn_observed rounds (unit_continuations Ls) Ls kept
             before))
       (tidied_before_keeps_scope rounds) Hclosed_lt Halike_lt)
    as (Hclosed_l & Halike_l).
  set
    (l :=
      tidied_before rounds
        (with_returns taken
           (lifted_in_turn_observed rounds (unit_continuations Ls) Ls kept
              before)))
    in *.
  set (a' := without_head_statics unused a).
  set (c := float_code (with_code_first a')).
  set (c' := with_statics_first Ls a').
  destruct
    (renames_alike_kept (without_slots slots) r r2 l
       (without_slots_scope slots) Hclosed_l Halike_l)
    as (Hclosed_slots & Halike_slots).
  destruct
    (renames_alike_kept (without_statics unused) r r2
       (without_slots slots l) (without_statics_scope unused) Hclosed_slots
       Halike_slots)
    as (Hclosed_without & Halike_without).
  destruct
    (renames_alike_kept float_code r r2
       (without_statics unused (without_slots slots l)) float_code_scope
       Hclosed_without Halike_without)
    as (Hclosed_floated & Halike_floated).
  destruct
    (renames_alike_kept (order_statics_as c) r r2
       (float_code (without_statics unused (without_slots slots l)))
       (order_statics_as_scope c) Hclosed_floated Halike_floated)
    as (Hclosed_last & Halike_last).
  destruct
    (renames_alike_kept (with_statics_first Ls) r r2
       (without_statics unused (without_slots slots l))
       (with_statics_first_scope Ls) Hclosed_without Halike_without)
    as (Hclosed_first & Halike_first).
  destruct
    (renames_alike_kept (order_statics_as c') r r2
       (with_statics_first Ls
          (without_statics unused (without_slots slots l)))
       (order_statics_as_scope c') Hclosed_first Halike_first)
    as (Hclosed_last' & Halike_last').
  rewrite
    (expr_eqb_renamed_alike r r2
       (order_statics_as c
          (float_code (without_statics unused (without_slots slots l))))
       c Hclosed_last Halike_last),
    (expr_eqb_renamed_alike r r2
       (order_statics_as c'
          (with_statics_first Ls
             (without_statics unused (without_slots slots l))))
       c' Hclosed_last' Halike_last').
  reflexivity.
Qed.

Theorem validate_unit_related_equivariant :
  validate_unit_related_equivariant_statement.
Proof.
  intros r imports rounds unused slots G Ls kept before after.
  apply validate_unit_related_renamed_alike.
  - split; intros name _; reflexivity.
  - intros k _. reflexivity.
Qed.
