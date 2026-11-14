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

(* Proofs of the four statements of FlambdaRegionsCheckedSpec.v: the state
   that a run of a checked term starts in has the property, a step keeps
   it, a state that has it stops without a step only where [step] has none,
   and in such a state what a step reads may be read. The lemmas are in the
   files FlambdaRegionsChecked...Proofs.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedInitialProofs FlambdaRegionsCheckedHeadProofs
  FlambdaRegionsCheckedReadsProofs FlambdaRegionsCheckedViewsProofs
  FlambdaRegionsCheckedFaultProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs FlambdaRegionsCheckedCallProofs
  FlambdaRegionsCheckedAppliedProofs FlambdaRegionsCheckedMovesProofs
  FlambdaRegionsCheckedDeepProofs FlambdaRegionsCheckedLetProofs
  FlambdaRegionsCheckedSharedProofs FlambdaRegionsCheckedSameStoreProofs
  FlambdaRegionsCheckedStoreProofs.
Import ListNotations.

(* What follows a marker is the body of the second [Let]. *)
Lemma marker_body (x : variable) (body : expr) (y : variable)
    (region : option simple) (body' : expr) :
  marker_of x body = Marker y region body' ->
  ∃ (y0 : variable) (kind0 : flambda_kind) (d0 : named),
    body = Let y0 kind0 d0 body'.
Proof.
  unfold marker_of.
  destruct body as [y0 kind0 d0 body0 | | | | | | | ]; try discriminate.
  intros H. exists y0, kind0, d0.
  destruct d0 as [ | [ | | | | [ | | | | ]] args | | | | | ];
    try discriminate.
  destruct args as [ | [x' | | ] rest]; try discriminate.
  destruct (Pos.eqb x x' && kind_eqb kind0 Value); [ | discriminate ].
  destruct rest as [ | [z | sym | c] [ | ? ?]]; try discriminate.
  - injection H as _ _ <-. reflexivity.
  - destruct (Pos.eqb z x); [ discriminate | ]. injection H as _ _ <-.
    reflexivity.
  - injection H as _ _ <-. reflexivity.
  - injection H as _ _ <-. reflexivity.
Qed.

(* The state that a run of a checked term starts in has the
   property. *)
Theorem checked_initial : checked_initial_statement.
Proof. exact initial_state_checked. Qed.

(* A state that has it stops without a step only where [step] has
   none. *)
Theorem checked_fault : checked_fault_statement.
Proof. exact checked_state_fault. Qed.

(* In a state that has it, what a step reads may be read. *)
Theorem checked_reads_live : checked_reads_live_statement.
Proof. exact checked_state_reads_live. Qed.

(* A step keeps it. *)
Theorem region_step_keeps_checked : region_step_keeps_checked_statement.
Proof.
  intros cert codes rs rs' Hchecked Hstep.
  pose proof (region_step_keeps_invariant rs rs' (proj1 Hchecked) Hstep)
    as Hinv'.
  pose proof
    (region_step_keeps_frames_ordered rs rs' Hstep (proj1 (proj2 Hchecked)))
    as Hordered'.
  pose proof (counted_checked cert codes rs Hchecked) as Hcounted.
  destruct Hchecked
    as (Hinv & _ & _ & _ & _ & _ & _ & _ & cs0 & ms0 & Hcheck0 & _ & Hrun0
        & Hconts0).
  pose proof
    (simples_ok_no_fault rs _ _
       (head_simples_ok cert codes rs cs0 ms0 (invariant_fresh rs Hinv)
          Hcheck0 Hrun0 Hconts0))
    as Hnofault.
  clear Hinv Hcheck0 Hrun0 Hconts0 cs0 ms0.
  pose proof Hcounted
    as (Hinv_c & Hord_c & Hcode_c & Hlocal_c & Hdeep_c & Hstatics_c & Hplain_c
        & Hpaps_c & cs & ms & Hcheck_c & Hincl_c & Hrun_c & Hconts_c).
  pose proof (invariant_fresh _ Hinv_c) as Hfresh_c.
  pose proof
    (proj2 (good_split cert codes (count_uses rs)) (conj Hplain_c Hpaps_c))
    as Hgood_c.
  pose proof Hgood_c as (Hgood_vars_c & Hgood_conts_c & Hgood_statics_c).
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He.
  - assert (He_c :
      state_expr (rs_state (count_uses rs)) = Let x kind d body)
      by exact He.
    destruct (region_named d) eqn:Hd.
    + (* A primitive of regions. *)
      destruct d as [s | [u | b | v | hp | p] args | | | | | ];
        try discriminate.
      rewrite (region_step_region_let rs x kind p args body He) in Hstep.
      pose proof Hcheck_c as Hc. rewrite He_c in Hc.
      rewrite check_expr_region_let in Hc.
      unfold region_let_step in Hstep. unfold check_region_let in Hc.
      destruct p, args as [ | [r | sym | c] [ | arg' args]];
        try discriminate.
      * apply andb_true_iff in Hc as [Hc Hbody].
        apply andb_true_iff in Hc as [_ Hx].
        eapply begin_checked; eassumption.
      * apply andb_true_iff in Hc as [Hc Hbody].
        apply andb_true_iff in Hc as [_ Hx].
        eapply begin_checked; eassumption.
      * apply andb_true_iff in Hc as [Hc Hbody].
        apply andb_true_iff in Hc as [Hc Hopen].
        apply andb_true_iff in Hc as [_ Hx].
        eapply end_checked; eassumption.
      * apply andb_true_iff in Hc as [Hc Hbody].
        apply andb_true_iff in Hc as [Hc Hopen].
        apply andb_true_iff in Hc as [_ Hx].
        eapply end_checked; eassumption.
    + (* Another [Let]. *)
      rewrite (region_step_let rs x kind d body He Hd) in Hstep
        by (rewrite He; exact Hnofault).
      unfold let_step in Hstep.
      assert (Hsimples :
        evaluated_simples (Let x kind d body) = simples_of_named d).
      { destruct d as [s | [u | b | v | hp | p] args | | | | | ];
          first [ reflexivity | discriminate ]. }
      destruct (marker_of x body) as [ | y region body' | ] eqn:Hmarker.
      * destruct (is_allocation d && _); [ discriminate | ].
        destruct (plain_step_step (count_uses rs) rs' Hstep)
          as (s' & Hs' & ->).
        apply (same_store_checked cert codes (count_uses rs));
          try reflexivity;
          [ exact Hinv' | exact Hordered' | exact Hcounted | ].
        change (entered cert codes (count_uses rs) s').
        eapply let_entered; eassumption.
      * pose proof Hcheck_c as Hc. rewrite He_c in Hc.
        rewrite (check_expr_let cert codes cs x kind d body Hd) in Hc.
        unfold check_let in Hc. cbv zeta in Hc. rewrite Hmarker in Hc.
        destruct (marker_body x body y region body' Hmarker)
          as (y0 & kind0 & d0 & ->).
        apply andb_true_iff in Hc as [Hc Hrest].
        apply andb_true_iff in Hc as [Hx Husable].
        apply andb_true_iff in Hrest as [Hrest Hbody].
        apply andb_true_iff in Hrest as [Hrest Hclosures].
        apply andb_true_iff in Hrest as [Hrest _].
        apply andb_true_iff in Hrest as [_ Hy].
        pose proof Hincl_c as Hi. rewrite He_c in Hi.
        eapply local_checked with (body' := body') (region := region);
          try eassumption.
        intros s Hin. apply (head_simples_below _ s Hfresh_c).
        rewrite He_c, Hsimples. exact Hin.
      * discriminate.
  - (* A [Let_static]. *)
    assert (He_c :
      state_expr (rs_state (count_uses rs)) = Let_static bindings body)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (existsb _ _); [ discriminate | ].
    pose proof Hcheck_c as Hc. rewrite He_c in Hc.
    eapply static_checked; eassumption.
  - assert (He_c :
      state_expr (rs_state (count_uses rs)) = Let_cont_non_recursive k h body)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & ->).
    apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
      [ exact Hinv' | exact Hordered' | exact Hcounted | ].
    change (entered cert codes (count_uses rs) s').
    eapply let_cont_entered; eassumption.
  - assert (He_c :
      state_expr (rs_state (count_uses rs)) = Let_cont_recursive handlers body)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & ->).
    apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
      [ exact Hinv' | exact Hordered' | exact Hcounted | ].
    change (entered cert codes (count_uses rs) s').
    eapply let_cont_rec_entered; eassumption.
  - assert (He_c : state_expr (rs_state (count_uses rs)) = Apply ap)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & ->).
    apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
      [ exact Hinv' | exact Hordered' | exact Hcounted | ].
    change (entered cert codes (count_uses rs) s').
    eapply apply_entered; eassumption.
  - assert (He_c : state_expr (rs_state (count_uses rs)) = Apply_cont ac)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (existsb (simple_open_mark (count_uses rs)) (ac_args ac))
      eqn:Hmark.
    + destruct (region_apply_cont_entered cert codes (count_uses rs) cs ms
                  Hfresh_c Hcode_c Hlocal_c Hdeep_c Hstatics_c Hgood_vars_c
                  Hgood_conts_c Hgood_statics_c Hcheck_c Hrun_c Hconts_c ac
                  rs' He_c Hmark Hstep)
        as (s' & -> & Hentered).
      apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
        [ exact Hinv' | exact Hordered' | exact Hcounted | ].
      exact Hentered.
    + destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & ->).
      apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
        [ exact Hinv' | exact Hordered' | exact Hcounted | ].
      change (entered cert codes (count_uses rs) s').
      eapply apply_cont_entered; eassumption.
  - assert (He_c :
      state_expr (rs_state (count_uses rs)) = Switch scrutinee arms)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & ->).
    apply (same_store_checked cert codes (count_uses rs)); try reflexivity;
      [ exact Hinv' | exact Hordered' | exact Hcounted | ].
    change (entered cert codes (count_uses rs) s').
    eapply switch_entered; eassumption.
  - assert (He_c : state_expr (rs_state (count_uses rs)) = Invalid)
      by exact He.
    unfold region_step in Hstep. cbv zeta in Hstep.
    rewrite He, Hnofault in Hstep.
    destruct (plain_step_step (count_uses rs) rs' Hstep) as (s' & Hs' & _).
    unfold with_statics in Hs'. rewrite He_c in Hs'. discriminate Hs'.
Qed.
