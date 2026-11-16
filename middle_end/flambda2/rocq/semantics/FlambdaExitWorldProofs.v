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

(* Proofs of the statements of FlambdaExitWorldSpec.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaExitWorldSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a run leaves at an exit                                             *)
(* ======================================================================== *)

(* A run that ends by an exit goes through a state whose step is the
   exit. *)
Lemma run_exit_last (n : nat) :
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n s = Exit k vs hp rd ->
    ∃ (m : nat) (last : state),
      run m s = Step last ∧ step last = Exit k vs hp rd.
Proof.
  induction n as [ | n IH]; intros s k vs hp rd Hrun; [ discriminate Hrun | ].
  rewrite run_succ in Hrun.
  destruct (step s) as [s' | k' vs' hp' rd' | ] eqn:Hstep;
    [ | | discriminate Hrun ].
  - destruct (IH s' k vs hp rd_ Hrun) as (m & last & Hm & Hlast).
    exists (S m), last. rewrite run_succ, Hstep. split; assumption.
  - exists O, s. split; [ reflexivity | ]. rewrite Hstep. exact Hrun.
Qed.

Lemma last_run_exit (n : nat) (s last : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  run n s = Step last ->
  step last = Exit k vs hp rd ->
  run (n + 1) s = Exit k vs hp rd.
Proof.
  intros Hrun Hstep. rewrite run_add, Hrun. simpl. rewrite Hstep. reflexivity.
Qed.

Theorem exits_world_behaves : exits_world_behaves_statement.
Proof.
  intros s k vs hp. split.
  - intros [fuel [rd Hrun]].
    destruct (run_exit_last fuel s k vs hp rd_ Hrun) as (m & last & Hm & Hlast).
    exists (state_statics last), rd, m, last. simpl. auto.
  - intros (statics & rd & n & last & Hrun & Hstep & _). simpl in Hstep.
    exists (n + 1), rd. exact (last_run_exit n s last k vs hp rd Hrun Hstep).
Qed.

Lemma exits_world_exits (s : state) (w : exit_world) :
  exits_world s w ->
  behaves s (Exits (exit_cont w) (exit_values w) (exit_heap w)).
Proof.
  intros (n & last & Hrun & Hstep & _). exists (n + 1), (exit_reads w).
  exact (last_run_exit n s last _ _ _ rd_ Hrun Hstep).
Qed.

(* Two runs of one state to a state: the longer goes through the
   shorter. *)
Lemma run_step_prefix (n m : nat) (s a b : state) :
  n <= m -> run n s = Step a -> run m s = Step b -> run (m - n) a = Step b.
Proof.
  intros Hle Ha Hb. replace m with (n + (m - n)) in Hb by lia.
  rewrite run_add, Ha in Hb. exact Hb.
Qed.

Theorem exits_world_unique : exits_world_unique_statement.
Proof.
  intros s w w' (n & last & Hrun & Hstep & Hst)
    (n' & last' & Hrun' & Hstep' & Hst').
  assert (Hsame : last = last').
  { destruct (Nat.le_gt_cases n n') as [Hle | Hgt].
    - pose proof (run_step_prefix n n' s last last' Hle Hrun Hrun') as H.
      destruct (n' - n) as [ | d]; [ injection H as <-; reflexivity | ].
      rewrite run_succ, Hstep in H. discriminate H.
    - assert (Hle : n' <= n) by lia.
      pose proof (run_step_prefix n' n s last' last Hle Hrun' Hrun) as H.
      destruct (n - n') as [ | d]; [ injection H as <-; reflexivity | ].
      rewrite run_succ, Hstep' in H. discriminate H. }
  subst last'. rewrite Hstep in Hstep'. rewrite Hst in Hst'.
  destruct w, w'. simpl in *. injection Hstep' as -> -> ->. subst.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Refinement with the worlds of two exits related                          *)
(* ======================================================================== *)

Theorem refines_world_same : refines_world_same_statement.
Proof.
  intros source target. split.
  - intros [Hexits Hends]. split.
    + intros k vs hp Hsource.
      apply exits_world_behaves in Hsource. destruct Hsource as [st [rd Hw]].
      destruct (Hexits _ Hw) as (w' & Hw' & Hk & Hvs & Hhp). simpl in *.
      subst. exact (exits_world_exits target w' Hw').
    + intros b Hnot Htarget.
      destruct (Hends b Hnot Htarget) as [Hwrong | [w Hw]];
        [ right; exact Hwrong | ].
      left. destruct (Hexits _ Hw) as (w' & Hw' & Hk & Hvs & Hhp).
      pose proof (exits_world_exits target w' Hw') as Hexit.
      rewrite (behaviour_unique target b _ Htarget Hexit).
      rewrite <- Hk, <- Hvs, <- Hhp. exact (exits_world_exits source w Hw).
  - intros [Hexits Hends]. split.
    + intros w Hw.
      pose proof (Hexits _ _ _ (exits_world_exits source w Hw)) as Htarget.
      apply exits_world_behaves in Htarget. destruct Htarget as [st' [rd Hw']].
      eexists. split; [ exact Hw' | ]. repeat split.
    + intros b Hnot Htarget.
      destruct (Hends b Hnot Htarget) as [Hsource | Hwrong];
        [ | left; exact Hwrong ].
      destruct b as [k vs hp | | ];
        [ | left; exact Hsource | contradiction Hnot; reflexivity ].
      right. apply exits_world_behaves in Hsource.
      destruct Hsource as [st [rd Hw]]. eauto.
Qed.

Theorem refines_world_trans : refines_world_trans_statement.
Proof.
  intros R1 R2 s1 s2 s3 [Hexits1 Hends1] [Hexits2 Hends2]. split.
  - intros w Hw. destruct (Hexits1 w Hw) as (w' & Hw' & HR1).
    destruct (Hexits2 w' Hw') as (w'' & Hw'' & HR2).
    exists w''. split; [ exact Hw'' | ]. exists w'. split; assumption.
  - intros b Hnot Hb.
    destruct (Hends2 b Hnot Hb) as [Hwrong | [w' Hw']].
    + apply (Hends1 Goes_wrong); [ discriminate | exact Hwrong ].
    + apply (Hends1 (Exits (exit_cont w') (exit_values w') (exit_heap w')));
        [ discriminate | exact (exits_world_exits s2 w' Hw') ].
Qed.

Theorem refines_world_weaken : refines_world_weaken_statement.
Proof.
  intros R R' source target Himpl [Hexits Hends]. split; [ | exact Hends ].
  intros w Hw. destruct (Hexits w Hw) as (w' & Hw' & HR).
  exists w'. split; [ exact Hw' | exact (Himpl _ _ HR) ].
Qed.
