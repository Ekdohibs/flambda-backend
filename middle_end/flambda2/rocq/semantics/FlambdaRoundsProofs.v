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

(* Simulations by rounds: one or more steps of the first state are
   answered by one or more steps of the second. Nothing here needs
   review: it is a way of proving [refines], beside
   FlambdaSemanticsProofs.simulation_refines, which asks that the two
   states step together, and FlambdaSimulationProofs.stuttering_refines,
   in which one step of the first is answered by one or more of the
   second.

   It serves where a rewrite makes several steps into one: an
   application of a continuation whose handler only applies another,
   made an application of that other (FlambdaForwardProofs.v). Each
   round uses up at least one step of each run, so a run of the first
   that exits is followed round by round to its exit, and a run of the
   second that has ended bounds the rounds of the first
   ([rounds_ends]); no measure is needed, and divergence never has to be
   considered. *)

From Stdlib Require Import ZArith List Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs.
Import ListNotations.

Section Rounds.

Variable R : state -> state -> Prop.

(* What some steps of the first state come to is answered by the second
   state: nothing is asked if the first run is undefined; an exit is
   answered by an exit through the same continuation with the same
   values; a state, by one or more steps to a related state. *)
Definition round_answered (result : step_result) (s' : state) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd => ∃ m : nat, run m s' = Exit k vs hp rd
  | Step t => ∃ (m : nat) (t' : state), run (S m) s' = Step t' ∧ R t t'
  end.

(* Two states that step together make a round of one step each. *)
Lemma step_round (s s' : state) :
  result_simulated R (step s) (step s') -> round_answered (run 1 s) s'.
Proof.
  unfold round_answered. rewrite run_succ.
  destruct (step s) as [t | k vs | ], (step s') as [t' | k' vs' | ] eqn:Hstep';
    cbn [result_simulated run]; try contradiction; try exact (fun _ => I).
  - intros Hrel. exists 0, t'. split; [ | exact Hrel ].
    cbn [run]. rewrite Hstep'. reflexivity.
  - intros [<- [<- [<- <-]]]. exists 1. cbn [run]. rewrite Hstep'. reflexivity.
Qed.

Hypothesis R_round :
  ∀ s s' : state, R s s' -> ∃ n : nat, round_answered (run (S n) s) s'.

(* A run that has come to an exit, or to nothing, stays there. *)
Lemma run_exit_stable (n m : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  run n s = Exit k vs hp rd -> run (n + m) s = Exit k vs hp rd.
Proof. intros Hrun. rewrite run_add, Hrun. reflexivity. Qed.

Lemma run_undefined_stable (n m : nat) (s : state) :
  run n s = Undefined -> run (n + m) s = Undefined.
Proof. intros Hrun. rewrite run_add, Hrun. reflexivity. Qed.

(* An exit of the first run is one of the second. *)
Lemma rounds_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s s' -> run n s = Exit k vs hp rd -> ∃ m : nat, run m s'
      = Exit k vs hp rd.
Proof.
  induction n as [n IH] using (well_founded_induction lt_wf).
  intros s s' k vs hp rd Hrel Hexit.
  destruct (R_round s s' Hrel) as [j Hanswer].
  destruct (Nat.le_gt_cases n (S j)) as [Hshort | Hlong].
  - (* the exit is within the round *)
    replace (S j) with (n + (S j - n)) in Hanswer by lia.
    rewrite (run_exit_stable n _ s k vs _ rd_ Hexit) in Hanswer. exact Hanswer.
  - replace n with (S j + (n - S j)) in Hexit by lia.
    rewrite run_add in Hexit.
    destruct (run (S j) s) as [t | k' vs' | ]; [ | | discriminate Hexit ].
    + destruct Hanswer as (m & t' & Hrun & Hrel').
      assert (Hless : n - S j < n) by lia.
      destruct (IH (n - S j) Hless t t' k vs hp rd_ Hrel' Hexit) as [m' Hm'].
      exists (S m + m'). rewrite run_add, Hrun. exact Hm'.
    + injection Hexit as -> -> -> ->. exact Hanswer.
Qed.

(* If the second run has ended, so has the first: each round uses up at
   least one step of the second. *)
Lemma rounds_ends (m : nat) :
  ∀ (s s' : state) (b : behaviour),
    R s s' ->
    ended (run m s') b ->
    ∃ n : nat,
      run n s = Undefined
      ∨ ∃ (k : continuation) (vs : list value) (hp : heap) (rd : read_record),
        run n s
        = Exit k vs hp rd.
Proof.
  induction m as [m IH] using (well_founded_induction lt_wf).
  intros s s' b Hrel Hended.
  destruct (R_round s s' Hrel) as [j Hanswer].
  destruct (run (S j) s) as [t | k vs hp' rd' | ] eqn:Hround.
  - destruct Hanswer as (i & t' & Hrun & Hrel').
    destruct (Nat.lt_ge_cases m (S i)) as [Hshort | Hlong].
    + exfalso. replace (S i) with (m + (S i - m)) in Hrun by lia.
      rewrite (ended_stable m _ s' b Hended) in Hrun.
      rewrite Hrun in Hended. destruct b; contradiction.
    + replace m with (S i + (m - S i)) in Hended by lia.
      rewrite run_add, Hrun in Hended.
      assert (Hless : m - S i < m) by lia.
      destruct (IH (m - S i) Hless t t' b Hrel' Hended) as [n Hn].
      exists (S j + n). rewrite run_add, Hround. exact Hn.
  - exists (S j). right. exists k, vs, hp', rd'. exact Hround.
  - exists (S j). left. exact Hround.
Qed.

(* The second state refines the first. *)
Lemma rounds_refines (s s' : state) : R s s' -> refines s s'.
Proof.
  intros Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp [n [rd Hn]].
    destruct (rounds_exit n s s' k vs _ rd_ Hrel Hn) as [m Hm].
    exists m, rd. exact Hm. }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (rounds_ends m s s' b Hrel Hended)
    as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
  - right. exists n. exact Hwrong.
  - left.
    assert (Hs : behaves s (Exits k vs hp))
      by (exists n, rd; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

End Rounds.
