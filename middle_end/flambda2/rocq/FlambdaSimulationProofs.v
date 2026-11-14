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

(* Simulations in which one step of the first state is answered by one or
   more steps of the second. Nothing here needs review: it is a way of
   proving [refines], beside FlambdaSemanticsProofs.simulation_refines,
   which asks that the two states step together.

   It serves where a rewrite makes one step into several: a [Let] made
   a [Let_static] followed by a [Let] (FlambdaLiftProofs.v), a call that
   then binds a variable. The second run never takes fewer steps than the
   first, which is what lets a run of the second that has ended bound the
   first ([stuttering_ends]); so no measure is needed, and divergence
   never has to be considered. What is asked of the values at an exit is
   a parameter ([Q]): equality gives [refines] ([stuttering_refines]),
   and a weaker relation gives the exit of the first run in the second
   up to it ([stuttering_exit]).

   The second section is the first with the heaps of the two exits
   left to the parameter too: the first asks that the second run exits
   with the heap of the first, which a rewrite that changes what a
   cell holds does not give (FlambdaLiftProofs.v: a cell of the second
   run holds a symbol where that of the first holds the block). *)

From Stdlib Require Import ZArith List Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs.
Import ListNotations.

(* ======================================================================== *)
(* Simulations in which one step is answered by one or more                 *)
(* ======================================================================== *)

Section Stuttering.

Variable R : state -> state -> Prop.

(* What is asked of the values at an exit, given the state that the
   second run exits from. *)
Variable Q : state -> list value -> list value -> Prop.

(* What a step of the first state comes to is answered by the second
   state: nothing is asked if it is undefined; an exit is answered by an
   exit through the same continuation; a step, by one or more steps to a
   related state. *)
Definition answered (result : step_result) (s' : state) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp rd => ∃ vs' : list value, step s' = Exit k vs' hp rd
    ∧ Q s' vs vs'
  | Step t => ∃ (n : nat) (t' : state), run (S n) s' = Step t' ∧ R t t'
  end.

Hypothesis R_step : ∀ s s' : state, R s s' -> answered (step s) s'.

(* An exit of the first run is one of the second. *)
Lemma stuttering_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s s' ->
    run n s = Exit k vs hp rd ->
    ∃ (m : nat) (last : state) (vs' : list value),
      run m s' = Step last ∧ step last = Exit k vs' hp rd ∧ Q last vs vs'.
Proof.
  induction n as [ | n IH]; intros s s' k vs hp rd Hrel; simpl;
    [ discriminate | ].
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k' vs0 | ]; unfold answered in Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel'). intros Hexit.
    destruct (IH t t' k vs hp rd_ Hrel' Hexit) as
      (m & last & vs' & Hm & Hlast & HQ).
    exists (S j + m), last, vs'. rewrite run_add, Hrun. auto.
  - intros [= <- <- -> ->]. destruct Hanswer as (vs' & Hstep & HQ).
    exists 0, s', vs'. auto.
  - discriminate.
Qed.

(* A state that the first run reaches is related to one that the second
   reaches. *)
Lemma stuttering_reaches (n : nat) :
  ∀ s s' last : state,
    R s s' ->
    run n s = Step last ->
    ∃ (m : nat) (last' : state), run m s' = Step last' ∧ R last last'.
Proof.
  induction n as [ | n IH]; intros s s' last Hrel; simpl.
  - intros [= <-]. exists 0, s'. split; [ reflexivity | exact Hrel ].
  - pose proof (R_step s s' Hrel) as Hanswer.
    destruct (step s) as [t | k vs hp rd | ]; unfold answered in Hanswer;
      [ | discriminate | discriminate ].
    destruct Hanswer as (j & t' & Hrun & Hrel'). intros Hlast.
    destruct (IH t t' last Hrel' Hlast) as (m & last' & Hm & Hrel'').
    exists (S j + m), last'. rewrite run_add, Hrun. auto.
Qed.

(* If the second run has ended, so has the first: each of its steps
   uses up at least one of the second's. *)
Lemma stuttering_ends (m : nat) :
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
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k vs hp' rd' | ] eqn:Hstep; unfold answered in
    Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel').
    destruct (Nat.lt_ge_cases m (S j)) as [Hshort | Hlong].
    + exfalso. replace (S j) with (m + (S j - m)) in Hrun by lia.
      rewrite (ended_stable m _ s' b Hended) in Hrun.
      rewrite Hrun in Hended. destruct b; contradiction.
    + replace m with (S j + (m - S j)) in Hended by lia.
      rewrite run_add, Hrun in Hended.
      assert (Hless : m - S j < m) by lia.
      destruct (IH (m - S j) Hless t t' b Hrel' Hended) as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
  - exists 1. right. exists k, vs, hp', rd'. simpl. rewrite Hstep. reflexivity.
  - exists 1. left. simpl. rewrite Hstep. reflexivity.
Qed.

(* Where the values at an exit are the same, the second state refines the
   first. *)
Lemma stuttering_refines (s s' : state) :
  (∀ (last : state) (vs vs' : list value), Q last vs vs' -> vs' = vs) ->
  R s s' ->
  refines s s'.
Proof.
  intros Hsame Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp [n [rd Hn]].
    destruct (stuttering_exit n s s' k vs _ rd_ Hrel Hn)
      as (m & last & vs' & Hm & Hlast & HQ).
    rewrite (Hsame last vs vs' HQ) in Hlast.
    exists (m + 1), rd. rewrite run_add, Hm. simpl. rewrite Hlast.
    reflexivity. }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (stuttering_ends m s s' b Hrel Hended)
    as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
  - right. exists n. exact Hwrong.
  - left.
    assert (Hs : behaves s (Exits k vs hp))
      by (exists n, rd; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

End Stuttering.

(* ======================================================================== *)
(* The same, with the heaps of two exits compared by the parameter          *)
(* ======================================================================== *)

Section Stuttering_heaps.

Variable R : state -> state -> Prop.

(* What is asked of the values and of the heaps at an exit, given the
   state that the second run exits from: the values and the heap of
   the first exit, then those of the second. *)
Variable Q : state -> list value -> heap -> list value -> heap -> Prop.

(* As [answered] above, but for the exit: the second state exits
   through the same continuation, with some values and some heap. *)
Definition answered_heaps (result : step_result) (s' : state) : Prop :=
  match result with
  | Undefined => True
  | Exit k vs hp _ =>
    ∃ (vs' : list value) (hp' : heap) (rd' : read_record),
      step s' = Exit k vs' hp' rd' ∧ Q s' vs hp vs' hp'
  | Step t => ∃ (n : nat) (t' : state), run (S n) s' = Step t' ∧ R t t'
  end.

Hypothesis R_step : ∀ s s' : state, R s s' -> answered_heaps (step s) s'.

(* An exit of the first run is answered by one of the second. *)
Lemma stuttering_heaps_exit (n : nat) :
  ∀ (s s' : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s s' ->
    run n s = Exit k vs hp rd ->
    ∃ (m : nat) (last : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m s' = Step last
      ∧ step last = Exit k vs' hp' rd'
      ∧ Q last vs hp vs' hp'.
Proof.
  induction n as [ | n IH]; intros s s' k vs hp rd Hrel; simpl;
    [ discriminate | ].
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k' vs0 hp0 rd0 | ]; unfold answered_heaps in
    Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel'). intros Hexit.
    destruct (IH t t' k vs hp rd Hrel' Hexit)
      as (m & last & vs' & hp' & rd' & Hm & Hlast & HQ).
    exists (S j + m), last, vs', hp', rd'. rewrite run_add, Hrun. auto.
  - intros [= <- <- <- <-].
    destruct Hanswer as (vs' & hp' & rd' & Hstep & HQ).
    exists 0, s', vs', hp', rd'. auto.
  - discriminate.
Qed.

(* A state that the first run reaches is related to one that the second
   reaches. *)
Lemma stuttering_heaps_reaches (n : nat) :
  ∀ s s' last : state,
    R s s' ->
    run n s = Step last ->
    ∃ (m : nat) (last' : state), run m s' = Step last' ∧ R last last'.
Proof.
  induction n as [ | n IH]; intros s s' last Hrel; simpl.
  - intros [= <-]. exists 0, s'. split; [ reflexivity | exact Hrel ].
  - pose proof (R_step s s' Hrel) as Hanswer.
    destruct (step s) as [t | k vs hp rd | ]; unfold answered_heaps in
      Hanswer;
      [ | discriminate | discriminate ].
    destruct Hanswer as (j & t' & Hrun & Hrel'). intros Hlast.
    destruct (IH t t' last Hrel' Hlast) as (m & last' & Hm & Hrel'').
    exists (S j + m), last'. rewrite run_add, Hrun. auto.
Qed.

(* If the second run has ended, so has the first. *)
Lemma stuttering_heaps_ends (m : nat) :
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
  pose proof (R_step s s' Hrel) as Hanswer.
  destruct (step s) as [t | k vs hp' rd' | ] eqn:Hstep;
    unfold answered_heaps in Hanswer.
  - destruct Hanswer as (j & t' & Hrun & Hrel').
    destruct (Nat.lt_ge_cases m (S j)) as [Hshort | Hlong].
    + exfalso. replace (S j) with (m + (S j - m)) in Hrun by lia.
      rewrite (ended_stable m _ s' b Hended) in Hrun.
      rewrite Hrun in Hended. destruct b; contradiction.
    + replace m with (S j + (m - S j)) in Hended by lia.
      rewrite run_add, Hrun in Hended.
      assert (Hless : m - S j < m) by lia.
      destruct (IH (m - S j) Hless t t' b Hrel' Hended) as [n Hn].
      exists (S n). simpl. rewrite Hstep. exact Hn.
  - exists 1. right. exists k, vs, hp', rd'. simpl. rewrite Hstep. reflexivity.
  - exists 1. left. simpl. rewrite Hstep. reflexivity.
Qed.

(* Where the values and the heap of the second exit are those of the
   first, for each exit that the first state has, the second state
   refines the first. *)
Lemma stuttering_heaps_refines (s s' : state) :
  (∀ (k : continuation) (vs : list value) (hp : heap) (last : state)
     (vs' : list value) (hp' : heap),
    behaves s (Exits k vs hp) ->
    Q last vs hp vs' hp' ->
    vs' = vs ∧ hp' = hp) ->
  R s s' ->
  refines s s'.
Proof.
  intros Hsame Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp Hs. pose proof Hs as [n [rd Hn]].
    destruct (stuttering_heaps_exit n s s' k vs hp rd Hrel Hn)
      as (m & last & vs' & hp' & rd' & Hm & Hlast & HQ).
    destruct (Hsame k vs hp last vs' hp' Hs HQ) as [-> ->].
    exists (m + 1), rd'. rewrite run_add, Hm. simpl. rewrite Hlast.
    reflexivity. }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (stuttering_heaps_ends m s s' b Hrel Hended)
    as [n [Hwrong | (k & vs & hp & rd & Hexit)]].
  - right. exists n. exact Hwrong.
  - left.
    assert (Hs : behaves s (Exits k vs hp))
      by (exists n, rd; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

End Stuttering_heaps.
