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


(* Simulations in which the first state may take steps that the second
   does not answer. Nothing here needs review: it is a way of proving
   [refines], beside FlambdaSemanticsProofs.simulation_refines, which
   asks that the two states step together, and
   FlambdaSimulationProofs.stuttering_refines, in which the second run
   never takes fewer steps than the first.

   It serves where a rewrite takes steps away: a [Let] at the head of
   a handler that becomes a parameter, or the [Let]s that a rewrite
   put there, seen from the result. The relation has a number, which
   bounds the steps that the first state may still take alone: a step
   of the first is answered by a step of the second, to states related
   at any number, or is not answered, and the states are then related
   at a smaller number. So the first run cannot go on alone for ever,
   and a run of the second that has ended bounds the first. *)

From Stdlib Require Import ZArith List Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs.
Import ListNotations.

Section Lagging.

Variable R : nat -> state -> state -> Prop.

Hypothesis R_step :
  ∀ (n : nat) (s s' : state),
    R n s s' ->
    match step s with
    | Undefined => True
    | Exit k vs hp rd => step s' = Exit k vs hp rd
    | Step t =>
      (∃ (m : nat) (t' : state), step s' = Step t' ∧ R m t t')
      ∨ ∃ m : nat, m < n ∧ R m t s'
    end.

(* An exit of the first run is one of the second. *)
Lemma counted_lagging_exit (fuel : nat) :
  ∀ (n : nat) (s s' : state) (k : continuation) (vs : list value)
    (hp : heap) (rd : read_record),
    R n s s' ->
    run fuel s = Exit k vs hp rd ->
    ∃ m : nat, run m s' = Exit k vs hp rd.
Proof.
  induction fuel as [ | fuel IH]; intros n s s' k vs hp rd Hrel; simpl;
    [ discriminate | ].
  pose proof (R_step n s s' Hrel) as Hanswer.
  destruct (step s) as [t | k' vs' hp' rd' | ]; [ | | discriminate ].
  - intros Hexit.
    destruct Hanswer as [(m & t' & Hstep & Hrel') | (m & _ & Hrel')].
    + destruct (IH m t t' k vs hp rd_ Hrel' Hexit) as [j Hj].
      exists (S j). simpl. rewrite Hstep. exact Hj.
    + exact (IH m t s' k vs hp rd_ Hrel' Hexit).
  - intros [= <- <- <- ->]. exists 1. simpl. rewrite Hanswer. reflexivity.
Qed.

(* If the second run has ended, so has the first: a step of the first
   uses up one of the second's, or brings the number down. *)
Lemma counted_lagging_ends (m : nat) :
  ∀ (n : nat) (s s' : state) (b : behaviour),
    R n s s' ->
    ended (run m s') b ->
    ∃ j : nat,
      run j s = Undefined
      ∨ ∃ (k : continuation) (vs : list value) (hp : heap) (rd : read_record),
          run j s = Exit k vs hp rd.
Proof.
  induction m as [m IHm] using (well_founded_induction lt_wf).
  intros n. induction n as [n IHn] using (well_founded_induction lt_wf).
  intros s s' b Hrel Hended.
  pose proof (R_step n s s' Hrel) as Hanswer.
  destruct (step s) as [t | k vs hp rd | ] eqn:Hstep.
  - destruct Hanswer as [(n' & t' & Hstep' & Hrel') | (n' & Hless & Hrel')].
    + destruct m as [ | m]; [ destruct b; contradiction | ].
      simpl in Hended. rewrite Hstep' in Hended.
      destruct (IHm m (Nat.lt_succ_diag_r m) n' t t' b Hrel' Hended)
        as [j Hj].
      exists (S j). simpl. rewrite Hstep. exact Hj.
    + destruct (IHn n' Hless t s' b Hrel' Hended) as [j Hj].
      exists (S j). simpl. rewrite Hstep. exact Hj.
  - exists 1. right. exists k, vs, hp, rd. simpl. rewrite Hstep. reflexivity.
  - exists 1. left. simpl. rewrite Hstep. reflexivity.
Qed.

Lemma counted_lagging_refines (n : nat) (s s' : state) :
  R n s s' -> refines s s'.
Proof.
  intros Hrel.
  assert (Hexits :
    ∀ (k : continuation) (vs : list value) (hp : heap),
      behaves s (Exits k vs hp) -> behaves s' (Exits k vs hp)).
  { intros k vs hp [fuel [rd Hfuel]].
    destruct (counted_lagging_exit fuel n s s' k vs hp rd_ Hrel Hfuel)
      as [m Hm].
    exists m, rd. exact Hm. }
  split; [ exact Hexits | ].
  intros b Hterm Hb. destruct (behaves_ended s' b Hterm Hb) as [m Hended].
  destruct (counted_lagging_ends m n s s' b Hrel Hended)
    as [j [Hwrong | (k & vs & hp & rd & Hexit)]].
  - right. exists j. exact Hwrong.
  - left.
    assert (Hs : behaves s (Exits k vs hp))
      by (exists j, rd; exact Hexit).
    rewrite (behaviour_unique s' b (Exits k vs hp) Hb (Hexits k vs hp Hs)).
    exact Hs.
Qed.

End Lagging.
