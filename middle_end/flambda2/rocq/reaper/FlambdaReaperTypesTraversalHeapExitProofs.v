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

(* The exit of the simulation of the local unboxing with the primitives
   of the heap (FlambdaReaperUnboxingRefinesHeapProofs.relatedh): where
   the run of the unit rebuilt without unboxing exits, the run of the
   unit rebuilt with it exits through the same continuation, with the
   same values and the same heap, from a state with the same static
   environment. For FlambdaReaperTypesTraversalHeapProofs.v. This file
   needs no review.

   The two rebuilt terms are related by [ubh]
   (FlambdaReaperUnboxingRefinesHeapProofs.rebuild_unbox_ubh_heap), whose
   simulation relates the states of the two runs ([relatedh]) with one
   static environment. [shrinking_exit_state]: in a simulation of
   FlambdaReaperRebuildProofs ([answered]), the state from which the
   first run exits is related to one from which the second exits. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaFreshQuantifierSpec
  FlambdaPrimitiveProofs FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaHeapProofs FlambdaReaperUnboxingRefinesHeapProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module URH := FlambdaReaperUnboxingRefinesHeapProofs.

(* ======================================================================== *)
(* The state of an exit, in a simulation                                    *)
(* ======================================================================== *)

Section Exit_state.

Variable R : state -> state -> Prop.
Variable E : list value -> list value -> Prop.
Variable measure : state -> nat.
Hypothesis R_answered : ∀ s t : state, R s t -> answered R E measure s t.

(* A state that the first run comes to, and from which it exits, is
   related to one that the second run comes to. *)
Lemma shrinking_reaches_exit (n : nat) :
  ∀ (s t s1 : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s t -> run n s = Step s1 -> step s1 = Exit k vs hp rd ->
    ∃ (m : nat) (t1 : state), run m t = Step t1 ∧ R s1 t1.
Proof.
  induction n as [ | n IH]; intros s t s1 k vs hp rd Hrel Hrun Hexit.
  - injection Hrun as <-. exists 0, t. split; [ reflexivity | exact Hrel ].
  - rewrite run_succ in Hrun. pose proof (R_answered s t Hrel) as Ha.
    unfold answered in Ha.
    destruct (step s) as [s' | k' vs' hp' rd' | ] eqn:Hs;
      [ | discriminate Hrun | discriminate Hrun ].
    destruct Ha as [Hund | [(t' & Ht & Hrel') | [(_ & Hrel')
                                                | (t' & n0 & Ht & Hrel')]]].
    + exfalso. destruct n as [ | n].
      * injection Hrun as <-. rewrite Hund in Hexit. discriminate Hexit.
      * rewrite run_succ, Hund in Hrun. discriminate Hrun.
    + destruct (IH s' t' s1 k vs hp rd Hrel' Hrun Hexit)
        as (m & t1 & Hm & Hr1).
      exists (S m), t1. rewrite run_succ, Ht. split; [ exact Hm | exact Hr1 ].
    + exact (IH s' t s1 k vs hp rd Hrel' Hrun Hexit).
    + destruct (IH s' t' s1 k vs hp rd Hrel' Hrun Hexit)
        as (m & t1 & Hm & Hr1).
      exists (S (S n0) + m), t1. rewrite run_add, Ht.
      split; [ exact Hm | exact Hr1 ].
Qed.

(* The state from which the first run exits is related to one from which
   the second exits, through the same continuation, with the same heap
   and values related by [E]. *)
Lemma shrinking_exit_state (n : nat) (s t s1 : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  R s t -> run n s = Step s1 -> step s1 = Exit k vs hp rd ->
  ∃ (m : nat) (t1 : state) (vs' : list value) (rd' : read_record),
    run m t = Step t1 ∧ R s1 t1 ∧ step t1 = Exit k vs' hp rd' ∧ E vs vs'.
Proof.
  intros Hrel Hrun Hexit.
  destruct (shrinking_reaches_exit n s t s1 k vs hp rd Hrel Hrun Hexit)
    as (m & t1 & Hm & Hr1).
  pose proof (R_answered s1 t1 Hr1) as Ha. unfold answered in Ha.
  rewrite Hexit in Ha. destruct Ha as (vs' & rd' & Ht1 & HE).
  exists m, t1, vs', rd'.
  split; [ exact Hm | split; [ exact Hr1 | split; [ exact Ht1 | exact HE ] ] ].
Qed.

End Exit_state.

(* ======================================================================== *)
(* The exit                                                                 *)
(* ======================================================================== *)

(* Where the run of the unit rebuilt without unboxing exits, the run of
   the unit rebuilt with unboxing exits through the same continuation,
   with the same values and heap, from a state with the same static
   environment. *)
Lemma unbox_exit_heap (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  run n
    (State
       (rebuild_unit_from_tables (traverse_unit flag u e) T
          preserve_direct_calls e)
       [] [] empty_static_env empty_heap no_reads)
  = Step s ->
  step s = Exit k vs hp rd ->
  ∃ (m : nat) (t : state) (rd' : read_record),
    run m
      (State
         (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
            code_deps dominated to_unbox preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)
    = Step t
    ∧ step t = Exit k vs hp rd' ∧ state_statics t = state_statics s.
Proof.
  intros Hslots Hfragment Hclosed Hlocal Hcheck Hrun Hexit.
  destruct (URH.rebuild_unbox_ubh_heap None flag is_local u e T
              cannot_change code_deps dominated to_unbox preserve_direct_calls
              Hslots Hfragment Hclosed Hlocal Hcheck)
    as (M & Hdistinct & Hub).
  assert (Hrel :
    URH.relatedh M
      (State
         (rebuild_unit_from_tables (traverse_unit flag u e) T
            preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)
      (State
         (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
            code_deps dominated to_unbox preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)).
  { exists []. split; [ exact Hub | split; [ apply URP.env_rel_empty | ] ].
    split; [ constructor | repeat split ]. }
  destruct (shrinking_exit_state (URH.relatedh M) eq measure
              (URH.relatedh_answered M Hdistinct) n _ _ s k vs hp rd Hrel
              Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Hr & Ht & <-).
  exists m, t, rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  destruct Hr as (K & _ & _ & _ & Hst & _). symmetry. exact Hst.
Qed.
