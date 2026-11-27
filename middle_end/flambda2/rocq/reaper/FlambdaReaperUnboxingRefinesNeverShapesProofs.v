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

(* The proofs of FlambdaReaperUnboxingRefinesNeverShapesSpec.v, by
   computation: the units and their rebuilds exit alike, so that each
   refines the other ([FlambdaReaperTupledProofs.same_exit_refines]); the
   first shape, admitted, by the statements of
   FlambdaReaperUnboxingRefinesNever2Spec.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesNever2Spec
  FlambdaReaperUnboxingRefinesNeverShapesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperTupledProofs FlambdaReaperCallingConventionsSpec
  FlambdaReaperUnboxingRefinesNever2Proofs.
Import ListNotations.

(* Two programs that exit alike, in [n] steps at most. *)
Lemma exit_alike_refines (e e' : expr) (n : nat) (k : continuation)
    (vs : list value) :
  run n (State e [] [] empty_static_env empty_heap no_reads)
  = Exit k vs empty_heap no_reads ->
  run n (State e' [] [] empty_static_env empty_heap no_reads)
  = Exit k vs empty_heap no_reads ->
  program_refines e e' ∧ program_refines_observably e e'.
Proof.
  intros He He'.
  assert (Hr : program_refines e e').
  { unfold program_refines.
    apply (FlambdaReaperTupledProofs.same_exit_refines _ _ k vs empty_heap);
      unfold behaves; exists n, no_reads; assumption. }
  exact (conj Hr (FlambdaReaperRebuildProofs.refines_observably_of_refines
                    _ _ Hr)).
Qed.

Theorem never_partial_example : never_partial_example_statement.
Proof.
  assert (Hfirst : PartialNeverExample.first = PartialNeverExample.cc_rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hfirst | ].
  split; [ vm_compute; reflexivity | ].
  rewrite Hfirst.
  destruct (exit_alike_refines PartialNeverExample.cc_rebuilt
              PartialNeverExample.rebuilt 80 2%positive
              [Value_const (Tagged_immediate 5)]
              ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity))
    as [Hcc _].
  destruct (exit_alike_refines PartialNeverExample.unit_expr
              PartialNeverExample.rebuilt 80 2%positive
              [Value_const (Tagged_immediate 5)]
              ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity))
    as [_ Hobs].
  exact (conj Hcc Hobs).
Qed.

Theorem never_over_example : never_over_example_statement.
Proof.
  assert (Hfirst : OverNeverExample.first = OverNeverExample.cc_rebuilt)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hfirst | ].
  split; [ vm_compute; reflexivity | ].
  rewrite Hfirst.
  destruct (exit_alike_refines OverNeverExample.cc_rebuilt
              OverNeverExample.rebuilt 80 2%positive
              [Value_const (Tagged_immediate 6)]
              ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity))
    as [Hcc _].
  destruct (exit_alike_refines OverNeverExample.unit_expr
              OverNeverExample.rebuilt 80 2%positive
              [Value_const (Tagged_immediate 6)]
              ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity))
    as [_ Hobs].
  exact (conj Hcc Hobs).
Qed.

(* ======================================================================== *)
(* The first shape, admitted                                                *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.
Module N2P := FlambdaReaperUnboxingRefinesNever2Proofs.

Theorem never_partial2_example : never_partial2_example_statement.
Proof.
  assert (Hprem :
    in_flow_fragment PartialNeverExample.unit_expr = true
    ∧ any_kind_arguments PartialNeverExample.unit_expr = true
    ∧ results_kinded [] [] PartialNeverExample.unit_expr = true
    ∧ well_formed_unit UE.interface PartialNeverExample.unit_expr = true
    ∧ exits_one_value UE.interface PartialNeverExample.unit_expr = true
    ∧ check_closed UE.no_local PartialNeverExample.facts
        PartialNeverExample.tables
      = true
    ∧ check_field_used_exact UE.no_local PartialNeverExample.facts
        PartialNeverExample.tables
      = true
    ∧ check_cc_closed PartialNeverExample.facts PartialNeverExample.tables
        PartialNeverExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        PartialNeverExample.facts PartialNeverExample.tables
      = true
    ∧ in_heap_fragment PartialNeverExample.first = true
    ∧ check_closed UE.no_local PartialNeverExample.facts1
        PartialNeverExample.tables1
      = true
    ∧ check_local_reads_used UE.no_local PartialNeverExample.facts1
        PartialNeverExample.tables1
      = true
    ∧ check_unboxed_reads PartialNeverExample.facts
        PartialNeverExample.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  (* The checks as the statements have them, which the elaborator is slow
     to convert the computed ones to. *)
  assert (Hcalls' :
    calls_unboxing_never2_nr PartialNeverExample.nr
      (traverse_unit Auto UE.interface PartialNeverExample.unit_expr)
      PartialNeverExample.tables PartialNeverExample.cannot_change
      (unit_code_deps Auto UE.interface PartialNeverExample.unit_expr) [] []
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface PartialNeverExample.unit_expr
            PartialNeverExample.tables PartialNeverExample.cannot_change
            true))
      PartialNeverExample.tables1 PartialNeverExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  - exact (N2P.rebuild_unbox_nr_ind2_refines_cc Auto UE.no_local UE.interface
             PartialNeverExample.unit_expr PartialNeverExample.tables
             PartialNeverExample.tables1 PartialNeverExample.cannot_change []
             [] true PartialNeverExample.nr Hslots Hfrag Hclosed Hfrag1
             Hclosed1 Hlrs1 Hcalls' Hreads).
  - exact (N2P.rebuild_unbox_nr_ind2_refines_observably Auto UE.no_local
             UE.interface PartialNeverExample.unit_expr
             PartialNeverExample.tables PartialNeverExample.tables1
             PartialNeverExample.cannot_change [] [] true
             PartialNeverExample.nr Hslots (or_introl (fun _ => eq_refl))
             Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1
             Hlrs1 Hcalls' Hreads).
Qed.
