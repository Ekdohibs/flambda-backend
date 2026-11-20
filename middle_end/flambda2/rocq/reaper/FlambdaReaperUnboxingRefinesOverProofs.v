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

(* The proofs of FlambdaReaperUnboxingRefinesOverSpec.v. The simulation
   is that of FlambdaReaperUnboxingRefinesCallsSimProofs.v, whose call of
   code that is not known is made at a shape that asks the plain result of
   an over-application only of a call that is not of known arity
   ([ind_targets_k]): a call of known arity is a full call or none
   ([known_arity_no_over]). FlambdaReaperUnboxingRefinesCallsProofs.v
   relates the two rebuilt units on the fragment [calls_unboxing_over_nr]
   ([rebuild_unbox_modes_refines_cc_over]), and its statements follow, as
   [calls_unboxing_nr] is in it ([calls_unboxing_in_over_nr]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperLeast
  FlambdaReaperLocalFields FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesCallsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Theorem calls_unboxing_in_over : calls_unboxing_in_over_statement.
Proof. exact calls_unboxing_in_over_nr. Qed.

Theorem rebuild_unbox_over_refines_cc :
  rebuild_unbox_over_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_modes_refines_cc_over flag is_local u e T
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_over_refines_observably_local :
  rebuild_unbox_over_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_over_refines_cc flag is_local u e T
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_over_refines_observably :
  rebuild_unbox_over_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hcalls Hcheck.
  exact (rebuild_unbox_over_refines_observably_local flag is_local u e T
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hcalls Hcheck).
Qed.

Module UE := FlambdaReaperUnboxing.Examples.

Theorem known_arity_example : known_arity_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_over KnownArityExample.facts
                     KnownArityExample.tables KnownArityExample.cannot_change
                     KnownArityExample.code_deps [] []
                     KnownArityExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads KnownArityExample.facts
                     KnownArityExample.tables [] []
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed KnownArityExample.facts KnownArityExample.tables
      KnownArityExample.cannot_change KnownArityExample.code_deps [] [] true
      KnownArityExample.unit_expr
    = KnownArityExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment KnownArityExample.unit_expr = true
    ∧ any_kind_arguments KnownArityExample.unit_expr = true
    ∧ results_kinded [] [] KnownArityExample.unit_expr = true
    ∧ well_formed_unit UE.interface KnownArityExample.unit_expr = true
    ∧ exits_one_value UE.interface KnownArityExample.unit_expr = true
    ∧ check_closed UE.no_local KnownArityExample.facts
        KnownArityExample.tables
      = true
    ∧ check_field_used_exact UE.no_local KnownArityExample.facts
        KnownArityExample.tables
      = true
    ∧ check_cc_closed KnownArityExample.facts KnownArityExample.tables
        KnownArityExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        KnownArityExample.facts KnownArityExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ exact Hfixed | ].
  rewrite <- Hfixed.
  apply (rebuild_unbox_over_refines_observably Auto UE.no_local
           UE.interface KnownArityExample.unit_expr
           KnownArityExample.tables KnownArityExample.cannot_change [] [] true
           (fun _ => false)); try assumption.
  left. intros f. reflexivity.
Qed.
