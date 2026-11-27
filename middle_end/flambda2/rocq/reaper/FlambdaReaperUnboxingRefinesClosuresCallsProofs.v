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


(* The proofs of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v.

   The relation of the two rebuilds is built in
   FlambdaReaperUnboxingRefinesClosuresProofs.v, by induction on a unit of
   the fragment of this file ([rebuild_repr_calls_modes_with_refines_cc],
   for any rewriting of what the parameters kept state), and the
   fragment of FlambdaReaperUnboxingRefinesClosuresSpec.v is within it
   ([repr_within_unit]). The statements up to observation compose the
   first with the refinements of the change of calling conventions, with
   types or without, as those of FlambdaReaperUnboxingRefinesClosuresSpec.v
   and FlambdaReaperTypesSpec.v do. The example is the statement applied,
   and runs. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesClosuresProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperCallingConventionsProofs
  FlambdaReaperFlowProofs FlambdaReaperTypes FlambdaReaperTypesProofs.
Import ListNotations.

Module RCP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module TP := FlambdaReaperTypesProofs.

Theorem repr_calls_holds_repr : repr_calls_holds_repr_statement.
Proof. exact RCP.repr_within_unit. Qed.

Theorem rebuild_repr_calls_refines_cc :
  rebuild_repr_calls_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr.
  exact (RCP.rebuild_repr_calls_modes_with_refines_cc None flag is_local u e
           T cannot_change dominated to_unbox tcr pdc nr).
Qed.

Theorem rebuild_repr_calls_local : rebuild_repr_calls_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hrepr
    Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_calls_refines_cc flag is_local u e T cannot_change
              dominated to_unbox tcr pdc nr Hslots Hfrag Hclosed Hrepr
              Hcheck)).
Qed.

Theorem rebuild_repr_calls_observably :
  rebuild_repr_calls_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hrepr
    Hcheck.
  exact (rebuild_repr_calls_local flag is_local u e T cannot_change
           dominated to_unbox tcr pdc nr Hslots Hfrag Hany Hrk Hwf Hone
           Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hrepr Hcheck).
Qed.

Theorem rebuild_repr_calls_with_types_refines_cc :
  rebuild_repr_calls_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr b nr
    Hslots Hfrag Hclosed Hrepr Hcheck.
  exact (RCP.rebuild_repr_calls_modes_with_refines_cc
           (FlambdaReaperTypes.subkinds_of_context
              (FlambdaReaperTypes.plain_context is_local
                 (traverse_unit flag u e) T [])
              (unit_code_deps flag u e) e)
           flag is_local u e T cannot_change dominated to_unbox tcr b nr
           Hslots Hfrag Hclosed Hrepr Hcheck).
Qed.

Theorem rebuild_repr_calls_with_types_local :
  rebuild_repr_calls_with_types_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr b nr Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hrepr
    Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (TP.rebuild_cc_with_types_refines_observably_local
              flag is_local u e T [] cannot_change b Hleast Hslots Hfrag Hany
              Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_calls_with_types_refines_cc flag is_local u e T
              cannot_change dominated to_unbox tcr b nr Hslots Hfrag Hclosed
              Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_calls_with_types_observably :
  rebuild_repr_calls_with_types_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr b nr Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hrepr Hcheck.
  exact (rebuild_repr_calls_with_types_local flag is_local u e T
           cannot_change dominated to_unbox tcr b nr Hleast Hslots Hfrag Hany
           Hrk Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hrepr Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

Theorem repr_calls_kept_code_example :
  repr_calls_kept_code_example_statement.
Proof.
  assert (Heq : rebuild_unit_unbox_repr_fixed no_local
                  ClosuresCallsExample.facts ClosuresCallsExample.tables
                  ClosuresCallsExample.cannot_change
                  ClosuresCallsExample.code_deps
                  ClosuresCallsExample.dominated
                  ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr true
                  ClosuresCallsExample.unit_expr
                = ClosuresCallsExample.rebuilt)
    by (vm_compute; reflexivity).
  do 11 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Heq | ].
  do 2 (split; [ vm_compute; reflexivity | ]).
  rewrite <- Heq.
  apply (rebuild_repr_calls_observably Auto no_local interface
           ClosuresCallsExample.unit_expr ClosuresCallsExample.tables
           ClosuresCallsExample.cannot_change ClosuresCallsExample.dominated
           ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr true
           (fun _ => false));
    try (vm_compute; reflexivity).
  - exact RCP.no_local_slots.
  - left. intros f. reflexivity.
Qed.
