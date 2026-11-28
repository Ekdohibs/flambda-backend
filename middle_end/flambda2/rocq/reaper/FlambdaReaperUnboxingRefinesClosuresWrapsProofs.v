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


(* The proofs of FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v.

   The relation of the two rebuilds is built in
   FlambdaReaperUnboxingRefinesClosuresProofs.v, by induction on a unit of
   the fragment ([rebuild_repr_wraps_modes_with_refines_cc]): a call through
   wrappers ([wraps_rebuilt]) is related by
   FlambdaReaperUnboxingRefinesClosuresSimProofs.rr_call_wraps, the two
   wrappers giving the continuation the same values (the arguments of the
   second rebuild's wrapper are those of the first's, of other parameters:
   [wrapper_args_u_fixed], [wrapper_args_rename]); the parameters of the
   wrappers are names after those of the unit, at which the environments
   within the wrappers are not related. The fragment of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v is within it
   ([repr_reach_within_wraps_unit]). The flow of the first rebuilt unit is
   that of FlambdaReaperUnboxingRefinesClosuresReachProofs.first_flow, and
   the statements up to observation compose the first with the refinements
   of the change of calling conventions, as there. The example is the
   statement applied, and runs. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionProofs
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesClosuresReachSpec
  FlambdaReaperUnboxingRefinesClosuresWrapsSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesClosuresProofs
  FlambdaReaperUnboxingRefinesClosuresReachProofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperCallingConventionsProofs
  FlambdaReaperFlowProofs FlambdaReaperTypes FlambdaReaperTypesSpec
  FlambdaReaperTypesProofs.
Import ListNotations.

Module RCP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module RRP := FlambdaReaperUnboxingRefinesClosuresReachProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module TP := FlambdaReaperTypesProofs.

Theorem repr_wraps_holds_reach : repr_wraps_holds_reach_statement.
Proof.
  intros nr is_local F T cannot_change code_deps dominated to_unbox tcr reach
    e.
  exact (RCP.repr_reach_within_wraps_unit nr is_local F T cannot_change
           code_deps dominated to_unbox tcr reach e).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_wraps_refines_cc :
  rebuild_repr_wraps_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_wraps_modes_with_refines_cc None flag is_local u e
           T cannot_change dominated to_unbox tcr pdc nr
           (flow_reach2
              (traverse_unit flag u
                 (first_rebuild flag u e T cannot_change pdc))
              T1)
           Hslots Hfrag Hclosed Hrepr Hcheck
           (RRP.first_flow flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_wraps_local : rebuild_repr_wraps_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_wraps_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr pdc nr Hslots Hfrag
              Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_wraps_observably :
  rebuild_repr_wraps_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_wraps_local flag is_local u e T T1 cannot_change
           dominated to_unbox tcr pdc nr Hslots Hfrag Hany Hrk Hwf Hone
           Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck).
Qed.

Theorem rebuild_repr_wraps_with_types_refines_cc :
  rebuild_repr_wraps_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_wraps_modes_with_refines_cc
           (FlambdaReaperTypes.subkinds_of_context
              (FlambdaReaperTypes.plain_context is_local
                 (traverse_unit flag u e) T [])
              (unit_code_deps flag u e) e)
           flag is_local u e T cannot_change dominated to_unbox tcr b nr
           (flow_reach2
              (traverse_unit flag u
                 (FlambdaReaperTypesSpec.first_rebuild_with_types flag
                    is_local u e T cannot_change b))
              T1)
           Hslots Hfrag Hclosed Hrepr Hcheck
           (RRP.first_flow flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_wraps_with_types_local :
  rebuild_repr_wraps_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (TP.rebuild_cc_with_types_refines_observably_local
              flag is_local u e T [] cannot_change b Hleast Hslots Hfrag Hany
              Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_wraps_with_types_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr b nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_wraps_with_types_observably :
  rebuild_repr_wraps_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_wraps_with_types_local flag is_local u e T T1
           cannot_change dominated to_unbox tcr b nr Hleast Hslots Hfrag Hany
           Hrk Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

Theorem repr_wraps_example : repr_wraps_example_statement.
Proof.
  assert (Heq : rebuild_unit_unbox_repr_fixed no_local WrapsExample.facts
                  WrapsExample.tables WrapsExample.cannot_change
                  WrapsExample.code_deps WrapsExample.dominated
                  WrapsExample.to_unbox WrapsExample.tcr true
                  WrapsExample.unit_expr
                = WrapsExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hrepr : repr_flow_wraps_nr (fun _ => false) no_local
                    WrapsExample.facts WrapsExample.tables
                    WrapsExample.cannot_change WrapsExample.code_deps
                    WrapsExample.dominated WrapsExample.to_unbox
                    WrapsExample.tcr WrapsExample.facts1
                    WrapsExample.tables1 WrapsExample.unit_expr
                  = true)
    by (vm_compute; reflexivity).
  do 8 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hrepr | ].
  do 2 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Heq | ].
  do 3 (split; [ vm_compute; reflexivity | ]).
  rewrite <- Heq.
  apply (rebuild_repr_wraps_observably Auto no_local interface
           WrapsExample.unit_expr WrapsExample.tables WrapsExample.tables1
           WrapsExample.cannot_change WrapsExample.dominated
           WrapsExample.to_unbox WrapsExample.tcr true (fun _ => false));
    try (vm_compute; reflexivity).
  - exact RCP.no_local_slots.
  - left. intros f. reflexivity.
Qed.
