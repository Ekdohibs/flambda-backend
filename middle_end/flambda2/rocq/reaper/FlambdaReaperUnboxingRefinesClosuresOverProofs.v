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


(* The proofs of FlambdaReaperUnboxingRefinesClosuresOverSpec.v.

   The relation of the two rebuilds is built in
   FlambdaReaperUnboxingRefinesClosuresProofs.v, by induction on a unit of
   the fragment ([rebuild_repr_over_modes_with_refines_cc]): a call that
   may over-apply ([over_call_rebuilt]) is related by
   FlambdaReaperUnboxingRefinesClosuresSimProofs.rr_call_over, whose code
   and the code that what is left of the over-application may call
   ([REM]) have their [my_closure] plain; what is left of an
   over-application is related in the frames of the two runs
   ([cont_rel_over]). The run of the first rebuilt unit is as
   [flow_reach2] and [rem_reach] say ([first_flow_o]: the flow of
   FlambdaReaperUnboxingRefinesClosuresReachProofs.first_flow, and
   FlambdaReaperUnboxingRefinesReach3Proofs.run_rem_used for what is left
   of an over-application). The statements up to observation compose the
   first with the refinements of the change of calling conventions. The
   example is the statement applied, and runs. *)

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
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesClosuresReachSpec
  FlambdaReaperUnboxingRefinesClosuresWrapsSpec
  FlambdaReaperUnboxingRefinesClosuresOverSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesClosuresProofs
  FlambdaReaperUnboxingRefinesClosuresSimProofs
  FlambdaReaperUnboxingRefinesClosuresReachProofs
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperCallingConventionsProofs
  FlambdaReaperFlowProofs FlambdaReaperLocalFieldsProofs FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesProofs.
Import ListNotations.

Module RCP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module RSP := FlambdaReaperUnboxingRefinesClosuresSimProofs.
Module RRP := FlambdaReaperUnboxingRefinesClosuresReachProofs.
Module R3P := FlambdaReaperUnboxingRefinesReach3Proofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module TP := FlambdaReaperTypesProofs.

Theorem repr_over_holds_wraps : repr_over_holds_wraps_statement.
Proof.
  intros nr is_local F T cannot_change code_deps dominated to_unbox tcr reach
    rem e.
  exact (RCP.repr_wraps_within_over_unit nr is_local F T cannot_change
           code_deps dominated to_unbox tcr reach rem e).
Qed.

(* ======================================================================== *)
(* The flow of the first rebuilt unit                                       *)
(* ======================================================================== *)

(* The run of the first rebuilt unit, as
   FlambdaReaperUnboxingRefinesClosuresReachProofs.first_flow has it, and
   what is left of an over-application calls code of [rem_reach] only
   (FlambdaReaperUnboxingRefinesReach3Proofs.run_rem_used). *)
Lemma first_flow_o (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e1 : expr)
    (T1 : tables) :
  is_local_slots_only is_local ->
  in_heap_fragment e1 = true ->
  check_closed is_local (traverse_unit flag u e1) T1 = true ->
  check_local_reads_used is_local (traverse_unit flag u e1) T1 = true ->
  RSP.flow_run
    (fun b f id => flow_reach2 (traverse_unit flag u e1) T1 b f id = true)
    (fun id : code_id => rem_reach (traverse_unit flag u e1) T1 id = true)
    (State e1 [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hfrag1 Hclosed1 Hlrs1 n s Hrun. split.
  - exact (proj1 (RRP.first_flow flag is_local u e1 T1 Hslots Hfrag1 Hclosed1
                    Hlrs1 n s Hrun)).
  - pose proof (R3P.run_rem_used flag is_local u e1 T1 n Hslots Hfrag1
                  (check_closed_sound is_local _ _ Hclosed1)
                  (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                     is_local _ _ Hlrs1))
      as Hrem.
    rewrite Hrun in Hrem.
    exact (R3P.step_rem_ok_mono _ _ s
             (R3P.rem_used_reach (traverse_unit flag u e1) T1) Hrem).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_over_refines_cc :
  rebuild_repr_over_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_over_modes_with_refines_cc None flag is_local u e
           T cannot_change dominated to_unbox tcr pdc nr
           (flow_reach2
              (traverse_unit flag u
                 (first_rebuild flag u e T cannot_change pdc))
              T1)
           (rem_reach
              (traverse_unit flag u
                 (first_rebuild flag u e T cannot_change pdc))
              T1)
           Hslots Hfrag Hclosed Hrepr Hcheck
           (first_flow_o flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_over_local : rebuild_repr_over_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_over_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr pdc nr Hslots Hfrag
              Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_over_observably :
  rebuild_repr_over_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_over_local flag is_local u e T T1 cannot_change
           dominated to_unbox tcr pdc nr Hslots Hfrag Hany Hrk Hwf Hone
           Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck).
Qed.

Theorem rebuild_repr_over_with_types_refines_cc :
  rebuild_repr_over_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_over_modes_with_refines_cc
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
           (rem_reach
              (traverse_unit flag u
                 (FlambdaReaperTypesSpec.first_rebuild_with_types flag
                    is_local u e T cannot_change b))
              T1)
           Hslots Hfrag Hclosed Hrepr Hcheck
           (first_flow_o flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_over_with_types_local :
  rebuild_repr_over_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (TP.rebuild_cc_with_types_refines_observably_local
              flag is_local u e T [] cannot_change b Hleast Hslots Hfrag Hany
              Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_over_with_types_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr b nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_over_with_types_observably :
  rebuild_repr_over_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_over_with_types_local flag is_local u e T T1
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

Theorem repr_over_example : repr_over_example_statement.
Proof.
  do 14 (split; [ vm_compute; reflexivity | ]).
  apply (rebuild_repr_over_observably Auto no_local interface
           OverExample.unit_expr OverExample.tables OverExample.tables1
           OverExample.cannot_change OverExample.dominated
           OverExample.to_unbox OverExample.tcr true (fun _ => false));
    try (vm_compute; reflexivity).
  - exact RCP.no_local_slots.
  - left. intros f. reflexivity.
Qed.
