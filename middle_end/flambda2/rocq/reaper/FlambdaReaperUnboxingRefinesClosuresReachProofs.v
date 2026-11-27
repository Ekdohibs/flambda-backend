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


(* The proofs of FlambdaReaperUnboxingRefinesClosuresReachSpec.v.

   The relation of the two rebuilds is built in
   FlambdaReaperUnboxingRefinesClosuresProofs.v, by induction on a unit of
   the fragment ([rebuild_repr_reach_modes_with_refines_cc], for any
   rewriting of what the parameters kept state and any flow [reach]),
   where the run of the first rebuilt unit is as [reach] says
   (FlambdaReaperUnboxingRefinesClosuresSimProofs.flow_run); the fragment
   of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v is within it
   ([repr_calls_within_reach_unit]). Here [flow_run] is proved of the
   first rebuilt unit, with or without types, with [reach] read off its
   own analysis ([flow_reach2]), as in
   FlambdaReaperUnboxingRefinesReach2Proofs.v: the run of
   FlambdaReaperFlowProofs.v puts every closure that a variable holds at
   an allocation that may come to the variable, with the witness of known
   arity of its code ([closure_reach]), and the closure of a partial
   application at the chain of the calls that made it ([partial_reach]);
   the partial applications are those of the machine. The statements up to
   observation compose the first with the refinements of the change of
   calling conventions, with types or without. The example is the
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
  FlambdaReaperUnboxingRefinesClosuresReachSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesClosuresProofs
  FlambdaReaperUnboxingRefinesClosuresSimProofs
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperCallingConventionsProofs
  FlambdaReaperFlowProofs FlambdaReaperLocalFieldsProofs FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesProofs.
Import ListNotations.

Module RCP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module RSP := FlambdaReaperUnboxingRefinesClosuresSimProofs.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module TP := FlambdaReaperTypesProofs.
Module R1P := FlambdaReaperUnboxingRefinesReachProofs.
Module R2P := FlambdaReaperUnboxingRefinesReach2Proofs.

Theorem repr_reach_holds_calls : repr_reach_holds_calls_statement.
Proof.
  intros nr is_local F T cannot_change code_deps dominated to_unbox tcr reach
    e.
  exact (RCP.repr_calls_within_reach_unit nr is_local F T cannot_change
           code_deps dominated to_unbox tcr reach e).
Qed.

(* ======================================================================== *)
(* The flow of the first rebuilt unit                                       *)
(* ======================================================================== *)

(* The run of a unit [e1] of the fragment of the heap, whose tables [T1]
   are closed and read local fields where used: at each call of a
   variable [f], what [f] holds has its partial applications made by the
   machine, and the code of its closure, or of the closure that a call of
   it calls, is that of [flow_reach2] of [f]. *)
Lemma first_flow (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e1 : expr)
    (T1 : tables) :
  is_local_slots_only is_local ->
  in_heap_fragment e1 = true ->
  check_closed is_local (traverse_unit flag u e1) T1 = true ->
  check_local_reads_used is_local (traverse_unit flag u e1) T1 = true ->
  RSP.flow_run
    (fun b f id => flow_reach2 (traverse_unit flag u e1) T1 b f id = true)
    (State e1 [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hfrag1 Hclosed1 Hlrs1 n s Hrun ap f v _ _ Hf.
  set (F1 := traverse_unit flag u e1) in *.
  pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
  destruct (R2P.run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1 Hc1
              (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                 is_local _ _ Hlrs1))
    as (located & cds & Hcds & Hnd & Hflows).
  fold F1 in Hc1, Hcds, Hflows. rewrite Hrun in Hflows.
  destruct Hflows as ((Hvars & _ & Hst & _) & Hchains & Hpa).
  assert (Hclosure :
    ∀ (v0 : value) (id0 : code_id) (slots0 : list (value_slot * value)),
      lookup f (state_vars s) = Some v0 ->
      closure_of (state_statics s) v0 = Some (id0, slots0) ->
      flow_reach F1 T1 f id0 = true)
    by (intros v0 id0 slots0 Hf0 Hclo0;
        exact (R1P.closure_reach F1 T1 cds (unit_kept e1) located
                 (state_vars s) (state_statics s) f v0 id0 slots0 Hcds Hvars
                 Hst Hf0 Hclo0)).
  split.
  { intros cl applied aa ->.
    destruct (Hpa f _ Hf) as ((id & slots & c & Hcl & Hc & Hlt & _) & _).
    exists id, slots, c. exact (conj Hcl (conj Hc Hlt)). }
  intros id slots. split; intros Hclo; cbn [flow_reach2].
  - exact (Hclosure v id slots Hf Hclo).
  - destruct v as [c | code vs0 | slot decls vs | sym | tag fields | elements
                   | contents | l | cl applied aa];
      try (unfold flow_reach_u; rewrite (Hclosure _ id slots Hf Hclo);
           reflexivity).
    exact (R2P.partial_reach is_local F1 T1 cds (unit_kept e1) located
             (state_statics s) f cl applied aa id slots Hc1 Hcds Hnd Hst
             Hchains (Hpa f _ Hf) (Hvars f _ Hf) Hclo).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_repr_reach_refines_cc :
  rebuild_repr_reach_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_reach_modes_with_refines_cc None flag is_local u e
           T cannot_change dominated to_unbox tcr pdc nr
           (flow_reach2
              (traverse_unit flag u
                 (first_rebuild flag u e T cannot_change pdc))
              T1)
           Hslots Hfrag Hclosed Hrepr Hcheck
           (first_flow flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_reach_local : rebuild_repr_reach_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_reach_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr pdc nr Hslots Hfrag
              Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_reach_observably :
  rebuild_repr_reach_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_reach_local flag is_local u e T T1 cannot_change
           dominated to_unbox tcr pdc nr Hslots Hfrag Hany Hrk Hwf Hone
           Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck).
Qed.

Theorem rebuild_repr_reach_with_types_refines_cc :
  rebuild_repr_reach_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (RCP.rebuild_repr_reach_modes_with_refines_cc
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
           (first_flow flag is_local u _ T1 Hslots Hfrag1 Hclosed1 Hlrs1)).
Qed.

Theorem rebuild_repr_reach_with_types_local :
  rebuild_repr_reach_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (TP.rebuild_cc_with_types_refines_observably_local
              flag is_local u e T [] cannot_change b Hleast Hslots Hfrag Hany
              Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_repr_reach_with_types_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox tcr b nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck)).
Qed.

Theorem rebuild_repr_reach_with_types_observably :
  rebuild_repr_reach_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox tcr b nr
    Hleast Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc
    Hwid Hfrag1 Hclosed1 Hlrs1 Hrepr Hcheck.
  exact (rebuild_repr_reach_with_types_local flag is_local u e T T1
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

Theorem repr_reach_ind_call_example :
  repr_reach_ind_call_example_statement.
Proof.
  assert (Heq : rebuild_unit_unbox_repr_fixed no_local
                  ClosuresReachExample.facts ClosuresReachExample.tables
                  ClosuresReachExample.cannot_change
                  ClosuresReachExample.code_deps
                  ClosuresReachExample.dominated
                  ClosuresReachExample.to_unbox ClosuresReachExample.tcr true
                  ClosuresReachExample.unit_expr
                = ClosuresReachExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hrepr : repr_flow_unboxing_nr (fun _ => false) no_local
                    ClosuresReachExample.facts ClosuresReachExample.tables
                    ClosuresReachExample.cannot_change
                    ClosuresReachExample.code_deps
                    ClosuresReachExample.dominated
                    ClosuresReachExample.to_unbox ClosuresReachExample.tcr
                    ClosuresReachExample.facts1 ClosuresReachExample.tables1
                    ClosuresReachExample.unit_expr
                  = true)
    by (vm_compute; reflexivity).
  do 15 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hrepr | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Heq | ].
  do 2 (split; [ vm_compute; reflexivity | ]).
  rewrite <- Heq.
  apply (rebuild_repr_reach_observably Auto no_local interface
           ClosuresReachExample.unit_expr ClosuresReachExample.tables
           ClosuresReachExample.tables1 ClosuresReachExample.cannot_change
           ClosuresReachExample.dominated ClosuresReachExample.to_unbox
           ClosuresReachExample.tcr true (fun _ => false));
    try (vm_compute; reflexivity).
  - exact RCP.no_local_slots.
  - left. intros f. reflexivity.
Qed.

Theorem repr_reach_never_example : repr_reach_never_example_statement.
Proof.
  do 10 (split; [ vm_compute; reflexivity | ]).
  apply (rebuild_repr_reach_observably Auto no_local interface
           ClosuresReachNeverExample.unit_expr
           ClosuresReachNeverExample.tables
           ClosuresReachNeverExample.tables1
           ClosuresReachNeverExample.cannot_change
           ClosuresReachNeverExample.dominated
           ClosuresReachNeverExample.to_unbox ClosuresReachNeverExample.tcr
           true ClosuresReachNeverExample.nr);
    try (vm_compute; reflexivity).
  - exact RCP.no_local_slots.
  - left. intros f. reflexivity.
Qed.
