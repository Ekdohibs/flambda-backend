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

(* The proofs of FlambdaReaperUnboxingRefinesNever2TypesSpec.v. Nothing
   here needs review.

   As those of FlambdaReaperTypesProofs.v (the statements with types of the
   fragment of FlambdaReaperUnboxingRefinesNeverSpec.v) and of
   FlambdaReaperUnboxingRefinesCallsHeapProofs.v, which they repeat, with
   FlambdaReaperUnboxingRefinesCallsProofs
   .rebuild_unbox_modes_with_refines_cc_never2_gen and
   FlambdaReaperUnboxingRefinesNever2Proofs.unit_never2_covered. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperLeastSpec FlambdaUnusedStaticsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec FlambdaReaperTypes
  FlambdaReaperUnboxingRefinesCallsHeapSpec
  FlambdaReaperUnboxingRefinesNever2Spec
  FlambdaReaperUnboxingRefinesNever2TypesSpec
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaStaticsInScopeSpec FlambdaReaperNeverReturnsSpec
  FlambdaReaperFlowProofs FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventionsProofs FlambdaReaperLocalFieldsProofs
  FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesHeapProofs
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesOver2Proofs
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesNever2Proofs FlambdaReaperTypesProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module URCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module URHP := FlambdaReaperUnboxingRefinesHeapProofs.
Module CCP := FlambdaReaperCallingConventionsProofs.
Module TP := FlambdaReaperTypesProofs.
Module URR1 := FlambdaReaperUnboxingRefinesReachProofs.
Module URR2 := FlambdaReaperUnboxingRefinesReach2Proofs.
Module URO2 := FlambdaReaperUnboxingRefinesOver2Proofs.
Module URR3 := FlambdaReaperUnboxingRefinesReach3Proofs.
Module N2P := FlambdaReaperUnboxingRefinesNever2Proofs.

(* ======================================================================== *)
(* With types                                                               *)
(* ======================================================================== *)

Lemma rebuild_unbox_nr_ind2_with_types_cc_gen (hb : bool) :
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    flow_fragment hb (unit_codes e) (static_names symbols_of_binding e) e
    && distinct (static_names code_ids_of_binding e) = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaUnusedStaticsSpec.program_refines
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hdist Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild_with_types, rebuild_unit_cc_with_types_from_tables,
    rebuild_unit_cc_with_types in *.
  set (types := subkinds_of_context
                  (plain_context is_local (traverse_unit flag u e) T [])
                  (FlambdaReaperCallingConventions.unit_code_deps flag u e) e)
    in *.
  set (e1 := FlambdaReaperCallingConventions.rebuild_unit_cc_with types
               (traverse_unit flag u e) (solution_of_tables T)
               (fun n => mem_node n cannot_change)
               (FlambdaReaperCallingConventions.unit_code_deps flag u e) pdc e)
    in *.
  set (F1 := traverse_unit flag u e1) in *.
  pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
  pose proof (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                is_local _ _ Hlrs1) as Hlr1.
  apply (URCP.rebuild_unbox_modes_with_refines_cc_never2_gen hb types flag
           is_local u e T cannot_change dominated to_unbox pdc nr
           (FlambdaReaperUnboxingRefinesReach2Spec.flow_reach2 F1 T1)
           (FlambdaReaperUnboxingRefinesOver2Spec.over_reach F1 T1)
           (FlambdaReaperUnboxingRefinesReach3Spec.rem_reach F1 T1)
           (TP.subkinds_of_context_unused is_local _ T [] _ e)
           (TP.subkinds_of_context_conts _ _ _ Hdist) Hslots Hfrag Hclosed
           Hcalls Hcheck).
  - (* The flow, as in FlambdaReaperUnboxingRefinesReach2Proofs.v. *)
    intros n s Hrun ap f v id slots _ _ Hf.
    destruct (URR2.run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1
                Hc1 Hlr1)
      as (located & cds & Hcds & Hnd & Hflows).
    fold F1 in Hcds, Hflows. fold e1 in Hrun. rewrite Hrun in Hflows.
    destruct Hflows as ((Hvars & _ & Hst & _) & Hchains & Hpa).
    assert (Hclosure :
      ∀ (v0 : value) (id0 : code_id) (slots0 : list (value_slot * value)),
        lookup f (state_vars s) = Some v0 ->
        closure_of (state_statics s) v0 = Some (id0, slots0) ->
        FlambdaReaperUnboxingRefinesReachSpec.flow_reach F1 T1 f id0 = true)
      by (intros v0 id0 slots0 Hf0 Hclo0;
          exact (URR1.closure_reach F1 T1 cds
                   (FlambdaReaperFlowSpec.unit_kept e1) located
                   (state_vars s) (state_statics s) f v0 id0 slots0 Hcds
                   Hvars Hst Hf0 Hclo0)).
    split; intros Hclo;
      cbn [FlambdaReaperUnboxingRefinesReach2Spec.flow_reach2].
    + exact (Hclosure v id slots Hf Hclo).
    + destruct v as [c | code vs0 | slot decls vs | sym | tag fields
                     | elements | contents | l | cl applied aa];
        try (unfold FlambdaReaperUnboxingRefinesReach2Spec.flow_reach_u;
             rewrite (Hclosure _ id slots Hf Hclo); reflexivity).
      exact (URR2.partial_reach is_local F1 T1 cds
               (FlambdaReaperFlowSpec.unit_kept e1) located
               (state_statics s) f cl applied aa id slots Hc1 Hcds Hnd Hst
               Hchains (Hpa f _ Hf) (Hvars f _ Hf) Hclo).
  - (* What a step over-applies. *)
    intros n s Hrun.
    pose proof (URO2.run_over_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                  Hlr1)
      as Hover.
    fold F1 in Hover. fold e1 in Hrun. rewrite Hrun in Hover.
    exact (URO2.step_over_ok_mono _ _ s (URO2.over_used_reach F1 T1) Hover).
  - (* What is left of an over-application calls. *)
    intros n s Hrun.
    pose proof (URR3.run_rem_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                  Hlr1)
      as Hrem.
    fold F1 in Hrem. fold e1 in Hrun. rewrite Hrun in Hrem.
    exact (URR3.step_rem_ok_mono _ _ s (URR3.rem_used_reach F1 T1) Hrem).
Qed.

Theorem rebuild_unbox_nr_ind2_with_types_refines_cc :
  rebuild_unbox_nr_ind2_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_nr_ind2_with_types_cc_gen false flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind2_with_types_local :
  rebuild_unbox_nr_ind2_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (URP.refines_observably_refines _ _ _
           (TP.rebuild_cc_with_types_refines_observably_local flag is_local u e
              T [] cannot_change b Hleast Hslots Hfrag Hany Hrk Hwf Hdist
              Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_nr_ind2_with_types_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox b nr Hslots Hfrag Hdist Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_nr_ind2_with_types_observably :
  rebuild_unbox_nr_ind2_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_local flag is_local u e T T1
           cannot_change dominated to_unbox b nr Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_refines_cc :
  rebuild_unbox_unit_never2_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hslots
    Hfrag Hdist Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_refines_cc flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hslots Hfrag Hdist Hclosed
           Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_local :
  rebuild_unbox_unit_never2_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_local flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_observably :
  rebuild_unbox_unit_never2_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_observably flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hproj Hfrag Hany
           Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

(* ======================================================================== *)
(* With the primitives of the heap                                          *)
(* ======================================================================== *)

Theorem rebuild_unbox_nr_ind2_cc_heap :
  rebuild_unbox_nr_ind2_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (N2P.rebuild_unbox_nr_ind2_refines_cc_gen true flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind2_heap_local :
  rebuild_unbox_nr_ind2_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (first_rebuild flag u e T cannot_change pdc)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (CCP.rebuild_cc_refines_up_to_heap_local flag is_local u e T
             cannot_change pdc Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs
             Hexact Hccc Hwid).
  - exact (rebuild_unbox_nr_ind2_cc_heap flag is_local u e T T1 cannot_change
             dominated to_unbox pdc nr Hslots Hfrag Hclosed Hfrag1 Hclosed1
             Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_nr_ind2_up_to_heap :
  rebuild_unbox_nr_ind2_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind2_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_cc_heap :
  rebuild_unbox_unit_never2_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_cc_heap flag is_local u e T T1 cannot_change
           dominated to_unbox pdc _ Hslots Hfrag Hclosed Hfrag1 Hclosed1
           Hlrs1 (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov)
           Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_heap_local :
  rebuild_unbox_unit_never2_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_up_to_heap :
  rebuild_unbox_unit_never2_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_up_to_heap flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hproj Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

(* With types. *)

Theorem rebuild_unbox_nr_ind2_with_types_cc_heap :
  rebuild_unbox_nr_ind2_with_types_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_nr_ind2_with_types_cc_gen true flag is_local u e T
           T1 cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind2_with_types_heap_local :
  rebuild_unbox_nr_ind2_with_types_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (first_rebuild_with_types flag is_local u e T
                     cannot_change b)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (TP.rebuild_cc_with_types_refines_up_to_heap_local flag is_local u
             e T [] cannot_change b Hleast Hslots Hfrag Hany Hrk Hwf Hdist
             Hone Hclosed Hlrs Hexact Hccc Hwid).
  - exact (rebuild_unbox_nr_ind2_with_types_cc_heap flag is_local u e T T1
             cannot_change dominated to_unbox b nr Hslots Hfrag Hdist Hclosed
             Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_nr_ind2_with_types_up_to_heap :
  rebuild_unbox_nr_ind2_with_types_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox b nr Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_cc_heap :
  rebuild_unbox_unit_never2_with_types_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hslots
    Hfrag Hdist Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_cc_heap flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hslots Hfrag Hdist Hclosed
           Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_heap_local :
  rebuild_unbox_unit_never2_with_types_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_with_types_up_to_heap :
  rebuild_unbox_unit_never2_with_types_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_with_types_up_to_heap flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hproj Hfrag Hany
           Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (N2P.unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

