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
(* The proofs of FlambdaReaperUnboxingRefinesNeverSpec.v. Nothing here
   needs review.

   FlambdaReaperUnboxingRefinesCallsProofs.v relates the two rebuilt units
   on the fragment [calls_unboxing_reach_never_nr], given [flow_run],
   [over_run] and [rem_run] of the first
   ([rebuild_unbox_modes_refines_cc_never]). A call whose code is not known
   that never returns is related to the call that the second rebuild makes
   return to a handler [Invalid]
   (FlambdaReaperUnboxingRefinesCallsSimProofs.uc_call_ind_never, built by
   [call_rebuilt_never]): the frames of the two runs are not related, and
   the code that the call enters is code of [nr], read off [flow_run]
   ([never_call_answered]), whose body is related with its return
   continuation out of the relation. The three runs are proved of the first
   rebuilt unit as in FlambdaReaperUnboxingRefinesReach3Proofs.v, whose
   proof this repeats. The fragment of
   FlambdaReaperUnboxingRefinesReach3Spec.v is in this one
   ([calls_unboxing_in_never_reach]). The proof holds of the units with
   the primitives of the heap ([rebuild_unbox_nr_ind_refines_cc_gen], if
   [hb]): the three runs are read off the first rebuilt unit, which the
   statement asks of [in_heap_fragment] already; the statement is its
   instance without them. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperLeast
  FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesOver2Proofs
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperPartialGroupsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.

Theorem calls_unboxing_flow3_in_never :
  calls_unboxing_flow3_in_never_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  exact (calls_unboxing_in_never_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach2 F1 T1) (over_reach F1 T1)
           (rem_reach F1 T1) e H).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The statement, of the units of the fragment with the primitives of the
   heap if [hb]. *)
Lemma rebuild_unbox_nr_ind_refines_cc_gen (hb : bool) :
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    flow_fragment hb (unit_codes e) (static_names symbols_of_binding e) e
    && distinct (static_names code_ids_of_binding e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild in *.
  set (e1 := rebuild_unit_cc_from_tables (traverse_unit flag u e) T
               cannot_change (unit_code_deps flag u e) pdc e) in *.
  set (F1 := traverse_unit flag u e1) in *.
  pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
  pose proof (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                is_local _ _ Hlrs1) as Hlr1.
  apply (rebuild_unbox_modes_refines_cc_never_gen hb flag is_local u e T
           cannot_change dominated to_unbox pdc nr (flow_reach2 F1 T1)
           (over_reach F1 T1) (rem_reach F1 T1) Hslots Hfrag Hclosed Hcalls
           Hcheck).
  - (* The flow, as in FlambdaReaperUnboxingRefinesReach2Proofs.v. *)
    intros n s Hrun ap f v id slots _ _ Hf.
    destruct (run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                Hlr1)
      as (located & cds & Hcds & Hnd & Hflows).
    fold F1 in Hcds, Hflows. unfold e1 in Hflows. rewrite Hrun in Hflows.
    destruct Hflows as ((Hvars & _ & Hst & _) & Hchains & Hpa).
    assert (Hclosure :
      ∀ (v0 : value) (id0 : code_id) (slots0 : list (value_slot * value)),
        lookup f (state_vars s) = Some v0 ->
        closure_of (state_statics s) v0 = Some (id0, slots0) ->
        flow_reach F1 T1 f id0 = true)
      by (intros v0 id0 slots0 Hf0 Hclo0;
          exact (closure_reach F1 T1 cds (unit_kept e1) located
                   (state_vars s) (state_statics s) f v0 id0 slots0 Hcds
                   Hvars Hst Hf0 Hclo0)).
    split; intros Hclo; cbn [flow_reach2].
    + exact (Hclosure v id slots Hf Hclo).
    + destruct v as [c | code vs0 | slot decls vs | sym | tag fields
                     | elements | contents | l | cl applied aa];
        try (unfold flow_reach_u; rewrite (Hclosure _ id slots Hf Hclo);
             reflexivity).
      exact (partial_reach is_local F1 T1 cds (unit_kept e1) located
               (state_statics s) f cl applied aa id slots Hc1 Hcds Hnd Hst
               Hchains (Hpa f _ Hf) (Hvars f _ Hf) Hclo).
  - (* What a step over-applies. *)
    intros n s Hrun.
    pose proof (run_over_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hover.
    fold F1 in Hover. unfold e1 in Hover. rewrite Hrun in Hover.
    exact (step_over_ok_mono _ _ s (over_used_reach F1 T1) Hover).
  - (* What is left of an over-application calls. *)
    intros n s Hrun.
    pose proof (run_rem_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hrem.
    fold F1 in Hrem. unfold e1 in Hrem. rewrite Hrun in Hrem.
    exact (step_rem_ok_mono _ _ s (rem_used_reach F1 T1) Hrem).
Qed.

Theorem rebuild_unbox_nr_ind_refines_cc :
  rebuild_unbox_nr_ind_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_nr_ind_refines_cc_gen false flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind_refines_observably_local :
  rebuild_unbox_nr_ind_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_nr_ind_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_nr_ind_refines_observably :
  rebuild_unbox_nr_ind_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind_refines_observably_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Theorem never_ind_example : never_ind_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_never_nr NeverIndExample.nr
                     NeverIndExample.facts NeverIndExample.tables
                     NeverIndExample.cannot_change NeverIndExample.code_deps
                     [] [] NeverIndExample.facts1 NeverIndExample.tables1
                     NeverIndExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment NeverIndExample.unit_expr = true
    ∧ any_kind_arguments NeverIndExample.unit_expr = true
    ∧ results_kinded [] [] NeverIndExample.unit_expr = true
    ∧ well_formed_unit UE.interface NeverIndExample.unit_expr = true
    ∧ exits_one_value UE.interface NeverIndExample.unit_expr = true
    ∧ check_closed UE.no_local NeverIndExample.facts NeverIndExample.tables
      = true
    ∧ check_field_used_exact UE.no_local NeverIndExample.facts
        NeverIndExample.tables
      = true
    ∧ check_cc_closed NeverIndExample.facts NeverIndExample.tables
        NeverIndExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        NeverIndExample.facts NeverIndExample.tables
      = true
    ∧ in_heap_fragment NeverIndExample.first = true
    ∧ check_closed UE.no_local NeverIndExample.facts1 NeverIndExample.tables1
      = true
    ∧ check_local_reads_used UE.no_local NeverIndExample.facts1
        NeverIndExample.tables1
      = true
    ∧ check_unboxed_reads NeverIndExample.facts NeverIndExample.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  (* The check as the statement has it, which the elaborator is slow to
     convert [Hcalls] to. *)
  assert (Hcalls' :
    calls_unboxing_never_nr NeverIndExample.nr
      (traverse_unit Auto UE.interface NeverIndExample.unit_expr)
      NeverIndExample.tables NeverIndExample.cannot_change
      (unit_code_deps Auto UE.interface NeverIndExample.unit_expr) [] []
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface NeverIndExample.unit_expr
            NeverIndExample.tables NeverIndExample.cannot_change true))
      NeverIndExample.tables1 NeverIndExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  exact (rebuild_unbox_nr_ind_refines_observably Auto UE.no_local
           UE.interface NeverIndExample.unit_expr NeverIndExample.tables
           NeverIndExample.tables1 NeverIndExample.cannot_change [] [] true
           NeverIndExample.nr Hslots (or_introl (fun _ => eq_refl)) Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls'
           Hreads).
Qed.
