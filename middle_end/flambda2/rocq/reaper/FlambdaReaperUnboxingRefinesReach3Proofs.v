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
(* The proofs of FlambdaReaperUnboxingRefinesReach3Spec.v. Nothing here
   needs review.

   FlambdaReaperUnboxingRefinesCallsProofs.v relates the two rebuilt units
   on the fragment [calls_unboxing_reach_rem_nr], given [flow_run],
   [over_run] and [rem_run] of the first
   ([rebuild_unbox_modes_refines_cc_over3]). Here they are proved of the
   first rebuilt unit with [flow_reach2], [over_reach] and [rem_reach] of
   its own analysis: [flow_run] as in
   FlambdaReaperUnboxingRefinesReach2Proofs.v, [over_run] as in
   FlambdaReaperUnboxingRefinesOver2Proofs.v, and [rem_run] by
   FlambdaReaperFlowProofs.step_rem_used, which gives, of each state of the
   run, that what is left of an over-application calls only code that is
   [rem_used] (the flow brings it to the first result, with a use, of a
   witness of code: where what is left of an over-application starts),
   which [rem_reach] reads off the facts and the tables
   ([rem_used_reach]). *)

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
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesOver2Proofs
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperPartialGroupsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.

Theorem calls_unboxing_flow2_in_flow3 :
  calls_unboxing_flow2_in_flow3_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  unfold calls_unboxing_flow3_nr.
  apply (calls_unboxing_in_over3_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach2 F1 T1) (flow_reach2 F1 T1));
    [ exact (fun _ _ _ H => H) | ].
  exact (calls_unboxing_in_over2_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach2 F1 T1) (over_reach F1 T1) e H).
Qed.

Theorem calls_unboxing_over2_in_flow3 :
  calls_unboxing_over2_in_flow3_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  unfold calls_unboxing_flow3_nr.
  apply (calls_unboxing_in_over3_reach nr F T cannot_change code_deps
           dominated to_unbox (fun _ _ _ => true) (flow_reach2 F1 T1));
    [ reflexivity | exact H ].
Qed.

(* ======================================================================== *)
(* The code that what is left of an over-application calls                 *)
(* ======================================================================== *)

(* [rem_reach] is [rem_used], read off the facts and the tables. *)
Lemma rem_used_reach (F1 : facts) (T1 : tables) (id : code_id) :
  rem_used (graph_of_facts F1) (solution_of_tables T1) id ->
  rem_reach F1 T1 id = true.
Proof.
  intros (W & R & (id' & Hcode) & Hret & Hused & Hsrc).
  unfold rem_reach. apply existsb_exists.
  exists (W, Field_code_id_of_call_witness, Node_code_id id').
  split; [ exact Hcode | ].
  rewrite field_eqb_refl. cbn [andb].
  apply existsb_exists. exists (W, Field_return_of_call (Normal 0), R).
  split; [ exact Hret | ].
  rewrite node_eqb_refl, field_eqb_refl.
  cbn [solution_of_tables s_has_usage] in Hused. rewrite Hused. cbn [andb].
  unfold node_reach.
  destruct Hsrc as [Has | (a & Hsa & Hcall)].
  { cbn [solution_of_tables s_any_source] in Has. rewrite Has.
    reflexivity. }
  cbn [solution_of_tables s_sources] in Hsa.
  apply mem_pair_In in Hsa.
  apply orb_true_iff. right. apply existsb_exists.
  exists (R, a). split; [ exact Hsa | ].
  rewrite node_eqb_refl. cbn [andb].
  destruct Hcall as [(w & Hkw & Hw) | (w & Huw & Hw)].
  - apply orb_true_iff. left.
    unfold alloc_has_code. apply existsb_exists.
    exists (a, Field_call_witness Known_arity_code_pointer, w).
    split; [ exact Hkw | ].
    rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
    destruct Hw as [Hany | Hw].
    + apply orb_true_iff. left. apply mem_node_In. exact Hany.
    + apply orb_true_iff. right. apply existsb_exists.
      exists (w, Field_code_id_of_call_witness, Node_code_id id).
      split; [ exact Hw | ].
      rewrite !node_eqb_refl, field_eqb_refl. reflexivity.
  - apply orb_true_iff. right.
    unfold alloc_unknown_code. apply existsb_exists.
    exists (a, Field_call_witness Unknown_arity_code_pointer, w).
    split; [ exact Huw | ].
    rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
    apply existsb_exists.
    exists (w, Field_code_id_of_call_witness, Node_code_id id).
    split; [ exact Hw | ].
    rewrite !node_eqb_refl, field_eqb_refl. reflexivity.
Qed.

(* What a step calls by what is left of an over-application, for a wider
   [REM]. *)
Lemma cont_rem_ok_mono (REM REM' : code_id -> Prop) (S : static_env) :
  (∀ id : code_id, REM id -> REM' id) ->
  ∀ (c : cont_value) (vs : list value),
    cont_rem_ok REM S c vs -> cont_rem_ok REM' S c vs.
Proof.
  intros Himp c.
  induction c as [h vars conts recursive | k | c IH extra
                 | remaining remaining_arity return_arity return_to exn_to
                     IHret _]
    using cont_value_over_ind;
    intros vs; cbn [cont_rem_ok]; [ exact (fun H => H) | exact (fun H => H)
                                  | apply IH | ].
  destruct vs as [ | v [ | v' vs]]; [ exact (fun H => H) | |
                                      exact (fun H => H) ].
  intros [Hcl Hsh]. split.
  - intros id slots Hclo. exact (Himp id (Hcl id slots Hclo)).
  - exact (shape_over_ok_mono _ _ S _ _ _ (fun _ H => H) IHret Hsh).
Qed.

Lemma step_rem_ok_mono (REM REM' : code_id -> Prop) (s : state) :
  (∀ id : code_id, REM id -> REM' id) ->
  step_rem_ok REM s -> step_rem_ok REM' s.
Proof.
  intros Himp. unfold step_rem_ok.
  destruct (state_expr s); try exact (fun H => H).
  - intros H callee vs Hcallee Hvs.
    exact (shape_over_ok_mono _ _ _ _ _ _ (fun _ H => H)
             (cont_rem_ok_mono REM REM' _ Himp _)
             (H callee vs Hcallee Hvs)).
  - intros H vs Hvs. exact (cont_rem_ok_mono REM REM' _ Himp _ _ (H vs Hvs)).
Qed.
(* Every state of the run of a unit of the fragment, with a solution closed
   for its facts, calls by what is left of an over-application only code
   that is [rem_used]. *)
Lemma run_rem_used (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (n : nat) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  match run n (State e [] [] empty_static_env empty_heap no_reads) with
  | Step s =>
    step_rem_ok
      (rem_used (graph_of_facts (traverse_unit flag u e))
         (solution_of_tables T))
      s
  | _ => True
  end.
Proof.
  intros Hlocal Hfrag Hc Hlrs.
  destruct (traverse_unit_facts_heap flag u e True Hfrag (or_introl I))
    as (noted & deps & cds & He0 & Hconsts & Himported & _ & Hdeps & Hnd
        & Hkeys).
  set (G := graph_of_facts (traverse_unit flag u e)) in *.
  set (sol := solution_of_tables T) in *.
  pose proof (static_names_edges G noted deps cds (unit_kept e)
                (unit_codes e) (static_names symbols_of_binding e)
                True True) as [Hnames _].
  destruct (Hnames e None (unit_conts u) He0) as [Hsym_names Hcode_names].
  assert (Hcds : ∀ (id : code_id) (cd : code_dep),
            lookup id cds = Some cd -> known_witness_edges G id cd).
  { intros id cd Hl.
    assert (Hid : id ∈ static_names code_ids_of_binding e).
    { apply Hkeys. apply (in_map fst (cds) (id, cd)).
      exact (lookup_in _ _ _ Hl). }
    destruct (Hcode_names id Hid) as (c & _ & Hcode).
    destruct c as [params mc ret exn metadata body].
    destruct Hcode as (cd' & Hin' & Hw & _).
    rewrite (lookup_nodup_in _ _ _ Hnd Hin') in Hl. injection Hl as <-.
    exact Hw. }
  assert (Hunit_syms : ∀ s : symbol,
            among s (static_names symbols_of_binding e) = true ->
            may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs. apply among_In in Hs.
    destruct (Hsym_names s Hs) as (const & Hb).
    destruct const as [ | slot syms decls vs | t fields | | | ];
      simpl in Hb; try contradiction.
    - destruct Hb as (Hslot & Hdep & _). unfold is_bound in Hslot.
      destruct (lookup slot decls) as [id | ] eqn:Hl; [ | discriminate Hslot ].
      exact (called_alloc is_local G sol cds Hc Hcds _ _
               (Hdeps _ (Hdep id eq_refl))).
    - destruct Hb as (_ & His & _).
      exact (alloc_self is_local G sol Hc _ _ _ His
               (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
    - exact (boxed_self is_local G sol Hc Hconsts _ _ Hb).
    - exact (any_source_self is_local G sol Hc _
               (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
    - destruct elements as [ | element elements ].
      + exact (any_source_self is_local G sol Hc _
                 (alias_consts_any_source is_local G sol Hc Hconsts _ Hb)).
      + destruct Hb as (_ & His & _).
        exact (alloc_self is_local G sol Hc _ _ _ His
                 (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hconsts))).
  }
  assert (Hsyms : ∀ s : symbol,
            noted s -> may_come_from sol (Node_symbol s) (Node_symbol s)).
  { intros s Hs.
    destruct (among s (static_names symbols_of_binding e)) eqn:Hunit;
      [ exact (Hunit_syms s Hunit) | ].
    apply (may_come_from_self is_local G sol _ Hc).
    exact (any_source_input _ _ _ Hc _ (Himported s Hs Hunit)). }
  destruct (located_exists n (State e [] [] empty_static_env empty_heap
                                no_reads))
    as (located & Hloc).
  pose proof (WithLocal.run_flow_invariant is_local G sol noted deps cds
                (unit_kept e) (unit_codes e) (static_names symbols_of_binding e)
                u True located True Hc Hlocal Hconsts Hsyms Hunit_syms Hdeps
                Hnd Hcds (fun _ => Hlrs) n _ Hloc
                (WithLocal.initial_flow_invariant G sol noted deps cds
                   (unit_kept e) (unit_codes e)
                   (static_names symbols_of_binding e) u True located True e
                   He0)
                ltac:(apply initial_heap_ok))
    as Hrun.
  destruct (run n (State e [] [] empty_static_env empty_heap no_reads))
    as [s | k vs hp rd | ]; [ | exact I | exact I ].
  destruct Hrun as [Hinv _].
  exact (WithLocal.step_rem_used is_local G sol noted deps cds (unit_kept e)
           (unit_codes e) (static_names symbols_of_binding e) u True located
           True Hc Hlocal Hconsts Hsyms Hnd Hcds s Hinv).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_unbox_flow3_refines_cc :
  rebuild_unbox_flow3_refines_cc_statement.
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
  apply (rebuild_unbox_modes_refines_cc_over3 flag is_local u e T
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

Theorem rebuild_unbox_flow3_refines_observably_local :
  rebuild_unbox_flow3_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_flow3_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_flow3_refines_observably :
  rebuild_unbox_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_flow3_refines_observably_local flag is_local u e T T1
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

Theorem flow3_example : flow3_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_flow3 Flow3Example.facts
                     Flow3Example.tables Flow3Example.cannot_change
                     Flow3Example.code_deps [] [] Flow3Example.facts1
                     Flow3Example.tables1 Flow3Example.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment Flow3Example.unit_expr = true
    ∧ any_kind_arguments Flow3Example.unit_expr = true
    ∧ results_kinded [] [] Flow3Example.unit_expr = true
    ∧ well_formed_unit UE.interface Flow3Example.unit_expr = true
    ∧ exits_one_value UE.interface Flow3Example.unit_expr = true
    ∧ check_closed UE.no_local Flow3Example.facts Flow3Example.tables = true
    ∧ check_field_used_exact UE.no_local Flow3Example.facts
        Flow3Example.tables
      = true
    ∧ check_cc_closed Flow3Example.facts Flow3Example.tables
        Flow3Example.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        Flow3Example.facts Flow3Example.tables
      = true
    ∧ in_heap_fragment Flow3Example.first = true
    ∧ check_closed UE.no_local Flow3Example.facts1 Flow3Example.tables1
      = true
    ∧ check_local_reads_used UE.no_local Flow3Example.facts1
        Flow3Example.tables1
      = true
    ∧ check_unboxed_reads Flow3Example.facts Flow3Example.tables [] []
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
  (* The check as the statement has it, which the elaborator is slow to
     convert [Hcalls] to. *)
  assert (Hcalls' :
    calls_unboxing_flow3_nr (fun _ => false)
      (traverse_unit Auto UE.interface Flow3Example.unit_expr)
      Flow3Example.tables Flow3Example.cannot_change
      (unit_code_deps Auto UE.interface Flow3Example.unit_expr) [] []
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface Flow3Example.unit_expr
            Flow3Example.tables Flow3Example.cannot_change true))
      Flow3Example.tables1 Flow3Example.unit_expr
    = true)
    by (vm_compute; reflexivity).
  exact (rebuild_unbox_flow3_refines_observably Auto UE.no_local
           UE.interface Flow3Example.unit_expr Flow3Example.tables
           Flow3Example.tables1 Flow3Example.cannot_change [] [] true
           (fun _ => false) Hslots (or_introl (fun _ => eq_refl)) Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls'
           Hreads).
Qed.
