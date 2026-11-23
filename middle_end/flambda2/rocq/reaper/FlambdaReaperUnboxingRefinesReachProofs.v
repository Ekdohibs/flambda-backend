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

(* The proofs of FlambdaReaperUnboxingRefinesReachSpec.v. Nothing here
   needs review.

   The simulation is that of FlambdaReaperUnboxingRefinesCallsSimProofs.v,
   whose call of known arity of [f], whose code is not known, may be made
   at a shape that only the code that the flow may bring to [f] fits
   ([ind_targets_f]); the run of the first unit then needs [flow_run]: at
   each call, the code of the closure that the callee holds is in [RCH] of
   the callee. FlambdaReaperUnboxingRefinesCallsProofs.v relates the two
   rebuilt units on the fragment [calls_unboxing_reach_nr], given
   [flow_run] ([rebuild_unbox_modes_refines_cc_reach]). Here [flow_run] is
   proved of the first rebuilt unit, with [RCH] read off its own analysis
   ([flow_reach]): the run of FlambdaReaperFlowProofs.v, kept with the
   closure deps of the code ([run_flows_called], as
   FlambdaReaperFlowProofs.flow_sound_gen_local but for [called_by]),
   puts every closure that a variable holds at an allocation that may come
   to the variable and has the witness of known arity of its code
   ([closure_reach]). *)

From Stdlib Require Import ZArith List Bool.
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
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesCallsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Theorem calls_unboxing_over1_in_flow :
  calls_unboxing_over1_in_flow_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  exact (calls_unboxing_in_flow_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach F1 T1) e H).
Qed.

(* ======================================================================== *)
(* The flow of the first rebuilt unit                                       *)
(* ======================================================================== *)

Import FlambdaReaperFlowProofs.

(* The run of FlambdaReaperFlowProofs.flow_sound_gen_local, with the
   closure deps of the code and what they give. *)
Lemma run_flows_called (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (n : nat) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  ∃ (located : location -> option node) (cds : finite_map code_dep),
    (∀ (id : code_id) (cd : code_dep),
       lookup id cds = Some cd ->
       known_witness_edges (graph_of_facts (traverse_unit flag u e)) id cd)
    ∧ match run n (State e [] [] empty_static_env empty_heap no_reads) with
      | Step s =>
        state_flows (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) (unit_kept e)
          (called_by (graph_of_facts (traverse_unit flag u e)) cds) located s
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
  exists located, cds. split; [ exact Hcds | ].
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
  destruct Hrun as [Hinv [Hheap _]].
  destruct (WithLocal.flow_invariant_flows _ _ _ _ _ _ _ _ _ _ _ _ _ Hinv)
    as (Hvars & Hconts & Hst).
  exact (conj Hvars (conj Hconts (conj Hst Hheap))).
Qed.

(* An allocation [a] that may come to [f], with the closure dep of the
   code [id], is one that [flow_reach] reads. *)
Lemma alloc_reach (F1 : facts) (T1 : tables) (cds : finite_map code_dep)
    (a : node) (f : variable) (id : code_id) :
  (∀ (id' : code_id) (cd : code_dep),
     lookup id' cds = Some cd ->
     known_witness_edges (graph_of_facts F1) id' cd) ->
  may_come_from (solution_of_tables T1) a (Node_variable f) ->
  called_by (graph_of_facts F1) cds a id ->
  flow_reach F1 T1 f id = true.
Proof.
  intros Hcds (_ & Hsrc & _) Hcalled. unfold flow_reach.
  destruct Hsrc as [Has | Hsrc].
  - cbn [solution_of_tables s_any_source] in Has. rewrite Has. reflexivity.
  - cbn [solution_of_tables s_sources] in Hsrc.
    apply mem_pair_In in Hsrc.
    apply orb_true_iff. right. apply existsb_exists.
    exists (Node_variable f, a). split; [ exact Hsrc | ].
    rewrite node_eqb_refl. cbn [andb].
    unfold called_by, closure_dep_facts in Hcalled. cbn in Hcalled.
    unfold alloc_has_code. apply existsb_exists.
    destruct (lookup id cds) as [cd | ] eqn:Hcd.
    + destruct Hcalled as (_ & Hkw & _).
      exists (a, Field_call_witness Known_arity_code_pointer,
              cd_known_arity_call_witness cd).
      split; [ exact Hkw | ].
      rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
      apply orb_true_iff. right. apply existsb_exists.
      destruct (Hcds id cd Hcd) as (_ & _ & _ & Hcode & _).
      exists (cd_known_arity_call_witness cd, Field_code_id_of_call_witness,
              Node_code_id id).
      split; [ exact Hcode | ].
      rewrite !node_eqb_refl, field_eqb_refl. reflexivity.
    + destruct Hcalled as (w & Hany & Hkw).
      exists (a, Field_call_witness Known_arity_code_pointer, w).
      split; [ exact Hkw | ].
      rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
      apply orb_true_iff. left. apply mem_node_In. exact Hany.
Qed.

(* The code of a closure that [f] holds is in [flow_reach] of [f]. *)
Lemma closure_reach (F1 : facts) (T1 : tables) (cds : finite_map code_dep)
    (keep : value_slot -> bool) (located : location -> option node)
    (vars : env) (S : static_env) (f : variable) (v : value) (id : code_id)
    (slots : list (value_slot * value)) :
  (∀ (id' : code_id) (cd : code_dep),
     lookup id' cds = Some cd ->
     known_witness_edges (graph_of_facts F1) id' cd) ->
  env_flows (graph_of_facts F1) (solution_of_tables T1) keep
    (called_by (graph_of_facts F1) cds) located vars ->
  statics_flow (graph_of_facts F1) (solution_of_tables T1) keep
    (called_by (graph_of_facts F1) cds) located S ->
  lookup f vars = Some v -> closure_of S v = Some (id, slots) ->
  flow_reach F1 T1 f id = true.
Proof.
  intros Hcds Hvars Hst Hf Hclo.
  pose proof (Hvars f v Hf) as Hfl.
  destruct v as [c | code vs0 | slot decls vs | sym | tag fields | elements
                 | contents | l | callee args aa];
    cbn [closure_of] in Hclo; try discriminate Hclo.
  - (* A closure of the older form, at no node. *)
    cbn [flows] in Hfl. contradiction.
  - (* A closure of a set. *)
    destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hclo ].
    cbn in Hclo. injection Hclo as <- _.
    cbn [flows] in Hfl. destruct Hfl as (sites & Hmc & Hall).
    destruct (Hall slot id0 Hl) as (_ & Hcalled & _).
    exact (alloc_reach F1 T1 cds _ f id0 Hcds Hmc Hcalled).
  - (* A symbol of a static set. *)
    cbn [flows] in Hfl.
    destruct (lookup sym (static_values S)) as [allocated | ] eqn:Hs;
      [ | discriminate Hclo ].
    pose proof (Hst sym allocated Hs) as (_ & Hsv).
    destruct allocated as [code vs0 | slot syms decls vs | tag fields
                           | contents | bytes | elements | c];
      try discriminate Hclo.
    + cbn in Hsv. contradiction.
    + destruct (lookup slot decls) as [id0 | ] eqn:Hl;
        [ | discriminate Hclo ].
      cbn in Hclo. injection Hclo as <- _.
      destruct Hsv as (Hcalled & _).
      exact (alloc_reach F1 T1 cds _ f id0 Hcds Hfl (Hcalled id0 eq_refl)).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_unbox_flow_refines_cc :
  rebuild_unbox_flow_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild in *.
  set (e1 := rebuild_unit_cc_from_tables (traverse_unit flag u e) T
               cannot_change (unit_code_deps flag u e) pdc e) in *.
  set (F1 := traverse_unit flag u e1) in *.
  apply (rebuild_unbox_modes_refines_cc_reach flag is_local u e T
           cannot_change dominated to_unbox pdc nr (flow_reach F1 T1) Hslots
           Hfrag Hclosed Hcalls Hcheck).
  intros n s Hrun ap f v id slots _ _ Hf.
  split; [ | intros _; reflexivity ].
  intros Hclo. cbn [reach2].
  destruct (run_flows_called flag is_local u e1 T1 n Hslots Hfrag1
              (check_closed_sound is_local _ _ Hclosed1)
              (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                 is_local _ _ Hlrs1))
    as (located & cds & Hcds & Hflows).
  fold F1 in Hcds, Hflows. unfold e1 in Hflows. rewrite Hrun in Hflows.
  destruct Hflows as (Hvars & _ & Hst & _).
  exact (closure_reach F1 T1 cds (unit_kept e1) located (state_vars s)
           (state_statics s) f v id slots Hcds Hvars Hst Hf Hclo).
Qed.

Theorem rebuild_unbox_flow_refines_observably_local :
  rebuild_unbox_flow_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_flow_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_flow_refines_observably :
  rebuild_unbox_flow_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_flow_refines_observably_local flag is_local u e T T1
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

Theorem flow_example : flow_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_flow FlowExample.facts FlowExample.tables
                     FlowExample.cannot_change FlowExample.code_deps [] []
                     FlowExample.facts1 FlowExample.tables1
                     FlowExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment FlowExample.unit_expr = true
    ∧ any_kind_arguments FlowExample.unit_expr = true
    ∧ results_kinded [] [] FlowExample.unit_expr = true
    ∧ well_formed_unit UE.interface FlowExample.unit_expr = true
    ∧ exits_one_value UE.interface FlowExample.unit_expr = true
    ∧ check_closed UE.no_local FlowExample.facts FlowExample.tables = true
    ∧ check_field_used_exact UE.no_local FlowExample.facts FlowExample.tables
      = true
    ∧ check_cc_closed FlowExample.facts FlowExample.tables
        FlowExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        FlowExample.facts FlowExample.tables
      = true
    ∧ in_heap_fragment FlowExample.first = true
    ∧ check_closed UE.no_local FlowExample.facts1 FlowExample.tables1 = true
    ∧ check_local_reads_used UE.no_local FlowExample.facts1
        FlowExample.tables1
      = true
    ∧ check_unboxed_reads FlowExample.facts FlowExample.tables [] [] = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  apply (rebuild_unbox_flow_refines_observably Auto UE.no_local
           UE.interface FlowExample.unit_expr FlowExample.tables
           FlowExample.tables1 FlowExample.cannot_change [] [] true
           (fun _ => false)); try assumption.
  left. intros f. reflexivity.
Qed.
