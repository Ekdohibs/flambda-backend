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

(* The proofs of FlambdaReaperUnboxingRefinesReach2Spec.v. Nothing here
   needs review.

   FlambdaReaperUnboxingRefinesCallsProofs.v relates the two rebuilt units
   on the fragment [calls_unboxing_reach2_nr], given [flow_run] of the
   first ([rebuild_unbox_modes_refines_cc_reach2]): at a call of [f], the
   code of a closure that [f] holds is in [reach true f], and that of the
   closure that a call of what [f] holds calls in [reach false f]. Here
   it is proved of the first rebuilt unit, with [flow_reach2] of its own
   analysis. A closure is as in FlambdaReaperUnboxingRefinesReachProofs.v
   ([closure_reach]). A partial application of a closure of the code [A],
   [k] groups given, was made by [k] calls, whose witnesses stand for the
   witnesses of unknown arity of what their callees may come from
   ([call_chain]); the code dep of [A] has a chain of such witnesses, the
   first result of each but the last an allocation whose call witness is
   the next ([code_chain_edges]), and the [k]-th call returns such an
   allocation, which may come to [f] ([chain_reach], [partial_reach]). *)
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
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesCallsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.

Theorem calls_unboxing_flow_in_flow2 :
  calls_unboxing_flow_in_flow2_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  exact (calls_unboxing_in_flow2_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach F1 T1) (flow_reach2 F1 T1) e
           (fun _ _ => eq_refl) H).
Qed.

(* ======================================================================== *)
(* The flow of the first rebuilt unit                                       *)
(* ======================================================================== *)

(* The witnesses of unknown arity of the code of the static environment,
   as its code dep has them. *)
Definition code_chains (G : graph) (cds : finite_map code_dep)
    (S : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id (static_code S) = Some c -> code_is_tupled c = false ->
    ∃ cd : code_dep,
      (id, cd) ∈ cds
      ∧ length (cd_unknown_arity_call_witnesses cd)
        = length
            (at_least_one_group
               (group_by_parameter (code_params_arity c) (cd_params cd)))
      ∧ code_chain_edges G id (cd_return cd) (cd_exn cd)
          (combine
             (at_least_one_group
                (group_by_parameter (code_params_arity c) (cd_params cd)))
             (cd_unknown_arity_call_witnesses cd)).

(* [run_flows_called], with the witnesses of the code and the partial
   applications of the state. *)
Lemma run_flows_called2 (flag : reaper_preserve_direct_calls)
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
    ∧ NoDup (map fst cds)
    ∧ match run n (State e [] [] empty_static_env empty_heap no_reads) with
      | Step s =>
        state_flows (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) (unit_kept e)
          (called_by (graph_of_facts (traverse_unit flag u e)) cds) located s
        ∧ code_chains (graph_of_facts (traverse_unit flag u e)) cds
            (state_statics s)
        ∧ env_pa_ok (state_statics s) (state_vars s)
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
  exists located, cds. split; [ exact Hcds | split; [ exact Hnd | ] ].
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
  split; [ exact (conj Hvars (conj Hconts (conj Hst Hheap))) | ].
  destruct Hinv as (ctx & ce & _ & _ & _ & _ & _ & _ & Hcodes & Hpa).
  split; [ | exact (proj1 Hpa) ].
  intros id c Hl Htup. specialize (Hcodes id c Hl).
  destruct c as [params mc ret exn md body].
  destruct Hcodes as (_ & cd & Hin & _ & Hch & _).
  exists cd. split; [ exact Hin | ]. exact (Hch Htup).
Qed.

(* The witnesses of a chain of calls stand for those of the code dep of
   [id], from the [j]-th on: what the last call returns has the call
   witness that comes [length ws] after. *)
Lemma chain_reach (is_local : field -> bool) (G : graph) (sol : solution)
    (id : code_id) (returns : list node) (exn : node)
    (L : list (list node * node)) :
  closed is_local G sol ->
  code_chain_edges G id returns exn L ->
  s_has_source sol (Node_code_id id) = true ->
  ∀ (ws : list node) (P : node -> Prop) (j : nat) (y W : node)
    (first : list node),
    call_chain G sol P ws -> P y ->
    g_constructor G y (Field_call_witness Unknown_arity_code_pointer) W ->
    nth_error L j = Some (first, W) -> j + length ws < length L ->
    ∃ (y' W' : node) (first' : list node),
      chain_end G sol P ws y'
      ∧ g_constructor G y' (Field_call_witness Unknown_arity_code_pointer) W'
      ∧ nth_error L (j + length ws) = Some (first', W').
Proof.
  intros Hc Hch Hhs ws.
  induction ws as [ | w ws IH];
    intros P j y W first Hchain Hy Hcons Hj Hlt.
  - exists y, W, first. cbn [chain_end length]. rewrite Nat.add_0_r.
    auto.
  - destruct Hchain as [Hcalled Hchain].
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hch Hj)
      as (_ & Hcode & _ & Hnext).
    pose proof (alloc_self is_local G sol Hc _ _ _ Hcode Hhs) as HWW.
    pose proof (Hcalled y W Hy Hcons HWW) as HWw.
    cbn [length] in Hlt.
    assert (Hj' : ∃ (first' : list node) (W' : node),
              nth_error L (S j) = Some (first', W')).
    { destruct (nth_error L (S j)) as [[f' W'] | ] eqn:E; [ eauto | ].
      apply nth_error_None in E. lia. }
    destruct Hj' as (first' & W' & HW').
    destruct (Hnext first' W' HW') as (v & Hv & Hvw).
    destruct (code_chain_nth G _ _ _ _ _ _ _ Hch HW') as (_ & Hcode' & _).
    pose proof (alloc_self is_local G sol Hc _ _ _ Hcode' Hhs) as HW'W'.
    pose proof (alloc_self is_local G sol Hc _ _ _ Hvw (proj1 HW'W')) as Hvv.
    assert (Hret : returned_at G sol w v) by (exists W, v; auto).
    destruct (IH (returned_at G sol w) (S j) v W' first' Hchain Hret Hvw HW'
                ltac:(lia))
      as (y' & W'' & first'' & Hend & Hcons' & Hnth).
    exists y', W'', first''. cbn [chain_end].
    split; [ exact Hend | split; [ exact Hcons' | ] ].
    cbn [length]. replace (j + S (length ws)) with (S j + length ws) by lia.
    exact Hnth.
Qed.

(* A closure at [m] may come from an allocation with the closure dep of
   its code. *)
Lemma closure_alloc (G : graph) (sol : solution) (keep : value_slot -> bool)
    (called : node -> code_id -> Prop) (located : location -> option node)
    (S : static_env) (v : value) (m : node) (id : code_id)
    (slots : list (value_slot * value)) :
  statics_flow G sol keep called located S ->
  flows G sol keep called located v m ->
  closure_of S v = Some (id, slots) ->
  ∃ a : node, may_come_from sol a m ∧ called a id.
Proof.
  intros Hst Hfl Hclo.
  destruct v as [c | code vs0 | slot decls vs | sym | tag fields | elements
                 | contents | l | callee args aa];
    cbn [closure_of] in Hclo; try discriminate Hclo.
  - cbn [flows] in Hfl. contradiction.
  - destruct (lookup slot decls) as [id0 | ] eqn:Hl; [ | discriminate Hclo ].
    cbn in Hclo. injection Hclo as <- _.
    cbn [flows] in Hfl. destruct Hfl as (sites & Hmc & Hall).
    destruct (Hall slot id0 Hl) as (_ & Hcalled & _).
    exists (sites slot). exact (conj Hmc Hcalled).
  - cbn [flows] in Hfl.
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
      exists (Node_symbol sym). exact (conj Hfl (Hcalled id0 eq_refl)).
Qed.

(* The code of the closure of a partial application that [f] holds is in
   [flow_reach_u] of [f]. *)
Lemma partial_reach (is_local : field -> bool) (F1 : facts) (T1 : tables)
    (cds : finite_map code_dep) (keep : value_slot -> bool)
    (located : location -> option node) (S : static_env) (f : variable)
    (cl : value) (applied : list value) (aa : arity) (id : code_id)
    (slots : list (value_slot * value)) :
  closed is_local (graph_of_facts F1) (solution_of_tables T1) ->
  (∀ (id' : code_id) (cd : code_dep),
     lookup id' cds = Some cd ->
     known_witness_edges (graph_of_facts F1) id' cd) ->
  NoDup (map fst cds) ->
  statics_flow (graph_of_facts F1) (solution_of_tables T1) keep
    (called_by (graph_of_facts F1) cds) located S ->
  code_chains (graph_of_facts F1) cds S ->
  pa_ok S (Value_partial_application cl applied aa) ->
  flows (graph_of_facts F1) (solution_of_tables T1) keep
    (called_by (graph_of_facts F1) cds) located
    (Value_partial_application cl applied aa) (Node_variable f) ->
  closure_of S cl = Some (id, slots) ->
  flow_reach_u F1 T1 f id = true.
Proof.
  intros Hc Hcds Hnd Hst Hchains Hpa Hfl Hclo.
  set (G := graph_of_facts F1) in *.
  set (sol := solution_of_tables T1) in *.
  destruct Hpa as (Hun & _ & _).
  destruct Hun as (id' & slots' & c & Hclo' & Hc' & Hlt & Htup).
  rewrite Hclo in Hclo'. injection Hclo' as <- <-.
  destruct (Hchains id c Hc' Htup) as (cd & Hin & Hlenw & Hch).
  pose proof (lookup_nodup_in _ _ _ Hnd Hin) as Hcd.
  cbn [flows] in Hfl.
  destruct Hfl as (m & ws & Hflcl & Hlen & Hchain & _ & Hend & _).
  destruct (closure_alloc G sol keep _ located S cl m id slots Hst Hflcl Hclo)
    as (a & Ham & Hcalled).
  unfold called_by, closure_dep_facts in Hcalled. cbn in Hcalled.
  rewrite Hcd in Hcalled. destruct Hcalled as (_ & _ & Hunk).
  pose proof (Hcds id cd Hcd) as (_ & _ & _ & _ & Hany).
  assert (Hhs : s_has_source sol (Node_code_id id) = true)
    by exact (as_hs _ _ _ Hc _ (any_source_input _ _ _ Hc _ Hany)).
  assert (Hne : code_params_arity c <> [])
    by (intros E; rewrite E in Hlt; cbn [length] in Hlt; lia).
  remember (at_least_one_group
              (group_by_parameter (code_params_arity c) (cd_params cd)))
    as groups eqn:Hgroups.
  assert (Hglen : length groups = length (code_params_arity c)).
  { rewrite Hgroups, (group_by_parameter_groups _ _ Hne).
    rewrite length_group_by_sizes, length_arity_sizes. reflexivity. }
  destruct (cd_unknown_arity_call_witnesses cd) as [ | w0 wrest] eqn:Hws.
  { exfalso. cbn [length] in Hlenw. rewrite Hglen in Hlenw.
    destruct (code_params_arity c); [ exact (Hne eq_refl) | ].
    discriminate Hlenw. }
  destruct groups as [ | g0 grest].
  { exfalso. cbn [length] in Hglen. destruct (code_params_arity c);
      [ exact (Hne eq_refl) | discriminate Hglen ]. }
  destruct (chain_reach is_local G sol id (cd_return cd) (cd_exn cd)
              (combine (g0 :: grest) (w0 :: wrest)) Hc Hch Hhs ws
              (fun y => may_come_from sol y m) 0 a w0 g0 Hchain Ham
              (Hunk w0 eq_refl) eq_refl)
    as (y' & W' & first' & Hy' & Hcons' & Hnth).
  { rewrite length_combine, Hlenw, Nat.min_id. cbn [Nat.add].
    rewrite Hlen. rewrite <- Hglen in Hlt. exact Hlt. }
  destruct (code_chain_nth G _ _ _ _ _ _ _ Hch Hnth) as (_ & Hcode' & _).
  destruct (Hend y' Hy') as (_ & Hsrc & _).
  unfold flow_reach_u, flow_reach.
  destruct Hsrc as [Has | Hsrc].
  { cbn [sol solution_of_tables s_any_source] in Has. rewrite Has.
    reflexivity. }
  cbn [sol solution_of_tables s_sources] in Hsrc.
  apply mem_pair_In in Hsrc.
  apply orb_true_iff. right. apply existsb_exists.
  exists (Node_variable f, y'). split; [ exact Hsrc | ].
  rewrite node_eqb_refl. cbn [andb].
  unfold alloc_unknown_code. apply existsb_exists.
  exists (y', Field_call_witness Unknown_arity_code_pointer, W').
  split; [ exact Hcons' | ].
  rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
  apply existsb_exists.
  exists (W', Field_code_id_of_call_witness, Node_code_id id).
  split; [ exact Hcode' | ].
  rewrite !node_eqb_refl, field_eqb_refl. reflexivity.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_unbox_flow2_refines_cc :
  rebuild_unbox_flow2_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild in *.
  set (e1 := rebuild_unit_cc_from_tables (traverse_unit flag u e) T
               cannot_change (unit_code_deps flag u e) pdc e) in *.
  set (F1 := traverse_unit flag u e1) in *.
  apply (rebuild_unbox_modes_refines_cc_reach2 flag is_local u e T
           cannot_change dominated to_unbox pdc nr (flow_reach2 F1 T1) Hslots
           Hfrag Hclosed Hcalls Hcheck).
  intros n s Hrun ap f v id slots _ _ Hf.
  pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
  destruct (run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1 Hc1
              (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                 is_local _ _ Hlrs1))
    as (located & cds & Hcds & Hnd & Hflows).
  fold F1 in Hc1, Hcds, Hflows. unfold e1 in Hflows. rewrite Hrun in Hflows.
  destruct Hflows as ((Hvars & _ & Hst & _) & Hchains & Hpa).
  assert (Hclosure :
    ∀ (v0 : value) (id0 : code_id) (slots0 : list (value_slot * value)),
      lookup f (state_vars s) = Some v0 ->
      closure_of (state_statics s) v0 = Some (id0, slots0) ->
      flow_reach F1 T1 f id0 = true)
    by (intros v0 id0 slots0 Hf0 Hclo0;
        exact (closure_reach F1 T1 cds (unit_kept e1) located (state_vars s)
                 (state_statics s) f v0 id0 slots0 Hcds Hvars Hst Hf0 Hclo0)).
  split; intros Hclo; cbn [flow_reach2].
  - exact (Hclosure v id slots Hf Hclo).
  - destruct v as [c | code vs0 | slot decls vs | sym | tag fields | elements
                   | contents | l | cl applied aa];
      try (unfold flow_reach_u; rewrite (Hclosure _ id slots Hf Hclo);
           reflexivity).
    exact (partial_reach is_local F1 T1 cds (unit_kept e1) located
             (state_statics s) f cl applied aa id slots Hc1 Hcds Hnd Hst
             Hchains (Hpa f _ Hf) (Hvars f _ Hf) Hclo).
Qed.

Theorem rebuild_unbox_flow2_refines_observably_local :
  rebuild_unbox_flow2_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_flow2_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_flow2_refines_observably :
  rebuild_unbox_flow2_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_flow2_refines_observably_local flag is_local u e T T1
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

Theorem flow2_example : flow2_example_statement.
Proof.
  assert (Hcalls : calls_unboxing_flow2 Flow2Example.facts
                     Flow2Example.tables Flow2Example.cannot_change
                     Flow2Example.code_deps [] [] Flow2Example.facts1
                     Flow2Example.tables1 Flow2Example.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment Flow2Example.unit_expr = true
    ∧ any_kind_arguments Flow2Example.unit_expr = true
    ∧ results_kinded [] [] Flow2Example.unit_expr = true
    ∧ well_formed_unit UE.interface Flow2Example.unit_expr = true
    ∧ exits_one_value UE.interface Flow2Example.unit_expr = true
    ∧ check_closed UE.no_local Flow2Example.facts Flow2Example.tables = true
    ∧ check_field_used_exact UE.no_local Flow2Example.facts
        Flow2Example.tables
      = true
    ∧ check_cc_closed Flow2Example.facts Flow2Example.tables
        Flow2Example.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        Flow2Example.facts Flow2Example.tables
      = true
    ∧ in_heap_fragment Flow2Example.first = true
    ∧ check_closed UE.no_local Flow2Example.facts1 Flow2Example.tables1
      = true
    ∧ check_local_reads_used UE.no_local Flow2Example.facts1
        Flow2Example.tables1
      = true
    ∧ check_unboxed_reads Flow2Example.facts Flow2Example.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  apply (rebuild_unbox_flow2_refines_observably Auto UE.no_local
           UE.interface Flow2Example.unit_expr Flow2Example.tables
           Flow2Example.tables1 Flow2Example.cannot_change [] [] true
           (fun _ => false)); try assumption.
  left. intros f. reflexivity.
Qed.
