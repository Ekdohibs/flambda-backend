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

(* The proofs of FlambdaReaperUnboxingRefinesWithinSpec.v. Nothing here
   needs review.

   The rules read the graph only in their premises, so tables closed under
   them on a graph are closed under them on any graph within it
   ([closed_within], [local_reads_used_within]); the reach, [over_reach]
   and [rem_reach] read the facts only through tests of membership, so
   that they read no more off facts within others ([flow_reach2_within],
   [over_reach_within], [rem_reach_within]). The run of the first rebuilt
   unit is as the reach of its own facts with any tables closed on them
   says ([first_runs], the part of the proof of
   FlambdaReaperUnboxingRefinesReach3Proofs.rebuild_unbox_flow3_refines_cc
   that reads the run), and so, with the unit's tables, as the reach of the
   unit's facts says, which is weaker ([flow_run_mono], [over_run_mono],
   [rem_run_mono]). FlambdaReaperUnboxingRefinesCallsProofs.v then relates
   the two rebuilt units on the fragments read off the unit's analysis
   ([rebuild_unbox_modes_refines_cc_over3],
   [rebuild_unbox_modes_refines_cc_never]). *)

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
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesWithinSpec
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesOver2Proofs
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperPartialGroupsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Facts within others                                                      *)
(* ======================================================================== *)

Lemma list_within_In {A : Type} (eq_dec : ∀ a b : A, {a = b} + {a <> b})
    (l l' : list A) :
  list_within eq_dec l l' = true -> ∀ a : A, a ∈ l -> a ∈ l'.
Proof.
  unfold list_within. rewrite forallb_forall. intros H a Ha.
  specialize (H a Ha).
  destruct (in_dec eq_dec a l') as [Hin | _]; [ exact Hin | discriminate H ].
Qed.

Lemma list_within_refl {A : Type} (eq_dec : ∀ a b : A, {a = b} + {a <> b})
    (l : list A) :
  list_within eq_dec l l = true.
Proof.
  unfold list_within. apply forallb_forall. intros a Ha.
  destruct (in_dec eq_dec a l) as [_ | Hn]; [ reflexivity | contradiction ].
Qed.

Lemma facts_within_refl (F : facts) : facts_within F F = true.
Proof. unfold facts_within. rewrite !list_within_refl. reflexivity. Qed.

(* The facts of [F1] are among those of [F], list by list. *)
Record lists_within (F1 F : facts) : Prop := {
  lw_alias : ∀ p, p ∈ f_alias F1 -> p ∈ f_alias F;
  lw_use : ∀ p, p ∈ f_use F1 -> p ∈ f_use F;
  lw_accessor : ∀ p, p ∈ f_accessor F1 -> p ∈ f_accessor F;
  lw_constructor : ∀ p, p ∈ f_constructor F1 -> p ∈ f_constructor F;
  lw_argument : ∀ p, p ∈ f_argument F1 -> p ∈ f_argument F;
  lw_parameter : ∀ p, p ∈ f_parameter F1 -> p ∈ f_parameter F;
  lw_propagate : ∀ p, p ∈ f_propagate F1 -> p ∈ f_propagate F;
  lw_alias_if_any_source : ∀ p,
    p ∈ f_alias_if_any_source F1 -> p ∈ f_alias_if_any_source F;
  lw_any_usage : ∀ p, p ∈ f_any_usage F1 -> p ∈ f_any_usage F;
  lw_any_source : ∀ p, p ∈ f_any_source F1 -> p ∈ f_any_source F;
  lw_any_source_if_used : ∀ p,
    p ∈ f_any_source_if_used F1 -> p ∈ f_any_source_if_used F;
  lw_zero_alloc_source : ∀ p,
    p ∈ f_zero_alloc_source F1 -> p ∈ f_zero_alloc_source F
}.

Lemma facts_within_lists (F1 F : facts) :
  facts_within F1 F = true -> lists_within F1 F.
Proof.
  unfold facts_within. intros H.
  apply andb_true_iff in H as [H Hz].
  apply andb_true_iff in H as [H Hasu].
  apply andb_true_iff in H as [H Has].
  apply andb_true_iff in H as [H Hau].
  apply andb_true_iff in H as [H Haif].
  apply andb_true_iff in H as [H Hpr].
  apply andb_true_iff in H as [H Hp].
  apply andb_true_iff in H as [H Har].
  apply andb_true_iff in H as [H Hc].
  apply andb_true_iff in H as [H Hac].
  apply andb_true_iff in H as [Ha Hu].
  split; intros p Hin; (eapply list_within_In; [ eassumption | exact Hin ]).
Qed.

(* The graph [G'] is within [G]. *)
Definition graph_within (G' G : graph) : Prop :=
  (∀ a b, g_alias G' a b -> g_alias G a b)
  ∧ (∀ a b, g_use G' a b -> g_use G a b)
  ∧ (∀ a r b, g_accessor G' a r b -> g_accessor G a r b)
  ∧ (∀ a r b, g_constructor G' a r b -> g_constructor G a r b)
  ∧ (∀ a i b, g_argument G' a i b -> g_argument G a i b)
  ∧ (∀ a i b, g_parameter G' a i b -> g_parameter G a i b)
  ∧ (∀ a b c, g_propagate G' a b c -> g_propagate G a b c)
  ∧ (∀ a b c,
       g_alias_if_any_source G' a b c -> g_alias_if_any_source G a b c)
  ∧ (∀ a, g_any_usage G' a -> g_any_usage G a)
  ∧ (∀ a, g_any_source G' a -> g_any_source G a)
  ∧ (∀ a b, g_any_source_if_used G' a b -> g_any_source_if_used G a b)
  ∧ (∀ a, g_zero_alloc_source G' a -> g_zero_alloc_source G a).

Lemma lists_within_graph (F1 F : facts) :
  lists_within F1 F -> graph_within (graph_of_facts F1) (graph_of_facts F).
Proof.
  intros Hw. cbn [graph_of_facts g_alias g_use g_accessor g_constructor
                  g_argument g_parameter g_propagate g_alias_if_any_source
                  g_any_usage g_any_source g_any_source_if_used
                  g_zero_alloc_source].
  repeat split; intros; [ apply (lw_alias _ _ Hw) | apply (lw_use _ _ Hw)
                        | apply (lw_accessor _ _ Hw)
                        | apply (lw_constructor _ _ Hw)
                        | apply (lw_argument _ _ Hw)
                        | apply (lw_parameter _ _ Hw)
                        | apply (lw_propagate _ _ Hw)
                        | apply (lw_alias_if_any_source _ _ Hw)
                        | apply (lw_any_usage _ _ Hw)
                        | apply (lw_any_source _ _ Hw)
                        | apply (lw_any_source_if_used _ _ Hw)
                        | apply (lw_zero_alloc_source _ _ Hw) ];
    assumption.
Qed.

(* Tables closed under the rules on a graph are closed under them on a
   graph within it: the rules read the graph in their premises only. *)
Lemma closed_within (is_local : field -> bool) (G' G : graph)
    (S : solution) :
  graph_within G' G -> closed is_local G S -> closed is_local G' S.
Proof.
  intros (Ha & Hu & Hac & Hc & Har & Hp & Hpr & Haif & Hau & Has & Hasu
          & Hz) H.
  destruct H. split; intros; eauto.
Qed.

Lemma local_reads_used_within (is_local : field -> bool) (G' G : graph)
    (S : solution) :
  graph_within G' G -> local_reads_used is_local G S ->
  local_reads_used is_local G' S.
Proof.
  intros (_ & _ & Hac & _) H base relation to_ Hs Hacc Hl.
  exact (H base relation to_ Hs (Hac _ _ _ Hacc) Hl).
Qed.

(* ======================================================================== *)
(* The reach reads no more off facts within others                          *)
(* ======================================================================== *)

Lemma existsb_within {A : Type} (p q : A -> bool) (l l' : list A) :
  (∀ a : A, a ∈ l -> a ∈ l') -> (∀ a : A, p a = true -> q a = true) ->
  existsb p l = true -> existsb q l' = true.
Proof.
  intros Hl Hpq H. apply existsb_exists in H as (a & Ha & Hpa).
  apply existsb_exists. exists a. exact (conj (Hl a Ha) (Hpq a Hpa)).
Qed.

Section Reach.

Variable F1 F : facts.
Variable T : tables.
Hypothesis Hcons : ∀ p, p ∈ f_constructor F1 -> p ∈ f_constructor F.
Hypothesis Hanys : ∀ p, p ∈ f_any_source F1 -> p ∈ f_any_source F.

Lemma alloc_has_code_within (a : node) (id : code_id) :
  alloc_has_code F1 a id = true -> alloc_has_code F a id = true.
Proof.
  unfold alloc_has_code.
  apply existsb_within; [ exact Hcons | ].
  intros [[b relation] w] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb].
  apply orb_true_iff in H2 as [H2 | H2]; apply orb_true_iff.
  - left. apply mem_node_In. apply Hanys.
    apply mem_node_In. exact H2.
  - right. revert H2.
    apply existsb_within; [ exact Hcons | ].
    intros c Hc. exact Hc.
Qed.

Lemma alloc_unknown_code_within (a : node) (id : code_id) :
  alloc_unknown_code F1 a id = true -> alloc_unknown_code F a id = true.
Proof.
  unfold alloc_unknown_code.
  apply existsb_within; [ exact Hcons | ].
  intros [[b relation] w] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb]. revert H2.
  apply existsb_within; [ exact Hcons | ].
  intros c Hc. exact Hc.
Qed.

Lemma flow_reach_within (f : variable) (id : code_id) :
  flow_reach F1 T f id = true -> flow_reach F T f id = true.
Proof.
  unfold flow_reach. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff; [ left; exact H | ].
  right. revert H. apply existsb_within; [ exact (fun a Ha => Ha) | ].
  intros [x a] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1, (alloc_has_code_within a id H2). reflexivity.
Qed.

Lemma flow_reach2_within (b : bool) (f : variable) (id : code_id) :
  flow_reach2 F1 T b f id = true -> flow_reach2 F T b f id = true.
Proof.
  destruct b; cbn [flow_reach2]; [ exact (flow_reach_within f id) | ].
  unfold flow_reach_u. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (flow_reach_within f id H) | right ].
  revert H. apply existsb_within; [ exact (fun a Ha => Ha) | ].
  intros [x a] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1, (alloc_unknown_code_within a id H2). reflexivity.
Qed.

Lemma over_reach_within (id : code_id) :
  over_reach F1 T id = true -> over_reach F T id = true.
Proof.
  unfold over_reach.
  apply existsb_within; [ exact Hcons | ].
  intros [[w relation] c] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb]. revert H2.
  apply existsb_within; [ exact Hcons | ].
  intros r Hr. exact Hr.
Qed.

Lemma node_reach_within (n : node) (id : code_id) :
  node_reach F1 T n id = true -> node_reach F T n id = true.
Proof.
  unfold node_reach. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff; [ left; exact H | ].
  right. revert H. apply existsb_within; [ exact (fun a Ha => Ha) | ].
  intros [x a] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb].
  apply orb_true_iff in H2 as [H2 | H2]; apply orb_true_iff;
    [ left; exact (alloc_has_code_within a id H2)
    | right; exact (alloc_unknown_code_within a id H2) ].
Qed.

Lemma rem_reach_within (id : code_id) :
  rem_reach F1 T id = true -> rem_reach F T id = true.
Proof.
  unfold rem_reach.
  apply existsb_within; [ exact Hcons | ].
  intros [[w relation] c] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H1 H2].
  rewrite H1. cbn [andb]. revert H2.
  apply existsb_within; [ exact Hcons | ].
  intros [[w' relation'] r] H. cbv beta iota in H |- *.
  apply andb_true_iff in H as [H H3].
  rewrite H, (node_reach_within r id H3). reflexivity.
Qed.

End Reach.

Theorem within_used_reach_covered : within_used_reach_covered_statement.
Proof.
  intros F1 F T e Hwithin.
  unfold code_facts_within in Hwithin.
  apply andb_true_iff in Hwithin as [Hc Has].
  pose proof (list_within_In _ _ _ Hc) as Hc'.
  pose proof (list_within_In _ _ _ Has) as Has'.
  unfold used_reach_covered. cbv zeta.
  apply andb_true_iff. split; apply forallb_forall.
  - intros f _.
    destruct (s_has_usage (solution_of_tables T) (Node_variable f));
      cbn [negb orb]; [ | reflexivity ].
    apply forallb_forall. intros id _. apply forallb_forall. intros b _.
    destruct (flow_reach2 F1 T b f id) eqn:H; [ | reflexivity ].
    rewrite (flow_reach2_within F1 F T Hc' Has' b f id H). reflexivity.
  - intros id _. apply andb_true_iff. split.
    + destruct (over_reach F1 T id) eqn:H; [ | reflexivity ].
      rewrite (over_reach_within F1 F T Hc' id H). reflexivity.
    + destruct (rem_reach F1 T id) eqn:H; [ | reflexivity ].
      rewrite (rem_reach_within F1 F T Hc' Has' id H). reflexivity.
Qed.

(* ======================================================================== *)
(* The run of the first rebuilt unit                                        *)
(* ======================================================================== *)

(* The run of a unit is as the reach of its facts with tables closed on
   them says: the part of the proof of
   FlambdaReaperUnboxingRefinesReach3Proofs.rebuild_unbox_flow3_refines_cc
   that reads the run. *)
Lemma first_runs (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e1 : expr)
    (T1 : tables) :
  is_local_slots_only is_local ->
  in_heap_fragment e1 = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e1))
    (solution_of_tables T1) ->
  local_reads_used is_local (graph_of_facts (traverse_unit flag u e1))
    (solution_of_tables T1) ->
  flow_run
    (fun b f id => flow_reach2 (traverse_unit flag u e1) T1 b f id = true)
    (State e1 [] [] empty_static_env empty_heap no_reads)
  ∧ over_run (fun id => over_reach (traverse_unit flag u e1) T1 id = true)
      (State e1 [] [] empty_static_env empty_heap no_reads)
  ∧ rem_run (fun id => rem_reach (traverse_unit flag u e1) T1 id = true)
      (State e1 [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hfrag1 Hc1 Hlr1.
  set (F1 := traverse_unit flag u e1) in *.
  split; [ | split ].
  - intros n s Hrun ap f v id slots _ _ Hf.
    destruct (run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                Hlr1)
      as (located & cds & Hcds & Hnd & Hflows).
    fold F1 in Hcds, Hflows. rewrite Hrun in Hflows.
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
  - intros n s Hrun.
    pose proof (run_over_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hover.
    fold F1 in Hover. rewrite Hrun in Hover.
    exact (step_over_ok_mono _ _ s (over_used_reach F1 T1) Hover).
  - intros n s Hrun.
    pose proof (run_rem_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hrem.
    fold F1 in Hrem. rewrite Hrun in Hrem.
    exact (step_rem_ok_mono _ _ s (rem_used_reach F1 T1) Hrem).
Qed.

Lemma flow_run_mono (RCH RCH' : bool -> variable -> code_id -> Prop)
    (s : state) :
  (∀ (b : bool) (f : variable) (id : code_id), RCH b f id -> RCH' b f id) ->
  flow_run RCH s -> flow_run RCH' s.
Proof.
  intros Himp H n s' Hrun ap f v id slots Hs Hc Hf.
  destruct (H n s' Hrun ap f v id slots Hs Hc Hf) as [H1 H2].
  split; intros Hclo; apply Himp; [ exact (H1 Hclo) | exact (H2 Hclo) ].
Qed.

Lemma over_run_mono (OV OV' : code_id -> Prop) (s : state) :
  (∀ id : code_id, OV id -> OV' id) -> over_run OV s -> over_run OV' s.
Proof.
  intros Himp H n s' Hrun. exact (step_over_ok_mono _ _ s' Himp (H n s' Hrun)).
Qed.

Lemma rem_run_mono (REM REM' : code_id -> Prop) (s : state) :
  (∀ id : code_id, REM id -> REM' id) -> rem_run REM s -> rem_run REM' s.
Proof.
  intros Himp H n s' Hrun. exact (step_rem_ok_mono _ _ s' Himp (H n s' Hrun)).
Qed.

(* The run of the first rebuilt unit, where its facts are within the
   unit's, is as the reach of the unit's facts with the unit's tables
   says. *)
Lemma within_runs (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (pdc : bool) :
  is_local_slots_only is_local ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  in_heap_fragment (first_rebuild flag u e T cannot_change pdc) = true ->
  facts_within
    (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc))
    (traverse_unit flag u e)
  = true ->
  flow_run
    (fun b f id => flow_reach2 (traverse_unit flag u e) T b f id = true)
    (State (first_rebuild flag u e T cannot_change pdc) [] []
       empty_static_env empty_heap no_reads)
  ∧ over_run (fun id => over_reach (traverse_unit flag u e) T id = true)
      (State (first_rebuild flag u e T cannot_change pdc) [] []
         empty_static_env empty_heap no_reads)
  ∧ rem_run (fun id => rem_reach (traverse_unit flag u e) T id = true)
      (State (first_rebuild flag u e T cannot_change pdc) [] []
         empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hclosed Hlrs Hfrag1 Hwithin.
  set (e1 := first_rebuild flag u e T cannot_change pdc) in *.
  pose proof (facts_within_lists _ _ Hwithin) as Hw.
  pose proof (lists_within_graph _ _ Hw) as Hg.
  pose proof (closed_within is_local _ _ _ Hg
                (check_closed_sound is_local _ _ Hclosed)) as Hc1.
  pose proof (local_reads_used_within is_local _ _ _ Hg
                (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                   is_local _ _ Hlrs)) as Hlr1.
  destruct (first_runs flag is_local u e1 T Hslots Hfrag1 Hc1 Hlr1)
    as (Hfl & Hov & Hrm).
  split; [ | split ].
  - exact (flow_run_mono _ _ _
             (fun b f id =>
                flow_reach2_within _ _ T (lw_constructor _ _ Hw)
                  (lw_any_source _ _ Hw) b f id)
             Hfl).
  - exact (over_run_mono _ _ _
             (fun id => over_reach_within _ _ T (lw_constructor _ _ Hw) id)
             Hov).
  - exact (rem_run_mono _ _ _
             (fun id =>
                rem_reach_within _ _ T (lw_constructor _ _ Hw)
                  (lw_any_source _ _ Hw) id)
             Hrm).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem within_flow3_refines_cc : within_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hwithin Hcalls Hcheck.
  destruct (within_runs flag is_local u e T cannot_change pdc Hslots Hclosed
              Hlrs Hfrag1 Hwithin)
    as (Hfl & Hov & Hrm).
  exact (rebuild_unbox_modes_refines_cc_over3 flag is_local u e T
           cannot_change dominated to_unbox pdc nr _ _ _ Hslots Hfrag Hclosed
           Hcalls Hcheck Hfl Hov Hrm).
Qed.

Theorem within_flow3_observably_local :
  within_flow3_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hwithin Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (within_flow3_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc nr Hslots Hfrag Hclosed Hlrs Hfrag1
              Hwithin Hcalls Hcheck)).
Qed.

Theorem within_flow3_refines_observably :
  within_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hwithin Hcalls Hcheck.
  exact (within_flow3_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

Theorem within_never_refines_cc : within_never_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hwithin Hcalls Hcheck.
  destruct (within_runs flag is_local u e T cannot_change pdc Hslots Hclosed
              Hlrs Hfrag1 Hwithin)
    as (Hfl & Hov & Hrm).
  exact (rebuild_unbox_modes_refines_cc_never flag is_local u e T
           cannot_change dominated to_unbox pdc _ _ _ _ Hslots Hfrag Hclosed
           Hcalls Hcheck Hfl Hov Hrm).
Qed.

Theorem within_never_observably_local :
  within_never_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hwithin Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (within_never_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc Hslots Hfrag Hclosed Hlrs Hfrag1
              Hwithin Hcalls Hcheck)).
Qed.

Theorem within_never_refines_observably :
  within_never_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hwithin Hcalls Hcheck.
  exact (within_never_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* The units that the first rebuild leaves as they are                     *)
(* ======================================================================== *)

(* A settled unit is of the class: its first rebuild is in the fragment
   of the heap, as the unit is, and has the unit's facts. *)
Lemma settled_within (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (pdc : bool) :
  in_flow_fragment e = true -> settled flag u e T cannot_change pdc ->
  in_heap_fragment (first_rebuild flag u e T cannot_change pdc) = true
  ∧ facts_within
      (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc))
      (traverse_unit flag u e)
    = true.
Proof.
  unfold settled. intros Hfrag Hs. rewrite Hs.
  exact (conj (FlambdaReaperFlowProofs.in_flow_fragment_in_heap_fragment e
                 Hfrag)
              (facts_within_refl _)).
Qed.

Theorem settled_flow3_refines_cc : settled_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hlrs Hs Hcalls Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  pose proof (within_flow3_refines_cc flag is_local u e T cannot_change
                dominated to_unbox pdc nr Hslots Hfrag Hclosed Hlrs Hfrag1
                Hwithin Hcalls Hcheck) as H.
  unfold settled in Hs. rewrite Hs in H. exact H.
Qed.

Theorem settled_flow3_observably_local :
  settled_flow3_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hs Hcalls
    Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  exact (within_flow3_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           Hlrs Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

Theorem settled_flow3_refines_observably :
  settled_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hs Hcalls
    Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  exact (within_flow3_refines_observably flag is_local u e T cannot_change
           dominated to_unbox pdc nr Hslots Hproj Hfrag Hany Hrk Hwf Hone
           Hclosed Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

Theorem settled_never_refines_cc : settled_never_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hlrs Hs Hcalls Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  pose proof (within_never_refines_cc flag is_local u e T cannot_change
                dominated to_unbox pdc Hslots Hfrag Hclosed Hlrs Hfrag1
                Hwithin Hcalls Hcheck) as H.
  unfold settled in Hs. rewrite Hs in H. exact H.
Qed.

Theorem settled_never_observably_local :
  settled_never_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hs Hcalls
    Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  exact (within_never_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           Hlrs Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

Theorem settled_never_refines_observably :
  settled_never_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hs Hcalls
    Hcheck.
  destruct (settled_within flag u e T cannot_change pdc Hfrag Hs)
    as [Hfrag1 Hwithin].
  exact (within_never_refines_observably flag is_local u e T cannot_change
           dominated to_unbox pdc Hslots Hproj Hfrag Hany Hrk Hwf Hone
           Hclosed Hexact Hccc Hwid Hfrag1 Hwithin Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Theorem within_example : within_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  apply (within_never_refines_observably Auto UE.no_local UE.interface
           NeverIndExample.unit_expr NeverIndExample.tables
           NeverIndExample.cannot_change [] [] true);
    [ intros f Hf; discriminate Hf | left; intros f; reflexivity
    | vm_compute; reflexivity .. ].
Qed.

Theorem settled_example : settled_example_statement.
Proof.
  assert (Hs : settled Auto UE.interface SettledExample.unit_expr
                 SettledExample.tables SettledExample.cannot_change true)
    by (unfold settled; vm_compute; reflexivity).
  split; [ exact Hs | ].
  split; [ vm_compute; reflexivity | ].
  apply (settled_flow3_refines_cc Auto UE.no_local UE.interface
           SettledExample.unit_expr SettledExample.tables
           SettledExample.cannot_change [] [] true (fun _ => false));
    [ intros f Hf; discriminate Hf | vm_compute; reflexivity
    | vm_compute; reflexivity | vm_compute; reflexivity | exact Hs
    | vm_compute; reflexivity | vm_compute; reflexivity ].
Qed.
