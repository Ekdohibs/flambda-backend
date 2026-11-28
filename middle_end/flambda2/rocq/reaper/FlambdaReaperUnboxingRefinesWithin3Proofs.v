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

(* The proofs of FlambdaReaperUnboxingRefinesWithin3Spec.v. Nothing here
   needs review.

   [poisoned_tables F T] differs from [T] in any source, a source and an
   alias to the constants, at nodes with no use ([P_alias],
   [P_any_source], [P_has_source]); a node with no use flows only to
   nodes with no use ([unused_alias], [unused_accessor], [unused_use]).
   [poison_closed]: these tables are closed on [poison_graph] (the facts
   of [F], the poison facts, and the simple facts that the tables
   satisfy), rule by rule, with what [unit_poison_ok] says of [F]
   ([unit_poison_ok_spec]) and what [poison_constructor_ok] and
   [poison_argument_ok] say of the poison ([poison_constructor_spec],
   [poison_argument_spec]). [facts_poison_graph]: the renamed facts of
   the first rebuilt unit that pass [facts_poison_ok] are in
   [poison_graph]. FlambdaReaperUnboxingRefinesWithin2Proofs.v then gives
   the tables of the first rebuilt unit, and at the variables with a use
   the reach of [poisoned_tables F T] is that of [T] ([flow_reach2_used],
   [rem_reach_used]). *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaBindersSpec FlambdaRenameApartSpec
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
  FlambdaReaperUnboxingRefinesWithin2Spec
  FlambdaReaperUnboxingRefinesWithin3Spec
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesUnitFlowProofs
  FlambdaReaperUnboxingRefinesWithinProofs
  FlambdaReaperUnboxingRefinesWithin2Proofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The unit's tables with poison                                            *)
(* ======================================================================== *)

Section Poison.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

Local Abbreviation S := (solution_of_tables T).
Local Abbreviation P := (solution_of_tables (poisoned_tables F T)).
Local Abbreviation U := (unused_nodes F T).
Local Abbreviation G := (graph_of_facts F).
Local Abbreviation consts := Node_all_constants.

Lemma in_unused (n : node) :
  n ∈ U <->
  n ∈ map fst (t_alias T) ++ nodes_of_facts F
  ∧ s_has_usage S n = false.
Proof.
  unfold unused_nodes. rewrite filter_In, negb_true_iff. reflexivity.
Qed.

Lemma P_alias (t f : node) :
  s_alias P t f = true <-> s_alias S t f = true ∨ (f = consts ∧ t ∈ U).
Proof.
  cbn [solution_of_tables s_alias poisoned_tables t_alias].
  rewrite !mem_pair_In, in_app_iff, in_map_iff. split.
  - intros [H | (n & Heq & Hn)]; [ left; exact H | right ].
    injection Heq as -> ->. split; [ reflexivity | exact Hn ].
  - intros [H | [-> Hn]]; [ left; exact H | right ].
    exists t. split; [ reflexivity | exact Hn ].
Qed.

Lemma P_any_source (n : node) :
  s_any_source P n = true <-> s_any_source S n = true ∨ n ∈ U.
Proof.
  cbn [solution_of_tables s_any_source poisoned_tables t_any_source].
  rewrite !mem_node_In, in_app_iff. reflexivity.
Qed.

Lemma P_has_source (n : node) :
  s_has_source P n = true <-> s_has_source S n = true ∨ n ∈ U.
Proof.
  cbn [solution_of_tables s_has_source poisoned_tables t_has_source].
  rewrite !mem_node_In, in_app_iff. reflexivity.
Qed.

Lemma P_any_source_false (n : node) :
  s_any_source P n = false -> s_any_source S n = false ∧ ~ n ∈ U.
Proof.
  intros H. split.
  - destruct (s_any_source S n) eqn:E; [ | reflexivity ].
    rewrite (proj2 (P_any_source n) (or_introl E)) in H. exact H.
  - intros Hn. rewrite (proj2 (P_any_source n) (or_intror Hn)) in H.
    discriminate H.
Qed.

Lemma nodes_accessor (t : node) (relation : field) (b : node) :
  (t, relation, b) ∈ f_accessor F -> t ∈ nodes_of_facts F.
Proof.
  intros H. unfold nodes_of_facts. rewrite !in_app_iff.
  right; right; left. apply in_flat_map. exists (t, relation, b).
  split; [ exact H | left; reflexivity ].
Qed.

Lemma nodes_use (t f : node) : (t, f) ∈ f_use F -> t ∈ nodes_of_facts F.
Proof.
  intros H. unfold nodes_of_facts. rewrite !in_app_iff.
  right; left. apply in_flat_map. exists (t, f).
  split; [ exact H | left; reflexivity ].
Qed.

Hypothesis HS : closed is_local G S.
Hypothesis Hall : s_any_usage S consts = true.

Lemma unused_any_usage (n : node) : n ∈ U -> s_any_usage S n = false.
Proof.
  intros Hn. apply in_unused in Hn as [_ Hu].
  destruct (s_any_usage S n) eqn:E; [ | reflexivity ].
  rewrite (has_usage_any _ _ _ HS n E) in Hu. exact Hu.
Qed.

Lemma consts_used : ~ consts ∈ U.
Proof.
  intros Hn. rewrite (unused_any_usage _ Hn) in Hall. discriminate Hall.
Qed.

Lemma unused_alias (t f : node) :
  f ∈ U -> s_alias S t f = true -> t ∈ U.
Proof.
  intros Hf Ha. apply in_unused. split.
  - apply in_app_iff. left. apply in_map_iff. exists (t, f).
    split; [ reflexivity | ].
    cbn [solution_of_tables s_alias] in Ha. apply mem_pair_In. exact Ha.
  - apply in_unused in Hf as [_ Hf].
    destruct (s_has_usage S t) eqn:E; [ | reflexivity ].
    rewrite (has_usage_alias _ _ _ HS t f E Ha) in Hf. exact Hf.
Qed.

Lemma unused_accessor (t : node) (relation : field) (b : node) :
  b ∈ U -> g_accessor G t relation b -> t ∈ U.
Proof.
  intros Hb Ha. apply in_unused. split.
  - apply in_app_iff. right. exact (nodes_accessor t relation b Ha).
  - apply in_unused in Hb as [_ Hb].
    destruct (s_has_usage S t) eqn:E; [ | reflexivity ].
    rewrite (has_usage_accessor _ _ _ HS t relation b E Ha) in Hb. exact Hb.
Qed.

Lemma unused_use (t f : node) : f ∈ U -> g_use G t f -> t ∈ U.
Proof.
  intros Hf Hu. apply in_unused. split.
  - apply in_app_iff. right. exact (nodes_use t f Hu).
  - pose proof (unused_any_usage f Hf) as Hf'.
    destruct (s_has_usage S t) eqn:E; [ | reflexivity ].
    rewrite (any_usage_use _ _ _ HS t f E Hu) in Hf'. exact Hf'.
Qed.

(* A constructor of [F] with poison in place of what it holds, where
   [poison_constructor_ok] holds. *)
Lemma poison_constructor_spec (b : node) (relation : field) (f : node) :
  poison_constructor_ok is_local F T b relation f = true ->
  f = consts
  ∧ (∃ y, (b, relation, y) ∈ f_constructor F)
  ∧ (∀ t base,
       (t, relation, base) ∈ f_accessor F ->
       (s_usages P b base = true ∧ s_any_usage P b = false)
       ∨ (is_local relation = true ∧ s_usages P b base = true)
       ∨ (s_sources P base b = true ∧ s_any_source P base = false)
       ∨ (is_local relation = true ∧ s_sources P base b = true) ->
       s_alias P t consts = true)
  ∧ s_has_source P b = true
  ∧ (s_any_usage P b = true -> is_local relation = true ->
     s_escaping_field P relation consts = true).
Proof.
  unfold poison_constructor_ok. intros H.
  apply andb_true_iff in H as [H Hesc]. apply andb_true_iff in H as [H Hhs].
  apply andb_true_iff in H as [H Hacc]. apply andb_true_iff in H as [H Hw].
  apply andb_true_iff in H as [Hf _]. apply node_eqb_eq in Hf.
  split; [ exact Hf | ]. split; [ | split; [ | split ] ].
  - apply existsb_exists in Hw as ([[b' rel'] y] & Hin & Hp).
    apply andb_true_iff in Hp as [Hb Hr]. apply node_eqb_eq in Hb.
    apply field_eqb_eq in Hr. subst b' rel'. exists y. exact Hin.
  - intros t base Hin Hcase. rewrite forallb_forall in Hacc.
    specialize (Hacc _ Hin). cbv beta iota in Hacc.
    rewrite field_eqb_refl in Hacc. cbn [andb] in Hacc.
    apply orb_true_iff in Hacc as [Hacc | Hacc]; [ | exact Hacc ].
    exfalso. apply negb_true_iff in Hacc.
    destruct Hcase as [[H1 H2] | [[H1 H2] | [[H1 H2] | [H1 H2]]]];
      rewrite ?H1, ?H2 in Hacc; cbn [negb andb orb] in Hacc;
      rewrite ?orb_true_r in Hacc; discriminate Hacc.
  - exact Hhs.
  - intros Hu Hl. rewrite Hu, Hl in Hesc. exact Hesc.
Qed.

(* An argument of [F] given poison, where [poison_argument_ok] holds. *)
Lemma poison_argument_spec (f : node) (i : nat) (w : node) :
  poison_argument_ok F T f i w = true ->
  f = consts
  ∧ (∃ y, (y, i, w) ∈ f_argument F)
  ∧ s_has_usage P w = true
  ∧ (∀ b t,
       (b, i, t) ∈ f_parameter F ->
       (s_usages P b w = true ∧ s_any_usage P b = false)
       ∨ (s_sources P w b = true ∧ s_any_source P w = false) ->
       s_alias P t consts = true).
Proof.
  unfold poison_argument_ok. intros H.
  apply andb_true_iff in H as [H Hpar]. apply andb_true_iff in H as [H Hu].
  apply andb_true_iff in H as [Hf Hw]. apply node_eqb_eq in Hf.
  split; [ exact Hf | ]. split; [ | split; [ exact Hu | ] ].
  - apply existsb_exists in Hw as ([[y i'] w'] & Hin & Hp).
    apply andb_true_iff in Hp as [Hi Hw']. apply Nat.eqb_eq in Hi.
    apply node_eqb_eq in Hw'. subst i' w'. exists y. exact Hin.
  - intros b t Hin Hcase. rewrite forallb_forall in Hpar.
    specialize (Hpar _ Hin). cbv beta iota in Hpar.
    rewrite Nat.eqb_refl in Hpar. cbn [andb] in Hpar.
    apply orb_true_iff in Hpar as [Hpar | Hpar]; [ | exact Hpar ].
    exfalso. apply negb_true_iff in Hpar.
    destruct Hcase as [[H1 H2] | [H1 H2]];
      rewrite ?H1, ?H2 in Hpar; cbn [negb andb orb] in Hpar;
      rewrite ?orb_true_r in Hpar; discriminate Hpar.
Qed.

(* The graph that [poisoned_tables F T] is closed on: the facts of [F],
   the constructors and arguments of [F] given poison, and the facts that
   the tables satisfy. *)
Definition poison_graph : graph :=
  {| g_alias := fun a b => g_alias G a b ∨ s_alias P a b = true;
     g_use := fun a b =>
       g_use G a b
       ∨ ((s_has_usage P a = true -> s_any_usage P b = true)
          ∧ (s_has_source P b = true -> s_any_source P a = true));
     g_accessor := g_accessor G;
     g_constructor := fun b relation f =>
       g_constructor G b relation f
       ∨ poison_constructor_ok is_local F T b relation f = true;
     g_argument := fun f i w =>
       g_argument G f i w ∨ poison_argument_ok F T f i w = true;
     g_parameter := g_parameter G;
     g_propagate := fun i t f =>
       g_propagate G i t f
       ∨ (s_any_usage P i = true -> s_alias P t f = true);
     g_alias_if_any_source := fun c t f =>
       g_alias_if_any_source G c t f
       ∨ (s_any_source P c = true -> s_alias P t f = true);
     g_any_usage := fun a => g_any_usage G a ∨ s_any_usage P a = true;
     g_any_source := g_any_source G;
     g_any_source_if_used := fun i x =>
       g_any_source_if_used G i x
       ∨ (s_any_usage P i = true -> s_any_source P x = true);
     g_zero_alloc_source := g_zero_alloc_source G |}.

(* What [unit_poison_ok] says of the facts of [F]. *)
Hypothesis Haias : ∀ c t f,
  (c, t, f) ∈ f_alias_if_any_source F -> c ∈ U -> s_alias P t f = true.
Hypothesis Harg_from : ∀ f i b,
  (f, i, b) ∈ f_argument F -> f ∈ U -> s_has_usage P b = true.
Hypothesis Harg_base : ∀ f i b,
  (f, i, b) ∈ f_argument F -> b ∈ U -> s_any_usage P f = true.
Hypothesis Hcons : ∀ b relation f,
  (b, relation, f) ∈ f_constructor F -> f ∈ U -> s_has_source P b = true.
Hypothesis Hacc : ∀ t relation b,
  (t, relation, b) ∈ f_accessor F -> b ∈ U -> is_local relation = false.

Lemma P_alias_S (t f : node) : s_alias S t f = true -> s_alias P t f = true.
Proof. intros H. apply P_alias. left. exact H. Qed.

Lemma P_alias_U (t : node) : t ∈ U -> s_alias P t consts = true.
Proof. intros H. apply P_alias. right. split; [ reflexivity | exact H ]. Qed.

Lemma P_any_source_S (n : node) :
  s_any_source S n = true -> s_any_source P n = true.
Proof. intros H. apply P_any_source. left. exact H. Qed.

Lemma P_any_source_U (n : node) : n ∈ U -> s_any_source P n = true.
Proof. intros H. apply P_any_source. right. exact H. Qed.

Lemma P_has_source_S (n : node) :
  s_has_source S n = true -> s_has_source P n = true.
Proof. intros H. apply P_has_source. left. exact H. Qed.

Lemma P_has_source_U (n : node) : n ∈ U -> s_has_source P n = true.
Proof. intros H. apply P_has_source. right. exact H. Qed.

Lemma unused_has_usage (n : node) : n ∈ U -> s_has_usage S n = false.
Proof. intros H. apply in_unused in H as [_ H]. exact H. Qed.

Lemma P_any_usage (n : node) : s_any_usage P n = s_any_usage S n.
Proof. reflexivity. Qed.

Lemma P_has_usage (n : node) : s_has_usage P n = s_has_usage S n.
Proof. reflexivity. Qed.

Lemma P_usages (a b : node) : s_usages P a b = s_usages S a b.
Proof. reflexivity. Qed.

Lemma P_sources (a b : node) : s_sources P a b = s_sources S a b.
Proof. reflexivity. Qed.

Lemma P_reading_field (f : field) (n : node) :
  s_reading_field P f n = s_reading_field S f n.
Proof. reflexivity. Qed.

Lemma P_escaping_field (f : field) (n : node) :
  s_escaping_field P f n = s_escaping_field S f n.
Proof. reflexivity. Qed.

Lemma P_field_used (n : node) (f : field) :
  s_field_used P n f = s_field_used S n f.
Proof. reflexivity. Qed.

(* [poisoned_tables F T] is closed on [poison_graph]: at the nodes with
   no use, that it adds any source to, the rules ask nothing that they
   do not give, the uses and sources being those of [T]. *)
Lemma poison_closed : closed is_local poison_graph P.
Proof.
  split; cbn [poison_graph g_alias g_use g_accessor g_constructor g_argument
              g_parameter g_propagate g_alias_if_any_source g_any_usage
              g_any_source g_any_source_if_used g_zero_alloc_source];
    intros;
    rewrite ?P_any_usage, ?P_has_usage, ?P_usages, ?P_sources,
      ?P_reading_field, ?P_escaping_field, ?P_field_used in *.
  - (* alias_input *)
    match goal with H : _ ∨ _ |- _ => destruct H as [H | H] end;
      [ apply P_alias_S; exact (alias_input _ _ _ HS _ _ H) | exact H ].
  - (* alias_propagate *)
    match goal with H : _ ∨ _ |- _ => destruct H as [H | H] end;
      [ apply P_alias_S; eapply (alias_propagate _ _ _ HS); eassumption
      | auto ].
  - (* alias_if_any_source *)
    rename H into Has, H0 into H.
    destruct H as [H | H]; [ | exact (H Has) ].
    apply P_any_source in Has as [Has | Has];
      [ apply P_alias_S; exact (alias_if_any_source _ _ _ HS _ _ _ Has H)
      | exact (Haias _ _ _ H Has) ].
  - (* alias_constructor_usage *)
    rename H into Hc, H0 into Hu, H1 into Ha, H2 into Hac.
    destruct Hc as [Hc | Hc];
      [ apply P_alias_S;
        exact (alias_constructor_usage _ _ _ HS _ _ _ _ _ Hc Hu Ha Hac) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _ & Hp & _).
    exact (Hp _ _ Hac (or_introl (conj Hu Ha))).
  - (* alias_local_constructor_usage *)
    rename H into Hl, H0 into Hc, H1 into Hu, H2 into Hac.
    destruct Hc as [Hc | Hc];
      [ apply P_alias_S;
        exact (alias_local_constructor_usage _ _ _ HS _ _ _ _ _ Hl Hc Hu Hac)
      | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _ & Hp & _).
    exact (Hp _ _ Hac (or_intror (or_introl (conj Hl Hu)))).
  - (* alias_parameter_usage *)
    rename H into Hp, H0 into Hu, H1 into Ha, H2 into Har.
    destruct Har as [Har | Har];
      [ apply P_alias_S;
        exact (alias_parameter_usage _ _ _ HS _ _ _ _ _ Hp Hu Ha Har) | ].
    destruct (poison_argument_spec _ _ _ Har) as (-> & _ & _ & Hpar).
    exact (Hpar _ _ Hp (or_introl (conj Hu Ha))).
  - (* alias_accessor_source *)
    rename H into Hac, H0 into Hs, H1 into Ha, H2 into Hc.
    destruct Hc as [Hc | Hc];
      [ apply P_alias_S;
        exact (alias_accessor_source _ _ _ HS _ _ _ _ _ Hac Hs
                 (proj1 (P_any_source_false _ Ha)) Hc) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _ & Hp & _).
    exact (Hp _ _ Hac (or_intror (or_intror (or_introl (conj Hs Ha))))).
  - (* alias_local_accessor_source *)
    rename H into Hl, H0 into Hac, H1 into Hs, H2 into Hc.
    destruct Hc as [Hc | Hc];
      [ apply P_alias_S;
        exact (alias_local_accessor_source _ _ _ HS _ _ _ _ _ Hl Hac Hs Hc)
      | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _ & Hp & _).
    exact (Hp _ _ Hac (or_intror (or_intror (or_intror (conj Hl Hs))))).
  - (* alias_escaping_reading *)
    apply P_alias_S. eapply (alias_escaping_reading _ _ _ HS); eassumption.
  - (* alias_argument_source *)
    rename H into Har, H0 into Hs, H1 into Ha, H2 into Hp.
    destruct Har as [Har | Har];
      [ apply P_alias_S;
        exact (alias_argument_source _ _ _ HS _ _ _ _ _ Har Hs
                 (proj1 (P_any_source_false _ Ha)) Hp) | ].
    destruct (poison_argument_spec _ _ _ Har) as (-> & _ & _ & Hpar).
    exact (Hpar _ _ Hp (or_intror (conj Hs Ha))).
  - (* has_usage_any *)
    eapply (has_usage_any _ _ _ HS); eassumption.
  - (* has_usage_alias *)
    rename H into Hh, H0 into Ha.
    apply P_alias in Ha as [Ha | [-> Hu]];
      [ exact (has_usage_alias _ _ _ HS _ _ Hh Ha)
      | rewrite (unused_has_usage _ Hu) in Hh; discriminate Hh ].
  - (* has_usage_accessor *)
    eapply (has_usage_accessor _ _ _ HS); eassumption.
  - (* has_usage_argument *)
    rename H into Hs, H0 into Har.
    destruct Har as [Har | Har].
    + apply P_has_source in Hs as [Hs | Hs];
        [ exact (has_usage_argument _ _ _ HS _ _ _ Hs Har)
        | exact (Harg_from _ _ _ Har Hs) ].
    + destruct (poison_argument_spec _ _ _ Har) as (_ & _ & Hu & _).
      exact Hu.
  - (* has_source_any *)
    rename H into Has.
    apply P_any_source in Has as [Has | Has];
      [ apply P_has_source_S; exact (has_source_any _ _ _ HS _ Has)
      | apply P_has_source_U; exact Has ].
  - (* has_source_alias *)
    rename H into Hs, H0 into Ha.
    apply P_alias in Ha as [Ha | [-> Hu]];
      [ | apply P_has_source_U; exact Hu ].
    apply P_has_source in Hs as [Hs | Hs];
      [ apply P_has_source_S; exact (has_source_alias _ _ _ HS _ _ Hs Ha)
      | apply P_has_source_U; exact (unused_alias _ _ Hs Ha) ].
  - (* has_source_constructor *)
    rename H into Hs, H0 into Hc.
    destruct Hc as [Hc | Hc].
    + apply P_has_source in Hs as [Hs | Hs];
        [ apply P_has_source_S;
          exact (has_source_constructor _ _ _ HS _ _ _ Hs Hc)
        | exact (Hcons _ _ _ Hc Hs) ].
    + destruct (poison_constructor_spec _ _ _ Hc) as (_ & _ & _ & Hh & _).
      exact Hh.
  - (* has_source_function_slot *)
    rename H into Hc, H0 into Hf.
    destruct Hc as [Hc | Hc];
      [ apply P_has_source_S;
        exact (has_source_function_slot _ _ _ HS _ _ _ Hc Hf) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    apply P_has_source_S. exact (has_source_function_slot _ _ _ HS _ _ _ Hy Hf).
  - (* has_source_parameter *)
    apply P_has_source_S. eapply (has_source_parameter _ _ _ HS); eassumption.
  - (* any_usage_use *)
    rename H into Hh, H0 into Hu.
    destruct Hu as [Hu | [Hu _]];
      [ exact (any_usage_use _ _ _ HS _ _ Hh Hu) | exact (Hu Hh) ].
  - (* any_usage_alias *)
    rename H into Hau, H0 into Ha.
    apply P_alias in Ha as [Ha | [-> Hu]];
      [ exact (any_usage_alias _ _ _ HS _ _ Hau Ha)
      | rewrite (unused_any_usage _ Hu) in Hau; discriminate Hau ].
  - (* any_usage_constructor *)
    rename H into Hau, H0 into Hc, H1 into Hl.
    destruct Hc as [Hc | Hc];
      [ exact (any_usage_constructor _ _ _ HS _ _ _ Hau Hc Hl) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _). exact Hall.
  - (* any_usage_argument *)
    rename H into Has, H0 into Har.
    destruct Har as [Har | Har];
      [ | destruct (poison_argument_spec _ _ _ Har) as (-> & _); exact Hall ].
    apply P_any_source in Has as [Has | Has];
      [ exact (any_usage_argument _ _ _ HS _ _ _ Has Har)
      | exact (Harg_base _ _ _ Har Has) ].
  - (* any_source_input *)
    apply P_any_source_S. eapply (any_source_input _ _ _ HS); eassumption.
  - (* any_source_zero_alloc *)
    apply P_any_source_S. eapply (any_source_zero_alloc _ _ _ HS);
      eassumption.
  - (* any_source_if_used *)
    rename H into Hau, H0 into H.
    destruct H as [H | H];
      [ apply P_any_source_S; exact (any_source_if_used _ _ _ HS _ _ Hau H)
      | exact (H Hau) ].
  - (* any_source_alias *)
    rename H into Ha, H0 into Has.
    apply P_alias in Ha as [Ha | [-> Hu]];
      [ | apply P_any_source_U; exact Hu ].
    apply P_any_source in Has as [Has | Has];
      [ apply P_any_source_S; exact (any_source_alias _ _ _ HS _ _ Ha Has)
      | apply P_any_source_U; exact (unused_alias _ _ Has Ha) ].
  - (* any_source_parameter *)
    apply P_any_source_S. eapply (any_source_parameter _ _ _ HS);
      eassumption.
  - (* any_source_accessor *)
    rename H into Has, H0 into Hac, H1 into Hl.
    apply P_any_source in Has as [Has | Has];
      [ apply P_any_source_S;
        exact (any_source_accessor _ _ _ HS _ _ _ Has Hac Hl)
      | apply P_any_source_U; exact (unused_accessor _ _ _ Has Hac) ].
  - (* any_source_use *)
    rename H into Hs, H0 into Hu.
    destruct Hu as [Hu | [_ Hu]]; [ | exact (Hu Hs) ].
    apply P_has_source in Hs as [Hs | Hs];
      [ apply P_any_source_S; exact (any_source_use _ _ _ HS _ _ Hs Hu)
      | apply P_any_source_U; exact (unused_use _ _ Hs Hu) ].
  - (* any_usage_input *)
    match goal with H : _ ∨ _ |- _ => destruct H as [H | H] end;
      [ exact (any_usage_input _ _ _ HS _ H) | exact H ].
  - (* usages_accessor *)
    eapply (usages_accessor _ _ _ HS); eassumption.
  - (* usages_argument *)
    rename H into Har, H0 into Ha.
    destruct Har as [Har | Har];
      [ exact (usages_argument _ _ _ HS _ _ _ Har Ha) | ].
    destruct (poison_argument_spec _ _ _ Har) as (_ & (y & Hy) & _).
    exact (usages_argument _ _ _ HS _ _ _ Hy Ha).
  - (* usages_alias *)
    rename H into Hu, H0 into Ha1, H1 into Ha, H2 into Ha2.
    apply P_alias in Ha as [Ha | [-> _]];
      [ exact (usages_alias _ _ _ HS _ _ _ Hu Ha1 Ha Ha2)
      | rewrite Hall in Ha2; discriminate Ha2 ].
  - (* sources_constructor *)
    rename H into Hc, H0 into Ha.
    apply P_any_source_false in Ha as [Ha _].
    destruct Hc as [Hc | Hc];
      [ exact (sources_constructor _ _ _ HS _ _ _ Hc Ha) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    exact (sources_constructor _ _ _ HS _ _ _ Hy Ha).
  - (* sources_parameter *)
    rename H into Hp, H0 into Ha.
    exact (sources_parameter _ _ _ HS _ _ _ Hp
             (proj1 (P_any_source_false _ Ha))).
  - (* sources_alias *)
    rename H into Hs, H0 into Ha1, H1 into Ha, H2 into Ha2.
    apply P_any_source_false in Ha1 as [Ha1 _].
    apply P_any_source_false in Ha2 as [Ha2 Hn2].
    apply P_alias in Ha as [Ha | [-> Hu]];
      [ exact (sources_alias _ _ _ HS _ _ _ Hs Ha1 Ha Ha2)
      | contradiction ].
  - (* escaping_field_rule *)
    rename H into Hau, H0 into Hc, H1 into Hl.
    destruct Hc as [Hc | Hc];
      [ exact (escaping_field_rule _ _ _ HS _ _ _ Hau Hc Hl) | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (-> & _ & _ & _ & He).
    exact (He Hau Hl).
  - (* reading_field_rule *)
    rename H into Has, H0 into Hac, H1 into Hl.
    apply P_any_source in Has as [Has | Has];
      [ exact (reading_field_rule _ _ _ HS _ _ _ Has Hac Hl)
      | rewrite (Hacc _ _ _ Hac Has) in Hl; discriminate Hl ].
  - (* field_used_any_usage *)
    rename H into Hc.
    destruct Hc as [Hc | Hc];
      [ eapply (field_used_any_usage _ _ _ HS); eassumption | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    eapply (field_used_any_usage _ _ _ HS _ _ y Hy); eassumption.
  - (* field_used_usage *)
    rename H into Hc.
    destruct Hc as [Hc | Hc];
      [ eapply (field_used_usage _ _ _ HS); eassumption | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    eapply (field_used_usage _ _ _ HS _ _ y _ _ Hy); eassumption.
  - (* field_used_reading *)
    rename H into Hc.
    destruct Hc as [Hc | Hc];
      [ eapply (field_used_reading _ _ _ HS); eassumption | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    eapply (field_used_reading _ _ _ HS _ _ y _ Hy); eassumption.
  - (* field_used_local_source *)
    rename H into Hc.
    destruct Hc as [Hc | Hc];
      [ eapply (field_used_local_source _ _ _ HS); eassumption | ].
    destruct (poison_constructor_spec _ _ _ Hc) as (_ & (y & Hy) & _).
    eapply (field_used_local_source _ _ _ HS _ _ _ y _ Hy); eassumption.
Qed.

Lemma poison_local_reads_used :
  local_reads_used is_local G S -> local_reads_used is_local poison_graph P.
Proof.
  intros H base relation to_ Has Hac Hl. cbn [poison_graph g_accessor] in Hac.
  rewrite P_has_usage.
  apply P_any_source in Has as [Has | Has];
    [ exact (H _ _ _ Has Hac Hl)
    | rewrite (Hacc _ _ _ Hac Has) in Hl; discriminate Hl ].
Qed.

End Poison.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

Section Unit.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

Local Abbreviation P := (solution_of_tables (poisoned_tables F T)).
Local Abbreviation U := (unused_nodes F T).

Lemma unit_poison_ok_spec :
  unit_poison_ok is_local F T = true ->
  s_any_usage (solution_of_tables T) Node_all_constants = true
  ∧ (∀ c t f,
       (c, t, f) ∈ f_alias_if_any_source F -> c ∈ U -> s_alias P t f = true)
  ∧ (∀ f i b,
       (f, i, b) ∈ f_argument F -> f ∈ U -> s_has_usage P b = true)
  ∧ (∀ f i b,
       (f, i, b) ∈ f_argument F -> b ∈ U -> s_any_usage P f = true)
  ∧ (∀ b relation f,
       (b, relation, f) ∈ f_constructor F -> f ∈ U ->
       s_has_source P b = true)
  ∧ (∀ t relation b,
       (t, relation, b) ∈ f_accessor F -> b ∈ U ->
       is_local relation = false).
Proof.
  unfold unit_poison_ok. intros H.
  apply andb_true_iff in H as [H Hac]. apply andb_true_iff in H as [H Hc].
  apply andb_true_iff in H as [H Har]. apply andb_true_iff in H as [Hall Ha].
  rewrite forallb_forall in Ha, Har, Hc, Hac.
  split; [ exact Hall | ]. split; [ | split; [ | split; [ | split ] ] ].
  - intros c t f Hin Hu. specialize (Ha _ Hin). cbv beta iota in Ha.
    rewrite (proj2 (mem_node_In _ _) Hu) in Ha. exact Ha.
  - intros f i b Hin Hu. specialize (Har _ Hin). cbv beta iota in Har.
    rewrite (proj2 (mem_node_In _ _) Hu) in Har.
    apply andb_true_iff in Har as [Har _]. exact Har.
  - intros f i b Hin Hu. specialize (Har _ Hin). cbv beta iota in Har.
    rewrite (proj2 (mem_node_In _ _) Hu) in Har.
    apply andb_true_iff in Har as [_ Har]. exact Har.
  - intros b relation f Hin Hu. specialize (Hc _ Hin). cbv beta iota in Hc.
    rewrite (proj2 (mem_node_In _ _) Hu) in Hc. exact Hc.
  - intros t relation b Hin Hu. specialize (Hac _ Hin). cbv beta iota in Hac.
    rewrite (proj2 (mem_node_In _ _) Hu) in Hac. cbn [andb negb] in Hac.
    destruct (is_local relation); [ discriminate Hac | reflexivity ].
Qed.

End Unit.

Lemma inb_In {A : Type} (eq_dec : ∀ a b : A, {a = b} + {a <> b}) (a : A)
    (l : list A) :
  inb eq_dec a l = true -> a ∈ l.
Proof.
  unfold inb. destruct (in_dec eq_dec a l); [ auto | discriminate ].
Qed.

Section Renamed.

Variable is_local : field -> bool.
Variable r : list (node * node).
Variables F1 F : facts.
Variable T : tables.

Local Abbreviation rn := (renamed r).
Local Abbreviation FF := (rename_facts r F1).

Lemma in_rename2 (l : list (node * node)) (a b : node) :
  (a, b) ∈ l -> (rn a, rn b) ∈ map (fun '(a, b) => (rn a, rn b)) l.
Proof. intros H. exact (in_map (fun '(a, b) => (rn a, rn b)) _ (_, _) H). Qed.

Lemma in_rename3 {A : Type} (l : list (node * A * node)) (a : node) (x : A)
    (b : node) :
  (a, x, b) ∈ l -> (rn a, x, rn b) ∈ map (fun '(a, x, b) => (rn a, x, rn b)) l.
Proof.
  intros H. exact (in_map (fun '(a, x, b) => (rn a, x, rn b)) _ (_, _, _) H).
Qed.

Lemma in_rename_triple (l : list (node * node * node)) (a b c : node) :
  (a, b, c) ∈ l ->
  (rn a, rn b, rn c) ∈ map (fun '(a, b, c) => (rn a, rn b, rn c)) l.
Proof.
  intros H. exact (in_map (fun '(a, b, c) => (rn a, rn b, rn c)) _ (_, _, _) H).
Qed.

(* The facts of [F1], renamed, are in the graph on which the unit's tables
   with poison are closed. *)
Lemma facts_poison_graph :
  facts_poison_ok is_local F T FF = true ->
  graph_renamed r (graph_of_facts F1) (poison_graph is_local F T).
Proof.
  unfold facts_poison_ok. intros H.
  apply andb_true_iff in H as [H Hz]. apply andb_true_iff in H as [H Hasu].
  apply andb_true_iff in H as [H Has]. apply andb_true_iff in H as [H Hau].
  apply andb_true_iff in H as [H Haif]. apply andb_true_iff in H as [H Hpr].
  apply andb_true_iff in H as [H Hp]. apply andb_true_iff in H as [H Har].
  apply andb_true_iff in H as [H Hc]. apply andb_true_iff in H as [H Hac].
  apply andb_true_iff in H as [Ha Hu].
  rewrite forallb_forall in Ha, Hu, Hc, Har, Hpr, Haif, Hau, Hasu.
  cbn [graph_of_facts poison_graph g_alias g_use g_accessor g_constructor
       g_argument g_parameter g_propagate g_alias_if_any_source g_any_usage
       g_any_source g_any_source_if_used g_zero_alloc_source].
  repeat split; intros.
  - specialize (Ha _ (in_rename2 _ _ _ H)). cbv beta iota in Ha.
    apply orb_true_iff in Ha as [Ha | Ha];
      [ left; apply mem_pair_In; exact Ha | right; exact Ha ].
  - specialize (Hu _ (in_rename2 _ _ _ H)). cbv beta iota in Hu.
    apply orb_true_iff in Hu as [Hu | Hu];
      [ left; apply mem_pair_In; exact Hu | right ].
    apply andb_true_iff in Hu as [H1 H2]. split; intros H3.
    + rewrite H3 in H1. exact H1.
    + rewrite H3 in H2. exact H2.
  - exact (list_within_In _ _ _ Hac _ (in_rename3 _ _ _ _ H)).
  - specialize (Hc _ (in_rename3 _ _ _ _ H)). cbv beta iota in Hc.
    apply orb_true_iff in Hc as [Hc | Hc];
      [ left; exact (inb_In _ _ _ Hc) | right; exact Hc ].
  - specialize (Har _ (in_rename3 _ _ _ _ H)). cbv beta iota in Har.
    apply orb_true_iff in Har as [Har | Har];
      [ left; exact (inb_In _ _ _ Har) | right; exact Har ].
  - exact (list_within_In _ _ _ Hp _ (in_rename3 _ _ _ _ H)).
  - specialize (Hpr _ (in_rename_triple _ _ _ _ H)). cbv beta iota in Hpr.
    apply orb_true_iff in Hpr as [Hpr | Hpr];
      [ apply orb_true_iff in Hpr as [Hpr | Hpr];
        [ left; exact (inb_In _ _ _ Hpr)
        | right; intros H'; rewrite H' in *; discriminate ]
      | right; intros _; exact Hpr ].
  - specialize (Haif _ (in_rename_triple _ _ _ _ H)). cbv beta iota in Haif.
    apply orb_true_iff in Haif as [Haif | Haif];
      [ apply orb_true_iff in Haif as [Haif | Haif];
        [ left; exact (inb_In _ _ _ Haif)
        | right; intros H'; rewrite H' in *; discriminate ]
      | right; intros _; exact Haif ].
  - specialize (Hau _ (in_map rn _ _ H)).
    apply orb_true_iff in Hau as [Hau | Hau];
      [ left; apply mem_node_In; exact Hau | right; exact Hau ].
  - exact (list_within_In _ _ _ Has _ (in_map rn _ _ H)).
  - specialize (Hasu _ (in_rename2 _ _ _ H)). cbv beta iota in Hasu.
    apply orb_true_iff in Hasu as [Hasu | Hasu];
      [ apply orb_true_iff in Hasu as [Hasu | Hasu];
        [ left; apply mem_pair_In; exact Hasu
        | right; intros H'; rewrite H' in *; discriminate ]
      | right; intros _; exact Hasu ].
  - exact (list_within_In _ _ _ Hz _ (in_map rn _ _ H)).
Qed.

End Renamed.

(* ======================================================================== *)
(* The reach of the unit's tables with poison, at nodes with a use          *)
(* ======================================================================== *)

Section Used.

Variable F : facts.
Variable T : tables.

Local Abbreviation T' := (poisoned_tables F T).

Lemma not_unused (n : node) :
  mem_node n (t_has_usage T) = true -> ~ n ∈ unused_nodes F T.
Proof.
  intros Hu Hn. unfold unused_nodes in Hn. apply filter_In in Hn as [_ Hn].
  rewrite Hu in Hn. discriminate Hn.
Qed.

Lemma any_source_used (n : node) :
  mem_node n (t_has_usage T) = true ->
  mem_node n (t_any_source T') = true -> mem_node n (t_any_source T) = true.
Proof.
  intros Hu H. cbn [poisoned_tables t_any_source] in H.
  apply mem_node_In, in_app_iff in H as [H | H];
    [ apply mem_node_In; exact H | exfalso; exact (not_unused n Hu H) ].
Qed.

Lemma flow_reach_used (f : variable) (id : code_id) :
  mem_node (Node_variable f) (t_has_usage T) = true ->
  flow_reach F T' f id = true -> flow_reach F T f id = true.
Proof.
  unfold flow_reach. intros Hu H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (any_source_used _ Hu H) | right; exact H ].
Qed.

Lemma flow_reach2_used (b : bool) (f : variable) (id : code_id) :
  mem_node (Node_variable f) (t_has_usage T) = true ->
  flow_reach2 F T' b f id = true -> flow_reach2 F T b f id = true.
Proof.
  intros Hu. destruct b; cbn [flow_reach2];
    [ exact (flow_reach_used f id Hu) | ].
  unfold flow_reach_u. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (flow_reach_used f id Hu H) | right; exact H ].
Qed.

Lemma over_reach_used (id : code_id) :
  over_reach F T' id = true -> over_reach F T id = true.
Proof. exact (fun H => H). Qed.

Lemma node_reach_used (n : node) (id : code_id) :
  mem_node n (t_has_usage T) = true ->
  node_reach F T' n id = true -> node_reach F T n id = true.
Proof.
  unfold node_reach. intros Hu H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (any_source_used _ Hu H) | right; exact H ].
Qed.

Lemma rem_reach_used (id : code_id) :
  rem_reach F T' id = true -> rem_reach F T id = true.
Proof.
  unfold rem_reach. intros H.
  apply existsb_exists in H as (x & Hin & Hp). apply existsb_exists.
  exists x. split; [ exact Hin | ]. destruct x as [[w relation] c].
  apply andb_true_iff in Hp as [Hp Hx]. rewrite Hp. cbn [andb].
  apply existsb_exists in Hx as (y & Hin' & Hq). apply existsb_exists.
  exists y. split; [ exact Hin' | ]. destruct y as [[w' relation'] n].
  apply andb_true_iff in Hq as [Hq Hn]. apply andb_true_iff in Hq as [Hq Hu].
  change (mem_node n (t_has_usage T) = true) in Hu.
  rewrite Hq, Hu, (node_reach_used n id Hu Hn). reflexivity.
Qed.

End Used.

(* ======================================================================== *)
(* The tables of the first rebuilt unit                                     *)
(* ======================================================================== *)

Theorem analysis_within_closed : analysis_within_closed_statement.
Proof.
  intros is_local r F1 F T Hw Hc Hl.
  unfold analysis_within in Hw. apply andb_true_iff in Hw as [Hu Hf].
  destruct (unit_poison_ok_spec is_local F T Hu)
    as (Hall & Haias & Harf & Harb & Hcons & Hacc).
  pose proof (facts_poison_graph is_local r F1 F T Hf) as Hg.
  pose proof (pull_tables_pulls r (poisoned_tables F T)) as Hp.
  pose proof (check_closed_sound is_local _ _ Hc) as HS.
  split.
  - exact (closed_renamed r is_local _ _ _ _ Hp Hg
             (poison_closed is_local F T HS Hall Haias Harf Harb Hcons Hacc)).
  - exact (local_reads_used_renamed r is_local _ _ _ _ Hp Hg
             (poison_local_reads_used is_local F T Hacc
                (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                   is_local _ _ Hl))).
Qed.

Theorem analysis_within_used_reach_covered :
  analysis_within_used_reach_covered_statement.
Proof.
  intros is_local r F1 F T e Hok Hw.
  unfold analysis_within in Hw. apply andb_true_iff in Hw as [_ Hf].
  pose proof (facts_poison_graph is_local r F1 F T Hf) as Hg.
  destruct Hg as (_ & _ & _ & Hc & _ & _ & _ & _ & _ & Has & _).
  cbn [graph_of_facts poison_graph g_constructor g_any_source] in Hc, Has.
  assert (Hc' : ∀ a f b, is_real_field f = false ->
                (a, f, b) ∈ f_constructor F1 ->
                (renamed r a, f, renamed r b) ∈ f_constructor F).
  { intros a f b Hf' Hin. destruct (Hc a f b Hin) as [H | H]; [ exact H | ].
    unfold poison_constructor_ok in H. rewrite Hf' in H.
    rewrite !andb_false_r, andb_false_l in H. discriminate H. }
  set (T' := poisoned_tables F T).
  assert (Tas : ∀ a, a ∈ t_any_source (pull_tables r T') ->
                     renamed r a ∈ t_any_source T')
    by (intros a; apply in_pull_node).
  assert (Tso : ∀ a b, (a, b) ∈ t_sources (pull_tables r T') ->
                       (renamed r a, renamed r b) ∈ t_sources T')
    by (intros a b; apply in_pull_pair).
  assert (Thu : ∀ a, a ∈ t_has_usage (pull_tables r T') ->
                     renamed r a ∈ t_has_usage T')
    by (intros a; apply in_pull_node).
  pose proof (fun id => renamed_code_id r e id Hok) as Hcid.
  apply used_reach_covered_of.
  - intros b f id Hf' Hu H.
    apply (flow_reach2_used F T b f id Hu).
    exact (flow_reach2_ren r F1 F _ T' Hc' Has Hcid Tas Tso b f id
             (renamed_variable r e f Hok Hf') H).
  - intros id H. apply (over_reach_used F T id).
    exact (over_reach_ren r F1 F _ T' Hc' Hcid Thu id H).
  - intros id H. apply (rem_reach_used F T id).
    exact (rem_reach_ren r F1 F _ T' Hc' Has Hcid Tas Tso Thu id H).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem analysis_within_flow3_refines_cc :
  analysis_within_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hok Hw Hcalls Hcheck.
  destruct (analysis_within_closed is_local r _ _ T Hw Hclosed Hlrs)
    as [Hc1 Hlr1].
  exact (tables_flow3_refines_cc flag is_local u e T _ cannot_change
           dominated to_unbox pdc nr Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1
           Hcalls (analysis_within_used_reach_covered is_local r _ _ T e Hok
              Hw) Hcheck).
Qed.

Theorem analysis_within_flow3_observably_local :
  analysis_within_flow3_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (analysis_within_flow3_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc nr r Hslots Hfrag Hclosed Hlrs Hfrag1
              Hok Hw Hcalls Hcheck)).
Qed.

Theorem analysis_within_flow3_refines_observably :
  analysis_within_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (analysis_within_flow3_observably_local flag is_local u e T
           cannot_change
           dominated to_unbox pdc nr r Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hok Hw Hcalls Hcheck).
Qed.

Theorem analysis_within_never_refines_cc :
  analysis_within_never_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hok Hw Hcalls Hcheck.
  destruct (analysis_within_closed is_local r _ _ T Hw Hclosed Hlrs)
    as [Hc1 Hlr1].
  exact (tables_never_refines_cc flag is_local u e T _ cannot_change
           dominated to_unbox pdc Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1
           Hcalls (analysis_within_used_reach_covered is_local r _ _ T e Hok
              Hw) Hcheck).
Qed.

Theorem analysis_within_never_observably_local :
  analysis_within_never_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (analysis_within_never_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc r Hslots Hfrag Hclosed Hlrs Hfrag1
              Hok Hw Hcalls Hcheck)).
Qed.

Theorem analysis_within_never_refines_observably :
  analysis_within_never_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (analysis_within_never_observably_local flag is_local u e T
           cannot_change
           dominated to_unbox pdc r Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hok Hw Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Lemma renamed_all_constants (r : list (node * node)) (e : expr) :
  renaming_ok r e = true ->
  renamed r Node_all_constants = Node_all_constants.
Proof.
  intros Hok. unfold renamed.
  destruct (find (fun p => node_eqb (fst p) Node_all_constants) r)
    as [[a b] | ] eqn:E; [ | reflexivity ].
  apply find_some in E as [Hin Heq]. cbn [fst] in Heq.
  apply node_eqb_eq in Heq. subst a.
  unfold renaming_ok in Hok. rewrite forallb_forall in Hok.
  specialize (Hok _ Hin). discriminate Hok.
Qed.

(* Facts of [F1] and those of [F1] renamed, by relation. *)
Lemma in_rename_argument (r : list (node * node)) (F1 : facts) (a : node)
    (i : nat) (b : node) :
  (a, i, b) ∈ f_argument F1 ->
  (renamed r a, i, renamed r b) ∈ f_argument (rename_facts r F1).
Proof. intros H. exact (in_rename3 r _ _ _ _ H). Qed.

Lemma in_rename_alias_if_any_source (r : list (node * node)) (F1 : facts)
    (a b c : node) :
  (a, b, c) ∈ f_alias_if_any_source F1 ->
  (renamed r a, renamed r b, renamed r c)
  ∈ f_alias_if_any_source (rename_facts r F1).
Proof. intros H. exact (in_rename_triple r _ _ _ _ H). Qed.

Lemma facts_within_renamed_lists (r : list (node * node)) (F1 F : facts) :
  facts_within_renamed r F1 F = true -> lists_within (rename_facts r F1) F.
Proof. exact (facts_within_lists _ _). Qed.

Theorem poison_example : poison_example_statement.
Proof.
  split; [ | split; [ vm_compute; reflexivity | ] ].
  - intros r Hok. destruct (facts_within_renamed r _ _) eqn:Hw;
      [ exfalso | reflexivity ].
    apply facts_within_renamed_lists in Hw.
    assert (Hp : (Node_all_constants, 0, Node_made 15)
                 ∈ f_argument PoisonReach.facts1)
      by (apply (inb_In node_nat_node_eq_dec); vm_compute; reflexivity).
    pose proof (lw_argument _ _ Hw _ (in_rename_argument r _ _ _ _ Hp))
      as Hin.
    rewrite (renamed_all_constants r _ Hok) in Hin.
    assert (Hno :
      forallb (fun '(a, _, _) => negb (node_eqb a Node_all_constants))
        (f_argument PoisonReach.facts)
      = true)
      by (vm_compute; reflexivity).
    rewrite forallb_forall in Hno. specialize (Hno _ Hin).
    cbv beta iota in Hno. rewrite node_eqb_refl in Hno. discriminate Hno.
  - split; [ vm_compute; reflexivity | ].
    (* The facts as the statement has them. *)
    assert (Hf : PoisonReach.facts
                 = traverse_unit Auto UE.interface PoisonReach.unit_expr)
      by reflexivity.
    rewrite Hf.
    apply (analysis_within_flow3_refines_observably Auto UE.no_local
             UE.interface PoisonReach.unit_expr PoisonReach.tables
             PoisonReach.cannot_change [] [] true (fun _ => false) []);
      [ intros f Hf'; discriminate Hf' | left; intros f; reflexivity
      | vm_compute; reflexivity .. ].
Qed.

Theorem direct_example : direct_example_statement.
Proof.
  split; [ | split; [ vm_compute; reflexivity | ] ].
  - intros r. destruct (facts_within_renamed r _ _) eqn:Hw;
      [ exfalso | reflexivity ].
    apply facts_within_renamed_lists in Hw.
    assert (Hp :
      (Node_variable 60%positive, Node_made 11, Node_variable 60%positive)
      ∈ f_alias_if_any_source DirectExample.facts1)
      by (apply (inb_In node_triple_eq_dec); vm_compute; reflexivity).
    pose proof (lw_alias_if_any_source _ _ Hw _
                  (in_rename_alias_if_any_source r _ _ _ _ Hp))
      as Hin.
    assert (Hnil : f_alias_if_any_source DirectExample.facts = [])
      by (vm_compute; reflexivity).
    rewrite Hnil in Hin. exact Hin.
  - split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    (* The facts and the code deps as the statement has them. *)
    assert (Hf : DirectExample.facts
                 = traverse_unit Auto UE.interface DirectExample.unit_expr)
      by reflexivity.
    assert (Hc : DirectExample.code_deps
                 = unit_code_deps Auto UE.interface DirectExample.unit_expr)
      by reflexivity.
    rewrite Hf, Hc.
    apply (analysis_within_flow3_refines_observably Auto UE.no_local
             UE.interface DirectExample.unit_expr DirectExample.tables
             DirectExample.cannot_change [] [] true (fun _ => false)
             DirectExample.renaming);
      [ intros f Hf'; discriminate Hf' | left; intros f; reflexivity
      | vm_compute; reflexivity .. ].
Qed.

Theorem wrapper_example : wrapper_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  (* The facts and the code deps as the statement has them. *)
  assert (Hf : FlowExample.facts
               = traverse_unit Auto UE.interface FlowExample.unit_expr)
    by reflexivity.
  assert (Hc : FlowExample.code_deps
               = unit_code_deps Auto UE.interface FlowExample.unit_expr)
    by reflexivity.
  rewrite Hf, Hc.
  apply (analysis_within_flow3_refines_observably Auto UE.no_local
           UE.interface FlowExample.unit_expr FlowExample.tables
           FlowExample.cannot_change [] [] true (fun _ => false)
           WrapperExample.renaming);
    [ intros f Hf'; discriminate Hf' | left; intros f; reflexivity
    | vm_compute; reflexivity .. ].
Qed.
