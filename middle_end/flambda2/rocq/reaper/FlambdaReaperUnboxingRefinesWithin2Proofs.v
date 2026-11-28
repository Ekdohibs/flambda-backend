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

(* The proofs of FlambdaReaperUnboxingRefinesWithin2Spec.v. Nothing here
   needs review.

   The solution of [pull_tables r T] is that of [T] at the renamed nodes
   ([pull_tables_pulls]). A solution of the rules on a graph, taken at the
   renamed nodes, is one on any graph that the renaming takes into it
   ([closed_renamed], [local_reads_used_renamed]): each rule is applied at
   the renamed nodes. The reach of renamed facts with tables whose tuples,
   renamed, are of other tables is within the reach of those
   ([flow_reach2_ren], [over_reach_ren], [rem_reach_ren], for the
   constructors of the fields that are not real and the inputs
   [any_source]), at the variables that the renaming keeps
   ([renamed_variable]). [tables_flow3_refines_cc] and
   [tables_never_refines_cc] are the statements of
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v with tables of the first
   rebuilt unit closed in the sense of the rules ([first_runs] of
   FlambdaReaperUnboxingRefinesWithinProofs.v gives the run premises). *)

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
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesUnitFlowProofs
  FlambdaReaperUnboxingRefinesWithinProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The renaming                                                             *)
(* ======================================================================== *)

Section Renaming.

Variable r : list (node * node).

Local Abbreviation rn := (renamed r).

Lemma preimage_In (m n : node) : m ∈ preimage r n <-> rn m = n.
Proof.
  unfold preimage. rewrite filter_In. split.
  - intros [_ H]. exact (node_eqb_eq _ _ H).
  - intros H. split; [ | rewrite H; apply node_eqb_refl ].
    unfold renamed in H.
    destruct (find (fun p => node_eqb (fst p) m) r) as [[a b] | ] eqn:E.
    + apply find_some in E as [Hin Heq]. cbn [fst] in Heq.
      apply node_eqb_eq in Heq. subst a. right.
      apply in_map_iff. exists (m, b). split; [ reflexivity | exact Hin ].
    + subst n. left. reflexivity.
Qed.

Lemma in_pull_node (l : list node) (a : node) :
  a ∈ flat_map (preimage r) l <-> rn a ∈ l.
Proof.
  rewrite in_flat_map. split.
  - intros (x & Hin & Ha). apply preimage_In in Ha. subst x. exact Hin.
  - intros Hin. exists (rn a). split; [ exact Hin | ].
    apply preimage_In. reflexivity.
Qed.

Lemma in_pull_pair (l : list (node * node)) (a b : node) :
  (a, b) ∈ flat_map (fun '(x, y) => list_prod (preimage r x) (preimage r y)) l
  <-> (rn a, rn b) ∈ l.
Proof.
  rewrite in_flat_map. split.
  - intros ([x y] & Hin & Hp). apply in_prod_iff in Hp as [Ha Hb].
    apply preimage_In in Ha, Hb. subst x y. exact Hin.
  - intros Hin. exists (rn a, rn b). split; [ exact Hin | ].
    apply in_prod_iff. split; apply preimage_In; reflexivity.
Qed.

Lemma in_pull_field_node (l : list (field * node)) (f : field) (a : node) :
  (f, a) ∈ flat_map (fun '(g, n) => map (pair g) (preimage r n)) l
  <-> (f, rn a) ∈ l.
Proof.
  rewrite in_flat_map. split.
  - intros ([g n] & Hin & Hp). apply in_map_iff in Hp as (m & Heq & Hm).
    injection Heq as <- <-. apply preimage_In in Hm. subst n. exact Hin.
  - intros Hin. exists (f, rn a). split; [ exact Hin | ].
    apply in_map_iff. exists a. split; [ reflexivity | ].
    apply preimage_In. reflexivity.
Qed.

Lemma in_pull_node_field (l : list (node * field)) (a : node) (f : field) :
  (a, f) ∈ flat_map (fun '(n, g) => map (fun m => (m, g)) (preimage r n)) l
  <-> (rn a, f) ∈ l.
Proof.
  rewrite in_flat_map. split.
  - intros ([n g] & Hin & Hp). apply in_map_iff in Hp as (m & Heq & Hm).
    injection Heq as <- <-. apply preimage_In in Hm. subst n. exact Hin.
  - intros Hin. exists (rn a, f). split; [ exact Hin | ].
    apply in_map_iff. exists a. split; [ reflexivity | ].
    apply preimage_In. reflexivity.
Qed.

Lemma bool_eq_iff (b b' : bool) : (b = true <-> b' = true) -> b = b'.
Proof. destruct b, b'; intros [H1 H2]; auto; symmetry; auto. Qed.

(* The solution of the tables at the renamed nodes is that of the tables
   at the nodes renamed. *)
Record pulls (S1 S : solution) : Prop := {
  pl_alias : ∀ a b, s_alias S1 a b = s_alias S (rn a) (rn b);
  pl_any_usage : ∀ a, s_any_usage S1 a = s_any_usage S (rn a);
  pl_any_source : ∀ a, s_any_source S1 a = s_any_source S (rn a);
  pl_has_usage : ∀ a, s_has_usage S1 a = s_has_usage S (rn a);
  pl_has_source : ∀ a, s_has_source S1 a = s_has_source S (rn a);
  pl_usages : ∀ a b, s_usages S1 a b = s_usages S (rn a) (rn b);
  pl_sources : ∀ a b, s_sources S1 a b = s_sources S (rn a) (rn b);
  pl_reading_field : ∀ f a,
    s_reading_field S1 f a = s_reading_field S f (rn a);
  pl_escaping_field : ∀ f a,
    s_escaping_field S1 f a = s_escaping_field S f (rn a);
  pl_field_used : ∀ a f, s_field_used S1 a f = s_field_used S (rn a) f
}.

Lemma pull_tables_pulls (T : tables) :
  pulls (solution_of_tables (pull_tables r T)) (solution_of_tables T).
Proof.
  split; intros; cbn [solution_of_tables s_alias s_any_usage s_any_source
                      s_has_usage s_has_source s_usages s_sources
                      s_reading_field s_escaping_field s_field_used
                      pull_tables t_alias t_any_usage t_any_source
                      t_has_usage t_has_source t_usages t_sources
                      t_reading_field t_escaping_field t_field_used];
    apply bool_eq_iff;
    rewrite ?mem_pair_In, ?mem_node_In, ?mem_field_node_In,
      ?mem_node_field_In;
    first [ apply in_pull_pair | apply in_pull_node
          | apply in_pull_field_node | apply in_pull_node_field ].
Qed.

(* [G1] renamed is within [G]. *)
Definition graph_renamed (G1 G : graph) : Prop :=
  (∀ a b, g_alias G1 a b -> g_alias G (rn a) (rn b))
  ∧ (∀ a b, g_use G1 a b -> g_use G (rn a) (rn b))
  ∧ (∀ a f b, g_accessor G1 a f b -> g_accessor G (rn a) f (rn b))
  ∧ (∀ a f b, g_constructor G1 a f b -> g_constructor G (rn a) f (rn b))
  ∧ (∀ a i b, g_argument G1 a i b -> g_argument G (rn a) i (rn b))
  ∧ (∀ a i b, g_parameter G1 a i b -> g_parameter G (rn a) i (rn b))
  ∧ (∀ a b c, g_propagate G1 a b c -> g_propagate G (rn a) (rn b) (rn c))
  ∧ (∀ a b c,
       g_alias_if_any_source G1 a b c ->
       g_alias_if_any_source G (rn a) (rn b) (rn c))
  ∧ (∀ a, g_any_usage G1 a -> g_any_usage G (rn a))
  ∧ (∀ a, g_any_source G1 a -> g_any_source G (rn a))
  ∧ (∀ a b, g_any_source_if_used G1 a b -> g_any_source_if_used G (rn a) (rn b))
  ∧ (∀ a, g_zero_alloc_source G1 a -> g_zero_alloc_source G (rn a)).

Lemma facts_within_renamed_graph (F1 F : facts) :
  facts_within_renamed r F1 F = true ->
  graph_renamed (graph_of_facts F1) (graph_of_facts F).
Proof.
  intros H. apply facts_within_lists in H.
  cbn [graph_of_facts g_alias g_use g_accessor g_constructor g_argument
       g_parameter g_propagate g_alias_if_any_source g_any_usage g_any_source
       g_any_source_if_used g_zero_alloc_source].
  repeat split; intros.
  - apply (lw_alias _ _ H). cbn [rename_facts f_alias].
    apply (in_map (fun '(a, b) => (rn a, rn b)) _ (_, _)). assumption.
  - apply (lw_use _ _ H). cbn [rename_facts f_use].
    apply (in_map (fun '(a, b) => (rn a, rn b)) _ (_, _)). assumption.
  - apply (lw_accessor _ _ H). cbn [rename_facts f_accessor].
    apply (in_map (fun '(a, f, b) => (rn a, f, rn b)) _ (_, _, _)).
    assumption.
  - apply (lw_constructor _ _ H). cbn [rename_facts f_constructor].
    apply (in_map (fun '(a, f, b) => (rn a, f, rn b)) _ (_, _, _)).
    assumption.
  - apply (lw_argument _ _ H). cbn [rename_facts f_argument].
    apply (in_map (fun '(a, i, b) => (rn a, i, rn b)) _ (_, _, _)).
    assumption.
  - apply (lw_parameter _ _ H). cbn [rename_facts f_parameter].
    apply (in_map (fun '(a, i, b) => (rn a, i, rn b)) _ (_, _, _)).
    assumption.
  - apply (lw_propagate _ _ H). cbn [rename_facts f_propagate].
    apply (in_map (fun '(a, b, c) => (rn a, rn b, rn c)) _ (_, _, _)).
    assumption.
  - apply (lw_alias_if_any_source _ _ H).
    cbn [rename_facts f_alias_if_any_source].
    apply (in_map (fun '(a, b, c) => (rn a, rn b, rn c)) _ (_, _, _)).
    assumption.
  - apply (lw_any_usage _ _ H). cbn [rename_facts f_any_usage].
    apply in_map. assumption.
  - apply (lw_any_source _ _ H). cbn [rename_facts f_any_source].
    apply in_map. assumption.
  - apply (lw_any_source_if_used _ _ H).
    cbn [rename_facts f_any_source_if_used].
    apply (in_map (fun '(a, b) => (rn a, rn b)) _ (_, _)). assumption.
  - apply (lw_zero_alloc_source _ _ H). cbn [rename_facts f_zero_alloc_source].
    apply in_map. assumption.
Qed.

(* Apply the rule [f] of the closed solution [H], then its premises. *)
Local Ltac use_rule H f := eapply (f _ _ _ H); eauto.

(* A solution of the rules on [G], at the renamed nodes, is one on any
   graph that the renaming takes into [G]: the rules read the graph in
   their premises only, and name no node. *)
Lemma closed_renamed (is_local : field -> bool) (G1 G : graph)
    (S1 S : solution) :
  pulls S1 S -> graph_renamed G1 G -> closed is_local G S ->
  closed is_local G1 S1.
Proof.
  intros Hp (Ha & Hu & Hac & Hc & Har & Hpa & Hpr & Haif & Hau & Has & Hasu
             & Hz) H.
  destruct Hp as [Pal Pau Pas Phu Phs Pus Pso Prf Pef Pfu].
  split; intros;
    rewrite ?Pal, ?Pau, ?Pas, ?Phu, ?Phs, ?Pus, ?Pso, ?Prf, ?Pef, ?Pfu in *;
    [ use_rule H alias_input
    | use_rule H alias_propagate
    | use_rule H alias_if_any_source
    | use_rule H alias_constructor_usage
    | use_rule H alias_local_constructor_usage
    | use_rule H alias_parameter_usage
    | use_rule H alias_accessor_source
    | use_rule H alias_local_accessor_source
    | use_rule H alias_escaping_reading
    | use_rule H alias_argument_source
    | use_rule H has_usage_any
    | use_rule H has_usage_alias
    | use_rule H has_usage_accessor
    | use_rule H has_usage_argument
    | use_rule H has_source_any
    | use_rule H has_source_alias
    | use_rule H has_source_constructor
    | use_rule H has_source_function_slot
    | use_rule H has_source_parameter
    | use_rule H any_usage_use
    | use_rule H any_usage_alias
    | use_rule H any_usage_constructor
    | use_rule H any_usage_argument
    | use_rule H any_source_input
    | use_rule H any_source_zero_alloc
    | use_rule H any_source_if_used
    | use_rule H any_source_alias
    | use_rule H any_source_parameter
    | use_rule H any_source_accessor
    | use_rule H any_source_use
    | use_rule H any_usage_input
    | use_rule H usages_accessor
    | use_rule H usages_argument
    | use_rule H usages_alias
    | use_rule H sources_constructor
    | use_rule H sources_parameter
    | use_rule H sources_alias
    | use_rule H escaping_field_rule
    | use_rule H reading_field_rule
    | use_rule H field_used_any_usage
    | use_rule H field_used_usage
    | use_rule H field_used_reading
    | use_rule H field_used_local_source ].
Qed.

Lemma local_reads_used_renamed (is_local : field -> bool) (G1 G : graph)
    (S1 S : solution) :
  pulls S1 S -> graph_renamed G1 G -> local_reads_used is_local G S ->
  local_reads_used is_local G1 S1.
Proof.
  intros Hp (_ & _ & Hac & _) H base relation to_ Hs Hacc Hl.
  rewrite (pl_any_source _ _ Hp) in Hs. rewrite (pl_has_usage _ _ Hp).
  exact (H _ _ _ Hs (Hac _ _ _ Hacc) Hl).
Qed.

End Renaming.

(* ======================================================================== *)
(* The reach reads no more off renamed facts                                *)
(* ======================================================================== *)

Section Reach.

Variable r : list (node * node).
Local Abbreviation rn := (renamed r).

Variables F1 F : facts.
Variables T1 T : tables.
(* The constructors of the fields that are not real (those that the reach
   reads), and the inputs [any_source]. *)
Hypothesis Hcons : ∀ a f b,
  is_real_field f = false -> (a, f, b) ∈ f_constructor F1 ->
  (rn a, f, rn b) ∈ f_constructor F.
Hypothesis Hanys : ∀ a, a ∈ f_any_source F1 -> rn a ∈ f_any_source F.
Hypothesis Hcid : ∀ id, rn (Node_code_id id) = Node_code_id id.
Hypothesis Tas : ∀ a, a ∈ t_any_source T1 -> rn a ∈ t_any_source T.
Hypothesis Tso : ∀ a b,
  (a, b) ∈ t_sources T1 -> (rn a, rn b) ∈ t_sources T.
Hypothesis Thu : ∀ a, a ∈ t_has_usage T1 -> rn a ∈ t_has_usage T.

Lemma cons_ren (a : node) (f : field) (b : node) :
  (a, f, b) ∈ f_constructor F1 -> is_real_field f = false ->
  (rn a, f, rn b) ∈ f_constructor F.
Proof. intros Hin Hf. exact (Hcons a f b Hf Hin). Qed.

(* The witness of code [id] of [w], renamed. *)
Lemma code_of_witness_ren (w : node) (id : code_id) :
  existsb
    (fun '(w', relation', c) =>
       node_eqb w' w && field_eqb relation' Field_code_id_of_call_witness
       && node_eqb c (Node_code_id id))
    (f_constructor F1)
  = true ->
  existsb
    (fun '(w', relation', c) =>
       node_eqb w' (rn w) && field_eqb relation' Field_code_id_of_call_witness
       && node_eqb c (Node_code_id id))
    (f_constructor F)
  = true.
Proof.
  intros H. apply existsb_exists in H as ([[w' rel'] c] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hc]. apply andb_true_iff in Hp as [Hw Hr].
  apply node_eqb_eq in Hw, Hc. apply field_eqb_eq in Hr. subst w' rel' c.
  apply existsb_exists.
  exists (rn w, Field_code_id_of_call_witness, Node_code_id id).
  split; [ rewrite <- (Hcid id); exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite !node_eqb_refl, field_eqb_refl. reflexivity.
Qed.

Lemma alloc_has_code_ren (a : node) (id : code_id) :
  alloc_has_code F1 a id = true -> alloc_has_code F (rn a) id = true.
Proof.
  unfold alloc_has_code. intros H.
  apply existsb_exists in H as ([[b rel] w] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hw]. apply andb_true_iff in Hp as [Hb Hr].
  apply node_eqb_eq in Hb. apply field_eqb_eq in Hr. subst b rel.
  apply existsb_exists.
  exists (rn a, Field_call_witness Known_arity_code_pointer, rn w).
  split; [ exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
  apply orb_true_iff in Hw as [Hw | Hw]; apply orb_true_iff.
  - left. apply mem_node_In. apply Hanys. apply mem_node_In. exact Hw.
  - right. exact (code_of_witness_ren w id Hw).
Qed.

Lemma alloc_unknown_code_ren (a : node) (id : code_id) :
  alloc_unknown_code F1 a id = true -> alloc_unknown_code F (rn a) id = true.
Proof.
  unfold alloc_unknown_code. intros H.
  apply existsb_exists in H as ([[b rel] w] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hw]. apply andb_true_iff in Hp as [Hb Hr].
  apply node_eqb_eq in Hb. apply field_eqb_eq in Hr. subst b rel.
  apply existsb_exists.
  exists (rn a, Field_call_witness Unknown_arity_code_pointer, rn w).
  split; [ exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
  exact (code_of_witness_ren w id Hw).
Qed.

(* A source of [x] with a property [P], renamed. *)
Lemma sources_ren (x : node) (P : node -> bool) (Q : node -> bool) :
  (∀ a, P a = true -> Q (rn a) = true) ->
  existsb (fun '(y, a) => node_eqb y x && P a) (t_sources T1) = true ->
  existsb (fun '(y, a) => node_eqb y (rn x) && Q a) (t_sources T) = true.
Proof.
  intros HPQ H. apply existsb_exists in H as ([y a] & Hin & Hp).
  apply andb_true_iff in Hp as [Hy Ha]. apply node_eqb_eq in Hy. subst y.
  apply existsb_exists. exists (rn x, rn a).
  split; [ exact (Tso _ _ Hin) | ].
  rewrite node_eqb_refl. exact (HPQ a Ha).
Qed.

Lemma any_source_ren (x : node) :
  mem_node x (t_any_source T1) = true ->
  mem_node (rn x) (t_any_source T) = true.
Proof. rewrite !mem_node_In. exact (Tas x). Qed.

Lemma flow_reach_ren (f : variable) (id : code_id) :
  rn (Node_variable f) = Node_variable f ->
  flow_reach F1 T1 f id = true -> flow_reach F T f id = true.
Proof.
  unfold flow_reach. intros Hf H. rewrite <- Hf.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (any_source_ren _ H) | right ].
  exact (sources_ren _ _ _ (fun a => alloc_has_code_ren a id) H).
Qed.

Lemma flow_reach2_ren (b : bool) (f : variable) (id : code_id) :
  rn (Node_variable f) = Node_variable f ->
  flow_reach2 F1 T1 b f id = true -> flow_reach2 F T b f id = true.
Proof.
  intros Hf.
  destruct b; cbn [flow_reach2]; [ exact (flow_reach_ren f id Hf) | ].
  unfold flow_reach_u. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (flow_reach_ren f id Hf H) | right ].
  rewrite <- Hf.
  exact (sources_ren _ _ _ (fun a => alloc_unknown_code_ren a id) H).
Qed.

(* The first result, with a use, of a witness of code [id] of [w]. *)
Lemma result_ren (w : node) (P : node -> bool) (Q : node -> bool) :
  (∀ x, P x = true -> Q (rn x) = true) ->
  existsb
    (fun '(w', relation', x) =>
       node_eqb w' w && field_eqb relation' (Field_return_of_call (Normal 0))
       && P x)
    (f_constructor F1)
  = true ->
  existsb
    (fun '(w', relation', x) =>
       node_eqb w' (rn w)
       && field_eqb relation' (Field_return_of_call (Normal 0)) && Q x)
    (f_constructor F)
  = true.
Proof.
  intros HPQ H. apply existsb_exists in H as ([[w' rel'] x] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hx]. apply andb_true_iff in Hp as [Hw Hr].
  apply node_eqb_eq in Hw. apply field_eqb_eq in Hr. subst w' rel'.
  apply existsb_exists. exists (rn w, Field_return_of_call (Normal 0), rn x).
  split; [ exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite node_eqb_refl, field_eqb_refl. exact (HPQ x Hx).
Qed.

Lemma has_usage_ren (x : node) :
  mem_node x (t_has_usage T1) = true ->
  mem_node (rn x) (t_has_usage T) = true.
Proof. rewrite !mem_node_In. exact (Thu x). Qed.

Lemma over_reach_ren (id : code_id) :
  over_reach F1 T1 id = true -> over_reach F T id = true.
Proof.
  unfold over_reach. intros H.
  apply existsb_exists in H as ([[w rel] c] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hx]. apply andb_true_iff in Hp as [Hr Hc].
  apply field_eqb_eq in Hr. apply node_eqb_eq in Hc. subst rel c.
  apply existsb_exists.
  exists (rn w, Field_code_id_of_call_witness, Node_code_id id).
  split; [ rewrite <- (Hcid id); exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite node_eqb_refl, field_eqb_refl. cbn [andb].
  exact (result_ren w _ _ has_usage_ren Hx).
Qed.

Lemma node_reach_ren (n : node) (id : code_id) :
  node_reach F1 T1 n id = true -> node_reach F T (rn n) id = true.
Proof.
  unfold node_reach. intros H.
  apply orb_true_iff in H as [H | H]; apply orb_true_iff;
    [ left; exact (any_source_ren _ H) | right ].
  refine (sources_ren _ _ _ _ H).
  intros a Ha. apply orb_true_iff in Ha as [Ha | Ha]; apply orb_true_iff;
    [ left; exact (alloc_has_code_ren a id Ha)
    | right; exact (alloc_unknown_code_ren a id Ha) ].
Qed.

Lemma rem_reach_ren (id : code_id) :
  rem_reach F1 T1 id = true -> rem_reach F T id = true.
Proof.
  unfold rem_reach. intros H.
  apply existsb_exists in H as ([[w rel] c] & Hin & Hp).
  apply andb_true_iff in Hp as [Hp Hx]. apply andb_true_iff in Hp as [Hr Hc].
  apply field_eqb_eq in Hr. subst rel.
  destruct c as [ | | cid | | ]; try discriminate Hc.
  apply existsb_exists.
  exists (rn w, Field_code_id_of_call_witness, Node_code_id cid).
  split; [ rewrite <- (Hcid cid); exact (cons_ren _ _ _ Hin eq_refl) | ].
  rewrite field_eqb_refl. cbn [andb].
  apply existsb_exists in Hx as ([[w' rel'] x] & Hin' & Hp).
  apply andb_true_iff in Hp as [Hp Hn]. apply andb_true_iff in Hp as [Hp Hu].
  apply andb_true_iff in Hp as [Hw Hr].
  apply node_eqb_eq in Hw. apply field_eqb_eq in Hr. subst w' rel'.
  apply existsb_exists. exists (rn w, Field_return_of_call (Normal 0), rn x).
  split; [ exact (cons_ren _ _ _ Hin' eq_refl) | ].
  rewrite node_eqb_refl, field_eqb_refl, (has_usage_ren x Hu),
    (node_reach_ren x id Hn).
  reflexivity.
Qed.

End Reach.

(* ======================================================================== *)
(* The renaming keeps the names of the unit                                 *)
(* ======================================================================== *)

Lemma renamed_variable (r : list (node * node)) (e : expr) (x : variable) :
  renaming_ok r e = true -> x ∈ variables_of e ->
  renamed r (Node_variable x) = Node_variable x.
Proof.
  intros Hok Hx. unfold renamed.
  destruct (find (fun p => node_eqb (fst p) (Node_variable x)) r)
    as [[a b] | ] eqn:E; [ | reflexivity ].
  apply find_some in E as [Hin Heq]. cbn [fst] in Heq.
  apply node_eqb_eq in Heq. subst a.
  unfold renaming_ok in Hok. rewrite forallb_forall in Hok.
  specialize (Hok _ Hin). cbn beta iota in Hok.
  rewrite (proj2 (among_In _ _) Hx) in Hok. discriminate Hok.
Qed.

Lemma renamed_code_id (r : list (node * node)) (e : expr) (id : code_id) :
  renaming_ok r e = true -> renamed r (Node_code_id id) = Node_code_id id.
Proof.
  intros Hok. unfold renamed.
  destruct (find (fun p => node_eqb (fst p) (Node_code_id id)) r)
    as [[a b] | ] eqn:E; [ | reflexivity ].
  apply find_some in E as [Hin Heq]. cbn [fst] in Heq.
  apply node_eqb_eq in Heq. subst a.
  unfold renaming_ok in Hok. rewrite forallb_forall in Hok.
  specialize (Hok _ Hin). discriminate Hok.
Qed.

(* [used_reach_covered] from the comparison at the variables of [e] with
   a use and at every code. *)
Lemma used_reach_covered_of (F : facts) (T : tables) (F1 : facts)
    (T1 : tables) (e : expr) :
  (∀ (b : bool) (f : variable) (id : code_id),
     f ∈ variables_of e ->
     s_has_usage (solution_of_tables T) (Node_variable f) = true ->
     flow_reach2 F1 T1 b f id = true -> flow_reach2 F T b f id = true) ->
  (∀ id : code_id, over_reach F1 T1 id = true -> over_reach F T id = true) ->
  (∀ id : code_id, rem_reach F1 T1 id = true -> rem_reach F T id = true) ->
  used_reach_covered F T F1 T1 e = true.
Proof.
  intros Hr Ho Hm. unfold used_reach_covered. cbv zeta.
  apply andb_true_iff. split; apply forallb_forall.
  - intros f Hf.
    destruct (s_has_usage (solution_of_tables T) (Node_variable f)) eqn:Hu;
      cbn [negb orb]; [ | reflexivity ].
    apply forallb_forall. intros id _. apply forallb_forall. intros b _.
    destruct (flow_reach2 F1 T1 b f id) eqn:H; [ | reflexivity ].
    rewrite (Hr b f id Hf Hu H). reflexivity.
  - intros id _. apply andb_true_iff. split.
    + destruct (over_reach F1 T1 id) eqn:H; [ | reflexivity ].
      rewrite (Ho id H). reflexivity.
    + destruct (rem_reach F1 T1 id) eqn:H; [ | reflexivity ].
      rewrite (Hm id H). reflexivity.
Qed.

Theorem within2_closed : within2_closed_statement.
Proof.
  intros is_local r F1 F T Hw Hc Hl.
  pose proof (facts_within_renamed_graph r F1 F Hw) as Hg.
  pose proof (pull_tables_pulls r T) as Hp.
  split.
  - exact (closed_renamed r is_local _ _ _ _ Hp Hg
             (check_closed_sound is_local _ _ Hc)).
  - exact (local_reads_used_renamed r is_local _ _ _ _ Hp Hg
             (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                is_local _ _ Hl)).
Qed.

Theorem within2_used_reach_covered : within2_used_reach_covered_statement.
Proof.
  intros r F1 F T e Hok Hw.
  destruct (facts_within_renamed_graph r F1 F Hw)
    as (_ & _ & _ & Hc & _ & _ & _ & _ & _ & Has & _).
  cbn [graph_of_facts g_constructor g_any_source] in Hc, Has.
  assert (Tas : ∀ a, a ∈ t_any_source (pull_tables r T) ->
                     renamed r a ∈ t_any_source T)
    by (intros a; apply in_pull_node).
  assert (Tso : ∀ a b, (a, b) ∈ t_sources (pull_tables r T) ->
                       (renamed r a, renamed r b) ∈ t_sources T)
    by (intros a b; apply in_pull_pair).
  assert (Thu : ∀ a, a ∈ t_has_usage (pull_tables r T) ->
                     renamed r a ∈ t_has_usage T)
    by (intros a; apply in_pull_node).
  pose proof (fun a f b (_ : is_real_field f = false) => Hc a f b) as Hc'.
  pose proof (renamed_code_id r e) as Hcid.
  apply used_reach_covered_of.
  - intros b f id Hf _.
    exact (flow_reach2_ren r F1 F _ T Hc' Has (fun id => Hcid id Hok) Tas
             Tso b f id (renamed_variable r e f Hok Hf)).
  - exact (over_reach_ren r F1 F _ T Hc' (fun id => Hcid id Hok) Thu).
  - exact (rem_reach_ren r F1 F _ T Hc' Has (fun id => Hcid id Hok) Tas
             Tso Thu).
Qed.

(* ======================================================================== *)
(* From tables of the first rebuilt unit                                    *)
(* ======================================================================== *)

(* The statements of FlambdaReaperUnboxingRefinesUnitFlowSpec.v with the
   tables of the first rebuilt unit closed in the sense of the rules
   rather than of their checks. *)
Lemma tables_flow3_refines_cc (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr)
    (T T1 : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  in_heap_fragment (first_rebuild flag u e T cannot_change pdc) = true ->
  closed is_local
    (graph_of_facts
       (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)))
    (solution_of_tables T1) ->
  local_reads_used is_local
    (graph_of_facts
       (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)))
    (solution_of_tables T1) ->
  calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  used_reach_covered (traverse_unit flag u e) T
    (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)) T1 e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines (first_rebuild flag u e T cannot_change pdc)
    (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) dominated to_unbox pdc e).
Proof.
  intros Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1 Hcalls Hcov Hcheck.
  destruct (first_runs flag is_local u _ T1 Hslots Hfrag1 Hc1 Hlr1)
    as (Hfl & Hov & Hrm).
  exact (rebuild_unbox_modes_refines_cc_over3 flag is_local u e T
           cannot_change dominated to_unbox pdc nr _ _ _ Hslots Hfrag Hclosed
           (unit_flow3_covered nr _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck Hfl
           Hov Hrm).
Qed.

Lemma tables_never_refines_cc (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr)
    (T T1 : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  in_heap_fragment (first_rebuild flag u e T cannot_change pdc) = true ->
  closed is_local
    (graph_of_facts
       (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)))
    (solution_of_tables T1) ->
  local_reads_used is_local
    (graph_of_facts
       (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)))
    (solution_of_tables T1) ->
  calls_unboxing_unit_never_nr
    (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
       (unit_code_deps flag u e))
    (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
    dominated to_unbox e
  = true ->
  used_reach_covered (traverse_unit flag u e) T
    (traverse_unit flag u (first_rebuild flag u e T cannot_change pdc)) T1 e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  program_refines (first_rebuild flag u e T cannot_change pdc)
    (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
       (unit_code_deps flag u e) dominated to_unbox pdc e).
Proof.
  intros Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1 Hcalls Hcov Hcheck.
  destruct (first_runs flag is_local u _ T1 Hslots Hfrag1 Hc1 Hlr1)
    as (Hfl & Hov & Hrm).
  exact (rebuild_unbox_modes_refines_cc_never flag is_local u e T
           cannot_change dominated to_unbox pdc _ _ _ _ Hslots Hfrag Hclosed
           (unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck Hfl
           Hov Hrm).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem renamed_flow3_refines_cc : renamed_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hok Hw Hcalls Hcheck.
  destruct (within2_closed is_local r _ _ T Hw Hclosed Hlrs) as [Hc1 Hlr1].
  exact (tables_flow3_refines_cc flag is_local u e T _ cannot_change
           dominated to_unbox pdc nr Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1
           Hcalls (within2_used_reach_covered r _ _ T e Hok Hw) Hcheck).
Qed.

Theorem renamed_flow3_observably_local :
  renamed_flow3_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (renamed_flow3_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc nr r Hslots Hfrag Hclosed Hlrs Hfrag1
              Hok Hw Hcalls Hcheck)).
Qed.

Theorem renamed_flow3_refines_observably :
  renamed_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr r
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (renamed_flow3_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc nr r Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hok Hw Hcalls Hcheck).
Qed.

Theorem renamed_never_refines_cc : renamed_never_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hfrag Hclosed Hlrs Hfrag1 Hok Hw Hcalls Hcheck.
  destruct (within2_closed is_local r _ _ T Hw Hclosed Hlrs) as [Hc1 Hlr1].
  exact (tables_never_refines_cc flag is_local u e T _ cannot_change
           dominated to_unbox pdc Hslots Hfrag Hclosed Hfrag1 Hc1 Hlr1
           Hcalls (within2_used_reach_covered r _ _ T e Hok Hw) Hcheck).
Qed.

Theorem renamed_never_observably_local :
  renamed_never_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (renamed_never_refines_cc flag is_local u e T cannot_change
              dominated to_unbox pdc r Hslots Hfrag Hclosed Hlrs Hfrag1
              Hok Hw Hcalls Hcheck)).
Qed.

Theorem renamed_never_refines_observably :
  renamed_never_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc r
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hok
    Hw Hcalls Hcheck.
  exact (renamed_never_observably_local flag is_local u e T cannot_change
           dominated to_unbox pdc r Hslots Hfrag Hany Hrk Hwf Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hok Hw Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Theorem renamed_example : renamed_example_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  apply (renamed_never_refines_observably Auto UE.no_local UE.interface
           RenamedExample.unit_expr RenamedExample.tables
           RenamedExample.cannot_change [] [] true RenamedExample.renaming);
    [ intros f Hf; discriminate Hf | left; intros f; reflexivity
    | vm_compute; reflexivity .. ].
Qed.
