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

(* The proofs of FlambdaReaperCallingConventionsSpec.v: the check of
   FlambdaReaperCallingConventions.v implies what it is said to mean, and
   the table it derives is the least, and the rebuild where no code
   changes is that of FlambdaReaperRebuild.v. The refinement with changes
   of calling convention is
   FlambdaReaperRebuildProofs.rebuild_cc_refines_checked, given what the
   checks of the table and of the facts say ([check_cc_closed_sound],
   [check_apply_widgets_sound], [check_witness_code_ids_sound]); the
   facts of the traversal pass the last ([traverse_witness_code_ids]).
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaWellFormedSpec FlambdaSemanticsSpec FlambdaReaperGraphSpec
  FlambdaReaperSolutionProofs
  FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs FlambdaSemanticsProofs
  FlambdaReaperFlowSpec FlambdaReaperFlowProofs.
Import ListNotations.

Section Witnesses.

Variable F : facts.
Variable T : tables.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation witness := (cannot_change_witness_calling_convention F T).

Lemma node_eqb_false : ∀ n n' : node, n <> n' -> node_eqb n n' = false.
Proof.
  intros n n' Hne. destruct (node_eqb n n') eqn:Heq; [ | reflexivity ].
  apply node_eqb_eq in Heq. contradiction.
Qed.

(* [code_id_read_with_usage] is the existential of the two premises it
   stands for. *)
Lemma code_id_read_with_usage_iff : ∀ alias : node,
  code_id_read_with_usage F T alias = true
  <-> ∃ v : node,
        g_accessor G v Field_code_id_of_call_witness alias
        ∧ s_has_usage S v = true.
Proof.
  intros alias. unfold code_id_read_with_usage. rewrite existsb_exists.
  cbn [graph_of_facts solution_of_tables g_accessor s_has_usage].
  split.
  - intros ([[v relation] base] & Hin & Htest).
    apply andb_true_iff in Htest as [Htest Hv].
    apply andb_true_iff in Htest as [Hrelation Hbase].
    apply field_eqb_eq in Hrelation. apply node_eqb_eq in Hbase. subst.
    exists v. auto.
  - intros (v & Hin & Hv). exists (v, Field_code_id_of_call_witness, alias).
    split; [ exact Hin | ].
    rewrite field_eqb_refl, node_eqb_refl, Hv. reflexivity.
Qed.

(* [witness] holds where one of its six rules does. *)
Lemma witness_intro : ∀ x : node,
  witness_any_usage T x = true ∨ witness_other_source F T x = true
  ∨ witness_any_source_usage F T x = true
  ∨ witness_other_own_source T x = true
  ∨ witness_not_own_source T x = true ∨ witness_any_source T x = true ->
  witness x = true.
Proof.
  intros x Hrule. unfold cannot_change_witness_calling_convention.
  destruct Hrule as [H | [H | [H | [H | [H | H]]]]];
    rewrite H; rewrite ?orb_true_r; reflexivity.
Qed.

(* The witnesses as the check computes them are closed under the six
   rules that conclude [cannot_change_witness_calling_convention]: each
   rule's premises are found in the lists that its test goes through. *)
Lemma witness_any_usage_closed : ∀ x,
  s_any_usage S x = true -> witness x = true.
Proof. intros x Hx. apply witness_intro. left. exact Hx. Qed.

Lemma witness_other_source_closed : ∀ allocation_id alias alias_source v,
  s_usages S allocation_id alias = true ->
  s_sources S alias alias_source = true ->
  s_has_source S alias_source = true ->
  alias_source <> allocation_id ->
  g_accessor G v Field_code_id_of_call_witness alias ->
  s_has_usage S v = true ->
  witness allocation_id = true.
Proof.
  intros allocation_id alias alias_source v Husage Hsource Hhas Hne Hv Hused.
  apply witness_intro. right. left.
  cbn [solution_of_tables s_usages s_sources s_has_source] in *.
  unfold witness_other_source. apply existsb_exists.
  exists (allocation_id, alias). split; [ apply mem_pair_In, Husage | ].
  rewrite node_eqb_refl. cbn.
  apply andb_true_iff. split.
  - apply existsb_exists. exists (alias, alias_source).
    split; [ apply mem_pair_In, Hsource | ].
    rewrite node_eqb_refl, Hhas, node_eqb_false by exact Hne. reflexivity.
  - apply code_id_read_with_usage_iff. exists v. auto.
Qed.

Lemma witness_any_source_usage_closed : ∀ allocation_id alias v,
  s_usages S allocation_id alias = true -> s_any_source S alias = true ->
  g_accessor G v Field_code_id_of_call_witness alias ->
  s_has_usage S v = true ->
  witness allocation_id = true.
Proof.
  intros allocation_id alias v Husage Hany Hv Hused.
  apply witness_intro. right. right. left.
  cbn [solution_of_tables s_usages s_any_source] in *.
  unfold witness_any_source_usage. apply existsb_exists.
  exists (allocation_id, alias). split; [ apply mem_pair_In, Husage | ].
  rewrite node_eqb_refl, Hany. cbn.
  apply code_id_read_with_usage_iff. exists v. auto.
Qed.

Lemma witness_other_own_source_closed : ∀ allocation_id source,
  s_sources S allocation_id source = true ->
  s_has_source S source = true -> source <> allocation_id ->
  witness allocation_id = true.
Proof.
  intros allocation_id source Hsource Hhas Hne.
  apply witness_intro. right. right. right. left.
  cbn [solution_of_tables s_sources s_has_source] in *.
  unfold witness_other_own_source. apply existsb_exists.
  exists (allocation_id, source). split; [ apply mem_pair_In, Hsource | ].
  rewrite node_eqb_refl, Hhas, node_eqb_false by exact Hne. reflexivity.
Qed.

Lemma witness_not_own_source_closed : ∀ allocation_id usage,
  s_usages S allocation_id usage = true -> s_has_usage S usage = true ->
  s_sources S allocation_id allocation_id = false ->
  witness allocation_id = true.
Proof.
  intros allocation_id usage Husage Hused Hown.
  apply witness_intro. right. right. right. right. left.
  cbn [solution_of_tables s_usages s_has_usage s_sources] in *.
  unfold witness_not_own_source. rewrite Hown, andb_true_r.
  apply existsb_exists.
  exists (allocation_id, usage). split; [ apply mem_pair_In, Husage | ].
  rewrite node_eqb_refl, Hused. reflexivity.
Qed.

Lemma witness_any_source_closed : ∀ allocation_id,
  s_any_source S allocation_id = true -> witness allocation_id = true.
Proof.
  intros allocation_id Hany. apply witness_intro.
  right. right. right. right. right. exact Hany.
Qed.

(* Conversely, [witness] holds only where one of the six rules gives it:
   so it holds in every table of the witnesses that is closed. *)
Lemma witness_forced : ∀ cannot_change_witness cannot_change : node -> bool,
  cc_closed G S cannot_change_witness cannot_change ->
  ∀ x : node, witness x = true -> cannot_change_witness x = true.
Proof.
  intros ccw cc Hclosed x Hx.
  unfold cannot_change_witness_calling_convention in Hx.
  repeat rewrite orb_true_iff in Hx.
  destruct Hx as [[[[[Hx | Hx] | Hx] | Hx] | Hx] | Hx].
  - (* [any_usage] *)
    exact (cc_witness_any_usage _ _ _ _ Hclosed x Hx).
  - (* used at a call that may find another witness *)
    unfold witness_other_source in Hx. apply existsb_exists in Hx.
    destruct Hx as ([allocation_id alias] & Hin & Htest).
    apply andb_true_iff in Htest as [Htest Hread].
    apply andb_true_iff in Htest as [Hx Hsources].
    apply node_eqb_eq in Hx. subst allocation_id.
    apply existsb_exists in Hsources.
    destruct Hsources as ([alias' alias_source] & Hin' & Htest).
    apply andb_true_iff in Htest as [Htest Hne].
    apply andb_true_iff in Htest as [Halias Hhas].
    apply node_eqb_eq in Halias. subst alias'.
    apply code_id_read_with_usage_iff in Hread as (v & Hv & Hused).
    apply (cc_witness_other_source _ _ _ _ Hclosed x alias alias_source v);
      auto.
    + apply mem_pair_In. exact Hin.
    + apply mem_pair_In. exact Hin'.
    + intros Heq. subst. rewrite node_eqb_refl in Hne. discriminate.
  - (* used at a call that may find any witness *)
    unfold witness_any_source_usage in Hx. apply existsb_exists in Hx.
    destruct Hx as ([allocation_id alias] & Hin & Htest).
    apply andb_true_iff in Htest as [Htest Hread].
    apply andb_true_iff in Htest as [Hx Hany].
    apply node_eqb_eq in Hx. subst allocation_id.
    apply code_id_read_with_usage_iff in Hread as (v & Hv & Hused).
    apply (cc_witness_any_source_usage _ _ _ _ Hclosed x alias v); auto.
    apply mem_pair_In. exact Hin.
  - (* may come from another allocation *)
    unfold witness_other_own_source in Hx. apply existsb_exists in Hx.
    destruct Hx as ([allocation_id source] & Hin & Htest).
    apply andb_true_iff in Htest as [Htest Hne].
    apply andb_true_iff in Htest as [Hx Hhas].
    apply node_eqb_eq in Hx. subst allocation_id.
    apply (cc_witness_other_own_source _ _ _ _ Hclosed x source); auto.
    + apply mem_pair_In. exact Hin.
    + intros Heq. subst. rewrite node_eqb_refl in Hne. discriminate.
  - (* used, and not its own source *)
    unfold witness_not_own_source in Hx.
    apply andb_true_iff in Hx as [Hx Hown]. apply negb_true_iff in Hown.
    apply existsb_exists in Hx.
    destruct Hx as ([allocation_id usage] & Hin & Htest).
    apply andb_true_iff in Htest as [Hx Hused].
    apply node_eqb_eq in Hx. subst allocation_id.
    apply (cc_witness_not_own_source _ _ _ _ Hclosed x usage); auto.
    apply mem_pair_In. exact Hin.
  - (* [any_source] *)
    exact (cc_witness_any_source _ _ _ _ Hclosed x Hx).
Qed.

(* What [derived_by_code_id_of_witness] lists: the code ids of the used
   witnesses that cannot change. *)
Lemma in_derived_by_code_id_of_witness : ∀ code_id : node,
  code_id ∈ derived_by_code_id_of_witness F T
  <-> ∃ call_witness : node,
        g_constructor G call_witness Field_code_id_of_call_witness code_id
        ∧ s_has_usage S call_witness = true ∧ witness call_witness = true.
Proof.
  intros code_id. unfold derived_by_code_id_of_witness. rewrite in_flat_map.
  cbn [graph_of_facts solution_of_tables g_constructor s_has_usage].
  split.
  - intros ([[call_witness relation] code_id'] & Hin & Hcode).
    destruct (field_eqb relation Field_code_id_of_call_witness
              && mem_node call_witness (t_has_usage T)
              && witness call_witness) eqn:Htest;
      [ | destruct Hcode ].
    destruct Hcode as [<- | []].
    apply andb_true_iff in Htest as [Htest Hwitness].
    apply andb_true_iff in Htest as [Hrelation Hused].
    apply field_eqb_eq in Hrelation. subst relation.
    exists call_witness. auto.
  - intros (call_witness & Hin & Hused & Hwitness).
    exists (call_witness, Field_code_id_of_call_witness, code_id).
    split; [ exact Hin | ].
    rewrite field_eqb_refl, Hused, Hwitness. left. reflexivity.
Qed.

(* What [derived_by_code_id_unknown_arity] lists: the code ids of the
   witnesses of the used witnesses of unknown arity. *)
Lemma in_derived_by_code_id_unknown_arity : ∀ code_id : node,
  code_id ∈ derived_by_code_id_unknown_arity F T
  <-> ∃ set_of_closures call_witness : node,
        g_constructor G set_of_closures
          (Field_call_witness Unknown_arity_code_pointer) call_witness
        ∧ s_has_usage S call_witness = true
        ∧ g_constructor G call_witness Field_code_id_of_call_witness code_id.
Proof.
  intros code_id. unfold derived_by_code_id_unknown_arity.
  rewrite in_flat_map.
  cbn [graph_of_facts solution_of_tables g_constructor s_has_usage].
  split.
  - intros ([[set_of_closures relation] call_witness] & Hin & Hcode).
    destruct (field_eqb relation
                (Field_call_witness Unknown_arity_code_pointer)
              && mem_node call_witness (t_has_usage T)) eqn:Htest;
      [ | destruct Hcode ].
    apply andb_true_iff in Htest as [Hrelation Hused].
    apply field_eqb_eq in Hrelation. subst relation.
    unfold code_ids_of_witness in Hcode. apply in_flat_map in Hcode.
    destruct Hcode as ([[call_witness' relation'] code_id'] & Hin' & Hcode).
    destruct (node_eqb call_witness call_witness'
              && field_eqb relation' Field_code_id_of_call_witness)
      eqn:Htest; [ | destruct Hcode ].
    destruct Hcode as [<- | []].
    apply andb_true_iff in Htest as [Hwitness Hrelation].
    apply node_eqb_eq in Hwitness. apply field_eqb_eq in Hrelation.
    subst call_witness' relation'.
    exists set_of_closures, call_witness. auto.
  - intros (set_of_closures & call_witness & Hin & Hused & Hcode).
    exists (set_of_closures, Field_call_witness Unknown_arity_code_pointer,
            call_witness).
    split; [ exact Hin | ].
    rewrite field_eqb_refl, Hused. cbn.
    unfold code_ids_of_witness. apply in_flat_map.
    exists (call_witness, Field_code_id_of_call_witness, code_id).
    split; [ exact Hcode | ].
    rewrite node_eqb_refl, field_eqb_refl. left. reflexivity.
Qed.

End Witnesses.

(* A table that passes the check is closed under the eight rules, with
   the witnesses that the check computes. *)
Theorem check_cc_closed_sound : check_cc_closed_sound_statement.
Proof.
  intros F T cannot_change Hcheck.
  exists (cannot_change_witness_calling_convention F T).
  unfold check_cc_closed, check_code_id_of_witness,
    check_code_id_unknown_arity in Hcheck.
  apply andb_true_iff in Hcheck as [Hwitness Hunknown].
  rewrite forallb_forall in Hwitness, Hunknown.
  constructor.
  - apply witness_any_usage_closed.
  - apply witness_other_source_closed.
  - apply witness_any_source_usage_closed.
  - apply witness_other_own_source_closed.
  - apply witness_not_own_source_closed.
  - apply witness_any_source_closed.
  - intros call_witness code_id Hin Hused Hcannot.
    apply Hwitness, in_derived_by_code_id_of_witness.
    exists call_witness. auto.
  - intros set_of_closures call_witness code_id Hin Hused Hcode.
    apply Hunknown, in_derived_by_code_id_unknown_arity.
    exists set_of_closures, call_witness. auto.
Qed.

(* The derived table has all that the two rules derive, and has it in
   every closed table, the witnesses of which have those that the check
   computes ([witness_forced]). *)
Theorem derive_cannot_change_least : derive_cannot_change_least_statement.
Proof.
  intros F T. split.
  - unfold check_cc_closed, check_code_id_of_witness,
      check_code_id_unknown_arity, derive_cannot_change.
    apply andb_true_iff. split; apply forallb_forall; intros code_id Hin;
      apply mem_node_In, in_or_app; auto.
  - intros ccw cc Hclosed n Hn.
    apply mem_node_In in Hn. unfold derive_cannot_change in Hn.
    apply in_app_or in Hn as [Hn | Hn].
    + apply in_derived_by_code_id_of_witness in Hn.
      destruct Hn as (call_witness & Hin & Hused & Hwitness).
      apply (cc_code_id_of_witness _ _ _ _ Hclosed call_witness);
        [ exact Hin | exact Hused | ].
      exact (witness_forced F T ccw cc Hclosed call_witness Hwitness).
    + apply in_derived_by_code_id_unknown_arity in Hn.
      destruct Hn as (set_of_closures & call_witness & Hin & Hused & Hcode).
      exact (cc_code_id_unknown_arity _ _ _ _ Hclosed set_of_closures
               call_witness n Hin Hused Hcode).
Qed.

(* The witnesses of known arity that calls read off a node have the same
   sources in a table that passes [check_apply_widgets]. *)
Lemma check_apply_widgets_sound (F : facts) (T : tables) :
  check_apply_widgets F T = true ->
  ∀ a a' n y : node,
    g_accessor (graph_of_facts F) a
      (Field_call_witness Known_arity_code_pointer) n ->
    g_accessor (graph_of_facts F) a'
      (Field_call_witness Known_arity_code_pointer) n ->
    (s_any_source (solution_of_tables T) a' = true ->
     s_any_source (solution_of_tables T) a = true)
    ∧ (s_sources (solution_of_tables T) a' y = true ->
       s_sources (solution_of_tables T) a y = true).
Proof.
  intros Hcheck a a' n y Ha Ha'. cbn [graph_of_facts g_accessor] in Ha, Ha'.
  cbn [solution_of_tables s_any_source s_sources].
  unfold check_apply_widgets in Hcheck. rewrite forallb_forall in Hcheck.
  specialize (Hcheck _ Ha'). cbv beta iota in Hcheck.
  rewrite forallb_forall in Hcheck. specialize (Hcheck _ Ha).
  cbv beta iota in Hcheck.
  assert (Hfield : field_eqb (Field_call_witness Known_arity_code_pointer)
                     (Field_call_witness Known_arity_code_pointer)
                   = true).
  { unfold field_eqb. destruct (field_eq_dec _ _); [ reflexivity | ].
    contradiction. }
  assert (Hnode : node_eqb n n = true).
  { unfold node_eqb. destruct (node_eq_dec n n); [ reflexivity | ].
    contradiction. }
  rewrite Hfield, Hnode in Hcheck. cbn [andb negb orb] in Hcheck.
  apply andb_true_iff in Hcheck as [Has Hsrc].
  split.
  - intros H. rewrite H in Has. destruct (mem_node a (t_any_source T));
      [ reflexivity | discriminate Has ].
  - intros H. rewrite forallb_forall in Hsrc.
    apply mem_pair_In in H. specialize (Hsrc _ H). cbv beta iota in Hsrc.
    assert (Ha'a' : node_eqb a' a' = true).
    { unfold node_eqb. destruct (node_eq_dec a' a'); [ reflexivity | ].
      contradiction. }
    rewrite Ha'a' in Hsrc. exact Hsrc.
Qed.

(* A witness gives one code id where [check_witness_code_ids] passes. *)
Lemma check_witness_code_ids_sound (F : facts) :
  check_witness_code_ids F = true ->
  ∀ (w : node) (id id' : code_id),
    g_constructor (graph_of_facts F) w Field_code_id_of_call_witness
      (Node_code_id id) ->
    g_constructor (graph_of_facts F) w Field_code_id_of_call_witness
      (Node_code_id id') ->
    id' = id.
Proof.
  intros Hcheck w id id' H H'. cbn [graph_of_facts g_constructor] in H, H'.
  unfold check_witness_code_ids in Hcheck. rewrite forallb_forall in Hcheck.
  specialize (Hcheck _ H). cbv beta iota in Hcheck.
  assert (Hfield : field_eqb Field_code_id_of_call_witness
                     Field_code_id_of_call_witness = true).
  { unfold field_eqb. destruct (field_eq_dec _ _); [ reflexivity | ].
    contradiction. }
  assert (Hw : node_eqb w w = true).
  { unfold node_eqb. destruct (node_eq_dec w w); [ reflexivity | ].
    contradiction. }
  rewrite Hfield in Hcheck. cbn [negb orb] in Hcheck.
  rewrite forallb_forall in Hcheck. specialize (Hcheck _ H').
  cbv beta iota in Hcheck. rewrite Hw, Hfield in Hcheck.
  cbn [andb negb orb] in Hcheck.
  unfold node_eqb in Hcheck.
  destruct (node_eq_dec (Node_code_id id') (Node_code_id id)) as [E | _];
    [ injection E as E; exact E | discriminate Hcheck ].
Qed.

(* ======================================================================== *)
(* The rebuild with no change of calling convention                         *)
(* ======================================================================== *)

(* With every code of the unit in the table, [rebuild_unit_cc] is
   [rebuild_unit]: the two rebuilds are the same case for case
   (FlambdaReaperRebuildProofs.rebuild_expr_cc_unchanged, which the proof
   of the rebuild runs on). *)
Theorem rebuild_unit_cc_unchanged : rebuild_unit_cc_unchanged_statement.
Proof.
  intros F sol cannot_change code_deps preserve_direct_calls e Hall.
  unfold rebuild_unit_cc, rebuild_unit.
  apply FlambdaReaperRebuildProofs.rebuild_expr_cc_unchanged. exact Hall.
Qed.

(* ======================================================================== *)
(* Runs of units whose code changes                                         *)
(* ======================================================================== *)

Theorem two_calls_example : two_calls_example_statement.
Proof.
  do 10 (split; [ vm_compute; reflexivity | ]).
  split; exists 30; vm_compute; reflexivity.
Qed.

Theorem poison_result_example : poison_result_example_statement.
Proof.
  do 11 (split; [ vm_compute; reflexivity | ]).
  split; exists 30; vm_compute; reflexivity.
Qed.

Theorem masked_return_example : masked_return_example_statement.
Proof.
  do 11 (split; [ vm_compute; reflexivity | ]).
  split; exists 30; vm_compute; reflexivity.
Qed.

Theorem tail_escape_example : tail_escape_example_statement.
Proof.
  do 14 (split; [ vm_compute; reflexivity | ]).
  split; exists 30; vm_compute; reflexivity.
Qed.

Theorem cc_fragment_examples : cc_fragment_examples_statement.
Proof. vm_compute. repeat split. Qed.

Theorem spurious_widget_counterexample :
  spurious_widget_counterexample_statement.
Proof.
  do 9 (split; [ vm_compute; reflexivity | ]).
  split; [ exists 30; vm_compute; reflexivity | ].
  do 6 (split; [ vm_compute; reflexivity | ]).
  assert (Hsource :
    behaves
      (State SpuriousWidgetExample.unit_expr [] [] empty_static_env
         empty_heap)
      (Exits 1%positive [Value_const (Tagged_immediate 9)] empty_heap))
    by (exists 30; vm_compute; reflexivity).
  split; [ exact Hsource | ].
  intros preserve_direct_calls (_ & _ & Hwrong).
  assert (Htarget :
    behaves
      (State
         (SpuriousWidgetExample.rebuilt SpuriousWidgetExample.spurious
            preserve_direct_calls)
         [] [] empty_static_env empty_heap)
      Goes_wrong)
    by (exists 30; destruct preserve_direct_calls; vm_compute; reflexivity).
  pose proof (FlambdaSemanticsProofs.behaviour_unique _ _ _ (Hwrong Htarget)
                Hsource) as Heq.
  discriminate Heq.
Qed.

(* What [check_cc_returns] says of the code that changes. *)
Lemma check_cc_returns_sound (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep) (e : expr) :
  check_cc_returns F T cannot_change code_deps e = true ->
  ∀ (id : code_id) (params : list variable) (mc : variable)
    (ret exn : continuation) (md : code_metadata) (body : expr)
    (pd rd : list param_decision),
    (id, Code params mc ret exn md body) ∈ FlambdaReaperFlowSpec.unit_codes e ->
    code_calling_convention_change (solution_of_tables T)
      (fun n => mem_node n cannot_change) code_deps
      (static_names code_ids_bound e) id md
    = Changing_calling_convention pd rd ->
    tail_calls_known F (solution_of_tables T)
      (fun n => mem_node n cannot_change) code_deps
      (static_names code_ids_bound e) (bound_code_metadata e) ret body
    = true.
Proof.
  intros Hcheck id params mc ret exn md body pd rd Hin Hch.
  unfold check_cc_returns in Hcheck. rewrite forallb_forall in Hcheck.
  specialize (Hcheck _ Hin). cbv beta iota in Hcheck. rewrite Hch in Hcheck.
  exact Hcheck.
Qed.

(* ======================================================================== *)
(* The traversal makes each witness give one code id                        *)
(* ======================================================================== *)

(* The witnesses that give a code id in the facts of an accumulator are
   made nodes before its counter, each giving one. *)
Definition wci_inv (a : acc) : Prop :=
  (∀ w c,
     Fact_constructor w Field_code_id_of_call_witness c ∈ acc_facts a ->
     ∃ n, w = Node_made n ∧ (n < acc_next a)%positive)
  ∧ (∀ w c c',
       Fact_constructor w Field_code_id_of_call_witness c ∈ acc_facts a ->
       Fact_constructor w Field_code_id_of_call_witness c' ∈ acc_facts a ->
       c' = c).

(* A step of the traversal keeps [wci_inv] and its counter grows. *)
Definition wstep (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive ∧ (wci_inv a -> wci_inv a').

Lemma wstep_refl (a : acc) : wstep a a.
Proof. split; [ lia | auto ]. Qed.

Lemma wstep_trans (a a' a'' : acc) :
  wstep a a' -> wstep a' a'' -> wstep a a''.
Proof. intros [H1 H2] [H3 H4]. split; [ lia | auto ]. Qed.

Definition no_wci (l : list fact) : Prop :=
  ∀ w c, ~ Fact_constructor w Field_code_id_of_call_witness c ∈ l.

(* Facts added whose witnesses that give a code id are made by the step
   and all give the same one. *)
Lemma wstep_add (a a' : acc) (l : list fact) (c0 : node) :
  acc_facts a' = l ++ acc_facts a ->
  (acc_next a <= acc_next a')%positive ->
  (∀ w c, Fact_constructor w Field_code_id_of_call_witness c ∈ l ->
     c = c0 ∧ ∃ n, w = Node_made n ∧ (acc_next a <= n)%positive
                 ∧ (n < acc_next a')%positive) ->
  wstep a a'.
Proof.
  intros Hf Hle Hl. split; [ exact Hle | ].
  intros [Hmade Hfun]. unfold wci_inv. rewrite Hf. split.
  - intros w c Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + destruct (Hl _ _ Hin) as (_ & n & -> & _ & Hn). exists n. auto.
    + destruct (Hmade _ _ Hin) as (n & -> & Hn). exists n.
      split; [ reflexivity | lia ].
  - intros w c c' Hin Hin'. apply in_app_or in Hin.
    apply in_app_or in Hin'.
    destruct Hin as [Hin | Hin], Hin' as [Hin' | Hin'].
    + destruct (Hl _ _ Hin) as [-> _]. destruct (Hl _ _ Hin') as [-> _].
      reflexivity.
    + exfalso. destruct (Hl _ _ Hin) as (_ & n & -> & Hn & _).
      destruct (Hmade _ _ Hin') as (n' & Heq & Hn'). injection Heq as <-.
      lia.
    + exfalso. destruct (Hl _ _ Hin') as (_ & n & -> & Hn & _).
      destruct (Hmade _ _ Hin) as (n' & Heq & Hn'). injection Heq as <-.
      lia.
    + exact (Hfun _ _ _ Hin Hin').
Qed.

Lemma wstep_clean (a a' : acc) (l : list fact) :
  acc_facts a' = l ++ acc_facts a ->
  (acc_next a <= acc_next a')%positive -> no_wci l -> wstep a a'.
Proof.
  intros Hf Hle Hl. apply (wstep_add a a' l Node_all_constants Hf Hle).
  intros w c Hin. destruct (Hl w c Hin).
Qed.

Lemma wstep_same (a a' : acc) :
  acc_facts a' = acc_facts a -> (acc_next a <= acc_next a')%positive ->
  wstep a a'.
Proof.
  intros Hf Hle. apply (wstep_clean a a' []); [ exact Hf | exact Hle | ].
  intros w c [].
Qed.

Lemma wstep_add_facts (l : list fact) (a : acc) :
  no_wci l -> wstep a (add_facts l a).
Proof.
  intros Hl. apply (wstep_clean _ _ l); [ reflexivity | cbn; lia | exact Hl ].
Qed.

Ltac wci_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in
    apply in_map_iff in H; destruct H as [y [H _]]
  | In _ (flat_map _ _) =>
    let y := fresh "y" in
    apply in_flat_map in H; destruct H as [y [_ H]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  | In _ (match ?t with _ => _ end) => destruct t
  end.

Ltac wci_close H :=
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  cbn in H; try congruence.

Ltac no_wci_tac :=
  let w := fresh "w" in let c := fresh "c" in let H := fresh "H" in
  intros w c H; wci_split H; wci_close H.

Lemma wstep_fresh (a : acc) : wstep a (snd (fresh a)).
Proof. apply wstep_same; cbn; [ reflexivity | lia ]. Qed.

Lemma fresh_list_spec (n : nat) (a : acc) :
  acc_facts (snd (fresh_list n a)) = acc_facts a
  ∧ (acc_next a <= acc_next (snd (fresh_list n a)))%positive
  ∧ ∀ x, x ∈ fst (fresh_list n a) ->
      ∃ m, x = Node_made m ∧ (acc_next a <= m)%positive
           ∧ (m < acc_next (snd (fresh_list n a)))%positive.
Proof.
  revert a. induction n as [ | n IH]; intros a; cbn [fresh_list].
  - cbn. split; [ reflexivity | split; [ lia | intros x [] ] ].
  - destruct (fresh a) as [x a1] eqn:E1.
    assert (Hf1 : acc_facts a1 = acc_facts a
                  ∧ acc_next a1 = Pos.succ (acc_next a)
                  ∧ x = Node_made (acc_next a)).
    { unfold fresh in E1. injection E1 as <- <-. cbn. auto. }
    destruct Hf1 as (Hf1 & Hn1 & ->).
    destruct (IH a1) as (H1 & H2 & H3).
    destruct (fresh_list n a1) as [xs a2] eqn:E2. cbn [fst snd] in *.
    split; [ congruence | split; [ lia | ] ].
    intros y [<- | Hy].
    + exists (acc_next a). split; [ reflexivity | lia ].
    + destruct (H3 y Hy) as (m & -> & Hm1 & Hm2). exists m.
      split; [ reflexivity | lia ].
Qed.

Lemma wstep_fresh_list (n : nat) (a : acc) : wstep a (snd (fresh_list n a)).
Proof.
  destruct (fresh_list_spec n a) as (H1 & H2 & _). apply wstep_same; assumption.
Qed.

Lemma wstep_note_simple (s : simple) (a : acc) : wstep a (note_simple s a).
Proof. destruct s; apply wstep_same; cbn; (reflexivity || lia). Qed.

Lemma wstep_fold {A : Type} (f : acc -> A -> acc) (l : list A) (a : acc) :
  (∀ x a, x ∈ l -> wstep a (f a x)) -> wstep a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | x l IH]; intros a H; cbn.
  - apply wstep_refl.
  - eapply wstep_trans; [ apply H; left; reflexivity | ].
    apply IH. intros y b Hy. apply H. right. exact Hy.
Qed.

Lemma wstep_note_simples (l : list simple) (a : acc) :
  wstep a (note_simples l a).
Proof. apply wstep_fold. intros s b _. apply wstep_note_simple. Qed.

Lemma wstep_push_apply_dep (d : apply_dep) (a : acc) :
  wstep a (push_apply_dep d a).
Proof. apply wstep_same; cbn; [ reflexivity | lia ]. Qed.

Lemma wstep_push_closure_dep (d : closure_dep) (a : acc) :
  wstep a (push_closure_dep d a).
Proof. apply wstep_same; cbn; [ reflexivity | lia ]. Qed.

Lemma wstep_add_code_dep (id : code_id) (cd : code_dep) (a : acc) :
  wstep a (add_code_dep id cd a).
Proof. apply wstep_same; cbn; [ reflexivity | lia ]. Qed.

Create HintDb wci discriminated.
#[local] Hint Resolve wstep_fresh wstep_fresh_list wstep_note_simple
  wstep_note_simples wstep_push_apply_dep wstep_push_closure_dep
  wstep_add_code_dep : wci.

(* A chain of steps, from the last. *)
Ltac wauto :=
  cbv zeta;
  match goal with
  | |- wstep ?a ?a => apply wstep_refl
  | |- wstep ?a (snd (?x, ?y)) => change (wstep a y); wauto
  | |- wstep ?a (add_facts ?l ?b) =>
    apply (wstep_trans a b); [ wauto | apply wstep_add_facts; no_wci_tac ]
  | |- wstep ?a (add_fact ?f ?b) =>
    apply (wstep_trans a b); [ wauto | apply wstep_add_facts; no_wci_tac ]
  | |- wstep ?a (snd (match ?t with _ => _ end)) =>
    lazymatch type of t with
    | (_ * acc)%type =>
      let Hs := fresh "Hs" in
      assert (Hs : wstep a (snd t)) by wauto;
      let x := fresh "x" in let b := fresh "b" in let E := fresh "E" in
      destruct t as [x b] eqn:E; try rewrite E in Hs; cbn [snd] in Hs;
      apply (wstep_trans a b); [ exact Hs | wauto ]
    | _ => destruct t; wauto
    end
  | |- wstep ?a (match ?t with _ => _ end) =>
    lazymatch type of t with
    | (_ * acc)%type =>
      let Hs := fresh "Hs" in
      assert (Hs : wstep a (snd t)) by wauto;
      let x := fresh "x" in let b := fresh "b" in let E := fresh "E" in
      destruct t as [x b] eqn:E; try rewrite E in Hs; cbn [snd] in Hs;
      apply (wstep_trans a b); [ exact Hs | wauto ]
    | _ => destruct t; wauto
    end
  | |- wstep ?a (snd (?g ?b)) =>
    apply (wstep_trans a b); [ wauto | solve [ eauto with wci ] ]
  | |- wstep ?a (?g ?b) =>
    apply (wstep_trans a b); [ wauto | solve [ eauto with wci ] ]
  end.

Lemma wstep_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  wstep a (cond_any_usage env s a).
Proof. unfold cond_any_usage. wauto. Qed.

Lemma wstep_cond_alias (env : traverse_env) (from to_ : node) (a : acc) :
  wstep a (cond_alias env from to_ a).
Proof. unfold cond_alias. wauto. Qed.

Lemma wstep_find_cont (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  wstep a (snd (find_cont env k n a)).
Proof. unfold find_cont. wauto. Qed.

#[local] Hint Resolve wstep_cond_any_usage wstep_cond_alias wstep_find_cont
  : wci.

Lemma wstep_block_deps (base : node) (fields : list simple) (a : acc) :
  wstep a (block_deps base fields a).
Proof. unfold block_deps. wauto. Qed.

#[local] Hint Resolve wstep_block_deps : wci.

Lemma wstep_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  wstep a (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  destruct p as [ u | b | v | h ]; [ destruct u | destruct b | destruct v
    | destruct h ]; wauto.
Qed.

Lemma wstep_record_set (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  wstep a (record_set_of_closures_deps env closures function_slots
             value_slots a).
Proof.
  unfold record_set_of_closures_deps. apply wstep_fold.
  intros [base code] b _. wauto.
Qed.

Lemma wstep_closures_of_set (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (function_decls : list (function_slot * code_id)) (a : acc) :
  wstep a (snd (closures_of_set x slot others function_decls a)).
Proof.
  revert a. induction function_decls as [ | [f id] decls IH]; intros a;
    cbn [closures_of_set].
  - apply wstep_refl.
  - destruct (if Pos.eqb f slot then _ else _) as [name b] eqn:E.
    assert (Hb : wstep a b).
    { destruct (Pos.eqb f slot);
        [ injection E as _ <-; apply wstep_refl | ].
      destruct (lookup f others);
        [ injection E as _ <-; apply wstep_refl | ].
      pose proof (wstep_fresh a) as H. rewrite E in H. exact H. }
    specialize (IH b).
    destruct (closures_of_set x slot others decls b) as [rest c].
    exact (wstep_trans _ _ _ Hb IH).
Qed.

#[local] Hint Resolve wstep_traverse_prim wstep_record_set
  wstep_closures_of_set : wci.

Lemma wstep_traverse_named (env : traverse_env) (x : variable)
    (defining_expr : named) (a : acc) :
  wstep a (traverse_named env x defining_expr a).
Proof.
  unfold traverse_named, traverse_closure_of_set.
  destruct defining_expr; wauto.
Qed.

Lemma wstep_traverse_static_binding (env : traverse_env)
    (binding : static_binding) (a : acc) :
  wstep a (traverse_static_binding env binding a).
Proof.
  unfold traverse_static_binding, boxed_number_facts.
  destruct binding as [ | | s const]; [ wauto | wauto | ].
  destruct const; wauto.
Qed.

Lemma fresh_eq (a : acc) :
  fresh a = (Node_made (acc_next a), snd (fresh a))
  ∧ acc_facts (snd (fresh a)) = acc_facts a
  ∧ acc_next (snd (fresh a)) = Pos.succ (acc_next a).
Proof. cbn. auto. Qed.

Lemma wstep_known_witness (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  wstep a (snd (create_known_arity_call_witness id params returns exn a)).
Proof.
  unfold create_known_arity_call_witness.
  destruct (fresh_eq a) as (E & Hf & Hn).
  destruct (fresh a) as [w b] eqn:E'. injection E as ->.
  cbn [snd] in Hf, Hn.
  cbv zeta. cbn [snd].
  eapply (wstep_add _ _ _ (Node_code_id id));
    [ cbn; rewrite Hf; reflexivity | cbn; lia | ].
  intros w' c H. wci_split H; wci_close H.
  injection H as <- <-. split; [ reflexivity | ].
  exists (acc_next a). cbn. split; [ reflexivity | lia ].
Qed.

Lemma wstep_tupled_witnesses (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  wstep a
    (snd (create_unknown_arity_tupled_call_witnesses id params returns exn
            a)).
Proof.
  unfold create_unknown_arity_tupled_call_witnesses.
  destruct (fresh_eq a) as (E & Hf & Hn).
  destruct (fresh a) as [w b] eqn:E'. injection E as ->.
  cbn [snd] in Hf, Hn.
  destruct (fresh_eq b) as (E2 & Hf2 & Hn2).
  destruct (fresh b) as [u b2] eqn:E2'. injection E2 as ->.
  cbn [snd] in Hf2, Hn2.
  destruct (fresh_eq b2) as (E3 & Hf3 & Hn3).
  destruct (fresh b2) as [k b3] eqn:E3'. injection E3 as ->.
  cbn [snd] in Hf3, Hn3.
  cbv zeta. cbn [snd].
  eapply (wstep_add _ _ _ (Node_code_id id));
    [ cbn; rewrite Hf3, Hf2, Hf; reflexivity | cbn; lia | ].
  intros w' c H. wci_split H; wci_close H.
  injection H as <- <-. split; [ reflexivity | ].
  exists (acc_next a). cbn. split; [ reflexivity | lia ].
Qed.

Lemma unknown_arity_witness_deps_spec (id : code_id) (returns : list node)
    (exn : node) (pw : list (list node * node)) (a : acc) :
  ∃ l,
    acc_facts (unknown_arity_witness_deps id returns exn pw a)
    = l ++ acc_facts a
    ∧ (acc_next a
       <= acc_next (unknown_arity_witness_deps id returns exn pw a))%positive
    ∧ ∀ w c, Fact_constructor w Field_code_id_of_call_witness c ∈ l ->
        c = Node_code_id id ∧ w ∈ map snd pw.
Proof.
  revert a. induction pw as [ | [first witness] rest IH]; intros a;
    cbn [unknown_arity_witness_deps].
  - exists []. split; [ reflexivity | split; [ lia | intros w c [] ] ].
  - destruct rest as [ | [g nw] rest'].
    + eexists. split;
        [ cbn [acc_facts add_facts]; rewrite app_assoc; reflexivity | ].
      split; [ cbn; lia | ].
      intros w c H. wci_split H; wci_close H.
      injection H as <- <-. split; [ reflexivity | left; reflexivity ].
    + cbv zeta.
      match goal with
      | |- context [fresh ?t] => destruct (fresh_eq t) as (E & Hf & Hn)
      end.
      destruct (fresh _) as [v b] eqn:E'. injection E as ->.
      cbn [snd] in Hf, Hn.
      match goal with
      | |- context [unknown_arity_witness_deps id returns exn _ ?t] =>
        destruct (IH t) as (l & Hl & Hle & Hw)
      end.
      eexists. split.
      * rewrite Hl. cbn [acc_facts add_facts]. rewrite Hf.
        cbn [acc_facts add_facts]. rewrite !app_assoc. reflexivity.
      * cbn [acc_next add_facts] in Hle, Hn.
        split; [ cbn [acc_next add_facts]; lia | ].
        intros w c H. wci_split H; wci_close H;
          first
            [ destruct (Hw _ _ H) as [-> Hin];
              split; [ reflexivity | right; exact Hin ]
            | injection H as <- <-;
              split; [ reflexivity | left; reflexivity ] ].
Qed.

Lemma wstep_non_tupled_witnesses (id : code_id) (params_arity : arity)
    (params returns : list node) (exn : node) (a : acc) :
  wstep a
    (snd (create_unknown_arity_non_tupled_call_witnesses id params_arity
            params returns exn a)).
Proof.
  unfold create_unknown_arity_non_tupled_call_witnesses. cbv zeta.
  set (groups := at_least_one_group _).
  destruct (fresh_list_spec (length groups) a) as (Hf & Hn & Hm).
  destruct (fresh_list (length groups) a) as [ws b] eqn:E.
  cbn [fst snd] in Hf, Hn, Hm.
  destruct (unknown_arity_witness_deps_spec id returns exn
              (combine groups ws) b) as (l & Hl & Hle & Hw).
  cbn [snd].
  apply (wstep_add _ _ l (Node_code_id id));
    [ rewrite Hl, Hf; reflexivity | lia | ].
  intros w c H. destruct (Hw _ _ H) as [-> Hin].
  split; [ reflexivity | ].
  apply in_map_iff in Hin. destruct Hin as [[g w'] [Heq Hin]].
  cbn in Heq. subst w'. apply in_combine_r in Hin.
  destruct (Hm _ Hin) as (m & -> & H1 & H2).
  exists m. split; [ reflexivity | lia ].
Qed.

Lemma wstep_unknown_witnesses (id : code_id) (is_tupled : bool)
    (params_arity : arity) (params returns : list node) (exn : node)
    (a : acc) :
  wstep a
    (snd (create_unknown_arity_call_witnesses id is_tupled params_arity
            params returns exn a)).
Proof.
  unfold create_unknown_arity_call_witnesses. destruct is_tupled;
    [ apply wstep_tupled_witnesses | apply wstep_non_tupled_witnesses ].
Qed.

#[local] Hint Resolve wstep_known_witness wstep_unknown_witnesses : wci.

Lemma wstep_prepare_binding (binding : static_binding) (a : acc) :
  wstep a (prepare_binding binding a).
Proof.
  destruct binding; cbn [prepare_binding]; [ | apply wstep_refl
    | apply wstep_refl ].
  unfold prepare_code. wauto.
Qed.

Lemma wstep_known_widget (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  wstep a (snd (make_known_arity_apply_widget env ap returns exn a)).
Proof. unfold make_known_arity_apply_widget. wauto. Qed.

Lemma wstep_unknown_apply_deps (returns : list node) (exn called : node)
    (aw : list (list simple * node)) (a : acc) :
  wstep a (unknown_arity_apply_deps returns exn called aw a).
Proof.
  revert a. induction aw as [ | [first witness] rest IH]; intros a;
    cbn [unknown_arity_apply_deps].
  - apply wstep_refl.
  - destruct rest as [ | [g nw] rest']; wauto.
Qed.

#[local] Hint Resolve wstep_known_widget wstep_unknown_apply_deps : wci.

Lemma wstep_unknown_widget (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  wstep a (snd (make_unknown_arity_apply_widget env ap returns exn a)).
Proof. unfold make_unknown_arity_apply_widget. wauto. Qed.

Lemma wstep_add_apply (env : traverse_env) (id : code_id)
    (callee call_widget : node) (b : bool) (a : acc) :
  wstep a (add_apply env id callee call_widget b a).
Proof. unfold add_apply. wauto. Qed.

#[local] Hint Resolve wstep_unknown_widget wstep_add_apply : wci.

Lemma wstep_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  wstep a (traverse_apply env ap a).
Proof. unfold traverse_apply, traverse_call_kind. wauto. Qed.

Lemma wstep_apply_cont_deps (env : traverse_env) (ac : apply_cont)
    (a : acc) :
  wstep a (apply_cont_deps env ac a).
Proof. unfold apply_cont_deps. wauto. Qed.

Lemma wstep_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  wstep a (traverse_switch env scrutinee arms a).
Proof.
  unfold traverse_switch.
  apply (wstep_trans a (cond_any_usage env scrutinee a));
    [ apply wstep_cond_any_usage | ].
  apply wstep_fold. intros [i ac] b _. apply wstep_apply_cont_deps.
Qed.

Lemma wstep_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  wstep a (exn_handler_deps env h a).
Proof. unfold exn_handler_deps. destruct h as [[ | p ps] md body]; wauto. Qed.

Lemma wstep_traverse :
  (∀ e env a, wstep a (traverse env e a))
  ∧ (∀ h env a, wstep a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a. cbn [traverse].
    exact (wstep_trans _ _ _ (wstep_traverse_named env x d a) (IH _ _)).
  - intros bindings body IHc IH env a. cbn [traverse]. cbv zeta.
    eapply wstep_trans; [ | apply IH ].
    eapply wstep_trans; [ | apply wstep_fold ].
    + eapply wstep_trans; [ | apply wstep_fold ].
      * apply wstep_fold. intros b a' _. apply wstep_prepare_binding.
      * intros b a' _. apply wstep_traverse_static_binding.
    + intros [id c | id | s const] a' Hin; cbn [traverse_binding];
        [ | apply wstep_refl | apply wstep_refl ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ];
        [ | apply wstep_refl ].
      eapply wstep_trans;
        [ | apply (IHc id (Code params mc ret exn md cbody) Hin) ].
      apply wstep_add_facts. no_wci_tac.
  - intros k h body IHh IH env a. cbn [traverse]. cbv zeta.
    eapply wstep_trans; [ | apply IH ].
    eapply wstep_trans; [ | apply wstep_exn_handler_deps ].
    apply IHh.
  - intros handlers body IHh IH env a. cbn [traverse]. cbv zeta.
    eapply wstep_trans; [ | apply IH ].
    apply wstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
  - intros ap env a. apply wstep_traverse_apply.
  - intros ac env a. apply wstep_apply_cont_deps.
  - intros scrutinee arms env a. apply wstep_traverse_switch.
  - intros env a. apply wstep_refl.
  - intros params md handler IH env a. cbn [traverse_handler]. apply IH.
Qed.

Lemma wstep_add_closure_dep (code_deps : finite_map code_dep)
    (d : closure_dep) (a : acc) :
  wstep a (add_closure_dep code_deps d a).
Proof.
  unfold add_closure_dep. destruct (lookup _ code_deps) as [cd | ].
  - wauto.
  - destruct (fresh_eq a) as (E & Hf & Hn).
    destruct (fresh a) as [w b] eqn:E'. injection E as ->.
    cbn [snd] in Hf, Hn.
    eapply (wstep_add _ _ _ (cld_let_bound_name_of_the_closure d));
      [ cbn [acc_facts add_facts]; rewrite Hf; reflexivity
      | cbn [acc_next add_facts]; lia | ].
    intros w' c H. wci_split H; wci_close H.
    injection H as <- <-. split; [ reflexivity | ].
    exists (acc_next a). split; [ reflexivity | cbn [acc_next add_facts]; lia ].
Qed.

Lemma wstep_resolve_delayed_deps (unit_symbols : list symbol) (a : acc) :
  wstep a (resolve_delayed_deps unit_symbols a).
Proof.
  unfold resolve_delayed_deps. cbv zeta.
  eapply wstep_trans; [ | apply wstep_add_facts; no_wci_tac ].
  eapply wstep_trans; [ | apply wstep_fold ].
  - apply wstep_add_facts.
    intros w c H. apply in_flat_map in H. destruct H as [d [_ H]].
    unfold apply_dep_facts, alias_for_caller in H.
    destruct (ad_function_containing_apply_expr d);
      wci_split H; wci_close H.
  - intros d b _. apply wstep_add_closure_dep.
Qed.

Lemma f_constructor_facts_of_list (l : list fact) (w : node) (f : field)
    (c : node) :
  (w, f, c) ∈ f_constructor (facts_of_list l) -> Fact_constructor w f c ∈ l.
Proof.
  cbn [facts_of_list f_constructor]. intros H. apply in_flat_map in H.
  destruct H as [fct [Hin H]].
  destruct fct; cbn in H; try contradiction.
  destruct H as [H | []]. injection H as -> -> ->. exact Hin.
Qed.

(* The traversal makes each witness give one code id. *)
Lemma traverse_unit_witness_code_ids (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (w c c' : node) :
  (w, Field_code_id_of_call_witness, c)
    ∈ f_constructor (traverse_unit flag u e) ->
  (w, Field_code_id_of_call_witness, c')
    ∈ f_constructor (traverse_unit flag u e) ->
  c' = c.
Proof.
  unfold traverse_unit. cbv zeta. cbn [fresh].
  match goal with
  | |- context [facts_of_list (acc_facts ?X)] =>
    assert (Hw : wstep empty_acc X)
  end.
  { eapply wstep_trans; [ | apply wstep_resolve_delayed_deps ].
    eapply wstep_trans; [ | apply wstep_add_facts; no_wci_tac ].
    eapply wstep_trans; [ | apply (proj1 wstep_traverse) ].
    apply (wstep_clean _ _ [Fact_any_usage (Node_made 2);
                            Fact_any_source Node_all_constants]);
      [ reflexivity | cbn; lia | no_wci_tac ]. }
  intros H H'. apply f_constructor_facts_of_list in H, H'.
  refine (proj2 (proj2 Hw _) _ _ _ H H').
  split; cbn; intros; contradiction.
Qed.

Theorem traverse_witness_code_ids : traverse_witness_code_ids_statement.
Proof.
  intros flag u e. unfold check_witness_code_ids. apply forallb_forall.
  intros [[w f] c] Hin. cbv beta iota.
  unfold field_eqb at 1.
  destruct (field_eq_dec f Field_code_id_of_call_witness) as [-> | ];
    [ | reflexivity ].
  cbn [negb orb]. apply forallb_forall.
  intros [[w' f'] c'] Hin'. cbv beta iota.
  unfold node_eqb at 1, field_eqb.
  destruct (node_eq_dec w' w) as [-> | ]; [ | reflexivity ].
  destruct (field_eq_dec f' Field_code_id_of_call_witness) as [-> | ];
    [ | reflexivity ].
  cbn [andb negb orb]. unfold node_eqb.
  destruct (node_eq_dec c' c) as [ | Hne ]; [ reflexivity | ].
  exfalso. apply Hne.
  exact (traverse_unit_witness_code_ids _ _ _ _ _ _ Hin Hin').
Qed.

Theorem rebuild_cc_refines_observably :
  rebuild_cc_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change b Hlocal Hslots Hfrag Hdc Hrk
    Hwf Hexits Hcheck Hcheck_exact Hccc Hwid.
  pose proof (traverse_witness_code_ids flag u e) as Hcid.
  assert (Hkinded : FlambdaReaperRebuildProofs.calls_kinded [] e = true).
  { apply (proj1 FlambdaReaperRebuildProofs.calls_kinded_of_results e [] []
             [] Hrk).
    intros k l Hl. discriminate Hl. }
  destruct (check_cc_closed_sound _ _ _ Hccc) as (ccw & Hcl).
  exact (proj1 (FlambdaReaperRebuildProofs.rebuild_cc_refines_checked false
                  flag is_local u e T (fun n => mem_node n cannot_change) ccw
                  b Hlocal Hslots Hfrag
                  (proj1 (FlambdaReaperRebuildProofs.static_constants_checked
                            (FlambdaReaperFlowSpec.unit_codes e)) e Hdc)
                  Hkinded Hwf Hexits Hcheck Hcheck_exact Hrk Hcl
                  (check_apply_widgets_sound _ _ Hwid)
                  (check_witness_code_ids_sound _ Hcid))).
Qed.

Theorem two_calls_refines : two_calls_refines_statement.
Proof.
  apply (rebuild_cc_refines_observably Auto Examples.no_local
           Examples.interface Examples.two_calls Examples.two_calls_tables []
           true);
    try (vm_compute; reflexivity).
  - intros f H. discriminate H.
  - left. intros f. reflexivity.
Qed.

Theorem poison_result_refines : poison_result_refines_statement.
Proof.
  apply (rebuild_cc_refines_observably Auto Examples.no_local
           Examples.interface PoisonResultExample.unit_expr
           PoisonResultExample.solution [Node_code_id 52%positive] true);
    try (vm_compute; reflexivity).
  - intros f H. discriminate H.
  - left. intros f. reflexivity.
Qed.

Theorem masked_return_refines : masked_return_refines_statement.
Proof.
  apply (rebuild_cc_refines_observably Auto Examples.no_local
           Examples.interface MaskedReturnExample.unit_expr
           MaskedReturnExample.solution [] true);
    try (vm_compute; reflexivity).
  - intros f H. discriminate H.
  - left. intros f. reflexivity.
Qed.

Theorem tail_escape_refines : tail_escape_refines_statement.
Proof.
  apply (rebuild_cc_refines_observably Auto Examples.no_local
           Examples.interface TailEscapeExample.unit_expr
           TailEscapeExample.solution [Node_code_id 51%positive] true);
    try (vm_compute; reflexivity).
  - intros f H. discriminate H.
  - left. intros f. reflexivity.
Qed.
