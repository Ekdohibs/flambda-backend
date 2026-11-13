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
   [check_apply_widgets_sound], [check_witness_code_ids_sound]). Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaWellFormedSpec FlambdaSemanticsSpec FlambdaReaperGraphSpec
  FlambdaReaperSolutionProofs
  FlambdaReaperTraversal FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs FlambdaSemanticsProofs
  FlambdaReaperFlowSpec.
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

Theorem rebuild_cc_refines_observably :
  rebuild_cc_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change b Hlocal Hslots Hfrag Hdc Hrk
    Hwf Hexits Hcheck Hcheck_exact Hccc Hwid Hcid.
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
