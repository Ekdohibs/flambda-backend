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

(* The proofs of FlambdaReaperSolutionSpec.v: the checks of
   FlambdaReaperSolution.v imply what they are said to mean, and the
   example passes. Nothing here needs review. *)

From Stdlib Require Import List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaReaperGraphSpec
  FlambdaReaperSolution FlambdaReaperSolutionSpec.
Import ListNotations.

(* ======================================================================== *)
(* Equality tests and membership                                            *)
(* ======================================================================== *)

Lemma node_eqb_refl : ∀ n : node, node_eqb n n = true.
Proof.
  intros n. unfold node_eqb. destruct (node_eq_dec n n); congruence.
Qed.

Lemma node_eqb_eq : ∀ n n' : node, node_eqb n n' = true -> n = n'.
Proof.
  intros n n'. unfold node_eqb.
  destruct (node_eq_dec n n'); [ auto | discriminate ].
Qed.

Lemma field_eqb_refl : ∀ f : field, field_eqb f f = true.
Proof.
  intros f. unfold field_eqb. destruct (field_eq_dec f f); congruence.
Qed.

Lemma field_eqb_eq : ∀ f f' : field, field_eqb f f' = true -> f = f'.
Proof.
  intros f f'. unfold field_eqb.
  destruct (field_eq_dec f f'); [ auto | discriminate ].
Qed.

Lemma mem_node_In : ∀ (n : node) (l : list node),
  mem_node n l = true <-> n ∈ l.
Proof.
  intros n l. unfold mem_node. rewrite existsb_exists. split.
  - intros (n' & Hin & Heq). apply node_eqb_eq in Heq. subst. exact Hin.
  - intros Hin. exists n. split; [ exact Hin | apply node_eqb_refl ].
Qed.

Lemma mem_pair_In : ∀ (n m : node) (l : list (node * node)),
  mem_pair n m l = true <-> (n, m) ∈ l.
Proof.
  intros n m l. unfold mem_pair. rewrite existsb_exists. split.
  - intros ([n' m'] & Hin & Heq). apply andb_true_iff in Heq as [Hn Hm].
    apply node_eqb_eq in Hn, Hm. subst. exact Hin.
  - intros Hin. exists (n, m). cbn.
    rewrite !node_eqb_refl. auto.
Qed.

Lemma mem_field_node_In : ∀ (f : field) (n : node) (l : list (field * node)),
  mem_field_node f n l = true <-> (f, n) ∈ l.
Proof.
  intros f n l. unfold mem_field_node. rewrite existsb_exists. split.
  - intros ([f' n'] & Hin & Heq). apply andb_true_iff in Heq as [Hf Hn].
    apply field_eqb_eq in Hf. apply node_eqb_eq in Hn. subst. exact Hin.
  - intros Hin. exists (f, n). cbn.
    rewrite field_eqb_refl, node_eqb_refl. auto.
Qed.

Lemma mem_node_field_In : ∀ (n : node) (f : field) (l : list (node * field)),
  mem_node_field n f l = true <-> (n, f) ∈ l.
Proof.
  intros n f l. unfold mem_node_field. rewrite existsb_exists. split.
  - intros ([n' f'] & Hin & Heq). apply andb_true_iff in Heq as [Hn Hf].
    apply field_eqb_eq in Hf. apply node_eqb_eq in Hn. subst. exact Hin.
  - intros Hin. exists (n, f). cbn.
    rewrite field_eqb_refl, node_eqb_refl. auto.
Qed.

(* ======================================================================== *)
(* Reading a test back                                                      *)
(* ======================================================================== *)

(* The relations of the graph and of the solution, as the lists of the
   facts and the tables. *)
Ltac unfold_relations :=
  cbn [graph_of_facts solution_of_tables g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* For each hypothesis that a table has a tuple, the tuple as a member of
   the list. *)
Ltac add_memberships :=
  repeat match goal with
  | Hm : mem_node ?n ?l = true |- _ =>
      lazymatch goal with
      | _ : n ∈ l |- _ => fail
      | _ => pose proof (proj1 (mem_node_In n l) Hm)
      end
  | Hm : mem_pair ?n ?m ?l = true |- _ =>
      lazymatch goal with
      | _ : (n, m) ∈ l |- _ => fail
      | _ => pose proof (proj1 (mem_pair_In n m l) Hm)
      end
  | Hm : mem_field_node ?f ?n ?l = true |- _ =>
      lazymatch goal with
      | _ : (f, n) ∈ l |- _ => fail
      | _ => pose proof (proj1 (mem_field_node_In f n l) Hm)
      end
  end.

(* Rewrite in [H] every test whose answer another hypothesis gives. *)
Ltac rewrite_known H :=
  repeat match goal with
  | Hb : ?t = ?v |- _ =>
      tryif constr_eq Hb H then fail else idtac;
      lazymatch v with true => idtac | false => idtac end;
      lazymatch type of H with context [t] => rewrite Hb in H end
  end.

(* [H] is a test [forallb] over a list, and the context has a member of
   the list: take the test at that member, and simplify it with what the
   context says, until what is left is no longer such a pass. *)
Ltac instantiate_passes H :=
  repeat (
    lazymatch type of H with
    | forallb _ ?l = true =>
        match goal with
        | Hin : ?t ∈ l |- _ =>
            rewrite forallb_forall in H; specialize (H t Hin)
        end
    end;
    cbn beta iota in H;
    rewrite ?node_eqb_refl, ?field_eqb_refl, ?Nat.eqb_refl in H;
    rewrite_known H;
    cbn [negb andb orb] in H).

(* The goal is a field of [closed], and [c] the test of that rule: the
   test, taken at the facts and the tuples that the premises name, is the
   conclusion. *)
Ltac solve_with c :=
  intros;
  match goal with
  | Hc : ?t = true |- _ =>
      lazymatch t with context [c] => idtac end;
      let H := fresh "H" in
      pose proof Hc as H; unfold c in H;
      unfold_relations; add_memberships; instantiate_passes H;
      exact H
  end.

(* ======================================================================== *)
(* Soundness                                                                *)
(* ======================================================================== *)

(* Each field of [closed] is the test named after it. *)
Theorem check_closed_sound : check_closed_sound_statement.
Proof.
  intros is_local F T Hcheck.
  unfold check_closed, check_alias_rules, check_has_usage_rules,
    check_has_source_rules, check_any_usage_rules, check_any_source_rules,
    check_usages_rules, check_sources_rules, check_local_rules,
    check_post_processing_rules in Hcheck.
  repeat match goal with
  | H : _ && _ = true |- _ => apply andb_true_iff in H as [? ?]
  end.
  constructor.
  - solve_with check_alias_input.
  - solve_with check_alias_propagate.
  - solve_with check_alias_if_any_source.
  - solve_with check_alias_constructor_usage.
  - solve_with check_alias_local_constructor_usage.
  - solve_with check_alias_parameter_usage.
  - solve_with check_alias_accessor_source.
  - solve_with check_alias_local_accessor_source.
  - solve_with check_alias_escaping_reading.
  - solve_with check_alias_argument_source.
  - solve_with check_has_usage_any.
  - solve_with check_has_usage_alias.
  - solve_with check_has_usage_accessor.
  - solve_with check_has_usage_argument.
  - solve_with check_has_source_any.
  - solve_with check_has_source_alias.
  - solve_with check_has_source_constructor.
  - solve_with check_has_source_function_slot.
  - solve_with check_has_source_parameter.
  - solve_with check_any_usage_use.
  - solve_with check_any_usage_alias.
  - solve_with check_any_usage_constructor.
  - solve_with check_any_usage_argument.
  - solve_with check_any_source_input.
  - solve_with check_any_source_zero_alloc.
  - solve_with check_any_source_if_used.
  - solve_with check_any_source_alias.
  - solve_with check_any_source_parameter.
  - solve_with check_any_source_accessor.
  - solve_with check_any_source_use.
  - solve_with check_any_usage_input.
  - solve_with check_usages_accessor.
  - solve_with check_usages_argument.
  - solve_with check_usages_alias.
  - solve_with check_sources_constructor.
  - solve_with check_sources_parameter.
  - solve_with check_sources_alias.
  - solve_with check_escaping_field_rule.
  - solve_with check_reading_field_rule.
  - solve_with check_field_used_any_usage.
  - solve_with check_field_used_usage.
  - solve_with check_field_used_reading.
  - solve_with check_field_used_local_source.
Qed.

(* Break the tests of a hypothesis into the facts they find: the members
   of the lists that an [existsb] finds, their components, and the
   equalities that [node_eqb] and [field_eqb] find. *)
Ltac break_tests :=
  repeat match goal with
  | H : _ && _ = true |- _ => apply andb_true_iff in H as [? ?]
  | H : negb _ = true |- _ => apply negb_true_iff in H
  | H : node_eqb _ _ = true |- _ => apply node_eqb_eq in H; subst
  | H : field_eqb _ _ = true |- _ => apply field_eqb_eq in H; subst
  | H : existsb _ _ = true |- _ =>
      let t := fresh "t" in
      apply existsb_exists in H as (t & ? & H)
  | t : (_ * _)%type |- _ => destruct t; cbn beta iota in *
  end.

(* A tuple of [field_of_constructor_is_used] is given by the rule whose
   test found it, and the member of [f_constructor] that the test found
   is the constructor of the rule. *)
Theorem check_field_used_exact_sound :
  check_field_used_exact_sound_statement.
Proof.
  intros is_local F T Hcheck base relation Hused.
  unfold_relations.
  apply mem_node_field_In in Hused.
  unfold check_field_used_exact in Hcheck.
  rewrite forallb_forall in Hcheck.
  specialize (Hcheck _ Hused). cbn beta iota in Hcheck.
  rewrite !orb_true_iff in Hcheck.
  destruct Hcheck as [[[Hrule | Hrule] | Hrule] | Hrule].
  - (* [any_usage base], and [relation] not local *)
    unfold derived_by_field_used_any_usage in Hrule. break_tests.
    eexists. split; [ eassumption | ].
    left. split; assumption.
  - (* a use of [base] whose field [relation] is read and used *)
    unfold derived_by_field_used_usage in Hrule. break_tests.
    eexists. split; [ eassumption | ].
    right; left. do 2 eexists.
    split; [ apply mem_pair_In; eassumption | ].
    split; eassumption.
  - (* [any_usage base], and a read of the local field that is used *)
    unfold derived_by_field_used_reading in Hrule. break_tests.
    eexists. split; [ eassumption | ].
    right; right; left. eexists.
    split; [ assumption | ].
    split; [ apply mem_field_node_In; eassumption | eassumption ].
  - (* a use of what [base] may flow to, through a local field *)
    unfold derived_by_field_used_local_source in Hrule. break_tests.
    eexists. split; [ eassumption | ].
    right; right; right. do 2 eexists.
    split; [ apply mem_pair_In; eassumption | ].
    split; [ assumption | ]. split; [ assumption | ].
    split; eassumption.
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Theorem example_passes : example_passes_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem example_each_tuple_needed : example_each_tuple_needed_statement.
Proof.
  intros T Hin.
  assert (Hall :
    forallb (fun T => negb (check_closed Example.is_local Example.graph T))
      (one_fewer_tables Example.solution) = true)
    by (vm_compute; reflexivity).
  rewrite forallb_forall in Hall.
  apply negb_true_iff, Hall, Hin.
Qed.

Theorem example_unread_field_rejected :
  example_unread_field_rejected_statement.
Proof. split; vm_compute; reflexivity. Qed.
