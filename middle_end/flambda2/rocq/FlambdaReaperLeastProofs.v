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

(* The proofs of FlambdaReaperLeastSpec.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperLeast
  FlambdaReaperLeastSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventionsProofs
  FlambdaReaperUnboxingRefinesProofs.
Import ListNotations.

(* ======================================================================== *)
(* What is derived holds of a closed solution                               *)
(* ======================================================================== *)

(* The relations of the graph and of the solution, as the lists of the
   facts and the tables. *)
Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

(* Each rule of [derivable] is the field of [closed] of its name, so a
   closed solution has every tuple that the rules derive with its own
   guards. *)
Lemma derivable_holds (is_local : field -> bool) (G : graph)
    (S : solution) (a : atom) :
  closed is_local G S -> derivable is_local G S a -> holds S a = true.
Proof.
  intros Hclosed Hd. induction Hd; cbn [holds] in *;
  [> eapply (alias_input _ _ _ Hclosed)
  | eapply (alias_propagate _ _ _ Hclosed)
  | eapply (alias_if_any_source _ _ _ Hclosed)
  | eapply (alias_constructor_usage _ _ _ Hclosed)
  | eapply (alias_local_constructor_usage _ _ _ Hclosed)
  | eapply (alias_parameter_usage _ _ _ Hclosed)
  | eapply (alias_accessor_source _ _ _ Hclosed)
  | eapply (alias_local_accessor_source _ _ _ Hclosed)
  | eapply (alias_escaping_reading _ _ _ Hclosed)
  | eapply (alias_argument_source _ _ _ Hclosed)
  | eapply (has_usage_any _ _ _ Hclosed)
  | eapply (has_usage_alias _ _ _ Hclosed)
  | eapply (has_usage_accessor _ _ _ Hclosed)
  | eapply (has_usage_argument _ _ _ Hclosed)
  | eapply (has_source_any _ _ _ Hclosed)
  | eapply (has_source_alias _ _ _ Hclosed)
  | eapply (has_source_constructor _ _ _ Hclosed)
  | eapply (has_source_function_slot _ _ _ Hclosed)
  | eapply (has_source_parameter _ _ _ Hclosed)
  | eapply (any_usage_use _ _ _ Hclosed)
  | eapply (any_usage_alias _ _ _ Hclosed)
  | eapply (any_usage_constructor _ _ _ Hclosed)
  | eapply (any_usage_argument _ _ _ Hclosed)
  | eapply (any_source_input _ _ _ Hclosed)
  | eapply (any_source_zero_alloc _ _ _ Hclosed)
  | eapply (any_source_if_used _ _ _ Hclosed)
  | eapply (any_source_alias _ _ _ Hclosed)
  | eapply (any_source_parameter _ _ _ Hclosed)
  | eapply (any_source_accessor _ _ _ Hclosed)
  | eapply (any_source_use _ _ _ Hclosed)
  | eapply (any_usage_input _ _ _ Hclosed)
  | eapply (usages_accessor _ _ _ Hclosed)
  | eapply (usages_argument _ _ _ Hclosed)
  | eapply (usages_alias _ _ _ Hclosed)
  | eapply (sources_constructor _ _ _ Hclosed)
  | eapply (sources_parameter _ _ _ Hclosed)
  | eapply (sources_alias _ _ _ Hclosed)
  | eapply (escaping_field_rule _ _ _ Hclosed)
  | eapply (reading_field_rule _ _ _ Hclosed)
  | eapply (field_used_any_usage _ _ _ Hclosed)
  | eapply (field_used_usage _ _ _ Hclosed)
  | eapply (field_used_reading _ _ _ Hclosed)
  | eapply (field_used_local_source _ _ _ Hclosed) ]; eassumption.
Qed.

(* Fewer guards hold of [S] than of [S']: what the rules derive with the
   guards of [S] they derive with those of [S']. *)
Lemma derivable_mono (is_local : field -> bool) (G : graph)
    (S S' : solution) (a : atom) :
  (∀ x, s_any_usage S' x = true -> s_any_usage S x = true) ->
  (∀ x, s_any_source S' x = true -> s_any_source S x = true) ->
  derivable is_local G S a -> derivable is_local G S' a.
Proof.
  intros Hu Hs Hd.
  assert (Gu : ∀ x, s_any_usage S x = false -> s_any_usage S' x = false).
  { intros x Hx. destruct (s_any_usage S' x) eqn:E; [ | reflexivity ].
    rewrite (Hu x E) in Hx. discriminate. }
  assert (Gs : ∀ x, s_any_source S x = false -> s_any_source S' x = false).
  { intros x Hx. destruct (s_any_source S' x) eqn:E; [ | reflexivity ].
    rewrite (Hs x E) in Hx. discriminate. }
  induction Hd;
  [> eapply by_alias_input
  | eapply by_alias_propagate
  | eapply by_alias_if_any_source
  | eapply by_alias_constructor_usage
  | eapply by_alias_local_constructor_usage
  | eapply by_alias_parameter_usage
  | eapply by_alias_accessor_source
  | eapply by_alias_local_accessor_source
  | eapply by_alias_escaping_reading
  | eapply by_alias_argument_source
  | eapply by_has_usage_any
  | eapply by_has_usage_alias
  | eapply by_has_usage_accessor
  | eapply by_has_usage_argument
  | eapply by_has_source_any
  | eapply by_has_source_alias
  | eapply by_has_source_constructor
  | eapply by_has_source_function_slot
  | eapply by_has_source_parameter
  | eapply by_any_usage_use
  | eapply by_any_usage_alias
  | eapply by_any_usage_constructor
  | eapply by_any_usage_argument
  | eapply by_any_source_input
  | eapply by_any_source_zero_alloc
  | eapply by_any_source_if_used
  | eapply by_any_source_alias
  | eapply by_any_source_parameter
  | eapply by_any_source_accessor
  | eapply by_any_source_use
  | eapply by_any_usage_input
  | eapply by_usages_accessor
  | eapply by_usages_argument
  | eapply by_usages_alias
  | eapply by_sources_constructor
  | eapply by_sources_parameter
  | eapply by_sources_alias
  | eapply by_escaping_field_rule
  | eapply by_reading_field_rule
  | eapply by_field_used_any_usage
  | eapply by_field_used_usage
  | eapply by_field_used_reading
  | eapply by_field_used_local_source ]; eauto.
Qed.

Section Least.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

Lemma least_closed : closed is_local G S.
Proof. exact (check_closed_sound is_local F T (proj1 Hleast)). Qed.

Lemma least_holds (a : atom) : D a -> holds S a = true.
Proof. exact (derivable_holds is_local G S a least_closed). Qed.

Lemma least_derivable (a : atom) : holds S a = true -> D a.
Proof. exact (proj2 Hleast a). Qed.

End Least.

Theorem least_solution_exact : least_solution_exact_statement.
Proof.
  intros is_local F T a Hleast. split;
    [ apply (least_derivable is_local F T Hleast)
    | apply (least_holds is_local F T Hleast) ].
Qed.

(* A rule that concludes [usages x _] has the guard [~~(any_usage x)],
   and one that concludes [sources x _] the guard [~~(any_source x)]. *)
Lemma derivable_usages_guard (is_local : field -> bool) (G : graph)
    (S : solution) (x y : node) :
  derivable is_local G S (Atom_usages x y) -> s_any_usage S x = false.
Proof. intros Hd. inversion Hd; subst; assumption. Qed.

Lemma derivable_sources_guard (is_local : field -> bool) (G : graph)
    (S : solution) (x y : node) :
  derivable is_local G S (Atom_sources x y) -> s_any_source S x = false.
Proof. intros Hd. inversion Hd; subst; assumption. Qed.

Theorem least_no_stale : least_no_stale_statement.
Proof.
  intros is_local F T x y Hleast. split; intros Hin.
  - apply (derivable_usages_guard is_local (graph_of_facts F)
             (solution_of_tables T) x y).
    apply (least_derivable is_local F T Hleast (Atom_usages x y)).
    exact Hin.
  - apply (derivable_sources_guard is_local (graph_of_facts F)
             (solution_of_tables T) x y).
    apply (least_derivable is_local F T Hleast (Atom_sources x y)).
    exact Hin.
Qed.

Theorem stale_tables : stale_tables_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  intros Hleast.
  destruct (least_no_stale _ _ _ (Node_made 8%positive)
              (Node_variable 36%positive) Hleast) as [_ Hs].
  specialize (Hs eq_refl). vm_compute in Hs. discriminate.
Qed.

(* ======================================================================== *)
(* [check_field_used_exact]                                                 *)
(* ======================================================================== *)

(* The tests in the goal that hypotheses answer, answered. *)
Ltac rewrite_bools :=
  repeat match goal with
  | H : ?t = true |- context [?t] => progress rewrite H
  | H : ?t = false |- context [?t] => progress rewrite H
  end.

(* An [existsb] in the goal, whose witnesses are among the hypotheses
   and whose tests they answer. *)
Ltac solve_existsb :=
  repeat first
    [ reflexivity
    | progress rewrite ?node_eqb_refl, ?field_eqb_refl, ?Nat.eqb_refl
    | progress rewrite_bools
    | progress cbn [andb orb negb]
    | apply andb_true_iff; split
    | apply existsb_exists; eexists; split; [ eassumption | cbn beta iota ]
    ].

(* Each tuple of [field_of_constructor_is_used] of a least solution is
   derived by one of the four rules of the post-processing, whose
   premises hold of the solution: the test of that rule finds them. *)
Theorem least_field_used_exact : least_field_used_exact_statement.
Proof.
  intros is_local F T Hleast.
  unfold check_field_used_exact. apply forallb_forall.
  intros [base relation] Hin.
  pose proof (least_derivable is_local F T Hleast
                (Atom_field_used base relation)) as Hd.
  unfold_rels. specialize (Hd (proj2 (mem_node_field_In _ _ _) Hin)).
  inversion Hd; subst; unfold_rels;
  repeat match goal with
  | H : derivable _ _ _ _ |- _ =>
      apply (least_holds is_local F T Hleast) in H; unfold_rels
  end;
  add_memberships; rewrite !orb_true_iff.
  - left; left; left. unfold derived_by_field_used_any_usage.
    solve_existsb.
  - left; left; right. unfold derived_by_field_used_usage.
    solve_existsb.
  - left; right. unfold derived_by_field_used_reading. solve_existsb.
  - right. unfold derived_by_field_used_local_source. solve_existsb.
Qed.

(* ======================================================================== *)
(* A node that is only read into                                            *)
(* ======================================================================== *)

(* What [only_read_into F x f base if_any_source] says, fact by fact. *)
Record read_only (F : facts) (x : node) (f : field) (base : node)
    (if_any_source : bool) : Prop := {
  ro_alias : ∀ from, (x, from) ∉ f_alias F;
  ro_propagate : ∀ c from, (c, x, from) ∉ f_propagate F;
  ro_if_any_source : ∀ c from,
    (c, x, from) ∈ f_alias_if_any_source F ->
    if_any_source = true ∧ c = base;
  ro_accessor : ∀ f' base',
    (x, f', base') ∈ f_accessor F -> f' = f ∧ base' = base;
  ro_parameter_to : ∀ b i, (b, i, x) ∉ f_parameter F;
  ro_parameter_base : ∀ i t, (x, i, t) ∉ f_parameter F;
  ro_constructor : ∀ r from, (x, r, from) ∉ f_constructor F;
  ro_any_source : x ∉ f_any_source F;
  ro_zero_alloc : x ∉ f_zero_alloc_source F;
  ro_if_used : ∀ c, (c, x) ∉ f_any_source_if_used F;
  ro_use : ∀ from, (x, from) ∉ f_use F
}.

(* The test, read back. *)
Lemma only_read_into_read_only (F : facts) (x : node) (f : field)
    (base : node) (if_any_source : bool) :
  only_read_into F x f base if_any_source = true ->
  read_only F x f base if_any_source.
Proof.
  unfold only_read_into. intros H.
  apply andb_true_iff in H as [H Huse].
  apply andb_true_iff in H as [H Hused].
  apply andb_true_iff in H as [H Hzero].
  apply andb_true_iff in H as [H Hany].
  apply andb_true_iff in H as [H Hcons].
  apply andb_true_iff in H as [H Hparam].
  apply andb_true_iff in H as [H Hacc].
  apply andb_true_iff in H as [H Hias].
  apply andb_true_iff in H as [Halias Hprop].
  rewrite forallb_forall in Halias, Hprop, Hias, Hacc, Hparam, Hcons,
    Hused, Huse.
  apply negb_true_iff in Hany, Hzero.
  constructor.
  - intros from Hin. specialize (Halias _ Hin). cbn in Halias.
    rewrite node_eqb_refl in Halias. discriminate.
  - intros c from Hin. specialize (Hprop _ Hin). cbn in Hprop.
    rewrite node_eqb_refl in Hprop. discriminate.
  - intros c from Hin. specialize (Hias _ Hin). cbn in Hias.
    rewrite node_eqb_refl in Hias. cbn in Hias.
    apply andb_true_iff in Hias as [Hb Hc]. apply node_eqb_eq in Hc.
    auto.
  - intros f' base' Hin. specialize (Hacc _ Hin). cbn in Hacc.
    rewrite node_eqb_refl in Hacc. cbn in Hacc.
    apply andb_true_iff in Hacc as [Hf Hb].
    apply field_eqb_eq in Hf. apply node_eqb_eq in Hb. auto.
  - intros b i Hin. specialize (Hparam _ Hin). cbn in Hparam.
    rewrite node_eqb_refl in Hparam. discriminate.
  - intros i t Hin. specialize (Hparam _ Hin). cbn in Hparam.
    rewrite node_eqb_refl, orb_true_r in Hparam. discriminate.
  - intros r from Hin. specialize (Hcons _ Hin). cbn in Hcons.
    rewrite node_eqb_refl in Hcons. discriminate.
  - intros Hin. apply mem_node_In in Hin. congruence.
  - intros Hin. apply mem_node_In in Hin. congruence.
  - intros c Hin. specialize (Hused _ Hin). cbn in Hused.
    rewrite node_eqb_refl in Hused. discriminate.
  - intros from Hin. specialize (Huse _ Hin). cbn in Huse.
    rewrite node_eqb_refl in Huse. discriminate.
Qed.

Section Read_only.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.
Variable x : node.
Variable f : field.
Variable base : node.
Variable if_any_source : bool.
Hypothesis Hro : read_only F x f base if_any_source.
Hypothesis Hnot_local : is_local f = false.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

(* The rules that may give [x] an alias: the two that read the field [f]
   off [base], through a constructor that [base] is a use of or a source
   of [base], and the aliases that hold if [base] has any source. *)
Lemma alias_into (from : node) :
  D (Atom_alias x from) ->
  (∃ b : node,
     (b, f, from) ∈ f_constructor F ∧ D (Atom_usages b base)
     ∧ s_any_usage S b = false)
  ∨ (∃ bs : node,
       D (Atom_sources base bs) ∧ s_any_source S base = false
       ∧ (bs, f, from) ∈ f_constructor F)
  ∨ (if_any_source = true ∧ D (Atom_any_source base)).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - exfalso. eapply (ro_alias _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_propagate _ _ _ _ _ Hro). eassumption.
  - match goal with
    | H : (_, x, _) ∈ f_alias_if_any_source F |- _ =>
        destruct (ro_if_any_source _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    right; right. auto.
  - match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    left. eexists. eauto.
  - match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    congruence.
  - exfalso. eapply (ro_parameter_to _ _ _ _ _ Hro). eassumption.
  - match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    right; left. eexists. eauto.
  - match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    congruence.
  - match goal with
    | H : derivable _ _ _ (Atom_reading_field _ _) |- _ =>
        inversion H; subst; unfold_rels
    end.
    match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    congruence.
  - exfalso. eapply (ro_parameter_to _ _ _ _ _ Hro). eassumption.
Qed.

(* [x] has any source through an alias, or as [base] has. *)
Lemma any_source_into :
  D (Atom_any_source x) ->
  (∃ from : node, D (Atom_alias x from) ∧ D (Atom_any_source from))
  ∨ D (Atom_any_source base).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - exfalso. eapply (ro_any_source _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_zero_alloc _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_if_used _ _ _ _ _ Hro). eassumption.
  - left. eauto.
  - exfalso. eapply (ro_parameter_to _ _ _ _ _ Hro). eassumption.
  - match goal with
    | H : (x, _, _) ∈ f_accessor F |- _ =>
        destruct (ro_accessor _ _ _ _ _ Hro _ _ H) as [-> ->]
    end.
    right. assumption.
  - exfalso. eapply (ro_use _ _ _ _ _ Hro). eassumption.
Qed.

(* [x] has the sources of what it is an alias of. *)
Lemma sources_into (s : node) :
  D (Atom_sources x s) ->
  ∃ from : node,
    D (Atom_sources from s) ∧ s_any_source S from = false
    ∧ D (Atom_alias x from) ∧ s_any_source S x = false.
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - exfalso. eapply (ro_constructor _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_parameter_base _ _ _ _ _ Hro). eassumption.
  - eexists. eauto.
Qed.

(* [x] has a source if it has any, or through an alias. *)
Lemma has_source_into :
  D (Atom_has_source x) ->
  D (Atom_any_source x)
  ∨ ∃ from : node, D (Atom_alias x from) ∧ D (Atom_has_source from).
Proof.
  intros Hd. inversion Hd; subst; unfold_rels.
  - left. assumption.
  - right. eauto.
  - exfalso. eapply (ro_constructor _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_constructor _ _ _ _ _ Hro). eassumption.
  - exfalso. eapply (ro_parameter_base _ _ _ _ _ Hro). eassumption.
Qed.

End Read_only.

(* ======================================================================== *)
(* [check_apply_widgets]                                                    *)
(* ======================================================================== *)

(* An alias of a node [a] only read off [n] with the field of known
   arity is one of each node [y] read off [n] alike, unless [n] has any
   source. *)
Lemma read_of (is_local : field -> bool) (F : facts) (T : tables)
    (a n y from : node) :
  least_solution is_local F T ->
  is_local (Field_call_witness Known_arity_code_pointer) = false ->
  read_only F a (Field_call_witness Known_arity_code_pointer) n true ->
  (y, Field_call_witness Known_arity_code_pointer, n) ∈ f_accessor F ->
  mem_pair a from (t_alias T) = true ->
  mem_node n (t_any_source T) = true ∨ mem_pair y from (t_alias T) = true.
Proof.
  intros Hleast Hknown Hro Hacc Hal.
  pose proof (least_closed is_local F T Hleast) as Hc.
  apply (least_derivable is_local F T Hleast (Atom_alias a from)) in Hal.
  destruct (alias_into is_local F T a _ n true Hro Hknown from Hal)
    as [(b & Hb & Hu & Hg) | [(bs & Hs & Hg & Hb) | [_ Has]]].
  - right. apply (least_holds is_local F T Hleast) in Hu. unfold_rels.
    exact (alias_constructor_usage _ _ _ Hc b n _ y from Hb Hu Hg Hacc).
  - right. apply (least_holds is_local F T Hleast) in Hs. unfold_rels.
    exact (alias_accessor_source _ _ _ Hc n bs _ y from Hacc Hs Hg Hb).
  - left. exact (least_holds is_local F T Hleast _ Has).
Qed.

(* What one of two nodes only read off [n] with the field of known arity
   has of a least solution the other has: an alias of the one is one of
   the other, unless [n] has any source, when both have. *)
Lemma widget_any_source (is_local : field -> bool) (F : facts)
    (T : tables) (a a' n : node) :
  least_solution is_local F T ->
  is_local (Field_call_witness Known_arity_code_pointer) = false ->
  read_only F a (Field_call_witness Known_arity_code_pointer) n true ->
  (a', Field_call_witness Known_arity_code_pointer, n) ∈ f_accessor F ->
  mem_node a (t_any_source T) = true -> mem_node a' (t_any_source T) = true.
Proof.
  intros Hleast Hknown Hro Hacc' Has.
  pose proof (least_closed is_local F T Hleast) as Hc.
  apply (least_derivable is_local F T Hleast (Atom_any_source a)) in Has.
  destruct (any_source_into is_local F T a _ n true Hro Has)
    as [(from & Hal & Hfrom) | Hn].
  - apply (least_holds is_local F T Hleast) in Hal, Hfrom.
    destruct (read_of is_local F T a n a' from Hleast Hknown Hro Hacc' Hal)
      as [Hn | Hal'].
    + exact (any_source_accessor _ _ _ Hc n _ a' Hn Hacc' Hknown).
    + exact (any_source_alias _ _ _ Hc from a' Hal' Hfrom).
  - apply (least_holds is_local F T Hleast) in Hn. unfold_rels.
    exact (any_source_accessor _ _ _ Hc n _ a' Hn Hacc' Hknown).
Qed.

Lemma widget_sources (is_local : field -> bool) (F : facts)
    (T : tables) (a a' n y : node) :
  least_solution is_local F T ->
  is_local (Field_call_witness Known_arity_code_pointer) = false ->
  read_only F a (Field_call_witness Known_arity_code_pointer) n true ->
  read_only F a' (Field_call_witness Known_arity_code_pointer) n true ->
  (a, Field_call_witness Known_arity_code_pointer, n) ∈ f_accessor F ->
  (a', Field_call_witness Known_arity_code_pointer, n) ∈ f_accessor F ->
  mem_pair a y (t_sources T) = true -> mem_pair a' y (t_sources T) = true.
Proof.
  intros Hleast Hknown Hro Hro' Hacc Hacc' Hs.
  pose proof (least_closed is_local F T Hleast) as Hc.
  apply (least_derivable is_local F T Hleast (Atom_sources a y)) in Hs.
  destruct (sources_into is_local F T a _ n true Hro y Hs)
    as (from & Hfrom & Hgf & Hal & Hga).
  apply (least_holds is_local F T Hleast) in Hfrom, Hal. unfold_rels.
  destruct (read_of is_local F T a n a' from Hleast Hknown Hro Hacc' Hal)
    as [Hn | Hal'].
  - pose proof (any_source_accessor _ _ _ Hc n _ a Hn Hacc Hknown).
    unfold_rels. congruence.
  - assert (Hga' : mem_node a' (t_any_source T) = false).
    { destruct (mem_node a' (t_any_source T)) eqn:E; [ | reflexivity ].
      rewrite (widget_any_source is_local F T a' a n Hleast Hknown Hro' Hacc
                 E) in Hga.
      discriminate. }
    exact (sources_alias _ _ _ Hc from a' y Hfrom Hgf Hal' Hga').
Qed.

(* Two nodes read off one node with the field of known arity, each only
   read into, have the same [any_source] and the same [sources]. *)
Theorem least_apply_widgets : least_apply_widgets_statement.
Proof.
  intros is_local F T Hslots Hleast Hcheck.
  assert (Hknown :
            is_local (Field_call_witness Known_arity_code_pointer) = false).
  { destruct (is_local (Field_call_witness Known_arity_code_pointer))
      eqn:E; [ | reflexivity ].
    destruct (Hslots _ E) as [[w Hw] | [fs Hfs]]; discriminate. }
  unfold check_widgets_read_only in Hcheck.
  rewrite forallb_forall in Hcheck.
  unfold check_apply_widgets. apply forallb_forall.
  intros [[a f] n] Hacc. apply forallb_forall.
  intros [[a' f'] n'] Hacc'. cbn beta iota.
  destruct (field_eqb f (Field_call_witness Known_arity_code_pointer)
            && field_eqb f' f && node_eqb n' n) eqn:E; [ | reflexivity ].
  apply andb_true_iff in E as [E Hn]. apply andb_true_iff in E as [Hf Hf'].
  apply field_eqb_eq in Hf, Hf'. apply node_eqb_eq in Hn. subst.
  pose proof (Hcheck _ Hacc) as Hro. pose proof (Hcheck _ Hacc') as Hro'.
  cbn beta iota in Hro, Hro'. rewrite field_eqb_refl in Hro, Hro'.
  cbn in Hro, Hro'.
  apply only_read_into_read_only in Hro, Hro'.
  cbn [negb orb]. apply andb_true_iff. split.
  - apply Bool.eqb_true_iff.
    destruct (mem_node a (t_any_source T)) eqn:Ea,
      (mem_node a' (t_any_source T)) eqn:Ea'; try reflexivity.
    + rewrite (widget_any_source is_local F T a a' n Hleast Hknown Hro Hacc'
                 Ea) in Ea'.
      discriminate.
    + rewrite (widget_any_source is_local F T a' a n Hleast Hknown Hro' Hacc
                 Ea') in Ea.
      discriminate.
  - apply forallb_forall. intros [x y] Hin. cbn beta iota.
    destruct (node_eqb x a) eqn:Ex; [ | reflexivity ].
    apply node_eqb_eq in Ex. subst x. cbn.
    exact (widget_sources is_local F T a a' n y Hleast Hknown Hro Hro'
             Hacc Hacc' (proj2 (mem_pair_In _ _ _) Hin)).
Qed.

(* ======================================================================== *)
(* [check_unboxed_reads]                                                    *)
(* ======================================================================== *)

(* [Keyed.dedup] keeps every node. *)
Lemma in_dedup (l : list node) (x : node) :
  x ∈ l -> x ∈ Keyed.dedup node_eqb l.
Proof.
  induction l as [ | y l IH]; cbn; [ easy | ].
  intros [<- | Hin]; [ left; reflexivity | ].
  destruct (node_eq_dec y x) as [<- | Hne]; [ left; reflexivity | ].
  right. unfold Keyed.without. apply filter_In. split; [ exact (IH Hin) | ].
  cbn. unfold node_eqb. destruct (node_eq_dec x y); [ congruence | ].
  reflexivity.
Qed.

(* The rounds of [cannot_unbox0] only add. *)
Lemma stored_rounds_incl (F : facts) (T : tables) (crr : list node)
    (fuel : nat) (cu0 : list node) (x : node) :
  x ∈ cu0 -> x ∈ stored_rounds F T crr fuel cu0.
Proof.
  revert cu0. induction fuel as [ | fuel IH]; intros cu0 Hin;
    cbn [stored_rounds]; [ exact Hin | ].
  destruct (Keyed.dedup node_eqb (stored_round F T crr cu0)) as [ | y l];
    [ exact Hin | ].
  apply IH. apply in_or_app. right. exact Hin.
Qed.

(* An allocation to unbox has no [any_usage]: [any_usage] is a seed of
   [cannot_unbox0]. *)
Lemma to_unbox_no_any_usage (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (a : node) :
  mem_node a (derive_to_unbox is_local F T cannot_change max_unbox_size
                lambda_lifting)
  = true ->
  mem_node a (t_any_usage T) = false.
Proof.
  intros Hin. apply mem_node_In in Hin.
  unfold derive_to_unbox in Hin. apply filter_In in Hin as [_ Hcu].
  apply negb_true_iff in Hcu.
  destruct (mem_node a (t_any_usage T)) eqn:E; [ | reflexivity ].
  exfalso. apply mem_node_In in E.
  apply Bool.not_true_iff_false in Hcu. apply Hcu. apply mem_node_In.
  unfold unboxing_relations. cbn [ut_cu]. apply in_dedup.
  apply in_or_app. left. apply stored_rounds_incl. apply in_dedup.
  unfold derived_cannot_unbox0_seeds. apply in_or_app. right.
  apply in_or_app. left. exact E.
Qed.

(* A name dominated by [a] has [a] as a source with a source, no
   [any_source], and no other source with a source. *)
Lemma dominated_spec (T : tables) (a y : node) :
  (a, y) ∈ derive_dominated T ->
  mem_pair y a (t_sources T) = true ∧ mem_node a (t_has_source T) = true
  ∧ mem_node y (t_any_source T) = false
  ∧ ∀ z : node,
      mem_pair y z (t_sources T) = true -> mem_node z (t_has_source T) = true ->
      z = a.
Proof.
  unfold derive_dominated. intros Hin.
  apply in_flat_map in Hin as ([y' a'] & Hs & Hin).
  destruct (mem_node a' (t_has_source T)
            && negb (multiple_allocation_points T y')) eqn:E;
    [ | destruct Hin ].
  destruct Hin as [Heq | []]. injection Heq as -> ->.
  apply andb_true_iff in E as [Ha Hm]. apply negb_true_iff in Hm.
  unfold multiple_allocation_points in Hm.
  apply orb_false_iff in Hm as [Hany Hm].
  split; [ apply mem_pair_In; exact Hs | ].
  split; [ exact Ha | ]. split; [ exact Hany | ].
  intros z Hz Hhz. apply mem_pair_In in Hz.
  destruct (node_eq_dec z a) as [ | Hne]; [ assumption | exfalso ].
  assert (Hex : existsb
                  (fun '(x', y0) =>
                     node_eqb y x' && mem_node y0 (t_has_source T)
                     && existsb
                          (fun '(x'', z0) =>
                             node_eqb y x'' && mem_node z0 (t_has_source T)
                             && negb (node_eqb y0 z0))
                          (t_sources T))
                  (t_sources T)
                = true).
  { apply existsb_exists. exists (y, a). split; [ exact Hs | ].
    cbn beta iota. rewrite node_eqb_refl, Ha. cbn.
    apply existsb_exists. exists (y, z). split; [ exact Hz | ].
    cbn beta iota. rewrite node_eqb_refl, Hhz. cbn.
    unfold node_eqb. destruct (node_eq_dec a z); [ congruence | ].
    reflexivity. }
  rewrite Hex in Hm. discriminate.
Qed.

(* [get_single_field_source] finds a source of the field [f] of [a] if
   what [a] holds there has any source, or a source with a source. *)
Lemma single_field_source_found (F : facts) (T : tables) (a from : node)
    (f : field) :
  (a, f, from) ∈ f_constructor F ->
  mem_node from (t_any_source T) = true
  ∨ (∃ s : node,
       mem_pair from s (t_sources T) = true
       ∧ mem_node s (t_has_source T) = true) ->
  match get_single_field_source F T a f with
  | No_source => false
  | One_source _ | Many_sources => true
  end
  = true.
Proof.
  intros Hc Hsrc. unfold get_single_field_source.
  destruct (existsb _ (f_constructor F)) eqn:Eany; [ reflexivity | ].
  assert (Hno : mem_node from (t_any_source T) = false).
  { destruct (mem_node from (t_any_source T)) eqn:E; [ | reflexivity ].
    assert (Hex : existsb
                    (fun '(base, f', source) =>
                       node_eqb base a && field_eqb f f'
                       && mem_node source (t_any_source T))
                    (f_constructor F)
                  = true).
    { apply existsb_exists. exists (a, f, from). split; [ exact Hc | ].
      cbn beta iota. rewrite node_eqb_refl, field_eqb_refl, E.
      reflexivity. }
    rewrite Hex in Eany. discriminate. }
  destruct Hsrc as [Hany | (s & Hs & Hhs)]; [ congruence | ].
  match goal with
  | |- context [Keyed.dedup node_eqb ?l] => assert (Hin : s ∈ l)
  end.
  { apply in_flat_map. exists (a, f, from). split; [ exact Hc | ].
    cbn beta iota. rewrite node_eqb_refl, field_eqb_refl. cbn.
    apply in_flat_map. exists (from, s).
    split; [ apply mem_pair_In; exact Hs | ].
    cbn beta iota. rewrite node_eqb_refl, Hhs. left. reflexivity. }
  apply in_dedup in Hin.
  destruct (Keyed.dedup node_eqb _) as [ | y [ | z l]];
    [ destruct Hin | reflexivity | reflexivity ].
Qed.

Section Flow.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Hypothesis Hleast : least_solution is_local F T.

Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable is_local G S).

(* A node with a source has any source, or a source that has a source:
   each rule that gives a node a source gives it one of its own, or
   carries one along an alias, or gives it any source. *)
Lemma has_source_allocation (n : node) :
  D (Atom_has_source n) ->
  mem_node n (t_any_source T) = true
  ∨ ∃ s : node,
      mem_pair n s (t_sources T) = true ∧ mem_node s (t_has_source T) = true.
Proof.
  pose proof (least_closed is_local F T Hleast) as Hc.
  set (P := fun a : atom =>
         match a with
         | Atom_has_source n =>
           mem_node n (t_any_source T) = true
           ∨ ∃ s : node,
               mem_pair n s (t_sources T) = true
               ∧ mem_node s (t_has_source T) = true
         | _ => True
         end).
  intros Hd. change (P (Atom_has_source n)).
  induction Hd; cbn [P] in *; trivial.
  - apply (least_holds is_local F T Hleast) in Hd. left. exact Hd.
  - (* along an alias *)
    apply (least_holds is_local F T Hleast) in Hd2. unfold_rels.
    destruct (mem_node from (t_any_source T)) eqn:Ef.
    { left. exact (any_source_alias _ _ _ Hc from to_ Hd2 Ef). }
    destruct (mem_node to_ (t_any_source T)) eqn:Et; [ left; reflexivity | ].
    destruct IHHd1 as [Hany | (s & Hs & Hhs)]; [ congruence | ].
    right. exists s. split; [ | exact Hhs ].
    exact (sources_alias _ _ _ Hc from to_ s Hs Ef Hd2 Et).
  - (* a constructor: its own source *)
    destruct (mem_node base (t_any_source T)) eqn:Eb; [ left; reflexivity | ].
    right. exists base. split.
    + exact (sources_constructor _ _ _ Hc from relation base H Eb).
    + apply (least_holds is_local F T Hleast) in Hd. unfold_rels.
      exact (has_source_constructor _ _ _ Hc from relation base Hd H).
  - destruct (mem_node base (t_any_source T)) eqn:Eb; [ left; reflexivity | ].
    right. exists base. split.
    + exact (sources_constructor _ _ _ Hc from relation base H Eb).
    + exact (has_source_function_slot _ _ _ Hc from relation base H H0).
  - destruct (mem_node base (t_any_source T)) eqn:Eb; [ left; reflexivity | ].
    right. exists base. split.
    + exact (sources_parameter _ _ _ Hc to_ relation base H Eb).
    + apply (least_holds is_local F T Hleast) in Hd. unfold_rels.
      exact (has_source_parameter _ _ _ Hc to_ relation base Hd H).
Qed.

(* [any_usage] goes back along the aliases that carry a source. *)
Lemma sources_any_usage (y s : node) :
  D (Atom_sources y s) ->
  mem_node y (t_any_usage T) = true -> mem_node s (t_any_usage T) = true.
Proof.
  pose proof (least_closed is_local F T Hleast) as Hc.
  set (P := fun a : atom =>
         match a with
         | Atom_sources y s =>
           mem_node y (t_any_usage T) = true ->
           mem_node s (t_any_usage T) = true
         | _ => True
         end).
  intros Hd. change (P (Atom_sources y s)).
  induction Hd; cbn [P] in *; trivial.
  intros Hto. apply IHHd1.
  apply (least_holds is_local F T Hleast) in Hd2. unfold_rels.
  exact (any_usage_alias _ _ _ Hc to_ from Hto Hd2).
Qed.

(* What [usages b u] says: [u] is reached from [b] along aliases, so
   that it has any source if [b] has, and the sources of [b] otherwise. *)
Lemma usages_flow (b u : node) :
  D (Atom_usages b u) ->
  (mem_node b (t_any_source T) = true -> mem_node u (t_any_source T) = true)
  ∧ ∀ s : node,
      mem_pair b s (t_sources T) = true ->
      mem_node u (t_any_source T) = true ∨ mem_pair u s (t_sources T) = true.
Proof.
  pose proof (least_closed is_local F T Hleast) as Hc.
  set (P := fun a : atom =>
         match a with
         | Atom_usages b u =>
           (mem_node b (t_any_source T) = true ->
            mem_node u (t_any_source T) = true)
           ∧ ∀ s : node,
               mem_pair b s (t_sources T) = true ->
               mem_node u (t_any_source T) = true
               ∨ mem_pair u s (t_sources T) = true
         | _ => True
         end).
  intros Hd. change (P (Atom_usages b u)).
  induction Hd; cbn [P] in *; trivial.
  - split; [ exact (fun H => H) | intros s Hs; right; exact Hs ].
  - split; [ exact (fun H => H) | intros s Hs; right; exact Hs ].
  - destruct IHHd1 as [IHany IHsrc].
    apply (least_holds is_local F T Hleast) in Hd2. unfold_rels.
    assert (Hany : mem_node from (t_any_source T) = true ->
                   mem_node usage (t_any_source T) = true).
    { intros Hf. apply IHany.
      exact (any_source_alias _ _ _ Hc from to_ Hd2 Hf). }
    split; [ exact Hany | ].
    intros s Hs.
    destruct (mem_node from (t_any_source T)) eqn:Ef;
      [ left; exact (Hany eq_refl) | ].
    destruct (mem_node to_ (t_any_source T)) eqn:Et;
      [ left; exact (IHany eq_refl) | ].
    apply IHsrc. exact (sources_alias _ _ _ Hc from to_ s Hs Ef Hd2 Et).
Qed.

End Flow.

(* A read [x] of a node [y] that the allocation [a] to unbox dominates:
   [y] has no [any_usage], as [a] would have it; and the source of [x]
   comes, through the one alias the read gives it, from a field [f] of
   an allocation that [y] may come from, which is [a], the one source of
   [y] that has a source. *)
Theorem least_unboxed_reads : least_unboxed_reads_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting Hleast
    Hcheck.
  pose proof (least_closed is_local F T Hleast) as Hc.
  set (to_unbox := derive_to_unbox is_local F T cannot_change
                     max_unbox_size lambda_lifting) in *.
  unfold check_reads_read_only in Hcheck. rewrite forallb_forall in Hcheck.
  unfold check_unboxed_reads. apply forallb_forall.
  intros [[x f] y] Hacc. cbn beta iota.
  destruct (mem_node x (t_has_source T) && mem_node x (t_has_usage T)
            && negb (has_to_be_unboxed (derive_dominated T) to_unbox x))
    eqn:Hx; [ | reflexivity ].
  apply andb_true_iff in Hx as [Hx _]. apply andb_true_iff in Hx as [Hhs _].
  cbn [negb orb]. apply forallb_forall. intros [a y'] Hd. cbn beta iota.
  destruct (node_eqb y y' && mem_node a to_unbox) eqn:Ha;
    [ | reflexivity ].
  apply andb_true_iff in Ha as [Hy Ha]. apply node_eqb_eq in Hy. subst y'.
  destruct (dominated_spec T a y Hd) as (Hya & Hha & Hnoany & Huniq).
  (* The read of [y] is only read into. *)
  assert (Hunboxed : has_to_be_unboxed (derive_dominated T) to_unbox y
                     = true).
  { unfold has_to_be_unboxed. apply existsb_exists. exists (a, y).
    split; [ exact Hd | ]. cbn beta iota. rewrite node_eqb_refl, Ha.
    reflexivity. }
  pose proof (Hcheck _ Hacc) as Hro. cbn beta iota in Hro.
  rewrite Hunboxed in Hro. cbn [negb orb] in Hro.
  apply andb_true_iff in Hro as [Hnl Hro]. apply negb_true_iff in Hnl.
  apply only_read_into_read_only in Hro.
  apply andb_true_iff. split.
  - (* [y] has a source [a] that would have [any_usage] *)
    apply negb_true_iff.
    destruct (mem_node y (t_any_usage T)) eqn:Hyu; [ | reflexivity ].
    exfalso.
    apply (least_derivable is_local F T Hleast (Atom_sources y a)) in Hya.
    pose proof (sources_any_usage is_local F T Hleast y a Hya Hyu) as Hau.
    rewrite (to_unbox_no_any_usage is_local F T cannot_change
               max_unbox_size lambda_lifting a Ha) in Hau.
    discriminate.
  - (* [x] has its source from the field [f] of [a] *)
    apply (least_derivable is_local F T Hleast (Atom_has_source x)) in Hhs.
    assert (Hal : ∃ from : node,
                    derivable is_local (graph_of_facts F)
                      (solution_of_tables T) (Atom_alias x from)
                    ∧ mem_node from (t_has_source T) = true).
    { destruct (has_source_into is_local F T x f y false Hro Hhs)
        as [Hany | (from & Hal & Hfrom)].
      - destruct (any_source_into is_local F T x f y false Hro Hany)
          as [(from & Hal & Hfrom) | Hyany].
        + exists from. split; [ exact Hal | ].
          apply (least_holds is_local F T Hleast) in Hfrom. unfold_rels.
          exact (has_source_any _ _ _ Hc from Hfrom).
        + apply (least_holds is_local F T Hleast) in Hyany. unfold_rels.
          congruence.
      - exists from. split; [ exact Hal | ].
        exact (least_holds is_local F T Hleast _ Hfrom). }
    destruct Hal as (from & Hal & Hfrom).
    assert (Hcons : (a, f, from) ∈ f_constructor F).
    { destruct (alias_into is_local F T x f y false Hro Hnl from Hal)
        as [(b & Hb & Hu & Hg) | [(bs & Hs & Hg & Hb) | [Habs _]]];
        [ | | discriminate ].
      - (* [b] flows to [y]: a source of [y] with a source *)
        apply (least_holds is_local F T Hleast) in Hu as Hu'.
        destruct (usages_flow is_local F T Hleast b y Hu) as [Hfa Hfs].
        assert (Hbs : mem_node b (t_any_source T) = false).
        { destruct (mem_node b (t_any_source T)) eqn:E; [ | reflexivity ].
          rewrite (Hfa eq_refl) in Hnoany. discriminate. }
        pose proof (sources_constructor _ _ _ Hc from f b Hb Hbs) as Hbb.
        unfold_rels.
        destruct (Hfs b Hbb) as [Hyany | Hyb]; [ congruence | ].
        rewrite (Huniq b Hyb
                   (has_source_constructor _ _ _ Hc from f b Hfrom Hb))
          in Hb.
        exact Hb.
      - (* [bs] is a source of [y] with a source *)
        apply (least_holds is_local F T Hleast) in Hs. unfold_rels.
        rewrite (Huniq bs Hs
                   (has_source_constructor _ _ _ Hc from f bs Hfrom Hb))
          in Hb.
        exact Hb. }
    apply (single_field_source_found F T a from f Hcons).
    apply (has_source_allocation is_local F T Hleast from).
    exact (least_derivable is_local F T Hleast (Atom_has_source from) Hfrom).
Qed.

(* ======================================================================== *)
(* The check and the computation                                            *)
(* ======================================================================== *)

(* [add_new] adds only the new elements. *)
Lemma add_new_in {A : Type} (eqb : A -> A -> bool) (old new : list A)
    (x : A) :
  x ∈ add_new eqb old new -> x ∈ old ∨ x ∈ new.
Proof.
  unfold add_new. revert old.
  induction new as [ | y new IH]; intros old Hin; cbn [fold_left] in Hin;
    [ left; exact Hin | ].
  apply IH in Hin as [Hin | Hin]; [ | right; right; exact Hin ].
  destruct (existsb (eqb y) old); [ left; exact Hin | ].
  destruct Hin as [<- | Hin]; [ right; left; reflexivity | left; exact Hin ].
Qed.

(* The membership of a list built by [new_alias] and its kind, broken
   into the case of each rule: the facts and the tuples it is made of,
   and the tests that held. *)
Ltac break_in H :=
  repeat match type of H with
  | _ ∈ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | _ ∈ flat_map _ _ =>
      let t := fresh "t" in let Ht := fresh "Hin" in
      apply in_flat_map in H; destruct H as (t & Ht & H);
      repeat match goal with p : (_ * _)%type |- _ => destruct p end;
      cbn beta iota in H
  | _ ∈ when ?b _ =>
      let E := fresh "E" in
      destruct b eqn:E; [ cbn [when] in H | destruct H ]
  | _ ∈ [_] => destruct H as [H | []]
  | _ ∈ [] => destruct H
  end.

Ltac clean_tests :=
  repeat match goal with
  | H : (_, _) = (_, _) |- _ => injection H as <- <-
  | H : (_, _, _) = (_, _, _) |- _ => injection H as <- <- <-
  | H : _ && _ = true |- _ => apply andb_true_iff in H as [? ?]
  | H : negb _ = true |- _ => apply negb_true_iff in H
  | H : node_eqb _ _ = true |- _ => apply node_eqb_eq in H; subst
  | H : field_eqb _ _ = true |- _ => apply field_eqb_eq in H; subst
  | H : Nat.eqb _ _ = true |- _ => apply Nat.eqb_eq in H; subst
  | H : ?a = ?b |- _ => is_var a; is_var b; subst a
  end.

(* A premise of a rule: a fact of the graph, a tuple of the tables,
   which [Hall] derives, or a test that held. *)
Ltac solve_premise Hall :=
  unfold_rels;
  first
    [ eassumption
    | apply Hall; unfold_rels;
      first
        [ eassumption
        | apply mem_pair_In; eassumption
        | apply mem_field_node_In; eassumption
        | apply mem_node_In; eassumption
        | apply mem_node_field_In; eassumption ] ].

Ltac derive_with Hall :=
  first
    [ eapply by_alias_input; solve [ repeat solve_premise Hall ]
    | eapply by_alias_propagate; solve [ repeat solve_premise Hall ]
    | eapply by_alias_if_any_source; solve [ repeat solve_premise Hall ]
    | eapply by_alias_constructor_usage; solve [ repeat solve_premise Hall ]
    | eapply by_alias_local_constructor_usage;
      solve [ repeat solve_premise Hall ]
    | eapply by_alias_parameter_usage; solve [ repeat solve_premise Hall ]
    | eapply by_alias_accessor_source; solve [ repeat solve_premise Hall ]
    | eapply by_alias_local_accessor_source;
      solve [ repeat solve_premise Hall ]
    | eapply by_alias_escaping_reading; solve [ repeat solve_premise Hall ]
    | eapply by_alias_argument_source; solve [ repeat solve_premise Hall ]
    | eapply by_has_usage_any; solve [ repeat solve_premise Hall ]
    | eapply by_has_usage_alias; solve [ repeat solve_premise Hall ]
    | eapply by_has_usage_accessor; solve [ repeat solve_premise Hall ]
    | eapply by_has_usage_argument; solve [ repeat solve_premise Hall ]
    | eapply by_has_source_any; solve [ repeat solve_premise Hall ]
    | eapply by_has_source_alias; solve [ repeat solve_premise Hall ]
    | eapply by_has_source_constructor; solve [ repeat solve_premise Hall ]
    | eapply by_has_source_function_slot; solve [ repeat solve_premise Hall ]
    | eapply by_has_source_parameter; solve [ repeat solve_premise Hall ]
    | eapply by_any_usage_use; solve [ repeat solve_premise Hall ]
    | eapply by_any_usage_alias; solve [ repeat solve_premise Hall ]
    | eapply by_any_usage_constructor; solve [ repeat solve_premise Hall ]
    | eapply by_any_usage_argument; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_input; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_zero_alloc; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_if_used; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_alias; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_parameter; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_accessor; solve [ repeat solve_premise Hall ]
    | eapply by_any_source_use; solve [ repeat solve_premise Hall ]
    | eapply by_any_usage_input; solve [ repeat solve_premise Hall ]
    | eapply by_usages_accessor; solve [ repeat solve_premise Hall ]
    | eapply by_usages_argument; solve [ repeat solve_premise Hall ]
    | eapply by_usages_alias; solve [ repeat solve_premise Hall ]
    | eapply by_sources_constructor; solve [ repeat solve_premise Hall ]
    | eapply by_sources_parameter; solve [ repeat solve_premise Hall ]
    | eapply by_sources_alias; solve [ repeat solve_premise Hall ]
    | eapply by_escaping_field_rule; solve [ repeat solve_premise Hall ]
    | eapply by_reading_field_rule; solve [ repeat solve_premise Hall ]
    | eapply by_field_used_any_usage; solve [ repeat solve_premise Hall ]
    | eapply by_field_used_usage; solve [ repeat solve_premise Hall ]
    | eapply by_field_used_reading; solve [ repeat solve_premise Hall ]
    | eapply by_field_used_local_source;
      solve [ repeat solve_premise Hall ] ].

(* A step adds only what the rules derive, with the guards of [Tg], from
   tuples that they derive: each list of [new_alias] and its kind is the
   conclusion of one rule, at the facts and tuples its premises name. *)
Lemma step_derivable (is_local : field -> bool) (F : facts)
    (Tg T : tables) :
  (∀ a : atom,
     holds (solution_of_tables T) a = true ->
     derivable is_local (graph_of_facts F) (solution_of_tables Tg) a) ->
  ∀ a : atom,
    holds (solution_of_tables (step is_local F Tg T)) a = true ->
    derivable is_local (graph_of_facts F) (solution_of_tables Tg) a.
Proof.
  intros Hall a Ha.
  destruct a; unfold_rels; cbn [step t_alias t_any_usage t_any_source
    t_has_usage t_has_source t_usages t_sources t_reading_field
    t_escaping_field t_field_used] in Ha;
  first
    [ apply mem_pair_In in Ha | apply mem_node_In in Ha
    | apply mem_field_node_In in Ha | apply mem_node_field_In in Ha ];
  apply add_new_in in Ha as [Ha | Ha];
  try solve [ apply Hall; unfold_rels;
              first
                [ apply mem_pair_In; exact Ha | apply mem_node_In; exact Ha
                | apply mem_field_node_In; exact Ha
                | apply mem_node_field_In; exact Ha ] ];
  [ unfold new_alias in Ha | unfold new_any_usage in Ha
  | unfold new_any_source in Ha | unfold new_has_usage in Ha
  | unfold new_has_source in Ha | unfold new_usages in Ha
  | unfold new_sources in Ha | unfold new_reading_field in Ha
  | unfold new_escaping_field in Ha | unfold new_field_used in Ha ];
  break_in Ha; clean_tests; derive_with Hall.
Qed.

(* The steps of a saturation add only what the rules derive. *)
Lemma saturate_derivable (is_local : field -> bool) (F : facts)
    (Tg : tables) (fuel : nat) (T : tables) :
  (∀ a : atom,
     holds (solution_of_tables T) a = true ->
     derivable is_local (graph_of_facts F) (solution_of_tables Tg) a) ->
  ∀ a : atom,
    holds (solution_of_tables (saturate is_local F Tg fuel T)) a = true ->
    derivable is_local (graph_of_facts F) (solution_of_tables Tg) a.
Proof.
  revert T. induction fuel as [ | fuel IH]; intros T Hall;
    cbn [saturate]; [ exact Hall | ].
  destruct (Nat.eqb _ _); [ exact Hall | ].
  apply IH. apply step_derivable. exact Hall.
Qed.

(* The empty tables have no tuple to derive. *)
Lemma empty_tables_derivable (is_local : field -> bool) (G : graph)
    (S : solution) (a : atom) :
  holds (solution_of_tables empty_tables) a = true ->
  derivable is_local G S a.
Proof. destruct a; discriminate. Qed.

(* [tables_incl], read back: each tuple of [T] is one of [T']. *)
Lemma tables_incl_holds (T T' : tables) (a : atom) :
  tables_incl T T' = true ->
  holds (solution_of_tables T) a = true ->
  holds (solution_of_tables T') a = true.
Proof.
  unfold tables_incl. intros H.
  repeat match type of H with
  | _ && _ = true =>
      let H' := fresh "Hi" in apply andb_true_iff in H as [H H']
  end.
  repeat match goal with
  | Hi : forallb _ _ = true |- _ => rewrite forallb_forall in Hi
  end.
  destruct a; unfold_rels; intros Ha;
  first
    [ apply mem_pair_In in Ha | apply mem_node_In in Ha
    | apply mem_field_node_In in Ha | apply mem_node_field_In in Ha ];
  match goal with
  | Hi : ∀ x, x ∈ ?l -> _ |- _ =>
      lazymatch type of Ha with _ ∈ l => exact (Hi _ Ha) end
  end.
Qed.

(* The tuples of [T] are among those that the rules derive from nothing
   with the guards of [T]. *)
Theorem check_least_sound : check_least_sound_statement.
Proof.
  intros is_local F fuel T Hcheck. unfold check_least in Hcheck.
  apply andb_true_iff in Hcheck as [Hclosed Hincl].
  split; [ exact Hclosed | ].
  intros a Ha.
  apply (saturate_derivable is_local F T fuel empty_tables).
  - intros a' Ha'. exact (empty_tables_derivable _ _ _ _ Ha').
  - exact (tables_incl_holds _ _ a Hincl Ha).
Qed.

(* What [alternate] gives is the saturation with the guards of some
   tables, closed, with no guard that those tables do not have. *)
Lemma alternate_some (is_local : field -> bool) (F : facts)
    (fuel rounds : nat) (Tg T : tables) :
  alternate is_local F fuel rounds Tg = Some T ->
  ∃ Tg' : tables,
    T = saturate is_local F Tg' fuel empty_tables
    ∧ check_closed is_local F T = true ∧ tops_within T Tg' = true.
Proof.
  revert Tg. induction rounds as [ | rounds IH]; intros Tg Halt;
    cbn [alternate] in Halt;
    (destruct (check_closed is_local F
                 (saturate is_local F Tg fuel empty_tables)
               && tops_within (saturate is_local F Tg fuel empty_tables) Tg)
       eqn:E;
     [ injection Halt as <-; apply andb_true_iff in E as [Hc Ht];
       exists Tg; auto
     | ]).
  - discriminate.
  - exact (IH _ Halt).
Qed.

(* The tuples that [alternate] returns are derived with the guards of
   tables that have every guard they have, so with their own. *)
Theorem compute_least_sound : compute_least_sound_statement.
Proof.
  intros is_local F fuel T Hcomp. unfold compute_least in Hcomp.
  apply alternate_some in Hcomp as (Tg & -> & Hclosed & Htops).
  split; [ exact Hclosed | ].
  unfold tops_within in Htops.
  apply andb_true_iff in Htops as [Htu Hts].
  rewrite forallb_forall in Htu, Hts.
  intros a Ha.
  apply (derivable_mono is_local (graph_of_facts F) (solution_of_tables Tg)).
  - intros x Hx. unfold_rels. apply Htu. apply mem_node_In. exact Hx.
  - intros x Hx. unfold_rels. apply Hts. apply mem_node_In. exact Hx.
  - apply (saturate_derivable is_local F Tg fuel empty_tables);
      [ intros a' Ha'; exact (empty_tables_derivable _ _ _ _ Ha') | ].
    exact Ha.
Qed.

(* ======================================================================== *)
(* The examples                                                             *)
(* ======================================================================== *)

Theorem least_examples : least_examples_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* The least tables of the counterexamples pass [check_least]; the rest
   is computed. *)
Ltac split_least Hleast :=
  repeat match goal with
  | |- least_solution _ _ _ => exact Hleast
  | |- _ ∧ _ => split
  end;
  vm_compute; reflexivity.

Theorem widgets_counterexample : widgets_counterexample_statement.
Proof.
  assert (Hleast : least_solution Examples.no_local
                     WidgetExample.unit_facts WidgetExample.least).
  { apply (check_least_sound _ _ 100). vm_compute. reflexivity. }
  unfold widgets_counterexample_statement. split_least Hleast.
Qed.

Theorem reads_counterexample : reads_counterexample_statement.
Proof.
  assert (Hleast : least_solution Examples.no_local
                     ShadowedReadExample.unit_facts
                     ShadowedReadExample.least).
  { apply (check_least_sound _ _ 100). vm_compute. reflexivity. }
  unfold reads_counterexample_statement. cbv zeta. split_least Hleast.
Qed.

(* ======================================================================== *)
(* The statements of the rebuild for the least solution                     *)
(* ======================================================================== *)

(* The statements of the rebuild, whose checks of the solution a least
   solution passes. *)
Theorem least_rebuild_refines_observably :
  least_rebuild_refines_observably_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls Hslots Hlocal Hfrag
    Hstatic Hkinded Hwf Hexits Hleast.
  apply (FlambdaReaperRebuildProofs.rebuild_refines_observably flag
           is_local); auto.
  - exact (proj1 Hleast).
  - exact (least_field_used_exact _ _ _ Hleast).
Qed.

Theorem least_rebuild_cc_refines_observably :
  least_rebuild_cc_refines_observably_statement.
Proof.
  intros flag is_local u e T preserve_direct_calls Hslots Hlocal Hfrag
    Hstatic Hkinded Hwf Hexits Hleast Hwidgets.
  apply (FlambdaReaperCallingConventionsProofs.rebuild_cc_refines_observably
           flag is_local); auto.
  - exact (proj1 Hleast).
  - exact (least_field_used_exact _ _ _ Hleast).
  - exact (proj1
             (FlambdaReaperCallingConventionsProofs.derive_cannot_change_least
                _ _)).
  - exact (least_apply_widgets _ _ _ Hslots Hleast Hwidgets).
Qed.

Theorem least_rebuild_unbox_refines_observably :
  least_rebuild_unbox_refines_observably_statement.
Proof.
  intros flag is_local u e T max_unbox_size lambda_lifting
    preserve_direct_calls Hslots Hlocal Hfrag Hstatic Hkinded Hwf Hexits
    Hleast Hunboxing Hreads.
  apply (FlambdaReaperUnboxingRefinesProofs.rebuild_unbox_refines_observably
           flag is_local); auto.
  - exact (proj1 Hleast).
  - exact (least_field_used_exact _ _ _ Hleast).
  - exact (least_unboxed_reads _ _ _ _ _ _ Hleast Hreads).
Qed.
