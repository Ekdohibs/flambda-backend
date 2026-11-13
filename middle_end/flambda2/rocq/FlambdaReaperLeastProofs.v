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

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperLeast
  FlambdaReaperLeastSpec.
From OxCamlFlambda2 Require FlambdaMapProofs FlambdaReaperFlowProofs
  FlambdaReaperRebuildProofs FlambdaReaperCallingConventionsProofs
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
(* The traversal with [Auto] reads each widget only                         *)
(* ======================================================================== *)

(* The node that a call of known arity reads off its callee (its widget)
   is made by the call, and with [Auto] the delayed facts of the call go
   to two other nodes that the call makes ([add_apply]). The invariant
   of the accumulator ([winv]): the widgets are made nodes, and nothing
   puts anything into one but its read and an alias that holds where its
   callee has any source; the nodes that facts put something into are
   made before the counter; and the nodes of the deps into which the
   delayed facts put something are not widgets ([good]). A step adds
   facts that put something only into nodes that are not made, good at
   the start of the step or made by it ([ext]); the call of known arity
   is the one step that makes a widget ([tstep_call_kind]). *)
Module AutoWidgets.

Abbreviation known := (Field_call_witness Known_arity_code_pointer).
Abbreviation env_conts := FlambdaReaperTraversal.env_conts.

(* The nodes that a fact puts where [only_read_into] looks. *)
Definition into_nodes (fct : fact) : list node :=
  match fct with
  | Fact_alias to_ _ | Fact_use to_ _ | Fact_propagate _ to_ _
  | Fact_alias_if_any_source _ to_ _ | Fact_accessor to_ _ _ => [to_]
  | Fact_parameter base _ to_ => [base; to_]
  | Fact_constructor base _ _ => [base]
  | Fact_any_source x | Fact_any_source_if_used _ x => [x]
  | Fact_argument _ _ _ | Fact_any_usage _ => []
  end.

(* What [only_read_into] lets a fact do to a widget read off [n]. *)
Definition widget_fact_ok (fct : fact) (n : node) : Prop :=
  match fct with
  | Fact_accessor _ f base => f = known ∧ base = n
  | Fact_alias_if_any_source c _ _ => c = n
  | _ => False
  end.

Definition made_lt (x : node) (p : positive) : Prop :=
  ∀ k, x = Node_made k -> (k < p)%positive.

(* [x] is read off a node with the field of known arity. *)
Definition widget (a : acc) (x : node) : Prop :=
  ∃ n, Fact_accessor x known n ∈ acc_facts a.

(* [x] is made before [a] and is not a widget of it. *)
Definition good (a : acc) (x : node) : Prop :=
  made_lt x (acc_next a) ∧ ~ widget a x.

Definition not_made (x : node) : Prop := ∀ k, x <> Node_made k.

(* [x] is not made, or good in [a0], or made from [a0] to [a]. *)
Definition ok_node (a0 a : acc) (x : node) : Prop :=
  not_made x ∨ good a0 x
  ∨ ∃ k, x = Node_made k ∧ (acc_next a0 <= k)%positive
         ∧ (k < acc_next a)%positive.

(* The nodes of a code dep that facts put something into. *)
Definition code_dep_nodes (cd : code_dep) : list node :=
  cd_my_closure cd :: cd_exn cd :: cd_return cd.

Record winv (a : acc) : Prop := {
  wi_into : ∀ fct x,
    fct ∈ acc_facts a -> x ∈ into_nodes fct -> made_lt x (acc_next a);
  wi_made : ∀ w, widget a w -> ∃ k, w = Node_made k;
  wi_widget : ∀ w n fct,
    Fact_accessor w known n ∈ acc_facts a -> fct ∈ acc_facts a ->
    w ∈ into_nodes fct -> widget_fact_ok fct n;
  wi_code_deps : ∀ id cd x,
    (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd -> good a x;
  wi_apply_deps : ∀ d,
    d ∈ acc_apply_deps a -> good a (ad_apply_call_witness d);
  wi_closure_deps : ∀ d,
    d ∈ acc_closure_deps a -> good a (cld_let_bound_name_of_the_closure d)
}.

Definition env_good (env : traverse_env) (a : acc) : Prop :=
  ∀ k params x, (k, params) ∈ env_conts env -> x ∈ params -> good a x.

(* The widgets of [a'] are those of [a] and nodes made from [a]. *)
Definition grows (a a' : acc) : Prop :=
  (acc_next a <= acc_next a')%positive
  ∧ ∀ w, widget a' w ->
      widget a w ∨ ∃ k, w = Node_made k ∧ (acc_next a <= k)%positive.

(* From [a] to [a'], facts are added that put something into nodes that
   are not made, good in [a0] or made from [a0], and no widget; and deps
   of such nodes. *)
Record ext (a0 a a' : acc) : Prop := {
  ext_next : (acc_next a <= acc_next a')%positive;
  ext_facts : ∃ L, acc_facts a' = L ++ acc_facts a
    ∧ (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a' x)
    ∧ (∀ w n, Fact_accessor w known n ∉ L);
  ext_code_deps : ∀ id cd, (id, cd) ∈ acc_code_deps a' ->
    (id, cd) ∈ acc_code_deps a
    ∨ ∀ x, x ∈ code_dep_nodes cd -> ok_node a0 a' x;
  ext_apply_deps : ∀ d, d ∈ acc_apply_deps a' ->
    d ∈ acc_apply_deps a ∨ ok_node a0 a' (ad_apply_call_witness d);
  ext_closure_deps : ∀ d, d ∈ acc_closure_deps a' ->
    d ∈ acc_closure_deps a
    ∨ ok_node a0 a' (cld_let_bound_name_of_the_closure d)
}.

(* ------------------------------------------------------------------------ *)
(* Basic lemmas                                                             *)
(* ------------------------------------------------------------------------ *)

Lemma made_lt_mono (x : node) (p q : positive) :
  made_lt x p -> (p <= q)%positive -> made_lt x q.
Proof. intros H Hle k Hk. specialize (H k Hk). lia. Qed.

Lemma not_made_simple (s : simple) : not_made (node_of_simple s).
Proof. intros k. destruct s; discriminate. Qed.

Lemma ok_node_mono (a0 a a' : acc) (x : node) :
  (acc_next a <= acc_next a')%positive -> ok_node a0 a x -> ok_node a0 a' x.
Proof.
  intros Hle [H | [H | (k & -> & H1 & H2)]]; [ left | right; left | ];
    auto.
  right; right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma ok_simple (a0 a : acc) (s : simple) : ok_node a0 a (node_of_simple s).
Proof. left. apply not_made_simple. Qed.

Lemma ok_not_made (a0 a : acc) (x : node) : not_made x -> ok_node a0 a x.
Proof. intros H. left. exact H. Qed.

Lemma ok_good (a0 a : acc) (x : node) : good a0 x -> ok_node a0 a x.
Proof. intros H. right; left. exact H. Qed.

Lemma ok_made (a0 a : acc) (k : positive) :
  (acc_next a0 <= k)%positive -> (k < acc_next a)%positive ->
  ok_node a0 a (Node_made k).
Proof. intros H1 H2. right; right. exists k. auto. Qed.

Lemma ext_trans (a0 a a1 a2 : acc) :
  ext a0 a a1 -> ext a0 a1 a2 -> ext a0 a a2.
Proof.
  intros [Hn1 (L1 & Hf1 & Hi1 & Hw1) Hc1 Hap1 Hcl1]
    [Hn2 (L2 & Hf2 & Hi2 & Hw2) Hc2 Hap2 Hcl2].
  constructor; [ lia | | | | ].
  - exists (L2 ++ L1). split; [ rewrite Hf2, Hf1, app_assoc; reflexivity | ].
    split.
    + intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hi2 _ _ Hin Hx) | ].
      exact (ok_node_mono _ _ _ _ Hn2 (Hi1 _ _ Hin Hx)).
    + intros w n Hin. apply in_app_or in Hin as [Hin | Hin];
        [ exact (Hw2 _ _ Hin) | exact (Hw1 _ _ Hin) ].
  - intros id cd Hin. destruct (Hc2 _ _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hc1 _ _ Hin') as [H' | H']; [ left; auto | right ].
    intros x Hx. exact (ok_node_mono _ _ _ _ Hn2 (H' x Hx)).
  - intros d Hin. destruct (Hap2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hap1 _ Hin') as [H' | H']; [ left; auto | right ].
    exact (ok_node_mono _ _ _ _ Hn2 H').
  - intros d Hin. destruct (Hcl2 _ Hin) as [Hin' | H]; [ | right; auto ].
    destruct (Hcl1 _ Hin') as [H' | H']; [ left; auto | right ].
    exact (ok_node_mono _ _ _ _ Hn2 H').
Qed.

Lemma ext_same (a0 a a' : acc) :
  (acc_next a <= acc_next a')%positive ->
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> ext a0 a a'.
Proof.
  intros Hn Hf Hc Hap Hcl. constructor; auto.
  - exists []. rewrite Hf. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id cd Hin. left. rewrite <- Hc. exact Hin.
  - intros d Hin. left. rewrite <- Hap. exact Hin.
  - intros d Hin. left. rewrite <- Hcl. exact Hin.
Qed.

Lemma ext_refl (a0 a : acc) : ext a0 a a.
Proof. apply ext_same; (reflexivity || lia). Qed.

Lemma ext_add_facts (a0 a : acc) (L : list fact) :
  (∀ fct x, fct ∈ L -> x ∈ into_nodes fct -> ok_node a0 a x) ->
  (∀ w n, Fact_accessor w known n ∉ L) ->
  ext a0 a (add_facts L a).
Proof.
  intros Hi Hw. constructor; cbn [acc_next acc_facts acc_code_deps
    acc_apply_deps acc_closure_deps add_facts]; [ lia | | | | ].
  - exists L. split; [ reflexivity | split; assumption ].
  - intros id cd Hin. left. exact Hin.
  - intros d Hin. left. exact Hin.
  - intros d Hin. left. exact Hin.
Qed.

Lemma ext_add_fact (a0 a : acc) (fct : fact) :
  (∀ x, x ∈ into_nodes fct -> ok_node a0 a x) ->
  (∀ w n, fct <> Fact_accessor w known n) ->
  ext a0 a (add_fact fct a).
Proof.
  intros Hi Hw. apply ext_add_facts.
  - intros fct' x [<- | []] Hx. exact (Hi x Hx).
  - intros w n [H | []]. exact (Hw w n H).
Qed.

Lemma ext_fresh (a0 a : acc) : ext a0 a (snd (fresh a)).
Proof. apply ext_same; cbn; (reflexivity || lia). Qed.

Lemma fresh_fst (a : acc) : fst (fresh a) = Node_made (acc_next a).
Proof. reflexivity. Qed.

Lemma fresh_next (a : acc) :
  acc_next (snd (fresh a)) = Pos.succ (acc_next a).
Proof. reflexivity. Qed.

Lemma ok_fresh (a0 a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ok_node a0 (snd (fresh a)) (fst (fresh a)).
Proof. intros H. apply ok_made; cbn; lia. Qed.

Lemma ext_note_simple (a0 : acc) (s : simple) (a : acc) :
  ext a0 a (note_simple s a).
Proof. destruct s; apply ext_same; cbn; (reflexivity || lia). Qed.

Lemma note_simples_fields (l : list simple) (a : acc) :
  acc_next (note_simples l a) = acc_next a
  ∧ acc_facts (note_simples l a) = acc_facts a
  ∧ acc_code_deps (note_simples l a) = acc_code_deps a
  ∧ acc_apply_deps (note_simples l a) = acc_apply_deps a
  ∧ acc_closure_deps (note_simples l a) = acc_closure_deps a.
Proof.
  unfold note_simples. revert a. induction l as [ | s l IH]; intros a;
    [ auto | ].
  cbn -[note_simple].
  destruct (IH (note_simple s a)) as (H1 & H2 & H3 & H4 & H5).
  rewrite H1, H2, H3, H4, H5. destruct s; cbn; auto.
Qed.

Lemma ext_note_simples (a0 : acc) (l : list simple) (a : acc) :
  ext a0 a (note_simples l a).
Proof.
  destruct (note_simples_fields l a) as (H1 & H2 & H3 & H4 & H5).
  apply ext_same; auto. rewrite H1. lia.
Qed.

Lemma ext_push_apply_dep (a0 : acc) (d : apply_dep) (a : acc) :
  ok_node a0 a (ad_apply_call_witness d) -> ext a0 a (push_apply_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id cd Hin. left. exact Hin.
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
Qed.

Lemma ext_push_closure_dep (a0 : acc) (d : closure_dep) (a : acc) :
  ok_node a0 a (cld_let_bound_name_of_the_closure d) ->
  ext a0 a (push_closure_dep d a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id cd Hin. left. exact Hin.
  - intros d' Hin. left. exact Hin.
  - intros d' [<- | Hin]; [ right; exact H | left; exact Hin ].
Qed.

Lemma ext_add_code_dep (a0 : acc) (id : code_id) (cd : code_dep)
    (a : acc) :
  (∀ x, x ∈ code_dep_nodes cd -> ok_node a0 a x) ->
  ext a0 a (add_code_dep id cd a).
Proof.
  intros H. constructor; cbn; [ lia | | | | ].
  - exists []. split; [ reflexivity | split ];
      [ intros fct x [] | intros w n [] ].
  - intros id' cd' [Heq | Hin]; [ injection Heq as <- <-; right; exact H
                                | left; exact Hin ].
  - intros d' Hin. left. exact Hin.
  - intros d' Hin. left. exact Hin.
Qed.

Lemma ext_fold {A : Type} (a0 : acc) (f : acc -> A -> acc) (l : list A)
    (a : acc) :
  (∀ y b, y ∈ l -> (acc_next a <= acc_next b)%positive -> ext a0 b (f b y)) ->
  ext a0 a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [List.fold_left].
  - apply ext_refl.
  - pose proof (H y a (or_introl eq_refl) (Pos.le_refl _)) as H1.
    apply (ext_trans _ _ _ _ H1). apply IH.
    intros y' b Hy' Hb. apply H; [ right; exact Hy' | ].
    pose proof (ext_next _ _ _ H1). lia.
Qed.

(* What [ext] gives where [a0] is the invariant. *)
Lemma widget_lt (a : acc) (w : node) :
  winv a -> widget a w -> made_lt w (acc_next a).
Proof.
  intros Hinv [n Hin]. apply (wi_into a Hinv _ w Hin). left. reflexivity.
Qed.

Lemma ok_close (a0 a' : acc) (x : node) :
  winv a0 -> ext a0 a0 a' -> ok_node a0 a' x -> good a' x.
Proof.
  intros Hinv [Hn (L & Hf & Hi & Hw) Hc Hap Hcl].
  assert (Hwid : ∀ w, widget a' w -> widget a0 w).
  { intros w [n Hin]. rewrite Hf in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ destruct (Hw _ _ Hin) | exists n; exact Hin ]. }
  intros [Hx | [[Hlt Hnw] | (k & -> & H1 & H2)]].
  - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
    intros Hwx. destruct (wi_made a0 Hinv x (Hwid x Hwx)) as [k ->].
    exact (Hx k eq_refl).
  - split; [ intros k Hk; specialize (Hlt k Hk); lia | ].
    intros Hwx. exact (Hnw (Hwid x Hwx)).
  - split; [ intros k' Hk; injection Hk as <-; exact H2 | ].
    intros Hwx. specialize (widget_lt a0 _ Hinv (Hwid _ Hwx) k eq_refl).
    lia.
Qed.

Lemma ext_close (a0 a' : acc) :
  winv a0 -> ext a0 a0 a' -> winv a' ∧ grows a0 a'.
Proof.
  intros Hinv [Hn (L & Hf & Hi & Hw) Hc Hap Hcl].
  assert (Hwid : ∀ w, widget a' w -> widget a0 w).
  { intros w [n Hin]. rewrite Hf in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ destruct (Hw _ _ Hin) | exists n; exact Hin ]. }
  assert (Hgood : ∀ x, ok_node a0 a' x -> good a' x).
  { intros x [Hx | [[Hlt Hnw] | (k & -> & H1 & H2)]].
    - split; [ intros k -> ; destruct (Hx k eq_refl) | ].
      intros Hwx. destruct (wi_made a0 Hinv x (Hwid x Hwx)) as [k ->].
      exact (Hx k eq_refl).
    - split; [ intros k Hk; specialize (Hlt k Hk); lia | ].
      intros Hwx. exact (Hnw (Hwid x Hwx)).
    - split; [ intros k' Hk; injection Hk as <-; exact H2 | ].
      intros Hwx. specialize (widget_lt a0 _ Hinv (Hwid _ Hwx) k eq_refl).
      lia. }
  assert (Hgood0 : ∀ x, good a0 x -> good a' x).
  { intros x Hx. apply Hgood. right; left. exact Hx. }
  split; [ constructor | split; [ exact Hn | ] ].
  - intros fct x Hin Hx. rewrite Hf in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + exact (proj1 (Hgood x (Hi _ _ Hin Hx))).
    + intros k Hk. specialize (wi_into a0 Hinv _ _ Hin Hx k Hk). lia.
  - intros w Hwx. exact (wi_made a0 Hinv w (Hwid w Hwx)).
  - intros w n fct Hw' Hfct Hx. rewrite Hf in Hw', Hfct.
    apply in_app_or in Hw' as [Hw' | Hw']; [ destruct (Hw _ _ Hw') | ].
    apply in_app_or in Hfct as [Hfct | Hfct].
    + exfalso. destruct (Hgood w (Hi _ _ Hfct Hx)) as [_ Hnw].
      apply Hnw. exists n. rewrite Hf. apply in_or_app. right. exact Hw'.
    + exact (wi_widget a0 Hinv w n fct Hw' Hfct Hx).
  - intros id cd x Hin Hx. destruct (Hc _ _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (wi_code_deps a0 Hinv _ _ _ Hin' Hx)).
    + exact (Hgood _ (H x Hx)).
  - intros d Hin. destruct (Hap _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (wi_apply_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
  - intros d Hin. destruct (Hcl _ Hin) as [Hin' | H].
    + exact (Hgood0 _ (wi_closure_deps a0 Hinv _ Hin')).
    + exact (Hgood _ H).
  - intros w Hwx. left. exact (Hwid w Hwx).
Qed.

Lemma grows_refl (a : acc) : grows a a.
Proof. split; [ lia | intros w H; left; exact H ]. Qed.

Lemma grows_trans (a a1 a2 : acc) : grows a a1 -> grows a1 a2 -> grows a a2.
Proof.
  intros [Hn1 Hw1] [Hn2 Hw2]. split; [ lia | ].
  intros w Hw. destruct (Hw2 w Hw) as [H | (k & -> & Hk)].
  - destruct (Hw1 _ H) as [H' | (k & -> & Hk)]; [ left; exact H' | ].
    right. exists k. split; [ reflexivity | lia ].
  - right. exists k. split; [ reflexivity | lia ].
Qed.

Lemma good_grows (a a' : acc) (x : node) :
  grows a a' -> good a x -> good a' x.
Proof.
  intros [Hn Hw] [Hlt Hnw]. split.
  - intros k Hk. specialize (Hlt k Hk). lia.
  - intros Hwx. destruct (Hw x Hwx) as [H | (k & -> & Hk)];
      [ exact (Hnw H) | ].
    specialize (Hlt k eq_refl). lia.
Qed.

Lemma env_good_grows (env : traverse_env) (a a' : acc) :
  grows a a' -> env_good env a -> env_good env a'.
Proof.
  intros Hg H k params x Hin Hx. exact (good_grows _ _ _ Hg (H _ _ _ Hin Hx)).
Qed.

(* A step of the traversal, from an accumulator of the invariant whose
   continuations in scope are good. *)
Definition tstep (env : traverse_env) (a a' : acc) : Prop :=
  winv a -> env_good env a -> winv a' ∧ grows a a'.

Lemma tstep_refl (env : traverse_env) (a : acc) : tstep env a a.
Proof. intros H _. split; [ exact H | apply grows_refl ]. Qed.

Lemma tstep_trans (env : traverse_env) (a a1 a2 : acc) :
  tstep env a a1 -> tstep env a1 a2 -> tstep env a a2.
Proof.
  intros H1 H2 Hinv Henv. destruct (H1 Hinv Henv) as [Hinv1 Hg1].
  destruct (H2 Hinv1 (env_good_grows _ _ _ Hg1 Henv)) as [Hinv2 Hg2].
  split; [ exact Hinv2 | exact (grows_trans _ _ _ Hg1 Hg2) ].
Qed.

Lemma tstep_ext (env : traverse_env) (a a' : acc) :
  (winv a -> env_good env a -> ext a a a') -> tstep env a a'.
Proof. intros H Hinv Henv. exact (ext_close a a' Hinv (H Hinv Henv)). Qed.

Lemma tstep_fold {A : Type} (env : traverse_env) (f : acc -> A -> acc)
    (l : list A) (a : acc) :
  (∀ y b, y ∈ l -> tstep env b (f b y)) -> tstep env a (List.fold_left f l a).
Proof.
  revert a. induction l as [ | y l IH]; intros a H; cbn [fold_left].
  - apply tstep_refl.
  - apply (tstep_trans _ _ (f a y)); [ apply H; left; reflexivity | ].
    apply IH. intros y' b Hy'. apply H. right. exact Hy'.
Qed.

(* ------------------------------------------------------------------------ *)
(* The steps of the traversal                                               *)
(* ------------------------------------------------------------------------ *)

Ltac into_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (flat_map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_flat_map in H; destruct H as [y [Hy H]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  | In _ (match ?t with _ => _ end) => destruct t
  end.

Ltac into_close H Hx :=
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst; cbn [into_nodes] in Hx;
  repeat match type of Hx with
  | In _ (_ :: _) => destruct Hx as [Hx | Hx]
  | In _ [] => destruct Hx
  end;
  subst.

Ltac no_widget :=
  let w := fresh "w" in let n := fresh "n" in let H := fresh "H" in
  intros w n H; into_split H;
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  try discriminate.

Ltac ok_tac :=
  first
    [ assumption
    | apply ok_simple
    | apply ok_not_made; intros ?k; discriminate
    | apply ok_made; cbn [acc_next]; lia
    | eapply ok_node_mono; [ | eassumption ]; cbn [acc_next]; lia ].

Lemma ext_cond_any_usage (a0 : acc) (env : traverse_env) (s : simple)
    (a : acc) :
  ext a0 a (cond_any_usage env s a).
Proof.
  unfold cond_any_usage.
  apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate).
  - intros x [<- | []]. ok_tac.
  - intros x [].
Qed.

Lemma ext_cond_alias (a0 : acc) (env : traverse_env) (from to_ : node)
    (a : acc) :
  ok_node a0 a to_ -> ext a0 a (cond_alias env from to_ a).
Proof.
  intros H. unfold cond_alias.
  destruct (env_current_code_id env) as [id | ]; apply ext_add_fact;
    try (intros ? ? ?; discriminate); intros x [<- | []]; exact H.
Qed.

Lemma fresh_list_spec (n : nat) (a : acc) :
  acc_facts (snd (fresh_list n a)) = acc_facts a
  ∧ acc_code_deps (snd (fresh_list n a)) = acc_code_deps a
  ∧ acc_apply_deps (snd (fresh_list n a)) = acc_apply_deps a
  ∧ acc_closure_deps (snd (fresh_list n a)) = acc_closure_deps a
  ∧ (acc_next a <= acc_next (snd (fresh_list n a)))%positive
  ∧ ∀ x, x ∈ fst (fresh_list n a) ->
      ∃ m, x = Node_made m ∧ (acc_next a <= m)%positive
           ∧ (m < acc_next (snd (fresh_list n a)))%positive.
Proof.
  revert a. induction n as [ | n IH]; intros a; cbn [fresh_list].
  - cbn. repeat split; [ lia | intros x [] ].
  - cbn [fresh]. cbv beta iota.
    match goal with
    | |- context [fresh_list n ?b] =>
      destruct (IH b) as (H1 & H2 & H3 & H4 & H5 & H6);
      destruct (fresh_list n b) as [xs a2] eqn:E
    end.
    cbn [fst snd acc_next acc_facts acc_code_deps acc_apply_deps
      acc_closure_deps] in *.
    repeat split; auto; [ lia | ].
    intros y [<- | Hy].
    + exists (acc_next a). split; [ reflexivity | lia ].
    + destruct (H6 y Hy) as (m & -> & Hm1 & Hm2). exists m.
      split; [ reflexivity | lia ].
Qed.

Lemma ext_fresh_list (a0 : acc) (n : nat) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (fresh_list n a))
  ∧ ∀ x, x ∈ fst (fresh_list n a) -> ok_node a0 (snd (fresh_list n a)) x.
Proof.
  intros Hle. destruct (fresh_list_spec n a) as (H1 & H2 & H3 & H4 & H5 & H6).
  split; [ apply ext_same; auto | ].
  intros x Hx. destruct (H6 x Hx) as (m & -> & Hm1 & Hm2).
  apply ok_made; lia.
Qed.

Lemma lookup_in_conts (env : traverse_env) (k : continuation)
    (params : list node) :
  lookup k (env_conts env) = Some params -> (k, params) ∈ env_conts env.
Proof. apply FlambdaMapProofs.lookup_in. Qed.

Lemma ext_find_cont (a0 : acc) (env : traverse_env) (k : continuation)
    (count : nat) (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (find_cont env k count a))
  ∧ ∀ x, x ∈ fst (find_cont env k count a) ->
      ok_node a0 (snd (find_cont env k count a)) x.
Proof.
  intros Henv Hle. unfold find_cont.
  destruct (lookup k (env_conts env)) as [params | ] eqn:E.
  - cbn [fst snd]. split; [ apply ext_refl | ].
    intros x Hx. apply ok_good.
    exact (Henv k params x (lookup_in_conts _ _ _ E) Hx).
  - destruct (ext_fresh_list a0 count a Hle) as [H1 H2].
    destruct (fresh_list count a) as [params b] eqn:Eb.
    cbn [fst snd] in *. split.
    + apply (ext_trans _ _ _ _ H1). apply ext_add_facts.
      * intros fct x Hin Hx. into_split Hin. into_close Hin Hx.
      * no_widget.
    + intros x Hx. exact (H2 x Hx).
Qed.

Lemma ext_block_deps (a0 : acc) (base : node) (fields : list simple)
    (a : acc) :
  ok_node a0 a base -> ext a0 a (block_deps base fields a).
Proof.
  intros H. unfold block_deps.
  destruct (note_simples_fields fields a) as (H1 & _).
  apply (ext_trans _ _ (note_simples fields a));
    [ apply ext_note_simples | ].
  apply ext_add_facts.
  - intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      eapply ok_node_mono; try eassumption; rewrite H1; lia.
  - no_widget.
Qed.

Lemma ext_traverse_prim (a0 : acc) (env : traverse_env) (x : variable)
    (p : prim) (args : list simple) (a : acc) :
  ext a0 a (traverse_prim env x p args a).
Proof.
  unfold traverse_prim.
  assert (Hacc : ∀ f s b,
             f <> known ->
             ext a0 b (add_fact (Fact_accessor (Node_variable x) f
                                   (node_of_simple s)) (note_simple s b))).
  { intros f s b Hf. apply (ext_trans _ _ (note_simple s b));
      [ apply ext_note_simple | ].
    apply ext_add_fact.
    - intros y [<- | []]. ok_tac.
    - intros w n Heq. injection Heq as _ Hf' _. exact (Hf Hf'). }
  assert (Hdefault :
    ext a0 a
      (add_facts
         (Fact_any_source (Node_variable x)
          :: map (fun from => Fact_use (Node_variable x) from)
               (free_names_of_simples args))
         (if prim_has_arbitrary_effects p
          then cond_any_usage env (Var x) a else a))).
  { apply (ext_trans _ _ (if prim_has_arbitrary_effects p
                          then cond_any_usage env (Var x) a else a)).
    - destruct (prim_has_arbitrary_effects p);
        [ apply ext_cond_any_usage | apply ext_refl ].
    - apply ext_add_facts.
      + intros fct y Hin Hy. into_split Hin; into_close Hin Hy; ok_tac.
      + no_widget. }
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- ext _ _ (match ?t with _ => _ end) => destruct t
    end;
    try exact Hdefault;
    try (apply ext_block_deps; ok_tac);
    try (apply Hacc; discriminate);
    try apply ext_refl.
  - match goal with
    | |- ext _ _ (add_fact _ ?b) => apply (ext_trans _ _ b)
    end; [ apply Hacc; discriminate | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
Qed.

Lemma ext_record_set (a0 : acc) (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (∀ base code, (base, code) ∈ closures -> ok_node a0 a base) ->
  ext a0 a (record_set_of_closures_deps env closures function_slots
              value_slots a).
Proof.
  intros H. unfold record_set_of_closures_deps. apply ext_fold.
  intros [base code] b Hin Hle. cbv beta iota.
  assert (Hb : ok_node a0 b base)
    by exact (ok_node_mono _ _ _ _ Hle (H _ _ Hin)).
  set (b1 := match code with
             | Some id => push_closure_dep _ b
             | None => b
             end).
  assert (H1 : ext a0 b b1).
  { subst b1. destruct code; [ apply ext_push_closure_dep; exact Hb
                             | apply ext_refl ]. }
  apply (ext_trans _ _ _ _ H1).
  match goal with
  | |- ext _ _ (add_facts _ (note_simples ?l _)) =>
    destruct (note_simples_fields l b1) as (Hn & _);
    apply (ext_trans _ _ (note_simples l b1)); [ apply ext_note_simples | ]
  end.
  apply ext_add_facts.
  - intros fct x Hfct Hx. into_split Hfct; into_close Hfct Hx;
      apply (ok_node_mono _ b); [ | exact Hb | | exact Hb ];
      rewrite Hn; exact (ext_next _ _ _ H1).
  - no_widget.
Qed.

Lemma ext_closures_of_set (a0 : acc) (x : variable) (slot : function_slot)
    (others : list (function_slot * variable))
    (decls : list (function_slot * code_id)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (snd (closures_of_set x slot others decls a))
  ∧ ∀ f name id, (f, name, id) ∈ fst (closures_of_set x slot others decls a) ->
      ok_node a0 (snd (closures_of_set x slot others decls a)) name.
Proof.
  revert a. induction decls as [ | [f id] decls IH]; intros a Hle;
    cbn [closures_of_set].
  - cbn [fst snd]. split; [ apply ext_refl | intros f name id [] ].
  - match goal with
    | |- context [let '(_, _) := ?t in _] =>
      assert (Ht : ext a0 a (snd t) ∧ ok_node a0 (snd t) (fst t));
      [ | destruct t as [name b] eqn:E ]
    end.
    { destruct (Pos.eqb f slot);
        [ cbn; split; [ apply ext_refl | ok_tac ] | ].
      destruct (lookup f others);
        [ cbn; split; [ apply ext_refl | ok_tac ] | ].
      split; [ apply ext_fresh | apply ok_fresh; exact Hle ]. }
    cbn [fst snd] in Ht. destruct Ht as [H1 Hname].
    destruct (IH b) as [H2 H3]; [ pose proof (ext_next _ _ _ H1); lia | ].
    destruct (closures_of_set x slot others decls b) as [rest c] eqn:E2.
    cbn [fst snd] in *. split; [ exact (ext_trans _ _ _ _ H1 H2) | ].
    intros f' name' id' [Heq | Hin].
    + injection Heq as <- <- <-.
      exact (ok_node_mono _ _ _ _ (ext_next _ _ _ H2) Hname).
    + exact (H3 _ _ _ Hin).
Qed.

Lemma ext_traverse_closure_of_set (a0 : acc) (env : traverse_env)
    (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_closure_of_set env x slot function_decls value_slots a).
Proof.
  intros Hle. unfold traverse_closure_of_set. cbv zeta.
  destruct (among slot _); [ | apply ext_refl ].
  match goal with
  | |- context [closures_of_set ?x ?s ?o ?d a] =>
    destruct (ext_closures_of_set a0 x s o d a Hle) as [H1 H2];
    destruct (closures_of_set x s o d a) as [closures b] eqn:E
  end.
  cbn [fst snd] in *. apply (ext_trans _ _ _ _ H1).
  apply ext_record_set. intros base code Hin.
  apply in_map_iff in Hin. destruct Hin as [[[f name] id] [Heq Hin]].
  injection Heq as <- _. exact (H2 _ _ _ Hin).
Qed.

Lemma ext_traverse_named (a0 : acc) (env : traverse_env) (x : variable)
    (defining_expr : named) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_named env x defining_expr a).
Proof.
  intros Hle. unfold traverse_named.
  destruct defining_expr as [ s | p args | id vs | slot decls vs | w c
                            | m w c | mf mt c ].
  - apply (ext_trans _ _ (note_simple s a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply ext_traverse_prim.
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_traverse_closure_of_set. exact Hle.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
  - apply (ext_trans _ _ (note_simple c a)); [ apply ext_note_simple | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate.
Qed.

Lemma ext_traverse_static_binding (a0 : acc) (env : traverse_env)
    (binding : static_binding) (a : acc) :
  ext a0 a (traverse_static_binding env binding a).
Proof.
  unfold traverse_static_binding.
  destruct binding as [ | | s const]; [ apply ext_refl | apply ext_refl | ].
  destruct const as [ id vs | slot syms decls vs | tag fields
                    | contents | str | [ | f fields] ].
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_record_set. intros base code [Heq | []].
    injection Heq as <- _. ok_tac.
  - apply ext_block_deps. ok_tac.
  - unfold boxed_number_facts. apply ext_add_facts.
    + intros fct y Hin Hy. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split Hin; into_close Hin Hy; ok_tac.
    + intros w n H. repeat match goal with
      | H : context [match ?t with _ => _ end] |- _ => destruct t
      end; into_split H; discriminate.
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ].
  - apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ].
  - apply ext_block_deps. ok_tac.
Qed.

Lemma fresh_spec (a0 c : acc) (x : node) (b : acc) :
  fresh c = (x, b) -> (acc_next a0 <= acc_next c)%positive ->
  x = Node_made (acc_next c) ∧ acc_next b = Pos.succ (acc_next c)
  ∧ ext a0 c b ∧ ok_node a0 b x.
Proof.
  intros E Hle. pose proof (ext_fresh a0 c) as H1.
  pose proof (ok_fresh a0 c Hle) as H2. rewrite E in H1, H2.
  unfold fresh in E. injection E as <- <-. cbn in *. auto.
Qed.

Lemma indexed_in {A : Type} (l : list A) (i : nat) (x : A) :
  (i, x) ∈ indexed l -> x ∈ l.
Proof. unfold indexed. apply in_combine_r. Qed.

Lemma group_by_sizes_in {A : Type} (sizes : list nat) (l g : list A)
    (x : A) :
  g ∈ group_by_sizes sizes l -> x ∈ g -> x ∈ l.
Proof.
  revert l. induction sizes as [ | n sizes IH]; intros l Hg Hx;
    cbn [group_by_sizes] in Hg; [ destruct Hg | ].
  rewrite <- (firstn_skipn n l). apply in_or_app.
  destruct Hg as [<- | Hg]; [ left; exact Hx | ].
  right. exact (IH _ Hg Hx).
Qed.

Lemma group_in {A : Type} (ar : arity) (l g : list A) (x : A) :
  g ∈ at_least_one_group (group_by_parameter ar l) -> x ∈ g -> x ∈ l.
Proof.
  unfold at_least_one_group, group_by_parameter.
  destruct (group_by_sizes _ l) eqn:E.
  - intros [<- | []] [].
  - intros Hg Hx. rewrite <- E in Hg. exact (group_by_sizes_in _ _ _ _ Hg Hx).
Qed.

Ltac nexts :=
  repeat match goal with
  | H : ext _ ?b ?c |- _ =>
    lazymatch goal with
    | _ : (acc_next b <= acc_next c)%positive |- _ => fail
    | _ => pose proof (ext_next _ _ _ H)
    end
  end.

Ltac mono :=
  nexts; eapply ok_node_mono; [ | eassumption ];
  cbn [acc_next add_fact add_facts push_apply_dep push_closure_dep
       add_code_dep] in *; lia.

Ltac fresh_step a0 c x b E Hx :=
  let Hn := fresh "Hn" in let He := fresh "He" in let Ho := fresh "Ho" in
  destruct (fresh c) as [x b] eqn:E;
  destruct (fresh_spec a0 c x b E) as (Hx & Hn & He & Ho);
  [ nexts; lia | ].

Lemma ext_create_known (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a (snd (create_known_arity_call_witness id params returns exn a))
  ∧ ok_node a0 (snd (create_known_arity_call_witness id params returns exn a))
      (fst (create_known_arity_call_witness id params returns exn a)).
Proof.
  intros Hle Hp. unfold create_known_arity_call_witness.
  fresh_step a0 a w b E Hw. cbv zeta. cbn [fst snd].
  split; [ | exact Ho ].
  apply (ext_trans _ _ _ _ He). apply ext_add_facts; [ | no_widget ].
  intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
    try exact Ho.
  eapply ok_node_mono; [ | apply Hp; eapply indexed_in; eassumption ].
  rewrite Hn. lia.
Qed.

Lemma ext_create_tupled (a0 : acc) (id : code_id) (params returns : list node)
    (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_tupled_call_witnesses id params returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_tupled_call_witnesses id params returns
                 exn a) ->
      ok_node a0
        (snd (create_unknown_arity_tupled_call_witnesses id params returns
                exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_tupled_call_witnesses.
  fresh_step a0 a w b E Hw.
  pose proof (ext_next _ _ _ He) as Hb.
  fresh_step a0 b u c E2 Hu.
  pose proof (ext_next _ _ _ He0) as Hc.
  fresh_step a0 c k d E3 Hk.
  cbv zeta. cbn [fst snd].
  assert (Hpd : ∀ x, x ∈ params -> ok_node a0 d x).
  { intros x Hx. eapply ok_node_mono; [ | exact (Hp x Hx) ]. lia. }
  assert (Hwd : ok_node a0 d w) by (eapply ok_node_mono; [ | exact Ho ]; lia).
  assert (Hud : ok_node a0 d u)
    by (eapply ok_node_mono; [ | exact Ho0 ]; lia).
  split.
  - apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ He0).
    apply (ext_trans _ _ _ _ He1). apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx;
      try ok_tac.
    apply Hpd. eapply indexed_in. eassumption.
  - intros x [<- | []]. cbn [acc_next add_facts].
    exact Hwd.
Qed.

Lemma ext_witness_deps (a0 : acc) (id : code_id) (returns : list node)
    (exn : node) (pw : list (list node * node)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ first w, (first, w) ∈ pw ->
     ok_node a0 a w ∧ ∀ x, x ∈ first -> ok_node a0 a x) ->
  ext a0 a (unknown_arity_witness_deps id returns exn pw a).
Proof.
  revert a. induction pw as [ | [first w] rest IH]; intros a Hle Hpw;
    cbn [unknown_arity_witness_deps]; [ apply ext_refl | ].
  destruct (Hpw first w (or_introl eq_refl)) as [Hw Hfirst].
  set (b := add_facts _ a).
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  assert (H1 : ext a0 a b).
  { apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hw.
    apply Hfirst. eapply indexed_in. eassumption. }
  apply (ext_trans _ _ _ _ H1).
  assert (Hrest : ∀ first' w', (first', w') ∈ rest ->
            ok_node a0 b w' ∧ ∀ x, x ∈ first' -> ok_node a0 b x).
  { intros first' w' Hin. destruct (Hpw _ _ (or_intror Hin)) as [H2 H3].
    split; [ exact H2 | exact H3 ]. }
  destruct rest as [ | [g nw] rest'].
  - apply ext_add_facts; [ | no_widget ].
    intros fct x Hin Hx. into_split Hin; into_close Hin Hx; exact Hw.
  - fresh_step a0 b v c E Hv.
    cbv zeta.
    destruct (Hrest g nw (or_introl eq_refl)) as [Hnw _].
    assert (Hcv : ok_node a0 c v) by exact Ho.
    set (d := add_facts _ c).
    assert (Hdn : acc_next d = acc_next c) by reflexivity.
    assert (H2 : ext a0 c d).
    { apply ext_add_facts; [ | no_widget ].
      intros fct x Hin Hx. into_split Hin; into_close Hin Hx; try exact Hcv.
      eapply ok_node_mono; [ | exact Hw ]. lia. }
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H2).
    apply IH; [ lia | ].
    intros first' w' Hin. destruct (Hrest _ _ Hin) as [H3 H4].
    split; [ eapply ok_node_mono; [ | exact H3 ] | intros x Hx;
             eapply ok_node_mono; [ | exact (H4 x Hx) ] ]; lia.
Qed.

Lemma ext_create_non_tupled (a0 : acc) (id : code_id) (ar : arity)
    (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_non_tupled_call_witnesses id ar params
            returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_non_tupled_call_witnesses id ar params
                 returns exn a) ->
      ok_node a0
        (snd (create_unknown_arity_non_tupled_call_witnesses id ar params
                returns exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_non_tupled_call_witnesses.
  cbv zeta.
  set (groups := at_least_one_group (group_by_parameter ar params)).
  destruct (ext_fresh_list a0 (length groups) a Hle) as [H1 H1o].
  destruct (fresh_list (length groups) a) as [ws b] eqn:E.
  cbn [fst snd] in *.
  pose proof (ext_next _ _ _ H1) as Hn1.
  assert (H2 : ext a0 b
                 (unknown_arity_witness_deps id returns exn (combine groups ws)
                    b)).
  { apply ext_witness_deps; [ lia | ].
    intros first w Hin. split.
    - exact (H1o w (in_combine_r _ _ _ _ Hin)).
    - intros x Hx. apply in_combine_l in Hin.
      eapply ok_node_mono; [ exact Hn1 | ].
      apply Hp. exact (group_in ar params first x Hin Hx). }
  split; [ exact (ext_trans _ _ _ _ H1 H2) | ].
  intros x Hx. eapply ok_node_mono; [ exact (ext_next _ _ _ H2) | ].
  exact (H1o x Hx).
Qed.

Lemma ext_create_unknown (a0 : acc) (id : code_id) (is_tupled : bool)
    (ar : arity) (params returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ params -> ok_node a0 a x) ->
  ext a0 a
    (snd (create_unknown_arity_call_witnesses id is_tupled ar params
            returns exn a))
  ∧ ∀ x,
      x ∈ fst (create_unknown_arity_call_witnesses id is_tupled ar params
                 returns exn a) ->
      ok_node a0
        (snd (create_unknown_arity_call_witnesses id is_tupled ar params
                returns exn a)) x.
Proof.
  intros Hle Hp. unfold create_unknown_arity_call_witnesses.
  destruct is_tupled;
    [ apply ext_create_tupled | apply ext_create_non_tupled ]; assumption.
Qed.

Lemma ext_prepare_code (a0 : acc) (id : code_id) (c : code) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (prepare_code id c a).
Proof.
  intros Hle. unfold prepare_code.
  destruct (ext_fresh_list a0 (length (code_result_kinds c)) a Hle)
    as [H1 H1o].
  destruct (fresh_list _ a) as [returns b1] eqn:E1. cbn [fst snd] in *.
  cbv beta iota.
  fresh_step a0 b1 exn b2 E2 Hexn. cbv beta iota.
  fresh_step a0 b2 mc b3 E3 Hmc. cbv beta iota.
  destruct (ext_fresh_list a0 (length (code_params_kinds c)) b3)
    as [H4 H4o]; [ nexts; lia | ].
  destruct (fresh_list _ b3) as [params b4] eqn:E4. cbn [fst snd] in *.
  cbv beta iota.
  destruct (ext_create_known a0 id params returns exn b4) as [H5 H5o];
    [ nexts; lia | exact H4o | ].
  destruct (create_known_arity_call_witness id params returns exn b4)
    as [known b5] eqn:E5. cbn [fst snd] in *. cbv beta iota.
  destruct (ext_create_unknown a0 id (code_is_tupled c) (code_params_arity c)
              params returns exn b5) as [H6 H6o];
    [ nexts; lia | intros x Hx; pose proof (H4o x Hx); mono | ].
  destruct (create_unknown_arity_call_witnesses _ _ _ _ _ _ b5)
    as [unknown b6] eqn:E6. cbn [fst snd] in *. cbv beta iota zeta.
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ He).
  apply (ext_trans _ _ _ _ He0). apply (ext_trans _ _ _ _ H4).
  apply (ext_trans _ _ _ _ H5). apply (ext_trans _ _ _ _ H6).
  match goal with
  | |- ext _ _ (add_code_dep _ _ ?b) => apply (ext_trans _ _ b)
  end.
  { apply ext_add_fact; [ intros y [<- | []]; ok_tac
                        | intros ? ? ?; discriminate ]. }
  apply ext_add_code_dep.
  cbn [code_dep_nodes cd_my_closure cd_exn cd_return acc_next add_fact
       add_facts].
  intros x [<- | [<- | Hx]]; [ mono | mono | ].
  pose proof (H1o x Hx). mono.
Qed.

Lemma ext_prepare_binding (a0 : acc) (binding : static_binding) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (prepare_binding binding a).
Proof.
  intros Hle. destruct binding; cbn [prepare_binding];
    [ apply ext_prepare_code; exact Hle | apply ext_refl | apply ext_refl ].
Qed.

(* ------------------------------------------------------------------------ *)
(* Calls                                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma make_known_spec (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  ext a0 a (snd (make_known_arity_apply_widget env ap returns exn a))
  ∧ fst (make_known_arity_apply_widget env ap returns exn a)
    = Node_made (Pos.succ (Pos.succ (acc_next a)))
  ∧ acc_next (snd (make_known_arity_apply_widget env ap returns exn a))
    = Pos.succ (Pos.succ (Pos.succ (acc_next a)))
  ∧ acc_code_deps (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_code_deps a
  ∧ acc_apply_deps (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_apply_deps a
  ∧ acc_closure_deps
      (snd (make_known_arity_apply_widget env ap returns exn a))
    = acc_closure_deps a
  ∧ ∀ fct x,
      fct ∈ acc_facts (snd (make_known_arity_apply_widget env ap returns exn
                              a)) ->
      x ∈ into_nodes fct ->
      fct ∈ acc_facts a
      ∨ made_lt x (Pos.succ (Pos.succ (acc_next a))).
Proof.
  intros Hle Hin. unfold make_known_arity_apply_widget.
  cbn [fresh]. cbv beta iota zeta. cbn [fst snd acc_next].
  destruct (note_simples_fields (ap_args ap)
              {| acc_next := Pos.succ (Pos.succ (Pos.succ (acc_next a)));
                 acc_facts := acc_facts a;
                 acc_code_deps := acc_code_deps a;
                 acc_apply_deps := acc_apply_deps a;
                 acc_closure_deps := acc_closure_deps a;
                 acc_symbols := acc_symbols a |})
    as (N1 & N2 & N3 & N4 & N5).
  cbn [acc_next acc_facts acc_code_deps acc_apply_deps acc_closure_deps]
    in N1, N2, N3, N4, N5.
  set (b := note_simples _ _) in *. clearbody b.
  assert (Hok : ∀ x, x ∈ exn :: returns -> made_lt x (acc_next a)).
  { intros x Hx k ->. destruct (Hin _ Hx) as [H | [[H _] | (m & Hm & _ & H)]].
    - destruct (H k eq_refl).
    - specialize (H k eq_refl). lia.
    - injection Hm as <-. exact H. }
  set (c := add_facts _ b).
  assert (Hext : ext a0 b (cond_alias env
                             (Node_made (Pos.succ (Pos.succ (acc_next a))))
                             (Node_made (acc_next a)) c)).
  { apply (ext_trans _ _ c).
    - apply ext_add_facts; [ | no_widget ].
      intros fct x H Hx. into_split H; into_close H Hx.
      + eapply ok_node_mono; [ | apply Hin; right; eapply indexed_in;
                                 eassumption ].
        rewrite N1. lia.
      + eapply ok_node_mono; [ | apply Hin; left; reflexivity ].
        rewrite N1. lia.
      + apply ok_made; try rewrite N1; lia.
    - apply ext_cond_alias. apply ok_made; cbn; try rewrite N1; lia. }
  split; [ apply (ext_trans _ _ b); [ | exact Hext ];
           apply ext_same; auto; rewrite N1; lia | ].
  split; [ reflexivity | ].
  split; [ unfold cond_alias; destruct (env_current_code_id env); cbn;
           exact N1 | ].
  split; [ unfold cond_alias; destruct (env_current_code_id env); cbn;
           exact N3 | ].
  split; [ unfold cond_alias; destruct (env_current_code_id env); cbn;
           exact N4 | ].
  split; [ unfold cond_alias; destruct (env_current_code_id env); cbn;
           exact N5 | ].
  intros fct x H Hx. unfold cond_alias in H.
    subst c. destruct (env_current_code_id env);
      cbn [acc_facts add_fact add_facts] in H; rewrite N2 in H;
      into_split H; try (left; exact H); right; into_close H Hx;
      try (intros k Hk; injection Hk as <-; lia);
      try (eapply made_lt_mono;
           [ apply Hok; right; eapply indexed_in; eassumption | lia ]);
      try (eapply made_lt_mono; [ apply Hok; left; reflexivity | lia ]).
Qed.

Lemma winv_add_widget (a : acc) (p : positive) (n : node) :
  winv a -> (p < acc_next a)%positive ->
  (∀ fct x, fct ∈ acc_facts a -> x ∈ into_nodes fct -> made_lt x p) ->
  (∀ id cd x, (id, cd) ∈ acc_code_deps a -> x ∈ code_dep_nodes cd ->
     made_lt x p) ->
  (∀ d, d ∈ acc_apply_deps a -> made_lt (ad_apply_call_witness d) p) ->
  (∀ d, d ∈ acc_closure_deps a ->
     made_lt (cld_let_bound_name_of_the_closure d) p) ->
  winv (add_fact (Fact_accessor (Node_made p) known n) a).
Proof.
  intros Hinv Hp Hfacts Hc Hap Hcl.
  assert (Hne : ∀ x, made_lt x p -> x <> Node_made p).
  { intros x Hx ->. specialize (Hx p eq_refl). lia. }
  assert (Hgood : ∀ x, good a x -> made_lt x p -> good
            (add_fact (Fact_accessor (Node_made p) known n) a) x).
  { intros x [Hlt Hnw] Hxp. split; [ exact Hlt | ].
    intros [m [Hin | Hin]];
      [ injection Hin as Hxe _; exact (Hne _ Hxp (eq_sym Hxe))
      | exact (Hnw (ex_intro _ m Hin)) ]. }
  constructor; cbn [acc_next acc_facts acc_code_deps acc_apply_deps
    acc_closure_deps add_fact add_facts app].
  - intros fct x [<- | Hin] Hx.
    + destruct Hx as [<- | []]. intros k Hk. injection Hk as <-. exact Hp.
    + exact (wi_into a Hinv _ _ Hin Hx).
  - intros w [m [Hin | Hin]]; [ injection Hin as <- _; exists p; reflexivity
                              | exact (wi_made a Hinv w (ex_intro _ m Hin)) ].
  - intros w m fct Hw Hfct Hx. destruct Hw as [Hw | Hw].
    + injection Hw as <- <-. destruct Hfct as [<- | Hfct];
        [ split; reflexivity | ].
      exfalso. exact (Hne _ (Hfacts _ _ Hfct Hx) eq_refl).
    + destruct Hfct as [<- | Hfct].
      * destruct Hx as [<- | []]. exfalso.
        apply (Hne (Node_made p) (Hfacts _ _ Hw (or_introl eq_refl))).
        reflexivity.
      * exact (wi_widget a Hinv _ _ _ Hw Hfct Hx).
  - intros id cd x Hin Hx.
    exact (Hgood _ (wi_code_deps a Hinv _ _ _ Hin Hx) (Hc _ _ _ Hin Hx)).
  - intros d Hin. exact (Hgood _ (wi_apply_deps a Hinv _ Hin) (Hap _ Hin)).
  - intros d Hin.
    exact (Hgood _ (wi_closure_deps a Hinv _ Hin) (Hcl _ Hin)).
Qed.

Lemma winv_fresh (a : acc) :
  winv a -> winv (snd (fresh a)) ∧ good (snd (fresh a)) (fst (fresh a))
            ∧ grows a (snd (fresh a)).
Proof.
  intros Hinv.
  pose proof (ext_close a (snd (fresh a)) Hinv (ext_fresh a a)) as [H1 H2].
  split; [ exact H1 | split; [ | exact H2 ] ].
  apply (ok_close a); [ exact Hinv | apply ext_fresh | ].
  apply ok_fresh. lia.
Qed.

Lemma winv_add_allowed (a : acc) (L : list fact) :
  winv a ->
  (∀ fct x, fct ∈ L -> x ∈ into_nodes fct ->
     good a x ∨ ∃ n, Fact_accessor x known n ∈ acc_facts a
                     ∧ widget_fact_ok fct n) ->
  (∀ w n, Fact_accessor w known n ∉ L) ->
  winv (add_facts L a) ∧ grows a (add_facts L a).
Proof.
  intros Hinv Hi Hw.
  assert (Hwid : ∀ w, widget (add_facts L a) w -> widget a w).
  { intros w [n Hin]. cbn in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ destruct (Hw _ _ Hin) | exists n; exact Hin ]. }
  assert (Hgood : ∀ x, good a x -> good (add_facts L a) x).
  { intros x [Hlt Hnw]. split; [ exact Hlt | ].
    intros Hx. exact (Hnw (Hwid _ Hx)). }
  split; [ | split; [ cbn; lia | intros w Hx; left; exact (Hwid _ Hx) ] ].
  constructor; cbn [acc_next acc_facts acc_code_deps acc_apply_deps
    acc_closure_deps add_facts].
  - intros fct x Hin Hx. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (wi_into a Hinv _ _ Hin Hx) ].
    destruct (Hi _ _ Hin Hx) as [[Hlt _] | (n & Hn & _)]; [ exact Hlt | ].
    exact (wi_into a Hinv _ _ Hn (or_introl eq_refl)).
  - intros w Hx. exact (wi_made a Hinv w (Hwid _ Hx)).
  - intros w n fct Hwn Hfct Hx. apply in_app_or in Hwn as [Hwn | Hwn];
      [ destruct (Hw _ _ Hwn) | ].
    apply in_app_or in Hfct as [Hfct | Hfct];
      [ | exact (wi_widget a Hinv _ _ _ Hwn Hfct Hx) ].
    destruct (Hi _ _ Hfct Hx) as [[_ Hnw] | (n' & Hn' & Hok)].
    + destruct (Hnw (ex_intro _ n Hwn)).
    + destruct (wi_widget a Hinv _ _ _ Hwn Hn' (or_introl eq_refl))
        as [_ <-].
      exact Hok.
  - intros id cd x Hin Hx. exact (Hgood _ (wi_code_deps a Hinv _ _ _ Hin Hx)).
  - intros d Hin. exact (Hgood _ (wi_apply_deps a Hinv _ Hin)).
  - intros d Hin. exact (Hgood _ (wi_closure_deps a Hinv _ Hin)).
Qed.

Lemma winv_push_apply_dep (a : acc) (d : apply_dep) :
  winv a -> good a (ad_apply_call_witness d) ->
  winv (push_apply_dep d a) ∧ grows a (push_apply_dep d a).
Proof.
  intros Hinv Hd.
  apply (ext_close a); [ exact Hinv | apply ext_push_apply_dep ].
  apply ok_good. exact Hd.
Qed.

Lemma ext_unknown_apply_deps (a0 : acc) (returns : list node)
    (exn called : node) (aw : list (list simple * node)) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: called :: returns -> ok_node a0 a x) ->
  (∀ first w, (first, w) ∈ aw -> ok_node a0 a w) ->
  ext a0 a (unknown_arity_apply_deps returns exn called aw a).
Proof.
  revert a. induction aw as [ | [first w] rest IH]; intros a Hle Hin Haw;
    cbn [unknown_arity_apply_deps]; [ apply ext_refl | ].
  set (b := add_facts _ a).
  assert (Hbn : acc_next b = acc_next a) by reflexivity.
  assert (H1 : ext a0 a b).
  { apply ext_add_facts; [ | no_widget ].
    intros fct x H Hx. into_split H; into_close H Hx; apply Hin;
      [ left | right; left ]; reflexivity. }
  apply (ext_trans _ _ _ _ H1).
  destruct rest as [ | [g nw] rest'].
  - apply ext_add_facts; [ | no_widget ].
    intros fct x H Hx. into_split H; into_close H Hx.
    match goal with
    | Hy : (?i, ?r) ∈ indexed returns |- _ =>
      assert (Hr : ok_node a0 a r)
        by (apply Hin; right; right; eapply indexed_in; exact Hy)
    end.
    mono.
  - fresh_step a0 b v c E Hv. cbv zeta.
    set (d := add_facts _ c).
    assert (Hdn : acc_next d = acc_next c) by reflexivity.
    assert (H2 : ext a0 c d).
    { pose proof (Haw g nw (or_intror (or_introl eq_refl))) as Hnw.
      apply ext_add_facts; [ | no_widget ].
      intros fct x H Hx. into_split H; into_close H Hx; [ exact Ho | ].
      mono. }
    apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H2).
    apply IH; [ lia | | ].
    + intros x Hx. pose proof (Hin x Hx). mono.
    + intros first' w' Hw'. pose proof (Haw _ _ (or_intror Hw')). mono.
Qed.

Lemma ext_make_unknown (a0 : acc) (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) :
  (acc_next a0 <= acc_next a)%positive ->
  (∀ x, x ∈ exn :: returns -> ok_node a0 a x) ->
  ext a0 a (snd (make_unknown_arity_apply_widget env ap returns exn a))
  ∧ ok_node a0 (snd (make_unknown_arity_apply_widget env ap returns exn a))
      (fst (make_unknown_arity_apply_widget env ap returns exn a)).
Proof.
  intros Hle Hin. unfold make_unknown_arity_apply_widget.
  fresh_step a0 a called b E Hc. cbv beta iota zeta.
  set (b1 := add_fact _ b).
  assert (H1 : ext a0 b b1).
  { apply ext_add_fact; [ intros x [] | intros ? ? ?; discriminate ]. }
  set (groups := at_least_one_group _).
  destruct (ext_fresh_list a0 (length groups) b1) as [H2 H2o];
    [ nexts; lia | ].
  destruct (fresh_list (length groups) b1) as [ws b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota zeta.
  destruct (note_simples_fields (ap_args ap) b2) as (N1 & _).
  pose proof (ext_note_simples a0 (ap_args ap) b2) as H3.
  set (b3 := note_simples (ap_args ap) b2) in *.
  assert (H4 : ext a0 b3
                 (unknown_arity_apply_deps returns exn called
                    (combine groups ws) b3)).
  { apply ext_unknown_apply_deps; [ nexts; lia | | ].
    - intros x [<- | [<- | Hx]].
      + pose proof (Hin exn (or_introl eq_refl)). mono.
      + mono.
      + pose proof (Hin x (or_intror Hx)). mono.
    - intros first w Hw. pose proof (H2o w (in_combine_r _ _ _ _ Hw)).
      mono. }
  set (b4 := unknown_arity_apply_deps _ _ _ _ b3) in *.
  fresh_step a0 b4 an b5 E5 Han.
  assert (H05 : ext a0 a b5).
  { apply (ext_trans _ _ _ _ He). apply (ext_trans _ _ _ _ H1).
    apply (ext_trans _ _ _ _ H2). apply (ext_trans _ _ _ _ H3).
    apply (ext_trans _ _ _ _ H4). exact He0. }
  destruct ws as [ | first rest]; cbn [fst snd].
  - split; [ exact H05 | exact Ho0 ].
  - pose proof (H2o first (or_introl eq_refl)) as Hf.
    assert (H6 : ext a0 b5 (cond_alias env an first b5)).
    { apply ext_cond_alias. mono. }
    split; [ exact (ext_trans _ _ _ _ H05 H6) | ].
    unfold cond_alias. destruct (env_current_code_id env); cbn;
      exact Ho0.
Qed.

Lemma note_simple_fields (s : simple) (a : acc) :
  acc_next (note_simple s a) = acc_next a
  ∧ acc_facts (note_simple s a) = acc_facts a
  ∧ acc_code_deps (note_simple s a) = acc_code_deps a
  ∧ acc_apply_deps (note_simple s a) = acc_apply_deps a
  ∧ acc_closure_deps (note_simple s a) = acc_closure_deps a.
Proof. destruct s; cbn; auto. Qed.

(* The step that makes a widget: the call of known arity, with [Auto]. *)
Lemma tstep_call_kind (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) :
  env_should_preserve_direct_calls env = Preserve_auto ->
  winv a -> (∀ x, x ∈ exn_arg :: return_args -> good a x) ->
  winv (traverse_call_kind env ap exn_arg return_args a)
  ∧ grows a (traverse_call_kind env ap exn_arg return_args a).
Proof.
  intros Hpres Hinv Hgood. unfold traverse_call_kind. cbv zeta.
  destruct (note_simple_fields (ap_callee ap) a) as (N1 & N2 & N3 & N4 & N5).
  pose proof (ext_note_simple a (ap_callee ap) a) as H1.
  set (a1 := note_simple (ap_callee ap) a) in *. clearbody a1.
  set (closure := node_of_simple (ap_callee ap)).
  assert (Hin1 : ∀ x, x ∈ exn_arg :: return_args -> ok_node a a1 x).
  { intros x Hx. apply ok_good. exact (Hgood x Hx). }
  destruct (ap_call_kind ap) as [id | ].
  - destruct (make_known_spec a env ap return_args exn_arg a1)
      as (Hk & Hcw & Hn & Hc & Hap & Hcl & Hf); [ lia | exact Hin1 | ].
    destruct (make_known_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. subst cw.
    cbv beta iota zeta. rewrite Hpres.
    destruct (ext_close a a1 Hinv H1) as [Hinv1 _].
    destruct (ext_close a a2 Hinv (ext_trans _ _ _ _ H1 Hk))
      as [Hinv2 Hg2].
    set (p := Pos.succ (Pos.succ (acc_next a1))) in *.
    set (a3 := add_fact (Fact_accessor (Node_made p) known closure) a2).
    assert (Hlt1 : ∀ x, made_lt x (acc_next a1) -> made_lt x p).
    { intros x Hx. apply (made_lt_mono _ _ _ Hx). subst p. lia. }
    assert (Hinv3 : winv a3).
    { apply winv_add_widget; [ exact Hinv2 | rewrite Hn; subst p; lia | | | | ].
      - intros fct x Hfct Hx. destruct (Hf _ _ Hfct Hx) as [H | H];
          [ | exact H ].
        apply Hlt1. exact (wi_into a1 Hinv1 _ _ H Hx).
      - intros i cd x Hin Hx. rewrite Hc in Hin. apply Hlt1.
        exact (proj1 (wi_code_deps a1 Hinv1 _ _ _ Hin Hx)).
      - intros d Hin. rewrite Hap in Hin. apply Hlt1.
        exact (proj1 (wi_apply_deps a1 Hinv1 _ Hin)).
      - intros d Hin. rewrite Hcl in Hin. apply Hlt1.
        exact (proj1 (wi_closure_deps a1 Hinv1 _ Hin)). }
    assert (Hg3 : grows a a3).
    { split; [ subst a3; cbn; destruct Hg2; lia | ].
      intros w [m [Hm | Hm]].
      - injection Hm as <- _. right. exists p. split; [ reflexivity | ].
        subst p. lia.
      - exact (proj2 Hg2 w (ex_intro _ m Hm)). }
    unfold add_apply.
    destruct (winv_fresh a3 Hinv3) as (Hinvb1 & Hcif & Hgb1).
    destruct (fresh a3) as [cif b1] eqn:Eb1. cbn [fst snd] in *.
    assert (Hfb1 : acc_facts b1 = acc_facts a3).
    { unfold fresh in Eb1. injection Eb1 as _ <-. reflexivity. }
    destruct (winv_fresh b1 Hinvb1) as (Hinvb2 & Hwif & Hgb2).
    destruct (fresh b1) as [wif b2] eqn:Eb2. cbn [fst snd] in *.
    assert (Hfb2 : acc_facts b2 = acc_facts b1).
    { unfold fresh in Eb2. injection Eb2 as _ <-. reflexivity. }
    cbv beta iota zeta.
    destruct (winv_add_allowed b2
                [Fact_alias_if_any_source closure cif closure;
                 Fact_alias_if_any_source closure (Node_made p) wif])
      as [Hinvb3 Hgb3]; [ exact Hinvb2 | | no_widget | ].
    { intros fct x H Hx. into_split H; into_close H Hx.
      - left. exact (good_grows _ _ _ Hgb2 Hcif).
      - right. exists closure. split; [ | reflexivity ].
        rewrite Hfb2, Hfb1. left. reflexivity. }
    destruct (winv_push_apply_dep
                (add_facts
                   [Fact_alias_if_any_source closure cif closure;
                    Fact_alias_if_any_source closure (Node_made p) wif] b2)
                {| ad_function_containing_apply_expr :=
                     env_current_code_id env;
                   ad_apply_code_id := id;
                   ad_apply_closure := cif;
                   ad_apply_call_witness := wif |})
      as [Hinv4 Hg4]; [ exact Hinvb3 | cbn; exact (good_grows _ _ _ Hgb3 Hwif)
                      | ].
    split; [ exact Hinv4 | ].
    apply (grows_trans _ _ _ Hg3). apply (grows_trans _ _ _ Hgb1).
    apply (grows_trans _ _ _ Hgb2). apply (grows_trans _ _ _ Hgb3).
    exact Hg4.
  - destruct (ext_make_unknown a env ap return_args exn_arg a1)
      as [Hk Hcw]; [ lia | exact Hin1 | ].
    destruct (make_unknown_arity_apply_widget env ap return_args exn_arg a1)
      as [cw a2] eqn:E. cbn [fst snd] in *. cbv beta iota.
    apply (ext_close a); [ exact Hinv | ].
    apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ Hk).
    apply ext_add_fact; [ intros x [<- | []]; exact Hcw | ].
    intros ? ? ?; discriminate.
Qed.

Lemma tstep_traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc) :
  env_should_preserve_direct_calls env = Preserve_auto ->
  tstep env a (traverse_apply env ap a).
Proof.
  intros Hpres Hinv Henv. unfold traverse_apply.
  destruct (ext_find_cont a env (ap_continuation ap)
              (length (unarize (ap_return_arity ap))) a Henv)
    as [H1 H1o]; [ lia | ].
  destruct (find_cont env (ap_continuation ap) _ a) as [rets b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota.
  destruct (ext_find_cont a env (ap_exn_continuation ap)
              (S (length (ap_exn_extra_args ap))) b1 Henv)
    as [H2 H2o]; [ nexts; lia | ].
  destruct (find_cont env (ap_exn_continuation ap) _ b1) as [eps b2] eqn:E2.
  cbn [fst snd] in *. cbv beta iota.
  assert (Hpre : ∀ (exn_arg : node) (b3 : acc),
            ext a b2 b3 -> ok_node a b3 exn_arg ->
            winv (traverse_call_kind env ap exn_arg rets b3)
            ∧ grows a (traverse_call_kind env ap exn_arg rets b3)).
  { intros exn_arg b3 H3 Hex.
    assert (H03 : ext a a b3)
      by exact (ext_trans _ _ _ _ (ext_trans _ _ _ _ H1 H2) H3).
    destruct (ext_close a b3 Hinv H03) as [Hinv3 Hg3].
    destruct (tstep_call_kind env ap exn_arg rets b3 Hpres Hinv3)
      as [Hinv4 Hg4].
    - intros x [<- | Hx]; apply (ok_close a); try assumption.
      pose proof (H1o x Hx). nexts. eapply ok_node_mono; [ | eassumption ].
      lia.
    - split; [ exact Hinv4 | exact (grows_trans _ _ _ Hg3 Hg4) ]. }
  destruct eps as [ | ep extra].
  - destruct (fresh b2) as [x b3] eqn:E3. cbv beta iota.
    destruct (fresh_spec a b2 x b3 E3) as (_ & _ & He & Ho);
      [ nexts; lia | ].
    exact (Hpre x b3 He Ho).
  - cbv zeta beta iota.
    pose proof (ext_note_simples a (exn_extra_simples ap) b2) as H3.
    destruct (note_simples_fields (exn_extra_simples ap) b2) as (N1 & _).
    set (b3 := note_simples _ b2) in *. clearbody b3.
    apply Hpre.
    + apply (ext_trans _ _ _ _ H3). apply ext_add_facts; [ | no_widget ].
      intros fct y H Hy. into_split H; into_close H Hy.
      match goal with
      | Hc : (?p, ?s) ∈ combine extra _ |- _ =>
        pose proof (H2o p (or_intror (in_combine_l _ _ _ _ Hc)))
      end.
      mono.
    + pose proof (H2o ep (or_introl eq_refl)). mono.
Qed.

Lemma ext_apply_cont_deps (a0 : acc) (env : traverse_env) (ac : apply_cont)
    (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (apply_cont_deps env ac a).
Proof.
  intros Henv Hle. unfold apply_cont_deps. cbv zeta.
  destruct (ext_find_cont a0 env (ac_continuation ac) (length (ac_args ac)) a
              Henv Hle) as [H1 H1o].
  destruct (find_cont env (ac_continuation ac) _ a) as [params b1] eqn:E1.
  cbn [fst snd] in *. cbv beta iota zeta.
  pose proof (ext_note_simples a0 (ac_args ac) b1) as H2.
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ H2).
  apply ext_add_facts; [ | no_widget ].
  intros fct y H Hy. into_split H; into_close H Hy.
  match goal with
  | Hc : (?p, ?s) ∈ combine params _ |- _ =>
    pose proof (H1o p (in_combine_l _ _ _ _ Hc))
  end.
  mono.
Qed.

Lemma ext_traverse_switch (a0 : acc) (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  env_good env a0 -> (acc_next a0 <= acc_next a)%positive ->
  ext a0 a (traverse_switch env scrutinee arms a).
Proof.
  intros Henv Hle. unfold traverse_switch. cbv zeta.
  pose proof (ext_cond_any_usage a0 env scrutinee a) as H1.
  apply (ext_trans _ _ _ _ H1). apply ext_fold.
  intros [i ac] b _ Hb. apply ext_apply_cont_deps; [ exact Henv | ].
  pose proof (ext_next _ _ _ H1). lia.
Qed.

Lemma ext_exn_handler_deps (a0 : acc) (env : traverse_env)
    (h : continuation_handler) (a : acc) :
  ext a0 a (exn_handler_deps env h a).
Proof.
  unfold exn_handler_deps. destruct h as [[ | p ps] md body];
    [ apply ext_refl | ].
  destruct (hm_is_exn_handler md); [ | apply ext_refl ].
  match goal with
  | |- ext _ _ (cond_any_usage _ _ ?b) => apply (ext_trans _ _ b)
  end; [ | apply ext_cond_any_usage ].
  apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
  intros ? ? ?; discriminate.
Qed.

(* ------------------------------------------------------------------------ *)
(* The traversal                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma good_not_made (a : acc) (x : node) :
  winv a -> not_made x -> good a x.
Proof.
  intros Hinv Hx. split; [ intros k -> ; destruct (Hx k eq_refl) | ].
  intros Hw. destruct (wi_made a Hinv x Hw) as [k ->].
  exact (Hx k eq_refl).
Qed.

Lemma tstep_env (env env' : traverse_env) (a a' : acc) :
  (winv a -> env_good env a -> env_good env' a) ->
  tstep env' a a' -> tstep env a a'.
Proof. intros H H' Hinv Henv. exact (H' Hinv (H Hinv Henv)). Qed.

Lemma env_good_add_cont (env : traverse_env) (k : continuation)
    (params : list node) (a : acc) :
  env_good env a -> (∀ x, x ∈ params -> good a x) ->
  env_good (add_cont env k params) a.
Proof.
  intros Henv Hp k' params' x [Heq | Hin] Hx.
  - injection Heq as <- <-. exact (Hp x Hx).
  - exact (Henv _ _ _ Hin Hx).
Qed.

Lemma handler_params_good (a : acc) (h : continuation_handler) :
  winv a -> ∀ x, x ∈ handler_params h -> good a x.
Proof.
  intros Hinv x Hx. destruct h as [params md body]. cbn in Hx.
  apply in_map_iff in Hx. destruct Hx as [y [<- _]].
  apply good_not_made; [ exact Hinv | intros k; discriminate ].
Qed.

Lemma env_good_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) (a : acc) :
  winv a -> env_good env a -> env_good (add_rec_conts env handlers) a.
Proof.
  intros Hinv Henv. unfold add_rec_conts.
  induction handlers as [ | [k h] handlers IH]; cbn [fold_right];
    [ exact Henv | ].
  apply env_good_add_cont; [ exact IH | ].
  apply handler_params_good. exact Hinv.
Qed.

Lemma preserve_add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) :
  env_should_preserve_direct_calls (add_rec_conts env handlers)
  = env_should_preserve_direct_calls env.
Proof.
  unfold add_rec_conts.
  induction handlers as [ | [k h] handlers IH]; cbn; [ reflexivity | ].
  exact IH.
Qed.

Lemma tstep_traverse :
  (∀ e env a, env_should_preserve_direct_calls env = Preserve_auto ->
     tstep env a (traverse env e a))
  ∧ (∀ h env a, env_should_preserve_direct_calls env = Preserve_auto ->
       tstep env a (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH env a Hpres. cbn [traverse].
    apply (tstep_trans _ _ (traverse_named env x d a)); [ | apply IH, Hpres ].
    apply tstep_ext. intros _ _. apply ext_traverse_named. lia.
  - intros bindings body IHc IH env a Hpres. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ | apply IH, Hpres ].
    eapply tstep_trans; [ | apply tstep_fold ].
    + eapply tstep_trans; [ | apply tstep_fold ].
      * apply tstep_fold. intros b a' _. apply tstep_ext. intros _ _.
        apply ext_prepare_binding. lia.
      * intros b a' _. apply tstep_ext. intros _ _.
        apply ext_traverse_static_binding.
    + intros [id c | id | s const] a' Hin; cbn [traverse_binding];
        [ | apply tstep_refl | apply tstep_refl ].
      destruct c as [params mc ret exn md cbody]. cbn [traverse_code].
      destruct (lookup id (acc_code_deps a')) as [cd | ] eqn:Ecd;
        [ | apply tstep_refl ].
      set (b := add_facts _ a').
      apply (tstep_trans _ _ b).
      * apply tstep_ext. intros _ _. apply ext_add_facts; [ | no_widget ].
        intros fct y H Hy. into_split H; into_close H Hy; ok_tac.
      * apply (tstep_env _ (code_env env id ret exn cd));
          [ | apply (IHc id (Code params mc ret exn md cbody) Hin); cbn;
              exact Hpres ].
        intros Hinvb _ k ps y Hk Hy.
        assert (Hin' : (id, cd) ∈ acc_code_deps b)
          by exact (FlambdaMapProofs.lookup_in _ _ _ Ecd).
        cbn [code_env env_conts] in Hk.
        destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
          apply (wi_code_deps b Hinvb id cd y Hin');
          cbn [code_dep_nodes];
          [ right; left; destruct Hy as [Hy | []]; exact Hy
          | right; right; exact Hy ].
  - intros k h body IHh IH env a Hpres. cbn [traverse]. cbv zeta.
    eapply tstep_trans; [ apply IHh, Hpres | ].
    eapply tstep_trans; [ apply tstep_ext; intros _ _;
                          apply ext_exn_handler_deps | ].
    apply (tstep_env _ (add_cont env k (handler_params h))); [ | apply IH;
      exact Hpres ].
    intros Hinv Henv. apply env_good_add_cont; [ exact Henv | ].
    apply handler_params_good. exact Hinv.
  - intros handlers body IHh IH env a Hpres. cbn [traverse]. cbv zeta.
    apply (tstep_env _ (add_rec_conts env handlers));
      [ intros Hinv Henv; apply env_good_add_rec_conts; assumption | ].
    eapply tstep_trans; [ | apply IH; rewrite preserve_add_rec_conts;
                            exact Hpres ].
    apply tstep_fold. intros [k h] a' Hin. apply (IHh k h Hin).
    rewrite preserve_add_rec_conts. exact Hpres.
  - intros ap env a Hpres. apply tstep_traverse_apply. exact Hpres.
  - intros ac env a Hpres. apply tstep_ext. intros _ Henv.
    apply ext_apply_cont_deps; [ exact Henv | lia ].
  - intros scrutinee arms env a Hpres. apply tstep_ext. intros _ Henv.
    apply ext_traverse_switch; [ exact Henv | lia ].
  - intros env a Hpres. apply tstep_refl.
  - intros params md handler IH env a Hpres. cbn [traverse_handler].
    apply IH. exact Hpres.
Qed.

(* ------------------------------------------------------------------------ *)
(* The unit                                                                 *)
(* ------------------------------------------------------------------------ *)

Lemma ext_resolve (a : acc) (unit_symbols : list symbol) :
  winv a -> ext a a (resolve_delayed_deps unit_symbols a).
Proof.
  intros Hinv. unfold resolve_delayed_deps. cbv zeta.
  assert (Hcd : ∀ id cd x, lookup id (acc_code_deps a) = Some cd ->
                  x ∈ code_dep_nodes cd -> ok_node a a x).
  { intros id cd x E Hx. apply ok_good.
    exact (wi_code_deps a Hinv id cd x (FlambdaMapProofs.lookup_in _ _ _ E)
             Hx). }
  set (b1 := add_facts _ a).
  assert (H1 : ext a a b1).
  { apply ext_add_facts.
    - intros fct y H Hy. apply in_flat_map in H. destruct H as [d [Hd H]].
      pose proof (ok_good a a _ (wi_apply_deps a Hinv d Hd)) as Hw.
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup (ad_apply_code_id d) (acc_code_deps a)) as [cd | ]
        eqn:E;
        destruct (ad_function_containing_apply_expr d);
        into_split H; into_close H Hy; try exact Hw; try ok_tac;
        apply (Hcd _ cd _ E); left; reflexivity.
    - intros w n H. apply in_flat_map in H. destruct H as [d [_ H]].
      unfold apply_dep_facts, alias_for_caller in H.
      destruct (lookup _ _); destruct (ad_function_containing_apply_expr d);
        into_split H; discriminate. }
  assert (Hcl : ∀ d, d ∈ acc_closure_deps b1 ->
                  ok_node a a (cld_let_bound_name_of_the_closure d)).
  { intros d Hd. apply ok_good. exact (wi_closure_deps a Hinv d Hd). }
  set (b2 := List.fold_left _ (acc_closure_deps b1) b1).
  assert (H2 : ext a b1 b2).
  { apply ext_fold. intros d b Hd Hb. unfold add_closure_dep.
    pose proof (Hcl d Hd) as Hc.
    destruct (lookup (cld_closure_code_id d) (acc_code_deps a)) as [cd | ]
      eqn:E.
    - apply ext_add_facts; [ | no_widget ].
      intros fct y H Hy. into_split H; into_close H Hy;
        try (eapply ok_node_mono; [ | exact Hc ];
             cbn [b1 acc_next add_facts] in Hb; lia).
      all: eapply ok_node_mono; [ | apply (Hcd _ cd _ E); left; reflexivity ];
        cbn [b1 acc_next add_facts] in Hb; lia.
    - fresh_step a b w b' E' Hw'.
      cbv beta iota. apply (ext_trans _ _ _ _ He).
      apply ext_add_facts; [ | no_widget ].
      intros fct y H Hy. into_split H; into_close H Hy; try exact Ho.
      all: eapply ok_node_mono; [ | exact Hc ];
        cbn [b1 acc_next add_facts] in Hb; lia. }
  apply (ext_trans _ _ _ _ H1). apply (ext_trans _ _ _ _ H2).
  apply ext_add_facts; [ | no_widget ].
  intros fct y H Hy. into_split H; into_close H Hy; ok_tac.
Qed.

Lemma winv_empty : winv empty_acc.
Proof.
  constructor; cbn; try (intros; contradiction).
  intros w [n []].
Qed.

Lemma winv_traverse_unit (u : unit_interface) (e : expr) :
  winv (traverse_unit_acc Auto u e).
Proof.
  unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (Hinv1 : winv a1).
  { apply (ext_close empty_acc); [ exact winv_empty | ].
    apply ext_add_fact; [ intros y [<- | []]; ok_tac | ].
    intros ? ? ?; discriminate. }
  destruct (winv_fresh a1 Hinv1) as (Hinv2 & Hr & Hg2).
  destruct (fresh a1) as [r a2] eqn:E2. cbn [fst snd] in *.
  destruct (winv_fresh a2 Hinv2) as (Hinv3 & Hex & Hg3).
  destruct (fresh a2) as [ex a3] eqn:E3. cbn [fst snd] in *.
  cbv beta iota zeta.
  set (a4 := add_fact (Fact_any_usage ex) a3).
  assert (H4 : winv a4 ∧ grows a3 a4).
  { apply (ext_close a3); [ exact Hinv3 | ].
    apply ext_add_fact; [ intros y [] | intros ? ? ?; discriminate ]. }
  destruct H4 as [Hinv4 Hg4].
  match goal with
  | |- winv (resolve_delayed_deps _ (add_fact _ (traverse ?env e a4))) =>
    set (env0 := env)
  end.
  assert (Henv : env_good env0 a4).
  { intros k ps x Hk Hx. cbn [env0 env_conts] in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hx as [<- | []].
    - exact (good_grows _ _ _ Hg4 Hex).
    - exact (good_grows _ _ _ Hg4 (good_grows _ _ _ Hg3 Hr)). }
  destruct (proj1 tstep_traverse e env0 a4 eq_refl Hinv4 Henv)
    as [Hinv5 _].
  set (a5 := traverse env0 e a4) in *.
  assert (Hinv6 : winv (add_fact (Fact_any_usage r) a5)).
  { apply (ext_close a5); [ exact Hinv5 | ].
    apply ext_add_fact; [ intros y [] | intros ? ? ?; discriminate ]. }
  apply (ext_close _ _ Hinv6). apply ext_resolve. exact Hinv6.
Qed.

Lemma facts_of_list_in {B : Type} (g : fact -> list B) (l : list fact)
    (y : B) :
  y ∈ flat_map g l -> ∃ fct, fct ∈ l ∧ y ∈ g fct.
Proof. intros H. apply in_flat_map in H. exact H. Qed.

(* One conjunct of [only_read_into] for a widget [x] whose facts are
   [widget_fact_ok] ([Hok]). *)
Ltac read_only_conj x Hok :=
  first
    [ lazymatch goal with
      | |- negb (mem_node x ?l) = true =>
        let Em := fresh "Em" in
        destruct (mem_node x l) eqn:Em; [ | reflexivity ];
        exfalso; apply mem_node_In in Em;
        cbn [f_any_source f_zero_alloc_source facts_of_list] in Em;
        first
          [ destruct Em
          | let fct := fresh "fct" in let Hf := fresh "Hf" in
            let Hy := fresh "Hy" in
            destruct (facts_of_list_in _ _ _ Em) as [fct [Hf Hy]];
            destruct fct; cbn in Hy; try contradiction;
            destruct Hy as [Hy | []]; subst;
            exact (Hok _ Hf (or_introl eq_refl)) ]
      end
    | let y := fresh "y" in let Hy := fresh "Hy" in
      let fct := fresh "fct" in let Hf := fresh "Hf" in
      let Hy' := fresh "Hy" in
      apply forallb_forall; intros y Hy;
      destruct (facts_of_list_in _ _ _ Hy) as [fct [Hf Hy']];
      destruct fct; cbn in Hy'; try contradiction;
      destruct Hy' as [<- | []]; cbv beta iota;
      repeat match goal with
      | |- context [node_eqb ?z x] =>
        let E := fresh "E" in
        destruct (node_eqb z x) eqn:E; [ apply node_eqb_eq in E; subst z | ]
      end; cbn [negb orb andb]; try reflexivity;
      let Hw := fresh "Hw" in
      pose proof (Hok _ Hf) as Hw; cbn [into_nodes widget_fact_ok] in Hw;
      first
        [ solve [ destruct (Hw (or_introl eq_refl)) ]
        | solve [ destruct (Hw (or_intror (or_introl eq_refl))) ]
        | rewrite (Hw (or_introl eq_refl)), node_eqb_refl; reflexivity
        | destruct (Hw (or_introl eq_refl)) as [-> ->];
          rewrite field_eqb_refl, node_eqb_refl; reflexivity ] ].

Lemma widgets_read_only (u : unit_interface) (e : expr) :
  check_widgets_read_only (traverse_unit Auto u e) = true.
Proof.
  pose proof (winv_traverse_unit u e) as Hinv.
  rewrite traverse_unit_acc_facts.
  set (a := traverse_unit_acc Auto u e) in *. clearbody a.
  unfold check_widgets_read_only. apply forallb_forall.
  intros [[x f] n] Hin. cbv beta iota.
  destruct (facts_of_list_in _ _ _ Hin) as [fct [Hfct H]].
  destruct fct; cbn in H; try contradiction.
  destruct H as [Heq | []]. injection Heq as -> -> ->.
  unfold field_eqb at 1.
  destruct (field_eq_dec f known) as [-> | ]; [ | reflexivity ].
  cbn [negb orb].
  assert (Hok : ∀ fct', fct' ∈ acc_facts a -> x ∈ into_nodes fct' ->
                  widget_fact_ok fct' n)
    by (intros fct' H1 H2; exact (wi_widget a Hinv _ _ _ Hfct H1 H2)).
  unfold only_read_into.
  repeat (apply andb_true_intro; split).
  all: read_only_conj x Hok.
Qed.

End AutoWidgets.

Theorem traverse_widgets_read_only : traverse_widgets_read_only_statement.
Proof. intros u e. exact (AutoWidgets.widgets_read_only u e). Qed.

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
(* A unit that binds each variable once reads only into its variables      *)
(* ======================================================================== *)

(* In a unit with no closure, projection, call or code ([closure_free]),
   a fact puts something into a variable only where it is bound: the
   [Let] of the variable ([from_let]: its read, and any source if the
   read is mutable, or no read), or a parameter of a handler ([var_ok]).
   Where the unit binds each variable once, the read into the variable
   of a [Let] is then the one fact into it, but for any source, which
   the fragment of [local_unboxing] excludes of a read of a node to
   unbox. *)
Module BoundOnce.

Abbreviation env_conts := FlambdaReaperTraversal.env_conts.
Abbreviation into_nodes := AutoWidgets.into_nodes.

(* The read that the [Let] of [x] to [d] makes. *)
Definition read_of (x : variable) (d : named) : option fact :=
  match d with
  | Prim (Unary (Block_load _ _ field)) [block] =>
    Some (Fact_accessor (Node_variable x) (Field_block (Z.to_nat field) Value)
            (node_of_simple block))
  | Prim (Unary (Is_int true)) [arg] =>
    Some (Fact_accessor (Node_variable x) Field_is_int (node_of_simple arg))
  | Prim (Unary Get_tag) [arg] =>
    Some (Fact_accessor (Node_variable x) Field_get_tag (node_of_simple arg))
  | Prim (Unary (Unbox_number bn)) [arg] =>
    Some (Fact_accessor (Node_variable x) (Field_boxed_number bn)
            (node_of_simple arg))
  | _ => None
  end.

Definition mutable_read (d : named) : bool :=
  match d with
  | Prim (Unary (Block_load _ Mutable _)) [_] => true
  | _ => false
  end.

(* What the [Let] of [x] to [d] may put into [x]: its read and, for a
   mutable one, any source; otherwise no read. *)
Definition from_let (x : variable) (d : named) (fct : fact) : Prop :=
  match read_of x d with
  | Some r =>
    fct = r ∨ (mutable_read d = true ∧ fct = Fact_any_source (Node_variable x))
  | None => ∀ f b, fct <> Fact_accessor (Node_variable x) f b
  end.

Section Bound.

Variable e0 : expr.

(* A fact reads into variables only, and puts something into a variable
   as its binder does: a parameter of a handler, with no read, or a
   [Let], as [from_let] says. *)
Definition var_ok (fct : fact) : Prop :=
  (∀ x f b, fct = Fact_accessor x f b -> ∃ v, x = Node_variable v)
  ∧ ∀ v, Node_variable v ∈ into_nodes fct ->
      (v ∈ param_variables e0
       ∧ ∀ f b, fct <> Fact_accessor (Node_variable v) f b)
      ∨ ∃ d, (v, d) ∈ let_bindings e0 ∧ from_let v d fct.

(* The invariant of the accumulator: its facts are [var_ok], the
   constants have any source, and it has no dep. *)
Definition vinv (a : acc) : Prop :=
  (∀ fct, fct ∈ acc_facts a -> var_ok fct)
  ∧ Fact_any_source Node_all_constants ∈ acc_facts a
  ∧ acc_code_deps a = [] ∧ acc_apply_deps a = [] ∧ acc_closure_deps a = [].

(* The variables among the parameters of the continuations in scope are
   parameters of handlers. *)
Definition venv (env : traverse_env) : Prop :=
  ∀ k ps v, (k, ps) ∈ env_conts env -> Node_variable v ∈ ps ->
    v ∈ param_variables e0.

Lemma vinv_add_facts (L : list fact) (a : acc) :
  (∀ fct, fct ∈ L -> var_ok fct) -> vinv a -> vinv (add_facts L a).
Proof.
  intros HL (Hf & Hc & H1 & H2 & H3). cbn.
  split; [ | split; [ apply in_or_app; right; exact Hc | auto ] ].
  intros fct Hin. apply in_app_or in Hin as [Hin | Hin]; auto.
Qed.

Lemma vinv_same (a a' : acc) :
  acc_facts a' = acc_facts a -> acc_code_deps a' = acc_code_deps a ->
  acc_apply_deps a' = acc_apply_deps a ->
  acc_closure_deps a' = acc_closure_deps a -> vinv a -> vinv a'.
Proof.
  intros E1 E2 E3 E4 (Hf & Hc & H1 & H2 & H3). unfold vinv.
  rewrite E1, E2, E3, E4. auto.
Qed.

Lemma vinv_note_simple (s : simple) (a : acc) :
  vinv a -> vinv (note_simple s a).
Proof. destruct s; apply vinv_same; reflexivity. Qed.

Lemma vinv_note_simples (l : list simple) (a : acc) :
  vinv a -> vinv (note_simples l a).
Proof.
  destruct (AutoWidgets.note_simples_fields l a)
    as (_ & H1 & H2 & H3 & H4).
  apply vinv_same; assumption.
Qed.

Lemma var_ok_not_var (fct : fact) :
  (∀ x f b, fct = Fact_accessor x f b -> ∃ v, x = Node_variable v) ->
  (∀ v, Node_variable v ∉ into_nodes fct) -> var_ok fct.
Proof. intros H1 H2. split; [ exact H1 | intros v Hv; destruct (H2 v Hv) ]. Qed.

Lemma var_ok_param (fct : fact) :
  (∀ x f b, fct <> Fact_accessor x f b) ->
  (∀ v, Node_variable v ∈ into_nodes fct -> v ∈ param_variables e0) ->
  var_ok fct.
Proof.
  intros H1 H2. split; [ intros x f b E; destruct (H1 x f b E) | ].
  intros v Hv. left. split; [ exact (H2 v Hv) | ].
  intros f b. apply H1.
Qed.

Lemma var_ok_let (x : variable) (d : named) (fct : fact) :
  (x, d) ∈ let_bindings e0 ->
  (∀ y f b, fct = Fact_accessor y f b -> y = Node_variable x) ->
  (∀ v, Node_variable v ∈ into_nodes fct -> v = x) ->
  from_let x d fct -> var_ok fct.
Proof.
  intros Hin Ha Hx Hfrom. split.
  - intros y f b E. exists x. exact (Ha y f b E).
  - intros v Hv. rewrite (Hx v Hv). right. exists d. auto.
Qed.

Lemma vinv_fresh (a : acc) : vinv a -> vinv (snd (fresh a)).
Proof. apply vinv_same; reflexivity. Qed.

Lemma vinv_fresh_list (n : nat) (a : acc) :
  vinv a -> vinv (snd (fresh_list n a)).
Proof.
  destruct (AutoWidgets.fresh_list_spec n a)
    as (H1 & H2 & H3 & H4 & _).
  apply vinv_same; assumption.
Qed.

Lemma find_cont_vars (env : traverse_env) (k : continuation) (n : nat)
    (a : acc) :
  venv env -> vinv a ->
  vinv (snd (find_cont env k n a))
  ∧ ∀ v, Node_variable v ∈ fst (find_cont env k n a) ->
      v ∈ param_variables e0.
Proof.
  intros Henv Ha. unfold find_cont.
  destruct (lookup k (env_conts env)) as [params | ] eqn:E.
  - split; [ exact Ha | ]. intros v Hv.
    exact (Henv k params v (FlambdaMapProofs.lookup_in _ _ _ E) Hv).
  - destruct (AutoWidgets.fresh_list_spec n a)
      as (_ & _ & _ & _ & _ & Hm).
    destruct (fresh_list n a) as [params b] eqn:Eb. cbn [fst snd] in *.
    split.
    + apply vinv_add_facts; [ | apply (vinv_fresh_list n a) in Ha;
                                rewrite Eb in Ha; exact Ha ].
      intros fct Hin. apply in_map_iff in Hin. destruct Hin as [p [<- _]].
      apply var_ok_not_var; [ intros ? ? ? ?; discriminate | ].
      intros v [].
    + intros v Hv. destruct (Hm _ Hv) as (m & Hm' & _). discriminate Hm'.
Qed.


Ltac fact_split H :=
  repeat match type of H with
  | In _ (_ ++ _) => apply in_app_or in H; destruct H as [H | H]
  | In _ (map _ _) =>
    let y := fresh "y" in let Hy := fresh "Hy" in
    apply in_map_iff in H; destruct H as [y [H Hy]]
  | In _ (_ :: _) => destruct H as [H | H]
  | In _ [] => destruct H
  end.

(* A fact of the [Let] of [x] to [d]. *)
Ltac let_fact Hin :=
  repeat match goal with y : prod _ _ |- _ => destruct y end;
  subst;
  eapply var_ok_let;
  [ exact Hin
  | let y := fresh "y" in let f := fresh "f" in let b := fresh "b" in
    let E := fresh "E" in
    intros y f b E; first [ discriminate E | injection E as <- _ _;
                                             reflexivity ]
  | let v := fresh "v" in let Hv := fresh "Hv" in
    intros v Hv; cbn [AutoWidgets.into_nodes] in Hv;
    destruct Hv as [Hv | []]; injection Hv as ->; reflexivity
  | cbn [from_let read_of mutable_read];
    first [ left; reflexivity | right; split; reflexivity
          | let f := fresh "f" in let b := fresh "b" in
            intros f b; discriminate ] ].

Lemma vinv_cond_any_usage (env : traverse_env) (s : simple) (a : acc) :
  vinv a -> vinv (cond_any_usage env s a).
Proof.
  intros Ha. unfold cond_any_usage. apply (vinv_note_simple s) in Ha.
  destruct (env_current_code_id env); apply vinv_add_facts; try exact Ha;
    intros fct [<- | []]; apply var_ok_not_var;
    try (intros ? ? ? ?; discriminate); intros v Hv; cbn in Hv;
    repeat (destruct Hv as [Hv | Hv]; [ discriminate | ]); contradiction.
Qed.

Lemma vinv_traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) :
  (x, Prim p args) ∈ let_bindings e0 -> vinv a ->
  vinv (traverse_prim env x p args a).
Proof.
  intros Hin Ha. unfold traverse_prim.
  destruct p as [ u | b | v | h | r ];
    [ destruct u | destruct b | destruct v | destruct h | destruct r ];
    repeat match goal with
    | |- vinv (match ?t with _ => _ end) => destruct t
    end.
  all: repeat first
    [ exact Ha
    | apply vinv_note_simple
    | apply vinv_note_simples
    | apply vinv_cond_any_usage
    | match goal with
      | |- vinv (if ?c then _ else _) => destruct c
      end
    | apply vinv_add_facts; [ intros ?fct ?Hfct; fact_split Hfct;
                              let_fact Hin | ]
    | unfold block_deps ].
Qed.


Lemma vinv_block_deps_symbol (sym : symbol) (fields : list simple)
    (a : acc) :
  vinv a -> vinv (block_deps (Node_symbol sym) fields a).
Proof.
  intros Ha. unfold block_deps. apply vinv_add_facts;
    [ | apply vinv_note_simples; exact Ha ].
  intros fct Hfct. fact_split Hfct;
    repeat match goal with y : prod _ _ |- _ => destruct y end; subst;
    apply var_ok_not_var; try (intros ? ? ? ?; discriminate);
    intros v [Hv | []]; discriminate.
Qed.

Lemma vinv_apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc) :
  venv env -> vinv a -> vinv (apply_cont_deps env ac a).
Proof.
  intros Henv Ha. unfold apply_cont_deps. cbv zeta.
  destruct (find_cont_vars env (ac_continuation ac) (length (ac_args ac)) a
              Henv Ha) as [H1 H2].
  destruct (find_cont env (ac_continuation ac) _ a) as [params b] eqn:E.
  cbn [fst snd] in *. cbv beta iota zeta.
  apply vinv_add_facts; [ | apply vinv_note_simples; exact H1 ].
  intros fct Hfct. fact_split Hfct.
  repeat match goal with y : prod _ _ |- _ => destruct y end; subst.
  apply var_ok_param; [ intros ? ? ? ?; discriminate | ].
  intros v [Hv | []]. subst.
  exact (H2 v (in_combine_l _ _ _ _ Hy)).
Qed.

Lemma vinv_traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) :
  venv env -> vinv a -> vinv (traverse_switch env scrutinee arms a).
Proof.
  intros Henv Ha. unfold traverse_switch. cbv zeta.
  apply vinv_cond_any_usage with (env := env) (s := scrutinee) in Ha.
  revert Ha. generalize (cond_any_usage env scrutinee a) as b.
  induction (Keyed.compact Z.eqb arms) as [ | [i ac] l IH]; intros b Hb;
    [ exact Hb | ].
  cbn [List.fold_left]. apply IH. apply vinv_apply_cont_deps; assumption.
Qed.

(* The variables of a part of the unit are those of the unit. *)
Definition sub_vars (lets : list (variable * named)) (params : list variable)
    : Prop :=
  (∀ x d, (x, d) ∈ lets -> (x, d) ∈ let_bindings e0)
  ∧ (∀ p, p ∈ params -> p ∈ param_variables e0).

Lemma vinv_exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) :
  (∀ p, p ∈ handler_param_variables h -> p ∈ param_variables e0) ->
  vinv a -> vinv (exn_handler_deps env h a).
Proof.
  intros Hp Ha. unfold exn_handler_deps.
  destruct h as [[ | p ps] md body]; [ exact Ha | ].
  destruct (hm_is_exn_handler md); [ | exact Ha ].
  apply vinv_cond_any_usage. apply vinv_add_facts; [ | exact Ha ].
  intros fct [<- | []]. apply var_ok_param; [ intros ? ? ? ?; discriminate | ].
  intros v [Hv | []]. injection Hv as <-. apply Hp. left. reflexivity.
Qed.

Lemma venv_add_cont (env : traverse_env) (k : continuation)
    (params : list variable) :
  venv env -> (∀ p, p ∈ params -> p ∈ param_variables e0) ->
  venv (add_cont env k (map Node_variable params)).
Proof.
  intros Henv Hp k' ps v [Heq | Hin] Hv.
  - injection Heq as <- <-. apply in_map_iff in Hv.
    destruct Hv as [p [Hpv Hp']]. injection Hpv as ->. exact (Hp _ Hp').
  - exact (Henv _ _ _ Hin Hv).
Qed.

Lemma vinv_traverse :
  (∀ e : expr, closure_free e = true ->
     sub_vars (let_bindings e) (param_variables e) ->
     ∀ (env : traverse_env) (a : acc), venv env -> vinv a ->
       vinv (traverse env e a))
  ∧ (∀ h : continuation_handler, handler_closure_free h = true ->
       sub_vars (handler_let_bindings h) (handler_param_variables h) ->
       ∀ (env : traverse_env) (a : acc), venv env -> vinv a ->
         vinv (traverse_handler env h a)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH Hcf [Hl Hp] env a Henv Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hd Hbody]. cbn [traverse].
    assert (Hx : (x, d) ∈ let_bindings e0) by (apply Hl; left; reflexivity).
    apply IH; [ exact Hbody | | exact Henv | ].
    { split; [ intros y d' Hy; apply Hl; right; exact Hy | exact Hp ]. }
    destruct d as [s | p args | | | | | ]; try discriminate Hd.
    + cbn [traverse_named]. apply vinv_add_facts;
        [ | apply vinv_note_simple; exact Ha ].
      intros fct [<- | []]. let_fact Hx.
    + apply vinv_traverse_prim; assumption.
  - intros bindings body _ IH Hcf [Hl Hp] env a Henv Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hbs Hbody].
    change (traverse env (Let_static bindings body) a)
      with (traverse env body
              (List.fold_left (fun a b => traverse_binding env b a) bindings
                 (List.fold_left (fun a b => traverse_static_binding env b a)
                    bindings
                    (List.fold_left (fun a b => prepare_binding b a)
                       bindings a)))).
    apply IH; [ exact Hbody | | exact Henv | ].
    { split; intros; [ apply Hl | apply Hp ];
        cbn [let_bindings param_variables];
        apply in_or_app; right; assumption. }
    clear Hl Hp IH Hbody.
    rewrite forallb_forall in Hbs.
    assert (Hblocks : ∀ b, b ∈ bindings ->
              ∃ s tag fields,
                b = Bind_symbol s (Static_const_block tag fields)).
    { intros b Hb. specialize (Hbs b Hb).
      destruct b as [ | | s [ | | tag fields | | | ]]; try discriminate Hbs.
      eauto. }
    clear Hbs.
    assert (Hprep : ∀ a,
              List.fold_left (fun a b => prepare_binding b a) bindings a = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    assert (Hcode : ∀ a,
              List.fold_left (fun a b => traverse_binding env b a) bindings a
              = a).
    { clear -Hblocks. induction bindings as [ | b bindings IHb]; intros a;
        [ reflexivity | ].
      destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
      simpl. apply IHb. intros b' Hb'. apply Hblocks. right. exact Hb'. }
    rewrite Hcode, Hprep. clear Hprep Hcode.
    revert a Ha. induction bindings as [ | b bindings IHb]; intros a Ha;
      [ exact Ha | ].
    destruct (Hblocks b (or_introl eq_refl)) as (s & tag & fields & ->).
    simpl. apply IHb; [ intros b' Hb'; apply Hblocks; right; exact Hb' | ].
    apply vinv_block_deps_symbol, Ha.
  - intros k h body IHh IH Hcf [Hl Hp] env a Henv Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hh Hbody]. cbn [traverse]. cbv zeta.
    cbn [let_bindings param_variables] in Hl, Hp.
    assert (Hph : ∀ p, p ∈ handler_param_variables h -> p ∈ param_variables e0)
      by (intros p Hin; apply Hp; apply in_or_app; left; exact Hin).
    apply IH; [ exact Hbody | | | ].
    + split; intros; [ apply Hl | apply Hp ]; apply in_or_app; right;
        assumption.
    + destruct h as [params md handler]. cbn [handler_params].
      apply venv_add_cont; [ exact Henv | ].
      intros q Hq. apply Hph. cbn. apply in_or_app. left. exact Hq.
    + apply vinv_exn_handler_deps; [ exact Hph | ].
      apply IHh; [ exact Hh | | exact Henv | exact Ha ].
      split; intros; [ apply Hl | apply Hp ]; apply in_or_app; left;
        assumption.
  - intros handlers body IHh IH Hcf [Hl Hp] env a Henv Ha. simpl in Hcf.
    apply andb_true_iff in Hcf as [Hhs Hbody]. cbn [traverse]. cbv zeta.
    cbn [let_bindings param_variables] in Hl, Hp.
    rewrite forallb_forall in Hhs.
    assert (Hsub : ∀ k h, (k, h) ∈ handlers ->
              sub_vars (handler_let_bindings h) (handler_param_variables h)).
    { intros k h Hin. split; intros; [ apply Hl | apply Hp ]; apply in_or_app;
        left; apply in_flat_map; exists (k, h); split; assumption. }
    assert (Henv' : venv (add_rec_conts env handlers)).
    { unfold add_rec_conts. clear -Henv Hsub.
      induction handlers as [ | [k h] hs IHhs]; cbn [fold_right];
        [ exact Henv | ].
      destruct h as [params md handler]. cbn [handler_params].
      apply venv_add_cont.
      - apply IHhs. intros k' h' Hin. apply (Hsub k' h'). right. exact Hin.
      - intros q Hq. apply (proj2 (Hsub k (Handler params md handler)
                                     (or_introl eq_refl))).
        cbn. apply in_or_app. left. exact Hq. }
    apply IH; [ exact Hbody | | exact Henv' | ].
    + split; intros; [ apply Hl | apply Hp ]; apply in_or_app; right;
        assumption.
    + revert a Ha. revert Henv'.
      generalize (add_rec_conts env handlers) as env'. intros env' Henv'.
      assert (Hsub' : ∀ k h, (k, h) ∈ handlers -> (k, h) ∈ handlers) by auto.
      revert Hsub'. generalize handlers at 1 3 as hs.
      intros hs Hsub'. induction hs as [ | [k h] hs IHhs]; intros a Ha;
        [ exact Ha | ].
      simpl. apply IHhs; [ intros k' h' Hin; apply Hsub'; right; exact Hin | ].
      assert (Hin : (k, h) ∈ handlers) by (apply Hsub'; left; reflexivity).
      exact (IHh k h Hin (Hhs _ Hin) (Hsub k h Hin) env' a Henv' Ha).
  - intros ap Hcf. discriminate Hcf.
  - intros ac _ _ env a Henv Ha. apply vinv_apply_cont_deps; assumption.
  - intros scrutinee arms _ _ env a Henv Ha.
    apply vinv_traverse_switch; assumption.
  - intros _ _ env a _ Ha. exact Ha.
  - intros params metadata handler IH Hcf [Hl Hp] env a Henv Ha.
    cbn [traverse_handler]. apply IH; [ exact Hcf | | exact Henv | exact Ha ].
    split; [ exact Hl | ]. intros p Hin. apply Hp. cbn. apply in_or_app.
    right. exact Hin.
Qed.

End Bound.

Lemma vinv_unit (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) :
  closure_free e = true -> vinv e (traverse_unit_acc flag u e).
Proof.
  intros Hcf. unfold traverse_unit_acc. cbv zeta.
  set (a1 := add_fact (Fact_any_source Node_all_constants) empty_acc).
  assert (H1 : vinv e a1).
  { split; [ | split; [ left; reflexivity | auto ] ].
    intros fct [<- | []]. apply var_ok_not_var;
      [ intros ? ? ? ?; discriminate | intros v [Hv | []]; discriminate ]. }
  cbn [fresh]. cbv beta iota.
  match goal with
  | |- vinv _ (resolve_delayed_deps _ (add_fact _ (traverse ?env e ?a))) =>
    assert (Henv : venv e env); [ | assert (Ha : vinv e a) ]
  end.
  { intros k ps v Hk Hv. cbn in Hk.
    destruct Hk as [Hk | [Hk | []]]; injection Hk as <- <-;
      destruct Hv as [Hv | []]; discriminate. }
  { apply vinv_add_facts.
    - intros fct [<- | []]. apply var_ok_not_var;
        [ intros ? ? ? ?; discriminate | intros v [] ].
    - apply (vinv_same e a1); try reflexivity. exact H1. }
  pose proof (proj1 (vinv_traverse e) e Hcf
                (conj (fun x d H => H) (fun p H => H)) _ _ Henv Ha) as H2.
  set (a5 := traverse _ e _) in *. clearbody a5.
  assert (H3 : vinv e (add_fact (Fact_any_usage (Node_made 1)) a5)).
  { apply vinv_add_facts; [ | exact H2 ].
    intros fct [<- | []]. apply var_ok_not_var;
      [ intros ? ? ? ?; discriminate | intros v [] ]. }
  set (a6 := add_fact _ a5) in *. clearbody a6.
  destruct H3 as (Hf & Hc & Hcd & Had & Hcl).
  unfold resolve_delayed_deps. rewrite Hcd, Had. cbn [flat_map].
  cbv zeta. cbn [add_facts acc_closure_deps acc_code_deps app List.fold_left].
  rewrite Hcl. cbn [List.fold_left].
  apply vinv_add_facts.
  - intros fct Hfct. apply in_map_iff in Hfct. destruct Hfct as [s [<- _]].
    apply var_ok_not_var;
      [ intros ? ? ? ?; discriminate | intros v [Hv | []]; discriminate ].
  - split; [ | split; [ | auto ] ]; cbn [acc_facts app]; assumption.
Qed.

(* The [Let]s of a unit of the fragment of [local_unboxing]: of an
   unboxed variable, or of [plain_binding_ok]. *)
Lemma local_lets (unboxed : list (node * unboxed_fields)) (sol : solution) :
  (∀ (e : expr) (kinds : finite_map flambda_kind)
     (conts : finite_map (list (variable * bool))),
     local_expr unboxed sol kinds conts e = true ->
     ∀ v d, (v, d) ∈ let_bindings e ->
       (∃ fs, fields_of unboxed v = Some fs)
       ∨ ∃ kind, plain_binding_ok unboxed v kind d = true)
  ∧ (∀ (h : continuation_handler) (kinds : finite_map flambda_kind)
       (conts : finite_map (list (variable * bool))) (mask : list bool),
       local_handler unboxed sol kinds conts mask h = true ->
       ∀ v d, (v, d) ∈ handler_let_bindings h ->
         (∃ fs, fields_of unboxed v = Some fs)
         ∨ ∃ kind, plain_binding_ok unboxed v kind d = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds conts Hloc v d' Hin. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hd Hbody].
    destruct Hin as [Heq | Hin]; [ | exact (IH _ _ Hbody v d' Hin) ].
    injection Heq as <- <-.
    destruct (fields_of unboxed x) as [fs | ] eqn:E; [ left; eauto | ].
    right. exists kind. exact Hd.
  - intros bindings body _ IH kinds conts Hloc v d Hin. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hbs Hbody].
    cbn [let_bindings] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH _ _ Hbody v d Hin) ].
    apply in_flat_map in Hin. destruct Hin as [b [Hb Hin]].
    rewrite forallb_forall in Hbs. specialize (Hbs b Hb).
    destruct b as [ | | s [ | | | | | ]]; try discriminate Hbs.
    destruct Hin.
  - intros k h body IHh IH kinds conts Hloc v d Hin. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hh Hbody].
    cbn [let_bindings] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ exact (IHh _ _ _ Hh v d Hin) | exact (IH _ _ Hbody v d Hin) ].
  - intros handlers body IHh IH kinds conts Hloc v d Hin. simpl in Hloc.
    apply andb_true_iff in Hloc as [Hhs Hbody].
    cbn [let_bindings] in Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (IH _ _ Hbody v d Hin) ].
    apply in_flat_map in Hin. destruct Hin as [[k h] [Hkh Hin]].
    rewrite forallb_forall in Hhs.
    exact (IHh k h Hkh _ _ _ (Hhs _ Hkh) v d Hin).
  - intros ap kinds conts Hloc. discriminate Hloc.
  - intros ac kinds conts _ v d [].
  - intros scrutinee arms kinds conts _ v d [].
  - intros kinds conts _ v d [].
  - intros params metadata handler IH kinds conts mask Hloc v d Hin.
    simpl in Hloc. apply andb_true_iff in Hloc as [_ Hbody].
    exact (IH _ _ Hbody v d Hin).
Qed.


Lemma nodup_app_disjoint {A : Type} (l l' : list A) (x : A) :
  NoDup (l ++ l') -> x ∈ l -> x ∈ l' -> False.
Proof.
  induction l as [ | y l IH]; intros Hnd Hx Hx'; [ destruct Hx | ].
  inversion Hnd as [ | y' l0 Hy Hnd' ]; subst.
  destruct Hx as [<- | Hx]; [ apply Hy; apply in_or_app; right; exact Hx'
                           | exact (IH Hnd' Hx Hx') ].
Qed.

Lemma nodup_map_fst {A B : Type} (l : list (A * B)) (x : A) (b b' : B) :
  NoDup (map fst l) -> (x, b) ∈ l -> (x, b') ∈ l -> b = b'.
Proof.
  induction l as [ | [y c] l IH]; intros Hnd Hb Hb'; [ destruct Hb | ].
  inversion Hnd as [ | y' l0 Hy Hnd' ]; subst.
  destruct Hb as [Hb | Hb], Hb' as [Hb' | Hb'].
  - congruence.
  - injection Hb as -> ->. exfalso. apply Hy.
    apply in_map_iff. exists (x, b'). auto.
  - injection Hb' as -> ->. exfalso. apply Hy.
    apply in_map_iff. exists (x, b). auto.
  - exact (IH Hnd' Hb Hb').
Qed.

Lemma read_of_field (x : variable) (d : named) (y : node) (f : field)
    (b : node) :
  read_of x d = Some (Fact_accessor y f b) ->
  (∃ i, f = Field_block i Value) ∨ f = Field_is_int ∨ f = Field_get_tag
  ∨ ∃ bn, f = Field_boxed_number bn.
Proof.
  unfold read_of. intros H.
  repeat match type of H with
  | match ?t with _ => _ end = _ => destruct t; try discriminate H
  end;
  injection H as _ <- _; eauto.
Qed.

(* One conjunct of [only_read_into] for a node [x] whose only fact is its
   read ([Hok]). *)
Ltac read_conj x Hok :=
  first
    [ lazymatch goal with
      | |- negb (mem_node x ?l) = true =>
        let Em := fresh "Em" in
        destruct (mem_node x l) eqn:Em; [ | reflexivity ];
        exfalso; apply mem_node_In in Em;
        cbn [f_any_source f_zero_alloc_source facts_of_list] in Em;
        first
          [ destruct Em
          | let fct := fresh "fct" in let Hf := fresh "Hf" in
            let Hy := fresh "Hy" in
            apply in_flat_map in Em; destruct Em as [fct [Hf Hy]];
            destruct fct; cbn in Hy; try contradiction;
            destruct Hy as [Hy | []]; subst;
            discriminate (Hok _ Hf (or_introl eq_refl)) ]
      end
    | let y0 := fresh "y" in let Hy0 := fresh "Hy" in
      let fct := fresh "fct" in let Hf := fresh "Hf" in
      let Hy' := fresh "Hy" in
      apply forallb_forall; intros y0 Hy0;
      apply in_flat_map in Hy0; destruct Hy0 as [fct [Hf Hy']];
      destruct fct; cbn in Hy'; try contradiction;
      destruct Hy' as [<- | []]; cbv beta iota;
      repeat match goal with
      | |- context [node_eqb ?z x] =>
        let E := fresh "E" in
        destruct (node_eqb z x) eqn:E; [ apply node_eqb_eq in E; subst z | ]
      end; cbn [negb orb andb]; try reflexivity;
      first
        [ discriminate (Hok _ Hf (or_introl eq_refl))
        | discriminate (Hok _ Hf (or_intror (or_introl eq_refl)))
        | injection (Hok _ Hf (or_introl eq_refl)) as -> ->;
          rewrite field_eqb_refl, node_eqb_refl; reflexivity ] ].

Lemma reads_read_only (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (to_unbox : list node) :
  is_local_slots_only is_local ->
  variables_bound_once e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  local_unboxing (traverse_unit flag u e) T (derive_dominated T) to_unbox e
  = true ->
  check_reads_read_only is_local (traverse_unit flag u e) (derive_dominated T)
    to_unbox
  = true.
Proof.
  intros Hslots Honce Hc Hloc.
  set (F := traverse_unit flag u e) in *.
  set (unboxed := fst (unit_unboxed_fields F T (derive_dominated T) to_unbox
                         e)).
  assert (Hloc' : local_expr unboxed (solution_of_tables T) [] [] e = true)
    by exact Hloc.
  pose proof (proj1 (FlambdaReaperUnboxingRefinesProofs.local_closure_free
                       unboxed (solution_of_tables T)) e [] [] Hloc')
    as Hcf.
  pose proof (vinv_unit flag u e Hcf) as Hv.
  pose proof (check_closed_sound _ _ _ Hc) as Hcl.
  assert (HF : F = facts_of_list (acc_facts (traverse_unit_acc flag u e)))
    by reflexivity.
  set (a := traverse_unit_acc flag u e) in *. clearbody a.
  destruct Hv as (Hvf & Hconsts & _).
  apply FlambdaMapProofs.distinct_NoDup in Honce.
  (* An allocation that dominates has no any source. *)
  assert (Hany : ∀ n, Fact_any_source n ∈ acc_facts a ->
            mem_node n (t_any_source T) = true).
  { intros n Hn. apply (any_source_input _ _ _ Hcl n). rewrite HF.
    cbn [graph_of_facts g_any_source facts_of_list f_any_source].
    apply in_flat_map. exists (Fact_any_source n).
    split; [ exact Hn | left; reflexivity ]. }
  assert (Hdom : ∀ n, has_to_be_unboxed (derive_dominated T) to_unbox n
                      = true ->
            n ∈ map fst unboxed
            ∧ mem_node n (t_any_source T) = false).
  { intros n Hn. unfold has_to_be_unboxed in Hn.
    apply existsb_exists in Hn. destruct Hn as [[a0 n'] [Hin Htest]].
    apply andb_true_iff in Htest as [Hn' Ha0]. apply node_eqb_eq in Hn'.
    subst n'. split.
    - exact (FlambdaReaperUnboxingRefinesProofs.unboxed_fields_of_names_bound
               F T (bound_code_my_closures e) (derive_dominated T) to_unbox
               _ a0 n Hin Ha0).
    - exact (proj1 (proj2 (proj2
               (dominated_spec T a0 n Hin)))). }
  unfold check_reads_read_only. apply forallb_forall.
  intros [[x f] y] Hacc. cbv beta iota.
  destruct (has_to_be_unboxed (derive_dominated T) to_unbox y) eqn:Hh;
    [ | reflexivity ].
  cbn [negb orb].
  rewrite HF in Hacc. cbn [facts_of_list f_accessor] in Hacc.
  apply in_flat_map in Hacc. destruct Hacc as [fct [Hfct Hacc]].
  destruct fct; cbn in Hacc; try contradiction.
  destruct Hacc as [Heq | []]. injection Heq as -> -> ->.
  destruct (proj1 (Hvf _ Hfct) x f y eq_refl) as [v ->].
  destruct (proj2 (Hvf _ Hfct) v (or_introl eq_refl))
    as [[_ Hna] | (d & Hd & Hfrom)];
    [ exfalso; exact (Hna f y eq_refl) | ].
  unfold from_let in Hfrom.
  destruct (read_of v d) as [r | ] eqn:Er;
    [ | exfalso; exact (Hfrom f y eq_refl) ].
  destruct Hfrom as [Hr | [_ Hr]]; [ subst r | discriminate Hr ].
  assert (Huniq : ∀ d', (v, d') ∈ let_bindings e -> d' = d).
  { intros d' Hd'. apply (nodup_map_fst (let_bindings e) v d' d); auto.
    exact (NoDup_app_remove_r _ _ Honce). }
  assert (Hnp : v ∉ param_variables e).
  { intros Hp. apply (nodup_app_disjoint _ _ v Honce); [ | exact Hp ].
    apply in_map_iff. exists (v, d). auto. }
  assert (Hok : ∀ fct, fct ∈ acc_facts a ->
            Node_variable v ∈ into_nodes fct ->
            fct = Fact_accessor (Node_variable v) f y).
  { intros fct Hfct' Hx.
    destruct (proj2 (Hvf _ Hfct') v Hx) as [[Hp _] | (d' & Hd' & Hfrom')];
      [ contradiction | ].
    rewrite (Huniq d' Hd') in Hfrom'. unfold from_let in Hfrom'.
    rewrite Er in Hfrom'.
    destruct Hfrom' as [-> | [Hmut ->]]; [ reflexivity | exfalso ].
    pose proof (Hany _ Hfct') as Hsv.
    destruct d as [s | p args | | | | | ]; try discriminate Hmut.
    destruct p as [u0 | | | | ]; try discriminate Hmut.
    destruct u0; try discriminate Hmut.
    match goal with
    | m : mutability |- _ => destruct m; try discriminate Hmut
    end.
    destruct args as [ | arg [ | ]]; try discriminate Hmut.
    cbn [read_of] in Er. injection Er as Ef Ey. subst f y.
    destruct (Hdom _ Hh) as [Hin_u Hnot_any].
    destruct (proj1 (local_lets unboxed (solution_of_tables T)) e [] []
                Hloc' v _ Hd) as [[fs Hfs] | [knd Hplain]].
    - apply FlambdaReaperUnboxingRefinesProofs.fields_of_in in Hfs.
      destruct (FlambdaReaperUnboxingRefinesProofs.unboxed_fields_of_names_dom
                  F T (bound_code_my_closures e) (derive_dominated T) to_unbox
                  _ _ _ Hfs) as (a0 & next' & Hin0 & _).
      rewrite (proj1 (proj2 (proj2
                 (dominated_spec T a0 _ Hin0))))
        in Hsv.
      discriminate Hsv.
    - destruct arg as [x' | sym | c].
      + destruct (fields_of unboxed x') as [fs | ] eqn:Ex'.
        * unfold plain_binding_ok, unboxed_read in Hplain.
          cbn [field_read] in Hplain. rewrite Ex' in Hplain.
          destruct knd; discriminate Hplain.
        * exact (FlambdaMapProofs.keyed_lookup_none_not_in node_eqb
                   FlambdaReaperUnboxingRefinesProofs.node_eqb_decides _ _
                   Ex' Hin_u).
      + unfold plain_binding_ok, unboxed_read in Hplain.
        cbn [forallb plain_simple] in Hplain.
        rewrite andb_true_r in Hplain. apply negb_true_iff in Hplain.
        unfold Keyed.is_bound in Hplain.
        destruct (Keyed.lookup node_eqb (Node_symbol sym) unboxed) eqn:El;
          [ discriminate Hplain | ].
        exact (FlambdaMapProofs.keyed_lookup_none_not_in node_eqb
                 FlambdaReaperUnboxingRefinesProofs.node_eqb_decides _ _
                 El Hin_u).
      + cbn [node_of_simple] in Hnot_any.
        rewrite (Hany _ Hconsts) in Hnot_any. discriminate Hnot_any. }
  apply andb_true_intro. split.
  - apply negb_true_iff.
    destruct (is_local f) eqn:El; [ | reflexivity ].
    destruct (Hslots f El) as [[w ->] | [fs ->]];
      destruct (read_of_field _ _ _ _ _ Er) as [[i Hi] | [Hi | [Hi | [bn Hi]]]];
      discriminate Hi.
  - rewrite HF. unfold only_read_into.
    repeat (apply andb_true_intro; split).
    all: read_conj (Node_variable v) Hok.
Qed.

End BoundOnce.

Theorem bound_once_reads_read_only : bound_once_reads_read_only_statement.
Proof. exact BoundOnce.reads_read_only. Qed.

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

(* With [Auto], the traversal passes the check of the graph. *)
Theorem least_rebuild_cc_auto_refines_observably :
  least_rebuild_cc_auto_refines_observably_statement.
Proof.
  intros is_local u e T preserve_direct_calls Hslots Hlocal Hfrag Hstatic
    Hkinded Hwf Hexits Hleast.
  exact (least_rebuild_cc_refines_observably Auto is_local u e T
           preserve_direct_calls Hslots Hlocal Hfrag Hstatic Hkinded Hwf
           Hexits Hleast (traverse_widgets_read_only u e)).
Qed.

(* ======================================================================== *)
(* The rebuild of the unit of [widgets_counterexample_statement]            *)
(* ======================================================================== *)

(* A tuple of a list that computes. *)
Ltac in_computed := vm_compute; repeat first [ left; reflexivity | right ].

(* With [Always] and [Never], [s80] has any usage, and so has the
   witness of [f50] in its field of known arity: [f50] cannot change. *)
Lemma widget_example_cannot_change (flag : reaper_preserve_direct_calls)
    (T : tables) :
  flag <> Auto ->
  check_closed Examples.no_local
    (traverse_unit flag Examples.interface WidgetExample.unit_expr) T
  = true ->
  mem_node (Node_code_id 50%positive)
    (derive_cannot_change
       (traverse_unit flag Examples.interface WidgetExample.unit_expr) T)
  = true.
Proof.
  intros Hflag Hc.
  set (F := traverse_unit flag Examples.interface WidgetExample.unit_expr).
  assert (Hfacts :
    Node_symbol 80%positive ∈ f_any_usage F
    ∧ (Node_symbol 80%positive, Field_call_witness Known_arity_code_pointer,
       Node_made 8%positive) ∈ f_constructor F
    ∧ (Node_made 8%positive, Field_code_id_of_call_witness,
       Node_code_id 50%positive) ∈ f_constructor F).
  { subst F. destruct flag; [ | | | contradiction ];
      (split; [ | split ]); in_computed. }
  destruct Hfacts as (H1 & H2 & H3).
  pose proof (check_closed_sound _ _ _ Hc) as Hcl.
  destruct (FlambdaReaperCallingConventionsProofs.check_cc_closed_sound F T
              (derive_cannot_change F T)
              (proj1
                 (FlambdaReaperCallingConventionsProofs
                  .derive_cannot_change_least F T)))
    as [ccw Hcc].
  assert (Hu : s_any_usage (solution_of_tables T) (Node_made 8%positive)
               = true).
  { apply (any_usage_constructor _ _ _ Hcl (Node_symbol 80%positive)
             (Field_call_witness Known_arity_code_pointer));
      [ exact (any_usage_input _ _ _ Hcl _ H1) | exact H2 | reflexivity ]. }
  exact (cc_code_id_of_witness _ _ _ _ Hcc _ _ H3
           (has_usage_any _ _ _ Hcl _ Hu)
           (cc_witness_any_usage _ _ _ _ Hcc _ Hu)).
Qed.

Theorem widgets_counterexample_refines :
  widgets_counterexample_refines_statement.
Proof.
  split; [ exact widget_example_cannot_change | ].
  intros flag T preserve_direct_calls Hleast.
  assert (Hslots : is_local_slots_only Examples.no_local)
    by (intros f H; discriminate H).
  destruct flag eqn:Ef.
  4: { exact (least_rebuild_cc_auto_refines_observably Examples.no_local
                Examples.interface WidgetExample.unit_expr T
                preserve_direct_calls Hslots (or_introl (fun f => eq_refl))
                ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
                ltac:(vm_compute; reflexivity) ltac:(vm_compute; reflexivity)
                ltac:(vm_compute; reflexivity) Hleast). }
  all: rewrite <- Ef in Hleast |- *.
  all: assert (Hna : flag <> Auto) by (rewrite Ef; discriminate).
  all: unfold rebuild_unit_cc_from_tables;
    rewrite FlambdaReaperCallingConventionsProofs.rebuild_unit_cc_unchanged;
    [ apply (least_rebuild_refines_observably flag Examples.no_local
               Examples.interface WidgetExample.unit_expr T
               preserve_direct_calls Hslots (or_introl (fun f => eq_refl)));
      [ vm_compute; reflexivity .. | exact Hleast ] | ].
  all: intros id Hid;
    change (static_names code_ids_bound WidgetExample.unit_expr)
      with [50%positive] in Hid;
    cbn [among existsb] in Hid; rewrite orb_false_r in Hid;
    apply Pos.eqb_eq in Hid; subst id;
    exact (widget_example_cannot_change flag T Hna (proj1 Hleast)).
Qed.

(* For a unit that binds each variable once, the traversal passes the
   check of the graph. *)
Theorem least_rebuild_unbox_bound_once_refines_observably :
  least_rebuild_unbox_bound_once_refines_observably_statement.
Proof.
  intros flag is_local u e T max_unbox_size lambda_lifting
    preserve_direct_calls Hslots Hlocal Hfrag Hstatic Hkinded Hwf Hexits
    Hbound Hleast Hunboxing.
  apply (least_rebuild_unbox_refines_observably flag is_local); auto.
  exact (bound_once_reads_read_only flag is_local u e T _ Hslots Hbound
           (proj1 Hleast) Hunboxing).
Qed.

