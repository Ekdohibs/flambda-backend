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

(* Proof of the statement of FlambdaRecoverComparisonScopeSpec.v.
   Nothing here needs review. It follows the proof for the sharing of
   primitives in FlambdaKeepsScopeProofs.v, with the arguments of what
   is known where that proof has the variables bound to it. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaCseSpec FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaRecoverComparisonSpec FlambdaRecoverComparisonProofs
  FlambdaRecoverComparisonScopeSpec.
From OxCamlFlambda2 Require FlambdaCseProofs.
Import ListNotations.

(* The variables among the arguments of what is known are among
   [names]. *)
Definition operands_within (names : list variable) (known : computed)
    : Prop :=
  ∀ entry : variable * (prim * list simple),
    In entry known ->
    incl (free_variables_of_simples (snd (snd entry))) names.

Lemma forget_computed_operands_within (names names' : list variable)
    (x : variable) (known : computed) :
  incl names names' ->
  operands_within names known ->
  operands_within names' (forget_computed x known).
Proof.
  intros Hincl Hknown entry Hentry. unfold forget_computed in Hentry.
  apply filter_In in Hentry.
  exact (incl_tran (Hknown entry (proj1 Hentry)) Hincl).
Qed.

Lemma forget_computed_vars_operands_within (names : list variable)
    (xs : list variable) :
  ∀ known : computed,
    operands_within names known ->
    operands_within names (forget_computed_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown;
    cbn [forget_computed_vars]; [ exact Hknown | ].
  apply IH, (forget_computed_operands_within names names), Hknown.
  apply incl_refl.
Qed.

Lemma computed_under_let_operands_within (names : list variable)
    (known : computed) (x : variable) (d : named) :
  operands_within names known ->
  operands_within (x :: names ++ free_variables_of_named d)
    (computed_under_let known x d).
Proof.
  intros Hknown. unfold computed_under_let. cbv zeta.
  assert (Hforgot :
    operands_within (x :: names ++ free_variables_of_named d)
      (forget_computed x known))
    by (apply (forget_computed_operands_within names); [ | exact Hknown ];
        apply incl_tl, incl_appl, incl_refl).
  destruct d as [ | p args | | | | | ]; try exact Hforgot.
  destruct (shareable p && negb (var_in_simples x args)); [ | exact Hforgot ].
  intros entry [<- | Hentry].
  - cbn [fst snd free_variables_of_named].
    apply incl_tl, incl_appr, incl_refl.
  - (* What the untagging of a variable adds has, for its argument, the
       variable that the [Let] binds. *)
    apply in_app_or in Hentry as [Hagain | Hentry];
      [ | exact (Hforgot entry Hentry) ].
    destruct (FlambdaCseProofs.tagged_again_spec x p args entry Hagain)
      as (t & _ & _ & ->).
    intros y [<- | []]. left. reflexivity.
Qed.

(* The arguments of a comparison that is found are those of an entry of
   what is known. *)
Lemma compared_from (known : computed) (tagged : simple)
    (kind : standard_int) (s : signedness) (operands : list simple) :
  compared known tagged = Some (kind, s, operands) ->
  ∃ entry : variable * (prim * list simple),
    In entry known ∧ snd (snd entry) = operands.
Proof.
  destruct tagged as [t | sym | c]; cbn [compared]; try discriminate.
  intros Hcompared.
  apply first_known_sound in Hcompared as (entry & _ & Hheld).
  apply tagged_comparison_held_sound in Hheld as (u & _ & Hcomparison).
  apply first_known_sound in Hcomparison as (entry' & Hentry' & Hheld').
  apply comparison_held_sound in Hheld'. subst entry'.
  eexists. split; [ exact Hentry' | reflexivity ].
Qed.

Lemma recover_one_way_free (names : list variable) (known : computed)
    (tagged zero : simple) (c : int_comparison) (d' : named) :
  operands_within names known ->
  recover_one_way known tagged zero c = Some d' ->
  incl (free_variables_of_named d') names.
Proof.
  intros Hknown. unfold recover_one_way.
  destruct (is_tagged_zero zero); [ | discriminate ].
  destruct (compared known tagged) as [[[kind s] operands] | ] eqn:Hcompared;
    [ | discriminate ].
  destruct (against_zero s c) as [c' | ]; [ | discriminate ].
  intros [= <-].
  destruct (compared_from known tagged kind s operands Hcompared)
    as (entry & Hentry & <-).
  cbn [free_variables_of_named]. exact (Hknown entry Hentry).
Qed.

Lemma recover_comparison_free (names : list variable) (known : computed)
    (d : named) :
  operands_within names known ->
  incl (free_variables_of_named (recover_comparison known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown. unfold recover_comparison.
  destruct d as [ | p args | | | | | ]; try (apply incl_appl, incl_refl).
  destruct args as [ | arg1 [ | arg2 [ | arg3 args]]];
    try (apply incl_appl, incl_refl).
  destruct (tested p) as [c | ]; [ | apply incl_appl, incl_refl ].
  destruct (recover_one_way known arg1 arg2 c) as [d' | ] eqn:Hone.
  - apply incl_appr.
    exact (recover_one_way_free names known arg1 arg2 c d' Hknown Hone).
  - destruct (recover_one_way known arg2 arg1 (swap_comparison c))
      as [d' | ] eqn:Hother; [ | apply incl_appl, incl_refl ].
    apply incl_appr.
    exact (recover_one_way_free names known arg2 arg1 (swap_comparison c)
      d' Hknown Hother).
Qed.

Fixpoint recover_comparisons_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : computed),
    operands_within names known ->
    scoped_under names e (recover_comparisons_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [recover_comparisons_knowing recover_comparisons_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply recover_comparison_free, Hknown | ].
    apply recover_comparisons_knowing_scoped,
      computed_under_let_operands_within, Hknown.
  - apply scoped_under_let_static. apply recover_comparisons_knowing_scoped.
    intros entry Hentry. apply incl_appl. exact (Hknown entry Hentry).
  - apply scoped_under_let_cont; apply recover_comparisons_knowing_scoped;
      [ apply forget_computed_vars_operands_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply recover_comparisons_knowing_scoped, Hknown.
Qed.

Lemma recover_known_comparisons_scoped (e : expr) :
  scoped e (recover_known_comparisons e).
Proof.
  apply scoped_under_none, recover_comparisons_knowing_scoped.
  intros entry [].
Qed.

Theorem recover_known_comparisons_keeps_scope :
  recover_known_comparisons_keeps_scope_statement.
Proof. apply keeps_scope_intro, recover_known_comparisons_scoped. Qed.
