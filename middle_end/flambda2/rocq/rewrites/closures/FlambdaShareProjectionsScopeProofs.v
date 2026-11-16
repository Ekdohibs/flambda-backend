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

(* Proof of the statement of FlambdaShareProjectionsScopeSpec.v.
   Nothing here needs review. It follows the proof for the sharing of
   primitives in FlambdaKeepsScopeProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaNormaliseSpec FlambdaCseSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaShareProjectionsSpec FlambdaShareProjectionsScopeSpec.
Import ListNotations.

(* The variables that are known to hold a projection are among
   [names]. *)
Definition projected_within (names : list variable) (known : projected)
    : Prop :=
  ∀ entry : variable * named, In entry known -> In (fst entry) names.

Lemma forget_projected_within (names names' : list variable) (x : variable)
    (known : projected) :
  incl names names' ->
  projected_within names known ->
  projected_within names' (forget_projected x known).
Proof.
  intros Hincl Hknown entry Hentry. unfold forget_projected in Hentry.
  apply filter_In in Hentry. exact (Hincl _ (Hknown entry (proj1 Hentry))).
Qed.

Lemma forget_projected_vars_within (names : list variable)
    (xs : list variable) :
  ∀ known : projected,
    projected_within names known ->
    projected_within names (forget_projected_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known Hknown;
    cbn [forget_projected_vars]; [ exact Hknown | ].
  apply IH, (forget_projected_within names names), Hknown. apply incl_refl.
Qed.

Lemma projected_under_let_within (names : list variable) (known : projected)
    (x : variable) (d : named) :
  projected_within names known ->
  projected_within (x :: names ++ free_variables_of_named d)
    (projected_under_let known x d).
Proof.
  intros Hknown. unfold projected_under_let. cbv zeta.
  assert (Hforgot :
    projected_within (x :: names ++ free_variables_of_named d)
      (forget_projected x known))
    by (apply (forget_projected_within names); [ | exact Hknown ];
        apply incl_tl, incl_appl, incl_refl).
  destruct (is_projection d && negb (var_in_simples x (projected_from d)));
    [ | exact Hforgot ].
  intros entry [<- | Hentry];
    [ left; reflexivity | exact (Hforgot entry Hentry) ].
Qed.

Lemma projected_by_from (known : projected) (d : named) (x : variable) :
  projected_by known d = Some x ->
  ∃ entry : variable * named, In entry known ∧ fst entry = x.
Proof.
  induction known as [ | [y d'] known IH]; cbn [projected_by];
    [ discriminate | ].
  destruct (same_projection d d').
  - intros Hfound. injection Hfound as ->.
    exists (x, d'). split; [ left; reflexivity | reflexivity ].
  - intros Hfound. destruct (IH Hfound) as (entry & Hentry & Hfst).
    exists entry. split; [ right; exact Hentry | exact Hfst ].
Qed.

Lemma share_projection_free (names : list variable) (known : projected)
    (d : named) :
  projected_within names known ->
  incl (free_variables_of_named (share_projection known d))
    (free_variables_of_named d ++ names).
Proof.
  intros Hknown. unfold share_projection.
  destruct (projected_by known d) as [x | ] eqn:Hprojected;
    [ | apply incl_appl, incl_refl ].
  destruct (projected_by_from known d x Hprojected) as (entry & Hentry & <-).
  cbn [free_variables_of_named free_variables_of_simple].
  intros y [<- | []]. apply in_or_app. right. exact (Hknown entry Hentry).
Qed.

Fixpoint share_projections_knowing_scoped (e : expr) :
  ∀ (names : list variable) (known : projected),
    projected_within names known ->
    scoped_under names e (share_projections_knowing known e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros names known Hknown;
    cbn [share_projections_knowing share_projections_in_handler];
    try apply scoped_under_refl.
  - apply scoped_under_let; [ apply share_projection_free, Hknown | ].
    apply share_projections_knowing_scoped, projected_under_let_within,
      Hknown.
  - apply scoped_under_let_static. apply share_projections_knowing_scoped.
    intros entry Hentry. apply in_or_app. left. exact (Hknown entry Hentry).
  - apply scoped_under_let_cont; apply share_projections_knowing_scoped;
      [ apply forget_projected_vars_within, Hknown | exact Hknown ].
  - apply scoped_under_let_cont_recursive.
    apply share_projections_knowing_scoped, Hknown.
Qed.

Lemma share_known_projections_scoped (e : expr) :
  scoped e (share_known_projections e).
Proof.
  apply scoped_under_none, share_projections_knowing_scoped. intros entry [].
Qed.

Theorem share_known_projections_keeps_scope :
  share_known_projections_keeps_scope_statement.
Proof. apply keeps_scope_intro, share_known_projections_scoped. Qed.
