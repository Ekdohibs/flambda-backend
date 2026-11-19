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

(* Proofs of the statements of FlambdaUntagAtBinderNamesSpec.v:
   FlambdaUntagAtBinderSpec.bind_untaggings commutes with renaming and
   keeps scope.

   This file needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceBaseProofs FlambdaScopedProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs
  FlambdaUntagParamsNamesProofs FlambdaUntagAtBinderSpec
  FlambdaUntagAtBinderProofs FlambdaUntagAtBinderNamesSpec
  FlambdaEquivarianceTactic FlambdaTermInductionProofs
  FlambdaScopedInductionProofs FlambdaScopeTactic.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma distinct_variables_renamed (r : renaming) (names : list variable) :
  distinct (map (rename_variable r) names) = distinct names.
Proof. equivariance. Qed.
#[local] Hint Rewrite distinct_variables_renamed : equivariance.

Lemma tagged_params_renamed : equivariance_of tagged_params.
Proof. equivariance. Qed.
#[local] Hint Rewrite tagged_params_renamed : equivariance.

Lemma untags_renamed : equivariance_of untags.
Proof. equivariance. Qed.
#[local] Hint Rewrite untags_renamed : equivariance.

Lemma untaggings_renamed : equivariance_of untaggings.
Proof. equivariance. Qed.
#[local] Hint Rewrite untaggings_renamed : equivariance.

Lemma first_untagging_renamed : equivariance_of first_untagging.
Proof.
  intros. unfold first_untagging. rewrite untaggings_renamed.
  destruct (untaggings _ _); reflexivity.
Qed.
#[local] Hint Rewrite first_untagging_renamed : equivariance.

Lemma drop_untagging_renamed : equivariance_of drop_untagging.
Proof. equivariance. Qed.
#[local] Hint Rewrite drop_untagging_renamed : equivariance.

Lemma untagging_at_head_renamed : equivariance_of untagging_at_head.
Proof. equivariance. Qed.
#[local] Hint Rewrite untagging_at_head_renamed : equivariance.

Lemma returns_to_renamed : equivariance_of returns_to.
Proof. equivariance. Qed.
#[local] Hint Rewrite returns_to_renamed : equivariance.

Theorem bind_untaggings_equivariant : bind_untaggings_equivariant_statement.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

(* The [Let]s taken away: their variable may come to be free. *)
Fixpoint drop_untagging_scoped (e : expr) :
  ∀ x u : variable, scoped_under [u] e (drop_untagging x u e)
with drop_untagging_in_handler_scoped (h : continuation_handler) :
  ∀ x u : variable,
    scoped_handler_under [u] h (drop_untagging_in_handler x u h).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros x u; simpl; try apply scoped_under_refl.
    + destruct (Pos.eqb y u) eqn:Hyu.
      * destruct (untags x kind d); [ | apply scoped_under_refl ].
        apply Pos.eqb_eq in Hyu. subst y.
        intros Hclosed. scope_in Hclosed.
        destruct (drop_untagging_scoped body x u Hclosed)
          as (Hclosed' & Hv & Hk).
        split; [ exact Hclosed' | split; [ | exact Hk ] ].
        scope. first [ names | names_cases ].
      * destruct (Pos.eqb y x); [ apply scoped_under_refl | ].
        apply scoped_under_let; [ names | ].
        apply (scoped_under_more [u]);
          [ names | apply drop_untagging_scoped ].
    + apply scoped_under_let_static.
      apply (scoped_under_more [u]); [ names | apply drop_untagging_scoped ].
    + apply scoped_under_let_cont_handler;
        [ apply drop_untagging_in_handler_scoped
        | apply drop_untagging_scoped ].
    + apply scoped_under_let_cont_recursive_handlers.
      * apply (map_fst_keyed (fun _ h => drop_untagging_in_handler x u h)).
      * apply
          (scoped_handlers_under_map_keyed [u]
             (fun _ h => drop_untagging_in_handler x u h)).
        induction handlers as [ | [k h] handlers IH]; [ apply Forall_nil | ].
        apply Forall_cons;
          [ apply drop_untagging_in_handler_scoped | exact IH ].
      * apply drop_untagging_scoped.
  - destruct h as [params metadata handler]. intros x u. simpl.
    destruct (among x params || among u params);
      [ intros Hclosed; split; [ exact Hclosed | split; names ] | ].
    intros Hclosed. scope_in Hclosed.
    destruct (drop_untagging_scoped handler x u Hclosed)
      as (Hclosed' & Hv & Hk).
    split; [ exact Hclosed' | split; [ | exact Hk ] ].
    scope. first [ names | names_cases ].
Qed.

(* An expression that untags a variable reads it. *)
Fixpoint untaggings_read (e : expr) :
  ∀ x u : variable,
    In u (untaggings x e) -> In x (free_variables_of_expr e)
with untaggings_in_handler_read (h : continuation_handler) :
  ∀ x u : variable,
    In u (untaggings_in_handler x h) -> In x (free_variables_of_handler h).
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      intros x u Hin; simpl in Hin; try contradiction; scope.
    + apply in_app_or in Hin as [Hin | Hin].
      * destruct (untags x kind d) eqn:Huntags; [ | contradiction ].
        destruct (untags_spec x kind d Huntags) as [_ ->].
        apply in_or_app. left. left. reflexivity.
      * destruct (Pos.eqb y x) eqn:Hyx; [ contradiction | ].
        pose proof (untaggings_read body x u Hin).
        names.
    + pose proof (untaggings_read body x u Hin). names.
    + apply in_app_or in Hin as [Hin | Hin].
      * pose proof (untaggings_in_handler_read h x u Hin). names.
      * pose proof (untaggings_read body x u Hin). names.
    + apply in_app_or in Hin as [Hin | Hin].
      * apply in_or_app. left.
        induction handlers as [ | [k h] handlers IH];
          [ contradiction | ].
        cbn [flat_map] in Hin |- *.
        apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
          [ left; exact (untaggings_in_handler_read h x u Hin)
          | right; exact (IH Hin) ].
      * pose proof (untaggings_read body x u Hin). names.
  - destruct h as [params metadata handler]. intros x u Hin. simpl in Hin.
    destruct (among x params) eqn:Hparams; [ contradiction | ].
    pose proof (untaggings_read handler x u Hin).
    scope. names.
Qed.

Lemma untagging_at_head_scoped (params : list variable) (x : variable)
    (handler : expr) :
  scoped handler (untagging_at_head params x handler).
Proof.
  unfold untagging_at_head.
  destruct (first_untagging x handler) as [u | ] eqn:Hfirst;
    [ | apply scoped_refl ].
  destruct (negb (among u params) && negb (var_free_b u handler));
    [ | apply scoped_refl ].
  assert (Hx : In x (free_variables_of_expr handler)).
  { apply (untaggings_read handler x u). unfold first_untagging in Hfirst.
    destruct (untaggings x handler) as [ | u' us]; [ discriminate Hfirst | ].
    injection Hfirst as ->. left. reflexivity. }
  intros Hclosed.
  destruct (drop_untagging_scoped handler x u Hclosed)
    as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  scope.
  change (free_variables_of_named (Prim (Unary Untag_immediate) [Var x]))
    with [x].
  first [ names | names_cases ].
Qed.

Lemma untaggings_at_head_scoped (params xs : list variable) :
  ∀ handler handler' : expr,
    scoped handler handler' ->
    scoped handler (fold_right (untagging_at_head params) handler' xs).
Proof.
  induction xs as [ | x xs IH]; intros handler handler' H; simpl;
    [ exact H | ].
  exact
    (scoped_trans _ _ _ (IH handler handler' H)
       (untagging_at_head_scoped params x _)).
Qed.

Fixpoint bind_untaggings_scoped (e : expr) : scoped e (bind_untaggings e)
with bind_untaggings_in_handler_scoped (h : continuation_handler) :
  ∀ returned_to : bool,
    scoped_handler h (bind_untaggings_in_handler returned_to h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply bind_untaggings_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply bind_untaggings_scoped ].
    + apply scoped_let_cont;
        [ apply bind_untaggings_in_handler_scoped
        | apply bind_untaggings_scoped ].
    + apply scoped_let_cont_recursive;
        [ reflexivity | apply scoped_handlers_refl
        | apply bind_untaggings_scoped ].
  - destruct h as [params metadata handler]. intros returned_to. simpl.
    apply scoped_handler_intro.
    destruct
      (returned_to && negb (hm_is_exn_handler metadata) && distinct params);
      [ apply untaggings_at_head_scoped | ]; apply bind_untaggings_scoped.
Qed.

Theorem bind_untaggings_keeps_scope : bind_untaggings_keeps_scope_statement.
Proof. apply keeps_scope_intro, bind_untaggings_scoped. Qed.
