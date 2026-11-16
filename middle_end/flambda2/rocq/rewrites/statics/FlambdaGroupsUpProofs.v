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


(* Proofs of the statements of FlambdaGroupsUpSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaStaticSpec FlambdaStaticProofs FlambdaNominalSpec
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaFloatCodeSpec
  FlambdaFloatCodeProofs FlambdaGroupsUpSpec FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* The equivalence                                                          *)
(* ======================================================================== *)

Lemma group_over_equivalent (wrap : expr -> expr) :
  (∀ (bindings : list static_binding) (body : expr),
    expr_equivalent (wrap (Let_static bindings body))
      (Let_static bindings (wrap body))) ->
  ∀ inner : expr, expr_equivalent (wrap inner) (group_over wrap inner).
Proof.
  intros Hover inner.
  destruct inner
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    unfold group_over; try
      (intros vars conts statics hp rd; apply equivalent_refl).
  apply Hover.
Qed.

Fixpoint groups_up_equivalent_aux (e : expr) :
  expr_equivalent e (groups_up e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [groups_up];
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _)
      (groups_up_equivalent_aux body)).
  - pose proof (plug_equivalent (Let_static_body bindings Hole) _ _
      (not_many _) (groups_up_equivalent_aux body)) as Hbody.
    simpl in Hbody.
    destruct (groups_up body)
      as [ x kind' d body' | bindings' body' | k h body' | handlers body' | ap
         | ac | scrutinee arms | ];
      try (apply (expr_equivalent_trans _ _ _ Hbody), code_first_equivalent).
    apply (expr_equivalent_trans _ _ _ Hbody).
    apply (expr_equivalent_trans _
      (Let_static (bindings ++ bindings') body')).
    + intros vars conts statics hp rd.
      apply equivalent_sym, let_static_split.
    + apply code_first_equivalent.
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (groups_up handler)) body)).
    { exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole body)
        _ _ (not_many _) (groups_up_equivalent_aux handler)). }
    apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (groups_up handler)) (groups_up body))).
    { exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (groups_up handler)) Hole)
        _ _ (not_many _) (groups_up_equivalent_aux body)). }
    apply (group_over_equivalent
      (Let_cont_non_recursive k
         (Handler params metadata (groups_up handler)))).
    intros bindings body'. apply float_over_let_cont.
  - apply (expr_equivalent_trans _
      (Let_cont_recursive handlers (groups_up body))).
    { exact (plug_equivalent (Let_cont_recursive_body handlers Hole)
        _ _ (not_many _) (groups_up_equivalent_aux body)). }
    apply (group_over_equivalent (Let_cont_recursive handlers)).
    intros bindings body'. apply float_over_let_cont_recursive.
Qed.

Theorem groups_up_equivalent : groups_up_equivalent_statement.
Proof. exact groups_up_equivalent_aux. Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* Proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v), its database applied by the one pass,
   the faster step: the proof passes with it. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Theorem groups_up_equivariant : groups_up_equivariant_statement.
Proof. equivariance. Qed.
