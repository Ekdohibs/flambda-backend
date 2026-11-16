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

(* Proofs of the statements of FlambdaSpineWalkNamesSpec.v. Nothing
   here needs review.

   Each is one induction on the term (FlambdaTermInductionProofs.v).
   For renaming: the hypothesis on the step, read from right to left,
   brings the renaming inside the step, where it meets what the
   induction gives. For scope: the lemma of FlambdaScopedProofs.v for
   the form of the node, then the hypothesis on the step. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaTermInductionProofs FlambdaSpineWalkSpec FlambdaSpineWalkNamesSpec.
Import ListNotations.

Theorem walk_spine_equivariant : walk_spine_equivariant_statement.
Proof.
  intros step Hstep r e. revert r.
  induction e using expr_deep_ind; intros r;
    cbn [walk_spine rename_expr rename_handler]; rewrite <- Hstep;
    cbn [rename_expr rename_handler];
    rewrite ?IHe, ?IHe1, ?IHe2; reflexivity.
Qed.

Lemma walk_spine_scoped (step : expr -> expr) :
  (∀ e : expr, scoped e (step e)) -> ∀ e : expr, scoped e (walk_spine step e).
Proof.
  intros Hstep e.
  induction e using expr_deep_ind; cbn [walk_spine];
    refine (scoped_trans _ _ _ _ (Hstep _)); try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | assumption ].
  - apply scoped_let_static; [ apply scoped_bindings_refl | assumption ].
  - apply scoped_let_cont; [ apply scoped_handler_intro | ]; assumption.
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl | assumption ].
Qed.

Theorem walk_spine_keeps_scope : walk_spine_keeps_scope_statement.
Proof.
  intros step Hstep.
  apply keeps_scope_intro, walk_spine_scoped, keeps_scope_scoped, Hstep.
Qed.
