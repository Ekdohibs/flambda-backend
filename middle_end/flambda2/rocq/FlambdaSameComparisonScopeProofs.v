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

(* Proof of the statement of FlambdaSameComparisonScopeSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaSameComparisonSpec FlambdaSameComparisonScopeSpec.
Import ListNotations.

Lemma fold_same_comparison_free (d : named) :
  incl (free_variables_of_named (fold_same_comparison d))
    (free_variables_of_named d).
Proof.
  destruct d as [s | p args | | | | | ]; try apply incl_refl.
  destruct p as [p | p | p | p | rp]; try apply incl_refl.
  destruct p as [k lk m | w | op | k o | k o | k b]; try apply incl_refl.
  destruct args as [ | a [ | b [ | c args]]]; try apply incl_refl.
  cbn [fold_same_comparison].
  destruct (FlambdaModel.simple_eq_dec a b); [ intros x [] | ].
  destruct a as [x1 | s1 | c1], b as [x2 | s2 | c2];
    first [ apply incl_refl | intros x [] ].
Qed.

Fixpoint fold_same_comparisons_scoped (e : expr) :
  scoped e (fold_same_comparisons e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [fold_same_comparisons];
    first
      [ apply scoped_let;
        [ apply fold_same_comparison_free
        | apply fold_same_comparisons_scoped ]
      | scoped_under_binders fold_same_comparisons_scoped ].
Qed.

Theorem fold_same_comparisons_keeps_scope :
  fold_same_comparisons_keeps_scope_statement.
Proof. apply keeps_scope_intro, fold_same_comparisons_scoped. Qed.
