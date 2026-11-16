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

(* Proofs of the statements of FlambdaGuardScopeSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaBindersSpec
  FlambdaUnusedStaticsSpec FlambdaGuardScopeSpec.
Import ListNotations.

(* What [guard_scope] gives is the term or what the rewrite gives. *)
Lemma guard_scope_cases (f : expr -> expr) (e : expr) :
  guard_scope f e = f e ∨ guard_scope f e = e.
Proof.
  unfold guard_scope. cbv zeta.
  destruct (negb (code_closed e) || scope_kept e (f e));
    [ left | right ]; reflexivity.
Qed.

Theorem guard_scope_refines : guard_scope_refines_statement.
Proof.
  intros f e Hrefines.
  destruct (guard_scope_cases f e) as [-> | ->]; [ exact Hrefines | ].
  intros vars conts statics hp rd. apply refines_refl.
Qed.

Theorem guard_scope_equivalent : guard_scope_equivalent_statement.
Proof.
  intros f e Hequivalent.
  destruct (guard_scope_cases f e) as [-> | ->]; [ exact Hequivalent | ].
  intros vars conts statics hp rd. apply equivalent_refl.
Qed.

Theorem guard_scope_program_refines : guard_scope_program_refines_statement.
Proof.
  intros f e Hrefines.
  destruct (guard_scope_cases f e) as [-> | ->]; [ exact Hrefines | ].
  apply refines_refl.
Qed.
