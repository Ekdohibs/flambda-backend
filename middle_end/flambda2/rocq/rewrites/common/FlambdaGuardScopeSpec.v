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

(* A rewrite made to keep scope by a test.

   This file is to be reviewed: [scope_kept] and [guard_scope], which
   are to be extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   A rewrite keeps scope if, on a term whose code is closed, the code of
   its result is closed and every variable or continuation that is free
   in the result is free in the term
   (FlambdaValidatorEquivarianceSpec.keeps_scope). The validation needs
   it of every rewrite that follows one which makes names. For most of
   its rewrites it is proved (FlambdaKeepsScopeSpec.v,
   FlambdaLoopsScopeSpec.v). [guard_scope f] is for a rewrite of which
   it is not proved: of a term whose code is closed, it gives what [f]
   gives where that passes the test, and the term as it was otherwise.

   The test ([scope_kept]) is what [keeps_scope] says, of the term and
   the result at hand: the code of the result is closed, and its free
   variables and continuations, as FlambdaBindersSpec.v lists them, are
   among those of the term. It is made only of a term whose code is
   closed, which is all that [keeps_scope] speaks of; of a term whose
   code is not closed [guard_scope f] gives what [f] gives.

   So [guard_scope f] is the term or what [f] gives, and it refines the
   term, or is equivalent to it, where what [f] gives does. That it
   keeps scope whatever [f] is, that it is [f] where [f] keeps scope,
   and that it commutes with renaming where [f] does are stated in
   FlambdaValidateScopeSpec.v, since the definitions they are stated
   with come after the validation, which is to run [guard_scope].

   Nothing is trusted here. What the test costs: whether the code of
   each of the two terms is closed, the lists of the free variables and
   of the free continuations of each, and for each name in a list of
   the result a search of the list of the term, which is quadratic in
   the lengths of the lists at worst. The proofs are in
   FlambdaGuardScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaUnusedStaticsSpec.
Import ListNotations.

(* The code of the second term is closed, and its free variables and
   continuations are among those of the first. *)
Definition scope_kept (e e' : expr) : bool :=
  code_closed e'
  && all_among (free_variables_of_expr e') (free_variables_of_expr e)
  && all_among (free_continuations_of_expr e')
       (free_continuations_of_expr e).

(* CR mshinwell: of a term whose code is not closed the test is not
   made, and what the rewrite gives goes through. [keeps_scope] says
   nothing of such a term, and this way the validation answers of it as
   it did before the test was there. The other choice is to leave such
   a term as it was. *)

(* What the rewrite gives, if the code of the term is not closed or the
   test passes; the term otherwise. *)
Definition guard_scope (f : expr -> expr) (e : expr) : expr :=
  let e' := f e in
  if negb (code_closed e) || scope_kept e e' then e' else e.

(* It refines the term where what the rewrite gives does, under all
   environments and for terms run on their own, and is equivalent to it
   where what the rewrite gives is. *)
Definition guard_scope_refines_statement : Prop :=
  ∀ (f : expr -> expr) (e : expr),
    expr_refines e (f e) -> expr_refines e (guard_scope f e).

Definition guard_scope_equivalent_statement : Prop :=
  ∀ (f : expr -> expr) (e : expr),
    expr_equivalent e (f e) -> expr_equivalent e (guard_scope f e).

Definition guard_scope_program_refines_statement : Prop :=
  ∀ (f : expr -> expr) (e : expr),
    program_refines e (f e) -> program_refines e (guard_scope f e).
