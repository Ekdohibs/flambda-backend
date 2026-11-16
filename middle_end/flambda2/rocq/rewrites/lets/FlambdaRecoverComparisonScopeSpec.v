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

(* Making a test of the result of a three-way comparison the comparison
   of its operands keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaRecoverComparisonSpec.recover_known_comparisons.

   The proof that the validator answers the same of renamed terms goes
   rewrite by rewrite, and uses of each that it keeps scope: on a term
   whose code is closed the code of the result is closed, and no
   variable or continuation is free in the result that is not free in
   the term. The pass puts, in the place of a test, a comparison of the
   arguments of a primitive that an enclosing [Let] binds, and those
   arguments are still what they were where it puts them: what is known
   of a variable is forgotten under a binder of it, or of a variable
   that its arguments mention. So it keeps scope.

   The statement is in a file of its own, and not in
   FlambdaRecoverComparisonSpec.v, because [keeps_scope] is defined
   after the validator, which applies the rewrite.

   The proof is in FlambdaRecoverComparisonScopeProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec FlambdaRecoverComparisonSpec.
Import ListNotations.

Definition recover_known_comparisons_keeps_scope_statement : Prop :=
  keeps_scope recover_known_comparisons.
