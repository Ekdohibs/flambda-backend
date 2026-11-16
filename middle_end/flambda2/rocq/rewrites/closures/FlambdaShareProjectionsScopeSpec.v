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

(* The sharing of the projections of value slots keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaShareProjectionsSpec.share_known_projections.

   The proof that the validator answers the same of renamed terms goes
   rewrite by rewrite, and uses of each that it keeps scope: on a term
   whose code is closed the code of the result is closed, and no
   variable or continuation is free in the result that is not free in
   the term. The pass replaces a projection by a variable that an
   enclosing [Let] binds, and that is still what the [Let] bound where
   it is put, so it keeps scope.

   The statement is in a file of its own, and not in
   FlambdaShareProjectionsSpec.v, because [keeps_scope] is defined
   after the validator, which applies the rewrite.

   The proof is in FlambdaShareProjectionsScopeProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec FlambdaShareProjectionsSpec.
Import ListNotations.

Definition share_known_projections_keeps_scope_statement : Prop :=
  keeps_scope share_known_projections.
