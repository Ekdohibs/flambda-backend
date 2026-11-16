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


(* The reading of a known immutable array keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaArrayLoadSpec.load_known_arrays.

   The proof that the validator answers the same of renamed terms goes
   rewrite by rewrite, and uses of each that it keeps scope: on a term
   whose code is closed the code of the result is closed, and no
   variable or continuation is free in the result that is not free in
   the term. The pass replaces a primitive by a constant, or by what an
   element of an array holds: a symbol, a constant, or a variable that
   is an argument of the [Let] that made the array, or of the static
   constant of its symbol, and that no binder has bound again since.
   So it keeps scope.

   The statement is in a file of its own, and not in
   FlambdaArrayLoadSpec.v, because [keeps_scope] is defined after the
   validator. The rewrite is not a step of the rounds of the
   validator (FlambdaValidateReadingSpec.v says why), so the proof
   about those rounds does not use this statement. A law under
   renaming of the function of that file, which applies the rewrite,
   would.

   The proof is in FlambdaArrayLoadScopeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec FlambdaArrayLoadSpec.
Import ListNotations.

Definition load_known_arrays_keeps_scope_statement : Prop :=
  keeps_scope load_known_arrays.
