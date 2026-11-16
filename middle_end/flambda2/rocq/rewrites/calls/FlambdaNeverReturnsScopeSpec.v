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

(* The rewrite of calls of code that never returns keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaNeverReturnsSpec.cut_returns.

   The proof that the validation answers the same of renamed terms
   goes rewrite by rewrite, and uses of each that it keeps scope: on a
   term whose code is closed the code of the result is closed, and no
   variable or continuation is free in the result that is not free in
   the term. The rewrite puts a call under a binding of the call's own
   return continuation, whose handler is [Invalid] and binds its
   parameters: nothing is free in the result that was not free in the
   call. That holds whatever code is known.

   The statement is in a file of its own, and not in
   FlambdaNeverReturnsSpec.v, because [keeps_scope] is defined after
   the validator, which is to apply the rewrite.

   The proof is in FlambdaNeverReturnsScopeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValidatorEquivarianceSpec
  FlambdaNeverReturnsSpec.
Import ListNotations.

Definition cut_returns_keeps_scope_statement : Prop :=
  ∀ known : list (code_id * code), keeps_scope (cut_returns known).
