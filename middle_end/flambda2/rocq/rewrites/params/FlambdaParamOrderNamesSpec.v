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

(* FlambdaParamOrderSpec.order_params commutes with renaming and keeps
   scope.

   This file is to be reviewed: the two statements. The proofs are in
   FlambdaParamOrderNamesProofs.v, which needs no review.

   The pass compares names by equality only and makes none. The order
   that it gives the parameters of a handler is read off the list of
   the free variables of the handler, which a renaming renames in
   place, so a renamed term gets the same order. The names that the
   result binds and mentions are those of the term: parameters and
   arguments change places only.

   "Keeps scope" is FlambdaValidatorEquivarianceSpec.keeps_scope: on a
   term whose code is closed, the code of the result is closed, and a
   variable or a continuation that is free in the result is free in
   the term. Symbols and code ids are not counted. The statement has
   that premise on the code; the equivalence of
   FlambdaParamOrderSpec.v has none. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaNominalSpec
  FlambdaValidatorEquivarianceSpec FlambdaParamOrderSpec.
Import ListNotations.

Definition order_params_equivariant_statement : Prop :=
  rewrite_equivariant order_params.

Definition order_params_keeps_scope_statement : Prop :=
  keeps_scope order_params.
