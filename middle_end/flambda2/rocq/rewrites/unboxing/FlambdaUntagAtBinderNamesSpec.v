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


(* FlambdaUntagAtBinderSpec.bind_untaggings commutes with renaming and
   keeps scope.

   This file is to be reviewed: the two statements. The proofs are in
   FlambdaUntagAtBinderNamesProofs.v, which needs no review.

   The pass compares names by equality only and makes none. Which
   [Let] it takes is decided by the structure of the term (the first
   in one walk), and its tests are whether a variable is a parameter
   of the handler and whether it is free in the handler, which a
   renaming keeps. So a renamed term gets the renamed result.

   The [Let] that it puts at the head of a handler reads a parameter
   of that handler and takes the name of the variable of an untagging
   that is in the handler's body. Another binder of that name may keep
   that untagging in its place. Every [Let] that is taken away is in
   the scope of the one at the head, with no binder of either variable
   between the two. So no variable is free in the result that is not
   free in the term, and no continuation is touched.

   "Keeps scope" is FlambdaValidatorEquivarianceSpec.keeps_scope: on a
   term whose code is closed, the code of the result is closed, and a
   variable or a continuation that is free in the result is free in
   the term. Symbols and code ids are not counted. The statement has
   that premise on the code; the equivalence of
   FlambdaUntagAtBinderSpec.v has none. Neither statement depends on
   what the machine tests: both are of the function alone. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaNominalSpec
  FlambdaValidatorEquivarianceSpec FlambdaUntagAtBinderSpec.
Import ListNotations.

Definition bind_untaggings_equivariant_statement : Prop :=
  rewrite_equivariant bind_untaggings.

Definition bind_untaggings_keeps_scope_statement : Prop :=
  keeps_scope bind_untaggings.
