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

(* Following aliases keeps scope.

   This file is to be reviewed: the statement.

   A rewrite keeps scope (FlambdaValidatorEquivarianceSpec.keeps_scope)
   if, on a term whose code is closed, the code of its result is closed
   and every variable or continuation that is free in the result is
   free in the term. FlambdaKeepsScopeSpec.v says it of the
   rewrites of the validation, one statement to a rewrite, and among
   them of the [follow_aliases] of FlambdaAliasesSpec.v, which the
   validator runs today. This is the same of the [follow_aliases] of
   FlambdaEnvAliasesFollowSpec.v, which is to take its place, from
   nothing known ([Known.empty]): the statement that the validator
   needs of it when it moves.

   Why it holds: what is known where a subterm is speaks only of
   variables that are free in the term or bound around the subterm
   (the names of the aliases, and the [Simple]s of the value slots of
   the closures recorded), and a variable that is bound again is
   forgotten first ([Known.bind]). The code that a [Let_static] binds
   is left as it is, and no continuation is touched.

   The proof is in FlambdaEnvAliasesFollowScopeProofs.v, which needs
   no review; it follows the one of
   FlambdaKeepsScopeProofs.v. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaEnvAliasesFollowSpec.
Import ListNotations.

Definition follow_aliases_knowing_nothing_keeps_scope_statement : Prop :=
  keeps_scope (FlambdaEnvAliasesFollowSpec.follow_aliases Known.empty).
