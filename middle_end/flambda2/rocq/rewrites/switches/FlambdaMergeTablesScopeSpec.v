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

(* Merging the arms of a [Switch] into a load from a table that is in
   scope keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaMergeTablesSpec.merge_tables.

   The proof that the validator answers the same of renamed terms goes
   rewrite by rewrite, and uses of each that it keeps scope: on a term
   whose code is closed the code of the result is closed, and no
   variable or continuation is free in the result that is not free in
   the term. The rewrite puts, in the place of a [Switch] on a
   variable, an application of the continuation of its first arm under
   [Let]s. Where one table has the arms, the [Let]s bind that variable
   again; they read that variable and the symbol of the table, and
   each argument of the application is an argument of the first arm or
   that variable. Where there is a way for each argument, the [Let]s
   bind names that the rewrite is given; they read the variable of the
   scrutinee, the name that they bind and the symbols of the tables,
   and each argument of the application is that variable, a name that
   the [Let]s bind or an argument of the first arm that is the same in
   every arm. So no variable or continuation is free in the result
   that is not free in the [Switch], and the code of the result is
   closed where that of the term is. The symbols of the tables are not
   counted: [keeps_scope] says nothing of symbols.

   The statement is in a file of its own, and not in
   FlambdaMergeTablesSpec.v, because [keeps_scope] is defined after the
   validator, which applies the rewrite.

   The proof is in FlambdaMergeTablesScopeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec FlambdaMergeTablesSpec.
Import ListNotations.

Definition merge_tables_keeps_scope_statement : Prop :=
  keeps_scope merge_tables.
