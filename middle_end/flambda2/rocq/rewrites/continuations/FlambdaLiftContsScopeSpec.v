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

(* Moving out the continuations that a handler binds on its way to its
   [Switch] keeps scope.

   This file is to be reviewed: the statement, against
   FlambdaValidatorEquivarianceSpec.keeps_scope and
   FlambdaLiftContsSpec.lift_handler_conts.

   A rewrite keeps scope if, on a term whose code is closed, the code
   of the result is closed, and no variable or continuation is free in
   the result that is not free in the term. The proof that the
   validation answers the same of renamed terms uses that of each
   rewrite that it applies.

   The pass gives a continuation that it moves the parameters, of the
   handler that it is moved out of, that the handler of the
   continuation may read, under their own names; so the moved handler
   has none of them free outside that handler, where they are not
   bound. The applications of the continuation mention those
   parameters where they are bound. The continuation moved is bound
   around more than it was, and the pass has found it free in none of
   that.

   The statement is in a file of its own, and not in
   FlambdaLiftContsSpec.v, because [keeps_scope] is defined after the
   validation.

   The proof is in FlambdaLiftContsScopeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel
  FlambdaValidatorEquivarianceSpec FlambdaLiftContsSpec.
Import ListNotations.

Definition lift_handler_conts_keeps_scope_statement : Prop :=
  keeps_scope lift_handler_conts.
