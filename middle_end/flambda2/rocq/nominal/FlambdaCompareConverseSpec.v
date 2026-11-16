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

(* The comparison of two terms accepts only terms that are
   alpha-equivalent.

   This file is to be reviewed: the statements.

   FlambdaCompareAlphaSpec.v says that the comparison of
   FlambdaCompareSpec.v, with no names paired to begin with, is an
   equivalence relation and accepts every two terms that are
   alpha-equivalent (FlambdaAlphaSpec.alpha_equivalent). Here is the
   converse, for the terms that have two of the conditions of
   well-formedness (FlambdaBindersSpec.v): the code that they bind is
   closed ([code_closed], row 6), and no binder of theirs repeats a name
   ([binders_distinct], row 5). On such terms the comparison and
   alpha-equivalence are one relation
   ([expr_eqb_iff_alpha_equivalent_statement]): the first decides the
   second.

   The two conditions say for which terms the converse is stated:
   those that well-formedness admits. Neither can be left out, as the
   two examples proved in FlambdaCompareAlphaSpec.v show. The
   comparison pairs the names of a binder by position, so it accepts a
   handler whose two parameters are one name against a handler with
   two names, and the two terms are not alpha-equivalent
   ([expr_eqb_repeated_parameter_statement]; the terms of that example
   bind no code, which the statement itself does not say). And each
   rule of [alpha_equivalent] that renames a bound name takes closed
   code as a premise: of terms whose binders are distinct and whose
   code is not closed the comparison accepts two that are not
   alpha-equivalent ([expr_eqb_open_code_statement]).

   With it the rule that compares two [Let]s at a fresh name
   (FlambdaFreshBindersSpec.alpha_let_fresh_statement) can be read both
   ways ([alpha_let_iff_statement]): two [Let]s are alpha-equivalent
   exactly when they have the same kind and the same defining
   expression and their bodies are alpha-equivalent once both bound
   variables are exchanged for a fresh name. That is the equality of
   name abstractions of FlambdaFreshAbstractionSpec.v, said of the
   terms. FlambdaAlphaFreshSpec.v says the same of the other binders
   that are not recursive, and FlambdaAlphaGroupSpec.v of the
   continuations of a recursive [Let_cont].

   The proof follows the compiler's way of comparing two binders
   ([Name_abstraction.pattern_match_pair]): at each binder the bound
   names of both sides are exchanged for the same fresh names, the
   scopes are then compared with no names paired, and the rule that
   compares two binders at fresh names (FlambdaFreshBindersSpec.v)
   concludes.

   Nothing is trusted here. The proofs are in
   FlambdaCompareConverseProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaBindersSpec
  FlambdaNominalSpec FlambdaAlphaSpec FlambdaCompareSpec
  FlambdaFreshQuantifierSpec FlambdaFreshBindersSpec.
Import ListNotations.

Definition expr_eqb_alpha_equivalent_statement : Prop :=
  ∀ e e' : expr,
    code_closed e = true ->
    code_closed e' = true ->
    binders_distinct e = true ->
    binders_distinct e' = true ->
    expr_eqb [] [] e e' = true ->
    alpha_equivalent e e'.

Definition expr_eqb_iff_alpha_equivalent_statement : Prop :=
  ∀ e e' : expr,
    code_closed e = true ->
    code_closed e' = true ->
    binders_distinct e = true ->
    binders_distinct e' = true ->
    (expr_eqb [] [] e e' = true <-> alpha_equivalent e e').

(* A [Let] is a name abstraction of its variable over its body. *)
Definition alpha_let_iff_statement : Prop :=
  ∀ (x x' : variable) (kind kind' : flambda_kind)
    (defining_expr defining_expr' : named) (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    binders_distinct body = true ->
    binders_distinct body' = true ->
    (alpha_equivalent (Let x kind defining_expr body)
       (Let x' kind' defining_expr' body')
     <-> kind = kind'
         ∧ defining_expr = defining_expr'
         ∧ И fresh,
             alpha_equivalent (rename_expr (swap_variable x fresh) body)
               (rename_expr (swap_variable x' fresh) body')).
