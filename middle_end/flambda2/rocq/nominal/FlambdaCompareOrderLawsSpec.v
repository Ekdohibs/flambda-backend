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

(* The comparison that lets the second term state less is transitive
   and does not depend on the names.

   This file is to be reviewed: the statements.

   [FlambdaCompareOrderSpec.expr_says_at_least] compares two terms up
   to their bound names, the first being the one that may state more
   of its parameters. Two laws of it, which the laws of renaming of
   the validation use ([expr_says_at_least_renamed_alike_statement]
   of FlambdaValidatorEquivarianceSpec.v):
   - with no names paired to begin with it is transitive
     ([expr_says_at_least_transitive_statement]): the order on what
     two binders state is
     (FlambdaSubkindOrderSpec.params_say_at_least_trans_statement).
     With [FlambdaCompareOrderSpec.expr_eqb_says_at_least_statement]
     that gives what the laws of renaming need: of two terms that
     the comparison accepts, the first may give its place to a term
     that [FlambdaCompareSpec.expr_eqb] accepts beside it;
   - it does not depend on the names
     ([expr_says_at_least_equivariant_statement]): its answer on two
     terms is its answer on the two terms renamed, with the names
     paired so far renamed too. A renaming changes no metadata. The
     statement is FlambdaCompareAlphaSpec.expr_eqb_equivariant_statement
     with the other comparison.

   It is reflexive, by [expr_eqb_says_at_least_statement] and
   FlambdaCompareAlphaSpec.expr_eqb_reflexive_statement; that is not
   stated again. It is not symmetric, and nothing here says that it
   is: a term whose handler states that a parameter is a tagged
   immediate is accepted before the same term whose handler states
   nothing, and not after it.

   Nothing is trusted here. The proofs are in
   FlambdaCompareOrderLawsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaCompareSpec FlambdaCompareAlphaSpec
  FlambdaCompareOrderSpec.
Import ListNotations.

Definition expr_says_at_least_transitive_statement : Prop :=
  ∀ e e' e'' : expr,
    expr_says_at_least [] [] e e' = true ->
    expr_says_at_least [] [] e' e'' = true ->
    expr_says_at_least [] [] e e'' = true.

(* Nothing is asked of the terms. *)
Definition expr_says_at_least_equivariant_statement : Prop :=
  ∀ (r : renaming) (vm km : name_map) (e e' : expr),
    expr_says_at_least (rename_name_map (rename_variable r) vm)
      (rename_name_map (rename_continuation r) km) (rename_expr r e)
      (rename_expr r e')
    = expr_says_at_least vm km e e'.
