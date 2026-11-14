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

(* A [Let] of a variable to the variable that the [Let] around it has
   just bound, where that variable is not free in what follows: the two
   are made one [Let]. A normalisation that is an equivalence.

   This file is to be reviewed: [merge_alias_lets], which is extracted
   and run by the validation of middle_end/flambda2/tests/semantics/ on
   the term after Simplify (FlambdaValidateSpec.normalise_expr_after
   and [normalise_after_round]), and the statement.

   The validation inlines the calls of known code in the term after as
   in the term before (FlambdaNormaliseSpec.inline_known_calls). The
   copy of the body of the code returns by applying the continuation
   that the call returned to. Once that continuation is inlined where
   it is applied, the copy ends with a [Let] of the parameter of the
   continuation to the variable that the body returns:

     let x = d in      the last thing that the body computes
     let y = x in      the result of the call, by the name that the
     rest              continuation gives it

   FlambdaAliasesSpec.follow_aliases then puts x for y in the rest,
   and the [Let] of y, left unused, is removed. It cannot do so where
   the rest binds x again, and it does where the same code is inlined
   at a second call in the rest: a body that is inlined twice is copied
   with the names that it binds, the round of the term after renames
   nothing apart (the round of the term before does,
   FlambdaRenameApartSpec.rename_apart_avoiding), and what is known of
   y is forgotten under the second binding of x. The [Let] of y then
   stays in the term after, and the term before has none.

   [merge_alias_lets] makes the two [Let]s one,

     let y = d in rest

   where the defining expression of the inner [Let] is the variable of
   the outer one, the two [Let]s are at the same kind, and the variable
   of the outer one is not free in the body of the inner one
   (FlambdaNormaliseSpec.var_free_b; the body may bind a variable of
   that name again and use it). It is applied before the aliases are
   followed, when x is not yet free in the rest. The body of the inner
   [Let] is rewritten first, so of three such [Let]s one is left. The
   handlers of non-recursive
   continuations are gone into; the handlers of recursive ones and the
   bodies of code are left as they are
   (FlambdaRecHandlersSpec.rewrite_rec_handlers and
   FlambdaNormaliseSpec.rewrite_code take a rewrite there).

   The two kinds must be the same: a [Let] has no step unless the
   value of its defining expression is of its kind
   (FlambdaSemanticsSpec.step), so where the kinds differ the two
   [Let]s are undefined whatever d is, and the one [Let] may not be.

   The result is equivalent to the term
   ([merge_alias_lets_equivalent_statement]), under all environments.

   Renaming: the pass compares variables for equality and makes no
   name, and it commutes with the renamings of FlambdaNominalSpec.v
   ([merge_alias_lets_renamed_statement]).

   The proof is in FlambdaAliasLetsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaNominalSpec.
Import ListNotations.

Fixpoint merge_alias_lets (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    match merge_alias_lets body with
    | Let y kind' (Simple (Var x')) body' =>
      if Pos.eqb x' x && kind_eqb kind' kind && negb (var_free_b x body')
      then Let y kind d body'
      else Let x kind d (Let y kind' (Simple (Var x')) body')
    | body' => Let x kind d body'
    end
  | Let_static bindings body => Let_static bindings (merge_alias_lets body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (merge_alias_lets handler))
      (merge_alias_lets body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (merge_alias_lets body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition merge_alias_lets_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (merge_alias_lets e).

(* The pass commutes with renaming. *)
Definition merge_alias_lets_renamed_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    merge_alias_lets (rename_expr r e) = rename_expr r (merge_alias_lets e).
