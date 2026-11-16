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

(* Rewriting the handlers of recursive continuations: what lets a
   normalisation into a loop.

   This file is to be reviewed: [rewrite_rec_handlers], which is
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the two statements.

   The rewrites of FlambdaNormaliseSpec.v and those written after it
   leave the handlers of a recursive [Let_cont] as they are. Simplify
   does not: it simplifies the handler of a loop as it does any other
   expression (simplify/simplify_let_cont_expr.ml: [simplify_handlers],
   the case [Recursive], goes by [simplify_recursive_handlers] through
   every handler that is reachable, and
   [simplify_single_recursive_handler] gives each to the expression
   simplifier, under the environment of the [Let_cont] with the
   parameters added at unknown types). So a unit with a [while] or a
   [for] loop, which the front end makes a recursive continuation, was
   never validated: the term after had its loop simplified and the term
   before had not.

   [rewrite_rec_handlers f] applies the rewrite [f] to the handler of
   every recursive continuation of a term, wherever the [Let_cont] is:
   in a body, in the handler of a continuation, recursive or not, and so
   in a loop inside a loop, the inner one first. It is to recursive
   handlers what FlambdaNormaliseSpec.rewrite_code is to the bodies of
   code, which it does not go into; [rewrite_code] takes it there.

   What a rewrite needs to be applied in such a handler is no more than
   what [rewrite_code] needs in code: that it refines, or is an
   equivalence, under all environments, as the handler runs under
   environments that are not those of its definition (its parameters are
   bound afresh each time round); and that it keeps whether the handler
   binds something static, as a recursive [Let_cont] one of whose
   handlers does is undefined (FlambdaSemanticsSpec.binds_statics). The
   first is the premise of the statements. The second is checked: a
   handler is left alone if the rewrite would change it
   (FlambdaNormaliseSpec.rewrite_code_body, which is used as it is).

   A rewrite that relies on what is known where the loop is defined (an
   alias bound outside the loop, a block made before it) is not covered:
   [f] is given the handler alone.

   The functions here act on variables and continuations only through
   [f]: [rewrite_rec_handlers f] commutes with a renaming where [f]
   does. That statement is owed to the nominal work and is not made
   here.

   The proofs are in FlambdaRecHandlersProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* [f] applied to the handler of every recursive continuation. In a
   handler that has a recursive [Let_cont] of its own, the handlers of
   that one are rewritten first. *)
Fixpoint rewrite_rec_handlers (f : expr -> expr) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (rewrite_rec_handlers f body)
  | Let_static bindings body =>
    Let_static bindings (rewrite_rec_handlers f body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (rewrite_rec_handlers_in_handler f h)
      (rewrite_rec_handlers f body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_rec_handler f h)) handlers)
      (rewrite_rec_handlers f body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_rec_handlers_in_handler (f : expr -> expr)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (rewrite_rec_handlers f handler)
  end
with rewrite_rec_handler (f : expr -> expr) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (rewrite_code_body f (rewrite_rec_handlers f handler))
  end.

(* A rewrite that refines, or is an equivalence, under all environments
   is one when applied to the handlers of recursive continuations. *)
Definition rewrite_rec_handlers_refines_statement : Prop :=
  ∀ f : expr -> expr,
    (∀ e : expr, expr_refines e (f e)) ->
    ∀ e : expr, expr_refines e (rewrite_rec_handlers f e).

Definition rewrite_rec_handlers_equivalent_statement : Prop :=
  ∀ f : expr -> expr,
    (∀ e : expr, expr_equivalent e (f e)) ->
    ∀ e : expr, expr_equivalent e (rewrite_rec_handlers f e).
