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

(* Removing the unused [Let]s whose defining expression has a value, in
   the handlers of recursive continuations too: a normalisation that is
   an equivalence.

   This file is to be reviewed: [remove_unused_defined_lets_in_loops],
   which is extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   FlambdaNormaliseSpec.remove_unused_defined_lets removes a [Let] that
   nothing uses where its defining expression is sure to have a value of
   the kind of the [Let]: a [Simple] or a closure made of constants,
   symbols and variables in scope, each variable with the kind of its
   binder. It leaves the handlers of recursive continuations as they are,
   and FlambdaRecHandlersSpec.rewrite_rec_handlers takes it into such a
   handler with no variable in scope, so a [Let] of a parameter of the
   loop, or of a variable bound outside it, stays. The term after
   Simplify has such [Let]s: where every application of a continuation
   gives one of its parameters the same value, Simplify removes the
   parameter and binds its name at the start of the handler
   (simplify/flow/control_flow_graph.ml,
   [minimize_extra_args_for_one_continuation], makes the
   [lets_to_introduce]; simplify/simplify_let_cont_expr.ml puts them
   around the rebuilt handler), and the name is often not used again.

   [remove_unused_defined_lets_in_loops] is [remove_unused_defined_lets]
   made to go into the handler of a recursive continuation as it goes
   into that of another, with the parameters of the handler, at the
   kinds that the handler gives them, and the variables in scope at the
   [Let_cont] in scope (FlambdaNormaliseSpec.params_in_scope).

   Why this is right although the handler runs many times: the
   variables in scope at the [Let_cont] are bound in the closure of the
   handler, and the parameters are bound each time it is applied, to
   values of the kinds of the handler
   (FlambdaSemanticsSpec.apply_cont_value).

   Renaming: the pass compares names by equality only and binds none; it
   commutes with a renaming as [remove_unused_defined_lets] does. That
   statement is owed to the nominal work and is not made here.

   The proofs are in FlambdaLoopLetsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

Fixpoint remove_unused_defined_lets_in_loops (kinds : kinds_in_scope)
    (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body' :=
      remove_unused_defined_lets_in_loops ((x, kind) :: kinds) body
    in
    if named_defined kinds kind d && negb (var_free_b x body')
    then body'
    else Let x kind d body'
  | Let_static bindings body =>
    Let_static bindings (remove_unused_defined_lets_in_loops kinds body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (remove_unused_defined_lets_in_handler kinds h)
      (remove_unused_defined_lets_in_loops kinds body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, remove_unused_defined_lets_in_handler kinds h))
         handlers)
      (remove_unused_defined_lets_in_loops kinds body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with remove_unused_defined_lets_in_handler (kinds : kinds_in_scope)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (remove_unused_defined_lets_in_loops
         (params_in_scope params (hm_params_kinds metadata) kinds) handler)
  end.

Definition remove_unused_defined_lets_in_loops_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_defined_lets_in_loops [] e).
