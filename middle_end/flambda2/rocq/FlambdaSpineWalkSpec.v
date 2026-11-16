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

(* A rewrite applied at every node of the spine of a term, the inside
   first, and what it has of the rewrite.

   This file is to be reviewed: [walk_spine] and the two statements.

   Many rewrites of the validation go down a term in the same way:
   into the body of a [Let] and of a [Let_static], into the handler
   and the body of a non-recursive [Let_cont], into the body of a
   recursive one, and not into the handlers of recursive continuations
   nor into the bodies of code (FlambdaRecHandlersSpec.v and
   FlambdaNormaliseSpec.rewrite_code take a rewrite there). Each is
   written as a [Fixpoint] of its own, and each of its laws is proved
   by an induction of its own that says the same thing: a context
   around what the induction gives, then what the rewrite does at the
   node.

   [walk_spine step] is that way of going down, with the rewrite of
   one node, [step], as an argument: [step] is applied to a node once
   what is inside the node has been rewritten. The statements say that
   the walk has a property as soon as the step has it: here, that its
   result refines the term, or is equivalent to it, under all
   environments; in FlambdaSpineWalkNamesSpec.v, that it commutes with
   renaming and that it keeps scope. So a rewrite written as a walk
   owes its laws for its step only, and a step is not recursive.

   The proofs are in FlambdaSpineWalkProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec.
Import ListNotations.

Fixpoint walk_spine (step : expr -> expr) (e : expr) {struct e} : expr :=
  step
    match e with
    | Let x kind d body => Let x kind d (walk_spine step body)
    | Let_static bindings body => Let_static bindings (walk_spine step body)
    | Let_cont_non_recursive k (Handler params metadata handler) body =>
      Let_cont_non_recursive k
        (Handler params metadata (walk_spine step handler))
        (walk_spine step body)
    | Let_cont_recursive handlers body =>
      Let_cont_recursive handlers (walk_spine step body)
    | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
    end.

Definition walk_spine_refines_statement : Prop :=
  ∀ step : expr -> expr,
    (∀ e : expr, expr_refines e (step e)) ->
    ∀ e : expr, expr_refines e (walk_spine step e).

Definition walk_spine_equivalent_statement : Prop :=
  ∀ step : expr -> expr,
    (∀ e : expr, expr_equivalent e (step e)) ->
    ∀ e : expr, expr_equivalent e (walk_spine step e).
