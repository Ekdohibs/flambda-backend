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


(* Removing continuations that are not used: a normalisation that is an
   equivalence.

   This file is to be reviewed: [remove_unused_conts] and [cont_free_b],
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify deletes a continuation that nothing applies
   ([Simplify_let_cont_expr]: a handler with no uses is not rebuilt). A
   match whose scrutinee becomes known, by inlining for one, leaves the
   continuations of the arms that are not taken in that state: the
   [Switch] is folded to one arm and the others are applied by nothing.

   [remove_unused_conts] removes a non-recursive [Let_cont] whose
   continuation is not free in its body, once the unused continuations
   of the body have themselves been removed, so that a continuation used
   only by the handler of an unused one goes too. It goes into the
   handlers of the non-recursive continuations that stay; recursive
   [Let_cont]s are kept, their handlers as they are, and the bodies of
   code are not gone into (FlambdaNormaliseSpec.rewrite_code takes the
   pass there).

   [cont_free_b] is the test. It may say that a continuation is free
   when it is not (it does not look at whether a recursive [Let_cont]
   binds the name), never the reverse.

   The result is equivalent to the term
   ([remove_unused_conts_equivalent_statement]), by
   FlambdaNominalSpec.unused_let_cont_statement at each [Let_cont]
   removed, so the pass may be applied on both sides of the comparison.

   The proofs are in FlambdaUnusedContsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2 Require Import FlambdaSemanticsSpec.
Import ListNotations.

(* Whether a continuation may be free in an expression. *)
Fixpoint cont_free_b (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => cont_free_b k body
  | Let_cont_non_recursive bound h body =>
    cont_free_in_handler_b k h
    || (negb (Pos.eqb k bound) && cont_free_b k body)
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => cont_free_in_handler_b k h) handlers
    || cont_free_b k body
  | Apply ap =>
    Pos.eqb k (ap_continuation ap) || Pos.eqb k (ap_exn_continuation ap)
  | Apply_cont ac => among k (apply_cont_continuations ac)
  | Switch _ arms =>
    existsb (fun arm => among k (apply_cont_continuations (snd arm))) arms
  | Invalid => false
  end
with cont_free_in_handler_b (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => cont_free_b k handler
  end.

Fixpoint remove_unused_conts (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (remove_unused_conts body)
  | Let_static bindings body => Let_static bindings (remove_unused_conts body)
  | Let_cont_non_recursive k h body =>
    let body' := remove_unused_conts body in
    if cont_free_b k body'
    then Let_cont_non_recursive k (remove_unused_conts_in_handler h) body'
    else body'
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_unused_conts body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with remove_unused_conts_in_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (remove_unused_conts handler)
  end.

Definition remove_unused_conts_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_conts e).
