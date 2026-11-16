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

(* Moving a continuation to the handler that alone uses it: a
   normalisation that is an equivalence.

   This file is to be reviewed: [sink_let_conts] and
   [sink_into_handler], which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify inlines a non-recursive continuation that is applied exactly
   once, by an [Apply_cont] that is not an arm of a [Switch], wherever
   that application is (simplify/simplify_let_cont_expr.ml,
   [is_single_inlinable_use], then
   [Upwards_env.add_linearly_used_inlinable_continuation];
   simplify/simplify_apply_cont_expr.ml,
   [inline_linearly_used_continuation], which puts the handler at the
   [Apply_cont] with a [Let] for each parameter).
   FlambdaNormaliseSpec.inline_tail_conts does so only where the
   application ends the body of the [Let_cont] itself, which is where
   inlining a call leaves it. The front end leaves it elsewhere: in
   the handler of a continuation bound further on, as in

     let_cont k1 (x) = h1 in
     let y = d in
     let_cont k2 (z) = cont k1 (z) in
     body

   where [body] applies [k2] and not [k1]. The body of a [while] loop is
   bound so, and so is the end of every compilation unit.

   [sink_let_conts] moves such a [Let_cont] to the head of the handler
   that uses it,

     let y = d in
     let_cont k2 (z) = (let_cont k1 (x) = h1 in cont k1 (z)) in
     body

   and leaves the inlining to [inline_tail_conts]. So the two are right
   separately, and this one, which changes no handler and removes
   nothing, is an equivalence.

   [sink_into_handler k h body] is [Let_cont_non_recursive k h body]
   with the binding of [k] moved, if it can be moved into a handler
   where [inline_tail_conts] then inlines it ([applied_at_tail]), and
   nothing otherwise. A [Let_cont] that would only go past other
   bindings, or into a handler that applies it twice or from an arm of
   a [Switch], stays where it is: Simplify does not move it, and the
   comparison of the two terms is of the bindings in their order. On
   its way the binding goes
   - past a [Let] whose variable is not free in the handler [h], which
     would otherwise come to mean that variable;
   - past a [Let_static], with no condition;
   - into the handler of the next [Let_cont], if [k] is not free in the
     body of that one, nor in [h] itself (where it would stand for a
     continuation bound outside), and no parameter of that handler is
     free in [h]. There it goes on, as far as it can;
   - failing that, past the next [Let_cont], of another name, if [k] is
     not free in the handler of that one and its name is not free in
     [h].
   It does not go into a recursive [Let_cont], or into the arm of a
   [Switch], which holds no handler.

   The tests of freedom are FlambdaNormaliseSpec.var_free_in_handler_b
   and FlambdaUnusedContsSpec.cont_free_b, which may say that a name is
   free when it is not, never the reverse; so the pass may fail to move
   a binding, and never moves one wrongly.

   Exceptions. A continuation that a trap action names is free where
   the trap action is ([cont_free_b]), so a binding is not moved away
   from one. An application with a trap action is not inlined by
   [inline_tail_conts], so no [Let_cont] is moved to meet one.

   Renaming: [sink_let_conts] compares names by equality only and binds
   none, so it commutes with a renaming. That statement is owed to the
   nominal work and is not made here.

   The proofs are in FlambdaSinkContsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUnusedContsSpec.
Import ListNotations.

(* Whether [inline_tail_conts] inlines the continuation [k] of handler
   [h] bound around [body]: [body] ends by applying it. *)
Definition applied_at_tail (k : continuation) (h : continuation_handler)
    (body : expr) : bool :=
  match h with
  | Handler params metadata handler =>
    match inline_cont_at_tail k params metadata handler body with
    | Some _ => true
    | None => false
    end
  end.

(* No parameter of the list is free in the handler. *)
Definition params_not_free (params : list variable)
    (h : continuation_handler) : bool :=
  forallb (fun param => negb (var_free_in_handler_b param h)) params.

Fixpoint sink_into_handler (k : continuation) (h : continuation_handler)
    (body : expr) : option expr :=
  match body with
  | Let x kind d body =>
    if var_free_in_handler_b x h then None
    else option_map (Let x kind d) (sink_into_handler k h body)
  | Let_static bindings body =>
    option_map (Let_static bindings) (sink_into_handler k h body)
  | Let_cont_non_recursive k' (Handler params' metadata' handler') body =>
    if negb (cont_free_b k body)
       && negb (cont_free_in_handler_b k h)
       && params_not_free params' h
    then
      match sink_into_handler k h handler' with
      | Some sunk =>
        Some (Let_cont_non_recursive k' (Handler params' metadata' sunk) body)
      | None =>
        if applied_at_tail k h handler'
        then
          Some
            (Let_cont_non_recursive k'
               (Handler params' metadata'
                  (Let_cont_non_recursive k h handler'))
               body)
        else None
      end
    else if
      negb (Pos.eqb k k')
      && negb (cont_free_b k handler')
      && negb (cont_free_in_handler_b k' h)
    then
      option_map
        (Let_cont_non_recursive k' (Handler params' metadata' handler'))
        (sink_into_handler k h body)
    else None
  | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Switch _ _ | Invalid =>
    None
  end.

(* Every non-recursive [Let_cont] moved where it can be, those of the
   handler and of the body of a [Let_cont] before that [Let_cont]
   itself. *)
Fixpoint sink_let_conts (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (sink_let_conts body)
  | Let_static bindings body => Let_static bindings (sink_let_conts body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let h := Handler params metadata (sink_let_conts handler) in
    let body := sink_let_conts body in
    match sink_into_handler k h body with
    | Some sunk => sunk
    | None => Let_cont_non_recursive k h body
    end
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (sink_let_conts body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition sink_let_conts_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (sink_let_conts e).
