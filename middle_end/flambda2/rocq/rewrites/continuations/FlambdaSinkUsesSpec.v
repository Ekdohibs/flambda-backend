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

(* Moving a continuation down past the bindings under it, and into the
   handler under it that alone names it: a normalisation that is an
   equivalence.

   This file is to be reviewed: [sink_conts_to_uses] and [sink_to_use],
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   The comparison of two terms is of their bindings in their order, and
   the two terms of a unit may bind a continuation at different places
   among the [Let]s around it.

   The front end binds the continuation that a call returns to before
   the arguments of the call are computed:

     let_cont k (r) = h in
     let x = d in
     apply f (x) -> k

   Simplify leaves such a term in that order. But where it changes the
   parameter of [k] (the boolean result of a call that an [if] tests is
   untagged; a result that [h] does not use is not passed), the call
   can no longer return to [k]. Simplify makes the call return to a new
   continuation, bound at the call, that applies [k]
   (simplify/simplify_apply_expr.ml rebuilds the call with
   [Expr_builder.rewrite_fixed_arity_apply]; the case [New_wrapper] of
   [Expr_builder.rewrite_fixed_arity_continuation] binds the new
   continuation around the [Apply]):

     let_cont k (u) = h' in
     let x = d in
     let_cont k' (r) = (let u = untag r in cont k (u)) in
     apply f (x) -> k'

   Once [k], applied once, is inlined in [k']
   (FlambdaSinkContsSpec.sink_let_conts, then the inlining of a
   continuation applied at the end of the body of its [Let_cont]), the
   handler is the one the term had before, bound after the [Let] and
   not before it. In that handler, the [Let] of the untagging then
   stands before the bindings of the continuations of the test, where
   the term before has it after them:

     let u = untag r in             let_cont k1 () = .. in
     let_cont k1 () = .. in         let u = untag r in
     switch u | 0 -> k1 () | ..     switch u | 0 -> k1 () | ..

   The same happens at a loop. The continuation that follows an
   expression is bound before what the expression computes first. A
   loop in the expression leaves by a continuation of its own, bound
   at the loop, whose handler applies that one. In the term before,
   the validation makes the loop leave by the outer continuation
   (FlambdaForwardSpec.forward_applications), which stays where it
   was bound. Simplify, where nothing else applies the outer
   continuation, inlines it in the one bound at the loop. The handler
   is then bound before a [Let] in one term and after it in the
   other, and only the handler of the recursive continuation names
   its continuation:

     let_cont k () = h in           let x = d in
     let x = d in                   let_cont k' () = h in
     let_cont rec loop (i) = ..     let_cont rec loop (i) = ..
     in cont loop (0)               in cont loop (0)

   And at a test of which one way does something first. Where a
   function is inlined, the continuation that the call returned to is
   bound before the arguments are computed, and the continuation
   where the two ways of a test in the function meet is bound in its
   body, after them; one applies the other. Of the two the
   validation keeps the second in the term before, and Simplify,
   which goes through a continuation that only applies another, the
   first. An arm of the test and the handler of the continuation of
   the other way both apply it:

     let x = d in                   let_cont k () = h in
     let_cont k () = h in           let x = d in
     let y = p (x) in               let y = p (x) in
     let_cont k1 () = .. k () in    let_cont k1 () = .. k () in
     switch y | 0 -> k () | 1 -> k1 ()
                                    switch y | 0 -> k () | 1 -> k1 ()

   And where the result of a call is the argument of a second call,
   whose result is tested ([if small (next x) then .. else ..],
   neither function inlined). The first call returns to a
   continuation that makes the second, and the front end binds the
   continuation that tests before that one. Simplify untags the
   boolean that is tested, so the second call returns to a new
   continuation bound at that call, as above, which is in the handler
   of the continuation that the first call returns to; and the
   continuation that tests, once inlined in the new one, stands in
   that handler too:

     let_cont k (b) = test in       let x = d in
     let x = d in                   let_cont k1 (y) =
     let_cont k1 (y) =                let_cont k' (b) = test in
       apply g (y) -> k               apply g (y) -> k'
     in                             in
     apply f (x) -> k1              apply f (x) -> k1

   [sink_conts_to_uses] gives the two terms one order: a non-recursive
   [Let_cont] goes down past the bindings under it, and into the
   handler under it that alone names its continuation, as far as it
   can. [sink_to_use k h body] is [Let_cont_non_recursive k h body]
   with the binding of [k] so moved. The binding goes
   - past a [Let] whose variable is not free in the handler [h], which
     would otherwise come to mean that variable;
   - past a [Let_static], with no condition;
   - past a non-recursive [Let_cont] of another name, if [k] is not
     free in the handler of that one and its name is not free in [h];
   - failing that, into the handler of a non-recursive [Let_cont], if
     [k] is not free in the body of that [Let_cont], nor in [h] itself
     (where it would stand for a continuation bound outside), and no
     parameter of that handler is free in [h]. There it goes on, as
     far as it can.
   It stops above the first thing that it can neither pass nor enter:
   a [Let] or a [Let_cont] that does not meet these conditions, a
   [Let_cont] of recursive continuations, or the [Apply], the
   [Apply_cont], the [Switch] or the [Invalid] that the bindings end
   with. So the binding of the continuation that a call returns to
   stands directly above the call, where only bindings that it can
   pass are between the two; that of a continuation that an arm of a
   test names, directly above the test or above the continuations of
   the other arms that apply it; that of the way out of a loop,
   directly above the loop; that of a continuation that only the
   handler of another names, in that handler, placed there in the
   same way. Each of the four moves is one of
   FlambdaSinkContsSpec.sink_into_handler, which makes the last only
   where the continuation is then inlined; this pass does not move a
   binding past a [Let_cont] of recursive continuations, nor into the
   handler of one.

   The tests of freedom are FlambdaNormaliseSpec.var_free_in_handler_b
   and FlambdaUnusedContsSpec.cont_free_b, which may say that a name is
   free when it is not, never the reverse; so the pass may stop a
   binding higher than it need, and never moves one wrongly.

   A move takes the handler with its binding, as it is, and the pass
   removes nothing: it is an equivalence. Going down a term, the pass
   also places the bindings that are inside the handlers of
   non-recursive continuations; the parameters of a handler, their
   kinds and the rest of its metadata are never changed. It does not
   go into the handlers of recursive continuations nor into the
   bodies of code (FlambdaRecHandlersSpec.rewrite_rec_handlers and
   FlambdaNormaliseSpec.rewrite_code take it there).

   It is for the two terms that are compared, once, when the rounds of
   their normalisation are over (FlambdaValidateSpec.placed). In a
   round it would change where a continuation is bound for the
   rewrites of the next round that look at that
   (FlambdaParamAliasesSpec.bind_aliased_params, the inlining of a call
   whose continuation was moved), which follow Simplify on the term as
   it was.

   Renaming: [sink_conts_to_uses] compares names by equality only and
   binds none, so it commutes with a renaming
   (FlambdaValidatorEquivarianceProofs.v).

   The proofs are in FlambdaSinkUsesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUnusedContsSpec FlambdaSinkContsSpec.
Import ListNotations.

Fixpoint sink_to_use (k : continuation) (h : continuation_handler)
    (body : expr) : expr :=
  match body with
  | Let x kind d rest =>
    if var_free_in_handler_b x h then Let_cont_non_recursive k h body
    else Let x kind d (sink_to_use k h rest)
  | Let_static bindings rest => Let_static bindings (sink_to_use k h rest)
  | Let_cont_non_recursive k' (Handler params' metadata' handler') rest =>
    if negb (Pos.eqb k k')
       && negb (cont_free_b k handler')
       && negb (cont_free_in_handler_b k' h)
    then
      Let_cont_non_recursive k' (Handler params' metadata' handler')
        (sink_to_use k h rest)
    else if
      negb (cont_free_b k rest)
      && negb (cont_free_in_handler_b k h)
      && params_not_free params' h
    then
      Let_cont_non_recursive k'
        (Handler params' metadata' (sink_to_use k h handler')) rest
    else Let_cont_non_recursive k h body
  | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Switch _ _ | Invalid =>
    Let_cont_non_recursive k h body
  end.

(* Every non-recursive [Let_cont] moved down, those of the handler and
   of the body of a [Let_cont] before that [Let_cont] itself. *)
Fixpoint sink_conts_to_uses (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (sink_conts_to_uses body)
  | Let_static bindings body =>
    Let_static bindings (sink_conts_to_uses body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    sink_to_use k (Handler params metadata (sink_conts_to_uses handler))
      (sink_conts_to_uses body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (sink_conts_to_uses body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition sink_conts_to_uses_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (sink_conts_to_uses e).
