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

(* A continuation whose handler is that of a continuation bound around
   it applies that one: a normalisation that is an equivalence.

   This file is to be reviewed: [share_handlers] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify keeps one handler where two non-recursive continuations
   have the same one. Where it rebuilds the handler of a non-recursive
   continuation that it does not inline, that is not an exception
   handler, and whose handler is neither one [Apply_cont] nor [Invalid]
   (simplify/simplify_let_cont_expr.ml,
   [rebuild_single_non_recursive_handler]), it looks among the
   continuations in scope whose handlers it has kept
   ([Upwards_env.find_unique_continuation_handler]) for one with the
   same handler. If there is one, the continuation is recorded as a
   shortcut to it ([Shortcut_to]), and every application of it is
   rebuilt as an application of the other
   (simplify/continuation_shortcut.ml, [apply]); if there is none, its
   own handler is kept for the continuations bound under it
   ([Upwards_env.add_non_inlinable_continuation]). The handlers that it
   compares (simplify/rebuilt_expr.ml, [Matching_for_unique_handler])
   are [Let]s of a simple or of a primitive, fewer than
   [max_hash_depth] of them, that end in an [Apply_cont] with no trap
   action; two are the same if they are so up to the variables that
   their [Let]s bind and to the order of their parameters. The two arms
   of a match that do the same thing are such a pair: the front end
   gives each its own continuation, and the [Switch] of the term after
   has both arms at one, or is gone.

   [share_with k h] goes down the body of the [Let_cont] that binds [k]
   to the handler [h]: under its [Let]s and [Let_static]s, and into the
   bodies of its non-recursive [Let_cont]s. The handler of such a
   [Let_cont] that is the same as [h] up to bound names
   (FlambdaCompareSpec.handler_eqb, with the parameters in the same
   order) becomes the application of [k] to its own parameters
   ([same_handler]). For that, the names free in the handler must be,
   where it is, what they are where [k] is bound: none is a variable
   that a [Let] has bound on the way ([xs]), and none is a continuation
   bound on the way or [k] itself ([ks]). The parameters must be
   distinct, for the application to give each the value that it was
   given. Where [k] is bound again, the name is another continuation,
   and what is under that [Let_cont] is left. The pass does not go into
   a handler to look for one that is the same as [h]: Simplify compares
   a handler where the term has it, and the rounds of the validation
   have by then put a continuation that is applied once where it is
   applied, so a handler that is inside another in the term of the
   validation may not be in the scope of [k] in that of Simplify.
   Nothing is done under a recursive [Let_cont] either, nor in the
   bodies of code.

   [share_handlers] does this for every non-recursive continuation of a
   term whose handler Simplify compares ([sharable]), the inner
   [Let_cont]s first. Such a handler is not an exception handler; it
   has one [Let] at least, as a handler that is one [Apply_cont] is a
   shortcut by itself, and 31 at most; its [Apply_cont] has no trap
   action; and none of its [Let]s is of what Simplify may bind to a
   symbol ([may_be_static]). A set of closures is not compared by
   Simplify. An immutable block or array, or a boxed number, is
   compared where its [Let] stays; but at the toplevel of a unit
   Simplify binds it to a symbol, in each handler one of its own, and
   the handlers are then no longer the same, where in the body of code
   it stays a [Let] unless it mentions no variable. The pass is applied
   to both and cannot tell them apart, so it leaves the handlers that
   make such a value.

   It is an equivalence ([share_handlers_equivalent_statement]): where
   the handler that was replaced is applied, the other now is, to the
   same values, under variables and continuations that are the same
   for the names free in it, and the two handlers are the same up to
   bound names (FlambdaCompareSpec.expr_eqb_sound_statement is that
   fact for two expressions under one environment).

   The applications of the continuation are not rewritten here: its
   handler is one [Apply_cont], and
   FlambdaForwardSpec.forward_applications and
   FlambdaShortcutSpec.shortcut_switches take them to the other
   continuation, after which FlambdaUnusedContsSpec.remove_unused_conts
   removes the binding.

   What it leaves, of what Simplify shares: two handlers that are the
   same with their parameters in another order; a handler inside
   another handler, or in a loop, that is the same as one bound around
   it; and the handlers that make a block, an array or a boxed number
   in the body of code. And it shares where Simplify does not: with
   an outer continuation that Simplify has marked cold, whose handler
   Simplify does not keep for the comparison; and with one that
   Simplify inlines, where the rounds of the validation, which inline
   before this pass, have not.

   Renaming: the pass makes no name and compares names by the functions
   above, so it commutes with a renaming and keeps scope; those
   statements are with the others of the validation
   (FlambdaValidatorEquivarianceSpec.share_handlers_equivariant_statement,
   FlambdaLoopsScopeSpec.share_handlers_keeps_scope_statement).

   The proofs are in FlambdaSameHandlersProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaCompareSpec FlambdaNormaliseSpec
  FlambdaUnusedContsSpec.
Import ListNotations.

(* What Simplify may bind to a symbol where the term has a [Let]: a
   set of closures, an immutable block or array, a boxed number. *)
Definition may_be_static (d : named) : bool :=
  match d with
  | Set_of_closures _ _ | Closure_of_set _ _ _ => true
  | Prim (Variadic _) _ | Prim (Unary (Box_number _)) _ => true
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    false
  end.

(* [Let]s, [lets] of them at most and none of what may be static, that
   end in an [Apply_cont] with no trap action. *)
Fixpoint straight (lets : nat) (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ d body =>
    if may_be_static d
    then false
    else
      match lets with
      | O => false
      | S lets => straight lets body
      end
  | Apply_cont ac => no_trap_action ac
  | Let_static _ _ | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _
  | Apply _ | Switch _ _ | Invalid =>
    false
  end.

(* A handler that Simplify compares with those in scope. *)
Definition sharable (h : continuation_handler) : bool :=
  match h with
  | Handler _ metadata (Let _ _ _ _ as handler) =>
    negb (hm_is_exn_handler metadata) && straight 31 handler
  | Handler _ _ _ => false
  end.

Definition handler_params (h : continuation_handler) : list variable :=
  match h with
  | Handler params _ _ => params
  end.

(* [h'] is the same as [h] up to bound names, its parameters are
   distinct, and it names no variable of [xs] and no continuation of
   [ks]. *)
Definition same_handler (h : continuation_handler) (xs : list variable)
    (ks : list continuation) (h' : continuation_handler) : bool :=
  handler_eqb [] [] h' h
  && distinct (handler_params h')
  && negb (existsb (fun x => var_free_in_handler_b x h') xs)
  && negb (existsb (fun c => cont_free_in_handler_b c h') ks).

(* Under the binding of [k] to [h]: a handler that is the same as [h]
   applies [k]. [xs] and [ks] are the variables and the continuations
   bound since, [k] among them. *)
Fixpoint share_with (k : continuation) (h : continuation_handler)
    (xs : list variable) (ks : list continuation) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (share_with k h (x :: xs) ks body)
  | Let_static bindings body => Let_static bindings (share_with k h xs ks body)
  | Let_cont_non_recursive k' (Handler params metadata handler) body =>
    if Pos.eqb k' k
    then e
    else
      Let_cont_non_recursive k'
        (if same_handler h xs ks (Handler params metadata handler)
         then
           Handler params metadata
             (Apply_cont (Apply_cont_expr k (map Var params) None))
         else Handler params metadata handler)
        (share_with k h xs (k' :: ks) body)
  | Let_cont_recursive _ _ | Apply _ | Apply_cont _ | Switch _ _ | Invalid =>
    e
  end.

Fixpoint share_handlers (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (share_handlers body)
  | Let_static bindings body => Let_static bindings (share_handlers body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let h := Handler params metadata (share_handlers handler) in
    let body := share_handlers body in
    Let_cont_non_recursive k h
      (if sharable h then share_with k h [] [k] body else body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (share_handlers body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition share_handlers_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (share_handlers e).
