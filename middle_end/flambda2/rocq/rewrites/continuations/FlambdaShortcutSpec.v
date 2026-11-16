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

(* An arm of a switch that applies a continuation which only applies
   another: a normalisation that is an equivalence.

   This file is to be reviewed: [shortcut_switches] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify does not keep a non-recursive continuation whose handler,
   once rebuilt, is a single [Apply_cont]: it records the continuation
   as a shortcut to the one its handler applies
   (simplify/simplify_let_cont_expr.ml, where the handler of a
   non-recursive continuation is rebuilt: [Shortcut_to], then
   [Upwards_env.add_continuation_shortcut]), and every application of it
   is rebuilt as an application of that one, with the arguments of the
   handler (simplify/continuation_shortcut.ml, [apply]). The back edge
   of a loop is one: the front end ends the body of a [while] or a [for]
   with a continuation of no parameters that applies the loop, and the
   [Switch] of the test has an arm to it.

   [shortcut_switches] does the case of that shape, which is a fact
   about two expressions on their own. Where a [Let_cont] binds a
   continuation of no parameters whose handler is one [Apply_cont], and
   its body is a [Switch], an arm that applies the continuation to no
   arguments becomes the [Apply_cont] of the handler ([shortcut_arm]):
   the handler runs under the variables of the [Let_cont], which are
   those of the [Switch]. The [Apply_cont] of the handler must not be of
   the name being bound, which in the handler is another continuation.
   Simplify records no shortcut for an exception handler, nor for a
   handler whose [Apply_cont] has a trap action
   (simplify/simplify_let_cont_expr.ml); so this pass makes none, and an
   arm that has a trap action of its own stays as it is. (A goto arm of
   a [Switch] does take the [Apply_cont] of a handler of no parameters
   with its trap action, by another route:
   simplify/simplify_switch_expr.ml, [rebuild_arm]. This pass does not;
   FlambdaForwardSpec.forward_applications does.) The handler
   must give no kinds of parameters either: one that has no parameters
   and gives a kind never runs (FlambdaSemanticsSpec.apply_cont_value),
   so an arm that applies it is undefined, and the arm made of its
   [Apply_cont] would not be. The [Let_cont] goes if no arm names its
   continuation any more (FlambdaNormaliseSpec.arms_apply).

   What it leaves: a continuation with parameters, whose shortcut needs
   the arguments put for the parameters; an application that is not an
   arm of the [Switch] directly under the [Let_cont], in another handler
   or under a [Let]. FlambdaSinkContsSpec.sink_let_conts and
   FlambdaNormaliseSpec.inline_tail_conts must come before this pass in
   a round: a continuation that is applied once by the handler of the
   one of no parameters is inlined there by Simplify, which makes no
   shortcut to a continuation it inlines
   (simplify/simplify_apply_cont_expr.ml, the comment in
   [rebuild_apply_cont]).

   Renaming: the functions here compare names by equality only and bind
   none, so the pass commutes with a renaming. That statement is owed to
   the nominal work and is not made here.

   The proofs are in FlambdaShortcutProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* An arm that applies [k] to no arguments, with no trap action, goes
   where [target] does. *)
Definition shortcut_arm (k : continuation) (target : apply_cont)
    (arm : Z * apply_cont) : Z * apply_cont :=
  match ac_args (snd arm) with
  | [] =>
    if Pos.eqb (ac_continuation (snd arm)) k && no_trap_action (snd arm)
    then (fst arm, target)
    else arm
  | _ :: _ => arm
  end.

Definition shortcut_switch (k : continuation) (h : continuation_handler)
    (body : expr) : expr :=
  match h, body with
  | Handler [] metadata (Apply_cont target), Switch scrutinee arms =>
    match hm_params_kinds metadata with
    | [] =>
      if Pos.eqb (ac_continuation target) k
         || hm_is_exn_handler metadata
         || negb (no_trap_action target)
      then Let_cont_non_recursive k h body
      else
        let arms := map (shortcut_arm k target) arms in
        if arms_apply k arms
        then Let_cont_non_recursive k h (Switch scrutinee arms)
        else Switch scrutinee arms
    | _ :: _ => Let_cont_non_recursive k h body
    end
  | _, _ => Let_cont_non_recursive k h body
  end.

Fixpoint shortcut_switches (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (shortcut_switches body)
  | Let_static bindings body => Let_static bindings (shortcut_switches body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    shortcut_switch k (Handler params metadata (shortcut_switches handler))
      (shortcut_switches body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (shortcut_switches body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition shortcut_switches_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (shortcut_switches e).
