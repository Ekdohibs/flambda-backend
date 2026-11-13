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

(* The rewrites of the validation for loops keep scope.

   This file is to be reviewed: the statements.

   FlambdaKeepsScopeSpec.v says what it is for a rewrite to keep scope
   (FlambdaValidatorEquivarianceSpec.keeps_scope), why it matters, and
   that the rewrites of a round do. This file says the same of the
   rewrites that were written for loops, one statement to a rewrite:
   those that a round applies as they are
   (FlambdaSinkContsSpec.sink_let_conts,
   FlambdaShortcutSpec.shortcut_switches,
   FlambdaSameHandlersSpec.share_handlers,
   FlambdaUnusedParamsSpec.drop_unused_params,
   FlambdaDeadParamsSpec.drop_dead_params,
   FlambdaParamAliasesSpec.bind_aliased_params,
   FlambdaForwardSpec.forward_applications); the going of an
   application to the arm of the number that it gives
   (FlambdaForwardArmSpec.forward_to_arms), the way of taking that
   pass that asks nothing of the kinds; the removal of the unused
   [Let]s that have a value, in loops too
   (FlambdaLoopLetsSpec.remove_unused_defined_lets_in_loops), whatever
   is taken to be in scope; the following of
   aliases into the handlers of recursive continuations
   (FlambdaLoopAliasesSpec.follow_aliases_in_loops), with nothing known
   to begin with, as a round applies it; a rewrite that keeps scope,
   taken into the handlers of the recursive continuations
   (FlambdaRecHandlersSpec.rewrite_rec_handlers); the moving of a
   continuation down past the bindings under it
   (FlambdaSinkUsesSpec.sink_conts_to_uses), which the validation
   applies once, when the rounds are over, to each term that it then
   compares; and loopify (FlambdaLoopifySpec.loopify_unit), which
   the validation applies to the term before after its first
   comparison fails, for the code and the names of loops that it is
   given.

   Closed code is all that is asked of a term, as [keeps_scope] asks
   it. Loopify binds a name that its caller gives: the code it makes is
   closed whatever that name is, since the loop binds it. Three of the
   rewrites put a variable where it was not: [bind_aliased_params]
   binds a parameter, at the start of its handler, to what every
   application gives it, [forward_applications] gives an application
   the arguments of the handler it goes to, and [forward_to_arms]
   gives it those of the arm that it goes to, the arm that the
   [Switch] of the handler has for the number given. Each keeps
   scope because the variable was free already: in the body of the
   [Let_cont], as an argument of an application there, or in the
   handler. The continuation that an application is made to go to by
   the last two was free in the handler too.
   [share_handlers] puts a continuation in a handler that did not
   name it: the one whose [Let_cont] the handler is under, which is
   bound there and so is not free in the term.

   Nothing is trusted here. The proofs are in
   FlambdaLoopsScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNormaliseSpec
  FlambdaDeadParamsSpec FlambdaForwardSpec FlambdaForwardArmSpec
  FlambdaLoopAliasesSpec
  FlambdaLoopLetsSpec FlambdaLoopifySpec FlambdaParamAliasesSpec
  FlambdaRecHandlersSpec FlambdaShortcutSpec FlambdaSameHandlersSpec
  FlambdaSinkContsSpec FlambdaSinkUsesSpec
  FlambdaUnusedParamsSpec FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec.
Import ListNotations.

(* The rewrites that a round applies as they are. *)
Definition sink_let_conts_keeps_scope_statement : Prop :=
  keeps_scope sink_let_conts.

Definition shortcut_switches_keeps_scope_statement : Prop :=
  keeps_scope shortcut_switches.

Definition share_handlers_keeps_scope_statement : Prop :=
  keeps_scope share_handlers.

Definition drop_unused_params_keeps_scope_statement : Prop :=
  keeps_scope drop_unused_params.

Definition drop_dead_params_keeps_scope_statement : Prop :=
  keeps_scope drop_dead_params.

Definition bind_aliased_params_keeps_scope_statement : Prop :=
  keeps_scope bind_aliased_params.

Definition forward_applications_keeps_scope_statement : Prop :=
  keeps_scope forward_applications.

(* The same with the calls taken on or not
   (FlambdaForwardSpec.forward_applications_with). *)
Definition forward_applications_with_keeps_scope_statement : Prop :=
  ∀ calls : bool, keeps_scope (forward_applications_with calls).

(* An application that gives a number goes to the arm of that number,
   with nothing asked of the kinds
   (FlambdaForwardArmSpec.forward_to_arms). *)
Definition forward_to_arms_keeps_scope_statement : Prop :=
  keeps_scope forward_to_arms.

(* Whatever the variables that are taken to be in scope, and the kinds
   they are taken to have. *)
Definition remove_unused_defined_lets_in_loops_keeps_scope_statement : Prop :=
  ∀ kinds : kinds_in_scope,
    keeps_scope (remove_unused_defined_lets_in_loops kinds).

(* As a round applies it: with nothing known to begin with. *)
Definition follow_aliases_in_loops_keeps_scope_statement : Prop :=
  keeps_scope (follow_aliases_in_loops []).

(* A rewrite that keeps scope, taken into the handlers of the recursive
   continuations. *)
Definition rewrite_rec_handlers_keeps_scope_statement : Prop :=
  ∀ f : expr -> expr, keeps_scope f -> keeps_scope (rewrite_rec_handlers f).

(* What the validation applies once, when the rounds are over. *)
Definition sink_conts_to_uses_keeps_scope_statement : Prop :=
  keeps_scope sink_conts_to_uses.

(* Loopify, whatever the code that is to be made a loop and the names
   given for the loops. *)
Definition loopify_unit_keeps_scope_statement : Prop :=
  ∀ G : loopified, keeps_scope (loopify_unit G).
