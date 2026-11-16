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

(* An application of a continuation of no parameters whose handler only
   applies another applies that other: a normalisation that is an
   equivalence.

   This file is to be reviewed: [forward_jumps] and what it is made of,
   and the statement. The validation of
   middle_end/flambda2/tests/semantics/ ran this pass on the term
   after. It runs FlambdaForwardKindedSpec.forward_kinded there now,
   which forwards what this pass forwards and applications with
   arguments too. The pass is kept as the case in which there is
   nothing to know of kinds, and is still extracted.

   FlambdaForwardSpec.forward_applications makes an application of a
   non-recursive continuation whose handler is one [Apply_cont] that
   [Apply_cont], wherever the application is, as Simplify does (the
   header of that file says where). It refines and is not an
   equivalence: an application binds the parameters, and what is put
   in its place does not look at an argument that the handler does not
   pass on. So it is for the term before Simplify.

   The term after Simplify needs the same in one case. Where every
   application of a continuation gives a parameter the same constant,
   Simplify removes the parameter and binds it by a [Let] at the head
   of the handler, once the handler has been rebuilt
   (simplify/simplify_let_cont_expr.ml, [add_lets_around_handler]; the
   parameters that the unboxing of a variant adds are often such). A
   [Switch] on that parameter in the handler was simplified when the
   parameter was not known, and it stays:

     let_cont k () =
       let x = 0 in
       switch x | 0 -> k1 (a) | 1 -> k2 ()
     in
     ... cont k () ... switch y | 0 -> k () | 1 -> ...

   The handler of [k] is not one [Apply_cont], so Simplify records no
   shortcut for [k] and its applications stay. The validation folds
   that [Switch], in the term after as in the term before
   (FlambdaNormaliseSpec.fold_constants), and the handler is then
   [cont k1 (a)]. The term before has its applications forwarded; the
   term after must have them forwarded too, by an equivalence.

   [forward_jumps] is [forward_applications] with entries for the
   continuations of no parameters only. The entry of a continuation
   ([jump_entry]) is that of FlambdaForwardSpec.forwarding if the
   continuation has no parameters and its handler gives no kinds of
   parameters, and none otherwise. An [Apply_cont] that applies such a
   continuation to no arguments, where neither it nor the [Apply_cont]
   of the handler has a trap action, becomes the [Apply_cont] of the
   handler (FlambdaForwardSpec.forward_apply_cont; with no parameters
   the arguments of the handler are left as they are), and an arm of a
   [Switch] in the same way. The [Let_cont] stays;
   FlambdaUnusedContsSpec.remove_unused_conts removes it if nothing
   names its continuation any more.

   Why this case is an equivalence: an application to no arguments
   evaluates nothing and binds nothing, and where the handler gives no
   kinds it tests nothing, so it runs the [Apply_cont] of the handler
   under the variables and continuations of the [Let_cont]; and
   [forwarding] has checked that what the [Apply_cont] names means the
   same where the application is.

   What stops it:
   - what stops [forwarding]: the [Apply_cont] of the handler names a
     continuation, or reads a variable, that is bound again somewhere
     in the body of the [Let_cont]; it names the continuation being
     bound; the continuation is an exception handler;
   - a continuation with parameters;
   - a handler that gives kinds of parameters: one that has no
     parameters and gives a kind never runs
     (FlambdaSemanticsSpec.apply_cont_value), so an application of it
     is undefined, and the [Apply_cont] put in its place might not be;
   - a trap action, on the application or on the [Apply_cont] of the
     handler. (A goto arm of a [Switch] takes the [Apply_cont] of a
     handler with its trap action in Simplify, and in
     FlambdaForwardSpec.forward_arm; not here);
   - recursive continuations are not forwarded, and an [Apply] that
     returns to a continuation still does;
   - the bodies of code are not gone into
     (FlambdaNormaliseSpec.rewrite_code takes the pass there).

   The tests of [forwarding] are on the syntax
   ([FlambdaForwardSpec.binds_cont], FlambdaParamAliasesSpec.binds_var)
   and may say that a name is bound again when it is not, never the
   reverse; an application is then left that could have gone on.

   Renaming: the functions here compare names by equality only and bind
   none, so the pass commutes with a renaming
   (FlambdaValidatorEquivarianceProofs.v).

   The proofs are in FlambdaForwardJumpsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaForwardSpec.
Import ListNotations.

(* The entry of [k] among the continuations in scope: that of
   [forwarding], for a continuation of no parameters whose handler
   gives no kinds. [handler] is the handler once the pass has been
   through it, and [body] the body of the [Let_cont] of [k]. *)
Definition jump_entry (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr)
    : option (list variable * apply_cont) :=
  match params, hm_params_kinds metadata with
  | [], [] => forwarding k [] metadata handler body
  | _, _ => None
  end.

Fixpoint forward_jumps_under (F : forwards) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (forward_jumps_under F body)
  | Let_static bindings body =>
    Let_static bindings (forward_jumps_under F body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler := forward_jumps_under F handler in
    Let_cont_non_recursive k (Handler params metadata handler)
      (forward_jumps_under
         ((k, jump_entry k params metadata handler body) :: F) body)
  | Let_cont_recursive handlers body =>
    let F := map (fun '(k, _) => (k, None)) handlers ++ F in
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_jumps_in_handler F h)) handlers)
      (forward_jumps_under F body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (forward_apply_cont F ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun arm => (fst arm, forward_apply_cont F (snd arm))) arms)
  end
with forward_jumps_in_handler (F : forwards) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (forward_jumps_under F handler)
  end.

Definition forward_jumps (e : expr) : expr := forward_jumps_under [] e.

Definition forward_jumps_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (forward_jumps e).
