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

(* A parameter that its handler does not use, of a continuation that a
   call returns to, is removed, and the call returns to a wrapper: a
   normalisation that refines.

   This file is to be reviewed: [wrap_returns] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. The
   validation applies the pass to the term before Simplify.

   Simplify removes a parameter that the handler of a continuation
   does not use, with the argument that each application gives it
   (simplify/simplify_let_cont_expr.ml,
   [decide_param_usage_non_recursive]: a parameter that the rebuilt
   handler does not mention is [Unused] in the [Apply_cont_rewrite] of
   the continuation). A call that returns to the continuation gives it
   the results of the call, which are not written in the term and
   cannot lose one: Simplify makes the call return to a new
   continuation, bound at the call, whose parameters are those results
   and whose handler applies the old one to those that are left
   (simplify/expr_builder.ml, [rewrite_fixed_arity_apply] and
   [rewrite_fixed_arity_continuation0], the case [New_wrapper]). The
   source

     let _ = if c then f x else 8 in ...

   is of this kind: the continuation after the test is applied to a
   constant in one branch and is returned to by the call in the
   other, and nothing uses its parameter.

   FlambdaUnusedParamsSpec.drop_unused_params and
   FlambdaDeadParamsSpec.drop_dead_params, which the rounds of the
   term before apply, remove the parameters that are not used and
   stop at a continuation that a call returns to
   (FlambdaUnusedParamsSpec.returned_to). [wrap_returns] is for those
   continuations. For the non-recursive [Let_cont] of [k], and the
   parameter at one position of its handler ([wrap_return]):

   - the handler loses the parameter;
   - every application of [k] in the body of the [Let_cont] loses its
     argument at that position (FlambdaUnaliasParamsSpec.drop_arg_of);
   - every call in the body that returns to [k] is put under a
     [Let_cont] of a wrapper ([wrapper]): a handler with the
     parameters and the kinds that the handler of [k] had, which
     applies [k] to its parameters but the one at that position.

   The wrapper is bound to the name [k] itself, and its parameters
   have the names of those of [k]. So the pass makes no name. The
   handler of a non-recursive [Let_cont] is not in the scope of the
   name that the [Let_cont] binds: in [let_cont k (x) = cont k () in
   apply f -> k] the [k] that is applied is the one bound outside, and
   the call returns to the one bound here. The validation gives every
   binder a name of its own in its next round
   (FlambdaRenameApartSpec.rename_apart), and its comparison is up to
   bound names.

   What the pass asks ([wrap_return]):
   - that the handler is not an exception handler, that its parameters
     are distinct, and that it does not use the parameter
     (FlambdaNormaliseSpec.var_free_b, which may say that a variable is
     used when it is not, never the reverse);
   - that every mention of [k] in the body of the [Let_cont] is an
     application of [k] or the return continuation of a call
     ([only_applied_or_returned_to]): no call raises to [k], no trap
     action names it, and no [Let_cont] inside binds [k] again. The
     bodies of code are not looked into: code runs with its own
     continuations, and a [k] there is not this one;
   - that [k] is mentioned twice at least ([uses_of]). Where the one
     mention of [k] is a call that returns to it, Simplify inlines the
     handler in the wrapper that it makes, which gives the term back
     with other names, and the pass leaves such a term as it is.
     Where no call returns to [k] the pass does what the two passes
     named above do.
   The positions are tried from the last to the first, and the pass
   stops at the first parameter that it removes ([wrap_from]): the
   wrappers bind [k] again, so the pass would refuse the next. The
   other parameters that are not used are removed in the rounds that
   follow, by the two passes named above: once the wrappers have names
   of their own, no call returns to [k]. Every non-recursive
   [Let_cont] of the term is looked at, the innermost first
   ([wrap_returns]). The handlers of recursive continuations and the
   bodies of code are not gone into:
   FlambdaNormaliseSpec.rewrite_code and [rewrite_rec_handlers] take a
   pass there.

   The result refines the term ([wrap_returns_refines_statement]),
   under all environments. Nothing is asked of the names of the term
   beyond what the pass checks itself. It is stated as a refinement
   and not as an equivalence: the machine tests the kind of a value
   where a handler is applied, and the test of the parameter that is
   removed goes with it where an [Apply_cont] applied [k]. (Where a
   call returned to [k] the wrapper makes the test.)

   Renaming: the functions here compare names by equality only and
   make none. The pass commutes with every renaming and keeps scope;
   those two statements are in FlambdaValidatorEquivarianceSpec.v and
   in FlambdaStaticsScopeSpec.v.

   The proofs are in FlambdaWrapReturnsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaUnaliasParamsSpec.
Import ListNotations.

(* No trap action of an [Apply_cont] names [k]. An application of [k]
   may carry a trap action that names another continuation. *)
Definition trap_free_of (k : continuation) (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some action => negb (Pos.eqb (trap_action_handler action) k)
  | None => true
  end.

(* Every mention of [k] in a term is an application of [k] or the
   return continuation of a call, and [k] is not bound again. *)
Fixpoint only_applied_or_returned_to (k : continuation) (e : expr)
    {struct e} : bool :=
  match e with
  | Let _ _ _ body => only_applied_or_returned_to k body
  | Let_static _ body => only_applied_or_returned_to k body
  | Let_cont_non_recursive k' h body =>
    negb (Pos.eqb k' k)
    && handler_only_applied_or_returned_to k h
    && only_applied_or_returned_to k body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k', h) =>
         negb (Pos.eqb k' k) && handler_only_applied_or_returned_to k h)
      handlers
    && only_applied_or_returned_to k body
  | Apply ap => negb (Pos.eqb (ap_exn_continuation ap) k)
  | Apply_cont ac => trap_free_of k ac
  | Switch _ arms => forallb (fun '(_, ac) => trap_free_of k ac) arms
  | Invalid => true
  end
with handler_only_applied_or_returned_to (k : continuation)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler => only_applied_or_returned_to k handler
  end.

(* How many times a term applies [k] or has a call that returns to
   [k]. *)
Definition one_if_same (k k' : continuation) : nat :=
  if Pos.eqb k' k then 1 else 0.

Fixpoint uses_of (k : continuation) (e : expr) {struct e} : nat :=
  match e with
  | Let _ _ _ body => uses_of k body
  | Let_static _ body => uses_of k body
  | Let_cont_non_recursive _ h body =>
    uses_in_handler k h + uses_of k body
  | Let_cont_recursive handlers body =>
    list_sum (map (fun '(_, h) => uses_in_handler k h) handlers)
    + uses_of k body
  | Apply ap => one_if_same k (ap_continuation ap)
  | Apply_cont ac => one_if_same k (ac_continuation ac)
  | Switch _ arms =>
    list_sum (map (fun '(_, ac) => one_if_same k (ac_continuation ac)) arms)
  | Invalid => 0
  end
with uses_in_handler (k : continuation) (h : continuation_handler)
    {struct h} : nat :=
  match h with
  | Handler _ _ handler => uses_of k handler
  end.

(* What a call that returned to [k] returns to once the handler of [k]
   has lost its parameter at position [i]: a handler with the
   parameters and the kinds that the handler of [k] had, which applies
   [k] to its parameters but that one. *)
Definition wrapper (k : continuation) (i : nat) (params : list variable)
    (metadata : handler_metadata) : continuation_handler :=
  Handler params metadata
    (Apply_cont
       {| ac_continuation := k;
          ac_args := map Var (drop_nth i params);
          ac_trap_action := None |}).

(* Every application of [k] without its argument at a position, and
   every call that returns to [k] under a [Let_cont] that binds [k] to
   the handler [w]. *)
Fixpoint wrap_uses (k : continuation) (i : nat) (w : continuation_handler)
    (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (wrap_uses k i w body)
  | Let_static bindings body => Let_static bindings (wrap_uses k i w body)
  | Let_cont_non_recursive k' h body =>
    Let_cont_non_recursive k' (wrap_uses_in_handler k i w h)
      (wrap_uses k i w body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k', h) => (k', wrap_uses_in_handler k i w h)) handlers)
      (wrap_uses k i w body)
  | Apply ap =>
    if Pos.eqb (ap_continuation ap) k
    then Let_cont_non_recursive k w (Apply ap)
    else e
  | Apply_cont ac => Apply_cont (drop_arg_of k i ac)
  | Switch scrutinee arms =>
    Switch scrutinee (map (fun '(n, ac) => (n, drop_arg_of k i ac)) arms)
  | Invalid => e
  end
with wrap_uses_in_handler (k : continuation) (i : nat)
    (w : continuation_handler) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (wrap_uses k i w handler)
  end.

(* The handler [h] of [k] and the body of its [Let_cont] without the
   parameter at position [i], which [h] does not use: the applications
   of [k] in the body lose the argument, and the calls that return to
   [k] return to a wrapper. An exception handler is left as it is, and
   so is a continuation that is mentioned once only. *)
Definition wrap_return (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (continuation_handler * expr) :=
  match h with
  | Handler params metadata handler =>
    match nth_error params i with
    | Some p =>
      if negb (hm_is_exn_handler metadata)
         && distinct params
         && negb (var_free_b p handler)
         && only_applied_or_returned_to k body
         && (2 <=? uses_of k body)
      then
        Some
          (Handler (drop_nth i params)
             {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
                hm_is_exn_handler := false |}
             handler,
           wrap_uses k i (wrapper k i params metadata) body)
      else None
    | None => None
    end
  end.

(* The same for the last position below [i] at which it can be done. *)
Fixpoint wrap_from (i : nat) (k : continuation) (h : continuation_handler)
    (body : expr) : continuation_handler * expr :=
  match i with
  | O => (h, body)
  | S i =>
    match wrap_return i k h body with
    | Some (h', body') => (h', body')
    | None => wrap_from i k h body
    end
  end.

(* The pass: every non-recursive [Let_cont] of a term, its handler and
   its body first. The handlers of recursive continuations and the
   bodies of code are left as they are. *)
Fixpoint wrap_returns (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (wrap_returns body)
  | Let_static bindings body => Let_static bindings (wrap_returns body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let h := Handler params metadata (wrap_returns handler) in
    let body := wrap_returns body in
    let '(h, body) := wrap_from (length params) k h body in
    Let_cont_non_recursive k h body
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (wrap_returns body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result refines the term, under all environments. *)
Definition wrap_returns_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (wrap_returns e).
