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

(* Removing the parameters of a continuation that its handler does not
   use: a normalisation that refines.

   This file is to be reviewed: [drop_unused_params] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   Simplify removes a parameter of a continuation that the handler, as
   rebuilt, does not use, and with it the argument at every application
   (simplify/simplify_let_cont_expr.ml: [decide_param_usage_non_recursive]
   asks the free names of the rebuilt handler, and
   [Apply_cont_rewrite.create] records the answer for each parameter;
   simplify/apply_cont_rewrite.ml, [make_rewrite], drops the arguments
   of the parameters that are [Unused] where simplify/expr_builder.ml
   rebuilds an [Apply_cont]). The front end makes such parameters all
   the time: the continuation after a [while] or a [for] loop takes the
   unit that the loop returns and every variable the loop assigns, and
   uses at most the latter.

   [drop_unused_params] does it for the continuations of a term. Going
   down the term it keeps, for each continuation in scope, which of its
   parameters are kept ([masks]): [None] for all of them, or one boolean
   for each. At a non-recursive [Let_cont] the handler is done first; the
   parameters that are then not free in it
   (FlambdaNormaliseSpec.var_free_b) are removed from the handler, with
   their kinds in its metadata ([keep_kinds]), and
   in the body of the [Let_cont] the arguments at their positions are
   removed from every [Apply_cont] and every arm of a [Switch] that
   applies the continuation ([drop_args]), wherever it is: in the
   handlers of other continuations and in those of recursive ones, so in
   a loop that exits through the continuation. At a recursive [Let_cont]
   the parameters removed from a handler are those that are not free in
   it as it stands, before anything is removed from it
   ([used_params_in_loop]); their arguments are removed from the
   applications in the handlers of the [Let_cont], which are then done,
   and in its body.

   What stops it:
   - a continuation that some [Apply] of the body has as its return or
     exception continuation keeps all its parameters ([returned_to]): the
     call passes it as it is. Simplify puts a wrapper continuation
     around such a call (simplify/expr_builder.ml, [New_wrapper]), or
     removes the extra arguments of its exception continuation; this
     pass has no fresh names to make a wrapper with, and leaves both;
   - the first parameter of an exception handler, which is the
     exception, is kept whether it is used or not, as Simplify keeps it
     ([decide_param_usage_non_recursive], its [exn_bucket]). The other
     parameters of an exception handler that only raises reach (an
     [Apply_cont] with a trap action is one) go as those of any
     continuation; the trap action of an application stays as it is;
   - a parameter of a recursive continuation that its handler uses only
     to pass it on, to itself or to another continuation whose handler
     does not use it, is kept. Simplify finds those too, by its data flow
     analysis ([decide_param_usage_recursive]), which this pass has not
     got; FlambdaDeadParamsSpec.drop_dead_params is this pass with a
     search for them. A parameter that the handler binds again at once, as
     FlambdaParamAliasesSpec.bind_aliased_params makes it do where every
     application gives the parameter the same thing, is one that the
     handler does not use;
   - the bodies of code are not gone into
     (FlambdaNormaliseSpec.rewrite_code takes the pass there).

   The result refines the term ([drop_unused_params_refines_statement]),
   under all environments. It is not equivalent to it: an argument that
   is removed is no longer evaluated, so a term that was undefined for a
   variable with no value there may become defined. So the pass is for
   the term before Simplify.

   What is checked and what is not: nothing is asked of the names of
   the term, and nothing of the arities. Where an application has not as
   many arguments as the continuation has parameters, the arguments are
   cut as the mask says and the application, which was undefined, may
   become defined. Where a parameter is bound twice in one handler, both
   are kept or both removed, as [var_free_b] answers of the name.
   [returned_to] and [var_free_b] may say yes when the answer is no,
   never the reverse.

   Renaming: the functions here compare names by equality only and bind
   none, so the pass commutes with a renaming. That statement is owed to
   the nominal work and is not made here.

   The proofs are in FlambdaUnusedParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* Which parameters are kept                                                *)
(* ======================================================================== *)

(* The elements of a list at the positions where the mask says yes. Past
   the end of the shorter of the two there is nothing. *)
Fixpoint keep {A : Type} (mask : list bool) (l : list A) : list A :=
  match mask, l with
  | true :: mask, a :: l => a :: keep mask l
  | false :: mask, _ :: l => keep mask l
  | _, _ => []
  end.

(* [None]: every element is kept. *)
Definition keep_all_or {A : Type} (mask : option (list bool)) (l : list A)
    : list A :=
  match mask with
  | Some mask => keep mask l
  | None => l
  end.

(* For the continuations in scope, the most recently bound first, which
   parameters are kept. A continuation that is not listed keeps them
   all. *)
Definition masks : Type := list (continuation * option (list bool)).

Definition mask_of (known : masks) (k : continuation) : option (list bool) :=
  match lookup k known with
  | Some mask => mask
  | None => None
  end.

(* An application with the arguments of removed parameters removed. Its
   trap action stays. *)
Definition drop_args (known : masks) (ac : apply_cont) : apply_cont :=
  Apply_cont_expr (ac_continuation ac)
    (keep_all_or (mask_of known (ac_continuation ac)) (ac_args ac))
    (ac_trap_action ac).

(* Whether some [Apply] of the expression may have [k] as its return or
   exception continuation. A binding of the same name in between is not
   looked at. *)
Fixpoint returned_to (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => returned_to k body
  | Let_cont_non_recursive _ h body =>
    returned_to_in_handler k h || returned_to k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => returned_to_in_handler k h) handlers
    || returned_to k body
  | Apply ap =>
    Pos.eqb k (ap_continuation ap) || Pos.eqb k (ap_exn_continuation ap)
  | Apply_cont _ | Switch _ _ | Invalid => false
  end
with returned_to_in_handler (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => returned_to k handler
  end.

(* Which parameters a handler uses, if not all. The first parameter of
   an exception handler, which is the exception, counts as used. *)
Definition used_params (is_exn_handler : bool) (params : list variable)
    (handler : expr) : option (list bool) :=
  let mask := map (fun param => var_free_b param handler) params in
  let mask :=
    match is_exn_handler, mask with
    | true, _ :: mask => true :: mask
    | _, _ => mask
    end
  in
  if forallb (fun used => used) mask then None else Some mask.

(* The metadata of a handler with the kinds of the parameters kept: a
   parameter that goes takes its kind with it. An exception handler
   stays one. *)
Definition keep_kinds (mask : option (list bool))
    (metadata : handler_metadata) : handler_metadata :=
  match mask with
  | Some mask =>
    let kinds := keep mask (hm_params_kinds metadata) in
    if hm_is_exn_handler metadata then make_exn_handler_metadata kinds
    else make_handler_metadata kinds
  | None => metadata
  end.

(* Which parameters the handler of a recursive continuation uses, if not
   all, as the handler stands. [whole] is the [Let_cont] of the
   continuation: an [Apply] in one of its handlers or in its body that
   returns to the continuation keeps all its parameters. *)
Definition used_params_in_loop (k : continuation) (h : continuation_handler)
    (whole : expr) : option (list bool) :=
  match h with
  | Handler params metadata handler =>
    if returned_to k whole then None
    else used_params (hm_is_exn_handler metadata) params handler
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

Fixpoint drop_params (known : masks) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (drop_params known body)
  | Let_static bindings body => Let_static bindings (drop_params known body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler := drop_params known handler in
    let mask :=
      if returned_to k body then None
      else used_params (hm_is_exn_handler metadata) params handler
    in
    Let_cont_non_recursive k
      (Handler (keep_all_or mask params) (keep_kinds mask metadata) handler)
      (drop_params ((k, mask) :: known) body)
  | Let_cont_recursive handlers body =>
    let known :=
      map (fun '(k, h) => (k, used_params_in_loop k h e)) handlers ++ known
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, drop_params_in_loop known (used_params_in_loop k h e) h))
         handlers)
      (drop_params known body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (drop_args known ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun arm => (fst arm, drop_args known (snd arm))) arms)
  end
with drop_params_in_loop (known : masks) (mask : option (list bool))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (keep_all_or mask params) (keep_kinds mask metadata)
      (drop_params known handler)
  end.

Definition drop_unused_params (e : expr) : expr := drop_params [] e.

Definition drop_unused_params_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (drop_unused_params e).
