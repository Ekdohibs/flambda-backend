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

(* Loopify: the self tail calls of a function become jumps to a recursive
   continuation around its body. A rewrite of the term before Simplify,
   which refines it.

   This file is to be reviewed: [loopify_unit] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   What Simplify does. The front end marks a function for it
   (from_lambda/closure_conversion.ml: [Default_loopify_and_tailrec] for a
   single recursive function whose recursive calls are all in tail
   position, [Always_loopify] for [@loop]; terms/loopify_attribute.ml,
   [should_loopify]). For such code, simplify/simplify_set_of_closures.ml
   ([simplify_function_body]) makes a continuation, [self], and
   simplify/simplify_expr.ml ([simplify_function_body], the case
   [Loopify]) simplifies, in place of the body,

     let_cont rec self (params) = body in cont self (params)

   with the parameters of the code, and their kinds, as those of the
   continuation. In it,
   simplify/simplify_apply_expr.ml ([loopify_decision_for_call], then
   [simplify_self_tail_call]) makes a call an [Apply_cont] of [self] to
   the arguments of the call when
   - its callee is, up to aliases, the [my_closure] of the code;
   - its return continuation is that of the code (or it never returns);
   - its exception continuation is that of the code, with no extra
     arguments.
   So a call counts as a tail call by where it returns, and nothing else
   is asked: not that the call names the code, not its region (see the
   CR at [is_self_tail_call]). The call is a full application of known
   code by then: Simplify makes the decision in
   [simplify_direct_full_application], and takes a call with fewer or
   more arguments than the code has parameters for a partial application
   or an over-application first ([simplify_direct_function_call]).

   What [loopify_unit] does. For each code that a [Let_static] of the
   term binds and whose code id is in the list given, with a
   continuation name for [self], it replaces the code by
   [looped_code], if [loopable] holds of it, and leaves it alone
   otherwise. The conditions, all checked:
   - [my_closure] is not a parameter, and no [Let] or handler of the
     body binds it ([loop_safe]): then a callee that is [Var my_closure]
     is the closure the code was called with;
   - no name is two of the parameters (FlambdaMapSpec.distinct): the
     loop is entered by a jump to [self] with the parameters as its
     arguments, and the handler of [self] has no step unless their
     values are of the kinds of the parameters
     (FlambdaSemanticsSpec.apply_cont_value). Were one name two
     parameters of different kinds, both would have the value of the
     second at the jump, which would then have no step where the call
     had one;
   - no [Let_cont] of the body binds the return or the exception
     continuation of the code ([loop_safe]): then a call that returns
     to them returns where the code does;
   - [self] is neither of those two, and does not occur in the body,
     bound or applied ([loop_safe]);
   - some call was made a jump (FlambdaUnusedContsSpec.cont_free_b of
     [self]): Simplify does not keep a loop that nothing jumps to;
   - the code is not tupled (FlambdaModel.code_is_tupled). A call of
     this fragment gives tupled code one argument, a tuple, which the
     machine takes apart (FlambdaSemanticsSpec.arguments_for_code); a
     jump binds the parameters to the arguments as they are, so it would
     not refine the call. The compiler's self tail call of tupled code is
     a direct call, which gives the code its arguments as the code takes
     them, and this fragment has not got that call yet.
   A self tail call ([is_self_tail_call]) is an [Apply] whose callee is
   [Var my_closure], whose two continuations are those of the code, the
   exception continuation with no extra arguments, and which is made at
   an arity of as many components as the code has parameters. With extra
   arguments, an exception of the body would reach the handler with them
   after the call and without them after the jump. With fewer or more
   components the call is a partial application or an over-application
   (FlambdaSemanticsSpec.shape_of_closure_call, which compares the same
   two lengths), which does not run the body with the arguments of the
   call, so the jump would not refine it.
   What is not asked, and why it is right without:
   - the call kind. A direct call that names other code than that of the
     closure is undefined (FlambdaSemanticsSpec.shape_of_closure_call),
     so anything refines it;
   - the kinds of the arguments of the call, and that its arities are
     those of the code. A full call that does not agree with the code in
     them is undefined (FlambdaSemanticsSpec.call_code), so the jump,
     which tests the kinds of its arguments against those of the
     parameters and their number, refines it; with the wrong number of
     arguments the jump is undefined as the call is;
   - that the callee has been proved to be this code: the closure a
     piece of code is called with is a closure of that code, by how a
     call finds the code.

   Why it is right: a self tail call runs the body again with the
   parameters bound to the arguments, [my_closure] to the same closure,
   and the same two continuations; there is no stack to push
   (FlambdaSemanticsSpec.call_code). The jump runs the looped body
   with the parameters rebound, under the variables and continuations of
   the first entry, where [my_closure] and the two continuations are the
   same. So the two runs go step for step, but at the entry of the code,
   where the looped code takes two more steps to bind [self] and jump
   to it.

   The statement ([loopify_unit_refines_statement]) is refinement under
   all environments. The list is a guess of the caller, and the check
   fails closed: code that is not [loopable] is not changed. The name
   for [self] is given by the caller too, and checked.
   FlambdaNameAbstractionSpec.fresh_name makes a name outside a list of
   names; taking [self] from it, over the continuations of the body,
   would spare the caller the guess, and may replace it.

   Regions and depths. The model has no regions, so the rewrite says
   nothing of the region that a self tail call passes for its result,
   which Simplify does not compare with the function's (the CR at
   [is_self_tail_call]). The depth variable of the call ([my_depth],
   [next_depth]) is dropped by the conversion to the model, as
   everywhere.

   Renaming: [loop_calls] compares names by equality only; [looped_code]
   binds the name given for [self], so it commutes with a renaming that
   is applied to that name too. That statement is owed to the nominal
   work and is not made here.

   The proofs are in FlambdaLoopifyProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaUnusedContsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The body of the loop                                                     *)
(* ======================================================================== *)

(* CR mshinwell: a self tail call passes the region for its result
   ([Apply.return_mode]) and has a position; Simplify's
   [loopify_decision_for_call] compares neither with the function's, and
   the model has no regions, so this rewrite says nothing about them. A
   trial from source found no call that passes the three tests with a
   return mode other than the function's own region parameters: the type
   checker refuses a local-returning call in tail position outside an
   exclave, and the conversion from Lambda gives a call the function's
   return continuation only where no region that the function opened is
   still open. The position can differ ([@nontail] inside an exclave),
   which looks harmless. Discrepancy V4-1. *)

(* A full call of the closure the code was called with, returning where
   the code returns. [params_arity] is the arity of the parameters of
   the code. *)
Definition is_self_tail_call (my_closure : variable) (params_arity : arity)
    (ret exn : continuation) (ap : apply_expr) : bool :=
  match ap_callee ap with
  | Var x => Pos.eqb x my_closure
  | Symbol _ | Const _ => false
  end
  && Nat.eqb (length (ap_args_arity ap)) (length params_arity)
  && Pos.eqb (ap_continuation ap) ret
  && Pos.eqb (ap_exn_continuation ap) exn
  && no_exn_extra_args ap.

(* The expression with its self tail calls made applications of [self] to
   the arguments of the call. *)
Fixpoint loop_calls (my_closure : variable) (params_arity : arity)
    (ret exn self : continuation) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (loop_calls my_closure params_arity ret exn self body)
  | Let_static bindings body =>
    Let_static bindings (loop_calls my_closure params_arity ret exn self body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (loop_calls_in_handler my_closure params_arity ret exn self h)
      (loop_calls my_closure params_arity ret exn self body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, loop_calls_in_handler my_closure params_arity ret exn self h))
         handlers)
      (loop_calls my_closure params_arity ret exn self body)
  | Apply ap =>
    if is_self_tail_call my_closure params_arity ret exn ap
    then Apply_cont (Apply_cont_expr self (ap_args ap) None)
    else e
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with loop_calls_in_handler (my_closure : variable) (params_arity : arity)
    (ret exn self : continuation) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (loop_calls my_closure params_arity ret exn self handler)
  end.

(* No [Let] or handler of the expression binds [my_closure]; no
   [Let_cont] binds [ret], [exn] or [self]; and [self] is not a
   continuation of any [Apply], nor the continuation or the name in the
   trap action of any [Apply_cont] or arm of a [Switch]. *)
Fixpoint loop_safe (my_closure : variable) (ret exn self : continuation)
    (e : expr) : bool :=
  match e with
  | Let x _ _ body =>
    negb (Pos.eqb x my_closure) && loop_safe my_closure ret exn self body
  | Let_static _ body => loop_safe my_closure ret exn self body
  | Let_cont_non_recursive k h body =>
    negb (Pos.eqb k ret) && negb (Pos.eqb k exn) && negb (Pos.eqb k self)
    && loop_safe_handler my_closure ret exn self h
    && loop_safe my_closure ret exn self body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k, h) =>
         negb (Pos.eqb k ret) && negb (Pos.eqb k exn) && negb (Pos.eqb k self)
         && loop_safe_handler my_closure ret exn self h)
      handlers
    && loop_safe my_closure ret exn self body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) self)
    && negb (Pos.eqb (ap_exn_continuation ap) self)
  | Apply_cont ac => negb (among self (apply_cont_continuations ac))
  | Switch _ arms =>
    forallb
      (fun arm => negb (among self (apply_cont_continuations (snd arm))))
      arms
  | Invalid => true
  end
with loop_safe_handler (my_closure : variable) (ret exn self : continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler =>
    negb (among my_closure params)
    && loop_safe my_closure ret exn self handler
  end.

(* ======================================================================== *)
(* Code                                                                     *)
(* ======================================================================== *)

(* The body of the code as the handler of [self], with its self tail
   calls made jumps. Its parameters are those of the code, with their
   kinds; it is not an exception handler, and the jumps have no trap
   action. *)
Definition loop_handler (self : continuation) (c : code)
    : continuation_handler :=
  Handler (code_params c) (make_handler_metadata (code_params_kinds c))
    (loop_calls (code_my_closure c) (code_params_arity c)
       (code_return_continuation c) (code_exn_continuation c) self
       (code_body c)).

(* The code with the loop for its body, and nothing else changed. *)
Definition looped_code (self : continuation) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata _ =>
    Code params my_closure ret exn metadata
      (Let_cont_recursive [(self, loop_handler self c)]
         (Apply_cont (Apply_cont_expr self (map Var params) None)))
  end.

Definition loopable (self : continuation) (c : code) : bool :=
  let looped :=
    loop_calls (code_my_closure c) (code_params_arity c)
      (code_return_continuation c) (code_exn_continuation c) self
      (code_body c)
  in
  negb (among (code_my_closure c) (code_params c))
  && distinct (code_params c)
  && negb (Pos.eqb self (code_return_continuation c))
  && negb (Pos.eqb self (code_exn_continuation c))
  && loop_safe (code_my_closure c) (code_return_continuation c)
       (code_exn_continuation c) self (code_body c)
  && cont_free_b self looped
  && negb (code_is_tupled c).

Definition loopify_code (self : continuation) (c : code) : code :=
  if loopable self c then looped_code self c else c.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The code to loopify, each with the name for its [self]. *)
Definition loopified : Type := list (code_id * continuation).

Definition loopify_binding (G : loopified) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id c =>
    match lookup id G with
    | Some self => Bind_code id (loopify_code self c)
    | None => binding
    end
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* The code of every [Let_static] of the term that may be executed: one
   in the body of code or in the handler of a recursive continuation is
   not, as such a call or [Let_cont] is undefined
   (FlambdaSemanticsSpec.binds_statics). *)
Fixpoint loopify_unit (G : loopified) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (loopify_unit G body)
  | Let_static bindings body =>
    Let_static (map (loopify_binding G) bindings) (loopify_unit G body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (loopify_unit G handler))
      (loopify_unit G body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (loopify_unit G body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition loopify_unit_refines_statement : Prop :=
  ∀ (G : loopified) (e : expr), expr_refines e (loopify_unit G e).
