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

(* An application of a continuation whose handler only applies another
   applies that other: a normalisation that refines.

   This file is to be reviewed: [forward_applications_with],
   [forward_applications] and what they are made of, which are
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   Simplify does not go through a non-recursive continuation whose
   handler, once rebuilt, is a single [Apply_cont]:

   - it records the continuation as a shortcut to the one its handler
     applies (simplify/simplify_let_cont_expr.ml, where the handler of a
     non-recursive continuation is rebuilt: [Shortcut_to], unless the
     continuation is an exception handler, is inlined, or the
     [Apply_cont] has a trap action; then
     [Upwards_env.add_continuation_shortcut]), and every application of
     it is rebuilt as an application of that one, with the arguments of
     the application put for the parameters in the arguments of the
     handler (simplify/continuation_shortcut.ml, [apply]), and with
     the trap action of the application, if it has one
     (simplify/expr_builder.ml, [apply_continuation_shortcuts]);
   - an arm of a [Switch] that is a goto, with no argument and no trap
     action, to a continuation of no parameters whose handler is known
     becomes the [Apply_cont] of the handler as it is, with its
     arguments and its trap action if it has one
     (simplify/simplify_switch_expr.ml, [rebuild_arm]);
   - a call that returns to such a continuation is rebuilt to return to
     the one its handler applies, if the arguments of the handler are
     its parameters, in their order
     (simplify/continuation_shortcut.ml, [to_alias];
     simplify/expr_builder.ml, [apply_continuation_aliases], which
     [rewrite_fixed_arity_apply] takes the return continuation of a
     call through, where nothing has changed the parameters of that
     continuation).

   FlambdaShortcutSpec.shortcut_switches does this where the [Switch] is
   directly under the [Let_cont] and the continuation has no parameter.
   The front end leaves many more: the raise of a failed test, in every
   division and every checked access, is an arm to a continuation of no
   parameters bound some way above; the way out of a [try] is a
   continuation that passes its parameter on.

   [forward_applications_with] does it wherever the continuation is
   applied, with the calls or without them ([calls]).
   Going down the term it keeps, for the non-recursive continuations in
   scope whose handler, once done itself, is one [Apply_cont], the
   parameters and that [Apply_cont] ([forwards]). An application of such
   a continuation becomes the [Apply_cont] of its handler with the
   arguments put for the parameters ([forwarded]): an [Apply_cont], if
   that of the handler has no trap action, and it keeps its own if it
   has one ([forward_apply_cont_keeping], [forwarded_keeping]); an arm
   of a [Switch] in the same way, and a goto to a continuation of no
   parameters whatever the trap action of the handler ([forward_arm]).
   ([forward_apply_cont] is the case where neither has a trap action,
   which FlambdaForwardJumpsSpec.forward_jumps is made of.) Where the
   calls are taken, an [Apply]
   that returns to such a continuation returns to the one that its
   handler applies, if the [Apply_cont] of the handler has no trap
   action and its arguments are the parameters, in their order
   ([passes_on], [alias_of], [forward_apply]); the exception
   continuation of the call is left, no entry being kept for an
   exception handler. The [Let_cont] stays;
   FlambdaUnusedContsSpec.remove_unused_conts removes it if nothing
   names its continuation any more.

   [forward_applications] is the pass without the calls, which the
   rounds of the term before apply. The calls of the term before are
   taken on once, when its rounds are over, and only if the term
   after has no call left that the pass would take on
   (FlambdaValidateSpec.with_returns). A handler that ends a
   region before it passes its parameters on is not an alias for
   Simplify, which leaves the call returning to it; it is one here,
   the model having no regions; and the term after cannot be rid of
   such a continuation by an equivalence, since the handler's test of
   the kinds of the results would go with it. So where Simplify left
   such a call, the calls of the term before are left too. The term
   before is refined either way.

   What stops it ([forwarding]):
   - a continuation that the [Apply_cont] of the handler names, the one
     it applies or the one of its trap action, or a variable of its
     arguments that is not a parameter, is bound again somewhere in the
     body of the [Let_cont]: there it would not mean what it means in the
     handler. Names that the binders of a term do not reuse pass;
   - the [Apply_cont] of the handler names the continuation being bound,
     by applying it or in its trap action: in the handler that name is
     another continuation;
   - an exception handler, which a raise applies;
   - parameters that are not distinct;
   - an application that has not as many arguments as the continuation
     has parameters stays as it is;
   - recursive continuations are not forwarded;
   - an [Apply] returns where it did if the handler of its return
     continuation does anything to its parameters but pass them all on
     in their order (one dropped, two exchanged, a constant or another
     variable among the arguments): there the call would give its
     results to a continuation that expects others;
   - the bodies of code are not gone into
     (FlambdaNormaliseSpec.rewrite_code takes the pass there).

   Simplify leaves some applications that the pass forwards. Where
   every application of a continuation gives one of its parameters
   the same variable, bound in the body of the [Let_cont], Simplify's
   analysis of the aliases of parameters removes the parameter,
   passes the variable as a parameter of its own and binds the old
   name to it at the start of the handler
   (simplify/simplify_let_cont_expr.ml,
   [extra_params_for_continuation_param_aliases] and
   [add_lets_around_handler], which comes before the handler is
   looked at for a shortcut): the handler it has rebuilt is not a
   single [Apply_cont], and no shortcut is recorded. It is how a loop
   in a function leaves for the return continuation with what it has
   computed. And a handler that ends a region before it applies
   another continuation is not a single [Apply_cont] for Simplify; it
   is one here, the model having no regions. In the term after, once
   that [Let] has been followed and removed, such a continuation is
   left with a handler that only applies another, and its
   applications still go through it. The pass does not look for
   those cases. The validation forwards those applications of the
   term after too, by an equivalence
   (FlambdaForwardKindedSpec.forward_kinded), which takes no call on:
   hence the choice above for the calls.

   The result refines the term, whether the calls are taken or not
   ([forward_applications_with_refines_statement]; the pass of the
   rounds is [forward_applications_refines_statement]), under all
   environments. It is not an equivalence as it is stated: an
   application of the first term binds the parameters, so one with a
   parameter that the handler does not pass on is undefined where its
   argument has no value, and the second term does not look at that
   argument. Likewise a call of the first term gives its results to
   its return continuation, which tests their number and their kinds
   against its own parameters before its handler passes them on, and
   the continuation they are passed on to tests them again against
   its own; the call of the second term gives them to that one at
   once. Nothing is asked of the kinds that the two continuations give
   their parameters: where they differ the first term is undefined,
   and the second may not be. So the pass is for the term before
   Simplify. The machine does not execute trap actions, so that a goto
   takes that of the handler, or that an application keeps its own
   where it goes on, is nothing to it.

   Nothing is asked of the names of the term beyond what the pass checks
   itself. [binds_cont] and FlambdaParamAliasesSpec.binds_var may say yes
   when the answer is no, never the reverse.

   Renaming: the functions here compare names by equality only and bind
   none, so the pass commutes with a renaming: that is
   forward_applications_equivariant_statement of
   FlambdaValidatorEquivarianceSpec.v.

   The proofs are in FlambdaForwardProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaParamAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* Where an application goes                                                *)
(* ======================================================================== *)

(* Whether an expression may bind the continuation, by a [Let_cont]. *)
Fixpoint binds_cont (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => binds_cont k body
  | Let_cont_non_recursive k' h body =>
    Pos.eqb k k' || binds_cont_in_handler k h || binds_cont k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(k', h) => Pos.eqb k k' || binds_cont_in_handler k h)
      handlers
    || binds_cont k body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with binds_cont_in_handler (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => binds_cont k handler
  end.

(* A [Simple] with the arguments put for the parameters. *)
Definition subst_simple (params : list variable) (args : list simple)
    (s : simple) : simple :=
  match s with
  | Var x =>
    match lookup x (combine params args) with
    | Some arg => arg
    | None => s
    end
  | Symbol _ | Const _ => s
  end.

(* The [Apply_cont] of a handler, for an application of its continuation
   to the arguments given. *)
Definition forwarded (params : list variable) (args : list simple)
    (target : apply_cont) : apply_cont :=
  Apply_cont_expr (ac_continuation target)
    (map (subst_simple params args) (ac_args target))
    (ac_trap_action target).

(* For the continuations in scope, the most recently bound first: the
   parameters and the [Apply_cont] of the handler, for one whose
   applications may go where its handler goes; [None] for another. *)
Definition forwards : Type :=
  list (continuation * option (list variable * apply_cont)).

Definition forward_of (F : forwards) (k : continuation)
    : option (list variable * apply_cont) :=
  match lookup k F with
  | Some entry => entry
  | None => None
  end.

(* An [Apply_cont] made to go where the handler of its continuation
   goes, if neither has a trap action. *)
Definition forward_apply_cont (F : forwards) (ac : apply_cont) : apply_cont :=
  match forward_of F (ac_continuation ac) with
  | Some (params, target) =>
    if
      Nat.eqb (length (ac_args ac)) (length params)
      && no_trap_action ac && no_trap_action target
    then forwarded params (ac_args ac) target
    else ac
  | None => ac
  end.

(* The same with a trap action given, in place of that of the handler. *)
Definition forwarded_keeping (action : option trap_action)
    (params : list variable) (args : list simple) (target : apply_cont)
    : apply_cont :=
  Apply_cont_expr (ac_continuation target)
    (map (subst_simple params args) (ac_args target))
    action.

(* An [Apply_cont] made to go where the handler of its continuation
   goes, if the [Apply_cont] of the handler has no trap action. It
   keeps its own trap action, if it has one. *)
Definition forward_apply_cont_keeping (F : forwards) (ac : apply_cont)
    : apply_cont :=
  match forward_of F (ac_continuation ac) with
  | Some (params, target) =>
    if Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target
    then forwarded_keeping (ac_trap_action ac) params (ac_args ac) target
    else ac
  | None => ac
  end.

(* The same for an arm of a [Switch]; and a goto to a continuation of no
   parameters goes where the handler goes whatever the trap action of
   the handler. *)
Definition forward_arm (F : forwards) (ac : apply_cont) : apply_cont :=
  match forward_of F (ac_continuation ac), ac_args ac with
  | Some ([], target), [] =>
    if no_trap_action ac
    then forwarded [] [] target
    else forward_apply_cont_keeping F ac
  | _, _ => forward_apply_cont_keeping F ac
  end.

(* Whether the arguments are the parameters, in their order. *)
Fixpoint passes_on (params : list variable) (args : list simple) : bool :=
  match params, args with
  | [], [] => true
  | p :: params, Var x :: args => Pos.eqb x p && passes_on params args
  | _, _ => false
  end.

(* The continuation that a continuation is another name for: its
   handler passes its parameters on to that one as they are. *)
Definition alias_of (F : forwards) (k : continuation) : option continuation :=
  match forward_of F k with
  | Some (params, target) =>
    if no_trap_action target && passes_on params (ac_args target)
    then Some (ac_continuation target)
    else None
  | None => None
  end.

(* A call that returns to a continuation which is another name for one
   returns to that one. *)
Definition forward_apply (F : forwards) (ap : apply_expr) : apply_expr :=
  match alias_of F (ap_continuation ap) with
  | Some k' =>
    Apply_expr (ap_callee ap) (ap_call_kind ap) (ap_args ap) k'
      (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
      (ap_return_arity ap)
  | None => ap
  end.

(* Whether the applications of [k], whose handler has become [handler],
   may go where it goes: the parameters and the [Apply_cont] if so.
   [body] is the body of the [Let_cont] of [k]. *)
Definition forwarding (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr)
    : option (list variable * apply_cont) :=
  match handler with
  | Apply_cont target =>
    if
      hm_is_exn_handler metadata
      || among k (apply_cont_continuations target)
      || negb (distinct params)
      || existsb (fun j => binds_cont j body) (apply_cont_continuations target)
      || existsb
           (fun s =>
              match s with
              | Var x => negb (among x params) && binds_var x body
              | Symbol _ | Const _ => false
              end)
           (ac_args target)
    then None
    else Some (params, target)
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Apply _ | Switch _ _ | Invalid =>
    None
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* [calls]: whether a call that returns to a continuation which is
   another name for one is made to return to that one. *)
Fixpoint forward_applications_under (calls : bool) (F : forwards)
    (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (forward_applications_under calls F body)
  | Let_static bindings body =>
    Let_static bindings (forward_applications_under calls F body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler := forward_applications_under calls F handler in
    Let_cont_non_recursive k (Handler params metadata handler)
      (forward_applications_under calls
         ((k, forwarding k params metadata handler body) :: F) body)
  | Let_cont_recursive handlers body =>
    let F := map (fun '(k, _) => (k, None)) handlers ++ F in
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_applications_in_handler calls F h))
         handlers)
      (forward_applications_under calls F body)
  | Apply ap => Apply (if calls then forward_apply F ap else ap)
  | Invalid => e
  | Apply_cont ac => Apply_cont (forward_apply_cont_keeping F ac)
  | Switch scrutinee arms =>
    Switch scrutinee (map (fun arm => (fst arm, forward_arm F (snd arm))) arms)
  end
with forward_applications_in_handler (calls : bool) (F : forwards)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (forward_applications_under calls F handler)
  end.

Definition forward_applications_with (calls : bool) (e : expr) : expr :=
  forward_applications_under calls [] e.

(* The pass of the rounds: the calls are left. *)
Definition forward_applications (e : expr) : expr :=
  forward_applications_with false e.

Definition forward_applications_with_refines_statement : Prop :=
  ∀ (calls : bool) (e : expr),
    expr_refines e (forward_applications_with calls e).

Definition forward_applications_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (forward_applications e).
