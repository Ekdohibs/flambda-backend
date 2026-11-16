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
   applies that other, where its arguments have the kinds of the
   parameters: a normalisation that is an equivalence.

   This file is to be reviewed: [forward_kinded] and
   [forward_kinded_in_code] with what they are made of, which are
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the two statements.

   FlambdaForwardSpec.forward_applications makes an application of a
   non-recursive continuation whose handler is one [Apply_cont] that
   [Apply_cont], with the arguments put for the parameters. It refines
   and is not an equivalence: an application evaluates its arguments,
   tests their kinds against those that the handler gives its
   parameters and binds them (FlambdaSemanticsSpec.apply_cont_value),
   and what is put in its place does none of that. So it is for the
   term before Simplify. FlambdaForwardJumpsSpec.forward_jumps is the
   case of a continuation of no parameters, where there is nothing to
   evaluate, to test or to bind: an equivalence.

   The term after needs the same for a continuation with parameters.
   Simplify leaves applications that go through a continuation whose
   handler, in the model, only applies another, in two cases more
   (the header of FlambdaForwardSpec.v says where it records
   shortcuts).

   Every application of the continuation gives one of its parameters
   the same variable, bound in the body of the [Let_cont]. Simplify
   passes that variable as a parameter of its own and binds the old
   name to it at the head of the handler, which is then not one
   [Apply_cont] when Simplify looks for a shortcut. It is how a loop
   in a function leaves for the return continuation with what it has
   computed. Once the validation has followed that [Let] and removed
   it, the handler is one [Apply_cont].

   Or the handler ends a region. A function whose body has a [try]
   has a region, which the continuation that follows the [try] ends
   before it passes its parameter on:

     let attempt n =
       match (try Some (check n) with _ -> None) with
       | Some v -> v
       | None -> -7

   Simplify keeps the region, which the [try] uses, so the handler of
   that continuation is the two primitives that end the region and
   then an [Apply_cont]. The model has no regions: the conversion
   drops those primitives, and the handler is one [Apply_cont] that
   passes the parameter on, in both terms. Where the function returns
   what the [try] gives, the body of the [try] applies that
   continuation with the trap action that pops the handler:

     let sum_or_zero g l =
       try fold_left (fun acc x -> acc + g x) 0 l with _ -> 0

   (the function has a region, in which the closure given to
   [fold_left] is allocated, and the continuation that follows the
   [try] ends it).

   In both cases the term before has its applications forwarded:
   FlambdaForwardSpec.forward_applications does not look for what
   Simplify left. The term after must have them forwarded too, by an
   equivalence.

   [forward_kinded] forwards an application where the three things
   that the application does cannot fail and cannot be seen:
   - every argument has a value. An argument is a constant, a symbol,
     or a variable that a [Let] or the parameters of a handler bind
     around the application, inside the term;
   - the values have the kinds of the parameters. The machine tests
     the kind of what a [Let] binds against the kind of the [Let], and
     the kinds of what a handler is applied to against the kinds of
     its parameters, and has no step otherwise
     (FlambdaSemanticsSpec.step, [apply_cont_value]); so a variable
     bound by one of them, where it is read under that binder, holds a
     value of the kind that the binder gives. Going down the term the
     pass keeps those kinds (FlambdaNormaliseSpec.kinds_in_scope:
     the most recently bound first, as the variables of the machine
     are; [params_in_scope] for the parameters of a handler), and
     asks that each argument have, by them, the kind that the handler
     gives the parameter it is given to ([have_kinds], with
     FlambdaNormaliseSpec.kind_of_simple_in_scope). A constant has
     its own kind and a symbol the kind [Value];
   - the binding succeeds: the application has as many arguments as
     the handler has parameters, and the handler gives as many kinds
     ([kinded_entry]).
   The application then runs the [Apply_cont] of the handler under the
   variables of the [Let_cont] with the parameters bound, and the
   [Apply_cont] put in its place is that one with the arguments put
   for the parameters, whose names mean the same where the
   application is (FlambdaForwardSpec.forwarding has checked it), and
   with the trap action of the application, if it has one
   (FlambdaForwardSpec.forwarded_keeping). The machine does not
   execute trap actions, so the one that is kept changes no run. It
   is kept for the comparison of the two terms, which tells a trap
   action from none: the term before keeps it
   (FlambdaForwardSpec.forward_apply_cont_keeping), as Simplify does.

   What the pass keeps, going down a term: the kinds of the variables
   bound around the point ([G]); for each continuation bound around
   it, the kinds that its handler gives its parameters ([K]); and the
   entries of FlambdaForwardSpec.forward_applications ([F]): for a
   non-recursive continuation whose handler, once done itself, is one
   [Apply_cont], the parameters and that [Apply_cont]. An
   [Apply_cont] and an arm of a [Switch] are treated alike
   ([forward_kinded_apply_cont]), but for one case of an arm
   ([forward_kinded_arm]), which comes next. The [Let_cont] stays;
   FlambdaUnusedContsSpec.remove_unused_conts removes it if nothing
   names its continuation any more.

   An arm of a [Switch] that applies, to no arguments and with no
   trap action of its own, a continuation that has no parameters and
   gives no kinds, and whose handler is one [Apply_cont], becomes
   that [Apply_cont] whatever its trap action, which it keeps: an
   arm may be an [Apply_cont] with a trap action. The other tests of
   FlambdaForwardSpec.forwarding are asked as for every entry (the
   continuation is no exception handler, and what stops the pass is
   listed below). FlambdaForwardSpec.forward_arm does the same
   for the term before, as Simplify does (the header of that file
   says where). The term after needs it where the validation has
   made such a continuation itself. An application that goes to the
   arm of the number that it gives (FlambdaForwardArmSpec.v) may
   leave the other applications of its continuation giving one
   constant; the parameter is then bound at the head of the handler
   (FlambdaBindConstParamsSpec.v) and the [Switch] on it folded, and
   where the arm that is left raises, the handler is one
   [Apply_cont] with the trap action of the raise, of a continuation
   of no parameters that the arms of the tests before it apply. The
   term before has the raise in those arms.

   Why that case is an equivalence: the continuation has no
   parameters and gives no kinds, so its application to no arguments
   is one that the machine accepts (a handler applied to values that
   are not of the kinds that it gives is undefined,
   FlambdaSemanticsSpec.apply_cont_value; here there are neither
   values nor kinds), and there is nothing to evaluate or to bind,
   as for FlambdaForwardJumpsSpec.forward_jumps. A continuation of
   no parameters that gave a kind is not taken: its application
   would be undefined where the [Apply_cont] of its handler need not
   be. And the trap action that
   is then where the arm was changes no run, because the machine
   does not execute trap actions (FlambdaSemanticsSpec.step reads
   the continuation and the arguments of an [Apply_cont], and
   nothing else of it). So the statement is of the machine as it is.
   For a machine that executes trap actions, that an arm which goes
   on to a handler with a trap action runs as the arm with that trap
   action does is not stated here, and is to be looked at again when
   the statements of the validation are made of such a machine.

   What stops it:
   - what stops FlambdaForwardSpec.forwarding: the [Apply_cont] of the
     handler names a continuation, or reads a variable, that is bound
     again somewhere in the body of the [Let_cont]; it names the
     continuation being bound; the continuation is an exception
     handler; its parameters are not distinct;
   - a handler that does not give as many kinds as it has parameters:
     it never runs, so an application of it is undefined, and the
     [Apply_cont] put in its place might not be;
   - an argument whose kind the pass does not know: a variable free
     in the term. Nothing is assumed of the environments under which
     the term runs. The parameters of code are free in its body;
     [forward_kinded_in_code] is for them;
   - an argument whose kind is not that of the parameter: the
     application is undefined;
   - a trap action on the [Apply_cont] of the handler, but for the
     arm above;
   - recursive continuations are not forwarded, and an [Apply] that
     returns to a continuation still does: the results of a call are
     tested where they are returned, and their kinds are not known
     where the call is;
   - the bodies of code are not gone into
     (FlambdaNormaliseSpec.rewrite_code takes the pass there, with
     nothing known of the parameters of the code).

   With no parameters, and no trap action on the application, this
   is FlambdaForwardJumpsSpec.forward_jumps: a handler of no
   parameters that gives no kinds, applied to no arguments. That pass
   leaves an application that has a trap action.

   [forward_kinded_in_code] applies the pass to the body of every
   piece of code that a term binds, wherever the binding is, as
   FlambdaNormaliseSpec.rewrite_code does, started with the kinds of
   the variables that a call binds
   (FlambdaCodeLetsSpec.code_in_scope, whose header says why those
   kinds): [my_closure] at the kind [Value], and the parameters at
   the kinds of the arity of the code, unarized. So an application in
   the body of a function that gives a continuation a parameter of
   the function goes on:

     let or_itself x =
       match (try Some (check x) with _ -> None) with
       | Some v -> v + 1
       | None -> x

   Nothing is kept for the return continuation and the exception
   continuation of the code, which no [Let_cont] binds: an
   application may be forwarded to them, and none of them is
   forwarded. The body of code that binds something static does not
   run at all; the pass does not change whether a body does.

   The tests of [forwarding] are on the syntax and may say that a name
   is bound again when it is not, never the reverse; an application is
   then left that could have gone on.

   Renaming: the functions here compare names by equality only and
   bind none, so the two passes commute with a renaming: that is
   forward_kinded_equivariant_statement and
   forward_kinded_in_code_equivariant_statement of
   FlambdaValidatorEquivarianceSpec.v.

   The proofs are in FlambdaForwardKindedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaCodeLetsSpec FlambdaForwardSpec.
Import ListNotations.

(* ======================================================================== *)
(* The kinds of what is bound around a point                                *)
(* ======================================================================== *)

(* The kinds that the handlers of the continuations bound around a
   point give their parameters. *)
Definition cont_kinds : Type := list (continuation * list flambda_kind).

(* Whether the arguments are known to have these kinds, one each. *)
Fixpoint have_kinds (G : kinds_in_scope) (args : list simple)
    (kinds : list flambda_kind) : bool :=
  match args, kinds with
  | [], [] => true
  | s :: args, kind :: kinds =>
    match kind_of_simple_in_scope G s with
    | Some kind' => kind_eqb kind' kind && have_kinds G args kinds
    | None => false
    end
  | _, _ => false
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The entry of [k] among the continuations in scope: that of
   FlambdaForwardSpec.forwarding, for a handler that gives as many
   kinds as it has parameters. [handler] is the handler once the pass
   has been through it, and [body] the body of the [Let_cont] of [k]. *)
Definition kinded_entry (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr)
    : option (list variable * apply_cont) :=
  if Nat.eqb (length params) (length (hm_params_kinds metadata))
  then forwarding k params metadata handler body
  else None.

(* An [Apply_cont] made to go where the handler of its continuation
   goes, if the [Apply_cont] of the handler has no trap action and its
   arguments have the kinds of the parameters. It keeps its own trap
   action, if it has one. *)
Definition forward_kinded_apply_cont (G : kinds_in_scope) (K : cont_kinds)
    (F : forwards) (ac : apply_cont) : apply_cont :=
  match forward_of F (ac_continuation ac), lookup (ac_continuation ac) K with
  | Some (params, target), Some kinds =>
    if
      Nat.eqb (length (ac_args ac)) (length params)
      && no_trap_action target
      && have_kinds G (ac_args ac) kinds
    then forwarded_keeping (ac_trap_action ac) params (ac_args ac) target
    else ac
  | _, _ => ac
  end.

(* The same for an arm of a [Switch], with one case more: an arm that
   applies, to no arguments and with no trap action, a continuation of
   no parameters whose handler gives no kinds goes where the handler
   goes whatever the trap action of the handler, as
   FlambdaForwardSpec.forward_arm has it. *)
Definition forward_kinded_arm (G : kinds_in_scope) (K : cont_kinds)
    (F : forwards) (ac : apply_cont) : apply_cont :=
  match
    forward_of F (ac_continuation ac), lookup (ac_continuation ac) K,
    ac_args ac
  with
  | Some ([], target), Some [], [] =>
    if no_trap_action ac then forwarded [] [] target
    else forward_kinded_apply_cont G K F ac
  | _, _, _ => forward_kinded_apply_cont G K F ac
  end.

Fixpoint forward_kinded_under (G : kinds_in_scope) (K : cont_kinds)
    (F : forwards) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (forward_kinded_under ((x, kind) :: G) K F body)
  | Let_static bindings body =>
    Let_static bindings (forward_kinded_under G K F body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let kinds := hm_params_kinds metadata in
    let handler :=
      forward_kinded_under (params_in_scope params kinds G) K F handler
    in
    Let_cont_non_recursive k (Handler params metadata handler)
      (forward_kinded_under G ((k, kinds) :: K)
         ((k, kinded_entry k params metadata handler body) :: F) body)
  | Let_cont_recursive handlers body =>
    let F := map (fun '(k, _) => (k, None)) handlers ++ F in
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_kinded_in_handler G K F h)) handlers)
      (forward_kinded_under G K F body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (forward_kinded_apply_cont G K F ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun arm => (fst arm, forward_kinded_arm G K F (snd arm))) arms)
  end
with forward_kinded_in_handler (G : kinds_in_scope) (K : cont_kinds)
    (F : forwards) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (forward_kinded_under
         (params_in_scope params (hm_params_kinds metadata) G) K F handler)
  end.

Definition forward_kinded (e : expr) : expr :=
  forward_kinded_under [] [] [] e.

Definition forward_kinded_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (forward_kinded e).

(* ======================================================================== *)
(* The pass in the bodies of code                                           *)
(* ======================================================================== *)

(* The pass in the body of a piece of code, with the kinds of what a
   call binds (FlambdaCodeLetsSpec.code_in_scope). *)
Definition forward_kinded_in_binding (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata
         (forward_kinded_under (code_in_scope params my_closure metadata) []
            [] body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

Fixpoint forward_kinded_in_code (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (forward_kinded_in_code body)
  | Let_static bindings body =>
    Let_static (map forward_kinded_in_binding bindings)
      (forward_kinded_in_code body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (forward_kinded_in_code_handler h)
      (forward_kinded_in_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_kinded_in_code_handler h)) handlers)
      (forward_kinded_in_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with forward_kinded_in_code_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (forward_kinded_in_code handler)
  end.

Definition forward_kinded_in_code_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (forward_kinded_in_code e).
