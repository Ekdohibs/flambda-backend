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

(* An application that gives a number to the parameter that the handler
   of its continuation tests goes where the arm of that number goes: a
   normalisation that refines, and an equivalence where the arguments
   are known to have the kinds of the parameters.

   This file is to be reviewed: [forward_to_arms],
   [forward_to_arms_kinded] and [forward_to_arms_in_code] with what
   they are made of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the three statements.
   The validation applies [forward_to_arms] to the term before
   (FlambdaValidateSpec.first_phase_round) and the other two to the
   term after (FlambdaValidateSpec.inline_once_applied_after,
   FlambdaValidateSpec.normalise_after_round).

   Where the handler of a non-recursive continuation [k] is a
   [Switch] on one of its parameters, an application of [k] that
   gives that parameter a naked immediate can only come to the arm of
   that number. In

     let_cont k (p, q) =
       switch p
         | 0 -> cont k' (q)
         | 1 -> cont j (7)
     in
     ... cont k (0, a) ... cont k (1, b) ...

   the two applications become [cont k' (a)] and [cont j (7)]: the
   [Apply_cont] of the arm with the arguments put for the parameters
   (FlambdaForwardSpec.forwarded). The [Let_cont] stays;
   FlambdaUnusedContsSpec.remove_unused_conts removes it if nothing
   names [k] any more.

   The front end leaves the shape where a variant is matched after
   the branches that build it have met, once the parameters of [k]
   are unboxed, by Simplify and by the validation, and the
   continuations that its handler binds are bound outside it
   (FlambdaLiftContsSpec.v, whose header shows the shape before
   that). The applications of [k] then give different numbers, each
   of which would decide the [Switch]. Simplify has a path for that
   shape, the specialisation of the continuation, enabled by
   [-flambda2-match-in-match], which no level of optimisation sets
   (driver/oxcaml_flags.ml, [Default.match_in_match]): where its
   other tests pass, it moves the continuations out of the handler
   and gives each application a handler of its own, whose [Switch]
   it decides by what the application gives
   (simplify/simplify_switch_expr.ml,
   [decide_continuation_specialization];
   simplify/simplify_let_cont_expr.ml,
   [specialize_continuation_if_needed]). The path is not needed for
   every such [Switch]: where one application alone comes to [k] and
   gives a known number, or all the applications give the same known
   number, the [Switch] can be decided without specialising. In the
   units that the pass was written for, none of them compiled with
   that flag, the term after keeps [k] and its [Switch], as the term
   before does. But Simplify knows in the handler what the
   applications give (simplify/join_points.ml,
   [compute_handler_env]), and the handler that it leaves is not
   always that of the term before: an arm whose number no
   application gives may read a constant where the term before
   reads a parameter. The validation has no rewrite that makes
   two such handlers the same. It makes the applications of the two
   terms go to their arms instead: where nothing then applies [k] in
   either term, the two handlers are removed.

   Going down the term the pass keeps, for each non-recursive
   continuation in scope whose handler, once done itself, is a
   [Switch] on one of its parameters, the parameters, the metadata,
   the position of the parameter tested, the arms, and the body of
   the [Let_cont] ([arm_entry], [arm_entry_of]). An [Apply_cont] and
   an arm of a [Switch] are treated alike
   ([forward_to_arm_apply_cont]): where the continuation has an entry
   and the argument at the position kept is a constant that is a
   naked immediate, the arm of that number is looked up
   (FlambdaSemanticsSpec.find_arm, as the machine does), and the
   application goes where that arm goes if the arm passes the tests
   of FlambdaForwardSpec.forwarding, those of a handler that would be
   that arm alone ([arm_of_apply_cont]).

   What stops it:
   - what stops [forwarding], for the arm: the arm names a
     continuation, or reads a variable that is not a parameter, that
     is bound again somewhere in the body of the [Let_cont]; it names
     [k]; [k] is an exception handler; its parameters are not
     distinct. The tests are made of each arm when an application
     comes to it, so an arm that fails them stops only the
     applications that give its number;
   - a handler that binds something on its way to its [Switch], or
     tests a variable that is not one of its parameters: no entry is
     kept;
   - an argument at the position of the tested parameter that is not
     a constant, or a number that no arm has;
   - an application that has not as many arguments as [k] has
     parameters;
   - a trap action on the application or on the arm;
   - recursive continuations have no entry, and an [Apply] that
     returns to a continuation still does;
   - the bodies of code are not gone into by [forward_to_arms] and
     [forward_to_arms_kinded] (FlambdaNormaliseSpec.rewrite_code
     takes a pass there; [forward_to_arms_in_code] is for the kinded
     one).

   The pass is taken two ways ([kinded]).

   [forward_to_arms] asks nothing of the kinds. Its result refines
   the term ([forward_to_arms_refines_statement]), under all
   environments, and is not equivalent to it as the statement is
   made: an application of the first term evaluates its arguments,
   tests their kinds against those that the handler gives its
   parameters and binds them
   (FlambdaSemanticsSpec.apply_cont_value), and the [Apply_cont] put
   in its place does none of that, so the first term is undefined
   where an argument that the arm does not read has no value or not
   the kind of its parameter, and the second need not be. It is for
   the term before Simplify.

   [forward_to_arms_kinded] is for the term after, where an
   equivalence is needed
   ([forward_to_arms_kinded_equivalent_statement]). It forwards an
   application only where the three things that the application does
   cannot fail, as FlambdaForwardKindedSpec.forward_kinded does and
   by the same means: going down the term it keeps the kinds of the
   variables bound around the point
   (FlambdaNormaliseSpec.kinds_in_scope), and asks that each
   argument have, by them, the kind that the handler gives the
   parameter it is given to (FlambdaForwardKindedSpec.have_kinds);
   and it keeps no entry for a handler that does not give as many
   kinds as it has parameters, which never runs. The [Switch] of the
   handler cannot fail either: the parameter tested is bound to the
   naked immediate given, and the arm of that number has been found.

   [forward_to_arms_in_code] applies the kinded pass to the body of
   every piece of code that a term binds, started with the kinds of
   the variables that a call binds
   (FlambdaCodeLetsSpec.code_in_scope), as
   FlambdaForwardKindedSpec.forward_kinded_in_code does; its result
   is equivalent to the term
   ([forward_to_arms_in_code_equivalent_statement]).

   The machine does not execute trap actions, so the statements say
   nothing of them.

   The tests of [forwarding] are on the syntax and may say that a
   name is bound again when it is not, never the reverse; an
   application is then left that could have gone to its arm.

   Renaming: the functions here compare names by equality only and
   bind none, so the passes commute with a renaming: that is
   forward_to_arms_equivariant_statement,
   forward_to_arms_kinded_equivariant_statement and
   forward_to_arms_in_code_equivariant_statement of
   FlambdaValidatorEquivarianceSpec.v. [forward_to_arms] keeps
   scope: forward_to_arms_keeps_scope_statement of
   FlambdaLoopsScopeSpec.v.

   The proofs are in FlambdaForwardArmProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaCodeLetsSpec FlambdaForwardSpec FlambdaForwardKindedSpec.
Import ListNotations.

(* ======================================================================== *)
(* Where an application goes                                                *)
(* ======================================================================== *)

(* The position of a variable among the parameters. *)
Fixpoint position_of (x : variable) (params : list variable) : option nat :=
  match params with
  | [] => None
  | y :: params =>
    if Pos.eqb x y then Some O
    else
      match position_of x params with
      | Some i => Some (S i)
      | None => None
      end
  end.

(* What is kept of a continuation whose handler is a [Switch] on one of
   its parameters: the parameters, the metadata, the position of the
   parameter tested, the arms, and the body of the [Let_cont]. *)
Definition arm_entry : Type :=
  list variable * handler_metadata * nat * list (Z * apply_cont) * expr.

(* For the continuations in scope, the most recently bound first: what
   is kept of one whose applications may go to an arm of its handler;
   [None] for another. *)
Definition arm_forwards : Type := list (continuation * option arm_entry).

(* The entry of a continuation with these parameters. [handler] is the
   handler once the pass has been through it, and [body] the body of
   the [Let_cont]. Where [kinded], none is kept for a handler that does
   not give as many kinds as it has parameters. *)

Definition arm_entry_of (kinded : bool) (params : list variable)
    (metadata : handler_metadata) (handler body : expr) : option arm_entry :=
  match handler with
  | Switch (Var p) arms =>
    if kinded
       && negb (Nat.eqb (length params) (length (hm_params_kinds metadata)))
    then None
    else
      match position_of p params with
      | Some i => Some (params, metadata, i, arms, body)
      | None => None
      end
  | _ => None
  end.

(* The arm that an application goes to, with the parameters of its
   continuation and their kinds: the continuation has an entry, the
   argument at the position of the parameter tested is a naked
   immediate, an arm has that number, and the arm passes the tests of
   FlambdaForwardSpec.forwarding, those of a handler that would be that
   arm alone. *)
Definition arm_of_apply_cont (F : arm_forwards) (ac : apply_cont)
    : option (list variable * list flambda_kind * apply_cont) :=
  match lookup (ac_continuation ac) F with
  | Some (Some (params, metadata, i, arms, body)) =>
    match nth_error (ac_args ac) i with
    | Some (Const (Naked_immediate n)) =>
      match find_arm n arms with
      | Some target =>
        match
          forwarding (ac_continuation ac) params metadata (Apply_cont target)
            body
        with
        | Some (params, target) =>
          Some (params, hm_params_kinds metadata, target)
        | None => None
        end
      | None => None
      end
    | _ => None
    end
  | _ => None
  end.

(* An [Apply_cont] made to go where the arm goes, if it has as many
   arguments as the continuation has parameters, neither it nor the arm
   has a trap action, and, where [kinded], the arguments have the kinds
   of the parameters. *)
Definition forward_to_arm_apply_cont (kinded : bool) (G : kinds_in_scope)
    (F : arm_forwards) (ac : apply_cont) : apply_cont :=
  match arm_of_apply_cont F ac with
  | Some (params, kinds, target) =>
    if
      Nat.eqb (length (ac_args ac)) (length params)
      && no_trap_action ac && no_trap_action target
      && (negb kinded || have_kinds G (ac_args ac) kinds)
    then forwarded params (ac_args ac) target
    else ac
  | None => ac
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* [G]: the kinds of the variables bound around the point, which only
   the kinded pass reads. *)
Fixpoint forward_to_arms_under (kinded : bool) (G : kinds_in_scope)
    (F : arm_forwards) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (forward_to_arms_under kinded ((x, kind) :: G) F body)
  | Let_static bindings body =>
    Let_static bindings (forward_to_arms_under kinded G F body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler :=
      forward_to_arms_under kinded
        (params_in_scope params (hm_params_kinds metadata) G) F handler
    in
    Let_cont_non_recursive k (Handler params metadata handler)
      (forward_to_arms_under kinded G
         ((k, arm_entry_of kinded params metadata handler body) :: F) body)
  | Let_cont_recursive handlers body =>
    let F := map (fun '(k, _) => (k, None)) handlers ++ F in
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_to_arms_in_handler kinded G F h))
         handlers)
      (forward_to_arms_under kinded G F body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (forward_to_arm_apply_cont kinded G F ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map
         (fun arm => (fst arm, forward_to_arm_apply_cont kinded G F (snd arm)))
         arms)
  end
with forward_to_arms_in_handler (kinded : bool) (G : kinds_in_scope)
    (F : arm_forwards) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (forward_to_arms_under kinded
         (params_in_scope params (hm_params_kinds metadata) G) F handler)
  end.

(* For the term before: nothing is asked of the kinds. *)
Definition forward_to_arms (e : expr) : expr :=
  forward_to_arms_under false [] [] e.

(* For the term after: the arguments are known to have the kinds of the
   parameters. *)
Definition forward_to_arms_kinded (e : expr) : expr :=
  forward_to_arms_under true [] [] e.

Definition forward_to_arms_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (forward_to_arms e).

Definition forward_to_arms_kinded_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (forward_to_arms_kinded e).

(* ======================================================================== *)
(* The pass in the bodies of code                                           *)
(* ======================================================================== *)

(* The kinded pass in the body of a piece of code, with the kinds of
   what a call binds (FlambdaCodeLetsSpec.code_in_scope). *)
Definition forward_to_arms_in_binding (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata
         (forward_to_arms_under true (code_in_scope params my_closure metadata)
            [] body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

Fixpoint forward_to_arms_in_code (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (forward_to_arms_in_code body)
  | Let_static bindings body =>
    Let_static (map forward_to_arms_in_binding bindings)
      (forward_to_arms_in_code body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (forward_to_arms_in_code_handler h)
      (forward_to_arms_in_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, forward_to_arms_in_code_handler h)) handlers)
      (forward_to_arms_in_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with forward_to_arms_in_code_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (forward_to_arms_in_code handler)
  end.

Definition forward_to_arms_in_code_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (forward_to_arms_in_code e).
