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


(* A parameter of a continuation that every application gives the
   number of the arm that the application is in, of a [Switch] on one
   variable, is given that variable: a normalisation that is an
   equivalence.

   This file is to be reviewed: [give_scrutinees] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the two statements. The
   validation applies the pass to each of the two terms that it
   compares, once it has placed their continuations, just before
   FlambdaUnaliasParamsSpec.unalias_params
   (FlambdaValidateSpec.tidied_before, [tidied_after]).

   Simplify gives a continuation a parameter for a primitive that has
   been computed on every path to it
   (simplify/common_subexpression_elimination.ml, [join];
   FlambdaUnaliasParamsSpec.v says more), and each application has
   for it what holds the result on its path. What holds it is taken
   in its canonical form in the environment of the application
   ([cse_with_eligible_lhs], which asks
   [Typing_env.get_canonical_simple_exn]); on a path that goes
   through an arm of a [Switch] on the result, that is the number of
   the arm, which Simplify knows the scrutinee to be there. The paths
   then differ, and the parameter is made ([join_one_cse_equation]).
   So for a loop that tests a flag
   kept in a reference before it tests something else,

     while (not !found) && !i < n do ... done

   the continuation that the two arms of the test of the flag come
   to has a parameter for the flag, which one arm gives [0] and the
   other gives [1], and its handler passes that parameter on where
   the term before Simplify passes the flag. An arm that has more to
   compute before it applies the continuation goes through a
   continuation of its own, bound just above the [Switch]; the
   application is then in the handler of that one.

   [give_scrutinee x k i] puts the variable [x] for the argument at
   position [i] of an application of [k] where that argument is the
   constant that [x] is known to be there. What is known ([known], a
   number or nothing) is found on the way down:
   - in an arm of a [Switch] on [x], [x] is the number of the arm;
   - in the handler of a non-recursive continuation whose [Let_cont]
     has for its body a [Switch] on [x], and that only arms of one
     number apply, [x] is that number ([arm_fact]): the handler runs
     when such an arm is taken, under the variables that the [Switch]
     was run under, and its parameters. The same where the body comes
     to the [Switch] through other non-recursive [Let_cont]s, each
     arm that has more to compute having a continuation of its own,
     if none of them binds the name again and none of their handlers
     mentions it: nothing but the [Switch] applies the continuation
     then, and no variable is bound between its [Let_cont] and the
     [Switch];
   - what is known where a continuation is defined is known in its
     handler, which runs under the variables of its definition and
     its parameters;
   - nothing is known of [x] under a [Let] of [x] or in a handler
     that has [x] among its parameters.
   The bindings of a [Let_static], and so the bodies of code, are
   left as they are. The constant is a naked immediate, as the
   scrutinee of a [Switch] is.

   The rewrite is an equivalence whatever [x], [k] and [i] are
   ([give_scrutinee_equivalent_statement]): an argument is changed
   only where the variable holds the constant. The pass
   ([give_scrutinees]) chooses them. For each non-recursive
   [Let_cont], and each position of its handler from the last to the
   first, it takes the variables to try from the first application of
   the continuation in the body ([first_scrutinee]: the variable that
   it gives, or the scrutinees of the [Switch]es that it is under,
   the nearest first, if it gives a naked immediate), makes the
   rewrite for each in turn ([give_first]), and keeps the first
   result for which FlambdaUnaliasParamsSpec.unalias_param would then
   take the parameter away: every application now gives that one
   variable, and the other conditions of that pass hold. If there is
   none the body is left as it was. That test is for the use of the
   result, and neither statement depends on it. The nearest [Switch]
   is not always the one: where an arm applies a continuation whose
   handler tests something else and applies the continuation in the
   arms of that test, the number given is that of the arm of the
   first [Switch].

   The handlers of recursive continuations and the bodies of code are
   not gone into by the pass: FlambdaNormaliseSpec.rewrite_code and
   FlambdaRecHandlersSpec.rewrite_rec_handlers take a pass there.

   The result is equivalent to the term
   ([give_scrutinees_equivalent_statement]), under all environments.

   Renaming: the functions here compare names by equality only and
   make none.

   The proofs are in FlambdaGiveScrutineeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaUnusedContsSpec
  FlambdaUnaliasParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known of the variable                                            *)
(* ======================================================================== *)

(* The number of the first arm that applies [k]. *)
Definition first_arm_of (k : continuation) (arms : list (Z * apply_cont))
    : option Z :=
  first_answer
    (fun '(d, ac) => if Pos.eqb (ac_continuation ac) k then Some d else None)
    arms.

(* Every arm that applies [k] has the number [c]. *)
Definition only_arm (k : continuation) (c : Z)
    (arms : list (Z * apply_cont)) : bool :=
  forallb
    (fun '(d, ac) => negb (Pos.eqb (ac_continuation ac) k) || Z.eqb d c)
    arms.

(* The number that [x] is whenever the handler of [k] runs, where the
   body of the [Let_cont] of [k] is a [Switch] on [x], or comes to one
   through non-recursive [Let_cont]s that do not bind [k] again and
   whose handlers do not mention it, and only arms of one number apply
   [k]. *)
Fixpoint arm_fact (x : variable) (k : continuation) (body : expr)
    {struct body} : option Z :=
  match body with
  | Switch (Var y) arms =>
    if Pos.eqb y x
    then
      match first_arm_of k arms with
      | Some c => if only_arm k c arms then Some c else None
      | None => None
      end
    else None
  | Let_cont_non_recursive k' h body =>
    if Pos.eqb k' k || cont_free_in_handler_b k h then None
    else arm_fact x k body
  | _ => None
  end.

(* What is known in an arm of a [Switch]. *)
Definition arm_known (x : variable) (scrutinee : simple) (d : Z)
    (known : option Z) : option Z :=
  match scrutinee with
  | Var y => if Pos.eqb y x then Some d else known
  | _ => known
  end.

(* What is known in a handler of what was known where it is defined. *)
Definition known_under (x : variable) (params : list variable)
    (known : option Z) : option Z :=
  if among x params then None else known.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The arguments with [x] at position [i], if the argument there is the
   naked immediate [c]. *)
Fixpoint give_nth (x : variable) (c : Z) (i : nat) (args : list simple)
    : list simple :=
  match i, args with
  | O, Const (Naked_immediate c') :: rest =>
    if Z.eqb c' c then Var x :: rest else args
  | S i, a :: rest => a :: give_nth x c i rest
  | _, _ => args
  end.

Definition give_in_apply_cont (x : variable) (k : continuation) (i : nat)
    (known : option Z) (ac : apply_cont) : apply_cont :=
  match known with
  | Some c =>
    if Pos.eqb (ac_continuation ac) k
    then
      {| ac_continuation := ac_continuation ac;
         ac_args := give_nth x c i (ac_args ac);
         ac_trap_action := ac_trap_action ac |}
    else ac
  | None => ac
  end.

Fixpoint give_scrutinee (x : variable) (k : continuation) (i : nat)
    (known : option Z) (e : expr) {struct e} : expr :=
  match e with
  | Let y kind d body =>
    Let y kind d
      (give_scrutinee x k i (if Pos.eqb y x then None else known) body)
  | Let_static bindings body =>
    Let_static bindings (give_scrutinee x k i known body)
  | Let_cont_non_recursive k' h body =>
    Let_cont_non_recursive k'
      (give_in_handler x k i
         match arm_fact x k' body with
         | Some c => Some c
         | None => known
         end
         h)
      (give_scrutinee x k i known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k', h) => (k', give_in_handler x k i known h)) handlers)
      (give_scrutinee x k i known body)
  | Apply_cont ac => Apply_cont (give_in_apply_cont x k i known ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map
         (fun '(d, ac) =>
            (d, give_in_apply_cont x k i (arm_known x scrutinee d known) ac))
         arms)
  | Apply _ | Invalid => e
  end
with give_in_handler (x : variable) (k : continuation) (i : nat)
    (known : option Z) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (give_scrutinee x k i (known_under x params known) handler)
  end.

(* The result behaves as the term does, under all environments,
   whatever the variable, the continuation and the position are. *)
Definition give_scrutinee_equivalent_statement : Prop :=
  ∀ (x : variable) (k : continuation) (i : nat) (e : expr),
    expr_equivalent e (give_scrutinee x k i None e).

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* What an application of [k] says of the variables to try: the
   variable that it gives, or [current], the scrutinees of the
   [Switch]es that it is under, the nearest first, if it gives a naked
   immediate. *)
Definition scrutinee_by (k : continuation) (i : nat)
    (current : list variable) (ac : apply_cont)
    : option (list variable) :=
  if Pos.eqb (ac_continuation ac) k
  then
    Some
      match nth_error (ac_args ac) i with
      | Some (Var y) => [y]
      | Some (Const (Naked_immediate _)) => current
      | _ => []
      end
  else None.

(* A scrutinee that is a variable, before those that there were; and
   that of an expression that is a [Switch], or comes to one through
   non-recursive [Let_cont]s. *)
Definition scrutinee_var (current : list variable) (scrutinee : simple)
    : list variable :=
  match scrutinee with
  | Var y => y :: current
  | _ => current
  end.

Fixpoint scrutinee_of (current : list variable) (e : expr) {struct e}
    : list variable :=
  match e with
  | Switch scrutinee _ => scrutinee_var current scrutinee
  | Let_cont_non_recursive _ _ body => scrutinee_of current body
  | _ => current
  end.

(* What the first application of [k] says. *)
Fixpoint first_scrutinee (k : continuation) (i : nat)
    (current : list variable) (e : expr) {struct e}
    : option (list variable) :=
  match e with
  | Let _ _ _ body => first_scrutinee k i current body
  | Let_static _ body => first_scrutinee k i current body
  | Let_cont_non_recursive _ h body =>
    match handler_first_scrutinee k i (scrutinee_of current body) h with
    | Some r => Some r
    | None => first_scrutinee k i current body
    end
  | Let_cont_recursive handlers body =>
    match
      first_answer (fun '(_, h) => handler_first_scrutinee k i current h)
        handlers
    with
    | Some r => Some r
    | None => first_scrutinee k i current body
    end
  | Apply_cont ac => scrutinee_by k i current ac
  | Switch scrutinee arms =>
    first_answer
      (fun '(_, ac) => scrutinee_by k i (scrutinee_var current scrutinee) ac)
      arms
  | Apply _ | Invalid => None
  end
with handler_first_scrutinee (k : continuation) (i : nat)
    (current : list variable) (h : continuation_handler) {struct h}
    : option (list variable) :=
  match h with
  | Handler _ _ handler => first_scrutinee k i current handler
  end.

(* The body of the [Let_cont] of [k], whose handler is [h], with the
   first of the variables [xs] given at position [i] for which
   FlambdaUnaliasParamsSpec.unalias_param would then take the parameter
   away; the body as it is if there is none. *)
Fixpoint give_first (xs : list variable) (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) : expr :=
  match xs with
  | [] => body
  | x :: xs =>
    let body' := give_scrutinee x k i None body in
    match unalias_param i k h body' with
    | Some _ => body'
    | None => give_first xs i k h body
    end
  end.

(* The same for the variables that the first application of [k] in
   the body says to try. *)
Definition give_at (i : nat) (k : continuation) (h : continuation_handler)
    (body : expr) : expr :=
  match first_scrutinee k i [] body with
  | Some xs => give_first xs i k h body
  | None => body
  end.

(* The same for every position below [i], the last first. *)
Fixpoint give_from (i : nat) (k : continuation) (h : continuation_handler)
    (body : expr) : expr :=
  match i with
  | O => body
  | S i => give_from i k h (give_at i k h body)
  end.

(* The pass: every non-recursive [Let_cont] of a term, its handler and
   its body first. The handlers of recursive continuations and the
   bodies of code are left as they are. *)
Fixpoint give_scrutinees (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (give_scrutinees body)
  | Let_static bindings body => Let_static bindings (give_scrutinees body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let h := Handler params metadata (give_scrutinees handler) in
    Let_cont_non_recursive k h
      (give_from (length params) k h (give_scrutinees body))
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (give_scrutinees body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result behaves as the term does, under all environments. *)
Definition give_scrutinees_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (give_scrutinees e).
