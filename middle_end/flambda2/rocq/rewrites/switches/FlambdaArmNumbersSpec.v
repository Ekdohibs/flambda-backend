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



(* Where a [Switch] on a variable has decided what the variable is,
   the variable is that number in the places, listed below, where it
   is read: a normalisation that is an equivalence.

   This file is to be reviewed: [numbers_in_arms] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the two statements. The
   validation applies the pass to each of the two terms that it
   compares, once it has placed their continuations, just after
   FlambdaUnaliasParamsSpec.unalias_params
   (FlambdaValidateSpec.tidied_before, [tidied_after]).

   Simplify simplifies an arm of a [Switch] in an environment in
   which the scrutinee is the number of the arm
   (simplify/simplify_switch_expr.ml: [simplify_switch] meets the type
   of the scrutinee with [T.this_naked_immediate arm] for each arm,
   and [simplify_arm] simplifies the arguments of the arm, and
   records the use of its continuation, under what the meet gives).
   An argument that is the scrutinee is then the constant; and the
   handler of a continuation that has that one use is simplified
   under the same knowledge, so that there a primitive on the
   scrutinee has a constant for an argument, and a second [Switch] on
   it is a [Switch] on a constant. Where the scrutinee is the
   untagging of a variable, the meet says of that variable that it
   is the tagged number, and a primitive on it is on a constant too.
   So for a flag that a loop tests before it sets it,

     for i = 0 to n do
       if not !seen then if p i then (seen := true; first := i)
     done

   the term after Simplify passes the constant [0] or [1] on where
   the term before passes the flag as it was read; and under the arm
   of a match on the length of an array,

     match Array.length a with 1 -> a.(0) | ...

   the test of the index against the length is the test of two
   constants, which Simplify decides.

   [number_in_arms x] puts, for the variable [x] at the places
   listed below where it is read, the naked immediate that [x] is
   known to be there. What is known ([known], a number or nothing)
   is found on the way down as
   FlambdaGiveScrutineeSpec.give_scrutinee finds it, by the functions
   of that file:
   - in an arm of a [Switch] on [x], [x] is the number of the arm
     ([arm_known]);
   - in the handler of a non-recursive continuation whose [Let_cont]
     has for its body a [Switch] on [x], or comes to one through
     non-recursive [Let_cont]s that do not bind the continuation
     again and whose handlers do not mention it, and that only arms
     of one number apply, [x] is that number ([arm_fact]);
   - what is known where a continuation is defined is known in its
     handler, which runs under the variables of its definition and
     its parameters;
   - nothing is known of [x] under a [Let] of [x] or in a handler
     that has [x] among its parameters ([known_under]).
   Where [x] is known to be a number:
   - an argument of an application of a continuation that is [x] is
     the number ([number_in_apply_cont]), in the arms of a [Switch]
     too;
   - a defining expression that is [x], an argument of a primitive
     that is [x], and the scrutinee of a [Switch] that is [x], are
     the number ([number_read]);
   - and where a [Let] in scope binds [x] to the untagging of a
     variable, and neither has been bound again ([source]: the
     variable untagged; [source_under_let], [source_under]), a
     defining expression, an argument of a primitive and the
     scrutinee of a [Switch] that are that variable are the tagged
     immediate of the number. An argument of an application of a
     continuation that is that variable stays: where Simplify merges
     the arms of a [Switch] it does so with the variable there.
   The callee and the arguments of a call, the value slots of a set
   of closures, the closure of a projection and the fields of a
   static binding keep the variable. The bindings of a [Let_static],
   and so the bodies of code, are left as they are.

   The rewrite is an equivalence whatever [x] is
   ([number_in_arms_equivalent_statement]): a variable is changed
   only where it holds the constant. The pass ([numbers_in_arms])
   makes it for each variable that is the scrutinee of a [Switch] of
   the term, in the order in which the term has them ([scrutinees];
   a variable that several have is taken once for each, and the
   rewrite changes nothing the second time). It goes into the
   handlers of recursive continuations; the bodies of code are not
   gone into: FlambdaNormaliseSpec.rewrite_code takes a pass there.

   For the arguments of applications it is the rewrite of
   FlambdaGiveScrutineeSpec.give_scrutinee the other way; the pass of
   that file keeps its rewrite only where
   FlambdaUnaliasParamsSpec.unalias_param would then take the
   parameter away. Applied to the result of that pass as it is, this
   one puts the numbers back. The validation applies that pass, then
   [unalias_params], which takes the parameter away and with it the
   arguments that were made the variable, and then this one: there
   it undoes nothing.

   The result is equivalent to the term
   ([numbers_in_arms_equivalent_statement]), under all environments.

   Renaming: the functions here compare names by equality only and
   make none.

   The proofs are in FlambdaArmNumbersProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaUnaliasParamsSpec
  FlambdaUntagParamsSpec FlambdaGiveScrutineeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The number [c] for the variable [x]. *)
Definition number_simple (x : variable) (c : Z) (s : simple) : simple :=
  match s with
  | Var y => if Pos.eqb y x then Const (Naked_immediate c) else s
  | _ => s
  end.

Definition number_in_apply_cont (x : variable) (known : option Z)
    (ac : apply_cont) : apply_cont :=
  match known with
  | Some c =>
    {| ac_continuation := ac_continuation ac;
       ac_args := map (number_simple x c) (ac_args ac);
       ac_trap_action := ac_trap_action ac |}
  | None => ac
  end.

(* The variable that [x] is the untagging of, under a [Let] of [y] to
   [d]: the one that [d] untags, if [y] is [x]; no longer the one
   that it was, if [y] is that one. *)
Definition source_under_let (x y : variable) (d : named)
    (source : option variable) : option variable :=
  if Pos.eqb y x
  then
    match untag_of d with
    | Some t => if Pos.eqb t x then None else Some t
    | None => None
    end
  else
    match source with
    | Some t => if Pos.eqb y t then None else source
    | None => None
    end.

(* The same in a handler: no variable if [x] or that variable is
   among the parameters. *)
Definition source_under (x : variable) (params : list variable)
    (source : option variable) : option variable :=
  if among x params then None
  else
    match source with
    | Some t => if among t params then None else source
    | None => None
    end.

(* The number [c] for the variable [x], and the tagged immediate of
   [c] for the variable that [x] is the untagging of. *)
Definition number_read (x : variable) (source : option variable) (c : Z)
    (s : simple) : simple :=
  match s with
  | Var y =>
    if Pos.eqb y x then Const (Naked_immediate c)
    else
      match source with
      | Some t => if Pos.eqb y t then Const (Tagged_immediate c) else s
      | None => s
      end
  | _ => s
  end.

Definition number_in_named (x : variable) (source : option variable)
    (known : option Z) (d : named) : named :=
  match known, d with
  | Some c, Simple s => Simple (number_read x source c s)
  | Some c, Prim p args => Prim p (map (number_read x source c) args)
  | _, _ => d
  end.

Definition number_in_scrutinee (x : variable) (source : option variable)
    (known : option Z) (scrutinee : simple) : simple :=
  match known with
  | Some c => number_read x source c scrutinee
  | None => scrutinee
  end.

Fixpoint number_in_region (x : variable) (source : option variable)
    (known : option Z) (e : expr) {struct e} : expr :=
  match e with
  | Let y kind d body =>
    Let y kind (number_in_named x source known d)
      (number_in_region x (source_under_let x y d source)
         (if Pos.eqb y x then None else known) body)
  | Let_static bindings body =>
    Let_static bindings (number_in_region x source known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (number_in_handler x source
         match arm_fact x k body with
         | Some c => Some c
         | None => known
         end
         h)
      (number_in_region x source known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, number_in_handler x source known h)) handlers)
      (number_in_region x source known body)
  | Apply_cont ac => Apply_cont (number_in_apply_cont x known ac)
  | Switch scrutinee arms =>
    Switch (number_in_scrutinee x source known scrutinee)
      (map
         (fun '(d, ac) =>
            (d, number_in_apply_cont x (arm_known x scrutinee d known) ac))
         arms)
  | Apply _ | Invalid => e
  end
with number_in_handler (x : variable) (source : option variable)
    (known : option Z) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (number_in_region x (source_under x params source)
         (known_under x params known) handler)
  end.

(* The rewrite for the whole of a term: no [Let] of [x] is in scope
   yet. *)
Definition number_in_arms (x : variable) (known : option Z) (e : expr)
    : expr :=
  number_in_region x None known e.

(* The result behaves as the term does, under all environments,
   whatever the variable is. *)
Definition number_in_arms_equivalent_statement : Prop :=
  ∀ (x : variable) (e : expr), expr_equivalent e (number_in_arms x None e).

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The variables that are the scrutinee of a [Switch] of a term,
   outside code. *)
Fixpoint scrutinees (e : expr) {struct e} : list variable :=
  match e with
  | Let _ _ _ body => scrutinees body
  | Let_static _ body => scrutinees body
  | Let_cont_non_recursive _ h body =>
    handler_scrutinees h ++ scrutinees body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_scrutinees h) handlers
    ++ scrutinees body
  | Switch (Var x) _ => [x]
  | Switch _ _ | Apply _ | Apply_cont _ | Invalid => []
  end
with handler_scrutinees (h : continuation_handler) {struct h}
    : list variable :=
  match h with
  | Handler _ _ handler => scrutinees handler
  end.

(* The pass: the rewrite for each of them in turn. *)
Definition numbers_in_arms (e : expr) : expr :=
  fold_left (fun e x => number_in_arms x None e) e (scrutinees e).

Definition numbers_in_arms_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (numbers_in_arms e).
