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

(* A parameter of a continuation that every application gives one
   symbol, or one constant, is that symbol or that constant: a
   normalisation that is an equivalence.

   This file is to be reviewed: [bind_const_params] and what it is made
   of, and the statement. The pass is written for the validation of
   middle_end/flambda2/tests/semantics/, which may apply it to either
   of the terms that it compares.

   It is FlambdaUnaliasParamsSpec.unalias_params for a symbol or a
   constant where that pass takes a variable. Simplify does as much:
   its analysis of aliases gives a parameter of a continuation the
   one thing that every path to the continuation gives it
   (simplify/flow/dominator_graph.ml, [dominator_analysis]; a symbol
   and a constant are their own dominators, [find_dom]), and it then
   removes the parameter and binds it at the start of the handler
   (simplify/flow/control_flow_graph.ml,
   [minimize_extra_args_for_one_continuation], and the [Let]s that
   simplify/simplify_let_cont_expr.ml makes of [lets_to_introduce]).
   So the term after Simplify has no such parameter where the term
   before has one; and either term has one where a rewrite of the
   validation has made the arguments of the applications one constant
   (FlambdaArmNumbersSpec.numbers_in_arms makes an argument that is
   the scrutinee of a [Switch] the number of its arm).

   [bind_const_params] takes the parameter away: the applications
   lose the argument (FlambdaUnaliasParamsSpec.drop_arg), the handler
   loses the parameter and begins with a [Let] of the parameter to the
   symbol or the constant, with the kind that the handler had for the
   parameter ([bind_const_param]). The machine tests the kind of a
   value at a [Let] as it does where a handler is applied, so the
   test that the application made is made by the [Let], and the pass
   is an equivalence: it may be applied to either term.
   FlambdaParamAliasesSpec.bind_aliased_params with
   FlambdaDeadParamsSpec.drop_dead_params, which the rounds of the term
   before apply, take such a parameter too, each as a refinement, and
   the second is no more than that: the parameters that it removes go
   with their tests.

   What the pass asks, of the parameter at a position of the handler
   of a non-recursive [Let_cont] of [k] ([bind_const_param]):
   - that the handler is not an exception handler, and has a kind for
     the parameter;
   - that the first application of [k] in the body of the [Let_cont]
     gives the parameter a symbol or a constant ([first_given],
     [is_plain]);
   - that every mention of [k] in the body of the [Let_cont] is an
     application of [k] that gives the parameter that same symbol or
     constant ([all_give]): no call returns or raises to [k], no trap
     action names it, and no [Let_cont] inside binds [k] again. An
     application of [k] may carry a trap action that names another
     continuation. The bodies of code are not looked into: code runs
     with its own continuations, and a [k] there is not this one;
   - that the parameter is not at another position of the handler too,
     where the [Let] would hide it.
   Nothing is asked of the binders between the [Let_cont] and an
   application, as [unalias_params] asks of them for its variable:
   FlambdaSemanticsSpec.eval_simple reads no environment for a symbol
   or a constant, so the argument has at the application the value
   that the [Let] gives the parameter in the handler, whatever is
   bound between the two. What a read through the symbol finds is in
   the static environment of the state, which the rewrite does not
   change.
   The positions are tried from the last to the first
   ([bind_const_from]), in every non-recursive [Let_cont] of the term,
   the innermost first ([bind_const_params]). The handlers of recursive
   continuations and the bodies of code are not gone into:
   FlambdaNormaliseSpec.rewrite_code and [rewrite_rec_handlers] take a
   pass there.

   The result is equivalent to the term
   ([bind_const_params_equivalent_statement]), under all environments.
   Nothing is asked of the names of the term beyond what the pass
   checks itself. A symbol that nothing binds passes: it is an argument
   in the term and the right-hand side of a [Let] in the result, and
   the machine makes the same value of it in both places.

   Renaming: the functions here compare names by equality only and
   make none. The pass commutes with every renaming and keeps scope;
   those two statements are in FlambdaValidatorEquivarianceSpec.v and
   in FlambdaStaticsScopeSpec.v. The [Let] that the handler gains
   mentions no variable; the symbol that it may mention is one that
   the body of the [Let_cont] mentions, and scope, in the sense of
   those files, does not count symbols.

   The proofs are in FlambdaBindConstParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaUnaliasParamsSpec.
Import ListNotations.

Definition same_simple (s s' : simple) : bool :=
  if simple_eq_dec s s' then true else false.

(* A constant or a symbol. *)
Definition is_plain (s : simple) : bool :=
  match s with
  | Var _ => false
  | Symbol _ | Const _ => true
  end.

(* No trap action of an [Apply_cont] names [k]; and if it applies [k],
   it gives the parameter [s]. *)
Definition apply_cont_gives (k : continuation) (i : nat) (s : simple)
    (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some action => negb (Pos.eqb (trap_action_handler action) k)
  | None => true
  end
  && (if Pos.eqb (ac_continuation ac) k
      then
        match nth_error (ac_args ac) i with
        | Some s' => same_simple s' s
        | None => false
        end
      else true).

Fixpoint all_give (k : continuation) (i : nat) (s : simple) (e : expr)
    {struct e} : bool :=
  match e with
  | Let _ _ _ body => all_give k i s body
  | Let_static _ body => all_give k i s body
  | Let_cont_non_recursive k' h body =>
    negb (Pos.eqb k' k) && handler_all_give k i s h && all_give k i s body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k', h) => negb (Pos.eqb k' k) && handler_all_give k i s h)
      handlers
    && all_give k i s body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) k)
    && negb (Pos.eqb (ap_exn_continuation ap) k)
  | Apply_cont ac => apply_cont_gives k i s ac
  | Switch _ arms => forallb (fun '(_, ac) => apply_cont_gives k i s ac) arms
  | Invalid => true
  end
with handler_all_give (k : continuation) (i : nat) (s : simple)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler => all_give k i s handler
  end.

(* What the first application of [k] gives. *)
Definition given_by (k : continuation) (i : nat) (ac : apply_cont)
    : option (option simple) :=
  if Pos.eqb (ac_continuation ac) k
  then Some (nth_error (ac_args ac) i)
  else None.

Fixpoint first_given (k : continuation) (i : nat) (e : expr) {struct e}
    : option (option simple) :=
  match e with
  | Let _ _ _ body => first_given k i body
  | Let_static _ body => first_given k i body
  | Let_cont_non_recursive _ h body =>
    match handler_first_given k i h with
    | Some r => Some r
    | None => first_given k i body
    end
  | Let_cont_recursive handlers body =>
    match first_answer (fun '(_, h) => handler_first_given k i h) handlers with
    | Some r => Some r
    | None => first_given k i body
    end
  | Apply_cont ac => given_by k i ac
  | Switch _ arms => first_answer (fun '(_, ac) => given_by k i ac) arms
  | Apply _ | Invalid => None
  end
with handler_first_given (k : continuation) (i : nat)
    (h : continuation_handler) {struct h} : option (option simple) :=
  match h with
  | Handler _ _ handler => first_given k i handler
  end.

Definition bind_const_param (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (continuation_handler * expr) :=
  match h with
  | Handler params metadata handler =>
    match nth_error params i, nth_error (hm_params_kinds metadata) i with
    | Some p, Some kind =>
      match first_given k i body with
      | Some (Some s) =>
        if is_plain s
           && negb (hm_is_exn_handler metadata)
           && all_give k i s body
           && negb (among p (drop_nth i params))
        then
          Some
            (Handler (drop_nth i params)
               {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
                  hm_is_exn_handler := false |}
               (Let p kind (Simple s) handler),
             drop_arg k i body)
        else None
      | _ => None
      end
    | _, _ => None
    end
  end.

Fixpoint bind_const_from (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : continuation_handler * expr :=
  match i with
  | O => (h, body)
  | S i =>
    match bind_const_param i k h body with
    | Some (h', body') => bind_const_from i k h' body'
    | None => bind_const_from i k h body
    end
  end.

Fixpoint bind_const_params (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (bind_const_params body)
  | Let_static bindings body => Let_static bindings (bind_const_params body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let h := Handler params metadata (bind_const_params handler) in
    let body := bind_const_params body in
    let '(h, body) := bind_const_from (length params) k h body in
    Let_cont_non_recursive k h body
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (bind_const_params body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The result behaves as the term does, under all environments. *)
Definition bind_const_params_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (bind_const_params e).
