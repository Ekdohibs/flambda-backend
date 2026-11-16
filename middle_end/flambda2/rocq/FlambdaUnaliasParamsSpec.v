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
   variable, which is at the handler what it is at the applications, is
   that variable: a normalisation that is an equivalence.

   This file is to be reviewed: [unalias_params] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. The
   validation applies the pass to each of the two terms that it
   compares, once it has placed their continuations
   (FlambdaValidateSpec.tidied_before, [tidied_after]).

   Simplify gives a continuation a parameter for a primitive that has
   been computed on every path to it
   (simplify/common_subexpression_elimination.ml, [join], which
   simplify/join_points.ml calls). Where the results are not held by
   one variable that is in scope where the continuation is defined
   ([join_one_cse_equation]: every case but [Rhs_in_scope]), the
   continuation has a new parameter, each application has for it the
   variable that holds the result on its path, and the handler is
   simplified knowing that the parameter is the result: an application
   of the primitive there becomes the parameter. The term before
   Simplify has no such parameter and computes the primitive again in
   the handler. A continuation with one application is a case of this:
   the primitive is computed between the [Let_cont] and the
   application, as for the test of the bounds of a string and the
   read that follows it.

   Once the continuations are placed, the [Let_cont] of such a
   continuation is under the [Let] that computes the primitive, and
   the variable given for the parameter is in scope at the handler.
   [unalias_params] then takes the parameter away: the applications
   lose the argument ([drop_arg]), the handler loses the parameter and
   begins with a [Let] of the parameter to that variable, with the
   kind that the handler had for the parameter ([unalias_param]).
   What follows aliases and removes a [Let] that is not used does the
   rest, and FlambdaCseSpec.share_known_primitives, on the term
   before, makes the second computation of the primitive a [Let] of
   the variable that holds the first.

   FlambdaParamAliasesSpec.bind_aliased_params with
   FlambdaDeadParamsSpec.drop_dead_params, which the rounds of the
   term before apply, do as much for a parameter that is always given
   one thing, and more: they take constants and symbols, and the
   parameters of a loop. Each is stated as a refinement, and the
   second is no more than that: the machine tests the kind of a value
   where a handler is applied, and the parameters that it removes go
   with their tests. A rewrite of the term after Simplify has to be
   refined by the term that it rewrites, so those two are for the term
   before only. [unalias_params] keeps the test, in the [Let], where
   the machine tests the kind in the same way; so it is an
   equivalence, and may be applied to either term. It also asks less
   of the variable: that no binder on the way from the [Let_cont] to
   an application binds it, where [bind_aliased_params] asks that
   nothing in the scope of the continuation binds it.

   A parameter that Simplify removes because every path gives it one
   variable in scope where the continuation is defined
   (simplify/flow/control_flow_graph.ml,
   [minimize_extra_args_for_one_continuation], from the aliases that
   simplify/flow/dominator_graph.ml finds; FlambdaParamAliasesSpec.v
   says how the rounds follow it) is of the same shape. This pass
   takes it too where the rounds leave it in the term before: where
   the variable comes into the scope of the handler only once the
   continuation is placed under the [Let] that binds it, or is bound
   again somewhere in the scope of the continuation, off the path to
   an application.

   What the pass asks, of the parameter at a position of the handler
   of a non-recursive [Let_cont] of [k] ([unalias_param]):
   - that the handler is not an exception handler, and has a kind for
     the parameter;
   - that every mention of [k] in the body of the [Let_cont] is an
     application of [k] ([all_give_var]): no call returns or raises to
     [k], no trap action names it, and no [Let_cont] inside binds [k]
     again. An application of [k] may carry a trap action that names
     another continuation;
   - that every such application gives the parameter a variable, the
     same in all ([first_given_var] finds it in the first), and that
     no [Let] and no parameter of a handler binds that variable
     between the [Let_cont] of [k] and the application, so that it is
     there the variable that the handler of [k] sees. The bodies of
     code are not looked into: code runs with its own continuations,
     and a [k] there is not this one;
   - that the variable is not a parameter of the handler, and that the
     parameter is not at another position of the handler too.
   The positions are tried from the last to the first ([unalias_from]),
   in every non-recursive [Let_cont] of the term, the innermost first
   ([unalias_params], which is FlambdaSpineWalkSpec.walk_spine of the
   rewrite of one [Let_cont], [unalias_node]). The handlers of
   recursive continuations and the bodies of code are not gone into:
   FlambdaNormaliseSpec.rewrite_code and [rewrite_rec_handlers] take a
   pass there.

   The result is equivalent to the term
   ([unalias_params_equivalent_statement]), under all environments.
   Nothing is asked of the names of the term beyond what the pass
   checks itself. A variable that is not bound at all passes: the
   application is undefined in the term, and the [Let] is in the
   result.

   Renaming: the functions here compare names by equality only and
   make none. The pass commutes with every renaming and keeps scope;
   those two statements are in FlambdaValidatorEquivarianceSpec.v and
   in FlambdaStaticsScopeSpec.v.

   The proofs are in FlambdaUnaliasParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaSpineWalkSpec.
Import ListNotations.

(* A list without its element at a position. *)
Definition drop_nth {A : Type} (i : nat) (l : list A) : list A :=
  firstn i l ++ skipn (S i) l.

(* The first answer that a function gives of an element of a list. *)
Fixpoint first_answer {A B : Type} (f : A -> option B) (l : list A)
    : option B :=
  match l with
  | [] => None
  | a :: l =>
    match f a with
    | Some b => Some b
    | None => first_answer f l
    end
  end.

(* No trap action of an [Apply_cont] names [k]; and if the [Apply_cont]
   applies [k], it gives the parameter the variable [x], where [x] is
   still the one that was in scope at the [Let_cont] of [k] ([ok]). An
   application of [k] may carry a trap action that names another
   continuation. *)
Definition apply_cont_gives_var (k : continuation) (i : nat) (x : variable)
    (ok : bool) (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some action => negb (Pos.eqb (trap_action_handler action) k)
  | None => true
  end
  && (if Pos.eqb (ac_continuation ac) k
      then
        match nth_error (ac_args ac) i with
        | Some (Var y) => ok && Pos.eqb x y
        | _ => false
        end
      else true).

(* Every mention of [k] in a term is such an application, and [k] is
   not bound again. *)
Fixpoint all_give_var (k : continuation) (i : nat) (x : variable)
    (ok : bool) (e : expr) {struct e} : bool :=
  match e with
  | Let y _ _ body => all_give_var k i x (ok && negb (Pos.eqb x y)) body
  | Let_static _ body => all_give_var k i x ok body
  | Let_cont_non_recursive k' h body =>
    negb (Pos.eqb k' k)
    && handler_all_give_var k i x ok h
    && all_give_var k i x ok body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k', h) =>
         negb (Pos.eqb k' k) && handler_all_give_var k i x ok h)
      handlers
    && all_give_var k i x ok body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) k)
    && negb (Pos.eqb (ap_exn_continuation ap) k)
  | Apply_cont ac => apply_cont_gives_var k i x ok ac
  | Switch _ arms =>
    forallb (fun '(_, ac) => apply_cont_gives_var k i x ok ac) arms
  | Invalid => true
  end
with handler_all_give_var (k : continuation) (i : nat) (x : variable)
    (ok : bool) (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params _ handler =>
    all_give_var k i x (ok && negb (among x params)) handler
  end.

(* The variable that the first application of [k] gives. *)
Definition given_var_by (k : continuation) (i : nat) (ac : apply_cont)
    : option (option variable) :=
  if Pos.eqb (ac_continuation ac) k
  then
    Some
      match nth_error (ac_args ac) i with
      | Some (Var y) => Some y
      | _ => None
      end
  else None.

Fixpoint first_given_var (k : continuation) (i : nat) (e : expr) {struct e}
    : option (option variable) :=
  match e with
  | Let _ _ _ body => first_given_var k i body
  | Let_static _ body => first_given_var k i body
  | Let_cont_non_recursive _ h body =>
    match handler_first_given_var k i h with
    | Some r => Some r
    | None => first_given_var k i body
    end
  | Let_cont_recursive handlers body =>
    match
      first_answer (fun '(_, h) => handler_first_given_var k i h) handlers
    with
    | Some r => Some r
    | None => first_given_var k i body
    end
  | Apply_cont ac => given_var_by k i ac
  | Switch _ arms => first_answer (fun '(_, ac) => given_var_by k i ac) arms
  | Apply _ | Invalid => None
  end
with handler_first_given_var (k : continuation) (i : nat)
    (h : continuation_handler) {struct h} : option (option variable) :=
  match h with
  | Handler _ _ handler => first_given_var k i handler
  end.

(* Every application of [k] without its argument at a position. *)
Definition drop_arg_of (k : continuation) (i : nat) (ac : apply_cont)
    : apply_cont :=
  if Pos.eqb (ac_continuation ac) k
  then
    {| ac_continuation := ac_continuation ac;
       ac_args := drop_nth i (ac_args ac);
       ac_trap_action := ac_trap_action ac |}
  else ac.

Fixpoint drop_arg (k : continuation) (i : nat) (e : expr) {struct e}
    : expr :=
  match e with
  | Let x kind d body => Let x kind d (drop_arg k i body)
  | Let_static bindings body => Let_static bindings (drop_arg k i body)
  | Let_cont_non_recursive k' h body =>
    Let_cont_non_recursive k' (drop_arg_in_handler k i h)
      (drop_arg k i body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k', h) => (k', drop_arg_in_handler k i h)) handlers)
      (drop_arg k i body)
  | Apply_cont ac => Apply_cont (drop_arg_of k i ac)
  | Switch scrutinee arms =>
    Switch scrutinee (map (fun '(n, ac) => (n, drop_arg_of k i ac)) arms)
  | Apply _ | Invalid => e
  end
with drop_arg_in_handler (k : continuation) (i : nat)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (drop_arg k i handler)
  end.

(* The handler [h] of [k] and the body of its [Let_cont] without the
   parameter at position [i], where every application of [k] in the
   body gives it one variable that no binder between the [Let_cont] and
   the application binds, and that is not a parameter of [h]. The new
   handler begins with a [Let] of the parameter to that variable, with
   the kind that [h] had for the parameter: the machine tests the kind
   at the [Let] as it did where [h] was applied. The parameter is at no
   other position of [h], where the [Let] would hide it. An exception
   handler is left as it is. *)
Definition unalias_param (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : option (continuation_handler * expr) :=
  match h with
  | Handler params metadata handler =>
    match nth_error params i, nth_error (hm_params_kinds metadata) i with
    | Some p, Some kind =>
      match first_given_var k i body with
      | Some (Some x) =>
        if negb (hm_is_exn_handler metadata)
           && all_give_var k i x true body
           && negb (among x params)
           && negb (among p (drop_nth i params))
        then
          Some
            (Handler (drop_nth i params)
               {| hm_params_kinds := drop_nth i (hm_params_kinds metadata);
                  hm_is_exn_handler := false |}
               (Let p kind (Simple (Var x)) handler),
             drop_arg k i body)
        else None
      | _ => None
      end
    | _, _ => None
    end
  end.

(* The same for every position below [i], the last first. *)
Fixpoint unalias_from (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr)
    : continuation_handler * expr :=
  match i with
  | O => (h, body)
  | S i =>
    match unalias_param i k h body with
    | Some (h', body') => unalias_from i k h' body'
    | None => unalias_from i k h body
    end
  end.

(* The pass at one node: the parameters of a non-recursive [Let_cont],
   the last first. Any other term is left as it is. *)
Definition unalias_node (e : expr) : expr :=
  match e with
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let '(h, body) :=
      unalias_from (length params) k (Handler params metadata handler) body
    in
    Let_cont_non_recursive k h body
  | Let _ _ _ _ | Let_static _ _ | Let_cont_recursive _ _ | Apply _
  | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The pass: that step at every non-recursive [Let_cont] of a term, its
   handler and its body first (FlambdaSpineWalkSpec.walk_spine). The
   handlers of recursive continuations and the bodies of code are left
   as they are. *)
Definition unalias_params (e : expr) : expr := walk_spine unalias_node e.

(* The result behaves as the term does, under all environments. *)
Definition unalias_params_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (unalias_params e).
