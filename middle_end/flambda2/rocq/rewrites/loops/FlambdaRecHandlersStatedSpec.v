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

(* Rewriting the handlers of recursive continuations with what is known
   of the variables in scope: what lets a rewrite that asks what a
   variable is stated to be work inside a loop.

   This file is to be reviewed: [rewrite_rec_handlers_stated], which
   the validation applies (FlambdaValidateSpec.tidy_code_after; the
   tool of middle_end/flambda2/tests/semantics/ runs the extraction of
   the validation), and the statement.

   FlambdaRecHandlersSpec.rewrite_rec_handlers applies a rewrite to the
   handler of every recursive continuation of a term, and gives the
   rewrite the handler alone. That is enough for a rewrite that is an
   equivalence under all environments. It is not for one that tests
   what is known of a variable (FlambdaStatedScopeSpec.is_stated): the
   machine tests, where a handler is applied, that each argument is of
   what the handler states of its parameter, and a [Let] tests the kind
   only (decision D25, at parameters), so a rewrite that takes such a
   test away asks that the argument is known to be of what is stated.
   Given the handler alone, such a rewrite knows nothing of the
   parameters of the loop that it is in, nor of the variables bound
   around the loop. Its test of one of them then passes only where
   the parameter to be taken away has no entry in what its handler
   states, or an entry that says no more than is known of a variable
   of which nothing is known (FlambdaStatedScopeSpec.is_stated_at).
   Where the handler states more, the rewrite refuses:
   FlambdaUnaliasParamsSpec.unalias_params, say, leaves in a loop a
   parameter of a continuation that every application gives the
   parameter of the loop, where the handler of that continuation states
   that its parameter is a tagged immediate and the handler of the loop
   states the same of its own.

   [rewrite_rec_handlers_stated f S] is [rewrite_rec_handlers] for a
   rewrite [f] that takes what is known of the variables in scope
   (FlambdaStatedScopeSpec.stated_in_scope). It goes down the term
   with what is known, as the rewrites of that kind do
   (FlambdaUnaliasParamsSpec.unalias_params_under): a [Let] adds what
   its defining expression makes, the handler of a non-recursive
   continuation what it states of its parameters; and it gives [f],
   at the handler of a recursive continuation, what was known where
   the [Let_cont] is with what that handler states of its parameters
   added. In a handler that has a recursive [Let_cont] of its own, the
   handlers of that one are rewritten first, and they are given what
   the outer handler states too. As [rewrite_rec_handlers], it does
   not go into the bodies of code. Where [f] would change whether a
   handler binds something static, what [f] gives of that handler is
   not taken (FlambdaNormaliseSpec.rewrite_code_body): the handler
   stays as it is once the recursive handlers inside it are
   rewritten, and those rewrites are kept.

   Why what is given holds where the handler runs. A handler of a
   recursive [Let_cont] runs under the variables of the [Let_cont]
   with its parameters bound, each time that it is entered, from
   outside or from a handler of the group. The variables of the
   [Let_cont] hold what they held when it ran. The static environment
   may have grown since, and what is known of a variable may rest on
   what a symbol is bound to; as it grows, what is bound in it stays
   bound to the same, so a value is still of what it was of
   (FlambdaStatedScopeSpec.stated_hold_extend_statement). So what was
   known of those variables at the [Let_cont] holds. The parameters
   are bound to the values given, and the machine has just tested each
   against what the handler states
   (FlambdaSemanticsSpec.apply_cont_value): a run in which one is not
   of it is undefined at the application, before the handler. What is
   known of a variable that a parameter hides is replaced by what is
   stated of the parameter (FlambdaStatedScopeSpec.params_stated).

   The statement ([rewrite_rec_handlers_stated_equivalent_statement]):
   if [f S] gives, of every term, a term that runs the same under the
   environments in which [S] holds
   (FlambdaStatedScopeSpec.stated_hold), then so does
   [rewrite_rec_handlers_stated f S]. With nothing known that is
   FlambdaSemanticsSpec.expr_equivalent.

   Not done: the bodies of code are given nothing of what the code
   states of its parameters; a rewrite that refines and is no
   equivalence has no statement here.

   The functions here act on variables and continuations only through
   [f] and through what is known. That [rewrite_rec_handlers_stated f]
   commutes with a renaming where [f] does is proved for the use that
   the validation makes of it
   (FlambdaValidatorEquivarianceProofs.tidy_code_after_equivariant) and
   is not stated here.

   The proofs are in FlambdaRecHandlersStatedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaStatedScopeSpec.
Import ListNotations.

(* [f], given what is known there, applied to the handler of every
   recursive continuation. [S]: what is known of the variables bound
   around the term. *)
Fixpoint rewrite_rec_handlers_stated (f : stated_in_scope -> expr -> expr)
    (S : stated_in_scope) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d
      (rewrite_rec_handlers_stated f ((x, stated_of_named S d) :: S) body)
  | Let_static bindings body =>
    Let_static bindings (rewrite_rec_handlers_stated f S body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (rewrite_rec_handlers_stated_in_handler f S h)
      (rewrite_rec_handlers_stated f S body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_rec_handler_stated f S h)) handlers)
      (rewrite_rec_handlers_stated f S body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_rec_handlers_stated_in_handler
    (f : stated_in_scope -> expr -> expr) (S : stated_in_scope)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (rewrite_rec_handlers_stated f
         (params_stated params (hm_params_subkinds metadata) S) handler)
  end
with rewrite_rec_handler_stated (f : stated_in_scope -> expr -> expr)
    (S : stated_in_scope) (h : continuation_handler) {struct h}
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (rewrite_code_body
         (f (params_stated params (hm_params_subkinds metadata) S))
         (rewrite_rec_handlers_stated f
            (params_stated params (hm_params_subkinds metadata) S) handler))
  end.

(* A rewrite that gives, under what is known, a term that runs the same
   under the environments in which that holds, is such a rewrite when
   applied to the handlers of recursive continuations. *)
Definition rewrite_rec_handlers_stated_equivalent_statement : Prop :=
  ∀ f : stated_in_scope -> expr -> expr,
    (∀ (S : stated_in_scope) (e : expr) (vars : env) (conts : cont_env)
       (statics : static_env) (hp : heap) (rd : read_record),
       stated_hold statics S vars ->
       equivalent (State e vars conts statics hp rd)
         (State (f S e) vars conts statics hp rd)) ->
    ∀ (S : stated_in_scope) (e : expr) (vars : env) (conts : cont_env)
      (statics : static_env) (hp : heap) (rd : read_record),
      stated_hold statics S vars ->
      equivalent (State e vars conts statics hp rd)
        (State (rewrite_rec_handlers_stated f S e) vars conts statics hp rd).
