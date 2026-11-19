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

(* Rewriting the bodies of code with what the code states of its
   parameters: what lets a rewrite that asks what a variable is stated
   to be work in the body of a function.

   This file is to be reviewed: [rewrite_code_stated], which the
   validation applies (the rounds
   FlambdaValidateSpec.first_phase_round, tidy_before_round and
   tidy_after_round; the tool of middle_end/flambda2/tests/semantics/
   runs the extraction of the validation), and the two statements.

   FlambdaNormaliseSpec.rewrite_code applies a rewrite to the body of
   every piece of code that a term binds, and gives the rewrite the
   body alone. A rewrite that tests what is known of a variable
   (FlambdaStatedScopeSpec.is_stated; decision D25, at parameters)
   then knows nothing of the parameters of the code, and its test of
   one of them passes only where the handler states, of the parameter
   to be taken away, no more than is known of a variable of which
   nothing is known. Where the handler states more, the rewrite
   refuses: FlambdaUnaliasRecParamsSpec.unalias_rec_params, say,
   leaves in the loop that a function has become a parameter that
   every turn passes on and that the loop is given the parameter of
   the function for, where the handler of the loop states that the
   parameter is not null and the code states the same of its own.

   [rewrite_code_stated g] is [rewrite_code] for a rewrite [g] that
   takes what is known of the variables in scope
   (FlambdaStatedScopeSpec.stated_in_scope): it gives [g], at the body
   of code, what the code states of its parameters and that its
   closure is not null (FlambdaStatedScopeSpec.code_stated). Nothing
   of the variables bound around the [Let_static] is given: in code
   no variable of the term is in scope. As [rewrite_code], it goes
   through the term and the handlers of its continuations, and a body
   is left alone where the rewrite would change whether it binds
   something static (FlambdaNormaliseSpec.rewrite_code_body).

   Why what is given holds where the body runs. The body of code runs
   at a call, under its parameters bound to the arguments and its
   closure variable bound to the callee, and nothing else. The machine
   enters code with a closure for callee
   (FlambdaSemanticsSpec.shape_of_call), which is not null, and tests
   the arguments against what the code states of its parameters
   (FlambdaSemanticsSpec.call_code): a call in which one is not of it
   is undefined before the body. That is
   FlambdaStatedScopeSpec.code_stated_hold_statement.

   The statements: if [g S] gives, of every term, a term that refines
   it under the environments in which [S] holds
   (FlambdaStatedScopeSpec.stated_hold), then [rewrite_code_stated g]
   refines, under all environments
   ([rewrite_code_stated_refines_statement]); and the same for a [g]
   that gives a term that runs the same, with an equivalence
   ([rewrite_code_stated_equivalent_statement]).

   Not done: what is known of the symbols that the static bindings in
   scope bind is not given (FlambdaUnboxInScopeSpec.rewrite_code_in_scope
   gives facts about them to another kind of rewrite).

   That [rewrite_code_stated g] commutes with a renaming where [g]
   does, and keeps scope where [g] does, is proved for the uses that
   the validation makes of it and is not stated here.

   The proofs are in FlambdaCodeStatedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaStatedScopeSpec.
Import ListNotations.

(* [g], given what the code states, applied to the body of code. *)
Definition rewrite_code_stated_in_binding
    (g : stated_in_scope -> expr -> expr) (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata
         (rewrite_code_body (g (code_stated params my_closure metadata))
            body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* That at the body of every piece of code that a term binds. *)
Fixpoint rewrite_code_stated (g : stated_in_scope -> expr -> expr)
    (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (rewrite_code_stated g body)
  | Let_static bindings body =>
    Let_static (map (rewrite_code_stated_in_binding g) bindings)
      (rewrite_code_stated g body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (rewrite_code_stated_in_handler g h)
      (rewrite_code_stated g body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_code_stated_in_handler g h)) handlers)
      (rewrite_code_stated g body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_code_stated_in_handler (g : stated_in_scope -> expr -> expr)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (rewrite_code_stated g handler)
  end.

(* A rewrite that gives, under what is known, a term that refines
   under the environments in which that holds, refines under all
   environments when it is applied to the bodies of code with what
   the code states. *)
Definition rewrite_code_stated_refines_statement : Prop :=
  ∀ g : stated_in_scope -> expr -> expr,
    (∀ (S : stated_in_scope) (e : expr) (vars : env) (conts : cont_env)
       (statics : static_env) (hp : heap) (rd : read_record),
       stated_hold statics S vars ->
       refines (State e vars conts statics hp rd)
         (State (g S e) vars conts statics hp rd)) ->
    ∀ e : expr, expr_refines e (rewrite_code_stated g e).

(* The same for a rewrite that gives a term that runs the same. *)
Definition rewrite_code_stated_equivalent_statement : Prop :=
  ∀ g : stated_in_scope -> expr -> expr,
    (∀ (S : stated_in_scope) (e : expr) (vars : env) (conts : cont_env)
       (statics : static_env) (hp : heap) (rd : read_record),
       stated_hold statics S vars ->
       equivalent (State e vars conts statics hp rd)
         (State (g S e) vars conts statics hp rd)) ->
    ∀ e : expr, expr_equivalent e (rewrite_code_stated g e).
