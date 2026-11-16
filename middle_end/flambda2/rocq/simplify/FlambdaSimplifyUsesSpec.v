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


(* The handler of a continuation, rewritten under what is known where
   the continuation is applied.

   This file is to be reviewed: the statements, against
   middle_end/flambda2/simplify/join_points.ml ([compute_handler_env],
   [compute_use_env_with_ids]) and
   middle_end/flambda2/simplify/simplify_switch_expr.ml (the
   environment of an arm). The functions that the statements are about
   ([arm_env], [use_env], [leaf_uses]) are in
   FlambdaSimplifyPassSpec.v, with the function that uses them, and are
   to be reviewed there.

   FlambdaSimplifyHandlersSpec.v rewrites a handler under the
   environment of its [Let_cont] with the parameters defined at their
   kinds: nothing is known of a parameter, and nothing of what was
   learnt between the [Let_cont] and an application of the
   continuation. Simplify knows more. It goes down the body of a
   non-recursive [Let_cont] first, records the environment at each
   application of the continuation with the types of the arguments,
   and rewrites the handler afterwards:
   - where the continuation is applied once, by an [Apply_cont] that
     is not an arm of a [Switch] and has no trap action, under the
     environment of that application, with the parameters defined
     there and given the types of the arguments: no join is made
     ([compute_handler_env], the case of one inlinable use; the
     continuation is not then the return continuation of a function
     or of the unit, whose uses are never inlinable);
   - otherwise under the join of the environments of the applications,
     each with the parameters defined and given the types of the
     arguments, from the environment of the [Let_cont] with the
     parameters defined (if there are several applications, only under
     -flambda2-join-points; without it no join of the types of the
     arguments is made, and a parameter has the unknown type of its
     subkind: [add_parameters_with_unknown_types]).
   The environment at an application that is an arm of a [Switch] is
   the one in which the type of the scrutinee has been met with the
   discriminant of the arm (simplify_switch_expr.ml).

   What is stated here, for a [Let_cont] whose body is an [Apply_cont]
   or a [Switch], so that the continuation is applied, if at all, as
   soon as the body runs:
   - the environment of an arm holds in the arm
     ([arm_env_satisfied_statement]), and the environment made for an
     application holds where the handler then runs
     ([use_env_satisfied_statement]);
   - where the body applies the continuation at one place, the handler
     may be rewritten under the environment made for that application
     ([let_cont_handler_at_use_statement]). An arm of a [Switch] is
     such a place: the model makes no join of one environment, where
     Simplify makes one;
   - whatever the number of applications, the handler may be rewritten
     under any environment that has the guarantees of a join
     (FlambdaJoinSpec.join_guarantees) of the environments made for
     the applications, from the environment of the [Let_cont] with the
     parameters defined ([let_cont_handler_joined_statement]). This is
     where the contract of the join meets the terms: its premise that
     one valuation satisfies a use and the source
     (FlambdaJoinSpec.use_and_source_satisfied) is proved of these
     environments, and so is that the variables of terms of the source
     are variables of terms at each use
     (FlambdaJoinSpec.terms_in_scope_at_uses). No join is named: the
     statement is of every environment with the guarantees.
   A [Let_cont] whose body binds something before it applies the
   continuation is brought to this case by moving the [Let_cont] down
   to the application: FlambdaSimplifyPassSpec.find_use looks for the
   application, and FlambdaSimplifyPassSpec.find_use_refines_statement
   is the statement for one application. The join is stated only for
   a body that is the application.

   Where the model and the compiler part:
   - The environment of an arm ([arm_env]) is made without the meet:
     the canonical element of the scrutinee, if it is a variable, is
     made an alias of the constant of the arm. So what the meet would
     learn of the name that the scrutinee was computed from (that it
     is an immediate, or has a tag) is not learnt, and no arm is found
     to be unreachable.
   - A parameter is made an alias of the canonical element of its
     argument ([use_env]), where Simplify gives it the type of the
     argument, which is the alias type of that element. It is made so
     only if that element may occur in terms where the handler is
     defined: Simplify, where it joins, makes the variables defined
     between the [Let_cont] and the application existential, and the
     model has no such step.
   - Only applications are uses here. A continuation that a call
     returns to, or raises to, is used by an [Apply], and nothing is
     said of a body that is one. A trap action that names the
     continuation is not counted as an application of it, and an
     [Apply_cont] or an arm that has a trap action is counted as any
     other ([leaf_uses] does not read trap actions): the statements
     are of the machine of FlambdaSemanticsSpec.step, whose step for
     an [Apply_cont] does not read the trap action either. Simplify
     does not take a use with a trap action to be inlinable.
   - The lifted constants, the extra parameters and arguments, and the
     subkinds of the parameters are not modelled.

   Renaming. That [arm_env], [use_env] and [leaf_uses] commute with
   renaming is in FlambdaSimplifyPassRenamingSpec.v, with the
   equations of the other functions of that section of
   FlambdaSimplifyPassSpec.v.

   Nothing is assumed beyond the premises of each statement.

   The proofs are in FlambdaSimplifyUsesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvRefinementSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaSimplifyHandlersSpec FlambdaSimplifyPassSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environments                                                         *)
(* ======================================================================== *)

(* The environment of an arm holds in the arm: in a state that
   satisfies the environment of the [Switch] and in which the scrutinee
   is the discriminant of the arm. *)
Definition arm_env_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (scrutinee : simple) (d : Z) (statics : static_env)
    (vars : env),
    env_satisfies_strictly tenv statics vars ->
    eval_simple vars scrutinee = Some (Value_const (Naked_immediate d)) ->
    env_satisfies_strictly (arm_env tenv scrutinee d) statics vars.

(* The environment made for an application holds where the handler
   runs: under the variables of the application, which satisfy its
   environment, with the parameters bound to the values of the
   arguments, which are of the kinds of the handler. (The handler runs
   under the variables of its own definition, with the parameters
   bound; the statements below are of a continuation that is applied
   where it is defined, so that the two are the same.) *)
Definition use_env_satisfied_statement : Prop :=
  ∀ (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple) (target : Typing_env.t)
    (statics : static_env) (vars : env) (vs : list value)
    (handler_vars : env),
    use_env scope tenv params metadata args = Some target ->
    env_satisfies_strictly tenv statics vars ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    env_satisfies_strictly target statics handler_vars.

(* ======================================================================== *)
(* One application                                                          *)
(* ======================================================================== *)

(* The body applies the continuation at one place: the handler may be
   rewritten under the environment made for that application. *)
Definition let_cont_handler_at_use_statement : Prop :=
  ∀ (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (tenv_use : Typing_env.t) (args : list simple) (target : Typing_env.t),
    leaf_uses tenv k body = Some [(tenv_use, args)] ->
    use_env scope tenv_use params metadata args = Some target ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* ======================================================================== *)
(* The join of the applications                                             *)
(* ======================================================================== *)

(* The handler may be rewritten under an environment that has the
   guarantees of a join whose source is the environment of the
   [Let_cont] with the parameters defined ([handler_env]) and whose
   uses are the environments made for the applications of the
   continuation in the body, in their order. The identifiers of the
   uses, the scope after which the join cuts, the variables that it
   avoids and its analysis are whatever they are: the guarantees are
   all that is asked. *)
Definition let_cont_handler_joined_statement : Prop :=
  ∀ (well_formed : Typing_env.t -> Prop) (A : Type) (avoid : list variable)
    (scope tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (applications : list (Typing_env.t * list simple))
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    leaf_uses tenv k body = Some applications ->
    handler_env tenv params metadata = Some (source inputs) ->
    Forall2
      (fun application use =>
         use_env scope (fst application) params metadata (snd application)
         = Some (snd use))
      applications (uses inputs) ->
    join_guarantees well_formed avoid inputs target analysis ->
    expr_refines_under target handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).
