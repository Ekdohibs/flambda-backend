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


(* The environment in which the handler of a continuation is rewritten,
   at the kinds that the handler gives its parameters.

   This file is to be reviewed: the definition and the statements,
   against middle_end/flambda2/simplify/simplify_let_cont_expr.ml and
   middle_end/flambda2/simplify/env/downwards_env.ml
   ([add_parameters_with_unknown_types]). It follows
   FlambdaTypingEnvRefinementSpec.v, whose two statements about a
   handler ([let_cont_handler_refines_under_statement],
   [let_cont_recursive_handler_refines_under_statement]) ask the
   rewrite of the handler under the parameters at every choice of
   kinds: they were written when the machine did not test the kind of
   a value where a handler binds it. It does now
   (FlambdaSemanticsSpec.apply_cont_value): the application
   of a handler has no step unless the arguments are of the kinds that
   the handler gives its parameters. So a function that rewrites a
   handler can do so under one environment, in which each parameter is
   defined at its kind.

   [handler_env tenv params metadata] is that environment: [tenv], the
   environment where the [Let_cont] is, with the parameters defined in
   order at the kinds of [metadata], with no types. The variables of
   the place where the handler is defined are in scope in it, and
   what was known of them there is known: a handler runs under those
   variables, whose values do not change, under a static environment
   that may have grown.

   Where the model and the compiler part:
   - The compiler knows more in a handler than this environment
     does. It rewrites the handler of a non-recursive [Let_cont] after
     the body, under the join of the environments at the uses of the
     continuation, in which a parameter has the join of the types of
     its arguments. For a recursive [Let_cont] it starts from the
     environment at the definition, as here, and gives the parameters
     of each handler unknown types
     ([add_parameters_with_unknown_types]); but the invariant
     parameters of the group have the join of the types of their
     arguments at the uses outside the handlers
     ([prepare_dacc_for_handlers]), an unknown type is of the subkind
     of the parameter and not only of its kind
     ([unknown_with_subkind]), and lifted constants and the unboxing
     of parameters add to the environment. Here there is no join of
     environments and no subkind: what was known at the definition,
     and the kind that the handler declares for each parameter.
   - [handler_env] tests, before it defines each parameter, that the
     environment passes the test of well-formedness and does not have
     the variable (FlambdaSimplifyCodeEnvSpec.define_variables_checked),
     and that the handler gives as many kinds as it has parameters.
     The compiler has no such tests. Where a test fails there is no
     environment, and the handler is to be left as it is.

   The statements:
   - the environment is satisfied where the handler runs: under the
     variables of the definition with the parameters bound to values
     of the kinds of the handler ([handler_env_satisfied_statement]);
   - the handler of a non-recursive [Let_cont] may be rewritten under
     it ([let_cont_handler_typed_statement]);
   - so may one handler of a recursive [Let_cont], if the two
     handlers agree on whether they bind something static, as in
     FlambdaTypingEnvRefinementSpec.v
     ([let_cont_recursive_handler_typed_statement]);
   - [handler_env] commutes with renaming
     ([handler_env_rename_statement]).

   Nothing else is assumed of the term or of the environment than
   what the statements say: [handler_env] answers, so that its tests
   have passed.

   The proofs are in FlambdaSimplifyHandlersProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvExamplesSpec FlambdaTypingEnvRefinementSpec
  FlambdaTypeRenameSpec FlambdaEnvRenameSpec FlambdaSimplifyCodeEnvSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The environment of a handler                                             *)
(* ======================================================================== *)

(* The environment under which a handler is rewritten, where it is
   defined under [tenv]: the parameters are defined at the kinds that
   the handler gives them. [None] where the handler has not as many
   kinds as parameters, or a test of [define_variables_checked] fails:
   the parameters are then not different variables that [tenv] does
   not have, or an environment on the way is not well formed. *)
Definition handler_env (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) : option Typing_env.t :=
  let kinds := hm_params_kinds metadata in
  if Nat.eqb (length kinds) (length params)
  then define_variables_checked tenv (combine params kinds)
  else None.

(* It is satisfied where the handler runs: under the variables of the
   place of its definition, which satisfy [tenv] with the static
   environment, and the parameters bound to values of the kinds of
   the handler. *)
Definition handler_env_satisfied_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (tenv_handler : Typing_env.t)
    (statics : static_env) (vars : env) (vs : list value)
    (handler_vars : env),
    handler_env tenv params metadata = Some tenv_handler ->
    env_satisfies_strictly tenv statics vars ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    env_satisfies_strictly tenv_handler statics handler_vars.

(* ======================================================================== *)
(* A handler rewritten                                                      *)
(* ======================================================================== *)

Definition let_cont_handler_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body : expr)
    (tenv_handler : Typing_env.t),
    handler_env tenv params metadata = Some tenv_handler ->
    expr_refines_under tenv_handler handler handler' ->
    expr_refines_under tenv
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params metadata handler') body).

(* One handler of a recursive [Let_cont]. A handler of a recursive
   [Let_cont] may not bind anything static, so the two handlers are to
   agree on whether they do. *)
Definition let_cont_recursive_handler_typed_statement : Prop :=
  ∀ (tenv : Typing_env.t)
    (before : list (continuation * continuation_handler)) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr)
    (after : list (continuation * continuation_handler)) (body : expr)
    (tenv_handler : Typing_env.t),
    handler_env tenv params metadata = Some tenv_handler ->
    binds_statics handler = binds_statics handler' ->
    expr_refines_under tenv_handler handler handler' ->
    expr_refines_under tenv
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler') :: after) body).

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

(* The environment of the handler made from the renamed environment
   and the renamed parameters is the environment of the handler,
   renamed. The metadata contains no variable. *)
Definition handler_env_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata),
    handler_env (rename_typing_env r tenv) (map (rename_variable r) params)
      metadata
    = option_map (rename_typing_env r) (handler_env tenv params metadata).
