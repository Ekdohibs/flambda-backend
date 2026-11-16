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


(* The function that rewrites from the types and removes arms of a
   [Switch] commutes with renaming.

   This file is to be reviewed: what is asked of the function that
   rewrites a [Switch] ([switch_rewrite_equivariant]), and the
   statements.

   [simplify_expr_pruning] (FlambdaSimplifyPruningSpec.v) takes a
   function that may rewrite a [Switch], a typing environment, a
   boolean and a term. The statement
   ([simplify_expr_pruning_equivariant_statement]) is that of the first
   function (FlambdaSimplifyPassRenamingSpec.v): given the renamed
   environment and the renamed term, with the same boolean, it answers
   the renamed answer. It has one premise, on the function that
   rewrites a [Switch]: that it commutes with renaming in the same
   form ([switch_rewrite_equivariant]). The function that removes the
   arms which a meet excludes does, for a meet that commutes with
   renaming ([simplify_switch_rewrite_equivariant_statement], from
   FlambdaSimplifySwitchSpec.simplify_switch_rename_statement); the
   meet that the contract is proved of, with the join that forgets,
   is such a meet (FlambdaMeetRenameSpec.v). So [prune_by_types] is a
   rewrite that commutes with renaming in the form of the other
   rewrites of the validation, with no premise
   ([prune_by_types_equivariant_statement], with
   FlambdaValidatorEquivarianceSpec.rewrite_equivariant).

   The second function makes the tests of the first and four more:
   those by which its boolean stays true at a [Let]
   ([let_keeps_fit]) and at a [Let_static] ([statics_keep_fit]), and
   those made at a [Switch] before the function that rewrites it is
   asked ([types_fit_b], and that the scrutinee may occur in terms).
   Each answers the same of the renamed environment and term
   ([types_fit_b_equivariant_statement],
   [alias_keeps_fit_equivariant_statement],
   [equation_keeps_fit_equivariant_statement],
   [let_keeps_fit_equivariant_statement],
   [statics_keep_fit_equivariant_statement]): a test reads kinds,
   whether a name is bound or canonical, whether a type is an alias,
   is unknown or is bottom, and none of these changes when variables
   are permuted.

   Where the function rewrites the handler of a continuation under
   the environment of its one application, its boolean stays true by
   the tests of the section "Where a continuation is applied" of
   FlambdaSimplifyPruningSpec.v, one for each function that builds
   that environment. Each answers the same of the renamed
   environments and terms ([arm_env_keeps_fit_equivariant_statement],
   [use_env_keeps_fit_equivariant_statement],
   [leaf_uses_keep_fit_equivariant_statement],
   [let_env_keeps_fit_equivariant_statement],
   [find_use_keeps_fit_equivariant_statement]), and so do the test
   for the applications that are gathered
   ([gather_uses_keeps_fit_equivariant_statement]) and the test for
   the one application under whose environment the function rewrites
   the handler ([find_one_use_keeps_fit_equivariant_statement]); that
   the functions themselves commute with renaming is in
   FlambdaSimplifyPassRenamingSpec.v.

   Nothing is trusted here. The proofs are in
   FlambdaSimplifyPruningRenamingProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaMeet FlambdaMeetSpec FlambdaTypeRenameSpec FlambdaEnvRenameSpec
  FlambdaMeetRenameSpec FlambdaSimplifyPrimitivesSpec FlambdaSimplifyPassSpec
  FlambdaSimplifyPruningSpec.
From OxCamlFlambda2Types Require FlambdaSimplifySwitchSpec.
Import ListNotations.

(* ======================================================================== *)
(* A function that rewrites a [Switch]                                      *)
(* ======================================================================== *)

(* Given the renamed environment, scrutinee and arms, it answers the
   renamed answer, and no answer where it gave none. The arms are
   renamed as [rename_expr] renames those of a [Switch]
   (FlambdaSimplifySwitchSpec.rename_arms). *)
Definition switch_rewrite_equivariant (prune : switch_rewrite) : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple)
    (arms : list (Z * apply_cont)),
    prune (rename_typing_env r tenv) (rename_simple r scrutinee)
      (FlambdaSimplifySwitchSpec.rename_arms r arms)
    = option_map (rename_expr r) (prune tenv scrutinee arms).

(* The function that removes the arms which a meet excludes is such a
   function, for a meet that commutes with renaming. *)
Definition simplify_switch_rewrite_equivariant_statement : Prop :=
  ∀ meet : meet_function,
    meet_function_equivariant meet ->
    switch_rewrite_equivariant
      (FlambdaSimplifySwitchSpec.simplify_switch meet).

(* So is the function that rewrites no [Switch]. *)
Definition no_pruning_equivariant_statement : Prop :=
  switch_rewrite_equivariant no_pruning.

(* ======================================================================== *)
(* The tests                                                                *)
(* ======================================================================== *)

Definition types_fit_b_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    types_fit_b (rename_typing_env r tenv) = types_fit_b tenv.

Definition alias_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (e1 e2 : simple)
    (kind : flambda_kind),
    alias_keeps_fit (rename_typing_env r tenv) (rename_simple r e1)
      (rename_simple r e2) kind
    = alias_keeps_fit tenv e1 e2 kind.

Definition equation_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    equation_keeps_fit (rename_typing_env r tenv) (rename_name r n)
      (rename_type r ty)
    = equation_keeps_fit tenv n ty.

Definition let_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable)
    (defining_expr : named),
    let_keeps_fit (rename_typing_env r tenv) (rename_variable r x)
      (rename_named r defining_expr)
    = let_keeps_fit tenv x defining_expr.

Definition statics_keep_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (bindings : list static_binding),
    statics_keep_fit (rename_typing_env r tenv)
      (map (rename_static_binding r) bindings)
    = statics_keep_fit tenv bindings.

(* ======================================================================== *)
(* Where a continuation is applied                                          *)
(* ======================================================================== *)

(* The tests of the section "Where a continuation is applied" of
   FlambdaSimplifyPruningSpec.v answer the same of the renamed
   environments and terms. *)

Definition arm_env_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (scrutinee : simple) (d : Z),
    arm_env_keeps_fit (rename_typing_env r tenv) (rename_simple r scrutinee) d
    = arm_env_keeps_fit tenv scrutinee d.

Definition use_env_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope tenv : Typing_env.t) (params : list variable)
    (metadata : handler_metadata) (args : list simple),
    use_env_keeps_fit (rename_typing_env r scope) (rename_typing_env r tenv)
      (map (rename_variable r) params) metadata (map (rename_simple r) args)
    = use_env_keeps_fit scope tenv params metadata args.

Definition leaf_uses_keep_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation) (e : expr),
    leaf_uses_keep_fit (rename_typing_env r tenv) (rename_continuation r k)
      (rename_expr r e)
    = leaf_uses_keep_fit tenv k e.

Definition let_env_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable)
    (declared : flambda_kind) (defining_expr : named),
    let_env_keeps_fit (rename_typing_env r tenv) (rename_variable r x)
      declared (rename_named r defining_expr)
    = let_env_keeps_fit tenv x declared defining_expr.

Definition find_use_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    find_use_keeps_fit (rename_typing_env r scope) (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = find_use_keeps_fit scope tenv k h body.

Definition gather_uses_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    gather_uses_keeps_fit (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = gather_uses_keeps_fit tenv k h body.

Definition find_one_use_keeps_fit_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (k : continuation)
    (h : continuation_handler) (body : expr),
    find_one_use_keeps_fit (rename_typing_env r tenv)
      (rename_continuation r k) (rename_handler r h) (rename_expr r body)
    = find_one_use_keeps_fit tenv k h body.

(* ======================================================================== *)
(* The function                                                             *)
(* ======================================================================== *)

Definition simplify_switch_pruning_equivariant_statement : Prop :=
  ∀ (prune : switch_rewrite) (r : renaming) (tenv : Typing_env.t)
    (fit : bool) (scrutinee : simple) (arms : list (Z * apply_cont)),
    switch_rewrite_equivariant prune ->
    simplify_switch_pruning prune (rename_typing_env r tenv) fit
      (rename_simple r scrutinee)
      (FlambdaSimplifySwitchSpec.rename_arms r arms)
    = rename_expr r (simplify_switch_pruning prune tenv fit scrutinee arms).

Definition simplify_expr_pruning_equivariant_statement : Prop :=
  ∀ (prune : switch_rewrite) (r : renaming) (tenv : Typing_env.t)
    (fit : bool) (e : expr),
    switch_rewrite_equivariant prune ->
    simplify_expr_pruning prune (rename_typing_env r tenv) fit
      (rename_expr r e)
    = rename_expr r (simplify_expr_pruning prune tenv fit e).

(* From the environment that knows nothing it is a rewrite of terms
   that commutes with renaming. *)
Definition simplify_by_types_pruning_equivariant_statement : Prop :=
  ∀ prune : switch_rewrite,
    switch_rewrite_equivariant prune ->
    rewrite_equivariant (simplify_by_types_pruning prune).

(* With the meet that the contract is proved of, for a join that
   commutes with renaming; and with the join that forgets, where
   nothing is asked. *)
Definition simplify_by_types_pruning_with_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_equivariant join ->
    rewrite_equivariant (simplify_by_types_pruning_with join fuel).

Definition prune_by_types_equivariant_statement : Prop :=
  ∀ fuel : nat, rewrite_equivariant (prune_by_types fuel).
