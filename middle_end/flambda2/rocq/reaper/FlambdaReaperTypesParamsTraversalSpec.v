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

(* The check of FlambdaReaperTypesParamsSpec.v ([params_aliased]) holds of
   the traversal of every unit of the fragment whose variables are bound
   once, and the statements of the result types at the parameters of code
   without it.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesParamsSpec.v (the check, and the statements whose
   premise it was). The proofs are in
   FlambdaReaperTypesParamsTraversalProofs.v, which needs no review.

   Why the traversal passes the check. The traversal puts something into
   a parameter [p] of the code of the unit only by the alias of
   [traverse_code] from the node [r] of its rank of the code dep
   (FlambdaReaperTraversal.traverse_code): the other facts that put
   something into a variable put it into a variable that the unit binds
   by a [Let], into a parameter of a handler (of the unit or of the body
   of code) or into the [my_closure] of code, none of which is [p] where
   the variables of the unit are bound once. The node [r] is made by
   [prepare_code], which puts the arguments of the calls into it through
   the witnesses of the calls (facts [parameter], and the reads from the
   tuple of a tupled call), and nothing else reads it but that alias: the
   facts of the steps of the traversal read only nodes that are not made,
   nodes made in the step, the parameters of the continuations in scope,
   and the nodes of the code deps that are not parameters. The code of the
   fragment binds no static constant in its body
   (FlambdaReaperFlowSpec.code_flow_fragment), so each code is traversed
   once, at the code dep of its code id, which the unit binds once.

   The statements:
   - [params_aliased_traversal_statement]: the traversal of a unit of
     [in_heap_fragment] (so of [in_flow_fragment]) whose variables are
     bound once passes [params_aliased].
   - [param_type_end_to_end_traversal_statement] and
     [cc_param_type_end_to_end_traversal_statement], each with
     [_landed], [_local] and [_local_landed]: the statements
     FlambdaReaperTypesParamsSpec.param_type_end_to_end_statement and
     [cc_param_type_end_to_end_statement] (and their forms) without the
     premise [params_aliased], which their other premises give
     ([used_premises]: [in_flow_fragment] and [variables_bound_once]).

   No premise is added. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperCallingConventions FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesReturnsSpec FlambdaReaperTypesUsedValuesSpec
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperTypesParamsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The check, of the traversal                                              *)
(* ======================================================================== *)

Definition params_aliased_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    in_heap_fragment e = true ->
    variables_bound_once e = true ->
    params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
    = true.

(* ======================================================================== *)
(* The statements without the check                                         *)
(* ======================================================================== *)

(* FlambdaReaperTypesParamsSpec.param_type_end_to_end_with without the
   premise [params_aliased]. *)
Definition param_type_end_to_end_traversal_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    param_premises flag u e T n s id i p r tenv canon valuation valuation'
      type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ lookup p (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition param_type_end_to_end_traversal_statement : Prop :=
  param_type_end_to_end_traversal_with false all_fixes.

Definition param_type_end_to_end_traversal_landed_statement : Prop :=
  param_type_end_to_end_traversal_with false landed_fixes.

Definition param_type_end_to_end_traversal_local_statement : Prop :=
  param_type_end_to_end_traversal_with true all_fixes.

Definition param_type_end_to_end_traversal_local_landed_statement : Prop :=
  param_type_end_to_end_traversal_with true landed_fixes.

(* FlambdaReaperTypesParamsSpec.cc_param_type_end_to_end_with without the
   premise [params_aliased]. *)
Definition cc_param_type_end_to_end_traversal_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    param_premises flag u e T n s id i p r tenv canon valuation valuation'
      type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ lookup p (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition cc_param_type_end_to_end_traversal_statement : Prop :=
  cc_param_type_end_to_end_traversal_with false all_fixes.

Definition cc_param_type_end_to_end_traversal_landed_statement : Prop :=
  cc_param_type_end_to_end_traversal_with false landed_fixes.

Definition cc_param_type_end_to_end_traversal_local_statement : Prop :=
  cc_param_type_end_to_end_traversal_with true all_fixes.

Definition cc_param_type_end_to_end_traversal_local_landed_statement : Prop :=
  cc_param_type_end_to_end_traversal_with true landed_fixes.
