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

(* The statements of FlambdaReaperTypesReturnsSpec.v with the wider check
   of static blocks of FlambdaReaperTypesUsedValuesSpec.v.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesReturnsSpec.v (whose statements these are, with
   FlambdaReaperTypesCallsUsedSpec.used_premises in place of
   [unit_premises]). The proofs are in
   FlambdaReaperTypesReturnsUsedProofs.v, which needs no review, with
   FlambdaReaperTypesReturnsProofs.v.

   As in FlambdaReaperTypesCallsUsedSpec.v: the premises of the statements
   of FlambdaReaperTypesReturnsSpec.v on the unit ([unit_premises]) ask
   every static block of the unit to hold values ([unit_of_values]); the
   statements read the static values of the symbols with a use only, and
   the simulation of the rebuild relates the static environments from
   those alone. [used_premises] asks it of the static blocks whose symbol
   has a use in the solution only ([used_values]). Each statement of
   FlambdaReaperTypesReturnsSpec.v but the example is the instance of the
   one here
   (FlambdaReaperTypesCallsUsedSpec.used_premises_instance_statement).

   The statements, named with [_used] after the name of their family,
   each with [_landed], [_local] and [_local_landed] as there:
   [code_result_type_end_to_end_used_statement] (of
   [code_result_type_end_to_end_statement], the plain rebuild) and
   [cc_code_result_type_end_to_end_used_statement] (of
   [cc_code_result_type_end_to_end_statement], the rebuild with changes
   of calling convention).

   No premise is added: the check of the static blocks is the only
   premise changed, and the wider check reads the solution. *)

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
  FlambdaReaperTypesCallsUsedSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec
  FlambdaReaperDeadFramesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* FlambdaReaperTypesReturnsSpec.code_result_type_end_to_end_with
   with [used_premises]. *)
Definition code_result_type_end_to_end_used_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    return_premises false flag u e T n s id i arg r tenv canon valuation
      valuation' type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ eval_simple (state_vars t) arg = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition code_result_type_end_to_end_used_statement : Prop :=
  code_result_type_end_to_end_used_with false all_fixes.

Definition code_result_type_end_to_end_used_landed_statement : Prop :=
  code_result_type_end_to_end_used_with false landed_fixes.

Definition code_result_type_end_to_end_used_local_statement : Prop :=
  code_result_type_end_to_end_used_with true all_fixes.

Definition code_result_type_end_to_end_used_local_landed_statement : Prop :=
  code_result_type_end_to_end_used_with true landed_fixes.

(* FlambdaReaperTypesReturnsSpec.cc_code_result_type_end_to_end_with
   with [used_premises]. *)
Definition cc_code_result_type_end_to_end_used_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    return_premises false flag u e T n s id i arg r tenv canon valuation
      valuation' type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ eval_simple (state_vars t) arg = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type_at fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon r type_var).

Definition cc_code_result_type_end_to_end_used_statement : Prop :=
  cc_code_result_type_end_to_end_used_with false all_fixes.

Definition cc_code_result_type_end_to_end_used_landed_statement : Prop :=
  cc_code_result_type_end_to_end_used_with false landed_fixes.

Definition cc_code_result_type_end_to_end_used_local_statement : Prop :=
  cc_code_result_type_end_to_end_used_with true all_fixes.

Definition cc_code_result_type_end_to_end_used_local_landed_statement : Prop :=
  cc_code_result_type_end_to_end_used_with true landed_fixes.
