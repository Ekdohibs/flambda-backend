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

(* The statements of FlambdaReaperTypesReturnsSpec.v of the units with the
   primitives of the heap (FlambdaReaperFlowSpec.in_heap_fragment): the
   result types of code at the results of the code, end to end.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesReturnsSpec.v (whose statements these are, with
   FlambdaReaperTypesCallsHeapSpec.heap_premises in place of
   [unit_premises]) and FlambdaReaperTypesCallsHeapSpec.v (the statements
   across calls with the heap). The proofs are in
   FlambdaReaperTypesReturnsHeapProofs.v, which needs no review.

   As in FlambdaReaperTypesCallsHeapSpec.v: the run of the unit and that
   of its rebuild (plain, or with changes of calling convention) are
   related by the simulation of FlambdaReaperRebuildProofs.v with the
   primitives of the heap, which relates a variable with a use bound in
   the first to its value in the second by [rebuilt_value_heap], for a
   correspondence of the locations of the allocations that the rebuild
   keeps. The traversal aliases the node of a result of code to the
   argument of that rank at each return of the code, with the heap as
   without it; what the rebuild relates at the node of the argument it
   relates at that of the result, and the types, which never read the
   heap, hold of a value as of the value with each location erased
   (FlambdaReaperTypesHeapSpec.type_satisfied_erase_statement).

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes), [_local] and [_local_landed] for a
   least solution of the rules with the fix of local fields
   (FlambdaReaperTypesCallsSpec.least_with):
   - [code_result_type_end_to_end_heap_statement]: at each state that the
     run of the unit comes to and from which it steps, at a return of the
     code [id] whose argument of rank [i] is [arg] and has the value [v],
     the type that the result types give [type_var], which holds of [v],
     rewritten under the metadata of the node [r] of the result of rank
     [i] of [id], holds of the value of [arg] at a state that the run of
     the plain rebuild comes to.
   - [cc_code_result_type_end_to_end_heap_statement]: the same of the
     rebuild with changes of calling convention.

   Of the premises: those of FlambdaReaperTypesReturnsUsedSpec.v with
   [heap_premises] (FlambdaReaperTypesCallsHeapSpec.v: [used_premises]
   with [in_heap_fragment] in place of [in_flow_fragment]) in place of
   [used_premises]. No premise is added. The statements of the fragment
   [loose] with the heap are in FlambdaReaperTypesLooseHeapSpec.v. *)

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
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperTypesCallsHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* FlambdaReaperTypesReturnsSpec.code_result_type_end_to_end_with with
   [heap_premises]. *)
Definition code_result_type_end_to_end_heap_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
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

Definition code_result_type_end_to_end_heap_statement : Prop :=
  code_result_type_end_to_end_heap_with false all_fixes.

Definition code_result_type_end_to_end_heap_landed_statement : Prop :=
  code_result_type_end_to_end_heap_with false landed_fixes.

Definition code_result_type_end_to_end_heap_local_statement : Prop :=
  code_result_type_end_to_end_heap_with true all_fixes.

Definition code_result_type_end_to_end_heap_local_landed_statement : Prop :=
  code_result_type_end_to_end_heap_with true landed_fixes.

(* FlambdaReaperTypesReturnsSpec.cc_code_result_type_end_to_end_with with
   [heap_premises]. *)
Definition cc_code_result_type_end_to_end_heap_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
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

Definition cc_code_result_type_end_to_end_heap_statement : Prop :=
  cc_code_result_type_end_to_end_heap_with false all_fixes.

Definition cc_code_result_type_end_to_end_heap_landed_statement : Prop :=
  cc_code_result_type_end_to_end_heap_with false landed_fixes.

Definition cc_code_result_type_end_to_end_heap_local_statement : Prop :=
  cc_code_result_type_end_to_end_heap_with true all_fixes.

Definition cc_code_result_type_end_to_end_heap_local_landed_statement : Prop :=
  cc_code_result_type_end_to_end_heap_with true landed_fixes.
