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

(* The statements of FlambdaReaperTypesParamsTraversalSpec.v (the result
   types of code at the parameters of the code, at the nodes of
   [code_dep.params]) of the units with the primitives of the heap
   (FlambdaReaperFlowSpec.in_heap_fragment).

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesParamsSpec.v and
   FlambdaReaperTypesParamsTraversalSpec.v (whose statements these are,
   with FlambdaReaperTypesCallsHeapSpec.heap_premises in place of
   [used_premises]), and the example. The proofs are in
   FlambdaReaperTypesParamsHeapProofs.v, which needs no review.

   As in FlambdaReaperTypesCallsHeapSpec.v: the run of the unit and that
   of its rebuild are related with the primitives of the heap, a variable
   with a use bound in the first to its value in the second by
   [rebuilt_value_heap]. The traversal of a unit of [in_heap_fragment]
   whose variables are bound once passes the check
   FlambdaReaperTypesParamsSpec.params_aliased
   (FlambdaReaperTypesParamsTraversalSpec
   .params_aliased_traversal_statement), so a value rebuilt at the node of
   a parameter is rebuilt under the metadata of the node of its rank of
   the code dep, with the heap as without it; the types never read the
   heap.

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes), [_local] and [_local_landed] for a
   least solution of the rules with the fix of local fields
   (FlambdaReaperTypesCallsSpec.least_with):
   - [param_type_end_to_end_heap_statement]: at each state that the run of
     the unit comes to and from which it steps, where the parameter [p] of
     rank [i] of the code [id] has the value [v], the type that the result
     types give [type_var], which holds of [v], rewritten under the
     metadata of the node [r] of rank [i] of the parameters of the code
     dep of [id], holds of the value of [p] at a state that the run of the
     plain rebuild comes to.
   - [cc_param_type_end_to_end_heap_statement]: the same of the rebuild
     with changes of calling convention.
   - [param_heap_example_statement] (and [_landed]): the unit of
     FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample, whose
     code 50 takes the block [x40] that holds a cell, of
     [in_heap_fragment] and not of [in_flow_fragment]. The node of its
     parameter is the made node 6; the result types give the parameter
     the type of a block of tag 0 whose field 0 is unknown
     (FlambdaReaperTypesCallsHeapSpec.CallsHeapTypesExample), which,
     rewritten under the metadata of the node, is itself; the run of the
     plain rebuild binds [x40] to the block of the location 2 of the cell
     at the [Let] of [x43] in the code (7 steps), of which it holds.

   Of the premises: those of FlambdaReaperTypesParamsTraversalSpec.v with
   [heap_premises] in place of [used_premises]. No premise is added. *)

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
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperTypesParamsSpec
  FlambdaReaperTypesCallsHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* FlambdaReaperTypesParamsTraversalSpec.param_type_end_to_end_traversal_with
   with [heap_premises]. *)
Definition param_type_end_to_end_heap_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
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

Definition param_type_end_to_end_heap_statement : Prop :=
  param_type_end_to_end_heap_with false all_fixes.

Definition param_type_end_to_end_heap_landed_statement : Prop :=
  param_type_end_to_end_heap_with false landed_fixes.

Definition param_type_end_to_end_heap_local_statement : Prop :=
  param_type_end_to_end_heap_with true all_fixes.

Definition param_type_end_to_end_heap_local_landed_statement : Prop :=
  param_type_end_to_end_heap_with true landed_fixes.

(* FlambdaReaperTypesParamsTraversalSpec
   .cc_param_type_end_to_end_traversal_with with [heap_premises]. *)
Definition cc_param_type_end_to_end_heap_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (p : variable)
    (r : node) (type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
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

Definition cc_param_type_end_to_end_heap_statement : Prop :=
  cc_param_type_end_to_end_heap_with false all_fixes.

Definition cc_param_type_end_to_end_heap_landed_statement : Prop :=
  cc_param_type_end_to_end_heap_with false landed_fixes.

Definition cc_param_type_end_to_end_heap_local_statement : Prop :=
  cc_param_type_end_to_end_heap_with true all_fixes.

Definition cc_param_type_end_to_end_heap_local_landed_statement : Prop :=
  cc_param_type_end_to_end_heap_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample
   (FlambdaReaperTypesCallsHeapSpec.CallsHeapTypesExample):

     code 50 (x40) =
       let x43 = x40.(0) in
       let x44 = x43.(0) <- 2 in
       return x44
     let x10 = mutable { 1 } in
     let x11 = [x10] in
     let x60 = closure 50 in
     let k20 (x30) = let x31 = x10.(0) in return x31 in
     x60 x11 -> k20

   meets the premises of [param_type_end_to_end_heap_statement] with the
   tables of the least solution, no local field and the set of closures
   of [x60]: the parameter [x40] of rank 0 of the code 50 has the node 6
   of the code dep, and, at the state of the run of the unit at the [Let]
   of [x43] in the code (6 steps), it is bound to the block of the
   location 2 of the cell, as is the variable 90 of the result types,
   whose type is that of a block of tag 0 whose field 0 is unknown. The
   unit is of [in_heap_fragment] and not of [in_flow_fragment]. The type
   rewritten under the metadata of the node is the type itself, and the
   run of the plain rebuild, at the [Let] of [x43] in the code (7 steps:
   the rebuilt call is under a handler), binds [x40] to the same block,
   of which it holds. *)
Definition param_heap_example_with (fx : fixes) : Prop :=
  heap_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface CallsHeapTypesExample.CH.unit_expr
    CallsHeapTypesExample.CH.tables CallsHeapTypesExample.sets
  ∧ in_flow_fragment CallsHeapTypesExample.CH.unit_expr = false
  ∧ param_node Auto FlambdaReaperUnboxing.Examples.interface
      CallsHeapTypesExample.CH.unit_expr 50%positive 0
    = Some (Node_made 6)
  ∧ (∃ s : state,
       param_premises Auto FlambdaReaperUnboxing.Examples.interface
         CallsHeapTypesExample.CH.unit_expr CallsHeapTypesExample.CH.tables 6
         s 50%positive 0 40%positive (Node_made 6) CallsHeapTypesExample.tenv
         CallsHeapTypesExample.canon CallsHeapTypesExample.valuation
         CallsHeapTypesExample.valuation 90%positive
         CallsHeapTypesExample.block)
  ∧ rewrite_result_type_at fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         CallsHeapTypesExample.CH.facts CallsHeapTypesExample.CH.tables
         CallsHeapTypesExample.sets)
      CallsHeapTypesExample.tenv CallsHeapTypesExample.canon (Node_made 6)
      90%positive
    = CallsHeapTypesExample.param_type
  ∧ ∃ t : state,
      run 7
        (initial_state
           (rebuild_unit_from_tables CallsHeapTypesExample.CH.facts
              CallsHeapTypesExample.CH.tables true
              CallsHeapTypesExample.CH.unit_expr))
      = Step t
      ∧ lookup 40%positive (state_vars t) = Some CallsHeapTypesExample.block
      ∧ type_satisfied (state_statics t) CallsHeapTypesExample.valuation
          CallsHeapTypesExample.block CallsHeapTypesExample.param_type.

Definition param_heap_example_statement : Prop :=
  param_heap_example_with all_fixes.

Definition param_heap_example_landed_statement : Prop :=
  param_heap_example_with landed_fixes.
