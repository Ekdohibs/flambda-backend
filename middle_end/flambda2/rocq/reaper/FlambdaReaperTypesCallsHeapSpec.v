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

(* The statements of FlambdaReaperTypesCallsSpec.v of the units with the
   primitives of the heap (FlambdaReaperFlowSpec.in_heap_fragment): the
   exported types of the symbols where the unit rebuilt with unboxing
   across calls exits, and the result types of code at the variables that
   the runs bind, the variables kept and those that the unboxing across
   calls unboxes.

   This file is to be reviewed: [heap_premises] and the statements,
   against FlambdaReaperTypesCallsSpec.v (whose statements these are, with
   [heap_premises] in place of [unit_premises]),
   FlambdaReaperTypesCallsUsedSpec.v ([used_premises]),
   FlambdaReaperTypesHeapSpec.v (the types with the heap) and
   FlambdaReaperUnboxingRefinesCallsHeapSpec.v (the unboxing across calls
   with the heap). The proofs are in FlambdaReaperTypesCallsHeapProofs.v,
   which needs no review.

   The runs. As in FlambdaReaperTypesCallsSpec.v: the run of the unit,
   that of the unit rebuilt with changes of calling convention (the plain
   rebuild where no code changes), and that of the unit rebuilt with
   unboxing across calls. The first two are related by the simulation of
   FlambdaReaperRebuildProofs.v with the primitives of the heap: a
   variable with a use bound in the first is bound in the second to the
   value that [rebuilt_value_heap] relates to it, for the correspondence
   [β] of the locations of the allocations that the rebuild keeps, and
   the static environments are related by [statics_rebuilt_heap]; where
   the first exits, the second exits through the same continuation, the
   exits observed equal up to a correspondence of their locations
   (FlambdaReaperRebuildSpec.exits_observed_equal). The last two are
   related by the simulation of the unboxing across calls ([uc]) with the
   heap, which makes the same primitives of the heap on the same values in
   both: their heaps are one, so the third run exits as the second with
   the same values and heap, and the variables of the fields of an
   unboxed variable hold the values at the paths of the fields in the
   value of the second run, locations included.

   How the types speak of the heap: they never read it. A type holds of a
   value as it holds of the value with each location erased
   (FlambdaReaperTypesHeapSpec.type_satisfied_erase_statement), and the
   statements of types of FlambdaReaperTypesHeapSpec.v give, from the
   relations above, the types rewritten of the values of the second run;
   the third run binds the same static values.

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes), [_local] and [_local_landed] for a
   least solution of the rules with the fix of local fields
   (FlambdaReaperTypesCallsSpec.least_with); two names are shortened,
   [unbox_calls_result_end_to_end_heap_local_landed_statement] and
   [calls_unboxed_result_end_to_end_heap_local_landed_statement]:
   - [heap_premises_instance_statement]: [used_premises] gives
     [heap_premises] (a unit of [in_flow_fragment] is of
     [in_heap_fragment]).
   - [unbox_calls_types_end_to_end_heap_statement]: where the unit exits,
     the unit rebuilt with unboxing across calls exits through the same
     continuation, the exits observed equal up to a correspondence of
     their locations, and the type of each symbol bound, rewritten, holds
     of the symbol there.
   - [result_type_end_to_end_heap_statement] (plain rebuild) and
     [cc_result_type_end_to_end_heap_statement] (rebuild with changes of
     calling convention): at each state that the run of the unit comes to
     and from which it steps, the type that the result types give
     [type_var], which holds of the value of the variable [code_var]
     there, rewritten under the metadata of [code_var], holds of the value
     of [code_var] at a state that the run of the rebuilt unit comes to.
   - [unbox_calls_result_type_end_to_end_heap_statement]: the same of the
     unit rebuilt with unboxing across calls, for a variable of the unit
     that it does not unbox, at a state of the run of the unit at a [Let]
     of a variable with a use, if the run of the unit does not go wrong;
     [calls_unboxed_result_type_end_to_end_heap_statement]: for one that
     it unboxes, the types that [rewrite_unboxed_result_type] gives its
     fields hold of the values of the variables of the fields.
   - [calls_heap_types_example_statement]: the unit of
     FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample, whose
     code takes a block holding a cell, of [in_heap_fragment] and not of
     [in_flow_fragment], meets the premises of
     [calls_unboxed_result_type_end_to_end_heap_statement] with the
     unboxed parameter of the code, whose field holds the location of the
     cell.

   Of the premises: those of FlambdaReaperTypesCallsUsedSpec.v
   ([used_premises]) with [in_heap_fragment] in place of
   [in_flow_fragment] ([heap_premises]), and those of
   FlambdaReaperTypesCallsSpec.v on the tables ([calls_premises]), the
   states ([result_premises]) and the third run ([third_run_premises]).
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
  FlambdaReaperTypesUsedValuesSpec FlambdaReaperTypesCallsUsedSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsHeapSpec
  FlambdaReaperCallingConventionsSpec FlambdaRenameApartSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsUsedSpec.used_premises with [in_heap_fragment]
   in place of [in_flow_fragment]. *)
Definition heap_premises (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) : Prop :=
  is_local_slots_only is_local
  ∧ least_with local is_local (traverse_unit flag u e) T e
  ∧ in_heap_fragment e = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments e = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] e = true
  ∧ well_formed_unit u e = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value u e = true
  ∧ variables_bound_once e = true
  ∧ used_values T e = true
  ∧ sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets.

Definition heap_premises_instance_statement : Prop :=
  ∀ (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures),
    used_premises local flag is_local u e T sets ->
    heap_premises local flag is_local u e T sets.

(* ======================================================================== *)
(* The exported types                                                       *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.unbox_calls_types_end_to_end_with with
   [heap_premises]: the exits are observed equal up to a correspondence of
   their locations. *)
Definition unbox_calls_types_end_to_end_heap_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    heap_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unbox_calls_types_end_to_end_heap_statement : Prop :=
  unbox_calls_types_end_to_end_heap_with false all_fixes.

Definition unbox_calls_types_end_to_end_heap_landed_statement : Prop :=
  unbox_calls_types_end_to_end_heap_with false landed_fixes.

Definition unbox_calls_types_end_to_end_heap_local_statement : Prop :=
  unbox_calls_types_end_to_end_heap_with true all_fixes.

Definition unbox_calls_types_end_to_end_heap_local_landed_statement : Prop :=
  unbox_calls_types_end_to_end_heap_with true landed_fixes.

(* ======================================================================== *)
(* The result types of code                                                 *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.result_type_end_to_end_with with
   [heap_premises]. *)
Definition result_type_end_to_end_heap_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition result_type_end_to_end_heap_statement : Prop :=
  result_type_end_to_end_heap_with false all_fixes.

Definition result_type_end_to_end_heap_landed_statement : Prop :=
  result_type_end_to_end_heap_with false landed_fixes.

Definition result_type_end_to_end_heap_local_statement : Prop :=
  result_type_end_to_end_heap_with true all_fixes.

Definition result_type_end_to_end_heap_local_landed_statement : Prop :=
  result_type_end_to_end_heap_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.cc_result_type_end_to_end_with with
   [heap_premises]. *)
Definition cc_result_type_end_to_end_heap_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition cc_result_type_end_to_end_heap_statement : Prop :=
  cc_result_type_end_to_end_heap_with false all_fixes.

Definition cc_result_type_end_to_end_heap_landed_statement : Prop :=
  cc_result_type_end_to_end_heap_with false landed_fixes.

Definition cc_result_type_end_to_end_heap_local_statement : Prop :=
  cc_result_type_end_to_end_heap_with true all_fixes.

Definition cc_result_type_end_to_end_heap_local_landed_statement : Prop :=
  cc_result_type_end_to_end_heap_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.unbox_calls_result_type_end_to_end_with
   with [heap_premises]. *)
Definition unbox_calls_result_type_end_to_end_heap_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value),
    heap_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    third_run_premises (traverse_unit flag u e) T e s code_var ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              (traverse_unit flag u e) T dominated to_unbox e))
    = None ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition unbox_calls_result_type_end_to_end_heap_statement : Prop :=
  unbox_calls_result_type_end_to_end_heap_with false all_fixes.

Definition unbox_calls_result_type_end_to_end_heap_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_heap_with false landed_fixes.

Definition unbox_calls_result_type_end_to_end_heap_local_statement : Prop :=
  unbox_calls_result_type_end_to_end_heap_with true all_fixes.

Definition unbox_calls_result_end_to_end_heap_local_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_heap_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.calls_unboxed_result_type_end_to_end_with
   with [heap_premises]. *)
Definition calls_unboxed_result_type_end_to_end_heap_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v : value),
    heap_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    third_run_premises (traverse_unit flag u e) T e s code_var ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              (traverse_unit flag u e) T dominated to_unbox e))
    = Some fields ->
    ∃ (m : nat) (t : state) (ws : list value),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ Forall2 (fun (xk : variable * flambda_kind) (w : value) =>
                   lookup (fst xk) (state_vars t) = Some w)
          (FlambdaReaperUnboxing.fields_leaves fields) ws
      ∧ Forall2 (type_satisfied (state_statics t) valuation') ws
          (rewrite_unboxed_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             (FlambdaReaperUnboxing.bound_code_my_closures e) canon code_var
             type_var fields).

Definition calls_unboxed_result_type_end_to_end_heap_statement : Prop :=
  calls_unboxed_result_type_end_to_end_heap_with false all_fixes.

Definition calls_unboxed_result_type_end_to_end_heap_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_heap_with false landed_fixes.

Definition calls_unboxed_result_type_end_to_end_heap_local_statement : Prop :=
  calls_unboxed_result_type_end_to_end_heap_with true all_fixes.

Definition calls_unboxed_result_end_to_end_heap_local_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_heap_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample:

     code 50 (x40) =
       let x43 = x40.(0) in
       let x44 = x43.(0) <- 2 in
       return x44
     let x10 = mutable { 1 } in
     let x11 = [x10] in
     let x60 = closure 50 in
     let k20 (x30) = let x31 = x10.(0) in return x31 in
     x60 x11 -> k20

   with its tables, which are the least solution. The rules unbox the
   parameter [x40] into the variable 61 of its field 0, which holds the
   location of the cell. The result types give the parameter, the
   variable 90 of the result types, the type of a block of tag 0 whose
   field 0 is unknown, which holds of the block [x11]. *)
Module CallsHeapTypesExample.

Local Open Scope positive_scope.

Module CH := FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample.

(* The set of closures of [x60]. *)
Definition sets : list set_of_closures :=
  [[(70, (Node_variable 60, Or_unknown.Known 50))]].

(* The unboxed fields of [x40]. *)
Definition fields : FlambdaReaperUnboxing.unboxed_fields :=
  [(Field_block 0 Value, FlambdaReaperUnboxing.Not_unboxed 61)].

Definition param_type : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [unknown Value].

(* The result types: the variable 90 of the type [param_type]. *)
Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create 90 Value
       Name_mode.normal)
    (Name_var 90) param_type.

(* The block [x11], holding the location 2 of the cell. *)
Definition block : value := Value_block 0 [Value_location 2].

Definition valuation : env := [(90, block)].

Definition canon : canon_fn := fun n _ _ => n.

End CallsHeapTypesExample.

(* The unit meets the premises of
   [calls_unboxed_result_type_end_to_end_heap_statement], with the tables
   of the least solution, no local field, the set of closures of [x60],
   no code taken to never return, and, at the state of the run of the
   unit at the [Let] of [x43] in the code (6 steps), the parameter [x40]
   bound to the block of the location 2, the variable 90 of the result
   types at that block; it is of [in_heap_fragment] and not of
   [in_flow_fragment]; the rules unbox [x40] into the variable 61 of its
   field 0; the rebuilt unit is
   FlambdaReaperUnboxingRefinesCallsHeapSpec.CallsHeapExample.rebuilt.
   The type of the field, rewritten, is unknown; the run of the rebuilt
   unit, at the [Let] of [x43] in the code (6 steps), binds 61 to the
   location 2, of which it holds. *)
Definition calls_heap_types_example_with (fx : fixes) : Prop :=
  heap_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface CallsHeapTypesExample.CH.unit_expr
    CallsHeapTypesExample.CH.tables CallsHeapTypesExample.sets
  ∧ in_flow_fragment CallsHeapTypesExample.CH.unit_expr = false
  ∧ calls_premises Auto FlambdaReaperUnboxing.Examples.interface
      CallsHeapTypesExample.CH.unit_expr CallsHeapTypesExample.CH.tables
      CallsHeapTypesExample.CH.cannot_change
      CallsHeapTypesExample.CH.dominated CallsHeapTypesExample.CH.to_unbox
      (fun _ => false)
  ∧ Keyed.lookup node_eqb (Node_variable 40%positive)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              CallsHeapTypesExample.CH.facts CallsHeapTypesExample.CH.tables
              CallsHeapTypesExample.CH.dominated
              CallsHeapTypesExample.CH.to_unbox
              CallsHeapTypesExample.CH.unit_expr))
    = Some CallsHeapTypesExample.fields
  ∧ calls_rebuilt Auto FlambdaReaperUnboxing.Examples.interface
      CallsHeapTypesExample.CH.unit_expr CallsHeapTypesExample.CH.tables
      CallsHeapTypesExample.CH.cannot_change
      CallsHeapTypesExample.CH.dominated CallsHeapTypesExample.CH.to_unbox
      true
    = CallsHeapTypesExample.CH.rebuilt
  ∧ (∃ s : state,
       result_premises CallsHeapTypesExample.CH.facts
         CallsHeapTypesExample.CH.tables CallsHeapTypesExample.CH.unit_expr 6
         s CallsHeapTypesExample.tenv CallsHeapTypesExample.canon
         CallsHeapTypesExample.valuation CallsHeapTypesExample.valuation
         40%positive 90%positive CallsHeapTypesExample.block
       ∧ third_run_premises CallsHeapTypesExample.CH.facts
           CallsHeapTypesExample.CH.tables CallsHeapTypesExample.CH.unit_expr
           s 40%positive)
  ∧ rewrite_unboxed_result_type fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         CallsHeapTypesExample.CH.facts CallsHeapTypesExample.CH.tables
         CallsHeapTypesExample.sets)
      CallsHeapTypesExample.tenv
      (FlambdaReaperUnboxing.bound_code_my_closures
         CallsHeapTypesExample.CH.unit_expr)
      CallsHeapTypesExample.canon 40%positive 90%positive
      CallsHeapTypesExample.fields
    = [unknown Value]
  ∧ ∃ t : state,
      run 6 (initial_state CallsHeapTypesExample.CH.rebuilt) = Step t
      ∧ lookup 61%positive (state_vars t) = Some (Value_location 2%positive)
      ∧ type_satisfied (state_statics t) CallsHeapTypesExample.valuation
          (Value_location 2%positive) (unknown Value).

Definition calls_heap_types_example_statement : Prop :=
  calls_heap_types_example_with all_fixes.

Definition calls_heap_types_example_landed_statement : Prop :=
  calls_heap_types_example_with landed_fixes.
