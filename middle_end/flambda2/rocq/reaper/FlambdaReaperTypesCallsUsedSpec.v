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

(* The statements of FlambdaReaperTypesCallsSpec.v with the wider check of
   static blocks of FlambdaReaperTypesUsedValuesSpec.v.

   This file is to be reviewed: [used_premises] and the statements,
   against FlambdaReaperTypesCallsSpec.v (whose statements these are, with
   [used_premises] in place of [unit_premises]) and
   FlambdaReaperTypesUsedValuesSpec.v (the check [used_values]). The
   proofs are in FlambdaReaperTypesCallsUsedProofs.v, which needs no
   review, with FlambdaReaperTypesCallsProofs.v.

   The premises of the statements of FlambdaReaperTypesCallsSpec.v on the
   unit ([unit_premises]) ask every static block of the unit to hold
   values ([unit_of_values]). The statements read the static values of
   the symbols with a use only (FlambdaReaperTypesSpec.statics_rebuilt),
   and the rebuild keeps the binding of a static block exactly where its
   symbol has a use. [used_premises] asks it of the static blocks whose
   symbol has a use in the solution only ([used_values]), as
   FlambdaReaperTypesUsedValuesSpec.v does for the statements of
   FlambdaReaperTypesTermSpec.v: the simulations of the rebuilds relate
   the static environments by [statics_rebuilt] from the static values of
   the symbols with a use alone.

   The statements:
   - [used_premises_instance_statement]: [unit_premises] gives
     [used_premises] (a unit that passes [unit_of_values] passes
     [used_values]), so that each statement of
     FlambdaReaperTypesCallsSpec.v is an instance of the one here.
   - Each statement of FlambdaReaperTypesCallsSpec.v but the example,
     with [used_premises] in place of [unit_premises], named with [_used]
     after the name of its family, each with [_landed], [_local] and
     [_local_landed] as there:
     [unbox_calls_types_end_to_end_used_statement],
     [repr_calls_types_end_to_end_used_statement],
     [result_type_end_to_end_used_statement],
     [cc_result_type_end_to_end_used_statement],
     [unbox_calls_result_type_end_to_end_used_statement] and
     [calls_unboxed_result_type_end_to_end_used_statement]; the
     [_local_landed] forms of the last two are shortened as there,
     [unbox_calls_result_end_to_end_used_local_landed_statement] and
     [calls_unboxed_result_end_to_end_used_local_landed_statement].

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
  FlambdaReaperTypesUsedValuesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperCallingConventionsSpec FlambdaRenameApartSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.unit_premises with the wider check of the
   static blocks ([used_values]) in place of [unit_of_values]. *)
Definition used_premises (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) : Prop :=
  is_local_slots_only is_local
  ∧ least_with local is_local (traverse_unit flag u e) T e
  ∧ in_flow_fragment e = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments e = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] e = true
  ∧ well_formed_unit u e = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value u e = true
  ∧ variables_bound_once e = true
  ∧ used_values T e = true
  ∧ sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets.

Definition used_premises_instance_statement : Prop :=
  ∀ (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures),
    unit_premises local flag is_local u e T sets ->
    used_premises local flag is_local u e T sets.

(* ======================================================================== *)
(* The exported types                                                       *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.unbox_calls_types_end_to_end_with with
   [used_premises]. *)
Definition unbox_calls_types_end_to_end_used_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    used_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unbox_calls_types_end_to_end_used_statement : Prop :=
  unbox_calls_types_end_to_end_used_with false all_fixes.

Definition unbox_calls_types_end_to_end_used_landed_statement : Prop :=
  unbox_calls_types_end_to_end_used_with false landed_fixes.

Definition unbox_calls_types_end_to_end_used_local_statement : Prop :=
  unbox_calls_types_end_to_end_used_with true all_fixes.

Definition unbox_calls_types_end_to_end_used_local_landed_statement : Prop :=
  unbox_calls_types_end_to_end_used_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.repr_calls_types_end_to_end_with with
   [used_premises]. *)
Definition repr_calls_types_end_to_end_used_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    used_premises local flag is_local u e T sets ->
    repr_premises flag is_local u e T cannot_change dominated to_unbox tcr nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (initial_state
           (repr_rebuilt flag is_local u e T cannot_change dominated to_unbox
              tcr preserve_direct_calls))
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition repr_calls_types_end_to_end_used_statement : Prop :=
  repr_calls_types_end_to_end_used_with false all_fixes.

Definition repr_calls_types_end_to_end_used_landed_statement : Prop :=
  repr_calls_types_end_to_end_used_with false landed_fixes.

Definition repr_calls_types_end_to_end_used_local_statement : Prop :=
  repr_calls_types_end_to_end_used_with true all_fixes.

Definition repr_calls_types_end_to_end_used_local_landed_statement : Prop :=
  repr_calls_types_end_to_end_used_with true landed_fixes.

(* ======================================================================== *)
(* The result types of code                                                 *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.result_type_end_to_end_with with
   [used_premises]. *)
Definition result_type_end_to_end_used_with
    (local : bool) (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
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

Definition result_type_end_to_end_used_statement : Prop :=
  result_type_end_to_end_used_with false all_fixes.

Definition result_type_end_to_end_used_landed_statement : Prop :=
  result_type_end_to_end_used_with false landed_fixes.

Definition result_type_end_to_end_used_local_statement : Prop :=
  result_type_end_to_end_used_with true all_fixes.

Definition result_type_end_to_end_used_local_landed_statement : Prop :=
  result_type_end_to_end_used_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.cc_result_type_end_to_end_with with
   [used_premises]. *)
Definition cc_result_type_end_to_end_used_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
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

Definition cc_result_type_end_to_end_used_statement : Prop :=
  cc_result_type_end_to_end_used_with false all_fixes.

Definition cc_result_type_end_to_end_used_landed_statement : Prop :=
  cc_result_type_end_to_end_used_with false landed_fixes.

Definition cc_result_type_end_to_end_used_local_statement : Prop :=
  cc_result_type_end_to_end_used_with true all_fixes.

Definition cc_result_type_end_to_end_used_local_landed_statement : Prop :=
  cc_result_type_end_to_end_used_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec.unbox_calls_result_type_end_to_end_with
   with [used_premises]. *)
Definition unbox_calls_result_type_end_to_end_used_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value),
    used_premises local flag is_local u e T sets ->
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

Definition unbox_calls_result_type_end_to_end_used_statement : Prop :=
  unbox_calls_result_type_end_to_end_used_with false all_fixes.

Definition unbox_calls_result_type_end_to_end_used_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_used_with false landed_fixes.

Definition unbox_calls_result_type_end_to_end_used_local_statement : Prop :=
  unbox_calls_result_type_end_to_end_used_with true all_fixes.

Definition unbox_calls_result_end_to_end_used_local_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_used_with true landed_fixes.

(* FlambdaReaperTypesCallsSpec
   .calls_unboxed_result_type_end_to_end_with with [used_premises]. *)
Definition calls_unboxed_result_type_end_to_end_used_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v : value),
    used_premises local flag is_local u e T sets ->
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

Definition calls_unboxed_result_type_end_to_end_used_statement : Prop :=
  calls_unboxed_result_type_end_to_end_used_with false all_fixes.

Definition calls_unboxed_result_type_end_to_end_used_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_used_with false landed_fixes.

Definition calls_unboxed_result_type_end_to_end_used_local_statement : Prop :=
  calls_unboxed_result_type_end_to_end_used_with true all_fixes.

Definition calls_unboxed_result_end_to_end_used_local_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_used_with true landed_fixes.
