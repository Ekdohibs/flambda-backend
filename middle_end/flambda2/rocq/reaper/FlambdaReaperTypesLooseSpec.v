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

(* The fragment [loose] (closure types that are not aliases) end to end,
   for the rebuilds of FlambdaReaperTypesCallsSpec.v and
   FlambdaReaperTypesReturnsSpec.v.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesSpec.v (the fragment, [env_in_fragment_with],
   [static_sets_whole]) and FlambdaReaperTypesCallsSpec.v and
   FlambdaReaperTypesReturnsSpec.v (whose statements these are, in the
   fragment [loose]). The proofs are in FlambdaReaperTypesLooseProofs.v,
   which needs no review, with FlambdaReaperTypesCcSetsProofs.v.

   The statements of the fragment [loose] ask that the static sets of
   closures be kept whole ([static_sets_whole]) by the runs they relate.
   FlambdaReaperTypesProofs.v proves it of the run of the unit and that
   of its plain rebuild; here it is of the rebuild with changes of
   calling convention too. That rebuild keeps or removes the bindings of
   a static set together, as the plain rebuild does, so the rebuilt unit
   of a well-formed unit binds its static sets whole wherever it runs to
   (FlambdaClosuresSpec.static_sets_bound), and the simulation binds in
   the second run each symbol of a static set that has a use, to a
   closure of the same set. The unit rebuilt with unboxing across calls,
   and with changes of representation, binds the same static values as
   the rebuild with changes of calling convention
   (FlambdaReaperTypesCallsSpec.v), so the static sets are whole there
   too.

   The statements, each the statement of FlambdaReaperTypesCallsSpec.v or
   FlambdaReaperTypesReturnsSpec.v named, with the types of the
   environment of the fragment [loose] ([env_in_fragment_with true]) in
   place of the first fragment; each with [_landed], [_local] and
   [_local_landed] as there:
   - [unaliased_result_type_end_to_end_statement] (of
     [result_type_end_to_end_statement], the plain rebuild) and
     [cc_unaliased_result_type_end_to_end_statement] (of
     [cc_result_type_end_to_end_statement], the rebuild with changes of
     calling convention): the result types at the variables that the run
     binds.
   - [unaliased_code_result_end_to_end_statement] (of
     [code_result_type_end_to_end_statement]) and
     [cc_unaliased_code_result_end_to_end_statement] (of
     [cc_code_result_type_end_to_end_statement]): at the results of code.
   - [calls_unaliased_types_end_to_end_statement] (of
     [unbox_calls_types_end_to_end_statement]),
     [calls_unaliased_result_end_to_end_statement] (of
     [unbox_calls_result_type_end_to_end_statement]) and
     [calls_unaliased_unboxed_end_to_end_statement] (of
     [calls_unboxed_result_type_end_to_end_statement]): the unit rebuilt
     with unboxing across calls.
   - [repr_unaliased_types_end_to_end_statement] (of
     [repr_calls_types_end_to_end_statement]): the unit rebuilt with
     unboxing and changes of representation.
   - [loose_example_statement]: the unit of
     FlambdaReaperUnboxingRefinesSpec.CallExample, at the state where its
     closure [x60] is bound, with result types that give the variable 93
     the type of a closure of the set of [x60] whose closure type at its
     own function slot is unknown, not an alias: the environment is of
     the fragment [loose] and not of the first, so that the statements
     of FlambdaReaperTypesCallsSpec.v do not apply, and those here do; the
     type rewritten holds of [x60] in the run of the unit rebuilt with
     changes of calling convention.

   No premise is added: [static_sets_whole] of each pair of runs follows
   from the premises of the statements named (from [well_formed_unit]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperCallingConventions FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesReturnsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* FlambdaReaperTypesCallsSpec.result_premises, with the types of the
   environment of the fragment [loose]. *)
Definition unaliased_result_premises (F : facts) (T : tables) (e : expr)
    (n : nat) (s : state) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (code_var type_var : variable) (v : value)
    : Prop :=
  run n (initial_state e) = Step s
  ∧ step s <> Undefined
  ∧ lookup code_var (state_vars s) = Some v
  ∧ s_has_usage (solution_of_tables T) (Node_variable code_var) = true
  ∧ typing_env_satisfied_strictly (state_statics s) valuation tenv
  ∧ env_in_fragment_with true valuation tenv
  ∧ copies_valued tenv canon valuation valuation'
  ∧ Typing_env.find tenv (Name_var type_var) None <> None
  ∧ lookup type_var valuation = Some v.

(* ======================================================================== *)
(* The result types, of the plain rebuild and of the rebuild with changes   *)
(* of calling convention                                                    *)
(* ======================================================================== *)

Definition unaliased_result_type_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    unaliased_result_premises (traverse_unit flag u e) T e n s tenv canon
      valuation valuation' code_var type_var v ->
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

Definition unaliased_result_type_end_to_end_statement : Prop :=
  unaliased_result_type_end_to_end_with false all_fixes.

Definition unaliased_result_type_end_to_end_landed_statement : Prop :=
  unaliased_result_type_end_to_end_with false landed_fixes.

Definition unaliased_result_type_end_to_end_local_statement : Prop :=
  unaliased_result_type_end_to_end_with true all_fixes.

Definition unaliased_result_type_end_to_end_local_landed_statement : Prop :=
  unaliased_result_type_end_to_end_with true landed_fixes.

Definition cc_unaliased_result_type_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    unaliased_result_premises (traverse_unit flag u e) T e n s tenv canon
      valuation valuation' code_var type_var v ->
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

Definition cc_unaliased_result_type_end_to_end_statement : Prop :=
  cc_unaliased_result_type_end_to_end_with false all_fixes.

Definition cc_unaliased_result_type_end_to_end_landed_statement : Prop :=
  cc_unaliased_result_type_end_to_end_with false landed_fixes.

Definition cc_unaliased_result_type_end_to_end_local_statement : Prop :=
  cc_unaliased_result_type_end_to_end_with true all_fixes.

Definition cc_unaliased_result_type_end_to_end_local_landed_statement : Prop :=
  cc_unaliased_result_type_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* The results of code                                                      *)
(* ======================================================================== *)

Definition unaliased_code_result_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    return_premises true flag u e T n s id i arg r tenv canon valuation
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

Definition unaliased_code_result_end_to_end_statement : Prop :=
  unaliased_code_result_end_to_end_with false all_fixes.

Definition unaliased_code_result_end_to_end_landed_statement : Prop :=
  unaliased_code_result_end_to_end_with false landed_fixes.

Definition unaliased_code_result_end_to_end_local_statement : Prop :=
  unaliased_code_result_end_to_end_with true all_fixes.

Definition unaliased_code_result_end_to_end_local_landed_statement : Prop :=
  unaliased_code_result_end_to_end_with true landed_fixes.

Definition cc_unaliased_code_result_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (id : code_id) (i : nat) (arg : simple) (r : node)
    (type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    return_premises true flag u e T n s id i arg r tenv canon valuation
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

Definition cc_unaliased_code_result_end_to_end_statement : Prop :=
  cc_unaliased_code_result_end_to_end_with false all_fixes.

Definition cc_unaliased_code_result_end_to_end_landed_statement : Prop :=
  cc_unaliased_code_result_end_to_end_with false landed_fixes.

Definition cc_unaliased_code_result_end_to_end_local_statement : Prop :=
  cc_unaliased_code_result_end_to_end_with true all_fixes.

Definition cc_unaliased_code_result_end_to_end_local_landed_statement : Prop :=
  cc_unaliased_code_result_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* The unit rebuilt with unboxing across calls                              *)
(* ======================================================================== *)

Definition calls_unaliased_types_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
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

Definition calls_unaliased_types_end_to_end_statement : Prop :=
  calls_unaliased_types_end_to_end_with false all_fixes.

Definition calls_unaliased_types_end_to_end_landed_statement : Prop :=
  calls_unaliased_types_end_to_end_with false landed_fixes.

Definition calls_unaliased_types_end_to_end_local_statement : Prop :=
  calls_unaliased_types_end_to_end_with true all_fixes.

Definition calls_unaliased_types_end_to_end_local_landed_statement : Prop :=
  calls_unaliased_types_end_to_end_with true landed_fixes.

Definition calls_unaliased_result_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    unaliased_result_premises (traverse_unit flag u e) T e n s tenv canon
      valuation valuation' code_var type_var v ->
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

Definition calls_unaliased_result_end_to_end_statement : Prop :=
  calls_unaliased_result_end_to_end_with false all_fixes.

Definition calls_unaliased_result_end_to_end_landed_statement : Prop :=
  calls_unaliased_result_end_to_end_with false landed_fixes.

Definition calls_unaliased_result_end_to_end_local_statement : Prop :=
  calls_unaliased_result_end_to_end_with true all_fixes.

Definition calls_unaliased_result_end_to_end_local_landed_statement : Prop :=
  calls_unaliased_result_end_to_end_with true landed_fixes.

Definition calls_unaliased_unboxed_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v : value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    unaliased_result_premises (traverse_unit flag u e) T e n s tenv canon
      valuation valuation' code_var type_var v ->
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

Definition calls_unaliased_unboxed_end_to_end_statement : Prop :=
  calls_unaliased_unboxed_end_to_end_with false all_fixes.

Definition calls_unaliased_unboxed_end_to_end_landed_statement : Prop :=
  calls_unaliased_unboxed_end_to_end_with false landed_fixes.

Definition calls_unaliased_unboxed_end_to_end_local_statement : Prop :=
  calls_unaliased_unboxed_end_to_end_with true all_fixes.

Definition calls_unaliased_unboxed_end_to_end_local_landed_statement : Prop :=
  calls_unaliased_unboxed_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* The unit rebuilt with unboxing and changes of representation             *)
(* ======================================================================== *)

Definition repr_unaliased_types_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    unit_premises local flag is_local u e T sets ->
    repr_premises flag is_local u e T cannot_change dominated to_unbox tcr nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
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

Definition repr_unaliased_types_end_to_end_statement : Prop :=
  repr_unaliased_types_end_to_end_with false all_fixes.

Definition repr_unaliased_types_end_to_end_landed_statement : Prop :=
  repr_unaliased_types_end_to_end_with false landed_fixes.

Definition repr_unaliased_types_end_to_end_local_statement : Prop :=
  repr_unaliased_types_end_to_end_with true all_fixes.

Definition repr_unaliased_types_end_to_end_local_landed_statement : Prop :=
  repr_unaliased_types_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesSpec.CallExample
   (FlambdaReaperTypesCallsSpec.UnboxCallsExample), at the state where
   [x60], the closure at the function slot 70 of the code 50, is bound.
   The result types give the variable 93 the type of a closure at the
   function slot 70 of a set of at least that slot, whose code is 50 and
   whose closure type at the slot 70 is unknown: a closure type that is
   not an alias, of the fragment [loose] and not of the first. *)
Module LooseExample.

Local Open Scope positive_scope.

Definition closure_type : flambda_type :=
  at_least_this_closure 70 [(70, Or_unknown.Known (Function_type 50))]
    [(70, unknown Value)] [] Alloc_mode.Heap.

Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create 93 Value
       Name_mode.normal)
    (Name_var 93) closure_type.

Definition closure : value := Value_closure_of_set 70 [(70, 50)] [].

Definition valuation : env := [(93, closure)].

(* The unit rebuilt with changes of calling convention. *)
Definition cc_rebuilt : expr :=
  rebuild_unit_cc_from_tables UnboxCallsExample.facts
    UnboxCallsExample.tables UnboxCallsExample.cannot_change
    (unit_code_deps Auto FlambdaReaperUnboxing.Examples.interface
       UnboxCallsExample.unit_expr)
    true UnboxCallsExample.unit_expr.

End LooseExample.

(* The unit meets the premises of
   [cc_unaliased_result_type_end_to_end_statement], with the tables of
   the least solution, no local field, the set of closures of [x60] and
   the table [cannot_change] of the example, at the state that the run of
   the unit comes to in 3 steps, where [x60] is bound to [closure], with
   the variable 93 of the result types at [closure]; the environment of
   the result types is not of the first fragment. The type that the
   rewriting gives the variable 93 under the metadata of [x60] is
   [closure_type] itself, and the run of the rebuilt unit comes in 3 steps
   to a state where [x60] is bound to [closure], of which it holds. *)
Definition loose_example_with (fx : fixes) : Prop :=
  unit_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface UnboxCallsExample.unit_expr
    UnboxCallsExample.tables UnboxCallsExample.sets
  ∧ check_cc_closed UnboxCallsExample.facts UnboxCallsExample.tables
      UnboxCallsExample.cannot_change
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      UnboxCallsExample.facts UnboxCallsExample.tables
    = true
  ∧ ~ env_in_fragment LooseExample.valuation LooseExample.tenv
  ∧ (∃ s : state,
       unaliased_result_premises UnboxCallsExample.facts
         UnboxCallsExample.tables UnboxCallsExample.unit_expr 3 s
         LooseExample.tenv UnboxCallsExample.canon LooseExample.valuation
         LooseExample.valuation 60%positive 93%positive LooseExample.closure)
  ∧ rewrite_result_type fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         UnboxCallsExample.facts UnboxCallsExample.tables
         UnboxCallsExample.sets)
      LooseExample.tenv UnboxCallsExample.canon 60%positive 93%positive
    = LooseExample.closure_type
  ∧ ∃ t : state,
      run 3 (initial_state LooseExample.cc_rebuilt) = Step t
      ∧ lookup 60%positive (state_vars t) = Some LooseExample.closure
      ∧ type_satisfied (state_statics t) LooseExample.valuation
          LooseExample.closure LooseExample.closure_type.

Definition loose_example_statement : Prop :=
  loose_example_with all_fixes.

Definition loose_example_landed_statement : Prop :=
  loose_example_with landed_fixes.
