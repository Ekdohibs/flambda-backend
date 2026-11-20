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

(* A premise on the term in place of
   FlambdaReaperTypesSpec.exits_of_values, a premise on the runs: the
   static blocks of the unit hold values where it exits.

   This file is to be reviewed: the check ([const_of_values],
   [expr_of_values], [unit_of_values]) and the statements, against
   FlambdaReaperTypesSpec.v and FlambdaSemanticsSpec.v. The proofs are in
   FlambdaReaperTypesTermProofs.v, which needs no review.

   The check: under the kinds that the binders in scope give the
   variables, the fields of a static block are of kind value, and the
   contents of a static boxed number of a kind that can be boxed, as
   FlambdaWellKindedSpec.static_const_well_kinded asks of them. Nothing
   else is asked: the check of FlambdaWellKindedSpec.v asks this among
   much else, and is not a premise that a statement about the terms of
   the compiler may take (its header). Code and the handlers of a
   recursive [Let_cont] are not looked into: the machine binds nothing
   static there (FlambdaSemanticsSpec.call_code and [step], by
   [binds_statics]). Nor is a handler that has not as many kinds as
   parameters: no value is ever bound to them
   (FlambdaSemanticsSpec.apply_cont_value).

   Why it is enough. The machine makes a static block of the values that
   its fields have in the environment of the [Let_static]
   (FlambdaSemanticsSpec.eval_static_const), and nothing changes it
   after. Where something static may be bound, a variable holds a value
   of the kind of its binder: the machine has no step at a [Let] of a
   value of another kind, nor where a handler is applied to values of
   other kinds than its parameters, and calls no code whose body binds
   something static. So in every state that the run reaches, the
   expression passes the check under the kinds of the values of the
   variables, and so does each handler that a continuation keeps, under
   the kinds of the values it keeps.

   The statements:
   - [unit_of_values_runs_statement]: in every state that the run of a
     unit that passes the check reaches, the static values are of the
     fragment of FlambdaReaperTypesSpec.statics_of_values;
     [unit_of_values_exits_statement]: so the unit has [exits_of_values].
   - [well_kinded_unit_of_values_statement]: the check of
     FlambdaWellKindedSpec.v asks it, among much else.
   - [naked_field_not_of_values_statement]: the unit of
     FlambdaReaperTypesSpec.naked_field_exit_relation_statement, a static
     block of a naked immediate, fails it.
   - The statements of FlambdaReaperTypesSpec.v that take
     [exits_of_values], with [unit_of_values] in its place:
     [exit_relation_term_statement]
     (of [rebuild_exit_relation_holds_statement]),
     [exit_relation_whole_term_statement]
     (of [rebuild_exit_relation_whole_holds_statement]),
     [types_end_to_end_term_statement]
     (of [exported_types_sound_end_to_end_statement]),
     [unaliased_types_end_to_end_term_statement]
     (of [exported_unaliased_types_sound_end_to_end_statement]), and
     each with [_local] (of the statement with [_local]).

   What is left out: static blocks whose fields are not all values. The
   compiler reads a static block at its shape, which the model has not
   got; such a unit fails the check. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaWellKindedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperLocalFields
  FlambdaReaperTypes FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The fields of a static block are values, and the contents of a static
   boxed number a number that can be boxed. *)
Definition const_of_values (kinds : kind_env) (const : static_const)
    : bool :=
  match const with
  | Static_const_block _ fields =>
    forallb (fun s => has_kind (kind_of_simple kinds s) Value) fields
  | Static_const_boxed_number contents =>
    match kind_of_simple kinds contents with
    | Some kind => is_some (boxable_number_of_kind kind)
    | None => false
    end
  | Static_const_set_of_closures _ _ | Static_const_closure_of_set _ _ _ _
  | Static_const_immutable_string _ | Static_const_immutable_value_array _ =>
    true
  end.

Definition binding_of_values (kinds : kind_env) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_symbol _ const => const_of_values kinds const
  | Bind_code _ _ | Bind_deleted_code _ => true
  end.

(* The constants of the static bindings that the term may run, under the
   kinds of the variables in scope. *)
Fixpoint expr_of_values (kinds : kind_env) (e : expr) : bool :=
  match e with
  | Let x kind _ body => expr_of_values ((x, kind) :: kinds) body
  | Let_static bindings body =>
    forallb (binding_of_values kinds) bindings && expr_of_values kinds body
  | Let_cont_non_recursive _ h body =>
    handler_of_values kinds h && expr_of_values kinds body
  | Let_cont_recursive _ body => expr_of_values kinds body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_of_values (kinds : kind_env) (h : continuation_handler)
    : bool :=
  match h with
  | Handler params metadata handler =>
    match bind_param_kinds params (hm_params_kinds metadata) kinds with
    | Some handler_kinds => expr_of_values handler_kinds handler
    | None => true
    end
  end.

(* A unit, with no variable in scope. *)
Definition unit_of_values (e : expr) : bool := expr_of_values [] e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition unit_of_values_runs_statement : Prop :=
  ∀ (e : expr) (n : nat) (s : state),
    unit_of_values e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    statics_of_values (state_statics s).

Definition unit_of_values_exits_statement : Prop :=
  ∀ e : expr, unit_of_values e = true -> exits_of_values e.

Definition well_kinded_unit_of_values_statement : Prop :=
  ∀ (slots : slot_kinds) (unit_return unit_exn : continuation) (e : expr),
    well_kinded_unit slots unit_return unit_exn e = true ->
    unit_of_values e = true.

Definition naked_field_not_of_values_statement : Prop :=
  unit_of_values NakedFieldExample.unit_expr = false.

(* [rebuild_exit_relation_holds_statement] of FlambdaReaperTypesSpec.v. *)
Definition exit_relation_term_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* [rebuild_exit_relation_whole_holds_statement] of FlambdaReaperTypesSpec.v. *)
Definition exit_relation_whole_term_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* [exported_types_sound_end_to_end_statement] of FlambdaReaperTypesSpec.v. *)
Definition types_end_to_end_term_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

(* [exported_unaliased_types_sound_end_to_end_statement]
   of FlambdaReaperTypesSpec.v. *)
Definition unaliased_types_end_to_end_term_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

(* ======================================================================== *)
(* Local fields                                                             *)
(* ======================================================================== *)

(* The statements with [_local] of FlambdaReaperTypesSpec.v, with
   [unit_of_values] in place of [exits_of_values]. *)
Definition exit_relation_term_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition exit_relation_whole_term_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition types_end_to_end_term_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_term_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type all_fixes
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).
