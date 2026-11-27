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

(* Static blocks whose fields are not all values: the check
   FlambdaReaperTypesTermSpec.unit_of_values, widened to the static blocks
   that the rebuild keeps, and what is left beyond it.

   This file is to be reviewed: the check ([binding_of_values_with],
   [expr_of_values_with], [unit_of_values_with], [used_values],
   [statics_of_used_values]), the statements and the counterexample,
   against FlambdaReaperTypesTermSpec.v (the check and the statements
   that this one widens), FlambdaReaperTypesSpec.v and
   FlambdaSemanticsSpec.v. The proofs are in
   FlambdaReaperTypesUsedValuesProofs.v, which needs no review.

   The model's static block has no shape: its traversal and its rebuild
   read every field at the kind value
   (FlambdaReaperTraversal.block_deps, FlambdaReaperRebuild.rebuild_fields,
   which makes an unread field poison of the kind value). The compiler
   reads a static block at its shape: its traversal gives each field the
   kind of the shape (traverse.ml, [traverse_block_like_static_const], by
   [Static_const.block_field_kind]), and its rebuild makes an unread field
   poison of that kind (rebuild.ml, [rewrite_static_const], by
   [K.Scannable_block_shape.element_kind]). On a static block of fields of
   other kinds than value the model's reading is not the compiler's.

   The wider check. The statements of the exported types read the static
   values of the symbols that have a use only
   (FlambdaReaperTypesSpec.statics_rebuilt; the rebuild keeps the binding
   of a static block exactly where its symbol has a use,
   FlambdaReaperRebuild.static_const_is_used). So the check asks that the
   fields be values of the static blocks whose symbol has a use in the
   solution only ([used_values]): it is the check of
   FlambdaReaperTypesTermSpec.v ([const_of_values]) for the static
   constants whose symbol [used] holds of ([unit_of_values_with]), and
   [unit_of_values] is the instance where [used] holds of every symbol.
   Of the other static blocks, the fields may be of any kind: the rebuild
   removes them, and nothing reads them.

   The statements:
   - [unit_of_values_with_runs_statement]: in every state that the run of
     a unit that passes the check of [used] reaches, the static blocks
     bound to a symbol that [used] holds of hold values
     ([statics_of_used_values]).
   - [unit_of_values_with_instance_statement]: a unit that passes the
     check of FlambdaReaperTypesTermSpec.v passes that of any [used].
   - The statements of FlambdaReaperTypesTermSpec.v with [used_values] in
     place of [unit_of_values]: [exit_relation_used_term_statement],
     [exit_relation_whole_used_term_statement],
     [types_end_to_end_used_term_statement],
     [unaliased_types_end_to_end_used_term_statement], each with [_local]
     as there, those of types each with [_landed].
   - [dead_naked_block_example_statement]: a unit that binds a static
     block of a naked immediate, whose symbol has no use, and returns
     another symbol: it fails [unit_of_values] and passes [used_values]
     with its least solution, and meets the premises of
     [exit_relation_used_term_statement]; its rebuild removes the block.
   - [naked_unread_counterexample_statement]: beyond the check, the
     statement of the types is false. The unit

       let symbol s50 = { 7; #3 } in
       let x10 = s50.(0) in
       return x10

     whose static block is of the shape of a record of a value and a
     naked immediate, with its least solution, meets every premise of
     FlambdaReaperTypesTermSpec.types_end_to_end_term_statement but the
     check (which it fails, as it fails [used_values]: [s50] has a use).
     The typing environment gives [s50] the type of a block of that shape
     whose field 0 is 7 and field 1 a naked immediate, which holds of it
     where the unit exits; the rewriting keeps that type (the field 1 has
     no use, and its type is the unknown type of its kind). The rebuild
     makes the field 1, which nothing reads, poison of the kind value, of
     which no type of a naked immediate holds, nor the shape: wherever
     the rebuilt unit exits, the type rewritten does not hold of [s50].
     With [_landed] and [_compiler], of the fixes to land and of the
     compiler's rewriting ([compiler_fixes]).

   The counterexample is the model's, not the compiler's: the compiler's
   rebuild makes the field poison of the kind of the shape, a naked
   immediate, of which the type holds, and its rewriting reads the field
   at the kind of its type (types_rewriter.ml, [block_slot]), which is
   that of the shape. A model of the static blocks with their shapes
   would read the fields as the compiler does, and lift the check. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaWellKindedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperLocalFields
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* [binding_of_values] of the static constants whose symbol [used] holds
   of. *)
Definition binding_of_values_with (used : symbol -> bool) (kinds : kind_env)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s const => negb (used s) || const_of_values kinds const
  | Bind_code _ _ | Bind_deleted_code _ => true
  end.

(* [expr_of_values] with [binding_of_values_with]. *)
Fixpoint expr_of_values_with (used : symbol -> bool) (kinds : kind_env)
    (e : expr) : bool :=
  match e with
  | Let x kind _ body => expr_of_values_with used ((x, kind) :: kinds) body
  | Let_static bindings body =>
    forallb (binding_of_values_with used kinds) bindings
    && expr_of_values_with used kinds body
  | Let_cont_non_recursive _ h body =>
    handler_of_values_with used kinds h && expr_of_values_with used kinds body
  | Let_cont_recursive _ body => expr_of_values_with used kinds body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_of_values_with (used : symbol -> bool) (kinds : kind_env)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    match bind_param_kinds params (hm_params_kinds metadata) kinds with
    | Some handler_kinds => expr_of_values_with used handler_kinds handler
    | None => true
    end
  end.

Definition unit_of_values_with (used : symbol -> bool) (e : expr) : bool :=
  expr_of_values_with used [] e.

(* The wider check: the static blocks whose symbol has a use in the
   solution, which the rebuild keeps, hold values. *)
Definition used_values (T : tables) (e : expr) : bool :=
  unit_of_values_with
    (fun s => s_has_usage (solution_of_tables T) (Node_symbol s)) e.

(* FlambdaReaperTypesSpec.statics_of_values of the symbols that [used]
   holds of. *)
Definition statics_of_used_values (used : symbol -> bool)
    (statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    used s = true ->
    match allocated with
    | Static_block _ fields =>
      ∀ v : value, v ∈ fields -> kind_of_value v = Value
    | Static_boxed_number contents =>
      boxable_number_of_kind (kind_of_value contents) <> None
    | Static_closure _ _ | Static_closure_of_set _ _ _ _ | Static_string _
    | Static_immutable_array _ | Static_imported _ =>
      True
    end.

(* ======================================================================== *)
(* The check and the runs                                                   *)
(* ======================================================================== *)

Definition unit_of_values_with_runs_statement : Prop :=
  ∀ (used : symbol -> bool) (e : expr) (n : nat) (s : state),
    unit_of_values_with used e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    statics_of_used_values used (state_statics s).

Definition unit_of_values_with_instance_statement : Prop :=
  ∀ (used : symbol -> bool) (e : expr),
    unit_of_values e = true -> unit_of_values_with used e = true.

(* ======================================================================== *)
(* The statements of FlambdaReaperTypesTermSpec.v, with [used_values]      *)
(* ======================================================================== *)

Definition exit_relation_used_term_statement : Prop :=
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
    used_values T e = true ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition exit_relation_whole_used_term_statement : Prop :=
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
    used_values T e = true ->
    rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition types_end_to_end_used_term_with (fx : fixes) : Prop :=
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
    used_values T e = true ->
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
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition types_end_to_end_used_term_statement : Prop :=
  types_end_to_end_used_term_with all_fixes.

Definition types_end_to_end_used_term_landed_statement : Prop :=
  types_end_to_end_used_term_with landed_fixes.

Definition unaliased_types_end_to_end_used_term_with (fx : fixes) : Prop :=
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
    used_values T e = true ->
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
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_used_term_statement : Prop :=
  unaliased_types_end_to_end_used_term_with all_fixes.

Definition unaliased_types_end_to_end_used_term_landed_statement : Prop :=
  unaliased_types_end_to_end_used_term_with landed_fixes.

Definition exit_relation_used_term_local_statement : Prop :=
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
    used_values T e = true ->
    rebuild_exit_relation (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition exit_relation_whole_used_term_local_statement : Prop :=
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
    used_values T e = true ->
    rebuild_exit_relation_whole (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition types_end_to_end_used_term_local_with (fx : fixes) : Prop :=
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
    used_values T e = true ->
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
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition types_end_to_end_used_term_local_statement : Prop :=
  types_end_to_end_used_term_local_with all_fixes.

Definition types_end_to_end_used_term_local_landed_statement : Prop :=
  types_end_to_end_used_term_local_with landed_fixes.

Definition unaliased_types_end_to_end_used_term_local_with (fx : fixes)
    : Prop :=
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
    used_values T e = true ->
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
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_used_term_local_statement : Prop :=
  unaliased_types_end_to_end_used_term_local_with all_fixes.

Definition unaliased_types_end_to_end_used_term_local_landed_statement : Prop :=
  unaliased_types_end_to_end_used_term_local_with landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* A static block of a naked immediate whose symbol has no use:

     let symbol s50 = { #3 } and s51 = { 7 } in
     return s51
*)
Module DeadNakedBlockExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [Const (Naked_immediate 3)]);
     Bind_symbol 51 (Static_const_block 0 [int 7])]
    (return_to 1 [Symbol 51]).

Definition facts : facts := traverse_unit Never interface unit_expr.

Definition s50 : node := Node_symbol 50.
Definition s51 : node := Node_symbol 51.
Definition returned : node := Node_made 1.
Definition raised : node := Node_made 2.
Definition constants : node := Node_all_constants.

(* The least solution: [s50] has no use. *)
Definition tables : tables :=
  {| t_alias := [(returned, s51)];
     t_any_usage := [constants; s51; raised; returned];
     t_any_source := [constants];
     t_has_usage := [constants; s51; returned; raised];
     t_has_source := [returned; s50; s51; constants];
     t_usages := [];
     t_sources := [(returned, s51); (s50, s50); (s51, s51)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used :=
       [(s51, Field_get_tag); (s51, Field_is_int); (s51, Field_block 0 Value)]
  |}.

(* The rebuild removes the binding of [s50]. *)
Definition rebuilt : expr :=
  Let_static [Bind_symbol 51 (Static_const_block 0 [int 7])]
    (return_to 1 [Symbol 51]).

End DeadNakedBlockExample.

(* The unit meets the premises of [exit_relation_used_term_statement] with
   its least solution and no local field: it passes [used_values] and
   fails [unit_of_values], so that
   FlambdaReaperTypesTermSpec.exit_relation_term_statement does not apply
   to it. Its rebuild removes the block of the naked immediate. *)
Definition dead_naked_block_example_statement : Prop :=
  least_solution FlambdaReaperUnboxing.Examples.no_local
    DeadNakedBlockExample.facts DeadNakedBlockExample.tables
  ∧ is_local_slots_only FlambdaReaperUnboxing.Examples.no_local
  ∧ (∀ f : field, FlambdaReaperUnboxing.Examples.no_local f = false)
  ∧ in_flow_fragment DeadNakedBlockExample.unit_expr = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments
      DeadNakedBlockExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] []
      DeadNakedBlockExample.unit_expr
    = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      DeadNakedBlockExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value
      FlambdaReaperUnboxing.Examples.interface DeadNakedBlockExample.unit_expr
    = true
  ∧ check_closed FlambdaReaperUnboxing.Examples.no_local
      DeadNakedBlockExample.facts DeadNakedBlockExample.tables
    = true
  ∧ check_field_used_exact FlambdaReaperUnboxing.Examples.no_local
      DeadNakedBlockExample.facts DeadNakedBlockExample.tables
    = true
  ∧ variables_bound_once DeadNakedBlockExample.unit_expr = true
  ∧ unit_of_values DeadNakedBlockExample.unit_expr = false
  ∧ used_values DeadNakedBlockExample.tables DeadNakedBlockExample.unit_expr
    = true
  ∧ ∀ preserve_direct_calls : bool,
      rebuild_unit_from_tables DeadNakedBlockExample.facts
        DeadNakedBlockExample.tables preserve_direct_calls
        DeadNakedBlockExample.unit_expr
      = DeadNakedBlockExample.rebuilt.

(* ======================================================================== *)
(* Beyond the check                                                         *)
(* ======================================================================== *)

(* The unit of the counterexample:

     let symbol s50 = { 7; #3 } in
     let x10 = s50.(0) in
     return x10

   with its least solution, the typing environment that gives [s50] the
   type of a block of the shape of a record of a value and a naked
   immediate, whose field 0 is 7 and field 1 a naked immediate, and the
   state from which the unit exits. *)
Module NakedUnreadExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [int 7; Const (Naked_immediate 3)])]
    (Let 10 Value (load 0 (Symbol 50)) (return_to 1 [Var 10])).

Definition facts : facts := traverse_unit Never interface unit_expr.

Definition s50 : node := Node_symbol 50.
Definition x10 : node := Node_variable 10.
Definition returned : node := Node_made 1.
Definition raised : node := Node_made 2.
Definition constants : node := Node_all_constants.

(* The least solution: the field 1 of [s50] has no use. *)
Definition tables : tables :=
  {| t_alias := [(x10, constants); (returned, x10)];
     t_any_usage := [constants; x10; raised; returned];
     t_any_source := [returned; x10; constants];
     t_has_usage := [s50; constants; x10; returned; raised];
     t_has_source := [returned; s50; x10; constants];
     t_usages := [(s50, s50)];
     t_sources := [(s50, s50)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used := [(s50, Field_block 0 Value)]
  |}.

Definition block_type : flambda_type :=
  immutable_block false 0%Z (Scannable (Mixed_record 1 [Kind_naked_immediate]))
    Alloc_mode.Heap
    [this_tagged_immediate 7%Z;
     FlambdaTypeGrammar.unknown (Naked_number Kind_naked_immediate)].

Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_symbol_definition Typing_env.create 50)
    (Name_symbol 50) block_type.

Definition canon : canon_fn := fun n _ _ => n.

Definition block : statically_allocated_value :=
  Static_block 0
    [Value_const (Tagged_immediate 7); Value_const (Naked_immediate 3)].

Definition exit_state : state :=
  State (return_to 1 [Var 10]) [(10, Value_const (Tagged_immediate 7))] []
    {| static_code := []; static_values := [(50, block)] |} empty_heap
    no_reads.

End NakedUnreadExample.

(* The unit meets every premise of
   FlambdaReaperTypesTermSpec.types_end_to_end_term_with but the check,
   which it fails, as it fails [used_values]; it exits from [exit_state],
   where [tenv] is satisfied and [s50] bound to [block]. The type of [s50]
   rewritten is [block_type], and wherever the rebuilt unit exits it does
   not hold of [s50]. *)
Definition naked_unread_counterexample_with (fx : fixes) : Prop :=
  is_local_slots_only FlambdaReaperUnboxing.Examples.no_local
  ∧ (∀ f : field, FlambdaReaperUnboxing.Examples.no_local f = false)
  ∧ in_flow_fragment NakedUnreadExample.unit_expr = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments NakedUnreadExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] NakedUnreadExample.unit_expr
    = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      NakedUnreadExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value
      FlambdaReaperUnboxing.Examples.interface NakedUnreadExample.unit_expr
    = true
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local
      NakedUnreadExample.facts NakedUnreadExample.tables
  ∧ variables_bound_once NakedUnreadExample.unit_expr = true
  ∧ unit_of_values NakedUnreadExample.unit_expr = false
  ∧ used_values NakedUnreadExample.tables NakedUnreadExample.unit_expr = false
  ∧ sets_of_graph (graph_of_facts NakedUnreadExample.facts) []
  ∧ run 2
      (State NakedUnreadExample.unit_expr [] [] empty_static_env empty_heap
         no_reads)
    = Step NakedUnreadExample.exit_state
  ∧ step NakedUnreadExample.exit_state
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ typing_env_satisfied_strictly
      (state_statics NakedUnreadExample.exit_state) [] NakedUnreadExample.tenv
  ∧ env_in_fragment [] NakedUnreadExample.tenv
  ∧ copies_valued NakedUnreadExample.tenv NakedUnreadExample.canon [] []
  ∧ lookup 50%positive
      (static_values (state_statics NakedUnreadExample.exit_state))
    = Some NakedUnreadExample.block
  ∧ rewrite_symbol_type fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         NakedUnreadExample.facts NakedUnreadExample.tables [])
      NakedUnreadExample.tenv NakedUnreadExample.canon 50%positive
    = NakedUnreadExample.block_type
  ∧ ∀ (preserve_direct_calls : bool) (m : nat) (t : state)
      (vs' : list value) (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables NakedUnreadExample.facts
              NakedUnreadExample.tables preserve_direct_calls
              NakedUnreadExample.unit_expr)
           [] [] empty_static_env empty_heap no_reads)
      = Step t ->
      step t = Exit 1%positive vs' empty_heap rd' ->
      ~ type_satisfied (state_statics t) [] (Value_symbol 50%positive)
          (rewrite_symbol_type fx
             (plain_context FlambdaReaperUnboxing.Examples.no_local
                NakedUnreadExample.facts NakedUnreadExample.tables [])
             NakedUnreadExample.tenv NakedUnreadExample.canon 50%positive).

Definition naked_unread_counterexample_statement : Prop :=
  naked_unread_counterexample_with all_fixes.

Definition naked_unread_counterexample_landed_statement : Prop :=
  naked_unread_counterexample_with landed_fixes.

Definition naked_unread_counterexample_compiler_statement : Prop :=
  naked_unread_counterexample_with compiler_fixes.
