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

(* The statements of FlambdaReaperUnboxingRefinesNever2Spec.v (the calls
   whose code is not known that never return, which may complete a partial
   application) with types and with the primitives of the heap, as those of
   the fragments of FlambdaReaperUnboxingRefinesNeverSpec.v and
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v are in
   FlambdaReaperTypesSpec.v and FlambdaReaperUnboxingRefinesCallsHeapSpec.v.

   This file is to be reviewed: the statements, which are those of the two
   files named with the fragments [calls_unboxing_never2_nr] and
   [calls_unboxing_unit_never2_nr] in place of [calls_unboxing_never_nr]
   and [calls_unboxing_unit_never_nr], and the same premises otherwise. The
   proofs are in FlambdaReaperUnboxingRefinesNever2TypesProofs.v, which
   needs no review.

   With types (FlambdaReaperTypesSpec.v), the flow read off the first
   rebuilt unit with types, analysed again:
   [rebuild_unbox_nr_ind2_with_types_refines_cc_statement],
   [rebuild_unbox_nr_ind2_with_types_local_statement],
   [rebuild_unbox_nr_ind2_with_types_observably_statement]; read off the
   unit's own analysis:
   [rebuild_unbox_unit_never2_with_types_refines_cc_statement],
   [rebuild_unbox_unit_never2_with_types_local_statement],
   [rebuild_unbox_unit_never2_with_types_observably_statement].

   With the primitives of the heap
   (FlambdaReaperUnboxingRefinesCallsHeapSpec.v), of the units of
   [in_heap_fragment], against the change of calling conventions, heaps
   included, and against the unit up to the heap:
   [rebuild_unbox_nr_ind2_{cc_heap,heap_local,up_to_heap}_statement],
   [rebuild_unbox_unit_never2_{cc_heap,heap_local,up_to_heap}_statement],
   and with types
   [rebuild_unbox_nr_ind2_with_types_{cc_heap,heap_local,up_to_heap}],
   [rebuild_unbox_unit_never2_with_types_{cc_heap,heap_local,up_to_heap}]
   (each a [_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperLeastSpec FlambdaUnusedStaticsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec FlambdaReaperTypes
  FlambdaReaperUnboxingRefinesCallsHeapSpec
  FlambdaReaperUnboxingRefinesNever2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaStaticsInScopeSpec FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* With types                                                               *)
(* ======================================================================== *)

Definition rebuild_unbox_nr_ind2_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaUnusedStaticsSpec.program_refines
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaUnusedStaticsSpec.program_refines
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_observably e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

(* ======================================================================== *)
(* With the primitives of the heap                                          *)
(* ======================================================================== *)

Definition rebuild_unbox_nr_ind2_cc_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_cc_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_with_types_cc_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaUnusedStaticsSpec.program_refines
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_with_types_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_with_types_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_never2_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_cc_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaUnusedStaticsSpec.program_refines
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_with_types_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNever2Spec.calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    FlambdaReaperRebuildSpec.program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).
