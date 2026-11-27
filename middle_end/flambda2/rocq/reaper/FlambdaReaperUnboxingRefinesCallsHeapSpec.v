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


(* The rebuild with unboxing across calls and code of
   FlambdaReaperUnboxingRefinesCallsSpec.v, in the mode of the fix, for
   the units with primitives of the heap
   (FlambdaReaperFlowSpec.in_heap_fragment).

   This file is to be reviewed: the statements, against
   FlambdaReaperUnboxingRefinesCallsSpec.v,
   FlambdaReaperUnboxingRefinesHeapSpec.v and
   FlambdaReaperCallingConventionsSpec.v. The proofs are in
   FlambdaReaperUnboxingRefinesCallsHeapProofs.v, which needs no review.

   How the unboxing across calls meets the heap: as the local unboxing
   does (FlambdaReaperUnboxingRefinesHeapSpec.v). What the fragment
   ([calls_unboxing]) unboxes is bound to an immutable block of values, a
   boxed number, an alias or the read of a field unboxed in turn, or is a
   parameter given such a name ([unboxed_binding_ok], [arg_ok]): never a
   cell. An unboxed name, a parameter of code or of a handler included, is
   no argument of a primitive ([plain_binding_ok]), of the heap or not, so
   no block unboxed is stored into a cell or compared physically. A field
   of an unboxed block may hold a location: the variable of the field
   holds it, is given across a call as an argument or returned as a
   result as any other field, and a read of the field becomes that
   variable, so that a store into the cell is seen alike through either.
   A parameter that the rebuild with unboxing keeps with no use, to which
   it may give poison, is no argument of a primitive either: both rebuilds
   make poison of a name with no use where they read it. So the two
   rebuilds make the same primitives of the heap on the same values, and
   their heaps are one.

   The statements:
   - [rebuild_unbox_fixed_refines_cc_heap_statement] and
     [rebuild_unbox_fixed_refines_cc_heap_never_returns_statement]:
     [rebuild_unbox_fixed_refines_cc_statement] of
     FlambdaReaperUnboxingRefinesCallsSpec.v and its form with the code
     [nr] taken to never return, of the units of [in_heap_fragment]. The
     comparison is [program_refines], heaps included: the same cells at
     the same locations.
   - [rebuild_unbox_fixed_refines_up_to_heap_statement],
     [rebuild_unbox_fixed_refines_up_to_heap_local_statement],
     [rebuild_unbox_fixed_up_to_heap_never_returns_statement] and
     [rebuild_unbox_fixed_heap_local_never_returns_statement]: the
     statements up to observation of
     FlambdaReaperUnboxingRefinesCallsSpec.v, without and with local
     fields, without and with code taken to never return, of the units
     of [in_heap_fragment], the exits compared up to a correspondence of
     the locations of the two runs ([program_refines_up_to_heap]): the
     first statements composed with
     [rebuild_cc_refines_up_to_heap_statement] of
     FlambdaReaperCallingConventionsSpec.v and its form with local
     fields. They have the premises of those
     statements, and [calls_unboxing] and [check_unboxed_reads].
   - [rebuild_unbox_fixed_with_types_cc_heap_nr_statement],
     [rebuild_unbox_fixed_with_types_heap_local_nr_statement] and
     [rebuild_unbox_fixed_with_types_up_to_heap_nr_statement]: those of
     FlambdaReaperTypesSpec.v for the reaper with types
     ([rebuild_unbox_fixed_with_types_refines_cc_nr_statement] and the
     two forms up to observation), of the units of [in_heap_fragment],
     the last two up to the heap, composed with
     [rebuild_cc_with_types_refines_up_to_heap_local_statement] of
     FlambdaReaperTypesSpec.v.
   - [rebuild_unbox_nr_ind_cc_heap_statement],
     [rebuild_unbox_nr_ind_heap_local_statement] and
     [rebuild_unbox_nr_ind_up_to_heap_statement]: those of
     FlambdaReaperUnboxingRefinesNeverSpec.v (calls whose code is not
     known that never return, the flow read off the first rebuilt unit,
     which they ask of [in_heap_fragment] already), of the units of
     [in_heap_fragment], the last two up to the heap; and
     [rebuild_unbox_unit_never_cc_heap_statement],
     [rebuild_unbox_unit_never_heap_local_statement] and
     [rebuild_unbox_unit_never_up_to_heap_statement], those of
     FlambdaReaperUnboxingRefinesUnitFlowSpec.v alike; and the six
     statements of the reaper with types of FlambdaReaperTypesSpec.v for
     the two fragments ([rebuild_unbox_nr_ind_with_types_cc_heap_statement]
     and the others with [_with_types]) alike.
   - [calls_heap_example_statement]: a unit of [in_heap_fragment] and not
     of [in_flow_fragment] whose code takes a block holding a cell, reads
     the cell from the block and stores into it; the rules of the fix
     unbox the block and the parameter, the call gives the code the
     location, and the unit and its rebuild exit alike, with the same
     heap.
   The statements of FlambdaReaperUnboxingRefinesCallsSpec.v against the
   change of calling conventions are the first two here on the units of
   [in_flow_fragment], which are of [in_heap_fragment]
   (FlambdaReaperFlowSpec.in_flow_fragment_in_heap_fragment_statement);
   those up to observation compare the exits by [observed_equal], with no
   heap, which the comparison up to the heap gives of a run with no
   primitive of the heap only
   (FlambdaReaperRebuildSpec.rebuild_refines_up_to_heap_flow_statement).
   No counterexample. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperLeastSpec FlambdaUnusedStaticsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec FlambdaReaperTypes.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaStaticsInScopeSpec FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Against the change of calling conventions                                *)
(* ======================================================================== *)

Definition rebuild_unbox_fixed_refines_cc_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same with calls of code that never returns: the fragment takes the
   code [nr] to never return ([calls_unboxing_nr]). The statement before is
   this one with no such code. *)
Definition rebuild_unbox_fixed_refines_cc_heap_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* Against the unit, up to the heap                                         *)
(* ======================================================================== *)

Definition rebuild_unbox_fixed_refines_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
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
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too. *)
Definition rebuild_unbox_fixed_refines_up_to_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
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
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The two statements before, with calls of code that never returns. *)
Definition rebuild_unbox_fixed_up_to_heap_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_fixed_heap_local_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* The reaper with types                                                    *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.rebuild_unbox_fixed_with_types_refines_cc_nr_statement
   of the units of [in_heap_fragment]: the rebuild with unboxing of the
   reaper with types in the mode of the fix refines the rebuild with types
   and changes of calling convention, heaps included. *)
Definition rebuild_unbox_fixed_with_types_cc_heap_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change (unit_code_deps flag u e)
         preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox [] preserve_direct_calls e).

(* The unit rebuilt with unboxing by the reaper with types in the mode of
   the fix refines the unit up to the heap, with the premises of
   FlambdaReaperTypesSpec.rebuild_unbox_fixed_with_types_local_nr_statement
   but [in_heap_fragment] in place of [in_flow_fragment]. *)
Definition rebuild_unbox_fixed_with_types_heap_local_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox [] preserve_direct_calls e).

(* The same without local fields. *)
Definition rebuild_unbox_fixed_with_types_up_to_heap_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox [] preserve_direct_calls e).

(* ======================================================================== *)
(* Calls whose code is not known that never return                          *)
(* ======================================================================== *)

(* The statements of FlambdaReaperUnboxingRefinesNeverSpec.v (the calls
   whose code is not known that never return, with the flow of the first
   rebuilt unit, which they ask of [in_heap_fragment] already) of the
   units of [in_heap_fragment]: against the change of calling
   conventions, heaps included, and against the unit up to the heap, with
   and without local fields. *)
Definition rebuild_unbox_nr_ind_cc_heap_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_nr_ind_heap_local_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind_up_to_heap_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same, for the fragment of
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v, the checks of the flow read
   off the analysis of the unit itself. *)
Definition rebuild_unbox_unit_never_cc_heap_statement : Prop :=
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
    calls_unboxing_unit_never_nr
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

Definition rebuild_unbox_unit_never_heap_local_statement : Prop :=
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
    calls_unboxing_unit_never_nr
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

Definition rebuild_unbox_unit_never_up_to_heap_statement : Prop :=
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
    calls_unboxing_unit_never_nr
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

(* The statements of the reaper with types of FlambdaReaperTypesSpec.v
   for the same fragments, of the units of [in_heap_fragment]. *)
Definition rebuild_unbox_nr_ind_with_types_cc_heap_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
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

Definition rebuild_unbox_nr_ind_with_types_heap_local_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
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

Definition rebuild_unbox_nr_ind_with_types_up_to_heap_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
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

Definition rebuild_unbox_unit_never_with_types_cc_heap_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
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

Definition rebuild_unbox_unit_never_with_types_heap_local_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
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

Definition rebuild_unbox_unit_never_with_types_up_to_heap_statement : Prop :=
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
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
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

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The code 50 takes a block [x40] holding a cell, reads the cell from it
   and stores 2 into the cell; the unit makes the cell [x10] and the block
   [x11], calls 50 directly with [x11], and returns what the cell then
   holds:

     code 50 (x40) =
       let x43 = x40.(0) in
       let x44 = x43.(0) <- 2 in
       return x44
     let x10 = mutable { 1 } in
     let x11 = [x10] in
     let x60 = closure 50 in
     let k20 (x30) = let x31 = x10.(0) in return x31 in    (a mutable load)
     x60 x11 -> k20

   With the least solution, the rules of the fix unbox [x11] and the
   parameter [x40], which it dominates; the cell has any use, being stored
   into, and is not unboxed. The rebuild with unboxing gives 50 the field
   of [x11], the variable 63 bound to [x10], which 50 takes as its
   parameter 61 and reads as the cell; the result of 50 and the parameter
   of [k20] have no use and are deleted. *)
Module CallsHeapExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition access : block_access_kind :=
  Block_access_values None None Any_value.

Definition store (cell : simple) (v : Z) : named :=
  Prim (Heap (Block_set access Assignment 0)) [cell; int v].

Definition callee : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 43 Value (load 0 (Var 40))
       (Let 44 Value (store (Var 43) 2) (return_to 3 [Var 44]))).

Definition reader : continuation_handler :=
  Handler [30] (make_handler_metadata [Value])
    (Let 31 Value (Prim (Unary (Block_load access Mutable 0)) [Var 10])
       (return_to 1 [Var 31])).

Definition unit_expr : expr :=
  Let_static [Bind_code 50 callee]
    (Let 10 Value
       (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
       (Let 11 Value (make_block [Var 10])
          (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
             (Let_cont_non_recursive 20 reader
                (Apply (call_50 [Var 11] 20)))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables cannot_change max_unbox_size
    false.

Definition unboxed_arity : arity := [Unboxed_product [Singleton Value]].

Definition rebuilt_metadata : code_metadata :=
  {| cm_params_arity := unboxed_arity;
     cm_params_subkinds := [];
     cm_result_arity := [Unboxed_product []];
     cm_result_subkinds := [];
     cm_is_tupled := false;
     cm_is_stub := false |}.

(* The rebuilt unit: 50 takes the cell as its parameter 61, which the read
   of the field becomes; the unit binds the field of [x11], the variable
   63, to the cell, and gives it to the call. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_code 50
       (Code [61] 41 3 4 rebuilt_metadata
          (Let 43 Value (Simple (Var 61))
             (Let 44 Value (store (Var 43) 2) (return_to 3 []))))]
    (Let 10 Value
       (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
       (Let 63 Value (Simple (Var 10))
          (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
             (Let_cont_non_recursive 20
                (Handler [] (make_handler_metadata [])
                   (Let 31 Value
                      (Prim (Unary (Block_load access Mutable 0)) [Var 10])
                      (return_to 1 [Var 31])))
                (Apply
                   {| ap_callee := Var 60; ap_call_kind := Direct 50;
                      ap_args := [Var 63]; ap_continuation := 20;
                      ap_exn_continuation := 2; ap_exn_extra_args := [];
                      ap_args_arity := unboxed_arity;
                      ap_return_arity := [Unboxed_product []] |}))))).

Definition cell (n : Z) : cell :=
  Cell_block Cell_mutable 0 [Value_const (Tagged_immediate n)].

Definition heap_at_exit : heap := [(2, cell 2%Z); (2, cell 1%Z)].

End CallsHeapExample.

Definition calls_heap_example_statement : Prop :=
  in_heap_fragment CallsHeapExample.unit_expr = true
  ∧ in_flow_fragment CallsHeapExample.unit_expr = false
  ∧ CallsHeapExample.to_unbox = [Node_variable 11%positive]
  ∧ calls_unboxing CallsHeapExample.facts CallsHeapExample.tables
      CallsHeapExample.cannot_change CallsHeapExample.code_deps
      CallsHeapExample.dominated CallsHeapExample.to_unbox
      CallsHeapExample.unit_expr
    = true
  ∧ check_unboxed_reads CallsHeapExample.facts CallsHeapExample.tables
      CallsHeapExample.dominated CallsHeapExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_fixed CallsHeapExample.facts CallsHeapExample.tables
      CallsHeapExample.cannot_change CallsHeapExample.code_deps
      CallsHeapExample.dominated CallsHeapExample.to_unbox true
      CallsHeapExample.unit_expr
    = CallsHeapExample.rebuilt
  ∧ run 20 (State CallsHeapExample.unit_expr [] [] empty_static_env
              empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 2)]
        CallsHeapExample.heap_at_exit no_reads
  ∧ run 20 (State CallsHeapExample.rebuilt [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 2)]
        CallsHeapExample.heap_at_exit no_reads
  ∧ program_refines
      (rebuild_unit_cc_from_tables CallsHeapExample.facts
         CallsHeapExample.tables CallsHeapExample.cannot_change
         CallsHeapExample.code_deps true CallsHeapExample.unit_expr)
      CallsHeapExample.rebuilt
  ∧ program_refines_up_to_heap CallsHeapExample.unit_expr
      CallsHeapExample.rebuilt.
