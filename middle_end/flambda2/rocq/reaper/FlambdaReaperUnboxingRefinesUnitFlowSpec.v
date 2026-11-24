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

(* The fragments of FlambdaReaperUnboxingRefinesReach3Spec.v and
   FlambdaReaperUnboxingRefinesNeverSpec.v, with the checks of the flow
   read off the analysis of the unit itself (its facts [F] and tables [T])
   rather than off that of the first rebuilt unit, analysed again ([F1],
   [T1]).

   This file is to be reviewed: the fragments
   ([calls_unboxing_unit_flow3_nr], [calls_unboxing_unit_never_nr]), the
   check that links the two analyses ([used_reach_covered]), the
   statements, the examples and the counterexample. The proofs are in
   FlambdaReaperUnboxingRefinesUnitFlowProofs.v, which needs no review.

   The reach fragments read three things off [F1] and [T1]: the code that
   the flow may bring to the callee [f] of a call ([flow_reach2]), the code
   that a call may over-apply ([over_reach]), and the code that what is
   left of an over-application may call ([rem_reach]). Each check asks
   something of the code it reads, so that a larger reach makes a stronger
   check: the fragment with the reach of the unit's own analysis implies
   the fragment with any smaller one, at the variables with a use (the
   callee of a call whose code is not known has one, which the fragment
   checks) and the code of the unit.

   [used_reach_covered F T F1 T1 e]: at every variable of [e] with a use in
   [T] and every code id of [e], the reach of [F1], [T1] is within that of
   [F], [T], both of known arity and not; and so are [over_reach] and
   [rem_reach] at every code id of [e]. That the first rebuild brings no
   more code anywhere, analysed again, is the open part: it needs the
   facts of the traversal of the first rebuilt unit to be those of the unit
   up to the names that the rebuild makes (wrappers, the parameters and
   results that the change of calling conventions renumbers, the calls
   that it makes direct), and the least solution to be monotone in the
   facts, which the rules give only up to "any source" (their guards read
   [s_any_source] and [s_any_usage] negated). It holds of the units of the
   examples of the reach fragments ([unit_flow_examples_statement]).

   It does not hold at variables with no use ([poison_reach_statement]):
   the first rebuild gives poison, a constant, to a parameter with no use,
   and in the analysis of the first rebuilt unit a constant has any
   source, so that the parameter may hold the closure of any code there.
   No check reads the reach at such a variable.

   [rebuild_unbox_unit_flow3_refines_cc_statement],
   [rebuild_unbox_unit_flow3_refines_observably_statement] and
   [rebuild_unbox_unit_flow3_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesReach3Spec.v with
   [calls_unboxing_unit_flow3_nr], the fragment read off the unit's own
   analysis, and [used_reach_covered] in place of
   [calls_unboxing_flow3_nr].

   [rebuild_unbox_unit_never_refines_cc_statement],
   [rebuild_unbox_unit_never_refines_observably_statement] and
   [rebuild_unbox_unit_never_refines_observably_local_statement]: the
   same for the fragment of FlambdaReaperUnboxingRefinesNeverSpec.v, with
   the code taken to never return that of the unit's own analysis
   ([used_never_returns]: used, some result of its code dep with no
   source), as in FlambdaReaperNeverReturnsSpec.v. The check of
   [live_conts] on that code stays in the fragment.

   [unit_flow_examples_statement]: the units of [flow_example_statement],
   [flow2_example_statement], [flow3_example_statement] and
   [never_ind_example_statement] are of the fragments read off their own
   analyses (the last with [used_never_returns]), and
   [used_reach_covered] holds of each, with the least solutions.
   [unit_flow3_example_statement] and [unit_never_example_statement]: the
   rebuilt units of [flow3_example_statement] and
   [never_ind_example_statement] refine the units up to observation, by the
   new statements.

   [poison_reach_statement]: a unit whose codes 50 and 51 have a parameter
   with no use, and which calls the closure of 50 or of 51 (chosen by a
   [Switch]) at known arity with the closure [x61] of 51 as argument. The
   first rebuild gives the call poison. With the least solutions, the
   parameter [x40] of 50 has no use; the unit's analysis brings it the
   closure of 51 only, that of the first rebuilt unit any code (poison has
   any source). At the callee [x31], which has a use, the two agree, and
   [used_reach_covered] holds. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragments read off the unit's own analysis                          *)
(* ======================================================================== *)

(* [calls_unboxing_flow3_nr] with the flow of the unit's facts [F] and
   tables [T]. *)
Definition calls_unboxing_unit_flow3_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  calls_unboxing_reach_rem_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F T) (over_reach F T) (rem_reach F T) e.

(* [calls_unboxing_never_nr] likewise. *)
Definition calls_unboxing_unit_never_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  calls_unboxing_reach_never_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F T) (over_reach F T) (rem_reach F T) e.

(* The analysis [F1], [T1] brings to the variables of [e] with a use in
   [T], and to the code of [e], no more code than [F], [T]. *)
Definition used_reach_covered (F : facts) (T : tables) (F1 : facts)
    (T1 : tables) (e : expr) : bool :=
  let ids := map fst (bound_code_metadata e) in
  forallb
    (fun f =>
       negb (s_has_usage (solution_of_tables T) (Node_variable f))
       || forallb
            (fun id =>
               forallb
                 (fun b =>
                    negb (flow_reach2 F1 T1 b f id)
                    || flow_reach2 F T b f id)
                 [true; false])
            ids)
    (variables_of e)
  && forallb
       (fun id =>
          (negb (over_reach F1 T1 id) || over_reach F T id)
          && (negb (rem_reach F1 T1 id) || rem_reach F T id))
       ids.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_unit_flow3_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
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
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
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

Definition rebuild_unbox_unit_flow3_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
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
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_flow3_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
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
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
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

Definition rebuild_unbox_unit_never_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
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
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
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
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Definition unit_flow_examples_statement : Prop :=
  used_reach_covered FlowExample.facts FlowExample.tables FlowExample.facts1
    FlowExample.tables1 FlowExample.unit_expr
  = true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) FlowExample.facts
      FlowExample.tables FlowExample.cannot_change FlowExample.code_deps [] []
      FlowExample.unit_expr
    = true
  ∧ used_reach_covered Flow2Example.facts Flow2Example.tables
      Flow2Example.facts1 Flow2Example.tables1 Flow2Example.unit_expr
    = true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) Flow2Example.facts
      Flow2Example.tables Flow2Example.cannot_change Flow2Example.code_deps
      [] [] Flow2Example.unit_expr
    = true
  ∧ used_reach_covered Flow3Example.facts Flow3Example.tables
      Flow3Example.facts1 Flow3Example.tables1 Flow3Example.unit_expr
    = true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) Flow3Example.facts
      Flow3Example.tables Flow3Example.cannot_change Flow3Example.code_deps
      [] [] Flow3Example.unit_expr
    = true
  ∧ used_reach_covered NeverIndExample.facts NeverIndExample.tables
      NeverIndExample.facts1 NeverIndExample.tables1
      NeverIndExample.unit_expr
    = true
  ∧ calls_unboxing_unit_never_nr NeverIndExample.nr NeverIndExample.facts
      NeverIndExample.tables NeverIndExample.cannot_change
      NeverIndExample.code_deps [] [] NeverIndExample.unit_expr
    = true.

Definition unit_flow3_example_statement : Prop :=
  program_refines_observably Flow3Example.unit_expr
    (rebuild_unit_unbox_fixed Flow3Example.facts Flow3Example.tables
       Flow3Example.cannot_change Flow3Example.code_deps [] [] true
       Flow3Example.unit_expr).

Definition unit_never_example_statement : Prop :=
  program_refines_observably NeverIndExample.unit_expr
    (rebuild_unit_unbox_fixed NeverIndExample.facts NeverIndExample.tables
       NeverIndExample.cannot_change NeverIndExample.code_deps [] [] true
       NeverIndExample.unit_expr).

(* ======================================================================== *)
(* Counterexample: the reach at a variable with no use                      *)
(* ======================================================================== *)

(* The codes 50 and 51 do not use their parameters [x40] and [x42]. The
   unit calls the closure [x60] of 50 or [x61] of 51, chosen by a
   [Switch], at known arity, with [x61] as argument. The first rebuild
   gives the call poison, which the parameters take. *)
Module PoisonReach.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call (callee : variable) (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Indirect_known_arity None;
     ap_args := args; ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := []; ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata (return_to 3 [int 0]));
     Bind_code 51 (Code [42] 43 5 6 metadata (return_to 5 [int 1]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (return_to 1 [Var 30]))
             (Let_cont_non_recursive 21
                (Handler [31] (make_handler_metadata [Value])
                   (Apply (call 31 [Var 61] 20)))
                (Switch (Const (Naked_immediate 0))
                   [(0%Z, jump 21 [Var 60]); (1%Z, jump 21 [Var 61])]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

End PoisonReach.

Definition poison_reach_statement : Prop :=
  s_has_usage (solution_of_tables PoisonReach.tables)
    (Node_variable 40%positive)
  = false
  ∧ map (flow_reach PoisonReach.facts PoisonReach.tables 40%positive)
      [50; 51]%positive
    = [false; true]
  ∧ map (flow_reach PoisonReach.facts1 PoisonReach.tables1 40%positive)
      [50; 51]%positive
    = [true; true]
  ∧ s_has_usage (solution_of_tables PoisonReach.tables)
      (Node_variable 31%positive)
    = true
  ∧ map (flow_reach PoisonReach.facts1 PoisonReach.tables1 31%positive)
      [50; 51]%positive
    = [true; true]
  ∧ map (flow_reach PoisonReach.facts PoisonReach.tables 31%positive)
      [50; 51]%positive
    = [true; true]
  ∧ used_reach_covered PoisonReach.facts PoisonReach.tables
      PoisonReach.facts1 PoisonReach.tables1 PoisonReach.unit_expr
    = true.
