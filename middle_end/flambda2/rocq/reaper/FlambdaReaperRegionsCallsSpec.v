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

(* The rebuild with unboxing in the mode of the fix on units with regions,
   calls, code and wrappers, through the erasure of the core, as
   FlambdaReaperRegionsFixedSpec.v has it of the units of the local
   unboxing.

   This file is to be reviewed: the fragment ([regions_calls],
   [sets_not_unboxed], [graph_reads_plain]), the example and the
   statements. FlambdaReaperRegionsCallsProofs.v proves them, with the
   lemmas of FlambdaReaperRegionsCallsRebuildProofs.v (the commutation),
   which need no review.

   WHAT THE REBUILD DOES WITH CALLS, CODE AND WRAPPERS

   (FlambdaReaperUnboxing.v, [rebuild_apply_u], [rebuild_code_u],
   [make_apply_wrapper_u].) The parts of a call are read off the code
   deps (whose nodes are made nodes, never the constants), the
   constructors, the accessors, the parameters and the sources, which
   the erasure keeps; so the call that the rebuild makes is the same in
   the two rebuilds. A wrapper has parameters of the kinds of the return
   arity of the call and of the leaves of the unboxed fields of the
   results, none of the kind [Region] here, so it is erased to itself.
   Code is erased in its body alone, and its decisions (those of its
   parameters, of its results and, in the mode of the fix, of its return
   and exception continuations) are the same in the two rebuilds. A set
   of closures that is not unboxed is rebuilt as the plain rebuild does.

   THE FRAGMENT

   [regions_calls]: that of the plain rebuild
   (FlambdaReaperRegionsSpec.regions_plain), with calls, code and
   closures, and no projection of another closure after the [Let] of a
   closure of a set ([sets_alone]): the closures of
   FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing
   ([closure_binding_ok]). [sets_not_unboxed]: no variable bound to a
   set of closures is dominated by an allocation to unbox, as
   [regions_not_unboxed] says of the variables of the regions; were one
   to, the rebuild would bind the fields of the closure and drop the
   projections that follow. [graph_reads_plain]: the graph reads no
   field of the kind [Region] (FlambdaReaperUnboxing.field_kind is
   [Region] only for a field of a block of that kind, which no primitive
   reads), so that no unboxed field is of that kind;
   FlambdaReaperRegionsUnboxTraverseProofs.closure_free_reads_plain
   proves it of the units of the local unboxing.

   STATEMENTS

   - [rebuild_unbox_fixed_calls_erase_commutes_statement]: on the
     fragment, for any tables, erasing the unit rebuilt in the mode of
     the fix is rebuilding the erased unit in that mode, with its facts,
     its code deps, the same tables, and the fields named after every
     name of the unit.
   - [rebuild_unbox_fixed_calls_regions_refines_statement]: on the
     fragment, where the largest name of the unit is kept by the erasure,
     for any tables whose completion
     (FlambdaReaperRegionsFixedSpec.alias_consts) passes on the erased
     unit the checks of FlambdaReaperUnboxingRefinesCallsSpec
     .rebuild_unbox_fixed_refines_observably_local_statement, the unit
     rebuilt in the mode of the fix refines the unit on the machine of
     regions, if the rebuilt unit is erasable, has no symbol at or above
     the base and passes the check of regions.
   - [regions_calls_example_statement]: a call in a region of code whose
     parameter is unboxed (the unit of
     FlambdaReaperUnboxingRefinesSpec.CallExample in a region).
   - [alias_consts_needs_consts_statement]: where the completion of the
     tables passes the check of the rules on a graph, the constants have
     any usage wherever a variable with any source has any usage. The
     variable of a region is such a variable, so the premise that the
     completion passes [check_closed] on the erased graph asks the
     constants to have any usage in the tables of the graph with regions.
   - [alias_consts_example_statement]: a unit that returns the end of its
     region, whose least solution gives the constants no usage: its
     completion fails the check on the erased graph.

   WHAT IS NOT STATED

   - That the rebuild keeps [erasable] and the freshness of the symbols
     on the fragment (proved of the plain rebuild, the change of calling
     conventions and the local unboxing): they are premises here.
   - [graph_reads_plain] of every unit of the fragment, and
     [sets_not_unboxed] of the least solution.
   - That the completion passes the checks wherever the least solution of
     the graph with regions does and gives the constants any usage: false
     without that usage ([alias_consts_example_statement]); with it, not
     proved (see S7).
   - Unboxed closures, sets with projections, units with a parameter of
     the kind [Region], code that returns a region. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsFixedSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* The variables that the unit binds to sets of closures, in its handlers
   and in the code that it binds too. *)
Fixpoint set_binders (e : expr) : list variable :=
  match e with
  | Let x _ (Set_of_closures _ _) body
  | Let x _ (Closure_of_set _ _ _) body =>
    x :: set_binders body
  | Let _ _ _ body => set_binders body
  | Let_static bindings body =>
    flat_map binding_set_binders bindings ++ set_binders body
  | Let_cont_non_recursive _ h body =>
    handler_set_binders h ++ set_binders body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_set_binders h) handlers
    ++ set_binders body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_set_binders (h : continuation_handler) : list variable :=
  match h with
  | Handler _ _ handler => set_binders handler
  end
with binding_set_binders (binding : static_binding) : list variable :=
  match binding with
  | Bind_code _ c => code_set_binders c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_set_binders (c : code) : list variable :=
  match c with
  | Code _ _ _ _ _ body => set_binders body
  end.

(* No [Let] of a closure of a set is followed by projections of the other
   closures of the set (FlambdaReaperRebuild.projections_of_set). *)
Fixpoint sets_alone (e : expr) : bool :=
  match e with
  | Let x _ d body =>
    match projections_of_set x d body with
    | [] => true
    | _ :: _ => false
    end
    && sets_alone body
  | Let_static bindings body =>
    forallb binding_sets_alone bindings && sets_alone body
  | Let_cont_non_recursive _ h body => handler_sets_alone h && sets_alone body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_sets_alone h) handlers
    && sets_alone body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_sets_alone (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => sets_alone handler
  end
with binding_sets_alone (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_sets_alone c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_sets_alone (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => sets_alone body
  end.

(* The fragment of the plain rebuild, with calls, code and closures whose
   sets are not followed by projections. *)
Definition regions_calls (e : expr) : bool :=
  regions_plain [] e && sets_alone e.

(* No variable of a set of closures is dominated by an allocation to
   unbox. *)
Definition sets_not_unboxed (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  forallb
    (fun x => negb (has_to_be_unboxed dominated to_unbox (Node_variable x)))
    (set_binders e).

(* The graph reads no field of the kind [Region]. *)
Definition graph_reads_plain (F : facts) : bool :=
  forallb (fun '(_, f, _) => kind_not_region (field_kind f)) (f_accessor F).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* Erasing the unit rebuilt with unboxing in the mode of the fix is
   rebuilding the erased unit in that mode, with its facts and code deps,
   the same tables, and the variables of the fields named after every
   name of the unit with its regions. *)
Definition rebuild_unbox_fixed_calls_erase_commutes_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    regions_calls e = true ->
    regions_not_unboxed dominated to_unbox e = true ->
    sets_not_unboxed dominated to_unbox e = true ->
    graph_reads_plain (traverse_unit flag u e) = true ->
    erase []
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e)
    = let '(unboxed, next) :=
        unboxed_fields_of_names (traverse_unit flag u (erase [] e)) T
          (bound_code_my_closures (erase [] e)) dominated to_unbox
          (fresh_name (names_of e))
      in
      rebuild_expr_u true (traverse_unit flag u (erase [] e))
        (solution_of_tables T) (fun n => mem_node n cannot_change)
        (unit_code_deps flag u (erase [] e)) preserve_direct_calls
        (static_names code_ids_bound (erase [] e))
        (bound_code_metadata (erase [] e)) unboxed next [] None
        empty_unbox_env (erase [] e).

(* The rebuild in the mode of the fix refines the unit on the machine of
   regions, for any tables whose completion passes on the erased unit the
   checks of FlambdaReaperUnboxingRefinesCallsSpec.rebuild_unbox_fixed
   _refines_observably_local_statement, where the largest name of the
   unit is one that the erasure keeps, and the rebuilt unit is erasable,
   has no symbol at or above the base and passes the check of regions. *)
Definition rebuild_unbox_fixed_calls_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_calls e = true ->
    regions_not_unboxed dominated to_unbox e = true ->
    sets_not_unboxed dominated to_unbox e = true ->
    graph_reads_plain (traverse_unit flag u e) = true ->
    fresh_name (names_of (erase [] e)) = fresh_name (names_of e) ->
    is_local_slots_only is_local ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    check_closed is_local (traverse_unit flag u (erase [] e)) (alias_consts T)
    = true ->
    check_local_reads_used is_local (traverse_unit flag u (erase [] e))
      (alias_consts T)
    = true ->
    check_field_used_exact is_local (traverse_unit flag u (erase [] e))
      (alias_consts T)
    = true ->
    check_cc_closed (traverse_unit flag u (erase [] e)) (alias_consts T)
      cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u (erase [] e)) (alias_consts T)
    = true ->
    calls_unboxing (traverse_unit flag u (erase [] e)) (alias_consts T)
      cannot_change (unit_code_deps flag u (erase [] e)) dominated to_unbox
      (erase [] e)
    = true ->
    check_unboxed_reads (traverse_unit flag u (erase [] e)) (alias_consts T)
      dominated to_unbox
    = true ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true ->
    erasable
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true ->
    region_check cert empty_static_env
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* The completion of the tables                                             *)
(* ======================================================================== *)

(* A completion that passes the check of the rules on a graph gives the
   constants any usage wherever a variable with any source has any usage:
   it has the alias of that variable to the constants. The variable of a
   region has any source (its begin), and any usage where the region has
   an end (in the least solutions of the examples, whether the variable
   of the end has a use or not), so the premise of
   [rebuild_unbox_fixed_calls_regions_refines_statement] and of
   FlambdaReaperRegionsFixedSpec.rebuild_unbox_fixed_regions_refines
   _statement that [alias_consts T] passes [check_closed] on the erased
   graph holds of a unit with a region only where the constants have any
   usage in [T] ([alias_consts_example_statement]). *)
Definition alias_consts_needs_consts_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables) (x : variable),
    check_closed is_local F (alias_consts T) = true ->
    mem_node (Node_variable x) (t_any_source T) = true ->
    mem_node (Node_variable x) (t_any_usage T) = true ->
    mem_node Node_all_constants (t_any_usage T) = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module RegionsCallsExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples.

Local Open Scope positive_scope.

Module E := FlambdaReaperUnboxing.Examples.
Module C := FlambdaReaperUnboxingRefinesSpec.CallExample.

(* The unit of FlambdaReaperUnboxingRefinesSpec.CallExample, with the call
   in a region that the handler of its result ends:

     let code f50 x40 =
       let x43 = x40.(1) in let x42 = [x43; 9] in return x42
     let x60 = closure of f50
     let r80 = begin_region in
     let_cont k20 x30 =
       let x81 = end_region r80 in
       let x31 = x30.(0) in
       return x31
     in
     let x10 = [7; 8] in
     f50 x60 x10 -> k20 *)
Definition call_in_region : expr :=
  Let_static [Bind_code 50 C.callee]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 80 Region (region Begin_region [])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let 81 Value (region End_region [Var 80])
                   (Let 31 Value (E.load 0 (Var 30))
                      (return_to 1 [Var 31]))))
             (Let 10 Value (E.make_block [E.int 7; E.int 8])
                (Apply (E.call_50 [Var 10] 20)))))).

(* The least solution of its graph: that of CallExample, with the
   variables of the region and of its end, which have any source and any
   usage. *)
Definition call_in_region_tables : tables :=
  {| t_alias := t_alias C.tables;
     t_any_usage := t_any_usage C.tables ++ [E.v 80; E.v 81];
     t_any_source := t_any_source C.tables ++ [E.v 80; E.v 81];
     t_has_usage := t_has_usage C.tables ++ [E.v 80; E.v 81];
     t_has_source := t_has_source C.tables ++ [E.v 80; E.v 81];
     t_usages := t_usages C.tables;
     t_sources := t_sources C.tables;
     t_reading_field := t_reading_field C.tables;
     t_escaping_field := t_escaping_field C.tables;
     t_field_used := t_field_used C.tables |}.

(* The rebuilt unit: the parameter of the code and its result are
   unboxed, the call gives the field [8] that the code reads and returns
   its field to [k20], which reads it in place of the load; the region is
   kept. *)
Definition call_in_region_rebuilt : expr :=
  Let_static
    [Bind_code 50
       (Code [87] 41 3 4
          {| cm_params_arity := [Unboxed_product [Singleton Value]];
             cm_params_subkinds := [];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_result_subkinds := [];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (Let 43 Value (Simple (Var 87))
             (Let 83 Value (Simple (Var 43)) (return_to 3 [Var 83]))))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 80 Region (region Begin_region [])
          (Let_cont_non_recursive 20
             (Handler [85] (make_handler_metadata [Value])
                (Let 81 Value (region End_region [Var 80])
                   (Let 31 Value (Simple (Var 85)) (return_to 1 [Var 31]))))
             (Let 82 Value (Simple (E.int 8))
                (Apply
                   {| ap_callee := Var 60; ap_call_kind := Direct 50;
                      ap_args := [Var 82]; ap_continuation := 20;
                      ap_exn_continuation := 2; ap_exn_extra_args := [];
                      ap_args_arity := [Unboxed_product [Singleton Value]];
                      ap_return_arity := [Unboxed_product [Singleton Value]]
                   |}))))).

(* The certificate of the check of regions of the rebuilt unit: [k20] is
   entered with [r80] open and its parameter living for ever, and the code
   50 is given no local value, has no local closure and returns no local
   value. *)
Definition call_in_region_cert : certificate :=
  Certificate [(20, Handler_entry [80] [Forever])]
    [(50, Code_entry [false] false false)] false false false.

(* A region whose end is returned:

     let r80 = begin_region in
     let x81 = end_region r80 in
     return x81

   and the least solution of its graph, in which nothing gives the
   constants a usage. *)
Definition end_returned : expr :=
  Let 80 Region (region Begin_region [])
    (Let 81 Value (region End_region [Var 80]) (return_to 1 [Var 81])).

Definition end_returned_tables : tables :=
  {| t_alias := [(E.m 1, E.v 81)];
     t_any_usage := [E.v 80; E.m 2; E.v 81; E.m 1];
     t_any_source := [E.m 1; E.consts; E.v 80; E.v 81];
     t_has_usage := [E.v 80; E.m 1; E.v 81; E.m 2];
     t_has_source := [E.m 1; E.v 81; E.v 80; E.consts];
     t_usages := [];
     t_sources := [];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [] |}.

End RegionsCallsExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsCallsExamples.

(* The statement applies to [call_in_region]: the tables are the least
   solution of its graph, and their completion is closed under the rules
   on the erased graph; the rules unbox the block [x10] and the result
   [x42] of the code; the rebuild in the mode of the fix makes
   [call_in_region_rebuilt], which passes the check of regions; both exit
   with 8 on the machine of regions; and, through
   [rebuild_unbox_fixed_calls_regions_refines_statement], the rebuilt
   unit refines the unit there. *)
Definition regions_calls_example_statement : Prop :=
  least_solution (fun _ => false)
    (traverse_unit Auto interface call_in_region) call_in_region_tables
  ∧ check_closed (fun _ => false)
      (traverse_unit Auto interface (erase [] call_in_region))
      (alias_consts call_in_region_tables)
    = true
  ∧ least_to_unbox (fun _ => false)
      (traverse_unit Auto interface call_in_region) call_in_region_tables 10
      false
    = [Node_variable 42%positive; Node_variable 10%positive]
  ∧ rebuild_unit_unbox_fixed (traverse_unit Auto interface call_in_region)
      call_in_region_tables
      (derive_cannot_change (traverse_unit Auto interface call_in_region)
         call_in_region_tables)
      (unit_code_deps Auto interface call_in_region)
      (derive_dominated call_in_region_tables)
      (least_to_unbox (fun _ => false)
         (traverse_unit Auto interface call_in_region) call_in_region_tables
         10 false)
      true call_in_region
    = call_in_region_rebuilt
  ∧ region_check call_in_region_cert empty_static_env call_in_region_rebuilt
    = true
  ∧ (∃ rs : region_state,
       region_run 40
         (initial_region_state call_in_region empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 8)] rs)
  ∧ (∃ rs : region_state,
       region_run 40
         (initial_region_state call_in_region_rebuilt empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 8)] rs)
  ∧ regions_refine_observably empty_static_env base call_in_region
      call_in_region_rebuilt.

(* [end_returned]: its tables are the least solution of its graph, the
   constants have no usage there, and their completion fails the check of
   the rules on the erased graph (the variables [r80] and [x81] have any
   source and any usage, and aliases to the constants there). *)
Definition alias_consts_example_statement : Prop :=
  least_solution (fun _ => false) (traverse_unit Auto interface end_returned)
    end_returned_tables
  ∧ mem_node Node_all_constants (t_any_usage end_returned_tables) = false
  ∧ check_closed (fun _ => false)
      (traverse_unit Auto interface (erase [] end_returned))
      (alias_consts end_returned_tables)
    = false.
