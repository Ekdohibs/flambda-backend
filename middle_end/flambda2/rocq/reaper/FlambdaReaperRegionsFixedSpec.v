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

(* The rebuild with unboxing in the mode of the fix (the unboxing across
   calls of FlambdaReaperUnboxingRefinesCallsSpec.v) on units with
   regions, through the erasure of the core, as
   FlambdaReaperRegionsUnboxSpec.v has it of the rebuild in the mode of
   the branch.

   This file is to be reviewed: [alias_consts], the example and the
   statements. FlambdaReaperRegionsFixedProofs.v proves them, with the
   lemmas of FlambdaReaperRegionsUnboxRebuildProofs.v (the commutation,
   in either mode) and FlambdaReaperRegionsUnboxKeepsProofs.v (what the
   rebuild keeps), which need no review.

   THE TABLES OF THE ERASED UNIT ([alias_consts])

   The statements of FlambdaReaperUnboxingRefinesCallsSpec.v ask the
   tables to pass the check of the rules ([check_closed]) on the graph of
   the unit they rebuild. The erasure binds the variable of each end of a
   region to the unit value, so the graph of the erased unit has an alias
   of that variable to the constants, which the tables of the graph with
   regions, the compiler's, need not have
   (FlambdaReaperRegionsUnboxSpec.regions_unbox_example_statement).
   [alias_consts T] adds to the aliases of [T] one of each variable with
   any source to the constants (the aliases of
   FlambdaReaperRegionsProofs.consts_used), and nothing else: the
   rebuilds read no alias, so they rebuild every unit with [alias_consts
   T] as with [T] ([alias_consts_rebuild_statement]). The statements of
   the calls are then applied, as they are, to the erased unit and
   [alias_consts T], with their checks.

   THE NAMES

   The rebuild names the variables of the unboxed fields after every name
   of the unit it rebuilds; the statements of the calls name them after
   the names of the erased unit, which lacks the variables of the begins.
   The statement here asks that the largest name of the unit be one that
   the erasure keeps ([fresh_name] of the two the same): otherwise the
   two rebuilt units differ by the names of those variables only, which
   no lemma relates. The compiler takes its names from a counter.

   STATEMENTS

   - [alias_consts_rebuild_statement]: the rebuild with unboxing, in
     either mode, of any unit with [alias_consts T] is that with [T].
   - [rebuild_unbox_fixed_erase_commutes_statement]: on the units of
     FlambdaReaperRegionsUnboxSpec.regions_local with
     [regions_not_unboxed], erasing the unit rebuilt in the mode of the
     fix is rebuilding the erased unit in that mode, with its facts, the
     same tables and the fields named after every name of the unit.
   - [rebuild_unbox_fixed_keeps_erasable_statement],
     [rebuild_unbox_fixed_keeps_fresh_statement]: the rebuilt unit is
     erasable, and has no symbol at or above the base, wherever the unit
     is and has none.
   - [rebuild_unbox_fixed_regions_refines_statement]: on the same
     fragment, with the names above, for any tables whose completion
     passes on the erased unit the checks of
     FlambdaReaperUnboxingRefinesCallsSpec.rebuild_unbox_fixed_refines
     _observably_local_statement (the premises of that statement, there),
     the unit rebuilt in the mode of the fix with the tables refines the
     unit on the machine of regions (regions_refine_observably) if it
     passes the check of regions.
   - [least_rebuild_unbox_fixed_regions_refines_statement]: the same for
     the least solution of the graph with regions and the tables that the
     rules give from it, where [regions_not_unboxed] follows.
   - [regions_fixed_example_statement]: the unit of
     FlambdaReaperRegionsUnboxSpec.RegionsUnboxExamples: its least
     solution is not closed under the rules on the erased graph, its
     completion is, the premises hold, the rebuild in the mode of the fix
     is that of the branch, and both exit with 7 on the machine of
     regions.

   WHAT IS NOT STATED

   - The commutation with calls, code and wrappers: the fragment is that
     of the local unboxing, with no closure, call or code (see S7).
   - That the completion passes the check wherever the least solution of
     the graph with regions does (it asks of the constants a use wherever
     a variable of an end has one), and the checks of the calls statement
     from those of the graph with regions: they are premises here. *)

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
  FlambdaReaperRegionsUnboxSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The tables of the erased unit                                            *)
(* ======================================================================== *)

(* The aliases to the constants of the variables with any source. *)
Definition consts_aliases (l : list node) : list (node * node) :=
  flat_map
    (fun n =>
       match n with
       | Node_variable _ => [(n, Node_all_constants)]
       | _ => []
       end)
    l.

(* The tables [T] with those aliases. *)
Definition alias_consts (T : tables) : tables :=
  {| t_alias := t_alias T ++ consts_aliases (t_any_source T);
     t_any_usage := t_any_usage T;
     t_any_source := t_any_source T;
     t_has_usage := t_has_usage T;
     t_has_source := t_has_source T;
     t_usages := t_usages T;
     t_sources := t_sources T;
     t_reading_field := t_reading_field T;
     t_escaping_field := t_escaping_field T;
     t_field_used := t_field_used T |}.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The rebuilds read no alias. *)
Definition alias_consts_rebuild_statement : Prop :=
  ∀ (fixed : bool) (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (e : expr),
    rebuild_unit_unbox_repr_mode fixed is_local F (alias_consts T)
      cannot_change code_deps dominated to_unbox tcr preserve_direct_calls e
    = rebuild_unit_unbox_repr_mode fixed is_local F T cannot_change code_deps
        dominated to_unbox tcr preserve_direct_calls e.

(* Erasing the unit rebuilt with unboxing in the mode of the fix is
   rebuilding the erased unit in that mode, with its facts, the same
   tables, and the variables of the fields named after every name of the
   unit with its regions. *)
Definition rebuild_unbox_fixed_erase_commutes_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    regions_local e = true ->
    regions_not_unboxed dominated to_unbox e = true ->
    erase []
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         code_deps dominated to_unbox preserve_direct_calls e)
    = let '(unboxed, next) :=
        unboxed_fields_of_names (traverse_unit flag u (erase [] e)) T
          (bound_code_my_closures (erase [] e)) dominated to_unbox
          (fresh_name (names_of e))
      in
      rebuild_expr_u true (traverse_unit flag u (erase [] e))
        (solution_of_tables T) (fun n => mem_node n cannot_change) code_deps
        preserve_direct_calls (static_names code_ids_bound (erase [] e))
        (bound_code_metadata (erase [] e)) unboxed next [] None
        empty_unbox_env (erase [] e).

Definition rebuild_unbox_fixed_keeps_erasable_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    regions_local e = true ->
    erasable e empty_static_env = true ->
    erasable
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         code_deps dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true.

Definition rebuild_unbox_fixed_keeps_fresh_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool) (base : symbol),
    regions_local e = true ->
    region_names_fresh base e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         code_deps dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true.

(* The rebuild in the mode of the fix refines the unit on the machine of
   regions, for any tables whose completion passes on the erased unit the
   checks of FlambdaReaperUnboxingRefinesCallsSpec.rebuild_unbox_fixed
   _refines_observably_local_statement (its premises, on the erased unit
   with the completion of the tables), where the largest name of the unit
   is one that the erasure keeps and the rebuilt unit passes the check of
   regions. *)
Definition rebuild_unbox_fixed_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_local e = true ->
    regions_not_unboxed dominated to_unbox e = true ->
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
    region_check cert empty_static_env
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same for the least solution of the graph of the unit with its
   regions and the tables that the rules give from it: no allocation to
   unbox dominates the variable of a region. *)
Definition least_rebuild_unbox_fixed_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (max_unbox_size : nat)
    (lambda_lifting preserve_direct_calls : bool) (cert : certificate)
    (base : symbol),
    regions_local e = true ->
    fresh_name (names_of (erase [] e)) = fresh_name (names_of e) ->
    is_local_slots_only is_local ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_closed is_local (traverse_unit flag u (erase [] e)) (alias_consts T)
    = true ->
    check_local_reads_used is_local (traverse_unit flag u (erase [] e))
      (alias_consts T)
    = true ->
    check_field_used_exact is_local (traverse_unit flag u (erase [] e))
      (alias_consts T)
    = true ->
    check_cc_closed (traverse_unit flag u (erase [] e)) (alias_consts T)
      (derive_cannot_change (traverse_unit flag u e) T)
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u (erase [] e)) (alias_consts T)
    = true ->
    calls_unboxing (traverse_unit flag u (erase [] e)) (alias_consts T)
      (derive_cannot_change (traverse_unit flag u e) T)
      (unit_code_deps flag u (erase [] e)) (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
      (erase [] e)
    = true ->
    check_unboxed_reads (traverse_unit flag u (erase [] e)) (alias_consts T)
      (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
    = true ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_check cert empty_static_env
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) (derive_dominated T)
         (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
            lambda_lifting)
         preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) (derive_dominated T)
         (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
            lambda_lifting)
         preserve_direct_calls e).

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsUnboxExamples.

(* The unit [local_pair] of FlambdaReaperRegionsUnboxSpec.v, rebuilt in
   the mode of the fix: the least solution of its graph is not closed
   under the rules on the graph of the erased unit, and its completion
   is; the largest name of the unit is not the variable of the region;
   the erased unit is of the fragment of the calls with the completion;
   the rebuild in the mode of the fix is that of the branch; and,
   through [least_rebuild_unbox_fixed_regions_refines_statement], the
   rebuilt unit refines the unit on the machine of regions. *)
Definition regions_fixed_example_statement : Prop :=
  check_closed (fun _ => false)
    (traverse_unit Auto interface (erase [] local_pair)) local_pair_tables
  = false
  ∧ check_closed (fun _ => false)
      (traverse_unit Auto interface (erase [] local_pair))
      (alias_consts local_pair_tables)
    = true
  ∧ fresh_name (names_of (erase [] local_pair))
    = fresh_name (names_of local_pair)
  ∧ calls_unboxing (traverse_unit Auto interface (erase [] local_pair))
      (alias_consts local_pair_tables)
      (derive_cannot_change (traverse_unit Auto interface local_pair)
         local_pair_tables)
      (unit_code_deps Auto interface (erase [] local_pair))
      (derive_dominated local_pair_tables) local_pair_to_unbox
      (erase [] local_pair)
    = true
  ∧ rebuild_unit_unbox_fixed (traverse_unit Auto interface local_pair)
      local_pair_tables
      (derive_cannot_change (traverse_unit Auto interface local_pair)
         local_pair_tables)
      (unit_code_deps Auto interface local_pair)
      (derive_dominated local_pair_tables) local_pair_to_unbox true local_pair
    = local_pair_rebuilt
  ∧ regions_refine_observably empty_static_env base local_pair
      local_pair_rebuilt.
