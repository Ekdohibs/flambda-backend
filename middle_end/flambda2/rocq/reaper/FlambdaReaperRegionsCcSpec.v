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

(* The rebuild of the reaper with changes of calling convention on units
   with regions, through the erasure of the core, as
   FlambdaReaperRegionsSpec.v has it of the plain rebuild.

   This file is to be reviewed: the fragment [regions_cc_plain], the
   example and the statements. FlambdaReaperRegionsCcProofs.v proves them,
   with the lemmas of FlambdaReaperRegionsCcRebuildProofs.v (the
   commutation), FlambdaReaperRegionsCcBaseProofs.v (the refinement of the
   rebuild for any first fresh name) and
   FlambdaReaperRegionsKeepsProofs.v (what the rebuild keeps), which need
   no review.

   WHAT THE REBUILD DOES WITH REGIONS

   What the plain rebuild does (FlambdaReaperRegionsSpec.v), case for
   case, but for a call and for code (FlambdaReaperCallingConventions.v).
   Code that changes its calling convention keeps the parameters and the
   results that have a use, and those of the kind [Region], which always
   have one ([is_node_used]). A call of such code gives the arguments
   kept and expects the results kept; where its continuation keeps other
   results than the callee returns, it returns to a wrapper, whose
   parameters are the results that the callee returns, at their kinds in
   the metadata of the callee.

   THE FRAGMENT ([regions_cc_plain])

   That of the plain rebuild ([regions_plain]), and no result of the code
   that the unit binds of the kind [Region] ([code_results_plain]).
   [regions_plain] asks it of the results of every call; this asks it of
   the results of the code, of which the parameters of a wrapper are: a
   wrapper with a parameter of that kind is not erased to itself.

   THE COMMUTATION ([rebuild_cc_erase_commutes_statement])

   On the fragment, for any tables and any table of the code that cannot
   change its calling convention: erasing the rebuilt unit is rebuilding
   the erased unit, with the facts and the code deps of the traversal of
   the erased unit, the metadata of its code, the same tables and table,
   and the first fresh name of the unit with its regions. The two
   traversals have the same code deps, and the least table of the code
   that cannot change is the same for both, as [derive_cannot_change]
   reads of the facts the constructors and the accessors only, which are
   the same ([cc_tables_erase_statement]).

   THE REFINEMENT ([least_rebuild_cc_regions_refines_statement])

   For a least solution of the graph of the unit with its regions, which
   is the analysis that the compiler runs, its least table of the code
   that cannot change ([derive_cannot_change]), and the premises of
   FlambdaReaperLeastSpec.least_rebuild_cc_refines_observably_statement
   on the erased unit, but [check_widgets_read_only], which is asked of
   the graph of the unit with its regions (the traversal passes it with
   the flag [Auto]: [least_rebuild_cc_auto_regions_refines_statement]): if
   the rebuilt unit passes the check of regions, with any certificate, it
   refines the unit on the machine of regions
   (FlambdaReaperRegionsSpec.regions_refine_observably). The rebuild keeps
   [erasable] and the freshness of the symbols on the fragment
   ([rebuild_cc_keeps_erasable_statement],
   [rebuild_cc_keeps_fresh_statement]), so nothing else is asked of the
   rebuilt unit.

   The proof is that of FlambdaReaperRegionsSpec.v: the erased unit is
   refined by its rebuild with the least solution of the unit with the
   tuples that the erasure adds (the constants have every usage), and the
   same table of the code that cannot change. That table is closed under
   the rules of [calling_convention_changes_rules] on the erased graph and
   that solution, with the table of the witnesses that cannot change and
   the constants: no constructor of the graph is of the constants, no
   accessor reads into them, and a least solution has [usages] of the
   constants of the constants only. The decisions of code are read off
   the nodes of its code dep, which are made nodes, so they are the same
   in both solutions. By the commutation that rebuild is the erasure of
   the rebuilt unit.

   THE EXAMPLE ([regions_cc_example_statement])

   A call in a region of code whose second parameter and two last results
   have no use: the code changes its calling convention, the call gives
   one argument and expects one result, and the unit and its rebuild
   exit with 7 on the machine of regions.

   WHAT IS NOT STATED

   - The rebuild with unboxing across calls. The local unboxing is
     stated in FlambdaReaperRegionsUnboxSpec.v.
   - Units with a parameter of the kind [Region], with a local set of
     several closures, or with code that returns a region.
   - Anything of a run that goes on for ever, or of the heap at the
     exit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeastSpec FlambdaRegionsSpec FlambdaRegionsErasureSpec
  FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* No result of the code that the unit binds, where
   FlambdaReaperCallingConventions.bound_code_metadata looks, is of the
   kind [Region]. *)
Definition code_results_plain (e : expr) : bool :=
  forallb
    (fun '(_, metadata) =>
       forallb kind_not_region (unarize (cm_result_arity metadata)))
    (bound_code_metadata e).

Definition regions_cc_plain (e : expr) : bool :=
  regions_plain [] e && code_results_plain e.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* Erasing the rebuilt unit is rebuilding the erased unit: with the facts
   and the code deps of the erased unit, the metadata of its code, the
   same tables and table, and the first fresh name of the unit with its
   regions. *)
Definition rebuild_cc_erase_commutes_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node) (preserve_direct_calls : bool),
    regions_cc_plain e = true ->
    erase []
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
    = rebuild_expr_cc (traverse_unit flag u (erase [] e))
        (solution_of_tables T) (fun n => mem_node n cannot_change)
        (unit_code_deps flag u (erase [] e)) preserve_direct_calls
        (static_names code_ids_bound (erase [] e))
        (bound_code_metadata (erase [] e)) (fresh_name (names_of e)) None
        FlambdaReaperRebuild.empty_env (erase [] e).

(* The least table of the code that cannot change its calling convention,
   and the code deps, are those of the erased unit. *)
Definition cc_tables_erase_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables),
    regions_plain [] e = true ->
    derive_cannot_change (traverse_unit flag u (erase [] e)) T
    = derive_cannot_change (traverse_unit flag u e) T
    ∧ unit_code_deps flag u (erase [] e) = unit_code_deps flag u e.

(* On the fragment, for any facts, tables, table and code deps, the
   rebuilt unit is erasable wherever the unit is, and has no symbol at or
   above the base wherever the unit has none. *)
Definition rebuild_cc_keeps_erasable_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr),
    regions_cc_plain e = true ->
    erasable e empty_static_env = true ->
    erasable
      (rebuild_unit_cc_from_tables F T cannot_change code_deps
         preserve_direct_calls e)
      empty_static_env
    = true.

Definition rebuild_cc_keeps_fresh_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr) (base : symbol),
    regions_cc_plain e = true ->
    region_names_fresh base e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_cc_from_tables F T cannot_change code_deps
         preserve_direct_calls e)
      empty_static_env
    = true.

(* The rebuild with changes of calling convention, for the least solution
   of the graph of the unit with its regions and its least table, refines
   the unit on the machine of regions, if the rebuilt unit passes the
   check of regions. The premises on the erased unit are those of
   FlambdaReaperLeastSpec.least_rebuild_cc_refines_observably_statement. *)
Definition least_rebuild_cc_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_cc_plain e = true ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free (erase [] e) = true ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_widgets_read_only (traverse_unit flag u e) = true ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_check cert empty_static_env
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) preserve_direct_calls e).

(* With [Auto], without the check of the graph. *)
Definition least_rebuild_cc_auto_regions_refines_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (cert : certificate) (base : symbol),
    regions_cc_plain e = true ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free (erase [] e) = true ->
    in_flow_fragment (erase [] e) = true ->
    any_kind_arguments (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_check cert empty_static_env
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
         (derive_cannot_change (traverse_unit Auto u e) T)
         (unit_code_deps Auto u e) preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
         (derive_cannot_change (traverse_unit Auto u e) T)
         (unit_code_deps Auto u e) preserve_direct_calls e).

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module RegionsCcExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples.

Local Open Scope positive_scope.

(* The code 50 has the parameters [x40] and [x42] and returns
   [x40; x40; x42] (that of FlambdaReaperCallingConventions.Examples). *)
Definition code_50 : code :=
  Code [40; 42] 41 3 4
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value; Singleton Value])
    (return_to 3 [Var 40; Var 40; Var 42]).

Definition call_50 (args : list simple) (k : continuation) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value; Singleton Value] |}.

(* A call of the code 50 in a region:

     let code f50 x40 x42 = return (x40, x40, x42)
     let x60 = closure of f50
     let x61 = 3 in
     let r10 = begin_region in
     let x11 = local [7] in
     let x12 = x11.(0) in
     let_cont k20 (x30, x31, x32) =
       let x14 = end_region r10 in
       return x30
     in
     f50 x60 (x12, x61) -> k20

   The second parameter and the two last results of the code have no
   use: the code changes its calling convention, and the call with it. *)
Definition call_in_region : expr :=
  Let_static [Bind_code 50 code_50]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Simple (Const (Tagged_immediate 3)))
          (Let 10 Region (region Begin_region [])
             (Let 11 Value (make_block [Const (Tagged_immediate 7)])
                (Let 11 Value (region Local_allocation [Var 11; Var 10])
                   (Let 12 Value (load_immediate 0 (Var 11))
                      (Let_cont_non_recursive 20
                         (Handler [30; 31; 32]
                            (make_handler_metadata [Value; Value; Value])
                            (Let 14 Value (region End_region [Var 10])
                               (return_to 1 [Var 30])))
                         (Apply (call_50 [Var 12; Var 61] 20))))))))).

(* Short names for the nodes and fields of the tables. *)
Definition v (x : positive) : node := Node_variable x.
Definition m (n : positive) : node := Node_made n.
Definition c (id : positive) : node := Node_code_id id.
Definition consts : node := Node_all_constants.
Definition known : field := Field_call_witness Known_arity_code_pointer.
Definition result (i : nat) : field := Field_return_of_call (Normal i).

(* The least solution of the graph of [call_in_region]. The code dep of
   50 is made first: its results 3, 4 and 5, its exception 6, its
   [my_closure] 7, its parameters 8 and 9, its witnesses 10 (of known
   arity), 11 and 12. The call makes the witness 14. Neither the
   parameter 9 nor the results 4 and 5 have a use. *)
Definition call_in_region_tables : tables :=
  {| t_alias :=
       [(m 7, v 60); (m 9, v 61); (m 8, v 12); (m 15, c 50); (m 2, m 6);
        (v 32, m 5); (v 31, m 4); (v 30, m 3); (v 12, consts);
        (m 16, m 10); (v 42, m 9); (v 40, m 8); (v 41, m 7); (m 5, v 42);
        (m 4, v 40); (m 3, v 40); (v 61, consts); (m 1, v 30);
        (m 14, m 16); (m 7, m 17); (m 18, m 10)];
     t_any_usage :=
       [consts; v 12; m 8; v 40; m 3; m 6; c 50; v 10; v 30; m 2; v 14;
        m 15; m 1];
     t_any_source :=
       [m 1; v 30; v 31; v 32; m 3; m 4; m 5; v 40; v 42; m 15; m 8; m 9;
        v 12; v 61; consts; c 50; v 10; v 14];
     t_has_usage :=
       [v 11; consts; v 12; m 8; v 40; v 60; m 10; m 3; m 6; c 50; m 16;
        v 10; m 14; v 30; m 1; m 15; v 14; m 2];
     t_has_source :=
       [m 1; v 41; v 30; v 31; v 32; m 3; m 4; m 5; m 7; v 40; v 42; m 14;
        m 15; m 8; m 9; m 13; m 18; m 16; m 10; m 11; m 12; v 11; v 12;
        v 61; v 14; v 10; c 50; consts; v 60];
     t_usages :=
       [(m 10, m 14); (m 16, m 14); (v 11, v 11); (m 14, m 14);
        (v 60, v 60)];
     t_sources :=
       [(v 41, v 60); (m 7, v 60); (m 14, m 10); (m 16, m 10);
        (m 18, m 10); (m 10, m 10); (m 13, m 13); (m 11, m 11);
        (m 12, m 12); (v 11, v 11); (v 60, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 11, Field_block 0 Value); (v 60, known);
        (m 10, Field_code_id_of_call_witness);
        (m 10, Field_return_of_call Exn); (m 10, result 0)] |}.

(* The rebuilt unit: the code keeps its first parameter and its first
   result, the call gives [x12] and returns to [k20] with [x30] only, and
   [x61], which has no use, is removed. *)
Definition call_in_region_rebuilt : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4
          {| cm_params_arity :=
               [Unboxed_product [Singleton Value]; Unboxed_product []];
             cm_params_subkinds := [];
             cm_result_arity := [Unboxed_product [Singleton Value]];
             cm_result_subkinds := [];
             cm_is_tupled := false;
             cm_is_stub := false |}
          (return_to 3 [Var 40]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 10 Region (region Begin_region [])
          (Let 11 Value (make_block [Const (Tagged_immediate 7)])
             (Let 11 Value (region Local_allocation [Var 11; Var 10])
                (Let 12 Value (load_immediate 0 (Var 11))
                   (Let_cont_non_recursive 20
                      (Handler [30] (make_handler_metadata [Value])
                         (Let 14 Value (region End_region [Var 10])
                            (return_to 1 [Var 30])))
                      (Apply
                         {| ap_callee := Var 60; ap_call_kind := Direct 50;
                            ap_args := [Var 12]; ap_continuation := 20;
                            ap_exn_continuation := 2;
                            ap_exn_extra_args := [];
                            ap_args_arity :=
                              [Unboxed_product [Singleton Value];
                               Unboxed_product []];
                            ap_return_arity :=
                              [Unboxed_product [Singleton Value]] |}))))))).

(* The certificate of the check of regions of the rebuilt unit: [k20] is
   entered with [r10] open and its parameter living for ever, and the code
   50 is given no local value, has no local closure and returns no local
   value. *)
Definition call_in_region_cert : certificate :=
  Certificate [(20, Handler_entry [10] [Forever])]
    [(50, Code_entry [false] false false)] false false false.

End RegionsCcExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsCcExamples.

(* The statement applies to [call_in_region]: it meets every premise, the
   code 50 changes its calling convention (it keeps its first parameter
   and its first result), the rebuilt unit passes the check of regions,
   and both exit with 7 on the machine of regions. *)
Definition regions_cc_example_statement : Prop :=
  regions_cc_plain call_in_region = true
  ∧ least_solution (fun _ => false)
      (traverse_unit Auto interface call_in_region) call_in_region_tables
  ∧ in_flow_fragment (erase [] call_in_region) = true
  ∧ any_kind_arguments (erase [] call_in_region) = true
  ∧ results_kinded [] [] (erase [] call_in_region) = true
  ∧ well_formed_unit interface (erase [] call_in_region) = true
  ∧ exits_one_value interface (erase [] call_in_region) = true
  ∧ region_names_fresh base call_in_region empty_static_env = true
  ∧ erasable call_in_region empty_static_env = true
  ∧ derive_cannot_change (traverse_unit Auto interface call_in_region)
      call_in_region_tables
    = []
  ∧ unit_calling_convention_changes call_in_region_tables []
      (unit_code_deps Auto interface call_in_region) call_in_region
    = [(50%positive,
        Changing_calling_convention [Keep Value; Delete]
          [Keep Value; Delete; Delete])]
  ∧ rebuild_unit_cc_from_tables (traverse_unit Auto interface call_in_region)
      call_in_region_tables
      (derive_cannot_change (traverse_unit Auto interface call_in_region)
         call_in_region_tables)
      (unit_code_deps Auto interface call_in_region) true call_in_region
    = call_in_region_rebuilt
  ∧ region_check call_in_region_cert empty_static_env call_in_region_rebuilt
    = true
  ∧ (∃ rs : region_state,
       region_run 30
         (initial_region_state call_in_region empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ (∃ rs : region_state,
       region_run 30
         (initial_region_state call_in_region_rebuilt empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ regions_refine_observably empty_static_env base call_in_region
      call_in_region_rebuilt.
