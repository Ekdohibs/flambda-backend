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

(* The rebuild of the reaper with local unboxing on units with regions,
   through the erasure of the core, as FlambdaReaperRegionsSpec.v has it
   of the plain rebuild.

   This file is to be reviewed: the fragment ([regions_local],
   [regions_not_unboxed]), the example and the statements.
   FlambdaReaperRegionsUnboxProofs.v proves them, with the lemmas of
   FlambdaReaperRegionsUnboxRebuildProofs.v (the commutation),
   FlambdaReaperRegionsUnboxBaseProofs.v (the local refinement for any
   first fresh name and a solution apart from the tables),
   FlambdaReaperRegionsUnboxTraverseProofs.v (what the graph of the unit
   has) and FlambdaReaperRegionsUnboxKeepsProofs.v (what the rebuild
   keeps), which need no review.

   WHAT THE REBUILD DOES WITH REGIONS

   (FlambdaReaperUnboxing.v, [rebuild_expr_u].) What the plain rebuild
   does (FlambdaReaperRegionsSpec.v), case for case, but for a variable
   that is unboxed: the [Let] of a block or a boxed number that the rules
   unbox binds the variables of its fields to what it holds there, and
   the [Local_allocation] after it is dropped with it
   ([rebuild_unboxed_binding], "The local allocation of [x] was unboxed by
   the [Let] before"). The variable of a region is unboxed nowhere where
   no allocation to unbox dominates it; were one to, the rebuild would
   make the [Let] of the begin [Invalid]. A field of a block that is read
   is of the kind of the read, never [Region], so the rebuild makes no
   parameter and no poison of that kind.

   THE FRAGMENT ([regions_local], [regions_not_unboxed])

   That of the plain rebuild ([regions_plain]), and no closure,
   projection, call or code (FlambdaReaperRebuildSpec.closure_free): the
   terms of the local unboxing (FlambdaReaperUnboxingRefinesSpec.v,
   [local_expr]). For the commutation with any tables, no variable of a
   region (one that a begin binds, [region_binders]) is dominated by an
   allocation to unbox; in tables closed under the rules on the graph of
   the unit with its regions, no allocation dominates such a variable, as
   the begin gives it any source ([regions_not_unboxed_closed_statement]).

   THE COMMUTATION ([rebuild_unbox_erase_commutes_statement])

   On the fragment, for any tables: erasing the unit rebuilt with
   unboxing is rebuilding the erased unit, with the facts of the erased
   unit, the same tables, and the variables of the fields named after
   every name of the unit with its regions. The erased unit has fewer
   names (the erasure drops the variables of the regions), so its own
   rebuild names those variables from a first name that may be smaller;
   the two are the same where the last name of the unit is not the
   variable of a region. The unboxed fields are read off the accessors
   and the constructors of the graph, which the erasure keeps
   ([unboxed_fields_erase_statement]), and so are the allocations to
   unbox that the rules give ([to_unbox_erase_statement]); the table of
   [dominated_by_allocation_point] is read off the solution alone. The
   fragment of the local unboxing reads of the unboxed fields their
   shapes only, not the names of their variables
   ([local_unboxing_names_statement]).

   THE REFINEMENT ([least_rebuild_unbox_regions_refines_statement])

   For a least solution of the graph of the unit with its regions, which
   is the analysis that the compiler runs, the tables of the rules
   ([derive_dominated], [least_to_unbox]), and the premises of
   FlambdaReaperLeastSpec.least_rebuild_unbox_refines_observably_statement
   on the erased unit, with the check of the reads on the graph with
   regions: if the rebuilt unit passes the check of regions, with any
   certificate, it refines the unit on the machine of regions
   (FlambdaReaperRegionsSpec.regions_refine_observably). The rebuild
   keeps [erasable] and the freshness of the symbols on the fragment
   ([rebuild_unbox_keeps_erasable_statement],
   [rebuild_unbox_keeps_fresh_statement]), so nothing else is asked of
   the rebuilt unit.

   The proof is that of FlambdaReaperRegionsSpec.v: the erased unit is
   refined by its plain rebuild with the least solution of the unit with
   the tuples that the erasure adds (the constants have every usage), and
   that by its rebuild with unboxing with that solution and the unboxed
   fields of the tables, whose variables are named after every name of
   the unit with its regions (the proof of
   FlambdaReaperUnboxingRefinesProofs.rebuild_unbox_with_refines_rebuild
   with the first fresh name and the solution left free; the solution
   agrees with the tables on the variables, the sources and the usages).
   By the commutation that rebuild is the erasure of the rebuilt unit.
   The check of the reads (FlambdaReaperLeastSpec.check_reads_read_only)
   is asked of the graph with regions, as the compiler has it: it reads
   facts that the erasure changes (the aliases: the erased graph has one
   of the variable of each end to the constants), where what the
   unboxing reads, the accessors and the constructors, are those of the
   erased graph.

   THE EXAMPLE ([regions_unbox_example_statement])

   A region with a local block of two fields, the first of which is read:
   the rules unbox the block, the rebuilt unit binds the variable of the
   field to what the block holds there, reads it in place of the load,
   drops the block and its [Local_allocation], passes the check of
   regions, and both exit with 7 on the machine of regions.

   WHAT IS NOT STATED

   - The unboxing across calls (FlambdaReaperUnboxingRefinesCallsSpec.v,
     the rebuild in the mode of the fix): its rebuild has calls, code and
     wrappers, whose commutation with the erasure is not proved here, and
     its statements ask the tables to be closed under the rules on the
     graph of the erased unit, which the least solution of the graph with
     regions is not in a unit that ends a region (it has not the alias of
     the variable of the end to the constants, which the erased graph
     has; [regions_unbox_example_statement] says it of [local_pair]).
   - That the rebuilt unit passes the check of regions wherever the unit
     does: false for the plain rebuild already
     (FlambdaReaperRegionsSpec.region_check_not_kept_statement), and the
     rebuild with unboxing makes a local boxed number that it does not
     unbox, and whose contents have no use, poison as the plain rebuild
     does.
   - Units with a parameter of the kind [Region], or a local set of
     several closures.
   - Anything of a run that goes on for ever, or of the heap at the
     exit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperLeastSpec FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* The variables of the regions that the unit begins, in its handlers
   and in the code that it binds too. *)
Fixpoint region_binders (e : expr) : list variable :=
  match e with
  | Let x _ (Prim (Region_primitive (Begin_region | Begin_try_region)) _)
      body =>
    x :: region_binders body
  | Let _ _ _ body => region_binders body
  | Let_static bindings body =>
    flat_map binding_region_binders bindings ++ region_binders body
  | Let_cont_non_recursive _ h body =>
    handler_region_binders h ++ region_binders body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_region_binders h) handlers
    ++ region_binders body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_region_binders (h : continuation_handler) : list variable :=
  match h with
  | Handler _ _ handler => region_binders handler
  end
with binding_region_binders (binding : static_binding) : list variable :=
  match binding with
  | Bind_code _ c => code_region_binders c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_region_binders (c : code) : list variable :=
  match c with
  | Code _ _ _ _ _ body => region_binders body
  end.

(* The fragment of the local unboxing: that of the plain rebuild, and no
   closure, projection, call or code
   (FlambdaReaperRebuildSpec.closure_free). *)
Definition regions_local (e : expr) : bool :=
  regions_plain [] e && closure_free e.

(* No variable of a region is dominated by an allocation to unbox. *)
Definition regions_not_unboxed (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  forallb
    (fun r => negb (has_to_be_unboxed dominated to_unbox (Node_variable r)))
    (region_binders e).

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The unboxed fields of the names are read off the accessors and the
   constructors of the graph, which the erasure keeps: those of the unit
   and of the erased unit are the same, from any first name of their
   variables. *)
Definition unboxed_fields_erase_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (dominated : list (node * node)) (to_unbox : list node)
    (next : positive),
    regions_plain [] e = true ->
    unboxed_fields_of_names (traverse_unit flag u (erase [] e)) T
      (bound_code_my_closures (erase [] e)) dominated to_unbox next
    = unboxed_fields_of_names (traverse_unit flag u e) T
        (bound_code_my_closures e) dominated to_unbox next.

(* So are the allocations to unbox that the rules give, for any table of
   the code that cannot change, and for the least one. (The table of
   [dominated_by_allocation_point], FlambdaReaperUnboxing.derive_dominated,
   is read off the solution alone.) *)
Definition to_unbox_erase_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (is_local : field -> bool) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool),
    regions_plain [] e = true ->
    derive_to_unbox is_local (traverse_unit flag u (erase [] e)) T
      cannot_change max_unbox_size lambda_lifting
    = derive_to_unbox is_local (traverse_unit flag u e) T cannot_change
        max_unbox_size lambda_lifting
    ∧ least_to_unbox is_local (traverse_unit flag u (erase [] e)) T
        max_unbox_size lambda_lifting
      = least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
          lambda_lifting.

(* The fragment of the local unboxing does not depend on the names of
   the variables of the fields: [local_unboxing], which names them after
   every name of the unit, holds where they are named from any first
   name. *)
Definition local_unboxing_names_statement : Prop :=
  ∀ (F : facts) (T : tables) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) (next : positive),
    local_unboxing F T dominated to_unbox e
    = local_expr
        (fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                dominated to_unbox next))
        (solution_of_tables T) [] [] e.

(* In tables closed under the rules on the graph of the unit with its
   regions, no allocation dominates the variable of a region: the begin
   of a region gives it any source, hence several allocations. *)
Definition regions_not_unboxed_closed_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (to_unbox : list node),
    regions_local e = true ->
    in_flow_fragment (erase [] e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    regions_not_unboxed (derive_dominated T) to_unbox e = true.

(* Erasing the unit rebuilt with unboxing (in the mode of the branch, with
   no change of representation) is rebuilding the erased unit, with the
   facts of the erased unit, the same tables, and the variables of the
   fields named after every name of the unit with its regions. *)
Definition rebuild_unbox_erase_commutes_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    regions_local e = true ->
    regions_not_unboxed dominated to_unbox e = true ->
    erase []
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e)
    = let '(unboxed, next) :=
        unboxed_fields_of_names (traverse_unit flag u (erase [] e)) T
          (bound_code_my_closures (erase [] e)) dominated to_unbox
          (fresh_name (names_of e))
      in
      rebuild_expr_u false (traverse_unit flag u (erase [] e))
        (solution_of_tables T) (fun n => mem_node n cannot_change) code_deps
        preserve_direct_calls (static_names code_ids_bound (erase [] e))
        (bound_code_metadata (erase [] e)) unboxed next [] None
        empty_unbox_env (erase [] e).

(* On the fragment, the unit rebuilt with unboxing, from the graph of the
   unit and any tables, is erasable wherever the unit is, and has no
   symbol at or above the base wherever the unit has none. *)
Definition rebuild_unbox_keeps_erasable_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    regions_local e = true ->
    erasable e empty_static_env = true ->
    erasable
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true.

Definition rebuild_unbox_keeps_fresh_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool) (base : symbol),
    regions_local e = true ->
    region_names_fresh base e empty_static_env = true ->
    region_names_fresh base
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e)
      empty_static_env
    = true.

(* The rebuild with unboxing, for the least solution of the graph of the
   unit with its regions and the tables that the rules give from it,
   refines the unit on the machine of regions if the rebuilt unit passes
   the check of regions. The premises on the erased unit are those of
   FlambdaReaperLeastSpec.least_rebuild_unbox_refines_observably_statement
   (the check of the reads is of the graph with regions, whose reads are
   those of the erased graph); that of no local field or no projection is
   left out, as the fragment of the local unboxing has no projection. *)
Definition least_rebuild_unbox_regions_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (max_unbox_size : nat)
    (lambda_lifting preserve_direct_calls : bool) (cert : certificate)
    (base : symbol),
    regions_plain [] e = true ->
    is_local_slots_only is_local ->
    in_flow_fragment (erase [] e) = true ->
    static_constants (unit_codes (erase [] e)) (erase [] e) = true ->
    results_kinded [] [] (erase [] e) = true ->
    well_formed_unit u (erase [] e) = true ->
    exits_one_value u (erase [] e) = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    local_unboxing (traverse_unit flag u (erase [] e)) T (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
      (erase [] e)
    = true ->
    check_reads_read_only is_local (traverse_unit flag u e)
      (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
    = true ->
    region_names_fresh base e empty_static_env = true ->
    erasable e empty_static_env = true ->
    region_check cert empty_static_env
      (rebuild_unit_unbox (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) (derive_dominated T)
         (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
            lambda_lifting)
         preserve_direct_calls e)
    = true ->
    regions_refine_observably empty_static_env base e
      (rebuild_unit_unbox (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) (derive_dominated T)
         (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
            lambda_lifting)
         preserve_direct_calls e).

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Module RegionsUnboxExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples.

Local Open Scope positive_scope.

(* A region with a local block of two fields, the first of which is
   read:

     let r10 = begin_region in
     let x11 = local [7; 8] in
     let x12 = x11.(0) in
     let x13 = end_region r10 in
     return x12

   The rules unbox [x11]: its field 0 is the variable 14, after every
   name of the unit, bound to 7; the read is that variable; the block and
   its [Local_allocation] are gone. *)
Definition local_pair : expr :=
  Let 10 Region (region Begin_region [])
    (Let 11 Value
       (make_block [Const (Tagged_immediate 7); Const (Tagged_immediate 8)])
       (Let 11 Value (region Local_allocation [Var 11; Var 10])
          (Let 12 Value (load_immediate 0 (Var 11))
             (Let 13 Value (region End_region [Var 10])
                (return_to 1 [Var 12]))))).

(* The least solution of its graph. *)
Definition local_pair_tables : tables :=
  {| t_alias :=
       [(Node_variable 12, Node_all_constants);
        (Node_made 1, Node_variable 12)];
     t_any_usage :=
       [Node_all_constants; Node_variable 10; Node_variable 12; Node_made 2;
        Node_variable 13; Node_made 1];
     t_any_source :=
       [Node_made 1; Node_variable 12; Node_all_constants; Node_variable 10;
        Node_variable 13];
     t_has_usage :=
       [Node_variable 11; Node_variable 10; Node_all_constants;
        Node_variable 12; Node_made 1; Node_variable 13; Node_made 2];
     t_has_source :=
       [Node_made 1; Node_variable 11; Node_variable 12; Node_variable 13;
        Node_variable 10; Node_all_constants];
     t_usages := [(Node_variable 11, Node_variable 11)];
     t_sources := [(Node_variable 11, Node_variable 11)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(Node_variable 11, Field_block 0 Value)] |}.

(* The allocations to unbox, as the rules give them with no local field,
   at most 10 fields and no lambda-lifting. *)
Definition local_pair_to_unbox : list node :=
  least_to_unbox (fun _ => false) (traverse_unit Auto interface local_pair)
    local_pair_tables 10 false.

Definition local_pair_rebuilt : expr :=
  Let 10 Region (region Begin_region [])
    (Let 14 Value (Simple (Const (Tagged_immediate 7)))
       (Let 12 Value (Simple (Var 14))
          (Let 13 Value (region End_region [Var 10])
             (return_to 1 [Var 12])))).

End RegionsUnboxExamples.

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsUnboxExamples.

(* The statement applies to [local_pair]: it meets every premise, the
   rules unbox the local block, its rebuild binds the variable of the
   field read and drops the block and its [Local_allocation], passes the
   check of regions, and both exit with 7 on the machine of regions. The
   least solution is not closed under the rules on the graph of the
   erased unit. *)
Definition regions_unbox_example_statement : Prop :=
  regions_local local_pair = true
  ∧ least_solution (fun _ => false) (traverse_unit Auto interface local_pair)
      local_pair_tables
  ∧ check_closed (fun _ => false)
      (traverse_unit Auto interface (erase [] local_pair)) local_pair_tables
    = false
  ∧ derive_dominated local_pair_tables
    = [(Node_variable 11%positive, Node_variable 11%positive)]
  ∧ local_pair_to_unbox = [Node_variable 11%positive]
  ∧ in_flow_fragment (erase [] local_pair) = true
  ∧ static_constants (unit_codes (erase [] local_pair)) (erase [] local_pair)
    = true
  ∧ results_kinded [] [] (erase [] local_pair) = true
  ∧ well_formed_unit interface (erase [] local_pair) = true
  ∧ exits_one_value interface (erase [] local_pair) = true
  ∧ local_unboxing (traverse_unit Auto interface (erase [] local_pair))
      local_pair_tables (derive_dominated local_pair_tables)
      local_pair_to_unbox (erase [] local_pair)
    = true
  ∧ check_reads_read_only (fun _ => false)
      (traverse_unit Auto interface local_pair)
      (derive_dominated local_pair_tables) local_pair_to_unbox
    = true
  ∧ region_names_fresh base local_pair empty_static_env = true
  ∧ erasable local_pair empty_static_env = true
  ∧ rebuild_unit_unbox (traverse_unit Auto interface local_pair)
      local_pair_tables
      (derive_cannot_change (traverse_unit Auto interface local_pair)
         local_pair_tables)
      (unit_code_deps Auto interface local_pair)
      (derive_dominated local_pair_tables) local_pair_to_unbox true local_pair
    = local_pair_rebuilt
  ∧ region_check cert empty_static_env local_pair_rebuilt = true
  ∧ (∃ rs : region_state,
       region_run 10 (initial_region_state local_pair empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ (∃ rs : region_state,
       region_run 10
         (initial_region_state local_pair_rebuilt empty_static_env base)
       = Region_exit 1%positive [Value_const (Tagged_immediate 7)] rs)
  ∧ regions_refine_observably empty_static_env base local_pair
      local_pair_rebuilt.
