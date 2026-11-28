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

(* The value slots of the closures of the traversal, and the refinement
   across changes of representation with a fragment that no longer asks
   that the slot of a field of a representation be one of the set.

   This file is to be reviewed: [repr_field_frag3_ok],
   [changed_set_frag3_ok], [repr_unboxing_frag3_nr] and the statements.
   The proofs are in FlambdaReaperClosuresTablesSlotsProofs.v, which needs
   no review.

   - [closure_value_slots_statement]: in the graph of the traversal (any
     mode) of a unit of the flow fragment that binds each variable once,
     the variable of a [Let x _ (Closure_of_set slot decls vs)] holds at a
     value slot [w] only what [vs] has there: [vs] has [w], at a simple
     whose node is what is held. Only the sets of closures make
     constructors at value slots
     (FlambdaReaperClosuresExactProofs.ns_ok), and the set of a [Let] of
     a closure makes them, at its own allocations, from its value slots;
     the allocations of another [Let] are its variable, those of the
     projections of its closures and made nodes, none of which is [x];
     those of a static set are symbols.
   - [changed_field_in_set_statement]: with the premises of
     FlambdaReaperClosuresTablesFieldsSpec.changed_field_used_statement,
     the value slot of each field of the representation of [x] is one of
     [vs]: the allocation of [tcr] whose set gives the representation
     holds the slot (FlambdaReaperClosuresTablesFieldsProofs.field_origin),
     and holds the value slots that [x] holds
     (FlambdaReaperClosuresExactProofs.closure_node_exact).
   - [repr_unboxing_frag3_sound_statement]: for such a unit and tables,
     [repr_unboxing_frag3_nr] implies
     FlambdaReaperClosuresTablesFieldsSpec.repr_unboxing_frag2_nr.
   - [rebuild_repr_frag3_refines_cc_statement],
     [rebuild_repr_frag3_observably_statement] and
     [rebuild_repr_frag3_local_statement]: the [rebuild_repr_frag2_]
     statements of FlambdaReaperClosuresTablesFieldsSpec.v with
     [repr_unboxing_frag3_nr].

   That the simple of a field not unboxed is not unboxed ([plain_simple]
   in [plain_c]) is derived in FlambdaReaperClosuresTablesPlainSpec.v,
   with an invariant of the traversal on the variables of the
   projections of value slots (FlambdaReaperTraversalSlotsSpec.v); the
   fragment there, [changed_set_frag4_ok], no longer asks it. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaUnusedStaticsSpec
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLocalFields FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperClosuresTablesSpec FlambdaReaperClosuresTablesFieldsSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment_frag3.

Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).

(* [repr_field_frag_ok], which asks nothing of a slot that [vs] does not
   have: there is none. *)
Definition repr_field_frag3_ok (vs : list (value_slot * simple))
    (fu : field * unboxed_field) : bool :=
  match fu with
  | (Field_value_slot w, u) =>
    match lookup w vs with
    | Some _ => repr_field_frag_ok unboxed changed vs fu
    | None => true
    end
  | _ => true
  end.

(* The conjuncts of [changed_set_frag2_ok] that are not derived. *)
Definition changed_set_frag3_ok (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) (r : closure_repr) : bool :=
  match d with
  | Closure_of_set _ _ vs =>
    kind_eqb kind Value
    && match projections_of_set x d body with [] => true | _ :: _ => false end
    && forallb (repr_field_frag3_ok vs) (cr_fields r)
  | _ => false
  end.

End Fragment_frag3.

(* [repr_unboxing_frag2_nr], with [changed_set_frag3_ok] for
   [changed_set_frag2_ok]. *)
Definition repr_unboxing_frag3_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  let changed :=
    unit_changed_representations is_local F T dominated to_unbox tcr e
  in
  repr_expr_with F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    changed (bound_code_my_closures e) nr
    (changed_set_frag3_ok unboxed changed) [] [] e
  && check_unboxed_uses F T unboxed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition closure_value_slots_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (w : value_slot) (b : node),
    in_flow_fragment e = true -> variables_bound_once e = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    g_constructor (graph_of_facts (traverse_unit flag u e)) (Node_variable x)
      (Field_value_slot w) b ->
    ∃ a : simple, lookup w vs = Some a ∧ b = node_of_simple a.

Definition changed_field_in_set_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (r : closure_repr) (w : value_slot)
    (uf : unboxed_field),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_changed_representations is_local (traverse_unit Auto u e) T
         dominated to_unbox tcr e)
    = Some r ->
    (Field_value_slot w, uf) ∈ cr_fields r ->
    lookup w vs <> None.

Definition repr_unboxing_frag3_sound_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag3_nr nr is_local (traverse_unit Auto u e) T
      cannot_change code_deps dominated to_unbox tcr e
    = true ->
    repr_unboxing_frag2_nr nr is_local (traverse_unit Auto u e) T
      cannot_change code_deps dominated to_unbox tcr e
    = true.

Definition rebuild_repr_frag3_refines_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag3_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_frag3_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_field_used_exact is_local (traverse_unit Auto u e) T = true ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag3_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_frag3_local_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_local_reads_used is_local (traverse_unit Auto u e) T = true ->
    check_field_used_exact is_local (traverse_unit Auto u e) T = true ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true ->
    analysis_tables is_local (traverse_unit Auto u e) T cannot_change
      max_unbox_size lambda_lifting dominated to_unbox tcr
    = true ->
    repr_unboxing_frag3_nr nr is_local (traverse_unit Auto u e) T
      cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).
