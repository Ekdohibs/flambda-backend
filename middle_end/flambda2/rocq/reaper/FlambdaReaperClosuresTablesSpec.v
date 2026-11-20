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

(* The checks of the tables in the fragment of the refinement across
   changes of representation (FlambdaReaperUnboxingRefinesClosuresSpec.v,
   [changed_set_ok]) that follow from the construction of the
   representations (FlambdaReaperUnboxing.changed_representations), and
   the refinement with a fragment that asks only the others.

   This file is to be reviewed: [changed_set_frag_ok], [repr_expr_frag]
   and [repr_unboxing_frag_nr] (a copy of [repr_expr] and
   [repr_unboxing_nr] with [changed_set_frag_ok] in place of
   [changed_set_ok]), and the statements. The proofs are in
   FlambdaReaperClosuresTablesProofs.v, which needs no review.

   Of a [Let x kind (Closure_of_set slot decls vs) body] to which the
   changes of representation give [r], [changed_set_ok] asks:
   - that [kind] is [Value], that no other closure of the set is
     projected, and that each field of [r] passes [repr_field_ok];
   - that [slot] is [cr_current r];
   - that the renamed declarations find at [current_slot r] the code
     that the declarations find at [slot];
   - that the leaves of [cr_fields r], the new value slots, are
     distinct.
   [changed_set_frag_ok] asks those of the first item, the checks of the
   term and [repr_field_ok]; the other three are derived:
   - [changed_leaves_distinct_statement]: of every representation that
     [unit_changed_representations] makes, for any tables:
     [mk_unboxed_fields] names its variables from [next] on, each once;
   - [changed_decls_renamed_statement]: of every such representation
     and every [Let] of a closure of the unit, for any tables, the
     renamed declarations find at the renamed slot what the declarations
     find at the slot: the new function slots of a representation are
     distinct and after every slot of the unit;
   - [changed_slot_current_statement]: [slot] is [cr_current r], for the
     graph of the traversal (with [Auto]) of a unit of the flow fragment,
     well formed and binding each variable once, a least solution and a
     table of [dominated_by_allocation_point] that passes its check (as
     the derived one does:
     FlambdaReaperWrapperAutoProofs.derive_dominated_checked). The
     traversal makes the variable a closure of its set at [slot]
     (FlambdaReaperFlowProofs.set_edges); the invariant of the proof of
     FlambdaReaperTypesTraversalSpec.reprs_of_sets_derived_statement
     gives the representation of a closure at the function slot at which
     its set holds it, the closures of the set being each of one
     allocation in the graph of the traversal
     (FlambdaReaperTypesTraversalUnaliasedSpec.v).
   - [repr_unboxing_frag_sound_statement]: for such a unit and tables,
     [repr_unboxing_frag_nr] implies [repr_unboxing_nr].
   - [rebuild_repr_frag_refines_cc_statement],
     [rebuild_repr_frag_observably_statement] and
     [rebuild_repr_frag_local_statement]: the statements
     [rebuild_repr_fixed_refines_cc_never_returns_statement],
     [rebuild_repr_fixed_observably_never_returns_statement] and
     [rebuild_repr_fixed_local_never_returns_statement] for the
     traversal with [Auto], with [repr_unboxing_frag_nr] in place of
     [repr_unboxing_nr] and the premises of
     [changed_slot_current_statement].

   Left as checks: the conjuncts of [repr_field_ok], that each field of
   [r] is a value slot of the set, with a use in a closure of the set
   ([s_field_used]), holding a plain name with a use or, for a field
   unboxed in turn, a used name unboxed alike. The first two would need
   the constructors at the closures of the traversal and the relation of
   [get_fields_of_constructors] to [s_field_used]; the others relate the
   unboxing of a field to that of the name stored in it, which the
   statements, taking any [to_unbox], do not tie. *)

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
  FlambdaReaperLocalFields FlambdaReaperUnboxingRefinesClosuresSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment_frag.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.

(* The conjuncts of [changed_set_ok] that are not derived. *)
Definition changed_set_frag_ok (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) (r : closure_repr) : bool :=
  match d with
  | Closure_of_set _ _ vs =>
    kind_eqb kind Value
    && match projections_of_set x d body with [] => true | _ :: _ => false end
    && forallb (repr_field_ok F sol unboxed changed x vs) (cr_fields r)
  | _ => false
  end.

(* [repr_expr], with [changed_set_frag_ok] for [changed_set_ok]. *)
Fixpoint repr_expr_frag (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name changed (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields
          && no_changed changed d)
         || unboxed_slot_read_ok unboxed changed x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed changed d)
         || plain_set_ok unboxed changed x d body
         || plain_slot_read_ok unboxed changed d
       | None, Some r => changed_set_frag_ok x kind d body r
       | Some _, Some _ => false
       end
    && repr_expr_frag ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_frag bindings && repr_expr_frag kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_frag kinds C ds h
    && repr_expr_frag kinds ((k, (ds, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map
        (fun '(k, h) =>
           (k, (handler_decisions true sol unboxed false h, true)))
        handlers
      ++ C
    in
    forallb
      (fun '(_, h) =>
         repr_handler_frag kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_frag kinds C' body
  | Apply ap =>
    repr_call F sol cannot_change code_deps unit_code_ids codes unboxed changed
      my_closures nr C ap
  | Apply_cont ac => apply_cont_ok_c unboxed changed C ac
  | Switch scrutinee arms =>
    plain_c unboxed changed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok_c unboxed changed C ac) arms
  | Invalid => true
  end
with repr_handler_frag (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided ds
    && forallb (fun x => plain_c unboxed changed (Var x)) params
    && repr_expr_frag
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with repr_binding_frag (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_frag id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb (plain_c unboxed changed) fields
  | Bind_symbol _ _ => false
  end
with repr_code_frag (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match lookup id my_closures with
       | Some mc => Pos.eqb mc my_closure
       | None => false
       end
    && match
         code_change_u sol cannot_change code_deps unit_code_ids unboxed id
           metadata
       with
       | Changing_u pd rd =>
         forallb decided pd && forallb decided rd
         && Nat.eqb (length params) (length pd)
         && Nat.eqb (length pd) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
         && forallb (fun x => plain_c unboxed changed (Var x)) params
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && is_none (fields_of unboxed my_closure)
         && negb (Pos.eqb ret exn)
         && repr_expr_frag
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true)); (exn, ([], false))] body
       | Not_changing_u => false
       end
    && (negb (nr id)
        || negb (among ret (live_conts F sol code_deps codes body)))
  end.

End Fragment_frag.

(* [repr_unboxing_nr], with [repr_expr_frag] for [repr_expr]. *)
Definition repr_unboxing_frag_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr_frag F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr [] [] e
  && check_unboxed_uses F T unboxed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition changed_leaves_distinct_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    (n : node) (r : closure_repr),
    Keyed.lookup node_eqb n
      (unit_changed_representations is_local F T dominated to_unbox tcr e)
    = Some r ->
    distinct (map fst (fields_leaves (cr_fields r))) = true.

Definition changed_decls_renamed_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (n : node) (r : closure_repr),
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb n
      (unit_changed_representations is_local F T dominated to_unbox tcr e)
    = Some r ->
    lookup (rename_function_slot r slot) (rename_function_decls r decls)
    = lookup slot decls.

Definition changed_slot_current_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (r : closure_repr),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_dominated T dominated = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_changed_representations is_local (traverse_unit Auto u e) T
         dominated to_unbox tcr e)
    = Some r ->
    slot = cr_current r.

Definition repr_unboxing_frag_sound_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_dominated T dominated = true ->
    repr_unboxing_frag_nr nr is_local (traverse_unit Auto u e) T cannot_change
      code_deps dominated to_unbox tcr e
    = true ->
    repr_unboxing_nr nr is_local (traverse_unit Auto u e) T cannot_change
      code_deps dominated to_unbox tcr e
    = true.

Definition rebuild_repr_frag_refines_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_dominated T dominated = true ->
    repr_unboxing_frag_nr nr is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_frag_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
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
    check_dominated T dominated = true ->
    repr_unboxing_frag_nr nr is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_frag_local_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
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
    check_dominated T dominated = true ->
    repr_unboxing_frag_nr nr is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).
