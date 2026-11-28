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


(* The rebuild with unboxing refines its unit across the changes of
   representation of sets of closures, on a fragment wider than that of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v: beside its calls, a
   call of code that changes its calling convention may return to a
   continuation that does not take what the callee returns as it is,
   through a wrapper that each rebuild makes.

   This file is to be reviewed: the fragment ([repr_wraps_unboxing_nr],
   [repr_flow_wraps_nr]), the statements and the example, against
   FlambdaReaperUnboxing.v ([make_apply_wrapper_u]) and
   FlambdaReaperCallingConventions.v ([make_apply_wrapper_cc]). The proofs
   are in FlambdaReaperUnboxingRefinesClosuresWrapsProofs.v,
   FlambdaReaperUnboxingRefinesClosuresProofs.v and
   FlambdaReaperUnboxingRefinesClosuresSimProofs.v, which need no review.

   A call of code that changes its calling convention returns what the
   decisions of the results of the code keep. If its continuation does
   not keep the same (it deletes a result that the callee returns, or
   keeps one that the callee does not, which it is given as poison), each
   rebuild makes the call return to a wrapper of its own, which takes what
   the callee returns and gives the continuation what it keeps. The two
   wrappers have other names, and their parameters other names (the first
   rebuild names them from [fresh_name (names_of e)], the second after the
   variables of the unboxed fields); given the same results, they give the
   continuation the same values. Neither states anything of its
   parameters, with types or without: a call of code that changes its
   calling convention gives its wrapper no subkind
   ([make_apply_wrapper_u] with no [wrapper_subkinds]).

   [repr_wraps_holds_reach_statement]: the fragment holds that of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v
   ([repr_reach_unboxing_nr]), for the same tables, code taken to never
   return and flow.

   [rebuild_repr_wraps_refines_cc_statement],
   [rebuild_repr_wraps_observably_statement],
   [rebuild_repr_wraps_local_statement] and their [_with_types] forms: the
   statements of FlambdaReaperUnboxingRefinesClosuresReachSpec.v with
   [repr_flow_wraps_nr] in place of [repr_flow_unboxing_nr]; those are
   corollaries.

   The fragment ([repr_wraps_unboxing_nr]) is that of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v with one more kind of
   call ([wraps_call]): a direct call, as [repr_call_w] has it, of code
   that changes its calling convention and whose results do not all have
   no source ([code_never_returns] is false), to a continuation [k]
   ([wraps_cont_ok]) that may be applied (a handler of the term, or the
   return continuation of the code the call is in), each result that [k]
   keeps of the kind that the call returns it at, or to a way out of the
   unit, which keeps every result; the decisions of the results of the
   code keep them at those kinds too ([kinds_kept]). Whatever [k] keeps,
   the two rebuilds agree on whether the call returns to a wrapper: both
   do, or neither.

   What is left out, for the proof: wrappers of calls of code that keeps
   its calling convention (whose wrapper, with types, states what the
   continuation keeps of its parameters), and the rest of what
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v leaves out.

   [repr_wraps_example_statement]: a unit of [repr_flow_wraps_nr] that is
   not of [repr_flow_unboxing_nr] ([WrapsExample]: a call of code that
   changes its calling convention returning to a handler that drops its
   result), whose two rebuilds make the call return to wrappers of other
   parameters; its rebuild in the mode of the fix refines it up to
   observation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesClosuresReachSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeast FlambdaReaperLeastSpec FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaStaticsInScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
(* The [my_closure] of each code of the unit. *)
Variable my_closures : finite_map variable.
(* The code taken to never return. *)
Variable nr : code_id -> bool.
(* The code that the flow may bring to the callee of a call
   (FlambdaReaperUnboxingRefinesClosuresReachSpec.v). *)
Variable reach : bool -> variable -> code_id -> bool.

Local Abbreviation plain_c := (plain_c unboxed changed).
Local Abbreviation changed_name := (changed_name changed).

(* The decisions [ds] keep each value they keep at its kind in [ks]. *)
Definition kinds_kept (ds : list decision) (ks : list flambda_kind) : bool :=
  forallb
    (fun '(d, kind) =>
       match d with
       | Decided (Keep kind') => kind_eqb kind' kind
       | _ => true
       end)
    (combine ds ks).

(* The continuation [k] of a call that returns values of the kinds [ks]:
   one that may be applied, which keeps each result it keeps at the kind
   that the call returns it at, or a way out of the unit. *)
Definition wraps_cont_ok (C : finite_map (list decision * bool))
    (k : continuation) (ks : list flambda_kind) : bool :=
  match lookup k C with
  | Some (ds, true) => kinds_kept ds ks
  | Some (_, false) => false
  | None => true
  end.

(* A direct call of code that changes its calling convention and returns,
   as [repr_call_w] has it, to a continuation of [wraps_cont_ok], which
   may take what the callee returns through a wrapper of each rebuild. *)
Definition wraps_call (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap with
  | Var f, Some id =>
    kind_calls (ap_call_kind ap) id
    && match lookup id codes, lookup id my_closures with
       | Some metadata, Some mc =>
         negb (code_never_returns sol code_deps id)
         && repr_option_eqb (changed_name (Node_variable f))
              (changed_name (Node_variable mc))
         && is_none (fields_of unboxed f)
         && arity_eqb (ap_args_arity ap) (cm_params_arity metadata)
         && arity_eqb (ap_return_arity ap) (cm_result_arity metadata)
         && forallb plain_c (ap_args ap)
         && extra_args_ok unboxed changed C ap
         && match
              code_change_u sol cannot_change code_deps unit_code_ids unboxed
                id metadata
            with
            | Changing_u pd rd =>
              forallb decided pd && forallb decided rd
              && is_none
                   (my_closure_decision_u cannot_change code_deps
                      unit_code_ids unboxed id)
              && Nat.eqb (length rd)
                   (length (unarize (cm_result_arity metadata)))
              && kinds_kept rd (unarize (ap_return_arity ap))
              && wraps_cont_ok C (ap_continuation ap)
                   (unarize (ap_return_arity ap))
            | Not_changing_u => false
            end
       | _, _ => false
       end
  | _, _ => false
  end.

(* A call of [repr_call_r], or one through wrappers. *)
Definition repr_call_x (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  repr_call_r F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr reach C ap
  || wraps_call C ap.

(* [repr_expr_r], [repr_handler_r], [repr_binding_r] and [repr_code_r] of
   FlambdaReaperUnboxingRefinesClosuresReachSpec.v, with [repr_call_x]. *)
Fixpoint repr_expr_x (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields
          && no_changed changed d)
         || unboxed_slot_read_ok unboxed changed x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed changed d)
         || plain_set_ok unboxed changed x d body
         || plain_slot_read_ok unboxed changed d
       | None, Some r => changed_set_ok F sol unboxed changed x kind d body r
       | Some _, Some _ => false
       end
    && repr_expr_x ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_x bindings && repr_expr_x kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_x kinds C ds h
    && repr_expr_x kinds ((k, (normed ds, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map
        (fun '(k, h) =>
           (k, (normed (handler_decisions true sol unboxed false h), true)))
        handlers
      ++ C
    in
    forallb
      (fun '(_, h) =>
         repr_handler_x kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_x kinds C' body
  | Apply ap => repr_call_x C ap
  | Apply_cont ac => apply_cont_ok_w unboxed changed C ac
  | Switch scrutinee arms =>
    plain_c scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok_w unboxed changed C ac) arms
  | Invalid => true
  end
with repr_handler_x (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided (normed ds)
    && forallb (fun x => plain_c (Var x)) params
    && repr_expr_x (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         C handler
  end
with repr_binding_x (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_x id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_code_x (id : code_id) (c : code) {struct c} : bool :=
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
    && forallb (fun x => plain_c (Var x)) params
    && is_none (fields_of unboxed my_closure)
    && negb (Pos.eqb ret exn)
    && match
         code_change_u sol cannot_change code_deps unit_code_ids unboxed id
           metadata
       with
       | Changing_u pd rd =>
         forallb decided pd && forallb decided rd
         && Nat.eqb (length params) (length pd)
         && Nat.eqb (length pd) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && repr_expr_x
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true));
               (exn, (code_exn_ds sol code_deps unboxed id, false))]
              body
       | Not_changing_u =>
         negb (cm_is_tupled metadata)
         && match lookup id code_deps with
            | Some cd =>
              let rds :=
                normed
                  (fixed_arity_decisions sol unboxed (cd_return cd)
                     (unarize (cm_result_arity metadata)))
              in
              forallb decided rds
              && repr_expr_x
                   (rev ((my_closure, Value)
                         :: with_kinds params
                              (unarize (cm_params_arity metadata))))
                   [(ret, (rds, true));
                    (exn, (code_exn_ds sol code_deps unboxed id, false))]
                   body
            | None => false
            end
       end
    && (negb (nr id)
        || negb (among ret (live_conts F sol code_deps codes body)))
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing and the changes of
   representation that the tables give from its graph [F], the code [nr]
   taken to never return, and the code [reach] that the flow may bring to
   the callee of a call. *)
Definition repr_wraps_unboxing_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (reach : bool -> variable -> code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr_x F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr reach [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow being read off the facts [F1] and the tables [T1] of the first
   rebuilt unit (FlambdaReaperUnboxingRefinesReach2Spec.flow_reach2). *)
Definition repr_flow_wraps_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (F1 : facts) (T1 : tables) (e : expr) : bool :=
  repr_wraps_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr (flow_reach2 F1 T1) e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition repr_wraps_holds_reach_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (reach : bool -> variable -> code_id -> bool)
    (e : expr),
    repr_reach_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr reach e
    = true ->
    repr_wraps_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr reach e
    = true.

Definition rebuild_repr_wraps_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
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
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_wraps_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
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
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_wraps_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
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
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* ======================================================================== *)
(* With types                                                               *)
(* ======================================================================== *)

(* The rebuild with unboxing and changes of representation of the reaper
   with types in the mode of the fix refines the rebuild with types and
   changes of calling convention, with no set of closures given to the
   rewriting, on the wider fragment, with the premises of
   [rebuild_repr_reach_with_types_refines_cc_statement]
   (FlambdaReaperUnboxingRefinesClosuresReachSpec.v). *)
Definition rebuild_repr_wraps_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The unit rebuilt with unboxing and changes of representation by the
   reaper with types refines the unit up to observation, with the
   premises of [rebuild_repr_reach_with_types_local_statement] but the
   fragment. *)
Definition rebuild_repr_wraps_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
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
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The same without local fields. *)
Definition rebuild_repr_wraps_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
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
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_wraps_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).


(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

(* The unit of [TwoFieldsExample] (the code 54 reads the block of two
   fields in the slot 80 of its closure [x60], whose set changes its
   representation, and adds them and its parameter), which calls 54 twice:
   with 32, returning to the handler 20 of the result [x30]; and from 20,
   with [x30], returning to the handler 21, whose parameter [x31] has no
   use. 21 returns [x30], 7 + 8 + 32. The result of 54 is used (at 20), so
   the code returns it, and 21 does not take it: each rebuild makes the
   second call return to a wrapper that drops it. *)
Module WrapsExample.

Local Open Scope positive_scope.

Definition jump (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

(* A call of 54 through [x60], returning to [k], at the arity [arity]. *)
Definition call (arity : arity) (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 54; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := arity; ap_return_arity := arity |}.

(* The handlers of the results of the two calls, the second call returning
   to [k] ([call2]). *)
Definition handlers (arity : arity) (call2 : continuation -> expr)
    (body : expr) : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value])
       (Let_cont_non_recursive 21 (Handler [] (make_handler_metadata [])
                                     (return_to 1 [Var 30]))
          (call2 21)))
    body.

Definition unit_expr : expr :=
  Let_static [Bind_code 54 TwoFieldsExample.code_54]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let_cont_non_recursive 21
                   (Handler [31] (make_handler_metadata [Value])
                      (return_to 1 [Var 30]))
                   (Apply (call [Singleton Value] [Var 30] 21))))
             (Apply (call [Singleton Value] [int 32] 20))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

(* The table of the rules for [cannot_change_calling_convention]. *)
Definition cannot_change : list node := derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The first rebuilt unit, and its analysis. *)
Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match FlambdaReaperLeast.compute_least no_local facts1 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

(* The metadata of 54 rebuilt, which changes its calling convention. *)
Definition changed_metadata : code_metadata :=
  {| cm_params_arity := [Unboxed_product [Singleton Value]];
     cm_params_subkinds := [];
     cm_result_arity := [Unboxed_product [Singleton Value]];
     cm_result_subkinds := [];
     cm_is_tupled := false;
     cm_is_stub := false |}.

(* The wrapper [w] of the second call, of parameter [p], which drops it. *)
Definition wrapper (w p : variable) (k : continuation) : expr :=
  Let_cont_non_recursive w (Handler [p] (make_handler_metadata [Value])
                              (jump k []))
    (Apply (call [Unboxed_product [Singleton Value]] [Var 30] w)).

(* The first rebuild: 54 changes its calling convention, the handler 21
   drops its parameter, and the second call returns to the wrapper 61 of
   parameter 61. *)
Definition first_rebuilt : expr :=
  Let_static
    [Bind_code 54
       (Code [40] 41 3 4 changed_metadata
          (code_body TwoFieldsExample.code_54))]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
          (handlers [Unboxed_product [Singleton Value]] (wrapper 61 61)
             (Apply (call [Unboxed_product [Singleton Value]] [int 32]
                       20))))).

(* The rebuild with unboxing and changes of representation: 54 reads the
   new slots 81 and 82 of its closure into the variables 61 and 62 of the
   fields of [x45], the block [x10] is the variables 63 and 64, and the
   second call returns to the wrapper 65 of parameter 65. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_code 54
       (Code [40] 41 3 4 changed_metadata
          (Let 62 Value (Project_value_slot_from 83 82 (Var 41))
             (Let 61 Value (Project_value_slot_from 83 81 (Var 41))
                (Let 46 Value (Simple (Var 61))
                   (Let 47 Value (Simple (Var 62))
                      (Let 48 Value (TwoFieldsExample.add (Var 46) (Var 47))
                         (Let 49 Value (TwoFieldsExample.add (Var 48) (Var 40))
                            (return_to 3 [Var 49]))))))))]
    (Let 64 Value (Simple (int 8))
       (Let 63 Value (Simple (int 7))
          (Let 60 Value
             (Closure_of_set 83 [(83, 54)] [(81, Var 63); (82, Var 64)])
             (handlers [Unboxed_product [Singleton Value]] (wrapper 65 65)
                (Apply (call [Unboxed_product [Singleton Value]] [int 32]
                          20)))))).

(* What the unit and its rebuilds return: 7 + 8 + 32. *)
Definition result : list value := [Value_const (Tagged_immediate 47)].

End WrapsExample.

(* With the tables of the least solution, of the unit and of the first
   rebuilt unit: the block [x10] is unboxed and the set of [x60] changes
   its representation; 54 may change its calling convention. The unit is
   not of the fragment of FlambdaReaperUnboxingRefinesClosuresReachSpec.v
   (the second call returns to a continuation that does not take its
   result), and is of [repr_flow_wraps_nr]; the first rebuild makes the
   second call return to the wrapper 61, and the rebuild in the mode of
   the fix to the wrapper 65, whose parameter is not the parameter 61 of
   the first, 61 being there a variable of a field of [x45]; the unit and
   both rebuilds return [result]; and the rebuilt unit refines the unit up
   to observation. *)
Definition repr_wraps_example_statement : Prop :=
  check_dominated WrapsExample.tables WrapsExample.dominated = true
  ∧ check_to_unbox no_local WrapsExample.facts WrapsExample.tables []
      max_unbox_size false WrapsExample.to_unbox
    = true
  ∧ check_to_change_representation no_local WrapsExample.facts
      WrapsExample.tables [] max_unbox_size false WrapsExample.to_unbox
      WrapsExample.tcr
    = true
  ∧ check_sets_whole WrapsExample.facts WrapsExample.tables
      WrapsExample.to_unbox WrapsExample.tcr
    = true
  ∧ WrapsExample.to_unbox = [v 10]
  ∧ mem_node (v 60) WrapsExample.tcr = true
  ∧ mem_node (c 54) WrapsExample.cannot_change = false
  ∧ repr_flow_unboxing_nr (fun _ => false) no_local WrapsExample.facts
      WrapsExample.tables WrapsExample.cannot_change WrapsExample.code_deps
      WrapsExample.dominated WrapsExample.to_unbox WrapsExample.tcr
      WrapsExample.facts1 WrapsExample.tables1 WrapsExample.unit_expr
    = false
  ∧ repr_flow_wraps_nr (fun _ => false) no_local WrapsExample.facts
      WrapsExample.tables WrapsExample.cannot_change WrapsExample.code_deps
      WrapsExample.dominated WrapsExample.to_unbox WrapsExample.tcr
      WrapsExample.facts1 WrapsExample.tables1 WrapsExample.unit_expr
    = true
  ∧ check_unboxed_reads WrapsExample.facts WrapsExample.tables
      WrapsExample.dominated WrapsExample.to_unbox
    = true
  ∧ WrapsExample.first = WrapsExample.first_rebuilt
  ∧ rebuild_unit_unbox_repr_fixed no_local WrapsExample.facts
      WrapsExample.tables WrapsExample.cannot_change WrapsExample.code_deps
      WrapsExample.dominated WrapsExample.to_unbox WrapsExample.tcr true
      WrapsExample.unit_expr
    = WrapsExample.rebuilt
  ∧ run 60 (on_its_own WrapsExample.unit_expr)
    = Exit 1%positive WrapsExample.result empty_heap no_reads
  ∧ run 60 (on_its_own WrapsExample.first_rebuilt)
    = Exit 1%positive WrapsExample.result empty_heap no_reads
  ∧ run 60 (on_its_own WrapsExample.rebuilt)
    = Exit 1%positive WrapsExample.result empty_heap no_reads
  ∧ program_refines_observably WrapsExample.unit_expr WrapsExample.rebuilt.
