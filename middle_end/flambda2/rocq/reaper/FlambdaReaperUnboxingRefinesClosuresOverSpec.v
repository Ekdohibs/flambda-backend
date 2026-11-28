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
   FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v: a call of unknown
   arity whose code is not known may over-apply.

   This file is to be reviewed: the fragment ([repr_over_unboxing_nr],
   [repr_flow_over_nr]), the statements and the example. The proofs are
   in FlambdaReaperUnboxingRefinesClosuresOverProofs.v,
   FlambdaReaperUnboxingRefinesClosuresProofs.v and
   FlambdaReaperUnboxingRefinesClosuresSimProofs.v, which need no review.

   A call of unknown arity with more groups of arguments than the code it
   calls has parameters calls that code with the first groups, and what is
   left of the application calls the value that the code returns with the
   others ([Over_application]); that call may over-apply in turn. A plain
   name holds the same value in both rebuilt units, so both runs call the
   same closure, which reads its [my_closure] as the first rebuild does
   only if that [my_closure] does not change. The flow says which code the
   call may call: that of [flow_reach2] of the callee, for the code it
   calls first, and that of [rem_reach]
   (FlambdaReaperUnboxingRefinesReach3Spec.v), the code that the first
   result, with a use, of a witness of code may be the closure of, for
   what is left of an over-application; both read off the first rebuilt
   unit, analysed again ([F1], [T1]).

   [repr_over_holds_wraps_statement]: the fragment holds that of
   FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v
   ([repr_wraps_unboxing_nr]), for the same tables, code taken to never
   return and flow, whatever [rem].

   [rebuild_repr_over_refines_cc_statement],
   [rebuild_repr_over_observably_statement],
   [rebuild_repr_over_local_statement] and their [_with_types] forms: the
   statements of FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v with
   [repr_flow_over_nr] in place of [repr_flow_wraps_nr]; those are
   corollaries.

   The fragment ([repr_over_unboxing_nr]) is that of
   FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v with one more kind of
   call ([over_call]): a call of unknown arity of a variable that is
   neither unboxed nor changed, whose code is not known, which returns to
   a continuation that keeps every result, with plain arguments at the
   same mask in both rebuilds, as [ind_call_r] has it, but with any
   number of groups of arguments; the code that the flow may bring to
   the closure of what the callee holds, and the code of [rem], have
   their [my_closure] plain.

   What is left out, for the proof: such calls that never return, and the
   rest of what FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v leaves
   out.

   [repr_over_example_statement]: a unit of [repr_flow_over_nr] that is
   not of [repr_flow_wraps_nr] ([OverExample]: a closure of code of one
   parameter called with two groups of arguments, the code returning the
   closure that what is left calls); its rebuild in the mode of the fix
   refines it up to observation. *)

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
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesClosuresReachSpec
  FlambdaReaperUnboxingRefinesClosuresWrapsSpec.
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
(* The code that what is left of an over-application may call. *)
Variable rem : code_id -> bool.

Local Abbreviation plain_c := (plain_c unboxed changed).
Local Abbreviation changed_name := (changed_name changed).

(* A call of unknown arity of a variable [f] whose code is not known,
   which may over-apply: as [ind_call_r] has a call of unknown arity that
   returns (of [f] neither unboxed nor changed, its arguments plain and at
   the same mask in both rebuilds, to a continuation that keeps every
   result), but with any number of groups of arguments; the code that
   the flow may bring to the closure of what [f] holds ([reach false f])
   and the code that what is left of an over-application may call
   ([rem]) have their [my_closure] plain. *)
Definition over_call (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap, ap_call_kind ap with
  | Var f, None, Indirect_unknown_arity =>
    is_none (fields_of unboxed f) && is_none (changed_name (Node_variable f))
    && forallb plain_c (ap_args ap)
    && forallb2 Bool.eqb (call_mask F sol f ap)
         (call_mask F (solution_without_unboxed sol unboxed) f ap)
    && extra_args_ok unboxed changed C ap
    && negb (apply_never_returns F sol code_deps codes ap)
    && returns_ok C (ap_continuation ap)
         (map (fun kind => Decided (Keep kind))
            (unarize (ap_return_arity ap)))
    && forallb
         (fun '(id, _) =>
            negb (reach false f id || rem id)
            || plain_my_closure changed my_closures id)
         codes
  | _, _, _ => false
  end.

(* A call of [repr_call_x], or one that may over-apply. *)
Definition repr_call_o (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  repr_call_x F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr reach C ap
  || over_call C ap.

(* [repr_expr_o], [repr_handler_o], [repr_binding_o] and [repr_code_o] of
   FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v, with [repr_call_o]. *)
Fixpoint repr_expr_o (kinds : finite_map flambda_kind)
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
    && repr_expr_o ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_o bindings && repr_expr_o kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_o kinds C ds h
    && repr_expr_o kinds ((k, (normed ds, true)) :: C) body
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
         repr_handler_o kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_o kinds C' body
  | Apply ap => repr_call_o C ap
  | Apply_cont ac => apply_cont_ok_w unboxed changed C ac
  | Switch scrutinee arms =>
    plain_c scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok_w unboxed changed C ac) arms
  | Invalid => true
  end
with repr_handler_o (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided (normed ds)
    && forallb (fun x => plain_c (Var x)) params
    && repr_expr_o (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         C handler
  end
with repr_binding_o (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_o id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_code_o (id : code_id) (c : code) {struct c} : bool :=
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
         && repr_expr_o
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
              && repr_expr_o
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
   taken to never return, the code [reach] that the flow may bring to the
   callee of a call, and the code [rem] that what is left of an
   over-application may call. *)
Definition repr_over_unboxing_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (reach : bool -> variable -> code_id -> bool) (rem : code_id -> bool)
    (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr_o F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr reach rem [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow and the code of what is left of an over-application being read
   off the facts [F1] and the tables [T1] of the first rebuilt unit. *)
Definition repr_flow_over_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (F1 : facts) (T1 : tables) (e : expr) : bool :=
  repr_over_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr (flow_reach2 F1 T1) (rem_reach F1 T1) e.
(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition repr_over_holds_wraps_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (reach : bool -> variable -> code_id -> bool)
    (rem : code_id -> bool) (e : expr),
    repr_wraps_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr reach e
    = true ->
    repr_over_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr reach rem e
    = true.

Definition rebuild_repr_over_refines_cc_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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

Definition rebuild_repr_over_observably_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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
Definition rebuild_repr_over_local_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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
   [rebuild_repr_wraps_with_types_refines_cc_statement]
   (FlambdaReaperUnboxingRefinesClosuresWrapsSpec.v). *)
Definition rebuild_repr_over_with_types_refines_cc_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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
   premises of [rebuild_repr_wraps_with_types_local_statement] but the
   fragment. *)
Definition rebuild_repr_over_with_types_local_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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
Definition rebuild_repr_over_with_types_observably_statement : Prop :=
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
    repr_flow_over_nr nr is_local (traverse_unit flag u e) T
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

(* The unit calls [x61], the closure of the code 56, with two groups of
   arguments, 32 and 32, a call of unknown arity whose code is not known:
   56, of one parameter, returns the closure [x44] of the code 57, which
   what is left of the over-application calls with 32, and which returns
   its parameter plus 1. The handler 22 of the result returns the block of
   it and [x61], which so escapes. No set changes its representation here:
   [rem_reach] holds of every code as soon as the first result, with a
   use, of some witness of code has any source in the tables of the first
   rebuilt unit; beside a call of 54 through [x60] of [TwoFieldsExample],
   it holds of 54, whose [my_closure] changes. *)
Module OverExample.

Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition code_56 : code :=
  Code [42] 43 5 6 metadata
    (Let 44 Value (Closure_of_set 73 [(73, 57)] [])
       (return_to 5 [Var 44])).

Definition code_57 : code :=
  Code [46] 47 7 8 metadata
    (Let 48 Value (TwoFieldsExample.add (Var 46) (int 1))
       (return_to 7 [Var 48])).

(* The call of [x61] with two groups of arguments. *)
Definition call_61 : apply_expr :=
  {| ap_callee := Var 61; ap_call_kind := Indirect_unknown_arity;
     ap_args := [int 32; int 32]; ap_continuation := 22;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 56 code_56; Bind_code 57 code_57]
    (Let 61 Value (Closure_of_set 71 [(71, 56)] [])
       (Let_cont_non_recursive 22
          (Handler [32] (make_handler_metadata [Value])
             (Let 33 Value (make_block [Var 32; Var 61])
                (return_to 1 [Var 33])))
          (Apply call_61))).

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

Definition rebuilt : expr :=
  rebuild_unit_unbox_repr_fixed no_local facts tables cannot_change code_deps
    dominated to_unbox tcr true unit_expr.

(* What the unit and its rebuild return: 32 + 1, and [x61]. *)
Definition result : list value :=
  [Value_block 0
     [Value_const (Tagged_immediate 33);
      Value_closure_of_set 71 [(71, 56)] []]].

End OverExample.

(* With the tables of the least solution, of the unit and of the first
   rebuilt unit: the set of [x61] keeps its representation; 56 and 57
   cannot change their calling convention; the flow brings 56 to the
   closure of what [x61] holds, and 57 is code that what is left of an
   over-application may call. The unit is not of
   [repr_flow_wraps_nr] (the call of [x61] has two groups of arguments),
   and is of [repr_flow_over_nr]; the unit and its rebuild in the mode of
   the fix return [result]; and the rebuilt unit refines the unit up to
   observation. *)
Definition repr_over_example_statement : Prop :=
  check_dominated OverExample.tables OverExample.dominated = true
  ∧ check_to_unbox no_local OverExample.facts OverExample.tables []
      max_unbox_size false OverExample.to_unbox
    = true
  ∧ check_to_change_representation no_local OverExample.facts
      OverExample.tables [] max_unbox_size false OverExample.to_unbox
      OverExample.tcr
    = true
  ∧ check_sets_whole OverExample.facts OverExample.tables
      OverExample.to_unbox OverExample.tcr
    = true
  ∧ mem_node (v 61) OverExample.tcr = false
  ∧ mem_node (c 56) OverExample.cannot_change = true
  ∧ mem_node (c 57) OverExample.cannot_change = true
  ∧ flow_reach2 OverExample.facts1 OverExample.tables1 false 61%positive
      56%positive
    = true
  ∧ rem_reach OverExample.facts1 OverExample.tables1 57%positive = true
  ∧ repr_flow_wraps_nr (fun _ => false) no_local OverExample.facts
      OverExample.tables OverExample.cannot_change OverExample.code_deps
      OverExample.dominated OverExample.to_unbox OverExample.tcr
      OverExample.facts1 OverExample.tables1 OverExample.unit_expr
    = false
  ∧ repr_flow_over_nr (fun _ => false) no_local OverExample.facts
      OverExample.tables OverExample.cannot_change OverExample.code_deps
      OverExample.dominated OverExample.to_unbox OverExample.tcr
      OverExample.facts1 OverExample.tables1 OverExample.unit_expr
    = true
  ∧ check_unboxed_reads OverExample.facts OverExample.tables
      OverExample.dominated OverExample.to_unbox
    = true
  ∧ run 60 (on_its_own OverExample.unit_expr)
    = Exit 1%positive OverExample.result empty_heap no_reads
  ∧ run 60 (on_its_own OverExample.rebuilt)
    = Exit 1%positive OverExample.result empty_heap no_reads
  ∧ program_refines_observably OverExample.unit_expr OverExample.rebuilt.
