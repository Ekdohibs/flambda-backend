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
(* The fragment of FlambdaReaperUnboxingRefinesOver2Spec.v, with the flow
   for calls that may over-apply: such a call, of unknown arity with two
   groups of arguments or more, asks [ind_code_ok_o1] only of the code that
   the flow may bring to its callee and of the code that what is left of
   an over-application may call; the code elsewhere may change its calling
   convention.

   This file is to be reviewed: the code that what is left of an
   over-application may call ([node_reach], [rem_reach]), the fragment
   ([calls_unboxing_reach_rem_nr], which is [calls_unboxing_reach_over_nr]
   but for [ind_call_ok_f3]; and [calls_unboxing_flow3_nr]), the statements
   and the example. The proofs are in
   FlambdaReaperUnboxingRefinesReach3Proofs.v, which needs no review.

   The flow is that of the first rebuilt unit, analysed again, as in
   FlambdaReaperUnboxingRefinesReach2Spec.v. A call of [f] that may
   over-apply calls first the closure of what [f] holds, or that of a
   partial application: the code of [flow_reach_u f]. Where it over-applies
   code [A], what is left of it calls what [A] returns with the groups that
   remain: [A]'s last witness of unknown arity is read on past by the call,
   so that its first result has a use (FlambdaReaperUnboxingRefinesOver2Spec.v),
   and what [A] returns may come to that result. So what is left of an
   over-application calls the code that the flow may bring to the first
   result, with a use, of a witness of code: [rem_reach]. [node_reach F1 T1
   n] is [flow_reach_u] at the node [n]: any code if [n] has any source,
   otherwise that of the call witnesses of the sources of [n], of known
   arity (or any, if the witness has any source) or of unknown arity.

   [ind_call_ok_f3] is [ind_call_ok_o2], but a call of [f] that may
   over-apply asks [ind_code_ok_o1] only of the code of [reach false f] and
   of [rem]. [calls_unboxing_flow3_nr] takes [flow_reach2 F1 T1] for
   [reach], [over_reach F1 T1] for [ovr] and [rem_reach F1 T1] for [rem].

   [rebuild_unbox_flow3_refines_cc_statement],
   [rebuild_unbox_flow3_refines_observably_statement] and
   [rebuild_unbox_flow3_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesReach2Spec.v for the fragment
   [calls_unboxing_flow3_nr], with the same premises.

   [calls_unboxing_flow2_in_flow3_statement] and
   [calls_unboxing_over2_in_flow3_statement]: every unit of
   [calls_unboxing_flow2_nr] or of [calls_unboxing_over2_nr] is of
   [calls_unboxing_flow3_nr], for the same analysis of the first rebuilt
   unit.

   [flow3_example_statement]: [TwoGroupsExample] of
   FlambdaReaperUnboxingRefinesOver2Spec.v with the least table of
   [cannot_change_calling_convention], which has 50 and 51 and not 52:
   code 52, called directly, changes its calling convention.
   [calls_unboxing_over2] refuses it (the call with two groups asks every
   code to keep its calling convention), [rem_reach] holds of 51 alone, and
   [calls_unboxing_flow3] accepts it with the least solution of the first
   rebuilt unit; the rebuilt unit refines the unit up to observation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
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
  FlambdaReaperUnboxingRefinesOver2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The code that what is left of an over-application may call              *)
(* ======================================================================== *)

(* The code that the flow may bring to the node [n], partial applications
   included: [flow_reach_u] at a node. *)
Definition node_reach (F1 : facts) (T1 : tables) (n : node) (id : code_id)
    : bool :=
  mem_node n (t_any_source T1)
  || existsb
       (fun '(x, a) =>
          node_eqb x n
          && (alloc_has_code F1 a id || alloc_unknown_code F1 a id))
       (t_sources T1).

(* The code that the flow may bring to the first result, with a use, of a
   witness of code. *)
Definition rem_reach (F1 : facts) (T1 : tables) (id : code_id) : bool :=
  existsb
    (fun '(w, relation, c) =>
       field_eqb relation Field_code_id_of_call_witness
       && match c with
          | Node_code_id _ => true
          | _ => false
          end
       && existsb
            (fun '(w', relation', r) =>
               node_eqb w' w
               && field_eqb relation' (Field_return_of_call (Normal 0))
               && mem_node r (t_has_usage T1)
               && node_reach F1 T1 r id)
            (f_constructor F1))
    (f_constructor F1).

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
Variable nr : code_id -> bool.

Local Abbreviation code_pdescs :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs sol cannot_change
     code_deps unit_code_ids unboxed).
Local Abbreviation args_ok2 :=
  (FlambdaReaperUnboxingRefinesCallsSpec.args_ok2 sol unboxed).
Local Abbreviation code_exn_decision :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_exn_decision sol code_deps
     unboxed).
Local Abbreviation exn_cont_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.exn_cont_ok sol unboxed).
Local Abbreviation kept_args_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.kept_args_ok F sol code_deps codes
     unboxed).
Local Abbreviation handler_pdescs :=
  (FlambdaReaperUnboxingRefinesCallsSpec.handler_pdescs sol unboxed).
Local Abbreviation apply_cont_ok2 :=
  (FlambdaReaperUnboxingRefinesCallsSpec.apply_cont_ok2 sol unboxed).
Local Abbreviation code_param_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_param_ok unboxed).
Local Abbreviation live_conts :=
  (FlambdaReaperUnboxingRefinesCallsSpec.live_conts F sol code_deps codes).
Local Abbreviation masked_args_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.masked_args_ok sol unboxed).
Local Abbreviation call_mask :=
  (FlambdaReaperUnboxingRefinesCallsSpec.call_mask F).
Local Abbreviation ind_exn_decision :=
  (FlambdaReaperUnboxingRefinesCallsSpec.ind_exn_decision sol code_deps
     codes unboxed).

(* The code that the flow may bring to a variable, for a call of known
   arity or not. *)
Variable reach : bool -> variable -> code_id -> bool.
(* The code that a call may over-apply. *)
Variable ovr : code_id -> bool.
(* The code that what is left of an over-application may call. *)
Variable rem : code_id -> bool.

Local Abbreviation never_over :=
  (FlambdaReaperUnboxingRefinesOver1Spec.never_over codes).
Local Abbreviation ind_code_ok_o1 :=
  (FlambdaReaperUnboxingRefinesOver1Spec.ind_code_ok_o1 sol cannot_change
     code_deps unit_code_ids unboxed).

(* [ind_call_ok_o2], asking [ind_code_ok_o1] of a call that may
   over-apply only for the code that the flow may bring to its callee [f],
   as the closure of what [f] holds ([reach false f]), and the code that
   what is left of an over-application may call ([rem]). *)
Definition ind_call_ok_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable)
    : bool :=
  plain_simple unboxed (Var f) && is_var_used sol f Value
  && negb (apply_never_returns F sol code_deps codes ap)
  && Nat.eqb (length (ap_args ap)) (length (unarize (ap_args_arity ap)))
  && masked_args_ok (call_mask sol f ap)
       (call_mask (solution_without_unboxed sol unboxed) f ap) (ap_args ap)
  && exn_cont_ok kinds C (ap_exn_continuation ap)
       (Pdesc ind_exn_decision true Value) (ap_exn_extra_args ap)
  && match lookup (ap_continuation ap) C with
     | Some (kps, true) =>
       let kinds_r := unarize (ap_return_arity ap) in
       Nat.eqb (length kps) (length kinds_r)
       && forallb2 result_ok kps (ind_results kps kinds_r)
       && forallb
            (fun '(id, md) =>
               never_over ap
               && negb (reach (known_arity_call (ap_call_kind ap)) f id)
               || negb (never_over ap) && negb (reach false f id || rem id)
               || ind_code_ok_o1 (never_over ap || negb (ovr id)) kps
                    (ap_return_arity ap) id md)
            codes
     | _ => false
     end.

(* [call_ok_o2], [calls_expr_o2], [calls_handler_o2], [calls_binding_o2]
   and [calls_code_o2] of FlambdaReaperUnboxingRefinesOver2Spec.v, with
   [ind_call_ok_f3]. *)
Definition call_ok_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap with
  | Var f, Some id =>
    kind_calls (ap_call_kind ap) id && plain_simple unboxed (Var f)
    && is_var_used sol f Value
    && (negb (code_never_returns sol code_deps id) || nr id)
    && match lookup id codes with
       | Some metadata =>
         match code_pdescs id metadata with
         | Some (pds, rds) =>
           arity_eqb (ap_args_arity ap) (cm_params_arity metadata)
           && arity_eqb (ap_return_arity ap) (cm_result_arity metadata)
           && is_none
                (my_closure_decision_u cannot_change code_deps unit_code_ids
                   unboxed id)
           && Nat.eqb (length (ap_args ap)) (length pds)
           && Nat.eqb (length pds)
                (length (unarize (cm_params_arity metadata)))
           && Nat.eqb (length rds)
                (length (unarize (cm_result_arity metadata)))
           && forallb pdesc_ok pds && forallb pdesc_ok rds
           && match
                code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id metadata
              with
              | Not_changing_u => true
              | Changing_u _ _ => args_ok2 kinds pds (ap_args ap)
              end
           && match code_exn_decision id with
              | Some d =>
                exn_cont_ok kinds C (ap_exn_continuation ap)
                  (Pdesc d true Value) (ap_exn_extra_args ap)
              | None => false
              end
           && match
                code_change_u sol cannot_change code_deps unit_code_ids
                  unboxed id metadata
              with
              | Not_changing_u => kept_args_ok f ap
              | Changing_u _ _ => true
              end
           && (code_never_returns sol code_deps id
               || match lookup (ap_continuation ap) C with
                  | Some (kps, true) => forallb2 result_ok kps rds
                  | _ => false
                  end)
         | None => false
         end
       | None => false
       end
  | Var f, None => ind_call_ok_f3 kinds C ap f
  | _, _ => false
  end.

Fixpoint calls_expr_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_f3 ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_f3 bindings && calls_expr_f3 kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_f3 kinds C ps h
    && calls_expr_f3 kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_f3 kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_f3 kinds C' body
  | Apply ap => call_ok_f3 kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_f3
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_f3 (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_f3 id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_f3 (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match code_pdescs id metadata with
       | Some (pds, rds) =>
         Nat.eqb (length params) (length pds)
         && Nat.eqb (length pds) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rds) (length (unarize (cm_result_arity metadata)))
         && forallb pdesc_ok pds && forallb pdesc_ok rds
         && forallb2 code_param_ok params pds
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && plain_simple unboxed (Var my_closure)
         && negb (Pos.eqb ret exn)
         && match code_exn_decision id with
            | Some d =>
              calls_expr_f3
                (rev ((my_closure, Value)
                      :: with_kinds params
                           (unarize (cm_params_arity metadata))))
                [(ret, (rds, true)); (exn, ([Pdesc d true Value], false))]
                body
            | None => false
            end
       | None => false
       end
    && (negb (nr id) || negb (among ret (live_conts body)))
  end.

End Fragment.

(* [calls_unboxing_reach_over_nr] with [ind_call_ok_f3]. *)
Definition calls_unboxing_reach_rem_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_f3 F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr reach ovr rem [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow, the code that a call may over-apply and the code that what is
   left of an over-application may call, read off the facts [F1] and the
   tables [T1] of the first rebuilt unit. *)
Definition calls_unboxing_flow3_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (F1 : facts) (T1 : tables) (e : expr) : bool :=
  calls_unboxing_reach_rem_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F1 T1) (over_reach F1 T1) (rem_reach F1 T1) e.

Definition calls_unboxing_flow3 (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) : bool :=
  calls_unboxing_flow3_nr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox F1 T1 e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_flow3_refines_cc_statement : Prop :=
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
    calls_unboxing_flow3_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_flow3_refines_observably_local_statement : Prop :=
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
    calls_unboxing_flow3_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_flow3_refines_observably_statement : Prop :=
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
    calls_unboxing_flow3_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragments of FlambdaReaperUnboxingRefinesReach2Spec.v and
   FlambdaReaperUnboxingRefinesOver2Spec.v are in this one, for the same
   analysis of the first rebuilt unit. *)
Definition calls_unboxing_flow2_in_flow3_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_flow2_nr nr F T cannot_change code_deps dominated to_unbox
      F1 T1 e
    = true ->
    calls_unboxing_flow3_nr nr F T cannot_change code_deps dominated
      to_unbox F1 T1 e
    = true.

Definition calls_unboxing_over2_in_flow3_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_over2_nr nr F T cannot_change code_deps dominated to_unbox
      F1 T1 e
    = true ->
    calls_unboxing_flow3_nr nr F T cannot_change code_deps dominated
      to_unbox F1 T1 e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* [TwoGroupsExample] of FlambdaReaperUnboxingRefinesOver2Spec.v with the
   least table of [cannot_change_calling_convention], which has 50 and 51
   and not 52: code 52, called directly, changes its calling convention.
   The tables of the unit and of the first rebuilt unit are the least
   solutions of the rules (FlambdaReaperLeast.compute_least). *)
Module Flow3Example.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr := TwoGroupsExample.unit_expr.

Definition facts : facts := TwoGroupsExample.facts.

Definition tables : tables := TwoGroupsExample.tables.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep := TwoGroupsExample.code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

End Flow3Example.

Definition flow3_example_statement : Prop :=
  mem_node (Node_code_id 52%positive) Flow3Example.cannot_change = false
  ∧ calls_unboxing_over2 Flow3Example.facts Flow3Example.tables
      Flow3Example.cannot_change Flow3Example.code_deps [] []
      Flow3Example.facts1 Flow3Example.tables1 Flow3Example.unit_expr
    = false
  ∧ map (rem_reach Flow3Example.facts1 Flow3Example.tables1)
      [50; 51; 52]%positive
    = [false; true; false]
  ∧ calls_unboxing_flow3 Flow3Example.facts Flow3Example.tables
      Flow3Example.cannot_change Flow3Example.code_deps [] []
      Flow3Example.facts1 Flow3Example.tables1 Flow3Example.unit_expr
    = true
  ∧ program_refines_observably Flow3Example.unit_expr
      (rebuild_unit_unbox_fixed Flow3Example.facts Flow3Example.tables
         Flow3Example.cannot_change Flow3Example.code_deps [] [] true
         Flow3Example.unit_expr).
