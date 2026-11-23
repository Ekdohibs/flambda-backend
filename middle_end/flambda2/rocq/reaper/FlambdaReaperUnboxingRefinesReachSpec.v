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

(* The fragment of FlambdaReaperUnboxingRefinesOver1Spec.v, widened by the
   flow: a call of known arity whose code is not known asks
   [ind_code_ok] only of the code that the flow may bring to its callee.

   This file is to be reviewed: the code that the flow brings to a
   variable ([flow_reach]), the fragment ([calls_unboxing_flow_nr], which
   is [calls_unboxing_over1_nr] but for [ind_call_ok_f]), the statements
   and the example. The proofs are in
   FlambdaReaperUnboxingRefinesReachProofs.v, which needs no review.

   [ind_call_ok_o1] asks [ind_code_ok] of every code of the unit, as the
   simulation of FlambdaReaperUnboxingRefinesCallsSimProofs.v relates the
   two rebuilt units with no flow invariant: the call may reach any code.
   So a unit with one such call and one code that changes its calling
   convention was out of the fragment.

   The flow is that of the first rebuilt unit [first_rebuild] (the change
   of calling conventions alone), analysed again: its facts [F1] and
   tables [T1], which pass [check_closed] and [check_local_reads_used].
   By the run of FlambdaReaperFlowSpec.flow_sound on it, every closure
   that a variable [f] of its run holds may come from an allocation that
   is a source of [f] in [T1], or [f] has any source; the proof has more
   than the statement says of that allocation: the witness of known arity
   of the code of the closure, which names its code id, or which has any
   source ([flow_reach]). The flow of the original unit would not do: its
   run may go wrong where the first rebuilt unit does not (a dead [Let]
   that the rebuild drops), and the refinement of the two rebuilt units
   must hold there too.

   [ind_call_ok_f] is [ind_call_ok_o1], but for a call of known arity,
   which asks [ind_code_ok_o1] only of the code in [flow_reach] of its
   callee. A call of unknown arity still asks it of every code: its
   callee may be a partial application, whose code the flow of a
   variable does not give.

   [rebuild_unbox_flow_refines_cc_statement]: on the fragment
   [calls_unboxing_flow_nr], with the analysis [F1], [T1] of the first
   rebuilt unit, the rebuild with unboxing in the mode of the fix
   refines the first rebuilt unit;
   [rebuild_unbox_flow_refines_observably_statement] and
   [rebuild_unbox_flow_refines_observably_local_statement]: so it refines
   the unit up to observation, with the premises of
   FlambdaReaperCallingConventionsSpec.rebuild_cc_refines_observably.

   [calls_unboxing_over1_in_flow_statement]: every unit of
   [calls_unboxing_over1_nr] is of [calls_unboxing_flow_nr], whatever the
   analysis of the first rebuilt unit.

   [flow_example_statement]: [KnownArityExample] of
   FlambdaReaperUnboxingRefinesOverSpec.v, where code 52, called
   directly, changes its calling convention (the least table of
   [cannot_change_calling_convention] has 50 and 51 only): the call of
   known arity, of 50 or 51, makes [calls_unboxing_over1] refuse the unit,
   as 52 does not keep its calling convention; [calls_unboxing_flow]
   accepts it, with the least solution of the first rebuilt unit, whose
   [flow_reach] of the callee is 50 and 51; and the rebuilt unit refines
   the unit up to observation. *)

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
  FlambdaReaperUnboxingRefinesOver1Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The flow                                                                 *)
(* ======================================================================== *)

(* The allocation [a] has a witness of known arity that names the code
   [id], or one with any source, which may be of any code. *)
Definition alloc_has_code (F1 : facts) (a : node) (id : code_id) : bool :=
  existsb
    (fun '(b, relation, w) =>
       node_eqb b a
       && field_eqb relation (Field_call_witness Known_arity_code_pointer)
       && (mem_node w (f_any_source F1)
           || existsb
                (fun '(w', relation', c) =>
                   node_eqb w' w
                   && field_eqb relation' Field_code_id_of_call_witness
                   && node_eqb c (Node_code_id id))
                (f_constructor F1)))
    (f_constructor F1).

(* The code that the flow may bring to [f]: any, if [f] has any source;
   otherwise that of a source of [f]. *)
Definition flow_reach (F1 : facts) (T1 : tables) (f : variable)
    (id : code_id) : bool :=
  mem_node (Node_variable f) (t_any_source T1)
  || existsb
       (fun '(x, a) => node_eqb x (Node_variable f) && alloc_has_code F1 a id)
       (t_sources T1).

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

(* The code that the flow may bring to a variable. *)
Variable reach : variable -> code_id -> bool.

Local Abbreviation never_over :=
  (FlambdaReaperUnboxingRefinesOver1Spec.never_over codes).
Local Abbreviation ind_code_ok_o1 :=
  (FlambdaReaperUnboxingRefinesOver1Spec.ind_code_ok_o1 sol cannot_change
     code_deps unit_code_ids unboxed).

(* [ind_call_ok_o1], asking [ind_code_ok_o1] of a call of known arity only
   for the code that the flow may bring to its callee [f]. *)
Definition ind_call_ok_f (kinds : finite_map flambda_kind)
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
               known_arity_call (ap_call_kind ap) && negb (reach f id)
               || ind_code_ok_o1 (never_over ap) kps (ap_return_arity ap) id
                    md)
            codes
     | _ => false
     end.

(* [call_ok_o1], [calls_expr_o1], [calls_handler_o1], [calls_binding_o1]
   and [calls_code_o1] of FlambdaReaperUnboxingRefinesOver1Spec.v, with
   [ind_call_ok_f]. *)
Definition call_ok_f (kinds : finite_map flambda_kind)
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
  | Var f, None => ind_call_ok_f kinds C ap f
  | _, _ => false
  end.

Fixpoint calls_expr_f (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_f ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_f bindings && calls_expr_f kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_f kinds C ps h
    && calls_expr_f kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_f kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_f kinds C' body
  | Apply ap => call_ok_f kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_f (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_f
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_f (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_f id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_f (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_f
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

(* [calls_unboxing_over1_nr] with [ind_call_ok_f]. *)
Definition calls_unboxing_reach_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : variable -> code_id -> bool) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_f F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr reach [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow being read off the facts [F1] and the tables [T1] of the first
   rebuilt unit. *)
Definition calls_unboxing_flow_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (F1 : facts) (T1 : tables) (e : expr) : bool :=
  calls_unboxing_reach_nr nr F T cannot_change code_deps dominated to_unbox
    (flow_reach F1 T1) e.

Definition calls_unboxing_flow (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) : bool :=
  calls_unboxing_flow_nr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox F1 T1 e.

(* The first rebuilt unit: the change of calling conventions alone. *)
Definition first_rebuild (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) : expr :=
  rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) preserve_direct_calls e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_flow_refines_cc_statement : Prop :=
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
    calls_unboxing_flow_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_flow_refines_observably_local_statement : Prop :=
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
    calls_unboxing_flow_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_flow_refines_observably_statement : Prop :=
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
    calls_unboxing_flow_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragment of FlambdaReaperUnboxingRefinesOver1Spec.v is in this one,
   whatever the analysis of the first rebuilt unit. *)
Definition calls_unboxing_over1_in_flow_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_over1_nr nr F T cannot_change code_deps dominated to_unbox
      e
    = true ->
    calls_unboxing_flow_nr nr F T cannot_change code_deps dominated to_unbox
      F1 T1 e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* [KnownArityExample] of FlambdaReaperUnboxingRefinesOverSpec.v with the
   least table of [cannot_change_calling_convention], 51 and 50: code 52,
   called directly, changes its calling convention, so [ind_code_ok_o1]
   refuses it for the call of known arity of 50 or 51. The tables of the
   unit and of the first rebuilt unit are the least solutions of the rules
   (FlambdaReaperLeast.compute_least). *)
Module FlowExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr := KnownArityExample.unit_expr.

Definition facts : facts := KnownArityExample.facts.

Definition tables : tables := KnownArityExample.tables.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep := KnownArityExample.code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

End FlowExample.

Definition flow_example_statement : Prop :=
  FlowExample.cannot_change = [Node_code_id 51%positive; Node_code_id
    50%positive]
  ∧ calls_unboxing_over1 FlowExample.facts FlowExample.tables
      FlowExample.cannot_change FlowExample.code_deps [] []
      FlowExample.unit_expr
    = false
  ∧ calls_unboxing_flow FlowExample.facts FlowExample.tables
      FlowExample.cannot_change FlowExample.code_deps [] []
      FlowExample.facts1 FlowExample.tables1 FlowExample.unit_expr
    = true
  ∧ program_refines_observably FlowExample.unit_expr
      (rebuild_unit_unbox_fixed FlowExample.facts FlowExample.tables
         FlowExample.cannot_change FlowExample.code_deps [] [] true
         FlowExample.unit_expr).
