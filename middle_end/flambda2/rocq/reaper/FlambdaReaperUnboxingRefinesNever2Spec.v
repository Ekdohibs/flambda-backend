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

(* The fragment of FlambdaReaperUnboxingRefinesNeverSpec.v, widened to the
   calls whose code is not known that never return and complete a partial
   application of code that never returns.

   This file is to be reviewed: the check of such a call
   ([never_code_ok2], [ind_never_ok2], [call_ok_n2]), the check of a term
   with a check of the calls given ([calls_expr_w]), the fragments
   ([calls_unboxing_never2_nr], read off the first rebuilt unit analysed
   again, and [calls_unboxing_unit_never2_nr], read off the unit's own
   analysis), the statements and the example. The proofs are in
   FlambdaReaperUnboxingRefinesNever2Proofs.v, which needs no review.

   [never_code_ok] asks the code that the flow may bring to the callee [f]
   of a call of unknown arity that never returns to have no more
   parameters than the call has groups of arguments, so that the call is
   a full call or an over-application of it, and never a partial
   application (which the first run would give to the frame that the
   second run does not keep). A call that completes a partial application
   gives the code fewer groups than it has parameters: the partial
   application holds the others. [never_code_ok2] admits besides code
   that has one parameter more than the call has groups where the flow
   never brings the closure of the code itself to [f] ([reach true f id]
   is false: [f] holds no closure of it, only partial applications of
   one): a partial application that the machine makes holds a group of
   arguments at least (a call with no argument is undefined), so that the
   call gives the code all its parameters. [reach true] is the reach of a
   call of known arity, whose callee is a closure: the code of the
   closures that may come to [f] (FlambdaReaperUnboxingRefinesReachSpec
   .flow_reach), which leaves out the partial applications; [reach false]
   has both.

   The bound is one group. A partial application of [k] groups is put by
   the flow at the allocation of the first result of the [k]-th witness of
   unknown arity of its code, so that [k], read off the chain of
   witnesses, would bound the groups that it holds more finely; the
   relation of the proofs (FlambdaReaperUnboxingRefinesCallsSimProofs.uc)
   would then need that bound as a run invariant of the first run, beside
   [flow_ok], of which [RCH true] and [RCH false] give one bit.

   [calls_expr_w], [calls_handler_w], [calls_binding_w] and
   [calls_code_w] are [calls_expr_n] and the others of
   FlambdaReaperUnboxingRefinesNeverSpec.v with the check of a call
   [call_ok] given: [calls_expr_n] is [calls_expr_w] of [call_ok_n], and
   the fragment here is [calls_expr_w] of [call_ok_n2].

   [rebuild_unbox_nr_ind2_refines_cc_statement],
   [rebuild_unbox_nr_ind2_refines_observably_statement] and
   [rebuild_unbox_nr_ind2_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesNeverSpec.v for
   [calls_unboxing_never2_nr], with the same premises.
   [rebuild_unbox_unit_never2_refines_cc_statement],
   [rebuild_unbox_unit_never2_refines_observably_statement] and
   [rebuild_unbox_unit_never2_observably_local_statement]: those of
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v for
   [calls_unboxing_unit_never2_nr], with the same premises
   ([used_reach_covered] among them).

   [calls_unboxing_never_in_never2_statement],
   [calls_unboxing_unit_never_in_never2_statement]: every unit of the
   fragments of FlambdaReaperUnboxingRefinesNeverSpec.v and
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v is of those here.

   The example of the shape is
   FlambdaReaperUnboxingRefinesNeverShapesSpec
   .never_partial2_example_statement. *)

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
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
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
Variable nr : code_id -> bool.

Local Abbreviation code_pdescs :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs sol cannot_change
     code_deps unit_code_ids unboxed).
Local Abbreviation code_exn_decision :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_exn_decision sol code_deps
     unboxed).
Local Abbreviation exn_cont_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.exn_cont_ok sol unboxed).
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
Local Abbreviation never_code_ok :=
  (FlambdaReaperUnboxingRefinesNeverSpec.never_code_ok sol cannot_change
     code_deps unit_code_ids unboxed nr).

Section Calls.

(* The check of a call. *)
Variable call_ok : finite_map flambda_kind -> finite_map (list pdesc * bool)
  -> apply_expr -> bool.

(* [calls_expr_n] and the others, with the check [call_ok] of a call. *)
Fixpoint calls_expr_w (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_w ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_w bindings && calls_expr_w kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_w kinds C ps h
    && calls_expr_w kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_w kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_w kinds C' body
  | Apply ap => call_ok kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_w (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_w
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_w (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_w id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_w (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_w
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


End Calls.

(* The code that the flow may bring to a variable, for a call of known
   arity or not; the code that a call may over-apply; the code that what
   is left of an over-application may call. *)
Variable reach : bool -> variable -> code_id -> bool.
Variable ovr : code_id -> bool.
Variable rem : code_id -> bool.

(* [never_code_ok], or, for a call [ap] of [f] of unknown arity, code of
   [nr] that keeps its calling convention, which has one parameter more
   than the call has groups of arguments at most, and whose closure the
   flow never brings to [f] itself. *)
Definition never_code_ok2 (known : bool) (ap : apply_expr) (f : variable)
    (id : code_id) (md : code_metadata) : bool :=
  never_code_ok known ap id md
  || never_code_ok true ap id md && negb (reach true f id)
     && Nat.leb (length (cm_params_arity md)) (S (length (ap_args_arity ap))).

(* [ind_never_ok] with [never_code_ok2]. *)
Definition ind_never_ok2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable)
    : bool :=
  plain_simple unboxed (Var f) && is_var_used sol f Value
  && negb (is_direct (ap_call_kind ap))
  && Nat.eqb (length (ap_args ap)) (length (unarize (ap_args_arity ap)))
  && masked_args_ok (call_mask sol f ap)
       (call_mask (solution_without_unboxed sol unboxed) f ap) (ap_args ap)
  && exn_cont_ok kinds C (ap_exn_continuation ap)
       (Pdesc ind_exn_decision true Value) (ap_exn_extra_args ap)
  && forallb
       (fun '(id, md) =>
          negb (reach (known_arity_call (ap_call_kind ap)) f id)
          || never_code_ok2 (known_arity_call (ap_call_kind ap)) ap f id md)
       codes.

(* [ind_call_ok_n] and [call_ok_n] with [ind_never_ok2]. *)
Definition ind_call_ok_n2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable)
    : bool :=
  if apply_never_returns F sol code_deps codes ap
  then ind_never_ok2 kinds C ap f
  else
    ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
      reach ovr rem kinds C ap f.

Definition call_ok_n2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap with
  | Var f, None => ind_call_ok_n2 kinds C ap f
  | _, _ =>
    call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr
      reach ovr rem kinds C ap
  end.

End Fragment.

(* [calls_unboxing_reach_never_nr] with [call_ok_n2]. *)
Definition calls_unboxing_reach_never2_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_w F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr
    (call_ok_n2 F (solution_of_tables T) (fun n => mem_node n cannot_change)
       code_deps (static_names code_ids_bound e) (bound_code_metadata e)
       unboxed nr reach ovr rem)
    [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow, the code that a call may over-apply and the code that what is
   left of an over-application may call, read off the facts [F1] and the
   tables [T1] of the first rebuilt unit. *)
Definition calls_unboxing_never2_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (F1 : facts) (T1 : tables) (e : expr) : bool :=
  calls_unboxing_reach_never2_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F1 T1) (over_reach F1 T1) (rem_reach F1 T1) e.

(* The same, read off the unit's own analysis. *)
Definition calls_unboxing_unit_never2_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  calls_unboxing_reach_never2_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F T) (over_reach F T) (rem_reach F T) e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_nr_ind2_refines_cc_statement : Prop :=
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
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_nr_ind2_refines_observably_local_statement : Prop :=
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
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind2_refines_observably_statement : Prop :=
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
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_unit_never2_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    calls_unboxing_unit_never2_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragments of FlambdaReaperUnboxingRefinesNeverSpec.v and
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v are in those here. *)
Definition calls_unboxing_never_in_never2_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_never_nr nr F T cannot_change code_deps dominated to_unbox
      F1 T1 e
    = true ->
    calls_unboxing_never2_nr nr F T cannot_change code_deps dominated
      to_unbox F1 T1 e
    = true.

Definition calls_unboxing_unit_never_in_never2_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr),
    calls_unboxing_unit_never_nr nr F T cannot_change code_deps dominated
      to_unbox e
    = true ->
    calls_unboxing_unit_never2_nr nr F T cannot_change code_deps dominated
      to_unbox e
    = true.
