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

(* The fragment of FlambdaReaperUnboxingRefinesReach3Spec.v, with calls
   whose code is not known that never return: a call of a variable [f],
   of known arity or of unknown arity, that [apply_never_returns] makes one
   that never returns, which the fix makes return to a handler [Invalid].
   The fragment of FlambdaReaperUnboxingRefinesCallsSpec.v asked
   [apply_never_returns] to be false of such a call ([ind_call_ok]), and so
   did every fragment after it.

   This file is to be reviewed: the check of such a call
   ([never_code_ok], [ind_never_ok], [ind_call_ok_n]), the fragment
   ([calls_unboxing_reach_never_nr], which is [calls_unboxing_reach_rem_nr]
   but for [ind_call_ok_n]; and [calls_unboxing_never_nr]), the statements
   and the example. The proofs are in
   FlambdaReaperUnboxingRefinesNeverProofs.v, which needs no review.

   The rebuild with unboxing gives such a call a handler [Invalid]; the
   change of calling conventions keeps its continuation, a handler, the
   return continuation of the code it is in, or a wrapper. The two frames
   are not related: the refinement holds only if the first run never
   returns to its frame. It does where the code that the call enters is
   code taken to never return ([nr]), whose body is related with its return
   continuation out of the relation, as for a call of known code. That code
   is read off the flow of the first rebuilt unit, analysed again, as in
   FlambdaReaperUnboxingRefinesReach2Spec.v: [reach true f] for a call of
   known arity, which calls the closure that [f] holds, and [reach false f]
   for a call of unknown arity, which calls that closure or the closure of
   a partial application.

   [ind_never_ok]: the call is not direct; [f] is used and not unboxed; the
   arguments are given at the mask of the call ([masked_args_ok]) and the
   exception continuation takes the exception of [ind_exn_decision], as
   for a call that may return; and every code of the unit that the flow
   may bring to [f] is code of [nr] that keeps its calling convention and,
   for a call of unknown arity, has no more parameters than the call has
   groups of arguments ([never_code_ok]), so that the call is a full call
   or an over-application of it, and never a partial application (which
   the first run would give to the frame). Its continuation is not
   checked. [ind_call_ok_n] is [ind_never_ok] for a call that
   [apply_never_returns], and [ind_call_ok_f3] otherwise.

   The flow is that of the first run, as [flow_reach2] reads it; the code
   that [apply_never_returns] reads is that of the analysis of the unit.
   The fragment asks that the two agree on such a call: the code that the
   flow of the first rebuilt unit brings to [f] is of [nr]. For [nr] the
   code that is used and some result of whose code dep has no source
   ([used_never_returns]), this is what
   FlambdaReaperIndirectNeverReturnsSpec.entered_used_never_returns_statement
   proves of the code that the run of the unit enters at a call of known
   arity.

   [rebuild_unbox_nr_ind_refines_cc_statement],
   [rebuild_unbox_nr_ind_refines_observably_statement] and
   [rebuild_unbox_nr_ind_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesReach3Spec.v for the fragment
   [calls_unboxing_never_nr], with the same premises.

   [calls_unboxing_flow3_in_never_statement]: every unit of
   [calls_unboxing_flow3_nr] is of [calls_unboxing_never_nr], for the same
   analysis of the first rebuilt unit.

   [never_ind_example_statement]: a unit whose codes 50 and 51 raise their
   parameter, so that no result of theirs has a source, and which calls a
   closure of 50 or 51 (chosen by a [Switch]) at known arity, returning to
   the handler 20, or at unknown arity, returning to the handler 24.
   [apply_never_returns] holds of both calls; [calls_unboxing_flow3_nr]
   refuses the unit, and [calls_unboxing_never_nr] accepts it, with the
   least solutions and the codes 50 and 51 taken to never return (the code
   of [used_never_returns]). The rebuild in the mode of the fix makes both
   calls return to a handler [Invalid] and drops the handlers 20 and 24,
   which the change of calling conventions alone keeps; the rebuilt unit
   refines the unit up to observation. *)

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
  FlambdaReaperUnboxingRefinesReach3Spec.
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

(* The code that the flow may bring to a variable, for a call of known
   arity or not. *)
Variable reach : bool -> variable -> code_id -> bool.
(* The code that a call may over-apply. *)
Variable ovr : code_id -> bool.
(* The code that what is left of an over-application may call. *)
Variable rem : code_id -> bool.

(* The code [id] of the unit, with [md], which a call [ap] whose code is
   not known and which never returns may enter ([known]: of known arity):
   code taken to never return, which keeps its calling convention, so that
   the two rebuilds enter it alike; a call of unknown arity gives it as
   many groups of arguments as it has parameters at least. *)
Definition never_code_ok (known : bool) (ap : apply_expr) (id : code_id)
    (md : code_metadata) : bool :=
  nr id
  && match
       code_change_u sol cannot_change code_deps unit_code_ids unboxed id md,
       code_pdescs id md, code_exn_decision id
     with
     | Not_changing_u, Some _, Some _ => true
     | _, _, _ => false
     end
  && (known
      || Nat.leb (length (cm_params_arity md)) (length (ap_args_arity ap))).

Definition is_direct (kind : call_kind) : bool :=
  match kind with
  | Direct _ => true
  | Indirect_known_arity _ | Indirect_unknown_arity => false
  end.

(* A call [ap] of [f] whose code is not known and which never returns: not
   direct; [f] is used and not unboxed; the arguments are given at the
   mask of the call; the exception continuation takes the exception of
   [ind_exn_decision]; and every code that the flow may bring to [f] is
   one it may enter ([never_code_ok]). Its continuation is not checked:
   the rebuild with unboxing gives the call a handler [Invalid]. *)
Definition ind_never_ok (kinds : finite_map flambda_kind)
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
          || never_code_ok (known_arity_call (ap_call_kind ap)) ap id md)
       codes.

(* A call of [f] whose code is not known: [ind_never_ok] if it never
   returns, [ind_call_ok_f3] otherwise. *)
Definition ind_call_ok_n (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable)
    : bool :=
  if apply_never_returns F sol code_deps codes ap
  then ind_never_ok kinds C ap f
  else
    ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
      reach ovr rem kinds C ap f.

(* [call_ok_f3], [calls_expr_f3], [calls_handler_f3], [calls_binding_f3]
   and [calls_code_f3] of FlambdaReaperUnboxingRefinesReach3Spec.v, with
   [ind_call_ok_n]. *)
Definition call_ok_n (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap with
  | Var f, None => ind_call_ok_n kinds C ap f
  | _, _ =>
    call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr
      reach ovr rem kinds C ap
  end.

Fixpoint calls_expr_n (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_n ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_n bindings && calls_expr_n kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_n kinds C ps h
    && calls_expr_n kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_n kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_n kinds C' body
  | Apply ap => call_ok_n kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_n (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_n
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_n (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_n id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_n (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_n
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

(* [calls_unboxing_reach_rem_nr] with [ind_call_ok_n]. *)
Definition calls_unboxing_reach_never_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_n F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr reach ovr rem [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow, the code that a call may over-apply and the code that what is
   left of an over-application may call, read off the facts [F1] and the
   tables [T1] of the first rebuilt unit. *)
Definition calls_unboxing_never_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (F1 : facts) (T1 : tables) (e : expr) : bool :=
  calls_unboxing_reach_never_nr nr F T cannot_change code_deps dominated
    to_unbox (flow_reach2 F1 T1) (over_reach F1 T1) (rem_reach F1 T1) e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_nr_ind_refines_cc_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_nr_ind_refines_observably_local_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_nr_ind_refines_observably_statement : Prop :=
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
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragment of FlambdaReaperUnboxingRefinesReach3Spec.v is in this
   one, for the same analysis of the first rebuilt unit. *)
Definition calls_unboxing_flow3_in_never_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_flow3_nr nr F T cannot_change code_deps dominated to_unbox
      F1 T1 e
    = true ->
    calls_unboxing_never_nr nr F T cannot_change code_deps dominated
      to_unbox F1 T1 e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* The codes 50 and 51 raise their parameter, so that no result of theirs
   has a source. The unit calls the closure [x60] of 50 or [x61] of 51,
   chosen by a [Switch]: at known arity in the handler 21, returning to 20,
   or at unknown arity in the handler 23, returning to 24. The tables of
   the unit and of the first rebuilt unit are the least solutions of the
   rules (FlambdaReaperLeast.compute_least); the code taken to never return
   is [used_never_returns]. *)
Module NeverIndExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call (callee : variable) (kind : call_kind) (args : list simple)
    (k : continuation) : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := kind; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

(* [x] raised to the exception continuation [exn] of the code. *)
Definition raise (exn : continuation) (x : variable) : expr :=
  Apply_cont {| ac_continuation := exn; ac_args := [Var x];
                ac_trap_action := Some (Pop exn None) |}.

(* The handler [k] of the parameter [x], running [handler], over [body]. *)
Definition handler (k : continuation) (x : variable) (handler : expr)
    (body : expr) : expr :=
  Let_cont_non_recursive k
    (Handler [x] (make_handler_metadata [Value]) handler) body.

(* The unit, with the handlers 20 and 24 [around] the rest, and the calls
   in 21 and 23 [call_21] and [call_23]. *)
Definition unit_with (around : expr -> expr) (call_21 call_23 : expr)
    : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata (raise 4 40));
     Bind_code 51 (Code [42] 43 5 6 metadata (raise 6 42))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (around
             (handler 21 31 call_21
                (handler 23 33 call_23
                   (Switch (Const (Naked_immediate 0))
                      [(0%Z, jump 21 [Var 60]); (1%Z, jump 21 [Var 61]);
                       (2%Z, jump 23 [Var 60]);
                       (3%Z, jump 23 [Var 61])])))))).

(* The call of known arity, and that of unknown arity, returning to [k]. *)
Definition known_call (ids : option (list code_id)) (k : continuation)
    : apply_expr :=
  call 31 (Indirect_known_arity ids) [int 5] k.

Definition unknown_call (k : continuation) : apply_expr :=
  call 33 Indirect_unknown_arity [int 6] k.

Definition unit_expr : expr :=
  unit_with
    (fun body => handler 20 30 (return_to 1 [Var 30])
                   (handler 24 34 (return_to 1 [Var 34]) body))
    (Apply (known_call None 20)) (Apply (unknown_call 24)).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition nr : code_id -> bool :=
  FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables tables)
    code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

(* The unit rebuilt with the change of calling conventions alone: the calls
   return where they did; the handlers 20 and 24 are kept, [Invalid], as
   their parameters have no source; the call of known arity lists the two
   codes. *)
Definition cc_rebuilt : expr :=
  unit_with
    (fun body => handler 20 30 Invalid (handler 24 34 Invalid body))
    (Apply (known_call (Some [51; 50]) 20)) (Apply (unknown_call 24)).

(* A call returning to the handler [Invalid] 62. *)
Definition invalid_call (ap : apply_expr) : expr :=
  handler 62 62 Invalid (Apply (with_continuation ap 62)).

(* The unit rebuilt in the mode of the fix: both calls return to a handler
   [Invalid], and the handlers 20 and 24 are dropped. *)
Definition rebuilt : expr :=
  unit_with (fun body => body)
    (invalid_call (known_call (Some [51; 50]) 20))
    (invalid_call (unknown_call 24)).

End NeverIndExample.

Definition never_ind_example_statement : Prop :=
  map
    (apply_never_returns NeverIndExample.facts
       (solution_of_tables NeverIndExample.tables) NeverIndExample.code_deps
       (bound_code_metadata NeverIndExample.unit_expr))
    [NeverIndExample.known_call None 20%positive;
     NeverIndExample.unknown_call 24%positive]
  = [true; true]
  ∧ map NeverIndExample.nr [50; 51]%positive = [true; true]
  ∧ calls_unboxing_flow3_nr NeverIndExample.nr NeverIndExample.facts
      NeverIndExample.tables NeverIndExample.cannot_change
      NeverIndExample.code_deps [] [] NeverIndExample.facts1
      NeverIndExample.tables1 NeverIndExample.unit_expr
    = false
  ∧ calls_unboxing_never_nr NeverIndExample.nr NeverIndExample.facts
      NeverIndExample.tables NeverIndExample.cannot_change
      NeverIndExample.code_deps [] [] NeverIndExample.facts1
      NeverIndExample.tables1 NeverIndExample.unit_expr
    = true
  ∧ NeverIndExample.first = NeverIndExample.cc_rebuilt
  ∧ rebuild_unit_unbox_fixed NeverIndExample.facts NeverIndExample.tables
      NeverIndExample.cannot_change NeverIndExample.code_deps [] [] true
      NeverIndExample.unit_expr
    = NeverIndExample.rebuilt
  ∧ program_refines_observably NeverIndExample.unit_expr
      (rebuild_unit_unbox_fixed NeverIndExample.facts NeverIndExample.tables
         NeverIndExample.cannot_change NeverIndExample.code_deps [] [] true
         NeverIndExample.unit_expr).
