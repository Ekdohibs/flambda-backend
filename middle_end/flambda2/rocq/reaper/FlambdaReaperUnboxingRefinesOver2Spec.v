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
(* The fragment of FlambdaReaperUnboxingRefinesReach2Spec.v, widened to
   calls of unknown arity with two groups of arguments or more, which may
   over-apply: such a call asks a single result of kind [Value] with a use
   only of the code that a call may over-apply, as the analysis of the
   first rebuilt unit says.

   This file is to be reviewed: the code that a call may over-apply
   ([over_reach]), the fragment ([calls_unboxing_reach_over_nr], which is
   [calls_unboxing_reach2_nr] but for [ind_call_ok_o2]; and
   [calls_unboxing_over2_nr], where every code may be reached), the
   statements and the example. The proofs are in
   FlambdaReaperUnboxingRefinesOver2Proofs.v, which needs no review.

   [ind_code_ok_o1] asks of every code, for a call that may over-apply
   ([never_over] fails), that a single result of kind [Value] have a use,
   because what is left of the over-application calls it. The traversal
   gives that use, for the code that the call may reach: a call of unknown
   arity reads a witness for each group of its arguments, the first
   result of each but the last into a variable from which it reads the
   next witness, and the code that the variable is given to as a witness
   has any usage (traverse_acc.ml, [make_unknown_arity_apply_widget]). Where
   the call over-applies code [A], the witness of the call for the last
   group of [A] stands for the last witness of unknown arity of [A], whose
   first result is that of [A]; the rules then alias the variable of the
   call to that result ([alias_constructor_usage]), which so has a use.
   Arity alone does not say which code a call may over-apply: a call of a
   partial application of [A], [k] groups given, over-applies [A] if [k]
   and the groups of the call are more than the parameters of [A], with
   as many groups as [A] has parameters or more.

   The use is that of the analysis of the unit whose run the proofs follow,
   the first rebuilt unit, analysed again (facts [F1], tables [T1]), as in
   FlambdaReaperUnboxingRefinesReachSpec.v; the decisions of the rebuild
   with unboxing are those of the tables [T] of the unit. [over_reach F1 T1
   id]: the code [id] has a witness, of known arity or not, whose first
   result has a use in [T1]; the code that a call over-applies is of it.
   [ind_call_ok_o2] is [ind_call_ok_f2], but a call that may over-apply asks
   [ind_code_ok_o1] with the use of a single result only of the code for
   which [ovr] holds. [calls_unboxing_over2_nr] takes every code to be
   reached and [over_reach F1 T1] for [ovr].

   [rebuild_unbox_over2_refines_cc_statement],
   [rebuild_unbox_over2_refines_observably_statement] and
   [rebuild_unbox_over2_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesReach2Spec.v for the fragment
   [calls_unboxing_over2_nr], with the same premises.

   [calls_unboxing_over1_in_over2_statement]: every unit of
   [calls_unboxing_over1_nr] is of [calls_unboxing_over2_nr], whatever the
   analysis of the first rebuilt unit.

   [two_groups_example_statement]: a unit whose code 50 returns a closure
   of 51, and which calls a closure of 50 by a call of unknown arity with
   two groups of arguments, which over-applies 50 and calls 51 with the
   second, and then calls 52 directly; the results of 51 and 52 have no
   use. With the least solution, [calls_unboxing_over1] and
   [calls_unboxing_flow2] refuse it (the call asks a used result of 51 and
   52), [calls_unboxing_over2] accepts it (only 50 is [over_reach]), the
   rebuilt unit, where 51 and 52 return poison, is the one written out, and
   it refines the unit up to observation. *)

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
  FlambdaReaperUnboxingRefinesReach2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The code that a call may over-apply                                      *)
(* ======================================================================== *)

(* The code [id] has a witness, of known arity or not, whose first result
   has a use. *)
Definition over_reach (F1 : facts) (T1 : tables) (id : code_id) : bool :=
  existsb
    (fun '(w, relation, c) =>
       field_eqb relation Field_code_id_of_call_witness
       && node_eqb c (Node_code_id id)
       && existsb
            (fun '(w', relation', r) =>
               node_eqb w' w
               && field_eqb relation' (Field_return_of_call (Normal 0))
               && mem_node r (t_has_usage T1))
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

Local Abbreviation never_over :=
  (FlambdaReaperUnboxingRefinesOver1Spec.never_over codes).
Local Abbreviation ind_code_ok_o1 :=
  (FlambdaReaperUnboxingRefinesOver1Spec.ind_code_ok_o1 sol cannot_change
     code_deps unit_code_ids unboxed).

(* [ind_call_ok_f2], asking a single result of kind [Value] with a use
   only of the code that a call may over-apply ([ovr]). *)
Definition ind_call_ok_o2 (kinds : finite_map flambda_kind)
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
               || ind_code_ok_o1 (never_over ap || negb (ovr id)) kps
                    (ap_return_arity ap) id md)
            codes
     | _ => false
     end.

(* [call_ok_f2], [calls_expr_f2], [calls_handler_f2], [calls_binding_f2]
   and [calls_code_f2] of FlambdaReaperUnboxingRefinesReach2Spec.v, with
   [ind_call_ok_o2]. *)
Definition call_ok_o2 (kinds : finite_map flambda_kind)
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
  | Var f, None => ind_call_ok_o2 kinds C ap f
  | _, _ => false
  end.

Fixpoint calls_expr_o2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_o2 ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_o2 bindings && calls_expr_o2 kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_o2 kinds C ps h
    && calls_expr_o2 kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_o2 kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_o2 kinds C' body
  | Apply ap => call_ok_o2 kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_o2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_o2
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_o2 (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_o2 id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_o2 (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_o2
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

(* [calls_unboxing_reach2_nr] with [ind_call_ok_o2]. *)
Definition calls_unboxing_reach_over_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr : code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_o2 F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr reach ovr [] [] e
  && check_unboxed_uses F T unboxed.

(* Every code reached, and the code that a call may over-apply read off the
   facts [F1] and the tables [T1] of the first rebuilt unit. *)
Definition calls_unboxing_over2_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (F1 : facts) (T1 : tables) (e : expr) : bool :=
  calls_unboxing_reach_over_nr nr F T cannot_change code_deps dominated
    to_unbox (fun _ _ _ => true) (over_reach F1 T1) e.

Definition calls_unboxing_over2 (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) : bool :=
  calls_unboxing_over2_nr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox F1 T1 e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_over2_refines_cc_statement : Prop :=
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
    calls_unboxing_over2_nr nr (traverse_unit flag u e) T cannot_change
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

Definition rebuild_unbox_over2_refines_observably_local_statement : Prop :=
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
    calls_unboxing_over2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_over2_refines_observably_statement : Prop :=
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
    calls_unboxing_over2_nr nr (traverse_unit flag u e) T cannot_change
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
Definition calls_unboxing_over1_in_over2_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr),
    calls_unboxing_over1_nr nr F T cannot_change code_deps dominated
      to_unbox e
    = true ->
    calls_unboxing_over2_nr nr F T cannot_change code_deps dominated
      to_unbox F1 T1 e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* A unit whose code 50 makes a closure of 51 and returns it, and which
   calls a closure of 50 with two groups of arguments, by a call of unknown
   arity that returns to the handler 21; 21 calls 52 directly, returning
   to 20, which leaves the unit. The call over-applies 50, whose result
   51 is called with the second group. The parameters of 20 and 21 have no
   use, so neither have the results of 51 and 52; the three codes keep
   their calling convention. The tables of the unit and of the first
   rebuilt unit are the least solutions of the rules
   (FlambdaReaperLeast.compute_least); nothing is unboxed. *)
Module TwoGroupsExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

(* A call of unknown arity of [callee] with two groups of one argument. *)
Definition call2 (callee : variable) (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Indirect_unknown_arity;
     ap_args := args; ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

(* The unit, the results of 51 and 52 being [r51] and [r52], the
   parameters of the handlers 20 and 21 [ps20] and [ps21], the call in 21
   [call_21] and the call of two groups [call_2]. *)
Definition unit_with (r51 r52 : simple) (ps20 ps21 : list variable)
    (call_21 call_2 : expr) : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4 KnownArityExample.metadata
          (Let 46 Value (Closure_of_set 71 [(71, 51)] [])
             (return_to 3 [Var 46])));
     Bind_code 51 (Code [42] 43 5 6 KnownArityExample.metadata
                     (return_to 5 [r51]));
     Bind_code 52 (Code [44] 45 7 8 KnownArityExample.metadata
                     (return_to 7 [r52]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 62 Value (Closure_of_set 72 [(72, 52)] [])
          (Let_cont_non_recursive 20
             (Handler ps20
                (make_handler_metadata (map (fun _ => Value) ps20))
                (return_to 1 [int 1]))
             (Let_cont_non_recursive 21
                (Handler ps21
                   (make_handler_metadata (map (fun _ => Value) ps21))
                   call_21)
                call_2)))).

Definition unit_expr : expr :=
  unit_with (Var 42) (int 0) [30] [31]
    (Apply (KnownArityExample.call 62 (Direct 52) [int 6] 20))
    (Apply (call2 60 [int 5; int 7] 21)).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := [c 50; c 51; c 52].

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

(* The rebuilt unit: 51 returns poison, the handlers 20 and 21 lose their
   parameter, and both calls return to a wrapper that drops the result;
   the call with two groups stays as it is. *)
Definition rebuilt : expr :=
  unit_with (Const (Poison Value)) (int 0) [] []
    (KnownArityExample.dropped 63 20
       (KnownArityExample.call 62 (Direct 52) [int 6] 63))
    (KnownArityExample.dropped 63 21 (call2 60 [int 5; int 7] 63)).

End TwoGroupsExample.

Definition two_groups_example_statement : Prop :=
  calls_unboxing_over1 TwoGroupsExample.facts TwoGroupsExample.tables
    TwoGroupsExample.cannot_change TwoGroupsExample.code_deps [] []
    TwoGroupsExample.unit_expr
  = false
  ∧ calls_unboxing_flow2 TwoGroupsExample.facts TwoGroupsExample.tables
      TwoGroupsExample.cannot_change TwoGroupsExample.code_deps [] []
      TwoGroupsExample.facts1 TwoGroupsExample.tables1
      TwoGroupsExample.unit_expr
    = false
  ∧ map (over_reach TwoGroupsExample.facts1 TwoGroupsExample.tables1)
      [50; 51; 52]%positive
    = [true; false; false]
  ∧ calls_unboxing_over2 TwoGroupsExample.facts TwoGroupsExample.tables
      TwoGroupsExample.cannot_change TwoGroupsExample.code_deps [] []
      TwoGroupsExample.facts1 TwoGroupsExample.tables1
      TwoGroupsExample.unit_expr
    = true
  ∧ rebuild_unit_unbox_fixed TwoGroupsExample.facts TwoGroupsExample.tables
      TwoGroupsExample.cannot_change TwoGroupsExample.code_deps [] [] true
      TwoGroupsExample.unit_expr
    = TwoGroupsExample.rebuilt
  ∧ program_refines_observably TwoGroupsExample.unit_expr
      TwoGroupsExample.rebuilt.
