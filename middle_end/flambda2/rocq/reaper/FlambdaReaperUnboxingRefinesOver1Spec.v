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

(* The fragment of FlambdaReaperUnboxingRefinesOverSpec.v, widened to
   calls with one group of arguments ([f x], as [List.iter] makes) in a
   unit whose code has parameters.

   This file is to be reviewed: the fragment ([calls_unboxing_over1_nr],
   which is [calls_unboxing_over_nr] but for [ind_code_ok_o1]), the
   statements and the example. The proofs are in
   FlambdaReaperUnboxingRefinesOver1Proofs.v, which needs no review.

   [ind_code_ok_o] asks of every code of the unit, for a call whose code
   is not known and that is not of known arity, that a single result of
   kind [Value] have a use, because an over-application calls it. A call
   with one group of arguments is no over-application either, where
   every code has a parameter: a closure is then given at most as many
   groups as its code has parameters, and a partial application that the
   machine makes holds fewer groups than its code has parameters
   (FlambdaReaperPartialGroupsProofs.one_group_no_over). In a unit of
   [FlambdaReaperFlowSpec.in_flow_fragment] with a call of unknown arity,
   every code that is not tupled has a parameter. So [ind_code_ok_o1]
   asks the use only of a call for which [never_over] fails.

   [rebuild_unbox_over1_refines_cc_statement],
   [rebuild_unbox_over1_refines_observably_statement] and
   [rebuild_unbox_over1_refines_observably_local_statement]: the
   statements of FlambdaReaperUnboxingRefinesOverSpec.v for the fragment
   [calls_unboxing_over1_nr].

   [calls_unboxing_over_in_over1_statement]: every unit of
   [calls_unboxing_over_nr] is of [calls_unboxing_over1_nr].

   [one_group_example_statement]: a unit whose code 52 returns a result
   with no use, called directly, and which calls the code 50 or 51,
   chosen by a [Switch], by a call of unknown arity with one group of
   arguments whose result has no use, with the tables of the least
   solution: it is not of [calls_unboxing_over] (the codes 50 and 51 do
   not pass [ind_code_ok_o]), it is of [calls_unboxing_over1], and the
   rebuilt unit, where 50 and 51 return poison, refines the unit up to
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
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesOverSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
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

(* Every code of the unit has a parameter. *)
Definition codes_have_params : bool :=
  forallb (fun '(_, md) => negb (Nat.eqb (length (cm_params_arity md)) 0))
    codes.

(* A call that over-applies nothing: of known arity, or with one group of
   arguments in a unit whose code has parameters. *)
Definition never_over (ap : apply_expr) : bool :=
  known_arity_call (ap_call_kind ap)
  || (Nat.eqb (length (ap_args_arity ap)) 1 && codes_have_params).

(* [ind_code_ok] for a call that over-applies nothing if [no_over]: a
   single result of kind [Value] has a use unless [no_over]. *)
Definition ind_code_ok_o1 (no_over : bool) (kps : list pdesc) (RA : arity)
    (id : code_id) (md : code_metadata) : bool :=
  match
    code_change_u sol cannot_change code_deps unit_code_ids unboxed id md,
    code_pdescs id md, code_exn_decision id
  with
  | Not_changing_u, Some (_, rds), Some _ =>
    Nat.eqb (length rds) (length (unarize (cm_result_arity md)))
    && (negb (arity_eqb (cm_result_arity md) RA)
        || forallb2 result_ok kps rds)
    && (no_over
        || negb (arity_eqb (cm_result_arity md) [Singleton Value])
        || forallb (fun r => is_kept_used (pd_decision r)) rds)
  | _, _, _ => false
  end.

(* [ind_call_ok] with [ind_code_ok_o1]. *)
Definition ind_call_ok_o1 (kinds : finite_map flambda_kind)
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
               ind_code_ok_o1 (never_over ap) kps (ap_return_arity ap) id
                 md)
            codes
     | _ => false
     end.

(* [call_ok_o], [calls_expr_o], [calls_handler_o], [calls_binding_o] and
   [calls_code_o] of FlambdaReaperUnboxingRefinesOverSpec.v, with
   [ind_call_ok_o1]. *)
Definition call_ok_o1 (kinds : finite_map flambda_kind)
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
  | Var f, None => ind_call_ok_o1 kinds C ap f
  | _, _ => false
  end.

Fixpoint calls_expr_o1 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_o1 ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_o1 bindings && calls_expr_o1 kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_o1 kinds C ps h
    && calls_expr_o1 kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_o1 kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_o1 kinds C' body
  | Apply ap => call_ok_o1 kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_o1 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_o1
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_binding_o1 (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_o1 id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_o1 (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_o1
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

(* [calls_unboxing_over_nr] with [ind_code_ok_o1]. *)
Definition calls_unboxing_over1_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_o1 F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr [] [] e
  && check_unboxed_uses F T unboxed.

Definition calls_unboxing_over1 (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : bool :=
  calls_unboxing_over1_nr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_over1_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing_over1_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_over1_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    calls_unboxing_over1_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_over1_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    calls_unboxing_over1_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragment of FlambdaReaperUnboxingRefinesOverSpec.v is in this
   one. *)
Definition calls_unboxing_over_in_over1_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr),
    calls_unboxing_over_nr nr F T cannot_change code_deps dominated to_unbox
      e
    = true ->
    calls_unboxing_over1_nr nr F T cannot_change code_deps dominated
      to_unbox e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* [KnownArityExample] of FlambdaReaperUnboxingRefinesOverSpec.v, its
   call of known arity made of unknown arity: [f 5], [f] the closure of
   50 or of 51, chosen by a [Switch]. The three codes keep their calling
   convention and have one parameter; no result of any code has a use,
   so [ind_code_ok_o] refuses 50 and 51 for that call. The tables are the
   least solution of the rules (FlambdaReaperLeast.compute_least); nothing
   is unboxed. *)
Module OneGroupExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  KnownArityExample.unit_with (Var 40) (Var 42) [30] [32]
    (Apply (KnownArityExample.call 31 Indirect_unknown_arity [int 5] 20))
    (Apply (KnownArityExample.call 62 (Direct 52) [int 6] 22)).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := [c 50; c 51; c 52].

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The rebuilt unit: that of [KnownArityExample], its call of unknown
   arity kept so. *)
Definition rebuilt : expr :=
  KnownArityExample.unit_with (Const (Poison Value)) (Const (Poison Value))
    [] []
    (KnownArityExample.dropped 63 20
       (KnownArityExample.call 31 Indirect_unknown_arity [int 5] 63))
    (KnownArityExample.dropped 63 22
       (KnownArityExample.call 62 (Direct 52) [int 6] 63)).

End OneGroupExample.

Definition one_group_example_statement : Prop :=
  calls_unboxing_over OneGroupExample.facts OneGroupExample.tables
    OneGroupExample.cannot_change OneGroupExample.code_deps [] []
    OneGroupExample.unit_expr
  = false
  ∧ calls_unboxing_over1 OneGroupExample.facts OneGroupExample.tables
      OneGroupExample.cannot_change OneGroupExample.code_deps [] []
      OneGroupExample.unit_expr
    = true
  ∧ check_unboxed_reads OneGroupExample.facts OneGroupExample.tables [] []
    = true
  ∧ rebuild_unit_unbox_fixed OneGroupExample.facts OneGroupExample.tables
      OneGroupExample.cannot_change OneGroupExample.code_deps [] [] true
      OneGroupExample.unit_expr
    = OneGroupExample.rebuilt
  ∧ program_refines_observably OneGroupExample.unit_expr
      OneGroupExample.rebuilt.
