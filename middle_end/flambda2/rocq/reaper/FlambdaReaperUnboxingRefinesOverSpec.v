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

(* The rebuild with unboxing, in the mode of the fix, refines its unit
   across calls, with a weaker check of the code that a call whose code
   is not known may call: the plain result that an over-application
   calls is asked only of code that a call that may over-apply may
   call.

   This file is to be reviewed: the fragment ([calls_unboxing_over_nr],
   which is [FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr] but
   for [ind_code_ok_o]), the statements and the example. The proofs are
   in FlambdaReaperUnboxingRefinesOverProofs.v, which needs no review.

   [FlambdaReaperUnboxingRefinesCallsSpec.ind_code_ok] asks of every code
   of the unit, for every call whose code is not known, that a single
   result of kind [Value] have a use, because an over-application calls
   it. A call of known arity ([Indirect_known_arity]) is never an
   over-application: the machine finds it undefined unless the code of
   its callee has as many parameters as it has groups of arguments
   (FlambdaSemanticsSpec.known_arity_full), and its callee is never a
   partial application. So [ind_code_ok_o] asks it only of a call of
   another kind: of unknown arity, or direct.

   [rebuild_unbox_over_refines_cc_statement],
   [rebuild_unbox_over_refines_observably_statement] and
   [rebuild_unbox_over_refines_observably_local_statement]: the statements
   of FlambdaReaperUnboxingRefinesCallsSpec.v with code [nr] that never
   returns ([rebuild_unbox_fixed_refines_cc_never_returns_statement],
   [rebuild_unbox_fixed_observably_never_returns_statement],
   [rebuild_unbox_fixed_local_never_returns_statement]), for the fragment
   [calls_unboxing_over_nr], which every unit of [calls_unboxing_nr] is
   of.

   [known_arity_example_statement]: a unit whose code 52 returns a result
   with no use, called directly, and which calls the code 50 or 51,
   chosen by a [Switch], by a call of known arity whose result has no
   use, with the tables of the least solution: it is not of
   [calls_unboxing] (the code 52 does not pass [ind_code_ok]), it is of
   [calls_unboxing_over], and the rebuilt unit refines the unit up to
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
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* A call of known arity, which is never an over-application. *)
Definition known_arity_call (kind : call_kind) : bool :=
  match kind with
  | Indirect_known_arity _ => true
  | Direct _ | Indirect_unknown_arity => false
  end.

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

(* [ind_code_ok] for a call of the kind [kind]: a single result of kind
   [Value] has a use unless the call is of known arity. *)
Definition ind_code_ok_o (kind : call_kind) (kps : list pdesc) (RA : arity)
    (id : code_id) (md : code_metadata) : bool :=
  match
    code_change_u sol cannot_change code_deps unit_code_ids unboxed id md,
    code_pdescs id md, code_exn_decision id
  with
  | Not_changing_u, Some (_, rds), Some _ =>
    Nat.eqb (length rds) (length (unarize (cm_result_arity md)))
    && (negb (arity_eqb (cm_result_arity md) RA)
        || forallb2 result_ok kps rds)
    && (known_arity_call kind
        || negb (arity_eqb (cm_result_arity md) [Singleton Value])
        || forallb (fun r => is_kept_used (pd_decision r)) rds)
  | _, _, _ => false
  end.

(* [ind_call_ok] with [ind_code_ok_o]. *)
Definition ind_call_ok_o (kinds : finite_map flambda_kind)
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
               ind_code_ok_o (ap_call_kind ap) kps (ap_return_arity ap) id
                 md)
            codes
     | _ => false
     end.

(* [call_ok], [calls_expr], [calls_handler], [calls_binding] and
   [calls_code] of FlambdaReaperUnboxingRefinesCallsSpec.v, with
   [ind_call_ok_o]. *)
Definition call_ok_o (kinds : finite_map flambda_kind)
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
  | Var f, None => ind_call_ok_o kinds C ap f
  | _, _ => false
  end.

Fixpoint calls_expr_o (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_expr_o ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_binding_o bindings && calls_expr_o kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_handler_o kinds C ps h
    && calls_expr_o kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) => calls_handler_o kinds C' (handler_pdescs false h) h)
      handlers
    && calls_expr_o kinds C' body
  | Apply ap => call_ok_o kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_handler_o (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_expr_o (rev (combine params (hm_params_kinds metadata)) ++ kinds) C
         handler
  end
with calls_binding_o (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => calls_code_o id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_code_o (id : code_id) (c : code) {struct c} : bool :=
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
              calls_expr_o
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

(* [calls_unboxing_nr] with [ind_code_ok_o]. *)
Definition calls_unboxing_over_nr (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_expr_o F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    nr [] [] e
  && check_unboxed_uses F T unboxed.

Definition calls_unboxing_over (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : bool :=
  calls_unboxing_over_nr (fun _ => false) F T cannot_change code_deps
    dominated to_unbox e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_over_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing_over_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_over_refines_observably_statement : Prop :=
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
    calls_unboxing_over_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_over_refines_observably_local_statement : Prop :=
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
    calls_unboxing_over_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The fragment of FlambdaReaperUnboxingRefinesCallsSpec.v is in this
   one. *)
Definition calls_unboxing_in_over_statement : Prop :=
  ∀ (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr),
    calls_unboxing_nr nr F T cannot_change code_deps dominated to_unbox e
    = true ->
    calls_unboxing_over_nr nr F T cannot_change code_deps dominated to_unbox e
    = true.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* A call of known arity whose code is not known, of a unit whose code
   returns a result with no use. The codes 50 and 51 return their
   parameter, and 52 the constant 0; all three keep their calling
   convention (they are in the table). The unit calls 52 directly and
   drops its result, then calls 50 or 51, the closure chosen by a
   [Switch], at known arity, and drops that result too. No result of any
   code has a use, so [ind_code_ok] refuses each code for the call of
   known arity. The tables are the least solution of the rules
   (FlambdaReaperLeast.compute_least); nothing is unboxed. *)
Module KnownArityExample.

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

(* The unit, the results of 50 and 51 being [r50] and [r51], the
   parameters of the handlers 20 and 22 [ps20] and [ps22], the call in 21
   [call_21] and that of 52 [call_52]. *)
Definition unit_with (r50 r51 : simple) (ps20 ps22 : list variable)
    (call_21 call_52 : expr) : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata (return_to 3 [r50]));
     Bind_code 51 (Code [42] 43 5 6 metadata (return_to 5 [r51]));
     Bind_code 52 (Code [44] 45 7 8 metadata (return_to 7 [int 0]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (Let 62 Value (Closure_of_set 72 [(72, 52)] [])
             (Let_cont_non_recursive 20
                (Handler ps20
                   (make_handler_metadata (map (fun _ => Value) ps20))
                   (return_to 1 [int 1]))
                (Let_cont_non_recursive 21
                   (Handler [31] (make_handler_metadata [Value]) call_21)
                   (Let_cont_non_recursive 22
                      (Handler ps22
                         (make_handler_metadata (map (fun _ => Value) ps22))
                         (Switch (Const (Naked_immediate 0))
                            [(0%Z, jump 21 [Var 60]);
                             (1%Z, jump 21 [Var 61])]))
                      call_52)))))).

Definition unit_expr : expr :=
  unit_with (Var 40) (Var 42) [30] [32]
    (Apply (call 31 (Indirect_known_arity None) [int 5] 20))
    (Apply (call 62 (Direct 52) [int 6] 22)).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := [c 50; c 51; c 52].

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* A call returning to a wrapper [w] that drops its result for [k]. *)
Definition dropped (w k : continuation) (ap : apply_expr) : expr :=
  Let_cont_non_recursive w
    (Handler [w] (make_handler_metadata [Value])
       (Apply_cont (jump k [])))
    (Apply ap).

(* The rebuilt unit: 50 and 51 return poison, the handlers 20 and 22 lose
   their parameter, both calls return to a wrapper that drops the result,
   and the call of known arity lists the two codes. *)
Definition rebuilt : expr :=
  unit_with (Const (Poison Value)) (Const (Poison Value)) [] []
    (dropped 63 20
       (call 31 (Indirect_known_arity (Some [51; 50])) [int 5] 63))
    (dropped 63 22 (call 62 (Direct 52) [int 6] 63)).

End KnownArityExample.

Definition known_arity_example_statement : Prop :=
  calls_unboxing KnownArityExample.facts KnownArityExample.tables
    KnownArityExample.cannot_change KnownArityExample.code_deps [] []
    KnownArityExample.unit_expr
  = false
  ∧ calls_unboxing_over KnownArityExample.facts KnownArityExample.tables
      KnownArityExample.cannot_change KnownArityExample.code_deps [] []
      KnownArityExample.unit_expr
    = true
  ∧ check_unboxed_reads KnownArityExample.facts KnownArityExample.tables [] []
    = true
  ∧ rebuild_unit_unbox_fixed KnownArityExample.facts KnownArityExample.tables
      KnownArityExample.cannot_change KnownArityExample.code_deps [] [] true
      KnownArityExample.unit_expr
    = KnownArityExample.rebuilt
  ∧ program_refines_observably KnownArityExample.unit_expr
      KnownArityExample.rebuilt.
