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


(* The premises of FlambdaReaperWrapperSpec.v that say what the shape of
   the traversal gives, proved of it where they hold of every unit, and
   the statements of that file with those premises dropped.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaReaperWrapperShapeProofs.v, which needs no review.

   - [traverse_used_code_ids_statement]: a least solution of the facts of
     any unit, with any flag, passes [check_used_code_ids]. The
     traversal puts a code id only into the field of the code id of a
     call witness, reads that field only into a node with any usage (the
     node [called] of a call), and never reads off a code id nor gives
     it an argument: the only alias from a code id is into such a node,
     so a use of a code id is any usage of it.
   - [traverse_argument_reads_statement]: the facts of any unit, with
     any flag, pass [check_argument_reads]: the traversal gives
     arguments only to the witness of a call, whose code id it reads
     into [called], which has any usage.
   - [arm_needs_no_source_traversal_statement],
     [arm_needs_no_source_auto_traversal_statement],
     [indirect_arm_needs_no_source_traversal_statement] and
     [kept_result_never_unboxed_traversal_statement]: the statements of
     FlambdaReaperWrapperSpec.v without those two premises.

   Not proved here: [check_known_witnesses], [check_unknown_widgets] and
   [check_unknown_witnesses]. Their facts hold of the traversal (a
   witness of code is made by [prepare_code] and is the base of no other
   constructor; nothing is aliased into it; a call reads off its callee
   only what the field of a closure holds, a witness of a code dep or,
   for code that the unit does not bind, a node with any source), but
   the proof needs an invariant of the made nodes for every flag, as
   [check_widgets_read_only] has for [Auto] (FlambdaReaperLeastProofs.v,
   [AutoWidgets]). [conts_counted] at toplevel and [plain_kinds] are
   checks of the term already: the first is the arity of each jump and
   each call, which [well_formed_unit] does not check and the compiler's
   terms have; the second is needed as the rebuild keeps a parameter of
   kind [Region] or [Rec_info] without a use. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The shape of the traversal                                               *)
(* ======================================================================== *)

Definition traverse_used_code_ids_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_used_code_ids T = true.

Definition traverse_argument_reads_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    check_argument_reads (traverse_unit flag u e) = true.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition arm_needs_no_source_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fd : decision) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false.

Definition arm_needs_no_source_auto_traversal_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox : list node),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fd : decision) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false.

Definition indirect_arm_needs_no_source_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_known_witnesses F T code_deps = true ->
    check_unknown_widgets F T = true ->
    check_unknown_witnesses F T = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (i : nat) (fd : decision)
      (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Indirect -> length (ap_args_arity ap) <= 1 ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      result_has_source_at F T unknown (node_of_simple (ap_callee ap)) i
      = false.

Definition kept_result_never_unboxed_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fields : unboxed_fields) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (Unbox fields, kind) ->
      arm_pair (apply_decision_at ads i kind) (Unbox fields) = false.
