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


(* The table of [to_unbox] of the fix derived from the solution passes
   [check_to_unbox_fixed], and the statements of
   FlambdaReaperWrapperAutoSpec.v and FlambdaReaperWrapperUnknownSpec.v
   with that table: they ask of the graph a least solution only.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaReaperWrapperDerivedProofs.v, which needs no review.

   - [derive_to_unbox_fixed_checked_statement]: for any tables, the table
     [derive_to_unbox_fixed] passes [check_to_unbox_fixed]. The tables
     of the unboxing that the model computes are closed under the rules:
     each relation but [cannot_unbox0] is one pass over the facts and the
     tuples, which gives what its rules conclude; the rounds of
     [cannot_unbox0] reach the fixpoint, as a round that adds something
     adds an allocation of the sources that was not there, and there are
     more rounds than sources. Neither a least solution nor the seeds of
     the fix matter: [derive_to_unbox_with] passes [check_to_unbox_with]
     for any seeds.
   - [arm_needs_no_source_auto_derived_statement],
     [kept_result_never_unboxed_auto_derived_statement] and
     [indirect_arm_needs_no_source_auto_derived_statement]: the statements
     [arm_needs_no_source_auto_end_to_end_statement],
     [kept_result_never_unboxed_auto_end_to_end_statement] and
     [indirect_arm_needs_no_source_auto_end_to_end_statement] with the
     table of [to_unbox] derived from the solution. What is left of the
     graph is a least solution. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec.
Import ListNotations.

(* ======================================================================== *)
(* The table of [to_unbox]                                                  *)
(* ======================================================================== *)

Definition derive_to_unbox_fixed_checked_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool),
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting
      (derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
         lambda_lifting)
    = true.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition arm_needs_no_source_auto_derived_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting : bool),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    let unboxed := unit_unboxed F T (derive_dominated T) to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (i : nat)
      (fd : decision) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap <> Indirect_unknown_arity ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false.

Definition kept_result_never_unboxed_auto_derived_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting : bool),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    let unboxed := unit_unboxed F T (derive_dominated T) to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (i : nat)
      (fields : unboxed_fields) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap <> Indirect_unknown_arity ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (Unbox fields, kind) ->
      arm_pair (apply_decision_at ads i kind) (Unbox fields) = false.

Definition indirect_arm_needs_no_source_auto_derived_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting : bool),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    let unboxed := unit_unboxed F T (derive_dominated T) to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (i : nat) (fd : decision)
      (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Indirect_unknown_arity ->
      length (ap_args_arity ap) <= 1 ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      result_has_source_at F T unknown (node_of_simple (ap_callee ap)) i
      = false.
