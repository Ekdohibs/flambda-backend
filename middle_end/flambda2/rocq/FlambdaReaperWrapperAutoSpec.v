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


(* The premise [check_known_witnesses] of FlambdaReaperWrapperSpec.v,
   proved of the traversal with the flag [Auto], and the statements of
   that file with that flag that ask of the unit and of the solution only
   what the rebuild needs of the term, a least solution and the table of
   [to_unbox].

   This file is to be reviewed: the statements. The proofs are in
   FlambdaReaperWrapperAutoProofs.v, which needs no review.

   - [traverse_known_witnesses_statement]: a least solution of the facts
     of any unit, with the flag [Auto], passes [check_known_witnesses].
     The witness of known arity of a code dep is made by [prepare_code],
     and nothing but [create_known_arity_call_witness] puts anything
     into it: the results and the code id of its code, and its
     parameters. So no alias goes into it, and it is its own one source.
     The field of known arity of a closure holds the witness of a code
     dep, or, for code of which there is no code dep, a node with any
     source ([add_closure_dep]); with [Auto], the node that a call reads
     off its callee with that field gets a source only from that read
     ([traverse_widgets_read_only_statement]). The proof is an invariant
     of the accumulator along the traversal, of the nodes that facts put
     something into, as for [traverse_widgets_read_only_statement]; it
     holds with every flag.
   - [arm_needs_no_source_auto_end_to_end_statement] and
     [kept_result_never_unboxed_auto_end_to_end_statement]: the
     statements of FlambdaReaperWrapperShapeSpec.v with the flag [Auto],
     without [check_known_witnesses] and [check_apply_widgets], and with
     the tables of [cannot_change_calling_convention] and of
     [dominated_by_allocation_point] derived from the solution
     ([derive_cannot_change], [derive_dominated]), which pass
     [check_cc_closed] and [check_dominated]. What is left of the graph is
     a least solution and a table of [to_unbox] that passes
     [check_to_unbox_fixed].

   [check_unknown_widgets] and [check_unknown_witnesses], which
   [indirect_arm_needs_no_source_statement] asks for the calls of unknown
   arity, are proved in FlambdaReaperWrapperUnknownSpec.v, with the
   statement for those calls; the table of [to_unbox] derived from the
   solution passes [check_to_unbox_fixed]
   (FlambdaReaperWrapperDerivedSpec.v, with the statements for that
   table). *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec.
Import ListNotations.

(* ======================================================================== *)
(* The witnesses of known arity                                             *)
(* ======================================================================== *)

Definition traverse_known_witnesses_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_known_witnesses (traverse_unit Auto u e) T (unit_code_deps Auto u e)
    = true.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition arm_needs_no_source_auto_end_to_end_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting : bool) (to_unbox : list node),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let unboxed := unit_unboxed F T (derive_dominated T) to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
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

Definition kept_result_never_unboxed_auto_end_to_end_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting : bool) (to_unbox : list node),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let unboxed := unit_unboxed F T (derive_dominated T) to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
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
