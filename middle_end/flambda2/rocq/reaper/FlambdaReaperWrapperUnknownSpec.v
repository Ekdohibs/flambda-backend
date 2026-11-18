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


(* The premises [check_unknown_widgets] and [check_unknown_witnesses] of
   FlambdaReaperWrapperSpec.v, proved of the traversal, and the statement
   of that file about the calls of unknown arity with the flag [Auto]
   that asks of the unit and of the solution only what the rebuild needs
   of the term, a least solution and the table of [to_unbox].

   This file is to be reviewed: the statements. The proofs are in
   FlambdaReaperWrapperUnknownProofs.v, which needs no review.

   - [traverse_unknown_widgets_statement]: a least solution of the facts
     of any unit, with any flag, passes [check_unknown_widgets]. The
     nodes that a call of unknown arity reads with the field of unknown
     arity, off its callee and, for a call of several groups, off the
     node between two witnesses, are made by the call, and nothing but
     that read puts anything into them; so two of them read off one node
     have the sources of that field of the node.
   - [traverse_unknown_witnesses_statement]: a least solution of the facts
     of any unit, with any flag, passes [check_unknown_witnesses]. The
     witnesses of unknown arity of a code dep are made by [prepare_code],
     with a code id; nothing but constructors of them and parameters
     that they are the base of puts anything into them, so each is its
     own one source. The field of unknown arity of a node holds such a
     witness: of a closure ([add_closure_dep]), the first one of its
     code dep, or, for code of which there is no code dep, a node with
     any source; of the node between two witnesses, the next one. The
     proof of both is an invariant of the accumulator along the
     traversal, as in FlambdaReaperWrapperAutoProofs.v, of the nodes read
     with the field of unknown arity and of the witnesses of unknown
     arity of the code deps.
   - [indirect_arm_needs_no_source_auto_end_to_end_statement]: the
     statement [indirect_arm_needs_no_source_traversal_statement] of
     FlambdaReaperWrapperShapeSpec.v with the flag [Auto], without the
     checks of the shape of the traversal, and with the tables of
     [cannot_change_calling_convention] and of
     [dominated_by_allocation_point] derived from the solution, as in
     FlambdaReaperWrapperAutoSpec.v. What is left of the graph is a least
     solution and a table of [to_unbox] that passes
     [check_to_unbox_fixed]. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec.
Import ListNotations.

(* ======================================================================== *)
(* The checks of unknown arity                                              *)
(* ======================================================================== *)

Definition traverse_unknown_widgets_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_unknown_widgets (traverse_unit flag u e) T = true.

Definition traverse_unknown_witnesses_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_unknown_witnesses (traverse_unit flag u e) T = true.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition indirect_arm_needs_no_source_auto_end_to_end_statement : Prop :=
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
