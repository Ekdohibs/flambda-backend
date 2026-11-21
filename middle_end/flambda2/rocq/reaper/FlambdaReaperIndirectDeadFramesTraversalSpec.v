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

(* The check of FlambdaReaperIndirectDeadFramesSpec.v holds of the graph
   of the traversal, so that the frame of a call of unknown arity that the
   fix makes one that never returns needs it not.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaReaperIndirectDeadFramesTraversalProofs.v, which needs no review.

   [traverse_unknown_chains_statement]: the traversal of every unit, with
   every flag, passes [check_unknown_chains]. The nodes that a call of
   unknown arity reads with that field (the read off the callee, the
   witnesses after the first), the first witness and the reads of the
   first result between two witnesses are made by the call, and only its
   facts put anything into them; the delayed facts give that field only
   to constructors.

   [indirect_dead_frame_traversal_statement]:
   FlambdaReaperIndirectDeadFramesSpec.indirect_dead_frame_statement
   without the premise [check_unknown_chains]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaReaperLeastSpec FlambdaReaperDeadFramesSpec
  FlambdaReaperIndirectDeadFramesSpec.
Import ListNotations.

Definition traverse_unknown_chains_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    check_unknown_chains (traverse_unit flag u e) = true.

Definition indirect_dead_frame_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state)
    (ap : apply_expr) (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_expr s = Apply ap ->
    ap_call_kind ap = Indirect_unknown_arity ->
    ap_exn_continuation ap <> ap_continuation ap ->
    apply_never_returns (traverse_unit flag u e) (solution_of_tables T)
      (unit_code_deps flag u e) (bound_code_metadata e) ap
    = true ->
    resolve (ap_continuation ap) (state_conts s)
    = Closure h vars conts recursive ->
    ∃ i : nat,
      i < length (unarize (ap_return_arity ap))
      ∧ ∀ (x : variable) (m : nat) (s' : state),
          s_has_source (solution_of_tables T) (Node_variable x) = false ->
          run m
            (State (Apply ap) (state_vars s)
               ((ap_continuation ap,
                 Closure (rename_param i x h) vars conts recursive)
                :: state_conts s)
               (state_statics s) (state_heap s) (state_reads s))
          = Step s' ->
          lookup x (state_vars s') = None.
