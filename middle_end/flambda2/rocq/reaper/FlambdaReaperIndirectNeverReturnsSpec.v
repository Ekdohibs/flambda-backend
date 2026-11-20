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

(* Calls whose code is not known that the fix makes never return
   ([apply_never_returns], FlambdaReaperUnboxing.v), in the runs of the
   unit: what the run of the unit gives of the code they enter.

   This file is to be reviewed: the statement. The proofs are in
   FlambdaReaperIndirectNeverReturnsProofs.v, which needs no review.

   The refinement of FlambdaReaperUnboxingRefinesCallsSpec.v asks such
   calls, of code that is not known ([ind_call_ok]), to have
   [apply_never_returns] false. The rebuild with unboxing makes such a
   call return to a handler [Invalid]; the relation of
   FlambdaReaperUnboxingRefinesCallsSimProofs.v skips the frame of a call
   only where the code entered is of [nr], whose return continuation is
   not live, and the code entered by a call whose code is not known is
   known only from the values of the run. Of the run of the unit, it is
   known here.

   [entered_used_never_returns_statement]: for a unit of the flow
   fragment, tables closed under the rules on its graph, any flag, and a
   state of its run at a call of known arity that [apply_never_returns]
   makes one that never returns, whether its code is known or not, and
   that steps: the code of the closure called is used and has a result
   with no source ([used_never_returns]), so it is of the code that the
   derived statements of FlambdaReaperNeverReturnsSpec.v take to never
   return. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperCallingConventions
  FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperNeverReturnsSpec.
Import ListNotations.

Definition entered_used_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s s1 : state)
    (ap : apply_expr) (callee : value) (id : code_id)
    (slots : list (value_slot * value)),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_expr s = Apply ap ->
    ap_call_kind ap <> Indirect_unknown_arity ->
    eval_simple (state_vars s) (ap_callee ap) = Some callee ->
    closure_of (state_statics s) callee = Some (id, slots) ->
    apply_never_returns (traverse_unit flag u e) (solution_of_tables T)
      (unit_code_deps flag u e) (bound_code_metadata e) ap
    = true ->
    step s = Step s1 ->
    FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
      (unit_code_deps flag u e) id
    = true.
