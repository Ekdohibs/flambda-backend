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

(* The frame of a call of code that never returns is never returned to, in
   the runs of the unit: part of what makes the fix proposed for the reaper
   sound where it makes such a call one that never returns
   (FlambdaReaperUnboxing.v, [apply_never_returns], [code_never_returns]).

   This file is to be reviewed: [rename_nth], [rename_param] and the
   statements. The proofs are in FlambdaReaperDeadFramesProofs.v, which
   needs no review.

   A frame of the machine is the continuation that a call gives its callee
   as its return continuation; the machine has no stack, and that value is
   not marked, so the statement marks it. [dead_frame_statement]: take a
   unit [e] of the flow fragment, tables closed under the rules on its
   graph (with no local field, or no projection), and a state [s] of its run
   at a call [ap] of known arity whose callee is a closure of the code
   [id], whose code dep has a result [r] of rank [i] with no source (as
   [code_never_returns] has it), and whose continuation is a handler [h]
   (the closure of [h] with [vars], [conts] and [recursive]). Rebind the
   continuation of the call in [s] to the closure of [h] whose parameter of
   rank [i] is renamed [x], a variable of no source: the run from there
   never binds [x]. The renamed handler is applied only where the frame of
   the call is returned to with a value for each of its parameters, and
   [x] would then be bound; so the frame is never returned to. The flag
   and the tables are any (the statement holds with [Auto] and a least
   solution in particular). The exception continuation of the call is
   another continuation.

   The call of known arity enters the code whose closure it calls
   (FlambdaSemanticsSpec.v), so [id] is the code that runs; a call of
   unknown arity, which may make a partial application or an
   over-application, is not covered.

   [dead_frame_of_call_statement]: the premise on the code holds of a call
   of known arity that [apply_never_returns] makes one that never returns
   and that is entered (it steps): the code entered has a result with no
   source, the one of the code the call is known to call
   ([code_id_actually_called]) or the one read off the call witness of
   its callee. [never_returns_frame_statement] is the two together: such
   a call returning to a handler, there is a rank [i] such that the
   handler with its parameter [i] renamed is never applied. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperFlowSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing.
Import ListNotations.

(* The list [l] with its element of rank [i] replaced by [x]; [l] if it has
   none. *)
Fixpoint rename_nth (i : nat) (x : variable) (l : list variable)
    : list variable :=
  match l, i with
  | [], _ => []
  | _ :: l, O => x :: l
  | y :: l, S i => y :: rename_nth i x l
  end.

(* The handler [h] with its parameter of rank [i] renamed [x]. *)
Definition rename_param (i : nat) (x : variable) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler (rename_nth i x params) metadata handler
  end.

Definition dead_frame_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n m : nat) (s s' : state)
    (ap : apply_expr) (callee : value) (id : code_id)
    (slots : list (value_slot * value)) (cd : code_dep) (i : nat) (r : node)
    (h : continuation_handler) (vars : env) (conts : cont_env)
    (recursive : list (continuation * continuation_handler)) (x : variable),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_expr s = Apply ap ->
    ap_call_kind ap <> Indirect_unknown_arity ->
    ap_exn_continuation ap <> ap_continuation ap ->
    eval_simple (state_vars s) (ap_callee ap) = Some callee ->
    closure_of (state_statics s) callee = Some (id, slots) ->
    lookup id (unit_code_deps flag u e) = Some cd ->
    nth_error (cd_return cd) i = Some r ->
    s_has_source (solution_of_tables T) r = false ->
    resolve (ap_continuation ap) (state_conts s)
    = Closure h vars conts recursive ->
    s_has_source (solution_of_tables T) (Node_variable x) = false ->
    run m
      (State (Apply ap) (state_vars s)
         ((ap_continuation ap,
           Closure (rename_param i x h) vars conts recursive)
          :: state_conts s)
         (state_statics s) (state_heap s) (state_reads s))
    = Step s' ->
    lookup x (state_vars s') = None.

(* The call of known arity [ap] of [s] that [apply_never_returns] makes one
   that never returns, entered (it steps), enters the code [id] of its
   callee, whose code dep has a result of some rank with no source: the
   premise of [dead_frame_statement] on the code. *)
Definition dead_frame_of_call_statement : Prop :=
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
    ∃ (cd : code_dep) (i : nat) (r : node),
      lookup id (unit_code_deps flag u e) = Some cd
      ∧ nth_error (cd_return cd) i = Some r
      ∧ s_has_source (solution_of_tables T) r = false.

(* The two statements above together: the frame of a call of known arity
   that [apply_never_returns] makes one that never returns, a handler, is
   never returned to with a value for each of its parameters. The rank [i]
   of the parameter renamed is that of the result with no source. *)
Definition never_returns_frame_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state)
    (ap : apply_expr) (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_expr s = Apply ap ->
    ap_call_kind ap <> Indirect_unknown_arity ->
    ap_exn_continuation ap <> ap_continuation ap ->
    apply_never_returns (traverse_unit flag u e) (solution_of_tables T)
      (unit_code_deps flag u e) (bound_code_metadata e) ap
    = true ->
    resolve (ap_continuation ap) (state_conts s)
    = Closure h vars conts recursive ->
    ∃ i : nat,
      ∀ (x : variable) (m : nat) (s' : state),
        s_has_source (solution_of_tables T) (Node_variable x) = false ->
        run m
          (State (Apply ap) (state_vars s)
             ((ap_continuation ap,
               Closure (rename_param i x h) vars conts recursive)
              :: state_conts s)
             (state_statics s) (state_heap s) (state_reads s))
        = Step s' ->
        lookup x (state_vars s') = None.
