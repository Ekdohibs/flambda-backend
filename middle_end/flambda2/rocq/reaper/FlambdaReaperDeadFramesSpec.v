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
   handler with its parameter [i] renamed is never applied.

   The statements above are of the runs of the unit. In the rebuilt terms
   the returns of such code are dropped by the terms themselves, with no
   invariant of their runs: [dead_return_dropped_statement], every
   [Apply_cont] of the body of the code to its return continuation
   ([applies_to]), and every arm of a switch, that gives a value at the
   rank of the result with no source gives it a name with no source, and
   both rebuilds drop it. What is left to apply the frame is a call given
   it as a continuation (in tail position) and the wrappers of the rebuilds
   around calls. This file is to be reviewed for [applies_to] too.

   [dead_returns_not_live_statement]: for code that [code_never_returns],
   with no premise on the arity of its returns, which the flow fragment
   gives ([conts_counted]), no such [Apply_cont] or arm is live in the
   sense of FlambdaReaperUnboxingRefinesCallsSpec.live_conts.

   [unused_never_returns_example_statement]: with the least solution, under
   [Auto], code may never return only because nothing uses it, and then a
   call that returns may be in tail position in it: its return
   continuation is live, and the check of [calls_code] fails with
   [code_never_returns] for [nr]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperFlowSpec
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperLeast FlambdaReaperLeastSpec
  FlambdaReaperUnboxingRefinesCallsSpec.
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

(* The [Apply_cont]s of [e] to [k], and the arms of its switches to [k]
   that the rebuilds keep ([Keyed.compact]), not under a binding of [k] nor
   in code that [e] binds. *)
Fixpoint applies_to (k : continuation) (e : expr) {struct e}
    : list apply_cont :=
  match e with
  | Let _ _ _ body | Let_static _ body => applies_to k body
  | Let_cont_non_recursive k' h body =>
    handler_applies_to k h
    ++ (if Pos.eqb k' k then [] else applies_to k body)
  | Let_cont_recursive handlers body =>
    if existsb (fun '(k', _) => Pos.eqb k' k) handlers
    then []
    else
      flat_map (fun '(_, h) => handler_applies_to k h) handlers
      ++ applies_to k body
  | Apply_cont ac => if Pos.eqb (ac_continuation ac) k then [ac] else []
  | Switch _ arms =>
    filter (fun ac => Pos.eqb (ac_continuation ac) k)
      (map snd (Keyed.compact Z.eqb arms))
  | Apply _ | Invalid => []
  end
with handler_applies_to (k : continuation) (h : continuation_handler)
    {struct h} : list apply_cont :=
  match h with
  | Handler _ _ handler => applies_to k handler
  end.

(* The frames of code whose result of rank [i] has no source, in the
   terms rebuilt: an [Apply_cont] of the body of the code to its return
   continuation, or an arm of a switch, that gives it a value at rank [i]
   gives it a name with no source, and both rebuilds drop it (the change of
   calling conventions, [rewrite_apply_cont], and the rebuild with
   unboxing, [rewrite_apply_cont_u]: [Invalid], or no arm). The return
   continuation of such code is then applied, in the rebuilt terms, only
   by calls given it as a continuation and by the wrappers that the
   rebuilds make around calls. *)
Definition dead_return_dropped_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (id : code_id) (c : code)
    (cd : code_dep) (i : nat) (r : node) (ac : apply_cont),
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    In (id, c) (unit_codes e) ->
    lookup id (unit_code_deps flag u e) = Some cd ->
    nth_error (cd_return cd) i = Some r ->
    s_has_source (solution_of_tables T) r = false ->
    In ac (applies_to (code_return_continuation c) (code_body c)) ->
    i < length (ac_args ac) ->
    existsb (simple_has_no_source (solution_of_tables T)) (ac_args ac)
    = true
    ∧ (∀ en : rebuild_env,
         rewrite_apply_cont (solution_of_tables T) en ac = None)
    ∧ (∀ (unboxed : list (node * unboxed_fields)) (en : unbox_env),
         rewrite_apply_cont_u (solution_of_tables T) unboxed en ac = None).

(* The returns of code whose result has no source are not live
   ([FlambdaReaperUnboxingRefinesCallsSpec.live_apply_cont]): an
   [Apply_cont] to its return continuation, or an arm kept, gives as many
   values as the code has results ([conts_counted]), so that
   [dead_return_dropped_statement] applies to it. What is left of the
   check of [calls_code] on such code is in its calls and in the arms that
   [Keyed.compact] drops. *)
Definition dead_returns_not_live_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (id : code_id) (c : code)
    (ac : apply_cont),
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    In (id, c) (unit_codes e) ->
    code_never_returns (solution_of_tables T) (unit_code_deps flag u e) id
    = true ->
    In ac (applies_to (code_return_continuation c) (code_body c)) ->
    FlambdaReaperUnboxingRefinesCallsSpec.live_apply_cont
      (solution_of_tables T) ac
    = [].

(* Code that nothing calls, taken to never return only for that: the code
   50 calls the code 51, which returns, in tail position. The facts of
   the call in 50 hold only where 50 is used ([cond_alias]), and nothing
   uses it, so the result of 50 has no source in the least solution. The
   call returns to the return continuation 3 of 50, which is then live:
   with the code that never returns for [nr], the check of [calls_code]
   fails. The fragment leaves the unit out by another row as well: the
   callee [x42] of the call has no use. *)
Module UnusedExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition call : apply_expr :=
  {| ap_callee := Var 42; ap_call_kind := Direct 51; ap_args := [Var 40];
     ap_continuation := 3; ap_exn_continuation := 4;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition body : expr :=
  Let 42 Value (Closure_of_set 71 [(71, 51)] []) (Apply call).

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata body);
     Bind_code 51 (Code [43] 44 5 6 metadata (return_to 5 [int 0]))]
    (return_to 1 [int 0]).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

End UnusedExample.

Definition unused_never_returns_example_statement : Prop :=
  let F := UnusedExample.facts in
  let T := UnusedExample.tables in
  let sol := solution_of_tables T in
  let cds := UnusedExample.code_deps in
  let e := UnusedExample.unit_expr in
  FlambdaReaperLeastSpec.least_solution FlambdaReaperUnboxing.Examples.no_local
    F T
  ∧ in_flow_fragment e = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface e = true
  ∧ s_any_usage sol (Node_code_id 50%positive) = false
  ∧ code_never_returns sol cds 50%positive = true
  ∧ code_never_returns sol cds 51%positive = false
  ∧ among 3%positive
      (FlambdaReaperUnboxingRefinesCallsSpec.live_conts F sol cds
         (bound_code_metadata e) UnusedExample.body)
    = true
  ∧ FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr
      (code_never_returns sol cds) F T [] cds [] [] e
    = false
  ∧ is_var_used sol 42%positive Value = false.
