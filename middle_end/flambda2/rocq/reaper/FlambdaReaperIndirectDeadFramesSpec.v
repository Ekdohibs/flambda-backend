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

(* The frame of a call of unknown arity that the fix makes one that never
   returns ([apply_never_returns], [never_returns_of_call]), in the runs
   of the unit: FlambdaReaperDeadFramesSpec.never_returns_frame_statement
   for calls of unknown arity.

   This file is to be reviewed: [only_alias_into], [check_unknown_chains],
   [ChainsExample] and the statements. The proofs are in
   FlambdaReaperIndirectDeadFramesProofs.v, which needs no review.

   [never_returns_of_call] reads, group by group, the witnesses of unknown
   arity held by the sources of the callee, the sources of their first
   results, and so on; it finds, off the witnesses of the last group, a
   result of rank [i] with no source. The call reads the same off its own
   nodes: the read of the call witness off the callee, the first witness
   (an alias of that read), and, between two witnesses, the read of the
   first result and the read of its call witness. In a least solution,
   those nodes have only the sources so read if no other fact puts
   anything into them: [check_unknown_chains], a test of the facts, which
   the traversal of every unit passes
   (FlambdaReaperIndirectDeadFramesTraversalSpec.v).

   A call of unknown arity may be a full call, a partial application,
   which returns a value to the frame at once, or an over-application.
   The statement is then that the parameter of rank [i] of the handler
   returned to, renamed [x], a variable with no source, is never bound,
   not that the frame is never applied.

   [indirect_dead_frame_statement]: a unit of the flow fragment, tables
   that are a least solution of the rules on its graph, which passes
   [check_unknown_chains], and a state of its run at a call of unknown
   arity that [apply_never_returns] makes one that never returns,
   returning to a handler. For a rank [i] of the results of the call,
   rebind the continuation of the call to the handler with its parameter
   of rank [i] renamed [x]: the run from there never binds [x].

   [chains_example_statement]: a unit with such a call, of two groups of
   arguments, of code whose body is [Invalid], meets the premises on the
   facts and the tables. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaReaperLeastSpec FlambdaReaperDeadFramesSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast.
Import ListNotations.

(* No fact of [F] puts [x] where a rule may conclude [alias x _],
   [any_source x] or [sources x _], but aliases and propagations from
   [from]. *)
Definition only_alias_into (F : facts) (x from : node) : bool :=
  forallb (fun '(to_, y) => negb (node_eqb to_ x) || node_eqb y from)
    (f_alias F)
  && forallb (fun '(_, to_, y) => negb (node_eqb to_ x) || node_eqb y from)
       (f_propagate F)
  && forallb (fun '(_, to_, _) => negb (node_eqb to_ x))
       (f_alias_if_any_source F)
  && forallb (fun '(to_, _, _) => negb (node_eqb to_ x)) (f_accessor F)
  && forallb
       (fun '(base, _, to_) => negb (node_eqb to_ x || node_eqb base x))
       (f_parameter F)
  && forallb (fun '(base, _, _) => negb (node_eqb base x))
       (f_constructor F)
  && negb (mem_node x (f_any_source F))
  && negb (mem_node x (f_zero_alloc_source F))
  && forallb (fun '(_, y) => negb (node_eqb y x)) (f_any_source_if_used F)
  && forallb (fun '(to_, _) => negb (node_eqb to_ x)) (f_use F).

Local Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).
Local Abbreviation first_result := (Field_return_of_call (Normal 0)).

(* The nodes of the calls of unknown arity are only read into. For each
   read [x] of the call witness of unknown arity off [n]: nothing else
   puts anything into [x]; if [x] is a witness of a call (the code id is
   read off it), [n] is only the read of the first result off some node;
   and what [x] is aliased to, or propagated to, is only that. *)
Definition check_unknown_chains (F : facts) : bool :=
  forallb
    (fun '(x, f, n) =>
       negb (field_eqb f unknown)
       || (only_read_into F x unknown n false
           && (negb
                 (existsb
                    (fun '(_, f', b) =>
                       field_eqb f' Field_code_id_of_call_witness
                       && node_eqb b x)
                    (f_accessor F))
               || existsb
                    (fun '(y, f', w) =>
                       node_eqb y n && field_eqb f' first_result
                       && only_read_into F n first_result w false)
                    (f_accessor F))
           && forallb
                (fun '(w, y) => negb (node_eqb y x) || only_alias_into F w x)
                (f_alias F)
           && forallb
                (fun '(_, w, y) =>
                   negb (node_eqb y x) || only_alias_into F w x)
                (f_propagate F)))
    (f_accessor F).

Definition indirect_dead_frame_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat) (s : state)
    (ap : apply_expr) (h : continuation_handler) (vars : env)
    (conts : cont_env)
    (recursive : list (continuation * continuation_handler)),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_unknown_chains (traverse_unit flag u e) = true ->
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

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The code 50 takes two parameters and its body is [Invalid]; the unit
   calls a closure of it with unknown arity, two groups of one argument,
   returning to [k20]:

     let code f50 x40 x41 = invalid
     let x42 = closure of f50 in
     let k20 x30 = return x30 in
     x42 9 7 -> k20

   With the least solution, the call is one that never returns. *)
Module ChainsExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value; Singleton Value] [Singleton Value].

Definition call : apply_expr :=
  {| ap_callee := Var 42; ap_call_kind := Indirect_unknown_arity;
     ap_args := [int 9; int 7]; ap_continuation := 20;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 50 (Code [40; 41] 43 3 4 metadata Invalid)]
    (Let 42 Value (Closure_of_set 71 [(71, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (return_to 1 [Var 30]))
          (Apply call))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

End ChainsExample.

Definition chains_example_statement : Prop :=
  let F := ChainsExample.facts in
  let T := ChainsExample.tables in
  let e := ChainsExample.unit_expr in
  least_solution FlambdaReaperUnboxing.Examples.no_local F T
  ∧ in_flow_fragment e = true
  ∧ check_unknown_chains F = true
  ∧ apply_never_returns F (solution_of_tables T) ChainsExample.code_deps
      (bound_code_metadata e) ChainsExample.call
    = true.
