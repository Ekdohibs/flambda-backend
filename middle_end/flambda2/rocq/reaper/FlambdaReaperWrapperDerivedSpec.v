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
     graph is a least solution.

   The fatal errors of the fix ([rebuild_fails_u] of
   FlambdaReaperUnboxing.v), with the flag [Auto], a least solution and
   the tables derived from it:
   - [fix_arm_never_returns_auto_derived_statement]: the arm of
     [make_apply_wrapper] that the fix stops on ([wrapper_fails_u]: the
     continuation unboxes a result that the callee keeps or deletes) is
     reached, by a call that the rebuild makes (in the unit or in code
     that is used), only if the call never returns
     ([apply_never_returns]), which the fix tests first: the arm is not
     reached. Of the calls: those that are not of unknown arity, which
     expect of the code they are known to call no more results than its
     code dep has (a premise on the arity, which the compiler's terms
     meet), and those of unknown arity with one group of arguments. The
     statements of FlambdaReaperWrapperSpec.v give no source to the
     result at the call witness that the call reads off its callee; the
     witness is that of the code dep of the code the call is known to
     call, which then has a result with no source
     ([code_never_returns]), or the sources of the witness that
     [never_returns_of_call] reads are those of the node read, by the
     rules ([fix_arm_never_returns_statement],
     [indirect_fix_arm_never_returns_statement] there).
   - [fix_arm_never_returns_auto_derived_full_statement]: the same for
     every call of unknown arity with a group of arguments or more. Each
     group reads a witness off the node that the previous one returns; a
     source with a source of the last witness is held in the field of
     unknown arity of some node ([check_unknown_witnesses]) and has a
     use, so its code cannot change its calling convention, and holds the
     result that the continuation unboxes only with no source
     ([cu0_result]): the call never returns.
   - [unboxed_callee_invalid_statement]: an unboxed callee of a call
     whose code keeps its calling convention is reached, and the call is
     rebuilt as [Invalid] ("Unboxed callee ... cannot actually be a
     function"). A block, of which a field is read, is also called at
     unknown arity in an arm that is not taken; nothing in the rules
     forbids to unbox it (a call reads the field of the call witness, not
     a real field, and the seeds of the fix need a use with another
     source or any source). The call, of no known code, keeps its calling
     convention; the rebuild of the fix does not stop (the fix stopped
     there with a fatal error for a while, and no longer does), and both
     the unit and its rebuilt unit exit with 1.
   - [check_never_returns_auto_statement]: [check_never_returns], the
     third error, is not reached by code that is used: the rebuilt body
     of code with a result of no source ([code_never_returns]) has its
     return continuation free nowhere. It holds of any tables of the
     rebuild, with a least solution. An [Apply_cont] to the return
     continuation passes the result an argument of which it is an
     alias, which so has no source, and is rebuilt as [Invalid]; a call
     that returns there gives it its own result, of which it is an alias
     where the code is used, so that the call never returns
     ([apply_never_returns]) and returns to the wrapper [Invalid]; every
     other term of the rebuild keeps the free continuations of its
     parts, or fewer. The premises: the exception handlers well formed
     ([exn_handlers_well_formed], of [well_formed_unit]: no call has the
     return continuation of code as its exception continuation), and two
     of the calls the rebuild makes, which the compiler's terms meet: one
     of code that is known expects of it no more results than its code
     dep has (as for the arm), and one of unknown arity has an argument
     (the compiler's analysis stops on one that has none). Code that is
     not used is not covered (see S7.md). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaValueSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperFlowSpec FlambdaReaperLeastSpec FlambdaReaperWrapperSpec.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec.
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

(* ======================================================================== *)
(* An unboxed callee                                                        *)
(* ======================================================================== *)

(* A block [x10] whose field is read into [x11], which the unit returns,
   and which an arm that is not taken calls at unknown arity:

     let k20 x30 = return x30 in
     let x10 = [1; 2] in
     let x11 = x10.(0) in
     let k21 () = return x11 in
     let k22 () = x10 3 (unknown arity) -> k20 in
     switch 0 [0 -> k21 (); 1 -> k22 ()]

   The tables are the least solution of the rules
   (FlambdaReaperLeast.compute_least), and those of [cannot_change],
   [dominated], [to_unbox] and [to_change_representation] the ones
   derived from it. *)
Module UnboxedCalleeExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition call : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Indirect_unknown_arity;
     ap_args := [int 3]; ap_continuation := 20; ap_exn_continuation := 2;
     ap_exn_extra_args := []; ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

(* The unit, with [k22] the handler of the arm that calls. *)
Definition unit_with (k22 : expr) : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value]) (return_to 1 [Var 30]))
    (Let 10 Value (make_block [int 1; int 2])
       (Let 11 Value (load 0 (Var 10))
          (Let_cont_non_recursive 21
             (Handler [] (make_handler_metadata []) (return_to 1 [Var 11]))
             (Let_cont_non_recursive 22
                (Handler [] (make_handler_metadata []) k22)
                (Switch (Const (Naked_immediate 0))
                   [(0%Z, jump 21 []); (1%Z, jump 22 [])]))))).

Definition unit_expr : expr := unit_with (Apply call).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables cannot_change max_unbox_size
    false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables cannot_change
    max_unbox_size false to_unbox.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The rebuilt unit: [x10] is the variable [x31] of its field, the call
   is [Invalid], and so is the continuation [k20], which nothing applies
   any more. *)
Definition rebuilt : expr :=
  Let 31 Value (Simple (int 1))
    (Let 11 Value (Simple (Var 31))
       (Let_cont_non_recursive 21
          (Handler [] (make_handler_metadata []) (return_to 1 [Var 11]))
          (Let_cont_non_recursive 22
             (Handler [] (make_handler_metadata []) Invalid)
             (Switch (Const (Naked_immediate 0))
                [(0%Z, jump 21 []); (1%Z, jump 22 [])])))).

End UnboxedCalleeExample.

Definition unboxed_callee_invalid_statement : Prop :=
  let F := UnboxedCalleeExample.facts in
  let T := UnboxedCalleeExample.tables in
  let e := UnboxedCalleeExample.unit_expr in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ projection_free e = true
  ∧ conts_counted (unit_counts Examples.interface) e = true
  ∧ plain_kinds e = true
  ∧ UnboxedCalleeExample.to_unbox = [Node_variable 10%positive]
  ∧ unit_unboxed F T UnboxedCalleeExample.dominated
      UnboxedCalleeExample.to_unbox e
    = [(Node_variable 10%positive,
        [(Field_block 0 Value, Not_unboxed 31%positive)])]
  ∧ rebuild_unit_unbox_repr_fixed_fails Examples.no_local F T
      UnboxedCalleeExample.cannot_change UnboxedCalleeExample.code_deps
      UnboxedCalleeExample.dominated UnboxedCalleeExample.to_unbox
      UnboxedCalleeExample.tcr true e
    = false
  ∧ rebuild_unit_unbox_repr_fixed Examples.no_local F T
      UnboxedCalleeExample.cannot_change UnboxedCalleeExample.code_deps
      UnboxedCalleeExample.dominated UnboxedCalleeExample.to_unbox
      UnboxedCalleeExample.tcr true e
    = UnboxedCalleeExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own e)
      (Exits 1%positive [Value_const (Tagged_immediate 1)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own UnboxedCalleeExample.rebuilt)
      (Exits 1%positive [Value_const (Tagged_immediate 1)] empty_heap).

Definition fix_arm_never_returns_auto_derived_statement : Prop :=
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
    ∀ (ads : list decision) (ap : apply_expr),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap <> Indirect_unknown_arity
      ∨ length (ap_args_arity ap) = 1 ->
      (∀ (id : code_id) (cd : code_dep),
         code_id_actually_called F (solution_of_tables T) ap = Some id ->
         (id, cd) ∈ code_deps ->
         length (unarize (ap_return_arity ap)) <= length (cd_return cd)) ->
      wrapper_fails_u ads
        (map fst
           (unit_callee_decisions F T cannot_change code_deps unboxed e ap))
      = true ->
      apply_never_returns F (solution_of_tables T) code_deps
        (bound_code_metadata e) ap
      = true.

Definition fix_arm_never_returns_auto_derived_full_statement : Prop :=
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
    ∀ (ads : list decision) (ap : apply_expr),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      (ap_call_kind ap = Indirect_unknown_arity -> ap_args_arity ap <> []) ->
      (∀ (id : code_id) (cd : code_dep),
         code_id_actually_called F (solution_of_tables T) ap = Some id ->
         (id, cd) ∈ code_deps ->
         length (unarize (ap_return_arity ap)) <= length (cd_return cd)) ->
      wrapper_fails_u ads
        (map fst
           (unit_callee_decisions F T cannot_change code_deps unboxed e ap))
      = true ->
      apply_never_returns F (solution_of_tables T) code_deps
        (bound_code_metadata e) ap
      = true.

Definition check_never_returns_auto_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (unboxed : list (node * unboxed_fields))
    (first_fresh : positive) (changed : list (node * closure_repr))
    (preserve_direct_calls : bool),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let S := solution_of_tables T in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    exn_handlers_well_formed e = true ->
    (∀ (ads : list decision) (ap : apply_expr),
       (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
       (ap_call_kind ap = Indirect_unknown_arity -> ap_args_arity ap <> [])
       ∧ ∀ (id : code_id) (cd : code_dep),
           code_id_actually_called F S ap = Some id ->
           (id, cd) ∈ code_deps ->
           length (unarize (ap_return_arity ap)) <= length (cd_return cd)) ->
    ∀ (id : code_id) (c : code),
      (id, c) ∈ unit_codes e -> is_code_id_used S id = true ->
      code_never_returns S code_deps id
      && among (code_return_continuation c)
           (free_continuations_of_expr
              (code_body
                 (rebuild_code_u true F S (fun n => mem_node n cannot_change)
                    code_deps preserve_direct_calls
                    (static_names code_ids_bound e) (bound_code_metadata e)
                    unboxed first_fresh changed id c)))
      = false.
