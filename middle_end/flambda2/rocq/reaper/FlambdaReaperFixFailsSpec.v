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


(* The fatal errors of the fix proposed for the reaper, end to end: with
   the flag [Auto], a least solution and the tables derived from it, the
   rebuild of the fix stops ([rebuild_unit_unbox_repr_fixed_fails] of
   FlambdaReaperUnboxing.v) only at an unboxed callee of a call whose code
   keeps its calling convention, which is reached
   (FlambdaReaperWrapperDerivedSpec.unboxed_callee_fails_statement).

   This file is to be reviewed: the checks and the statement. The proofs
   are in FlambdaReaperFixFailsProofs.v, which needs no review.

   - [fix_fails_only_by_unboxed_callee_statement]: if the rebuild of the
     fix stops, a call that the rebuild makes ([unit_rebuilt_calls]) keeps
     its calling convention and has an unboxed callee. The walk of
     [rebuild_fails_u] reaches the calls of [unit_rebuilt_calls], under the
     same decisions of their continuations, and the code of [unit_codes]
     (code that is not used stops nothing where its [my_closure] is dead:
     its body is [Invalid]). At a
     call, the arm of [make_apply_wrapper] is not reached
     ([fix_arm_never_returns_auto_derived_statement]; the continuation of
     a call has no more parameters than the call has results, by
     [conts_counted], so that the walk and the rebuild read the same
     decisions); at code that is used, [check_never_returns] is not
     ([check_never_returns_auto_statement]).

   The premises, besides those of these two statements:
   - [one_group_unknown_calls]: every call of unknown arity of the term
     has one group of arguments. The arm is proved of such calls only;
     the compiler's calls of unknown arity have one group for each
     argument, so that this leaves out those of several arguments.
   - [results_within_codes]: every call of the term that is not of
     unknown arity expects no more results than any code of the unit
     returns: a check of the term for the premise on the arity of the
     two statements (a call expects no more results than the code dep of
     the code it is known to call has). The compiler's units meet it
     where every code returns as many values as the calls expect (one,
     but for unboxed products).
   - [unused_code_dead]: the [my_closure] of code that is not used has no
     source, so that its body is [Invalid]. This is of the solution, not
     of the term: its proof needs an inversion of the least solution on
     the facts that reach that node (see S7.md).

   [fix_fails_only_by_unboxed_callee_any_groups_statement]: the same,
   with [unknown_calls_have_args] in place of [one_group_unknown_calls]:
   every call of unknown arity has a group of arguments, of any number
   (the compiler's analysis stops on a call with none). The arm is not
   reached at such a call
   ([fix_arm_never_returns_auto_derived_full_statement]): the codes the
   chain of witnesses reaches cannot change their calling convention, so
   that, where the continuation unboxes a result, no source with a source
   of the last witness holds that result with a source ([cu0_result]),
   and the call [never_returns]. [unused_code_dead] stays.
   [one_group_unknown_calls_have_args_statement]: the new check is the
   weaker.

   [unboxed_callee_meets_premises_statement]: the unit of
   [unboxed_callee_fails_statement], whose rebuild with the fix stops,
   meets these premises and the others (proved there), so that the
   statement is not vacuous; it meets [unknown_calls_have_args] too, the
   weaker check. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperDerivedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

(* The calls of a term, in its handlers and in the body of every code it
   binds. *)
Fixpoint calls_of (e : expr) : list apply_expr :=
  match e with
  | Let _ _ _ body => calls_of body
  | Let_static bindings body =>
    flat_map binding_calls_of bindings ++ calls_of body
  | Let_cont_non_recursive _ h body => handler_calls_of h ++ calls_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_calls_of h) handlers ++ calls_of body
  | Apply ap => [ap]
  | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_calls_of (h : continuation_handler) : list apply_expr :=
  match h with
  | Handler _ _ handler => calls_of handler
  end
with binding_calls_of (binding : static_binding) : list apply_expr :=
  match binding with
  | Bind_code _ c => code_calls_of c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_calls_of (c : code) : list apply_expr :=
  match c with
  | Code _ _ _ _ _ body => calls_of body
  end.

(* Every call of unknown arity has one group of arguments. *)
Definition one_group_unknown_calls (e : expr) : bool :=
  forallb
    (fun ap =>
       match ap_call_kind ap with
       | Indirect_unknown_arity => Nat.eqb (length (ap_args_arity ap)) 1
       | Direct _ | Indirect_known_arity _ => true
       end)
    (calls_of e).

(* Every call of unknown arity has a group of arguments. *)
Definition unknown_calls_have_args (e : expr) : bool :=
  forallb
    (fun ap =>
       match ap_call_kind ap, ap_args_arity ap with
       | Indirect_unknown_arity, [] => false
       | _, _ => true
       end)
    (calls_of e).

(* Every call that is not of unknown arity expects no more results than
   any code of the unit returns. *)
Definition results_within_codes (e : expr) : bool :=
  forallb
    (fun ap =>
       match ap_call_kind ap with
       | Indirect_unknown_arity => true
       | Direct _ | Indirect_known_arity _ =>
         forallb
           (fun '(_, c) =>
              Nat.leb (length (unarize (ap_return_arity ap)))
                (length (code_result_kinds c)))
           (unit_codes e)
       end)
    (calls_of e).

(* The [my_closure] of code of the unit that is not used has no
   source. *)
Definition unused_code_dead (S : solution) (e : expr) : bool :=
  forallb
    (fun '(id, c) =>
       is_code_id_used S id
       || negb (s_has_source S (Node_variable (code_my_closure c))))
    (unit_codes e).

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition fix_fails_only_by_unboxed_callee_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting preserve_direct_calls : bool)
    (tcr : list node),
    let F := traverse_unit Auto u e in
    let S := solution_of_tables T in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let dominated := derive_dominated T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    exn_handlers_well_formed e = true ->
    one_group_unknown_calls e = true ->
    results_within_codes e = true ->
    unused_code_dead S e = true ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = true ->
    ∃ (ads : list decision) (ap : apply_expr) (fields : unboxed_fields),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
      ∧ apply_change_u F S (fun n => mem_node n cannot_change) code_deps
          (static_names code_ids_bound e) (bound_code_metadata e) unboxed ap
        = Not_changing_u
      ∧ unboxed_of unboxed (ap_callee ap) = Some fields.

(* The same for calls of unknown arity of any number of groups. *)
Definition fix_fails_only_by_unboxed_callee_any_groups_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting preserve_direct_calls : bool)
    (tcr : list node),
    let F := traverse_unit Auto u e in
    let S := solution_of_tables T in
    let code_deps := unit_code_deps Auto u e in
    let cannot_change := derive_cannot_change F T in
    let dominated := derive_dominated T in
    let to_unbox :=
      derive_to_unbox_fixed is_local F T cannot_change max_unbox_size
        lambda_lifting
    in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    exn_handlers_well_formed e = true ->
    unknown_calls_have_args e = true ->
    results_within_codes e = true ->
    unused_code_dead S e = true ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = true ->
    ∃ (ads : list decision) (ap : apply_expr) (fields : unboxed_fields),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e
      ∧ apply_change_u F S (fun n => mem_node n cannot_change) code_deps
          (static_names code_ids_bound e) (bound_code_metadata e) unboxed ap
        = Not_changing_u
      ∧ unboxed_of unboxed (ap_callee ap) = Some fields.

(* The check of one group is stronger. *)
Definition one_group_unknown_calls_have_args_statement : Prop :=
  ∀ e : expr,
    one_group_unknown_calls e = true -> unknown_calls_have_args e = true.

(* The unit at which the rebuild of the fix stops at an unboxed callee
   meets the premises that [unboxed_callee_fails_statement] does not
   state. *)
Definition unboxed_callee_meets_premises_statement : Prop :=
  let e := UnboxedCalleeExample.unit_expr in
  exn_handlers_well_formed e = true
  ∧ one_group_unknown_calls e = true
  ∧ results_within_codes e = true
  ∧ unused_code_dead (solution_of_tables UnboxedCalleeExample.tables) e
    = true.
