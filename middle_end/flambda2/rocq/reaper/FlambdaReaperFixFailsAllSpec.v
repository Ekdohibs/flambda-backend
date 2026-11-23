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

(* The failure sites of the fix proposed for the reaper, besides the two
   of FlambdaReaperFixFailsSpec.v: the fatal errors, assertions and
   exceptions of its rebuild and analyses, each with the model's
   counterpart, are listed in the status of S7 (the entry "The failure
   sites of the fix's rebuild and analyses"). Most are unreachable by
   construction, outside the fragment, or excluded by a premise; where
   the model does something defined in their place, this file proves of
   some that the model never does it on the fragment, and gives units
   that reach two others.

   This file is to be reviewed: the definitions and the statements. The
   proofs are in FlambdaReaperFixFailsAllProofs.v.

   - [fix_never_fails_all_statement]: under the premises of
     [fix_never_fails_full_statement], its conclusion, and [uses_fit]:
     each continuation that the unit or its code applies or gives a call
     has, in the rebuild of the fix, as many decisions as it is given
     values ([cont_uses]), or none, being one of the unit, which the
     compiler keeps whole. rebuild.ml would stop otherwise: [get_args]
     and [get_args_with_kinds] ([List.fold_left2]) at an [Apply_cont] and
     at the extra arguments of an exception continuation,
     [Continuation.Map.find] of the decisions of a continuation not bound
     and [List.tl] of those of an exception continuation
     ([rewrite_apply_cont_expr], [rebuild_apply], [make_apply_wrapper]),
     [List.map2] of the return arity of a call ([rebuild_apply]), and
     [List.fold_left2] of [make_apply_wrapper] at a call that keeps its
     calling convention. The model keeps the arguments beyond the
     decisions, and has no decisions for a continuation not bound.

   - [more_results_statement]: the [List.fold_left2] of
     [make_apply_wrapper] at a call that changes its calling convention
     is reached. The code 21 returns two values, and a direct call of it
     expects one; [results_within_codes] allows it. The model cuts the
     decisions of the callee to the results of the call
     ([callee_decisions], [combine]); rebuild.ml raises
     [Invalid_argument].

   - [empty_callee_codes_statement]: the fatal error "Empty potentially
     called code_ids" of
     [decide_whether_apply_needs_calling_convention_change] is reached: a
     call of known arity of a block, whose callee has a source but holds
     no closure. The model keeps the call kind there
     ([FlambdaReaperRebuild.called_call_kind]), and makes this call
     [Invalid], its callee being unboxed; rebuild.ml decides the call
     kind before it tests the callee.

   - [no_parameter_code_statement]: the fatal error "add_deps: no params
     for code ID" of [create_unknown_arity_non_tupled_call_witnesses]
     (traverse_acc.ml) is reached by any code with no parameter, which
     [in_flow_fragment] allows where the unit makes no call of unknown
     arity. The model gives such code one group of none
     ([at_least_one_group]).

   The units meet the premises of [fix_never_fails_callees_statement]
   and of [fix_never_fails_full_statement], with a least solution. The
   compiler builds none of them: a call whose return arity is not that of
   its code, a call of known arity of what is not a closure, or code with
   no parameter, is not made by its front end or by Simplify from a
   well-typed program. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperDerivedSpec
  FlambdaReaperFixFailsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The uses of continuations                                                *)
(* ======================================================================== *)

Section Uses.

Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable unboxed : list (node * unboxed_fields).

(* Each continuation that [e] applies or gives a call, with the decisions
   that the rebuild of the fix reads for it under the decisions [ds] of
   the continuations in scope, and the number of values it is given: the
   return and exception continuations of a call (the exception and the
   extra arguments), and the continuation of an [Apply_cont] and of each
   arm of a [Switch]. The continuations are bound as [rebuild_expr_u]
   binds them; every code of the unit is gone through, as the compiler
   rebuilds all code. *)
Fixpoint cont_uses (ds : finite_map (list decision)) (e : expr) {struct e}
    : list (continuation * option (list decision) * nat) :=
  match e with
  | Let _ _ _ body => cont_uses ds body
  | Let_static bindings body =>
    flat_map binding_cont_uses bindings ++ cont_uses ds body
  | Let_cont_non_recursive k h body =>
    handler_cont_uses ds h
    ++ cont_uses
         ((k, handler_decisions true sol unboxed (handler_is_exn_handler h)
                h)
          :: ds)
         body
  | Let_cont_recursive handlers body =>
    let ds :=
      map (fun '(k, h) => (k, handler_decisions true sol unboxed false h))
        handlers
      ++ ds
    in
    flat_map (fun '(_, h) => handler_cont_uses ds h) handlers
    ++ cont_uses ds body
  | Apply ap =>
    [(ap_continuation ap, lookup (ap_continuation ap) ds,
      length (unarize (ap_return_arity ap)));
     (ap_exn_continuation ap, lookup (ap_exn_continuation ap) ds,
      S (length (ap_exn_extra_args ap)))]
  | Apply_cont ac =>
    [(ac_continuation ac, lookup (ac_continuation ac) ds, length (ac_args ac))]
  | Switch _ arms =>
    map
      (fun '(_, ac) =>
         (ac_continuation ac, lookup (ac_continuation ac) ds,
          length (ac_args ac)))
      arms
  | Invalid => []
  end
with handler_cont_uses (ds : finite_map (list decision))
    (h : continuation_handler) {struct h}
    : list (continuation * option (list decision) * nat) :=
  match h with
  | Handler _ _ handler => cont_uses ds handler
  end
with binding_cont_uses (binding : static_binding) {struct binding}
    : list (continuation * option (list decision) * nat) :=
  match binding with
  | Bind_code id c => code_cont_uses id c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_cont_uses (id : code_id) (c : code) {struct c}
    : list (continuation * option (list decision) * nat) :=
  match c with
  | Code _ _ return_continuation exn_continuation metadata body =>
    cont_uses
      (code_continuation_decisions sol code_deps unboxed id metadata
         (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
            metadata)
         return_continuation exn_continuation)
      body
  end.

End Uses.

Definition unit_cont_uses (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (unboxed : list (node * unboxed_fields))
    (e : expr) : list (continuation * option (list decision) * nat) :=
  cont_uses (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) unboxed [] e.

(* A use fits: as many decisions as values, or no decisions for a
   continuation of the unit ([counts]) given as many values as it has
   parameters. *)
Definition use_fits (counts : finite_map nat)
    (use : continuation * option (list decision) * nat) : bool :=
  match use with
  | (k, Some decisions, n) => Nat.eqb (length decisions) n
  | (k, None, n) => count_ok counts k n
  end.

Definition uses_fit (counts : finite_map nat)
    (uses : list (continuation * option (list decision) * nat)) : bool :=
  forallb (use_fits counts) uses.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition fix_never_fails_all_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (max_unbox_size : nat) (lambda_lifting preserve_direct_calls : bool)
    (tcr : list node),
    let F := traverse_unit Auto u e in
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
    variables_bound_once e = true ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = false
    ∧ uses_fit (unit_counts u)
        (unit_cont_uses T cannot_change code_deps unboxed e)
      = true.

(* ======================================================================== *)
(* Two units that reach a failure site                                      *)
(* ======================================================================== *)

(* [more_results] calls the code 21 of CalleesExample, which returns its
   two parameters swapped, directly, for one result:

     let code 21 (x40, x41) my_closure x42 = return (x41, x40) in
     let x10 = closure 21 in
     let_cont k7 (x14) = return x14 in
     direct(21) x10 (1, 2) -> k7, raise

   [block_callee] calls a block at known arity:

     let x10 = [5] in
     let_cont k7 (x14) = return x14 in
     x10 (1) -> k7, raise

   [no_parameter] binds code with no parameter:

     let code 22 () my_closure x43 = return 3 in
     return 0 *)
Module FixFailsAllExamples.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition call_one : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Direct 21;
     ap_args := [int 1; int 2]; ap_continuation := 7;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition return_k7 (body : expr) : expr :=
  Let_cont_non_recursive 7
    (Handler [14] (make_handler_metadata [Value]) (return_to 1 [Var 14]))
    body.

Definition more_results : expr :=
  Let_static [Bind_code 21 CalleesExample.code_21]
    (Let 10 Value (Closure_of_set 50 [(50, 21)] [])
       (return_k7 (Apply call_one))).

Definition call_block : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Indirect_known_arity None;
     ap_args := [int 1]; ap_continuation := 7;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition block_callee : expr :=
  Let 10 Value (make_block [int 5]) (return_k7 (Apply call_block)).

Definition code_22 : code :=
  Code [] 43 5 6 (make_code_metadata [] [Singleton Value])
    (return_to 5 [int 3]).

Definition no_parameter : expr :=
  Let_static [Bind_code 22 code_22] (return_to 1 [int 0]).

(* The premises of [fix_never_fails_callees_statement] and of
   [fix_never_fails_full_statement] but the least solution, for no local
   field. *)
Definition premises (e : expr) : bool :=
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  in_flow_fragment e && projection_free e
  && conts_counted (unit_counts interface) e && plain_kinds e
  && exn_handlers_well_formed e && unknown_calls_have_args e
  && results_within_codes e && results_within_callees e
  && callees_promised F T e && variables_bound_once e.

(* The arguments of the rebuild of the fix for [e], with
   [max_unbox_size] and without lambda lifting. *)
Definition cannot_change_of (e : expr) : list node :=
  derive_cannot_change (CalleesExample.facts_of e)
    (CalleesExample.tables_of e).

Definition code_deps_of (e : expr) : finite_map code_dep :=
  unit_code_deps Auto interface e.

Definition unboxed_of_unit (e : expr) : list (node * unboxed_fields) :=
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  unit_unboxed F T (derive_dominated T)
    (derive_to_unbox_fixed no_local F T (cannot_change_of e) max_unbox_size
       false)
    e.

Definition rebuilt_calls_of (e : expr) : list (list decision * apply_expr) :=
  unit_rebuilt_calls (CalleesExample.tables_of e) (cannot_change_of e)
    (code_deps_of e) (unboxed_of_unit e) e.

End FixFailsAllExamples.

(* [more_results] meets the premises; its call is rebuilt, with one
   decision for its continuation, and changes its calling convention
   with two decisions for the results of the code, with its
   [my_closure] kept; it may return, and its callee has a source. The
   decisions of the callee that the model reads are cut to one. *)
Definition more_results_statement : Prop :=
  let e := FixFailsAllExamples.more_results in
  let ap := FixFailsAllExamples.call_one in
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  let S := solution_of_tables T in
  let cc := fun n => mem_node n (FixFailsAllExamples.cannot_change_of e) in
  let cds := FixFailsAllExamples.code_deps_of e in
  let uci := static_names code_ids_bound e in
  let codes := bound_code_metadata e in
  let unboxed := FixFailsAllExamples.unboxed_of_unit e in
  least_solution Examples.no_local F T
  ∧ FixFailsAllExamples.premises e = true
  ∧ FixFailsAllExamples.rebuilt_calls_of e = [([Decided (Keep Value)], ap)]
  ∧ s_has_source S (Node_variable 10%positive) = true
  ∧ apply_change_u F S cc cds uci codes unboxed ap
    = Changing_u [Decided Delete; Decided (Keep Value)]
        [Decided (Keep Value); Decided Delete]
  ∧ apply_my_closure_u F S cc cds uci codes unboxed ap = None
  ∧ apply_never_returns F S cds codes ap = false
  ∧ unit_callee_decisions F T (FixFailsAllExamples.cannot_change_of e) cds
      unboxed e ap
    = [(Decided (Keep Value), Value)].

(* [block_callee] meets the premises; its call is rebuilt, its callee has
   a source and its argument is a constant, and the flow finds no code
   that the callee may call. *)
Definition empty_callee_codes_statement : Prop :=
  let e := FixFailsAllExamples.block_callee in
  let ap := FixFailsAllExamples.call_block in
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  let S := solution_of_tables T in
  least_solution Examples.no_local F T
  ∧ FixFailsAllExamples.premises e = true
  ∧ FixFailsAllExamples.rebuilt_calls_of e = [([Decided (Keep Value)], ap)]
  ∧ s_has_source S (Node_variable 10%positive) = true
  ∧ code_id_actually_directly_called F S (Node_variable 10%positive) = Some [].

(* [no_parameter] meets the premises, and binds code with no group of
   parameters. *)
Definition no_parameter_code_statement : Prop :=
  let e := FixFailsAllExamples.no_parameter in
  least_solution Examples.no_local (CalleesExample.facts_of e)
    (CalleesExample.tables_of e)
  ∧ FixFailsAllExamples.premises e = true
  ∧ unit_codes e = [(22%positive, FixFailsAllExamples.code_22)]
  ∧ code_params_arity FixFailsAllExamples.code_22 = [].
