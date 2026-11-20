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
   rebuild of the fix never stops ([rebuild_unit_unbox_repr_fixed_fails]
   of FlambdaReaperUnboxing.v is [false]).

   This file is to be reviewed: the checks and the statement. The proofs
   are in FlambdaReaperFixFailsProofs.v, which needs no review.

   - [fix_never_fails_statement]. The walk of [rebuild_fails_u] reaches
     the calls of [unit_rebuilt_calls], under the same decisions of their
     continuations, and the code of [unit_codes] (code that is not used
     stops nothing where its [my_closure] is dead: its body is
     [Invalid]). At a call, the arm of [make_apply_wrapper] is not
     reached ([fix_arm_never_returns_auto_derived_full_statement]; the
     continuation of a call has no more parameters than the call has
     results, by [conts_counted], so that the walk and the rebuild read
     the same decisions); at code that is used, [check_never_returns] is
     not ([check_never_returns_auto_statement]). An unboxed callee of a
     call whose code keeps its calling convention makes the call
     [Invalid] and stops nothing: the fix stopped there with a fatal
     error, which the model and an OCaml program reached
     (FlambdaReaperWrapperDerivedSpec.unboxed_callee_invalid_statement),
     and its author made it [Invalid] again.

   The premises, besides those of these two statements:
   - [unknown_calls_have_args]: every call of unknown arity of the term
     has a group of arguments, of any number (the compiler's analysis
     stops on a call with none). The arm is not reached at such a call:
     the codes the chain of witnesses reaches cannot change their calling
     convention, so that, where the continuation unboxes a result, no
     source with a source of the last witness holds that result with a
     source ([cu0_result]), and the call [never_returns].
   - [results_within_codes]: every call of the term that is not of
     unknown arity expects no more results than any code of the unit
     returns: a check of the term for the premise on the arity of the
     two statements (a call expects no more results than the code dep of
     the code it is known to call has). The compiler's units meet it
     where every code returns as many values as the calls expect (one,
     but for unboxed products).
   - [unused_code_dead]: the [my_closure] of code that is not used has no
     source, so that its body is [Invalid]. This is of the solution, not
     of the term. FlambdaReaperUnusedCodeProofs.v proves it of a least
     solution where the facts into that node, into the [my_closure] of
     the code dep and into the nodes of the direct calls of the code have
     the shape that the traversal gives them, and
     FlambdaReaperUnusedCodeTraversalProofs.v that the traversal does so
     where every variable is bound once.

   [fix_never_fails_full_statement]: the same, with [variables_bound_once]
   (FlambdaReaperLeastSpec.v) in place of [unused_code_dead].

   [fix_never_fails_callees_statement]: the same, with two checks in
   place of [results_within_codes], which fails of a unit where a call
   expects more results than code that it does not call returns
   ([known_arity_callees_statement]):
   - [results_within_callees]: every call expects no more results than
     the code of the unit that it promises to call returns
     ([call_promises], as the machine reads the call: the code of a
     direct call; for a call of known arity, code that is not tupled,
     has a parameter for each group of arguments and is allowed by the
     list, if any). A check of the term, weaker than
     [results_within_codes] ([results_within_codes_callees_statement]).
     A direct call of code that the unit does not bind promises no code
     of the unit.
   - [callees_promised]: the code that the flow finds a call to call is
     code that the call promises to call. The premise on the arity of
     the two statements is of that code, which the flow reads off the
     closures that reach the callee, whatever the call's kind: a check
     of the term alone does not give it
     ([unpromised_callee_statement]: a direct call of the code 21 whose
     callee is a closure of the code 20). The machine runs a call only
     with a closure of code that it promises to call; the flow also
     follows closures into arms that no run takes.

   [fix_fails_only_by_unboxed_callee_statement] and
   [fix_fails_only_by_unboxed_callee_any_groups_statement], from before
   the fix made the unboxed callee [Invalid] again: if the rebuild of the
   fix stops, a call that the rebuild makes keeps its calling convention
   and has an unboxed callee. They hold, as the rebuild does not stop;
   the first with [one_group_unknown_calls] (every call of unknown arity
   has one group of arguments) in place of [unknown_calls_have_args],
   which is the weaker ([one_group_unknown_calls_have_args_statement]).

   [unboxed_callee_meets_premises_statement]: the unit of
   [unboxed_callee_invalid_statement] meets these premises and the
   others (proved there), so that they hold of a unit with an unboxed
   callee of a call whose code keeps its calling convention. *)

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

(* The code [id] of the unit, [c], is code that a call promises to
   call, by its kind, as the machine reads it
   (FlambdaSemanticsSpec.shape_of_closure_call): that of a direct call;
   for a call of known arity, code that its list allows, not tupled,
   with as many parameters as the call has groups of arguments, one at
   least. *)
Definition call_promises (ap : apply_expr) (id : code_id) (c : code)
    : bool :=
  match ap_call_kind ap with
  | Direct named => Pos.eqb named id
  | Indirect_known_arity code_ids =>
    match ap_args_arity ap with
    | [] => false
    | _ :: _ =>
      match code_ids with
      | Some ids => among id ids
      | None => true
      end
      && negb (code_is_tupled c)
      && Nat.eqb (length (ap_args_arity ap)) (length (code_params_arity c))
    end
  | Indirect_unknown_arity => false
  end.

(* Every call expects no more results than each code of the unit that
   it promises to call returns. *)
Definition results_within_callees (e : expr) : bool :=
  forallb
    (fun ap =>
       forallb
         (fun '(id, c) =>
            negb (call_promises ap id c)
            || Nat.leb (length (unarize (ap_return_arity ap)))
                 (length (code_result_kinds c)))
         (unit_codes e))
    (calls_of e).

(* The code of the unit that the flow finds a call to call
   ([code_id_actually_called]) is code that the call promises to
   call. *)
Definition callees_promised (F : facts) (T : tables) (e : expr) : bool :=
  forallb
    (fun ap =>
       match code_id_actually_called F (solution_of_tables T) ap with
       | Some id =>
         forallb
           (fun '(id', c) => negb (Pos.eqb id' id) || call_promises ap id' c)
           (unit_codes e)
       | None => true
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

Definition fix_never_fails_statement : Prop :=
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
    = false.

(* [fix_never_fails_statement] with [variables_bound_once] in place of
   [unused_code_dead], which a least solution of the facts of such a unit
   meets. *)
Definition fix_never_fails_full_statement : Prop :=
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
    = false.

(* [fix_never_fails_full_statement] with [results_within_callees] and
   [callees_promised] in place of [results_within_codes]. *)
Definition fix_never_fails_callees_statement : Prop :=
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
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    exn_handlers_well_formed e = true ->
    unknown_calls_have_args e = true ->
    results_within_callees e = true ->
    callees_promised F T e = true ->
    variables_bound_once e = true ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = false.

(* The check of every code is the stronger. *)
Definition results_within_codes_callees_statement : Prop :=
  ∀ e : expr,
    results_within_codes e = true -> results_within_callees e = true.

(* The statements from before the unboxed callee was [Invalid] again. *)
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

(* The unit with an unboxed callee meets the premises that
   [unboxed_callee_invalid_statement] does not state. *)
Definition unboxed_callee_meets_premises_statement : Prop :=
  let e := UnboxedCalleeExample.unit_expr in
  exn_handlers_well_formed e = true
  ∧ one_group_unknown_calls e = true
  ∧ results_within_codes e = true
  ∧ unused_code_dead (solution_of_tables UnboxedCalleeExample.tables) e
    = true.

(* ======================================================================== *)
(* The finer checks on two units                                            *)
(* ======================================================================== *)

(* The code 20 returns its parameter, the code 21 its two parameters.
   [known_unit] is the unit of reaper/cc_known_arity.fl, of the corpus:
   the closure [x11] of 21 goes through the closure [x10] of 20, by a
   direct call, and what comes back is called at known arity, with no
   list, for two results:

     let code 20 (x30) my_closure x31 = return x30 in
     let code 21 (x40, x41) my_closure x42 = return (x41, x40) in
     let x10 = closure 20 in
     let x11 = closure 21 in
     let_cont k7 (x14, x15) = return x14 in
     let_cont k8 (x12) = x12 (1, 2) -> k7, raise in
     direct(20) x10 (x11) -> k8, raise

   [unpromised_unit] calls the closure [x10] of 20 by a direct call of
   21, for two results, which the machine finds undefined:

     let code 20 ... in let code 21 ... in
     let x10 = closure 20 in
     let_cont k7 (x14, x15) = return x14 in
     direct(21) x10 (1, 2) -> k7, raise *)
Module CalleesExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition code_20 : code :=
  Code [30] 31 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 30]).

Definition code_21 : code :=
  Code [40; 41] 42 5 6
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value])
    (return_to 5 [Var 41; Var 40]).

Definition two_values : arity := [Singleton Value; Singleton Value].

Definition call_pass : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Direct 20; ap_args := [Var 11];
     ap_continuation := 8; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition call_known : apply_expr :=
  {| ap_callee := Var 12; ap_call_kind := Indirect_known_arity None;
     ap_args := [int 1; int 2]; ap_continuation := 7;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := two_values; ap_return_arity := two_values |}.

Definition call_unpromised : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Direct 21;
     ap_args := [int 1; int 2]; ap_continuation := 7;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := two_values; ap_return_arity := two_values |}.

Definition with_codes (body : expr) : expr :=
  Let_static [Bind_code 20 code_20; Bind_code 21 code_21]
    (Let 10 Value (Closure_of_set 50 [(50, 20)] [])
       (Let_cont_non_recursive 7
          (Handler [14; 15] (make_handler_metadata [Value; Value])
             (return_to 1 [Var 14]))
          body)).

Definition known_unit : expr :=
  with_codes
    (Let 11 Value (Closure_of_set 51 [(51, 21)] [])
       (Let_cont_non_recursive 8
          (Handler [12] (make_handler_metadata [Value]) (Apply call_known))
          (Apply call_pass))).

Definition unpromised_unit : expr := with_codes (Apply call_unpromised).

Definition facts_of (e : expr) : facts := traverse_unit Auto interface e.

Definition tables_of (e : expr) : tables :=
  match compute_least no_local (facts_of e) 40 with
  | Some T => T
  | None => empty_tables
  end.

End CalleesExample.

(* [known_unit] meets the premises of [fix_never_fails_callees_statement],
   with its least solution, but not [results_within_codes]: the call of
   [x12] expects two results, and the code 20 returns one. The flow finds
   that it calls 21. *)
Definition known_arity_callees_statement : Prop :=
  let e := CalleesExample.known_unit in
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ projection_free e = true
  ∧ conts_counted (unit_counts Examples.interface) e = true
  ∧ plain_kinds e = true ∧ exn_handlers_well_formed e = true
  ∧ unknown_calls_have_args e = true ∧ results_within_callees e = true
  ∧ callees_promised F T e = true ∧ variables_bound_once e = true
  ∧ results_within_codes e = false
  ∧ code_id_actually_called F (solution_of_tables T)
      CalleesExample.call_known
    = Some 21%positive.

(* [results_within_callees] alone does not give what the proof of
   [fix_never_fails_statement] reads of the arities (the premise on the
   arity of [fix_arm_never_returns_auto_derived_full_statement] and of
   [check_never_returns_auto_statement]): of [unpromised_unit], the flow
   finds that its call calls the code 20, whose code dep has one result,
   and the call expects two. *)
Definition unpromised_callee_statement : Prop :=
  let e := CalleesExample.unpromised_unit in
  let F := CalleesExample.facts_of e in
  let T := CalleesExample.tables_of e in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true
  ∧ results_within_callees e = true ∧ callees_promised F T e = false
  ∧ CalleesExample.call_unpromised ∈ calls_of e
  ∧ code_id_actually_called F (solution_of_tables T)
      CalleesExample.call_unpromised
    = Some 20%positive
  ∧ option_map (fun cd => length (cd_return cd))
      (lookup 20%positive (unit_code_deps Auto Examples.interface e))
    = Some 1
  ∧ length (unarize (ap_return_arity CalleesExample.call_unpromised)) = 2.
