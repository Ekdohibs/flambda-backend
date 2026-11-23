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

(* Three premises that exclude the units of FlambdaReaperFixFailsAllSpec.v
   that reach a failure site of the fix, and the statement that, with
   them, none of the three sites is reached.

   This file is to be reviewed: the checks and the statement. The proofs
   are in FlambdaReaperFixFailsAll2Proofs.v.

   The premises, each of the term or, as [callees_promised], of the flow:
   - [results_match_callees]: every call expects exactly as many results
     as each code of the unit that it promises to call returns
     ([call_promises]): [results_within_callees] with an equality.
     Simplify checks it of every full application of code it knows
     (simplify_apply_expr.ml, "Wrong return arity for direct OCaml
     function call": a fatal error under kind checks, else [Invalid]).
   - [unit_codes_have_params]: every code of the unit has a parameter
     ([codes_have_params] of FlambdaReaperUnboxingRefinesOver1Spec.v, of
     [bound_code_metadata e], which lists the same code). Lambda makes
     no function without one ([Lambda.lfunction'] asserts it).
   - [known_callees_have_code]: the flow finds code for the callee, if it
     has a source, of every direct call and call of known arity whose
     callee is a name ([code_id_actually_directly_called] is not the
     empty list). Simplify makes such a call only of a callee whose type
     is a closure of known code, which is where the flow finds the
     closure's call witness; the traversal gives none to a closure of
     [Deleted] code.

   - [call_args_fit]: every call has as many arguments as its arity of
     arguments has components ([Apply.create] checks it of every call,
     apply_expr.ml, "Length of argument and arity lists disagree").
     [call_args_grouped_statement]: with it, the groups of the
     arguments of a call ([group_by_parameter]) cut or drop none, where
     rebuild.ml ([rebuild_apply]) and traverse_acc.ml
     ([make_unknown_arity_apply_widget]) would stop
     ([Flambda_arity.group_by_parameter]). The flow fragment does not
     ask it: a call of two arguments of the arity [[Singleton Value]] is
     of the fragment.

   [fix_never_fails_all2_statement]: with the premises of
   [fix_never_fails_callees_statement] (with [results_match_callees] in
   place of [results_within_callees]) and these, the conclusion of
   [fix_never_fails_all_statement] and:
   - [calls_fit]: each call that the rebuild makes, if its code changes
     its calling convention, has as many return decisions from the code
     as it expects results (the [List.fold_left2] of
     [make_apply_wrapper]: [more_results_statement]), and a group of
     arguments ("Empty argument groups in direct apply rewrite"); and
     the flow finds code for its callee if it has a source ("Empty
     potentially called code_ids": [empty_callee_codes_statement]).
   - [codes_grouped]: the parameters of the code dep of each code of the
     unit fall in a group at least ("add_deps: no params for code ID",
     "Empty parameter groups when changing calling convention":
     [no_parameter_code_statement]); the traversal's
     [at_least_one_group] then changes nothing. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperDerivedSpec
  FlambdaReaperFixFailsSpec FlambdaReaperFixFailsAllSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* Every call expects as many results as each code of the unit that it
   promises to call returns. *)
Definition results_match_callees (e : expr) : bool :=
  forallb
    (fun ap =>
       forallb
         (fun '(id, c) =>
            negb (call_promises ap id c)
            || Nat.eqb (length (unarize (ap_return_arity ap)))
                 (length (code_result_kinds c)))
         (unit_codes e))
    (calls_of e).

(* Every code of the unit has a parameter. *)
Definition unit_codes_have_params (e : expr) : bool :=
  forallb (fun '(_, c) => negb (Nat.eqb (length (code_params_arity c)) 0))
    (unit_codes e).

(* The flow finds code for the callee [callee] of a direct call or of a
   call of known arity, if it has a source. *)
Definition callee_has_code (F : facts) (S : solution) (ap : apply_expr)
    : bool :=
  match ap_call_kind ap, ap_callee ap with
  | Indirect_unknown_arity, _ | _, Const _ => true
  | _, callee =>
    negb (s_has_source S (node_of_simple callee))
    || match code_id_actually_directly_called F S (node_of_simple callee)
       with
       | Some [] => false
       | Some (_ :: _) | None => true
       end
  end.

Definition known_callees_have_code (F : facts) (T : tables) (e : expr)
    : bool :=
  forallb (callee_has_code F (solution_of_tables T)) (calls_of e).

(* ======================================================================== *)
(* The conclusion                                                           *)
(* ======================================================================== *)

(* A call that the rebuild makes: if its code changes its calling
   convention, as many return decisions as results, and a group of
   arguments; and code for its callee. *)
Definition call_fits (F : facts) (S : solution) (cannot_change : node -> bool)
    (code_deps : finite_map code_dep) (unit_code_ids : list code_id)
    (codes : finite_map code_metadata)
    (unboxed : list (node * unboxed_fields)) (ap : apply_expr) : bool :=
  match
    apply_change_u F S cannot_change code_deps unit_code_ids codes unboxed ap
  with
  | Not_changing_u => true
  | Changing_u _ return_decisions =>
    Nat.eqb (length return_decisions) (length (unarize (ap_return_arity ap)))
    && negb (Nat.eqb (length (ap_args_arity ap)) 0)
  end
  && callee_has_code F S ap.

Definition calls_fit (F : facts) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep)
    (unboxed : list (node * unboxed_fields)) (e : expr) : bool :=
  forallb
    (fun '(_, ap) =>
       call_fits F (solution_of_tables T) (fun n => mem_node n cannot_change)
         code_deps (static_names code_ids_bound e) (bound_code_metadata e)
         unboxed ap)
    (unit_rebuilt_calls T cannot_change code_deps unboxed e).

(* Each code of the unit has a code dep, whose parameters fall in a group
   at least. *)
Definition codes_grouped (code_deps : finite_map code_dep) (e : expr)
    : bool :=
  forallb
    (fun '(id, c) =>
       match lookup id code_deps with
       | Some cd =>
         negb (Nat.eqb
                 (length (group_by_parameter (code_params_arity c)
                            (cd_params cd)))
                 0)
       | None => false
       end)
    (unit_codes e).

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition fix_never_fails_all2_statement : Prop :=
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
    results_match_callees e = true ->
    callees_promised F T e = true ->
    variables_bound_once e = true ->
    unit_codes_have_params e = true ->
    known_callees_have_code F T e = true ->
    rebuild_unit_unbox_repr_fixed_fails is_local F T cannot_change code_deps
      dominated to_unbox tcr preserve_direct_calls e
    = false
    ∧ uses_fit (unit_counts u)
        (unit_cont_uses T cannot_change code_deps unboxed e)
      = true
    ∧ calls_fit F T cannot_change code_deps unboxed e = true
    ∧ codes_grouped code_deps e = true.

(* The equality is finer than [results_within_callees]. *)
Definition results_match_within_statement : Prop :=
  ∀ e : expr,
    results_match_callees e = true -> results_within_callees e = true.

(* Each premise fails of the unit of FlambdaReaperFixFailsAllSpec.v that
   it excludes, and holds of the other two. *)
Definition premises_exclude_units_statement : Prop :=
  let e1 := FixFailsAllExamples.more_results in
  let e2 := FixFailsAllExamples.block_callee in
  let e3 := FixFailsAllExamples.no_parameter in
  let T1 := CalleesExample.tables_of e1 in
  let T2 := CalleesExample.tables_of e2 in
  let T3 := CalleesExample.tables_of e3 in
  let F1 := CalleesExample.facts_of e1 in
  let F2 := CalleesExample.facts_of e2 in
  let F3 := CalleesExample.facts_of e3 in
  results_match_callees e1 = false
  ∧ unit_codes_have_params e1 = true
  ∧ known_callees_have_code F1 T1 e1 = true
  ∧ results_match_callees e2 = true
  ∧ unit_codes_have_params e2 = true
  ∧ known_callees_have_code F2 T2 e2 = false
  ∧ results_match_callees e3 = true
  ∧ unit_codes_have_params e3 = false
  ∧ known_callees_have_code F3 T3 e3 = true.

(* Every call has as many arguments as its arity of arguments has
   components. *)
Definition call_args_fit (e : expr) : bool :=
  forallb
    (fun ap =>
       Nat.eqb (length (ap_args ap)) (length (unarize (ap_args_arity ap))))
    (calls_of e).

Definition call_args_grouped_statement : Prop :=
  ∀ e : expr,
    call_args_fit e = true ->
    ∀ ap : apply_expr,
      ap ∈ calls_of e ->
      concat (group_by_parameter (ap_args_arity ap) (ap_args ap))
      = ap_args ap.
