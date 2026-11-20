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

(* The fragment of the refinement across calls with code that never
   returns (FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr) asks
   of each code [id] that it takes to never return ([nr id]) that its
   return continuation is not live in its body ([calls_code]). Here [nr]
   is the code that is used and some result of which has no source
   ([used_never_returns]), and that check is derived. The same of the
   fragment across changes of representation
   (FlambdaReaperUnboxingRefinesClosuresSpec.repr_unboxing_nr, [RCS] below,
   the check of [repr_code]).

   This file is to be reviewed: [calls_unchecked_expr] against
   FlambdaReaperUnboxingRefinesCallsSpec.calls_expr, [repr_unchecked_expr]
   against RCS.repr_expr, [used_never_returns], [keys_and_groups_ok] and
   the statements. The proofs are in FlambdaReaperNeverReturnsProofs.v,
   which needs no review.

   [calls_unchecked_expr] is [calls_expr] but for the last conjunct of
   [calls_code], the check of [live_conts]. [calls_unboxing_unchecked] is
   [calls_unboxing_nr] with [calls_unchecked_expr] for [calls_expr].
   [repr_unchecked_expr] is [RCS.repr_expr] but for the last conjunct of
   [RCS.repr_code], the same check; [repr_unboxing_unchecked] is
   [RCS.repr_unboxing_nr] with it.

   [never_returns_derived_observably_statement]: the
   premises of [rebuild_unbox_fixed_observably_never_returns_statement]
   (FlambdaReaperUnboxingRefinesCallsSpec.v) with the flag [Auto], [nr]
   the code of [used_never_returns], the fragment without the check,
   tables that are a least solution, and a check of the term
   ([keys_and_groups_ok]); the rebuilt unit refines the unit up to
   observation.
   [never_returns_derived_refines_cc_statement]: the
   same for the refinement of the change of calling conventions, with
   the exception handlers of the unit well formed.
   [never_returns_derived_local_statement]: the premises of
   [rebuild_unbox_fixed_local_never_returns_statement] in the same way, the
   tables a least solution of the rules with the fix of local fields
   ([least_solution_local]); the checks that such tables pass are not
   asked ([check_closed], [check_local_reads_used],
   [check_field_used_exact], [check_apply_widgets]).
   [never_returns_derived_example_statement]: the unit of
   FlambdaReaperUnboxingRefinesCallsSpec.never_returns_example_statement,
   with the tables of the least solution, meets the premises of the first,
   its code 50 taken to never return.
   [repr_never_returns_derived_observably_statement],
   [repr_never_returns_derived_local_statement] and
   [repr_never_returns_derived_refines_cc_statement]: the same of
   RCS.rebuild_repr_fixed_observably_never_returns_statement,
   RCS.rebuild_repr_fixed_local_never_returns_statement and
   RCS.rebuild_repr_fixed_refines_cc_never_returns_statement, with
   [repr_unboxing_unchecked]; the checks that a least solution passes are
   not asked. [repr_never_returns_derived_example_statement]: the unit of
   RCS.repr_never_returns_example_statement, with its tables, meets the
   premises of the first.
   [never_returns_local_cc_statement] and
   [repr_never_returns_local_cc_statement]: the two [_refines_cc] forms
   with tables that are a least solution with the fix of local fields;
   [check_closed] is not asked.

   Why the return continuation [ret] of such code is not live in its body
   (FlambdaReaperDeadFramesSpec.v for the first):
   - An [Apply_cont] to [ret], or an arm, gives a name with no source
     ([dead_return_dropped_statement]).
   - A call that returns to [ret] gives it its results, which flow into
     those of the code where it is used; the call never returns
     ([apply_never_returns]): its code is known to call code a result of
     which has no source, or the witnesses that it reads give no source
     at that rank. With [Auto] and a least solution the witness read off
     the callee is that of the code it is known to call
     (FlambdaReaperWrapperDerivedProofs.call_never_returns).
   - No call raises to [ret]: [well_formed_unit] (or
     [exn_handlers_well_formed]) asks it of the free continuations of the
     body of code other than its exception continuation.
   - [live_conts] counts every arm of a switch, the traversal only those
     that [Keyed.compact] keeps: [keys_and_groups_ok] asks that the keys
     of a switch are distinct.
   - A call of unknown arity with no group of arguments reads no witness
     ([never_returns_of_call] is [false]) and may return to [ret] where
     its results have no source: [keys_and_groups_ok] asks a group of
     every call of unknown arity, as the compiler's terms have.
   - A call of known code expects of it the results its code has: the
     row of [call_ok] on the return arity.
   Code that is not used is left out
   (FlambdaReaperDeadFramesSpec.unused_never_returns_example_statement). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields FlambdaReaperLeastSpec
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxingSpec
  FlambdaReaperUnboxingRefinesClosuresSpec.
Import ListNotations.

Module RCS := FlambdaReaperUnboxingRefinesClosuresSpec.

(* ======================================================================== *)
(* The fragment without the check                                           *)
(* ======================================================================== *)

Section Unchecked.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.

Local Abbreviation handler_pdescs :=
  (FlambdaReaperUnboxingRefinesCallsSpec.handler_pdescs sol unboxed).
Local Abbreviation call_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.call_ok F sol cannot_change
     code_deps unit_code_ids codes unboxed nr).
Local Abbreviation apply_cont_ok2 :=
  (FlambdaReaperUnboxingRefinesCallsSpec.apply_cont_ok2 sol unboxed).
Local Abbreviation code_pdescs :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs sol cannot_change
     code_deps unit_code_ids unboxed).
Local Abbreviation code_param_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_param_ok unboxed).
Local Abbreviation code_exn_decision :=
  (FlambdaReaperUnboxingRefinesCallsSpec.code_exn_decision sol code_deps
     unboxed).

(* [calls_expr], but for the check of [live_conts] in [calls_code]. *)
Fixpoint calls_unchecked_expr (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x with
       | Some fields => unboxed_binding_ok unboxed sol kinds x kind d fields
       | None =>
         plain_binding_ok unboxed x kind d || closure_binding_ok x d body
       end
    && calls_unchecked_expr ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb calls_unchecked_binding bindings
    && calls_unchecked_expr kinds C body
  | Let_cont_non_recursive k h body =>
    let ps := handler_pdescs (handler_is_exn_handler h) h in
    calls_unchecked_handler kinds C ps h
    && calls_unchecked_expr kinds ((k, (ps, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map (fun '(k, h) => (k, (handler_pdescs false h, true))) handlers ++ C
    in
    forallb
      (fun '(_, h) =>
         calls_unchecked_handler kinds C' (handler_pdescs false h) h)
      handlers
    && calls_unchecked_expr kinds C' body
  | Apply ap => call_ok kinds C ap
  | Apply_cont ac => apply_cont_ok2 kinds C ac
  | Switch scrutinee arms =>
    plain_simple unboxed scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok2 kinds C ac) arms
  | Invalid => true
  end
with calls_unchecked_handler (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ps : list pdesc)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb pdesc_ok ps
    && calls_unchecked_expr
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with calls_unchecked_binding (binding : static_binding) {struct binding}
    : bool :=
  match binding with
  | Bind_code id c => calls_unchecked_code id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => local_binding unboxed binding
  end
with calls_unchecked_code (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match code_pdescs id metadata with
       | Some (pds, rds) =>
         Nat.eqb (length params) (length pds)
         && Nat.eqb (length pds) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rds) (length (unarize (cm_result_arity metadata)))
         && forallb pdesc_ok pds && forallb pdesc_ok rds
         && forallb2 code_param_ok params pds
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && plain_simple unboxed (Var my_closure)
         && negb (Pos.eqb ret exn)
         && match code_exn_decision id with
            | Some d =>
              calls_unchecked_expr
                (rev ((my_closure, Value)
                      :: with_kinds params
                           (unarize (cm_params_arity metadata))))
                [(ret, (rds, true)); (exn, ([Pdesc d true Value], false))]
                body
            | None => false
            end
       | None => false
       end
  end.

End Unchecked.

(* [calls_unboxing_nr] without the check of [live_conts]. *)
Definition calls_unboxing_unchecked (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  calls_unchecked_expr F (solution_of_tables T)
    (fun n => mem_node n cannot_change) code_deps
    (static_names code_ids_bound e) (bound_code_metadata e) unboxed nr [] []
    e
  && check_unboxed_uses F T unboxed.

Section ReprUnchecked.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.

Local Abbreviation changed_name := (RCS.changed_name changed).
Local Abbreviation plain_c := (RCS.plain_c unboxed changed).
Local Abbreviation no_changed := (RCS.no_changed changed).
Local Abbreviation plain_set_ok := (RCS.plain_set_ok unboxed changed).
Local Abbreviation plain_slot_read_ok :=
  (RCS.plain_slot_read_ok unboxed changed).
Local Abbreviation unboxed_slot_read_ok :=
  (RCS.unboxed_slot_read_ok unboxed changed).
Local Abbreviation changed_set_ok := (RCS.changed_set_ok F sol unboxed changed).
Local Abbreviation apply_cont_ok_c := (RCS.apply_cont_ok_c unboxed changed).
Local Abbreviation repr_call :=
  (RCS.repr_call F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr).

(* [RCS.repr_expr], but for the check of [live_conts] in [repr_code]. *)
Fixpoint repr_unchecked_expr (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields && no_changed d)
         || unboxed_slot_read_ok x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed d)
         || plain_set_ok x d body || plain_slot_read_ok d
       | None, Some r => changed_set_ok x kind d body r
       | Some _, Some _ => false
       end
    && repr_unchecked_expr ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_unchecked_binding bindings && repr_unchecked_expr kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_unchecked_handler kinds C ds h
    && repr_unchecked_expr kinds ((k, (ds, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map
        (fun '(k, h) =>
           (k, (handler_decisions true sol unboxed false h, true)))
        handlers
      ++ C
    in
    forallb
      (fun '(_, h) =>
         repr_unchecked_handler kinds C'
           (handler_decisions true sol unboxed false h) h)
      handlers
    && repr_unchecked_expr kinds C' body
  | Apply ap => repr_call C ap
  | Apply_cont ac => apply_cont_ok_c C ac
  | Switch scrutinee arms =>
    plain_c scrutinee && forallb (fun '(_, ac) => apply_cont_ok_c C ac) arms
  | Invalid => true
  end
with repr_unchecked_handler (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb RCS.decided ds
    && forallb (fun x => plain_c (Var x)) params
    && repr_unchecked_expr
         (rev (combine params (hm_params_kinds metadata)) ++ kinds) C handler
  end
with repr_unchecked_binding (binding : static_binding) {struct binding}
    : bool :=
  match binding with
  | Bind_code id c => repr_unchecked_code id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_unchecked_code (id : code_id) (c : code) {struct c} : bool :=
  match c with
  | Code params my_closure ret exn metadata body =>
    match lookup id codes with
    | Some metadata' => code_metadata_eqb metadata metadata'
    | None => false
    end
    && match lookup id my_closures with
       | Some mc => Pos.eqb mc my_closure
       | None => false
       end
    && match
         code_change_u sol cannot_change code_deps unit_code_ids unboxed id
           metadata
       with
       | Changing_u pd rd =>
         forallb RCS.decided pd && forallb RCS.decided rd
         && Nat.eqb (length params) (length pd)
         && Nat.eqb (length pd) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
         && forallb (fun x => plain_c (Var x)) params
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && is_none (fields_of unboxed my_closure)
         && negb (Pos.eqb ret exn)
         && repr_unchecked_expr
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true)); (exn, ([], false))] body
       | Not_changing_u => false
       end
  end.

End ReprUnchecked.

(* [RCS.repr_unboxing_nr] without the check of [live_conts]. *)
Definition repr_unboxing_unchecked (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_unchecked_expr F (solution_of_tables T)
    (fun n => mem_node n cannot_change) code_deps
    (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr [] [] e
  && check_unboxed_uses F T unboxed.

(* The code taken to never return: code that is used, some result of
   whose code dep has no source. *)
Definition used_never_returns (sol : solution)
    (code_deps : finite_map code_dep) (id : code_id) : bool :=
  code_never_returns sol code_deps id && s_any_usage sol (Node_code_id id).

(* The check of the term: the keys of each switch are distinct, and each
   call of unknown arity has a group of arguments, in the bodies of code
   too. *)
Fixpoint keys_and_groups_ok (e : expr) : bool :=
  match e with
  | Let _ _ _ body => keys_and_groups_ok body
  | Let_static bindings body =>
    forallb binding_keys_and_groups_ok bindings && keys_and_groups_ok body
  | Let_cont_non_recursive _ h body =>
    handler_keys_and_groups_ok h && keys_and_groups_ok body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_keys_and_groups_ok h) handlers
    && keys_and_groups_ok body
  | Apply ap =>
    match ap_call_kind ap, ap_args_arity ap with
    | Indirect_unknown_arity, [] => false
    | _, _ => true
    end
  | Switch _ arms => Keyed.distinct Z.eqb (map fst arms)
  | Apply_cont _ | Invalid => true
  end
with handler_keys_and_groups_ok (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => keys_and_groups_ok handler
  end
with binding_keys_and_groups_ok (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ (Code _ _ _ _ _ body) => keys_and_groups_ok body
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition never_returns_derived_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    keys_and_groups_ok e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    check_field_used_exact is_local (traverse_unit Auto u e) T = true ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true ->
    calls_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      (traverse_unit Auto u e) T cannot_change (unit_code_deps Auto u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) dominated to_unbox preserve_direct_calls e).

Definition never_returns_derived_refines_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    exn_handlers_well_formed e = true ->
    keys_and_groups_ok e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_closed is_local (traverse_unit Auto u e) T = true ->
    calls_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      (traverse_unit Auto u e) T cannot_change (unit_code_deps Auto u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) dominated to_unbox preserve_direct_calls e).

Definition never_returns_derived_local_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    keys_and_groups_ok e = true ->
    least_solution_local is_local (traverse_unit Auto u e) T ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    calls_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      (traverse_unit Auto u e) T cannot_change (unit_code_deps Auto u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) dominated to_unbox preserve_direct_calls e).

Definition repr_never_returns_derived_observably_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    keys_and_groups_ok e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    repr_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition repr_never_returns_derived_local_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    keys_and_groups_ok e = true ->
    least_solution_local is_local (traverse_unit Auto u e) T ->
    check_cc_closed (traverse_unit Auto u e) T cannot_change = true ->
    repr_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition repr_never_returns_derived_refines_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    exn_handlers_well_formed e = true ->
    keys_and_groups_ok e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    repr_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The two [_refines_cc] forms with the fix of local fields. *)
Definition never_returns_local_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    exn_handlers_well_formed e = true ->
    keys_and_groups_ok e = true ->
    least_solution_local is_local (traverse_unit Auto u e) T ->
    calls_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      (traverse_unit Auto u e) T cannot_change (unit_code_deps Auto u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) dominated to_unbox preserve_direct_calls e).

Definition repr_never_returns_local_cc_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    exn_handlers_well_formed e = true ->
    keys_and_groups_ok e = true ->
    least_solution_local is_local (traverse_unit Auto u e) T ->
    repr_unboxing_unchecked
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      is_local (traverse_unit Auto u e) T cannot_change
      (unit_code_deps Auto u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit Auto u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T cannot_change
         (unit_code_deps Auto u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit Auto u e) T
         cannot_change (unit_code_deps Auto u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesCallsSpec.NeverExample, whose
   code 50 calls itself through its closure in tail position and is
   called by the unit, with the tables of the least solution: the code 50
   is of [used_never_returns], the unit is of the fragment without the
   check and passes the check of the term, so that
   [never_returns_derived_observably_statement]
   applies to it. *)
Definition never_returns_derived_example_statement : Prop :=
  let F := NeverExample.facts in
  let T := NeverExample.tables in
  let cds := NeverExample.code_deps in
  let e := NeverExample.unit_expr in
  least_solution FlambdaReaperUnboxing.Examples.no_local F T
  ∧ keys_and_groups_ok e = true
  ∧ used_never_returns (solution_of_tables T) cds 50%positive = true
  ∧ calls_unboxing_unchecked (used_never_returns (solution_of_tables T) cds)
      F T NeverExample.cannot_change cds [] [] e
    = true
  ∧ program_refines_observably e
      (rebuild_unit_unbox_fixed F T NeverExample.cannot_change cds [] []
         true e).

(* The unit of RCS.repr_never_returns_example_statement, whose code 50
   reads a slot of its closure, whose set changes its representation, and
   calls itself in tail position, with the tables of the least solution:
   50 is of [used_never_returns], the unit is of the closures fragment
   without the check and passes the check of the term, so that
   [repr_never_returns_derived_observably_statement] applies to it. *)
Definition repr_never_returns_derived_example_statement : Prop :=
  let F := RCS.ReprNeverExample.facts in
  let T := RCS.ReprNeverExample.tables in
  let cds := RCS.ReprNeverExample.code_deps in
  let dom := RCS.ReprNeverExample.dominated in
  let to_unbox := RCS.ReprNeverExample.to_unbox in
  let tcr := RCS.ReprNeverExample.tcr in
  let e := RCS.ReprNeverExample.unit_expr in
  least_solution FlambdaReaperUnboxing.Examples.no_local F T
  ∧ keys_and_groups_ok e = true
  ∧ used_never_returns (solution_of_tables T) cds 50%positive = true
  ∧ repr_unboxing_unchecked (used_never_returns (solution_of_tables T) cds)
      FlambdaReaperUnboxing.Examples.no_local F T [] cds dom to_unbox tcr e
    = true
  ∧ program_refines_observably e
      (rebuild_unit_unbox_repr_fixed FlambdaReaperUnboxing.Examples.no_local
         F T [] cds dom to_unbox tcr true e).
