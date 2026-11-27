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


(* The rebuild with unboxing refines its unit across the changes of
   representation of sets of closures, on a fragment wider than that of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v: beside the sets whose
   representation changes, calls whose code is not known, of known arity
   or of unknown arity, that return or that never return, where the flow
   of the first rebuilt unit brings to the callee only code whose
   [my_closure] does not change.

   This file is to be reviewed: the fragment ([repr_reach_unboxing_nr],
   [repr_flow_unboxing_nr]), the statements and the example, against
   FlambdaReaperUnboxing.v, FlambdaReaperUnboxingRefinesClosuresCallsSpec.v
   and FlambdaReaperUnboxingRefinesReach2Spec.v. The proofs are in
   FlambdaReaperUnboxingRefinesClosuresReachProofs.v and
   FlambdaReaperUnboxingRefinesClosuresProofs.v, which need no review.

   A call whose code is not known calls the code of whatever closure its
   callee holds. A plain name holds the same value in both rebuilt units,
   never a closure in the new representation of its set (a changed name
   flows to no plain name), so the code called reads its [my_closure] as
   the first rebuild does only if that [my_closure] does not change. The
   flow says which code may be called: that of [flow_reach2] of the
   callee in the first rebuilt unit, analysed again ([F1], [T1]), as in
   FlambdaReaperUnboxingRefinesReach2Spec.v. Code whose [my_closure]
   changes, called so, would read the value slots of a closure in the
   old representation at the new slots.

   [repr_reach_holds_calls_statement]: the fragment holds that of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v
   ([repr_calls_unboxing_nr]), for the same tables and the same code taken
   to never return, whatever the flow.

   [rebuild_repr_reach_refines_cc_statement],
   [rebuild_repr_reach_observably_statement] and
   [rebuild_repr_reach_local_statement]: the statements
   [rebuild_repr_calls_refines_cc_statement],
   [rebuild_repr_calls_observably_statement] and
   [rebuild_repr_calls_local_statement] of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v with
   [repr_flow_unboxing_nr] in place of [repr_calls_unboxing_nr], and the
   premises of FlambdaReaperUnboxingRefinesReach2Spec.v on the first
   rebuilt unit, whose run the flow describes: it is of the fragment of
   the heap ([in_heap_fragment]), and its tables [T1] pass [check_closed]
   and [check_local_reads_used]. The statements of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v are corollaries.

   [rebuild_repr_reach_with_types_refines_cc_statement],
   [rebuild_repr_reach_with_types_local_statement] and
   [rebuild_repr_reach_with_types_observably_statement]: the same for the
   rebuilds of the reaper with types, as
   [rebuild_repr_calls_with_types_refines_cc_statement] and the two after
   it, the flow read off the first rebuilt unit with types
   (FlambdaReaperTypesSpec.first_rebuild_with_types).

   The fragment ([repr_reach_unboxing_nr], for the code [reach b f] that
   the flow may bring to the callee [f] of a call, of known arity if [b])
   is that of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v
   ([repr_calls_unboxing_nr]) with one more kind of call ([repr_call_r]):
   a call whose code is not known ([ind_call_r]: not direct, with no code
   id that [code_id_actually_called] gives), of a variable [f] that is
   neither unboxed nor changed. Its arguments and the extra arguments of
   its exception continuation are plain, as for the calls of
   [repr_call_w]; both rebuilds give the arguments at the mask of the call
   read off the callee ([call_mask], of the solution and of the solution
   without the unboxed names), which must be the same. A call of known
   arity asks that the code that the flow may bring to [f] ([reach true
   f]) have its [my_closure] plain ([plain_my_closure]); a call of unknown
   arity, that it have one group of arguments, and that the code that the
   flow may bring to the closure of what [f] holds ([reach false f],
   partial applications included) have its [my_closure] plain and a
   parameter, so that the call over-applies nothing: the partial
   applications are those of the machine, given fewer groups of arguments
   than their code has parameters. A call that never returns
   ([apply_never_returns]) asks that code be code of [nr], taken to never
   return, with one parameter for a call of unknown arity, so that the
   call is a full call: the fix makes it return to a handler [Invalid],
   and it may return anywhere in the first rebuild, through a wrapper or
   not. Any other call returns to a continuation that keeps every result,
   or to a way out of the unit, so that neither rebuild makes a wrapper.

   What is left out, for the proof: calls of unknown arity with more than
   one group of arguments or of code with no parameter (they may
   over-apply, and what is left of an over-application calls a result,
   whose code the flow of a variable does not give), wrappers of calls
   that return, unboxed parameters, extra
   arguments for an exception continuation that is not a handler of the
   term, projections of other closures of a set, static sets whose
   representation changes, and closures that are unboxed.

   [repr_reach_ind_call_example_statement]: a unit of
   [repr_flow_unboxing_nr] that is not of [repr_calls_unboxing_nr]
   ([ClosuresReachExample]): the set of [x60] changes its representation
   beside a call of known arity whose code is not known, of the closure of
   56 or of 57; and its rebuild in the mode of the fix refines it up to
   observation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeast FlambdaReaperLeastSpec FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaStaticsInScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
(* The [my_closure] of each code of the unit. *)
Variable my_closures : finite_map variable.
(* The code taken to never return. *)
Variable nr : code_id -> bool.
(* The code that the flow may bring to the callee [f] of a call: to what
   [f] holds, for a call of known arity ([reach true f]); to the closure
   that a call of what [f] holds calls, of a partial application or of
   [f] itself ([reach false f]). *)
Variable reach : bool -> variable -> code_id -> bool.

Local Abbreviation plain_c := (plain_c unboxed changed).
Local Abbreviation changed_name := (changed_name changed).

(* The [my_closure] of the code [id] of the unit does not change. *)
Definition plain_my_closure (id : code_id) : bool :=
  match lookup id my_closures with
  | Some mc => is_none (changed_name (Node_variable mc))
  | None => false
  end.

(* A call [ap] of [f] whose code is not known: [f] neither unboxed nor
   changed; plain arguments, at the same mask in both rebuilds; and the
   code that the flow may bring to it has its [my_closure] plain (and,
   for a call of unknown arity, which has one group of arguments, a
   parameter). If the call never returns ([apply_never_returns]), that
   code is code of [nr] (with one parameter, for a call of unknown arity)
   and the call may return anywhere, as the fix makes it return to a
   handler [Invalid]; otherwise its continuation keeps every result, so
   that neither rebuild makes a wrapper. *)
Definition ind_call_r (C : finite_map (list decision * bool))
    (ap : apply_expr) (f : variable) : bool :=
  let never := apply_never_returns F sol code_deps codes ap in
  is_none (fields_of unboxed f) && is_none (changed_name (Node_variable f))
  && forallb plain_c (ap_args ap)
  && forallb2 Bool.eqb (call_mask F sol f ap)
       (call_mask F (solution_without_unboxed sol unboxed) f ap)
  && extra_args_ok unboxed changed C ap
  && (never
      || returns_ok C (ap_continuation ap)
           (map (fun kind => Decided (Keep kind))
              (unarize (ap_return_arity ap))))
  && match ap_call_kind ap with
     | Direct _ => false
     | Indirect_known_arity _ =>
       forallb
         (fun '(id, _) =>
            negb (reach true f id)
            || plain_my_closure id && (negb never || nr id))
         codes
     | Indirect_unknown_arity =>
       Nat.eqb (length (ap_args_arity ap)) 1
       && forallb
            (fun '(id, md) =>
               negb (reach false f id)
               || plain_my_closure id
                  && negb (Nat.eqb (length (cm_params_arity md)) 0)
                  && (negb never
                      || nr id && Nat.eqb (length (cm_params_arity md)) 1))
            codes
     end.

(* A call of [repr_call_w], or one whose code is not known. *)
Definition repr_call_r (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  repr_call_w F sol cannot_change code_deps unit_code_ids codes unboxed
    changed my_closures nr C ap
  || match ap_callee ap, code_id_actually_called F sol ap with
     | Var f, None => ind_call_r C ap f
     | _, _ => false
     end.

(* [repr_expr_w], [repr_handler_w], [repr_binding_w] and [repr_code_w] of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v, with [repr_call_r]. *)
Fixpoint repr_expr_r (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (e : expr) {struct e} : bool :=
  match e with
  | Let x kind d body =>
    kind_ok kind
    && match fields_of unboxed x, changed_name (Node_variable x) with
       | Some fields, None =>
         (unboxed_binding_ok unboxed sol kinds x kind d fields
          && no_changed changed d)
         || unboxed_slot_read_ok unboxed changed x kind d fields
       | None, None =>
         (plain_binding_ok unboxed x kind d && no_changed changed d)
         || plain_set_ok unboxed changed x d body
         || plain_slot_read_ok unboxed changed d
       | None, Some r => changed_set_ok F sol unboxed changed x kind d body r
       | Some _, Some _ => false
       end
    && repr_expr_r ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_r bindings && repr_expr_r kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_r kinds C ds h
    && repr_expr_r kinds ((k, (normed ds, true)) :: C) body
  | Let_cont_recursive handlers body =>
    let C' :=
      map
        (fun '(k, h) =>
           (k, (normed (handler_decisions true sol unboxed false h), true)))
        handlers
      ++ C
    in
    forallb
      (fun '(_, h) =>
         repr_handler_r kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_r kinds C' body
  | Apply ap => repr_call_r C ap
  | Apply_cont ac => apply_cont_ok_w unboxed changed C ac
  | Switch scrutinee arms =>
    plain_c scrutinee
    && forallb (fun '(_, ac) => apply_cont_ok_w unboxed changed C ac) arms
  | Invalid => true
  end
with repr_handler_r (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided (normed ds)
    && forallb (fun x => plain_c (Var x)) params
    && repr_expr_r (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         C handler
  end
with repr_binding_r (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_r id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_code_r (id : code_id) (c : code) {struct c} : bool :=
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
    && forallb (fun x => plain_c (Var x)) params
    && is_none (fields_of unboxed my_closure)
    && negb (Pos.eqb ret exn)
    && match
         code_change_u sol cannot_change code_deps unit_code_ids unboxed id
           metadata
       with
       | Changing_u pd rd =>
         forallb decided pd && forallb decided rd
         && Nat.eqb (length params) (length pd)
         && Nat.eqb (length pd) (length (unarize (cm_params_arity metadata)))
         && Nat.eqb (length rd) (length (unarize (cm_result_arity metadata)))
         && is_none
              (my_closure_decision_u cannot_change code_deps unit_code_ids
                 unboxed id)
         && repr_expr_r
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true));
               (exn, (code_exn_ds sol code_deps unboxed id, false))]
              body
       | Not_changing_u =>
         negb (cm_is_tupled metadata)
         && match lookup id code_deps with
            | Some cd =>
              let rds :=
                normed
                  (fixed_arity_decisions sol unboxed (cd_return cd)
                     (unarize (cm_result_arity metadata)))
              in
              forallb decided rds
              && repr_expr_r
                   (rev ((my_closure, Value)
                         :: with_kinds params
                              (unarize (cm_params_arity metadata))))
                   [(ret, (rds, true));
                    (exn, (code_exn_ds sol code_deps unboxed id, false))]
                   body
            | None => false
            end
       end
    && (negb (nr id)
        || negb (among ret (live_conts F sol code_deps codes body)))
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing and the changes of
   representation that the tables give from its graph [F], the code [nr]
   taken to never return, and the code [reach] that the flow may bring to
   the callee of a call. *)
Definition repr_reach_unboxing_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (reach : bool -> variable -> code_id -> bool) (e : expr) : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr_r F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr reach [] [] e
  && check_unboxed_uses F T unboxed.

(* The flow being read off the facts [F1] and the tables [T1] of the first
   rebuilt unit (FlambdaReaperUnboxingRefinesReach2Spec.flow_reach2). *)
Definition repr_flow_unboxing_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (F1 : facts) (T1 : tables) (e : expr) : bool :=
  repr_reach_unboxing_nr nr is_local F T cannot_change code_deps dominated
    to_unbox tcr (flow_reach2 F1 T1) e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition repr_reach_holds_calls_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (reach : bool -> variable -> code_id -> bool)
    (e : expr),
    repr_calls_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr e
    = true ->
    repr_reach_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr reach e
    = true.

Definition rebuild_repr_reach_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_reach_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_reach_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* ======================================================================== *)
(* With types                                                               *)
(* ======================================================================== *)

(* The rebuild with unboxing and changes of representation of the reaper
   with types in the mode of the fix refines the rebuild with types and
   changes of calling convention, with no set of closures given to the
   rewriting, on the wider fragment, with the premises of
   [rebuild_repr_calls_with_types_refines_cc_statement]
   (FlambdaReaperUnboxingRefinesClosuresCallsSpec.v) and the flow read off
   the first rebuilt unit with types. *)
Definition rebuild_repr_reach_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The unit rebuilt with unboxing and changes of representation by the
   reaper with types refines the unit up to observation, with the
   premises of [rebuild_repr_calls_with_types_local_statement] but the
   fragment, and the flow read off the first rebuilt unit with types. *)
Definition rebuild_repr_reach_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The same without local fields. *)
Definition rebuild_repr_reach_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
         cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1
    = true ->
    repr_flow_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
      (traverse_unit flag u
         (FlambdaReaperTypesSpec.first_rebuild_with_types flag is_local u e T
            cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperUnboxing.Examples.

(* The unit of [TwoFieldsExample] (the code 54 reads the block of two
   fields in the slot 80 of its closure [x60], whose set changes its
   representation, and adds them and its parameter), with the closures
   [x61] of the code 56, which adds 1 to its parameter, and [x62] of the
   code 57, which adds 2. The unit calls 54 with 32; the handler 20 of its
   result [x30] jumps to the handler 21 with [x61] or [x62], in an arm that
   the run does not take for the second; 21 calls its parameter [x31] with
   [x30], a call of known arity whose code is not known, of 56 or of 57,
   and the handler 22 of the result returns the block of the result and of
   [x31]. The two closures escape, so their sets keep their representation
   and 56 and 57 their calling convention. *)
Module ClosuresReachExample.

Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition code_56 : code :=
  Code [42] 43 5 6 metadata
    (Let 44 Value (TwoFieldsExample.add (Var 42) (int 1))
       (return_to 5 [Var 44])).

Definition code_57 : code :=
  Code [46] 47 7 8 metadata
    (Let 48 Value (TwoFieldsExample.add (Var 46) (int 2))
       (return_to 7 [Var 48])).

Definition call (callee : variable) (kind : call_kind) (args : list simple)
    (k : continuation) : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := kind; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

(* The call whose code is not known, of the kind [kind]. *)
Definition call_31 (kind : call_kind) : apply_expr :=
  call 31 kind [Var 30] 22.

(* The closures [x61] and [x62] and the handlers, the call of 31 being of
   the kind [kind] and that of 54 [call_54]. *)
Definition tail (kind : call_kind) (call_54 : expr) : expr :=
  Let 61 Value (Closure_of_set 71 [(71, 56)] [])
    (Let 62 Value (Closure_of_set 72 [(72, 57)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let_cont_non_recursive 21
                (Handler [31] (make_handler_metadata [Value])
                   (Let_cont_non_recursive 22
                      (Handler [32] (make_handler_metadata [Value])
                         (Let 33 Value (make_block [Var 32; Var 31])
                            (return_to 1 [Var 33])))
                      (Apply (call_31 kind))))
                (Switch (Const (Naked_immediate 0))
                   [(0%Z, jump 21 [Var 61]); (1%Z, jump 21 [Var 62])])))
          call_54)).

Definition unit_expr : expr :=
  Let_static
    [Bind_code 54 TwoFieldsExample.code_54; Bind_code 56 code_56;
     Bind_code 57 code_57]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
          (tail (Indirect_known_arity None)
             (Apply (call 60 (Direct 54) [int 32] 20))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

(* The table of the rules for [cannot_change_calling_convention]. *)
Definition cannot_change : list node := derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The first rebuilt unit, and its analysis. *)
Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match FlambdaReaperLeast.compute_least no_local facts1 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

(* The rebuild: 54 reads the new slots 81 and 82 of its closure, at the new
   function slot 83, into the variables 63 and 64 of the fields of [x45];
   the call of [x31] lists the codes 57 and 56; the rest is as it was. *)
Definition changed_54 : code :=
  Code [40] 41 3 4
    {| cm_params_arity := [Unboxed_product [Singleton Value]];
       cm_params_subkinds := [];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_result_subkinds := [];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (Let 64 Value (Project_value_slot_from 83 82 (Var 41))
       (Let 63 Value (Project_value_slot_from 83 81 (Var 41))
          (Let 46 Value (Simple (Var 63))
             (Let 47 Value (Simple (Var 64))
                (Let 48 Value (TwoFieldsExample.add (Var 46) (Var 47))
                   (Let 49 Value (TwoFieldsExample.add (Var 48) (Var 40))
                      (return_to 3 [Var 49]))))))).

Definition rebuilt : expr :=
  Let_static
    [Bind_code 54 changed_54; Bind_code 56 code_56; Bind_code 57 code_57]
    (Let 66 Value (Simple (int 8))
       (Let 65 Value (Simple (int 7))
          (Let 60 Value
             (Closure_of_set 83 [(83, 54)] [(81, Var 65); (82, Var 66)])
             (tail (Indirect_known_arity (Some [57; 56]))
                (Apply
                   {| ap_callee := Var 60; ap_call_kind := Direct 54;
                      ap_args := [int 32]; ap_continuation := 20;
                      ap_exn_continuation := 2; ap_exn_extra_args := [];
                      ap_args_arity := [Unboxed_product [Singleton Value]];
                      ap_return_arity :=
                        [Unboxed_product [Singleton Value]] |}))))).

(* What the unit and its rebuild return: 7 + 8 + 32 + 1, and [x61]. *)
Definition result : list value :=
  [Value_block 0
     [Value_const (Tagged_immediate 48);
      Value_closure_of_set 71 [(71, 56)] []]].

End ClosuresReachExample.

(* With the tables of the least solution, of the unit and of the first
   rebuilt unit: the set of [x60] changes its representation and those of
   [x61] and [x62] do not; 56 and 57 cannot change their calling
   convention; the call of [x31] has no code that the solution gives, and
   the flow brings 56 and 57 to [x31], not 54. The unit is of
   [repr_flow_unboxing_nr], not of [repr_calls_unboxing_nr]; the rebuild in
   the mode of the fix is [rebuilt]; both return [result]; and the rebuilt
   unit refines the unit up to observation. *)
Definition repr_reach_ind_call_example_statement : Prop :=
  check_dominated ClosuresReachExample.tables ClosuresReachExample.dominated
  = true
  ∧ check_to_unbox no_local ClosuresReachExample.facts
      ClosuresReachExample.tables [] max_unbox_size false
      ClosuresReachExample.to_unbox
    = true
  ∧ check_to_change_representation no_local ClosuresReachExample.facts
      ClosuresReachExample.tables [] max_unbox_size false
      ClosuresReachExample.to_unbox ClosuresReachExample.tcr
    = true
  ∧ check_sets_whole ClosuresReachExample.facts ClosuresReachExample.tables
      ClosuresReachExample.to_unbox ClosuresReachExample.tcr
    = true
  ∧ ClosuresReachExample.to_unbox = [v 10]
  ∧ mem_node (v 60) ClosuresReachExample.tcr = true
  ∧ mem_node (v 61) ClosuresReachExample.tcr = false
  ∧ mem_node (v 62) ClosuresReachExample.tcr = false
  ∧ mem_node (c 56) ClosuresReachExample.cannot_change = true
  ∧ mem_node (c 57) ClosuresReachExample.cannot_change = true
  ∧ code_id_actually_called ClosuresReachExample.facts
      (solution_of_tables ClosuresReachExample.tables)
      (ClosuresReachExample.call_31 (Indirect_known_arity None))
    = None
  ∧ flow_reach2 ClosuresReachExample.facts1 ClosuresReachExample.tables1
      true 31%positive 56%positive
    = true
  ∧ flow_reach2 ClosuresReachExample.facts1 ClosuresReachExample.tables1
      true 31%positive 57%positive
    = true
  ∧ flow_reach2 ClosuresReachExample.facts1 ClosuresReachExample.tables1
      true 31%positive 54%positive
    = false
  ∧ repr_calls_unboxing_nr (fun _ => false) no_local
      ClosuresReachExample.facts ClosuresReachExample.tables
      ClosuresReachExample.cannot_change ClosuresReachExample.code_deps
      ClosuresReachExample.dominated ClosuresReachExample.to_unbox
      ClosuresReachExample.tcr ClosuresReachExample.unit_expr
    = false
  ∧ repr_flow_unboxing_nr (fun _ => false) no_local
      ClosuresReachExample.facts ClosuresReachExample.tables
      ClosuresReachExample.cannot_change ClosuresReachExample.code_deps
      ClosuresReachExample.dominated ClosuresReachExample.to_unbox
      ClosuresReachExample.tcr ClosuresReachExample.facts1
      ClosuresReachExample.tables1 ClosuresReachExample.unit_expr
    = true
  ∧ check_unboxed_reads ClosuresReachExample.facts
      ClosuresReachExample.tables ClosuresReachExample.dominated
      ClosuresReachExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_repr_fixed no_local ClosuresReachExample.facts
      ClosuresReachExample.tables ClosuresReachExample.cannot_change
      ClosuresReachExample.code_deps ClosuresReachExample.dominated
      ClosuresReachExample.to_unbox ClosuresReachExample.tcr true
      ClosuresReachExample.unit_expr
    = ClosuresReachExample.rebuilt
  ∧ run 60 (on_its_own ClosuresReachExample.unit_expr)
    = Exit 1%positive ClosuresReachExample.result empty_heap no_reads
  ∧ run 60 (on_its_own ClosuresReachExample.rebuilt)
    = Exit 1%positive ClosuresReachExample.result empty_heap no_reads
  ∧ program_refines_observably ClosuresReachExample.unit_expr
      ClosuresReachExample.rebuilt.
