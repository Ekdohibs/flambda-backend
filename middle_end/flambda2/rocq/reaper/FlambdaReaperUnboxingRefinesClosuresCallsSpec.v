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
   FlambdaReaperUnboxingRefinesClosuresSpec.v: beside the sets whose
   representation changes, code that keeps its calling convention and the
   calls of it, raising from code, extra arguments for the exception
   continuation of a call, and parameters of handlers kept with no use.

   This file is to be reviewed: the fragment ([repr_calls_unboxing_nr]),
   the statements and the example, against FlambdaReaperUnboxing.v and
   FlambdaReaperUnboxingRefinesClosuresSpec.v. The proofs are in
   FlambdaReaperUnboxingRefinesClosuresCallsProofs.v and
   FlambdaReaperUnboxingRefinesClosuresProofs.v, which need no review.

   [repr_calls_holds_repr_statement]: the fragment holds that of
   FlambdaReaperUnboxingRefinesClosuresSpec.v ([repr_unboxing_nr]), for
   the same tables and the same code taken to never return.

   [rebuild_repr_calls_refines_cc_statement],
   [rebuild_repr_calls_observably_statement] and
   [rebuild_repr_calls_local_statement]: the statements
   [rebuild_repr_fixed_refines_cc_never_returns_statement],
   [rebuild_repr_fixed_observably_never_returns_statement] and
   [rebuild_repr_fixed_local_never_returns_statement] of
   FlambdaReaperUnboxingRefinesClosuresSpec.v with [repr_calls_unboxing_nr]
   in place of [repr_unboxing_nr], with the same premises otherwise: the
   unit rebuilt with unboxing and changes of representation in the mode
   of the fix refines the unit rebuilt with the change of calling
   conventions alone, and the unit up to observation (with local fields
   for the last). The statements of that file are corollaries.

   [rebuild_repr_calls_with_types_refines_cc_statement],
   [rebuild_repr_calls_with_types_local_statement] and
   [rebuild_repr_calls_with_types_observably_statement]: the same for the
   rebuilds of the reaper with types, as
   FlambdaReaperTypesSpec.rebuild_repr_fixed_with_types_refines_cc_nr_statement
   and the two after it, with the same premises but the fragment.

   The fragment ([repr_calls_unboxing_nr]) is that of
   FlambdaReaperUnboxingRefinesClosuresSpec.v ([repr_unboxing_nr]: names
   plain, unboxed or changed; the [Let]s, the sets of closures, those whose
   representation changes and the reads of their slots alike; static
   constants; [check_unboxed_uses]), but:
   - A continuation is described by the decisions of the rebuild with
     unboxing read as the two rebuilds read them ([normed]): a parameter
     kept with no use ([Keep_unused]) is kept, as one with a use is; the
     rebuilds give it what is given ([get_args]), keep it
     ([params_of_decisions]) and state of it what the first rebuild does
     ([decisions_subkinds]). So a handler may keep a parameter with no use.
   - Code ([repr_code_w]) that keeps its calling convention is of the
     fragment: it is not tupled, it has a code dep, and its results are
     not unboxed, so that its return continuation keeps every result in
     both rebuilds (the first has no decision for it, the second those of
     a continuation of fixed arity, [fixed_arity_decisions], each read as
     kept). Its parameters are plain and its [my_closure] is not unboxed,
     as for code that changes its calling convention, and the return
     continuation of code of [nr] is not live in its body.
   - Code raises: an [Apply_cont] (or an arm) to the exception continuation
     of the code it is in, with plain arguments, is of the fragment if the
     exception of the code is not unboxed ([code_exn_ds]: the decision of a
     continuation of fixed arity, [code_exn_decision], read as kept). Both
     rebuilds give it its arguments as they are.
   - Calls ([repr_call_w]) of code that keeps its calling convention:
     direct, of the code they are known to call, or of known arity, at the
     arities of the code, by a variable of the class of the [my_closure] of
     the code (plain, or changed with the same representation), with plain
     arguments; the first rebuild gives the arguments at the mask of a
     call of known arity, the second at the parameters of the code dep
     with a use, which must agree, an argument given as it is a name with
     a use (FlambdaReaperUnboxingRefinesCallsSpec.kept_args_ok); the call
     returns to a continuation that keeps every result (or a way out of the
     unit), so that neither rebuild makes a wrapper, or is of code of
     [nr], which the fix makes return to a handler [Invalid]. Calls of code
     that changes its calling convention are those of [repr_call].
   - A call, of code that changes its calling convention or keeps it, may
     give extra arguments to its exception continuation if that is a
     handler of the term ([extra_args_ok]): they are plain, and both
     rebuilds give those that the handler keeps.

   What is left out, for the proof: unboxed parameters (of code or of
   handlers), wrappers but for calls that never return, calls whose code
   is not known (of known or of unknown arity: a closure that a plain name
   holds could be one whose code reads its [my_closure] in the new
   representation; the flow would have to exclude it), extra arguments
   for an exception continuation that is not a handler of the term,
   projections of other closures of a set, static sets whose
   representation changes, and closures that are unboxed.

   [repr_calls_kept_code_example_statement]: a unit of
   [repr_calls_unboxing_nr] that is not of [repr_unboxing_nr]
   ([ClosuresCallsExample]): the set of [x60] changes its representation
   (its code 54 reads the block of two fields in its slot, which is
   unboxed into two new slots), and the code 56, whose closure the unit
   returns in a block, keeps its calling convention, is called directly
   under a handler of its exception with an extra argument, and raises in
   an arm that is not taken. With the tables of the least solution and the
   table of the rules for [cannot_change_calling_convention], the rebuild
   in the mode of the fix is the term [rebuilt], the unit and its rebuild
   return the same block, and the rebuilt unit refines the unit up to
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
  FlambdaReaperUnboxingRefinesClosuresSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperLeast FlambdaReaperLeastSpec FlambdaReaperTypes
  FlambdaStaticsInScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* A decision as the two rebuilds read it: a parameter kept with no use is
   kept. *)
Definition norm_decision (d : decision) : decision :=
  match d with
  | Keep_unused kind => Decided (Keep kind)
  | Decided _ | Unbox _ => d
  end.

Definition normed (ds : list decision) : list decision :=
  map norm_decision ds.

(* The results of code that keeps its calling convention, as its callers
   take them: every one kept. *)
Definition kept_results (metadata : code_metadata) : list decision :=
  map (fun kind => Decided (Keep kind)) (unarize (cm_result_arity metadata)).

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

Local Abbreviation plain_c := (plain_c unboxed changed).
Local Abbreviation changed_name := (changed_name changed).

(* The exception continuation of the code [id]: [[d]], [d] the decision of
   a continuation of fixed arity that keeps the exception, if the code may
   raise to it; [[]] if its exception is unboxed. *)
Definition code_exn_ds (id : code_id) : list decision :=
  match code_exn_decision sol code_deps unboxed id with
  | Some d => [norm_decision d]
  | None => []
  end.

(* An [Apply_cont] to [C]'s continuations: plain arguments, to a
   continuation that may be applied, to a way out of the unit, or to the
   exception continuation of the code it is in, if the code may raise to
   it. *)
Definition apply_cont_ok_w (C : finite_map (list decision * bool))
    (ac : apply_cont) : bool :=
  forallb plain_c (ac_args ac)
  && match lookup (ac_continuation ac) C with
     | Some (_, true) | None => true
     | Some ([], false) => false
     | Some (_ :: _, false) => true
     end.

(* The extra arguments of the exception continuation of a call: plain, and
   only for a handler of the term, whose parameters after the first take
   them. *)
Definition extra_args_ok (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  forallb plain_c (exn_extra_simples ap)
  && match ap_exn_extra_args ap with
     | [] => true
     | _ :: _ =>
       match lookup (ap_exn_continuation ap) C with
       | Some (_, true) => true
       | Some (_, false) | None => false
       end
     end.

Definition repr_call_w (C : finite_map (list decision * bool))
    (ap : apply_expr) : bool :=
  match ap_callee ap, code_id_actually_called F sol ap with
  | Var f, Some id =>
    kind_calls (ap_call_kind ap) id
    && match lookup id codes, lookup id my_closures with
       | Some metadata, Some mc =>
         (negb (code_never_returns sol code_deps id) || nr id)
         && repr_option_eqb (changed_name (Node_variable f))
              (changed_name (Node_variable mc))
         && is_none (fields_of unboxed f)
         && arity_eqb (ap_args_arity ap) (cm_params_arity metadata)
         && arity_eqb (ap_return_arity ap) (cm_result_arity metadata)
         && forallb plain_c (ap_args ap)
         && extra_args_ok C ap
         && match
              code_change_u sol cannot_change code_deps unit_code_ids unboxed
                id metadata
            with
            | Changing_u pd rd =>
              forallb decided pd && forallb decided rd
              && is_none
                   (my_closure_decision_u cannot_change code_deps
                      unit_code_ids unboxed id)
              && Nat.eqb (length rd)
                   (length (unarize (cm_result_arity metadata)))
              && (code_never_returns sol code_deps id
                  || returns_ok C (ap_continuation ap) rd)
            | Not_changing_u =>
              kept_args_ok F sol code_deps codes unboxed f ap
              && (code_never_returns sol code_deps id
                  || returns_ok C (ap_continuation ap)
                       (kept_results metadata))
            end
       | _, _ => false
       end
  | _, _ => false
  end.

Fixpoint repr_expr_w (kinds : finite_map flambda_kind)
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
    && repr_expr_w ((x, kind) :: kinds) C body
  | Let_static bindings body =>
    forallb repr_binding_w bindings && repr_expr_w kinds C body
  | Let_cont_non_recursive k h body =>
    let ds := handler_decisions true sol unboxed (handler_is_exn_handler h) h in
    repr_handler_w kinds C ds h
    && repr_expr_w kinds ((k, (normed ds, true)) :: C) body
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
         repr_handler_w kinds C' (handler_decisions true sol unboxed false h)
           h)
      handlers
    && repr_expr_w kinds C' body
  | Apply ap => repr_call_w C ap
  | Apply_cont ac => apply_cont_ok_w C ac
  | Switch scrutinee arms =>
    plain_c scrutinee && forallb (fun '(_, ac) => apply_cont_ok_w C ac) arms
  | Invalid => true
  end
with repr_handler_w (kinds : finite_map flambda_kind)
    (C : finite_map (list decision * bool)) (ds : list decision)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb kind_ok (hm_params_kinds metadata)
    && forallb decided (normed ds)
    && forallb (fun x => plain_c (Var x)) params
    && repr_expr_w (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         C handler
  end
with repr_binding_w (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code id c => repr_code_w id c
  | Bind_deleted_code _ => true
  | Bind_symbol _ (Static_const_block _ fields) => forallb plain_c fields
  | Bind_symbol _ _ => false
  end
with repr_code_w (id : code_id) (c : code) {struct c} : bool :=
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
         && repr_expr_w
              (rev ((my_closure, Value)
                    :: with_kinds params (unarize (cm_params_arity metadata))))
              [(ret, (rd, true)); (exn, (code_exn_ds id, false))] body
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
              && repr_expr_w
                   (rev ((my_closure, Value)
                         :: with_kinds params
                              (unarize (cm_params_arity metadata))))
                   [(ret, (rds, true)); (exn, (code_exn_ds id, false))] body
            | None => false
            end
       end
    && (negb (nr id)
        || negb (among ret (live_conts F sol code_deps codes body)))
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing and the changes of
   representation that the tables give from its graph [F], and the code
   [nr] taken to never return. *)
Definition repr_calls_unboxing_nr (nr : code_id -> bool)
    (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  repr_expr_w F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (unit_changed_representations is_local F T dominated to_unbox tcr e)
    (bound_code_my_closures e) nr [] [] e
  && check_unboxed_uses F T unboxed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition repr_calls_holds_repr_statement : Prop :=
  ∀ (nr : code_id -> bool) (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr),
    repr_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr e
    = true ->
    repr_calls_unboxing_nr nr is_local F T cannot_change code_deps dominated
      to_unbox tcr e
    = true.

Definition rebuild_repr_calls_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition rebuild_repr_calls_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too (FlambdaReaperLocalFieldsSpec.v). *)
Definition rebuild_repr_calls_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
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
   FlambdaReaperTypesSpec.rebuild_repr_fixed_with_types_refines_cc_nr_statement:
   code that keeps its calling convention states the same in both
   rebuilds ([kept_code_metadata]), and a handler states of a parameter
   kept with no use what the first rebuild does. *)
Definition rebuild_repr_calls_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (FlambdaReaperTypes.rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change (unit_code_deps flag u e)
         preserve_direct_calls e)
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The unit rebuilt with unboxing and changes of representation by the
   reaper with types refines the unit up to observation, with the
   premises of
   FlambdaReaperTypesSpec.rebuild_repr_fixed_with_types_local_nr_statement
   but the fragment. *)
Definition rebuild_repr_calls_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

(* The same without local fields. *)
Definition rebuild_repr_calls_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool),
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
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
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
   fields in the slot 80 of its closure, whose set changes its
   representation, and adds them and its parameter), with the code 56,
   which returns its parameter plus 1 or, in an arm that is not taken,
   raises it, and whose closure [x61] the unit returns in a block with the
   result: so it escapes, and 56 keeps its calling convention. The unit
   calls 54 with 32, then 56 with the result under a handler of its
   exception, and returns the block of 48 and [x61]. *)
Module ClosuresCallsExample.

Local Open Scope positive_scope.

Definition jump (k : continuation) (args : list simple)
    (trap : option trap_action) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := trap |}.

Definition code_56 : code :=
  Code [50] 51 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 52 Value (TwoFieldsExample.add (Var 50) (int 1))
       (Let 53 (Naked_number Kind_naked_immediate)
          (Prim (Unary Untag_immediate) [Var 50])
          (Switch (Var 53)
             [(0%Z, {| ac_continuation := 6; ac_args := [Var 50];
                       ac_trap_action := Some (Pop 6 None) |});
              (47%Z, {| ac_continuation := 5; ac_args := [Var 52];
                        ac_trap_action := None |})]))).

Definition call (id : code_id) (callee : variable) (args : list simple)
    (k exn : continuation) (extra : list (simple * flambda_kind))
    : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Direct id; ap_args := args;
     ap_continuation := k; ap_exn_continuation := exn;
     ap_exn_extra_args := extra;
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

(* The handler of the result [x30] of 54: it calls 56 with it under the
   handler 22 of its exception, to which it gives [x61] as an extra
   argument and which returns it; the handler 21 of the result of 56
   returns the block of it and [x61]. *)
Definition try_56 : expr :=
  Let_cont_non_recursive 22
    (Handler [33; 34] (make_exn_handler_metadata [Value; Value])
       (return_to 1 [Var 34]))
    (Let_cont_non_recursive 21
       (Handler [31] (make_handler_metadata [Value])
          (Let 32 Value (make_block [Var 31; Var 61])
             (jump 1 [Var 32] (Some (Pop 22 None)))))
       (Let_cont_non_recursive 23
          (Handler [] (make_handler_metadata [])
             (Apply (call 56 61 [Var 30] 21 22 [(Var 61, Value)])))
          (jump 23 [] (Some (Push 22))))).

(* [body], the call of 54, returns to the handler 20 that runs [try_56]. *)
Definition after (body : expr) : expr :=
  Let_cont_non_recursive 20
    (Handler [30] (make_handler_metadata [Value]) try_56) body.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 54 TwoFieldsExample.code_54; Bind_code 56 code_56]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
          (Let 61 Value (Closure_of_set 71 [(71, 56)] [])
             (after (Apply (call 54 60 [int 32] 20 2 [])))))).

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

(* The rebuild: 54 is rebuilt as in [TwoFieldsExample.rebuilt], reading
   the new slots 81 and 82 of its closure into the variables 62 and 63 of
   the fields of [x45]; 56, its raise, its call and the handlers are as
   they were. *)
Definition changed_54 : code :=
  Code [40] 41 3 4
    {| cm_params_arity := [Unboxed_product [Singleton Value]];
       cm_params_subkinds := [];
       cm_result_arity := [Unboxed_product [Singleton Value]];
       cm_result_subkinds := [];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (Let 63 Value (Project_value_slot_from 83 82 (Var 41))
       (Let 62 Value (Project_value_slot_from 83 81 (Var 41))
          (Let 46 Value (Simple (Var 62))
             (Let 47 Value (Simple (Var 63))
                (Let 48 Value (TwoFieldsExample.add (Var 46) (Var 47))
                   (Let 49 Value (TwoFieldsExample.add (Var 48) (Var 40))
                      (return_to 3 [Var 49]))))))).

Definition rebuilt : expr :=
  Let_static [Bind_code 54 changed_54; Bind_code 56 code_56]
    (Let 65 Value (Simple (int 8))
       (Let 64 Value (Simple (int 7))
          (Let 60 Value
             (Closure_of_set 83 [(83, 54)] [(81, Var 64); (82, Var 65)])
             (Let 61 Value (Closure_of_set 71 [(71, 56)] [])
                (after
                   (Apply
                      {| ap_callee := Var 60; ap_call_kind := Direct 54;
                         ap_args := [int 32]; ap_continuation := 20;
                         ap_exn_continuation := 2; ap_exn_extra_args := [];
                         ap_args_arity := [Unboxed_product [Singleton Value]];
                         ap_return_arity :=
                           [Unboxed_product [Singleton Value]] |})))))).

(* What the unit and its rebuild return. *)
Definition result : list value :=
  [Value_block 0
     [Value_const (Tagged_immediate 48);
      Value_closure_of_set 71 [(71, 56)] []]].

End ClosuresCallsExample.

Definition repr_calls_kept_code_example_statement : Prop :=
  check_dominated ClosuresCallsExample.tables ClosuresCallsExample.dominated
  = true
  ∧ check_to_unbox no_local ClosuresCallsExample.facts
      ClosuresCallsExample.tables [] max_unbox_size false
      ClosuresCallsExample.to_unbox
    = true
  ∧ check_to_change_representation no_local ClosuresCallsExample.facts
      ClosuresCallsExample.tables [] max_unbox_size false
      ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr
    = true
  ∧ check_sets_whole ClosuresCallsExample.facts ClosuresCallsExample.tables
      ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr
    = true
  ∧ ClosuresCallsExample.to_unbox = [v 10]
  ∧ mem_node (v 60) ClosuresCallsExample.tcr = true
  ∧ mem_node (c 56) ClosuresCallsExample.cannot_change = true
  ∧ mem_node (c 54) ClosuresCallsExample.cannot_change = false
  ∧ repr_unboxing no_local ClosuresCallsExample.facts
      ClosuresCallsExample.tables ClosuresCallsExample.cannot_change
      ClosuresCallsExample.code_deps ClosuresCallsExample.dominated
      ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr
      ClosuresCallsExample.unit_expr
    = false
  ∧ repr_calls_unboxing_nr (fun _ => false) no_local
      ClosuresCallsExample.facts ClosuresCallsExample.tables
      ClosuresCallsExample.cannot_change ClosuresCallsExample.code_deps
      ClosuresCallsExample.dominated ClosuresCallsExample.to_unbox
      ClosuresCallsExample.tcr ClosuresCallsExample.unit_expr
    = true
  ∧ check_unboxed_reads ClosuresCallsExample.facts
      ClosuresCallsExample.tables ClosuresCallsExample.dominated
      ClosuresCallsExample.to_unbox
    = true
  ∧ rebuild_unit_unbox_repr_fixed no_local ClosuresCallsExample.facts
      ClosuresCallsExample.tables ClosuresCallsExample.cannot_change
      ClosuresCallsExample.code_deps ClosuresCallsExample.dominated
      ClosuresCallsExample.to_unbox ClosuresCallsExample.tcr true
      ClosuresCallsExample.unit_expr
    = ClosuresCallsExample.rebuilt
  ∧ run 60 (on_its_own ClosuresCallsExample.unit_expr)
    = Exit 1%positive ClosuresCallsExample.result empty_heap no_reads
  ∧ run 60 (on_its_own ClosuresCallsExample.rebuilt)
    = Exit 1%positive ClosuresCallsExample.result empty_heap no_reads
  ∧ program_refines_observably ClosuresCallsExample.unit_expr
      ClosuresCallsExample.rebuilt.
