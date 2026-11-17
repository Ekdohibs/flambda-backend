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

(* The case of [make_apply_wrapper] (rebuild.ml) where the decision of
   the continuation of a call for a parameter and that of the callee for
   its result disagree on unboxing,
     | Unbox _, (Keep _ | Delete) | Keep _, Unbox _ ->
         ... (* CR ncourant: restore this check *)
         let has_any_source = false in
         if has_any_source then error () else Invalid
   in the rebuild of the fix of the unboxing ([Keep_unused] first, the
   seeds [fix_seeds]), and whether it could be a fatal error.

   This file is to be reviewed: the arm ([arm_pair], against
   FlambdaReaperUnboxing.wrapper_args_u), the calls that the rebuild in
   the mode of the fix gives [make_apply_wrapper_u] ([rebuilt_calls],
   against FlambdaReaperUnboxing.rebuild_expr_u and rebuild_apply_u),
   the test of a source ([result_has_source_at]), the checks, the
   example and the statements. The proofs are in
   FlambdaReaperWrapperProofs.v, which needs no review.

   The answer: the arm is reached only where the result of the callee at
   that position has no source, so restoring the test [has_any_source]
   would never stop the compiler, but an unconditional fatal error
   would.
   - [arm_needs_no_source_statement]: for every direct call of the unit
     that the rebuild with the fix makes (code that changes its calling
     convention, code that keeps it, and code that is not known), at
     every position, the decisions are the arm only where no node that
     the call reads off its callee with the field of known arity has any
     source, or a source whose result of that rank has a source
     ([callee_result_has_source]: [not_local_field_has_source] of
     points_to_analysis.ml at that node and field). As a result of the
     callee that the callee unboxes has a source, [Keep _, Unbox _] is
     never reached ([kept_result_never_unboxed_statement]); and a call
     whose callee keeps its calling convention never returns to an
     unboxed parameter but where its result has no source.
   - [indirect_arm_needs_no_source_statement]: the same of the calls of
     unknown arity with one group of arguments, whose callee keeps every
     result, at the nodes read off the callee with the field of unknown
     arity ([result_has_source_at]), with two more checks
     ([check_unknown_widgets], [check_unknown_witnesses]): such a call
     reaches only code that cannot change its calling convention, whose
     results with a source [cu0_result] keeps from an unboxed
     parameter.
   - [result_flow_statement], the argument, for any call: where the
     result [r] of the callee flows into the parameter [x] of the
     continuation of the call ([alias x r]), a used [x] is dominated by
     every allocation to unbox that dominates [r] (the two seeds of the
     fix forbid the other case: a use of [x] ends in a read of a field
     with a used result, or in [any_usage]); [r] is dominated by every
     allocation that dominates [x], if it has a source; and an [x]
     dominated by an allocation to unbox takes no result with a source
     of code that cannot change its calling convention ([cu0_result]).
   - [no_source_arm_statement]: the arm is reached, in a unit that meets
     every premise, by a call of code that never returns (it raises): its
     result has a use and no source, and the parameter of the
     continuation of the call is unboxed from the allocation that another
     jump gives it. The rebuilt unit, whose call returns to a wrapper
     [Invalid], exits as the unit does. The compiler with the fix reaches
     the arm too (repro-wrapper/x.ml: a [[@loop never]] recursive
     function that raises, whose result Simplify does not know to be
     bottom).

   The premises of [arm_needs_no_source_statement], besides tables that
   are a least solution of the rules and tables of
   [dominated_by_allocation_point] and of [to_unbox] that pass the checks
   of the fix ([check_dominated], [check_to_unbox_fixed]):
   - the unit is of the fragment of FlambdaReaperFlowSpec.v, with no
     local field or no projection, as FlambdaReaperFlowProofs.v reads the
     facts of its traversal;
   - [check_cc_closed]: the table of [cannot_change_calling_convention]
     is closed under its rules, so that code called with two code ids is
     in it;
   - [check_apply_widgets] (FlambdaReaperCallingConventionsSpec.v): the
     nodes that calls read off one callee with the field of known arity
     have the same sources; a least solution passes it with the flag
     [Auto] ([arm_needs_no_source_auto_statement]);
   - [check_used_code_ids]: a code id with a use has any usage, so that
     the conditional aliases of the calls in the code that the rebuild
     keeps hold;
   - [check_argument_reads]: a node that is given arguments has a field
     read into a node with any usage (the call witness of a call, whose
     code id is read), so that a use of a parameter ends in a read;
   - [check_known_witnesses]: what a call reads off its callee with the
     field of known arity has for its sources with a source only call
     witnesses of known arity of code deps, each of which holds at the
     result of rank [i] the [i]-th result of its code dep and at its code
     id that of its code;
   - [conts_counted] at toplevel, as FlambdaReaperFlowSpec.v asks of
     code, so that each result of a call has a parameter of its
     continuation; and [plain_kinds]: no parameter of a handler and no
     result of code is of kind [Region] or [Rec_info], which are kept
     without a use.
   [check_used_code_ids], [check_argument_reads],
   [check_known_witnesses] and, for calls of unknown arity,
   [check_unknown_widgets] and [check_unknown_witnesses] say what the
   shape of the traversal gives (a code id is read only by calls of its
   code, arguments are given only to the witnesses of calls, a witness is
   put only in the field of its arity of the closures of its code, and
   the nodes read off a callee with one field have the sources of that
   field of the callee); they are not proved of the traversal.

   Calls of unknown arity with several groups of arguments are not lifted
   to the unit: their results are read through a chain of witnesses, one
   for each group. [result_flow_statement] holds of them as of any
   call. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaValueSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaReaperLeast FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec
  FlambdaReaperCallingConventionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The arm                                                                  *)
(* ======================================================================== *)

(* The pairs of [wrapper_args_u] that are the arm, after [Decided Delete]
   and [Keep_unused]: the decision of the continuation for its parameter
   [ad] and that of the callee for its result [fd] (which
   [freshen_decisions] makes a [Wrapper_keep], [Wrapper_delete] or
   [Wrapper_unbox]). *)
Definition arm_pair (ad fd : decision) : bool :=
  match ad, fd with
  | Unbox _, (Decided _ | Keep_unused _) => true
  | Decided (Keep _), Unbox _ => true
  | _, _ => false
  end.

(* The decision of the continuation at the result of rank [i], of kind
   [kind]: [Keep] beyond its decisions, as [wrapper_args_u] has it. *)
Definition apply_decision_at (ads : list decision) (i : nat)
    (kind : flambda_kind) : decision :=
  match nth_error ads i with
  | Some d => d
  | None => Decided (Keep kind)
  end.

(* ======================================================================== *)
(* The calls of the rebuild                                                 *)
(* ======================================================================== *)

Section Calls.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).

(* The decisions of the callee that [rebuild_apply_u] gives
   [make_apply_wrapper_u], each with the kind of the result in the return
   arity of the call. *)
Definition callee_decisions (ap : apply_expr)
    : list (decision * flambda_kind) :=
  match
    apply_change_u F sol cannot_change code_deps unit_code_ids codes unboxed
      ap
  with
  | Not_changing_u =>
    map (fun kind => (Decided (Keep kind), kind)) (unarize (ap_return_arity ap))
  | Changing_u _ return_decisions =>
    combine return_decisions (unarize (ap_return_arity ap))
  end.

(* The calls of [e] that [rebuild_expr_u] in the mode of the fix rebuilds,
   under the decisions [ds] of the continuations in scope, each with the
   decisions of its continuation that [make_apply_wrapper_u] reads: the
   continuations bound as [rebuild_expr_u] binds them
   ([handler_decisions], [code_continuation_decisions]), and the code
   that it keeps ([is_code_id_used]). Where the rebuild drops a term
   ([Invalid] for a dead name, a handler not applied), its calls are
   kept here. *)
Fixpoint rebuilt_calls (ds : finite_map (list decision)) (e : expr)
    {struct e} : list (list decision * apply_expr) :=
  match e with
  | Let _ _ _ body => rebuilt_calls ds body
  | Let_static bindings body =>
    flat_map binding_rebuilt_calls bindings ++ rebuilt_calls ds body
  | Let_cont_non_recursive k h body =>
    handler_rebuilt_calls ds h
    ++ rebuilt_calls
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
    flat_map (fun '(_, h) => handler_rebuilt_calls ds h) handlers
    ++ rebuilt_calls ds body
  | Apply ap =>
    [(match lookup (ap_continuation ap) ds with
      | Some decisions => decisions
      | None => []
      end,
      ap)]
  | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_rebuilt_calls (ds : finite_map (list decision))
    (h : continuation_handler) {struct h}
    : list (list decision * apply_expr) :=
  match h with
  | Handler _ _ handler => rebuilt_calls ds handler
  end
with binding_rebuilt_calls (binding : static_binding) {struct binding}
    : list (list decision * apply_expr) :=
  match binding with
  | Bind_code id c => if is_code_id_used sol id then code_rebuilt_calls id c
                      else []
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_rebuilt_calls (id : code_id) (c : code) {struct c}
    : list (list decision * apply_expr) :=
  match c with
  | Code _ _ return_continuation exn_continuation metadata body =>
    rebuilt_calls
      (code_continuation_decisions sol code_deps unboxed id metadata
         (code_change_u sol cannot_change code_deps unit_code_ids unboxed id
            metadata)
         return_continuation exn_continuation)
      body
  end.

End Calls.

(* The calls of the unit [e] that [rebuild_unit_unbox_repr_fixed] (and
   its variant [_tail], whose decisions are the same) rebuilds, with the
   arguments that it gives [rebuild_expr_u]; and the decisions of the
   callee of each. *)
Definition unit_rebuilt_calls (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (unboxed : list (node * unboxed_fields))
    (e : expr) : list (list decision * apply_expr) :=
  rebuilt_calls (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) unboxed [] e.

Definition unit_callee_decisions (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (unboxed : list (node * unboxed_fields)) (e : expr) (ap : apply_expr)
    : list (decision * flambda_kind) :=
  callee_decisions F (solution_of_tables T) (fun n => mem_node n cannot_change)
    code_deps (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    ap.

(* The unboxed fields of the names of the unit, as
   [rebuild_unit_unbox_repr_modes] computes them. *)
Definition unit_unboxed (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : list (node * unboxed_fields) :=
  fst (unit_unboxed_fields F T dominated to_unbox e).

(* ======================================================================== *)
(* A source                                                                 *)
(* ======================================================================== *)

Abbreviation known := (Field_call_witness Known_arity_code_pointer).
Abbreviation unknown := (Field_call_witness Unknown_arity_code_pointer).

(* The test that the arm comments out, [not_local_field_has_source] of
   points_to_analysis.ml, at what the graph now has in place of the field
   [Apply (entry, Normal i)] of the callee: a node [a] that a call reads
   off the callee with the field [entry] has any source, or a source with
   a field [Field_return_of_call (Normal i)] that has a source. *)
Definition result_has_source_at (F : facts) (T : tables) (entry : field)
    (callee : node) (i : nat) : bool :=
  existsb
    (fun '(a, f, base) =>
       field_eqb f entry && node_eqb base callee
       && (mem_node a (t_any_source T)
           || existsb
                (fun '(a', s) =>
                   node_eqb a' a
                   && existsb
                        (fun '(s', f', r) =>
                           node_eqb s' s
                           && field_eqb f' (Field_return_of_call (Normal i))
                           && mem_node r (t_has_source T))
                        (f_constructor F))
                (t_sources T)))
    (f_accessor F).

(* For a direct call, the field of known arity. *)
Definition callee_result_has_source (F : facts) (T : tables) (callee : node)
    (i : nat) : bool :=
  result_has_source_at F T known callee i.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

(* A code id with a use has any usage. *)
Definition check_used_code_ids (T : tables) : bool :=
  forallb
    (fun n =>
       match n with
       | Node_code_id _ => mem_node n (t_any_usage T)
       | _ => true
       end)
    (t_has_usage T).

(* A node given arguments has a field read into a node with any usage. *)
Definition check_argument_reads (F : facts) : bool :=
  forallb
    (fun '(_, _, base) =>
       existsb
         (fun '(v, _, base') =>
            node_eqb base base' && mem_node v (f_any_usage F))
         (f_accessor F))
    (f_argument F).

(* The sources with a source of a node read off a callee with the field of
   known arity are witnesses of known arity of code deps; such a witness
   holds at the result of rank [i] only the [i]-th result of its code
   dep, and at its code id only that of its code. *)
Definition check_known_witnesses (F : facts) (T : tables)
    (code_deps : finite_map code_dep) : bool :=
  forallb
    (fun '(a, f, _) =>
       negb (field_eqb f known)
       || forallb
            (fun '(a', s) =>
               negb (node_eqb a' a) || negb (mem_node s (t_has_source T))
               || existsb
                    (fun '(_, cd) =>
                       node_eqb s (cd_known_arity_call_witness cd))
                    code_deps)
            (t_sources T))
    (f_accessor F)
  && forallb
       (fun '(id, cd) =>
          forallb
            (fun '(w, f, r) =>
               negb (node_eqb w (cd_known_arity_call_witness cd))
               || match f with
                  | Field_return_of_call (Normal i) =>
                    match nth_error (cd_return cd) i with
                    | Some r' => node_eqb r r'
                    | None => false
                    end
                  | Field_code_id_of_call_witness =>
                    node_eqb r (Node_code_id id)
                  | _ => true
                  end)
            (f_constructor F))
       code_deps.

(* [check_apply_widgets] for the field of unknown arity: the nodes that
   calls read off one callee with that field have the same sources. *)
Definition check_unknown_widgets (F : facts) (T : tables) : bool :=
  forallb
    (fun '(a, f, n) =>
       forallb
         (fun '(a', f', n') =>
            negb (field_eqb f unknown && field_eqb f' f && node_eqb n' n)
            || (Bool.eqb (mem_node a (t_any_source T))
                  (mem_node a' (t_any_source T))
                && forallb
                     (fun '(x, y) =>
                        negb (node_eqb x a) || mem_pair a' y (t_sources T))
                     (t_sources T)))
         (f_accessor F))
    (f_accessor F).

(* The sources with a source of a node read off a callee with the field
   of unknown arity are held in the field of unknown arity of a node, and
   have a code id. *)
Definition check_unknown_witnesses (F : facts) (T : tables) : bool :=
  forallb
    (fun '(a, f, _) =>
       negb (field_eqb f unknown)
       || forallb
            (fun '(a', s) =>
               negb (node_eqb a' a) || negb (mem_node s (t_has_source T))
               || (existsb
                     (fun '(_, f', s') => field_eqb f' unknown && node_eqb s' s)
                     (f_constructor F)
                   && existsb
                        (fun '(s', f', _) =>
                           node_eqb s' s
                           && field_eqb f' Field_code_id_of_call_witness)
                        (f_constructor F)))
            (t_sources T))
    (f_accessor F).

(* The number of parameters of each continuation of the unit. *)
Definition unit_counts (u : unit_interface) : finite_map nat :=
  map (fun '(k, ns) => (k, length ns)) (unit_conts u).

Definition plain_kind (kind : flambda_kind) : bool :=
  match kind with
  | Value | Naked_number _ => true
  | Region | Rec_info => false
  end.

(* No parameter of a handler and no result of code is of kind [Region]
   or [Rec_info]. *)
Fixpoint plain_kinds (e : expr) : bool :=
  match e with
  | Let _ _ _ body => plain_kinds body
  | Let_static bindings body =>
    forallb binding_plain_kinds bindings && plain_kinds body
  | Let_cont_non_recursive _ h body => handler_plain_kinds h && plain_kinds body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_plain_kinds h) handlers && plain_kinds body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_plain_kinds (h : continuation_handler) : bool :=
  match h with
  | Handler _ metadata handler =>
    forallb plain_kind (hm_params_kinds metadata) && plain_kinds handler
  end
with binding_plain_kinds (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_plain_kinds c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_plain_kinds (c : code) : bool :=
  match c with
  | Code _ _ _ _ metadata body =>
    forallb plain_kind (unarize (cm_result_arity metadata)) && plain_kinds body
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition result_flow_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox : list node) (x r : node),
    is_local_slots_only is_local ->
    least_solution is_local F T ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    check_argument_reads F = true ->
    mem_pair x r (t_alias T) = true ->
    (∀ a : node,
       mem_pair a r dominated = true -> mem_node a to_unbox = true ->
       mem_node x (t_has_usage T) = true -> mem_pair a x dominated = true)
    ∧ (∀ a : node,
         mem_pair a x dominated = true -> mem_node r (t_has_source T) = true ->
         mem_pair a r dominated = true)
    ∧ (∀ (a w codeid : node) (f : field),
         mem_pair a x dominated = true -> mem_node a to_unbox = true ->
         mem_node r (t_has_source T) = true ->
         (w, f, r) ∈ f_constructor F -> is_return_of_call f = true ->
         (w, Field_code_id_of_call_witness, codeid) ∈ f_constructor F ->
         mem_node codeid cannot_change = false).

Definition arm_needs_no_source_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_used_code_ids T = true ->
    check_argument_reads F = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fd : decision) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false.

(* The same with the flag [Auto], the compiler's default, for which a
   least solution passes [check_apply_widgets]. *)
Definition arm_needs_no_source_auto_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (dominated : list (node * node))
    (to_unbox : list node),
    let F := traverse_unit Auto u e in
    let code_deps := unit_code_deps Auto u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    check_used_code_ids T = true ->
    check_argument_reads F = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fd : decision) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false.

(* Calls of unknown arity with one group of arguments: their callee
   keeps every result, and the decisions are the arm only where no node
   that the call reads off its callee with the field of unknown arity has
   any source, or a source whose result of rank [i] has a source. *)
Definition indirect_arm_needs_no_source_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_used_code_ids T = true ->
    check_argument_reads F = true ->
    check_known_witnesses F T code_deps = true ->
    check_unknown_widgets F T = true ->
    check_unknown_witnesses F T = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (i : nat) (fd : decision)
      (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Indirect -> length (ap_args_arity ap) <= 1 ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (fd, kind) ->
      arm_pair (apply_decision_at ads i kind) fd = true ->
      result_has_source_at F T unknown (node_of_simple (ap_callee ap)) i
      = false.

(* A result that the callee unboxes is never given to a parameter that
   the continuation keeps: of the arm, only [Unbox _, (Keep _ | Delete)]
   is reached. *)
Definition kept_result_never_unboxed_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (max_unbox_size : nat) (lambda_lifting : bool)
    (dominated : list (node * node)) (to_unbox : list node),
    let F := traverse_unit flag u e in
    let code_deps := unit_code_deps flag u e in
    let unboxed := unit_unboxed F T dominated to_unbox e in
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    least_solution is_local F T ->
    check_cc_closed F T cannot_change = true ->
    check_dominated T dominated = true ->
    check_to_unbox_fixed is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true ->
    check_used_code_ids T = true ->
    check_argument_reads F = true ->
    check_known_witnesses F T code_deps = true ->
    conts_counted (unit_counts u) e = true ->
    plain_kinds e = true ->
    ∀ (ads : list decision) (ap : apply_expr) (id : code_id) (i : nat)
      (fields : unboxed_fields) (kind : flambda_kind),
      (ads, ap) ∈ unit_rebuilt_calls T cannot_change code_deps unboxed e ->
      ap_call_kind ap = Direct id ->
      nth_error
        (unit_callee_decisions F T cannot_change code_deps unboxed e ap) i
      = Some (Unbox fields, kind) ->
      arm_pair (apply_decision_at ads i kind) (Unbox fields) = false.

(* ======================================================================== *)
(* The arm reached                                                          *)
(* ======================================================================== *)

(* A call of code that never returns, whose result has a use and no
   source, to a continuation whose parameter the rules unbox:

     let code f50 x40 = raise x40 (to its exception continuation 4)
     let x60 = closure of f50 in
     let k20 x30 = (let x31 = x30.(0) in return x31) in
     let k21 () = x60 7 (direct f50) -> k20 in
     let x10 = [1] in
     switch 0 [0 -> k21 (); 1 -> k20 x10]

   [x30] has the one source [x10], which is unboxed: its only use is the
   read of its field. The result 3 of the code dep of 50 is used through
   [x30] and has no source; 50 changes its calling convention and keeps
   it. The call has the decision [Unbox] of [x30] for a result that 50
   keeps: the arm, and the call returns to the wrapper [k63], whose
   handler is [Invalid]. The unit raises 7, and so does the rebuilt unit.
   The tables are the least solution of the rules
   (FlambdaReaperLeast.compute_least), and those of [cannot_change],
   [dominated] and [to_unbox] the ones that the rules give. *)
Module NoSourceExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition call_f (k : continuation) (arity : arity) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := [int 7];
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := []; ap_args_arity := arity;
     ap_return_arity := arity |}.

Definition unit_with (metadata : code_metadata) (slot : function_slot)
    (k20 k21 : continuation_handler) (x : variable) (d : named) : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4 metadata (Apply_cont (jump 4 [Var 40])))]
    (Let 60 Value (Closure_of_set slot [(slot, 50)] [])
       (Let_cont_non_recursive 20 k20
          (Let_cont_non_recursive 21 k21
             (Let x Value d
                (Switch (Const (Naked_immediate 0))
                   [(0%Z, jump 21 []); (1%Z, jump 20 [Var x])]))))).

Definition unit_expr : expr :=
  unit_with (make_code_metadata [Singleton Value] [Singleton Value]) 70
    (Handler [30] (make_handler_metadata [Value])
       (Let 31 Value (load 0 (Var 30)) (return_to 1 [Var 31])))
    (Handler [] (make_handler_metadata [])
       (Apply (call_f 20 [Singleton Value])))
    10 (make_block [int 1]).

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

Definition unboxed : list (node * unboxed_fields) :=
  unit_unboxed facts tables dominated to_unbox unit_expr.

(* The rebuilt unit: [x30] is the variable [x61] of its field, the block
   [x10] is the variable [x62] of its field, and the call returns to the
   wrapper [k63]. *)
Definition rebuilt : expr :=
  unit_with
    (make_code_metadata [Unboxed_product [Singleton Value]]
       [Unboxed_product [Singleton Value]])
    71
    (Handler [61] (make_handler_metadata [Value])
       (Let 31 Value (Simple (Var 61)) (return_to 1 [Var 31])))
    (Handler [] (make_handler_metadata [])
       (Let_cont_non_recursive 63
          (Handler [63] (make_handler_metadata [Value]) Invalid)
          (Apply (call_f 63 [Unboxed_product [Singleton Value]]))))
    62 (Simple (int 1)).

End NoSourceExample.

(* The unit meets every premise of [arm_needs_no_source_statement]; its
   one call has the decision [Unbox] of [x30] and the decision [Keep] of
   the callee for the result 3, which has a use and no source; the
   rebuild with the fix (and its variant) makes the wrapper [Invalid];
   and the unit and the rebuilt unit raise 7. *)
Definition no_source_arm_statement : Prop :=
  let F := NoSourceExample.facts in
  let T := NoSourceExample.tables in
  let e := NoSourceExample.unit_expr in
  least_solution Examples.no_local F T
  ∧ in_flow_fragment e = true ∧ projection_free e = true
  ∧ check_cc_closed F T NoSourceExample.cannot_change = true
  ∧ check_dominated T NoSourceExample.dominated = true
  ∧ check_to_unbox_fixed Examples.no_local F T NoSourceExample.cannot_change
      Examples.max_unbox_size false NoSourceExample.to_unbox
    = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true
  ∧ check_used_code_ids T = true
  ∧ check_argument_reads F = true
  ∧ check_known_witnesses F T NoSourceExample.code_deps = true
  ∧ conts_counted (unit_counts Examples.interface) e = true
  ∧ plain_kinds e = true
  ∧ NoSourceExample.to_unbox = [Node_variable 10%positive]
  ∧ unit_rebuilt_calls T NoSourceExample.cannot_change
      NoSourceExample.code_deps NoSourceExample.unboxed e
    = [([Unbox [(Field_block 0 Value, Not_unboxed 61%positive)]],
        NoSourceExample.call_f 20%positive [Singleton Value])]
  ∧ unit_callee_decisions F T NoSourceExample.cannot_change
      NoSourceExample.code_deps NoSourceExample.unboxed e
      (NoSourceExample.call_f 20%positive [Singleton Value])
    = [(Decided (Keep Value), Value)]
  ∧ arm_pair (Unbox [(Field_block 0 Value, Not_unboxed 61%positive)])
      (Decided (Keep Value))
    = true
  ∧ lookup 50%positive NoSourceExample.code_deps
    = Some {| cd_params := [Node_made 6%positive];
              cd_my_closure := Node_made 5%positive;
              cd_return := [Node_made 3%positive];
              cd_exn := Node_made 4%positive;
              cd_known_arity_call_witness := Node_made 7%positive;
              cd_unknown_arity_call_witnesses := [Node_made 8%positive] |}
  ∧ mem_node (Node_made 3%positive) (t_has_usage T) = true
  ∧ mem_node (Node_made 3%positive) (t_has_source T) = false
  ∧ callee_result_has_source F T (Node_variable 60%positive) 0 = false
  ∧ rebuild_unit_unbox_repr_fixed Examples.no_local F T
      NoSourceExample.cannot_change NoSourceExample.code_deps
      NoSourceExample.dominated NoSourceExample.to_unbox NoSourceExample.tcr
      true e
    = NoSourceExample.rebuilt
  ∧ rebuild_unit_unbox_repr_fixed_tail Examples.no_local F T
      NoSourceExample.cannot_change NoSourceExample.code_deps
      NoSourceExample.dominated NoSourceExample.to_unbox NoSourceExample.tcr
      true e
    = NoSourceExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own e)
      (Exits 2%positive [Value_const (Tagged_immediate 7)] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own NoSourceExample.rebuilt)
      (Exits 2%positive [Value_const (Tagged_immediate 7)] empty_heap).
