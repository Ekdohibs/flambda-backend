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

(* The changes of calling convention of the reaper, transcribed, without
   unboxing: which code changes its calling convention, what becomes of
   its parameters and results, and the rebuild of the code and of its
   calls.

   This file transcribes, in middle_end/flambda2/reaper/:
   - unboxing_analysis.ml: the rules [calling_convention_changes_rules],
     as a check of a table ([check_cc_closed]) and as the least table
     ([derive_cannot_change]); [cannot_change_calling_convention];
     [compute_code_changes], of the parameters and results of code
     ([code_calling_convention_change]) and of its metadata
     ([changed_code_metadata], with [arity_of_decisions] and the arities
     of [get_arity_and_modes]); [get_calling_convention_change];
   - rebuild.ml: [decide_whether_apply_needs_calling_convention_change]
     ([code_id_actually_called]), the case [Changing_calling_convention]
     of [rebuild_apply] ([rebuild_apply_cc], with [get_args_with_kinds]),
     [make_apply_wrapper] given the decisions of the callee
     ([make_apply_wrapper_cc]), and the case [Changing_calling_convention]
     of [rebuild_function_params_and_body] ([rebuild_code_cc]).
   It needs no review: FlambdaReaperCallingConventionsSpec.v says what
   the check means and relates the rebuild to that of
   FlambdaReaperRebuild.v, and the test tool compares the decisions and
   the rebuilt term with those of the compiler's reaper. It is extracted
   ([check_cc_closed], [derive_cannot_change], [unit_code_deps],
   [unit_calling_convention_changes], [rebuild_unit_cc] and
   [rebuild_unit_cc_from_tables]).

   What is transcribed is the reaper run with
   [-reaper-change-calling-conventions] and [-no-reaper-unbox]:
   [Unboxing_analysis.perform_analysis] then runs
   [calling_convention_changes_rules] and not [perform_analysis0], so
   that [get_unboxed_fields] and [get_changed_representation] find
   nothing. Every decision is then [Keep] or [Delete], that of
   [my_closure] is [Keep_my_closure], and the branches of rebuild.ml
   that read an unboxing are left out ([Unbox], [Unbox_my_closure], the
   [Invalid] of [make_apply_wrapper] where the decisions of the call and
   of the callee disagree on one).

   The table. [cannot_change_calling_convention] is a relation of the
   Datalog database, derived from
   [cannot_change_witness_calling_convention], which is derived from the
   tables of points_to_analysis.ml; no rule of the group is recursive.
   Neither relation is exported by unboxing_analysis.mli, but the
   decision is, through [compute_code_changes] and
   [is_changing_calling_convention]. So the table is an input, as the
   solution is: the nodes of the code ids whose calling convention does
   not change, those of other units included, as
   [Unboxing_analysis.cannot_change_calling_convention] has them. It is
   checked: [check_cc_closed] tests the two rules that conclude
   [cannot_change_calling_convention], with
   [cannot_change_witness_calling_convention] computed from the tables by
   the six rules that conclude it, which read the tables of
   points_to_analysis.ml only. A table larger than the least passes, as
   is safe: more code keeps its calling convention. For a solution that
   the model computes, [derive_cannot_change] is the least table (a node
   may be listed more than once).

   The code deps. The decisions are taken on the nodes that stand for
   the parameters and the results of code ([Code_dep.params],
   [Code_dep.return]), which FlambdaReaperTraversal.prepare_code makes
   and which are not among the facts. [unit_code_deps] runs the
   traversal of the unit and returns them ([traverse_unit_acc], of which
   [traverse_unit] is the facts: [traverse_unit_acc_facts]).

   The rebuild. [rebuild_expr_cc] is a function of its own, not a
   generalisation of FlambdaReaperRebuild.rebuild_expr: it is that
   function case for case, but for a call ([rebuild_apply_cc]) and for
   code ([rebuild_code_cc]), and every other case calls the functions of
   FlambdaReaperRebuild.v. A call whose code does not change its calling
   convention is [FlambdaReaperRebuild.rebuild_apply], and code that does
   not change is rebuilt as [FlambdaReaperRebuild.rebuild_code] does it.
   In the body of code that changes, the return continuation has the
   decisions of its results ([cont_params_to_keep] of
   [rebuild_function_params_and_body]), which an [Apply_cont] of it and a
   call returning to it read, whatever the call. With every code of the
   unit in the table, the rebuild is that of FlambdaReaperRebuild.v
   (FlambdaReaperCallingConventionsSpec.rebuild_unit_cc_unchanged_statement).

   The deliberate differences from the OCaml, each where the model cannot
   follow it or does not have what it reads:
   - Decisions (unboxing_analysis.ml: [param_decision]). A decision is
     [Keep kind] or [Delete]. The compiler's [Keep] also holds a
     variable: that of the code dep in [compute_code_changes], replaced
     by the parameter of the code in rebuild.ml:
     [rebuild_function_params_and_body], and freshened for a wrapper in
     rebuild.ml: [make_apply_wrapper] ([freshen_decisions]). The model
     takes the parameter of the code at the same place, and names of its
     own for the wrapper, as FlambdaReaperRebuild.make_apply_wrapper
     does.
   - The poison of a wrapper (rebuild.ml: [make_apply_wrapper]). For a
     result that the callee no longer returns and that the continuation
     of the call keeps, the wrapper gives poison of the kind of the
     return arity of the call at that place, where the compiler takes
     the kind of the parameter of the continuation. In a term whose kinds
     agree the two are the same.
   - The code of the unit (unboxing_analysis.ml:
     [cannot_change_calling_convention], [Analysis_scope.contains_unit]).
     The code ids of the unit are those of [static_names code_ids_bound],
     as in FlambdaReaperRebuild.v, and the metadata of code is that of its
     [Bind_code] ([bound_code_metadata]), found where [static_names]
     looks.
   - The callee (rebuild.ml: [rebuild_apply], [rewrite_simple_opt]). The
     callee of a call that changes its calling convention is kept, as in
     FlambdaReaperRebuild.v for every call.
   - Subkinds, as in FlambdaReaperRebuild.v: the reaper without types,
     whose [rewrite_kind_with_subkind] is [erase_subkind], after which
     an entry states nothing. Code that keeps its calling convention has
     what its parameters and results state erased
     ([erase_code_subkinds]); code that changes states nothing of its
     parameters and results ([changed_code_metadata]): what is kept is
     erased ([Keep] of [compute_code_changes]).
   - What the model has not got: [only_full_applications] (rebuild.ml:
     [rewrite_set_of_closures], which sets it for code that changes),
     [first_complex_local_param], [param_modes] and [is_my_closure_used]
     of the metadata (unboxing_analysis.ml: [compute_code_changes]) and
     result types ([rewrite_result_types]).
   - Where the OCaml stops with a fatal error, the model does something
     defined, which no term that the compiler accepts reaches: lists that
     [List.map2] or [List.fold_left2] would refuse for their lengths
     (unboxing_analysis.ml: [compute_code_changes]; rebuild.ml:
     [get_args_with_kinds], [rebuild_apply], [make_apply_wrapper]) are
     cut to the shorter ([combine]), and the groups of
     [Flambda_arity.group_by_parameter] are cut or dropped
     (FlambdaReaperTraversal.group_by_parameter); a code id of the unit
     that has no code dep (deleted code) or no metadata does not change
     (unboxing_analysis.ml: [get_calling_convention_change]); a
     continuation that has fewer decisions than the callee has results
     keeps those beyond, as [FlambdaReaperRebuild.filter_kept] does; code
     with no parameter has an arity of no parameter
     (unboxing_analysis.ml: [compute_code_changes], "Empty parameter
     groups"); a direct call or a call of known arity whose callee
     calls no code (rebuild.ml:
     [decide_whether_apply_needs_calling_convention_change], an empty
     [Code_id.Set.t]) does not change, as it keeps its call kind in
     FlambdaReaperRebuild.v.

   The examples are a code with an unused parameter and an unused result,
   called twice, directly, through a known closure: its decisions, and
   the term that rebuild.ml makes of the unit, a wrapper included; and
   the same unit with the closure escaping, which the check keeps from
   changing. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaFreshQuantifierSpec
  FlambdaRenameApartSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperRebuild FlambdaReaperSolution.
Import ListNotations.

(* ======================================================================== *)
(* The rules: [calling_convention_changes_rules]                            *)
(* ======================================================================== *)

Section Checks.

Variable F : facts.
Variable T : tables.

(* The relations of the solution, as the tables have them. *)
Abbreviation any_usage x := (mem_node x (t_any_usage T)).
Abbreviation any_source x := (mem_node x (t_any_source T)).
Abbreviation has_usage x := (mem_node x (t_has_usage T)).
Abbreviation has_source x := (mem_node x (t_has_source T)).
Abbreviation sources x y := (mem_pair x y (t_sources T)).

(* [rev_accessor ~base:alias code_id_of_call_witness ~to_:v; has_usage v]
   for some [v]. *)
Definition code_id_read_with_usage (alias : node) : bool :=
  existsb
    (fun '(v, relation, base) =>
       field_eqb relation Field_code_id_of_call_witness && node_eqb base alias
       && has_usage v)
    (f_accessor F).

(* The six rules that conclude
   [cannot_change_witness_calling_convention allocation_id], each a test
   at [allocation_id], named after the field of
   FlambdaReaperCallingConventionsSpec.cc_closed that states it, without
   its prefix [cc_]. *)
Definition witness_any_usage (allocation_id : node) : bool :=
  any_usage allocation_id.

Definition witness_other_source (allocation_id : node) : bool :=
  existsb
    (fun '(allocation_id', alias) =>
       node_eqb allocation_id allocation_id'
       && existsb
            (fun '(alias', alias_source) =>
               node_eqb alias alias' && has_source alias_source
               && negb (node_eqb alias_source allocation_id))
            (t_sources T)
       && code_id_read_with_usage alias)
    (t_usages T).

Definition witness_any_source_usage (allocation_id : node) : bool :=
  existsb
    (fun '(allocation_id', alias) =>
       node_eqb allocation_id allocation_id' && any_source alias
       && code_id_read_with_usage alias)
    (t_usages T).

Definition witness_other_own_source (allocation_id : node) : bool :=
  existsb
    (fun '(allocation_id', source) =>
       node_eqb allocation_id allocation_id' && has_source source
       && negb (node_eqb source allocation_id))
    (t_sources T).

Definition witness_not_own_source (allocation_id : node) : bool :=
  existsb
    (fun '(allocation_id', usage) =>
       node_eqb allocation_id allocation_id' && has_usage usage)
    (t_usages T)
  && negb (sources allocation_id allocation_id).

Definition witness_any_source (allocation_id : node) : bool :=
  any_source allocation_id.

(* [cannot_change_witness_calling_convention], as the six rules give it:
   no rule concludes it from itself, so this is the relation of the
   compiler for these tables. *)
Definition cannot_change_witness_calling_convention (allocation_id : node)
    : bool :=
  witness_any_usage allocation_id || witness_other_source allocation_id
  || witness_any_source_usage allocation_id
  || witness_other_own_source allocation_id
  || witness_not_own_source allocation_id || witness_any_source allocation_id.

(* The two rules that conclude [cannot_change_calling_convention
   code_id]: as derived, for the least table, and as tests of a table
   [cannot_change]. *)
Definition derived_by_code_id_of_witness : list node :=
  flat_map
    (fun '(call_witness, relation, code_id) =>
       if field_eqb relation Field_code_id_of_call_witness
          && has_usage call_witness
          && cannot_change_witness_calling_convention call_witness
       then [code_id]
       else [])
    (f_constructor F).

(* The code ids of [call_witness]: [constructor ~base:call_witness
   code_id_of_call_witness ~from:code_id]. *)
Definition code_ids_of_witness (call_witness : node) : list node :=
  flat_map
    (fun '(call_witness', relation, code_id) =>
       if node_eqb call_witness call_witness'
          && field_eqb relation Field_code_id_of_call_witness
       then [code_id]
       else [])
    (f_constructor F).

Definition derived_by_code_id_unknown_arity : list node :=
  flat_map
    (fun '(_, relation, call_witness) =>
       if field_eqb relation (Field_call_witness Unknown_arity_code_pointer)
          && has_usage call_witness
       then code_ids_of_witness call_witness
       else [])
    (f_constructor F).

(* The least table of [cannot_change_calling_convention]. *)
Definition derive_cannot_change : list node :=
  derived_by_code_id_of_witness ++ derived_by_code_id_unknown_arity.

Variable cannot_change : list node.

Definition check_code_id_of_witness : bool :=
  forallb (fun code_id => mem_node code_id cannot_change)
    derived_by_code_id_of_witness.

Definition check_code_id_unknown_arity : bool :=
  forallb (fun code_id => mem_node code_id cannot_change)
    derived_by_code_id_unknown_arity.

(* Whether the table [cannot_change] is closed under
   [calling_convention_changes_rules]. *)
Definition check_cc_closed : bool :=
  check_code_id_of_witness && check_code_id_unknown_arity.

End Checks.

(* ======================================================================== *)
(* The decisions: [compute_code_changes]                                    *)
(* ======================================================================== *)

(* [Unboxing_analysis.param_decision], without [Unbox], and without the
   variable of [Keep]. *)
Inductive param_decision : Type :=
  | Keep (kind : flambda_kind)
  | Delete.

(* [Unboxing_analysis.calling_convention_change], without the decision
   of [my_closure], which is [Keep_my_closure]. *)
Inductive calling_convention_change : Type :=
  | Not_changing_calling_convention
  | Changing_calling_convention (params_decisions : list param_decision)
      (return_decisions : list param_decision).

Definition is_kept (decision : param_decision) : bool :=
  match decision with
  | Keep _ => true
  | Delete => false
  end.

(* The kinds of what is kept ([arity_of_decisions] and [get_parameters],
   unarized). *)
Definition kept_kinds (decisions : list param_decision) : list flambda_kind :=
  flat_map
    (fun decision =>
       match decision with
       | Keep kind => [kind]
       | Delete => []
       end)
    decisions.

(* [Unboxing_analysis.arity_of_decisions]: one product. *)
Definition arity_of_decisions (decisions : list param_decision) : arity :=
  [Unboxed_product (map Singleton (kept_kinds decisions))].

(* The arity of [get_arity_and_modes]: one product for each parameter of
   the code, of the kinds kept. The decision of [my_closure], [Delete],
   which it puts first, adds nothing. *)
Definition arity_of_grouped_decisions (groups : list (list param_decision))
    : arity :=
  map (fun decisions => Unboxed_product (map Singleton (kept_kinds decisions)))
    groups.

(* The metadata of code that changes its calling convention
   ([compute_code_changes]): the arities of what is kept, which states
   nothing beyond its kinds (its entries are erased), and not tupled
   ([with_is_tupled false]). *)
Definition changed_code_metadata (metadata : code_metadata)
    (params_decisions return_decisions : list param_decision)
    : code_metadata :=
  {| cm_params_arity :=
       arity_of_grouped_decisions
         (group_by_parameter (cm_params_arity metadata) params_decisions);
     cm_params_subkinds := [];
     cm_result_arity := arity_of_decisions return_decisions;
     cm_result_subkinds := [];
     cm_is_tupled := false;
     cm_is_stub := cm_is_stub metadata |}.

(* The same, what the parameters kept state rewritten as [types] says
   ([Keep] of [compute_code_changes], with types), the parameters of the
   code being [params]. *)
Definition changed_code_metadata_with (types : subkinds_rewriting)
    (params : list variable) (metadata : code_metadata)
    (params_decisions return_decisions : list param_decision)
    : code_metadata :=
  match types with
  | None => changed_code_metadata metadata params_decisions return_decisions
  | Some rewrite =>
    with_code_params_subkinds
      (rewritten_subkinds rewrite (map is_kept params_decisions) params
         (cm_params_subkinds metadata))
      (changed_code_metadata metadata params_decisions return_decisions)
  end.

Definition metadata_of_code (c : code) : code_metadata :=
  match c with Code _ _ _ _ metadata _ => metadata end.

(* The metadata of the code that the [Let_static]s of an expression bind,
   where [static_names] looks. *)
Fixpoint bound_code_metadata (e : expr) : finite_map code_metadata :=
  match e with
  | Let _ _ _ body => bound_code_metadata body
  | Let_static bindings body =>
    flat_map
      (fun binding =>
         match binding with
         | Bind_code id c => [(id, metadata_of_code c)]
         | Bind_deleted_code _ | Bind_symbol _ _ => []
         end)
      bindings
    ++ bound_code_metadata body
  | Let_cont_non_recursive _ h body =>
    handler_bound_code_metadata h ++ bound_code_metadata body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_bound_code_metadata h) handlers
    ++ bound_code_metadata body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_bound_code_metadata (h : continuation_handler)
    : finite_map code_metadata :=
  match h with
  | Handler _ _ handler => bound_code_metadata handler
  end.

Section Decisions.

Variable sol : solution.
(* The table of [cannot_change_calling_convention]. *)
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
(* The code ids of the unit. *)
Variable unit_code_ids : list code_id.

(* [is_var_used] of [compute_code_changes], of a node of a code dep, whose
   kind is that of the arity of the code. *)
Definition is_node_used (n : node) (kind : flambda_kind) : bool :=
  match kind with
  | Region | Rec_info => true
  | Value | Naked_number _ => s_has_usage sol n
  end.

(* The decisions of [compute_code_changes] for the nodes of a code dep and
   their kinds: [Keep] if used, [Delete] if not. *)
Definition decisions_of (nodes : list node) (kinds : list flambda_kind)
    : list param_decision :=
  map (fun '(n, kind) => if is_node_used n kind then Keep kind else Delete)
    (combine nodes kinds).

(* [Unboxing_analysis.cannot_change_calling_convention], with the flag
   [-reaper-change-calling-conventions]: code of another unit, or code
   that the table has. *)
Definition cannot_change_calling_convention (id : code_id) : bool :=
  negb (among id unit_code_ids) || cannot_change (Node_code_id id).

(* [compute_code_changes], of the code [id] with [metadata]. *)
Definition code_calling_convention_change (id : code_id)
    (metadata : code_metadata) : calling_convention_change :=
  if cannot_change_calling_convention id
  then Not_changing_calling_convention
  else
    match lookup id code_deps with
    | Some cd =>
      Changing_calling_convention
        (decisions_of (cd_params cd) (unarize (cm_params_arity metadata)))
        (decisions_of (cd_return cd) (unarize (cm_result_arity metadata)))
    | None => Not_changing_calling_convention
    end.

(* [get_calling_convention_change], of a code id, for the code of the
   unit whose metadata is [codes]. *)
Definition get_calling_convention_change (codes : finite_map code_metadata)
    (id : code_id) : calling_convention_change :=
  match lookup id codes with
  | Some metadata => code_calling_convention_change id metadata
  | None => Not_changing_calling_convention
  end.

End Decisions.

(* ======================================================================== *)
(* The rebuild                                                              *)
(* ======================================================================== *)

(* The arguments given to the parameters kept, each with the kind of its
   parameter ([get_args_with_kinds]), of one group. *)
Definition get_args_with_kinds (sol : solution) (en : rebuild_env)
    (decisions : list param_decision) (args : list simple)
    : list (simple * flambda_kind) :=
  flat_map
    (fun '(arg, decision) =>
       match decision with
       | Keep kind => [(rewrite_simple sol en arg, kind)]
       | Delete => []
       end)
    (combine args decisions).

(* The arity of the arguments of a call, one product to a group. *)
Definition arity_of_groups (groups : list (list (simple * flambda_kind)))
    : arity :=
  map
    (fun group =>
       Unboxed_product (map (fun '(_, kind) => Singleton kind) group))
    groups.

(* The decisions of the continuation [k] of a call for the [n] results
   of the callee: those of [cont_params_to_keep], and [true] beyond them
   or where the continuation has none. *)
Definition apply_decisions (en : rebuild_env) (k : continuation) (n : nat)
    : list bool :=
  let mask :=
    match lookup k (env_conts en) with
    | Some mask => mask
    | None => []
    end
  in
  firstn n (mask ++ repeat true n).

(* The arguments that the wrapper gives the continuation of the call
   ([rev_args] of [make_apply_wrapper]): for each result, if the
   continuation keeps it, the parameter of the wrapper for it if the
   callee returns it and poison of its kind otherwise. The parameters are
   the names from [next], one for each result that the callee returns. *)
Fixpoint wrapper_args (next : positive) (apply_mask : list bool)
    (func_decisions : list (param_decision * flambda_kind)) : list simple :=
  match apply_mask, func_decisions with
  | keep :: apply_mask, (Keep _, _) :: func_decisions =>
    (if keep then [Var next] else [])
    ++ wrapper_args (Pos.succ next) apply_mask func_decisions
  | keep :: apply_mask, (Delete, kind) :: func_decisions =>
    (if keep then [Const (Poison kind)] else [])
    ++ wrapper_args next apply_mask func_decisions
  | [], _ | _, [] => []
  end.

Section Rebuild.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
(* [should_preserve_direct_calls] is [Yes] or [Auto]. *)
Variable preserve_direct_calls : bool.
(* The code ids of the unit, and the metadata of its code. *)
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
(* The first of the names that the rebuild makes. *)
Variable first_fresh : positive.
(* How what a parameter kept states is rewritten
   (FlambdaReaperRebuild.subkinds_rewriting). *)
Variable types : subkinds_rewriting.

(* The code that a call is known to call
   ([decide_whether_apply_needs_calling_convention_change]): the one code
   id that [code_id_actually_directly_called] finds for the callee of a
   direct call or of a call of known arity. *)
Definition code_id_actually_called (ap : apply_expr) : option code_id :=
  match ap_call_kind ap with
  | Indirect_unknown_arity => None
  | Direct _ | Indirect_known_arity _ =>
    match ap_callee ap with
    | Const _ => None
    | Var _ | Symbol _ =>
      match
        code_id_actually_directly_called F sol (node_of_simple (ap_callee ap))
      with
      | Some ids => Keyed.get_singleton Pos.eqb ids
      | None => None
      end
    end
  end.

(* The change of calling convention of a call: that of the code it is
   known to call. *)
Definition apply_calling_convention_change (ap : apply_expr)
    : calling_convention_change :=
  match code_id_actually_called ap with
  | Some id =>
    get_calling_convention_change sol cannot_change code_deps unit_code_ids
      codes id
  | None => Not_changing_calling_convention
  end.

(* [make_apply_wrapper] given the decisions of the callee, each with the
   kind of the result in the return arity of the call: no wrapper if the
   continuation keeps what the callee returns; otherwise the call returns
   to a wrapper, whose parameters are the results that the callee
   returns. *)
Definition make_apply_wrapper_cc (en : rebuild_env) (ap : apply_expr)
    (func_decisions : list (param_decision * flambda_kind)) : expr :=
  let apply_mask :=
    apply_decisions en (ap_continuation ap) (length func_decisions)
  in
  if forallb (fun '(keep, (decision, _)) => Bool.eqb keep (is_kept decision))
       (combine apply_mask func_decisions)
  then Apply ap
  else
    let kinds := kept_kinds (map fst func_decisions) in
    Let_cont_non_recursive first_fresh
      (Handler (names_from (length kinds) first_fresh)
         (make_handler_metadata kinds)
         (Apply_cont
            {| ac_continuation := ap_continuation ap;
               ac_args := wrapper_args first_fresh apply_mask func_decisions;
               ac_trap_action := None |}))
      (Apply (with_continuation ap first_fresh)).

(* [rebuild_apply]: a call that changes its calling convention gives the
   parameters kept, a product for each group, and expects the results
   kept. *)
Definition rebuild_apply_cc (en : rebuild_env) (ap : apply_expr) : expr :=
  match apply_calling_convention_change ap with
  | Not_changing_calling_convention =>
    rebuild_apply F sol preserve_direct_calls unit_code_ids first_fresh en ap
  | Changing_calling_convention params_decisions return_decisions =>
    if simple_is_dead sol en (ap_callee ap)
       || existsb (simple_is_dead sol en) (ap_args ap)
    then Invalid
    else
      let groups :=
        map
          (fun '(args, decisions) => get_args_with_kinds sol en decisions args)
          (combine (group_by_parameter (ap_args_arity ap) (ap_args ap))
             (group_by_parameter (ap_args_arity ap) params_decisions))
      in
      make_apply_wrapper_cc en
        {| ap_callee := ap_callee ap;
           ap_call_kind :=
             rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap;
           ap_args := flat_map (map fst) groups;
           ap_continuation := ap_continuation ap;
           ap_exn_continuation := ap_exn_continuation ap;
           ap_exn_extra_args := rebuild_exn_extra_args sol en ap;
           ap_args_arity := arity_of_groups groups;
           ap_return_arity := arity_of_decisions return_decisions |}
        (combine return_decisions (unarize (ap_return_arity ap)))
  end.

(* [rebuild_expr] of FlambdaReaperRebuild.v, with [rebuild_apply_cc] and
   [rebuild_code_cc]. *)
Fixpoint rebuild_expr_cc (en : rebuild_env) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind defining_expr body =>
    if is_pending_projection en x defining_expr
    then
      Let x kind defining_expr
        (rebuild_expr_cc
           (bind_variables_then en [(x, kind)] (tl (env_projections en)))
           body)
    else
      match defining_expr with
      | Set_of_closures _ _ | Closure_of_set _ _ _ =>
        let projections := projections_of_set x defining_expr body in
        if is_dead_var sol x kind
           || existsb (fun y => is_dead_var sol y Value) projections
        then Invalid
        else if is_var_used sol x kind
                || existsb (fun y => is_var_used sol y Value) projections
        then
          Let x kind
            (rebuild_set sol en (set_closure_nodes F (Node_variable x))
               defining_expr)
            (rebuild_expr_cc (bind_variables_then en [(x, kind)] projections)
               body)
        else rebuild_expr_cc (bind_variables en [(x, kind)]) body
      | Simple _ | Prim _ _ | Project_value_slot _ _
      | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
        if is_dead_var sol x kind
        then Invalid
        else if is_var_used sol x kind
        then
          Let x kind (rebuild_named sol en x defining_expr)
            (rebuild_expr_cc (bind_variables en [(x, kind)]) body)
        else rebuild_expr_cc (bind_variables en [(x, kind)]) body
      end
  | Let_static bindings body =>
    Let_static (flat_map (rebuild_binding_cc en) bindings)
      (rebuild_expr_cc (bind_variables en []) body)
  | Let_cont_non_recursive k h body =>
    let mask := handler_mask sol (handler_is_exn_handler h) h in
    let body := rebuild_expr_cc (bind_continuations en [(k, mask)]) body in
    if among k (free_continuations_of_expr body)
    then Let_cont_non_recursive k (rebuild_handler_cc en true mask h) body
    else body
  | Let_cont_recursive handlers body =>
    let en :=
      bind_continuations en
        (map (fun '(k, h) => (k, handler_mask sol false h)) handlers)
    in
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, rebuild_handler_cc en false (handler_mask sol false h) h))
         handlers)
      (rebuild_expr_cc en body)
  | Apply ap => rebuild_apply_cc en ap
  | Apply_cont ac => rebuild_apply_cont sol en ac
  | Switch scrutinee arms => rebuild_switch sol en scrutinee arms
  | Invalid => Invalid
  end
(* [rebuild_handler]. *)
with rebuild_handler_cc (en : rebuild_env) (dead_check : bool)
    (mask : list bool) (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let bound := with_kinds params (hm_params_kinds metadata) in
    Handler (filter_kept mask params)
      (kept_handler_metadata types mask params metadata)
      (if dead_check
          && existsb (fun '(x, kind) => is_dead_var sol x kind) bound
       then Invalid
       else rebuild_expr_cc (bind_variables en bound) handler)
  end
(* [rebuild_binding]. *)
with rebuild_binding_cc (en : rebuild_env) (binding : static_binding)
    : list static_binding :=
  match binding with
  | Bind_code id c =>
    if is_code_id_used sol id
    then [Bind_code id (rebuild_code_cc id c)]
    else [Bind_deleted_code id]
  | Bind_deleted_code _ => [binding]
  | Bind_symbol s const =>
    if static_const_is_used sol s const
    then [Bind_symbol s (rebuild_static_const sol en s const)]
    else []
  end
(* [rebuild_code] and [rebuild_function_params_and_body]: code that
   changes its calling convention keeps the parameters kept, has the
   arities of [changed_code_metadata], and its return continuation keeps
   the results kept. *)
with rebuild_code_cc (id : code_id) (c : code) : code :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    let change :=
      code_calling_convention_change sol cannot_change code_deps
        unit_code_ids id metadata
    in
    let bound :=
      (my_closure, Value)
      :: with_kinds params (unarize (cm_params_arity metadata))
    in
    let en :=
      match change with
      | Not_changing_calling_convention => empty_env
      | Changing_calling_convention _ return_decisions =>
        bind_continuations empty_env
          [(return_continuation, map is_kept return_decisions)]
      end
    in
    let body :=
      if existsb (fun '(x, kind) => is_dead_var sol x kind) bound
      then Invalid
      else rebuild_expr_cc (bind_variables en bound) body
    in
    match change with
    | Not_changing_calling_convention =>
      Code params my_closure return_continuation exn_continuation
        (kept_code_metadata types params metadata) body
    | Changing_calling_convention params_decisions return_decisions =>
      Code (filter_kept (map is_kept params_decisions) params) my_closure
        return_continuation exn_continuation
        (changed_code_metadata_with types params metadata params_decisions
           return_decisions)
        body
    end
  end.

End Rebuild.

(* [Rebuild.rebuild] with changes of calling convention: the rebuilt
   unit, given the facts of its graph, the solution, the table of
   [cannot_change_calling_convention] and the code deps of the
   traversal. *)
Definition rebuild_unit_cc_with (types : subkinds_rewriting) (F : facts)
    (sol : solution) (cannot_change : node -> bool)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr) : expr :=
  rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
    (static_names code_ids_bound e) (bound_code_metadata e)
    (fresh_name (names_of e)) types empty_env e.

(* The reaper without types. *)
Definition rebuild_unit_cc (F : facts) (sol : solution)
    (cannot_change : node -> bool) (code_deps : finite_map code_dep)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_expr_cc F sol cannot_change code_deps preserve_direct_calls
    (static_names code_ids_bound e) (bound_code_metadata e)
    (fresh_name (names_of e)) None empty_env e.

(* The same, with the solution and the table given by their tuples. *)
Definition rebuild_unit_cc_from_tables (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (preserve_direct_calls : bool) (e : expr) : expr :=
  rebuild_unit_cc F (solution_of_tables T)
    (fun n => mem_node n cannot_change) code_deps preserve_direct_calls e.

(* The change of calling convention of each code of the unit, for the
   test tool to compare with the compiler's
   ([Analysis.get_calling_convention_change]). *)
Definition unit_calling_convention_changes (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep) (e : expr)
    : list (code_id * calling_convention_change) :=
  map
    (fun '(id, metadata) =>
       (id,
        code_calling_convention_change (solution_of_tables T)
          (fun n => mem_node n cannot_change) code_deps
          (static_names code_ids_bound e) id metadata))
    (bound_code_metadata e).

(* ======================================================================== *)
(* The code deps                                                            *)
(* ======================================================================== *)

(* [traverse_unit], up to the accumulator, of which [traverse_unit] takes
   the facts and [unit_code_deps] the code deps. *)
Definition traverse_unit_acc (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) : acc :=
  let a := add_fact (Fact_any_source Node_all_constants) empty_acc in
  let '(dummy_toplevel_return, a) := fresh a in
  let '(dummy_toplevel_exn, a) := fresh a in
  let a := add_fact (Fact_any_usage dummy_toplevel_exn) a in
  let env :=
    {| FlambdaReaperTraversal.env_conts :=
         [(unit_exn u, [dummy_toplevel_exn]);
          (unit_return u, [dummy_toplevel_return])];
       env_current_code_id := None;
       env_should_preserve_direct_calls := should_preserve_of_flag flag;
       env_value_slots_to_keep := projected_value_slots e;
       env_unit_code_ids := static_names all_code_ids_of_binding e;
       env_set_closures := set_closures e |}
  in
  let a := traverse env e a in
  let a := add_fact (Fact_any_usage dummy_toplevel_return) a in
  resolve_delayed_deps (static_names symbols_of_binding e) a.

Lemma traverse_unit_acc_facts :
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    traverse_unit flag u e
    = facts_of_list (acc_facts (traverse_unit_acc flag u e)).
Proof. reflexivity. Qed.

(* The code deps of the traversal of the unit [e] with the interface
   [u]. *)
Definition unit_code_deps (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) : finite_map code_dep :=
  acc_code_deps (traverse_unit_acc flag u e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Local Open Scope positive_scope.

(* The units return to 1 and raise to 2. *)
Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition make_block (fields : list simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) fields.

(* The code 50 has the parameters [x40] and [x42] and returns
   [x40; x40; x42]. *)
Definition code_50 : code :=
  Code [40; 42] 41 3 4
    (make_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value; Singleton Value; Singleton Value])
    (return_to 3 [Var 40; Var 40; Var 42]).

Definition call_50 (args : list simple) (k : continuation) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value; Singleton Value] |}.

(* The unit makes the closure [x60] of the code 50 and calls it twice,
   directly: first with 9 and [x61], returning to 21, whose handler puts
   the first two results in a block [x36] and calls the code again with
   the block and [x61], returning to 20, whose handler is [h]. *)
Definition two_calls_then (h : expr) : expr :=
  Let_static [Bind_code 50 code_50]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Simple (Const (Tagged_immediate 3)))
          (Let_cont_non_recursive 20
             (Handler [30; 31; 32]
                (make_handler_metadata [Value; Value; Value]) h)
             (Let_cont_non_recursive 21
                (Handler [33; 34; 35]
                   (make_handler_metadata [Value; Value; Value])
                   (Let 36 Value (make_block [Var 33; Var 34])
                      (Apply (call_50 [Var 36; Var 61] 20))))
                (Apply (call_50 [Const (Tagged_immediate 9); Var 61] 21)))))).

(* The unit returns the first result of the second call. *)
Definition two_calls : expr := two_calls_then (return_to 1 [Var 30]).

(* The unit returns the first result of the second call and the closure,
   in a block: the closure escapes. *)
Definition escaping_closure : expr :=
  two_calls_then
    (Let 37 Value (make_block [Var 30; Var 60]) (return_to 1 [Var 37])).

Definition two_calls_facts : facts := traverse_unit Auto interface two_calls.

Definition escaping_closure_facts : facts :=
  traverse_unit Auto interface escaping_closure.

(* Short names for the nodes and fields of the tables below. *)
Definition v (x : positive) : node := Node_variable x.
Definition m (n : positive) : node := Node_made n.
Definition c (id : positive) : node := Node_code_id id.
Definition consts : node := Node_all_constants.
Definition known : field := Field_call_witness Known_arity_code_pointer.
Definition unknown : field := Field_call_witness Unknown_arity_code_pointer.
Definition result (i : nat) : field := Field_return_of_call (Normal i).

(* No field is local. *)
Definition no_local (_ : field) : bool := false.

(* The solution of [two_calls_facts], as the solver of the test tool
   computes it, in the order of the compiler's schedule. The code dep of
   50 is made first: its results 3, 4 and 5, its exception 6, its
   [my_closure] 7, its parameters 8 and 9, its witnesses 10 (of known
   arity), 11 and 12 (of unknown arity, one to a parameter). The calls
   make the witnesses 19 and 14, which read the code id of what they call
   into 20 and 15. Neither witness of the code is used but through the
   calls, and each call finds the witness 10 and no other: the code may
   change its calling convention. Its parameter 8 is used, as [x40] is
   returned; 9 is not, as [x42] is returned only as the third result,
   which no continuation uses; the first two results are used, by the
   block [x36] and by the return of the unit. *)
Definition two_calls_tables : tables :=
  {| t_alias :=
       [(m 23, m 10); (m 7, m 22); (m 18, m 10); (m 7, m 17); (m 19, m 21);
        (m 14, m 16); (m 1, v 30); (v 61, consts); (m 3, v 40); (m 4, v 40);
        (m 5, v 42); (v 41, m 7); (v 40, m 8); (v 42, m 9); (m 16, m 10);
        (m 21, m 10); (v 33, m 3); (v 30, m 3); (v 34, m 4); (v 31, m 4);
        (v 35, m 5); (v 32, m 5); (m 2, m 6); (m 20, c 50); (m 15, c 50);
        (m 8, consts); (m 8, v 36); (m 9, v 61); (m 7, v 60)];
     t_any_usage :=
       [m 1; m 20; m 15; m 2; v 30; m 3; m 6; c 50; v 40; m 8; consts; v 36;
        v 33; v 34; m 4];
     t_any_source :=
       [c 50; consts; v 61; m 20; m 15; m 8; m 9; v 40; v 42; m 3; m 4; m 5;
        v 33; v 30; v 34; v 31; v 35; v 32; m 1];
     t_has_usage :=
       [m 1; m 20; m 15; m 2; v 30; m 19; m 14; m 21; m 16; m 10; m 3; m 6;
        c 50; v 40; m 8; consts; v 36; v 33; v 34; m 4; v 60];
     t_has_source :=
       [c 50; consts; v 61; v 36; v 60; m 12; m 13; m 11; m 10; m 23; m 18;
        m 16; m 21; m 20; m 15; m 8; m 9; m 19; m 14; v 40; v 42; m 7; m 3;
        m 4; m 5; v 41; v 33; v 30; v 34; v 31; v 35; v 32; m 1];
     t_usages :=
       [(v 60, v 60); (m 19, m 19); (m 14, m 14); (m 21, m 19); (m 16, m 14);
        (m 10, m 14); (m 10, m 19)];
     t_sources :=
       [(v 60, v 60); (v 36, v 36); (m 12, m 12); (m 11, m 11); (m 13, m 13);
        (m 10, m 10); (m 23, m 10); (m 18, m 10); (m 16, m 10); (m 21, m 10);
        (m 19, m 10); (m 14, m 10); (m 8, v 36); (v 40, v 36); (m 3, v 36);
        (m 4, v 36); (v 33, v 36); (v 30, v 36); (v 34, v 36); (v 31, v 36);
        (m 1, v 36); (m 7, v 60); (v 41, v 60)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used :=
       [(v 60, known); (v 36, Field_block 0 Value);
        (v 36, Field_block 1 Value); (v 36, Field_is_int);
        (v 36, Field_get_tag); (m 10, result 0); (m 10, result 1);
        (m 10, Field_return_of_call Exn);
        (m 10, Field_code_id_of_call_witness)] |}.

Example two_calls_solution :
  check_closed no_local two_calls_facts two_calls_tables = true
  ∧ check_field_used_exact no_local two_calls_facts two_calls_tables = true.
Proof. vm_compute. auto. Qed.

(* No code is in the least table, and the code 50 keeps its first
   parameter and its first two results. *)
Example two_calls_changes :
  derive_cannot_change two_calls_facts two_calls_tables = []
  ∧ unit_calling_convention_changes two_calls_tables []
      (unit_code_deps Auto interface two_calls) two_calls =
    [(50, Changing_calling_convention [Keep Value; Delete]
            [Keep Value; Keep Value; Delete])].
Proof. vm_compute. auto. Qed.

(* What rebuild.ml makes of the unit. The code loses its second
   parameter and its third result: its arities are a product of what it
   keeps for each parameter, and one for its results
   ([compute_code_changes]), and its return gives the first two results
   ([cont_params_to_keep] of its return continuation). The calls give
   the first argument only ([get_args_with_kinds]), at those arities, so
   that [x61], given for the parameter removed, and its [Let] go. The
   first call returns to 21, which keeps the two results that the code
   returns: no wrapper. The second returns to 20, which keeps the first
   only: it returns to a wrapper, the continuation 62, the name after
   every name of the unit, whose parameters are the two results that the
   code returns, and which applies 20 to the first
   ([make_apply_wrapper]). *)
Definition code_50_changed : code :=
  Code [40] 41 3 4
    {| cm_params_arity :=
         [Unboxed_product [Singleton Value]; Unboxed_product []];
       cm_params_subkinds := [];
       cm_result_arity := [Unboxed_product [Singleton Value; Singleton Value]];
       cm_result_subkinds := [];
       cm_is_tupled := false;
       cm_is_stub := false |}
    (return_to 3 [Var 40; Var 40]).

Definition call_50_changed (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50; ap_args := args;
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Unboxed_product [Singleton Value]; Unboxed_product []];
     ap_return_arity :=
       [Unboxed_product [Singleton Value; Singleton Value]] |}.

Example rebuild_two_calls :
  rebuild_unit_cc_from_tables two_calls_facts two_calls_tables []
    (unit_code_deps Auto interface two_calls) true two_calls =
  Let_static [Bind_code 50 code_50_changed]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (return_to 1 [Var 30]))
          (Let_cont_non_recursive 21
             (Handler [33; 34] (make_handler_metadata [Value; Value])
                (Let 36 Value (make_block [Var 33; Var 34])
                   (Let_cont_non_recursive 62
                      (Handler [62; 63]
                         (make_handler_metadata [Value; Value])
                         (return_to 20 [Var 62]))
                      (Apply (call_50_changed [Var 36] 62)))))
             (Apply (call_50_changed [Const (Tagged_immediate 9)] 21))))).
Proof. vm_compute. reflexivity. Qed.

(* The solution of [escaping_closure_facts], as the solver of the test
   tool computes it. The closure escapes, and with it its witnesses 10
   and 11, which have [any_usage]. *)
Definition escaping_closure_tables : tables :=
  {| t_alias :=
       [(m 23, m 10); (m 7, m 22); (m 18, m 10); (m 7, m 17); (m 19, m 21);
        (m 14, m 16); (m 1, v 37); (v 61, consts); (m 3, v 40); (m 4, v 40);
        (m 5, v 42); (v 41, m 7); (v 40, m 8); (v 42, m 9); (m 7, v 60);
        (m 21, m 10); (m 16, m 10); (v 33, m 3); (v 34, m 4); (v 35, m 5);
        (m 2, m 6); (m 20, c 50); (v 30, m 3); (v 31, m 4); (v 32, m 5);
        (m 15, c 50); (m 8, consts); (m 9, v 61); (m 8, v 36)];
     t_any_usage :=
       [m 1; m 20; m 15; m 2; v 37; v 30; v 60; consts; m 10; m 11; m 13;
        m 12; c 50; m 3; m 4; m 5; m 6; v 40; v 42; m 8; m 9; v 61; v 36;
        v 33; v 34];
     t_any_source :=
       [c 50; consts; v 61; m 9; m 8; v 40; v 42; m 3; m 4; m 5; v 33; v 34;
        v 35; m 20; v 30; v 31; v 32; m 15];
     t_has_usage :=
       [m 1; m 20; m 15; m 2; v 37; v 30; v 60; consts; m 10; m 11; m 13;
        m 12; c 50; m 3; m 4; m 5; m 6; v 40; v 42; m 8; m 9; m 19; m 14;
        m 21; m 16; v 61; v 36; v 33; v 34];
     t_has_source :=
       [c 50; consts; v 61; m 9; m 8; v 40; v 42; m 3; m 4; m 5; v 36; v 37;
        v 60; m 12; m 13; m 11; m 10; m 23; m 18; m 1; m 7; m 21; m 16; v 33;
        v 34; v 35; m 20; v 30; v 31; v 32; m 15; m 19; m 14; v 41];
     t_usages :=
       [(m 19, m 19); (m 14, m 14); (m 21, m 19); (m 16, m 14)];
     t_sources :=
       [(v 60, v 60); (v 36, v 36); (v 37, v 37); (m 12, m 12); (m 11, m 11);
        (m 13, m 13); (m 10, m 10); (m 23, m 10); (m 18, m 10); (m 1, v 37);
        (m 7, v 60); (v 41, v 60); (m 21, m 10); (m 16, m 10); (m 19, m 10);
        (m 14, m 10)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used :=
       [(v 60, known); (v 60, unknown); (v 36, Field_block 0 Value);
        (v 36, Field_block 1 Value); (v 36, Field_is_int);
        (v 36, Field_get_tag); (v 37, Field_block 0 Value);
        (v 37, Field_block 1 Value); (v 37, Field_is_int);
        (v 37, Field_get_tag); (v 60, Field_function_slot 70);
        (m 12, Field_return_of_call Exn); (m 12, result 0); (m 12, result 1);
        (m 12, result 2); (m 12, Field_code_id_of_call_witness);
        (m 11, result 0); (m 13, unknown);
        (m 11, Field_code_id_of_call_witness); (m 10, result 0);
        (m 10, result 1); (m 10, result 2); (m 10, Field_return_of_call Exn);
        (m 10, Field_code_id_of_call_witness)] |}.

Example escaping_closure_solution :
  check_closed no_local escaping_closure_facts escaping_closure_tables = true
  ∧ check_field_used_exact no_local escaping_closure_facts
      escaping_closure_tables = true.
Proof. vm_compute. auto. Qed.

(* The code 50 is in the least table: the check rejects a table without
   it, which would have the code change its calling convention, and
   accepts it. With it, nothing changes its calling convention, and the
   rebuild is that of FlambdaReaperRebuild.v. *)
Example escaping_closure_cannot_change :
  mem_node (c 50)
    (derive_cannot_change escaping_closure_facts escaping_closure_tables)
  = true
  ∧ check_cc_closed escaping_closure_facts escaping_closure_tables [] = false
  ∧ check_cc_closed escaping_closure_facts escaping_closure_tables [c 50]
    = true
  ∧ rebuild_unit_cc_from_tables escaping_closure_facts
      escaping_closure_tables [c 50]
      (unit_code_deps Auto interface escaping_closure) true escaping_closure
    = rebuild_unit_from_tables escaping_closure_facts escaping_closure_tables
        true escaping_closure.
Proof. vm_compute. auto. Qed.

End Examples.
