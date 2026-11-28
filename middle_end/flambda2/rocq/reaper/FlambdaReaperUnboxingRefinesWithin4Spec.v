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

(* The relation between the traversal of a unit and that of its first
   rebuild, for the plain rebuild: the first rebuild drops [Let]s, code,
   handlers, calls and primitives, and gives poison, but changes no
   calling convention, makes no call direct that the traversal reads
   otherwise, and adds no wrapper.

   This file is to be reviewed: the check [facts_within_poison], the
   class ([plain_unit], [plain_rebuild]), the statements and the
   example. The proofs are in
   FlambdaReaperUnboxingRefinesWithin4SimProofs.v and
   FlambdaReaperUnboxingRefinesWithin4Proofs.v, which need no review.

   The check of FlambdaReaperUnboxingRefinesWithin3Spec.v,
   [analysis_within is_local r F1 F T], is [unit_poison_ok] and
   [facts_poison_ok] of the facts [F1] of the first rebuilt unit renamed
   by [r]. Its conditions on a fact of [F1] are of two sorts: that the
   fact is one of [F], or one that the unit's tables with poison satisfy
   (the facts part), and, for a fact that gives poison to a field of a
   block or to an argument of a call, that the rules alias to the
   constants only nodes with no use (the solution's side:
   [poison_constructor_ok], [poison_argument_ok]). [facts_within_poison F T
   F1] is the facts part alone: [facts_poison_ok] with, for a field or an
   argument given poison, only that [F] has a fact at the same place,
   whose node has no use, or whose field has none
   ([poison_constructor_site], [poison_argument_site]). With the
   solution's side of each poison fact ([poison_sites_met]) and
   [unit_poison_ok], it is [analysis_within]
   ([within_poison_analysis_within_statement]).

   The class. [plain_unit e]: the unit is a [Let_static] of code (code
   and deleted code, of distinct code ids) followed by a term that binds
   no static, or such a term; no body binds a static; the [Let]s bind no
   set of closures of the older form ([Set_of_closures]) and no
   projection of the older form ([Project_value_slot]); a set of closures
   has one function; there is no [Local_allocation]; a [Let] of a kind
   other than [Value] and the naked numbers binds a primitive of regions.
   [plain_rebuild F sol pdc pres e], of the unit's facts and solution:
   - the continuation of every call keeps all its parameters, and its
     exception continuation its first, in the rebuild (no wrapper; the
     exception of the call goes to the parameter it went to);
   - the call kind that the rebuild gives a call is read by the traversal
     as the call kind of the unit ([call_kind_kept]): the same, or a call
     of known arity for a call of known arity or a direct call, or, where
     direct calls are not preserved ([Preserve_no]), a direct call of
     code of the unit for either; and a direct call that the traversal
     gives an apply dep calls code that has a use or that the unit does
     not bind;
   - a set of closures with a use has code with a use or code that the
     unit does not bind (no closure of deleted code);
   - a boxed number with a use has a use of its field (the rebuild gives
     poison to a boxed number whose field has no use, as a [Simple]: an
     alias of a node with a use to the constants, which no renaming takes
     to a fact of the unit; [box_poison_statement]).
   With every code of the unit in the table [cannot_change] (no change of
   calling convention), the first rebuild is the rebuild of
   FlambdaReaperRebuild.v
   (FlambdaReaperCallingConventionsSpec.rebuild_unit_cc_unchanged_statement).

   The statement this file is for, left open: for every unit of the
   class whose tables are closed, with every code of the unit in
   [cannot_change], a renaming [r] with [renaming_ok r e] takes the facts
   of the first rebuilt unit within the unit's up to poison:
   [facts_within_poison (traverse_unit flag u e) T (rename_facts r
   (traverse_unit flag u (first_rebuild ...)))]. The renaming is that of
   the two runs of the traversal: each node that the traversal of the
   first rebuilt unit makes goes to the node that the traversal of the
   unit makes at the same place, the traversal of the unit making more
   nodes where the rebuild drops a term. The steps of the two runs, for
   every construct of the class but the [Let_static] of the code, are
   related in FlambdaReaperUnboxingRefinesWithin4SimProofs.v; what is
   missing is the induction over the terms that composes them, the code
   deps of [prepare_code] and the delayed deps of [resolve_delayed_deps].

   [box_poison_statement]: a unit that boxes a float and reads [Is_int]
   of the box is of the class but for the boxed number (its premises
   other than [plain_rebuild] hold, and [plain_rebuild] fails at the box
   only), and is of the fragment read off its own analysis; no renaming
   with [renaming_ok] takes the facts of its first rebuild within the
   unit's ([analysis_within] fails for each), as the box, which has a use,
   is an alias of the constants there. So not every unit of the fragments
   passes [analysis_within]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperLeast
  FlambdaReaperUnboxing FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesWithinSpec
  FlambdaReaperUnboxingRefinesWithin2Spec
  FlambdaReaperUnboxingRefinesWithin3Spec.
Import ListNotations.

(* ======================================================================== *)
(* The facts within up to poison                                            *)
(* ======================================================================== *)

Section Check.

Variable F : facts.
Variable T : tables.

Local Abbreviation S := (solution_of_tables T).
Local Abbreviation P := (solution_of_tables (poisoned_tables F T)).
Local Abbreviation consts := Node_all_constants.

(* A field [relation] of [b] given poison: [F] has the field, from a node
   with no use, or the field has no use. *)
Definition poison_constructor_site (b : node) (relation : field) (f : node)
    : bool :=
  node_eqb f consts
  && existsb
       (fun '(b', relation', f') =>
          node_eqb b' b && field_eqb relation' relation
          && (negb (s_has_usage S f') || negb (s_field_used S b relation)))
       (f_constructor F).

(* An argument number [i] of the call [w] given poison: [F] has one. *)
Definition poison_argument_site (f : node) (i : nat) (w : node) : bool :=
  node_eqb f consts
  && existsb (fun '(_, i', w') => Nat.eqb i' i && node_eqb w' w)
       (f_argument F).

(* [facts_poison_ok] with the sites of poison in place of
   [poison_constructor_ok] and [poison_argument_ok]. *)
Definition facts_within_poison (F1 : facts) : bool :=
  forallb (fun '(a, b) => mem_pair a b (f_alias F) || s_alias P a b)
    (f_alias F1)
  && forallb
       (fun '(a, b) =>
          mem_pair a b (f_use F)
          || ((negb (s_has_usage P a) || s_any_usage P b)
              && (negb (s_has_source P b) || s_any_source P a)))
       (f_use F1)
  && list_within node_field_node_eq_dec (f_accessor F1) (f_accessor F)
  && forallb
       (fun '(b, relation, f) =>
          inb node_field_node_eq_dec (b, relation, f) (f_constructor F)
          || poison_constructor_site b relation f)
       (f_constructor F1)
  && forallb
       (fun '(f, i, w) =>
          inb node_nat_node_eq_dec (f, i, w) (f_argument F)
          || poison_argument_site f i w)
       (f_argument F1)
  && list_within node_nat_node_eq_dec (f_parameter F1) (f_parameter F)
  && forallb
       (fun '(i, t, f) =>
          inb node_triple_eq_dec (i, t, f) (f_propagate F)
          || negb (s_any_usage P i) || s_alias P t f)
       (f_propagate F1)
  && forallb
       (fun '(c, t, f) =>
          inb node_triple_eq_dec (c, t, f) (f_alias_if_any_source F)
          || negb (s_any_source P c) || s_alias P t f)
       (f_alias_if_any_source F1)
  && forallb (fun a => mem_node a (f_any_usage F) || s_any_usage P a)
       (f_any_usage F1)
  && list_within node_eq_dec (f_any_source F1) (f_any_source F)
  && forallb
       (fun '(i, x) =>
          mem_pair i x (f_any_source_if_used F)
          || negb (s_any_usage P i) || s_any_source P x)
       (f_any_source_if_used F1)
  && list_within node_eq_dec (f_zero_alloc_source F1) (f_zero_alloc_source F).

End Check.

(* The solution's side of the poison facts of [F1]: each field and each
   argument of [F1] that [F] does not have meets the conditions of
   [analysis_within]. *)
Definition poison_sites_met (is_local : field -> bool) (F : facts)
    (T : tables) (F1 : facts) : bool :=
  forallb
    (fun '(b, relation, f) =>
       inb node_field_node_eq_dec (b, relation, f) (f_constructor F)
       || poison_constructor_ok is_local F T b relation f)
    (f_constructor F1)
  && forallb
       (fun '(f, i, w) =>
          inb node_nat_node_eq_dec (f, i, w) (f_argument F)
          || poison_argument_ok F T f i w)
       (f_argument F1).

Definition within_poison_analysis_within_statement : Prop :=
  ∀ (is_local : field -> bool) (r : list (node * node)) (F1 F : facts)
    (T : tables),
    unit_poison_ok is_local F T = true ->
    facts_within_poison F T (rename_facts r F1) = true ->
    poison_sites_met is_local F T (rename_facts r F1) = true ->
    analysis_within is_local r F1 F T = true.

(* ======================================================================== *)
(* The class                                                                *)
(* ======================================================================== *)

(* A kind of a variable that the rebuild may find with no use. *)
Definition value_or_naked (kind : flambda_kind) : bool :=
  match kind with
  | Value | Naked_number _ => true
  | Region | Rec_info => false
  end.

(* A [Let] of the class. *)
Definition plain_named (kind : flambda_kind) (defining_expr : named) : bool :=
  match defining_expr with
  | Prim (Region_primitive Local_allocation) _ => false
  | Prim (Region_primitive _) _ => true
  | Simple _ | Prim _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ => value_or_naked kind
  | Closure_of_set _ [_] _ => value_or_naked kind
  | Closure_of_set _ _ _ | Set_of_closures _ _ | Project_value_slot _ _ =>
    false
  end.

(* A term of the class, which binds no static. *)
Fixpoint plain_expr (e : expr) : bool :=
  match e with
  | Let _ kind defining_expr body =>
    plain_named kind defining_expr && plain_expr body
  | Let_static _ _ => false
  | Let_cont_non_recursive _ h body => plain_handler h && plain_expr body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => plain_handler h) handlers && plain_expr body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with plain_handler (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => plain_expr handler
  end.

Definition plain_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => plain_expr (code_body c)
  | Bind_deleted_code _ => true
  | Bind_symbol _ _ => false
  end.

(* The unit: code first, of distinct code ids, then a term of the
   class. *)
Definition plain_unit (e : expr) : bool :=
  match e with
  | Let_static bindings body =>
    forallb plain_binding bindings
    && distinct (flat_map code_ids_bound bindings)
    && plain_expr body
  | _ => plain_expr e
  end.

(* The code ids that the unit binds to code (not to deleted code): those
   that the traversal gives a code dep. *)
Definition unit_code_ids_prepared (e : expr) : list code_id :=
  map fst (FlambdaReaperFlowSpec.unit_codes e).

Section Plain.

Variable F : facts.
Variable sol : solution.
Variable preserve_direct_calls : bool.
Variable pres : should_preserve_direct_calls.
(* The code ids of the unit, and those that it binds to code. *)
Variable unit_code_ids : list code_id.
Variable prepared : list code_id.

(* A code id that the apply dep of a call may name: one with a use, or
   one that the unit does not bind to code. *)
Definition apply_code_ok (id : code_id) : bool :=
  negb (among id prepared) || s_has_usage sol (Node_code_id id).

(* The call kind [ck1] that the rebuild gives a call of kind [ck] is read
   by the traversal as [ck]: the same, or of known arity, or, where direct
   calls are not preserved, a direct call of code of the unit. *)
Definition call_kind_kept (ck1 ck : call_kind) : bool :=
  let direct_as_known (id1 : code_id) :=
    match pres with
    | Preserve_no => among id1 unit_code_ids
    | Preserve_yes | Preserve_auto => false
    end
  in
  match ck1, ck with
  | Indirect_unknown_arity, Indirect_unknown_arity => true
  | Indirect_known_arity _, (Indirect_known_arity _ | Direct _) => true
  | Direct id1, Direct id =>
    (Pos.eqb id1 id && apply_code_ok id) || direct_as_known id1
  | Direct id1, Indirect_known_arity _ => direct_as_known id1
  | _, _ => false
  end.

(* The continuations of a call keep what the call gives them. *)
Definition call_conts_kept (masks : finite_map (list bool)) (ap : apply_expr)
    : bool :=
  match lookup (ap_continuation ap) masks with
  | Some mask => forallb (fun keep => keep) mask
  | None => true
  end
  && match lookup (ap_exn_continuation ap) masks with
     | Some (keep :: _) => keep
     | Some [] | None => true
     end.

(* What the class asks of the rebuild of a term, with the masks of the
   continuations in scope ([env_conts] of the rebuild). *)
Fixpoint plain_rebuild_expr (masks : finite_map (list bool)) (e : expr)
    : bool :=
  match e with
  | Let x _ defining_expr body =>
    match defining_expr with
    | Closure_of_set _ function_decls _ =>
      negb (s_has_usage sol (Node_variable x))
      || forallb (fun '(_, id) => apply_code_ok id) function_decls
    | Prim (Unary (Box_number bn)) _ =>
      negb (s_has_usage sol (Node_variable x))
      || s_field_used sol (Node_variable x) (Field_boxed_number bn)
    | _ => true
    end
    && plain_rebuild_expr masks body
  | Let_static _ body => plain_rebuild_expr masks body
  | Let_cont_non_recursive k h body =>
    plain_rebuild_handler masks h
    && plain_rebuild_expr
         ((k, handler_mask sol (handler_is_exn_handler h) h) :: masks) body
  | Let_cont_recursive handlers body =>
    let masks :=
      map (fun '(k, h) => (k, handler_mask sol false h)) handlers ++ masks
    in
    forallb (fun '(_, h) => plain_rebuild_handler masks h) handlers
    && plain_rebuild_expr masks body
  | Apply ap =>
    call_conts_kept masks ap
    && call_kind_kept
         (rewrite_call_kind F sol preserve_direct_calls unit_code_ids ap)
         (ap_call_kind ap)
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with plain_rebuild_handler (masks : finite_map (list bool))
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => plain_rebuild_expr masks handler
  end.

End Plain.

(* What the class asks of the rebuild of the unit: of the body of each
   code, with no continuation in scope, and of the term after it. *)
Definition plain_rebuild (F : facts) (sol : solution)
    (preserve_direct_calls : bool) (pres : should_preserve_direct_calls)
    (e : expr) : bool :=
  let ids := static_names code_ids_bound e in
  let prepared := unit_code_ids_prepared e in
  let check :=
    plain_rebuild_expr F sol preserve_direct_calls pres ids prepared []
  in
  match e with
  | Let_static bindings body =>
    forallb
      (fun binding =>
         match binding with
         | Bind_code _ c => check (code_body c)
         | Bind_deleted_code _ | Bind_symbol _ _ => true
         end)
      bindings
    && check body
  | _ => check e
  end.

(* ======================================================================== *)
(* The boxed number                                                         *)
(* ======================================================================== *)

(* A unit that boxes the float 0 into [x10] and returns [Is_int] of the
   box. [x10] has a use (its field [Is_int] is read), and its field of
   boxed number has none, so that the first rebuild binds [x10] to poison,
   as a [Simple]: the traversal of the first rebuilt unit gives the alias
   of [x10] to the constants, which [x10], a variable of the unit with a
   use, does not have in the unit's tables with poison. *)
Module BoxExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let 10 Value
    (Prim (Unary (Box_number Boxable_naked_float)) [Const (Naked_float 0)])
    (Let 11 Value (Prim (Unary (Is_int true)) [Var 10])
       (return_to 1 [Var 11])).

Definition facts : facts := traverse_unit Never interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition first : expr :=
  first_rebuild Never interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Never interface first.

End BoxExample.

Definition box_poison_statement : Prop :=
  plain_unit BoxExample.unit_expr = true
  ∧ check_closed (fun _ => false) BoxExample.facts BoxExample.tables = true
  ∧ plain_rebuild BoxExample.facts (solution_of_tables BoxExample.tables)
      true (should_preserve_of_flag Never) BoxExample.unit_expr
    = false
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) BoxExample.facts
      BoxExample.tables BoxExample.cannot_change
      (unit_code_deps Never FlambdaReaperUnboxing.Examples.interface
         BoxExample.unit_expr)
      [] [] BoxExample.unit_expr
    = true
  ∧ (∀ r : list (node * node),
       renaming_ok r BoxExample.unit_expr = true ->
       analysis_within (fun _ => false) r BoxExample.facts1 BoxExample.facts
         BoxExample.tables
       = false).
