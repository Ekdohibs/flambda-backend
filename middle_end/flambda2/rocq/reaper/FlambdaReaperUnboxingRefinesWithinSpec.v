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

(* The statements of FlambdaReaperUnboxingRefinesUnitFlowSpec.v without a
   second analysis: the unit's own tables [T] serve as the tables of the
   first rebuilt unit, for the units whose first rebuild only drops facts
   of the traversal.

   This file is to be reviewed: the check [facts_within], the statements
   and the examples. The proofs are in
   FlambdaReaperUnboxingRefinesWithinProofs.v, which needs no review.

   The statements of FlambdaReaperUnboxingRefinesUnitFlowSpec.v take
   tables [T1] of the first rebuilt unit, closed under the rules on its
   facts [F1], and the check [used_reach_covered] that compares the reach
   of [F1], [T1] with that of the unit's facts [F] and tables [T]. That
   check is not derived: in general it needs the facts of the traversal of
   the first rebuilt unit related to those of the unit up to the names the
   rebuild makes, and the least solutions compared through the negative
   guards of the rules.

   Here [T1] is [T]. The rules have the facts as premises only, so that
   tables closed under the rules on [F] are closed under them on any
   facts within [F]; and the reach, [over_reach] and [rem_reach] read the
   facts only through tests of membership, so that with the same tables
   they read no more off fewer facts. So where every fact of [F1] is a
   fact of [F] ([facts_within F1 F]), the run of the first rebuilt unit
   is as the reach of [F], [T] says (FlambdaReaperFlowSpec.flow_sound on
   the first rebuilt unit with [T]), and the fragments read off the unit's
   own analysis ([calls_unboxing_unit_flow3_nr],
   [calls_unboxing_unit_never_nr]) are the fragments of
   FlambdaReaperUnboxingRefinesReach3Spec.v and
   FlambdaReaperUnboxingRefinesNeverSpec.v for that reach. Neither the
   tables [T1] nor their checks nor [used_reach_covered] are asked; what is
   left of the first rebuilt unit is [in_heap_fragment] and
   [facts_within], two tests of the first rebuild of the unit with its own
   tables.

   [within_flow3_refines_cc_statement],
   [within_flow3_refines_observably_statement],
   [within_flow3_observably_local_statement]: the statements of
   [rebuild_unbox_unit_flow3_refines_cc_statement] and its two forms up to
   observation, with [facts_within] in place of the tables [T1], their
   checks and [used_reach_covered]. The form [_refines_cc] asks
   [check_local_reads_used] of [T], which the other two ask or derive.
   [within_never_refines_cc_statement],
   [within_never_refines_observably_statement],
   [within_never_observably_local_statement]: the same for the statements
   of [rebuild_unbox_unit_never_refines_cc_statement].

   [within_used_reach_covered_statement]: with [T] for [T1], the check of
   the earlier statements holds where the facts that the reach reads (the
   constructors and the inputs [any_source]) are within the unit's
   ([code_facts_within]), so that those statements apply with [T] for [T1]
   wherever [T] passes [check_closed] and [check_local_reads_used] on the
   facts of the first rebuilt unit.

   The units that the first rebuild leaves as they are ([settled]: the
   first rebuild of the unit with its tables is the unit) are of the
   class, and for them the premises are of the unit alone:
   [settled_flow3_refines_cc_statement],
   [settled_flow3_refines_observably_statement],
   [settled_flow3_observably_local_statement],
   [settled_never_refines_cc_statement],
   [settled_never_refines_observably_statement] and
   [settled_never_observably_local_statement].

   Which units are of the class, and why it is not wider:
   - The rebuild drops [Let]s with no use, the bodies of handlers whose
     parameters have no source (made [Invalid]) and code with no use, and
     lists the code of a call of known arity, which adds no fact (the
     traversal reads no list). Such a rebuild keeps every fact that it
     does not drop, under the same name, only if what it drops makes no
     fresh node ([Node_made]): the traversal numbers fresh nodes in turn,
     so that dropping a call, a primitive or code renumbers the fresh nodes
     of what comes after.
   - The rebuild adds facts in three places: poison given to a parameter
     with no use (an alias to [Node_all_constants], which has any source;
     [poison_reach_statement]), a call made direct (the call widget of
     known arity and, with [Auto], aliases under any source of the callee
     and an apply dep), and the wrappers and renumbered parameters of a
     change of calling convention. A unit where any of these happen is not
     of the class.
   - Wider classes need the tables of the first rebuilt unit built from
     [T]: a renaming of the fresh nodes between the two traversals, any
     source added at the parameters given poison (which needs that a node
     with no use flows only to nodes with no use), and the facts of the
     calls made direct and of the wrappers satisfied by [T]. That is the
     relation between the two traversals that the earlier statements
     leave open; it is not done here.

   [within_example_statement]: the unit of [never_ind_example_statement]
   is of the class (its first rebuild makes the bodies of the handlers 20
   and 24 [Invalid] and lists the code of the call of known arity), and
   the statements of the fragment of code that never returns read off its
   own analysis give that its rebuilt unit refines it up to observation,
   with no analysis of the first rebuilt unit. So are the units of
   FlambdaReaperUnboxingRefinesNeverShapesSpec.v (not stated: the fragment
   refuses them); the units of the examples of the reach fragments with a
   change of calling convention and that of [poison_reach_statement] are
   not.

   [settled_example_statement]: a unit that applies the closure of one of
   two codes that return their parameter at unknown arity, and returns the
   result, is settled, and refines its rebuilt unit by
   [settled_flow3_refines_cc_statement]. Settled units are few: the first
   rebuild is not idempotent (the first rebuild of the first rebuild of the
   unit of [never_ind_example_statement] drops the parameters of the
   handlers made [Invalid] and adds a wrapper), so that a unit with any
   code, binding or parameter that the rebuild drops or changes is not
   settled. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* Every element of [l] is one of [l']. *)
Definition list_within {A : Type} (eq_dec : ∀ a b : A, {a = b} + {a <> b})
    (l l' : list A) : bool :=
  forallb (fun a => if in_dec eq_dec a l' then true else false) l.

Definition node_pair_eq_dec : ∀ p q : node * node, {p = q} + {p <> q}.
Proof. decide equality; apply node_eq_dec. Defined.

Definition node_triple_eq_dec :
  ∀ p q : node * node * node, {p = q} + {p <> q}.
Proof. decide equality; [ apply node_eq_dec | apply node_pair_eq_dec ]. Defined.

Definition node_field_node_eq_dec :
  ∀ p q : node * field * node, {p = q} + {p <> q}.
Proof.
  decide equality; [ apply node_eq_dec | ].
  decide equality; [ apply field_eq_dec | apply node_eq_dec ].
Defined.

Definition node_nat_node_eq_dec :
  ∀ p q : node * nat * node, {p = q} + {p <> q}.
Proof.
  decide equality; [ apply node_eq_dec | ].
  decide equality; [ apply Nat.eq_dec | apply node_eq_dec ].
Defined.

(* Every fact of [F1] is a fact of [F]. *)
Definition facts_within (F1 F : facts) : bool :=
  list_within node_pair_eq_dec (f_alias F1) (f_alias F)
  && list_within node_pair_eq_dec (f_use F1) (f_use F)
  && list_within node_field_node_eq_dec (f_accessor F1) (f_accessor F)
  && list_within node_field_node_eq_dec (f_constructor F1)
       (f_constructor F)
  && list_within node_nat_node_eq_dec (f_argument F1) (f_argument F)
  && list_within node_nat_node_eq_dec (f_parameter F1) (f_parameter F)
  && list_within node_triple_eq_dec (f_propagate F1) (f_propagate F)
  && list_within node_triple_eq_dec (f_alias_if_any_source F1)
       (f_alias_if_any_source F)
  && list_within node_eq_dec (f_any_usage F1) (f_any_usage F)
  && list_within node_eq_dec (f_any_source F1) (f_any_source F)
  && list_within node_pair_eq_dec (f_any_source_if_used F1)
       (f_any_source_if_used F)
  && list_within node_eq_dec (f_zero_alloc_source F1)
       (f_zero_alloc_source F).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition within_flow3_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition within_flow3_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
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
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition within_flow3_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
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
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition within_never_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition within_never_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition within_never_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    facts_within
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e)
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The facts that the reach, [over_reach] and [rem_reach] read: the
   constructors and the inputs [any_source]. *)
Definition code_facts_within (F1 F : facts) : bool :=
  list_within node_field_node_eq_dec (f_constructor F1) (f_constructor F)
  && list_within node_eq_dec (f_any_source F1) (f_any_source F).

(* With [T] for the tables of the first rebuilt unit, the check of
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v holds where the facts that
   the reach reads are within the unit's. *)
Definition within_used_reach_covered_statement : Prop :=
  ∀ (F1 F : facts) (T : tables) (e : expr),
    code_facts_within F1 F = true -> used_reach_covered F T F1 T e = true.

(* ======================================================================== *)
(* The units that the first rebuild leaves as they are                     *)
(* ======================================================================== *)

Definition settled (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) : Prop :=
  first_rebuild flag u e T cannot_change preserve_direct_calls = e.

Definition settled_flow3_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition settled_flow3_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
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
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition settled_flow3_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
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
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition settled_never_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition settled_never_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition settled_never_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
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
    settled flag u e T cannot_change preserve_direct_calls ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* The unit of [never_ind_example_statement] is of the class: its first
   rebuild is in the fragment of the heap, its facts are within the
   unit's, and the statement of the fragment read off the unit's own
   analysis gives that its rebuilt unit refines it up to observation. *)
Definition within_example_statement : Prop :=
  in_heap_fragment NeverIndExample.first = true
  ∧ facts_within NeverIndExample.facts1 NeverIndExample.facts = true
  ∧ calls_unboxing_unit_never_nr NeverIndExample.nr NeverIndExample.facts
      NeverIndExample.tables NeverIndExample.cannot_change
      NeverIndExample.code_deps [] [] NeverIndExample.unit_expr
    = true
  ∧ program_refines_observably NeverIndExample.unit_expr
      (rebuild_unit_unbox_fixed NeverIndExample.facts NeverIndExample.tables
         NeverIndExample.cannot_change NeverIndExample.code_deps [] [] true
         NeverIndExample.unit_expr).

(* A unit that the first rebuild leaves as it is: the codes 50 and 51
   return their parameter; the unit applies the closure of one of them
   (chosen by a [Switch]) to 5 at unknown arity and returns the result.
   With the least solution, it is of the fragment read off its own
   analysis, with no code taken to never return, and it refines its
   rebuilt unit by the statement of settled units. *)
Module SettledExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition handler (k : continuation) (x : variable) (handler : expr)
    (body : expr) : expr :=
  Let_cont_non_recursive k
    (Handler [x] (make_handler_metadata [Value]) handler) body.

Definition call : apply_expr :=
  {| ap_callee := Var 31; ap_call_kind := Indirect_unknown_arity;
     ap_args := [int 5]; ap_continuation := 20; ap_exn_continuation := 2;
     ap_exn_extra_args := []; ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 (Code [40] 41 3 4 metadata (return_to 3 [Var 40]));
     Bind_code 51 (Code [42] 43 5 6 metadata (return_to 5 [Var 42]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (handler 20 30 (return_to 1 [Var 30])
             (handler 21 31 (Apply call)
                (Switch (Const (Naked_immediate 0))
                   [(0%Z, jump 21 [Var 60]); (1%Z, jump 21 [Var 61])]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

End SettledExample.

Definition settled_example_statement : Prop :=
  settled Auto FlambdaReaperUnboxing.Examples.interface
    SettledExample.unit_expr SettledExample.tables
    SettledExample.cannot_change true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) SettledExample.facts
      SettledExample.tables SettledExample.cannot_change
      SettledExample.code_deps [] [] SettledExample.unit_expr
    = true
  ∧ program_refines SettledExample.unit_expr
      (rebuild_unit_unbox_fixed SettledExample.facts SettledExample.tables
         SettledExample.cannot_change SettledExample.code_deps [] [] true
         SettledExample.unit_expr).
