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

(* The statements of FlambdaReaperUnboxingRefinesWithinSpec.v up to a
   renaming of the nodes that the traversal makes: the unit's own tables,
   renamed, serve as the tables of the first rebuilt unit, for the units
   whose first rebuild drops code, calls or primitives, so that the fresh
   nodes of what follows are numbered otherwise.

   This file is to be reviewed: the renaming ([renamed], [renaming_ok]),
   the check [facts_within_renamed], the tables [pull_tables], the
   statements and the example. The proofs are in
   FlambdaReaperUnboxingRefinesWithin2Proofs.v, which needs no review.

   The traversal numbers the nodes it makes ([Node_made]) in turn, so that
   a first rebuild that drops a call, a primitive or code makes the nodes
   of what follows under other numbers. A renaming [r] (a list of pairs,
   the first pair of a node giving its image; a node of no pair is its own
   image) takes the nodes of the traversal of the first rebuilt unit to
   those of the unit. [renaming_ok r e]: [r] renames only made nodes and
   variables that are not of [e] (the fresh names of the first rebuild), so
   that it keeps the code ids and the variables of the unit, at which the
   reach is read.

   [facts_within_renamed r F1 F]: every fact of [F1], its nodes renamed by
   [r], is a fact of [F]. The rules read the graph only in their premises
   and speak of no node by its name, so that a renaming that takes the
   facts of [F1] into those of [F] takes a solution of the rules on [F] to
   one on [F1]: the solution at the renamed nodes ([pull_tables r T], whose
   tuples are those of [T] with each node replaced by each node that [r]
   takes to it). Its reach, [over_reach] and [rem_reach] are within those
   of [F], [T] at the variables and code of the unit, as the reach reads
   the constructors and the inputs [any_source] only by membership.

   [within2_closed_statement]: [pull_tables r T] is closed under the rules
   on [F1], and [local_reads_used] holds of it, where [T] is on [F].
   [within2_used_reach_covered_statement]: [used_reach_covered] of
   FlambdaReaperUnboxingRefinesUnitFlowSpec.v, the premise of the
   statements of that file that compares the two analyses, holds of [F1]
   and [pull_tables r T].

   [renamed_flow3_refines_cc_statement],
   [renamed_flow3_observably_local_statement],
   [renamed_flow3_refines_observably_statement],
   [renamed_never_refines_cc_statement],
   [renamed_never_observably_local_statement] and
   [renamed_never_refines_observably_statement]: those of
   FlambdaReaperUnboxingRefinesWithinSpec.v with [facts_within_renamed r]
   for a renaming [r] with [renaming_ok] in place of [facts_within]. The
   earlier ones are the case where [r] is empty.

   [renamed_example_statement]: the unit of
   [never_ind_example_statement] with a call in a handler whose parameter
   has no source. The first rebuild makes the body of the handler
   [Invalid], so that the traversal of the first rebuilt unit makes the
   nodes of the calls that follow under smaller numbers: its facts are not
   within the unit's, but are up to the renaming given, and the statement
   of the fragment of code that never returns read off the unit's own
   analysis gives that its rebuilt unit refines it up to observation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesWithinSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The renaming                                                             *)
(* ======================================================================== *)

(* The image of [n]: that of the first pair of [n] in [r], if any. *)
Definition renamed (r : list (node * node)) (n : node) : node :=
  match find (fun p => node_eqb (fst p) n) r with
  | Some (_, m) => m
  | None => n
  end.

(* [r] renames only made nodes and variables that are not of [e]. *)
Definition renaming_ok (r : list (node * node)) (e : expr) : bool :=
  forallb
    (fun '(a, _) =>
       match a with
       | Node_made _ => true
       | Node_variable x => negb (among x (variables_of e))
       | Node_symbol _ | Node_code_id _ | Node_all_constants => false
       end)
    r.

(* The facts of [F], their nodes renamed. *)
Definition rename_facts (r : list (node * node)) (F : facts) : facts :=
  let rn := renamed r in
  {| f_alias := map (fun '(a, b) => (rn a, rn b)) (f_alias F);
     f_use := map (fun '(a, b) => (rn a, rn b)) (f_use F);
     f_accessor := map (fun '(a, f, b) => (rn a, f, rn b)) (f_accessor F);
     f_constructor :=
       map (fun '(a, f, b) => (rn a, f, rn b)) (f_constructor F);
     f_argument := map (fun '(a, i, b) => (rn a, i, rn b)) (f_argument F);
     f_parameter := map (fun '(a, i, b) => (rn a, i, rn b)) (f_parameter F);
     f_propagate :=
       map (fun '(a, b, c) => (rn a, rn b, rn c)) (f_propagate F);
     f_alias_if_any_source :=
       map (fun '(a, b, c) => (rn a, rn b, rn c)) (f_alias_if_any_source F);
     f_any_usage := map rn (f_any_usage F);
     f_any_source := map rn (f_any_source F);
     f_any_source_if_used :=
       map (fun '(a, b) => (rn a, rn b)) (f_any_source_if_used F);
     f_zero_alloc_source := map rn (f_zero_alloc_source F) |}.

(* Every fact of [F1], renamed by [r], is a fact of [F]. *)
Definition facts_within_renamed (r : list (node * node)) (F1 F : facts)
    : bool :=
  facts_within (rename_facts r F1) F.

(* The nodes that [r] takes to [n]. *)
Definition preimage (r : list (node * node)) (n : node) : list node :=
  filter (fun m => node_eqb (renamed r m) n) (n :: map fst r).

(* The tables at the renamed nodes: a tuple for each tuple of [T] and
   nodes that [r] takes to its nodes. *)
Definition pull_tables (r : list (node * node)) (T : tables) : tables :=
  let pre := preimage r in
  {| t_alias :=
       flat_map (fun '(a, b) => list_prod (pre a) (pre b)) (t_alias T);
     t_any_usage := flat_map pre (t_any_usage T);
     t_any_source := flat_map pre (t_any_source T);
     t_has_usage := flat_map pre (t_has_usage T);
     t_has_source := flat_map pre (t_has_source T);
     t_usages :=
       flat_map (fun '(a, b) => list_prod (pre a) (pre b)) (t_usages T);
     t_sources :=
       flat_map (fun '(a, b) => list_prod (pre a) (pre b)) (t_sources T);
     t_reading_field :=
       flat_map (fun '(f, n) => map (pair f) (pre n)) (t_reading_field T);
     t_escaping_field :=
       flat_map (fun '(f, n) => map (pair f) (pre n)) (t_escaping_field T);
     t_field_used :=
       flat_map (fun '(n, f) => map (fun m => (m, f)) (pre n))
         (t_field_used T) |}.

(* ======================================================================== *)
(* The tables of the first rebuilt unit                                     *)
(* ======================================================================== *)

Definition within2_closed_statement : Prop :=
  ∀ (is_local : field -> bool) (r : list (node * node)) (F1 F : facts)
    (T : tables),
    facts_within_renamed r F1 F = true ->
    check_closed is_local F T = true ->
    check_local_reads_used is_local F T = true ->
    closed is_local (graph_of_facts F1) (solution_of_tables (pull_tables r T))
    ∧ local_reads_used is_local (graph_of_facts F1)
        (solution_of_tables (pull_tables r T)).

Definition within2_used_reach_covered_statement : Prop :=
  ∀ (r : list (node * node)) (F1 F : facts) (T : tables) (e : expr),
    renaming_ok r e = true ->
    facts_within_renamed r F1 F = true ->
    used_reach_covered F T F1 (pull_tables r T) e = true.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition renamed_flow3_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool) (r : list (node * node)),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    renaming_ok r e = true ->
    facts_within_renamed r
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

Definition renamed_flow3_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool) (r : list (node * node)),
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
    renaming_ok r e = true ->
    facts_within_renamed r
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

Definition renamed_flow3_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool) (r : list (node * node)),
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
    renaming_ok r e = true ->
    facts_within_renamed r
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

Definition renamed_never_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (r : list (node * node)),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    renaming_ok r e = true ->
    facts_within_renamed r
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

Definition renamed_never_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (r : list (node * node)),
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
    renaming_ok r e = true ->
    facts_within_renamed r
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

Definition renamed_never_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (r : list (node * node)),
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
    renaming_ok r e = true ->
    facts_within_renamed r
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

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesNeverSpec.NeverIndExample with
   a call in the handler 20, which applies its parameter [x30] to 1 at
   unknown arity. [x30] has no source (the codes 50 and 51 raise), so the
   first rebuild makes the body of the handler [Invalid], and the
   traversal of the first rebuilt unit does not make the three nodes of
   the call: those it makes for the calls that follow come three numbers
   lower than in the traversal of the unit. *)
Module RenamedExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Module N := NeverIndExample.

Definition unit_expr : expr :=
  N.unit_with
    (fun body =>
       N.handler 20 30 (Apply (N.call 30 Indirect_unknown_arity [int 1] 1))
         (N.handler 24 34 (return_to 1 [Var 34]) body))
    (Apply (N.known_call None 20)) (Apply (N.unknown_call 24)).

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

Definition nr : code_id -> bool :=
  FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables tables)
    code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

(* The made nodes of the traversal of the first rebuilt unit that the
   renaming moves, each with that of the unit. *)
Definition renaming : list (node * node) :=
  map (fun '(a, b) => (Node_made a, Node_made b))
    [(15, 18); (16, 19); (17, 20); (18, 21); (19, 22); (20, 23)].

End RenamedExample.

Definition renamed_example_statement : Prop :=
  facts_within RenamedExample.facts1 RenamedExample.facts = false
  ∧ renaming_ok RenamedExample.renaming RenamedExample.unit_expr = true
  ∧ facts_within_renamed RenamedExample.renaming RenamedExample.facts1
      RenamedExample.facts
    = true
  ∧ calls_unboxing_unit_never_nr RenamedExample.nr RenamedExample.facts
      RenamedExample.tables RenamedExample.cannot_change
      RenamedExample.code_deps [] [] RenamedExample.unit_expr
    = true
  ∧ program_refines_observably RenamedExample.unit_expr
      (rebuild_unit_unbox_fixed RenamedExample.facts RenamedExample.tables
         RenamedExample.cannot_change RenamedExample.code_deps [] [] true
         RenamedExample.unit_expr).
