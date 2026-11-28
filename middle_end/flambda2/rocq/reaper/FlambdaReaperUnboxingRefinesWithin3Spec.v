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

(* The statements of FlambdaReaperUnboxingRefinesWithin2Spec.v for units
   whose first rebuild also gives poison to nodes with no use, makes calls
   direct and changes calling conventions: the unit's tables, with any
   source at the nodes with no use and renamed, serve as the tables of the
   first rebuilt unit.

   This file is to be reviewed: the tables [poisoned_tables], the check
   [analysis_within] (with [unit_poison_ok], [poison_constructor_ok],
   [poison_argument_ok] and [facts_poison_ok]), the statements and the
   examples. The proofs are in FlambdaReaperUnboxingRefinesWithin3Proofs.v,
   which needs no review.

   What the first rebuild adds to the facts of the traversal, beyond the
   renaming of the made nodes:
   - Poison. The rebuild gives poison, a constant ([Node_all_constants],
     which has any source), in place of a variable with no use, and of the
     argument of a call whose parameter has no use. So a fact of the unit
     with a variable [y] becomes one with the constants: an alias into a
     parameter of a continuation or a result of code, a field of a block,
     an argument of a call. No renaming takes it to a fact of the unit
     ([poison_reach_statement]). The rules then alias to the constants
     the nodes that read the poison, and give them any source; these
     have no use, as the reader of a node with no use has none
     (FlambdaReaperGraphSpec.has_usage_alias, [has_usage_accessor],
     [any_usage_use]), so that the reach at the variables with a use is
     unchanged. [poisoned_tables F T] is [T] with any source, a source,
     and an alias to the constants at every node with no use
     ([unused_nodes]: those of the facts and of the aliases of [T] that
     [T] gives no use).
   - Calls made direct. A call of known arity whose code is known becomes
     a direct call, which with [Auto] the traversal gives two fresh nodes,
     the aliases of the call widget and of the callee on any source of the
     callee, and the aliases of the apply dep (the fresh node of the
     witness to the witness of known arity of the code, [my_closure] of
     the code to the fresh node of the closure). These are aliases, which
     the rules read only in [alias_input], [alias_propagate] and
     [alias_if_any_source]: a renaming that takes the fresh node of the
     witness to the node of the call of the unit, and that of the closure
     to the closure, makes them aliases that the unit's tables have.
   - Changes of calling convention. A parameter or a result deleted is a
     node that the traversal of the first rebuilt unit does not make, and
     an argument it does not give: fewer facts, the nodes that follow
     renamed.

   The check, [analysis_within is_local r F1 F T], of the facts [F1] of
   the first rebuilt unit, renamed by [r], against the facts [F] and the
   tables [T] of the unit:
   - [unit_poison_ok]: the constants have any usage in [T]; at the nodes
     with no use, which [poisoned_tables] gives any source, the rules on
     [F] ask nothing more (an alias on any source of such a node holds in
     [poisoned_tables F T]; such a node given as an argument goes to a
     call with a use, and is not a call given arguments; a node that holds
     one in a field has a source; no local field is read from one).
   - [facts_poison_ok]: each fact of [F1], renamed, is one of [F], or one
     that [poisoned_tables F T] satisfies (an alias, a use, an alias on a
     usage or on any source, an input [any_usage] or
     [any_source_if_used]), or poison: a field of a block or an argument
     of a call that [F] has with another node, whose readers through the
     uses and sources of [T] (resp. the parameters) have no use; the
     constructors, accessors, parameters and inputs [any_source], which
     the reach reads, are those of [F] but for poison in real fields.

   [analysis_within_closed_statement]: where the check holds,
   [pull_tables r (poisoned_tables F T)] is closed under the rules on
   [F1], and [local_reads_used] holds of it.
   [analysis_within_used_reach_covered_statement]: [used_reach_covered]
   holds of [F1] and these tables, at the variables with a use in [T],
   which [poisoned_tables] leaves as they are. The statements of
   FlambdaReaperUnboxingRefinesWithin2Spec.v follow, with
   [analysis_within] in place of [facts_within_renamed]:
   [analysis_within_flow3_refines_cc_statement],
   [analysis_within_flow3_observably_local_statement],
   [analysis_within_flow3_refines_observably_statement],
   [analysis_within_never_refines_cc_statement],
   [analysis_within_never_observably_local_statement] and
   [analysis_within_never_refines_observably_statement].

   [poison_example_statement]: the unit of [poison_reach_statement], whose
   first rebuild gives the call poison, is of the class with no renaming
   and of none of the earlier ones, and refines its rebuilt unit up to
   observation. [direct_example_statement]: a unit whose first rebuild
   makes a call direct and deletes the parameter of the code, so that the
   call gives no argument, is of the class with the renaming given and of
   none of the earlier ones, and refines its rebuilt unit up to
   observation.

   [wrapper_example_statement]: the unit of [flow_example_statement],
   whose first rebuild gives poison to the results of 50 and 51, makes the
   call of 52 return to a wrapper and changes the calling convention of
   52, is of the class with the renaming given, which takes the parameter
   of the wrapper, a fresh variable, to that of the continuation of the
   call, and refines its rebuilt unit up to observation.

   Not covered: a parameter deleted before a kept one, which renumbers the
   parameters and arguments that follow (their number is not a node); and
   closures of code deleted, whose witness the traversal makes fresh, with
   any source and two constructors that no renaming takes to the unit's. *)

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
  FlambdaReaperUnboxingRefinesWithinSpec
  FlambdaReaperUnboxingRefinesWithin2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The unit's tables with poison                                            *)
(* ======================================================================== *)

(* The nodes of the facts. *)
Definition nodes_of_facts (F : facts) : list node :=
  flat_map (fun '(a, b) => [a; b]) (f_alias F)
  ++ flat_map (fun '(a, b) => [a; b]) (f_use F)
  ++ flat_map (fun '(a, _, b) => [a; b]) (f_accessor F)
  ++ flat_map (fun '(a, _, b) => [a; b]) (f_constructor F)
  ++ flat_map (fun '(a, _, b) => [a; b]) (f_argument F)
  ++ flat_map (fun '(a, _, b) => [a; b]) (f_parameter F)
  ++ flat_map (fun '(a, b, c) => [a; b; c]) (f_propagate F)
  ++ flat_map (fun '(a, b, c) => [a; b; c]) (f_alias_if_any_source F)
  ++ f_any_usage F ++ f_any_source F
  ++ flat_map (fun '(a, b) => [a; b]) (f_any_source_if_used F)
  ++ f_zero_alloc_source F.

(* The nodes with no use among those that the aliases of [T] start from
   and those of [F]. *)
Definition unused_nodes (F : facts) (T : tables) : list node :=
  filter (fun n => negb (mem_node n (t_has_usage T)))
    (map fst (t_alias T) ++ nodes_of_facts F).

(* [T] where every node with no use has any source and may be a
   constant: what poison, a constant given to a node with no use, adds. *)
Definition poisoned_tables (F : facts) (T : tables) : tables :=
  let U := unused_nodes F T in
  {| t_alias := t_alias T ++ map (fun n => (n, Node_all_constants)) U;
     t_any_usage := t_any_usage T;
     t_any_source := t_any_source T ++ U;
     t_has_usage := t_has_usage T;
     t_has_source := t_has_source T ++ U;
     t_usages := t_usages T;
     t_sources := t_sources T;
     t_reading_field := t_reading_field T;
     t_escaping_field := t_escaping_field T;
     t_field_used := t_field_used T |}.

(* Membership, by a decision of equality. *)
Definition inb {A : Type} (eq_dec : ∀ a b : A, {a = b} + {a <> b}) (a : A)
    (l : list A) : bool :=
  if in_dec eq_dec a l then true else false.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Section Check.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.

Local Abbreviation P := (solution_of_tables (poisoned_tables F T)).
Local Abbreviation unused := (fun n => mem_node n (unused_nodes F T)).
Local Abbreviation consts := Node_all_constants.

(* Where a node with no use has any source, the rules give no more than
   [poisoned_tables F T] at the facts of [F]: an alias on any source of it
   holds; a node with no use given as an argument goes to a call with a
   use, and no argument is given to such a node; a node that holds one
   with no use in a field has a source; no local field is read from a
   node with no use. *)
Definition unit_poison_ok : bool :=
  mem_node consts (t_any_usage T)
  && forallb (fun '(c, t, f) => negb (unused c) || s_alias P t f)
       (f_alias_if_any_source F)
  && forallb
       (fun '(f, _, b) =>
          (negb (unused f) || s_has_usage P b)
          && (negb (unused b) || s_any_usage P f))
       (f_argument F)
  && forallb (fun '(b, _, f) => negb (unused f) || s_has_source P b)
       (f_constructor F)
  && forallb (fun '(_, relation, b) => negb (unused b && is_local relation))
       (f_accessor F).

(* A field [relation] of [b] given poison, where [F] has it: whatever the
   rules alias to the field through its uses and sources has no use, [b]
   has a source, and a local field with any usage escapes with the
   constants. *)
Definition poison_constructor_ok (b : node) (relation : field) (f : node)
    : bool :=
  node_eqb f consts && is_real_field relation
  && existsb (fun '(b', relation', _) =>
                node_eqb b' b && field_eqb relation' relation)
       (f_constructor F)
  && forallb
       (fun '(t, relation', base) =>
          negb (field_eqb relation' relation
                && ((s_usages P b base && negb (s_any_usage P b))
                    || (is_local relation && s_usages P b base)
                    || (s_sources P base b && negb (s_any_source P base))
                    || (is_local relation && s_sources P base b)))
          || s_alias P t consts)
       (f_accessor F)
  && s_has_source P b
  && (negb (s_any_usage P b && is_local relation)
      || s_escaping_field P relation consts).

(* An argument number [i] of the call [w] given poison, where [F] has
   one: the call has a use, and the parameters that the rules alias to it
   have none. *)
Definition poison_argument_ok (f : node) (i : nat) (w : node) : bool :=
  node_eqb f consts
  && existsb (fun '(_, i', w') => Nat.eqb i' i && node_eqb w' w)
       (f_argument F)
  && s_has_usage P w
  && forallb
       (fun '(b, i', t) =>
          negb (Nat.eqb i' i
                && ((s_usages P b w && negb (s_any_usage P b))
                    || (s_sources P w b && negb (s_any_source P w))))
          || s_alias P t consts)
       (f_parameter F).

(* Every fact of [F1] is one of [F], or poison as above, or a fact that
   [poisoned_tables F T] already satisfies (an alias, a use, an alias on
   a use or on any source, a usage); the constructors, accessors,
   parameters and inputs [any_source], which the reach reads, are those
   of [F] but for poison. *)
Definition facts_poison_ok (F1 : facts) : bool :=
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
          || poison_constructor_ok b relation f)
       (f_constructor F1)
  && forallb
       (fun '(f, i, w) =>
          inb node_nat_node_eq_dec (f, i, w) (f_argument F)
          || poison_argument_ok f i w)
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

(* The analysis of [F1], renamed by [r], is within that of [F], [T] up to
   poison. *)
Definition analysis_within (is_local : field -> bool) (r : list (node * node))
    (F1 F : facts) (T : tables) : bool :=
  unit_poison_ok is_local F T
  && facts_poison_ok is_local F T (rename_facts r F1).

(* ======================================================================== *)
(* The tables of the first rebuilt unit                                     *)
(* ======================================================================== *)

Definition analysis_within_closed_statement : Prop :=
  ∀ (is_local : field -> bool) (r : list (node * node)) (F1 F : facts)
    (T : tables),
    analysis_within is_local r F1 F T = true ->
    check_closed is_local F T = true ->
    check_local_reads_used is_local F T = true ->
    closed is_local (graph_of_facts F1)
      (solution_of_tables (pull_tables r (poisoned_tables F T)))
    ∧ local_reads_used is_local (graph_of_facts F1)
        (solution_of_tables (pull_tables r (poisoned_tables F T))).

Definition analysis_within_used_reach_covered_statement : Prop :=
  ∀ (is_local : field -> bool) (r : list (node * node)) (F1 F : facts)
    (T : tables) (e : expr),
    renaming_ok r e = true ->
    analysis_within is_local r F1 F T = true ->
    used_reach_covered F T F1 (pull_tables r (poisoned_tables F T)) e = true.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition analysis_within_flow3_refines_cc_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition analysis_within_flow3_observably_local_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition analysis_within_flow3_refines_observably_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_observably e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition analysis_within_never_refines_cc_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
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

Definition analysis_within_never_observably_local_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
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

Definition analysis_within_never_refines_observably_statement : Prop :=
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
    analysis_within is_local r
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      (traverse_unit flag u e) T
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
(* Examples                                                                 *)
(* ======================================================================== *)

(* The unit of [poison_reach_statement]: its first rebuild gives the call
   poison, which no renaming takes to a fact of the unit, as the unit
   gives no constant as an argument. Its analysis is within that of the
   unit up to poison, and the statement of the fragment of
   FlambdaReaperUnboxingRefinesReach3Spec.v read off the unit's own
   analysis gives that its rebuilt unit refines it up to observation. *)
Definition poison_example_statement : Prop :=
  (∀ r : list (node * node),
     renaming_ok r PoisonReach.unit_expr = true ->
     facts_within_renamed r PoisonReach.facts1 PoisonReach.facts = false)
  ∧ analysis_within (fun _ => false) [] PoisonReach.facts1 PoisonReach.facts
      PoisonReach.tables
    = true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) PoisonReach.facts
      PoisonReach.tables PoisonReach.cannot_change
      (unit_code_deps Auto FlambdaReaperUnboxing.Examples.interface
         PoisonReach.unit_expr)
      [] [] PoisonReach.unit_expr
    = true
  ∧ program_refines_observably PoisonReach.unit_expr
      (rebuild_unit_unbox_fixed PoisonReach.facts PoisonReach.tables
         PoisonReach.cannot_change
         (unit_code_deps Auto FlambdaReaperUnboxing.Examples.interface
            PoisonReach.unit_expr)
         [] [] true PoisonReach.unit_expr).

(* A unit that applies the closure [x60] of the code 50 at known arity to
   5, and returns the result; 50 returns 0, and does not use its
   parameter. The first rebuild makes the call direct and changes the
   calling convention of 50, which loses its parameter, so that the call
   gives no argument. The traversal of the first rebuilt unit gives the
   direct call, with [Auto], the alias of its witness on any source of
   [x60] and the aliases of the apply dep, into two fresh nodes, and makes
   the nodes of 50 and of the call that follow the parameter one number
   lower. Its analysis is within that of the unit up to the renaming
   given, which takes the fresh node of the closure to [x60] and that of
   the witness to the node of the call of the unit, which the unit's
   tables alias to the witness of known arity of 50 (and [my_closure] of
   50 to [x60]). *)
Module DirectExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition call : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Indirect_known_arity None;
     ap_args := [int 5]; ap_continuation := 20; ap_exn_continuation := 2;
     ap_exn_extra_args := []; ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 50 (Code [40] 41 3 4 metadata (return_to 3 [int 0]))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (SettledExample.handler 20 30 (return_to 1 [Var 30]) (Apply call))).

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

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition renaming : list (node * node) :=
  [(Node_made 6, Node_made 7); (Node_made 7, Node_made 8);
   (Node_made 8, Node_made 9); (Node_made 9, Node_made 10);
   (Node_made 10, Node_made 11); (Node_made 11, Node_variable 60);
   (Node_made 12, Node_made 11)].

End DirectExample.

Definition direct_example_statement : Prop :=
  (∀ r : list (node * node),
     facts_within_renamed r DirectExample.facts1 DirectExample.facts = false)
  ∧ renaming_ok DirectExample.renaming DirectExample.unit_expr = true
  ∧ analysis_within (fun _ => false) DirectExample.renaming
      DirectExample.facts1 DirectExample.facts DirectExample.tables
    = true
  ∧ calls_unboxing_unit_flow3_nr (fun _ => false) DirectExample.facts
      DirectExample.tables DirectExample.cannot_change DirectExample.code_deps
      [] [] DirectExample.unit_expr
    = true
  ∧ program_refines_observably DirectExample.unit_expr
      (rebuild_unit_unbox_fixed DirectExample.facts DirectExample.tables
         DirectExample.cannot_change DirectExample.code_deps [] [] true
         DirectExample.unit_expr).

(* The renaming of the unit of [flow_example_statement]: the nodes of 52,
   which loses its parameter and its result, and those of the calls that
   follow, and the parameter [x63] of the wrapper, to [x30]. *)
Module WrapperExample.

Local Open Scope positive_scope.

Definition renaming : list (node * node) :=
  map (fun '(a, b) => (Node_made a, Node_made b))
    [(15, 16); (16, 17); (17, 19); (18, 20); (19, 21); (20, 22); (21, 23);
     (22, 24); (23, 25); (24, 26); (25, 27); (26, 28)]
  ++ [(Node_variable 63, Node_variable 30)].

End WrapperExample.

Definition wrapper_example_statement : Prop :=
  facts_within_renamed WrapperExample.renaming FlowExample.facts1
    FlowExample.facts
  = false
  ∧ renaming_ok WrapperExample.renaming FlowExample.unit_expr = true
  ∧ analysis_within (fun _ => false) WrapperExample.renaming
      FlowExample.facts1 FlowExample.facts FlowExample.tables
    = true
  ∧ program_refines_observably FlowExample.unit_expr
      (rebuild_unit_unbox_fixed FlowExample.facts FlowExample.tables
         FlowExample.cannot_change FlowExample.code_deps [] [] true
         FlowExample.unit_expr).
