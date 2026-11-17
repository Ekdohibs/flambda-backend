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

(* Local fields: what goes wrong, and the rule that fixes it.

   This file is to be reviewed: the rule ([local_reads_used]), against
   the fix of middle_end/flambda2/reaper/points_to_analysis.ml (the last
   rule of [local_rules], and the last paragraph of the explanation of
   local fields at the top of that file), the two examples and the
   statements. The checks and the solver are
   FlambdaReaperLocalFields.v, which needs no review; the proofs are in
   FlambdaReaperLocalFieldsProofs.v.

   What goes wrong. A field is local when it is a value or a function
   slot of the unit, with [-reaper-local-fields]. The rules of
   points_to_analysis.ml give a local field neither any usage, when it
   is stored in a block with any usage, nor any source, when it is read
   from a block with any source; they find the reads of an allocation
   [a] that holds [u] in a local field [f], made from a node [b] into
   [v], by cases on how [a] reaches [b]: [b] is among the usages of
   [a] ([alias_local_constructor_usage]), or [a] among the sources of
   [b] ([alias_local_accessor_source]), or else [a] has any usage and
   [b] any source, and then [f] escapes with [u] and is read into [v]
   ([escaping_field_rule], [reading_field_rule],
   [alias_escaping_reading]). Each case gives the alias [let v = u].
   The cases leave one out. A value reaches the result of a [use] (a
   primitive of the default case of [traverse_prim]: [Array_load] from
   an immutable array, [Opaque_identity], ...) without an alias; the
   result has any source as soon as the argument has a source
   ([any_source_use]), but the argument has any usage only once the
   result has a usage ([any_usage_use]). If [b] is such a result, or
   comes from one, and nothing uses [v], nothing uses [b] either: [a]
   has no usage, [b] is not among its usages, [b] has any source and
   so no sources, and [v] gets [reading_field] and nothing else. With
   no source, [v] is taken for a variable whose binding is never
   reached, and the rebuild replaces by [Invalid] what passes it on,
   which the unit runs (FlambdaReaperFlowSpec.v,
   [local_slot_counterexample_statement]). The compiler does the same:
   a closure put in [[: clo :]] and read back with [Iarray.get], whose
   call is inlined, and the slot it reads passed to a function that
   ignores it, runs [Invalid] at [-O4] ("Dead variable in bound
   pattern"). So does a closure put in a pair, the pair in the array.

   Which rule. [v] needs a source, not a usage: nothing reads it, and
   the field may go from the allocation and the read with it. The case
   that should have applied is the third: [b] has any source. It needs
   [a] to have any usage, which [any_usage_use] would give if [b] had a
   usage. The fix makes a read of a local field from a node with any
   source a usage of that node:

     any_source base, accessor ~to_ relation ~base, is_local relation
     ==> has_usage base

   in points_to_analysis.ml the rule with the premises of
   [reading_field] and the conclusion [has_usage base], the last of
   [local_rules]. From [b], [has_usage] goes back along aliases and
   reads ([has_usage_alias], [has_usage_accessor]) to the result of the
   [use], which gives the argument any usage ([any_usage_use]), and any
   usage goes back along aliases and fields that are not local
   ([any_usage_alias], [any_usage_constructor]) to [a]: the third case
   applies. The other way, making [f] not local where [a] reaches a
   [use], would need the rules to read whether a field is local off
   what they derive, and negate it where the field is local; here
   [is_local] is fixed before the rules run, as
   [Analysis_scope.is_local_field] is. A seed of the traversal would
   only approximate the rule: [alias_if_any_source] can make an alias
   conditional on [any_source base], but the graph has no node to alias
   [base] to that has a usage and nothing more. A node with any usage
   would make every allocation that reaches [base] escape, whether it
   came by a [use] or not; a node read from, as the keep-alive of
   tupled code reads [Is_int] (FlambdaReaperTupledSpec.v), would also
   make the field read used in every allocation that [base] may come
   from.

   In the model the fix is a closure condition beside [closed]
   ([local_reads_used], and [closed_local] for both): the rule only
   adds tuples of [has_usage], so the tables that the fixed rules give
   are closed under the rules before the fix, and every statement that
   asks [check_closed] holds of them. [check_closed_local] tests both
   ([check_closed_local_sound_statement]). Without local fields the
   rule says nothing ([local_reads_used_no_local_statement]), and where
   the read of a local field is used, its node has a usage already: the
   example of FlambdaReaperSolutionSpec.v, whose value slot is local
   and read by a read that is used, passes the new check as it is
   ([example_local_reads_used_statement]).

   The fix pays where the read of a local field is not used, from a
   node with any source that nothing else uses: the node keeps its
   binding, and what reaches it by a [use] has any usage. It is what
   the third case asks for: the allocation does escape into the
   container, which the analysis does not follow.

   The examples. On the unit of [local_slot_counterexample_statement],
   the least solution of the rules before the fix fails the new check,
   and the least solution of the fixed rules ([LocalSlotFixed]), which
   the solver of FlambdaReaperLocalFields.v finds, passes it and every
   check that the statement of the rebuild asks; the rebuilt unit exits
   as the unit does ([local_slot_fixed_statement]). The closure now has
   any usage, its value slot escapes with the constants and [x13] has
   them for source; the rebuild keeps the closure, without its slot,
   the array and the load, and removes the read. The second unit
   ([NestedSlot]) puts the closure in a block and the block in the
   array, and reads the block's field before the slot: the usage goes
   back through a read of a field that is not local and two [use]s.
   Its least solution before the fix passes [check_closed] and the
   rebuild goes wrong; that of the fixed rules passes and the rebuild
   exits as the unit does ([nested_slot_statement]).

   The statements of the flow and of the rebuild with local fields
   (the [_local] statements of FlambdaReaperFlowSpec.v and after) ask
   [check_local_reads_used] in place of no local field or no
   projection, which gives it
   (FlambdaReaperFlowSpec.old_premise_local_reads_used_statement). The
   step of a projection of a local field uses
   [local_read_alias_statement]; the rules before the fix do not give
   it ([local_read_alias_needs_fix_statement], on the counterexample).
   A read that the rebuild keeps finds its field used in every
   allocation that it may come from without the fix
   ([local_read_field_used_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLocalFields.
Import ListNotations.

(* ======================================================================== *)
(* The rule                                                                 *)
(* ======================================================================== *)

Section Rule.

Variable is_local : field -> bool.
Variable G : graph.
Variable S : solution.

(* A read of a local field from a node with any source is a usage of
   that node. *)
Definition local_reads_used : Prop :=
  ∀ base relation to_,
    s_any_source S base = true -> g_accessor G to_ relation base ->
    is_local relation = true -> s_has_usage S base = true.

(* The rules with the fix. *)
Definition closed_local : Prop := closed is_local G S ∧ local_reads_used.

End Rule.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

Definition check_local_reads_used_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    check_local_reads_used is_local F T = true ->
    local_reads_used is_local (graph_of_facts F) (solution_of_tables T).

Definition check_closed_local_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    check_closed_local is_local F T = true ->
    closed_local is_local (graph_of_facts F) (solution_of_tables T).

(* Without local fields, the rules with the fix are those without. *)
Definition local_reads_used_no_local_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (S : solution),
    (∀ f : field, is_local f = false) -> local_reads_used is_local G S.

(* The example of FlambdaReaperSolutionSpec.v, whose local value slot is
   read by a read that is used, passes the check of the fix. *)
Definition example_local_reads_used_statement : Prop :=
  check_local_reads_used Example.is_local Example.graph Example.solution
  = true.

(* ======================================================================== *)
(* The counterexample, fixed                                                *)
(* ======================================================================== *)

(* The least solution of the fixed rules on the unit of
   [local_slot_counterexample_statement]: besides the tuples of the
   least solution before the fix, [x12] has a usage, which makes [x11]
   and [x10] have any usage; the slot 60 escapes with the constants,
   which [x13] reads, and the code 20 is used, with the fields of the
   witnesses of the closure. *)
Module LocalSlotFixed.

Local Open Scope positive_scope.

Definition v (n : positive) : node := Node_variable n.
Definition m (n : positive) : node := Node_made n.

Definition facts : FlambdaReaperGraphSpec.facts :=
  traverse_unit Never LocalSlotCounterexample.interface
    LocalSlotCounterexample.unit_expr.

Definition solution : tables :=
  {| t_alias :=
       [(m 5, v 10); (v 13, Node_all_constants); (v 30, m 6); (v 31, m 5);
        (m 3, v 30); (m 1, Node_all_constants); (v 14, v 13)];
     t_any_usage :=
       [m 6; v 30; Node_code_id 20; m 3; m 4; m 8; m 7; v 10; v 11;
        Node_all_constants; m 2; m 1];
     t_any_source :=
       [v 14; m 3; v 30; v 13; m 6; m 1; Node_all_constants; Node_code_id 20;
        v 11; v 12];
     t_has_usage :=
       [m 6; v 30; m 4; m 3; Node_code_id 20; m 7; m 8; v 10; v 11;
        Node_all_constants; v 12; m 1; m 2];
     t_has_source :=
       [v 31; m 3; v 14; m 5; v 30; v 13; m 6; m 7; m 8; m 1; v 12; v 11;
        Node_code_id 20; Node_all_constants; v 10];
     t_usages := [(v 12, v 12)];
     t_sources :=
       [(v 31, v 10); (m 5, v 10); (m 7, m 7); (m 8, m 8); (v 10, v 10)];
     t_reading_field := [(Field_value_slot 60, v 13)];
     t_escaping_field := [(Field_value_slot 60, Node_all_constants)];
     t_field_used :=
       [(m 7, Field_code_id_of_call_witness);
        (m 7, Field_return_of_call Exn);
        (m 7, Field_return_of_call (Normal 0));
        (m 8, Field_code_id_of_call_witness);
        (m 8, Field_return_of_call (Normal 0));
        (m 8, Field_return_of_call Exn); (v 10, Field_function_slot 40);
        (v 10, Field_call_witness Unknown_arity_code_pointer);
        (v 10, Field_call_witness Known_arity_code_pointer)] |}.

End LocalSlotFixed.

Definition local_slot_fixed_statement : Prop :=
  check_local_reads_used LocalSlotCounterexample.is_local
    LocalSlotFixed.facts LocalSlotCounterexample.solution
  = false
  ∧ compute_least_local LocalSlotCounterexample.is_local LocalSlotFixed.facts
      50
    = Some LocalSlotFixed.solution
  ∧ check_closed_local LocalSlotCounterexample.is_local LocalSlotFixed.facts
      LocalSlotFixed.solution
    = true
  ∧ check_field_used_exact LocalSlotCounterexample.is_local
      LocalSlotFixed.facts LocalSlotFixed.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables LocalSlotFixed.solution ->
       check_closed_local LocalSlotCounterexample.is_local
         LocalSlotFixed.facts T
       = false)
  ∧ run 10
      (State LocalSlotCounterexample.unit_expr [] [] empty_static_env
         empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads
  ∧ run 10
      (State
         (rebuild_unit_from_tables LocalSlotFixed.facts
            LocalSlotFixed.solution false LocalSlotCounterexample.unit_expr)
         [] [] empty_static_env empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads.

(* ======================================================================== *)
(* A read of a local field, with the fix                                    *)
(* ======================================================================== *)

(* What the flow invariant needs of a read of a local field: what an
   allocation that the value read from may come from holds in the field
   is where the read puts it. With the fix, the three cases of the rules
   cover every allocation that [may_come_from] allows. *)
Definition local_read_alias_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (S : solution)
    (a base to_ from : node) (f : field),
    closed_local is_local G S -> is_local f = true ->
    g_constructor G a f from -> g_accessor G to_ f base ->
    may_come_from S a base -> s_alias S to_ from = true.

(* Without the fix it does not hold: on the unit of
   [local_slot_counterexample_statement], the closure [x10] may be at
   [x12], from which the slot 60 is read into [x13], and the least
   solution of the rules before the fix has no alias of [x13] to the
   constants that the slot holds. *)
Definition local_read_alias_needs_fix_statement : Prop :=
  closed LocalSlotCounterexample.is_local
    (graph_of_facts LocalSlotFixed.facts)
    (solution_of_tables LocalSlotCounterexample.solution)
  ∧ may_come_from (solution_of_tables LocalSlotCounterexample.solution)
      (LocalSlotFixed.v 10) (LocalSlotFixed.v 12)
  ∧ g_constructor (graph_of_facts LocalSlotFixed.facts) (LocalSlotFixed.v 10)
      (Field_value_slot 60%positive) Node_all_constants
  ∧ g_accessor (graph_of_facts LocalSlotFixed.facts) (LocalSlotFixed.v 13)
      (Field_value_slot 60%positive) (LocalSlotFixed.v 12)
  ∧ s_alias (solution_of_tables LocalSlotCounterexample.solution)
      (LocalSlotFixed.v 13) Node_all_constants
    = false.

(* What the rebuild needs of a read of a local field that it keeps: the
   field is used in every allocation that the value read from may come
   from. This needs no fix: the read has a usage, which its node has
   then too. *)
Definition local_read_field_used_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (S : solution)
    (a base to_ from : node) (f : field),
    closed is_local G S -> is_local f = true ->
    g_constructor G a f from -> g_accessor G to_ f base ->
    may_come_from S a base -> s_has_usage S to_ = true ->
    s_field_used S a f = true.

(* ======================================================================== *)
(* A closure in a block in the array                                        *)
(* ======================================================================== *)

(* The closure goes into a block [x15], the block into the array, and
   the block loaded back has its field 0 read before the slot:

     let code 20 (x30) my_closure x31 = return x30 in
     let x10 = closure 20 { 60 = 5 } in
     let x15 = { x10 } in
     let x11 = [| x15 |] in
     let x12 = x11.(0) in
     let x16 = x12.0 in
     let x13 = x16.60 in
     let_cont k5 x14 = return 0 in
     k5 x13

   The field 0 of [x15] is not local: [x16] has any source, read from
   [x12], which has any source. Before the fix nothing has a usage
   ([solution]); with it, [x16] has one, so [x12] has one, [x11] and
   [x15] have any usage, and so has [x10], through the field 0 of
   [x15] ([fixed_solution]). *)
Module NestedSlot.

Local Open Scope positive_scope.

Definition v (n : positive) : node := Node_variable n.
Definition m (n : positive) : node := Node_made n.

Definition unit_expr : expr :=
  Let_static [Bind_code 20 LocalSlotCounterexample.code20]
    (Let 10 Value
       (Closure_of_set 40 [(40, 20)] [(60, Const (Tagged_immediate 5))])
       (Let 15 Value (FlowExample.make_block [Var 10])
          (Let 11 Value
             (Prim (Variadic (Make_immutable_array Array_values)) [Var 15])
             (Let 12 Value
                (Prim (Binary (Array_load Array_values Load_values Immutable))
                   [Var 11; Const (Tagged_immediate 0)])
                (Let 16 Value (FlowExample.load 0 (Var 12))
                   (Let 13 Value (Project_value_slot_from 40 60 (Var 16))
                      (Let_cont_non_recursive 5
                         (Handler [14] (make_handler_metadata [Value])
                            (LocalSlotCounterexample.return_to 1
                               [Const (Tagged_immediate 0)]))
                         (LocalSlotCounterexample.return_to 5
                            [Var 13])))))))).

Definition facts : FlambdaReaperGraphSpec.facts :=
  traverse_unit Never LocalSlotCounterexample.interface unit_expr.

(* The least solution of the rules before the fix. *)
Definition solution : tables :=
  {| t_alias :=
       [(v 30, m 6); (v 31, m 5); (m 3, v 30); (m 1, Node_all_constants);
        (v 14, v 13)];
     t_any_usage := [Node_all_constants; m 2; m 1];
     t_any_source :=
       [v 16; m 1; Node_all_constants; Node_code_id 20; v 11; v 12];
     t_has_usage := [Node_all_constants; m 1; m 2];
     t_has_source :=
       [m 7; m 8; m 1; v 16; v 15; v 12; v 11; Node_code_id 20;
        Node_all_constants; v 10];
     t_usages := [(v 12, v 12); (v 16, v 16)];
     t_sources := [(m 7, m 7); (m 8, m 8); (v 15, v 15); (v 10, v 10)];
     t_reading_field := [(Field_value_slot 60, v 13)];
     t_escaping_field := [];
     t_field_used := [] |}.

(* The least solution of the fixed rules. *)
Definition fixed_solution : tables :=
  {| t_alias :=
       [(m 5, v 10); (v 13, Node_all_constants); (v 30, m 6); (v 31, m 5);
        (m 3, v 30); (m 1, Node_all_constants); (v 14, v 13)];
     t_any_usage :=
       [m 6; v 30; Node_code_id 20; m 3; m 4; m 8; m 7; v 10; v 15; v 11;
        Node_all_constants; m 2; m 1];
     t_any_source :=
       [v 14; m 3; v 30; v 13; m 6; v 16; m 1; Node_all_constants;
        Node_code_id 20; v 11; v 12];
     t_has_usage :=
       [m 6; v 30; m 4; m 3; Node_code_id 20; m 7; m 8; v 10; v 15; v 11;
        v 12; v 16; Node_all_constants; m 1; m 2];
     t_has_source :=
       [v 31; m 3; v 14; m 5; v 30; v 13; m 6; m 7; m 8; m 1; v 16; v 15;
        v 12; v 11; Node_code_id 20; Node_all_constants; v 10];
     t_usages := [(v 12, v 12); (v 16, v 16)];
     t_sources :=
       [(v 31, v 10); (m 5, v 10); (m 7, m 7); (m 8, m 8); (v 15, v 15);
        (v 10, v 10)];
     t_reading_field := [(Field_value_slot 60, v 13)];
     t_escaping_field := [(Field_value_slot 60, Node_all_constants)];
     t_field_used :=
       [(m 7, Field_code_id_of_call_witness);
        (m 7, Field_return_of_call Exn);
        (m 7, Field_return_of_call (Normal 0));
        (m 8, Field_code_id_of_call_witness);
        (m 8, Field_return_of_call (Normal 0));
        (m 8, Field_return_of_call Exn); (v 10, Field_function_slot 40);
        (v 10, Field_call_witness Unknown_arity_code_pointer);
        (v 10, Field_call_witness Known_arity_code_pointer);
        (v 15, Field_get_tag); (v 15, Field_is_int);
        (v 15, Field_block 0%nat Value)] |}.

End NestedSlot.

Definition nested_slot_statement : Prop :=
  in_heap_fragment NestedSlot.unit_expr = true
  ∧ check_closed LocalSlotCounterexample.is_local NestedSlot.facts
      NestedSlot.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables NestedSlot.solution ->
       check_closed LocalSlotCounterexample.is_local NestedSlot.facts T
       = false)
  ∧ check_local_reads_used LocalSlotCounterexample.is_local NestedSlot.facts
      NestedSlot.solution
    = false
  ∧ compute_least_local LocalSlotCounterexample.is_local NestedSlot.facts 50
    = Some NestedSlot.fixed_solution
  ∧ check_closed_local LocalSlotCounterexample.is_local NestedSlot.facts
      NestedSlot.fixed_solution
    = true
  ∧ check_field_used_exact LocalSlotCounterexample.is_local NestedSlot.facts
      NestedSlot.fixed_solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables NestedSlot.fixed_solution ->
       check_closed_local LocalSlotCounterexample.is_local NestedSlot.facts T
       = false)
  ∧ run 12
      (State NestedSlot.unit_expr [] [] empty_static_env empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads
  ∧ run 12
      (State
         (rebuild_unit_from_tables NestedSlot.facts NestedSlot.solution false
            NestedSlot.unit_expr)
         [] [] empty_static_env empty_heap no_reads)
    = Undefined
  ∧ run 12
      (State
         (rebuild_unit_from_tables NestedSlot.facts NestedSlot.fixed_solution
            false NestedSlot.unit_expr)
         [] [] empty_static_env empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap no_reads.
