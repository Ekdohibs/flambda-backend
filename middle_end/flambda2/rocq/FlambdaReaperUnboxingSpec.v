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

(* Which allocations the reaper may unbox, and what the checks of the two
   tables that say so mean.

   This file is to be reviewed: [dominated_by_allocation_point] against
   the last rules of [post_processing_rules] of
   middle_end/flambda2/reaper/points_to_analysis.ml, [unboxing_closed]
   against [unboxing_rules] of middle_end/flambda2/reaper/
   unboxing_analysis.ml and the seeds of [cannot_unbox0] that
   [perform_analysis0] puts in the table before running them, and the
   statements. The checks are FlambdaReaperUnboxing.v, which needs no
   review; the proofs are FlambdaReaperUnboxingProofs.v.

   An allocation is unboxed ([to_unbox]) if it has a use and
   [cannot_unbox] does not hold of it; every name whose one allocation
   with a source is that allocation ([dominated_by_allocation_point]) is
   then replaced by variables for the fields read from it. Both tables are
   read off the compiler, and nothing is believed of them:

   - [check_dominated] tests that a table of
     [dominated_by_allocation_point] is that relation, both ways
     ([check_dominated_sound_statement]): a name missing from it would
     keep a reference to an allocation that is no longer made, and a name
     wrongly in it would be replaced by the fields of an allocation it
     may not be.
   - [check_to_unbox] tests that a table of [to_unbox] is the relation
     that the rules give, both ways, for tables of the other relations of
     [unboxing_rules] that are closed under the rules
     ([check_to_unbox_sound_statement]): an allocation is in [to_unbox]
     if and only if it has a use and is not in [cannot_unbox], leaving
     out allocations with no source, which dominate no name. Both ways
     are needed: an allocation that a rule forbids to unbox is unboxed
     wrongly, and one that is left out may hold in a field an allocation
     that is unboxed on the assumption that it is unboxed too (rule
     [cu0_stored]), whose field would then be poison. The statement does
     not say that the closed tables are the least: tables that are larger
     make [cannot_unbox] larger and [to_unbox] smaller, consistently, and
     what the rebuild is to rely on is that agreement. The rules are
     monotone in the relations they derive (their negated premises are of
     the tables of points_to_analysis.ml and of the flags), so the least
     tables, which the compiler computes, are in any closed ones; the
     check computes them too, [cannot_unbox0] by rounds of its one
     recursive rule, and tests that they are closed, so that the table of
     the compiler passes.
   - [check_to_change_representation] tests the same of a table of
     [to_change_representation], the last rule, for a table of
     [to_unbox] that passes its check
     ([check_to_change_representation_sound_statement]): an allocation
     changes its representation if and only if it has a use, is not in
     [cannot_change_representation] and is not unboxed, leaving out
     allocations with no source. Both ways are needed: a closure that
     changes its representation wrongly may be read by code that expects
     its old slots, and one that is left out keeps in a slot a block
     unboxed on the assumption that the closure would hold its fields
     (rule [cu0_stored]).
   - [check_sets_whole] tests that the closures of a set that have a use
     are unboxed alike and change their representation alike
     ([check_sets_whole_sound_statement]): the rebuild unboxes a set, or
     changes its representation, whole.

   [unboxing_closed] says that five tables, those of
   [cannot_change_representation0], [cannot_change_representation1],
   [cannot_change_representation], [cannot_unbox0] and [cannot_unbox],
   are closed under the rules, each a field named after the relation it
   concludes and the rule; the rules of [to_unbox] and
   [to_change_representation] are the statements of their checks.
   Nothing is said of a table that fails a check; the examples of
   FlambdaReaperUnboxing.v have tables that the checks reject, and tables
   that pass. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaReaperGraphSpec
  FlambdaReaperUnboxing.
Import ListNotations.

(* ======================================================================== *)
(* Allocation points                                                        *)
(* ======================================================================== *)

Section Allocation_points.

Variable S : solution.

(* [multiple_allocation_points x]: [x] may come from any allocation, or
   from two different ones that have a source. *)
Definition multiple_allocation_points_of (x : node) : Prop :=
  s_any_source S x = true
  ∨ ∃ y z : node,
      s_sources S x y = true ∧ s_has_source S y = true
      ∧ s_sources S x z = true ∧ s_has_source S z = true ∧ y <> z.

(* [dominated_by_allocation_point a x]: [a] is the one allocation with a
   source that [x] may come from. *)
Definition dominated_by_allocation_point (a x : node) : Prop :=
  s_sources S x a = true ∧ s_has_source S a = true
  ∧ ¬ multiple_allocation_points_of x.

End Allocation_points.

Definition check_dominated_sound_statement : Prop :=
  ∀ (T : tables) (dominated : list (node * node)),
    check_dominated T dominated = true ->
    ∀ a x : node,
      mem_pair a x dominated = true
      <-> dominated_by_allocation_point (solution_of_tables T) a x.

(* ======================================================================== *)
(* The rules of the unboxing                                                *)
(* ======================================================================== *)

Section Rules.

Variable is_local : field -> bool.
Variable G : graph.
Variable S : solution.
(* [cannot_change_calling_convention]. *)
Variable cannot_change : node -> bool.
(* [Flambda_features.reaper_max_unbox_size] and the extra option
   [reaper-lambda-lifting]. *)
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
(* [cannot_change_representation0], [cannot_change_representation1],
   [cannot_change_representation], [cannot_unbox0], [cannot_unbox]. *)
Variables crr0 crr1 crr cu0 cu : node -> bool.

Abbreviation any_usage := (s_any_usage S).
Abbreviation any_source := (s_any_source S).
Abbreviation has_usage := (s_has_usage S).
Abbreviation has_source := (s_has_source S).
Abbreviation usages := (s_usages S).
Abbreviation sources := (s_sources S).

(* The rules of [unboxing_rules], in their order there, and the seeds of
   [cannot_unbox0] of [perform_analysis0] ([cu0_max_unbox_size]), each a
   field named after the relation it concludes and the rule; each says
   that the tables have the conclusion of the rule wherever they have the
   premises. *)
Record unboxing_closed : Prop := {
  (* An allocation that escapes with a real field that is not local. *)
  crr0_any_usage : ∀ x field y,
    any_usage x = true -> is_local field = false ->
    is_real_field field = true -> g_constructor G x field y ->
    crr0 x = true;
  (* An allocation that escapes with a local field that is read from an
     allocation of any source. *)
  crr0_local_reading : ∀ x field y z,
    any_usage x = true -> is_local field = true ->
    s_reading_field S field z = true -> g_constructor G x field y ->
    crr0 x = true;
  (* A local field read from a value of two sources. *)
  crr0_local_sources : ∀ usage field v source1 source2,
    g_accessor G v field usage -> has_usage v = true ->
    is_local field = true ->
    sources usage source1 = true -> has_source source1 = true ->
    sources usage source2 = true -> has_source source2 = true ->
    source1 <> source2 ->
    crr0 source1 = true;
  (* A use that reads a real field of a value of another source too. *)
  crr0_other_source : ∀ allocation_id alias alias_source field v,
    usages allocation_id alias = true ->
    sources alias alias_source = true -> has_source alias_source = true ->
    alias_source <> allocation_id -> is_real_field field = true ->
    g_accessor G v field alias -> has_usage v = true ->
    crr0 allocation_id = true;
  (* A use that reads a real field of a value of any source. *)
  crr0_any_source_usage : ∀ allocation_id alias field v,
    usages allocation_id alias = true -> any_source alias = true ->
    is_real_field field = true -> g_accessor G v field alias ->
    has_usage v = true ->
    crr0 allocation_id = true;
  crr0_other_own_source : ∀ allocation_id source,
    sources allocation_id source = true -> has_source source = true ->
    source <> allocation_id ->
    crr0 allocation_id = true;
  crr0_not_own_source : ∀ allocation_id usage,
    usages allocation_id usage = true -> has_usage usage = true ->
    sources allocation_id allocation_id = false ->
    crr0 allocation_id = true;
  crr0_any_source : ∀ allocation_id,
    any_source allocation_id = true -> crr0 allocation_id = true;
  crr0_call_witness : ∀ call_witness code_id,
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    crr0 call_witness = true;
  crr1_crr0 : ∀ x, crr0 x = true -> crr1 x = true;
  (* The closures of a set go together. *)
  crr1_function_slot : ∀ x field y,
    g_constructor G x field y -> is_function_slot field = true ->
    crr0 x = true -> crr1 y = true;
  crr_crr1 : ∀ x, crr1 x = true -> crr x = true;
  (* Blocks and boxed numbers never change their representation. *)
  crr_block_like : ∀ x field y,
    g_constructor G x field y -> is_block_like_field field = true ->
    crr x = true;
  cu0_cannot_change_representation : ∀ x, crr1 x = true -> cu0 x = true;
  cu0_any_usage : ∀ x, any_usage x = true -> cu0 x = true;
  (* A closure of code that cannot change its calling convention. *)
  cu0_called_cannot_change : ∀ x call_witness codeid,
    g_constructor G x (Field_call_witness Known_arity_code_pointer)
      call_witness ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 x = true;
  (* A closure that is called, unless closures are lambda-lifted. *)
  cu0_called : ∀ x call_witness,
    g_constructor G x (Field_call_witness Known_arity_code_pointer)
      call_witness ->
    has_usage call_witness = true -> lambda_lifting = false ->
    cu0 x = true;
  (* A result of code that cannot change its calling convention. *)
  cu0_result : ∀ alias allocation_id relation call_witness codeid,
    sources alias allocation_id = true -> has_source allocation_id = true ->
    g_constructor G call_witness relation alias ->
    is_return_of_call relation = true ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 allocation_id = true;
  (* A parameter of code that cannot change its calling convention. *)
  cu0_parameter : ∀ alias allocation_id relation call_witness codeid,
    sources alias allocation_id = true -> has_source allocation_id = true ->
    g_parameter G call_witness relation alias ->
    g_constructor G call_witness Field_code_id_of_call_witness codeid ->
    cannot_change codeid = true ->
    cu0 allocation_id = true;
  cu0_symbol : ∀ x source,
    sources x source = true -> is_symbol_node x = true -> cu0 x = true;
  (* An allocation stored in a used real field of one that can neither
     change its representation nor be unboxed. *)
  cu0_stored : ∀ alias relation to_ allocation_id,
    sources alias allocation_id = true ->
    g_constructor G to_ relation alias ->
    s_field_used S to_ relation = true -> crr to_ = true ->
    is_real_field relation = true -> cu0 to_ = true ->
    cu0 allocation_id = true;
  (* An allocation with more used real fields, function slots apart, than
     [max_unbox_size]. *)
  cu0_max_unbox_size : ∀ (x : node) (fields : list field),
    NoDup fields ->
    (∀ f : field,
       In f fields ->
       s_field_used S x f = true ∧ is_real_field f = true
       ∧ is_function_slot f = false) ->
    max_unbox_size < length fields ->
    cu0 x = true;
  cu_cu0 : ∀ x, cu0 x = true -> cu x = true;
  cu_function_slot : ∀ x field y,
    cu0 x = true -> g_constructor G x field y ->
    is_function_slot field = true ->
    cu y = true
}.

End Rules.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Definition check_to_unbox_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (to_unbox : list node),
    check_to_unbox is_local F T cannot_change max_unbox_size lambda_lifting
      to_unbox = true ->
    ∃ crr0 crr1 crr cu0 cu : node -> bool,
      unboxing_closed is_local (graph_of_facts F) (solution_of_tables T)
        (fun n => mem_node n cannot_change) max_unbox_size lambda_lifting
        crr0 crr1 crr cu0 cu
      ∧ (∀ x : node,
           mem_node x to_unbox = true ->
           s_has_usage (solution_of_tables T) x = true ∧ cu x = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> cu x = false ->
           mem_node x to_unbox = true).

Definition check_to_change_representation_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (to_unbox tcr : list node),
    check_to_change_representation is_local F T cannot_change max_unbox_size
      lambda_lifting to_unbox tcr = true ->
    ∃ crr0 crr1 crr cu0 cu : node -> bool,
      unboxing_closed is_local (graph_of_facts F) (solution_of_tables T)
        (fun n => mem_node n cannot_change) max_unbox_size lambda_lifting
        crr0 crr1 crr cu0 cu
      ∧ (∀ x : node,
           mem_node x to_unbox = true ->
           s_has_usage (solution_of_tables T) x = true ∧ cu x = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> cu x = false ->
           mem_node x to_unbox = true)
      ∧ (∀ x : node,
           mem_node x tcr = true ->
           s_has_usage (solution_of_tables T) x = true ∧ crr x = false
           ∧ mem_node x to_unbox = false)
      ∧ (∀ x : node,
           s_has_usage (solution_of_tables T) x = true ->
           s_has_source (solution_of_tables T) x = true -> crr x = false ->
           mem_node x to_unbox = false ->
           mem_node x tcr = true).

(* ======================================================================== *)
(* Sets of closures                                                         *)
(* ======================================================================== *)

(* Two closures of a set, one held by the other in a function slot, that
   have a use are both unboxed or neither, and both change their
   representation or neither. *)
Definition check_sets_whole_sound_statement : Prop :=
  ∀ (F : facts) (T : tables) (to_unbox tcr : list node),
    check_sets_whole F T to_unbox tcr = true ->
    ∀ (x y : node) (f : field),
      g_constructor (graph_of_facts F) x f y -> is_function_slot f = true ->
      s_has_usage (solution_of_tables T) x = true ->
      s_has_usage (solution_of_tables T) y = true ->
      mem_node x to_unbox = mem_node y to_unbox
      ∧ mem_node x tcr = mem_node y tcr.
