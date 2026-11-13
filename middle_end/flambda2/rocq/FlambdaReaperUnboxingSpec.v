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
   [perform_analysis0] puts in the table before running them, the unit of
   the counterexample below against its .fl, [check_called_my_closure]
   against the fix it is said to be, and the statements. The checks are
   FlambdaReaperUnboxing.v, which needs no review; the proofs are
   FlambdaReaperUnboxingProofs.v.

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
   that pass.

   A counterexample, found by fuzzing: a crash of the compiler's reaper
   with lambda lifting, which the model shares
   (middle_end/flambda2/tests/semantics/reaper/
   lambda_lifting_dead_callee.fl, whose comment gives the OCaml). With
   lambda lifting, a closure that is only called may be unboxed, and so
   is the [my_closure] of its code if the closure dominates it: a call
   then gives the fields of the closure for those of [my_closure], and
   has no callee ([Unbox_my_closure]). Two closures of one code, as two
   inlined copies of a function make them, are each unboxed, but
   [my_closure], of which they are the two sources, is dominated by
   neither and is not unboxed. The decision on [my_closure] is then
   [Keep_my_closure], and the rebuild keeps the callee of each call: a
   closure whose binding the unboxing removed.
   [lambda_lifting_dead_callee_statement] says that the tables of such a
   unit pass every check, with lambda lifting and without, that the unit
   exits, and that rebuilt with lambda lifting it goes wrong, and without
   it exits as the unit does; [lambda_lifting_dead_callee_cause_statement]
   says why.

   Of the two fixes of the fuzzing report, the model has the first: the
   analysis must not unbox a called closure unless the [my_closure] of
   its code is unboxed too. (The second makes [rebuild_apply] drop the
   callee under [Keep_my_closure] when the callee is unboxed and
   [my_closure] has no use.) [check_called_my_closure] is what the fix
   asks of a table of [to_unbox]. As a rule of [unboxing_rules], the fix
   is a seed of [cannot_unbox0] like [cu0_called], with the premise that
   [my_closure] has [multiple_allocation_points] in place of
   [lambda_lifting = false]: a closure of code that has a use is a source
   of the [my_closure] of the code, which is unboxed with the closure if
   the closure dominates it. [lambda_lifting_dead_callee_statement] says
   too that the check rejects the tables of the unit with lambda lifting,
   and passes those without and those of the example [unboxed_closure] of
   FlambdaReaperUnboxing.v, whose closure dominates the [my_closure] of
   its code. With lambda lifting, the rule seeds [cannot_unbox0] with the
   two closures of the unit and nothing else, as [cu0_called] does
   without: the analysis with the fix gives the tables without lambda
   lifting, whose rebuild exits as the unit does. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaBindersSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperGraphSpec
  FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec.
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

(* ======================================================================== *)
(* A counterexample: lambda lifting and a dead callee                       *)
(* ======================================================================== *)

(* The first fix, as what it asks of a table of [to_unbox]: a closure to
   unbox that is called, its witness of known arity, that of the code dep
   of its code, having a use (as in [cu0_called]), has the [my_closure]
   of that code dep unboxed too, dominated by an allocation to unbox. *)
Definition check_called_my_closure (F : facts) (T : tables)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) : bool :=
  forallb
    (fun '(_, cd) =>
       forallb
         (fun '(x, field, call_witness) =>
            negb (field_eqb field
                    (Field_call_witness Known_arity_code_pointer))
            || negb (node_eqb call_witness (cd_known_arity_call_witness cd))
            || negb (mem_node call_witness (t_has_usage T))
            || negb (mem_node x to_unbox)
            || existsb
                 (fun '(a, y) =>
                    node_eqb y (cd_my_closure cd) && mem_node a to_unbox)
                 dominated)
         (f_constructor F))
    code_deps.

Import Examples.
Local Open Scope positive_scope.

(* The code 54 returns its parameter, and does not use its closure. The
   unit makes two closures of it, [x60] and [x61], as two inlined copies
   of a function would; it calls the code through [x60] with 9, then
   through [x61] with 9, and returns the result. *)
Definition dead_callee_code : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 40]).

Definition dead_callee_unit : expr :=
  Let_static [Bind_code 54 dead_callee_code]
    (Let 60 Value (Closure_of_set 70 [(70, 54)] [])
       (Let 61 Value (Closure_of_set 70 [(70, 54)] [])
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Apply (call 54 (Var 61))))
             (Apply (with_continuation (call 54 (Var 60)) 20))))).

Definition dead_callee_facts : facts :=
  traverse_unit Auto interface dead_callee_unit.

Definition dead_callee_code_deps : finite_map code_dep :=
  unit_code_deps Auto interface dead_callee_unit.

(* The solution of the unit, as the solver of the test tool computes it.
   The code dep of 54 has the result 3, the exception 4, [my_closure] 5,
   the parameter 6 and the witnesses 7 (of known arity) and 8; the calls
   make the nodes 9 to 18. [my_closure] has the two sources [x60] and
   [x61]. *)
Definition dead_callee_tables : tables :=
  {| t_alias := [(m 18, m 7); (m 5, m 17); (m 13, m 7); (m 5, m 12);
       (m 14, m 16); (m 9, m 11); (m 3, v 40); (v 41, m 5); (v 40, m 6);
       (m 16, m 7); (m 11, m 7); (m 1, m 3); (v 30, m 3); (m 2, m 4);
       (m 10, c 54); (m 15, c 54); (m 6, consts); (m 5, v 60);
       (m 5, v 61)];
     t_any_usage := [m 1; m 15; m 10; m 2; m 3; m 4; c 54; v 40; m 6;
       consts];
     t_any_source := [c 54; consts; m 10; m 15; m 6; v 40; m 3; m 1; v 30];
     t_has_usage := [m 1; m 15; m 10; m 2; m 14; m 9; m 16; m 11; m 7; m 3;
       m 4; c 54; v 40; m 6; consts; v 60; v 61];
     t_has_source := [c 54; consts; v 61; v 60; m 8; m 7; m 18; m 13; m 16;
       m 11; m 10; m 15; m 6; m 14; m 9; v 40; m 5; m 3; v 41; m 1; v 30];
     t_usages := [(v 60, v 60); (m 14, m 14); (v 61, v 61); (m 9, m 9);
       (m 16, m 14); (m 11, m 9); (m 7, m 14); (m 7, m 9)];
     t_sources := [(v 60, v 60); (v 61, v 61); (m 8, m 8); (m 7, m 7);
       (m 18, m 7); (m 13, m 7); (m 16, m 7); (m 11, m 7); (m 14, m 7);
       (m 9, m 7); (m 5, v 60); (m 5, v 61); (v 41, v 61); (v 41, v 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 60, known); (v 61, known); (m 7, result 0);
       (m 7, Field_return_of_call Exn);
       (m 7, Field_code_id_of_call_witness)] |}.

Definition dead_callee_cannot_change : list node :=
  derive_cannot_change dead_callee_facts dead_callee_tables.

Definition dead_callee_dominated : list (node * node) :=
  derive_dominated dead_callee_tables.

(* The tables of [to_unbox] and [to_change_representation] that the rules
   give, with lambda lifting or without, and the unit rebuilt with them. *)
Definition dead_callee_to_unbox (lambda_lifting : bool) : list node :=
  derive_to_unbox no_local dead_callee_facts dead_callee_tables
    dead_callee_cannot_change max_unbox_size lambda_lifting.

Definition dead_callee_tcr (lambda_lifting : bool) : list node :=
  derive_to_change_representation no_local dead_callee_facts
    dead_callee_tables dead_callee_cannot_change max_unbox_size
    lambda_lifting (dead_callee_to_unbox lambda_lifting).

Definition dead_callee_rebuilt (lambda_lifting : bool) : expr :=
  rebuild_unit_unbox_repr no_local dead_callee_facts dead_callee_tables
    dead_callee_cannot_change dead_callee_code_deps dead_callee_dominated
    (dead_callee_to_unbox lambda_lifting) (dead_callee_tcr lambda_lifting)
    true dead_callee_unit.

Definition dead_callee_exit : behaviour :=
  Exits 1 [Value_const (Tagged_immediate 9)] empty_heap.

(* With lambda lifting and without, the tables pass every check. The unit
   exits with 9; rebuilt with lambda lifting, it goes wrong, and rebuilt
   without, it exits with 9. The check of the fix rejects the tables with
   lambda lifting, and passes those without, and those of
   [unboxed_closure]. *)
Definition lambda_lifting_dead_callee_statement : Prop :=
  (∀ lambda_lifting : bool,
     let to_unbox := dead_callee_to_unbox lambda_lifting in
     let tcr := dead_callee_tcr lambda_lifting in
     check_closed no_local dead_callee_facts dead_callee_tables = true
     ∧ check_field_used_exact no_local dead_callee_facts dead_callee_tables
       = true
     ∧ check_cc_closed dead_callee_facts dead_callee_tables
         dead_callee_cannot_change = true
     ∧ check_dominated dead_callee_tables dead_callee_dominated = true
     ∧ check_to_unbox no_local dead_callee_facts dead_callee_tables
         dead_callee_cannot_change max_unbox_size lambda_lifting to_unbox
       = true
     ∧ check_to_change_representation no_local dead_callee_facts
         dead_callee_tables dead_callee_cannot_change max_unbox_size
         lambda_lifting to_unbox tcr = true
     ∧ check_sets_whole dead_callee_facts dead_callee_tables to_unbox tcr
       = true)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own dead_callee_unit)
      dead_callee_exit
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (dead_callee_rebuilt true))
      Goes_wrong
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own (dead_callee_rebuilt false))
      dead_callee_exit
  ∧ check_called_my_closure dead_callee_facts dead_callee_tables
      dead_callee_code_deps dead_callee_dominated (dead_callee_to_unbox true)
    = false
  ∧ check_called_my_closure dead_callee_facts dead_callee_tables
      dead_callee_code_deps dead_callee_dominated
      (dead_callee_to_unbox false)
    = true
  ∧ check_called_my_closure unboxed_closure_facts unboxed_closure_tables
      (unit_code_deps Auto interface unboxed_closure)
      unboxed_closure_dominated [v 60]
    = true.

(* Why. With lambda lifting, the two closures are unboxed, to no field;
   [my_closure] has two allocation points, and is not: the decision on it
   is [Keep_my_closure] ([None]). The rebuilt unit calls the code through
   [x60] and [x61], which it no longer binds. Without lambda lifting,
   nothing is unboxed. *)
Definition lambda_lifting_dead_callee_cause_statement : Prop :=
  dead_callee_to_unbox true = [v 60; v 61]
  ∧ fst (unit_unboxed_fields dead_callee_facts dead_callee_tables
           dead_callee_dominated (dead_callee_to_unbox true)
           dead_callee_unit)
    = [(v 60, []); (v 61, [])]
  ∧ multiple_allocation_points dead_callee_tables (m 5) = true
  ∧ unit_my_closure_decisions_u dead_callee_facts dead_callee_tables
      dead_callee_cannot_change dead_callee_code_deps dead_callee_dominated
      (dead_callee_to_unbox true) dead_callee_unit
    = [(54, None)]
  ∧ free_variables_of_expr dead_callee_unit = []
  ∧ free_variables_of_expr (dead_callee_rebuilt true) = [61; 60]
  ∧ dead_callee_to_unbox false = [].
