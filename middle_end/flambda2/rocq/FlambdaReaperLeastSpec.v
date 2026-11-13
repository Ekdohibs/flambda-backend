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

(* The least solution of the reaper's rules, and the statements of the
   rebuild for it, in place of the checks of the solution that a closed
   one may fail.

   This file is to be reviewed: [derivable] against the fields of
   FlambdaReaperGraphSpec.closed, of which it has one constructor each,
   named after the field ([by_alias_input] for [alias_input]);
   [least_solution]; the two checks of the graph ([only_read_into] and
   what is made of it); the examples; and the statements. The check and
   the computation of a least solution are FlambdaReaperLeast.v, which
   needs no review; the proofs are FlambdaReaperLeastProofs.v.

   The guards. The rules of points_to_analysis.ml have one kind of
   negation, the guard of [nontop_usages] and [nontop_sources]:
   [~~(any_usage x)] or [~~(any_source x)], a premise of the rules that
   conclude [usages] or [sources] and of four that conclude [alias]. The
   program is not stratified: [alias] gives [any_usage] and [any_source]
   ([any_usage_alias], [any_source_alias]). The compiler's schedule
   ([make_schedule]) is a fixpoint of two saturations, the rules [let$],
   which have no guard and conclude every [any_usage] and [any_source],
   then the rules [let$$], which have every guard: a guard is read where
   the relation it negates is saturated for the tuples so far, and a
   later round may make it false, the conclusion staying (the comment of
   [usages] says that a node may have both [usages] and [any_usage]).
   Here the guards are read off the solution itself, which is where the
   compiler's two strata agree: [derivable G S] is the least set of
   tuples closed under the rules of [closed] on the graph [G], each
   guard [any_usage x = false] read off [S]; and tables [T] are a least
   solution ([least_solution]) if they pass [check_closed] and the rules
   derive each of their tuples, the guards read off [T]. The tuples of a
   least solution are then exactly those derivable
   ([least_solution_exact_statement]): it is the least model of the
   rules with its own guards (a stable model of the program). The
   compiler's tables are that where no guard that a rule read turns
   false later; where one does, they have the conclusions of the rules
   that read it too, which no rule gives at the end. Such tables pass
   [check_closed] and are not least: a least solution has no [usages x _]
   with [any_usage x] nor [sources x _] with [any_source x]
   ([least_no_stale_statement]), and the tables of the solver of the
   test tool for the unit [two_calls] of
   FlambdaReaperCallingConventions.v, which follows the compiler's
   schedule, have nine [sources x x36] with [any_source x]
   ([stale_tables_statement]).

   The check and the computation. [check_least] tests that tables are a
   least solution ([check_least_sound_statement]), and
   [compute_least], if it gives tables, gives a least solution
   ([compute_least_sound_statement]); FlambdaReaperLeast.v says how. The
   least tables written out by hand or by the solver in the examples of
   the other files pass [check_least], and the closed tables that they
   are compared with fail it ([least_examples_statement]).

   What a least solution gives of the checks that the statements of the
   rebuild ask of the solution besides [check_closed]:
   - [check_field_used_exact]: every least solution passes it
     ([least_field_used_exact_statement]).
   - [check_apply_widgets] (FlambdaReaperCallingConventionsSpec.v): a
     least solution passes it where the nodes that calls read off a
     closure with the field of known arity get a source from that read
     only, or from an alias that holds where the closure has any source
     ([check_widgets_read_only], [least_apply_widgets_statement]). With
     the flag [Auto], the delayed facts of a call give aliases and any
     source to a node of their own ([add_apply],
     [only_if_closure_any_source]), and the traversal of a unit is
     expected to pass that check: the unit below does; that every unit
     does is not proved. With [Always], or with [Never] for code that is
     not of the unit, they give them to the node read off the closure,
     and two calls of one closure that name two codes make the least
     solution fail [check_apply_widgets]
     ([widgets_counterexample_statement]: a unit of the fragment of the
     statements, a call that runs and one in a handler that is never
     applied, which names code that the unit does not bind).
   - [check_unboxed_reads] (FlambdaReaperUnboxingRefinesSpec.v): a least
     solution passes it where each read of a node that an allocation to
     unbox dominates is of a field that is not local, into a node that
     gets a source from that read only ([check_reads_read_only],
     [least_unboxed_reads_statement]). Without that, it may fail: in a
     unit of [local_unboxing], a variable bound twice, by a read in a
     handler that is never applied and by a constant that the unit
     returns, has a source and a use, and the field that it reads none
     ([reads_counterexample_statement]). A well-formed unit may bind a
     variable twice: [binders_distinct] is of the names of one binder.

   The statements of the rebuild for the least solution
   ([least_rebuild_refines_observably_statement],
   [least_rebuild_cc_refines_observably_statement],
   [least_rebuild_unbox_refines_observably_statement]) are those of
   FlambdaReaperRebuildSpec.v, FlambdaReaperCallingConventionsSpec.v and
   FlambdaReaperUnboxingRefinesSpec.v, with the solution a least one in
   place of [check_closed], [check_field_used_exact], [check_apply_widgets]
   and [check_unboxed_reads]; the tables of
   [cannot_change_calling_convention], [dominated_by_allocation_point]
   and [to_unbox] those that the rules give of it
   ([derive_cannot_change], the least table that passes
   [check_cc_closed]; [derive_dominated]; [derive_to_unbox]); and, where
   a least solution may fail the check, the check of the graph that
   gives it. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperFlowSpec FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperLeast.
Import ListNotations.

(* ======================================================================== *)
(* The tuples of a solution                                                 *)
(* ======================================================================== *)

(* A tuple of one of the tables of a solution
   (FlambdaReaperGraphSpec.solution). *)
Inductive atom : Type :=
  | Atom_alias (to_ from : node)
  | Atom_any_usage (x : node)
  | Atom_any_source (x : node)
  | Atom_has_usage (x : node)
  | Atom_has_source (x : node)
  | Atom_usages (x usage : node)
  | Atom_sources (x source : node)
  | Atom_reading_field (relation : field) (x : node)
  | Atom_escaping_field (relation : field) (x : node)
  | Atom_field_used (base : node) (relation : field).

(* Whether the solution has the tuple. *)
Definition holds (S : solution) (a : atom) : bool :=
  match a with
  | Atom_alias to_ from => s_alias S to_ from
  | Atom_any_usage x => s_any_usage S x
  | Atom_any_source x => s_any_source S x
  | Atom_has_usage x => s_has_usage S x
  | Atom_has_source x => s_has_source S x
  | Atom_usages x usage => s_usages S x usage
  | Atom_sources x source => s_sources S x source
  | Atom_reading_field relation x => s_reading_field S relation x
  | Atom_escaping_field relation x => s_escaping_field S relation x
  | Atom_field_used base relation => s_field_used S base relation
  end.

(* ======================================================================== *)
(* What the rules derive                                                    *)
(* ======================================================================== *)

Section Derivation.

Variable is_local : field -> bool.
Variable G : graph.
(* The solution that the guards are read off. *)
Variable S : solution.

(* The tuples that the rules of FlambdaReaperGraphSpec.closed derive
   from the graph, a constructor for each field, in their order there,
   with the premises on the solution derived in turn and the guards read
   off [S]. *)
Inductive derivable : atom -> Prop :=
  (* alias_rules *)
  | by_alias_input : ∀ to_ from,
      g_alias G to_ from -> derivable (Atom_alias to_ from)
  | by_alias_propagate : ∀ if_used to_ from,
      derivable (Atom_any_usage if_used) -> g_propagate G if_used to_ from ->
      derivable (Atom_alias to_ from)
  | by_alias_if_any_source : ∀ if_any_source to_ from,
      derivable (Atom_any_source if_any_source) ->
      g_alias_if_any_source G if_any_source to_ from ->
      derivable (Atom_alias to_ from)
  | by_alias_constructor_usage : ∀ base base_use relation to_ from,
      g_constructor G base relation from ->
      derivable (Atom_usages base base_use) -> s_any_usage S base = false ->
      g_accessor G to_ relation base_use ->
      derivable (Atom_alias to_ from)
  | by_alias_local_constructor_usage : ∀ base base_use relation to_ from,
      is_local relation = true ->
      g_constructor G base relation from ->
      derivable (Atom_usages base base_use) ->
      g_accessor G to_ relation base_use ->
      derivable (Atom_alias to_ from)
  | by_alias_parameter_usage : ∀ base base_use relation to_ from,
      g_parameter G base relation to_ ->
      derivable (Atom_usages base base_use) -> s_any_usage S base = false ->
      g_argument G from relation base_use ->
      derivable (Atom_alias to_ from)
  | by_alias_accessor_source : ∀ base base_source relation to_ from,
      g_accessor G to_ relation base ->
      derivable (Atom_sources base base_source) ->
      s_any_source S base = false ->
      g_constructor G base_source relation from ->
      derivable (Atom_alias to_ from)
  | by_alias_local_accessor_source : ∀ base base_source relation to_ from,
      is_local relation = true ->
      g_accessor G to_ relation base ->
      derivable (Atom_sources base base_source) ->
      g_constructor G base_source relation from ->
      derivable (Atom_alias to_ from)
  | by_alias_escaping_reading : ∀ relation to_ from,
      derivable (Atom_escaping_field relation from) ->
      derivable (Atom_reading_field relation to_) ->
      derivable (Atom_alias to_ from)
  | by_alias_argument_source : ∀ base base_source relation to_ from,
      g_argument G from relation base ->
      derivable (Atom_sources base base_source) ->
      s_any_source S base = false ->
      g_parameter G base_source relation to_ ->
      derivable (Atom_alias to_ from)
  (* has_usage_rules *)
  | by_has_usage_any : ∀ x,
      derivable (Atom_any_usage x) -> derivable (Atom_has_usage x)
  | by_has_usage_alias : ∀ to_ from,
      derivable (Atom_has_usage to_) -> derivable (Atom_alias to_ from) ->
      derivable (Atom_has_usage from)
  | by_has_usage_accessor : ∀ to_ relation base,
      derivable (Atom_has_usage to_) -> g_accessor G to_ relation base ->
      derivable (Atom_has_usage base)
  | by_has_usage_argument : ∀ from relation base,
      derivable (Atom_has_source from) -> g_argument G from relation base ->
      derivable (Atom_has_usage base)
  (* has_source_rules *)
  | by_has_source_any : ∀ x,
      derivable (Atom_any_source x) -> derivable (Atom_has_source x)
  | by_has_source_alias : ∀ from to_,
      derivable (Atom_has_source from) -> derivable (Atom_alias to_ from) ->
      derivable (Atom_has_source to_)
  | by_has_source_constructor : ∀ from relation base,
      derivable (Atom_has_source from) -> g_constructor G base relation from ->
      derivable (Atom_has_source base)
  | by_has_source_function_slot : ∀ from relation base,
      g_constructor G base relation from -> is_function_slot relation = true ->
      derivable (Atom_has_source base)
  | by_has_source_parameter : ∀ to_ relation base,
      derivable (Atom_has_usage to_) -> g_parameter G base relation to_ ->
      derivable (Atom_has_source base)
  (* any_usage_rules *)
  | by_any_usage_use : ∀ to_ from,
      derivable (Atom_has_usage to_) -> g_use G to_ from ->
      derivable (Atom_any_usage from)
  | by_any_usage_alias : ∀ to_ from,
      derivable (Atom_any_usage to_) -> derivable (Atom_alias to_ from) ->
      derivable (Atom_any_usage from)
  | by_any_usage_constructor : ∀ base relation from,
      derivable (Atom_any_usage base) -> g_constructor G base relation from ->
      is_local relation = false -> derivable (Atom_any_usage from)
  | by_any_usage_argument : ∀ base relation from,
      derivable (Atom_any_source base) -> g_argument G from relation base ->
      derivable (Atom_any_usage from)
  (* any_source_rules *)
  | by_any_source_input : ∀ x,
      g_any_source G x -> derivable (Atom_any_source x)
  | by_any_source_zero_alloc : ∀ x,
      g_zero_alloc_source G x -> derivable (Atom_any_source x)
  | by_any_source_if_used : ∀ if_used x,
      derivable (Atom_any_usage if_used) -> g_any_source_if_used G if_used x ->
      derivable (Atom_any_source x)
  | by_any_source_alias : ∀ from to_,
      derivable (Atom_alias to_ from) -> derivable (Atom_any_source from) ->
      derivable (Atom_any_source to_)
  | by_any_source_parameter : ∀ base relation to_,
      derivable (Atom_any_usage base) -> g_parameter G base relation to_ ->
      derivable (Atom_any_source to_)
  | by_any_source_accessor : ∀ base relation to_,
      derivable (Atom_any_source base) -> g_accessor G to_ relation base ->
      is_local relation = false -> derivable (Atom_any_source to_)
  | by_any_source_use : ∀ from to_,
      derivable (Atom_has_source from) -> g_use G to_ from ->
      derivable (Atom_any_source to_)
  (* The input [any_usage]. *)
  | by_any_usage_input : ∀ x,
      g_any_usage G x -> derivable (Atom_any_usage x)
  (* usages_rules, with the guard of [nontop_usages] *)
  | by_usages_accessor : ∀ to_ relation base,
      g_accessor G to_ relation base -> s_any_usage S base = false ->
      derivable (Atom_usages base base)
  | by_usages_argument : ∀ from relation base,
      g_argument G from relation base -> s_any_usage S base = false ->
      derivable (Atom_usages base base)
  | by_usages_alias : ∀ to_ from usage,
      derivable (Atom_usages to_ usage) -> s_any_usage S to_ = false ->
      derivable (Atom_alias to_ from) -> s_any_usage S from = false ->
      derivable (Atom_usages from usage)
  (* sources_rules, with the guard of [nontop_sources] *)
  | by_sources_constructor : ∀ from relation base,
      g_constructor G base relation from -> s_any_source S base = false ->
      derivable (Atom_sources base base)
  | by_sources_parameter : ∀ to_ relation base,
      g_parameter G base relation to_ -> s_any_source S base = false ->
      derivable (Atom_sources base base)
  | by_sources_alias : ∀ from to_ source,
      derivable (Atom_sources from source) -> s_any_source S from = false ->
      derivable (Atom_alias to_ from) -> s_any_source S to_ = false ->
      derivable (Atom_sources to_ source)
  (* local_rules *)
  | by_escaping_field_rule : ∀ base relation from,
      derivable (Atom_any_usage base) -> g_constructor G base relation from ->
      is_local relation = true ->
      derivable (Atom_escaping_field relation from)
  | by_reading_field_rule : ∀ base relation to_,
      derivable (Atom_any_source base) -> g_accessor G to_ relation base ->
      is_local relation = true ->
      derivable (Atom_reading_field relation to_)
  (* post_processing_rules: [field_of_constructor_is_used] *)
  | by_field_used_any_usage : ∀ base relation from,
      g_constructor G base relation from ->
      derivable (Atom_any_usage base) -> is_local relation = false ->
      derivable (Atom_field_used base relation)
  | by_field_used_usage : ∀ base relation from usage v,
      g_constructor G base relation from ->
      derivable (Atom_usages base usage) -> g_accessor G v relation usage ->
      derivable (Atom_has_usage v) ->
      derivable (Atom_field_used base relation)
  | by_field_used_reading : ∀ base relation from x,
      g_constructor G base relation from ->
      derivable (Atom_any_usage base) ->
      derivable (Atom_reading_field relation x) ->
      derivable (Atom_has_usage x) ->
      derivable (Atom_field_used base relation)
  | by_field_used_local_source : ∀ usage base relation from v,
      g_constructor G base relation from ->
      derivable (Atom_sources usage base) -> is_local relation = true ->
      derivable (Atom_any_usage base) -> g_accessor G v relation usage ->
      derivable (Atom_has_usage v) ->
      derivable (Atom_field_used base relation).

End Derivation.

(* Tables [T] are a least solution of the rules on the facts [F]: they
   are closed under the rules, and the rules derive each of their tuples,
   the guards read off [T]. *)
Definition least_solution (is_local : field -> bool) (F : facts)
    (T : tables) : Prop :=
  check_closed is_local F T = true
  ∧ ∀ a : atom,
      holds (solution_of_tables T) a = true ->
      derivable is_local (graph_of_facts F) (solution_of_tables T) a.

Definition least_solution_exact_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables) (a : atom),
    least_solution is_local F T ->
    holds (solution_of_tables T) a = true
    <-> derivable is_local (graph_of_facts F) (solution_of_tables T) a.

Definition least_no_stale_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables) (x y : node),
    least_solution is_local F T ->
    (mem_pair x y (t_usages T) = true -> mem_node x (t_any_usage T) = false)
    ∧ (mem_pair x y (t_sources T) = true ->
       mem_node x (t_any_source T) = false).

(* The tables of the solver of the test tool for the unit [two_calls]
   (FlambdaReaperCallingConventions.Examples.two_calls_tables) are closed
   and not least: they have [sources x x36] for nine nodes [x] with
   [any_source], which the schedule of the compiler derives before it
   derives the [any_source]. *)
Definition stale_tables_statement : Prop :=
  check_closed Examples.no_local Examples.two_calls_facts
    Examples.two_calls_tables
  = true
  ∧ mem_pair (Node_made 8%positive) (Node_variable 36%positive)
      (t_sources Examples.two_calls_tables)
    = true
  ∧ mem_node (Node_made 8%positive) (t_any_source Examples.two_calls_tables)
    = true
  ∧ ~ least_solution Examples.no_local Examples.two_calls_facts
        Examples.two_calls_tables.

(* ======================================================================== *)
(* The check and the computation                                            *)
(* ======================================================================== *)

Definition check_least_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (fuel : nat) (T : tables),
    check_least is_local F fuel T = true -> least_solution is_local F T.

Definition compute_least_sound_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (fuel : nat) (T : tables),
    compute_least is_local F fuel = Some T -> least_solution is_local F T.

(* The tables of the examples that are said to be least are, and those
   that are closed and not least are not: the least solution of the
   example of FlambdaReaperSolutionSpec.v, and the least tables of
   [spurious_widget_counterexample_statement] and of
   [unboxed_reads_example_statement], with the closed tables they are
   compared with there. *)
Definition least_examples_statement : Prop :=
  check_least Example.is_local Example.graph 100 Example.solution = true
  ∧ check_least Examples.no_local SpuriousWidgetExample.unit_facts 100
      SpuriousWidgetExample.least
    = true
  ∧ check_least Examples.no_local SpuriousWidgetExample.unit_facts 100
      SpuriousWidgetExample.spurious
    = false
  ∧ check_least Examples.no_local DeadReadExample.facts 100
      DeadReadExample.least
    = true
  ∧ check_least Examples.no_local DeadReadExample.facts 100
      DeadReadExample.tables
    = false.

(* ======================================================================== *)
(* [check_field_used_exact]                                                 *)
(* ======================================================================== *)

Definition least_field_used_exact_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    least_solution is_local F T -> check_field_used_exact is_local F T = true.

(* ======================================================================== *)
(* [check_apply_widgets]                                                    *)
(* ======================================================================== *)

(* No fact of [F] puts [x] where a rule may conclude [alias x _],
   [any_source x] or [sources x _], but the read of the field [f] off
   [base] into [x] and, if [if_any_source], aliases to [x] that hold if
   [base] has any source: [x] is not the [to_] of an alias, of a
   propagation, of another read or of a parameter, nor the base of a
   constructor or of a parameter, nor has any source as an input, by
   [any_source_if_used] or by a use. *)
Definition only_read_into (F : facts) (x : node) (f : field) (base : node)
    (if_any_source : bool) : bool :=
  forallb (fun '(to_, _) => negb (node_eqb to_ x)) (f_alias F)
  && forallb (fun '(_, to_, _) => negb (node_eqb to_ x)) (f_propagate F)
  && forallb
       (fun '(c, to_, _) =>
          negb (node_eqb to_ x) || (if_any_source && node_eqb c base))
       (f_alias_if_any_source F)
  && forallb
       (fun '(to_, f', base') =>
          negb (node_eqb to_ x) || (field_eqb f' f && node_eqb base' base))
       (f_accessor F)
  && forallb
       (fun '(base', _, to_) => negb (node_eqb to_ x || node_eqb base' x))
       (f_parameter F)
  && forallb (fun '(base', _, _) => negb (node_eqb base' x))
       (f_constructor F)
  && negb (mem_node x (f_any_source F))
  && negb (mem_node x (f_zero_alloc_source F))
  && forallb (fun '(_, y) => negb (node_eqb y x)) (f_any_source_if_used F)
  && forallb (fun '(to_, _) => negb (node_eqb to_ x)) (f_use F).

(* Each node that a call reads off a closure with the field of known
   arity, its [apply_widget] (traverse.ml), has no source but from that
   read and from aliases that hold where the closure has any source. *)
Definition check_widgets_read_only (F : facts) : bool :=
  forallb
    (fun '(a, f, n) =>
       negb (field_eqb f (Field_call_witness Known_arity_code_pointer))
       || only_read_into F a f n true)
    (f_accessor F).

Definition least_apply_widgets_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables),
    is_local_slots_only is_local ->
    least_solution is_local F T ->
    check_widgets_read_only F = true ->
    check_apply_widgets F T = true.

(* The unit of [spurious_widget_counterexample_statement] with a second
   call of [s80], in a handler that is never applied, which names the
   code 99, which the unit does not bind:

     let code f50 x40 x42 = x40
     let symbol s80 = closure of f50
     let k20 x30 = return x30 in
     let k21 () = s80 9 7 (direct f99) -> k20 in
     s80 9 7 (direct f50) -> k20

   With the flag [Always] (or [Never], as 99 is not of the unit), the
   delayed facts of the calls give the node that the call of [f50]
   reads off [s80] ([m 17]) an alias of the witness of known arity of
   [f50] ([m 8]), and the node that the call of [f99] reads ([m 14]) any
   source. The least solution keeps them apart: [m 14] has any source,
   and [m 17] has [m 8] for its one source. With [Auto], the graph of
   the same unit passes [check_widgets_read_only], so that each least
   solution of it passes [check_apply_widgets]. *)
Module WidgetExample.

Local Open Scope positive_scope.
Import FlambdaReaperCallingConventions.Examples.

Definition s80 : node := Node_symbol 80.

Definition call_s80 (id : code_id) (k : continuation) : apply_expr :=
  {| ap_callee := Symbol 80; ap_call_kind := Direct id;
     ap_args := [Const (Tagged_immediate 9); Const (Tagged_immediate 7)];
     ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value; Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50 SpuriousWidgetExample.code_50;
     Bind_symbol 80 (Static_const_closure_of_set 70 [(70, 80)] [(70, 50)] [])]
    (Let_cont_non_recursive 20
       (Handler [30] (make_handler_metadata [Value]) (return_to 1 [Var 30]))
       (Let_cont_non_recursive 21
          (Handler [] (make_handler_metadata []) (Apply (call_s80 99 20)))
          (Apply (call_s80 50 20)))).

Definition unit_facts : facts := traverse_unit Always interface unit_expr.

Definition least : tables :=
  {| t_alias :=
       [(m 7, consts); (m 6, consts); (m 16, c 50); (m 2, m 4); (v 30, m 3);
        (m 14, m 8); (v 42, m 7); (v 40, m 6); (v 41, m 5); (m 3, v 40);
        (m 1, v 30); (m 12, m 14); (m 15, m 17); (m 5, s80); (m 17, m 8)];
     t_any_usage :=
       [m 6; m 10; v 40; consts; m 4; m 3; c 50; m 11; m 9; m 8; v 30; m 2;
        m 13; m 16; m 1; s80];
     t_any_source :=
       [m 3; m 16; m 1; v 40; v 42; m 13; m 2; v 30; m 7; m 6; m 12; consts;
        c 50; m 14];
     t_has_usage :=
       [m 6; v 40; m 10; m 17; m 14; m 11; c 50; m 3; m 4; consts; m 12;
        m 15; v 30; m 8; m 9; s80; m 1; m 16; m 13; m 2];
     t_has_source :=
       [m 3; m 15; m 16; v 42; v 40; m 1; m 11; m 17; m 6; m 7; v 30; m 2;
        m 13; m 8; m 9; m 10; v 41; m 12; m 5; m 14; c 50; consts; s80];
     t_usages := [(m 17, m 15); (m 14, m 12); (m 12, m 12); (m 15, m 15)];
     t_sources :=
       [(m 15, m 8); (v 41, s80); (m 5, s80); (m 17, m 8); (m 8, m 8);
        (m 11, m 11); (m 9, m 9); (m 10, m 10); (s80, s80)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(m 10, Field_code_id_of_call_witness); (m 10, result 0);
        (m 10, Field_return_of_call Exn); (m 11, unknown);
        (m 8, Field_code_id_of_call_witness);
        (m 8, Field_return_of_call Exn); (m 8, result 0);
        (m 9, Field_code_id_of_call_witness); (m 9, result 0);
        (s80, Field_function_slot 70); (s80, unknown); (s80, known)] |}.

End WidgetExample.

Definition widgets_counterexample_statement : Prop :=
  in_flow_fragment WidgetExample.unit_expr = true
  ∧ static_constants (unit_codes WidgetExample.unit_expr)
      WidgetExample.unit_expr
    = true
  ∧ results_kinded [] [] WidgetExample.unit_expr = true
  ∧ well_formed_unit Examples.interface WidgetExample.unit_expr = true
  ∧ exits_one_value Examples.interface WidgetExample.unit_expr = true
  ∧ least_solution Examples.no_local WidgetExample.unit_facts
      WidgetExample.least
  ∧ check_widgets_read_only WidgetExample.unit_facts = false
  ∧ check_apply_widgets WidgetExample.unit_facts WidgetExample.least = false
  ∧ check_widgets_read_only
      (traverse_unit Auto Examples.interface WidgetExample.unit_expr)
    = true.

(* ======================================================================== *)
(* [check_unboxed_reads]                                                    *)
(* ======================================================================== *)

(* Each read of a node that an allocation of [to_unbox] dominates is of
   a field that is not local, into a node that has no source but from
   that read. *)
Definition check_reads_read_only (is_local : field -> bool) (F : facts)
    (dominated : list (node * node)) (to_unbox : list node) : bool :=
  forallb
    (fun '(x, f, y) =>
       negb (has_to_be_unboxed dominated to_unbox y)
       || (negb (is_local f) && only_read_into F x f y false))
    (f_accessor F).

(* The allocations to unbox, as the rules give them from the solution
   [T] and the least table of [cannot_change_calling_convention]. *)
Definition least_to_unbox (is_local : field -> bool) (F : facts)
    (T : tables) (max_unbox_size : nat) (lambda_lifting : bool)
    : list node :=
  derive_to_unbox is_local F T (derive_cannot_change F T) max_unbox_size
    lambda_lifting.

Definition least_unboxed_reads_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool),
    least_solution is_local F T ->
    check_reads_read_only is_local F (derive_dominated T)
      (derive_to_unbox is_local F T cannot_change max_unbox_size
         lambda_lifting)
    = true ->
    check_unboxed_reads F T (derive_dominated T)
      (derive_to_unbox is_local F T cannot_change max_unbox_size
         lambda_lifting)
    = true.

(* The unit of [unboxed_reads_example_statement], which returns a
   variable [x32] bound to 7 in place of 7:

     let rec k20 x30 = (let x31 = [x30] in let x32 = x31.(0) in
                        return x32) in
     let x32 = 7 in return x32

   [x32] is bound twice, once in the handler, which is never applied, to
   the read of the block [x31] that the rules unbox. Its other binding
   gives it a source, and its return a use; the field 0 of [x31] holds
   [x30], which has none. The unit is of the fragment of the statements
   and of [local_unboxing]. *)
Module ShadowedReadExample.

Local Open Scope positive_scope.
Import FlambdaReaperUnboxing.Examples.

Definition unit_expr : expr :=
  Let_cont_recursive
    [(20%positive,
      Handler [30] (make_handler_metadata [Value])
        (Let 31 Value (make_block [Var 30])
           (Let 32 Value (load 0 (Var 31))
              (return_to 1 [Var 32]))))]
    (Let 32 Value (Simple (int 7)) (return_to 1 [Var 32])).

Definition unit_facts : facts := traverse_unit Auto interface unit_expr.

Definition least : tables :=
  {| t_alias := [(v 32, v 30); (v 32, consts); (m 1, v 32)];
     t_any_usage := [consts; v 30; v 32; m 2; m 1];
     t_any_source := [m 1; v 32; consts];
     t_has_usage := [v 31; v 30; consts; v 32; m 1; m 2];
     t_has_source := [m 1; v 31; v 32; consts];
     t_usages := [(v 31, v 31)];
     t_sources := [(v 31, v 31)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used := [(v 31, Field_block 0 Value)] |}.

Definition to_unbox : list node :=
  least_to_unbox no_local unit_facts least max_unbox_size false.

End ShadowedReadExample.

Definition reads_counterexample_statement : Prop :=
  let e := ShadowedReadExample.unit_expr in
  let F := ShadowedReadExample.unit_facts in
  let T := ShadowedReadExample.least in
  in_flow_fragment e = true
  ∧ static_constants (unit_codes e) e = true
  ∧ results_kinded [] [] e = true
  ∧ well_formed_unit Examples.interface e = true
  ∧ exits_one_value Examples.interface e = true
  ∧ least_solution Examples.no_local F T
  ∧ ShadowedReadExample.to_unbox = [Node_variable 31%positive]
  ∧ local_unboxing F T (derive_dominated T) ShadowedReadExample.to_unbox e
    = true
  ∧ check_reads_read_only Examples.no_local F (derive_dominated T)
      ShadowedReadExample.to_unbox
    = false
  ∧ check_unboxed_reads F T (derive_dominated T) ShadowedReadExample.to_unbox
    = false.

(* ======================================================================== *)
(* The statements of the rebuild for the least solution                     *)
(* ======================================================================== *)

Definition least_rebuild_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    program_refines_observably e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition least_rebuild_cc_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    check_widgets_read_only (traverse_unit flag u e) = true ->
    program_refines_observably e
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) preserve_direct_calls e).

Definition least_rebuild_unbox_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (max_unbox_size : nat)
    (lambda_lifting preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    local_unboxing (traverse_unit flag u e) T (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
      e
    = true ->
    check_reads_read_only is_local (traverse_unit flag u e)
      (derive_dominated T)
      (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
         lambda_lifting)
    = true ->
    program_refines_observably e
      (rebuild_unit_unbox (traverse_unit flag u e) T
         (derive_cannot_change (traverse_unit flag u e) T)
         (unit_code_deps flag u e) (derive_dominated T)
         (least_to_unbox is_local (traverse_unit flag u e) T max_unbox_size
            lambda_lifting)
         preserve_direct_calls e).
