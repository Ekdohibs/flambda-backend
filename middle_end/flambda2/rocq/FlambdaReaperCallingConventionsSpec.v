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

(* Which code may change its calling convention, what the check of the
   table that says so means, and the rebuild when no code changes.

   This file is to be reviewed: [cc_closed] against the rules
   [calling_convention_changes_rules] of
   middle_end/flambda2/reaper/unboxing_analysis.ml, and the statements.
   The check, the table and the rebuild are
   FlambdaReaperCallingConventions.v, which needs no review; the proofs
   are FlambdaReaperCallingConventionsProofs.v.

   The reaper changes the calling convention of code of the unit (it
   removes its unused parameters and results) unless
   [cannot_change_calling_convention] holds of its code id. That relation
   is derived by eight rules from the solution of points_to_analysis.ml
   (FlambdaReaperGraphSpec.solution), through a second relation,
   [cannot_change_witness_calling_convention], of the call witnesses
   (the allocations of [Field.code_id_of_call_witness]). A witness
   cannot change if it escapes ([cc_witness_any_usage]); if a call that
   uses it and reads the code id of what it calls may also find another
   witness there, or any ([cc_witness_other_source],
   [cc_witness_any_source_usage]); if it may come from another
   allocation, or from any ([cc_witness_other_own_source],
   [cc_witness_any_source]); or if it is used and is not its own source
   ([cc_witness_not_own_source]). The code of a used witness that cannot
   change cannot change ([cc_code_id_of_witness]), nor can code whose
   witness of unknown arity is used ([cc_code_id_unknown_arity]).
   [cc_closed] says that two tables are closed under the eight rules,
   each a field named [cc_] and the rule. What the rebuild is to rely on
   is that code not in the table is called only by calls that find its
   witness and no other.

   The table of [cannot_change_calling_convention] is read off the
   compiler, and nothing is believed of it: [check_cc_closed] tests it,
   and a table that passes is, with a table of the witnesses, closed
   under the rules ([check_cc_closed_sound_statement]). The table of the
   witnesses is not an input: no rule concludes it from itself, and the
   check computes it from the solution. [derive_cannot_change] is the
   least table: it passes the check, and is in every table that is
   closed under the rules ([derive_cannot_change_least_statement]).
   Nothing is said of a table that fails. The examples of
   FlambdaReaperCallingConventions.v have a table that the check
   rejects.

   [rebuild_unit_cc], the rebuild with changes of calling convention, is
   [FlambdaReaperRebuild.rebuild_unit] when every code of the unit is in
   the table ([rebuild_unit_cc_unchanged_statement]): what is proved of
   the one is to be carried to the other, for the code that does not
   change. *)

From Stdlib Require Import List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperRebuild FlambdaReaperCallingConventions.
Import ListNotations.

(* ======================================================================== *)
(* The rules                                                                *)
(* ======================================================================== *)

Section Rules.

Variable G : graph.
Variable S : solution.
(* [cannot_change_witness_calling_convention]. *)
Variable cannot_change_witness : node -> bool.
(* [cannot_change_calling_convention]. *)
Variable cannot_change : node -> bool.

Abbreviation any_usage := (s_any_usage S).
Abbreviation any_source := (s_any_source S).
Abbreviation has_usage := (s_has_usage S).
Abbreviation has_source := (s_has_source S).
Abbreviation usages := (s_usages S).
Abbreviation sources := (s_sources S).

(* The rules of [calling_convention_changes_rules], in their order there,
   each a field named [cc_] and the rule; each says that the tables have
   the conclusion of the rule wherever they have the premises. *)
Record cc_closed : Prop := {
  cc_witness_any_usage : ∀ x,
    any_usage x = true -> cannot_change_witness x = true;
  cc_witness_other_source : ∀ allocation_id alias alias_source v,
    usages allocation_id alias = true ->
    sources alias alias_source = true -> has_source alias_source = true ->
    alias_source <> allocation_id ->
    g_accessor G v Field_code_id_of_call_witness alias ->
    has_usage v = true ->
    cannot_change_witness allocation_id = true;
  cc_witness_any_source_usage : ∀ allocation_id alias v,
    usages allocation_id alias = true -> any_source alias = true ->
    g_accessor G v Field_code_id_of_call_witness alias ->
    has_usage v = true ->
    cannot_change_witness allocation_id = true;
  cc_witness_other_own_source : ∀ allocation_id source,
    sources allocation_id source = true -> has_source source = true ->
    source <> allocation_id ->
    cannot_change_witness allocation_id = true;
  cc_witness_not_own_source : ∀ allocation_id usage,
    usages allocation_id usage = true -> has_usage usage = true ->
    sources allocation_id allocation_id = false ->
    cannot_change_witness allocation_id = true;
  cc_witness_any_source : ∀ allocation_id,
    any_source allocation_id = true ->
    cannot_change_witness allocation_id = true;
  cc_code_id_of_witness : ∀ call_witness code_id,
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    has_usage call_witness = true ->
    cannot_change_witness call_witness = true ->
    cannot_change code_id = true;
  cc_code_id_unknown_arity : ∀ set_of_closures call_witness code_id,
    g_constructor G set_of_closures
      (Field_call_witness Unknown_arity_code_pointer) call_witness ->
    has_usage call_witness = true ->
    g_constructor G call_witness Field_code_id_of_call_witness code_id ->
    cannot_change code_id = true
}.

End Rules.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Definition check_cc_closed_sound_statement : Prop :=
  ∀ (F : facts) (T : tables) (cannot_change : list node),
    check_cc_closed F T cannot_change = true ->
    ∃ cannot_change_witness : node -> bool,
      cc_closed (graph_of_facts F) (solution_of_tables T)
        cannot_change_witness (fun n => mem_node n cannot_change).

Definition derive_cannot_change_least_statement : Prop :=
  ∀ (F : facts) (T : tables),
    check_cc_closed F T (derive_cannot_change F T) = true
    ∧ ∀ cannot_change_witness cannot_change : node -> bool,
        cc_closed (graph_of_facts F) (solution_of_tables T)
          cannot_change_witness cannot_change ->
        ∀ n : node,
          mem_node n (derive_cannot_change F T) = true ->
          cannot_change n = true.

(* ======================================================================== *)
(* The rebuild                                                              *)
(* ======================================================================== *)

Definition rebuild_unit_cc_unchanged_statement : Prop :=
  ∀ (F : facts) (sol : solution) (cannot_change : node -> bool)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (e : expr),
    (∀ id : code_id,
       among id (static_names code_ids_bound e) = true ->
       cannot_change (Node_code_id id) = true) ->
    rebuild_unit_cc F sol cannot_change code_deps preserve_direct_calls e
    = rebuild_unit F sol preserve_direct_calls e.
