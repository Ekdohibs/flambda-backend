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

(* The types of the reaper applied to the runs of the unit rebuilt with
   unboxing across calls, and with changes of representation of sets of
   closures, end to end: the exported types of the symbols where the
   rebuilt unit exits, and the result types of code where the run of the
   unit and its rebuild bind the variables of the code. The
   statements of FlambdaReaperTypesSpec.v take the relation of the
   values of the runs at a return of code as a premise; here it is that
   of the simulations of the rebuilds, which the proofs read off them.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesSpec.v (the types), FlambdaReaperTypesTermSpec.v (the
   end-to-end statements of the plain rebuild, whose premises these
   take), FlambdaReaperCallingConventionsSpec.v and
   FlambdaReaperUnboxingRefinesCallsSpec.v and
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v (the rebuilds and
   their fragments). The proofs are in FlambdaReaperTypesCallsProofs.v,
   which needs no review, with FlambdaReaperTypesCallsExitProofs.v and
   FlambdaReaperTypesReprExitProofs.v (the states that the simulations
   relate).

   The runs. The run of the unit, that of the unit rebuilt with changes
   of calling convention ([rebuild_unit_cc_from_tables], or the plain
   rebuild, where no code changes), and that of the unit rebuilt with
   unboxing in the mode of the fix
   (FlambdaReaperUnboxingRefinesCallsSpec.rebuild_unit_unbox_fixed). The
   first two are related by the simulation of
   FlambdaReaperRebuildProofs.v, the last two by that of
   FlambdaReaperUnboxingRefinesCallsSimProofs.v ([uc]). Each state that
   the run of the unit comes to, and from which it steps, is related to a
   state that the second run comes to: a variable with a use bound in the
   first is bound in the second, to the value that [rebuilt_value]
   relates to it at the node of the variable, and the static environments
   are related by [statics_rebuilt]. The second and third runs bind the
   same static values; a variable of the second that is neither unboxed
   nor without a use has one value in both, and an unboxed one has the
   variables of its fields bound in the third to the values at their
   paths ([unboxed_values]), at the states that [uc] relates with their
   environments: all but those within a wrapper of both runs, about to
   apply the continuation of a call ([related_pending]), which carry none.
   The rebuild keeps a [Let] of a variable with a use, so the states of
   the second run that are the rebuild of such a [Let] are not of those.

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes) and, but the example, [_local] for a
   least solution of the rules with the fix of local fields (whose check
   replaces the premise that no field is local or that no slot is
   projected); two names are shortened,
   [unbox_calls_result_end_to_end_local_landed_statement] and
   [calls_unboxed_result_end_to_end_local_landed_statement]:
   - [unbox_calls_types_end_to_end_statement]: the statement of the runs
     of FlambdaReaperTypesTermSpec.types_end_to_end_term_statement, of
     the unit rebuilt with unboxing across calls (for any table
     [cannot_change] that passes [check_cc_closed] and
     [check_apply_widgets], and any code [nr] taken to never return, of
     the fragment [calls_unboxing_nr] with [check_unboxed_reads]) in
     place of the plain rebuild: where the unit exits, the rebuilt unit
     exits through the same continuation with the same heap, and the type
     of each symbol bound, rewritten, holds of the symbol there. The
     unboxing changes no representation of a closure, so the types are
     those of the plain rewriting ([plain_context]).
   - [repr_calls_types_end_to_end_statement]: the same of the unit
     rebuilt with unboxing and changes of representation of sets of
     closures (FlambdaReaperUnboxing.rebuild_unit_unbox_repr_fixed, for a
     table [tcr] of the sets that change, of the fragment
     [repr_calls_unboxing_nr] of
     FlambdaReaperUnboxingRefinesClosuresCallsSpec.v), whose simulation
     ([rr], FlambdaReaperUnboxingRefinesClosuresSimProofs.v) binds the
     same static values in the two rebuilds: a closure that changes is
     bound by a [Let], neither held in a field nor static. The types are
     those of the plain rewriting.
   - [result_type_end_to_end_statement]: the statement of result types
     (FlambdaReaperTypesSpec.rewrite_result_type_sound_statement) with no
     premise on the runs: at each state that the run of the unit comes to
     and from which it steps (a return of code among them), the type that
     the result types give [type_var], which holds of the value of the
     variable [code_var] there (a parameter of the code, or any variable
     with a use that the run binds), rewritten under the metadata of
     [code_var], holds of the value of [code_var] at a state that the run
     of the plain rebuild comes to. [cc_result_type_end_to_end_statement]:
     the same of the rebuild with changes of calling convention.
   - [unbox_calls_result_type_end_to_end_statement]: the same of the
     unit rebuilt with unboxing across calls, for a variable of the unit
     that it does not unbox, at a state of the run of the unit at a [Let]
     of a variable with a use, if the run of the unit does not go wrong.
     [calls_unboxed_result_type_end_to_end_statement]: the same for a
     variable that it unboxes, with the statement of the new parameters
     and results of an unboxed variable
     (FlambdaReaperTypesSpec.rewrite_unboxed_result_type_sound_statement):
     the types that [rewrite_unboxed_result_type] gives the fields of
     [code_var] hold of the values of the variables of the fields in the
     run of the unit rebuilt with unboxing. The value of the variable in
     the plain rebuild, which that statement asks, is that of the rebuild
     with changes of calling convention, which the unboxing across calls
     refines.
   - [unbox_calls_example_statement]: the unit of
     FlambdaReaperUnboxingRefinesSpec.CallExample, whose code takes a
     pair of which it reads a field, unboxed across the call, with its
     tables, which are the least solution; its result types give the
     parameter of the code the type of a block whose field 1 is 8. It
     meets the premises of the statements of the result types, the rules
     unbox the parameter into the variable of its field, the rebuild with
     unboxing is computed, the type of the field rewritten is [= 8], and
     the run of the rebuilt unit binds the variable of the field to 8, of
     which it holds.

   Of the premises: those of the end-to-end statements of the plain
   rebuild (FlambdaReaperTypesTermSpec.v) and of the refinement of the
   unit by the rebuild with unboxing across calls
   (FlambdaReaperUnboxingRefinesCallsSpec
   .rebuild_unbox_fixed_observably_never_returns_statement) or with
   changes of representation (FlambdaReaperUnboxingRefinesClosuresCallsSpec
   .rebuild_repr_calls_observably_statement), and, for the
   result types, a variable with a use bound at a state from which the
   run of the unit steps. The variable has a use: the rebuild keeps no
   other, and the change of calling conventions deletes a parameter or a
   result with no use, whose type it drops. For the third run, the unit
   does not go wrong, so that the run of its first rebuild, which
   refines it, does not either, and the state is at a [Let] of a variable
   with a use (above).

   What is left out:
   - The results of code: the model's traversal gives the results of
     code nodes that are not variables ([Node_made], where the compiler
     has the variables of [code_dep.return]), and
     FlambdaReaperTypes.rewrite_result_type reads the metadata of a
     variable; the statements are of the variables that the run binds.
   - The fragment [loose] (closure types that are not aliases), which asks
     that the static sets be kept whole, of the run of the rebuild with
     changes of calling convention: FlambdaReaperTypesProofs.v proves it
     of the plain rebuild only (its section [Rebuilt_sets]).
   - The states within a wrapper of both runs, for the third run.
   - The closure types of the sets whose representation changes, of the
     runs of [rr]: the statements of those types
     (FlambdaReaperTypesSpec.rewrite_changed_type_sound_slots_statement
     and those after it) ask of the closure of the third run
     [changed_closure_rebuilt]: that it holds the new value slots of the
     representation and no other, in their order. What [rr] has of it
     ([crel]) is that each new value slot holds the value at its path,
     with nothing of the other value slots of the closure. The rebuild
     makes the closure with exactly those value slots
     (FlambdaReaperUnboxing.changed_value_slots), but the relation of the
     simulation does not carry it; applying the statements to its runs
     needs [crel] to say so, an addition to that relation.
   - The rewriting of the types of the symbols with the changes of
     representation: the statements of symbols are of the plain rewriting
     ([plain_context], where no set changes). With the changes,
     FlambdaReaperTypes.rewrite forgets the closure types of the sets
     that change, and the compiler rewrites them into the new
     representation (FlambdaReaperTypes.rewrite_arbitrary_u); the
     fragment of [rr] holds no closure that changes in a symbol. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperCallingConventions.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperCallingConventionsSpec FlambdaRenameApartSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* The least solution: of the rules with the fix of local fields if
   [local], otherwise of the rules, with no field local or no slot
   projected. *)
Definition least_with (local : bool) (is_local : field -> bool)
    (F : facts) (T : tables) (e : expr) : Prop :=
  if local then least_solution_local is_local F T
  else
    least_solution is_local F T
    ∧ ((∀ f : field, is_local f = false) ∨ projection_free e = true).

(* The premises on the unit of the end-to-end statements of the plain
   rebuild (FlambdaReaperTypesTermSpec.types_end_to_end_term_statement),
   with the least solution [least_with local]. *)
Definition unit_premises (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) : Prop :=
  is_local_slots_only is_local
  ∧ least_with local is_local (traverse_unit flag u e) T e
  ∧ in_flow_fragment e = true
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments e = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] e = true
  ∧ well_formed_unit u e = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value u e = true
  ∧ variables_bound_once e = true
  ∧ unit_of_values e = true
  ∧ sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets.

(* The premises on the tables of the refinement of the unit by its
   rebuild with unboxing across calls
   (FlambdaReaperUnboxingRefinesCallsSpec
   .rebuild_unbox_fixed_observably_never_returns_statement): the table
   [cannot_change] passes its checks, and the unit is of the fragment for
   the code [nr] taken to never return. *)
Definition calls_premises (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (nr : code_id -> bool) : Prop :=
  check_cc_closed (traverse_unit flag u e) T cannot_change = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true
  ∧ FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr nr
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true
  ∧ FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true.

(* The unit rebuilt with unboxing across calls. *)
Definition calls_rebuilt (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) : expr :=
  FlambdaReaperUnboxingRefinesCallsSpec.rebuild_unit_unbox_fixed
    (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
    dominated to_unbox preserve_direct_calls e.

(* The state that a unit starts in, run on its own. *)
Definition initial_state (e : expr) : state :=
  State e [] [] empty_static_env empty_heap no_reads.

(* ======================================================================== *)
(* The exported types                                                       *)
(* ======================================================================== *)

(* FlambdaReaperTypesTermSpec.types_end_to_end_term_with, of the unit
   rebuilt with unboxing across calls in place of the plain rebuild, for
   a least solution [least_with local]. *)
Definition unbox_calls_types_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unbox_calls_types_end_to_end_statement : Prop :=
  unbox_calls_types_end_to_end_with false all_fixes.

Definition unbox_calls_types_end_to_end_landed_statement : Prop :=
  unbox_calls_types_end_to_end_with false landed_fixes.

Definition unbox_calls_types_end_to_end_local_statement : Prop :=
  unbox_calls_types_end_to_end_with true all_fixes.

Definition unbox_calls_types_end_to_end_local_landed_statement : Prop :=
  unbox_calls_types_end_to_end_with true landed_fixes.

(* The premises on the tables of the refinement of the unit by its
   rebuild with unboxing and changes of representation
   (FlambdaReaperUnboxingRefinesClosuresCallsSpec
   .rebuild_repr_calls_observably_statement), for the table [tcr] of the
   sets whose representation changes. *)
Definition repr_premises (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (nr : code_id -> bool) : Prop :=
  check_cc_closed (traverse_unit flag u e) T cannot_change = true
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true
  ∧ FlambdaReaperUnboxingRefinesClosuresCallsSpec.repr_calls_unboxing_nr nr
      is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true
  ∧ FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true.

(* The unit rebuilt with unboxing and changes of representation. *)
Definition repr_rebuilt (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool) : expr :=
  FlambdaReaperUnboxing.rebuild_unit_unbox_repr_fixed is_local
    (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
    dominated to_unbox tcr preserve_direct_calls e.

(* [unbox_calls_types_end_to_end_with] of the unit rebuilt with unboxing
   and changes of representation, of the fragment of
   FlambdaReaperUnboxingRefinesClosuresCallsSpec.v. The types are those
   of the plain rewriting ([plain_context]). *)
Definition repr_calls_types_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    unit_premises local flag is_local u e T sets ->
    repr_premises flag is_local u e T cannot_change dominated to_unbox tcr nr ->
    run n (initial_state e) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
      run m
        (initial_state
           (repr_rebuilt flag is_local u e T cannot_change dominated to_unbox
              tcr preserve_direct_calls))
      = Step t
      ∧ step t = Exit k vs' hp rd'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition repr_calls_types_end_to_end_statement : Prop :=
  repr_calls_types_end_to_end_with false all_fixes.

Definition repr_calls_types_end_to_end_landed_statement : Prop :=
  repr_calls_types_end_to_end_with false landed_fixes.

Definition repr_calls_types_end_to_end_local_statement : Prop :=
  repr_calls_types_end_to_end_with true all_fixes.

Definition repr_calls_types_end_to_end_local_landed_statement : Prop :=
  repr_calls_types_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* The result types of code                                                 *)
(* ======================================================================== *)

(* The premises on a state [s] of the run of the unit and the variable
   [code_var] that it binds to [v]: the run of the unit comes to [s] and
   steps from it, [code_var] has a use, and a valuation of the names of
   the result types [tenv] that gives [type_var] the value [v] satisfies
   them, with the copies of the rewriting valued as [copies_valued]
   says. *)
Definition result_premises (F : facts) (T : tables) (e : expr) (n : nat)
    (s : state) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (code_var type_var : variable) (v : value)
    : Prop :=
  run n (initial_state e) = Step s
  ∧ step s <> Undefined
  ∧ lookup code_var (state_vars s) = Some v
  ∧ s_has_usage (solution_of_tables T) (Node_variable code_var) = true
  ∧ typing_env_satisfied_strictly (state_statics s) valuation tenv
  ∧ env_in_fragment valuation tenv
  ∧ copies_valued tenv canon valuation valuation'
  ∧ Typing_env.find tenv (Name_var type_var) None <> None
  ∧ lookup type_var valuation = Some v.

(* FlambdaReaperTypesSpec.rewrite_result_type_sound_with of the runs of
   the unit and of the plain rebuild: the rewritten type holds of the
   value of [code_var] at a state that the run of the rebuilt unit comes
   to. *)
Definition result_type_end_to_end_with (local : bool) (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition result_type_end_to_end_statement : Prop :=
  result_type_end_to_end_with false all_fixes.

Definition result_type_end_to_end_landed_statement : Prop :=
  result_type_end_to_end_with false landed_fixes.

Definition result_type_end_to_end_local_statement : Prop :=
  result_type_end_to_end_with true all_fixes.

Definition result_type_end_to_end_local_landed_statement : Prop :=
  result_type_end_to_end_with true landed_fixes.

(* The same of the rebuild with changes of calling convention, for a
   table [cannot_change] that passes its checks. *)
Definition cc_result_type_end_to_end_with (local : bool) (fx : fixes)
    : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) preserve_direct_calls
              e))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition cc_result_type_end_to_end_statement : Prop :=
  cc_result_type_end_to_end_with false all_fixes.

Definition cc_result_type_end_to_end_landed_statement : Prop :=
  cc_result_type_end_to_end_with false landed_fixes.

Definition cc_result_type_end_to_end_local_statement : Prop :=
  cc_result_type_end_to_end_with true all_fixes.

Definition cc_result_type_end_to_end_local_landed_statement : Prop :=
  cc_result_type_end_to_end_with true landed_fixes.

(* The premises of the third run: the run of the unit does not go wrong,
   and [s] is at a [Let] of a variable with a use; [code_var] is a
   variable of the unit. *)
Definition third_run_premises (F : facts) (T : tables) (e : expr)
    (s : state) (code_var : variable) : Prop :=
  ~ behaves (initial_state e) Goes_wrong
  ∧ (∃ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
       state_expr s = Let y kind d body
       ∧ s_has_usage (solution_of_tables T) (Node_variable y) = true)
  ∧ code_var ∈ FlambdaRenameApartSpec.variables_of e.

(* The same of the unit rebuilt with unboxing across calls, for a
   variable that it does not unbox. *)
Definition unbox_calls_result_type_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    third_run_premises (traverse_unit flag u e) T e s code_var ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              (traverse_unit flag u e) T dominated to_unbox e))
    = None ->
    ∃ (m : nat) (t : state) (v' : value),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ lookup code_var (state_vars t) = Some v'
      ∧ type_satisfied (state_statics t) valuation' v'
          (rewrite_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon code_var type_var).

Definition unbox_calls_result_type_end_to_end_statement : Prop :=
  unbox_calls_result_type_end_to_end_with false all_fixes.

Definition unbox_calls_result_type_end_to_end_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_with false landed_fixes.

Definition unbox_calls_result_type_end_to_end_local_statement : Prop :=
  unbox_calls_result_type_end_to_end_with true all_fixes.

Definition unbox_calls_result_end_to_end_local_landed_statement : Prop :=
  unbox_calls_result_type_end_to_end_with true landed_fixes.

(* FlambdaReaperTypesSpec.rewrite_unboxed_result_type_sound_with of the
   runs of the unit and of its rebuild with unboxing across calls, for a
   variable [code_var] that the rebuild unboxes with [fields]: the types
   of the fields hold of the values [ws] of their variables at a state
   that the run of the rebuilt unit comes to. *)
Definition calls_unboxed_result_type_end_to_end_with (local : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v : value),
    unit_premises local flag is_local u e T sets ->
    calls_premises flag u e T cannot_change dominated to_unbox nr ->
    result_premises (traverse_unit flag u e) T e n s tenv canon valuation
      valuation' code_var type_var v ->
    third_run_premises (traverse_unit flag u e) T e s code_var ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              (traverse_unit flag u e) T dominated to_unbox e))
    = Some fields ->
    ∃ (m : nat) (t : state) (ws : list value),
      run m
        (initial_state
           (calls_rebuilt flag u e T cannot_change dominated to_unbox
              preserve_direct_calls))
      = Step t
      ∧ Forall2 (fun (xk : variable * flambda_kind) (w : value) =>
                   lookup (fst xk) (state_vars t) = Some w)
          (FlambdaReaperUnboxing.fields_leaves fields) ws
      ∧ Forall2 (type_satisfied (state_statics t) valuation') ws
          (rewrite_unboxed_result_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             (FlambdaReaperUnboxing.bound_code_my_closures e) canon code_var
             type_var fields).

Definition calls_unboxed_result_type_end_to_end_statement : Prop :=
  calls_unboxed_result_type_end_to_end_with false all_fixes.

Definition calls_unboxed_result_type_end_to_end_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_with false landed_fixes.

Definition calls_unboxed_result_type_end_to_end_local_statement : Prop :=
  calls_unboxed_result_type_end_to_end_with true all_fixes.

Definition calls_unboxed_result_end_to_end_local_landed_statement : Prop :=
  calls_unboxed_result_type_end_to_end_with true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesSpec.CallExample:

     code 50 (x40) =                  (my_closure 41, return 3)
       let x43 = x40.(1) in
       let x42 = [x43; 9] in
       return x42
     let x60 = closure 70 of the code 50 in
     let_cont k20 (x30) = let x31 = x30.(0) in return x31 in
     let x10 = [7; 8] in
     apply x60 x10 -> k20

   with its tables, which are the least solution. The rules unbox the
   parameter [x40] into the variable 66 of its field 1, and the result,
   so that the call gives the code 8 and the code returns 8. The result
   types of the code give its parameter, the variable 90 of the result
   types, the type of a block of tag 0 whose field 1 is 8, which holds
   of [7; 8]. *)
Module UnboxCallsExample.

Local Open Scope positive_scope.

Definition unit_expr : expr :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.unit_expr.

Definition facts : facts := FlambdaReaperUnboxingRefinesSpec.CallExample.facts.

Definition tables : tables :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.tables.

Definition cannot_change : list node :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.cannot_change.

Definition dominated : list (node * node) :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.dominated.

Definition to_unbox : list node :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.to_unbox.

Definition rebuilt : expr :=
  FlambdaReaperUnboxingRefinesSpec.CallExample.rebuilt.

(* The set of closures of [x60]. *)
Definition sets : list set_of_closures :=
  [[(70, (Node_variable 60, Or_unknown.Known 50))]].

(* The unboxed fields of [x40]. *)
Definition fields : FlambdaReaperUnboxing.unboxed_fields :=
  [(Field_block 1 Value, FlambdaReaperUnboxing.Not_unboxed 66)].

Definition param_type : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [unknown Value; this_tagged_immediate 8%Z].

(* The result types: the variable 90 of the type [param_type]. *)
Definition tenv : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create 90 Value
       Name_mode.normal)
    (Name_var 90) param_type.

Definition pair : value :=
  Value_block 0 [Value_const (Tagged_immediate 7);
                 Value_const (Tagged_immediate 8)].

Definition valuation : env := [(90, pair)].

Definition canon : canon_fn := fun n _ _ => n.

End UnboxCallsExample.

(* The unit meets the premises of
   [calls_unboxed_result_type_end_to_end_statement], with the tables of
   the least solution, no local field, the set of closures of [x60], the
   code 50 taken to never return by none, and, at the state of the run of
   the unit at the [Let] of [x43], the parameter [x40] bound to [7; 8],
   the variable 90 of the result types at [7; 8]; the rules unbox [x40]
   into the variable 66 of its field 1; the rebuilt unit is [rebuilt]. The
   type of the field, rewritten, is [= 8]; the run of [rebuilt], at the
   [Let] of [x43] in the code, binds 66 to 8, of which it holds. *)
Definition unbox_calls_example_with (fx : fixes) : Prop :=
  unit_premises false Auto FlambdaReaperUnboxing.Examples.no_local
    FlambdaReaperUnboxing.Examples.interface UnboxCallsExample.unit_expr
    UnboxCallsExample.tables UnboxCallsExample.sets
  ∧ calls_premises Auto FlambdaReaperUnboxing.Examples.interface
      UnboxCallsExample.unit_expr UnboxCallsExample.tables
      UnboxCallsExample.cannot_change UnboxCallsExample.dominated
      UnboxCallsExample.to_unbox (fun _ => false)
  ∧ Keyed.lookup node_eqb (Node_variable 40%positive)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields UnboxCallsExample.facts
              UnboxCallsExample.tables UnboxCallsExample.dominated
              UnboxCallsExample.to_unbox UnboxCallsExample.unit_expr))
    = Some UnboxCallsExample.fields
  ∧ calls_rebuilt Auto FlambdaReaperUnboxing.Examples.interface
      UnboxCallsExample.unit_expr UnboxCallsExample.tables
      UnboxCallsExample.cannot_change UnboxCallsExample.dominated
      UnboxCallsExample.to_unbox true
    = UnboxCallsExample.rebuilt
  ∧ (∃ s : state,
       result_premises UnboxCallsExample.facts UnboxCallsExample.tables
         UnboxCallsExample.unit_expr 5 s UnboxCallsExample.tenv
         UnboxCallsExample.canon UnboxCallsExample.valuation
         UnboxCallsExample.valuation 40%positive 90%positive
         UnboxCallsExample.pair
       ∧ third_run_premises UnboxCallsExample.facts UnboxCallsExample.tables
           UnboxCallsExample.unit_expr s 40%positive)
  ∧ rewrite_unboxed_result_type fx
      (plain_context FlambdaReaperUnboxing.Examples.no_local
         UnboxCallsExample.facts UnboxCallsExample.tables
         UnboxCallsExample.sets)
      UnboxCallsExample.tenv
      (FlambdaReaperUnboxing.bound_code_my_closures
         UnboxCallsExample.unit_expr)
      UnboxCallsExample.canon 40%positive 90%positive
      UnboxCallsExample.fields
    = [this_tagged_immediate 8%Z]
  ∧ ∃ t : state,
      run 5 (initial_state UnboxCallsExample.rebuilt) = Step t
      ∧ lookup 66%positive (state_vars t)
        = Some (Value_const (Tagged_immediate 8))
      ∧ type_satisfied (state_statics t) UnboxCallsExample.valuation
          (Value_const (Tagged_immediate 8)) (this_tagged_immediate 8%Z).

Definition unbox_calls_example_statement : Prop :=
  unbox_calls_example_with all_fixes.

Definition unbox_calls_example_landed_statement : Prop :=
  unbox_calls_example_with landed_fixes.
