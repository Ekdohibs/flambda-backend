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


(* What a refinement says of the symbols at an exit, with two tests
   of its target.

   This file is to be reviewed: [exit_statics_bound],
   [same_exit_bound] and the three statements.

   [FlambdaSemanticsSpec.refines] compares two exits by their
   continuation, their values and their heap, and a symbol is a value
   whether it is bound or not (FlambdaValueSpec.v). So a target may
   exit with a symbol that it never binds, where the source binds it
   and neither reads it: the two exits are the same, and the target
   refines the source. The statements of the validation conclude a
   refinement (FlambdaValidateSpec.validate_unit_sound_statement and
   those of the other functions that validate). So the refinement
   that a verdict concludes does not by itself say that what the term
   after exits with is bound. Decision D57.

   What says it is two tests of the term after, which no statement
   of the validation concludes of it.
   FlambdaStaticsInScopeSpec.statics_bound_in_run_statement is of a
   unit that passes [well_formed_unit] and [statics_in_scope], and of
   a run of it that starts with no variable and no continuation
   bound, in a static environment that has no code and binds the
   imports and nothing else, each to something that mentions no
   symbol but an import and no code, and with a heap whose cells hold
   only such values. Of every value that a state of that run holds,
   those in the cells of its heap included, and of every value that
   it exits with: a symbol that the value is or holds is bound in the
   static environment of that state, and the code of a closure that
   it is or holds is bound there or is declared deleted by the unit.

   This file joins that to a refinement of two units run from the
   same imports (FlambdaImportsSpec.unit_refines). Each of the two
   runs starts with no variable and no continuation bound, with the
   empty heap, and in the static environment that binds each import
   to something imported and nothing else
   (FlambdaImportsSpec.imports_statics): nothing is said of a run
   from another heap, or from imports that have contents. The
   statements are in the form of FlambdaExitWorldSpec.refines_world,
   which relates what the two runs leave at their exits:
   - [unit_refines_exits_bound_statement]: where the target of such a
     refinement passes the two tests, an exit of the source is
     answered by an exit of the target through the same continuation,
     with the same values and the same heap; every symbol that those
     values, or the values in the cells of that heap, are or hold is
     bound in the static environment of the target when it exits,
     where an import is bound from the start; and the code of every
     closure that they are or hold is bound there or is declared
     deleted by the target;
   - [validated_exits_bound_statement]: so of a verdict of
     [validate_unit], of the two terms run on their own, where the
     term after passes the two tests.

   The relation is asked at the exits of the target that answer exits
   of the source. The second half of [refines_world] allows a source
   that goes wrong, and asks no related exit in that case. The two
   tests of the target separately give, of every exit of the target
   run so, that it is as [exit_statics_bound] has it
   (FlambdaStaticsInScopeSpec.statics_bound_in_run_statement, and
   FlambdaSemanticsSpec.step_exit_heap_statement for the heap): that
   is a consequence of those tests and of the start, and is not what
   the relation says.

   What these two statements do not say. That a symbol is bound is
   not that it is bound to what the source bound it to: the two
   static environments are not compared ([same_exit] compares
   neither, and [exit_statics_bound] looks at the second alone). The
   two tests are not asked of the source, and nothing is said of
   what it binds. The two tests of the target are premises: nothing
   here concludes them from a verdict.

   That a refinement alone does not say it is shown by two units
   ([refined_exit_unbound_statement]): the second refines the first,
   and exits with a symbol that it has not bound. The two are not
   said to be a pair that the validation accepts: the statement is of
   the refinement, which is what a verdict concludes.

   The proofs are in FlambdaExitBoundProofs.v, which needs no
   review. *)

From Stdlib Require Import List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaHeapSpec FlambdaSemanticsSpec FlambdaExitWorldSpec
  FlambdaWellFormedSpec FlambdaStaticsInScopeSpec
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaLiftSpec
  FlambdaValidateSpec FlambdaImportsSpec.
Import ListNotations.

(* What the run exits with mentions only what its static environment
   binds when it exits, and the code that [deleted] names
   (FlambdaStaticsInScopeSpec.value_statics_bound): the values that it
   exits with, and the values that the cells of its heap hold
   (FlambdaHeapSpec.heap_values_all). *)
Definition exit_statics_bound (deleted : list code_id) (w : exit_world)
    : Prop :=
  Forall (value_statics_bound deleted (exit_statics w)) (exit_values w)
  ∧ heap_values_all (value_statics_bound deleted (exit_statics w))
      (exit_heap w).

(* The two exits are the same, as [FlambdaExitWorldSpec.same_exit] has
   it, and the second is as above. *)
Definition same_exit_bound (deleted : list code_id) (w w' : exit_world)
    : Prop :=
  same_exit w w' ∧ exit_statics_bound deleted w'.

(* A refinement of two units run from the same imports
   (FlambdaImportsSpec.unit_refines) whose target passes the two
   tests of FlambdaStaticsInScopeSpec.statics_bound_in_run_statement,
   given the symbols that are imported: the exits are related by
   [same_exit_bound], with the code that the target declares
   deleted. *)
Definition unit_refines_exits_bound_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (source target : expr),
    well_formed_unit u target = true ->
    statics_in_scope (map fst imports) target = true ->
    unit_refines imports source target ->
    refines_world (same_exit_bound (deleted_code_ids target))
      (State source [] [] (imports_statics imports) empty_heap no_reads)
      (State target [] [] (imports_statics imports) empty_heap no_reads).

(* So of a verdict of [validate_unit]: the two terms are run on their
   own, and the term after imports nothing. *)
Definition validated_exits_bound_statement : Prop :=
  ∀ (u : unit_interface) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    validate_unit rounds unused slots G Ls kept before after
    <> Not_validated ->
    well_formed_unit u after = true ->
    statics_in_scope [] after = true ->
    refines_world (same_exit_bound (deleted_code_ids after))
      (State before [] [] empty_static_env empty_heap no_reads)
      (State after [] [] empty_static_env empty_heap no_reads).

(* That a refinement does not say it, and that the tests do. There are
   two units of which the second refines the first, each run on its
   own, such that: the first passes the two tests, and exits with a
   symbol and the empty heap; the second exits through the same
   continuation with the same symbol and the same heap, and the
   static environment of its state that exits does not bind the
   symbol; and the second passes [well_formed_unit] and fails
   [statics_in_scope]. Nothing here says that the validation accepts
   the two.

   One witness, which is that of the proof: the first unit binds a
   symbol and returns it. One handler of the second binds the symbol
   and returns it, another returns the symbol alone, and the test of
   the second unit goes to the handler that returns it alone. That
   unit passes [well_formed_unit]: the handler that binds the symbol
   is outside loops and code
   (FlambdaWellFormedSpec.statics_outside_loops_and_code). It fails
   [statics_in_scope]: the handler that returns the symbol alone
   mentions it, and the way of the test to that handler binds
   nothing (FlambdaStaticsInScopeSpec.handler_scope). *)
Definition refined_exit_unbound_statement : Prop :=
  ∃ (u : unit_interface) (source target : expr) (n : nat) (last : state)
    (k : continuation) (s : symbol),
    well_formed_unit u source = true
    ∧ statics_in_scope [] source = true
    ∧ behaves (State source [] [] empty_static_env empty_heap no_reads)
        (Exits k [Value_symbol s] empty_heap)
    ∧ program_refines source target
    ∧ run n (State target [] [] empty_static_env empty_heap no_reads)
      = Step last
    ∧ step last = Exit k [Value_symbol s] empty_heap no_reads
    ∧ symbol_bound (state_statics last) s = false
    ∧ well_formed_unit u target = true
    ∧ statics_in_scope [] target = false.
