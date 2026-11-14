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

(* The two actions on what may follow a unit.

   This file is to be reviewed: the statements. It has no definition.

   FlambdaObserversSpec.v says what a unit may be followed by
   ([observer], [observers]), by what two results of a step are
   compared there ([observed_refines], which compares an exit by its
   continuation, by the constants and the locations among its values
   ([same_scalars]) and by the locations that its heap binds), when no
   observer tells two worlds apart ([worlds_observed]), and the form in
   which a pass proves it of two units ([refines_under_observers]).
   FlambdaEquivarianceSpec.v has the action of the renamings of
   variables and continuations on states, FlambdaLocationsSpec.v that
   of the permutations of locations, and FlambdaExitWorldActionsSpec.v
   the two on a world. This file says how the definitions of
   FlambdaObserversSpec.v go through them.

   A renaming. What is renamed is renamed jointly: the continuation of
   an exit, what a continuation stands for, a continuation environment,
   the code of a static environment and the two terms, by one renaming.
   - What a step comes to has the behaviours of what it comes to
     renamed, renamed ([step_result_behaves_equivariant_statement]),
     and [observed_refines] holds of two results exactly when it holds
     of the two renamed ([observed_refines_equivariant_statement]): an
     exit is compared by its continuation, which is renamed on both
     sides, and by values and a heap, which hold no variables or
     continuations.
   - The three tests of an observer answer the same of a term renamed:
     a renaming brings no static binding in or out
     ([binds_no_static_equivariant_statement]), it renames no symbol
     and no code id ([expr_statics_in_scope_equivariant_statement]),
     and no value slot
     (FlambdaUnusedSlotsSpec.slots_unused_equivariant_statement). So
     what a continuation stands for is an observer exactly when it is
     one renamed ([observer_equivariant_statement]), and so of a
     continuation environment ([observers_equivariant_statement]).
   - So [worlds_observed] relates two worlds exactly when it relates
     the two renamed ([worlds_observed_equivariant_statement]): every
     renaming keeps it, in the sense of
     FlambdaExitWorldActionsSpec.refines_world_kept_statement, and the
     comparison of two units is kept both ways
     ([refines_world_observed_equivariant_statement]).
   - The form in which a pass proves it holds of two terms, from a
     static environment, exactly when it holds of the two terms and
     of the code of the static environment renamed
     ([refines_under_observers_equivariant_statement]). The heap and
     the read record are left: they hold no variables or
     continuations.
   None of these has a premise.

   A class of what may follow a unit. FlambdaObserversSpec.v has the two
   relations over a class of continuation values that is given, in the
   place of the observers ([worlds_observed_by], [refines_observed_by]);
   the two over the observers are those at the class [observer slots].
   For every class that a renaming keeps, in that what is in the class
   is in it renamed, the two relations, and the comparison of two units
   by the first, hold exactly when they hold of what is renamed
   ([worlds_observed_by_equivariant_statement],
   [refines_world_observed_by_equivariant_statement],
   [refines_observed_by_equivariant_statement]). The three statements
   above that have a universal over the observers on each side are
   these at the class [observer slots], which a renaming keeps
   ([observer_equivariant_statement]); and these say the same of a
   class of some of the observers, where a renaming keeps it. The class
   is under an outer universal and nothing is asked of it but that: no
   observer is named.

   A permutation of locations. One permutation, applied to both sides.
   - A value is a constant or a location exactly when the value moved
     is ([scalar_permuted_statement]), so two lists of values have the
     same constants and the same locations at the same places exactly
     when the two lists moved have
     ([same_scalars_permuted_statement]).
   - The locations that a heap moved binds are those that the heap
     binds, moved, in their order and as many times
     ([heap_domain_permuted_statement]).
   - So what the way out compares of two worlds
     (FlambdaObserversSpec.worlds_observed_way_out_statement: the
     continuation, the constants and the locations among the values,
     the locations that the heap binds) is the same of the two worlds
     exactly when it is the same of the two moved by one permutation
     ([way_out_permuted_statement]).

   What is not here. [observed_refines] of two results moved, and
   [worlds_observed] of two worlds moved. What is proved of a run
   under a permutation of locations
   (FlambdaLocationsMachineSpec.run_permuted_statement,
   FlambdaExitWorldActionsSpec.exits_world_permuted_statement) gives
   each run its exit up to a second permutation, which agrees with the
   first on the locations bound at the start and of which nothing else
   is said; the source and the target each have their own. The
   comparison by [same_scalars] and by the locations that the heaps
   bind is by [=], so those statements do not by themselves give the
   two laws. A handler that is an observer may allocate: a [Let] of
   the machine of FlambdaSemanticsSpec.v runs the primitives of the
   heap. Whether the two laws hold is not said here, either way.
   Nothing is said with two permutations.

   Nothing is trusted here. The proofs are in
   FlambdaObserversActionsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaEquivarianceSpec FlambdaStaticsInScopeSpec FlambdaHeapSpec
  FlambdaExitWorldSpec FlambdaRunUnderSpec FlambdaObserversSpec
  FlambdaLocationsSpec FlambdaExitWorldActionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A renaming                                                               *)
(* ======================================================================== *)

(* The behaviours of what a step comes to, renamed, are the behaviours
   of what it comes to, renamed. *)
Definition step_result_behaves_equivariant_statement : Prop :=
  ∀ (r : renaming) (result : step_result) (b : behaviour),
    step_result_behaves (rename_result r result) (rename_behaviour r b)
    <-> step_result_behaves result b.

(* Two results are compared as the two renamed are. *)
Definition observed_refines_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : step_result),
    observed_refines (rename_result r source) (rename_result r target)
    <-> observed_refines source target.

(* A renaming brings no static binding in or out. *)
Definition binds_no_static_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    binds_no_static (rename_expr r e) = binds_no_static e.

(* Nor does it rename a symbol or a code id: the static names of a
   term renamed are in scope where those of the term are. *)
Definition expr_statics_in_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (syms : list symbol) (ids : list code_id) (e : expr),
    expr_statics_in_scope syms ids (rename_expr r e)
    = expr_statics_in_scope syms ids e.

(* What a continuation stands for is an observer exactly when it is one
   renamed. The value slots are not renamed. *)
Definition observer_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (c : cont_value),
    observer slots (rename_cont_value r c) <-> observer slots c.

Definition observers_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (conts : cont_env),
    observers slots (rename_cont_env r conts) <-> observers slots conts.

(* No observer tells two worlds apart exactly when none tells the two
   renamed apart. *)
Definition worlds_observed_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (w w' : exit_world),
    worlds_observed slots (rename_exit_world r w) (rename_exit_world r w')
    <-> worlds_observed slots w w'.

(* So the comparison of two units is that of the two renamed. *)
Definition refines_world_observed_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (source target : state),
    refines_world (worlds_observed slots) (rename_state r source)
      (rename_state r target)
    <-> refines_world (worlds_observed slots) source target.

(* The form in which a pass proves it: of the two terms and of the code
   of the static environment renamed. *)
Definition refines_under_observers_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (statics : static_env)
    (hp : heap) (rd : read_record) (source target : expr),
    refines_under_observers slots (rename_static_env r statics) hp rd
      (rename_expr r source) (rename_expr r target)
    <-> refines_under_observers slots statics hp rd source target.

(* ======================================================================== *)
(* A class of what may follow a unit                                        *)
(* ======================================================================== *)

(* Of a class of continuation values that a renaming keeps: what is in
   the class is in it renamed. The other way follows, by the inverse
   renaming. No two worlds are told apart by the class exactly when the
   two renamed are not. *)
Definition worlds_observed_by_equivariant_statement : Prop :=
  ∀ O : cont_value -> Prop,
    (∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c)) ->
    ∀ (r : renaming) (w w' : exit_world),
      worlds_observed_by O (rename_exit_world r w) (rename_exit_world r w')
      <-> worlds_observed_by O w w'.

(* So the comparison of two units by the class is that of the two
   renamed. *)
Definition refines_world_observed_by_equivariant_statement : Prop :=
  ∀ O : cont_value -> Prop,
    (∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c)) ->
    ∀ (r : renaming) (source target : state),
      refines_world (worlds_observed_by O) (rename_state r source)
        (rename_state r target)
      <-> refines_world (worlds_observed_by O) source target.

(* The form in which a pass proves it, from the continuation
   environments of the class: of the two terms and of the code of the
   static environment renamed. *)
Definition refines_observed_by_equivariant_statement : Prop :=
  ∀ O : cont_value -> Prop,
    (∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c)) ->
    ∀ (r : renaming) (statics : static_env) (hp : heap) (rd : read_record)
      (source target : expr),
      refines_observed_by O (rename_static_env r statics) hp rd
        (rename_expr r source) (rename_expr r target)
      <-> refines_observed_by O statics hp rd source target.

(* ======================================================================== *)
(* A permutation of locations                                               *)
(* ======================================================================== *)

(* A constant stays the constant, a location goes to a location, and
   nothing else becomes either. *)
Definition scalar_permuted_statement : Prop :=
  ∀ (p : permutation) (v : value), scalar (permute_value p v) <-> scalar v.

(* One permutation, applied to both lists. *)
Definition same_scalars_permuted_statement : Prop :=
  ∀ (p : permutation) (vs vs' : list value),
    same_scalars (map (permute_value p) vs) (map (permute_value p) vs')
    <-> same_scalars vs vs'.

(* The locations that the heap moved binds: those that the heap binds,
   moved, in their order, one for each binding. *)
Definition heap_domain_permuted_statement : Prop :=
  ∀ (p : permutation) (hp : heap),
    map fst (permute_heap p hp) = map (permute p) (map fst hp).

(* What the way out compares of two worlds is the same of the two
   exactly when it is the same of the two moved by one permutation. *)
Definition way_out_permuted_statement : Prop :=
  ∀ (p : permutation) (w w' : exit_world),
    (exit_cont (permute_exit_world p w) = exit_cont (permute_exit_world p w')
     ∧ same_scalars (exit_values (permute_exit_world p w))
         (exit_values (permute_exit_world p w'))
     ∧ map fst (exit_heap (permute_exit_world p w))
       = map fst (exit_heap (permute_exit_world p w')))
    <-> (exit_cont w = exit_cont w'
         ∧ same_scalars (exit_values w) (exit_values w')
         ∧ map fst (exit_heap w) = map fst (exit_heap w')).
