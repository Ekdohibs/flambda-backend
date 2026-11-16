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

(* The two actions on what a run leaves at an exit.

   This file is to be reviewed: [rename_exit_world],
   [rename_world_relation], [permute_exit_world],
   [world_locations_bound], and the statements.

   FlambdaExitWorldSpec.v has what a run leaves at an exit
   ([exit_world]: the continuation, the values, the heap, the read
   record, and the static environment of the state that takes the
   last step), that a state leaves a world ([exits_world]), and
   refinement with the worlds of the two exits related
   ([refines_world]).
   FlambdaEquivarianceSpec.v has the action of the renamings of
   variables and continuations on states, and FlambdaLocationsSpec.v
   that of the permutations of locations. This file extends both to a
   world, and says how [exits_world] and [refines_world] go through
   them.

   A renaming. A world is renamed as the state that leaves it is
   ([rename_exit_world]): its continuation is renamed, as the
   continuation of an exit is; the code of its static environment is
   renamed; its values, its heap and its read record hold no
   variables or continuations and are left. The action has inverses
   ([rename_exit_world_inverse_statement]). A state leaves a world
   exactly when the state renamed leaves the world renamed
   ([exits_world_equivariant_statement]). It has no premise, as the
   statements about a step and a run have none.
   Refinement with the worlds related is kept with the relation
   renamed as the two states are ([rename_world_relation],
   [refines_world_equivariant_statement]). So it is kept, at the
   relation as it is, where the renaming keeps the relation
   ([refines_world_kept_statement]). [same_exit], at which
   [refines_world] is FlambdaSemanticsSpec.refines, is such a
   relation for every renaming ([same_exit_equivariant_statement]).

   A permutation of locations. A world is moved in its values, in the
   statically allocated values of its static environment, in its heap
   and in its read record ([permute_exit_world]); its continuation is
   no location and code holds none. The premise is that of
   FlambdaLocationsMachineSpec.v: the heap of the state binds every
   location of the state. Under it the world that the state leaves
   has its locations bound by its heap ([world_locations_bound]),
   those of its static environment and of its read record among
   them, and a location that the heap of the state bound is bound by
   the heap of the world
   ([exits_world_keeps_locations_bound_statement]). A run that
   allocates chooses its locations, so the state moved leaves the
   world moved up to a second permutation, which agrees with the
   first on every location that the heap of the state binds
   ([exits_world_permuted_statement]). That agreement is all that is
   said of the second permutation. [same_exit] is kept where the two
   worlds are moved by one permutation
   ([same_exit_permuted_statement]). The two actions commute
   ([rename_permute_exit_world_statement]).

   What is not here. Refinement with the worlds related, under a
   permutation of locations: the source and the target each leave
   their world up to a permutation of their own, so the relation
   would have to be kept where its two worlds are moved by two
   permutations that agree on the locations bound at the start.
   [same_exit] is not: it compares values and heaps by [=]. It does
   not compare the read records.

   Nothing is trusted here. The proofs are in
   FlambdaExitWorldActionsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaEquivarianceSpec FlambdaHeapSpec FlambdaExitWorldSpec
  FlambdaLocationsSpec FlambdaLocationsPrimitiveSpec
  FlambdaLocationsMachineSpec.
Import ListNotations.

(* ======================================================================== *)
(* A renaming                                                               *)
(* ======================================================================== *)

(* The continuation of the exit is renamed, and the code of the static
   environment. The values, the heap and the read record hold no
   variables or continuations. *)
Definition rename_exit_world (r : renaming) (w : exit_world) : exit_world :=
  Exit_world (rename_continuation r (exit_cont w)) (exit_values w)
    (rename_static_env r (exit_statics w)) (exit_heap w) (exit_reads w).

Definition rename_exit_world_inverse_statement : Prop :=
  ∀ (r : renaming) (w : exit_world),
    rename_exit_world (inverse_renaming r) (rename_exit_world r w) = w
    ∧ rename_exit_world r (rename_exit_world (inverse_renaming r) w) = w.

(* A state leaves a world exactly when the state renamed leaves the
   world renamed. *)
Definition exits_world_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state) (w : exit_world),
    exits_world (rename_state r s) (rename_exit_world r w)
    <-> exits_world s w.

(* A relation between worlds, renamed: it relates two worlds that are
   the renamings of two worlds that the relation relates. *)
Definition rename_world_relation (r : renaming)
    (R : exit_world -> exit_world -> Prop)
    : exit_world -> exit_world -> Prop :=
  fun w w' =>
    R (rename_exit_world (inverse_renaming r) w)
      (rename_exit_world (inverse_renaming r) w').

(* Refinement with the worlds related is kept, the relation renamed
   with the two states. *)
Definition refines_world_equivariant_statement : Prop :=
  ∀ (r : renaming) (R : exit_world -> exit_world -> Prop)
    (source target : state),
    refines_world R source target ->
    refines_world (rename_world_relation r R) (rename_state r source)
      (rename_state r target).

(* So it is kept at the relation as it is, where the renaming keeps the
   relation. *)
Definition refines_world_kept_statement : Prop :=
  ∀ (r : renaming) (R : exit_world -> exit_world -> Prop)
    (source target : state),
    (∀ w w' : exit_world,
      R w w' -> R (rename_exit_world r w) (rename_exit_world r w')) ->
    refines_world R source target ->
    refines_world R (rename_state r source) (rename_state r target).

(* Every renaming keeps [same_exit]. *)
Definition same_exit_equivariant_statement : Prop :=
  ∀ (r : renaming) (w w' : exit_world),
    same_exit (rename_exit_world r w) (rename_exit_world r w')
    <-> same_exit w w'.

(* ======================================================================== *)
(* A permutation of locations                                               *)
(* ======================================================================== *)

(* The values, the statically allocated values of the static
   environment, the heap and the read record are moved. The
   continuation of the exit is not a location. *)
Definition permute_exit_world (p : permutation) (w : exit_world)
    : exit_world :=
  Exit_world (exit_cont w) (map (permute_value p) (exit_values w))
    (permute_static_env p (exit_statics w)) (permute_heap p (exit_heap w))
      (permute_reads p (exit_reads w)).

(* The heap of the world binds every location of the world. *)
Definition world_locations_bound (w : exit_world) : Prop :=
  locations_bound (exit_heap w)
    (locations_of_values (exit_values w)
     ++ locations_of_static_env (exit_statics w)
     ++ locations_of_heap (exit_heap w)
     ++ locations_of_reads (exit_reads w)).

(* The world that a state leaves has it where the state has, and a
   location that the heap of the state binds is bound by the heap of
   the world. *)
Definition exits_world_keeps_locations_bound_statement : Prop :=
  ∀ (s : state) (w : exit_world),
    state_locations_bound s ->
    exits_world s w ->
    world_locations_bound w
    ∧ (∀ l : location,
         is_bound l (state_heap s) = true -> is_bound l (exit_heap w) = true).

(* The state moved leaves the world moved, up to the locations
   allocated: by a second permutation, which agrees with the first on
   the locations that the heap of the state binds. *)
Definition exits_world_permuted_statement : Prop :=
  ∀ (p : permutation) (s : state) (w : exit_world),
    state_locations_bound s ->
    exits_world s w ->
    ∃ p' : permutation,
      (∀ l : location,
         is_bound l (state_heap s) = true -> permute p' l = permute p l)
      ∧ exits_world (permute_state p s) (permute_exit_world p' w).

(* One permutation, applied to both worlds, keeps [same_exit]. *)
Definition same_exit_permuted_statement : Prop :=
  ∀ (p : permutation) (w w' : exit_world),
    same_exit w w' ->
    same_exit (permute_exit_world p w) (permute_exit_world p w').

(* ======================================================================== *)
(* The two actions                                                          *)
(* ======================================================================== *)

(* They commute: the first moves nothing in a value, the second nothing
   in code or in a continuation. *)
Definition rename_permute_exit_world_statement : Prop :=
  ∀ (r : renaming) (p : permutation) (w : exit_world),
    rename_exit_world r (permute_exit_world p w)
    = permute_exit_world p (rename_exit_world r w).
