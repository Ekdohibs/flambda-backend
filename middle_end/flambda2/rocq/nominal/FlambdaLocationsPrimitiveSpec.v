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

(* The primitives under a permutation of locations.

   This file is to be reviewed: [locations_bound], [binds_location],
   [hidden_entry_heap], and the five statements.

   FlambdaLocationsSpec.v has the action of a permutation of locations
   on what the machine holds, and says of nothing that the machine
   does that it goes through the action. This file begins with the
   primitives: FlambdaPrimitiveSpec.eval_prim, which does not touch
   the heap, and FlambdaHeapSpec.exec_prim, which reads it, writes it
   and allocates in it.

   A primitive that does not touch the heap commutes with the action
   ([eval_prim_permuted_statement]): its value on the arguments moved,
   under the static environment moved, is its value moved, and it has
   no value on the ones where it has none on the others. This is the
   law FlambdaPrimitiveSpec.eval_prim_related_statement, that a
   primitive reads an allocated value through the observations of
   FlambdaValueSpec.v only, taken at the relation between a value and
   the value moved, and at the inverse permutation for the converse.

   So does a primitive that reads the heap or writes it
   ([exec_prim_permuted_exactly_statement]): every primitive but the
   two that bind a location ([binds_location]), under no premise.

   A primitive that binds a location does not commute with the
   action. The location is FlambdaHeapSpec.fresh_location of the heap,
   which is outside the domain of the heap and is computed from the
   numbers that the bound locations are, and the fresh location of the
   heap moved is not the fresh location moved. So the statement for
   every primitive ([exec_prim_permuted_statement]) is up to a second
   permutation: what the primitive does in the heap moved by [p] is
   what it does in the heap, moved by some [p'] that agrees with [p]
   on every location that the heap binds. (For an allocation [p'] is
   [p] followed by the swap of the fresh location moved and the fresh
   location of the heap moved; the statement does not say which it
   is.)

   The premise of that statement. The second permutation moves the
   heap and the arguments as the first does only where it agrees with
   the first on their locations, so the heap is asked to bind each of
   them ([locations_bound]). A location that the heap or an argument
   holds and that the heap does not bind may be the fresh location,
   and then the allocation captures it in the one heap and not in the
   other. The locations of a heap are those of all its entries
   (FlambdaLocationsSpec.locations_of_heap). A write is a new entry in
   front of the one that it replaces, which stays in the list, and a
   location that only such a hidden entry holds counts: the equation
   is [=] between heaps, which are lists. This asks more of the heap
   than the first half of FlambdaHeapSpec.memory_closed, which speaks
   of the cells that a lookup finds, and it asks nothing of kinds. The
   heap in which location 1 is bound to a block of no field, in front
   of an entry that binds it to a block whose field is location 2, is
   closed in that sense; its fresh location is 2; and an allocation in
   it, and in it moved by the swap of 2 and 5, give heaps that no
   permutation relates ([hidden_entry_statement]).

   The premise is kept ([exec_prim_keeps_locations_bound_statement]):
   if the heap binds the locations of its entries, of the arguments
   and of the static environment, then after the primitive the heap
   binds the locations of its entries and of the value of the
   primitive, and every location that it bound before.

   What is not here. A step of the machine and a run, up to such a
   permutation, under the premise that the heap of the state binds
   every location of the state. The link with the relation between
   outcomes of FlambdaHeapMachineSpec.v, which pairs locations by a
   correspondence.

   Nothing is trusted here. The proofs are in
   FlambdaLocationsPrimitiveProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaHeapSpec
  FlambdaLocationsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A primitive that does not touch the heap                                 *)
(* ======================================================================== *)

Definition eval_prim_permuted_statement : Prop :=
  ∀ (p : permutation) (statics : static_env) (prim : prim)
    (vs : list value),
    eval_prim (permute_static_env p statics) prim (map (permute_value p) vs)
    = option_map (permute_value p) (eval_prim statics prim vs).

(* ======================================================================== *)
(* A primitive that reads the heap or writes it                             *)
(* ======================================================================== *)

(* The value and the heap that a primitive gives, moved. *)
Definition permute_value_and_heap (p : permutation) (result : value * heap)
    : value * heap :=
  (permute_value p (fst result), permute_heap p (snd result)).

(* The primitives that bind a location: the two allocations of
   FlambdaHeapSpec.exec_heap_primitive. *)
Definition binds_location (prim : prim) : bool :=
  match prim with
  | Heap (Make_block_with_identity _ _) | Heap (Make_mutable_array _) => true
  | _ => false
  end.

Definition exec_prim_permuted_exactly_statement : Prop :=
  ∀ (p : permutation) (statics : static_env) (h : heap) (prim : prim)
    (vs : list value),
    binds_location prim = false ->
    exec_prim (permute_static_env p statics) (permute_heap p h) prim
      (map (permute_value p) vs)
    = option_map (permute_value_and_heap p) (exec_prim statics h prim vs).

(* ======================================================================== *)
(* Every primitive, up to the choice of a location                          *)
(* ======================================================================== *)

(* The heap binds each of these locations. *)
Definition locations_bound (h : heap) (locations : list location) : Prop :=
  ∀ l : location, l ∈ locations -> is_bound l h = true.

Definition locations_of_values (vs : list value) : list location :=
  flat_map locations_of_value vs.

Definition exec_prim_permuted_statement : Prop :=
  ∀ (p : permutation) (statics : static_env) (h : heap) (prim : prim)
    (vs : list value),
    locations_bound h (locations_of_heap h) ->
    locations_bound h (locations_of_values vs) ->
    ∃ p' : permutation,
      (∀ l : location, is_bound l h = true -> permute p' l = permute p l)
      ∧ exec_prim (permute_static_env p statics) (permute_heap p h) prim
          (map (permute_value p) vs)
        = option_map (permute_value_and_heap p')
            (exec_prim statics h prim vs).

(* The premise is kept, with what a state needs of the value. *)
Definition exec_prim_keeps_locations_bound_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (prim : prim) (vs : list value)
    (v : value),
    locations_bound h (locations_of_heap h) ->
    locations_bound h (locations_of_values vs) ->
    locations_bound h (locations_of_static_env statics) ->
    exec_prim statics h prim vs = Some (v, h') ->
    locations_bound h' (locations_of_heap h')
    ∧ locations_bound h' (locations_of_value v)
    ∧ (∀ l : location, is_bound l h = true -> is_bound l h' = true).

(* ======================================================================== *)
(* The premise counts the entries that a write hides                        *)
(* ======================================================================== *)

(* Location 1 bound to a block of no field, in front of an entry that
   binds it to a block whose field is location 2. *)
Definition hidden_entry_heap : heap :=
  [(1%positive, Cell_block Cell_mutable 0 []);
   (1%positive, Cell_block Cell_mutable 0 [Value_location 2%positive])].

(* It is closed as FlambdaHeapSpec.memory_closed says, and no
   permutation relates what the allocation of a block does in it to
   what the allocation does in it moved by the swap of 2 and 5. *)
Definition hidden_entry_statement : Prop :=
  memory_closed empty_static_env hidden_entry_heap
  ∧ ¬ (∃ p' : permutation,
         exec_prim
           (permute_static_env [(2%positive, 5%positive)] empty_static_env)
           (permute_heap [(2%positive, 5%positive)] hidden_entry_heap)
           (Heap (Make_block_with_identity Cell_mutable 0)) []
         = option_map (permute_value_and_heap p')
             (exec_prim empty_static_env hidden_entry_heap
                (Heap (Make_block_with_identity Cell_mutable 0)) [])).
