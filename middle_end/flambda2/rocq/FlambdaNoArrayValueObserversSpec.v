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

(* What the array value leaves as it was, of what may follow a unit.

   FlambdaObserversSpec.v says what may follow a unit by the terms that
   may be run next: an [observer] is a way out, or a handler with no
   environment of its own that meets three conditions on its text.
   [Make_immutable_array] (FlambdaPrimitiveSpec.v) is a primitive more
   that such a handler may be written with, so with it there are
   observers more. Two relations of that file are about every observer:
   [worlds_observed], of two worlds that exits leave, and
   [refines_under_observers], of two units. Of two worlds, and of two
   units, they ask what they asked and more: what the observers that
   make an array come to as well.

   Whether they relate fewer pairs than they did is not known. That an
   observer which makes an array tells apart only what some observer
   which makes none tells apart is not stated in this tree, and nor is
   the contrary. So:
   - a statement that concludes one of the two relations says more
     than it did, and gives what it gave
     ([worlds_observed_gives_free_of_arrays_statement],
     [refines_under_observers_gives_free_of_arrays_statement]);
   - a statement that has one in a premise, or one on each side of an
     equivalence, is not given back by the statement about every
     observer, whose premise asks more. There are six in
     FlambdaObserversSpec.v and FlambdaObserversActionsSpec.v: the way
     out, the composition, the two forms, and the three that say that a
     renaming keeps the relations. And a seventh has
     FlambdaSemanticsSpec.expr_refines in its premise, which is about
     every environment, those that hold an array value or keep a
     handler that makes one included.
   This file states the seven for the observers that make no array.

   The observers that make no array are a class of what a continuation
   may stand for, [observer_free_of_arrays slots], and the two
   relations of that class are those of FlambdaObserversSpec.v
   ([worlds_observed_by], [refines_observed_by]): the two relations
   with every observer that makes an array left out, and nothing else
   changed. An environment every entry of which is in the class is an
   environment of observers that is free of arrays
   ([observers_free_of_arrays_statement]), which is what a state free
   of arrays asks of its continuations
   (FlambdaNoArrayValueSpec.conts_free_of_arrays).
   [expr_refines_free_of_arrays] is [expr_refines] of the whole source
   states that are free of arrays.

   In six of the seven the worlds, the units, the static environment
   and the heap are any, as in the statement that is restated. The
   seventh, [expr_refines_free_of_arrays_under_observers_statement], is
   of a source that makes no array, from a static environment and a
   heap that are free of arrays: its premise says nothing of another
   start.

   That, of two worlds and of two units in which there is no array,
   these are the relations that there were before the array value is no
   statement of this file. A world with no array in it is more than one
   whose values hold no array value: what its symbols are bound to and
   its heap hold none either, and no code of its static environment
   makes an array (FlambdaNoArrayValueSpec.statics_no_array_value,
   [heap_no_array_value], [code_makes_no_array]); a world whose values
   hold none may keep code that makes one, which an observer may call.
   A unit with no array in it makes none, and starts from a static
   environment and a heap of that kind. What is said then rests on
   FlambdaNoArrayValueSpec.v: an observer that makes no array, applied
   to such a world, starts a state free of arrays
   ([state_free_of_arrays]), whose run is the run that there was; and
   the world that such a unit leaves is such a world
   ([cont_free_of_arrays] is of that file, and says of a handler with no
   environment that it makes no array, at any depth).

   This file is to be reviewed: the two definitions and the ten
   statements. The proofs are in FlambdaNoArrayValueObserversProofs.v,
   which needs no review. Nothing here is a premise of another file. *)

From Stdlib Require Import List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaEquivarianceSpec
  FlambdaBindersSpec FlambdaHeapSpec FlambdaExitWorldSpec FlambdaRunUnderSpec
  FlambdaObserversSpec FlambdaExitWorldActionsSpec FlambdaNoArrayValueSpec.
Import ListNotations.

(* ======================================================================== *)
(* The observers that make no array                                         *)
(* ======================================================================== *)

(* An observer that makes no array: a way out, or a handler that is an
   observer and in which the primitive that makes an array does not
   occur. For some value slots it is a class of what a continuation may
   stand for, in the sense of FlambdaObserversSpec.v. *)
Definition observer_free_of_arrays (slots : list value_slot)
    (c : cont_value) : Prop :=
  observer slots c ∧ cont_free_of_arrays c.

(* A continuation environment every entry of which is in the class, the
   ones that an earlier entry hides included, is an environment of
   observers that is free of arrays. *)
Definition observers_free_of_arrays_statement : Prop :=
  ∀ (slots : list value_slot) (conts : cont_env),
    Forall (fun kc => observer_free_of_arrays slots (snd kc)) conts
    <-> observers slots conts ∧ conts_free_of_arrays conts.

(* [FlambdaSemanticsSpec.expr_refines] restricted to the whole source
   states that are free of arrays. As in
   FlambdaNoArrayValueSpec.heap_expr_refines_free_of_arrays, the
   condition is on the source expression with its environments and its
   memory, so it has [makes_array source = false] in it: of a source
   that makes an array it holds of no state, so such a source is
   refined by every target in this sense, and that tells nothing. The
   condition says nothing of the target. *)
Definition expr_refines_free_of_arrays (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    state_free_of_arrays (State source vars conts statics hp rd) ->
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

(* ======================================================================== *)
(* A statement that concludes one of the two gives what it gave             *)
(* ======================================================================== *)

Definition worlds_observed_gives_free_of_arrays_statement : Prop :=
  ∀ (slots : list value_slot) (w w' : exit_world),
    worlds_observed slots w w' ->
    worlds_observed_by (observer_free_of_arrays slots) w w'.

Definition refines_under_observers_gives_free_of_arrays_statement : Prop :=
  ∀ (slots : list value_slot) (statics : static_env) (hp : heap)
    (rd : read_record)
    (source target : expr),
    refines_under_observers slots statics hp rd source target ->
    refines_observed_by (observer_free_of_arrays slots) statics hp rd source
      target.

(* ======================================================================== *)
(* The statements of FlambdaObserversSpec.v                                 *)
(* ======================================================================== *)

(* FlambdaObserversSpec.worlds_observed_way_out_statement: the way out
   makes no array. *)
Definition worlds_observed_free_of_arrays_way_out_statement : Prop :=
  ∀ (slots : list value_slot) (w w' : exit_world),
    worlds_observed_by (observer_free_of_arrays slots) w w' ->
    exit_cont w = exit_cont w'
    ∧ same_scalars (exit_values w) (exit_values w')
    ∧ map fst (exit_heap w) = map fst (exit_heap w').

(* FlambdaObserversSpec.expr_refines_under_observers_statement, from a
   static environment and a heap that are free of arrays, for a source
   that makes no array. *)
Definition expr_refines_free_of_arrays_under_observers_statement : Prop :=
  ∀ (slots : list value_slot) (statics : static_env) (hp : heap)
    (rd : read_record)
    (source target : expr),
    makes_array source = false ->
    code_makes_no_array statics ->
    statics_no_array_value statics ->
    heap_no_array_value hp ->
    expr_refines_free_of_arrays source target ->
    refines_observed_by (observer_free_of_arrays slots) statics hp rd source
      target.

(* FlambdaObserversSpec.refines_under_observers_trans_statement. *)
Definition refines_under_observers_free_of_arrays_trans_statement : Prop :=
  ∀ (slots : list value_slot) (statics : static_env) (hp : heap)
    (rd : read_record)
    (e1 e2 e3 : expr),
    refines_observed_by (observer_free_of_arrays slots) statics hp rd e1 e2 ->
    refines_observed_by (observer_free_of_arrays slots) statics hp rd e2 e3 ->
    refines_observed_by (observer_free_of_arrays slots) statics hp rd e1 e3.

(* FlambdaObserversSpec.under_observers_worlds_statement: the two forms
   say the same. *)
Definition under_observers_free_of_arrays_worlds_statement : Prop :=
  ∀ (slots : list value_slot) (statics : static_env) (hp : heap)
    (rd : read_record)
    (source target : expr),
    code_closed source = true ->
    code_closed target = true ->
    static_code_closed statics ->
    (refines_observed_by (observer_free_of_arrays slots) statics hp rd source
       target
     <-> refines_world (worlds_observed_by (observer_free_of_arrays slots))
           (State source [] [] statics hp rd)
           (State target [] [] statics hp rd)).

(* ======================================================================== *)
(* The statements of FlambdaObserversActionsSpec.v                          *)
(* ======================================================================== *)

(* A renaming brings the primitive neither in nor out
   (FlambdaNoArrayValueSpec.makes_array_renamed_statement), so an
   observer makes no array exactly when it makes none renamed. *)

(* FlambdaObserversActionsSpec.worlds_observed_equivariant_statement. *)
Definition worlds_observed_free_of_arrays_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (w w' : exit_world),
    worlds_observed_by (observer_free_of_arrays slots)
      (rename_exit_world r w) (rename_exit_world r w')
    <-> worlds_observed_by (observer_free_of_arrays slots) w w'.

(* FlambdaObserversActionsSpec
   .refines_world_observed_equivariant_statement. *)
Definition refines_world_observed_free_of_arrays_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (source target : state),
    refines_world (worlds_observed_by (observer_free_of_arrays slots))
      (rename_state r source) (rename_state r target)
    <-> refines_world (worlds_observed_by (observer_free_of_arrays slots))
          source target.

(* FlambdaObserversActionsSpec
   .refines_under_observers_equivariant_statement. *)
Definition refines_observed_free_of_arrays_equivariant_statement : Prop :=
  ∀ (r : renaming) (slots : list value_slot) (statics : static_env)
    (hp : heap) (rd : read_record) (source target : expr),
    refines_observed_by (observer_free_of_arrays slots)
      (rename_static_env r statics) hp rd (rename_expr r source)
      (rename_expr r target)
    <-> refines_observed_by (observer_free_of_arrays slots) statics hp rd source
          target.
