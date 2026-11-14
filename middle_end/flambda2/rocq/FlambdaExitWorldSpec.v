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

(* The world that a run leaves at an exit, and refinement with the worlds
   of two exits related.

   [FlambdaSemanticsSpec.refines] compares two exits by their
   continuation, their values and their heap, each of which must be the
   same. It does not look at the static environment that a run has built
   when it exits: a behaviour does not carry it, and two runs that
   [refines] relates need not build the same (a static binding that is
   not used may be removed, and code is rewritten).

   A statement that relates two exits otherwise than by equality needs
   that static environment: where a value of the second exit is a symbol
   that stands for a value of the first, what the symbol is bound to is
   in the static environment of the second run. Such a statement is
   that of a pass that changes values
   (FlambdaLiftSpec.lift_simulates_statement, which takes the last state
   of the second run by hand). This file gives the form once:
   - [exit_world], what a run leaves at an exit: the continuation, the
     values, the heap and the read record that the exit carries, and
     the static environment of the state that takes the last step;
   - [exits_world], that a state exits and leaves a world. The witness
     is a run of that state, so the static environment is that of a
     state of this run and of no other term;
   - [refines_world R], which is [refines] with the worlds of the two
     exits related by [R];
   - [same_exit], the relation at which it is [refines]: the static
     environments are not compared
     ([refines_world_same_statement]).
   Nothing here changes a statement that exists; no relation between
   worlds other than [same_exit] is defined here.

   What is proved is in FlambdaExitWorldProofs.v: every [_statement]
   here. *)

From Stdlib Require Import List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What a run leaves at an exit                                             *)
(* ======================================================================== *)

(* The continuation named by the exit; the values given to it; the
   heap; the read record; and the static environment of the state that
   takes the last step. The first three are those of
   [FlambdaSemanticsSpec.Exits], which forgets the read record that
   [FlambdaSemanticsSpec.Exit] carries. *)
Record exit_world : Type := Exit_world {
  exit_cont : continuation;
  exit_values : list value;
  exit_statics : static_env;
  exit_heap : heap;
  exit_reads : read_record
}.

(* The state exits and leaves this world: some run of it reaches a state
   whose step is the exit, and the static environment of the world is
   that of this state. *)
Definition exits_world (s : state) (w : exit_world) : Prop :=
  ∃ (n : nat) (last : state),
    run n s = Step last
    ∧ step last
      = Exit (exit_cont w) (exit_values w) (exit_heap w) (exit_reads w)
    ∧ state_statics last = exit_statics w.

(* A state exits with some values and heap exactly when it leaves a
   world that has them. *)
Definition exits_world_behaves_statement : Prop :=
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap),
    behaves s (Exits k vs hp)
    <-> ∃ (statics : static_env) (rd : read_record),
          exits_world s (Exit_world k vs statics hp rd).

(* A state leaves at most one world. *)
Definition exits_world_unique_statement : Prop :=
  ∀ (s : state) (w w' : exit_world),
    exits_world s w -> exits_world s w' -> w = w'.

(* ======================================================================== *)
(* Refinement with the worlds of two exits related                          *)
(* ======================================================================== *)

(* [FlambdaSemanticsSpec.refines], with the worlds of the two exits
   related by [R] where [refines] asks for the same continuation, values
   and heap. An exit of the source is answered by an exit of the target
   whose world [R] relates to that of the source. And the target ends,
   by an exit or by going wrong, only if the source exits or goes
   wrong: so a source that diverges has a target that does not end.
   Nothing is said of a source that goes wrong. The second part does
   not say that the two end alike; with the first part and
   [FlambdaSemanticsSpec.behaviour_unique_statement], a source that
   exits has a target that exits, with a related world. *)
Definition refines_world (R : exit_world -> exit_world -> Prop)
    (source target : state) : Prop :=
  (∀ w : exit_world,
    exits_world source w ->
    ∃ w' : exit_world, exits_world target w' ∧ R w w')
  ∧ (∀ b : behaviour,
      b <> Diverges ->
      behaves target b ->
      behaves source Goes_wrong ∨ ∃ w : exit_world, exits_world source w).

(* The relation at which it is [refines]: the same continuation, values
   and heap. Neither the static environments nor the read records are
   compared, as [refines] compares neither. *)
Definition same_exit (w w' : exit_world) : Prop :=
  exit_cont w = exit_cont w'
  ∧ exit_values w = exit_values w'
  ∧ exit_heap w = exit_heap w'.

Definition refines_world_same_statement : Prop :=
  ∀ source target : state,
    refines_world same_exit source target <-> refines source target.

(* Two of them compose, at the composite of the two relations. *)
Definition refines_world_trans_statement : Prop :=
  ∀ (R1 R2 : exit_world -> exit_world -> Prop) (s1 s2 s3 : state),
    refines_world R1 s1 s2 ->
    refines_world R2 s2 s3 ->
    refines_world (fun w w'' => ∃ w' : exit_world, R1 w w' ∧ R2 w' w'')
      s1 s3.

(* A relation may be made larger. *)
Definition refines_world_weaken_statement : Prop :=
  ∀ (R R' : exit_world -> exit_world -> Prop) (source target : state),
    (∀ w w' : exit_world, R w w' -> R' w w') ->
    refines_world R source target ->
    refines_world R' source target.
