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

(* A second check of a unit against the unit it came from, for the
   units whose exits a lifting changes.

   This file awaits the human's confirmation (decision D43). What it
   proves of the check is not [refines]; whether it is what a verdict
   is to mean is the human's to decide, between this form and a
   structural one (below, "What that is, and is not"). The test tool
   asks the check only of a unit that it reports as not validated, and
   prints what the check accepts on a line of its own, which has not
   the word "validated": such a unit stays not validated, and is in no
   count of the units validated.

   This file is to be reviewed: the definitions and the statements.

   FlambdaValidateSpec.validate_unit answers [Same_once_lifted] of a
   unit only where each lifting passes its test of the exits
   (FlambdaLiftSpec.exits_plain_knowing): a lifting gives a symbol where the
   unit before gave the value that it lifts
   (FlambdaLiftSpec.lift_named), and FlambdaSemanticsSpec.refines asks
   of two exits the same values. A unit that exits with something that
   Simplify lifted is so refused.

   The check here follows the two comparisons of lifted terms of
   [validate_unit], with the liftings made without that test
   (FlambdaLiftObservedSpec.with_lifting_observed), and concludes
   another relation: not [refines], but that an exit of the unit
   before is answered by an exit of the unit after that nothing which
   may follow a unit tells apart from it
   (FlambdaObserversSpec.worlds_observed), and that the unit after
   ends only if the unit before exits or goes wrong. Of a unit before
   that goes wrong it says nothing. Neither relation is said to give
   the other: [refines] does not compare the static environments that
   two exits leave, under which what may follow is run here; and two
   exits related here need not have the same values, nor heaps that
   hold the same.

   What that is, and is not. What may follow a unit
   (FlambdaObserversSpec.observer) is a way out, or a handler with no
   environment of its own, which binds nothing static and names no
   symbol and no code id, and which projects none of the value slots
   that the check is given to remove; it may read, store, allocate,
   project and call. The two statements are over that whole class,
   [observer slots] at the value slots given, and over no smaller
   one; a handler that makes an immutable array is in it. Their
   relations are those of FlambdaObserversSpec.v over
   every observer, which are the relations of a class at that class
   (FlambdaObserversSpec.worlds_observed_by_observer_statement,
   FlambdaObserversSpec.refines_observed_by_observer_statement). With
   a term more in the language there are observers more, and the two
   statements then conclude more. So nothing is said here of two
   units that are linked: a unit that follows another names its
   symbols. Nor does what may follow name an import: of two exits
   that each carry an imported symbol, a predefined exception for
   one, alone or inside a block, it is not said that it is the same
   symbol. Two exits so related have
   the same constants and the same locations among their values, and
   heaps that bind the same locations
   (FlambdaObserversSpec.worlds_observed_way_out_statement); of an
   allocated value at an exit, only what such a handler can come to
   observe of it is compared. Whether this is what a verdict on a
   pass that changes values is to mean has not been decided: the
   other form would relate the values of the two exits and every
   entry of the two memories structurally. The verdict has a type of
   its own ([related_validation]) and is no [validation]. The check
   does not ask what [validate_unit] answers of the two terms: it is
   the tool that asks it only of a unit not validated.

   The route. The unit before is prepared as [validate_unit] prepares
   it; each lifting given is applied without the test of the exits
   and followed by FlambdaValidateSpec.normalise_lifted_own (the
   rounds of [normalise_lifted], with the calls of the closures that
   a [Let] makes made direct in them where the lifting says so, and
   with what [my_closure] is known to be from its code used in them
   where the lifting says so), and
   after the last the known value slots are read and what is known is
   used after the demotion, as there ([lifted_in_turn_observed]); the
   calls are taken on and the term tidied as there; the value slots
   given and then the static bindings given are removed
   (FlambdaValidateSpec.without_slots,
   FlambdaValidateSpec.without_statics). The unit after is normalised
   and tidied and the unused static bindings of its head are removed,
   as there. The two are then compared as there, twice: with the code
   of the unit after bound first, up to the order of the static
   bindings and the place of the code; and, if that fails, with the
   static bindings of both that need nothing from around them bound
   first (FlambdaValidateSpec.with_statics_first).

   The unit is run from the static environment of its imports
   (FlambdaImportsSpec.imports_statics). The check computes what the
   laws of the steps ask of that environment, as
   FlambdaValidateImportsSpec.lifted_holds_from does for
   [validate_unit]: that it binds no name that a lifting or the use
   of what is known binds ([lifted_observed_free_in]), nor a name of
   the head of the unit after where bindings are removed from it
   (FlambdaValidateImportsSpec.head_removal_free_in), nor, for the
   second comparison, a name that either term compared binds
   (FlambdaValidateImportsSpec.statics_first_free_in). It also asks
   that the code of the two units be closed
   (FlambdaBindersSpec.code_closed): the passage from the form in
   which the steps are proved to the statement about the two units
   run on their own needs it
   (FlambdaObserversSpec.under_observers_worlds_statement).

   The proofs are in FlambdaValidateRelatedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaBindersSpec FlambdaCompareSpec FlambdaNormaliseSpec
  FlambdaStaticsOrderSpec FlambdaUnusedStaticsSpec
  FlambdaUnusedHeadStaticsSpec FlambdaUnusedSlotsSpec FlambdaLoopifySpec
  FlambdaLiftSpec FlambdaKnownSlotsSpec
  FlambdaFloatCodeSpec FlambdaStubCallsSpec FlambdaNeverReturnsSpec
  FlambdaValidateSpec FlambdaImportsSpec FlambdaValidateImportsSpec
  FlambdaExitWorldSpec FlambdaRunUnderSpec FlambdaObserversSpec
  FlambdaLiftObservedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The lifted term                                                          *)
(* ======================================================================== *)

(* FlambdaValidateSpec.lifted_in_turn with each lifting made without
   the test of the exits. What ends it is what ends that one. *)
Fixpoint lifted_in_turn_observed (rounds : nat) (avoid : list continuation)
    (Ls : list lifting) (kept : list code_id) (e : expr) : expr :=
  match Ls with
  | [] => demote_and_tidy_known rounds (read_known_slots_anywhere e)
  | L :: Ls =>
    lifted_in_turn_observed rounds avoid Ls kept
      (normalise_lifted_own (made_closures_called L) (own_closures_known L)
         rounds avoid kept (with_lifting_observed L e))
  end.

(* What FlambdaLiftObservedSpec.with_lifting_observed_refines_statement
   asks of the static environment: it binds no name that the term
   binds after the first, the second and the third lifting, nor one
   that the result binds. *)
Definition lifting_observed_free_in (statics : static_env) (L : lifting)
    (e : expr) : bool :=
  let at_head := to_head_observed L e in
  let out_of_code := out_of_code_observed L at_head in
  let lifted := lets_observed L out_of_code in
  static_names_free_in statics at_head
  && static_names_free_in statics out_of_code
  && static_names_free_in statics lifted
  && static_names_free_in statics (with_lifting_observed L e).

(* That, and what the rounds that follow ask
   (FlambdaValidateImportsSpec.normalise_lifted_own_free_in), of
   each of the liftings as [lifted_in_turn_observed] applies them; and
   after the last, what the use of what is known after the demotion
   asks (FlambdaValidateImportsSpec.demote_and_tidy_known_free_in), of
   the term with its known value slots read. *)
Fixpoint lifted_observed_free_in (statics : static_env) (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id)
    (e : expr) : bool :=
  match Ls with
  | [] =>
    demote_and_tidy_known_free_in statics rounds (read_known_slots_anywhere e)
  | L :: Ls =>
    lifting_observed_free_in statics L e
    && normalise_lifted_own_free_in statics (made_closures_called L)
         (own_closures_known L) rounds avoid kept
         (with_lifting_observed L e)
    && lifted_observed_free_in statics rounds avoid Ls kept
         (normalise_lifted_own (made_closures_called L)
            (own_closures_known L) rounds avoid kept
            (with_lifting_observed L e))
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

Inductive related_validation : Type :=
  | Related_once_lifted
  | Not_related.

Definition validate_unit_related (imports : list (symbol * import_class))
    (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr) : related_validation :=
  let statics := imports_statics imports in
  let source := loopify_unit G (cut_returns [] (demote_stub_calls before)) in
  let tidied := tidied_after rounds (normalise_after rounds kept after) in
  let taken := negb (keeps_a_return tidied) in
  let target := without_head_statics unused tidied in
  let lifted :=
    without_statics unused
      (without_slots slots
         (tidied_before rounds
            (with_returns taken
               (lifted_in_turn_observed rounds (unit_continuations Ls) Ls
                  kept source))))
  in
  if
    code_closed before
    && code_closed after
    && head_removal_free_in statics unused tidied
    && lifted_observed_free_in statics rounds (unit_continuations Ls) Ls kept
         source
    && (expr_eqb [] []
          (order_statics_as (float_code (with_code_first target))
             (float_code lifted))
          (float_code (with_code_first target))
        || statics_first_free_in statics Ls target
           && statics_first_free_in statics Ls lifted
           && expr_eqb [] []
                (order_statics_as (with_statics_first Ls target)
                   (with_statics_first Ls lifted))
                (with_statics_first Ls target))
  then Related_once_lifted
  else Not_related.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* In the form in which the steps are proved: the two units, run from
   one continuation environment of what may follow a unit and from the
   static environment of the imports, come to results that differ in
   nothing that an exit is compared by. What may follow projects none
   of the value slots given: the class is [observer slots], every
   observer of those slots. *)
Definition validate_unit_related_under_observers_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    validate_unit_related imports rounds unused slots G Ls kept before after
    = Related_once_lifted ->
    refines_under_observers slots (imports_statics imports) empty_heap
      before after.

(* Of the two units run on their own from their imports: an exit of
   the unit before is answered by an exit of the unit after that
   nothing which may follow a unit tells apart from it, and the unit
   after ends only if the unit before exits or goes wrong. The class
   is the same, [observer slots]. *)
Definition validate_unit_related_sound_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    validate_unit_related imports rounds unused slots G Ls kept before after
    = Related_once_lifted ->
    refines_world (worlds_observed slots)
      (State before [] [] (imports_statics imports) empty_heap)
      (State after [] [] (imports_statics imports) empty_heap).
