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


(* The validation asked of the term before with its known arrays read.

   This file is to be reviewed: [read_arrays_in_code], [read_arrays],
   [normalise_before_avoiding], [reading_round], [normalise_reading],
   [before_read], [validate_unit_reading] and
   [validate_unit_knowing_reading], which are extracted and run by
   the validation of middle_end/flambda2/tests/semantics/; and the
   statements.

   Why it is not in the rounds of the validation.
   FlambdaArrayLoadSpec.load_known_arrays makes a constant of the
   length of an immutable array that the term makes or binds to a
   symbol, and the element of a load from one at a constant index. It
   refines the term and is not an equivalence, so it is for the term
   before only. Simplify does the same where it knows the array, and
   it does not know it where the array is made by one function and
   read by another and neither call is inlined. Told to leave the
   calls of the code that the term after calls, the validation
   compares such a unit with the calls in place on both sides. Told
   to leave the calls of recursive code only, which it is for a unit
   in which Simplify inlines a function at one call and not at
   another, it inlines the calls of the two functions on both sides:
   the term before then has the array and its read in one piece of
   code and the term after has them too, as primitives. With the fold
   in the rounds the term before would have the constant there and
   the term after the primitive, which no rewrite of the term after
   that is an equivalence makes the constant in general, and a unit
   that FlambdaValidateSpec.validate_unit accepts would be refused.

   So the rounds of FlambdaValidateSpec.validate_unit do not have the
   fold, and it is here, in a function beside:

   - [read_arrays] is the fold in the term, in the handlers of its
     recursive continuations and in the bodies of its code;
   - [normalise_reading] is the term before after rounds of
     [reading_round]. A round makes the loops of the term, as
     FlambdaValidateSpec.validate_unit makes them first, normalises
     it as that function normalises the term before, and reads the
     arrays of what that gives: a read may be of an array that only
     the inlining of a call brings beside it; and the round after
     normalises what was read, as a constant that a read makes may
     let a call be inlined. That normalisation is
     [normalise_before_avoiding]: FlambdaValidateSpec.normalise_before
     with its rounds given continuations that the names they make are
     not to be ([normalise_before_avoiding_nothing_statement]: with
     none given it is that function). A round that reads nothing
     gives the term back as it was, not normalised: so of a term in
     which the first round reads nothing, [normalise_reading] is the
     term itself ([normalise_reading_nothing_read_statement]), and
     the two functions below are then the functions that they ask,
     of the two terms as they are given. What is accepted here and
     not there is accepted with a read;
   - [validate_unit_reading] is FlambdaValidateSpec.validate_unit, of
     that term and of the term after, and
     [validate_unit_knowing_reading] is
     FlambdaValidateKnowingSpec.validate_unit_knowing of the same.
     The continuations given to the rounds are those of the units of
     the liftings and those of the loops, which the function asked
     reads and the term need not mention, as the second asking gives
     them to its own rounds.

   What they accept is correct for the unit as what the two functions
   accept is: [normalise_reading] refines the term under all
   environments ([normalise_reading_refines_statement]), so a term
   after that refines what it gives refines the term before. The
   statements are of the unit run on its own from empty environments
   and of the unit run from its imports; the second is given the test
   of FlambdaValidateImportsSpec.v on the term that the function asked
   was given for the term before ([before_read]).

   A caller asks FlambdaValidateSpec.validate_unit first and this
   only of a unit that it does not validate: nothing that the first
   accepts is lost to the fold, and what only the fold makes the same
   is accepted here.

   The proofs are in FlambdaValidateReadingProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaCompareSpec FlambdaNormaliseSpec FlambdaRecHandlersSpec
  FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaStubCallsSpec FlambdaNeverReturnsSpec
  FlambdaLiftSpec FlambdaGuardScopeSpec FlambdaArrayLoadSpec
  FlambdaValidateSpec FlambdaValidateKnowingSpec FlambdaImportsSpec
  FlambdaValidateImportsSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The arrays read                                                          *)
(* ======================================================================== *)

(* The fold in the body of code: in the body, and in the handler of
   each of its recursive continuations, a loop in a loop included,
   with nothing known at the start of a handler. *)
Definition read_arrays_in_code (body : expr) : expr :=
  rewrite_rec_handlers load_known_arrays (load_known_arrays body).

(* The fold in a unit: in the term itself, and in the body of each
   piece of code. *)
Definition read_arrays (e : expr) : expr :=
  rewrite_code read_arrays_in_code (read_arrays_in_code e).

(* FlambdaValidateSpec.normalise_before, with the rounds of its second
   phase given the continuations [avoid]: no name that they make is
   one of those. *)
Definition normalise_before_avoiding (avoid : list continuation)
    (rounds : nat) (kept : list code_id) (e : expr) : expr :=
  demote_and_tidy rounds
    (iterate rounds (normalise_before_round_avoiding avoid kept)
       (guard_scope FlambdaSimplifyCarriedSpec.simplify_by_types_carried
          (iterate rounds (normalise_before_round_avoiding avoid kept)
             (iterate rounds normalise_params_round e)))).

(* One round of the reading: the loops of the term made, as
   FlambdaValidateSpec.validate_unit makes them first; the term
   normalised, with [avoid] for the names that the normalisation
   makes; and the arrays read. Where nothing is read (the comparison
   up to bound names, FlambdaCompareSpec.expr_eqb, finds what is read
   the same as the term normalised), the term is given back as it
   was: without a read the normalisation is not kept.

   The three steps that make the loops are written here as
   FlambdaValidateSpec.validate_unit has them at its head, and no
   statement holds the two together: if those of [validate_unit]
   change, these are to be changed with them. *)
Definition reading_round (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) (e : expr) : expr :=
  let normalised :=
    normalise_before_avoiding avoid rounds kept
      (loopify_unit G (cut_returns [] (demote_stub_calls e)))
  in
  let read := read_arrays normalised in
  if expr_eqb [] [] normalised read then e else read.

(* The term before as it is given to the validation here: rounds of
   the reading, until one leaves the term as it was. *)
Definition normalise_reading (rounds : nat) (avoid : list continuation)
    (G : loopified) (kept : list code_id) (e : expr) : expr :=
  iterate rounds (reading_round avoid rounds G kept) e.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The term before as the two functions below give it to the function
   that they ask. The continuations that the rounds of
   [normalise_reading] are given to avoid are those of the units of
   the liftings and those given for the loops, as in
   FlambdaValidateKnowingSpec.validate_unit_knowing. *)
Definition before_read (rounds : nat) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before : expr) : expr :=
  normalise_reading rounds (unit_continuations Ls ++ map snd G) G kept
    before.

Definition validate_unit_reading (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit rounds unused slots G Ls kept
    (before_read rounds G Ls kept before) after.

(* The same of the second asking. *)
Definition validate_unit_knowing_reading (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) : validation :=
  validate_unit_knowing rounds unused slots G Ls kept
    (before_read rounds G Ls kept before) after.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What is read refines the term, under all environments. *)
Definition read_arrays_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (read_arrays e).

(* With no continuation to avoid, the normalisation of the rounds is
   that of FlambdaValidateSpec.validate_unit. *)
Definition normalise_before_avoiding_nothing_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id) (e : expr),
    normalise_before_avoiding [] rounds kept e
    = normalise_before rounds kept e.

Definition normalise_reading_refines_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (G : loopified)
    (kept : list code_id) (e : expr),
    expr_refines e (normalise_reading rounds avoid G kept e).

(* Where a round gives a term back as it was, which it does where it
   reads nothing, the rounds give that term. *)
Definition normalise_reading_nothing_read_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (G : loopified)
    (kept : list code_id) (e : expr),
    reading_round avoid rounds G kept e = e ->
    normalise_reading rounds avoid G kept e = e.

(* A unit that is validated here refines the unit it came from, run on
   its own from empty environments. *)
Definition validate_unit_reading_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_reading rounds unused slots G Ls kept before after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_knowing_reading_sound_statement : Prop :=
  ∀ (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    validate_unit_knowing_reading rounds unused slots G Ls kept before
      after
    <> Not_validated ->
    program_refines before after.

(* A unit whose verdict here the test of FlambdaValidateImportsSpec.v
   accepts at the static environment of its imports, of the term that
   the function asked was given for the term before ([before_read]) and
   of the term after, refines the unit it came from, each run on its
   own from those imports. *)
Definition validate_unit_reading_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls
      kept (before_read rounds G Ls kept before) after
      (validate_unit_reading rounds unused slots G Ls kept before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_reading_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    verdict_knowing_holds_from (imports_statics imports) rounds unused
      slots G Ls kept (before_read rounds G Ls kept before) after
      (validate_unit_knowing_reading rounds unused slots G Ls kept before
         after)
    = true ->
    unit_refines imports before after.
