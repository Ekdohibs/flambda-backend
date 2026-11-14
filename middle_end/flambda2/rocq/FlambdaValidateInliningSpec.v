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


(* The validation asked of the term before with chosen calls inlined
   first.

   This file is to be reviewed: [before_inlined],
   [validate_unit_inlining] and
   [validate_unit_knowing_inlining], which are extracted and run
   by the validation of middle_end/flambda2/tests/semantics/; and the
   statements.

   Why it is not in the rounds of the validation.
   FlambdaValidateSpec.validate_unit is told the code whose calls its
   rounds leave ([kept]), one list for the two terms and for every
   round: the choice is by code. Simplify chooses by call
   (FlambdaInlineCallsAtSpec.v says how), and of a unit in which it
   has inlined one call of a piece of code and left another it can
   be that no list makes the two terms come to the same. A choice
   by place cannot be given to the rounds as they are: each round
   renames the term and moves its parts, so that a place of the
   term as it is given is not a place of the term that a round is
   given.

   So the choice is made before the rounds, in a function beside:

   - [inlined_in_turn] is a term with the calls of each choice in
     turn rewritten by FlambdaInlineCallsAtSpec.inline_calls_at. A
     choice is the places of the calls outside code and the places of
     the calls in the body of each piece of code, in the term as that
     application is given it: a call that the choice before brought
     into the term is at a place of the term that results. After each
     application the term is renamed apart
     (FlambdaRenameApartSpec.rename_apart_avoiding), as a round of
     the validation renames it once its calls are inlined
     (FlambdaValidateSpec.normalise_before_round_avoiding). A body
     that is inlined binds the parameters of its code, and
     FlambdaNormaliseSpec.inline_apply leaves a call an argument of
     which is a parameter of the code called: a call of the same code
     that the body brings, which passes them, is rewritten by the
     next choice only once they are renamed. The renaming is given
     continuations that the names it makes are not to be;
   - [before_inlined] is that of the term before, the continuations
     given being those of the units of the liftings and those of the
     loops, which the function asked reads and the term need not
     mention, as FlambdaValidateReadingSpec.before_read gives them to
     its rounds;
   - [validate_unit_inlining] is
     FlambdaValidateSpec.validate_unit, of that term and of the term
     after, and [validate_unit_knowing_inlining] is
     FlambdaValidateKnowingSpec.validate_unit_knowing of the same.

   The choices are a guess of the caller and are not trusted. What
   the two functions accept is correct for the unit whatever they
   are, as what the functions that they ask accept is:
   [inlined_in_turn] refines the term under all environments
   ([inlined_in_turn_refines_statement]), so a term after that
   refines what [before_inlined] gives refines the term before. With
   no choice that term is the term before, and the two functions are
   the functions that they ask.

   The statements are of the unit run on its own from empty
   environments and of the unit run from its imports; the second is
   given the test of FlambdaValidateImportsSpec.v on the term that
   the function asked was given for the term before
   ([before_inlined]).

   A caller asks FlambdaValidateSpec.validate_unit first and this
   only of a unit that it does not validate: nothing that the first
   accepts is asked again, and what only an inlining by place makes
   the same is accepted here.

   The proofs are in FlambdaValidateInliningProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaLiftSpec
  FlambdaRenameApartSpec FlambdaInlineCallsAtSpec FlambdaValidateSpec
  FlambdaValidateKnowingSpec FlambdaImportsSpec
  FlambdaValidateImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The calls inlined                                                        *)
(* ======================================================================== *)

(* A choice: the places of the calls to rewrite outside code, and
   those in the body of each piece of code. *)
Definition inlining_choice : Type := (list path * calls_in_code)%type.

(* For each choice in turn, the calls at its places rewritten and the
   term renamed apart, no name that the renaming makes being one of
   the continuations [avoid]. *)
Fixpoint inlined_in_turn (avoid : list continuation)
    (choices : list inlining_choice) (e : expr) : expr :=
  match choices with
  | [] => e
  | (sites, K) :: choices =>
    inlined_in_turn avoid choices
      (rename_apart_avoiding avoid (inline_calls_at sites K e))
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The term before as the two functions below give it to the function
   that they ask. The continuations that the renamings of
   [inlined_in_turn] are given to avoid are those of the units of the
   liftings and those given for the loops, as in
   FlambdaValidateReadingSpec.before_read. *)
Definition before_inlined (choices : list inlining_choice) (G : loopified)
    (Ls : list lifting) (before : expr) : expr :=
  inlined_in_turn (unit_continuations Ls ++ map snd G) choices before.

Definition validate_unit_inlining (choices : list inlining_choice)
    (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr) : validation :=
  validate_unit rounds unused slots G Ls kept
    (before_inlined choices G Ls before) after.

(* The same of the second asking. *)
Definition validate_unit_knowing_inlining (choices : list inlining_choice)
    (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr) : validation :=
  validate_unit_knowing rounds unused slots G Ls kept
    (before_inlined choices G Ls before) after.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The term with the calls of the choices inlined refines the term,
   under all environments. *)
Definition inlined_in_turn_refines_statement : Prop :=
  ∀ (avoid : list continuation) (choices : list inlining_choice)
    (e : expr),
    expr_refines e (inlined_in_turn avoid choices e).

(* A unit that is validated here refines the unit it came from, run on
   its own from empty environments. *)
Definition validate_unit_inlining_sound_statement : Prop :=
  ∀ (choices : list inlining_choice) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    validate_unit_inlining choices rounds unused slots G Ls kept
      before after
    <> Not_validated ->
    program_refines before after.

Definition validate_unit_knowing_inlining_sound_statement : Prop :=
  ∀ (choices : list inlining_choice) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    validate_unit_knowing_inlining choices rounds unused slots G Ls
      kept before after
    <> Not_validated ->
    program_refines before after.

(* A unit whose verdict here the test of FlambdaValidateImportsSpec.v
   accepts at the static environment of its imports, of the term that
   the function asked was given for the term before ([before_inlined])
   and of the term after, refines the unit it came from, each run on
   its own from those imports. *)
Definition validate_unit_inlining_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class))
    (choices : list inlining_choice) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    verdict_holds_from (imports_statics imports) rounds unused slots G Ls
      kept (before_inlined choices G Ls before) after
      (validate_unit_inlining choices rounds unused slots G Ls kept
         before after)
    = true ->
    unit_refines imports before after.

(* The same of the second asking. *)
Definition validate_unit_knowing_inlining_verdict_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class))
    (choices : list inlining_choice) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    verdict_knowing_holds_from (imports_statics imports) rounds unused
      slots G Ls kept (before_inlined choices G Ls before) after
      (validate_unit_knowing_inlining choices rounds unused slots G
         Ls kept before after)
    = true ->
    unit_refines imports before after.
