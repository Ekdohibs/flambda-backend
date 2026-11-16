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


(* The validation asked of the term before with its known arrays read,
   under renaming.

   This file is to be reviewed: the statements. It has no definition.

   FlambdaValidateReadingSpec.v has a function beside the validation:
   the term before goes through rounds that make its loops, normalise
   it and read its known arrays ([normalise_reading], of rounds of
   [reading_round]), and the validation is asked of what they give
   ([validate_unit_reading], [validate_unit_knowing_reading]). That
   FlambdaValidateSpec.validate_unit and the second asking answer the
   same of a pair of terms and of the pair renamed is
   FlambdaValidatorEquivarianceSpec.validate_unit_equivariant_statement
   and
   FlambdaValidateKnowingSpec.validate_unit_knowing_equivariant_statement.
   This file states the same of the two functions of that file, and,
   on the way, of what they are made of. The statements are about the
   functions as they are defined, and not about what is proved of a
   pair that they accept.

   What a renaming is here: FlambdaNominalSpec.renaming, which acts on
   variables and on continuations. It moves no symbol and no code id.
   The liftings and the continuations given for the loops hold
   continuations, and are renamed with the terms
   (FlambdaValidatorEquivarianceSpec.rename_lifting,
   [rename_loopified]).

   The forms are those of FlambdaValidatorEquivarianceSpec.v.
   - The read ([read_arrays]) makes no name: it commutes with
     renaming, as an equation, and it keeps scope.
   - The normalisation of a round ([normalise_before_avoiding]) has
     the rounds of the validation, one of whose rewrites makes names.
     So it commutes with renaming up to the names that it makes, the
     second renaming doing to the continuations that it is given to
     avoid what the first does ([equivariant_up_to_avoiding]); and it
     keeps scope.
   - A round ([reading_round]) is that normalisation of the term with
     its loops made, then the read, and the term as it was where the
     comparison finds that nothing was read. The comparison is of two
     terms that are, for a renamed term, under the same second
     renaming, so it answers the same and the same branch is taken.
     The loops are given by continuations that the term need not
     mention; they are renamed with it.
   - The rounds ([normalise_reading]) are given the same loops at
     every round, and the second round is asked of a term that is
     under the second renaming of the first. So the statement asks
     that the continuations of the loops are among those to avoid:
     the second renaming then does to them what the first does. The
     function as the checks call it ([before_read]) gives the rounds
     the continuations of the units of the liftings and those of the
     loops to avoid, so it has the law with no such premise.
   - The two checks then answer the same of a pair and of the pair
     renamed, where the code of the term before is closed: the term
     that each gives to the function that it asks is, for a renamed
     pair, that term under a second renaming that does to its free
     names, and to the continuations of the liftings and of the loops,
     what the first does, which is what
     FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement
     asks.
   - The tests from the imports. A caller that has a unit with imports
     asks, of the verdict that a check gives, whether it holds of the
     unit run from the static environment of its imports
     (FlambdaValidateImportsSpec.verdict_holds_from and
     [verdict_knowing_holds_from]), and asks it of the term that
     [before_read] gives, as the statements of
     FlambdaValidateReadingSpec.v with the imports have it. Each test,
     of the verdict that its check gives, answers the same of a pair
     and of the pair renamed, where the code of the term before is
     closed: the form of the two laws of
     FlambdaValidateImportsRenameSpec.v for [validate_unit] and for
     the second asking
     ([validate_unit_holds_from_equivariant_statement] and the one
     after it). The loops and the liftings are renamed with the
     terms, and every other argument, the static environment among
     them, is the same on both sides; nothing is asked of the term
     after.

   What is not said. Of a term before with open code only the
   equation of the read is said: the laws of scope, the laws up to
   the names made, those of the two checks and those of the two tests
   ask that its code is closed. The statements take the loops and the
   liftings as they are given: nothing is said of how a caller finds
   them, nor that what it finds for the term renamed is what it finds
   for the term, renamed with it.

   The proofs are in FlambdaValidateReadingRenameProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaLiftSpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidateKnowingSpec FlambdaValidateImportsSpec
  FlambdaValidateReadingSpec.
Import ListNotations.

(* ======================================================================== *)
(* The read                                                                 *)
(* ======================================================================== *)

Definition read_arrays_equivariant_statement : Prop :=
  rewrite_equivariant read_arrays.

Definition read_arrays_keeps_scope_statement : Prop :=
  keeps_scope read_arrays.

(* ======================================================================== *)
(* The normalisation of a round                                             *)
(* ======================================================================== *)

Definition normalise_before_avoiding_keeps_scope_statement : Prop :=
  ∀ (avoid : list continuation) (rounds : nat) (kept : list code_id),
    keeps_scope (normalise_before_avoiding avoid rounds kept).

Definition normalise_before_avoiding_equivariant_up_to_statement : Prop :=
  ∀ (rounds : nat) (kept : list code_id),
    equivariant_up_to_avoiding
      (fun avoid => normalise_before_avoiding avoid rounds kept).

(* ======================================================================== *)
(* A round, and the rounds                                                  *)
(* ======================================================================== *)

Definition reading_round_keeps_scope_statement : Prop :=
  ∀ (avoid : list continuation) (rounds : nat) (G : loopified)
    (kept : list code_id),
    keeps_scope (reading_round avoid rounds G kept).

(* The round of the renamed term, with the continuations to avoid and
   those of the loops renamed, gives the result of the round under a
   second renaming, which does to the free names of that result and to
   the continuations to avoid what the first does. *)
Definition reading_round_renamed_statement : Prop :=
  ∀ (r : renaming) (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) (e : expr),
    code_closed e = true ->
    ∃ r' : renaming,
      renames_alike r r' (reading_round avoid rounds G kept e)
      ∧ rename_conts_alike r r' avoid
      ∧ reading_round (map (rename_continuation r) avoid) rounds
          (rename_loopified r G) kept (rename_expr r e)
        = rename_expr r' (reading_round avoid rounds G kept e).

Definition normalise_reading_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (G : loopified)
    (kept : list code_id),
    keeps_scope (normalise_reading rounds avoid G kept).

(* The same of the rounds, where the continuations of the loops are
   among those to avoid. *)
Definition normalise_reading_renamed_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (avoid : list continuation)
    (G : loopified) (kept : list code_id) (e : expr),
    code_closed e = true ->
    (∀ k : continuation,
       among k (map snd G) = true -> among k avoid = true) ->
    ∃ r' : renaming,
      renames_alike r r' (normalise_reading rounds avoid G kept e)
      ∧ rename_conts_alike r r' avoid
      ∧ normalise_reading rounds (map (rename_continuation r) avoid)
          (rename_loopified r G) kept (rename_expr r e)
        = rename_expr r' (normalise_reading rounds avoid G kept e).

(* ======================================================================== *)
(* The term that the checks give                                            *)
(* ======================================================================== *)

Definition before_read_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (G : loopified) (Ls : list lifting)
    (kept : list code_id),
    keeps_scope (before_read rounds G Ls kept).

(* For a renamed term, with the loops and the liftings renamed, it is
   the term that the checks give under a second renaming, which does to
   the free names of that term, and to the continuations of the units
   of the liftings and of the loops, what the first does. *)
Definition before_read_renamed_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before : expr),
    code_closed before = true ->
    ∃ r' : renaming,
      renames_alike r r' (before_read rounds G Ls kept before)
      ∧ rename_conts_alike r r' (unit_continuations Ls ++ map snd G)
      ∧ before_read rounds (rename_loopified r G)
          (map (rename_lifting r) Ls) kept (rename_expr r before)
        = rename_expr r' (before_read rounds G Ls kept before).

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

(* Each answers the same on a pair of terms and on the pair renamed,
   the names given for the loops and the liftings being renamed with
   them, where the code of the term before is closed. *)
Definition validate_unit_reading_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_reading rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit_reading rounds unused slots G Ls kept before after.

Definition validate_unit_knowing_reading_equivariant_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_knowing_reading rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r before) (rename_expr r after)
    = validate_unit_knowing_reading rounds unused slots G Ls kept before
        after.

(* ======================================================================== *)
(* The verdict that the checks give                                         *)
(* ======================================================================== *)

(* Whether the verdict that [validate_unit_reading] gives of two terms
   holds of the unit run from a static environment, asked of the term
   that [before_read] gives: the same of the two terms renamed, the
   names given for the loops and the liftings being renamed with them,
   where the code of the term before is closed. *)
Definition validate_unit_reading_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    verdict_holds_from statics rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept
      (before_read rounds (rename_loopified r G)
         (map (rename_lifting r) Ls) kept (rename_expr r before))
      (rename_expr r after)
      (validate_unit_reading rounds unused slots (rename_loopified r G)
         (map (rename_lifting r) Ls) kept (rename_expr r before)
         (rename_expr r after))
    = verdict_holds_from statics rounds unused slots G Ls kept
        (before_read rounds G Ls kept before) after
        (validate_unit_reading rounds unused slots G Ls kept before after).

(* The same of the second asking. *)
Definition validate_unit_knowing_reading_holds_from_equivariant_statement : Prop
  :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    verdict_knowing_holds_from statics rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (before_read rounds (rename_loopified r G)
         (map (rename_lifting r) Ls) kept (rename_expr r before))
      (rename_expr r after)
      (validate_unit_knowing_reading rounds unused slots
         (rename_loopified r G) (map (rename_lifting r) Ls) kept
         (rename_expr r before) (rename_expr r after))
    = verdict_knowing_holds_from statics rounds unused slots G Ls kept
        (before_read rounds G Ls kept before) after
        (validate_unit_knowing_reading rounds unused slots G Ls kept
           before after).
