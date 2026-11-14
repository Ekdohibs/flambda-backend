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
   first, under renaming.

   This file is to be reviewed: the statements. It has no definition.

   FlambdaValidateInliningSpec.v has a function beside the validation:
   the calls of the term before that stand at chosen places are
   inlined, one choice after another, the term being renamed apart
   after each ([inlined_in_turn], [before_inlined]), and the
   validation is asked of what that gives
   ([validate_unit_inlining],
   [validate_unit_knowing_inlining]). That
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
   variables and on continuations. It moves no symbol and no code id,
   and it moves no call: a place is told by position, so the choices
   are the same for a term and for the term renamed. The liftings and
   the continuations given for the loops hold continuations, and are
   renamed with the terms
   (FlambdaValidatorEquivarianceSpec.rename_lifting,
   [rename_loopified]).

   The forms are those of FlambdaValidatorEquivarianceSpec.v.
   - The inlining at chosen places
     (FlambdaInlineCallsAtSpec.inline_calls_at) makes no name: it
     commutes with renaming, as an equation
     (FlambdaInlineCallsAtSpec.inline_calls_at_equivariant_statement),
     and it keeps scope.
   - The choices in turn ([inlined_in_turn]) rename the term apart
     after each, and that makes names. So they commute with renaming
     up to the names that they make, the second renaming doing to the
     continuations that they are given to avoid what the first does
     ([equivariant_up_to_avoiding]); and they keep scope.
   - The function as the checks call it ([before_inlined]) gives them
     the continuations of the units of the liftings and those of the
     loops to avoid, which are renamed with the term.
   - The two checks then answer the same of a pair and of the pair
     renamed, where the code of the term before is closed: the term
     that each gives to the function that it asks is, for a renamed
     pair, that term under a second renaming that does to its free
     names, and to the continuations of the liftings and of the loops,
     what the first does, which is what
     FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement
     asks.

   What is not said. Of a term before with open code only the
   equation of the inlining is said: the laws of scope, the law up to
   the names made and those of the two checks ask that its code is
   closed. Nothing is said here of the tests from the imports on the
   term that [before_inlined] gives
   (FlambdaValidateImportsSpec.verdict_holds_from): a law of those
   tests under renaming, where there is one, applies to that term by
   [before_inlined_renamed_statement].

   The proofs are in FlambdaValidateInliningRenameProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec FlambdaBindersSpec FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec FlambdaLiftSpec FlambdaInlineCallsAtSpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidateKnowingSpec FlambdaValidateInliningSpec.
Import ListNotations.

(* ======================================================================== *)
(* The inlining at chosen places                                            *)
(* ======================================================================== *)

Definition inline_calls_at_keeps_scope_statement : Prop :=
  ∀ (sites : list path) (K : calls_in_code),
    keeps_scope (inline_calls_at sites K).

(* ======================================================================== *)
(* The choices in turn                                                      *)
(* ======================================================================== *)

Definition inlined_in_turn_keeps_scope_statement : Prop :=
  ∀ (avoid : list continuation) (choices : list inlining_choice),
    keeps_scope (inlined_in_turn avoid choices).

Definition inlined_in_turn_equivariant_up_to_statement : Prop :=
  ∀ choices : list inlining_choice,
    equivariant_up_to_avoiding (fun avoid => inlined_in_turn avoid choices).

(* ======================================================================== *)
(* The term that the checks give                                            *)
(* ======================================================================== *)

Definition before_inlined_keeps_scope_statement : Prop :=
  ∀ (choices : list inlining_choice) (G : loopified) (Ls : list lifting),
    keeps_scope (before_inlined choices G Ls).

(* For a renamed term, with the loops and the liftings renamed, it is
   the term that the checks give under a second renaming, which does to
   the free names of that term, and to the continuations of the units
   of the liftings and of the loops, what the first does. *)
Definition before_inlined_renamed_statement : Prop :=
  ∀ (r : renaming) (choices : list inlining_choice) (G : loopified)
    (Ls : list lifting) (before : expr),
    code_closed before = true ->
    ∃ r' : renaming,
      renames_alike r r' (before_inlined choices G Ls before)
      ∧ rename_conts_alike r r' (unit_continuations Ls ++ map snd G)
      ∧ before_inlined choices (rename_loopified r G)
          (map (rename_lifting r) Ls) (rename_expr r before)
        = rename_expr r' (before_inlined choices G Ls before).

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

(* Each answers the same on a pair of terms and on the pair renamed,
   the names given for the loops and the liftings being renamed with
   them, where the code of the term before is closed. *)
Definition validate_unit_inlining_equivariant_statement : Prop :=
  ∀ (r : renaming) (choices : list inlining_choice) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_inlining choices rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r before) (rename_expr r after)
    = validate_unit_inlining choices rounds unused slots G Ls kept
        before after.

Definition validate_unit_knowing_inlining_equivariant_statement : Prop :=
  ∀ (r : renaming) (choices : list inlining_choice) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    code_closed before = true ->
    validate_unit_knowing_inlining choices rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r before) (rename_expr r after)
    = validate_unit_knowing_inlining choices rounds unused slots G Ls
        kept before after.
