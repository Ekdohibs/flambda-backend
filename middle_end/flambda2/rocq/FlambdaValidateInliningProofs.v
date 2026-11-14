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


(* Proofs of the statements of FlambdaValidateInliningSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNormaliseSpec FlambdaNormaliseProofs
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaLiftSpec
  FlambdaRenameApartSpec FlambdaRenameApartProofs
  FlambdaInlineCallsAtSpec FlambdaInlineCallsAtProofs
  FlambdaValidateSpec FlambdaValidateProofs FlambdaValidateKnowingSpec
  FlambdaValidateKnowingProofs FlambdaImportsSpec FlambdaImportsProofs
  FlambdaValidateImportsSpec FlambdaValidateImportsProofs
  FlambdaValidateInliningSpec.
Import ListNotations.

Theorem inlined_in_turn_refines : inlined_in_turn_refines_statement.
Proof.
  intros avoid choices.
  induction choices as [ | [sites K] choices IH]; intros e;
    cbn [inlined_in_turn]; [ apply expr_refines_refl | ].
  apply (expr_refines_trans _ _ _ (inline_calls_at_refines sites K e)).
  apply (expr_refines_trans _ _ _
    (expr_equivalent_refines _ _
       (rename_apart_avoiding_equivalent avoid (inline_calls_at sites K e)))).
  apply IH.
Qed.

Lemma before_inlined_refines (choices : list inlining_choice)
    (G : loopified) (Ls : list lifting) (before : expr) :
  expr_refines before (before_inlined choices G Ls before).
Proof. apply inlined_in_turn_refines. Qed.

Theorem validate_unit_inlining_sound
    : validate_unit_inlining_sound_statement.
Proof.
  intros choices rounds unused slots G Ls kept before after Hvalidated.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (before_inlined_refines choices G Ls before))).
  exact (validate_unit_sound rounds unused slots G Ls kept _ after
    Hvalidated).
Qed.

Theorem validate_unit_knowing_inlining_sound
    : validate_unit_knowing_inlining_sound_statement.
Proof.
  intros choices rounds unused slots G Ls kept before after Hvalidated.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (before_inlined_refines choices G Ls before))).
  exact (validate_unit_knowing_sound rounds unused slots G Ls kept _ after
    Hvalidated).
Qed.

Theorem validate_unit_inlining_verdict_imports
    : validate_unit_inlining_verdict_imports_statement.
Proof.
  intros imports choices rounds unused slots G Ls kept before after Hholds.
  apply (refines_trans _ _ _
    (expr_refines_unit_refines imports _ _
       (before_inlined_refines choices G Ls before))).
  exact (validate_unit_sound_verdict_imports imports rounds unused slots G
    Ls kept _ after Hholds).
Qed.

Theorem validate_unit_knowing_inlining_verdict_imports
    : validate_unit_knowing_inlining_verdict_imports_statement.
Proof.
  intros imports choices rounds unused slots G Ls kept before after Hholds.
  apply (refines_trans _ _ _
    (expr_refines_unit_refines imports _ _
       (before_inlined_refines choices G Ls before))).
  exact (validate_unit_knowing_sound_verdict_imports imports rounds unused
    slots G Ls kept _ after Hholds).
Qed.
