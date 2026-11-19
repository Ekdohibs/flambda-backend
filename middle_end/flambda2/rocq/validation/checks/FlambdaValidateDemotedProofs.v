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


(* Proofs of the statements of FlambdaValidateDemotedSpec.v. Nothing
   here needs review: the two of the control are computations. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaUnusedStaticsSpec FlambdaLoopifySpec
  FlambdaLiftSpec FlambdaDemoteSpec FlambdaDemoteProofs
  FlambdaValidateSpec FlambdaValidateProofs FlambdaValidateKnowingSpec
  FlambdaValidateKnowingProofs FlambdaImportsSpec FlambdaImportsProofs
  FlambdaValidateImportsSpec FlambdaValidateImportsProofs
  FlambdaValidateReadingSpec FlambdaValidateReadingProofs
  FlambdaValidateDemotedSpec.
Import ListNotations.

Theorem after_demoted_refines : after_demoted_refines_statement.
Proof.
  intros after. unfold after_demoted.
  apply expr_equivalent_refines, expr_equivalent_sym,
    demote_handlers_equivalent.
Qed.

Theorem demoted_program_refines : demoted_program_refines_statement.
Proof.
  intros before after Hrefines.
  exact (refines_trans _ _ _ Hrefines
    (expr_refines_program _ _ (after_demoted_refines after))).
Qed.

Theorem demoted_unit_refines : demoted_unit_refines_statement.
Proof.
  intros imports before after Hrefines.
  exact (refines_trans _ _ _ Hrefines
    (expr_refines_unit_refines imports _ _ (after_demoted_refines after))).
Qed.

Theorem validate_unit_demoted_sound : validate_unit_demoted_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply demoted_program_refines.
  exact (validate_unit_sound rounds unused slots G Ls kept before _
    Hvalidated).
Qed.

Theorem validate_unit_knowing_demoted_sound
    : validate_unit_knowing_demoted_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply demoted_program_refines.
  exact (validate_unit_knowing_sound rounds unused slots G Ls kept before _
    Hvalidated).
Qed.

Theorem validate_unit_reading_demoted_sound
    : validate_unit_reading_demoted_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply demoted_program_refines.
  exact (validate_unit_reading_sound rounds unused slots G Ls kept before _
    Hvalidated).
Qed.

Theorem validate_unit_knowing_reading_demoted_sound
    : validate_unit_knowing_reading_demoted_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply demoted_program_refines.
  exact (validate_unit_knowing_reading_sound rounds unused slots G Ls kept
    before _ Hvalidated).
Qed.

Theorem validate_unit_demoted_imports
    : validate_unit_demoted_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply demoted_unit_refines.
  exact (validate_unit_sound_verdict_imports imports rounds unused slots G
    Ls kept before _ Hholds).
Qed.

Theorem validate_unit_knowing_demoted_imports
    : validate_unit_knowing_demoted_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply demoted_unit_refines.
  exact (validate_unit_knowing_sound_verdict_imports imports rounds unused
    slots G Ls kept before _ Hholds).
Qed.

Theorem validate_unit_reading_demoted_imports
    : validate_unit_reading_demoted_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply demoted_unit_refines.
  exact (validate_unit_reading_verdict_imports imports rounds unused slots
    G Ls kept before _ Hholds).
Qed.

Theorem validate_unit_knowing_reading_demoted_imports
    : validate_unit_knowing_reading_demoted_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply demoted_unit_refines.
  exact (validate_unit_knowing_reading_verdict_imports imports rounds
    unused slots G Ls kept before _ Hholds).
Qed.

Theorem kept_handler_validated : kept_handler_validated_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

Theorem kept_handler_wrong_refused : kept_handler_wrong_refused_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.
