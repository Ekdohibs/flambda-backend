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


(* Proofs of the statements of FlambdaValidateReadingSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaCompareSpec FlambdaNormaliseSpec
  FlambdaNormaliseProofs
  FlambdaRecHandlersSpec FlambdaRecHandlersProofs
  FlambdaUnusedStaticsSpec FlambdaLoopifySpec FlambdaLoopifyProofs
  FlambdaStubCallsSpec FlambdaStubCallsProofs FlambdaNeverReturnsSpec
  FlambdaNeverReturnsProofs FlambdaLiftSpec FlambdaArrayLoadSpec
  FlambdaArrayLoadProofs FlambdaValidateSpec FlambdaValidateProofs
  FlambdaValidateKnowingSpec FlambdaValidateKnowingProofs
  FlambdaImportsSpec FlambdaImportsProofs FlambdaValidateImportsSpec
  FlambdaValidateImportsProofs FlambdaGuardScopeSpec
  FlambdaGuardScopeProofs FlambdaValidateReadingSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
Import ListNotations.

Lemma read_arrays_in_code_refines (body : expr) :
  expr_refines body (read_arrays_in_code body).
Proof.
  unfold read_arrays_in_code.
  exact (expr_refines_trans _ _ _ (load_known_arrays_refines body)
    (rewrite_rec_handlers_refines _ load_known_arrays_refines _)).
Qed.

Theorem read_arrays_refines : read_arrays_refines_statement.
Proof.
  intros e. unfold read_arrays.
  eapply expr_refines_trans;
    [ | apply rewrite_code_refines, read_arrays_in_code_refines ].
  apply read_arrays_in_code_refines.
Qed.

Theorem normalise_before_avoiding_nothing
    : normalise_before_avoiding_nothing_statement.
Proof. intros rounds kept e. reflexivity. Qed.

Lemma normalise_before_avoiding_refines (avoid : list continuation)
    (rounds : nat) (kept : list code_id) (e : expr) :
  expr_refines e (normalise_before_avoiding avoid rounds kept e).
Proof.
  unfold normalise_before_avoiding.
  eapply expr_refines_trans; [ | apply demote_and_tidy_refines ].
  eapply expr_refines_trans;
    [ | exact (iterate_refines _
                 (normalise_before_round_avoiding_refines avoid kept)
                 rounds _) ].
  eapply expr_refines_trans;
    [ | apply guard_scope_refines,
          FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines ].
  exact (expr_refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds e)
    (iterate_refines _ (normalise_before_round_avoiding_refines avoid kept)
       rounds _)).
Qed.

Lemma reading_round_refines (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) (e : expr) :
  expr_refines e (reading_round avoid rounds G kept e).
Proof.
  unfold reading_round. cbv zeta.
  destruct (expr_eqb [] [] _ _); [ apply expr_refines_refl | ].
  apply (expr_refines_trans _ _ _ (demote_full_stub_calls_refines e)).
  apply (expr_refines_trans _ _ _
    (cut_returns_refines (demote_full_stub_calls e))).
  apply (expr_refines_trans _ _ _
    (loopify_unit_refines G (cut_returns [] (demote_full_stub_calls e)))).
  exact (expr_refines_trans _ _ _
    (normalise_before_avoiding_refines avoid rounds kept _)
    (read_arrays_refines _)).
Qed.

Theorem normalise_reading_refines : normalise_reading_refines_statement.
Proof.
  intros rounds avoid G kept e. unfold normalise_reading.
  apply iterate_refines. intros e'. apply reading_round_refines.
Qed.

Theorem normalise_reading_nothing_read
    : normalise_reading_nothing_read_statement.
Proof.
  intros rounds avoid G kept e Hround. unfold normalise_reading.
  induction rounds as [ | rounds' IH] in |- * at 1; cbn [iterate];
    [ reflexivity | ].
  cbv zeta. rewrite Hround.
  destruct (expr_eqb [] [] e e); [ reflexivity | exact IH ].
Qed.

Lemma before_read_refines (rounds : nat) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before : expr) :
  expr_refines before (before_read rounds G Ls kept before).
Proof. apply normalise_reading_refines. Qed.

Theorem validate_unit_reading_sound : validate_unit_reading_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (before_read_refines rounds G Ls kept before))).
  exact (validate_unit_sound rounds unused slots G Ls kept _ after
    Hvalidated).
Qed.

Theorem validate_unit_knowing_reading_sound
    : validate_unit_knowing_reading_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after Hvalidated.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (before_read_refines rounds G Ls kept before))).
  exact (validate_unit_knowing_sound rounds unused slots G Ls kept _ after
    Hvalidated).
Qed.

Theorem validate_unit_reading_verdict_imports
    : validate_unit_reading_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply (refines_trans _ _ _
    (expr_refines_unit_refines imports _ _
       (before_read_refines rounds G Ls kept before))).
  exact (validate_unit_sound_verdict_imports imports rounds unused slots G
    Ls kept _ after Hholds).
Qed.

Theorem validate_unit_knowing_reading_verdict_imports
    : validate_unit_knowing_reading_verdict_imports_statement.
Proof.
  intros imports rounds unused slots G Ls kept before after Hholds.
  apply (refines_trans _ _ _
    (expr_refines_unit_refines imports _ _
       (before_read_refines rounds G Ls kept before))).
  exact (validate_unit_knowing_sound_verdict_imports imports rounds unused
    slots G Ls kept _ after Hholds).
Qed.
