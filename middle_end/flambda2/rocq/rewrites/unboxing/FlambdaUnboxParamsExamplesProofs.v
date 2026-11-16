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


(* Proofs of the statements of FlambdaUnboxParamsExamplesSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaLoopAliasesSpec FlambdaLoopAliasesProofs
  FlambdaShortcutSpec FlambdaShortcutProofs FlambdaSinkContsSpec
  FlambdaSinkContsProofs FlambdaUnboxSpec FlambdaUnboxProofs
  FlambdaUnusedContsSpec FlambdaUnusedContsProofs FlambdaUnusedParamsSpec
  FlambdaUnusedParamsProofs FlambdaValidateSpec FlambdaValidateProofs
  FlambdaUnboxParamsSpec FlambdaUnboxParamsProofs
  FlambdaUnboxParamsExamplesSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec FlambdaAliasesProofs.
Import ListNotations.

(* The round refines, as each of its rewrites does. *)
Lemma round_with_unboxing_refines (e : expr) :
  expr_refines e (round_with_unboxing e).
Proof.
  unfold round_with_unboxing. cbv zeta.
  repeat
    (eapply expr_refines_trans;
     [ | solve [ first [ apply unbox_known_params_refines | refinement_step ] ]
     ]).
  apply expr_refines_refl.
Qed.

Lemma rounds_with_unboxing_refine (rounds : nat) :
  ∀ e : expr, expr_refines e (iterate rounds round_with_unboxing e).
Proof.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate].
  - apply expr_refines_refl.
  - cbv zeta.
    match goal with
    | |- context [if ?same then _ else _] =>
      destruct same; [ apply round_with_unboxing_refines | ]
    end.
    exact (expr_refines_trans _ _ _ (round_with_unboxing_refines e)
      (IH (round_with_unboxing e))).
Qed.

(* What is the same as the body after, once the rounds are done, is
   refined by it. *)
Lemma validated_refines (rounds : nat) (before after : expr) :
  expr_eqb [] [] (iterate rounds round_with_unboxing before) after = true ->
  expr_refines before after.
Proof.
  intros Hsame.
  exact (expr_refines_trans _ _ _ (rounds_with_unboxing_refine rounds before)
    (expr_equivalent_refines _ _ (expr_eqb_sound _ _ Hsame))).
Qed.

Theorem pair_computed_validated : pair_computed_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem pair_computed_not_validated_today :
  pair_computed_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem pair_computed_refines : pair_computed_refines_statement.
Proof. exact (validated_refines 5 _ _ pair_computed_validated). Qed.

Theorem pair_built_unboxed : pair_built_unboxed_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem pair_built_not_validated : pair_built_not_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_built_validated : int64_built_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_built_not_validated_today :
  int64_built_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_built_refines : int64_built_refines_statement.
Proof. exact (validated_refines 5 _ _ int64_built_validated). Qed.
