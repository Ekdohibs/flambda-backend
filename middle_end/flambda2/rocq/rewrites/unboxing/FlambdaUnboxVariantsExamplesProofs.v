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


(* Proofs of the statements of FlambdaUnboxVariantsExamplesSpec.v.
   Nothing here needs review. *)

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
  FlambdaUnboxParamsSpec FlambdaUnboxParamsExamplesSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxVariantsProofs
  FlambdaUnboxVariantsExamplesSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec FlambdaAliasesProofs.
Import ListNotations.

(* The round refines, as each of its rewrites does. *)
Lemma round_with_variants_refines (e : expr) :
  expr_refines e (round_with_variants e).
Proof.
  unfold round_with_variants. cbv zeta.
  repeat
    (eapply expr_refines_trans;
     [ | solve
           [ first [ apply unbox_variant_params_refines | refinement_step ] ]
     ]).
  apply expr_refines_refl.
Qed.

Lemma rounds_with_variants_refine (rounds : nat) :
  ∀ e : expr, expr_refines e (iterate rounds round_with_variants e).
Proof.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate].
  - apply expr_refines_refl.
  - cbv zeta.
    match goal with
    | |- context [if ?same then _ else _] =>
      destruct same; [ apply round_with_variants_refines | ]
    end.
    exact (expr_refines_trans _ _ _ (round_with_variants_refines e)
      (IH (round_with_variants e))).
Qed.

(* What is the same as the body after, once the rounds are done, is
   refined by it. *)
Lemma validated_refines (rounds : nat) (before after : expr) :
  expr_eqb [] [] (iterate rounds round_with_variants before) after = true ->
  expr_refines before after.
Proof.
  intros Hsame.
  exact (expr_refines_trans _ _ _ (rounds_with_variants_refine rounds before)
    (expr_equivalent_refines _ _ (expr_eqb_sound _ _ Hsame))).
Qed.

Theorem option_heap_validated : option_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem option_heap_not_validated_today :
  option_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem option_heap_refines : option_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ option_heap_validated). Qed.

Theorem list_heap_validated : list_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem list_heap_not_validated_today :
  list_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem list_heap_refines : list_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ list_heap_validated). Qed.

Theorem shape_heap_validated : shape_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem shape_heap_not_validated_today :
  shape_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem shape_heap_refines : shape_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ shape_heap_validated). Qed.

Theorem shape2_heap_validated : shape2_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem shape2_heap_not_validated_today :
  shape2_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem shape2_heap_refines : shape2_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ shape2_heap_validated). Qed.

Theorem nested_heap_validated : nested_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem nested_heap_not_validated_today :
  nested_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem nested_heap_refines : nested_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ nested_heap_validated). Qed.

Theorem either_heap_validated : either_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem either_heap_not_validated_today :
  either_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem either_heap_refines : either_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ either_heap_validated). Qed.

Theorem int64_option_heap_validated : int64_option_heap_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_option_heap_not_validated_today :
  int64_option_heap_not_validated_today_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_option_heap_refines : int64_option_heap_refines_statement.
Proof. exact (validated_refines 5 _ _ int64_option_heap_validated). Qed.

Theorem earlier_examples_validated : earlier_examples_validated_statement.
Proof. split; vm_compute; reflexivity. Qed.

Theorem bodies_validated_by_first_phase :
  bodies_validated_by_first_phase_statement.
Proof. repeat split; vm_compute; reflexivity. Qed.

(* The rounds of the first phase of the validation refine, and what
   they take to the body after is refined by it. *)
Lemma first_phase_rounds_refine (rounds : nat) :
  ∀ e : expr, expr_refines e (iterate rounds normalise_expr_params e).
Proof.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate].
  - apply expr_refines_refl.
  - cbv zeta.
    match goal with
    | |- context [if ?same then _ else _] =>
      destruct same; [ apply normalise_expr_params_refines | ]
    end.
    exact (expr_refines_trans _ _ _ (normalise_expr_params_refines e)
      (IH (normalise_expr_params e))).
Qed.

Lemma validated_by_first_phase_refines (before after : expr) :
  validated_by_first_phase before after -> expr_refines before after.
Proof.
  intros Hsame.
  exact (expr_refines_trans _ _ _ (first_phase_rounds_refine 5 before)
    (expr_equivalent_refines _ _ (expr_eqb_sound _ _ Hsame))).
Qed.

Theorem while_two_tests_validated : while_two_tests_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem while_two_tests_refines : while_two_tests_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ while_two_tests_validated).
Qed.

Theorem shape_constant_validated : shape_constant_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem shape_constant_refines : shape_constant_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ shape_constant_validated).
Qed.

(* The rounds of the first phase for a whole term refine, and what they
   take to the term after is refined by it. *)
Lemma validated_by_rounds_refines (before after : expr) :
  validated_by_rounds before after -> expr_refines before after.
Proof.
  intros Hsame.
  exact (expr_refines_trans _ _ _
    (iterate_refines _ normalise_params_round_refines 5 before)
    (expr_equivalent_refines _ _ (expr_eqb_sound _ _ Hsame))).
Qed.

Theorem constant_in_code_validated : constant_in_code_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem constant_in_code_refines : constant_in_code_refines_statement.
Proof.
  exact (validated_by_rounds_refines _ _ constant_in_code_validated).
Qed.

Theorem constant_below_code_left : constant_below_code_left_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem constant_in_loop_validated : constant_in_loop_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem constant_in_loop_refines : constant_in_loop_refines_statement.
Proof.
  exact (validated_by_rounds_refines _ _ constant_in_loop_validated).
Qed.

Theorem pair_constant_validated : pair_constant_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem pair_constant_refines : pair_constant_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ pair_constant_validated).
Qed.

Theorem option_pair_validated : option_pair_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem option_pair_refines : option_pair_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ option_pair_validated).
Qed.

Theorem option_fields_validated : option_fields_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem option_fields_refines : option_fields_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ option_fields_validated).
Qed.

Theorem two_tags_validated : two_tags_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem two_tags_refines : two_tags_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ two_tags_validated).
Qed.

Theorem int64_max_validated : int64_max_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_max_refines : int64_max_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ int64_max_validated).
Qed.

Theorem matched_twice_validated : matched_twice_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem matched_twice_refines : matched_twice_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ matched_twice_validated).
Qed.

Theorem pair_loop_validated : pair_loop_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem pair_loop_refines : pair_loop_refines_statement.
Proof.
  exact (validated_by_first_phase_refines _ _ pair_loop_validated).
Qed.

Theorem int64_passed_on_validated : int64_passed_on_validated_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem int64_passed_on_refines : int64_passed_on_refines_statement.
Proof.
  exact (validated_by_rounds_refines _ _ int64_passed_on_validated).
Qed.
