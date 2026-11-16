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


(* Proofs of the statements of FlambdaJoinTypesExamplesSpec.v, by
   computation. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaJoinEnv FlambdaJoinTypes
  FlambdaJoinTypesExamplesSpec.
Import ListNotations.

Theorem example_join_naked_immediates : example_join_naked_immediates_statement.
Proof. reflexivity. Qed.

Theorem example_join_tagged_immediates
  : example_join_tagged_immediates_statement.
Proof. reflexivity. Qed.

Theorem example_join_boxed_floats : example_join_boxed_floats_statement.
Proof. reflexivity. Qed.

Theorem example_join_strings : example_join_strings_statement.
Proof. reflexivity. Qed.

Theorem example_join_one_constant : example_join_one_constant_statement.
Proof. reflexivity. Qed.

Theorem example_join_two_constants : example_join_two_constants_statement.
Proof. reflexivity. Qed.

Theorem example_join_constant_and_set : example_join_constant_and_set_statement.
Proof. reflexivity. Qed.

Theorem example_join_unknown : example_join_unknown_statement.
Proof. reflexivity. Qed.

Theorem example_join_bottom : example_join_bottom_statement.
Proof. reflexivity. Qed.

Theorem example_join_all_bottom : example_join_all_bottom_statement.
Proof. reflexivity. Qed.

Theorem example_join_string_and_immediate
  : example_join_string_and_immediate_statement.
Proof. reflexivity. Qed.

Theorem example_join_null_and_immediate
  : example_join_null_and_immediate_statement.
Proof. reflexivity. Qed.

Theorem example_join_mutable_blocks : example_join_mutable_blocks_statement.
Proof. reflexivity. Qed.

Theorem example_join_fields : example_join_fields_statement.
Proof. reflexivity. Qed.

Theorem example_join_same_fields : example_join_same_fields_statement.
Proof. reflexivity. Qed.

Theorem example_join_fields_of_two_variables
  : example_join_fields_of_two_variables_statement.
Proof. reflexivity. Qed.

Theorem example_join_two_sizes : example_join_two_sizes_statement.
Proof. reflexivity. Qed.

Theorem example_join_two_shapes : example_join_two_shapes_statement.
Proof. reflexivity. Qed.

Theorem example_join_two_tags : example_join_two_tags_statement.
Proof. reflexivity. Qed.

Theorem example_join_immediate_and_block
  : example_join_immediate_and_block_statement.
Proof. reflexivity. Qed.

Theorem example_join_variants : example_join_variants_statement.
Proof. reflexivity. Qed.

Theorem example_join_closures : example_join_closures_statement.
Proof. reflexivity. Qed.

Theorem example_join_closures_of_two_codes
  : example_join_closures_of_two_codes_statement.
Proof. reflexivity. Qed.

Theorem example_join_closures_one_without_function_type
  : example_join_closures_one_without_function_type_statement.
Proof. reflexivity. Qed.

Theorem example_join_no_type : example_join_no_type_statement.
Proof. reflexivity. Qed.

Theorem example_join_two_kinds : example_join_two_kinds_statement.
Proof. reflexivity. Qed.

Theorem example_join_bound_reached : example_join_bound_reached_statement.
Proof. reflexivity. Qed.

Theorem example_join_closure_types_of_another_kind
  : example_join_closure_types_of_another_kind_statement.
Proof. reflexivity. Qed.

Theorem example_join_alias_of_another_kind
  : example_join_alias_of_another_kind_statement.
Proof. reflexivity. Qed.
