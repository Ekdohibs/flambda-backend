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


(* The proofs of the examples of FlambdaJoinMeetExamplesSpec.v, by
   computation. This file needs no review. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaTypingEnv FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec
  FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaJoinMeet FlambdaJoinMeetExamplesSpec.
Import ListNotations.

Theorem example_join_env_two_constants
  : example_join_env_two_constants_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_one_constant
  : example_join_env_one_constant_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_set_and_constant
  : example_join_env_set_and_constant_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_one_alias : example_join_env_one_alias_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_two_aliases
  : example_join_env_two_aliases_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_pairs : example_join_env_pairs_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_immediate_and_block
  : example_join_env_immediate_and_block_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_two_tags : example_join_env_two_tags_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_variable_of_one_use
  : example_join_env_variable_of_one_use_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_variable_of_both_uses
  : example_join_env_variable_of_both_uses_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_bottom_use : example_join_env_bottom_use_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_join_env_no_use : example_join_env_no_use_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_recovered_first_join : example_recovered_first_join_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_recovered_second_join
  : example_recovered_second_join_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_recovered_third_join : example_recovered_third_join_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_recovered_bottom : example_recovered_bottom_statement.
Proof. vm_compute. repeat split. Qed.

Theorem example_meet_joins_alternatives
  : example_meet_joins_alternatives_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_meet_immediate_alternative
  : example_meet_immediate_alternative_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_meet_block_alternative
  : example_meet_block_alternative_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_nested_join_names : example_nested_join_names_statement.
Proof. vm_compute. reflexivity. Qed.

Theorem example_nested_join_avoids : example_nested_join_avoids_statement.
Proof. vm_compute. reflexivity. Qed.
