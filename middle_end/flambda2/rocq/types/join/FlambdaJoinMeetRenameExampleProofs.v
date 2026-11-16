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

(* Proofs of FlambdaJoinMeetRenameExampleSpec.v: each is a computation.
   This file needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaEnvRenameSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinTypesExamplesSpec FlambdaMeetGrammar FlambdaMeet FlambdaJoinMeet
  FlambdaJoinMeetExamplesSpec FlambdaEnvRenameJoinSpec
  FlambdaJoinMeetRenameSpec FlambdaJoinMeetRenameExampleSpec.
Import ListNotations.

Theorem example_join_makes_variable : example_join_makes_variable_statement.
Proof. vm_compute. reflexivity. Qed.

(* The two answers differ in the variable that their analyses define. *)
Theorem example_join_one_source : example_join_one_source_statement.
Proof.
  intros Hsame.
  apply
    (f_equal
       (option_map
          (fun answer : Typing_env.t * Analysis.t unit =>
             map fst (Analysis.definitions_in_joined_envs (snd answer)))))
    in Hsame.
  vm_compute in Hsame. discriminate Hsame.
Qed.

Theorem example_join_renamed_source : example_join_renamed_source_statement.
Proof. vm_compute. reflexivity. Qed.
