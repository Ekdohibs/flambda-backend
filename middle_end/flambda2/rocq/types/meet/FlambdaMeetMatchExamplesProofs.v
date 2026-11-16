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


(* Proofs of the statements of FlambdaMeetMatchExamplesSpec.v. Nothing
   here needs review: each is by running the functions. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec FlambdaMeetMatchExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* Both meets answer what the meet that is to be proved answers with
   the join that forgets, if the join is not called. *)
Local Ltac run_meets :=
  intros join; split; vm_compute; reflexivity.

Theorem match_is_int : match_is_int_statement.
Proof.
  split; [ | split ]; [ | vm_compute; reflexivity .. ].
  intros join choices. vm_compute. reflexivity.
Qed.

Theorem match_arm_immediate : match_arm_immediate_statement.
Proof. split; [ run_meets | split; vm_compute; reflexivity ]. Qed.

Theorem match_arm_block : match_arm_block_statement.
Proof. split; [ run_meets | split; vm_compute; reflexivity ]. Qed.

Theorem match_no_other_arm : match_no_other_arm_statement.
Proof. run_meets. Qed.

Theorem match_untag : match_untag_statement.
Proof. split; [ run_meets | split; vm_compute; reflexivity ]. Qed.

Theorem match_arm_constructor : match_arm_constructor_statement.
Proof. split; [ run_meets | split; vm_compute; reflexivity ]. Qed.

Theorem match_inner_arm : match_inner_arm_statement.
Proof. run_meets. Qed.

Theorem match_load : match_load_statement.
Proof. split; [ run_meets | split; vm_compute; reflexivity ]. Qed.

Theorem match_load_of_immediate : match_load_of_immediate_statement.
Proof. run_meets. Qed.
