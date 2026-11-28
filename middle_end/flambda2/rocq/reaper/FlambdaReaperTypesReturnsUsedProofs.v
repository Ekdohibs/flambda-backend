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

(* Proofs of FlambdaReaperTypesReturnsUsedSpec.v. This file needs no
   review. The proofs are those of FlambdaReaperTypesReturnsProofs.v
   (each [_used_gen]), of which the statements of
   FlambdaReaperTypesReturnsSpec.v are the instances. *)

From OxCamlFlambda2 Require Import FlambdaReaperTypes
  FlambdaReaperTypesReturnsUsedSpec.
From OxCamlFlambda2 Require FlambdaReaperTypesProofs
  FlambdaReaperTypesReturnsProofs.

Module TP := FlambdaReaperTypesProofs.
Module RT := FlambdaReaperTypesReturnsProofs.

Theorem code_result_type_end_to_end_used :
  code_result_type_end_to_end_used_statement.
Proof. exact (RT.plain_return_used_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_used_landed :
  code_result_type_end_to_end_used_landed_statement.
Proof.
  exact (RT.plain_return_used_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem code_result_type_end_to_end_used_local :
  code_result_type_end_to_end_used_local_statement.
Proof. exact (RT.plain_return_used_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_used_local_landed :
  code_result_type_end_to_end_used_local_landed_statement.
Proof.
  exact (RT.plain_return_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_used :
  cc_code_result_type_end_to_end_used_statement.
Proof.
  exact (RT.cc_return_type_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_used_landed :
  cc_code_result_type_end_to_end_used_landed_statement.
Proof.
  exact (RT.cc_return_type_used_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_used_local :
  cc_code_result_type_end_to_end_used_local_statement.
Proof.
  exact (RT.cc_return_type_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_used_local_landed :
  cc_code_result_type_end_to_end_used_local_landed_statement.
Proof.
  exact (RT.cc_return_type_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.
