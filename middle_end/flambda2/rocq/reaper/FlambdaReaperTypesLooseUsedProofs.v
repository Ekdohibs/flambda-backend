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

(* Proofs of FlambdaReaperTypesLooseUsedSpec.v. This file needs no
   review. The proofs are those of FlambdaReaperTypesLooseProofs.v (each
   [_used]), of which the statements of FlambdaReaperTypesLooseSpec.v are
   the instances. *)

From OxCamlFlambda2 Require Import FlambdaReaperTypes
  FlambdaReaperTypesLooseUsedSpec.
From OxCamlFlambda2 Require FlambdaReaperTypesProofs
  FlambdaReaperTypesLooseProofs.

Module TP := FlambdaReaperTypesProofs.
Module LP := FlambdaReaperTypesLooseProofs.

Theorem unaliased_result_type_end_to_end_used :
  unaliased_result_type_end_to_end_used_statement.
Proof.
  exact (LP.plain_result_loose_used false all_fixes TP.all_fixes_sound).
Qed.

Theorem unaliased_result_type_end_to_end_used_landed :
  unaliased_result_type_end_to_end_used_landed_statement.
Proof.
  exact (LP.plain_result_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unaliased_result_type_end_to_end_used_local :
  unaliased_result_type_end_to_end_used_local_statement.
Proof.
  exact (LP.plain_result_loose_used true all_fixes TP.all_fixes_sound).
Qed.

Theorem unaliased_result_end_to_end_used_local_landed :
  unaliased_result_end_to_end_used_local_landed_statement.
Proof.
  exact (LP.plain_result_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_result_type_end_to_end_used :
  cc_unaliased_result_type_end_to_end_used_statement.
Proof. exact (LP.cc_result_loose_used false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_type_end_to_end_used_landed :
  cc_unaliased_result_type_end_to_end_used_landed_statement.
Proof.
  exact (LP.cc_result_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_result_type_end_to_end_used_local :
  cc_unaliased_result_type_end_to_end_used_local_statement.
Proof. exact (LP.cc_result_loose_used true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_end_to_end_used_local_landed :
  cc_unaliased_result_end_to_end_used_local_landed_statement.
Proof.
  exact (LP.cc_result_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unaliased_code_result_end_to_end_used :
  unaliased_code_result_end_to_end_used_statement.
Proof.
  exact (LP.plain_return_loose_used false all_fixes TP.all_fixes_sound).
Qed.

Theorem unaliased_code_result_end_to_end_used_landed :
  unaliased_code_result_end_to_end_used_landed_statement.
Proof.
  exact (LP.plain_return_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unaliased_code_result_end_to_end_used_local :
  unaliased_code_result_end_to_end_used_local_statement.
Proof.
  exact (LP.plain_return_loose_used true all_fixes TP.all_fixes_sound).
Qed.

Theorem unaliased_code_result_used_local_landed :
  unaliased_code_result_used_local_landed_statement.
Proof.
  exact (LP.plain_return_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_code_result_end_to_end_used :
  cc_unaliased_code_result_end_to_end_used_statement.
Proof. exact (LP.cc_return_loose_used false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_end_to_end_used_landed :
  cc_unaliased_code_result_end_to_end_used_landed_statement.
Proof.
  exact (LP.cc_return_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_code_result_end_to_end_used_local :
  cc_unaliased_code_result_end_to_end_used_local_statement.
Proof. exact (LP.cc_return_loose_used true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_used_local_landed :
  cc_unaliased_code_result_used_local_landed_statement.
Proof.
  exact (LP.cc_return_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_types_end_to_end_used :
  calls_unaliased_types_end_to_end_used_statement.
Proof.
  exact (LP.calls_types_loose_used false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_types_end_to_end_used_landed :
  calls_unaliased_types_end_to_end_used_landed_statement.
Proof.
  exact (LP.calls_types_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_types_end_to_end_used_local :
  calls_unaliased_types_end_to_end_used_local_statement.
Proof. exact (LP.calls_types_loose_used true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_types_used_local_landed :
  calls_unaliased_types_used_local_landed_statement.
Proof.
  exact (LP.calls_types_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_result_end_to_end_used :
  calls_unaliased_result_end_to_end_used_statement.
Proof.
  exact (LP.calls_result_loose_used false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_result_end_to_end_used_landed :
  calls_unaliased_result_end_to_end_used_landed_statement.
Proof.
  exact (LP.calls_result_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_result_end_to_end_used_local :
  calls_unaliased_result_end_to_end_used_local_statement.
Proof.
  exact (LP.calls_result_loose_used true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_result_used_local_landed :
  calls_unaliased_result_used_local_landed_statement.
Proof.
  exact (LP.calls_result_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_used :
  calls_unaliased_unboxed_end_to_end_used_statement.
Proof.
  exact (LP.calls_unboxed_loose_used false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_used_landed :
  calls_unaliased_unboxed_end_to_end_used_landed_statement.
Proof.
  exact (LP.calls_unboxed_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_used_local :
  calls_unaliased_unboxed_end_to_end_used_local_statement.
Proof.
  exact (LP.calls_unboxed_loose_used true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_used_local_landed :
  calls_unaliased_unboxed_used_local_landed_statement.
Proof.
  exact (LP.calls_unboxed_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem repr_unaliased_types_end_to_end_used :
  repr_unaliased_types_end_to_end_used_statement.
Proof. exact (LP.repr_types_loose_used false all_fixes TP.all_fixes_sound). Qed.

Theorem repr_unaliased_types_end_to_end_used_landed :
  repr_unaliased_types_end_to_end_used_landed_statement.
Proof.
  exact (LP.repr_types_loose_used false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem repr_unaliased_types_end_to_end_used_local :
  repr_unaliased_types_end_to_end_used_local_statement.
Proof. exact (LP.repr_types_loose_used true all_fixes TP.all_fixes_sound). Qed.

Theorem repr_unaliased_types_end_to_end_used_local_landed :
  repr_unaliased_types_end_to_end_used_local_landed_statement.
Proof.
  exact (LP.repr_types_loose_used true landed_fixes TP.landed_fixes_sound).
Qed.
