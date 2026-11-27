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

(* Proofs of FlambdaReaperTypesCallsUsedSpec.v. This file needs no
   review. The proofs are those of FlambdaReaperTypesCallsProofs.v (each
   [_used_gen]), of which the statements of FlambdaReaperTypesCallsSpec.v
   are the instances ([unit_premises_used]). *)

From OxCamlFlambda2 Require Import FlambdaReaperTypes
  FlambdaReaperTypesCallsSpec FlambdaReaperTypesCallsUsedSpec.
From OxCamlFlambda2 Require FlambdaReaperTypesProofs
  FlambdaReaperTypesCallsProofs.

Module TP := FlambdaReaperTypesProofs.
Module CP := FlambdaReaperTypesCallsProofs.

Theorem used_premises_instance : used_premises_instance_statement.
Proof. exact CP.unit_premises_used. Qed.

Theorem unbox_calls_types_end_to_end_used :
  unbox_calls_types_end_to_end_used_statement.
Proof.
  exact (CP.unbox_calls_types_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_used_landed :
  unbox_calls_types_end_to_end_used_landed_statement.
Proof.
  exact (CP.unbox_calls_types_used_gen false landed_fixes
           TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_used_local :
  unbox_calls_types_end_to_end_used_local_statement.
Proof.
  exact (CP.unbox_calls_types_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_used_local_landed :
  unbox_calls_types_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.unbox_calls_types_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem repr_calls_types_end_to_end_used :
  repr_calls_types_end_to_end_used_statement.
Proof.
  exact (CP.repr_calls_types_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem repr_calls_types_end_to_end_used_landed :
  repr_calls_types_end_to_end_used_landed_statement.
Proof.
  exact (CP.repr_calls_types_used_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem repr_calls_types_end_to_end_used_local :
  repr_calls_types_end_to_end_used_local_statement.
Proof.
  exact (CP.repr_calls_types_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem repr_calls_types_end_to_end_used_local_landed :
  repr_calls_types_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.repr_calls_types_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem result_type_end_to_end_used :
  result_type_end_to_end_used_statement.
Proof.
  exact (CP.plain_result_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem result_type_end_to_end_used_landed :
  result_type_end_to_end_used_landed_statement.
Proof.
  exact (CP.plain_result_used_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem result_type_end_to_end_used_local :
  result_type_end_to_end_used_local_statement.
Proof.
  exact (CP.plain_result_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem result_type_end_to_end_used_local_landed :
  result_type_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.plain_result_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_used :
  cc_result_type_end_to_end_used_statement.
Proof.
  exact (CP.cc_result_type_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_used_landed :
  cc_result_type_end_to_end_used_landed_statement.
Proof.
  exact (CP.cc_result_type_used_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_used_local :
  cc_result_type_end_to_end_used_local_statement.
Proof.
  exact (CP.cc_result_type_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_used_local_landed :
  cc_result_type_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.cc_result_type_used_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_used :
  unbox_calls_result_type_end_to_end_used_statement.
Proof.
  exact (CP.unbox_calls_result_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_used_landed :
  unbox_calls_result_type_end_to_end_used_landed_statement.
Proof.
  exact (CP.unbox_calls_result_used_gen false landed_fixes
           TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_used_local :
  unbox_calls_result_type_end_to_end_used_local_statement.
Proof.
  exact (CP.unbox_calls_result_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_result_end_to_end_used_local_landed :
  unbox_calls_result_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.unbox_calls_result_used_gen true landed_fixes
           TP.landed_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_used :
  calls_unboxed_result_type_end_to_end_used_statement.
Proof.
  exact (CP.calls_unboxed_result_used_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_used_landed :
  calls_unboxed_result_type_end_to_end_used_landed_statement.
Proof.
  exact (CP.calls_unboxed_result_used_gen false landed_fixes
           TP.landed_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_used_local :
  calls_unboxed_result_type_end_to_end_used_local_statement.
Proof.
  exact (CP.calls_unboxed_result_used_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_end_to_end_used_local_landed :
  calls_unboxed_result_end_to_end_used_local_landed_statement.
Proof.
  exact (CP.calls_unboxed_result_used_gen true landed_fixes
           TP.landed_fixes_sound).
Qed.
