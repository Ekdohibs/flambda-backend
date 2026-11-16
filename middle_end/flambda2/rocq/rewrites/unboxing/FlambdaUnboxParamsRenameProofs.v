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


(* Proofs of the statements of
   FlambdaUnboxParamsRenameSpec.v: they are those of
   FlambdaUnboxParamsNamesProofs.v, with the equations of renaming that
   it takes as premises given from
   FlambdaValidatorEquivarianceProofs.v. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaNominalSpec
  FlambdaBlockLoadSpec FlambdaUnboxSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceProofs FlambdaUnboxParamsSpec
  FlambdaUnboxParamsRenameSpec FlambdaUnboxVariantsSpec
  FlambdaUnboxVariantsRenameSpec.
From OxCamlFlambda2 Require FlambdaUnboxParamsNamesProofs.
Import ListNotations.

Module Names := FlambdaUnboxParamsNamesProofs.

(* The equations that FlambdaUnboxParamsNamesProofs.v takes as
   premises. *)
Ltac known_renamed :=
  first
    [ exact var_free_b_equivariant
    | exact remove_unused_conts_equivariant
    | exact forget_vars_equivariant
    | exact known_under_let_equivariant
    | exact known_under_statics_equivariant
    | exact known_block_of_equivariant
    | exact field_loaded_equivariant
    | exact load_known_named_equivariant
    | exact forget_wrapped_vars_equivariant
    | exact numbers_under_let_equivariant
    | exact numbers_under_statics_equivariant
    | exact unwrapped_equivariant
    | exact unbox_known_named_equivariant ].

Theorem projection_of_equivariant : projection_of_equivariant_statement.
Proof. apply Names.projection_of_equivariant; known_renamed. Qed.

Theorem extend_applications_equivariant :
  extend_applications_equivariant_statement.
Proof. apply Names.extend_applications_equivariant; known_renamed. Qed.

Theorem unbox_params_knowing_equivariant :
  unbox_params_knowing_equivariant_statement.
Proof. apply Names.unbox_params_knowing_equivariant; known_renamed. Qed.

Theorem unbox_known_params_equivariant :
  unbox_known_params_equivariant_statement.
Proof. apply Names.unbox_known_params_equivariant; known_renamed. Qed.

Theorem unbox_known_params_keeps_scope :
  unbox_known_params_keeps_scope_statement.
Proof. exact Names.unbox_known_params_keeps_scope. Qed.
