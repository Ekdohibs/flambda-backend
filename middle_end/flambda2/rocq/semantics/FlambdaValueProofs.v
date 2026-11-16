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

(* Proofs of the statements of FlambdaValueSpec.v. Nothing here needs
   review. FlambdaSemanticsProofs.v exports this file. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec.

(* By cases on the value and, for a symbol, on what it is bound to. No
   case names a constructor of [value] by its position. *)
Theorem observations_disjoint : observations_disjoint_statement.
Proof.
  intros statics v. unfold closure_of, block_of, boxed_number_of, string_of.
  destruct v; try (repeat split; congruence).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) as [allocated | ];
      [ destruct allocated | ]
  end;
    repeat split; congruence.
Qed.

Theorem array_of_disjoint : array_of_disjoint_statement.
Proof.
  intros statics v.
  unfold array_of, closure_of, block_of, boxed_number_of, string_of.
  destruct v; try congruence; try (intros _; repeat split; reflexivity).
  match goal with
  | |- context [lookup ?s (static_values statics)] =>
    destruct (lookup s (static_values statics)) as [allocated | ];
      [ destruct allocated | ]
  end;
    try congruence.
  intros _. repeat split; reflexivity.
Qed.
