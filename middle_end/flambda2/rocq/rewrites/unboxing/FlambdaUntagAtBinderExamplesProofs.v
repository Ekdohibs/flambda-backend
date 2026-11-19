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

(* Proofs of the statements of FlambdaUntagAtBinderExamplesSpec.v, by
   computation. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaRecHandlersSpec FlambdaUntagParamsSpec FlambdaSubkindSpec
  FlambdaUntagAtBinderSpec FlambdaUntagAtBinderExamplesSpec.
Import ListNotations.

(* A conjunction of equalities between closed terms, or such an
   equality of each element of a list. *)
Ltac computed :=
  vm_compute; repeat first [ split | constructor ]; reflexivity.

Theorem late_bound : late_bound_statement.
Proof. computed. Qed.

Theorem at_head_kept : at_head_kept_statement.
Proof. computed. Qed.

Theorem repeated_bound : repeated_bound_statement.
Proof. computed. Qed.

Theorem rebound_u_bound : rebound_u_bound_statement.
Proof. computed. Qed.

Theorem rebound_x_bound : rebound_x_bound_statement.
Proof. computed. Qed.

Theorem below_binder_bound : below_binder_bound_statement.
Proof. computed. Qed.

Theorem below_binder_runs : below_binder_runs_statement.
Proof. computed. Qed.

Theorem below_param_bound : below_param_bound_statement.
Proof. computed. Qed.

Theorem in_loop_bound : in_loop_bound_statement.
Proof. computed. Qed.

Theorem in_loop_of_x_kept : in_loop_of_x_kept_statement.
Proof. computed. Qed.

Theorem in_loops_bound : in_loops_bound_statement.
Proof. computed. Qed.

Theorem loop_returned_to_kept : loop_returned_to_kept_statement.
Proof. computed. Qed.

Theorem inside_loop_kept : inside_loop_kept_statement.
Proof. computed. Qed.

Theorem inside_loop_bound : inside_loop_bound_statement.
Proof. computed. Qed.

Theorem free_u_kept : free_u_kept_statement.
Proof. computed. Qed.

Theorem wrong_kind_kept : wrong_kind_kept_statement.
Proof. computed. Qed.

Theorem wrong_kind_first_bound : wrong_kind_first_bound_statement.
Proof. computed. Qed.

Theorem u_is_param_kept : u_is_param_kept_statement.
Proof. computed. Qed.

Theorem twice_param_kept : twice_param_kept_statement.
Proof. computed. Qed.

Theorem exn_handler_kept : exn_handler_kept_statement.
Proof. computed. Qed.

Theorem jumped_to_kept : jumped_to_kept_statement.
Proof. computed. Qed.

Theorem nullable_kept : nullable_kept_statement.
Proof. computed. Qed.

Theorem variant_with_block_kept : variant_with_block_kept_statement.
Proof. computed. Qed.

Theorem nothing_stated_kept : nothing_stated_kept_statement.
Proof. computed. Qed.

Theorem constants_only_bound : constants_only_bound_statement.
Proof. computed. Qed.

Theorem entered_in_range : entered_in_range_statement.
Proof. computed. Qed.

Theorem not_entered : not_entered_statement.
Proof. computed. Qed.

Theorem untagging_defined : untagging_defined_statement.
Proof. computed. Qed.

Theorem constants_only_accepted : constants_only_accepted_statement.
Proof. computed. Qed.

Theorem seen_in_handler_bound : seen_in_handler_bound_statement.
Proof. computed. Qed.

Theorem hidden_in_body_bound : hidden_in_body_bound_statement.
Proof. computed. Qed.

Theorem hidden_in_recursive_handler_kept
  : hidden_in_recursive_handler_kept_statement.
Proof. computed. Qed.

Theorem hidden_in_recursive_body_kept
  : hidden_in_recursive_body_kept_statement.
Proof. computed. Qed.

Theorem seen_in_recursive_handler_bound
  : seen_in_recursive_handler_bound_statement.
Proof. computed. Qed.

Theorem two_params_bound : two_params_bound_statement.
Proof. computed. Qed.

Theorem reused_bound : reused_bound_statement.
Proof. computed. Qed.

Theorem reused_runs : reused_runs_statement.
Proof. computed. Qed.

Theorem reused_again : reused_again_statement.
Proof. computed. Qed.
