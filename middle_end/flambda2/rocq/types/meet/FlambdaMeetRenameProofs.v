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


(* Proofs of the statements of FlambdaMeetRenameSpec.v. Nothing here
   needs review. The laws of the functions of the meet are in
   FlambdaEnvRenameMeetProofs.v, FlambdaEnvRenameMeetPartsProofs.v
   and FlambdaEnvRenameMeetProductsProofs.v (the environment of the
   meet, the functions that put a meet together, the products and the
   parts that are not values), FlambdaMeetRenameScopesProofs.v and
   FlambdaMeetRenameRowLikeProofs.v (the meet of two alternatives and
   of two row-like structures), FlambdaEnvRenameMeetValuesProofs.v
   (the heads of kind value, given the rows) and
   FlambdaMeetRenameHeadsProofs.v (the rows, and the meet of two
   expanded types). *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaProversRenameSpec FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec FlambdaMeetRenameSpec
  FlambdaEnvRenameMeetProofs FlambdaEnvRenameMeetPartsProofs
  FlambdaEnvRenameMeetProductsProofs FlambdaMeetRenameScopesProofs
  FlambdaMeetRenameRowLikeProofs FlambdaMeetRenameHeadsProofs.
Import ListNotations.

Theorem compiler_choices_closed : compiler_choices_closed_statement.
Proof.
  exact FlambdaEnvRenameMeetProductsProofs.compiler_choices_closed.
Qed.

Theorem repaired_choices_closed : repaired_choices_closed_statement.
Proof.
  exact FlambdaEnvRenameMeetProductsProofs.repaired_choices_closed.
Qed.

(* ======================================================================== *)
(* Two joins, two meets, one renaming                                       *)
(* ======================================================================== *)

Theorem meet_expanded_head_equivariant_iff_related :
  meet_expanded_head_equivariant_iff_related_statement.
Proof. intros m. split; intros H r; exact (H r). Qed.

Theorem meet_function_equivariant_iff_related :
  meet_function_equivariant_iff_related_statement.
Proof. intros meet. split; intros H r; exact (H r). Qed.

Theorem join_related_gives_equivariant :
  join_related_gives_equivariant_statement.
Proof.
  intros join H r m cut_after env tenv scoped_envs Hm.
  exact (H r m m (Hm r) cut_after env tenv scoped_envs).
Qed.

Theorem meet_expanded_head_at_fuel_related :
  meet_expanded_head_at_fuel_related_statement.
Proof.
  intros r join join' choices fuel Hjoin Hchoices.
  induction fuel as [ | fuel IH]; [ intros env e1 e2; reflexivity | ].
  cbn [Meet.meet_expanded_head].
  exact (meet_expanded_head_step_renamed r join join' choices _ _
           (Hjoin _ _ IH) IH Hchoices).
Qed.

Theorem meet_of_types_related : meet_of_types_related_statement.
Proof.
  intros r join join' choices fuel Hjoin Hchoices.
  apply exposed_meet_renamed.
  exact (meet_expanded_head_at_fuel_related r join join' choices fuel Hjoin
           Hchoices).
Qed.

Theorem typing_env_functions_related :
  typing_env_functions_related_statement.
Proof.
  intros r join join' choices fuel Hjoin Hchoices.
  pose proof
    (meet_expanded_head_at_fuel_related r join join' choices fuel Hjoin
       Hchoices)
    as Hmeet.
  split; [ | split ].
  - intros tenv name ty.
    exact (typing_env_add_equation_renamed r join join' choices fuel Hmeet
             tenv name ty).
  - intros tenv extension.
    exact (typing_env_add_env_extension_renamed r join join' choices fuel
             Hmeet tenv extension).
  - intros tenv params param_types.
    exact (typing_env_add_equations_on_params_renamed r join join' choices
             fuel Hmeet tenv params param_types).
Qed.

(* ======================================================================== *)
(* A join that commutes with every renaming                                 *)
(* ======================================================================== *)

Theorem meet_expanded_head_at_fuel_equivariant :
  meet_expanded_head_at_fuel_equivariant_statement.
Proof.
  intros join choices fuel Hjoin Hchoices.
  induction fuel as [ | fuel IH]; [ intros r env e1 e2; reflexivity | ].
  intros r. cbn [Meet.meet_expanded_head].
  exact (meet_expanded_head_step_renamed r join join choices _ _
           (fun cut_after env tenv scoped_envs =>
              Hjoin r _ cut_after env tenv scoped_envs IH)
           (IH r) Hchoices).
Qed.

Theorem meet_of_types_equivariant : meet_of_types_equivariant_statement.
Proof.
  intros join choices fuel Hjoin Hchoices r.
  apply (exposed_meet_renamed r join join choices fuel).
  exact (meet_expanded_head_at_fuel_equivariant join choices fuel Hjoin
           Hchoices r).
Qed.

Theorem meet_to_prove_equivariant : meet_to_prove_equivariant_statement.
Proof.
  intros join fuel Hjoin.
  exact (meet_of_types_equivariant join repaired_choices fuel Hjoin
           repaired_choices_closed).
Qed.

Theorem compiler_meet_equivariant : compiler_meet_equivariant_statement.
Proof.
  intros join fuel Hjoin.
  exact (meet_of_types_equivariant join compiler_choices fuel Hjoin
           compiler_choices_closed).
Qed.

Theorem forgetful_join_equivariant : forgetful_join_equivariant_statement.
Proof. intros r m cut_after env tenv scoped_envs Hm. reflexivity. Qed.

Theorem meet_with_forgetful_join_equivariant :
  meet_with_forgetful_join_equivariant_statement.
Proof.
  intros fuel.
  exact (meet_to_prove_equivariant forgetful_join fuel
           forgetful_join_equivariant).
Qed.

Theorem typing_env_functions_equivariant :
  typing_env_functions_equivariant_statement.
Proof.
  intros join choices fuel Hjoin Hchoices.
  pose proof
    (meet_expanded_head_at_fuel_equivariant join choices fuel Hjoin Hchoices)
    as Hmeet.
  split; [ | split ].
  - intros r tenv name ty.
    exact (typing_env_add_equation_renamed r join join choices fuel
             (Hmeet r) tenv name ty).
  - intros r tenv extension.
    exact (typing_env_add_env_extension_renamed r join join choices fuel
             (Hmeet r) tenv extension).
  - intros r tenv params param_types.
    exact (typing_env_add_equations_on_params_renamed r join join choices
             fuel (Hmeet r) tenv params param_types).
Qed.
