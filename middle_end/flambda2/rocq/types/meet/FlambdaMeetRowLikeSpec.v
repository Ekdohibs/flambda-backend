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


(* What the meets of row-like structures keep: the rows of the blocks
   of a variant, and the rows of the closures of a set.

   This file is to be reviewed: the statement and the proposition,
   which are about [meet_row_like_for_blocks] and
   [meet_row_like_for_closures] of FlambdaMeet.v. They are in the form
   of FlambdaMeetHeadsSpec.v ([type_part_meet_ok]): a row fits an
   environment, and is true of the value in hand, as the type that it
   makes does. They are the two premises of
   [meet_value_heads_ok_statement] (FlambdaMeetHeadsSpec.v), and with
   them the meet of two heads of kind value is right. The proofs are
   in FlambdaMeetRowLikeProofs.v, which needs no review.

   What they suppose, beyond what the statements of the heads do: the
   choice of the meet that is to be proved lifts the variables of a
   case that is alone ([lift_variables_of_single_case]; FlambdaMeet.v
   says what the compiler does).

   The first, of the rows of blocks, is proved
   ([meet_row_like_for_blocks_ok_statement]). The second, of the rows
   of closures, is not proved here, and is not named as a statement.
   What is proved for both, in FlambdaMeetRowLikeProofs.v, is the part
   that the two share ([row_like_body_ok]): [meet_row_like] is right
   whatever its cases are of, given that the meet of the products of
   two cases is, and what the indexes and the map of the cases are to
   have. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvKindedSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetStepsSpec FlambdaMeetEnvSpec FlambdaMeetScopesSpec
  FlambdaMeetProductsSpec FlambdaMeetHeadsSpec.
Import ListNotations.

(* The meet of the rows of two sets of blocks is right, for any
   value. *)
Definition meet_row_like_for_blocks_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    lift_variables_of_single_case choices = true ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok blocks_type v
      (meet_row_like_for_blocks join choices mh).

(* The meet of the rows of two sets of closures is right, for any
   value. *)
Definition meet_row_like_for_closures_right : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    lift_variables_of_single_case choices = true ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok (create_closures Alloc_mode.Heap) v
      (meet_row_like_for_closures join choices mh).
