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

(* The meet of the rows of two sets of closures; and the contract of
   the meet, from the contract of the join.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaMeetClosuresProofs.v, which needs no review.

   - The meet of the rows of the closures of two sets is right
     ([meet_row_like_for_closures_ok_statement]), in the form of
     FlambdaMeetHeadsSpec.v ([type_part_meet_ok]): a row fits an
     environment, and is true of the value in hand, as the type of
     closures that it makes does. It is the proposition
     [meet_row_like_for_closures_right] of FlambdaMeetRowLikeSpec.v,
     and with the rows of blocks, proved there, it is what the meet of
     two heads of kind value was waiting for
     (FlambdaMeetHeadsSpec.v, [meet_value_heads_ok_statement]). It
     supposes what the rows of blocks suppose: the join's contract,
     the two choices of the meet that is to be proved (a variable
     that the environment knows is not defined again; the variables
     of a case that is alone are lifted), and the two contracts of
     the meet of expanded types.

   - So the meet that is to be proved has its contract
     ([meet_to_prove_sound_statement]): [transcription_sound
     meet_to_prove] of FlambdaMeetSpec.v. For every join that has
     [join_scoped_envs_sound] and every fuel, [meet_to_prove join
     fuel] has [meet_contract]. This is what FlambdaMeetSpec.v says
     is to be proved of the transcribed meet.

   - And the functions that add to a typing environment, made of that
     meet, have their contracts
     ([typing_env_functions_of_meet_to_prove_statement]):
     [Typing_env.add_equation] and [Typing_env.add_env_extension] as
     FlambdaMeetSpec.v states them, and
     [Typing_env.add_equations_on_params] for parameters that the
     environment knows (FlambdaMeetFuelSpec.v says why).

   - And the meet of expanded types that it is made of has its two
     contracts at every fuel
     ([meet_to_prove_expanded_heads_statement]). They are the
     premises of the statements of FlambdaMeetFuelSpec.v about the
     meet of types and the functions that add to a typing
     environment; the two statements above are those statements with
     this premise discharged.

   What these do not say. The contracts say nothing of a meet that
   does not answer: the fuel may run out, and it is not shown that
   some fuel is enough. The meet is the one with the choices that are
   to be proved, which is not the compiler's on the three points of
   [meet_choices] (FlambdaMeet.v), nor on the one state at which the
   model stops and the compiler does not (FlambdaMeetEnv.v). And the
   join is any function that has [join_scoped_envs_sound]; that the
   compiler's join has it is not shown here, and it has it only for
   the runs that the conditions of the join's own statements
   describe. *)

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
  FlambdaMeetProductsSpec FlambdaMeetHeadsSpec FlambdaMeetRowLikeSpec
  FlambdaMeetImmediateSpec FlambdaMeetFuelSpec.
Import ListNotations.

(* The meet of the rows of two sets of closures is right, for any
   value. *)
Definition meet_row_like_for_closures_ok_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head) (v : value),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    lift_variables_of_single_case choices = true ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    type_part_meet_ok (create_closures Alloc_mode.Heap) v
      (meet_row_like_for_closures join choices mh).

(* The meet that is to be proved has its contract, with a join that
   has its own, whatever the fuel. *)
Definition meet_to_prove_sound_statement : Prop :=
  transcription_sound meet_to_prove.

(* The functions that add to a typing environment, made of that meet. *)
Definition typing_env_functions_of_meet_to_prove_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join ->
    add_equation_sound (Flambda2_types.add_equation join repaired_choices fuel)
    ∧ add_env_extension_sound
        (Flambda2_types.add_env_extension join repaired_choices fuel)
    ∧ add_equations_on_params_sound_where_known
        (Flambda2_types.add_equations_on_params join repaired_choices fuel).

(* The meet of expanded types that the meet that is to be proved is
   made of has its two contracts, with a join that has its own,
   whatever the fuel. *)
Definition meet_to_prove_expanded_heads_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join ->
    meet_expanded_head_keeps
      (Meet.meet_expanded_head join repaired_choices fuel)
    ∧ meet_expanded_head_sound
        (Meet.meet_expanded_head join repaired_choices fuel).
