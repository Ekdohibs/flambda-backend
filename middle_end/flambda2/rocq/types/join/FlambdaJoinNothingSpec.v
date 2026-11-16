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


(* The join that the meet calls, of environments to which nothing was
   added.

   This file is to be reviewed: the two statements. The proofs are in
   FlambdaJoinNothingProofs.v, which needs no review.

   The meet of two variants, and of two row-like structures, enters a
   scope for each alternative and joins the environments of the
   alternatives that are left, even where no name was met in them:
   it then joins environments that are each the environment of the
   meet one scope further, with nothing added. The statements about
   the fuel of the meet (FlambdaMeetDepthSpec.v) suppose of the join
   that it then answers one environment, whatever meet of expanded
   types it is given, and that this environment has the typing
   environment of the environment of the meet ([join_of_nothing],
   same file). Here that is shown of the join of the model
   ([model_join_scoped_envs], FlambdaJoinMeet.v), with any test for
   recovering relations and at any depth, where the fuel of the join
   of environments is not zero; the first statement is the body of
   that predicate at that join, under that premise on the fuel:
   - the join answers an environment that has the typing environment
     of the environment of the meet, the same whatever meet of
     expanded types it is given, and whatever the two other amounts
     of fuel ([model_join_scoped_envs_of_nothing_statement]). No
     invariant is asked of the environment, which may be bottom. The
     environment answered need not be the one given: its names to
     avoid are those of the join.
   - With no fuel for the join of environments, and an environment
     that is not bottom, the join answers nothing
     ([model_join_scoped_envs_no_fuel_statement]): the first of the
     rounds that join the types kept reads the fuel before it looks
     at what there is to join. So the premise on the fuel is needed.

   Nothing is assumed beyond the premises of each statement. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaJoinEnv FlambdaMeetEnv FlambdaMeetSpec FlambdaJoinMeet.
Import ListNotations.

(* The join of environments, one at least, that are each the
   environment of the meet one scope further with nothing added,
   answers an environment that has the typing environment of the
   environment of the meet; and it answers that one environment
   whatever meet of expanded types it is given. *)
Definition model_join_scoped_envs_of_nothing_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels)
    (env : Meet_env.t) (scoped_envs : list Typing_env.t),
    fuel_of_env fuel <> 0%nat ->
    scoped_envs <> [] ->
    (∀ scoped : Typing_env.t,
      scoped ∈ scoped_envs ->
      scoped = Typing_env.increment_scope (Meet_env.typing_env env)) ->
    ∃ result : Meet_env.t,
      Meet_env.typing_env result = Meet_env.typing_env env
      ∧ ∀ mh : meet_expanded_head,
          model_join_scoped_envs join_choices join_depth fuel mh
            (Typing_env.current_scope (Meet_env.typing_env env)) env
            (Meet_env.typing_env env) scoped_envs
          = Some result.

(* With no fuel for the join of environments, the join of such
   environments answers nothing, if the environment of the meet is
   not bottom. *)
Definition model_join_scoped_envs_no_fuel_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels)
    (mh : meet_expanded_head) (env : Meet_env.t)
    (scoped_envs : list Typing_env.t),
    fuel_of_env fuel = 0%nat ->
    Typing_env.is_bottom (Meet_env.typing_env env) = false ->
    scoped_envs <> [] ->
    (∀ scoped : Typing_env.t,
      scoped ∈ scoped_envs ->
      scoped = Typing_env.increment_scope (Meet_env.typing_env env)) ->
    model_join_scoped_envs join_choices join_depth fuel mh
      (Typing_env.current_scope (Meet_env.typing_env env)) env
      (Meet_env.typing_env env) scoped_envs
    = None.
