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

(* A meet adds no variable that may occur in terms.

   This file is to be reviewed: the definition and the statement. The
   proof is in FlambdaMeetTermsProofs.v, which needs no review.

   What it is for. After a meet, Simplify goes on in the environment
   that the meet answers, with the terms that it had before: so in a
   [Switch] that it prunes with the type of the scrutinee
   (FlambdaSimplifySwitchSpec.v). A meet runs nothing: it gives the
   machine no binding. What satisfies the environment given is known
   to bind the variables that may occur in terms under it
   ([in_scope_in_terms] and [bound_in_terms], FlambdaTypingEnvSpec.v);
   of a variable that could newly occur in terms nothing would say
   that it is bound. So the environment answered must have no
   variable that may occur in terms and that the environment given
   has not. That is an inclusion of the variables that may occur in
   terms, and all that is stated: nothing is said of what the machine
   binds beside them.

   The definition ([meet_keeps_terms]) is of any function of the type
   of the meet, as the contracts of FlambdaMeetSpec.v are: a variable
   that may occur in terms under the environment answered may occur
   in terms under the environment given. It says nothing of the
   variables that may not occur in terms: a meet defines such
   variables, and the join that it calls makes them.

   The statement ([meet_to_prove_keeps_terms_statement]): the
   transcribed meet that the contract is proved of ([meet_to_prove],
   FlambdaMeetSpec.v) has it, at every fuel and for every join that
   has its contract ([join_scoped_envs_sound]). It follows from two
   clauses of the contracts of that file. A variable that a meet of
   expanded types has defined may not occur in terms
   ([new_names_avoided], a clause of [meet_expanded_head_keeps]);
   and a variable that the environment given knew has in the
   environment answered the name mode that it had
   ([meet_env_extends]). The first asks the same of the join: a
   variable of its result that the first environment does not know
   may not occur in terms. The join that forgets has that contract
   ([forgetful_join_sound_statement], FlambdaMeetExamplesSpec.v), so
   the statement has an instance with no premise.

   The same conclusion under another hypothesis on the join
   ([meet_to_prove_keeps_terms_of_keeps_statement]): that it has
   the conclusions of its contract that are not about values, with
   every meet of expanded types that keeps what a meet keeps
   ([join_scoped_envs_keeps], FlambdaMeetKeepsSpec.v). It is not a
   weaker hypothesis than the contract, nor a stronger one: the
   contract gives its conclusions only with a meet that loses no
   value, and this one gives fewer conclusions with more meets. The
   two clauses that the statement follows from are in the half of
   the meet's contract that such a join gives it. It is the
   hypothesis that the join of the model has
   (FlambdaJoinMeetFitsSpec.v).

   The compiler. The meet defines variables in two places, both with
   the name mode of the variables that exist only in types
   ([Name_mode.in_types]): [add_defined_vars], and the fold over the
   extra variables after the join in [meet_row_like]
   (middle_end/flambda2/types/meet_and_n_way_join.ml). The n-way
   join defines the variables of its target with the same name mode
   ([define_or_eliminate_variables_and_add_equations],
   types/env/join_env.ml). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetKeepsSpec.
Import ListNotations.

(* A meet adds no variable that may occur in terms: a variable that
   may occur in terms under the environment that it answers may occur
   in terms under the environment that it was given. *)
Definition meet_keeps_terms (meet : meet_function) : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (t1 t2 t : flambda_type),
    meet_inputs_well_formed tenv t1 t2 ->
    meet tenv t1 t2 = Some (Or_bottom.Ok (t, tenv')) ->
    ∀ x : variable, in_scope_in_terms tenv' x -> in_scope_in_terms tenv x.

(* The meet that the contract is proved of has it, with a join that
   has its contract, whatever the fuel. *)
Definition meet_to_prove_keeps_terms_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join -> meet_keeps_terms (meet_to_prove join fuel).

(* The same conclusion, with a join that has the conclusions of its
   contract that are not about values with every meet that keeps
   ([join_scoped_envs_keeps]): another hypothesis than
   [join_scoped_envs_sound], which neither implies nor is implied by
   it. *)
Definition meet_to_prove_keeps_terms_of_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_keeps join -> meet_keeps_terms (meet_to_prove join fuel).
