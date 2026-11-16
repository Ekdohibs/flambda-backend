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


(* Proofs of the statements of FlambdaJoinMeetRenameSpec.v. Nothing
   here needs review.

   Each is one of the general statements at the model's functions: the
   functions of [Meet_env] are related by the lemmas of
   FlambdaEnvRenameMeetProofs.v; the join that the meet calls and the
   join that Simplify calls by the statements of
   FlambdaEnvRenameJoinSpec.v, with the join of types of the model
   related by the statement there; the meet by the statements of
   FlambdaMeetRenameSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinTypesWithEnvSpec
  FlambdaMeetGrammar FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaJoinMeet FlambdaMeetRenameSpec FlambdaMeetRenameProofs
  FlambdaEnvRenameJoinSpec FlambdaEnvRenameJoinProofs
  FlambdaJoinMeetRenameSpec.
From OxCamlFlambda2Types Require FlambdaEnvRenameMeetProofs.
Import ListNotations.

Module Meet_env_renamed := FlambdaEnvRenameMeetProofs.

Theorem model_join_scoped_envs_with :
  model_join_scoped_envs_with_statement.
Proof. intros join_choices join_depth fuel. reflexivity. Qed.

Theorem model_join_with : model_join_with_statement.
Proof. intros A join_choices meet_choices join_depth fuel. reflexivity. Qed.

Theorem model_meet_env_functions_related :
  model_meet_env_functions_related_statement.
Proof.
  intros r. unfold meet_env_functions_related.
  cbn [meet_env_functions Join_env.create_meet_env
       Join_env.current_typing_env Join_env.add_variable_definition
       Join_env.add_env_extension Join_env.make_bottom
       Join_env.final_typing_env Join_env.names_to_avoid
       Join_env.with_names_to_avoid].
  split; [ exact (Meet_env_renamed.create_renamed r) | ].
  split; [ exact (Meet_env_renamed.current_typing_env_renamed r) | ].
  split; [ exact (Meet_env_renamed.add_variable_definition_renamed r) | ].
  split.
  { intros env extension m m' Hm.
    exact (Meet_env_renamed.add_env_extension_renamed r m m' Hm env extension).
  }
  split; [ exact (Meet_env_renamed.make_bottom_renamed r) | ].
  split; [ exact (Meet_env_renamed.final_typing_env_renamed r) | ].
  split; [ intros env; reflexivity | ].
  exact (Meet_env_renamed.with_names_to_avoid_renamed r).
Qed.

Lemma join_of_types_with_related (r : renaming) (p p' : Join_env.parameters)
    (fuel : fuels) :
  parameters_related r p p' ->
  n_way_join_type_related r (join_of_types_with p fuel)
    (join_of_types_with p' fuel).
Proof.
  intros Hp.
  exact
    (n_way_join_of_types_related r p p' (fuel_of_env fuel)
       (fuel_of_types fuel) Hp).
Qed.

Theorem join_scoped_envs_with_related :
  join_scoped_envs_with_related_statement.
Proof.
  intros r p p' choices choices' fuel Hp Hchoices m m' Hm cut_after
    initial_env initial_tenv scoped_envs.
  exact
    (cut_and_n_way_join_related Meet_env.t meet_expanded_head r
       (rename_meet_env r) (meet_expanded_head_related r) meet_env_functions
       meet_env_functions p p' (fuel_of_env fuel) choices choices'
       (join_of_types_with p fuel) (join_of_types_with p' fuel) m m'
       cut_after initial_env initial_tenv scoped_envs
       (model_meet_env_functions_related r) Hp Hchoices
       (join_of_types_with_related r p p' fuel Hp) Hm).
Qed.

Theorem model_join_scoped_envs_related :
  model_join_scoped_envs_related_statement.
Proof.
  intros r join_depth fuel.
  split;
    apply (join_scoped_envs_with_related r (join_parameters join_depth));
    try apply join_parameters_related;
    [ apply compiler_choices_related | apply repaired_choices_related ].
Qed.

Theorem model_meet_related : model_meet_related_statement.
Proof.
  intros r meet_choices join_depth fuel Hclosed.
  destruct (model_join_scoped_envs_related r join_depth fuel)
    as [Hcompiler Hrepaired].
  split;
    [ exact
        (meet_of_types_related r _ _ meet_choices (fuel_of_meet fuel)
           Hcompiler Hclosed)
    | exact
        (meet_of_types_related r _ _ meet_choices (fuel_of_meet fuel)
           Hrepaired Hclosed) ].
Qed.

Theorem join_with_parameters_related :
  join_with_parameters_related_statement.
Proof.
  intros A r p p' choices choices' meet_choices fuel avoid inputs Hp Hchoices
    Hclosed.
  unfold join_with_parameters, join_with.
  cbn [rename_join_inputs source cut_after uses].
  exact
    (cut_and_n_way_join_with_analysis_related A Meet_env.t meet_expanded_head
       r (rename_meet_env r) (meet_expanded_head_related r)
       meet_env_functions meet_env_functions p p' (fuel_of_env fuel) choices
       choices' (join_of_types_with p fuel) (join_of_types_with p' fuel) _ _
       avoid (cut_after inputs) (source inputs) (uses inputs)
       (model_meet_env_functions_related r) Hp Hchoices
       (join_of_types_with_related r p p' fuel Hp)
       (meet_expanded_head_at_fuel_related r _ _ meet_choices
          (fuel_of_meet fuel)
          (join_scoped_envs_with_related r p p' choices choices' fuel Hp
             Hchoices)
          Hclosed)).
Qed.

Theorem model_join_related : model_join_related_statement.
Proof.
  intros A r meet_choices join_depth fuel avoid inputs Hclosed.
  exact
    (join_with_parameters_related A r (join_parameters join_depth)
       (renamed_join_parameters r join_depth) Join_env.compiler_choices
       Join_env.compiler_choices meet_choices fuel avoid inputs
       (join_parameters_related r join_depth) (compiler_choices_related r)
       Hclosed).
Qed.
