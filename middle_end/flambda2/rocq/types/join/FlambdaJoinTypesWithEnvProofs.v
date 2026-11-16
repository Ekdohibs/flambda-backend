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


(* Proofs of the statements of FlambdaJoinTypesWithEnvSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeEqualSpec FlambdaTypeEqualProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvInvariantProofs
  FlambdaTypingEnvKindedSpec FlambdaTypingEnvKindedProofs FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinEnvProofs
  FlambdaJoinStateSpec FlambdaJoinStateProofs FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesProofs FlambdaJoinTypesKeepsSpec
  FlambdaJoinTypesKeepsProofs FlambdaJoinTypesWithEnvSpec.
Import ListNotations.

(* The environments joined have [constants_of_kind]: the invariant has
   the aliases consistent, and the environments are kinded. *)
Lemma joined_envs_have_constants_of_kind :
  joined_envs_constants_of_kind typing_env_invariant Join_env.joined_env
    state_valid.
Proof.
  intros s index use _ Hwf Hkinded Huse.
  apply typing_env_kinded_constants_of_kind;
    [ | exact (Hkinded index use Huse) ].
  destruct (Hwf index use Huse) as [Htest _].
  apply typing_env_well_formed_iff in Htest.
  destruct Htest as (_ & _ & Hconsistent & _). exact Hconsistent.
Qed.

Theorem n_way_join_of_types_sound : n_way_join_of_types_sound_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate
    fuel Hfresh Heqb Hexpand Hreduce.
  unfold n_way_join_of_types.
  apply FlambdaJoinTypesProofs.n_way_join_sound.
  - split; [ exact state_valid_has_witnesses | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.n_way_join_simples_sound p fuel_of_env Hfresh)
      | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.import_type_sound p fuel_of_env Hfresh)
      | ].
    exact
      (FlambdaJoinEnvProofs.n_way_join_env_extension_sound p fuel_of_env
         Hfresh Heqb).
  - exact joined_envs_have_constants_of_kind.
  - exact Hexpand.
  - exact Hreduce.
  - intros t t' Hequal statics vars v.
    exact (type_equal_satisfied statics vars v t t' Hequal).
  - intros shape shape' Hequal. exact (proj1 (block_shape_eqb_eq _ _) Hequal).
Qed.

Theorem n_way_join_of_types_keeps_definitions
  : n_way_join_of_types_keeps_definitions_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate fuel.
  exact
    (n_way_join_keeps_definitions p fuel_of_env fuel expand_head
       reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb).
Qed.

Theorem n_way_join_of_types_true : n_way_join_of_types_true_statement.
Proof.
  intros source p fuel_of_env expand_head reduce_head_of_kind_naked_immediate
    fuel Hfresh Hparameters Heqb Hexpand Hreduce.
  unfold n_way_join_of_types.
  apply FlambdaJoinTypesProofs.n_way_join_sound.
  - split; [ exact (state_true_has_witnesses source) | ].
    split;
      [ exact (n_way_join_simples_true p fuel_of_env source Hfresh Hparameters)
      | ].
    split;
      [ exact (import_type_true p fuel_of_env source Hfresh Hparameters) | ].
    exact
      (n_way_join_env_extension_true p fuel_of_env source Hfresh Hparameters
         Heqb).
  - intros s index use [Hvalid _].
    exact (joined_envs_have_constants_of_kind s index use Hvalid).
  - exact Hexpand.
  - exact Hreduce.
  - intros t t' Hequal statics vars v.
    exact (type_equal_satisfied statics vars v t t' Hequal).
  - intros shape shape' Hequal. exact (proj1 (block_shape_eqb_eq _ _) Hequal).
Qed.

Theorem n_way_join_of_types_equations_kinded
  : n_way_join_of_types_equations_kinded_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate
    fuel Hfresh Heqb Hexpand Hequations Hreduce.
  unfold n_way_join_of_types.
  apply
    (FlambdaJoinTypesProofs.n_way_join_equations_kinded typing_env_invariant
       _ Join_env.joined_env Join_env.definitions_of_existentials
       kind_in_target_env state_valid state_kinded).
  - split; [ exact state_valid_has_witnesses | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.n_way_join_simples_sound p fuel_of_env Hfresh)
      | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.import_type_sound p fuel_of_env Hfresh)
      | ].
    exact
      (FlambdaJoinEnvProofs.n_way_join_env_extension_sound p fuel_of_env
         Hfresh Heqb).
  - exact
      (FlambdaJoinEnvProofs.import_type_equations_kinded p fuel_of_env Hfresh).
  - exact
      (FlambdaJoinEnvProofs.n_way_join_env_extension_equations_kinded p
         fuel_of_env Hfresh).
  - exact joined_envs_have_constants_of_kind.
  - exact Hexpand.
  - exact Hequations.
  - exact Hreduce.
  - intros t t' Hequal statics vars v.
    exact (type_equal_satisfied statics vars v t t' Hequal).
  - intros shape shape' Hequal. exact (proj1 (block_shape_eqb_eq _ _) Hequal).
Qed.

Theorem n_way_join_of_types_well_formed
  : n_way_join_of_types_well_formed_statement.
Proof.
  intros p fuel_of_env expand_head reduce_head_of_kind_naked_immediate
    fuel Hfresh Heqb Hexpand Hexpand_wf Hreduce.
  unfold n_way_join_of_types.
  apply
    (FlambdaJoinTypesProofs.n_way_join_well_formed typing_env_invariant
       _ Join_env.joined_env Join_env.definitions_of_existentials
       kind_in_target_env state_valid state_kinded).
  - split; [ exact state_valid_has_witnesses | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.n_way_join_simples_sound p fuel_of_env Hfresh)
      | ].
    split;
      [ exact
          (FlambdaJoinEnvProofs.import_type_sound p fuel_of_env Hfresh)
      | ].
    exact
      (FlambdaJoinEnvProofs.n_way_join_env_extension_sound p fuel_of_env
         Hfresh Heqb).
  - (* The extension answered is one of those given, or the empty
       one. *)
    intros s s' extensions joined _ Hgiven Hjoin.
    destruct
      (n_way_join_env_extension_answers_given p fuel_of_env s s' extensions
         joined Hjoin)
      as [-> | [index Hin]];
      [ reflexivity | exact (Hgiven index joined Hin) ].
  - exact joined_envs_have_constants_of_kind.
  - exact Hexpand.
  - exact Hexpand_wf.
  - exact Hreduce.
  - intros t t' Hequal statics vars v.
    exact (type_equal_satisfied statics vars v t t' Hequal).
  - intros shape shape' Hequal. exact (proj1 (block_shape_eqb_eq _ _) Hequal).
Qed.
