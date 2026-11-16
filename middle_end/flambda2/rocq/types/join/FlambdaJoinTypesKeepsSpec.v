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


(* What the n-way join of types does to the state of the join of
   environments: nothing but what the functions of the environment
   side do.

   To be reviewed: the two statements of this file. The functions are
   FlambdaJoinTypes.v.

   The join of types threads the state of the join of environments,
   and changes it only by calling the three functions of the
   environment side that it is given: the join of [Simple]s, the
   import of a type, and the join of environment extensions. So a
   relation between states that those three keep, and that is
   reflexive and transitive, is kept by the join of types: between the
   state that it is given and the state that it answers, whatever the
   types and whatever the state. Nothing is asked of the state, of the
   types or of the other functions that the join of types is given
   ([Expand_head], the reduction of naked immediates, the two tests of
   equality): this is about where the state goes, not about what the
   answer means.

   What it is for. The contract of the join of types
   ([n_way_join_type_sound], FlambdaJoinSpec.v) says that the state
   grows only at a valid state and for types in scope and kinded. The
   statements about the rounds of the join of environments need less
   of the state and with no premise: that the environments joined and
   the definitions made are kept, at every state. That is this
   statement at that relation, and the second statement of this file
   says so: the join of types, given the three functions of the join
   of environments (FlambdaJoinEnv.v), keeps the definitions
   ([join_of_types_keeps_definitions], FlambdaJoinEnvSpec.v), which is
   what the statements about the rounds suppose of the join of types
   that they are given.

   One place deserves a look: [n_way_join_variant] answers the state
   of before the extensions were joined when both joined extensions
   are empty. That state is one that the function went through, so
   the relation holds of it too.

   The proofs are in FlambdaJoinTypesKeepsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaJoinEnv FlambdaJoinEnvSpec
  FlambdaJoinTypes.
Import ListNotations.

(* A reflexive and transitive relation on states that the three
   functions of the environment side keep is kept by the join of
   types. *)
Definition n_way_join_keeps_statement : Prop :=
  ∀ (state : Type) (keeps : state -> state -> Prop)
    (joined_env : state -> Index.t -> option Typing_env.t)
    (n_way_join_simples :
       state -> flambda_kind -> list (Index.t * simple)
       -> option (Or_bottom.t simple * state))
    (import_type : state -> flambda_type -> option state)
    (join_env_n_way_join_env_extension :
       state -> list (Index.t * env_extension)
       -> option (Or_bottom.t (env_extension * state)))
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool) (fuel : nat),
    (∀ s : state, keeps s s) ->
    (∀ s1 s2 s3 : state, keeps s1 s2 -> keeps s2 s3 -> keeps s1 s3) ->
    (∀ (s s' : state) (kind : flambda_kind)
       (simples : list (Index.t * simple)) (result : Or_bottom.t simple),
       n_way_join_simples s kind simples = Some (result, s') -> keeps s s') ->
    (∀ (s s' : state) (ty : flambda_type),
       import_type s ty = Some s' -> keeps s s') ->
    (∀ (s s' : state) (extensions : list (Index.t * env_extension))
       (joined : env_extension),
       join_env_n_way_join_env_extension s extensions
       = Some (Or_bottom.Ok (joined, s')) ->
       keeps s s') ->
    ∀ (s s' : state) (ts : list (Index.t * flambda_type))
      (joined : Or_unknown.t flambda_type),
      n_way_join joined_env n_way_join_simples import_type
        join_env_n_way_join_env_extension expand_head
        reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel s
        ts
      = Some (joined, s') ->
      keeps s s'.

(* The join of types, given the three functions of the join of
   environments, keeps the definitions of the state: at every state,
   for every list of types, and whatever the other four functions that
   it is given. The two amounts of fuel are that of the functions of
   the join of environments and that of the join of types. *)
Definition n_way_join_keeps_definitions_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    join_of_types_keeps_definitions
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).
