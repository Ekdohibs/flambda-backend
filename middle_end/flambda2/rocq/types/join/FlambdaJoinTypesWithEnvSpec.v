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


(* The n-way join of types with the functions of the join of
   environments: the two halves of the n-way join put together.

   To be reviewed: the definition and the five statements of this
   file.

   FlambdaJoinTypesSpec.v says what the type traversal guarantees
   against an interface (FlambdaJoinSpec.v): of the state of the join
   of environments it knows three readings and two invariants, and of
   the functions of the environment side it knows their contracts.
   FlambdaJoinEnvSpec.v says that the functions of FlambdaJoinEnv.v
   meet those contracts, at its own readings and invariants of the
   state and at the invariant of typing environments
   (FlambdaTypingEnvInvariantSpec.v). This file puts the two together.
   [n_way_join_of_types] is the join of types that is given those
   functions and the two tests of equality of FlambdaTypeEqualSpec.v;
   it is what the join of environments is to be given for its rounds.
   The first statement is that it meets the contract of the join of
   types ([n_way_join_type_sound], FlambdaJoinSpec.v) at those readings
   and invariants, whatever the two numbers that bound the recursions.
   The second, which has no premise, is that it keeps the definitions
   of the state ([join_of_types_keeps_definitions],
   FlambdaJoinEnvSpec.v), which the statements about the rounds of the
   join of environments suppose of the join of types that they are
   given: it is the statement of FlambdaJoinTypesKeepsSpec.v, said of
   this function. The third is the first at another invariant of the
   state, [state_true source] (FlambdaJoinStateSpec.v), which says what
   the state means (the equations of the target made so far are true
   at every input, for a join whose source is [source]) and not only
   what its names are. It is the hypothesis under which the rounds of
   the join of environments are proved to keep the equations true
   ([first_phases_true_statement] of that file). Its premises are
   those of the first and one more, [parameters_type_eqb_sound]: it is
   about the equality of types that the join of environments is given
   for its own use, which is not the equality that the traversal
   uses. The fourth is that the function meets the contract for the
   names of the equations inside the type that it answers
   ([n_way_join_type_equations_kinded], FlambdaJoinSpec.v), which the
   target of a join is to have at their kinds if a meet is to be made
   in it. It has the premises of the first and one more,
   [expand_head_equations_kinded] (FlambdaJoinTypesSpec.v). The
   invariant of the state is only a premise of that contract, so the
   statement gives the same at [state_true source].
   The fifth is that the type that the function answers is well
   formed ([n_way_join_type_well_formed], FlambdaJoinTypesSpec.v). It
   has the premises of the first and one more,
   [expand_head_well_formed] (FlambdaJoinTypesSpec.v). What that file
   assumes of the join of extensions is true of the function of the
   join of environments, which answers an extension that it was
   given or the empty one
   ([n_way_join_env_extension_answers_given_statement],
   FlambdaJoinEnvSpec.v).

   What is still assumed, each a premise of the first statement:
   - of [Expand_head] and of the reduction of heads of naked
     immediates, what FlambdaJoinTypesSpec.v assumes of them
     ([expand_head_sound], at the invariant of typing environments,
     and [reduce_head_of_kind_naked_immediate_sound]). They are
     functions of the workstream on meet, whose files are not under
     this one, and stay arguments of [n_way_join_of_types];
   - what the theorems about the functions of the environment side
     assume (FlambdaJoinEnvSpec.v): [fresh_avoids], of the source of
     new names, and [env_extension_eqb_sound], of the equality of
     extensions that the join of environments is given.

   What is assumed in FlambdaJoinTypesSpec.v and not here:
   - the premise about the constants of the environments joined
     ([joined_envs_constants_of_kind]). The invariant of typing
     environments has the aliases of an environment consistent, an
     environment that is kinded and has consistent aliases has
     [constants_of_kind] (FlambdaTypingEnvKindedSpec.v), and the
     environments joined are kinded by a premise of the contract;
   - what is assumed of the two tests of equality: those of
     FlambdaTypeEqualSpec.v have it.

   The proofs are in FlambdaJoinTypesWithEnvProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeEqualSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinTypes
  FlambdaJoinTypesSpec.
Import ListNotations.

(* The join of types that the join of environments is given: the
   traversal of FlambdaJoinTypes.v with the three functions of the
   join of environments and the two tests of equality of the grammar.
   [fuel_of_env] bounds the recursion of those three functions and
   [fuel] the depth of the join of types. *)
Definition n_way_join_of_types (p : Join_env.parameters)
    (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat) : Join_env.n_way_join_type :=
  n_way_join Join_env.joined_env (Join_env.n_way_join_simples p fuel_of_env)
    (Join_env.import_type p fuel_of_env)
    (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
    reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel.

(* It meets the contract that the join of environments assumes of the
   join of types that it is given. *)
Definition n_way_join_of_types_sound_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    fresh_avoids p ->
    env_extension_eqb_sound p ->
    expand_head_sound typing_env_invariant expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env state_valid
      state_kinded
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* It keeps the definitions of the state, at every state and for every
   list of types. *)
Definition n_way_join_of_types_keeps_definitions_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    join_of_types_keeps_definitions
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* It meets the contract at the invariant of the state that says what
   the state means, for a join of any source. *)
Definition n_way_join_of_types_true_statement : Prop :=
  ∀ (source : Typing_env.t) (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    env_extension_eqb_sound p ->
    expand_head_sound typing_env_invariant expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* It meets the contract for the names of the equations inside the
   type that it answers. *)
Definition n_way_join_of_types_equations_kinded_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    fresh_avoids p ->
    env_extension_eqb_sound p ->
    expand_head_sound typing_env_invariant expand_head ->
    expand_head_equations_kinded expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    n_way_join_type_equations_kinded typing_env_invariant Join_env.joined_env
      kind_in_target_env state_valid state_kinded
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* The type that it answers is well formed, if the types given are
   and the environments joined have well-formed types. *)
Definition n_way_join_of_types_well_formed_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    fresh_avoids p ->
    env_extension_eqb_sound p ->
    expand_head_sound typing_env_invariant expand_head ->
    expand_head_well_formed expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    n_way_join_type_well_formed typing_env_invariant Join_env.joined_env
      state_valid state_kinded
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).
