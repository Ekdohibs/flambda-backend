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



(* The names of an n-way join of environments: the names to avoid that
   it is given are kept, and the variables that it makes are none of
   them.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinNamesProofs.v, which needs no review.

   A variable that the compiler makes differs from every other. The
   model makes a variable outside a list of names to avoid, which the
   state of the join carries and to which the variable is added
   (FlambdaJoinEnv.v, "New names"). The invariant of the state
   ([state_valid], FlambdaJoinEnvSpec.v) says that the list has the
   variables that the environments know and those that the join has
   made. It does not say what the list has of the names that the join
   was given, which may be names that no environment of this join
   knows: the join that the meet calls is given the names to avoid of
   the meet's environment, among them the variables that a join in
   another branch of the meet has made. The contract that the meet
   has for its join ([join_scoped_envs_sound], FlambdaMeetSpec.v)
   asks that a variable that the join makes is none of the names to
   avoid of the environment given, and that none of those is
   forgotten. So, here:
   - a relation between two states ([names_kept]): the names to avoid
     of the first are among those of the second, and a variable that
     the join has made at the second was made at the first or is
     none of the names to avoid of the first;
   - the first phases keep it from the state that the join starts
     from, which has made no variable and has the names given for
     names to avoid ([first_phases_names_statement]), given a join
     of types that keeps it ([join_of_types_keeps_names]) and a
     source of new names that avoids the names given
     ([fresh_avoids], FlambdaJoinEnvSpec.v);
   - the join of types of FlambdaJoinTypes.v, given the three
     functions of the join of environments, keeps it
     ([n_way_join_keeps_names_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaJoinEnv FlambdaJoinEnvSpec FlambdaJoinTypes.
Import ListNotations.

(* Between two states of a join: the names to avoid of the first are
   among those of the second, and a variable that the join has made
   at the second was made at the first or is none of the names to
   avoid of the first. *)
Definition names_kept (s s' : Join_env.t) : Prop :=
  (∀ x : variable, x ∈ names_to_avoid s -> x ∈ names_to_avoid s')
  ∧ ∀ x : variable,
      is_existential s' x -> is_existential s x ∨ ~ x ∈ names_to_avoid s.

(* What is asked of the join of types: it keeps the names. *)
Definition join_of_types_keeps_names (jt : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') -> names_kept s s'.

(* The join of types of FlambdaJoinTypes.v, given the three functions
   of the join of environments, keeps them, if a new name is none of
   the names to avoid. (By [n_way_join_keeps_statement],
   FlambdaJoinTypesKeepsSpec.v.) *)
Definition n_way_join_keeps_names_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    fresh_avoids p ->
    join_of_types_keeps_names
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* The first phases, from the state that the join starts from: the
   names that the join was given are among the names to avoid of the
   state that they answer, and a variable that the join has made is
   none of them. *)
Definition first_phases_names_statement : Prop :=
  ∀ (jt : Join_env.n_way_join_type) (choices : Join_env.choices)
    (p : Join_env.parameters) (fuel : nat) (avoid : list variable)
    (source : Typing_env.t) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    join_of_types_keeps_names jt ->
    fresh_avoids p ->
    Join_env.join_aliases_into_bindings p fuel
      (Join_env.create envs
         (Bindings_in_target_env.from_source_env avoid source))
      envs
    = Some s1 ->
    Join_env.n_way_join_delayed_equations fuel choices jt s1 []
      (Join_env.new_equations_in_joined_envs s1
         (Join_env.create envs
            (Bindings_in_target_env.from_source_env avoid source)))
    = Some (relations2, s2) ->
    Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
      relations2
      (Join_env.new_definitions_of_existentials s2
         (Join_env.create envs
            (Bindings_in_target_env.from_source_env avoid source)))
    = Some (relations3, s3) ->
    (∀ x : variable, x ∈ avoid -> x ∈ names_to_avoid s3)
    ∧ ∀ x : variable, is_existential s3 x -> ~ x ∈ avoid.
