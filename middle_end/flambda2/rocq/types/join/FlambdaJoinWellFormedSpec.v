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


(* The types of the equations that the state of an n-way join of
   environments holds are well formed, through its first phases, and
   so are those of the equations that its last phase adds.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinWellFormedProofs.v, which needs no
   review.

   An extension that the environment of the meet is given is supposed
   to have well-formed types ([env_extension_fits], FlambdaMeetSpec.v).
   Of the two extensions that the join adds, that is the first part
   of what FlambdaJoinMeetSpec.v leaves as a premise
   ([types_fit_otherwise]). The equations of the two extensions are
   made, by the last phase, from the equations that the state holds.
   This file says what the first phases keep of the types of those
   ([equations_well_formed]): the type of each equation of the target
   made so far is well formed ([type_well_formed],
   FlambdaTypeWellFormedSpec.v), and so is each type kept to be
   joined.

   The types that the state holds have four origins: a type that a
   joined environment has for a name (the levels cut from them, the
   variables brought in, the [Simple]s that a variable made stands
   for); the type that the parameters give for a constant
   ([Join_env.expand_const]); an alias type, which is always well
   formed; and a type that the join of types answers, with or without
   the relations recovered from it. So the statements suppose that
   the joined environments have well-formed types
   ([types_well_formed], FlambdaTypingEnvInvariantSpec.v), that the
   type of a constant is well formed ([expand_const_well_formed]),
   and of the join of types that the type it answers is well formed
   ([n_way_join_type_well_formed], FlambdaJoinTypesSpec.v) and that it
   keeps [equations_well_formed] ([join_of_types_keeps_well_formed]).

   The statements:
   - the join of types of FlambdaJoinTypes.v, given the three
     functions of the join of environments, keeps
     [equations_well_formed], whatever the other four functions that
     it is given and with no premise on the parameters
     ([n_way_join_keeps_well_formed_statement]). That the type which
     it answers is well formed is
     [n_way_join_of_types_well_formed_statement] of
     FlambdaJoinTypesWithEnvSpec.v;
   - recovering the relations of a well-formed type answers a
     well-formed type
     ([recover_inverse_relations_well_formed_statement]);
   - the first phases answer a state that has [equations_well_formed]
     ([first_phases_well_formed_statement]), under the premises of
     [first_phases_true_statement] (FlambdaJoinStateSpec.v) and those
     named above;
   - each type of the two extensions that the last phase adds
     ([last_phase_adds], FlambdaJoinAddsSpec.v) is well formed, if the
     equations of the target that the state holds are
     ([last_phase_adds_well_formed_statement]);
   - of what the join is given in FlambdaJoinMeet.v, what the
     statements above suppose: the join of types
     ([n_way_join_of_types], FlambdaJoinTypesWithEnvSpec.v) keeps
     [equations_well_formed], whatever its arguments
     ([n_way_join_of_types_keeps_well_formed_statement]); the type
     that the parameters give for a constant is well formed
     ([join_parameters_well_formed_statement]); and the expansion of
     heads answers a well-formed type for a well-formed one in an
     environment whose types are well formed
     ([expand_head_for_join_well_formed_statement]), which the
     well-formedness of the type that the join of types answers
     supposes of it ([n_way_join_of_types_well_formed_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinFitsSpec FlambdaJoinAddsSpec FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesWithEnvSpec FlambdaJoinMeet.
Import ListNotations.

(* ======================================================================== *)
(* What the state holds                                                     *)
(* ======================================================================== *)

(* The types that the state holds are well formed: the type of each
   equation of the target made so far, and each type kept to be
   joined. *)
Definition equations_well_formed (s : Join_env.t) : Prop :=
  (∀ (x : variable) (ty : flambda_type),
     lookup x (Join_env.types_in_target_env s) = Some ty ->
     type_well_formed ty = true)
  ∧ ∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type)
      (index : Index.t) (t : flambda_type),
      lookup x (Join_env.types_in_joined_envs s) = Some types ->
      Keyed.lookup Index.equal index types = Some t ->
      type_well_formed t = true.

(* The type that the parameters give for a constant is well formed. *)
Definition expand_const_well_formed (p : Join_env.parameters) : Prop :=
  ∀ c : const, type_well_formed (Join_env.expand_const p c) = true.

(* ======================================================================== *)
(* The join of types                                                        *)
(* ======================================================================== *)

(* What is asked of the join of types: it keeps
   [equations_well_formed], where the joined environments have
   well-formed types. *)
Definition join_of_types_keeps_well_formed (jt : Join_env.n_way_join_type)
    : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') ->
    joined_envs_are types_well_formed s ->
    equations_well_formed s ->
    equations_well_formed s'.

(* The join of types of FlambdaJoinTypes.v, given the three functions
   of the join of environments, keeps it: whatever the other four
   functions that it is given. *)
Definition n_way_join_keeps_well_formed_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    join_of_types_keeps_well_formed
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* ======================================================================== *)
(* The relations recovered                                                  *)
(* ======================================================================== *)

(* The type that [Join_env.recover_inverse_relations] answers for a
   well-formed type is well formed. *)
Definition recover_inverse_relations_well_formed_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (n : name)
    (ty ty' : flambda_type),
    Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
    ->
    type_well_formed ty = true ->
    type_well_formed ty' = true.

(* ======================================================================== *)
(* The first phases                                                         *)
(* ======================================================================== *)

(* The first phases answer a state whose types are well formed: under
   the premises of [first_phases_true_statement]
   (FlambdaJoinStateSpec.v), with a join of types that answers
   well-formed types and keeps [equations_well_formed], a type of
   constants that is well formed, and joined environments that have
   well-formed types. *)
Definition first_phases_well_formed_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    n_way_join_type_well_formed typing_env_invariant Join_env.joined_env
      state_valid state_kinded jt ->
    join_of_types_keeps_well_formed jt ->
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    expand_const_sound p ->
    expand_const_well_formed p ->
    (∀ x : variable, knows source x -> x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       knows use x ->
       x ∈ avoid) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       typing_env_invariant use) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       typing_env_kinded use) ->
    (∀ (index : Index.t) (use : Typing_env.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       types_well_formed use) ->
    levels_found envs ->
    (∀ (index : Index.t) (use : Typing_env.t) (x : variable)
       (kind : flambda_kind) (ty : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
       Joined_envs.get_nth_joined_env envs index = Some use ->
       simple_of_kind use (Var x) kind ->
       Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types source)
       = Some (ty, binding_time_and_mode) ->
       FlambdaTypeGrammar.kind ty = kind) ->
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
    equations_well_formed s3.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

(* Each type of the two extensions that the last phase adds is well
   formed, if the types of the equations of the target that the state
   holds are. *)
Definition last_phase_adds_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    (∀ (x : variable) (ty : flambda_type),
       lookup x (Join_env.types_in_target_env s) = Some ty ->
       type_well_formed ty = true) ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations ++ relation_equations ->
      type_well_formed ty = true.

(* ======================================================================== *)
(* What the join is given in FlambdaJoinMeet.v                              *)
(* ======================================================================== *)

(* The join of types with the functions of the join of environments
   ([n_way_join_of_types], FlambdaJoinTypesWithEnvSpec.v) keeps
   [equations_well_formed]. *)
Definition n_way_join_of_types_keeps_well_formed_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    join_of_types_keeps_well_formed
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* The type that the parameters fixed there give for a constant is
   well formed. *)
Definition join_parameters_well_formed_statement : Prop :=
  ∀ join_depth : nat, expand_const_well_formed (join_parameters join_depth).

(* The expansion of heads that the join of types is given there has
   what the join of types supposes of it for well-formedness. *)
Definition expand_head_for_join_well_formed_statement : Prop :=
  FlambdaJoinTypesSpec.expand_head_well_formed expand_head_for_join.
