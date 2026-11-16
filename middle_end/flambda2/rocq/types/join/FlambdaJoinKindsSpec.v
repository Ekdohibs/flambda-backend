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


(* The kinds of the equations that the state of an n-way join of
   environments holds, through its first phases, and of those that
   its last phase adds.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinKindsProofs.v, which needs no review.

   What is stated of adding an extension to the meet's environment
   supposes that its types are kinded in it and that each name of an
   equation is of the kind of its type ([env_extension_fits],
   [env_extension_kinded], FlambdaMeetSpec.v and
   FlambdaMeetEnvSpec.v; these are premises of statements, and not
   tests that the function makes). The equations of the
   two extensions that the join adds are made, by its last phase,
   from the equations that its state holds. This file says what the
   first phases keep of the kinds of those:

   - the aliases of the type of an equation, each with the kind at
     which it stands, are of that kind "in reach"
     ([equations_kinded_in_reach]): in the target so far, or in one
     of the joined environments. (The join adds an equation before it
     brings in the variables of its type, so "in the target" is not
     kept; where the variables are all in the target it follows,
     [scoped_equations_kinded_statement].)
   - the variable of an equation is of the kind of its type
     ([equations_of_kinds]);
   - the types kept to be joined for a variable are of its kind, and
     there is one at least ([types_to_join_of_kinds]);
   - a variable that is to be an alias of a variable that the join
     has made is of the kind of that one ([aliases_of_kinds]);
   - the definition of a variable that the join has made has a
     [Simple] ([definitions_nonempty]: with none, the join of types
     would be given no type, and its contract says nothing of the
     kind of what it then answers).
   Together: [equations_kinded]. And of the relations that the
   rounds recover: the variable that holds the result of one is a
   naked immediate ([relations_kinded]).

   The statements:
   - the first phases answer a state that has these, and relations
     that have theirs ([first_phases_kinds_statement]), under the
     premises of [first_phases_true_statement] (FlambdaJoinStateSpec.v)
     and one more: the join of types keeps [equations_kinded]
     ([join_of_types_keeps_kinds]);
   - the join of types of FlambdaJoinTypes.v, given the three
     functions of the join of environments, keeps [equations_kinded]
     ([n_way_join_keeps_kinds_statement]);
   - where every variable of an equation is in the target, "in
     reach" is "in the target" ([scoped_equations_kinded_statement]).

   What is asked of the join of types has the kinds of the state
   that it answers as a premise ([state_kinded], of the state
   answered and not of the state given): the functions of the join of
   environments keep [state_kinded] only for arguments that are
   kinded, which a statement about every call cannot suppose; but the
   kinds of an earlier state agree with the joined environments if
   those of a later state do, and the rounds have [state_kinded] of
   the state answered from the contract of the join of types.

   And the last phase: each equation of the two extensions that it
   adds is on a variable that the target has at the kind of the type
   of the equation, and the aliases of that type are of their kinds
   in the target ([last_phase_adds_kinded_statement]). And each
   definition that it answers is of a variable that the state has at
   the kind of the definition, and is a definition that the state has
   ([last_phase_adds_definitions_kinded_statement]).

   Not here: the kinds of the names of the equations inside types. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinFitsSpec FlambdaJoinAddsSpec FlambdaJoinTypes.
Import ListNotations.

(* ======================================================================== *)
(* What the state holds, with its kinds                                     *)
(* ======================================================================== *)

(* A [Simple] is of a kind in reach: in the target so far, or in one
   of the joined environments. *)
Definition simple_of_kind_in_reach (s : Join_env.t) (simple : simple)
    (kind : flambda_kind) : Prop :=
  simple_of_kind_in_target Join_env.joined_env kind_in_target_env s simple kind
  ∨ ∃ (index : Index.t) (use : Typing_env.t),
      Join_env.joined_env s index = Some use ∧ simple_of_kind use simple kind.

(* The aliases of the types of the equations, with their kinds, are of
   those kinds in reach. *)
Definition equations_kinded_in_reach (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    ∀ (simple : simple) (kind : flambda_kind),
      (simple, kind) ∈ aliases_with_kinds ty ->
      simple_of_kind_in_reach s simple kind.

(* The variable of an equation is of the kind of its type. *)
Definition equations_of_kinds (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    kind_in_target_env s x = Some (kind ty).

(* Types to be joined for a variable: there is one, and each is of the
   kind of the variable. *)
Definition types_of_kind (s : Join_env.t) (x : variable)
    (types : Keyed.finite_map Index.t flambda_type) : Prop :=
  (∃ (index : Index.t) (t : flambda_type),
     Keyed.lookup Index.equal index types = Some t)
  ∧ ∀ (index : Index.t) (t : flambda_type),
      Keyed.lookup Index.equal index types = Some t ->
      kind_in_target_env s x = Some (kind t).

Definition types_to_join_of_kinds (s : Join_env.t) : Prop :=
  ∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type),
    lookup x (Join_env.types_in_joined_envs s) = Some types ->
    types_of_kind s x types.

(* A variable that is to be an alias of a variable that the join has
   made is of the kind of that one. *)
Definition aliases_of_kinds (s : Join_env.t) : Prop :=
  ∀ e x : variable,
    x
    ∈ Aliases_of_existentials.aliases_of_existential_var
        (Join_env.aliases_of_existentials s) e ->
    ∃ kind : flambda_kind,
      kind_in_target_env s x = Some kind ∧ kind_in_target_env s e = Some kind.

(* The definition of a variable that the join has made has a
   [Simple]. *)
Definition definitions_nonempty (s : Join_env.t) : Prop :=
  ∀ (x : variable) (simples : Simples_in_joined_envs.t)
    (kind : flambda_kind),
    lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind) ->
    ∃ (index : Index.t) (simple : simple),
      Simples_in_joined_envs.find index simples = Some simple.

(* The five together. *)
Definition equations_kinded (s : Join_env.t) : Prop :=
  equations_kinded_in_reach s
  ∧ equations_of_kinds s
  ∧ types_to_join_of_kinds s
  ∧ aliases_of_kinds s
  ∧ definitions_nonempty s.

(* The variable that holds the result of a relation kept is a naked
   immediate. *)
Definition relations_kinded (s : Join_env.t)
    (relations : Join_env.inverse_relations) : Prop :=
  ∀ (r : variable) (names : list (Relation.t * name)),
    lookup r relations = Some names ->
    kind_in_target_env s r = Some (Naked_number Kind_naked_immediate).

(* The equations of a state are kinded in the target so far. *)
Definition equations_kinded_in_target (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    type_kinded_in_target Join_env.joined_env kind_in_target_env s ty.

(* Where the types of the equations mention only variables of the
   target, "in reach" is "in the target": a [Simple] that a joined
   environment has at a kind, and whose variable the target has, is
   of that kind in the target, at a state that is valid and kinded
   with joined environments that have their invariant and are kinded.
   (The invariant is for the symbols: a symbol is of its kind in the
   target if an environment joined knows it, FlambdaJoinSpec.v, and
   one that has a type in an environment is known to it by the
   invariant.) *)
Definition scoped_equations_kinded_statement : Prop :=
  ∀ s : Join_env.t,
    state_valid s ->
    state_kinded s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    equations_in_target s ->
    equations_kinded_in_reach s ->
    equations_kinded_in_target s.

(* ======================================================================== *)
(* The first phases                                                         *)
(* ======================================================================== *)

(* What is asked of the join of types: it keeps [equations_kinded],
   at a valid state whose joined environments have their invariant
   and are kinded, if the state that it answers is kinded. *)
Definition join_of_types_keeps_kinds (jt : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') ->
    state_valid s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    state_kinded s' ->
    equations_kinded s ->
    equations_kinded s'.

(* The join of types of FlambdaJoinTypes.v, given the three functions
   of the join of environments with a source of names that are new,
   keeps it: whatever the other four functions that it is given. *)
Definition n_way_join_keeps_kinds_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    fresh_avoids p ->
    join_of_types_keeps_kinds
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* The first phases answer a state whose equations are kinded, and
   relations that are: under the premises of
   [first_phases_true_statement] (FlambdaJoinStateSpec.v), with a join
   of types that keeps [equations_kinded]. *)
Definition first_phases_kinds_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    join_of_types_keeps_kinds jt ->
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    expand_const_sound p ->
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
    equations_kinded s3 ∧ relations_kinded s3 relations3.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

(* What the last phase adds ([last_phase_adds], FlambdaJoinAddsSpec.v)
   is kinded in the target: each equation of the two extensions is on
   a variable that the target has at the kind of the type of the
   equation, and each alias of that type, with the kind at which it
   stands, is of that kind in the target.

   The premises are of the state that the first phases answer:
   [first_phases_kinds_statement] has [equations_of_kinds],
   [aliases_of_kinds] and [relations_kinded] of it, and
   [scoped_equations_kinded_statement] has
   [equations_kinded_in_target].

   "In the target" is by the kinds that the bindings of the state
   record ([kind_in_target_env]), which has every variable that the
   join has made. Those that the last phase does not keep are not in
   the types of the equations
   ([last_phase_adds_in_scope_statement]); what the environment of
   the meet then knows of the kinds is
   [last_phase_kinds_known_statement] of FlambdaJoinMeetSpec.v. *)
Definition last_phase_adds_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_valid s ->
    equations_kinded_in_target s ->
    equations_of_kinds s ->
    aliases_of_kinds s ->
    relations_kinded s relations ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations ++ relation_equations ->
      type_kinded_in_target Join_env.joined_env kind_in_target_env s ty
      ∧ ∃ x : variable,
          n = Name_var x ∧ kind_in_target_env s x = Some (kind ty).

(* The definitions that the last phase answers: each is of a variable
   that the state has at the kind of the definition, and is a
   definition that the state has for a variable that the join has
   made: that of a variable that the phase defines, or one that has
   been moved to the variable of the target that stands for a
   variable that it removes, which is of the kind of that one
   ([aliases_of_kinds]). (That the variable is one that the target is
   to have is [last_phase_adds_definitions_statement],
   FlambdaJoinAddsSpec.v.) *)
Definition last_phase_adds_definitions_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (source : Typing_env.t)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    aliases_of_kinds s ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
      lookup x definitions = Some definition ->
      kind_in_target_env s x = Some (snd definition)
      ∧ ∃ e : variable,
          lookup e (Join_env.definitions_of_existentials s) = Some definition.
