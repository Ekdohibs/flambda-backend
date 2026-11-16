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


(* The names of the equations inside the types that the state of an
   n-way join of environments holds are of their kinds, through its
   first phases, and so are those inside the types of the equations
   that its last phase adds.

   This file is to be reviewed: the definitions and the statements.
   The proofs are in FlambdaJoinEquationNamesProofs.v, which needs no
   review.

   A type has names with equations inside it, in the extensions of
   the cases of a row-like structure and of a variant, each at the
   kind of the type of its equation ([equation_names_with_kinds],
   FlambdaTypeGrammar.v). An extension that the environment of the
   meet is given is supposed to have types in which those names are
   of those kinds in that environment ([type_equations_kinded],
   FlambdaTypingEnvKindedSpec.v). Of the two extensions that the join
   adds, that is the second part of what FlambdaJoinMeetSpec.v leaves
   as a premise ([types_fit_otherwise]). This file follows
   FlambdaJoinKindsSpec.v, which says the same of the aliases of the
   types:

   - the names of the equations inside the type of an equation of
     the target made so far are of their kinds "in reach"
     ([simple_of_kind_in_reach], FlambdaJoinKindsSpec.v): in the
     target so far, or in one of the joined environments (the join
     adds an equation before it brings in the variables of its type);
   - each type kept to be joined is at the index of a joined
     environment, in which the names of the equations inside it are
     of their kinds.
   Together: [equation_names_kinded].

   The statements:
   - the join of types of FlambdaJoinTypes.v, given the three
     functions of the join of environments with a source of names
     that are new, keeps [equation_names_kinded]
     ([n_way_join_keeps_equation_names_statement]);
   - where the name of an equation inside the type that
     [Join_env.recover_inverse_relations] answers comes from
     ([recover_equation_names_statement]): each such name, with its
     kind, is one of the type given, or is at the kind of naked
     immediates and is reached, in no step or more, through the
     aliases between naked immediates of an extension whose aliases
     are aliases of the type given, from a variable that the type
     given has an alias to at that kind. (The function adds one
     equation at most, on the variable of the tag or on a name that
     the extension for blocks makes an alias of it. The statement
     says less, and what the kind of a name needs: it does not say
     that one name only is new, nor which variable and which
     extension they are, nor that the names of the type given are
     kept.)
   - the first phases answer a state that has [equation_names_kinded]
     ([first_phases_equation_names_statement]), under the premises of
     [first_phases_true_statement] (FlambdaJoinStateSpec.v) and four
     more: the join of types meets its contract for those names
     ([n_way_join_type_equations_kinded], FlambdaJoinSpec.v) and keeps
     [equation_names_kinded]; the type that the parameters give for a
     constant has no equation inside it; and the joined environments
     have the names of the equations inside their types at their
     kinds ([types_equations_kinded]);
   - where every variable of an equation is in the target, "in
     reach" is "in the target"
     ([scoped_equation_names_statement]);
   - each type of the two extensions that the last phase adds
     ([last_phase_adds], FlambdaJoinAddsSpec.v) has the names of the
     equations inside it at their kinds in the target, by the kinds
     that the bindings of the state record
     ([last_phase_adds_equation_names_statement]). As for the aliases
     ([last_phase_adds_kinded_statement], FlambdaJoinKindsSpec.v),
     what the environment of the meet then knows of those kinds is
     for FlambdaJoinMeetSpec.v;
   - of what the join is given in FlambdaJoinMeet.v: the join of
     types ([n_way_join_of_types], FlambdaJoinTypesWithEnvSpec.v)
     keeps [equation_names_kinded], and the type that the parameters
     give for a constant has no equation inside it
     ([n_way_join_of_types_keeps_equation_names_statement],
     [join_parameters_no_equations_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaBindingTime FlambdaEnvAliases FlambdaTypingEnv
  FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinFitsSpec FlambdaJoinAddsSpec FlambdaJoinKindsSpec
  FlambdaJoinTypes FlambdaJoinTypesSpec FlambdaJoinTypesWithEnvSpec
  FlambdaJoinMeet.
Import ListNotations.

(* ======================================================================== *)
(* What the state holds                                                     *)
(* ======================================================================== *)

(* The names of the equations inside a type are of their kinds in
   reach: in the target so far, or in one of the joined
   environments. *)
Definition equation_names_in_reach (s : Join_env.t) (ty : flambda_type)
    : Prop :=
  ∀ (n : name) (kind : flambda_kind),
    (n, kind) ∈ equation_names_with_kinds ty ->
    simple_of_kind_in_reach s (simple_of_name n) kind.

(* Types to be joined: each is at the index of a joined environment
   in which the names of the equations inside it are of their
   kinds. *)
Definition types_equations_kinded_in_envs (s : Join_env.t)
    (types : Keyed.finite_map Index.t flambda_type) : Prop :=
  ∀ (index : Index.t) (t : flambda_type),
    Keyed.lookup Index.equal index types = Some t ->
    ∃ use : Typing_env.t,
      Join_env.joined_env s index = Some use ∧ type_equations_kinded use t.

(* The two together, of the equations of the target made so far and
   of the types kept to be joined. *)
Definition equation_names_kinded (s : Join_env.t) : Prop :=
  (∀ (x : variable) (ty : flambda_type),
     lookup x (Join_env.types_in_target_env s) = Some ty ->
     equation_names_in_reach s ty)
  ∧ ∀ (x : variable) (types : Keyed.finite_map Index.t flambda_type),
      lookup x (Join_env.types_in_joined_envs s) = Some types ->
      types_equations_kinded_in_envs s types.

(* The type that the parameters give for a constant has no equation
   inside it. *)
Definition expand_const_no_equations (p : Join_env.parameters) : Prop :=
  ∀ c : const, equation_names_with_kinds (Join_env.expand_const p c) = [].

(* ======================================================================== *)
(* The join of types                                                        *)
(* ======================================================================== *)

(* What is asked of the join of types: it keeps
   [equation_names_kinded], at a valid state whose joined
   environments have the names of the equations inside their types
   at their kinds. *)
Definition join_of_types_keeps_equation_names
    (jt : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') ->
    state_valid s ->
    joined_envs_are types_equations_kinded s ->
    equation_names_kinded s ->
    equation_names_kinded s'.

(* The join of types of FlambdaJoinTypes.v, given the three functions
   of the join of environments with a source of names that are new,
   keeps it: whatever the other four functions that it is given. *)
Definition n_way_join_keeps_equation_names_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    fresh_avoids p ->
    join_of_types_keeps_equation_names
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* ======================================================================== *)
(* The relations recovered                                                  *)
(* ======================================================================== *)

(* A name is reached from another through the aliases that the
   equations of an extension make between naked immediates: in no
   step; or the extension has an equation that makes the first an
   alias, as a naked immediate, of a [Simple] that is a name, and the
   name is reached from that one. *)
Inductive aliased_in (extension : env_extension) : name -> name -> Prop :=
  | Aliased_here (n : name) : aliased_in extension n n
  | Aliased_step (n n' n'' : name) (simple : simple) :
      Keyed.lookup name_eqb n (Join_env.equations_of extension)
      = Some (Type_naked_immediate (Ok (Equals simple))) ->
      name_of_simple simple = Some n' ->
      aliased_in extension n' n'' ->
      aliased_in extension n n''.

(* Where the name of an equation inside the type that
   [Join_env.recover_inverse_relations] answers comes from. Each such
   name, with its kind, is one of the type given; or it is at the
   kind of naked immediates and is reached ([aliased_in], in no step
   or more) from a variable that the type given has an alias to at
   that kind, through an extension whose aliases are all aliases of
   the type given. That is enough for the kind of the name. It is
   less than what the function does (it adds one equation at most, in
   the extension for blocks of a type of variants, on the variable of
   the tag or on a name that this extension makes an alias of it):
   the variable and the extension are not said to be those, it is
   not said that one name only is new, and it is not said that the
   names of the type given are kept. *)
Definition recover_equation_names_statement : Prop :=
  ∀ (relations relations' : Join_env.inverse_relations) (n : name)
    (ty ty' : flambda_type),
    Join_env.recover_inverse_relations relations n ty = Some (ty', relations')
    ->
    ∀ equation : name * flambda_kind,
      equation ∈ equation_names_with_kinds ty' ->
      equation ∈ equation_names_with_kinds ty
      ∨ ∃ (g : variable) (extension : env_extension) (n' : name),
          (Var g, Naked_number Kind_naked_immediate) ∈ aliases_with_kinds ty
          ∧ (∀ alias : simple * flambda_kind,
               alias ∈ aliases_of_env_extension extension ->
               alias ∈ aliases_with_kinds ty)
          ∧ aliased_in extension (Name_var g) n'
          ∧ equation = (n', Naked_number Kind_naked_immediate).

(* ======================================================================== *)
(* The first phases                                                         *)
(* ======================================================================== *)

(* The first phases answer a state that has [equation_names_kinded]:
   under the premises of [first_phases_true_statement]
   (FlambdaJoinStateSpec.v), with a join of types that meets its
   contract for the names of equations and keeps
   [equation_names_kinded], a type of constants with no equation
   inside it, and joined environments that have the names of the
   equations inside their types at their kinds. *)
Definition first_phases_equation_names_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    n_way_join_type_equations_kinded typing_env_invariant Join_env.joined_env
      kind_in_target_env state_valid state_kinded jt ->
    join_of_types_keeps_equation_names jt ->
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    expand_const_sound p ->
    expand_const_no_equations p ->
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
       types_equations_kinded use) ->
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
    equation_names_kinded s3.

(* The names of the equations inside the types of the equations of a
   state are of their kinds in the target so far. *)
Definition equation_names_kinded_in_target (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    type_equations_kinded_in_target Join_env.joined_env kind_in_target_env s
      ty.

(* Where the types of the equations mention only variables of the
   target, "in reach" is "in the target": as
   [scoped_equations_kinded_statement] (FlambdaJoinKindsSpec.v). *)
Definition scoped_equation_names_statement : Prop :=
  ∀ s : Join_env.t,
    state_valid s ->
    state_kinded s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    equations_in_target s ->
    equation_names_kinded s ->
    equation_names_kinded_in_target s.

(* ======================================================================== *)
(* The last phase                                                           *)
(* ======================================================================== *)

(* Each type of the two extensions that the last phase adds has the
   names of the equations inside it at their kinds in the target, if
   those of the equations of the target that the state holds are.
   "In the target" is by the kinds that the bindings of the state
   record, as in [last_phase_adds_kinded_statement]
   (FlambdaJoinKindsSpec.v). *)
Definition last_phase_adds_equation_names_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    equation_names_kinded_in_target s ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations ++ relation_equations ->
      type_equations_kinded_in_target Join_env.joined_env kind_in_target_env s
      ty.

(* ======================================================================== *)
(* What the join is given in FlambdaJoinMeet.v                              *)
(* ======================================================================== *)

(* The join of types with the functions of the join of environments
   ([n_way_join_of_types], FlambdaJoinTypesWithEnvSpec.v) keeps
   [equation_names_kinded]. *)
Definition n_way_join_of_types_keeps_equation_names_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (fuel : nat),
    fresh_avoids p ->
    join_of_types_keeps_equation_names
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel).

(* The type that the parameters fixed there give for a constant has
   no equation inside it. *)
Definition join_parameters_no_equations_statement : Prop :=
  ∀ join_depth : nat, expand_const_no_equations (join_parameters join_depth).
