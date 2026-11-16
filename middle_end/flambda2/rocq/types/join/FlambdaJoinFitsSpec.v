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


(* That what the n-way join adds to the target fits it: the variables,
   the kinds of the aliases and the kinds of the names of equations
   of the types that the last phase of the join makes.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaJoinFitsProofs.v, which needs no review.

   The meet's environment adds an extension only if it fits
   ([env_extension_fits], FlambdaMeetSpec.v): the environment knows
   the names of its equations and the variables of their types, the
   aliases of those types and the names of the equations inside them
   are of the kinds at which the types have them, and the types are
   well formed. The types of the two extensions that the join adds
   are made by [project_variables_out] (FlambdaTypeProject.v) from
   the types of the state. What is here so far is about that
   function alone, for any variables to remove and any types put for
   them:
   - the variables ([project_variables_out_scope_statement]): if the
     variables of the type that are not removed have a property, and
     so have the variables of the types put for those removed, then
     the variables of the type answered have it. (The property is to
     be "a variable of the target".)
   - the aliases ([project_variables_out_kinded_statement]): the same
     of the aliases with their kinds ([aliases_with_kinds]), where an
     alias of a variable that is removed is supposed to be at the
     kind of the type put for it. (An alias is replaced by that type
     only if the kinds agree; but a variable that holds the result of
     [Is_int], [Get_tag] or [Is_null], and a variable that an inverse
     relation is on, are replaced by what the type put for them is an
     alias of, whatever the kind of that type.)
   - the names of the equations inside the type
     ([project_variables_out_equations_kinded_statement]): the same
     of [equation_names_with_kinds]. An equation on a variable that
     is removed is dropped, so nothing is supposed of it.
   - the kind ([project_variables_out_kind_statement]): the type
     answered is of the kind of the type given.
   - well-formedness ([project_variables_out_well_formed_statement]):
     the type answered is well formed ([type_well_formed],
     FlambdaTypeWellFormedSpec.v) if the type given is and the types
     put for the variables removed are.
   Each but the kind is stated of a type and of an environment
   extension.

   And about the function that the last phase gives the projection
   for the types to put ([Join_env.expand_variable], FlambdaJoinEnv.v:
   the type kept for a variable that is removed, in which the
   variables that are removed are replaced in turn): the variables of
   a type that it answers have a property if those of the types kept
   that are not removed have it ([expand_variable_scope_statement]);
   and the type is well formed if the types kept are
   ([expand_variable_well_formed_statement]).

   And about the variables of the two extensions, for the join of
   environments up to what it adds to the target:
   - the last phase ([define_variables_in_scope_statement]): the
     equations of the two extensions that it adds are on variables
     that the target is to have, and their types mention only such
     variables: those that the source knows, those brought in, and
     those of the variables that the join has made that the phase
     defines ([in_final_target]). This is of a state whose equations
     mention only variables of the target so far
     ([equations_in_target]), and where no variable that is to be an
     alias of one that the join has made is itself one that the join
     has made ([aliases_not_made]).
   - the first phases ([first_phases_scoped_statement]): the state
     that they answer has those two properties ([state_scoped], and
     [scoped_aliases_not_made_statement]), given a join of types that
     has its contract and keeps them
     ([join_of_types_keeps_scoped]).
   - the join of types of FlambdaJoinTypes.v, given the three
     functions of the join of environments, keeps them
     ([n_way_join_keeps_scoped_statement]).
   - together ([cut_and_n_way_join0_in_scope_statement]): the
     equations of the two extensions that [cut_and_n_way_join0] adds
     are on such variables and mention only such variables.

   [in_final_target] is a set of variables read from the state of the
   join: it is what the target is to know, not what the environment
   answered knows. The functions of the meet's environment are any
   functions here, and nothing is supposed of them: that the
   environment that they answer knows these variables is for the
   statements about those functions. A symbol of an equation is said
   nothing of (the target is to know it too). And the witnesses of
   these statements (the variables kept, the two lists of equations)
   are not said to be those of the statements of
   FlambdaJoinDefineSpec.v about what the equations mean: the proofs
   take the same ones, and a statement that has both conclusions is
   still to be made.

   Not here yet: the kinds of what [Join_env.expand_variable] answers;
   and the kinds and the well-formedness of the equations that the
   state of a join holds, through the phases of the join, which is
   what would give them of the two extensions. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaTypeProject FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec
  FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinStateSpec
  FlambdaJoinDefineSpec FlambdaJoinTypes.
Import ListNotations.

(* The type answered is of the kind of the type given. *)
Definition project_variables_out_kind_statement : Prop :=
  ∀ (to_project : list variable) (expand : variable -> option flambda_type)
    (t t' : flambda_type),
    project_variables_out to_project expand t = Some t' -> kind t' = kind t.

(* The variables. *)
Definition project_variables_out_scope_statement : Prop :=
  ∀ (to_project : list variable) (expand : variable -> option flambda_type)
    (in_target : variable -> Prop),
    (∀ (e : variable) (ty : flambda_type) (x : variable),
       among e to_project = true ->
       expand e = Some ty ->
       x ∈ free_variables ty ->
       in_target x) ->
    (∀ t t' : flambda_type,
       project_variables_out to_project expand t = Some t' ->
       (∀ x : variable,
          x ∈ free_variables t -> among x to_project = false -> in_target x)
       ->
       ∀ x : variable, x ∈ free_variables t' -> in_target x)
    ∧ ∀ extension extension' : env_extension,
        project_env_extension to_project expand extension = Some extension'
        ->
        (∀ x : variable,
           x ∈ free_variables_of_env_extension extension ->
           among x to_project = false ->
           in_target x) ->
        ∀ x : variable,
          x ∈ free_variables_of_env_extension extension' -> in_target x.

(* What is supposed of an alias of the type given, at a kind: if it is
   of a variable that is removed, the type put for the variable is of
   that kind; otherwise it has the property. *)
Definition alias_before (to_project : list variable)
    (expand : variable -> option flambda_type)
    (of_kind : simple -> flambda_kind -> Prop)
    (alias : simple * flambda_kind) : Prop :=
  match fst alias with
  | Var e =>
    if among e to_project
    then ∀ ty : flambda_type, expand e = Some ty -> kind ty = snd alias
    else of_kind (fst alias) (snd alias)
  | Symbol _ | Const _ => of_kind (fst alias) (snd alias)
  end.

(* The aliases, with their kinds. *)
Definition project_variables_out_kinded_statement : Prop :=
  ∀ (to_project : list variable) (expand : variable -> option flambda_type)
    (of_kind : simple -> flambda_kind -> Prop),
    (∀ (e : variable) (ty : flambda_type) (s : simple) (k : flambda_kind),
       among e to_project = true ->
       expand e = Some ty ->
       (s, k) ∈ aliases_with_kinds ty ->
       of_kind s k) ->
    (∀ t t' : flambda_type,
       project_variables_out to_project expand t = Some t' ->
       (∀ alias : simple * flambda_kind,
          alias ∈ aliases_with_kinds t ->
          alias_before to_project expand of_kind alias) ->
       ∀ (s : simple) (k : flambda_kind),
         (s, k) ∈ aliases_with_kinds t' -> of_kind s k)
    ∧ ∀ extension extension' : env_extension,
        project_env_extension to_project expand extension = Some extension'
        ->
        (∀ alias : simple * flambda_kind,
           alias ∈ aliases_of_env_extension extension ->
           alias_before to_project expand of_kind alias) ->
        ∀ (s : simple) (k : flambda_kind),
          (s, k) ∈ aliases_of_env_extension extension' -> of_kind s k.

(* A name that is not a variable that is removed. *)
Definition name_kept (to_project : list variable) (n : name) : Prop :=
  match n with
  | Name_var e => among e to_project = false
  | Name_symbol _ => True
  end.

(* The names of the equations inside the type, with their kinds. *)
Definition project_variables_out_equations_kinded_statement : Prop :=
  ∀ (to_project : list variable) (expand : variable -> option flambda_type)
    (of_kind : name -> flambda_kind -> Prop),
    (∀ (e : variable) (ty : flambda_type) (n : name) (k : flambda_kind),
       among e to_project = true ->
       expand e = Some ty ->
       (n, k) ∈ equation_names_with_kinds ty ->
       of_kind n k) ->
    (∀ t t' : flambda_type,
       project_variables_out to_project expand t = Some t' ->
       (∀ (n : name) (k : flambda_kind),
          (n, k) ∈ equation_names_with_kinds t ->
          name_kept to_project n ->
          of_kind n k) ->
       ∀ (n : name) (k : flambda_kind),
         (n, k) ∈ equation_names_with_kinds t' -> of_kind n k)
    ∧ ∀ extension extension' : env_extension,
        project_env_extension to_project expand extension = Some extension'
        ->
        (∀ (n : name) (k : flambda_kind),
           (n, k) ∈ equation_names_of_env_extension extension ->
           name_kept to_project n ->
           of_kind n k) ->
        ∀ (n : name) (k : flambda_kind),
          (n, k) ∈ equation_names_of_env_extension extension' -> of_kind n k.

(* Well-formedness. *)
Definition project_variables_out_well_formed_statement : Prop :=
  ∀ (to_project : list variable) (expand : variable -> option flambda_type),
    (∀ (e : variable) (ty : flambda_type),
       among e to_project = true ->
       expand e = Some ty ->
       type_well_formed ty = true) ->
    (∀ t t' : flambda_type,
       project_variables_out to_project expand t = Some t' ->
       type_well_formed t = true ->
       type_well_formed t' = true)
    ∧ ∀ extension extension' : env_extension,
        project_env_extension to_project expand extension = Some extension'
        ->
        env_extension_well_formed extension = true ->
        env_extension_well_formed extension' = true.

(* ======================================================================== *)
(* The types put for the variables that the last phase removes              *)
(* ======================================================================== *)

(* The variables of a type that [Join_env.expand_variable] answers:
   they have a property if every variable of a type kept that is not
   itself removed has it. (With "is not removed" as part of the
   property: no variable that is removed is left.) *)
Definition expand_variable_scope_statement : Prop :=
  ∀ (to_expand : finite_map flambda_type) (in_target : variable -> Prop),
    (∀ (e : variable) (ty : flambda_type) (x : variable),
       lookup e to_expand = Some ty ->
       x ∈ free_variables ty ->
       among x (domain to_expand) = false ->
       in_target x) ->
    ∀ (fuel : nat) (e : variable) (ty : flambda_type) (x : variable),
      Join_env.expand_variable fuel to_expand e = Some ty ->
      x ∈ free_variables ty ->
      in_target x.

(* A type that it answers is well formed if the types kept are. *)
Definition expand_variable_well_formed_statement : Prop :=
  ∀ to_expand : finite_map flambda_type,
    (∀ (e : variable) (ty : flambda_type),
       lookup e to_expand = Some ty -> type_well_formed ty = true) ->
    ∀ (fuel : nat) (e : variable) (ty : flambda_type),
      Join_env.expand_variable fuel to_expand e = Some ty ->
      type_well_formed ty = true.

(* ======================================================================== *)
(* The variables of the two extensions that the last phase adds             *)
(* ======================================================================== *)

(* The types of the equations that the state holds mention only
   variables of the target so far. (The invariant of the state asks
   less: a variable of such a type may be one that only a joined
   environment knows, as it is between the adding of an equation and
   the bringing in of the variables of its type.) *)
Definition equations_in_target (s : Join_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type),
    lookup x (Join_env.types_in_target_env s) = Some ty ->
    type_in_target kind_in_target_env s ty.

(* No variable that is to be an alias of a variable that the join has
   made is itself one that the join has made. *)
Definition aliases_not_made (s : Join_env.t) : Prop :=
  ∀ e x : variable,
    x
    ∈ Aliases_of_existentials.aliases_of_existential_var
        (Join_env.aliases_of_existentials s) e ->
    ~ is_existential s x.

(* A variable that the target that the last phase answers is to have,
   given the variables that the join has made that the phase defines:
   one that the source knows, one brought in, or one of those. (This
   is read from the state of the join. That the environment answered
   knows such a variable depends on the functions of the meet's
   environment, of which nothing is supposed here.) *)
Definition in_final_target (s : Join_env.t)
    (kept : list (variable * flambda_kind)) (x : variable) : Prop :=
  knows (Join_env.source_env s) x ∨ is_imported s x ∨ x ∈ map fst kept.

(* The equations of an extension are on such variables, and their
   types mention only such variables. *)
Definition equations_in_final_target (s : Join_env.t)
    (kept : list (variable * flambda_kind))
    (equations : list (name * flambda_type)) : Prop :=
  ∀ (n : name) (ty : flambda_type),
    (n, ty) ∈ equations ->
    (∀ x : variable,
       x ∈ free_variables_of_name n -> in_final_target s kept x)
    ∧ ∀ x : variable, x ∈ free_variables ty -> in_final_target s kept x.

(* The last phase, at a state that has the invariant and the two
   properties above, with relations that are true of it: the target
   is as [define_variables_true_statement] (FlambdaJoinDefineSpec.v)
   has it, the environment given with the variables brought in and
   some of the variables made defined in it and two extensions added;
   and the equations of the two extensions are on variables that the
   target is to have, and mention only such variables. *)
Definition define_variables_in_scope_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (mh : meet_expanded_head) (source : Typing_env.t)
    (s : Join_env.t) (menv target : meet_env)
    (relations : Join_env.inverse_relations)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    relations_hold s relations ->
    relations_nonempty relations ->
    equations_in_target s ->
    aliases_not_made s ->
    Join_env.define_or_eliminate_variables_and_add_equations ops fuel mh s
      menv relations
    = Some (target, definitions) ->
    ∃ (kept : list (variable * flambda_kind))
      (equations relation_equations : list (name * flambda_type))
      (menv' : meet_env),
      (∀ binding : variable * flambda_kind,
         binding ∈ kept ->
         binding
         ∈ Bindings_in_target_env.existential_variables_in_order
             (Join_env.bindings s))
      ∧ Join_env.add_env_extension ops
          (variables_defined ops menv
             (Bindings_in_target_env.imported_variables_in_order
                (Join_env.bindings s)
              ++ kept))
          (Env_extension equations) mh
        = Some menv'
      ∧ Join_env.add_env_extension ops menv' (Env_extension relation_equations)
          mh
        = Some target
      ∧ equations_in_final_target s kept equations
      ∧ equations_in_final_target s kept relation_equations.

(* ======================================================================== *)
(* The state that the first phases answer                                   *)
(* ======================================================================== *)

(* A variable that is to be an alias of a variable that the join has
   made is one that the source knows, or one brought in. *)
Definition aliases_from_envs (s : Join_env.t) : Prop :=
  ∀ e x : variable,
    x
    ∈ Aliases_of_existentials.aliases_of_existential_var
        (Join_env.aliases_of_existentials s) e ->
    knows (Join_env.source_env s) x ∨ is_imported s x.

(* The variables listed as brought in are recorded with a kind (the
   first part of [imports_recorded], FlambdaJoinEnvSpec.v, which a
   valid state has). *)
Definition imports_listed (s : Join_env.t) : Prop :=
  ∀ x : variable,
    among x (Join_env.imported_vars s) = true -> is_imported s x.

(* What the first phases keep of the variables of the state: the types
   of its equations mention only variables of the target so far, and
   the aliases of the variables made are of the source or brought
   in. *)
Definition state_scoped (s : Join_env.t) : Prop :=
  equations_in_target s ∧ aliases_from_envs s.

(* At a valid state, the second gives the premise of
   [define_variables_in_scope_statement]: no environment knows a
   variable that the join has made. *)
Definition scoped_aliases_not_made_statement : Prop :=
  ∀ s : Join_env.t,
    state_valid s -> aliases_from_envs s -> aliases_not_made s.

(* What is asked of the join of types: it keeps the two properties,
   at a state whose imports are listed. (Its contract,
   [n_way_join_type_sound] of FlambdaJoinSpec.v, says that the type
   answered mentions only variables of the target, and nothing of the
   equations that the state holds.) *)
Definition join_of_types_keeps_scoped (jt : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') ->
    imports_listed s ->
    state_scoped s ->
    state_scoped s'.

(* The join of types of FlambdaJoinTypes.v, given the three functions
   of the join of environments, keeps them: at every state, for every
   list of types, and whatever the other four functions that it is
   given. (By [n_way_join_keeps_statement], FlambdaJoinTypesKeepsSpec.v:
   the three functions keep them, between any two states.) *)
Definition n_way_join_keeps_scoped_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    join_of_types_keeps_scoped
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* The first phases answer a state that has the two properties: under
   the premises of [first_phases_true_statement]
   (FlambdaJoinStateSpec.v), with a join of types that keeps them. *)
Definition first_phases_scoped_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    join_of_types_keeps_scoped jt ->
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
    state_scoped s3.

(* ======================================================================== *)
(* The join, up to what it adds to the target                               *)
(* ======================================================================== *)

(* [cut_and_n_way_join0], where there is an environment to join, under
   the premises of [cut_and_n_way_join0_true_statement]
   (FlambdaJoinDefineSpec.v) and with a join of types that keeps the
   two properties: there is a state that has the invariant, whose
   names to avoid are the names answered, such that the target
   answered is the environment given with those names for names to
   avoid, the variables brought in and some of the variables made
   defined in it, and two extensions added; and the equations of the
   two extensions are on variables
   that the target is to have, and mention only such variables. *)
Definition cut_and_n_way_join0_in_scope_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat) (choices : Join_env.choices)
    (jt : Join_env.n_way_join_type) (mh : meet_expanded_head)
    (avoid : list variable) (menv target : meet_env) (source : Typing_env.t)
    (envs : Joined_envs.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
    join_of_types_keeps_scoped jt ->
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
    Keyed.is_empty envs = false ->
    Join_env.cut_and_n_way_join0 ops p fuel choices jt mh avoid menv source
      envs
    = Some (target, definitions, names) ->
    ∃ (s : Join_env.t) (kept : list (variable * flambda_kind))
      (equations relation_equations : list (name * flambda_type))
      (menv' : meet_env),
      state_true source s
      ∧ names = Bindings_in_target_env.names_to_avoid (Join_env.bindings s)
      ∧ (∀ binding : variable * flambda_kind,
           binding ∈ kept ->
           binding
           ∈ Bindings_in_target_env.existential_variables_in_order
               (Join_env.bindings s))
      ∧ Join_env.add_env_extension ops
          (variables_defined ops (Join_env.with_names_to_avoid ops menv names)
             (Bindings_in_target_env.imported_variables_in_order
                (Join_env.bindings s)
              ++ kept))
          (Env_extension equations) mh
        = Some menv'
      ∧ Join_env.add_env_extension ops menv' (Env_extension relation_equations)
          mh
        = Some target
      ∧ equations_in_final_target s kept equations
      ∧ equations_in_final_target s kept relation_equations.
