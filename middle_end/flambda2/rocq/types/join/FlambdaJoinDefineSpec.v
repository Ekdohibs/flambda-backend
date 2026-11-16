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


(* The last phase of the n-way join: the variables of the target are
   defined, and its equations are added.

   To be reviewed: every definition and statement of this file. The
   proofs are in FlambdaJoinDefineProofs.v.

   The phase is [define_or_eliminate_variables_and_add_equations]
   (FlambdaJoinEnv.v, after types/env/join_env.ml). It is given the
   state and the relations that the first phases leave
   ([first_phases_true_statement], FlambdaJoinStateSpec.v). It defines
   in the target the variables that the join has brought in; then, for
   each variable that the join has made: if a variable of the target is
   to be its alias, that one takes its place (its definition, its
   equation and its relations move to it, and the other aliases get an
   equation to it); if it is mentioned once and no relation is about it,
   it is removed, and its type is put where it was mentioned; otherwise
   it is defined. The equations, with the variables that have gone
   projected out of their types, are then added to the target through
   the environment of the meet, and after them the relations, as
   equations on the variables that hold their results.

   What is here:

   - The projection keeps what a type says
     ([project_variables_out_sound_statement], and the same of an
     environment extension): if, in a valuation, each variable that is
     projected out has the type that is put for it, a value of a type
     has the type with those variables projected out. The valuation
     gives values to the variables that are projected out: it is a
     valuation of the state of the join, and not only of the target.
   - The type that is put for a variable that goes
     ([expand_variable_sound_statement]): [Join_env.expand_variable]
     follows the types of the variables that go, and what it answers
     is true of the variable if each of those types is.
   - What the phase does and what it adds
     ([define_variables_true_statement]). The target that it answers
     is the environment given, with the variables brought in and some
     of the variables made defined in it, to which two environment
     extensions have been added: the equations, and those of the
     relations. At the run of an input, in the valuation of the target
     that the state gives ([target_valuation], FlambdaJoinStateSpec.v),
     every equation of the two is true of the value of its variable,
     where it has one, and the definitions answered hold. This is of
     the runs that have two conditions, below.
   - Of the relations given, the phase needs that no variable is kept
     with an empty list of them ([relations_nonempty]): the rounds
     keep that ([delayed_equations_relations_nonempty_statement],
     [loop_relations_nonempty_statement]), and start from no
     relation.
   - An extension asks more than that its equations are true where
     their names have values: each name is to have a value
     (FlambdaTypeMeaningSpec.v). [equations_true_with_values_statement]
     is the step from one to the other.

   The two conditions on a run, which nothing in the compiler
   establishes (there is a CR at each):

   - [aliases_have_values]: a variable of the target that is to be an
     alias of a variable that the join has made has a value where that
     one has. The state knows the converse ([alias_holds]), and this
     follows from two conditions of the contract read on the state
     ([aliases_have_values_of_values_statement]).
   - [relation_scrutinees_have_values]: where the variable that holds
     the result of a relation kept has a value, the variable that the
     relation is on has one.

   They are stated of the state of the join and of its valuation, and
   not of the target answered as the conditions of the contract are
   ([values_are_read], FlambdaJoinSpec.v). How the two sets are related
   is not stated here; see the CRs.

   - The join up to what it adds to the target
     ([cut_and_n_way_join0_true_statement]): the first phases
     ([first_phases_true_statement], FlambdaJoinStateSpec.v), which
     also keep the state kinded ([first_phases_kinded_statement]),
     and then this phase.

   - The same of the join that Simplify calls, from the premises of
     the contract ([join_with_true_statement], whose conclusion is
     [join_adds_true]): the joined environments are the uses that are
     not bottom, with the levels cut from them, and two of the
     conditions on the run are the contract's own. And of the
     compiler's join with the join of types of the model
     ([compiler_join_true_statement]).
   - The valuation of the target is in range if that of the input is
     and the constants of the definitions are
     ([target_valuation_in_range_statement]).

   Not here yet: that the two extensions fit the environment that they
   are added to (the variables of their types are variables of the
   target, at their kinds), which the functions of the meet's
   environment ask before they say anything; and the contract of the
   whole join. *)

From Stdlib Require Import ZArith List Bool PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaRangeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeProject FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinTypes
  FlambdaJoinTypesSpec FlambdaJoinTypesWithEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The projection of variables out of a type                                *)
(* ======================================================================== *)

(* In a valuation, each variable that is projected out and has a value
   has the type that is put for it. *)
Definition expansions_true (statics : static_env) (vars : env)
    (to_project : list variable) (expand : variable -> option flambda_type)
    : Prop :=
  ∀ (e : variable) (ty : flambda_type) (w : value),
    among e to_project = true ->
    expand e = Some ty ->
    lookup e vars = Some w ->
    type_satisfied statics vars w ty.

(* A value of a type has the type with the variables projected out. *)
Definition project_variables_out_sound_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (to_project : list variable)
    (expand : variable -> option flambda_type) (t t' : flambda_type)
    (v : value),
    expansions_true statics vars to_project expand ->
    project_variables_out to_project expand t = Some t' ->
    type_satisfied statics vars v t ->
    type_satisfied statics vars v t'.

(* The same of an environment extension. (The equations on the
   variables that are projected out are dropped.) *)
Definition project_env_extension_sound_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (to_project : list variable)
    (expand : variable -> option flambda_type)
    (extension extension' : env_extension),
    expansions_true statics vars to_project expand ->
    project_env_extension to_project expand extension = Some extension' ->
    env_extension_satisfied statics vars extension ->
    env_extension_satisfied statics vars extension'.

(* The type that [Join_env.expand_variable] answers for a variable is
   true of it, if the type kept for each variable is true of that
   variable. *)
Definition expand_variable_sound_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (to_expand : finite_map flambda_type) (fuel : nat) (e : variable)
    (ty : flambda_type) (w : value),
    (∀ (e' : variable) (ty' : flambda_type) (w' : value),
       lookup e' to_expand = Some ty' ->
       lookup e' vars = Some w' ->
       type_satisfied statics vars w' ty') ->
    Join_env.expand_variable fuel to_expand e = Some ty ->
    lookup e vars = Some w ->
    type_satisfied statics vars w ty.

(* ======================================================================== *)
(* Two conditions on a run                                                  *)
(* ======================================================================== *)

(* At the run of an input, a variable of the target that is to be an
   alias of a variable that the join has made has a value where that
   one has. (It is then the same value: [alias_holds].) *)
(* CR mshinwell: nothing in the compiler establishes this. The
   variable that takes the place of a variable that the join has made
   gets its definition, and the others get an equation to it: so, on
   a run in which the vector of [Simple]s of the definition has a
   value at the input and the variable has none, the target asks of
   the variable a value that it has not. The variable is one of the
   source that has an equation in the level of every input, or one
   brought in from the inputs that know it, and the vector has a
   [Simple] only at inputs that know the variable. So the condition
   follows from two others, which say that such a variable has a
   value at an input that knows it ([imported_have_values] and
   [joined_have_values] below,
   [aliases_have_values_of_values_statement]): the invariant of the
   state records where an alias is known and that an alias that the
   source knows has an equation in every level ([alias_holds],
   FlambdaJoinStateSpec.v). Those two are two conditions of the
   contract, [imported_variables_have_values] and
   [joined_variables_have_values] (FlambdaJoinSpec.v), read on the
   state of the join; the contract's are about the target answered
   and the levels cut from the uses, and the step from them is not
   stated yet. Discrepancies T5-5 and T5-10. *)
Definition aliases_have_values (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env) : Prop :=
  ∀ (e x : variable) (v : value),
    x
    ∈ Aliases_of_existentials.aliases_of_existential_var
        (Join_env.aliases_of_existentials s) e ->
    lookup e (target_valuation s index use valuation) = Some v ->
    ∃ w : value, lookup x (target_valuation s index use valuation) = Some w.

(* At the run of an input, a variable that the join has brought in,
   and that the input knows, has a value; and a variable of the source
   that has an equation in the level of every joined environment has
   one. *)
Definition imported_have_values (s : Join_env.t) (use : Typing_env.t)
    (valuation : env) : Prop :=
  ∀ x : variable,
    is_imported s x ->
    knows use x ->
    ∃ v : value, lookup x valuation = Some v.

Definition joined_have_values (s : Join_env.t) (valuation : env) : Prop :=
  ∀ x : variable,
    knows (Join_env.source_env s) x ->
    joined_everywhere s x ->
    ∃ v : value, lookup x valuation = Some v.

Definition aliases_have_values_of_values_statement : Prop :=
  ∀ (source : Typing_env.t) (s : Join_env.t) (index : Index.t)
    (use : Typing_env.t) (valuation : env),
    state_true source s ->
    Join_env.joined_env s index = Some use ->
    imported_have_values s use valuation ->
    joined_have_values s valuation ->
    aliases_have_values s index use valuation.

(* At the run of an input, where the variable that holds the result of
   a relation kept has a value, the variable that the relation is on
   has one. *)
(* CR mshinwell: nothing in the compiler establishes this; it is
   [scrutinees_have_values] of FlambdaJoinSpec.v (see the CR there),
   asked of the relations that the join has kept and not of the
   relations that the target answered has. The two differ: the
   relations kept are added to the target by a meet, with the types
   that the target has already for the variables that hold their
   results, and a relation whose result has an alias or a constant
   for type there is not at the head of a type of the target
   afterwards, though the meet has used it. The condition here is the
   one that the proof needs, since the meet's environment is told
   that the equations added are true. Which of the two the contract
   should ask is open. Discrepancy T5-10. *)
Definition relation_scrutinees_have_values (s : Join_env.t)
    (relations : Join_env.inverse_relations) (index : Index.t)
    (use : Typing_env.t) (valuation : env) : Prop :=
  ∀ (r : variable) (relation : Relation.t) (x : variable) (v : value),
    relation_recorded relations r relation (Name_var x) ->
    lookup r (target_valuation s index use valuation) = Some v ->
    ∃ w : value, lookup x (target_valuation s index use valuation) = Some w.

(* ======================================================================== *)
(* What the phase does, and what it adds                                    *)
(* ======================================================================== *)

(* Every variable that holds the result of a relation kept has one
   kept for it. (The rounds add a relation to those kept for a
   variable, and start from none.) *)
Definition relations_nonempty (relations : Join_env.inverse_relations)
    : Prop :=
  ∀ (r : variable) (names : list (Relation.t * name)),
    lookup r relations = Some names -> names <> [].

(* The rounds keep it, whatever the join of types. *)
Definition delayed_equations_relations_nonempty_statement : Prop :=
  ∀ (fuel : nat) (choices : Join_env.choices)
    (jt : Join_env.n_way_join_type) (s s' : Join_env.t)
    (relations relations' : Join_env.inverse_relations)
    (equations : finite_map (Keyed.finite_map Index.t flambda_type)),
    relations_nonempty relations ->
    Join_env.n_way_join_delayed_equations fuel choices jt s relations
      equations
    = Some (relations', s') ->
    relations_nonempty relations'.

Definition loop_relations_nonempty_statement : Prop :=
  ∀ (depth : nat) (p : Join_env.parameters) (fuel : nat)
    (choices : Join_env.choices) (jt : Join_env.n_way_join_type)
    (s s' : Join_env.t) (relations relations' : Join_env.inverse_relations)
    (existentials : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    relations_nonempty relations ->
    Join_env.n_way_join_loop depth p fuel choices jt s relations existentials
    = Some (relations', s') ->
    relations_nonempty relations'.

(* In a valuation, each equation of a list is true of the value of its
   name, where it has one. *)
Definition equations_true_at (statics : static_env) (vars : env)
    (equations : list (name * flambda_type)) : Prop :=
  ∀ (n : name) (ty : flambda_type) (v : value),
    (n, ty) ∈ equations ->
    value_of_name vars n = Some v ->
    type_satisfied statics vars v ty.

(* An extension is satisfied if its equations are true and each of its
   names has a value. *)
Definition equations_true_with_values_statement : Prop :=
  ∀ (statics : static_env) (vars : env)
    (equations : list (name * flambda_type)),
    equations_true_at statics vars equations ->
    (∀ (n : name) (ty : flambda_type),
       (n, ty) ∈ equations -> ∃ v : value, value_of_name vars n = Some v) ->
    env_extension_satisfied statics vars (Env_extension equations).

(* The environment of the meet in which variables have been defined,
   in order, each at its kind and at the name mode of the variables
   that exist only in types. *)
Definition variables_defined {meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (menv : meet_env) (vars : list (variable * flambda_kind)) : meet_env :=
  List.fold_left
    (fun menv binding =>
       Join_env.add_variable_definition ops menv (fst binding) (snd binding)
         Name_mode.in_types)
    vars menv.

(* [define_or_eliminate_variables_and_add_equations], given a state
   that has the invariant and relations that are true of it. The
   target answered is made from the environment given by defining the
   variables brought in and some of the variables made, and adding two
   extensions. At the run of an input that has the two conditions
   above, and whose blocks have tags in range, each equation of the
   two extensions is true, in the valuation of the target that the
   state gives, of the value of its variable where it has one; and the
   definitions answered hold in that valuation. *)
Definition define_variables_true_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (mh : meet_expanded_head) (source : Typing_env.t)
    (s : Join_env.t) (menv target : meet_env)
    (relations : Join_env.inverse_relations)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    state_kinded s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    relations_nonempty relations ->
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
      ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
          (valuation : env),
          Join_env.joined_env s index = Some use ->
          typing_env_satisfied_strictly statics valuation use ->
          tags_at s use statics valuation ->
          aliases_have_values s index use valuation ->
          relation_scrutinees_have_values s relations index use valuation ->
          equations_true_at statics (target_valuation s index use valuation)
            equations
          ∧ equations_true_at statics
              (target_valuation s index use valuation) relation_equations
          ∧ definitions_hold definitions index valuation
              (target_valuation s index use valuation).

(* ======================================================================== *)
(* The join, up to what it adds to the target                               *)
(* ======================================================================== *)

(* The first phases keep the state kinded, under the premises of
   [first_phases_true_statement] (FlambdaJoinStateSpec.v). The last
   phase needs it: the value that a variable that the join has made
   has at an input is of the kind of the variable. *)
Definition first_phases_kinded_statement : Prop :=
  ∀ (source : Typing_env.t) (jt : Join_env.n_way_join_type)
    (choices : Join_env.choices) (p : Join_env.parameters) (fuel : nat)
    (avoid : list variable) (envs : Joined_envs.t) (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true source) state_kinded jt ->
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
    state_kinded s3.


(* [cut_and_n_way_join0], where there is an environment to join, given
   a join of types that meets its contract with the invariant, and
   joined environments as [first_phases_true_statement] asks. There is
   a state that has the invariant, is kinded, and has grown from the
   state that the join starts from, and relations that are true of
   it, such that: the names answered are the names to avoid of that
   state; the target answered is the environment given, with those
   names for names to avoid (so that a join that the meets of the
   two extensions call makes none of them: FlambdaJoinEnv.v, "New
   names"), with the variables brought in and some of the variables
   made defined in it, and two extensions added; and at the run of an
   input that has the
   two conditions, and whose blocks have tags in range, the equations
   of the two extensions are true where their variables have values,
   and the definitions answered hold, in the valuation of the target
   that the state gives. Nothing is said of what the functions of the
   meet's environment answer. *)
Definition cut_and_n_way_join0_true_statement : Prop :=
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
    ∃ (s : Join_env.t) (relations : Join_env.inverse_relations)
      (kept : list (variable * flambda_kind))
      (equations relation_equations : list (name * flambda_type))
      (menv' : meet_env),
      state_true source s
      ∧ state_kinded s
      ∧ relations_hold s relations
      ∧ state_grows Join_env.joined_env Join_env.definitions_of_existentials
          kind_in_target_env
          (Join_env.create envs
             (Bindings_in_target_env.from_source_env avoid source))
          s
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
      ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
          (valuation : env),
          Join_env.joined_env s index = Some use ->
          typing_env_satisfied_strictly statics valuation use ->
          tags_at s use statics valuation ->
          aliases_have_values s index use valuation ->
          relation_scrutinees_have_values s relations index use valuation ->
          equations_true_at statics (target_valuation s index use valuation)
            equations
          ∧ equations_true_at statics
              (target_valuation s index use valuation) relation_equations
          ∧ definitions_hold definitions index valuation
              (target_valuation s index use valuation).

(* ======================================================================== *)
(* The join that Simplify calls, up to what it adds to the target           *)
(* ======================================================================== *)

(* The join of types keeps the joined environments, with the levels
   kept with them. This is more than its contract gives ([state_grows]
   is about the environments and not about their levels). It is so of
   a join of types that changes the state only through the three
   functions of the environment side, which keep them at every
   state. *)
Definition join_of_types_keeps_envs (jt : Join_env.n_way_join_type) : Prop :=
  ∀ (s s' : Join_env.t) (ts : list (Index.t * flambda_type))
    (joined : Or_unknown.t flambda_type),
    jt s ts = Some (joined, s') ->
    Join_env.joined_envs s' = Join_env.joined_envs s.

Definition n_way_join_simples_keeps_envs_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (kind : flambda_kind) (simples : list (Index.t * simple))
    (result : Or_bottom.t simple),
    Join_env.n_way_join_simples p fuel s kind simples = Some (result, s') ->
    Join_env.joined_envs s' = Join_env.joined_envs s.

Definition import_type_keeps_envs_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (ty : flambda_type),
    Join_env.import_type p fuel s ty = Some s' ->
    Join_env.joined_envs s' = Join_env.joined_envs s.

Definition n_way_join_env_extension_keeps_envs_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel : nat) (s s' : Join_env.t)
    (extensions : list (Index.t * env_extension)) (joined : env_extension),
    Join_env.n_way_join_env_extension p fuel s extensions
    = Some (Or_bottom.Ok (joined, s')) ->
    Join_env.joined_envs s' = Join_env.joined_envs s.

(* So the n-way join of types (FlambdaJoinTypes.v), given the three
   functions, keeps them: by the statement of
   FlambdaJoinTypesKeepsSpec.v about the relations that the three
   functions keep. *)
Definition n_way_join_keeps_envs_statement : Prop :=
  ∀ (p : Join_env.parameters) (fuel_of_env fuel : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (type_eqb : flambda_type -> flambda_type -> bool)
    (block_shape_eqb : block_shape -> block_shape -> bool),
    join_of_types_keeps_envs
      (n_way_join Join_env.joined_env
         (Join_env.n_way_join_simples p fuel_of_env)
         (Join_env.import_type p fuel_of_env)
         (Join_env.n_way_join_env_extension p fuel_of_env) expand_head
         reduce_head_of_kind_naked_immediate type_eqb block_shape_eqb fuel).

(* What the join that Simplify calls adds to the target is true: the
   conclusion of the two statements below. There is a state with the
   invariant, kinded, whose joined environments are the uses that are
   not bottom, each at its index, and relations true of it, such that
   the target answered is the final environment of the meet's
   environment made from the source, with the names to avoid of the
   state, by defining variables and adding two extensions; and at
   the run of a use that is not bottom, the
   equations of the two extensions are true where their variables
   have values, and the two maps of definitions of the analysis hold,
   in the valuation of the target that the state gives.

   The run is supposed to have: its blocks with tags in range, and
   the variables joined with values ([blocks_have_tags] and
   [joined_variables_have_values], two of the four conditions of
   [values_are_read], as they are in FlambdaJoinSpec.v); the
   variables brought in with values ([imported_have_values]: the
   contract's [imported_variables_have_values], read on the state);
   and the scrutinees of the relations kept with values
   ([relation_scrutinees_have_values], which is not the contract's
   [scrutinees_have_values]: see its CR).

   Nothing is said of what the functions of the meet's environment
   answer: so nothing of whether the target is satisfied, of what it
   keeps of the source, nor of the kinds of the variables of the
   analysis in it. *)
Definition join_adds_true {A meet_env meet_expanded_head : Type}
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (mh : meet_expanded_head) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A) : Prop :=
  ∃ (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (menv' menv'' : meet_env),
    state_true (source inputs) s
    ∧ state_kinded s
    ∧ relations_hold s relations
    ∧ (∀ (index : Index.t) (use : Typing_env.t),
         Join_env.joined_env s index = Some use
         <-> use_at inputs index use ∧ Typing_env.is_bottom use = false)
    ∧ (∀ binding : variable * flambda_kind,
         binding ∈ kept ->
         binding
         ∈ Bindings_in_target_env.existential_variables_in_order
             (Join_env.bindings s))
    ∧ Join_env.add_env_extension ops
        (variables_defined ops
           (Join_env.with_names_to_avoid ops
              (Join_env.create_meet_env ops (source inputs))
              (Bindings_in_target_env.names_to_avoid (Join_env.bindings s)))
           (Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings s)
            ++ kept))
        (Env_extension equations) mh
      = Some menv'
    ∧ Join_env.add_env_extension ops menv' (Env_extension relation_equations)
        mh
      = Some menv''
    ∧ target = Join_env.final_typing_env ops menv''
    ∧ ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
        (valuation : env),
        use_at inputs index use ->
        Typing_env.is_bottom use = false ->
        typing_env_satisfied_strictly statics valuation use ->
        blocks_have_tags (known_at_use inputs index) statics valuation ->
        joined_variables_have_values inputs valuation ->
        imported_have_values s use valuation ->
        relation_scrutinees_have_values s relations index use valuation ->
        equations_true_at statics (target_valuation s index use valuation)
          equations
        ∧ equations_true_at statics (target_valuation s index use valuation)
            relation_equations
        ∧ definitions_hold (Analysis.definitions_in_joined_envs analysis)
            index valuation (target_valuation s index use valuation)
        ∧ definitions_hold
            (Analysis.canonical_definitions_at_normal_mode analysis) index
            valuation (target_valuation s index use valuation).

(* [join_with] (FlambdaJoinEnvSpec.v), which is
   [cut_and_n_way_join_with_analysis] on the inputs of the contract,
   where some use is not bottom. Supposed: a join of types that meets
   its contract with the invariant and keeps the joined environments;
   the parameters as before; the inputs of the contract
   ([join_inputs_well_formed], [uses_kinded]); and that the uses have
   the names of the source that they know at the source's kinds
   ([uses_agree_with_source_on_kinds], which the contract does not
   ask). Then what the join adds to the target is true
   ([join_adds_true]). *)
Definition join_with_true_statement : Prop :=
  ∀ (A meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat) (choices : Join_env.choices)
    (jt : Join_env.n_way_join_type) (mh : meet_expanded_head)
    (avoid : list variable) (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    n_way_join_type_sound typing_env_invariant Join_env.joined_env
      Join_env.definitions_of_existentials kind_in_target_env
      (state_true (source inputs)) state_kinded jt ->
    join_of_types_keeps_envs jt ->
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    expand_const_sound p ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    (∃ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ∧ Typing_env.is_bottom use = false) ->
    join_with choices ops p fuel jt mh avoid inputs = Some (target, analysis)
    ->
    join_adds_true ops mh inputs target analysis.

(* The same of the compiler's join ([compiler_join],
   FlambdaJoinEnvSpec.v: the choices of the compiler) with the join
   of types of the model ([n_way_join_of_types],
   FlambdaJoinTypes.v). The join of types is no longer a premise:
   that it meets its contract with the invariant is
   [n_way_join_of_types_true] (FlambdaJoinTypesWithEnvSpec.v), and
   that it keeps the joined environments is
   [n_way_join_keeps_envs_statement] above. What is supposed of the
   parameters is what those two ask: of the expansion of heads and of
   the reduction of heads of kind naked immediate, their contracts
   (FlambdaJoinTypesSpec.v). *)
Definition compiler_join_true_statement : Prop :=
  ∀ (A meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel fuel_of_env fuel_of_types : nat)
    (expand_head : Typing_env.t -> flambda_type -> option Expanded_type.t)
    (reduce_head_of_kind_naked_immediate :
       Typing_env.t -> head_of_kind_naked_immediate
       -> Or_bottom.t head_of_kind_naked_immediate)
    (mh : meet_expanded_head) (avoid : list variable) (inputs : join_inputs A)
    (target : Typing_env.t) (analysis : Analysis.t A),
    fresh_avoids p ->
    parameters_type_eqb_sound p ->
    env_extension_eqb_sound p ->
    expand_const_sound p ->
    expand_head_sound typing_env_invariant expand_head ->
    reduce_head_of_kind_naked_immediate_sound
      reduce_head_of_kind_naked_immediate ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    (∃ (index : Index.t) (use : Typing_env.t),
       use_at inputs index use ∧ Typing_env.is_bottom use = false) ->
    compiler_join ops p fuel
      (n_way_join_of_types p fuel_of_env expand_head
         reduce_head_of_kind_naked_immediate fuel_of_types)
      mh avoid inputs
    = Some (target, analysis) ->
    join_adds_true ops mh inputs target analysis.

(* ---------------------------------------------------------------------- *)
(* The valuation of the target is in range                                 *)
(* ---------------------------------------------------------------------- *)

(* The constants of the definitions of the variables that the join
   has made are in range ([value_in_range], FlambdaRangeSpec.v). A
   definition is a vector of [Simple]s of the joined environments, so
   these are constants of the types of the uses. *)
Definition definitions_in_range (s : Join_env.t) : Prop :=
  ∀ (x : variable) (simples : Simples_in_joined_envs.t) (kind : flambda_kind)
    (index : Index.t) (c : const),
    lookup x (Join_env.definitions_of_existentials s) = Some (simples, kind)
    ->
    Simples_in_joined_envs.find index simples = Some (Const c) ->
    value_in_range (Value_const c) = true.

(* The valuation of the target that the state gives
   ([target_valuation], FlambdaJoinStateSpec.v) is in range
   ([env_in_range]) if the valuation of the use is and the constants
   of the definitions are: a variable of it has the value that the
   valuation of the use gives to a variable, or the address of a
   symbol, or a constant of a definition.

   That the constants of the definitions are in range is not shown
   of a state that the join reaches: it would follow from the
   constants of the types of the uses being in range, which no
   statement of the environments asks here. *)
Definition target_valuation_in_range_statement : Prop :=
  ∀ (s : Join_env.t) (index : Index.t) (use : Typing_env.t) (valuation : env),
    definitions_in_range s ->
    env_in_range valuation ->
    env_in_range (target_valuation s index use valuation).
