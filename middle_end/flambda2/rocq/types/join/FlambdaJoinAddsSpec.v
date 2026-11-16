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


(* What the last phase of the n-way join of environments adds to the
   target, as data: the variables that it defines and the equations
   of the two extensions that it adds.

   This file is to be reviewed: the definitions at its start, and the
   statements. The proofs are in FlambdaJoinAddsProofs.v, which needs
   no review.

   The statements of FlambdaJoinDefineSpec.v and of
   FlambdaJoinFitsSpec.v about the last phase
   ([Join_env.define_or_eliminate_variables_and_add_equations],
   FlambdaJoinEnv.v) each say "there are variables and two lists of
   equations such that the target is the environment given with the
   variables defined and the two extensions added, and ...". Two such
   statements cannot be used together: nothing says that the
   variables and the equations of the one are those of the other.
   Here they are named.

   The last phase uses the environment of the meet through functions
   that it is given, and does not look at what they answer but to
   choose, among the aliases of a variable that the join has made,
   the one that the target binds first (it reads the typing
   environment once, after it has defined the variables brought in).
   So what it defines and what it adds can be had by running it with
   functions that only write down what they are asked ([recording]):
   the variables defined, in order, and the extensions added, in
   order. [last_phase_adds] says of some variables, two lists of
   equations and a map of definitions that they are what such a run
   writes down and answers, for a given typing environment in which
   to choose aliases.

   - That relation has at most one answer
     ([last_phase_adds_functional_statement]).
   - A run of the last phase with any functions is the run that
     records, replayed ([last_phase_replayed_statement]): if it
     answers, there are variables and equations that the last phase
     adds, for the typing environment that those functions answer
     after the variables brought in are defined, and the target is
     the environment given with the variables brought in and those
     variables defined and the two extensions added.
   - The same for the join up to what it adds
     ([cut_and_n_way_join0_replayed_statement]), which also names the
     states and the relations that the first phases answer, by the
     equations that determine them.
   - The variables that the last phase defines are variables that the
     join has made, each once ([last_phase_adds_kept_statement]).
   - The equations are on variables
     ([last_phase_adds_on_variables_statement]).
   - What the equations mean ([last_phase_adds_true_statement]): the
     conclusion of [define_variables_true_statement], of these
     equations.
   - The variables of the equations
     ([last_phase_adds_in_scope_statement]): the conclusion of
     [define_variables_in_scope_statement], of these equations.
   - The definitions answered are of variables that the target is to
     have ([last_phase_adds_definitions_statement]).

   The two before the last are those statements at the functions that
   record. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinFitsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The functions that record                                                *)
(* ======================================================================== *)

(* What has been asked of the environment of the meet: the variables
   defined, with their kinds, in the order in which they were
   defined; and the extensions added, in the order in which they were
   added. *)
Record recorded : Type := {
  recorded_variables : list (variable * flambda_kind);
  recorded_extensions : list env_extension
}.

Definition nothing_recorded : recorded :=
  {| recorded_variables := []; recorded_extensions := [] |}.

(* The functions of the meet's environment that only write down what
   they are asked. The typing environment that they answer is always
   the one given; adding an extension never fails; the name mode of a
   definition is not written down (the last phase defines every
   variable at the mode of variables that exist only in types). *)
Definition recording (tenv : Typing_env.t)
    : Join_env.meet_env_functions recorded unit :=
  {| Join_env.create_meet_env := fun _ => nothing_recorded;
     Join_env.current_typing_env := fun _ => tenv;
     Join_env.add_variable_definition :=
       fun r var kind _ =>
         {| recorded_variables := recorded_variables r ++ [(var, kind)];
            recorded_extensions := recorded_extensions r |};
     Join_env.add_env_extension :=
       fun r extension _ =>
         Some
           {| recorded_variables := recorded_variables r;
              recorded_extensions := recorded_extensions r ++ [extension] |};
     Join_env.make_bottom := fun r => r;
     Join_env.final_typing_env := fun _ => tenv;
     Join_env.names_to_avoid := fun _ => [];
     Join_env.with_names_to_avoid := fun r _ => r |}.

(* ======================================================================== *)
(* What the last phase adds                                                 *)
(* ======================================================================== *)

(* The variables, the equations of the first extension, those of the
   second and the definitions are what the last phase adds at a state,
   with relations, where aliases are chosen in [tenv]: the run that
   records, from nothing recorded, answers the definitions, and has
   written down the variables brought in and then the variables, and
   the two extensions. *)
Definition last_phase_adds (tenv : Typing_env.t) (fuel : nat)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    : Prop :=
  Join_env.define_or_eliminate_variables_and_add_equations (recording tenv)
    fuel tt s nothing_recorded relations
  = Some
      ({| recorded_variables :=
            Bindings_in_target_env.imported_variables_in_order
              (Join_env.bindings s)
            ++ kept;
          recorded_extensions :=
            [Env_extension equations; Env_extension relation_equations] |},
       definitions).

(* There is at most one answer. *)
Definition last_phase_adds_functional_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept kept' : list (variable * flambda_kind))
    (equations equations' relation_equations relation_equations'
     : list (name * flambda_type))
    (definitions definitions'
     : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    last_phase_adds tenv fuel s relations kept' equations' relation_equations'
      definitions' ->
    kept' = kept
    ∧ equations' = equations
    ∧ relation_equations' = relation_equations
    ∧ definitions' = definitions.

(* A run of the last phase with any functions is that run, replayed:
   aliases are chosen in the typing environment that the functions
   answer once the variables brought in are defined. *)
Definition last_phase_replayed_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (fuel : nat) (mh : meet_expanded_head) (s : Join_env.t)
    (menv target : meet_env) (relations : Join_env.inverse_relations)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    Join_env.define_or_eliminate_variables_and_add_equations ops fuel mh s
      menv relations
    = Some (target, definitions) ->
    ∃ (kept : list (variable * flambda_kind))
      (equations relation_equations : list (name * flambda_type))
      (menv' : meet_env),
      last_phase_adds
        (Join_env.current_typing_env ops
           (variables_defined ops menv
              (Bindings_in_target_env.imported_variables_in_order
                 (Join_env.bindings s))))
        fuel s relations kept equations relation_equations definitions
      ∧ Join_env.add_env_extension ops
          (variables_defined ops menv
             (Bindings_in_target_env.imported_variables_in_order
                (Join_env.bindings s)
              ++ kept))
          (Env_extension equations) mh
        = Some menv'
      ∧ Join_env.add_env_extension ops menv' (Env_extension relation_equations)
          mh
        = Some target.

(* The same for the join up to what it adds, where there is an
   environment to join. The states and the relations that the first
   phases answer are named by the equations that determine them. The
   environment that the last phase is given is the one given to the
   join with the names of the last state for names to avoid. *)
Definition cut_and_n_way_join0_replayed_statement : Prop :=
  ∀ (meet_env meet_expanded_head : Type)
    (ops : Join_env.meet_env_functions meet_env meet_expanded_head)
    (p : Join_env.parameters) (fuel : nat) (choices : Join_env.choices)
    (jt : Join_env.n_way_join_type) (mh : meet_expanded_head)
    (avoid : list variable) (menv target : meet_env) (source : Typing_env.t)
    (envs : Joined_envs.t)
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (names : list variable),
    Keyed.is_empty envs = false ->
    Join_env.cut_and_n_way_join0 ops p fuel choices jt mh avoid menv source
      envs
    = Some (target, definitions, names) ->
    ∃ (s1 s2 s3 : Join_env.t)
      (relations2 relations3 : Join_env.inverse_relations)
      (kept : list (variable * flambda_kind))
      (equations relation_equations : list (name * flambda_type))
      (menv' : meet_env),
      Join_env.join_aliases_into_bindings p fuel
        (Join_env.create envs
           (Bindings_in_target_env.from_source_env avoid source))
        envs
      = Some s1
      ∧ Join_env.n_way_join_delayed_equations fuel choices jt s1 []
          (Join_env.new_equations_in_joined_envs s1
             (Join_env.create envs
                (Bindings_in_target_env.from_source_env avoid source)))
        = Some (relations2, s2)
      ∧ Join_env.n_way_join_loop (Join_env.join_depth p) p fuel choices jt s2
          relations2
          (Join_env.new_definitions_of_existentials s2
             (Join_env.create envs
                (Bindings_in_target_env.from_source_env avoid source)))
        = Some (relations3, s3)
      ∧ names = Bindings_in_target_env.names_to_avoid (Join_env.bindings s3)
      ∧ last_phase_adds
          (Join_env.current_typing_env ops
             (variables_defined ops
                (Join_env.with_names_to_avoid ops menv names)
                (Bindings_in_target_env.imported_variables_in_order
                   (Join_env.bindings s3))))
          fuel s3 relations3 kept equations relation_equations definitions
      ∧ Join_env.add_env_extension ops
          (variables_defined ops (Join_env.with_names_to_avoid ops menv names)
             (Bindings_in_target_env.imported_variables_in_order
                (Join_env.bindings s3)
              ++ kept))
          (Env_extension equations) mh
        = Some menv'
      ∧ Join_env.add_env_extension ops menv' (Env_extension relation_equations)
          mh
        = Some target.

(* ======================================================================== *)
(* Of the variables and the equations that it adds                          *)
(* ======================================================================== *)

(* The variables that the last phase defines are variables that the
   join has made, and none is defined twice. *)
Definition last_phase_adds_kept_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    NoDup (map fst kept)
    ∧ ∀ binding : variable * flambda_kind,
        binding ∈ kept ->
        binding
        ∈ Bindings_in_target_env.existential_variables_in_order
            (Join_env.bindings s).

(* Every equation is on a variable. *)
Definition last_phase_adds_on_variables_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (s : Join_env.t)
    (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations ++ relation_equations ->
      ∃ x : variable, n = Name_var x.

(* What the equations mean: as [define_variables_true_statement]
   (FlambdaJoinDefineSpec.v). *)
Definition last_phase_adds_true_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (source : Typing_env.t)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    state_kinded s ->
    joined_envs_are typing_env_kinded s ->
    relations_hold s relations ->
    relations_nonempty relations ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (index : Index.t) (use : Typing_env.t) (statics : static_env)
      (valuation : env),
      Join_env.joined_env s index = Some use ->
      typing_env_satisfied_strictly statics valuation use ->
      tags_at s use statics valuation ->
      aliases_have_values s index use valuation ->
      relation_scrutinees_have_values s relations index use valuation ->
      equations_true_at statics (target_valuation s index use valuation)
        equations
      ∧ equations_true_at statics (target_valuation s index use valuation)
          relation_equations
      ∧ definitions_hold definitions index valuation
          (target_valuation s index use valuation).

(* The variables of the equations: as
   [define_variables_in_scope_statement] (FlambdaJoinFitsSpec.v). *)
Definition last_phase_adds_in_scope_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (source : Typing_env.t)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    relations_hold s relations ->
    relations_nonempty relations ->
    equations_in_target s ->
    aliases_not_made s ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    equations_in_final_target s kept equations
    ∧ equations_in_final_target s kept relation_equations.

(* The definitions that the last phase answers are of variables that
   the target is to have ([in_final_target], FlambdaJoinFitsSpec.v):
   a definition is that of a variable that the join has made and the
   phase defines, or has been moved to the variable of the target
   that stands for one that it removes. *)
Definition last_phase_adds_definitions_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (source : Typing_env.t)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    aliases_not_made s ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    ∀ (x : variable) (definition : Simples_in_joined_envs.t * flambda_kind),
      lookup x definitions = Some definition -> in_final_target s kept x.
