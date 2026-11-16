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


(* The types of the equations that the join adds fit the environment
   of the meet: what FlambdaJoinMeetSpec.v leaves as a premise
   ([types_fit_otherwise], [join_adds_types_that_fit_otherwise]) is
   true of the join put together in FlambdaJoinMeet.v.

   This file is to be reviewed: the statements. The proofs are in
   FlambdaJoinMeetFitsProofs.v, which needs no review.

   An extension that the environment of the meet is given is supposed
   to have types that are well formed and in which the names of the
   equations inside them are of the kinds of their types
   ([type_fits], FlambdaMeetSpec.v, with the variables and the aliases
   that FlambdaJoinMeetSpec.v has). Of the two extensions that the
   last phase of the join adds:

   - the types are well formed: FlambdaJoinWellFormedSpec.v, for the
     state that the first phases answer and for what the last phase
     makes of it;
   - the names of the equations inside the types are of their kinds
     in the target, by the kinds that the bindings of the state
     record: FlambdaJoinEquationNamesSpec.v. Here: so they are in the
     environment of the meet in which the variables of the target
     are defined ([last_phase_equation_names_known_statement]), as
     [last_phase_kinds_known_statement] (FlambdaJoinMeetSpec.v) says
     of the aliases, and under its premises;
   - so the join that the meet calls adds types that fit otherwise
     ([model_join_adds_types_that_fit_otherwise_statement]), and has
     the conclusions of the contract that the meet has for its join
     that are not about values, with no premise left about the join
     itself ([model_join_scoped_envs_keeps_outright_statement]:
     [model_join_scoped_envs_keeps_statement] of FlambdaJoinMeetSpec.v
     with its premise discharged). The same of the conclusion about
     values that FlambdaJoinMeetSpec.v has on the runs that have its
     conditions ([model_join_scoped_envs_satisfied_outright_statement]:
     [model_join_scoped_envs_satisfied_statement] with its premise
     about the join discharged, and nothing else changed).

   The expansion of heads that the join of types is given has what
   the join of types supposes of it for the names of equations
   ([expand_head_for_join_equations_kinded_statement]).

   The same of the join that Simplify calls ([model_join]), in a
   section of its own. The statements of FlambdaJoinMeetSpec.v about
   it have the premise [types_fit_otherwise] for the run that they
   name.
   Here it follows from their premises about the inputs and one
   more, [uses_fit_otherwise]: the types of the uses that are joined
   are well formed and have the names of their equations at their
   kinds ([model_join_run_adds_types_that_fit_otherwise_statement]).
   So that join has five of the six parts of [join_guarantees]
   (FlambdaJoinSpec.v) with no premise about a run: the four that
   are about the target ([model_join_keeps_outright_statement]) and
   what the analysis speaks of
   ([model_join_describes_uses_outright_statement]). What stays a
   premise of the two: that the meet of expanded types of the model
   has the half of what the meet's contract for its join supposes of
   the one it is given that is not about values
   ([meet_expanded_head_keeps]). The sixth part, about values, is
   had on the runs that have the conditions, of that join made with
   any meet of expanded types that has both halves
   ([model_join_with_admits_use_outright_statement]); it is not
   stated at the meet of the model, which is not shown to have the
   half about values and may not have it.

   The meet of the model has the half of that which is not about
   values ([meet_expanded_head_keeps]), for the choices of the meet
   that its contract is proved of
   ([model_meet_expanded_head_keeps_statement]): the meet has it at
   every fuel from a join that has the conclusions of its contract
   that are not about values (FlambdaMeetKeepsSpec.v), and the join
   of the model has them outright (above). So the join that Simplify
   calls has those five parts of [join_guarantees] under premises
   about its inputs and the choices of the meet only
   ([model_join_guarantees_on_inputs_statement]): nothing is supposed
   of a function of the model. And the meet of the model, with
   such choices, adds no variable that may occur in terms
   ([model_meet_keeps_terms_statement]: [meet_keeps_terms],
   FlambdaMeetTermsSpec.v, which is what the simplification of a
   [Switch] supposes of the meet that it is given,
   FlambdaSimplifySwitchArmsSpec.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaJoinEnv FlambdaJoinSpec
  FlambdaJoinEnvSpec FlambdaJoinStateSpec FlambdaJoinDefineSpec
  FlambdaJoinFitsSpec FlambdaJoinAddsSpec FlambdaJoinKindsSpec
  FlambdaJoinTypes FlambdaJoinTypesSpec FlambdaMeetGrammar FlambdaMeetEnv
  FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec FlambdaMeetEnvSpec
  FlambdaMeetKeepsSpec FlambdaJoinMeet FlambdaJoinMeetSpec
  FlambdaJoinEquationNamesSpec FlambdaMeetTermsSpec.
Import ListNotations.

(* The expansion of heads that the join of types is given in
   FlambdaJoinMeet.v keeps the names of the equations inside a type
   at their kinds. *)
Definition expand_head_for_join_equations_kinded_statement : Prop :=
  FlambdaJoinTypesSpec.expand_head_equations_kinded expand_head_for_join.

(* The names of the equations inside the types of the equations that
   the last phase adds are of their kinds in the environment in which
   the variables are defined. The premises are those of
   [last_phase_kinds_known_statement] (FlambdaJoinMeetSpec.v), with
   [equation_names_kinded] (FlambdaJoinEquationNamesSpec.v) of the
   state where that has the kinds of its aliases. *)
Definition last_phase_equation_names_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (fuel : nat) (source : Typing_env.t)
    (s : Join_env.t) (relations : Join_env.inverse_relations)
    (menv : Meet_env.t) (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    state_true source s ->
    relations_hold s relations ->
    relations_nonempty relations ->
    state_scoped s ->
    state_kinded s ->
    joined_envs_are typing_env_invariant s ->
    joined_envs_are typing_env_kinded s ->
    symbols_of_joined_envs_in_source source s ->
    equation_names_kinded s ->
    Meet_env.typing_env menv = source ->
    typing_env_fits (Meet_env.typing_env menv) ->
    last_phase_adds tenv fuel s relations kept equations relation_equations
      definitions ->
    let menv' :=
      variables_defined meet_env_functions menv
        (Bindings_in_target_env.imported_variables_in_order
           (Join_env.bindings s)
         ++ kept)
    in
    ∀ (n : name) (ty : flambda_type),
      (n, ty) ∈ equations ++ relation_equations ->
      type_equations_kinded (Meet_env.typing_env menv') ty.

(* The join that the meet calls adds types that fit otherwise. *)
Definition model_join_adds_types_that_fit_otherwise_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels),
    join_adds_types_that_fit_otherwise join_choices join_depth fuel.

(* So it has the conclusions of the contract that the meet has for
   its join that are not about values. *)
Definition model_join_scoped_envs_keeps_outright_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels),
    join_scoped_envs_keeps
      (model_join_scoped_envs join_choices join_depth fuel).

(* And it has the conclusion of that contract about values, on the
   runs that have the conditions of FlambdaJoinMeetSpec.v: the
   statement there ([model_join_scoped_envs_satisfied_statement]),
   word for word, without its premise about the join. *)
Definition model_join_scoped_envs_satisfied_outright_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels)
    (meet_head : meet_expanded_head) (initial_env result : Meet_env.t)
    (scoped_envs : list Typing_env.t) (envs : Joined_envs.t)
    (s1 s2 s3 : Join_env.t)
    (relations2 relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (scoped_env : Typing_env.t) (statics : static_env)
    (valuation : env),
    let initial_tenv := Meet_env.typing_env initial_env in
    let p := join_parameters join_depth in
    let jt := model_join_of_types join_depth fuel in
    let avoid :=
      Meet_env.names_to_avoid initial_env
      ++ Join_env.variables_of_env initial_tenv
      ++ flat_map Join_env.variables_of_env scoped_envs
    in
    let s0 :=
      Join_env.create envs
        (Bindings_in_target_env.from_source_env avoid initial_tenv)
    in
    let menv :=
      Meet_env.with_names_to_avoid initial_env
        (Bindings_in_target_env.names_to_avoid (Join_env.bindings s3))
    in
    meet_expanded_head_sound meet_head ->
    meet_expanded_head_keeps meet_head ->
    typing_env_fits initial_tenv ->
    (∀ scoped_env : Typing_env.t,
      scoped_env ∈ scoped_envs -> scoped_env_of initial_tenv scoped_env) ->
    scoped_envs_agree scoped_envs ->
    model_join_scoped_envs join_choices join_depth fuel meet_head
      (Typing_env.current_scope initial_tenv) initial_env initial_tenv
      scoped_envs
    = Some result ->
    Join_env.cut_for_joins (Typing_env.current_scope initial_tenv)
      (Index.number scoped_envs)
    = Some envs ->
    Join_env.join_aliases_into_bindings p (fuel_of_env fuel) s0 envs
    = Some s1 ->
    Join_env.n_way_join_delayed_equations (fuel_of_env fuel) join_choices jt
      s1 [] (Join_env.new_equations_in_joined_envs s1 s0)
    = Some (relations2, s2) ->
    Join_env.n_way_join_loop join_depth p (fuel_of_env fuel) join_choices jt
      s2 relations2 (Join_env.new_definitions_of_existentials s2 s0)
    = Some (relations3, s3) ->
    last_phase_adds
      (Meet_env.typing_env
         (variables_defined meet_env_functions menv
            (Bindings_in_target_env.imported_variables_in_order
               (Join_env.bindings s3))))
      (fuel_of_env fuel) s3 relations3 kept equations relation_equations
      definitions ->
    Join_env.joined_env s3 index = Some scoped_env ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation initial_tenv ->
    typing_env_satisfied_strictly statics valuation scoped_env ->
    tags_at s3 scoped_env statics valuation ->
    aliases_have_values s3 index scoped_env valuation ->
    relation_scrutinees_have_values s3 relations3 index scoped_env valuation
    ->
    (∀ (n : name) (ty : flambda_type),
       (n, ty) ∈ equations ++ relation_equations ->
       ∃ v : value,
         value_of_name (target_valuation s3 index scoped_env valuation) n
         = Some v) ->
    ∃ valuation' : env,
      extended_from scoped_env valuation valuation'
      ∧ typing_env_satisfied_strictly statics valuation'
          (Meet_env.typing_env result)
      ∧ keeps_equations_valued initial_tenv (Meet_env.typing_env result)
          valuation valuation'.

(* ======================================================================== *)
(* The join that Simplify calls                                             *)
(* ======================================================================== *)

(* What is supposed more of the uses that are joined (those that are
   not bottom), beside the premises of the contract of
   FlambdaJoinSpec.v: the types of each are well formed, and the names
   of the equations inside them are of the kinds of their types. With
   the invariant and the absence of bottom types that the contract
   supposes, and the kinds ([uses_kinded]), it is that a meet may be
   made in each such use ([typing_env_fits], FlambdaMeetSpec.v). *)
(* CR mshinwell: the contract of the join (FlambdaJoinSpec.v) does not
   suppose this of the uses, as it does not suppose that a meet may be
   made in the source. It is supposed here because the join puts
   types of the uses, and joins of them, into the target, and a meet
   may be made in the target only if those are well formed and have
   the names of their equations at their kinds. Whether every
   environment that Simplify gives the join has it is not known. *)
Definition uses_fit_otherwise {A : Type} (inputs : join_inputs A) : Prop :=
  ∀ (index : Index.t) (use : Typing_env.t),
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    types_well_formed use ∧ types_equations_kinded use.

(* The types of the equations that the last phase of a run of that
   join adds fit otherwise, in the environment in which its variables
   are defined: the premise [types_fit_otherwise] of the statements
   of FlambdaJoinMeetSpec.v about that join. The premises are theirs
   about the inputs, and [uses_fit_otherwise]. *)
Definition model_join_run_adds_types_that_fit_otherwise_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (avoid : list variable) (inputs : join_inputs A)
    (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind)),
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join_run join_choices join_depth fuel avoid inputs envs s3
      relations3 kept equations relation_equations definitions ->
    types_fit_otherwise
      (Meet_env.typing_env (model_join_defined inputs s3 kept))
      (equations ++ relation_equations).

(* So the join that Simplify calls has the four parts of
   [join_guarantees] that [model_join_keeps_statement]
   (FlambdaJoinMeetSpec.v) concludes, with no premise about a run: a
   meet may be made in the target, the target keeps the source, and
   the variables that it has and the source has not are as the
   contract says. What is left of the premises of that statement:
   those about the inputs, and that the meet of expanded types of the
   model has what the meet's contract for its join supposes of the one
   it is given ([meet_expanded_head_keeps]). *)
Definition model_join_keeps_outright_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    meet_expanded_head_keeps
      (model_meet_expanded_head join_choices meet_choices join_depth fuel) ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join join_choices meet_choices join_depth fuel avoid inputs
    = Some (target, analysis) ->
    typing_env_fits target
    ∧ join_keeps_source (source inputs) target
    ∧ join_introduces avoid inputs target.

(* And what the analysis speaks of ([analysis_describes_uses],
   FlambdaJoinSpec.v): the conclusion of
   [model_join_describes_uses_statement] (FlambdaJoinMeetSpec.v),
   likewise with no premise about a run. *)
Definition model_join_describes_uses_outright_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    meet_expanded_head_keeps
      (model_meet_expanded_head join_choices meet_choices join_depth fuel) ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join join_choices meet_choices join_depth fuel avoid inputs
    = Some (target, analysis) ->
    analysis_describes_uses inputs target analysis.

(* And the valuations, on the runs that have the conditions, for
   that join made with any meet of expanded types that has the two
   contracts ([model_join_with], FlambdaJoinMeetSpec.v):
   [model_join_with_admits_use_statement] of that file, word for
   word, with [uses_fit_otherwise] of the inputs in the place of its
   premise about the types that the last phase adds. Its conclusion
   is that of [join_admits_use] (FlambdaJoinSpec.v). The join that
   Simplify calls is this join at the meet of the model; that meet is
   not shown to have the second contract ([meet_expanded_head_sound])
   and may not have it, so nothing is stated here at that meet. *)
Definition model_join_with_admits_use_outright_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (join_depth : nat)
    (fuel : fuels) (mh : meet_expanded_head) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t) (analysis : Analysis.t A)
    (envs : Joined_envs.t) (s3 : Join_env.t)
    (relations3 : Join_env.inverse_relations)
    (kept : list (variable * flambda_kind))
    (equations relation_equations : list (name * flambda_type))
    (definitions : finite_map (Simples_in_joined_envs.t * flambda_kind))
    (index : Index.t) (use : Typing_env.t) (statics : static_env)
    (valuation : env),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join_with join_choices join_depth fuel mh avoid inputs
    = Some (target, analysis) ->
    model_join_run join_choices join_depth fuel avoid inputs envs s3
      relations3 kept equations relation_equations definitions ->
    use_at inputs index use ->
    Typing_env.is_bottom use = false ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation use ->
    typing_env_satisfied_strictly statics valuation (source inputs) ->
    tags_at s3 use statics valuation ->
    aliases_have_values s3 index use valuation ->
    relation_scrutinees_have_values s3 relations3 index use valuation ->
    (∀ (n : name) (ty : flambda_type),
       (n, ty) ∈ equations ++ relation_equations ->
       ∃ v : value,
         value_of_name (target_valuation s3 index use valuation) n = Some v) ->
    ∃ valuation' : env,
      typing_env_satisfied_strictly statics valuation' target
      ∧ agrees_on_known use valuation valuation'
      ∧ agrees_on_known (source inputs) valuation valuation'
      ∧ definitions_hold (Analysis.definitions_in_joined_envs analysis) index
          valuation valuation'
      ∧ definitions_hold
          (Analysis.canonical_definitions_at_normal_mode analysis) index
          valuation valuation'.

(* ======================================================================== *)
(* The join and the meet of the model together                              *)
(* ======================================================================== *)

(* The meet of expanded types of the model keeps what a meet keeps
   ([meet_expanded_head_keeps], FlambdaMeetSpec.v), whatever the
   choices of the join, the depth and the fuel, for choices of the
   meet as its contract has them: the type of the [Get_tag] step that
   is true, no variable defined again, the variables of a single case
   lifted ([repaired_choices], FlambdaMeet.v, has the three). *)
Definition model_meet_expanded_head_keeps_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels),
    type_of_get_tag_scrutinee meet_choices
    = immediate_or_blocks_with_these_tags ->
    define_again meet_choices = false ->
    lift_variables_of_single_case meet_choices = true ->
    meet_expanded_head_keeps
      (model_meet_expanded_head join_choices meet_choices join_depth fuel).

(* So the meet of the model, with such choices, adds no variable
   that may occur in terms ([meet_keeps_terms],
   FlambdaMeetTermsSpec.v). *)
Definition model_meet_keeps_terms_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels),
    type_of_get_tag_scrutinee meet_choices
    = immediate_or_blocks_with_these_tags ->
    define_again meet_choices = false ->
    lift_variables_of_single_case meet_choices = true ->
    meet_keeps_terms (model_meet join_choices meet_choices join_depth fuel).

(* And the join that Simplify calls, with such choices of the meet,
   has five of the six parts of [join_guarantees] (FlambdaJoinSpec.v)
   under premises about its inputs only: the conclusions of
   [model_join_keeps_outright_statement] and of
   [model_join_describes_uses_outright_statement], without their
   premise about the meet. (That a meet may be made in the target
   says that the target has the invariant of typing environments and
   no bottom type, the first two parts.) The sixth part,
   [join_admits_use], is not here: see
   [model_join_with_admits_use_outright_statement]. *)
Definition model_join_guarantees_on_inputs_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (inputs : join_inputs A) (target : Typing_env.t)
    (analysis : Analysis.t A),
    type_of_get_tag_scrutinee meet_choices
    = immediate_or_blocks_with_these_tags ->
    define_again meet_choices = false ->
    lift_variables_of_single_case meet_choices = true ->
    join_inputs_well_formed typing_env_invariant avoid inputs ->
    uses_kinded inputs ->
    uses_agree_with_source_on_kinds inputs ->
    typing_env_fits (source inputs) ->
    uses_fit_otherwise inputs ->
    model_join join_choices meet_choices join_depth fuel avoid inputs
    = Some (target, analysis) ->
    typing_env_fits target
    ∧ join_keeps_source (source inputs) target
    ∧ join_introduces avoid inputs target
    ∧ analysis_describes_uses inputs target analysis.
