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


(* The join and the meet of the model, put together, under a renaming
   of variables.

   This file is to be reviewed: the definitions and the statements.
   FlambdaJoinMeet.v puts the join of environments, the join of types
   and the meet together, each given the others. The join of
   environments under renaming is FlambdaEnvRenameJoinSpec.v, and the
   meet under renaming is FlambdaMeetRenameSpec.v, which says what
   follows for the meet from a join that a renaming relates to another
   ([join_scoped_envs_related]). This file says that the model's join
   is such a join.

   The join makes variables, with a source of new names that is one of
   its parameters, and no source commutes with every renaming
   (FlambdaEnvRenameJoinSpec.no_source_commutes_statement). So the
   statements are about two joins: the join with some parameters, and
   the join with parameters that the renaming relates to them
   (FlambdaEnvRenameJoinSpec.parameters_related). For the model's own
   parameters the second is the model's join with the source of new
   names renamed (FlambdaEnvRenameJoinSpec.rename_fresh: the renaming
   is undone on the names to avoid, the source is asked, and what it
   answers is renamed), and nothing else changed.

   - [join_scoped_envs_with] is the join that the meet calls, with the
     parameters as an argument; the model's is that at the model's
     parameters ([model_join_scoped_envs_with_statement]), and
     [join_with_parameters] is the same for the join that Simplify
     calls ([model_join_with_statement]).
   - The functions of the environment of the meet that the join uses
     are related to themselves by every renaming, at the action on the
     environments of the meet and the relation between meets of
     expanded types of FlambdaMeetRenameSpec.v
     ([model_meet_env_functions_related_statement]).
   - Two joins with related parameters and related choices are related
     ([join_scoped_envs_with_related_statement]); so the model's join
     and the model's join with the renamed source are, with the
     compiler's choices and with the repaired ones
     ([model_join_scoped_envs_related_statement]).
   - So the meet of the model, of a renamed environment and renamed
     types, with the renamed source, answers the renamed answer
     ([model_meet_related_statement]), for choices of the meet that a
     renaming fixes (FlambdaMeetRenameSpec.meet_choices_closed), as
     the compiler's and the repaired ones are.
   - The join that Simplify calls, with the renamed source, of the
     renamed names to avoid and the renamed inputs, answers the
     renamed environment and the renamed analysis
     ([model_join_related_statement]).

   What this does not say: that the model's join or meet with one
   source of new names answers, of renamed arguments, the answer
   renamed up to the variables made. That needs one more fact, that
   two sources give answers that differ by a renaming of the variables
   made, which is not stated. Nothing is said of an answer [None] but
   that the two sides answer it together, and the fuel is the same on
   both sides.

   The proofs are in FlambdaJoinMeetRenameProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaJoinEnv
  FlambdaJoinSpec FlambdaJoinEnvSpec FlambdaJoinTypesWithEnvSpec
  FlambdaMeetGrammar FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaJoinMeet FlambdaMeetRenameSpec FlambdaEnvRenameJoinSpec.
Import ListNotations.

(* ======================================================================== *)
(* The joins of the model, with the parameters as an argument               *)
(* ======================================================================== *)

(* The join of types that the join of environments is given. *)
Definition join_of_types_with (p : Join_env.parameters) (fuel : fuels)
    : Join_env.n_way_join_type :=
  n_way_join_of_types p (fuel_of_env fuel) expand_head_for_join
    reduce_head_of_kind_naked_immediate (fuel_of_types fuel).

(* The join that the meet calls. *)
Definition join_scoped_envs_with (p : Join_env.parameters)
    (join_choices : Join_env.choices) (fuel : fuels) : join_scoped_envs :=
  fun meet_head cut_after menv tenv envs =>
    Join_env.cut_and_n_way_join meet_env_functions p (fuel_of_env fuel)
      join_choices (join_of_types_with p fuel) meet_head cut_after menv tenv
      envs.

(* The join that Simplify calls. *)
Definition join_with_parameters {A : Type} (p : Join_env.parameters)
    (join_choices : Join_env.choices) (meet_choices : meet_choices)
    (fuel : fuels) : join_of_environments A :=
  join_with join_choices meet_env_functions p (fuel_of_env fuel)
    (join_of_types_with p fuel)
    (Meet.meet_expanded_head (join_scoped_envs_with p join_choices fuel)
       meet_choices (fuel_of_meet fuel)).

(* The model's joins are these, at the model's parameters. *)
Definition model_join_scoped_envs_with_statement : Prop :=
  ∀ (join_choices : Join_env.choices) (join_depth : nat) (fuel : fuels),
    model_join_scoped_envs join_choices join_depth fuel
    = join_scoped_envs_with (join_parameters join_depth) join_choices fuel.

Definition model_join_with_statement : Prop :=
  ∀ (A : Type) (join_choices : Join_env.choices)
    (meet_choices : meet_choices) (join_depth : nat) (fuel : fuels),
    @model_join join_choices meet_choices join_depth fuel A
    = join_with_parameters (join_parameters join_depth) join_choices
        meet_choices fuel.

(* The model's parameters, with the source of new names renamed. *)
Definition renamed_join_parameters (r : renaming) (join_depth : nat)
    : Join_env.parameters :=
  with_fresh (rename_fresh r fresh_name) (join_parameters join_depth).

(* ======================================================================== *)
(* The join that the meet calls                                             *)
(* ======================================================================== *)

(* The functions of the environment of the meet that the join uses. *)
Definition model_meet_env_functions_related_statement : Prop :=
  ∀ r : renaming,
    meet_env_functions_related r (rename_meet_env r)
      (meet_expanded_head_related r) meet_env_functions meet_env_functions.

(* Two joins with related parameters and related choices. *)
Definition join_scoped_envs_with_related_statement : Prop :=
  ∀ (r : renaming) (p p' : Join_env.parameters)
    (choices choices' : Join_env.choices) (fuel : fuels),
    parameters_related r p p' ->
    choices_related r choices choices' ->
    join_scoped_envs_related r (join_scoped_envs_with p choices fuel)
      (join_scoped_envs_with p' choices' fuel).

(* The model's join, and the model's join with the renamed source. *)
Definition model_join_scoped_envs_related_statement : Prop :=
  ∀ (r : renaming) (join_depth : nat) (fuel : fuels),
    join_scoped_envs_related r
      (model_join_scoped_envs Join_env.compiler_choices join_depth fuel)
      (join_scoped_envs_with (renamed_join_parameters r join_depth)
         Join_env.compiler_choices fuel)
    ∧ join_scoped_envs_related r
        (model_join_scoped_envs Join_env.repaired_choices join_depth fuel)
        (join_scoped_envs_with (renamed_join_parameters r join_depth)
           Join_env.repaired_choices fuel).

(* ======================================================================== *)
(* The meet                                                                 *)
(* ======================================================================== *)

(* The meet of the model, and the meet with the join that has the
   renamed source: of a renamed environment and renamed types, the
   second answers the renamed answer of the first. *)
Definition model_meet_related_statement : Prop :=
  ∀ (r : renaming) (meet_choices : meet_choices) (join_depth : nat)
    (fuel : fuels),
    meet_choices_closed meet_choices ->
    meet_function_related r
      (model_meet Join_env.compiler_choices meet_choices join_depth fuel)
      (Meet.meet
         (join_scoped_envs_with (renamed_join_parameters r join_depth)
            Join_env.compiler_choices fuel)
         meet_choices (fuel_of_meet fuel))
    ∧ meet_function_related r
        (model_meet Join_env.repaired_choices meet_choices join_depth fuel)
        (Meet.meet
           (join_scoped_envs_with (renamed_join_parameters r join_depth)
              Join_env.repaired_choices fuel)
           meet_choices (fuel_of_meet fuel)).

(* ======================================================================== *)
(* The join that Simplify calls                                             *)
(* ======================================================================== *)

(* The inputs of a join: the source environment and the use
   environments are renamed; the scope and the identifiers of the uses
   are kept. *)
Definition rename_join_inputs {A : Type} (r : renaming)
    (inputs : join_inputs A) : join_inputs A :=
  {| source := rename_typing_env r (source inputs);
     cut_after := cut_after inputs;
     uses :=
       map
         (fun use : A * Typing_env.t =>
            (fst use, rename_typing_env r (snd use)))
         (uses inputs) |}.

(* Two joins with related parameters and related choices of the join,
   and choices of the meet that a renaming fixes: of the renamed names
   to avoid and the renamed inputs, the second answers the renamed
   environment and the renamed analysis of the first. *)
Definition join_with_parameters_related_statement : Prop :=
  ∀ (A : Type) (r : renaming) (p p' : Join_env.parameters)
    (choices choices' : Join_env.choices) (meet_choices : meet_choices)
    (fuel : fuels) (avoid : list variable) (inputs : join_inputs A),
    parameters_related r p p' ->
    choices_related r choices choices' ->
    meet_choices_closed meet_choices ->
    join_with_parameters p' choices' meet_choices fuel
      (map (rename_variable r) avoid) (rename_join_inputs r inputs)
    = option_map
        (fun result : Typing_env.t * Analysis.t A =>
           (rename_typing_env r (fst result), rename_analysis r (snd result)))
        (join_with_parameters p choices meet_choices fuel avoid inputs).

(* The model's join, and the model's join with the renamed source,
   with the compiler's choices of the join. *)
Definition model_join_related_statement : Prop :=
  ∀ (A : Type) (r : renaming) (meet_choices : meet_choices)
    (join_depth : nat) (fuel : fuels) (avoid : list variable)
    (inputs : join_inputs A),
    meet_choices_closed meet_choices ->
    join_with_parameters (renamed_join_parameters r join_depth)
      Join_env.compiler_choices meet_choices fuel
      (map (rename_variable r) avoid) (rename_join_inputs r inputs)
    = option_map
        (fun result : Typing_env.t * Analysis.t A =>
           (rename_typing_env r (fst result), rename_analysis r (snd result)))
        (@model_join Join_env.compiler_choices meet_choices join_depth fuel
           A avoid inputs).
