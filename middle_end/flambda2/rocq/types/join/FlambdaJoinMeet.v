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


(* The n-way join of typing environments with the meet: the functions
   of FlambdaJoinEnv.v, FlambdaJoinTypes.v and FlambdaMeet.v put
   together, with nothing left as an argument but the choices of the
   model, the depth of the join and fuel.

   This file is to be reviewed: it holds definitions only, and each is
   one function of those files applied to others.

   The join of environments (FlambdaJoinEnv.v) takes the functions of
   the meet's environment, the meet of expanded types and the join of
   types as arguments; the join of types (FlambdaJoinTypes.v) takes
   the three functions of the join of environments that it calls, the
   expansion of heads and the reduction of heads of naked immediates;
   the meet (FlambdaMeet.v) takes the join that it calls. Here each is
   given the others:
   - the functions of the meet's environment are those of [Meet_env]
     (FlambdaMeetEnv.v);
   - the join of types is [n_way_join_of_types]
     (FlambdaJoinTypesWithEnvSpec.v), with [expand_head]
     (FlambdaExpandHead.v) and [reduce_head_of_kind_naked_immediate]
     (FlambdaMeet.v);
   - the join that the meet calls is [Join_env.cut_and_n_way_join];
   - the meet of expanded types that the join is given is
     [Meet.meet_expanded_head] with that join.
   The compiler ties the same knot: types/meet_and_n_way_join.ml gives
   [Join_env.cut_and_n_way_join] its own [n_way_join] and
   [meet_expanded_head], and types/join_levels.ml does the same for
   the join that Simplify calls.

   What the functions here take and the compiler's do not:
   - the two choices of the model: the test for recovering relations
     ([Join_env.choices]) and the steps of the meet that have a
     repaired form ([meet_choices]); [Join_env.compiler_choices] and
     [compiler_choices] are the compiler's;
   - the depth at which the join stops looking at the variables that
     it makes, which the compiler reads from
     [Flambda_features.join_depth] (5 unless a flag says otherwise);
   - fuel, in three numbers ([fuels]): for the recursion of the three
     functions of the join of environments, for the depth of the join
     of types, and for the meet.
   An answer [None] is a fatal error of the compiler, or too little
   fuel, or the one state at which the model's meet stops and the
   compiler does not: an equation is being added to a name that the
   meet in between has made an alias of another (FlambdaMeetEnv.v,
   "A name that is no longer canonical"; discrepancy T4-10). The
   functions put together here are the transcriptions, each with the
   differences from the compiler that its file says: that nothing is
   stood in for does not make the whole the compiler's function on
   every input.
   The other parameters of the join of environments are fixed
   ([join_parameters]): a new variable is one greater than those to
   avoid ([fresh_name], FlambdaNameAbstractionSpec.v); the tests of
   equality that stand for the compiler's physical equality are those
   of FlambdaTypeEqualSpec.v; the type of a constant is that of
   [Expanded_type.create_const].

   One thing is converted. FlambdaJoinTypes.v and FlambdaExpandHead.v
   each have a transcription of [Expand_head.Expanded_type], with the
   same constructors: [join_expanded] takes the second to the first,
   and [expand_head_for_join] is [expand_head] followed by it. One of
   the two transcriptions should go. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypeEqualSpec
  FlambdaTypingEnv FlambdaJoinEnv FlambdaJoinSpec FlambdaJoinEnvSpec
  FlambdaJoinTypes FlambdaJoinTypesWithEnvSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec.
Import ListNotations.

(* The functions of the meet's environment that the join uses: those
   of [Meet_env]. *)
Definition meet_env_functions
    : Join_env.meet_env_functions Meet_env.t meet_expanded_head :=
  {| Join_env.create_meet_env := Meet_env.create;
     Join_env.current_typing_env := Meet_env.current_typing_env;
     Join_env.add_variable_definition := Meet_env.add_variable_definition;
     Join_env.add_env_extension := Meet_env.add_env_extension;
     Join_env.make_bottom := Meet_env.make_bottom;
     Join_env.final_typing_env := Meet_env.final_typing_env;
     Join_env.names_to_avoid := Meet_env.names_to_avoid;
     Join_env.with_names_to_avoid := Meet_env.with_names_to_avoid |}.

(* The parameters of the join of environments, but for the depth. *)
Definition join_parameters (join_depth : nat) : Join_env.parameters :=
  {| Join_env.fresh := fresh_name;
     Join_env.type_eqb := type_eqb;
     Join_env.env_extension_eqb := env_extension_eqb;
     Join_env.expand_const :=
       fun c =>
         FlambdaExpandHead.Expanded_type.to_type
           (FlambdaExpandHead.Expanded_type.create_const c);
     Join_env.join_depth := join_depth |}.

(* An expanded type of FlambdaExpandHead.v, as one of
   FlambdaJoinTypes.v. *)
Definition join_expanded (expanded : FlambdaExpandHead.Expanded_type.t)
    : FlambdaJoinTypes.Expanded_type.t :=
  FlambdaJoinTypes.Expanded_type.create
    (FlambdaExpandHead.Expanded_type.kind expanded)
    match FlambdaExpandHead.Expanded_type.descr expanded with
    | Unknown => Unknown
    | Bottom => Bottom
    | Ok (FlambdaExpandHead.Expanded_type.Head_value head) =>
      Ok (FlambdaJoinTypes.Expanded_type.Head_value head)
    | Ok (FlambdaExpandHead.Expanded_type.Head_naked_immediate head) =>
      Ok (FlambdaJoinTypes.Expanded_type.Head_naked_immediate head)
    | Ok (FlambdaExpandHead.Expanded_type.Head_naked_number k head) =>
      Ok (FlambdaJoinTypes.Expanded_type.Head_naked_number k head)
    | Ok (FlambdaExpandHead.Expanded_type.Head_rec_info head) =>
      Ok (FlambdaJoinTypes.Expanded_type.Head_rec_info head)
    | Ok (FlambdaExpandHead.Expanded_type.Head_region head) =>
      Ok (FlambdaJoinTypes.Expanded_type.Head_region head)
    end.

(* [Expand_head.expand_head], for the join of types. *)
Definition expand_head_for_join (tenv : Typing_env.t) (ty : flambda_type)
    : option FlambdaJoinTypes.Expanded_type.t :=
  option_map join_expanded (FlambdaExpandHead.expand_head tenv ty).

(* The fuel. *)
Record fuels : Type := {
  fuel_of_env : nat;
  fuel_of_types : nat;
  fuel_of_meet : nat
}.

Section Model.

Variable join_choices : Join_env.choices.
Variable meet_choices : meet_choices.
Variable join_depth : nat.
Variable fuel : fuels.

(* The join of types that the join of environments is given. *)
Definition model_join_of_types : Join_env.n_way_join_type :=
  n_way_join_of_types (join_parameters join_depth) (fuel_of_env fuel)
    expand_head_for_join reduce_head_of_kind_naked_immediate
    (fuel_of_types fuel).

(* The join that the meet calls: [Join_env.cut_and_n_way_join], given
   the meet of expanded types by the meet that calls it. *)
Definition model_join_scoped_envs : join_scoped_envs :=
  fun meet_head cut_after menv tenv envs =>
    Join_env.cut_and_n_way_join meet_env_functions
      (join_parameters join_depth) (fuel_of_env fuel) join_choices
      model_join_of_types meet_head cut_after menv tenv envs.

(* [Meet_and_n_way_join.meet_expanded_head]. *)
Definition model_meet_expanded_head : meet_expanded_head :=
  Meet.meet_expanded_head model_join_scoped_envs meet_choices
    (fuel_of_meet fuel).

(* [Meet.meet]. *)
Definition model_meet : meet_function :=
  Meet.meet model_join_scoped_envs meet_choices (fuel_of_meet fuel).

(* [Typing_env.add_equation] of flambda2_types.ml. *)
Definition model_add_equation : add_equation_function :=
  Flambda2_types.add_equation model_join_scoped_envs meet_choices
    (fuel_of_meet fuel).

(* [Join_env.cut_and_n_way_join_with_analysis], the join that Simplify
   calls, as a join of environments (FlambdaJoinSpec.v): the first
   argument is the names that new variables must avoid. *)
Definition model_join {A : Type} : join_of_environments A :=
  join_with join_choices meet_env_functions (join_parameters join_depth)
    (fuel_of_env fuel) model_join_of_types model_meet_expanded_head.

End Model.
