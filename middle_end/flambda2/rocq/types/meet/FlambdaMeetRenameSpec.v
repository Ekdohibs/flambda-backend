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


(* The meet commutes with a renaming of variables.

   This file is to be reviewed: the action of a renaming on what the
   functions of the meet take and answer, and the statements. The
   renaming of types is FlambdaTypeRenameSpec.v, of typing
   environments FlambdaEnvRenameSpec.v, of an expanded type
   FlambdaProversRenameSpec.v ([rename_expanded], with the statement
   that [expand_head] commutes with it). The meet is FlambdaMeet.v and
   FlambdaMeetEnv.v; its contract, FlambdaMeetSpec.v, does not speak
   of renaming.

   The compiler has no such statement: its meet is not given renamed
   environments. It is here because every function of the model that
   reads names is to commute with renaming, and because the rewrites
   that ask the meet (the arms of a [Switch] that the type of the
   scrutinee excludes, FlambdaSimplifySwitchSpec.v) are to commute
   with it for the validation to use them.

   What is stated is an equation: of a renamed environment and
   renamed types, the meet answers the renamed answer. The meet makes
   no variable and goes by no order of names; where it chooses between
   two names it is by their binding times, which a renaming keeps.
   The join that it is given makes variables, and is a parameter: the
   statement is for a join that commutes with renaming in the same
   sense ([join_scoped_envs_equivariant]). The join that forgets
   (FlambdaMeetExamplesSpec.v) does. A join that makes new variables
   by a choice that looks at the names it is given has not that
   property: the second part of the file is for such a join, with
   its source of new names renamed. A statement up to the choice of
   the new variables, for one source of names, is not made and not
   proved: it would need a contract of the join that says which
   variables are new and where they may occur.

   The statements, proved in FlambdaMeetRenameProofs.v, which needs
   no review.
   - The meet of expanded types commutes with renaming at every fuel
     ([meet_expanded_head_at_fuel_equivariant_statement]), and so
     does the meet of types ([meet_of_types_equivariant_statement]):
     for every join that commutes, and for choices ([meet_choices],
     FlambdaMeet.v) whose type for the scrutinee of a tag every
     renaming of variables fixes ([meet_choices_closed]). The other
     two choices are booleans.
   - Both records of choices are such
     ([compiler_choices_closed_statement],
     [repaired_choices_closed_statement]); so the statement holds of
     the meet that the contract is proved of and of the meet as the
     compiler has it ([meet_to_prove_equivariant_statement],
     [compiler_meet_equivariant_statement]).
   - The join that forgets commutes
     ([forgetful_join_equivariant_statement]); so the meet that the
     contract is proved of, with that join, commutes with no premise
     ([meet_with_forgetful_join_equivariant_statement]). That is the
     meet that the rewrite of the [Switch] is stated for.
   - The functions that add to a typing environment commute, as the
     meet of types does
     ([typing_env_functions_equivariant_statement]).

   Two joins, two meets, one renaming (the second part). A join that
   makes variables takes them from a source of new names. For a
   statement that holds of every input, the correspondence that is
   meant renames the source as well, and the join with the renamed
   source is another function. So the second part relates two
   functions by one renaming, the second being the one that is given
   the renamed arguments: [meet_expanded_head_related],
   [join_scoped_envs_related], [meet_function_related].
   - The meets of two joins that a renaming relates are related by
     it, at every fuel
     ([meet_expanded_head_at_fuel_related_statement],
     [meet_of_types_related_statement]), and so are the functions
     that add to a typing environment
     ([typing_env_functions_related_statement]).
   - At one function the relations are the properties of the first
     part ([meet_expanded_head_equivariant_iff_related_statement],
     [meet_function_equivariant_iff_related_statement]). A join that
     every renaming relates to itself commutes with renaming
     ([join_related_gives_equivariant_statement]). The converse is
     not stated and is not expected: to commute asks less of a join,
     which is then asked only about a meet that commutes with every
     renaming.
   Nothing here says of a particular join that makes variables that
   a renaming relates it to the join with the renamed source; that
   is for the files of that join.

   What the statements do not say. Nothing of a meet that stops: where
   the one answers [None] the other does, and that is all. Nothing of
   the compiler's own join, which is not transcribed as a function
   that commutes on the nose (FlambdaJoinSpec.v). Nothing of how an
   answer depends on the source of new names. And the fuel is the
   same on both sides: a renaming changes no depth. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaProversRenameSpec FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetExamplesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The action of a renaming                                                 *)
(* ======================================================================== *)

(* An environment of the meet: its typing environment, the names that
   an equation is being added to, and the names to avoid. *)
Definition rename_meet_env (r : renaming) (env : Meet_env.t) : Meet_env.t :=
  {| Meet_env.typing_env := rename_typing_env r (Meet_env.typing_env env);
     Meet_env.adding_equations_for_names :=
       map (rename_name r) (Meet_env.adding_equations_for_names env);
     Meet_env.names_to_avoid :=
       map (rename_variable r) (Meet_env.names_to_avoid env) |}.

(* What a meet returns: a new result is renamed. *)
Definition rename_return_value {A : Type} (f : A -> A)
    (res : meet_return_value A) : meet_return_value A :=
  match res with
  | New_result a => New_result (f a)
  | Left_input => Left_input
  | Right_input => Right_input
  | Both_inputs => Both_inputs
  end.

(* The result of a meet: what it returns, and the environment. *)
Definition rename_meet_result {A : Type} (r : renaming) (f : A -> A)
    (result : Meet_result.t A) : Meet_result.t A :=
  match result with
  | Meet_result.Bottom res => Meet_result.Bottom res
  | Meet_result.Ok res env =>
    Meet_result.Ok (rename_return_value f res) (rename_meet_env r env)
  end.

(* A meet of two parts of a type commutes with renaming, the parts
   being renamed by [rename_a] and what it answers by [rename_b]. *)
Definition meet_of_parts_equivariant {A B : Type}
    (rename_a : renaming -> A -> A) (rename_b : renaming -> B -> B)
    (m : Meet_env.t -> A -> A -> option (Meet_result.t B)) : Prop :=
  ∀ (r : renaming) (env : Meet_env.t) (a1 a2 : A),
    m (rename_meet_env r env) (rename_a r a1) (rename_a r a2)
    = option_map (rename_meet_result r (rename_b r)) (m env a1 a2).

(* A meet of expanded types commutes with renaming. This is what is
   asked of the function that the functions of [Meet_env] are given,
   and what is proved of the meet of expanded types at every fuel. It
   is [meet_of_parts_equivariant] at expanded types. *)
Definition meet_expanded_head_equivariant (m : meet_expanded_head) : Prop :=
  ∀ (r : renaming) (env : Meet_env.t) (e1 e2 : Expanded_type.t),
    m (rename_meet_env r env) (rename_expanded r e1) (rename_expanded r e2)
    = option_map (rename_meet_result r (rename_expanded r)) (m env e1 e2).

(* The join that the meet is given commutes with renaming: given a
   meet of expanded types that does, of the renamed environments it
   answers the renamed answer. The scope is not renamed. *)
Definition join_scoped_envs_equivariant (join : join_scoped_envs) : Prop :=
  ∀ (r : renaming) (m : meet_expanded_head) (cut_after : Scope.t)
    (initial_env : Meet_env.t) (initial_tenv : Typing_env.t)
    (scoped_envs : list Typing_env.t),
    meet_expanded_head_equivariant m ->
    join m cut_after (rename_meet_env r initial_env)
      (rename_typing_env r initial_tenv)
      (map (rename_typing_env r) scoped_envs)
    = option_map (rename_meet_env r)
        (join m cut_after initial_env initial_tenv scoped_envs).

(* A meet of types commutes with renaming: of a renamed environment
   and renamed types it answers the renamed answer. *)
Definition meet_function_equivariant (meet : meet_function) : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (t1 t2 : flambda_type),
    meet (rename_typing_env r tenv) (rename_type r t1) (rename_type r t2)
    = option_map
        (fun result =>
           match result with
           | Or_bottom.Bottom => Or_bottom.Bottom
           | Or_bottom.Ok (t, tenv') =>
             Or_bottom.Ok (rename_type r t, rename_typing_env r tenv')
           end)
        (meet tenv t1 t2).

(* A renaming of variables fixes the type that the meet, with these
   choices, gives a name of which some tags are [Get_tag]. Nothing is
   said of the symbols, the code ids or the slots of that type: a
   renaming does not touch them. *)
Definition meet_choices_closed (choices : meet_choices) : Prop :=
  ∀ (r : renaming) (tags : list Z) (alloc_mode : Alloc_mode.t),
    rename_type r (type_of_get_tag_scrutinee choices tags alloc_mode)
    = type_of_get_tag_scrutinee choices tags alloc_mode.

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The choices of the compiler, and those that the contract is proved
   of, are such. *)
Definition compiler_choices_closed_statement : Prop :=
  meet_choices_closed compiler_choices.

Definition repaired_choices_closed_statement : Prop :=
  meet_choices_closed repaired_choices.

(* The meet of expanded types commutes with renaming, whatever the
   fuel, for a join that does and choices that are such. *)
Definition meet_expanded_head_at_fuel_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    join_scoped_envs_equivariant join ->
    meet_choices_closed choices ->
    meet_expanded_head_equivariant (Meet.meet_expanded_head join choices fuel).

(* So does the meet of types. *)
Definition meet_of_types_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    join_scoped_envs_equivariant join ->
    meet_choices_closed choices ->
    meet_function_equivariant (Meet.meet join choices fuel).

(* The meet that the contract is proved of, and the meet as the
   compiler has it. *)
Definition meet_to_prove_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_equivariant join ->
    meet_function_equivariant (meet_to_prove join fuel).

Definition compiler_meet_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_equivariant join ->
    meet_function_equivariant (compiler_meet join fuel).

(* The join that forgets commutes with renaming; so the meet that the
   contract is proved of, with that join, does, with no premise. *)
Definition forgetful_join_equivariant_statement : Prop :=
  join_scoped_envs_equivariant forgetful_join.

Definition meet_with_forgetful_join_equivariant_statement : Prop :=
  ∀ fuel : nat, meet_function_equivariant (meet_to_prove forgetful_join fuel).

(* The functions that add to a typing environment
   ([Flambda2_types], FlambdaMeet.v) commute with renaming as the meet
   does. A parameter is renamed and keeps its kind. *)
Definition typing_env_functions_equivariant_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    join_scoped_envs_equivariant join ->
    meet_choices_closed choices ->
    (∀ (r : renaming) (tenv : Typing_env.t) (name : name)
       (ty : flambda_type),
      Flambda2_types.add_equation join choices fuel (rename_typing_env r tenv)
        (rename_name r name) (rename_type r ty)
      = option_map (rename_typing_env r)
          (Flambda2_types.add_equation join choices fuel tenv name ty))
    ∧ (∀ (r : renaming) (tenv : Typing_env.t) (extension : env_extension),
        Flambda2_types.add_env_extension join choices fuel
          (rename_typing_env r tenv) (rename_env_extension r extension)
        = option_map (rename_typing_env r)
            (Flambda2_types.add_env_extension join choices fuel tenv
               extension))
    ∧ ∀ (r : renaming) (tenv : Typing_env.t)
        (params : list (variable * flambda_kind))
        (param_types : list flambda_type),
        Flambda2_types.add_equations_on_params join choices fuel
          (rename_typing_env r tenv)
          (map (fun param => (rename_variable r (fst param), snd param))
             params)
          (map (rename_type r) param_types)
        = option_map (rename_typing_env r)
            (Flambda2_types.add_equations_on_params join choices fuel tenv
               params param_types).

(* ======================================================================== *)
(* Two joins, two meets, one renaming                                       *)
(* ======================================================================== *)

(* What is above is for a join that commutes with every renaming. A
   join that makes variables takes them from a source of new names;
   the correspondence that is meant renames the source as well, so on
   the renamed side the join is another function. What follows is
   for one renaming and two functions, the second of which is given
   the renamed arguments. *)

(* Two meets of two parts of a type, the parts being renamed by
   [rename_a] and what the meets answer by [rename_b]. *)
Definition meet_of_parts_related {A B : Type} (r : renaming)
    (rename_a : A -> A) (rename_b : B -> B)
    (meet meet' : Meet_env.t -> A -> A -> option (Meet_result.t B)) : Prop :=
  ∀ (env : Meet_env.t) (a1 a2 : A),
    meet' (rename_meet_env r env) (rename_a a1) (rename_a a2)
    = option_map (rename_meet_result r rename_b) (meet env a1 a2).

(* Two meets of expanded types. *)
Definition meet_expanded_head_related (r : renaming)
    (m m' : meet_expanded_head) : Prop :=
  ∀ (env : Meet_env.t) (e1 e2 : Expanded_type.t),
    m' (rename_meet_env r env) (rename_expanded r e1) (rename_expanded r e2)
    = option_map (rename_meet_result r (rename_expanded r)) (m env e1 e2).

(* Two joins, each given its meet of expanded types. *)
Definition join_related_at (r : renaming) (join join' : join_scoped_envs)
    (m m' : meet_expanded_head) : Prop :=
  ∀ (cut_after : Scope.t) (initial_env : Meet_env.t)
    (initial_tenv : Typing_env.t) (scoped_envs : list Typing_env.t),
    join' m' cut_after (rename_meet_env r initial_env)
      (rename_typing_env r initial_tenv)
      (map (rename_typing_env r) scoped_envs)
    = option_map (rename_meet_env r)
        (join m cut_after initial_env initial_tenv scoped_envs).

(* Two joins: given two meets of expanded types that the renaming
   relates, the second answers the renamed answer of the first. *)
Definition join_scoped_envs_related (r : renaming)
    (join join' : join_scoped_envs) : Prop :=
  ∀ m m' : meet_expanded_head,
    meet_expanded_head_related r m m' -> join_related_at r join join' m m'.

(* Two meets of types. *)
Definition meet_function_related (r : renaming) (meet meet' : meet_function)
    : Prop :=
  ∀ (tenv : Typing_env.t) (t1 t2 : flambda_type),
    meet' (rename_typing_env r tenv) (rename_type r t1) (rename_type r t2)
    = option_map
        (fun result =>
           match result with
           | Or_bottom.Bottom => Or_bottom.Bottom
           | Or_bottom.Ok (t, tenv') =>
             Or_bottom.Ok (rename_type r t, rename_typing_env r tenv')
           end)
        (meet tenv t1 t2).

(* The relations, at one function, are the properties above; and a
   join that every renaming relates to itself commutes with
   renaming. *)
Definition meet_expanded_head_equivariant_iff_related_statement : Prop :=
  ∀ m : meet_expanded_head,
    meet_expanded_head_equivariant m
    <-> ∀ r : renaming, meet_expanded_head_related r m m.

Definition meet_function_equivariant_iff_related_statement : Prop :=
  ∀ meet : meet_function,
    meet_function_equivariant meet
    <-> ∀ r : renaming, meet_function_related r meet meet.

Definition join_related_gives_equivariant_statement : Prop :=
  ∀ join : join_scoped_envs,
    (∀ r : renaming, join_scoped_envs_related r join join) ->
    join_scoped_envs_equivariant join.

(* The meets of expanded types of two joins that a renaming relates
   are related by it, whatever the fuel, for choices that are fixed
   by renaming. *)
Definition meet_expanded_head_at_fuel_related_statement : Prop :=
  ∀ (r : renaming) (join join' : join_scoped_envs) (choices : meet_choices)
    (fuel : nat),
    join_scoped_envs_related r join join' ->
    meet_choices_closed choices ->
    meet_expanded_head_related r (Meet.meet_expanded_head join choices fuel)
      (Meet.meet_expanded_head join' choices fuel).

(* So are the meets of types. *)
Definition meet_of_types_related_statement : Prop :=
  ∀ (r : renaming) (join join' : join_scoped_envs) (choices : meet_choices)
    (fuel : nat),
    join_scoped_envs_related r join join' ->
    meet_choices_closed choices ->
    meet_function_related r (Meet.meet join choices fuel)
      (Meet.meet join' choices fuel).

(* And the functions that add to a typing environment. *)
Definition typing_env_functions_related_statement : Prop :=
  ∀ (r : renaming) (join join' : join_scoped_envs) (choices : meet_choices)
    (fuel : nat),
    join_scoped_envs_related r join join' ->
    meet_choices_closed choices ->
    (∀ (tenv : Typing_env.t) (name : name) (ty : flambda_type),
      Flambda2_types.add_equation join' choices fuel
        (rename_typing_env r tenv) (rename_name r name) (rename_type r ty)
      = option_map (rename_typing_env r)
          (Flambda2_types.add_equation join choices fuel tenv name ty))
    ∧ (∀ (tenv : Typing_env.t) (extension : env_extension),
        Flambda2_types.add_env_extension join' choices fuel
          (rename_typing_env r tenv) (rename_env_extension r extension)
        = option_map (rename_typing_env r)
            (Flambda2_types.add_env_extension join choices fuel tenv
               extension))
    ∧ ∀ (tenv : Typing_env.t) (params : list (variable * flambda_kind))
        (param_types : list flambda_type),
        Flambda2_types.add_equations_on_params join' choices fuel
          (rename_typing_env r tenv)
          (map (fun param => (rename_variable r (fst param), snd param))
             params)
          (map (rename_type r) param_types)
        = option_map (rename_typing_env r)
            (Flambda2_types.add_equations_on_params join choices fuel tenv
               params param_types).
