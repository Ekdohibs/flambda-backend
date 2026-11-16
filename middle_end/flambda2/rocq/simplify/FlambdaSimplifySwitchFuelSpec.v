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

(* The slices of the [Switch] and the fuel of the meet: under the
   hypotheses below the slices answer the same from two units of fuel
   on, whatever the scrutinee.

   This file is to be reviewed: the definitions and the statements.
   The slices are those of FlambdaSimplifySwitchSpec.v,
   FlambdaSimplifySwitchArmsSpec.v and FlambdaSimplifySwitchMergeSpec.v.
   What it is for the fuel of the meet to be enough is
   FlambdaMeetTerminationSpec.v; the meet for an arm is
   FlambdaMeetRelationFuelSpec.v. The proofs are in
   FlambdaSimplifySwitchFuelProofs.v, which needs no review.

   The slices take a meet as a parameter. The meet that the model has
   for the compiler's is defined by recursion on a number, the fuel,
   which the compiler's has not. So a slice applied to the model's
   meet depends on the fuel, and with too little of it may answer
   nothing: the meet answers nothing where the fuel runs out at a
   meet that it asks of itself. What is stated here is from which
   fuel on the slices no longer depend on it.

   What is stated.
   - [slices_of_agreeing_meets_statement]: the three slices ask one
     meet for each arm, that of the type which the environment
     answers for the scrutinee with the type of the number of the
     arm, and no other. So with two meets that answer the same on
     those ([arm_meets_agree]) each slice answers the same.
   - [switch_fuel_statement]: the meets of every fuel from two units
     on answer the same on those as the meet of two units, whatever
     the scrutinee and whatever the environment has for it, under
     what FlambdaMeetRelationFuelSpec.v supposes of the environment
     (its types of kind value are aliases or have no extension,
     [values_extension_free]; its aliases are consistent), of the
     choices of the meet ([get_tag_shapes]) and of the join
     ([join_of_nothing]). So each slice answers the same from two
     units on ([switch_slices_fuel_statement]).
   - [switch_arm_keeps_statement]: under the same hypotheses, the
     environment that the meet for an arm answers has again the two
     things that are supposed of the environment. So a [Switch] that
     is simplified in the environment of an arm, on the same
     scrutinee or on another, is under the same hypotheses, and so
     are those in its arms.
   - [model_switch_fuel_statement]: the same of the meet of the model
     (FlambdaJoinMeet.model_meet), whose join has what is supposed
     where there is fuel for the join of environments
     (FlambdaMeetDepthSpec.model_join_of_nothing_statement): two
     meets of the model that differ in the fuel of the meet only, two
     units or more in both, answer the same for the arms.
   - [flat_switch_fuel_statement]: one unit is enough for a scrutinee
     whose type is flat (a set of numbers, as the result of a
     comparison is; or a type that says nothing or has no value), and
     of the environment only that its aliases are consistent is
     supposed there.

   Why nothing is asked of the scrutinee. The type that the
   environment answers for it is an alias, and the meet reads for it
   the canonical element of the class of what it is an alias of (or
   nothing, where the environment does not bind that, and then the
   meet answers nothing at every fuel). That element is a constant,
   and then the meet asks no meet of itself; or it is a name, and
   then the type that the environment has for the name is flat, or
   is a head of naked immediates with its inverse relations (the
   result of [Is_int], [Get_tag] or [Is_null]), or the environment
   has no type for the name at the kind of the naked immediates, or
   has an alias for it, and in these two the meet answers nothing at
   every fuel. Where the type of the scrutinee is of another kind
   the meet answers nothing at every fuel too. The cases are those
   of [naked_immediate_fuel_enough_statement],
   FlambdaMeetRelationFuelSpec.v. (In an environment that is bottom
   [Meet.meet] answers bottom before any of this, whatever the
   types.)

   [arm_meets_agree] is about the type that the environment answers
   for the scrutinee, if it answers one: that it answers one is not
   said, and where it answers none the slices answer nothing, whatever
   the meet.

   Four statements that were here supposed of the scrutinee that the
   meet reads a canonical name for it, whose type is a head of naked
   immediates. [switch_fuel_statement], [switch_arm_keeps_statement]
   and [model_switch_fuel_statement] have their conclusions without
   that hypothesis. FlambdaSimplifySwitchFuelProofs.v keeps the four,
   with their texts, as cases of these: [relation_switch_fuel] of the
   first, [relation_switch_arm_keeps] and
   [relation_switch_arm_keeps_aliases] of the two parts of the
   second, [model_relation_switch_fuel] of the third.

   What [flat_switch_fuel_statement] supposes of the scrutinee
   ([scrutinee_has] of [flat_type]): the type that the environment
   answers for it is an alias whose canonical element, as the meet
   reads it, is a canonical name, and the type that the environment
   has for that name is flat. A flat type may be of any kind; where it
   is not one of naked immediates, the two types that are met for an
   arm are of different kinds, and the meet answers the same at every
   fuel.

   Two examples. [example_switch_fuel_statement]: the environment
   after [let tag = Get_tag p] of FlambdaMeetExamplesSpec.v and a
   [Switch] on [tag]; the first slice answers the same from two units
   on, whatever the arms. [example_switch_under_arm_statement]: a
   [Switch] on [tag] in the environment that the meet for the arm for
   1 of such a [Switch] answers, where [tag] is the constant 1; the
   first slice answers the same from two units on, and keeps the arm
   for 1 only.

   What is not here: that the slices applied to the meet of that fuel
   answer what the compiler does (the places where the model of the
   meet is not the compiler's are listed in FlambdaMeet.v and
   FlambdaMeetEnv.v); and the environments in which a type of kind
   value has an extension. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaCseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetTerminationSpec FlambdaMeetDepthSpec FlambdaMeetRelationFuelSpec
  FlambdaSimplifyPrimitivesSpec FlambdaSimplifySwitchSpec
  FlambdaSimplifySwitchArmsSpec FlambdaSimplifySwitchMergeSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec FlambdaJoinEnv
  FlambdaJoinMeet.
Import ListNotations.

(* ======================================================================== *)
(* The meets that the slices ask                                            *)
(* ======================================================================== *)

(* Two meets answer the same on the type that the environment answers
   for the scrutinee, if it answers one, and the type of each
   number. *)
Definition arm_meets_agree (meet meet' : meet_function) (tenv : Typing_env.t)
    (scrutinee : simple) : Prop :=
  ∀ (scrutinee_ty : flambda_type) (canonical : simple) (i : Z),
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    meet tenv scrutinee_ty (this_naked_immediate i)
    = meet' tenv scrutinee_ty (this_naked_immediate i).

(* The three slices ask no other meet: with two meets that agree there
   they answer the same. *)
Definition slices_of_agreeing_meets_statement : Prop :=
  ∀ (meet meet' : meet_function) (rewrite : apply_cont_rewrite)
    (known : computed) (fresh : list variable) (tenv : Typing_env.t)
    (scrutinee : simple) (arms : list (Z * apply_cont)),
    arm_meets_agree meet meet' tenv scrutinee ->
    simplify_switch meet tenv scrutinee arms
    = simplify_switch meet' tenv scrutinee arms
    ∧ simplify_switch_arms meet rewrite tenv scrutinee arms
      = simplify_switch_arms meet' rewrite tenv scrutinee arms
    ∧ simplify_switch_merging meet rewrite known fresh tenv scrutinee arms
      = simplify_switch_merging meet' rewrite known fresh tenv scrutinee arms.

(* ======================================================================== *)
(* Every scrutinee                                                          *)
(* ======================================================================== *)

(* From two units of fuel on, the meets for the arms are the same,
   whatever the scrutinee is and whatever the environment has for
   it. *)
Definition switch_fuel_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (scrutinee : simple) (fuel : nat),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    2 <= fuel ->
    arm_meets_agree (Meet.meet join choices fuel) (Meet.meet join choices 2)
      tenv scrutinee.

(* The environment that the meet for an arm answers has again the two
   things that are supposed of the environment of the [Switch]. *)
Definition switch_arm_keeps_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (tenv tenv' : Typing_env.t) (scrutinee : simple) (fuel : nat)
    (scrutinee_ty ty : flambda_type) (canonical : simple) (i : Z),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    2 <= fuel ->
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    Meet.meet join choices fuel tenv scrutinee_ty (this_naked_immediate i)
    = Some (Or_bottom.Ok (ty, tenv')) ->
    values_extension_free tenv'
    ∧ aliases_consistent (Typing_env.aliases tenv').

(* So each of the three slices answers the same from two units on. *)
Definition switch_slices_fuel_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (rewrite : apply_cont_rewrite) (known : computed) (fresh : list variable)
    (tenv : Typing_env.t) (scrutinee : simple) (arms : list (Z * apply_cont))
    (fuel : nat),
    join_of_nothing join ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    2 <= fuel ->
    simplify_switch (Meet.meet join choices fuel) tenv scrutinee arms
    = simplify_switch (Meet.meet join choices 2) tenv scrutinee arms
    ∧ simplify_switch_arms (Meet.meet join choices fuel) rewrite tenv
        scrutinee arms
      = simplify_switch_arms (Meet.meet join choices 2) rewrite tenv scrutinee
          arms
    ∧ simplify_switch_merging (Meet.meet join choices fuel) rewrite known
        fresh tenv scrutinee arms
      = simplify_switch_merging (Meet.meet join choices 2) rewrite known fresh
          tenv scrutinee arms.

(* The same of the meet of the model, where there is fuel for the join
   of environments: two meets of the model that differ in the fuel of
   the meet only, two units or more in both, answer the same for the
   arms. *)
Definition model_switch_fuel_statement : Prop :=
  ∀ (join_choices : FlambdaJoinEnv.Join_env.choices) (choices : meet_choices)
    (join_depth : nat) (fuel fuel' : FlambdaJoinMeet.fuels)
    (tenv : Typing_env.t) (scrutinee : simple),
    FlambdaJoinMeet.fuel_of_env fuel <> 0%nat ->
    FlambdaJoinMeet.fuel_of_env fuel' = FlambdaJoinMeet.fuel_of_env fuel ->
    FlambdaJoinMeet.fuel_of_types fuel' = FlambdaJoinMeet.fuel_of_types fuel ->
    2 <= FlambdaJoinMeet.fuel_of_meet fuel ->
    2 <= FlambdaJoinMeet.fuel_of_meet fuel' ->
    get_tag_shapes choices ->
    values_extension_free tenv ->
    aliases_consistent (Typing_env.aliases tenv) ->
    arm_meets_agree
      (FlambdaJoinMeet.model_meet join_choices choices join_depth fuel)
      (FlambdaJoinMeet.model_meet join_choices choices join_depth fuel')
      tenv scrutinee.

(* ======================================================================== *)
(* A scrutinee whose type is flat                                           *)
(* ======================================================================== *)

(* What the environment has for the scrutinee: the type that it answers
   for it, if it answers one, is an alias whose canonical element, as
   the meet reads it, is a canonical name, and the type of that name
   has [P]. *)
Definition scrutinee_has (P : flambda_type -> Prop) (tenv : Typing_env.t)
    (scrutinee : simple) : Prop :=
  ∀ (scrutinee_ty : flambda_type) (canonical : simple),
    Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) scrutinee
    = Some (Some (scrutinee_ty, canonical)) ->
    ∃ (n : name) (ty : flambda_type),
      is_canonical (Typing_env.aliases tenv) (simple_of_name n)
      ∧ Typing_env.get_alias_then_canonical_simple_exn tenv
          (Some Name_mode.in_types) None scrutinee_ty
        = Some (Some (simple_of_name n))
      ∧ Typing_env.find tenv n (Some (kind scrutinee_ty)) = Some ty
      ∧ P ty.

(* The type of the scrutinee is flat: a set of numbers, or a type that
   says nothing or has no value. *)
Definition flat_type (ty : flambda_type) : Prop :=
  ∃ e : Expanded_type.t,
    Expanded_type.of_non_alias_type ty = Some e ∧ flat e = true.

(* From one unit of fuel on, the meets for the arms are the same, for
   a scrutinee whose type is flat. *)
Definition flat_switch_fuel_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (scrutinee : simple) (fuel : nat),
    aliases_consistent (Typing_env.aliases tenv) ->
    scrutinee_has flat_type tenv scrutinee ->
    1 <= fuel ->
    arm_meets_agree (Meet.meet join choices fuel) (Meet.meet join choices 1)
      tenv scrutinee.

(* ======================================================================== *)
(* Two examples                                                             *)
(* ======================================================================== *)

(* The environment after [let tag = Get_tag p] of
   FlambdaMeetExamplesSpec.v, and a [Switch] on [tag]: the first slice
   answers the same from two units of fuel on, whatever the arms, with
   every join and all choices that have what is supposed of them. *)
Definition example_switch_fuel_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (arms : list (Z * apply_cont)),
    join_of_nothing join ->
    get_tag_shapes choices ->
    2 <= fuel ->
    simplify_switch (Meet.meet join choices fuel)
      FlambdaMeetExamplesSpec.example_get_tag_env
      (Var FlambdaMeetExamplesSpec.example_tag) arms
    = simplify_switch (Meet.meet join choices 2)
        FlambdaMeetExamplesSpec.example_get_tag_env
        (Var FlambdaMeetExamplesSpec.example_tag) arms.

(* A [Switch] under an arm of another. The environment of
   [example_two_relations_env] (FlambdaMeetRelationFuelSpec.v), in
   which [tag] is the result of [Get_tag] and of [Is_int], and the
   environment that the meet for the arm for 1 of a [Switch] on [tag]
   answers there, with the meet that is to be proved and the join
   that forgets, at two units of fuel. *)
Definition example_arm_env : option Typing_env.t :=
  match
    Meet.meet FlambdaMeetExamplesSpec.forgetful_join repaired_choices 2
      example_two_relations_env FlambdaMeetExamplesSpec.example_tag_type
      (this_naked_immediate 1)
  with
  | Some (Or_bottom.Ok (_, tenv')) => Some tenv'
  | Some Or_bottom.Bottom | None => None
  end.

(* There is one, and the type that it answers for [tag] is that of
   the constant 1: the meet reads a constant for it, and no name. A
   [Switch] on [tag] in that environment: the first slice answers the
   same from two units of fuel on, whatever the arms, with every join
   and all choices that have what is supposed of them; and of an arm
   for 0 and an arm for 1 it keeps the second, so that what it makes
   is the [Apply_cont] of that arm. *)
Definition example_switch_under_arm_statement : Prop :=
  ∃ tenv' : Typing_env.t,
    example_arm_env = Some tenv'
    ∧ Typing_env.type_simple_in_term_exn tenv' (Some Name_mode.normal)
        (Var FlambdaMeetExamplesSpec.example_tag)
      = Some (Some (this_naked_immediate 1, Const (Naked_immediate 1)))
    ∧ (∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
         (arms : list (Z * apply_cont)),
        join_of_nothing join ->
        get_tag_shapes choices ->
        2 <= fuel ->
        simplify_switch (Meet.meet join choices fuel) tenv'
          (Var FlambdaMeetExamplesSpec.example_tag) arms
        = simplify_switch (Meet.meet join choices 2) tenv'
            (Var FlambdaMeetExamplesSpec.example_tag) arms)
    ∧ ∀ ac0 ac1 : apply_cont,
        simplify_switch
          (Meet.meet FlambdaMeetExamplesSpec.forgetful_join repaired_choices 2)
          tenv' (Var FlambdaMeetExamplesSpec.example_tag)
          [(0%Z, ac0); (1%Z, ac1)]
        = Some (Apply_cont ac1).
