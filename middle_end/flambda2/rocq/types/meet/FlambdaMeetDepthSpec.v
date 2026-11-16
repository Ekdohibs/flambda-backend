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

(* The fuel of the meet of types that have no name inside: their depth
   is enough. And of the meet of a shape with a type that has no
   extension: two units are enough, and what that meet makes has no
   extension.

   This file is to be reviewed: the definitions and the statements.
   They go on from FlambdaMeetTerminationSpec.v, whose header says what
   the fuel is and what it is for it to be enough ([fuel_enough],
   [meet_fuel_enough]); the functions are those of FlambdaMeet.v. The
   proofs are in FlambdaMeetDepthProofs.v, which needs no review.

   What is stated. A type has no name inside, within a depth
   ([nameless_within]), if
   - it is not an alias of a name, and no type inside it is one. An
     alias of a constant is taken as the type that the constant
     expands to ([Expanded_type.create_const]): the meet of such an
     alias with a type that is not an alias is the meet of the head of
     that type with the head of the expansion, and the meet of two
     such aliases asks no meet;
   - no variable stands in it for the result of a relation: a variant
     has none for [Is_int] nor for [Get_tag], a head of values none
     for [Is_null], and a head of naked immediates has no inverse
     relation;
   - it has no extension: a variant has [No_extensions], and a case of
     a row-like structure the empty extension;
   - the types inside it (the fields of a block, the immediates of a
     variant, the contents of a boxed number, the closures and the
     value slots of a set of closures) are so within one less. The
     immediates of a variant are not counted if they have no value:
     the meet of two variants then answers that the immediates have
     none without a meet of types ([meet_unknown]).
   So within depth 1 are the types that have no type inside, the flat
   ones of FlambdaMeetTerminationSpec.v among them
   ([flat_nameless_statement]), and no type is so within depth 0. A
   type that is so within a depth is so within any greater
   ([nameless_within_more_statement]). The alias of a constant is so
   within depth 2, and within depth 1 unless the constant is a tagged
   immediate, which expands to a variant whose immediates are the
   alias of a naked immediate ([const_nameless_statement]).

   For two expanded types that have no name inside within a depth,
   that many units of fuel are enough ([nameless_fuel_enough_statement]),
   whatever the environment and the choices of the meet. Enough is as
   in FlambdaMeetTerminationSpec.v: the answer is the same whatever is
   answered where the fuel runs out. That the meet answers is not
   said: the answer that is the same may be none ([None]), as it is
   for two heads of different kinds. And the meet adds nothing: where
   it answers an environment, the typing environment of that one is
   the typing environment of the one it was given
   ([nameless_meet_adds_nothing_statement]). The same for the meet of
   two types ([nameless_types_fuel_enough_statement],
   [nameless_types_meet_adds_nothing_statement]).

   What is supposed, of the join that the meet is given
   ([join_of_nothing]): the join of environments that are each the
   environment of the meet one scope further, with nothing added,
   answers an environment that has the typing environment of the
   environment of the meet, and answers that one environment whatever
   meet the join is given. The join comes into it although the types
   have no name: the meet of two variants, and of two row-like
   structures with more cases than one, meets each alternative in a
   scope of its own and joins the environments that result, which here
   are the ones that were entered. The rest of the environment that
   the join answers (the names to avoid, which are the model's own,
   and the names that an equation is being added to) may be other than
   it was given: so the meet is said to keep the typing environment,
   and no more.

   The join that forgets has the property
   ([forgetful_join_of_nothing_statement]). The join of the model
   (FlambdaJoinMeet.v) has it where its fuel for the join of
   environments is not zero, whatever its test for recovering
   relations, its depth and its two other amounts of fuel
   ([model_join_of_nothing_statement], which is
   FlambdaJoinNothingSpec.model_join_scoped_envs_of_nothing_statement
   said with the predicate of this file). With no fuel for the join
   of environments it has it not: an example is run
   ([example_model_join_statement]) in which, with fuel, the
   environment that it answers has the typing environment that it was
   given and is not the environment that it was given, and with no
   fuel for the environment it answers nothing
   (FlambdaJoinNothingSpec.model_join_scoped_envs_no_fuel_statement
   says that of every environment that is not bottom). So what is
   stated in this file under the hypothesis holds of the meet of the
   model with such a fuel; it is said once, of the types with no name
   inside ([model_meet_nameless_statement]). Nothing is stated of the
   compiler's join.

   The bound is by the depth of the two types and not by that of the
   environment: the functions of the meet read no type of the
   environment, as no name is met. What the join reads is not
   constrained: the hypothesis is on what it answers. A depth that is
   less is not enough in general: two examples are run, one with no
   alias ([example_depth_fuel_statement]) and one with the alias of a
   constant ([example_const_fuel_statement]).

   The meet with a shape. A relation gives the name that it is a
   relation of a type that says little: [any_block],
   [any_tagged_immediate], [null], [any_non_null_value], or the blocks
   of some tags. Such a type is a shape ([shape]): it has no name
   inside, and the types inside it each say nothing or have no value;
   those five are shapes. The type that a shape is met with is one
   that the environment has for a name, and has names inside in
   general. What is asked of it is that it
   has no extension ([extension_free]): it is not an alias; a variant
   in it has [No_extensions], and a case of a row-like structure the
   empty extension; and if it is a head of naked immediates it has no
   inverse relation. It may have variables for relations, and nothing
   is asked of the types inside it, which may be aliases of names.
   For a shape and such a type, in either order, two units of fuel
   are enough ([shape_fuel_enough_statement]), and the meet adds
   nothing to the typing environment
   ([shape_meet_adds_nothing_statement]), under the same hypothesis on
   the join. Here the meet does read the environment: where a type
   inside the second is an alias, it is met with a type that says
   nothing or has no value, and the meet looks up what the alias is
   an alias of and the type that the environment has for it. It adds
   no equation, since that meet answers the type that the name has,
   or bottom. Two units are reached: an example is run
   ([example_shape_fuel_statement]), with a shape that has a type
   inside which says nothing. The shapes that the relations give have
   no type inside, but for immediates that have no value: they are
   bare shapes ([bare_shape], [relation_shapes_statement]; a bare
   shape is a shape, [bare_shape_shape_statement]). For a bare shape
   and a type that has no extension, in either order, one unit is
   enough, and the meet adds nothing to the typing environment
   ([bare_shape_fuel_enough_statement],
   [bare_shape_meet_adds_nothing_statement]): no meet of types is
   asked, so the functions of the meet read no type of the
   environment.

   What the meet with a shape makes. A meet answers one of its two
   inputs as it is, or a type that it makes ([New_result]). Where the
   meet of a shape with a type that has no extension makes a type,
   that type has no extension ([shape_meet_extension_free_statement],
   at two units of fuel; [bare_shape_meet_extension_free_statement],
   at one unit for a bare shape; in either order). That the type made
   differs from the two inputs is not said: the constructor says only
   that the meet built it. The extensions of a variant and of the
   cases of a row-like structure are what was added to the
   environments of the alternatives, and nothing is added there; a
   head of naked immediates that is made of two that have no inverse
   relation has none. Nothing is said of the types inside what is
   made, as nothing is asked of them in a type that has no extension.
   So a name whose type has no extension, once the type that a
   relation gives is added to it, has a type with no extension again:
   the shape, the type that it had, or the type that the meet makes.
   FlambdaMeetRelationFuelSpec.v rests on this for a scrutinee with
   several relations.

   The two classes are one test ([plain_type]) with two parameters:
   whether a head of values may have variables for relations, and
   what is asked of the types inside. A type with no name inside
   allows no variable and asks the same of the types inside, one
   level less ([nameless_type]); a shape allows no variable and asks
   that the types inside say nothing or have no value; a bare shape
   allows no variable and no type inside; a type with no extension
   allows variables and asks nothing of the types inside. In all of
   them the immediates of a variant may have no value.

   What is not here: names, but for the meet with a shape. An alias of
   a name, a variable for a relation on both sides, or an extension
   makes the meet read the environment and add to it, and then the
   types that are met are those that the environment has, which the
   meet itself changes. FlambdaMeetTerminationSpec.v says what a bound
   would rest on there, and what is in its way. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetTerminationSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec FlambdaJoinEnv
  FlambdaJoinMeet.
Import ListNotations.

(* ======================================================================== *)
(* Types with no extension and no alias                                     *)
(* ======================================================================== *)

Section Plain.

(* Whether a head of values may have variables that stand for the
   result of a relation. *)
Variable relations : bool.

(* What is asked of the types inside. *)
Variable inside : flambda_type -> bool.

Definition relation_variable_allowed (var : option variable) : bool :=
  match var with
  | None => true
  | Some _ => relations
  end.

Definition extension_is_empty (extension : env_extension) : bool :=
  match extension with
  | Env_extension equations => Keyed.is_empty equations
  end.

Definition plain_block_case (case : row_like_block_case) : bool :=
  forallb inside (case_maps_to case)
  && extension_is_empty (case_env_extension case).

Definition plain_blocks (blocks : row_like_for_blocks) : bool :=
  match blocks with
  | Row_like_for_blocks known_tags other_tags _ =>
    forallb
      (fun known =>
         match snd known with
         | Or_unknown.Unknown => true
         | Or_unknown.Known case => plain_block_case case
         end)
      known_tags
    && match other_tags with
       | Or_bottom.Bottom => true
       | Or_bottom.Ok case => plain_block_case case
       end
  end.

Definition plain_closures_entry (entry : closures_entry) : bool :=
  match entry with
  | Closures_entry _ closure_types value_slot_types =>
    forallb (fun closure => inside (snd closure)) closure_types
    && forallb (fun slot => inside (snd slot)) value_slot_types
  end.

Definition plain_closures (closures : row_like_for_closures) : bool :=
  match closures with
  | Row_like_for_closures known_closures =>
    forallb
      (fun known =>
         plain_closures_entry (case_maps_to (snd known))
         && extension_is_empty (case_env_extension (snd known)))
      known_closures
  end.

Definition plain_non_null (head : head_of_kind_value_non_null) : bool :=
  match head with
  | Variant is_int immediates get_tag blocks extensions _ =>
    relation_variable_allowed is_int && relation_variable_allowed get_tag
    && match immediates with
       | Or_unknown.Unknown => true
       | Or_unknown.Known ty => inside ty || is_obviously_bottom ty
       end
    && match blocks with
       | Or_unknown.Unknown => true
       | Or_unknown.Known blocks => plain_blocks blocks
       end
    && match extensions with
       | No_extensions => true
       | Ext _ _ => false
       end
  | Mutable_block _ | String _ => true
  | Boxed_number _ contents _ => inside contents
  | Closures by_function_slot _ => plain_closures by_function_slot
  end.

Definition plain_value (head : head_of_kind_value) : bool :=
  match head with
  | Head_of_kind_value non_null is_null =>
    match non_null with
    | Unknown | Bottom => true
    | Ok non_null => plain_non_null non_null
    end
    && match is_null with
       | Not_null => true
       | Maybe_null is_null => relation_variable_allowed is_null
       end
  end.

(* A head of naked immediates has no inverse relation, whether or not
   variables for relations are allowed: an inverse relation makes the
   meet add an equation. *)
Definition plain_naked_immediate (head : head_of_kind_naked_immediate)
    : bool :=
  Keyed.is_empty (inverse_relations head).

Definition plain_descr {head : Type} (plain_head : head -> bool)
    (ty : type_descr head) : bool :=
  match ty with
  | Unknown | Bottom => true
  | Ok (No_alias head) => plain_head head
  | Ok (Equals _) => false
  end.

(* Of a type that is not an alias. *)
Definition plain_type (ty : flambda_type) : bool :=
  match ty with
  | Type_value ty => plain_descr plain_value ty
  | Type_naked_immediate ty => plain_descr plain_naked_immediate ty
  | Type_naked_number _ ty => plain_descr (fun _ => true) ty
  | Type_rec_info ty => plain_descr (fun _ => true) ty
  | Type_region ty => plain_descr (fun _ => true) ty
  end.

End Plain.

(* ======================================================================== *)
(* Types with no name inside                                                *)
(* ======================================================================== *)

(* A type that is not an alias and has no variable for a relation,
   given what is asked of the types inside it. *)
Definition nameless_type (inside : flambda_type -> bool)
    : flambda_type -> bool :=
  plain_type false inside.

(* An alias of a constant is taken as the type that the constant
   expands to, which is not an alias; an alias of a name is not
   allowed. *)
Definition nameless_or_const (inside : flambda_type -> bool)
    (ty : flambda_type) : bool :=
  match get_alias_opt ty with
  | None => nameless_type inside ty
  | Some (Const c) =>
    nameless_type inside
      (Expanded_type.to_type (Expanded_type.create_const c))
  | Some (Var _ | Symbol _) => false
  end.

(* A type has no name inside, and what is inside it is no deeper than
   [depth]. *)
Fixpoint nameless_within (depth : nat) (ty : flambda_type) : bool :=
  match depth with
  | O => false
  | S depth => nameless_or_const (nameless_within depth) ty
  end.

(* The same of an expanded type. *)
Definition nameless_expanded_within (depth : nat) (expanded : Expanded_type.t)
    : bool :=
  nameless_within depth (Expanded_type.to_type expanded).

Definition nameless_within_more_statement : Prop :=
  ∀ (depth depth' : nat) (ty : flambda_type),
    depth <= depth' ->
    nameless_within depth ty = true ->
    nameless_within depth' ty = true.

(* The flat expanded types have no name inside, within depth 1. *)
Definition flat_nameless_statement : Prop :=
  ∀ expanded : Expanded_type.t,
    flat expanded = true -> nameless_expanded_within 1 expanded = true.

(* An alias of a constant has no name inside within depth 2, and
   within depth 1 unless the constant is a tagged immediate. *)
Definition const_nameless_statement : Prop :=
  ∀ (ty : flambda_type) (c : const),
    get_alias_opt ty = Some (Const c) ->
    nameless_within 2 ty = true
    ∧ nameless_within 1 ty
      = match c with
        | Tagged_immediate _ => false
        | _ => true
        end.

(* ======================================================================== *)
(* The join of environments to which nothing was added                      *)
(* ======================================================================== *)

(* The join of environments, one at least, that are each the
   environment of the meet one scope further with nothing added,
   answers an environment that has the typing environment of the
   environment of the meet; and it answers that one environment
   whatever meet it is given. *)
Definition join_of_nothing (join : join_scoped_envs) : Prop :=
  ∀ (env : Meet_env.t) (scoped_envs : list Typing_env.t),
    scoped_envs <> [] ->
    (∀ scoped : Typing_env.t,
      scoped ∈ scoped_envs ->
      scoped = Typing_env.increment_scope (Meet_env.typing_env env)) ->
    ∃ result : Meet_env.t,
      Meet_env.typing_env result = Meet_env.typing_env env
      ∧ ∀ mh : meet_expanded_head,
          join mh (Typing_env.current_scope (Meet_env.typing_env env)) env
            (Meet_env.typing_env env) scoped_envs
          = Some result.

Definition forgetful_join_of_nothing_statement : Prop :=
  join_of_nothing FlambdaMeetExamplesSpec.forgetful_join.

(* The join of the model, of one environment: that of one variable,
   one scope further with nothing added. The meet that the join is
   given is that of expanded types with the choices of the meet that
   is to be proved, over the join that forgets. *)
Definition example_join_env : Typing_env.t :=
  Typing_env.add_variable_definition Typing_env.create 1%positive Value
    Name_mode.normal.

Definition example_model_join_answer (fuel : FlambdaJoinMeet.fuels)
    : option Meet_env.t :=
  FlambdaJoinMeet.model_join_scoped_envs
    FlambdaJoinEnv.Join_env.compiler_choices 5 fuel
    (Meet.meet_expanded_head FlambdaMeetExamplesSpec.forgetful_join
       repaired_choices 3)
    (Typing_env.current_scope example_join_env)
    (Meet_env.create example_join_env) example_join_env
    [Typing_env.increment_scope example_join_env].

Definition example_join_fuel (fuel_of_env : nat) : FlambdaJoinMeet.fuels :=
  {| FlambdaJoinMeet.fuel_of_env := fuel_of_env;
     FlambdaJoinMeet.fuel_of_types := 8;
     FlambdaJoinMeet.fuel_of_meet := 3 |}.

(* With fuel, the environment that the join of the model answers has
   the typing environment that the join was given, and is not the
   environment that it was given; with no fuel for the environment the
   join answers nothing, and so has not the property. *)
Definition example_model_join_statement : Prop :=
  option_map Meet_env.typing_env
    (example_model_join_answer (example_join_fuel 8))
  = Some example_join_env
  ∧ example_model_join_answer (example_join_fuel 8)
    <> Some (Meet_env.create example_join_env)
  ∧ example_model_join_answer (example_join_fuel 0) = None
  ∧ ¬ join_of_nothing
        (FlambdaJoinMeet.model_join_scoped_envs
           FlambdaJoinEnv.Join_env.compiler_choices 5 (example_join_fuel 0)).

(* The join of the model has the property, where its fuel for the join
   of environments is not zero. *)
Definition model_join_of_nothing_statement : Prop :=
  ∀ (join_choices : FlambdaJoinEnv.Join_env.choices) (join_depth : nat)
    (fuel : FlambdaJoinMeet.fuels),
    FlambdaJoinMeet.fuel_of_env fuel <> 0%nat ->
    join_of_nothing
      (FlambdaJoinMeet.model_join_scoped_envs join_choices join_depth fuel).

(* ======================================================================== *)
(* The depth is enough                                                      *)
(* ======================================================================== *)

Definition nameless_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (depth : nat)
    (menv : Meet_env.t) (e1 e2 : Expanded_type.t),
    join_of_nothing join ->
    nameless_expanded_within depth e1 = true ->
    nameless_expanded_within depth e2 = true ->
    fuel_enough join choices depth menv e1 e2.

(* The meet of two such types adds nothing to the typing environment:
   where it answers an environment, that has the typing environment
   that the meet was given. *)
Definition nameless_meet_adds_nothing_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (depth : nat)
    (menv menv' : Meet_env.t) (e1 e2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t),
    join_of_nothing join ->
    nameless_expanded_within depth e1 = true ->
    nameless_expanded_within depth e2 = true ->
    Meet.meet_expanded_head join choices depth menv e1 e2
    = Some (Meet_result.Ok r menv') ->
    Meet_env.typing_env menv' = Meet_env.typing_env menv.

(* The same for the meet of two types. *)
Definition nameless_types_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (depth : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type),
    join_of_nothing join ->
    nameless_within depth t1 = true ->
    nameless_within depth t2 = true ->
    meet_fuel_enough join choices depth tenv t1 t2.

(* And the meet of two such types, as it is exposed, answers the
   typing environment that it was given, where it answers one. *)
Definition nameless_types_meet_adds_nothing_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (depth : nat)
    (tenv tenv' : Typing_env.t) (t1 t2 ty : flambda_type),
    join_of_nothing join ->
    nameless_within depth t1 = true ->
    nameless_within depth t2 = true ->
    Meet.meet join choices depth tenv t1 t2 = Some (Or_bottom.Ok (ty, tenv')) ->
    tenv' = tenv.

(* The meet of the model (FlambdaJoinMeet.model_meet), of two such
   types, with fuel for the join of environments and a fuel of the
   meet that is the depth or more: that fuel of the meet is enough,
   and the meet answers the typing environment that it was given,
   where it answers one. *)
Definition model_meet_nameless_statement : Prop :=
  ∀ (join_choices : FlambdaJoinEnv.Join_env.choices) (choices : meet_choices)
    (join_depth depth : nat) (fuel : FlambdaJoinMeet.fuels)
    (tenv tenv' : Typing_env.t) (t1 t2 ty : flambda_type),
    FlambdaJoinMeet.fuel_of_env fuel <> 0%nat ->
    depth <= FlambdaJoinMeet.fuel_of_meet fuel ->
    nameless_within depth t1 = true ->
    nameless_within depth t2 = true ->
    meet_fuel_enough
      (FlambdaJoinMeet.model_join_scoped_envs join_choices join_depth fuel)
      choices (FlambdaJoinMeet.fuel_of_meet fuel) tenv t1 t2
    ∧ (FlambdaJoinMeet.model_meet join_choices choices join_depth fuel tenv
         t1 t2
       = Some (Or_bottom.Ok (ty, tenv')) ->
       tenv' = tenv).

(* ======================================================================== *)
(* The meet with a shape                                                    *)
(* ======================================================================== *)

(* A type that says nothing or that has no value; it is not an
   alias. *)
Definition says_nothing_or_no_value (ty : flambda_type) : bool :=
  is_obviously_unknown ty || is_obviously_bottom ty.

(* A shape: a type that is not an alias, has no variable for a
   relation and no extension, and whose types inside each say nothing
   or have no value. *)
Definition shape (ty : flambda_type) : bool :=
  plain_type false says_nothing_or_no_value ty.

(* A type that is not an alias and has no extension, nor an inverse
   relation if it is one of naked immediates. It may have variables
   for relations, and nothing is asked of the types inside it. *)
Definition extension_free (ty : flambda_type) : bool :=
  plain_type true (fun _ => true) ty.

(* A bare shape: a shape with no type inside, but for immediates that
   have no value. *)
Definition bare_shape (ty : flambda_type) : bool :=
  plain_type false (fun _ => false) ty.

Definition bare_shape_shape_statement : Prop :=
  ∀ ty : flambda_type, bare_shape ty = true -> shape ty = true.

(* Two units of fuel are enough for the meet of a shape with a type
   that has no extension, in either order. *)
Definition shape_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t),
    join_of_nothing join ->
    shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    fuel_enough join choices 2 menv e1 e2
    ∧ fuel_enough join choices 2 menv e2 e1.

(* And that meet adds nothing to the typing environment. *)
Definition shape_meet_adds_nothing_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (menv menv' : Meet_env.t) (e1 e2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t),
    join_of_nothing join ->
    shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    (Meet.meet_expanded_head join choices 2 menv e1 e2
     = Some (Meet_result.Ok r menv') ->
     Meet_env.typing_env menv' = Meet_env.typing_env menv)
    ∧ (Meet.meet_expanded_head join choices 2 menv e2 e1
       = Some (Meet_result.Ok r menv') ->
       Meet_env.typing_env menv' = Meet_env.typing_env menv).

(* What that meet makes has no extension: where its answer is a type
   that it makes ([New_result]) and not one of the two inputs as it is,
   that type has none. That it is not equal to an input is not said:
   the constructor says only that the meet built it. *)
Definition shape_meet_extension_free_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (menv menv' : Meet_env.t) (e1 e2 e : Expanded_type.t),
    join_of_nothing join ->
    shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    (Meet.meet_expanded_head join choices 2 menv e1 e2
     = Some (Meet_result.Ok (New_result e) menv') ->
     extension_free (Expanded_type.to_type e) = true)
    ∧ (Meet.meet_expanded_head join choices 2 menv e2 e1
       = Some (Meet_result.Ok (New_result e) menv') ->
       extension_free (Expanded_type.to_type e) = true).

(* One unit is enough where the shape is bare. *)
Definition bare_shape_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t),
    join_of_nothing join ->
    bare_shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    fuel_enough join choices 1 menv e1 e2
    ∧ fuel_enough join choices 1 menv e2 e1.

Definition bare_shape_meet_adds_nothing_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (menv menv' : Meet_env.t) (e1 e2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t),
    join_of_nothing join ->
    bare_shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    (Meet.meet_expanded_head join choices 1 menv e1 e2
     = Some (Meet_result.Ok r menv') ->
     Meet_env.typing_env menv' = Meet_env.typing_env menv)
    ∧ (Meet.meet_expanded_head join choices 1 menv e2 e1
       = Some (Meet_result.Ok r menv') ->
       Meet_env.typing_env menv' = Meet_env.typing_env menv).

(* And what it makes has no extension. *)
Definition bare_shape_meet_extension_free_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (menv menv' : Meet_env.t) (e1 e2 e : Expanded_type.t),
    join_of_nothing join ->
    bare_shape (Expanded_type.to_type e1) = true ->
    extension_free (Expanded_type.to_type e2) = true ->
    (Meet.meet_expanded_head join choices 1 menv e1 e2
     = Some (Meet_result.Ok (New_result e) menv') ->
     extension_free (Expanded_type.to_type e) = true)
    ∧ (Meet.meet_expanded_head join choices 1 menv e2 e1
       = Some (Meet_result.Ok (New_result e) menv') ->
       extension_free (Expanded_type.to_type e) = true).

(* The types that a relation gives the name it is a relation of
   ([reduce_inverse_relations], FlambdaMeet.v) are bare shapes; the
   last two are what the compiler's choice and the repaired choice
   give for [Get_tag]. *)
Definition relation_shapes_statement : Prop :=
  bare_shape any_non_null_value = true
  ∧ bare_shape null = true
  ∧ bare_shape any_block = true
  ∧ bare_shape any_tagged_immediate = true
  ∧ ∀ (tags : list Z) (alloc_mode : Alloc_mode.t),
      bare_shape (blocks_with_these_tags tags alloc_mode) = true
      ∧ bare_shape (immediate_or_blocks_with_these_tags tags alloc_mode)
        = true.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* A block of one field of which nothing is known, met with itself. It
   has no name inside within depth 2, and not within depth 1. With the
   meet that is to be proved and the join that forgets, the meet
   answers with two units of fuel, and with one it does not: the meet
   of the fields is then asked of a meet that has no fuel. *)
Definition example_depth_block : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap
    [unknown Value].

Definition example_depth_fuel_statement : Prop :=
  nameless_within 2 example_depth_block = true
  ∧ nameless_within 1 example_depth_block = false
  ∧ (∃ answer : flambda_type * Typing_env.t,
      meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 2 Typing_env.create
        example_depth_block example_depth_block
      = Some (Or_bottom.Ok answer))
  ∧ meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 1 Typing_env.create
      example_depth_block example_depth_block
    = None.

(* A block of one field that is the tagged immediate 3, an alias of
   the constant, met with a block of one field that is the tagged
   immediate 3 or 4, which is not an alias. Both have no name inside
   within depth 3, and the first not within depth 2: the alias is
   taken as the variant that the constant expands to, whose
   immediates are the alias of the naked immediate 3. The meet answers
   with three units of fuel and not with two. *)
Definition example_const_block (field : flambda_type) : flambda_type :=
  immutable_block false 0%Z (Scannable Value_only) Alloc_mode.Heap [field].

Definition example_const_fuel_statement : Prop :=
  let block1 := example_const_block (this_tagged_immediate 3) in
  let block2 := example_const_block (these_tagged_immediates [3%Z; 4%Z]) in
  nameless_within 3 block1 = true
  ∧ nameless_within 3 block2 = true
  ∧ nameless_within 2 block1 = false
  ∧ (∃ answer : flambda_type * Typing_env.t,
      meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 3 Typing_env.create
        block1 block2
      = Some (Or_bottom.Ok answer))
  ∧ meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 2 Typing_env.create
      block1 block2
    = None.

(* A boxed float of which nothing is known, which is a shape, met with
   the boxed float whose contents are the variable [x]. The meet
   answers with two units of fuel and not with one: the type that the
   environment has for [x] is met with the contents of the shape. *)
Definition example_shape_env : Typing_env.t :=
  Typing_env.add_variable_definition Typing_env.create 1%positive
    (Naked_number Kind_naked_float) Name_mode.normal.

Definition example_shape_fuel_statement : Prop :=
  let any_float := any_boxed_number Boxable_naked_float in
  let boxed_x :=
    boxed_number_alias_to Boxable_naked_float 1%positive Alloc_mode.Heap
  in
  shape any_float = true
  ∧ extension_free boxed_x = true
  ∧ (∃ answer : flambda_type * Typing_env.t,
      meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 2 example_shape_env
        any_float boxed_x
      = Some (Or_bottom.Ok answer))
  ∧ meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 1 example_shape_env
      any_float boxed_x
    = None.
