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


(* The meet and its fuel.

   This file is to be reviewed: the definitions and the statements,
   which are about the functions of FlambdaMeet.v and FlambdaMeetEnv.v.
   The contract of the meet is FlambdaMeetSpec.v. The proofs are in
   FlambdaMeetTerminationProofs.v, which needs no review.

   The meet of expanded types is defined by recursion on a number, the
   fuel ([Meet.meet_expanded_head], FlambdaMeet.v): with no fuel it
   answers nothing, and with one more it is [meet_expanded_head_step]
   of the meet with one less. The compiler has no fuel: its meet is a
   recursive function, and nothing in it says why it ends. So a [None]
   of the model is a fatal error of the compiler, or the one state at
   which the model stops and the compiler does not (FlambdaMeetEnv.v,
   "A name that is no longer canonical"), or fuel that has run out; and
   the contract of the meet, which is about the answers that are not
   [None], says nothing of an input on which no fuel is enough.

   What is stated here is what can be said of the fuel without a
   bound on it. First, an answer, once given, is the answer at every
   greater fuel.
   - [meet_head_le mh mh']: wherever the meet of expanded types [mh]
     answers, [mh'] gives the same answer.
   - One step is monotone ([meet_expanded_head_step_monotone_statement]):
     the step of a meet that answers more answers more. The join that
     the meet is given takes the meet of less fuel too, so it is asked
     to be monotone in it ([join_monotone]); the join that forgets
     (FlambdaMeetExamplesSpec.v) is
     ([forgetful_join_monotone_statement]). That the compiler's join is
     monotone in its meet is not stated here.
   - So the meet of expanded types answers at a greater fuel what it
     answers at a lesser ([meet_expanded_head_fuel_monotone_statement]),
     and so do the meet of types ([meet_fuel_monotone_statement]),
     [Meet.meet_shape] ([meet_shape_fuel_monotone_statement]) and the
     functions that add to a typing environment
     ([typing_env_functions_fuel_monotone_statement]).
   - So an input has one answer at most, whatever the fuel
     ([meet_answer_unique_statement]).
   Nothing is supposed of the inputs, of the choices of the meet
   ([meet_choices]) or of the contract of the join.

   Second, what it is for the fuel to be enough, [None] included. A
   [None] of the model does not say whether the fuel ran out, so that
   cannot be said of the answers alone. The meet of no fuel answers
   [None] on every input; [meet_expanded_head_over] is the meet with
   any meet [mh0] in its place. The fuel is enough on an input
   ([fuel_enough], [meet_fuel_enough]) if the answer is the same
   whatever [mh0] is. That is a property of the answers and not of how
   the meet runs: it does not say that the meet never gets to the
   bottom of its fuel on that input (the join, which is any function
   here, may ask the meet that it is given and not use the answer),
   only that what is answered there makes no difference.
   - Where the fuel is enough, more fuel is enough and the answer is
     the same, [None] included ([fuel_enough_stable_statement],
     [meet_fuel_enough_stable_statement]). So a [None] at a fuel that
     is enough is a [None] at every greater fuel: more of this fuel
     does not remedy it. What it is otherwise is not said here: a
     fatal error of the compiler, the one state at which the model
     stops, or a [None] of the join, which where it is the transcribed
     join has fuel of its own. This supposes nothing of the join.
   - A fuel at which the meet answers is enough
     ([answer_fuel_enough_statement], [meet_answer_fuel_enough_statement]),
     for a join that is monotone.

   Third, the inputs for which a fuel is known, which are few. An
   expanded type is flat ([flat]) if it says nothing, or has no value,
   or is a set of numbers, with no relation if they are naked
   immediates. A flat type has no type and no name inside. Not every
   type of which that is so is flat: no head of values is, not even
   that of null or of a string.
   - The meet of two flat expanded types asks for no other meet, so
     one unit of fuel is enough ([flat_fuel_enough_statement]); and so
     is it where one of the two says nothing or has no value, whatever
     the other is ([unknown_or_bottom_fuel_enough_statement]).
   - For two types that are not aliases, the meet of types reads
     nothing of the environment before it asks the meet of their
     heads, in the environment as it was given: a fuel that is enough
     for that meet is enough for the meet of the types
     ([non_alias_fuel_enough_statement]). (The meet of the heads is
     given the environment and may read it: a head that is not an
     alias may have aliases inside.) So one unit is enough for two
     flat types that are not aliases
     ([flat_types_fuel_enough_statement]).
   - A type that is not an alias, met with the type of a constant,
     which is an alias of the constant: the one meet that is asked for
     is that of the head of the type with the head of the constant
     ([const_fuel_enough_statement]).
   - An alias of a name, met with the type of a constant, where the
     name is canonical: the name is demoted, and the one meet that is
     asked for is that of the type that the environment had for the
     name with the head of the constant
     ([alias_const_fuel_enough_statement]). So one unit is enough
     where that type is flat and the constant is a naked number
     ([alias_flat_const_fuel_enough_statement]). That is the meet
     that Simplify makes for an arm of a [Switch] on a variable whose
     type is a set of numbers, as the result of a comparison is; an
     example is run ([example_alias_const_fuel_statement]).
   These hold of every join and of every choice of the meet. They are
   the bottom of a bound by the depth of the types and no more than
   that (the bound by the depth, for the types with no name inside,
   is in FlambdaMeetDepthSpec.v). They say nothing of a type of
   values with a block or a closure in it, of two aliases of names, of
   an alias met with a type that is not that of a constant, or of a
   naked immediate that is the result of a relation: the types that
   Simplify meets at a [Switch] on the tag of a variant are of those.
   (For the last there is FlambdaMeetRelationFuelSpec.v: two units
   are enough, whatever the relations. That is not of every join
   and every choice: it is under what that file supposes of the join
   ([join_of_nothing]), of the choices of the meet ([get_tag_shapes]),
   and of the environment and its aliases.)

   What is not here: a bound for the other inputs. That is, a number
   computed from the environment and the two types at which the fuel
   is enough, [meet_fuel_enough join choices (bound tenv t1 t2) tenv
   t1 t2], for the inputs that a meet may be given. With it and
   [meet_fuel_enough_stable_statement] the answer of the meet does not
   change from the bound on. No argument for it is written down in the
   compiler, and nothing here proves any part of what follows.

   What a proof would rest on. A meet of expanded types of less fuel
   is asked for in three ways: for two parts of the types in hand,
   which are smaller; for a type and the type that the environment has
   for a canonical name, where an equation is added to the name; and,
   where two canonical elements are made equal, for the type that the
   name demoted had and the type of the other, or the constant that
   the other is. In the second the name is among those that an
   equation is being added to until that meet answers
   ([Meet_env.adding_equation_for_name]), and an equation on such a
   name is passed over; in the third a name ceases to be canonical. So
   along the meets that are inside one another those of the second and
   third ways are no more than twice the names of the environment, and
   between two of them the types get smaller.

   What is in the way of a bound. The types of the environment change
   while the meet runs: a meet gives a name the type that it answers,
   and that type holds the extensions made from the equations added in
   the scopes of a disjunction, so it can be deeper than the two types
   met. And a join may define variables, which are new names; the join
   that forgets defines none.

   What comes next, in the order of what it costs. For types with no
   name inside, the depth of the two types is fuel enough, with a join
   that answers the typing environment of the meet where nothing was
   added to the environments joined: that is stated and proved in
   FlambdaMeetDepthSpec.v. For types with names, the same with the
   invariant on the environment that the paragraph above asks for.
   For a join that defines variables, nothing is known. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec.
From OxCamlFlambda2Types Require FlambdaMeetExamplesSpec.
Import ListNotations.

(* ======================================================================== *)
(* A meet that answers more                                                 *)
(* ======================================================================== *)

(* Wherever [mh] answers, [mh'] gives the same answer. *)
Definition meet_head_le (mh mh' : meet_expanded_head) : Prop :=
  ∀ (menv : Meet_env.t) (e1 e2 : Expanded_type.t)
    (result : Meet_result.t Expanded_type.t),
    mh menv e1 e2 = Some result -> mh' menv e1 e2 = Some result.

(* A join answers the same with a meet that answers more. *)
Definition join_monotone (join : join_scoped_envs) : Prop :=
  ∀ (mh mh' : meet_expanded_head) (cut_after : Scope.t)
    (initial_env result : Meet_env.t) (initial_tenv : Typing_env.t)
    (scoped_envs : list Typing_env.t),
    meet_head_le mh mh' ->
    join mh cut_after initial_env initial_tenv scoped_envs = Some result ->
    join mh' cut_after initial_env initial_tenv scoped_envs = Some result.

Definition forgetful_join_monotone_statement : Prop :=
  join_monotone FlambdaMeetExamplesSpec.forgetful_join.

(* ======================================================================== *)
(* More fuel changes no answer                                              *)
(* ======================================================================== *)

(* One step. *)
Definition meet_expanded_head_step_monotone_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices)
    (mh mh' : meet_expanded_head),
    join_monotone join ->
    meet_head_le mh mh' ->
    meet_head_le (meet_expanded_head_step join choices mh)
      (meet_expanded_head_step join choices mh').

(* The meet of expanded types. *)
Definition meet_expanded_head_fuel_monotone_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat),
    join_monotone join ->
    fuel <= fuel' ->
    meet_head_le (Meet.meet_expanded_head join choices fuel)
      (Meet.meet_expanded_head join choices fuel').

(* The meet of types. *)
Definition meet_fuel_monotone_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (result : Or_bottom.t (flambda_type * Typing_env.t)),
    join_monotone join ->
    fuel <= fuel' ->
    Meet.meet join choices fuel tenv t1 t2 = Some result ->
    Meet.meet join choices fuel' tenv t1 t2 = Some result.

Definition meet_shape_fuel_monotone_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (tenv : Typing_env.t) (t shape : flambda_type)
    (result : Or_bottom.t Typing_env.t),
    join_monotone join ->
    fuel <= fuel' ->
    Meet.meet_shape join choices fuel tenv t shape = Some result ->
    Meet.meet_shape join choices fuel' tenv t shape = Some result.

(* An input has one answer at most, whatever the fuel. *)
Definition meet_answer_unique_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (result result' : Or_bottom.t (flambda_type * Typing_env.t)),
    join_monotone join ->
    Meet.meet join choices fuel tenv t1 t2 = Some result ->
    Meet.meet join choices fuel' tenv t1 t2 = Some result' ->
    result = result'.

(* The functions that add to a typing environment
   (types/flambda2_types.ml). *)
Definition typing_env_functions_fuel_monotone_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (tenv tenv' : Typing_env.t),
    join_monotone join ->
    fuel <= fuel' ->
    (∀ (n : name) (ty : flambda_type),
      Flambda2_types.add_equation join choices fuel tenv n ty = Some tenv' ->
      Flambda2_types.add_equation join choices fuel' tenv n ty = Some tenv')
    ∧ (∀ extension : env_extension,
        Flambda2_types.add_env_extension join choices fuel tenv extension
        = Some tenv' ->
        Flambda2_types.add_env_extension join choices fuel' tenv extension
        = Some tenv')
    ∧ (∀ (params : list (variable * flambda_kind))
         (param_types : list flambda_type),
        Flambda2_types.add_equations_on_params join choices fuel tenv params
          param_types
        = Some tenv' ->
        Flambda2_types.add_equations_on_params join choices fuel' tenv params
          param_types
        = Some tenv')
    ∧ (∀ (n : name) (scrutinee : simple) (name_ty : flambda_type)
         (scrutinee_head : variable -> head_of_kind_value),
        Flambda2_types.add_relation join choices fuel tenv n scrutinee name_ty
          scrutinee_head
        = Some tenv' ->
        Flambda2_types.add_relation join choices fuel' tenv n scrutinee
          name_ty scrutinee_head
        = Some tenv').

(* ======================================================================== *)
(* A fuel that is enough                                                    *)
(* ======================================================================== *)

(* The meet of expanded types, with the meet [mh0] where the fuel runs
   out. *)
Fixpoint meet_expanded_head_over (join : join_scoped_envs)
    (choices : meet_choices) (mh0 : meet_expanded_head) (fuel : nat)
    : meet_expanded_head :=
  match fuel with
  | O => mh0
  | S fuel =>
    meet_expanded_head_step join choices
      (meet_expanded_head_over join choices mh0 fuel)
  end.

(* [Meet.meet_expanded_head] is that, over the meet that answers
   nothing. *)
Definition meet_expanded_head_over_none_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    meet_expanded_head_over join choices (fun _ _ _ => None) fuel
    = Meet.meet_expanded_head join choices fuel.

(* The fuel is enough for the meet of two expanded types: the answer
   is the same whatever is answered where the fuel runs out. *)
Definition fuel_enough (join : join_scoped_envs) (choices : meet_choices)
    (fuel : nat) (menv : Meet_env.t) (e1 e2 : Expanded_type.t) : Prop :=
  ∀ mh0 : meet_expanded_head,
    meet_expanded_head_over join choices mh0 fuel menv e1 e2
    = Meet.meet_expanded_head join choices fuel menv e1 e2.

(* The same for the meet of two types ([Meet.meet], which is made of
   [Meet_env.meet_type] in the environment of the meet). *)
Definition meet_fuel_enough (join : join_scoped_envs)
    (choices : meet_choices) (fuel : nat) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) : Prop :=
  ∀ mh0 : meet_expanded_head,
    Meet_env.meet_type (Meet_env.create tenv) t1 t2
      (meet_expanded_head_over join choices mh0 fuel)
    = Meet_env.meet_type (Meet_env.create tenv) t1 t2
        (Meet.meet_expanded_head join choices fuel).

(* Where the fuel is enough, more fuel is enough, and the answer is the
   same, [None] included. *)
Definition fuel_enough_stable_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (menv : Meet_env.t) (e1 e2 : Expanded_type.t),
    fuel_enough join choices fuel menv e1 e2 ->
    fuel <= fuel' ->
    fuel_enough join choices fuel' menv e1 e2
    ∧ Meet.meet_expanded_head join choices fuel' menv e1 e2
      = Meet.meet_expanded_head join choices fuel menv e1 e2.

Definition meet_fuel_enough_stable_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel fuel' : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type),
    meet_fuel_enough join choices fuel tenv t1 t2 ->
    fuel <= fuel' ->
    meet_fuel_enough join choices fuel' tenv t1 t2
    ∧ Meet.meet join choices fuel' tenv t1 t2
      = Meet.meet join choices fuel tenv t1 t2.

(* A fuel at which the meet answers is enough. *)
Definition answer_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (menv : Meet_env.t) (e1 e2 : Expanded_type.t)
    (result : Meet_result.t Expanded_type.t),
    join_monotone join ->
    Meet.meet_expanded_head join choices fuel menv e1 e2 = Some result ->
    fuel_enough join choices fuel menv e1 e2.

(* (In an environment that is bottom [Meet.meet] answers bottom at
   every fuel and makes no meet of types.) *)
Definition meet_answer_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type)
    (result : Or_bottom.t (flambda_type * Typing_env.t)),
    join_monotone join ->
    Typing_env.is_bottom tenv = false ->
    Meet.meet join choices fuel tenv t1 t2 = Some result ->
    meet_fuel_enough join choices fuel tenv t1 t2.

(* ======================================================================== *)
(* Inputs for which a fuel is known                                         *)
(* ======================================================================== *)

(* The expanded types of which the statements below are made: one that
   says nothing or has no value, or a set of numbers, with no relation
   if they are naked immediates. (A head of [Rec_info] or of a region
   has nothing in it in the model.) They have no type and no name
   inside. They are not all the types of which that is so: every head
   of values is left out, that of null and that of a string among
   them. *)
Definition flat_head (head : Expanded_type.head) : bool :=
  match head with
  | Expanded_type.Head_value _ => false
  | Expanded_type.Head_naked_immediate head =>
    Keyed.is_empty (inverse_relations head)
  | Expanded_type.Head_naked_number _ _ | Expanded_type.Head_rec_info _
  | Expanded_type.Head_region _ => true
  end.

Definition flat (expanded : Expanded_type.t) : bool :=
  match Expanded_type.descr expanded with
  | Unknown | Bottom => true
  | Ok head => flat_head head
  end.

(* The meet of two flat expanded types asks for no other meet: one
   unit of fuel is enough, whatever the environment, the join and the
   choices. *)
Definition flat_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t),
    flat e1 = true ->
    flat e2 = true ->
    fuel_enough join choices 1 menv e1 e2.

(* So is it where one of the two says nothing or has no value,
   whatever the other is. *)
Definition unknown_or_bottom_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (menv : Meet_env.t)
    (e1 e2 : Expanded_type.t),
    Expanded_type.is_unknown e1 || Expanded_type.is_bottom e1
    || Expanded_type.is_unknown e2 || Expanded_type.is_bottom e2
    = true ->
    fuel_enough join choices 1 menv e1 e2.

(* For two types that are not aliases, a fuel that is enough for the
   meet of their heads, in the environment as the meet of types was
   given it, is enough for the meet of the types: the meet of types
   reads nothing of the environment before it asks that meet. *)
Definition non_alias_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv : Typing_env.t) (t1 t2 : flambda_type) (e1 e2 : Expanded_type.t),
    Expanded_type.of_non_alias_type t1 = Some e1 ->
    Expanded_type.of_non_alias_type t2 = Some e2 ->
    fuel_enough join choices fuel (Meet_env.create tenv) e1 e2 ->
    meet_fuel_enough join choices fuel tenv t1 t2.

(* So one unit of fuel is enough for the meet of two types that are
   not aliases and are flat. *)
Definition flat_types_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (t1 t2 : flambda_type) (e1 e2 : Expanded_type.t),
    Expanded_type.of_non_alias_type t1 = Some e1 ->
    Expanded_type.of_non_alias_type t2 = Some e2 ->
    flat e1 = true ->
    flat e2 = true ->
    meet_fuel_enough join choices 1 tenv t1 t2.

(* A type that is not an alias, met with the type of a constant, on
   either side: the one meet that is asked for is that of the head of
   the type with the head of the constant. *)
Definition const_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv : Typing_env.t) (t : flambda_type) (e : Expanded_type.t)
    (c : const),
    Expanded_type.of_non_alias_type t = Some e ->
    fuel_enough join choices fuel (Meet_env.create tenv) e
      (Expanded_type.create_const c) ->
    meet_fuel_enough join choices fuel tenv t (this_const c)
    ∧ meet_fuel_enough join choices fuel tenv (this_const c) t.

(* A type that is an alias, met with the type of a constant, where the
   canonical element that the meet reads for the alias is a name: the
   name is demoted to the constant, and the one meet that is asked for
   is that of the type that the environment has for the name with the
   head of the constant, in an environment that the demotion has
   changed. (For each arm of a [Switch] whose scrutinee is a variable,
   Simplify makes such a meet.) What is supposed of the aliases of the
   environment is part of the invariant of typing environments
   (FlambdaTypingEnvInvariantSpec.v). *)
Definition alias_const_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat)
    (tenv : Typing_env.t) (t ty : flambda_type) (n : name)
    (e : Expanded_type.t) (c : const),
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) (simple_of_name n) ->
    Typing_env.get_alias_then_canonical_simple_exn tenv
      (Some Name_mode.in_types) None t
    = Some (Some (simple_of_name n)) ->
    Typing_env.find tenv n (Some (kind t)) = Some ty ->
    Expanded_type.of_non_alias_type ty = Some e ->
    (∀ menv : Meet_env.t,
      fuel_enough join choices fuel menv e (Expanded_type.create_const c)) ->
    meet_fuel_enough join choices fuel tenv t (this_const c).

(* So one unit of fuel is enough for it where the type that the
   environment has for the name is flat and the constant is a naked
   number or poison (the head of a tagged immediate and that of null
   are heads of values, which are not flat). *)
Definition alias_flat_const_fuel_enough_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (tenv : Typing_env.t)
    (t ty : flambda_type) (n : name) (e : Expanded_type.t) (c : const),
    aliases_consistent (Typing_env.aliases tenv) ->
    is_canonical (Typing_env.aliases tenv) (simple_of_name n) ->
    Typing_env.get_alias_then_canonical_simple_exn tenv
      (Some Name_mode.in_types) None t
    = Some (Some (simple_of_name n)) ->
    Typing_env.find tenv n (Some (kind t)) = Some ty ->
    Expanded_type.of_non_alias_type ty = Some e ->
    flat e = true ->
    flat (Expanded_type.create_const c) = true ->
    meet_fuel_enough join choices 1 tenv t (this_const c).

(* An example: a variable of which nothing is known but that it is a
   naked immediate, and the type that is an alias of it, met with the
   type of the number 0, as for an arm of a [Switch]. One unit of fuel
   is enough, whatever the join and the choices. And with the meet
   that is to be proved and the join that forgets, the meet answers
   with one unit, and with none it does not. *)
Definition example_fuel_env : Typing_env.t :=
  Typing_env.add_variable_definition Typing_env.create 1%positive
    (Naked_number Kind_naked_immediate) Name_mode.normal.

Definition example_fuel_alias : flambda_type :=
  alias_type_of (Naked_number Kind_naked_immediate) (Var 1%positive).

Definition example_alias_const_fuel_statement : Prop :=
  (∀ (join : join_scoped_envs) (choices : meet_choices),
    meet_fuel_enough join choices 1 example_fuel_env example_fuel_alias
      (this_const (Naked_immediate 0%Z)))
  ∧ (∃ answer : flambda_type * Typing_env.t,
      meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 1 example_fuel_env
        example_fuel_alias (this_const (Naked_immediate 0%Z))
      = Some (Or_bottom.Ok answer))
  ∧ meet_to_prove FlambdaMeetExamplesSpec.forgetful_join 0 example_fuel_env
      example_fuel_alias (this_const (Naked_immediate 0%Z))
    = None.
