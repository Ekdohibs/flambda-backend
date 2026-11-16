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


(* The meet of two things that are each one of two alternatives, each
   alternative met in a scope of its own.

   This file is to be reviewed: the statements, which are about
   [meet_disjunction] of FlambdaMeet.v and the functions that it is
   made of. The proofs are in FlambdaMeetScopesProofs.v, which needs no
   review.

   [meet_disjunction] is the meet of the two sides of a variant (an
   immediate or a block) and of a value that may be null (not null, or
   null). Each input has a part for each side and, for the variant, an
   extension for each side: equations that hold where the value is of
   that side. The function enters a scope ([Meet_env.enter_scope]),
   and for each side adds the extensions that the two inputs have for
   it and meets the two parts, in the environment one scope further.
   If one side is bottom, what the other added goes to the environment
   of the meet: the level of its scope is cut ([Typing_env.cut]), the
   variables that the level defines are defined there
   ([add_defined_vars]) and its equations are added as an extension
   ([extension_of_level]). If neither is bottom, the two environments
   are joined, the variables that the two levels define and that the
   result of the join does not know are defined in it, and the
   equations of the two levels are the extensions of the answer.

   What is stated ([meet_disjunction_ok_statement], which is the
   proposition [meet_disjunction_right]): the meet so made is right,
   in the sense of FlambdaMeetStepsSpec.v ([part_meet_ok]), if the
   meets of the two sides are, given a join that has its contract
   ([join_scoped_envs_sound], FlambdaMeetSpec.v) and a meet of
   expanded types that has its two.
   - A thing with two sides fits an environment if its two parts do
     and each of its extensions is well formed, fits the environment
     and is kinded there ([disjunction_fits]).
   - It is true of the value in hand if one side is: the part is, and
     the extension for that side, if there is one
     ([disjunction_satisfied]). The meaning of a variant
     (FlambdaTypeMeaningSpec.v) is of that form. That of a value that
     may be null says more: on the side of a value that is not null it
     says something of the part for null. The statement does not carry
     that; a statement of what the parts of the answer are is to be
     put beside it by the meet of that head.
   - The two sides exclude each other: no value has a part of one
     side and a part of the other. Without that the meet would be
     wrong: a value that is of one side by the first input and of the
     other by the second may have neither of the two meets.
   - The parts that stand for a side that is bottom fit every
     environment.
   - The join is called as its contract supposes: the two
     environments joined are made from the environment of the meet
     ([scoped_env_of]), and no simple is at two kinds in them
     ([scoped_envs_agree]), the second side being given the names to
     avoid of the first.
   - The choice of the meet that is to be proved: a variable that
     the environment knows is not defined again ([define_again] is
     false; FlambdaMeet.v says what the compiler does).
   - What it is for a part to fit an environment is to depend on the
     environment only through the names that it knows and their kinds
     ([fits_by_kinds]): the part that the meet of a side answers fits
     the environment of that side, and the answer is to fit the
     environment that the join answers, which is not the environment
     of the side extended.

   Nothing is said of how precise the answer is, nor of a call that
   does not answer.

   The meet of row-like structures uses the same functions. What they
   keep is in the proofs file, as lemmas: the environment of an
   alternative, the level that is cut from it, the extension made of
   that level, and the variables of a level defined in another
   environment. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaMeetGrammar FlambdaExpandHead
  FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec FlambdaMeetStepsSpec
  FlambdaMeetEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* What it is to fit, by names and kinds                                    *)
(* ======================================================================== *)

(* The second environment knows every name that the first knows, and
   has every simple at the kinds at which the first has it. *)
Definition knows_at_kinds (tenv tenv' : Typing_env.t) : Prop :=
  (∀ n : name,
     Typing_env.mem None tenv n = true -> Typing_env.mem None tenv' n = true)
  ∧ (∀ (s : simple) (kind : flambda_kind),
       simple_of_kind tenv s kind -> simple_of_kind tenv' s kind).

(* What it is for a part to fit depends on the environment only
   through the names that it knows and their kinds. [type_fits] is
   so. *)
Definition fits_by_kinds {A : Type} (fits : Typing_env.t -> A -> Prop)
    : Prop :=
  ∀ (tenv tenv' : Typing_env.t) (a : A),
    knows_at_kinds tenv tenv' -> fits tenv a -> fits tenv' a.

(* ======================================================================== *)
(* A thing with two sides                                                   *)
(* ======================================================================== *)

(* An extension inside a type fits an environment: it is well formed,
   the environment knows its names, the types of its equations fit,
   and each name is of the kind of the type of its equation. *)
Definition extension_fits (tenv : Typing_env.t) (extension : env_extension)
    : Prop :=
  env_extension_well_formed extension = true
  ∧ env_extension_fits tenv extension
  ∧ env_extension_kinded tenv extension.

(* The extensions that a thing with two sides has for the first side
   and for the second. *)
Definition extension_when_a (extensions : variant_extensions)
    : option env_extension :=
  match extensions with
  | No_extensions => None
  | Ext when_a _ => Some when_a
  end.

Definition extension_when_b (extensions : variant_extensions)
    : option env_extension :=
  match extensions with
  | No_extensions => None
  | Ext _ when_b => Some when_b
  end.

Definition optional_extension_fits (tenv : Typing_env.t)
    (extension : option env_extension) : Prop :=
  match extension with
  | None => True
  | Some extension => extension_fits tenv extension
  end.

Definition optional_extension_satisfied (statics : static_env)
    (valuation : env) (extension : option env_extension) : Prop :=
  match extension with
  | None => True
  | Some extension => env_extension_satisfied statics valuation extension
  end.

Section Disjunction.

Context {A B : Type}.

Variable fits_a : Typing_env.t -> A -> Prop.
Variable satisfied_a : static_env -> env -> A -> Prop.
Variable fits_b : Typing_env.t -> B -> Prop.
Variable satisfied_b : static_env -> env -> B -> Prop.

(* A thing with two sides fits an environment: its two parts do, and
   so do its extensions. *)
Definition disjunction_fits (tenv : Typing_env.t)
    (x : A * B * variant_extensions) : Prop :=
  let '(a, b, extensions) := x in
  fits_a tenv a
  ∧ fits_b tenv b
  ∧ optional_extension_fits tenv (extension_when_a extensions)
  ∧ optional_extension_fits tenv (extension_when_b extensions).

(* It is true of the value in hand: one side is, with its
   extension. *)
Definition disjunction_satisfied (statics : static_env) (valuation : env)
    (x : A * B * variant_extensions) : Prop :=
  let '(a, b, extensions) := x in
  (satisfied_a statics valuation a
   ∧ optional_extension_satisfied statics valuation
       (extension_when_a extensions))
  ∨ (satisfied_b statics valuation b
     ∧ optional_extension_satisfied statics valuation
         (extension_when_b extensions)).

End Disjunction.

(* [meet_disjunction], of two things with two sides. *)
Definition meet_disjunction_of {A B : Type} (join : join_scoped_envs)
    (choices : meet_choices) (mh : meet_expanded_head)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B) (menv : Meet_env.t)
    (x1 x2 : A * B * variant_extensions)
    : option (Meet_result.t (A * B * variant_extensions)) :=
  let '(a1, b1, extensions1) := x1 in
  let '(a2, b2, extensions2) := x2 in
  meet_disjunction join choices mh meet_a meet_b bottom_a bottom_b menv a1 b1
    extensions1 a2 b2 extensions2.

(* The meet of two things with two sides is right, if the meets of the
   two sides are. *)
Definition meet_disjunction_right : Prop :=
  ∀ (A B : Type)
    (fits_a : Typing_env.t -> A -> Prop)
    (satisfied_a : static_env -> env -> A -> Prop)
    (fits_b : Typing_env.t -> B -> Prop)
    (satisfied_b : static_env -> env -> B -> Prop)
    (join : join_scoped_envs) (choices : meet_choices)
    (mh : meet_expanded_head)
    (meet_a : Meet_env.t -> A -> A -> option (Meet_result.t A))
    (meet_b : Meet_env.t -> B -> B -> option (Meet_result.t B))
    (bottom_a : unit -> A) (bottom_b : unit -> B),
    join_scoped_envs_sound join ->
    define_again choices = false ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    part_stable fits_a satisfied_a ->
    part_stable fits_b satisfied_b ->
    fits_by_kinds fits_a ->
    fits_by_kinds fits_b ->
    part_meet_ok fits_a satisfied_a meet_a ->
    part_meet_ok fits_b satisfied_b meet_b ->
    (∀ (statics : static_env) (valuation : env) (a : A) (b : B),
       satisfied_a statics valuation a ->
       satisfied_b statics valuation b ->
       False) ->
    (∀ tenv : Typing_env.t, fits_a tenv (bottom_a tt)) ->
    (∀ tenv : Typing_env.t, fits_b tenv (bottom_b tt)) ->
    part_meet_ok (disjunction_fits fits_a fits_b)
      (disjunction_satisfied satisfied_a satisfied_b)
      (meet_disjunction_of join choices mh meet_a meet_b bottom_a bottom_b).

Definition meet_disjunction_ok_statement : Prop := meet_disjunction_right.
