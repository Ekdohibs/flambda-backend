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

(* The meet of two heads of naked immediates.

   This file is to be reviewed: the statements, which are about
   [meet_head_of_kind_naked_immediate] of FlambdaMeet.v and the two
   functions that it is made of. The proofs are in
   FlambdaMeetImmediateProofs.v, which needs no review.

   A head of naked immediates has the numbers that the value may be,
   if they are known, and relations: pairs of a relation ([Is_null],
   [Is_int], [Get_tag]) and a name, which say that the number is the
   result of the relation on the value of the name. The meet

   - reduces each head in the environment
     ([reduce_head_of_kind_naked_immediate]): a relation on a name
     whose canonical element is a constant gives the number and goes,
     and the other relations are put on canonical names;
   - meets the numbers, and puts the relations of the two together;
   - adds to the environment what the numbers that are left say of
     the names of the relations that had not those numbers before
     ([reduce_inverse_relations]): the name of an [Is_null] that can
     only be 0 is not null, the name of a [Get_tag] that is among some
     tags is an immediate or a block with one of those tags, and so
     on. Each is an equation added with [Meet_env.add_equation_strict],
     which runs the meet of expanded types.

   What is stated.

   - Reducing a head keeps its fit
     ([reduce_head_of_kind_naked_immediate_fits_statement]). That it
     loses no value is in FlambdaMeetStepsSpec.v.
   - The meet of two heads is right, in the sense of [part_meet_ok]
     (FlambdaMeetStepsSpec.v), for a meet of expanded types that has
     its two contracts and the third below, and for the type of the
     [Get_tag] step that is true
     ([meet_head_of_kind_naked_immediate_ok_statement]). A head fits
     an environment, and is true of the value in hand, as the type
     that it makes does. The step for [Get_tag] as the compiler has it
     is not sound (FlambdaMeetSpec.v,
     [get_tag_reduction_unsound_statement]).

   The kinds of the names of relations. A type does not say of what
   kind the name of a relation is, and [type_fits] asks nothing of it:
   the names of relations are not among the aliases of a type
   ([aliases_with_kinds], FlambdaTypeGrammar.v), and a type that fits
   may have a relation on a name of any kind. The statements about
   adding an equation (FlambdaMeetEnvSpec.v) suppose that the name is
   of the kind of the type, and here the types added are of kind
   value. So that nothing more is asked of the inputs, the proof goes
   by what the function does where the kind is another: it reads the
   type of a canonical name at the kind of the type that it adds, a
   fatal error if the two differ; it ignores an equation on a name
   that an equation is being added to; and for a canonical element
   that is a constant it runs the meet of expanded types on the type
   and the type of the constant. That last case is why a third thing
   is supposed of the meet of expanded types
   ([meet_expanded_head_other_kinds]): on two types of different
   kinds it adds nothing to the environment. The transcribed meet has
   it at every fuel ([meet_expanded_head_fuel_other_kinds_statement]):
   two heads of different kinds are a fatal error, and where one of
   the two is unknown the environment answered is the one given. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetStepsSpec.
Import ListNotations.

(* The type that a head of naked immediates makes. *)
Definition naked_immediate_head_type (head : head_of_kind_naked_immediate)
    : flambda_type :=
  Type_naked_immediate (Ok (No_alias head)).

(* [reduce_head_of_kind_naked_immediate] keeps the fit of a head: the
   head answered fits the environment that the first was reduced
   in. *)
Definition reduce_head_of_kind_naked_immediate_fits_statement : Prop :=
  ∀ (tenv : Typing_env.t) (head head' : head_of_kind_naked_immediate),
    typing_env_fits tenv ->
    type_fits tenv (naked_immediate_head_type head) ->
    reduce_head_of_kind_naked_immediate tenv head = Or_bottom.Ok head' ->
    type_fits tenv (naked_immediate_head_type head').

(* The meet of expanded types adds nothing where the two types are of
   different kinds. *)
(* CR mshinwell: this is said of the meet as it is transcribed, and it
   is what the compiler does: an assertion fails where two heads are
   of different kinds, and where one of the two types is unknown the
   other is answered with the environment as it was. It is supposed
   here so that nothing is asked of the kinds of the names of
   relations, on which a type that fits says nothing (decisions D10,
   D12 and D32: on kinds the model asks no more than the compiler
   checks). The other way is a premise that those names are of kind
   value, in [type_fits]. *)
Definition meet_expanded_head_other_kinds (meet_head : meet_expanded_head)
    : Prop :=
  ∀ (menv menv' : Meet_env.t) (expanded1 expanded2 : Expanded_type.t)
    (r : meet_return_value Expanded_type.t),
    kind (Expanded_type.to_type expanded1)
    <> kind (Expanded_type.to_type expanded2) ->
    meet_head menv expanded1 expanded2 = Some (Meet_result.Ok r menv') ->
    menv' = menv.

(* The transcribed meet of expanded types has it, whatever the join,
   the choices and the fuel. *)
Definition meet_expanded_head_fuel_other_kinds_statement : Prop :=
  ∀ (join : join_scoped_envs) (choices : meet_choices) (fuel : nat),
    meet_expanded_head_other_kinds
      (Meet.meet_expanded_head join choices fuel).

(* [meet_head_of_kind_naked_immediate] is right, for any value: with
   the type of the [Get_tag] step that is true (an immediate, or a
   block with one of the tags), and a meet of expanded types that has
   its contracts. *)
Definition meet_head_of_kind_naked_immediate_ok_statement : Prop :=
  ∀ (choices : meet_choices) (mh : meet_expanded_head) (v : value),
    type_of_get_tag_scrutinee choices = immediate_or_blocks_with_these_tags ->
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    meet_expanded_head_other_kinds mh ->
    part_meet_ok
      (fun (tenv : Typing_env.t) (head : head_of_kind_naked_immediate) =>
         type_fits tenv (naked_immediate_head_type head))
      (fun (statics : static_env) (valuation : env)
           (head : head_of_kind_naked_immediate) =>
         type_satisfied statics valuation v (naked_immediate_head_type head))
      (meet_head_of_kind_naked_immediate choices mh).
