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



(* What expanding the head of a type keeps.

   This file is to be reviewed: the statements, which are about
   [expand_head] of FlambdaExpandHead.v
   ([Expand_head.expand_head], types/expand_head.ml): the type that the
   environment has for the canonical element of an alias, or the type
   itself if it is not an alias. Meet and the join work on what it
   answers. The proofs are in FlambdaExpandHeadProofs.v, which needs
   no review.

   What is stated, of [expand_head tenv ty = Some expanded]:
   - values are kept ([expand_head_sound_statement]): a value that has
     the type, under a valuation that satisfies the environment
     strictly, has the type expanded. Nothing is asked of the
     environment or of the type: the canonical element has the value
     of the alias, and a name has the type that the environment has
     for it (FlambdaTypingEnvSpec.v). It is false with a name exempt
     (that file has the example);
   - names ([expand_head_closed_statement]): if the types of the
     environment mention only variables that it knows, and the type
     does, so does the type expanded;
   - kinds of aliases ([expand_head_kinded_statement]): if the types
     of the environment and the type are kinded
     (FlambdaTypingEnvKindedSpec.v), so is the type expanded;
   - kinds of equations ([expand_head_equations_kinded_statement]):
     the same of the names that have an equation in an extension
     inside the types ([type_equations_kinded] of that file);
   - well-formedness ([expand_head_well_formed_statement]): if the
     types of the environment and the type are well formed
     (FlambdaTypeWellFormedSpec.v), so is the type expanded;
   - the kind ([expand_head_kind_statement]): the type expanded is of
     the kind of the type, if the type is kinded and the environment
     has [constants_of_kind]. The compiler does not check this where
     the canonical element is a constant: it answers the type of the
     constant, whatever the kind of the alias
     ([expand_head_of_alias_type]; for a name [Typing_env.find] checks
     the kind). So an alias of kind value whose canonical element is a
     naked immediate expands to a type of naked immediates;
   - the head ([expand_head_head_of_kind_statement]): a head answered
     is of the kind answered ([head_kind]). A type made with the
     constructor of the naked numbers at the kind of the immediates
     expands to a head of naked numbers, at that kind, and not to a
     head of naked immediates: that such a type is not well formed is
     for the predicate of well formed types to say. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeWellFormedSpec FlambdaBindingTime
  FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec.
Import ListNotations.

(* The kind of a head of an expanded type. *)
Definition head_kind (head : Expanded_type.head) : flambda_kind :=
  match head with
  | Expanded_type.Head_value _ => Value
  | Expanded_type.Head_naked_immediate _ => Naked_number Kind_naked_immediate
  | Expanded_type.Head_naked_number k _ => Naked_number k
  | Expanded_type.Head_rec_info _ => Rec_info
  | Expanded_type.Head_region _ => Region
  end.

(* That a constant which is the canonical element of a [Simple] of
   some kind is of that kind is [constants_of_kind], of
   FlambdaTypingEnvKindedSpec.v. It follows from the condition that
   two [Simple]s known to be equal are of one kind
   ([aliases_kinded_constants_of_kind_statement] there). *)

(* Values are kept. *)
Definition expand_head_sound_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (tenv : Typing_env.t)
    (ty : flambda_type) (expanded : Expanded_type.t) (v : value),
    typing_env_satisfied_strictly statics valuation tenv ->
    type_satisfied statics valuation v ty ->
    expand_head tenv ty = Some expanded ->
    type_satisfied statics valuation v (Expanded_type.to_type expanded).

(* The variables of the type expanded are known. *)
Definition expand_head_closed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    types_closed tenv ->
    (∀ x : variable,
      x ∈ free_variables ty ->
      Typing_env.mem None tenv (Name_var x) = true) ->
    expand_head tenv ty = Some expanded ->
    ∀ x : variable,
      x ∈ free_variables (Expanded_type.to_type expanded) ->
      Typing_env.mem None tenv (Name_var x) = true.

(* The type expanded is kinded. *)
Definition expand_head_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    types_kinded tenv ->
    type_kinded tenv ty ->
    expand_head tenv ty = Some expanded ->
    type_kinded tenv (Expanded_type.to_type expanded).

(* The equations inside the type expanded are kinded. *)
Definition expand_head_equations_kinded_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    types_equations_kinded tenv ->
    type_equations_kinded tenv ty ->
    expand_head tenv ty = Some expanded ->
    type_equations_kinded tenv (Expanded_type.to_type expanded).

(* The type expanded is well formed. *)
Definition expand_head_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    expand_head tenv ty = Some expanded ->
    types_well_formed tenv ->
    type_well_formed ty = true ->
    type_well_formed (Expanded_type.to_type expanded) = true.

(* The type expanded is of the kind of the type. *)
Definition expand_head_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t),
    constants_of_kind tenv ->
    type_kinded tenv ty ->
    expand_head tenv ty = Some expanded ->
    Expanded_type.kind expanded = kind ty
    ∧ kind (Expanded_type.to_type expanded) = kind ty.

(* A head answered is of the kind answered. *)
Definition expand_head_head_of_kind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (ty : flambda_type) (expanded : Expanded_type.t)
    (head : Expanded_type.head),
    expand_head tenv ty = Some expanded ->
    Expanded_type.descr expanded = Ok head ->
    head_kind head = Expanded_type.kind expanded.

(* The compiler's answer where the kind is not checked: an alias of
   kind value to the naked immediate 3 expands, in the empty
   environment, to a type of naked immediates. *)
Definition expand_head_other_kind_statement : Prop :=
  ∃ expanded : Expanded_type.t,
    expand_head Typing_env.create
      (alias_type_of Value (Const (Naked_immediate 3)))
    = Some expanded
    ∧ Expanded_type.kind expanded = Naked_number Kind_naked_immediate.
