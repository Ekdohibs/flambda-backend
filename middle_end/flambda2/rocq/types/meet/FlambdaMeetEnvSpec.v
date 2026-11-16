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



(* What the functions of [Meet_env] keep, and that what they add is
   true.

   This file is to be reviewed: the statements, which are about the
   functions of FlambdaMeetEnv.v ([Meet_env], types/env/meet_env.ml):
   the functions through which a meet, and the join that it calls,
   change a typing environment. The proofs are in
   FlambdaMeetEnvProofs.v, which needs no review.

   There are two statements for a function. The first says that it
   answers an environment that is the one given, extended
   ([meet_env_extends], FlambdaMeetSpec.v), and in which a meet may be
   made ([typing_env_fits]): [meet_env_kept]. No value and no
   valuation is supposed. The second (its name begins with [meet_env_]
   and ends in [_sound]) is about a valuation that satisfies the
   environment given, strictly, and under which what is added is
   true: the name has a value, of the type given for it. The static
   environment is valid ([statics_valid]: FlambdaMeetSpec.v says
   why). Then the
   function does not answer bottom, and a valuation that agrees with
   the first on what the environment knew satisfies the environment
   answered; and no equation has been added on a name that has no
   value ([meet_env_added]). The second
   statement supposes of the meet of expanded types both
   [meet_expanded_head_keeps] and [meet_expanded_head_sound].

   What is stated in the first form:
   - of [Meet_env.make_bottom] ([make_bottom_keeps_statement]);
   - of [Meet_env.add_variable_definition]
     ([add_variable_definition_keeps_statement]), for a variable that
     is not bound (the compiler refuses one that is, when invariant
     checks are on, and the invariant of typing environments is kept
     only then);
   - of the functions that add an equation, given a meet of expanded
     types that has [meet_expanded_head_keeps]:
     [Meet_env.add_equation_on_simple], [Meet_env.add_equation],
     [Meet_env.add_equation_strict], [Meet_env.add_env_extension],
     [Meet_env.add_env_extension_strict], [Meet_env.add_alias] and
     [Meet_env.add_equations_on_params]. The strict ones and
     [add_alias] answer bottom or an environment; the statement is
     about the environment. The others answer the environment made
     bottom where the equation has no value. These functions run the
     meet of expanded types, which may define variables (the join
     that it calls makes some): what they keep says also that the
     variables defined are new ([meet_env_kept_by_meet], with
     [new_names_avoided] of FlambdaMeetSpec.v);
   - of [Meet_env.meet_type], the meet of two types of which either
     may be an alias ([meet_type_keeps_statement]): the type answered
     fits the environment answered and is of the kind of the
     inputs.

   What is asked of an equation, beside that its type fits: the name,
   or the [Simple], is of the kind of the type ([simple_of_kind],
   FlambdaTypingEnvKindedSpec.v; for an extension,
   [env_extension_kinded]). The functions check the kind themselves
   only where the type is not an alias and the canonical element of
   the name is a name ([Typing_env.find] with the kind, a fatal
   error). Where the type is an alias, two canonical elements are
   made aliases whatever their kinds; where the canonical element is
   a constant, the type is met with the type of the constant. An
   environment in which two [Simple]s known to be equal are of two
   kinds is not kinded, and the join that the meet calls needs a
   kinded one. The condition is about kinds, on which the model may
   not ask more than the compiler: there is a CR at
   [env_extension_kinded]. The functions of [Typing_env] that are
   called from outside a meet ([Typing_env.add_equation], ...,
   FlambdaMeetSpec.v) have no such premise: no equation is being
   added when they are called, and what is to be shown of them is
   that a name of another kind is then a fatal error.

   [Meet_env.add_alias] reads the kind of the first canonical element
   ([kind_of_simple], FlambdaMeetGrammar.v), and takes a symbol to be
   of the kind of values: a kinded environment has no symbol of
   another kind ([symbols_of_kind_value]).

   What is stated in the second form: of the same functions that add
   an equation, and of [Meet_env.meet_type], for which the two forms
   are put together as [part_meet_ok] of FlambdaMeetStepsSpec.v
   ([meet_type_ok_statement]): the meet of two types is right, given
   a meet of expanded types that is. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvKindedSpec
  FlambdaMeetGrammar FlambdaExpandHead FlambdaMeetEnv FlambdaMeet
  FlambdaMeetSpec FlambdaMeetStepsSpec.
Import ListNotations.

(* What a function of [Meet_env] keeps: the environment that it
   answers is the one that it was given, extended, and a meet may be
   made in it. *)
Definition meet_env_kept (menv menv' : Meet_env.t) : Prop :=
  meet_env_extends menv menv' ∧ typing_env_fits (Meet_env.typing_env menv').

(* The same of a function that runs the meet of expanded types: and the
   variables that the second environment knows and the first does not
   are new. *)
Definition meet_env_kept_by_meet (menv menv' : Meet_env.t) : Prop :=
  meet_env_kept menv menv' ∧ new_names_avoided menv menv'.

(* [Meet_env.make_bottom]. *)
Definition make_bottom_keeps_statement : Prop :=
  ∀ menv : Meet_env.t,
    typing_env_fits (Meet_env.typing_env menv) ->
    meet_env_kept menv (Meet_env.make_bottom menv).

(* [Meet_env.add_variable_definition], of a variable that is not
   bound. *)
Definition add_variable_definition_keeps_statement : Prop :=
  ∀ (menv : Meet_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    typing_env_fits (Meet_env.typing_env menv) ->
    Typing_env.mem None (Meet_env.typing_env menv) (Name_var var) = false ->
    meet_env_kept menv
      (Meet_env.add_variable_definition menv var kind name_mode).

(* ======================================================================== *)
(* The functions that add an equation                                       *)
(* ======================================================================== *)

(* The names that an extension has equations on are of the kinds of
   the types of their equations. *)
(* CR mshinwell: this is a premise about kinds, on which the model may
   not ask more than the compiler checks (decisions D10, D12 and
   D32). The compiler checks the kind of the name of an equation only
   where the type is not an alias and the canonical element of the
   name is a name. It is not known whether an extension that has not
   this can reach [Meet_env.add_env_extension] in the middle of a
   meet. Discrepancy T5-9. *)
Definition env_extension_kinded (tenv : Typing_env.t)
    (extension : env_extension) : Prop :=
  match extension with
  | Env_extension equations =>
    ∀ (n : name) (ty : flambda_type),
      Keyed.lookup name_eqb n equations = Some ty ->
      simple_of_kind tenv (simple_of_name n) (kind ty)
  end.

(* [Meet_env.add_equation_on_simple]. *)
Definition add_equation_on_simple_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t) (s : simple)
    (ty : flambda_type),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) s (kind ty) ->
    Meet_env.add_equation_on_simple menv s ty mh = Some menv' ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_equation]. *)
Definition add_equation_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t) (n : name)
    (ty : flambda_type),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) (simple_of_name n) (kind ty) ->
    Meet_env.add_equation menv n ty mh = Some menv' ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_equation_strict]. *)
Definition add_equation_strict_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t) (n : name)
    (ty : flambda_type),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) (simple_of_name n) (kind ty) ->
    Meet_env.add_equation_strict menv n ty mh = Some (Or_bottom.Ok menv') ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_env_extension]. *)
Definition add_env_extension_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (extension : env_extension),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    env_extension_fits (Meet_env.typing_env menv) extension ->
    env_extension_kinded (Meet_env.typing_env menv) extension ->
    Meet_env.add_env_extension menv extension mh = Some menv' ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_env_extension_strict]. *)
Definition add_env_extension_strict_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (extension : env_extension),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    env_extension_fits (Meet_env.typing_env menv) extension ->
    env_extension_kinded (Meet_env.typing_env menv) extension ->
    Meet_env.add_env_extension_strict menv extension mh
    = Some (Or_bottom.Ok menv') ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_alias], of two [Simple]s of one kind. *)
Definition add_alias_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t) (s1 s2 : simple)
    (kind : flambda_kind),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    simple_of_kind (Meet_env.typing_env menv) s1 kind ->
    simple_of_kind (Meet_env.typing_env menv) s2 kind ->
    Meet_env.add_alias menv s1 s2 mh = Some (Or_bottom.Ok menv') ->
    meet_env_kept_by_meet menv menv'.

(* [Meet_env.add_equations_on_params]: each parameter is of the kind
   of the type given for it. *)
Definition add_equations_on_params_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (params : list (variable * flambda_kind))
    (param_types : list flambda_type),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    Forall2
      (fun (param : variable * flambda_kind) (ty : flambda_type) =>
         type_fits (Meet_env.typing_env menv) ty
         ∧ simple_of_kind (Meet_env.typing_env menv) (Var (fst param))
             (kind ty))
      params param_types ->
    Meet_env.add_equations_on_params menv params param_types mh = Some menv' ->
    meet_env_kept_by_meet menv menv'.

(* ======================================================================== *)
(* The meet of two types                                                    *)
(* ======================================================================== *)

(* [Meet_env.meet_type]: the meet of two types that fit, of which
   either may be an alias. Where the answer is not bottom, the
   environment is kept, and the type answered fits it and is of the
   kind of the inputs. Types of two kinds are a fatal error. *)
Definition meet_type_keeps_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t)
    (t1 t2 : flambda_type) (r : meet_return_value flambda_type),
    meet_expanded_head_keeps mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) t1 ->
    type_fits (Meet_env.typing_env menv) t2 ->
    Meet_env.meet_type menv t1 t2 mh = Some (Meet_result.Ok r menv') ->
    meet_env_kept_by_meet menv menv'
    ∧ type_fits (Meet_env.typing_env menv') (extract_value r t1 t2)
    ∧ kind (extract_value r t1 t2) = kind t1.

(* ======================================================================== *)
(* What the functions add is true                                           *)
(* ======================================================================== *)

(* What was added to an environment is true: a valuation that agrees
   with [valuation] on what the first environment knew satisfies the
   second strictly, and no equation has been added on a name that has
   no value. *)
Definition meet_env_added (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) : Prop :=
  let tenv := Meet_env.typing_env menv in
  let tenv' := Meet_env.typing_env menv' in
  ∃ valuation' : env,
    extended_from tenv valuation valuation'
    ∧ typing_env_satisfied_strictly statics valuation' tenv'
    ∧ keeps_equations_valued tenv tenv' valuation valuation'.

(* The same of a function that may answer bottom: it does not. *)
Definition meet_env_sound (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (result : Or_bottom.t Meet_env.t) : Prop :=
  ∃ menv' : Meet_env.t,
    result = Or_bottom.Ok menv' ∧ meet_env_added statics valuation menv menv'.

(* [Meet_env.add_equation_on_simple]: the [Simple] has a value, of the
   type. *)
Definition meet_env_add_equation_on_simple_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) (s : simple) (ty : flambda_type) (v : value),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) s (kind ty) ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    eval_simple valuation s = Some v ->
    type_satisfied statics valuation v ty ->
    Meet_env.add_equation_on_simple menv s ty mh = Some menv' ->
    meet_env_added statics valuation menv menv'.

(* [Meet_env.add_equation]. *)
Definition meet_env_add_equation_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) (n : name) (ty : flambda_type) (v : value),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) (simple_of_name n) (kind ty) ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    value_of_name valuation n = Some v ->
    type_satisfied statics valuation v ty ->
    Meet_env.add_equation menv n ty mh = Some menv' ->
    meet_env_added statics valuation menv menv'.

(* [Meet_env.add_equation_strict]. *)
Definition meet_env_add_equation_strict_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (n : name) (ty : flambda_type) (v : value)
    (result : Or_bottom.t Meet_env.t),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    simple_of_kind (Meet_env.typing_env menv) (simple_of_name n) (kind ty) ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    value_of_name valuation n = Some v ->
    type_satisfied statics valuation v ty ->
    Meet_env.add_equation_strict menv n ty mh = Some result ->
    meet_env_sound statics valuation menv result.

(* [Meet_env.add_env_extension]: the extension is true, which says
   that each of its names has a value, of the type of its equation
   (FlambdaTypeMeaningSpec.v). *)
Definition meet_env_add_env_extension_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) (extension : env_extension),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    env_extension_fits (Meet_env.typing_env menv) extension ->
    env_extension_kinded (Meet_env.typing_env menv) extension ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    env_extension_satisfied statics valuation extension ->
    Meet_env.add_env_extension menv extension mh = Some menv' ->
    meet_env_added statics valuation menv menv'.

(* [Meet_env.add_env_extension_strict]. *)
Definition meet_env_add_env_extension_strict_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (extension : env_extension)
    (result : Or_bottom.t Meet_env.t),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    env_extension_fits (Meet_env.typing_env menv) extension ->
    env_extension_kinded (Meet_env.typing_env menv) extension ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    env_extension_satisfied statics valuation extension ->
    Meet_env.add_env_extension_strict menv extension mh = Some result ->
    meet_env_sound statics valuation menv result.

(* [Meet_env.add_alias]: the two [Simple]s have one value. *)
Definition meet_env_add_alias_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv : Meet_env.t) (s1 s2 : simple) (kind : flambda_kind) (v : value)
    (result : Or_bottom.t Meet_env.t),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    simple_of_kind (Meet_env.typing_env menv) s1 kind ->
    simple_of_kind (Meet_env.typing_env menv) s2 kind ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    eval_simple valuation s1 = Some v ->
    eval_simple valuation s2 = Some v ->
    Meet_env.add_alias menv s1 s2 mh = Some result ->
    meet_env_sound statics valuation menv result.

(* [Meet_env.add_equations_on_params]: each parameter has a value, of
   the type given for it. *)
Definition meet_env_add_equations_on_params_sound_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) (params : list (variable * flambda_kind))
    (param_types : list flambda_type),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    Forall2
      (fun (param : variable * flambda_kind) (ty : flambda_type) =>
         type_fits (Meet_env.typing_env menv) ty
         ∧ simple_of_kind (Meet_env.typing_env menv) (Var (fst param))
             (kind ty))
      params param_types ->
    statics_valid statics ->
    typing_env_satisfied_strictly statics valuation (Meet_env.typing_env menv)
    ->
    Forall2
      (fun (param : variable * flambda_kind) (ty : flambda_type) =>
         ∃ v : value,
           lookup (fst param) valuation = Some v
           ∧ type_satisfied statics valuation v ty)
      params param_types ->
    Meet_env.add_equations_on_params menv params param_types mh = Some menv' ->
    meet_env_added statics valuation menv menv'.

(* [Meet_env.meet_type] is right, as a meet of parts
   (FlambdaMeetStepsSpec.v): the part is a type, which fits as a type
   does and is true where the value in hand has it. *)
Definition meet_type_ok_statement : Prop :=
  ∀ (mh : meet_expanded_head) (v : value),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    part_meet_ok type_fits
      (fun (statics : static_env) (valuation : env) (ty : flambda_type) =>
         type_satisfied statics valuation v ty)
      (fun (menv : Meet_env.t) (t1 t2 : flambda_type) =>
         Meet_env.meet_type menv t1 t2 mh).
