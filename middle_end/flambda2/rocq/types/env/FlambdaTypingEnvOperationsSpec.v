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


(* What the functions that build a typing environment do to what
   satisfies it.

   This file is to be reviewed: the statements. The functions are those
   of FlambdaTypingEnv.v ([Typing_env] of
   middle_end/flambda2/types/env/typing_env.ml); what satisfies an
   environment is FlambdaTypingEnvSpec.v; the invariant, and that the
   functions keep it, is FlambdaTypingEnvInvariantSpec.v.

   Each statement is about [typing_env_satisfied_but] with any names
   exempt, the same before and after, so that it holds of strict
   satisfaction, which is the notion for the statements about types.
   It does not follow for [typing_env_satisfied], whose exempt names
   depend on the types of the environment.

   The functions, and what is stated of each:
   - [increment_scope], [closure_env], [add_symbol_definition]: the
     same valuations satisfy the environment before and after. They do
     not change the types of the names or the aliases, which is all
     that satisfaction reads. ([closure_env] changes which variables
     are in scope, so the machine is another matter; that is for the
     statements about terms.)
   - [add_variable_definition]: a valuation that satisfies the
     environment satisfies the new one if the value that it has for the
     variable, if it has one, is of the kind; and so the valuation with
     the variable bound to any value of the kind does, if the
     environment does not mention the variable, which is so of a
     variable that is not bound in an environment that is well formed.
     The other way: what satisfies the new environment satisfies the
     old.
   - [add_definitions_of_params]: the valuation with the parameters
     bound as the machine binds them, each to a value of its kind,
     satisfies the new environment, if the parameters are different
     variables that the environment does not mention.
   - [replace_equation] and [add_alias]: what each leaves as it is
     ([replace_equation_frame_statement], the two on [lookup],
     [add_alias_frame_statement]), for the statements about
     [Meet_env], which calls them in turn.
   - [replace_equation]: a valuation that satisfies the environment
     satisfies the new one if the value of the name, if it has one, has
     the new type. The other way, what satisfies the new environment
     satisfies the old if the value of the name has the type that the
     name had: [replace_equation] does not compare the two types. That
     the new type implies the old is for the meet to give
     ([Meet_env.add_equation] gives the name the meet of the two).
   - [add_alias]: a valuation that satisfies the environment and gives
     the two the same value, or gives neither a value, satisfies the
     new one; and where the answer is bottom, no valuation gives the
     two the same value.
   - The machine: binding a variable that the environment does not
     know, and the static environment growing, keep satisfaction.

   The proofs are in FlambdaTypingEnvOperationsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec.
Import ListNotations.

(* ======================================================================== *)
(* Scopes and symbols                                                       *)
(* ======================================================================== *)

Definition increment_scope_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t),
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.increment_scope tenv)
    <-> typing_env_satisfied_but exempt statics valuation tenv.

Definition closure_env_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t),
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.closure_env tenv)
    <-> typing_env_satisfied_but exempt statics valuation tenv.

Definition add_symbol_definition_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (sym : symbol),
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.add_symbol_definition tenv sym)
    <-> typing_env_satisfied_but exempt statics valuation tenv.

(* The scope of a level is not read by the machine either. *)
Definition increment_scope_env_satisfies_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies_strictly (Typing_env.increment_scope tenv) statics vars
    <-> env_satisfies_strictly tenv statics vars.

(* ======================================================================== *)
(* Defining a variable                                                      *)
(* ======================================================================== *)

(* The valuation as it is: the value that it has for the variable, if
   it has one, is of the kind. So a valuation that gives the variable
   no value satisfies the new environment. *)
Definition add_variable_definition_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    typing_env_satisfied_but exempt statics valuation tenv ->
    (∀ v : value, lookup var valuation = Some v -> kind_of_value v = kind) ->
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* The valuation with the variable bound to a value of the kind, any
   one, poison too: the environment is not to mention the variable. *)
Definition add_variable_definition_satisfied_bind_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t) (v : value),
    typing_env_satisfied_but exempt statics valuation tenv ->
    ~ typing_env_mentions tenv var ->
    kind_of_value v = kind ->
    typing_env_satisfied_but exempt statics ((var, v) :: valuation)
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* An environment that is well formed does not mention a variable that
   is not bound in it. *)
Definition well_formed_not_mentioned_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var var) = false ->
    ~ typing_env_mentions tenv var.

(* The other way: what satisfies the environment with the variable
   defined satisfies the environment without, if the variable was not
   bound. *)
Definition add_variable_definition_satisfied_inv_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    Typing_env.mem None tenv (Name_var var) = false ->
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.add_variable_definition tenv var kind name_mode) ->
    typing_env_satisfied_but exempt statics valuation tenv.

(* A variable that the environment with a variable defined mentions is
   that variable, or one that the environment mentioned. *)
Definition add_variable_definition_mentions_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t) (x : variable),
    typing_env_mentions
      (Typing_env.add_variable_definition tenv var kind name_mode) x ->
    x = var ∨ typing_env_mentions tenv x.

(* Parameters are defined, and the valuation binds them as the machine
   does at a call or a jump ([bind_params]), each to a value of its
   kind: they are different variables, and the environment mentions
   none of them. *)
Definition add_definitions_of_params_satisfied_bind_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env)
    (valuation valuation' : env) (tenv : Typing_env.t)
    (params : list (variable * flambda_kind)) (vs : list value),
    typing_env_satisfied_but exempt statics valuation tenv ->
    distinct (map fst params) = true ->
    (∀ x : variable, x ∈ map fst params -> ~ typing_env_mentions tenv x) ->
    Forall2 (fun param v => kind_of_value v = snd param) params vs ->
    bind_params (map fst params) vs valuation = Some valuation' ->
    typing_env_satisfied_but exempt statics valuation'
      (Typing_env.add_definitions_of_params tenv params).

(* ======================================================================== *)
(* A new type for a name                                                    *)
(* ======================================================================== *)

(* The value of the name, if it has one, has the new type. *)
Definition replace_equation_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    typing_env_satisfied_but exempt statics valuation tenv ->
    (∀ v : value,
      value_of_name valuation n = Some v ->
      has_type_unless (exempt n) statics valuation v ty) ->
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.replace_equation tenv n ty).

(* What [Typing_env.replace_equation] leaves as it is: all but the
   type of the name and the equations of the current level. *)
Definition replace_equation_frame_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    let tenv' := Typing_env.replace_equation tenv n ty in
    Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv
    ∧ Typing_env.prev_levels tenv' = Typing_env.prev_levels tenv
    ∧ Typing_env.current_scope tenv' = Typing_env.current_scope tenv
    ∧ Typing_env_level.defined_vars
        (Typing_env.One_level.level (Typing_env.current_level tenv'))
      = Typing_env_level.defined_vars
          (Typing_env.One_level.level (Typing_env.current_level tenv))
    ∧ Typing_env_level.binding_times
        (Typing_env.One_level.level (Typing_env.current_level tenv'))
      = Typing_env_level.binding_times
          (Typing_env.One_level.level (Typing_env.current_level tenv))
    ∧ Typing_env.next_binding_time tenv' = Typing_env.next_binding_time tenv
    ∧ Typing_env.min_binding_time tenv' = Typing_env.min_binding_time tenv
    ∧ Typing_env.is_bottom tenv' = Typing_env.is_bottom tenv
    ∧ Typing_env.aliases tenv' = Typing_env.aliases tenv.

(* An equation of the current level afterwards, counted or not, is the
   new one of the name or was an equation of the current level. *)
Definition replace_equation_equations_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n m : name) (ty ty' : flambda_type),
    (m, ty')
    ∈ Typing_env_level.equations
        (Typing_env.One_level.level
           (Typing_env.current_level (Typing_env.replace_equation tenv n ty)))
    ->
    (m = n ∧ ty' = ty)
    ∨ (m, ty')
      ∈ Typing_env_level.equations
          (Typing_env.One_level.level (Typing_env.current_level tenv)).

(* The type of another name, with its binding time and its name mode,
   is as it was. *)
Definition replace_equation_lookup_other_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n m : name) (ty : flambda_type),
    m <> n ->
    Keyed.lookup name_eqb m
      (Typing_env.names_to_types (Typing_env.replace_equation tenv n ty))
    = Keyed.lookup name_eqb m (Typing_env.names_to_types tenv).

(* A variable that has a type gets the new one and keeps its binding
   time and its name mode; one that has none gets none. A symbol gets
   the type, with the binding time and the name mode of symbols. *)
Definition replace_equation_lookup_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    Keyed.lookup name_eqb n
      (Typing_env.names_to_types (Typing_env.replace_equation tenv n ty))
    = match n with
      | Name_var _ =>
        option_map (fun entry => (ty, snd entry))
          (Keyed.lookup name_eqb n (Typing_env.names_to_types tenv))
      | Name_symbol _ => Some (ty, Binding_time.With_name_mode.symbols)
      end.

(* The other way, the value of the name is to have the type that the
   name had: nothing in [Typing_env.replace_equation] makes the new
   type imply the old. *)
Definition replace_equation_satisfied_inv_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    (∀ (ty' : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t) (v : value),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty', binding_time_and_mode) ->
      value_of_name valuation n = Some v ->
      has_type_unless (exempt n) statics valuation v ty') ->
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.replace_equation tenv n ty) ->
    typing_env_satisfied_but exempt statics valuation tenv.

(* ======================================================================== *)
(* Two canonical elements made aliases                                      *)
(* ======================================================================== *)

(* The valuation gives the two the same value, or gives neither a
   value. *)
Definition add_alias_satisfied_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    typing_env_satisfied_but exempt statics valuation tenv ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    eval_simple valuation canonical_element1
    = eval_simple valuation canonical_element2 ->
    typing_env_satisfied_but exempt statics valuation
      (Typing_env.add_alias_result_t result).

(* What [Typing_env.add_alias] answers: the environment with the
   aliases that [Aliases.add] answers, and nothing else changed; the
   canonical element and the name demoted are those of [Aliases.add]. *)
Definition add_alias_frame_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    let tenv' := Typing_env.add_alias_result_t result in
    (∃ added : Aliases.add_result,
      Aliases.add (Typing_env.binding_time_resolver tenv)
        (Typing_env.names_to_types tenv) (Typing_env.aliases tenv)
        canonical_element1 canonical_element2
      = Some (Or_bottom.Ok added)
      ∧ Typing_env.aliases tenv' = Aliases.add_result_t added
      ∧ Typing_env.canonical_element result
        = Aliases.canonical_element added
      ∧ Typing_env.demoted_name result = Aliases.demoted_name added)
    ∧ Typing_env.defined_symbols tenv' = Typing_env.defined_symbols tenv
    ∧ Typing_env.prev_levels tenv' = Typing_env.prev_levels tenv
    ∧ Typing_env.current_level tenv' = Typing_env.current_level tenv
    ∧ Typing_env.names_to_types tenv' = Typing_env.names_to_types tenv
    ∧ Typing_env.next_binding_time tenv' = Typing_env.next_binding_time tenv
    ∧ Typing_env.min_binding_time tenv' = Typing_env.min_binding_time tenv
    ∧ Typing_env.is_bottom tenv' = Typing_env.is_bottom tenv.

(* Where the answer is bottom the two are different constants: no
   valuation gives them the same value. *)
Definition add_alias_bottom_statement : Prop :=
  ∀ (valuation : env) (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple),
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some Bottom ->
    eval_simple valuation canonical_element1
    <> eval_simple valuation canonical_element2.

(* ======================================================================== *)
(* The machine and the static environment                                   *)
(* ======================================================================== *)

(* The static environment grows as a program runs: what satisfied an
   environment still does. *)
Definition satisfied_but_statics_extend_statement : Prop :=
  ∀ (exempt : name -> Prop) (statics statics' : static_env)
    (valuation : env) (tenv : Typing_env.t),
    statics_extend statics statics' ->
    typing_env_satisfied_but exempt statics valuation tenv ->
    typing_env_satisfied_but exempt statics' valuation tenv.

Definition env_satisfies_strictly_statics_extend_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics statics' : static_env) (vars : env),
    statics_extend statics statics' ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly tenv statics' vars.

(* The machine binds a variable that the environment does not know: a
   variable of the term that the typing environment has not been told
   of, or has no use for. *)
Definition env_satisfies_strictly_bind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (x : variable) (v : value),
    Typing_env.mem None tenv (Name_var x) = false ->
    env_satisfies_strictly tenv statics vars ->
    env_satisfies_strictly tenv statics ((x, v) :: vars).

Definition env_satisfies_bind_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (x : variable) (v : value),
    Typing_env.mem None tenv (Name_var x) = false ->
    env_satisfies tenv statics vars ->
    env_satisfies tenv statics ((x, v) :: vars).
