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


(* Typing environments and runs whose numbers are in range.

   This file is to be reviewed: the definition and the statements. What
   it is for the machine to satisfy a typing environment is
   FlambdaTypingEnvSpec.v ([env_satisfies_strictly]); what it is for a
   state of the machine to be in range is
   FlambdaRangeSpec.v ([state_in_range]), where every state
   of the run of a well-formed unit is shown to be.

   Some statements about types hold only of values in range: the tag
   of a block has to be a tag for a [Get_tag] relation to relate the
   block to it (FlambdaTypeMeaningSpec.v, [relation_holds]), and the
   join of environments makes such relations. A machine state in range
   has such values in its variables. But a typing environment is
   satisfied through a valuation, which also gives values to the
   variables that exist only in types, and of those the machine says
   nothing: they were chosen where the environment was built, by a
   join or a meet, or are the values that variables had that have gone
   out of scope. So that the machine satisfies an environment
   ([env_satisfies_strictly]) and is in range does not give a
   valuation in range.

   [env_satisfies_strictly_in_range] is the notion that does: the
   machine satisfies the environment through a valuation all of whose
   values are in range. It is kept by the steps by which Simplify's
   environment follows the term, where the values that the machine
   binds are in range, which they are in a run in range
   (FlambdaRangeSpec.step_in_range_statement). What a join or a meet
   does to it is for their statements to say: the witnesses that they
   choose have to be in range where the valuations that they are given
   are.

   The statements:
   - the notion gives the plain one;
   - under a valuation and a static environment in range, every block
     that a [Simple] evaluates to has a tag in range, which is a tag,
     and fields in range: the condition on runs that the join of
     environments takes, for any variables;
   - a valuation that agrees with the machine gives every variable
     that may occur in a term the value that the machine has for it,
     so such a variable has a value in the run. (Of a variable that
     exists only in types no agreement is asked: the valuation need
     not give it what the machine has, if the machine has anything
     for it.)
   - the environment that knows nothing is satisfied so by any
     machine, and one that the machine's variables satisfy is, if they
     are in range;
   - the steps keep it: a new scope, a variable defined, parameters
     defined, a type given to a variable in scope, two [Simple]s made
     equal, and the static environment growing;
   - an environment that a machine in range satisfies strictly, and
     through no valuation in range: a variable that exists only in
     types is, by its type, a number that no tagged immediate of the
     machine is, and a variable of the machine is the result of a
     relation on it ([example_out_of_range_statement]).

   Not here: a notion of refinement for runs in range. The statements
   that go under a handler (FlambdaTypingEnvRefinementSpec.v) are of
   handlers that refine whatever values, continuations and static
   environment they run under; for handlers that refine only in runs
   in range, the lemma of the core that they rest on has to be about
   such runs.

   The proofs are in FlambdaTypingEnvRangeProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaRangeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaTypingEnvInvariantSpec FlambdaTypingEnvExamplesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Satisfaction through a valuation in range                                *)
(* ======================================================================== *)

(* The static environment and the variables of a state of the machine
   satisfy a typing environment strictly, through a valuation all of
   whose values are in range. *)
Definition env_satisfies_strictly_in_range (tenv : Typing_env.t)
    (statics : static_env) (vars : env) : Prop :=
  bound_in_terms tenv vars
  ∧ ∃ valuation : env,
      typing_env_satisfied_strictly statics valuation tenv
      ∧ agrees_in_terms tenv valuation vars
      ∧ env_in_range valuation.

Definition satisfies_in_range_strictly_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly tenv statics vars.

(* ======================================================================== *)
(* What a valuation in range gives                                          *)
(* ======================================================================== *)

(* Every block that a [Simple] evaluates to, made by a primitive or
   static, has a tag in range, which is a tag, and fields in range. *)
Definition in_range_blocks_have_tags_statement : Prop :=
  ∀ (statics : static_env) (valuation : env) (simple : simple) (v : value)
    (tag : Z) (fields : list value),
    statics_in_range statics ->
    env_in_range valuation ->
    eval_simple valuation simple = Some v ->
    block_of statics v = Some (tag, fields) ->
    tag_in_range tag = true
    ∧ is_tag tag
    ∧ ∀ field : value, field ∈ fields -> value_in_range field = true.

(* A valuation that agrees with the machine gives every variable that
   may occur in a term the value that the machine has for it. *)
Definition agrees_in_terms_has_value_statement : Prop :=
  ∀ (tenv : Typing_env.t) (valuation vars : env) (x : variable),
    bound_in_terms tenv vars ->
    agrees_in_terms tenv valuation vars ->
    in_scope_in_terms tenv x ->
    ∃ v : value, lookup x valuation = Some v ∧ lookup x vars = Some v.

(* ======================================================================== *)
(* Where it holds                                                           *)
(* ======================================================================== *)

(* Any machine satisfies so the environment that knows nothing. *)
Definition create_satisfies_in_range_statement : Prop :=
  ∀ (statics : static_env) (vars : env),
    env_satisfies_strictly_in_range Typing_env.create statics vars.

(* The variables of the machine that give every name its type satisfy
   the environment so, if they are in range. *)
Definition satisfied_strictly_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    bound_in_terms tenv vars ->
    typing_env_satisfied_strictly statics vars tenv ->
    env_in_range vars ->
    env_satisfies_strictly_in_range tenv statics vars.

(* ======================================================================== *)
(* The environment follows the term                                         *)
(* ======================================================================== *)

(* As in FlambdaTypingEnvRefinementSpec.v, with the values that the
   machine binds in range. *)

Definition increment_scope_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies_strictly_in_range (Typing_env.increment_scope tenv) statics
      vars
    <-> env_satisfies_strictly_in_range tenv statics vars.

Definition add_variable_definition_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (x : variable)
    (kind : flambda_kind) (v : value),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = false ->
    kind_of_value v = kind ->
    value_in_range v = true ->
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly_in_range
      (Typing_env.add_variable_definition tenv x kind Name_mode.normal)
      statics ((x, v) :: vars).

Definition add_definitions_of_params_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars vars' : env)
    (params : list (variable * flambda_kind)) (vs : list value),
    typing_env_invariant tenv ->
    distinct (map fst params) = true ->
    (∀ x : variable,
      x ∈ map fst params -> Typing_env.mem None tenv (Name_var x) = false) ->
    Forall2 (fun param v => kind_of_value v = snd param) params vs ->
    (∀ v : value, v ∈ vs -> value_in_range v = true) ->
    bind_params (map fst params) vs vars = Some vars' ->
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly_in_range
      (Typing_env.add_definitions_of_params tenv params) statics vars'.

Definition replace_equation_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env) (x : variable)
    (ty : flambda_type),
    in_scope_in_terms tenv x ->
    (∀ y : variable, y ∈ free_variables ty -> in_scope_in_terms tenv y) ->
    (∀ v : value, lookup x vars = Some v -> type_satisfied statics vars v ty)
    ->
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly_in_range
      (Typing_env.replace_equation tenv (Name_var x) ty) statics vars.

Definition add_alias_satisfies_in_range_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics : static_env) (vars : env)
    (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    (∀ x : variable,
      canonical_element1 = Var x ∨ canonical_element2 = Var x ->
      in_scope_in_terms tenv x) ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    eval_simple vars canonical_element1 = eval_simple vars canonical_element2
    ->
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly_in_range (Typing_env.add_alias_result_t result)
      statics vars.

(* The static environment grows as a program runs. *)
Definition satisfies_in_range_statics_extend_statement : Prop :=
  ∀ (tenv : Typing_env.t) (statics statics' : static_env) (vars : env),
    statics_extend statics statics' ->
    env_satisfies_strictly_in_range tenv statics vars ->
    env_satisfies_strictly_in_range tenv statics' vars.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The variables are those of FlambdaTypingEnvExamplesSpec.v. [p]
   exists only in types and is, by its type, the tagged immediate of a
   number that no immediate of the machine is. [i] may occur in terms
   and is [Is_int] of [p]. (The type of [p] is written with
   [tag_immediate], so that it is not an alias: the environment has
   its invariant.) *)
Definition example_big_number : Z := 2 ^ 70.

Definition example_out_of_range_env : Typing_env.t :=
  let tenv := Typing_env.create in
  let tenv :=
    Typing_env.add_variable_definition tenv example_p Value
      Name_mode.in_types
  in
  let tenv :=
    Typing_env.replace_equation tenv (Name_var example_p)
      (tag_immediate (these_naked_immediates [example_big_number]))
  in
  let tenv :=
    Typing_env.add_variable_definition tenv example_i
      (Naked_number Kind_naked_immediate) Name_mode.normal
  in
  Typing_env.replace_equation tenv (Name_var example_i)
    (is_int_for_scrutinee (Name_var example_p)).

Definition example_out_of_range_vars : env :=
  [(example_i, Value_const (Naked_immediate 1))].

(* The environment has its invariant, and its types are well formed.
   The machine that binds [i] to 1 is in range and satisfies the
   environment strictly, through the valuation that gives [p] that
   number. A valuation that satisfies the environment and agrees with
   the machine gives [p] a value, since [i] is the result of a
   relation on it, and so that number: none is in range. *)
Definition example_out_of_range_statement : Prop :=
  typing_env_invariant example_out_of_range_env
  ∧ types_well_formed example_out_of_range_env
  ∧ env_satisfies_strictly example_out_of_range_env empty_static_env
      example_out_of_range_vars
  ∧ env_in_range example_out_of_range_vars
  ∧ ~ env_satisfies_strictly_in_range example_out_of_range_env
        empty_static_env example_out_of_range_vars.
