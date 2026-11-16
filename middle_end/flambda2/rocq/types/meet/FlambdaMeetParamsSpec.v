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


(* The equations on the parameters of a continuation, for parameters
   that the environment does not know.

   This file is to be reviewed: three definitions, an example and
   eight statements. The proofs are in FlambdaMeetParamsProofs.v, which
   needs no review.

   The question. The contract agreed for
   [Typing_env.add_equations_on_params]
   ([add_equations_on_params_sound], FlambdaMeetSpec.v) does not
   suppose that the environment knows the parameters. What is proved
   of the function of the model supposes it
   ([add_equations_on_params_sound_where_known],
   FlambdaMeetFuelSpec.v). Is the contract true without the premise?

   It is not. The equations are added one parameter after the other.
   The meet for one parameter may call the join, and a join that has
   its contract ([join_scoped_envs_sound]) may make a variable: any
   variable that the environments it is given do not know and that
   is not among the names that the meet is to avoid. A later
   parameter that the environment does not know is such a variable.
   If the join makes it and gives it a type, the equation of that
   parameter is then met with that type, and the value that the
   parameter has need not have it. [naming_join] is a join that
   makes one variable of a given name, a naked immediate that is 5
   or 6; it has the contract ([naming_join_sound_statement]). With
   it, on an environment that knows the first of two parameters and
   not the second, where the second holds 7 and is given the type of
   7 and 8, the function answers the bottom environment
   ([example_params_answer_statement]), though every premise of the
   contract holds ([example_params_premises_statement]); so it has
   not the contract ([add_equations_on_params_not_sound_statement]).

   This is of the model and not of the compiler. A variable that the
   compiler's join makes ([Variable.create]) differs from every
   variable that exists, parameters included, whether or not the
   environment knows them. The model's join has no such supply: it
   avoids the names that it is given to avoid and those that the
   environments know (FlambdaMeetEnv.v, [names_to_avoid]), and the
   function starts with no name to avoid.

   The premise under which the contract is true. Every parameter
   that the environment does not know is among the names that the
   meet is to avoid
   ([meet_env_add_equations_on_params_avoided_statement], for the
   function of [Meet_env], which takes those names in its
   environment). A parameter that is not known and is to be avoided
   stays unknown, since a variable that a meet defines is not among
   the names to avoid ([new_names_avoided], FlambdaMeetSpec.v), and
   an equation on a variable that is not known adds nothing: where
   the function answers, the environment is as it was
   ([add_equation_on_unknown_statement]). It need not answer: the
   compiler stops with a fatal error on a name that is not bound.
   Two instances:
   - the function as transcribed starts with no name to avoid, so
     the premise is that every parameter is known: it is the
     statement that was proved;
   - a function that starts with the parameters as the names to
     avoid ([add_equations_on_params_avoiding]) has the contract as
     agreed, for a join that has its own
     ([add_equations_on_params_avoiding_sound_statement]). This
     function is not the transcription: it is here to show what
     would make the contract true. On the example it does not
     answer ([example_params_avoiding_answer_statement]); that is
     one input, and does not show that it never answers where a
     parameter is not known. What is shown in general is the
     statement above: of such a parameter the environment answered
     says nothing.

   A premise on the join alone ("the join makes no variable that is
   among the parameters") is not stated: a variable becomes known to
   the environment of the meet through the meet's own definitions
   after a join too, and what the contract of the meet says of those
   is said with the names to avoid.

   The join of the model. The variable that it makes is one greater
   than every name to avoid and every variable that the environments
   it is given know ([fresh_name], FlambdaJoinMeet.v). A parameter
   that the environment does not know can be that one, so the join
   of the model does not by itself keep off the parameters. With the
   parameters among the names to avoid it does: that a variable made
   is not among those names is in the half of the join's contract
   that is not about values, which the join of the model has
   ([model_join_scoped_envs_keeps_outright_statement],
   FlambdaJoinMeetFitsSpec.v). The last statement here supposes the
   whole contract of the join, as every statement about values
   does. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaClosuresSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvKindedSpec FlambdaMeetGrammar
  FlambdaExpandHead FlambdaMeetEnv FlambdaMeet FlambdaMeetSpec
  FlambdaMeetEnvSpec FlambdaMeetImmediateSpec FlambdaMeetFuelSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* A join that makes a variable of a given name                             *)
(* ======================================================================== *)

(* The type that the join gives the variable: the naked immediates 5
   and 6. *)
Definition naming_join_type : flambda_type := these_naked_immediates [5; 6].

(* The join that forgets the environments joined, as
   [forgetful_join] (FlambdaMeetExamplesSpec.v) does, and makes the
   variable [q], with that type, if no environment that it is given
   knows [q] and [q] is not to be avoided. *)
Definition naming_join (q : variable) : join_scoped_envs :=
  fun _meet_expanded_head _cut_after initial_env _initial_tenv scoped_envs =>
    let tenv := Meet_env.typing_env initial_env in
    if Typing_env.mem None tenv (Name_var q)
       || existsb
            (fun scoped_env => Typing_env.mem None scoped_env (Name_var q))
            scoped_envs
       || among q (Meet_env.names_to_avoid initial_env)
    then Some initial_env
    else
      let tenv :=
        Typing_env.add_variable_definition tenv q
          (Naked_number Kind_naked_immediate) Name_mode.in_types
      in
      let tenv :=
        Typing_env.replace_equation tenv (Name_var q) naming_join_type
      in
      Some
        (Meet_env.with_names_to_avoid
           (Meet_env.with_typing_env initial_env tenv)
           (q :: Meet_env.names_to_avoid initial_env)).

(* It has the contract of the join that the meet is given. *)
Definition naming_join_sound_statement : Prop :=
  ∀ q : variable, join_scoped_envs_sound (naming_join q).

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Definition example_known_param : variable := 1%positive.
Definition example_unknown_param : variable := 2%positive.

(* A value that is not null: an immediate or a block, of which no
   more is known. *)
Definition example_any_variant : flambda_type :=
  Type_value
    (Ok
       (No_alias
          (Head_of_kind_value
             (Ok
                (Variant None Or_unknown.Unknown None Or_unknown.Unknown
                   No_extensions false))
             Not_null))).

(* The environment knows the first parameter, with that type. It does
   not know the second. *)
Definition example_params_env : Typing_env.t :=
  Typing_env.replace_equation
    (Typing_env.add_variable_definition Typing_env.create example_known_param
       Value Name_mode.normal)
    (Name_var example_known_param) example_any_variant.

Definition example_params : list (variable * flambda_kind) :=
  [(example_known_param, Value);
   (example_unknown_param, Naked_number Kind_naked_immediate)].

(* The types given for the parameters: an immediate or a block of tag
   0; the naked immediates 7 and 8. Meeting the first with what the
   environment has calls the join once, for the two alternatives of
   the variant (an immediate, a block), in the environment of the
   meet: what the join adds there is added to the environment
   answered. *)
Definition example_param_types : list flambda_type :=
  [immediate_or_blocks_with_these_tags [0] Alloc_mode.Heap;
   these_naked_immediates [7; 8]].

(* A run: the first parameter is an immediate, the second is 7. *)
Definition example_params_valuation : env :=
  [(example_known_param, Value_const (Tagged_immediate 3));
   (example_unknown_param, Value_const (Naked_immediate 7))].

(* The function, with the join that makes the second parameter. *)
Definition example_add_equations_on_params
    : add_equations_on_params_function :=
  Flambda2_types.add_equations_on_params (naming_join example_unknown_param)
    repaired_choices 3.

(* Every premise of [add_equations_on_params_sound] holds of the
   example. *)
Definition example_params_premises_statement : Prop :=
  typing_env_fits example_params_env
  ∧ (∀ ty : flambda_type,
      ty ∈ example_param_types -> type_fits example_params_env ty)
  ∧ statics_valid empty_static_env
  ∧ typing_env_satisfied_strictly empty_static_env example_params_valuation
      example_params_env
  ∧ Forall2
      (fun (param : variable * flambda_kind) (ty : flambda_type) =>
         ∃ v : value,
           lookup (fst param) example_params_valuation = Some v
           ∧ type_satisfied empty_static_env example_params_valuation v ty)
      example_params example_param_types.

(* The function answers the bottom environment, which no valuation
   satisfies. *)
Definition example_params_answer_statement : Prop :=
  ∃ tenv' : Typing_env.t,
    example_add_equations_on_params example_params_env example_params
      example_param_types
    = Some tenv'
    ∧ Typing_env.is_bottom tenv' = true.

(* So the function has not the contract as agreed, though the join
   has its own. *)
Definition add_equations_on_params_not_sound_statement : Prop :=
  join_scoped_envs_sound (naming_join example_unknown_param)
  ∧ ~ add_equations_on_params_sound example_add_equations_on_params.

(* ======================================================================== *)
(* The premise under which the contract is true                             *)
(* ======================================================================== *)

(* An equation on a variable that the environment does not know adds
   nothing: where the function answers, the environment is as it
   was. No equation is being added when it is called. *)
Definition add_equation_on_unknown_statement : Prop :=
  ∀ (mh : meet_expanded_head) (menv menv' : Meet_env.t) (x : variable)
    (ty : flambda_type),
    meet_expanded_head_other_kinds mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    type_fits (Meet_env.typing_env menv) ty ->
    Meet_env.adding_equations_for_names menv = [] ->
    Typing_env.mem None (Meet_env.typing_env menv) (Name_var x) = false ->
    Meet_env.add_equation menv (Name_var x) ty mh = Some menv' ->
    menv' = menv.

(* [Meet_env.add_equations_on_params]: what is added is true, if
   every parameter that the environment does not know is among the
   names to avoid. No equation is being added when it is called. *)
Definition meet_env_add_equations_on_params_avoided_statement : Prop :=
  ∀ (mh : meet_expanded_head) (statics : static_env) (valuation : env)
    (menv menv' : Meet_env.t) (params : list (variable * flambda_kind))
    (param_types : list flambda_type),
    meet_expanded_head_keeps mh ->
    meet_expanded_head_sound mh ->
    meet_expanded_head_other_kinds mh ->
    typing_env_fits (Meet_env.typing_env menv) ->
    Meet_env.adding_equations_for_names menv = [] ->
    (∀ ty : flambda_type,
       ty ∈ param_types -> type_fits (Meet_env.typing_env menv) ty) ->
    (∀ param : variable * flambda_kind,
       param ∈ params ->
       Typing_env.mem None (Meet_env.typing_env menv) (Name_var (fst param))
       = false ->
       among (fst param) (Meet_env.names_to_avoid menv) = true) ->
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

(* [Flambda2_types.add_equations_on_params] (FlambdaMeet.v), but that
   the meet is given the parameters as names to avoid. Not the
   transcription. *)
Definition add_equations_on_params_avoiding (join : join_scoped_envs)
    (choices : meet_choices) (fuel : nat)
    : add_equations_on_params_function :=
  fun tenv params param_types =>
    option_map Meet_env.final_typing_env
      (Meet_env.add_equations_on_params
         (Meet_env.with_names_to_avoid (Meet_env.create tenv)
            (map fst params))
         params param_types (Meet.meet_expanded_head join choices fuel)).

(* It has the contract as agreed, made of the meet that is to be
   proved with a join that has its own. *)
Definition add_equations_on_params_avoiding_sound_statement : Prop :=
  ∀ (join : join_scoped_envs) (fuel : nat),
    join_scoped_envs_sound join ->
    add_equations_on_params_sound
      (add_equations_on_params_avoiding join repaired_choices fuel).

(* On the example it does not answer, with the join that makes the
   second parameter (which now may not make it): the second parameter
   is not known when its equation is added. This is one input. *)
Definition example_params_avoiding_answer_statement : Prop :=
  add_equations_on_params_avoiding (naming_join example_unknown_param)
    repaired_choices 3 example_params_env example_params example_param_types
  = None.
