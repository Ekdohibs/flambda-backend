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



(* The environment that a meet works in, what a meet answers, and the
   functions that add an equation to a typing environment, as the model
   sees them.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/env/meet_env.ml (with meet_env.mli). It
   holds definitions only and it extracts. What each function
   guarantees is FlambdaMeetSpec.v.

   What the model has of [Meet_env], in the order of the file:
   - [meet_return_value] and [meet_result]: in full. The constructors of
     [meet_result] are in a module, [Meet_result.Bottom] and
     [Meet_result.Ok], as those of [Or_bottom.t] are in
     FlambdaTypeGrammar.v. [map_result]: in full.
   - [meet_expanded_head]: the type of the function that meets two
     expanded types, which every function here that may meet is given
     as its last argument ([~meet_expanded_head] in the compiler). It
     answers an [option]: [None] is a fatal error of the compiler, a
     meet that was not given enough fuel (FlambdaMeetSpec.v,
     "Termination"), or, in one place, a state at which the model
     stops and the compiler does not (below, "A name that is no longer
     canonical").
   - [t]: [typing_env] and [adding_equations_for_names] in full, the
     latter as a list of names. [create], [with_typing_env],
     [map_typing_env], [current_typing_env], [make_bottom]: in full.
   - [replace_concrete_equation],
     [add_concrete_equation_on_canonical_name],
     [adding_equation_for_name],
     [add_or_delay_concrete_equation_on_canonical_name],
     [add_concrete_equation_on_const],
     [add_or_delay_concrete_equation_on_canonical],
     [add_concrete_equation_on_simple], [record_demotion],
     [add_alias_between_canonicals]: in full, for the default strategy,
     but for one test that the model adds in
     [replace_concrete_equation] (below, "A name that is no longer
     canonical").
   - [add_equation_on_simple], [add_equation], [add_env_extension] and
     [add_alias]: the compiler defines each twice, first with an
     argument [~raise_on_bottom] and then, under the same name, the
     function that its interface shows. The first of each is here with
     a [0] after its name ([add_equation0], ...).
   - [add_equation_strict], [add_env_extension_strict],
     [add_env_extension_maybe_bottom], [add_equations_on_params],
     [enter_scope], [add_definition], [add_variable_definition],
     [meet], [meet_type], [final_typing_env],
     [final_typing_env_strict], [use_meet_env],
     [use_meet_env_strict]: in full.
   Left out: [add_env_extension_with_extra_variables] and
   [add_env_extension_from_level] (the first needs extensions with
   extra variables, which FlambdaTypingEnv.v has not yet; the second is
   for the binary join); [add_symbol_projection] (symbol projections
   are not in the model's environment); [code_age_relation],
   [code_age_relation_resolver] and [machine_width] (neither is in the
   model's environment).

   The differences:
   - Strategies. [delayed_equations] is left out, with
     [delay_equation_on_simple] and the loop of [final_typing_env]. The
     compiler has three strategies for a meet ([meet_strategy]: [Dfs],
     [Bfs], [Hybrid]), chosen by an option for debugging; the default,
     [Dfs], never delays an equation, and it is the one modelled. With
     it [final_typing_env] is [current_typing_env] and
     [final_typing_env_strict] never answers bottom.
   - Exceptions. The compiler raises [Bottom_equation] where an
     equation has no value, if [raise_on_bottom] is set, and the
     functions of the interface catch it. Here a function that may
     raise it answers an [Or_bottom.t], [Bottom] being the exception,
     and a function given more than one equation stops at the first
     [Bottom].
   - A fatal error of the compiler is [None]: a kind that is not the
     one expected, a name that is not bound ([Typing_env.find]), a type
     that is an alias where a concrete one is expected, an alias to a
     constant that is refused, a constant that is demoted, lists of
     different lengths in [add_equations_on_params]. What the compiler
     checks only when a flag asks for invariants is not checked
     ([record_demotion], [check_params_and_types]). Not every [None]
     is a fatal error of the compiler: the next item but one has a
     [None] of the model's own.
   - Coercions. A [Simple] has none (FlambdaModel.v), so the cases for
     a coercion that is not the identity are not there.
   - A name that is no longer canonical. The model is stricter than
     the compiler here (decision D32).
     [add_concrete_equation_on_canonical_name] is given a name that is
     canonical, meets the type to add with the type of the name, and
     then gives the name the type answered
     ([replace_concrete_equation]), taking the name to be canonical
     still. Nothing in the compiler says that it is: the meet in
     between may have made the name an alias of an earlier one, if the
     types that it met lead back to the name. Where the type answered
     has one constant for only value, the compiler then calls
     [Typing_env.add_alias] on a name that is not canonical, which
     [Aliases.add] forbids and does not check. The model answers
     [None] there. Where the type answered is not such a type the
     model does as the compiler does: the name, which is an alias in
     the aliases, gets a type that is not an alias. Discrepancy T4-10.
     This [None] is no fatal error of the compiler and no lack of
     fuel: on an input that reaches it the compiler answers and the
     model does not. No choice of [meet_choices] (FlambdaMeet.v)
     governs it, so the meet with the compiler's choices has it too.
     There are such inputs among types built by hand with the
     compiler's library of types; whether Simplify reaches one is not
     known. A comparison of the model with the compiler is to account
     for such an input, and a statement that the model answers what
     the compiler answers is to exclude it.
   - [add_alias] reads the kind of a [Simple] from the environment
     ([kind_of_simple], FlambdaMeetGrammar.v): a variable of the model
     does not carry its kind.
   - The equations of an extension are added in the order of the list
     that the extension is, a name with two equations having the first.
     The compiler adds them in the order of the names
     ([Name.Map.fold]), which the model may not look at. The order can
     show in the answer (which fatal error or bottom comes first, what
     a later meet reads, the order of the bindings answered), and an
     extension that the model makes from a level is in the order in
     which the level was built: FlambdaMeet.v, "The order of what is
     gone through", says what is and is not claimed.
   - [names_to_avoid] is not in the compiler. A variable that the
     compiler makes ([Variable.create], in types/env/join_env.ml; meet
     itself makes none) is different from every variable made before.
     The model has no such supply: the join is to make variables that
     are not in this list and that no environment it is given knows,
     and to add them to it, and meet hands the list on from one branch
     of a disjunction to the next, so that two branches do not make the
     same variable. Nothing reads the list but to avoid its names; no
     statement depends on which names are made. Every function here
     keeps the list of the environment it is given.
   - The record [t] is defined in a module of its own, [Meet_env_type],
     which [Meet_env] includes: [meet_result] mentions the record, the
     functions mention [meet_result], and a module of Rocq cannot be
     reopened. [Meet_env.t] is that record. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases FlambdaTypingEnv FlambdaMeetGrammar FlambdaExpandHead.
Import ListNotations.

(* [Meet_env.meet_return_value]: the result of a meet is one of its
   inputs, either of them, or something else. *)
Inductive meet_return_value (A : Type) : Type :=
  | Left_input
  | Right_input
  | Both_inputs
  | New_result (a : A).

Arguments Left_input {A}.
Arguments Right_input {A}.
Arguments Both_inputs {A}.
Arguments New_result {A} a.

(* [Meet_and_n_way_join.extract_value]. *)
Definition extract_value {A : Type} (res : meet_return_value A)
    (left right : A) : A :=
  match res with
  | Left_input => left
  | Right_input => right
  | Both_inputs => left
  | New_result value => value
  end.

(* [Meet_env.t]. *)
Module Meet_env_type.

  Record t : Type := {
    typing_env : Typing_env.t;
    adding_equations_for_names : list name;
    names_to_avoid : list variable
  }.

End Meet_env_type.

(* [Meet_env.meet_result]. *)
Module Meet_result.
  Inductive t (A : Type) : Type :=
    | Bottom (r : meet_return_value unit)
    | Ok (r : meet_return_value A) (env : Meet_env_type.t).
  Arguments Bottom {A} r.
  Arguments Ok {A} r env.
End Meet_result.

(* [Meet_env.meet_expanded_head]. *)
Definition meet_expanded_head : Type :=
  Meet_env_type.t -> Expanded_type.t -> Expanded_type.t ->
  option (Meet_result.t Expanded_type.t).

(* [Meet_env.map_result]. *)
Definition map_result {A B : Type} (f : A -> B) (result : Meet_result.t A)
    : Meet_result.t B :=
  match result with
  | Meet_result.Bottom r => Meet_result.Bottom r
  | Meet_result.Ok Left_input env => Meet_result.Ok Left_input env
  | Meet_result.Ok Right_input env => Meet_result.Ok Right_input env
  | Meet_result.Ok Both_inputs env => Meet_result.Ok Both_inputs env
  | Meet_result.Ok (New_result x) env => Meet_result.Ok (New_result (f x)) env
  end.

Module Meet_env.

  Include Meet_env_type.

  (* [Meet_env.create]. *)
  Definition create (typing_env : Typing_env.t) : t :=
    {| typing_env := typing_env;
       adding_equations_for_names := [];
       names_to_avoid := [] |}.

  Definition with_typing_env (env : t) (typing_env : Typing_env.t) : t :=
    {| typing_env := typing_env;
       adding_equations_for_names := adding_equations_for_names env;
       names_to_avoid := names_to_avoid env |}.

  Definition map_typing_env (env : t) (f : Typing_env.t -> Typing_env.t) : t :=
    with_typing_env env (f (typing_env env)).

  (* The environment with the names to avoid of another: not in the
     compiler (see the header). *)
  Definition with_names_to_avoid (env : t) (names : list variable) : t :=
    {| typing_env := typing_env env;
       adding_equations_for_names := adding_equations_for_names env;
       names_to_avoid := names |}.

  (* [Meet_env.current_typing_env]. *)
  Definition current_typing_env (env : t) : Typing_env.t := typing_env env.

  (* [Meet_env.make_bottom]. *)
  Definition make_bottom (env : t) : t :=
    map_typing_env env Typing_env.make_bottom.

  (* [replace_concrete_equation]: the type of a name becomes a type that
     is not an alias; if that type has one constant for only value, the
     name becomes an alias of the constant instead. The model stops
     ([None]) where the name is to become an alias of a constant and is
     not canonical: see "The differences" in the header. *)
  (* CR mshinwell: the model stops where the compiler goes on with a
     name that is no longer canonical. Whether the meet in between can
     demote the name, in an environment that Simplify builds, is not
     known. If it cannot, what excludes it is to be a condition on
     environments, and this test goes. Discrepancy T4-10. *)
  Definition replace_concrete_equation (tenv : Typing_env.t) (name : name)
      (ty : flambda_type) : option Typing_env.t :=
    let* singleton := must_be_singleton ty in
    match singleton with
    | None =>
      match get_alias_opt ty with
      | None => Some (Typing_env.replace_equation tenv name ty)
      | Some _ => None
      end
    | Some const =>
      match
        Aliases.canonical (Typing_env.aliases tenv) (simple_of_name name)
      with
      | Aliases.Alias_of_canonical _ => None
      | Aliases.Is_canonical =>
        let* result :=
          Typing_env.add_alias tenv (simple_of_name name) (Const const)
        in
        match result with
        | Bottom | Unknown => None
        | Ok result =>
          if name_eqb (Typing_env.demoted_name result) name
             && simple_equal (Typing_env.canonical_element result)
                  (Const const)
          then
            Some
              (Typing_env.replace_equation
                 (Typing_env.add_alias_result_t result)
                 (Typing_env.demoted_name result)
                 (alias_type_of (kind_of_const const)
                    (Typing_env.canonical_element result)))
          else None
        end
      end
    end.

  (* [add_concrete_equation_on_canonical_name]: the type of a canonical
     name becomes the meet of the type it has and of [ty]. [Bottom] is
     the exception [Bottom_equation]. *)
  Definition add_concrete_equation_on_canonical_name (raise_on_bottom : bool)
      (env : t) (name : name) (ty : Expanded_type.t)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    let kind := Expanded_type.kind ty in
    let* existing_ty_of_name :=
      Typing_env.find (typing_env env) name (Some kind)
    in
    let* existing_ty := Expanded_type.of_non_alias_type existing_ty_of_name in
    let* result := meet_expanded_head env ty existing_ty in
    match result with
    | Meet_result.Bottom _ =>
      if raise_on_bottom
      then Some Or_bottom.Bottom
      else
        Some
          (Or_bottom.Ok
             (map_typing_env env
                (fun tenv =>
                   Typing_env.replace_equation tenv name (bottom kind))))
    | Meet_result.Ok (Right_input | Both_inputs) env => Some (Or_bottom.Ok env)
    | Meet_result.Ok Left_input env =>
      let* tenv :=
        replace_concrete_equation (typing_env env) name
          (Expanded_type.to_type ty)
      in
      Some (Or_bottom.Ok (with_typing_env env tenv))
    | Meet_result.Ok (New_result ty') env =>
      let* tenv :=
        replace_concrete_equation (typing_env env) name
          (Expanded_type.to_type ty')
      in
      Some (Or_bottom.Ok (with_typing_env env tenv))
    end.

  (* [adding_equation_for_name]: [f] runs with the name among those that
     an equation is being added to, and the names are then as they
     were. *)
  Definition adding_equation_for_name (env : t) (name : name)
      (f : t -> option (Or_bottom.t t)) : option (Or_bottom.t t) :=
    let adding_equations_for_names := adding_equations_for_names env in
    let env :=
      {| typing_env := typing_env env;
         adding_equations_for_names := name :: adding_equations_for_names;
         names_to_avoid := names_to_avoid env |}
    in
    let* result := f env in
    match result with
    | Or_bottom.Bottom => Some Or_bottom.Bottom
    | Or_bottom.Ok env =>
      Some
        (Or_bottom.Ok
           {| typing_env := typing_env env;
              adding_equations_for_names := adding_equations_for_names;
              names_to_avoid := names_to_avoid env |})
    end.

  (* [add_or_delay_concrete_equation_on_canonical_name], for the
     strategy [Dfs]: an equation on a name that an equation is being
     added to is ignored. *)
  Definition add_or_delay_concrete_equation_on_canonical_name
      (raise_on_bottom : bool) (env : t) (name : name) (ty : Expanded_type.t)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    if Keyed.among name_eqb name (adding_equations_for_names env)
    then Some (Or_bottom.Ok env)
    else
      adding_equation_for_name env name
        (fun env =>
           add_concrete_equation_on_canonical_name raise_on_bottom env name ty
             meet_expanded_head).

  (* [add_concrete_equation_on_const]: a constant has no type to record;
     its type is met with [ty] to see that the two have a value in
     common. *)
  Definition add_concrete_equation_on_const (raise_on_bottom : bool) (env : t)
      (const : const) (ty : Expanded_type.t)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    let* result :=
      meet_expanded_head env ty (Expanded_type.create_const const)
    in
    match result with
    | Meet_result.Ok _ env => Some (Or_bottom.Ok env)
    | Meet_result.Bottom _ =>
      if raise_on_bottom then Some Or_bottom.Bottom else Some (Or_bottom.Ok env)
    end.

  (* [add_or_delay_concrete_equation_on_canonical]. *)
  Definition add_or_delay_concrete_equation_on_canonical
      (raise_on_bottom : bool) (env : t) (simple : simple)
      (ty : Expanded_type.t) (meet_expanded_head : meet_expanded_head)
      : option (Or_bottom.t t) :=
    match simple with
    | Const const =>
      add_concrete_equation_on_const raise_on_bottom env const ty
        meet_expanded_head
    | Var x =>
      add_or_delay_concrete_equation_on_canonical_name raise_on_bottom env
        (Name_var x) ty meet_expanded_head
    | Symbol s =>
      add_or_delay_concrete_equation_on_canonical_name raise_on_bottom env
        (Name_symbol s) ty meet_expanded_head
    end.

  (* [add_concrete_equation_on_simple], for the strategy [Dfs]. *)
  Definition add_concrete_equation_on_simple (raise_on_bottom : bool)
      (env : t) (simple : simple) (ty : Expanded_type.t)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    let canonical :=
      Typing_env.get_canonical_simple_ignoring_name_mode (typing_env env)
        simple
    in
    add_or_delay_concrete_equation_on_canonical raise_on_bottom env canonical
      ty meet_expanded_head.

  (* [record_demotion]: [demoted], which was canonical, is now an alias
     of [canonical]. Its type becomes that alias, and the type that it
     had is added to [canonical]. *)
  Definition record_demotion (raise_on_bottom : bool) (env : t)
      (kind : flambda_kind) (demoted : name) (canonical : simple)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    let* ty_of_demoted :=
      Typing_env.find (typing_env env) demoted (Some kind)
    in
    let env :=
      map_typing_env env
        (fun tenv =>
           Typing_env.replace_equation tenv demoted
             (alias_type_of kind canonical))
    in
    let* ty_of_demoted :=
      expand_head0 (typing_env env) ty_of_demoted
        (Some (simple_of_name demoted))
    in
    add_or_delay_concrete_equation_on_canonical raise_on_bottom env canonical
      ty_of_demoted meet_expanded_head.

  (* [add_alias_between_canonicals]: two canonical elements become
     aliases. *)
  Definition add_alias_between_canonicals (raise_on_bottom : bool) (env : t)
      (kind : flambda_kind) (canonical_element1 canonical_element2 : simple)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    if simple_equal canonical_element1 canonical_element2
    then Some (Or_bottom.Ok env)
    else
      let* result :=
        Typing_env.add_alias (typing_env env) canonical_element1
          canonical_element2
      in
      match result with
      | Bottom =>
        if raise_on_bottom
        then Some Or_bottom.Bottom
        else Some (Or_bottom.Ok env)
      | Unknown => Some (Or_bottom.Ok env)
      | Ok result =>
        record_demotion raise_on_bottom
          (with_typing_env env (Typing_env.add_alias_result_t result))
          kind
          (Typing_env.demoted_name result)
          (Typing_env.canonical_element result)
          meet_expanded_head
      end.

  (* [add_equation_on_simple ~raise_on_bottom]: a [Simple] is given a
     type. A type that is not an alias goes to the canonical element of
     the [Simple]; an alias makes two canonical elements aliases. *)
  Definition add_equation_on_simple0 (raise_on_bottom : bool) (env : t)
      (simple : simple) (ty : flambda_type)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    match get_alias_opt ty with
    | None =>
      let* ty := Expanded_type.of_non_alias_type ty in
      add_concrete_equation_on_simple raise_on_bottom env simple ty
        meet_expanded_head
    | Some alias =>
      let canonical_element1 :=
        Typing_env.get_canonical_simple_ignoring_name_mode (typing_env env)
          simple
      in
      let canonical_element2 :=
        Typing_env.get_canonical_simple_ignoring_name_mode (typing_env env)
          alias
      in
      add_alias_between_canonicals raise_on_bottom env (kind ty)
        canonical_element1 canonical_element2 meet_expanded_head
    end.

  (* [add_equation ~raise_on_bottom]. *)
  Definition add_equation0 (raise_on_bottom : bool) (env : t) (name : name)
      (ty : flambda_type) (meet_expanded_head : meet_expanded_head)
      : option (Or_bottom.t t) :=
    add_equation_on_simple0 raise_on_bottom env (simple_of_name name) ty
      meet_expanded_head.

  (* The equations of a list are added one after the other, until one
     has no value. *)
  Fixpoint add_equations0 (raise_on_bottom : bool) (env : t)
      (equations : list (name * flambda_type))
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    match equations with
    | [] => Some (Or_bottom.Ok env)
    | (name, ty) :: equations =>
      let* result :=
        add_equation0 raise_on_bottom env name ty meet_expanded_head
      in
      match result with
      | Or_bottom.Bottom => Some Or_bottom.Bottom
      | Or_bottom.Ok env =>
        add_equations0 raise_on_bottom env equations meet_expanded_head
      end
    end.

  (* [add_env_extension ~raise_on_bottom]. *)
  Definition add_env_extension0 (raise_on_bottom : bool) (env : t)
      (env_extension : env_extension) (meet_expanded_head : meet_expanded_head)
      : option (Or_bottom.t t) :=
    match env_extension with
    | Env_extension equations =>
      add_equations0 raise_on_bottom env (Keyed.compact name_eqb equations)
        meet_expanded_head
    end.

  (* [Meet_env.add_equation_strict]. *)
  Definition add_equation_strict (env : t) (name : name) (ty : flambda_type)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    if Typing_env.is_bottom (typing_env env)
    then Some Or_bottom.Bottom
    else add_equation0 true env name ty meet_expanded_head.

  (* [Meet_env.add_env_extension_strict]. *)
  Definition add_env_extension_strict (env : t) (env_extension : env_extension)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    if Typing_env.is_bottom (typing_env env)
    then Some Or_bottom.Bottom
    else add_env_extension0 true env env_extension meet_expanded_head.

  (* Where the compiler catches [Bottom_equation] and makes the
     environment bottom. *)
  Definition bottom_on_exception (env : t) (result : option (Or_bottom.t t))
      : option t :=
    let* result := result in
    match result with
    | Or_bottom.Ok env => Some env
    | Or_bottom.Bottom => Some (make_bottom env)
    end.

  (* [Meet_env.add_env_extension_maybe_bottom]. With [raise_on_bottom]
     unset no exception is raised. *)
  Definition add_env_extension_maybe_bottom (env : t)
      (env_extension : env_extension) (meet_expanded_head : meet_expanded_head)
      : option t :=
    let* result :=
      add_env_extension0 false env env_extension meet_expanded_head
    in
    match result with
    | Or_bottom.Ok env => Some env
    | Or_bottom.Bottom => None
    end.

  (* [Meet_env.add_equation]. *)
  Definition add_equation (env : t) (name : name) (ty : flambda_type)
      (meet_expanded_head : meet_expanded_head) : option t :=
    bottom_on_exception env
      (add_equation0 true env name ty meet_expanded_head).

  (* [Meet_env.add_equation_on_simple]. *)
  Definition add_equation_on_simple (env : t) (simple : simple)
      (ty : flambda_type) (meet_expanded_head : meet_expanded_head)
      : option t :=
    bottom_on_exception env
      (add_equation_on_simple0 true env simple ty meet_expanded_head).

  (* [Meet_env.add_env_extension]. *)
  Definition add_env_extension (env : t) (env_extension : env_extension)
      (meet_expanded_head : meet_expanded_head) : option t :=
    bottom_on_exception env
      (add_env_extension0 true env env_extension meet_expanded_head).

  (* [Meet_env.add_equations_on_params]: each parameter is given its
     type. [None] if there are not as many types as parameters. *)
  Fixpoint add_equations_on_params (env : t)
      (params : list (variable * flambda_kind))
      (param_types : list flambda_type)
      (meet_expanded_head : meet_expanded_head) : option t :=
    match params, param_types with
    | [], [] => Some env
    | param :: params, param_type :: param_types =>
      let* env :=
        add_equation env (Name_var (fst param)) param_type meet_expanded_head
      in
      add_equations_on_params env params param_types meet_expanded_head
    | [], _ :: _ | _ :: _, [] => None
    end.

  (* [Meet_env.enter_scope]: the scope to cut after, the typing
     environment as it is, and the environment one scope further. *)
  Definition enter_scope (env : t) : Scope.t * Typing_env.t * t :=
    let tenv := typing_env env in
    let current_scope := Typing_env.current_scope tenv in
    let env :=
      {| typing_env := Typing_env.increment_scope tenv;
         adding_equations_for_names := adding_equations_for_names env;
         names_to_avoid := names_to_avoid env |}
    in
    (current_scope, tenv, env).

  (* [Meet_env.add_definition], of a [Bound_name.t] given as its name and
     its name mode. *)
  Definition add_definition (env : t) (name : name) (name_mode : Name_mode.t)
      (kind : flambda_kind) : option t :=
    let* tenv :=
      Typing_env.add_definition (typing_env env) name name_mode kind
    in
    Some (with_typing_env env tenv).

  (* [Meet_env.add_variable_definition]. *)
  Definition add_variable_definition (env : t) (var : variable)
      (kind : flambda_kind) (name_mode : Name_mode.t) : t :=
    map_typing_env env
      (fun tenv => Typing_env.add_variable_definition tenv var kind name_mode).

  (* [add_alias ~raise_on_bottom]. *)
  Definition add_alias0 (raise_on_bottom : bool) (env : t)
      (simple1 simple2 : simple) (meet_expanded_head : meet_expanded_head)
      : option (Or_bottom.t t) :=
    let canonical1 :=
      Typing_env.get_canonical_simple_ignoring_name_mode (typing_env env)
        simple1
    in
    let canonical2 :=
      Typing_env.get_canonical_simple_ignoring_name_mode (typing_env env)
        simple2
    in
    let* kind := kind_of_simple (typing_env env) canonical1 in
    add_alias_between_canonicals raise_on_bottom env kind canonical1
      canonical2 meet_expanded_head.

  (* [Meet_env.add_alias]. *)
  Definition add_alias (env : t) (simple1 simple2 : simple)
      (meet_expanded_head : meet_expanded_head) : option (Or_bottom.t t) :=
    add_alias0 true env simple1 simple2 meet_expanded_head.

  (* [Meet_env.meet]: the meet of two types, of which either may be an
     alias. The meet of a type with an alias is the alias, with an
     equation added on what it is an alias to; the meet of two aliases
     is either, the two being made aliases. *)
  Definition meet (env : t) (t1 t2 : flambda_type)
      (meet_expanded_head : meet_expanded_head)
      : option (Meet_result.t Expanded_type.t) :=
    if negb (kind_eqb (kind t1) (kind t2))
    then None
    else
      let kind := kind t1 in
      let tenv := typing_env env in
      let* simple1 :=
        Typing_env.get_alias_then_canonical_simple_exn tenv
          (Some Name_mode.in_types) None t1
      in
      let* simple2 :=
        Typing_env.get_alias_then_canonical_simple_exn tenv
          (Some Name_mode.in_types) None t2
      in
      let of_exception (result : Or_bottom.t t)
          (return_value : meet_return_value Expanded_type.t)
          : Meet_result.t Expanded_type.t :=
        match result with
        | Or_bottom.Bottom => Meet_result.Bottom (New_result tt)
        | Or_bottom.Ok env => Meet_result.Ok return_value env
        end
      in
      match simple1 with
      | None =>
        let* expanded1 := expand_head0 tenv t1 simple1 in
        match simple2 with
        | None =>
          let* expanded2 := expand_head0 tenv t2 simple2 in
          meet_expanded_head env expanded1 expanded2
        | Some simple2 =>
          let* result :=
            add_or_delay_concrete_equation_on_canonical true env simple2
              expanded1 meet_expanded_head
          in
          Some (of_exception result Right_input)
        end
      | Some simple1 =>
        match simple2 with
        | None =>
          let* expanded2 := expand_head0 tenv t2 simple2 in
          let* result :=
            add_or_delay_concrete_equation_on_canonical true env simple1
              expanded2 meet_expanded_head
          in
          Some (of_exception result Left_input)
        | Some simple2 =>
          let* result :=
            add_alias_between_canonicals true env kind simple1 simple2
              meet_expanded_head
          in
          Some (of_exception result Both_inputs)
        end
      end.

  (* [Meet_env.meet_type]. *)
  Definition meet_type (env : t) (t1 t2 : flambda_type)
      (meet_expanded_head : meet_expanded_head)
      : option (Meet_result.t flambda_type) :=
    option_map (map_result Expanded_type.to_type)
      (meet env t1 t2 meet_expanded_head).

  (* [Meet_env.final_typing_env] and [final_typing_env_strict], for the
     strategy [Dfs]: no equation is delayed. *)
  Definition final_typing_env (env : t) : Typing_env.t := typing_env env.

  Definition final_typing_env_strict (env : t) : Or_bottom.t Typing_env.t :=
    Or_bottom.Ok (typing_env env).

  (* [Meet_env.use_meet_env]. *)
  Definition use_meet_env (tenv : Typing_env.t) (f : t -> option t)
      : option Typing_env.t :=
    option_map final_typing_env (f (create tenv)).

  (* [Meet_env.use_meet_env_strict]. *)
  Definition use_meet_env_strict (tenv : Typing_env.t) (f : t -> option t)
      : option (Or_bottom.t Typing_env.t) :=
    if Typing_env.is_bottom tenv
    then Some Or_bottom.Bottom
    else
      let* env := f (create tenv) in
      let tenv := final_typing_env env in
      if Typing_env.is_bottom tenv
      then Some Or_bottom.Bottom
      else Some (Or_bottom.Ok tenv).

End Meet_env.
