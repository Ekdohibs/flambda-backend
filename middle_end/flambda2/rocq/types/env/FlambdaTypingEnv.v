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


(* Typing environments, as the model sees them: what Simplify knows of
   the names in scope at a point of a term.

   This file is to be reviewed, line by line against
   middle_end/flambda2/types/env/typing_env.ml (with typing_env.mli),
   typing_env_level.ml and cached_level.ml, as FlambdaTypeGrammar.v is
   against type_grammar.ml. It holds definitions only, in modules that
   have the compiler's names, so that [Typing_env.find] is that
   function; and it extracts, for the differential tests to run these
   functions against the compiler's. What it is for the values of a run
   to satisfy a typing environment, and what each function guarantees of
   that, is FlambdaTypingEnvSpec.v. The aliases that an environment
   holds are FlambdaEnvAliases.v; binding times and name modes are
   FlambdaBindingTime.v.

   What an environment is: [Typing_env.t], field by field.
   - [defined_symbols]: in full.
   - [prev_levels] and [current_level]: in full, but for the
     [just_after_level] of each level (next item). A level
     ([One_level.t]) is a scope and what the level adds
     ([Typing_env_level.t]): the variables it defines and the equations
     it adds. The levels are the history of the environment; only [cut]
     reads them.
   - [just_after_level]: the compiler has one in every level. The model
     has the one of the current level, as a field of the environment.
     It is the environment proper ([Cached_level.t]): [names_to_types],
     the type, the binding time and the name mode of every name that
     has them, and [aliases]. Every function of the compiler reads the
     one of the current level, and those of the earlier levels are read
     only to print them. For all its name it is not a cache: it cannot
     be had from the levels, which do not record name modes and which
     drop an equation whose type is unknown where this map keeps the
     type.
   - [next_binding_time], [min_binding_time], [is_bottom]: in full.
   - [machine_width] is left out: the numbers of the model are not
     restricted to a width (FlambdaModel.v).
   - [resolver] and [binding_time_resolver] are left out: the model has
     one compilation unit. No name is imported and no .cmx file is read,
     or missing; so [variable_is_from_missing_cmx_file] is left out,
     and the cases of [find], [mem], [replace_equation] and [add_alias]
     for a name of another unit are not there. [Aliases] asks the
     resolver for the binding time of a name where two names have the
     same one in [names_to_types]: [Typing_env.binding_time_resolver]
     here is the function that reads that map again. In the compiler the
     resolver stops with a fatal error for a name of the current unit;
     no two variables of one environment have the same binding time, so
     it is not asked.
   - [code_age_relation] is left out, with [add_to_code_age_relation]
     and [with_code_age_relation]: a function type of the model names
     its code exactly (FlambdaTypeMeaningSpec.v). It is for the
     workstream on calls (S4) to add.

   [Typing_env_level.t]: [defined_vars], [binding_times] and [equations]
   in full. [symbol_projections] is left out, there and in
   [Cached_level.t], with [add_symbol_projection] and
   [find_symbol_projection]: a symbol projection says that a variable is
   a field of a symbol; lifting reads it (simplify/lifting), and no
   function here does. It is for the workstream on lifting (S5) to add,
   with what it means.

   The functions that are left out:
   - what a unit exports and imports: [Serializable],
     [Pre_serializable], [Cached_level.clean_for_export],
     [remove_unused_value_slots_and_shortcut_aliases],
     [create_from_closure_conversion_approx],
     [Typing_env_level.ids_for_export];
   - [Meet_or_join_env_base] and [Join_env], which are for join;
   - [find_params]: the parameters of the model have no kinds yet;
   - [stable_compare_simples], an order on names by compilation unit
     and stamp, which the model's names have not;
   - printing;
   - not yet here: [aliases_of_simple] and [compute_joined_aliases]
     (they need [Aliases.Alias_set.filter] and [Aliases.add_alias_set]),
     [free_names_transitive], and [Typing_env_extension].

   The differences:
   - A fatal error of the compiler that no flag guards is [None]: of
     [find], for a name that is not bound or a type that is not of the
     kind expected; of [add_definition], for a symbol whose name mode is
     not the normal one; of [Typing_env_level.concat], and so of [cut];
     of [add_alias], where [Aliases.add] has one.
   - An exception [Not_found] that the callers catch is a [None] inside
     the first: [type_simple_in_term_exn], [get_canonical_simple_exn]
     and [get_alias_then_canonical_simple_exn] answer an
     [option (option _)].
   - What the compiler checks only when a flag asks for invariants is
     not checked by the functions here: that [add_variable_definition]
     is given a variable that is not bound yet ([check_invariants]);
     that the names of the type given to [replace_equation] are bound
     ([invariant_for_new_equation]), and that the name is not canonical
     if the type is an alias ([invariant_for_alias]); that an equation
     is not an alias of a name to itself
     ([More_type_creators.check_equation]); that the two maps of a level
     have the same variables ([Typing_env_level.create]); that a level
     made into an extension defines nothing
     ([as_extension_without_bindings]). The model follows the compiler
     with its flags off. FlambdaTypingEnvSpec.v says which of these the
     statements need, as premises or as the invariant of an
     environment.
   - A variable of the compiler carries its kind ([Variable.kind]), and
     [add_variable_definition] stops if it is not the kind given. A
     variable of the model is a name and no more; the kind of a
     variable is that of its type in [names_to_types].
   - [replace_equation] also checks, under a flag for debugging
     ([concrete_types_only_on_canonicals]), that a type that is not an
     alias is given to a canonical name only. That is not here.
   - [add_alias] never answers [Unknown], which the compiler answers
     where a .cmx file is missing.
   - [find_with_binding_time_and_mode] makes the pair of the binding
     time and the scoped name mode anew; the compiler returns the pair
     it found if the name mode is unchanged, which is the same pair.
   - An optional argument is an [option]. A [Bound_name.t] is a name and
     its name mode, two arguments; [Bound_parameters.t] is a list of
     variables with their kinds.
   - Maps and sets are those of FlambdaMapSpec.v: a map from variables
     is a [finite_map], one from the names of types a
     [Keyed.finite_map name] read with [name_eqb], one from binding
     times a [Keyed.finite_map Binding_time.t] read with [Z.eqb]. To
     add a binding is to cons it, so the union of two maps that takes
     the bindings of the second where both have one
     ([union_right_biased], in [concat]) is the second followed by the
     first.
   - The variables that a level has at one binding time are a set in
     the compiler and a list here, to which [add_definition] conses; a
     variable that is twice in the list is in the set once.
     [Typing_env_level.fold_on_defined_vars] goes through the binding
     times in increasing order, as the compiler does, and through the
     variables of one binding time, each once, in the order of their
     first places in the list, where the compiler goes through its set
     in its order on variables (by compilation unit and stamp), which
     the model's names have not. Where a binding time has one variable
     the two are the same; that is so of the levels of an environment
     that is well formed (FlambdaTypingEnvInvariantSpec.v).
   - [Flambda_kind.equal] is the core's [kind_eqb]
     (FlambdaKindModel.v). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Scopes                                                                   *)
(* ======================================================================== *)

(* [Scope.t]: the number of a level of a typing environment. *)
Module Scope.

  Definition t : Type := Z.

  Definition initial : t := 0.

  Definition next (scope : t) : t := scope + 1.

End Scope.

(* ======================================================================== *)
(* Levels                                                                   *)
(* ======================================================================== *)

(* [Typing_env_level]: what one level of a typing environment adds to
   the levels before it. *)
Module Typing_env_level.

  (* [Typing_env_level.t], with [create]: the variables that the level
     defines, each with its kind; the same variables by binding time;
     and the equations that the level adds, the type that each gives a
     name standing for the types that earlier levels give it. *)
  Record t : Type := create {
    defined_vars : finite_map flambda_kind;
    binding_times : Keyed.finite_map Binding_time.t (list variable);
    equations : Keyed.finite_map name flambda_type
  }.

  Definition defined_variables (level : t) : list variable :=
    domain (defined_vars level).

  Definition defined_variables_with_kinds (level : t)
      : finite_map flambda_kind :=
    defined_vars level.

  Definition defined_names (level : t) : list name :=
    map Name_var (defined_variables level).

  Definition variables_by_binding_time (level : t)
      : Keyed.finite_map Binding_time.t (list variable) :=
    binding_times level.

  (* [None] where the compiler raises [Not_found]. *)
  Definition find_kind (level : t) (var : variable) : option flambda_kind :=
    lookup var (defined_vars level).

  Definition variable_is_defined (level : t) (var : variable) : bool :=
    is_bound var (defined_vars level).

  Definition empty : t := create [] [] [].

  Definition is_empty (level : t) : bool :=
    Keyed.is_empty (defined_vars level)
    && Keyed.is_empty (binding_times level)
    && Keyed.is_empty (equations level).

  Definition add_definition (level : t) (var : variable)
      (kind : flambda_kind) (binding_time : Binding_time.t) : t :=
    let vars :=
      match Keyed.lookup Z.eqb binding_time (binding_times level) with
      | None => [var]
      | Some prev_vars => var :: prev_vars
      end
    in
    create ((var, kind) :: defined_vars level)
      ((binding_time, vars) :: binding_times level) (equations level).

  (* An equation whose type is the unknown one is not recorded: the
     level then has no equation on the name, whatever it had. *)
  Definition add_or_replace_equation (level : t) (name : name)
      (ty : flambda_type) : t :=
    if is_obviously_unknown ty
    then
      create (defined_vars level) (binding_times level)
        (Keyed.unbind name_eqb name (equations level))
    else
      create (defined_vars level) (binding_times level)
        ((name, ty) :: equations level).

  (* The two levels as one. A name with an equation in both keeps that
     of the later level. [None] is a fatal error of the compiler: the
     two levels define a variable in common, or have variables at a
     binding time in common. *)
  Definition concat (earlier later : t) : option t :=
    if Keyed.is_empty
         (inter (domain (defined_vars earlier)) (domain (defined_vars later)))
       && Keyed.is_empty
            (Keyed.inter Z.eqb
               (Keyed.domain Z.eqb (binding_times earlier))
               (Keyed.domain Z.eqb (binding_times later)))
    then
      Some
        (create (defined_vars later ++ defined_vars earlier)
           (binding_times later ++ binding_times earlier)
           (equations later ++ equations earlier))
    else None.

  (* One equation for each name that has one, the one that counts: the
     map of the compiler has one for each name. *)
  Definition as_extension_without_bindings (level : t) : env_extension :=
    Env_extension (Keyed.compact name_eqb (equations level)).

  (* Binding times in increasing order. *)
  Fixpoint insert_binding_time (binding_time : Binding_time.t)
      (sorted : list Binding_time.t) : list Binding_time.t :=
    match sorted with
    | [] => [binding_time]
    | first :: rest =>
      if binding_time <=? first
      then binding_time :: sorted
      else first :: insert_binding_time binding_time rest
    end.

  Definition sort_binding_times (binding_times : list Binding_time.t)
      : list Binding_time.t :=
    fold_right insert_binding_time [] binding_times.

  (* The variables that the level has by binding time, those of the
     earliest binding time first. Those of one binding time are each
     once, in the order of the list that the model has for them; the
     compiler has a set there, and goes through it in its order on
     variables. *)
  Definition variables_in_order (level : t) : list variable :=
    flat_map
      (fun binding_time =>
         match Keyed.lookup Z.eqb binding_time (binding_times level) with
         | Some vars => dedup vars
         | None => []
         end)
      (sort_binding_times (Keyed.domain Z.eqb (binding_times level))).

  (* [f] is given, by binding time, each variable that the level has at
     one, with the kind that the level defines it with. [None] is the
     compiler's exception [Not_found], which nothing catches: the level
     has at a binding time a variable that it does not define. *)
  Definition fold_on_defined_vars {A : Type}
      (f : variable -> flambda_kind -> A -> A) (level : t) (init : A)
      : option A :=
    fold_left
      (fun acc var =>
         let* acc := acc in
         let* kind := find_kind level var in
         Some (f var kind acc))
      (variables_in_order level) (Some init).

End Typing_env_level.

(* ======================================================================== *)
(* What a typing environment knows of its names                             *)
(* ======================================================================== *)

(* [Cached_level]: the type, the binding time and the name mode of every
   name that has them, and the aliases. *)
Module Cached_level.

  Record t : Type := {
    names_to_types :
      Keyed.finite_map name (flambda_type * Binding_time.With_name_mode.t);
    aliases : Aliases.t
  }.

  Definition empty : t :=
    {| names_to_types := []; aliases := Aliases.empty |}.

  Definition add_or_replace_binding (cached : t) (name : name)
      (ty : flambda_type) (binding_time : Binding_time.t)
      (name_mode : Name_mode.t) : t :=
    let binding_time_and_mode :=
      Binding_time.With_name_mode.create binding_time name_mode
    in
    {| names_to_types :=
         (name, (ty, binding_time_and_mode)) :: names_to_types cached;
       aliases := aliases cached |}.

  (* The type of a variable is replaced and its binding time and name
     mode kept. Nothing is done for a variable that has none. *)
  Definition replace_variable_binding (cached : t) (var : variable)
      (ty : flambda_type) : t :=
    match Keyed.lookup name_eqb (Name_var var) (names_to_types cached) with
    | Some (_, binding_time_and_mode) =>
      {| names_to_types :=
           (Name_var var, (ty, binding_time_and_mode))
           :: names_to_types cached;
         aliases := aliases cached |}
    | None => cached
    end.

  Definition with_aliases (cached : t) (aliases : Aliases.t) : t :=
    {| names_to_types := names_to_types cached; aliases := aliases |}.

End Cached_level.

(* ======================================================================== *)
(* Typing environments                                                      *)
(* ======================================================================== *)

Module Typing_env.

  (* [Typing_env.One_level.t], without its [just_after_level]: the scope
     of a level and what the level adds. *)
  Module One_level.

    Record t : Type := create {
      scope : Scope.t;
      level : Typing_env_level.t
    }.

    Definition create_empty (scope : Scope.t) : t :=
      create scope Typing_env_level.empty.

    Definition is_empty (one_level : t) : bool :=
      Typing_env_level.is_empty (level one_level).

    Definition bump_scope (one_level : t) : t :=
      create (Scope.next (scope one_level)) (level one_level).

  End One_level.

  (* [Typing_env.t]. [prev_levels] has the greatest scope at its head.
     [just_after_level] is that of the current level. A variable bound
     before [min_binding_time] is out of scope, and may occur only in
     types. *)
  Record t : Type := {
    defined_symbols : list symbol;
    prev_levels : list One_level.t;
    current_level : One_level.t;
    just_after_level : Cached_level.t;
    next_binding_time : Binding_time.t;
    min_binding_time : Binding_time.t;
    is_bottom : bool
  }.

  Definition make_bottom (tenv : t) : t :=
    {| defined_symbols := defined_symbols tenv;
       prev_levels := prev_levels tenv;
       current_level := current_level tenv;
       just_after_level := just_after_level tenv;
       next_binding_time := next_binding_time tenv;
       min_binding_time := min_binding_time tenv;
       is_bottom := true |}.

  Definition aliases (tenv : t) : Aliases.t :=
    Cached_level.aliases (just_after_level tenv).

  Definition names_to_types (tenv : t)
      : Keyed.finite_map name (flambda_type * Binding_time.With_name_mode.t)
      :=
    Cached_level.names_to_types (just_after_level tenv).

  Definition current_scope (tenv : t) : Scope.t :=
    One_level.scope (current_level tenv).

  (* The scope of the first level is not the initial one, so that there
     is a scope before it to cut after. *)
  Definition create : t :=
    {| defined_symbols := [];
       prev_levels := [];
       current_level := One_level.create_empty (Scope.next Scope.initial);
       just_after_level := Cached_level.empty;
       next_binding_time := Binding_time.earliest_var;
       min_binding_time := Binding_time.earliest_var;
       is_bottom := false |}.

  Definition with_current_level (tenv : t) (current_level : One_level.t)
      (just_after_level : Cached_level.t) : t :=
    {| defined_symbols := defined_symbols tenv;
       prev_levels := prev_levels tenv;
       current_level := current_level;
       just_after_level := just_after_level;
       next_binding_time := next_binding_time tenv;
       min_binding_time := min_binding_time tenv;
       is_bottom := is_bottom tenv |}.

  Definition with_current_level_and_next_binding_time (tenv : t)
      (current_level : One_level.t) (just_after_level : Cached_level.t)
      (next_binding_time : Binding_time.t) : t :=
    {| defined_symbols := defined_symbols tenv;
       prev_levels := prev_levels tenv;
       current_level := current_level;
       just_after_level := just_after_level;
       next_binding_time := next_binding_time;
       min_binding_time := min_binding_time tenv;
       is_bottom := is_bottom tenv |}.

  Definition with_aliases (tenv : t) (aliases : Aliases.t) : t :=
    with_current_level tenv (current_level tenv)
      (Cached_level.with_aliases (just_after_level tenv) aliases).

  Definition cached (tenv : t) : Cached_level.t := just_after_level tenv.

  (* A new level, with nothing in it yet, becomes the current one. *)
  Definition increment_scope (tenv : t) : t :=
    {| defined_symbols := defined_symbols tenv;
       prev_levels := current_level tenv :: prev_levels tenv;
       current_level :=
         One_level.create (Scope.next (current_scope tenv))
           Typing_env_level.empty;
       just_after_level := just_after_level tenv;
       next_binding_time := next_binding_time tenv;
       min_binding_time := min_binding_time tenv;
       is_bottom := is_bottom tenv |}.

  Definition bump_current_level_scope (tenv : t) : t :=
    with_current_level tenv (One_level.bump_scope (current_level tenv))
      (just_after_level tenv).

  Definition name_domain (tenv : t) : list name :=
    Keyed.domain name_eqb (names_to_types tenv)
    ++ map Name_symbol (defined_symbols tenv).

  (* The type of a symbol that is defined and has no equation. *)
  Definition initial_symbol_type
      : flambda_type * Binding_time.With_name_mode.t :=
    (any_non_null_value, Binding_time.With_name_mode.symbols).

  Definition check_optional_kind_matches (ty : flambda_type)
      (kind_opt : option flambda_kind) : bool :=
    match kind_opt with
    | None => true
    | Some kind => kind_eqb kind (FlambdaTypeGrammar.kind ty)
    end.

  (* The type of a name with its binding time and name mode. Every type
     is the bottom one in an environment that is bottom, but that of a
     symbol with no equation. [None] is a fatal error of the compiler:
     the name is not bound, or its type is not of the kind expected. *)
  Definition find_with_binding_time_and_mode' (tenv : t) (name : name)
      (kind : option flambda_kind)
      : option (flambda_type * Binding_time.With_name_mode.t) :=
    match Keyed.lookup name_eqb name (names_to_types tenv) with
    | None =>
      match name with
      | Name_var _ => None
      | Name_symbol sym =>
        if among sym (defined_symbols tenv)
           && check_optional_kind_matches (fst initial_symbol_type) kind
        then Some initial_symbol_type
        else None
      end
    | Some (ty, binding_time_and_mode) =>
      if check_optional_kind_matches ty kind
      then
        if is_bottom tenv
        then Some (bottom (FlambdaTypeGrammar.kind ty), binding_time_and_mode)
        else Some (ty, binding_time_and_mode)
      else None
    end.

  Definition find (tenv : t) (name : name) (kind : option flambda_kind)
      : option flambda_type :=
    option_map fst (find_with_binding_time_and_mode' tenv name kind).

  (* The same, with the name mode that the name has where only the
     variables bound at [min_binding_time] or later are in scope. *)
  Definition find_with_binding_time_and_mode (tenv : t) (name : name)
      (kind : option flambda_kind)
      : option (flambda_type * Binding_time.With_name_mode.t) :=
    let* '(ty, binding_time_and_mode) :=
      find_with_binding_time_and_mode' tenv name kind
    in
    let scoped_mode :=
      Binding_time.With_name_mode.scoped_name_mode binding_time_and_mode
        (min_binding_time tenv)
    in
    Some
      (ty,
       Binding_time.With_name_mode.create
         (Binding_time.With_name_mode.binding_time binding_time_and_mode)
         scoped_mode).

  Definition binding_time_and_mode (tenv : t) (name : name)
      : option Binding_time.With_name_mode.t :=
    match name with
    | Name_var _ =>
      option_map snd (find_with_binding_time_and_mode tenv name None)
    | Name_symbol _ => Some Binding_time.With_name_mode.symbols
    end.

  Definition binding_time_and_mode_of_simple (tenv : t) (simple : simple)
      : option Binding_time.With_name_mode.t :=
    match simple with
    | Const _ => Some Binding_time.With_name_mode.consts
    | Var x => binding_time_and_mode tenv (Name_var x)
    | Symbol s => binding_time_and_mode tenv (Name_symbol s)
    end.

  (* Whether a name is bound, with a name mode of at least
     [min_name_mode] if one is given. *)
  Definition mem (min_name_mode : option Name_mode.t) (tenv : t) (name : name)
      : bool :=
    match name with
    | Name_var _ =>
      match Keyed.lookup name_eqb name (names_to_types tenv) with
      | None => false
      | Some (_, binding_time_and_mode) =>
        let name_mode :=
          Binding_time.With_name_mode.scoped_name_mode binding_time_and_mode
            (min_binding_time tenv)
        in
        match min_name_mode with
        | None => true
        | Some min_name_mode =>
          match Name_mode.compare_partial_order min_name_mode name_mode with
          | None => false
          | Some c => c <=? 0
          end
        end
      end
    | Name_symbol sym => among sym (defined_symbols tenv)
    end.

  Definition mem_simple (min_name_mode : option Name_mode.t) (tenv : t)
      (simple : simple) : bool :=
    match simple with
    | Const _ => true
    | Var x => mem min_name_mode tenv (Name_var x)
    | Symbol s => mem min_name_mode tenv (Name_symbol s)
    end.

  Definition alias_is_bound_strictly_earlier (tenv : t) (bound_name : name)
      (alias : simple) : option bool :=
    let* time_of_name := binding_time_and_mode tenv bound_name in
    let* time_of_alias := binding_time_and_mode_of_simple tenv alias in
    Some
      (Binding_time.strictly_earlier
         (Binding_time.With_name_mode.binding_time time_of_alias)
         (Binding_time.With_name_mode.binding_time time_of_name)).

  (* [Typing_env.binding_time_resolver], for one compilation unit: the
     binding time and the name mode that the environment has for a
     name. *)
  Definition binding_time_resolver (tenv : t) (name : name)
      : Binding_time.With_name_mode.t :=
    match Keyed.lookup name_eqb name (names_to_types tenv) with
    | Some (_, binding_time_and_mode) => binding_time_and_mode
    | None =>
      match name with
      | Name_var _ => Binding_time.With_name_mode.imported_variables
      | Name_symbol _ => Binding_time.With_name_mode.symbols
      end
    end.

  (* A variable is defined, at the next binding time, with the unknown
     type of its kind. *)
  Definition add_variable_definition (tenv : t) (var : variable)
      (kind : flambda_kind) (name_mode : Name_mode.t) : t :=
    let level :=
      Typing_env_level.add_definition (One_level.level (current_level tenv))
        var kind (next_binding_time tenv)
    in
    let just_after_level :=
      Cached_level.add_or_replace_binding (cached tenv) (Name_var var)
        (unknown kind) (next_binding_time tenv) name_mode
    in
    let current_level := One_level.create (current_scope tenv) level in
    with_current_level_and_next_binding_time tenv current_level
      just_after_level
      (Binding_time.succ (next_binding_time tenv)).

  Definition add_symbol_definition (tenv : t) (sym : symbol) : t :=
    {| defined_symbols := sym :: defined_symbols tenv;
       prev_levels := prev_levels tenv;
       current_level := current_level tenv;
       just_after_level := just_after_level tenv;
       next_binding_time := next_binding_time tenv;
       min_binding_time := min_binding_time tenv;
       is_bottom := is_bottom tenv |}.

  Definition add_symbol_definitions (tenv : t) (syms : list symbol) : t :=
    {| defined_symbols := syms ++ defined_symbols tenv;
       prev_levels := prev_levels tenv;
       current_level := current_level tenv;
       just_after_level := just_after_level tenv;
       next_binding_time := next_binding_time tenv;
       min_binding_time := min_binding_time tenv;
       is_bottom := is_bottom tenv |}.

  (* [Typing_env.add_definition], of a [Bound_name.t] given as its name
     and its name mode. [None] is a fatal error of the compiler: a
     symbol is defined with a name mode that is not the normal one. *)
  Definition add_definition (tenv : t) (name : name) (name_mode : Name_mode.t)
      (kind : flambda_kind) : option t :=
    match name with
    | Name_var var =>
      Some (add_variable_definition tenv var kind name_mode)
    | Name_symbol sym =>
      if Name_mode.equal name_mode Name_mode.normal
      then Some (add_symbol_definition tenv sym)
      else None
    end.

  (* The parameters of a continuation or of a function, each with its
     kind, are defined in order with the normal name mode. *)
  Definition add_definitions_of_params (tenv : t)
      (params : list (variable * flambda_kind)) : t :=
    fold_left
      (fun tenv param =>
         add_variable_definition tenv (fst param) (snd param)
           Name_mode.normal)
      params tenv.

  (* The type of a name becomes [ty]: an equation is added to the
     current level, and the type that the environment has for the name
     is replaced. Nothing is asked of [ty] here; the caller is to have
     met it with the type that the name had (Meet_env.add_equation). *)
  Definition replace_equation (tenv : t) (name : name) (ty : flambda_type)
      : t :=
    let level :=
      Typing_env_level.add_or_replace_equation
        (One_level.level (current_level tenv)) name ty
    in
    let just_after_level :=
      match name with
      | Name_var var =>
        Cached_level.replace_variable_binding (just_after_level tenv) var ty
      | Name_symbol _ =>
        Cached_level.add_or_replace_binding (just_after_level tenv) name ty
          Binding_time.symbols Name_mode.normal
      end
    in
    with_current_level tenv (One_level.create (current_scope tenv) level)
      just_after_level.

  Definition aliases_add (tenv : t)
      (canonical_element1 canonical_element2 : simple)
      : option (Or_bottom.t Aliases.add_result) :=
    Aliases.add (binding_time_resolver tenv) (names_to_types tenv)
      (aliases tenv) canonical_element1 canonical_element2.

  Record add_alias_result : Type := {
    canonical_element : simple;
    demoted_name : name;
    add_alias_result_t : t
  }.

  (* Two canonical elements are made aliases: one of them stays
     canonical and the other, a name, is demoted. Only the aliases of
     the environment change; the types of the two are for the caller to
     bring up to date. [None] is a fatal error of the compiler in
     [Aliases.add]; [Bottom] is two different constants; [Unknown],
     which the compiler answers where a .cmx file is missing, is never
     answered here. *)
  Definition add_alias (tenv : t)
      (canonical_element1 canonical_element2 : simple)
      : option (or_unknown_or_bottom add_alias_result) :=
    match aliases_add tenv canonical_element1 canonical_element2 with
    | None => None
    | Some Or_bottom.Bottom => Some Bottom
    | Some (Or_bottom.Ok result) =>
      Some
        (Ok
           {| canonical_element := Aliases.canonical_element result;
              demoted_name := Aliases.demoted_name result;
              add_alias_result_t :=
                with_aliases tenv (Aliases.add_result_t result) |})
    end.

  (* The levels after the scope [cut_after], as one level. [None] is a
     fatal error of the compiler, in [Typing_env_level.concat]. *)
  Fixpoint cut_loop (cut_after : Scope.t) (result : Typing_env_level.t)
      (levels : list One_level.t) : option Typing_env_level.t :=
    match levels with
    | [] => Some result
    | one_level :: levels =>
      if cut_after <? One_level.scope one_level
      then
        let* result :=
          Typing_env_level.concat (One_level.level one_level) result
        in
        cut_loop cut_after result levels
      else Some result
    end.

  Definition cut (tenv : t) (cut_after : Scope.t)
      : option Typing_env_level.t :=
    if current_scope tenv <=? cut_after
    then Some Typing_env_level.empty
    else
      cut_loop cut_after (One_level.level (current_level tenv))
        (prev_levels tenv).

  Definition cut_as_extension (tenv : t) (cut_after : Scope.t)
      : option env_extension :=
    option_map Typing_env_level.as_extension_without_bindings
      (cut tenv cut_after).

  (* The type of a [Simple] that occurs in a term, as an alias to its
     canonical element of name mode at least [min_name_mode] (by default
     that of the [Simple]), with that element. The outer [None] is a
     fatal error of the compiler (a name that is not bound); the inner
     one is its exception [Not_found] (no alias has such a name mode). *)
  Definition type_simple_in_term_exn (tenv : t)
      (min_name_mode : option Name_mode.t) (simple : simple)
      : option (option (flambda_type * FlambdaModel.simple)) :=
    let* '(ty, binding_time_and_name_mode_simple) :=
      match simple with
      | Const const =>
        Some (type_for_const const, Binding_time.With_name_mode.consts)
      | Var x => find_with_binding_time_and_mode tenv (Name_var x) None
      | Symbol s => find_with_binding_time_and_mode tenv (Name_symbol s) None
      end
    in
    let name_mode_simple :=
      Binding_time.With_name_mode.name_mode binding_time_and_name_mode_simple
    in
    let kind := FlambdaTypeGrammar.kind ty in
    let min_name_mode :=
      match min_name_mode with
      | None => name_mode_simple
      | Some name_mode => name_mode
      end
    in
    Some
      (match
         Aliases.get_canonical_element_exn (binding_time_resolver tenv)
           (names_to_types tenv) (aliases tenv) simple name_mode_simple
           min_name_mode (min_binding_time tenv)
       with
       | None => None
       | Some alias => Some (alias_type_of kind alias, alias)
       end).

  Definition get_canonical_simple_ignoring_name_mode (tenv : t)
      (simple : simple) : FlambdaModel.simple :=
    match simple with
    | Const _ => simple
    | Var x =>
      Aliases.get_canonical_ignoring_name_mode (aliases tenv) (Name_var x)
    | Symbol s =>
      Aliases.get_canonical_ignoring_name_mode (aliases tenv) (Name_symbol s)
    end.

  (* The canonical element of a [Simple], of name mode at least
     [min_name_mode] (by default that of the [Simple]). The two [None]s
     are as for [type_simple_in_term_exn]. *)
  Definition get_canonical_simple_exn (tenv : t)
      (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
      (simple : simple) : option (option FlambdaModel.simple) :=
    let* name_mode_simple :=
      match name_mode_of_existing_simple with
      | Some name_mode => Some name_mode
      | None =>
        option_map Binding_time.With_name_mode.name_mode
          (binding_time_and_mode_of_simple tenv simple)
      end
    in
    let min_name_mode :=
      match min_name_mode with
      | None => name_mode_simple
      | Some name_mode => name_mode
      end
    in
    Some
      (Aliases.get_canonical_element_exn (binding_time_resolver tenv)
         (names_to_types tenv) (aliases tenv) simple name_mode_simple
         min_name_mode (min_binding_time tenv)).

  (* The same of the [Simple] that a type is an alias to; [Not_found]
     if the type is not an alias. *)
  Definition get_alias_then_canonical_simple_exn (tenv : t)
      (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
      (typ : flambda_type) : option (option simple) :=
    match get_alias_opt typ with
    | None => Some None
    | Some simple =>
      get_canonical_simple_exn tenv min_name_mode
        name_mode_of_existing_simple simple
    end.

  (* The environment in which the body of a function is simplified: the
     variables bound so far go out of scope, and stay known to the
     types. *)
  Definition closure_env (tenv : t) : t :=
    increment_scope
      {| defined_symbols := defined_symbols tenv;
         prev_levels := prev_levels tenv;
         current_level := current_level tenv;
         just_after_level := just_after_level tenv;
         next_binding_time := next_binding_time tenv;
         min_binding_time := next_binding_time tenv;
         is_bottom := is_bottom tenv |}.

End Typing_env.
