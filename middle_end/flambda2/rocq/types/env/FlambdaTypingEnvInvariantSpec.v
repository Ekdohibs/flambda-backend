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


(* The invariant of a typing environment: what the functions that build
   an environment keep, and what the statements about meet, join and
   cutting take as a premise.

   This file is to be reviewed: the conditions, with what is said of
   where the compiler has each; the test [typing_env_well_formed]; and
   the statements. The environment is FlambdaTypingEnv.v, to be read
   against middle_end/flambda2/types/env/typing_env.ml and
   typing_env_level.ml. What satisfies an environment is
   FlambdaTypingEnvSpec.v, which has no premise on the environment;
   the invariant is about the environment alone, not about values.

   The compiler has no one statement of this. Some of the conditions
   are checked by it when invariant checks are on, some hold by the way
   its functions are written, and some are relied on with nothing that
   states them. This file is to be the one statement, for the typing
   environment, of what a well-formed environment is; the aliases have
   theirs in FlambdaEnvAliasesSpec.v, which is a part of this one.

   The conditions, and for each where the compiler establishes or
   checks it. "Stricter" marks those where the model asks for what the
   compiler does not check, or checks only under a flag (decision D32);
   each has a comment at its definition.

   Of the names and their types ([names_to_types]):
   - [binding_times_bounded], [binding_times_in_range]: the binding
     time of a variable is one that a definition gave. Holds by
     construction: [Typing_env.add_variable_definition] gives the next
     binding time and moves on; [closure_env] sets [min_binding_time]
     to the next. Nothing checks it.
   - [binding_times_distinct]: different variables have different
     binding times. By construction, for the same reason. Stricter:
     nothing checks it, and the order of the aliases relies on it
     (discrepancy T3-1).
   - [types_closed]: the variables of every type are bound. Checked by
     [invariant_for_new_equation] when invariant checks are on.
     Stricter (discrepancy T2-2). The compiler's check is of symbols
     too; here it is of variables only until the types have their free
     names.
   - [symbols_defined]: a symbol that has a type is defined. Stricter:
     nothing checks it, and [replace_equation] gives a type to any
     symbol (discrepancy T2-1).
   - [symbols_recorded]: a symbol that has a type has the binding time
     and the name mode of symbols. By construction: only
     [replace_equation] gives a symbol a type, with those. Nothing
     checks it.

   Of the levels:
   - [levels_pairwise_disjoint]: no two levels define one variable or
     have one binding time. [Typing_env_level.concat] stops with a
     fatal error otherwise, so [Typing_env.cut] checks it of the levels
     that it cuts; [add_variable_definition] refuses a variable that is
     bound only when invariant checks are on (discrepancy T2-2). With
     it [cut] always answers ([cut_defined_statement]).
   - [defined_variables_bound]: a variable that a level defines has a
     type, of the kind it was defined with. By construction:
     [add_variable_definition] gives the unknown type of the kind, and
     [Meet_env] reads the type of a name with the kind of the type it
     is about to give it ([Typing_env.find] with a kind, a fatal error
     whatever the flags if they differ). [replace_equation] itself
     checks nothing (discrepancy T2-8). This is the one condition that
     speaks of kinds, and it is of two records of one definition, the
     level's and the environment's. Nothing here is about the kind of a
     value, or the kinds of the aliases inside a type (below).
   - [levels_indexed]: the variables that a level has by binding time
     are those that it defines. Checked by [Typing_env_level.create]
     when invariant checks are on; [fold_on_defined_vars] relies on it
     whatever the flags. Stricter (discrepancy T2-2). With it
     [fold_on_defined_vars] always answers
     ([fold_on_defined_vars_defined_statement]).
   - [level_binding_times_recorded]: a variable that a level has at a
     binding time has that binding time in the environment. By
     construction: [add_variable_definition] gives the level and the
     environment the one binding time, and a new type for a variable
     leaves its binding time. Nothing checks it.
   - [level_binding_times_in_range]: the binding times of the levels
     are before the next. By construction.
   - [scopes_decreasing]: each level has a greater scope than the one
     before it. By construction ([increment_scope]).
   - [variables_defined]: a variable that has a type is defined by one
     of the levels; the converse of [defined_variables_bound]. By
     construction: of the variables of the unit being compiled only
     [add_variable_definition] gives one an entry, and it defines it
     in the current level. Nothing checks it. (The compiler also gives
     an entry to a variable of another compilation unit that gets an
     equation, which no level defines; the model has no such
     variable.)
   - [levels_agree]: the equation that the latest level has for a name
     is the type that the environment has for the name. Stricter: the
     compiler relies on it ("We rely on the fact that equations in
     later levels are more precise", [Typing_env_level.concat]) and
     nothing checks it; two things break it, a level dropping an
     equation whose type is unknown (discrepancy T2-4) and an equation
     on a variable that is not bound (discrepancy T2-1). With it the
     equations of the level that [cut] answers are the types that the
     environment has ([cut_equations_found_statement]). It is a
     proposition and not a test: it compares types, and the types have
     no test of equality yet. So the invariant
     ([typing_env_invariant]) is the test and this.

   Of the aliases:
   - [aliases_consistent] and [name_modes_recorded]
     (FlambdaEnvAliasesSpec.v, where each says how the model is
     stricter than the compiler). The order of the canonical elements
     ([canonical_elements_earliest]) is not here: [Aliases.add] does
     not keep it for two symbols (discrepancy T3-2), and a statement
     that needs it takes it as a premise.
   - [aliases_known]: the names that the aliases speak of are bound: a
     variable has a type, a symbol is defined. Stricter: nothing checks
     it; [Aliases.add] reads the binding times of its arguments from
     the environment, and takes a variable with no entry for one of
     another compilation unit.
   - [canonical_names_concrete]: a name whose type is an alias is not
     canonical. Checked by [invariant_for_alias] when light invariant
     checks are on; [Expand_head] relies on it always. Stricter
     (discrepancy T2-2).

   Beside the invariant for now: that every type that the environment
   has for a name is well formed ([types_well_formed], with the test of
   FlambdaTypeWellFormedSpec.v). Every function keeps it, with the
   premise for [replace_equation] that the type given is well formed.
   It is to be a conjunct of the invariant when the meet says that the
   types that it answers are well formed; until then
   [replace_equation_invariant_statement] does not ask it.

   What is not here, and is to come: the free names of an environment
   and its renaming.

   What is not here because it is not so: that a name that is not
   canonical has the alias type of its canonical element. It has it
   when it is demoted ([Meet_env.record_demotion]); when the element is
   demoted in turn the aliases move the name to the new element, and
   its type still names the old one. Between the two steps of
   [record_demotion] it has not even an alias type, and the invariant
   holds there ([add_alias_invariant_statement]).

   What is not here, and is not to be: that an alias inside a type is
   at the kind that the environment has for its target. The compiler
   checks that only where it follows an alias ([Typing_env.find] with a
   kind), and programs that it accepts break agreement of kinds, so it
   is a premise of the statements that need it, in a file of its own,
   and no part of the invariant.

   The statements: the test answers [true] exactly when the conditions
   hold; what the conditions give to the aliases and to [cut]; and that
   each function that builds an environment keeps the invariant, under
   what it needs. What the functions do to satisfaction is
   FlambdaTypingEnvOperationsSpec.v.

   The proofs are in FlambdaTypingEnvInvariantProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeWellFormedSpec FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesInvariantSpec
  FlambdaTypingEnv.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The names and their types                                                *)
(* ======================================================================== *)

(* The binding time of an entry of [names_to_types]. *)
Definition binding_time_of
    (entry : flambda_type * Binding_time.With_name_mode.t) : Binding_time.t :=
  Binding_time.With_name_mode.binding_time (snd entry).

(* The binding times that definitions give are from the first one of a
   variable on, and the variables in scope are those bound from
   [min_binding_time] on, which is not after the next binding time. *)
Definition binding_times_bounded (tenv : Typing_env.t) : Prop :=
  Binding_time.earliest_var <= Typing_env.min_binding_time tenv
  ∧ Typing_env.min_binding_time tenv <= Typing_env.next_binding_time tenv.

Definition binding_times_bounded_b (tenv : Typing_env.t) : bool :=
  (Binding_time.earliest_var <=? Typing_env.min_binding_time tenv)
  && (Typing_env.min_binding_time tenv <=? Typing_env.next_binding_time tenv).

(* The binding time of a variable is one that a definition gave: not
   before the first, and before the next. *)
Definition binding_times_in_range (tenv : Typing_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    Binding_time.earliest_var
    <= Binding_time.With_name_mode.binding_time binding_time_and_mode
    ∧ Binding_time.With_name_mode.binding_time binding_time_and_mode
      < Typing_env.next_binding_time tenv.

Definition binding_times_in_range_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n entry =>
       match n with
       | Name_var _ =>
         (Binding_time.earliest_var <=? binding_time_of entry)
         && (binding_time_of entry <? Typing_env.next_binding_time tenv)
       | Name_symbol _ => true
       end)
    (Typing_env.names_to_types tenv).

(* Different variables have different binding times. *)
(* The model is stricter than the compiler here (decision D32). The
   compiler checks nothing of the kind. Its environments have the
   condition by the way they are made: an entry for a variable of the
   unit being compiled is added only by
   [Typing_env.add_variable_definition], with the next binding time,
   which then moves on. (A level that a join makes may give several
   variables one binding time; such a level is added to an environment
   by [Meet_env.add_env_extension_from_level], which defines each of
   its variables anew.) The order of the aliases relies on it:
   [Aliases.add] asks which of two variables is the earlier, and for
   two of one binding time the compiler stops with a fatal error.
   Discrepancy T3-1. *)
Definition binding_times_distinct (tenv : Typing_env.t) : Prop :=
  ∀ (x y : variable) (ty ty' : flambda_type)
    (binding_time_and_mode binding_time_and_mode'
      : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    Keyed.lookup name_eqb (Name_var y) (Typing_env.names_to_types tenv)
    = Some (ty', binding_time_and_mode') ->
    Binding_time.With_name_mode.binding_time binding_time_and_mode
    = Binding_time.With_name_mode.binding_time binding_time_and_mode' ->
    x = y.

Definition binding_times_distinct_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n entry =>
       Keyed.for_all_bindings_b name_eqb
         (fun n' entry' =>
            match n, n' with
            | Name_var x, Name_var y =>
              Pos.eqb x y
              || negb (binding_time_of entry =? binding_time_of entry')
            | _, _ => true
            end)
         (Typing_env.names_to_types tenv))
    (Typing_env.names_to_types tenv).

(* The variables that the type of a name mentions are bound. *)
(* The model is stricter than the compiler here (decision D32). The
   compiler checks it of the type that [Typing_env.replace_equation]
   gives a name, in [invariant_for_new_equation], when invariant checks
   are on, and not otherwise. Its check is of the symbols that the type
   mentions too. Discrepancy T2-2. *)
Definition types_closed (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (x : variable),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    x ∈ free_variables ty ->
    Keyed.is_bound name_eqb (Name_var x) (Typing_env.names_to_types tenv)
    = true.

Definition types_closed_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun _ entry =>
       forallb
         (fun x =>
            Keyed.is_bound name_eqb (Name_var x)
              (Typing_env.names_to_types tenv))
         (free_variables (fst entry)))
    (Typing_env.names_to_types tenv).

(* A symbol that has a type is defined. *)
(* The model is stricter than the compiler here (decision D32). Nothing
   in the compiler checks it: [Typing_env.replace_equation] gives a
   type to any symbol, after which [Typing_env.find] answers for the
   symbol and [Typing_env.mem] says that it is not bound. Discrepancy
   T2-1. *)
Definition symbols_defined (tenv : Typing_env.t) : Prop :=
  ∀ (s : symbol) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_symbol s) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    among s (Typing_env.defined_symbols tenv) = true.

(* A symbol that has a type has the binding time and the name mode of
   symbols. *)
(* Nothing in the compiler checks this. Its environments have it by the
   way they are made: a symbol of the unit being compiled gets a type
   only from [Typing_env.replace_equation], which gives it
   [Binding_time.symbols] and the normal name mode, whatever it had. *)
Definition symbols_recorded (tenv : Typing_env.t) : Prop :=
  ∀ (s : symbol) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_symbol s) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    binding_time_and_mode = Binding_time.With_name_mode.symbols.

Definition symbols_recorded_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n entry =>
       match n with
       | Name_symbol _ =>
         Binding_time.With_name_mode.equal (snd entry)
           Binding_time.With_name_mode.symbols
       | Name_var _ => true
       end)
    (Typing_env.names_to_types tenv).

Definition symbols_defined_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n _ =>
       match n with
       | Name_symbol s => among s (Typing_env.defined_symbols tenv)
       | Name_var _ => true
       end)
    (Typing_env.names_to_types tenv).

(* ======================================================================== *)
(* The levels                                                               *)
(* ======================================================================== *)

(* The levels of an environment, the latest first; their scopes; and
   their equations as one map, in which the equation of the latest
   level that has one for a name is the one that counts. *)
Definition all_levels (tenv : Typing_env.t) : list Typing_env_level.t :=
  map Typing_env.One_level.level
    (Typing_env.current_level tenv :: Typing_env.prev_levels tenv).

Definition all_scopes (tenv : Typing_env.t) : list Scope.t :=
  map Typing_env.One_level.scope
    (Typing_env.current_level tenv :: Typing_env.prev_levels tenv).

Definition all_equations (tenv : Typing_env.t)
    : Keyed.finite_map name flambda_type :=
  flat_map Typing_env_level.equations (all_levels tenv).

(* Two levels define no variable in common and have no binding time in
   common. This is what [Typing_env_level.concat] asks of its
   arguments, with a fatal error. *)
Definition levels_disjoint (earlier later : Typing_env_level.t) : Prop :=
  (∀ x : variable,
    is_bound x (Typing_env_level.defined_vars earlier) = true ->
    is_bound x (Typing_env_level.defined_vars later) = false)
  ∧ (∀ binding_time : Binding_time.t,
      Keyed.is_bound Z.eqb binding_time
        (Typing_env_level.binding_times earlier)
      = true ->
      Keyed.is_bound Z.eqb binding_time (Typing_env_level.binding_times later)
      = false).

Definition levels_disjoint_b (earlier later : Typing_env_level.t) : bool :=
  Keyed.is_empty
    (inter (domain (Typing_env_level.defined_vars earlier))
       (domain (Typing_env_level.defined_vars later)))
  && Keyed.is_empty
       (Keyed.inter Z.eqb
          (Keyed.domain Z.eqb (Typing_env_level.binding_times earlier))
          (Keyed.domain Z.eqb (Typing_env_level.binding_times later))).

(* No two of the levels, given the latest first, have a variable or a
   binding time in common. *)
(* The model is stricter than the compiler here (decision D32), in
   that it asks this of every environment: the compiler finds out when
   it cuts ([Typing_env_level.concat] stops with a fatal error), and
   [Typing_env.add_variable_definition] refuses a variable that is
   bound only when invariant checks are on. Discrepancy T2-2. *)
Fixpoint levels_pairwise_disjoint (levels : list Typing_env_level.t)
    : Prop :=
  match levels with
  | [] => True
  | later :: levels =>
    (∀ earlier : Typing_env_level.t,
      earlier ∈ levels -> levels_disjoint earlier later)
    ∧ levels_pairwise_disjoint levels
  end.

Fixpoint levels_pairwise_disjoint_b (levels : list Typing_env_level.t)
    : bool :=
  match levels with
  | [] => true
  | later :: levels =>
    forallb (fun earlier => levels_disjoint_b earlier later) levels
    && levels_pairwise_disjoint_b levels
  end.

(* A variable that a level defines has a type in the environment, of
   the kind that the level defined it with. *)
(* The kind here is of two records of one definition: the kind that the
   level has for the variable, and the kind of the type that the
   environment has for it. It is not about the kind of a value, and not
   about the kinds of the aliases inside a type.
   [Typing_env.add_variable_definition] makes the two agree: it gives
   the level the kind and the environment the unknown type of that
   kind (and stops with a fatal error, whatever the flags, if the kind
   is not the variable's own). The model is stricter than the compiler
   in one place (decision D32): [Typing_env.replace_equation] and
   [Cached_level.replace_variable_binding] take a type of any kind for
   a variable and check nothing. It is their callers in [Meet_env] that
   keep the kind: each reads the type of the name with the kind of the
   type that it is about to give ([Typing_env.find] with a kind, in
   [add_concrete_equation_on_canonical_name] and [record_demotion]),
   which is a fatal error, whatever the flags, if the kinds differ.
   [replace_equation_invariant_statement] has that as a premise.
   Discrepancy T2-8. *)
Definition defined_variables_bound (tenv : Typing_env.t) : Prop :=
  ∀ (level : Typing_env_level.t) (x : variable) (k : flambda_kind),
    level ∈ all_levels tenv ->
    lookup x (Typing_env_level.defined_vars level) = Some k ->
    ∃ (ty : flambda_type)
      (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode)
      ∧ kind ty = k.

Definition defined_variables_bound_b (tenv : Typing_env.t) : bool :=
  forallb
    (fun level =>
       for_all_bindings_b
         (fun x k =>
            match
              Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv)
            with
            | Some (ty, _) => kind_eqb (kind ty) k
            | None => false
            end)
         (Typing_env_level.defined_vars level))
    (all_levels tenv).

(* A level has a variable at a binding time: the variables that it has
   for the binding time ([Typing_env_level.variables_by_binding_time])
   include it. *)
Definition indexed_at (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) : Prop :=
  ∃ vars : list variable,
    Keyed.lookup Z.eqb binding_time (Typing_env_level.binding_times level)
    = Some vars
    ∧ x ∈ vars.

Definition indexed_at_b (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) : bool :=
  match
    Keyed.lookup Z.eqb binding_time (Typing_env_level.binding_times level)
  with
  | Some vars => among x vars
  | None => false
  end.

(* The variables that a level has by binding time are those that it
   defines. *)
(* The model is stricter than the compiler here (decision D32), in
   that it asks this whatever the flags. [Typing_env_level.create]
   checks it (the keys of [defined_vars] are the union of the sets of
   [binding_times]) when invariant checks are on.
   [Typing_env_level.fold_on_defined_vars] relies on it always: it goes
   through the variables by binding time and looks up the kind of each
   among the defined variables, with [Variable.Map.find], which raises
   [Not_found] for a variable that is not defined; and it does not
   reach a defined variable that is at no binding time.
   Discrepancy T2-2. *)
Definition level_indexed (level : Typing_env_level.t) : Prop :=
  ∀ x : variable,
    is_bound x (Typing_env_level.defined_vars level) = true
    <-> ∃ binding_time : Binding_time.t, indexed_at level binding_time x.

Definition level_indexed_b (level : Typing_env_level.t) : bool :=
  Keyed.for_all_bindings_b Z.eqb
    (fun _ vars =>
       all_among vars (domain (Typing_env_level.defined_vars level)))
    (Typing_env_level.binding_times level)
  && forallb
       (fun x =>
          existsb (fun binding_time => indexed_at_b level binding_time x)
            (Keyed.domain Z.eqb (Typing_env_level.binding_times level)))
       (domain (Typing_env_level.defined_vars level)).

Definition levels_indexed (tenv : Typing_env.t) : Prop :=
  ∀ level : Typing_env_level.t,
    level ∈ all_levels tenv -> level_indexed level.

Definition levels_indexed_b (tenv : Typing_env.t) : bool :=
  forallb level_indexed_b (all_levels tenv).

(* A variable that a level has at a binding time has that binding time
   in the environment. *)
(* Nothing in the compiler checks this. Its environments have it by
   the way they are made: [Typing_env.add_variable_definition] gives
   the level and the environment the one binding time, the next;
   [Typing_env.replace_equation] leaves the binding time of a variable
   and the levels' variables by binding time as they are. The levels
   that a join makes, in which several variables may be at one binding
   time, are not levels of an environment: an environment takes one by
   defining each of its variables anew
   ([Meet_env.add_env_extension_from_level]). *)
Definition level_binding_times_recorded (tenv : Typing_env.t) : Prop :=
  ∀ (level : Typing_env_level.t) (binding_time : Binding_time.t)
    (x : variable),
    level ∈ all_levels tenv ->
    indexed_at level binding_time x ->
    ∃ (ty : flambda_type)
      (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode)
      ∧ Binding_time.With_name_mode.binding_time binding_time_and_mode
        = binding_time.

Definition level_binding_times_recorded_b (tenv : Typing_env.t) : bool :=
  forallb
    (fun level =>
       Keyed.for_all_bindings_b Z.eqb
         (fun binding_time vars =>
            forallb
              (fun x =>
                 match
                   Keyed.lookup name_eqb (Name_var x)
                     (Typing_env.names_to_types tenv)
                 with
                 | Some entry => binding_time_of entry =? binding_time
                 | None => false
                 end)
              vars)
         (Typing_env_level.binding_times level))
    (all_levels tenv).

(* The binding times of the levels are before the next one. *)
Definition level_binding_times_in_range (tenv : Typing_env.t) : Prop :=
  ∀ (level : Typing_env_level.t) (binding_time : Binding_time.t),
    level ∈ all_levels tenv ->
    Keyed.is_bound Z.eqb binding_time (Typing_env_level.binding_times level)
    = true ->
    binding_time < Typing_env.next_binding_time tenv.

Definition level_binding_times_in_range_b (tenv : Typing_env.t) : bool :=
  forallb
    (fun level =>
       forallb
         (fun binding_time =>
            binding_time <? Typing_env.next_binding_time tenv)
         (Keyed.domain Z.eqb (Typing_env_level.binding_times level)))
    (all_levels tenv).

(* Of the scopes of the levels, the latest first, each is greater than
   the one after it. *)
Fixpoint scopes_decreasing (scopes : list Scope.t) : Prop :=
  match scopes with
  | [] => True
  | later :: scopes =>
    match scopes with
    | [] => True
    | earlier :: _ => earlier < later
    end
    ∧ scopes_decreasing scopes
  end.

Fixpoint scopes_decreasing_b (scopes : list Scope.t) : bool :=
  match scopes with
  | [] => true
  | later :: scopes =>
    match scopes with
    | [] => true
    | earlier :: _ => earlier <? later
    end
    && scopes_decreasing_b scopes
  end.

(* A variable that has a type is defined by one of the levels. *)
(* The compiler's environments have this of the variables of the unit
   being compiled, by the way they are made: only
   [Typing_env.add_variable_definition] gives such a variable an
   entry, and it defines the variable in the current level;
   [Cached_level.replace_variable_binding] replaces an entry that there
   is. Nothing checks it. A variable of another compilation unit gets
   an entry where it gets an equation ([Typing_env.replace_equation],
   at [Binding_time.imported_variables]), and no level defines it; the
   model has no such variable. *)
Definition variables_defined (tenv : Typing_env.t) : Prop :=
  ∀ (x : variable) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    ∃ level : Typing_env_level.t,
      level ∈ all_levels tenv
      ∧ is_bound x (Typing_env_level.defined_vars level) = true.

Definition variables_defined_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n _ =>
       match n with
       | Name_var x =>
         existsb
           (fun level => is_bound x (Typing_env_level.defined_vars level))
           (all_levels tenv)
       | Name_symbol _ => true
       end)
    (Typing_env.names_to_types tenv).

(* The equation that the latest level has for a name is the type that
   the environment has for the name. *)
(* The model is stricter than the compiler here (decision D32). The
   compiler relies on it where it cuts ("We rely on the fact that
   equations in later levels are more precise",
   [Typing_env_level.concat]) and nothing checks it. Two things break
   it. [Typing_env_level.add_or_replace_equation] drops the equation of
   a name whose new type is obviously unknown, so that an equation of
   an earlier level becomes the latest (discrepancy T2-4).
   [Typing_env.replace_equation] on a variable that is not bound adds
   an equation and no type (discrepancy T2-1).
   [replace_equation_invariant_statement] has the premises that exclude
   the two. *)
Definition levels_agree (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type),
    Keyed.lookup name_eqb n (all_equations tenv) = Some ty ->
    ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode).

(* ======================================================================== *)
(* The aliases                                                              *)
(* ======================================================================== *)

(* A name that has been demoted, and the canonical element that it has
   been demoted to, are bound: a variable has a type, a symbol is
   defined. With [aliases_consistent] this is so of every name that the
   aliases speak of ([well_formed_mentioned_known_statement]). *)
(* The model is stricter than the compiler here (decision D32). Nothing
   in the compiler checks it. [Aliases.add] reads the binding time and
   the name mode of each of its arguments from the environment, and
   takes a variable that has no entry there for a variable of another
   compilation unit. *)
Definition aliases_known (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (canonical_element : simple),
    demoted_to (Typing_env.aliases tenv) n canonical_element ->
    Typing_env.mem None tenv n = true
    ∧ Typing_env.mem_simple None tenv canonical_element = true.

Definition aliases_known_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n canonical_element =>
       Typing_env.mem None tenv n
       && Typing_env.mem_simple None tenv canonical_element)
    (Aliases.canonical_elements (Typing_env.aliases tenv)).

(* A name whose type is an alias is not canonical. *)
(* The model is stricter than the compiler here (decision D32). The
   compiler checks it of the type that [Typing_env.replace_equation]
   gives a name, in [invariant_for_alias], when light invariant checks
   are on. [Expand_head.expand_head] relies on it whatever the flags:
   it stops with a fatal error on a canonical name whose type is an
   alias. Discrepancy T2-2. *)
Definition canonical_names_concrete (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t) (alias : simple),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    get_alias_opt ty = Some alias ->
    ~ is_canonical (Typing_env.aliases tenv) (simple_of_name n).

Definition canonical_names_concrete_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n entry =>
       match get_alias_opt (fst entry) with
       | Some _ =>
         negb (is_canonical_b (Typing_env.aliases tenv) (simple_of_name n))
       | None => true
       end)
    (Typing_env.names_to_types tenv).

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

(* The conditions that have a test, and the test. *)
Definition typing_env_conditions (tenv : Typing_env.t) : Prop :=
  (binding_times_bounded tenv
   ∧ binding_times_in_range tenv
   ∧ binding_times_distinct tenv
   ∧ types_closed tenv
   ∧ symbols_defined tenv
   ∧ symbols_recorded tenv)
  ∧ (levels_pairwise_disjoint (all_levels tenv)
     ∧ defined_variables_bound tenv
     ∧ levels_indexed tenv
     ∧ level_binding_times_recorded tenv
     ∧ level_binding_times_in_range tenv
     ∧ scopes_decreasing (all_scopes tenv)
     ∧ variables_defined tenv)
  ∧ (aliases_consistent (Typing_env.aliases tenv)
     ∧ name_modes_recorded (Typing_env.names_to_types tenv)
         (Typing_env.aliases tenv)
     ∧ aliases_known tenv
     ∧ canonical_names_concrete tenv).

Definition typing_env_well_formed (tenv : Typing_env.t) : bool :=
  (binding_times_bounded_b tenv
   && binding_times_in_range_b tenv
   && binding_times_distinct_b tenv
   && types_closed_b tenv
   && symbols_defined_b tenv
   && symbols_recorded_b tenv)
  && (levels_pairwise_disjoint_b (all_levels tenv)
      && defined_variables_bound_b tenv
      && levels_indexed_b tenv
      && level_binding_times_recorded_b tenv
      && level_binding_times_in_range_b tenv
      && scopes_decreasing_b (all_scopes tenv)
      && variables_defined_b tenv)
  && (aliases_consistent_b (Typing_env.aliases tenv)
      && name_modes_recorded_b (Typing_env.names_to_types tenv)
           (Typing_env.aliases tenv)
      && aliases_known_b tenv
      && canonical_names_concrete_b tenv).

(* The invariant of a typing environment. *)
Definition typing_env_invariant (tenv : Typing_env.t) : Prop :=
  typing_env_well_formed tenv = true ∧ levels_agree tenv.

(* The test of two levels answers [true] exactly when they are
   disjoint, which is when [Typing_env_level.concat] answers. *)
Definition levels_disjoint_b_iff_statement : Prop :=
  ∀ earlier later : Typing_env_level.t,
    levels_disjoint_b earlier later = true <-> levels_disjoint earlier later.

Definition concat_defined_statement : Prop :=
  ∀ earlier later : Typing_env_level.t,
    Typing_env_level.concat earlier later <> None
    <-> levels_disjoint earlier later.

(* The test of the index of a level answers [true] exactly when the
   variables that the level has by binding time are those that it
   defines. *)
Definition level_indexed_b_iff_statement : Prop :=
  ∀ level : Typing_env_level.t,
    level_indexed_b level = true <-> level_indexed level.

(* [Typing_env_level.concat] keeps it, and has a variable at a binding
   time only where one of the two levels has it. *)
Definition concat_indexed_statement : Prop :=
  ∀ earlier later result : Typing_env_level.t,
    Typing_env_level.concat earlier later = Some result ->
    level_indexed earlier ->
    level_indexed later ->
    level_indexed result
    ∧ ∀ (binding_time : Binding_time.t) (x : variable),
        indexed_at result binding_time x ->
        indexed_at earlier binding_time x ∨ indexed_at later binding_time x.

(* The test of an environment answers [true] exactly when the
   conditions hold. *)
Definition typing_env_well_formed_iff_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_well_formed tenv = true <-> typing_env_conditions tenv.

(* ======================================================================== *)
(* What the invariant gives                                                 *)
(* ======================================================================== *)

(* For the order of the aliases (FlambdaEnvAliasesSpec.v,
   [simples_comparable_statement]): two variables that are bound and
   have the same binding time in the environment are the same. *)
Definition well_formed_binding_times_distinct_statement : Prop :=
  ∀ (tenv : Typing_env.t) (x y : variable),
    typing_env_well_formed tenv = true ->
    Typing_env.mem None tenv (Name_var x) = true ->
    Typing_env.mem None tenv (Name_var y) = true ->
    binding_time_in (Typing_env.names_to_types tenv) x
    = binding_time_in (Typing_env.names_to_types tenv) y ->
    x = y.

(* Every name that the aliases speak of is bound. *)
Definition well_formed_mentioned_known_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name),
    typing_env_well_formed tenv = true ->
    aliases_mention (Typing_env.aliases tenv) n ->
    Typing_env.mem None tenv n = true.

(* The variables of a type that [Typing_env.find] answers are bound. *)
Definition find_closed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type) (x : variable),
    types_closed tenv ->
    Typing_env.find tenv n kind = Some ty ->
    x ∈ free_variables ty ->
    Typing_env.mem None tenv (Name_var x) = true.

(* The canonical element that is answered at a name mode is bound at
   that name mode, if the [Simple] asked about is bound. (A symbol that
   is not defined, and that the aliases do not speak of, is its own
   canonical element whatever the name mode asked for, and is not
   bound: [Typing_env.binding_time_and_mode] answers for any symbol.) *)
Definition get_canonical_simple_bound_statement : Prop :=
  ∀ (tenv : Typing_env.t) (min_name_mode : Name_mode.t)
    (s canonical : simple),
    typing_env_well_formed tenv = true ->
    Typing_env.mem_simple None tenv s = true ->
    Typing_env.get_canonical_simple_exn tenv (Some min_name_mode) None s
    = Some (Some canonical) ->
    Typing_env.mem_simple (Some min_name_mode) tenv canonical = true.

(* The extension that a level makes has the equations of the level,
   one for each name. *)
Definition as_extension_without_bindings_equations_statement : Prop :=
  ∀ (level : Typing_env_level.t)
    (equations : Keyed.finite_map name flambda_type),
    Typing_env_level.as_extension_without_bindings level
    = Env_extension equations ->
    (∀ n : name,
      Keyed.lookup name_eqb n equations
      = Keyed.lookup name_eqb n (Typing_env_level.equations level))
    ∧ Keyed.distinct name_eqb (map fst equations) = true.

(* [Typing_env.cut] answers, whatever the scope: the fatal error of
   [Typing_env_level.concat] does not happen. *)
Definition cut_defined_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t),
    typing_env_well_formed tenv = true ->
    ∃ level : Typing_env_level.t, Typing_env.cut tenv cut_after = Some level.

(* An equation of the level that [Typing_env.cut] answers is the type
   that the environment has for the name... *)
Definition cut_equations_found_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t)
    (n : name) (ty : flambda_type),
    levels_agree tenv ->
    Typing_env.cut tenv cut_after = Some level ->
    Keyed.lookup name_eqb n (Typing_env_level.equations level) = Some ty ->
    ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode).

(* ... as is any equation of that level, counted or not, an equation
   of one of the levels after the scope... *)
Definition cut_equations_member_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t)
    (n : name) (ty : flambda_type),
    Typing_env.cut tenv cut_after = Some level ->
    (n, ty) ∈ Typing_env_level.equations level ->
    ∃ one_level : Typing_env.One_level.t,
      one_level ∈ Typing_env.current_level tenv :: Typing_env.prev_levels tenv
      ∧ cut_after < Typing_env.One_level.scope one_level
      ∧ (n, ty)
        ∈ Typing_env_level.equations (Typing_env.One_level.level one_level).

(* ... and a variable that it defines has a type in the environment, of
   the kind that the level has for it. *)
Definition cut_defined_variables_found_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t)
    (x : variable) (k : flambda_kind),
    typing_env_well_formed tenv = true ->
    Typing_env.cut tenv cut_after = Some level ->
    lookup x (Typing_env_level.defined_vars level) = Some k ->
    ∃ (ty : flambda_type)
      (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
      = Some (ty, binding_time_and_mode)
      ∧ kind ty = k.

(* The variables that the level that [Typing_env.cut] answers defines
   are those that the levels after the scope define, each with the
   kind that its level has for it. *)
Definition cut_defined_variables_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t)
    (x : variable) (k : flambda_kind),
    typing_env_well_formed tenv = true ->
    Typing_env.cut tenv cut_after = Some level ->
    (lookup x (Typing_env_level.defined_vars level) = Some k
     <-> ∃ one_level : Typing_env.One_level.t,
           one_level
           ∈ Typing_env.current_level tenv :: Typing_env.prev_levels tenv
           ∧ cut_after < Typing_env.One_level.scope one_level
           ∧ lookup x
               (Typing_env_level.defined_vars
                  (Typing_env.One_level.level one_level))
             = Some k).

(* The variables that the level that [Typing_env.cut] answers has by
   binding time are those that it defines, and each is at the binding
   time that the environment has for it. So
   [Typing_env_level.fold_on_defined_vars] on that level finds the kind
   of every variable that it comes to, and comes to every variable that
   the level defines. *)
Definition cut_indexed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t),
    typing_env_well_formed tenv = true ->
    Typing_env.cut tenv cut_after = Some level ->
    level_indexed level
    ∧ ∀ (binding_time : Binding_time.t) (x : variable),
        indexed_at level binding_time x ->
        ∃ (ty : flambda_type)
          (binding_time_and_mode : Binding_time.With_name_mode.t),
          Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
          = Some (ty, binding_time_and_mode)
          ∧ Binding_time.With_name_mode.binding_time binding_time_and_mode
            = binding_time.

(* [Typing_env_level.fold_on_defined_vars] goes through the variables
   that a level has at a binding time, each once for each binding time
   that it is at... *)
Definition variables_in_order_indexed_statement : Prop :=
  ∀ (level : Typing_env_level.t) (x : variable),
    x ∈ Typing_env_level.variables_in_order level
    <-> ∃ binding_time : Binding_time.t, indexed_at level binding_time x.

(* ... so once in all, where no variable is at two binding times... *)
Definition variables_in_order_distinct_statement : Prop :=
  ∀ level : Typing_env_level.t,
    (∀ (x : variable) (binding_time binding_time' : Binding_time.t),
      indexed_at level binding_time x ->
      indexed_at level binding_time' x ->
      binding_time = binding_time') ->
    NoDup (Typing_env_level.variables_in_order level).

(* ... by binding time: one that it comes to before another is at a
   binding time that is not later. *)
Definition variables_in_order_sorted_statement : Prop :=
  ∀ (level : Typing_env_level.t) (before between after : list variable)
    (x y : variable),
    Typing_env_level.variables_in_order level
    = before ++ x :: between ++ y :: after ->
    ∃ binding_time binding_time' : Binding_time.t,
      indexed_at level binding_time x
      ∧ indexed_at level binding_time' y
      ∧ binding_time <= binding_time'.

(* On a level that has its index it goes through the variables that
   the level defines, and it answers: the exception of the compiler
   does not happen. What it answers is the fold, with the kind of each
   variable. *)
Definition variables_in_order_defined_statement : Prop :=
  ∀ (level : Typing_env_level.t) (x : variable),
    level_indexed level ->
    (x ∈ Typing_env_level.variables_in_order level
     <-> is_bound x (Typing_env_level.defined_vars level) = true).

Definition fold_on_defined_vars_defined_statement : Prop :=
  ∀ (A : Type) (f : variable -> flambda_kind -> A -> A)
    (level : Typing_env_level.t) (init : A),
    level_indexed level ->
    Typing_env_level.fold_on_defined_vars f level init
    = Some
        (List.fold_left
           (fun acc x =>
              match Typing_env_level.find_kind level x with
              | Some kind => f x kind acc
              | None => acc
              end)
           (Typing_env_level.variables_in_order level) init).

(* On the level that [Typing_env.cut] answers it comes to each variable
   once. *)
Definition cut_variables_in_order_distinct_statement : Prop :=
  ∀ (tenv : Typing_env.t) (cut_after : Scope.t) (level : Typing_env_level.t),
    typing_env_well_formed tenv = true ->
    Typing_env.cut tenv cut_after = Some level ->
    NoDup (Typing_env_level.variables_in_order level).

(* ======================================================================== *)
(* The functions keep the invariant                                         *)
(* ======================================================================== *)

Definition create_invariant_statement : Prop :=
  typing_env_invariant Typing_env.create.

Definition make_bottom_invariant_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_invariant tenv ->
    typing_env_invariant (Typing_env.make_bottom tenv).

Definition increment_scope_invariant_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_invariant tenv ->
    typing_env_invariant (Typing_env.increment_scope tenv).

Definition closure_env_invariant_statement : Prop :=
  ∀ tenv : Typing_env.t,
    typing_env_invariant tenv ->
    typing_env_invariant (Typing_env.closure_env tenv).

Definition add_symbol_definition_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol),
    typing_env_invariant tenv ->
    typing_env_invariant (Typing_env.add_symbol_definition tenv sym).

(* A variable is defined that is not bound. (The compiler refuses one
   that is bound when invariant checks are on.) *)
Definition add_variable_definition_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    typing_env_invariant tenv ->
    Typing_env.mem None tenv (Name_var var) = false ->
    typing_env_invariant
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* Parameters are defined: they are different variables, and none is
   bound. *)
Definition add_definitions_of_params_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (params : list (variable * flambda_kind)),
    typing_env_invariant tenv ->
    distinct (map fst params) = true ->
    (∀ x : variable,
      x ∈ map fst params -> Typing_env.mem None tenv (Name_var x) = false) ->
    typing_env_invariant (Typing_env.add_definitions_of_params tenv params).

(* A name gets a new type. The name is bound; the variables of the type
   are bound; the type has the kind of the one that the name had; if
   the type is an alias, the name is not canonical; and either the type
   is not obviously unknown, or no level before the current one has an
   equation on the name (discrepancy T2-4: a level drops an equation
   whose type is obviously unknown, and an equation of an earlier level
   would become the latest). [Meet_env] is to call it so. *)
Definition replace_equation_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    typing_env_invariant tenv ->
    Typing_env.mem None tenv n = true ->
    (∀ x : variable,
      x ∈ free_variables ty ->
      Typing_env.mem None tenv (Name_var x) = true) ->
    (∀ (ty' : flambda_type)
       (binding_time_and_mode : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
      = Some (ty', binding_time_and_mode) ->
      kind ty = kind ty') ->
    (∀ alias : simple,
      get_alias_opt ty = Some alias ->
      ~ is_canonical (Typing_env.aliases tenv) (simple_of_name n)) ->
    is_obviously_unknown ty = false
    ∨ Keyed.lookup name_eqb n
        (flat_map Typing_env_level.equations
           (map Typing_env.One_level.level (Typing_env.prev_levels tenv)))
      = None ->
    typing_env_invariant (Typing_env.replace_equation tenv n ty).

(* Two canonical elements that are bound are made aliases. *)
Definition add_alias_invariant_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    typing_env_invariant tenv ->
    is_canonical (Typing_env.aliases tenv) canonical_element1 ->
    is_canonical (Typing_env.aliases tenv) canonical_element2 ->
    Typing_env.mem_simple None tenv canonical_element1 = true ->
    Typing_env.mem_simple None tenv canonical_element2 = true ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    typing_env_invariant (Typing_env.add_alias_result_t result).

(* And [Typing_env.add_alias] answers for two canonical elements that
   are different: the fatal errors of [Aliases.add] do not happen. *)
Definition add_alias_defined_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple),
    typing_env_invariant tenv ->
    is_canonical (Typing_env.aliases tenv) canonical_element1 ->
    is_canonical (Typing_env.aliases tenv) canonical_element2 ->
    canonical_element1 <> canonical_element2 ->
    Typing_env.add_alias tenv canonical_element1 canonical_element2 <> None.

(* ======================================================================== *)
(* Well-formed types                                                        *)
(* ======================================================================== *)

(* Every type that the environment has for a name is well formed. *)
(* This is beside the invariant for now, and is to be a conjunct of it.
   The test of well-formed types is meant to be true of every type
   that the compiler builds (FlambdaTypeWellFormedSpec.v says, for each
   of its conditions, where the compiler establishes it and where
   nothing does). *)
Definition types_well_formed (tenv : Typing_env.t) : Prop :=
  ∀ (n : name) (ty : flambda_type)
    (binding_time_and_mode : Binding_time.With_name_mode.t),
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
    = Some (ty, binding_time_and_mode) ->
    type_well_formed ty = true.

Definition types_well_formed_b (tenv : Typing_env.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun _ entry => type_well_formed (fst entry))
    (Typing_env.names_to_types tenv).

Definition types_well_formed_b_iff_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_well_formed_b tenv = true <-> types_well_formed tenv.

(* The type found for a name is well formed. *)
Definition find_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (kind : option flambda_kind)
    (ty : flambda_type),
    types_well_formed tenv ->
    Typing_env.find tenv n kind = Some ty ->
    type_well_formed ty = true.

Definition create_types_well_formed_statement : Prop :=
  types_well_formed Typing_env.create.

Definition make_bottom_types_well_formed_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_well_formed tenv -> types_well_formed (Typing_env.make_bottom tenv).

Definition increment_scope_types_well_formed_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_well_formed tenv ->
    types_well_formed (Typing_env.increment_scope tenv).

Definition closure_env_types_well_formed_statement : Prop :=
  ∀ tenv : Typing_env.t,
    types_well_formed tenv -> types_well_formed (Typing_env.closure_env tenv).

Definition add_symbol_definition_types_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (sym : symbol),
    types_well_formed tenv ->
    types_well_formed (Typing_env.add_symbol_definition tenv sym).

Definition add_variable_definition_types_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (var : variable) (kind : flambda_kind)
    (name_mode : Name_mode.t),
    types_well_formed tenv ->
    types_well_formed
      (Typing_env.add_variable_definition tenv var kind name_mode).

(* A name gets a type that is well formed. *)
Definition replace_equation_types_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (n : name) (ty : flambda_type),
    type_well_formed ty = true ->
    types_well_formed tenv ->
    types_well_formed (Typing_env.replace_equation tenv n ty).

Definition add_alias_types_well_formed_statement : Prop :=
  ∀ (tenv : Typing_env.t) (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result),
    Typing_env.add_alias tenv canonical_element1 canonical_element2
    = Some (Ok result) ->
    types_well_formed tenv ->
    types_well_formed (Typing_env.add_alias_result_t result).
