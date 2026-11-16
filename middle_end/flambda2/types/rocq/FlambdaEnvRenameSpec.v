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


(* Renaming the variables of a typing environment: the action, the free
   variables, and that the functions of the environment commute with it.

   This file is to be reviewed: the action ([rename_level],
   [rename_cached_level], [rename_one_level], [rename_typing_env]), the
   free variables ([level_free_variables], [typing_env_free_variables]),
   the two instances of the interface for things that contain names, and
   the statements. The compiler has no [apply_renaming] for
   [Typing_env.t] or for a level. It has one for what a unit exports of
   its environment ([Typing_env.Serializable.apply_renaming], which
   renames each name and its type and keeps the binding time and the
   name mode, as [rename_cached_level] does of the same map), and one
   for an extension with its existential variables
   ([Typing_env_extension.With_extra_variables.apply_renaming]). The
   model has not the first structure; the second, with its action, is
   in FlambdaTypingEnvExtension.v. The action is here
   because everything that holds names is to have its renaming and its
   free names, with the laws of the interface for things that contain
   names (FlambdaNameAbstractionSpec.v), and the
   functions on it are to commute with renaming.

   A renaming (FlambdaNominalSpec.renaming) permutes variables; a typing
   environment holds no continuation. Symbols, constants, kinds, scopes,
   binding times and name modes are not renamed. The action renames
   every variable that the environment writes:
   - in a level, the variables that the level defines, in both of the
     maps that list them, and its equations: the name that each is on
     ([FlambdaTypeRenameSpec.rename_name]) and its type
     ([FlambdaTypeRenameSpec.rename_type]);
   - in what the environment knows of its names
     ([Cached_level.names_to_types]), each name and its type; the
     binding time and the name mode of a name stay with it;
   - the aliases, by the action of FlambdaEnvAliasesRenamingSpec.v
     ([rename_aliases]).

   The free variables of an environment are every variable written in
   it, also in a binding that is shadowed. A typing environment is the
   context of a term, and it records the identity of each variable
   that its levels define: those in scope in the term, and also those
   that no term may mention, which are the variables of the name modes
   [In_types] and [Phantom] and the variables bound before
   [min_binding_time], which are out of scope. All of them are free in
   the environment: the action renames them and abstracts none. They
   are not bound by it. (Where a level is taken out of
   an environment and its variables are to be made anew, as for the
   levels that a join is given, the level with what it scopes over is
   a name abstraction; that is not stated here.) So the equality of the
   two instances is that of Rocq, as for types and for the aliases.

   The statements:
   - the renamings act on levels and on environments as a group, two
     renamings that rename the free variables alike rename alike, and
     one that fixes them changes nothing; the free variables of what is
     renamed are the renamed free variables. So a level and an
     environment are things that contain names, with the laws of the
     interface ([level_names_nominal_statement],
     [typing_env_names_nominal_statement]);
   - the functions of a level ([Typing_env_level]) and of what the
     environment knows of its names ([Cached_level]) commute with
     renaming;
   - so does what an environment is asked ([Typing_env]): [find] and
     the functions beside it, the binding time and the name mode of a
     name, [mem], and the canonical element of a [Simple] at a name
     mode;
   - and so does what is done to an environment: the definition of a
     variable or of a symbol, [replace_equation], [add_alias], [cut],
     and the changes of scope;
   - the renamed valuation satisfies the renamed environment exactly
     when the valuation satisfies the environment, for each of the
     senses of FlambdaTypingEnvSpec.v. The values are not renamed: a
     value holds no variable;
   - the renamed environment has the invariant of
     FlambdaTypingEnvInvariantSpec.v exactly when the environment has
     it, and the test of the invariant answers the same of the two; the
     same of its types being well formed, with the test of that, and of
     its being kinded, in the two senses of FlambdaTypingEnvKindedSpec.v.

   Each statement about a function is an equation and holds of every
   level or environment: nothing is asked of it, its invariant
   included. One has a premise, on an argument that is a function
   ([Typing_env_level.fold_on_defined_vars]). No function here makes a
   name, and none goes by the order of names. Where the order in which
   names were bound matters (the variables of a level in order, the
   choice of a canonical element and of the name that is demoted), it
   is read from binding times, which a name keeps when it is renamed;
   and a tie is settled by the order of a list of the model, which the
   action keeps. The compiler settles such a tie by an order on names
   (FlambdaEnvAliasesSpec.v, "Differences"; FlambdaTypingEnv.v, on
   [fold_on_defined_vars]).

   Nothing is trusted here. The proofs are in FlambdaEnvRenameProofs.v,
   which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaKindModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaNameAbstractionSpec FlambdaNominalInstancesSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaEnvAliasesRenamingSpec FlambdaTypingEnv
  FlambdaTypingEnvSpec FlambdaTypingEnvInvariantSpec
  FlambdaTypingEnvKindedSpec FlambdaTypeRenameSpec FlambdaTypeNominalSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* The action                                                               *)
(* ======================================================================== *)

(* A level: the variables it defines, in the map of their kinds and in
   the lists by binding time, and its equations. *)
Definition rename_level (r : renaming) (level : Typing_env_level.t)
    : Typing_env_level.t :=
  Typing_env_level.create
    (Keyed.map_keys (rename_variable r) (Typing_env_level.defined_vars level))
    (Keyed.map_values (map (rename_variable r))
       (Typing_env_level.binding_times level))
    (Keyed.map_bindings (rename_name r) (rename_type r)
       (Typing_env_level.equations level)).

(* What the environment knows of a name: the type is renamed, and the
   binding time and the name mode are kept. *)
Definition rename_binding (r : renaming)
    (binding : flambda_type * Binding_time.With_name_mode.t)
    : flambda_type * Binding_time.With_name_mode.t :=
  (rename_type r (fst binding), snd binding).

Definition rename_cached_level (r : renaming) (cached : Cached_level.t)
    : Cached_level.t :=
  {| Cached_level.names_to_types :=
       Keyed.map_bindings (rename_name r) (rename_binding r)
         (Cached_level.names_to_types cached);
     Cached_level.aliases := rename_aliases r (Cached_level.aliases cached) |}.

Definition rename_one_level (r : renaming)
    (one_level : Typing_env.One_level.t) : Typing_env.One_level.t :=
  Typing_env.One_level.create (Typing_env.One_level.scope one_level)
    (rename_level r (Typing_env.One_level.level one_level)).

Definition rename_typing_env (r : renaming) (tenv : Typing_env.t)
    : Typing_env.t :=
  {| Typing_env.defined_symbols := Typing_env.defined_symbols tenv;
     Typing_env.prev_levels :=
       map (rename_one_level r) (Typing_env.prev_levels tenv);
     Typing_env.current_level :=
       rename_one_level r (Typing_env.current_level tenv);
     Typing_env.just_after_level :=
       rename_cached_level r (Typing_env.just_after_level tenv);
     Typing_env.next_binding_time := Typing_env.next_binding_time tenv;
     Typing_env.min_binding_time := Typing_env.min_binding_time tenv;
     Typing_env.is_bottom := Typing_env.is_bottom tenv |}.

(* ======================================================================== *)
(* Free variables                                                           *)
(* ======================================================================== *)

(* The variables written in a level. A variable may be listed more than
   once, as for a type (FlambdaTypeGrammar.free_variables). *)
Definition level_free_variables (level : Typing_env_level.t)
    : list variable :=
  map fst (Typing_env_level.defined_vars level)
  ++ flat_map snd (Typing_env_level.binding_times level)
  ++ flat_map
       (fun equation =>
          free_variables_of_name (fst equation)
          ++ free_variables (snd equation))
       (Typing_env_level.equations level).

Definition cached_level_free_variables (cached : Cached_level.t)
    : list variable :=
  flat_map
    (fun binding =>
       free_variables_of_name (fst binding)
       ++ free_variables (fst (snd binding)))
    (Cached_level.names_to_types cached)
  ++ aliases_free_variables (Cached_level.aliases cached).

Definition typing_env_free_variables (tenv : Typing_env.t) : list variable :=
  flat_map
    (fun one_level =>
       level_free_variables (Typing_env.One_level.level one_level))
    (Typing_env.prev_levels tenv)
  ++ level_free_variables
       (Typing_env.One_level.level (Typing_env.current_level tenv))
  ++ cached_level_free_variables (Typing_env.just_after_level tenv).

(* ======================================================================== *)
(* The renamings act on levels and on environments                          *)
(* ======================================================================== *)

Definition rename_level_identity_statement : Prop :=
  ∀ level : Typing_env_level.t, rename_level identity_renaming level = level.

Definition rename_level_compose_statement : Prop :=
  ∀ (second first : renaming) (level : Typing_env_level.t),
    rename_level (compose_renaming second first) level
    = rename_level second (rename_level first level).

(* Two renamings that rename the free variables of a level alike rename
   the level alike. *)
Definition rename_level_extensional_statement : Prop :=
  ∀ (r r' : renaming) (level : Typing_env_level.t),
    (∀ x : variable,
      x ∈ level_free_variables level ->
      rename_variable r x = rename_variable r' x) ->
    rename_level r level = rename_level r' level.

(* So a renaming that fixes the free variables of a level leaves it as
   it is. *)
Definition rename_level_fixes_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    (∀ x : variable, x ∈ level_free_variables level -> rename_variable r x = x)
    ->
    rename_level r level = level.

Definition rename_level_inverse_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    rename_level (inverse_renaming r) (rename_level r level) = level.

Definition free_variables_rename_level_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    level_free_variables (rename_level r level)
    = map (rename_variable r) (level_free_variables level).

(* The same of an environment. *)
Definition rename_typing_env_identity_statement : Prop :=
  ∀ tenv : Typing_env.t, rename_typing_env identity_renaming tenv = tenv.

Definition rename_typing_env_compose_statement : Prop :=
  ∀ (second first : renaming) (tenv : Typing_env.t),
    rename_typing_env (compose_renaming second first) tenv
    = rename_typing_env second (rename_typing_env first tenv).

Definition rename_typing_env_extensional_statement : Prop :=
  ∀ (r r' : renaming) (tenv : Typing_env.t),
    (∀ x : variable,
      x ∈ typing_env_free_variables tenv ->
      rename_variable r x = rename_variable r' x) ->
    rename_typing_env r tenv = rename_typing_env r' tenv.

Definition rename_typing_env_fixes_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    (∀ x : variable,
      x ∈ typing_env_free_variables tenv -> rename_variable r x = x) ->
    rename_typing_env r tenv = tenv.

Definition rename_typing_env_inverse_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    rename_typing_env (inverse_renaming r) (rename_typing_env r tenv) = tenv.

Definition free_variables_rename_typing_env_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    typing_env_free_variables (rename_typing_env r tenv)
    = map (rename_variable r) (typing_env_free_variables tenv).

(* ======================================================================== *)
(* Levels and environments are things that contain names                    *)
(* ======================================================================== *)

(* [Contains_names.S] at the sort of variables: a permutation of the
   variables acts by the renaming that permutes no continuation
   (FlambdaNominalInstancesSpec.variables_renaming), the free names are the
   free variables, and the equality is that of Rocq. *)
Definition level_names : contains_names Typing_env_level.t :=
  Contains_names (fun p level => rename_level (variables_renaming p) level)
    level_free_variables eq.

Definition typing_env_names : contains_names Typing_env.t :=
  Contains_names (fun p tenv => rename_typing_env (variables_renaming p) tenv)
    typing_env_free_variables eq.

Definition level_names_nominal_statement : Prop := nominal level_names.

Definition typing_env_names_nominal_statement : Prop :=
  nominal typing_env_names.

(* ======================================================================== *)
(* The functions of a level                                                 *)
(* ======================================================================== *)

Definition level_empty_equivariant_statement : Prop :=
  ∀ r : renaming,
    rename_level r Typing_env_level.empty = Typing_env_level.empty.

Definition level_defined_variables_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    Typing_env_level.defined_variables (rename_level r level)
    = map (rename_variable r) (Typing_env_level.defined_variables level).

Definition level_defined_names_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    Typing_env_level.defined_names (rename_level r level)
    = map (rename_name r) (Typing_env_level.defined_names level).

Definition level_find_kind_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t) (var : variable),
    Typing_env_level.find_kind (rename_level r level) (rename_variable r var)
    = Typing_env_level.find_kind level var.

Definition level_variable_is_defined_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t) (var : variable),
    Typing_env_level.variable_is_defined (rename_level r level)
      (rename_variable r var)
    = Typing_env_level.variable_is_defined level var.

Definition level_is_empty_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    Typing_env_level.is_empty (rename_level r level)
    = Typing_env_level.is_empty level.

Definition level_add_definition_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t) (var : variable)
    (kind : flambda_kind) (binding_time : Binding_time.t),
    Typing_env_level.add_definition (rename_level r level)
      (rename_variable r var) kind binding_time
    = rename_level r
        (Typing_env_level.add_definition level var kind binding_time).

Definition level_add_or_replace_equation_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t) (name : name)
    (ty : flambda_type),
    Typing_env_level.add_or_replace_equation (rename_level r level)
      (rename_name r name) (rename_type r ty)
    = rename_level r (Typing_env_level.add_or_replace_equation level name ty).

(* Where the two levels cannot be made one, the two renamed cannot. *)
Definition level_concat_equivariant_statement : Prop :=
  ∀ (r : renaming) (earlier later : Typing_env_level.t),
    Typing_env_level.concat (rename_level r earlier) (rename_level r later)
    = option_map (rename_level r) (Typing_env_level.concat earlier later).

Definition level_as_extension_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    Typing_env_level.as_extension_without_bindings (rename_level r level)
    = rename_env_extension r
        (Typing_env_level.as_extension_without_bindings level).

(* The variables of a level in the order of their binding times, and
   within one binding time in the order of the model's list: a renaming
   keeps both. *)
Definition level_variables_in_order_equivariant_statement : Prop :=
  ∀ (r : renaming) (level : Typing_env_level.t),
    Typing_env_level.variables_in_order (rename_level r level)
    = map (rename_variable r) (Typing_env_level.variables_in_order level).

(* The fold over the variables that a level defines, for a function that
   commutes with the renaming of the variable it is given and of what it
   accumulates. *)
Definition level_fold_on_defined_vars_equivariant_statement : Prop :=
  ∀ (A : Type) (r : renaming) (rename_acc : A -> A)
    (f f' : variable -> flambda_kind -> A -> A) (level : Typing_env_level.t)
    (init : A),
    (∀ (var : variable) (kind : flambda_kind) (acc : A),
      f' (rename_variable r var) kind (rename_acc acc)
      = rename_acc (f var kind acc)) ->
    Typing_env_level.fold_on_defined_vars f' (rename_level r level)
      (rename_acc init)
    = option_map rename_acc
        (Typing_env_level.fold_on_defined_vars f level init).

(* ======================================================================== *)
(* What the environment knows of its names                                  *)
(* ======================================================================== *)

Definition cached_level_empty_equivariant_statement : Prop :=
  ∀ r : renaming, rename_cached_level r Cached_level.empty = Cached_level.empty.

Definition cached_level_add_or_replace_binding_equivariant_statement : Prop :=
  ∀ (r : renaming) (cached : Cached_level.t) (name : name)
    (ty : flambda_type) (binding_time : Binding_time.t)
    (name_mode : Name_mode.t),
    Cached_level.add_or_replace_binding (rename_cached_level r cached)
      (rename_name r name) (rename_type r ty) binding_time name_mode
    = rename_cached_level r
        (Cached_level.add_or_replace_binding cached name ty binding_time
           name_mode).

Definition cached_level_replace_variable_binding_equivariant_statement : Prop :=
  ∀ (r : renaming) (cached : Cached_level.t) (var : variable)
    (ty : flambda_type),
    Cached_level.replace_variable_binding (rename_cached_level r cached)
      (rename_variable r var) (rename_type r ty)
    = rename_cached_level r
        (Cached_level.replace_variable_binding cached var ty).

Definition cached_level_with_aliases_equivariant_statement : Prop :=
  ∀ (r : renaming) (cached : Cached_level.t) (aliases : Aliases.t),
    Cached_level.with_aliases (rename_cached_level r cached)
      (rename_aliases r aliases)
    = rename_cached_level r (Cached_level.with_aliases cached aliases).

(* ======================================================================== *)
(* What an environment is asked                                             *)
(* ======================================================================== *)

Definition create_equivariant_statement : Prop :=
  ∀ r : renaming, rename_typing_env r Typing_env.create = Typing_env.create.

Definition aliases_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.aliases (rename_typing_env r tenv)
    = rename_aliases r (Typing_env.aliases tenv).

Definition names_to_types_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.names_to_types (rename_typing_env r tenv)
    = Keyed.map_bindings (rename_name r) (rename_binding r)
        (Typing_env.names_to_types tenv).

Definition current_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.current_scope (rename_typing_env r tenv)
    = Typing_env.current_scope tenv.

Definition name_domain_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.name_domain (rename_typing_env r tenv)
    = map (rename_name r) (Typing_env.name_domain tenv).

(* [Typing_env.find_with_binding_time_and_mode'], the one that does not
   scope the name mode: the type found is the renamed type, with the
   same binding time and name mode; where the one is a fatal error the
   other is. *)
Definition find_unscoped_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name)
    (kind : option flambda_kind),
    Typing_env.find_with_binding_time_and_mode' (rename_typing_env r tenv)
      (rename_name r name) kind
    = option_map (rename_binding r)
        (Typing_env.find_with_binding_time_and_mode' tenv name kind).

Definition find_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name)
    (kind : option flambda_kind),
    Typing_env.find (rename_typing_env r tenv) (rename_name r name) kind
    = option_map (rename_type r) (Typing_env.find tenv name kind).

Definition find_with_binding_time_and_mode_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name)
    (kind : option flambda_kind),
    Typing_env.find_with_binding_time_and_mode (rename_typing_env r tenv)
      (rename_name r name) kind
    = option_map (rename_binding r)
        (Typing_env.find_with_binding_time_and_mode tenv name kind).

(* A name keeps its binding time and its name mode. *)
Definition binding_time_and_mode_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name),
    Typing_env.binding_time_and_mode (rename_typing_env r tenv)
      (rename_name r name)
    = Typing_env.binding_time_and_mode tenv name.

Definition binding_time_and_mode_of_simple_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (simple : simple),
    Typing_env.binding_time_and_mode_of_simple (rename_typing_env r tenv)
      (rename_simple r simple)
    = Typing_env.binding_time_and_mode_of_simple tenv simple.

Definition binding_time_resolver_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name),
    Typing_env.binding_time_resolver (rename_typing_env r tenv)
      (rename_name r name)
    = Typing_env.binding_time_resolver tenv name.

Definition mem_equivariant_statement : Prop :=
  ∀ (r : renaming) (min_name_mode : option Name_mode.t) (tenv : Typing_env.t)
    (name : name),
    Typing_env.mem min_name_mode (rename_typing_env r tenv)
      (rename_name r name)
    = Typing_env.mem min_name_mode tenv name.

Definition mem_simple_equivariant_statement : Prop :=
  ∀ (r : renaming) (min_name_mode : option Name_mode.t) (tenv : Typing_env.t)
    (simple : simple),
    Typing_env.mem_simple min_name_mode (rename_typing_env r tenv)
      (rename_simple r simple)
    = Typing_env.mem_simple min_name_mode tenv simple.

Definition alias_is_bound_strictly_earlier_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (bound_name : name) (alias : simple),
    Typing_env.alias_is_bound_strictly_earlier (rename_typing_env r tenv)
      (rename_name r bound_name) (rename_simple r alias)
    = Typing_env.alias_is_bound_strictly_earlier tenv bound_name alias.

(* The canonical element of a renamed [Simple] is the renamed canonical
   element. The order that chooses it is by binding times, which names
   keep; where two are equally early the model takes the first of a
   list, which a renaming keeps (FlambdaEnvAliasesRenamingSpec.v). *)
Definition get_canonical_ignoring_name_mode_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (simple : simple),
    Typing_env.get_canonical_simple_ignoring_name_mode
      (rename_typing_env r tenv) (rename_simple r simple)
    = rename_simple r
        (Typing_env.get_canonical_simple_ignoring_name_mode tenv simple).

Definition get_canonical_simple_exn_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t)
    (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
    (simple : simple),
    Typing_env.get_canonical_simple_exn (rename_typing_env r tenv)
      min_name_mode name_mode_of_existing_simple (rename_simple r simple)
    = option_map (option_map (rename_simple r))
        (Typing_env.get_canonical_simple_exn tenv min_name_mode
           name_mode_of_existing_simple simple).

Definition get_alias_then_canonical_simple_exn_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t)
    (min_name_mode name_mode_of_existing_simple : option Name_mode.t)
    (typ : flambda_type),
    Typing_env.get_alias_then_canonical_simple_exn (rename_typing_env r tenv)
      min_name_mode name_mode_of_existing_simple (rename_type r typ)
    = option_map (option_map (rename_simple r))
        (Typing_env.get_alias_then_canonical_simple_exn tenv min_name_mode
           name_mode_of_existing_simple typ).

(* The type of a [Simple] in a term, with the canonical element that the
   type is an alias to. *)
Definition type_simple_in_term_exn_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (min_name_mode : option Name_mode.t)
    (simple : simple),
    Typing_env.type_simple_in_term_exn (rename_typing_env r tenv)
      min_name_mode (rename_simple r simple)
    = option_map
        (option_map
           (fun typed =>
              (rename_type r (fst typed), rename_simple r (snd typed))))
        (Typing_env.type_simple_in_term_exn tenv min_name_mode simple).

(* ======================================================================== *)
(* What is done to an environment                                           *)
(* ======================================================================== *)

Definition make_bottom_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.make_bottom (rename_typing_env r tenv)
    = rename_typing_env r (Typing_env.make_bottom tenv).

Definition increment_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.increment_scope (rename_typing_env r tenv)
    = rename_typing_env r (Typing_env.increment_scope tenv).

Definition bump_current_level_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.bump_current_level_scope (rename_typing_env r tenv)
    = rename_typing_env r (Typing_env.bump_current_level_scope tenv).

Definition closure_env_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    Typing_env.closure_env (rename_typing_env r tenv)
    = rename_typing_env r (Typing_env.closure_env tenv).

Definition with_aliases_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (aliases : Aliases.t),
    Typing_env.with_aliases (rename_typing_env r tenv)
      (rename_aliases r aliases)
    = rename_typing_env r (Typing_env.with_aliases tenv aliases).

(* A variable is defined: the one given, renamed. The environment makes
   no name of its own. *)
Definition add_variable_definition_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (var : variable)
    (kind : flambda_kind) (name_mode : Name_mode.t),
    Typing_env.add_variable_definition (rename_typing_env r tenv)
      (rename_variable r var) kind name_mode
    = rename_typing_env r
        (Typing_env.add_variable_definition tenv var kind name_mode).

Definition add_symbol_definition_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (sym : symbol),
    Typing_env.add_symbol_definition (rename_typing_env r tenv) sym
    = rename_typing_env r (Typing_env.add_symbol_definition tenv sym).

Definition add_symbol_definitions_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (syms : list symbol),
    Typing_env.add_symbol_definitions (rename_typing_env r tenv) syms
    = rename_typing_env r (Typing_env.add_symbol_definitions tenv syms).

Definition add_definition_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name)
    (name_mode : Name_mode.t) (kind : flambda_kind),
    Typing_env.add_definition (rename_typing_env r tenv) (rename_name r name)
      name_mode kind
    = option_map (rename_typing_env r)
        (Typing_env.add_definition tenv name name_mode kind).

Definition add_definitions_of_params_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t)
    (params : list (variable * flambda_kind)),
    Typing_env.add_definitions_of_params (rename_typing_env r tenv)
      (map (fun param => (rename_variable r (fst param), snd param)) params)
    = rename_typing_env r (Typing_env.add_definitions_of_params tenv params).

Definition replace_equation_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (name : name) (ty : flambda_type),
    Typing_env.replace_equation (rename_typing_env r tenv)
      (rename_name r name) (rename_type r ty)
    = rename_typing_env r (Typing_env.replace_equation tenv name ty).

(* Two canonical elements are made aliases. Which of the two is demoted
   goes by their binding times, which names keep. *)
Definition aliases_add_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple),
    Typing_env.aliases_add (rename_typing_env r tenv)
      (rename_simple r canonical_element1)
      (rename_simple r canonical_element2)
    = option_map (or_bottom_map (rename_add_result r))
        (Typing_env.aliases_add tenv canonical_element1 canonical_element2).

Definition rename_add_alias_result (r : renaming)
    (result : Typing_env.add_alias_result) : Typing_env.add_alias_result :=
  {| Typing_env.canonical_element :=
       rename_simple r (Typing_env.canonical_element result);
     Typing_env.demoted_name := rename_name r (Typing_env.demoted_name result);
     Typing_env.add_alias_result_t :=
       rename_typing_env r (Typing_env.add_alias_result_t result) |}.

Definition add_alias_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple),
    Typing_env.add_alias (rename_typing_env r tenv)
      (rename_simple r canonical_element1)
      (rename_simple r canonical_element2)
    = option_map (or_unknown_or_bottom_map (rename_add_alias_result r))
        (Typing_env.add_alias tenv canonical_element1 canonical_element2).

(* The levels after a scope, as one level: that of the renamed
   environment is the renamed level. *)
Definition cut_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (cut_after : Scope.t),
    Typing_env.cut (rename_typing_env r tenv) cut_after
    = option_map (rename_level r) (Typing_env.cut tenv cut_after).

Definition cut_as_extension_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (cut_after : Scope.t),
    Typing_env.cut_as_extension (rename_typing_env r tenv) cut_after
    = option_map (rename_env_extension r)
        (Typing_env.cut_as_extension tenv cut_after).

(* ======================================================================== *)
(* What satisfies a renamed environment                                     *)
(* ======================================================================== *)

(* A valuation is renamed as the variables of the machine are
   ([FlambdaAlphaSpec.rename_env]): each variable is renamed and keeps
   its value. *)

(* The names that are exempt, renamed: a name is exempt when the name
   that it was before the renaming is. *)
Definition rename_exempt (r : renaming) (exempt : name -> Prop) (n : name)
    : Prop :=
  exempt (rename_name (inverse_renaming r) n).

(* The names that are exempt where poison is, for the renamed valuation
   and the renamed environment, are those that were, renamed. *)
Definition exempt_by_poison_rename_statement : Prop :=
  ∀ (r : renaming) (valuation : env) (tenv : Typing_env.t) (n : name),
    exempt_by_poison (rename_env r valuation) (rename_typing_env r tenv)
      (rename_name r n)
    <-> exempt_by_poison valuation tenv n.

(* The renamed valuation satisfies the renamed environment, but for the
   renamed exempt names, exactly when the valuation satisfies the
   environment but for the exempt names. *)
Definition typing_env_satisfied_but_rename_statement : Prop :=
  ∀ (r : renaming) (exempt : name -> Prop) (statics : static_env)
    (valuation : env) (tenv : Typing_env.t),
    typing_env_satisfied_but (rename_exempt r exempt) statics
      (rename_env r valuation) (rename_typing_env r tenv)
    <-> typing_env_satisfied_but exempt statics valuation tenv.

(* The same with no name exempt, and with the names exempt that are
   where poison is. *)
Definition typing_env_satisfied_strictly_rename_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t),
    typing_env_satisfied_strictly statics (rename_env r valuation)
      (rename_typing_env r tenv)
    <-> typing_env_satisfied_strictly statics valuation tenv.

Definition typing_env_satisfied_rename_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (valuation : env)
    (tenv : Typing_env.t),
    typing_env_satisfied statics (rename_env r valuation)
      (rename_typing_env r tenv)
    <-> typing_env_satisfied statics valuation tenv.

(* A variable may occur in a term under the renamed environment exactly
   when the variable that it was may under the environment. *)
Definition in_scope_in_terms_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (x : variable),
    in_scope_in_terms (rename_typing_env r tenv) (rename_variable r x)
    <-> in_scope_in_terms tenv x.

(* The renamed variables of the machine satisfy the renamed environment
   exactly when the variables satisfy the environment. *)
Definition env_satisfies_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies (rename_typing_env r tenv) statics (rename_env r vars)
    <-> env_satisfies tenv statics vars.

Definition env_satisfies_strictly_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (statics : static_env) (vars : env),
    env_satisfies_strictly (rename_typing_env r tenv) statics
      (rename_env r vars)
    <-> env_satisfies_strictly tenv statics vars.

(* ======================================================================== *)
(* The invariant of a renamed environment                                   *)
(* ======================================================================== *)

(* The test of the invariant of an environment
   (FlambdaTypingEnvInvariantSpec.v) answers the same of the renamed
   environment. So does the condition that has no test, that the latest
   equation on a name is the type that the environment has for it; and
   the invariant, which is the two together. *)
Definition typing_env_well_formed_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    typing_env_well_formed (rename_typing_env r tenv)
    = typing_env_well_formed tenv.

Definition levels_agree_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    levels_agree (rename_typing_env r tenv) <-> levels_agree tenv.

Definition typing_env_invariant_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    typing_env_invariant (rename_typing_env r tenv)
    <-> typing_env_invariant tenv.

(* The types of the renamed environment are well formed exactly when
   those of the environment are. *)
Definition types_well_formed_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    types_well_formed (rename_typing_env r tenv) <-> types_well_formed tenv.

(* The test of that condition answers the same of the two. *)
Definition types_well_formed_b_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    types_well_formed_b (rename_typing_env r tenv) = types_well_formed_b tenv.

(* ======================================================================== *)
(* The kinds of a renamed environment                                       *)
(* ======================================================================== *)

(* A renamed [Simple] is of a kind in the renamed environment exactly
   when the [Simple] is of that kind in the environment
   (FlambdaTypingEnvKindedSpec.v); and the renamed environment is
   kinded exactly when the environment is. *)
Definition simple_of_kind_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (s : simple) (kind : flambda_kind),
    simple_of_kind (rename_typing_env r tenv) (rename_simple r s) kind
    <-> simple_of_kind tenv s kind.

Definition typing_env_kinded_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    typing_env_kinded (rename_typing_env r tenv) <-> typing_env_kinded tenv.

(* The second condition on kinds of that file, of the names that have
   an equation in an extension inside a type: it holds of a renamed
   type in the renamed environment exactly when it holds of the type
   in the environment, and of the renamed environment exactly when it
   holds of the environment. *)
Definition type_equations_kinded_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (ty : flambda_type),
    type_equations_kinded (rename_typing_env r tenv) (rename_type r ty)
    <-> type_equations_kinded tenv ty.

Definition types_equations_kinded_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t),
    types_equations_kinded (rename_typing_env r tenv)
    <-> types_equations_kinded tenv.
