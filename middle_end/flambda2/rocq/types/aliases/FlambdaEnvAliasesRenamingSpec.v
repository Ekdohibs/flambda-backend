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

(* Renaming the aliases of the typing environment, and their free
   variables.

   This file is to be reviewed: the renamings ([rename_aliases],
   [rename_binding_times], [rename_resolver], [rename_alias_set],
   [rename_add_result]), [aliases_free_variables], and the statements.
   The compiler has nothing to compare them with: [Aliases.t] has
   neither [apply_renaming] nor [free_names] there. They are here
   because everything that holds names is to have its renaming and its
   free names, and the functions on it are to commute with renaming
   (FlambdaNominalSpec.v; the compiler's directory nominal/). A name is
   renamed by FlambdaTypeRenameSpec.rename_name, and a map has its keys
   renamed by FlambdaMapSpec.Keyed.map_bindings.

   A renaming (FlambdaNominalSpec.renaming) permutes variables. It does
   not touch symbols or constants. The structure is renamed in all three
   of its maps, keys and values. What the functions of the aliases take
   of the typing environment is renamed with it: the map of binding
   times and name modes in its keys, and the resolver by the inverse
   renaming of its argument, so that a name keeps its binding time and
   its name mode when it is renamed.

   The free variables of the structure are every variable written in it:
   in the keys and in the values of the three maps, in a binding that is
   shadowed and in the key of an empty set of aliases too. Nothing in
   the structure binds a name. They are more than the variables the
   structure speaks of ([aliases_mention], FlambdaEnvAliasesSpec.v),
   which are those that a query can show: a renaming that fixes the
   variables the structure speaks of may still change it, as written,
   where it renames a key that no query reads. The laws of the renaming
   are stated with the first; what the structure means depends on the
   second only (FlambdaEnvAliasesSpec.v, [respected_coincide]).

   The statements:
   - the renamings of the structure act as the renamings do: identity,
     composition, and two renamings that rename every variable alike
     rename a structure alike;
   - the free variables of the renamed structure are the renamed free
     variables, and a renaming that fixes the free variables of a
     structure leaves it as it is, with [=];
   - the variables the structure speaks of are among its free
     variables, and the names the renamed structure speaks of are the
     renamed names;
   - the equality test of simples, the order, the queries and
     [Aliases.add] commute with renaming; respect and the invariant are
     kept by it, and the compiler's two checks ([Aliases.invariant],
     [Aliases.invariant_add_result]) answer the same.
   No statement about a function has a premise: the functions compare
   names for equality only and read binding times through the map and
   the resolver. In particular where two aliases are equally early the
   model takes the first of a list, which renaming keeps; the compiler
   takes the one that an order on names puts first, which renaming need
   not keep (FlambdaEnvAliasesSpec.v, "Differences").

   The structure is an instance, for variables and with [=] as its
   equality, of the interface for things that contain names
   (FlambdaNameAbstractionSpec.v, after the compiler's
   [Contains_names.S]): [aliases_contains_names], which has the laws of
   that interface ([aliases_nominal_statement]).

   The proofs are in FlambdaEnvAliasesRenamingProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaNameAbstractionSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypeRenameSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The renamings                                                            *)
(* ======================================================================== *)

Definition rename_aliases_of_canonical_element (r : renaming)
    (aliases : Aliases.Aliases_of_canonical_element.t)
    : Aliases.Aliases_of_canonical_element.t :=
  Keyed.map_bindings (rename_name r) (fun name_mode => name_mode) aliases.

(* The structure renamed: every name in it, in the three maps. *)
Definition rename_aliases (r : renaming) (t : Aliases.t) : Aliases.t :=
  Aliases.make
    (Keyed.map_bindings (rename_name r) (rename_simple r)
       (Aliases.canonical_elements t))
    (Keyed.map_bindings (rename_name r) (rename_aliases_of_canonical_element r)
       (Aliases.aliases_of_canonical_names t))
    (Keyed.map_values (rename_aliases_of_canonical_element r)
       (Aliases.aliases_of_consts t)).

(* The binding times and name modes of the typing environment, each now
   that of the renamed name. *)
Definition rename_binding_times {A : Type} (r : renaming)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    : Keyed.finite_map name (A * Binding_time.With_name_mode.t) :=
  Keyed.map_bindings (rename_name r) (fun entry => entry)
    binding_times_and_modes.

Definition rename_resolver (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    : name -> Binding_time.With_name_mode.t :=
  fun n => binding_time_resolver (rename_name (inverse_renaming r) n).

(* The result of [Aliases.add], renamed. *)
Definition rename_add_result (r : renaming) (result : Aliases.add_result)
    : Aliases.add_result :=
  Aliases.make_add_result
    (rename_aliases r (Aliases.add_result_t result))
    (rename_name r (Aliases.demoted_name result))
    (rename_simple r (Aliases.canonical_element result)).

(* A set of aliases, renamed. *)
Definition rename_alias_set (r : renaming) (set : Aliases.Alias_set.t)
    : Aliases.Alias_set.t :=
  Aliases.Alias_set.make (Aliases.Alias_set.set_const set)
    (map (rename_name r) (Aliases.Alias_set.set_names set)).

(* ======================================================================== *)
(* Free variables                                                           *)
(* ======================================================================== *)

Definition free_variables_of_aliases_of_canonical_element
    (aliases : Aliases.Aliases_of_canonical_element.t) : list variable :=
  flat_map (fun binding => free_variables_of_name (fst binding)) aliases.

(* The variables written in the structure. A variable may be listed more
   than once, and the variables of a binding that is shadowed are listed
   too, as for a type (FlambdaTypeGrammar.v, [free_variables]). *)
Definition aliases_free_variables (t : Aliases.t) : list variable :=
  flat_map
    (fun binding =>
       free_variables_of_name (fst binding)
       ++ free_variables_of_simple (snd binding))
    (Aliases.canonical_elements t)
  ++ flat_map
       (fun binding =>
          free_variables_of_name (fst binding)
          ++ free_variables_of_aliases_of_canonical_element (snd binding))
       (Aliases.aliases_of_canonical_names t)
  ++ flat_map
       (fun binding =>
          free_variables_of_aliases_of_canonical_element (snd binding))
       (Aliases.aliases_of_consts t).

(* ======================================================================== *)
(* The renamings of a structure act as the renamings do                     *)
(* ======================================================================== *)

Definition rename_aliases_identity_statement : Prop :=
  ∀ t : Aliases.t, rename_aliases identity_renaming t = t.

Definition rename_aliases_compose_statement : Prop :=
  ∀ (second first : renaming) (t : Aliases.t),
    rename_aliases (compose_renaming second first) t
    = rename_aliases second (rename_aliases first t).

Definition rename_aliases_extensional_statement : Prop :=
  ∀ (r r' : renaming) (t : Aliases.t),
    (∀ x : variable, rename_variable r x = rename_variable r' x) ->
    rename_aliases r t = rename_aliases r' t.

(* The free variables of the renamed structure are the renamed free
   variables. *)
Definition rename_free_variables_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t),
    aliases_free_variables (rename_aliases r t)
    = map (rename_variable r) (aliases_free_variables t).

(* A renaming that fixes the free variables of a structure leaves it as
   it is. *)
Definition rename_aliases_fixes_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t),
    (∀ x : variable,
      among x (aliases_free_variables t) = true -> rename_variable r x = x) ->
    rename_aliases r t = t.

(* The variables that the structure speaks of are among its free
   variables. The converse fails: [x] is free in the structure whose
   only binding gives [x] no alias,
   [Aliases.make [] [(Name_var x, [])] []], which speaks of no name. *)
Definition mentioned_free_statement : Prop :=
  ∀ (t : Aliases.t) (x : variable),
    aliases_mention t (Name_var x) ->
    among x (aliases_free_variables t) = true.

(* The names that the renamed structure speaks of are the renamed
   names. *)
Definition rename_mentioned_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t) (n : name),
    aliases_mention (rename_aliases r t) (rename_name r n)
    <-> aliases_mention t n.

(* ======================================================================== *)
(* The functions commute with renaming                                      *)
(* ======================================================================== *)

(* The equality test of simples. *)
Definition rename_simple_equal_statement : Prop :=
  ∀ (r : renaming) (s s' : simple),
    simple_equal (rename_simple r s) (rename_simple r s') = simple_equal s s'.

(* The order. *)
Definition rename_defined_earlier_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e e' : simple),
    Aliases.defined_earlier (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes) (rename_simple r e)
      (rename_simple r e')
    = Aliases.defined_earlier binding_time_resolver binding_times_and_modes
        e e'.

(* The canonical element. *)
Definition rename_canonical_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t) (n : name),
    Aliases.get_canonical_ignoring_name_mode (rename_aliases r t)
      (rename_name r n)
    = rename_simple r (Aliases.get_canonical_ignoring_name_mode t n).

(* The aliases of an element. *)
Definition rename_get_aliases_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t) (element : simple),
    Aliases.get_aliases (rename_aliases r t) (rename_simple r element)
    = rename_alias_set r (Aliases.get_aliases t element).

(* The canonical element at a name mode. *)
Definition rename_canonical_at_mode_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t),
    Aliases.get_canonical_element_exn (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
      (rename_simple r element) elt_name_mode min_name_mode min_binding_time
    = option_map (rename_simple r)
        (Aliases.get_canonical_element_exn binding_time_resolver
           binding_times_and_modes t element elt_name_mode min_name_mode
           min_binding_time).

(* Adding an alias. *)
Definition rename_add_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple),
    Aliases.add (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
      (rename_simple r canonical_element1)
      (rename_simple r canonical_element2)
    = match
        Aliases.add binding_time_resolver binding_times_and_modes t
          canonical_element1 canonical_element2
      with
      | None => None
      | Some Or_bottom.Bottom => Some Or_bottom.Bottom
      | Some (Or_bottom.Ok result) =>
        Some (Or_bottom.Ok (rename_add_result r result))
      end.

(* Respect: the renamed environment respects the renamed structure. *)
Definition rename_respected_statement : Prop :=
  ∀ (r : renaming) (t : Aliases.t) (vars : env),
    aliases_respected (rename_aliases r t) (rename_env r vars)
    <-> aliases_respected t vars.

(* The invariant: the renamed structure has it for the renamed typing
   environment. *)
Definition rename_well_formed_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    aliases_well_formed (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
    <-> aliases_well_formed binding_time_resolver binding_times_and_modes t.

(* What the compiler's two checks answer, which it makes with its
   invariant checks on: [Aliases.invariant] answers of the renamed
   structure what it answers of the structure, and
   [Aliases.invariant_add_result] of the renamed result of
   [Aliases.add] what it answers of the result. *)
Definition rename_invariant_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    Aliases.invariant (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes) (rename_aliases r t)
    = Aliases.invariant binding_time_resolver binding_times_and_modes t.

Definition rename_invariant_add_result_statement : Prop :=
  ∀ (A : Type) (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (result : Aliases.add_result),
    Aliases.invariant_add_result (rename_resolver r binding_time_resolver)
      (rename_binding_times r binding_times_and_modes)
      (rename_add_result r result)
    = Aliases.invariant_add_result binding_time_resolver
        binding_times_and_modes result.

(* ======================================================================== *)
(* The structure as a thing that contains names                             *)
(* ======================================================================== *)

(* [Contains_names.S] for the aliases, the names being the variables: a
   permutation of variables acts by [rename_aliases], the free names
   are the free variables, and equality is [=]. *)
Definition aliases_contains_names : Nominal.contains_names Aliases.t :=
  Nominal.Contains_names (fun p t => rename_aliases (Renaming p []) t)
    aliases_free_variables eq.

(* It has the laws of the interface. *)
Definition aliases_nominal_statement : Prop :=
  Nominal.nominal aliases_contains_names.
