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

(* The invariant of the aliases as a test, and the checks that the
   compiler makes under a flag.

   This file is to be reviewed: the tests, the statements, and what is
   said of [Aliases.invariant] and [Aliases.invariant_add_result]
   against [invariant] and [invariant_add_result] of
   middle_end/flambda2/types/env/aliases.ml.

   The invariant that the statements about the aliases use is
   [aliases_well_formed] (FlambdaEnvAliasesSpec.v): the three maps
   agree ([aliases_consistent]), every canonical element is earlier
   than the names demoted to it ([canonical_elements_earliest]), and
   the name mode recorded for an alias is the one the typing
   environment gives it ([name_modes_recorded]). It is a proposition
   about every name. Here each part is a function that answers [true]
   or [false] by going through the three maps, and the statements say
   that the function answers [true] exactly when the part holds. The
   functions are what a test can run on a structure taken from the
   compiler.

   The compiler has two checks of its own, which it runs only with
   heavy invariant checks; [Aliases.invariant] and
   [Aliases.invariant_add_result] are the model's copies
   (FlambdaEnvAliases.v). They ask less than [aliases_well_formed], and
   the model is stricter than the compiler there (decision D32):
   - a structure that is consistent and whose canonical elements are
     earliest passes [Aliases.invariant]
     ([well_formed_invariant_statement]);
   - the converse fails, since [Aliases.invariant] never reads
     [canonical_elements] ([invariant_not_consistent_statement];
     discrepancy T3-3);
   - where [Aliases.add] keeps the invariant, its result passes
     [Aliases.invariant_add_result] ([add_invariant_statement]);
   - [Aliases.add] of two different symbols answers, and its result
     fails both checks ([add_two_symbols_invariant_statement];
     discrepancy T3-2, and the question at [add_earliest_statement] in
     FlambdaEnvAliasesSpec.v).

   The proofs are in FlambdaEnvAliasesInvariantProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The three maps agree                                                     *)
(* ======================================================================== *)

(* [demoted_to] and [is_canonical], as tests. *)
Definition demoted_to_b (t : Aliases.t) (n : name)
    (canonical_element : simple) : bool :=
  match Keyed.lookup name_eqb n (Aliases.canonical_elements t) with
  | Some canonical_element' => simple_equal canonical_element' canonical_element
  | None => false
  end.

Definition is_canonical_b (t : Aliases.t) (e : simple) : bool :=
  match Aliases.canonical t e with
  | Aliases.Is_canonical => true
  | Aliases.Alias_of_canonical _ => false
  end.

Definition demoted_to_b_iff_statement : Prop :=
  ∀ (t : Aliases.t) (n : name) (canonical_element : simple),
    demoted_to_b t n canonical_element = true
    <-> demoted_to t n canonical_element.

Definition is_canonical_b_iff_statement : Prop :=
  ∀ (t : Aliases.t) (e : simple),
    is_canonical_b t e = true <-> is_canonical t e.

(* [canonical_elements_canonical], as a test: it goes through the names
   that have been demoted. *)
Definition canonical_elements_canonical_b (t : Aliases.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun _ canonical_element => is_canonical_b t canonical_element)
    (Aliases.canonical_elements t).

(* The names listed among the aliases of [canonical_element], which are
   [aliases], have been demoted to it. *)
Definition listed_aliases_demoted_b (t : Aliases.t)
    (canonical_element : simple)
    (aliases : Aliases.Aliases_of_canonical_element.t) : bool :=
  forallb (fun n => demoted_to_b t n canonical_element)
    (Aliases.Aliases_of_canonical_element.all aliases).

(* [aliases_listed], as a test: every name that has been demoted is
   listed among the aliases of its canonical element, and every name
   listed among the aliases of a canonical element, a name or a
   constant, has been demoted to it. *)
Definition aliases_listed_b (t : Aliases.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n canonical_element =>
       Aliases.Aliases_of_canonical_element.mem
         (Aliases.get_aliases_of_canonical_element t canonical_element) n)
    (Aliases.canonical_elements t)
  && Keyed.for_all_bindings_b name_eqb
       (fun canonical_name aliases =>
          listed_aliases_demoted_b t (simple_of_name canonical_name) aliases)
       (Aliases.aliases_of_canonical_names t)
  && Keyed.for_all_bindings_b const_eqb
       (fun canonical_const aliases =>
          listed_aliases_demoted_b t (Const canonical_const) aliases)
       (Aliases.aliases_of_consts t).

Definition aliases_consistent_b (t : Aliases.t) : bool :=
  canonical_elements_canonical_b t && aliases_listed_b t.

Definition aliases_consistent_b_iff_statement : Prop :=
  ∀ t : Aliases.t, aliases_consistent_b t = true <-> aliases_consistent t.

(* ======================================================================== *)
(* The order and the name modes                                             *)
(* ======================================================================== *)

Definition canonical_elements_earliest_b {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n canonical_element =>
       Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         canonical_element (simple_of_name n))
    (Aliases.canonical_elements t).

Definition canonical_elements_earliest_b_iff_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    canonical_elements_earliest_b binding_time_resolver
      binding_times_and_modes t
    = true
    <-> canonical_elements_earliest binding_time_resolver
          binding_times_and_modes t.

(* The name modes recorded for the aliases of one canonical element are
   those of the typing environment. *)
Definition name_modes_recorded_of_b {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (aliases : Aliases.Aliases_of_canonical_element.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun n name_mode =>
       Name_mode.equal name_mode
         (Aliases.compute_name_mode_unscoped binding_times_and_modes
            (simple_of_name n)))
    aliases.

Definition name_modes_recorded_b {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : bool :=
  Keyed.for_all_bindings_b name_eqb
    (fun _ aliases => name_modes_recorded_of_b binding_times_and_modes aliases)
    (Aliases.aliases_of_canonical_names t)
  && Keyed.for_all_bindings_b const_eqb
       (fun _ aliases =>
          name_modes_recorded_of_b binding_times_and_modes aliases)
       (Aliases.aliases_of_consts t).

Definition name_modes_recorded_b_iff_statement : Prop :=
  ∀ (A : Type)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    name_modes_recorded_b binding_times_and_modes t = true
    <-> name_modes_recorded binding_times_and_modes t.

(* ======================================================================== *)
(* The invariant                                                            *)
(* ======================================================================== *)

Definition aliases_well_formed_b {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) : bool :=
  aliases_consistent_b t
  && canonical_elements_earliest_b binding_time_resolver
       binding_times_and_modes t
  && name_modes_recorded_b binding_times_and_modes t.

Definition aliases_well_formed_b_iff_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    aliases_well_formed_b binding_time_resolver binding_times_and_modes t
    = true
    <-> aliases_well_formed binding_time_resolver binding_times_and_modes t.

(* ======================================================================== *)
(* The compiler's checks                                                    *)
(* ======================================================================== *)

(* A structure whose three maps agree and whose canonical elements are
   earliest passes the compiler's check. The name modes recorded are
   not needed: the check does not read them. *)
Definition well_formed_invariant_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    aliases_consistent t ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    Aliases.invariant binding_time_resolver binding_times_and_modes t = true.

(* The compiler's check passes on structures whose maps do not agree:
   the model is stricter than the compiler (decision D32; discrepancy
   T3-3). The structure of the proof lists [y] among the aliases of [x]
   and has not demoted [y]. *)
Definition invariant_not_consistent_statement : Prop :=
  ∃ (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (unit * Binding_time.With_name_mode.t))
    (t : Aliases.t),
    Aliases.invariant binding_time_resolver binding_times_and_modes t = true
    ∧ ~ aliases_consistent t.

(* Where [Aliases.add] keeps the invariant (the conditions are those of
   [add_earliest_statement]), its result passes the check that the
   compiler makes of it. *)
Definition add_invariant_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result),
    aliases_consistent t ->
    canonical_elements_earliest binding_time_resolver
      binding_times_and_modes t ->
    is_canonical t canonical_element1 ->
    is_canonical t canonical_element2 ->
    comparable binding_time_resolver binding_times_and_modes
      canonical_element1 canonical_element2 ->
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result) ->
    Aliases.invariant_add_result binding_time_resolver
      binding_times_and_modes result
    = true.

(* [Aliases.add] of two different symbols to the structure that knows
   nothing answers, and its result fails the two checks of the
   compiler, whatever the typing environment (discrepancy T3-2). *)
Definition add_two_symbols_invariant_statement : Prop :=
  ∀ (A : Type) (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (s s' : symbol),
    s <> s' ->
    ∃ result : Aliases.add_result,
      Aliases.add binding_time_resolver binding_times_and_modes Aliases.empty
        (Symbol s) (Symbol s')
      = Some (Or_bottom.Ok result)
      ∧ Aliases.invariant binding_time_resolver binding_times_and_modes
          (Aliases.add_result_t result)
        = false
      ∧ Aliases.invariant_add_result binding_time_resolver
          binding_times_and_modes result
        = false.
