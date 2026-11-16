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

(* The aliases of the typing environment, replayed: a list of calls of
   the functions of FlambdaEnvAliases.v, and the list of their answers.

   This file holds definitions only, and extracts. It is for the tests
   that compare the model with the compiler: the compiler's [Aliases.t]
   is abstract, so such a test cannot compare two structures; it makes
   the same calls on both sides, starting from [Aliases.empty], and
   compares the answers. [Aliases_replay.run] is the model's side: it
   goes through the calls in order, keeping the structure that the
   calls so far have made, and answers what each call answers. The
   resolver and the map of binding times and name modes are the same
   for every call.

   What is to be read here: that each call is the function it is named
   after, with its arguments in the order of the function; and what
   becomes of the structure where a call has no structure to give:
   - where the model's function answers [None], which is a fatal error
     of the compiler, the answer is [Fatal_error] and the structure is
     kept as it was (a test stops there on the compiler's side);
   - where [Aliases.add] answers bottom, the answer is [Added_bottom]
     and the structure is kept as it was.
   [Added] carries, beside the name demoted and the canonical element,
   what the compiler's check of the result answers
   ([Aliases.invariant_add_result]), which the compiler makes only with
   its invariant checks on: so one replay serves a test with the checks
   off, which ignores it, and one with them on, where [false] is a
   fatal error of the compiler.

   Which answers the two sides must agree on, and how (exactly, as
   sets, or up to a choice among equally early names), is not said
   here: FlambdaEnvAliasesSpec.v and FlambdaEnvAliasesSetSpec.v have
   the statements, and the status of the workstream the rule drawn
   from them. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaBindingTime
  FlambdaEnvAliases.
Import ListNotations.

Module Aliases_replay.

(* A call of one of the functions of the aliases. *)
Inductive call : Type :=
  | Add (canonical_element1 canonical_element2 : simple)
  | Add_alias_set (name : name) (aliases : Aliases.Alias_set.t)
  | Get_canonical_ignoring_name_mode (name : name)
  | Get_canonical_element_exn (element : simple)
      (elt_name_mode min_name_mode : Name_mode.t)
      (min_binding_time : Binding_time.t)
  | Get_aliases (element : simple)
  | Invariant.

(* What a call answers. [Added], [Added_bottom] and [Fatal_error] are
   the answers of [Add]; [Added_set] and [Fatal_error] those of
   [Add_alias_set]; [Canonical_element] that of the two queries for a
   canonical element, and [Not_found] that of the second where the
   compiler raises [Not_found]. *)
Inductive answer : Type :=
  | Added (demoted_name : name) (canonical_element : simple)
      (invariant_add_result : bool)
  | Added_bottom
  | Added_set
  | Fatal_error
  | Canonical_element (element : simple)
  | Not_found
  | Aliases_of (aliases : Aliases.Alias_set.t)
  | Invariant_holds (holds : bool).

Section Binding_times.

Context {A : Type}
  (binding_time_resolver : name -> Binding_time.With_name_mode.t)
  (binding_times_and_modes
    : Keyed.finite_map name (A * Binding_time.With_name_mode.t)).

(* One call: the structure afterwards, and the answer. *)
Definition one (t : Aliases.t) (c : call) : Aliases.t * answer :=
  match c with
  | Add canonical_element1 canonical_element2 =>
    match
      Aliases.add binding_time_resolver binding_times_and_modes t
        canonical_element1 canonical_element2
    with
    | Some (Or_bottom.Ok result) =>
      (Aliases.add_result_t result,
       Added (Aliases.demoted_name result) (Aliases.canonical_element result)
         (Aliases.invariant_add_result binding_time_resolver
            binding_times_and_modes result))
    | Some Or_bottom.Bottom => (t, Added_bottom)
    | None => (t, Fatal_error)
    end
  | Add_alias_set name aliases =>
    match
      Aliases.add_alias_set binding_time_resolver binding_times_and_modes t
        name aliases
    with
    | Some t' => (t', Added_set)
    | None => (t, Fatal_error)
    end
  | Get_canonical_ignoring_name_mode name =>
    (t, Canonical_element (Aliases.get_canonical_ignoring_name_mode t name))
  | Get_canonical_element_exn element elt_name_mode min_name_mode
      min_binding_time =>
    match
      Aliases.get_canonical_element_exn binding_time_resolver
        binding_times_and_modes t element elt_name_mode min_name_mode
        min_binding_time
    with
    | Some canonical_element => (t, Canonical_element canonical_element)
    | None => (t, Not_found)
    end
  | Get_aliases element => (t, Aliases_of (Aliases.get_aliases t element))
  | Invariant =>
    (t,
     Invariant_holds
       (Aliases.invariant binding_time_resolver binding_times_and_modes t))
  end.

(* The calls in order, from the structure [t]: the structure at the end
   and the answers. *)
Fixpoint run_from (t : Aliases.t) (calls : list call)
    : Aliases.t * list answer :=
  match calls with
  | [] => (t, [])
  | c :: calls =>
    let '(t, a) := one t c in
    let '(t, answers) := run_from t calls in
    (t, a :: answers)
  end.

(* The answers of the calls made in order from the structure that knows
   nothing. *)
Definition run (calls : list call) : list answer :=
  snd (run_from Aliases.empty calls).

End Binding_times.

End Aliases_replay.
