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

(* Name abstractions by the freshness quantifier.

   This file is to be reviewed: [concretion], [concretion_names] and the
   statements, against [equal_abstraction],
   [pattern_match_pair_statement] and
   [pattern_match_well_defined_statement] of
   FlambdaNameAbstractionSpec.v, which they restate, and against
   middle_end/flambda2/nominal/name_abstraction.ml ([pattern_match],
   [pattern_match_pair]), which opens an abstraction at a fresh name.

   FlambdaNameAbstractionSpec.v says when two name abstractions are
   equal with permutations: there are two, each moving no free name of
   its abstraction, that make the bound names the same and the scopes
   equal. The papers on nominal sets say it with the quantifier of
   FlambdaFreshQuantifierSpec.v: [a]x = [b]y when, for a fresh name c,
   x with a exchanged for c equals y with b exchanged for c. Here the
   two are shown to be one relation.

   One bound name. [concretion a fresh] is the scope of [a] with its
   bound name exchanged for [fresh]: it is what [pattern_match] gives
   when the permutation is that swap, and the swap opens the
   abstraction when [fresh] is not free in it
   ([concretion_pattern_match_statement]). Then:
   - two abstractions are equal exactly when their concretions at a
     fresh name are equal ([equal_abstraction_fresh_statement]);
   - "a fresh name" can be read as every name free in neither
     abstraction, or as some such name
     ([equal_abstraction_some_any_statement]). The first reading is
     [pattern_match_pair_statement] for swaps, and the second is
     [equal_abstraction] for swaps: for one bound name,
     [pattern_match_pair] is the agreement of the readings of the
     quantifier;
   - [pattern_match_pair_statement] holds as it stands with the
     quantifier in the place of [equal_abstraction]
     ([pattern_match_pair_fresh_statement]);
   - abstractions of one name, with the equality said with the
     quantifier, are things that contain names and obey every law
     ([fresh_abstraction_names], [fresh_abstraction_nominal_statement]).
     So a binder of one name (the variable of a [Let], the continuation
     of a [Let_cont]) may be given this equality where it is made an
     instance of the interface, and the equality with permutations is
     then a lemma about it.

   A list of bound names. One fresh name does not open an abstraction
   that binds a list (the parameters of a handler, the continuations of
   a recursive group). What does is a list of as many fresh names, all
   different, and the quantifier for that is [И[n] fresh, ...]. The
   permutation is the swaps of each bound name with the fresh name at
   its position ([concretion_names]). It sends the bound names to the
   fresh ones only if the bound names are different from one another
   and from the fresh ones, so the statement is for lists that repeat
   no name, which is what the compiler's bindables are, and the names
   to stay clear of include the bound names themselves
   ([concretion_names_pattern_match_statement]). With that, two
   abstractions of lists are equal exactly when the lists have one
   length and the concretions at fresh names are equal
   ([equal_abstraction_names_fresh_statement]).

   So the equality with permutations stays the definition, and the
   quantifier gives a second way to read it, for one name and for a
   list of different names. That the names of one binder are different
   is a condition of well-formedness
   (FlambdaBindersSpec.binders_distinct), so every well-formed term
   meets it. For a bindable that is not a name or a list of names there
   is no such statement: the interface gives no way to make, from a
   bindable and fresh names, the permutation that sends the one to the
   others (the compiler's [Bindable.S] has [renaming] for that). The
   work on general binders in the literature defines equality of
   abstractions with a permutation too (Urban and Kaliszyk, "General
   bindings and alpha-equivalence in Nominal Isabelle", 2012).

   Functions of an opened abstraction.
   [pattern_match_well_defined_statement] asks that no bound name be
   free in the result of the function. With the quantifier that is
   the freshness condition for binders of the papers: for a fresh
   name, the name is not free in the result whatever the scope. For an
   equivariant function the two say the same
   ([freshness_condition_statement]), so the statement holds with
   either ([pattern_match_well_defined_fresh_statement]). For one
   bound name its conclusion can be said with no permutation: the
   function applied at a fresh name to the concretion at that name
   gives equal results, whichever fresh names and whichever of two
   equal abstractions ([concretion_well_defined_statement]).

   Functions that make fresh names. A function of the model that makes
   names (a join that invents variables, inlining once it opens code at
   fresh names) is extracted and run, so it takes the names to avoid
   and makes its names with [fresh_name]: it computes one result, and a
   statement about it is about that result. The quantifier does not
   change that. What it can say is what the choice does not affect.
   Write the function as one that is given the names it is to use,
   [g (names, x)]. If [g] is equivariant, its result with those names
   bound is the same abstraction whichever fresh names it is given
   ([fresh_choice_statement]); and an equivariant property of the names
   and the argument holds for all fresh names if it holds for some, in
   particular for those that [fresh_name] gives
   (FlambdaFreshQuantifierSpec.some_any_names_nominal_statement). The
   names are fresh for the argument: the names to avoid that the caller
   gives must include its free names.

   Nothing here mentions the terms of Flambda 2. Nothing is trusted. The
   proofs are in FlambdaFreshAbstractionProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaNominalSpec
  FlambdaNameAbstractionSpec FlambdaFreshQuantifierSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* One bound name                                                           *)
(* ======================================================================== *)

(* The concretion of an abstraction of one name at a name: the scope,
   with the bound name exchanged for the name given. *)
Definition concretion {T : Type} (NT : contains_names T)
    (a : name_abstraction positive T) (fresh : positive) : T :=
  apply_renaming NT [(bound a, fresh)] (scope a).

(* It is [pattern_match] with the swap, which opens the abstraction at
   a name that is not free in it. *)
Definition concretion_pattern_match_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T) (a : name_abstraction positive T)
    (fresh : positive),
    among fresh (free_names_abstraction single_name NT a) = false ->
    opens single_name NT [(bound a, fresh)] a
    ∧ pattern_match single_name NT [(bound a, fresh)] a
      = (fresh, concretion NT a fresh).

(* Equality of abstractions, as in the papers. *)
Definition equal_abstraction_fresh_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T) (a a' : name_abstraction positive T),
    nominal NT ->
    (equal_abstraction single_name NT a a'
     <-> И fresh, equal NT (concretion NT a fresh) (concretion NT a' fresh)).

(* The same with the names said: every name free in neither, or some
   such name. *)
Definition equal_abstraction_some_any_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T) (a a' : name_abstraction positive T),
    nominal NT ->
    (equal_abstraction single_name NT a a'
     <-> (∀ fresh : positive,
           among fresh (free_names_abstraction single_name NT a) = false ->
           among fresh (free_names_abstraction single_name NT a') = false ->
           equal NT (concretion NT a fresh) (concretion NT a' fresh)))
    ∧ (equal_abstraction single_name NT a a'
       <-> (∃ fresh : positive,
             among fresh (free_names_abstraction single_name NT a) = false
             ∧ among fresh (free_names_abstraction single_name NT a')
               = false
             ∧ equal NT (concretion NT a fresh) (concretion NT a' fresh))).

(* [pattern_match_pair_statement] for one bound name, from the
   quantifier. *)
Definition pattern_match_pair_fresh_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T) (a a' : name_abstraction positive T)
    (p p' : permutation),
    nominal NT ->
    (И fresh, equal NT (concretion NT a fresh) (concretion NT a' fresh)) ->
    opens single_name NT p a ->
    opens single_name NT p' a' ->
    fst (pattern_match single_name NT p a)
    = fst (pattern_match single_name NT p' a') ->
    equal NT (snd (pattern_match single_name NT p a))
      (snd (pattern_match single_name NT p' a')).

(* Abstractions of one name with that equality: the action and the free
   names are those of [abstraction_names]. *)
Definition equal_at_fresh {T : Type} (NT : contains_names T)
    (a a' : name_abstraction positive T) : Prop :=
  И fresh, equal NT (concretion NT a fresh) (concretion NT a' fresh).

Definition fresh_abstraction_names {T : Type} (NT : contains_names T)
    : contains_names (name_abstraction positive T) :=
  Contains_names (apply_renaming_abstraction single_name NT)
    (free_names_abstraction single_name NT) (equal_at_fresh NT).

Definition fresh_abstraction_nominal_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T),
    nominal NT -> nominal (fresh_abstraction_names NT).

(* ======================================================================== *)
(* A list of bound names                                                    *)
(* ======================================================================== *)

(* The concretion at a list of names: each bound name is exchanged for
   the name at its position. *)
Definition concretion_names {T : Type} (NT : contains_names T)
    (a : name_abstraction (list positive) T) (fresh : list positive) : T :=
  apply_renaming NT (combine (bound a) fresh) (scope a).

(* It is [pattern_match] with those swaps, which open the abstraction
   when it binds different names and the names given are as many,
   different from one another, and neither bound by the abstraction nor
   free in it. *)
Definition concretion_names_pattern_match_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T)
    (a : name_abstraction (list positive) T) (fresh : list positive),
    distinct (bound a) = true ->
    length fresh = length (bound a) ->
    distinct_outside
      (bound a ++ free_names_abstraction list_of_names NT a) fresh ->
    opens list_of_names NT (combine (bound a) fresh) a
    ∧ pattern_match list_of_names NT (combine (bound a) fresh) a
      = (fresh, concretion_names NT a fresh).

Definition equal_abstraction_names_fresh_statement : Prop :=
  ∀ (T : Type) (NT : contains_names T)
    (a a' : name_abstraction (list positive) T),
    nominal NT ->
    distinct (bound a) = true ->
    distinct (bound a') = true ->
    (equal_abstraction list_of_names NT a a'
     <-> length (bound a) = length (bound a')
         ∧ И[length (bound a)] fresh,
             equal NT (concretion_names NT a fresh)
               (concretion_names NT a' fresh)).

(* ======================================================================== *)
(* Functions of an opened abstraction                                       *)
(* ======================================================================== *)

(* The freshness condition for binders, with the quantifier and
   without. *)
Definition freshness_condition_statement : Prop :=
  ∀ (B T X : Type) (NB : contains_names B) (NT : contains_names T)
    (NX : contains_names X) (f : B * T -> X),
    nominal NB ->
    nominal NX ->
    equivariant (pair_names NB NT) NX f ->
    ((И name,
       ∀ ab : B * T,
         among name (free_names NB (fst ab)) = true ->
         among name (free_names NX (f ab)) = false)
     <-> (∀ (ab : B * T) (name : positive),
           among name (free_names NB (fst ab)) = true ->
           among name (free_names NX (f ab)) = false)).

(* [pattern_match_well_defined_statement], with the freshness condition
   said with the quantifier. *)
Definition pattern_match_well_defined_fresh_statement : Prop :=
  ∀ (B T X : Type) (NB : contains_names B) (NT : contains_names T)
    (NX : contains_names X) (f : B * T -> X)
    (a a' : name_abstraction B T) (p p' : permutation),
    bindable NB ->
    nominal NT ->
    nominal NX ->
    equivariant (pair_names NB NT) NX f ->
    respects_equal (pair_names NB NT) NX f ->
    (И name,
      ∀ ab : B * T,
        among name (free_names NB (fst ab)) = true ->
        among name (free_names NX (f ab)) = false) ->
    equal_abstraction NB NT a a' ->
    opens NB NT p a ->
    opens NB NT p' a' ->
    equal NX (f (pattern_match NB NT p a)) (f (pattern_match NB NT p' a')).

(* For one bound name, with no permutation: the function applied at a
   fresh name does not depend on the name, nor on which of two equal
   abstractions. *)
Definition concretion_well_defined_statement : Prop :=
  ∀ (T X : Type) (NT : contains_names T) (NX : contains_names X)
    (f : positive * T -> X) (a a' : name_abstraction positive T),
    nominal NT ->
    nominal NX ->
    equivariant (pair_names single_name NT) NX f ->
    respects_equal (pair_names single_name NT) NX f ->
    (И name, ∀ t : T, among name (free_names NX (f (name, t))) = false) ->
    equal_abstraction single_name NT a a' ->
    И fresh fresh',
      equal NX (f (fresh, concretion NT a fresh))
        (f (fresh', concretion NT a' fresh')).

(* ======================================================================== *)
(* Functions that make fresh names                                          *)
(* ======================================================================== *)

(* The result of an equivariant function of the names it is to use and
   an argument, with those names bound, does not depend on the names,
   so long as they are fresh for the argument. *)
Definition fresh_choice_statement : Prop :=
  ∀ (X Y : Type) (NX : contains_names X) (NY : contains_names Y)
    (g : list positive * X -> Y) (x : X) (count : nat),
    nominal NX ->
    nominal NY ->
    equivariant (pair_names list_of_names NX) NY g ->
    respects_equal (pair_names list_of_names NX) NY g ->
    И[count] names,
      И[count] names',
        equal_abstraction list_of_names NY (create names (g (names, x)))
          (create names' (g (names', x))).
