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

(* The freshness quantifier: "for all but finitely many names".

   This file is to be reviewed: [for_fresh] and its notation, [supports],
   [propositions], [names_from], [fresh_names], [fresh_names_in_turn],
   and the statements. The compiler has nothing to check it against: it
   makes a fresh name from a counter and states nothing of it. The
   definition and the laws are those of the literature on nominal sets,
   where the quantifier is written with a reversed N: Gabbay and Pitts,
   "A new approach to abstract syntax with variable binding" (2002), and
   Pitts, "Nominal Sets: Names and Symmetry in Computer Science" (2013),
   section 3.2.

   [И a, P a] says that [P a] holds of all but finitely many names [a].
   It is the way to say "for a fresh [a]" without saying what [a] must be
   fresh for. Three readings of "[P] holds of a fresh name" agree when
   [P] depends on finitely many names only, its support:
   - [P] holds of every name outside some finite set ([for_fresh]);
   - [P] holds of every name outside the support;
   - [P] holds of some name outside the support.
   That they agree is [some_any_statement]. One proves the third (choose
   one fresh name and show [P] of it) and uses the second (it then holds
   of whichever fresh name is at hand).

   The definition. [for_fresh P] is that there is a list of names off
   which [P] holds. The reasons for this form.
   - It names no support, which is the point of the quantifier. The
     second and third readings above each need one.
   - The list bounds the exceptions and need not be exactly the names at
     which [P] fails. So nothing asks that [P] be decidable, and no law
     below needs the excluded middle: no statement here depends on an
     axiom. "The set of names at which [P] fails is finite" would need
     both.
   - It is a proposition. A proof of it cannot be asked for a name: the
     functions that are extracted take the names to avoid and call
     [fresh_name] (FlambdaNameAbstractionSpec.v), or [fresh_names]
     below for several names. The quantifier is for stating what is
     true of their results.
   It is the cofinite quantification of locally nameless developments
   (Aydemir, Chargueraud, Pierce, Pollack and Weirich, "Engineering
   formal metatheory", 2008), with names in place of indices.

   The notation. The letter is the Cyrillic capital I, U+0418, as in the
   papers. For Rocq it is a letter and not a symbol, so the notation
   makes it a keyword, as Rocq's own Utf8_core does for the Greek letter
   of its notation for functions, and a blank must follow it: [И a, P a].
   It takes binders as [∀] does: [И a b, P] is [И a, И b, P]. A line is
   measured in characters, of which the letter is one.

   Which laws need a support. In the papers every predicate has a finite
   support, because the logic is that of nominal sets. In Rocq a
   predicate on names can be anything: it can look at the order of
   names, for one, and then [И a b, a < b] holds and [И b a, a < b] does
   not ([for_fresh_swap_needs_support_statement]). So the support is a
   premise of the laws that need it, said with [supports]: a permutation
   that fixes the names of the support keeps the predicate.
   - No support is needed for introduction and elimination, for
     monotonicity (modus ponens under the quantifier), for conjunction,
     for a proposition that does not mention the name, and for one
     direction each of disjunction, negation, [∀] and [∃].
   - A support is needed for the agreement of the three readings, for
     the other direction of disjunction and negation, for implication as
     an equivalence, and for exchanging two quantified names.
   Two names quantified one after the other are different names
   ([for_fresh_distinct_statement]).

   With the interface of FlambdaNameAbstractionSpec.v: the propositions
   are a type that contains no names, with [<->] as its equality
   ([propositions]); a predicate of a name and a value is then a function
   to which [equivariant] and [respects_equal] apply as they stand, and
   such a predicate, at a value, is supported by the free names of the
   value ([equivariant_supports_statement]). So for an equivariant
   predicate the three readings agree with the free names of its
   argument as the support ([some_any_nominal_statement], and
   [some_any_names_nominal_statement] for several names).

   Several names at once. A binder of Flambda 2 may bind a list of
   names (the parameters of a handler, the continuations of a recursive
   group), and what is fresh for it is a list of as many names, all
   different. [И[n] names, P names] says that [P] holds of every list of
   [n] different names outside some finite set ([for_fresh_names]). It
   gives the quantifier taken [n] times, one name after another
   ([for_fresh_names_next_statement]), and is the same thing for a
   predicate with a support ([for_fresh_names_iterated_statement]); the
   three readings agree for it too ([some_any_names_statement]).
   [fresh_names] makes such a list from the names to avoid
   ([fresh_names_fresh_statement]): the name that [fresh_name] makes for
   them and the names that follow it, which are the names that
   [fresh_name] makes one at a time, each avoiding those made before
   ([fresh_names_equal_in_turn_statement]).

   Nothing here mentions the terms of Flambda 2. Nothing is trusted. The
   proofs are in FlambdaFreshQuantifierProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaNominalSpec
  FlambdaNameAbstractionSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* The quantifier                                                           *)
(* ======================================================================== *)

(* [P] holds of every name outside a finite set. *)
Definition for_fresh (P : positive -> Prop) : Prop :=
  ∃ avoid : list positive,
    ∀ name : positive, among name avoid = false -> P name.

Notation "'И' x .. y , P" :=
  (for_fresh (fun x => .. (for_fresh (fun y => P)) ..))
  (at level 10, x binder, y binder, P at level 200,
   format "'[  ' '[  ' 'И'  x  ..  y ']' ,  '/' P ']'") : type_scope.

(* Introduction: name the finite set. *)
Definition for_fresh_intro_statement : Prop :=
  ∀ (P : positive -> Prop) (avoid : list positive),
    (∀ name : positive, among name avoid = false -> P name) ->
    И name, P name.

(* Elimination: there is a name of which [P] holds, and one outside any
   finite set, there being infinitely many names. *)
Definition for_fresh_elim_statement : Prop :=
  ∀ (P : positive -> Prop) (names : list positive),
    (И name, P name) ->
    ∃ name : positive, among name names = false ∧ P name.

(* Monotonicity, as modus ponens under the quantifier. *)
Definition for_fresh_mono_statement : Prop :=
  ∀ P Q : positive -> Prop,
    (И name, P name -> Q name) -> (И name, P name) -> И name, Q name.

(* ======================================================================== *)
(* The connectives, for any predicate                                       *)
(* ======================================================================== *)

Definition for_fresh_and_statement : Prop :=
  ∀ P Q : positive -> Prop,
    (И name, P name ∧ Q name) <-> (И name, P name) ∧ (И name, Q name).

(* A proposition that does not mention the name. So [И name, True] holds
   and [И name, False] does not. *)
Definition for_fresh_constant_statement : Prop :=
  ∀ Q : Prop, (И _, Q) <-> Q.

Definition for_fresh_or_of_either_statement : Prop :=
  ∀ P Q : positive -> Prop,
    (И name, P name) ∨ (И name, Q name) -> И name, P name ∨ Q name.

Definition for_fresh_not_gives_not_statement : Prop :=
  ∀ P : positive -> Prop, (И name, ~ P name) -> ~ (И name, P name).

Definition for_fresh_forall_statement : Prop :=
  ∀ (X : Type) (P : positive -> X -> Prop),
    (И name, ∀ x : X, P name x) -> ∀ x : X, И name, P name x.

Definition for_fresh_exists_statement : Prop :=
  ∀ (X : Type) (P : positive -> X -> Prop),
    (∃ x : X, И name, P name x) -> И name, ∃ x : X, P name x.

(* A fresh name is outside any finite set given beforehand, and two names
   quantified one after the other differ. *)
Definition for_fresh_avoids_statement : Prop :=
  ∀ names : list positive, И name, among name names = false.

Definition for_fresh_distinct_statement : Prop := И a b, a <> b.

(* ======================================================================== *)
(* Predicates with a support                                                *)
(* ======================================================================== *)

(* The names of [support] support the predicate [P], on a type whose
   values contain names: a permutation that moves none of them keeps
   [P]. *)
Definition supports {X : Type} (N : contains_names X)
    (support : list positive) (P : X -> Prop) : Prop :=
  ∀ (p : permutation) (x : X),
    fixes p support -> P x -> P (apply_renaming N p x).

(* The three readings agree: all but finitely many names; every name
   outside the support; some name outside the support. *)
Definition some_any_statement : Prop :=
  ∀ (support : list positive) (P : positive -> Prop),
    supports single_name support P ->
    ((И name, P name)
     <-> (∀ name : positive, among name support = false -> P name))
    ∧ ((И name, P name)
       <-> (∃ name : positive, among name support = false ∧ P name)).

(* The connectives that need it. *)
Definition for_fresh_or_statement : Prop :=
  ∀ (support support' : list positive) (P Q : positive -> Prop),
    supports single_name support P ->
    supports single_name support' Q ->
    ((И name, P name ∨ Q name) <-> (И name, P name) ∨ (И name, Q name)).

Definition for_fresh_not_statement : Prop :=
  ∀ (support : list positive) (P : positive -> Prop),
    supports single_name support P ->
    ((И name, ~ P name) <-> ~ (И name, P name)).

Definition for_fresh_impl_statement : Prop :=
  ∀ (support support' : list positive) (P Q : positive -> Prop),
    supports single_name support P ->
    supports single_name support' Q ->
    ((И name, P name -> Q name) <-> ((И name, P name) -> И name, Q name)).

(* Two quantified names: any two different names outside the support. So
   the order in which they are quantified does not matter. *)
Definition for_fresh_two_statement : Prop :=
  ∀ (support : list positive) (P : positive * positive -> Prop),
    supports (pair_names single_name single_name) support P ->
    ((И a b, P (a, b))
     <-> (∀ a b : positive,
           among a support = false -> among b support = false -> a <> b ->
           P (a, b))).

Definition for_fresh_swap_statement : Prop :=
  ∀ (support : list positive) (P : positive * positive -> Prop),
    supports (pair_names single_name single_name) support P ->
    ((И a b, P (a, b)) <-> (И b a, P (a, b))).

(* Without a support the order matters: a predicate that looks at the
   order of names. *)
Definition for_fresh_swap_needs_support_statement : Prop :=
  (И a b, (a < b)%positive) ∧ ~ (И b a, (a < b)%positive).

(* ======================================================================== *)
(* Equivariant predicates                                                   *)
(* ======================================================================== *)

(* The propositions contain no names, and are the same when
   equivalent. *)
Definition propositions : contains_names Prop :=
  Contains_names (fun _ P => P) (fun _ => []) iff.

Definition propositions_nominal_statement : Prop := nominal propositions.

(* An equivariant predicate of two things that contain names is
   supported, at a value of the second, by the free names of that value.
   The first may be a name or a list of names. *)
Definition equivariant_supports_statement : Prop :=
  ∀ (B X : Type) (NB : contains_names B) (N : contains_names X)
    (R : B * X -> Prop) (x : X),
    nominal NB ->
    nominal N ->
    equivariant (pair_names NB N) propositions R ->
    respects_equal (pair_names NB N) propositions R ->
    supports NB (free_names N x) (fun b => R (b, x)).

Definition some_any_nominal_statement : Prop :=
  ∀ (X : Type) (N : contains_names X) (R : positive * X -> Prop) (x : X),
    nominal N ->
    equivariant (pair_names single_name N) propositions R ->
    respects_equal (pair_names single_name N) propositions R ->
    ((И name, R (name, x))
     <-> (∀ name : positive,
           among name (free_names N x) = false -> R (name, x)))
    ∧ ((И name, R (name, x))
       <-> (∃ name : positive,
             among name (free_names N x) = false ∧ R (name, x))).

(* ======================================================================== *)
(* Several fresh names                                                      *)
(* ======================================================================== *)

(* The names of a list are different from one another and outside a
   set. *)
Definition distinct_outside (avoid names : list positive) : Prop :=
  distinct names = true
  ∧ ∀ name : positive, among name names = true -> among name avoid = false.

(* A name and those that follow it, as many as asked for. *)
Fixpoint names_from (count : nat) (first : positive) : list positive :=
  match count with
  | O => []
  | S count => first :: names_from count (Pos.succ first)
  end.

(* As many names as asked for, different from one another and outside a
   list: the name that [fresh_name] (FlambdaNameAbstractionSpec.v) makes
   for the list, and those that follow it. A function that makes several
   names takes the names to avoid and calls this. The greatest of the
   names to avoid is looked for once, whatever the number of names
   asked for. *)
Definition fresh_names (count : nat) (avoid : list positive)
    : list positive :=
  names_from count (fresh_name avoid).

Definition fresh_names_fresh_statement : Prop :=
  ∀ (count : nat) (avoid : list positive),
    length (fresh_names count avoid) = count
    ∧ distinct_outside avoid (fresh_names count avoid).

(* The same names made one at a time: [fresh_name] again and again, each
   name made being one that the next avoids. It goes through the whole
   list to avoid, which grows by each name made, once for each name
   that it makes; it is here to say what [fresh_names] computes, and no
   function calls it. *)
Fixpoint fresh_names_in_turn (count : nat) (avoid : list positive)
    : list positive :=
  match count with
  | O => []
  | S count =>
    let name := fresh_name avoid in
    name :: fresh_names_in_turn count (name :: avoid)
  end.

Definition fresh_names_equal_in_turn_statement : Prop :=
  ∀ (count : nat) (avoid : list positive),
    fresh_names count avoid = fresh_names_in_turn count avoid.

(* [P] holds of every list of [count] different names outside a finite
   set. *)
Definition for_fresh_names (count : nat) (P : list positive -> Prop)
    : Prop :=
  ∃ avoid : list positive,
    ∀ names : list positive,
      length names = count -> distinct_outside avoid names -> P names.

Notation "'И' [ n ] x , P" := (for_fresh_names n (fun x => P))
  (at level 10, x name, P at level 200,
   format "'[  ' '[  ' 'И' [ n ]  x ']' ,  '/' P ']'") : type_scope.

(* Elimination: such a list can be had outside any finite set. *)
Definition for_fresh_names_elim_statement : Prop :=
  ∀ (count : nat) (P : list positive -> Prop) (avoid : list positive),
    (И[count] names, P names) ->
    ∃ names : list positive,
      length names = count ∧ distinct_outside avoid names ∧ P names.

(* No name, and one name more. *)
Definition for_fresh_names_none_statement : Prop :=
  ∀ P : list positive -> Prop, (И[0] names, P names) <-> P [].

Definition for_fresh_names_next_statement : Prop :=
  ∀ (count : nat) (P : list positive -> Prop),
    (И[S count] names, P names) ->
    И name, И[count] names, P (name :: names).

(* For a predicate with a support the three readings agree, and the
   quantifier is the one for a single name, repeated. *)
Definition some_any_names_statement : Prop :=
  ∀ (count : nat) (support : list positive) (P : list positive -> Prop),
    supports list_of_names support P ->
    ((И[count] names, P names)
     <-> (∀ names : list positive,
           length names = count -> distinct_outside support names ->
           P names))
    ∧ ((И[count] names, P names)
       <-> (∃ names : list positive,
             length names = count
             ∧ distinct_outside support names
             ∧ P names)).

Definition for_fresh_names_iterated_statement : Prop :=
  ∀ (count : nat) (support : list positive) (P : list positive -> Prop),
    supports list_of_names support P ->
    ((И[S count] names, P names)
     <-> И name, И[count] names, P (name :: names)).

Definition some_any_names_nominal_statement : Prop :=
  ∀ (count : nat) (X : Type) (N : contains_names X)
    (R : list positive * X -> Prop) (x : X),
    nominal N ->
    equivariant (pair_names list_of_names N) propositions R ->
    respects_equal (pair_names list_of_names N) propositions R ->
    ((И[count] names, R (names, x))
     <-> (∀ names : list positive,
           length names = count ->
           distinct_outside (free_names N x) names ->
           R (names, x)))
    ∧ ((И[count] names, R (names, x))
       <-> (∃ names : list positive,
             length names = count
             ∧ distinct_outside (free_names N x) names
             ∧ R (names, x))).
