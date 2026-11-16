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

(* Things that contain names, and name abstractions over them: the
   interface of the compiler's nominal/ directory, with its laws.

   This file is to be reviewed, against
   middle_end/flambda2/nominal/contains_names.ml ([contains_names]),
   bindable.ml ([bindable]), permutation.ml (the permutations of
   FlambdaNominalSpec.v), name_abstraction.ml and .mli
   ([name_abstraction], [create], [pattern_match],
   [pattern_match_pair_statement]), and the statements.

   The compiler has one signature for a type whose values contain names,
   [Contains_names.S]: [apply_renaming] and [free_names]. A binder is a
   [Name_abstraction.t] of a [Bindable.t] and a term, and the only way to
   read one is [pattern_match], which gives the bound names and the term
   renamed so that the bound names are fresh. That is the theory of
   nominal sets: a permutation action with a finite support, and atom
   abstraction with concretion at a fresh atom.

   Here is the same interface with what the compiler leaves unsaid: the
   laws, and the equality. Rocq has no quotients, so a type is given with
   the relation that stands for equality of what its values denote
   ([equal]): for a type with no binder it is [=], and for expressions it
   is alpha-equivalence. Three relations are kept apart. Equality of
   values as they are written, [=]: the laws of the action (identity,
   composition) and the free names of a renamed value are stated with
   it. [equal]: the support law is stated with it, since a permutation
   that fixes the free names of a term with binders gives another tree.
   And what a client means by two values behaving alike, which is not
   part of the interface.

   The laws ([nominal]) are that [equal] is an equivalence, that the
   permutations act as a group, that the action preserves [equal] and
   commutes with [free_names], that [equal] values have the same free
   names, and that a permutation fixing the free names of a value gives
   an [equal] value. The last is what makes [free_names] a support.

   A name abstraction is a pair; its equality is that of nominal sets,
   stated with permutations: two abstractions are equal when each can be
   renamed, without touching its free names, so that the bound names are
   the same and the scopes equal ([equal_abstraction]). The statements:
   - abstractions over a bindable and a term are again things that
     contain names, with all the laws ([abstraction_nominal_statement]).
     Among them is the defining property: an abstraction renamed by a
     permutation that fixes its free names is equal to it;
   - pairs are too ([pair_nominal_statement]);
   - [pattern_match_pair_statement]: equal abstractions opened at the
     same bound names have equal scopes, whichever names those are. So
     comparing two abstractions by opening both at one fresh choice, as
     the compiler does, does not depend on the choice;
   - [invents_no_names_statement]: an equivariant function introduces no
     name;
   - the instances that need no term: a type with no names in it
     ([no_names]: run-time values are one), a name and a list of names
     as bindables ([single_name], [list_of_names]: the variable of a
     [Let], the parameters of a handler);
   - [pattern_match_well_defined_statement]: a function defined through
     [pattern_match] gives equal results on equal abstractions, whichever
     permutation opens each, if it is equivariant and no bound name is
     free in its result. This is what replaces "well defined on
     classes".

   Differences from the compiler.
   - One sort of name at a time. The compiler's [Renaming.t] is a
     permutation for each sort (variables, continuations) and its
     [Name_occurrences.t] a set for each; here an instance is for one
     sort, with a permutation ([FlambdaNominalSpec.permutation]) and a
     set of names. A type that holds names of two sorts is an instance
     twice, once for each, and a new sort of name, such as the locations
     of a heap, needs no change here.
   - [apply_renaming] takes the permutation first, as [rename_expr]
     takes the renaming first.
   - [pattern_match] takes the permutation that opens the abstraction.
     The compiler builds one, from the delayed renaming and fresh names
     ([Bindable.import_and_rename]); any permutation that fixes the free
     names of the abstraction serves ([opens]), and the last statement
     says the choice does not matter. [fresh_name] is the one function
     that looks at names other than to compare them for equality.
   - A bindable here may repeat a name; the compiler's may not
     ([Bound_continuations.create], and [Bound_parameters.create] under
     invariant checks). No law needs it.
   - There are no name modes and no counts of occurrences.
   - No renaming is delayed, and there is no import.

   The interface is in a module, [Nominal], so that its short names
   ([free_names], [equal], [create]) are had only by a file that imports
   the module. The permutations and [fresh_name] are outside it.

   Nothing here mentions the terms of Flambda 2. Nothing is trusted. The
   proofs are in FlambdaNameAbstractionProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* Permutations and sets of names                                           *)
(* ======================================================================== *)

(* [Permutation.compose ~second ~first], [Permutation.inverse] and
   [Permutation.empty], on the lists of swaps of FlambdaNominalSpec.v. *)
Definition compose_permutation (second first : permutation) : permutation :=
  second ++ first.

Definition inverse_permutation (p : permutation) : permutation := rev p.

Definition identity_permutation : permutation := [].

(* Two lists of swaps that are the same permutation. *)
Definition same_permutation (p p' : permutation) : Prop :=
  ∀ name : positive, permute p name = permute p' name.

(* The permutation moves none of the names. *)
Definition fixes (p : permutation) (names : list positive) : Prop :=
  ∀ name : positive, among name names = true -> permute p name = name.

(* A name that is not in a set: one greater than all of them. *)
Definition fresh_name (names : list positive) : positive :=
  Pos.succ (fold_right Pos.max xH names).

Definition fresh_name_fresh_statement : Prop :=
  ∀ names : list positive, among (fresh_name names) names = false.

(* ======================================================================== *)
(* Things that contain names                                                *)
(* ======================================================================== *)

Module Nominal.

(* [Contains_names.S] for one sort of name, and the relation that stands
   for equality. *)
Record contains_names (T : Type) : Type := Contains_names {
  apply_renaming : permutation -> T -> T;
  free_names : T -> list positive;
  equal : T -> T -> Prop
}.

Arguments Contains_names {T} apply_renaming free_names equal.
Arguments apply_renaming {T} _ _ _.
Arguments free_names {T} _ _.
Arguments equal {T} _ _ _.

(* The laws. *)
Record nominal {T : Type} (N : contains_names T) : Prop := Nominal {
  equal_refl : ∀ t : T, equal N t t;
  equal_sym : ∀ t t' : T, equal N t t' -> equal N t' t;
  equal_trans :
    ∀ t t' t'' : T, equal N t t' -> equal N t' t'' -> equal N t t'';
  apply_renaming_identity :
    ∀ t : T, apply_renaming N identity_permutation t = t;
  apply_renaming_compose :
    ∀ (second first : permutation) (t : T),
      apply_renaming N (compose_permutation second first) t
      = apply_renaming N second (apply_renaming N first t);
  apply_renaming_same :
    ∀ (p p' : permutation) (t : T),
      same_permutation p p' -> apply_renaming N p t = apply_renaming N p' t;
  equal_equivariant :
    ∀ (p : permutation) (t t' : T),
      equal N t t' -> equal N (apply_renaming N p t) (apply_renaming N p t');
  free_names_equivariant :
    ∀ (p : permutation) (t : T) (name : positive),
      among (permute p name) (free_names N (apply_renaming N p t))
      = among name (free_names N t);
  free_names_equal :
    ∀ t t' : T,
      equal N t t' -> same_names (free_names N t) (free_names N t');
  apply_renaming_fixes :
    ∀ (p : permutation) (t : T),
      fixes p (free_names N t) -> equal N (apply_renaming N p t) t
}.

(* [Bindable.S]: names in binding position. They are compared by
   equality, and a permutation is known on them by what it makes of
   them. *)
Record bindable {B : Type} (N : contains_names B) : Prop := Bindable {
  bindable_nominal : nominal N;
  bindable_equal : ∀ b b' : B, equal N b b' <-> b = b';
  bindable_faithful :
    ∀ (p p' : permutation) (b : B) (name : positive),
      apply_renaming N p b = apply_renaming N p' b ->
      among name (free_names N b) = true ->
      permute p name = permute p' name
}.

(* A function is equivariant, and gives equal results on equal
   arguments. *)
Definition equivariant {X Y : Type} (NX : contains_names X)
    (NY : contains_names Y) (f : X -> Y) : Prop :=
  ∀ (p : permutation) (x : X),
    equal NY (f (apply_renaming NX p x)) (apply_renaming NY p (f x)).

Definition respects_equal {X Y : Type} (NX : contains_names X)
    (NY : contains_names Y) (f : X -> Y) : Prop :=
  ∀ x x' : X, equal NX x x' -> equal NY (f x) (f x').

(* Pairs. *)
Definition pair_names {A B : Type} (NA : contains_names A)
    (NB : contains_names B) : contains_names (A * B) :=
  Contains_names
    (fun p ab => (apply_renaming NA p (fst ab), apply_renaming NB p (snd ab)))
    (fun ab => free_names NA (fst ab) ++ free_names NB (snd ab))
    (fun ab ab' => equal NA (fst ab) (fst ab') ∧ equal NB (snd ab) (snd ab')).

(* A type whose values hold no name of the sort: nothing is renamed. *)
Definition no_names (T : Type) : contains_names T :=
  Contains_names (fun _ t => t) (fun _ => []) eq.

(* A name, and a list of names. *)
Definition single_name : contains_names positive :=
  Contains_names permute (fun name => [name]) eq.

Definition list_of_names : contains_names (list positive) :=
  Contains_names (fun p names => map (permute p) names) (fun names => names)
    eq.

(* Name abstractions. *)

(* [Name_abstraction.t], with [create]: the bound names and their scope. *)
Record name_abstraction (B T : Type) : Type := create {
  bound : B;
  scope : T
}.

Arguments create {B T} bound scope.
Arguments bound {B T} _.
Arguments scope {B T} _.

Section Abstraction.

Context {B T : Type}.

Variable NB : contains_names B.

Variable NT : contains_names T.

(* [Name_abstraction.apply_renaming]: bound and free names alike. *)
Definition apply_renaming_abstraction (p : permutation)
    (a : name_abstraction B T) : name_abstraction B T :=
  create (apply_renaming NB p (bound a)) (apply_renaming NT p (scope a)).

(* [Name_abstraction.free_names]: those of the scope that are not
   bound. *)
Definition free_names_abstraction (a : name_abstraction B T)
    : list positive :=
  without (free_names NB (bound a)) (free_names NT (scope a)).

(* A permutation with which an abstraction may be opened: it moves none
   of its free names. The compiler's sends the bound names to fresh
   ones. *)
Definition opens (p : permutation) (a : name_abstraction B T) : Prop :=
  fixes p (free_names_abstraction a).

(* [Name_abstraction.pattern_match]: the bound names and the scope, both
   renamed. It is to be used with a permutation that [opens]. *)
Definition pattern_match (p : permutation) (a : name_abstraction B T)
    : B * T :=
  (apply_renaming NB p (bound a), apply_renaming NT p (scope a)).

(* Two abstractions are the same: opened suitably, they bind the same
   names and their scopes are equal. *)
Definition equal_abstraction (a a' : name_abstraction B T) : Prop :=
  ∃ p p' : permutation,
    opens p a
    ∧ opens p' a'
    ∧ fst (pattern_match p a) = fst (pattern_match p' a')
    ∧ equal NT (snd (pattern_match p a)) (snd (pattern_match p' a')).

Definition abstraction_names : contains_names (name_abstraction B T) :=
  Contains_names apply_renaming_abstraction free_names_abstraction
    equal_abstraction.

End Abstraction.

End Nominal.

Import Nominal.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition invents_no_names_statement : Prop :=
  ∀ (X Y : Type) (NX : contains_names X) (NY : contains_names Y)
    (f : X -> Y) (x : X) (name : positive),
    nominal NX ->
    nominal NY ->
    equivariant NX NY f ->
    respects_equal NX NY f ->
    among name (free_names NY (f x)) = true ->
    among name (free_names NX x) = true.

Definition no_names_nominal_statement : Prop :=
  ∀ T : Type, nominal (no_names T).

Definition single_name_bindable_statement : Prop := bindable single_name.

Definition list_of_names_bindable_statement : Prop := bindable list_of_names.

Definition pair_nominal_statement : Prop :=
  ∀ (A B : Type) (NA : contains_names A) (NB : contains_names B),
    nominal NA -> nominal NB -> nominal (pair_names NA NB).

Definition abstraction_nominal_statement : Prop :=
  ∀ (B T : Type) (NB : contains_names B) (NT : contains_names T),
    bindable NB -> nominal NT -> nominal (abstraction_names NB NT).

(* [Name_abstraction.pattern_match_pair]: two equal abstractions, opened
   so that they bind the same names, have equal scopes. *)
Definition pattern_match_pair_statement : Prop :=
  ∀ (B T : Type) (NB : contains_names B) (NT : contains_names T)
    (a a' : name_abstraction B T) (p p' : permutation),
    bindable NB ->
    nominal NT ->
    equal_abstraction NB NT a a' ->
    opens NB NT p a ->
    opens NB NT p' a' ->
    fst (pattern_match NB NT p a) = fst (pattern_match NB NT p' a') ->
    equal NT (snd (pattern_match NB NT p a))
      (snd (pattern_match NB NT p' a')).

(* A function of the bound names and the scope of an abstraction, applied
   to what [pattern_match] gives, does not depend on the permutation that
   opened the abstraction, nor on which of two equal abstractions it was,
   provided it is equivariant, gives equal results on equal arguments,
   and no bound name is free in its result. *)
Definition pattern_match_well_defined_statement : Prop :=
  ∀ (B T X : Type) (NB : contains_names B) (NT : contains_names T)
    (NX : contains_names X) (f : B * T -> X)
    (a a' : name_abstraction B T) (p p' : permutation),
    bindable NB ->
    nominal NT ->
    nominal NX ->
    equivariant (pair_names NB NT) NX f ->
    respects_equal (pair_names NB NT) NX f ->
    (∀ (ab : B * T) (name : positive),
      among name (free_names NB (fst ab)) = true ->
      among name (free_names NX (f ab)) = false) ->
    equal_abstraction NB NT a a' ->
    opens NB NT p a ->
    opens NB NT p' a' ->
    equal NX (f (pattern_match NB NT p a)) (f (pattern_match NB NT p' a')).
