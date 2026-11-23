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

(* The order that a code age relation stands for, and what the functions
   of the compiler answer in terms of it.

   This file is to be reviewed: [same_or_newer], the order;
   [well_formed], the condition that [Code_age_relation.add] does not
   check; [same_or_newer_b], which decides the order; [set_allows], what
   a set of code ids stands for; the three functions of the compiler for
   one compilation unit ([has_older_version_in_unit], [meet_set_in_unit],
   [meet_in_unit]); and the statements. The relation and its functions
   are FlambdaCodeAgeRelation.v, a transcription of
   types/env/code_age_relation.ml.

   The order. A relation maps a code id to the code id of which it is a
   newer version. [same_or_newer rel id id'] says that [id] is [id'] or a
   newer version of it: that [id'] is reached from [id] by following the
   relation, in no step or more. It is the order that the compiler means
   where a code id stands for "that code or a newer version of it": in a
   function type, in a direct call, and in the list of a call of known
   arity (terms/call_kind.mli: "or a simplified version thereof").
   - It is reflexive and transitive, whatever the relation.
   - The older versions of one code are a chain: of two codes that one
     code is a newer version of, one is a newer version of the other
     ([older_versions_chain_statement]). A relation gives a code id one
     older version, so the codes reached from one code id are reached one
     after the other. Two newer versions of one code need not be
     comparable: they are siblings.
   - It is antisymmetric where the relation is [well_formed]: where no
     code id is reached from its own older version, so that following
     the relation never comes back. The compiler says of its relation
     that it "forms a partial order", and that is so of a well-formed
     one. The compiler's [add] does not check it (its own comment asks
     for the check): adding to a well-formed relation keeps it
     well-formed exactly when the older code is not the newer one or a
     newer version of it ([add_well_formed_statement]). [add] gives the
     newer code id a new older version and so forgets the one it had:
     what is said here is of well-formedness, and not that the order of
     the relation before is kept.
     [add_unchecked_statement] has a relation that two calls of [add]
     make from the empty one and that is not well-formed, and a question
     that [has_older_version_in] does not answer on it in any number of
     steps: there the compiler's function does not return.

   Deciding the order. [same_or_newer_b] follows the relation from a
   code id, for one step more than the relation has pairs, and decides
   the order of every relation, one with a cycle too
   ([same_or_newer_b_decides_statement]): a code id that is reached is reached
   before a code id is met twice. It is the model's own function and not
   one of the compiler's.

   What the compiler's functions answer. The statements are of the three
   functions FOR ONE COMPILATION UNIT ([has_older_version_in_unit] and
   the two others): every code id is of the current unit, so the
   resolver is never asked, and the relation given is the only one that
   is followed. They are not statements of the functions with a resolver
   that answers: there the compiler leaves the relation that it was given
   and goes on in that of another unit, which [same_or_newer] of the
   first relation does not see, and it may answer [Unknown]. Of those
   only what the last four statements say is claimed here, each of one
   step or two of the function: a missing relation makes
   [has_older_version_in] and [meet] answer [Unknown] and makes [meet_set]
   keep both code ids, which is then more than the codes that both sets
   allow; and a relation that the resolver gives is followed.
   For one compilation unit, and with more steps than the relation has
   pairs:
   - [has_older_version_in_unit] answers [Known true] exactly where the
     code id is one of the candidates or a newer version of one
     ([set_allows]); on a well-formed relation it always answers, and
     never [Unknown];
   - [meet_in_unit], on a well-formed relation, answers the newer of two
     code ids of which one is the other or a newer version of it, and
     [Bottom] for two that are not so related;
   - [meet_set_in_unit] answers a set that allows exactly the code ids
     that both sets allow, and answers [Bottom] only where the two sets
     allow no code id in common. It does not answer [Bottom] wherever
     they allow none: two empty sets are equal, and the answer is then
     the empty set ([meet_set_in_unit_empty_statement]). On a well-formed
     relation it always answers.
   The number of steps is that of the model's bounded functions
   (FlambdaCodeAgeRelation.v): nothing here is a bound on the compiler's
   own recursion. Of a relation that is not well-formed, what is said
   is the first of the three and, of [meet_set], what an answer means
   where there is one.

   The proofs are in FlambdaCodeAgeRelationProofs.v, which needs no
   review. *)

From Stdlib Require Import PArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation.
Import ListNotations.

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

(* [id] is [id'] or a newer version of it. *)
Inductive same_or_newer (rel : Code_age_relation.t)
    : code_id -> code_id -> Prop :=
  | Same_version (id : code_id) : same_or_newer rel id id
  | Newer_version (id older id' : code_id) :
      Code_age_relation.get_older_version_of rel id = Some older ->
      same_or_newer rel older id' ->
      same_or_newer rel id id'.

(* No code id is reached from its own older version: following the
   relation from a code id, in one step or more, never comes back to
   it. *)
(* CR mshinwell: [Code_age_relation.add] checks nothing, and the comment
   before it in the compiler says that there should be a check of
   well-formedness there, "otherwise the functions below may not work
   correctly". [add_well_formed_statement] says what would keep a
   relation well-formed in this sense: that the older code is not the
   newer one or a newer version of it. Is that the check that is meant?
   And may the model take every relation that the compiler makes to be
   well-formed? Where one is not, [has_older_version_in] need not
   return ([add_unchecked_statement]), and the order is not a partial
   order. *)
Definition well_formed (rel : Code_age_relation.t) : Prop :=
  ∀ id older : code_id,
    Code_age_relation.get_older_version_of rel id = Some older ->
    ~ same_or_newer rel older id.

(* Whether [id'] is one of the first [steps] code ids met by following
   the relation from [id], which is the first of them. *)
Fixpoint same_or_newer_within (steps : nat) (rel : Code_age_relation.t)
    (id id' : code_id) : bool :=
  match steps with
  | O => false
  | S steps =>
    Pos.eqb id id'
    || match Code_age_relation.get_older_version_of rel id with
       | Some older => same_or_newer_within steps rel older id'
       | None => false
       end
  end.

(* Whether [id] is [id'] or a newer version of it. *)
Definition same_or_newer_b (rel : Code_age_relation.t) (id id' : code_id)
    : bool :=
  same_or_newer_within (S (length rel)) rel id id'.

(* What a set of code ids stands for: [id] is one of [ids] or a newer
   version of one. *)
Definition set_allows (rel : Code_age_relation.t) (ids : list code_id)
    (id : code_id) : Prop :=
  ∃ member : code_id, member ∈ ids ∧ same_or_newer rel id member.

Definition same_or_newer_trans_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (id1 id2 id3 : code_id),
    same_or_newer rel id1 id2 ->
    same_or_newer rel id2 id3 ->
    same_or_newer rel id1 id3.

(* The older versions of one code are a chain. *)
Definition older_versions_chain_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (id older older' : code_id),
    same_or_newer rel id older ->
    same_or_newer rel id older' ->
    same_or_newer rel older older' ∨ same_or_newer rel older' older.

(* On a well-formed relation the order is antisymmetric: a partial
   order. *)
Definition well_formed_antisymmetric_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (id id' : code_id),
    well_formed rel ->
    same_or_newer rel id id' ->
    same_or_newer rel id' id ->
    id = id'.

Definition empty_well_formed_statement : Prop :=
  well_formed Code_age_relation.empty.

(* What [add] would have to check: adding to a well-formed relation
   keeps it well-formed exactly when the older code is not the newer one
   or a newer version of it. *)
Definition add_well_formed_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (newer older : code_id),
    well_formed rel ->
    (well_formed (Code_age_relation.add rel newer older)
     <-> ~ same_or_newer rel older newer).

(* [same_or_newer_b] decides the order, of every relation. *)
Definition same_or_newer_b_decides_statement : Prop :=
  ∀ (rel : Code_age_relation.t) (id id' : code_id),
    same_or_newer_b rel id id' = true <-> same_or_newer rel id id'.

(* ======================================================================== *)
(* The functions of the compiler, for one compilation unit                  *)
(* ======================================================================== *)

(* The three functions where every code id is of the current unit: there
   is one compilation unit, and no relation of another unit. *)
Definition has_older_version_in_unit (steps : nat)
    (rel : Code_age_relation.t) (id : code_id) (candidates : list code_id)
    : option (Or_unknown.t bool) :=
  Code_age_relation.has_older_version_in unit (fun _ => tt) tt
    (fun _ _ => true) steps rel (fun _ => None) id candidates.

Definition meet_set_in_unit (steps : nat) (rel : Code_age_relation.t)
    (ids1 ids2 : list code_id) : option (Or_bottom.t (list code_id)) :=
  Code_age_relation.meet_set unit (fun _ => tt) tt (fun _ _ => true) steps
    rel (fun _ => None) ids1 ids2.

Definition meet_in_unit (steps : nat) (rel : Code_age_relation.t)
    (id1 id2 : code_id) : option (or_unknown_or_bottom code_id) :=
  Code_age_relation.meet unit (fun _ => tt) tt (fun _ _ => true) steps rel
    (fun _ => None) id1 id2.

(* [has_older_version_in] answers [Known true] exactly where the code id
   is one of the candidates or a newer version of one. *)
Definition has_older_version_in_unit_true_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (id : code_id)
    (candidates : list code_id),
    length rel < steps ->
    (has_older_version_in_unit steps rel id candidates
     = Some (Or_unknown.Known true)
     <-> set_allows rel candidates id).

(* On a well-formed relation it answers, and not [Unknown]. *)
Definition has_older_version_in_unit_answers_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (id : code_id)
    (candidates : list code_id),
    well_formed rel ->
    length rel < steps ->
    ∃ answer : bool,
      has_older_version_in_unit steps rel id candidates
      = Some (Or_unknown.Known answer).

(* [meet] answers the newer of two code ids of which one is the other or
   a newer version of it, [Bottom] for two that are not so related, and
   nothing else. *)
Definition meet_in_unit_ok_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (id1 id2 id : code_id),
    well_formed rel ->
    length rel < steps ->
    (meet_in_unit steps rel id1 id2 = Some (Ok id)
     <-> (same_or_newer rel id2 id1 ∧ id = id2)
         ∨ (same_or_newer rel id1 id2 ∧ id = id1)).

Definition meet_in_unit_bottom_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (id1 id2 : code_id),
    well_formed rel ->
    length rel < steps ->
    (meet_in_unit steps rel id1 id2 = Some Bottom
     <-> ~ same_or_newer rel id1 id2 ∧ ~ same_or_newer rel id2 id1).

Definition meet_in_unit_answers_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (id1 id2 : code_id),
    well_formed rel ->
    length rel < steps ->
    (∃ id : code_id, meet_in_unit steps rel id1 id2 = Some (Ok id))
    ∨ meet_in_unit steps rel id1 id2 = Some Bottom.

(* [meet_set] answers a set that allows exactly the code ids that both
   sets allow. *)
Definition meet_set_in_unit_ok_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (ids1 ids2 ids : list code_id),
    length rel < steps ->
    meet_set_in_unit steps rel ids1 ids2 = Some (Or_bottom.Ok ids) ->
    ∀ id : code_id,
      set_allows rel ids id
      <-> set_allows rel ids1 id ∧ set_allows rel ids2 id.

(* It answers [Bottom] only where the two sets allow no code id in
   common. *)
Definition meet_set_in_unit_bottom_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (ids1 ids2 : list code_id),
    length rel < steps ->
    meet_set_in_unit steps rel ids1 ids2 = Some Or_bottom.Bottom ->
    ∀ id : code_id, ~ (set_allows rel ids1 id ∧ set_allows rel ids2 id).

(* Two empty sets allow no code id, and the answer is not [Bottom]: the
   two sets are equal, and the answer is the first of them. *)
Definition meet_set_in_unit_empty_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t),
    meet_set_in_unit steps rel [] [] = Some (Or_bottom.Ok []).

(* On a well-formed relation it answers. *)
Definition meet_set_in_unit_answers_statement : Prop :=
  ∀ (steps : nat) (rel : Code_age_relation.t) (ids1 ids2 : list code_id),
    well_formed rel ->
    length rel < steps ->
    ∃ answer : Or_bottom.t (list code_id),
      meet_set_in_unit steps rel ids1 ids2 = Some answer.

(* ======================================================================== *)
(* What [add] does not check                                                *)
(* ======================================================================== *)

(* Two calls of [add] from the empty relation make one that is not
   well-formed: each of two code ids is the older version of the other.
   Whether the first is a newer version of a third code id is then not
   answered in any number of steps. *)
Definition add_unchecked_statement : Prop :=
  let rel :=
    Code_age_relation.add
      (Code_age_relation.add Code_age_relation.empty 1%positive 2%positive)
      2%positive 1%positive
  in
  ~ well_formed rel
  ∧ ∀ steps : nat,
      has_older_version_in_unit steps rel 1%positive [3%positive] = None.

(* ======================================================================== *)
(* Several compilation units                                                *)
(* ======================================================================== *)

(* A code id of another unit, whose relation the resolver has not got:
   it is not known whether it is a newer version of a candidate. *)
Definition has_older_version_in_unknown_statement : Prop :=
  ∀ (compilation_unit : Type)
    (get_compilation_unit : code_id -> compilation_unit)
    (current_unit : compilation_unit)
    (compilation_unit_equal : compilation_unit -> compilation_unit -> bool)
    (steps : nat) (rel : Code_age_relation.t)
    (resolver : compilation_unit -> option Code_age_relation.t)
    (id : code_id) (candidates : list code_id),
    among id candidates = false ->
    Code_age_relation.get_older_version_of rel id = None ->
    compilation_unit_equal (get_compilation_unit id) current_unit = false ->
    resolver (get_compilation_unit id) = None ->
    Code_age_relation.has_older_version_in compilation_unit
      get_compilation_unit current_unit compilation_unit_equal (S steps) rel
      resolver id candidates
    = Some Or_unknown.Unknown.

(* A code id of another unit whose relation the resolver gives: that
   relation is followed, and not the one given. *)
Definition has_older_version_in_resolved_statement : Prop :=
  ∀ (compilation_unit : Type)
    (get_compilation_unit : code_id -> compilation_unit)
    (current_unit : compilation_unit)
    (compilation_unit_equal : compilation_unit -> compilation_unit -> bool)
    (steps : nat) (rel rel' : Code_age_relation.t)
    (resolver : compilation_unit -> option Code_age_relation.t)
    (id older : code_id) (candidates : list code_id),
    among id candidates = false ->
    Code_age_relation.get_older_version_of rel id = None ->
    compilation_unit_equal (get_compilation_unit id) current_unit = false ->
    resolver (get_compilation_unit id) = Some rel' ->
    Code_age_relation.get_older_version_of rel' id = Some older ->
    among older candidates = true ->
    Code_age_relation.has_older_version_in compilation_unit
      get_compilation_unit current_unit compilation_unit_equal (S (S steps))
      rel resolver id candidates
    = Some (Or_unknown.Known true).

(* Two different code ids of other units, of which the relation given
   says nothing and whose relations the resolver has not got: [meet]
   answers [Unknown], and [meet_set] keeps both. *)
Definition meet_unknown_statement : Prop :=
  ∀ (compilation_unit : Type)
    (get_compilation_unit : code_id -> compilation_unit)
    (current_unit : compilation_unit)
    (compilation_unit_equal : compilation_unit -> compilation_unit -> bool)
    (steps : nat) (rel : Code_age_relation.t)
    (resolver : compilation_unit -> option Code_age_relation.t)
    (id1 id2 : code_id),
    id1 <> id2 ->
    (∀ id : code_id,
      id = id1 ∨ id = id2 ->
      Code_age_relation.get_older_version_of rel id = None
      ∧ compilation_unit_equal (get_compilation_unit id) current_unit = false
      ∧ resolver (get_compilation_unit id) = None) ->
    Code_age_relation.meet compilation_unit get_compilation_unit current_unit
      compilation_unit_equal (S steps) rel resolver id1 id2
    = Some Unknown.

Definition meet_set_unknown_statement : Prop :=
  ∀ (compilation_unit : Type)
    (get_compilation_unit : code_id -> compilation_unit)
    (current_unit : compilation_unit)
    (compilation_unit_equal : compilation_unit -> compilation_unit -> bool)
    (steps : nat) (rel : Code_age_relation.t)
    (resolver : compilation_unit -> option Code_age_relation.t)
    (id1 id2 : code_id),
    id1 <> id2 ->
    (∀ id : code_id,
      id = id1 ∨ id = id2 ->
      Code_age_relation.get_older_version_of rel id = None
      ∧ compilation_unit_equal (get_compilation_unit id) current_unit = false
      ∧ resolver (get_compilation_unit id) = None) ->
    Code_age_relation.meet_set compilation_unit get_compilation_unit
      current_unit compilation_unit_equal (S steps) rel resolver [id1] [id2]
    = Some (Or_bottom.Ok [id2; id1]).
