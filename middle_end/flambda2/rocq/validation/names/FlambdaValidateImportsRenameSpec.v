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

(* The tests of the validation of a unit run from its imports, under
   renaming.

   This file is to be reviewed: the statements. It has no definition.

   FlambdaValidateSpec.validate_unit gives a verdict of two terms, and
   FlambdaValidateImportsSpec.v has the tests by which that verdict is
   one that holds of the unit run from a static environment
   ([holds_from], [lifted_holds_from], [verdict_holds_from], and
   [verdict_knowing_holds_from] for the second asking). That
   [validate_unit] gives the same verdict of a pair of terms and of
   the pair renamed is
   FlambdaValidatorEquivarianceSpec.validate_unit_equivariant_statement.
   This file states the same of the tests: each answers of the renamed
   terms what it answers of the terms. With the two together, whether
   a unit is validated with its imports does not depend on the names
   of its variables and of its continuations
   ([validate_unit_holds_from_equivariant_statement],
   [validate_unit_knowing_holds_from_equivariant_statement]).

   What a renaming is here: FlambdaNominalSpec.renaming, which acts on
   variables and on continuations. It moves no symbol and no code id,
   and it does not act on a static environment: the static
   environment, the names to remove and the value slots are the same
   on the two sides of every statement. The liftings and the
   continuations given for the loops hold continuations, and are
   renamed with the terms
   (FlambdaValidatorEquivarianceSpec.rename_lifting,
   [rename_loopified]).

   The tests are of two sorts.

   Those that read a term for the symbols and the code ids that it
   binds: FlambdaImportsSpec.static_names_free_in,
   FlambdaUnusedHeadStaticsSpec.head_free_in, and
   [head_removal_free_in], which asks the second where a removal is
   made. A renaming changes none of what they read, so they are
   stated by an equation with no premise, of any term.

   Those that compute again terms that the validation computes, and
   ask a test of the first sort of some of them: [known_free_in] and
   [normalise_lifted_free_in], each with its twin for the rounds that
   also call directly the closures that a [Let] makes
   ([known_calling_free_in], [normalise_lifted_calling_free_in]) and
   its twin for the rounds that also use what [my_closure] is known
   to be from its code ([known_own_free_in],
   [normalise_lifted_own_free_in]), [demote_and_tidy_known_free_in],
   [lifted_free_in], [statics_first_free_in], and the tests of a
   verdict. Some of them go again through rounds that make names
   (FlambdaRenameApartSpec.rename_apart_avoiding): [known_free_in],
   [normalise_lifted_free_in], their four twins, [lifted_free_in],
   [lifted_holds_from] and, with it, [verdict_holds_from] and
   [verdict_knowing_holds_from]. The terms that those compute of a
   renamed term are those that they compute of the term under another
   renaming
   (FlambdaValidatorEquivarianceSpec.equivariant_up_to_avoiding), which
   a test of the first sort cannot tell from the first. The rounds
   commute with renaming in that sense where the code of the term is
   closed (FlambdaBindersSpec.code_closed), and that is the premise of
   the statements of those tests and of no other statement here:
   [demote_and_tidy_known_free_in], [statics_first_free_in] and
   [holds_from] go through no round that makes a name, and their
   statements have no such premise. The rounds stop after the first
   that gives back its term, up to bound names, and the comparison
   that decides it answers the same of the renamed terms.

   Where liftings are given, they are renamed by one renaming and the
   term may be under another, as in
   FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement:
   after the first lifting the term is the result of rounds, under
   the renaming that they chose. What is asked of the second renaming
   is that it rename as the first does the continuations to avoid
   (FlambdaValidatorEquivarianceSpec.rename_conts_alike), among which
   are the two continuations of the unit of each lifting; a test of a
   verdict compares terms too, and there the second renaming is also
   to rename the free names of the term before as the first does
   ([renames_alike]). The statements with one renaming are the case
   in which the two are the same. The test of a verdict of the second
   asking is stated in both forms, as that of the first is.

   The tests of a verdict are stated of any verdict given, as the
   functions take one; the two statements named above are their case
   at the verdict that the validation gives.

   What is not here: nothing is said of the verdict itself, which
   FlambdaValidatorEquivarianceSpec.v and FlambdaValidateKnowingSpec.v
   state; and the tests on one lifting ([lifting_free_in],
   [made_free_in], [tables_free_in]) have their laws as lemmas of the
   proofs, being steps of the law of [lifted_free_in].

   The proofs are in FlambdaValidateImportsRenameProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaUnusedStaticsSpec
  FlambdaUnusedHeadStaticsSpec FlambdaLiftSpec FlambdaLoopifySpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidateKnowingSpec FlambdaImportsSpec FlambdaValidateImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The tests that read the static names of a term                           *)
(* ======================================================================== *)

(* Whether a static environment binds no symbol and no code id that a
   term binds: the same of the term renamed. *)
Definition static_names_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (e : expr),
    static_names_free_in statics (rename_expr r e)
    = static_names_free_in statics e.

(* The same of the names that the groups at the head of a term bind. *)
Definition head_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (e : expr),
    head_free_in statics (rename_expr r e) = head_free_in statics e.

(* And where the removal of static bindings from the head of a term is
   made: it is made of the renamed term where it is made of the
   term. *)
Definition head_removal_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (D : dropped) (e : expr),
    head_removal_free_in statics D (rename_expr r e)
    = head_removal_free_in statics D e.

(* ======================================================================== *)
(* The tests along the rounds                                               *)
(* ======================================================================== *)

(* The test of what ends the liftings made in turn
   (FlambdaValidateSpec.demote_and_tidy_known): its rounds make no
   name. *)
Definition demote_and_tidy_known_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (e : expr),
    demote_and_tidy_known_free_in statics rounds (rename_expr r e)
    = demote_and_tidy_known_free_in statics rounds e.

(* The test of the rounds that use what the unit binds its symbols to:
   with the continuations to avoid renamed with the term, where the
   code of the term is closed. *)
Definition known_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    known_free_in statics rounds (map (rename_continuation r) avoid) kept
      (rename_expr r e)
    = known_free_in statics rounds avoid kept e.

(* The same of the rounds of FlambdaValidateSpec.normalise_lifted. *)
Definition normalise_lifted_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    normalise_lifted_free_in statics rounds
      (map (rename_continuation r) avoid) kept (rename_expr r e)
    = normalise_lifted_free_in statics rounds avoid kept e.

(* The same of the tests of the rounds that also make direct, where a
   lifting asks for it, the calls of the closures that a [Let] makes
   (FlambdaValidateSpec.normalise_lifted_calling), whether or not it
   is asked. *)
Definition known_calling_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    known_calling_free_in statics calling rounds
      (map (rename_continuation r) avoid) kept (rename_expr r e)
    = known_calling_free_in statics calling rounds avoid kept e.

Definition normalise_lifted_calling_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    normalise_lifted_calling_free_in statics calling rounds
      (map (rename_continuation r) avoid) kept (rename_expr r e)
    = normalise_lifted_calling_free_in statics calling rounds avoid kept e.

(* The same of the tests of the rounds that also use, where a lifting
   asks for it, what [my_closure] is known to be from its code
   (FlambdaValidateSpec.normalise_lifted_own), whether or not that or
   the calls are asked. *)
Definition known_own_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    known_own_free_in statics calling own rounds
      (map (rename_continuation r) avoid) kept (rename_expr r e)
    = known_own_free_in statics calling own rounds avoid kept e.

Definition normalise_lifted_own_free_in_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (e : expr),
    code_closed e = true ->
    normalise_lifted_own_free_in statics calling own rounds
      (map (rename_continuation r) avoid) kept (rename_expr r e)
    = normalise_lifted_own_free_in statics calling own rounds avoid kept e.

(* The test of the liftings made in turn. The liftings and the
   continuations to avoid are renamed by [r], and the term by a
   renaming that renames those continuations as [r] does; the two
   continuations of the unit of each lifting are among them, as
   [validate_unit] gives them. *)
Definition lifted_free_in_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id)
    (e : expr),
    (∀ L : lifting,
      In L Ls ->
      among (unit_return (lifting_unit L)) avoid = true
      ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
    code_closed e = true ->
    rename_conts_alike r r1 avoid ->
    lifted_free_in statics rounds (map (rename_continuation r) avoid)
      (map (rename_lifting r) Ls) kept (rename_expr r1 e)
    = lifted_free_in statics rounds avoid Ls kept e.

(* The test of the static bindings taken to the head of a term, for the
   unit of the first lifting: the liftings renamed by [r], and the term
   by a renaming that renames the continuations of their units as [r]
   does. *)
Definition statics_first_free_in_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (Ls : list lifting) (e : expr),
    rename_conts_alike r r1 (unit_continuations Ls) ->
    statics_first_free_in statics (map (rename_lifting r) Ls)
      (rename_expr r1 e)
    = statics_first_free_in statics Ls e.

(* ======================================================================== *)
(* The tests of a verdict                                                   *)
(* ======================================================================== *)

(* Of a verdict that does not lift: the test reads the term after
   alone, normalised and tidied by rewrites that make no name. *)
Definition holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat)
    (kept : list code_id) (after : expr) (v : validation),
    holds_from statics rounds kept (rename_expr r after) v
    = holds_from statics rounds kept after v.

(* Of the verdict that lifts. The liftings, the continuations of the
   loops and the term after are renamed by [r]; the term before, whose
   code is closed, by a renaming that renames its free names, and the
   continuations of the units of the liftings and of the loops, as [r]
   does. *)
Definition lifted_holds_from_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    code_closed before = true ->
    renames_alike r r1 before ->
    rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
    lifted_holds_from statics rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r1 before)
      (rename_expr r after)
    = lifted_holds_from statics rounds unused slots G Ls kept before after.

(* Of any verdict given: the two above. *)
Definition verdict_holds_from_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr)
    (v : validation),
    code_closed before = true ->
    renames_alike r r1 before ->
    rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
    verdict_holds_from statics rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r1 before)
      (rename_expr r after) v
    = verdict_holds_from statics rounds unused slots G Ls kept before after v.

(* With one renaming. *)
Definition verdict_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) (v : validation),
    code_closed before = true ->
    verdict_holds_from statics rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after) v
    = verdict_holds_from statics rounds unused slots G Ls kept before after v.

(* The same of a verdict of the second asking: the test is of the two
   terms that FlambdaValidateKnowingSpec.validate_unit_knowing gives
   [validate_unit], and the rounds that make the first of them make
   names. *)
Definition verdict_knowing_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr) (v : validation),
    code_closed before = true ->
    verdict_knowing_holds_from statics rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r before) (rename_expr r after) v
    = verdict_knowing_holds_from statics rounds unused slots G Ls kept
        before after v.

(* The same where the liftings, the continuations of the loops and the
   term after are renamed by [r] and the term before, whose code is
   closed, by a renaming that renames its free names, and the
   continuations of the units of the liftings and of the loops, as [r]
   does: the form of [verdict_holds_from_renamed_alike_statement]. *)
Definition verdict_knowing_holds_from_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr)
    (v : validation),
    code_closed before = true ->
    renames_alike r r1 before ->
    rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
    verdict_knowing_holds_from statics rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r1 before) (rename_expr r after) v
    = verdict_knowing_holds_from statics rounds unused slots G Ls kept
        before after v.

(* ======================================================================== *)
(* The verdict that the validation gives                                    *)
(* ======================================================================== *)

(* Whether the verdict that [validate_unit] gives of two terms holds of
   the unit run from a static environment: the same of the two terms
   renamed, where the code of the term before is closed. *)
Definition validate_unit_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    verdict_holds_from statics rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after)
      (validate_unit rounds unused slots (rename_loopified r G)
         (map (rename_lifting r) Ls) kept (rename_expr r before)
         (rename_expr r after))
    = verdict_holds_from statics rounds unused slots G Ls kept before after
        (validate_unit rounds unused slots G Ls kept before after).

(* The same of the second asking. *)
Definition validate_unit_knowing_holds_from_equivariant_statement : Prop :=
  ∀ (r : renaming) (statics : static_env) (rounds : nat) (unused : dropped)
    (slots : list value_slot) (G : loopified) (Ls : list lifting)
    (kept : list code_id) (before after : expr),
    code_closed before = true ->
    verdict_knowing_holds_from statics rounds unused slots
      (rename_loopified r G) (map (rename_lifting r) Ls) kept
      (rename_expr r before) (rename_expr r after)
      (validate_unit_knowing rounds unused slots (rename_loopified r G)
         (map (rename_lifting r) Ls) kept (rename_expr r before)
         (rename_expr r after))
    = verdict_knowing_holds_from statics rounds unused slots G Ls kept
        before after
        (validate_unit_knowing rounds unused slots G Ls kept before after).
