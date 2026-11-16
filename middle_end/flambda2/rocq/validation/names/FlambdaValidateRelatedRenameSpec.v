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

(* The second check of a unit, under renaming.

   This file is to be reviewed: the statements. It has no definition.

   FlambdaValidateRelatedSpec.validate_unit_related checks a unit
   against the unit it came from where a lifting changes its exits.
   That FlambdaValidateSpec.validate_unit answers the same of a pair
   of terms and of the pair renamed is
   FlambdaValidatorEquivarianceSpec.validate_unit_equivariant_statement;
   the check is another function, and this file states the same of
   it ([validate_unit_related_equivariant_statement]). The statements
   are about the function as it is defined, and not about what is
   proved of a pair that it accepts.

   What a renaming is here: FlambdaNominalSpec.renaming, which acts on
   variables and on continuations. It moves no symbol and no code id:
   the imports, the names to remove and the value slots are the same
   on the two sides. The liftings and the continuations given for the
   loops hold continuations, and are renamed with the terms
   (FlambdaValidatorEquivarianceSpec.rename_lifting,
   [rename_loopified]).

   The check follows the two comparisons of lifted terms of
   [validate_unit] with the liftings made without the test of the
   exits ([lifted_in_turn_observed]), and asks tests on the static
   names of the terms that it computes ([lifted_observed_free_in],
   and those of FlambdaValidateImportsSpec.v, whose laws of renaming
   are in FlambdaValidateImportsRenameSpec.v). So what is stated on
   the way is what is stated of [validate_unit] on the way: the
   liftings made in turn keep scope and commute with renaming up to
   the names that the rounds make, and their test answers the same of
   the term renamed.

   The check tests itself that the code of both units is closed
   (FlambdaBindersSpec.code_closed), and answers [Not_related] where
   it is not; a renaming does not change that test. So its statements
   have no premise that the code is closed, where
   [validate_unit_equivariant_statement] has one for the term before.
   The form with a second renaming of the term before asks of that
   renaming what
   FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement
   asks: that it rename the free names of the term before, and the
   continuations of the units of the liftings and of the loops, as
   the first does.

   What is not here: the test on one lifting
   ([lifting_observed_free_in]) has its law as a lemma of the proofs,
   being a step of the law of [lifted_observed_free_in].

   The proofs are in FlambdaValidateRelatedRenameProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaUnusedStaticsSpec FlambdaLiftSpec
  FlambdaLoopifySpec FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaImportsSpec FlambdaValidateImportsSpec FlambdaValidateRelatedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The liftings made in turn without the two tests of the unit              *)
(* ======================================================================== *)

(* They keep scope, as FlambdaValidateSpec.lifted_in_turn does. *)
Definition lifted_in_turn_observed_keeps_scope_statement : Prop :=
  ∀ (rounds : nat) (avoid : list continuation) (kept : list code_id)
    (Ls : list lifting),
    keeps_scope (lifted_in_turn_observed rounds avoid Ls kept).

(* They commute with renaming up to the names that the normalisations
   make, in the form of
   FlambdaValidatorEquivarianceSpec.lifted_in_turn_equivariant_up_to_statement:
   the liftings and the continuations to avoid are renamed by the
   first renaming, the term may be under another that does to its
   free names and to those continuations what the first does, and the
   two continuations of the unit of each lifting are among those to
   avoid. *)
Definition lifted_in_turn_observed_equivariant_up_to_statement : Prop :=
  ∀ (r : renaming) (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting),
    (∀ L : lifting,
      In L Ls ->
      among (unit_return (lifting_unit L)) avoid = true
      ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
    ∀ (r1 : renaming) (e : expr),
      code_closed e = true ->
      renames_alike r r1 e ->
      rename_conts_alike r r1 avoid ->
      ∃ r' : renaming,
        renames_alike r r' (lifted_in_turn_observed rounds avoid Ls kept e)
        ∧ rename_conts_alike r r' avoid
        ∧ lifted_in_turn_observed rounds (map (rename_continuation r) avoid)
            (map (rename_lifting r) Ls) kept (rename_expr r1 e)
          = rename_expr r' (lifted_in_turn_observed rounds avoid Ls kept e).

(* Their test on static names: the liftings and the continuations to
   avoid renamed by [r], and the term by a renaming that renames
   those continuations as [r] does. *)
Definition lifted_observed_free_in_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (statics : static_env) (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id)
    (e : expr),
    (∀ L : lifting,
      In L Ls ->
      among (unit_return (lifting_unit L)) avoid = true
      ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
    code_closed e = true ->
    rename_conts_alike r r1 avoid ->
    lifted_observed_free_in statics rounds
      (map (rename_continuation r) avoid) (map (rename_lifting r) Ls) kept
      (rename_expr r1 e)
    = lifted_observed_free_in statics rounds avoid Ls kept e.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The liftings, the continuations of the loops and the term after are
   renamed by [r]; the term before by a renaming that renames its free
   names, and the continuations of the units of the liftings and of
   the loops, as [r] does. *)
Definition validate_unit_related_renamed_alike_statement : Prop :=
  ∀ (r r1 : renaming) (imports : list (symbol * import_class))
    (rounds : nat) (unused : dropped) (slots : list value_slot)
    (G : loopified) (Ls : list lifting) (kept : list code_id)
    (before after : expr),
    renames_alike r r1 before ->
    rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
    validate_unit_related imports rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r1 before)
      (rename_expr r after)
    = validate_unit_related imports rounds unused slots G Ls kept before
        after.

(* With one renaming: the check answers of the two units renamed what
   it answers of the two units. *)
Definition validate_unit_related_equivariant_statement : Prop :=
  ∀ (r : renaming) (imports : list (symbol * import_class)) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr),
    validate_unit_related imports rounds unused slots (rename_loopified r G)
      (map (rename_lifting r) Ls) kept (rename_expr r before)
      (rename_expr r after)
    = validate_unit_related imports rounds unused slots G Ls kept before
        after.
