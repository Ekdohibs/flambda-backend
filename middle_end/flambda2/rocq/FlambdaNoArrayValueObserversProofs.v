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

(* Proofs of the statements of FlambdaNoArrayValueObserversSpec.v.
   Nothing here needs review.

   The four of FlambdaObserversSpec.v are instances of its statements
   for a class, or with no observer in them, at the class of the
   observers that make no array: the way out is in it. The three of
   FlambdaObserversActionsSpec.v are instances of its statements for a
   class that a renaming keeps: a renaming keeps an observer
   (FlambdaObserversActionsProofs.observer_renamed) and keeps a
   continuation value free of arrays
   (FlambdaNoArrayValueProofs.rename_cont_free_of_arrays). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaEquivarianceSpec
  FlambdaBindersSpec
  FlambdaHeapSpec FlambdaExitWorldSpec FlambdaExitWorldProofs
  FlambdaRunUnderSpec FlambdaRunUnderProofs FlambdaObserversSpec
  FlambdaObserversProofs FlambdaExitWorldActionsSpec
  FlambdaExitWorldActionsProofs FlambdaObserversActionsSpec
  FlambdaObserversActionsProofs FlambdaNoArrayValueSpec
  FlambdaNoArrayValueProofs FlambdaNoArrayValueObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The observers that make no array                                         *)
(* ======================================================================== *)

Theorem observers_free_of_arrays : observers_free_of_arrays_statement.
Proof.
  intros slots conts. unfold observers, conts_free_of_arrays.
  rewrite !Forall_forall. split.
  - intros Hconts. split.
    + intros kc Hin. exact (proj1 (Hconts kc Hin)).
    + intros k c Hin. exact (proj2 (Hconts (k, c) Hin)).
  - intros [Hobservers Hfree] [k c] Hin. split.
    + exact (Hobservers (k, c) Hin).
    + exact (Hfree k c Hin).
Qed.

(* ======================================================================== *)
(* A statement that concludes one of the two gives what it gave             *)
(* ======================================================================== *)

Theorem worlds_observed_gives_free_of_arrays :
  worlds_observed_gives_free_of_arrays_statement.
Proof.
  intros slots w w' [Hk Hobserved]. split; [ exact Hk | ].
  intros c [Hc _]. exact (Hobserved c Hc).
Qed.

Theorem refines_under_observers_gives_free_of_arrays :
  refines_under_observers_gives_free_of_arrays_statement.
Proof.
  intros slots statics hp rd source target Hrefines conts Hconts.
  apply Hrefines.
  exact (proj1 (proj1 (observers_free_of_arrays slots conts) Hconts)).
Qed.

(* ======================================================================== *)
(* The statements of FlambdaObserversSpec.v                                 *)
(* ======================================================================== *)

Theorem worlds_observed_free_of_arrays_way_out :
  worlds_observed_free_of_arrays_way_out_statement.
Proof.
  intros slots w w' [Hk Hobserved]. split; [ exact Hk | ].
  exact
    (observed_refines_way_out (exit_cont w) _ _ _ _ _ _ rd_ rd_
       (Hobserved (Outer (exit_cont w))
          (conj I (Outer_free_of_arrays (exit_cont w))))).
Qed.

Theorem expr_refines_free_of_arrays_under_observers :
  expr_refines_free_of_arrays_under_observers_statement.
Proof.
  intros slots statics hp rd source target Hsource Hcode Hstatics Hheap
    Hrefines conts Hconts.
  apply refines_gives_observed_refines, Hrefines.
  destruct (proj1 (observers_free_of_arrays slots conts) Hconts)
    as [_ Hfree].
  split; [ | split; [ | split; [ | split; [ | split ] ] ] ]; simpl;
    try assumption.
  intros x v [].
Qed.

Theorem refines_under_observers_free_of_arrays_trans :
  refines_under_observers_free_of_arrays_trans_statement.
Proof.
  intros slots.
  exact (refines_observed_by_trans (observer_free_of_arrays slots)).
Qed.

Theorem under_observers_free_of_arrays_worlds :
  under_observers_free_of_arrays_worlds_statement.
Proof.
  intros slots statics hp rd source target.
  exact
    (refines_observed_by_worlds (observer_free_of_arrays slots) statics hp rd
       source target (fun k => conj I (Outer_free_of_arrays k))).
Qed.

(* ======================================================================== *)
(* The statements of FlambdaObserversActionsSpec.v                          *)
(* ======================================================================== *)

(* A renaming keeps the class. *)
Lemma observer_free_of_arrays_renamed (slots : list value_slot)
    (r : renaming) (c : cont_value) :
  observer_free_of_arrays slots c ->
  observer_free_of_arrays slots (rename_cont_value r c).
Proof.
  intros [Hobserver Hfree]. split.
  - exact (observer_renamed slots r c Hobserver).
  - exact (rename_cont_free_of_arrays r c Hfree).
Qed.

Theorem worlds_observed_free_of_arrays_equivariant :
  worlds_observed_free_of_arrays_equivariant_statement.
Proof.
  intros r slots w w'.
  exact
    (worlds_observed_by_equivariant (observer_free_of_arrays slots)
       (observer_free_of_arrays_renamed slots) r w w').
Qed.

Theorem refines_world_observed_free_of_arrays_equivariant :
  refines_world_observed_free_of_arrays_equivariant_statement.
Proof.
  intros r slots source target.
  exact
    (refines_world_observed_by_equivariant (observer_free_of_arrays slots)
       (observer_free_of_arrays_renamed slots) r source target).
Qed.

Theorem refines_observed_free_of_arrays_equivariant :
  refines_observed_free_of_arrays_equivariant_statement.
Proof.
  intros r slots statics hp rd source target.
  exact
    (refines_observed_by_equivariant (observer_free_of_arrays slots)
       (observer_free_of_arrays_renamed slots) r statics hp rd_ source target).
Qed.
