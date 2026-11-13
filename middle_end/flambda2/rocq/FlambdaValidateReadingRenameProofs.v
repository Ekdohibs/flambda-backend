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


(* Proofs of the statements of FlambdaValidateReadingRenameSpec.v.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaNominalSpec FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNormaliseSpec FlambdaRecHandlersSpec FlambdaUnusedStaticsSpec
  FlambdaLoopifySpec
  FlambdaStubCallsSpec FlambdaStubCallsProofs FlambdaNeverReturnsSpec
  FlambdaNeverReturnsProofs FlambdaNeverReturnsScopeSpec
  FlambdaNeverReturnsScopeProofs FlambdaLoopsScopeSpec
  FlambdaLoopsScopeProofs FlambdaStaticsScopeSpec FlambdaStaticsScopeProofs
  FlambdaLiftSpec FlambdaGuardScopeSpec FlambdaArrayLoadSpec
  FlambdaArrayLoadProofs FlambdaArrayLoadScopeSpec
  FlambdaArrayLoadScopeProofs FlambdaKeepsScopeSpec FlambdaKeepsScopeProofs
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceProofs FlambdaValidateKnowingSpec
  FlambdaValidateKnowingProofs FlambdaValidateReadingSpec
  FlambdaValidateReadingRenameSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The read                                                                 *)
(* ======================================================================== *)

Lemma read_arrays_in_code_equivariant :
  rewrite_equivariant read_arrays_in_code.
Proof.
  intros r e. unfold read_arrays_in_code.
  rewrite load_known_arrays_equivariant.
  exact
    (rewrite_rec_handlers_equivariant load_known_arrays
       load_known_arrays_equivariant r _).
Qed.

Theorem read_arrays_equivariant : read_arrays_equivariant_statement.
Proof.
  intros r e. unfold read_arrays.
  rewrite read_arrays_in_code_equivariant.
  exact
    (rewrite_code_equivariant read_arrays_in_code
       read_arrays_in_code_equivariant r _).
Qed.

Lemma read_arrays_in_code_scope : keeps_scope read_arrays_in_code.
Proof.
  exact
    (keeps_scope_compose load_known_arrays
       (rewrite_rec_handlers load_known_arrays)
       load_known_arrays_keeps_scope
       (rewrite_rec_handlers_keeps_scope load_known_arrays
          load_known_arrays_keeps_scope)).
Qed.

Theorem read_arrays_keeps_scope : read_arrays_keeps_scope_statement.
Proof.
  exact
    (keeps_scope_compose read_arrays_in_code
       (rewrite_code read_arrays_in_code) read_arrays_in_code_scope
       (rewrite_code_keeps_scope read_arrays_in_code
          read_arrays_in_code_scope)).
Qed.

(* ======================================================================== *)
(* The normalisation of a round                                             *)
(* ======================================================================== *)

(* The phases with the round that is given the continuations
   (FlambdaValidatorEquivarianceProofs.phases_with), then the demotion
   of the exception handlers with its rounds, which commutes with
   renaming. *)
Theorem normalise_before_avoiding_keeps_scope :
  normalise_before_avoiding_keeps_scope_statement.
Proof.
  intros avoid rounds kept.
  exact
    (keeps_scope_compose
       (phases_with rounds (normalise_before_round_avoiding avoid kept))
       (demote_and_tidy rounds)
       (phases_with_scope rounds (normalise_before_round_avoiding avoid kept)
          (normalise_before_round_avoiding_scope avoid kept))
       (demote_and_tidy_scope rounds)).
Qed.

Theorem normalise_before_avoiding_equivariant_up_to :
  normalise_before_avoiding_equivariant_up_to_statement.
Proof.
  intros rounds kept.
  exact
    (equivariant_up_to_avoiding_after
       (fun avoid =>
          phases_with rounds (normalise_before_round_avoiding avoid kept))
       (demote_and_tidy rounds)
       (fun avoid =>
          phases_with_scope rounds
            (normalise_before_round_avoiding avoid kept)
            (normalise_before_round_avoiding_scope avoid kept))
       (demote_and_tidy_scope rounds) (demote_and_tidy_equivariant rounds)
       (phases_with_up_to rounds
          (fun avoid => normalise_before_round_avoiding avoid kept)
          (fun avoid => normalise_before_round_avoiding_scope avoid kept)
          (normalise_before_round_avoiding_up_to kept))).
Qed.

(* ======================================================================== *)
(* A round                                                                  *)
(* ======================================================================== *)

(* The loops of the term made, as the validation makes them first. *)
Definition made_loops (G : loopified) (e : expr) : expr :=
  loopify_unit G (cut_returns [] (demote_stub_calls e)).

Lemma made_loops_scope (G : loopified) : keeps_scope (made_loops G).
Proof.
  exact
    (keeps_scope_compose (fun e => cut_returns [] (demote_stub_calls e))
       (loopify_unit G)
       (keeps_scope_compose demote_stub_calls (cut_returns [])
          FlambdaStaticsScopeProofs.demote_stub_calls_keeps_scope
          (FlambdaNeverReturnsScopeProofs.cut_returns_keeps_scope []))
       (FlambdaLoopsScopeProofs.loopify_unit_keeps_scope G)).
Qed.

Lemma made_loops_renamed (r : renaming) (G : loopified) (e : expr) :
  made_loops (rename_loopified r G) (rename_expr r e)
  = rename_expr r (made_loops G e).
Proof.
  unfold made_loops.
  rewrite FlambdaStubCallsProofs.demote_stub_calls_equivariant,
    FlambdaNeverReturnsProofs.cut_returns_unit_equivariant.
  apply loopify_unit_equivariant.
Qed.

(* The term normalised with its loops made, and what is read of it. *)
Definition normalised_for_read (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) (e : expr) : expr :=
  normalise_before_avoiding avoid rounds kept (made_loops G e).

Lemma normalised_for_read_scope (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) :
  keeps_scope (normalised_for_read avoid rounds G kept).
Proof.
  exact
    (keeps_scope_compose (made_loops G)
       (normalise_before_avoiding avoid rounds kept) (made_loops_scope G)
       (normalise_before_avoiding_keeps_scope avoid rounds kept)).
Qed.

Lemma reading_round_unfold (avoid : list continuation) (rounds : nat)
    (G : loopified) (kept : list code_id) (e : expr) :
  reading_round avoid rounds G kept e
  = if
      expr_eqb [] [] (normalised_for_read avoid rounds G kept e)
        (read_arrays (normalised_for_read avoid rounds G kept e))
    then e
    else read_arrays (normalised_for_read avoid rounds G kept e).
Proof. reflexivity. Qed.

Theorem reading_round_keeps_scope : reading_round_keeps_scope_statement.
Proof.
  intros avoid rounds G kept e Hclosed. rewrite reading_round_unfold.
  destruct (expr_eqb [] [] _ _).
  - split; [ exact Hclosed | ]. split; intros; assumption.
  - exact
      (keeps_scope_compose (normalised_for_read avoid rounds G kept)
         read_arrays (normalised_for_read_scope avoid rounds G kept)
         read_arrays_keeps_scope e Hclosed).
Qed.

(* The term normalised is, for the renamed term, under a second
   renaming; what is read of it is under the same one, so the
   comparison answers alike. Where nothing is read the round gives the
   term, renamed by the first. *)
Theorem reading_round_renamed : reading_round_renamed_statement.
Proof.
  intros r avoid rounds G kept e Hclosed. rewrite !reading_round_unfold.
  destruct (made_loops_scope G e Hclosed) as (Hclosed0 & _ & _).
  destruct
    (normalise_before_avoiding_equivariant_up_to rounds kept r avoid
       (made_loops G e) Hclosed0)
    as (r1 & Halike1 & Havoid1 & Heq1).
  cbv beta in Halike1, Heq1.
  assert
    (Heq :
       normalised_for_read (map (rename_continuation r) avoid) rounds
         (rename_loopified r G) kept (rename_expr r e)
       = rename_expr r1 (normalised_for_read avoid rounds G kept e)).
  { unfold normalised_for_read. rewrite made_loops_renamed. exact Heq1. }
  rewrite Heq, read_arrays_equivariant, expr_eqb_renamed.
  destruct
    (expr_eqb [] [] (normalised_for_read avoid rounds G kept e)
       (read_arrays (normalised_for_read avoid rounds G kept e))).
  - exists r. split; [ split; intros; reflexivity | ].
    split; [ intros k _; reflexivity | reflexivity ].
  - exists r1. split; [ | split; [ exact Havoid1 | reflexivity ] ].
    destruct (normalised_for_read_scope avoid rounds G kept e Hclosed)
      as (Hclosed1 & _ & _).
    exact
      (proj2
         (renames_alike_kept read_arrays r r1
            (normalised_for_read avoid rounds G kept e)
            read_arrays_keeps_scope Hclosed1 Halike1)).
Qed.

(* ======================================================================== *)
(* The rounds                                                               *)
(* ======================================================================== *)

Theorem normalise_reading_keeps_scope :
  normalise_reading_keeps_scope_statement.
Proof.
  intros rounds avoid G kept.
  exact
    (keeps_scope_iterate rounds (reading_round avoid rounds G kept)
       (reading_round_keeps_scope avoid rounds G kept)).
Qed.

(* Any number of rounds: after one, the term is under the second
   renaming, which does to the continuations to avoid, and so to those
   of the loops, what the first does; so the next round is given the
   same list and the same loops. *)
Lemma reading_rounds_renamed (n rounds : nat) (avoid : list continuation)
    (G : loopified) (kept : list code_id) :
  (∀ k : continuation,
     among k (map snd G) = true -> among k avoid = true) ->
  ∀ (r : renaming) (e : expr),
    code_closed e = true ->
    ∃ r' : renaming,
      renames_alike r r' (iterate n (reading_round avoid rounds G kept) e)
      ∧ rename_conts_alike r r' avoid
      ∧ iterate n
          (reading_round (map (rename_continuation r) avoid) rounds
             (rename_loopified r G) kept)
          (rename_expr r e)
        = rename_expr r' (iterate n (reading_round avoid rounds G kept) e).
Proof.
  intros Hloops.
  induction n as [ | n IH]; intros r e Hclosed.
  - exists r. split; [ split; intros; reflexivity | ].
    split; [ intros k _; reflexivity | reflexivity ].
  - cbn [iterate]. cbv zeta.
    destruct (reading_round_renamed r avoid rounds G kept e Hclosed)
      as (r1 & Halike1 & Havoid1 & Heq1).
    destruct (reading_round_keeps_scope avoid rounds G kept e Hclosed)
      as (Hclosed_f & _ & _).
    rewrite Heq1,
      (same_after_renamed r r1 e (reading_round avoid rounds G kept e)
         Hclosed_f Halike1).
    destruct (expr_eqb [] [] e (reading_round avoid rounds G kept e)).
    { exists r1. split; [ exact Halike1 | ].
      split; [ exact Havoid1 | reflexivity ]. }
    destruct (IH r1 (reading_round avoid rounds G kept e) Hclosed_f)
      as (r2 & Halike2 & Havoid2 & Heq2).
    rewrite (map_rename_conts_alike r r1 avoid Havoid1) in Heq2.
    assert (HG : rename_loopified r G = rename_loopified r1 G).
    { apply rename_loopified_alike. intros k Hk.
      apply Havoid1, Hloops, Hk. }
    rewrite <- HG in Heq2.
    exists r2. split; [ | split ].
    + exact
        (renames_alike_then (iterate n (reading_round avoid rounds G kept))
           r r1 r2 (reading_round avoid rounds G kept e)
           (keeps_scope_iterate n (reading_round avoid rounds G kept)
              (reading_round_keeps_scope avoid rounds G kept))
           Hclosed_f Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid1, Hk.
    + exact Heq2.
Qed.

Theorem normalise_reading_renamed : normalise_reading_renamed_statement.
Proof.
  intros r rounds avoid G kept e Hclosed Hloops.
  exact (reading_rounds_renamed rounds rounds avoid G kept Hloops r e Hclosed).
Qed.

(* ======================================================================== *)
(* The term that the checks give                                            *)
(* ======================================================================== *)

Theorem before_read_keeps_scope : before_read_keeps_scope_statement.
Proof.
  intros rounds G Ls kept.
  exact
    (normalise_reading_keeps_scope rounds
       (unit_continuations Ls ++ map snd G) G kept).
Qed.

Theorem before_read_renamed : before_read_renamed_statement.
Proof.
  intros r rounds G Ls kept before Hclosed. unfold before_read.
  rewrite unit_continuations_renamed, loop_continuations_renamed, <- map_app.
  apply normalise_reading_renamed; [ exact Hclosed | ].
  intros k Hk. rewrite among_app, Hk. apply orb_true_r.
Qed.

(* ======================================================================== *)
(* The checks                                                               *)
(* ======================================================================== *)

Theorem validate_unit_reading_equivariant :
  validate_unit_reading_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls kept before after Hclosed.
  unfold validate_unit_reading.
  destruct (before_read_renamed r rounds G Ls kept before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_renamed_alike.
  - exact (proj1 (before_read_keeps_scope rounds G Ls kept before Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.

(* The second asking where the term before is under a second renaming,
   which does to its free names, and to the continuations of the units
   of the liftings and of the loops, what the first does: the form of
   FlambdaValidatorEquivarianceSpec.validate_unit_renamed_alike_statement.
   The rounds of the term before are given those continuations renamed
   by the first renaming, which are those renamed by the second. *)
Lemma validate_unit_knowing_renamed_alike (r r1 : renaming) (rounds : nat)
    (unused : dropped) (slots : list value_slot) (G : loopified)
    (Ls : list lifting) (kept : list code_id) (before after : expr) :
  code_closed before = true ->
  renames_alike r r1 before ->
  rename_conts_alike r r1 (unit_continuations Ls ++ map snd G) ->
  validate_unit_knowing rounds unused slots (rename_loopified r G)
    (map (rename_lifting r) Ls) kept (rename_expr r1 before)
    (rename_expr r after)
  = validate_unit_knowing rounds unused slots G Ls kept before after.
Proof.
  intros Hclosed Halike Hheld. unfold validate_unit_knowing.
  rewrite unit_continuations_renamed, loop_continuations_renamed, <- map_app.
  set (avoid := unit_continuations Ls ++ map snd G) in *.
  rewrite <- (map_rename_conts_alike r r1 avoid Hheld).
  destruct
    (normalise_knowing_equivariant_up_to_avoiding
       (fun avoid => normalise_before_round_avoiding avoid kept) rounds
       (fun avoid => normalise_before_round_avoiding_scope avoid kept)
       (normalise_before_round_avoiding_up_to kept) r1 avoid before Hclosed)
    as (r2 & Halike2 & Havoid2 & Heq).
  cbv beta in Halike2, Heq. rewrite Heq.
  rewrite
    (normalise_knowing_equivariant (normalise_after_round kept) rounds
       (normalise_after_round_equivariant kept) r after).
  apply validate_unit_renamed_alike.
  - exact
      (proj1
         (normalise_knowing_keeps_scope
            (normalise_before_round_avoiding avoid kept) rounds
            (normalise_before_round_avoiding_scope avoid kept) before
            Hclosed)).
  - exact
      (renames_alike_then
         (normalise_knowing (normalise_before_round_avoiding avoid kept)
            rounds)
         r r1 r2 before
         (normalise_knowing_keeps_scope
            (normalise_before_round_avoiding avoid kept) rounds
            (normalise_before_round_avoiding_scope avoid kept))
         Hclosed Halike Halike2).
  - intros k Hk. rewrite (Havoid2 k Hk). exact (Hheld k Hk).
Qed.

Theorem validate_unit_knowing_reading_equivariant :
  validate_unit_knowing_reading_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls kept before after Hclosed.
  unfold validate_unit_knowing_reading.
  destruct (before_read_renamed r rounds G Ls kept before Hclosed)
    as (r1 & Halike & Havoid & Heq).
  rewrite Heq.
  apply validate_unit_knowing_renamed_alike.
  - exact (proj1 (before_read_keeps_scope rounds G Ls kept before Hclosed)).
  - exact Halike.
  - exact Havoid.
Qed.
