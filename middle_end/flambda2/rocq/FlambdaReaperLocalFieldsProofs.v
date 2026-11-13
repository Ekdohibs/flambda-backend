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

(* Proofs of the statements of FlambdaReaperLocalFieldsSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec.
Import ListNotations.

Theorem check_local_reads_used_sound :
  check_local_reads_used_sound_statement.
Proof.
  intros is_local F T Hcheck base relation to_ Hsource Hacc Hlocal.
  unfold check_local_reads_used in Hcheck.
  rewrite forallb_forall in Hcheck.
  specialize (Hcheck (to_, relation, base) Hacc). cbn in Hcheck, Hsource.
  rewrite Hsource, Hlocal in Hcheck. exact Hcheck.
Qed.

Theorem check_closed_local_sound : check_closed_local_sound_statement.
Proof.
  intros is_local F T Hcheck.
  unfold check_closed_local in Hcheck.
  apply andb_true_iff in Hcheck as [Hclosed Hlocal].
  split.
  - exact (check_closed_sound is_local F T Hclosed).
  - exact (check_local_reads_used_sound is_local F T Hlocal).
Qed.

Theorem local_reads_used_no_local : local_reads_used_no_local_statement.
Proof.
  intros is_local G S Hno base relation to_ _ _ Hlocal.
  rewrite Hno in Hlocal. discriminate Hlocal.
Qed.

Theorem example_local_reads_used : example_local_reads_used_statement.
Proof. vm_compute. reflexivity. Qed.

(* Every tables of a list fails the check of the fixed rules. *)
Lemma all_fail_local (is_local : field -> bool) (F : facts)
    (l : list tables) :
  forallb (fun T => negb (check_closed_local is_local F T)) l = true ->
  ∀ T : tables, T ∈ l -> check_closed_local is_local F T = false.
Proof.
  intros H T Hin. rewrite forallb_forall in H.
  apply negb_true_iff. exact (H T Hin).
Qed.

(* Every tables of a list fails the check of the rules before the
   fix. *)
Lemma all_fail (is_local : field -> bool) (F : facts) (l : list tables) :
  forallb (fun T => negb (check_closed is_local F T)) l = true ->
  ∀ T : tables, T ∈ l -> check_closed is_local F T = false.
Proof.
  intros H T Hin. rewrite forallb_forall in H.
  apply negb_true_iff. exact (H T Hin).
Qed.

Theorem local_slot_fixed : local_slot_fixed_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail_local; vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem nested_slot : nested_slot_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail; vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ apply all_fail_local; vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; vm_compute; reflexivity.
Qed.

Theorem local_read_alias : local_read_alias_statement.
Proof.
  intros is_local G S a base to_ from f [Hc Hfix] Hl Hcons Hacc
    (_ & Hsrc & _ & Hau & Hus).
  destruct (s_any_source S base) eqn:Eb.
  - pose proof (Hfix base f to_ Eb Hacc Hl) as Hhu.
    destruct (s_any_usage S a) eqn:Ea.
    + apply (alias_escaping_reading _ _ _ Hc f to_ from).
      * exact (escaping_field_rule _ _ _ Hc a f from Ea Hcons Hl).
      * exact (reading_field_rule _ _ _ Hc base f to_ Eb Hacc Hl).
    + assert (Eau : s_any_usage S base = false).
      { destruct (s_any_usage S base);
          [ discriminate (Hau eq_refl) | reflexivity ]. }
      pose proof (usages_accessor _ _ _ Hc to_ f base Hacc Eau) as Hbb.
      destruct (Hus Hhu base Hbb) as [Hab | Hab];
        [ | congruence ].
      exact (alias_local_constructor_usage _ _ _ Hc a base f to_ from Hl
               Hcons Hab Hacc).
  - destruct Hsrc as [Hsrc | Hsrc]; [ congruence | ].
    exact (alias_local_accessor_source _ _ _ Hc base a f to_ from Hl Hacc
             Hsrc Hcons).
Qed.

Theorem local_read_alias_needs_fix : local_read_alias_needs_fix_statement.
Proof.
  split.
  { apply check_closed_sound. vm_compute. reflexivity. }
  split.
  { unfold may_come_from. vm_compute.
    split; [ reflexivity | ].
    split; [ left; reflexivity | ].
    split; [ intros H; discriminate H | ].
    split; [ intros H; discriminate H | ].
    intros H; discriminate H. }
  split; [ vm_compute; tauto | ].
  split; [ vm_compute; tauto | ].
  vm_compute. reflexivity.
Qed.

Theorem local_read_field_used : local_read_field_used_statement.
Proof.
  intros is_local G S a base to_ from f Hc Hl Hcons Hacc
    (_ & Hsrc & _ & Hau & Hus) Hhu.
  pose proof (has_usage_accessor _ _ _ Hc to_ f base Hhu Hacc) as Hhub.
  destruct (s_any_usage S a) eqn:Ea.
  - destruct (s_any_source S base) eqn:Eb.
    + exact (field_used_reading _ _ _ Hc a f from to_ Hcons Ea
               (reading_field_rule _ _ _ Hc base f to_ Eb Hacc Hl) Hhu).
    + destruct Hsrc as [Hsrc | Hsrc];
        [ congruence | ].
      exact (field_used_local_source _ _ _ Hc base a f from to_ Hcons Hsrc
               Hl Ea Hacc Hhu).
  - assert (Eau : s_any_usage S base = false).
    { destruct (s_any_usage S base);
        [ discriminate (Hau eq_refl) | reflexivity ]. }
    pose proof (usages_accessor _ _ _ Hc to_ f base Hacc Eau) as Hbb.
    destruct (Hus Hhub base Hbb) as [Hab | Hab];
      [ | congruence ].
    exact (field_used_usage _ _ _ Hc a f from base to_ Hcons Hab Hacc Hhu).
Qed.
