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

(* Proofs of the statements of FlambdaCodeAgeRelationSpec.v. Nothing here
   needs review. *)

From Stdlib Require Import PArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation FlambdaCodeAgeRelationSpec.
Import ListNotations.

(* The function, where the theorem of FlambdaMapProofs.v has its name. *)
Local Abbreviation all_among_b := FlambdaMapSpec.all_among (only parsing).

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

Theorem same_or_newer_trans : same_or_newer_trans_statement.
Proof.
  intros rel id1 id2 id3 H12 H23.
  induction H12 as [id | id older id' Holder H IH]; [ exact H23 | ].
  exact (Newer_version rel id older id3 Holder (IH H23)).
Qed.

(* A code is a newer version of its older version. *)
Lemma same_or_newer_older (rel : Code_age_relation.t) (id older : code_id) :
  Code_age_relation.get_older_version_of rel id = Some older ->
  same_or_newer rel id older.
Proof.
  intros Holder.
  exact (Newer_version rel id older older Holder (Same_version rel older)).
Qed.

(* Two codes reached from one code: the relation gives a code one older
   version, so the one that is reached first is a newer version of the
   other. *)
Theorem older_versions_chain : older_versions_chain_statement.
Proof.
  intros rel id older older' H H'. revert H'.
  induction H as [id | id next id' Hnext H IH]; intros H'; [ left; exact H' | ].
  inversion H' as [ id0 | id0 next' id0' Hnext' Hrest ]; subst.
  - right. exact (Newer_version rel _ next id' Hnext H).
  - rewrite Hnext in Hnext'. injection Hnext' as <-. exact (IH Hrest).
Qed.

Theorem well_formed_antisymmetric : well_formed_antisymmetric_statement.
Proof.
  intros rel id id' Hwf H H'.
  destruct H as [id | id older id' Holder H]; [ reflexivity | ].
  exfalso. apply (Hwf id older Holder).
  exact (same_or_newer_trans rel older id' id H H').
Qed.

Theorem empty_well_formed : empty_well_formed_statement.
Proof. intros id older Holder. discriminate Holder. Qed.

(* The older version of a code id after [add]. *)
Lemma get_older_version_of_add (rel : Code_age_relation.t)
    (newer older id : code_id) :
  Code_age_relation.get_older_version_of
    (Code_age_relation.add rel newer older) id
  = if Pos.eqb id newer then Some older
    else Code_age_relation.get_older_version_of rel id.
Proof. apply lookup_cons. Qed.

(* What reaches [newer] before [add] reaches it after: the way to its
   first occurrence does not go by the older version of [newer]. *)
Lemma same_or_newer_add_to_newer (rel : Code_age_relation.t)
    (newer older id id' : code_id) :
  same_or_newer rel id id' ->
  id' = newer ->
  same_or_newer (Code_age_relation.add rel newer older) id id'.
Proof.
  intros H Hnewer.
  induction H as [id | id next id' Hnext H IH]; [ constructor | ].
  destruct (Pos.eqb_spec id newer) as [-> | Hne].
  - subst id'. constructor.
  - apply (Newer_version _ id next id'); [ | exact (IH Hnewer) ].
    rewrite get_older_version_of_add.
    destruct (Pos.eqb_spec id newer); [ contradiction | exact Hnext ].
Qed.

(* What is reached after [add] was reached before, or goes by [newer]
   and is reached from [older]. *)
Lemma same_or_newer_add_cases (rel : Code_age_relation.t)
    (newer older id id' : code_id) :
  same_or_newer (Code_age_relation.add rel newer older) id id' ->
  same_or_newer rel id id'
  ∨ (same_or_newer rel id newer
     ∧ same_or_newer (Code_age_relation.add rel newer older) older id').
Proof.
  intros H.
  induction H as [id | id next id' Hnext H IH]; [ left; constructor | ].
  rewrite get_older_version_of_add in Hnext.
  destruct (Pos.eqb_spec id newer) as [-> | Hne].
  - injection Hnext as <-. right. split; [ constructor | exact H ].
  - destruct IH as [IH | [IH IH']].
    + left. exact (Newer_version rel id next id' Hnext IH).
    + right. split; [ exact (Newer_version rel id next newer Hnext IH) | ].
      exact IH'.
Qed.

Theorem add_well_formed : add_well_formed_statement.
Proof.
  intros rel newer older Hwf. split.
  - intros Hadd Hreached. apply (Hadd newer older).
    + rewrite get_older_version_of_add, Pos.eqb_refl. reflexivity.
    + exact (same_or_newer_add_to_newer rel newer older older newer Hreached
               eq_refl).
  - intros Hnot id next Hnext Hback.
    rewrite get_older_version_of_add in Hnext.
    assert (Hno : ∀ id' : code_id,
              same_or_newer (Code_age_relation.add rel newer older) older id' ->
              same_or_newer rel id' newer -> False).
    { intros id' H H'.
      destruct (same_or_newer_add_cases rel newer older older id' H)
        as [Hold | [Hold _]]; [ | exact (Hnot Hold) ].
      exact (Hnot (same_or_newer_trans rel older id' newer Hold H')). }
    destruct (Pos.eqb_spec id newer) as [-> | Hne].
    + injection Hnext as <-. apply (Hno newer Hback). constructor.
    + destruct (same_or_newer_add_cases rel newer older next id Hback)
        as [Hold | [Hold Hrest]]; [ exact (Hwf id next Hnext Hold) | ].
      apply (Hno id Hrest).
      exact (Newer_version rel id next newer Hnext Hold).
Qed.

(* ======================================================================== *)
(* Deciding the order                                                       *)
(* ======================================================================== *)

Lemma same_or_newer_within_sound (steps : nat) (rel : Code_age_relation.t)
    (id id' : code_id) :
  same_or_newer_within steps rel id id' = true -> same_or_newer rel id id'.
Proof.
  revert id. induction steps as [ | steps IH]; intros id H; simpl in H;
    [ discriminate | ].
  apply orb_true_iff in H. destruct H as [H | H].
  - apply Pos.eqb_eq in H. subst id'. constructor.
  - destruct (Code_age_relation.get_older_version_of rel id) as [older | ]
      eqn:Holder; [ | discriminate ].
    exact (Newer_version rel id older id' Holder (IH older H)).
Qed.

(* The relation without the older version of one code id. *)
Lemma get_older_version_of_unbind (rel : Code_age_relation.t)
    (removed id : code_id) :
  Code_age_relation.get_older_version_of (unbind removed rel) id
  = if Pos.eqb id removed then None
    else Code_age_relation.get_older_version_of rel id.
Proof. apply lookup_unbind. Qed.

Lemma unbind_length_le (rel : Code_age_relation.t) (removed : code_id) :
  length (unbind removed rel : Code_age_relation.t) <= length rel.
Proof.
  unfold unbind. induction rel as [ | binding rel IH]; simpl; [ lia | ].
  destruct (negb _); simpl; lia.
Qed.

Lemma unbind_length_lt (rel : Code_age_relation.t) (removed older : code_id) :
  Code_age_relation.get_older_version_of rel removed = Some older ->
  length (unbind removed rel : Code_age_relation.t) < length rel.
Proof.
  unfold Code_age_relation.get_older_version_of.
  induction rel as [ | [bound older'] rel IH]; simpl; [ discriminate | ].
  intros Hfound. unfold unbind. simpl.
  destruct (Pos.eqb removed bound); simpl.
  - pose proof (unbind_length_le rel removed) as Hle. unfold unbind in Hle.
    lia.
  - specialize (IH Hfound). unfold unbind in IH. lia.
Qed.

Lemma same_or_newer_unbind (rel : Code_age_relation.t)
    (removed id id' : code_id) :
  same_or_newer (unbind removed rel) id id' -> same_or_newer rel id id'.
Proof.
  intros H. induction H as [id | id next id' Hnext H IH]; [ constructor | ].
  rewrite get_older_version_of_unbind in Hnext.
  destruct (Pos.eqb id removed); [ discriminate | ].
  exact (Newer_version rel id next id' Hnext IH).
Qed.

(* A code that is reached, and is not [removed], is reached without the
   older version of [removed], from where it was or from that older
   version: the way from the last occurrence of [removed] on. *)
Lemma same_or_newer_avoiding (rel : Code_age_relation.t)
    (removed older id id' : code_id) :
  removed <> id' ->
  Code_age_relation.get_older_version_of rel removed = Some older ->
  same_or_newer rel id id' ->
  same_or_newer (unbind removed rel) id id'
  ∨ same_or_newer (unbind removed rel) older id'.
Proof.
  intros Hne Holder H. revert Hne.
  induction H as [id | id next id' Hnext H IH]; intros Hne;
    [ left; constructor | ].
  specialize (IH Hne).
  destruct (Pos.eqb_spec id removed) as [-> | Hid].
  - rewrite Holder in Hnext. injection Hnext as <-.
    right. destruct IH; assumption.
  - destruct IH as [IH | IH]; [ left | right; exact IH ].
    apply (Newer_version _ id next id'); [ | exact IH ].
    rewrite get_older_version_of_unbind.
    destruct (Pos.eqb_spec id removed); [ contradiction | exact Hnext ].
Qed.

Lemma same_or_newer_within_unbind (steps : nat) (rel : Code_age_relation.t)
    (removed id id' : code_id) :
  same_or_newer_within steps (unbind removed rel) id id' = true ->
  same_or_newer_within steps rel id id' = true.
Proof.
  revert id. induction steps as [ | steps IH]; intros id H; simpl in *;
    [ discriminate | ].
  apply orb_true_iff in H. apply orb_true_iff.
  destruct H as [H | H]; [ left; exact H | right ].
  rewrite get_older_version_of_unbind in H.
  destruct (Pos.eqb id removed); [ discriminate | ].
  destruct (Code_age_relation.get_older_version_of rel id) as [older | ];
    [ exact (IH older H) | discriminate ].
Qed.

(* One step more than the relation has pairs is enough: by induction on
   the number of pairs, a code that is reached, and is not the first, is
   reached from the older version of the first without going by the
   first again. *)
Lemma same_or_newer_within_complete (n : nat) :
  ∀ (rel : Code_age_relation.t) (id id' : code_id),
    length rel <= n ->
    same_or_newer rel id id' ->
    same_or_newer_within (S n) rel id id' = true.
Proof.
  induction n as [ | n IH]; intros rel id id' Hlen H.
  - destruct rel as [ | binding rel]; [ | simpl in Hlen; lia ].
    destruct H as [id | id older id' Holder _];
      [ simpl; rewrite Pos.eqb_refl; reflexivity | discriminate Holder ].
  - destruct (Pos.eqb_spec id id') as [-> | Hne];
      [ simpl; rewrite Pos.eqb_refl; reflexivity | ].
    destruct H as [id | id older id' Holder H]; [ contradiction | ].
    change (same_or_newer_within (S (S n)) rel id id')
      with (Pos.eqb id id'
            || match Code_age_relation.get_older_version_of rel id with
               | Some older => same_or_newer_within (S n) rel older id'
               | None => false
               end).
    rewrite Holder. apply orb_true_iff. right.
    apply (same_or_newer_within_unbind (S n) rel id older id'). apply IH.
    + pose proof (unbind_length_lt rel id older Holder). lia.
    + destruct (same_or_newer_avoiding rel id older older id' Hne Holder H);
        assumption.
Qed.

Theorem same_or_newer_b_decides : same_or_newer_b_decides_statement.
Proof.
  intros rel id id'. split.
  - apply same_or_newer_within_sound.
  - apply same_or_newer_within_complete. lia.
Qed.

Lemma same_or_newer_b_refl (rel : Code_age_relation.t) (id : code_id) :
  same_or_newer_b rel id id = true.
Proof. apply same_or_newer_b_decides. constructor. Qed.

(* With the empty relation the order is equality. *)
Lemma same_or_newer_b_empty (id id' : code_id) :
  same_or_newer_b Code_age_relation.empty id id' = Pos.eqb id id'.
Proof. unfold same_or_newer_b. simpl. apply orb_false_r. Qed.

(* ======================================================================== *)
(* What a set of code ids stands for                                        *)
(* ======================================================================== *)

Lemma set_allows_singleton (rel : Code_age_relation.t) (id' id : code_id) :
  set_allows rel [id'] id <-> same_or_newer rel id id'.
Proof.
  split.
  - intros [member [[<- | []] H]]. exact H.
  - intros H. exists id'. split; [ left; reflexivity | exact H ].
Qed.

Lemma set_allows_newer (rel : Code_age_relation.t) (ids : list code_id)
    (id id' : code_id) :
  same_or_newer rel id id' -> set_allows rel ids id' -> set_allows rel ids id.
Proof.
  intros H [member [Hin H']]. exists member. split; [ exact Hin | ].
  exact (same_or_newer_trans rel id id' member H H').
Qed.

(* ======================================================================== *)
(* [has_older_version_in], for one compilation unit                         *)
(* ======================================================================== *)

Lemma has_older_version_in_unit_succ (steps : nat)
    (rel : Code_age_relation.t) (id : code_id) (candidates : list code_id) :
  has_older_version_in_unit (S steps) rel id candidates
  = if among id candidates then Some (Or_unknown.Known true)
    else
      match Code_age_relation.get_older_version_of rel id with
      | None => Some (Or_unknown.Known false)
      | Some older => has_older_version_in_unit steps rel older candidates
      end.
Proof. reflexivity. Qed.

Lemma has_older_version_in_unit_sound (steps : nat)
    (rel : Code_age_relation.t) (candidates : list code_id) :
  ∀ id : code_id,
    has_older_version_in_unit steps rel id candidates
    = Some (Or_unknown.Known true) ->
    set_allows rel candidates id.
Proof.
  induction steps as [ | steps IH]; intros id H; [ discriminate H | ].
  rewrite has_older_version_in_unit_succ in H.
  destruct (among id candidates) eqn:Hamong.
  - exists id. split; [ apply among_In; exact Hamong | constructor ].
  - destruct (Code_age_relation.get_older_version_of rel id) as [older | ]
      eqn:Holder; [ | discriminate ].
    apply (set_allows_newer rel candidates id older);
      [ apply same_or_newer_older; exact Holder | exact (IH older H) ].
Qed.

Lemma has_older_version_in_unit_unbind (steps : nat)
    (rel : Code_age_relation.t) (removed : code_id)
    (candidates : list code_id) :
  ∀ id : code_id,
    has_older_version_in_unit steps (unbind removed rel) id candidates
    = Some (Or_unknown.Known true) ->
    has_older_version_in_unit steps rel id candidates
    = Some (Or_unknown.Known true).
Proof.
  induction steps as [ | steps IH]; intros id H; [ discriminate H | ].
  rewrite has_older_version_in_unit_succ in *.
  destruct (among id candidates); [ exact H | ].
  rewrite get_older_version_of_unbind in H.
  destruct (Pos.eqb id removed); [ discriminate | ].
  destruct (Code_age_relation.get_older_version_of rel id) as [older | ];
    [ exact (IH older H) | discriminate ].
Qed.

Lemma has_older_version_in_unit_complete (n : nat) :
  ∀ (rel : Code_age_relation.t) (id : code_id) (candidates : list code_id),
    length rel <= n ->
    set_allows rel candidates id ->
    has_older_version_in_unit (S n) rel id candidates
    = Some (Or_unknown.Known true).
Proof.
  induction n as [ | n IH]; intros rel id candidates Hlen [member [Hin H]];
    rewrite has_older_version_in_unit_succ;
    destruct (among id candidates) eqn:Hamong; try reflexivity.
  - destruct rel as [ | binding rel]; [ | simpl in Hlen; lia ].
    destruct H as [id | id older id' Holder _]; [ | discriminate Holder ].
    apply among_In in Hin. congruence.
  - destruct (Pos.eqb_spec id member) as [-> | Hne];
      [ apply among_In in Hin; congruence | ].
    destruct H as [id | id older id' Holder H]; [ contradiction | ].
    rewrite Holder.
    apply (has_older_version_in_unit_unbind (S n) rel id candidates).
    apply IH.
    + pose proof (unbind_length_lt rel id older Holder). lia.
    + exists id'. split; [ exact Hin | ].
      destruct (same_or_newer_avoiding rel id older older id' Hne Holder H);
        assumption.
Qed.

Theorem has_older_version_in_unit_true :
  has_older_version_in_unit_true_statement.
Proof.
  intros steps rel id candidates Hlen. split.
  - apply has_older_version_in_unit_sound.
  - destruct steps as [ | steps]; [ lia | ].
    apply has_older_version_in_unit_complete. lia.
Qed.

Lemma well_formed_unbind (rel : Code_age_relation.t) (removed : code_id) :
  well_formed rel -> well_formed (unbind removed rel).
Proof.
  intros Hwf id older Holder Hback.
  rewrite get_older_version_of_unbind in Holder.
  destruct (Pos.eqb id removed); [ discriminate | ].
  exact (Hwf id older Holder (same_or_newer_unbind rel removed older id Hback)).
Qed.

(* From a code id that does not reach [removed], the function does not
   look at the older version of [removed]. *)
Lemma has_older_version_in_unit_unbind_eq (steps : nat)
    (rel : Code_age_relation.t) (removed : code_id)
    (candidates : list code_id) :
  ∀ id : code_id,
    ~ same_or_newer rel id removed ->
    has_older_version_in_unit steps (unbind removed rel) id candidates
    = has_older_version_in_unit steps rel id candidates.
Proof.
  induction steps as [ | steps IH]; intros id Hid; [ reflexivity | ].
  rewrite !has_older_version_in_unit_succ.
  destruct (among id candidates); [ reflexivity | ].
  rewrite get_older_version_of_unbind.
  destruct (Pos.eqb_spec id removed) as [-> | Hne];
    [ exfalso; apply Hid; constructor | ].
  destruct (Code_age_relation.get_older_version_of rel id) as [older | ]
    eqn:Holder; [ | reflexivity ].
  apply IH. intros Hreached. apply Hid.
  exact (Newer_version rel id older removed Holder Hreached).
Qed.

Lemma has_older_version_in_unit_answers_aux (n : nat) :
  ∀ (rel : Code_age_relation.t) (id : code_id) (candidates : list code_id),
    well_formed rel ->
    length rel <= n ->
    ∃ answer : bool,
      has_older_version_in_unit (S n) rel id candidates
      = Some (Or_unknown.Known answer).
Proof.
  induction n as [ | n IH]; intros rel id candidates Hwf Hlen;
    rewrite has_older_version_in_unit_succ;
    destruct (among id candidates); try (exists true; reflexivity).
  - destruct rel as [ | binding rel]; [ | simpl in Hlen; lia ].
    exists false. reflexivity.
  - destruct (Code_age_relation.get_older_version_of rel id) as [older | ]
      eqn:Holder; [ | exists false; reflexivity ].
    rewrite <- (has_older_version_in_unit_unbind_eq (S n) rel id candidates
                  older (Hwf id older Holder)).
    apply IH; [ apply well_formed_unbind; exact Hwf | ].
    pose proof (unbind_length_lt rel id older Holder). lia.
Qed.

Theorem has_older_version_in_unit_answers :
  has_older_version_in_unit_answers_statement.
Proof.
  intros steps rel id candidates Hwf Hlen.
  destruct steps as [ | steps]; [ lia | ].
  apply has_older_version_in_unit_answers_aux; [ exact Hwf | lia ].
Qed.

(* An answer [Known], with enough steps, says whether the code id is
   allowed. *)
Lemma has_older_version_in_unit_known (steps : nat)
    (rel : Code_age_relation.t) (id : code_id) (candidates : list code_id)
    (answer : bool) :
  length rel < steps ->
  has_older_version_in_unit steps rel id candidates
  = Some (Or_unknown.Known answer) ->
  (answer = true <-> set_allows rel candidates id).
Proof.
  intros Hlen H.
  rewrite <- (has_older_version_in_unit_true steps rel id candidates Hlen), H.
  split; [ intros ->; reflexivity | intros [= ->]; reflexivity ].
Qed.

(* For one compilation unit it never answers [Unknown]. *)
Lemma has_older_version_in_unit_not_unknown (steps : nat)
    (rel : Code_age_relation.t) (candidates : list code_id) :
  ∀ id : code_id,
    has_older_version_in_unit steps rel id candidates
    <> Some Or_unknown.Unknown.
Proof.
  induction steps as [ | steps IH]; intros id; [ discriminate | ].
  rewrite has_older_version_in_unit_succ.
  destruct (among id candidates); [ discriminate | ].
  destruct (Code_age_relation.get_older_version_of rel id) as [older | ];
    [ apply IH | discriminate ].
Qed.

(* ======================================================================== *)
(* [meet], for one compilation unit                                         *)
(* ======================================================================== *)

Lemma meet_in_unit_eq (steps : nat) (rel : Code_age_relation.t)
    (id1 id2 : code_id) :
  meet_in_unit steps rel id1 id2
  = if Pos.eqb id1 id2
    then Some (Ok id1)
    else
      let* older1 := has_older_version_in_unit steps rel id2 [id1] in
      let* older2 := has_older_version_in_unit steps rel id1 [id2] in
      Some
        match older1, older2 with
        | Or_unknown.Known true, _ => Ok id2
        | _, Or_unknown.Known true => Ok id1
        | Or_unknown.Known false, Or_unknown.Known false => Bottom
        | Or_unknown.Unknown, _ | _, Or_unknown.Unknown => Unknown
        end.
Proof. reflexivity. Qed.

(* The two questions that [meet] asks, answered. *)
Lemma meet_in_unit_questions (steps : nat) (rel : Code_age_relation.t)
    (id1 id2 : code_id) :
  well_formed rel ->
  length rel < steps ->
  ∃ answer1 answer2 : bool,
    has_older_version_in_unit steps rel id2 [id1]
    = Some (Or_unknown.Known answer1)
    ∧ has_older_version_in_unit steps rel id1 [id2]
      = Some (Or_unknown.Known answer2)
    ∧ (answer1 = true <-> same_or_newer rel id2 id1)
    ∧ (answer2 = true <-> same_or_newer rel id1 id2).
Proof.
  intros Hwf Hlen.
  destruct (has_older_version_in_unit_answers steps rel id2 [id1] Hwf Hlen)
    as [answer1 H1].
  destruct (has_older_version_in_unit_answers steps rel id1 [id2] Hwf Hlen)
    as [answer2 H2].
  exists answer1, answer2. split; [ exact H1 | ]. split; [ exact H2 | ].
  split.
  - rewrite <- set_allows_singleton.
    exact (has_older_version_in_unit_known steps rel id2 [id1] answer1 Hlen
             H1).
  - rewrite <- set_allows_singleton.
    exact (has_older_version_in_unit_known steps rel id1 [id2] answer2 Hlen
             H2).
Qed.

Theorem meet_in_unit_ok : meet_in_unit_ok_statement.
Proof.
  intros steps rel id1 id2 id Hwf Hlen. rewrite meet_in_unit_eq.
  destruct (Pos.eqb_spec id1 id2) as [-> | Hne].
  - split.
    + intros [= <-]. right. split; [ constructor | reflexivity ].
    + intros [[_ ->] | [_ ->]]; reflexivity.
  - destruct (meet_in_unit_questions steps rel id1 id2 Hwf Hlen)
      as [answer1 [answer2 [H1 [H2 [Hanswer1 Hanswer2]]]]].
    rewrite H1, H2.
    destruct answer1, answer2; simpl; split.
    + intros [= <-]. left. split; [ apply Hanswer1 | ]; reflexivity.
    + intros [[_ ->] | [H ->]]; [ reflexivity | ].
      exfalso. apply Hne.
      apply (well_formed_antisymmetric rel id1 id2 Hwf H).
      apply Hanswer1. reflexivity.
    + intros [= <-]. left. split; [ apply Hanswer1 | ]; reflexivity.
    + intros [[_ ->] | [H ->]]; [ reflexivity | ].
      apply Hanswer2 in H. discriminate.
    + intros [= <-]. right. split; [ apply Hanswer2 | ]; reflexivity.
    + intros [[H ->] | [_ ->]]; [ | reflexivity ].
      apply Hanswer1 in H. discriminate.
    + discriminate.
    + intros [[H _] | [H _]];
        [ apply Hanswer1 in H | apply Hanswer2 in H ]; discriminate.
Qed.

Theorem meet_in_unit_bottom : meet_in_unit_bottom_statement.
Proof.
  intros steps rel id1 id2 Hwf Hlen. rewrite meet_in_unit_eq.
  destruct (Pos.eqb_spec id1 id2) as [-> | Hne].
  - split; [ discriminate | ]. intros [H _]. exfalso. apply H. constructor.
  - destruct (meet_in_unit_questions steps rel id1 id2 Hwf Hlen)
      as [answer1 [answer2 [H1 [H2 [Hanswer1 Hanswer2]]]]].
    rewrite H1, H2.
    destruct answer1, answer2; simpl; split; try discriminate.
    + intros [_ H]. exfalso. apply H, Hanswer1. reflexivity.
    + intros [_ H]. exfalso. apply H, Hanswer1. reflexivity.
    + intros [H _]. exfalso. apply H, Hanswer2. reflexivity.
    + intros _. split; intros H;
        [ apply Hanswer2 in H | apply Hanswer1 in H ]; discriminate.
    + intros _. reflexivity.
Qed.

Theorem meet_in_unit_answers : meet_in_unit_answers_statement.
Proof.
  intros steps rel id1 id2 Hwf Hlen. rewrite meet_in_unit_eq.
  destruct (Pos.eqb id1 id2); [ left; eexists; reflexivity | ].
  destruct (meet_in_unit_questions steps rel id1 id2 Hwf Hlen)
    as [answer1 [answer2 [H1 [H2 _]]]].
  rewrite H1, H2.
  destruct answer1, answer2; simpl;
    try (left; eexists; reflexivity); right; reflexivity.
Qed.

(* ======================================================================== *)
(* [meet_set], for one compilation unit                                     *)
(* ======================================================================== *)

(* [should_keep_id] for one compilation unit. *)
Definition should_keep_id_in_unit (steps : nat) (rel : Code_age_relation.t)
    (other_ids : list code_id) (id : code_id) : option bool :=
  Code_age_relation.should_keep_id unit (fun _ => tt) tt (fun _ _ => true)
    steps rel (fun _ => None) other_ids id.

Lemma should_keep_id_in_unit_eq (steps : nat) (rel : Code_age_relation.t)
    (other_ids : list code_id) (id : code_id) :
  should_keep_id_in_unit steps rel other_ids id
  = match has_older_version_in_unit steps rel id other_ids with
    | Some (Or_unknown.Known answer) => Some answer
    | Some Or_unknown.Unknown => Some true
    | None => None
    end.
Proof. reflexivity. Qed.

Lemma meet_set_in_unit_eq (steps : nat) (rel : Code_age_relation.t)
    (ids1 ids2 : list code_id) :
  meet_set_in_unit steps rel ids1 ids2
  = if all_among_b ids1 ids2 && all_among_b ids2 ids1
    then Some (Or_bottom.Ok ids1)
    else
      let* kept2 :=
        Code_age_relation.filter_ids (should_keep_id_in_unit steps rel ids1)
          ids2
      in
      let* kept1 :=
        Code_age_relation.filter_ids (should_keep_id_in_unit steps rel ids2)
          ids1
      in
      match kept2 ++ kept1 with
      | [] => Some Or_bottom.Bottom
      | ids => Some (Or_bottom.Ok ids)
      end.
Proof. reflexivity. Qed.

(* A code id is kept exactly where the other set allows it. *)
Lemma should_keep_id_in_unit_true (steps : nat) (rel : Code_age_relation.t)
    (other_ids : list code_id) (id : code_id) :
  length rel < steps ->
  (should_keep_id_in_unit steps rel other_ids id = Some true
   <-> set_allows rel other_ids id).
Proof.
  intros Hlen. rewrite should_keep_id_in_unit_eq.
  rewrite <- (has_older_version_in_unit_true steps rel id other_ids Hlen).
  pose proof (has_older_version_in_unit_not_unknown steps rel other_ids id)
    as Hknown.
  destruct (has_older_version_in_unit steps rel id other_ids)
    as [[answer | ] | ]; [ | contradiction | ].
  - split; [ intros [= ->]; reflexivity | intros [= ->]; reflexivity ].
  - split; discriminate.
Qed.

(* What [filter_ids] keeps. *)
Lemma filter_ids_some (keep : code_id -> option bool) :
  ∀ ids kept : list code_id,
    Code_age_relation.filter_ids keep ids = Some kept ->
    ∀ id : code_id, id ∈ kept <-> id ∈ ids ∧ keep id = Some true.
Proof.
  induction ids as [ | first ids IH]; intros kept H id; simpl in H.
  - injection H as <-. simpl. tauto.
  - destruct (keep first) as [answer | ] eqn:Hfirst; [ | discriminate ].
    destruct (Code_age_relation.filter_ids keep ids) as [rest | ];
      [ | discriminate ].
    simpl in H. injection H as <-. specialize (IH rest eq_refl id).
    destruct answer; simpl; rewrite ?IH; split.
    + intros [<- | [Hin Hkeep]];
        [ split; [ left; reflexivity | exact Hfirst ] | ].
      split; [ right; exact Hin | exact Hkeep ].
    + intros [[<- | Hin] Hkeep]; [ left; reflexivity | right ].
      split; assumption.
    + intros [Hin Hkeep]. split; [ right; exact Hin | exact Hkeep ].
    + intros [[<- | Hin] Hkeep]; [ congruence | split; assumption ].
Qed.

Lemma filter_ids_total (keep : code_id -> option bool) (ids : list code_id) :
  (∀ id : code_id, ∃ answer : bool, keep id = Some answer) ->
  ∃ kept : list code_id, Code_age_relation.filter_ids keep ids = Some kept.
Proof.
  intros Hkeep. induction ids as [ | first ids [rest IH]]; simpl;
    [ eexists; reflexivity | ].
  destruct (Hkeep first) as [answer ->]. rewrite IH. simpl.
  eexists. reflexivity.
Qed.

(* The code ids kept allow exactly what both sets allow: a code id that
   both allow is a newer version of a member of each, and of those two
   one is a newer version of the other, and is kept. *)
Lemma kept_allow (steps : nat) (rel : Code_age_relation.t)
    (ids1 ids2 kept1 kept2 : list code_id) :
  length rel < steps ->
  Code_age_relation.filter_ids (should_keep_id_in_unit steps rel ids1) ids2
  = Some kept2 ->
  Code_age_relation.filter_ids (should_keep_id_in_unit steps rel ids2) ids1
  = Some kept1 ->
  ∀ id : code_id,
    set_allows rel (kept2 ++ kept1) id
    <-> set_allows rel ids1 id ∧ set_allows rel ids2 id.
Proof.
  intros Hlen H2 H1 id.
  pose proof (filter_ids_some _ ids2 kept2 H2) as Hkept2.
  pose proof (filter_ids_some _ ids1 kept1 H1) as Hkept1.
  split.
  - intros [member [Hin H]]. apply in_app_or in Hin.
    destruct Hin as [Hin | Hin].
    + apply Hkept2 in Hin. destruct Hin as [Hin Hkeep].
      apply (should_keep_id_in_unit_true steps rel ids1 member Hlen) in Hkeep.
      split; [ exact (set_allows_newer rel ids1 id member H Hkeep) | ].
      exists member. split; assumption.
    + apply Hkept1 in Hin. destruct Hin as [Hin Hkeep].
      apply (should_keep_id_in_unit_true steps rel ids2 member Hlen) in Hkeep.
      split; [ | exact (set_allows_newer rel ids2 id member H Hkeep) ].
      exists member. split; assumption.
  - intros [[member1 [Hin1 Hge1]] [member2 [Hin2 Hge2]]].
    destruct (older_versions_chain rel id member1 member2 Hge1 Hge2)
      as [Hchain | Hchain].
    + exists member1. split; [ | exact Hge1 ].
      apply in_or_app. right. apply Hkept1. split; [ exact Hin1 | ].
      apply (should_keep_id_in_unit_true steps rel ids2 member1 Hlen).
      exists member2. split; assumption.
    + exists member2. split; [ | exact Hge2 ].
      apply in_or_app. left. apply Hkept2. split; [ exact Hin2 | ].
      apply (should_keep_id_in_unit_true steps rel ids1 member2 Hlen).
      exists member1. split; assumption.
Qed.

Theorem meet_set_in_unit_ok : meet_set_in_unit_ok_statement.
Proof.
  intros steps rel ids1 ids2 ids Hlen H id.
  rewrite meet_set_in_unit_eq in H.
  destruct (all_among_b ids1 ids2 && all_among_b ids2 ids1) eqn:Hsame.
  - injection H as <-. apply andb_true_iff in Hsame.
    destruct Hsame as [Hsub _].
    pose proof (proj1 (FlambdaMapProofs.all_among ids1 ids2) Hsub) as Hsub'.
    split.
    + intros Hallows. split; [ exact Hallows | ].
      destruct Hallows as [member [Hin H]]. exists member.
      split; [ | exact H ]. apply among_In, Hsub', among_In, Hin.
    + intros [Hallows _]. exact Hallows.
  - destruct (Code_age_relation.filter_ids
                (should_keep_id_in_unit steps rel ids1) ids2)
      as [kept2 | ] eqn:H2; [ | discriminate ].
    destruct (Code_age_relation.filter_ids
                (should_keep_id_in_unit steps rel ids2) ids1)
      as [kept1 | ] eqn:H1; [ | discriminate ].
    simpl in H.
    pose proof (kept_allow steps rel ids1 ids2 kept1 kept2 Hlen H2 H1 id)
      as Hallow.
    destruct (kept2 ++ kept1) as [ | first rest]; [ discriminate | ].
    injection H as <-. exact Hallow.
Qed.

Theorem meet_set_in_unit_bottom : meet_set_in_unit_bottom_statement.
Proof.
  intros steps rel ids1 ids2 Hlen H id Hboth.
  rewrite meet_set_in_unit_eq in H.
  destruct (all_among_b ids1 ids2 && all_among_b ids2 ids1);
    [ discriminate | ].
  destruct (Code_age_relation.filter_ids
              (should_keep_id_in_unit steps rel ids1) ids2)
    as [kept2 | ] eqn:H2; [ | discriminate ].
  destruct (Code_age_relation.filter_ids
              (should_keep_id_in_unit steps rel ids2) ids1)
    as [kept1 | ] eqn:H1; [ | discriminate ].
  simpl in H.
  apply (kept_allow steps rel ids1 ids2 kept1 kept2 Hlen H2 H1 id) in Hboth.
  destruct (kept2 ++ kept1) as [ | first rest]; [ | discriminate ].
  destruct Hboth as [member [[] _]].
Qed.

Theorem meet_set_in_unit_empty : meet_set_in_unit_empty_statement.
Proof. intros steps rel. reflexivity. Qed.

Theorem meet_set_in_unit_answers : meet_set_in_unit_answers_statement.
Proof.
  intros steps rel ids1 ids2 Hwf Hlen. rewrite meet_set_in_unit_eq.
  destruct (all_among_b ids1 ids2 && all_among_b ids2 ids1);
    [ eexists; reflexivity | ].
  assert (Hkeep : ∀ (other_ids : list code_id) (id : code_id),
            ∃ answer : bool,
              should_keep_id_in_unit steps rel other_ids id = Some answer).
  { intros other_ids id. rewrite should_keep_id_in_unit_eq.
    destruct (has_older_version_in_unit_answers steps rel id other_ids Hwf
                Hlen) as [answer ->].
    exists answer. reflexivity. }
  destruct (filter_ids_total _ ids2 (Hkeep ids1)) as [kept2 ->].
  destruct (filter_ids_total _ ids1 (Hkeep ids2)) as [kept1 ->].
  simpl. destruct (kept2 ++ kept1); eexists; reflexivity.
Qed.

(* ======================================================================== *)
(* What [add] does not check                                                *)
(* ======================================================================== *)

Theorem add_unchecked : add_unchecked_statement.
Proof.
  cbv zeta.
  set (rel :=
         Code_age_relation.add
           (Code_age_relation.add Code_age_relation.empty 1%positive
              2%positive)
           2%positive 1%positive).
  split.
  - intros Hwf. apply (Hwf 1%positive 2%positive eq_refl).
    exact (Newer_version rel 2%positive 1%positive 1%positive eq_refl
             (Same_version rel 1%positive)).
  - assert (H : ∀ steps : nat,
              has_older_version_in_unit steps rel 1%positive [3%positive]
              = None
              ∧ has_older_version_in_unit steps rel 2%positive [3%positive]
                = None).
    { induction steps as [ | steps [IH1 IH2]]; [ split; reflexivity | ].
      rewrite !has_older_version_in_unit_succ. split; assumption. }
    intros steps. apply H.
Qed.

(* ======================================================================== *)
(* Several compilation units                                                *)
(* ======================================================================== *)

Theorem has_older_version_in_unknown :
  has_older_version_in_unknown_statement.
Proof.
  intros compilation_unit get_compilation_unit current_unit
    compilation_unit_equal steps rel resolver id candidates Hamong Holder
    Hunit Hresolver.
  unfold Code_age_relation.get_older_version_of in Holder. simpl.
  rewrite Hamong, Holder, Hunit, Hresolver. reflexivity.
Qed.

Theorem has_older_version_in_resolved :
  has_older_version_in_resolved_statement.
Proof.
  intros compilation_unit get_compilation_unit current_unit
    compilation_unit_equal steps rel rel' resolver id older candidates Hamong
    Holder Hunit Hresolver Holder' Hamong'.
  unfold Code_age_relation.get_older_version_of in Holder, Holder'.
  change (Code_age_relation.has_older_version_in compilation_unit
            get_compilation_unit current_unit compilation_unit_equal
            (S (S steps)) rel resolver id candidates)
    with (if among id candidates then Some (Or_unknown.Known true)
          else
            match lookup id rel with
            | None =>
              if compilation_unit_equal (get_compilation_unit id) current_unit
              then Some (Or_unknown.Known false)
              else
                match resolver (get_compilation_unit id) with
                | None => Some Or_unknown.Unknown
                | Some rel' =>
                  match lookup id rel' with
                  | None => Some (Or_unknown.Known false)
                  | Some older =>
                    Code_age_relation.has_older_version_in compilation_unit
                      get_compilation_unit current_unit
                      compilation_unit_equal (S steps) rel' resolver older
                      candidates
                  end
                end
            | Some older =>
              Code_age_relation.has_older_version_in compilation_unit
                get_compilation_unit current_unit compilation_unit_equal
                (S steps) rel resolver older candidates
            end).
  rewrite Hamong, Holder, Hunit, Hresolver, Holder'. simpl.
  rewrite Hamong'. reflexivity.
Qed.

Theorem meet_unknown : meet_unknown_statement.
Proof.
  intros compilation_unit get_compilation_unit current_unit
    compilation_unit_equal steps rel resolver id1 id2 Hne Hids.
  destruct (Hids id1 (or_introl eq_refl)) as [Holder1 [Hunit1 Hresolver1]].
  destruct (Hids id2 (or_intror eq_refl)) as [Holder2 [Hunit2 Hresolver2]].
  unfold Code_age_relation.meet.
  destruct (Pos.eqb_spec id1 id2); [ contradiction | ].
  rewrite (has_older_version_in_unknown compilation_unit get_compilation_unit
             current_unit compilation_unit_equal steps rel resolver id2 [id1]);
    try assumption.
  - rewrite (has_older_version_in_unknown compilation_unit
               get_compilation_unit current_unit compilation_unit_equal steps
               rel resolver id1 [id2]); try assumption; [ reflexivity | ].
    simpl. destruct (Pos.eqb_spec id1 id2); [ contradiction | reflexivity ].
  - simpl. destruct (Pos.eqb_spec id2 id1); [ congruence | reflexivity ].
Qed.

Theorem meet_set_unknown : meet_set_unknown_statement.
Proof.
  intros compilation_unit get_compilation_unit current_unit
    compilation_unit_equal steps rel resolver id1 id2 Hne Hids.
  destruct (Hids id1 (or_introl eq_refl)) as [Holder1 [Hunit1 Hresolver1]].
  destruct (Hids id2 (or_intror eq_refl)) as [Holder2 [Hunit2 Hresolver2]].
  assert (Hamong12 : among id1 [id2] = false).
  { simpl. destruct (Pos.eqb_spec id1 id2); [ contradiction | reflexivity ]. }
  assert (Hamong21 : among id2 [id1] = false).
  { simpl. destruct (Pos.eqb_spec id2 id1); [ congruence | reflexivity ]. }
  assert (Hsets : all_among_b [id1] [id2] = false).
  { unfold FlambdaMapSpec.all_among. cbn [forallb]. rewrite Hamong12.
    reflexivity. }
  unfold Code_age_relation.meet_set. rewrite Hsets.
  cbn [andb Code_age_relation.filter_ids].
  unfold Code_age_relation.should_keep_id.
  rewrite (has_older_version_in_unknown compilation_unit get_compilation_unit
             current_unit compilation_unit_equal steps rel resolver id2 [id1]
             Hamong21 Holder2 Hunit2 Hresolver2).
  rewrite (has_older_version_in_unknown compilation_unit get_compilation_unit
             current_unit compilation_unit_equal steps rel resolver id1 [id2]
             Hamong12 Holder1 Hunit1 Hresolver1).
  reflexivity.
Qed.
