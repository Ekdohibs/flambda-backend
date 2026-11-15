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

(* Lemmas for the proofs of the statements of FlambdaRegionsEndsSpec.v:
   the marks of the regions that a handler is entered with. By the
   ordinary test of a jump they are chosen, in order, among the marks
   of the open regions, and every plain frame drawn since the code
   was entered is theirs; those of a handler that is entered stale
   with the oldest of the open regions are the oldest of the marks,
   and the plain frames below the newest of them are theirs. And
   where the code returns, no plain frame was drawn since it was
   entered. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedOutlivesProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs FlambdaRegionsEndsHandlersProofs.
Import ListNotations.

(* A symbol before the mark of an open region, or before the floor, is
   before where a state that is stale stops. *)
Lemma stale_stop_below (stale : bool) (floor : symbol) (ms : list mark)
    (m m' : mark) :
  marks_decrease ms = true ->
  forallb (fun m0 => Pos.leb floor m0) ms = true ->
  In m' ms ∨ m' = floor ->
  (m < m')%positive ->
  before (stale_stop stale floor ms) m = true.
Proof.
  intros Hdecrease Hfloor Hm' Hlt. unfold stale_stop.
  destruct stale; [ | reflexivity ]. cbn [before]. apply Pos.ltb_lt.
  destruct ms as [ | m1 ms]; cbn [hd].
  - destruct Hm' as [[] | ->]. exact Hlt.
  - assert (Hle : (m' <= m1)%positive).
    { destruct Hm' as [[<- | Hin] | ->].
      - lia.
      - pose proof (marks_decrease_lt _ _ Hdecrease _ Hin). lia.
      - cbn [forallb] in Hfloor. apply andb_true_iff in Hfloor.
        destruct Hfloor as [H _]. apply Pos.leb_le in H. exact H. }
    lia.
Qed.

(* A handler that is entered stale, with the oldest of the open
   regions. *)
Lemma oldest_marks (frames : list frame) (vars : env) (es : ends_state)
    (ms : list mark) (floor : symbol) (stop : option symbol)
    (named : list variable) :
  ends_placed vars frames es ms floor ->
  frames_tight frames floor stop ms ->
  (∀ m m' : mark,
     In m' ms ∨ m' = floor -> (m < m')%positive -> before stop m = true) ->
  oldest_of named (es_open es) = true ->
  ∃ ms_h : list mark,
    open_marks vars named = Some ms_h
    ∧ marks_placed frames ms_h = true
    ∧ (∀ r : variable, In r named -> In r (es_open es))
    ∧ frames_tight frames floor (stale_stop true floor ms_h) ms_h.
Proof.
  intros (Hmarks & Hmp & Hfloor & _) Htight Hstop Holdest.
  destruct (oldest_of_suffix _ _ Holdest) as [zs Hopen].
  rewrite Hopen in Hmarks.
  destruct (open_marks_app _ _ _ _ Hmarks) as (ms1 & ms2 & -> & _ & Hms2).
  unfold marks_placed in Hmp. apply andb_true_iff in Hmp.
  destruct Hmp as [Hon Hdecrease].
  destruct (marks_decrease_app _ _ Hdecrease) as (_ & Hd2 & _).
  exists ms2. split; [ exact Hms2 | ]. split.
  - unfold marks_placed. apply andb_true_iff. split; [ | exact Hd2 ].
    apply forallb_forall. intros m Hm.
    apply (proj1 (forallb_forall _ _) Hon). apply in_or_app. right. exact Hm.
  - split.
    + intros r Hr. rewrite Hopen. apply in_or_app. right. exact Hr.
    + apply (frames_tight_oldest frames floor stop ms1 ms2 Hdecrease Htight).
      intros m Hbefore. cbn [before] in Hbefore. apply Pos.ltb_lt in Hbefore.
      apply (Hstop m (hd floor ms2)); [ | exact Hbefore ].
      destruct ms2 as [ | m2 ms2]; cbn [hd]; [ right; reflexivity | ].
      left. apply in_or_app. right. left. reflexivity.
Qed.

(* A handler that is entered by the ordinary test of a jump: the
   regions that it is entered with are among the open regions, in
   order, and every open region that is not one of them is of a
   [try]. *)
Lemma selected_marks (frames : list frame) (top : symbol) (vars : env)
    (es : ends_state) (ms : list mark) (floor : symbol)
    (named : list variable) :
  frames_ordered_below top frames = true ->
  ends_placed vars frames es ms floor ->
  frames_tight frames floor None ms ->
  in_order named (es_open es) = true ->
  forallb (fun r => among r (es_tries es) || among r named) (es_open es)
  = true ->
  ∃ ms_h : list mark,
    open_marks vars named = Some ms_h
    ∧ marks_placed frames ms_h = true
    ∧ (∀ r : variable, In r named -> In r (es_open es))
    ∧ frames_tight frames floor None ms_h.
Proof.
  intros Hordered (Hmarks & Hmp & Hfloor & _ & _ & Hkinds) Htight Horder Hall.
  destruct
    (in_order_marks vars (es_open es) named ms Horder
       (proj1 (open_marks_marked vars _ ms) Hmarks))
    as (ms_h & Hmarked & Hsel).
  pose proof (proj2 (open_marks_marked vars named ms_h) Hmarked) as Hms_h.
  exists ms_h. split; [ exact Hms_h | ].
  split; [ exact (selection_placed frames ms_h ms Hsel Hmp) | ]. split.
  - intros r Hr. exact (in_order_in _ _ r Horder Hr).
  - apply (frames_tight_fewer frames floor None ms ms_h Htight).
    intros m Hm Hplain.
    destruct
      (in_combine_exists (es_open es) ms m (open_marks_length _ _ _ Hmarks) Hm)
      as [r Hpair].
    pose proof
      (proj1 (forallb_forall _ _) Hall r (in_combine_l _ _ _ _ Hpair)) as Hr.
    cbn beta in Hr. apply orb_true_iff in Hr. destruct Hr as [Htry | Hnamed].
    + exfalso.
      exact
        (try_mark_not_plain frames top m Hordered (Hkinds r m Hpair Htry)
           Hplain).
    + destruct (combine_marked vars _ ms r m Hmarks Hpair) as [_ Hlookup].
      exact
        (in_combine_r _ _ _ _
           (marked_combine vars named ms_h r m Hms_h
              (proj1 (among_In _ _) Hnamed) Hlookup)).
Qed.

(* What is asked of the frames of a state that is not stale is asked of
   them where it is. *)
Lemma frames_tight_stale (frames : list frame) (floor : symbol)
    (stop : option symbol) (ms : list mark) :
  frames_tight frames floor None ms -> frames_tight frames floor stop ms.
Proof.
  intros Htight m Hplain Hfloor _. exact (Htight m Hplain Hfloor eq_refl).
Qed.

(* Where every open region is of a [try], and the state is not stale,
   no plain frame was drawn since the code was entered. *)
Lemma return_no_plain (frames : list frame) (top : symbol) (vars : env)
    (es : ends_state) (ms : list mark) (floor : symbol) :
  frames_ordered_below top frames = true ->
  ends_placed vars frames es ms floor ->
  frames_tight frames floor None ms ->
  forallb (fun r => among r (es_tries es)) (es_open es) = true ->
  ∀ m : mark, In m (plain_marks frames) -> Pos.ltb m floor = true.
Proof.
  intros Hordered (Hmarks & _ & _ & _ & _ & Hkinds) Htight Hall m Hplain.
  destruct (Pos.ltb m floor) eqn:Hlt; [ reflexivity | exfalso ].
  apply Pos.ltb_ge in Hlt.
  assert (Hin : In m ms).
  { apply (Htight m Hplain); [ apply Pos.leb_le; exact Hlt | reflexivity ]. }
  destruct
    (in_combine_exists (es_open es) ms m (open_marks_length _ _ _ Hmarks) Hin)
    as [r Hpair].
  pose proof
    (proj1 (forallb_forall _ _) Hall r (in_combine_l _ _ _ _ Hpair)) as Htry.
  cbn beta in Htry.
  exact
    (try_mark_not_plain frames top m Hordered (Hkinds r m Hpair Htry) Hplain).
Qed.
