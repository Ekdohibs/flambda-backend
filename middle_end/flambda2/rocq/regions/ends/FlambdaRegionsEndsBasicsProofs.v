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
   first facts about its definitions. Lists of variables of regions;
   the marks of a list of open regions; the plain frames of a stack
   and the kind of the frame of a mark; [frames_tight] under a push,
   under a cut, for fewer marks and for a smaller range. Nothing here
   is about a step of the machine, and nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaRegionsSpec FlambdaRegionsProofs FlambdaRegionsCallsSpec
  FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedSharedProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Lists of variables                                                       *)
(* ======================================================================== *)

Lemma variables_eqb_eq (xs : list variable) :
  ∀ ys : list variable, variables_eqb xs ys = true -> xs = ys.
Proof.
  induction xs as [ | x xs IH]; intros [ | y ys] H; simpl in H;
    try discriminate; [ reflexivity | ].
  apply andb_true_iff in H. destruct H as [Hxy Hrest].
  apply Pos.eqb_eq in Hxy. subst y. f_equal. exact (IH ys Hrest).
Qed.

(* The oldest of a list are its last. *)
Lemma oldest_of_suffix (xs ys : list variable) :
  oldest_of xs ys = true -> ∃ zs : list variable, ys = zs ++ xs.
Proof.
  unfold oldest_of. intros H. apply variables_eqb_eq in H.
  exists (firstn (length ys - length xs) ys).
  rewrite H at 2. symmetry. apply firstn_skipn.
Qed.

(* An open region parts the open regions: those above it, itself,
   those below it. *)
Lemma regions_above_below (r : variable) (open : list variable) :
  among r open = true ->
  open = regions_above r open ++ r :: regions_below r open.
Proof.
  induction open as [ | x open IH]; intros H; [ discriminate | ].
  rewrite among_cons in H. simpl.
  destruct (Pos.eqb x r) eqn:Hx.
  - apply Pos.eqb_eq in Hx. subst x. reflexivity.
  - rewrite Pos.eqb_sym, Hx in H. simpl in H.
    simpl. f_equal. exact (IH H).
Qed.

Lemma regions_above_among (r r' : variable) (open : list variable) :
  among r' (regions_above r open) = true -> among r' open = true.
Proof.
  induction open as [ | x open IH]; simpl; [ exact (fun H => H) | ].
  destruct (Pos.eqb x r); [ discriminate | ].
  rewrite !among_cons. intros H. apply orb_true_iff in H.
  apply orb_true_iff. destruct H as [H | H];
    [ left; exact H | right; exact (IH H) ].
Qed.

(* ======================================================================== *)
(* The marks of open regions                                                *)
(* ======================================================================== *)

(* The marks of two lists of open regions, one after the other. *)
Lemma open_marks_app (vars : env) (xs ys : list variable) (ms : list mark) :
  open_marks vars (xs ++ ys) = Some ms ->
  ∃ ms1 ms2 : list mark,
    ms = ms1 ++ ms2
    ∧ open_marks vars xs = Some ms1
    ∧ open_marks vars ys = Some ms2.
Proof.
  intros H. apply open_marks_marked in H.
  apply Forall2_app_inv_l in H. destruct H as (ms1 & ms2 & H1 & H2 & ->).
  exists ms1, ms2.
  split; [ reflexivity | ].
  split; apply open_marks_marked; assumption.
Qed.

Lemma open_marks_length (vars : env) (xs : list variable) (ms : list mark) :
  open_marks vars xs = Some ms -> length ms = length xs.
Proof.
  intros H. apply open_marks_marked in H. symmetry.
  exact (Forall2_length H).
Qed.

(* Marks that decrease, in two parts: each of the first part was drawn
   after each of the second. *)
Lemma marks_decrease_app (ms1 ms2 : list mark) :
  marks_decrease (ms1 ++ ms2) = true ->
  marks_decrease ms1 = true
  ∧ marks_decrease ms2 = true
  ∧ ∀ m1 m2 : mark, In m1 ms1 -> In m2 ms2 -> (m2 < m1)%positive.
Proof.
  induction ms1 as [ | m ms1 IH]; intros H.
  - split; [ reflexivity | ]. split; [ exact H | ].
    intros m1 m2 [].
  - change ((m :: ms1) ++ ms2) with (m :: ms1 ++ ms2) in H.
    apply marks_decrease_cons in H. destruct H as [Hlt Hrest].
    destruct (IH Hrest) as (H1 & H2 & Hcross).
    split.
    + apply marks_decrease_cons. split; [ | exact H1 ].
      intros m' Hin. apply Hlt. apply in_or_app. left. exact Hin.
    + split; [ exact H2 | ].
      intros m1 m2 [<- | Hin1] Hin2.
      * apply Hlt. apply in_or_app. right. exact Hin2.
      * exact (Hcross m1 m2 Hin1 Hin2).
Qed.

(* ======================================================================== *)
(* The plain frames, and the kind of the frame of a mark                    *)
(* ======================================================================== *)

Lemma plain_marks_cons (f : frame) (frames : list frame) :
  plain_marks (f :: frames)
  = if is_plain f then frame_mark f :: plain_marks frames
    else plain_marks frames.
Proof. unfold plain_marks. simpl. destruct (is_plain f); reflexivity. Qed.

Lemma plain_marks_app (above below : list frame) :
  plain_marks (above ++ below) = plain_marks above ++ plain_marks below.
Proof. unfold plain_marks. rewrite filter_app, map_app. reflexivity. Qed.

Lemma plain_marks_in (m : mark) (frames : list frame) :
  In m (plain_marks frames)
  <-> ∃ f : frame, In f frames ∧ is_plain f = true ∧ frame_mark f = m.
Proof.
  unfold plain_marks. rewrite in_map_iff. split.
  - intros (f & Hm & Hin). apply filter_In in Hin.
    destruct Hin as [Hin Hplain]. exists f. auto.
  - intros (f & Hin & Hplain & Hm). exists f.
    split; [ exact Hm | ]. apply filter_In. auto.
Qed.

(* The plain frames are those of the marks and the kinds: the local
   allocations of a frame do not count. *)
Lemma plain_marks_same (frames : list frame) :
  ∀ frames' : list frame,
    map (fun f => (frame_mark f, frame_kind f)) frames
    = map (fun f => (frame_mark f, frame_kind f)) frames' ->
    plain_marks frames = plain_marks frames'.
Proof.
  induction frames as [ | f frames IH]; intros [ | f' frames'] H;
    try discriminate; [ reflexivity | ].
  simpl in H. injection H as Hmark Hkind Hrest.
  rewrite !plain_marks_cons. unfold is_plain.
  rewrite Hmark, Hkind, (IH frames' Hrest). reflexivity.
Qed.

(* In a stack whose frames are in the order of their marks, the frame
   of a mark is found by its mark. *)
Lemma find_frame_of_mark (frames : list frame) :
  ∀ (top : symbol) (f : frame),
    frames_ordered_below top frames = true ->
    In f frames ->
    find (fun g => Pos.eqb (frame_mark g) (frame_mark f)) frames = Some f.
Proof.
  induction frames as [ | g frames IH]; intros top f Hord Hin;
    [ contradiction | ].
  simpl in Hord. apply andb_true_iff in Hord. destruct Hord as [_ Hrest].
  simpl. destruct Hin as [-> | Hin].
  - rewrite Pos.eqb_refl. reflexivity.
  - pose proof (frames_ordered_below_marks frames (frame_mark g) f Hrest Hin)
      as Hlt.
    destruct (Pos.eqb (frame_mark g) (frame_mark f)) eqn:Heq.
    + apply Pos.eqb_eq in Heq. lia.
    + exact (IH (frame_mark g) f Hrest Hin).
Qed.

Lemma kind_on_stack_in (frames : list frame) (top : symbol) (f : frame) :
  frames_ordered_below top frames = true ->
  In f frames ->
  kind_on_stack (frame_mark f) frames = Some (frame_kind f).
Proof.
  intros Hord Hin. unfold kind_on_stack.
  rewrite (find_frame_of_mark frames top f Hord Hin). reflexivity.
Qed.

(* A mark whose frame is of a [try] is the mark of no plain frame. *)
Lemma try_mark_not_plain (frames : list frame) (top : symbol) (m : mark) :
  frames_ordered_below top frames = true ->
  kind_on_stack m frames = Some Try_mark ->
  ~ In m (plain_marks frames).
Proof.
  intros Hord Hkind Hin. apply plain_marks_in in Hin.
  destruct Hin as (f & Hin & Hplain & <-).
  rewrite (kind_on_stack_in frames top f Hord Hin) in Hkind.
  unfold is_plain in Hplain. injection Hkind as Hkind.
  rewrite Hkind in Hplain. discriminate.
Qed.

(* The frames below a frame of the stack have marks that were drawn
   before its mark. *)
Lemma plain_marks_below_lt (above : list frame) (f : frame)
    (below : list frame) (top : symbol) (m : mark) :
  frames_ordered_below top (above ++ f :: below) = true ->
  In m (plain_marks below) ->
  (m < frame_mark f)%positive.
Proof.
  intros Hord Hin. apply plain_marks_in in Hin.
  destruct Hin as (g & Hin & _ & <-).
  exact
    (frames_ordered_below_marks below (frame_mark f) g
       (frames_ordered_tail above top f below Hord) Hin).
Qed.

(* ======================================================================== *)
(* [frames_tight]                                                           *)
(* ======================================================================== *)

(* A frame is pushed, and its mark is that of a region that is now
   open. *)
Lemma frames_tight_push (f : frame) (frames : list frame) (floor : symbol)
    (ms : list mark) :
  frames_tight frames floor None ms ->
  frames_tight (f :: frames) floor None (frame_mark f :: ms).
Proof.
  intros Htight m Hin Hfloor Hbefore. rewrite plain_marks_cons in Hin.
  destruct (is_plain f).
  - destruct Hin as [<- | Hin]; [ left; reflexivity | ].
    right. exact (Htight m Hin Hfloor Hbefore).
  - right. exact (Htight m Hin Hfloor Hbefore).
Qed.

(* Frames with the same marks and kinds. *)
Lemma frames_tight_same (frames frames' : list frame) (floor : symbol)
    (stop : option symbol) (ms : list mark) :
  map (fun f => (frame_mark f, frame_kind f)) frames
  = map (fun f => (frame_mark f, frame_kind f)) frames' ->
  frames_tight frames floor stop ms ->
  frames_tight frames' floor stop ms.
Proof.
  intros Hsame Htight m Hin. rewrite <- (plain_marks_same _ _ Hsame) in Hin.
  exact (Htight m Hin).
Qed.

(* Fewer marks: those left out are the marks of no plain frame. *)
Lemma frames_tight_fewer (frames : list frame) (floor : symbol)
    (stop : option symbol) (ms ms' : list mark) :
  frames_tight frames floor stop ms ->
  (∀ m : mark, In m ms -> In m (plain_marks frames) -> In m ms') ->
  frames_tight frames floor stop ms'.
Proof.
  intros Htight Hkept m Hin Hfloor Hbefore.
  exact (Hkept m (Htight m Hin Hfloor Hbefore) Hin).
Qed.

(* A smaller range. *)
Lemma frames_tight_range (frames : list frame) (floor floor' : symbol)
    (stop stop' : option symbol) (ms : list mark) :
  frames_tight frames floor stop ms ->
  (floor <= floor')%positive ->
  (∀ m : mark, before stop' m = true -> before stop m = true) ->
  frames_tight frames floor' stop' ms.
Proof.
  intros Htight Hfloor Hstop m Hin Hle Hbefore.
  apply (Htight m Hin); [ | exact (Hstop m Hbefore) ].
  apply Pos.leb_le. apply Pos.leb_le in Hle. lia.
Qed.

(* The stack is cut at the frame of an open region: what is left is
   tight for the regions below it, with no bound, whatever the bound
   was, if the bound was above the mark of the region. *)
Lemma frames_tight_cut (above : list frame) (f : frame) (below : list frame)
    (top floor : symbol) (stop : option symbol) (ms1 ms2 : list mark) :
  frames_ordered_below top (above ++ f :: below) = true ->
  marks_decrease (ms1 ++ frame_mark f :: ms2) = true ->
  frames_tight (above ++ f :: below) floor stop
    (ms1 ++ frame_mark f :: ms2) ->
  (∀ m : mark, (m < frame_mark f)%positive -> before stop m = true) ->
  frames_tight below floor None ms2.
Proof.
  intros Hord Hdecrease Htight Hstop m Hin Hfloor _.
  pose proof (plain_marks_below_lt above f below top m Hord Hin) as Hlt.
  assert (Hin' : In m (plain_marks (above ++ f :: below))).
  { rewrite plain_marks_app. apply in_or_app. right.
    rewrite plain_marks_cons. destruct (is_plain f);
      [ right; exact Hin | exact Hin ]. }
  pose proof (Htight m Hin' Hfloor (Hstop m Hlt)) as Hms.
  destruct (marks_decrease_app _ _ Hdecrease) as (_ & _ & Hcross).
  apply in_app_or in Hms. destruct Hms as [Hms | [Hms | Hms]].
  - pose proof (Hcross m (frame_mark f) Hms (or_introl eq_refl)). lia.
  - lia.
  - exact Hms.
Qed.

(* The bound of a stale state is above the mark of each open region
   but the newest, and of anything drawn before them. *)
Lemma stale_stop_above (stale : bool) (floor : symbol) (ms1 : list mark)
    (m0 : mark) (ms2 : list mark) (m : mark) :
  marks_decrease (ms1 ++ m0 :: ms2) = true ->
  (m < m0)%positive ->
  before (stale_stop stale floor (ms1 ++ m0 :: ms2)) m = true.
Proof.
  intros Hdecrease Hlt. unfold stale_stop.
  destruct stale; [ | reflexivity ].
  simpl. apply Pos.ltb_lt.
  destruct ms1 as [ | m1 ms1]; simpl; [ exact Hlt | ].
  destruct (marks_decrease_app _ _ Hdecrease) as (_ & _ & Hcross).
  pose proof
    (Hcross m1 m0 (or_introl eq_refl) (or_introl eq_refl)).
  lia.
Qed.

(* The oldest of the open regions, for a state that is stale: what is
   asked of the frames below the newest of them was asked before. *)
Lemma frames_tight_oldest (frames : list frame) (floor : symbol)
    (stop : option symbol) (ms1 ms2 : list mark) :
  marks_decrease (ms1 ++ ms2) = true ->
  frames_tight frames floor stop (ms1 ++ ms2) ->
  (∀ m : mark,
     before (Some (hd floor ms2)) m = true -> before stop m = true) ->
  frames_tight frames floor (stale_stop true floor ms2) ms2.
Proof.
  intros Hdecrease Htight Hstop m Hin Hfloor Hbefore.
  unfold stale_stop in Hbefore.
  pose proof (Htight m Hin Hfloor (Hstop m Hbefore)) as Hms.
  apply in_app_or in Hms. destruct Hms as [Hms | Hms]; [ | exact Hms ].
  exfalso. simpl in Hbefore. apply Pos.ltb_lt in Hbefore.
  apply Pos.leb_le in Hfloor.
  destruct ms2 as [ | m2 ms2]; simpl in Hbefore; [ lia | ].
  destruct (marks_decrease_app _ _ Hdecrease) as (_ & _ & Hcross).
  pose proof (Hcross m m2 Hms (or_introl eq_refl)). lia.
Qed.

(* ======================================================================== *)
(* The frames that were drawn before a symbol                               *)
(* ======================================================================== *)

Lemma frames_before_cons (bottom : symbol) (g : frame) (frames : list frame) :
  frames_before bottom (g :: frames)
  = if Pos.ltb (frame_mark g) bottom
    then (frame_mark g, frame_kind g) :: frames_before bottom frames
    else frames_before bottom frames.
Proof.
  unfold frames_before. simpl.
  destruct (Pos.ltb (frame_mark g) bottom); reflexivity.
Qed.

(* Those drawn before an earlier symbol are among them. *)
Lemma frames_before_filter (bottom' bottom : symbol) (frames : list frame) :
  (bottom' <= bottom)%positive ->
  frames_before bottom' frames
  = filter (fun p => Pos.ltb (fst p) bottom') (frames_before bottom frames).
Proof.
  intros Hle. induction frames as [ | g frames IH]; [ reflexivity | ].
  rewrite !frames_before_cons.
  destruct (Pos.ltb (frame_mark g) bottom) eqn:H0;
    destruct (Pos.ltb (frame_mark g) bottom') eqn:H'.
  - simpl. rewrite H'. f_equal. exact IH.
  - simpl. rewrite H'. exact IH.
  - apply Pos.ltb_lt in H'. apply Pos.ltb_ge in H0. lia.
  - exact IH.
Qed.

Lemma frames_before_le (bottom' bottom : symbol) (frames frames' : list frame) :
  (bottom' <= bottom)%positive ->
  frames_before bottom frames' = frames_before bottom frames ->
  frames_before bottom' frames' = frames_before bottom' frames.
Proof.
  intros Hle H.
  rewrite !(frames_before_filter bottom' bottom _ Hle), H. reflexivity.
Qed.

(* Whether a mark that was drawn before the symbol is on the stack, the
   kind of its frame, and whether its frame is plain, are read from
   those frames. *)
Lemma on_stack_before (m bottom : symbol) (frames : list frame) :
  (m < bottom)%positive ->
  on_stack m frames
  = existsb (fun p => Pos.eqb (fst p) m) (frames_before bottom frames).
Proof.
  intros Hlt. unfold on_stack.
  induction frames as [ | g frames IH]; [ reflexivity | ].
  rewrite frames_before_cons. simpl.
  destruct (Pos.ltb (frame_mark g) bottom) eqn:H0.
  - simpl. rewrite IH. reflexivity.
  - rewrite IH. destruct (Pos.eqb (frame_mark g) m) eqn:Heq; [ | reflexivity ].
    apply Pos.eqb_eq in Heq. apply Pos.ltb_ge in H0. lia.
Qed.

Lemma kind_on_stack_before (m bottom : symbol) (frames : list frame) :
  (m < bottom)%positive ->
  kind_on_stack m frames
  = option_map snd
      (find (fun p => Pos.eqb (fst p) m) (frames_before bottom frames)).
Proof.
  intros Hlt. unfold kind_on_stack.
  induction frames as [ | g frames IH]; [ reflexivity | ].
  rewrite frames_before_cons. simpl.
  destruct (Pos.ltb (frame_mark g) bottom) eqn:H0.
  - simpl. destruct (Pos.eqb (frame_mark g) m); [ reflexivity | exact IH ].
  - destruct (Pos.eqb (frame_mark g) m) eqn:Heq; [ | exact IH ].
    apply Pos.eqb_eq in Heq. apply Pos.ltb_ge in H0. lia.
Qed.

Lemma plain_before (m bottom : symbol) (frames : list frame) :
  (m < bottom)%positive ->
  In m (plain_marks frames)
  <-> In (m, Plain_mark) (frames_before bottom frames).
Proof.
  intros Hlt. rewrite plain_marks_in. unfold frames_before.
  rewrite in_map_iff. split.
  - intros (g & Hin & Hplain & Hm). exists g. split.
    + unfold is_plain in Hplain. rewrite Hm.
      destruct (frame_kind g); [ reflexivity | discriminate ].
    + apply filter_In. split; [ exact Hin | ].
      apply Pos.ltb_lt. lia.
  - intros (g & Heq & Hin). apply filter_In in Hin.
    destruct Hin as [Hin _]. injection Heq as Hm Hkind.
    exists g. split; [ exact Hin | ].
    split; [ unfold is_plain; rewrite Hkind; reflexivity | exact Hm ].
Qed.

(* So what is said of marks that were drawn before a symbol is kept
   where those frames are. *)
Section Kept.

Variables frames frames' : list frame.
Variable bottom : symbol.
Hypothesis Hkept : frames_before bottom frames' = frames_before bottom frames.

Lemma kept_on_stack (m : mark) :
  (m < bottom)%positive -> on_stack m frames' = on_stack m frames.
Proof.
  intros Hlt.
  rewrite !(on_stack_before m bottom _ Hlt), Hkept. reflexivity.
Qed.

Lemma kept_kind (m : mark) :
  (m < bottom)%positive -> kind_on_stack m frames' = kind_on_stack m frames.
Proof.
  intros Hlt.
  rewrite !(kind_on_stack_before m bottom _ Hlt), Hkept. reflexivity.
Qed.

Lemma kept_plain (m : mark) :
  (m < bottom)%positive ->
  In m (plain_marks frames') -> In m (plain_marks frames).
Proof.
  intros Hlt Hin. apply (plain_before m bottom _ Hlt).
  rewrite <- Hkept. apply (plain_before m bottom _ Hlt). exact Hin.
Qed.

Lemma kept_marks_placed (ms : list mark) :
  (∀ m : mark, In m ms -> (m < bottom)%positive) ->
  marks_placed frames ms = true ->
  marks_placed frames' ms = true.
Proof.
  intros Hbelow Hplaced. unfold marks_placed in *.
  apply andb_true_iff in Hplaced. destruct Hplaced as [Hon Hdecrease].
  apply andb_true_iff. split; [ | exact Hdecrease ].
  apply forallb_forall. intros m Hin.
  rewrite (kept_on_stack m (Hbelow m Hin)).
  exact (proj1 (forallb_forall _ _) Hon m Hin).
Qed.

Lemma kept_ends_placed (vars : env) (es : ends_state) (ms : list mark)
    (floor : symbol) :
  (∀ m : mark, In m ms -> (m < bottom)%positive) ->
  ends_placed vars frames es ms floor ->
  ends_placed vars frames' es ms floor.
Proof.
  intros Hbelow (Hmarks & Hplaced & Hfloor & Hopen & Htries & Hkinds).
  split; [ exact Hmarks | ].
  split; [ exact (kept_marks_placed ms Hbelow Hplaced) | ].
  split; [ exact Hfloor | ]. split; [ exact Hopen | ].
  split; [ exact Htries | ].
  intros r m Hin Htry.
  rewrite (kept_kind m (Hbelow m (in_combine_r _ _ _ _ Hin))).
  exact (Hkinds r m Hin Htry).
Qed.

Lemma kept_tight (floor stop : symbol) (ms : list mark) :
  (stop <= bottom)%positive ->
  frames_tight frames floor (Some stop) ms ->
  frames_tight frames' floor (Some stop) ms.
Proof.
  intros Hstop Htight m Hin Hfloor Hbefore.
  apply (Htight m); [ | exact Hfloor | exact Hbefore ].
  apply kept_plain; [ | exact Hin ].
  simpl in Hbefore. apply Pos.ltb_lt in Hbefore. lia.
Qed.

End Kept.

(* ======================================================================== *)
(* Frames with the same marks and kinds, a frame pushed, a stack cut        *)
(* ======================================================================== *)

Lemma on_stack_same (m : mark) (frames : list frame) :
  ∀ frames' : list frame,
    map (fun f => (frame_mark f, frame_kind f)) frames
    = map (fun f => (frame_mark f, frame_kind f)) frames' ->
    on_stack m frames = on_stack m frames'.
Proof.
  induction frames as [ | f frames IH]; intros [ | f' frames'] H;
    try discriminate; [ reflexivity | ].
  cbn in H. injection H as Hmark Hkind Hrest.
  unfold on_stack in *. cbn. rewrite Hmark, (IH frames' Hrest). reflexivity.
Qed.

Lemma kind_on_stack_same (m : mark) (frames : list frame) :
  ∀ frames' : list frame,
    map (fun f => (frame_mark f, frame_kind f)) frames
    = map (fun f => (frame_mark f, frame_kind f)) frames' ->
    kind_on_stack m frames = kind_on_stack m frames'.
Proof.
  induction frames as [ | f frames IH]; intros [ | f' frames'] H;
    try discriminate; [ reflexivity | ].
  cbn in H. injection H as Hmark Hkind Hrest.
  unfold kind_on_stack in *. cbn. rewrite Hmark.
  destruct (Pos.eqb (frame_mark f') m); [ cbn; rewrite Hkind; reflexivity | ].
  exact (IH frames' Hrest).
Qed.

Lemma frames_before_same (bottom : symbol) (frames : list frame) :
  ∀ frames' : list frame,
    map (fun f => (frame_mark f, frame_kind f)) frames
    = map (fun f => (frame_mark f, frame_kind f)) frames' ->
    frames_before bottom frames = frames_before bottom frames'.
Proof.
  induction frames as [ | f frames IH]; intros [ | f' frames'] H;
    try discriminate; [ reflexivity | ].
  cbn in H. injection H as Hmark Hkind Hrest.
  rewrite !frames_before_cons, Hmark, Hkind, (IH frames' Hrest). reflexivity.
Qed.

(* A frame that is pushed was not drawn before a symbol that is at or
   before its mark. *)
Lemma frames_before_push (bottom : symbol) (f : frame) (frames : list frame) :
  (bottom <= frame_mark f)%positive ->
  frames_before bottom (f :: frames) = frames_before bottom frames.
Proof.
  intros Hle. rewrite frames_before_cons.
  destruct (Pos.ltb (frame_mark f) bottom) eqn:Hlt; [ | reflexivity ].
  apply Pos.ltb_lt in Hlt. lia.
Qed.

(* The frames that a cut takes away were not drawn before a symbol
   that is at or before the mark of the cut. *)
Lemma frames_before_cut (above : list frame) :
  ∀ (top : symbol) (f : frame) (below : list frame) (bottom : symbol),
    frames_ordered_below top (above ++ f :: below) = true ->
    (bottom <= frame_mark f)%positive ->
    frames_before bottom (above ++ f :: below) = frames_before bottom below.
Proof.
  induction above as [ | g above IH]; intros top f below bottom Hord Hle.
  - exact (frames_before_push bottom f below Hle).
  - change ((g :: above) ++ f :: below) with (g :: above ++ f :: below).
    pose proof
      (frames_above_marks (g :: above) top f below g Hord
         (or_introl eq_refl))
      as Hg.
    rewrite frames_before_push; [ | lia ].
    cbn in Hord. apply andb_true_iff in Hord. destruct Hord as [_ Hrest].
    exact (IH (frame_mark g) f below bottom Hrest Hle).
Qed.

(* Pairs of the last parts of two lists that are paired. *)
Lemma combine_app_r {A B : Type} (l1 : list A) :
  ∀ (m1 : list B) (l2 : list A) (m2 : list B) (a : A) (b : B),
    length m1 = length l1 ->
    In (a, b) (combine l2 m2) ->
    In (a, b) (combine (l1 ++ l2) (m1 ++ m2)).
Proof.
  induction l1 as [ | x l1 IH]; intros [ | y m1] l2 m2 a b Hlength Hin;
    cbn in *; try discriminate; [ exact Hin | ].
  right. injection Hlength as Hlength. exact (IH m1 l2 m2 a b Hlength Hin).
Qed.

(* ======================================================================== *)
(* [ends_placed] under other variables and other frames                     *)
(* ======================================================================== *)

Lemma ends_placed_same (vars : env) (frames frames' : list frame)
    (es : ends_state) (ms : list mark) (floor : symbol) :
  map (fun f => (frame_mark f, frame_kind f)) frames
  = map (fun f => (frame_mark f, frame_kind f)) frames' ->
  ends_placed vars frames es ms floor ->
  ends_placed vars frames' es ms floor.
Proof.
  intros Hsame (Hmarks & Hplaced & Hfloor & Hopen & Htries & Hkinds).
  split; [ exact Hmarks | ]. split.
  - unfold marks_placed in *. apply andb_true_iff in Hplaced.
    destruct Hplaced as [Hon Hdecrease].
    apply andb_true_iff. split; [ | exact Hdecrease ].
    apply forallb_forall. intros m Hin.
    rewrite <- (on_stack_same m frames frames' Hsame).
    exact (proj1 (forallb_forall _ _) Hon m Hin).
  - split; [ exact Hfloor | ]. split; [ exact Hopen | ].
    split; [ exact Htries | ].
    intros r m Hin Htry.
    rewrite <- (kind_on_stack_same m frames frames' Hsame).
    exact (Hkinds r m Hin Htry).
Qed.

(* A variable that is not the variable of a region is bound. *)
Lemma ends_placed_bind (vars : env) (frames : list frame) (es : ends_state)
    (ms : list mark) (floor : symbol) (x : variable) (v : value) :
  among x (es_regions es) = false ->
  ends_placed vars frames es ms floor ->
  ends_placed ((x, v) :: vars) frames es ms floor.
Proof.
  intros Hnew (Hmarks & Hrest).
  split; [ | exact Hrest ].
  destruct Hrest as (_ & _ & Hopen & _).
  rewrite open_marks_bind; [ exact Hmarks | ].
  destruct (among x (es_open es)) eqn:Hx; [ | reflexivity ].
  unfold FlambdaMapSpec.all_among in Hopen.
  pose proof (proj1 (forallb_forall _ _) Hopen x (proj1 (among_In _ _) Hx))
    as Hregion.
  cbn beta in Hregion. rewrite Hnew in Hregion. discriminate.
Qed.

(* Pairs of the first parts of two lists that are paired. *)
Lemma combine_app_l {A B : Type} (l1 : list A) :
  ∀ (m1 : list B) (l2 : list A) (m2 : list B) (a : A) (b : B),
    In (a, b) (combine l1 m1) ->
    In (a, b) (combine (l1 ++ l2) (m1 ++ m2)).
Proof.
  induction l1 as [ | x l1 IH]; intros [ | y m1] l2 m2 a b Hin;
    cbn in *; try contradiction.
  destruct Hin as [Heq | Hin]; [ left; exact Heq | right ].
  exact (IH m1 l2 m2 a b Hin).
Qed.

(* No frame of a list is plain. *)
Lemma filter_plain_nil (frames : list frame) :
  (∀ g : frame, In g frames -> is_plain g = false) ->
  filter is_plain frames = [].
Proof.
  induction frames as [ | g frames IH]; intros H; [ reflexivity | ].
  cbn. rewrite (H g (or_introl eq_refl)).
  exact (IH (fun g' Hin => H g' (or_intror Hin))).
Qed.
