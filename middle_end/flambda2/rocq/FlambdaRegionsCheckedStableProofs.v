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

(* Lemmas for the proofs of the statements of FlambdaRegionsCheckedSpec.v:
   what a continuation is as its entry says to ([cont_ok]) is kept when the
   store grows and no frame is popped, when a region is ended, and when the
   user changes. The end of a region is by the order of the frames
   (FlambdaRegionsCallsSpec.frames_ordered). Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs.
Import ListNotations.

(* ======================================================================== *)
(* Marks                                                                    *)
(* ======================================================================== *)

Lemma marks_decrease_lt (m : mark) (ms : list mark) :
  marks_decrease (m :: ms) = true ->
  ∀ m' : mark, In m' ms -> (m' < m)%positive.
Proof.
  revert m. induction ms as [ | m1 ms IH]; intros m; [ contradiction | ].
  cbn [marks_decrease]. intros H. apply andb_true_iff in H as [Hlt Hrest].
  apply Pos.ltb_lt in Hlt. intros m' [<- | Hin]; [ exact Hlt | ].
  pose proof (IH m1 Hrest m' Hin). lia.
Qed.

Lemma marks_decrease_tail (m : mark) (ms : list mark) :
  marks_decrease (m :: ms) = true -> marks_decrease ms = true.
Proof.
  cbn [marks_decrease]. intros H. apply andb_true_iff in H as [_ H]. exact H.
Qed.

(* The place of an open region: its mark is one of the marks, and its
   bound is at most the bound of the newest and at most every mark
   above its own. *)
Lemma region_place_bound (opens : list variable) :
  ∀ (ms : list mark) (above : symbol) (r : variable) (m : mark)
    (stop : option symbol),
    region_place opens ms (Some above) r = Some (m, stop) ->
    (∀ m' : mark, In m' ms -> (m' < above)%positive) ->
    marks_decrease ms = true ->
    ∃ b : symbol,
      stop = Some b
      ∧ (b <= above)%positive
      ∧ In m ms
      ∧ ∀ m0 : mark, In m0 ms -> (m < m0)%positive -> (b <= m0)%positive.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m stop;
    [ discriminate | ].
  destruct ms as [ | m1 ms]; [ discriminate | ]. cbn [region_place].
  destruct (Pos.eqb x r).
  - intros [= <- <-] Hbelow Hdecrease. exists above.
    split; [ reflexivity | ]. split; [ lia | ].
    split; [ left; reflexivity | ].
    intros m0 [<- | Hin] Hlt; [ lia | ].
    pose proof (marks_decrease_lt m1 ms Hdecrease m0 Hin). lia.
  - intros Hplace Hbelow Hdecrease.
    destruct (IH ms m1 r m stop Hplace (marks_decrease_lt m1 ms Hdecrease)
                (marks_decrease_tail m1 ms Hdecrease))
      as (b & -> & Hb & Hin & Hup).
    exists b. split; [ reflexivity | ].
    pose proof (Hbelow m1 (or_introl eq_refl)).
    split; [ lia | ]. split; [ right; exact Hin | ].
    intros m0 [<- | Hin0] Hlt; [ exact Hb | exact (Hup m0 Hin0 Hlt) ].
Qed.

(* The same with any bound of the newest: the mark is one of the
   marks. *)
Lemma region_place_in (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable) (m : mark)
    (stop : option symbol),
    region_place opens ms above r = Some (m, stop) -> In m ms.
Proof.
  induction opens as [ | x opens IH]; intros ms above r m stop;
    [ discriminate | ].
  destruct ms as [ | m1 ms]; [ discriminate | ]. cbn [region_place].
  destruct (Pos.eqb x r).
  - intros [= <- _]. left. reflexivity.
  - intros H. right. exact (IH _ _ _ _ _ H).
Qed.

(* The mark of the oldest open region is one of the marks, and at most
   each. *)
Lemma lowest_least (ms : list mark) :
  ∀ b : mark,
    lowest ms = Some b ->
    marks_decrease ms = true ->
    In b ms ∧ ∀ m0 : mark, In m0 ms -> (b <= m0)%positive.
Proof.
  induction ms as [ | m ms IH]; intros b; [ discriminate | ].
  destruct ms as [ | m1 ms].
  - intros [= <-] _. split; [ left; reflexivity | ].
    intros m0 [<- | []]. lia.
  - change (lowest (m :: m1 :: ms)) with (lowest (m1 :: ms)).
    intros Hlowest Hdecrease.
    destruct (IH b Hlowest (marks_decrease_tail _ _ Hdecrease)) as [Hin Hle].
    split; [ right; exact Hin | ].
    intros m0 [<- | Hin0]; [ | exact (Hle m0 Hin0) ].
    pose proof (marks_decrease_lt m (m1 :: ms) Hdecrease b Hin). lia.
Qed.

Lemma lowest_none (ms : list mark) : lowest ms = None -> ms = [].
Proof.
  induction ms as [ | m ms IH]; [ reflexivity | ].
  destruct ms as [ | m1 ms]; [ discriminate | ].
  change (lowest (m :: m1 :: ms)) with (lowest (m1 :: ms)).
  intros H. discriminate (IH H).
Qed.

(* The bound of what lives as long as the code, with a bound of the
   newest region above every mark. *)
Lemma caller_bound_least (ms : list mark) (above : symbol) :
  (∀ m' : mark, In m' ms -> (m' < above)%positive) ->
  marks_decrease ms = true ->
  ∃ b : symbol,
    caller_bound ms (Some above) = Some b
    ∧ (b <= above)%positive
    ∧ ∀ m0 : mark, In m0 ms -> (b <= m0)%positive.
Proof.
  intros Hbelow Hdecrease. unfold caller_bound.
  destruct (lowest ms) as [b | ] eqn:Hlowest.
  - destruct (lowest_least ms b Hlowest Hdecrease) as [Hin Hle].
    exists b. split; [ reflexivity | ].
    pose proof (Hbelow b Hin). split; [ lia | exact Hle ].
  - exists above. split; [ reflexivity | ]. split; [ lia | ].
    intros m0 Hin. rewrite (lowest_none ms Hlowest) in Hin. contradiction.
Qed.

(* ======================================================================== *)
(* The store grows                                                          *)
(* ======================================================================== *)

(* No frame is popped: a symbol of the heap stays one, a symbol that a
   frame binds stays bound, and the marks drawn so far are on the stack
   as they were. *)
Record store_grows (rs rs' : region_state) : Prop := Store_grows {
  grows_next : (rs_next rs <= rs_next rs')%positive;
  grows_heap :
    ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true;
  grows_local :
    ∀ s : symbol,
      local_in (rs_frames rs) s = true -> local_in (rs_frames rs') s = true;
  grows_stack :
    ∀ m : mark,
      (m < rs_next rs)%positive ->
      on_stack m (rs_frames rs') = on_stack m (rs_frames rs)
}.

Section Grows.

Variables rs rs' : region_state.
Hypothesis Hgrows : store_grows rs rs'.

Lemma grows_heap_value (v : value) :
  value_all (heap_symbol rs) v = true -> value_all (heap_symbol rs') v = true.
Proof. apply value_all_mono. exact (grows_heap rs rs' Hgrows). Qed.

Lemma grows_within (stop : option symbol) (v : value) :
  value_within rs stop v = true -> value_within rs' stop v = true.
Proof.
  apply value_all_mono. intros s. unfold symbol_within. intros H.
  apply orb_true_iff in H as [H | H].
  - rewrite (grows_heap rs rs' Hgrows s H). reflexivity.
  - apply andb_true_iff in H as [Hlocal Hbefore].
    rewrite (grows_local rs rs' Hgrows s Hlocal), Hbefore. apply orb_true_r.
Qed.

Lemma grows_lifetime (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  (∀ m : mark, In m ms -> (m < rs_next rs)%positive) ->
  lifetime_holds rs opens ms above l v ->
  lifetime_holds rs' opens ms above l v.
Proof.
  intros Hms. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact (grows_heap_value v).
  - exact (grows_within _ v).
  - destruct (region_place opens ms above r) as [[m stop] | ] eqn:Hplace;
      [ | exact (fun H => H) ].
    intros H Hon. apply grows_within. apply H.
    rewrite <- (grows_stack rs rs' Hgrows m
                  (Hms m (region_place_in _ _ _ _ _ _ Hplace))).
    exact Hon.
  - exact (fun H => H).
Qed.

Lemma grows_placed (ms : list mark) :
  (∀ m : mark, In m ms -> (m < rs_next rs)%positive) ->
  marks_placed (rs_frames rs) ms = true ->
  marks_placed (rs_frames rs') ms = true.
Proof.
  intros Hms. unfold marks_placed. intros H.
  apply andb_true_iff in H as [Hon Hdecrease].
  apply andb_true_iff. split; [ | exact Hdecrease ].
  apply forallb_forall. intros m Hin.
  rewrite (grows_stack rs rs' Hgrows m (Hms m Hin)).
  exact (forallb_in _ _ m Hon Hin).
Qed.

Variable cert : certificate.
Variable codes : list (code_id * code).

Lemma forallb_ltb_in (next : symbol) (ms : list mark) (m : mark) :
  forallb (fun m => Pos.ltb m next) ms = true -> In m ms ->
  (m < next)%positive.
Proof.
  intros H Hin. apply Pos.ltb_lt. exact (forallb_in _ _ m H Hin).
Qed.

Lemma grows_made_at (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  (u_limit u <= rs_next rs)%positive ->
  made_at cert codes rs u vars recursive cs base ms next ->
  made_at cert codes rs' u vars recursive cs base ms next.
Proof.
  intros Hlimit
    (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11).
  repeat (split; [ assumption | ]). split; [ | split; assumption ].
  intros x l v Hl Hv. apply grows_lifetime; [ | exact (H9 x l v Hl Hv) ].
  intros m Hin. pose proof (forallb_ltb_in next ms m H7 Hin).
  apply Pos.leb_le in H8. lia.
Qed.

Lemma grows_called_from (u : user) (a : caller) :
  (u_limit u <= rs_next rs)%positive ->
  called_from rs u a -> called_from rs' u a.
Proof.
  intros Hlimit (H1 & H2 & H3 & H4 & H5 & H6).
  split; [ exact H1 | ]. split; [ | repeat split; assumption ].
  apply grows_placed; [ | exact H2 ].
  intros m Hin. pose proof (forallb_ltb_in _ _ m H4 Hin).
  apply Pos.leb_le in H5. lia.
Qed.

Definition caller_bounded (a : caller) : Prop :=
  forallb (fun m => Pos.ltb m (caller_next a)) (caller_marks a) = true.

Lemma grows_caller_holds (a : caller) (l : lifetime) (v : value) :
  caller_bounded a ->
  (caller_next a <= rs_next rs)%positive ->
  caller_holds rs a l v -> caller_holds rs' a l v.
Proof.
  intros Hbounded Hnext [Husable Hholds]. split; [ exact Husable | ].
  apply grows_lifetime; [ | exact Hholds ].
  intros m Hin. pose proof (forallb_ltb_in _ _ m Hbounded Hin). lia.
Qed.

Lemma grows_conts :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     (u_limit u <= rs_next rs)%positive ->
     cont_ok cert codes rs' u ce c)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c ->
       caller_bounded a ->
       (caller_next a <= rs_next rs)%positive ->
       ret_ok cert codes rs' a l c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c ->
       caller_bounded a ->
       (caller_next a <= rs_next rs)%positive ->
       raise_ok cert codes rs' a c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              (u_limit u <= rs_next rs)%positive ->
              cont_ok cert codes rs' u ce c)
           (fun a l c _ =>
              caller_bounded a ->
              (caller_next a <= rs_next rs)%positive ->
              ret_ok cert codes rs' a l c)
           (fun a c _ =>
              caller_bounded a ->
              (caller_next a <= rs_next rs)%positive ->
              raise_ok cert codes rs' a c)).
  - intros u params metadata handler entry vars conts recursive cs base ms
      next Hmade Hfits Hcheck Hincl Hconts IH Hnone Hlimit.
    apply (Ok_handler cert codes rs' u params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    + exact (grows_made_at u vars recursive cs base ms next Hlimit Hmade).
    + intros k ce Hk. apply (IH k ce Hk). cbn [u_limit].
      destruct Hmade as (_ & _ & _ & _ & _ & _ & _ & H8 & _).
      apply Pos.leb_le in H8. lia.
  - intros u c _. apply Ok_unreachable.
  - intros u local a c Hcalled Hret IH Hlimit.
    apply (Ok_return cert codes rs' u local a c);
      [ exact (grows_called_from u a Hlimit Hcalled) | ].
    destruct Hcalled as (_ & _ & _ & H4 & H5 & _).
    apply IH; [ exact H4 | ]. apply Pos.leb_le in H5. lia.
  - intros u a c Hcalled Hraise IH Hlimit.
    apply (Ok_raise cert codes rs' u a c);
      [ exact (grows_called_from u a Hlimit Hcalled) | ].
    destruct Hcalled as (_ & _ & _ & H4 & H5 & _).
    apply IH; [ exact H4 | ]. apply Pos.leb_le in H5. lia.
  - intros a l k c Hjump Hsome IH Houter Hbounded Hnext.
    apply (Ret_to cert codes rs' a l k c Hjump); [ | exact Houter ].
    intros ce Hk. exact (IH ce Hk Hnext).
  - intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise IHraise Hbounded Hnext.
    apply (Ret_over cert codes rs' a l l' ls remaining remaining_arity
             return_arity return_to exn_to);
      [ | exact Hunknown | exact (IHret Hbounded Hnext)
        | exact (IHraise Hbounded Hnext) ].
    revert Hholds. apply Forall2_impl. intros l0 v.
    exact (grows_caller_holds a l0 v Hbounded Hnext).
  - intros a l l' c Hlasts Hret IH Hbounded Hnext.
    exact (Ret_longer cert codes rs' a l l' c Hlasts (IH Hbounded Hnext)).
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hbounded
      Hnext.
    apply (Raise_to cert codes rs' a k simples extra c);
      [ exact Heval | | exact Hjump | | exact Houter ].
    + revert Hholds. apply Forall2_impl. intros s v.
      exact (grows_caller_holds a _ v Hbounded Hnext).
    + intros ce Hk. exact (IH ce Hk Hnext).
Qed.

End Grows.

(* ======================================================================== *)
(* A region is ended                                                        *)
(* ======================================================================== *)

(* A symbol that a frame binds and that is below the mark of [f] is
   bound by a frame below [f]. *)
Lemma end_keeps_local (above0 : list frame) (above : symbol) (f : frame)
    (below : list frame) (s : symbol) :
  frames_ordered_below above (above0 ++ f :: below) = true ->
  local_in (above0 ++ f :: below) s = true ->
  (s < frame_mark f)%positive ->
  local_in below s = true.
Proof.
  intros Hord Hlocal Hs. unfold local_in in *.
  replace (above0 ++ f :: below) with ((above0 ++ [f]) ++ below) in Hlocal
    by (rewrite <- app_assoc; reflexivity).
  rewrite live_locals_app, lookup_app in Hlocal.
  destruct (lookup s (live_locals (above0 ++ [f]))) as [allocated | ] eqn:Hl;
    [ | exact Hlocal ].
  pose proof (frames_ordered_cut _ _ _ _ _ _ Hord (lookup_in _ _ _ Hl)). lia.
Qed.

(* What is below a frame is in order below its mark, and the frames
   above it have greater marks. *)
Lemma frames_ordered_tail (above : list frame) :
  ∀ (top : symbol) (f : frame) (below : list frame),
    frames_ordered_below top (above ++ f :: below) = true ->
    frames_ordered_below (frame_mark f) below = true.
Proof.
  induction above as [ | g above IH]; intros top f below;
    cbn [app frames_ordered_below]; intros H;
    apply andb_true_iff in H as [_ H]; [ exact H | exact (IH _ _ _ H) ].
Qed.

Lemma frames_above_marks (above : list frame) :
  ∀ (top : symbol) (f : frame) (below : list frame) (g : frame),
    frames_ordered_below top (above ++ f :: below) = true ->
    In g above ->
    (frame_mark f < frame_mark g)%positive.
Proof.
  induction above as [ | g0 above IH]; intros top f below g;
    [ contradiction | ].
  cbn [app frames_ordered_below]. intros H [<- | Hin].
  - apply andb_true_iff in H as [_ H].
    assert (Hf : In f (above ++ f :: below))
      by (apply in_app_iff; right; left; reflexivity).
    exact (frames_ordered_below_marks _ _ f H Hf).
  - apply andb_true_iff in H as [_ H]. exact (IH _ _ _ _ H Hin).
Qed.

(* The frames down to that of the mark [m0] are popped; nothing else
   changes that a value is asked about. *)
Record store_cut (rs rs' : region_state) (m0 : mark) : Prop := Store_cut {
  cut_frames :
    ∃ (above : list frame) (f : frame),
      rs_frames rs = above ++ f :: rs_frames rs' ∧ frame_mark f = m0;
  cut_heap : ∀ s : symbol, heap_symbol rs' s = heap_symbol rs s
}.

Section Cut.

Variables rs rs' : region_state.
Variable m0 : mark.
Hypothesis Hcut : store_cut rs rs' m0.
Hypothesis Hordered : frames_ordered rs = true.

Lemma cut_heap_value (v : value) :
  value_all (heap_symbol rs) v = true -> value_all (heap_symbol rs') v = true.
Proof.
  apply value_all_mono. intros s H. rewrite (cut_heap rs rs' m0 Hcut). exact H.
Qed.

(* A symbol that a frame binds and that was drawn before the mark is
   bound by a frame that stays. *)
Lemma cut_local (s : symbol) :
  local_in (rs_frames rs) s = true ->
  (s < m0)%positive ->
  local_in (rs_frames rs') s = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in *.
  exact (end_keeps_local above _ f _ s Hordered).
Qed.

Lemma cut_within (b : symbol) (v : value) :
  (b <= m0)%positive ->
  value_within rs (Some b) v = true ->
  value_within rs' (Some b) v = true.
Proof.
  intros Hb. apply value_all_mono. intros s. unfold symbol_within.
  rewrite (cut_heap rs rs' m0 Hcut). intros H.
  apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
  apply andb_true_iff in H as [Hlocal Hbefore]. cbn [before] in *.
  rewrite Hbefore, andb_true_r.
  apply Pos.ltb_lt in Hbefore.
  rewrite (cut_local s Hlocal); [ apply orb_true_r | lia ].
Qed.

(* A mark that is on the stack after was on it before; one that was on
   it and was drawn before the mark stays. *)
Lemma cut_stack_back (m : mark) :
  on_stack m (rs_frames rs') = true -> on_stack m (rs_frames rs) = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & -> & _).
  intros H. rewrite on_stack_app. apply orb_true_iff. right.
  unfold on_stack in *. cbn [existsb]. rewrite H. apply orb_true_r.
Qed.

Lemma cut_stack_below (m : mark) :
  on_stack m (rs_frames rs') = true -> (m < m0)%positive.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in Hordered.
  intros Hon. destruct (on_stack_frame m _ Hon) as (g & Hin & <-).
  exact
    (frames_ordered_below_marks _ _ g
       (frames_ordered_tail above _ f _ Hordered) Hin).
Qed.

Lemma cut_stack_keeps (m : mark) :
  on_stack m (rs_frames rs) = true ->
  (m < m0)%positive ->
  on_stack m (rs_frames rs') = true.
Proof.
  destruct (cut_frames rs rs' m0 Hcut) as (above & f & Hframes & <-).
  unfold frames_ordered in Hordered. rewrite Hframes in *.
  rewrite on_stack_app. intros Hon Hlt.
  apply orb_true_iff in Hon as [Hon | Hon].
  - exfalso. destruct (on_stack_frame m _ Hon) as (g & Hin & <-).
    pose proof (frames_above_marks above _ f _ g Hordered Hin). lia.
  - unfold on_stack in *. cbn [existsb] in Hon.
    apply orb_true_iff in Hon as [Hon | Hon]; [ | exact Hon ].
    apply Pos.eqb_eq in Hon. lia.
Qed.

(* A lifetime of a state that was made or of a caller, whose bounds are
   at most the mark. *)
Lemma cut_lifetime (opens : list variable) (ms : list mark) (next : symbol)
    (l : lifetime) (v : value) :
  (∀ m : mark, In m ms -> (m < next)%positive) ->
  marks_decrease ms = true ->
  (In m0 ms ∨ (next <= m0)%positive) ->
  lifetime_holds rs opens ms (Some next) l v ->
  lifetime_holds rs' opens ms (Some next) l v.
Proof.
  intros Hms Hdecrease Hm0. destruct l as [ | | r | ]; cbn [lifetime_holds].
  - exact (cut_heap_value v).
  - destruct (caller_bound_least ms next Hms Hdecrease) as (b & -> & Hb & Hle).
    apply cut_within. destruct Hm0 as [Hin | Hnext];
      [ exact (Hle m0 Hin) | lia ].
  - destruct (region_place opens ms (Some next) r) as [[m stop] | ] eqn:Hplace;
      [ | exact (fun H => H) ].
    destruct (region_place_bound opens ms next r m stop Hplace Hms Hdecrease)
      as (b & -> & Hb & Hin & Hup).
    intros H Hon. apply cut_within; [ | exact (H (cut_stack_back m Hon)) ].
    destruct Hm0 as [Hin0 | Hnext]; [ | lia ].
    exact (Hup m0 Hin0 (cut_stack_below m Hon)).
  - exact (fun H => H).
Qed.

Variable cert : certificate.
Variable codes : list (code_id * code).

Lemma cut_made_at (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  (In m0 ms ∨ (next <= m0)%positive) ->
  made_at cert codes rs u vars recursive cs base ms next ->
  made_at cert codes rs' u vars recursive cs base ms next.
Proof.
  intros Hm0 (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11).
  repeat (split; [ assumption | ]). split; [ | split; assumption ].
  intros x l v Hl Hv.
  exact
    (cut_lifetime (cs_open cs) ms next l v
       (fun m => forallb_ltb_in next ms m H7)
       H6 Hm0 (H9 x l v Hl Hv)).
Qed.

(* The mark is one of the marks of the closure, or was drawn after the
   closure was made. *)
Lemma made_at_mark (u : user) (vars : env)
    (recursive : list (continuation * continuation_handler))
    (cs : check_state) (base : list (continuation * cont_entry))
    (ms : list mark) (next : symbol) :
  made_at cert codes rs u vars recursive cs base ms next ->
  (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
  In m0 ms ∨ (next <= m0)%positive.
Proof.
  intros (_ & _ & _ & _ & _ & _ & _ & H8 & _ & H10 & _) Hm0.
  apply Pos.leb_le in H8.
  destruct Hm0 as [Hin | Hlimit]; [ | right; lia ].
  destruct (Pos.ltb m0 next) eqn:Hlt;
    [ left; exact (H10 m0 Hin Hlt) | right; apply Pos.ltb_ge; exact Hlt ].
Qed.

Lemma cut_called_from (u : user) (a : caller) :
  (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
  called_from rs u a ->
  called_from rs' u a ∧ (caller_next a <= m0)%positive.
Proof.
  intros Hm0 (H1 & H2 & H3 & H4 & H5 & H6).
  assert (Hnext : (caller_next a <= m0)%positive).
  { apply Pos.leb_le in H5. destruct Hm0 as [Hin | Hlimit]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m0 H6 Hin). }
  split; [ | exact Hnext ].
  split; [ exact H1 | ]. split; [ | repeat split; assumption ].
  unfold marks_placed in *. apply andb_true_iff in H2 as [Hon Hdecrease].
  apply andb_true_iff. split; [ | exact Hdecrease ].
  apply forallb_forall. intros m Hin.
  apply cut_stack_keeps; [ exact (forallb_in _ _ m Hon Hin) | ].
  pose proof (forallb_ltb_in _ _ m H4 Hin). lia.
Qed.

(* The marks of a caller decrease: they are placed. *)
Definition caller_decreasing (a : caller) : Prop :=
  marks_decrease (caller_marks a) = true.

Lemma cut_caller_holds (a : caller) (l : lifetime) (v : value) :
  caller_bounded a ->
  caller_decreasing a ->
  (caller_next a <= m0)%positive ->
  caller_holds rs a l v -> caller_holds rs' a l v.
Proof.
  intros Hbounded Hdecrease Hnext [Husable Hholds]. split; [ exact Husable | ].
  exact
    (cut_lifetime _ _ _ l v (fun m => forallb_ltb_in _ _ m Hbounded) Hdecrease
       (or_intror Hnext) Hholds).
Qed.

Lemma cut_conts :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
     cont_ok cert codes rs' u ce c)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c ->
       caller_bounded a ->
       caller_decreasing a ->
       (caller_next a <= m0)%positive ->
       ret_ok cert codes rs' a l c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c ->
       caller_bounded a ->
       caller_decreasing a ->
       (caller_next a <= m0)%positive ->
       raise_ok cert codes rs' a c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              (In m0 (u_marks u) ∨ (u_limit u <= m0)%positive) ->
              cont_ok cert codes rs' u ce c)
           (fun a l c _ =>
              caller_bounded a ->
              caller_decreasing a ->
              (caller_next a <= m0)%positive ->
              ret_ok cert codes rs' a l c)
           (fun a c _ =>
              caller_bounded a ->
              caller_decreasing a ->
              (caller_next a <= m0)%positive ->
              raise_ok cert codes rs' a c)).
  - intros u params metadata handler entry vars conts recursive cs base ms
      next Hmade Hfits Hcheck Hincl Hconts IH Hnone Hm0.
    pose proof (made_at_mark u vars recursive cs base ms next Hmade Hm0)
      as Hm0'.
    apply (Ok_handler cert codes rs' u params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    + exact (cut_made_at u vars recursive cs base ms next Hm0' Hmade).
    + intros k ce Hk. exact (IH k ce Hk Hm0').
  - intros u c _. apply Ok_unreachable.
  - intros u local a c Hcalled Hret IH Hm0.
    destruct (cut_called_from u a Hm0 Hcalled) as [Hcalled' Hnext].
    apply (Ok_return cert codes rs' u local a c Hcalled').
    destruct Hcalled as (_ & H2 & _ & H4 & _).
    apply IH; [ exact H4 | | exact Hnext ].
    unfold marks_placed in H2. apply andb_true_iff in H2 as [_ H2]. exact H2.
  - intros u a c Hcalled Hraise IH Hm0.
    destruct (cut_called_from u a Hm0 Hcalled) as [Hcalled' Hnext].
    apply (Ok_raise cert codes rs' u a c Hcalled').
    destruct Hcalled as (_ & H2 & _ & H4 & _).
    apply IH; [ exact H4 | | exact Hnext ].
    unfold marks_placed in H2. apply andb_true_iff in H2 as [_ H2]. exact H2.
  - intros a l k c Hjump Hsome IH Houter Hbounded Hdecrease Hnext.
    apply (Ret_to cert codes rs' a l k c Hjump); [ | exact Houter ].
    intros ce Hk. exact (IH ce Hk (or_intror Hnext)).
  - intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise IHraise Hbounded Hdecrease Hnext.
    apply (Ret_over cert codes rs' a l l' ls remaining remaining_arity
             return_arity return_to exn_to);
      [ | exact Hunknown | exact (IHret Hbounded Hdecrease Hnext)
        | exact (IHraise Hbounded Hdecrease Hnext) ].
    revert Hholds. apply Forall2_impl. intros l0 v.
    exact (cut_caller_holds a l0 v Hbounded Hdecrease Hnext).
  - intros a l l' c Hlasts Hret IH Hbounded Hdecrease Hnext.
    exact
      (Ret_longer cert codes rs' a l l' c Hlasts (IH Hbounded Hdecrease Hnext)).
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hbounded
      Hdecrease Hnext.
    apply (Raise_to cert codes rs' a k simples extra c);
      [ exact Heval | | exact Hjump | | exact Houter ].
    + revert Hholds. apply Forall2_impl. intros s v.
      exact (cut_caller_holds a _ v Hbounded Hdecrease Hnext).
    + intros ce Hk. exact (IH ce Hk (or_intror Hnext)).
Qed.

End Cut.

(* ======================================================================== *)
(* The user changes                                                         *)
(* ======================================================================== *)

(* A later user: the variables of regions that the first has are bound
   as they were; each of its marks is one of the first's or was drawn
   at or after the first's counter; its counter is no earlier. *)
Definition user_extends (u u' : user) : Prop :=
  (∀ r : variable,
     among r (u_regions u) = true ->
     lookup r (u_vars u') = lookup r (u_vars u)
     ∧ among r (u_regions u') = true)
  ∧ (∀ m : mark,
       In m (u_marks u') -> In m (u_marks u) ∨ (u_limit u <= m)%positive)
  ∧ (u_limit u <= u_limit u')%positive.

Lemma user_conts (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) (u u' : user) (ce : cont_entry) (c : cont_value) :
  user_extends u u' ->
  cont_ok cert codes rs u ce c ->
  cont_ok cert codes rs u' ce c.
Proof.
  intros (Hregions & Hmarks & Hlimit) Hok.
  destruct Hok
    as [u params metadata handler entry vars conts recursive cs base ms next
          Hmade Hfits Hcheck Hincl Hconts Hnone
       | u c | u local a c Hcalled Hret | u a c Hcalled Hraise].
  - apply (Ok_handler cert codes rs u' params metadata handler entry vars conts
             recursive cs base ms next);
      try assumption.
    destruct Hmade as (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10 & H11).
    apply Pos.leb_le in H8.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ]. split; [ exact H9 | ]. split.
    + intros m Hin Hlt. apply Pos.ltb_lt in Hlt.
      destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
      apply (H10 m Hin'). apply Pos.ltb_lt. exact Hlt.
    + intros r Hr. destruct (H11 r Hr) as [Hlookup Hamong].
      destruct (Hregions r Hamong) as [Hlookup' Hamong'].
      split; [ rewrite Hlookup'; exact Hlookup | exact Hamong' ].
  - apply Ok_unreachable.
  - apply (Ok_return cert codes rs u' local a c); [ | exact Hret ].
    destruct Hcalled as (H1 & H2 & H3 & H4 & H5 & H6).
    apply Pos.leb_le in H5.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ].
    apply forallb_forall. intros m Hin. apply Pos.leb_le.
    destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m H6 Hin').
  - apply (Ok_raise cert codes rs u' a c); [ | exact Hraise ].
    destruct Hcalled as (H1 & H2 & H3 & H4 & H5 & H6).
    apply Pos.leb_le in H5.
    repeat (split; [ assumption | ]).
    split; [ apply Pos.leb_le; lia | ].
    apply forallb_forall. intros m Hin. apply Pos.leb_le.
    destruct (Hmarks m Hin) as [Hin' | Hle]; [ | lia ].
    apply Pos.leb_le. exact (forallb_in _ _ m H6 Hin').
Qed.
