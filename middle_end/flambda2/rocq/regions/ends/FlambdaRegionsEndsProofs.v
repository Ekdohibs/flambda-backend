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

(* Proofs of the eight statements of FlambdaRegionsEndsSpec.v. A step
   is read through FlambdaRegionsEndsViewProofs.region_step_viewed,
   and each of its cases keeps the property by a lemma of
   FlambdaRegionsEndsLetProofs.v or of FlambdaRegionsEndsCallProofs.v
   ([view_keeps]). The statements of a run follow by induction on the
   fuel, and the last from
   FlambdaRegionsCheckSoundProofs.validated_and_checked_exits_agree.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaImportsSpec
  FlambdaCompareSpec FlambdaUnusedStaticsSpec FlambdaLoopifySpec
  FlambdaLiftSpec FlambdaValidateSpec FlambdaValidateImportsSpec
  FlambdaRegionsSpec FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec
  FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec
  FlambdaRegionsCheckSoundSpec FlambdaRegionsCheckSoundProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs FlambdaRegionsEndsStableProofs
  FlambdaRegionsEndsViewProofs FlambdaRegionsEndsHandlersProofs
  FlambdaRegionsEndsLetProofs FlambdaRegionsEndsCallProofs.
Import ListNotations.

(* ======================================================================== *)
(* A step                                                                   *)
(* ======================================================================== *)

(* What a step leaves: a state that has the property, with the frames
   below as they were; or, at an exit through the return continuation,
   no plain frame drawn since what is run was entered. *)
Lemma view_keeps (cert : certificate) (opts : ends_options)
    (ret : continuation) (bottom : symbol) (rs : region_state)
    (result : region_step_result) :
  step_view rs result ->
  ∀ (es : ends_state) (ms : list mark) (floor : symbol),
    frames_ordered rs = true ->
    code_ends cert opts rs ->
    ends_at cert opts ret bottom rs es ms floor ->
    match result with
    | Region_step rs' =>
      frames_ordered rs' = true ->
      state_ends cert opts ret bottom rs'
      ∧ frames_before bottom (rs_frames rs')
        = frames_before bottom (rs_frames rs)
    | Region_exit k vs rs' =>
      frames_before bottom (rs_frames rs') = frames_before bottom (rs_frames rs)
      ∧ (k = ret ->
         ∀ m : mark,
           In m (plain_marks (rs_frames rs')) -> Pos.ltb m bottom = true)
    | Region_undefined _ => True
    end.
Proof.
  intros Hview es ms floor Hordered Hcode Hat.
  pose proof Hat as (_ & Hbottom & Hnext & _).
  apply Pos.leb_le in Hbottom. apply Pos.leb_le in Hnext.
  destruct Hview
    as [fault
       | x kind p mark_kind body rs' He Hp Hexpr' Hvars' Hconts' Hstatics'
           Hframes' Hnext' Hcuts
       | x kind p region body m above f below rs' He Hp Heval Hframes Hf
           Hexpr' Hvars' Hconts' Hstatics' Hframes' Hnext' Hcuts
       | x kind d y kind' args body' v rs' He Hd Hexpr' Hvars' Hconts'
           Hstatics' Hframes' Hnext' Hcuts
       | bindings body statics' rs' He Heval Hexpr' Hvars' Hconts' Hstatics'
           Hframes' Hnext' Hcuts
       | ac params metadata handler def_vars def_conts recursive vs
           handler_vars rs' He Hresolve Heval Hbind Hexpr' Hvars' Hconts'
           Hstatics' Hframes' Hnext' Hcuts
       | live s' rs' Hnot_region Hnot_static Hlive Hstep Hstate Hframes'
           Hnext' Hcuts
       | live k vs hp rd rs' Hlive Hstep Hframes'].
  - exact I.
  - intros Hordered'.
    destruct
      (begin_keeps cert opts ret bottom rs rs' es ms floor x kind p mark_kind
         body Hordered Hat He Hp Hexpr' Hvars' Hconts' Hframes' Hnext')
      as (es' & ms' & _ & Hat').
    split.
    + exists es'.
      apply (state_ends_at_intro cert opts ret bottom rs' es' ms' floor
               Hordered');
        [ | exact Hat' ].
      apply (code_ends_same cert opts rs rs'); [ | exact Hcode ].
      rewrite Hstatics'. reflexivity.
    + rewrite Hframes'. apply frames_before_push. cbn [frame_mark]. lia.
  - intros Hordered'.
    destruct
      (end_keeps cert opts ret bottom rs rs' es ms floor x kind p region body
         m above f below Hordered Hat He Hp Heval Hframes Hf Hexpr' Hvars'
         Hconts' Hframes' Hnext')
      as (Hfloor_m & _ & es' & ms' & _ & Hat').
    split.
    + exists es'.
      apply (state_ends_at_intro cert opts ret bottom rs' es' ms' floor
               Hordered');
        [ | exact Hat' ].
      apply (code_ends_same cert opts rs rs'); [ | exact Hcode ].
      rewrite Hstatics'. reflexivity.
    + rewrite Hframes', Hframes. symmetry.
      apply (frames_before_cut above (rs_next rs) f below bottom).
      * unfold frames_ordered in Hordered. rewrite Hframes in Hordered.
        exact Hordered.
      * rewrite Hf. lia.
  - intros Hordered'. split.
    + exists es.
      apply (state_ends_at_intro cert opts ret bottom rs' es ms floor
               Hordered');
        [ exact (code_ends_same cert opts rs rs' Hstatics' Hcode) | ].
      exact
        (local_keeps cert opts ret bottom rs rs' es ms floor x kind d y kind'
           args body' v Hat He Hd Hexpr' Hvars' Hconts' Hframes' Hnext').
    + exact (frames_before_same bottom _ _ Hframes').
  - intros Hordered'.
    destruct
      (static_keeps cert opts ret bottom rs rs' es ms floor bindings body Hat
         He Hexpr' Hvars' Hconts' Hframes' Hnext')
      as [Hbindings Hat'].
    split; [ | rewrite Hframes'; reflexivity ].
    exists es.
    apply (state_ends_at_intro cert opts ret bottom rs' es ms floor Hordered');
      [ | exact Hat' ].
    exact
      (static_code_ends cert opts rs rs' bindings statics' Hcode Hbindings
         Heval Hstatics').
  - intros Hordered'. split; [ | rewrite Hframes'; reflexivity ].
    pose proof
      (apply_cont_lands cert opts ret bottom rs Hordered Hcode es ms floor ac
         vs (state_statics (rs_state rs)) (state_heap (rs_state rs))
         (state_reads (rs_state rs)) Hat He Heval eq_refl)
      as Hlands.
    rewrite Hresolve in Hlands. cbn [apply_unkinded] in Hlands.
    rewrite Hbind in Hlands. cbn [lands] in Hlands.
    apply (Hlands rs'); cbn [state_expr state_vars state_conts];
      [ exact Hexpr' | exact Hvars' | exact Hconts'
      | rewrite Hstatics'; reflexivity | exact Hframes' | exact Hnext' ].
  - intros Hordered'. split; [ | rewrite Hframes'; reflexivity ].
    assert (Hstatics' :
              static_code (state_statics (rs_state rs'))
              = static_code (state_statics (rs_state rs))).
    { rewrite Hstate. reflexivity. }
    assert (Hforms :
              ((∃ ap : apply_expr, state_expr (rs_state rs) = Apply ap)
               ∨ (∃ ac : apply_cont, state_expr (rs_state rs) = Apply_cont ac))
              ∨ ((∀ ap : apply_expr, state_expr (rs_state rs) <> Apply ap)
                 ∧ (∀ ac : apply_cont,
                      state_expr (rs_state rs) <> Apply_cont ac))).
    { destruct (state_expr (rs_state rs))
        as [x kind d body | bindings body | k h body | handlers body | ap
           | ac | scrutinee arms | ];
        try (right; split; intros; discriminate).
      - left. left. exists ap. reflexivity.
      - left. right. exists ac. reflexivity. }
    destruct Hforms as [Hform | [Hnot_apply Hnot_jump]].
    + pose proof
        (step_lands cert opts ret bottom rs Hordered Hcode es ms floor live
           Hat Hlive Hform)
        as Hlands.
      rewrite Hstep in Hlands. cbn [lands] in Hlands.
      apply (Hlands rs'); try (rewrite Hstate; reflexivity);
        [ exact Hframes' | exact Hnext' ].
    + destruct
        (plain_keeps cert opts ret bottom rs rs' es ms floor live s' Hat
           Hnot_region Hnot_static Hnot_apply Hnot_jump Hstep Hstate Hframes'
           Hnext')
        as (es' & _ & Hat').
      exists es'.
      exact
        (state_ends_at_intro cert opts ret bottom rs' es' ms floor Hordered'
           (code_ends_same cert opts rs rs' Hstatics' Hcode) Hat').
  - split; [ rewrite Hframes'; reflexivity | ].
    intros Hk. rewrite Hframes'.
    pose proof
      (step_lands cert opts ret bottom rs Hordered Hcode es ms floor live Hat
         Hlive (step_exit_forms _ k vs hp rd Hstep))
      as Hlands.
    rewrite Hstep in Hlands. cbn [lands] in Hlands. exact (Hlands Hk).
Qed.

(* With the option, a step from a state that is not stale counts no
   plain frame cut by the end of another region. *)
Lemma view_own_end (cert : certificate) (opts : ends_options)
    (ret : continuation) (bottom : symbol) (rs : region_state)
    (result : region_step_result) :
  step_view rs result ->
  ∀ (es : ends_state) (ms : list mark) (floor : symbol),
    eo_own_end opts = true ->
    frames_ordered rs = true ->
    ends_at cert opts ret bottom rs es ms floor ->
    es_stale es = false ->
    match result with
    | Region_step rs' => cuts_kept rs rs'
    | Region_exit _ _ _ | Region_undefined _ => True
    end.
Proof.
  intros Hview es ms floor Hopt Hordered Hat Hstale.
  destruct Hview
    as [fault
       | x kind p mark_kind body rs' He Hp Hexpr' Hvars' Hconts' Hstatics'
           Hframes' Hnext' Hcuts
       | x kind p region body m above f below rs' He Hp Heval Hframes Hf
           Hexpr' Hvars' Hconts' Hstatics' Hframes' Hnext' Hcuts
       | x kind d y kind' args body' v rs' He Hd Hexpr' Hvars' Hconts'
           Hstatics' Hframes' Hnext' Hcuts
       | bindings body statics' rs' He Heval Hexpr' Hvars' Hconts' Hstatics'
           Hframes' Hnext' Hcuts
       | ac params metadata handler def_vars def_conts recursive vs
           handler_vars rs' He Hresolve Heval Hbind Hexpr' Hvars' Hconts'
           Hstatics' Hframes' Hnext' Hcuts
       | live s' rs' Hnot_region Hnot_static Hlive Hstep Hstate Hframes'
           Hnext' Hcuts
       | live k vs hp rd rs' Hlive Hstep Hframes'];
    try exact I; try exact Hcuts.
  apply Hcuts.
  exact
    (proj1
       (proj2
          (end_keeps cert opts ret bottom rs rs' es ms floor x kind p region
             body m above f below Hordered Hat He Hp Heval Hframes Hf Hexpr'
             Hvars' Hconts' Hframes' Hnext'))
       Hstale Hopt).
Qed.

(* ======================================================================== *)
(* The state that is entered                                                *)
(* ======================================================================== *)

(* A state that has no continuation, whose expression passes as the
   body of code does with [ret] and [exn] its two continuations, has
   the property with nothing above the counter. *)
Lemma entry_ends (cert : certificate) (opts : ends_options)
    (ret exn : continuation) (rs : region_state) :
  frames_ordered rs = true ->
  code_ends cert opts rs ->
  Pos.eqb ret exn = false ->
  ends_expr cert opts (ends_code_state ret exn) (state_expr (rs_state rs))
  = true ->
  state_conts (rs_state rs) = [] ->
  state_ends cert opts ret (rs_next rs) rs.
Proof.
  intros Hordered Hcode Hdistinct Hexpr Hconts.
  exists (ends_code_state ret exn).
  split; [ exact Hordered | ]. split; [ exact Hcode | ].
  exists [], (rs_next rs).
  split; [ exact Hexpr | ]. split; [ apply Pos.leb_refl | ].
  split; [ apply Pos.leb_refl | ]. split.
  - split.
    + unfold ends_placed. cbn [ends_code_state es_open es_tries es_regions].
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. split; [ reflexivity | ].
      split; [ reflexivity | ]. intros r m [].
    + cbn [ends_code_state es_stale stale_stop]. intros m Hplain Hfloor _.
      exfalso. apply plain_marks_in in Hplain.
      destruct Hplain as (g & Hg & _ & <-).
      pose proof
        (frames_ordered_below_marks (rs_frames rs) (rs_next rs) g Hordered Hg).
      apply Pos.leb_le in Hfloor. lia.
  - cbn [ends_code_state es_conts es_regions es_tries]. rewrite Hconts.
    split.
    + intros k ec Hlookup. unfold resolve. cbn [lookup] in *.
      destruct (Pos.eqb k ret) eqn:Hk.
      * injection Hlookup as <-. apply Pos.eqb_eq in Hk. subst k.
        apply (Ends_ok_exit cert opts ret (rs_next rs) rs). reflexivity.
      * destruct (Pos.eqb k exn) eqn:Hk'; [ | discriminate ].
        injection Hlookup as <-.
        exact (Ends_ok_raise_out cert opts ret (rs_next rs) rs _ k [] Hk).
    + intros k Hlookup. reflexivity.
Qed.

Theorem ends_initial : ends_initial_statement.
Proof.
  intros cert opts ret exn statics base e Hcheck.
  unfold ends_check in Hcheck. rewrite !andb_true_iff in Hcheck.
  destruct Hcheck as [[Hcodes Hdistinct] Hexpr].
  apply negb_true_iff in Hdistinct.
  apply (entry_ends cert opts ret exn (initial_region_state e statics base));
    [ apply frames_ordered_initial | | exact Hdistinct | exact Hexpr
    | reflexivity ].
  intros id c Hlookup.
  exact (proj1 (forallb_forall _ _) Hcodes (id, c) (lookup_in _ _ _ Hlookup)).
Qed.

Theorem ends_code_entry : ends_code_entry_statement.
Proof.
  intros cert opts c rs Hc Hordered Hcode Hexpr Hconts.
  destruct (ends_code_parts cert opts c Hc) as [Hdistinct Hbody].
  apply
    (entry_ends cert opts (code_return_continuation c)
       (code_exn_continuation c) rs Hordered Hcode Hdistinct);
    [ rewrite Hexpr; exact Hbody | exact Hconts ].
Qed.

(* ======================================================================== *)
(* A step, and an exit                                                      *)
(* ======================================================================== *)

Theorem region_step_keeps_ends : region_step_keeps_ends_statement.
Proof.
  intros cert opts ret bottom rs rs' (es & Hordered & Hcode & ms & floor & Hat)
    Hstep.
  exact
    (view_keeps cert opts ret bottom rs (Region_step rs')
       (region_step_viewed rs _ Hstep) es ms floor Hordered Hcode Hat
       (region_step_keeps_frames_ordered rs rs' Hstep Hordered)).
Qed.

Theorem ends_exit : ends_exit_statement.
Proof.
  intros cert opts ret bottom rs rs' vs
    (es & Hordered & Hcode & ms & floor & Hat) Hstep.
  destruct
    (view_keeps cert opts ret bottom rs (Region_exit ret vs rs')
       (region_step_viewed rs _ Hstep) es ms floor Hordered Hcode Hat)
    as [Hbefore Hplain].
  split; [ exact Hbefore | exact (Hplain eq_refl) ].
Qed.

Theorem ends_own_end : ends_own_end_statement.
Proof.
  intros cert opts ret bottom rs rs' es Hopt
    (Hordered & Hcode & ms & floor & Hat) Hstale Hstep.
  exact
    (view_own_end cert opts ret bottom rs (Region_step rs')
       (region_step_viewed rs _ Hstep) es ms floor Hopt Hordered Hat Hstale).
Qed.

(* ======================================================================== *)
(* A run                                                                    *)
(* ======================================================================== *)

(* A run from a state that has the property, where it exits through
   [ret]. *)
Lemma run_exit (cert : certificate) (opts : ends_options)
    (ret : continuation) (bottom : symbol) (fuel : nat) :
  ∀ (rs rs' : region_state) (vs : list value),
    state_ends cert opts ret bottom rs ->
    region_run fuel rs = Region_exit ret vs rs' ->
    frames_before bottom (rs_frames rs') = frames_before bottom (rs_frames rs)
    ∧ ∀ m : mark,
        In m (plain_marks (rs_frames rs')) -> Pos.ltb m bottom = true.
Proof.
  induction fuel as [ | fuel IH]; intros rs rs' vs Hends Hrun;
    cbn [region_run] in Hrun; [ discriminate | ].
  destruct (region_step rs) as [rs1 | k ws rs1 | fault] eqn:Hstep;
    try discriminate.
  - destruct (region_step_keeps_ends cert opts ret bottom rs rs1 Hends Hstep)
      as [Hends1 Hbefore1].
    destruct (IH rs1 rs' vs Hends1 Hrun) as [Hbefore Hplain].
    split; [ rewrite Hbefore; exact Hbefore1 | exact Hplain ].
  - injection Hrun as -> -> ->.
    exact (ends_exit cert opts ret bottom rs rs' vs Hends Hstep).
Qed.

Theorem ends_unit_exit : ends_unit_exit_statement.
Proof.
  intros cert opts ret exn statics base e fuel vs rs Hcheck Hrun.
  destruct
    (run_exit cert opts ret base fuel _ rs vs
       (ends_initial cert opts ret exn statics base e Hcheck) Hrun)
    as [Hbefore Hplain].
  change (frames_before base (rs_frames rs) = []) in Hbefore.
  destruct (plain_marks (rs_frames rs)) as [ | m l] eqn:Hmarks;
    [ reflexivity | exfalso ].
  pose proof (Hplain m (or_introl eq_refl)) as Hlt. apply Pos.ltb_lt in Hlt.
  assert (Hin : In m (plain_marks (rs_frames rs))).
  { rewrite Hmarks. left. reflexivity. }
  pose proof (proj1 (plain_before m base (rs_frames rs) Hlt) Hin) as Hbelow.
  rewrite Hbefore in Hbelow. exact Hbelow.
Qed.

(* The plain marks of frames all of which were drawn before a symbol,
   from the marks and the kinds of the frames drawn before it. *)
Lemma plain_marks_before (bottom : symbol) (frames : list frame) :
  (∀ m : mark, In m (plain_marks frames) -> Pos.ltb m bottom = true) ->
  plain_marks frames
  = map fst
      (filter
         (fun p =>
            match snd p with
            | Plain_mark => true
            | Try_mark => false
            end)
         (frames_before bottom frames)).
Proof.
  induction frames as [ | f frames IH]; intros Hlt; [ reflexivity | ].
  assert (Hrest :
            ∀ m : mark,
              In m (plain_marks frames) -> Pos.ltb m bottom = true).
  { intros m Hm. apply Hlt. rewrite plain_marks_cons.
    destruct (is_plain f); [ right; exact Hm | exact Hm ]. }
  rewrite plain_marks_cons, frames_before_cons.
  destruct (is_plain f) eqn:Hplain.
  - assert (Hf : Pos.ltb (frame_mark f) bottom = true).
    { apply Hlt. rewrite plain_marks_cons, Hplain. left. reflexivity. }
    rewrite Hf. unfold is_plain in Hplain.
    destruct (frame_kind f); [ | discriminate ].
    cbn [filter snd map fst]. f_equal. exact (IH Hrest).
  - destruct (Pos.ltb (frame_mark f) bottom); [ | exact (IH Hrest) ].
    unfold is_plain in Hplain. destruct (frame_kind f); [ discriminate | ].
    cbn [filter snd]. exact (IH Hrest).
Qed.

Theorem ends_code_return : ends_code_return_statement.
Proof.
  intros cert opts c rs rs' fuel vs Hc Hordered Hcode Hexpr Hconts Hrun.
  destruct
    (run_exit cert opts (code_return_continuation c) (rs_next rs) fuel rs rs'
       vs (ends_code_entry cert opts c rs Hc Hordered Hcode Hexpr Hconts) Hrun)
    as [Hbefore Hplain].
  rewrite (plain_marks_before (rs_next rs) (rs_frames rs') Hplain), Hbefore.
  symmetry. apply plain_marks_before.
  intros m Hm. apply Pos.ltb_lt. apply plain_marks_in in Hm.
  destruct Hm as (g & Hg & _ & <-).
  exact (frames_ordered_below_marks (rs_frames rs) (rs_next rs) g Hordered Hg).
Qed.

Theorem validated_and_ends : validated_and_ends_statement.
Proof.
  intros imports rounds unused slots G Ls kept cert opts ret exn base before
    after validated_before validated_after statics Hverdict Hbefore Hafter
    Hdefined Hcheck Hfresh Hfresh' Herasable Herasable' fuel vs rs Hrun.
  unfold region_check_ends in Hcheck. apply andb_true_iff in Hcheck.
  destruct Hcheck as [Hregion Hends].
  destruct
    (validated_and_checked_exits_agree imports rounds unused slots G Ls kept
       cert base before after validated_before validated_after Hverdict
       Hbefore Hafter Hdefined Hregion Hfresh Hfresh' Herasable Herasable'
       fuel ret vs rs Hrun)
    as (fuel' & ws & rs' & us & Hrun' & _).
  exists fuel', ws, rs'. split; [ exact Hrun' | ].
  exact
    (ends_unit_exit cert opts ret exn statics base after fuel' ws rs' Hends
       Hrun').
Qed.
