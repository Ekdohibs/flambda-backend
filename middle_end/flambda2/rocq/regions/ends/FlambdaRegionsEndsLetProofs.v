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
   the steps that are not a jump and not a call keep the property. A
   region is begun; a region is ended; a local allocation; another
   [Let]; a [Let_static]; a [Let_cont]; a [Switch]. Each is said of
   what the step leaves (FlambdaRegionsEndsViewProofs.step_view).
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsStepsProofs FlambdaRegionsCallsSpec
  FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedSharedProofs FlambdaRegionsCheckedViewsProofs
  FlambdaRegionsCheckedStoreProofs.
From OxCamlFlambda2 Require Import FlambdaRegionsEndsSpec
  FlambdaRegionsEndsBasicsProofs FlambdaRegionsEndsStableProofs
  FlambdaRegionsEndsViewProofs FlambdaRegionsEndsHandlersProofs.
Import ListNotations.

Section Let_steps.

Variable cert : certificate.
Variable opts : ends_options.
Variable ret : continuation.
Variable bottom : symbol.

(* ======================================================================== *)
(* The parts of the property that are of one state of the check             *)
(* ======================================================================== *)

Definition ends_at (rs : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) : Prop :=
  ends_expr cert opts es (state_expr (rs_state rs)) = true
  ∧ Pos.leb bottom floor = true
  ∧ Pos.leb floor (rs_next rs) = true
  ∧ ends_running rs es ms floor
  ∧ conts_ends cert opts ret bottom rs
      (Ends_user (state_vars (rs_state rs)) (es_regions es) (es_tries es)
         floor)
      (es_conts es) (state_conts (rs_state rs)).

Lemma state_ends_at_intro (rs : region_state) (es : ends_state)
    (ms : list mark) (floor : symbol) :
  frames_ordered rs = true ->
  code_ends cert opts rs ->
  ends_at rs es ms floor ->
  state_ends_at cert opts ret bottom rs es.
Proof.
  intros Hordered Hcode Hat.
  split; [ exact Hordered | ]. split; [ exact Hcode | ].
  exists ms, floor. exact Hat.
Qed.

(* The continuations, for other frames that are the same before the
   floor and for a later user. *)
Lemma conts_ends_moved (rs rs' : region_state) (u u' : ends_user)
    (entries : list (continuation * ends_cont)) (conts : cont_env) :
  frames_before (eu_floor u) (rs_frames rs')
  = frames_before (eu_floor u) (rs_frames rs) ->
  ends_user_extends u u' ->
  conts_ends cert opts ret bottom rs u entries conts ->
  conts_ends cert opts ret bottom rs' u' entries conts.
Proof.
  intros Hkept Hextends Hconts.
  apply (user_conts_ends cert opts ret bottom rs' u u' entries conts Hextends).
  exact (kept_conts_ends cert opts ret bottom rs rs' u entries conts Hkept
           Hconts).
Qed.

(* A user that binds one variable more, which is not the variable of a
   region, and has the same variables of regions. *)
Lemma user_extends_bind (vars : env) (regions tries : list variable)
    (floor : symbol) (x : variable) (v : value) :
  among x regions = false ->
  ends_user_extends (Ends_user vars regions tries floor)
    (Ends_user ((x, v) :: vars) regions tries floor).
Proof.
  intros Hnew. split; [ | split; [ | reflexivity ] ].
  - intros r Hr. cbn [eu_vars eu_regions] in *. split; [ | exact Hr ].
    cbn. destruct (Pos.eqb r x) eqn:Heq; [ | reflexivity ].
    apply Pos.eqb_eq in Heq. subst r. rewrite Hnew in Hr. discriminate.
  - intros r Hr. exact Hr.
Qed.

(* ======================================================================== *)
(* What the check says of a [Let]                                           *)
(* ======================================================================== *)

Lemma ends_expr_begin (es : ends_state) (x : variable) (kind : flambda_kind)
    (p : region_primitive) (mark_kind : mark_kind) (body : expr) :
  begin_mark_kind p = Some mark_kind ->
  ends_expr cert opts es (Let x kind (Prim (Region_primitive p) []) body)
  = true ->
  among x (es_regions es) = false
  ∧ es_stale es = false
  ∧ ends_expr cert opts
      (Ends_state (x :: es_open es)
         (match mark_kind with
          | Plain_mark => es_tries es
          | Try_mark => x :: es_tries es
          end)
         (x :: es_regions es) false (es_conts es))
      body
    = true.
Proof.
  intros Hp H.
  destruct p; try discriminate Hp; injection Hp as <-; cbn in H;
    rewrite !andb_true_iff in H; destruct H as [Hnew [Hstale Hbody]];
    (split; [ apply negb_true_iff; exact Hnew | ]);
    (split; [ apply negb_true_iff; exact Hstale | exact Hbody ]).
Qed.

Lemma ends_expr_end (es : ends_state) (x : variable) (kind : flambda_kind)
    (p : region_primitive) (region : simple) (body : expr) :
  is_end p = true ->
  ends_expr cert opts es (Let x kind (Prim (Region_primitive p) [region]) body)
  = true ->
  ∃ r : variable,
    region = Var r
    ∧ among x (es_regions es) = false
    ∧ among r (es_open es) = true
    ∧ (eo_own_end opts = true ->
       forallb (fun r' => among r' (es_tries es))
         (regions_above r (es_open es))
       = true)
    ∧ ends_expr cert opts
        (Ends_state (regions_below r (es_open es)) (es_tries es)
           (es_regions es) false (es_conts es))
        body
      = true.
Proof.
  intros Hp H.
  destruct p; try discriminate Hp;
    (destruct region as [r | sym | c]; cbn in H;
     [ | rewrite andb_false_r in H; discriminate
       | rewrite andb_false_r in H; discriminate ];
     rewrite !andb_true_iff in H;
     destruct H as [Hnew [[Hopen Hown] Hbody]];
     exists r;
     split; [ reflexivity | ];
     split; [ apply negb_true_iff; exact Hnew | ];
     split; [ exact Hopen | ];
     split; [ | exact Hbody ];
     intros Hopt; rewrite Hopt in Hown; exact Hown).
Qed.

Lemma ends_expr_let (es : ends_state) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  region_named d = false ->
  ends_expr cert opts es (Let x kind d body)
  = negb (among x (es_regions es)) && ends_expr cert opts es body.
Proof.
  destruct d as [s | [u | b | v | hp | rp] args | | | | | ]; intros Hd;
    try reflexivity.
  discriminate.
Qed.

Lemma ends_expr_local (es : ends_state) (y : variable) (kind : flambda_kind)
    (args : list simple) (body : expr) :
  ends_expr cert opts es
    (Let y kind (Prim (Region_primitive Local_allocation) args) body)
  = negb (among y (es_regions es)) && ends_expr cert opts es body.
Proof. destruct args as [ | a [ | b args]]; reflexivity. Qed.

(* ======================================================================== *)
(* A region is begun                                                        *)
(* ======================================================================== *)

(* The marks of the open regions were drawn before the counter. *)
Lemma placed_below_next (rs : region_state) (ms : list mark) (m : mark) :
  frames_ordered rs = true ->
  marks_placed (rs_frames rs) ms = true ->
  In m ms ->
  (m < rs_next rs)%positive.
Proof.
  intros Hordered Hplaced Hin.
  destruct (on_stack_frame m (rs_frames rs)
              (marks_placed_in (rs_frames rs) ms m Hplaced Hin))
    as (g & Hg & <-).
  exact (frames_ordered_below_marks (rs_frames rs) (rs_next rs) g Hordered Hg).
Qed.

Lemma begin_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (x : variable) (kind : flambda_kind)
    (p : region_primitive) (mark_kind : mark_kind) (body : expr) :
  frames_ordered rs = true ->
  ends_at rs es ms floor ->
  state_expr (rs_state rs) = Let x kind (Prim (Region_primitive p) []) body ->
  begin_mark_kind p = Some mark_kind ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs')
  = (x, Value_symbol (rs_next rs)) :: state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  rs_frames rs' = Frame (rs_next rs) mark_kind [] :: rs_frames rs ->
  rs_next rs' = Pos.succ (rs_next rs) ->
  ∃ (es' : ends_state) (ms' : list mark),
    es_stale es' = false ∧ ends_at rs' es' ms' floor.
Proof.
  intros Hordered (Hexpr & Hbottom & Hnext & [Hplaced Htight] & Hconts) He Hp
    Hexpr' Hvars' Hconts' Hframes' Hnext'.
  rewrite He in Hexpr.
  destruct (ends_expr_begin es x kind p mark_kind body Hp Hexpr)
    as (Hnew & Hstale & Hbody).
  pose proof Hplaced as (Hmarks & Hmp & Hfloor & Hopen & Htries & Hkinds).
  unfold FlambdaMapSpec.all_among in Hopen, Htries.
  assert (Hx_open : among x (es_open es) = false).
  { destruct (among x (es_open es)) eqn:Hx; [ | reflexivity ].
    pose proof (proj1 (forallb_forall _ _) Hopen x (proj1 (among_In _ _) Hx))
      as Hregion.
    cbn beta in Hregion. rewrite Hnew in Hregion. discriminate. }
  assert (Hx_tries : among x (es_tries es) = false).
  { destruct (among x (es_tries es)) eqn:Hx; [ | reflexivity ].
    pose proof (proj1 (forallb_forall _ _) Htries x (proj1 (among_In _ _) Hx))
      as Hregion.
    cbn beta in Hregion. rewrite Hnew in Hregion. discriminate. }
  assert (Hlt : ∀ m : mark, In m ms -> (m < rs_next rs)%positive).
  { intros m Hin. exact (placed_below_next rs ms m Hordered Hmp Hin). }
  apply Pos.leb_le in Hnext.
  rewrite Hstale in Htight. cbn [stale_stop] in Htight.
  set (tries' :=
         match mark_kind with
         | Plain_mark => es_tries es
         | Try_mark => x :: es_tries es
         end) in *.
  assert (Htries_x :
            ∀ r : variable,
              among r tries' = true ->
              Pos.eqb r x = false ->
              among r (es_tries es) = true).
  { intros r Hr Hrx. subst tries'. destruct mark_kind; [ exact Hr | ].
    rewrite among_cons, Hrx in Hr. exact Hr. }
  exists
    (Ends_state (x :: es_open es) tries' (x :: es_regions es) false
       (es_conts es)),
    (rs_next rs :: ms).
  split; [ reflexivity | ].
  unfold ends_at, ends_running, ends_placed.
  rewrite Hexpr', Hvars', Hconts', Hnext', Hframes'.
  cbn [es_open es_tries es_regions es_stale es_conts stale_stop].
  split; [ exact Hbody | ]. split; [ exact Hbottom | ].
  split; [ apply Pos.leb_le; lia | ]. split.
  - split.
    + split.
      { cbn [open_marks]. rewrite lookup_head.
        rewrite (open_marks_bind _ _ _ _ Hx_open), Hmarks. reflexivity. }
      split.
      { unfold marks_placed in *. apply andb_true_iff in Hmp.
        destruct Hmp as [Hon Hdecrease]. apply andb_true_iff. split.
        - cbn [forallb]. apply andb_true_iff. split.
          + unfold on_stack. cbn. rewrite Pos.eqb_refl. reflexivity.
          + apply forallb_forall. intros m Hin. unfold on_stack. cbn.
            apply orb_true_iff. right.
            exact (proj1 (forallb_forall _ _) Hon m Hin).
        - apply marks_decrease_cons. split; [ exact Hlt | exact Hdecrease ]. }
      split.
      { cbn [forallb]. apply andb_true_iff. split;
          [ apply Pos.leb_le; lia | exact Hfloor ]. }
      split.
      { unfold FlambdaMapSpec.all_among. cbn [forallb].
        apply andb_true_iff. split.
        - rewrite among_cons, Pos.eqb_refl. reflexivity.
        - apply forallb_forall. intros r Hr. cbn beta.
          rewrite among_cons. apply orb_true_iff. right.
          exact (proj1 (forallb_forall _ _) Hopen r Hr). }
      split.
      { unfold FlambdaMapSpec.all_among. apply forallb_forall. intros r Hr.
        cbn beta. rewrite among_cons.
        destruct (Pos.eqb r x) eqn:Hrx; [ reflexivity | ]. cbn.
        exact
          (proj1 (forallb_forall _ _) Htries r
             (proj1 (among_In _ _)
                (Htries_x r (proj2 (among_In _ _) Hr) Hrx))). }
      intros r m Hin Htry. cbn [combine] in Hin.
      destruct Hin as [Heq | Hin].
      * injection Heq as <- <-.
        unfold kind_on_stack. cbn. rewrite Pos.eqb_refl. cbn.
        subst tries'. destruct mark_kind; [ | reflexivity ].
        rewrite Hx_tries in Htry. discriminate.
      * assert (Hrx : Pos.eqb r x = false).
        { destruct (Pos.eqb r x) eqn:Hrx; [ | reflexivity ].
          apply Pos.eqb_eq in Hrx. subst r.
          rewrite (proj2 (among_In _ _) (in_combine_l _ _ _ _ Hin))
            in Hx_open.
          discriminate. }
        pose proof (Hkinds r m Hin (Htries_x r Htry Hrx)) as Hkind.
        pose proof (Hlt m (in_combine_r _ _ _ _ Hin)) as Hm.
        unfold kind_on_stack in *. cbn.
        destruct (Pos.eqb (rs_next rs) m) eqn:Heq; [ | exact Hkind ].
        apply Pos.eqb_eq in Heq. lia.
    + exact
        (frames_tight_push (Frame (rs_next rs) mark_kind []) (rs_frames rs)
           floor ms Htight).
  - apply (conts_ends_moved rs rs'
             (Ends_user (state_vars (rs_state rs)) (es_regions es)
                (es_tries es) floor));
      [ | | exact Hconts ].
    + cbn [eu_floor]. rewrite Hframes'.
      apply frames_before_push. cbn. exact Hnext.
    + split; [ | split; [ | reflexivity ] ].
      * intros r Hr. cbn [eu_vars eu_regions] in *. split.
        -- cbn. destruct (Pos.eqb r x) eqn:Heq; [ | reflexivity ].
           apply Pos.eqb_eq in Heq. subst r. rewrite Hnew in Hr.
           discriminate.
        -- rewrite among_cons. apply orb_true_iff. right. exact Hr.
      * intros r Hr. cbn [eu_tries] in *. subst tries'.
        destruct mark_kind; [ exact Hr | ].
        rewrite among_cons. apply orb_true_iff. right. exact Hr.
Qed.

(* ======================================================================== *)
(* A region is ended                                                        *)
(* ======================================================================== *)

(* The stack is cut at the frame of an open region. The mark of the
   region was drawn from the floor on; with the option, in a state
   that is not stale, no plain frame is above its frame; and the state
   after the end has the parts of the property. *)
Lemma end_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (x : variable) (kind : flambda_kind)
    (p : region_primitive) (region : simple) (body : expr) (m : mark)
    (above : list frame) (f : frame) (below : list frame) :
  frames_ordered rs = true ->
  ends_at rs es ms floor ->
  state_expr (rs_state rs)
  = Let x kind (Prim (Region_primitive p) [region]) body ->
  is_end p = true ->
  eval_simple (state_vars (rs_state rs)) region = Some (Value_symbol m) ->
  rs_frames rs = above ++ f :: below ->
  frame_mark f = m ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs') = (x, unit_value) :: state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  rs_frames rs' = below ->
  rs_next rs' = rs_next rs ->
  (floor <= m)%positive
  ∧ (es_stale es = false ->
     eo_own_end opts = true ->
     filter is_plain above = [])
  ∧ ∃ (es' : ends_state) (ms' : list mark),
      es_stale es' = false ∧ ends_at rs' es' ms' floor.
Proof.
  intros Hordered (Hexpr & Hbottom & Hnext & [Hplaced Htight] & Hconts) He Hp
    Heval Hframes Hf Hexpr' Hvars' Hconts' Hframes' Hnext'.
  rewrite He in Hexpr.
  destruct (ends_expr_end es x kind p region body Hp Hexpr)
    as (r & -> & Hnew & Hr & Hown & Hbody).
  cbn in Heval.
  pose proof Hplaced as (Hmarks & Hmp & Hfloor & Hopen & Htries & Hkinds).
  unfold FlambdaMapSpec.all_among in Hopen.
  pose proof (regions_above_below r (es_open es) Hr) as Hsplit.
  rewrite Hsplit in Hmarks.
  destruct (open_marks_app _ _ _ _ Hmarks) as (ms1 & ms' & Hms & Hms1 & Hms').
  cbn [open_marks] in Hms'. rewrite Heval in Hms'.
  destruct
    (open_marks (state_vars (rs_state rs)) (regions_below r (es_open es)))
    as [ms2 | ] eqn:Hms2; cbn in Hms'; [ | discriminate ].
  injection Hms' as <-. subst ms. subst m.
  unfold marks_placed in Hmp. apply andb_true_iff in Hmp.
  destruct Hmp as [Hon Hdecrease].
  destruct (marks_decrease_app _ _ Hdecrease) as (Hd1 & Hd2 & Hcross).
  unfold frames_ordered in Hordered. rewrite Hframes in Hordered.
  rewrite Hframes in Htight.
  assert (Hfloor_m : (floor <= frame_mark f)%positive).
  { apply Pos.leb_le.
    apply (proj1 (forallb_forall _ _) Hfloor). apply in_or_app. right.
    left. reflexivity. }
  assert (Hbefore :
            ∀ bottom' : symbol,
              (bottom' <= frame_mark f)%positive ->
              frames_before bottom' below
              = frames_before bottom' (above ++ f :: below)).
  { intros bottom' Hle. symmetry.
    exact (frames_before_cut above (rs_next rs) f below bottom' Hordered Hle). }
  assert (Hlength1 :
            length ms1 = length (regions_above r (es_open es))).
  { exact (open_marks_length _ _ _ Hms1). }
  split; [ exact Hfloor_m | ]. split.
  - intros Hstale Hopt. rewrite Hstale in Htight. cbn [stale_stop] in Htight.
    apply filter_plain_nil. intros g Hg.
    destruct (is_plain g) eqn:Hplain; [ exfalso | reflexivity ].
    pose proof (frames_above_marks above (rs_next rs) f below g Hordered Hg)
      as Hgt.
    assert (Hin : In (frame_mark g) (plain_marks (above ++ f :: below))).
    { apply plain_marks_in. exists g.
      split; [ apply in_or_app; left; exact Hg | ].
      split; [ exact Hplain | reflexivity ]. }
    assert (Hfloor_g : Pos.leb floor (frame_mark g) = true).
    { apply Pos.leb_le. lia. }
    pose proof (Htight (frame_mark g) Hin Hfloor_g eq_refl) as Hms.
    apply in_app_or in Hms. destruct Hms as [Hms | [Hms | Hms]].
    + destruct
        (in_combine_exists (regions_above r (es_open es)) ms1 (frame_mark g)
           Hlength1 Hms)
        as [r1 Hpair].
      pose proof
        (proj1 (forallb_forall _ _) (Hown Hopt) r1 (in_combine_l _ _ _ _ Hpair))
        as Htry.
      cbn beta in Htry.
      assert (Hpair' :
                In (r1, frame_mark g)
                  (combine (es_open es) (ms1 ++ frame_mark f :: ms2))).
      { rewrite Hsplit. apply combine_app_l. exact Hpair. }
      pose proof (Hkinds r1 (frame_mark g) Hpair' Htry) as Hkind.
      rewrite Hframes in Hkind.
      exact
        (try_mark_not_plain (above ++ f :: below) (rs_next rs) (frame_mark g)
           Hordered Hkind Hin).
    + lia.
    + pose proof (marks_decrease_lt _ _ Hd2 _ Hms). lia.
  - exists
      (Ends_state (regions_below r (es_open es)) (es_tries es)
         (es_regions es) false (es_conts es)),
      ms2.
    split; [ reflexivity | ].
    unfold ends_at, ends_running.
    rewrite Hexpr', Hvars', Hconts', Hnext', Hframes'.
    cbn [es_open es_tries es_regions es_stale es_conts stale_stop].
    split; [ exact Hbody | ]. split; [ exact Hbottom | ].
    split; [ exact Hnext | ]. split.
    + split.
      * apply ends_placed_bind; [ exact Hnew | ].
        unfold ends_placed. cbn [es_open es_tries es_regions].
        split; [ exact Hms2 | ]. split.
        { unfold marks_placed. apply andb_true_iff.
          split; [ | exact (marks_decrease_tail _ _ Hd2) ].
          apply forallb_forall. intros m2 Hm2.
          pose proof (marks_decrease_lt _ _ Hd2 _ Hm2) as Hlt.
          rewrite
            (kept_on_stack _ _ _
               (Hbefore (frame_mark f) (Pos.le_refl _)) m2 Hlt).
          rewrite <- Hframes.
          apply (proj1 (forallb_forall _ _) Hon). apply in_or_app. right.
          right. exact Hm2. }
        split.
        { apply forallb_forall. intros m2 Hm2.
          apply (proj1 (forallb_forall _ _) Hfloor). apply in_or_app. right.
          right. exact Hm2. }
        split.
        { unfold FlambdaMapSpec.all_among. apply forallb_forall.
          intros r' Hr'. cbn beta.
          apply (proj1 (forallb_forall _ _) Hopen). apply among_In.
          apply (regions_below_among r). apply among_In. exact Hr'. }
        split; [ exact Htries | ].
        intros r2 m2 Hpair Htry.
        assert (Hpair' :
                  In (r2, m2)
                    (combine (es_open es) (ms1 ++ frame_mark f :: ms2))).
        { rewrite Hsplit. apply combine_app_r; [ exact Hlength1 | ].
          right. exact Hpair. }
        pose proof (Hkinds r2 m2 Hpair' Htry) as Hkind.
        pose proof (marks_decrease_lt _ _ Hd2 _ (in_combine_r _ _ _ _ Hpair))
          as Hlt.
        rewrite
          (kept_kind _ _ _ (Hbefore (frame_mark f) (Pos.le_refl _)) m2 Hlt).
        rewrite <- Hframes. exact Hkind.
      * apply
          (frames_tight_cut above f below (rs_next rs) floor
             (stale_stop (es_stale es) floor (ms1 ++ frame_mark f :: ms2)) ms1
             ms2 Hordered Hdecrease Htight).
        intros m' Hlt.
        exact (stale_stop_above _ _ _ _ _ _ Hdecrease Hlt).
    + apply (conts_ends_moved rs rs'
               (Ends_user (state_vars (rs_state rs)) (es_regions es)
                  (es_tries es) floor));
        [ | | exact Hconts ].
      * cbn [eu_floor]. rewrite Hframes', Hframes.
        exact (Hbefore floor Hfloor_m).
      * apply user_extends_bind. exact Hnew.
Qed.

(* ======================================================================== *)
(* The steps that leave the frames                                          *)
(* ======================================================================== *)

(* The code of a state whose static environment has the same code. *)
Lemma code_ends_same (rs rs' : region_state) :
  static_code (state_statics (rs_state rs'))
  = static_code (state_statics (rs_state rs)) ->
  code_ends cert opts rs ->
  code_ends cert opts rs'.
Proof.
  intros Hsame Hcode id c Hlookup. rewrite Hsame in Hlookup.
  exact (Hcode id c Hlookup).
Qed.

(* A local allocation: the frames have the marks and the kinds that
   they had. *)
Lemma local_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (x : variable) (kind : flambda_kind) (d : named)
    (y : variable) (kind' : flambda_kind) (args : list simple) (body' : expr)
    (v : value) :
  ends_at rs es ms floor ->
  state_expr (rs_state rs)
  = Let x kind d
      (Let y kind' (Prim (Region_primitive Local_allocation) args) body') ->
  region_named d = false ->
  state_expr (rs_state rs') = body' ->
  state_vars (rs_state rs') = (y, v) :: (x, v) :: state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  map (fun f => (frame_mark f, frame_kind f)) (rs_frames rs')
  = map (fun f => (frame_mark f, frame_kind f)) (rs_frames rs) ->
  (rs_next rs <= rs_next rs')%positive ->
  ends_at rs' es ms floor.
Proof.
  intros (Hexpr & Hbottom & Hnext & [Hplaced Htight] & Hconts) He Hd Hexpr'
    Hvars' Hconts' Hframes' Hnext'.
  rewrite He in Hexpr.
  rewrite (ends_expr_let es x kind d _ Hd), ends_expr_local in Hexpr.
  rewrite !andb_true_iff in Hexpr. destruct Hexpr as [Hx [Hy Hbody]].
  apply negb_true_iff in Hx. apply negb_true_iff in Hy.
  apply Pos.leb_le in Hnext.
  unfold ends_at, ends_running. rewrite Hexpr', Hvars', Hconts'.
  split; [ exact Hbody | ]. split; [ exact Hbottom | ].
  split; [ apply Pos.leb_le; lia | ]. split.
  - split.
    + apply ends_placed_bind; [ exact Hy | ].
      apply ends_placed_bind; [ exact Hx | ].
      exact (ends_placed_same _ _ _ _ _ _ (eq_sym Hframes') Hplaced).
    + exact (frames_tight_same _ _ _ _ _ (eq_sym Hframes') Htight).
  - apply
      (user_conts_ends cert opts ret bottom rs'
         (Ends_user ((x, v) :: state_vars (rs_state rs)) (es_regions es)
            (es_tries es) floor));
      [ apply user_extends_bind; exact Hy | ].
    apply (conts_ends_moved rs rs'
             (Ends_user (state_vars (rs_state rs)) (es_regions es)
                (es_tries es) floor));
      [ | apply user_extends_bind; exact Hx | exact Hconts ].
    cbn [eu_floor]. exact (frames_before_same floor _ _ Hframes').
Qed.

(* A [Let] that is not of a region: one variable more. *)
Lemma let_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (v : value) :
  ends_at rs es ms floor ->
  state_expr (rs_state rs) = Let x kind d body ->
  region_named d = false ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs') = (x, v) :: state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  ends_at rs' es ms floor.
Proof.
  intros (Hexpr & Hbottom & Hnext & [Hplaced Htight] & Hconts) He Hd Hexpr'
    Hvars' Hconts' Hframes' Hnext'.
  rewrite He in Hexpr. rewrite (ends_expr_let es x kind d _ Hd) in Hexpr.
  apply andb_true_iff in Hexpr. destruct Hexpr as [Hx Hbody].
  apply negb_true_iff in Hx.
  unfold ends_at, ends_running.
  rewrite Hexpr', Hvars', Hconts', Hframes', Hnext'.
  split; [ exact Hbody | ]. split; [ exact Hbottom | ].
  split; [ exact Hnext | ]. split.
  - split; [ | exact Htight ].
    apply ends_placed_bind; [ exact Hx | exact Hplaced ].
  - apply (conts_ends_moved rs rs'
             (Ends_user (state_vars (rs_state rs)) (es_regions es)
                (es_tries es) floor));
      [ | apply user_extends_bind; exact Hx | exact Hconts ].
    rewrite Hframes'. reflexivity.
Qed.

(* A step that changes the expression only. *)
Lemma expr_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) :
  ends_at rs es ms floor ->
  ends_expr cert opts es (state_expr (rs_state rs')) = true ->
  state_vars (rs_state rs') = state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  ends_at rs' es ms floor.
Proof.
  intros (_ & Hbottom & Hnext & Hrunning & Hconts) Hexpr' Hvars' Hconts'
    Hframes' Hnext'.
  unfold ends_at, ends_running in *.
  rewrite Hvars', Hconts', Hframes', Hnext'.
  split; [ exact Hexpr' | ]. split; [ exact Hbottom | ].
  split; [ exact Hnext | ]. split; [ exact Hrunning | ].
  apply (kept_conts_ends cert opts ret bottom rs rs'); [ | exact Hconts ].
  rewrite Hframes'. reflexivity.
Qed.

(* A [Let_static]: the code that it binds passes. *)
Lemma static_code_ends (rs rs' : region_state)
    (bindings : list static_binding) (statics' : static_env) :
  code_ends cert opts rs ->
  forallb (ends_binding cert opts) bindings = true ->
  eval_static_bindings (state_vars (rs_state rs)) bindings
    (state_statics (rs_state rs))
  = Some statics' ->
  state_statics (rs_state rs') = statics' ->
  code_ends cert opts rs'.
Proof.
  intros Hcode Hbindings Heval Hstatics id c Hlookup.
  rewrite Hstatics in Hlookup.
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & Hnew & _ & Hbound & _).
  rewrite Hnew, lookup_app in Hlookup.
  destruct (lookup id new_code) as [c' | ] eqn:Hnew_c.
  - injection Hlookup as ->.
    pose proof (Hbound id c (lookup_in _ _ _ Hnew_c)) as Hin.
    exact (proj1 (forallb_forall _ _) Hbindings _ Hin).
  - exact (Hcode id c Hlookup).
Qed.

Lemma static_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (bindings : list static_binding) (body : expr) :
  ends_at rs es ms floor ->
  state_expr (rs_state rs) = Let_static bindings body ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs') = state_vars (rs_state rs) ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  forallb (ends_binding cert opts) bindings = true ∧ ends_at rs' es ms floor.
Proof.
  intros Hat He Hexpr' Hvars' Hconts' Hframes' Hnext'.
  pose proof Hat as (Hexpr & _). rewrite He in Hexpr. cbn in Hexpr.
  apply andb_true_iff in Hexpr. destruct Hexpr as [Hbindings Hbody].
  split; [ exact Hbindings | ].
  apply (expr_keeps rs rs' es ms floor Hat);
    [ rewrite Hexpr'; exact Hbody | exact Hvars' | exact Hconts'
    | exact Hframes' | exact Hnext' ].
Qed.

(* ======================================================================== *)
(* A [Let_cont]                                                             *)
(* ======================================================================== *)

(* The state of the check with continuations more describes the same
   open regions. *)
Lemma ends_at_with_conts (rs rs' : region_state) (es : ends_state)
    (ms : list mark) (floor : symbol)
    (entries : list (continuation * ends_cont)) :
  ends_at rs es ms floor ->
  ends_expr cert opts (ends_with_conts es entries) (state_expr (rs_state rs'))
  = true ->
  state_vars (rs_state rs') = state_vars (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  conts_ends cert opts ret bottom rs'
    (Ends_user (state_vars (rs_state rs)) (es_regions es) (es_tries es) floor)
    (entries ++ es_conts es) (state_conts (rs_state rs')) ->
  ends_at rs' (ends_with_conts es entries) ms floor.
Proof.
  intros (_ & Hbottom & Hnext & Hrunning & _) Hexpr' Hvars' Hframes' Hnext'
    Hconts'.
  unfold ends_at, ends_running, ends_placed in *.
  rewrite Hvars', Hframes', Hnext'.
  cbn [ends_with_conts es_open es_tries es_regions es_stale es_conts].
  split; [ exact Hexpr' | ]. split; [ exact Hbottom | ].
  split; [ exact Hnext | ]. split; [ exact Hrunning | exact Hconts' ].
Qed.

(* The user that a state is, of the closures that it makes. *)
Lemma made_here (vars : env) (es : ends_state) (ms : list mark)
    (frames : list frame) (floor : symbol)
    (recursive : list (continuation * continuation_handler))
    (es' : ends_state) (base : list (continuation * ends_cont)) :
  ends_placed vars frames es ms floor ->
  es_open es' = es_open es ->
  es_tries es' = es_tries es ->
  es_regions es' = es_regions es ->
  es_conts es'
  = map (fun '(k, h) => (k, ends_cont_of cert opts k h)) recursive ++ base ->
  forallb (fun '(k, h) => ends_handler cert opts es' k h) recursive = true ->
  ends_made_at cert opts (Ends_user vars (es_regions es) (es_tries es) floor)
    vars recursive es' base.
Proof.
  intros (_ & _ & _ & Hopen & Htries & _) Hsame_open Hsame_tries Hsame_regions
    Hentries Hhandlers.
  unfold ends_made_at. rewrite Hsame_open, Hsame_tries, Hsame_regions.
  cbn [eu_vars eu_regions eu_tries].
  split; [ exact Hentries | ]. split; [ exact Hhandlers | ].
  split; [ exact Hopen | ]. split; [ exact Htries | ]. split.
  - intros r Hr. split; [ reflexivity | exact Hr ].
  - intros r Hr. exact Hr.
Qed.

Lemma let_cont_keeps (rs rs' : region_state) (es : ends_state)
    (ms : list mark) (floor : symbol) (k : continuation)
    (h : continuation_handler) (body : expr) :
  ends_at rs es ms floor ->
  state_expr (rs_state rs) = Let_cont_non_recursive k h body ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs') = state_vars (rs_state rs) ->
  state_conts (rs_state rs')
  = (k, Closure h (state_vars (rs_state rs)) (state_conts (rs_state rs)) [])
    :: state_conts (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  ends_at rs' (ends_with_conts es [(k, ends_cont_of cert opts k h)]) ms floor.
Proof.
  intros Hat He Hexpr' Hvars' Hconts' Hframes' Hnext'.
  pose proof Hat as (Hexpr & _ & _ & [Hplaced _] & Hconts).
  rewrite He in Hexpr. cbn [ends_expr] in Hexpr.
  apply andb_true_iff in Hexpr. destruct Hexpr as [Hhandler Hbody].
  assert (Hmoved :
            conts_ends cert opts ret bottom rs'
              (Ends_user (state_vars (rs_state rs)) (es_regions es)
                 (es_tries es) floor)
              (es_conts es) (state_conts (rs_state rs))).
  { apply (kept_conts_ends cert opts ret bottom rs rs'); [ | exact Hconts ].
    rewrite Hframes'. reflexivity. }
  apply (ends_at_with_conts rs rs' es ms floor _ Hat);
    [ rewrite Hexpr'; exact Hbody | exact Hvars' | exact Hframes'
    | exact Hnext' | ].
  rewrite Hconts'. destruct Hmoved as [Hbound Hunbound]. split.
  - intros k' ec Hlookup. unfold resolve. cbn [app lookup] in *.
    destruct (Pos.eqb k' k) eqn:Hk.
    + injection Hlookup as <-.
      destruct h as [params metadata handler]. cbn [ends_cont_of].
      cbn [ends_handler] in Hhandler.
      destruct (lookup k (cert_handlers cert)) as [entry | ] eqn:Hentry;
        [ | apply Ends_ok_unreachable ].
      rewrite !andb_true_iff in Hhandler.
      destruct Hhandler as [[[[Hdistinct Hnew] Hdistinct_open] Hin] Hh].
      eapply (Ends_ok_handler cert opts ret bottom rs' _ params metadata
                handler entry _ _ [] es (es_conts es));
        [ | exact Hh | exact Hdistinct | exact Hnew | exact Hdistinct_open
        | exact Hin | exact Hbound | exact Hunbound ].
      exact
        (made_here _ es ms _ floor [] es (es_conts es) Hplaced eq_refl
           eq_refl eq_refl eq_refl eq_refl).
    + exact (Hbound k' ec Hlookup).
  - intros k' Hlookup. cbn [app lookup] in *.
    destruct (Pos.eqb k' k); [ discriminate | ].
    exact (Hunbound k' Hlookup).
Qed.

Lemma let_cont_rec_keeps (rs rs' : region_state) (es : ends_state)
    (ms : list mark) (floor : symbol)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  ends_at rs es ms floor ->
  state_expr (rs_state rs) = Let_cont_recursive handlers body ->
  state_expr (rs_state rs') = body ->
  state_vars (rs_state rs') = state_vars (rs_state rs) ->
  state_conts (rs_state rs')
  = rec_closures handlers (state_vars (rs_state rs))
      (state_conts (rs_state rs))
    ++ state_conts (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  ends_at rs'
    (ends_with_conts es
       (map (fun '(k, h) => (k, ends_cont_of cert opts k h)) handlers))
    ms floor.
Proof.
  intros Hat He Hexpr' Hvars' Hconts' Hframes' Hnext'.
  pose proof Hat as (Hexpr & _ & _ & [Hplaced _] & Hconts).
  rewrite He in Hexpr. cbn [ends_expr] in Hexpr.
  apply andb_true_iff in Hexpr. destruct Hexpr as [Hhandlers Hbody].
  assert (Hmoved :
            conts_ends cert opts ret bottom rs'
              (Ends_user (state_vars (rs_state rs)) (es_regions es)
                 (es_tries es) floor)
              (es_conts es) (state_conts (rs_state rs))).
  { apply (kept_conts_ends cert opts ret bottom rs rs'); [ | exact Hconts ].
    rewrite Hframes'. reflexivity. }
  apply (ends_at_with_conts rs rs' es ms floor _ Hat);
    [ rewrite Hexpr'; exact Hbody | exact Hvars' | exact Hframes'
    | exact Hnext' | ].
  rewrite Hconts'. destruct Hmoved as [Hbound Hunbound].
  set (es' :=
         ends_with_conts es
           (map (fun '(k, h) => (k, ends_cont_of cert opts k h)) handlers))
    in *.
  change
    (map (fun '(k, h) => (k, ends_cont_of cert opts k h)) handlers
     ++ es_conts es)
    with (es_conts es').
  apply
    (closures_conts_ends cert opts ret bottom rs'
       (Ends_user (state_vars (rs_state rs)) (es_regions es) (es_tries es)
          floor)
       _ _ _ handlers es' (es_conts es));
    [ | exact Hbound | exact Hunbound | ].
  - exact
      (made_here _ es ms _ floor handlers es' (es_conts es) Hplaced eq_refl
         eq_refl eq_refl eq_refl Hhandlers).
  - split; [ | split; [ | reflexivity ] ].
    + intros r Hr. split; [ reflexivity | exact Hr ].
    + intros r Hr. exact Hr.
Qed.

(* ======================================================================== *)
(* The step of the machine without regions                                  *)
(* ======================================================================== *)

(* What a step of FlambdaSemanticsSpec.step leaves, for the expressions
   that are not a call and not a jump. *)
Lemma step_leaves (s s1 : state) (live : static_env) (s' : state) :
  step (with_statics s live) = Step s' ->
  s1 = with_statics s' (state_statics s) ->
  match state_expr s with
  | Let x kind d body =>
    ∃ v : value,
      state_expr s1 = body
      ∧ state_vars s1 = (x, v) :: state_vars s
      ∧ state_conts s1 = state_conts s
  | Let_cont_non_recursive k h body =>
    state_expr s1 = body
    ∧ state_vars s1 = state_vars s
    ∧ state_conts s1
      = (k, Closure h (state_vars s) (state_conts s) []) :: state_conts s
  | Let_cont_recursive handlers body =>
    state_expr s1 = body
    ∧ state_vars s1 = state_vars s
    ∧ state_conts s1
      = rec_closures handlers (state_vars s) (state_conts s) ++ state_conts s
  | Switch scrutinee arms =>
    ∃ (i : Z) (ac : apply_cont),
      In (i, ac) arms
      ∧ state_expr s1 = Apply_cont ac
      ∧ state_vars s1 = state_vars s
      ∧ state_conts s1 = state_conts s
  | Invalid => False
  | Let_static _ _ | Apply _ | Apply_cont _ => True
  end.
Proof.
  destruct s as [e vars conts statics hp rd].
  unfold with_statics.
  cbn [state_expr state_vars state_conts state_statics state_heap
         state_reads].
  intros Hstep ->.
  destruct e as [x kind d body | bindings body | k h body | handlers body
                | ap | ac | scrutinee arms | ];
    autorewrite with step in Hstep; try exact I.
  - destruct (exec_named live hp rd vars d) as [[[v hp'] rd'] | ];
      [ | discriminate ].
    destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
    injection Hstep as <-. exists v. repeat split.
  - injection Hstep as <-. repeat split.
  - destruct (handlers_bind_statics handlers); [ discriminate | ].
    injection Hstep as <-. repeat split.
  - destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
    destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
    destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
    injection Hstep as <-.
    destruct (find_arm_some_in i arms ac Harm) as [i' Hin].
    exists i', ac. repeat split. exact Hin.
  - discriminate.
Qed.

(* A step that is not of a region, not a [Let_static], not a call and
   not a jump keeps the parts of the property. *)
Lemma plain_keeps (rs rs' : region_state) (es : ends_state) (ms : list mark)
    (floor : symbol) (live : static_env) (s' : state) :
  ends_at rs es ms floor ->
  (∀ (x : variable) (kind : flambda_kind) (p : region_primitive)
     (args : list simple) (body : expr),
     state_expr (rs_state rs)
     <> Let x kind (Prim (Region_primitive p) args) body) ->
  (∀ (bindings : list static_binding) (body : expr),
     state_expr (rs_state rs) <> Let_static bindings body) ->
  (∀ ap : apply_expr, state_expr (rs_state rs) <> Apply ap) ->
  (∀ ac : apply_cont, state_expr (rs_state rs) <> Apply_cont ac) ->
  step (with_statics (rs_state rs) live) = Step s' ->
  rs_state rs' = with_statics s' (state_statics (rs_state rs)) ->
  rs_frames rs' = rs_frames rs ->
  rs_next rs' = rs_next rs ->
  ∃ es' : ends_state, es_stale es' = es_stale es ∧ ends_at rs' es' ms floor.
Proof.
  intros Hat Hnot_region Hnot_static Hnot_apply Hnot_jump Hstep Hstate
    Hframes' Hnext'.
  pose proof (step_leaves (rs_state rs) (rs_state rs') live s' Hstep Hstate)
    as Hleaves.
  destruct (state_expr (rs_state rs))
    as [x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ] eqn:He.
  - destruct Hleaves as (v & Hexpr' & Hvars' & Hconts').
    exists es. split; [ reflexivity | ].
    apply (let_keeps rs rs' es ms floor x kind d body v Hat He);
      [ | exact Hexpr' | exact Hvars' | exact Hconts' | exact Hframes'
      | exact Hnext' ].
    destruct d as [s | [u | b | va | hp | rp] args | | | | | ];
      try reflexivity.
    exfalso. exact (Hnot_region x kind rp args body eq_refl).
  - exfalso. exact (Hnot_static bindings body eq_refl).
  - destruct Hleaves as (Hexpr' & Hvars' & Hconts').
    exists (ends_with_conts es [(k, ends_cont_of cert opts k h)]).
    split; [ reflexivity | ].
    exact
      (let_cont_keeps rs rs' es ms floor k h body Hat He Hexpr' Hvars' Hconts'
         Hframes' Hnext').
  - destruct Hleaves as (Hexpr' & Hvars' & Hconts').
    exists
      (ends_with_conts es
         (map (fun '(k, h) => (k, ends_cont_of cert opts k h)) handlers)).
    split; [ reflexivity | ].
    exact
      (let_cont_rec_keeps rs rs' es ms floor handlers body Hat He Hexpr'
         Hvars' Hconts' Hframes' Hnext').
  - exfalso. exact (Hnot_apply ap eq_refl).
  - exfalso. exact (Hnot_jump ac eq_refl).
  - destruct Hleaves as (i & ac & Hin & Hexpr' & Hvars' & Hconts').
    exists es. split; [ reflexivity | ].
    apply (expr_keeps rs rs' es ms floor Hat);
      [ | exact Hvars' | exact Hconts' | exact Hframes' | exact Hnext' ].
    rewrite Hexpr'. pose proof Hat as (Hexpr & _). rewrite He in Hexpr.
    cbn [ends_expr] in Hexpr.
    exact (proj1 (forallb_forall _ _) Hexpr (i, ac) Hin).
  - destruct Hleaves.
Qed.

End Let_steps.
