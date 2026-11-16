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
   the steps that leave the store as it was (a [Let_cont], a [Switch], an
   [Apply_cont], an [Apply], a [Let] that is no local allocation). Each
   comes to [entered]. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaKnownAritySpec FlambdaKnownArityProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedReadsProofs
  FlambdaRegionsCheckedViewsProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs FlambdaRegionsCheckedCallProofs
  FlambdaRegionsCheckedAppliedProofs FlambdaRegionsCheckedMovesProofs
  FlambdaRegionsCheckedDeepProofs FlambdaRegionsCheckedLetProofs
  FlambdaRegionsCheckedSharedProofs.
Import ListNotations.

Lemma Forall2_of_map {A B C : Type} (R : B -> C -> Prop) (f : A -> B)
    (l : list A) :
  ∀ l' : list C,
    Forall2 (fun a c => R (f a) c) l l' -> Forall2 R (map f l) l'.
Proof.
  induction l as [ | a l IH]; intros l' H; inversion H; subst; cbn [map];
    constructor; [ assumption | apply IH; assumption ].
Qed.

Lemma forallb_of_map {A B : Type} (f : B -> bool) (g : A -> B) (l : list A) :
  forallb f (map g l) = forallb (fun a => f (g a)) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite IH. reflexivity.
Qed.

Lemma lookup_closures (k : continuation)
    (group recursive : list (continuation * continuation_handler))
    (vars : env) (conts : cont_env) :
  lookup k (map (fun '(k, h) => (k, Closure h vars conts recursive)) group)
  = option_map (fun h => Closure h vars conts recursive) (lookup k group).
Proof.
  exact
    (lookup_map_entry (fun _ h => Closure h vars conts recursive) k group).
Qed.

Section Same.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Variable cs : check_state.

Variable ms : list mark.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hordered : frames_ordered rs = true.

Hypothesis Hcode : code_checked cert codes rs.

Hypothesis Hlocal : local_closures_fit cert rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hcells : cells_deep rs = true.

Hypothesis Hgood_vars :
  env_values_all (good cert codes rs) (state_vars (rs_state rs)).

Hypothesis Hgood_conts :
  conts_values_all (good cert codes rs) (state_conts (rs_state rs)).

Hypothesis Hgood_statics :
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    (static_values (live_statics rs)).

Hypothesis Hgood_heap :
  heap_values_all (good cert codes rs) (state_heap (rs_state rs)).

Hypothesis Hcheck :
  check_expr cert codes cs (state_expr (rs_state rs)) = true.

Hypothesis Hincl : incl (code_of_expr (state_expr (rs_state rs))) codes.

Hypothesis Hrun : running_checked rs cs ms.

Hypothesis Hconts :
  conts_checked cert codes rs (running_user rs cs ms) (cs_conts cs)
    (state_conts (rs_state rs)).

Local Abbreviation vars := (state_vars (rs_state rs)).

Local Abbreviation conts := (state_conts (rs_state rs)).

Local Abbreviation entered := (entered cert codes rs).

Local Abbreviation good := (good cert codes rs).

Lemma head_below (s : simple) :
  In s (evaluated_simples (state_expr (rs_state rs))) ->
  simple_below (rs_base rs) s = true.
Proof. exact (head_simples_below rs s Hfresh). Qed.

Lemma running_frame : frame_ok rs cs (running_user rs cs ms).
Proof.
  destruct Hrun as (Hopen & Hplaced & _). constructor.
  - exact (proj1 (open_marks_marked _ _ _) Hopen).
  - exact Hplaced.
  - apply Pos.le_refl.
Qed.

(* The state that runs, as a caller. *)
Definition running_caller : caller := Caller cs vars ms (rs_next rs).

Lemma running_caller_ok : caller_ok rs running_caller.
Proof.
  destruct Hrun as (Hopen & Hplaced & Hscoped & _). constructor.
  - exact Hopen.
  - exact Hplaced.
  - apply andb_true_iff in Hscoped as [H _]. exact H.
  - exact (placed_below rs ms Hfresh Hplaced).
  - apply Pos.le_refl.
Qed.

Lemma simples_good (simples : list simple) (vs : list value) :
  eval_simples vars simples = Some vs -> Forall good vs.
Proof.
  intros Hvs. apply Forall_forall. intros v Hin.
  destruct (eval_simples_in vars simples vs v Hvs Hin) as (s & _ & Hs).
  exact (simple_good cert codes rs vars Hgood_vars s v Hs).
Qed.

Lemma simples_caller_hold (simples : list simple) (vs : list value) :
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  forallb (check_usable cs) simples = true ->
  eval_simples vars simples = Some vs ->
  Forall2
    (fun s v => caller_holds rs running_caller (lifetime_of cs s) v)
    simples vs.
Proof.
  intros Hb Hu Hvs. pose proof (eval_simples_Forall2 _ _ _ Hvs) as H.
  revert H. apply Forall2_impl_in. intros s v Hin Hv.
  split; [ exact (forallb_in _ _ s Hu Hin) | ].
  exact
    (holds_next rs _ _ _ v Hfresh
       (simple_holds rs vars cs ms Hrun s v (Hb s Hin) Hv)).
Qed.

Lemma jump_args_hold (simples : list simple) (vs : list value) :
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  eval_simples vars simples = Some vs ->
  Forall2 (arg_holds rs cs (running_user rs cs ms))
    (map (fun s => (Some s, lifetime_of cs s)) simples) vs.
Proof.
  intros Hb Hvs. apply Forall2_of_map.
  pose proof (eval_simples_Forall2 _ _ _ Hvs) as H. revert H.
  apply Forall2_impl_in. intros s v Hin Hv. split.
  - exact (simple_holds rs vars cs ms Hrun s v (Hb s Hin) Hv).
  - cbn [fst]. intros r [= ->]. exact Hv.
Qed.

(* A simple that may be evaluated is no mark. *)
Lemma usable_no_mark (simples : list simple) :
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  forallb (check_usable cs) simples = true ->
  existsb (simple_open_mark rs) simples = false.
Proof.
  intros Hb Hu.
  destruct (existsb (simple_open_mark rs) simples) eqn:Hex; [ | reflexivity ].
  apply existsb_exists in Hex as (s & Hin & Hmark).
  pose proof (open_mark_unusable rs s Hmark) as Hunusable.
  unfold FlambdaRegionsSpec.simple_usable in Hunusable.
  destruct (eval_simple vars s) as [v | ] eqn:Hv; [ | discriminate ].
  rewrite
    (within_usable rs None v
       (running_usable_within rs cs ms s v Hrun (Hb s Hin)
          (forallb_in _ _ s Hu Hin) Hv))
    in Hunusable.
  discriminate.
Qed.

(* ======================================================================== *)
(* A [Let_cont]                                                             *)
(* ======================================================================== *)

(* The closures of a group of handlers, made now, are as their entries
   say. [recursive] is the group, or nothing for a handler that is not
   recursive. *)
Lemma group_conts
    (recursive group : list (continuation * continuation_handler)) :
  let csr :=
    with_conts cs
      (map (fun '(k, h) => (k, cont_entry_of cert k h)) recursive)
  in
  forallb (fun '(k, h) => check_handler cert codes csr k h) group = true ->
  (∀ (k : continuation) (h : continuation_handler),
     In (k, h) group -> incl (code_of_handler h) codes) ->
  recursive = [] ∨ recursive = group ->
  conts_checked cert codes rs (running_user rs cs ms)
    (map (fun '(k, h) => (k, cont_entry_of cert k h)) group ++ cs_conts cs)
    (map (fun '(k, h) => (k, Closure h vars conts recursive)) group ++ conts).
Proof.
  intros csr Hgroup Hgroup_incl Hrec.
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  destruct Hconts as [Hsome Hnone].
  assert (Hmade :
    made_at cert codes rs (running_user rs cs ms) vars recursive csr
      (cs_conts cs) ms (rs_next rs)).
  { split; [ reflexivity | ].
    split; [ destruct Hrec as [-> | ->]; [ reflexivity | exact Hgroup ] | ].
    split;
      [ destruct Hrec as [-> | ->]; [ intros k h [] | exact Hgroup_incl ] | ].
    split; [ exact Hscoped | ]. split; [ exact Hopen | ].
    split; [ exact (placed_decrease _ _ Hplaced) | ].
    split; [ exact (placed_below rs ms Hfresh Hplaced) | ].
    split; [ apply Pos.leb_refl | ].
    split;
      [ intros x l v Hl Hv;
        exact (holds_next rs _ _ l v Hfresh (Hvalues x l v Hl Hv))
      | ].
    split;
      [ intros x l v Hl Hv;
        exact (contents_next rs _ _ l v Hfresh (Hcontents x l v Hl Hv))
      | ].
    split; [ intros m Hin _; exact Hin | ].
    intros r Hr. split; [ reflexivity | exact Hr ]. }
  split.
  - intros k ce. rewrite lookup_app, lookup_entries. unfold resolve.
    rewrite lookup_app, lookup_closures.
    destruct (lookup k group) as [[params metadata handler] | ] eqn:Hk;
      cbn [option_map].
    + intros [= <-]. unfold cont_entry_of.
      destruct (lookup k (cert_handlers cert)) as [entry | ] eqn:Hentry;
        [ | apply Ok_unreachable ].
      pose proof (forallb_in _ _ _ Hgroup (lookup_in _ _ _ Hk)) as Hhandler.
      cbn [check_handler] in Hhandler. rewrite Hentry in Hhandler.
      apply andb_true_iff in Hhandler as [Hfits Hcheck_h].
      apply (Ok_handler cert codes rs (running_user rs cs ms) params metadata
               handler entry vars conts recursive csr (cs_conts cs) ms
               (rs_next rs) Hmade Hfits Hcheck_h).
      * exact (Hgroup_incl k _ (lookup_in _ _ _ Hk)).
      * exact Hsome.
      * exact Hnone.
    + intros Hbase. exact (Hsome k ce Hbase).
  - intros k.
    rewrite !lookup_app, lookup_entries, lookup_closures.
    destruct (lookup k group); cbn [option_map]; [ discriminate | ].
    exact (Hnone k).
Qed.

Lemma let_cont_entered (k : continuation) (h : continuation_handler)
    (body : expr) (s' : state) :
  state_expr (rs_state rs) = Let_cont_non_recursive k h body ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hstep. unfold with_statics in Hstep. rewrite He in Hstep.
  rewrite step_let_cont_non_recursive in Hstep. injection Hstep as <-.
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  apply andb_true_iff in Hc as [Hhandler Hbody].
  pose proof Hincl as Hi. rewrite He in Hi. cbn [code_of_expr] in Hi.
  exists (with_conts cs [(k, cont_entry_of cert k h)]), ms.
  cbn [state_expr state_vars state_conts].
  split; [ exact Hbody | ].
  split; [ intros b Hin; apply Hi; apply in_or_app; right; exact Hin | ].
  split; [ exact Hrun | ]. split.
  - apply (group_conts [] [(k, h)]).
    + destruct h as [params metadata handler]. cbn [forallb].
      rewrite andb_true_r. exact Hhandler.
    + intros k0 h0 [[= <- <-] | []] b Hin. apply Hi. apply in_or_app. left.
      exact Hin.
    + left. reflexivity.
  - split; [ exact Hgood_vars | ].
    split; [ split; [ exact Hgood_vars | exact Hgood_conts ] | ].
    exact Hgood_conts.
Qed.

Lemma let_cont_rec_entered
    (handlers : list (continuation * continuation_handler)) (body : expr)
    (s' : state) :
  state_expr (rs_state rs) = Let_cont_recursive handlers body ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hstep. unfold with_statics in Hstep. rewrite He in Hstep.
  rewrite step_let_cont_recursive in Hstep.
  destruct (handlers_bind_statics handlers); [ discriminate | ].
  injection Hstep as <-.
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  cbv zeta in Hc. apply andb_true_iff in Hc as [Hhandlers Hbody].
  pose proof Hincl as Hi. rewrite He in Hi. cbn [code_of_expr] in Hi.
  exists
    (with_conts cs (map (fun '(k, h) => (k, cont_entry_of cert k h)) handlers)),
    ms.
  cbn [state_expr state_vars state_conts].
  split; [ exact Hbody | ].
  split; [ intros b Hin; apply Hi; apply in_or_app; right; exact Hin | ].
  split; [ exact Hrun | ]. split.
  - apply (group_conts handlers handlers);
      [ exact Hhandlers | | right; reflexivity ].
    intros k h Hin b Hb. apply Hi. apply in_or_app. left. apply in_flat_map.
    exists (k, h). split; [ exact Hin | exact Hb ].
  - split; [ exact Hgood_vars | ].
    apply conts_values_all_app; [ | exact Hgood_conts ].
    exact (rec_closures_values_all _ vars conts handlers Hgood_vars
             Hgood_conts).
Qed.

(* ======================================================================== *)
(* A [Switch]                                                               *)
(* ======================================================================== *)

Lemma switch_entered (scrutinee : simple) (arms : list (Z * apply_cont))
    (s' : state) :
  state_expr (rs_state rs) = Switch scrutinee arms ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hstep. unfold with_statics in Hstep.
  rewrite He, step_switch in Hstep.
  destruct (eval_simple vars scrutinee) as [v | ]; [ | discriminate ].
  destruct (naked_immediate_of_value v) as [i | ]; [ | discriminate ].
  destruct (find_arm i arms) as [ac | ] eqn:Harm; [ | discriminate ].
  injection Hstep as <-.
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  apply andb_true_iff in Hc as [_ Harms].
  destruct (find_arm_some_in i arms ac Harm) as (i' & Hin).
  pose proof (forallb_in _ _ _ Harms Hin) as Hjump. cbv beta iota in Hjump.
  exists cs, ms. cbn [state_expr state_vars state_conts].
  split; [ exact Hjump | ]. split; [ intros b [] | ].
  split; [ exact Hrun | ]. split; [ exact Hconts | ].
  split; [ exact Hgood_vars | exact Hgood_conts ].
Qed.

(* ======================================================================== *)
(* A jump                                                                   *)
(* ======================================================================== *)

(* A jump that the check passes, by the step of [step] or, where a
   region is given, by the step of this machine. *)
Lemma jump_entered (k : continuation) (simples : list simple)
    (vs : list value) (s' : state) :
  jump cs k (map (fun s => (Some s, lifetime_of cs s)) simples) = true ->
  (∀ s : simple, In s simples -> simple_below (rs_base rs) s = true) ->
  eval_simples vars simples = Some vs ->
  apply_cont_value (resolve k conts) vs (live_statics rs)
    (state_heap (rs_state rs)) (state_reads (rs_state rs))
  = Step s'
  ∨ (existsb (simple_open_mark rs) simples = true
     ∧ ∃ (params : list variable) (metadata : handler_metadata)
         (handler : expr) (def_vars : env) (def_conts : cont_env)
         (recursive : list (continuation * continuation_handler))
         (handler_vars : env),
         resolve k conts
         = Closure (Handler params metadata handler) def_vars def_conts
             recursive
         ∧ bind_params params vs def_vars = Some handler_vars
         ∧ s'
           = State handler handler_vars
               (rec_closures recursive def_vars def_conts ++ def_conts)
               (live_statics rs) (state_heap (rs_state rs))
                 (state_reads (rs_state rs))) ->
  entered s'.
Proof.
  intros Hjump Hb Hvs Hstep.
  destruct Hconts as [Hsome Hnone].
  pose proof (jump_args_hold simples vs Hb Hvs) as Hargs.
  pose proof (simples_good simples vs Hvs) as Hgood_vs.
  pose proof (resolve_values_all _ k conts Hgood_conts) as Hgood_c.
  destruct (lookup k (cs_conts cs)) as [ce | ] eqn:Hk.
  - pose proof (Hsome k ce Hk) as Hok.
    rewrite (jump_some cs k ce _ Hk) in Hjump.
    destruct Hstep
      as [Hstep
         | (Hmark & params & metadata & handler & def_vars & def_conts
            & recursive & handler_vars & Hc & Hbind & ->)].
    + destruct (applied_all cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
                  Hordered Hcells Hgood_statics (state_heap (rs_state rs))
                  (state_reads (rs_state rs)))
        as (Hcont_all & _ & _).
      exact
        (Hcont_all _ ce _ Hok cs _ vs s' running_frame Hjump Hargs Hgood_vs
           Hgood_c Hstep).
    + destruct ce as [params0 kinds entry | | may_be_local | ].
      * destruct (ok_handler_inv cert codes rs _ _ _ _ _ Hok)
          as (metadata0 & handler0 & vars0 & conts0 & recursive0 & cs0 & base
              & ms0 & next & Hc0 & Hkinds & _).
        rewrite Hc in Hc0. injection Hc0 as <- <- <- <- <- <-. subst kinds.
        rewrite Hc in Hok, Hgood_c.
        cbn [jump_entry] in Hjump.
        destruct (region_args cs params (hm_params_kinds metadata) _)
          as [given | ] eqn:Hregion; [ | discriminate ].
        apply andb_true_iff in Hjump as [Horder Hvalues].
        destruct (enter_handler cert codes rs Hordered Hcells cs _ _ _ _ _ _
                    _ _ _ vs handler_vars given running_frame Hok Hregion
                    Horder Hvalues Hargs Hbind)
          as (cs' & ms' & Hcheck' & Hincl' & Hrun' & Hconts').
        destruct Hgood_c as [Hgood_def Hgood_dconts].
        exists cs', ms'. cbn [state_expr state_vars state_conts].
        split; [ exact Hcheck' | ]. split; [ exact Hincl' | ].
        split; [ exact Hrun' | ]. split; [ exact Hconts' | ]. split.
        -- exact (bind_params_values_all _ params vs def_vars handler_vars
                    Hgood_vs Hgood_def Hbind).
        -- apply conts_values_all_app; [ | exact Hgood_dconts ].
           apply rec_closures_values_all; assumption.
      * discriminate Hjump.
      * cbn [jump_entry] in Hjump.
        rewrite usable_no_mark in Hmark; [ discriminate | exact Hb | ].
        rewrite forallb_of_map in Hjump. revert Hjump. apply forallb_impl.
        intros s _ H. apply andb_true_iff in H as [H _]. exact H.
      * cbn [jump_entry] in Hjump.
        rewrite usable_no_mark in Hmark; [ discriminate | exact Hb | ].
        rewrite forallb_of_map in Hjump. revert Hjump. apply forallb_impl.
        intros s _ H. exact (forever_usable cs _ H).
  - assert (Hc : resolve k conts = Outer k)
      by (unfold resolve; rewrite (Hnone k Hk); reflexivity).
    rewrite Hc in Hstep.
    destruct Hstep
      as [Hstep
         | (_ & params & metadata & handler & def_vars & def_conts
            & recursive & handler_vars & H & _)];
      [ discriminate Hstep | discriminate H ].
Qed.

Lemma apply_cont_entered (ac : apply_cont) (s' : state) :
  state_expr (rs_state rs) = Apply_cont ac ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hstep. unfold with_statics in Hstep.
  rewrite He, step_apply_cont in Hstep.
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
    [ | discriminate ].
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  apply (jump_entered (ac_continuation ac) (ac_args ac) vs s' Hc);
    [ | exact Hvs | left; exact Hstep ].
  intros s Hin. apply head_below. rewrite He. exact Hin.
Qed.

(* An [Apply_cont] that gives a region. *)
Lemma region_apply_cont_entered (ac : apply_cont) (rs' : region_state) :
  state_expr (rs_state rs) = Apply_cont ac ->
  existsb (simple_open_mark rs) (ac_args ac) = true ->
  region_apply_cont rs ac = Region_step rs' ->
  ∃ s' : state,
    rs'
    = Region_state (with_statics s' (state_statics (rs_state rs)))
        (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs)
        (rs_cells rs)
    ∧ entered s'.
Proof.
  intros He Hmark Hstep. unfold region_apply_cont in Hstep. cbv zeta in Hstep.
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  assert (Hb :
    ∀ s : simple, In s (ac_args ac) -> simple_below (rs_base rs) s = true).
  { intros s Hin. apply head_below. rewrite He. exact Hin. }
  destruct (resolve (ac_continuation ac) conts)
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to]
    eqn:Hc_eq.
  2-4: destruct (plain_step_step rs rs' Hstep) as (s' & Hs' & ->);
    exists s'; split; [ reflexivity | exact (apply_cont_entered ac s' He Hs') ].
  destruct (eval_simples vars (ac_args ac)) as [vs | ] eqn:Hvs;
    [ | discriminate ].
  destruct (kinds_eqb _ _); [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ] eqn:Hbind;
    [ | discriminate ].
  injection Hstep as <-.
  exists
    (State handler handler_vars
       (rec_closures recursive def_vars def_conts ++ def_conts)
       (live_statics rs) (state_heap (rs_state rs))
         (state_reads (rs_state rs))).
  split; [ reflexivity | ].
  apply (jump_entered (ac_continuation ac) (ac_args ac) vs _ Hc Hb Hvs).
  right. split; [ exact Hmark | ].
  exists params, metadata, handler, def_vars, def_conts, recursive,
    handler_vars.
  split; [ exact Hc_eq | ]. split; [ exact Hbind | reflexivity ].
Qed.

(* ======================================================================== *)
(* A call                                                                   *)
(* ======================================================================== *)

Lemma apply_entered (ap : apply_expr) (s' : state) :
  state_expr (rs_state rs) = Apply ap ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hstep.
  pose proof Hcheck as Hc. rewrite He in Hc. cbn [check_expr] in Hc.
  apply andb_true_iff in Hc as [Hc Hcall].
  apply andb_true_iff in Hc as [Hc Husable_extra].
  apply andb_true_iff in Hc as [Husable_callee Husable_args].
  destruct (call_result cert codes cs ap) as [result | ] eqn:Hresult;
    [ | discriminate ].
  apply andb_true_iff in Hcall as [Hjump_ret Hjump_exn].
  assert (Hb :
    ∀ s : simple,
      In s (ap_callee ap :: ap_args ap ++ exn_extra_simples ap) ->
      simple_below (rs_base rs) s = true).
  { intros s Hin. apply head_below. rewrite He. exact Hin. }
  unfold with_statics in Hstep. rewrite He, step_apply in Hstep.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ] eqn:Hcallee;
    [ | discriminate ].
  destruct (eval_simples vars (ap_args ap)) as [vs | ] eqn:Hvs;
    [ | discriminate ].
  destruct (eval_simples vars (exn_extra_simples ap)) as [extra | ]
    eqn:Hextra; [ | discriminate ].
  set (a := running_caller).
  pose proof running_caller_ok as Hcaller.
  destruct Hconts as [Hsome Hnone].
  destruct (applied_all cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
              Hordered Hcells Hgood_statics (state_heap (rs_state rs))
              (state_reads (rs_state rs)))
    as (_ & Hret_all & _).
  assert (Hret :
    ret_ok cert codes rs a result (resolve (ap_continuation ap) conts)).
  { apply (Ret_to cert codes rs a result (ap_continuation ap)).
    - exact Hjump_ret.
    - intros ce Hk. exact (Hsome _ ce Hk).
    - intros Hk. unfold resolve. rewrite (Hnone _ Hk). reflexivity. }
  assert (Hraise :
    raise_ok cert codes rs a
      (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)).
  { apply (Raise_to cert codes rs a (ap_exn_continuation ap)
             (exn_extra_simples ap) extra).
    - exact Hextra.
    - apply simples_caller_hold; [ | exact Husable_extra | exact Hextra ].
      intros s Hin. apply Hb. right. apply in_or_app. right. exact Hin.
    - exact Hjump_exn.
    - intros ce Hk. exact (Hsome _ ce Hk).
    - intros Hk. unfold resolve. rewrite (Hnone _ Hk). reflexivity. }
  assert (Hgood_ret :
    cont_values_all good (resolve (ap_continuation ap) conts))
    by exact (resolve_values_all _ _ _ Hgood_conts).
  assert (Hgood_exn :
    cont_values_all good
      (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)).
  { apply with_extra_args_values_all;
      [ exact (resolve_values_all _ _ _ Hgood_conts)
      | exact (simples_good _ _ Hextra) ]. }
  assert (Hcallee_below : simple_below (rs_base rs) (ap_callee ap) = true)
    by (apply Hb; left; reflexivity).
  assert (Hargs :
    Forall2 (caller_holds rs a) (map (lifetime_of cs) (ap_args ap)) vs).
  { apply Forall2_of_map.
    apply simples_caller_hold; [ | exact Husable_args | exact Hvs ].
    intros s Hin. apply Hb. right. apply in_or_app. left. exact Hin. }
  pose proof
    (simple_holds rs vars cs ms Hrun _ callee Hcallee_below Hcallee)
    as Hcallee_holds.
  pose proof (simple_good cert codes rs vars Hgood_vars _ callee Hcallee)
    as Hcallee_good.
  pose proof
    (running_usable_within rs cs ms _ callee Hrun Hcallee_below
       Husable_callee Hcallee)
    as Hcallee_live.
  pose proof (Hret_all a result _ Hret Hcaller Hgood_ret) as Hreturns.
  unfold call_result in Hresult.
  destruct (ap_call_kind ap) as [id | | code_ids] eqn:Hkind.
  3: (* A call of known arity that has a step is the call of unknown
        arity. *)
    rewrite <- (known_arity_call callee code_ids vs _ _ _ _ _ _ _) in Hstep
      by (rewrite Hstep; discriminate).
  - exact
      (call_direct cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
         Hordered Hcells Hgood_statics _ _ a id callee _ _ vs _ _ _ _ result
         s' Hcaller Hcallee_good Hcallee_holds (simples_good _ _ Hvs) Hargs
         Hresult Hret Hraise Hreturns Hgood_ret Hgood_exn Hstep).
  - exact
      (call_unknown cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
         Hordered Hcells Hgood_statics _ _ a callee _ _ vs _ _ _ _ result s'
         Hcaller Hcallee_good Hcallee_live Hcallee_holds
         (simples_good _ _ Hvs) Hargs Hresult Hret Hraise Hreturns Hgood_ret
         Hgood_exn Hstep).
  - exact
      (call_unknown cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
         Hordered Hcells Hgood_statics _ _ a callee _ _ vs _ _ _ _ result s'
         Hcaller Hcallee_good Hcallee_live Hcallee_holds
         (simples_good _ _ Hvs) Hargs Hresult Hret Hraise Hreturns Hgood_ret
         Hgood_exn Hstep).
Qed.

(* ======================================================================== *)
(* A [Let] that is no local allocation                                      *)
(* ======================================================================== *)

Lemma let_entered (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (s' : state) :
  state_expr (rs_state rs) = Let x kind d body ->
  region_named d = false ->
  named_changes_heap d = false ->
  marker_of x body = No_marker ->
  step (with_statics (rs_state rs) (live_statics rs)) = Step s' ->
  entered s'.
Proof.
  intros He Hd Hsame Hmarker Hstep.
  unfold with_statics in Hstep. rewrite He, step_let_exec in Hstep.
  destruct (exec_named _ _ _ _ d) as [[[v hp'] rd'] | ] eqn:Hv;
    [ | discriminate ].
  destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
  injection Hstep as <-.
  pose proof Hcheck as Hc. rewrite He in Hc.
  rewrite (check_expr_let cert codes cs x kind d body Hd) in Hc.
  unfold check_let in Hc. cbv zeta in Hc. rewrite Hmarker in Hc.
  apply andb_true_iff in Hc as [Hc Hrest].
  apply andb_true_iff in Hc as [Hx Husable].
  pose proof Hincl as Hi. rewrite He in Hi. cbn [code_of_expr] in Hi.
  assert (Hsimples :
    evaluated_simples (Let x kind d body) = simples_of_named d).
  { destruct d as [s | [u | b | v0 | hp | p] args | | | | | ];
      first [ reflexivity | discriminate ]. }
  assert (Hb :
    ∀ s : simple,
      In s (simples_of_named d) -> simple_below (rs_base rs) s = true).
  { intros s Hin. apply head_below. rewrite He, Hsimples. exact Hin. }
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
  assert (Hx_region : among x (cs_regions cs) = false)
    by (unfold not_a_region in Hx; apply negb_true_iff in Hx; exact Hx).
  destruct (scoped_parts cs Hscoped) as [Hs_env Hs_contents].
  assert (Hbound :
    ∀ l lc : lifetime,
      lifetime_holds rs (cs_open cs) ms None l v ->
      contents_hold rs (cs_open cs) ms None lc v ->
      (∀ r : variable, l = Of_region r -> among r (cs_regions cs) = true) ->
      (∀ r : variable, lc = Of_region r -> among r (cs_regions cs) = true) ->
      good v ->
      check_expr cert codes (bind_lifetimes x l lc cs) body = true ->
      entered
        (State body ((x, v) :: vars) conts (live_statics rs) hp' rd')).
  { intros l lc Hholds Hcontents_v Hregion Hregion_c Hgood_v Hbody.
    exists (bind_lifetimes x l lc cs), ms.
    cbn [state_expr state_vars state_conts].
    split; [ exact Hbody | ]. split; [ exact Hi | ]. split.
    - split;
        [ change (open_marks ((x, v) :: vars) (cs_open cs) = Some ms);
          rewrite (open_marks_bind vars x v _ Hx_open); exact Hopen
        | ].
      split; [ exact Hplaced | ].
      split;
        [ exact (bind_lifetimes_scoped cs x l lc Hscoped Hregion Hregion_c) | ].
      split.
      + intros y l0 w. cbn [bind_lifetimes cs_env cs_open].
        rewrite !lookup_cons.
        destruct (Pos.eqb y x);
          [ intros [= <-] [= <-]; exact Hholds | exact (Hvalues y l0 w) ].
      + intros y l0 w. cbn [bind_lifetimes cs_contents cs_open].
        rewrite !lookup_cons.
        destruct (Pos.eqb y x);
          [ intros [= <-] [= <-]; exact Hcontents_v
          | exact (Hcontents y l0 w) ].
    - split.
      + destruct Hconts as [Hsome Hnone]. split; [ | exact Hnone ].
        intros k ce Hk.
        apply (user_conts cert codes rs (running_user rs cs ms));
          [ | exact (Hsome k ce Hk) ].
        split; [ | split ].
        * intros r Hr.
          cbn [u_vars u_regions running_user bind_lifetimes cs_regions].
          split; [ | exact Hr ].
          rewrite lookup_cons.
          destruct (Pos.eqb r x) eqn:Hrx; [ | reflexivity ].
          apply Pos.eqb_eq in Hrx. subst r.
          cbn [u_regions running_user] in Hr. rewrite Hx_region in Hr.
          discriminate.
        * intros m Hin. left. exact Hin.
        * apply Pos.le_refl.
      + split; [ apply Forall_cons; [ exact Hgood_v | exact Hgood_vars ] | ].
        exact Hgood_conts. }
  destruct (is_allocation d) eqn:Halloc.
  - cbn [orb] in Hrest. apply andb_true_iff in Hrest as [Hforever Hbody].
    destruct (exec_named_makes _ _ _ _ _ _ _ _ Halloc Hv) as [Heval _].
    destruct (made_good cert codes rs vars cs ms Hrun Hgood_vars d v Halloc
                Hb Hforever Heval)
      as [Hgood_v Hheap].
    apply (Hbound Forever Forever);
      [ exact Hheap
      | exact
          (holds_contents rs _ ms Forever v Hordered Hcells Hplaced Hheap)
      | intros r H; discriminate H
      | intros r H; discriminate H
      | exact Hgood_v
      | exact Hbody ].
  - assert (Hcell : is_cell_allocation d = false).
    { destruct (is_cell_allocation d) eqn:H; [ | reflexivity ].
      rewrite (cell_allocation_changes_heap d H) in Hsame. discriminate. }
    rewrite Hcell in Hrest. cbn [orb] in Hrest.
    apply andb_true_iff in Hrest as [_ Hbody].
    apply (Hbound (named_lifetime cs d) (named_contents cs d)).
    + exact
        (named_holds rs Hfresh Hordered Hdeep Hstatics Hcells vars cs ms Hrun
           d v _ _ _ Halloc Hsame Hb Hv).
    + exact
        (named_contents_hold rs Hfresh Hordered Hdeep Hstatics Hcells vars cs
           ms Hrun d v _ _ _ Halloc Hsame Hb Hv).
    + intros r Hr. exact (named_scoped cs d r Hs_env Hs_contents Hr).
    + intros r Hr.
      exact (named_contents_scoped cs d r Hs_env Hs_contents Hr).
    + exact
        (named_good cert codes rs Hgood_statics Hgood_heap vars Hgood_vars d
           v _ _ _ Halloc Hsame Hv).
    + exact Hbody.
Qed.

End Same.
