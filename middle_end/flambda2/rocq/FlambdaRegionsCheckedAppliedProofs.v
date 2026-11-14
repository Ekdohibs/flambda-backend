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
   applying a continuation that is as its entry says, to values that are as
   the check of the jump says, comes to [entered]. By induction on
   [cont_ok], [ret_ok] and [raise_ok]. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs FlambdaRegionsCheckedCallProofs.
Import ListNotations.

(* What [jump] asks of the arguments, by the entry of the
   continuation. *)
Definition jump_entry (cs : check_state) (ce : cont_entry)
    (args : list (option simple * lifetime)) : bool :=
  match ce with
  | Unreachable => false
  | To_return may_be_local =>
    forallb
      (fun '(_, l) => lifetime_usable cs l && (may_be_local || is_forever l))
      args
  | To_raise => forallb (fun '(_, l) => is_forever l) args
  | To_handler params kinds entry =>
    match region_args cs params kinds args with
    | None => false
    | Some given =>
      let named := map (region_at_jump given) (he_open entry) in
      in_order named (cs_open cs)
      && value_args cs given named kinds args (he_params entry)
    end
  end.

Lemma jump_some (cs : check_state) (k : continuation) (ce : cont_entry)
    (args : list (option simple * lifetime)) :
  lookup k (cs_conts cs) = Some ce -> jump cs k args = jump_entry cs ce args.
Proof. intros H. unfold jump. rewrite H. destruct ce; reflexivity. Qed.

Lemma eval_simples_Forall2 (vars : env) (args : list simple) :
  ∀ vs : list value,
    eval_simples vars args = Some vs ->
    Forall2 (fun s v => eval_simple vars s = Some v) args vs.
Proof.
  induction args as [ | s args IH]; cbn [eval_simples]; intros vs.
  - intros [= <-]. constructor.
  - destruct (eval_simple vars s) as [v | ] eqn:Hs; [ | discriminate ].
    destruct (eval_simples vars args) as [vs' | ]; [ | discriminate ].
    cbn. intros [= <-]. constructor; [ exact Hs | exact (IH vs' eq_refl) ].
Qed.

Lemma apply_with_extra (c : cont_value) (extra vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) :
  apply_cont_value (with_extra_args c extra) vs statics hp rd
  = apply_cont_value c (vs ++ extra) statics hp rd.
Proof.
  destruct extra as [ | v extra]; [ rewrite app_nil_r | ]; reflexivity.
Qed.

Lemma good_with_extra (P : value -> Prop) (c : cont_value)
    (extra : list value) :
  cont_values_all P (with_extra_args c extra) ->
  cont_values_all P c ∧ Forall P extra.
Proof.
  destruct extra as [ | v extra]; cbn [with_extra_args cont_values_all].
  - intros H. split; [ exact H | constructor ].
  - intros [Hextra Hc]. split; assumption.
Qed.

Lemma map_const_same {A B C : Type} (x : C) (l : list A) (l' : list B) :
  length l = length l' -> map (fun _ => x) l = map (fun _ => x) l'.
Proof.
  revert l'. induction l as [ | a l IH]; intros [ | b l']; cbn [length map];
    try discriminate; [ reflexivity | ].
  intros [= H]. rewrite (IH l' H). reflexivity.
Qed.

Section Applied.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hcode : code_checked cert codes rs.

Hypothesis Hlocal : local_closures_fit cert rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hgood_statics :
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    (static_values (live_statics rs)).

Variable hp : heap.
Variable rd : read_record.

Local Abbreviation good := (good cert codes rs).

Local Abbreviation entered := (entered cert codes rs).

(* The frame of a caller. *)
Lemma caller_frame (a : caller) :
  caller_ok rs a -> frame_ok rs (caller_check a) (caller_user a).
Proof.
  intros Hcaller. constructor.
  - exact (caller_marked rs a Hcaller).
  - exact (caller_placed rs a Hcaller).
  - exact (caller_limit rs a Hcaller).
Qed.

(* A value that is live is as the lifetime of the newest region of a
   caller says, with no bound. *)
Lemma newest_holds (a : caller) (v : value) :
  caller_ok rs a ->
  live rs v ->
  lifetime_holds rs (cs_open (caller_check a)) (caller_marks a) None
    (newest_lifetime (caller_check a)) v.
Proof.
  intros Hcaller Hlive. pose proof (caller_marked rs a Hcaller) as Hmarked.
  unfold newest_lifetime.
  destruct (cs_open (caller_check a)) as [ | r opens].
  - inversion Hmarked. exact Hlive.
  - inversion Hmarked as [ | r0 m opens0 ms Hr Hrest]; subst.
    cbn [lifetime_holds region_place]. rewrite Pos.eqb_refl.
    intros _. exact Hlive.
Qed.

(* Arguments that all have one lifetime. *)
Lemma same_lifetime_args (cs : check_state) (u : user) (l : lifetime)
    (vs : list value) :
  Forall (lifetime_holds rs (cs_open cs) (u_marks u) None l) vs ->
  Forall2 (arg_holds rs cs u) (map (fun _ => (None, l)) vs) vs.
Proof.
  induction 1 as [ | v vs Hv _ IH]; cbn [map]; constructor; [ | exact IH ].
  split; [ exact Hv | ]. intros r H. discriminate H.
Qed.

Definition applied_cont (u : user) (ce : cont_entry) (c : cont_value)
    : Prop :=
  ∀ (cs : check_state) (args : list (option simple * lifetime))
    (vs : list value) (s' : state),
    frame_ok rs cs u ->
    jump_entry cs ce args = true ->
    Forall2 (arg_holds rs cs u) args vs ->
    Forall good vs ->
    cont_values_all good c ->
    apply_cont_value c vs (live_statics rs) hp rd = Step s' ->
    entered s'.

Definition applied_ret (a : caller) (l : lifetime) (c : cont_value) : Prop :=
  caller_ok rs a ->
  cont_values_all good c ->
  returns_ok cert codes rs hp rd a l c.

Definition applied_raise (a : caller) (c : cont_value) : Prop :=
  caller_ok rs a ->
  cont_values_all good c ->
  ∀ (vs : list value) (s' : state),
    Forall good vs ->
    Forall (fun v => value_all (heap_symbol rs) v = true) vs ->
    apply_cont_value c vs (live_statics rs) hp rd = Step s' ->
    entered s'.

Lemma applied_all :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c -> applied_cont u ce c)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c -> applied_ret a l c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c -> applied_raise a c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ => applied_cont u ce c)
           (fun a l c _ => applied_ret a l c)
           (fun a c _ => applied_raise a c)).
  - (* A handler. *)
    intros u params metadata handler entry vars conts recursive cs0 base ms
      next Hmade Hfits Hcheck Hincl Hconts _ Hnone cs args vs s' Hframe Hjump
      Hargs Hgood_vs Hgood_c Hstep.
    cbn [jump_entry] in Hjump.
    destruct (region_args cs params (hm_params_kinds metadata) args)
      as [given | ] eqn:Hregion; [ | discriminate ].
    apply andb_true_iff in Hjump as [Horder Hvalues].
    cbn [apply_cont_value] in Hstep.
    destruct (kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata));
      [ | discriminate ].
    destruct (bind_params params vs vars) as [vars' | ] eqn:Hbind;
      [ | discriminate ].
    injection Hstep as <-.
    destruct (enter_handler cert codes rs cs u params metadata handler entry
                vars conts recursive args vs vars' given Hframe
                (Ok_handler cert codes rs u params metadata handler entry vars
                   conts recursive cs0 base ms next Hmade Hfits Hcheck Hincl
                   Hconts Hnone)
                Hregion Horder Hvalues Hargs Hbind)
      as (cs' & ms' & Hcheck' & Hincl' & Hrunning & Hconts').
    destruct Hgood_c as [Hgood_vars Hgood_conts].
    exists cs', ms'. cbn [state_expr state_vars state_conts].
    split; [ exact Hcheck' | ]. split; [ exact Hincl' | ].
    split; [ exact Hrunning | ]. split; [ exact Hconts' | ]. split.
    + exact (bind_params_values_all _ params vs vars vars' Hgood_vs Hgood_vars
               Hbind).
    + apply conts_values_all_app; [ | exact Hgood_conts ].
      apply rec_closures_values_all; assumption.
  - (* A handler with no entry. *)
    intros u c cs args vs s' _ Hjump. discriminate Hjump.
  - (* The return continuation of code. *)
    intros u may_be_local a c Hcalled Hret IH cs args vs s' Hframe Hjump Hargs
      Hgood_vs Hgood_c Hstep.
    pose proof
      (called_from_caller_ok rs u a Hcalled (frame_limit rs cs u Hframe))
      as Hcaller.
    cbn [jump_entry] in Hjump.
    apply (IH Hcaller Hgood_c vs s' Hgood_vs); [ | | exact Hstep ].
    + apply (Forall2_right_in _ _ args vs Hargs). intros [s l] v Hin [Hholds _].
      pose proof (forallb_in _ _ (s, l) Hjump Hin) as H. cbv beta iota in H.
      apply andb_true_iff in H as [Husable _].
      apply (holds_live rs (u_vars u) (cs_open cs) (u_marks u) l v
               (frame_marks rs cs u Hframe) (frame_placed rs cs u Hframe));
        [ | exact Hholds ].
      apply usable_rank. exact Husable.
    + apply (Forall2_right_in _ _ args vs Hargs). intros [s l] v Hin [Hholds _].
      pose proof (forallb_in _ _ (s, l) Hjump Hin) as H. cbv beta iota in H.
      apply andb_true_iff in H as [Husable Hlocal_or].
      cbn [snd] in Hholds.
      destruct may_be_local.
      * apply (newest_holds a v Hcaller).
        apply (holds_live rs (u_vars u) (cs_open cs) (u_marks u) l v
                 (frame_marks rs cs u Hframe) (frame_placed rs cs u Hframe));
          [ | exact Hholds ].
        apply usable_rank. exact Husable.
      * cbn [orb] in Hlocal_or. destruct l; try discriminate. exact Hholds.
  - (* The exception continuation of code. *)
    intros u a c Hcalled Hraise IH cs args vs s' Hframe Hjump Hargs Hgood_vs
      Hgood_c Hstep.
    pose proof
      (called_from_caller_ok rs u a Hcalled (frame_limit rs cs u Hframe))
      as Hcaller.
    cbn [jump_entry] in Hjump.
    apply (IH Hcaller Hgood_c vs s' Hgood_vs); [ | exact Hstep ].
    apply (Forall2_right_in _ _ args vs Hargs). intros [s l] v Hin [Hholds _].
    pose proof (forallb_in _ _ (s, l) Hjump Hin) as H. cbv beta iota in H.
    cbn [snd] in Hholds. destruct l; try discriminate. exact Hholds.
  - (* A continuation of the caller. *)
    intros a l k c Hjump Hsome IH Houter Hcaller Hgood_c vs s' Hgood_vs
      Hlive_vs Hholds_vs Hstep.
    destruct (lookup k (cs_conts (caller_check a))) as [ce | ] eqn:Hk;
      [ | rewrite (Houter eq_refl) in Hstep; discriminate Hstep ].
    apply (IH ce eq_refl (caller_check a) (map (fun _ => (None, l)) vs) vs s'
             (caller_frame a Hcaller));
      [ | exact (same_lifetime_args _ (caller_user a) l vs Hholds_vs)
        | exact Hgood_vs | exact Hgood_c | exact Hstep ].
    unfold jump_result in Hjump. rewrite Hk in Hjump.
    destruct ce as [params kinds entry | | may_be_local | ].
    + rewrite (jump_some _ k _ _ Hk) in Hjump.
      destruct (ok_handler_inv cert codes rs _ _ _ _ _ (Hsome _ eq_refl))
        as (metadata & handler & vars & conts & recursive & _ & _ & _ & _
            & Hc & _).
      rewrite Hc in Hstep. cbn [apply_cont_value] in Hstep.
      destruct (kinds_eqb _ _); [ | discriminate ].
      destruct (bind_params params vs vars) as [vars' | ] eqn:Hbind;
        [ | discriminate ].
      rewrite (map_const_same (None, l) vs params
                 (bind_params_same_length _ _ _ _ Hbind)).
      exact Hjump.
    + rewrite (jump_some _ k _ _ Hk) in Hjump. discriminate Hjump.
    + rewrite (jump_some _ k _ _ Hk) in Hjump. cbn [jump_entry forallb] in *.
      rewrite andb_true_r in Hjump. apply forallb_forall.
      intros [s l0] Hin. apply in_map_iff in Hin as (v & [= <- <-] & _).
      exact Hjump.
    + rewrite (jump_some _ k _ _ Hk) in Hjump. cbn [jump_entry forallb] in *.
      rewrite andb_true_r in Hjump. apply forallb_forall.
      intros [s l0] Hin. apply in_map_iff in Hin as (v & [= <- <-] & _).
      exact Hjump.
  - (* What is left of an application with too many arguments. *)
    intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise _ Hcaller Hgood_c vs s' Hgood_vs
      Hlive_vs Hholds_vs Hstep.
    destruct Hgood_c as (Hgood_remaining & Hgood_ret & Hgood_exn).
    cbn [apply_cont_value] in Hstep.
    destruct vs as [ | v [ | v' vs]]; try discriminate Hstep.
    exact
      (call_unknown cert codes rs Hfresh Hcode Hlocal Hdeep Hstatics
         Hgood_statics hp rd a v l ls remaining remaining_arity return_arity
         return_to exn_to l' s' Hcaller (Forall_inv Hgood_vs)
         (Forall_inv Hlive_vs) (Forall_inv Hholds_vs) Hgood_remaining Hholds
         Hunknown Hret Hraise (IHret Hcaller Hgood_ret) Hgood_ret Hgood_exn
         Hstep).
  - (* A longer lifetime. *)
    intros a l l' c Hlasts Hret IH Hcaller Hgood_c vs s' Hgood_vs Hlive_vs
      Hholds_vs Hstep.
    apply (IH Hcaller Hgood_c vs s' Hgood_vs Hlive_vs); [ | exact Hstep ].
    revert Hholds_vs. apply Forall_impl. intros v.
    exact (holds_rank rs (caller_vars a) _ _ l l' v
             (caller_marked rs a Hcaller) (caller_placed rs a Hcaller)
             (lasts_rank _ _ _ Hlasts)).
  - (* A continuation of the caller, with extra arguments. *)
    intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hcaller
      Hgood_c vs s' Hgood_vs Hheap_vs Hstep.
    rewrite apply_with_extra in Hstep.
    apply good_with_extra in Hgood_c as [Hgood_c Hgood_extra].
    destruct (lookup k (cs_conts (caller_check a))) as [ce | ] eqn:Hk;
      [ | rewrite (Houter eq_refl) in Hstep; discriminate Hstep ].
    set (cs := caller_check a) in *.
    set (extras := map (fun s => (Some s, lifetime_of cs s)) simples) in *.
    apply (IH ce eq_refl cs (map (fun _ => (None, Forever)) vs ++ extras)
             (vs ++ extra) s' (caller_frame a Hcaller));
      [ | | apply Forall_app; split; assumption | exact Hgood_c | exact Hstep ].
    + rewrite (jump_some _ k _ _ Hk) in Hjump.
      destruct ce as [params kinds entry | | may_be_local | ].
      * destruct (ok_handler_inv cert codes rs _ _ _ _ _ (Hsome _ eq_refl))
          as (metadata & handler & vars & conts & recursive & _ & _ & _ & _
              & Hc & _).
        rewrite Hc in Hstep. cbn [apply_cont_value] in Hstep.
        destruct (kinds_eqb _ _); [ | discriminate ].
        destruct (bind_params params (vs ++ extra) vars) as [vars' | ]
          eqn:Hbind; [ | discriminate ].
        pose proof (bind_params_same_length _ _ _ _ Hbind) as Hlength.
        cbn [jump_entry] in Hjump.
        destruct (region_args cs params kinds ((None, Forever) :: extras))
          as [given | ] eqn:Hregion; [ | discriminate ].
        destruct (region_args_spec cs params kinds _ given Hregion)
          as (_ & Hlen_args & _).
        pose proof (Forall2_length Hholds) as Hlen_extra.
        unfold extras in Hlen_args.
        rewrite length_app in Hlength. cbn [length] in Hlen_args.
        rewrite length_map in Hlen_args.
        destruct vs as [ | v [ | v' vs]]; cbn [length] in Hlength; try lia.
        cbn [map app jump_entry]. rewrite Hregion. exact Hjump.
      * discriminate Hjump.
      * cbn [jump_entry forallb] in *.
        apply andb_true_iff in Hjump as [_ Hextras].
        rewrite forallb_app, Hextras, andb_true_r. apply forallb_forall.
        intros [s l0] Hin. apply in_map_iff in Hin as (v & [= <- <-] & _).
        cbn [lifetime_usable is_forever andb]. apply orb_true_r.
      * cbn [jump_entry forallb] in *.
        apply andb_true_iff in Hjump as [_ Hextras].
        rewrite forallb_app, Hextras, andb_true_r. apply forallb_forall.
        intros [s l0] Hin. apply in_map_iff in Hin as (v & [= <- <-] & _).
        reflexivity.
    + apply Forall2_app.
      * apply (same_lifetime_args cs (caller_user a) Forever vs).
        exact Hheap_vs.
      * unfold extras. clear Hjump Hstep.
        pose proof (eval_simples_Forall2 _ _ _ Heval) as Hvalues.
        revert Hvalues Hholds. generalize extra. clear.
        induction simples as [ | s simples IH]; intros extra Hvalues Hholds;
          inversion Hvalues; subst; inversion Hholds; subst; cbn [map];
          constructor.
        -- split.
           ++ cbn [snd]. match goal with
              | H : caller_holds _ _ _ _ |- _ =>
                exact (holds_none _ _ _ _ _ _ (proj2 H))
              end.
           ++ intros r [= ->]. assumption.
        -- apply IH; assumption.
Qed.

End Applied.
