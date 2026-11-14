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
   what the cases of the proof that a step keeps the property share. The
   values that the state holds, taken together; the continuations when the
   store changes; a step that leaves the store as it was; the counts.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs
  FlambdaValueSpec FlambdaPrimitiveSpec FlambdaPrimitiveProofs
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
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
  FlambdaRegionsCheckedDeepProofs FlambdaRegionsCheckedLetProofs.
Import ListNotations.

(* ======================================================================== *)
(* The values that the state holds                                          *)
(* ======================================================================== *)

Lemma allocated_values_impl (P Q : value -> Prop)
    (allocated : statically_allocated_value) :
  (∀ v : value, P v -> Q v) ->
  allocated_values P allocated ->
  allocated_values Q allocated.
Proof.
  intros Himpl H. apply allocated_values_in. intros w Hin.
  exact (Himpl w (proj1 (allocated_values_in P allocated) H w Hin)).
Qed.

Lemma allocated_values_and (P Q : value -> Prop)
    (allocated : statically_allocated_value) :
  allocated_values P allocated ->
  allocated_values Q allocated ->
  allocated_values (fun v => P v ∧ Q v) allocated.
Proof.
  intros HP HQ. apply allocated_values_in. intros w Hin. split.
  - exact (proj1 (allocated_values_in P allocated) HP w Hin).
  - exact (proj1 (allocated_values_in Q allocated) HQ w Hin).
Qed.

(* [values_plain] and [paps_fit] together: every value that the state
   holds is good. *)
Lemma good_split (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) :
  state_values (good cert codes rs) rs
  <-> values_plain rs ∧ paps_fit cert codes rs.
Proof.
  unfold values_plain, paps_fit, state_values. cbv zeta. split.
  - intros (Hvars & Hconts & Hstatics). split; (split; [ | split ]).
    + revert Hvars. apply Forall_impl. intros b H. exact (proj1 H).
    + revert Hconts. apply conts_values_all_impl. intros v H. exact (proj1 H).
    + revert Hstatics. apply Forall_impl. intros b.
      apply allocated_values_impl. intros v H. exact (proj1 H).
    + revert Hvars. apply Forall_impl. intros b H. exact (proj2 H).
    + revert Hconts. apply conts_values_all_impl. intros v H. exact (proj2 H).
    + revert Hstatics. apply Forall_impl. intros b.
      apply allocated_values_impl. intros v H. exact (proj2 H).
  - intros ((Hv1 & Hc1 & Hs1) & (Hv2 & Hc2 & Hs2)).
    split; [ exact (env_values_all_and _ _ _ Hv1 Hv2) | ].
    split; [ exact (conts_values_all_and _ _ _ Hc1 Hc2) | ].
    apply Forall_forall. intros b Hin.
    apply allocated_values_and.
    + exact (proj1 (Forall_forall _ _) Hs1 b Hin).
    + exact (proj1 (Forall_forall _ _) Hs2 b Hin).
Qed.

(* ======================================================================== *)
(* Marks                                                                    *)
(* ======================================================================== *)

(* The marks that are on the stack were drawn. *)
Lemma placed_below (rs : region_state) (ms : list mark) :
  symbols_fresh rs = true ->
  marks_placed (rs_frames rs) ms = true ->
  forallb (fun m => Pos.ltb m (rs_next rs)) ms = true.
Proof.
  intros Hfresh Hplaced. apply forallb_forall. intros m Hin.
  destruct (on_stack_frame m _ (marks_placed_in _ _ m Hplaced Hin))
    as (f & Hf & <-).
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & _ & _ & _ & Hframes & _).
  destruct (Hframes f Hf) as (_ & Hlt & _). apply Pos.ltb_lt. exact Hlt.
Qed.

Lemma placed_decrease (frames : list frame) (ms : list mark) :
  marks_placed frames ms = true -> marks_decrease ms = true.
Proof.
  unfold marks_placed. intros H. apply andb_true_iff in H as [_ H]. exact H.
Qed.

(* A variable that is no region is bound: the marks of the open
   regions are found as they were. *)
Lemma open_marks_bind (vars : env) (x : variable) (v : value)
    (opens : list variable) :
  among x opens = false ->
  open_marks ((x, v) :: vars) opens = open_marks vars opens.
Proof.
  induction opens as [ | r opens IH]; [ reflexivity | ].
  rewrite among_cons. intros H. apply orb_false_iff in H as [Hxr Hrest].
  cbn [open_marks]. rewrite lookup_cons, Pos.eqb_sym, Hxr, (IH Hrest).
  reflexivity.
Qed.

Lemma not_region_not_open (cs : check_state) (x : variable) :
  check_state_scoped cs = true ->
  not_a_region cs x = true ->
  among x (cs_open cs) = false.
Proof.
  intros Hscoped Hx. unfold not_a_region in Hx. apply negb_true_iff in Hx.
  apply andb_true_iff in Hscoped as [Hopen _].
  destruct (among x (cs_open cs)) eqn:H; [ | reflexivity ].
  rewrite (all_among_among _ _ x Hopen H) in Hx. discriminate.
Qed.

(* ======================================================================== *)
(* The continuations when the store changes                                 *)
(* ======================================================================== *)

Lemma conts_grow (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state) (u u' : user)
    (entries : list (continuation * cont_entry)) (conts : cont_env) :
  store_grows rs rs' ->
  (u_limit u <= rs_next rs)%positive ->
  user_extends u u' ->
  conts_checked cert codes rs u entries conts ->
  conts_checked cert codes rs' u' entries conts.
Proof.
  intros Hgrows Hlimit Hextends [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ce Hk. apply (user_conts cert codes rs' u u' ce _ Hextends).
  exact
    (proj1 (grows_conts rs rs' Hgrows cert codes) u ce _ (Hsome k ce Hk)
       Hlimit).
Qed.

Lemma conts_cut (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state) (m0 : mark) (u u' : user)
    (entries : list (continuation * cont_entry)) (conts : cont_env) :
  store_cut rs rs' m0 ->
  frames_ordered rs = true ->
  In m0 (u_marks u) ∨ (u_limit u <= m0)%positive ->
  user_extends u u' ->
  conts_checked cert codes rs u entries conts ->
  conts_checked cert codes rs' u' entries conts.
Proof.
  intros Hcut Hordered Hmark Hextends [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ce Hk. apply (user_conts cert codes rs' u u' ce _ Hextends).
  exact
    (proj1 (cut_conts rs rs' m0 Hcut Hordered cert codes) u ce _
       (Hsome k ce Hk) Hmark).
Qed.

(* ======================================================================== *)
(* What the store binds, when it changes                                    *)
(* ======================================================================== *)

Lemma frames_deep_mono (rs rs' : region_state) (frames : list frame) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  frames_deep rs frames = true ->
  frames_deep rs' frames = true.
Proof.
  intros Hheap. induction frames as [ | f below IH]; [ reflexivity | ].
  cbn [frames_deep]. intros H. apply andb_true_iff in H as [Hf Hbelow].
  apply andb_true_iff. split; [ | exact (IH Hbelow) ].
  revert Hf. apply forallb_impl. intros [s allocated] _.
  apply allocated_all_mono. intros s' H'. apply orb_true_iff in H' as [H' | H'].
  - rewrite (Hheap s' H'). reflexivity.
  - rewrite H'. apply orb_true_r.
Qed.

Lemma statics_good_moves (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state)
    (bindings : list (symbol * statically_allocated_value)) :
  store_moves rs rs' ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
     In (s, allocated) bindings ->
     allocated_below (rs_next rs) allocated = true) ->
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    bindings ->
  Forall (fun binding => allocated_values (good cert codes rs') (snd binding))
    bindings.
Proof.
  intros Hmoves Hbelow Hgood. apply Forall_forall. intros [s allocated] Hin.
  cbn [snd].
  apply (allocated_good_moves cert codes rs rs' Hmoves allocated
           (Hbelow s allocated Hin)).
  exact (proj1 (Forall_forall _ _) Hgood (s, allocated) Hin).
Qed.

(* What the state holds, by its parts, with what is below the
   counter. *)
Lemma state_below (rs : region_state) :
  symbols_fresh rs = true ->
  env_values_all (fun v => value_below (rs_next rs) v = true)
    (state_vars (rs_state rs))
  ∧ conts_values_all (fun v => value_below (rs_next rs) v = true)
      (state_conts (rs_state rs))
  ∧ ∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) (static_values (live_statics rs)) ->
      (s < rs_next rs)%positive
      ∧ allocated_below (rs_next rs) allocated = true.
Proof.
  intros Hfresh.
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & [Hvars Hconts] & Hstore).
  split; [ exact Hvars | ]. split; [ exact Hconts | ].
  intros s allocated Hin.
  exact (store_bound_below _ _ _ _ s allocated Hstore Hin).
Qed.

(* ======================================================================== *)
(* A step that leaves the store as it was                                   *)
(* ======================================================================== *)

Lemma same_store_checked (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state) :
  rs_frames rs' = rs_frames rs ->
  rs_base rs' = rs_base rs ->
  rs_next rs' = rs_next rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs ->
  entered cert codes rs (rs_state rs') ->
  state_checked cert codes rs'.
Proof.
  destruct rs' as
    [[e' vars' conts' statics' hp' rd'] frames' base' next' counts'].
  cbn [rs_frames rs_base rs_next rs_state state_statics].
  intros -> -> -> -> Hinv' Hordered'
    (Hinv & _ & Hcode & Hlocal & Hdeep & Hstatics & Hplain & Hpaps & _)
    (cs' & ms' & Hcheck & Hincl & Hrun & Hconts & Hgood_vars & Hgood_conts).
  destruct (proj2 (good_split cert codes rs) (conj Hplain Hpaps))
    as (_ & _ & Hgood_statics).
  match goal with
  | |- state_checked _ _ ?r => set (rs' := r) in *
  end.
  destruct (state_below rs (invariant_fresh rs Hinv)) as (_ & _ & Hbound).
  destruct (state_below rs' (invariant_fresh rs' Hinv'))
    as (Hbelow_vars & Hbelow_conts & _).
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - intros s H. exact H.
    - intros s _.
      destruct (lookup s (static_values (live_statics rs))) as [allocated | ]
        eqn:Hs; [ left; exact Hs | intros _; exact Hs ]. }
  split; [ exact Hinv' | ]. split; [ exact Hordered' | ].
  split; [ exact Hcode | ]. split; [ exact Hlocal | ].
  split; [ exact (frames_deep_mono rs rs' _ (fun s H => H) Hdeep) | ].
  split; [ exact Hstatics | ].
  assert (Hgood' : state_values (good cert codes rs') rs').
  { split;
      [ exact
          (env_good_moves cert codes rs rs' Hmoves _ Hgood_vars Hbelow_vars)
      | ].
    split;
      [ exact
          (conts_good_moves cert codes rs rs' Hmoves _ Hgood_conts
             Hbelow_conts)
      | ].
    apply (statics_good_moves cert codes rs rs' _ Hmoves);
      [ | exact Hgood_statics ].
    intros s allocated Hin. exact (proj2 (Hbound s allocated Hin)). }
  apply good_split in Hgood' as [Hplain' Hpaps'].
  split; [ exact Hplain' | ]. split; [ exact Hpaps' | ].
  exists cs', ms'. split; [ exact Hcheck | ]. split; [ exact Hincl | ].
  split; [ exact Hrun | ].
  assert (Hgrows : store_grows rs rs').
  { constructor.
    - apply Pos.le_refl.
    - intros s H. exact H.
    - intros s H. exact H.
    - intros m _. reflexivity. }
  destruct Hconts as [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ce Hk.
  exact
    (proj1 (grows_conts rs rs' Hgrows cert codes) _ ce _ (Hsome k ce Hk)
       (Pos.le_refl _)).
Qed.

Lemma checked_entered (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) :
  state_checked cert codes rs -> entered cert codes rs (rs_state rs).
Proof.
  intros (_ & _ & _ & _ & _ & _ & Hplain & Hpaps & cs & ms & Hcheck & Hincl
          & Hrun & Hconts).
  destruct (proj2 (good_split cert codes rs) (conj Hplain Hpaps))
    as (Hgood_vars & Hgood_conts & _).
  exists cs, ms. split; [ exact Hcheck | ]. split; [ exact Hincl | ].
  split; [ exact Hrun | ]. split; [ exact Hconts | ].
  split; [ exact Hgood_vars | exact Hgood_conts ].
Qed.

(* The property does not look at the counts. *)
Lemma counted_checked (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) :
  state_checked cert codes rs -> state_checked cert codes (count_uses rs).
Proof.
  intros H.
  apply (same_store_checked cert codes rs (count_uses rs)); try reflexivity.
  - pose proof (proj1 H) as Hinv. unfold region_invariant in Hinv |- *.
    apply andb_true_iff in Hinv as [Hfresh Hlive]. apply andb_true_iff.
    split; [ exact Hfresh | exact (values_live_count_uses rs Hlive) ].
  - exact (proj1 (proj2 H)).
  - exact H.
  - exact (checked_entered cert codes rs H).
Qed.
