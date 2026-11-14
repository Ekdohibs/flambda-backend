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
  FlambdaHeapSpec FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsCellsProofs FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
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

(* The values of the cells of a heap, as a property of each entry and
   as FlambdaHeapSpec.heap_values_all has them. *)
Lemma heap_values_all_Forall (P : value -> Prop) (hp : heap) :
  Forall (fun binding => Forall P (values_of_cell (snd binding))) hp
  <-> heap_values_all P hp.
Proof.
  unfold heap_values_all. rewrite Forall_forall. split.
  - intros H l c v Hin Hv.
    exact (proj1 (Forall_forall _ _) (H (l, c) Hin) v Hv).
  - intros H [l c] Hin. apply Forall_forall. intros v Hv.
    exact (H l c v Hin Hv).
Qed.

(* [values_plain] and [paps_fit] together: every value that the state
   holds is good. *)
Lemma good_split (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) :
  state_values (good cert codes rs) rs
  <-> values_plain rs ∧ paps_fit cert codes rs.
Proof.
  unfold values_plain, paps_fit, state_values. cbv zeta. split.
  - intros (Hvars & Hconts & Hstatics & Hheap).
    split; (split; [ | split; [ | split ] ]).
    + revert Hvars. apply Forall_impl. intros b H. exact (proj1 H).
    + revert Hconts. apply conts_values_all_impl. intros v H. exact (proj1 H).
    + revert Hstatics. apply Forall_impl. intros b.
      apply allocated_values_impl. intros v H. exact (proj1 H).
    + revert Hheap. apply Forall_impl. intros b. apply Forall_impl.
      intros v H. exact (proj1 H).
    + revert Hvars. apply Forall_impl. intros b H. exact (proj2 H).
    + revert Hconts. apply conts_values_all_impl. intros v H. exact (proj2 H).
    + revert Hstatics. apply Forall_impl. intros b.
      apply allocated_values_impl. intros v H. exact (proj2 H).
    + revert Hheap. apply Forall_impl. intros b. apply Forall_impl.
      intros v H. exact (proj2 H).
  - intros ((Hv1 & Hc1 & Hs1 & Hh1) & (Hv2 & Hc2 & Hs2 & Hh2)).
    split; [ exact (env_values_all_and _ _ _ Hv1 Hv2) | ].
    split; [ exact (conts_values_all_and _ _ _ Hc1 Hc2) | ]. split.
    + apply Forall_forall. intros b Hin.
      apply allocated_values_and.
      * exact (proj1 (Forall_forall _ _) Hs1 b Hin).
      * exact (proj1 (Forall_forall _ _) Hs2 b Hin).
    + apply Forall_forall. intros b Hin. apply Forall_forall. intros v Hv.
      split.
      * exact
          (proj1 (Forall_forall _ _) (proj1 (Forall_forall _ _) Hh1 b Hin) v
             Hv).
      * exact
          (proj1 (Forall_forall _ _) (proj1 (Forall_forall _ _) Hh2 b Hin) v
             Hv).
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
  heap_values_all (closed rs) (state_heap (rs_state rs)) ->
  conts_values_all (closed rs) conts ->
  (u_limit u <= rs_next rs)%positive ->
  user_extends u u' ->
  conts_checked cert codes rs u entries conts ->
  conts_checked cert codes rs' u' entries conts.
Proof.
  intros Hgrows Hheap Hclosed Hlimit Hextends [Hsome Hnone].
  split; [ | exact Hnone ].
  intros k ce Hk. apply (user_conts cert codes rs' u u' ce _ Hextends).
  exact
    (proj1 (grows_conts rs rs' Hgrows cert codes Hheap) u ce _ (Hsome k ce Hk)
       (resolve_values_all _ k conts Hclosed) Hlimit).
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

(* The same of locations and of cells. What is asked of the locations
   of a value is kept where the cells are listed as they were at
   those locations. *)
Lemma cells_all_same (rs rs' : region_state) (frames frames' : list frame)
    (v : value) :
  (∀ l : location,
     In l (locations_of_value v) ->
     lookup l (rs_cells rs') = lookup l (rs_cells rs)) ->
  (∀ m : mark, on_stack m frames = true -> on_stack m frames' = true) ->
  value_all_cells (fun l => heap_location rs l || cell_in rs frames l) v
  = true ->
  value_all_cells (fun l => heap_location rs' l || cell_in rs' frames' l) v
  = true.
Proof.
  intros Hsame Hstack H. apply value_all_cells_iff. intros l Hl.
  pose proof (proj1 (value_all_cells_iff _ v) H l Hl) as Hold.
  cbv beta in Hold. unfold heap_location, cell_in in *.
  rewrite (Hsame l Hl).
  destruct (lookup l (rs_cells rs)) as [m | ]; [ | reflexivity ].
  cbn [is_some negb orb] in *. exact (Hstack m Hold).
Qed.

Lemma heap_cells_same (rs rs' : region_state) (v : value) :
  (∀ l : location,
     In l (locations_of_value v) ->
     lookup l (rs_cells rs') = lookup l (rs_cells rs)) ->
  value_all_cells (heap_location rs) v = true ->
  value_all_cells (heap_location rs') v = true.
Proof.
  intros Hsame H. apply value_all_cells_iff. intros l Hl.
  pose proof (proj1 (value_all_cells_iff _ v) H l Hl) as Hold.
  unfold heap_location in *. rewrite (Hsame l Hl). exact Hold.
Qed.

Lemma frames_deep_cells_same (rs rs' : region_state) (frames : list frame) :
  (∀ (s : symbol) (allocated : statically_allocated_value) (w : value)
     (l : location),
     In (s, allocated) (live_locals frames) ->
     In w (values_of_static allocated) ->
     In l (locations_of_value w) ->
     lookup l (rs_cells rs') = lookup l (rs_cells rs)) ->
  frames_deep_cells rs frames = true ->
  frames_deep_cells rs' frames = true.
Proof.
  induction frames as [ | f below IH]; intros Hsame; [ reflexivity | ].
  cbn [frames_deep_cells]. intros H. apply andb_true_iff in H as [Hf Hbelow].
  apply andb_true_iff. split.
  - apply forallb_forall. intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hf Hin) as Hall.
    cbv beta iota in Hall. apply forallb_forall. intros w Hw.
    apply (cells_all_same rs rs' (f :: below) (f :: below) w);
      [ | exact (fun m H => H) | exact (forallb_in _ _ w Hall Hw) ].
    intros l Hl. apply (Hsame s allocated w l); [ | exact Hw | exact Hl ].
    change (live_locals (f :: below))
      with (frame_locals f ++ live_locals below).
    apply in_or_app. left. exact Hin.
  - apply IH; [ | exact Hbelow ]. intros s allocated w l Hin.
    apply (Hsame s allocated w l).
    change (live_locals (f :: below))
      with (frame_locals f ++ live_locals below).
    apply in_or_app. right. exact Hin.
Qed.

Lemma statics_cells_same (rs rs' : region_state)
    (values : list (symbol * statically_allocated_value)) :
  (∀ (s : symbol) (allocated : statically_allocated_value) (w : value)
     (l : location),
     In (s, allocated) values ->
     In w (values_of_static allocated) ->
     In l (locations_of_value w) ->
     lookup l (rs_cells rs') = lookup l (rs_cells rs)) ->
  forallb
    (fun '(_, allocated) =>
       forallb (value_all_cells (heap_location rs))
         (values_of_static allocated))
    values
  = true ->
  forallb
    (fun '(_, allocated) =>
       forallb (value_all_cells (heap_location rs'))
         (values_of_static allocated))
    values
  = true.
Proof.
  intros Hsame H. apply forallb_forall. intros [s allocated] Hin.
  pose proof (forallb_in _ _ (s, allocated) H Hin) as Hall.
  cbv beta iota in Hall. apply forallb_forall. intros w Hw.
  apply (heap_cells_same rs rs' w); [ | exact (forallb_in _ _ w Hall Hw) ].
  intros l Hl. exact (Hsame s allocated w l Hin Hw Hl).
Qed.

(* A cell: its location and those of its values are listed as they
   were, the heap has the symbols that it had, and the frames from
   that of its mark down are frames that bind what they bound. *)
Lemma cell_deep_same (rs rs' : region_state) (l : location) (c : cell) :
  lookup l (rs_cells rs') = lookup l (rs_cells rs) ->
  (∀ (w : value) (l' : location),
     In w (values_of_cell c) ->
     In l' (locations_of_value w) ->
     lookup l' (rs_cells rs') = lookup l' (rs_cells rs)) ->
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  (∀ (m : mark) (fr' : list frame),
     lookup l (rs_cells rs) = Some m ->
     frames_from m (rs_frames rs') = Some fr' ->
     ∃ fr : list frame,
       frames_from m (rs_frames rs) = Some fr
       ∧ (∀ s : symbol, local_in fr s = true -> local_in fr' s = true)
       ∧ ∀ m' : mark, on_stack m' fr = true -> on_stack m' fr' = true) ->
  cell_deep rs l c = true ->
  cell_deep rs' l c = true.
Proof.
  intros Hl Hsame Hheap Hframes. unfold cell_deep. rewrite Hl.
  destruct (lookup l (rs_cells rs)) as [m | ] eqn:Hm.
  - destruct (frames_from m (rs_frames rs')) as [fr' | ] eqn:Hfr';
      [ | intros _; reflexivity ].
    destruct (Hframes m fr' eq_refl Hfr') as (fr & Hfr & Hlocal & Hstack).
    rewrite Hfr. intros H. apply forallb_forall. intros w Hw.
    pose proof (forallb_in _ _ w H Hw) as Hold. cbv beta in Hold.
    apply andb_true_iff in Hold as [Hsymbols Hcells].
    apply andb_true_iff. split.
    + revert Hsymbols. apply value_all_mono. intros s Hs.
      apply orb_true_iff in Hs as [Hs | Hs].
      * rewrite (Hheap s Hs). reflexivity.
      * rewrite (Hlocal s Hs). apply orb_true_r.
    + exact
        (cells_all_same rs rs' fr fr' w (fun l' Hl' => Hsame w l' Hw Hl')
           Hstack Hcells).
  - intros H. apply forallb_forall. intros w Hw.
    pose proof (forallb_in _ _ w H Hw) as Hold. unfold value_for_ever in *.
    apply andb_true_iff in Hold as [Hsymbols Hcells].
    rewrite (value_all_mono _ _ w Hheap Hsymbols). cbn [andb].
    exact (heap_cells_same rs rs' w (fun l' Hl' => Hsame w l' Hw Hl') Hcells).
Qed.

(* The whole, where the frames, the list of cells, the static
   environment and the heap are as they were. *)
Lemma cells_deep_same (rs rs' : region_state) :
  rs_frames rs' = rs_frames rs ->
  rs_cells rs' = rs_cells rs ->
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  state_heap (rs_state rs') = state_heap (rs_state rs) ->
  cells_deep rs = true ->
  cells_deep rs' = true.
Proof.
  intros Hframes Hcells Hheap Hstatics Hhp. unfold cells_deep.
  rewrite Hframes, Hstatics, Hhp. intros H.
  apply andb_true_iff in H as [H H3]. apply andb_true_iff in H as [H1 H2].
  apply andb_true_iff. split; [ apply andb_true_iff; split | ].
  - apply (frames_deep_cells_same rs rs'); [ | exact H1 ].
    intros s allocated w l _ _ _. rewrite Hcells. reflexivity.
  - apply (statics_cells_same rs rs'); [ | exact H2 ].
    intros s allocated w l _ _ _. rewrite Hcells. reflexivity.
  - apply forallb_forall. intros [l c] Hin.
    apply (cell_deep_same rs rs' l c);
      [ rewrite Hcells; reflexivity
      | intros w l' _ _; rewrite Hcells; reflexivity
      | exact Hheap
      |
      | exact (forallb_in _ _ (l, c) H3 Hin) ].
    intros m fr' _ Hfr'. rewrite Hframes in Hfr'. exists fr'.
    split; [ exact Hfr' | ]. split; intros x Hx; exact Hx.
Qed.

Lemma statics_good_moves (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state)
    (bindings : list (symbol * statically_allocated_value)) :
  store_moves rs rs' ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
     In (s, allocated) bindings -> allocated_values (old rs) allocated) ->
  Forall (fun binding => allocated_values (good cert codes rs) (snd binding))
    bindings ->
  Forall (fun binding => allocated_values (good cert codes rs') (snd binding))
    bindings.
Proof.
  intros Hmoves Hold Hgood. apply Forall_forall. intros [s allocated] Hin.
  cbn [snd].
  apply (allocated_good_moves cert codes rs rs' Hmoves allocated
           (Hold s allocated Hin)).
  exact (proj1 (Forall_forall _ _) Hgood (s, allocated) Hin).
Qed.

Lemma heap_good_moves (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state) (hp : heap) :
  store_moves rs rs' ->
  heap_values_all (old rs) hp ->
  Forall
    (fun binding => Forall (good cert codes rs) (values_of_cell (snd binding)))
    hp ->
  Forall
    (fun binding =>
       Forall (good cert codes rs') (values_of_cell (snd binding)))
    hp.
Proof.
  intros Hmoves Hold Hgood. apply Forall_forall. intros [l c] Hin.
  apply Forall_forall. intros v Hv.
  apply (good_moves cert codes rs rs' Hmoves v). split.
  - exact
      (proj1 (Forall_forall _ _) (proj1 (Forall_forall _ _) Hgood (l, c) Hin)
         v Hv).
  - exact (Hold l c v Hin Hv).
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

(* The heap binds every location that the state names. *)
Lemma state_closed (rs : region_state) :
  region_invariant rs = true ->
  env_values_all (closed rs) (state_vars (rs_state rs))
  ∧ conts_values_all (closed rs) (state_conts (rs_state rs))
  ∧ heap_values_all (closed rs) (state_heap (rs_state rs))
  ∧ ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
      In (s, allocated) (static_values (live_statics rs)) ->
      In w (FlambdaRegionsReadsProofs.allocated_values allocated) ->
      closed rs w.
Proof.
  intros Hinv. apply region_invariant_parts in Hinv as (_ & _ & _ & Hbound).
  destruct
    (proj1 (bound_parts rs) (proj1 (locations_all_bound_iff rs) Hbound))
    as ([Hvars Hconts] & Hframes & Hstatics & _ & Hheap & _).
  split; [ exact Hvars | ]. split; [ exact Hconts | ].
  split; [ exact Hheap | ].
  intros s allocated w Hin Hw. unfold live_statics in Hin.
  cbn [static_values] in Hin. apply in_app_or in Hin as [Hin | Hin].
  - exact (Hframes s allocated w Hin Hw).
  - exact (Hstatics s allocated w Hin Hw).
Qed.

(* What the state holds, it may hold: it names no symbol that is not
   drawn yet, and the heap binds every location that it names. *)
Lemma state_old (rs : region_state) :
  region_invariant rs = true ->
  env_values_all (old rs) (state_vars (rs_state rs))
  ∧ conts_values_all (old rs) (state_conts (rs_state rs))
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
       In (s, allocated) (static_values (live_statics rs)) ->
       allocated_values (old rs) allocated)
  ∧ heap_values_all (old rs) (state_heap (rs_state rs)).
Proof.
  intros Hinv.
  destruct (state_closed rs Hinv)
    as (Hc_vars & Hc_conts & Hc_heap & Hc_statics).
  pose proof (proj1 (region_invariant_parts rs) Hinv)
    as (Hfresh & _ & Hcells & _).
  destruct (state_below rs Hfresh) as (Hb_vars & Hb_conts & Hb_statics).
  pose proof (proj1 (cells_live_parts rs) Hcells) as (_ & _ & _ & Hb_heap & _).
  split; [ exact (env_values_all_and _ _ _ Hb_vars Hc_vars) | ].
  split; [ exact (conts_values_all_and _ _ _ Hb_conts Hc_conts) | ]. split.
  - intros s allocated Hin. apply allocated_values_in. intros w Hw. split.
    + destruct (Hb_statics s allocated Hin) as [_ Hbelow].
      rewrite allocated_below_parts in Hbelow.
      apply andb_true_iff in Hbelow as [Hbelow _].
      exact (forallb_in _ _ w Hbelow Hw).
    + exact (Hc_statics s allocated w Hin Hw).
  - intros l c v Hin Hv.
    exact (conj (Hb_heap l c v Hin Hv) (Hc_heap l c v Hin Hv)).
Qed.

(* ======================================================================== *)
(* A step that leaves the store as it was                                   *)
(* ======================================================================== *)

Lemma same_store_checked (cert : certificate) (codes : list (code_id * code))
    (rs rs' : region_state) :
  rs_frames rs' = rs_frames rs ->
  rs_base rs' = rs_base rs ->
  rs_next rs' = rs_next rs ->
  rs_cells rs' = rs_cells rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  state_heap (rs_state rs') = state_heap (rs_state rs) ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs ->
  entered cert codes rs (rs_state rs') ->
  state_checked cert codes rs'.
Proof.
  destruct rs' as
    [[e' vars' conts' statics' hp' rd'] frames' base' next' counts' cells'].
  cbn [rs_frames rs_base rs_next rs_cells rs_state state_statics state_heap].
  intros -> -> -> -> -> -> Hinv' Hordered'
    (Hinv & _ & Hcode & Hlocal & Hdeep & Hstatics & Hplain & Hpaps & Hcells
     & _)
    (cs' & ms' & Hcheck & Hincl & Hrun & Hconts & Hgood_vars & Hgood_conts).
  destruct (proj2 (good_split cert codes rs) (conj Hplain Hpaps))
    as (_ & _ & Hgood_statics & Hgood_heap).
  match goal with
  | |- state_checked _ _ ?r => set (rs' := r) in *
  end.
  destruct (state_old rs Hinv) as (_ & _ & Hold_statics & Hold_heap).
  destruct (state_old rs' Hinv') as (Hold_vars & Hold_conts & _).
  destruct (state_closed rs Hinv) as (_ & _ & Hclosed_heap & _).
  destruct (state_closed rs' Hinv') as (_ & Hclosed_conts & _).
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - intros s H. exact H.
    - intros s _.
      destruct (lookup s (static_values (live_statics rs))) as [allocated | ]
        eqn:Hs; [ left; exact Hs | intros _; exact Hs ].
    - intros l _ H. exact H. }
  split; [ exact Hinv' | ]. split; [ exact Hordered' | ].
  split; [ exact Hcode | ]. split; [ exact Hlocal | ].
  split; [ exact (frames_deep_mono rs rs' _ (fun s H => H) Hdeep) | ].
  split; [ exact Hstatics | ].
  assert (Hgood' : state_values (good cert codes rs') rs').
  { split;
      [ exact (env_good_moves cert codes rs rs' Hmoves _ Hgood_vars Hold_vars)
      | ].
    split;
      [ exact
          (conts_good_moves cert codes rs rs' Hmoves _ Hgood_conts Hold_conts)
      | ].
    split;
      [ exact
          (statics_good_moves cert codes rs rs' _ Hmoves Hold_statics
             Hgood_statics)
      | ].
    exact (heap_good_moves cert codes rs rs' _ Hmoves Hold_heap Hgood_heap). }
  apply good_split in Hgood' as [Hplain' Hpaps'].
  split; [ exact Hplain' | ]. split; [ exact Hpaps' | ].
  split;
    [ exact
        (cells_deep_same rs rs' eq_refl eq_refl (fun s H => H) eq_refl eq_refl
           Hcells)
    | ].
  exists cs', ms'. split; [ exact Hcheck | ]. split; [ exact Hincl | ].
  split; [ exact Hrun | ].
  assert (Hgrows : store_grows rs rs').
  { constructor.
    - apply Pos.le_refl.
    - intros s H. exact H.
    - intros s H. exact H.
    - intros m _. reflexivity.
    - intros m H. exact H.
    - intros l _. reflexivity.
    - intros l c' w _ Hl Hw. left. exists c'. split; [ exact Hl | exact Hw ]. }
  destruct Hconts as [Hsome Hnone]. split; [ | exact Hnone ].
  intros k ce Hk.
  exact
    (proj1 (grows_conts rs rs' Hgrows cert codes Hclosed_heap) _ ce _
       (Hsome k ce Hk) (resolve_values_all _ k conts' Hclosed_conts)
       (Pos.le_refl _)).
Qed.

Lemma checked_entered (cert : certificate) (codes : list (code_id * code))
    (rs : region_state) :
  state_checked cert codes rs -> entered cert codes rs (rs_state rs).
Proof.
  intros (_ & _ & _ & _ & _ & _ & Hplain & Hpaps & _ & cs & ms & Hcheck
          & Hincl & Hrun & Hconts).
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
  - pose proof (proj1 H) as Hinv.
    apply region_invariant_parts in Hinv as (Hfresh & Hlive & Hcells & Hbound).
    apply region_invariant_parts. split; [ exact Hfresh | ].
    split; [ exact (values_live_count_uses rs Hlive) | ].
    split; [ exact (cells_live_count_uses rs Hcells) | exact Hbound ].
  - exact (proj1 (proj2 H)).
  - exact H.
  - exact (checked_entered cert codes rs H).
Qed.

(* A variable is bound in a state of the check that is in scope. *)
Lemma bind_lifetimes_scoped (cs : check_state) (x : variable)
    (l lc : lifetime) :
  check_state_scoped cs = true ->
  (∀ r : variable, l = Of_region r -> among r (cs_regions cs) = true) ->
  (∀ r : variable, lc = Of_region r -> among r (cs_regions cs) = true) ->
  check_state_scoped (bind_lifetimes x l lc cs) = true.
Proof.
  intros Hscoped Hl Hlc. unfold check_state_scoped in *.
  apply andb_true_iff in Hscoped as [Hs1 Hs2].
  apply andb_true_iff. split; [ exact Hs1 | ].
  cbn [bind_lifetimes cs_env cs_regions cs_contents].
  rewrite forallb_app in Hs2 |- *. apply andb_true_iff in Hs2 as [He Hc].
  cbn [forallb]. rewrite He, Hc, !andb_true_r.
  apply andb_true_iff. split.
  - destruct l as [ | | r | ]; try reflexivity. exact (Hl r eq_refl).
  - destruct lc as [ | | r | ]; try reflexivity. exact (Hlc r eq_refl).
Qed.
