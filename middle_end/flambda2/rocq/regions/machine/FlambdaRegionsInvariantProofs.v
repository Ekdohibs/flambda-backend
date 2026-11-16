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

(* Proofs of the statements of FlambdaRegionsInvariantSpec.v. Nothing
   here needs review. The first part of the invariant, [symbols_fresh],
   is kept by FlambdaRegionsFreshProofs.v, and the fourth,
   [locations_all_bound], by FlambdaRegionsCellsProofs.v; here are the
   second, [values_live], and the third, [cells_live], and what
   follows of the four. The four statements on [no_poison] are proved
   in FlambdaRegionsNoPoisonProofs.v, and are stated again at the end
   of this file. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsInvariantSpec FlambdaRegionsReadsProofs
  FlambdaRegionsStepsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsCellsProofs.
From OxCamlFlambda2 Require FlambdaRegionsNoPoisonProofs.
Import ListNotations.

(* ======================================================================== *)
(* The tests of what a value holds, for a wider set of symbols              *)
(* ======================================================================== *)

Lemma value_holds_mono (ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_holds ok v = true ->
  value_holds ok' v = true.
Proof.
  intros Hok.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; auto.
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    apply andb_true_iff. split.
    + destruct callee; try reflexivity; exact (IHcallee Hcallee).
    + apply forallb_forall. intros arg Hin.
      rewrite Forall_forall in IHargs.
      pose proof (forallb_in _ _ arg Hargs Hin) as Harg.
      cbv beta in Harg.
      destruct arg; try reflexivity; exact (IHargs _ Hin Harg).
Qed.

Lemma value_holds_kept (ok : symbol -> bool) (v : value) :
  value_holds ok v = true -> value_kept ok v = true.
Proof. destruct v; simpl; auto. Qed.

Lemma value_kept_mono (ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_kept ok v = true ->
  value_kept ok' v = true.
Proof.
  intros Hok.
  destruct v;
    first [ exact (value_holds_mono ok ok' _ Hok) | intros _; reflexivity ].
Qed.

Lemma value_placed_holds (near ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, near s = true -> ok' s = true) ->
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_placed near ok v = true ->
  value_holds ok' v = true.
Proof.
  intros Hnear Hok.
  destruct v; first [ exact (Hnear _) | exact (value_holds_mono ok ok' _ Hok) ].
Qed.

Lemma value_placed_mono (near near' ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, near s = true -> near' s = true) ->
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_placed near ok v = true ->
  value_placed near' ok' v = true.
Proof.
  intros Hnear Hok.
  destruct v; first [ exact (Hnear _) | exact (value_holds_mono ok ok' _ Hok) ].
Qed.

Lemma value_placed_kept (near ok : symbol -> bool) (v : value) :
  value_placed near ok v = true -> value_kept ok v = true.
Proof. destruct v; simpl; auto. Qed.

Lemma allocated_holds_mono (ok ok' : symbol -> bool)
    (allocated : statically_allocated_value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  allocated_holds ok allocated = true ->
  allocated_holds ok' allocated = true.
Proof.
  intros Hok. rewrite !allocated_holds_parts. intros H.
  apply andb_true_iff in H as [Hv Hs]. apply andb_true_iff. split.
  - exact (forallb_impl _ _ _ (fun w _ => value_holds_mono ok ok' w Hok) Hv).
  - exact (forallb_impl _ _ _ (fun s _ => Hok s) Hs).
Qed.

Lemma allocated_placed_mono (near near' ok ok' : symbol -> bool)
    (allocated : statically_allocated_value) :
  (∀ s : symbol, near s = true -> near' s = true) ->
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  allocated_placed near ok allocated = true ->
  allocated_placed near' ok' allocated = true.
Proof.
  intros Hnear Hok. rewrite !allocated_placed_parts. intros H.
  apply andb_true_iff in H as [Hv Hs]. apply andb_true_iff. split.
  - exact
      (forallb_impl _ _ _
         (fun w _ => value_placed_mono near near' ok ok' w Hnear Hok) Hv).
  - exact (forallb_impl _ _ _ (fun s _ => Hnear s) Hs).
Qed.

Lemma statics_hold_mono (ok ok' : symbol -> bool)
    (values : list (symbol * statically_allocated_value)) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  forallb (fun '(_, allocated) => allocated_holds ok allocated) values = true ->
  forallb (fun '(_, allocated) => allocated_holds ok' allocated) values = true.
Proof.
  intros Hok. apply forallb_impl. intros [s allocated] _.
  exact (allocated_holds_mono ok ok' allocated Hok).
Qed.

(* A value all of whose symbols are below [base] holds only symbols of
   a set that has every symbol below [base]. *)
Lemma value_below_holds (base : symbol) (ok : symbol -> bool) (v : value) :
  (∀ s : symbol, symbol_below base s = true -> ok s = true) ->
  value_below base v = true ->
  value_holds ok v = true.
Proof.
  intros Hok.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; auto.
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH. exact (IH w Hin (forallb_in _ _ w H Hin)).
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    apply andb_true_iff. split.
    + destruct callee; try reflexivity; exact (IHcallee Hcallee).
    + apply forallb_forall. intros arg Hin.
      rewrite Forall_forall in IHargs.
      pose proof (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)) as Harg.
      destruct arg; try reflexivity; exact Harg.
Qed.

Lemma existsb_false_in {A : Type} (f : A -> bool) (l : list A) (a : A) :
  existsb f l = false -> In a l -> f a = false.
Proof.
  intros Hnone Hin. destruct (f a) eqn:Hfa; [ | reflexivity ].
  assert (Hsome : existsb f l = true)
    by (apply existsb_exists; exists a; split; assumption).
  rewrite Hnone in Hsome. discriminate.
Qed.

(* ======================================================================== *)
(* The symbols of the heap, and the frames                                  *)
(* ======================================================================== *)

Lemma heap_symbol_extends (rs rs' : region_state)
    (new : list (symbol * statically_allocated_value)) :
  rs_base rs' = rs_base rs ->
  static_values (state_statics (rs_state rs'))
  = new ++ static_values (state_statics (rs_state rs)) ->
  ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true.
Proof.
  intros Hbase Hvalues s. unfold heap_symbol, drawn.
  rewrite Hbase, Hvalues, is_some_lookup_app.
  destruct (negb (Pos.leb (rs_base rs) s)); simpl; [ reflexivity | ].
  intros ->. apply orb_true_r.
Qed.

Lemma heap_symbol_same (rs rs' : region_state) :
  rs_base rs' = rs_base rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true.
Proof.
  intros Hbase Hstatics.
  apply (heap_symbol_extends rs rs' []); [ exact Hbase | ].
  rewrite Hstatics. reflexivity.
Qed.

(* A symbol that may be evaluated and is in no frame is of the heap. *)
Lemma usable_not_local_heap (rs : region_state) (s : symbol) :
  value_usable rs (Value_symbol s) = true ->
  value_local rs (Value_symbol s) = false ->
  heap_symbol rs s = true.
Proof.
  unfold value_usable, value_local, heap_symbol, live_statics.
  cbn [static_values]. rewrite is_some_lookup_app.
  intros Hu Hl. rewrite Hl in Hu. exact Hu.
Qed.

(* A symbol that may be evaluated is of the heap or in a frame. *)
Lemma usable_near (rs : region_state) (s : symbol) :
  value_usable rs (Value_symbol s) = true ->
  heap_symbol rs s || local_in (rs_frames rs) s = true.
Proof.
  unfold value_usable, heap_symbol, local_in, live_statics.
  cbn [static_values]. rewrite is_some_lookup_app.
  destruct (negb (drawn rs s)),
    (is_some (lookup s (live_locals (rs_frames rs)))),
    (is_some (lookup s (static_values (state_statics (rs_state rs)))));
    simpl; congruence.
Qed.

Lemma heap_live (rs : region_state) (s : symbol) :
  heap_symbol rs s = true -> live_symbol rs s = true.
Proof.
  unfold live_symbol, value_usable, heap_symbol, live_statics.
  cbn [static_values]. rewrite is_some_lookup_app.
  destruct (negb (drawn rs s)),
    (is_some (lookup s (live_locals (rs_frames rs)))),
    (is_some (lookup s (static_values (state_statics (rs_state rs)))));
    simpl; congruence.
Qed.

Lemma near_live (rs : region_state) (s : symbol) :
  heap_symbol rs s || local_in (rs_frames rs) s = true ->
  live_symbol rs s = true.
Proof.
  unfold live_symbol, value_usable, heap_symbol, local_in, live_statics.
  cbn [static_values]. rewrite is_some_lookup_app.
  destruct (negb (drawn rs s)),
    (is_some (lookup s (live_locals (rs_frames rs)))),
    (is_some (lookup s (static_values (state_statics (rs_state rs)))));
    simpl; congruence.
Qed.

Lemma frames_hold_mono (rs rs' : region_state) (frames : list frame) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  frames_hold rs frames = true ->
  frames_hold rs' frames = true.
Proof.
  intros Hheap. induction frames as [ | f below IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hf Hbelow].
  apply andb_true_iff. split; [ | exact (IH Hbelow) ].
  revert Hf. apply forallb_impl. intros [s allocated] _.
  apply allocated_placed_mono; [ | exact Hheap ].
  intros s0 H0. apply orb_true_iff in H0 as [H0 | H0];
    [ rewrite (Hheap _ H0); reflexivity | rewrite H0; apply orb_true_r ].
Qed.

Lemma frames_hold_app (rs : region_state) (above rest : list frame) :
  frames_hold rs (above ++ rest) = true -> frames_hold rs rest = true.
Proof.
  induction above as [ | f above IH]; simpl; [ auto | ].
  intros H. apply andb_true_iff in H as [_ H]. exact (IH H).
Qed.

Lemma local_in_cons (f : frame) (below : list frame) (s : symbol) :
  local_in (f :: below) s
  = is_some (lookup s (frame_locals f)) || local_in below s.
Proof. unfold local_in, live_locals. simpl. apply is_some_lookup_app. Qed.

(* What a frame binds a symbol to. *)
Lemma frames_hold_in (rs : region_state) (frames : list frame) (s : symbol)
    (allocated : statically_allocated_value) :
  frames_hold rs frames = true ->
  In (s, allocated) (live_locals frames) ->
  allocated_placed (fun s' => heap_symbol rs s' || local_in frames s')
    (heap_symbol rs) allocated
  = true.
Proof.
  induction frames as [ | f below IH]; simpl; [ intros _ [] | ].
  intros H Hin. apply andb_true_iff in H as [Hf Hbelow].
  apply in_app_iff in Hin as [Hin | Hin].
  - exact (forallb_in _ _ (s, allocated) Hf Hin).
  - apply
      (allocated_placed_mono
         (fun s' => heap_symbol rs s' || local_in below s') _
         (heap_symbol rs) _ allocated);
      [ | auto | exact (IH Hbelow Hin) ].
    intros s' H'. rewrite local_in_cons.
    apply orb_true_iff in H' as [H' | H']; rewrite H';
      rewrite ?orb_true_r; reflexivity.
Qed.

Lemma local_in_app (above frames : list frame) (s : symbol) :
  local_in (above ++ frames) s = local_in above s || local_in frames s.
Proof. unfold local_in. rewrite live_locals_app. apply is_some_lookup_app. Qed.

(* What a cell that is not dead holds: of symbols and of locations,
   what a variable may hold; and it may be evaluated. *)
Lemma live_cell_holds (rs : region_state) (l : location) (c : cell)
    (w : value) :
  cell_holds rs l c = true ->
  cell_dead rs l = false ->
  In w (values_of_cell c) ->
  value_kept (heap_symbol rs) w = true
  ∧ value_kept_cells (heap_location rs) w = true
  ∧ value_usable rs w = true.
Proof.
  unfold cell_holds, cell_dead.
  destruct (lookup l (rs_cells rs)) as [m | ].
  - intros Hall Hdead Hin. apply negb_false_iff in Hdead.
    destruct (on_stack_frames_from m _ Hdead) as (fr & Hfr).
    rewrite Hfr in Hall.
    destruct (frames_from_app m _ fr Hfr) as (above & Hframes & _ & _).
    pose proof (forallb_in _ _ w Hall Hin) as Hw. unfold value_placed_in in Hw.
    apply andb_true_iff in Hw as [Hs Hc].
    split; [ exact (value_placed_kept _ _ w Hs) | ].
    split; [ exact (value_placed_cells_kept _ _ w Hc) | ].
    destruct w; try reflexivity.
    + apply near_live. cbn [value_placed] in Hs.
      rewrite Hframes, local_in_app.
      apply orb_true_iff in Hs as [Hs | Hs]; rewrite Hs;
        rewrite ?orb_true_r; reflexivity.
    + cbn [value_placed_cells] in Hc.
      apply orb_true_iff in Hc as [Hc | Hc].
      * exact (heap_location_usable rs _ Hc).
      * exact (cell_in_usable rs above fr _ Hframes Hc).
  - intros Hall _ Hin. pose proof (forallb_in _ _ w Hall Hin) as Hw.
    unfold value_holds_heap in Hw.
    apply andb_true_iff in Hw as [Hs Hc].
    split; [ exact (value_holds_kept _ w Hs) | ].
    split; [ exact (value_holds_cells_kept _ w Hc) | ].
    destruct w; try reflexivity.
    + exact (heap_live rs _ Hs).
    + exact (heap_location_usable rs _ Hc).
Qed.

(* ======================================================================== *)
(* The parts of [values_live]                                               *)
(* ======================================================================== *)

Lemma values_live_parts (rs : region_state) :
  values_live rs = true
  <-> forallb
        (fun '(_, allocated) => allocated_holds (heap_symbol rs) allocated)
        (static_values (state_statics (rs_state rs)))
      = true
      ∧ frames_hold rs (rs_frames rs) = true
      ∧ forallb (fun '(_, v) => value_kept (heap_symbol rs) v)
          (state_vars (rs_state rs))
        = true
      ∧ forallb (fun '(_, c) => cont_kept (heap_symbol rs) c)
          (state_conts (rs_state rs))
        = true.
Proof.
  unfold values_live. cbv zeta. split.
  - intros H. apply andb_true_iff in H as [H L4].
    apply andb_true_iff in H as [H L3].
    apply andb_true_iff in H as [L1 L2]. auto.
  - intros (L1 & L2 & L3 & L4). rewrite L1, L2, L3, L4. reflexivity.
Qed.

(* The test of the variables and of the continuations, as
   FlambdaRegionsStepsProofs.vals_ok. *)
Definition kept (rs : region_state) (v : value) : Prop :=
  value_kept (heap_symbol rs) v = true.

Lemma kept_vals (rs : region_state) (s : state) :
  forallb (fun '(_, v) => value_kept (heap_symbol rs) v) (state_vars s) = true
  ∧ forallb (fun '(_, c) => cont_kept (heap_symbol rs) c) (state_conts s)
    = true
  <-> vals_ok (kept rs) s.
Proof.
  unfold vals_ok, kept, env_values_all.
  destruct (forallb_pairs_Forall (value_kept (heap_symbol rs)) (state_vars s))
    as [Hv1 Hv2].
  destruct (conts_kept_split (heap_symbol rs) (state_conts s)) as [Hc1 Hc2].
  split; intros [Hv Hc]; split; auto.
Qed.

Lemma kept_partial (rs : region_state) (callee : value) (args : list value)
    (args_arity : arity) :
  kept rs (Value_partial_application callee args args_arity)
  <-> kept rs callee ∧ Forall (kept rs) args.
Proof.
  unfold kept. simpl. split.
  - intros H. apply andb_true_iff in H as [Hcallee Hargs].
    split; [ exact Hcallee | ].
    exact (proj1 (forallb_Forall _ args) Hargs).
  - intros [Hcallee Hargs]. apply andb_true_iff.
    split; [ exact Hcallee | ].
    exact (proj2 (forallb_Forall (value_kept (heap_symbol rs)) args) Hargs).
Qed.

Lemma vals_ok_mono (P P' : value -> Prop) (s : state) :
  (∀ v : value, P v -> P' v) -> vals_ok P s -> vals_ok P' s.
Proof.
  intros Himpl [Hv Hc]. split.
  - exact (Forall_impl _ (fun binding => Himpl (snd binding)) Hv).
  - exact (conts_values_all_impl P P' _ Himpl Hc).
Qed.

(* A state that has the test, from one whose symbols of the heap are
   among its own: what the static environment binds, the frames, and
   the variables and the continuations, these by the test of the state
   before. *)
Lemma values_live_build (rs rs' : region_state) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  forallb
    (fun '(_, allocated) => allocated_holds (heap_symbol rs') allocated)
    (static_values (state_statics (rs_state rs')))
  = true ->
  frames_hold rs' (rs_frames rs') = true ->
  vals_ok (kept rs) (rs_state rs') ->
  values_live rs' = true.
Proof.
  intros Hheap L1 L2 Hvals. apply values_live_parts.
  refine (conj L1 (conj L2 _)). apply kept_vals.
  exact (vals_ok_mono (kept rs) (kept rs') _
           (fun v => value_kept_mono _ _ v Hheap) Hvals).
Qed.

(* A step that leaves the static environment and the frames. *)
Lemma values_live_transfer (rs rs' : region_state) :
  rs_base rs' = rs_base rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  values_live rs = true ->
  vals_ok (kept rs) (rs_state rs') ->
  values_live rs' = true.
Proof.
  intros Hbase Hstatics Hframes Hlive Hvals.
  pose proof (heap_symbol_same rs rs' Hbase Hstatics) as Hheap.
  apply values_live_parts in Hlive as (L1 & L2 & _ & _).
  apply (values_live_build rs rs' Hheap); [ | | exact Hvals ].
  - rewrite Hstatics. exact (statics_hold_mono _ _ _ Hheap L1).
  - rewrite Hframes. exact (frames_hold_mono rs rs' _ Hheap L2).
Qed.

Lemma values_live_count_uses (rs : region_state) :
  values_live rs = true -> values_live (count_uses rs) = true.
Proof.
  intros Hlive.
  apply (values_live_transfer rs (count_uses rs)); try reflexivity;
    [ exact Hlive | ].
  apply values_live_parts in Hlive as (_ & _ & L3 & L4).
  exact (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)).
Qed.

(* ======================================================================== *)
(* The values of the simples                                                *)
(* ======================================================================== *)

Lemma eval_simple_kept (ok : symbol -> bool) (vars : env) (sim : simple)
    (v : value) :
  forallb (fun '(_, v) => value_kept ok v) vars = true ->
  eval_simple vars sim = Some v ->
  value_kept ok v = true.
Proof.
  intros Hvars. destruct sim as [x | sym | c]; simpl.
  - exact (lookup_forallb (value_kept ok) x v vars Hvars).
  - intros [= <-]. reflexivity.
  - intros [= <-]. reflexivity.
Qed.

(* The value of a simple that may be evaluated and is in no frame holds
   symbols of the heap only. *)
Lemma simple_holds_heap (rs : region_state) (sim : simple) (v : value) :
  forallb (fun '(_, v) => value_kept (heap_symbol rs) v)
    (state_vars (rs_state rs))
  = true ->
  simple_usable rs sim = true ->
  simple_local rs sim = false ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_holds (heap_symbol rs) v = true.
Proof.
  intros Hvars Hu Hl Hv. unfold simple_usable, simple_local in Hu, Hl.
  rewrite Hv in Hu, Hl.
  pose proof (eval_simple_kept _ _ sim v Hvars Hv) as Hk.
  destruct v; try exact Hk.
  exact (usable_not_local_heap rs s Hu Hl).
Qed.

(* The value of a simple that may be evaluated is what a frame may bind
   a symbol to hold. *)
Lemma simple_placed (rs : region_state) (sim : simple) (v : value) :
  forallb (fun '(_, v) => value_kept (heap_symbol rs) v)
    (state_vars (rs_state rs))
  = true ->
  simple_usable rs sim = true ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_placed (fun s => heap_symbol rs s || local_in (rs_frames rs) s)
    (heap_symbol rs) v
  = true.
Proof.
  intros Hvars Hu Hv. unfold simple_usable in Hu. rewrite Hv in Hu.
  pose proof (eval_simple_kept _ _ sim v Hvars Hv) as Hk.
  destruct v; try exact Hk.
  exact (usable_near rs s Hu).
Qed.

(* ======================================================================== *)
(* The value of a named expression                                          *)
(* ======================================================================== *)

Lemma kept_parts (ok : symbol -> bool) (v w : value) :
  value_kept ok v = true -> In w (value_parts v) -> value_kept ok w = true.
Proof.
  intros Hv Hin.
  destruct v; simpl in Hin; try contradiction;
    apply value_holds_kept;
    exact (forallb_in _ _ w (value_holds_parts ok _ Hv) Hin).
Qed.

(* What a symbol is bound to, in a frame or in the static environment
   of the state, holds directly values that a variable may hold. *)
Lemma kept_static (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  values_live rs = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  ∀ w : value,
    In w (allocated_values allocated) -> value_kept (heap_symbol rs) w = true.
Proof.
  intros Hlive Hlookup w Hin.
  apply values_live_parts in Hlive as (L1 & L2 & _ & _).
  unfold live_statics in Hlookup. cbn [static_values] in Hlookup.
  rewrite lookup_app in Hlookup.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated' | ]
    eqn:Hlocals.
  - injection Hlookup as ->.
    pose proof (frames_hold_in rs _ s allocated L2 (lookup_in _ _ _ Hlocals))
      as Hp.
    rewrite allocated_placed_parts in Hp. apply andb_true_iff in Hp as [Hp _].
    exact (value_placed_kept _ _ w (forallb_in _ _ w Hp Hin)).
  - pose proof
      (lookup_forallb (allocated_holds (heap_symbol rs)) s allocated _ L1
         Hlookup)
      as Hh.
    rewrite allocated_holds_parts in Hh. apply andb_true_iff in Hh as [Hh _].
    exact (value_holds_kept _ w (forallb_in _ _ w Hh Hin)).
Qed.

Lemma kept_fields (rs : region_state) :
  values_live rs = true -> P_fields (kept rs) (live_statics rs).
Proof.
  intros Hlive.
  apply (reads_fields (kept rs) (live_statics rs)).
  - exact (kept_parts (heap_symbol rs)).
  - intros s allocated _ Hlookup.
    split; [ exact (kept_static rs s allocated Hlive Hlookup) | ].
    intros s' _. reflexivity.
Qed.

(* The value of a [Let] that is not a local allocation: the simples may
   be evaluated, and if it allocates none of them is in a frame. *)
Lemma eval_named_kept (rs : region_state) (d : named) (v : value) :
  values_live rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  is_allocation d && existsb (simple_local rs) (simples_of_named d) = false ->
  eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
  value_kept (heap_symbol rs) v = true.
Proof.
  intros Hlive Husable Hheap Heval.
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & L3 & _).
  destruct (is_allocation d) eqn:His.
  - simpl in Hheap.
    destruct (eval_named_makes _ _ d v His Heval) as [Hmade Hparts].
    apply value_holds_kept, value_holds_made; [ exact Hmade | ].
    apply forallb_forall. intros w Hin.
    destruct (Hparts w Hin) as (sim & Hsim & Hw).
    exact
      (simple_holds_heap rs sim w L3 (forallb_in _ _ sim Husable Hsim)
         (existsb_false_in _ _ sim Hheap Hsim) Hw).
  - apply
      (eval_named_reads (kept rs) (live_statics rs)
         (fun c _ => eq_refl) (kept_parts (heap_symbol rs))
         (fun slot slot' decls slots H => H)
         (fun s allocated _ Hlookup =>
            conj (kept_static rs s allocated Hlive Hlookup)
              (fun s' _ => eq_refl))
         (state_vars (rs_state rs)) d v His);
      [ | exact Heval ].
    intros sim w _. exact (eval_simple_kept _ _ sim w L3).
Qed.

(* The same of what a [Let] runs on the heap: a primitive of the heap
   gives a location or the unit value, and a read of a cell gives what
   the cell holds, which is what a variable may hold as the cell is
   not dead. *)
Lemma exec_named_kept (rs : region_state) (d : named) (v : value)
    (hp' : heap) (rd' : read_record) :
  values_live rs = true ->
  (∀ (l : location) (c : cell),
    In (l, c) (state_heap (rs_state rs)) -> cell_holds rs l c = true) ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  is_allocation d && existsb (simple_local rs) (simples_of_named d) = false ->
  exec_named (live_statics rs) (state_heap (rs_state rs))
    (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
  = Some (v, hp', rd') ->
  value_kept (heap_symbol rs) v = true.
Proof.
  intros Hlive Hcells Husable Hheap Hexec.
  destruct (is_allocation d) eqn:His.
  - destruct (exec_named_makes _ _ _ _ _ _ d v His Hexec) as [Heval _].
    apply (eval_named_kept rs d v Hlive Husable); [ | exact Heval ].
    rewrite His. exact Hheap.
  - destruct (named_changes_heap d) eqn:Hchanges.
    + destruct (exec_named_heap_value _ _ _ _ _ _ d v Hchanges Hexec)
        as [-> | ->];
        reflexivity.
    + pose proof Hlive as HL.
      apply values_live_parts in HL as (_ & _ & L3 & _).
      apply
        (exec_named_reads (kept rs) (live_statics rs)
           (fun c _ => eq_refl) (kept_parts (heap_symbol rs))
           (fun slot slot' decls slots H => H)
           (fun s allocated _ Hlookup =>
              conj (kept_static rs s allocated Hlive Hlookup)
                (fun s' _ => eq_refl))
           (state_heap (rs_state rs)) hp' (state_reads (rs_state rs)) rd'
           (state_vars (rs_state rs)) d v His Hchanges);
        [ | | exact Hexec ].
      * intros sim w _. exact (eval_simple_kept _ _ sim w L3).
      * intros sim l c w Hsim Heval Hlookup Hw.
        refine
          (proj1
             (live_cell_holds rs l c w (Hcells l c (lookup_in _ _ _ Hlookup))
                _ Hw)).
        pose proof (forallb_in _ _ sim Husable Hsim) as Hu.
        unfold simple_usable in Hu. rewrite Heval in Hu.
        apply negb_true_iff in Hu. exact Hu.
Qed.

(* ======================================================================== *)
(* A step keeps [values_live]                                               *)
(* ======================================================================== *)

(* What is shown of a step: the state after has the test; and the
   values of an exit are each what a variable may hold, in the state
   that the run left by. *)
Definition live_result (result : region_step_result) : Prop :=
  match result with
  | Region_step rs' => values_live rs' = true
  | Region_exit _ vs rs' => values_live rs' = true ∧ Forall (kept rs') vs
  | Region_undefined _ => True
  end.

Lemma plain_step_live (rs : region_state) :
  values_live rs = true ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value) (hp' : heap) (rd' : read_record),
    state_expr (rs_state rs) = Let x kind d body ->
    exec_named (live_statics rs) (state_heap (rs_state rs))
      (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
    = Some (v, hp', rd') ->
    kept rs v) ->
  live_result (plain_step rs).
Proof.
  intros Hlive Hlet.
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & L3 & L4).
  pose proof
    (plain_step_values (kept rs) (kept_partial rs) rs (kept_fields rs Hlive)
       (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) Hlet
       (fun sim v _ => eval_simple_kept _ _ sim v L3))
    as Hvals.
  destruct (plain_step rs) as [rs' | k vs rs' | fault] eqn:Hstep;
    unfold region_result_vals_ok in Hvals; unfold live_result.
  - destruct (plain_step_step rs rs' Hstep) as (s' & _ & ->).
    apply (values_live_transfer rs); try reflexivity; assumption.
  - rewrite (plain_step_exit rs rs' k vs Hstep).
    exact (conj Hlive Hvals).
  - exact I.
Qed.

Lemma region_apply_cont_live (rs : region_state) (ac : apply_cont) :
  values_live rs = true ->
  state_expr (rs_state rs) = Apply_cont ac ->
  live_result (region_apply_cont rs ac).
Proof.
  intros Hlive He.
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & L3 & L4).
  pose proof
    (region_apply_cont_values (kept rs) (kept_partial rs) rs ac He
       (kept_fields rs Hlive)
       (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4))
       (fun sim v _ => eval_simple_kept _ _ sim v L3))
    as Hvals.
  destruct (region_apply_cont rs ac) as [rs' | k vs rs' | fault] eqn:Hstep;
    unfold region_result_vals_ok in Hvals; unfold live_result.
  - destruct (region_apply_cont_step rs rs' ac Hstep)
      as (Hframes & Hbase & _ & Hstatics).
    exact (values_live_transfer rs rs' Hbase Hstatics Hframes Hlive Hvals).
  - rewrite (region_apply_cont_exit rs rs' ac k vs Hstep).
    exact (conj Hlive Hvals).
  - exact I.
Qed.

Lemma begin_region_live (rs : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  values_live rs = true ->
  live_result (begin_region rs x kind mark_kind body).
Proof.
  intros Hlive. unfold begin_region.
  destruct (kind_eqb kind Region); [ | exact I ].
  unfold live_result.
  apply values_live_parts in Hlive as (L1 & L2 & L3 & L4).
  apply (values_live_build rs); unfold bind_let;
    cbn [rs_state rs_frames rs_base state_statics state_vars state_conts].
  - intros s H. exact H.
  - exact L1.
  - simpl. revert L2. apply frames_hold_mono. intros s H. exact H.
  - destruct (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) as [Hv Hc].
    split; [ apply Forall_cons; [ reflexivity | exact Hv ] | exact Hc ].
Qed.

Lemma end_region_live (rs : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  values_live rs = true ->
  live_result (end_region rs x kind ended_by region body).
Proof.
  intros Hlive. unfold end_region. cbv zeta.
  destruct (kind_eqb kind Value); [ | exact I ].
  destruct (eval_simple (state_vars (rs_state rs)) region) as [v | ];
    [ | exact I ].
  destruct v as [ | | | m | | | | | ]; try exact I.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | exact I ].
  destruct (cut_app _ _ _ _ Hcut) as (f & Hframes).
  unfold live_result.
  apply values_live_parts in Hlive as (L1 & L2 & L3 & L4).
  apply (values_live_build rs); unfold bind_let;
    cbn [rs_state rs_frames rs_base state_statics state_vars state_conts].
  - intros s H. exact H.
  - exact L1.
  - rewrite Hframes in L2. apply frames_hold_app in L2. simpl in L2.
    apply andb_true_iff in L2 as [_ L2].
    revert L2. apply frames_hold_mono. intros s H. exact H.
  - destruct (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) as [Hv Hc].
    split; [ apply Forall_cons; [ reflexivity | exact Hv ] | exact Hc ].
Qed.

Lemma local_allocation_live (rs : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  is_cell_allocation d && kind_eqb kind Value = false ->
  values_live rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  live_result (local_allocation rs x y kind d region body).
Proof.
  intros Hnot Hlive Husable.
  destruct (local_allocation rs x y kind d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep; unfold live_result.
  2: destruct
       (local_allocation_no_exit rs x y kind d region body k vs rs' Hstep).
  2: exact I.
  destruct (local_allocation_step rs rs' x y kind d region body Hnot Hstep)
    as (v & sym & bindings & His & Heval & Hlocalise & _ & Hvars & Hconts
        & Hbase & _ & _ & Hframes).
  pose proof Hlive as HL. apply values_live_parts in HL as (L1 & L2 & L3 & L4).
  destruct (eval_named_makes _ _ d v His Heval) as [Hmade Hparts].
  destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hbound).
  assert (Hvals : vals_ok (kept rs) (rs_state rs')).
  { destruct (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) as [Hv Hc].
    split; [ rewrite Hvars | rewrite Hconts; exact Hc ].
    apply Forall_cons; [ reflexivity | ].
    apply Forall_cons; [ reflexivity | exact Hv ]. }
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames] eqn:Hrs_frames.
  - (* No frame: the bindings go to the static environment. *)
    destruct Hframes as [Hframes' Hvalues'].
    pose proof (heap_symbol_extends rs rs' bindings Hbase Hvalues') as Hheap.
    apply (values_live_build rs rs' Hheap); [ | | exact Hvals ].
    + rewrite Hvalues', forallb_app. apply andb_true_iff. split;
        [ | exact (statics_hold_mono _ _ _ Hheap L1) ].
      apply forallb_forall. intros [s allocated] Hin.
      destruct (Hbound s allocated Hin) as (_ & _ & Hvalues & Hsymbols).
      rewrite allocated_holds_parts, Hvalues. apply andb_true_iff. split.
      * apply forallb_forall. intros w Hw.
        destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
        apply (value_holds_mono (heap_symbol rs) _ w Hheap).
        exact
          (simple_holds_heap rs sim w L3 (forallb_in _ _ sim Husable Hsim)
             (simple_local_no_frames rs sim Hrs_frames) Hsim_w).
      * apply forallb_forall. intros s' Hs'.
        destruct (Hsymbols s' Hs') as (allocated' & Hin').
        unfold heap_symbol. rewrite Hvalues', is_some_lookup_app,
          (in_is_some_lookup s' allocated' bindings Hin').
        rewrite orb_true_l. apply orb_true_r.
    + rewrite Hframes'. reflexivity.
  - (* The newest frame takes the bindings. *)
    destruct Hframes as [Hframes' Hstatics'].
    pose proof (heap_symbol_same rs rs' Hbase Hstatics') as Hheap.
    apply (values_live_build rs rs' Hheap); [ | | exact Hvals ].
    + rewrite Hstatics'. exact (statics_hold_mono _ _ _ Hheap L1).
    + rewrite Hframes'. simpl in L2 |- *.
      apply andb_true_iff in L2 as [Lf Lbelow].
      apply andb_true_iff. split;
        [ | exact (frames_hold_mono rs rs' _ Hheap Lbelow) ].
      assert (Hnear :
        ∀ s : symbol,
          heap_symbol rs s
          || local_in (Frame m mark_kind locals :: frames) s
          = true ->
          heap_symbol rs' s
          || local_in (Frame m mark_kind (bindings ++ locals) :: frames) s
          = true).
      { intros s H. apply orb_true_iff in H as [H | H];
          [ rewrite (Hheap s H); reflexivity | ].
        rewrite local_in_cons in H |- *. cbn [frame_locals] in H |- *.
        rewrite is_some_lookup_app.
        apply orb_true_iff in H as [H | H]; rewrite H;
          rewrite ?orb_true_r; reflexivity. }
      rewrite forallb_app. apply andb_true_iff. split.
      * apply forallb_forall. intros [s allocated] Hin.
        destruct (Hbound s allocated Hin) as (_ & _ & Hvalues & Hsymbols).
        rewrite allocated_placed_parts, Hvalues. apply andb_true_iff. split.
        -- apply forallb_forall. intros w Hw.
           destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
           pose proof
             (simple_placed rs sim w L3 (forallb_in _ _ sim Husable Hsim)
                Hsim_w)
             as Hplaced.
           rewrite Hrs_frames in Hplaced.
           exact (value_placed_mono _ _ _ _ w Hnear Hheap Hplaced).
        -- apply forallb_forall. intros s' Hs'.
           destruct (Hsymbols s' Hs') as (allocated' & Hin').
           rewrite local_in_cons. cbn [frame_locals].
           rewrite is_some_lookup_app,
             (in_is_some_lookup s' allocated' bindings Hin').
           rewrite orb_true_l. apply orb_true_r.
      * revert Lf. apply forallb_impl. intros [s allocated] _.
        exact (allocated_placed_mono _ _ _ _ allocated Hnear Hheap).
Qed.

Lemma local_cell_allocation_live (rs : region_state) (x y : variable)
    (d : named) (region : option simple) (body : expr) :
  values_live rs = true ->
  live_result (local_cell_allocation rs x y d region body).
Proof.
  intros Hlive.
  destruct (local_cell_allocation rs x y d region body)
    as [rs' | k vs rs' | fault] eqn:Hstep; unfold live_result.
  2: destruct
       (local_cell_allocation_no_exit rs x y d region body k vs rs' Hstep).
  2: exact I.
  destruct (local_cell_allocation_step rs rs' x y d region body Hstep)
    as (l & hp & rd & _ & _ & Hstate & Hframes & Hbase & _).
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & L3 & L4).
  apply (values_live_transfer rs rs' Hbase);
    [ rewrite Hstate; reflexivity | exact Hframes | exact Hlive | ].
  destruct (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) as [Hv Hc].
  rewrite Hstate. split; cbn [state_vars state_conts]; [ | exact Hc ].
  apply Forall_cons; [ reflexivity | ].
  apply Forall_cons; [ reflexivity | exact Hv ].
Qed.

Lemma symbols_fresh_expr (rs : region_state) :
  symbols_fresh rs = true ->
  expr_below (rs_base rs) (state_expr (rs_state rs)) = true.
Proof.
  unfold symbols_fresh. cbv zeta. intros H.
  do 7 (apply andb_true_iff in H as [H _]).
  apply andb_true_iff in H as [_ H]. exact H.
Qed.

Lemma below_base_heap (rs : region_state) (s : symbol) :
  symbol_below (rs_base rs) s = true -> heap_symbol rs s = true.
Proof.
  unfold symbol_below, heap_symbol, drawn. intros H.
  apply Pos.ltb_lt in H.
  destruct (Pos.leb (rs_base rs) s) eqn:Hle; [ | reflexivity ].
  apply Pos.leb_le in Hle. lia.
Qed.

Lemma static_step_live (rs : region_state) (bindings : list static_binding)
    (body : expr) :
  values_live rs = true ->
  expr_below (rs_base rs) (state_expr (rs_state rs)) = true ->
  state_expr (rs_state rs) = Let_static bindings body ->
  forallb (simple_usable rs) (flat_map simples_of_binding bindings) = true ->
  existsb (simple_local rs) (flat_map simples_of_binding bindings) = false ->
  live_result (static_step rs).
Proof.
  intros Hlive Hbelow He Husable Hlocal.
  destruct (static_step rs) as [rs' | k vs rs' | fault] eqn:Hstep;
    unfold live_result.
  2: { revert Hstep. unfold static_step.
       destruct (rs_state rs) as [e vars conts statics hp rd]. simpl in He.
       subst e. rewrite step_let_static.
       destruct (eval_static_bindings vars bindings statics); discriminate. }
  2: exact I.
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & _ & Hvalues & _ & Hnew).
  pose proof Hlive as HL. apply values_live_parts in HL as (L1 & L2 & L3 & L4).
  rewrite He in Hbelow. simpl in Hbelow.
  apply andb_true_iff in Hbelow as [Hbindings _].
  assert (Hheap :
    ∀ s : symbol,
      heap_symbol rs s = true ->
      heap_symbol
        (Region_state
           (State body (state_vars (rs_state rs)) (state_conts (rs_state rs))
              statics' (state_heap (rs_state rs)) (state_reads (rs_state rs)))
           (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs)
           (rs_cells rs))
        s
      = true)
    by (apply (heap_symbol_extends rs _ new_values); [ reflexivity | ];
        exact Hvalues).
  apply (values_live_build rs _ Hheap);
    cbn [rs_state rs_frames state_statics state_vars state_conts].
  - rewrite Hvalues, forallb_app. apply andb_true_iff. split;
      [ | exact (statics_hold_mono _ _ _ Hheap L1) ].
    apply forallb_forall. intros [s allocated] Hin.
    destruct (Hnew s allocated Hin) as (const & Hconst_in & Hconst).
    destruct (eval_static_const_makes _ const allocated Hconst)
      as [Hparts Hsymbols].
    pose proof (forallb_in _ _ _ Hbindings Hconst_in) as Hbinding.
    simpl in Hbinding. apply andb_true_iff in Hbinding as [_ Hconst_below].
    unfold static_const_below in Hconst_below.
    apply andb_true_iff in Hconst_below as [_ Hset_below].
    rewrite allocated_holds_parts. apply andb_true_iff. split.
    + apply forallb_forall. intros w Hw.
      destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
      assert (Hsim_in : In sim (flat_map simples_of_binding bindings))
        by (apply in_flat_map; exists (Bind_symbol s const);
            split; [ exact Hconst_in | exact Hsim ]).
      apply (value_holds_mono (heap_symbol rs) _ w Hheap).
      exact
        (simple_holds_heap rs sim w L3 (forallb_in _ _ sim Husable Hsim_in)
           (existsb_false_in _ _ sim Hlocal Hsim_in) Hsim_w).
    + rewrite Hsymbols.
      destruct const
        as [code value_slots | slot closure_symbols function_decls value_slots
           | tag fields | contents | bytes | elements];
        try reflexivity.
      rewrite forallb_pairs in Hset_below.
      revert Hset_below. apply forallb_impl. intros s' _ Hs'.
      exact (Hheap s' (below_base_heap rs s' Hs')).
  - exact (frames_hold_mono rs _ _ Hheap L2).
  - exact (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)).
Qed.

Theorem region_step_live (rs : region_state) :
  symbols_fresh rs = true ->
  values_live rs = true ->
  (∀ (l : location) (c : cell),
    In (l, c) (state_heap (rs_state rs)) -> cell_holds rs l c = true) ->
  live_result (region_step rs).
Proof.
  intros Hfresh Hlive Hcells.
  pose proof (values_live_count_uses rs Hlive) as Hlive1.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  apply region_step_cases.
  - intros fault. exact I.
  - intros x kind d mark_kind body _.
    exact (begin_region_live _ _ _ _ _ Hlive1).
  - intros x kind d ended_by region body _.
    exact (end_region_live _ _ _ _ _ _ Hlive1).
  - intros x kind d body y region body' _ _ Husable Hnot.
    exact (local_allocation_live _ _ _ _ _ _ _ Hnot Hlive1 Husable).
  - intros x kind d body y region body' _ _ _ _.
    exact (local_cell_allocation_live _ _ _ _ _ _ Hlive1).
  - intros x kind d body He Husable Hheap _.
    apply (plain_step_live _ Hlive1).
    intros x' kind' d' body' v hp' rd' He' Hexec.
    change (state_expr (rs_state (count_uses rs)))
      with (state_expr (rs_state rs)) in He'.
    rewrite He in He'. injection He' as <- <- <- <-.
    apply (exec_named_kept (count_uses rs) d v hp' rd' Hlive1 Hcells Husable);
      [ | exact Hexec ].
    destruct (is_allocation d); [ exact Hheap | reflexivity ].
  - intros bindings body He Husable Hlocal.
    exact
      (static_step_live (count_uses rs) bindings body Hlive1 Hbelow He Husable
         Hlocal).
  - intros ac He. exact (region_apply_cont_live _ ac Hlive1 He).
  - intros Hnot_let _. apply (plain_step_live _ Hlive1).
    intros x kind d body v hp' rd' He.
    contradiction (Hnot_let x kind d body He).
Qed.

(* ======================================================================== *)
(* The cells: what lives as long as what                                    *)
(* ======================================================================== *)

Lemma local_in_frame (frames : list frame) (f : frame) (s : symbol) :
  In f frames ->
  is_some (lookup s (frame_locals f)) = true ->
  local_in frames s = true.
Proof.
  induction frames as [ | f0 below IH]; [ intros [] | ].
  rewrite local_in_cons. intros [<- | Hin] Hs.
  - rewrite Hs. reflexivity.
  - rewrite (IH Hin Hs). apply orb_true_r.
Qed.

Lemma local_in_find (frames : list frame) (s : symbol) :
  local_in frames s = true ->
  ∃ f : frame,
    find (fun f => is_some (lookup s (frame_locals f))) frames = Some f.
Proof.
  induction frames as [ | f0 below IH];
    [ intros H; unfold local_in in H; discriminate | ].
  rewrite local_in_cons. simpl.
  destruct (is_some (lookup s (frame_locals f0))); simpl.
  - intros _. exists f0. reflexivity.
  - exact IH.
Qed.

(* A value that lives as long as what is in the frame of a mark, as
   the test of a store has it ([lives_as_long]), is one that a cell of
   that frame may hold ([value_placed_in]), if it may be evaluated and
   is what a variable may hold. The frame that binds a symbol is the
   one frame of its mark. *)
Lemma lives_as_long_lives_in (rs : region_state) (v : value) (m : mark)
    (fr : list frame) :
  distinct (map frame_mark (rs_frames rs)) = true ->
  frames_from m (rs_frames rs) = Some fr ->
  value_usable rs v = true ->
  value_kept (heap_symbol rs) v = true ->
  value_kept_cells (heap_location rs) v = true ->
  lives_as_long rs v m = true ->
  value_placed_in rs fr v = true.
Proof.
  intros Hdistinct Hfr Hu Hk Hkc. unfold lives_as_long. rewrite Hfr.
  destruct (frames_from_app m _ fr Hfr) as (above & Hframes & _ & _).
  unfold value_placed_in.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c
       | l | callee args args_arity];
    try
      (intros _;
       cbn [value_placed value_placed_cells value_kept value_kept_cells]
         in Hk, Hkc |- *;
       rewrite Hk, Hkc; reflexivity).
  - (* A symbol: of the heap, or bound in a frame at or below that of
       the mark. *)
    cbn [value_frame value_placed value_placed_cells value_holds_cells].
    rewrite andb_true_r.
    destruct (heap_symbol rs s) eqn:Hheap; [ intros _; reflexivity | ].
    pose proof (usable_near rs s Hu) as Hnear. rewrite Hheap in Hnear.
    cbn [orb] in Hnear.
    destruct (local_in_find _ s Hnear) as (f & Hfind). rewrite Hfind.
    cbn [option_map orb]. intros Hon.
    destruct (find_some _ _ Hfind) as [Hin Hs].
    apply (local_in_frame fr f s); [ | exact Hs ].
    rewrite Hframes in Hin, Hdistinct.
    apply in_app_or in Hin as [Hin | Hin]; [ | exact Hin ].
    destruct (distinct_marks_app above fr Hdistinct) as [_ Habove].
    pose proof (Habove _ Hon) as Hnot.
    assert (Hyes : on_stack (frame_mark f) above = true)
      by (apply on_stack_marks, in_map; exact Hin).
    rewrite Hyes in Hnot. discriminate.
  - (* A location: of the heap, or of a cell in such a frame. *)
    revert Hu.
    cbn [value_usable value_frame value_placed value_placed_cells value_holds
         andb].
    unfold cell_dead, local_cell, heap_location, cell_in.
    destruct (lookup l (rs_cells rs)) as [m' | ]; [ | intros _ _; reflexivity ].
    rewrite negb_involutive. intros Hu. rewrite Hu. simpl.
    intros H. exact H.
Qed.

(* A value that the heap and the static environment may hold, in a
   state with more symbols of the heap and that lists the same cells
   at the locations of the value. *)
Lemma value_holds_heap_transfer (hp : heap) (rs rs' : region_state)
    (v : value) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  cells_same_on hp rs rs' ->
  value_closed hp v ->
  value_holds_heap rs v = true ->
  value_holds_heap rs' v = true.
Proof.
  intros Hheap Hsame Hclosed H. unfold value_holds_heap in H |- *.
  apply andb_true_iff in H as [H1 H2]. apply andb_true_iff. split.
  - exact (value_holds_mono _ _ v Hheap H1).
  - exact (value_holds_cells_same hp rs rs' v Hsame Hclosed H2).
Qed.

(* The same of a value that what is in some frames may hold, under
   frames of the same marks that bind as much. *)
Lemma value_placed_in_transfer (hp : heap) (rs rs' : region_state)
    (frames frames' : list frame) (v : value) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  (∀ s : symbol, local_in frames s = true -> local_in frames' s = true) ->
  map frame_mark frames' = map frame_mark frames ->
  cells_same_on hp rs rs' ->
  value_closed hp v ->
  value_placed_in rs frames v = true ->
  value_placed_in rs' frames' v = true.
Proof.
  intros Hheap Hlocal Hmarks Hsame Hclosed H.
  unfold value_placed_in in H |- *.
  apply andb_true_iff in H as [H1 H2]. apply andb_true_iff. split.
  - revert H1. apply value_placed_mono; [ | exact Hheap ].
    intros s Hs. apply orb_true_iff in Hs as [Hs | Hs];
      [ rewrite (Hheap s Hs); reflexivity
      | rewrite (Hlocal s Hs); apply orb_true_r ].
  - exact
      (value_placed_cells_same hp rs rs' frames frames' v Hsame Hmarks
         Hclosed H2).
Qed.

(* What a cell holds lives as long as the cell, in a state after: one
   with more symbols of the heap, that lists the same cells at the
   locations that the heap bound, and in which the frames from that of
   the cell down, if they are still there, are frames that were there,
   with the same marks, binding as much. *)
Lemma cell_holds_transfer (hp : heap) (rs rs' : region_state) (l : location)
    (c : cell) :
  (∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true) ->
  cells_same_on hp rs rs' ->
  is_bound l hp = true ->
  (∀ w : value, In w (values_of_cell c) -> value_closed hp w) ->
  (∀ (m : mark) (fr' : list frame),
    lookup l (rs_cells rs) = Some m ->
    frames_from m (rs_frames rs') = Some fr' ->
    ∃ fr : list frame,
      frames_from m (rs_frames rs) = Some fr
      ∧ map frame_mark fr' = map frame_mark fr
      ∧ ∀ s : symbol, local_in fr s = true -> local_in fr' s = true) ->
  cell_holds rs l c = true ->
  cell_holds rs' l c = true.
Proof.
  intros Hheap Hsame Hl Hclosed Hframes. unfold cell_holds.
  rewrite (Hsame l Hl).
  destruct (lookup l (rs_cells rs)) as [m | ].
  - destruct (frames_from m (rs_frames rs')) as [fr' | ] eqn:Hfr';
      [ | reflexivity ].
    destruct (Hframes m fr' eq_refl Hfr') as (fr & -> & Hmarks & Hlocal).
    apply forallb_impl. intros w Hw.
    exact
      (value_placed_in_transfer hp rs rs' fr fr' w Hheap Hlocal Hmarks Hsame
         (Hclosed w Hw)).
  - apply forallb_impl. intros w Hw.
    exact
      (value_holds_heap_transfer hp rs rs' w Hheap Hsame (Hclosed w Hw)).
Qed.

(* ======================================================================== *)
(* The third part of the invariant                                          *)
(* ======================================================================== *)

(* [cells_live] but for its fourth test, of the symbols that the heap
   names, which FlambdaRegionsFreshProofs.region_step_cells_below
   keeps. *)
Definition cells_ok (rs : region_state) : Prop :=
  distinct (map frame_mark (rs_frames rs)) = true
  ∧ (∀ (l : location) (m : mark),
      In (l, m) (rs_cells rs) ->
      is_bound l (state_heap (rs_state rs)) = true
      ∧ (rs_base rs <= m)%positive
      ∧ (m < rs_next rs)%positive)
  ∧ (∀ (l : location) (c : cell),
      In (l, c) (state_heap (rs_state rs)) -> cell_holds rs l c = true)
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
      In (s, allocated) (static_values (state_statics (rs_state rs))) ->
      In w (allocated_values allocated) ->
      value_holds_cells (heap_location rs) w = true)
  ∧ frames_hold_cells rs (rs_frames rs) = true
  ∧ vals_ok (kept_cells rs) (rs_state rs).

Lemma cells_live_ok (rs : region_state) :
  cells_live rs = true
  <-> cells_ok rs ∧ cells_below (rs_next rs) (state_heap (rs_state rs)).
Proof. rewrite cells_live_parts. unfold cells_ok. tauto. Qed.

(* The tests that depend on the cells listed and on nothing else of
   the state. *)
Lemma heap_location_cells (rs rs' : region_state) :
  rs_cells rs' = rs_cells rs -> heap_location rs' = heap_location rs.
Proof. unfold heap_location. intros ->. reflexivity. Qed.

Lemma frames_hold_cells_cells (rs rs' : region_state) (frames : list frame) :
  rs_cells rs' = rs_cells rs ->
  frames_hold_cells rs' frames = frames_hold_cells rs frames.
Proof.
  destruct rs, rs'. simpl. intros ->.
  induction frames as [ | f below IH]; simpl; [ reflexivity | ].
  rewrite IH. reflexivity.
Qed.

(* In a goal on what the frames hold, of a state that lists the cells
   of [rs]: the same of [rs]. *)
Ltac same_cells rs :=
  match goal with
  | |- frames_hold_cells ?r ?frames = true =>
    rewrite (frames_hold_cells_cells rs r frames eq_refl)
  end.

Lemma cells_ok_count_uses (rs : region_state) :
  cells_ok rs -> cells_ok (count_uses rs).
Proof.
  intros (L1 & L2 & L3 & L4 & L5 & L6). unfold cells_ok.
  split; [ exact L1 | ]. split; [ exact L2 | ]. split; [ exact L3 | ].
  split; [ exact L4 | ]. split; [ | exact L6 ].
  rewrite (frames_hold_cells_cells rs (count_uses rs) _ eq_refl). exact L5.
Qed.

Lemma cells_live_count_uses (rs : region_state) :
  cells_live rs = true -> cells_live (count_uses rs) = true.
Proof.
  intros H. apply cells_live_ok in H as [Hok Hbelow]. apply cells_live_ok.
  split; [ exact (cells_ok_count_uses rs Hok) | exact Hbelow ].
Qed.

(* The marks of the frames were drawn. *)
Lemma symbols_fresh_marks (rs : region_state) :
  symbols_fresh rs = true ->
  ∀ f : frame,
    In f (rs_frames rs) ->
    (rs_base rs <= frame_mark f)%positive
    ∧ (frame_mark f < rs_next rs)%positive.
Proof.
  intros Hfresh f Hin.
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & _ & _ & _ & Hframes & _).
  destruct (Hframes f Hin) as (H1 & H2 & _). exact (conj H1 H2).
Qed.

Lemma is_bound_cons {A : Type} (l l0 : positive) (a : A)
    (bindings : list (positive * A)) :
  is_bound l bindings = true -> is_bound l ((l0, a) :: bindings) = true.
Proof.
  unfold is_bound. rewrite lookup_cons. destruct (Pos.eqb l l0); auto.
Qed.

Lemma is_bound_head {A : Type} (l : positive) (a : A)
    (bindings : list (positive * A)) :
  is_bound l ((l, a) :: bindings) = true.
Proof. unfold is_bound. rewrite lookup_cons, Pos.eqb_refl. reflexivity. Qed.

(* The value of a simple that may be evaluated: one that a cell of the
   newest frame may hold; and, if it is in no frame, one that the heap
   may hold. *)
Lemma simple_lives_in (rs : region_state) (sim : simple) (v : value) :
  values_live rs = true ->
  cells_ok rs ->
  simple_usable rs sim = true ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_placed_in rs (rs_frames rs) v = true.
Proof.
  intros Hlive Hok Hu Hv.
  apply values_live_parts in Hlive as (_ & _ & L3 & _).
  destruct Hok as (_ & _ & _ & _ & _ & Hvals).
  apply kept_cells_vals in Hvals as [Hvars _]. unfold value_placed_in.
  rewrite (simple_placed rs sim v L3 Hu Hv),
    (simple_placed_cells rs sim v Hvars Hu Hv).
  reflexivity.
Qed.

Lemma simple_lives_for_ever (rs : region_state) (sim : simple) (v : value) :
  values_live rs = true ->
  cells_ok rs ->
  simple_usable rs sim = true ->
  simple_local rs sim = false ->
  eval_simple (state_vars (rs_state rs)) sim = Some v ->
  value_holds_heap rs v = true.
Proof.
  intros Hlive Hok Hu Hl Hv.
  apply values_live_parts in Hlive as (_ & _ & L3 & _).
  destruct Hok as (_ & _ & _ & _ & _ & Hvals).
  apply kept_cells_vals in Hvals as [Hvars _]. unfold value_holds_heap.
  rewrite (simple_holds_heap rs sim v L3 Hu Hl Hv),
    (simple_holds_heap_cells rs sim v Hvars Hu Hl Hv).
  reflexivity.
Qed.

(* What a symbol is bound to, in a frame or in the static environment
   of the state, holds directly what a variable may hold, of
   locations. *)
Lemma kept_cells_static (rs : region_state) :
  (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
    In (s, allocated) (static_values (state_statics (rs_state rs))) ->
    In w (allocated_values allocated) ->
    value_holds_cells (heap_location rs) w = true) ->
  frames_hold_cells rs (rs_frames rs) = true ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    kept_cells rs (Value_symbol s) ->
    lookup s (static_values (live_statics rs)) = Some allocated ->
    (∀ w : value, In w (allocated_values allocated) -> kept_cells rs w)
    ∧ (∀ s' : symbol,
        In s' (allocated_symbols allocated) -> kept_cells rs (Value_symbol s')).
Proof.
  intros L4 L5 s allocated _ Hlookup. split; [ | intros s' _; reflexivity ].
  intros w Hw. apply lookup_in in Hlookup. unfold live_statics in Hlookup.
  cbn [static_values] in Hlookup.
  apply in_app_or in Hlookup as [Hin | Hin].
  - exact (frames_hold_cells_in rs _ s allocated w L5 Hin Hw).
  - exact (value_holds_cells_kept _ w (L4 s allocated w Hin Hw)).
Qed.

Lemma kept_cells_fields (rs : region_state) :
  (∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
    In (s, allocated) (static_values (state_statics (rs_state rs))) ->
    In w (allocated_values allocated) ->
    value_holds_cells (heap_location rs) w = true) ->
  frames_hold_cells rs (rs_frames rs) = true ->
  P_fields (kept_cells rs) (live_statics rs).
Proof.
  intros L4 L5. apply (reads_fields (kept_cells rs) (live_statics rs)).
  - exact (kept_cells_parts (heap_location rs)).
  - exact (kept_cells_static rs L4 L5).
Qed.

Lemma cell_allocation_changes_heap (d : named) :
  is_cell_allocation d = true -> named_changes_heap d = true.
Proof.
  destruct d as [s | [u | b | va | q | rp] args | | | | | ];
    try discriminate.
  intros _. reflexivity.
Qed.

Lemma stored_simples (d : named) (target new_simple : simple) :
  stored d = Some (target, new_simple) ->
  In target (simples_of_named d) ∧ In new_simple (simples_of_named d).
Proof.
  destruct d as [s | [u | b | va | q | rp] args | | | | | ];
    try discriminate.
  destruct q as [mut tag | kind init field | kind | kind set_kind];
    try discriminate.
  - destruct args as [ | a [ | b [ | c args]]]; try discriminate.
    intros [= <- <-]. simpl. auto.
  - destruct args as [ | a [ | b [ | c [ | e args]]]]; try discriminate.
    intros [= <- <-]. simpl. auto.
Qed.

(* The state after a [Let] that [plain_step] runs. *)
Lemma plain_step_let (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  state_expr (rs_state rs) = Let x kind d body ->
  plain_step rs = Region_step rs' ->
  ∃ (v : value) (hp' : heap) (rd' : read_record),
    exec_named (live_statics rs) (state_heap (rs_state rs))
      (state_reads (rs_state rs)) (state_vars (rs_state rs)) d
    = Some (v, hp', rd')
    ∧ rs'
      = Region_state
          (State body ((x, v) :: state_vars (rs_state rs))
             (state_conts (rs_state rs)) (state_statics (rs_state rs)) hp' rd')
          (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs)
          (rs_cells rs).
Proof.
  intros He Hstep. destruct (plain_step_step _ _ Hstep) as (s' & Hs & ->).
  revert Hs. unfold with_statics at 1. rewrite He, step_let_exec.
  destruct
    (exec_named (live_statics rs) (state_heap (rs_state rs))
       (state_reads (rs_state rs)) (state_vars (rs_state rs)) d)
    as [[[v hp'] rd'] | ];
    [ | discriminate ].
  destruct (kind_eqb (kind_of_value v) kind); [ | discriminate ].
  intros [= <-]. exists v, hp', rd'. split; reflexivity.
Qed.

Lemma plain_step_cells_listed (rs rs' : region_state) :
  plain_step rs = Region_step rs' -> rs_cells rs' = rs_cells rs.
Proof.
  intros H. destruct (plain_step_step _ _ H) as (s' & _ & ->). reflexivity.
Qed.

Lemma region_apply_cont_cells_listed (rs rs' : region_state)
    (ac : apply_cont) :
  region_apply_cont rs ac = Region_step rs' -> rs_cells rs' = rs_cells rs.
Proof.
  unfold region_apply_cont.
  destruct (resolve (ac_continuation ac) (state_conts (rs_state rs)))
    as [[params metadata handler] def_vars def_conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    try exact (plain_step_cells_listed rs rs').
  destruct (eval_simples (state_vars (rs_state rs)) (ac_args ac)) as [vs | ];
    [ | discriminate ].
  destruct
    (kinds_eqb (map (region_kind_of_value rs) vs) (hm_params_kinds metadata));
    [ | discriminate ].
  destruct (bind_params params vs def_vars) as [handler_vars | ];
    [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

(* ======================================================================== *)
(* A step keeps the third part                                              *)
(* ======================================================================== *)

(* A step that leaves the static environment, the frames, the heap,
   the counter and the cells listed: the test follows from that of the
   variables and of the continuations. *)
Lemma cells_ok_same_store (rs rs' : region_state) :
  rs_base rs' = rs_base rs ->
  rs_next rs' = rs_next rs ->
  state_statics (rs_state rs') = state_statics (rs_state rs) ->
  rs_frames rs' = rs_frames rs ->
  state_heap (rs_state rs') = state_heap (rs_state rs) ->
  rs_cells rs' = rs_cells rs ->
  bound rs ->
  cells_ok rs ->
  vals_ok (kept_cells rs) (rs_state rs') ->
  cells_ok rs'.
Proof.
  intros Hbase Hnext Hstatics Hframes Hheap Hcells Hb
    (L1 & L2 & L3 & L4 & L5 & _) Hvals.
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as (_ & _ & _ & Hkeys & Hclosed & _).
  unfold cells_ok. rewrite Hframes, Hheap, Hcells, Hbase, Hnext, Hstatics.
  split; [ exact L1 | ]. split; [ exact L2 | ]. split.
  { intros l c Hin.
    apply
      (cell_holds_transfer (state_heap (rs_state rs)) rs rs' l c
         (heap_symbol_same rs rs' Hbase Hstatics)
         (cells_same_on_same _ rs rs' Hcells) (Hkeys l c Hin)
         (fun w Hw => Hclosed l c w Hin Hw));
      [ | exact (L3 l c Hin) ].
    intros m fr' _ Hfr'. rewrite Hframes in Hfr'. exists fr'.
    split; [ exact Hfr' | ]. split; [ reflexivity | auto ]. }
  split.
  { intros s allocated w Hin Hw. rewrite (heap_location_cells rs rs' Hcells).
    exact (L4 s allocated w Hin Hw). }
  split; [ rewrite (frames_hold_cells_cells rs rs' _ Hcells); exact L5 | ].
  unfold kept_cells. rewrite (heap_location_cells rs rs' Hcells). exact Hvals.
Qed.

Lemma begin_region_cells (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (mark_kind : mark_kind) (body : expr) :
  (∀ f : frame,
    In f (rs_frames rs) ->
    (rs_base rs <= frame_mark f)%positive
    ∧ (frame_mark f < rs_next rs)%positive) ->
  bound rs ->
  cells_ok rs ->
  begin_region rs x kind mark_kind body = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hmarks Hb (L1 & L2 & L3 & L4 & L5 & L6) Hstep.
  rewrite (begin_region_step _ _ _ _ _ _ Hstep).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as (_ & _ & _ & Hkeys & Hclosed & _).
  unfold cells_ok, bind_let.
  cbn [rs_state rs_frames rs_base rs_next rs_cells state_vars state_conts
       state_statics state_heap].
  split; [ | split; [ | split; [ | split; [ | split ] ] ] ].
  - (* The new mark is the counter, and the marks are below it. *)
    apply (distinct_marks_cons (Frame (rs_next rs) mark_kind []));
      [ | exact L1 ].
    cbn [frame_mark].
    destruct (on_stack (rs_next rs) (rs_frames rs)) eqn:Hon; [ | reflexivity ].
    apply on_stack_marks, in_map_iff in Hon as (f & Hmark & Hin).
    destruct (Hmarks f Hin) as [_ Hlt]. rewrite Hmark in Hlt. lia.
  - intros l m Hin. destruct (L2 l m Hin) as (H1 & H2 & H3).
    split; [ exact H1 | ]. split; [ exact H2 | lia ].
  - intros l c Hin.
    match goal with
    | |- cell_holds ?r _ _ = true =>
      apply
        (cell_holds_transfer (state_heap (rs_state rs)) rs r l c
           (fun s H => H) (cells_same_on_same _ rs r eq_refl) (Hkeys l c Hin)
           (fun w Hw => Hclosed l c w Hin Hw))
    end;
      [ | exact (L3 l c Hin) ].
    intros m fr' Hlookup.
    destruct (L2 l m (lookup_in _ _ _ Hlookup)) as (_ & _ & Hlt).
    cbn [rs_frames frames_from frame_mark].
    destruct (Pos.eqb (rs_next rs) m) eqn:Heq;
      [ apply Pos.eqb_eq in Heq; exfalso; lia | ].
    intros Hfr'. exists fr'.
    split; [ exact Hfr' | ]. split; [ reflexivity | auto ].
  - exact L4.
  - simpl. same_cells rs. exact L5.
  - destruct L6 as [Hv Hc].
    split; [ apply Forall_cons; [ reflexivity | exact Hv ] | exact Hc ].
Qed.

Lemma end_region_cells (rs rs' : region_state) (x : variable)
    (kind : flambda_kind) (ended_by : mark_kind) (region : simple)
    (body : expr) :
  bound rs ->
  cells_ok rs ->
  end_region rs x kind ended_by region body = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hb (L1 & L2 & L3 & L4 & L5 & L6) Hstep.
  destruct (end_region_step _ _ _ _ _ _ _ Hstep)
    as (m & above & f & below & counts & _ & Hframes & ->).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as (_ & _ & _ & Hkeys & Hclosed & _).
  assert (Hsplit : rs_frames rs = (above ++ [f]) ++ below)
    by (rewrite <- app_assoc; exact Hframes).
  rewrite Hsplit in L1, L5.
  destruct (distinct_marks_app (above ++ [f]) below L1) as [Hdistinct _].
  unfold cells_ok, bind_let.
  cbn [rs_state rs_frames rs_base rs_next rs_cells state_vars state_conts
       state_statics state_heap].
  split; [ exact Hdistinct | ]. split; [ exact L2 | ]. split.
  { intros l c Hin.
    match goal with
    | |- cell_holds ?r _ _ = true =>
      apply
        (cell_holds_transfer (state_heap (rs_state rs)) rs r l c
           (fun s H => H) (cells_same_on_same _ rs r eq_refl) (Hkeys l c Hin)
           (fun w Hw => Hclosed l c w Hin Hw))
    end;
      [ | exact (L3 l c Hin) ].
    intros m0 fr' _ Hfr'. cbn [rs_frames] in Hfr'. exists fr'.
    split; [ | split; [ reflexivity | auto ] ].
    rewrite Hsplit. exact (frames_from_below m0 _ below fr' L1 Hfr'). }
  split; [ exact L4 | ]. split.
  { same_cells rs. exact (frames_hold_cells_app rs _ below L5). }
  destruct L6 as [Hv Hc].
  split; [ apply Forall_cons; [ reflexivity | exact Hv ] | exact Hc ].
Qed.

Lemma local_allocation_cells (rs rs' : region_state) (x y : variable)
    (kind : flambda_kind) (d : named) (region : option simple)
    (body : expr) :
  is_cell_allocation d && kind_eqb kind Value = false ->
  bound rs ->
  cells_ok rs ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  local_allocation rs x y kind d region body = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hnot Hb Hok Husable Hstep.
  destruct (local_allocation_step rs rs' x y kind d region body Hnot Hstep)
    as (v & sym & bindings & His & Heval & Hlocalise & _ & Hvars & Hconts
        & Hbase & Hnext & _ & Hframes).
  destruct (local_allocation_memory rs rs' x y kind d region body Hnot Hstep)
    as (Hheap & _ & Hcells).
  destruct (eval_named_makes _ _ d v His Heval) as [_ Hparts].
  destruct (localise_makes _ _ _ _ Hlocalise) as (_ & _ & Hbound).
  pose proof (advance_le (rs_next rs) (length bindings)) as Hle.
  pose proof Hok as (L1 & L2 & L3 & L4 & L5 & L6).
  destruct (proj2 (kept_cells_vals rs (rs_state rs)) L6) as [Lvars _].
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as (_ & _ & _ & Hkeys & Hclosed & _).
  pose proof (cells_same_on_same (state_heap (rs_state rs)) rs rs' Hcells)
    as Hsame.
  (* What the bindings hold are the values of the simples. *)
  assert (Hnew :
    ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
      In (s, allocated) bindings ->
      In w (allocated_values allocated) ->
      ∃ sim : simple,
        In sim (simples_of_named d)
        ∧ eval_simple (state_vars (rs_state rs)) sim = Some w).
  { intros s allocated w Hin Hw.
    destruct (Hbound s allocated Hin) as (_ & _ & Hvalues & _).
    rewrite Hvalues in Hw. exact (Hparts w Hw). }
  assert (Hvals' : vals_ok (kept_cells rs') (rs_state rs')).
  { destruct L6 as [Hv Hc]. unfold kept_cells.
    rewrite (heap_location_cells rs rs' Hcells).
    split; [ rewrite Hvars | rewrite Hconts; exact Hc ].
    apply Forall_cons; [ reflexivity | ].
    apply Forall_cons; [ reflexivity | exact Hv ]. }
  assert (Hlisted :
    ∀ (l : location) (m : mark),
      In (l, m) (rs_cells rs) ->
      is_bound l (state_heap (rs_state rs)) = true
      ∧ (rs_base rs <= m)%positive
      ∧ (m < advance (rs_next rs) (length bindings))%positive).
  { intros l m Hin. destruct (L2 l m Hin) as (H1 & H2 & H3).
    split; [ exact H1 | ]. split; [ exact H2 | lia ]. }
  unfold cells_ok. rewrite Hheap, Hcells, Hbase, Hnext.
  destruct (rs_frames rs) as [ | [m mark_kind locals] frames] eqn:Hrs_frames.
  - (* No frame: the bindings go to the static environment. *)
    destruct Hframes as [Hframes' Hvalues'].
    pose proof (heap_symbol_extends rs rs' bindings Hbase Hvalues')
      as Hheap_symbol.
    rewrite Hframes', Hvalues'.
    split; [ reflexivity | ]. split; [ exact Hlisted | ]. split.
    { intros l c Hin.
      apply
        (cell_holds_transfer (state_heap (rs_state rs)) rs rs' l c
           Hheap_symbol Hsame (Hkeys l c Hin)
           (fun w Hw => Hclosed l c w Hin Hw));
        [ | exact (L3 l c Hin) ].
      intros m0 fr' _. rewrite Hframes'. discriminate. }
    split.
    { intros s allocated w Hin Hw.
      rewrite (heap_location_cells rs rs' Hcells).
      apply in_app_or in Hin as [Hin | Hin];
        [ | exact (L4 s allocated w Hin Hw) ].
      destruct (Hnew s allocated w Hin Hw) as (sim & Hsim & Hsim_w).
      exact
        (simple_holds_heap_cells rs sim w Lvars
           (forallb_in _ _ sim Husable Hsim)
           (simple_local_no_frames rs sim Hrs_frames) Hsim_w). }
    split; [ reflexivity | exact Hvals' ].
  - (* The newest frame takes the bindings. *)
    destruct Hframes as [Hframes' Hstatics'].
    pose proof (heap_symbol_same rs rs' Hbase Hstatics') as Hheap_symbol.
    rewrite Hframes', Hstatics'.
    split; [ exact L1 | ]. split; [ exact Hlisted | ]. split.
    { intros l c Hin.
      apply
        (cell_holds_transfer (state_heap (rs_state rs)) rs rs' l c
           Hheap_symbol Hsame (Hkeys l c Hin)
           (fun w Hw => Hclosed l c w Hin Hw));
        [ | exact (L3 l c Hin) ].
      intros m0 fr' _. rewrite Hframes', Hrs_frames.
      cbn [frames_from frame_mark].
      destruct (Pos.eqb m m0).
      - intros [= <-]. exists (Frame m mark_kind locals :: frames).
        split; [ reflexivity | ]. split; [ reflexivity | ].
        intros s. rewrite !local_in_cons. cbn [frame_locals].
        rewrite is_some_lookup_app. intros H.
        apply orb_true_iff in H as [H | H]; rewrite H;
          rewrite ?orb_true_r; reflexivity.
      - intros Hfr'. exists fr'.
        split; [ exact Hfr' | ]. split; [ reflexivity | auto ]. }
    split.
    { intros s allocated w Hin Hw.
      rewrite (heap_location_cells rs rs' Hcells).
      exact (L4 s allocated w Hin Hw). }
    split; [ | exact Hvals' ].
    rewrite (frames_hold_cells_cells rs rs' _ Hcells).
    simpl in L5 |- *. apply andb_true_iff in L5 as [Lf Lbelow].
    apply andb_true_iff. split; [ | exact Lbelow ].
    rewrite forallb_app. apply andb_true_iff. split.
    + apply forallb_forall. intros [s allocated] Hin.
      apply forallb_forall. intros w Hw.
      rewrite values_of_static_values in Hw.
      destruct (Hnew s allocated w Hin Hw) as (sim & Hsim & Hsim_w).
      pose proof
        (simple_placed_cells rs sim w Lvars (forallb_in _ _ sim Husable Hsim)
           Hsim_w)
        as Hplaced.
      rewrite Hrs_frames in Hplaced. revert Hplaced.
      apply value_placed_cells_in; [ intros l _ H; exact H | auto ].
    + revert Lf. apply forallb_impl. intros [s allocated] _.
      apply forallb_impl. intros w _.
      apply value_placed_cells_in; [ intros l _ H; exact H | auto ].
Qed.

Lemma local_cell_allocation_cells (rs rs' : region_state) (x y : variable)
    (d : named) (region : option simple) (body : expr) :
  (∀ f : frame,
    In f (rs_frames rs) ->
    (rs_base rs <= frame_mark f)%positive
    ∧ (frame_mark f < rs_next rs)%positive) ->
  values_live rs = true ->
  bound rs ->
  cells_ok rs ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  is_cell_allocation d = true ->
  local_cell_allocation rs x y d region body = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hmarks Hlive Hb Hok Husable His Hstep.
  destruct (local_cell_allocation_step rs rs' x y d region body Hstep)
    as (l & hp' & rd' & _ & Hexec & Hstate & Hframes & Hbase & Hnext & Hcells).
  pose proof Hok as (L1 & L2 & L3 & L4 & L5 & L6).
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb
    as (Hvals_closed & Hlocals_closed & Hstatics_closed & Hkeys & Hclosed & _).
  (* The cell is made of the values of the simples, at a location that
     the heap did not bind and that is not listed. *)
  destruct (exec_named_heap_cases _ _ _ _ _ _ d _ Hexec)
    as [[Hno _]
       | [(_ & vs & c & Hvs & Hc & Hl & Hhp)
         | (t & n & l0 & new & c & c' & i & _ & _ & _ & _ & _ & Hunit & _)]].
  { rewrite (cell_allocation_changes_heap d His) in Hno. discriminate. }
  2: discriminate Hunit.
  injection Hl as ->.
  pose proof (FlambdaHeapProofs.fresh_location (state_heap (rs_state rs)))
    as Hfresh.
  assert (Hunbound :
    is_bound (fresh_location (state_heap (rs_state rs)))
      (state_heap (rs_state rs))
    = false)
    by (unfold is_bound; rewrite Hfresh; reflexivity).
  assert (Hunlisted :
    lookup (fresh_location (state_heap (rs_state rs))) (rs_cells rs) = None).
  { destruct
      (lookup (fresh_location (state_heap (rs_state rs))) (rs_cells rs))
      as [m | ] eqn:Hlisted;
      [ | reflexivity ].
    destruct (L2 _ m (lookup_in _ _ _ Hlisted)) as [Hbound_lf _].
    rewrite Hunbound in Hbound_lf. discriminate. }
  assert (Hsame : cells_same_on (state_heap (rs_state rs)) rs rs').
  { intros l0 Hl0. rewrite Hcells.
    destruct (rs_frames rs) as [ | f frames]; [ reflexivity | ].
    rewrite lookup_cons.
    destruct (Pos.eqb l0 (fresh_location (state_heap (rs_state rs))))
      eqn:Heq;
      [ | reflexivity ].
    apply Pos.eqb_eq in Heq. rewrite Heq, Hunbound in Hl0. discriminate. }
  assert (Hstatics' :
    state_statics (rs_state rs') = state_statics (rs_state rs))
    by (rewrite Hstate; reflexivity).
  pose proof (heap_symbol_same rs rs' Hbase Hstatics') as Hheap_symbol.
  assert (Hmade :
    ∀ w : value,
      In w vs ->
      value_closed (state_heap (rs_state rs)) w
      ∧ ∃ sim : simple,
          simple_usable rs sim = true
          ∧ eval_simple (state_vars (rs_state rs)) sim = Some w).
  { intros w Hw. destruct (Hvs w Hw) as (sim & Hsim & Hsim_w).
    split;
      [ exact (eval_simple_closed _ _ sim w (proj1 Hvals_closed) Hsim_w) | ].
    exists sim.
    split; [ exact (forallb_in _ _ sim Husable Hsim) | exact Hsim_w ]. }
  unfold cells_ok. rewrite Hframes, Hbase, Hnext, Hstate.
  cbn [state_vars state_conts state_statics state_heap].
  split; [ exact L1 | ]. split.
  { (* The cells listed. *)
    intros l0 m. rewrite Hcells, Hhp. intros Hin.
    assert (Hold :
      In (l0, m) (rs_cells rs) ->
      is_bound l0
        ((fresh_location (state_heap (rs_state rs)), c)
         :: state_heap (rs_state rs))
      = true
      ∧ (rs_base rs <= m)%positive
      ∧ (m < rs_next rs)%positive).
    { intros H. destruct (L2 l0 m H) as (H1 & H2).
      exact (conj (is_bound_cons l0 _ c _ H1) H2). }
    revert Hin Hmarks. destruct (rs_frames rs) as [ | f frames];
      intros Hin Hmarks; [ exact (Hold Hin) | ].
    destruct Hin as [[= <- <-] | Hin]; [ | exact (Hold Hin) ].
    split; [ apply is_bound_head | exact (Hmarks f (or_introl eq_refl)) ]. }
  split.
  { (* What the cells hold. *)
    intros l0 c0. rewrite Hhp. intros [[= <- <-] | Hin].
    - unfold cell_holds. rewrite Hcells, Hframes, Hc.
      destruct (rs_frames rs) as [ | f frames] eqn:Hrs_frames.
      + rewrite Hunlisted. apply forallb_forall. intros w Hw.
        destruct (Hmade w Hw) as (Hcl & sim & Hu & Hsim_w).
        exact
          (value_holds_heap_transfer _ rs rs' w Hheap_symbol Hsame Hcl
             (simple_lives_for_ever rs sim w Hlive Hok Hu
                (simple_local_no_frames rs sim Hrs_frames) Hsim_w)).
      + rewrite lookup_cons, Pos.eqb_refl. cbn [frames_from].
        rewrite Pos.eqb_refl. apply forallb_forall. intros w Hw.
        destruct (Hmade w Hw) as (Hcl & sim & Hu & Hsim_w).
        pose proof (simple_lives_in rs sim w Hlive Hok Hu Hsim_w) as Hin_frames.
        rewrite Hrs_frames in Hin_frames.
        exact
          (value_placed_in_transfer _ rs rs' _ _ w Hheap_symbol (fun s H => H)
             eq_refl Hsame Hcl Hin_frames).
    - apply
        (cell_holds_transfer (state_heap (rs_state rs)) rs rs' l0 c0
           Hheap_symbol Hsame (Hkeys l0 c0 Hin)
           (fun w Hw => Hclosed l0 c0 w Hin Hw));
        [ | exact (L3 l0 c0 Hin) ].
      intros m fr' _ Hfr'. rewrite Hframes in Hfr'. exists fr'.
      split; [ exact Hfr' | ]. split; [ reflexivity | auto ]. }
  split.
  { intros s allocated w Hin Hw.
    exact
      (value_holds_cells_same _ rs rs' w Hsame
         (Hstatics_closed s allocated w Hin Hw) (L4 s allocated w Hin Hw)). }
  split;
    [ exact (frames_hold_cells_same _ rs rs' _ Hsame Hlocals_closed L5) | ].
  pose proof
    (vals_ok_closed (kept_cells rs) (kept_cells rs') _ (rs_state rs)
       (fun v Hcl => value_kept_cells_same _ rs rs' v Hsame Hcl) Hvals_closed
       L6)
    as [Hv Hc0].
  split; [ | exact Hc0 ].
  apply Forall_cons; [ reflexivity | ].
  apply Forall_cons; [ reflexivity | exact Hv ].
Qed.

(* A [Let] that is not a local allocation: it leaves the heap, or
   makes a cell on the heap out of what is in no frame (decision D46),
   or stores what lives as long as the cell (decision D54). *)
Lemma let_cells (rs rs' : region_state) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  values_live rs = true ->
  cells_ok rs ->
  state_expr (rs_state rs) = Let x kind d body ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  (is_allocation d || is_cell_allocation d)
  && existsb (simple_local rs) (simples_of_named d)
  = false ->
  store_fault rs d = None ->
  plain_step rs = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hlive Hok He Husable Hheap Hfault Hstep.
  destruct (plain_step_let rs rs' x kind d body He Hstep)
    as (v & hp' & rd' & Hexec & ->).
  pose proof Hok as (L1 & L2 & L3 & L4 & L5 & L6).
  destruct (proj2 (kept_cells_vals rs (rs_state rs)) L6) as [Lvars _].
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & Lsym & _).
  assert (Hresult :
    (∀ (l : location) (c : cell), In (l, c) hp' -> cell_holds rs l c = true)
    ∧ (∀ l : location,
        is_bound l (state_heap (rs_state rs)) = true -> is_bound l hp' = true)
    ∧ kept_cells rs v).
  { destruct (exec_named_heap_cases _ _ _ _ _ _ d v Hexec)
      as [[Hno ->]
         | [(His & vs & c & Hvs & Hc & -> & ->)
           | (t & n & l & new & c & c' & i & Hstored & Ht & Hn & Hl & Hwith
              & -> & ->)]].
    - (* The heap is as it was. *)
      split; [ exact L3 | ]. split; [ auto | ].
      destruct (is_allocation d) eqn:Halloc.
      + assert (Hnot_local :
          existsb (simple_local rs) (simples_of_named d) = false)
          by (revert Hheap; try rewrite Halloc; simpl; auto).
        destruct (exec_named_makes _ _ _ _ _ _ d v Halloc Hexec) as [Heval _].
        destruct (eval_named_makes _ _ d v Halloc Heval) as [Hmade Hparts].
        apply value_holds_cells_kept, value_holds_cells_made;
          [ exact Hmade | ].
        apply forallb_forall. intros w Hin.
        destruct (Hparts w Hin) as (sim & Hsim & Hw).
        exact
          (simple_holds_heap_cells rs sim w Lvars
             (forallb_in _ _ sim Husable Hsim)
             (existsb_false_in _ _ sim Hnot_local Hsim) Hw).
      + apply
          (exec_named_reads (kept_cells rs) (live_statics rs)
             (fun c _ => eq_refl) (kept_cells_parts (heap_location rs))
             (fun slot slot' decls slots H => H) (kept_cells_static rs L4 L5)
             (state_heap (rs_state rs)) (state_heap (rs_state rs))
             (state_reads (rs_state rs)) rd' (state_vars (rs_state rs)) d v
             Halloc Hno);
          [ | | exact Hexec ].
        * intros sim w _. exact (eval_simple_kept_cells _ _ sim w Lvars).
        * intros sim l c w Hsim Heval Hlookup Hw.
          refine
            (proj1
               (proj2
                  (live_cell_holds rs l c w (L3 l c (lookup_in _ _ _ Hlookup))
                     _ Hw))).
          pose proof (forallb_in _ _ sim Husable Hsim) as Hu.
          unfold simple_usable in Hu. rewrite Heval in Hu.
          apply negb_true_iff in Hu. exact Hu.
    - (* A cell is made on the heap, out of what is in no frame. *)
      assert (Hnot_local :
        existsb (simple_local rs) (simples_of_named d) = false)
        by (revert Hheap; rewrite His, orb_true_r; simpl; auto).
      split;
        [ | split; [ intros l H; apply is_bound_cons; exact H | reflexivity ] ].
      intros l c0 [[= <- <-] | Hin]; [ | exact (L3 l c0 Hin) ].
      assert (Hunlisted :
        lookup (fresh_location (state_heap (rs_state rs))) (rs_cells rs)
        = None).
      { destruct
          (lookup (fresh_location (state_heap (rs_state rs))) (rs_cells rs))
          as [m | ] eqn:Hlisted;
          [ | reflexivity ].
        destruct (L2 _ m (lookup_in _ _ _ Hlisted)) as [Hbound_lf _].
        unfold is_bound in Hbound_lf.
        rewrite (FlambdaHeapProofs.fresh_location (state_heap (rs_state rs)))
          in Hbound_lf.
        discriminate. }
      unfold cell_holds. rewrite Hunlisted, Hc.
      apply forallb_forall. intros w Hw.
      destruct (Hvs w Hw) as (sim & Hsim & Hsim_w).
      exact
        (simple_lives_for_ever rs sim w Hlive Hok
           (forallb_in _ _ sim Husable Hsim)
           (existsb_false_in _ _ sim Hnot_local Hsim) Hsim_w).
    - (* A store: what is stored lives as long as the cell. *)
      destruct (stored_simples d t n Hstored) as [_ Hn_in].
      pose proof (forallb_in _ _ n Husable Hn_in) as Hu.
      split;
        [ | split; [ intros l0 H; apply is_bound_cons; exact H | ];
            reflexivity ].
      intros l0 c0 [[= <- <-] | Hin]; [ | exact (L3 l0 c0 Hin) ].
      pose proof (L3 l c (lookup_in _ _ _ Hl)) as Hold.
      revert Hfault Hold. unfold store_fault, cell_holds, local_cell.
      rewrite Hstored. cbv beta iota zeta. rewrite Ht, Hn. cbv beta iota.
      destruct (lookup l (rs_cells rs)) as [m | ].
      + destruct (frames_from m (rs_frames rs)) as [fr | ] eqn:Hfr;
          [ | intros _ _; reflexivity ].
        rewrite (frames_from_on_stack m _ fr Hfr). cbv beta iota.
        destruct (lives_as_long rs new m) eqn:Hlives; [ | discriminate ].
        intros _ Hold. apply forallb_forall. intros w Hw.
        destruct (cell_with_values i new c c' w Hwith Hw) as [-> | Hw_old];
          [ | exact (forallb_in _ _ w Hold Hw_old) ].
        unfold simple_usable in Hu. rewrite Hn in Hu.
        exact
          (lives_as_long_lives_in rs new m fr L1 Hfr Hu
             (eval_simple_kept _ _ n new Lsym Hn)
             (eval_simple_kept_cells _ _ n new Lvars Hn) Hlives).
      + destruct (value_local rs new) eqn:Hlocal; [ discriminate | ].
        intros _ Hold. apply forallb_forall. intros w Hw.
        destruct (cell_with_values i new c c' w Hwith Hw) as [-> | Hw_old];
          [ | exact (forallb_in _ _ w Hold Hw_old) ].
        apply (simple_lives_for_ever rs n new Hlive Hok Hu); [ | exact Hn ].
        unfold simple_local. rewrite Hn. exact Hlocal. }
  destruct Hresult as (Hholds & Hkept & Hv).
  unfold cells_ok.
  cbn [rs_state rs_frames rs_base rs_next rs_cells state_vars state_conts
       state_statics state_heap].
  split; [ exact L1 | ]. split.
  { intros l m Hin. destruct (L2 l m Hin) as (H1 & H2).
    exact (conj (Hkept l H1) H2). }
  split; [ exact Hholds | ]. split; [ exact L4 | ].
  split; [ same_cells rs; exact L5 | ].
  destruct L6 as [Hvs Hcs].
  split; [ apply Forall_cons; [ exact Hv | exact Hvs ] | exact Hcs ].
Qed.

Lemma static_step_cells (rs rs' : region_state)
    (bindings : list static_binding) (body : expr) :
  bound rs ->
  cells_ok rs ->
  state_expr (rs_state rs) = Let_static bindings body ->
  forallb (simple_usable rs) (flat_map simples_of_binding bindings) = true ->
  existsb (simple_local rs) (flat_map simples_of_binding bindings) = false ->
  static_step rs = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hb Hok He Husable Hlocal Hstep.
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & _ & Hvalues & _ & Hnew).
  pose proof Hok as (L1 & L2 & L3 & L4 & L5 & L6).
  destruct (proj2 (kept_cells_vals rs (rs_state rs)) L6) as [Lvars _].
  apply bound_parts in Hb. cbv zeta in Hb.
  destruct Hb as (_ & _ & _ & Hkeys & Hclosed & _).
  unfold cells_ok.
  cbn [rs_state rs_frames rs_base rs_next rs_cells state_vars state_conts
       state_statics state_heap].
  split; [ exact L1 | ]. split; [ exact L2 | ]. split.
  { intros l c Hin.
    apply (cell_holds_transfer (state_heap (rs_state rs)) rs _ l c);
      [ | exact (cells_same_on_same _ _ _ eq_refl) | exact (Hkeys l c Hin)
      | exact (fun w Hw => Hclosed l c w Hin Hw) | | exact (L3 l c Hin) ].
    - apply (heap_symbol_extends rs _ new_values); [ reflexivity | ].
      exact Hvalues.
    - intros m fr' _ Hfr'. exists fr'.
      split; [ exact Hfr' | ]. split; [ reflexivity | auto ]. }
  split.
  { intros s allocated w Hin Hw. rewrite Hvalues in Hin.
    apply in_app_or in Hin as [Hin | Hin];
      [ | exact (L4 s allocated w Hin Hw) ].
    destruct (Hnew s allocated Hin) as (const & Hconst_in & Hconst).
    destruct (eval_static_const_makes _ const allocated Hconst)
      as [Hparts _].
    destruct (Hparts w Hw) as (sim & Hsim & Hsim_w).
    assert (Hsim_in : In sim (flat_map simples_of_binding bindings))
      by (apply in_flat_map; exists (Bind_symbol s const);
          split; [ exact Hconst_in | exact Hsim ]).
    exact
      (simple_holds_heap_cells rs sim w Lvars
         (forallb_in _ _ sim Husable Hsim_in)
         (existsb_false_in _ _ sim Hlocal Hsim_in) Hsim_w). }
  split; [ same_cells rs; exact L5 | ].
  exact L6.
Qed.

Lemma region_apply_cont_cells (rs rs' : region_state) (ac : apply_cont) :
  bound rs ->
  cells_ok rs ->
  state_expr (rs_state rs) = Apply_cont ac ->
  region_apply_cont rs ac = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hb Hok He Hstep.
  pose proof Hok as (_ & _ & _ & L4 & L5 & L6).
  destruct (proj2 (kept_cells_vals rs (rs_state rs)) L6) as [Lvars _].
  pose proof
    (region_apply_cont_values (kept_cells rs) (kept_cells_partial rs) rs ac He
       (kept_cells_fields rs L4 L5) L6
       (fun sim v _ => eval_simple_kept_cells _ _ sim v Lvars))
    as Hvals.
  rewrite Hstep in Hvals. cbn [region_result_vals_ok] in Hvals.
  destruct (region_apply_cont_step rs rs' ac Hstep)
    as (Hframes & Hbase & Hnext & Hstatics).
  exact
    (cells_ok_same_store rs rs' Hbase Hnext Hstatics Hframes
       (region_apply_cont_heap rs rs' ac He Hstep)
       (region_apply_cont_cells_listed rs rs' ac Hstep) Hb Hok Hvals).
Qed.

Lemma plain_step_cells (rs rs' : region_state) :
  bound rs ->
  cells_ok rs ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr),
    state_expr (rs_state rs) <> Let x kind d body) ->
  plain_step rs = Region_step rs' ->
  cells_ok rs'.
Proof.
  intros Hb Hok Hnot_let Hstep.
  pose proof Hok as (_ & _ & _ & L4 & L5 & L6).
  destruct (proj2 (kept_cells_vals rs (rs_state rs)) L6) as [Lvars _].
  pose proof
    (plain_step_values (kept_cells rs) (kept_cells_partial rs) rs
       (kept_cells_fields rs L4 L5) L6
       (fun x kind d body v hp' rd' He _ =>
          False_ind _ (Hnot_let x kind d body He))
       (fun sim v _ => eval_simple_kept_cells _ _ sim v Lvars))
    as Hvals.
  rewrite Hstep in Hvals. cbn [region_result_vals_ok] in Hvals.
  assert (Hheap : state_heap (rs_state rs') = state_heap (rs_state rs)).
  { destruct (plain_step_heap rs rs' Hstep)
      as [H | (x & kind & d & body & v & rd' & He & _)];
      [ exact H | ].
    destruct (Hnot_let x kind d body He). }
  destruct (plain_step_step _ _ Hstep) as (s' & _ & Hrs').
  rewrite Hrs' in Hheap, Hvals |- *.
  match goal with
  | |- cells_ok ?r =>
    exact
      (cells_ok_same_store rs r eq_refl eq_refl eq_refl eq_refl Hheap eq_refl
         Hb Hok Hvals)
  end.
Qed.

(* A step keeps the third part of the invariant. *)
Theorem region_step_cells (rs rs' : region_state) :
  symbols_fresh rs = true ->
  values_live rs = true ->
  cells_live rs = true ->
  locations_all_bound rs = true ->
  region_step rs = Region_step rs' ->
  cells_live rs' = true.
Proof.
  intros Hfresh Hlive Hcells Hbound Hstep.
  apply cells_live_ok in Hcells as [Hok Hbelow]. apply cells_live_ok.
  split; [ | exact (region_step_cells_below rs rs' Hfresh Hbelow Hstep) ].
  apply locations_all_bound_iff in Hbound.
  pose proof (symbols_fresh_marks rs Hfresh) as Hmarks.
  pose proof (values_live_count_uses rs Hlive) as Hlive1.
  pose proof (cells_ok_count_uses rs Hok) as Hok1.
  revert rs' Hstep.
  apply
    (region_step_cases rs
       (fun result =>
          ∀ rs' : region_state, result = Region_step rs' -> cells_ok rs')).
  - intros fault rs'. discriminate.
  - intros x kind d mark_kind body _ rs'.
    exact
      (begin_region_cells (count_uses rs) rs' x kind mark_kind body Hmarks
         Hbound Hok1).
  - intros x kind d ended_by region body _ rs'.
    exact
      (end_region_cells (count_uses rs) rs' x kind ended_by region body Hbound
         Hok1).
  - intros x kind d body y region body' _ _ Husable Hnot rs'.
    exact
      (local_allocation_cells (count_uses rs) rs' x y kind d region body' Hnot
         Hbound Hok1 Husable).
  - intros x kind d body y region body' _ _ Husable His rs'.
    exact
      (local_cell_allocation_cells (count_uses rs) rs' x y d region body'
         Hmarks Hlive1 Hbound Hok1 Husable His).
  - intros x kind d body He Husable Hheap Hfault rs'.
    exact
      (let_cells (count_uses rs) rs' x kind d body Hlive1 Hok1 He Husable
         Hheap Hfault).
  - intros bindings body He Husable Hlocal rs'.
    exact
      (static_step_cells (count_uses rs) rs' bindings body Hbound Hok1 He
         Husable Hlocal).
  - intros ac He rs'.
    exact (region_apply_cont_cells (count_uses rs) rs' ac Hbound Hok1 He).
  - intros Hnot_let _ rs'.
    exact (plain_step_cells (count_uses rs) rs' Hbound Hok1 Hnot_let).
Qed.

(* The cells that a step lists: those that were listed, and with them
   the location of the cell that a local allocation makes, which the
   heap did not bind. *)
Lemma region_step_listed (rs rs' : region_state) :
  region_step rs = Region_step rs' ->
  rs_cells rs' = rs_cells rs
  ∨ ∃ m : mark,
      rs_cells rs'
      = (fresh_location (state_heap (rs_state rs)), m) :: rs_cells rs.
Proof.
  revert rs'.
  apply
    (region_step_cases rs
       (fun result =>
          ∀ rs' : region_state,
            result = Region_step rs' ->
            rs_cells rs' = rs_cells rs
            ∨ ∃ m : mark,
                rs_cells rs'
                = (fresh_location (state_heap (rs_state rs)), m)
                  :: rs_cells rs)).
  - intros fault rs'. discriminate.
  - intros x kind d mark_kind body _ rs' Hstep. left.
    rewrite (begin_region_step _ _ _ _ _ _ Hstep). reflexivity.
  - intros x kind d ended_by region body _ rs' Hstep. left.
    destruct (end_region_step _ _ _ _ _ _ _ Hstep)
      as (m & above & f & below & counts & _ & _ & ->).
    reflexivity.
  - intros x kind d body y region body' _ _ _ Hnot rs' Hstep. left.
    exact
      (proj2
         (proj2 (local_allocation_memory _ rs' x y kind d region body' Hnot
                   Hstep))).
  - intros x kind d body y region body' _ _ _ His rs' Hstep.
    destruct (local_cell_allocation_step _ rs' x y d region body' Hstep)
      as (l & hp' & rd' & _ & Hexec & _ & _ & _ & _ & Hcells).
    destruct (exec_named_heap_cases _ _ _ _ _ _ d _ Hexec)
      as [[Hno _]
         | [(_ & vs & c & _ & _ & Hl & _)
           | (t & n & l0 & new & c & c' & i & _ & _ & _ & _ & _ & Hunit & _)]].
    + rewrite (cell_allocation_changes_heap d His) in Hno. discriminate.
    + injection Hl as ->. revert Hcells.
      change (rs_frames (count_uses rs)) with (rs_frames rs).
      change (rs_cells (count_uses rs)) with (rs_cells rs).
      destruct (rs_frames rs) as [ | f frames]; intros Hcells;
        [ left; exact Hcells | ].
      right. exists (frame_mark f). exact Hcells.
    + discriminate Hunit.
  - intros x kind d body _ _ _ _ rs' Hstep. left.
    exact (plain_step_cells_listed _ rs' Hstep).
  - intros bindings body He _ _ rs' Hstep. left.
    destruct (static_step_step (count_uses rs) rs' bindings body He Hstep)
      as (statics' & _ & ->).
    reflexivity.
  - intros ac _ rs' Hstep. left.
    exact (region_apply_cont_cells_listed _ rs' ac Hstep).
  - intros _ _ rs' Hstep. left. exact (plain_step_cells_listed _ rs' Hstep).
Qed.

(* A value that names no location holds locations of any set. *)
Lemma value_holds_cells_any (v : value) :
  value_holds_cells (fun _ => true) v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    simpl; auto.
  - apply forallb_forall. intros [vslot w] Hin. rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin).
  - apply forallb_forall. intros [vslot w] Hin. rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin).
  - apply forallb_forall. intros w Hin. rewrite Forall_forall in IH.
    exact (IH w Hin).
  - apply forallb_forall. intros w Hin. rewrite Forall_forall in IH.
    exact (IH w Hin).
  - apply andb_true_iff. split.
    + destruct callee; try reflexivity; exact IHcallee.
    + apply forallb_forall. intros arg Hin. rewrite Forall_forall in IHargs.
      pose proof (IHargs _ Hin) as Harg.
      destruct arg; try reflexivity; exact Harg.
Qed.

Lemma no_location_holds_cells (ok : location -> bool) (v : value) :
  locations_of_value v = [] -> value_holds_cells ok v = true.
Proof.
  intros Hnone.
  apply (value_holds_cells_in (fun _ => true) ok v);
    [ | exact (value_holds_cells_any v) ].
  intros l Hl. rewrite Hnone in Hl. destruct Hl.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma region_invariant_parts (rs : region_state) :
  region_invariant rs = true
  <-> symbols_fresh rs = true
      ∧ values_live rs = true
      ∧ cells_live rs = true
      ∧ locations_all_bound rs = true.
Proof. unfold region_invariant. rewrite !andb_true_iff. tauto. Qed.

Theorem region_invariant_initial : region_invariant_initial_statement.
Proof.
  intros e statics base Hfresh. apply region_invariant_parts.
  split; [ exact (symbols_fresh_initial e statics base Hfresh) | ].
  unfold region_names_fresh in Hfresh.
  apply andb_true_iff in Hfresh as [Hfresh Hno_location].
  apply andb_true_iff in Hfresh as [_ Hstatics].
  assert (Hnone :
    ∀ (s : symbol) (allocated : statically_allocated_value) (w : value),
      In (s, allocated) (static_values statics) ->
      In w (allocated_values allocated) ->
      locations_of_value w = []).
  { intros s allocated w Hin Hw.
    unfold statics_hold_no_location in Hno_location.
    pose proof (forallb_in _ _ (s, allocated) Hno_location Hin) as H.
    cbv beta iota in H. rewrite values_of_static_values in H.
    pose proof (forallb_in _ _ w H Hw) as Hw_none. cbv beta in Hw_none.
    destruct (locations_of_value w); [ reflexivity | discriminate ]. }
  split; [ | split ].
  - apply values_live_parts. unfold initial_region_state.
    cbn [rs_state rs_frames state_statics state_vars state_conts].
    repeat split; try reflexivity.
    unfold statics_below in Hstatics.
    apply andb_true_iff in Hstatics as [_ Hvalues].
    revert Hvalues. apply forallb_impl. intros [s allocated] _ H.
    apply andb_true_iff in H as [_ H].
    rewrite allocated_below_parts in H. apply andb_true_iff in H as [Hv Hs].
    rewrite allocated_holds_parts. apply andb_true_iff. split.
    + revert Hv. apply forallb_impl. intros w _.
      apply value_below_holds. intros s'.
      exact (below_base_heap (initial_region_state e statics base) s').
    + revert Hs. apply forallb_impl. intros s' _.
      exact (below_base_heap (initial_region_state e statics base) s').
  - (* No frame, no cell, an empty heap; the static environment holds
       no location. *)
    apply cells_live_parts. unfold initial_region_state.
    cbn [rs_state rs_frames rs_cells rs_base rs_next state_statics state_vars
         state_conts state_heap].
    split; [ reflexivity | ]. split; [ intros l m [] | ].
    split; [ intros l c [] | ]. split; [ intros l c v [] | ]. split.
    { intros s allocated w Hin Hw.
      exact (no_location_holds_cells _ w (Hnone s allocated w Hin Hw)). }
    split; [ reflexivity | ]. split; [ apply Forall_nil | exact I ].
  - apply locations_all_bound_iff, bound_parts. cbv zeta.
    unfold initial_region_state.
    cbn [rs_state rs_frames rs_cells state_statics state_vars state_conts
         state_heap state_reads].
    split; [ split; [ apply Forall_nil | exact I ] | ].
    split; [ intros s allocated w [] | ]. split.
    { intros s allocated w Hin Hw l Hl.
      rewrite (Hnone s allocated w Hin Hw) in Hl. destruct Hl. }
    split; [ intros l c [] | ]. split; [ intros l c v [] | intros l [] ].
Qed.

Theorem region_step_keeps_invariant : region_step_keeps_invariant_statement.
Proof.
  intros rs rs' Hinv Hstep.
  apply region_invariant_parts in Hinv as (Hfresh & Hlive & Hcells & Hbound).
  apply region_invariant_parts.
  pose proof (proj1 (cells_live_parts rs) Hcells)
    as (_ & _ & Hholds & Hbelow & _).
  split;
    [ exact (region_step_keeps_symbols_fresh rs rs' Hfresh Hbelow Hstep) | ].
  split.
  { pose proof (region_step_live rs Hfresh Hlive Hholds) as H.
    rewrite Hstep in H. exact H. }
  split;
    [ exact (region_step_cells rs rs' Hfresh Hlive Hcells Hbound Hstep)
    | exact (region_step_keeps_bound rs rs' Hbound Hstep) ].
Qed.

Theorem region_run_keeps_invariant : region_run_keeps_invariant_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault] eqn:Hstep;
      try discriminate.
    apply IH. exact (region_step_keeps_invariant rs rs1 Hinv Hstep).
Qed.

Theorem live_holds_live : live_holds_live_statement.
Proof.
  intros rs s allocated Hinv Hlookup.
  apply region_invariant_parts in Hinv as (_ & Hlive & _).
  apply values_live_parts in Hlive as (L1 & L2 & _ & _).
  unfold live_statics in Hlookup. cbn [static_values] in Hlookup.
  rewrite lookup_app in Hlookup.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated' | ]
    eqn:Hlocals.
  - injection Hlookup as ->.
    pose proof (frames_hold_in rs _ s allocated L2 (lookup_in _ _ _ Hlocals))
      as Hp.
    rewrite allocated_placed_parts in Hp. apply andb_true_iff in Hp as [Hv Hs].
    rewrite allocated_holds_parts. apply andb_true_iff. split.
    + revert Hv. apply forallb_impl. intros w _.
      apply value_placed_holds; [ exact (near_live rs) | exact (heap_live rs) ].
    + revert Hs. apply forallb_impl. intros s' _. exact (near_live rs s').
  - pose proof
      (lookup_forallb (allocated_holds (heap_symbol rs)) s allocated _ L1
         Hlookup)
      as Hh.
    exact (allocated_holds_mono _ _ allocated (heap_live rs) Hh).
Qed.

(* A symbol of the heap is in no frame: the frames bind symbols that
   the machine drew and that the static environment does not bind. *)
Lemma heap_not_local (rs : region_state) (s : symbol) :
  symbols_fresh rs = true ->
  heap_symbol rs s = true ->
  not_local_symbol rs s = true.
Proof.
  intros Hfresh Hheap. unfold not_local_symbol.
  rewrite (heap_live rs s Hheap). simpl. unfold value_local.
  destruct (lookup s (live_locals (rs_frames rs))) as [allocated | ]
    eqn:Hlocals; [ | reflexivity ].
  apply lookup_in in Hlocals.
  unfold symbols_fresh in Hfresh. cbv zeta in Hfresh.
  apply andb_true_iff in Hfresh as [Hfresh F9].
  apply andb_true_iff in Hfresh as [Hfresh _].
  apply andb_true_iff in Hfresh as [_ F7].
  pose proof (forallb_in _ _ (s, allocated) F9 Hlocals) as Hunbound.
  cbv beta iota in Hunbound.
  apply in_flat_map in Hlocals as (f & Hf & Hin).
  pose proof (forallb_in _ _ f F7 Hf) as Hframe. unfold frame_fresh in Hframe.
  apply andb_true_iff in Hframe as [_ Hframe].
  pose proof (forallb_in _ _ (s, allocated) Hframe Hin) as Hdrawn.
  cbv beta iota in Hdrawn. apply andb_true_iff in Hdrawn as [Hdrawn _].
  unfold symbol_between in Hdrawn. apply andb_true_iff in Hdrawn as [Hdrawn _].
  unfold heap_symbol, drawn in Hheap. rewrite Hdrawn in Hheap.
  simpl in Hheap. rewrite Hheap in Hunbound. discriminate.
Qed.

Theorem heap_holds_no_local : heap_holds_no_local_statement.
Proof.
  intros rs s allocated Hinv Hlookup.
  apply region_invariant_parts in Hinv as (Hfresh & Hlive & _).
  apply values_live_parts in Hlive as (L1 & _ & _ & _).
  pose proof
    (lookup_forallb (allocated_holds (heap_symbol rs)) s allocated _ L1
       Hlookup)
    as Hh.
  exact
    (allocated_holds_mono _ _ allocated
       (fun s' => heap_not_local rs s' Hfresh) Hh).
Qed.

Theorem kept_holds_no_local : kept_holds_no_local_statement.
Proof.
  intros rs x v Hinv Hlookup.
  apply region_invariant_parts in Hinv as (Hfresh & Hlive & _).
  apply values_live_parts in Hlive as (_ & _ & L3 & _).
  pose proof (lookup_forallb (value_kept (heap_symbol rs)) x v _ L3 Hlookup)
    as Hk.
  exact (value_kept_mono _ _ v (fun s' => heap_not_local rs s' Hfresh) Hk).
Qed.

Theorem exit_values_kept : exit_values_kept_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' k vs Hinv; simpl;
    [ discriminate | ].
  destruct (region_step rs) as [rs1 | k1 vs1 rs1 | fault] eqn:Hstep;
    [ | | discriminate ].
  - apply IH. exact (region_step_keeps_invariant rs rs1 Hinv Hstep).
  - intros [= <- <- <-].
    apply region_invariant_parts in Hinv as (Hfresh & Hlive & Hcells & Hbound).
    pose proof (proj1 (cells_live_parts rs) Hcells) as (_ & _ & Hholds & _).
    pose proof (region_step_live rs Hfresh Hlive Hholds) as H.
    rewrite Hstep in H. destruct H as [Hlive1 Hvs].
    split.
    + apply region_invariant_parts.
      split;
        [ exact (region_step_exit_symbols_fresh rs k1 vs1 rs1 Hfresh Hstep) | ].
      split; [ exact Hlive1 | ].
      rewrite (region_step_exit rs rs1 k1 vs1 Hstep).
      split; [ exact (cells_live_count_uses rs Hcells) | exact Hbound ].
    + apply forallb_forall. intros v Hin. rewrite Forall_forall in Hvs.
      exact (value_kept_mono _ _ v (heap_live rs1) (Hvs v Hin)).
Qed.

Theorem gone_stays_gone_run : gone_stays_gone_run_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' s Hinv; simpl.
  - intros [= <-] Hgone. exact Hgone.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault] eqn:Hstep;
      try discriminate.
    intros Hrun Hgone.
    apply (IH rs1 rs' s (region_step_keeps_invariant rs rs1 Hinv Hstep) Hrun).
    apply region_invariant_parts in Hinv as (Hfresh & _).
    exact (gone_stays_gone_fresh rs rs1 s Hfresh Hstep Hgone).
Qed.

(* ======================================================================== *)
(* What the invariant says of cells                                         *)
(* ======================================================================== *)

Theorem dead_cell_stays_dead : dead_cell_stays_dead_statement.
Proof.
  intros rs rs' l Hinv Hstep.
  apply region_invariant_parts in Hinv as (Hfresh & _ & Hcells & _).
  apply cells_live_parts in Hcells as (_ & L2 & _).
  unfold cell_dead.
  destruct (lookup l (rs_cells rs)) as [m | ] eqn:Hl; [ | discriminate ].
  intros Hdead. apply negb_true_iff in Hdead.
  destruct (L2 l m (lookup_in _ _ _ Hl)) as (Hbound & _ & Hlt).
  (* The cell is listed after the step, with its mark. *)
  assert (Hl' : lookup l (rs_cells rs') = Some m).
  { destruct (region_step_listed rs rs' Hstep) as [-> | (m' & ->)];
      [ exact Hl | ].
    rewrite lookup_cons.
    destruct (Pos.eqb l (fresh_location (state_heap (rs_state rs)))) eqn:Heq;
      [ | exact Hl ].
    apply Pos.eqb_eq in Heq. unfold is_bound in Hbound.
    rewrite Heq,
      (FlambdaHeapProofs.fresh_location (state_heap (rs_state rs)))
      in Hbound.
    discriminate. }
  rewrite Hl'. apply negb_true_iff.
  (* And its mark does not come back: a mark that is pushed is the
     counter, which the mark of a cell is below. *)
  destruct (region_step_shape _ _ Hstep) as [_ Hshape].
  destruct Hshape
    as [Hframes _ _ _
       | bindings body _ Hframes _ _
       | kind Hframes _ _ _
       | above f Hframes _ _ _
       | v sym bindings _ _ Hframes _].
  - rewrite Hframes. exact Hdead.
  - rewrite Hframes. exact Hdead.
  - rewrite Hframes, on_stack_cons. cbn [frame_mark].
    rewrite Hdead, orb_false_r. apply Pos.eqb_neq. lia.
  - rewrite Hframes, on_stack_app, on_stack_cons in Hdead.
    apply orb_false_iff in Hdead as [_ Hdead].
    apply orb_false_iff in Hdead as [_ Hdead]. exact Hdead.
  - revert Hframes Hdead. destruct (rs_frames rs) as [ | f frames].
    + intros [-> _] _. reflexivity.
    + intros [-> _] Hdead. exact Hdead.
Qed.

Theorem live_cell_holds_usable : live_cell_holds_usable_statement.
Proof.
  intros rs l c v Hinv Hlookup Hdead Hin.
  apply region_invariant_parts in Hinv as (_ & _ & Hcells & _).
  apply cells_live_parts in Hcells as (_ & _ & L3 & _).
  exact
    (proj2
       (proj2
          (live_cell_holds rs l c v (L3 l c (lookup_in _ _ _ Hlookup)) Hdead
             Hin))).
Qed.

Theorem heap_cell_holds_no_local : heap_cell_holds_no_local_statement.
Proof.
  intros rs l c v Hinv Hlookup Hunlisted Hin.
  apply region_invariant_parts in Hinv as (_ & _ & Hcells & _).
  apply cells_live_parts in Hcells as (_ & _ & L3 & _).
  pose proof (L3 l c (lookup_in _ _ _ Hlookup)) as Hholds.
  unfold cell_holds in Hholds. rewrite Hunlisted in Hholds.
  exact (forallb_in _ _ v Hholds Hin).
Qed.

(* ======================================================================== *)
(* No poison of the kind [Region]                                           *)
(* ======================================================================== *)

(* The four statements on [no_poison], as FlambdaRegionsNoPoisonProofs.v
   proves them. *)
Theorem no_poison_initial : no_poison_initial_statement.
Proof. exact FlambdaRegionsNoPoisonProofs.no_poison_initial. Qed.

Theorem region_step_keeps_no_poison : region_step_keeps_no_poison_statement.
Proof. exact FlambdaRegionsNoPoisonProofs.region_step_keeps_no_poison. Qed.

Theorem region_run_keeps_no_poison : region_run_keeps_no_poison_statement.
Proof. exact FlambdaRegionsNoPoisonProofs.region_run_keeps_no_poison. Qed.

Theorem usable_not_region : usable_not_region_statement.
Proof. exact FlambdaRegionsNoPoisonProofs.usable_not_region. Qed.
