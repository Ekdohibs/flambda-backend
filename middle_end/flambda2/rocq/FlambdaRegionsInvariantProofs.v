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
   here needs review. The first half of the invariant, [symbols_fresh],
   is kept by FlambdaRegionsFreshProofs.v; here is the second,
   [values_live], and what follows of the two. The four statements on
   [no_poison] are proved in FlambdaRegionsNoPoisonProofs.v, and are
   stated again at the end of this file. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveModel FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsNeedsHeapProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsReadsProofs
  FlambdaRegionsStepsProofs FlambdaRegionsFreshProofs.
From OxCamlFlambda2 Require FlambdaRegionsNoPoisonProofs.
From OxCamlFlambda2 Require FlambdaRegionsNoHeapPrimitiveProofs.
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
  | Region_undefined _ | Region_let_not_run => True
  end.

Lemma plain_step_live (rs : region_state) :
  values_live rs = true ->
  needs_no_heap (rs_state rs) ->
  (∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
     (v : value),
    state_expr (rs_state rs) = Let x kind d body ->
    eval_named (live_statics rs) (state_vars (rs_state rs)) d = Some v ->
    kept rs v) ->
  live_result (plain_step rs).
Proof.
  intros Hlive Hneeds Hlet.
  pose proof Hlive as HL. apply values_live_parts in HL as (_ & _ & L3 & L4).
  pose proof
    (plain_step_values (kept rs) (kept_partial rs) rs (kept_fields rs Hlive)
       (proj1 (kept_vals rs (rs_state rs)) (conj L3 L4)) Hneeds Hlet
       (fun sim v _ => eval_simple_kept _ _ sim v L3))
    as Hvals.
  destruct (plain_step rs) as [rs' | k vs rs' | fault | ] eqn:Hstep;
    unfold region_result_vals_ok in Hvals; unfold live_result.
  - destruct (plain_step_step rs rs' Hstep) as (s' & _ & ->).
    apply (values_live_transfer rs); try reflexivity; assumption.
  - rewrite (plain_step_exit rs rs' k vs Hstep).
    exact (conj Hlive Hvals).
  - exact I.
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
  destruct (region_apply_cont rs ac) as [rs' | k vs rs' | fault | ] eqn:Hstep;
    unfold region_result_vals_ok in Hvals; unfold live_result.
  - destruct (region_apply_cont_step rs rs' ac Hstep)
      as (Hframes & Hbase & _ & Hstatics).
    exact (values_live_transfer rs rs' Hbase Hstatics Hframes Hlive Hvals).
  - rewrite (region_apply_cont_exit rs rs' ac k vs Hstep).
    exact (conj Hlive Hvals).
  - exact I.
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
  values_live rs = true ->
  forallb (simple_usable rs) (simples_of_named d) = true ->
  live_result (local_allocation rs x y kind d region body).
Proof.
  intros Hlive Husable.
  destruct (local_allocation rs x y kind d region body)
    as [rs' | k vs rs' | fault | ] eqn:Hstep; unfold live_result.
  2: { revert Hstep. unfold local_allocation. cbv zeta.
       destruct (is_allocation d && kind_eqb kind Value); [ | discriminate ].
       destruct (named_region rs region); [ | discriminate ].
       destruct (eval_named (live_statics rs) (state_vars (rs_state rs)) d)
         as [v | ]; [ | discriminate ].
       destruct (localise (rs_next rs) v) as [[sym bindings] | ];
         [ | discriminate ].
       destruct (rs_frames rs) as [ | [m mark_kind locals] frames];
         discriminate. }
  2, 3: exact I.
  destruct (local_allocation_step rs rs' x y kind d region body Hstep)
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
  destruct (static_step rs) as [rs' | k vs rs' | fault | ] eqn:Hstep;
    unfold live_result.
  2: { revert Hstep. unfold static_step.
       destruct (rs_state rs) as [e vars conts statics hp rd]. simpl in He.
       subst e. rewrite step_let_static.
       destruct (eval_static_bindings vars bindings statics); discriminate. }
  2, 3: exact I.
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
           (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs))
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
  live_result (region_step rs).
Proof.
  intros Hfresh Hlive.
  pose proof (values_live_count_uses rs Hlive) as Hlive1.
  pose proof (symbols_fresh_expr rs Hfresh) as Hbelow.
  apply region_step_cases.
  - intros fault. exact I.
  - intros x kind d body _ _ _. exact I.
  - intros x kind d mark_kind body _.
    exact (begin_region_live _ _ _ _ _ Hlive1).
  - intros x kind d ended_by region body _.
    exact (end_region_live _ _ _ _ _ _ Hlive1).
  - intros x kind d body y region body' _ _ _ Husable.
    exact (local_allocation_live _ _ _ _ _ _ _ Hlive1 Husable).
  - intros x kind d body He Hneeds Husable Hheap.
    apply (plain_step_live _ Hlive1).
    + intros x' kind' d' body' He'.
      change (state_expr (rs_state (count_uses rs)))
        with (state_expr (rs_state rs)) in He'.
      rewrite He in He'. injection He' as <- <- <- <-. exact Hneeds.
    + intros x' kind' d' body' v He' Heval.
      change (state_expr (rs_state (count_uses rs)))
        with (state_expr (rs_state rs)) in He'.
      rewrite He in He'. injection He' as <- <- <- <-.
      exact (eval_named_kept (count_uses rs) d v Hlive1 Husable Hheap Heval).
  - intros bindings body He Husable Hlocal.
    exact
      (static_step_live (count_uses rs) bindings body Hlive1 Hbelow He Husable
         Hlocal).
  - intros ac He. exact (region_apply_cont_live _ ac Hlive1 He).
  - intros Hnot_let _. apply (plain_step_live _ Hlive1).
    + intros x kind d body He. contradiction (Hnot_let x kind d body He).
    + intros x kind d body v He. contradiction (Hnot_let x kind d body He).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem region_invariant_initial : region_invariant_initial_statement.
Proof.
  intros e statics base Hfresh. unfold region_invariant.
  rewrite (symbols_fresh_initial e statics base Hfresh). simpl.
  apply values_live_parts. unfold initial_region_state.
  cbn [rs_state rs_frames state_statics state_vars state_conts].
  repeat split; try reflexivity.
  unfold region_names_fresh in Hfresh.
  apply andb_true_iff in Hfresh as [_ Hstatics].
  unfold statics_below in Hstatics.
  apply andb_true_iff in Hstatics as [_ Hvalues].
  revert Hvalues. apply forallb_impl. intros [s allocated] _ H.
  apply andb_true_iff in H as [_ H].
  rewrite allocated_below_parts in H. apply andb_true_iff in H as [Hv Hs].
  rewrite allocated_holds_parts. apply andb_true_iff. split.
  - revert Hv. apply forallb_impl. intros w _.
    apply value_below_holds. intros s'.
    exact (below_base_heap (initial_region_state e statics base) s').
  - revert Hs. apply forallb_impl. intros s' _.
    exact (below_base_heap (initial_region_state e statics base) s').
Qed.

Theorem region_step_keeps_invariant : region_step_keeps_invariant_statement.
Proof.
  intros rs rs' Hinv Hstep. unfold region_invariant in Hinv |- *.
  apply andb_true_iff in Hinv as [Hfresh Hlive].
  rewrite (region_step_keeps_symbols_fresh rs rs' Hfresh Hstep). simpl.
  pose proof (region_step_live rs Hfresh Hlive) as H.
  rewrite Hstep in H. exact H.
Qed.

Theorem region_run_keeps_invariant : region_run_keeps_invariant_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' Hinv; simpl.
  - intros [= <-]. exact Hinv.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault | ] eqn:Hstep;
      try discriminate.
    apply IH. exact (region_step_keeps_invariant rs rs1 Hinv Hstep).
Qed.

Theorem live_holds_live : live_holds_live_statement.
Proof.
  intros rs s allocated Hinv Hlookup. unfold region_invariant in Hinv.
  apply andb_true_iff in Hinv as [_ Hlive].
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
  intros rs s allocated Hinv Hlookup. unfold region_invariant in Hinv.
  apply andb_true_iff in Hinv as [Hfresh Hlive].
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
  intros rs x v Hinv Hlookup. unfold region_invariant in Hinv.
  apply andb_true_iff in Hinv as [Hfresh Hlive].
  apply values_live_parts in Hlive as (_ & _ & L3 & _).
  pose proof (lookup_forallb (value_kept (heap_symbol rs)) x v _ L3 Hlookup)
    as Hk.
  exact (value_kept_mono _ _ v (fun s' => heap_not_local rs s' Hfresh) Hk).
Qed.

Theorem exit_values_kept : exit_values_kept_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' k vs Hinv; simpl;
    [ discriminate | ].
  destruct (region_step rs) as [rs1 | k1 vs1 rs1 | fault | ] eqn:Hstep;
    [ | | discriminate | discriminate ].
  - apply IH. exact (region_step_keeps_invariant rs rs1 Hinv Hstep).
  - intros [= <- <- <-]. unfold region_invariant in Hinv |- *.
    apply andb_true_iff in Hinv as [Hfresh Hlive].
    pose proof (region_step_live rs Hfresh Hlive) as H.
    rewrite Hstep in H. destruct H as [Hlive1 Hvs].
    rewrite (region_step_exit_symbols_fresh rs k1 vs1 rs1 Hfresh Hstep),
      Hlive1.
    split; [ reflexivity | ].
    apply forallb_forall. intros v Hin. rewrite Forall_forall in Hvs.
    exact (value_kept_mono _ _ v (heap_live rs1) (Hvs v Hin)).
Qed.

Theorem gone_stays_gone_run : gone_stays_gone_run_statement.
Proof.
  intros fuel. induction fuel as [ | fuel IH]; intros rs rs' s Hinv; simpl.
  - intros [= <-] Hgone. exact Hgone.
  - destruct (region_step rs) as [rs1 | k vs rs1 | fault | ] eqn:Hstep;
      try discriminate.
    intros Hrun Hgone.
    apply (IH rs1 rs' s (region_step_keeps_invariant rs rs1 Hinv Hstep) Hrun).
    unfold region_invariant in Hinv.
    apply andb_true_iff in Hinv as [Hfresh _].
    exact (gone_stays_gone_fresh rs rs1 s Hfresh Hstep Hgone).
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

(* The statement on the terms with no primitive of the heap (decision
   D53), as FlambdaRegionsNoHeapPrimitiveProofs.v proves it. *)
Theorem no_heap_primitive_no_let_not_run :
  no_heap_primitive_no_let_not_run_statement.
Proof.
  exact FlambdaRegionsNoHeapPrimitiveProofs.no_heap_primitive_no_let_not_run.
Qed.
