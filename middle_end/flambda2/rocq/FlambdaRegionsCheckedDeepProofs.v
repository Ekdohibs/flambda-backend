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
   what a symbol within a bound is bound to is within the bound; so what a
   named expression reads from a value that is as a lifetime says is as the
   lifetime says. Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaClosuresProofs FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs
  FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs FlambdaRegionsCheckedCallProofs.
Import ListNotations.

(* ======================================================================== *)
(* Parts                                                                    *)
(* ======================================================================== *)

Lemma value_all_parts (ok : symbol -> bool) (v w : value) :
  value_all ok v = true -> In w (value_parts v) -> value_all ok w = true.
Proof.
  destruct v
    as [c | code slots | slot decls slots | s | tag fields | elements | c | l
       | callee args args_arity];
    cbn [value_parts value_all]; try contradiction.
  - intros H Hin. apply in_map_iff in Hin as ([vslot w0] & <- & Hin).
    exact (forallb_in _ _ (vslot, w0) H Hin).
  - intros H Hin. apply in_map_iff in Hin as ([vslot w0] & <- & Hin).
    exact (forallb_in _ _ (vslot, w0) H Hin).
  - intros H Hin. exact (forallb_in _ _ w H Hin).
  - intros H Hin. exact (forallb_in _ _ w H Hin).
  - intros _ [<- | []]. reflexivity.
Qed.

Lemma allocated_all_parts (ok : symbol -> bool)
    (allocated : statically_allocated_value) :
  allocated_all ok allocated = true ->
  (∀ w : value,
     In w (FlambdaRegionsReadsProofs.allocated_values allocated) ->
     value_all ok w = true)
  ∧ (∀ s : symbol, In s (allocated_symbols allocated) -> ok s = true).
Proof.
  destruct allocated; cbn [allocated_all];
    cbn [FlambdaRegionsReadsProofs.allocated_values allocated_symbols];
    intros H.
  all: try (split; [ | intros s [] ]).
  all: try (intros w Hin; apply in_map_iff in Hin as ([vslot w0] & <- & Hin);
            exact (forallb_in _ _ (vslot, w0) H Hin)).
  all: try (intros w Hin; exact (forallb_in _ _ w H Hin)).
  all: try (intros w [<- | []]; exact H).
  all: try (intros w []).
  apply andb_true_iff in H as [Hsymbols Hslots]. split.
  - intros w Hin. apply in_map_iff in Hin as ([vslot w0] & <- & Hin).
    exact (forallb_in _ _ (vslot, w0) Hslots Hin).
  - intros s1 Hin. apply in_map_iff in Hin as ([fslot s0] & <- & Hin).
    exact (forallb_in _ _ (fslot, s0) Hsymbols Hin).
Qed.

(* ======================================================================== *)
(* The frames                                                               *)
(* ======================================================================== *)

(* What frames in order bind was drawn before what is above them. *)
Lemma locals_below_top (frames : list frame) :
  ∀ (top s : symbol) (allocated : statically_allocated_value),
    frames_ordered_below top frames = true ->
    In (s, allocated) (live_locals frames) ->
    (s < top)%positive.
Proof.
  induction frames as [ | f below IH]; intros top s allocated;
    [ contradiction | ].
  cbn [frames_ordered_below]. intros H.
  apply andb_true_iff in H as [H Hbelow].
  apply andb_true_iff in H as [Hmark Hlocals].
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below).
  intros Hin. apply in_app_iff in Hin as [Hin | Hin].
  - pose proof (forallb_in _ _ (s, allocated) Hlocals Hin) as H.
    cbv beta iota in H. apply andb_true_iff in H as [_ H].
    apply Pos.ltb_lt. exact H.
  - pose proof (IH (frame_mark f) s allocated Hbelow Hin).
    apply Pos.ltb_lt in Hmark. lia.
Qed.

Lemma local_in_cons (f : frame) (below : list frame) (s : symbol) :
  local_in below s = true -> local_in (f :: below) s = true.
Proof. exact (local_in_below [f] below s). Qed.

Lemma frames_deep_all (rs : region_state) (frames : list frame) :
  frames_deep rs frames = true ->
  ∀ (s : symbol) (allocated : statically_allocated_value),
    In (s, allocated) (live_locals frames) ->
    allocated_all (fun s' => heap_symbol rs s' || local_in frames s')
      allocated
    = true.
Proof.
  induction frames as [ | f below IH]; intros H s allocated Hin;
    [ contradiction | ].
  cbn [frames_deep] in H. apply andb_true_iff in H as [Hf Hbelow].
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below)
    in Hin.
  apply in_app_iff in Hin as [Hin | Hin].
  - exact (forallb_in _ _ (s, allocated) Hf Hin).
  - pose proof (IH Hbelow s allocated Hin) as Hall. revert Hall.
    apply allocated_all_mono. intros s' H.
    apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
    rewrite (local_in_cons f below s' H). apply orb_true_r.
Qed.

(* What a frame binds a symbol to that was drawn before a mark on the
   stack reaches symbols of the heap, or that frames bind and that
   were drawn before the mark. *)
Lemma deep_marked (rs : region_state) (frames : list frame) :
  ∀ (top : symbol) (b s : symbol) (allocated : statically_allocated_value),
    frames_ordered_below top frames = true ->
    frames_deep rs frames = true ->
    on_stack b frames = true ->
    In (s, allocated) (live_locals frames) ->
    (s < b)%positive ->
    allocated_all
      (fun s' => heap_symbol rs s' || (local_in frames s' && Pos.ltb s' b))
      allocated
    = true.
Proof.
  induction frames as [ | f below IH];
    intros top b s allocated Hordered Hdeep Hstack Hin Hlt; [ contradiction | ].
  cbn [frames_ordered_below] in Hordered.
  apply andb_true_iff in Hordered as [Hordered Hbelow_ordered].
  apply andb_true_iff in Hordered as [Hmark Hlocals].
  cbn [frames_deep] in Hdeep. apply andb_true_iff in Hdeep as [Hf Hbelow_deep].
  unfold on_stack in Hstack. cbn [existsb] in Hstack.
  change (live_locals (f :: below)) with (frame_locals f ++ live_locals below)
    in Hin.
  apply in_app_iff in Hin as [Hin | Hin].
  - exfalso.
    pose proof (forallb_in _ _ (s, allocated) Hlocals Hin) as H.
    cbv beta iota in H. apply andb_true_iff in H as [H _].
    apply Pos.ltb_lt in H.
    apply orb_true_iff in Hstack as [Hb | Hb].
    + apply Pos.eqb_eq in Hb. lia.
    + apply existsb_exists in Hb as (g & Hg & Hb). apply Pos.eqb_eq in Hb.
      pose proof
        (frames_ordered_below_marks below (frame_mark f) g Hbelow_ordered Hg).
      lia.
  - apply orb_true_iff in Hstack as [Hb | Hb].
    + apply Pos.eqb_eq in Hb.
      pose proof (frames_deep_all rs below Hbelow_deep s allocated Hin) as Hall.
      revert Hall. apply allocated_all_mono. intros s' H.
      apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
      rewrite (local_in_cons f below s' H). cbn [andb].
      unfold local_in in H.
      destruct (lookup s' (live_locals below)) as [a' | ] eqn:Hs';
        [ | discriminate ].
      pose proof (locals_below_top below (frame_mark f) s' a' Hbelow_ordered
                    (lookup_in _ _ _ Hs'))
        as Hlt'.
      assert (H' : Pos.ltb s' b = true) by (apply Pos.ltb_lt; lia).
      rewrite H'. apply orb_true_r.
    + pose proof (IH (frame_mark f) b s allocated Hbelow_ordered Hbelow_deep
                    Hb Hin Hlt)
        as Hall.
      revert Hall. apply allocated_all_mono. intros s' H.
      apply orb_true_iff in H as [-> | H]; [ reflexivity | ].
      apply andb_true_iff in H as [H1 H2].
      rewrite (local_in_cons f below s' H1), H2. apply orb_true_r.
Qed.

(* ======================================================================== *)
(* What is read from a value within a bound                                 *)
(* ======================================================================== *)

Section Deep.

Variable rs : region_state.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hordered : frames_ordered rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

(* What a symbol of the heap is bound to reaches symbols of the heap. *)
Lemma heap_bound (s : symbol) (allocated : statically_allocated_value) :
  heap_symbol rs s = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  allocated_all (heap_symbol rs) allocated = true.
Proof.
  intros Hheap Hlookup. unfold live_statics in Hlookup.
  cbn [static_values] in Hlookup. rewrite lookup_app in Hlookup.
  destruct (lookup s (live_locals (rs_frames rs))) as [local | ] eqn:Hlocal.
  - pose proof (heap_not_local rs s Hfresh Hheap) as Hnot.
    unfold not_local_symbol, value_local in Hnot. rewrite Hlocal in Hnot.
    cbn [is_some negb] in Hnot. rewrite andb_false_r in Hnot. discriminate.
  - exact (forallb_in _ _ (s, allocated) Hstatics (lookup_in _ _ _ Hlookup)).
Qed.

(* What a symbol within a bound is bound to is within the bound, the
   bound being a mark on the stack, or none. *)
Lemma within_bound (stop : option symbol) (s : symbol)
    (allocated : statically_allocated_value) :
  (∀ b : symbol, stop = Some b -> on_stack b (rs_frames rs) = true) ->
  symbol_within rs stop s = true ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  allocated_all (symbol_within rs stop) allocated = true.
Proof.
  intros Hstop Hs Hlookup.
  destruct (heap_symbol rs s) eqn:Hheap.
  - pose proof (heap_bound s allocated Hheap Hlookup) as Hall. revert Hall.
    apply allocated_all_mono. intros s' H. unfold symbol_within. rewrite H.
    reflexivity.
  - unfold symbol_within in Hs. rewrite Hheap in Hs. cbn [orb] in Hs.
    apply andb_true_iff in Hs as [Hlocal Hbefore].
    unfold live_statics in Hlookup. cbn [static_values] in Hlookup.
    rewrite lookup_app in Hlookup. unfold local_in in Hlocal.
    destruct (lookup s (live_locals (rs_frames rs))) as [local | ] eqn:Hs;
      [ | discriminate ].
    injection Hlookup as ->.
    destruct stop as [b | ].
    + cbn [before] in Hbefore. apply Pos.ltb_lt in Hbefore.
      exact (deep_marked rs (rs_frames rs) (rs_next rs) b s allocated Hordered
               Hdeep (Hstop b eq_refl) (lookup_in _ _ _ Hs) Hbefore).
    + pose proof (frames_deep_all rs (rs_frames rs) Hdeep s allocated
                    (lookup_in _ _ _ Hs))
        as Hall.
      revert Hall. apply allocated_all_mono. intros s' H.
      unfold symbol_within. cbn [before]. rewrite andb_true_r. exact H.
Qed.

Variable vars : env.

Variable opens : list variable.

Variable ms : list mark.

Hypothesis Hmarked : Forall2 (marked vars) opens ms.

Hypothesis Hplaced : marks_placed (rs_frames rs) ms = true.

Variable l : lifetime.

Local Abbreviation holds :=
  (lifetime_holds rs opens ms None l).

Lemma holds_const (c : const) : holds (Value_const c).
Proof. apply heap_holds. reflexivity. Qed.

Lemma holds_parts (v w : value) : holds v -> In w (value_parts v) -> holds w.
Proof.
  intros H Hin. destruct l as [ | | r | ]; cbn [lifetime_holds] in *.
  - exact (value_all_parts _ v w H Hin).
  - exact (value_all_parts _ v w H Hin).
  - destruct (region_place opens ms None r) as [[m stop] | ]; [ | exact I ].
    intros Hstack. exact (value_all_parts _ v w (H Hstack) Hin).
  - exact I.
Qed.

Lemma holds_move (slot slot' : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) :
  holds (Value_closure_of_set slot function_decls value_slots) ->
  holds (Value_closure_of_set slot' function_decls value_slots).
Proof. destruct l; intros H; exact H. Qed.

Lemma holds_static (s : symbol) (allocated : statically_allocated_value) :
  holds (Value_symbol s) ->
  lookup s (static_values (live_statics rs)) = Some allocated ->
  (∀ w : value,
     In w (FlambdaRegionsReadsProofs.allocated_values allocated) -> holds w)
  ∧ (∀ s' : symbol,
       In s' (allocated_symbols allocated) -> holds (Value_symbol s')).
Proof.
  pose proof (proj2 (proj1 (andb_true_iff _ _) Hplaced)) as Hdecrease.
  assert (Hbound :
    ∀ stop : option symbol,
      (∀ b : symbol, stop = Some b -> In b ms) ->
      symbol_within rs stop s = true ->
      lookup s (static_values (live_statics rs)) = Some allocated ->
      (∀ w : value,
         In w (FlambdaRegionsReadsProofs.allocated_values allocated) ->
         value_within rs stop w = true)
      ∧ (∀ s' : symbol,
           In s' (allocated_symbols allocated) ->
           symbol_within rs stop s' = true)).
  { intros stop Hstop Hs Hlookup.
    apply allocated_all_parts.
    apply (within_bound stop s allocated); [ | exact Hs | exact Hlookup ].
    intros b Hb. exact (marks_placed_in _ _ b Hplaced (Hstop b Hb)). }
  intros H Hlookup. destruct l as [ | | r | ]; cbn [lifetime_holds] in *.
  - apply allocated_all_parts. exact (heap_bound s allocated H Hlookup).
  - apply (Hbound _); [ | exact H | exact Hlookup ].
    unfold caller_bound. destruct (lowest ms) as [b0 | ] eqn:Hlowest;
      [ | discriminate ].
    intros b [= <-]. exact (proj1 (lowest_least ms b0 Hlowest Hdecrease)).
  - destruct (region_place opens ms None r) as [[m stop] | ] eqn:Hplace;
      [ | split; intros; exact I ].
    destruct (region_place_least vars opens ms r m stop Hmarked Hdecrease
                Hplace)
      as (_ & _ & Hstop).
    assert (Hin : ∀ b : symbol, stop = Some b -> In b ms).
    { intros b ->. exact (proj1 Hstop). }
    split.
    + intros w Hw Hstack.
      exact (proj1 (Hbound stop Hin (H Hstack) Hlookup) w Hw).
    + intros s' Hs' Hstack.
      exact (proj2 (Hbound stop Hin (H Hstack) Hlookup) s' Hs').
  - split; intros; exact I.
Qed.

(* What a named expression that does not allocate reads from values
   that are as the lifetime says is as the lifetime says. *)
Lemma holds_named (vars0 : env) (d : named) (v : value) :
  is_allocation d = false ->
  (∀ (s : simple) (w : value),
     In s (simples_of_named d) -> eval_simple vars0 s = Some w -> holds w) ->
  eval_named (live_statics rs) vars0 d = Some v ->
  holds v.
Proof.
  exact
    (eval_named_reads (lifetime_holds rs opens ms None l) (live_statics rs)
       (fun c _ => holds_const c) holds_parts holds_move holds_static vars0 d
       v).
Qed.

Lemma holds_unary (p : unary_primitive) (arg v : value) :
  (∀ kind : boxable_number, p <> Box_number kind) ->
  holds arg ->
  eval_unary_primitive (live_statics rs) p arg = Some v ->
  holds v.
Proof.
  exact
    (eval_unary_primitive_reads (lifetime_holds rs opens ms None l)
       (live_statics rs) (fun c _ => holds_const c) holds_parts holds_static p
       arg v).
Qed.

Lemma holds_binary (p : binary_primitive) (arg1 arg2 v : value) :
  holds arg1 ->
  eval_binary_primitive (live_statics rs) p arg1 arg2 = Some v ->
  holds v.
Proof.
  exact
    (eval_binary_primitive_reads (lifetime_holds rs opens ms None l)
       (live_statics rs) (fun c _ => holds_const c) holds_parts holds_static p
       arg1 arg2 v).
Qed.

End Deep.
