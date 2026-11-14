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
   the steps that change the store (the beginning and the end of a region,
   a local allocation, a [Let_static]) keep the property. Nothing here
   needs review. *)

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
  FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCallsProofs FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec FlambdaWellFormedSpec FlambdaWellFormedProofs.
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

(* ======================================================================== *)
(* The lifetimes when a region is begun or ended                            *)
(* ======================================================================== *)

Lemma lowest_some (ms : list mark) :
  ∀ m : mark, ∃ b : mark, lowest (m :: ms) = Some b.
Proof.
  induction ms as [ | m1 ms IH]; intros m; [ exists m; reflexivity | ].
  destruct (IH m1) as (b & Hb). exists b. exact Hb.
Qed.

Lemma lowest_cons (m : mark) (ms : list mark) :
  lowest (m :: ms)
  = match lowest ms with
    | Some b => Some b
    | None => Some m
    end.
Proof.
  destruct ms as [ | m1 ms]; [ reflexivity | ].
  destruct (lowest_some ms m1) as (b & Hb). rewrite Hb. exact Hb.
Qed.

Lemma region_place_closed (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable),
    ~ In r opens -> region_place opens ms above r = None.
Proof.
  induction opens as [ | x opens IH]; intros [ | m ms] above r Hr;
    try reflexivity.
  cbn [region_place]. destruct (Pos.eqb x r) eqn:Hxr.
  - apply Pos.eqb_eq in Hxr. subst. contradiction Hr. left. reflexivity.
  - apply IH. intros H. apply Hr. right. exact H.
Qed.

(* A region is begun: what held with its mark as the bound of the
   newest region holds with the region open. *)
Lemma holds_push (rs : region_state) (opens : list variable)
    (ms : list mark) (x : variable) (m : mark) (l : lifetime) (v : value) :
  l <> Of_region x ->
  lifetime_holds rs opens ms (Some m) l v ->
  lifetime_holds rs (x :: opens) (m :: ms) None l v.
Proof.
  intros Hl. destruct l as [ | | r | ]; cbn [lifetime_holds];
    try exact (fun H => H).
  - unfold caller_bound. rewrite lowest_cons.
    destruct (lowest ms); exact (fun H => H).
  - cbn [region_place]. destruct (Pos.eqb x r) eqn:Hxr;
      [ | exact (fun H => H) ].
    apply Pos.eqb_eq in Hxr. subst. contradiction Hl. reflexivity.
Qed.

(* The newest region is no longer open. *)
Lemma holds_pop (rs : region_state) (opens : list variable) (ms : list mark)
    (x : variable) (m : mark) (above : option symbol) (l : lifetime)
    (v : value) :
  ~ In x opens ->
  lifetime_holds rs (x :: opens) (m :: ms) above l v ->
  lifetime_holds rs opens ms None l v.
Proof.
  intros Hx. destruct l as [ | | r | ]; cbn [lifetime_holds];
    try exact (fun H => H).
  - unfold caller_bound. rewrite lowest_cons.
    destruct (lowest ms) as [b | ]; [ exact (fun H => H) | ].
    apply within_later. intros b' H. discriminate H.
  - cbn [region_place]. destruct (Pos.eqb x r) eqn:Hxr.
    + apply Pos.eqb_eq in Hxr. subst r. intros _.
      rewrite (region_place_closed opens ms None x Hx). exact I.
    + rewrite (region_place_above opens ms (Some m) r).
      destruct (region_place opens ms None r) as [[m' [b | ]] | ];
        try exact (fun H => H).
      intros H Hon. apply (within_later rs (Some m) None);
        [ intros b' Hb; discriminate Hb | exact (H Hon) ].
Qed.

Lemma holds_empty (rs : region_state) (opens : list variable)
    (ms : list mark) (above : option symbol) (l : lifetime) (v : value) :
  lifetime_holds rs opens ms above l v ->
  lifetime_holds rs [] [] None l v.
Proof.
  destruct l as [ | | r | ]; cbn [lifetime_holds]; try exact (fun H => H).
  - apply within_later. intros b' H. discriminate H.
  - intros _. exact I.
Qed.

(* The marks of the open regions below [r]. *)
Fixpoint marks_below (r : variable) (opens : list variable) (ms : list mark)
    : list mark :=
  match opens, ms with
  | x :: opens, m :: ms => if Pos.eqb x r then ms else marks_below r opens ms
  | _, _ => []
  end.

Section Below.

Variable vars : env.

Variable r : variable.

Lemma marks_below_marked (opens : list variable) (ms : list mark) :
  Forall2 (marked vars) opens ms ->
  Forall2 (marked vars) (regions_below r opens) (marks_below r opens ms).
Proof.
  induction 1 as [ | x m opens ms Hx Hrest IH]; [ constructor | ].
  cbn [regions_below marks_below].
  destruct (Pos.eqb x r); [ exact Hrest | exact IH ].
Qed.

Lemma marks_below_in (opens : list variable) :
  ∀ (ms : list mark) (m' : mark),
    In m' (marks_below r opens ms) -> In m' ms.
Proof.
  induction opens as [ | x opens IH]; intros [ | m ms] m';
    cbn [marks_below]; try contradiction.
  destruct (Pos.eqb x r); intros H; right; [ exact H | exact (IH ms m' H) ].
Qed.

Lemma marks_below_decrease (opens : list variable) :
  ∀ ms : list mark,
    marks_decrease ms = true ->
    marks_decrease (marks_below r opens ms) = true.
Proof.
  induction opens as [ | x opens IH]; intros [ | m ms] Hdecrease;
    cbn [marks_below]; try reflexivity.
  pose proof (marks_decrease_tail m ms Hdecrease) as Htail.
  destruct (Pos.eqb x r); [ exact Htail | exact (IH ms Htail) ].
Qed.

Lemma marks_below_lt (opens : list variable) (ms : list mark) (m : mark) :
  Forall2 (marked vars) opens ms ->
  marks_decrease ms = true ->
  marked vars r m ->
  ∀ m' : mark, In m' (marks_below r opens ms) -> (m' < m)%positive.
Proof.
  intros Hmarked. revert m.
  induction Hmarked as [ | x m0 opens ms Hx Hrest IH]; intros m Hdecrease Hr m';
    cbn [marks_below]; [ contradiction | ].
  destruct (Pos.eqb x r) eqn:Hxr.
  - apply Pos.eqb_eq in Hxr. subst x. unfold marked in Hx, Hr.
    rewrite Hx in Hr. injection Hr as <-.
    exact (marks_decrease_lt m0 ms Hdecrease m').
  - exact (IH m (marks_decrease_tail m0 ms Hdecrease) Hr m').
Qed.

Lemma regions_below_among (opens : list variable) (r' : variable) :
  among r' (regions_below r opens) = true -> among r' opens = true.
Proof.
  induction opens as [ | x opens IH]; cbn [regions_below]; [ discriminate | ].
  rewrite among_cons. destruct (Pos.eqb x r); intros H.
  - rewrite H. apply orb_true_r.
  - rewrite (IH H). apply orb_true_r.
Qed.

(* The regions down to [r] are ended. *)
Lemma holds_below (rs : region_state) (l : lifetime) (v : value)
    (opens : list variable) (ms : list mark) :
  Forall2 (marked vars) opens ms ->
  NoDup opens ->
  ∀ above : option symbol,
    lifetime_holds rs opens ms above l v ->
    lifetime_holds rs (regions_below r opens) (marks_below r opens ms) None l
      v.
Proof.
  induction 1 as [ | x m opens ms Hx Hrest IH]; intros Hnodup above Hholds.
  - exact (holds_empty rs [] [] above l v Hholds).
  - inversion Hnodup as [ | x0 opens0 Hnot Hnodup']; subst.
    pose proof (holds_pop rs opens ms x m above l v Hnot Hholds) as Hpopped.
    cbn [regions_below marks_below].
    destruct (Pos.eqb x r); [ exact Hpopped | exact (IH Hnodup' None Hpopped) ].
Qed.

End Below.

(* ======================================================================== *)
(* What the frames bind                                                     *)
(* ======================================================================== *)

Lemma frames_deep_app (rs : region_state) (above rest : list frame) :
  frames_deep rs (above ++ rest) = true -> frames_deep rs rest = true.
Proof.
  induction above as [ | f above IH]; [ exact (fun H => H) | ].
  cbn [app frames_deep]. intros H. apply andb_true_iff in H as [_ H].
  exact (IH H).
Qed.

Lemma frames_deep_cells_app (rs : region_state) (above rest : list frame) :
  frames_deep_cells rs (above ++ rest) = true ->
  frames_deep_cells rs rest = true.
Proof.
  induction above as [ | f above IH]; [ exact (fun H => H) | ].
  cbn [app frames_deep_cells]. intros H. apply andb_true_iff in H as [_ H].
  exact (IH H).
Qed.

Lemma cells_deep_build (rs : region_state) :
  frames_deep_cells rs (rs_frames rs) = true ->
  forallb
    (fun '(_, allocated) =>
       forallb (value_all_cells (heap_location rs))
         (values_of_static allocated))
    (static_values (state_statics (rs_state rs)))
  = true ->
  forallb (fun '(l, c) => cell_deep rs l c) (state_heap (rs_state rs))
  = true ->
  cells_deep rs = true.
Proof.
  intros H1 H2 H3. unfold cells_deep. rewrite H1, H2, H3. reflexivity.
Qed.

(* What a lifetime says of what is read from a value follows what it
   says of a value, on the same heap. *)
Lemma contents_hold_impl (rs rs' : region_state)
    (opens opens' : list variable) (ms ms' : list mark)
    (above above' : option symbol) (l l' : lifetime) (v : value) :
  state_heap (rs_state rs') = state_heap (rs_state rs) ->
  (∀ w : value,
     lifetime_holds rs opens ms above l w ->
     lifetime_holds rs' opens' ms' above' l' w) ->
  contents_hold rs opens ms above l v ->
  contents_hold rs' opens' ms' above' l' v.
Proof.
  intros Hheap Himpl.
  destruct v; cbn [contents_hold]; try exact (Himpl _).
  rewrite Hheap. intros H c w Hc Hw. exact (Himpl w (H c w Hc Hw)).
Qed.

(* A symbol that a frame binds was drawn. *)
Lemma local_drawn (rs : region_state) (s : symbol)
    (allocated : statically_allocated_value) :
  symbols_fresh rs = true ->
  In (s, allocated) (live_locals (rs_frames rs)) ->
  drawn rs s = true.
Proof.
  intros Hfresh Hin.
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & _ & _ & _ & Hframes & _).
  unfold live_locals in Hin. apply in_flat_map in Hin as (f & Hf & Hin).
  destruct (Hframes f Hf) as (_ & _ & Hlocals).
  destruct (Hlocals s allocated Hin) as (Hle & _).
  unfold drawn. apply Pos.leb_le. exact Hle.
Qed.

(* ======================================================================== *)
(* Putting the state together                                               *)
(* ======================================================================== *)

Section Moved.

Variable cert : certificate.

Variable codes : list (code_id * code).

(* The values that the state holds after a step that moves the store:
   new variables in front of those that were, the continuations that
   were, and what the store binds. *)
Lemma moved_values (rs rs' : region_state) (new_vars : env) :
  region_invariant rs = true ->
  store_moves rs rs' ->
  state_values (good cert codes rs) rs ->
  state_vars (rs_state rs') = new_vars ++ state_vars (rs_state rs) ->
  env_values_all (good cert codes rs') new_vars ->
  state_conts (rs_state rs') = state_conts (rs_state rs) ->
  (∀ (s : symbol) (allocated : statically_allocated_value),
     In (s, allocated) (static_values (live_statics rs')) ->
     In (s, allocated) (static_values (live_statics rs))
     ∨ allocated_values (good cert codes rs') allocated) ->
  (∀ (l : location) (c : cell),
     In (l, c) (state_heap (rs_state rs')) ->
     In (l, c) (state_heap (rs_state rs))
     ∨ Forall (good cert codes rs') (values_of_cell c)) ->
  state_values (good cert codes rs') rs'.
Proof.
  intros Hinv Hmoves (Hgood_vars & Hgood_conts & Hgood_statics & Hgood_heap)
    Hvars Hnew Hconts Hstatics Hheap.
  destruct (state_old rs Hinv)
    as (Hold_vars & Hold_conts & Hold_statics & Hold_heap).
  unfold state_values. cbv zeta. rewrite Hvars, Hconts.
  split; [ | split; [ | split ] ].
  - apply Forall_app. split; [ exact Hnew | ].
    exact (env_good_moves cert codes rs rs' Hmoves _ Hgood_vars Hold_vars).
  - exact (conts_good_moves cert codes rs rs' Hmoves _ Hgood_conts
             Hold_conts).
  - apply Forall_forall. intros [s allocated] Hin. cbn [snd].
    destruct (Hstatics s allocated Hin) as [Hold | Hgood]; [ | exact Hgood ].
    apply (allocated_good_moves cert codes rs rs' Hmoves allocated
             (Hold_statics s allocated Hold)).
    exact (proj1 (Forall_forall _ _) Hgood_statics (s, allocated) Hold).
  - apply Forall_forall. intros [l c] Hin. cbn [snd].
    destruct (Hheap l c Hin) as [Hold | Hgood]; [ | exact Hgood ].
    exact
      (proj1 (Forall_forall _ _)
         (heap_good_moves cert codes rs rs' _ Hmoves Hold_heap Hgood_heap)
         (l, c) Hold).
Qed.

Lemma moved_checked (rs' : region_state) (cs' : check_state)
    (ms' : list mark) :
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  code_checked cert codes rs' ->
  local_closures_fit cert rs' = true ->
  frames_deep rs' (rs_frames rs') = true ->
  statics_deep rs' = true ->
  cells_deep rs' = true ->
  state_values (good cert codes rs') rs' ->
  check_expr cert codes cs' (state_expr (rs_state rs')) = true ->
  incl (code_of_expr (state_expr (rs_state rs'))) codes ->
  running_checked rs' cs' ms' ->
  conts_checked cert codes rs' (running_user rs' cs' ms') (cs_conts cs')
    (state_conts (rs_state rs')) ->
  state_checked cert codes rs'.
Proof.
  intros Hinv Hordered Hcode Hlocal Hdeep Hstatics Hcells Hgood Hcheck Hincl
    Hrun Hconts.
  apply good_split in Hgood as [Hplain Hpaps].
  split; [ exact Hinv | ]. split; [ exact Hordered | ].
  split; [ exact Hcode | ]. split; [ exact Hlocal | ].
  split; [ exact Hdeep | ]. split; [ exact Hstatics | ].
  split; [ exact Hplain | ]. split; [ exact Hpaps | ].
  split; [ exact Hcells | ].
  exists cs', ms'. split; [ exact Hcheck | ]. split; [ exact Hincl | ].
  split; [ exact Hrun | exact Hconts ].
Qed.

End Moved.

(* ======================================================================== *)
(* What a local allocation binds                                            *)
(* ======================================================================== *)

#[local] Transparent eval_prim.

Lemma allocated_all_of_parts (ok : symbol -> bool)
    (allocated : statically_allocated_value) :
  (∀ w : value,
     In w (FlambdaRegionsReadsProofs.allocated_values allocated) ->
     value_all ok w = true) ->
  (∀ s : symbol, In s (allocated_symbols allocated) -> ok s = true) ->
  allocated_all ok allocated = true.
Proof.
  destruct allocated
    as [code slots | slot symbols decls slots | tag fields | contents | bytes
       | elements | c];
    cbn [allocated_all FlambdaRegionsReadsProofs.allocated_values
           allocated_symbols];
    intros Hvalues Hsymbols; try reflexivity.
  - apply forallb_forall. intros [vslot w] Hin.
    exact (Hvalues w (in_map snd _ (vslot, w) Hin)).
  - apply andb_true_iff. split.
    + apply forallb_forall. intros [fslot s] Hin.
      exact (Hsymbols s (in_map snd _ (fslot, s) Hin)).
    + apply forallb_forall. intros [vslot w] Hin.
      exact (Hvalues w (in_map snd _ (vslot, w) Hin)).
  - apply forallb_forall. intros w Hin. exact (Hvalues w Hin).
  - exact (Hvalues contents (or_introl eq_refl)).
  - apply forallb_forall. intros w Hin. exact (Hvalues w Hin).
Qed.

Lemma bound_is_some {A : Type} (s : positive) (a : A)
    (bindings : list (positive * A)) :
  In (s, a) bindings -> is_some (lookup s bindings) = true.
Proof.
  intros Hin. destruct (lookup s bindings) eqn:Hlookup; [ reflexivity | ].
  exfalso. apply (lookup_none_not_in s bindings Hlookup).
  exact (in_map fst bindings (s, a) Hin).
Qed.

(* The closures that a local allocation binds are of code whose
   closures may be local. *)
Lemma new_closures_fit (cert : certificate) (statics : static_env)
    (vars : env) (d : named) (v : value) (next sym : symbol)
    (bindings : list (symbol * statically_allocated_value)) :
  is_allocation d = true ->
  eval_named statics vars d = Some v ->
  localise next v = Some (sym, bindings) ->
  forallb (closure_may_be_local cert) (closure_code d) = true ->
  forallb
    (fun '(_, allocated) =>
       match allocated with
       | Static_closure code _ => closure_may_be_local cert code
       | Static_closure_of_set _ _ function_decls _ =>
         forallb (fun '(_, code) => closure_may_be_local cert code)
           function_decls
       | _ => true
       end)
    bindings
  = true.
Proof.
  destruct d as [s | [u | b | p | hp | rp] args | code slots
                | slot decls slots | | | ];
    try discriminate; cbn [is_allocation eval_named closure_code].
  - destruct u; try discriminate. intros _.
    destruct (eval_simples vars args) as [[ | w [ | w' vs]] | ]; cbn;
      try discriminate.
    destruct w; try discriminate.
    unfold eval_prim, eval_unary_primitive.
    destruct (is_number_of_kind kind c); [ | discriminate ].
    intros [= <-]. cbn [localise]. intros [= <- <-] _. reflexivity.
  - intros _. destruct (eval_simples vars args) as [vs | ]; cbn;
      [ | discriminate ].
    unfold eval_prim, eval_variadic_primitive.
    destruct p as [[tag] | kind]; cbn.
    + destruct (_ && _); [ | discriminate ]. intros [= <-]. cbn [localise].
      intros [= <- <-] _. reflexivity.
    + destruct (array_made kind vs) as [elements | ]; cbn; [ | discriminate ].
      destruct (forallb _ vs); [ | discriminate ]. intros [= <-].
      cbn [localise]. intros [= <- <-] _. reflexivity.
  - intros _. destruct (eval_value_slots vars slots) as [vs | ]; cbn;
      [ | discriminate ].
    intros [= <-]. cbn [localise]. intros [= <- <-] Hcode.
    cbn [forallb] in Hcode |- *. exact Hcode.
  - intros _. destruct (lookup slot decls) as [code | ] eqn:Hslot; cbn;
      [ | discriminate ].
    destruct (eval_value_slots vars slots) as [vs | ]; cbn; [ | discriminate ].
    intros [= <-]. cbn [localise].
    destruct (lookup slot (number_slots decls next)) as [s | ]; cbn;
      [ | discriminate ].
    intros [= <- <-] Hcode. apply forallb_forall. intros [s0 allocated] Hin.
    apply in_map_iff in Hin as ([slot' s'] & [= <- <-] & _).
    apply forallb_forall. intros [fslot code'] Hin'.
    exact (forallb_in _ _ code' Hcode (in_map snd _ (fslot, code') Hin')).
Qed.

(* A value that may be evaluated is as the lifetime of the newest
   region says. *)
Lemma newest_live (rs : region_state) (vars : env) (cs : check_state)
    (ms : list mark) (v : value) :
  Forall2 (marked vars) (cs_open cs) ms ->
  value_within rs None v = true ->
  lifetime_holds rs (cs_open cs) ms None (newest_lifetime cs) v.
Proof.
  intros Hmarked Hlive. unfold newest_lifetime.
  destruct (cs_open cs) as [ | r opens].
  - inversion Hmarked. exact Hlive.
  - inversion Hmarked as [ | r0 m opens0 ms1 Hr Hrest]; subst.
    cbn [lifetime_holds region_place]. rewrite Pos.eqb_refl. intros _.
    exact Hlive.
Qed.

Section Store.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Variable cs : check_state.

Variable ms : list mark.

Hypothesis Hinv : region_invariant rs = true.

Hypothesis Hfresh : symbols_fresh rs = true.

Hypothesis Hordered : frames_ordered rs = true.

Hypothesis Hcode : code_checked cert codes rs.

Hypothesis Hlocal : local_closures_fit cert rs = true.

Hypothesis Hdeep : frames_deep rs (rs_frames rs) = true.

Hypothesis Hstatics : statics_deep rs = true.

Hypothesis Hcells : cells_deep rs = true.

Hypothesis Hgood : state_values (good cert codes rs) rs.

Hypothesis Hincl : incl (code_of_expr (state_expr (rs_state rs))) codes.

Hypothesis Hrun : running_checked rs cs ms.

Hypothesis Hconts :
  conts_checked cert codes rs (running_user rs cs ms) (cs_conts cs)
    (state_conts (rs_state rs)).

Local Abbreviation vars := (state_vars (rs_state rs)).

Local Abbreviation conts := (state_conts (rs_state rs)).

Lemma region_not_bound (x r : variable) :
  not_a_region cs x = true ->
  among r (cs_regions cs) = true ->
  Pos.eqb r x = false.
Proof.
  intros Hx Hr. unfold not_a_region in Hx. apply negb_true_iff in Hx.
  destruct (Pos.eqb r x) eqn:Hrx; [ | reflexivity ].
  apply Pos.eqb_eq in Hrx. subst r. rewrite Hx in Hr. discriminate.
Qed.

(* A variable that is no region is bound: a later user. *)
Lemma bound_extends (x : variable) (v : value) (regions : list variable)
    (ms' : list mark) (next' : symbol) :
  not_a_region cs x = true ->
  (∀ r : variable,
     among r (cs_regions cs) = true -> among r regions = true) ->
  (∀ m : mark, In m ms' -> In m ms ∨ (rs_next rs <= m)%positive) ->
  (rs_next rs <= next')%positive ->
  user_extends (running_user rs cs ms)
    (User ((x, v) :: vars) regions ms' next').
Proof.
  intros Hx Hregions Hmarks Hnext. split; [ | split ].
  - intros r Hr. split; [ | exact (Hregions r Hr) ].
    change (lookup r ((x, v) :: vars) = lookup r vars).
    rewrite lookup_cons, (region_not_bound x r Hx Hr). reflexivity.
  - exact Hmarks.
  - exact Hnext.
Qed.

(* The heap binds every location that the state holds. *)
Lemma store_closed :
  env_values_all (closed rs) vars
  ∧ conts_values_all (closed rs) conts
  ∧ heap_values_all (closed rs) (state_heap (rs_state rs)).
Proof.
  destruct (state_closed rs Hinv) as (H1 & H2 & H3 & _).
  split; [ exact H1 | ]. split; [ exact H2 | exact H3 ].
Qed.

Lemma var_closed (x : variable) (v : value) :
  lookup x vars = Some v -> closed rs v.
Proof.
  exact (env_values_all_lookup (closed rs) vars x v (proj1 store_closed)).
Qed.

Lemma simple_closed (s : simple) (v : value) :
  eval_simple vars s = Some v -> closed rs v.
Proof.
  exact
    (eval_simple_closed (state_heap (rs_state rs)) vars s v
       (proj1 store_closed)).
Qed.

Lemma simple_old (s : simple) (v : value) :
  simple_below (rs_base rs) s = true ->
  eval_simple vars s = Some v ->
  old rs v.
Proof.
  intros Hbelow Hv. split; [ | exact (simple_closed s v Hv) ].
  destruct (state_below rs Hfresh) as (Hbelow_vars & _ & _).
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & _ & Hbase_next & _).
  exact
    (eval_simple_below (rs_base rs) (rs_next rs) vars s v Hbase_next Hbelow
       Hbelow_vars Hv).
Qed.

Lemma running_below_next :
  ∀ m : mark, In m ms -> (m < rs_next rs)%positive.
Proof.
  destruct Hrun as (_ & Hplaced & _). intros m Hin.
  exact (forallb_ltb_in _ _ m (placed_below rs ms Hfresh Hplaced) Hin).
Qed.

(* A cell that a local allocation made is bound in the heap, and its
   mark was drawn. *)
Lemma listed_mark (l : location) (m : mark) :
  lookup l (rs_cells rs) = Some m ->
  is_bound l (state_heap (rs_state rs)) = true
  ∧ (rs_base rs <= m)%positive
  ∧ (m < rs_next rs)%positive.
Proof.
  intros Hl. pose proof Hinv as H.
  apply region_invariant_parts in H as (_ & _ & H & _).
  exact
    (proj1 (proj2 (proj1 (cells_live_parts rs) H)) l m (lookup_in _ _ _ Hl)).
Qed.

(* The store grows: what the variables hold is as it was. *)
Lemma grown_holds (rs' : region_state) (above : option symbol)
    (x : variable) (l : lifetime) (v : value) :
  store_grows rs rs' ->
  lookup x vars = Some v ->
  lifetime_holds rs (cs_open cs) ms above l v ->
  lifetime_holds rs' (cs_open cs) ms above l v.
Proof.
  intros Hgrows Hv.
  exact
    (grows_lifetime rs rs' Hgrows _ _ _ _ _ (var_closed x v Hv)
       running_below_next).
Qed.

Lemma grown_contents (rs' : region_state) (above : option symbol)
    (x : variable) (l : lifetime) (v : value) :
  store_grows rs rs' ->
  lookup x vars = Some v ->
  contents_hold rs (cs_open cs) ms above l v ->
  contents_hold rs' (cs_open cs) ms above l v.
Proof.
  intros Hgrows Hv.
  exact
    (grows_contents rs rs' Hgrows _ _ _ _ _ (proj2 (proj2 store_closed))
       (var_closed x v Hv) running_below_next).
Qed.

Lemma grown_conts (rs' : region_state) (u' : user) :
  store_grows rs rs' ->
  user_extends (running_user rs cs ms) u' ->
  conts_checked cert codes rs' u' (cs_conts cs) conts.
Proof.
  intros Hgrows Hextends. destruct store_closed as (_ & Hc & Hh).
  exact
    (conts_grow cert codes rs rs' (running_user rs cs ms) u' _ _ Hgrows Hh Hc
       (Pos.le_refl _) Hextends Hconts).
Qed.

(* ======================================================================== *)
(* The beginning of a region                                                *)
(* ======================================================================== *)

Lemma begin_checked (x : variable) (kind : flambda_kind)
    (mark_kind : mark_kind) (d : named) (body : expr) (rs' : region_state) :
  state_expr (rs_state rs) = Let x kind d body ->
  not_a_region cs x = true ->
  check_expr cert codes
    (Check_state ((x, Gone) :: cs_env cs) (x :: cs_open cs)
       (x :: cs_regions cs) (cs_conts cs) ((x, Gone) :: cs_contents cs))
    body
  = true ->
  begin_region rs x kind mark_kind body = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hx Hbody Hstep Hinv' Hordered'.
  pose proof (begin_region_step rs rs' x kind mark_kind body Hstep) as Heq.
  subst rs'.
  set (m := rs_next rs) in *.
  set (rs' :=
    bind_let rs x (Value_symbol m) body (state_statics (rs_state rs))
      (Frame m mark_kind [] :: rs_frames rs) (Pos.succ m) (rs_counts rs))
    in *.
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (placed_below rs ms Hfresh Hplaced) as Hbelow_next.
  pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
  destruct (scoped_parts cs Hscoped) as [Hs_env Hs_contents].
  pose proof Hscoped as Hscoped'.
  apply andb_true_iff in Hscoped' as [Hs1 Hs2].
  assert (Hgrows : store_grows rs rs').
  { constructor.
    - change (m <= Pos.succ m)%positive. lia.
    - intros s H. exact H.
    - intros s H. exact H.
    - intros m0 Hlt.
      change
        (Pos.eqb m m0 || on_stack m0 (rs_frames rs)
         = on_stack m0 (rs_frames rs)).
      destruct (Pos.eqb m m0) eqn:Heq; [ | reflexivity ].
      apply Pos.eqb_eq in Heq. subst m0. unfold m in Hlt. lia.
    - intros m0 H.
      change (Pos.eqb m m0 || on_stack m0 (rs_frames rs) = true).
      rewrite H. apply orb_true_r.
    - intros l _. reflexivity.
    - intros l c' w _ Hc' Hw. left. exists c'.
      split; [ exact Hc' | exact Hw ]. }
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - intros s H. exact H.
    - intros s _.
      change (static_values (live_statics rs'))
        with (static_values (live_statics rs)).
      destruct (lookup s (static_values (live_statics rs))) eqn:Hs;
        [ left; reflexivity | intros _; reflexivity ].
    - intros l _ H. exact H. }
  apply (moved_checked cert codes rs'
           (Check_state ((x, Gone) :: cs_env cs) (x :: cs_open cs)
              (x :: cs_regions cs) (cs_conts cs)
              ((x, Gone) :: cs_contents cs))
           (m :: ms) Hinv' Hordered').
  - exact Hcode.
  - exact Hlocal.
  - change (frames_deep rs' (Frame m mark_kind [] :: rs_frames rs) = true).
    cbn [frames_deep frame_locals forallb andb].
    exact (frames_deep_mono rs rs' _ (fun s H => H) Hdeep).
  - exact Hstatics.
  - destruct (cells_deep_parts rs Hcells)
      as (Hc_frames & Hc_statics & Hc_heap).
    apply cells_deep_build.
    + change
        (frames_deep_cells rs' (Frame m mark_kind [] :: rs_frames rs) = true).
      cbn [frames_deep_cells frame_locals forallb andb].
      exact
        (frames_deep_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
           Hc_frames).
    + exact
        (statics_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
           Hc_statics).
    + change
        (forallb (fun '(l, c) => cell_deep rs' l c)
           (state_heap (rs_state rs))
         = true).
      apply forallb_forall. intros [l c] Hin.
      apply (cell_deep_same rs rs' l c eq_refl (fun _ _ _ _ => eq_refl)
               (fun s H => H));
        [ | exact (forallb_in _ _ (l, c) Hc_heap Hin) ].
      intros m0 fr' Hm0 Hfrom.
      change
        (frames_from m0 (Frame m mark_kind [] :: rs_frames rs) = Some fr')
        in Hfrom.
      cbn [frames_from frame_mark] in Hfrom.
      destruct (Pos.eqb m m0) eqn:Hmm0.
      * exfalso. apply Pos.eqb_eq in Hmm0. subst m0.
        destruct (listed_mark l m Hm0) as (_ & _ & Hlt). unfold m in Hlt.
        lia.
      * exists fr'. split; [ exact Hfrom | ].
        split; intros ? H; exact H.
  - apply (moved_values cert codes rs rs' [(x, Value_symbol m)] Hinv Hmoves
             Hgood).
    + reflexivity.
    + apply Forall_cons; [ split; reflexivity | apply Forall_nil ].
    + reflexivity.
    + intros s allocated Hin. left. exact Hin.
    + intros l c Hin. left. exact Hin.
  - exact Hbody.
  - pose proof Hincl as Hi. rewrite He in Hi. exact Hi.
  - split; [ | split; [ | split; [ | split ] ] ].
    + change
        (open_marks ((x, Value_symbol m) :: vars) (x :: cs_open cs)
         = Some (m :: ms)).
      cbn [open_marks]. rewrite lookup_head.
      rewrite (open_marks_bind vars x _ _ Hx_open), Hopen. reflexivity.
    + change
        (marks_placed (Frame m mark_kind [] :: rs_frames rs) (m :: ms) = true).
      unfold marks_placed. apply andb_true_iff. split.
      * cbn [forallb]. apply andb_true_iff. split.
        -- cbn [on_stack existsb frame_mark]. rewrite Pos.eqb_refl.
           reflexivity.
        -- apply forallb_forall. intros m0 Hin.
           change (on_stack m0 (rs_frames rs') = true).
           rewrite
             (grows_stack rs rs' Hgrows m0
                (forallb_ltb_in _ _ m0 Hbelow_next Hin)).
           exact (marks_placed_in _ _ m0 Hplaced Hin).
      * apply marks_decrease_cons. split.
        -- intros m' Hin. exact (forallb_ltb_in _ _ m' Hbelow_next Hin).
        -- exact (placed_decrease _ _ Hplaced).
    + unfold check_state_scoped. cbn [cs_open cs_regions cs_env cs_contents].
      apply andb_true_iff. split.
      * apply all_among_iff. intros name. rewrite !among_cons.
        destruct (Pos.eqb name x); [ reflexivity | ]. cbn [orb].
        exact (all_among_among _ _ name Hs1).
      * rewrite forallb_app in Hs2 |- *.
        apply andb_true_iff in Hs2 as [Hs2e Hs2c].
        cbn [forallb andb].
        apply andb_true_iff;
          split; [ revert Hs2e | revert Hs2c ]; apply forallb_impl;
          intros [y [ | | r | ]] _; try (intros _; reflexivity);
          intros H; rewrite among_cons, H; apply orb_true_r.
    + intros y l v.
      change
        (lookup y ((x, Gone) :: cs_env cs) = Some l ->
         lookup y ((x, Value_symbol m) :: vars) = Some v ->
         lifetime_holds rs' (x :: cs_open cs) (m :: ms) None l v).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x); [ intros [= <-] _; exact I | ].
      intros Hl Hv. apply holds_push.
      * intros ->.
        pose proof
          (lookup_forallb
             (fun l =>
                match l with
                | Of_region r => among r (cs_regions cs)
                | Forever | Of_caller | Gone => true
                end)
             y (Of_region x) (cs_env cs) Hs_env Hl)
          as Hregion.
        cbv beta iota in Hregion. unfold not_a_region in Hx.
        rewrite Hregion in Hx. discriminate.
      * apply (grown_holds rs' _ y l v Hgrows Hv).
        exact (holds_next rs _ _ l v Hfresh (Hvalues y l v Hl Hv)).
    + intros y l v.
      change
        (lookup y ((x, Gone) :: cs_contents cs) = Some l ->
         lookup y ((x, Value_symbol m) :: vars) = Some v ->
         contents_hold rs' (x :: cs_open cs) (m :: ms) None l v).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x); [ intros [= <-] [= <-]; exact I | ].
      intros Hl Hv.
      apply (contents_hold_impl rs' rs' (cs_open cs) (x :: cs_open cs) ms
               (m :: ms) (Some m) None l l v eq_refl).
      * intros w. apply holds_push. intros ->.
        pose proof
          (lookup_forallb
             (fun l =>
                match l with
                | Of_region r => among r (cs_regions cs)
                | Forever | Of_caller | Gone => true
                end)
             y (Of_region x) (cs_contents cs) Hs_contents Hl)
          as Hregion.
        cbv beta iota in Hregion. unfold not_a_region in Hx.
        rewrite Hregion in Hx. discriminate.
      * apply (grown_contents rs' _ y l v Hgrows Hv).
        exact (contents_next rs _ _ l v Hfresh (Hcontents y l v Hl Hv)).
  - apply (grown_conts rs' _ Hgrows).
    apply bound_extends.
    + exact Hx.
    + intros r Hr. change (among r (x :: cs_regions cs) = true).
      rewrite among_cons, Hr. apply orb_true_r.
    + intros m0 [<- | Hin]; [ right; apply Pos.le_refl | left; exact Hin ].
    + change (m <= Pos.succ m)%positive. lia.
Qed.

(* ======================================================================== *)
(* The end of a region                                                      *)
(* ======================================================================== *)

Lemma end_checked (x : variable) (kind : flambda_kind) (ended_by : mark_kind)
    (r : variable) (d : named) (body : expr) (rs' : region_state) :
  state_expr (rs_state rs) = Let x kind d body ->
  not_a_region cs x = true ->
  among r (cs_open cs) = true ->
  check_expr cert codes
    (Check_state ((x, Forever) :: cs_env cs) (regions_below r (cs_open cs))
       (cs_regions cs) (cs_conts cs) ((x, Forever) :: cs_contents cs))
    body
  = true ->
  end_region rs x kind ended_by (Var r) body = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hx Hr Hbody Hstep Hinv' Hordered'.
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (proj1 (open_marks_marked _ _ _) Hopen) as Hmarked.
  pose proof (placed_below rs ms Hfresh Hplaced) as Hbelow_next.
  pose proof (placed_decrease _ _ Hplaced) as Hdecrease.
  pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
  pose proof Hscoped as Hscoped'.
  apply andb_true_iff in Hscoped' as [Hs1 Hs2].
  destruct (open_marks_in vars (cs_open cs) ms r Hopen Hr)
    as (m & Hrm & Hm_in).
  unfold end_region in Hstep. cbv zeta in Hstep.
  destruct (kind_eqb kind Value); [ | discriminate ].
  cbn [eval_simple] in Hstep. rewrite Hrm in Hstep.
  destruct (cut m (rs_frames rs)) as [[above below] | ] eqn:Hcut;
    [ | discriminate ].
  destruct (cut_mark _ _ _ _ Hcut) as (f & Hframes & Hmark).
  injection Hstep as Heq. subst rs'.
  match goal with
  | |- state_checked _ _ ?t => set (rs' := t) in *
  end.
  pose proof Hordered as Hord. unfold frames_ordered in Hord.
  rewrite Hframes in Hord.
  assert (Hsplit : above ++ f :: below = (above ++ [f]) ++ below)
    by (rewrite <- app_assoc; reflexivity).
  assert (Hcut_store : store_cut rs rs' m).
  { constructor.
    - exists above, f. split; [ exact Hframes | exact Hmark ].
    - intros s. reflexivity.
    - reflexivity.
    - reflexivity. }
  assert (Hdistinct :
    distinct (map frame_mark ((above ++ [f]) ++ below)) = true).
  { pose proof Hinv as H.
    apply region_invariant_parts in H as (_ & _ & H & _).
    pose proof (proj1 (proj1 (cells_live_parts rs) H)) as Hd.
    rewrite Hframes, Hsplit in Hd. exact Hd. }
  assert (Hlive :
    static_values (live_statics rs)
    = live_locals (above ++ [f]) ++ static_values (live_statics rs')).
  { change (static_values (live_statics rs'))
      with (live_locals below
            ++ static_values (state_statics (rs_state rs))).
    change (static_values (live_statics rs))
      with (live_locals (rs_frames rs)
            ++ static_values (state_statics (rs_state rs))).
    rewrite Hframes, Hsplit, live_locals_app, <- app_assoc. reflexivity. }
  assert (Hpopped :
    ∀ (s : symbol) (allocated : statically_allocated_value),
      In (s, allocated) (live_locals (above ++ [f])) ->
      lookup s (static_values (live_statics rs')) = None
      ∧ drawn rs s = true).
  { intros s allocated Hin.
    assert (Hin_all : In (s, allocated) (live_locals (rs_frames rs))).
    { rewrite Hframes, Hsplit, live_locals_app. apply in_or_app. left.
      exact Hin. }
    split; [ | exact (local_drawn rs s allocated Hfresh Hin_all) ].
    change
      (lookup s
         (live_locals below ++ static_values (state_statics (rs_state rs)))
       = None).
    rewrite lookup_app.
    pose proof
      (frames_ordered_cut above (rs_next rs) f below s allocated Hord Hin)
      as Hgt.
    rewrite (lookup_not_in s (live_locals below)).
    - destruct (proj1 (symbols_fresh_parts rs) Hfresh)
        as (_ & _ & _ & _ & _ & _ & Hapart).
      exact (Hapart s allocated Hin_all).
    - intros a Ha.
      pose proof
        (locals_below_top below (frame_mark f) s a
           (frames_ordered_tail above (rs_next rs) f below Hord) Ha).
      lia. }
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - intros s H. exact H.
    - intros s _. rewrite Hlive, lookup_app.
      destruct (lookup s (live_locals (above ++ [f]))) as [allocated | ]
        eqn:Hs.
      + right. exact (Hpopped s allocated (lookup_in _ _ _ Hs)).
      + destruct (lookup s (static_values (live_statics rs'))) eqn:Hs';
          [ left; reflexivity | intros _; reflexivity ].
    - intros l _ H. exact H. }
  pose
    (cs1 :=
       Check_state ((x, Forever) :: cs_env cs) (regions_below r (cs_open cs))
         (cs_regions cs) (cs_conts cs) ((x, Forever) :: cs_contents cs)).
  set (ms' := marks_below r (cs_open cs) ms).
  assert (Hms'_in : ∀ m' : mark, In m' ms' -> In m' ms)
    by exact (marks_below_in r (cs_open cs) ms).
  assert (Hms'_lt : ∀ m' : mark, In m' ms' -> (m' < m)%positive)
    by exact (marks_below_lt vars r (cs_open cs) ms m Hmarked Hdecrease Hrm).
  apply (moved_checked cert codes rs' cs1 ms' Hinv' Hordered').
  - exact Hcode.
  - pose proof Hlocal as Hl. unfold local_closures_fit in Hl |- *.
    apply forallb_forall. intros b Hin. apply (forallb_in _ _ b Hl).
    rewrite Hframes, live_locals_app. apply in_or_app. right.
    change (In b (frame_locals f ++ live_locals below)). apply in_or_app.
    right. exact Hin.
  - change (frames_deep rs' below = true).
    apply (frames_deep_mono rs rs' below (fun s H => H)).
    pose proof Hdeep as Hd. rewrite Hframes in Hd.
    apply (frames_deep_app rs (above ++ [f]) below). rewrite <- app_assoc.
    exact Hd.
  - exact Hstatics.
  - destruct (cells_deep_parts rs Hcells)
      as (Hc_frames & Hc_statics & Hc_heap).
    apply cells_deep_build.
    + change (frames_deep_cells rs' below = true).
      apply (frames_deep_cells_same rs rs' below
               (fun _ _ _ _ _ _ _ => eq_refl)).
      rewrite Hframes in Hc_frames.
      apply (frames_deep_cells_app rs (above ++ [f]) below).
      rewrite <- app_assoc. exact Hc_frames.
    + exact
        (statics_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
           Hc_statics).
    + change
        (forallb (fun '(l, c) => cell_deep rs' l c)
           (state_heap (rs_state rs))
         = true).
      apply forallb_forall. intros [l c] Hin.
      apply (cell_deep_same rs rs' l c eq_refl (fun _ _ _ _ => eq_refl)
               (fun s H => H));
        [ | exact (forallb_in _ _ (l, c) Hc_heap Hin) ].
      intros m1 fr' _ Hfrom. change (frames_from m1 below = Some fr') in Hfrom.
      exists fr'. split; [ | split; intros ? H; exact H ].
      rewrite Hframes, Hsplit.
      exact (frames_from_below m1 (above ++ [f]) below fr' Hdistinct Hfrom).
  - apply (moved_values cert codes rs rs' [(x, unit_value)] Hinv Hmoves
             Hgood).
    + reflexivity.
    + apply Forall_cons; [ split; reflexivity | apply Forall_nil ].
    + reflexivity.
    + intros s allocated Hin. left. rewrite Hlive. apply in_or_app. right.
      exact Hin.
    + intros l c Hin. left. exact Hin.
  - exact Hbody.
  - pose proof Hincl as Hi. rewrite He in Hi. exact Hi.
  - split; [ | split; [ | split; [ | split ] ] ].
    + change
        (open_marks ((x, unit_value) :: vars) (regions_below r (cs_open cs))
         = Some ms').
      rewrite open_marks_bind.
      * exact
          (proj2 (open_marks_marked _ _ _)
             (marks_below_marked vars r _ _ Hmarked)).
      * destruct (among x (regions_below r (cs_open cs))) eqn:H;
          [ | reflexivity ].
        rewrite (regions_below_among r _ x H) in Hx_open. discriminate.
    + change (marks_placed below ms' = true). unfold marks_placed.
      apply andb_true_iff. split.
      * apply forallb_forall. intros m' Hin.
        exact
          (cut_stack_keeps rs rs' m Hcut_store Hordered m'
             (marks_placed_in _ _ m' Hplaced (Hms'_in m' Hin))
             (Hms'_lt m' Hin)).
      * exact (marks_below_decrease r _ ms Hdecrease).
    + unfold check_state_scoped, cs1.
      cbn [cs_open cs_regions cs_env cs_contents].
      apply andb_true_iff. split.
      * apply all_among_iff. intros name Hname.
        exact
          (all_among_among _ _ name Hs1 (regions_below_among r _ name Hname)).
      * rewrite forallb_app in Hs2 |- *. cbn [forallb andb]. exact Hs2.
    + intros y l v.
      change
        (lookup y ((x, Forever) :: cs_env cs) = Some l ->
         lookup y ((x, unit_value) :: vars) = Some v ->
         lifetime_holds rs' (regions_below r (cs_open cs)) ms' None l v).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x); [ intros [= <-] [= <-]; reflexivity | ].
      intros Hl Hv.
      apply (holds_below vars r rs' l v (cs_open cs) ms Hmarked
               (marked_NoDup vars _ ms Hmarked Hdecrease)
               (Some (rs_next rs))).
      apply (cut_lifetime rs rs' m Hcut_store Hordered);
        [ intros m0 Hin; exact (forallb_ltb_in _ _ m0 Hbelow_next Hin)
        | exact Hdecrease
        | left; exact Hm_in
        | ].
      exact (holds_next rs _ _ l v Hfresh (Hvalues y l v Hl Hv)).
    + intros y l v.
      change
        (lookup y ((x, Forever) :: cs_contents cs) = Some l ->
         lookup y ((x, unit_value) :: vars) = Some v ->
         contents_hold rs' (regions_below r (cs_open cs)) ms' None l v).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x); [ intros [= <-] [= <-]; reflexivity | ].
      intros Hl Hv.
      apply (contents_hold_impl rs' rs' (cs_open cs)
               (regions_below r (cs_open cs)) ms ms' (Some (rs_next rs)) None
               l l v eq_refl).
      * intros w.
        exact
          (holds_below vars r rs' l w (cs_open cs) ms Hmarked
             (marked_NoDup vars _ ms Hmarked Hdecrease) (Some (rs_next rs))).
      * apply (cut_contents rs rs' m Hcut_store Hordered);
          [ intros m0 Hin; exact (forallb_ltb_in _ _ m0 Hbelow_next Hin)
          | exact Hdecrease
          | left; exact Hm_in
          | ].
        exact (contents_next rs _ _ l v Hfresh (Hcontents y l v Hl Hv)).
  - apply (conts_cut cert codes rs rs' m (running_user rs cs ms));
      [ exact Hcut_store
      | exact Hordered
      | left; exact Hm_in
      |
      | exact Hconts ].
    apply bound_extends.
    + exact Hx.
    + intros r0 Hr0. exact Hr0.
    + intros m0 Hin. left. exact (Hms'_in m0 Hin).
    + apply Pos.le_refl.
Qed.

(* ======================================================================== *)
(* A local allocation                                                       *)
(* ======================================================================== *)

Lemma local_checked (x y : variable) (kind : flambda_kind) (d : named)
    (region : option simple) (body body' : expr) (rs' : region_state) :
  state_expr (rs_state rs) = Let x kind d body ->
  incl (code_of_expr body') codes ->
  not_a_region cs x = true ->
  not_a_region cs y = true ->
  is_cell_allocation d && kind_eqb kind Value = false ->
  forallb (check_usable cs) (simples_of_named d) = true ->
  (∀ s : simple,
     In s (simples_of_named d) -> simple_below (rs_base rs) s = true) ->
  forallb (closure_may_be_local cert) (closure_code d) = true ->
  check_expr cert codes
    (bind_lifetime y (newest_lifetime cs)
       (bind_lifetime x (newest_lifetime cs) cs))
    body'
  = true ->
  local_allocation rs x y kind d region body' = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hincl' Hx Hy Hno_cell Husable Hb Hclosures Hbody Hstep Hinv'
    Hordered'.
  destruct (local_allocation_step rs rs' x y kind d region body' Hno_cell
              Hstep)
    as (v & sym & bindings & Halloc & Hv & Hlocalise & Hexpr' & Hvars'
        & Hconts' & Hbase' & Hnext' & Hcode' & Hstore').
  destruct (local_allocation_memory rs rs' x y kind d region body' Hno_cell
              Hstep)
    as (Hheap' & _ & Hcells').
  destruct (localise_makes _ _ _ _ Hlocalise)
    as (_ & (allocated_sym & Hsym) & Hbindings).
  destruct (eval_named_makes _ _ d v Halloc Hv) as [_ Hparts].
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (proj1 (open_marks_marked _ _ _) Hopen) as Hmarked.
  pose proof (placed_below rs ms Hfresh Hplaced) as Hbelow_next.
  pose proof Hgood
    as (Hgood_vars & Hgood_conts & Hgood_statics & Hgood_heap).
  destruct (state_below rs Hfresh) as (Hbelow_vars & Hbelow_conts & Hbound).
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as (_ & _ & Hbase_next & _).
  pose proof (advance_le (rs_next rs) (length bindings)) as Hadvance.
  pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
  pose proof (not_region_not_open cs y Hscoped Hy) as Hy_open.
  pose proof Hscoped as Hscoped'.
  apply andb_true_iff in Hscoped' as [Hs1 Hs2].
  destruct (cells_deep_parts rs Hcells)
    as (Hc_frames & Hc_statics & Hc_heap).
  assert (Hsame_cells :
    ∀ l : location, lookup l (rs_cells rs') = lookup l (rs_cells rs))
    by (intros l; rewrite Hcells'; reflexivity).
  (* The values that the allocation holds. *)
  assert (Hpart :
    ∀ w : value,
      In w (value_parts v) ->
      value_within rs None w = true ∧ good cert codes rs w ∧ old rs w).
  { intros w Hin. destruct (Hparts w Hin) as (s & Hs_in & Hs).
    split;
      [ exact
          (running_usable_within rs cs ms s w Hrun (Hb s Hs_in)
             (forallb_in _ _ s Husable Hs_in) Hs)
      | ].
    split; [ exact (simple_good cert codes rs vars Hgood_vars s w Hs) | ].
    split; [ | exact (simple_closed s w Hs) ].
    exact
      (eval_simple_below (rs_base rs) (rs_next rs) vars s w Hbase_next
         (Hb s Hs_in) Hbelow_vars Hs). }
  (* What is the same whether there is a frame or not. *)
  assert (Hcommon :
    static_values (live_statics rs')
    = bindings ++ static_values (live_statics rs) ->
    store_grows rs rs' ->
    symbol_within rs' None sym = true ->
    local_closures_fit cert rs' = true ->
    frames_deep rs' (rs_frames rs') = true ->
    statics_deep rs' = true ->
    cells_deep rs' = true ->
    state_checked cert codes rs').
  { intros Hlive' Hgrows Hsym_within Hlocal' Hdeep' Hstatics' Hcells_deep'.
    assert (Hmoves : store_moves rs rs').
    { constructor.
      - exact Hbase'.
      - exact (grows_heap rs rs' Hgrows).
      - intros s Hlt. rewrite Hlive', lookup_app.
        rewrite (lookup_not_in s bindings).
        + destruct (lookup s (static_values (live_statics rs))) eqn:Hs;
            [ left; reflexivity | intros _; reflexivity ].
        + intros a Ha. destruct (Hbindings s a Ha) as (Hfrom & _). lia.
      - intros l _ H. unfold heap_location in H |- *. rewrite Hcells'.
        exact H. }
    assert (Hsym_value : value_within rs' None (Value_symbol sym) = true).
    { unfold value_within. cbn [value_all value_all_cells].
      rewrite Hsym_within. reflexivity. }
    assert (Hnewest :
      ∀ r0 : variable,
        newest_lifetime cs = Of_region r0 ->
        among r0 (cs_regions cs) = true).
    { intros r0. unfold newest_lifetime.
      destruct (cs_open cs) as [ | r1 opens] eqn:Hopens; [ discriminate | ].
      intros [= <-]. apply (all_among_among _ _ r1 Hs1).
      try rewrite Hopens. rewrite among_cons, Pos.eqb_refl. reflexivity. }
    apply (moved_checked cert codes rs'
             (bind_lifetime y (newest_lifetime cs)
                (bind_lifetime x (newest_lifetime cs) cs))
             ms Hinv' Hordered').
    - unfold code_checked. rewrite Hcode'. exact Hcode.
    - exact Hlocal'.
    - exact Hdeep'.
    - exact Hstatics'.
    - exact Hcells_deep'.
    - apply (moved_values cert codes rs rs'
               [(y, Value_symbol sym); (x, Value_symbol sym)] Hinv Hmoves
               Hgood).
      + exact Hvars'.
      + repeat (apply Forall_cons; [ split; reflexivity | ]).
        apply Forall_nil.
      + exact Hconts'.
      + intros s allocated Hin. rewrite Hlive' in Hin.
        apply in_app_or in Hin as [Hnew | Hold]; [ right | left; exact Hold ].
        apply allocated_values_in. intros w Hw.
        destruct (Hbindings s allocated Hnew) as (_ & _ & Hvalues_eq & _).
        rewrite Hvalues_eq in Hw.
        destruct (Hpart w Hw) as (_ & Hgood_w & Hold_w).
        exact (good_moves cert codes rs rs' Hmoves w (conj Hgood_w Hold_w)).
      + intros l c Hin. left. rewrite Hheap' in Hin. exact Hin.
    - rewrite Hexpr'. exact Hbody.
    - rewrite Hexpr'. exact Hincl'.
    - split; [ | split; [ | split; [ | split ] ] ].
      + rewrite Hvars'. cbn [bind_lifetime bind_lifetimes cs_open].
        rewrite (open_marks_bind _ y _ _ Hy_open),
          (open_marks_bind _ x _ _ Hx_open).
        exact Hopen.
      + apply (grows_placed rs rs' Hgrows ms);
          [ intros m0 Hin; exact (forallb_ltb_in _ _ m0 Hbelow_next Hin)
          | exact Hplaced ].
      + apply bind_lifetimes_scoped;
          [ apply bind_lifetimes_scoped; [ exact Hscoped | | ] | | ];
          exact Hnewest.
      + intros y0 l w. rewrite Hvars'.
        cbn [bind_lifetime bind_lifetimes cs_env cs_open].
        rewrite !lookup_cons.
        assert (Hsym_holds :
          lifetime_holds rs' (cs_open cs) ms None (newest_lifetime cs)
            (Value_symbol sym))
          by exact
               (newest_live rs' vars cs ms (Value_symbol sym) Hmarked
                  Hsym_value).
        destruct (Pos.eqb y0 y);
          [ intros [= <-] [= <-]; exact Hsym_holds | ].
        destruct (Pos.eqb y0 x);
          [ intros [= <-] [= <-]; exact Hsym_holds | ].
        intros Hl Hw.
        exact (grown_holds rs' None y0 l w Hgrows Hw (Hvalues y0 l w Hl Hw)).
      + intros y0 l w. rewrite Hvars'.
        cbn [bind_lifetime bind_lifetimes cs_contents cs_open].
        rewrite !lookup_cons.
        assert (Hsym_holds :
          lifetime_holds rs' (cs_open cs) ms None (newest_lifetime cs)
            (Value_symbol sym))
          by exact
               (newest_live rs' vars cs ms (Value_symbol sym) Hmarked
                  Hsym_value).
        destruct (Pos.eqb y0 y);
          [ intros [= <-] [= <-]; exact Hsym_holds | ].
        destruct (Pos.eqb y0 x);
          [ intros [= <-] [= <-]; exact Hsym_holds | ].
        intros Hl Hw.
        exact
          (grown_contents rs' None y0 l w Hgrows Hw (Hcontents y0 l w Hl Hw)).
    - rewrite Hconts'. apply (grown_conts rs' _ Hgrows).
      split; [ | split ].
      + intros r0 Hr0. split; [ | exact Hr0 ].
        change (lookup r0 (state_vars (rs_state rs')) = lookup r0 vars).
        rewrite Hvars', !lookup_cons, (region_not_bound y r0 Hy Hr0),
          (region_not_bound x r0 Hx Hr0).
        reflexivity.
      + intros m0 Hin. left. exact Hin.
      + change (rs_next rs <= rs_next rs')%positive. rewrite Hnext'.
        exact Hadvance. }
  revert Hstore'.
  destruct (rs_frames rs) as [ | [m0 mk locals] frames0] eqn:Hframes;
    intros [Hf' Hs'].
  - (* No frame: the allocation goes to the static environment. *)
    assert (Hheap :
      ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true)
      by exact (heap_symbol_extends rs rs' bindings Hbase' Hs').
    assert (Hgrows : store_grows rs rs').
    { constructor.
      - rewrite Hnext'. exact Hadvance.
      - exact Hheap.
      - intros s. rewrite Hframes. discriminate.
      - intros m0 _. rewrite Hf', Hframes. reflexivity.
      - intros m0. rewrite Hframes. discriminate.
      - exact (cells_same_on_same _ rs rs' Hcells').
      - intros l c' w _ Hc' Hw. left. exists c'. rewrite Hheap' in Hc'.
        split; [ exact Hc' | exact Hw ]. }
    assert (Hheap_cells :
      ∀ w : value,
        value_all_cells (cell_within rs None) w = true ->
        value_all_cells (heap_location rs') w = true).
    { intros w. apply value_all_cells_mono. intros l.
      unfold cell_within, heap_location. rewrite Hcells', Hframes.
      destruct (lookup l (rs_cells rs)); [ | intros _; reflexivity ].
      simpl. discriminate. }
    assert (Hnew_heap :
      ∀ (s : symbol) (a : statically_allocated_value),
        In (s, a) bindings -> heap_symbol rs' s = true).
    { intros s a Hin. unfold heap_symbol. rewrite Hs', lookup_app.
      pose proof (bound_is_some s a bindings Hin) as Hsome.
      destruct (lookup s bindings); [ apply orb_true_r | discriminate Hsome ]. }
    apply Hcommon.
    + unfold live_statics. cbn [static_values]. rewrite Hf', Hs', Hframes.
      reflexivity.
    + exact Hgrows.
    + unfold symbol_within. rewrite (Hnew_heap sym allocated_sym Hsym).
      reflexivity.
    + unfold local_closures_fit. rewrite Hf'. reflexivity.
    + rewrite Hf'. reflexivity.
    + unfold statics_deep. rewrite Hs', forallb_app. apply andb_true_iff. split.
      * apply forallb_forall. intros [s a] Hin.
        destruct (Hbindings s a Hin) as (_ & _ & Hvalues_eq & Hsymbols).
        apply allocated_all_of_parts.
        -- intros w Hw. rewrite Hvalues_eq in Hw.
           destruct (Hpart w Hw) as (Hwithin & _).
           unfold value_within in Hwithin.
           apply andb_true_iff in Hwithin as [Hwithin _].
           revert Hwithin. apply value_all_mono.
           intros s0 H0. unfold symbol_within in H0. rewrite Hframes in H0.
           change (heap_symbol rs s0 || false = true) in H0.
           rewrite orb_false_r in H0. exact (Hheap s0 H0).
        -- intros s0 Hs0. destruct (Hsymbols s0 Hs0) as (a0 & Ha0).
           exact (Hnew_heap s0 a0 Ha0).
      * pose proof Hstatics as Hst. unfold statics_deep in Hst. revert Hst.
        apply forallb_impl. intros [s a] _. apply allocated_all_mono.
        exact Hheap.
    + apply cells_deep_build.
      * rewrite Hf'. reflexivity.
      * rewrite Hs', forallb_app. apply andb_true_iff. split.
        -- apply forallb_forall. intros [s a] Hin.
           destruct (Hbindings s a Hin) as (_ & _ & Hvalues_eq & _).
           rewrite values_of_static_values, Hvalues_eq.
           apply forallb_forall. intros w Hw.
           destruct (Hpart w Hw) as (Hwithin & _).
           unfold value_within in Hwithin.
           apply andb_true_iff in Hwithin as [_ Hwithin].
           exact (Hheap_cells w Hwithin).
        -- exact
             (statics_cells_same rs rs' _
                (fun _ _ _ l _ _ _ => Hsame_cells l) Hc_statics).
      * rewrite Hheap'. apply forallb_forall. intros [l c] Hin.
        apply (cell_deep_same rs rs' l c (Hsame_cells l)
                 (fun _ l' _ _ => Hsame_cells l') Hheap);
          [ | exact (forallb_in _ _ (l, c) Hc_heap Hin) ].
        intros m1 fr' _ Hfrom. rewrite Hf' in Hfrom. discriminate Hfrom.
  - (* The allocation goes to the newest frame. *)
    assert (Hheap :
      ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true).
    { intros s. unfold heap_symbol, drawn. rewrite Hbase', Hs'.
      exact (fun H => H). }
    assert (Hlocal_grows :
      ∀ s : symbol,
        local_in (Frame m0 mk locals :: frames0) s = true ->
        local_in (Frame m0 mk (bindings ++ locals) :: frames0) s = true).
    { intros s. unfold local_in. cbn [live_locals flat_map frame_locals].
      rewrite <- app_assoc, (lookup_app s bindings).
      destruct (lookup s bindings); [ reflexivity | exact (fun H => H) ]. }
    assert (Hnew_local :
      ∀ (s : symbol) (a : statically_allocated_value),
        In (s, a) bindings ->
        local_in (Frame m0 mk (bindings ++ locals) :: frames0) s = true).
    { intros s a Hin. unfold local_in. cbn [live_locals flat_map frame_locals].
      rewrite <- app_assoc, (lookup_app s bindings).
      pose proof (bound_is_some s a bindings Hin) as Hsome.
      destruct (lookup s bindings); [ reflexivity | discriminate Hsome ]. }
    assert (Hgrows : store_grows rs rs').
    { constructor.
      - rewrite Hnext'. exact Hadvance.
      - exact Hheap.
      - intros s. rewrite Hframes, Hf'. exact (Hlocal_grows s).
      - intros m1 _. rewrite Hf', Hframes. reflexivity.
      - intros m1. rewrite Hf', Hframes. exact (fun H => H).
      - exact (cells_same_on_same _ rs rs' Hcells').
      - intros l c' w _ Hc' Hw. left. exists c'. rewrite Hheap' in Hc'.
        split; [ exact Hc' | exact Hw ]. }
    assert (Hframe_cells :
      ∀ w : value,
        value_all_cells (cell_within rs None) w = true ->
        value_all_cells
          (fun l =>
             heap_location rs' l
             || cell_in rs' (Frame m0 mk (bindings ++ locals) :: frames0) l)
          w
        = true).
    { intros w. apply value_all_cells_mono. intros l.
      unfold cell_within, heap_location, cell_in. rewrite Hcells', Hframes.
      destruct (lookup l (rs_cells rs)); [ | exact (fun H => H) ].
      cbn [before]. rewrite andb_true_r. exact (fun H => H). }
    apply Hcommon.
    + unfold live_statics. cbn [static_values]. rewrite Hf', Hs', Hframes.
      cbn [live_locals flat_map frame_locals]. rewrite <- !app_assoc.
      reflexivity.
    + exact Hgrows.
    + unfold symbol_within. rewrite Hf', (Hnew_local sym allocated_sym Hsym).
      cbn [before andb]. apply orb_true_r.
    + pose proof Hlocal as Hl. unfold local_closures_fit in Hl |- *.
      rewrite Hframes in Hl. rewrite Hf'.
      cbn [live_locals flat_map frame_locals] in Hl |- *.
      rewrite <- app_assoc, forallb_app. apply andb_true_iff.
      split; [ | exact Hl ].
      exact
        (new_closures_fit cert _ _ d v _ sym bindings Halloc Hv Hlocalise
           Hclosures).
    + pose proof Hdeep as Hd. try rewrite Hframes in Hd. rewrite Hf'.
      cbn [frames_deep frame_locals] in Hd |- *.
      apply andb_true_iff in Hd as [Hd_locals Hd_below]. apply andb_true_iff.
      split; [ | exact (frames_deep_mono rs rs' frames0 Hheap Hd_below) ].
      rewrite forallb_app. apply andb_true_iff. split.
      * apply forallb_forall. intros [s a] Hin.
        destruct (Hbindings s a Hin) as (_ & _ & Hvalues_eq & Hsymbols).
        apply allocated_all_of_parts.
        -- intros w Hw. rewrite Hvalues_eq in Hw.
           destruct (Hpart w Hw) as (Hwithin & _).
           unfold value_within in Hwithin.
           apply andb_true_iff in Hwithin as [Hwithin _].
           revert Hwithin. apply value_all_mono.
           intros s0 H0. unfold symbol_within in H0.
           apply orb_true_iff in H0 as [H0 | H0];
             [ rewrite (Hheap s0 H0); reflexivity | ].
           apply andb_true_iff in H0 as [H0 _]. rewrite Hframes in H0.
           rewrite (Hlocal_grows s0 H0). apply orb_true_r.
        -- intros s0 Hs0. destruct (Hsymbols s0 Hs0) as (a0 & Ha0).
           rewrite (Hnew_local s0 a0 Ha0). apply orb_true_r.
      * revert Hd_locals. apply forallb_impl. intros [s a] _.
        apply allocated_all_mono. intros s0 H0.
        apply orb_true_iff in H0 as [H0 | H0];
          [ rewrite (Hheap s0 H0); reflexivity | ].
        rewrite (Hlocal_grows s0 H0). apply orb_true_r.
    + pose proof Hstatics as Hst. unfold statics_deep in Hst |- *.
      rewrite Hs'. revert Hst. apply forallb_impl. intros [s a] _.
      apply allocated_all_mono. exact Hheap.
    + apply cells_deep_build.
      * rewrite Hf'.
        cbn [frames_deep_cells frame_locals] in Hc_frames |- *.
        apply andb_true_iff in Hc_frames as [Hc_locals Hc_below].
        apply andb_true_iff.
        split;
          [
          | exact
              (frames_deep_cells_same rs rs' frames0
                 (fun _ _ _ l _ _ _ => Hsame_cells l) Hc_below) ].
        rewrite forallb_app. apply andb_true_iff. split.
        -- apply forallb_forall. intros [s a] Hin.
           destruct (Hbindings s a Hin) as (_ & _ & Hvalues_eq & _).
           rewrite values_of_static_values, Hvalues_eq.
           apply forallb_forall. intros w Hw.
           destruct (Hpart w Hw) as (Hwithin & _).
           unfold value_within in Hwithin.
           apply andb_true_iff in Hwithin as [_ Hwithin].
           exact (Hframe_cells w Hwithin).
        -- revert Hc_locals. apply forallb_impl. intros [s a] _.
           apply forallb_impl. intros w _.
           apply (cells_all_same rs rs' _ _ w (fun l _ => Hsame_cells l)).
           intros m1 H. exact H.
      * rewrite Hs'.
        exact
          (statics_cells_same rs rs' _ (fun _ _ _ l _ _ _ => Hsame_cells l)
             Hc_statics).
      * rewrite Hheap'. apply forallb_forall. intros [l c] Hin.
        apply (cell_deep_same rs rs' l c (Hsame_cells l)
                 (fun _ l' _ _ => Hsame_cells l') Hheap);
          [ | exact (forallb_in _ _ (l, c) Hc_heap Hin) ].
        intros m1 fr' _ Hfrom. rewrite Hf' in Hfrom. rewrite Hframes.
        cbn [frames_from frame_mark] in Hfrom |- *.
        destruct (Pos.eqb m0 m1).
        -- injection Hfrom as <-. exists (Frame m0 mk locals :: frames0).
           split; [ reflexivity | ].
           split; [ exact Hlocal_grows | intros m' H; exact H ].
        -- exists fr'. split; [ exact Hfrom | ].
           split; intros ? H; exact H.
Qed.

(* ======================================================================== *)
(* A local allocation of a cell                                             *)
(* ======================================================================== *)

Lemma local_cell_checked (x y : variable) (kind : flambda_kind) (d : named)
    (region : option simple) (body body' : expr) (rs' : region_state) :
  state_expr (rs_state rs) = Let x kind d body ->
  incl (code_of_expr body') codes ->
  not_a_region cs x = true ->
  not_a_region cs y = true ->
  is_cell_allocation d = true ->
  forallb (check_usable cs) (simples_of_named d) = true ->
  (∀ s : simple,
     In s (simples_of_named d) -> simple_below (rs_base rs) s = true) ->
  check_expr cert codes
    (bind_lifetimes y (newest_lifetime cs)
       (shortest cs (map (lifetime_of cs) (simples_of_named d)))
       (bind_lifetimes x (newest_lifetime cs)
          (shortest cs (map (lifetime_of cs) (simples_of_named d))) cs))
    body'
  = true ->
  local_cell_allocation rs x y d region body' = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hincl' Hx Hy His Husable Hb Hbody Hstep Hinv' Hordered'.
  destruct (local_cell_allocation_step rs rs' x y d region body' Hstep)
    as (l & hp & rd & _ & Hexec & Hstate' & Hframes' & Hbase' & Hnext'
        & Hcells').
  destruct (exec_named_heap_cases _ _ _ _ _ _ _ _ Hexec)
    as [[Hsame _]
       | [(_ & vs & c & Hvs & Hc & Hl & ->)
         | (target & new_simple & l0 & new & c & c' & i & _ & _ & _ & _ & _
            & Hunit & _)]].
  - rewrite (cell_allocation_changes_heap d His) in Hsame. discriminate.
  - injection Hl as ->.
    set (lf := FlambdaHeapSpec.fresh_location (state_heap (rs_state rs)))
      in *.
    pose proof (FlambdaHeapProofs.fresh_location (state_heap (rs_state rs)))
      as Hfresh_l.
    fold lf in Hfresh_l.
    assert (Hunlisted : lookup lf (rs_cells rs) = None).
    { destruct (lookup lf (rs_cells rs)) as [m | ] eqn:Hl; [ | reflexivity ].
      destruct (listed_mark lf m Hl) as (Hbound & _).
      apply is_bound_lookup in Hbound as (c0 & Hc0).
      rewrite Hfresh_l in Hc0. discriminate. }
    assert (Hnot_bound :
      ∀ l0 : location,
        is_bound l0 (state_heap (rs_state rs)) = true ->
        Pos.eqb l0 lf = false).
    { intros l0 Hbound. destruct (Pos.eqb l0 lf) eqn:Heq; [ | reflexivity ].
      apply Pos.eqb_eq in Heq. subst l0.
      apply is_bound_lookup in Hbound as (c0 & Hc0).
      rewrite Hfresh_l in Hc0. discriminate. }
    pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
    pose proof (proj1 (open_marks_marked _ _ _) Hopen) as Hmarked.
    pose proof Hgood
      as (Hgood_vars & Hgood_conts & Hgood_statics & Hgood_heap).
    pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
    pose proof (not_region_not_open cs y Hscoped Hy) as Hy_open.
    destruct (scoped_parts cs Hscoped) as [Hs_env Hs_contents].
    pose proof Hscoped as Hscoped'.
    apply andb_true_iff in Hscoped' as [Hs1 Hs2].
    destruct (cells_deep_parts rs Hcells)
      as (Hc_frames & Hc_statics & Hc_heap).
    destruct (state_closed rs Hinv)
      as (_ & _ & Hclosed_heap & Hclosed_statics).
    (* What the cell holds. *)
    assert (Hpart :
      ∀ w : value,
        In w (values_of_cell c) ->
        ∃ sim : simple,
          In sim (simples_of_named d)
          ∧ eval_simple vars sim = Some w
          ∧ value_within rs None w = true
          ∧ good cert codes rs w
          ∧ old rs w).
    { intros w Hw. rewrite Hc in Hw.
      destruct (Hvs w Hw) as (sim & Hsim_in & Hsim).
      exists sim. split; [ exact Hsim_in | ]. split; [ exact Hsim | ].
      split;
        [ exact
            (running_usable_within rs cs ms sim w Hrun (Hb sim Hsim_in)
               (forallb_in _ _ sim Husable Hsim_in) Hsim)
        | ].
      split; [ exact (simple_good cert codes rs vars Hgood_vars sim w Hsim) | ].
      exact (simple_old sim w (Hb sim Hsim_in) Hsim). }
    (* The cells that are listed. *)
    assert (Hsame_cells :
      ∀ l0 : location,
        is_bound l0 (state_heap (rs_state rs)) = true ->
        lookup l0 (rs_cells rs') = lookup l0 (rs_cells rs)).
    { intros l0 Hbound. rewrite Hcells'.
      destruct (rs_frames rs) as [ | f frames0]; [ reflexivity | ].
      rewrite lookup_cons, (Hnot_bound l0 Hbound). reflexivity. }
    assert (Hlisted' :
      lookup lf (rs_cells rs')
      = match rs_frames rs with
        | f :: _ => Some (frame_mark f)
        | [] => None
        end).
    { rewrite Hcells'.
      destruct (rs_frames rs); [ exact Hunlisted | apply lookup_head ]. }
    assert (Hwithin_l : cell_within rs' None lf = true).
    { unfold cell_within, heap_location. rewrite Hlisted', Hframes'.
      destruct (rs_frames rs) as [ | f frames0]; [ reflexivity | ].
      cbn [is_some negb orb before].
      rewrite on_stack_cons, Pos.eqb_refl. reflexivity. }
    assert (Hsym : ∀ s : symbol, heap_symbol rs' s = heap_symbol rs s).
    { intros s. unfold heap_symbol, drawn. rewrite Hbase', Hstate'.
      reflexivity. }
    assert (Hlive : live_statics rs' = live_statics rs).
    { unfold live_statics. rewrite Hframes', Hstate'. reflexivity. }
    assert (Hgrows : store_grows rs rs').
    { constructor.
      - rewrite Hnext'. apply Pos.le_refl.
      - intros s H. rewrite Hsym. exact H.
      - intros s H. rewrite Hframes'. exact H.
      - intros m0 _. rewrite Hframes'. reflexivity.
      - intros m0 H. rewrite Hframes'. exact H.
      - exact Hsame_cells.
      - intros l0 c0 w Hbound Hc0 Hw. left. exists c0.
        split; [ | exact Hw ].
        rewrite Hstate' in Hc0. cbn [state_heap] in Hc0.
        rewrite lookup_cons, (Hnot_bound l0 Hbound) in Hc0. exact Hc0. }
    assert (Hmoves : store_moves rs rs').
    { constructor.
      - exact Hbase'.
      - intros s H. rewrite Hsym. exact H.
      - intros s _. rewrite Hlive.
        destruct (lookup s (static_values (live_statics rs)));
          [ left; reflexivity | intros _; reflexivity ].
      - intros l0 Hbound H. unfold heap_location in H |- *.
        rewrite (Hsame_cells l0 Hbound). exact H. }
    assert (Hnewest :
      ∀ r0 : variable,
        newest_lifetime cs = Of_region r0 ->
        among r0 (cs_regions cs) = true).
    { intros r0. unfold newest_lifetime.
      destruct (cs_open cs) as [ | r1 opens] eqn:Hopens; [ discriminate | ].
      intros [= <-]. apply (all_among_among _ _ r1 Hs1).
      try rewrite Hopens. rewrite among_cons, Pos.eqb_refl. reflexivity. }
    assert (Hloc_holds :
      lifetime_holds rs' (cs_open cs) ms None (newest_lifetime cs)
        (Value_location lf)).
    { apply (newest_live rs' vars cs ms (Value_location lf) Hmarked).
      unfold value_within. cbn [value_all value_all_cells andb].
      exact Hwithin_l. }
    assert (Hloc_contents :
      contents_hold rs' (cs_open cs) ms None
        (shortest cs (map (lifetime_of cs) (simples_of_named d)))
        (Value_location lf)).
    { intros c0 w Hc0 Hw. rewrite Hstate' in Hc0. cbn [state_heap] in Hc0.
      rewrite lookup_head in Hc0. injection Hc0 as <-.
      destruct (Hpart w Hw) as (sim & Hsim_in & Hsim & _ & _ & [_ Hclosed_w]).
      apply (grows_lifetime rs rs' Hgrows _ _ _ _ _ Hclosed_w
               running_below_next).
      apply (holds_rank rs vars _ _ (lifetime_of cs sim) _ w Hmarked Hplaced).
      - apply rank_shortest_le. apply in_map. exact Hsim_in.
      - exact (simple_holds rs vars cs ms Hrun sim w (Hb sim Hsim_in) Hsim). }
    apply (moved_checked cert codes rs'
             (bind_lifetimes y (newest_lifetime cs)
                (shortest cs (map (lifetime_of cs) (simples_of_named d)))
                (bind_lifetimes x (newest_lifetime cs)
                   (shortest cs (map (lifetime_of cs) (simples_of_named d)))
                   cs))
             ms Hinv' Hordered').
    + unfold code_checked. rewrite Hstate'. exact Hcode.
    + unfold local_closures_fit. rewrite Hframes'. exact Hlocal.
    + rewrite Hframes'.
      apply (frames_deep_mono rs rs' _); [ | exact Hdeep ].
      intros s H. rewrite Hsym. exact H.
    + unfold statics_deep. rewrite Hstate'. cbn [state_statics].
      pose proof Hstatics as Hst. unfold statics_deep in Hst. revert Hst.
      apply forallb_impl. intros [s a] _. apply allocated_all_mono.
      intros s0 H. rewrite Hsym. exact H.
    + apply cells_deep_build.
      * rewrite Hframes'.
        apply (frames_deep_cells_same rs rs' (rs_frames rs));
          [ | exact Hc_frames ].
        intros s allocated w l0 Hin Hw Hl0. apply Hsame_cells.
        apply (Hclosed_statics s allocated w);
          [ | rewrite <- values_of_static_values; exact Hw | exact Hl0 ].
        change
          (In (s, allocated)
             (live_locals (rs_frames rs)
              ++ static_values (state_statics (rs_state rs)))).
        apply in_or_app. left. exact Hin.
      * rewrite Hstate'. cbn [state_statics].
        apply (statics_cells_same rs rs' _); [ | exact Hc_statics ].
        intros s allocated w l0 Hin Hw Hl0. apply Hsame_cells.
        apply (Hclosed_statics s allocated w);
          [ | rewrite <- values_of_static_values; exact Hw | exact Hl0 ].
        change
          (In (s, allocated)
             (live_locals (rs_frames rs)
              ++ static_values (state_statics (rs_state rs)))).
        apply in_or_app. right. exact Hin.
      * rewrite Hstate'. cbn [state_heap forallb]. apply andb_true_iff. split.
        -- (* The cell that is made. *)
           unfold cell_deep. rewrite Hlisted', Hframes'.
           destruct (rs_frames rs) as [ | f frames0] eqn:Hframes.
           ++ apply forallb_forall. intros w Hw.
              destruct (Hpart w Hw)
                as (_ & _ & _ & Hwithin & _ & [_ Hclosed_w]).
              unfold value_within in Hwithin.
              apply andb_true_iff in Hwithin as [Hw1 Hw2].
              unfold value_for_ever. apply andb_true_iff. split.
              ** revert Hw1. apply value_all_mono. intros s H0.
                 unfold symbol_within in H0. rewrite Hframes in H0.
                 change (heap_symbol rs s || false = true) in H0.
                 rewrite orb_false_r in H0. rewrite Hsym. exact H0.
              ** apply value_all_cells_iff. intros l0 Hl0.
                 pose proof (proj1 (value_all_cells_iff _ w) Hw2 l0 Hl0) as H0.
                 unfold cell_within in H0. rewrite Hframes in H0.
                 unfold heap_location in H0 |- *.
                 rewrite (Hsame_cells l0 (Hclosed_w l0 Hl0)).
                 destruct (lookup l0 (rs_cells rs));
                   [ simpl in H0; discriminate H0 | reflexivity ].
           ++ cbn [frames_from]. rewrite Pos.eqb_refl.
              apply forallb_forall. intros w Hw.
              destruct (Hpart w Hw)
                as (_ & _ & _ & Hwithin & _ & [_ Hclosed_w]).
              unfold value_within in Hwithin.
              apply andb_true_iff in Hwithin as [Hw1 Hw2].
              apply andb_true_iff. split.
              ** revert Hw1. apply value_all_mono. intros s H0.
                 unfold symbol_within in H0. rewrite Hframes in H0.
                 cbn [before] in H0. rewrite andb_true_r in H0.
                 rewrite Hsym. exact H0.
              ** apply value_all_cells_iff. intros l0 Hl0.
                 pose proof (proj1 (value_all_cells_iff _ w) Hw2 l0 Hl0) as H0.
                 unfold cell_within in H0. rewrite Hframes in H0.
                 unfold heap_location, cell_in in H0 |- *.
                 rewrite (Hsame_cells l0 (Hclosed_w l0 Hl0)).
                 destruct (lookup l0 (rs_cells rs)); [ | exact H0 ].
                 cbn [before] in H0. rewrite andb_true_r in H0. exact H0.
        -- (* The cells that were. *)
           apply forallb_forall. intros [l0 c0] Hin.
           assert (Hbound0 : is_bound l0 (state_heap (rs_state rs)) = true).
           { apply is_bound_lookup.
             pose proof (bound_is_some l0 c0 _ Hin) as Hsome.
             destruct (lookup l0 (state_heap (rs_state rs))) as [c1 | ];
               [ exists c1; reflexivity | discriminate Hsome ]. }
           apply (cell_deep_same rs rs' l0 c0 (Hsame_cells l0 Hbound0)).
           ++ intros w l1 Hw Hl1. apply Hsame_cells.
              exact (Hclosed_heap l0 c0 w Hin Hw l1 Hl1).
           ++ intros s H. rewrite Hsym. exact H.
           ++ intros m1 fr' _ Hfrom. rewrite Hframes' in Hfrom. exists fr'.
              split; [ exact Hfrom | ]. split; intros ? H; exact H.
           ++ exact (forallb_in _ _ (l0, c0) Hc_heap Hin).
    + apply (moved_values cert codes rs rs'
               [(y, Value_location lf); (x, Value_location lf)] Hinv Hmoves
               Hgood).
      * rewrite Hstate'. reflexivity.
      * repeat (apply Forall_cons; [ split; reflexivity | ]).
        apply Forall_nil.
      * rewrite Hstate'. reflexivity.
      * intros s allocated Hin. left. rewrite Hlive in Hin. exact Hin.
      * intros l0 c0 Hin. rewrite Hstate' in Hin. cbn [state_heap] in Hin.
        destruct Hin as [[= <- <-] | Hin]; [ right | left; exact Hin ].
        apply Forall_forall. intros w Hw.
        destruct (Hpart w Hw) as (_ & _ & _ & _ & Hgood_w & Hold_w).
        exact (good_moves cert codes rs rs' Hmoves w (conj Hgood_w Hold_w)).
    + rewrite Hstate'. exact Hbody.
    + rewrite Hstate'. exact Hincl'.
    + split; [ | split; [ | split; [ | split ] ] ].
      * rewrite Hstate'. cbn [state_vars bind_lifetimes cs_open].
        rewrite (open_marks_bind _ y _ _ Hy_open),
          (open_marks_bind _ x _ _ Hx_open).
        exact Hopen.
      * rewrite Hframes'. exact Hplaced.
      * apply bind_lifetimes_scoped;
          [ apply bind_lifetimes_scoped; [ exact Hscoped | | ] | | ];
          first
            [ exact Hnewest
            | exact (fun r => shortest_scoped cs _ r Hs_env) ].
      * intros y0 l0 w. rewrite Hstate'.
        cbn [state_vars bind_lifetimes cs_env cs_open]. rewrite !lookup_cons.
        destruct (Pos.eqb y0 y);
          [ intros [= <-] [= <-]; exact Hloc_holds | ].
        destruct (Pos.eqb y0 x);
          [ intros [= <-] [= <-]; exact Hloc_holds | ].
        intros Hl Hw.
        exact (grown_holds rs' None y0 l0 w Hgrows Hw (Hvalues y0 l0 w Hl Hw)).
      * intros y0 l0 w. rewrite Hstate'.
        cbn [state_vars bind_lifetimes cs_contents cs_open].
        rewrite !lookup_cons.
        destruct (Pos.eqb y0 y);
          [ intros [= <-] [= <-]; exact Hloc_contents | ].
        destruct (Pos.eqb y0 x);
          [ intros [= <-] [= <-]; exact Hloc_contents | ].
        intros Hl Hw.
        exact
          (grown_contents rs' None y0 l0 w Hgrows Hw
             (Hcontents y0 l0 w Hl Hw)).
    + rewrite Hstate'. cbn [state_conts]. apply (grown_conts rs' _ Hgrows).
      split; [ | split ].
      * intros r0 Hr0. split; [ | exact Hr0 ].
        cbn [u_vars running_user]. rewrite Hstate'. cbn [state_vars].
        rewrite !lookup_cons, (region_not_bound y r0 Hy Hr0),
          (region_not_bound x r0 Hx Hr0).
        reflexivity.
      * intros m0 Hin. left. exact Hin.
      * change (rs_next rs <= rs_next rs')%positive. rewrite Hnext'.
        apply Pos.le_refl.
  - unfold unit_value in Hunit. discriminate Hunit.
Qed.

(* ======================================================================== *)
(* A [Let_static]                                                           *)
(* ======================================================================== *)

Lemma static_checked (bindings : list static_binding) (body : expr)
    (rs' : region_state) :
  state_expr (rs_state rs) = Let_static bindings body ->
  check_expr cert codes cs (Let_static bindings body) = true ->
  static_step rs = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hcheck Hstep Hinv' Hordered'.
  cbn [check_expr] in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck Hbody].
  apply andb_true_iff in Hcheck as [Hforever Hbindings_check].
  destruct (static_step_step rs rs' bindings body He Hstep)
    as (statics' & Heval & ->).
  match goal with
  | |- state_checked _ _ ?t => set (rs' := t) in *
  end.
  destruct (eval_static_bindings_parts _ _ _ _ Heval)
    as (new_code & new_values & Hcode_eq & Hvalues_eq & Hnew_code
        & Hnew_values).
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (placed_below rs ms Hfresh Hplaced) as Hbelow_next.
  pose proof Hgood
    as (Hgood_vars & Hgood_conts & Hgood_statics & Hgood_heap).
  destruct (state_below rs Hfresh) as (Hbelow_vars & _ & _).
  destruct (proj1 (symbols_fresh_parts rs) Hfresh)
    as ((Hexpr_below & _) & _ & Hbase_next & _).
  rewrite He in Hexpr_below. cbn [expr_below] in Hexpr_below.
  apply andb_true_iff in Hexpr_below as [Hbindings_below _].
  assert (Hb :
    ∀ s : simple,
      In s (flat_map simples_of_binding bindings) ->
      simple_below (rs_base rs) s = true).
  { intros s Hin. apply (head_simples_below rs s Hfresh). rewrite He.
    exact Hin. }
  assert (Hheap :
    ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true)
    by exact (heap_symbol_extends rs rs' new_values eq_refl Hvalues_eq).
  assert (Hgrows : store_grows rs rs').
  { constructor.
    - apply Pos.le_refl.
    - exact Hheap.
    - intros s H. exact H.
    - intros m0 _. reflexivity.
    - intros m0 H. exact H.
    - intros l _. reflexivity.
    - intros l c' w _ Hc' Hw. left. exists c'.
      split; [ exact Hc' | exact Hw ]. }
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - exact Hheap.
    - intros s _.
      change (static_values (live_statics rs'))
        with (live_locals (rs_frames rs) ++ static_values statics').
      change (static_values (live_statics rs))
        with (live_locals (rs_frames rs)
              ++ static_values (state_statics (rs_state rs))).
      rewrite !lookup_app.
      destruct (lookup s (live_locals (rs_frames rs))) as [a | ];
        [ left; reflexivity | ].
      destruct (lookup s (static_values (state_statics (rs_state rs))))
        as [a | ] eqn:Hs.
      + left. exact (proj2 (eval_static_bindings_extend _ _ _ _ Heval) s a Hs).
      + intros Hdrawn.
        rewrite
          (eval_static_bindings_lookup _ _ _ _ (rs_base rs) s Hbindings_below
             (proj1 (Pos.leb_le _ _) Hdrawn) Heval).
        exact Hs.
    - intros l _ H. exact H. }
  apply (moved_checked cert codes rs' cs ms Hinv' Hordered').
  - destruct Hcode as [Hdistinct Hcodes]. split; [ exact Hdistinct | ].
    intros id c Hlookup.
    change (lookup id (static_code statics') = Some c) in Hlookup.
    rewrite Hcode_eq, lookup_app in Hlookup.
    destruct (lookup id new_code) as [c0 | ] eqn:Hid;
      [ | exact (Hcodes id c Hlookup) ].
    injection Hlookup as ->.
    pose proof (Hnew_code id c (lookup_in _ _ _ Hid)) as Hbound.
    split.
    + apply lookup_NoDup_in;
        [ exact (proj1 (distinct_NoDup _) Hdistinct) | ].
      apply Hincl. rewrite He. cbn [code_of_expr]. apply in_or_app. left.
      apply in_flat_map. exists (Bind_code id c).
      split; [ exact Hbound | left; reflexivity ].
    + exact (forallb_in _ _ _ Hbindings_check Hbound).
  - exact Hlocal.
  - exact (frames_deep_mono rs rs' _ Hheap Hdeep).
  - unfold statics_deep.
    change (static_values (state_statics (rs_state rs')))
      with (static_values statics').
    rewrite Hvalues_eq, forallb_app. apply andb_true_iff. split.
    + apply forallb_forall. intros [s a] Hin.
      destruct (Hnew_values s a Hin) as (const & Hconst_in & Hconst).
      destruct (eval_static_const_makes _ const a Hconst)
        as [Hconst_values Hconst_symbols].
      pose proof (forallb_in _ _ _ Hbindings_below Hconst_in) as Hconst_below.
      cbn [binding_below] in Hconst_below.
      apply andb_true_iff in Hconst_below as [_ Hconst_below].
      unfold static_const_below in Hconst_below.
      apply andb_true_iff in Hconst_below as [_ Hsymbols_below].
      apply allocated_all_of_parts.
      * intros w Hw. destruct (Hconst_values w Hw) as (s0 & Hs0_in & Hs0).
        assert (Hin0 : In s0 (flat_map simples_of_binding bindings)).
        { apply in_flat_map. exists (Bind_symbol s const).
          split; [ exact Hconst_in | exact Hs0_in ]. }
        apply (value_all_mono (heap_symbol rs) (heap_symbol rs') w Hheap).
        pose proof
          (simple_forever rs vars cs ms Hrun s0 w (Hb s0 Hin0)
             (forallb_in _ _ s0 Hforever Hin0) Hs0)
          as Hw_ever.
        unfold value_for_ever in Hw_ever.
        apply andb_true_iff in Hw_ever as [Hw_ever _]. exact Hw_ever.
      * intros s0 Hs0. rewrite Hconst_symbols in Hs0.
        destruct const as [ | slot closure_symbols decls slots | | | | ];
          try contradiction.
        apply in_map_iff in Hs0 as ([fslot s1] & <- & Hin1).
        pose proof (forallb_in _ _ _ Hsymbols_below Hin1) as Hbelow1.
        cbv beta iota in Hbelow1. apply Hheap.
        exact (below_base_heap rs s1 Hbelow1).
    + pose proof Hstatics as Hst. unfold statics_deep in Hst. revert Hst.
      apply forallb_impl. intros [s a] _. apply allocated_all_mono.
      exact Hheap.
  - destruct (cells_deep_parts rs Hcells)
      as (Hc_frames & Hc_statics & Hc_heap).
    apply cells_deep_build.
    + exact
        (frames_deep_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
           Hc_frames).
    + change (static_values (state_statics (rs_state rs')))
        with (static_values statics').
      rewrite Hvalues_eq, forallb_app. apply andb_true_iff. split.
      * apply forallb_forall. intros [s a] Hin.
        destruct (Hnew_values s a Hin) as (const & Hconst_in & Hconst).
        destruct (eval_static_const_makes _ const a Hconst)
          as [Hconst_values _].
        rewrite values_of_static_values. apply forallb_forall. intros w Hw.
        destruct (Hconst_values w Hw) as (s0 & Hs0_in & Hs0).
        assert (Hin0 : In s0 (flat_map simples_of_binding bindings)).
        { apply in_flat_map. exists (Bind_symbol s const).
          split; [ exact Hconst_in | exact Hs0_in ]. }
        pose proof
          (simple_forever rs vars cs ms Hrun s0 w (Hb s0 Hin0)
             (forallb_in _ _ s0 Hforever Hin0) Hs0)
          as Hw_ever.
        unfold value_for_ever in Hw_ever.
        apply andb_true_iff in Hw_ever as [_ Hw_ever]. exact Hw_ever.
      * exact
          (statics_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
             Hc_statics).
    + change
        (forallb (fun '(l, c) => cell_deep rs' l c)
           (state_heap (rs_state rs))
         = true).
      apply forallb_forall. intros [l c] Hin.
      apply (cell_deep_same rs rs' l c eq_refl (fun _ _ _ _ => eq_refl)
               Hheap);
        [ | exact (forallb_in _ _ (l, c) Hc_heap Hin) ].
      intros m1 fr' _ Hfrom. exists fr'. split; [ exact Hfrom | ].
      split; intros ? H; exact H.
  - apply (moved_values cert codes rs rs' [] Hinv Hmoves Hgood).
    + reflexivity.
    + apply Forall_nil.
    + reflexivity.
    + intros s a Hin.
      change (static_values (live_statics rs'))
        with (live_locals (rs_frames rs) ++ static_values statics')
        in Hin.
      rewrite Hvalues_eq in Hin.
      apply in_app_or in Hin as [Hin | Hin];
        [ left; apply in_or_app; left; exact Hin | ].
      apply in_app_or in Hin as [Hin | Hin];
        [ right | left; apply in_or_app; right; exact Hin ].
      destruct (Hnew_values s a Hin) as (const & Hconst_in & Hconst).
      destruct (eval_static_const_makes _ const a Hconst)
        as [Hconst_values _].
      apply allocated_values_in. intros w Hw.
      destruct (Hconst_values w Hw) as (s0 & Hs0_in & Hs0).
      apply (good_moves cert codes rs rs' Hmoves w). split.
      * exact (simple_good cert codes rs vars Hgood_vars s0 w Hs0).
      * split; [ | exact (simple_closed s0 w Hs0) ].
        apply (eval_simple_below (rs_base rs) (rs_next rs) vars s0 w
                 Hbase_next);
          [ | exact Hbelow_vars | exact Hs0 ].
        apply Hb. apply in_flat_map. exists (Bind_symbol s const).
        split; [ exact Hconst_in | exact Hs0_in ].
    + intros l c Hin. left. exact Hin.
  - exact Hbody.
  - pose proof Hincl as Hi. rewrite He in Hi. cbn [code_of_expr] in Hi.
    intros b Hin. apply Hi. apply in_or_app. right. exact Hin.
  - split; [ exact Hopen | ]. split; [ exact Hplaced | ].
    split; [ exact Hscoped | ]. split.
    + intros x l v Hl Hv.
      exact (grown_holds rs' None x l v Hgrows Hv (Hvalues x l v Hl Hv)).
    + intros x l v Hl Hv.
      exact (grown_contents rs' None x l v Hgrows Hv (Hcontents x l v Hl Hv)).
  - apply (grown_conts rs' _ Hgrows).
    split; [ | split ].
    + intros r Hr. split; [ reflexivity | exact Hr ].
    + intros m0 Hin. left. exact Hin.
    + apply Pos.le_refl.
Qed.

(* ======================================================================== *)
(* A [Let] that changes the heap                                            *)
(* ======================================================================== *)

(* An entry is put in front of the heap, and a variable is bound to a
   value that lives for ever: a cell that is made, at a location that
   the heap did not bind, the variable being bound to the location; or
   the cell of a location with a value stored in it. Each value of
   the entry was in the cell that the heap bound the location to, or
   lives for ever. *)
Lemma heap_added_checked (x : variable) (v : value) (body : expr)
    (l : location) (c' : cell) (rd' : read_record) (rs' : region_state) :
  rs'
  = Region_state
      (State body ((x, v) :: vars) conts (state_statics (rs_state rs))
         ((l, c') :: state_heap (rs_state rs)) rd')
      (rs_frames rs) (rs_base rs) (rs_next rs) (rs_counts rs) (rs_cells rs) ->
  incl (code_of_expr body) codes ->
  not_a_region cs x = true ->
  check_expr cert codes (bind_lifetime x Forever cs) body = true ->
  (∀ w : value,
     In w (values_of_cell c') ->
     (∃ c : cell,
        lookup l (state_heap (rs_state rs)) = Some c
        ∧ In w (values_of_cell c))
     ∨ (value_for_ever rs w = true ∧ good cert codes rs w ∧ old rs w)) ->
  good cert codes rs' v ->
  value_for_ever rs v = true ->
  (∀ l0 : location,
     v = Value_location l0 ->
     l0 = l ∧ lookup l (state_heap (rs_state rs)) = None) ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros -> Hincl' Hx Hbody Hvalues_c Hgood_v Hever_v Hloc_v Hinv' Hordered'.
  match goal with
  | |- state_checked _ _ ?t => set (rs' := t) in *
  end.
  pose proof Hrun as (Hopen & Hplaced & Hscoped & Hvalues & Hcontents).
  pose proof (not_region_not_open cs x Hscoped Hx) as Hx_open.
  pose proof Hgood
    as (Hgood_vars & Hgood_conts & Hgood_statics & Hgood_heap).
  destruct (cells_deep_parts rs Hcells)
    as (Hc_frames & Hc_statics & Hc_heap).
  destruct (state_old rs Hinv) as (_ & _ & _ & Hold_heap).
  assert (Hgrows : store_grows rs rs').
  { constructor.
    - apply Pos.le_refl.
    - intros s H. exact H.
    - intros s H. exact H.
    - intros m0 _. reflexivity.
    - intros m0 H. exact H.
    - intros l0 _. reflexivity.
    - intros l0 c0 w _ Hc0 Hw.
      change (lookup l0 ((l, c') :: state_heap (rs_state rs)) = Some c0)
        in Hc0.
      rewrite lookup_cons in Hc0.
      destruct (Pos.eqb l0 l) eqn:Hl0.
      + apply Pos.eqb_eq in Hl0. subst l0. injection Hc0 as <-.
        destruct (Hvalues_c w Hw) as [Hold | [Hever _]];
          [ left; exact Hold | right; exact Hever ].
      + left. exists c0. split; [ exact Hc0 | exact Hw ]. }
  assert (Hmoves : store_moves rs rs').
  { constructor.
    - reflexivity.
    - intros s H. exact H.
    - intros s _.
      change (static_values (live_statics rs'))
        with (static_values (live_statics rs)).
      destruct (lookup s (static_values (live_statics rs))) eqn:Hs;
        [ left; reflexivity | intros _; reflexivity ].
    - intros l0 _ H. exact H. }
  apply (moved_checked cert codes rs' (bind_lifetime x Forever cs) ms Hinv'
           Hordered').
  - exact Hcode.
  - exact Hlocal.
  - exact (frames_deep_mono rs rs' _ (fun s H => H) Hdeep).
  - exact Hstatics.
  - apply cells_deep_build;
      [ exact
          (frames_deep_cells_same rs rs' _ (fun _ _ _ _ _ _ _ => eq_refl)
             Hc_frames)
      | exact Hc_statics
      | ].
    change
      (forallb (fun '(l0, c0) => cell_deep rs l0 c0)
         ((l, c') :: state_heap (rs_state rs))
       = true).
    cbn [forallb]. apply andb_true_iff. split; [ | exact Hc_heap ].
    unfold cell_deep.
    destruct (lookup l (rs_cells rs)) as [m | ] eqn:Hlisted.
    + destruct (frames_from m (rs_frames rs)) as [fr | ] eqn:Hfrom;
        [ | reflexivity ].
      apply forallb_forall. intros w Hw.
      destruct (Hvalues_c w Hw) as [(c & Hc & Hw_c) | [Hever _]].
      * pose proof (forallb_in _ _ (l, c) Hc_heap (lookup_in _ _ _ Hc))
          as Hcd.
        cbv beta iota in Hcd. unfold cell_deep in Hcd.
        rewrite Hlisted, Hfrom in Hcd. exact (forallb_in _ _ w Hcd Hw_c).
      * unfold value_for_ever in Hever.
        apply andb_true_iff in Hever as [H1 H2].
        apply andb_true_iff. split.
        -- revert H1. apply value_all_mono. intros s H. rewrite H.
           reflexivity.
        -- revert H2. apply value_all_cells_mono. intros l0 H. rewrite H.
           reflexivity.
    + apply forallb_forall. intros w Hw.
      destruct (Hvalues_c w Hw) as [(c & Hc & Hw_c) | [Hever _]];
        [ | exact Hever ].
      pose proof (forallb_in _ _ (l, c) Hc_heap (lookup_in _ _ _ Hc)) as Hcd.
      cbv beta iota in Hcd. unfold cell_deep in Hcd.
      rewrite Hlisted in Hcd. exact (forallb_in _ _ w Hcd Hw_c).
  - apply (moved_values cert codes rs rs' [(x, v)] Hinv Hmoves Hgood).
    + reflexivity.
    + apply Forall_cons; [ exact Hgood_v | apply Forall_nil ].
    + reflexivity.
    + intros s allocated Hin. left. exact Hin.
    + intros l0 c0 [[= <- <-] | Hin]; [ right | left; exact Hin ].
      apply Forall_forall. intros w Hw.
      apply (good_moves cert codes rs rs' Hmoves w).
      destruct (Hvalues_c w Hw) as [(c & Hc & Hw_c) | (_ & Hgood_w & Hold_w)];
        [ | exact (conj Hgood_w Hold_w) ].
      split.
      * exact
          (proj1 (Forall_forall _ _)
             (proj1 (Forall_forall _ _) Hgood_heap (l, c)
                (lookup_in _ _ _ Hc))
             w Hw_c).
      * exact (Hold_heap l c w (lookup_in _ _ _ Hc) Hw_c).
  - exact Hbody.
  - exact Hincl'.
  - split; [ | split; [ | split; [ | split ] ] ].
    + change (open_marks ((x, v) :: vars) (cs_open cs) = Some ms).
      rewrite (open_marks_bind vars x v _ Hx_open). exact Hopen.
    + exact Hplaced.
    + apply bind_lifetimes_scoped; [ exact Hscoped | | ];
        intros r H; discriminate H.
    + intros y l0 w.
      change
        (lookup y ((x, Forever) :: cs_env cs) = Some l0 ->
         lookup y ((x, v) :: vars) = Some w ->
         lifetime_holds rs' (cs_open cs) ms None l0 w).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x); [ intros [= <-] [= <-]; exact Hever_v | ].
      intros Hl Hw.
      exact (grown_holds rs' None y l0 w Hgrows Hw (Hvalues y l0 w Hl Hw)).
    + intros y l0 w.
      change
        (lookup y ((x, Forever) :: cs_contents cs) = Some l0 ->
         lookup y ((x, v) :: vars) = Some w ->
         contents_hold rs' (cs_open cs) ms None l0 w).
      rewrite !lookup_cons.
      destruct (Pos.eqb y x).
      * intros [= <-] [= <-].
        assert (Hcase :
          (∃ l1 : location, v = Value_location l1)
          ∨ contents_hold rs' (cs_open cs) ms None Forever v
            = (value_for_ever rs v = true)).
        { destruct v;
            first [ right; reflexivity | left; eexists; reflexivity ]. }
        destruct Hcase as [[l1 Hv_eq] | Hcase];
          [ | rewrite Hcase; exact Hever_v ].
        destruct (Hloc_v l1 Hv_eq) as [-> Hnone]. rewrite Hv_eq.
        intros c0 w Hc0 Hw.
        change (lookup l ((l, c') :: state_heap (rs_state rs)) = Some c0)
          in Hc0.
        rewrite lookup_head in Hc0. injection Hc0 as <-.
        destruct (Hvalues_c w Hw) as [(c & Hc & _) | [Hever _]];
          [ | exact Hever ].
        rewrite Hnone in Hc. discriminate Hc.
      * intros Hl Hw.
        exact
          (grown_contents rs' None y l0 w Hgrows Hw (Hcontents y l0 w Hl Hw)).
  - apply (grown_conts rs' _ Hgrows). apply bound_extends.
    + exact Hx.
    + intros r Hr. exact Hr.
    + intros m0 Hin. left. exact Hin.
    + apply Pos.le_refl.
Qed.

(* A store is no allocation, and what it binds lives for ever. *)
Lemma stored_no_allocation (d : named) (target new : simple) :
  stored d = Some (target, new) ->
  is_allocation d = false
  ∧ is_cell_allocation d = false
  ∧ ∀ cs0 : check_state,
      named_lifetime cs0 d = Forever ∧ named_contents cs0 d = Forever.
Proof.
  destruct d as [s | [u | b | va | q | rp] args | | | | | ];
    try discriminate.
  destruct q as [mut tag | kind init field | kind | kind set_kind];
    try discriminate; intros _;
    (split; [ reflexivity | ]; split; [ reflexivity | ]; intros cs0;
     split; reflexivity).
Qed.

(* A [Let] of a primitive of the heap that is no local allocation. *)
Lemma heap_let_checked (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) (rs' : region_state) :
  state_expr (rs_state rs) = Let x kind d body ->
  named_changes_heap d = true ->
  not_a_region cs x = true ->
  (if is_allocation d || is_cell_allocation d
   then
     forallb (fun s => is_forever (lifetime_of cs s)) (simples_of_named d)
     && check_expr cert codes (bind_lifetime x Forever cs) body
   else
     match stored d with
     | Some (_, new) => is_forever (lifetime_of cs new)
     | None => true
     end
     && check_expr cert codes
          (bind_lifetimes x (named_lifetime cs d) (named_contents cs d) cs)
          body)
  = true ->
  plain_step rs = Region_step rs' ->
  region_invariant rs' = true ->
  frames_ordered rs' = true ->
  state_checked cert codes rs'.
Proof.
  intros He Hchanges Hx Hrest Hstep Hinv' Hordered'.
  destruct (plain_step_let rs rs' x kind d body He Hstep)
    as (v & hp' & rd' & Hexec & ->).
  pose proof Hincl as Hi. rewrite He in Hi. cbn [code_of_expr] in Hi.
  pose proof Hgood as (Hgood_vars & _).
  assert (Hsimples :
    evaluated_simples (Let x kind d body) = simples_of_named d).
  { destruct d as [s | [u | b | v0 | hp | p] args | | | | | ];
      first [ reflexivity | discriminate ]. }
  assert (Hb :
    ∀ s : simple,
      In s (simples_of_named d) -> simple_below (rs_base rs) s = true).
  { intros s Hin. apply (head_simples_below rs s Hfresh).
    rewrite He, Hsimples. exact Hin. }
  destruct (exec_named_heap_cases _ _ _ _ _ _ _ _ Hexec)
    as [[Hsame _]
       | [(His & vs & c & Hvs & Hc & -> & ->)
         | (target & new_simple & l & new & c & c' & i & Hstored & Htarget
            & Hnew & Hlookup & Hwith & -> & ->)]].
  - rewrite Hsame in Hchanges. discriminate.
  - (* A cell is made. *)
    rewrite His, orb_true_r in Hrest.
    apply andb_true_iff in Hrest as [Hforever Hbody].
    pose proof (FlambdaHeapProofs.fresh_location (state_heap (rs_state rs)))
      as Hfresh_l.
    assert (Hunlisted :
      lookup (FlambdaHeapSpec.fresh_location (state_heap (rs_state rs)))
        (rs_cells rs)
      = None).
    { destruct (lookup _ (rs_cells rs)) as [m | ] eqn:Hl; [ | reflexivity ].
      destruct (listed_mark _ m Hl) as (Hbound & _).
      apply is_bound_lookup in Hbound as (c0 & Hc0).
      rewrite Hfresh_l in Hc0. discriminate. }
    apply (heap_added_checked x _ body _ c rd' _ eq_refl Hi Hx Hbody).
    + intros w Hw. right. rewrite Hc in Hw.
      destruct (Hvs w Hw) as (sim & Hsim_in & Hsim).
      split; [ | split ].
      * exact
          (simple_forever rs vars cs ms Hrun sim w (Hb sim Hsim_in)
             (forallb_in _ _ sim Hforever Hsim_in) Hsim).
      * exact (simple_good cert codes rs vars Hgood_vars sim w Hsim).
      * exact (simple_old sim w (Hb sim Hsim_in) Hsim).
    + split; reflexivity.
    + unfold value_for_ever. cbn [value_all value_all_cells].
      unfold heap_location. rewrite Hunlisted. reflexivity.
    + intros l0 [= <-]. split; [ reflexivity | exact Hfresh_l ].
    + exact Hinv'.
    + exact Hordered'.
  - (* A store. *)
    destruct (stored_no_allocation d target new_simple Hstored)
      as (Halloc & Hcell & Hforever_d).
    rewrite Halloc, Hcell, Hstored, (proj1 (Hforever_d cs)),
      (proj2 (Hforever_d cs))
      in Hrest.
    cbn [orb] in Hrest. apply andb_true_iff in Hrest as [Hnew_forever Hbody].
    destruct (stored_simples d target new_simple Hstored) as [_ Hnew_in].
    apply (heap_added_checked x unit_value body l c' rd' _ eq_refl Hi Hx
             Hbody).
    + intros w Hw.
      destruct (cell_with_values i new c c' w Hwith Hw) as [-> | Hold].
      * right. split; [ | split ].
        -- exact
             (simple_forever rs vars cs ms Hrun new_simple new
                (Hb _ Hnew_in) Hnew_forever Hnew).
        -- exact
             (simple_good cert codes rs vars Hgood_vars new_simple new Hnew).
        -- exact (simple_old new_simple new (Hb _ Hnew_in) Hnew).
      * left. exists c. split; [ exact Hlookup | exact Hold ].
    + split; reflexivity.
    + reflexivity.
    + intros l0 H. discriminate H.
    + exact Hinv'.
    + exact Hordered'.
Qed.

End Store.
