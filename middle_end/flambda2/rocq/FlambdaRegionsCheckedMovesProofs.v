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
   the values that are good stay good when the store moves: when symbols
   are bound, when a cell is made, and when frames are popped. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaBindersProofs FlambdaValueSpec
  FlambdaHeapSpec FlambdaHeapProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaClosuresProofs FlambdaRegionsSpec
  FlambdaRegionsProofs FlambdaRegionsReadsProofs FlambdaRegionsFreshProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsCellsProofs
  FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs FlambdaRegionsCheckedUnknownProofs
  FlambdaRegionsCheckedParamsProofs FlambdaRegionsCheckedJumpProofs
  FlambdaRegionsCheckedStateProofs.
Import ListNotations.

(* A plain value stays plain with more symbols, and with a set of
   locations that has every one of the first set that the value
   names. *)
Lemma value_plain_in (ok ok' : symbol -> bool)
    (ok_cell ok_cell' : location -> bool) (v : value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  (∀ l : location,
    In l (locations_of_value v) -> ok_cell l = true -> ok_cell' l = true) ->
  value_plain ok ok_cell v = true ->
  value_plain ok' ok_cell' v = true.
Proof.
  intros Hok.
  assert (Hboth :
    ∀ w : value,
      (∀ l : location,
        In l (locations_of_value w) -> ok_cell l = true -> ok_cell' l = true) ->
      value_all ok w && value_all_cells ok_cell w = true ->
      value_all ok' w && value_all_cells ok_cell' w = true).
  { intros w Hcells H. apply andb_true_iff in H as [H H'].
    rewrite (value_all_mono ok ok' w Hok H). cbn [andb].
    apply value_all_cells_iff. intros l Hl.
    exact (Hcells l Hl (proj1 (value_all_cells_iff ok_cell w) H' l Hl)). }
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    try (exact (Hboth _)); try (intros _ _; reflexivity).
  cbn [value_plain locations_of_value]. intros Hcells H.
  apply andb_true_iff in H as [Hcallee Hargs].
  rewrite (IHcallee (fun l Hl => Hcells l (in_or_app _ _ l (or_introl Hl)))
             Hcallee).
  cbn [andb].
  apply forallb_forall. intros arg Hin. rewrite Forall_forall in IHargs.
  apply (IHargs _ Hin); [ | exact (forallb_in _ _ arg Hargs Hin) ].
  intros l Hl. apply Hcells, in_or_app. right. apply in_flat_map.
  exists arg. split; assumption.
Qed.

Section Moves.

Variable cert : certificate.

Variable codes : list (code_id * code).

(* The store moves: the first symbol that the machine drew is the
   same; a symbol of the heap stays one; of a symbol drawn so far (or
   of the term), what it is bound to is kept, or it comes to be bound
   to nothing and is a symbol that the machine drew; one that the
   machine drew and that is bound to nothing stays so; a location of
   the heap that the heap binds stays one of the heap: a cell that is
   made is at a location that the heap did not bind. *)
Record store_moves (rs rs' : region_state) : Prop := Store_moves {
  moves_base : rs_base rs' = rs_base rs;
  moves_heap :
    ∀ s : symbol, heap_symbol rs s = true -> heap_symbol rs' s = true;
  moves_bound :
    ∀ s : symbol,
      (s < rs_next rs)%positive ->
      match lookup s (static_values (live_statics rs)) with
      | Some allocated =>
        lookup s (static_values (live_statics rs')) = Some allocated
        ∨ (lookup s (static_values (live_statics rs')) = None
           ∧ drawn rs s = true)
      | None =>
        drawn rs s = true ->
        lookup s (static_values (live_statics rs')) = None
      end;
  moves_cells :
    ∀ l : location,
      is_bound l (state_heap (rs_state rs)) = true ->
      heap_location rs l = true ->
      heap_location rs' l = true
}.

Variables rs rs' : region_state.

Hypothesis Hmoves : store_moves rs rs'.

(* A value that the first state may hold: it names no symbol that is
   not drawn yet, and the heap binds every location that it names. *)
Definition old (v : value) : Prop :=
  value_below (rs_next rs) v = true
  ∧ value_closed (state_heap (rs_state rs)) v.

Lemma moves_heap_value (v : value) :
  value_all (heap_symbol rs) v = true ->
  value_all (heap_symbol rs') v = true.
Proof. apply value_all_mono. exact (moves_heap rs rs' Hmoves). Qed.

Lemma moves_for_ever (v : value) :
  value_closed (state_heap (rs_state rs)) v ->
  value_for_ever rs v = true ->
  value_for_ever rs' v = true.
Proof.
  intros Hclosed. unfold value_for_ever. intros H.
  apply andb_true_iff in H as [H H']. rewrite (moves_heap_value v H).
  cbn [andb]. apply value_all_cells_iff. intros l Hl.
  exact
    (moves_cells rs rs' Hmoves l (Hclosed l Hl)
       (proj1 (value_all_cells_iff _ v) H' l Hl)).
Qed.

Lemma moves_values_fit (flags : list bool) :
  ∀ args : list value,
    Forall (value_closed (state_heap (rs_state rs))) args ->
    values_fit rs flags args = true ->
    values_fit rs' flags args = true.
Proof.
  induction flags as [ | flag flags IH]; intros [ | v args] Hclosed;
    cbn [values_fit]; try (intros H; exact H).
  inversion Hclosed as [ | v0 args0 Hv Hargs]; subst.
  intros H. apply andb_true_iff in H as [Hhead Hrest].
  rewrite (IH args Hargs Hrest), andb_true_r.
  apply orb_true_iff in Hhead as [Hheap | ->]; [ | apply orb_true_r ].
  rewrite (moves_for_ever v Hv Hheap). reflexivity.
Qed.

Lemma moves_drawn (s : symbol) : drawn rs' s = drawn rs s.
Proof. unfold drawn. rewrite (moves_base rs rs' Hmoves). reflexivity. Qed.

(* A partial application that may be called stays one. *)
Lemma pap_moves (callee : value) (args : list value) (args_arity : arity) :
  value_below (rs_next rs) callee = true ->
  value_closed (state_heap (rs_state rs)) callee ->
  Forall (value_closed (state_heap (rs_state rs))) args ->
  pap_fits cert codes rs callee args args_arity = true ->
  pap_fits cert codes rs' callee args args_arity = true.
Proof.
  intros Hbelow Hclosed Hclosed_args.
  assert (Hbody :
    ∀ (id : code_id) (slots : list (value_slot * value)),
      match lookup id codes, lookup id (cert_code cert) with
      | Some c, Some entry =>
        negb (code_is_tupled c)
        && (length args_arity <? length (code_params_arity c))
        && ((value_for_ever rs callee && forallb (value_for_ever rs) args)
            || (cert_partial_local cert
                && values_fit rs (ce_params_local entry) args))
      | _, _ => true
      end
      = true ->
      match lookup id codes, lookup id (cert_code cert) with
      | Some c, Some entry =>
        negb (code_is_tupled c)
        && (length args_arity <? length (code_params_arity c))
        && ((value_for_ever rs' callee
             && forallb (value_for_ever rs') args)
            || (cert_partial_local cert
                && values_fit rs' (ce_params_local entry) args))
      | _, _ => true
      end
      = true).
  { intros id slots.
    destruct (lookup id codes) as [c | ]; [ | intros H; exact H ].
    destruct (lookup id (cert_code cert)) as [entry | ];
      [ | intros H; exact H ].
    intros H. apply andb_true_iff in H as [Hfirst Hkeeps].
    rewrite Hfirst. cbn [andb].
    apply orb_true_iff in Hkeeps as [H | H];
      apply andb_true_iff in H as [H1 H2].
    - rewrite (moves_for_ever callee Hclosed H1). cbn [andb].
      assert (H3 : forallb (value_for_ever rs') args = true).
      { apply forallb_forall. intros arg Hin.
        exact
          (moves_for_ever arg (proj1 (Forall_forall _ _) Hclosed_args arg Hin)
             (forallb_in _ _ arg H2 Hin)). }
      rewrite H3. reflexivity.
    - rewrite H1, (moves_values_fit _ args Hclosed_args H2).
      apply orb_true_r. }
  unfold pap_fits.
  destruct callee
    as [c | code slots0 | slot decls slots0 | s | tag fields | elements | c
       | l | callee0 args0 arity0];
    cbn [closure_of]; try (intros H; exact H).
  - exact (Hbody code slots0).
  - destruct (lookup slot decls) as [code | ]; [ | intros H; exact H ].
    exact (Hbody code slots0).
  - cbn [value_below] in Hbelow. unfold symbol_below in Hbelow.
    apply Pos.ltb_lt in Hbelow.
    pose proof (moves_bound rs rs' Hmoves s Hbelow) as Hbound.
    destruct (lookup s (static_values (live_statics rs))) as [allocated | ].
    + destruct Hbound as [-> | [-> Hdrawn]].
      * destruct allocated
          as [code slots0 | slot symbols decls slots0 | | | | | ];
          try (rewrite moves_drawn; intros H; exact H).
        -- exact (Hbody code slots0).
        -- destruct (lookup slot decls) as [code | ];
             [ exact (Hbody code slots0) | ].
           rewrite moves_drawn. intros H. exact H.
      * intros _. rewrite moves_drawn. exact Hdrawn.
    + intros Hdrawn. rewrite (Hbound Hdrawn), moves_drawn. exact Hdrawn.
Qed.

Lemma paps_move (v : value) :
  value_below (rs_next rs) v = true ->
  value_closed (state_heap (rs_state rs)) v ->
  value_paps (pap_fits cert codes rs) v = true ->
  value_paps (pap_fits cert codes rs') v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    cbn [value_below value_paps]; try (intros _ _ H; exact H).
  - intros Hbelow Hclosed H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    apply (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hbelow Hin));
      [ | exact (forallb_in _ _ (vslot, w) H Hin) ].
    intros l Hl. apply Hclosed. cbn [locations_of_value]. apply in_flat_map.
    exists (vslot, w). split; assumption.
  - intros Hbelow Hclosed H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    apply (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hbelow Hin));
      [ | exact (forallb_in _ _ (vslot, w) H Hin) ].
    intros l Hl. apply Hclosed. cbn [locations_of_value]. apply in_flat_map.
    exists (vslot, w). split; assumption.
  - intros Hbelow Hclosed H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    apply (IH w Hin (forallb_in _ _ w Hbelow Hin));
      [ | exact (forallb_in _ _ w H Hin) ].
    intros l Hl. apply Hclosed. cbn [locations_of_value]. apply in_flat_map.
    exists w. split; assumption.
  - intros Hbelow Hclosed H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    apply (IH w Hin (forallb_in _ _ w Hbelow Hin));
      [ | exact (forallb_in _ _ w H Hin) ].
    intros l Hl. apply Hclosed. cbn [locations_of_value]. apply in_flat_map.
    exists w. split; assumption.
  - intros Hbelow Hclosed H.
    apply value_closed_partial in Hclosed as [Hclosed_callee Hclosed_args].
    apply andb_true_iff in Hbelow as [Hbelow_callee Hbelow_args].
    apply andb_true_iff in H as [H Hargs].
    apply andb_true_iff in H as [Hpap Hc].
    rewrite (pap_moves callee args args_arity Hbelow_callee Hclosed_callee
               Hclosed_args Hpap),
      (IHcallee Hbelow_callee Hclosed_callee Hc).
    cbn [andb]. apply forallb_forall. intros arg Hin.
    rewrite Forall_forall in IHargs.
    exact (IHargs _ Hin (forallb_in _ _ arg Hbelow_args Hin)
             (proj1 (Forall_forall _ _) Hclosed_args arg Hin)
             (forallb_in _ _ arg Hargs Hin)).
Qed.

(* A value that is good, and that the first state may hold, stays
   good. *)
Lemma good_moves (v : value) :
  good cert codes rs v ∧ old v -> good cert codes rs' v.
Proof.
  intros [[Hplain Hfits] [Hbelow Hclosed]]. split.
  - revert Hplain. apply value_plain_in; [ exact (moves_heap rs rs' Hmoves) | ].
    intros l Hl. exact (moves_cells rs rs' Hmoves l (Hclosed l Hl)).
  - exact (paps_move v Hbelow Hclosed Hfits).
Qed.

Lemma env_good_moves (vars : env) :
  env_values_all (good cert codes rs) vars ->
  env_values_all old vars ->
  env_values_all (good cert codes rs') vars.
Proof.
  intros Hgood Hold.
  pose proof (env_values_all_and _ _ vars Hgood Hold) as H.
  revert H. apply Forall_impl. intros binding. exact (good_moves _).
Qed.

Lemma conts_good_moves (conts : cont_env) :
  conts_values_all (good cert codes rs) conts ->
  conts_values_all old conts ->
  conts_values_all (good cert codes rs') conts.
Proof.
  intros Hgood Hold.
  pose proof (conts_values_all_and _ _ conts Hgood Hold) as H.
  revert H. apply conts_values_all_impl. exact good_moves.
Qed.

Lemma allocated_good_moves (allocated : statically_allocated_value) :
  allocated_values old allocated ->
  allocated_values (good cert codes rs) allocated ->
  allocated_values (good cert codes rs') allocated.
Proof.
  assert (Hslots :
    ∀ slots : list (value_slot * value),
      Forall (fun binding => old (snd binding)) slots ->
      Forall (fun binding => good cert codes rs (snd binding)) slots ->
      Forall (fun binding => good cert codes rs' (snd binding)) slots).
  { intros slots Hold Hgood. apply Forall_forall. intros [slot v] Hin.
    apply good_moves.
    split;
      [ exact (proj1 (Forall_forall _ _) Hgood (slot, v) Hin)
      | exact (proj1 (Forall_forall _ _) Hold (slot, v) Hin) ]. }
  assert (Hlist :
    ∀ vs : list value,
      Forall old vs ->
      Forall (good cert codes rs) vs ->
      Forall (good cert codes rs') vs).
  { intros vs Hold Hgood. apply Forall_forall. intros v Hin.
    apply good_moves.
    split;
      [ exact (proj1 (Forall_forall _ _) Hgood v Hin)
      | exact (proj1 (Forall_forall _ _) Hold v Hin) ]. }
  destruct allocated; cbn [allocated_values];
    first
      [ exact (Hslots _)
      | exact (Hlist _)
      | intros _ _; exact I
      | intros Hold Hgood; exact (good_moves _ (conj Hgood Hold)) ].
Qed.

End Moves.
