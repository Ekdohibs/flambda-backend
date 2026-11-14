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
   are bound, and when frames are popped. Nothing here needs review. *)

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
  FlambdaRegionsCheckedStateProofs.
Import ListNotations.

(* Two properties of every value that a continuation holds. *)
Fixpoint cont_values_all_and (P Q : value -> Prop) (c : cont_value) :
  cont_values_all P c ->
  cont_values_all Q c ->
  cont_values_all (fun v => P v ∧ Q v) c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    cbn [cont_values_all]; [ | auto | | ].
  - intros [Hvars Hconts] [Hvars' Hconts']. split.
    + apply Forall_forall. intros binding Hin.
      exact (conj (proj1 (Forall_forall _ _) Hvars binding Hin)
               (proj1 (Forall_forall _ _) Hvars' binding Hin)).
    + induction conts as [ | [k c] conts IH]; cbn [fold_right]; [ exact I | ].
      destruct Hconts as [Hc Hrest]. destruct Hconts' as [Hc' Hrest'].
      split;
        [ exact (cont_values_all_and P Q c Hc Hc') | exact (IH Hrest Hrest') ].
  - intros [Hextra Hc] [Hextra' Hc'].
    split; [ | exact (cont_values_all_and P Q c Hc Hc') ].
    apply Forall_forall. intros v Hin.
    exact (conj (proj1 (Forall_forall _ _) Hextra v Hin)
             (proj1 (Forall_forall _ _) Hextra' v Hin)).
  - intros (Hremaining & Hreturn & Hexn) (Hremaining' & Hreturn' & Hexn').
    split; [ | split ].
    + apply Forall_forall. intros v Hin.
      exact (conj (proj1 (Forall_forall _ _) Hremaining v Hin)
               (proj1 (Forall_forall _ _) Hremaining' v Hin)).
    + exact (cont_values_all_and P Q return_to Hreturn Hreturn').
    + exact (cont_values_all_and P Q exn_to Hexn Hexn').
Qed.

Lemma conts_values_all_and (P Q : value -> Prop) (conts : cont_env) :
  conts_values_all P conts ->
  conts_values_all Q conts ->
  conts_values_all (fun v => P v ∧ Q v) conts.
Proof.
  unfold conts_values_all.
  induction conts as [ | [k c] conts IH]; cbn [fold_right]; [ auto | ].
  intros [Hc Hrest] [Hc' Hrest'].
  split; [ exact (cont_values_all_and P Q c Hc Hc') | exact (IH Hrest Hrest') ].
Qed.

Lemma env_values_all_and (P Q : value -> Prop) (vars : env) :
  env_values_all P vars ->
  env_values_all Q vars ->
  env_values_all (fun v => P v ∧ Q v) vars.
Proof.
  unfold env_values_all. intros H H'. apply Forall_forall. intros binding Hin.
  exact (conj (proj1 (Forall_forall _ _) H binding Hin)
           (proj1 (Forall_forall _ _) H' binding Hin)).
Qed.

Lemma value_plain_mono (ok ok' : symbol -> bool) (v : value) :
  (∀ s : symbol, ok s = true -> ok' s = true) ->
  value_plain ok v = true ->
  value_plain ok' v = true.
Proof.
  intros Hok.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    try (exact (value_all_mono ok ok' _ Hok)); try (intros _; reflexivity).
  cbn [value_plain]. intros H. apply andb_true_iff in H as [Hcallee Hargs].
  rewrite (IHcallee Hcallee). cbn [andb].
  apply forallb_forall. intros arg Hin. rewrite Forall_forall in IHargs.
  exact (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)).
Qed.

Section Moves.

Variable cert : certificate.

Variable codes : list (code_id * code).

(* The store moves: the first symbol that the machine drew is the
   same; a symbol of the heap stays one; of a symbol drawn so far (or
   of the term), what it is bound to is kept, or it comes to be bound
   to nothing and is a symbol that the machine drew; one that the
   machine drew and that is bound to nothing stays so. *)
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
      end
}.

Variables rs rs' : region_state.

Hypothesis Hmoves : store_moves rs rs'.

Lemma moves_heap_value (v : value) :
  value_all (heap_symbol rs) v = true ->
  value_all (heap_symbol rs') v = true.
Proof. apply value_all_mono. exact (moves_heap rs rs' Hmoves). Qed.

Lemma moves_values_fit (flags : list bool) :
  ∀ args : list value,
    values_fit rs flags args = true -> values_fit rs' flags args = true.
Proof.
  induction flags as [ | flag flags IH]; intros [ | v args]; cbn [values_fit];
    try (intros H; exact H).
  intros H. apply andb_true_iff in H as [Hhead Hrest].
  rewrite (IH args Hrest), andb_true_r.
  apply orb_true_iff in Hhead as [Hheap | ->]; [ | apply orb_true_r ].
  rewrite (moves_heap_value v Hheap). reflexivity.
Qed.

Lemma moves_drawn (s : symbol) : drawn rs' s = drawn rs s.
Proof. unfold drawn. rewrite (moves_base rs rs' Hmoves). reflexivity. Qed.

(* A partial application that may be called stays one. *)
Lemma pap_moves (callee : value) (args : list value) (args_arity : arity) :
  value_below (rs_next rs) callee = true ->
  pap_fits cert codes rs callee args args_arity = true ->
  pap_fits cert codes rs' callee args args_arity = true.
Proof.
  intros Hbelow.
  assert (Hbody :
    ∀ (id : code_id) (slots : list (value_slot * value)),
      match lookup id codes, lookup id (cert_code cert) with
      | Some c, Some entry =>
        negb (code_is_tupled c)
        && (length args_arity <? length (code_params_arity c))
        && ((value_all (heap_symbol rs) callee
             && forallb (value_all (heap_symbol rs)) args)
            || (cert_partial_local cert
                && values_fit rs (ce_params_local entry) args))
      | _, _ => true
      end
      = true ->
      match lookup id codes, lookup id (cert_code cert) with
      | Some c, Some entry =>
        negb (code_is_tupled c)
        && (length args_arity <? length (code_params_arity c))
        && ((value_all (heap_symbol rs') callee
             && forallb (value_all (heap_symbol rs')) args)
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
    - rewrite (moves_heap_value callee H1). cbn [andb].
      assert (H3 : forallb (value_all (heap_symbol rs')) args = true).
      { apply forallb_forall. intros arg Hin.
        exact (moves_heap_value arg (forallb_in _ _ arg H2 Hin)). }
      rewrite H3. reflexivity.
    - rewrite H1, (moves_values_fit _ args H2). apply orb_true_r. }
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
  value_paps (pap_fits cert codes rs) v = true ->
  value_paps (pap_fits cert codes rs') v = true.
Proof.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    cbn [value_below value_paps]; try (intros _ H; exact H).
  - intros Hbelow H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hbelow Hin)
             (forallb_in _ _ (vslot, w) H Hin)).
  - intros Hbelow H. apply forallb_forall. intros [vslot w] Hin.
    rewrite Forall_forall in IH.
    exact (IH (vslot, w) Hin (forallb_in _ _ (vslot, w) Hbelow Hin)
             (forallb_in _ _ (vslot, w) H Hin)).
  - intros Hbelow H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    exact (IH w Hin (forallb_in _ _ w Hbelow Hin) (forallb_in _ _ w H Hin)).
  - intros Hbelow H. apply forallb_forall. intros w Hin.
    rewrite Forall_forall in IH.
    exact (IH w Hin (forallb_in _ _ w Hbelow Hin) (forallb_in _ _ w H Hin)).
  - intros Hbelow H.
    apply andb_true_iff in Hbelow as [Hbelow_callee Hbelow_args].
    apply andb_true_iff in H as [H Hargs].
    apply andb_true_iff in H as [Hpap Hc].
    rewrite (pap_moves callee args args_arity Hbelow_callee Hpap),
      (IHcallee Hbelow_callee Hc).
    cbn [andb]. apply forallb_forall. intros arg Hin.
    rewrite Forall_forall in IHargs.
    exact (IHargs _ Hin (forallb_in _ _ arg Hbelow_args Hin)
             (forallb_in _ _ arg Hargs Hin)).
Qed.

(* A value that is good, and names no symbol that is not drawn yet,
   stays good. *)
Lemma good_moves (v : value) :
  good cert codes rs v ∧ value_below (rs_next rs) v = true ->
  good cert codes rs' v.
Proof.
  intros [[Hplain Hfits] Hbelow]. split.
  - exact (value_plain_mono _ _ v (moves_heap rs rs' Hmoves) Hplain).
  - exact (paps_move v Hbelow Hfits).
Qed.

Lemma env_good_moves (vars : env) :
  env_values_all (good cert codes rs) vars ->
  env_values_all (fun v => value_below (rs_next rs) v = true) vars ->
  env_values_all (good cert codes rs') vars.
Proof.
  intros Hgood Hbelow.
  pose proof (env_values_all_and _ _ vars Hgood Hbelow) as H.
  revert H. apply Forall_impl. intros binding. exact (good_moves _).
Qed.

Lemma conts_good_moves (conts : cont_env) :
  conts_values_all (good cert codes rs) conts ->
  conts_values_all (fun v => value_below (rs_next rs) v = true) conts ->
  conts_values_all (good cert codes rs') conts.
Proof.
  intros Hgood Hbelow.
  pose proof (conts_values_all_and _ _ conts Hgood Hbelow) as H.
  revert H. apply conts_values_all_impl. exact good_moves.
Qed.

Lemma allocated_good_moves (allocated : statically_allocated_value) :
  allocated_below (rs_next rs) allocated = true ->
  allocated_values (good cert codes rs) allocated ->
  allocated_values (good cert codes rs') allocated.
Proof.
  assert (Hslots :
    ∀ slots : list (value_slot * value),
      forallb (fun '(_, v) => value_below (rs_next rs) v) slots = true ->
      Forall (fun binding => good cert codes rs (snd binding)) slots ->
      Forall (fun binding => good cert codes rs' (snd binding)) slots).
  { intros slots Hbelow Hgood. apply Forall_forall. intros [slot v] Hin.
    apply good_moves.
    split;
      [ exact (proj1 (Forall_forall _ _) Hgood (slot, v) Hin)
      | exact (forallb_in _ _ (slot, v) Hbelow Hin) ]. }
  assert (Hlist :
    ∀ vs : list value,
      forallb (value_below (rs_next rs)) vs = true ->
      Forall (good cert codes rs) vs ->
      Forall (good cert codes rs') vs).
  { intros vs Hbelow Hgood. apply Forall_forall. intros v Hin.
    apply good_moves.
    split;
      [ exact (proj1 (Forall_forall _ _) Hgood v Hin)
      | exact (forallb_in _ _ v Hbelow Hin) ]. }
  destruct allocated; cbn [allocated_below allocated_values];
    first
      [ exact (Hslots _)
      | exact (Hlist _)
      | intros _ _; exact I
      | intros H; apply andb_true_iff in H as [_ H]; exact (Hslots _ H)
      | intros Hbelow Hgood; exact (good_moves _ (conj Hgood Hbelow)) ].
Qed.

End Moves.
