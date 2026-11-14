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
   the state that a run of a checked term starts in has the property.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsReadsProofs
  FlambdaRegionsInvariantSpec FlambdaRegionsInvariantProofs
  FlambdaRegionsErasureSpec FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec
  FlambdaRegionsCheckedSpec.
Import ListNotations.

(* A value all of whose symbols are below [base] reaches symbols of
   [ok] only, if [ok] has every symbol below [base]. *)
Lemma value_below_all (base : symbol) (ok : symbol -> bool) (v : value) :
  (∀ s : symbol, symbol_below base s = true -> ok s = true) ->
  value_below base v = true ->
  value_all ok v = true.
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
    apply andb_true_iff. split; [ exact (IHcallee Hcallee) | ].
    apply forallb_forall. intros arg Hin.
    rewrite Forall_forall in IHargs.
    exact (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)).
Qed.

(* A value all of whose symbols are below [base] is a symbol, a
   partial application of such values, or reaches symbols of [ok]
   only. *)
Lemma value_below_plain (base : symbol) (ok : symbol -> bool) (v : value) :
  (∀ s : symbol, symbol_below base s = true -> ok s = true) ->
  value_below base v = true ->
  value_plain ok v = true.
Proof.
  intros Hok.
  induction v
    as [c | code slots IH | slot decls slots IH | s | tag fields IH
       | elements IH | c | l | callee args args_arity IHcallee IHargs]
    using value_induction;
    try exact (value_below_all base ok _ Hok); try reflexivity.
  cbn [value_below value_plain]. intros H.
  apply andb_true_iff in H as [Hcallee Hargs].
  apply andb_true_iff. split; [ exact (IHcallee Hcallee) | ].
  apply forallb_forall. intros arg Hin.
  rewrite Forall_forall in IHargs.
  exact (IHargs _ Hin (forallb_in _ _ arg Hargs Hin)).
Qed.

(* A value that holds no partial application passes any test of
   them. *)
Lemma value_no_partial_paps (ok : value -> list value -> arity -> bool)
    (v : value) :
  value_no_partial v = true -> value_paps ok v = true.
Proof.
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
  - discriminate.
Qed.

Lemma initial_state_checked : checked_initial_statement.
Proof.
  intros cert statics base e Hcheck Hfresh.
  unfold region_check in Hcheck.
  apply andb_true_iff in Hcheck as [Hcheck He].
  apply andb_true_iff in Hcheck as [Hdistinct Hcode].
  apply andb_true_iff in Hdistinct as [Hdistinct Hnopap].
  set (rs := initial_region_state e statics base).
  assert (Hheap : ∀ s : symbol,
            symbol_below base s = true -> heap_symbol rs s = true).
  { intros s Hs. unfold heap_symbol, drawn, symbol_below in *. cbn.
    apply Pos.ltb_lt in Hs.
    destruct (Pos.leb_spec base s); [ lia | reflexivity ]. }
  pose proof Hfresh as Hstatics. unfold region_names_fresh in Hstatics.
  apply andb_true_iff in Hstatics as [_ Hstatics].
  unfold statics_below in Hstatics.
  apply andb_true_iff in Hstatics as [_ Hvalues].
  assert (Hdeep : statics_deep rs = true).
  { unfold statics_deep. cbn. apply forallb_forall.
    intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hvalues Hin) as Hbelow.
    cbn in Hbelow. apply andb_true_iff in Hbelow as [_ Hbelow].
    destruct allocated; cbn in *; try reflexivity.
    + apply forallb_forall. intros [slot w] Hw.
      exact (value_below_all _ _ _ Hheap (forallb_in _ _ (slot, w) Hbelow Hw)).
    + apply andb_true_iff in Hbelow as [Hsymbols Hslots].
      apply andb_true_iff. split.
      * apply forallb_forall. intros [slot' s'] Hs'.
        exact (Hheap s' (forallb_in _ _ (slot', s') Hsymbols Hs')).
      * apply forallb_forall. intros [slot' w] Hw.
        exact (value_below_all _ _ _ Hheap
                 (forallb_in _ _ (slot', w) Hslots Hw)).
    + apply forallb_forall. intros w Hw.
      exact (value_below_all _ _ _ Hheap (forallb_in _ _ w Hbelow Hw)).
    + exact (value_below_all _ _ _ Hheap Hbelow).
    + apply forallb_forall. intros w Hw.
      exact (value_below_all _ _ _ Hheap (forallb_in _ _ w Hbelow Hw)). }
  assert (Hpaps : paps_fit cert (all_code statics e) rs).
  { unfold paps_fit, state_values. cbn.
    split; [ constructor | ]. split; [ exact I | ].
    apply Forall_forall. intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hnopap Hin) as Hno.
    cbn in Hno.
    destruct allocated; cbn in *; try exact I.
    + apply Forall_forall. intros [slot w] Hw.
      exact (value_no_partial_paps _ w (forallb_in _ _ (slot, w) Hno Hw)).
    + apply Forall_forall. intros [slot' w] Hw.
      exact (value_no_partial_paps _ w (forallb_in _ _ (slot', w) Hno Hw)).
    + apply Forall_forall. intros w Hw.
      exact (value_no_partial_paps _ w (forallb_in _ _ w Hno Hw)).
    + exact (value_no_partial_paps _ _ Hno).
    + apply Forall_forall. intros w Hw.
      exact (value_no_partial_paps _ w (forallb_in _ _ w Hno Hw)). }
  assert (Hplain : values_plain rs).
  { unfold values_plain, state_values. cbn.
    split; [ constructor | ]. split; [ exact I | ].
    apply Forall_forall. intros [s allocated] Hin.
    pose proof (forallb_in _ _ (s, allocated) Hvalues Hin) as Hbelow.
    cbn in Hbelow. apply andb_true_iff in Hbelow as [_ Hbelow].
    destruct allocated; cbn in *; try exact I.
    + apply Forall_forall. intros [slot w] Hw.
      exact
        (value_below_plain _ _ w Hheap (forallb_in _ _ (slot, w) Hbelow Hw)).
    + apply andb_true_iff in Hbelow as [_ Hslots].
      apply Forall_forall. intros [slot' w] Hw.
      exact
        (value_below_plain _ _ w Hheap
           (forallb_in _ _ (slot', w) Hslots Hw)).
    + apply Forall_forall. intros w Hw.
      exact (value_below_plain _ _ w Hheap (forallb_in _ _ w Hbelow Hw)).
    + exact (value_below_plain _ _ _ Hheap Hbelow).
    + apply Forall_forall. intros w Hw.
      exact (value_below_plain _ _ w Hheap (forallb_in _ _ w Hbelow Hw)). }
  assert (Hcodes : code_checked cert (all_code statics e) rs).
  { split; [ exact Hdistinct | ]. intros id c H. cbn in H. split.
    - unfold all_code. rewrite lookup_app, H. reflexivity.
    - apply lookup_in in H. exact (forallb_in _ _ (id, c) Hcode H). }
  unfold state_checked.
  split; [ exact (region_invariant_initial e statics base Hfresh) | ].
  split; [ reflexivity | ].
  split; [ exact Hcodes | ].
  split; [ reflexivity | ].
  split; [ reflexivity | ].
  split; [ exact Hdeep | ].
  split; [ exact Hplain | ].
  split; [ exact Hpaps | ].
  exists initial_check_state, [].
  split; [ exact He | ].
  split; [ unfold all_code; apply incl_appr, incl_refl | ].
  split.
  - repeat split. intros x l v H. discriminate.
  - split; intros; [ discriminate | reflexivity ].
Qed.
