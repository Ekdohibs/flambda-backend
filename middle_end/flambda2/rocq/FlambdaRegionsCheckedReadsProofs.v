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
   in a state that has the property, what a step reads may be read. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsStepsProofs
  FlambdaRegionsFreshProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs.
Import ListNotations.

Definition spine_live (rs : region_state) (c : cont_value) : Prop :=
  forallb (value_all (live_symbol rs)) (cont_spine_values c) = true.

(* The open regions of a caller are at its marks, which are on the
   stack. *)
Definition caller_placed (rs : region_state) (a : caller) : Prop :=
  open_marks (caller_vars a) (cs_open (caller_check a))
  = Some (caller_marks a)
  ∧ marks_placed (rs_frames rs) (caller_marks a) = true.

Lemma called_from_placed (rs : region_state) (u : user) (a : caller) :
  called_from rs u a -> caller_placed rs a.
Proof. intros (H1 & H2 & _). exact (conj H1 H2). Qed.

(* A value that a caller gave and that a continuation holds since
   reaches only symbols that may be evaluated. *)
Lemma caller_holds_live (rs : region_state) (a : caller) (l : lifetime)
    (v : value) :
  caller_placed rs a ->
  caller_holds rs a l v ->
  value_all (live_symbol rs) v = true.
Proof.
  intros [Hms Hplaced] [Husable Hholds].
  destruct l as [ | | r | ]; cbn [lifetime_usable lifetime_holds] in *.
  - exact (value_all_mono _ _ v (heap_live rs) Hholds).
  - exact (value_within_live rs _ v Hholds).
  - destruct (region_place_open _ _ (caller_marks a) (Some (caller_next a)) r
                Hms Husable)
      as (m & stop & Hplace & Hin).
    rewrite Hplace in Hholds.
    exact
      (value_within_live rs stop v
         (Hholds (marks_placed_in _ _ _ Hplaced Hin))).
  - discriminate.
Qed.

Section Spine.

Variable cert : certificate.
Variable codes : list (code_id * code).
Variable rs : region_state.

(* Where a jump goes: a way out, or a continuation that is as an entry
   says which is not that of a handler that nothing may jump to. *)
Lemma jump_spine_live (cs : check_state) (u : user) (k : continuation)
    (args : list (option simple * lifetime)) (c : cont_value) :
  jump cs k args = true ->
  (∀ ce : cont_entry,
     lookup k (cs_conts cs) = Some ce ->
     cont_ok cert codes rs u ce c
     ∧ match ce with
       | To_return _ | To_raise => spine_live rs c
       | _ => True
       end) ->
  (lookup k (cs_conts cs) = None -> c = Outer k) ->
  spine_live rs c.
Proof.
  intros Hjump Hsome Houter. unfold jump in Hjump.
  destruct (lookup k (cs_conts cs)) as [ce | ] eqn:Hk;
    [ | rewrite (Houter eq_refl); reflexivity ].
  destruct (Hsome ce eq_refl) as [Hok Hce].
  destruct ce as [params kinds entry | | local | ];
    [ | discriminate | exact Hce | exact Hce ].
  inversion Hok; subst. reflexivity.
Qed.

Lemma jump_result_jump (cs : check_state) (k : continuation) (l : lifetime) :
  jump_result cs k l = true ->
  ∃ args : list (option simple * lifetime), jump cs k args = true.
Proof.
  unfold jump_result. intros H.
  destruct (lookup k (cs_conts cs)) as [[params kinds entry | | local | ] | ];
    eexists; exact H.
Qed.

Lemma callers_spine_live :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c ->
     match ce with
     | To_return _ | To_raise => spine_live rs c
     | _ => True
     end)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c -> caller_placed rs a -> spine_live rs c)
  ∧ (∀ (a : caller) (c : cont_value),
       raise_ok cert codes rs a c -> caller_placed rs a -> spine_live rs c).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun u ce c _ =>
              match ce with
              | To_return _ | To_raise => spine_live rs c
              | _ => True
              end)
           (fun a l c _ => caller_placed rs a -> spine_live rs c)
           (fun a c _ => caller_placed rs a -> spine_live rs c)).
  - intros. exact I.
  - intros. exact I.
  - intros u local a c Hcalled Hret IH.
    exact (IH (called_from_placed rs u a Hcalled)).
  - intros u a c Hcalled Hraise IH.
    exact (IH (called_from_placed rs u a Hcalled)).
  - intros a l k c Hjump Hsome IH Houter Hplaced.
    destruct (jump_result_jump _ _ _ Hjump) as (args & Hjump').
    apply (jump_spine_live (caller_check a) (caller_user a) k args c Hjump');
      [ | exact Houter ].
    intros ce Hk. exact (conj (Hsome ce Hk) (IH ce Hk)).
  - intros a l l' ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise IHraise Hplaced.
    unfold spine_live. cbn [cont_spine_values]. rewrite forallb_app.
    apply andb_true_iff. split; [ | exact (IHret Hplaced) ].
    apply forallb_forall. intros v Hin.
    destruct (Forall2_in_right _ _ _ _ Hholds Hin) as (l0 & _ & Hv).
    exact (caller_holds_live rs a l0 v Hplaced Hv).
  - intros a l l' c Hlasts Hret IH. exact IH.
  - intros a k simples extra c Heval Hholds Hjump Hsome IH Houter Hplaced.
    assert (Hc : spine_live rs c).
    { apply (jump_spine_live (caller_check a) (caller_user a) k _ c Hjump);
        [ | exact Houter ].
      intros ce Hk. exact (conj (Hsome ce Hk) (IH ce Hk)). }
    destruct extra as [ | v0 extra]; [ exact Hc | ].
    cbn [with_extra_args]. unfold spine_live. cbn [cont_spine_values].
    rewrite forallb_app. apply andb_true_iff. split; [ | exact Hc ].
    apply forallb_forall. intros v Hin.
    destruct (Forall2_in_right _ _ _ _ Hholds Hin) as (s & _ & Hv).
    exact (caller_holds_live rs a _ v Hplaced Hv).
Qed.

End Spine.

Lemma invariant_fresh (rs : region_state) :
  region_invariant rs = true -> symbols_fresh rs = true.
Proof.
  unfold region_invariant. intros H. apply andb_true_iff in H as [H _].
  exact H.
Qed.

(* The continuation that a jump of the state that runs goes to. *)
Lemma head_jump_spine_live (cert : certificate)
    (codes : list (code_id * code)) (rs : region_state) (cs : check_state)
    (ms : list mark) (k : continuation)
    (args : list (option simple * lifetime)) :
  conts_checked cert codes rs (running_user rs cs ms) (cs_conts cs)
    (state_conts (rs_state rs)) ->
  jump cs k args = true ->
  spine_live rs (resolve k (state_conts (rs_state rs))).
Proof.
  intros [Hconts Houter] Hjump.
  apply (jump_spine_live cert codes rs cs (running_user rs cs ms) k args _
           Hjump).
  - intros ce Hk. split; [ exact (Hconts k ce Hk) | ].
    exact (proj1 (callers_spine_live cert codes rs) _ _ _ (Hconts k ce Hk)).
  - intros Hk. unfold resolve. rewrite (Houter k Hk). reflexivity.
Qed.

Lemma checked_state_reads_live : checked_reads_live_statement.
Proof.
  intros cert codes rs
    (Hinv & Hord & Hcode & Hlocal & Hframes & Hstatics & Hplain & Hpaps & cs
     & ms & Hcheck & Hincl & Hrun & Hconts).
  pose proof (invariant_fresh rs Hinv) as Hfresh.
  unfold reads_live. cbv zeta. split; [ | split ].
  - apply simples_ok_live.
    exact (head_simples_ok cert codes rs cs ms Hfresh Hcheck Hrun Hconts).
  - exact (bindings_live rs Hframes Hstatics).
  - unfold head_conts.
    destruct (state_expr (rs_state rs))
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      try reflexivity.
    + cbn [check_expr] in Hcheck.
      apply andb_true_iff in Hcheck as [_ Hcheck].
      destruct (call_result cert codes cs ap) as [result | ];
        [ | discriminate ].
      apply andb_true_iff in Hcheck as [Hreturn _].
      destruct (jump_result_jump _ _ _ Hreturn) as (args & Hjump).
      cbn [flat_map]. rewrite app_nil_r.
      exact (head_jump_spine_live cert codes rs cs ms _ args Hconts Hjump).
    + cbn [check_expr] in Hcheck. cbn [flat_map]. rewrite app_nil_r.
      exact (head_jump_spine_live cert codes rs cs ms _ _ Hconts Hcheck).
Qed.
