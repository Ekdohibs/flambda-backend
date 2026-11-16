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
   the call of a callee that is not known
   (FlambdaRegionsCheckSpec.unknown_call) along a chain of calls, and that
   what takes a value of a lifetime takes one of a longer lifetime. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedHeadProofs FlambdaRegionsCheckedStableProofs
  FlambdaRegionsCheckedMarksProofs FlambdaRegionsCheckedOutlivesProofs
  FlambdaRegionsCheckedRankProofs.
Import ListNotations.

Section Unknown.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable cs : check_state.

Local Abbreviation R := (rank (cs_open cs)).

(* What a callee that is not known returns. *)
Definition unknown_lifetime : lifetime :=
  if cert_unknown_returns_local cert then newest_lifetime cs else Forever.

(* Whether a link of a call with [count] arguments may make a partial
   application. *)
Definition may_be_partial (count : nat) : bool :=
  (count <? max_full_arity codes)
  || (later_call codes count && (1 <? max_full_arity codes)).

Lemma may_be_partial_one (count : nat) :
  1 <= count -> may_be_partial count = (1 <? max_full_arity codes).
Proof.
  intros Hcount. unfold may_be_partial, later_call.
  destruct (1 <? max_full_arity codes) eqn:Hwidest.
  - apply Nat.ltb_lt in Hwidest.
    destruct (count <? max_full_arity codes) eqn:Hlt; [ reflexivity | ].
    apply Nat.ltb_ge in Hlt. cbn [orb]. rewrite andb_true_r.
    assert (H : (1 <? count) = true) by (apply Nat.ltb_lt; lia).
    rewrite H. cbn [andb]. apply orb_true_r.
  - apply Nat.ltb_ge in Hwidest. rewrite andb_false_r, orb_false_r.
    apply Nat.ltb_ge. lia.
Qed.

Lemma later_call_positive (count : nat) :
  later_call codes count = true -> 1 <= count.
Proof.
  unfold later_call. destruct count as [ | count]; [ | lia ].
  rewrite andb_false_r, orb_false_r. intros H.
  apply existsb_exists in H as ([id c] & _ & H). apply Nat.ltb_lt in H. lia.
Qed.

Lemma forallb_incl {A : Type} (f : A -> bool) (l l' : list A) :
  (∀ a : A, In a l' -> In a l) ->
  forallb f l = true -> forallb f l' = true.
Proof.
  intros Hincl Hall. apply forallb_forall. intros a Hin.
  exact (proj1 (forallb_forall f l) Hall a (Hincl a Hin)).
Qed.

(* The result is what an unknown callee returns, or no longer. *)
Lemma unknown_call_le (callee : lifetime) (args : list lifetime) (count : nat)
    (l : lifetime) :
  unknown_call cert codes cs callee args count = Some l ->
  R l <= R unknown_lifetime.
Proof.
  unfold unknown_call. cbv zeta. fold unknown_lifetime.
  destruct (forallb is_forever args || cert_indirect_local cert);
    [ | discriminate ].
  destruct (cert_partial_local cert).
  - intros [= <-]. destruct (_ || _); [ | lia ]. rewrite rank_shorter. lia.
  - destruct (_ || _); [ discriminate | ]. intros [= <-]. lia.
Qed.

(* A later link of a chain. The link before returned [first], which
   lives for ever or is what an unknown callee returns; the arguments
   are some of those of the call. *)
Lemma unknown_call_later (callee first : lifetime) (args args' : list lifetime)
    (count count' : nat) (l : lifetime) :
  unknown_call cert codes cs callee args count = Some l ->
  later_call codes count = true ->
  1 <= count' ->
  (∀ l0 : lifetime, In l0 args' -> In l0 args) ->
  first = Forever ∨ first = unknown_lifetime ->
  ∃ l' : lifetime,
    unknown_call cert codes cs first args' count' = Some l' ∧ R l <= R l'.
Proof.
  intros Hcall Hlater Hcount' Hincl Hfirst.
  pose proof (later_call_positive count Hlater) as Hcount.
  revert Hcall. unfold unknown_call. cbv zeta.
  fold unknown_lifetime. fold (may_be_partial count).
  fold (may_be_partial count').
  rewrite (may_be_partial_one count Hcount),
    (may_be_partial_one count' Hcount'), Hlater.
  destruct (forallb is_forever args) eqn:Hargs; cbn [orb negb andb].
  - rewrite (forallb_incl is_forever args args' Hincl Hargs).
    cbn [orb negb andb].
    destruct (cert_partial_local cert).
    + intros [= <-]. eexists. split; [ reflexivity | ].
      destruct (1 <? max_full_arity codes); [ | lia ].
      rewrite !rank_shorter.
      assert (Hkept :
        Nat.min (R unknown_lifetime)
          (Nat.min (R callee) (R (shortest cs args)))
        <= R (shortest cs (first :: args'))).
      { apply rank_shortest_ge.
        - pose proof (rank_le (cs_open cs) unknown_lifetime). lia.
        - intros l0 [<- | Hin].
          + destruct Hfirst as [-> | ->]; [ | lia ].
            pose proof (rank_le (cs_open cs) unknown_lifetime).
            cbn [rank]. lia.
          + pose proof (rank_shortest_le cs args l0 (Hincl l0 Hin)). lia. }
      lia.
    + destruct (negb (is_forever callee)
                && (count <? max_local_closure_arity cert codes));
        [ discriminate | ]. cbn [orb].
      destruct (negb (is_forever unknown_lifetime)
                && (1 <? max_local_closure_arity cert codes)) eqn:Hthird;
        [ discriminate | ].
      intros [= <-].
      assert (Hsecond :
        negb (is_forever first)
        && (count' <? max_local_closure_arity cert codes)
        = false).
      { destruct Hfirst as [-> | ->]; [ reflexivity | ].
        destruct (is_forever unknown_lifetime); [ reflexivity | ].
        cbn [negb andb] in *. apply Nat.ltb_ge in Hthird. apply Nat.ltb_ge.
        lia. }
      rewrite Hsecond, <- andb_assoc, Hthird, andb_false_r. cbn [orb].
      eexists. split; [ reflexivity | lia ].
  - destruct (cert_indirect_local cert); [ | discriminate ].
    rewrite orb_true_r.
    destruct (cert_partial_local cert).
    + intros [= <-]. eexists. split; [ reflexivity | ].
      destruct (1 <? max_full_arity codes); [ | lia ].
      rewrite !rank_shorter.
      assert (Hkept :
        Nat.min (R unknown_lifetime)
          (Nat.min (R callee) (R (shortest cs args)))
        <= R (shortest cs (first :: args'))).
      { apply rank_shortest_ge.
        - pose proof (rank_le (cs_open cs) unknown_lifetime). lia.
        - intros l0 [<- | Hin].
          + destruct Hfirst as [-> | ->]; [ | lia ].
            pose proof (rank_le (cs_open cs) unknown_lifetime).
            cbn [rank]. lia.
          + pose proof (rank_shortest_le cs args l0 (Hincl l0 Hin)). lia. }
      lia.
    + destruct (1 <? max_full_arity codes) eqn:Hwidest;
        [ discriminate | ]. cbn [orb andb].
      destruct (negb (is_forever callee)
                && (count <? max_local_closure_arity cert codes));
        [ discriminate | ]. cbn [orb].
      destruct (negb (is_forever unknown_lifetime)
                && (1 <? max_local_closure_arity cert codes)) eqn:Hthird;
        [ discriminate | ].
      intros [= <-].
      assert (Hsecond :
        negb (is_forever first)
        && (count' <? max_local_closure_arity cert codes)
        = false).
      { destruct Hfirst as [-> | ->]; [ reflexivity | ].
        destruct (is_forever unknown_lifetime); [ reflexivity | ].
        cbn [negb andb] in *. apply Nat.ltb_ge in Hthird. apply Nat.ltb_ge.
        lia. }
      rewrite andb_false_r, Hsecond, <- andb_assoc, Hthird, andb_false_r.
      cbn [orb]. eexists. split; [ reflexivity | lia ].
Qed.

(* A call that passes with a callee that is dead passes with any, to
   the same result unless the result is dead. *)
Lemma unknown_call_dead (callee callee' : lifetime) (args : list lifetime)
    (count : nat) (l : lifetime) :
  unknown_call cert codes cs callee args count = Some l ->
  R callee = 0 ->
  ∃ l' : lifetime,
    unknown_call cert codes cs callee' args count = Some l'
    ∧ (l' = l ∨ R l = 0).
Proof.
  intros Hcall Hdead. revert Hcall. unfold unknown_call. cbv zeta.
  destruct (forallb is_forever args || cert_indirect_local cert);
    [ | discriminate ].
  destruct (cert_partial_local cert).
  - intros [= <-]. eexists. split; [ reflexivity | ].
    destruct (_ || _); [ | left; reflexivity ].
    right. rewrite !rank_shorter. lia.
  - assert (Hcallee : is_forever callee = false).
    { destruct (is_forever callee) eqn:H; [ | reflexivity ].
      apply (is_forever_rank (cs_open cs)) in H. lia. }
    rewrite Hcallee. cbn [negb andb].
    destruct (negb (forallb is_forever args) && _); [ discriminate | ].
    cbn [orb].
    destruct (count <? max_local_closure_arity cert codes);
      [ discriminate | ].
    rewrite andb_false_r. cbn [orb].
    destruct (_ && _); [ discriminate | ].
    intros [= <-]. eexists. split; [ reflexivity | left; reflexivity ].
Qed.

End Unknown.

(* ======================================================================== *)
(* What takes a value of a lifetime takes one of a longer lifetime          *)
(* ======================================================================== *)

Lemma usable_dead (cs : check_state) (l : lifetime) :
  rank (cs_open cs) l = 0 -> lifetime_usable cs l = false.
Proof.
  intros Hdead. destruct (lifetime_usable cs l) eqn:H; [ | reflexivity ].
  apply usable_rank in H. lia.
Qed.

Lemma forever_dead (open : list variable) (l : lifetime) :
  rank open l = 0 -> is_forever l = false.
Proof.
  intros Hdead. destruct (is_forever l) eqn:H; [ | reflexivity ].
  apply (is_forever_rank open) in H. lia.
Qed.

(* A result that is dead goes only to a handler with no parameter,
   which takes a result of any lifetime. *)
Lemma jump_result_dead (cs : check_state) (k : continuation)
    (l l' : lifetime) :
  jump_result cs k l = true ->
  rank (cs_open cs) l = 0 ->
  jump_result cs k l' = true.
Proof.
  intros Hjump Hdead. revert Hjump. unfold jump_result, jump.
  pose proof (usable_dead cs l Hdead) as Husable.
  pose proof (forever_dead (cs_open cs) l Hdead) as Hforever.
  destruct (lookup k (cs_conts cs)) as [[params kinds entry | | local | ] | ];
    cbn [forallb]; rewrite ?Husable, ?Hforever; cbn [andb];
    try discriminate.
  destruct params as [ | p params]; [ intros H; exact H | ].
  cbn [map region_args].
  destruct kinds as [ | kind kinds]; [ discriminate | ].
  destruct (region_args cs params kinds _) as [rest | ]; simpl;
    [ | discriminate ].
  destruct (kind_eqb kind Region) eqn:Hkind; [ discriminate | ].
  cbn [value_args].
  destruct (he_params entry) as [ | w wanted];
    [ rewrite andb_false_r; discriminate | ].
  rewrite Husable. cbn [orb andb]. rewrite andb_false_r. discriminate.
Qed.

Section Longer.

Variable cert : certificate.

Variable codes : list (code_id * code).

Variable rs : region_state.

Lemma ret_dead_all :
  (∀ (u : user) (ce : cont_entry) (c : cont_value),
     cont_ok cert codes rs u ce c -> True)
  ∧ (∀ (a : caller) (l : lifetime) (c : cont_value),
       ret_ok cert codes rs a l c ->
       rank (cs_open (caller_check a)) l = 0 ->
       ∀ l' : lifetime, ret_ok cert codes rs a l' c)
  ∧ (∀ (a : caller) (c : cont_value), raise_ok cert codes rs a c -> True).
Proof.
  apply (conts_ok_mutind cert codes rs
           (fun _ _ _ _ => True)
           (fun a l c _ =>
              rank (cs_open (caller_check a)) l = 0 ->
              ∀ l' : lifetime, ret_ok cert codes rs a l' c)
           (fun _ _ _ => True));
    try (intros; exact I).
  - intros a l k c Hjump Hsome _ Houter Hdead l'.
    exact
      (Ret_to cert codes rs a l' k c
         (jump_result_dead (caller_check a) k l l' Hjump Hdead) Hsome Houter).
  - intros a l l2 ls remaining remaining_arity return_arity return_to exn_to
      Hholds Hunknown Hret IHret Hraise _ Hdead l'.
    destruct (unknown_call_dead cert codes (caller_check a) l l' ls
                (length remaining_arity) l2 Hunknown Hdead)
      as (l3 & Hcall & [-> | Hdead2]).
    + exact
        (Ret_over cert codes rs a l' l2 ls remaining remaining_arity
           return_arity return_to exn_to Hholds Hcall Hret Hraise).
    + exact
        (Ret_over cert codes rs a l' l3 ls remaining remaining_arity
           return_arity return_to exn_to Hholds Hcall (IHret Hdead2 l3)
           Hraise).
  - intros a l l2 c Hlasts Hret IH Hdead l'. apply IH.
    pose proof (lasts_rank _ _ _ Hlasts). lia.
Qed.

Lemma ret_dead (a : caller) (l : lifetime) (c : cont_value) :
  ret_ok cert codes rs a l c ->
  rank (cs_open (caller_check a)) l = 0 ->
  ∀ l' : lifetime, ret_ok cert codes rs a l' c.
Proof. exact (proj1 (proj2 ret_dead_all) a l c). Qed.

(* What takes a value of a lifetime takes one of a lifetime that is no
   shorter. *)
Lemma ret_longer (a : caller) (l l' : lifetime) (c : cont_value) :
  ret_ok cert codes rs a l' c ->
  rank (cs_open (caller_check a)) l' <= rank (cs_open (caller_check a)) l ->
  ret_ok cert codes rs a l c.
Proof.
  intros Hret Hle.
  destruct (Nat.eq_dec (rank (cs_open (caller_check a)) l') 0)
    as [Hdead | Hlive].
  - exact (ret_dead a l' c Hret Hdead l).
  - apply (Ret_longer cert codes rs a l l' c); [ | exact Hret ].
    apply rank_lasts; lia.
Qed.

End Longer.
