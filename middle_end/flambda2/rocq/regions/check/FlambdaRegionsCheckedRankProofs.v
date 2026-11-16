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
   the lifetimes in order. With the open regions given, a lifetime has a
   rank: 0 for [Gone] and for a region that is not open, then the open
   regions from the newest, then [Of_caller], then [Forever].
   FlambdaRegionsCheckSpec.shorter is the minimum. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia PeanoNat.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs
  FlambdaRegionsCheckedOutlivesProofs.
Import ListNotations.

(* The place of an open region, from the newest. *)
Fixpoint position (r : variable) (open : list variable) : option nat :=
  match open with
  | [] => None
  | x :: open =>
    if Pos.eqb x r then Some 0 else option_map S (position r open)
  end.

Definition rank (open : list variable) (l : lifetime) : nat :=
  match l with
  | Gone => 0
  | Of_region r =>
    match position r open with
    | Some i => S i
    | None => 0
    end
  | Of_caller => S (length open)
  | Forever => S (S (length open))
  end.

Lemma position_among (r : variable) (open : list variable) :
  among r open = true <-> ∃ i : nat, position r open = Some i.
Proof.
  induction open as [ | x open IH]; cbn [position].
  - split; [ discriminate | intros [i H]; discriminate ].
  - rewrite among_cons, (Pos.eqb_sym r x).
    destruct (Pos.eqb x r); cbn [orb].
    + split; [ intros _; exists 0; reflexivity | reflexivity ].
    + rewrite IH. split.
      * intros [i ->]. exists (S i). reflexivity.
      * intros [i H]. destruct (position r open) as [j | ];
          [ exists j; reflexivity | discriminate ].
Qed.

Lemma position_none (r : variable) (open : list variable) :
  position r open = None <-> among r open = false.
Proof.
  destruct (among r open) eqn:Hr.
  - apply position_among in Hr as [i ->]. split; discriminate.
  - destruct (position r open) as [i | ] eqn:Hp; [ | split; reflexivity ].
    rewrite (proj2 (position_among r open) (ex_intro _ i Hp)) in Hr.
    discriminate.
Qed.

Lemma position_below (r : variable) (open : list variable) :
  ∀ i : nat, position r open = Some i -> i < length open.
Proof.
  induction open as [ | x open IH]; intros i; cbn [position length];
    [ discriminate | ].
  destruct (Pos.eqb x r); [ intros [= <-]; lia | ].
  destruct (position r open) as [j | ]; [ | discriminate ].
  intros [= <-]. pose proof (IH j eq_refl). lia.
Qed.

Lemma position_inj (open : list variable) :
  ∀ (r r' : variable) (i : nat),
    position r open = Some i -> position r' open = Some i -> r = r'.
Proof.
  induction open as [ | x open IH]; intros r r' i; cbn [position];
    [ discriminate | ].
  destruct (Pos.eqb x r) eqn:Hr, (Pos.eqb x r') eqn:Hr'.
  - apply Pos.eqb_eq in Hr, Hr'. congruence.
  - intros [= <-]. destruct (position r' open); discriminate.
  - destruct (position r open); [ | discriminate ]. intros [= <-]. discriminate.
  - destruct (position r open) as [j | ] eqn:Hj; [ | discriminate ].
    destruct (position r' open) as [j' | ] eqn:Hj'; [ | discriminate ].
    cbn [option_map]. intros [= <-] [= Heq]. subst j'.
    exact (IH r r' j Hj Hj').
Qed.

(* A region is among those down to another when it is at or above
   it. *)
Lemma among_down_to (open : list variable) :
  ∀ (r r' : variable) (i i' : nat),
    position r open = Some i ->
    position r' open = Some i' ->
    among r (down_to r' open) = true <-> i <= i'.
Proof.
  induction open as [ | x open IH]; intros r r' i i';
    cbn [position down_to]; [ discriminate | ].
  destruct (Pos.eqb x r') eqn:Hr'.
  - intros Hi [= <-]. rewrite among_cons, (Pos.eqb_sym r x).
    destruct (Pos.eqb x r); cbn [orb among existsb].
    + injection Hi as <-. split; [ lia | reflexivity ].
    + destruct (position r open); [ | discriminate ]. injection Hi as <-.
      split; [ discriminate | lia ].
  - intros Hi Hi'. rewrite among_cons, (Pos.eqb_sym r x).
    destruct (position r' open) as [j' | ] eqn:Hj'; [ | discriminate ].
    injection Hi' as <-.
    destruct (Pos.eqb x r); cbn [orb].
    + injection Hi as <-. split; [ lia | reflexivity ].
    + destruct (position r open) as [j | ] eqn:Hj; [ | discriminate ].
      injection Hi as <-. rewrite (IH r r' j j' Hj Hj'). lia.
Qed.

Lemma rank_region_le (open : list variable) (r : variable) :
  rank open (Of_region r) <= length open.
Proof.
  cbn [rank]. destruct (position r open) as [i | ] eqn:Hi; [ | lia ].
  pose proof (position_below r open i Hi). lia.
Qed.

Lemma rank_le (open : list variable) (l : lifetime) :
  rank open l <= S (S (length open)).
Proof.
  destruct l as [ | | r | ]; try (cbn [rank]; lia).
  pose proof (rank_region_le open r). lia.
Qed.

(* [shorter] is the minimum, of any two lifetimes. *)
Lemma rank_shorter (open : list variable) (a b : lifetime) :
  rank open (shorter open a b) = Nat.min (rank open a) (rank open b).
Proof.
  destruct a as [ | | r | ], b as [ | | r' | ]; cbn [shorter];
    try (cbn [rank]; lia);
    try (pose proof (rank_region_le open r); cbn [rank] in *; lia);
    try (pose proof (rank_region_le open r'); cbn [rank] in *; lia).
  destruct (among r open) eqn:Hr; cbn [andb].
  - destruct (among r' open) eqn:Hr'.
    + apply position_among in Hr as [i Hi], Hr' as [i' Hi'].
      pose proof (among_down_to open r r' i i' Hi Hi') as Hdown.
      destruct (among r (down_to r' open)); cbn [rank]; rewrite Hi, Hi'.
      * pose proof (proj1 Hdown eq_refl). lia.
      * assert (¬ i <= i') by (intros H; apply Hdown in H; discriminate).
        lia.
    + apply position_none in Hr'. cbn [rank]. rewrite Hr'. lia.
  - apply position_none in Hr. cbn [rank]. rewrite Hr. lia.
Qed.

Lemma rank_shortest_le (cs : check_state) (ls : list lifetime) (l : lifetime) :
  In l ls -> rank (cs_open cs) (shortest cs ls) <= rank (cs_open cs) l.
Proof.
  unfold shortest. induction ls as [ | l0 ls IH]; [ contradiction | ].
  cbn [fold_right]. rewrite rank_shorter. intros [<- | Hin]; [ lia | ].
  specialize (IH Hin). lia.
Qed.

Lemma rank_shortest_ge (cs : check_state) (ls : list lifetime) (n : nat) :
  n <= S (S (length (cs_open cs))) ->
  (∀ l : lifetime, In l ls -> n <= rank (cs_open cs) l) ->
  n <= rank (cs_open cs) (shortest cs ls).
Proof.
  intros Htop. unfold shortest. induction ls as [ | l0 ls IH]; intros Hall.
  - exact Htop.
  - cbn [fold_right]. rewrite rank_shorter.
    pose proof (Hall l0 (or_introl eq_refl)).
    pose proof (IH (fun l Hin => Hall l (or_intror Hin))). lia.
Qed.

Lemma rank_newest (cs : check_state) :
  rank (cs_open cs) (newest_lifetime cs) = 1.
Proof.
  unfold newest_lifetime. destruct (cs_open cs) as [ | r open];
    [ reflexivity | ].
  cbn [rank position]. rewrite Pos.eqb_refl. reflexivity.
Qed.

Lemma is_forever_rank (open : list variable) (l : lifetime) :
  is_forever l = true <-> rank open l = S (S (length open)).
Proof.
  destruct l as [ | | r | ]; cbn [is_forever].
  - split; reflexivity.
  - cbn [rank]. split; [ discriminate | lia ].
  - pose proof (rank_region_le open r). split; [ discriminate | lia ].
  - cbn [rank]. split; [ discriminate | lia ].
Qed.

Lemma usable_rank (cs : check_state) (l : lifetime) :
  lifetime_usable cs l = true <-> 0 < rank (cs_open cs) l.
Proof.
  destruct l as [ | | r | ]; cbn [lifetime_usable rank].
  - split; [ lia | reflexivity ].
  - split; [ lia | reflexivity ].
  - destruct (among r (cs_open cs)) eqn:Hr.
    + apply position_among in Hr as [i ->]. split; [ lia | reflexivity ].
    + apply position_none in Hr. rewrite Hr. split; [ discriminate | lia ].
  - split; [ discriminate | lia ].
Qed.

Lemma lifetime_eqb_eq (l l' : lifetime) : lifetime_eqb l l' = true <-> l = l'.
Proof.
  destruct l as [ | | r | ], l' as [ | | r' | ]; cbn [lifetime_eqb];
    try (split; [ reflexivity | reflexivity ]);
    try (split; discriminate).
  rewrite Pos.eqb_eq. split; [ intros ->; reflexivity | intros [= ->] ];
    reflexivity.
Qed.

(* A lifetime that is not dead is the only one of its rank. *)
Lemma rank_inj (open : list variable) (l l' : lifetime) :
  0 < rank open l -> rank open l = rank open l' -> l = l'.
Proof.
  pose proof (rank_region_le open) as Hregion.
  destruct l as [ | | r | ], l' as [ | | r' | ]; cbn [rank];
    try reflexivity; try lia;
    try (pose proof (Hregion r); cbn [rank] in *; lia);
    try (pose proof (Hregion r'); cbn [rank] in *; lia).
  destruct (position r open) as [i | ] eqn:Hi; [ | lia ].
  destruct (position r' open) as [i' | ] eqn:Hi'; [ | lia ].
  intros _ [= <-]. rewrite (position_inj open r r' i Hi Hi'). reflexivity.
Qed.

Lemma lasts_rank (open : list variable) (l l' : lifetime) :
  lasts open l l' = true -> rank open l' <= rank open l.
Proof.
  unfold lasts. intros H. apply lifetime_eqb_eq in H.
  pose proof (rank_shorter open l l') as Hmin. rewrite H in Hmin. lia.
Qed.

Lemma rank_lasts (open : list variable) (l l' : lifetime) :
  0 < rank open l' -> rank open l' <= rank open l -> lasts open l l' = true.
Proof.
  intros Hlive Hle. unfold lasts. apply lifetime_eqb_eq. symmetry.
  apply (rank_inj open); [ exact Hlive | ]. rewrite rank_shorter. lia.
Qed.

(* A dead lifetime says nothing. *)
Lemma region_place_none (opens : list variable) :
  ∀ (ms : list mark) (above : option symbol) (r : variable),
    position r opens = None -> region_place opens ms above r = None.
Proof.
  induction opens as [ | x opens IH]; intros ms above r; [ reflexivity | ].
  cbn [position region_place]. destruct ms as [ | m ms]; [ reflexivity | ].
  destruct (Pos.eqb x r); [ discriminate | ].
  destruct (position r opens) eqn:Hr; [ discriminate | ]. intros _.
  exact (IH ms (Some m) r Hr).
Qed.

Lemma dead_holds (rs : region_state) (opens : list variable) (ms : list mark)
    (above : option symbol) (l : lifetime) (v : value) :
  rank opens l = 0 -> lifetime_holds rs opens ms above l v.
Proof.
  destruct l as [ | | r | ]; cbn [rank lifetime_holds];
    [ discriminate | discriminate | | intros _; exact I ].
  destruct (position r opens) as [i | ] eqn:Hr; [ discriminate | ]. intros _.
  rewrite (region_place_none opens ms above r Hr). exact I.
Qed.

Lemma selection_refl {A : Type} (l : list A) : selection l l.
Proof. induction l; constructor; assumption. Qed.

Lemma in_order_refl (l : list variable) : in_order l l = true.
Proof.
  induction l as [ | x l IH]; [ reflexivity | ]. cbn [in_order].
  rewrite Pos.eqb_refl. exact IH.
Qed.

Lemma nearest_named_self (opens : list variable) (r : variable) :
  among r opens = true -> nearest_named opens opens r = Some (Some r).
Proof.
  intros Hr. unfold nearest_named. rewrite Hr. cbn [find]. rewrite Hr.
  reflexivity.
Qed.

(* Of two lifetimes with the same open regions, the longer outlives
   the shorter. *)
Lemma rank_outlives (opens : list variable) (l l' : lifetime) :
  0 < rank opens l' -> rank opens l' <= rank opens l ->
  outlives opens opens l l' = true.
Proof.
  pose proof (rank_region_le opens) as Hregion.
  destruct l as [ | | r | ], l' as [ | | r' | ]; cbn [rank outlives];
    try reflexivity; try lia;
    try (pose proof (Hregion r); cbn [rank] in *; lia);
    try (pose proof (Hregion r'); cbn [rank] in *; lia).
  - destruct (position r' opens) as [i' | ] eqn:Hi'; [ | lia ]. intros _ _.
    apply position_among. exists i'. exact Hi'.
  - destruct (position r opens) as [i | ] eqn:Hi;
      destruct (position r' opens) as [i' | ] eqn:Hi'; try lia.
    intros _ Hle.
    assert (Hr : among r opens = true)
      by (apply position_among; exists i; exact Hi).
    assert (Hr' : among r' opens = true)
      by (apply position_among; exists i'; exact Hi').
    rewrite Hr', (nearest_named_self opens r Hr). cbn [andb].
    apply (among_down_to opens r' r i' i Hi' Hi). lia.
Qed.

(* A value as a lifetime says is as a shorter one says. *)
Lemma holds_rank (rs : region_state) (vars : env) (opens : list variable)
    (ms : list mark) (l l' : lifetime) (v : value) :
  Forall2 (marked vars) opens ms ->
  marks_placed (rs_frames rs) ms = true ->
  rank opens l' <= rank opens l ->
  lifetime_holds rs opens ms None l v ->
  lifetime_holds rs opens ms None l' v.
Proof.
  intros Hopens Hplaced Hle Hholds.
  destruct (Nat.eq_dec (rank opens l') 0) as [Hdead | Hlive];
    [ exact (dead_holds rs opens ms None l' v Hdead) | ].
  apply (outlives_holds rs vars opens opens ms ms Hopens Hplaced Hopens
           (selection_refl ms) (in_order_refl opens) l l' v);
    [ | exact Hholds ].
  apply rank_outlives; lia.
Qed.
