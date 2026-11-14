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
   an argument of a jump that lives as long as the handler takes its
   parameter to (FlambdaRegionsCheckSpec.outlives) is as the lifetime of
   the parameter says, with the regions that the handler is entered with.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs FlambdaRegionsCheckedMarksProofs.
Import ListNotations.

Lemma marked_fun (vars : env) (r : variable) (m m' : mark) :
  marked vars r m -> marked vars r m' -> m = m'.
Proof. unfold marked. intros H H'. rewrite H in H'. congruence. Qed.

Lemma Forall2_in_left {A B : Type} (R : A -> B -> Prop) (l : list A)
    (l' : list B) (a : A) :
  Forall2 R l l' -> In a l -> ∃ b : B, In b l' ∧ R a b.
Proof.
  induction 1 as [ | a0 b0 l l' Hab _ IH]; intros Hin; [ contradiction | ].
  destruct Hin as [<- | Hin].
  - exists b0. split; [ left; reflexivity | exact Hab ].
  - destruct (IH Hin) as (b & Hb & Hr).
    exists b. split; [ right; exact Hb | exact Hr ].
Qed.

(* What is selected in order is among the open regions. *)
Lemma in_order_in (opens : list variable) :
  ∀ (named : list variable) (n : variable),
    in_order named opens = true -> In n named -> In n opens.
Proof.
  induction opens as [ | y opens IH]; intros named n; cbn [in_order].
  - destruct named; [ contradiction | discriminate ].
  - destruct named as [ | x named]; [ contradiction | ].
    destruct (Pos.eqb x y) eqn:Heq.
    + apply Pos.eqb_eq in Heq. subst x. intros Horder [<- | Hin];
        [ left; reflexivity | right; exact (IH named n Horder Hin) ].
    + intros Horder Hin. right. exact (IH (x :: named) n Horder Hin).
Qed.

Section Marks.

Variable vars : env.

(* The first of a list of open regions that passes a test: its mark is
   the greatest of those that pass. *)
Lemma find_first (f : variable -> bool) :
  ∀ (l : list variable) (ms : list mark) (found : option variable),
    Forall2 (marked vars) l ms ->
    marks_decrease ms = true ->
    find f l = found ->
    match found with
    | None => ∀ n : variable, In n l -> f n = false
    | Some n =>
      f n = true
      ∧ ∃ m_n : mark,
          marked vars n m_n
          ∧ In m_n ms
          ∧ ∀ (n' : variable) (m' : mark),
              In n' l -> f n' = true -> marked vars n' m' ->
              (m' <= m_n)%positive
    end.
Proof.
  induction l as [ | x l IH]; intros ms found Hmarked Hdecrease <-.
  - intros n [].
  - inversion Hmarked as [ | x0 m_x l0 ms0 Hx Hrest]; subst.
    apply marks_decrease_cons in Hdecrease as [Hlt Htail].
    cbn [find]. destruct (f x) eqn:Hfx.
    + split; [ exact Hfx | ]. exists m_x. split; [ exact Hx | ].
      split; [ left; reflexivity | ].
      intros n' m' [<- | Hin] _ Hn'.
      * rewrite (marked_fun vars x m' m_x Hn' Hx). lia.
      * destruct (Forall2_in_left _ _ _ n' Hrest Hin) as (m'' & Hin'' & Hn'').
        rewrite (marked_fun vars n' m' m'' Hn' Hn'').
        pose proof (Hlt m'' Hin''). lia.
    + specialize (IH ms0 (find f l) Hrest Htail eq_refl).
      destruct (find f l) as [n | ].
      * destruct IH as (Hfn & m_n & Hn & Hin & Hgreatest).
        split; [ exact Hfn | ]. exists m_n. split; [ exact Hn | ].
        split; [ right; exact Hin | ].
        intros n' m' [<- | Hin'] Hfn' Hn'; [ congruence | ].
        exact (Hgreatest n' m' Hin' Hfn' Hn').
      * intros n [<- | Hin]; [ exact Hfx | exact (IH n Hin) ].
Qed.

(* The nearest of the regions [named] at or below [r]. *)
Lemma nearest_named_marks (named : list variable) :
  ∀ (opens : list variable) (ms : list mark) (r : variable)
    (nearest : option variable),
    Forall2 (marked vars) opens ms ->
    marks_decrease ms = true ->
    nearest_named named opens r = Some nearest ->
    ∃ m_r : mark,
      marked vars r m_r
      ∧ In m_r ms
      ∧ match nearest with
        | None =>
          ∀ (n : variable) (m_n : mark),
            among n named = true -> In n opens -> marked vars n m_n ->
            (m_r < m_n)%positive
        | Some n =>
          among n named = true
          ∧ ∃ m_n : mark,
              marked vars n m_n
              ∧ In m_n ms
              ∧ (m_n <= m_r)%positive
              ∧ ∀ (n' : variable) (m' : mark),
                  among n' named = true -> In n' opens -> marked vars n' m' ->
                  (m' <= m_n)%positive ∨ (m_r < m')%positive
        end.
Proof.
  unfold nearest_named.
  induction opens as [ | x opens IH]; intros ms r nearest Hmarked Hdecrease;
    [ discriminate | ].
  inversion Hmarked as [ | x0 m_x opens0 ms0 Hx Hrest]; subst.
  pose proof Hdecrease as Hall.
  apply marks_decrease_cons in Hdecrease as [Hlt Htail].
  replace (among r (x :: opens)) with (Pos.eqb r x || among r opens)
    by reflexivity.
  cbn [regions_below]. rewrite (Pos.eqb_sym x r).
  destruct (Pos.eqb r x) eqn:Heq; cbn [orb].
  - apply Pos.eqb_eq in Heq. subst x. intros Hnearest.
    assert (Hfound : find (fun x => among x named) (r :: opens) = nearest)
      by (injection Hnearest as Hfound; exact Hfound).
    pose proof (find_first _ _ _ _ Hmarked Hall Hfound) as Hfind.
    cbv beta in Hfind.
    exists m_x. split; [ exact Hx | ]. split; [ left; reflexivity | ].
    destruct nearest as [n | ].
    + destruct Hfind as (Hn & m_n & Hmn & Hin & Hgreatest).
      split; [ exact Hn | ]. exists m_n. split; [ exact Hmn | ].
      split; [ exact Hin | ]. split.
      * destruct Hin as [<- | Hin]; [ lia | ]. pose proof (Hlt m_n Hin). lia.
      * intros n' m' Hnamed Hin' Hn'. left.
        exact (Hgreatest n' m' Hin' Hnamed Hn').
    + intros n m_n Hnamed Hin _. rewrite (Hfind n Hin) in Hnamed.
      discriminate.
  - intros Hnearest.
    destruct (IH ms0 r nearest Hrest Htail Hnearest)
      as (m_r & Hmr & Hin & Hspec).
    pose proof (Hlt m_r Hin) as Hbelow.
    exists m_r. split; [ exact Hmr | ]. split; [ right; exact Hin | ].
    destruct nearest as [n | ].
    + destruct Hspec as (Hn & m_n & Hmn & Hin_n & Hle & Hothers).
      split; [ exact Hn | ]. exists m_n. split; [ exact Hmn | ].
      split; [ right; exact Hin_n | ]. split; [ exact Hle | ].
      intros n' m' Hnamed [<- | Hin'] Hn'.
      * right. rewrite (marked_fun vars x m' m_x Hn' Hx). exact Hbelow.
      * exact (Hothers n' m' Hnamed Hin' Hn').
    + intros n m_n Hnamed [<- | Hin'] Hn.
      * rewrite (marked_fun vars x m_n m_x Hn Hx). exact Hbelow.
      * exact (Hspec n m_n Hnamed Hin' Hn).
Qed.

(* A region among those down to [n] is at or above [n]. *)
Lemma down_to_marks (n : variable) :
  ∀ (opens : list variable) (ms : list mark) (r : variable) (m_r m_n : mark),
    Forall2 (marked vars) opens ms ->
    marks_decrease ms = true ->
    In n opens ->
    among r (down_to n opens) = true ->
    marked vars r m_r ->
    marked vars n m_n ->
    (m_n <= m_r)%positive.
Proof.
  induction opens as [ | x opens IH]; intros ms r m_r m_n Hmarked Hdecrease;
    [ contradiction | ].
  inversion Hmarked as [ | x0 m_x opens0 ms0 Hx Hrest]; subst.
  apply marks_decrease_cons in Hdecrease as [Hlt Htail].
  cbn [down_to]. destruct (Pos.eqb x n) eqn:Heq.
  - apply Pos.eqb_eq in Heq. subst x. intros _ Hr Hmr Hmn.
    apply among_In in Hr. destruct Hr as [<- | []].
    rewrite (marked_fun vars n m_r m_n Hmr Hmn). lia.
  - intros [-> | Hin]; [ rewrite Pos.eqb_refl in Heq; discriminate | ].
    intros Hr Hmr Hmn. apply among_In in Hr. destruct Hr as [<- | Hr].
    + rewrite (marked_fun vars x m_r m_x Hmr Hx).
      destruct (Forall2_in_left _ _ _ n Hrest Hin) as (m' & Hin' & Hn').
      rewrite (marked_fun vars n m_n m' Hmn Hn').
      pose proof (Hlt m' Hin'). lia.
    + apply among_In in Hr. exact (IH ms0 r m_r m_n Hrest Htail Hin Hr Hmr Hmn).
Qed.

End Marks.

Lemma nearest_named_open (named opens : list variable) (r : variable)
    (nearest : option variable) :
  nearest_named named opens r = Some nearest -> In r opens.
Proof.
  unfold nearest_named. destruct (among r opens) eqn:Hr; [ | discriminate ].
  intros _. apply among_In. exact Hr.
Qed.

(* ======================================================================== *)
(* An argument that outlives what the handler takes its parameter to        *)
(* ======================================================================== *)

Section Outlives.

Variable rs : region_state.

Variable vars : env.

Variables opens named : list variable.

Variables ms ms' : list mark.

Hypothesis Hopens : Forall2 (marked vars) opens ms.

Hypothesis Hplaced : marks_placed (rs_frames rs) ms = true.

Hypothesis Hnamed : Forall2 (marked vars) named ms'.

Hypothesis Hselection : selection ms' ms.

Hypothesis Horder : in_order named opens = true.

Lemma jump_marks_decrease : marks_decrease ms = true.
Proof.
  unfold marks_placed in Hplaced. apply andb_true_iff in Hplaced as [_ H].
  exact H.
Qed.

Lemma named_marks_decrease : marks_decrease ms' = true.
Proof. exact (selection_decrease _ _ Hselection jump_marks_decrease). Qed.

(* A mark of the regions that the handler is entered with. *)
Lemma named_mark (b : mark) :
  In b ms' ->
  In b ms
  ∧ ∃ n : variable,
      among n named = true ∧ In n opens ∧ marked vars n b.
Proof.
  intros Hin. split; [ exact (selection_in _ _ b Hselection Hin) | ].
  destruct (Forall2_in_right _ _ _ b Hnamed Hin) as (n & Hn & Hmarked).
  exists n. split; [ apply among_In; exact Hn | ].
  split; [ exact (in_order_in opens named n Horder Hn) | exact Hmarked ].
Qed.

(* What lives as long as an open region is within every mark above
   that of the region. *)
Lemma of_region_within (r : variable) (m_r : mark) (v : value)
    (stop' : option symbol) :
  marked vars r m_r ->
  In r opens ->
  lifetime_holds rs opens ms None (Of_region r) v ->
  (∀ b' : symbol, stop' = Some b' -> In b' ms ∧ (m_r < b')%positive) ->
  value_within rs stop' v = true.
Proof.
  intros Hr Hin Hholds Habove.
  pose proof (proj2 (open_marks_marked vars opens ms) Hopens) as Hmarks.
  destruct (region_place_open vars opens ms None r Hmarks
              (proj2 (among_In r opens) Hin))
    as (m & stop & Hplace & Hm).
  cbn [lifetime_holds] in Hholds. rewrite Hplace in Hholds.
  destruct (region_place_least vars opens ms r m stop Hopens
              jump_marks_decrease Hplace)
    as (Hrm & _ & Hstop).
  rewrite (marked_fun vars r m m_r Hrm Hr) in *.
  apply (within_later rs stop stop' v);
    [ | exact (Hholds (marks_placed_in _ _ m_r Hplaced Hm)) ].
  intros b' Hb'. destruct (Habove b' Hb') as [Hin' Hlt].
  destruct stop as [b | ].
  - exists b. split; [ reflexivity | ].
    destruct Hstop as (_ & _ & Hleast). exact (Hleast b' Hin' Hlt).
  - pose proof (Hstop b' Hin'). lia.
Qed.

(* What lives as long as the code is within every mark. *)
Lemma of_caller_within (v : value) (stop' : option symbol) :
  lifetime_holds rs opens ms None Of_caller v ->
  (∀ b' : symbol, stop' = Some b' -> In b' ms) ->
  value_within rs stop' v = true.
Proof.
  cbn [lifetime_holds]. intros Hholds Hin.
  apply (within_later rs (caller_bound ms None) stop' v); [ | exact Hholds ].
  intros b' Hb'. specialize (Hin b' Hb'). unfold caller_bound.
  destruct (lowest ms) as [b | ] eqn:Hlowest.
  - exists b. split; [ reflexivity | ].
    exact (proj2 (lowest_least ms b Hlowest jump_marks_decrease) b' Hin).
  - rewrite (lowest_none ms Hlowest) in Hin. contradiction.
Qed.

(* The bound of what lives as long as the code of the handler. *)
Lemma named_caller_bound (b' : symbol) :
  caller_bound ms' None = Some b' -> In b' ms'.
Proof.
  unfold caller_bound. destruct (lowest ms') as [b | ] eqn:Hlowest;
    [ | discriminate ].
  intros [= <-].
  exact (proj1 (lowest_least ms' b Hlowest named_marks_decrease)).
Qed.

Lemma outlives_holds (l w : lifetime) (v : value) :
  outlives named opens l w = true ->
  lifetime_holds rs opens ms None l v ->
  lifetime_holds rs named ms' None w v.
Proof.
  destruct w as [ | | r' | ]; [ | | | intros _ _; exact I ].
  - (* For ever. *)
    destruct l; cbn [outlives]; try discriminate. intros _ H. exact H.
  - (* As long as the code. *)
    destruct l as [ | | r | ]; cbn [outlives]; try discriminate.
    + intros _ H. exact (heap_within rs _ v H).
    + intros _ H. apply (of_caller_within v _ H).
      intros b' Hb'. exact (proj1 (named_mark b' (named_caller_bound b' Hb'))).
    + destruct (nearest_named named opens r) as [[n | ] | ] eqn:Hnearest;
        try discriminate.
      intros _ H.
      destruct (nearest_named_marks vars named opens ms r None Hopens
                  jump_marks_decrease Hnearest)
        as (m_r & Hr & _ & Habove).
      apply (of_region_within r m_r v _ Hr
               (nearest_named_open _ _ _ _ Hnearest) H).
      intros b' Hb'.
      destruct (named_mark b' (named_caller_bound b' Hb'))
        as (Hin & n & Hn & Hopen & Hmarked).
      split; [ exact Hin | exact (Habove n b' Hn Hopen Hmarked) ].
  - (* As long as a region that the handler is entered with. *)
    cbn [lifetime_holds].
    destruct (region_place named ms' None r') as [[m' stop'] | ] eqn:Hplace;
      [ | intros _ _; exact I ].
    destruct (region_place_least vars named ms' r' m' stop' Hnamed
                named_marks_decrease Hplace)
      as (Hr' & Hm' & Hstop').
    assert (Hbound :
      ∀ b' : symbol,
        stop' = Some b' -> In b' ms' ∧ (m' < b')%positive).
    { intros b' ->. destruct Hstop' as (H & Hlt & _). exact (conj H Hlt). }
    destruct l as [ | | r | ]; cbn [outlives]; try discriminate.
    + intros _ H _. exact (heap_within rs _ v H).
    + intros _ H _. apply (of_caller_within v _ H).
      intros b' Hb'. exact (proj1 (named_mark b' (proj1 (Hbound b' Hb')))).
    + intros Houtlives H _.
      apply andb_true_iff in Houtlives as [Hnamed' Houtlives].
      destruct (nearest_named named opens r) as [[n | ] | ] eqn:Hnearest;
        try discriminate.
      * destruct (nearest_named_marks vars named opens ms r (Some n) Hopens
                    jump_marks_decrease Hnearest)
          as (m_r & Hr & _ & Hn & m_n & Hmn & _ & Hle & Hothers).
        apply (of_region_within r m_r v _ Hr
                 (nearest_named_open _ _ _ _ Hnearest) H).
        intros b' Hb'. destruct (Hbound b' Hb') as [Hin' Hlt].
        destruct (named_mark b' Hin') as (Hin & n' & Hn' & Hopen & Hmarked).
        split; [ exact Hin | ].
        pose proof (down_to_marks vars n opens ms r' m' m_n Hopens
                      jump_marks_decrease
                      (in_order_in opens named n Horder
                         (proj1 (among_In n named) Hn))
                      Houtlives Hr' Hmn)
          as Hdown.
        destruct (Hothers n' b' Hn' Hopen Hmarked) as [Hbelow | Habove];
          [ lia | exact Habove ].
      * destruct (nearest_named_marks vars named opens ms r None Hopens
                    jump_marks_decrease Hnearest)
          as (m_r & Hr & _ & Habove).
        apply (of_region_within r m_r v _ Hr
                 (nearest_named_open _ _ _ _ Hnearest) H).
        intros b' Hb'. destruct (Hbound b' Hb') as [Hin' _].
        destruct (named_mark b' Hin') as (Hin & n' & Hn' & Hopen & Hmarked).
        split; [ exact Hin | exact (Habove n' b' Hn' Hopen Hmarked) ].
Qed.

End Outlives.
