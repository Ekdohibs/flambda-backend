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
   the open regions of the check and the marks that their variables are
   bound to; the bound of a region as the least mark above its own; a
   selection of the open regions, in their order. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaHeapProofs
  FlambdaSemanticsSpec FlambdaRegionsSpec FlambdaRegionsProofs
  FlambdaRegionsReadsProofs FlambdaRegionsInvariantSpec
  FlambdaRegionsInvariantProofs FlambdaRegionsErasureSpec
  FlambdaRegionsCallsSpec FlambdaRegionsCheckSpec FlambdaRegionsCheckedSpec.
From OxCamlFlambda2 Require Import FlambdaRegionsCheckedBasicsProofs
  FlambdaRegionsCheckedStableProofs.
Import ListNotations.

(* A variable is bound to a mark. *)
Definition marked (vars : env) (r : variable) (m : mark) : Prop :=
  lookup r vars = Some (Value_symbol m).

Lemma open_marks_marked (vars : env) (opens : list variable) :
  ∀ ms : list mark,
    open_marks vars opens = Some ms <-> Forall2 (marked vars) opens ms.
Proof.
  induction opens as [ | r opens IH]; intros ms; cbn [open_marks].
  - split; [ intros [= <-]; constructor | intros H; inversion H; reflexivity ].
  - split.
    + destruct (lookup r vars) as [[ | | | m | | | | | ] | ] eqn:Hr;
        try discriminate.
      destruct (open_marks vars opens) as [ms' | ] eqn:Hms; [ | discriminate ].
      cbn. intros [= <-]. constructor; [ exact Hr | ].
      exact (proj1 (IH ms') eq_refl).
    + intros H. inversion H as [ | r0 m opens0 ms' Hr Hrest]; subst.
      unfold marked in Hr. rewrite Hr.
      rewrite (proj2 (IH ms') Hrest). reflexivity.
Qed.

(* The marks decrease: each is above those that follow it. *)
Lemma marks_decrease_cons (m : mark) (ms : list mark) :
  marks_decrease (m :: ms) = true
  <-> (∀ m' : mark, In m' ms -> (m' < m)%positive)
      ∧ marks_decrease ms = true.
Proof.
  split.
  - intros H.
    exact (conj (marks_decrease_lt m ms H) (marks_decrease_tail m ms H)).
  - intros [Hlt Hrest]. cbn [marks_decrease]. rewrite Hrest, andb_true_r.
    destruct ms as [ | m1 ms]; [ reflexivity | ].
    apply Pos.ltb_lt. exact (Hlt m1 (or_introl eq_refl)).
Qed.

(* A selection of a list, in its order. *)
Inductive selection {A : Type} : list A -> list A -> Prop :=
  | Selection_nil (l : list A) : selection [] l
  | Selection_take (x : A) (xs l : list A) :
    selection xs l -> selection (x :: xs) (x :: l)
  | Selection_skip (x : A) (xs l : list A) :
    selection xs l -> selection xs (x :: l).

Lemma selection_in {A : Type} (xs l : list A) (x : A) :
  selection xs l -> In x xs -> In x l.
Proof.
  induction 1 as [l | y xs l _ IH | y xs l _ IH]; intros Hin.
  - contradiction.
  - destruct Hin as [<- | Hin]; [ left; reflexivity | right; exact (IH Hin) ].
  - right. exact (IH Hin).
Qed.

Lemma selection_decrease (ms' ms : list mark) :
  selection ms' ms -> marks_decrease ms = true -> marks_decrease ms' = true.
Proof.
  induction 1 as [l | m ms' l Hsel IH | m ms' l Hsel IH]; intros Hdecrease.
  - reflexivity.
  - apply marks_decrease_cons in Hdecrease as [Hlt Hrest].
    apply marks_decrease_cons. split; [ | exact (IH Hrest) ].
    intros m' Hin. exact (Hlt m' (selection_in _ _ m' Hsel Hin)).
  - exact (IH (marks_decrease_tail _ _ Hdecrease)).
Qed.

(* The marks of a selection of the open regions, in their order, are a
   selection of the marks. *)
Lemma in_order_marks (vars : env) :
  ∀ (opens named : list variable) (ms : list mark),
    in_order named opens = true ->
    Forall2 (marked vars) opens ms ->
    ∃ ms' : list mark, Forall2 (marked vars) named ms' ∧ selection ms' ms.
Proof.
  induction opens as [ | y opens IH]; intros named ms; cbn [in_order].
  - destruct named; [ | discriminate ]. intros _ H. inversion H; subst.
    exists []. split; constructor.
  - intros Horder H. inversion H as [ | y0 m opens0 ms0 Hy Hrest]; subst.
    destruct named as [ | x named].
    + exists []. split; constructor.
    + destruct (Pos.eqb x y) eqn:Heq.
      * apply Pos.eqb_eq in Heq. subst x.
        destruct (IH named ms0 Horder Hrest) as (ms' & Hmarked & Hsel).
        exists (m :: ms'). split; constructor; assumption.
      * destruct (IH (x :: named) ms0 Horder Hrest) as (ms' & Hmarked & Hsel).
        exists ms'. split; [ exact Hmarked | ]. constructor. exact Hsel.
Qed.

Lemma selection_placed (frames : list frame) (ms' ms : list mark) :
  selection ms' ms ->
  marks_placed frames ms = true ->
  marks_placed frames ms' = true.
Proof.
  intros Hsel. unfold marks_placed. intros H.
  apply andb_true_iff in H as [Hon Hdecrease].
  apply andb_true_iff.
  split; [ | exact (selection_decrease _ _ Hsel Hdecrease) ].
  apply forallb_forall. intros m Hin.
  exact (forallb_in _ _ m Hon (selection_in _ _ m Hsel Hin)).
Qed.

(* The place of an open region, with no bound for the newest: its
   mark is that of its variable; its bound, if it has one, is the
   least mark above its own; if it has none, no mark is above its
   own. *)
Lemma region_place_least (vars : env) (opens : list variable) :
  ∀ (ms : list mark) (r : variable) (m : mark) (stop : option symbol),
    Forall2 (marked vars) opens ms ->
    marks_decrease ms = true ->
    region_place opens ms None r = Some (m, stop) ->
    marked vars r m
    ∧ In m ms
    ∧ match stop with
      | Some b =>
        In b ms
        ∧ (m < b)%positive
        ∧ ∀ m0 : mark, In m0 ms -> (m < m0)%positive -> (b <= m0)%positive
      | None => ∀ m0 : mark, In m0 ms -> (m0 <= m)%positive
      end.
Proof.
  assert (Hgen :
    ∀ (opens : list variable) (ms : list mark) (above : option symbol)
      (r : variable) (m : mark) (stop : option symbol),
      Forall2 (marked vars) opens ms ->
      marks_decrease ms = true ->
      (∀ b : symbol,
         above = Some b -> ∀ m' : mark, In m' ms -> (m' < b)%positive) ->
      region_place opens ms above r = Some (m, stop) ->
      marked vars r m
      ∧ In m ms
      ∧ match stop with
        | Some b =>
          (above = Some b ∨ In b ms)
          ∧ (m < b)%positive
          ∧ ∀ m0 : mark, In m0 ms -> (m < m0)%positive -> (b <= m0)%positive
        | None => above = None ∧ ∀ m0 : mark, In m0 ms -> (m0 <= m)%positive
        end).
  { clear opens. induction opens as [ | x opens IH];
      intros ms above r m stop Hmarked Hdecrease Habove; [ discriminate | ].
    inversion Hmarked as [ | x0 m1 opens0 ms0 Hx Hrest]; subst.
    cbn [region_place]. apply marks_decrease_cons in Hdecrease as [Hlt Htail].
    destruct (Pos.eqb x r) eqn:Heq.
    - apply Pos.eqb_eq in Heq. subst x. intros [= <- <-].
      split; [ exact Hx | ]. split; [ left; reflexivity | ].
      destruct above as [b | ].
      + split; [ left; reflexivity | ].
        split; [ exact (Habove b eq_refl m1 (or_introl eq_refl)) | ].
        intros m0 [<- | Hin] Hm0; [ lia | ].
        pose proof (Hlt m0 Hin). lia.
      + split; [ reflexivity | ].
        intros m0 [<- | Hin]; [ lia | ]. pose proof (Hlt m0 Hin). lia.
    - intros Hplace.
      destruct (IH ms0 (Some m1) r m stop Hrest Htail) as (Hr & Hin & Hstop);
        [ intros b [= <-]; exact Hlt | exact Hplace | ].
      split; [ exact Hr | ]. split; [ right; exact Hin | ].
      destruct stop as [b | ].
      + destruct Hstop as (Hb & Hmb & Hleast).
        split.
        * right. destruct Hb as [[= <-] | Hb];
            [ left; reflexivity | right; exact Hb ].
        * split; [ exact Hmb | ].
          intros m0 [<- | Hin0] Hm0; [ | exact (Hleast m0 Hin0 Hm0) ].
          destruct Hb as [[= <-] | Hb]; [ lia | ].
          pose proof (Hlt b Hb). lia.
      + destruct Hstop as [Hnone _]. discriminate. }
  intros ms r m stop Hmarked Hdecrease Hplace.
  destruct (Hgen opens ms None r m stop Hmarked Hdecrease)
    as (Hr & Hin & Hstop);
    [ discriminate | exact Hplace | ].
  split; [ exact Hr | ]. split; [ exact Hin | ].
  destruct stop as [b | ].
  - destruct Hstop as ([Hb | Hb] & Hrest); [ discriminate | ].
    exact (conj Hb Hrest).
  - exact (proj2 Hstop).
Qed.

(* A value within a bound is within a later one, and within none. *)
Lemma within_later (rs : region_state) (stop stop' : option symbol)
    (v : value) :
  (∀ b' : symbol,
     stop' = Some b' -> ∃ b : symbol, stop = Some b ∧ (b <= b')%positive) ->
  value_within rs stop v = true ->
  value_within rs stop' v = true.
Proof.
  intros Hlater.
  assert (Hbefore :
    ∀ x : symbol, before stop x = true -> before stop' x = true).
  { intros x. destruct stop' as [b' | ]; [ | reflexivity ].
    destruct (Hlater b' eq_refl) as (b & -> & Hle). cbn [before].
    intros H. apply Pos.ltb_lt in H. apply Pos.ltb_lt. lia. }
  unfold value_within. intros H.
  apply andb_true_iff in H as [Hsymbols Hcells]. apply andb_true_iff. split.
  - revert Hsymbols. apply value_all_mono. intros s. unfold symbol_within.
    destruct (heap_symbol rs s); [ reflexivity | ]. cbn [orb].
    intros H. apply andb_true_iff in H as [-> H]. cbn [andb].
    exact (Hbefore s H).
  - revert Hcells. apply value_all_cells_mono. intros l. unfold cell_within.
    destruct (heap_location rs l); [ reflexivity | ]. cbn [orb].
    destruct (lookup l (rs_cells rs)) as [m | ]; [ | exact (fun H => H) ].
    intros H. apply andb_true_iff in H as [-> H]. cbn [andb].
    exact (Hbefore m H).
Qed.
