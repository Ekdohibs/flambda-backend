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

(* Proofs of the statements of FlambdaNameAbstractionSpec.v. Nothing here
   needs review.

   The laws of a name abstraction come from those of its two parts by
   reasoning about permutations alone. A permutation is undone by its
   reverse, so "the free names of a renamed value" can be read either
   way round ([among_apply_renaming]). Two abstractions that are equal
   have the same free names ([related_free]); transitivity composes the
   two pairs of opening permutations through the middle abstraction;
   equivariance conjugates them. [pattern_match_pair] and
   [pattern_match_well_defined] each build one permutation that carries
   one opening to the other, and show that it moves no name that
   matters. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaMapProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaNameAbstractionSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* Permutations and fresh names                                             *)
(* ======================================================================== *)

Lemma permute_fixed_rev (p : permutation) (name : positive) :
  permute p name = name -> permute (rev p) name = name.
Proof.
  intros Hfixed. rewrite <- Hfixed at 1. apply permute_rev_l.
Qed.

Lemma swap_other (a b name : positive) :
  name <> a -> name <> b -> swap a b name = name.
Proof.
  unfold swap. intros Ha Hb.
  destruct (Pos.eqb_spec name a); [ contradiction | ].
  destruct (Pos.eqb_spec name b); [ contradiction | reflexivity ].
Qed.

Lemma swap_first (a b : positive) : swap a b a = b.
Proof. unfold swap. rewrite Pos.eqb_refl. reflexivity. Qed.

Lemma fresh_name_above (names : list positive) (name : positive) :
  name ∈ names -> (name <= fold_right Pos.max xH names)%positive.
Proof.
  induction names as [ | first names IH]; simpl; [ intros [] | ].
  intros [-> | Hin]; [ apply Pos.le_max_l | ].
  eapply Pos.le_trans; [ exact (IH Hin) | apply Pos.le_max_r ].
Qed.

Theorem fresh_name_fresh : fresh_name_fresh_statement.
Proof.
  intros names.
  destruct (among (fresh_name names) names) eqn:Hamong; [ | reflexivity ].
  apply among_In, fresh_name_above in Hamong. unfold fresh_name in Hamong.
  lia.
Qed.

Lemma exists_fresh (names : list positive) :
  ∃ name : positive, among name names = false.
Proof. exists (fresh_name names). apply fresh_name_fresh. Qed.

(* ======================================================================== *)
(* What the laws give                                                       *)
(* ======================================================================== *)

Section Laws.

Context {T : Type} (N : contains_names T) (HN : nominal N).

Lemma apply_renaming_app (p q : permutation) (t : T) :
  apply_renaming N (p ++ q) t = apply_renaming N p (apply_renaming N q t).
Proof. exact (apply_renaming_compose N HN p q t). Qed.

Lemma apply_renaming_rev_l (p : permutation) (t : T) :
  apply_renaming N (rev p) (apply_renaming N p t) = t.
Proof.
  rewrite <- apply_renaming_app.
  rewrite (apply_renaming_same N HN (rev p ++ p) identity_permutation).
  - apply (apply_renaming_identity N HN).
  - intros name. rewrite permute_app. apply permute_rev_l.
Qed.

Lemma apply_renaming_rev_r (p : permutation) (t : T) :
  apply_renaming N p (apply_renaming N (rev p) t) = t.
Proof.
  rewrite <- apply_renaming_app.
  rewrite (apply_renaming_same N HN (p ++ rev p) identity_permutation).
  - apply (apply_renaming_identity N HN).
  - intros name. rewrite permute_app. apply permute_rev_r.
Qed.

(* The free names of a renamed value, read from the value. *)
Lemma among_apply_renaming (p : permutation) (t : T) (name : positive) :
  among name (free_names N (apply_renaming N p t))
  = among (permute (rev p) name) (free_names N t).
Proof.
  rewrite <- (free_names_equivariant N HN p t (permute (rev p) name)),
    permute_rev_r.
  reflexivity.
Qed.

End Laws.

(* ======================================================================== *)
(* An equivariant function invents no names                                 *)
(* ======================================================================== *)

(* If [name] were free in the result and not in the argument, swap it
   with a fresh name: the argument does not change, so the result does
   not, but [name] has left it. *)
Theorem invents_no_names : invents_no_names_statement.
Proof.
  intros X Y NX NY f x name HX HY Hequivariant Hrespects Hin.
  destruct (among name (free_names NX x)) eqn:Hfree; [ reflexivity | ].
  exfalso.
  destruct (exists_fresh (name :: free_names NX x ++ free_names NY (f x)))
    as [b Hb].
  rewrite among_cons, among_app in Hb.
  apply orb_false_iff in Hb as [Hb_name Hb].
  apply orb_false_iff in Hb as [Hb_x Hb_fx].
  assert (Hfixes : fixes [(name, b)] (free_names NX x)).
  { intros n Hn. simpl. apply swap_other; intros ->; congruence. }
  assert (Hequal :
    equal NY (apply_renaming NY [(name, b)] (f x)) (f x)).
  { apply (equal_trans NY HY) with (f (apply_renaming NX [(name, b)] x)).
    - apply (equal_sym NY HY), Hequivariant.
    - apply Hrespects, (apply_renaming_fixes NX HX), Hfixes. }
  apply (free_names_equal NY HY) in Hequal. specialize (Hequal b).
  rewrite Hb_fx in Hequal.
  pose proof (free_names_equivariant NY HY [(name, b)] (f x) name) as Hmoved.
  simpl in Hmoved. rewrite swap_first in Hmoved. congruence.
Qed.

(* ======================================================================== *)
(* The instances that need no term                                          *)
(* ======================================================================== *)

(* The laws of an instance whose [equal] is [=]: those that say it is
   an equivalence, and that it is preserved. *)
Ltac equal_is_eq :=
  simpl; intros; subst; first [ reflexivity | intros ?; reflexivity ].

Theorem no_names_nominal : no_names_nominal_statement.
Proof. intros T. constructor; equal_is_eq. Qed.

Lemma among_permuted (p : permutation) (name : positive)
    (names : list positive) :
  among (permute p name) (map (permute p) names) = among name names.
Proof.
  induction names as [ | first names IH]; [ reflexivity | ]. cbn [map].
  rewrite !among_cons, permute_eqb, IH. reflexivity.
Qed.

Lemma list_of_names_nominal : nominal list_of_names.
Proof.
  constructor.
  - equal_is_eq.
  - equal_is_eq.
  - intros names names' names'' Hequal Hequal'. simpl in *. congruence.
  - intros names. simpl.
    induction names as [ | name names IH]; simpl; congruence.
  - intros second first names. simpl. rewrite map_map. apply map_ext.
    intros name. apply permute_app.
  - intros p p' names Hsame. simpl. apply map_ext, Hsame.
  - equal_is_eq.
  - intros p names name. apply among_permuted.
  - equal_is_eq.
  - intros p names Hfixes. simpl in *.
    induction names as [ | name names IH]; [ reflexivity | ]. cbn [map].
    rewrite (Hfixes name), IH; [ reflexivity | | ].
    + intros other Hother. apply Hfixes. rewrite among_cons, Hother.
      apply orb_true_r.
    + rewrite among_cons, Pos.eqb_refl. reflexivity.
Qed.

Theorem list_of_names_bindable : list_of_names_bindable_statement.
Proof.
  constructor.
  - exact list_of_names_nominal.
  - intros names names'. simpl. split; auto.
  - intros p p' names name. simpl.
    induction names as [ | first names IH]; [ discriminate | ]. cbn [map].
    intros [= Hfirst Hrest]. rewrite among_cons.
    destruct (Pos.eqb_spec name first) as [-> | _]; [ auto | ].
    exact (IH Hrest).
Qed.

Lemma single_name_nominal : nominal single_name.
Proof.
  constructor.
  - equal_is_eq.
  - equal_is_eq.
  - intros name name' name'' Hequal Hequal'. simpl in *. congruence.
  - equal_is_eq.
  - intros second first name. exact (permute_app second first name).
  - intros p p' name Hsame. exact (Hsame name).
  - equal_is_eq.
  - intros p name other. exact (among_permuted p other [name]).
  - equal_is_eq.
  - intros p name Hfixes. simpl in *. apply Hfixes.
    rewrite among_cons, Pos.eqb_refl. reflexivity.
Qed.

Theorem single_name_bindable : single_name_bindable_statement.
Proof.
  constructor.
  - exact single_name_nominal.
  - intros name name'. simpl. split; auto.
  - intros p p' b name Hsame Hname. simpl in *.
    rewrite orb_false_r in Hname. apply Pos.eqb_eq in Hname. subst name.
    exact Hsame.
Qed.

(* ======================================================================== *)
(* Pairs                                                                    *)
(* ======================================================================== *)

Theorem pair_nominal : pair_nominal_statement.
Proof.
  intros A B NA NB HA HB. constructor; simpl.
  - intros t. split; [ apply (equal_refl NA HA) | apply (equal_refl NB HB) ].
  - intros t t' [Ha Hb].
    split; [ apply (equal_sym NA HA), Ha | apply (equal_sym NB HB), Hb ].
  - intros t t' t'' [Ha Hb] [Ha' Hb'].
    split;
      [ exact (equal_trans NA HA _ _ _ Ha Ha')
      | exact (equal_trans NB HB _ _ _ Hb Hb') ].
  - intros [a b]. simpl.
    rewrite (apply_renaming_identity NA HA), (apply_renaming_identity NB HB).
    reflexivity.
  - intros second first [a b]. simpl.
    rewrite (apply_renaming_compose NA HA), (apply_renaming_compose NB HB).
    reflexivity.
  - intros p p' [a b] Hsame. simpl.
    rewrite (apply_renaming_same NA HA p p' a Hsame),
      (apply_renaming_same NB HB p p' b Hsame).
    reflexivity.
  - intros p t t' [Ha Hb].
    split;
      [ apply (equal_equivariant NA HA), Ha
      | apply (equal_equivariant NB HB), Hb ].
  - intros p [a b] name. simpl.
    rewrite !among_app, (free_names_equivariant NA HA),
      (free_names_equivariant NB HB).
    reflexivity.
  - intros t t' [Ha Hb] name.
    rewrite !among_app, (free_names_equal NA HA _ _ Ha name),
      (free_names_equal NB HB _ _ Hb name).
    reflexivity.
  - intros p [a b] Hfixes. simpl in *. split.
    + apply (apply_renaming_fixes NA HA). intros name Hname.
      apply Hfixes. rewrite among_app, Hname. reflexivity.
    + apply (apply_renaming_fixes NB HB). intros name Hname.
      apply Hfixes. rewrite among_app, Hname. apply orb_true_r.
Qed.

(* ======================================================================== *)
(* Name abstractions                                                        *)
(* ======================================================================== *)

Section Abstractions.

Context {B T : Type} (NB : contains_names B) (NT : contains_names T).

Hypothesis HB : nominal NB.

Hypothesis HT : nominal NT.

Lemma among_free_names_abstraction (a : name_abstraction B T)
    (name : positive) :
  among name (free_names_abstraction NB NT a)
  = among name (free_names NT (scope a))
    && negb (among name (free_names NB (bound a))).
Proof. apply among_without. Qed.

Lemma free_names_abstraction_equivariant (p : permutation)
    (a : name_abstraction B T) (name : positive) :
  among (permute p name)
    (free_names_abstraction NB NT (apply_renaming_abstraction NB NT p a))
  = among name (free_names_abstraction NB NT a).
Proof.
  rewrite !among_free_names_abstraction. simpl.
  rewrite (free_names_equivariant NT HT), (free_names_equivariant NB HB).
  reflexivity.
Qed.

Lemma among_renamed_abstraction (p : permutation)
    (a : name_abstraction B T) (name : positive) :
  among name
    (free_names_abstraction NB NT (apply_renaming_abstraction NB NT p a))
  = among (permute (rev p) name) (free_names_abstraction NB NT a).
Proof.
  rewrite <- (free_names_abstraction_equivariant p a (permute (rev p) name)),
    permute_rev_r.
  reflexivity.
Qed.

(* A name free in one of two abstractions that open to the same bound
   names and equal scopes is free in the other. *)
Lemma related_free (p p' : permutation) (a a' : name_abstraction B T)
    (name : positive) :
  opens NB NT p a ->
  opens NB NT p' a' ->
  apply_renaming NB p (bound a) = apply_renaming NB p' (bound a') ->
  equal NT (apply_renaming NT p (scope a)) (apply_renaming NT p' (scope a')) ->
  among name (free_names_abstraction NB NT a) = true ->
  among name (free_names_abstraction NB NT a') = true.
Proof.
  intros Hopens Hopens' Hbound Hscope Hfree.
  pose proof (Hopens name Hfree) as Hfixed.
  rewrite among_free_names_abstraction in Hfree.
  apply andb_true_iff in Hfree as [Hin_scope Hnot_bound].
  apply negb_true_iff in Hnot_bound.
  assert (Hin_scope' :
    among (permute (rev p') name) (free_names NT (scope a')) = true).
  { rewrite <- (among_apply_renaming NT HT p' (scope a') name),
      <- (free_names_equal NT HT _ _ Hscope name),
      (among_apply_renaming NT HT p), (permute_fixed_rev p name Hfixed).
    exact Hin_scope. }
  assert (Hnot_bound' :
    among (permute (rev p') name) (free_names NB (bound a')) = false).
  { destruct (among (permute (rev p') name) (free_names NB (bound a')))
      eqn:Hbound'; [ exfalso | reflexivity ].
    assert (Hbound_name : among name (free_names NB (bound a)) = true).
    { rewrite <- (permute_fixed_rev p name Hfixed) at 1.
      rewrite <- (among_apply_renaming NB HB), Hbound,
        (among_apply_renaming NB HB).
      exact Hbound'. }
    congruence. }
  assert (Hfree' :
    among (permute (rev p') name) (free_names_abstraction NB NT a') = true)
    by (rewrite among_free_names_abstraction, Hin_scope', Hnot_bound';
        reflexivity).
  pose proof (Hopens' _ Hfree') as Hfixed'.
  rewrite permute_rev_r in Hfixed'. rewrite <- Hfixed' in Hfree'.
  exact Hfree'.
Qed.

Lemma equal_abstraction_refl (a : name_abstraction B T) :
  equal_abstraction NB NT a a.
Proof.
  exists identity_permutation, identity_permutation.
  split; [ intros name _; reflexivity | ].
  split; [ intros name _; reflexivity | ].
  split; [ reflexivity | apply (equal_refl NT HT) ].
Qed.

Lemma equal_abstraction_sym (a a' : name_abstraction B T) :
  equal_abstraction NB NT a a' -> equal_abstraction NB NT a' a.
Proof.
  intros (p & p' & Hopens & Hopens' & Hbound & Hscope). exists p', p.
  split; [ exact Hopens' | ]. split; [ exact Hopens | ].
  split; [ symmetry; exact Hbound | apply (equal_sym NT HT), Hscope ].
Qed.

Lemma equal_abstraction_free_names (a a' : name_abstraction B T) :
  equal_abstraction NB NT a a' ->
  same_names (free_names_abstraction NB NT a)
    (free_names_abstraction NB NT a').
Proof.
  intros (p & p' & Hopens & Hopens' & Hbound & Hscope) name.
  simpl in Hbound, Hscope. apply eq_true_iff_eq. split.
  - apply (related_free p p'); assumption.
  - apply (related_free p' p);
      [ assumption | assumption | symmetry; assumption
      | apply (equal_sym NT HT); assumption ].
Qed.

(* Through the middle abstraction: open the first as the second pair
   opens the middle one. *)
Lemma equal_abstraction_trans (a a' a'' : name_abstraction B T) :
  equal_abstraction NB NT a a' ->
  equal_abstraction NB NT a' a'' ->
  equal_abstraction NB NT a a''.
Proof.
  intros Hequal Hequal'.
  pose proof (equal_abstraction_free_names a a' Hequal) as Hnames.
  destruct Hequal as (p1 & p1' & Hopens1 & Hopens1' & Hbound1 & Hscope1).
  destruct Hequal' as (p2 & p2' & Hopens2 & Hopens2' & Hbound2 & Hscope2).
  simpl in Hbound1, Hscope1, Hbound2, Hscope2.
  exists ((p2 ++ rev p1') ++ p1), p2'.
  split; [ | split; [ exact Hopens2' | split ] ]; simpl.
  - intros name Hfree.
    assert (Hfree' : among name (free_names_abstraction NB NT a') = true)
      by (rewrite <- (Hnames name); exact Hfree).
    rewrite !permute_app, (Hopens1 name Hfree),
      (permute_fixed_rev p1' name (Hopens1' name Hfree')).
    exact (Hopens2 name Hfree').
  - rewrite !(apply_renaming_app NB HB), Hbound1,
      (apply_renaming_rev_l NB HB).
    exact Hbound2.
  - rewrite !(apply_renaming_app NT HT).
    apply (equal_trans NT HT) with (apply_renaming NT p2 (scope a'));
      [ | exact Hscope2 ].
    pose proof
      (equal_equivariant NT HT p2 _ _
         (equal_equivariant NT HT (rev p1') _ _ Hscope1))
      as Hmoved.
    rewrite (apply_renaming_rev_l NT HT) in Hmoved. exact Hmoved.
Qed.

(* The opening permutations of the renamed abstractions are the
   conjugates. *)
Lemma equal_abstraction_equivariant (r : permutation)
    (a a' : name_abstraction B T) :
  equal_abstraction NB NT a a' ->
  equal_abstraction NB NT (apply_renaming_abstraction NB NT r a)
    (apply_renaming_abstraction NB NT r a').
Proof.
  intros (p & p' & Hopens & Hopens' & Hbound & Hscope).
  simpl in Hbound, Hscope.
  exists ((r ++ p) ++ rev r), ((r ++ p') ++ rev r).
  assert (Hconjugate :
    ∀ (q : permutation) (c : name_abstraction B T),
      opens NB NT q c ->
      opens NB NT ((r ++ q) ++ rev r)
        (apply_renaming_abstraction NB NT r c)).
  { intros q c Hq name Hfree. rewrite among_renamed_abstraction in Hfree.
    rewrite !permute_app, (Hq _ Hfree). apply permute_rev_r. }
  split; [ apply Hconjugate, Hopens | ].
  split; [ apply Hconjugate, Hopens' | ]. simpl. split.
  - rewrite !(apply_renaming_app NB HB), !(apply_renaming_rev_l NB HB),
      Hbound.
    reflexivity.
  - rewrite !(apply_renaming_app NT HT), !(apply_renaming_rev_l NT HT).
    apply (equal_equivariant NT HT), Hscope.
Qed.

(* The defining property of an abstraction: its free names are a
   support. *)
Lemma apply_renaming_abstraction_fixes (r : permutation)
    (a : name_abstraction B T) :
  fixes r (free_names_abstraction NB NT a) ->
  equal_abstraction NB NT (apply_renaming_abstraction NB NT r a) a.
Proof.
  intros Hfixes. exists (rev r), identity_permutation. split.
  { intros name Hfree. rewrite among_renamed_abstraction in Hfree.
    pose proof (Hfixes _ Hfree) as Hfixed.
    rewrite permute_rev_r in Hfixed. symmetry. exact Hfixed. }
  split; [ intros name _; reflexivity | ]. simpl.
  rewrite (apply_renaming_rev_l NB HB), (apply_renaming_rev_l NT HT),
    (apply_renaming_identity NB HB), (apply_renaming_identity NT HT).
  split; [ reflexivity | apply (equal_refl NT HT) ].
Qed.

End Abstractions.

Theorem abstraction_nominal : abstraction_nominal_statement.
Proof.
  intros B T NB NT Hbindable HT.
  pose proof (bindable_nominal NB Hbindable) as HB.
  constructor; simpl.
  - apply equal_abstraction_refl, HT.
  - apply equal_abstraction_sym, HT.
  - apply equal_abstraction_trans; assumption.
  - intros [b t]. unfold apply_renaming_abstraction. simpl.
    rewrite (apply_renaming_identity NB HB), (apply_renaming_identity NT HT).
    reflexivity.
  - intros second first [b t]. unfold apply_renaming_abstraction. simpl.
    rewrite (apply_renaming_compose NB HB), (apply_renaming_compose NT HT).
    reflexivity.
  - intros p p' [b t] Hsame. unfold apply_renaming_abstraction. simpl.
    rewrite (apply_renaming_same NB HB p p' b Hsame),
      (apply_renaming_same NT HT p p' t Hsame).
    reflexivity.
  - apply equal_abstraction_equivariant; assumption.
  - apply free_names_abstraction_equivariant; assumption.
  - apply equal_abstraction_free_names; assumption.
  - apply apply_renaming_abstraction_fixes; assumption.
Qed.

(* ======================================================================== *)
(* Opening two abstractions                                                 *)
(* ======================================================================== *)

(* With [p0] and [p0'] the permutations by which the two are equal, [p]
   after the inverse of [p0] and [p'] after the inverse of [p0'] agree
   on every name free in the opened scope: on a bound name because they
   agree on the bound names, and on any other because none of the four
   moves it. *)
Theorem pattern_match_pair : pattern_match_pair_statement.
Proof.
  intros B T NB NT a a' p p' Hbindable HT Hequal Hopens Hopens' Hbound.
  pose proof (bindable_nominal NB Hbindable) as HB.
  pose proof (equal_abstraction_free_names NB NT HB HT a a' Hequal)
    as Hnames.
  destruct Hequal as (p0 & p0' & Hopens0 & Hopens0' & Hbound0 & Hscope0).
  simpl in Hbound, Hbound0, Hscope0. simpl.
  assert (Hfixes :
    fixes (rev (p' ++ rev p0') ++ (p ++ rev p0))
      (free_names NT (apply_renaming NT p0' (scope a')))).
  { intros name Hname. rewrite (among_apply_renaming NT HT) in Hname.
    rewrite permute_app.
    assert (Hagree :
      permute (p ++ rev p0) name = permute (p' ++ rev p0') name).
    { destruct (among (permute (rev p0') name) (free_names NB (bound a')))
        eqn:Hbound_name.
      - apply (bindable_faithful NB Hbindable)
          with (b := apply_renaming NB p0 (bound a)).
        + transitivity (apply_renaming NB p (bound a)).
          * rewrite (apply_renaming_app NB HB), (apply_renaming_rev_l NB HB).
            reflexivity.
          * rewrite Hbound0, (apply_renaming_app NB HB),
              (apply_renaming_rev_l NB HB).
            exact Hbound.
        + rewrite Hbound0, (among_apply_renaming NB HB).
          exact Hbound_name.
      - assert (Hfree' :
          among (permute (rev p0') name) (free_names_abstraction NB NT a')
          = true)
          by (rewrite among_free_names_abstraction, Hname, Hbound_name;
              reflexivity).
        pose proof (Hopens0' _ Hfree') as Hfixed.
        rewrite permute_rev_r in Hfixed. rewrite <- Hfixed in Hfree'.
        assert (Hfree : among name (free_names_abstraction NB NT a) = true)
          by (rewrite (Hnames name); exact Hfree').
        rewrite !permute_app,
          (permute_fixed_rev p0 name (Hopens0 name Hfree)),
          (Hopens name Hfree),
          (permute_fixed_rev p0' name (Hopens0' name Hfree')),
          (Hopens' name Hfree').
        reflexivity. }
    rewrite Hagree. apply permute_rev_l. }
  apply (apply_renaming_fixes NT HT) in Hfixes.
  apply (equal_equivariant NT HT (p' ++ rev p0')) in Hfixes.
  rewrite (apply_renaming_app NT HT (rev (p' ++ rev p0'))),
    (apply_renaming_rev_r NT HT) in Hfixes.
  rewrite !(apply_renaming_app NT HT), !(apply_renaming_rev_l NT HT)
    in Hfixes.
  apply (equal_trans NT HT)
    with (apply_renaming NT p
            (apply_renaming NT (rev p0) (apply_renaming NT p0' (scope a'))));
    [ | exact Hfixes ].
  pose proof
    (equal_equivariant NT HT p _ _
       (equal_equivariant NT HT (rev p0) _ _ Hscope0))
    as Hmoved.
  rewrite (apply_renaming_rev_l NT HT) in Hmoved. exact Hmoved.
Qed.

(* One permutation carries the first opening to the second. The function
   is equivariant, so it carries the first result to the second; and it
   moves no free name of the first result, since such a name is not
   bound (the premise) and so is free in both abstractions. *)
Theorem pattern_match_well_defined : pattern_match_well_defined_statement.
Proof.
  intros B T X NB NT NX f a a' p p' Hbindable HT HX Hequivariant Hrespects
    Hnot_bound Hequal Hopens Hopens'.
  pose proof (bindable_nominal NB Hbindable) as HB.
  pose proof (pair_nominal B T NB NT HB HT) as Hpair.
  pose proof (equal_abstraction_free_names NB NT HB HT a a' Hequal)
    as Hnames.
  destruct Hequal as (p0 & p0' & Hopens0 & Hopens0' & Hbound0 & Hscope0).
  simpl in Hbound0, Hscope0. unfold pattern_match.
  set (opened := (apply_renaming NB p (bound a),
                  apply_renaming NT p (scope a))).
  set (carry := ((p' ++ rev p0') ++ p0) ++ rev p).
  assert (Hfixes : fixes carry (free_names NX (f opened))).
  { intros name Hname.
    pose proof
      (invents_no_names _ _ (pair_names NB NT) NX f opened name Hpair HX
         Hequivariant Hrespects Hname)
      as Hin.
    specialize (Hnot_bound opened name).
    simpl in Hin, Hnot_bound. rewrite among_app in Hin.
    destruct (among name (free_names NB (apply_renaming NB p (bound a))))
      eqn:Hbound_name;
      [ specialize (Hnot_bound eq_refl); congruence | ].
    simpl in Hin.
    assert (Hfree :
      among name
        (free_names_abstraction NB NT (apply_renaming_abstraction NB NT p a))
      = true)
      by (rewrite among_free_names_abstraction; simpl;
          rewrite Hin, Hbound_name; reflexivity).
    rewrite (among_renamed_abstraction NB NT HB HT) in Hfree.
    pose proof (Hopens _ Hfree) as Hfixed.
    rewrite permute_rev_r in Hfixed. rewrite <- Hfixed in Hfree.
    assert (Hfree' : among name (free_names_abstraction NB NT a') = true)
      by (rewrite <- (Hnames name); exact Hfree).
    unfold carry. rewrite !permute_app, <- Hfixed, (Hopens0 name Hfree),
      (permute_fixed_rev p0' name (Hopens0' name Hfree')).
    exact (Hopens' name Hfree'). }
  apply (apply_renaming_fixes NX HX) in Hfixes.
  apply (equal_trans NX HX) with (apply_renaming NX carry (f opened));
    [ apply (equal_sym NX HX), Hfixes | ].
  apply (equal_trans NX HX)
    with (f (apply_renaming (pair_names NB NT) carry opened));
    [ apply (equal_sym NX HX), Hequivariant | ].
  apply Hrespects. unfold carry, opened. simpl. split.
  - rewrite !(apply_renaming_app NB HB), (apply_renaming_rev_l NB HB),
      Hbound0, (apply_renaming_rev_l NB HB).
    apply (equal_refl NB HB).
  - rewrite !(apply_renaming_app NT HT), (apply_renaming_rev_l NT HT).
    pose proof
      (equal_equivariant NT HT p' _ _
         (equal_equivariant NT HT (rev p0') _ _ Hscope0))
      as Hmoved.
    rewrite (apply_renaming_rev_l NT HT) in Hmoved. exact Hmoved.
Qed.
