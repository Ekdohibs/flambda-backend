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

(* Proofs of the statements of FlambdaFreshQuantifierSpec.v. Nothing here
   needs review.

   Each law that needs no support is proved by putting two finite sets
   together, or by taking a name outside one with [fresh_name]. Each law
   that needs a support goes through the agreement of the three readings
   ([some_any]), whose proof is one swap: a name outside the support at
   which the predicate holds is exchanged for any other name outside
   it. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaMapProofs
  FlambdaNominalSpec FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaFreshQuantifierSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* Names outside a list                                                     *)
(* ======================================================================== *)

Lemma among_app_false (name : positive) (names names' : list positive) :
  among name (names ++ names') = false
  <-> among name names = false ∧ among name names' = false.
Proof. rewrite among_app. apply orb_false_iff. Qed.

Lemma among_cons_false (name first : positive) (names : list positive) :
  among name (first :: names) = false
  <-> name <> first ∧ among name names = false.
Proof. rewrite among_cons, orb_false_iff, Pos.eqb_neq. reflexivity. Qed.

Lemma swap_second (a b : positive) : swap a b b = a.
Proof.
  unfold swap. destruct (Pos.eqb_spec b a) as [-> | _]; [ reflexivity | ].
  rewrite Pos.eqb_refl. reflexivity.
Qed.

(* A swap of two names outside a list moves none of its names. *)
Lemma fixes_swap (a b : positive) (names : list positive) :
  among a names = false -> among b names = false -> fixes [(a, b)] names.
Proof.
  intros Ha Hb name Hname. simpl. apply swap_other; intros ->; congruence.
Qed.

(* ======================================================================== *)
(* The quantifier                                                           *)
(* ======================================================================== *)

Theorem for_fresh_intro : for_fresh_intro_statement.
Proof. intros P avoid Hall. exists avoid. exact Hall. Qed.

Theorem for_fresh_elim : for_fresh_elim_statement.
Proof.
  intros P names [avoid Hall]. exists (fresh_name (avoid ++ names)).
  pose proof (fresh_name_fresh (avoid ++ names)) as Hfresh.
  apply among_app_false in Hfresh as [Havoid Hnames].
  split; [ exact Hnames | exact (Hall _ Havoid) ].
Qed.

Theorem for_fresh_mono : for_fresh_mono_statement.
Proof.
  intros P Q [avoid Himpl] [avoid' Hall]. exists (avoid ++ avoid').
  intros name Hname. apply among_app_false in Hname as [Hname Hname'].
  exact (Himpl name Hname (Hall name Hname')).
Qed.

(* The form in which monotonicity is used most. *)
Lemma for_fresh_weaken (P Q : positive -> Prop) :
  (∀ name : positive, P name -> Q name) ->
  (И name, P name) -> И name, Q name.
Proof.
  intros Himpl [avoid Hall]. exists avoid. intros name Hname.
  exact (Himpl name (Hall name Hname)).
Qed.

(* ======================================================================== *)
(* The connectives, for any predicate                                       *)
(* ======================================================================== *)

Theorem for_fresh_and : for_fresh_and_statement.
Proof.
  intros P Q. split.
  - intros Hboth.
    split; [ apply (for_fresh_weaken _ P) in Hboth
           | apply (for_fresh_weaken _ Q) in Hboth ];
      solve [ exact Hboth | intros name [HP HQ]; assumption ].
  - intros [[avoid HP] [avoid' HQ]]. exists (avoid ++ avoid').
    intros name Hname. apply among_app_false in Hname as [Hname Hname'].
    split; [ exact (HP name Hname) | exact (HQ name Hname') ].
Qed.

Theorem for_fresh_constant : for_fresh_constant_statement.
Proof.
  intros Q. split.
  - intros Hfresh.
    destruct (for_fresh_elim (fun _ => Q) [] Hfresh) as (_ & _ & HQ).
    exact HQ.
  - intros HQ. exists []. intros _ _. exact HQ.
Qed.

Theorem for_fresh_or_of_either : for_fresh_or_of_either_statement.
Proof.
  intros P Q [HP | HQ];
    [ apply (for_fresh_weaken P) | apply (for_fresh_weaken Q) ];
    auto.
Qed.

Theorem for_fresh_not_gives_not : for_fresh_not_gives_not_statement.
Proof.
  intros P Hnot HP. apply (for_fresh_constant False).
  apply (for_fresh_mono P); [ | exact HP ].
  apply (for_fresh_weaken (fun name => ~ P name)); [ | exact Hnot ].
  intros name HnotP HPname. exact (HnotP HPname).
Qed.

Theorem for_fresh_forall : for_fresh_forall_statement.
Proof.
  intros X P Hall x.
  apply (for_fresh_weaken (fun name => ∀ x : X, P name x)); [ | exact Hall ].
  intros name Hname. exact (Hname x).
Qed.

Theorem for_fresh_exists : for_fresh_exists_statement.
Proof.
  intros X P [x Hx].
  apply (for_fresh_weaken (fun name => P name x)); [ | exact Hx ].
  intros name Hname. exists x. exact Hname.
Qed.

Theorem for_fresh_avoids : for_fresh_avoids_statement.
Proof. intros names. exists names. intros name Hname. exact Hname. Qed.

Theorem for_fresh_distinct : for_fresh_distinct_statement.
Proof.
  exists []. intros a _. exists [a]. intros b Hb.
  apply among_cons_false in Hb as [Hb _]. intros ->. exact (Hb eq_refl).
Qed.

(* ======================================================================== *)
(* Predicates with a support                                                *)
(* ======================================================================== *)

(* A predicate that holds of one name outside its support holds of every
   other: exchange the two. *)
Lemma supported_some_any (support : list positive) (P : positive -> Prop)
    (a b : positive) :
  supports single_name support P ->
  among a support = false ->
  among b support = false ->
  P a -> P b.
Proof.
  intros Hsupports Ha Hb HP.
  pose proof (Hsupports [(a, b)] a (fixes_swap a b support Ha Hb) HP)
    as Hmoved.
  simpl in Hmoved. rewrite swap_first in Hmoved. exact Hmoved.
Qed.

Lemma supported_any (support : list positive) (P : positive -> Prop) :
  supports single_name support P ->
  (И name, P name) ->
  ∀ name : positive, among name support = false -> P name.
Proof.
  intros Hsupports Hfresh name Hname.
  destruct (for_fresh_elim P support Hfresh) as (witness & Hwitness & HP).
  exact (supported_some_any support P witness name Hsupports Hwitness Hname
           HP).
Qed.

Lemma supported_some (support : list positive) (P : positive -> Prop)
    (witness : positive) :
  supports single_name support P ->
  among witness support = false ->
  P witness ->
  И name, P name.
Proof.
  intros Hsupports Hwitness HP. exists support. intros name Hname.
  exact (supported_some_any support P witness name Hsupports Hwitness Hname
           HP).
Qed.

Theorem some_any : some_any_statement.
Proof.
  intros support P Hsupports. split; split.
  - apply supported_any, Hsupports.
  - intros Hall. exists support. exact Hall.
  - apply for_fresh_elim.
  - intros (witness & Hwitness & HP).
    exact (supported_some support P witness Hsupports Hwitness HP).
Qed.

Theorem for_fresh_or : for_fresh_or_statement.
Proof.
  intros support support' P Q HsupportsP HsupportsQ.
  split; [ | apply for_fresh_or_of_either ].
  intros Heither.
  destruct
    (for_fresh_elim (fun name => P name ∨ Q name) (support ++ support')
       Heither)
    as (witness & Hwitness & [HP | HQ]);
    apply among_app_false in Hwitness as [Hwitness Hwitness'].
  - left. exact (supported_some support P witness HsupportsP Hwitness HP).
  - right. exact (supported_some support' Q witness HsupportsQ Hwitness' HQ).
Qed.

Theorem for_fresh_not : for_fresh_not_statement.
Proof.
  intros support P Hsupports. split; [ apply for_fresh_not_gives_not | ].
  intros Hnot. exists support. intros name Hname HP. apply Hnot.
  exact (supported_some support P name Hsupports Hname HP).
Qed.

Theorem for_fresh_impl : for_fresh_impl_statement.
Proof.
  intros support support' P Q HsupportsP HsupportsQ.
  split; [ apply for_fresh_mono | ].
  intros Himpl. exists (support ++ support'). intros name Hname HP.
  apply among_app_false in Hname as [Hname Hname'].
  apply (supported_any support' Q HsupportsQ); [ | exact Hname' ].
  apply Himpl. exact (supported_some support P name HsupportsP Hname HP).
Qed.

(* Two fresh names at which the predicate holds are carried to any two
   different names outside the support by two swaps. *)
Lemma supported_two (support : list positive)
    (P : positive * positive -> Prop) :
  supports (pair_names single_name single_name) support P ->
  ((И a b, P (a, b))
   <-> (∀ a b : positive,
         among a support = false -> among b support = false -> a <> b ->
         P (a, b))).
Proof.
  intros Hsupports. split.
  - intros Hfresh a b Ha Hb Hdifferent.
    destruct
      (for_fresh_elim (fun a => И b, P (a, b)) (a :: b :: support) Hfresh)
      as (a' & Ha' & Hfresh').
    destruct
      (for_fresh_elim (fun b => P (a', b)) (a :: b :: a' :: support) Hfresh')
      as (b' & Hb' & HP).
    apply among_cons_false in Ha' as [Ha'_a Ha'].
    apply among_cons_false in Ha' as [Ha'_b Ha'].
    apply among_cons_false in Hb' as [Hb'_a Hb'].
    apply among_cons_false in Hb' as [Hb'_b Hb'].
    apply among_cons_false in Hb' as [Hb'_a' Hb'].
    assert (Hfixes : fixes [(a, a'); (b, b')] support).
    { intros name Hname. simpl.
      rewrite (swap_other b b'), (swap_other a a');
        solve [ reflexivity | intros ->; congruence ]. }
    pose proof (Hsupports _ _ Hfixes HP) as Hmoved. simpl in Hmoved.
    rewrite (swap_other b b' a'), (swap_second a a'), (swap_second b b'),
      (swap_other a a' b) in Hmoved;
      solve [ exact Hmoved | congruence ].
  - intros Hany. exists support. intros a Ha. exists (a :: support).
    intros b Hb. apply among_cons_false in Hb as [Hdifferent Hb].
    apply Hany; [ exact Ha | exact Hb | congruence ].
Qed.

Theorem for_fresh_two : for_fresh_two_statement.
Proof. exact supported_two. Qed.

Theorem for_fresh_swap : for_fresh_swap_statement.
Proof.
  intros support P Hsupports.
  rewrite (supported_two support P Hsupports).
  assert (Hsupports' :
    supports (pair_names single_name single_name) support
      (fun ba => P (snd ba, fst ba))).
  { intros p [b a] Hfixes HP. exact (Hsupports p (a, b) Hfixes HP). }
  pose proof (supported_two support _ Hsupports') as Hswapped.
  simpl in Hswapped. rewrite Hswapped. split.
  - intros Hany b a Hb Ha Hdifferent. apply Hany; congruence.
  - intros Hany a b Ha Hb Hdifferent. apply Hany; congruence.
Qed.

(* The names up to a given one, for the predicate that looks at their
   order. *)
Definition names_up_to (a : positive) : list positive :=
  map Pos.of_succ_nat (seq 0 (Pos.to_nat a)).

Lemma among_names_up_to (a b : positive) :
  (b <= a)%positive -> among b (names_up_to a) = true.
Proof.
  intros Hle. apply among_In. unfold names_up_to. apply in_map_iff.
  exists (Nat.pred (Pos.to_nat b)). split; [ | apply in_seq; lia ].
  rewrite <- (Pos2Nat.id b) at 2.
  destruct (Pos2Nat.is_succ b) as [n Hn]. rewrite Hn. cbn [Nat.pred].
  apply Pos.of_nat_succ.
Qed.

Theorem for_fresh_swap_needs_support : for_fresh_swap_needs_support_statement.
Proof.
  split.
  - exists []. intros a _. exists (names_up_to a). intros b Hb.
    destruct (Pos.ltb_spec a b) as [Hlt | Hle]; [ exact Hlt | ].
    rewrite (among_names_up_to a b Hle) in Hb. discriminate.
  - intros Hfresh.
    destruct (for_fresh_elim _ [] Hfresh) as (b & _ & Hfresh').
    destruct (for_fresh_elim _ (names_up_to b) Hfresh') as (a & Ha & Hlt).
    rewrite among_names_up_to in Ha; [ discriminate | lia ].
Qed.

(* ======================================================================== *)
(* Equivariant predicates                                                   *)
(* ======================================================================== *)

Theorem propositions_nominal : propositions_nominal_statement.
Proof.
  constructor; simpl; try reflexivity.
  - intros t t' Hequal. symmetry. exact Hequal.
  - intros t t' t'' Hequal Hequal'. rewrite Hequal. exact Hequal'.
  - intros p t t' Hequal. exact Hequal.
  - intros t t' _ name. reflexivity.
Qed.

(* Rename the name and the value together, then put the value back: it
   is equal to its renaming, since the permutation fixes its free
   names. *)
Theorem equivariant_supports : equivariant_supports_statement.
Proof.
  intros B X NB N R x HB HN Hequivariant Hrespects p b Hfixes HR.
  apply (Hrespects (apply_renaming NB p b, apply_renaming N p x)).
  - split; [ apply (equal_refl NB HB) | ]. simpl.
    exact (apply_renaming_fixes N HN p x Hfixes).
  - apply (Hequivariant p (b, x)). exact HR.
Qed.

Theorem some_any_nominal : some_any_nominal_statement.
Proof.
  intros X N R x HN Hequivariant Hrespects.
  exact (some_any (free_names N x) (fun name => R (name, x))
           (equivariant_supports positive X single_name N R x
              single_name_nominal HN Hequivariant Hrespects)).
Qed.

(* ======================================================================== *)
(* Several fresh names                                                      *)
(* ======================================================================== *)

(* The name made for a list with the name made for it is the next
   one. *)
Lemma fresh_name_after_fresh_name (avoid : list positive) :
  fresh_name (fresh_name avoid :: avoid) = Pos.succ (fresh_name avoid).
Proof.
  unfold fresh_name. cbn [fold_right]. f_equal.
  apply Pos.max_l, Pos.lt_le_incl, Pos.lt_succ_diag_r.
Qed.

Lemma fresh_names_as_in_turn (count : nat) (avoid : list positive) :
  fresh_names count avoid = fresh_names_in_turn count avoid.
Proof.
  unfold fresh_names. revert avoid.
  induction count as [ | count IH]; intros avoid; [ reflexivity | ].
  cbn [names_from fresh_names_in_turn]. f_equal.
  rewrite <- IH, fresh_name_after_fresh_name. reflexivity.
Qed.

Theorem fresh_names_equal_in_turn : fresh_names_equal_in_turn_statement.
Proof. exact fresh_names_as_in_turn. Qed.

Lemma fresh_names_in_turn_length (count : nat) (avoid : list positive) :
  length (fresh_names_in_turn count avoid) = count.
Proof.
  revert avoid. induction count as [ | count IH]; intros avoid; simpl;
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Lemma fresh_names_in_turn_outside (count : nat) (avoid : list positive) :
  distinct_outside avoid (fresh_names_in_turn count avoid).
Proof.
  revert avoid. induction count as [ | count IH]; intros avoid.
  - split; [ reflexivity | intros name Hname; discriminate ].
  - cbn [fresh_names_in_turn]. set (first := fresh_name avoid).
    destruct (IH (first :: avoid)) as [Hdistinct Houtside]. split.
    + cbn [distinct]. rewrite Hdistinct, andb_true_r.
      destruct (among first (fresh_names_in_turn count (first :: avoid)))
        eqn:Hamong; [ | reflexivity ].
      apply Houtside, among_cons_false in Hamong as [Hfirst _].
      contradiction.
    + intros name Hname. rewrite among_cons in Hname.
      apply orb_true_iff in Hname as [Hname | Hname].
      * apply Pos.eqb_eq in Hname. subst name. apply fresh_name_fresh.
      * apply Houtside, among_cons_false in Hname as [_ Hname].
        exact Hname.
Qed.

Lemma fresh_names_length (count : nat) (avoid : list positive) :
  length (fresh_names count avoid) = count.
Proof.
  rewrite fresh_names_as_in_turn. apply fresh_names_in_turn_length.
Qed.

Lemma fresh_names_outside (count : nat) (avoid : list positive) :
  distinct_outside avoid (fresh_names count avoid).
Proof.
  rewrite fresh_names_as_in_turn. apply fresh_names_in_turn_outside.
Qed.

Theorem fresh_names_fresh : fresh_names_fresh_statement.
Proof.
  intros count avoid.
  split; [ apply fresh_names_length | apply fresh_names_outside ].
Qed.

Lemma distinct_outside_app (avoid avoid' names : list positive) :
  distinct_outside (avoid ++ avoid') names ->
  distinct_outside avoid names ∧ distinct_outside avoid' names.
Proof.
  intros [Hdistinct Houtside].
  split; (split; [ exact Hdistinct | ]); intros name Hname;
    apply Houtside, among_app_false in Hname as [Hname Hname'];
    assumption.
Qed.

Lemma distinct_outside_cons (avoid : list positive) (name : positive)
    (names : list positive) :
  among name avoid = false ->
  distinct_outside (name :: avoid) names ->
  distinct_outside avoid (name :: names).
Proof.
  intros Hname [Hdistinct Houtside]. split.
  - cbn [distinct]. rewrite Hdistinct, andb_true_r.
    destruct (among name names) eqn:Hamong; [ | reflexivity ].
    apply Houtside, among_cons_false in Hamong as [Hdifferent _].
    contradiction.
  - intros other Hother. rewrite among_cons in Hother.
    apply orb_true_iff in Hother as [Hother | Hother].
    + apply Pos.eqb_eq in Hother. subst other. exact Hname.
    + apply Houtside, among_cons_false in Hother as [_ Hother].
      exact Hother.
Qed.

Theorem for_fresh_names_elim : for_fresh_names_elim_statement.
Proof.
  intros count P avoid [avoid' Hall].
  exists (fresh_names count (avoid' ++ avoid)).
  destruct
    (distinct_outside_app avoid' avoid _
       (fresh_names_outside count (avoid' ++ avoid)))
    as [Houtside' Houtside].
  split; [ apply fresh_names_length | ].
  split; [ exact Houtside | ].
  apply Hall; [ apply fresh_names_length | exact Houtside' ].
Qed.

Theorem for_fresh_names_none : for_fresh_names_none_statement.
Proof.
  intros P. split.
  - intros [avoid Hall]. apply Hall; [ reflexivity | ].
    split; [ reflexivity | intros name Hname; discriminate ].
  - intros HP. exists []. intros names Hlength _.
    destruct names; [ exact HP | discriminate ].
Qed.

Theorem for_fresh_names_next : for_fresh_names_next_statement.
Proof.
  intros count P [avoid Hall]. exists avoid. intros name Hname.
  exists (name :: avoid). intros names Hlength Houtside.
  apply Hall; [ simpl; rewrite Hlength; reflexivity | ].
  apply distinct_outside_cons; assumption.
Qed.

(* The swaps of two lists of names, one after another. *)

Lemma permute_combine_other (names names' : list positive)
    (name : positive) :
  among name names = false ->
  among name names' = false ->
  permute (combine names names') name = name.
Proof.
  revert names'. induction names as [ | first names IH];
    intros [ | first' names'] Hname Hname'; try reflexivity.
  apply among_cons_false in Hname as [Hfirst Hname].
  apply among_cons_false in Hname' as [Hfirst' Hname'].
  cbn [combine permute]. rewrite (IH names' Hname Hname').
  apply swap_other; assumption.
Qed.

Lemma map_swap_other (a b : positive) (names : list positive) :
  among a names = false -> among b names = false ->
  map (swap a b) names = names.
Proof.
  induction names as [ | first names IH]; [ reflexivity | ].
  intros Ha Hb. apply among_cons_false in Ha as [Ha_first Ha].
  apply among_cons_false in Hb as [Hb_first Hb]. cbn [map].
  rewrite (IH Ha Hb), swap_other; [ reflexivity | | ]; congruence.
Qed.

(* They send the first list to the second, when each list has different
   names and the two have none in common. *)
Lemma permute_combine (names names' : list positive) :
  length names = length names' ->
  distinct names = true ->
  distinct_outside names names' ->
  map (permute (combine names names')) names = names'.
Proof.
  revert names'. induction names as [ | first names IH];
    intros [ | first' names'] Hlength Hdistinct [Hdistinct' Houtside];
    try discriminate; [ reflexivity | ].
  cbn [distinct] in Hdistinct, Hdistinct'.
  apply andb_true_iff in Hdistinct as [Hfirst Hdistinct].
  apply andb_true_iff in Hdistinct' as [Hfirst' Hdistinct'].
  apply negb_true_iff in Hfirst, Hfirst'.
  assert (Hhead : among first' (first :: names) = false)
    by (apply Houtside; rewrite among_cons, Pos.eqb_refl; reflexivity).
  apply among_cons_false in Hhead as [Hhead_first Hhead].
  assert (Htail :
    ∀ name : positive,
      among name names' = true -> among name (first :: names) = false)
    by (intros name Hname; apply Houtside; rewrite among_cons, Hname;
        apply orb_true_r).
  assert (Hfirst_names' : among first names' = false).
  { destruct (among first names') eqn:Hamong; [ | reflexivity ].
    apply Htail, among_cons_false in Hamong as [Hdifferent _].
    contradiction. }
  cbn [combine map permute].
  rewrite (permute_combine_other names names' first Hfirst Hfirst_names'),
    swap_first.
  f_equal.
  rewrite <- (map_map (permute (combine names names')) (swap first first')).
  rewrite IH.
  - apply map_swap_other; assumption.
  - simpl in Hlength. congruence.
  - exact Hdistinct.
  - split; [ exact Hdistinct' | ]. intros name Hname.
    apply Htail, among_cons_false in Hname as [_ Hname]. exact Hname.
Qed.

Lemma fixes_combine (support names names' : list positive) :
  (∀ name : positive, among name names = true -> among name support = false)
  ->
  (∀ name : positive, among name names' = true -> among name support = false)
  ->
  fixes (combine names names') support.
Proof.
  intros Houtside Houtside' name Hname.
  apply permute_combine_other.
  - destruct (among name names) eqn:Hamong; [ | reflexivity ].
    apply Houtside in Hamong. congruence.
  - destruct (among name names') eqn:Hamong; [ | reflexivity ].
    apply Houtside' in Hamong. congruence.
Qed.

(* A predicate that holds of one list of different names outside its
   support holds of every other of the same length: go by a third list
   that has no name of either. *)
Lemma supported_names_move (support : list positive)
    (P : list positive -> Prop) (names names' : list positive) :
  supports list_of_names support P ->
  length names = length names' ->
  distinct_outside support names ->
  distinct_outside (support ++ names) names' ->
  P names -> P names'.
Proof.
  intros Hsupports Hlength [Hdistinct Houtside] Houtside' HP.
  destruct (distinct_outside_app support names names' Houtside')
    as [[Hdistinct' Hsupport'] Hnames'].
  pose proof
    (Hsupports (combine names names') names
       (fixes_combine support names names' Houtside Hsupport') HP)
    as Hmoved.
  simpl in Hmoved.
  rewrite (permute_combine names names' Hlength Hdistinct Hnames') in Hmoved.
  exact Hmoved.
Qed.

Lemma supported_names_some_any (support : list positive)
    (P : list positive -> Prop) (names names' : list positive) :
  supports list_of_names support P ->
  length names = length names' ->
  distinct_outside support names ->
  distinct_outside support names' ->
  P names -> P names'.
Proof.
  intros Hsupports Hlength Houtside Houtside' HP.
  set (between :=
    fresh_names (length names) ((support ++ names) ++ support ++ names')).
  pose proof
    (fresh_names_outside (length names)
       ((support ++ names) ++ support ++ names'))
    as Hbetween.
  apply distinct_outside_app in Hbetween as [Hbetween Hbetween'].
  fold between in Hbetween, Hbetween'.
  assert (Hlength_between : length between = length names)
    by apply fresh_names_length.
  assert (HP_between : P between).
  { apply (supported_names_move support P names between); try assumption.
    symmetry. exact Hlength_between. }
  apply (supported_names_move support P between names'); try assumption.
  - congruence.
  - apply (distinct_outside_app support names between Hbetween).
  - (* The names of [names'] are outside [between], since those of
       [between] are outside [names']. *)
    destruct Houtside' as [Hdistinct' Hsupport']. split; [ exact Hdistinct' | ].
    intros name Hname. apply among_app_false.
    split; [ exact (Hsupport' name Hname) | ].
    destruct (among name between) eqn:Hamong; [ | reflexivity ].
    destruct Hbetween' as [_ Hbetween'].
    apply Hbetween', among_app_false in Hamong as [_ Hamong]. congruence.
Qed.

Theorem some_any_names : some_any_names_statement.
Proof.
  intros count support P Hsupports.
  assert (Hsome :
    (∃ names : list positive,
      length names = count ∧ distinct_outside support names ∧ P names) ->
    И[count] names, P names).
  { intros (witness & Hlength & Houtside & HP). exists support.
    intros names Hlength' Houtside'.
    apply (supported_names_some_any support P witness names);
      solve [ assumption | congruence ]. }
  split; split.
  - intros Hfresh names Hlength Houtside.
    destruct (for_fresh_names_elim count P support Hfresh)
      as (witness & Hlength' & Houtside' & HP).
    apply (supported_names_some_any support P witness names);
      solve [ assumption | congruence ].
  - intros Hall. exists support. exact Hall.
  - apply for_fresh_names_elim.
  - exact Hsome.
Qed.

Theorem for_fresh_names_iterated : for_fresh_names_iterated_statement.
Proof.
  intros count support P Hsupports.
  split; [ apply for_fresh_names_next | ].
  intros Hfresh. apply (some_any_names (S count) support P Hsupports).
  destruct
    (for_fresh_elim (fun name => И[count] names, P (name :: names)) support
       Hfresh)
    as (name & Hname & Hfresh').
  destruct
    (for_fresh_names_elim count (fun names => P (name :: names))
       (name :: support) Hfresh')
    as (names & Hlength & Houtside & HP).
  exists (name :: names).
  split; [ simpl; rewrite Hlength; reflexivity | ].
  split; [ apply distinct_outside_cons; assumption | exact HP ].
Qed.

Theorem some_any_names_nominal : some_any_names_nominal_statement.
Proof.
  intros count X N R x HN Hequivariant Hrespects.
  exact (some_any_names count (free_names N x) (fun names => R (names, x))
           (equivariant_supports (list positive) X list_of_names N R x
              list_of_names_nominal HN Hequivariant Hrespects)).
Qed.
