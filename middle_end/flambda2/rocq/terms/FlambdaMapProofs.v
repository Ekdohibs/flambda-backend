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

(* Lemmas about the finite maps and the sets of names of FlambdaMapSpec.v,
   and the proofs of its statements. Nothing here needs review.

   A lemma about [lookup], [among] or another operation of
   FlambdaMapSpec.v that does not mention the terms or the machine goes
   here, so that there is one place to look.

   Each lemma is proved once, for keys of any type. The lemmas on names
   come after, under the names they have always had, and each is its
   lemma on keys at the equality test of names. *)

From Stdlib Require Import PArith ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec.
Import ListNotations.

(* ======================================================================== *)
(* Keys of any type                                                         *)
(* ======================================================================== *)

(* The lemmas about the module [Keyed] of FlambdaMapSpec.v: the equality
   test of the keys is an argument, and a lemma that depends on it takes
   a proof that it decides equality. A lemma here applies as it stands to
   a goal about the definitions on names, which are those of [Keyed] at
   [Pos.eqb] by computation ([names_are_keys]); give it [Pos.eqb_eq]
   where it asks for the proof, and [Z.eqb_eq] for integers. *)

Theorem pos_eqb_decides_equality : pos_eqb_decides_equality_statement.
Proof. exact Pos.eqb_eq. Qed.

Theorem z_eqb_decides_equality : z_eqb_decides_equality_statement.
Proof. exact Z.eqb_eq. Qed.

Theorem names_are_keys : names_are_keys_statement.
Proof. repeat split. Qed.

Lemma keyed_eqb_refl {K : Type} (eqb : K -> K -> bool) :
  decides_equality eqb -> ∀ key : K, eqb key key = true.
Proof. intros Heqb key. apply Heqb. reflexivity. Qed.

Lemma keyed_eqb_spec {K : Type} (eqb : K -> K -> bool) :
  decides_equality eqb ->
  ∀ key key' : K, reflect (key = key') (eqb key key').
Proof.
  intros Heqb key key'. destruct (eqb key key') eqn:Htest; constructor.
  - apply Heqb, Htest.
  - intros Heq. apply Heqb in Heq. congruence.
Qed.

(* ------------------------------------------------------------------------ *)
(* Sets of keys                                                             *)
(* ------------------------------------------------------------------------ *)

Theorem keyed_among_in : keyed_among_in_statement.
Proof.
  intros K eqb Heqb key keys. unfold Keyed.among.
  rewrite existsb_exists. split.
  - intros [x [Hin Heq]]. apply Heqb in Heq. subst x. exact Hin.
  - intros Hin. exists key.
    split; [ exact Hin | apply keyed_eqb_refl, Heqb ].
Qed.

Lemma keyed_among_cons {K : Type} (eqb : K -> K -> bool) (key first : K)
    (keys : list K) :
  Keyed.among eqb key (first :: keys)
  = eqb key first || Keyed.among eqb key keys.
Proof. reflexivity. Qed.

Theorem keyed_among_app : keyed_among_app_statement.
Proof. intros K eqb key keys keys'. apply existsb_app. Qed.

(* The keys of a filtered list are those the filter keeps. *)
Lemma keyed_among_filter {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (keep : K -> bool) (key : K)
    (keys : list K) :
  Keyed.among eqb key (filter keep keys)
  = Keyed.among eqb key keys && keep key.
Proof.
  unfold Keyed.among.
  induction keys as [ | first keys IH]; simpl; [ reflexivity | ].
  destruct (keep first) eqn:Hkeep; simpl; rewrite IH;
    destruct (keyed_eqb_spec eqb Heqb key first) as [-> | _]; simpl;
    rewrite ?Hkeep, ?andb_false_r; reflexivity.
Qed.

Theorem keyed_among_without : keyed_among_without_statement.
Proof.
  intros K eqb Heqb key removed keys. apply (keyed_among_filter eqb Heqb).
Qed.

Theorem keyed_among_inter : keyed_among_inter_statement.
Proof.
  intros K eqb Heqb key keys keys'. apply (keyed_among_filter eqb Heqb).
Qed.

Theorem keyed_all_among : keyed_all_among_statement.
Proof.
  intros K eqb Heqb keys allowed. unfold Keyed.all_among.
  rewrite forallb_forall. split.
  - intros Hall key Hin. apply Hall, (keyed_among_in K eqb Heqb), Hin.
  - intros Hall key Hin. apply Hall, (keyed_among_in K eqb Heqb), Hin.
Qed.

Lemma keyed_distinct_NoDup {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (keys : list K) :
  Keyed.distinct eqb keys = true <-> NoDup keys.
Proof.
  induction keys as [ | k keys IH]; simpl.
  - split; [ intros _; apply NoDup_nil | reflexivity ].
  - rewrite andb_true_iff, negb_true_iff, IH. split.
    + intros [Hk Hd]. apply NoDup_cons; [ | exact Hd ].
      intros Hin. apply (keyed_among_in K eqb Heqb) in Hin. congruence.
    + intros H. inversion H as [ | ? ? Hk Hd]. subst.
      split; [ | exact Hd ].
      destruct (Keyed.among eqb k keys) eqn:Hamong; [ | reflexivity ].
      apply (keyed_among_in K eqb Heqb) in Hamong. contradiction.
Qed.

(* Filtering keeps a list without repetition so. *)
Lemma keyed_distinct_filter {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (keep : K -> bool) (keys : list K) :
  Keyed.distinct eqb keys = true ->
  Keyed.distinct eqb (filter keep keys) = true.
Proof.
  induction keys as [ | first keys IH]; simpl; [ reflexivity | ].
  intros Hdistinct. apply andb_true_iff in Hdistinct as [Hfirst Hrest].
  destruct (keep first); simpl; [ | exact (IH Hrest) ].
  rewrite (keyed_among_filter eqb Heqb), (IH Hrest).
  apply negb_true_iff in Hfirst. rewrite Hfirst. reflexivity.
Qed.

Theorem keyed_among_dedup : keyed_among_dedup_statement.
Proof.
  intros K eqb Heqb key keys.
  induction keys as [ | first keys IH]; [ reflexivity | ].
  cbn [Keyed.dedup]. rewrite !keyed_among_cons.
  rewrite (keyed_among_without K eqb Heqb), IH.
  cbn [Keyed.among existsb]. destruct (eqb key first); simpl;
    [ reflexivity | apply andb_true_r ].
Qed.

Theorem keyed_dedup_distinct : keyed_dedup_distinct_statement.
Proof.
  intros K eqb Heqb keys.
  induction keys as [ | first keys IH]; [ reflexivity | ].
  cbn [Keyed.dedup Keyed.distinct].
  rewrite (keyed_among_without K eqb Heqb). cbn [Keyed.among existsb].
  rewrite (keyed_eqb_refl eqb Heqb), andb_false_r. simpl.
  apply (keyed_distinct_filter eqb Heqb), IH.
Qed.

Lemma keyed_same_keys_refl {K : Type} (eqb : K -> K -> bool)
    (keys : list K) :
  Keyed.same_keys eqb keys keys.
Proof. intros key. reflexivity. Qed.

Lemma keyed_same_keys_sym {K : Type} (eqb : K -> K -> bool)
    (keys keys' : list K) :
  Keyed.same_keys eqb keys keys' -> Keyed.same_keys eqb keys' keys.
Proof. intros Hsame key. symmetry. apply Hsame. Qed.

Lemma keyed_same_keys_trans {K : Type} (eqb : K -> K -> bool)
    (keys keys' keys'' : list K) :
  Keyed.same_keys eqb keys keys' -> Keyed.same_keys eqb keys' keys'' ->
  Keyed.same_keys eqb keys keys''.
Proof. intros H H' key. rewrite (H key). apply H'. Qed.

Lemma keyed_dedup_same_keys {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (keys : list K) :
  Keyed.same_keys eqb (Keyed.dedup eqb keys) keys.
Proof. intros key. apply (keyed_among_dedup K eqb Heqb). Qed.

Theorem keyed_same_keys_b : keyed_same_keys_b_statement.
Proof.
  intros K eqb Heqb keys keys'. unfold Keyed.same_keys_b.
  rewrite andb_true_iff, !(keyed_all_among K eqb Heqb). split.
  - intros [Hsub Hsub'] key.
    destruct (Keyed.among eqb key keys) eqn:Hkey.
    + symmetry. apply Hsub, Hkey.
    + destruct (Keyed.among eqb key keys') eqn:Hkey'; [ | reflexivity ].
      rewrite (Hsub' key Hkey') in Hkey. discriminate.
  - intros Hsame. split; intros key Hkey.
    + rewrite <- (Hsame key). exact Hkey.
    + rewrite (Hsame key). exact Hkey.
Qed.

Theorem keyed_is_empty_among : keyed_is_empty_among_statement.
Proof.
  intros K eqb Heqb [ | first keys]; split; try reflexivity.
  - discriminate.
  - intros Hnone. specialize (Hnone first).
    rewrite keyed_among_cons, (keyed_eqb_refl eqb Heqb) in Hnone.
    discriminate.
Qed.

Theorem keyed_get_singleton : keyed_get_singleton_statement.
Proof.
  intros K eqb Heqb key keys. unfold Keyed.get_singleton.
  pose proof (keyed_dedup_distinct K eqb Heqb keys) as Hdistinct.
  assert (Hsame :
    ∀ key' : K,
      Keyed.among eqb key' keys
      = Keyed.among eqb key' (Keyed.dedup eqb keys)).
  { intros key'. symmetry. apply (keyed_among_dedup K eqb Heqb). }
  destruct (Keyed.dedup eqb keys) as [ | first [ | second rest]].
  - split; [ discriminate | ]. intros Hkeys. specialize (Hkeys key).
    rewrite Hsame, (keyed_eqb_refl eqb Heqb) in Hkeys. discriminate.
  - split.
    + intros [= ->] key'. rewrite Hsame. simpl. apply orb_false_r.
    + intros Hkeys. specialize (Hkeys first).
      rewrite Hsame, keyed_among_cons, (keyed_eqb_refl eqb Heqb) in Hkeys.
      symmetry in Hkeys. apply Heqb in Hkeys. congruence.
  - split; [ discriminate | ]. intros Hkeys. exfalso.
    assert (Hfirst : first = key).
    { apply Heqb. rewrite <- Hkeys, Hsame, keyed_among_cons.
      rewrite (keyed_eqb_refl eqb Heqb). reflexivity. }
    assert (Hsecond : second = key).
    { apply Heqb. rewrite <- Hkeys, Hsame, !keyed_among_cons.
      rewrite (keyed_eqb_refl eqb Heqb). apply orb_true_r. }
    subst first second. simpl in Hdistinct.
    rewrite (keyed_eqb_refl eqb Heqb) in Hdistinct. discriminate.
Qed.

(* ------------------------------------------------------------------------ *)
(* Lookups                                                                  *)
(* ------------------------------------------------------------------------ *)

Lemma keyed_lookup_cons {K : Type} (eqb : K -> K -> bool) {A : Type}
    (key bound : K) (a : A) (bindings : list (K * A)) :
  Keyed.lookup eqb key ((bound, a) :: bindings)
  = if eqb key bound then Some a else Keyed.lookup eqb key bindings.
Proof. reflexivity. Qed.

(* The binding just added is the one found. *)
Lemma keyed_lookup_head {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K) (a : A)
    (bindings : list (K * A)) :
  Keyed.lookup eqb key ((key, a) :: bindings) = Some a.
Proof. simpl. rewrite (keyed_eqb_refl eqb Heqb). reflexivity. Qed.

(* A binding of another key changes nothing. *)
Lemma keyed_lookup_other {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key bound : K) (a : A)
    (bindings : list (K * A)) :
  key <> bound ->
  Keyed.lookup eqb key ((bound, a) :: bindings)
  = Keyed.lookup eqb key bindings.
Proof.
  intros Hne. simpl.
  destruct (keyed_eqb_spec eqb Heqb key bound);
    [ contradiction | reflexivity ].
Qed.

(* A binding added for a key that was not bound. *)
Lemma keyed_lookup_fresh {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key fresh : K) (a added : A)
    (bindings : list (K * A)) :
  Keyed.lookup eqb fresh bindings = None ->
  Keyed.lookup eqb key bindings = Some a ->
  Keyed.lookup eqb key ((fresh, added) :: bindings) = Some a.
Proof.
  intros Hfresh Hkey. simpl.
  destruct (keyed_eqb_spec eqb Heqb key fresh) as [-> | _];
    [ congruence | exact Hkey ].
Qed.

Lemma keyed_lookup_app {K : Type} (eqb : K -> K -> bool) {A : Type}
    (key : K) (bindings bindings' : list (K * A)) :
  Keyed.lookup eqb key (bindings ++ bindings')
  = match Keyed.lookup eqb key bindings with
    | Some a => Some a
    | None => Keyed.lookup eqb key bindings'
    end.
Proof.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  destruct (eqb key bound); [ reflexivity | exact IH ].
Qed.

(* What is found is bound by one of the pairs, whatever the test. *)
Lemma keyed_lookup_snd_in {K : Type} (eqb : K -> K -> bool) {A : Type}
    (key : K) (a : A) (bindings : list (K * A)) :
  Keyed.lookup eqb key bindings = Some a ->
  ∃ bound : K, (bound, a) ∈ bindings ∧ eqb key bound = true.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl; [ discriminate | ].
  destruct (eqb key bound) eqn:Htest.
  - intros [= ->]. exists bound. split; [ left; reflexivity | exact Htest ].
  - intros Hlookup. destruct (IH Hlookup) as (bound' & Hin & Htest').
    exists bound'. split; [ right; exact Hin | exact Htest' ].
Qed.

(* What is found is one of the bindings. *)
Lemma keyed_lookup_in {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K) (a : A)
    (bindings : list (K * A)) :
  Keyed.lookup eqb key bindings = Some a -> (key, a) ∈ bindings.
Proof.
  intros Hlookup.
  destruct (keyed_lookup_snd_in eqb key a bindings Hlookup)
    as (bound & Hin & Htest).
  apply Heqb in Htest. subst bound. exact Hin.
Qed.

Lemma keyed_lookup_none_not_in {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K)
    (bindings : list (K * A)) :
  Keyed.lookup eqb key bindings = None -> key ∉ map fst bindings.
Proof.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ auto | ].
  destruct (keyed_eqb_spec eqb Heqb key bound) as [-> | Hne];
    [ discriminate | ].
  intros Hlookup [Heq | Hin]; [ congruence | exact (IH Hlookup Hin) ].
Qed.

(* What holds of every binding holds of the one found. *)
Lemma keyed_lookup_forallb {K : Type} (eqb : K -> K -> bool) {A : Type}
    (f : A -> bool) (key : K) (a : A) (bindings : list (K * A)) :
  forallb (fun '(_, a) => f a) bindings = true ->
  Keyed.lookup eqb key bindings = Some a -> f a = true.
Proof.
  induction bindings as [ | [bound b] bindings IH]; simpl; [ discriminate | ].
  intros Hall. apply andb_true_iff in Hall as [Hb Hrest].
  destruct (eqb key bound); [ intros [= <-]; exact Hb | exact (IH Hrest) ].
Qed.

(* A key is bound when it is among the keys of the bindings. *)
Lemma keyed_is_bound_among {K : Type} (eqb : K -> K -> bool) {A : Type}
    (key : K) (bindings : Keyed.finite_map K A) :
  Keyed.is_bound eqb key bindings = Keyed.among eqb key (map fst bindings).
Proof.
  unfold Keyed.is_bound, Keyed.among.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  destruct (eqb key bound); [ reflexivity | exact IH ].
Qed.

Lemma keyed_is_bound_lookup {K : Type} (eqb : K -> K -> bool) {A : Type}
    (key : K) (bindings : Keyed.finite_map K A) :
  Keyed.is_bound eqb key bindings = true
  <-> ∃ a : A, Keyed.lookup eqb key bindings = Some a.
Proof.
  unfold Keyed.is_bound.
  destruct (Keyed.lookup eqb key bindings) as [a | ]; split.
  - intros _. exists a. reflexivity.
  - reflexivity.
  - discriminate.
  - intros [a [=]].
Qed.

(* ------------------------------------------------------------------------ *)
(* The operations on maps                                                   *)
(* ------------------------------------------------------------------------ *)

Theorem keyed_lookup_unbind : keyed_lookup_unbind_statement.
Proof.
  intros K eqb Heqb A key removed bindings. unfold Keyed.unbind.
  induction bindings as [ | [bound a] bindings IH]; simpl.
  - destruct (eqb key removed); reflexivity.
  - destruct (keyed_eqb_spec eqb Heqb removed bound) as [<- | Hne]; simpl;
      rewrite IH.
    + destruct (eqb key removed); reflexivity.
    + destruct (keyed_eqb_spec eqb Heqb key bound) as [-> | _];
        [ | reflexivity ].
      destruct (keyed_eqb_spec eqb Heqb bound removed);
        [ congruence | reflexivity ].
Qed.

Theorem keyed_lookup_map_values : keyed_lookup_map_values_statement.
Proof.
  intros K eqb A B f key bindings. unfold Keyed.map_values.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  destruct (eqb key bound); [ reflexivity | exact IH ].
Qed.

Theorem keyed_among_domain : keyed_among_domain_statement.
Proof.
  intros K eqb Heqb A key bindings. unfold Keyed.domain.
  rewrite (keyed_among_dedup K eqb Heqb), keyed_is_bound_among.
  reflexivity.
Qed.

Theorem keyed_domain_distinct : keyed_domain_distinct_statement.
Proof.
  intros K eqb Heqb A bindings. apply (keyed_dedup_distinct K eqb Heqb).
Qed.

Theorem keyed_lookup_tabulate : keyed_lookup_tabulate_statement.
Proof.
  intros K eqb Heqb A f key keys. unfold Keyed.tabulate, Keyed.among.
  induction keys as [ | first keys IH]; simpl; [ reflexivity | ].
  destruct (f first) as [a | ] eqn:Hfirst; simpl;
    destruct (keyed_eqb_spec eqb Heqb key first) as [-> | _]; simpl;
    try exact IH.
  - symmetry. exact Hfirst.
  - rewrite IH, Hfirst.
    destruct (existsb (eqb first) keys); reflexivity.
Qed.

(* The keys a tabulated map binds are those of the list for which the
   function gives something, in the order of the list. *)
Lemma keyed_tabulate_keys {K A : Type} (f : K -> option A) (keys : list K) :
  map fst (Keyed.tabulate f keys)
  = filter
      (fun key =>
         match f key with
         | Some _ => true
         | None => false
         end)
      keys.
Proof.
  unfold Keyed.tabulate.
  induction keys as [ | first keys IH]; simpl; [ reflexivity | ].
  destruct (f first); simpl; rewrite IH; reflexivity.
Qed.

Lemma keyed_tabulate_distinct {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (f : K -> option A)
    (keys : list K) :
  Keyed.distinct eqb keys = true ->
  Keyed.distinct eqb (map fst (Keyed.tabulate f keys)) = true.
Proof.
  rewrite keyed_tabulate_keys. apply (keyed_distinct_filter eqb Heqb).
Qed.

Theorem keyed_lookup_compact : keyed_lookup_compact_statement.
Proof.
  intros K eqb Heqb A key bindings. unfold Keyed.compact.
  rewrite (keyed_lookup_tabulate K eqb Heqb).
  rewrite (keyed_among_domain K eqb Heqb). unfold Keyed.is_bound.
  destruct (Keyed.lookup eqb key bindings); reflexivity.
Qed.

Theorem keyed_compact_distinct : keyed_compact_distinct_statement.
Proof.
  intros K eqb Heqb A bindings.
  apply (keyed_tabulate_distinct eqb Heqb).
  apply (keyed_domain_distinct K eqb Heqb).
Qed.

(* The bindings of the compact map are the ones [lookup] finds. *)
Lemma keyed_in_compact {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K) (a : A)
    (bindings : Keyed.finite_map K A) :
  (key, a) ∈ Keyed.compact eqb bindings
  <-> Keyed.lookup eqb key bindings = Some a.
Proof.
  split.
  - unfold Keyed.compact, Keyed.tabulate. rewrite in_flat_map.
    intros (bound & _ & Hin).
    destruct (Keyed.lookup eqb bound bindings) as [b | ] eqn:Hlookup;
      [ | contradiction ].
    destruct Hin as [[= -> ->] | []]. exact Hlookup.
  - intros Hlookup. apply (keyed_lookup_in eqb Heqb).
    rewrite (keyed_lookup_compact K eqb Heqb). exact Hlookup.
Qed.

Theorem keyed_lookup_filter_bindings :
  keyed_lookup_filter_bindings_statement.
Proof.
  intros K eqb Heqb A keep key bindings. unfold Keyed.filter_bindings.
  rewrite (keyed_lookup_tabulate K eqb Heqb).
  rewrite (keyed_among_domain K eqb Heqb). unfold Keyed.is_bound.
  destruct (Keyed.lookup eqb key bindings); reflexivity.
Qed.

Theorem keyed_lookup_merge : keyed_lookup_merge_statement.
Proof.
  intros K eqb Heqb A B C f key bindings bindings' Hnone.
  unfold Keyed.merge.
  rewrite (keyed_lookup_tabulate K eqb Heqb).
  rewrite (keyed_among_dedup K eqb Heqb), keyed_among_app.
  rewrite <- !keyed_is_bound_among. unfold Keyed.is_bound.
  destruct (Keyed.lookup eqb key bindings),
    (Keyed.lookup eqb key bindings'); simpl; try reflexivity.
  symmetry. exact Hnone.
Qed.

Theorem keyed_lookup_union_with : keyed_lookup_union_with_statement.
Proof.
  intros K eqb Heqb A f key bindings bindings'. unfold Keyed.union_with.
  rewrite (keyed_lookup_merge K eqb Heqb); reflexivity.
Qed.

Theorem keyed_lookup_inter_with : keyed_lookup_inter_with_statement.
Proof.
  intros K eqb Heqb A B C f key bindings bindings'.
  unfold Keyed.inter_with.
  rewrite (keyed_lookup_merge K eqb Heqb); reflexivity.
Qed.

Theorem keyed_is_empty_lookup : keyed_is_empty_lookup_statement.
Proof.
  intros K eqb Heqb A [ | [bound a] bindings]; simpl; split;
    try reflexivity.
  - discriminate.
  - intros Hnone. specialize (Hnone bound).
    rewrite (keyed_eqb_refl eqb Heqb) in Hnone. discriminate.
Qed.

Theorem keyed_get_singleton_binding : keyed_get_singleton_binding_statement.
Proof.
  intros K eqb Heqb A key a bindings. unfold Keyed.get_singleton_binding.
  pose proof (keyed_compact_distinct K eqb Heqb A bindings) as Hdistinct.
  assert (Hsame :
    ∀ key' : K,
      Keyed.lookup eqb key' bindings
      = Keyed.lookup eqb key' (Keyed.compact eqb bindings)).
  { intros key'. symmetry. apply (keyed_lookup_compact K eqb Heqb). }
  destruct (Keyed.compact eqb bindings)
    as [ | [first b] [ | [second b'] rest]].
  - split; [ discriminate | ]. intros Hmap. specialize (Hmap key).
    rewrite Hsame, (keyed_eqb_refl eqb Heqb) in Hmap. discriminate.
  - split.
    + intros [= -> ->] key'. rewrite Hsame. reflexivity.
    + intros Hmap. specialize (Hmap first).
      rewrite Hsame, keyed_lookup_cons, (keyed_eqb_refl eqb Heqb) in Hmap.
      destruct (keyed_eqb_spec eqb Heqb first key) as [-> | _];
        [ | discriminate ].
      injection Hmap as ->. reflexivity.
  - split; [ discriminate | ]. intros Hmap. exfalso.
    assert (Hfirst : first = key).
    { pose proof (Hmap first) as Hat.
      rewrite Hsame, keyed_lookup_cons, (keyed_eqb_refl eqb Heqb) in Hat.
      destruct (keyed_eqb_spec eqb Heqb first key);
        [ assumption | discriminate ]. }
    assert (Hsecond : second = key).
    { pose proof (Hmap second) as Hat.
      rewrite Hsame, !keyed_lookup_cons, (keyed_eqb_refl eqb Heqb) in Hat.
      destruct (keyed_eqb_spec eqb Heqb second key); [ assumption | ].
      destruct (eqb second first); discriminate. }
    subst first second. simpl in Hdistinct.
    rewrite (keyed_eqb_refl eqb Heqb) in Hdistinct. discriminate.
Qed.

(* ------------------------------------------------------------------------ *)
(* Maps that are the same, and maps that are included                       *)
(* ------------------------------------------------------------------------ *)

Lemma keyed_map_equal_refl {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings.
Proof. intros key. reflexivity. Qed.

Lemma keyed_map_equal_sym {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  Keyed.map_equal eqb bindings' bindings.
Proof. intros Hequal key. symmetry. apply Hequal. Qed.

Lemma keyed_map_equal_trans {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings bindings' bindings'' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  Keyed.map_equal eqb bindings' bindings'' ->
  Keyed.map_equal eqb bindings bindings''.
Proof. intros H H' key. rewrite (H key). apply H'. Qed.

Lemma keyed_compact_map_equal {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type}
    (bindings : Keyed.finite_map K A) :
  Keyed.map_equal eqb (Keyed.compact eqb bindings) bindings.
Proof. intros key. apply (keyed_lookup_compact K eqb Heqb). Qed.

Lemma keyed_map_included_refl {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings : Keyed.finite_map K A) :
  Keyed.map_included eqb bindings bindings.
Proof. intros key a Hlookup. exact Hlookup. Qed.

Lemma keyed_map_included_trans {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings bindings' bindings'' : Keyed.finite_map K A) :
  Keyed.map_included eqb bindings bindings' ->
  Keyed.map_included eqb bindings' bindings'' ->
  Keyed.map_included eqb bindings bindings''.
Proof. intros H H' key a Hlookup. apply H', H, Hlookup. Qed.

(* Two maps are the same when each is included in the other. *)
Lemma keyed_map_equal_included {K : Type} (eqb : K -> K -> bool) {A : Type}
    (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings'
  <-> Keyed.map_included eqb bindings bindings'
      ∧ Keyed.map_included eqb bindings' bindings.
Proof.
  split.
  - intros Hequal. split; intros key a Hlookup.
    + rewrite <- (Hequal key). exact Hlookup.
    + rewrite (Hequal key). exact Hlookup.
  - intros [Hincluded Hincluded'] key.
    destruct (Keyed.lookup eqb key bindings) as [a | ] eqn:Hlookup.
    + symmetry. exact (Hincluded key a Hlookup).
    + destruct (Keyed.lookup eqb key bindings') as [a' | ] eqn:Hlookup';
        [ | reflexivity ].
      rewrite (Hincluded' key a' Hlookup') in Hlookup. discriminate.
Qed.

(* A binding added for a key that was not bound includes what was. *)
Lemma keyed_map_included_fresh {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (fresh : K) (added : A)
    (bindings : Keyed.finite_map K A) :
  Keyed.lookup eqb fresh bindings = None ->
  Keyed.map_included eqb bindings ((fresh, added) :: bindings).
Proof.
  intros Hfresh key a Hlookup.
  exact (keyed_lookup_fresh eqb Heqb key fresh a added bindings Hfresh
           Hlookup).
Qed.

(* The operations give the same map on maps that are the same. *)
Lemma keyed_unbind_map_equal {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (key : K)
    (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  Keyed.map_equal eqb (Keyed.unbind eqb key bindings)
    (Keyed.unbind eqb key bindings').
Proof.
  intros Hequal other.
  rewrite !(keyed_lookup_unbind K eqb Heqb), (Hequal other). reflexivity.
Qed.

Lemma keyed_map_values_map_equal {K : Type} (eqb : K -> K -> bool)
    {A B : Type} (f : A -> B) (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  Keyed.map_equal eqb (Keyed.map_values f bindings)
    (Keyed.map_values f bindings').
Proof.
  intros Hequal key.
  rewrite !keyed_lookup_map_values, (Hequal key). reflexivity.
Qed.

Lemma keyed_merge_map_equal {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A B C : Type}
    (f : option A -> option B -> option C)
    (left left' : Keyed.finite_map K A)
    (right right' : Keyed.finite_map K B) :
  f None None = None ->
  Keyed.map_equal eqb left left' -> Keyed.map_equal eqb right right' ->
  Keyed.map_equal eqb (Keyed.merge eqb f left right)
    (Keyed.merge eqb f left' right').
Proof.
  intros Hnone Hleft Hright key.
  rewrite !(keyed_lookup_merge K eqb Heqb), (Hleft key), (Hright key);
    auto.
Qed.

(* ------------------------------------------------------------------------ *)
(* Going through a list or a map                                            *)
(* ------------------------------------------------------------------------ *)

Theorem every_in : every_in_statement.
Proof.
  intros A P l. induction l as [ | first l IH]; simpl.
  - split; [ intros _ a [] | intros _; exact I ].
  - rewrite IH. split.
    + intros [Hfirst Hrest] a [<- | Hin];
        [ exact Hfirst | exact (Hrest a Hin) ].
    + intros Hall. split.
      * apply Hall. left. reflexivity.
      * intros a Hin. apply Hall. right. exact Hin.
Qed.

(* What follows from [P] on the elements of the list holds of every one
   if [P] does. *)
Lemma every_impl {A : Type} (P Q : A -> Prop) (l : list A) :
  (∀ a : A, a ∈ l -> P a -> Q a) -> every P l -> every Q l.
Proof.
  rewrite !(every_in A). intros Himpl HP a Hin.
  exact (Himpl a Hin (HP a Hin)).
Qed.

Theorem every_b_every : every_b_every_statement.
Proof.
  intros A p l. induction l as [ | first l IH]; simpl.
  - split; [ intros _; exact I | reflexivity ].
  - rewrite andb_true_iff, IH. reflexivity.
Qed.

(* A test that agrees with [P] on the elements of the list. *)
Lemma every_b_reflects {A : Type} (P : A -> Prop) (p : A -> bool)
    (l : list A) :
  (∀ a : A, a ∈ l -> (p a = true <-> P a)) ->
  (every_b p l = true <-> every P l).
Proof.
  intros Hreflects. rewrite (every_b_every A), !(every_in A). split.
  - intros Hall a Hin. apply (Hreflects a Hin), Hall, Hin.
  - intros Hall a Hin. apply (Hreflects a Hin), Hall, Hin.
Qed.

Theorem keyed_for_all_bindings_lookup :
  keyed_for_all_bindings_lookup_statement.
Proof.
  intros K eqb Heqb A P bindings. unfold Keyed.for_all_bindings.
  rewrite (every_in (K * A)%type). split.
  - intros Hall key a Hlookup.
    exact (Hall (key, a) (keyed_lookup_in eqb Heqb key a bindings Hlookup)
             Hlookup).
  - intros Hall [key a] _ Hlookup. exact (Hall key a Hlookup).
Qed.

Theorem keyed_at_binding_lookup : keyed_at_binding_lookup_statement.
Proof.
  intros K eqb A P key bindings. unfold Keyed.at_binding.
  rewrite (every_in (K * A)%type). split.
  - intros Hall a Hlookup.
    destruct (keyed_lookup_snd_in eqb key a bindings Hlookup)
      as (bound & Hin & _).
    exact (Hall (bound, a) Hin Hlookup).
  - intros Hall [bound a] _ Hlookup. exact (Hall a Hlookup).
Qed.

Theorem for_all_bindings_lookup : for_all_bindings_lookup_statement.
Proof.
  exact (keyed_for_all_bindings_lookup positive Pos.eqb Pos.eqb_eq).
Qed.

Theorem at_binding_lookup : at_binding_lookup_statement.
Proof. exact (keyed_at_binding_lookup positive Pos.eqb). Qed.

(* What follows from [P] on the bindings that count holds of every one if
   [P] does. *)
Lemma keyed_for_all_bindings_impl {K : Type} (eqb : K -> K -> bool)
    {A : Type} (P Q : K -> A -> Prop) (bindings : Keyed.finite_map K A) :
  (∀ (key : K) (a : A),
     Keyed.lookup eqb key bindings = Some a -> P key a -> Q key a) ->
  Keyed.for_all_bindings eqb P bindings ->
  Keyed.for_all_bindings eqb Q bindings.
Proof.
  intros Himpl. unfold Keyed.for_all_bindings. apply every_impl.
  intros [key a] _ HP Hlookup. exact (Himpl key a Hlookup (HP Hlookup)).
Qed.

Lemma keyed_at_binding_impl {K : Type} (eqb : K -> K -> bool) {A : Type}
    (P Q : A -> Prop) (key : K) (bindings : Keyed.finite_map K A) :
  (∀ a : A, Keyed.lookup eqb key bindings = Some a -> P a -> Q a) ->
  Keyed.at_binding eqb P key bindings ->
  Keyed.at_binding eqb Q key bindings.
Proof.
  intros Himpl. unfold Keyed.at_binding. apply every_impl.
  intros [bound a] _ HP Hlookup. exact (Himpl a Hlookup (HP Hlookup)).
Qed.

(* The two say the same of maps that are the same. *)
Lemma keyed_for_all_bindings_map_equal {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (P : K -> A -> Prop)
    (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  (Keyed.for_all_bindings eqb P bindings
   <-> Keyed.for_all_bindings eqb P bindings').
Proof.
  intros Hequal. rewrite !(keyed_for_all_bindings_lookup K eqb Heqb).
  split; intros Hall key a Hlookup; apply Hall.
  - rewrite (Hequal key). exact Hlookup.
  - rewrite <- (Hequal key). exact Hlookup.
Qed.

Lemma keyed_at_binding_map_equal {K : Type} (eqb : K -> K -> bool)
    {A : Type} (P : A -> Prop) (key : K)
    (bindings bindings' : Keyed.finite_map K A) :
  Keyed.map_equal eqb bindings bindings' ->
  (Keyed.at_binding eqb P key bindings
   <-> Keyed.at_binding eqb P key bindings').
Proof.
  intros Hequal. rewrite !(keyed_at_binding_lookup K eqb), (Hequal key).
  reflexivity.
Qed.

(* The test of the binding of a key is the test of what [lookup] finds. *)
Lemma keyed_at_binding_b_lookup {K : Type} (eqb : K -> K -> bool)
    {A : Type} (p : A -> bool) (key : K)
    (bindings : Keyed.finite_map K A) :
  Keyed.at_binding_b eqb p key bindings
  = match Keyed.lookup eqb key bindings with
    | Some a => p a
    | None => true
    end.
Proof.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  destruct (eqb key bound); [ reflexivity | exact IH ].
Qed.

Theorem keyed_at_binding_b : keyed_at_binding_b_statement.
Proof.
  intros K eqb A p key bindings.
  rewrite (keyed_at_binding_lookup K eqb), keyed_at_binding_b_lookup.
  destruct (Keyed.lookup eqb key bindings) as [a | ]; split.
  - intros Hp b [= <-]. exact Hp.
  - intros Hall. apply Hall. reflexivity.
  - intros _ b [=].
  - reflexivity.
Qed.

Theorem keyed_for_all_bindings_b : keyed_for_all_bindings_b_statement.
Proof.
  intros K eqb Heqb A p bindings.
  rewrite (keyed_for_all_bindings_lookup K eqb Heqb).
  unfold Keyed.for_all_bindings_b.
  match goal with
  | |- ?go [] bindings = true <-> _ =>
    assert (Hgo :
      ∀ (bindings : list (K * A)) (seen : list K),
        go seen bindings = true
        <-> (∀ (key : K) (a : A),
              Keyed.lookup eqb key bindings = Some a ->
              Keyed.among eqb key seen = false -> p key a = true))
  end.
  { clear bindings.
    induction bindings as [ | [bound b] bindings IH]; intros seen; simpl.
    - split; [ intros _ key a [=] | reflexivity ].
    - rewrite andb_true_iff, orb_true_iff, IH. split.
      + intros [Hbound Hrest] key a.
        destruct (keyed_eqb_spec eqb Heqb key bound) as [-> | Hne].
        * intros [= <-] Hunseen. destruct Hbound; congruence.
        * intros Hlookup Hunseen. apply (Hrest key a Hlookup).
          rewrite keyed_among_cons, Hunseen.
          destruct (keyed_eqb_spec eqb Heqb key bound);
            [ contradiction | reflexivity ].
      + intros Hall. split.
        * destruct (Keyed.among eqb bound seen) eqn:Hseen;
            [ left; reflexivity | right ].
          apply Hall; [ | exact Hseen ].
          rewrite (keyed_eqb_refl eqb Heqb). reflexivity.
        * intros key a Hlookup Hunseen.
          rewrite keyed_among_cons in Hunseen.
          apply orb_false_iff in Hunseen as [Hne Hunseen].
          apply Hall; [ | exact Hunseen ]. rewrite Hne. exact Hlookup. }
  rewrite Hgo. split.
  - intros Hall key a Hlookup. apply (Hall key a Hlookup). reflexivity.
  - intros Hall key a Hlookup _. exact (Hall key a Hlookup).
Qed.

(* A test that agrees with [P] on the bindings that count. *)
Lemma keyed_for_all_bindings_b_reflects {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {A : Type} (P : K -> A -> Prop)
    (p : K -> A -> bool) (bindings : Keyed.finite_map K A) :
  (∀ (key : K) (a : A),
     Keyed.lookup eqb key bindings = Some a -> (p key a = true <-> P key a)) ->
  (Keyed.for_all_bindings_b eqb p bindings = true
   <-> Keyed.for_all_bindings eqb P bindings).
Proof.
  intros Hreflects. rewrite (keyed_for_all_bindings_b K eqb Heqb).
  split; apply keyed_for_all_bindings_impl; intros key a Hlookup;
    apply (Hreflects key a Hlookup).
Qed.

Lemma keyed_at_binding_b_reflects {K : Type} (eqb : K -> K -> bool)
    {A : Type} (P : A -> Prop) (p : A -> bool) (key : K)
    (bindings : Keyed.finite_map K A) :
  (∀ a : A, Keyed.lookup eqb key bindings = Some a -> (p a = true <-> P a)) ->
  (Keyed.at_binding_b eqb p key bindings = true
   <-> Keyed.at_binding eqb P key bindings).
Proof.
  intros Hreflects. rewrite (keyed_at_binding_b K eqb).
  split; apply keyed_at_binding_impl; intros a Hlookup;
    apply (Hreflects a Hlookup).
Qed.

(* ------------------------------------------------------------------------ *)
(* Changing the keys                                                        *)
(* ------------------------------------------------------------------------ *)

(* An [f] that sends different keys to different keys keeps the test. *)
Lemma keyed_eqb_injective {K K' : Type} (eqb : K -> K -> bool)
    (eqb' : K' -> K' -> bool) (Heqb : decides_equality eqb)
    (Heqb' : decides_equality eqb') (f : K -> K') :
  (∀ key key' : K, f key = f key' -> key = key') ->
  ∀ key key' : K, eqb' (f key) (f key') = eqb key key'.
Proof.
  intros Hf key key'.
  destruct (keyed_eqb_spec eqb Heqb key key') as [-> | Hne].
  - apply (keyed_eqb_refl eqb' Heqb').
  - destruct (keyed_eqb_spec eqb' Heqb' (f key) (f key')) as [Heq | _];
      [ | reflexivity ].
    elim Hne. exact (Hf _ _ Heq).
Qed.

Lemma keyed_map_bindings_keys {K K' A B : Type} (f : K -> K') (g : A -> B)
    (bindings : Keyed.finite_map K A) :
  map fst (Keyed.map_bindings f g bindings) = map f (map fst bindings).
Proof.
  unfold Keyed.map_bindings.
  induction bindings as [ | [bound a] bindings IH]; simpl;
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

Theorem keyed_lookup_map_bindings : keyed_lookup_map_bindings_statement.
Proof.
  intros K K' eqb eqb' Heqb Heqb' A B f g Hf key bindings.
  unfold Keyed.map_bindings.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  rewrite (keyed_eqb_injective eqb eqb' Heqb Heqb' f Hf).
  destruct (eqb key bound); [ reflexivity | exact IH ].
Qed.

Theorem keyed_lookup_map_bindings_off_image :
  keyed_lookup_map_bindings_off_image_statement.
Proof.
  intros K K' eqb' Heqb' A B f g key' bindings Hoff.
  unfold Keyed.map_bindings.
  induction bindings as [ | [bound a] bindings IH]; simpl; [ reflexivity | ].
  destruct (keyed_eqb_spec eqb' Heqb' key' (f bound)) as [Heq | _];
    [ | exact IH ].
  elim (Hoff bound). symmetry. exact Heq.
Qed.

Theorem keyed_lookup_map_bindings_inverse :
  keyed_lookup_map_bindings_inverse_statement.
Proof.
  intros K K' eqb eqb' Heqb Heqb' A B f f_inverse g Hinverse key' bindings.
  unfold Keyed.map_bindings.
  induction bindings as [ | [bound a] bindings IH]; simpl.
  - destruct (eqb' (f (f_inverse key')) key'); reflexivity.
  - destruct (keyed_eqb_spec eqb' Heqb' key' (f bound)) as [-> | Hne].
    + rewrite Hinverse, (keyed_eqb_refl eqb' Heqb'),
        (keyed_eqb_refl eqb Heqb).
      reflexivity.
    + rewrite IH.
      destruct (keyed_eqb_spec eqb' Heqb' (f (f_inverse key')) key')
        as [Himage | _];
        [ | reflexivity ].
      destruct (keyed_eqb_spec eqb Heqb (f_inverse key') bound)
        as [Hbound | _];
        [ | reflexivity ].
      elim Hne. rewrite <- Hbound. symmetry. exact Himage.
Qed.

Theorem keyed_among_map : keyed_among_map_statement.
Proof.
  intros K K' eqb eqb' Heqb Heqb' f Hf key keys. unfold Keyed.among.
  induction keys as [ | first keys IH]; simpl; [ reflexivity | ].
  rewrite (keyed_eqb_injective eqb eqb' Heqb Heqb' f Hf), IH. reflexivity.
Qed.

(* Whether a key is [f] of one of the keys of a list. *)
Lemma keyed_among_map_in {K K' : Type} (eqb' : K' -> K' -> bool)
    (Heqb' : decides_equality eqb') (f : K -> K') (key' : K')
    (keys : list K) :
  Keyed.among eqb' key' (map f keys) = true
  <-> ∃ key : K, key ∈ keys ∧ f key = key'.
Proof.
  split.
  - intros H. apply (keyed_among_in K' eqb' Heqb') in H.
    apply in_map_iff in H. destruct H as (key & Hkey & Hin).
    exists key. split; assumption.
  - intros (key & Hin & Hkey). apply (keyed_among_in K' eqb' Heqb').
    apply in_map_iff. exists key. split; assumption.
Qed.

Theorem keyed_domain_map_bindings : keyed_domain_map_bindings_statement.
Proof.
  intros K K' eqb eqb' Heqb Heqb' A B f g bindings key'.
  unfold Keyed.domain.
  rewrite (keyed_among_dedup K' eqb' Heqb'), keyed_map_bindings_keys.
  apply eq_true_iff_eq. split; intros H.
  - apply (keyed_among_map_in eqb' Heqb') in H.
    destruct H as (key & Hin & Hkey).
    apply (keyed_among_map_in eqb' Heqb'). exists key.
    split; [ | exact Hkey ].
    apply (keyed_among_in K eqb Heqb).
    rewrite (keyed_among_dedup K eqb Heqb).
    apply (keyed_among_in K eqb Heqb). exact Hin.
  - apply (keyed_among_map_in eqb' Heqb') in H.
    destruct H as (key & Hin & Hkey).
    apply (keyed_among_map_in eqb' Heqb'). exists key.
    split; [ | exact Hkey ].
    apply (keyed_among_in K eqb Heqb).
    rewrite <- (keyed_among_dedup K eqb Heqb).
    apply (keyed_among_in K eqb Heqb). exact Hin.
Qed.

Theorem keyed_map_bindings_map_equal :
  keyed_map_bindings_map_equal_statement.
Proof.
  intros K K' eqb eqb' Heqb Heqb' A B f g Hf bindings bindings' Hequal key'.
  assert (Himage :
            ∀ (m : Keyed.finite_map K A) (b : B),
              Keyed.lookup eqb' key' (Keyed.map_bindings f g m) = Some b ->
              ∃ key : K, f key = key').
  { intros m b Hlookup. apply (keyed_lookup_in eqb' Heqb') in Hlookup.
    apply (in_map fst) in Hlookup.
    rewrite keyed_map_bindings_keys in Hlookup. simpl in Hlookup.
    apply in_map_iff in Hlookup. destruct Hlookup as (key & Hkey & _).
    exists key. exact Hkey. }
  assert (Hsame :
            ∀ key : K,
              Keyed.lookup eqb' (f key) (Keyed.map_bindings f g bindings)
              = Keyed.lookup eqb' (f key) (Keyed.map_bindings f g bindings')).
  { intros key.
    rewrite !(keyed_lookup_map_bindings K K' eqb eqb' Heqb Heqb' A B f g Hf).
    rewrite (Hequal key). reflexivity. }
  destruct (Keyed.lookup eqb' key' (Keyed.map_bindings f g bindings))
    as [b | ] eqn:Hleft.
  - destruct (Himage bindings b Hleft) as (key & <-).
    rewrite <- Hsame. symmetry. exact Hleft.
  - destruct (Keyed.lookup eqb' key' (Keyed.map_bindings f g bindings'))
      as [b' | ] eqn:Hright;
      [ | reflexivity ].
    destruct (Himage bindings' b' Hright) as (key & <-).
    rewrite Hsame in Hleft. congruence.
Qed.

(* ------------------------------------------------------------------------ *)
(* Folds that thread a state                                                *)
(* ------------------------------------------------------------------------ *)

Lemma keyed_fold2_on_nil {K : Type} (eqb : K -> K -> bool) {A B S : Type}
    (f : K -> option A -> option B -> S -> S) (m1 : Keyed.finite_map K A)
    (m2 : Keyed.finite_map K B) (init : S) :
  Keyed.fold2_on eqb [] f m1 m2 init = init.
Proof. reflexivity. Qed.

Lemma keyed_fold2_on_cons {K : Type} (eqb : K -> K -> bool) {A B S : Type}
    (f : K -> option A -> option B -> S -> S) (key : K) (keys : list K)
    (m1 : Keyed.finite_map K A) (m2 : Keyed.finite_map K B) (init : S) :
  Keyed.fold2_on eqb (key :: keys) f m1 m2 init
  = Keyed.fold2_on eqb keys f m1 m2
      (f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2) init).
Proof. reflexivity. Qed.

Theorem keyed_among_keys_of_both : keyed_among_keys_of_both_statement.
Proof.
  intros K eqb Heqb A B key m1 m2. unfold Keyed.keys_of_both.
  rewrite (keyed_among_dedup K eqb Heqb), keyed_among_app,
    !keyed_is_bound_among.
  reflexivity.
Qed.

Theorem keyed_keys_of_both_distinct : keyed_keys_of_both_distinct_statement.
Proof.
  intros K eqb Heqb A B m1 m2. apply (keyed_dedup_distinct K eqb Heqb).
Qed.

(* One step of [tabulate_fold], by projections. *)
Lemma keyed_tabulate_fold_cons {K A S : Type} (f : K -> S -> option A * S)
    (key : K) (keys : list K) (s : S) :
  Keyed.tabulate_fold f (key :: keys) s
  = (match fst (f key s) with
     | Some a => (key, a) :: fst (Keyed.tabulate_fold f keys (snd (f key s)))
     | None => fst (Keyed.tabulate_fold f keys (snd (f key s)))
     end,
     snd (Keyed.tabulate_fold f keys (snd (f key s)))).
Proof.
  simpl. destruct (f key s) as [a s']. simpl.
  destruct (Keyed.tabulate_fold f keys s') as [bindings s'']. reflexivity.
Qed.

Theorem keyed_tabulate_fold_stateless :
  keyed_tabulate_fold_stateless_statement.
Proof.
  intros K A S f keys s. induction keys as [ | key keys IH];
    [ reflexivity | ].
  rewrite keyed_tabulate_fold_cons. cbn [fst snd]. rewrite IH.
  cbn [fst snd]. unfold Keyed.tabulate. simpl.
  destruct (f key); reflexivity.
Qed.

(* A fold over two lists of keys, one after the other. *)
Lemma keyed_tabulate_fold_app {K A S : Type} (f : K -> S -> option A * S)
    (keys keys' : list K) :
  ∀ s : S,
    Keyed.tabulate_fold f (keys ++ keys') s
    = (fst (Keyed.tabulate_fold f keys s)
       ++ fst
            (Keyed.tabulate_fold f keys'
               (snd (Keyed.tabulate_fold f keys s))),
       snd
         (Keyed.tabulate_fold f keys' (snd (Keyed.tabulate_fold f keys s)))).
Proof.
  induction keys as [ | key keys IH]; intros s.
  - simpl. destruct (Keyed.tabulate_fold f keys' s); reflexivity.
  - change ((key :: keys) ++ keys') with (key :: (keys ++ keys')).
    rewrite !keyed_tabulate_fold_cons, IH. cbn [fst snd].
    destruct (fst (f key s)); reflexivity.
Qed.

(* The map binds none but the keys given. *)
Lemma keyed_tabulate_fold_keys {K A S : Type} (f : K -> S -> option A * S)
    (keys : list K) :
  ∀ (s : S) (key : K),
    key ∈ map fst (fst (Keyed.tabulate_fold f keys s)) -> key ∈ keys.
Proof.
  induction keys as [ | first keys IH]; intros s key.
  - simpl. exact (fun H => H).
  - rewrite keyed_tabulate_fold_cons. cbn [fst snd].
    destruct (fst (f first s)) as [a | ]; simpl.
    + intros [-> | Hin]; [ left; reflexivity | right; exact (IH _ _ Hin) ].
    + intros Hin. right. exact (IH _ _ Hin).
Qed.

Theorem keyed_lookup_tabulate_fold_other :
  keyed_lookup_tabulate_fold_other_statement.
Proof.
  intros K eqb Heqb A S f key keys s Hamong.
  destruct (Keyed.lookup eqb key (fst (Keyed.tabulate_fold f keys s)))
    as [a | ] eqn:Hlookup;
    [ | reflexivity ].
  apply (keyed_lookup_in eqb Heqb) in Hlookup.
  apply (in_map fst) in Hlookup. simpl in Hlookup.
  apply keyed_tabulate_fold_keys in Hlookup.
  apply (keyed_among_in K eqb Heqb) in Hlookup. congruence.
Qed.

(* A key of a list of distinct keys is not among those before it nor
   among those after it. *)
Lemma keyed_distinct_app_cons {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (before : list K) (key : K)
    (after : list K) :
  Keyed.distinct eqb (before ++ key :: after) = true ->
  Keyed.among eqb key before = false ∧ Keyed.among eqb key after = false.
Proof.
  intros H. apply (keyed_distinct_NoDup eqb Heqb) in H.
  apply NoDup_remove_2 in H.
  split.
  - destruct (Keyed.among eqb key before) eqn:Hamong; [ | reflexivity ].
    elim H. apply in_or_app. left.
    apply (keyed_among_in K eqb Heqb). exact Hamong.
  - destruct (Keyed.among eqb key after) eqn:Hamong; [ | reflexivity ].
    elim H. apply in_or_app. right.
    apply (keyed_among_in K eqb Heqb). exact Hamong.
Qed.

Theorem keyed_lookup_tabulate_fold : keyed_lookup_tabulate_fold_statement.
Proof.
  intros K eqb Heqb A S f before key after s Hdistinct.
  destruct (keyed_distinct_app_cons eqb Heqb before key after Hdistinct)
    as [Hbefore Hafter].
  rewrite keyed_tabulate_fold_app. cbn [fst snd].
  rewrite keyed_lookup_app,
    (keyed_lookup_tabulate_fold_other K eqb Heqb A S f key before s Hbefore).
  rewrite keyed_tabulate_fold_cons. cbn [fst snd].
  destruct (fst (f key (snd (Keyed.tabulate_fold f before s)))) as [a | ].
  - apply (keyed_lookup_head eqb Heqb).
  - apply (keyed_lookup_tabulate_fold_other K eqb Heqb). exact Hafter.
Qed.

Theorem keyed_merge_fold_on : keyed_merge_fold_on_statement.
Proof.
  intros K eqb A B C S keys f m1 m2 init. unfold Keyed.merge_fold_on.
  enough
    (H :
       ∀ (keys : list K) (acc : Keyed.finite_map K C) (s : S),
         List.fold_left
           (fun (acc : Keyed.finite_map K C * S) key =>
              let '(c, s) :=
                f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2)
                  (snd acc)
              in
              (match c with
               | Some c => fst acc ++ [(key, c)]
               | None => fst acc
               end,
               s))
           keys (acc, s)
         = (acc
            ++ fst
                 (Keyed.tabulate_fold
                    (fun key =>
                       f key (Keyed.lookup eqb key m1)
                         (Keyed.lookup eqb key m2))
                    keys s),
            snd
              (Keyed.tabulate_fold
                 (fun key =>
                    f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2))
                 keys s))).
  { refine (eq_trans (H keys [] init) _). simpl.
    destruct
      (Keyed.tabulate_fold
         (fun key =>
            f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2))
         keys init);
      reflexivity. }
  clear keys init. induction keys as [ | key keys IH]; intros acc s.
  - simpl. rewrite app_nil_r. reflexivity.
  - cbn [List.fold_left]. rewrite keyed_tabulate_fold_cons. cbn [fst snd].
    destruct (f key (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2) s)
      as [c s'].
    cbn [fst snd]. rewrite IH.
    destruct c as [c | ]; [ | reflexivity ].
    rewrite <- app_assoc. reflexivity.
Qed.

Theorem keyed_merge_fold_stateless : keyed_merge_fold_stateless_statement.
Proof.
  intros K eqb A B C S f m1 m2 init. unfold Keyed.merge_fold.
  rewrite keyed_merge_fold_on.
  exact
    (keyed_tabulate_fold_stateless K C S
       (fun key => f (Keyed.lookup eqb key m1) (Keyed.lookup eqb key m2))
       (Keyed.keys_of_both eqb m1 m2) init).
Qed.

Theorem keyed_tabulate_fold_opt_total :
  keyed_tabulate_fold_opt_total_statement.
Proof.
  intros K A S f keys. induction keys as [ | key keys IH]; intros s;
    [ reflexivity | ].
  simpl. destruct (f key s) as [a s']. rewrite IH.
  destruct (Keyed.tabulate_fold f keys s') as [bindings s'']. reflexivity.
Qed.

(* Two functions that answer the same give the same fold. *)
Lemma keyed_tabulate_fold_opt_ext {K A S : Type}
    (f f' : K -> S -> option (option A * S)) (keys : list K) :
  (∀ (key : K) (s : S), f key s = f' key s) ->
  ∀ s : S, Keyed.tabulate_fold_opt f keys s = Keyed.tabulate_fold_opt f' keys s.
Proof.
  intros Hext. induction keys as [ | key keys IH]; intros s; simpl;
    [ reflexivity | ].
  rewrite <- Hext. destruct (f key s) as [[a s'] | ]; [ | reflexivity ].
  rewrite IH. reflexivity.
Qed.

Theorem keyed_tabulate_fold_opt_invariant :
  keyed_tabulate_fold_opt_invariant_statement.
Proof.
  intros K A S f P Q keys.
  induction keys as [ | key keys IH]; intros s s' bindings Hstep Hs; simpl.
  - intros [= <- <-].
    split; [ exact Hs | split; [ exact I | intros key [] ] ].
  - destruct (f key s) as [[a s1] | ] eqn:Hf; [ | discriminate ].
    destruct (Keyed.tabulate_fold_opt f keys s1) as [[rest s2] | ] eqn:Hrest;
      [ | discriminate ].
    intros [= <- <-].
    destruct (Hstep key s s1 a (or_introl eq_refl) Hs Hf) as [Hs1 Ha].
    destruct
      (IH s1 s2 rest
         (fun key' t t' a' Hin => Hstep key' t t' a' (or_intror Hin)) Hs1
         Hrest)
      as (Hs2 & Hrest_q & Hkeys).
    split; [ exact Hs2 | ].
    destruct a as [a | ]; simpl.
    + split; [ split; [ exact (Ha a eq_refl) | exact Hrest_q ] | ].
      intros key' [<- | Hin];
        [ left; reflexivity | right; exact (Hkeys key' Hin) ].
    + split; [ exact Hrest_q | ].
      intros key' Hin. right. exact (Hkeys key' Hin).
Qed.

(* The same of a fold that cannot fail. *)
Lemma keyed_tabulate_fold_invariant {K A S : Type}
    (f : K -> S -> option A * S) (P : S -> Prop) (Q : K -> A -> Prop)
    (keys : list K) (s : S) :
  (∀ (key : K) (s : S),
     key ∈ keys ->
     P s ->
     P (snd (f key s)) ∧ ∀ a : A, fst (f key s) = Some a -> Q key a) ->
  P s ->
  P (snd (Keyed.tabulate_fold f keys s))
  ∧ every (fun binding => Q (fst binding) (snd binding))
      (fst (Keyed.tabulate_fold f keys s)).
Proof.
  intros Hstep Hs.
  destruct
    (keyed_tabulate_fold_opt_invariant K A S (fun key s => Some (f key s)) P
       Q keys s
       (snd (Keyed.tabulate_fold f keys s))
       (fst (Keyed.tabulate_fold f keys s)))
    as (H1 & H2 & _).
  - intros key t t' a Hin Ht [= Hf].
    destruct (Hstep key t Hin Ht) as [Hp Hq]. rewrite Hf in Hp, Hq.
    split; [ exact Hp | exact Hq ].
  - exact Hs.
  - rewrite keyed_tabulate_fold_opt_total.
    destruct (Keyed.tabulate_fold f keys s); reflexivity.
  - exact (conj H1 H2).
Qed.

Theorem keyed_filter_map_fold_stateless :
  keyed_filter_map_fold_stateless_statement.
Proof.
  intros K eqb A B S f bindings s.
  unfold Keyed.filter_map_fold, Keyed.filter_map_fold_on.
  pose
    (h :=
       fun key =>
         match Keyed.lookup eqb key bindings with
         | Some a => f key a
         | None => None
         end).
  refine
    (eq_trans
       (keyed_tabulate_fold_opt_ext _ (fun key s => Some (h key, s)) _ _ _) _).
  - intros key s0. unfold h.
    destruct (Keyed.lookup eqb key bindings); reflexivity.
  - refine
      (eq_trans
         (keyed_tabulate_fold_opt_total K B S (fun key s => (h key, s)) _ _)
         _).
    exact (f_equal Some (keyed_tabulate_fold_stateless K B S h _ _)).
Qed.

(* ======================================================================== *)
(* Names                                                                    *)
(* ======================================================================== *)

(* What is above, for the definitions on names, which the machine and all
   that is written with it use. Each definition on names is the one of
   [Keyed] at [Pos.eqb] by computation, so each proof here is the lemma
   on keys, at [Pos.eqb] and with [Pos.eqb_eq] where it asks for a proof
   that the test decides equality. A lemma on names is added here only
   beside its lemma on keys above. *)

(* ------------------------------------------------------------------------ *)
(* Sets of names                                                            *)
(* ------------------------------------------------------------------------ *)

Lemma among_In (name : positive) (names : list positive) :
  among name names = true <-> name ∈ names.
Proof. exact (keyed_among_in positive Pos.eqb Pos.eqb_eq name names). Qed.

Lemma among_cons (name first : positive) (names : list positive) :
  among name (first :: names) = Pos.eqb name first || among name names.
Proof. exact (keyed_among_cons Pos.eqb name first names). Qed.

Lemma among_app (name : positive) (names names' : list positive) :
  among name (names ++ names') = among name names || among name names'.
Proof. exact (keyed_among_app positive Pos.eqb name names names'). Qed.

(* The names of a filtered list are those the filter keeps. *)
Lemma among_filter (keep : positive -> bool) (name : positive)
    (names : list positive) :
  among name (filter keep names) = among name names && keep name.
Proof. exact (keyed_among_filter Pos.eqb Pos.eqb_eq keep name names). Qed.

Theorem among_without : among_without_statement.
Proof. exact (keyed_among_without positive Pos.eqb Pos.eqb_eq). Qed.

Theorem among_inter : among_inter_statement.
Proof. exact (keyed_among_inter positive Pos.eqb Pos.eqb_eq). Qed.

Theorem all_among : all_among_statement.
Proof. exact (keyed_all_among positive Pos.eqb Pos.eqb_eq). Qed.

Lemma distinct_NoDup (names : list positive) :
  distinct names = true <-> NoDup names.
Proof. exact (keyed_distinct_NoDup Pos.eqb Pos.eqb_eq names). Qed.

(* Filtering keeps a list without repetition so. *)
Lemma distinct_filter (keep : positive -> bool) (names : list positive) :
  distinct names = true -> distinct (filter keep names) = true.
Proof. exact (keyed_distinct_filter Pos.eqb Pos.eqb_eq keep names). Qed.

Theorem among_dedup : among_dedup_statement.
Proof. exact (keyed_among_dedup positive Pos.eqb Pos.eqb_eq). Qed.

Theorem dedup_distinct : dedup_distinct_statement.
Proof. exact (keyed_dedup_distinct positive Pos.eqb Pos.eqb_eq). Qed.

Lemma same_names_refl (names : list positive) : same_names names names.
Proof. exact (keyed_same_keys_refl Pos.eqb names). Qed.

Lemma same_names_sym (names names' : list positive) :
  same_names names names' -> same_names names' names.
Proof. exact (keyed_same_keys_sym Pos.eqb names names'). Qed.

Lemma same_names_trans (names names' names'' : list positive) :
  same_names names names' -> same_names names' names'' ->
  same_names names names''.
Proof. exact (keyed_same_keys_trans Pos.eqb names names' names''). Qed.

Lemma dedup_same_names (names : list positive) :
  same_names (dedup names) names.
Proof. exact (keyed_dedup_same_keys Pos.eqb Pos.eqb_eq names). Qed.

(* ------------------------------------------------------------------------ *)
(* Lookups                                                                  *)
(* ------------------------------------------------------------------------ *)

Lemma lookup_cons {A : Type} (name bound : positive) (a : A)
    (bindings : list (positive * A)) :
  lookup name ((bound, a) :: bindings)
  = if Pos.eqb name bound then Some a else lookup name bindings.
Proof. exact (keyed_lookup_cons Pos.eqb name bound a bindings). Qed.

(* The binding just added is the one found. *)
Lemma lookup_head {A : Type} (name : positive) (a : A)
    (bindings : list (positive * A)) :
  lookup name ((name, a) :: bindings) = Some a.
Proof. exact (keyed_lookup_head Pos.eqb Pos.eqb_eq name a bindings). Qed.

(* A binding of another name changes nothing. *)
Lemma lookup_other {A : Type} (name bound : positive) (a : A)
    (bindings : list (positive * A)) :
  name <> bound -> lookup name ((bound, a) :: bindings) = lookup name bindings.
Proof.
  exact (keyed_lookup_other Pos.eqb Pos.eqb_eq name bound a bindings).
Qed.

(* A binding added for a name that was not bound. *)
Lemma lookup_fresh {A : Type} (name fresh : positive) (a added : A)
    (bindings : list (positive * A)) :
  lookup fresh bindings = None ->
  lookup name bindings = Some a ->
  lookup name ((fresh, added) :: bindings) = Some a.
Proof.
  exact (keyed_lookup_fresh Pos.eqb Pos.eqb_eq name fresh a added bindings).
Qed.

Lemma lookup_app {A : Type} (name : positive)
    (bindings bindings' : list (positive * A)) :
  lookup name (bindings ++ bindings')
  = match lookup name bindings with
    | Some a => Some a
    | None => lookup name bindings'
    end.
Proof. exact (keyed_lookup_app Pos.eqb name bindings bindings'). Qed.

(* What is found is one of the bindings. *)
Lemma lookup_in {A : Type} (name : positive) (a : A)
    (bindings : list (positive * A)) :
  lookup name bindings = Some a -> (name, a) ∈ bindings.
Proof. exact (keyed_lookup_in Pos.eqb Pos.eqb_eq name a bindings). Qed.

Lemma lookup_none_not_in {A : Type} (name : positive)
    (bindings : list (positive * A)) :
  lookup name bindings = None -> name ∉ map fst bindings.
Proof. exact (keyed_lookup_none_not_in Pos.eqb Pos.eqb_eq name bindings). Qed.

(* What holds of every binding holds of the one found. *)
Lemma lookup_forallb {A : Type} (f : A -> bool) (name : positive) (a : A)
    (bindings : list (positive * A)) :
  forallb (fun '(_, a) => f a) bindings = true ->
  lookup name bindings = Some a -> f a = true.
Proof. exact (keyed_lookup_forallb Pos.eqb f name a bindings). Qed.

(* A name is bound when it is among the names of the bindings. *)
Lemma is_bound_among {A : Type} (name : positive) (bindings : finite_map A) :
  is_bound name bindings = among name (map fst bindings).
Proof. exact (keyed_is_bound_among Pos.eqb name bindings). Qed.

Lemma is_bound_lookup {A : Type} (name : positive) (bindings : finite_map A) :
  is_bound name bindings = true <-> ∃ a : A, lookup name bindings = Some a.
Proof. exact (keyed_is_bound_lookup Pos.eqb name bindings). Qed.

(* ------------------------------------------------------------------------ *)
(* The operations on maps                                                   *)
(* ------------------------------------------------------------------------ *)

Theorem lookup_unbind : lookup_unbind_statement.
Proof. exact (keyed_lookup_unbind positive Pos.eqb Pos.eqb_eq). Qed.

Theorem lookup_map_values : lookup_map_values_statement.
Proof. exact (keyed_lookup_map_values positive Pos.eqb). Qed.

Theorem among_domain : among_domain_statement.
Proof. exact (keyed_among_domain positive Pos.eqb Pos.eqb_eq). Qed.

Theorem domain_distinct : domain_distinct_statement.
Proof. exact (keyed_domain_distinct positive Pos.eqb Pos.eqb_eq). Qed.

Theorem lookup_tabulate : lookup_tabulate_statement.
Proof. exact (keyed_lookup_tabulate positive Pos.eqb Pos.eqb_eq). Qed.

(* The names a tabulated map binds are those of the list for which the
   function gives something, in the order of the list. *)
Lemma tabulate_names {A : Type} (f : positive -> option A)
    (names : list positive) :
  map fst (tabulate f names)
  = filter
      (fun name =>
         match f name with
         | Some _ => true
         | None => false
         end)
      names.
Proof. exact (keyed_tabulate_keys f names). Qed.

Lemma tabulate_distinct {A : Type} (f : positive -> option A)
    (names : list positive) :
  distinct names = true -> distinct (map fst (tabulate f names)) = true.
Proof. exact (keyed_tabulate_distinct Pos.eqb Pos.eqb_eq f names). Qed.

Theorem lookup_compact : lookup_compact_statement.
Proof. exact (keyed_lookup_compact positive Pos.eqb Pos.eqb_eq). Qed.

Theorem compact_distinct : compact_distinct_statement.
Proof. exact (keyed_compact_distinct positive Pos.eqb Pos.eqb_eq). Qed.

(* The bindings of the compact map are the ones [lookup] finds. *)
Lemma in_compact {A : Type} (name : positive) (a : A)
    (bindings : finite_map A) :
  (name, a) ∈ compact bindings <-> lookup name bindings = Some a.
Proof. exact (keyed_in_compact Pos.eqb Pos.eqb_eq name a bindings). Qed.

Theorem lookup_filter_bindings : lookup_filter_bindings_statement.
Proof. exact (keyed_lookup_filter_bindings positive Pos.eqb Pos.eqb_eq). Qed.

Theorem lookup_merge : lookup_merge_statement.
Proof. exact (keyed_lookup_merge positive Pos.eqb Pos.eqb_eq). Qed.

Theorem lookup_union_with : lookup_union_with_statement.
Proof. exact (keyed_lookup_union_with positive Pos.eqb Pos.eqb_eq). Qed.

Theorem lookup_inter_with : lookup_inter_with_statement.
Proof. exact (keyed_lookup_inter_with positive Pos.eqb Pos.eqb_eq). Qed.

(* ------------------------------------------------------------------------ *)
(* Maps that are the same, and maps that are included                       *)
(* ------------------------------------------------------------------------ *)

Lemma map_equal_refl {A : Type} (bindings : finite_map A) :
  map_equal bindings bindings.
Proof. exact (keyed_map_equal_refl Pos.eqb bindings). Qed.

Lemma map_equal_sym {A : Type} (bindings bindings' : finite_map A) :
  map_equal bindings bindings' -> map_equal bindings' bindings.
Proof. exact (keyed_map_equal_sym Pos.eqb bindings bindings'). Qed.

Lemma map_equal_trans {A : Type}
    (bindings bindings' bindings'' : finite_map A) :
  map_equal bindings bindings' -> map_equal bindings' bindings'' ->
  map_equal bindings bindings''.
Proof.
  exact (keyed_map_equal_trans Pos.eqb bindings bindings' bindings'').
Qed.

Lemma compact_map_equal {A : Type} (bindings : finite_map A) :
  map_equal (compact bindings) bindings.
Proof. exact (keyed_compact_map_equal Pos.eqb Pos.eqb_eq bindings). Qed.

Lemma map_included_refl {A : Type} (bindings : finite_map A) :
  map_included bindings bindings.
Proof. exact (keyed_map_included_refl Pos.eqb bindings). Qed.

Lemma map_included_trans {A : Type}
    (bindings bindings' bindings'' : finite_map A) :
  map_included bindings bindings' -> map_included bindings' bindings'' ->
  map_included bindings bindings''.
Proof.
  exact (keyed_map_included_trans Pos.eqb bindings bindings' bindings'').
Qed.

(* Two maps are the same when each is included in the other. *)
Lemma map_equal_included {A : Type} (bindings bindings' : finite_map A) :
  map_equal bindings bindings'
  <-> map_included bindings bindings' ∧ map_included bindings' bindings.
Proof. exact (keyed_map_equal_included Pos.eqb bindings bindings'). Qed.

(* A binding added for a name that was not bound includes what was. *)
Lemma map_included_fresh {A : Type} (fresh : positive) (added : A)
    (bindings : finite_map A) :
  lookup fresh bindings = None ->
  map_included bindings ((fresh, added) :: bindings).
Proof.
  exact (keyed_map_included_fresh Pos.eqb Pos.eqb_eq fresh added bindings).
Qed.

(* The operations give the same map on maps that are the same. *)
Lemma unbind_map_equal {A : Type} (name : positive)
    (bindings bindings' : finite_map A) :
  map_equal bindings bindings' ->
  map_equal (unbind name bindings) (unbind name bindings').
Proof.
  exact (keyed_unbind_map_equal Pos.eqb Pos.eqb_eq name bindings bindings').
Qed.

Lemma map_values_map_equal {A B : Type} (f : A -> B)
    (bindings bindings' : finite_map A) :
  map_equal bindings bindings' ->
  map_equal (map_values f bindings) (map_values f bindings').
Proof. exact (keyed_map_values_map_equal Pos.eqb f bindings bindings'). Qed.

Lemma merge_map_equal {A B C : Type} (f : option A -> option B -> option C)
    (left left' : finite_map A) (right right' : finite_map B) :
  f None None = None ->
  map_equal left left' -> map_equal right right' ->
  map_equal (merge f left right) (merge f left' right').
Proof.
  exact (keyed_merge_map_equal Pos.eqb Pos.eqb_eq f left left' right right').
Qed.
