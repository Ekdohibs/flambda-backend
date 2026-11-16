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

(* Proofs of the statements of FlambdaEnvAliasesInvariantSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesInvariantSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Going through a map                                                      *)
(* ======================================================================== *)

Lemma for_all_bindings_b_lookup {K : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) {B : Type} (p : K -> B -> bool)
    (bindings : Keyed.finite_map K B) :
  Keyed.for_all_bindings_b eqb p bindings = true
  <-> (∀ (key : K) (b : B),
        Keyed.lookup eqb key bindings = Some b -> p key b = true).
Proof.
  rewrite (keyed_for_all_bindings_b K eqb Heqb),
    (keyed_for_all_bindings_lookup K eqb Heqb).
  reflexivity.
Qed.

(* What is listed for a canonical element, by the map that lists it. *)
Lemma listed_alias_cases (t : Aliases.t) (canonical_element : simple)
    (n : name) :
  listed_alias t canonical_element n ->
  (∃ (canonical_name : name)
     (aliases : Aliases.Aliases_of_canonical_element.t),
      canonical_element = simple_of_name canonical_name
      ∧ Keyed.lookup name_eqb canonical_name
          (Aliases.aliases_of_canonical_names t)
        = Some aliases
      ∧ Aliases.Aliases_of_canonical_element.mem aliases n = true)
  ∨ (∃ (canonical_const : const)
       (aliases : Aliases.Aliases_of_canonical_element.t),
        canonical_element = Const canonical_const
        ∧ Keyed.lookup const_eqb canonical_const (Aliases.aliases_of_consts t)
          = Some aliases
        ∧ Aliases.Aliases_of_canonical_element.mem aliases n = true).
Proof.
  unfold listed_alias. intros Hlisted.
  destruct (name_of_simple canonical_element) as [canonical_name | ] eqn:Hname.
  - apply name_of_simple_some in Hname. subst canonical_element.
    rewrite get_aliases_of_name in Hlisted.
    destruct
      (Keyed.lookup name_eqb canonical_name
         (Aliases.aliases_of_canonical_names t))
      as [aliases | ] eqn:Hlookup;
      [ | discriminate ].
    left. exists canonical_name, aliases. auto.
  - apply name_of_simple_none in Hname as [canonical_const ->].
    rewrite get_aliases_of_const in Hlisted.
    destruct
      (Keyed.lookup const_eqb canonical_const (Aliases.aliases_of_consts t))
      as [aliases | ] eqn:Hlookup;
      [ | discriminate ].
    right. exists canonical_const, aliases. auto.
Qed.

Lemma listed_alias_of_name (t : Aliases.t) (canonical_name n : name)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  Keyed.lookup name_eqb canonical_name (Aliases.aliases_of_canonical_names t)
  = Some aliases ->
  Aliases.Aliases_of_canonical_element.mem aliases n = true ->
  listed_alias t (simple_of_name canonical_name) n.
Proof.
  intros Hlookup Hmem. unfold listed_alias.
  rewrite get_aliases_of_name, Hlookup. exact Hmem.
Qed.

Lemma listed_alias_of_const (t : Aliases.t) (canonical_const : const)
    (n : name) (aliases : Aliases.Aliases_of_canonical_element.t) :
  Keyed.lookup const_eqb canonical_const (Aliases.aliases_of_consts t)
  = Some aliases ->
  Aliases.Aliases_of_canonical_element.mem aliases n = true ->
  listed_alias t (Const canonical_const) n.
Proof.
  intros Hlookup Hmem. unfold listed_alias.
  rewrite get_aliases_of_const, Hlookup. exact Hmem.
Qed.

Lemma in_all_mem (aliases : Aliases.Aliases_of_canonical_element.t)
    (n : name) :
  n ∈ Aliases.Aliases_of_canonical_element.all aliases
  <-> Aliases.Aliases_of_canonical_element.mem aliases n = true.
Proof.
  rewrite <- (keyed_among_in name name_eqb name_eqb_eq), among_all.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The three maps agree                                                     *)
(* ======================================================================== *)

Theorem demoted_to_b_iff : demoted_to_b_iff_statement.
Proof.
  intros t n canonical_element. unfold demoted_to_b, demoted_to.
  destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t))
    as [canonical_element' | ].
  - rewrite (simple_equal_eq canonical_element' canonical_element).
    split; [ intros ->; reflexivity | intros [= ->]; reflexivity ].
  - split; discriminate.
Qed.

Theorem is_canonical_b_iff : is_canonical_b_iff_statement.
Proof.
  intros t e. unfold is_canonical_b, is_canonical.
  destruct (Aliases.canonical t e); split;
    solve [ reflexivity | discriminate ].
Qed.

Lemma canonical_elements_canonical_b_iff (t : Aliases.t) :
  canonical_elements_canonical_b t = true <-> canonical_elements_canonical t.
Proof.
  unfold canonical_elements_canonical_b, canonical_elements_canonical,
    demoted_to.
  rewrite (for_all_bindings_b_lookup name_eqb name_eqb_eq). split.
  - intros Hall n canonical_element Hlookup.
    apply is_canonical_b_iff, (Hall n canonical_element Hlookup).
  - intros Hall n canonical_element Hlookup.
    apply is_canonical_b_iff, (Hall n canonical_element Hlookup).
Qed.

Lemma listed_aliases_demoted_b_iff (t : Aliases.t)
    (canonical_element : simple)
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  listed_aliases_demoted_b t canonical_element aliases = true
  <-> (∀ n : name,
        Aliases.Aliases_of_canonical_element.mem aliases n = true ->
        demoted_to t n canonical_element).
Proof.
  unfold listed_aliases_demoted_b. rewrite forallb_forall. split.
  - intros Hall n Hmem. apply demoted_to_b_iff, Hall, in_all_mem, Hmem.
  - intros Hall n Hin. apply demoted_to_b_iff, Hall, in_all_mem, Hin.
Qed.

Lemma aliases_listed_b_iff (t : Aliases.t) :
  aliases_listed_b t = true <-> aliases_listed t.
Proof.
  unfold aliases_listed_b, aliases_listed.
  rewrite !andb_true_iff, !(for_all_bindings_b_lookup name_eqb name_eqb_eq),
    (for_all_bindings_b_lookup const_eqb const_eqb_eq).
  split.
  - intros [[Hdemoted Hnames] Hconsts] n canonical_element. split.
    + intros Hd. apply Hdemoted, Hd.
    + intros Hlisted.
      apply listed_alias_cases in Hlisted
        as [(canonical_name & aliases & -> & Hlookup & Hmem)
           | (canonical_const & aliases & -> & Hlookup & Hmem)].
      * apply (listed_aliases_demoted_b_iff t _ aliases);
          [ apply Hnames, Hlookup | exact Hmem ].
      * apply (listed_aliases_demoted_b_iff t _ aliases);
          [ apply Hconsts, Hlookup | exact Hmem ].
  - intros Hlisted. repeat split.
    + intros n canonical_element Hlookup.
      apply (Hlisted n canonical_element), Hlookup.
    + intros canonical_name aliases Hlookup.
      apply listed_aliases_demoted_b_iff. intros n Hmem.
      apply Hlisted, (listed_alias_of_name t _ _ aliases); assumption.
    + intros canonical_const aliases Hlookup.
      apply listed_aliases_demoted_b_iff. intros n Hmem.
      apply Hlisted, (listed_alias_of_const t _ _ aliases); assumption.
Qed.

Theorem aliases_consistent_b_iff : aliases_consistent_b_iff_statement.
Proof.
  intros t. unfold aliases_consistent_b, aliases_consistent.
  rewrite andb_true_iff, canonical_elements_canonical_b_iff,
    aliases_listed_b_iff.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The order and the name modes                                             *)
(* ======================================================================== *)

Theorem canonical_elements_earliest_b_iff :
  canonical_elements_earliest_b_iff_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t.
  unfold canonical_elements_earliest_b, canonical_elements_earliest,
    demoted_to.
  apply (for_all_bindings_b_lookup name_eqb name_eqb_eq).
Qed.

Lemma name_modes_recorded_of_b_iff {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (aliases : Aliases.Aliases_of_canonical_element.t) :
  name_modes_recorded_of_b binding_times_and_modes aliases = true
  <-> (∀ (n : name) (name_mode : Name_mode.t),
        Keyed.lookup name_eqb n aliases = Some name_mode ->
        name_mode
        = Aliases.compute_name_mode_unscoped binding_times_and_modes
            (simple_of_name n)).
Proof.
  unfold name_modes_recorded_of_b.
  rewrite (for_all_bindings_b_lookup name_eqb name_eqb_eq). split.
  - intros Hall n name_mode Hlookup.
    apply name_mode_equal_eq, Hall, Hlookup.
  - intros Hall n name_mode Hlookup.
    apply name_mode_equal_eq, Hall, Hlookup.
Qed.

Theorem name_modes_recorded_b_iff : name_modes_recorded_b_iff_statement.
Proof.
  intros A binding_times_and_modes t.
  unfold name_modes_recorded_b, name_modes_recorded.
  rewrite andb_true_iff, (for_all_bindings_b_lookup name_eqb name_eqb_eq),
    (for_all_bindings_b_lookup const_eqb const_eqb_eq).
  split.
  - intros [Hnames Hconsts] canonical_element n name_mode Hlookup.
    destruct (name_of_simple canonical_element) as [canonical_name | ]
      eqn:Hname.
    + apply name_of_simple_some in Hname. subst canonical_element.
      rewrite get_aliases_of_name in Hlookup.
      destruct
        (Keyed.lookup name_eqb canonical_name
           (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Haliases;
        [ | discriminate ].
      exact
        (proj1 (name_modes_recorded_of_b_iff binding_times_and_modes aliases)
           (Hnames canonical_name aliases Haliases) n name_mode Hlookup).
    + apply name_of_simple_none in Hname as [canonical_const ->].
      rewrite get_aliases_of_const in Hlookup.
      destruct
        (Keyed.lookup const_eqb canonical_const (Aliases.aliases_of_consts t))
        as [aliases | ] eqn:Haliases;
        [ | discriminate ].
      exact
        (proj1 (name_modes_recorded_of_b_iff binding_times_and_modes aliases)
           (Hconsts canonical_const aliases Haliases) n name_mode Hlookup).
  - intros Hrecorded. split.
    + intros canonical_name aliases Haliases.
      apply name_modes_recorded_of_b_iff. intros n name_mode Hlookup.
      apply (Hrecorded (simple_of_name canonical_name)).
      rewrite get_aliases_of_name, Haliases. exact Hlookup.
    + intros canonical_const aliases Haliases.
      apply name_modes_recorded_of_b_iff. intros n name_mode Hlookup.
      apply (Hrecorded (Const canonical_const)).
      rewrite get_aliases_of_const, Haliases. exact Hlookup.
Qed.

Theorem aliases_well_formed_b_iff : aliases_well_formed_b_iff_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t.
  unfold aliases_well_formed_b, aliases_well_formed.
  rewrite !andb_true_iff, (aliases_consistent_b_iff t),
    (canonical_elements_earliest_b_iff A binding_time_resolver
       binding_times_and_modes t),
    (name_modes_recorded_b_iff A binding_times_and_modes t).
  tauto.
Qed.

(* ======================================================================== *)
(* The compiler's checks                                                    *)
(* ======================================================================== *)

Lemma NoDup_map_injective {B C : Type} (f : B -> C) (l : list B) :
  (∀ b b' : B, f b = f b' -> b = b') -> NoDup l -> NoDup (map f l).
Proof.
  intros Hinjective Hdistinct.
  induction Hdistinct as [ | b l Hnotin Hdistinct IH]; simpl;
    [ apply NoDup_nil | ].
  apply NoDup_cons; [ | exact IH ].
  intros Hin. apply in_map_iff in Hin as (b' & Heq & Hin).
  apply Hinjective in Heq. subst b'. contradiction.
Qed.

Lemma NoDup_app_apart {B : Type} (l l' : list B) :
  NoDup l -> NoDup l' -> (∀ b : B, b ∈ l -> ~ b ∈ l') -> NoDup (l ++ l').
Proof.
  intros Hdistinct Hdistinct' Hapart.
  induction Hdistinct as [ | b l Hnotin Hdistinct IH]; simpl;
    [ exact Hdistinct' | ].
  apply NoDup_cons.
  - intros Hin. apply in_app_or in Hin as [Hin | Hin]; [ contradiction | ].
    apply (Hapart b); [ left; reflexivity | exact Hin ].
  - apply IH. intros b' Hin. apply Hapart. right. exact Hin.
Qed.

(* Sets of aliases, each given with its canonical element, are disjoint
   when the canonical elements are different and each name of a set has
   been demoted to the canonical element of the set. *)
Lemma disjoint_alias_sets_of (t : Aliases.t)
    (sets : list (simple * list name)) :
  NoDup (map fst sets) ->
  (∀ (canonical_element : simple) (names : list name) (n : name),
    (canonical_element, names) ∈ sets ->
    n ∈ names ->
    demoted_to t n canonical_element) ->
  Aliases.disjoint_alias_sets (map snd sets) = true.
Proof.
  induction sets as [ | [canonical_element names] sets IH];
    intros Hdistinct Hdemoted; simpl; [ reflexivity | ].
  apply andb_true_iff. split.
  - apply forallb_forall. intros names' Hnames'.
    apply in_map_iff in Hnames' as ([canonical_element' names''] & Heq & Hin).
    simpl in Heq. subst names''.
    apply (keyed_is_empty_among name name_eqb name_eqb_eq). intros n.
    rewrite (keyed_among_inter name name_eqb name_eqb_eq).
    destruct (Keyed.among name_eqb n names) eqn:Hn; [ | reflexivity ].
    destruct (Keyed.among name_eqb n names') eqn:Hn'; [ | reflexivity ].
    exfalso.
    apply (keyed_among_in name name_eqb name_eqb_eq) in Hn, Hn'.
    assert (Hfirst : demoted_to t n canonical_element).
    { apply (Hdemoted canonical_element names);
        [ left; reflexivity | exact Hn ]. }
    assert (Hsecond : demoted_to t n canonical_element').
    { apply (Hdemoted canonical_element' names');
        [ right; exact Hin | exact Hn' ]. }
    unfold demoted_to in Hfirst, Hsecond. rewrite Hfirst in Hsecond.
    injection Hsecond as <-.
    inversion Hdistinct as [ | ? ? Hnotin _]. subst. apply Hnotin.
    apply in_map_iff. exists (canonical_element, names').
    split; [ reflexivity | exact Hin ].
  - apply IH.
    + inversion Hdistinct. assumption.
    + intros canonical_element' names' n Hin. apply Hdemoted. right.
      exact Hin.
Qed.

Theorem well_formed_invariant : well_formed_invariant_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t
    [Hcanonical Hlisted] Hearliest.
  unfold Aliases.invariant. cbv zeta. apply andb_true_iff. split.
  - apply forallb_forall. intros [canonical_name aliases] Hin.
    apply (keyed_in_compact name_eqb name_eqb_eq) in Hin.
    assert
      (Hdemoted
        : ∀ n : name,
            Aliases.Aliases_of_canonical_element.mem aliases n = true ->
            demoted_to t n (simple_of_name canonical_name)).
    { intros n Hmem.
      apply Hlisted, (listed_alias_of_name t _ _ aliases); assumption. }
    apply andb_true_iff. split.
    + apply forallb_forall. intros n Hn.
      rewrite <- defined_earlier_names.
      apply Hearliest, Hdemoted, in_all_mem, Hn.
    + destruct
        (Aliases.Aliases_of_canonical_element.mem aliases canonical_name)
        eqn:Hmem;
        [ | reflexivity ].
      pose proof (Hearliest _ _ (Hdemoted _ Hmem)) as Hearlier.
      rewrite
        (proj1
           (defined_earlier_strict A binding_time_resolver
              binding_times_and_modes (simple_of_name canonical_name)
              (simple_of_name canonical_name)
              (simple_of_name canonical_name)))
        in Hearlier.
      discriminate.
  - set (of_names
          := Keyed.compact name_eqb (Aliases.aliases_of_canonical_names t)).
    set (of_consts := Keyed.compact const_eqb (Aliases.aliases_of_consts t)).
    replace
      (map
         (fun binding => Aliases.Aliases_of_canonical_element.all (snd binding))
         of_names
       ++ map
            (fun binding =>
               Aliases.Aliases_of_canonical_element.all (snd binding))
            of_consts)
      with
        (map snd
           (map
              (fun binding =>
                 (simple_of_name (fst binding),
                  Aliases.Aliases_of_canonical_element.all (snd binding)))
              of_names
            ++ map
                 (fun binding =>
                    (Const (fst binding),
                     Aliases.Aliases_of_canonical_element.all (snd binding)))
                 of_consts))
      by (rewrite map_app, !map_map; reflexivity).
    apply (disjoint_alias_sets_of t).
    + rewrite map_app, !map_map. cbn [fst].
      rewrite <- (map_map fst simple_of_name), <- (map_map fst Const).
      apply NoDup_app_apart.
      * apply NoDup_map_injective; [ exact simple_of_name_inj | ].
        apply (keyed_distinct_NoDup name_eqb name_eqb_eq),
          (keyed_compact_distinct name name_eqb name_eqb_eq).
      * apply NoDup_map_injective; [ intros c c' [= ->]; reflexivity | ].
        apply (keyed_distinct_NoDup const_eqb const_eqb_eq),
          (keyed_compact_distinct const const_eqb const_eqb_eq).
      * intros e Hname Hconst.
        apply in_map_iff in Hname as (n & <- & _).
        apply in_map_iff in Hconst as (c & Heq & _).
        destruct n; discriminate.
    + intros canonical_element names n Hin Hn.
      apply in_app_or in Hin as [Hin | Hin];
        apply in_map_iff in Hin as ([key aliases] & [= <- <-] & Hin);
        apply Hlisted; cbn [fst snd] in *.
      * apply (keyed_in_compact name_eqb name_eqb_eq) in Hin.
        apply (listed_alias_of_name t _ _ aliases);
          [ exact Hin | apply in_all_mem, Hn ].
      * apply (keyed_in_compact const_eqb const_eqb_eq) in Hin.
        apply (listed_alias_of_const t _ _ aliases);
          [ exact Hin | apply in_all_mem, Hn ].
Qed.

Theorem invariant_not_consistent : invariant_not_consistent_statement.
Proof.
  exists (fun _ => Binding_time.With_name_mode.symbols).
  exists
    [(Name_var 1%positive,
      (tt, Binding_time.With_name_mode.create 3 Name_mode.normal));
     (Name_var 2%positive,
      (tt, Binding_time.With_name_mode.create 4 Name_mode.normal))].
  exists
    (Aliases.make []
       [(Name_var 1%positive, [(Name_var 2%positive, Name_mode.normal)])] []).
  split; [ reflexivity | ].
  intros [_ Hlisted].
  assert
    (Hdemoted
      : demoted_to
          (Aliases.make []
             [(Name_var 1%positive, [(Name_var 2%positive, Name_mode.normal)])]
             [])
          (Name_var 2%positive) (Var 1%positive)).
  { apply Hlisted. reflexivity. }
  discriminate Hdemoted.
Qed.

Theorem add_invariant : add_invariant_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result Hconsistent Hearliest Hcanonical1 Hcanonical2
    Hcomparable Hadd.
  pose proof
    (add_consistent A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result Hconsistent Hcanonical1
       Hcanonical2 Hadd)
    as Hconsistent'.
  pose proof
    (add_earliest A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result Hconsistent Hearliest
       Hcanonical1 Hcanonical2 Hcomparable Hadd)
    as Hearliest'.
  (* The canonical element is earlier than the name demoted, which was
     canonical. *)
  assert
    (Hearlier
      : Aliases.defined_earlier binding_time_resolver binding_times_and_modes
          (Aliases.canonical_element result)
          (simple_of_name (Aliases.demoted_name result))
        = true
        ∧ is_canonical t (simple_of_name (Aliases.demoted_name result))).
  { pose proof
      (add_result A binding_time_resolver binding_times_and_modes t
         canonical_element1 canonical_element2 result Hadd)
      as Hresult.
    destruct
      (Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         canonical_element1 canonical_element2)
      eqn:Hfirst;
      destruct Hresult as [-> ->].
    - split; assumption.
    - split; [ | assumption ].
      destruct Hcomparable as [Hsecond | Hsecond];
        [ congruence | exact Hsecond ]. }
  destruct Hearlier as [Hearlier Hwas_canonical].
  unfold Aliases.invariant_add_result.
  rewrite (well_formed_invariant A _ _ _ Hconsistent' Hearliest'), Hearlier.
  cbn [andb].
  pose proof
    (add_canonical_element A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result
       (simple_of_name (Aliases.demoted_name result)) Hconsistent Hcanonical1
       Hcanonical2 Hadd)
    as Hnow.
  rewrite (canonical_element_of_canonical t _ Hwas_canonical) in Hnow.
  rewrite (proj2 (simple_equal_eq _ _) eq_refl) in Hnow.
  unfold canonical_element_of in Hnow.
  destruct
    (Aliases.canonical (Aliases.add_result_t result)
       (simple_of_name (Aliases.demoted_name result)));
    [ | reflexivity ].
  rewrite <- Hnow in Hearlier.
  rewrite
    (proj1
       (defined_earlier_strict A binding_time_resolver binding_times_and_modes
          (simple_of_name (Aliases.demoted_name result))
          (simple_of_name (Aliases.demoted_name result))
          (simple_of_name (Aliases.demoted_name result))))
    in Hearlier.
  discriminate.
Qed.

Theorem add_two_symbols_invariant : add_two_symbols_invariant_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes s s' Hdifferent.
  assert
    (Hearlier
      : ∀ s1 s2 : symbol,
          Aliases.name_defined_earlier binding_time_resolver
            binding_times_and_modes (Name_symbol s1) (Name_symbol s2)
          = false).
  { intros s1 s2. unfold Aliases.name_defined_earlier. cbn.
    rewrite orb_true_r. reflexivity. }
  assert (Hnot_equal : Pos.eqb s' s = false).
  { apply Pos.eqb_neq. congruence. }
  eexists. split; [ | split ].
  - unfold Aliases.add, Aliases.add0, Aliases.add_alias,
      Aliases.choose_canonical_element_to_be_demoted, Aliases.defined_earlier.
    cbn [name_of_simple]. rewrite Hearlier.
    unfold Aliases.add_alias_between_canonical_elements.
    cbn [simple_equal]. rewrite Hnot_equal. cbn. reflexivity.
  - unfold Aliases.invariant. cbn. rewrite Pos.eqb_refl. cbn.
    rewrite Hearlier. reflexivity.
  - unfold Aliases.invariant_add_result, Aliases.invariant. cbn.
    rewrite Pos.eqb_refl. cbn. rewrite Hearlier. reflexivity.
Qed.
