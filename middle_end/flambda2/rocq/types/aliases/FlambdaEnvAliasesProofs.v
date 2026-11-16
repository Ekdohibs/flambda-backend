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

(* Proofs of the statements of FlambdaEnvAliasesSpec.v, and examples.
   Nothing here needs review. The examples are in the module [Examples]
   at the end: structures made by [Aliases.add], the environments that
   do and do not respect them, and what the queries answer. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec.
From OxCamlFlambda2 Require FlambdaCompareSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeMeaningProofs
  FlambdaBindingTime FlambdaEnvAliases FlambdaEnvAliasesSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Names, constants and simples                                             *)
(* ======================================================================== *)

Lemma name_eqb_refl (n : name) : name_eqb n n = true.
Proof. apply (keyed_eqb_refl name_eqb name_eqb_eq). Qed.

Lemma const_eqb_eq : decides_equality const_eqb.
Proof.
  intros c c'. unfold FlambdaCompareSpec.const_eqb.
  destruct (FlambdaCompareSpec.const_eq_dec c c') as [Heq | Hne].
  - split; [ intros _; exact Heq | reflexivity ].
  - split; [ discriminate | intros Heq; contradiction ].
Qed.

Lemma name_mode_equal_eq : decides_equality Name_mode.equal.
Proof.
  intros [ | | ] [ | | ]; simpl; split;
    solve [ reflexivity | discriminate ].
Qed.

Lemma simple_of_name_inj (n n' : name) :
  simple_of_name n = simple_of_name n' -> n = n'.
Proof. destruct n, n'; simpl; intros [=]; subst; reflexivity. Qed.

Lemma name_of_simple_of_name (n : name) :
  name_of_simple (simple_of_name n) = Some n.
Proof. destruct n; reflexivity. Qed.

Lemma name_of_simple_some (s : simple) (n : name) :
  name_of_simple s = Some n -> s = simple_of_name n.
Proof. destruct s; simpl; intros [= <-]; reflexivity. Qed.

Lemma name_of_simple_none (s : simple) :
  name_of_simple s = None -> ∃ c : const, s = Const c.
Proof.
  destruct s; simpl; try discriminate. intros _. eexists. reflexivity.
Qed.

Lemma simple_equal_spec (s s' : simple) :
  reflect (s = s') (simple_equal s s').
Proof.
  destruct s as [x | sym | c], s' as [x' | sym' | c']; simpl;
    try (constructor; discriminate).
  - destruct (Pos.eqb_spec x x') as [-> | Hne]; constructor;
      [ reflexivity | intros [= Heq]; contradiction ].
  - destruct (Pos.eqb_spec sym sym') as [-> | Hne]; constructor;
      [ reflexivity | intros [= Heq]; contradiction ].
  - destruct (keyed_eqb_spec const_eqb const_eqb_eq c c') as [-> | Hne];
      constructor; [ reflexivity | intros [= Heq]; contradiction ].
Qed.

(* The proof to give with [simple_equal] wherever [Keyed] asks for one. *)
Lemma simple_equal_eq : decides_equality simple_equal.
Proof.
  intros s s'. destruct (simple_equal_spec s s') as [Heq | Hne].
  - split; [ intros _; exact Heq | reflexivity ].
  - split; [ discriminate | intros Heq; contradiction ].
Qed.

Theorem simple_equal_decides_equality :
  simple_equal_decides_equality_statement.
Proof. exact simple_equal_eq. Qed.

(* ======================================================================== *)
(* The classes                                                              *)
(* ======================================================================== *)

Lemma canonical_of_name (t : Aliases.t) (n : name) :
  Aliases.canonical t (simple_of_name n)
  = match Keyed.lookup name_eqb n (Aliases.canonical_elements t) with
    | None => Aliases.Is_canonical
    | Some canonical_element => Aliases.Alias_of_canonical canonical_element
    end.
Proof.
  unfold Aliases.canonical. rewrite name_of_simple_of_name. reflexivity.
Qed.

(* A simple is canonical, or is a name that has been demoted. *)
Lemma canonical_cases (t : Aliases.t) (e : simple) :
  (is_canonical t e ∧ canonical_element_of t e = e)
  ∨ (∃ (n : name) (canonical_element : simple),
       e = simple_of_name n
       ∧ demoted_to t n canonical_element
       ∧ Aliases.canonical t e = Aliases.Alias_of_canonical canonical_element
       ∧ canonical_element_of t e = canonical_element).
Proof.
  unfold is_canonical, canonical_element_of, demoted_to, Aliases.canonical.
  destruct (name_of_simple e) as [n | ] eqn:Hname; [ | left; auto ].
  apply name_of_simple_some in Hname. subst e.
  destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t))
    as [canonical_element | ] eqn:Hlookup; [ | left; auto ].
  right. exists n, canonical_element. auto.
Qed.

Lemma canonical_element_of_demoted (t : Aliases.t) (n : name)
    (canonical_element : simple) :
  demoted_to t n canonical_element ->
  canonical_element_of t (simple_of_name n) = canonical_element.
Proof.
  unfold demoted_to, canonical_element_of. intros Hdemoted.
  rewrite canonical_of_name, Hdemoted. reflexivity.
Qed.

Lemma canonical_element_of_canonical (t : Aliases.t) (e : simple) :
  is_canonical t e -> canonical_element_of t e = e.
Proof.
  unfold is_canonical, canonical_element_of. intros ->. reflexivity.
Qed.

(* ======================================================================== *)
(* What the structure means                                                 *)
(* ======================================================================== *)

Theorem empty_respected : empty_respected_statement.
Proof.
  intros vars. split.
  - intros n canonical_element Hdemoted. discriminate Hdemoted.
  - intros canonical_element n Hlisted. unfold listed_alias in Hlisted.
    destruct canonical_element; discriminate Hlisted.
Qed.

(* The canonical element of a simple has its value. *)
Lemma respected_canonical_element_of (t : Aliases.t) (vars : env)
    (e : simple) :
  aliases_respected t vars ->
  eval_simple vars (canonical_element_of t e) = eval_simple vars e.
Proof.
  intros [Hcanonical _].
  destruct (canonical_cases t e)
    as [[_ ->] | (n & canonical_element & -> & Hdemoted & _ & ->)];
    [ reflexivity | ].
  symmetry. exact (Hcanonical n canonical_element Hdemoted).
Qed.

Theorem respected_aliased : respected_aliased_statement.
Proof.
  intros t vars e e' Hrespected Haliased.
  rewrite <- (respected_canonical_element_of t vars e Hrespected).
  rewrite <- (respected_canonical_element_of t vars e' Hrespected).
  rewrite Haliased. reflexivity.
Qed.

Theorem respected_canonical : respected_canonical_statement.
Proof.
  intros t vars n Hrespected.
  exact (respected_canonical_element_of t vars (simple_of_name n) Hrespected).
Qed.

(* ------------------------------------------------------------------------ *)
(* The aliases of one canonical element                                     *)
(* ------------------------------------------------------------------------ *)

(* The keys of a map are the names it binds. *)
Lemma in_keys_lookup {A : Type} (n : name)
    (bindings : Keyed.finite_map name A) :
  n ∈ map fst bindings <-> ∃ a : A, Keyed.lookup name_eqb n bindings = Some a.
Proof.
  rewrite <- (keyed_among_in name name_eqb name_eqb_eq).
  rewrite <- keyed_is_bound_among. apply keyed_is_bound_lookup.
Qed.

Lemma mem_lookup (aliases : Aliases.Aliases_of_canonical_element.t)
    (n : name) :
  Aliases.Aliases_of_canonical_element.mem aliases n = true
  <-> ∃ name_mode : Name_mode.t,
        Keyed.lookup name_eqb n aliases = Some name_mode.
Proof. apply keyed_is_bound_lookup. Qed.

(* The aliases at a name mode are the names recorded with it. *)
Lemma in_at_mode (aliases : Aliases.Aliases_of_canonical_element.t)
    (order : Name_mode.t) (n : name) :
  n ∈ Aliases.Aliases_of_canonical_element.at_mode aliases order
  <-> Keyed.lookup name_eqb n aliases = Some order.
Proof.
  unfold Aliases.Aliases_of_canonical_element.at_mode.
  rewrite in_keys_lookup. split.
  - intros [name_mode Hlookup].
    rewrite (keyed_lookup_filter_bindings name name_eqb name_eqb_eq)
      in Hlookup.
    destruct (Keyed.lookup name_eqb n aliases) as [recorded | ];
      [ | discriminate ].
    destruct (Name_mode.equal recorded order) eqn:Hequal; [ | discriminate ].
    apply name_mode_equal_eq in Hequal. subst recorded. reflexivity.
  - intros Hlookup. exists order.
    rewrite (keyed_lookup_filter_bindings name name_eqb name_eqb_eq).
    rewrite Hlookup, (keyed_eqb_refl Name_mode.equal name_mode_equal_eq).
    reflexivity.
Qed.

(* The test that [Aliases.get_canonical_element_exn] and
   [find_earliest_candidates] make of two name modes. *)
Definition name_mode_at_least_b (name_mode min_name_mode : Name_mode.t)
    : bool :=
  match Name_mode.compare_partial_order name_mode min_name_mode with
  | None => false
  | Some c => 0 <=? c
  end.

Lemma name_mode_at_least_b_iff (name_mode min_name_mode : Name_mode.t) :
  name_mode_at_least_b name_mode min_name_mode = true
  <-> name_mode_at_least name_mode min_name_mode.
Proof.
  unfold name_mode_at_least_b, name_mode_at_least.
  destruct (Name_mode.compare_partial_order name_mode min_name_mode)
    as [c | ]; split.
  - intros Hle. exists c. split; [ reflexivity | apply Z.leb_le, Hle ].
  - intros (c' & [= <-] & Hle). apply Z.leb_le, Hle.
  - discriminate.
  - intros (c' & [=] & _).
Qed.

(* Every name mode is at least [In_types], and only [In_types] is below
   every name mode. *)
Lemma name_mode_at_least_in_types (name_mode : Name_mode.t) :
  name_mode_at_least name_mode Name_mode.In_types.
Proof.
  apply name_mode_at_least_b_iff. destruct name_mode; reflexivity.
Qed.

Lemma in_types_at_least (min_name_mode : Name_mode.t) :
  name_mode_at_least Name_mode.In_types min_name_mode ->
  min_name_mode = Name_mode.In_types.
Proof.
  intros Hle. apply name_mode_at_least_b_iff in Hle.
  destruct min_name_mode; solve [ reflexivity | discriminate Hle ].
Qed.

(* What [find_earliest_candidates] answers is the aliases of one name
   mode that is at least the one asked for, and there is one. *)
Lemma find_earliest_candidates_some
    (aliases : Aliases.Aliases_of_canonical_element.t)
    (filter_by_scope : Name_mode.t -> list name -> list name)
    (min_name_mode : Name_mode.t) (candidates : list name) :
  Aliases.Aliases_of_canonical_element.find_earliest_candidates aliases
    filter_by_scope min_name_mode
  = Some candidates ->
  ∃ order : Name_mode.t,
    name_mode_at_least order min_name_mode
    ∧ candidates
      = filter_by_scope order
          (Aliases.Aliases_of_canonical_element.at_mode aliases order)
    ∧ candidates <> [].
Proof.
  unfold Aliases.Aliases_of_canonical_element.find_earliest_candidates.
  cbn [List.fold_left].
  assert (Hcase :
    ∀ order : Name_mode.t,
      name_mode_at_least_b order min_name_mode = true ->
      (if Keyed.is_empty
            (filter_by_scope order
               (Aliases.Aliases_of_canonical_element.at_mode aliases order))
       then None
       else
         Some
           (filter_by_scope order
              (Aliases.Aliases_of_canonical_element.at_mode aliases order)))
      = Some candidates ->
      ∃ order : Name_mode.t,
        name_mode_at_least order min_name_mode
        ∧ candidates
          = filter_by_scope order
              (Aliases.Aliases_of_canonical_element.at_mode aliases order)
        ∧ candidates <> []).
  { intros order Hle Hsome. exists order.
    split; [ apply name_mode_at_least_b_iff, Hle | ].
    destruct (filter_by_scope order
                (Aliases.Aliases_of_canonical_element.at_mode aliases order))
      as [ | first rest ]; simpl in Hsome; [ discriminate | ].
    injection Hsome as <-. split; [ reflexivity | discriminate ]. }
  destruct min_name_mode; simpl Name_mode.compare_partial_order; cbv iota;
    simpl Z.leb; cbv iota.
  - (* Asked for [Normal]: the aliases at [Normal]. *)
    apply (Hcase Name_mode.Normal). reflexivity.
  - (* Asked for [Phantom]: the aliases at [Phantom], or at [Normal]. *)
    destruct (Keyed.is_empty
                (filter_by_scope Name_mode.Phantom
                   (Aliases.Aliases_of_canonical_element.at_mode aliases
                      Name_mode.Phantom))) eqn:Hphantom.
    + apply (Hcase Name_mode.Normal). reflexivity.
    + intros Hsome. apply (Hcase Name_mode.Phantom); [ reflexivity | ].
      rewrite Hphantom. exact Hsome.
  - (* Asked for [In_types]: the aliases at the first of the three name
       modes that has some. *)
    destruct (Keyed.is_empty
                (filter_by_scope Name_mode.In_types
                   (Aliases.Aliases_of_canonical_element.at_mode aliases
                      Name_mode.In_types))) eqn:Hin_types.
    + destruct (Keyed.is_empty
                  (filter_by_scope Name_mode.Phantom
                     (Aliases.Aliases_of_canonical_element.at_mode aliases
                        Name_mode.Phantom))) eqn:Hphantom.
      * apply (Hcase Name_mode.Normal). reflexivity.
      * intros Hsome. apply (Hcase Name_mode.Phantom); [ reflexivity | ].
        rewrite Hphantom. exact Hsome.
    + intros Hsome. apply (Hcase Name_mode.In_types); [ reflexivity | ].
      rewrite Hin_types. exact Hsome.
Qed.

(* The earliest of a list, as [find_earliest_alias] finds it, is one of
   the list. *)
Lemma fold_earliest_in (earlier : name -> name -> bool) (names : list name)
    (first : name) :
  List.fold_left
    (fun min_elt elt => if earlier elt min_elt then elt else min_elt)
    names first
  ∈ first :: names.
Proof.
  revert first. induction names as [ | n names IH]; intros first; simpl.
  - left. reflexivity.
  - destruct (earlier n first).
    + right. apply IH.
    + destruct (IH first) as [Heq | Hin]; [ left; exact Heq | ].
      right. right. exact Hin.
Qed.

(* What [find_earliest_alias] answers is an alias listed for the canonical
   element, recorded with a name mode that is at least the one asked
   for, and, unless that is [In_types], in scope with it. *)
Lemma find_earliest_alias_some {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (min_name_mode : Name_mode.t)
    (answer : simple) :
  Aliases.find_earliest_alias binding_time_resolver binding_times_and_modes
    t canonical_element min_binding_time min_name_mode
  = Some answer ->
  ∃ (n : name) (order : Name_mode.t),
    answer = simple_of_name n
    ∧ Keyed.lookup name_eqb n
        (Aliases.get_aliases_of_canonical_element t canonical_element)
      = Some order
    ∧ name_mode_at_least order min_name_mode
    ∧ (order = Name_mode.In_types
       ∨ Aliases.compute_name_mode binding_times_and_modes (simple_of_name n)
           min_binding_time
         = order).
Proof.
  unfold Aliases.find_earliest_alias.
  match goal with
  | |- match ?candidates with _ => _ end = _ -> _ =>
    destruct candidates as [at_earliest_mode | ] eqn:Hcandidates
  end; [ | discriminate ].
  destruct at_earliest_mode as [ | first rest ]; [ discriminate | ].
  intros [= <-].
  apply find_earliest_candidates_some in Hcandidates
    as (order & Hle & Hfiltered & _).
  eexists. exists order. split; [ reflexivity | ].
  match goal with
  | |- Keyed.lookup name_eqb ?earliest _ = _ ∧ _ =>
    assert (Hmember : earliest ∈ first :: rest);
      [ | revert Hmember; generalize earliest; intros n Hmember ]
  end.
  { destruct (fold_earliest_in
                (Aliases.name_defined_earlier binding_time_resolver
                   binding_times_and_modes)
                (first :: rest) first) as [Heq | Hin];
      [ left; exact Heq | exact Hin ]. }
  rewrite Hfiltered in Hmember.
  destruct (Name_mode.equal order Name_mode.in_types) eqn:Hin_types.
  - apply name_mode_equal_eq in Hin_types.
    apply in_at_mode in Hmember. auto.
  - apply filter_In in Hmember as [Hmember Hscoped].
    apply in_at_mode in Hmember. apply name_mode_equal_eq in Hscoped. auto.
Qed.

(* ------------------------------------------------------------------------ *)
(* The canonical element at a name mode, in one equation                    *)
(* ------------------------------------------------------------------------ *)

(* [Aliases.get_canonical_element_exn] answers the canonical element if
   its name mode is at least the one asked for, and otherwise what
   [find_earliest_alias] finds among its aliases. *)
Lemma get_canonical_element_exn_eq {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) :
  Aliases.get_canonical_element_exn binding_time_resolver
    binding_times_and_modes t element elt_name_mode min_name_mode
    min_binding_time
  = if name_mode_at_least_b
         (canonical_name_mode binding_times_and_modes t element elt_name_mode
            min_binding_time)
         min_name_mode
    then Some (canonical_element_of t element)
    else
      Aliases.find_earliest_alias binding_time_resolver
        binding_times_and_modes t (canonical_element_of t element)
        min_binding_time min_name_mode.
Proof.
  unfold Aliases.get_canonical_element_exn, canonical_name_mode,
    canonical_element_of.
  fold (name_mode_at_least_b elt_name_mode min_name_mode).
  destruct (Aliases.canonical t element) as [ | canonical_element ].
  - destruct (name_mode_at_least_b elt_name_mode min_name_mode) eqn:Hle;
      [ reflexivity | ].
    cbv beta iota zeta.
    fold (name_mode_at_least_b elt_name_mode min_name_mode).
    rewrite Hle. reflexivity.
  - cbv beta iota zeta.
    match goal with
    | |- (if ?test then _ else _) = _ =>
      change test
        with (name_mode_at_least_b
                (Aliases.compute_name_mode binding_times_and_modes
                   canonical_element min_binding_time)
                min_name_mode)
    end.
    destruct (name_mode_at_least_b elt_name_mode min_name_mode); reflexivity.
Qed.

Theorem respected_canonical_at_mode : respected_canonical_at_mode_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t vars element
    elt_name_mode min_name_mode min_binding_time answer Hrespected Hanswer.
  rewrite get_canonical_element_exn_eq in Hanswer.
  rewrite <- (respected_canonical_element_of t vars element Hrespected).
  destruct (name_mode_at_least_b _ _).
  - injection Hanswer as <-. reflexivity.
  - apply find_earliest_alias_some in Hanswer
      as (n & order & -> & Hlookup & _ & _).
    destruct Hrespected as [_ Hlisted]. apply Hlisted.
    apply mem_lookup. exists order. exact Hlookup.
Qed.

Theorem respected_coincide : respected_coincide_statement.
Proof.
  intros t vars vars' Hsame [Hcanonical Hlisted].
  (* A canonical element has the same value in the two environments. *)
  assert (Hsimple :
    ∀ (n : name) (canonical_element : simple),
      demoted_to t n canonical_element ∨ listed_alias t canonical_element n ->
      eval_simple vars canonical_element
      = eval_simple vars' canonical_element).
  { intros n canonical_element Hmentioned.
    destruct (name_of_simple canonical_element) as [n' | ] eqn:Hname.
    - apply name_of_simple_some in Hname. subst canonical_element.
      apply (Hsame n').
      destruct Hmentioned as [Hdemoted | Hlisted'].
      + right. left. exists n. exact Hdemoted.
      + right. right. right. exists n. exact Hlisted'.
    - apply name_of_simple_none in Hname as [c ->]. reflexivity. }
  split.
  - intros n canonical_element Hdemoted.
    rewrite <- (Hsame n); [ | left; exists canonical_element; exact Hdemoted ].
    rewrite <- (Hsimple n canonical_element (or_introl Hdemoted)).
    exact (Hcanonical n canonical_element Hdemoted).
  - intros canonical_element n Hlisted'.
    rewrite <- (Hsame n);
      [ | right; right; left; exists canonical_element; exact Hlisted' ].
    rewrite <- (Hsimple n canonical_element (or_intror Hlisted')).
    exact (Hlisted canonical_element n Hlisted').
Qed.

Theorem respected_alias_type : respected_alias_type_statement.
Proof.
  intros t statics vars v k n Hrespected.
  rewrite !(alias_satisfied statics vars v).
  rewrite (respected_canonical t vars n Hrespected). reflexivity.
Qed.

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

Theorem defined_earlier_names : defined_earlier_names_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes alias than.
  unfold Aliases.defined_earlier. rewrite !name_of_simple_of_name.
  reflexivity.
Qed.

Theorem constant_earliest : constant_earliest_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes c e n.
  unfold Aliases.defined_earlier. rewrite name_of_simple_of_name.
  split; reflexivity.
Qed.

Theorem symbol_earlier : symbol_earlier_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes s x n.
  split; [ reflexivity | ].
  unfold Aliases.name_defined_earlier. destruct n as [y | s']; simpl.
  - reflexivity.
  - rewrite orb_true_r. reflexivity.
Qed.

(* The order on two variables, written with the binding times of the
   statements. *)
Lemma name_defined_earlier_variables {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (x y : variable) :
  Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
    (Name_var x) (Name_var y)
  = if Pos.eqb x y
    then false
    else if binding_time_in binding_times_and_modes x
            <? binding_time_in binding_times_and_modes y
    then true
    else if negb
              (binding_time_in binding_times_and_modes x
               =? binding_time_in binding_times_and_modes y)
    then false
    else
      resolved_binding_time binding_time_resolver x
      <? resolved_binding_time binding_time_resolver y.
Proof.
  unfold Aliases.name_defined_earlier, binding_time_in,
    resolved_binding_time, Aliases.binding_time_and_name_mode,
    Binding_time.strictly_earlier, Binding_time.equal.
  simpl name_eqb. simpl name_is_symbol. rewrite orb_false_r.
  cbv beta iota zeta. simpl andb. cbv iota.
  destruct (Keyed.lookup name_eqb (Name_var x) binding_times_and_modes)
    as [[a time_and_mode] | ];
    destruct (Keyed.lookup name_eqb (Name_var y) binding_times_and_modes)
      as [[a' time_and_mode'] | ];
    reflexivity.
Qed.

Theorem variable_earlier : variable_earlier_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes x y.
  rewrite name_defined_earlier_variables.
  destruct (Pos.eqb_spec x y) as [-> | Hne].
  - split; [ discriminate | lia ].
  - destruct (Z.ltb_spec (binding_time_in binding_times_and_modes x)
                (binding_time_in binding_times_and_modes y)).
    + split; [ intros _; left; assumption | reflexivity ].
    + destruct (Z.eqb_spec (binding_time_in binding_times_and_modes x)
                  (binding_time_in binding_times_and_modes y)); simpl.
      * rewrite Z.ltb_lt. split; [ intros Hlt; right; auto | ].
        intros [Hlt | [_ Hlt]]; [ lia | exact Hlt ].
      * split; [ discriminate | lia ].
Qed.

Lemma name_defined_earlier_irreflexive {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (n : name) :
  Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
    n n
  = false.
Proof.
  unfold Aliases.name_defined_earlier. rewrite name_eqb_refl. reflexivity.
Qed.

Lemma name_defined_earlier_transitive {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (n n' n'' : name) :
  Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
    n n'
  = true ->
  Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
    n' n''
  = true ->
  Aliases.name_defined_earlier binding_time_resolver binding_times_and_modes
    n n''
  = true.
Proof.
  pose proof (symbol_earlier A binding_time_resolver binding_times_and_modes)
    as Hsymbol.
  destruct n as [x | s], n' as [y | s'], n'' as [z | s''];
    intros Hfirst Hsecond;
    try (rewrite (proj2 (Hsymbol _ 1%positive _)) in Hfirst;
         discriminate Hfirst);
    try (rewrite (proj2 (Hsymbol _ 1%positive _)) in Hsecond;
         discriminate Hsecond).
  - (* Three variables: the order of the binding times is transitive. *)
    apply variable_earlier in Hfirst, Hsecond. apply variable_earlier. lia.
  - exact (proj1 (Hsymbol s z (Name_var z))).
Qed.

Theorem defined_earlier_strict : defined_earlier_strict_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes e e' e''.
  unfold Aliases.defined_earlier. split.
  - destruct (name_of_simple e);
      [ apply name_defined_earlier_irreflexive | reflexivity ].
  - destruct (name_of_simple e'') as [n'' | ]; [ | intros _; discriminate ].
    destruct (name_of_simple e') as [n' | ]; [ | discriminate ].
    destruct (name_of_simple e) as [n | ]; [ | reflexivity ].
    apply name_defined_earlier_transitive.
Qed.

(* ======================================================================== *)
(* The canonical element                                                    *)
(* ======================================================================== *)

Theorem get_canonical_ignoring_name_mode :
  get_canonical_ignoring_name_mode_statement.
Proof. intros t n. reflexivity. Qed.

Theorem canonical_element_canonical : canonical_element_canonical_statement.
Proof.
  intros t e [Hcanonical _].
  destruct (canonical_cases t e)
    as [[Hself ->] | (n & canonical_element & _ & Hdemoted & _ & ->)].
  - exact Hself.
  - exact (Hcanonical n canonical_element Hdemoted).
Qed.

Theorem canonical_element_idempotent :
  canonical_element_idempotent_statement.
Proof.
  intros t e Hconsistent. apply canonical_element_of_canonical.
  apply canonical_element_canonical, Hconsistent.
Qed.

Theorem canonical_element_mentioned : canonical_element_mentioned_statement.
Proof.
  intros t n n' Hcanonical.
  change (canonical_element_of t (simple_of_name n) = simple_of_name n')
    in Hcanonical.
  destruct (canonical_cases t (simple_of_name n))
    as [[_ Hself] | (m & canonical_element & Hname & Hdemoted & _ & Hof)].
  - left. rewrite Hself in Hcanonical.
    symmetry. exact (simple_of_name_inj _ _ Hcanonical).
  - right. right. left. exists m.
    rewrite Hof in Hcanonical. rewrite <- Hcanonical. exact Hdemoted.
Qed.

Theorem canonical_element_earliest : canonical_element_earliest_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t e Hearliest Hother.
  destruct (canonical_cases t e)
    as [[_ Hself] | (n & canonical_element & -> & Hdemoted & _ & ->)].
  - contradiction.
  - exact (Hearliest n canonical_element Hdemoted).
Qed.

Theorem canonical_element_determined :
  canonical_element_determined_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t t' Hconsistent
    Hconsistent' Hearliest Hearliest' Hclasses e.
  destruct (simple_equal_spec (canonical_element_of t e)
              (canonical_element_of t' e)) as [Heq | Hne]; [ exact Heq | ].
  exfalso.
  (* Each of the two is in the class of which the other is the canonical
     element, so each is earlier than the other. *)
  assert (Hc :
    canonical_element_of t' (canonical_element_of t e)
    = canonical_element_of t' e).
  { symmetry. apply (proj1 (Hclasses e (canonical_element_of t e))).
    unfold aliased. symmetry.
    apply canonical_element_idempotent, Hconsistent. }
  assert (Hc' :
    canonical_element_of t (canonical_element_of t' e)
    = canonical_element_of t e).
  { symmetry. apply (proj2 (Hclasses e (canonical_element_of t' e))).
    unfold aliased. symmetry.
    apply canonical_element_idempotent, Hconsistent'. }
  pose proof
    (canonical_element_earliest A binding_time_resolver
       binding_times_and_modes t' (canonical_element_of t e) Hearliest')
    as H1.
  rewrite Hc in H1. specialize (H1 (fun H => Hne (eq_sym H))).
  pose proof
    (canonical_element_earliest A binding_time_resolver
       binding_times_and_modes t (canonical_element_of t' e) Hearliest)
    as H2.
  rewrite Hc' in H2. specialize (H2 Hne).
  destruct
    (defined_earlier_strict A binding_time_resolver binding_times_and_modes
       (canonical_element_of t e) (canonical_element_of t' e)
       (canonical_element_of t e))
    as [Hirreflexive Htransitive].
  rewrite (Htransitive H2 H1) in Hirreflexive. discriminate Hirreflexive.
Qed.

Theorem canonical_element_constant : canonical_element_constant_statement.
Proof. intros t e c Haliased. exact Haliased. Qed.

Theorem canonical_element_symbol : canonical_element_symbol_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t e s Hearliest
    Haliased.
  rewrite Haliased.
  destruct (canonical_cases t (Symbol s))
    as [[_ ->] | (n & canonical_element & Hname & Hdemoted & _ & ->)];
    [ left; reflexivity | ].
  right.
  change (Symbol s) with (simple_of_name (Name_symbol s)) in Hname.
  apply simple_of_name_inj in Hname. subst n.
  specialize (Hearliest (Name_symbol s) canonical_element Hdemoted).
  destruct (name_of_simple canonical_element) as [n | ] eqn:Hcanonical.
  - (* A name is not earlier than a symbol. *)
    apply name_of_simple_some in Hcanonical. subst canonical_element.
    rewrite defined_earlier_names in Hearliest.
    rewrite (proj2 (symbol_earlier A binding_time_resolver
                      binding_times_and_modes s 1%positive n)) in Hearliest.
    discriminate Hearliest.
  - exact (name_of_simple_none canonical_element Hcanonical).
Qed.

(* ======================================================================== *)
(* The canonical element at a name mode                                     *)
(* ======================================================================== *)

Theorem canonical_at_mode : canonical_at_mode_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time Hle.
  rewrite get_canonical_element_exn_eq.
  apply name_mode_at_least_b_iff in Hle. rewrite Hle. reflexivity.
Qed.

Theorem canonical_at_mode_name_mode : canonical_at_mode_name_mode_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time answer Hanswer.
  rewrite get_canonical_element_exn_eq in Hanswer.
  destruct (name_mode_at_least_b _ _) eqn:Hle.
  - (* The canonical element. *)
    injection Hanswer as <-. apply name_mode_at_least_b_iff in Hle.
    unfold canonical_name_mode in Hle.
    destruct (canonical_cases t element)
      as [[Hself ->] | (n & canonical_element & _ & _ & Hcanonical & ->)].
    + left. rewrite Hself in Hle. auto.
    + right. rewrite Hcanonical in Hle. exact Hle.
  - (* An alias: it is in scope with its name mode, or its name mode was
       [In_types], and then that is what was asked for. *)
    right.
    apply find_earliest_alias_some in Hanswer
      as (n & order & -> & _ & Horder & [-> | ->]).
    + apply in_types_at_least in Horder. subst min_name_mode.
      apply name_mode_at_least_in_types.
    + exact Horder.
Qed.

Theorem canonical_at_mode_aliased : canonical_at_mode_aliased_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time answer Hconsistent Hanswer.
  rewrite get_canonical_element_exn_eq in Hanswer. unfold aliased.
  destruct (name_mode_at_least_b _ _).
  - injection Hanswer as <-. apply canonical_element_of_canonical.
    apply canonical_element_canonical, Hconsistent.
  - apply find_earliest_alias_some in Hanswer
      as (n & order & -> & Hlookup & _ & _).
    apply canonical_element_of_demoted. apply Hconsistent.
    apply mem_lookup. exists order. exact Hlookup.
Qed.

(* ------------------------------------------------------------------------ *)
(* Which alias is the answer                                                *)
(* ------------------------------------------------------------------------ *)

Lemma is_empty_nil {B : Type} (l : list B) :
  Keyed.is_empty l = true <-> l = [].
Proof. destruct l; simpl; split; solve [ reflexivity | discriminate ]. Qed.

(* Settles what is left of a case of [find_earliest_candidates_cases]: a
   comparison of two name modes, by computation, or a list of candidates
   that is empty and is not. *)
Ltac settle_name_modes :=
  first
    [ apply name_mode_at_least_b_iff; reflexivity
    | match goal with
      | Hle : name_mode_at_least _ _ |- _ =>
        apply name_mode_at_least_b_iff in Hle; discriminate Hle
      end
    | match goal with
      | Hne : ?c <> [], Hempty : Keyed.is_empty ?c = true |- _ =>
        exfalso; apply Hne, is_empty_nil, Hempty
      end
    | match goal with
      | Hempty : Keyed.is_empty ?c = true |- ?c = [] =>
        apply is_empty_nil, Hempty
      end ].

(* [find_earliest_candidates] in full: the answer is the candidates of the
   least name mode that is at least the one asked for and has some; there
   is none only if no such name mode has a candidate. *)
Lemma find_earliest_candidates_cases
    (aliases : Aliases.Aliases_of_canonical_element.t)
    (filter_by_scope : Name_mode.t -> list name -> list name)
    (min_name_mode : Name_mode.t) :
  match
    Aliases.Aliases_of_canonical_element.find_earliest_candidates aliases
      filter_by_scope min_name_mode
  with
  | Some candidates =>
    ∃ order : Name_mode.t,
      name_mode_at_least order min_name_mode
      ∧ candidates
        = filter_by_scope order
            (Aliases.Aliases_of_canonical_element.at_mode aliases order)
      ∧ candidates <> []
      ∧ ∀ order' : Name_mode.t,
          name_mode_at_least order' min_name_mode ->
          filter_by_scope order'
            (Aliases.Aliases_of_canonical_element.at_mode aliases order')
          <> [] ->
          name_mode_at_least order' order
  | None =>
    ∀ order : Name_mode.t,
      name_mode_at_least order min_name_mode ->
      filter_by_scope order
        (Aliases.Aliases_of_canonical_element.at_mode aliases order)
      = []
  end.
Proof.
  unfold Aliases.Aliases_of_canonical_element.find_earliest_candidates.
  cbn [List.fold_left].
  remember (filter_by_scope Name_mode.In_types
              (Aliases.Aliases_of_canonical_element.at_mode aliases
                 Name_mode.In_types)) as in_types eqn:Hin_types.
  remember (filter_by_scope Name_mode.Phantom
              (Aliases.Aliases_of_canonical_element.at_mode aliases
                 Name_mode.Phantom)) as phantom eqn:Hphantom.
  remember (filter_by_scope Name_mode.Normal
              (Aliases.Aliases_of_canonical_element.at_mode aliases
                 Name_mode.Normal)) as normal eqn:Hnormal.
  destruct (Keyed.is_empty in_types) eqn:Hempty_in_types;
    destruct (Keyed.is_empty phantom) eqn:Hempty_phantom;
    destruct (Keyed.is_empty normal) eqn:Hempty_normal;
    destruct min_name_mode; simpl;
    rewrite ?Hempty_in_types, ?Hempty_phantom, ?Hempty_normal;
    try (match goal with
         | Heq : ?c = filter_by_scope ?order _
           |- ∃ order' : Name_mode.t, _ ∧ ?c = _ ∧ _ =>
           exists order
         end;
         split; [ settle_name_modes | ];
         split; [ assumption | ];
         split;
           [ intros Hnil; apply is_empty_nil in Hnil; congruence | ];
         intros order' Hle Hne;
         destruct order';
         rewrite <- ?Hin_types, <- ?Hphantom, <- ?Hnormal in Hne;
         settle_name_modes);
    intros order Hle; destruct order;
    rewrite <- ?Hin_types, <- ?Hphantom, <- ?Hnormal;
    settle_name_modes.
Qed.

(* The earliest of a list, as [find_earliest_alias] finds it: it is the
   name it starts from or is earlier than it, and no name of the list is
   earlier than it. *)
Section Earliest.

Variable earlier : name -> name -> bool.

Hypothesis earlier_irreflexive : ∀ n : name, earlier n n = false.

Hypothesis earlier_transitive :
  ∀ n n' n'' : name,
    earlier n n' = true -> earlier n' n'' = true -> earlier n n'' = true.

Lemma fold_earliest_before (names : list name) (first : name) :
  let earliest :=
    List.fold_left
      (fun min_elt elt => if earlier elt min_elt then elt else min_elt)
      names first
  in
  earliest = first ∨ earlier earliest first = true.
Proof.
  revert first. induction names as [ | n names IH]; intros first; simpl.
  - left. reflexivity.
  - destruct (earlier n first) eqn:Hn.
    + right. destruct (IH n) as [-> | Hbefore]; [ exact Hn | ].
      exact (earlier_transitive _ _ _ Hbefore Hn).
    + apply IH.
Qed.

Lemma fold_earliest_minimal (names : list name) (first n : name) :
  n ∈ first :: names ->
  earlier n
    (List.fold_left
       (fun min_elt elt => if earlier elt min_elt then elt else min_elt)
       names first)
  = false.
Proof.
  revert first. induction names as [ | n' names IH]; intros first Hin.
  - destruct Hin as [<- | []]. apply earlier_irreflexive.
  - simpl. destruct (earlier n' first) eqn:Hn'.
    + (* [n'] is the earliest so far. *)
      destruct Hin as [<- | Hin]; [ | apply IH; exact Hin ].
      match goal with
      | |- ?test = false => destruct test eqn:Hfirst; [ | reflexivity ]
      end.
      exfalso.
      destruct (fold_earliest_before names n') as [Heq | Hbefore];
        simpl in *.
      * rewrite Heq in Hfirst.
        pose proof (earlier_transitive _ _ _ Hfirst Hn') as Hloop.
        rewrite earlier_irreflexive in Hloop. discriminate Hloop.
      * pose proof (earlier_transitive _ _ _ Hfirst Hbefore) as Hfirst'.
        pose proof (earlier_transitive _ _ _ Hfirst' Hn') as Hloop.
        rewrite earlier_irreflexive in Hloop. discriminate Hloop.
    + (* [first] still is. *)
      destruct Hin as [<- | [<- | Hin]];
        [ apply IH; left; reflexivity | | apply IH; right; exact Hin ].
      match goal with
      | |- ?test = false => destruct test eqn:Hearlier; [ | reflexivity ]
      end.
      exfalso.
      destruct (fold_earliest_before names first) as [Heq | Hbefore];
        simpl in *.
      * rewrite Heq in Hearlier. congruence.
      * pose proof (earlier_transitive _ _ _ Hearlier Hbefore) as Hn.
        congruence.
Qed.

End Earliest.

(* The filter that [find_earliest_alias] gives to
   [find_earliest_candidates]. *)
Definition filter_by_scope {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (min_binding_time : Binding_time.t) (name_mode : Name_mode.t)
    (names : list name) : list name :=
  if Name_mode.equal name_mode Name_mode.in_types
  then names
  else
    filter
      (fun name =>
         Name_mode.equal name_mode
           (Aliases.compute_name_mode binding_times_and_modes
              (simple_of_name name) min_binding_time))
      names.

(* What it keeps of the aliases recorded with a name mode is the
   candidates of the Spec. *)
Lemma in_filter_by_scope {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (order : Name_mode.t) (n : name) :
  n ∈ filter_by_scope binding_times_and_modes min_binding_time order
        (Aliases.Aliases_of_canonical_element.at_mode
           (Aliases.get_aliases_of_canonical_element t canonical_element)
           order)
  <-> alias_candidate binding_times_and_modes t canonical_element
        min_binding_time order n.
Proof.
  unfold filter_by_scope, alias_candidate.
  destruct (Name_mode.equal order Name_mode.in_types) eqn:Hin_types.
  - apply name_mode_equal_eq in Hin_types. rewrite in_at_mode.
    split; [ auto | intros [Hlookup _]; exact Hlookup ].
  - rewrite filter_In, in_at_mode. split; intros [Hlookup Hscoped].
    + apply name_mode_equal_eq in Hscoped. auto.
    + split; [ exact Hlookup | ]. apply name_mode_equal_eq.
      destruct Hscoped as [-> | Hscoped]; [ discriminate Hin_types | ].
      symmetry. exact Hscoped.
Qed.

Lemma find_earliest_alias_eq {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (min_name_mode : Name_mode.t) :
  Aliases.find_earliest_alias binding_time_resolver binding_times_and_modes
    t canonical_element min_binding_time min_name_mode
  = match
      Aliases.Aliases_of_canonical_element.find_earliest_candidates
        (Aliases.get_aliases_of_canonical_element t canonical_element)
        (filter_by_scope binding_times_and_modes min_binding_time)
        min_name_mode
    with
    | Some [] | None => None
    | Some (first :: rest) =>
      Some
        (simple_of_name
           (List.fold_left
              (fun min_elt elt =>
                 if Aliases.name_defined_earlier binding_time_resolver
                      binding_times_and_modes elt min_elt
                 then elt
                 else min_elt)
              (first :: rest) first))
    end.
Proof.
  unfold Aliases.find_earliest_alias.
  fold (filter_by_scope binding_times_and_modes min_binding_time).
  destruct (Aliases.Aliases_of_canonical_element.find_earliest_candidates
              _ _ _) as [[ | first rest ] | ];
    reflexivity.
Qed.

Theorem earliest_alias : earliest_alias_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time answer Hnot Hanswer.
  rewrite get_canonical_element_exn_eq in Hanswer.
  destruct (name_mode_at_least_b _ _) eqn:Hle;
    [ exfalso; apply Hnot, name_mode_at_least_b_iff, Hle | ].
  clear Hnot Hle. revert Hanswer.
  generalize (canonical_element_of t element). intros canonical_element.
  rewrite find_earliest_alias_eq.
  pose proof (find_earliest_candidates_cases
    (Aliases.get_aliases_of_canonical_element t canonical_element)
    (filter_by_scope binding_times_and_modes min_binding_time)
    min_name_mode) as Hcases.
  destruct (Aliases.Aliases_of_canonical_element.find_earliest_candidates
              _ _ _) as [candidates | ]; [ | discriminate ].
  destruct Hcases as (order & Hle & Hcandidates & _ & Hleast).
  destruct candidates as [ | first rest ]; [ discriminate | ].
  intros [= <-]. eexists. exists order.
  split; [ reflexivity | ]. split; [ exact Hle | ]. split.
  - (* The answer is a candidate. *)
    apply in_filter_by_scope. rewrite <- Hcandidates.
    destruct (fold_earliest_in
                (Aliases.name_defined_earlier binding_time_resolver
                   binding_times_and_modes)
                (first :: rest) first) as [Heq | Hin];
      [ left; exact Heq | exact Hin ].
  - intros n' order' Hle' Hcandidate. apply in_filter_by_scope in Hcandidate.
    split.
    + (* The name mode of any other candidate is at least its. *)
      apply Hleast; [ exact Hle' | ]. intros Hnil.
      rewrite Hnil in Hcandidate. exact Hcandidate.
    + (* No candidate of its name mode is earlier. *)
      intros ->. rewrite <- Hcandidates in Hcandidate.
      rewrite name_defined_earlier_irreflexive.
      apply fold_earliest_minimal.
      * apply name_defined_earlier_irreflexive.
      * apply name_defined_earlier_transitive.
      * exact Hcandidate.
Qed.

Theorem no_alias : no_alias_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t element
    elt_name_mode min_name_mode min_binding_time Hnot.
  rewrite get_canonical_element_exn_eq.
  destruct (name_mode_at_least_b _ _) eqn:Hle;
    [ exfalso; apply Hnot, name_mode_at_least_b_iff, Hle | ].
  clear Hnot Hle.
  generalize (canonical_element_of t element). intros canonical_element.
  rewrite find_earliest_alias_eq.
  pose proof (find_earliest_candidates_cases
    (Aliases.get_aliases_of_canonical_element t canonical_element)
    (filter_by_scope binding_times_and_modes min_binding_time)
    min_name_mode) as Hcases.
  destruct (Aliases.Aliases_of_canonical_element.find_earliest_candidates
              _ _ _) as [candidates | ].
  - (* There are candidates: the first of them is one. *)
    destruct Hcases as (order & Hle & Hcandidates & Hne & _).
    destruct candidates as [ | first rest ]; [ contradiction | ].
    split; [ discriminate | ]. intros Hnone. exfalso.
    apply (Hnone first order Hle). apply in_filter_by_scope.
    rewrite <- Hcandidates. left. reflexivity.
  - split; [ | reflexivity ]. intros _ n order Hle Hcandidate.
    apply in_filter_by_scope in Hcandidate.
    rewrite (Hcases order Hle) in Hcandidate. exact Hcandidate.
Qed.

(* ======================================================================== *)
(* Adding an alias                                                          *)
(* ======================================================================== *)

Lemma simple_equal_names (n n' : name) :
  simple_equal (simple_of_name n) (simple_of_name n') = name_eqb n n'.
Proof. destruct n, n'; reflexivity. Qed.

(* A map that binds each name of a list to one thing. *)
Lemma lookup_constant_map {B : Type} (b : B) (n : name) (names : list name) :
  Keyed.lookup name_eqb n (map (fun alias => (alias, b)) names)
  = if Keyed.among name_eqb n names then Some b else None.
Proof.
  induction names as [ | first names IH]; simpl; [ reflexivity | ].
  destruct (name_eqb n first); [ reflexivity | exact IH ].
Qed.

Lemma among_all (aliases : Aliases.Aliases_of_canonical_element.t)
    (n : name) :
  Keyed.among name_eqb n (Aliases.Aliases_of_canonical_element.all aliases)
  = Aliases.Aliases_of_canonical_element.mem aliases n.
Proof. apply (keyed_among_domain name name_eqb name_eqb_eq). Qed.

Lemma mem_union (aliases aliases' : Aliases.Aliases_of_canonical_element.t)
    (n : name) :
  Aliases.Aliases_of_canonical_element.mem
    (Aliases.Aliases_of_canonical_element.union aliases aliases') n
  = Aliases.Aliases_of_canonical_element.mem aliases n
    || Aliases.Aliases_of_canonical_element.mem aliases' n.
Proof.
  unfold Aliases.Aliases_of_canonical_element.mem,
    Aliases.Aliases_of_canonical_element.union, Keyed.is_bound.
  rewrite keyed_lookup_app.
  destruct (Keyed.lookup name_eqb n aliases); reflexivity.
Qed.

Lemma simple_equal_name_const (n : name) (c : const) :
  simple_equal (simple_of_name n) (Const c) = false.
Proof. destruct n; reflexivity. Qed.

Lemma simple_equal_const_name (c : const) (n : name) :
  simple_equal (Const c) (simple_of_name n) = false.
Proof. destruct n; reflexivity. Qed.

Lemma get_aliases_of_name (t : Aliases.t) (n : name) :
  Aliases.get_aliases_of_canonical_element t (simple_of_name n)
  = match Keyed.lookup name_eqb n (Aliases.aliases_of_canonical_names t) with
    | None => Aliases.Aliases_of_canonical_element.empty
    | Some aliases => aliases
    end.
Proof. destruct n; reflexivity. Qed.

Lemma get_aliases_of_const (t : Aliases.t) (c : const) :
  Aliases.get_aliases_of_canonical_element t (Const c)
  = match Keyed.lookup const_eqb c (Aliases.aliases_of_consts t) with
    | None => Aliases.Aliases_of_canonical_element.empty
    | Some aliases => aliases
    end.
Proof. reflexivity. Qed.

(* What [add_alias_between_canonical_elements] makes of the structure,
   where the two elements are different: every lookup in the new maps, in
   terms of the old ones. [aliases] is the new list of aliases of the
   canonical element: the name demoted, its aliases, and those the
   canonical element had. *)
Lemma add_alias_between_canonical_elements_ok {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t t' : Aliases.t) (canonical_element to_be_demoted : simple) :
  Aliases.add_alias_between_canonical_elements binding_times_and_modes t
    canonical_element to_be_demoted
  = Some (Or_bottom.Ok t') ->
  canonical_element <> to_be_demoted ->
  ∃ (name_to_be_demoted : name)
    (aliases : Aliases.Aliases_of_canonical_element.t),
    to_be_demoted = simple_of_name name_to_be_demoted
    ∧ aliases
      = (name_to_be_demoted,
         Aliases.compute_name_mode_unscoped binding_times_and_modes
           to_be_demoted)
        :: Aliases.get_aliases_of_canonical_element t to_be_demoted
        ++ Aliases.get_aliases_of_canonical_element t canonical_element
    ∧ Aliases.Aliases_of_canonical_element.mem
        (Aliases.get_aliases_of_canonical_element t to_be_demoted
         ++ Aliases.get_aliases_of_canonical_element t canonical_element)
        name_to_be_demoted
      = false
    ∧ (∀ n : name,
        Keyed.lookup name_eqb n (Aliases.canonical_elements t')
        = if name_eqb n name_to_be_demoted
          then Some canonical_element
          else if Aliases.Aliases_of_canonical_element.mem
                    (Aliases.get_aliases_of_canonical_element t
                       to_be_demoted)
                    n
          then Some canonical_element
          else Keyed.lookup name_eqb n (Aliases.canonical_elements t))
    ∧ (∀ e : simple,
        Aliases.get_aliases_of_canonical_element t' e
        = if simple_equal e canonical_element
          then aliases
          else if simple_equal e to_be_demoted
          then Aliases.Aliases_of_canonical_element.empty
          else Aliases.get_aliases_of_canonical_element t e).
Proof.
  unfold Aliases.add_alias_between_canonical_elements.
  intros Hadd Hne.
  destruct (simple_equal_spec canonical_element to_be_demoted)
    as [Heq | _]; [ contradiction | ].
  destruct (name_of_simple to_be_demoted) as [name_to_be_demoted | ]
    eqn:Hname; [ | discriminate Hadd ].
  apply name_of_simple_some in Hname.
  unfold Aliases.Aliases_of_canonical_element.add in Hadd.
  destruct (Aliases.Aliases_of_canonical_element.mem _ name_to_be_demoted)
    eqn:Hfresh; [ discriminate Hadd | ].
  exists name_to_be_demoted. eexists.
  split; [ exact Hname | ]. split; [ reflexivity | ].
  split; [ exact Hfresh | ].
  (* The new structure, whichever kind of simple the canonical element
     is. *)
  assert (Ht' :
    Aliases.canonical_elements t'
    = (name_to_be_demoted, canonical_element)
      :: map (fun alias => (alias, canonical_element))
           (Aliases.Aliases_of_canonical_element.all
              (Aliases.get_aliases_of_canonical_element t to_be_demoted))
      ++ Aliases.canonical_elements t).
  { destruct canonical_element; injection Hadd as <-; reflexivity. }
  split.
  - intros n. rewrite Ht'. cbn [Keyed.lookup].
    destruct (name_eqb n name_to_be_demoted); [ reflexivity | ].
    rewrite keyed_lookup_app, lookup_constant_map, among_all.
    destruct (Aliases.Aliases_of_canonical_element.mem _ n); reflexivity.
  - intros e.
    (* The two inverse maps of the new structure. *)
    assert (Hmaps :
      (∃ canonical_name : name,
         canonical_element = simple_of_name canonical_name
         ∧ Aliases.aliases_of_canonical_names t'
           = (canonical_name,
              (name_to_be_demoted,
               Aliases.compute_name_mode_unscoped binding_times_and_modes
                 to_be_demoted)
              :: Aliases.get_aliases_of_canonical_element t to_be_demoted
              ++ Aliases.get_aliases_of_canonical_element t
                   canonical_element)
             :: Keyed.unbind name_eqb name_to_be_demoted
                  (Aliases.aliases_of_canonical_names t)
         ∧ Aliases.aliases_of_consts t' = Aliases.aliases_of_consts t)
      ∨ (∃ c : const,
           canonical_element = Const c
           ∧ Aliases.aliases_of_canonical_names t'
             = Keyed.unbind name_eqb name_to_be_demoted
                 (Aliases.aliases_of_canonical_names t)
           ∧ Aliases.aliases_of_consts t'
             = (c,
                (name_to_be_demoted,
                 Aliases.compute_name_mode_unscoped binding_times_and_modes
                   to_be_demoted)
                :: Aliases.get_aliases_of_canonical_element t to_be_demoted
                ++ Aliases.get_aliases_of_canonical_element t
                     canonical_element)
               :: Aliases.aliases_of_consts t)).
    { destruct canonical_element as [x | sym | c]; injection Hadd as <-;
        [ left; exists (Name_var x) | left; exists (Name_symbol sym)
        | right; exists c ];
        repeat split; reflexivity. }
    clear Hadd Ht' Hfresh. subst to_be_demoted.
    remember (Aliases.get_aliases_of_canonical_element t
                (simple_of_name name_to_be_demoted)) as of_demoted
      eqn:Hof_demoted.
    remember (Aliases.get_aliases_of_canonical_element t canonical_element)
      as of_canonical eqn:Hof_canonical.
    clear Hof_demoted Hof_canonical.
    destruct (name_of_simple e) as [n | ] eqn:He;
      [ apply name_of_simple_some in He
      | apply name_of_simple_none in He as [c' He] ];
      subst e;
      destruct Hmaps as [(canonical_name & Hcanonical & Hnames & Hconsts)
                        | (c & Hcanonical & Hnames & Hconsts)];
      subst canonical_element;
      rewrite ?get_aliases_of_name, ?get_aliases_of_const, ?simple_equal_names,
        ?simple_equal_name_const, ?simple_equal_const_name, ?Hnames,
        ?Hconsts;
      cbn [Keyed.lookup simple_equal].
    + destruct (name_eqb n canonical_name); [ reflexivity | ].
      rewrite (keyed_lookup_unbind name name_eqb name_eqb_eq).
      destruct (name_eqb n name_to_be_demoted); reflexivity.
    + rewrite (keyed_lookup_unbind name name_eqb name_eqb_eq).
      destruct (name_eqb n name_to_be_demoted); reflexivity.
    + reflexivity.
    + destruct (const_eqb c' c); reflexivity.
Qed.

(* [add] in one equation: which of the two is to be demoted, and then
   [add_alias_between_canonical_elements]. *)
Lemma add_eq {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple) :
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = let '(canonical_element, to_be_demoted) :=
      if Aliases.defined_earlier binding_time_resolver
           binding_times_and_modes canonical_element1 canonical_element2
      then (canonical_element1, canonical_element2)
      else (canonical_element2, canonical_element1)
    in
    match
      Aliases.add_alias_between_canonical_elements binding_times_and_modes
        t canonical_element to_be_demoted
    with
    | None => None
    | Some Or_bottom.Bottom => Some Or_bottom.Bottom
    | Some (Or_bottom.Ok t') =>
      match name_of_simple to_be_demoted with
      | None => None
      | Some n =>
        Some (Or_bottom.Ok (Aliases.make_add_result t' n canonical_element))
      end
    end.
Proof.
  unfold Aliases.add, Aliases.add0, Aliases.add_alias,
    Aliases.choose_canonical_element_to_be_demoted.
  destruct (Aliases.defined_earlier binding_time_resolver
              binding_times_and_modes canonical_element1 canonical_element2);
    cbv beta iota zeta;
    destruct (Aliases.add_alias_between_canonical_elements _ _ _ _)
      as [[t' | ] | ];
    reflexivity.
Qed.

Lemma add_ok {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) :
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = Some (Or_bottom.Ok result) ->
  (if Aliases.defined_earlier binding_time_resolver binding_times_and_modes
        canonical_element1 canonical_element2
   then
     Aliases.canonical_element result = canonical_element1
     ∧ simple_of_name (Aliases.demoted_name result) = canonical_element2
   else
     Aliases.canonical_element result = canonical_element2
     ∧ simple_of_name (Aliases.demoted_name result) = canonical_element1)
  ∧ Aliases.add_alias_between_canonical_elements binding_times_and_modes t
      (Aliases.canonical_element result)
      (simple_of_name (Aliases.demoted_name result))
    = Some (Or_bottom.Ok (Aliases.add_result_t result)).
Proof.
  rewrite add_eq.
  destruct (Aliases.defined_earlier binding_time_resolver
              binding_times_and_modes canonical_element1 canonical_element2);
    cbv beta iota zeta;
    destruct (Aliases.add_alias_between_canonical_elements _ _ _ _)
      as [[t' | ] | ] eqn:Hbetween;
    try discriminate;
    match goal with
    | |- context [name_of_simple ?d] =>
      destruct (name_of_simple d) as [n | ] eqn:Hname
    end;
    try discriminate;
    intros [= <-]; apply name_of_simple_some in Hname;
    cbn [Aliases.canonical_element Aliases.demoted_name Aliases.add_result_t];
    rewrite <- Hname; auto.
Qed.

Theorem add_result : add_result_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result Hadd.
  exact (proj1 (add_ok _ _ _ _ _ _ Hadd)).
Qed.

(* The two arguments of [add] are the canonical element and the name
   demoted, in one order or the other. *)
Lemma add_arguments {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) :
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = Some (Or_bottom.Ok result) ->
  (Aliases.canonical_element result = canonical_element1
   ∧ simple_of_name (Aliases.demoted_name result) = canonical_element2)
  ∨ (Aliases.canonical_element result = canonical_element2
     ∧ simple_of_name (Aliases.demoted_name result) = canonical_element1).
Proof.
  intros Hadd. pose proof (add_result _ _ _ _ _ _ _ Hadd) as Hresult.
  destruct (Aliases.defined_earlier _ _ _ _); auto.
Qed.

(* What is earlier than something is not a constant. *)
Lemma defined_earlier_name {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (e e' : simple) :
  Aliases.defined_earlier binding_time_resolver binding_times_and_modes e e'
  = true ->
  ∃ n : name, e' = simple_of_name n.
Proof.
  unfold Aliases.defined_earlier.
  destruct (name_of_simple e') as [n | ] eqn:Hname; [ | discriminate ].
  intros _. exists n. apply name_of_simple_some, Hname.
Qed.

(* [add_alias_between_canonical_elements] answers bottom when it is to
   demote a constant, and stops with a fatal error when the name to be
   demoted is already listed among the aliases of one of the two. *)
Lemma add_alias_between_canonical_elements_bottom {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element to_be_demoted : simple) :
  Aliases.add_alias_between_canonical_elements binding_times_and_modes t
    canonical_element to_be_demoted
  = Some Or_bottom.Bottom
  <-> canonical_element <> to_be_demoted
      ∧ (∃ c : const, to_be_demoted = Const c).
Proof.
  unfold Aliases.add_alias_between_canonical_elements.
  destruct (simple_equal_spec canonical_element to_be_demoted) as [Heq | Hne].
  - split; [ discriminate | intros [Hne _]; contradiction ].
  - destruct (name_of_simple to_be_demoted) as [n | ] eqn:Hname.
    + split.
      * destruct (Aliases.Aliases_of_canonical_element.add _ _ _);
          [ destruct canonical_element | ]; discriminate.
      * intros [_ [c ->]]. discriminate Hname.
    + split; [ | reflexivity ]. intros _.
      split; [ exact Hne | exact (name_of_simple_none _ Hname) ].
Qed.

Lemma add_alias_between_canonical_elements_none {A : Type}
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element : simple) (name_to_be_demoted : name) :
  Aliases.add_alias_between_canonical_elements binding_times_and_modes t
    canonical_element (simple_of_name name_to_be_demoted)
  = None ->
  listed_alias t (simple_of_name name_to_be_demoted) name_to_be_demoted
  ∨ listed_alias t canonical_element name_to_be_demoted.
Proof.
  unfold Aliases.add_alias_between_canonical_elements.
  destruct (simple_equal _ _); [ discriminate | ].
  rewrite name_of_simple_of_name.
  unfold Aliases.Aliases_of_canonical_element.add.
  destruct (Aliases.Aliases_of_canonical_element.mem _ name_to_be_demoted)
    eqn:Hmem.
  - intros _. rewrite mem_union in Hmem.
    apply orb_true_iff in Hmem. exact Hmem.
  - destruct canonical_element; discriminate.
Qed.

Theorem add_bottom : add_bottom_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2.
  rewrite add_eq.
  (* The answer is bottom when that of
     [add_alias_between_canonical_elements] is. *)
  assert (Hbottom :
    ∀ canonical_element to_be_demoted : simple,
      match
        Aliases.add_alias_between_canonical_elements binding_times_and_modes
          t canonical_element to_be_demoted
      with
      | None => None
      | Some Or_bottom.Bottom => Some Or_bottom.Bottom
      | Some (Or_bottom.Ok t') =>
        match name_of_simple to_be_demoted with
        | None => None
        | Some n =>
          Some
            (Or_bottom.Ok (Aliases.make_add_result t' n canonical_element))
        end
      end
      = Some Or_bottom.Bottom
      <-> canonical_element <> to_be_demoted
          ∧ (∃ c : const, to_be_demoted = Const c)).
  { intros canonical_element to_be_demoted.
    rewrite <- (add_alias_between_canonical_elements_bottom
                  binding_times_and_modes t).
    destruct (Aliases.add_alias_between_canonical_elements _ _ _ _)
      as [[t' | ] | ];
      [ destruct (name_of_simple to_be_demoted) | | ];
      split; solve [ discriminate | reflexivity ]. }
  destruct (Aliases.defined_earlier binding_time_resolver
              binding_times_and_modes canonical_element1 canonical_element2)
    eqn:Hearlier; cbv beta iota zeta; rewrite Hbottom.
  - (* The second is to be demoted, and it is a name. *)
    destruct (defined_earlier_name _ _ _ _ Hearlier) as [n ->].
    split.
    + intros [_ [c Hconst]]. destruct n; discriminate Hconst.
    + intros (c1 & c2 & _ & Hconst & _). destruct n; discriminate Hconst.
  - (* The first is to be demoted. If it is a constant, so is the
       second, as a constant is earlier than any name. *)
    split.
    + intros [Hne [c1 ->]].
      destruct canonical_element2 as [x | sym | c2];
        try discriminate Hearlier.
      exists c1, c2. repeat split. intros ->. contradiction.
    + intros (c1 & c2 & -> & -> & Hne).
      split; [ intros [= ->]; contradiction | exists c1; reflexivity ].
Qed.

Theorem add_defined : add_defined_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 [_ Hlisted] Hcanonical1 Hcanonical2 Hne.
  rewrite add_eq.
  (* A canonical name is not listed among the aliases of anything. *)
  assert (Hnone :
    ∀ canonical_element to_be_demoted : simple,
      is_canonical t to_be_demoted ->
      canonical_element <> to_be_demoted ->
      match
        Aliases.add_alias_between_canonical_elements binding_times_and_modes
          t canonical_element to_be_demoted
      with
      | None => None
      | Some Or_bottom.Bottom => Some Or_bottom.Bottom
      | Some (Or_bottom.Ok t') =>
        match name_of_simple to_be_demoted with
        | None => None
        | Some n =>
          Some
            (Or_bottom.Ok (Aliases.make_add_result t' n canonical_element))
        end
      end
      <> None).
  { intros canonical_element to_be_demoted Hcanonical Hdifferent.
    destruct (name_of_simple to_be_demoted) as [n | ] eqn:Hname.
    - apply name_of_simple_some in Hname. subst to_be_demoted.
      destruct (Aliases.add_alias_between_canonical_elements _ _ _ _)
        as [[t' | ] | ] eqn:Hbetween; try discriminate.
      exfalso.
      unfold is_canonical in Hcanonical.
      rewrite canonical_of_name in Hcanonical.
      destruct (add_alias_between_canonical_elements_none _ _ _ _ Hbetween)
        as [Hself | Hother];
        apply Hlisted in Hself || apply Hlisted in Hother;
        unfold demoted_to in *;
        [ rewrite Hself in Hcanonical | rewrite Hother in Hcanonical ];
        discriminate Hcanonical.
    - apply name_of_simple_none in Hname as [c Hconst].
      rewrite (proj2 (add_alias_between_canonical_elements_bottom _ t
                        canonical_element to_be_demoted));
        [ discriminate | ].
      split; [ exact Hdifferent | exists c; exact Hconst ]. }
  destruct (Aliases.defined_earlier binding_time_resolver
              binding_times_and_modes canonical_element1 canonical_element2);
    cbv beta iota zeta; apply Hnone; auto.
Qed.

(* ------------------------------------------------------------------------ *)
(* The structure after [add]                                                *)
(* ------------------------------------------------------------------------ *)

(* [t'] is [t] with the class of the canonical name [demoted] merged into
   that of [canonical_element]: what is demoted to what, and what is
   listed for what, in terms of [t]. *)
Definition merged (t t' : Aliases.t) (canonical_element : simple)
    (demoted : name) : Prop :=
  (∀ (n : name) (e : simple),
    demoted_to t' n e
    <-> (e = canonical_element
         ∧ (n = demoted ∨ listed_alias t (simple_of_name demoted) n))
        ∨ (n <> demoted
           ∧ ~ listed_alias t (simple_of_name demoted) n
           ∧ demoted_to t n e))
  ∧ (∀ (e : simple) (n : name),
      listed_alias t' e n
      <-> (e = canonical_element
           ∧ (n = demoted
              ∨ listed_alias t (simple_of_name demoted) n
              ∨ listed_alias t canonical_element n))
          ∨ (e <> canonical_element
             ∧ e <> simple_of_name demoted
             ∧ listed_alias t e n)).

(* A successful [add] leaves the structure as it is, if its two arguments
   are the same, or merges the two classes. *)
Lemma add_merged {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) :
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = Some (Or_bottom.Ok result) ->
  (Aliases.canonical_element result
   = simple_of_name (Aliases.demoted_name result)
   ∧ Aliases.add_result_t result = t)
  ∨ (Aliases.canonical_element result
     <> simple_of_name (Aliases.demoted_name result)
     ∧ merged t (Aliases.add_result_t result)
         (Aliases.canonical_element result) (Aliases.demoted_name result)
     ∧ ∀ (e : simple) (n : name) (name_mode : Name_mode.t),
         Keyed.lookup name_eqb n
           (Aliases.get_aliases_of_canonical_element
              (Aliases.add_result_t result) e)
         = Some name_mode ->
         (n = Aliases.demoted_name result
          ∧ name_mode
            = Aliases.compute_name_mode_unscoped binding_times_and_modes
                (simple_of_name n))
         ∨ (∃ e' : simple,
             Keyed.lookup name_eqb n
               (Aliases.get_aliases_of_canonical_element t e')
             = Some name_mode)).
Proof.
  intros Hadd. apply add_ok in Hadd as [_ Hbetween].
  set (canonical_element := Aliases.canonical_element result) in *.
  set (demoted := Aliases.demoted_name result) in *.
  set (t' := Aliases.add_result_t result) in *.
  destruct (simple_equal_spec canonical_element (simple_of_name demoted))
    as [Heq | Hne].
  - left. split; [ exact Heq | ].
    unfold Aliases.add_alias_between_canonical_elements in Hbetween.
    rewrite Heq in Hbetween.
    destruct (simple_equal_spec (simple_of_name demoted)
                (simple_of_name demoted)) as [_ | Hne];
      [ | contradiction ].
    injection Hbetween as <-. reflexivity.
  - right. split; [ exact Hne | ].
    destruct (add_alias_between_canonical_elements_ok _ _ _ _ _ Hbetween Hne)
      as (name_to_be_demoted & aliases & Hname & Haliases & _ & Hcanonical
          & Hlisted).
    apply simple_of_name_inj in Hname. subst name_to_be_demoted.
    (* What is listed in the new list of aliases of the canonical
       element. *)
    assert (Hmem :
      ∀ n : name,
        Aliases.Aliases_of_canonical_element.mem aliases n = true
        <-> n = demoted
            ∨ listed_alias t (simple_of_name demoted) n
            ∨ listed_alias t canonical_element n).
    { intros n. subst aliases. unfold listed_alias.
      unfold Aliases.Aliases_of_canonical_element.mem at 1.
      unfold Keyed.is_bound. cbn [Keyed.lookup].
      destruct (keyed_eqb_spec name_eqb name_eqb_eq n demoted)
        as [-> | Hdifferent]; [ split; auto | ].
      fold (Keyed.is_bound name_eqb n
              (Aliases.get_aliases_of_canonical_element t
                 (simple_of_name demoted)
               ++ Aliases.get_aliases_of_canonical_element t
                    canonical_element)).
      change (Keyed.is_bound name_eqb n
                (Aliases.get_aliases_of_canonical_element t
                   (simple_of_name demoted)
                 ++ Aliases.get_aliases_of_canonical_element t
                      canonical_element))
        with (Aliases.Aliases_of_canonical_element.mem
                (Aliases.Aliases_of_canonical_element.union
                   (Aliases.get_aliases_of_canonical_element t
                      (simple_of_name demoted))
                   (Aliases.get_aliases_of_canonical_element t
                      canonical_element))
                n).
      rewrite mem_union, orb_true_iff. split; [ auto | ].
      intros [Heq | Hor]; [ contradiction | exact Hor ]. }
    split; [ split | ].
    + (* What is demoted. *)
      intros n e. unfold demoted_to. rewrite Hcanonical.
      fold (listed_alias t (simple_of_name demoted) n).
      destruct (keyed_eqb_spec name_eqb name_eqb_eq n demoted)
        as [-> | Hdifferent].
      * split.
        -- intros [= <-]. left. auto.
        -- intros [[-> _] | [Hdifferent _]];
             [ reflexivity | contradiction ].
      * unfold listed_alias.
        destruct (Aliases.Aliases_of_canonical_element.mem
                    (Aliases.get_aliases_of_canonical_element t
                       (simple_of_name demoted))
                    n) eqn:Hof_demoted.
        -- split.
           ++ intros [= <-]. left. auto.
           ++ intros [[-> _] | (_ & Hnot & _)];
                [ reflexivity | contradiction ].
        -- split.
           ++ intros Hold. right. repeat split; auto; discriminate.
           ++ intros [[_ [Heq | Hyes]] | (_ & _ & Hold)];
                [ contradiction | discriminate Hyes | exact Hold ].
    + (* What is listed. *)
      intros e n. unfold listed_alias at 1. rewrite Hlisted.
      destruct (simple_equal_spec e canonical_element) as [-> | Hother].
      * rewrite Hmem. split.
        -- intros Hor. left. auto.
        -- intros [[_ Hor] | [Hother _]]; [ exact Hor | contradiction ].
      * destruct (simple_equal_spec e (simple_of_name demoted))
          as [-> | Hother'].
        -- split; [ discriminate | ].
           intros [[Heq _] | (_ & Hself & _)]; contradiction.
        -- split.
           ++ intros Hold. right. auto.
           ++ intros [[Heq _] | (_ & _ & Hold)];
                [ contradiction | exact Hold ].
    + (* The name modes recorded. *)
      intros e n name_mode. rewrite Hlisted.
      destruct (simple_equal e canonical_element).
      * subst aliases. cbn [Keyed.lookup].
        destruct (keyed_eqb_spec name_eqb name_eqb_eq n demoted)
          as [-> | Hdifferent].
        -- intros [= <-]. left. auto.
        -- rewrite keyed_lookup_app. intros Hlookup. right.
           destruct (Keyed.lookup name_eqb n
                       (Aliases.get_aliases_of_canonical_element t
                          (simple_of_name demoted))) eqn:Hof_demoted.
           ++ exists (simple_of_name demoted). congruence.
           ++ exists canonical_element. exact Hlookup.
      * destruct (simple_equal e (simple_of_name demoted));
          [ discriminate | ].
        intros Hlookup. right. exists e. exact Hlookup.
Qed.

Theorem add_respected : add_respected_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result vars Hadd Hrespected Hvalues.
  (* The name demoted has the value of the canonical element. *)
  assert (Hdemoted :
    value_of_name vars (Aliases.demoted_name result)
    = eval_simple vars (Aliases.canonical_element result)).
  { unfold value_of_name.
    destruct (add_arguments _ _ _ _ _ _ Hadd) as [[Hc Hd] | [Hc Hd]];
      rewrite Hc, Hd; congruence. }
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[_ ->] | (_ & [Hdem Hlis] & _)]; [ exact Hrespected | ].
  destruct Hrespected as [Hcanonical Hlisted]. split.
  - intros n e Hd. apply Hdem in Hd as [[-> [-> | Hl]] | (_ & _ & Hold)].
    + exact Hdemoted.
    + rewrite (Hlisted _ _ Hl). exact Hdemoted.
    + exact (Hcanonical _ _ Hold).
  - intros e n Hl.
    apply Hlis in Hl as [[-> [-> | [Hl | Hl]]] | (_ & _ & Hold)].
    + exact Hdemoted.
    + rewrite (Hlisted _ _ Hl). exact Hdemoted.
    + exact (Hlisted _ _ Hl).
    + exact (Hlisted _ _ Hold).
Qed.

(* A name that is canonical has not been demoted. *)
Lemma canonical_not_demoted (t : Aliases.t) (n : name) (e : simple) :
  is_canonical t (simple_of_name n) -> ~ demoted_to t n e.
Proof.
  unfold is_canonical, demoted_to. rewrite canonical_of_name.
  intros Hcanonical Hdemoted. rewrite Hdemoted in Hcanonical.
  discriminate Hcanonical.
Qed.

Theorem add_respected_equal : add_respected_equal_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result vars Hadd Hrespected.
  destruct (add_merged _ _ _ _ _ _ Hadd) as [[Hsame _] | (_ & [Hdem _] & _)].
  - destruct (add_arguments _ _ _ _ _ _ Hadd) as [[Hc Hd] | [Hc Hd]];
      rewrite <- Hc, <- Hd, Hsame; reflexivity.
  - assert (Hdemoted :
      value_of_name vars (Aliases.demoted_name result)
      = eval_simple vars (Aliases.canonical_element result)).
    { apply (proj1 Hrespected). apply Hdem. left. auto. }
    unfold value_of_name in Hdemoted.
    destruct (add_arguments _ _ _ _ _ _ Hadd) as [[Hc Hd] | [Hc Hd]];
      rewrite <- Hc, <- Hd; congruence.
Qed.

Theorem add_respected_before : add_respected_before_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result vars Hconsistent Hcanonical1 Hcanonical2 Hadd
    Hrespected.
  pose proof
    (add_respected_equal A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result vars Hadd Hrespected)
    as Hequal.
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[_ Hsame] | (_ & [Hdem _] & _)];
    [ rewrite <- Hsame; exact Hrespected | ].
  (* The name demoted was canonical, and has the value of the canonical
     element. *)
  assert (Hwas : is_canonical t (simple_of_name (Aliases.demoted_name result))).
  { destruct (add_arguments _ _ _ _ _ _ Hadd) as [[_ Hd] | [_ Hd]];
      rewrite Hd; assumption. }
  assert (Hdemoted :
    eval_simple vars (simple_of_name (Aliases.demoted_name result))
    = eval_simple vars (Aliases.canonical_element result)).
  { destruct (add_arguments _ _ _ _ _ _ Hadd) as [[Hc Hd] | [Hc Hd]];
      rewrite Hc, Hd; congruence. }
  destruct Hconsistent as [_ Hlisted].
  assert (Hcanonical :
    ∀ (n : name) (e : simple),
      demoted_to t n e -> value_of_name vars n = eval_simple vars e).
  { intros n e Hd.
    destruct (simple_equal_spec e
                (simple_of_name (Aliases.demoted_name result)))
      as [-> | Hne].
    - rewrite Hdemoted. apply (proj1 Hrespected). apply Hdem. left.
      split; [ reflexivity | ]. right. apply Hlisted, Hd.
    - apply (proj1 Hrespected). apply Hdem. right. split; [ | split ].
      + intros ->. exact (canonical_not_demoted t _ e Hwas Hd).
      + intros Hl. apply Hlisted in Hl. unfold demoted_to in Hd, Hl.
        congruence.
      + exact Hd. }
  split; [ exact Hcanonical | ].
  intros e n Hl. apply Hcanonical, Hlisted, Hl.
Qed.

(* The two arguments of [add] being canonical, so are the canonical
   element and the name demoted. *)
Lemma add_canonical_arguments {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (result : Aliases.add_result) :
  is_canonical t canonical_element1 ->
  is_canonical t canonical_element2 ->
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = Some (Or_bottom.Ok result) ->
  is_canonical t (Aliases.canonical_element result)
  ∧ is_canonical t (simple_of_name (Aliases.demoted_name result)).
Proof.
  intros Hcanonical1 Hcanonical2 Hadd.
  destruct (add_arguments _ _ _ _ _ _ Hadd) as [[-> ->] | [-> ->]]; auto.
Qed.

Theorem add_consistent : add_consistent_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result [Hcanonical Hlisted] Hcanonical1 Hcanonical2
    Hadd.
  destruct (add_canonical_arguments _ _ _ _ _ _ Hcanonical1 Hcanonical2 Hadd)
    as [Hcanonical_c Hcanonical_d].
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[_ ->] | (Hne & [Hdem Hlis] & _)]; [ split; assumption | ].
  set (canonical_element := Aliases.canonical_element result) in *.
  set (demoted := Aliases.demoted_name result) in *.
  set (t' := Aliases.add_result_t result) in *.
  (* What was canonical, but for the name demoted, still is. *)
  assert (Hstay :
    ∀ e : simple,
      is_canonical t e -> e <> simple_of_name demoted -> is_canonical t' e).
  { intros e He Hother.
    destruct (name_of_simple e) as [n | ] eqn:Hname.
    - apply name_of_simple_some in Hname. subst e.
      unfold is_canonical. rewrite canonical_of_name.
      destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t'))
        as [e | ] eqn:Hlookup; [ exfalso | reflexivity ].
      apply Hdem in Hlookup as [[_ [-> | Hl]] | (_ & _ & Hold)].
      + apply Hother. reflexivity.
      + exact (canonical_not_demoted t n _ He (proj2 (Hlisted _ _) Hl)).
      + exact (canonical_not_demoted t n _ He Hold).
    - apply name_of_simple_none in Hname as [c ->]. reflexivity. }
  split.
  - intros n e Hd. apply Hdem in Hd as [[-> _] | (_ & Hnot & Hold)].
    + apply Hstay; assumption.
    + apply Hstay; [ exact (Hcanonical _ _ Hold) | ].
      intros ->. apply Hnot, Hlisted, Hold.
  - intros n e. split.
    + intros Hd. apply Hlis.
      apply Hdem in Hd as [[-> Hor] | (_ & Hnot & Hold)].
      * left. split; [ reflexivity | ]. destruct Hor; auto.
      * destruct (simple_equal_spec e canonical_element) as [-> | Hother].
        -- left. split; [ reflexivity | ]. right. right.
           apply Hlisted, Hold.
        -- right. repeat split; [ exact Hother | | apply Hlisted, Hold ].
           intros ->. apply Hnot, Hlisted, Hold.
    + intros Hl. apply Hdem.
      apply Hlis in Hl as [[-> Hor] | (Hother & Hother' & Hold)].
      * destruct Hor as [-> | [Hl | Hl]]; [ left; auto | left; auto | ].
        apply Hlisted in Hl.
        destruct (keyed_eqb_spec name_eqb name_eqb_eq n demoted)
          as [-> | Hdifferent]; [ left; auto | ].
        destruct (Aliases.Aliases_of_canonical_element.mem
                    (Aliases.get_aliases_of_canonical_element t
                       (simple_of_name demoted))
                    n) eqn:Hof_demoted.
        -- left. auto.
        -- right. repeat split; [ exact Hdifferent | | exact Hl ].
           unfold listed_alias. rewrite Hof_demoted. discriminate.
      * apply Hlisted in Hold. right. repeat split; [ | | exact Hold ].
        -- intros ->.
           exact (canonical_not_demoted t demoted e Hcanonical_d Hold).
        -- intros Hl. apply Hlisted in Hl. unfold demoted_to in *.
           rewrite Hold in Hl. injection Hl as ->. contradiction.
Qed.

Theorem add_canonical_element : add_canonical_element_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result e [Hcanonical Hlisted] Hcanonical1 Hcanonical2
    Hadd.
  destruct (add_canonical_arguments _ _ _ _ _ _ Hcanonical1 Hcanonical2 Hadd)
    as [Hcanonical_c Hcanonical_d].
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[Heq ->] | (Hne & [Hdem Hlis] & _)].
  - (* The two arguments were the same: nothing has changed. *)
    destruct (simple_equal_spec (canonical_element_of t e)
                (simple_of_name (Aliases.demoted_name result)))
      as [Hdemoted | _]; [ | reflexivity ].
    rewrite Heq. exact Hdemoted.
  - set (canonical_element := Aliases.canonical_element result) in *.
    set (demoted := Aliases.demoted_name result) in *.
    set (t' := Aliases.add_result_t result) in *.
    destruct (canonical_cases t' e)
      as [[Hself ->] | (n & e' & -> & Hd & _ & ->)].
    + (* [e] is canonical afterwards: it was, and it is not the name
         demoted. *)
      destruct (canonical_cases t e)
        as [[_ ->] | (n & e' & -> & Hd & _ & _)].
      * destruct (simple_equal_spec e (simple_of_name demoted))
          as [-> | _]; [ exfalso | reflexivity ].
        apply (canonical_not_demoted t' demoted canonical_element Hself).
        apply Hdem. left. auto.
      * exfalso.
        assert (Hafter : ∃ e'' : simple, demoted_to t' n e'').
        { destruct (keyed_eqb_spec name_eqb name_eqb_eq n demoted)
            as [-> | Hdifferent].
          - exists canonical_element. apply Hdem. left. auto.
          - destruct (Aliases.Aliases_of_canonical_element.mem
                        (Aliases.get_aliases_of_canonical_element t
                           (simple_of_name demoted))
                        n) eqn:Hof_demoted.
            + exists canonical_element. apply Hdem. left. auto.
            + exists e'. apply Hdem. right.
              repeat split; [ exact Hdifferent | | exact Hd ].
              unfold listed_alias. rewrite Hof_demoted. discriminate. }
        destruct Hafter as [e'' Hafter].
        exact (canonical_not_demoted t' n e'' Hself Hafter).
    + (* [e] is a name that is demoted afterwards. *)
      apply Hdem in Hd as [[-> Hor] | (_ & Hnot & Hold)].
      * (* To the canonical element: it was the name demoted, or one of
           its aliases. *)
        assert (Hbefore :
          canonical_element_of t (simple_of_name n) = simple_of_name demoted).
        { destruct Hor as [-> | Hl].
          - exact (canonical_element_of_canonical t _ Hcanonical_d).
          - apply canonical_element_of_demoted, Hlisted, Hl. }
        rewrite Hbefore.
        destruct (simple_equal_spec (simple_of_name demoted)
                    (simple_of_name demoted));
          [ reflexivity | contradiction ].
      * (* As before. *)
        rewrite (canonical_element_of_demoted t n e' Hold).
        destruct (simple_equal_spec e' (simple_of_name demoted))
          as [-> | _]; [ exfalso | reflexivity ].
        apply Hnot, Hlisted, Hold.
Qed.

Theorem add_aliased : add_aliased_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result e e' Hconsistent Hcanonical1 Hcanonical2 Hadd.
  unfold aliased.
  rewrite !(add_canonical_element _ _ _ _ _ _ _ _ Hconsistent Hcanonical1
              Hcanonical2 Hadd).
  rewrite (canonical_element_of_canonical t _ Hcanonical1).
  rewrite (canonical_element_of_canonical t _ Hcanonical2).
  generalize (canonical_element_of t e) (canonical_element_of t e').
  intros c c'.
  destruct (add_arguments _ _ _ _ _ _ Hadd) as [[-> ->] | [-> ->]];
    destruct (simple_equal_spec c canonical_element2);
    destruct (simple_equal_spec c' canonical_element2);
    destruct (simple_equal_spec c canonical_element1);
    destruct (simple_equal_spec c' canonical_element1);
    subst; intuition congruence.
Qed.

Theorem canonical_element_unmentioned :
  canonical_element_unmentioned_statement.
Proof.
  intros t n Hunmentioned. unfold Aliases.get_canonical_ignoring_name_mode.
  rewrite canonical_of_name.
  destruct (Keyed.lookup name_eqb n (Aliases.canonical_elements t))
    as [canonical_element | ] eqn:Hlookup;
    [ | reflexivity ].
  exfalso. apply Hunmentioned. left. exists canonical_element. exact Hlookup.
Qed.

Lemma add_name_constant_either {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (n : name) (c : const)
    (canonical_element1 canonical_element2 : simple) :
  aliases_consistent t ->
  is_canonical t (simple_of_name n) ->
  (canonical_element1 = simple_of_name n ∧ canonical_element2 = Const c)
  ∨ (canonical_element1 = Const c ∧ canonical_element2 = simple_of_name n) ->
  ∃ result : Aliases.add_result,
    Aliases.add binding_time_resolver binding_times_and_modes t
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok result)
    ∧ Aliases.demoted_name result = n
    ∧ Aliases.canonical_element result = Const c.
Proof.
  intros Hconsistent Hcanonical Harguments.
  destruct
    (Aliases.add binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2)
    as [[result | ] | ] eqn:Hadd.
  - exists result. split; [ reflexivity | ].
    pose proof
      (add_result A binding_time_resolver binding_times_and_modes t
         canonical_element1 canonical_element2 result Hadd)
      as Hresult.
    destruct Harguments as [[-> ->] | [-> ->]].
    + rewrite
        (proj2
           (constant_earliest A binding_time_resolver binding_times_and_modes
              c (simple_of_name n) n))
        in Hresult.
      destruct Hresult as [Hcanonical_element Hdemoted].
      split; [ apply simple_of_name_inj, Hdemoted | exact Hcanonical_element ].
    + rewrite
        (proj1
           (constant_earliest A binding_time_resolver binding_times_and_modes
              c (simple_of_name n) n))
        in Hresult.
      destruct Hresult as [Hcanonical_element Hdemoted].
      split; [ apply simple_of_name_inj, Hdemoted | exact Hcanonical_element ].
  - exfalso.
    apply add_bottom in Hadd as (c1 & c2 & Hfirst & Hsecond & _).
    destruct Harguments as [[-> _] | [_ ->]]; destruct n; discriminate.
  - exfalso. revert Hadd.
    apply add_defined; try assumption.
    + destruct Harguments as [[-> ->] | [-> ->]];
        [ exact Hcanonical | reflexivity ].
    + destruct Harguments as [[-> ->] | [-> ->]];
        [ reflexivity | exact Hcanonical ].
    + destruct Harguments as [[-> ->] | [-> ->]]; destruct n; discriminate.
Qed.

Theorem add_name_constant : add_name_constant_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t n c Hconsistent
    Hcanonical.
  split; apply add_name_constant_either; auto.
Qed.

(* In a consistent structure the canonical simples are those that are
   their own canonical element. *)
Lemma is_canonical_iff (t : Aliases.t) (e : simple) :
  aliases_consistent t ->
  (is_canonical t e <-> canonical_element_of t e = e).
Proof.
  intros [Hcanonical _]. split; [ apply canonical_element_of_canonical | ].
  intros Hself.
  destruct (canonical_cases t e)
    as [[His_canonical _]
       | (n & canonical_element & -> & Hdemoted & _ & Hcanonical_element)];
    [ exact His_canonical | ].
  rewrite Hcanonical_element in Hself. subst canonical_element.
  pose proof (Hcanonical n _ Hdemoted) as His_canonical.
  unfold is_canonical in His_canonical.
  rewrite canonical_of_name in His_canonical.
  unfold demoted_to in Hdemoted. rewrite Hdemoted in His_canonical.
  discriminate.
Qed.

Theorem add_is_canonical : add_is_canonical_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result e Hconsistent Hcanonical1 Hcanonical2
    Hdifferent Hadd.
  pose proof
    (add_consistent A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result Hconsistent Hcanonical1
       Hcanonical2 Hadd)
    as Hconsistent'.
  rewrite (is_canonical_iff _ e Hconsistent'),
    (is_canonical_iff t e Hconsistent).
  rewrite
    (add_canonical_element A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result e Hconsistent Hcanonical1
       Hcanonical2 Hadd).
  (* The canonical element and the name demoted are the two arguments:
     different, and canonical before. *)
  assert
    (Harguments
      : Aliases.canonical_element result
        <> simple_of_name (Aliases.demoted_name result)
        ∧ is_canonical t (Aliases.canonical_element result)
        ∧ is_canonical t (simple_of_name (Aliases.demoted_name result))).
  { pose proof
      (add_result A binding_time_resolver binding_times_and_modes t
         canonical_element1 canonical_element2 result Hadd)
      as Hresult.
    destruct
      (Aliases.defined_earlier binding_time_resolver binding_times_and_modes
         canonical_element1 canonical_element2);
      destruct Hresult as [-> ->]; auto. }
  destruct Harguments as (Hapart & Hcanonical & Hdemoted).
  apply canonical_element_of_canonical in Hcanonical, Hdemoted.
  destruct
    (simple_equal (canonical_element_of t e)
       (simple_of_name (Aliases.demoted_name result)))
    eqn:Hequal.
  - apply simple_equal_eq in Hequal. split.
    + intros Hself. exfalso. subst e. rewrite Hcanonical in Hequal.
      exact (Hapart Hequal).
    + intros [Hself Hnot_demoted]. exfalso. apply Hnot_demoted.
      rewrite <- Hself. exact Hequal.
  - split; [ | tauto ].
    intros Hself. split; [ exact Hself | ].
    intros ->. rewrite Hdemoted in Hequal.
    rewrite (proj2 (simple_equal_eq _ _) eq_refl) in Hequal. discriminate.
Qed.

Theorem add_classes_uniform : add_classes_uniform_statement.
Proof.
  intros A B binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2 result property Hconsistent
    Hcanonical1 Hcanonical2 Hadd Huniform Harguments e e' Haliased.
  apply
    (add_aliased A binding_time_resolver binding_times_and_modes t
       canonical_element1 canonical_element2 result e e' Hconsistent
       Hcanonical1 Hcanonical2 Hadd)
    in Haliased as [Hbefore | [[Hfirst Hsecond] | [Hfirst Hsecond]]].
  - apply Huniform, Hbefore.
  - rewrite (Huniform _ _ Hfirst), (Huniform _ _ Hsecond). exact Harguments.
  - rewrite (Huniform _ _ Hfirst), (Huniform _ _ Hsecond).
    exact (eq_sym Harguments).
Qed.

Theorem add_name_modes_recorded : add_name_modes_recorded_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result Hrecorded Hadd.
  destruct (add_merged _ _ _ _ _ _ Hadd) as [[_ ->] | (_ & _ & Hmodes)];
    [ exact Hrecorded | ].
  intros e n name_mode Hlookup.
  destruct (Hmodes e n name_mode Hlookup) as [[_ ->] | [e' Hold]];
    [ reflexivity | exact (Hrecorded e' n name_mode Hold) ].
Qed.

Theorem add_earliest : add_earliest_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result [_ Hlisted] Hearliest Hcanonical1 Hcanonical2
    Hcomparable Hadd.
  (* The canonical element is earlier than the name demoted. *)
  assert (Hdemoted :
    Aliases.defined_earlier binding_time_resolver binding_times_and_modes
      (Aliases.canonical_element result)
      (simple_of_name (Aliases.demoted_name result))
    = true).
  { pose proof (add_result _ _ _ _ _ _ _ Hadd) as Hresult.
    destruct (Aliases.defined_earlier binding_time_resolver
                binding_times_and_modes canonical_element1
                canonical_element2) eqn:Hearlier;
      destruct Hresult as [-> ->]; [ exact Hearlier | ].
    destruct Hcomparable as [Hfirst | Hsecond]; [ congruence | exact Hsecond ].
  }
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[_ ->] | (_ & [Hdem _] & _)]; [ exact Hearliest | ].
  intros n e Hd. apply Hdem in Hd as [[-> [-> | Hl]] | (_ & _ & Hold)].
  - exact Hdemoted.
  - (* An alias of the name demoted: the order is transitive. *)
    apply (proj2 (defined_earlier_strict A binding_time_resolver
                    binding_times_and_modes _
                    (simple_of_name (Aliases.demoted_name result)) _));
      [ exact Hdemoted | ].
    apply Hearliest, Hlisted, Hl.
  - exact (Hearliest _ _ Hold).
Qed.

Theorem simples_comparable : simples_comparable_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes e e' Hne.
  destruct e as [x | s | c], e' as [y | s' | c']; try exact I;
    unfold comparable;
    try (left; reflexivity); try (right; reflexivity).
  intros Htimes.
  change (Var x) with (simple_of_name (Name_var x)).
  change (Var y) with (simple_of_name (Name_var y)).
  rewrite !defined_earlier_names.
  destruct (Z.lt_trichotomy (binding_time_in binding_times_and_modes x)
              (binding_time_in binding_times_and_modes y))
    as [Hlt | [Heq | Hgt]]; [ | contradiction | ].
  - left. apply variable_earlier. left. exact Hlt.
  - right. apply variable_earlier. left. lia.
Qed.

Theorem add_mentioned : add_mentioned_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result n Hadd Hmention.
  destruct (add_merged _ _ _ _ _ _ Hadd)
    as [[_ Hsame] | (_ & [Hdem Hlis] & _)].
  { rewrite Hsame in Hmention. left. exact Hmention. }
  (* It was mentioned, or is the canonical element or the name demoted. *)
  assert (Hgoal :
    aliases_mention t n
    ∨ simple_of_name n = Aliases.canonical_element result
    ∨ n = Aliases.demoted_name result).
  { destruct Hmention as [[e Hd] | [[n' Hd] | [[e Hl] | [n' Hl]]]].
    - apply Hdem in Hd as [[_ [-> | Hl]] | (_ & _ & Hold)].
      + auto.
      + left. right. right. left. eexists. exact Hl.
      + left. left. exists e. exact Hold.
    - apply Hdem in Hd as [[Heq _] | (_ & _ & Hold)].
      + auto.
      + left. right. left. exists n'. exact Hold.
    - apply Hlis in Hl as [[_ [-> | [Hl | Hl]]] | (_ & _ & Hold)].
      + auto.
      + left. right. right. left. eexists. exact Hl.
      + left. right. right. left. eexists. exact Hl.
      + left. right. right. left. exists e. exact Hold.
    - apply Hlis in Hl as [[Heq _] | (_ & _ & Hold)].
      + auto.
      + left. right. right. right. exists n'. exact Hold. }
  destruct Hgoal as [Hbefore | [Hcanonical | ->]]; [ left; exact Hbefore | | ];
    destruct (add_arguments _ _ _ _ _ _ Hadd) as [[Hc Hd] | [Hc Hd]];
    rewrite <- ?Hc, <- ?Hd; auto.
Qed.

(* The order reads the typing environment only through the binding times
   it gives the two simples. *)
Lemma defined_earlier_coincide {A A' : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (binding_times_and_modes'
      : Keyed.finite_map name (A' * Binding_time.With_name_mode.t))
    (e e' : simple) :
  Aliases.binding_time_and_name_mode binding_times_and_modes e
  = Aliases.binding_time_and_name_mode binding_times_and_modes' e ->
  Aliases.binding_time_and_name_mode binding_times_and_modes e'
  = Aliases.binding_time_and_name_mode binding_times_and_modes' e' ->
  Aliases.defined_earlier binding_time_resolver binding_times_and_modes e e'
  = Aliases.defined_earlier binding_time_resolver binding_times_and_modes'
      e e'.
Proof.
  intros He He'.
  destruct e as [x | s | c], e' as [y | s' | c']; try reflexivity.
  - change (Var x) with (simple_of_name (Name_var x)).
    change (Var y) with (simple_of_name (Name_var y)).
    rewrite !defined_earlier_names, !name_defined_earlier_variables.
    unfold binding_time_in. cbn [simple_of_name] in *. rewrite He, He'.
    reflexivity.
  - (* Two symbols: neither is earlier, whatever the environment. *)
    change (Symbol s) with (simple_of_name (Name_symbol s)).
    change (Symbol s') with (simple_of_name (Name_symbol s')).
    rewrite !defined_earlier_names.
    rewrite !(proj2 (symbol_earlier _ _ _ s' 1%positive (Name_symbol s))).
    reflexivity.
Qed.

Theorem well_formed_coincide : well_formed_coincide_statement.
Proof.
  intros A A' binding_time_resolver binding_times_and_modes
    binding_times_and_modes' t Hsame (Hconsistent & Hearliest & Hrecorded).
  split; [ exact Hconsistent | split ].
  - intros n e Hd.
    rewrite <- (defined_earlier_coincide binding_time_resolver
                  binding_times_and_modes binding_times_and_modes').
    + exact (Hearliest n e Hd).
    + destruct (name_of_simple e) as [n' | ] eqn:Hname.
      * apply name_of_simple_some in Hname. subst e. apply Hsame.
        right. left. exists n. exact Hd.
      * apply name_of_simple_none in Hname as [c ->]. reflexivity.
    + apply Hsame. left. exists e. exact Hd.
  - intros e n name_mode Hlookup.
    rewrite (Hrecorded e n name_mode Hlookup).
    unfold Aliases.compute_name_mode_unscoped.
    rewrite Hsame; [ reflexivity | ].
    right. right. left. exists e. apply mem_lookup. exists name_mode.
    exact Hlookup.
Qed.

Theorem name_modes_recorded_coincide :
  name_modes_recorded_coincide_statement.
Proof.
  intros A A' binding_times_and_modes binding_times_and_modes' t Hsame
    Hrecorded e n name_mode Hlookup.
  rewrite (Hrecorded e n name_mode Hlookup). apply Hsame.
  right. right. left. exists e. apply mem_lookup. exists name_mode.
  exact Hlookup.
Qed.

Theorem empty_well_formed : empty_well_formed_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes.
  assert (Hdemoted :
    ∀ (n : name) (e : simple), ~ demoted_to Aliases.empty n e).
  { intros n e Hd. discriminate Hd. }
  assert (Hlisted :
    ∀ (e : simple) (n : name) (name_mode : Name_mode.t),
      Keyed.lookup name_eqb n
        (Aliases.get_aliases_of_canonical_element Aliases.empty e)
      <> Some name_mode).
  { intros [x | s | c] n name_mode Hlookup; discriminate Hlookup. }
  split; [ split | split ].
  - intros n e Hd. destruct (Hdemoted n e Hd).
  - intros n e. split.
    + intros Hd. destruct (Hdemoted n e Hd).
    + intros Hl. apply mem_lookup in Hl as [name_mode Hl].
      destruct (Hlisted e n name_mode Hl).
  - intros n e Hd. destruct (Hdemoted n e Hd).
  - intros e n name_mode Hl. destruct (Hlisted e n name_mode Hl).
Qed.

Theorem add_well_formed : add_well_formed_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t canonical_element1
    canonical_element2 result (Hconsistent & Hearliest & Hrecorded)
    Hcanonical1 Hcanonical2 Hcomparable Hadd.
  split; [ | split ].
  - exact (add_consistent _ _ _ _ _ _ _ Hconsistent Hcanonical1 Hcanonical2
             Hadd).
  - exact (add_earliest _ _ _ _ _ _ _ Hconsistent Hearliest Hcanonical1
             Hcanonical2 Hcomparable Hadd).
  - exact (add_name_modes_recorded _ _ _ _ _ _ _ Hrecorded Hadd).
Qed.

(* ======================================================================== *)
(* The aliases of an element                                                *)
(* ======================================================================== *)

Lemma get_aliases_eq (t : Aliases.t) (element : simple) :
  Aliases.get_aliases t element
  = Aliases.Alias_set.create_aliases_of_element
      (canonical_element_of t element)
      (Aliases.Aliases_of_canonical_element.all
         (Aliases.get_aliases_of_canonical_element t
            (canonical_element_of t element))).
Proof. reflexivity. Qed.

Lemma alias_set_has_name (set : Aliases.Alias_set.t) (n : name) :
  alias_set_has set (simple_of_name n)
  <-> n ∈ Aliases.Alias_set.set_names set.
Proof. destruct n; reflexivity. Qed.

Lemma set_names_create (canonical_element : simple) (names : list name) :
  Aliases.Alias_set.set_names
    (Aliases.Alias_set.create_aliases_of_element canonical_element names)
  = match name_of_simple canonical_element with
    | Some canonical_name => canonical_name :: names
    | None => names
    end.
Proof. destruct canonical_element; reflexivity. Qed.

Lemma set_const_create (canonical_element : simple) (names : list name)
    (c : const) :
  Aliases.Alias_set.set_const
    (Aliases.Alias_set.create_aliases_of_element canonical_element names)
  = Some c
  <-> canonical_element = Const c.
Proof.
  destruct canonical_element; simpl; split; intros Heq;
    solve [ discriminate | congruence ].
Qed.

(* The names of the list of aliases of a canonical element are those
   listed for it. *)
Lemma in_all_listed (t : Aliases.t) (canonical_element : simple) (n : name) :
  n ∈ Aliases.Aliases_of_canonical_element.all
        (Aliases.get_aliases_of_canonical_element t canonical_element)
  <-> listed_alias t canonical_element n.
Proof.
  rewrite <- (keyed_among_in name name_eqb name_eqb_eq), among_all.
  reflexivity.
Qed.

Theorem get_aliases : get_aliases_statement.
Proof.
  intros t element e Hconsistent. rewrite get_aliases_eq. unfold aliased.
  pose proof (canonical_element_canonical t element Hconsistent)
    as Hcanonical.
  generalize dependent (canonical_element_of t element).
  intros canonical_element Hcanonical.
  destruct (name_of_simple e) as [n | ] eqn:Hname.
  - apply name_of_simple_some in Hname. subst e.
    rewrite alias_set_has_name, set_names_create. split.
    + intros Hin.
      assert (Hlisted :
        n ∈ Aliases.Aliases_of_canonical_element.all
              (Aliases.get_aliases_of_canonical_element t canonical_element)
        -> canonical_element_of t (simple_of_name n) = canonical_element).
      { intros Hall. apply canonical_element_of_demoted.
        apply Hconsistent, in_all_listed, Hall. }
      destruct (name_of_simple canonical_element) as [canonical_name | ]
        eqn:Hcanonical_name; [ | exact (Hlisted Hin) ].
      destruct Hin as [<- | Hin]; [ | exact (Hlisted Hin) ].
      apply name_of_simple_some in Hcanonical_name. subst canonical_element.
      exact (canonical_element_of_canonical t _ Hcanonical).
    + intros Hof.
      destruct (canonical_cases t (simple_of_name n))
        as [[_ Hself] | (n' & e & Hn & Hd & _ & He)].
      * rewrite Hself in Hof. subst canonical_element.
        rewrite name_of_simple_of_name. left. reflexivity.
      * apply simple_of_name_inj in Hn. subst n'. rewrite He in Hof. subst e.
        assert (Hall :
          n ∈ Aliases.Aliases_of_canonical_element.all
                (Aliases.get_aliases_of_canonical_element t
                   canonical_element)).
        { apply in_all_listed, Hconsistent, Hd. }
        destruct (name_of_simple canonical_element);
          [ right; exact Hall | exact Hall ].
  - apply name_of_simple_none in Hname as [c ->].
    change (alias_set_has ?set (Const c))
      with (Aliases.Alias_set.set_const set = Some c).
    rewrite set_const_create.
    split; intros Heq; [ rewrite Heq; reflexivity | ].
    symmetry. exact Heq.
Qed.

Theorem respected_aliases : respected_aliases_statement.
Proof.
  intros t vars element e Hrespected Hhas.
  rewrite get_aliases_eq in Hhas.
  rewrite <- (respected_canonical_element_of t vars element Hrespected).
  generalize dependent (canonical_element_of t element).
  intros canonical_element Hhas.
  destruct (name_of_simple e) as [n | ] eqn:Hname.
  - apply name_of_simple_some in Hname. subst e.
    rewrite alias_set_has_name, set_names_create in Hhas.
    assert (Hlisted :
      n ∈ Aliases.Aliases_of_canonical_element.all
            (Aliases.get_aliases_of_canonical_element t canonical_element)
      -> eval_simple vars (simple_of_name n)
         = eval_simple vars canonical_element).
    { intros Hall. apply in_all_listed in Hall.
      exact (proj2 Hrespected canonical_element n Hall). }
    destruct (name_of_simple canonical_element) as [canonical_name | ]
      eqn:Hcanonical_name; [ | exact (Hlisted Hhas) ].
    destruct Hhas as [<- | Hin]; [ | exact (Hlisted Hin) ].
    apply name_of_simple_some in Hcanonical_name. rewrite Hcanonical_name.
    reflexivity.
  - apply name_of_simple_none in Hname as [c ->].
    change (Aliases.Alias_set.set_const
              (Aliases.Alias_set.create_aliases_of_element canonical_element
                 (Aliases.Aliases_of_canonical_element.all
                    (Aliases.get_aliases_of_canonical_element t
                       canonical_element)))
            = Some c) in Hhas.
    apply set_const_create in Hhas. rewrite Hhas. reflexivity.
Qed.

(* ======================================================================== *)
(* The functions read the binding times and the name modes only             *)
(* ======================================================================== *)

Lemma fold_left_alike {A B : Type} (f f' : A -> B -> A) (l : list B)
    (init : A) :
  (∀ (a : A) (b : B), f a b = f' a b) ->
  List.fold_left f l init = List.fold_left f' l init.
Proof.
  intros Hf. revert init.
  induction l as [ | b l IH]; intros init; simpl; [ reflexivity | ].
  rewrite Hf. apply IH.
Qed.

Lemma forallb_alike {A : Type} (f f' : A -> bool) (l : list A) :
  (∀ a : A, f a = f' a) -> forallb f l = forallb f' l.
Proof.
  intros Hf. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hf, IH. reflexivity.
Qed.

Lemma find_earliest_candidates_alike
    (t : Aliases.Aliases_of_canonical_element.t)
    (filter_by_scope filter_by_scope' : Name_mode.t -> list name -> list name)
    (min_name_mode : Name_mode.t) :
  (∀ (name_mode : Name_mode.t) (names : list name),
    filter_by_scope name_mode names = filter_by_scope' name_mode names) ->
  Aliases.Aliases_of_canonical_element.find_earliest_candidates t
    filter_by_scope min_name_mode
  = Aliases.Aliases_of_canonical_element.find_earliest_candidates t
      filter_by_scope' min_name_mode.
Proof.
  intros Hfilter.
  unfold Aliases.Aliases_of_canonical_element.find_earliest_candidates.
  apply fold_left_alike. intros res_opt order. rewrite Hfilter. reflexivity.
Qed.

Section Binding_times_alike.

Context {A B : Type}
  (resolver resolver' : name -> Binding_time.With_name_mode.t)
  (times : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
  (times' : Keyed.finite_map name (B * Binding_time.With_name_mode.t)).

Hypothesis resolvers_alike : ∀ n : name, resolver n = resolver' n.

Hypothesis times_alike : binding_times_alike times times'.

Lemma binding_time_and_name_mode_alike (elt : simple) :
  Aliases.binding_time_and_name_mode times elt
  = Aliases.binding_time_and_name_mode times' elt.
Proof.
  destruct elt as [x | sym | c]; simpl; try reflexivity.
  pose proof (times_alike (Name_var x)) as Hx.
  destruct (Keyed.lookup name_eqb (Name_var x) times) as [[a bt] | ],
    (Keyed.lookup name_eqb (Name_var x) times') as [[b bt'] | ];
    simpl in Hx; try discriminate; [ | reflexivity ].
  injection Hx as ->. reflexivity.
Qed.

Lemma name_defined_earlier_alike (alias than : name) :
  Aliases.name_defined_earlier resolver times alias than
  = Aliases.name_defined_earlier resolver' times' alias than.
Proof.
  unfold Aliases.name_defined_earlier.
  pose proof (times_alike alias) as Halias.
  pose proof (times_alike than) as Hthan.
  rewrite !resolvers_alike.
  destruct (Keyed.lookup name_eqb alias times) as [[a bt] | ],
    (Keyed.lookup name_eqb alias times') as [[b bt'] | ];
    simpl in Halias; try discriminate;
    destruct (Keyed.lookup name_eqb than times) as [[a2 bt2] | ],
      (Keyed.lookup name_eqb than times') as [[b2 bt2'] | ];
    simpl in Hthan; try discriminate;
    repeat
      match goal with
      | H : Some _ = Some _ |- _ => injection H as ->
      end;
    reflexivity.
Qed.

Lemma defined_earlier_alike_here (alias than : simple) :
  Aliases.defined_earlier resolver times alias than
  = Aliases.defined_earlier resolver' times' alias than.
Proof.
  unfold Aliases.defined_earlier.
  destruct (name_of_simple than); [ | reflexivity ].
  destruct (name_of_simple alias); [ | reflexivity ].
  apply name_defined_earlier_alike.
Qed.

Lemma compute_name_mode_unscoped_alike (elt : simple) :
  Aliases.compute_name_mode_unscoped times elt
  = Aliases.compute_name_mode_unscoped times' elt.
Proof.
  unfold Aliases.compute_name_mode_unscoped.
  rewrite binding_time_and_name_mode_alike. reflexivity.
Qed.

Lemma compute_name_mode_alike (elt : simple)
    (min_binding_time : Binding_time.t) :
  Aliases.compute_name_mode times elt min_binding_time
  = Aliases.compute_name_mode times' elt min_binding_time.
Proof.
  unfold Aliases.compute_name_mode.
  rewrite binding_time_and_name_mode_alike. reflexivity.
Qed.

Lemma add_alias_between_canonical_elements_alike (t : Aliases.t)
    (canonical_element to_be_demoted : simple) :
  Aliases.add_alias_between_canonical_elements times t canonical_element
    to_be_demoted
  = Aliases.add_alias_between_canonical_elements times' t canonical_element
      to_be_demoted.
Proof.
  unfold Aliases.add_alias_between_canonical_elements.
  rewrite compute_name_mode_unscoped_alike. reflexivity.
Qed.

Lemma add_alike_here (t : Aliases.t)
    (canonical_element1 canonical_element2 : simple) :
  Aliases.add resolver times t canonical_element1 canonical_element2
  = Aliases.add resolver' times' t canonical_element1 canonical_element2.
Proof.
  unfold Aliases.add, Aliases.add0, Aliases.add_alias,
    Aliases.choose_canonical_element_to_be_demoted.
  rewrite defined_earlier_alike_here.
  destruct
    (Aliases.defined_earlier resolver' times' canonical_element1
       canonical_element2);
    rewrite add_alias_between_canonical_elements_alike; reflexivity.
Qed.

Lemma find_earliest_alias_alike (t : Aliases.t) (canonical_element : simple)
    (min_binding_time : Binding_time.t) (min_name_mode : Name_mode.t) :
  Aliases.find_earliest_alias resolver times t canonical_element
    min_binding_time min_name_mode
  = Aliases.find_earliest_alias resolver' times' t canonical_element
      min_binding_time min_name_mode.
Proof.
  unfold Aliases.find_earliest_alias.
  rewrite
    (find_earliest_candidates_alike _ _
       (fun name_mode names =>
          if Name_mode.equal name_mode Name_mode.in_types
          then names
          else
            filter
              (fun name =>
                 Name_mode.equal name_mode
                   (Aliases.compute_name_mode times' (simple_of_name name)
                      min_binding_time))
              names)).
  - destruct
      (Aliases.Aliases_of_canonical_element.find_earliest_candidates _ _ _)
      as [[ | first rest] | ];
      try reflexivity.
    do 2 f_equal. apply fold_left_alike. intros min_elt elt.
    rewrite name_defined_earlier_alike. reflexivity.
  - intros name_mode names.
    destruct (Name_mode.equal name_mode Name_mode.in_types);
      [ reflexivity | ].
    apply filter_ext. intros name. rewrite compute_name_mode_alike.
    reflexivity.
Qed.

Lemma get_canonical_element_exn_alike_here (t : Aliases.t) (element : simple)
    (elt_name_mode min_name_mode : Name_mode.t)
    (min_binding_time : Binding_time.t) :
  Aliases.get_canonical_element_exn resolver times t element elt_name_mode
    min_name_mode min_binding_time
  = Aliases.get_canonical_element_exn resolver' times' t element
      elt_name_mode min_name_mode min_binding_time.
Proof.
  unfold Aliases.get_canonical_element_exn.
  destruct (Aliases.canonical t element) as [ | canonical_element];
    cbv beta iota zeta;
    rewrite ?compute_name_mode_alike, ?find_earliest_alias_alike;
    reflexivity.
Qed.

Lemma invariant_alike_here (t : Aliases.t) :
  Aliases.invariant resolver times t = Aliases.invariant resolver' times' t.
Proof.
  unfold Aliases.invariant. cbv zeta. f_equal.
  apply forallb_alike. intros [canonical_element aliases]. f_equal.
  apply forallb_alike. intros alias. apply name_defined_earlier_alike.
Qed.

Lemma well_formed_alike_here (t : Aliases.t) :
  aliases_well_formed resolver times t ->
  aliases_well_formed resolver' times' t.
Proof.
  intros (Hconsistent & Hearliest & Hrecorded).
  split; [ exact Hconsistent | split ].
  - intros n canonical_element Hdemoted.
    rewrite <- defined_earlier_alike_here.
    exact (Hearliest n canonical_element Hdemoted).
  - intros canonical_element n name_mode Hlookup.
    rewrite <- compute_name_mode_unscoped_alike.
    exact (Hrecorded canonical_element n name_mode Hlookup).
Qed.

End Binding_times_alike.

Theorem defined_earlier_alike : defined_earlier_alike_statement.
Proof.
  intros A B resolver resolver' times times' alias than Hresolvers Htimes.
  exact (defined_earlier_alike_here resolver resolver' times times'
    Hresolvers Htimes alias than).
Qed.

Theorem add_alike : add_alike_statement.
Proof.
  intros A B resolver resolver' times times' t canonical_element1
    canonical_element2 Hresolvers Htimes.
  exact (add_alike_here resolver resolver' times times' Hresolvers Htimes t
    canonical_element1 canonical_element2).
Qed.

Theorem get_canonical_element_exn_alike :
  get_canonical_element_exn_alike_statement.
Proof.
  intros A B resolver resolver' times times' t element elt_name_mode
    min_name_mode min_binding_time Hresolvers Htimes.
  exact (get_canonical_element_exn_alike_here resolver resolver' times
    times' Hresolvers Htimes t element elt_name_mode min_name_mode
    min_binding_time).
Qed.

Theorem invariant_alike : invariant_alike_statement.
Proof.
  intros A B resolver resolver' times times' t Hresolvers Htimes.
  exact (invariant_alike_here resolver resolver' times times' Hresolvers
    Htimes t).
Qed.

Theorem well_formed_alike : well_formed_alike_statement.
Proof.
  intros A B resolver resolver' times times' t Hresolvers Htimes. split.
  - exact (well_formed_alike_here resolver resolver' times times' Hresolvers
      Htimes t).
  - apply (well_formed_alike_here resolver' resolver times' times).
    + intros n. symmetry. apply Hresolvers.
    + intros n. symmetry. apply Htimes.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* Respect of a structure written out as its lists is a property of the
   pairs in them. *)
Lemma respected_of_pairs (t : Aliases.t) (vars : env) :
  (∀ (n : name) (canonical_element : simple),
    (n, canonical_element) ∈ Aliases.canonical_elements t ->
    value_of_name vars n = eval_simple vars canonical_element) ->
  (∀ (canonical_name : name)
     (aliases : Aliases.Aliases_of_canonical_element.t) (n : name)
     (name_mode : Name_mode.t),
    (canonical_name, aliases) ∈ Aliases.aliases_of_canonical_names t ->
    (n, name_mode) ∈ aliases ->
    value_of_name vars n = value_of_name vars canonical_name) ->
  (∀ (c : const) (aliases : Aliases.Aliases_of_canonical_element.t)
     (n : name) (name_mode : Name_mode.t),
    (c, aliases) ∈ Aliases.aliases_of_consts t ->
    (n, name_mode) ∈ aliases ->
    value_of_name vars n = Some (Value_const c)) ->
  aliases_respected t vars.
Proof.
  intros Hcanonical Hnames Hconsts. split.
  - intros n canonical_element Hdemoted. apply Hcanonical.
    exact (keyed_lookup_in name_eqb name_eqb_eq _ _ _ Hdemoted).
  - intros canonical_element n Hlisted. unfold listed_alias in Hlisted.
    apply mem_lookup in Hlisted as [name_mode Hlookup].
    apply (keyed_lookup_in name_eqb name_eqb_eq) in Hlookup.
    unfold Aliases.get_aliases_of_canonical_element in Hlookup.
    destruct canonical_element as [x | s | c].
    + destruct (Keyed.lookup name_eqb (Name_var x)
                  (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Haliases; [ | contradiction ].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Haliases.
      exact (Hnames _ _ _ _ Haliases Hlookup).
    + destruct (Keyed.lookup name_eqb (Name_symbol s)
                  (Aliases.aliases_of_canonical_names t))
        as [aliases | ] eqn:Haliases; [ | contradiction ].
      apply (keyed_lookup_in name_eqb name_eqb_eq) in Haliases.
      exact (Hnames _ _ _ _ Haliases Hlookup).
    + destruct (Keyed.lookup const_eqb c (Aliases.aliases_of_consts t))
        as [aliases | ] eqn:Haliases; [ | contradiction ].
      apply (keyed_lookup_in const_eqb const_eqb_eq) in Haliases.
      exact (Hconsts _ _ _ _ Haliases Hlookup).
Qed.

(* Goes through the pairs of a structure written out as its lists. *)
Ltac respected_by_pairs :=
  apply respected_of_pairs; simpl; intros;
  repeat
    match goal with
    | H : _ ∨ _ |- _ => destruct H
    | H : False |- _ => contradiction
    | H : (_, _) = (_, _) |- _ => injection H as <- <-
    | H : _ ∈ _ |- _ => simpl in H
    end;
  subst; reflexivity.

Module Examples.

(* Three variables, a symbol and two constants. *)
Definition w : variable := 1%positive.
Definition x : variable := 2%positive.
Definition y : variable := 3%positive.
Definition s : symbol := 1%positive.
Definition s' : symbol := 2%positive.
Definition five : const := Tagged_immediate 5.
Definition six : const := Tagged_immediate 6.

Definition entry (time : Binding_time.t) (name_mode : Name_mode.t)
    : unit * Binding_time.With_name_mode.t :=
  (tt, Binding_time.With_name_mode.create time name_mode).

(* A typing environment that bound [w], then [x], then [y], all three with
   a value at run time. *)
Definition times
    : Keyed.finite_map name (unit * Binding_time.With_name_mode.t) :=
  [(Name_var w, entry 3 Name_mode.Normal);
   (Name_var x, entry 4 Name_mode.Normal);
   (Name_var y, entry 5 Name_mode.Normal)].

Definition resolver (n : name) : Binding_time.With_name_mode.t :=
  Binding_time.With_name_mode.imported_variables.

(* ------------------------------------------------------------------------ *)
(* Two variables                                                            *)
(* ------------------------------------------------------------------------ *)

(* [y] is demoted to [x], which was bound first. *)
Definition x_y : Aliases.t :=
  Aliases.make [(Name_var y, Var x)]
    [(Name_var x, [(Name_var y, Name_mode.Normal)])] [].

(* Whichever of the two is given first, the later is demoted to the
   earlier. *)
Example add_two_variables :
  Aliases.add resolver times Aliases.empty (Var y) (Var x)
  = Some (Or_bottom.Ok (Aliases.make_add_result x_y (Name_var y) (Var x)))
  ∧ Aliases.add resolver times Aliases.empty (Var x) (Var y)
    = Some (Or_bottom.Ok (Aliases.make_add_result x_y (Name_var y) (Var x))).
Proof. split; reflexivity. Qed.

Example canonical_of_two_variables :
  Aliases.get_canonical_ignoring_name_mode x_y (Name_var y) = Var x
  ∧ Aliases.get_canonical_ignoring_name_mode x_y (Name_var x) = Var x
  ∧ Aliases.get_canonical_ignoring_name_mode x_y (Name_var w) = Var w.
Proof. repeat split. Qed.

(* An environment in which the two have the same value respects the
   structure... *)
Example two_variables_respected :
  aliases_respected x_y [(x, Value_const five); (y, Value_const five)].
Proof. respected_by_pairs. Qed.

(* ... and one in which they have different values does not. *)
Example two_variables_not_respected :
  ~ aliases_respected x_y [(x, Value_const five); (y, Value_const six)].
Proof.
  intros [Hcanonical _].
  discriminate (Hcanonical (Name_var y) (Var x) eq_refl).
Qed.

(* An environment that binds neither respects the structure; one that
   binds only one of them does not. *)
Example two_unbound_variables_respected : aliases_respected x_y [].
Proof. respected_by_pairs. Qed.

Example one_unbound_variable_not_respected :
  ~ aliases_respected x_y [(x, Value_const five)].
Proof.
  intros [Hcanonical _].
  discriminate (Hcanonical (Name_var y) (Var x) eq_refl).
Qed.

(* The structure is well formed for that typing environment: the
   statements that ask for it are about something. *)
Example two_variables_well_formed : aliases_well_formed resolver times x_y.
Proof.
  (* What is demoted, and what is listed. *)
  assert (Hdemoted :
    ∀ (n : name) (canonical_element : simple),
      demoted_to x_y n canonical_element ->
      n = Name_var y ∧ canonical_element = Var x).
  { intros n canonical_element Hdemoted. unfold demoted_to in Hdemoted.
    cbn [x_y Aliases.canonical_elements Keyed.lookup] in Hdemoted.
    destruct (name_eqb n (Name_var y)) eqn:Hn; [ | discriminate Hdemoted ].
    apply name_eqb_eq in Hn. injection Hdemoted as <-. auto. }
  assert (Hlisted :
    ∀ (canonical_element : simple) (n : name) (name_mode : Name_mode.t),
      Keyed.lookup name_eqb n
        (Aliases.get_aliases_of_canonical_element x_y canonical_element)
      = Some name_mode ->
      n = Name_var y ∧ canonical_element = Var x
      ∧ name_mode = Name_mode.Normal).
  { intros canonical_element n name_mode Hlookup.
    assert (Hx : canonical_element = Var x).
    { destruct canonical_element as [v | sym | c];
        try discriminate Hlookup.
      unfold Aliases.get_aliases_of_canonical_element in Hlookup.
      cbn [x_y Aliases.aliases_of_canonical_names Keyed.lookup] in Hlookup.
      destruct (name_eqb (Name_var v) (Name_var x)) eqn:Hv;
        [ | discriminate Hlookup ].
      apply name_eqb_eq in Hv. injection Hv as ->. reflexivity. }
    subst canonical_element.
    change (Keyed.lookup name_eqb n [(Name_var y, Name_mode.Normal)]
            = Some name_mode) in Hlookup.
    cbn [Keyed.lookup] in Hlookup.
    destruct (name_eqb n (Name_var y)) eqn:Hn; [ | discriminate Hlookup ].
    apply name_eqb_eq in Hn. injection Hlookup as <-. auto. }
  split; [ split | split ].
  - intros n canonical_element Hd.
    destruct (Hdemoted n canonical_element Hd) as [-> ->]. reflexivity.
  - intros n canonical_element. split.
    + intros Hd. destruct (Hdemoted n canonical_element Hd) as [-> ->].
      reflexivity.
    + intros Hl. apply mem_lookup in Hl as [name_mode Hl].
      destruct (Hlisted canonical_element n name_mode Hl) as (-> & -> & _).
      reflexivity.
  - intros n canonical_element Hd.
    destruct (Hdemoted n canonical_element Hd) as [-> ->]. reflexivity.
  - intros canonical_element n name_mode Hl.
    destruct (Hlisted canonical_element n name_mode Hl) as (-> & -> & ->).
    reflexivity.
Qed.

(* The same from the statements about [add]: the structure that knows
   nothing is well formed, [x] and [y] are canonical in it, and one is
   earlier than the other. *)
Example two_variables_well_formed_by_add :
  aliases_well_formed resolver times x_y.
Proof.
  apply (add_well_formed unit resolver times Aliases.empty (Var y) (Var x)
           (Aliases.make_add_result x_y (Name_var y) (Var x))).
  - apply empty_well_formed.
  - reflexivity.
  - reflexivity.
  - right. reflexivity.
  - reflexivity.
Qed.

(* After it, [x] and [y] are known to be equal, and [w] is known to be
   equal only to itself. *)
Example two_variables_aliased :
  aliased x_y (Var x) (Var y) ∧ ~ aliased x_y (Var w) (Var x).
Proof. split; [ reflexivity | discriminate ]. Qed.

(* The aliases of [y]: itself and [x]. *)
Example aliases_of_two_variables :
  Aliases.get_aliases x_y (Var y)
  = Aliases.Alias_set.make None [Name_var x; Name_var y].
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------------ *)
(* A variable and a constant                                                *)
(* ------------------------------------------------------------------------ *)

(* The constant is the canonical element, whichever is given first. *)
Definition x_five : Aliases.t :=
  Aliases.make [(Name_var x, Const five)] []
    [(five, [(Name_var x, Name_mode.Normal)])].

Example add_variable_constant :
  Aliases.add resolver times Aliases.empty (Var x) (Const five)
  = Some
      (Or_bottom.Ok (Aliases.make_add_result x_five (Name_var x) (Const five)))
  ∧ Aliases.add resolver times Aliases.empty (Const five) (Var x)
    = Some
        (Or_bottom.Ok
           (Aliases.make_add_result x_five (Name_var x) (Const five))).
Proof. split; reflexivity. Qed.

(* The variable must hold the constant: not another, and not nothing. *)
Example constant_respected :
  aliases_respected x_five [(x, Value_const five)].
Proof. respected_by_pairs. Qed.

Example constant_not_respected :
  ~ aliases_respected x_five [(x, Value_const six)]
  ∧ ~ aliases_respected x_five [].
Proof.
  split; intros [Hcanonical _];
    discriminate (Hcanonical (Name_var x) (Const five) eq_refl).
Qed.

(* A class with a constant and two variables: [y] and [x] are aliases,
   then [x] is the constant. Both variables now have it as their
   canonical element. *)
Example add_constant_to_class :
  ∃ t : Aliases.t,
    Aliases.add resolver times x_y (Var x) (Const five)
    = Some (Or_bottom.Ok (Aliases.make_add_result t (Name_var x) (Const five)))
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var y) = Const five
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var x) = Const five.
Proof. eexists. repeat split. Qed.

(* Two different constants are never equal: bottom. *)
Example add_two_constants :
  Aliases.add resolver times Aliases.empty (Const five) (Const six)
  = Some Or_bottom.Bottom.
Proof. reflexivity. Qed.

(* ------------------------------------------------------------------------ *)
(* A variable and a symbol                                                  *)
(* ------------------------------------------------------------------------ *)

(* The symbol is the canonical element, whichever is given first. *)
Definition x_s : Aliases.t :=
  Aliases.make [(Name_var x, Symbol s)]
    [(Name_symbol s, [(Name_var x, Name_mode.Normal)])] [].

Example add_variable_symbol :
  Aliases.add resolver times Aliases.empty (Var x) (Symbol s)
  = Some (Or_bottom.Ok (Aliases.make_add_result x_s (Name_var x) (Symbol s)))
  ∧ Aliases.add resolver times Aliases.empty (Symbol s) (Var x)
    = Some
        (Or_bottom.Ok (Aliases.make_add_result x_s (Name_var x) (Symbol s))).
Proof. split; reflexivity. Qed.

(* The variable must hold the symbol itself, the pointer. A closure made
   by a [Let] is another value, whatever the symbol is bound to. *)
Example symbol_respected : aliases_respected x_s [(x, Value_symbol s)].
Proof. respected_by_pairs. Qed.

Example symbol_not_respected :
  ~ aliases_respected x_s [(x, Value_closure 1%positive [])].
Proof.
  intros [Hcanonical _].
  discriminate (Hcanonical (Name_var x) (Symbol s) eq_refl).
Qed.

(* Two symbols are not ordered. [add] demotes the first of the two it is
   given, and the result does not pass the check that the OCaml makes
   under its flag for invariants. *)
Example add_two_symbols :
  ∃ t : Aliases.t,
    Aliases.add resolver times Aliases.empty (Symbol s) (Symbol s')
    = Some
        (Or_bottom.Ok (Aliases.make_add_result t (Name_symbol s) (Symbol s')))
    ∧ Aliases.invariant resolver times t = false.
Proof. eexists. split; reflexivity. Qed.

(* ------------------------------------------------------------------------ *)
(* The canonical element at a name mode                                     *)
(* ------------------------------------------------------------------------ *)

(* A typing environment in which [w], bound first, exists only in types,
   and [x] and [y] have values at run time. *)
Definition times_in_types
    : Keyed.finite_map name (unit * Binding_time.With_name_mode.t) :=
  [(Name_var w, entry 3 Name_mode.In_types);
   (Name_var x, entry 4 Name_mode.Normal);
   (Name_var y, entry 5 Name_mode.Normal)].

(* [x] and [y] are demoted to [w]. *)
Definition w_x_y : Aliases.t :=
  Aliases.make [(Name_var y, Var w); (Name_var x, Var w)]
    [(Name_var w,
      [(Name_var y, Name_mode.Normal); (Name_var x, Name_mode.Normal)])]
    [].

Example add_in_types :
  ∃ t : Aliases.t,
    Aliases.add resolver times_in_types Aliases.empty (Var x) (Var w)
    = Some (Or_bottom.Ok (Aliases.make_add_result t (Name_var x) (Var w)))
    ∧ ∃ t' : Aliases.t,
        Aliases.add resolver times_in_types t (Var y) (Var w)
        = Some (Or_bottom.Ok (Aliases.make_add_result t' (Name_var y) (Var w)))
        ∧ ∀ n : name,
            Aliases.get_canonical_ignoring_name_mode t' n
            = Aliases.get_canonical_ignoring_name_mode w_x_y n.
Proof.
  eexists. split; [ reflexivity | ]. eexists. split; [ reflexivity | ].
  intros n. unfold Aliases.get_canonical_ignoring_name_mode.
  rewrite !canonical_of_name. simpl.
  destruct (name_eqb n (Name_var y)); reflexivity.
Qed.

(* The canonical element of [y] is [w], which has no value at run time.
   Asked for a name that has one, the answer is [x], the earliest alias
   that has; in a scope where [x] is not (only what was bound at time 5
   or later is), it is [y] itself. Asked for a name that may occur in
   types, the answer is [w]. *)
Example canonical_at_normal :
  Aliases.get_canonical_element_exn resolver times_in_types w_x_y (Var y)
    Name_mode.Normal Name_mode.Normal 3
  = Some (Var x)
  ∧ Aliases.get_canonical_element_exn resolver times_in_types w_x_y (Var y)
      Name_mode.Normal Name_mode.Normal 5
    = Some (Var y)
  ∧ Aliases.get_canonical_element_exn resolver times_in_types w_x_y (Var y)
      Name_mode.Normal Name_mode.In_types 3
    = Some (Var w).
Proof. repeat split. Qed.

(* Where no alias may be used, there is no answer: here nothing bound
   before time 6 is in scope. *)
Example canonical_at_normal_none :
  Aliases.get_canonical_element_exn resolver times_in_types w_x_y (Var y)
    Name_mode.Normal Name_mode.Normal 6
  = None.
Proof. reflexivity. Qed.

(* The same with [y] a name for debugging information. Asked for a name
   that may be used there, the answer for [x] is [y], recorded as
   [Phantom], though [x] itself, recorded as [Normal] and bound earlier,
   may be used there too: the aliases of the least name mode that will
   do come first. *)
Definition times_phantom
    : Keyed.finite_map name (unit * Binding_time.With_name_mode.t) :=
  [(Name_var w, entry 3 Name_mode.In_types);
   (Name_var x, entry 4 Name_mode.Normal);
   (Name_var y, entry 5 Name_mode.Phantom)].

Definition w_x_y_phantom : Aliases.t :=
  Aliases.make [(Name_var y, Var w); (Name_var x, Var w)]
    [(Name_var w,
      [(Name_var y, Name_mode.Phantom); (Name_var x, Name_mode.Normal)])]
    [].

Example canonical_at_phantom :
  Aliases.get_canonical_element_exn resolver times_phantom w_x_y_phantom
    (Var x) Name_mode.Normal Name_mode.Phantom 3
  = Some (Var y)
  ∧ Aliases.get_canonical_element_exn resolver times_phantom w_x_y_phantom
      (Var x) Name_mode.Normal Name_mode.Normal 3
    = Some (Var x).
Proof. split; reflexivity. Qed.

End Examples.
