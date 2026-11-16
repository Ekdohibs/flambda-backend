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

(* Proofs of the statements of FlambdaEnvAliasesSetSpec.v, and examples.
   Nothing here needs review. The examples are in the module [Examples]
   at the end: what the binary join does with the aliases of a variable
   at a join of two uses, and what each function answers on small
   sets. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec FlambdaMapProofs
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeMeaningSpec FlambdaTypeRenameSpec
  FlambdaTypeRenameProofs FlambdaBindingTime FlambdaEnvAliases
  FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs FlambdaEnvAliasesRenamingSpec
  FlambdaEnvAliasesRenamingProofs FlambdaEnvAliasesSetSpec
  FlambdaEnvAliasesReplay.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The functions on sets                                                    *)
(* ======================================================================== *)

Lemma alias_set_has_const (set : Aliases.Alias_set.t) (c : const) :
  alias_set_has set (Const c) <-> Aliases.Alias_set.set_const set = Some c.
Proof. reflexivity. Qed.

(* A simple is a name or a constant. *)
Lemma simple_cases (e : simple) :
  (∃ n : name, e = simple_of_name n) ∨ (∃ c : const, e = Const c).
Proof.
  destruct e as [x | s | c];
    [ left; exists (Name_var x) | left; exists (Name_symbol s)
    | right; exists c ];
    reflexivity.
Qed.

Theorem singleton_has : singleton_has_statement.
Proof.
  intros [x | s | c] [x' | s' | c']; simpl; intuition congruence.
Qed.

Theorem empty_has : empty_has_statement.
Proof.
  intros [x | s | c] Hhas; simpl in Hhas;
    [ contradiction | contradiction | discriminate ].
Qed.

Theorem is_empty_has : is_empty_has_statement.
Proof.
  intros [[c | ] [ | n names]]; unfold Aliases.Alias_set.is_empty; simpl;
    split; try discriminate; try reflexivity.
  - intros Hnone. exfalso. apply (Hnone (Const c)). reflexivity.
  - intros Hnone. exfalso. apply (Hnone (Const c)). reflexivity.
  - intros _ [x | s | c] Hhas; simpl in Hhas;
      [ contradiction | contradiction | discriminate ].
  - intros Hnone. exfalso. apply (Hnone (simple_of_name n)).
    apply alias_set_has_name. left. reflexivity.
Qed.

Theorem inter_has : inter_has_statement.
Proof.
  intros set set' e.
  destruct (simple_cases e) as [[n ->] | [c ->]].
  - rewrite !alias_set_has_name. unfold Aliases.Alias_set.inter. simpl.
    rewrite <- !(keyed_among_in name name_eqb name_eqb_eq),
      (keyed_among_inter name name_eqb name_eqb_eq), andb_true_iff.
    reflexivity.
  - rewrite !alias_set_has_const. unfold Aliases.Alias_set.inter. simpl.
    destruct (Aliases.Alias_set.set_const set) as [c1 | ];
      destruct (Aliases.Alias_set.set_const set') as [c2 | ];
      try (split; [ discriminate | intros [H1 H2]; discriminate ]).
    destruct (const_eqb c1 c2) eqn:Hequal.
    + apply const_eqb_eq in Hequal. subst c2. intuition congruence.
    + split; [ discriminate | ].
      intros [[= ->] [= ->]].
      rewrite (proj2 (const_eqb_eq c c) eq_refl) in Hequal. discriminate.
Qed.

Theorem filter_has : filter_has_statement.
Proof.
  intros set f e.
  destruct (simple_cases e) as [[n ->] | [c ->]].
  - rewrite !alias_set_has_name. unfold Aliases.Alias_set.filter. simpl.
    apply filter_In.
  - rewrite !alias_set_has_const. unfold Aliases.Alias_set.filter. simpl.
    destruct (Aliases.Alias_set.set_const set) as [c' | ];
      [ | split; [ discriminate | intros [H _]; discriminate ] ].
    destruct (f (Const c')) eqn:Hkept.
    + split; [ intros [= ->]; auto | intros [H _]; exact H ].
    + split; [ discriminate | ].
      intros [[= ->] Hkept']. congruence.
Qed.

(* ======================================================================== *)
(* Choosing an element                                                      *)
(* ======================================================================== *)

Lemma alias_rank_name (n : name) :
  alias_rank (simple_of_name n) = if name_is_symbol n then 1 else 2.
Proof. destruct n; reflexivity. Qed.

Lemma alias_rank_bounds (e : simple) : 0 <= alias_rank e <= 2.
Proof. destruct e; simpl; lia. Qed.

(* What [find_best] answers for a set with no constant: the first symbol
   of its names, or with no symbol the first of its names. *)
Lemma find_best_names (set : Aliases.Alias_set.t) (n : name) :
  Aliases.Alias_set.set_const set = None ->
  Aliases.Alias_set.find_best set = Some (simple_of_name n) ->
  n ∈ Aliases.Alias_set.set_names set
  ∧ (name_is_symbol n = false ->
     ∀ n' : name,
       n' ∈ Aliases.Alias_set.set_names set -> name_is_symbol n' = false).
Proof.
  intros Hconst. unfold Aliases.Alias_set.find_best. rewrite Hconst.
  cbv zeta.
  destruct (filter name_is_symbol (Aliases.Alias_set.set_names set))
    as [ | symbol symbols] eqn:Hsymbols.
  - destruct
      (filter (fun name => negb (name_is_symbol name))
         (Aliases.Alias_set.set_names set))
      as [ | var vars] eqn:Hvars;
      [ discriminate | ].
    intros [= Heq]. apply simple_of_name_inj in Heq. subst var.
    assert
      (Hin
        : n ∈ filter (fun name => negb (name_is_symbol name))
                (Aliases.Alias_set.set_names set)).
    { rewrite Hvars. left. reflexivity. }
    apply filter_In in Hin as [Hin _]. split; [ exact Hin | ].
    intros _ n' Hn'.
    destruct (name_is_symbol n') eqn:Hsymbol; [ | reflexivity ].
    assert
      (Hfiltered
        : n' ∈ filter name_is_symbol (Aliases.Alias_set.set_names set)).
    { apply filter_In. split; assumption. }
    rewrite Hsymbols in Hfiltered. contradiction.
  - intros [= Heq]. apply simple_of_name_inj in Heq. subst symbol.
    assert
      (Hin : n ∈ filter name_is_symbol (Aliases.Alias_set.set_names set)).
    { rewrite Hsymbols. left. reflexivity. }
    apply filter_In in Hin as [Hin Hsymbol]. split; [ exact Hin | ].
    congruence.
Qed.

Lemma find_best_const (set : Aliases.Alias_set.t) (e : simple) :
  Aliases.Alias_set.find_best set = Some e ->
  match Aliases.Alias_set.set_const set with
  | Some c => e = Const c
  | None => ∃ n : name, e = simple_of_name n
  end.
Proof.
  unfold Aliases.Alias_set.find_best.
  destruct (Aliases.Alias_set.set_const set) as [c | ];
    [ intros [= <-]; reflexivity | ].
  cbv zeta.
  destruct (filter name_is_symbol (Aliases.Alias_set.set_names set))
    as [ | symbol symbols].
  - destruct
      (filter (fun name => negb (name_is_symbol name))
         (Aliases.Alias_set.set_names set))
      as [ | var vars];
      [ discriminate | ].
    intros [= <-]. exists var. reflexivity.
  - intros [= <-]. exists symbol. reflexivity.
Qed.

Theorem find_best_has : find_best_has_statement.
Proof.
  intros set e Hbest. pose proof (find_best_const set e Hbest) as Hcases.
  destruct (Aliases.Alias_set.set_const set) as [c | ] eqn:Hconst.
  - subst e. exact Hconst.
  - destruct Hcases as [n ->]. apply alias_set_has_name.
    apply (find_best_names set n Hconst Hbest).
Qed.

Theorem find_best_best : find_best_best_statement.
Proof.
  intros set e e' Hbest Hhas.
  pose proof (find_best_const set e Hbest) as Hcases.
  destruct (Aliases.Alias_set.set_const set) as [c | ] eqn:Hconst.
  - subst e. simpl. apply alias_rank_bounds.
  - destruct Hcases as [n ->].
    destruct (simple_cases e') as [[n' ->] | [c' ->]].
    + rewrite !alias_rank_name.
      destruct (name_is_symbol n) eqn:Hsymbol;
        [ destruct (name_is_symbol n'); lia | ].
      apply alias_set_has_name in Hhas.
      rewrite
        (proj2 (find_best_names set n Hconst Hbest) Hsymbol n' Hhas).
      lia.
    + apply alias_set_has_const in Hhas. congruence.
Qed.

Theorem find_best_none : find_best_none_statement.
Proof.
  intros set. unfold Aliases.Alias_set.find_best, Aliases.Alias_set.is_empty.
  destruct (Aliases.Alias_set.set_const set) as [c | ];
    [ split; discriminate | ].
  cbv zeta.
  destruct (Aliases.Alias_set.set_names set) as [ | n names];
    [ split; reflexivity | ].
  simpl. destruct (name_is_symbol n); simpl; [ split; discriminate | ].
  destruct (filter name_is_symbol names); split; discriminate.
Qed.

Theorem get_singleton : get_singleton_statement.
Proof.
  intros set e. unfold Aliases.Alias_set.get_singleton.
  destruct (Aliases.Alias_set.set_const set) as [c | ] eqn:Hconst.
  - destruct (Aliases.Alias_set.set_names set) as [ | n names] eqn:Hnames;
      simpl.
    + split.
      * intros [= <-] e'.
        destruct (simple_cases e') as [[n' ->] | [c' ->]].
        -- rewrite alias_set_has_name, Hnames.
           split; [ contradiction | destruct n'; discriminate ].
        -- rewrite alias_set_has_const, Hconst. intuition congruence.
      * intros Hall. f_equal. apply Hall. exact Hconst.
    + split; [ discriminate | ].
      intros Hall. exfalso.
      assert (Hfirst : Const c = e) by (apply Hall; exact Hconst).
      assert (Hsecond : simple_of_name n = e).
      { apply Hall, alias_set_has_name. rewrite Hnames. left. reflexivity. }
      rewrite <- Hfirst in Hsecond. destruct n; discriminate.
  - destruct
      (Keyed.get_singleton name_eqb (Aliases.Alias_set.set_names set))
      as [n | ] eqn:Hsingleton;
      simpl.
    + pose proof
        (proj1 (keyed_get_singleton name name_eqb name_eqb_eq n _) Hsingleton)
        as Hamong.
      clear Hsingleton. rename Hamong into Hsingleton. split.
      * intros [= <-] e'.
        destruct (simple_cases e') as [[n' ->] | [c' ->]].
        -- rewrite alias_set_has_name,
             <- (keyed_among_in name name_eqb name_eqb_eq), Hsingleton,
             (name_eqb_eq n' n).
           split; [ intros ->; reflexivity | apply simple_of_name_inj ].
        -- rewrite alias_set_has_const, Hconst.
           split; [ discriminate | destruct n; discriminate ].
      * intros Hall. f_equal. apply Hall, alias_set_has_name.
        apply (keyed_among_in name name_eqb name_eqb_eq).
        rewrite Hsingleton. apply name_eqb_refl.
    + split; [ discriminate | ].
      intros Hall. exfalso.
      destruct (simple_cases e) as [[n ->] | [c ->]].
      * assert
          (Hsingleton'
            : Keyed.get_singleton name_eqb (Aliases.Alias_set.set_names set)
              = Some n).
        { apply (proj2 (keyed_get_singleton name name_eqb name_eqb_eq n _)).
          intros n'.
          destruct (name_eqb n' n) eqn:Hequal.
          - apply name_eqb_eq in Hequal. subst n'.
            apply (keyed_among_in name name_eqb name_eqb_eq),
              alias_set_has_name, Hall.
            reflexivity.
          - destruct
              (Keyed.among name_eqb n' (Aliases.Alias_set.set_names set))
              eqn:Hamong;
              [ | reflexivity ].
            apply (keyed_among_in name name_eqb name_eqb_eq),
              alias_set_has_name, Hall, simple_of_name_inj in Hamong.
            subst n'. rewrite name_eqb_refl in Hequal. discriminate. }
        congruence.
      * assert (Hhas : alias_set_has set (Const c))
          by (apply Hall; reflexivity).
        apply alias_set_has_const in Hhas. congruence.
Qed.

Theorem choose_opt_has : choose_opt_has_statement.
Proof.
  intros set e. unfold Aliases.Alias_set.choose_opt.
  destruct (Aliases.Alias_set.set_const set) as [c | ] eqn:Hconst.
  - destruct (Keyed.is_empty (Aliases.Alias_set.set_names set));
      [ | discriminate ].
    intros [= <-]. exact Hconst.
  - destruct (Aliases.Alias_set.set_names set) as [ | n names] eqn:Hnames;
      [ discriminate | ].
    intros [= <-]. apply alias_set_has_name. rewrite Hnames. left.
    reflexivity.
Qed.

Theorem choose_opt_none : choose_opt_none_statement.
Proof.
  intros set. unfold Aliases.Alias_set.choose_opt, Aliases.Alias_set.is_empty.
  destruct (Aliases.Alias_set.set_const set) as [c | ] eqn:Hconst.
  - destruct (Aliases.Alias_set.set_names set) as [ | n names] eqn:Hnames;
      simpl.
    + split; [ discriminate | ].
      intros [Hempty | (c' & n & _ & Hname)]; [ discriminate | ].
      apply alias_set_has_name in Hname. rewrite Hnames in Hname.
      contradiction.
    + split; [ | reflexivity ].
      intros _. right. exists c, n.
      split; [ exact Hconst | ].
      apply alias_set_has_name. rewrite Hnames. left. reflexivity.
  - destruct (Aliases.Alias_set.set_names set) as [ | n names] eqn:Hnames;
      simpl.
    + split; [ intros _; left; reflexivity | reflexivity ].
    + split; [ discriminate | ].
      intros [Hempty | (c & n' & Hhas & _)]; [ discriminate | ].
      apply alias_set_has_const in Hhas. congruence.
Qed.

(* ======================================================================== *)
(* Adding a set of aliases: the fold                                        *)
(* ======================================================================== *)

(* The elements of a set, in the order in which [Aliases.add_alias_set]
   goes through them. *)
Definition elements (aliases : Aliases.Alias_set.t) : list simple :=
  match Aliases.Alias_set.set_const aliases with
  | Some c => [Const c]
  | None => []
  end
  ++ map simple_of_name (Aliases.Alias_set.set_names aliases).

Lemma in_elements (aliases : Aliases.Alias_set.t) (e : simple) :
  e ∈ elements aliases <-> alias_set_has aliases e.
Proof.
  unfold elements. rewrite in_app_iff, in_map_iff.
  destruct (simple_cases e) as [[n ->] | [c ->]].
  - rewrite alias_set_has_name. split.
    + intros [Hin | (n' & Heq & Hin)].
      * destruct (Aliases.Alias_set.set_const aliases); simpl in Hin;
          [ destruct Hin as [Heq | []]; destruct n; discriminate
          | contradiction ].
      * apply simple_of_name_inj in Heq. subst n'. exact Hin.
    + intros Hin. right. exists n. split; [ reflexivity | exact Hin ].
  - rewrite alias_set_has_const. split.
    + intros [Hin | (n' & Heq & _)].
      * destruct (Aliases.Alias_set.set_const aliases) as [c' | ];
          simpl in Hin;
          [ destruct Hin as [[= ->] | []]; reflexivity | contradiction ].
      * destruct n'; discriminate.
    + intros ->. left. left. reflexivity.
Qed.

Section Add_alias_set.

Context {A : Type}
  (binding_time_resolver : name -> Binding_time.With_name_mode.t)
  (binding_times_and_modes
    : Keyed.finite_map name (A * Binding_time.With_name_mode.t)).

(* One element of the set: its canonical element becomes an alias of the
   canonical element so far. *)
Definition step (acc : option (Aliases.t * simple)) (e : simple)
    : option (Aliases.t * simple) :=
  match acc with
  | None => None
  | Some acc =>
    Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
      acc
      (canonical_element_of (fst acc) e)
  end.

Local Arguments Aliases.add_alias_of_set : simpl never.

Lemma fold_step_none (todo : list simple) :
  List.fold_left step todo None = None.
Proof. induction todo as [ | e todo IH]; simpl; [ reflexivity | exact IH ]. Qed.

Lemma add_alias_set_eq (t : Aliases.t) (name : name)
    (aliases : Aliases.Alias_set.t) :
  Aliases.add_alias_set binding_time_resolver binding_times_and_modes t name
    aliases
  = option_map fst
      (List.fold_left step (elements aliases)
         (Some (t, canonical_element_of t (simple_of_name name)))).
Proof.
  unfold Aliases.add_alias_set, elements. cbv zeta.
  rewrite List.fold_left_app, get_canonical_ignoring_name_mode.
  assert
    (Hnames
      : ∀ (names : list FlambdaTypeGrammar.name)
          (acc : option (Aliases.t * simple)),
          List.fold_left
            (fun acc name =>
               match acc with
               | Some acc =>
                 Aliases.add_alias_of_set binding_time_resolver
                   binding_times_and_modes acc
                   (Aliases.get_canonical_ignoring_name_mode (fst acc) name)
               | None => None
               end)
            names acc
          = List.fold_left step (map simple_of_name names) acc).
  { induction names as [ | n names IH]; intros acc; simpl;
      [ reflexivity | ].
    rewrite IH. reflexivity. }
  destruct (Aliases.Alias_set.set_const aliases) as [c | ]; simpl.
  - change (canonical_element_of t (Const c)) with (Const c).
    destruct
      (Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
         (t, canonical_element_of t (simple_of_name name)) (Const c))
      as [acc | ].
    + rewrite Hnames.
      destruct
        (List.fold_left step
           (map simple_of_name (Aliases.Alias_set.set_names aliases))
           (Some acc));
        reflexivity.
    + rewrite fold_step_none. reflexivity.
  - rewrite Hnames.
    destruct
      (List.fold_left step
         (map simple_of_name (Aliases.Alias_set.set_names aliases))
         (Some (t, canonical_element_of t (simple_of_name name))));
      reflexivity.
Qed.

(* What holds of the structure and of the canonical element so far when
   it holds before the first element and each element keeps it. *)
Lemma fold_step_invariant (P : list simple -> Aliases.t -> simple -> Prop)
    (all : list simple) :
  (∀ (done : list simple) (t : Aliases.t) (c e : simple) (t' : Aliases.t)
     (c' : simple),
    e ∈ all ->
    P done t c ->
    Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
      (t, c) (canonical_element_of t e)
    = Some (t', c') ->
    P (done ++ [e]) t' c') ->
  ∀ (todo done : list simple) (t : Aliases.t) (c : simple) (t' : Aliases.t)
    (c' : simple),
    (∀ e : simple, e ∈ todo -> e ∈ all) ->
    P done t c ->
    List.fold_left step todo (Some (t, c)) = Some (t', c') ->
    P (done ++ todo) t' c'.
Proof.
  intros Hstep.
  induction todo as [ | e todo IH]; intros done t c t' c' Hall Hdone Hfold.
  - simpl in Hfold. injection Hfold as <- <-. rewrite app_nil_r. exact Hdone.
  - simpl in Hfold.
    destruct
      (Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
         (t, c) (canonical_element_of t e))
      as [[t1 c1] | ] eqn:Hone;
      [ | rewrite fold_step_none in Hfold; discriminate ].
    replace (done ++ e :: todo) with ((done ++ [e]) ++ todo)
      by (rewrite <- app_assoc; reflexivity).
    apply (IH (done ++ [e]) t1 c1).
    + intros e' Hin. apply Hall. right. exact Hin.
    + apply (Hstep done t c e); [ apply Hall; left; reflexivity | | ];
        assumption.
    + exact Hfold.
Qed.

(* ======================================================================== *)
(* Adding a set of aliases: one element                                     *)
(* ======================================================================== *)

Lemma add0_eq (t : Aliases.t) (canonical_element1 canonical_element2 : simple) :
  Aliases.add0 binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = let '(canonical_element, to_be_demoted, which_element) :=
      if Aliases.defined_earlier binding_time_resolver
           binding_times_and_modes canonical_element1 canonical_element2
      then
        (canonical_element1, canonical_element2,
         Aliases.Demote_canonical_element2)
      else
        (canonical_element2, canonical_element1,
         Aliases.Demote_canonical_element1)
    in
    match
      Aliases.add_alias_between_canonical_elements binding_times_and_modes
        t canonical_element to_be_demoted
    with
    | None => None
    | Some Or_bottom.Bottom => Some Or_bottom.Bottom
    | Some (Or_bottom.Ok t') => Some (Or_bottom.Ok (t', which_element))
    end.
Proof.
  unfold Aliases.add0, Aliases.add_alias,
    Aliases.choose_canonical_element_to_be_demoted.
  destruct
    (Aliases.defined_earlier binding_time_resolver binding_times_and_modes
       canonical_element1 canonical_element2);
    reflexivity.
Qed.

(* A simple that is demoted to a different one is a name. *)
Lemma demoted_is_name (t t' : Aliases.t)
    (canonical_element to_be_demoted : simple) :
  canonical_element <> to_be_demoted ->
  Aliases.add_alias_between_canonical_elements binding_times_and_modes t
    canonical_element to_be_demoted
  = Some (Or_bottom.Ok t') ->
  ∃ n : name, name_of_simple to_be_demoted = Some n.
Proof.
  intros Hdifferent. unfold Aliases.add_alias_between_canonical_elements.
  destruct (simple_equal canonical_element to_be_demoted) eqn:Hequal;
    [ apply simple_equal_eq in Hequal; contradiction | ].
  destruct (name_of_simple to_be_demoted) as [n | ]; [ | discriminate ].
  intros _. exists n. reflexivity.
Qed.

(* One element, by [Aliases.add]. *)
Lemma add_alias_of_set_eq (t : Aliases.t) (c elt : simple) :
  Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
    (t, c) elt
  = if simple_equal c elt
    then Some (t, c)
    else
      match
        Aliases.add binding_time_resolver binding_times_and_modes t c elt
      with
      | Some (Or_bottom.Ok result) =>
        Some (Aliases.add_result_t result, Aliases.canonical_element result)
      | Some Or_bottom.Bottom | None => None
      end.
Proof.
  unfold Aliases.add_alias_of_set.
  destruct (simple_equal c elt) eqn:Hequal; [ reflexivity | ].
  assert (Hdifferent : c <> elt).
  { intros Heq. apply simple_equal_eq in Heq. congruence. }
  rewrite add0_eq, add_eq.
  destruct
    (Aliases.defined_earlier binding_time_resolver binding_times_and_modes c
       elt).
  - destruct
      (Aliases.add_alias_between_canonical_elements binding_times_and_modes
         t c elt)
      as [[t' | ] | ] eqn:Hadd;
      try reflexivity.
    destruct (demoted_is_name t t' c elt Hdifferent Hadd) as [n ->].
    reflexivity.
  - destruct
      (Aliases.add_alias_between_canonical_elements binding_times_and_modes
         t elt c)
      as [[t' | ] | ] eqn:Hadd;
      try reflexivity.
    destruct (demoted_is_name t t' elt c (fun Heq => Hdifferent (eq_sym Heq))
                Hadd)
      as [n ->].
    reflexivity.
Qed.

Lemma add_alias_of_set_cases (t : Aliases.t) (c elt : simple) (t' : Aliases.t)
    (c' : simple) :
  Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
    (t, c) elt
  = Some (t', c') ->
  (c = elt ∧ t' = t ∧ c' = c)
  ∨ (c <> elt
     ∧ ∃ result : Aliases.add_result,
         Aliases.add binding_time_resolver binding_times_and_modes t c elt
         = Some (Or_bottom.Ok result)
         ∧ Aliases.add_result_t result = t'
         ∧ Aliases.canonical_element result = c').
Proof.
  rewrite add_alias_of_set_eq.
  destruct (simple_equal c elt) eqn:Hequal.
  - intros [= <- <-]. left. apply simple_equal_eq in Hequal. auto.
  - intros Hadd. right. split.
    + intros Heq. apply simple_equal_eq in Heq. congruence.
    + destruct
        (Aliases.add binding_time_resolver binding_times_and_modes t c elt)
        as [[result | ] | ];
        try discriminate.
      injection Hadd as <- <-. exists result. auto.
Qed.

Lemma aliased_canonical_element (t : Aliases.t) (e e' : simple) :
  aliases_consistent t ->
  (aliased t e (canonical_element_of t e') <-> aliased t e e').
Proof.
  intros Hconsistent. unfold aliased.
  rewrite (canonical_element_idempotent t e' Hconsistent). reflexivity.
Qed.

(* One element keeps consistency, and the canonical element so far is
   that of the name. *)
Lemma step_consistent (t : Aliases.t) (name : name) (c e : simple)
    (t' : Aliases.t) (c' : simple) :
  aliases_consistent t ->
  c = canonical_element_of t (simple_of_name name) ->
  Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
    (t, c) (canonical_element_of t e)
  = Some (t', c') ->
  aliases_consistent t' ∧ c' = canonical_element_of t' (simple_of_name name).
Proof.
  intros Hconsistent Hc Hone.
  apply add_alias_of_set_cases in Hone
    as [(_ & -> & ->) | (Hdifferent & result & Hadd & <- & <-)];
    [ auto | ].
  assert (Hcanonical : is_canonical t c).
  { rewrite Hc. apply canonical_element_canonical, Hconsistent. }
  assert (Hcanonical' : is_canonical t (canonical_element_of t e)).
  { apply canonical_element_canonical, Hconsistent. }
  split.
  - apply
      (add_consistent A binding_time_resolver binding_times_and_modes t c
         (canonical_element_of t e) result Hconsistent Hcanonical Hcanonical'
         Hadd).
  - rewrite
      (add_canonical_element A binding_time_resolver binding_times_and_modes
         t c (canonical_element_of t e) result (simple_of_name name)
         Hconsistent Hcanonical Hcanonical' Hadd).
    rewrite <- Hc.
    destruct
      (simple_equal c (simple_of_name (Aliases.demoted_name result)))
      eqn:Hdemoted;
      [ reflexivity | ].
    destruct
      (add_arguments binding_time_resolver binding_times_and_modes t c
         (canonical_element_of t e) result Hadd)
      as [[Hfirst _] | [_ Hsecond]];
      [ exact Hfirst | ].
    rewrite Hsecond in Hdemoted.
    rewrite (proj2 (simple_equal_eq c c) eq_refl) in Hdemoted. discriminate.
Qed.

(* One element keeps respect, and the canonical element so far has the
   value of the name. *)
Lemma step_respected (t : Aliases.t) (name : name) (c e : simple)
    (t' : Aliases.t) (c' : simple) (vars : env) :
  aliases_respected t vars ->
  eval_simple vars c = value_of_name vars name ->
  eval_simple vars e = value_of_name vars name ->
  Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
    (t, c) (canonical_element_of t e)
  = Some (t', c') ->
  aliases_respected t' vars ∧ eval_simple vars c' = value_of_name vars name.
Proof.
  intros Hrespected Hc He Hone.
  assert
    (Helt
      : eval_simple vars (canonical_element_of t e) = value_of_name vars name).
  { rewrite (respected_canonical_element_of t vars e Hrespected). exact He. }
  apply add_alias_of_set_cases in Hone
    as [(_ & -> & ->) | (Hdifferent & result & Hadd & <- & <-)];
    [ auto | ].
  split.
  - apply
      (add_respected A binding_time_resolver binding_times_and_modes t c
         (canonical_element_of t e) result vars Hadd Hrespected).
    rewrite Hc, Helt. reflexivity.
  - destruct
      (add_arguments binding_time_resolver binding_times_and_modes t c
         (canonical_element_of t e) result Hadd)
      as [[-> _] | [-> _]];
      assumption.
Qed.

(* One element has no fatal error when an environment witnesses that it
   has the value of the name. *)
Lemma step_defined (t : Aliases.t) (name : name) (c e : simple) (vars : env) :
  aliases_consistent t ->
  c = canonical_element_of t (simple_of_name name) ->
  aliases_respected t vars ->
  eval_simple vars c = value_of_name vars name ->
  eval_simple vars e = value_of_name vars name ->
  Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
    (t, c) (canonical_element_of t e)
  <> None.
Proof.
  intros Hconsistent Hc Hrespected Hvalue He.
  rewrite add_alias_of_set_eq.
  destruct (simple_equal c (canonical_element_of t e)) eqn:Hequal;
    [ discriminate | ].
  assert (Hdifferent : c <> canonical_element_of t e).
  { intros Heq. apply simple_equal_eq in Heq. congruence. }
  destruct
    (Aliases.add binding_time_resolver binding_times_and_modes t c
       (canonical_element_of t e))
    as [[result | ] | ] eqn:Hadd;
    [ discriminate | | ].
  - exfalso. apply add_bottom in Hadd as (c1 & c2 & Hfirst & Hsecond & Hapart).
    apply Hapart.
    assert
      (Hsame
        : eval_simple vars c
          = eval_simple vars (canonical_element_of t e)).
    { rewrite (respected_canonical_element_of t vars e Hrespected), Hvalue, He.
      reflexivity. }
    rewrite Hfirst, Hsecond in Hsame. simpl in Hsame. congruence.
  - exfalso. revert Hadd.
    apply add_defined; try assumption.
    + rewrite Hc. apply canonical_element_canonical, Hconsistent.
    + apply canonical_element_canonical, Hconsistent.
Qed.

End Add_alias_set.

#[local] Arguments Aliases.add_alias_of_set : simpl never.

(* ======================================================================== *)
(* Adding a set of aliases: the statements                                  *)
(* ======================================================================== *)

Lemma add_alias_set_invariant {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (P : list simple -> Aliases.t -> simple -> Prop) (t : Aliases.t)
    (name : name) (aliases : Aliases.Alias_set.t) (t' : Aliases.t) :
  Aliases.add_alias_set binding_time_resolver binding_times_and_modes t name
    aliases
  = Some t' ->
  P [] t (canonical_element_of t (simple_of_name name)) ->
  (∀ (done : list simple) (t1 : Aliases.t) (c e : simple) (t2 : Aliases.t)
     (c2 : simple),
    alias_set_has aliases e ->
    P done t1 c ->
    Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
      (t1, c) (canonical_element_of t1 e)
    = Some (t2, c2) ->
    P (done ++ [e]) t2 c2) ->
  ∃ c' : simple, P (elements aliases) t' c'.
Proof.
  intros Hadd Hfirst Hstep. rewrite add_alias_set_eq in Hadd.
  destruct
    (List.fold_left (step binding_time_resolver binding_times_and_modes)
       (elements aliases)
       (Some (t, canonical_element_of t (simple_of_name name))))
    as [[t1 c'] | ] eqn:Hfold;
    [ | discriminate ].
  injection Hadd as ->. exists c'.
  apply
    (fold_step_invariant binding_time_resolver binding_times_and_modes P
       (elements aliases))
    with (done := []) in Hfold;
    [ exact Hfold | | auto | exact Hfirst ].
  intros done t1 c e t2 c2 Hin. apply Hstep, in_elements, Hin.
Qed.

Theorem add_alias_set_respected : add_alias_set_respected_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t'
    vars Hadd Hrespected Hvalues.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun _ t c =>
          aliases_respected t vars
          ∧ eval_simple vars c = value_of_name vars name)
       t name aliases t' Hadd)
    as [c' [Hrespected' _]];
    [ | | exact Hrespected' ].
  - split; [ exact Hrespected | ].
    apply (respected_canonical_element_of t vars _ Hrespected).
  - intros done t1 c e t2 c2 Hhas [Hrespected1 Hvalue] Hone.
    apply
      (step_respected binding_time_resolver binding_times_and_modes t1 name c
         e t2 c2 vars);
      auto.
Qed.

Theorem add_alias_set_consistent : add_alias_set_consistent_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t'
    Hconsistent Hadd.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun _ t c =>
          aliases_consistent t
          ∧ c = canonical_element_of t (simple_of_name name))
       t name aliases t' Hadd)
    as [c' [Hconsistent' _]];
    [ | | exact Hconsistent' ].
  - auto.
  - intros done t1 c e t2 c2 _ [Hconsistent1 Hc] Hone.
    apply
      (step_consistent binding_time_resolver binding_times_and_modes t1 name c
         e t2 c2);
      assumption.
Qed.

Theorem add_alias_set_name_modes_recorded :
  add_alias_set_name_modes_recorded_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t'
    Hrecorded Hadd.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun _ t _ => name_modes_recorded binding_times_and_modes t)
       t name aliases t' Hadd)
    as [c' Hrecorded'];
    [ | | exact Hrecorded' ].
  - exact Hrecorded.
  - intros done t1 c e t2 c2 _ Hrecorded1 Hone.
    apply add_alias_of_set_cases in Hone
      as [(_ & -> & _) | (_ & result & Hadd1 & <- & _)];
      [ exact Hrecorded1 | ].
    apply
      (add_name_modes_recorded A binding_time_resolver binding_times_and_modes
         t1 c (canonical_element_of t1 e) result Hrecorded1 Hadd1).
Qed.

Lemma fold_step_defined {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (name : name) (vars : env) (todo : list simple) :
  ∀ (t : Aliases.t) (c : simple),
    aliases_consistent t ->
    c = canonical_element_of t (simple_of_name name) ->
    aliases_respected t vars ->
    eval_simple vars c = value_of_name vars name ->
    (∀ e : simple, e ∈ todo -> eval_simple vars e = value_of_name vars name) ->
    List.fold_left (step binding_time_resolver binding_times_and_modes) todo
      (Some (t, c))
    <> None.
Proof.
  induction todo as [ | e todo IH];
    intros t c Hconsistent Hc Hrespected Hvalue Hall; simpl;
    [ discriminate | ].
  destruct
    (Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
       (t, c) (canonical_element_of t e))
    as [[t1 c1] | ] eqn:Hone.
  - destruct
      (step_consistent binding_time_resolver binding_times_and_modes t name c
         e t1 c1 Hconsistent Hc Hone)
      as [Hconsistent1 Hc1].
    destruct
      (step_respected binding_time_resolver binding_times_and_modes t name c
         e t1 c1 vars Hrespected Hvalue (Hall e (or_introl eq_refl)) Hone)
      as [Hrespected1 Hvalue1].
    apply IH; auto.
    intros e' Hin. apply Hall. right. exact Hin.
  - exfalso. revert Hone.
    apply
      (step_defined binding_time_resolver binding_times_and_modes t name c e
         vars);
      auto.
    apply Hall. left. reflexivity.
Qed.

Theorem add_alias_set_defined : add_alias_set_defined_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases vars
    Hconsistent Hrespected Hvalues.
  rewrite add_alias_set_eq.
  destruct
    (List.fold_left (step binding_time_resolver binding_times_and_modes)
       (elements aliases)
       (Some (t, canonical_element_of t (simple_of_name name))))
    as [acc | ] eqn:Hfold;
    [ discriminate | ].
  exfalso. revert Hfold.
  apply
    (fold_step_defined binding_time_resolver binding_times_and_modes name
       vars);
    auto.
  - apply (respected_canonical_element_of t vars _ Hrespected).
  - intros e Hin. apply Hvalues, in_elements, Hin.
Qed.

Theorem add_alias_set_mentioned : add_alias_set_mentioned_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t' n
    Hadd Hmentioned.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun done t1 c =>
          (∀ n : FlambdaTypeGrammar.name,
            aliases_mention t1 n ->
            aliases_mention t n ∨ n = name ∨ simple_of_name n ∈ done)
          ∧ (∀ n : FlambdaTypeGrammar.name,
              simple_of_name n = c ->
              aliases_mention t n ∨ n = name ∨ simple_of_name n ∈ done))
       t name aliases t' Hadd)
    as [c' [Hmentions _]].
  - split; [ auto | ].
    intros n' Heq.
    destruct (canonical_element_mentioned t name n' (eq_sym Heq)); auto.
  - intros done t1 c e t2 c2 _ [Hmentions Hc] Hone.
    assert
      (Hweaken
        : ∀ n' : FlambdaTypeGrammar.name,
            aliases_mention t n' ∨ n' = name ∨ simple_of_name n' ∈ done ->
            aliases_mention t n' ∨ n' = name
            ∨ simple_of_name n' ∈ done ++ [e]).
    { intros n' [H | [H | H]]; auto.
      right. right. apply in_or_app. left. exact H. }
    assert
      (Helt
        : ∀ n' : FlambdaTypeGrammar.name,
            simple_of_name n' = canonical_element_of t1 e ->
            aliases_mention t n' ∨ n' = name
            ∨ simple_of_name n' ∈ done ++ [e]).
    { intros n' Heq.
      destruct (simple_cases e) as [[ne ->] | [c0 ->]].
      - destruct (canonical_element_mentioned t1 ne n' (eq_sym Heq))
          as [-> | Hmentioned1].
        + right. right. apply in_or_app. right. left. reflexivity.
        + apply Hweaken, Hmentions, Hmentioned1.
      - destruct n'; discriminate. }
    apply add_alias_of_set_cases in Hone
      as [(_ & -> & ->) | (_ & result & Hadd1 & <- & <-)].
    + split; intros n' H;
        [ apply Hweaken, Hmentions, H | apply Hweaken, Hc, H ].
    + split.
      * intros n' Hmentioned2.
        destruct
          (add_mentioned A binding_time_resolver binding_times_and_modes t1 c
             (canonical_element_of t1 e) result n' Hadd1 Hmentioned2)
          as [H | [H | H]].
        -- apply Hweaken, Hmentions, H.
        -- apply Hweaken, Hc, H.
        -- apply Helt, H.
      * intros n' Heq.
        destruct
          (add_arguments binding_time_resolver binding_times_and_modes t1 c
             (canonical_element_of t1 e) result Hadd1)
          as [[Hfirst _] | [Hsecond _]].
        -- apply Hweaken, Hc. congruence.
        -- apply Helt. congruence.
  - destruct (Hmentions n Hmentioned) as [H | [H | H]]; auto.
    right. right. apply in_elements, H.
Qed.

(* The simples that the elements gone through put in the class of the
   name. *)
Definition merged (t : Aliases.t) (name : name) (done : list simple)
    (e : simple) : Prop :=
  aliased t e (simple_of_name name)
  ∨ (∃ x : simple, x ∈ done ∧ aliased t e x).

Lemma aliased_refl (t : Aliases.t) (e : simple) : aliased t e e.
Proof. reflexivity. Qed.

Lemma aliased_sym (t : Aliases.t) (e e' : simple) :
  aliased t e e' -> aliased t e' e.
Proof. unfold aliased. congruence. Qed.

Lemma aliased_trans (t : Aliases.t) (e e' e'' : simple) :
  aliased t e e' -> aliased t e' e'' -> aliased t e e''.
Proof. unfold aliased. congruence. Qed.

Lemma merged_closed (t : Aliases.t) (name : name) (done : list simple)
    (e x : simple) :
  aliased t e x -> merged t name done x -> merged t name done e.
Proof.
  intros Haliased [Hname | (y & Hin & Hy)].
  - left. exact (aliased_trans t _ _ _ Haliased Hname).
  - right. exists y. split; [ exact Hin | ].
    exact (aliased_trans t _ _ _ Haliased Hy).
Qed.

Lemma merged_app (t : Aliases.t) (name : name) (done : list simple)
    (e x : simple) :
  merged t name (done ++ [x]) e <-> merged t name done e ∨ aliased t e x.
Proof.
  unfold merged. split.
  - intros [Hname | (y & Hin & Hy)]; [ auto | ].
    apply in_app_or in Hin as [Hin | [<- | []]]; [ | auto ].
    left. right. exists y. auto.
  - intros [[Hname | (y & Hin & Hy)] | Hx].
    + auto.
    + right. exists y. split; [ apply in_or_app; auto | exact Hy ].
    + right. exists x.
      split; [ apply in_or_app; right; left; reflexivity | exact Hx ].
Qed.

Theorem add_alias_set_aliased : add_alias_set_aliased_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t' e
    e' Hconsistent Hadd.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun done t1 c =>
          aliases_consistent t1
          ∧ c = canonical_element_of t1 (simple_of_name name)
          ∧ ∀ e e' : simple,
              aliased t1 e e'
              <-> aliased t e e'
                  ∨ (merged t name done e ∧ merged t name done e'))
       t name aliases t' Hadd)
    as [c' (_ & _ & Hclasses)].
  - split; [ exact Hconsistent | ]. split; [ reflexivity | ].
    intros e1 e2. split; [ auto | ].
    intros [Hbefore | [[H1 | (x & [] & _)] [H2 | (y & [] & _)]]];
      [ exact Hbefore | ].
    exact (aliased_trans t _ _ _ H1 (aliased_sym t _ _ H2)).
  - intros done t1 c x t2 c2 _ (Hconsistent1 & Hc & Hclasses) Hone.
    destruct
      (step_consistent binding_time_resolver binding_times_and_modes t1 name c
         x t2 c2 Hconsistent1 Hc Hone)
      as [Hconsistent2 Hc2].
    split; [ exact Hconsistent2 | ]. split; [ exact Hc2 | ].
    (* The class of the name in [t1] is what has been merged so far. *)
    assert
      (Hname
        : ∀ e1 : simple,
            aliased t1 e1 (simple_of_name name) <-> merged t name done e1).
    { intros e1. rewrite Hclasses. split.
      - intros [Hbefore | [Hmerged _]]; [ left; exact Hbefore | exact Hmerged ].
      - intros Hmerged. right. split; [ exact Hmerged | ].
        left. apply aliased_refl. }
    assert
      (Hbefore
        : ∀ e1 e2 : simple, aliased t e1 e2 -> aliased t1 e1 e2).
    { intros e1 e2 H. apply Hclasses. left. exact H. }
    intros e1 e2. rewrite !merged_app.
    apply add_alias_of_set_cases in Hone
      as [(Hsame & -> & _) | (_ & result & Hadd1 & <- & _)].
    + (* The element was in the class of the name already. *)
      assert (Hx : merged t name done x).
      { apply Hname. unfold aliased. rewrite <- Hc. exact (eq_sym Hsame). }
      rewrite Hclasses. split.
      * intros [H | [H1 H2]]; auto.
      * intros [H | [[H1 | H1] [H2 | H2]]]; auto; right; split; auto;
          solve [ exact (merged_closed t name done _ x H1 Hx)
                | exact (merged_closed t name done _ x H2 Hx) ].
    + assert (Hcanonical : is_canonical t1 c).
      { rewrite Hc. apply canonical_element_canonical, Hconsistent1. }
      assert (Hcanonical' : is_canonical t1 (canonical_element_of t1 x)).
      { apply canonical_element_canonical, Hconsistent1. }
      rewrite
        (add_aliased A binding_time_resolver binding_times_and_modes t1 c
           (canonical_element_of t1 x) result e1 e2 Hconsistent1 Hcanonical
           Hcanonical' Hadd1).
      rewrite Hc, !(aliased_canonical_element t1 _ _ Hconsistent1).
      split.
      * intros [H | [[H1 H2] | [H1 H2]]].
        -- apply Hclasses in H as [H | [H1 H2]]; auto.
        -- apply Hname in H1. apply Hclasses in H2 as [H2 | [H2 _]]; auto.
        -- apply Hname in H2. apply Hclasses in H1 as [H1 | [H1 _]]; auto.
      * intros [H | [[H1 | H1] [H2 | H2]]].
        -- left. apply Hbefore, H.
        -- left. apply Hclasses. auto.
        -- right. left. split; [ apply Hname, H1 | apply Hbefore, H2 ].
        -- right. right. split; [ apply Hbefore, H1 | apply Hname, H2 ].
        -- left. apply Hbefore.
           exact (aliased_trans t _ _ _ H1 (aliased_sym t _ _ H2)).
  - rewrite Hclasses.
    assert
      (Hmerged
        : ∀ e1 : simple,
            merged t name (elements aliases) e1
            <-> merged_by_set t name aliases e1).
    { intros e1. unfold merged, merged_by_set. split.
      - intros [H | (x & Hin & Hx)]; [ auto | ].
        right. exists x. split; [ apply in_elements, Hin | exact Hx ].
      - intros [H | (x & Hhas & Hx)]; [ auto | ].
        right. exists x. split; [ apply in_elements, Hhas | exact Hx ]. }
    rewrite !Hmerged. reflexivity.
Qed.

Theorem add_alias_set_earliest : add_alias_set_earliest_statement.
Proof.
  intros A binding_time_resolver binding_times_and_modes t name aliases t'
    Hconsistent Hearliest Hcomparable Hadd.
  destruct
    (add_alias_set_invariant binding_time_resolver binding_times_and_modes
       (fun _ t1 c =>
          aliases_consistent t1
          ∧ c = canonical_element_of t1 (simple_of_name name)
          ∧ canonical_elements_earliest binding_time_resolver
              binding_times_and_modes t1
          ∧ (∀ x : simple, is_canonical t1 x -> is_canonical t x)
          ∧ ∀ a b : simple,
              aliased t1 a b ->
              aliased t a b
              ∨ (merged_by_set t name aliases a
                 ∧ merged_by_set t name aliases b))
       t name aliases t' Hadd)
    as [c' (_ & _ & Hearliest' & _)]; [ | | exact Hearliest' ].
  - split; [ exact Hconsistent | ]. split; [ reflexivity | ].
    split; [ exact Hearliest | ]. split; auto.
  - intros done t1 c e t2 c2 Hhas
      (Hconsistent1 & Hc & Hearliest1 & Hcanonical1 & Hclasses1) Hone.
    destruct
      (step_consistent binding_time_resolver binding_times_and_modes t1 name c
         e t2 c2 Hconsistent1 Hc Hone)
      as [Hconsistent2 Hc2].
    split; [ exact Hconsistent2 | ]. split; [ exact Hc2 | ].
    apply add_alias_of_set_cases in Hone
      as [(_ & -> & _) | (Hdifferent & result & Hadd1 & <- & _)];
      [ split; [ exact Hearliest1 | split; assumption ] | ].
    assert (Hcan : is_canonical t1 c).
    { rewrite Hc. apply canonical_element_canonical, Hconsistent1. }
    assert (Hcan' : is_canonical t1 (canonical_element_of t1 e)).
    { apply canonical_element_canonical, Hconsistent1. }
    (* What is in the class of the name or of the element is merged. *)
    assert (Hmerged :
      ∀ a : simple,
        aliased t1 a c ∨ aliased t1 a (canonical_element_of t1 e) ->
        merged_by_set t name aliases a).
    { intros a [Ha | Ha].
      - assert (Ha' : aliased t1 a (simple_of_name name)).
        { rewrite Hc in Ha.
          apply (aliased_canonical_element t1 _ _ Hconsistent1), Ha. }
        destruct (Hclasses1 _ _ Ha') as [H | [H _]];
          [ left; exact H | exact H ].
      - assert (Ha' : aliased t1 a e).
        { apply (aliased_canonical_element t1 _ _ Hconsistent1), Ha. }
        destruct (Hclasses1 _ _ Ha') as [H | [H _]]; [ | exact H ].
        right. exists e. split; [ exact Hhas | exact H ]. }
    split; [ | split ].
    + apply
        (add_earliest A binding_time_resolver binding_times_and_modes t1 c
           (canonical_element_of t1 e) result Hconsistent1 Hearliest1 Hcan
           Hcan');
        [ | exact Hadd1 ].
      apply Hcomparable;
        [ apply Hcanonical1, Hcan
        | apply Hcanonical1, Hcan'
        | apply Hmerged; left; apply aliased_refl
        | apply Hmerged; right; apply aliased_refl
        | exact Hdifferent ].
    + intros x Hx.
      apply
        (add_is_canonical A binding_time_resolver binding_times_and_modes t1
           c (canonical_element_of t1 e) result x Hconsistent1 Hcan Hcan'
           Hdifferent Hadd1)
        in Hx.
      destruct Hx as [Hx _]. apply Hcanonical1, Hx.
    + intros a b Hab.
      apply
        (add_aliased A binding_time_resolver binding_times_and_modes t1 c
           (canonical_element_of t1 e) result a b Hconsistent1 Hcan Hcan'
           Hadd1)
        in Hab.
      destruct Hab as [H | [[Ha Hb] | [Ha Hb]]];
        [ exact (Hclasses1 _ _ H) | | ];
        right; split; apply Hmerged; auto.
Qed.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Theorem rename_alias_set_has : rename_alias_set_has_statement.
Proof.
  intros r set e.
  destruct (simple_cases e) as [[n ->] | [c ->]].
  - rewrite rename_simple_of_name, !alias_set_has_name. simpl.
    rewrite <- !(keyed_among_in name name_eqb name_eqb_eq),
      (among_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    reflexivity.
  - reflexivity.
Qed.

Theorem rename_singleton : rename_singleton_statement.
Proof. intros r [x | sym | c]; reflexivity. Qed.

Theorem rename_is_empty : rename_is_empty_statement.
Proof.
  intros r [[c | ] names]; unfold Aliases.Alias_set.is_empty; simpl;
    [ reflexivity | apply is_empty_map ].
Qed.

Lemma inter_renamed (r : renaming) (names names' : list name) :
  Keyed.inter name_eqb (map (rename_name r) names)
    (map (rename_name r) names')
  = map (rename_name r) (Keyed.inter name_eqb names names').
Proof.
  unfold Keyed.inter. apply filter_renamed. intros n.
  apply (among_keys_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
Qed.

Theorem rename_inter : rename_inter_statement.
Proof.
  intros r [const names] [const' names'].
  unfold Aliases.Alias_set.inter, rename_alias_set. simpl.
  rewrite inter_renamed. reflexivity.
Qed.

Theorem rename_filter : rename_filter_statement.
Proof.
  intros r [const names] f.
  unfold Aliases.Alias_set.filter, rename_alias_set. simpl. f_equal.
  apply filter_renamed. intros n. rewrite rename_simple_of_name.
  reflexivity.
Qed.

Theorem rename_find_best : rename_find_best_statement.
Proof.
  intros r [[c | ] names]; unfold Aliases.Alias_set.find_best; simpl;
    [ reflexivity | ].
  rewrite (filter_renamed (rename_name r) name_is_symbol name_is_symbol),
    (filter_renamed (rename_name r)
       (fun name => negb (name_is_symbol name))
       (fun name => negb (name_is_symbol name)));
    try (intros n; rewrite name_is_symbol_renamed; reflexivity).
  destruct (filter name_is_symbol names) as [ | symbol symbols]; simpl.
  - destruct (filter (fun name => negb (name_is_symbol name)) names)
      as [ | var vars];
      simpl; [ reflexivity | ].
    rewrite rename_simple_of_name. reflexivity.
  - rewrite rename_simple_of_name. reflexivity.
Qed.

Theorem rename_get_singleton : rename_get_singleton_statement.
Proof.
  intros r [[c | ] names]; unfold Aliases.Alias_set.get_singleton; simpl.
  - rewrite is_empty_map. destruct (Keyed.is_empty names); reflexivity.
  - unfold Keyed.get_singleton.
    rewrite (dedup_renamed name_eqb (rename_name r) (name_eqb_renamed r)).
    destruct (Keyed.dedup name_eqb names) as [ | n [ | n' rest]]; simpl;
      try reflexivity.
    rewrite rename_simple_of_name. reflexivity.
Qed.

Theorem rename_choose_opt : rename_choose_opt_statement.
Proof.
  intros r [[c | ] names]; unfold Aliases.Alias_set.choose_opt; simpl.
  - rewrite is_empty_map. destruct (Keyed.is_empty names); reflexivity.
  - destruct names as [ | n names]; simpl; [ reflexivity | ].
    rewrite rename_simple_of_name. reflexivity.
Qed.

(* The structure and the canonical element so far, renamed. *)
Definition rename_state (r : renaming) (acc : Aliases.t * simple)
    : Aliases.t * simple :=
  (rename_aliases r (fst acc), rename_simple r (snd acc)).

Lemma canonical_element_of_renamed (r : renaming) (t : Aliases.t)
    (e : simple) :
  canonical_element_of (rename_aliases r t) (rename_simple r e)
  = rename_simple r (canonical_element_of t e).
Proof.
  unfold canonical_element_of. rewrite canonical_renamed.
  destruct (Aliases.canonical t e); reflexivity.
Qed.

Lemma add_alias_of_set_renamed {A : Type} (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (c elt : simple) :
  Aliases.add_alias_of_set (rename_resolver r binding_time_resolver)
    (rename_binding_times r binding_times_and_modes)
    (rename_aliases r t, rename_simple r c) (rename_simple r elt)
  = option_map (rename_state r)
      (Aliases.add_alias_of_set binding_time_resolver binding_times_and_modes
         (t, c) elt).
Proof.
  rewrite !add_alias_of_set_eq, simple_equal_renamed, rename_add.
  destruct (simple_equal c elt); [ reflexivity | ].
  destruct
    (Aliases.add binding_time_resolver binding_times_and_modes t c elt)
    as [[result | ] | ];
    reflexivity.
Qed.

Lemma fold_step_renamed {A : Type} (r : renaming)
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (todo : list simple) :
  ∀ acc : option (Aliases.t * simple),
    List.fold_left
      (step (rename_resolver r binding_time_resolver)
         (rename_binding_times r binding_times_and_modes))
      (map (rename_simple r) todo)
      (option_map (rename_state r) acc)
    = option_map (rename_state r)
        (List.fold_left (step binding_time_resolver binding_times_and_modes)
           todo acc).
Proof.
  induction todo as [ | e todo IH]; intros acc; simpl; [ reflexivity | ].
  rewrite <- IH. f_equal.
  destruct acc as [[t c] | ]; simpl; [ | reflexivity ].
  rewrite canonical_element_of_renamed. apply add_alias_of_set_renamed.
Qed.

Lemma elements_renamed (r : renaming) (aliases : Aliases.Alias_set.t) :
  elements (rename_alias_set r aliases)
  = map (rename_simple r) (elements aliases).
Proof.
  unfold elements, rename_alias_set. simpl. rewrite map_app, !map_map.
  f_equal.
  - destruct (Aliases.Alias_set.set_const aliases); reflexivity.
  - apply map_ext. intros n. symmetry. apply rename_simple_of_name.
Qed.

Theorem rename_add_alias_set : rename_add_alias_set_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t name aliases.
  rewrite !add_alias_set_eq, elements_renamed, <- rename_simple_of_name,
    canonical_element_of_renamed.
  change
    (Some
       (rename_aliases r t,
        rename_simple r (canonical_element_of t (simple_of_name name))))
    with
      (option_map (rename_state r)
         (Some (t, canonical_element_of t (simple_of_name name)))).
  rewrite
    (fold_step_renamed r binding_time_resolver binding_times_and_modes
       (elements aliases)
       (Some (t, canonical_element_of t (simple_of_name name)))).
  destruct
    (List.fold_left (step binding_time_resolver binding_times_and_modes)
       (elements aliases)
       (Some (t, canonical_element_of t (simple_of_name name))))
    as [[t' c'] | ];
    reflexivity.
Qed.

(* ======================================================================== *)
(* The calls replayed                                                       *)
(* ======================================================================== *)

Theorem rename_replay_one : rename_replay_one_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t c.
  destruct c
    as [canonical_element1 canonical_element2 | name aliases | name
       | element elt_name_mode min_name_mode min_binding_time | element | ];
    simpl.
  - rewrite
      (rename_add A r binding_time_resolver binding_times_and_modes t
         canonical_element1 canonical_element2).
    destruct
      (Aliases.add binding_time_resolver binding_times_and_modes t
         canonical_element1 canonical_element2)
      as [[result | ] | ];
      simpl; try reflexivity.
    rewrite
      (rename_invariant_add_result A r binding_time_resolver
         binding_times_and_modes result).
    reflexivity.
  - rewrite
      (rename_add_alias_set A r binding_time_resolver binding_times_and_modes
         t name aliases).
    destruct
      (Aliases.add_alias_set binding_time_resolver binding_times_and_modes t
         name aliases);
      reflexivity.
  - rewrite (rename_canonical r t name). reflexivity.
  - rewrite
      (rename_canonical_at_mode A r binding_time_resolver
         binding_times_and_modes t element elt_name_mode min_name_mode
         min_binding_time).
    destruct
      (Aliases.get_canonical_element_exn binding_time_resolver
         binding_times_and_modes t element elt_name_mode min_name_mode
         min_binding_time);
      reflexivity.
  - rewrite (rename_get_aliases r t element). reflexivity.
  - rewrite
      (rename_invariant A r binding_time_resolver binding_times_and_modes t).
    reflexivity.
Qed.

Theorem rename_replay_from : rename_replay_from_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes t calls.
  revert t.
  induction calls as [ | c calls IH]; intros t; simpl; [ reflexivity | ].
  rewrite
    (rename_replay_one A r binding_time_resolver binding_times_and_modes t c).
  destruct
    (Aliases_replay.one binding_time_resolver binding_times_and_modes t c)
    as [t' a].
  simpl. rewrite IH.
  destruct
    (Aliases_replay.run_from binding_time_resolver binding_times_and_modes t'
       calls)
    as [t'' answers].
  reflexivity.
Qed.

Theorem rename_replay : rename_replay_statement.
Proof.
  intros A r binding_time_resolver binding_times_and_modes calls.
  unfold Aliases_replay.run.
  change Aliases.empty with (rename_aliases r Aliases.empty) at 1.
  rewrite
    (rename_replay_from A r binding_time_resolver binding_times_and_modes
       Aliases.empty calls).
  reflexivity.
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Import FlambdaEnvAliasesProofs.Examples.

(* What the binary join does with the aliases of [y], a variable defined
   before a fork, at a join of two uses. At the first use [x] and [y]
   have been found equal ([x_y]). At the second they have, and [x] has
   been found to hold five. *)
Definition x_y_five : Aliases.t :=
  match Aliases.add resolver times x_y (Var x) (Const five) with
  | Some (Or_bottom.Ok result) => Aliases.add_result_t result
  | Some Or_bottom.Bottom | None => Aliases.empty
  end.

Example aliases_at_first_use :
  Aliases.get_aliases x_y (Var y)
  = Aliases.Alias_set.make None [Name_var x; Name_var y].
Proof. reflexivity. Qed.

Example aliases_at_second_use :
  Aliases.get_aliases x_y_five (Var y)
  = Aliases.Alias_set.make (Some five) [Name_var x; Name_var y].
Proof. reflexivity. Qed.

(* The aliases common to the two uses: the constant is known at one use
   only. *)
Definition common : Aliases.Alias_set.t :=
  Aliases.Alias_set.inter (Aliases.get_aliases x_y (Var y))
    (Aliases.get_aliases x_y_five (Var y)).

Example common_aliases :
  common = Aliases.Alias_set.make None [Name_var x; Name_var y].
Proof. reflexivity. Qed.

(* Without [y] itself, as the join filters them, one alias is left. *)
Definition not_y (e : simple) : bool := negb (simple_equal e (Var y)).

Example common_aliases_but_itself :
  Aliases.Alias_set.filter common not_y
  = Aliases.Alias_set.make None [Name_var x].
Proof. reflexivity. Qed.

Example the_common_alias :
  Aliases.Alias_set.get_singleton (Aliases.Alias_set.filter common not_y)
  = Some (Var x)
  ∧ Aliases.Alias_set.find_best (Aliases.Alias_set.filter common not_y)
    = Some (Var x).
Proof. split; reflexivity. Qed.

(* The join records it: in a structure that knows nothing, [y] is made
   an alias of the set, and the structure is the one in which [x] and
   [y] are aliases. *)
Example add_common_alias :
  Aliases.add_alias_set resolver times Aliases.empty (Name_var y)
    (Aliases.Alias_set.filter common not_y)
  = Some x_y.
Proof. reflexivity. Qed.

(* The best alias of a set is its constant, and failing that a
   symbol. *)
Example best_is_constant :
  Aliases.Alias_set.find_best (Aliases.get_aliases x_y_five (Var y))
  = Some (Const five).
Proof. reflexivity. Qed.

Example best_is_symbol :
  Aliases.Alias_set.find_best
    (Aliases.Alias_set.make None [Name_var x; Name_symbol s])
  = Some (Symbol s).
Proof. reflexivity. Qed.

(* A set with a constant and names is not empty and has no singleton,
   and [choose_opt] has no answer for it. *)
Example choose_of_constant_and_names :
  Aliases.Alias_set.choose_opt (Aliases.get_aliases x_y_five (Var y)) = None
  ∧ Aliases.Alias_set.is_empty (Aliases.get_aliases x_y_five (Var y)) = false.
Proof. split; reflexivity. Qed.

(* A set with a constant and a name: both names end in the class of the
   constant. *)
Example add_constant_and_name :
  ∃ t : Aliases.t,
    Aliases.add_alias_set resolver times Aliases.empty (Name_var y)
      (Aliases.Alias_set.make (Some five) [Name_var x])
    = Some t
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var x) = Const five
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var y) = Const five.
Proof. eexists. split; [ reflexivity | split; reflexivity ]. Qed.

(* Three variables: the earliest, [w], is the canonical element. *)
Example add_two_names :
  ∃ t : Aliases.t,
    Aliases.add_alias_set resolver times Aliases.empty (Name_var y)
      (Aliases.Alias_set.make None [Name_var x; Name_var w])
    = Some t
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var x) = Var w
    ∧ Aliases.get_canonical_ignoring_name_mode t (Name_var y) = Var w.
Proof. eexists. split; [ reflexivity | split; reflexivity ]. Qed.

(* [x] is known to hold five; a set that says it holds six is the fatal
   error. No environment gives [x] both values. *)
Example add_other_constant :
  Aliases.add_alias_set resolver times x_five (Name_var x)
    (Aliases.Alias_set.make (Some six) [])
  = None.
Proof. reflexivity. Qed.

(* The calls replayed (FlambdaEnvAliasesReplay.v), as a test that
   compares the model with the compiler makes them. [y] and [x] are
   found equal, then [x] is found to hold five: the answers on the
   way. *)
Example replay_answers :
  Aliases_replay.run resolver times
    [Aliases_replay.Add (Var y) (Var x);
     Aliases_replay.Get_canonical_ignoring_name_mode (Name_var y);
     Aliases_replay.Add (Var x) (Const five);
     Aliases_replay.Get_aliases (Var y);
     Aliases_replay.Get_canonical_element_exn (Var y) Name_mode.Normal
       Name_mode.Normal 3;
     Aliases_replay.Invariant]
  = [Aliases_replay.Added (Name_var y) (Var x) true;
     Aliases_replay.Canonical_element (Var x);
     Aliases_replay.Added (Name_var x) (Const five) true;
     Aliases_replay.Aliases_of
       (Aliases.Alias_set.make (Some five) [Name_var x; Name_var y]);
     Aliases_replay.Canonical_element (Const five);
     Aliases_replay.Invariant_holds true].
Proof. reflexivity. Qed.

(* Where a call has no structure to give, the structure is kept. A set
   with another constant than the one [x] holds is a fatal error; two
   constants are bottom; two symbols are made aliases, and the result
   does not pass the compiler's check. *)
Example replay_stops :
  Aliases_replay.run resolver times
    [Aliases_replay.Add (Var x) (Const five);
     Aliases_replay.Add_alias_set (Name_var w)
       (Aliases.Alias_set.make (Some six) [Name_var x]);
     Aliases_replay.Add (Const five) (Const six);
     Aliases_replay.Get_canonical_ignoring_name_mode (Name_var x);
     Aliases_replay.Add (Symbol s) (Symbol s');
     Aliases_replay.Invariant]
  = [Aliases_replay.Added (Name_var x) (Const five) true;
     Aliases_replay.Fatal_error;
     Aliases_replay.Added_bottom;
     Aliases_replay.Canonical_element (Const five);
     Aliases_replay.Added (Name_symbol s) (Symbol s') false;
     Aliases_replay.Invariant_holds false].
Proof. reflexivity. Qed.

End Examples.
