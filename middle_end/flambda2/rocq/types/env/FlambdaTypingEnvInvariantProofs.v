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


(* Proofs of the statements of FlambdaTypingEnvInvariantSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool Lia Setoid Sorted.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaPrimitiveProofs FlambdaSemanticsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeGrammarProofs FlambdaTypeWellFormedSpec
  FlambdaTypeWellFormedProofs FlambdaTypeMeaningSpec FlambdaBindingTime
  FlambdaEnvAliases FlambdaEnvAliasesSpec FlambdaEnvAliasesProofs
  FlambdaEnvAliasesInvariantSpec FlambdaEnvAliasesInvariantProofs
  FlambdaTypingEnv FlambdaTypingEnvSpec FlambdaTypingEnvProofs
  FlambdaTypingEnvInvariantSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The tests and the conditions                                             *)
(* ======================================================================== *)

(* A test that goes through a map of names answers [true] exactly when
   what it tests holds of every binding that counts. *)
Lemma for_all_names_b_iff {A : Type} (P : name -> A -> Prop)
    (p : name -> A -> bool) (m : Keyed.finite_map name A) :
  (∀ (n : name) (a : A),
     Keyed.lookup name_eqb n m = Some a -> (p n a = true <-> P n a)) ->
  (Keyed.for_all_bindings_b name_eqb p m = true
   <-> (∀ (n : name) (a : A), Keyed.lookup name_eqb n m = Some a -> P n a)).
Proof.
  intros Hreflects. eapply iff_trans.
  - exact (keyed_for_all_bindings_b_reflects name_eqb name_eqb_eq P p m
             Hreflects).
  - exact (keyed_for_all_bindings_lookup name name_eqb name_eqb_eq A P m).
Qed.

Lemma binding_times_bounded_b_iff (tenv : Typing_env.t) :
  binding_times_bounded_b tenv = true <-> binding_times_bounded tenv.
Proof.
  unfold binding_times_bounded_b, binding_times_bounded.
  rewrite andb_true_iff, !Z.leb_le. reflexivity.
Qed.

Lemma binding_times_in_range_b_iff (tenv : Typing_env.t) :
  binding_times_in_range_b tenv = true <-> binding_times_in_range tenv.
Proof.
  unfold binding_times_in_range_b, binding_times_in_range.
  eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n entry =>
                match n with
                | Name_var _ =>
                  Binding_time.earliest_var <= binding_time_of entry
                  ∧ binding_time_of entry < Typing_env.next_binding_time tenv
                | Name_symbol _ => True
                end)).
    intros [x | s] entry _.
    + rewrite andb_true_iff, Z.leb_le, Z.ltb_lt. reflexivity.
    + split; [ intros _; exact I | reflexivity ].
  - split.
    + intros Hall x ty btm Hlookup.
      exact (Hall (Name_var x) (ty, btm) Hlookup).
    + intros Hall [x | s] [ty btm] Hlookup;
        [ exact (Hall x ty btm Hlookup) | exact I ].
Qed.

Lemma binding_times_distinct_b_iff (tenv : Typing_env.t) :
  binding_times_distinct_b tenv = true <-> binding_times_distinct tenv.
Proof.
  unfold binding_times_distinct_b, binding_times_distinct.
  eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n entry =>
                ∀ (n' : name)
                  (entry' : flambda_type * Binding_time.With_name_mode.t),
                  Keyed.lookup name_eqb n' (Typing_env.names_to_types tenv)
                  = Some entry' ->
                  match n, n' with
                  | Name_var x, Name_var y =>
                    x = y ∨ binding_time_of entry <> binding_time_of entry'
                  | _, _ => True
                  end)).
    intros n entry _. apply for_all_names_b_iff. intros n' entry' _.
    destruct n as [x | s], n' as [y | s'];
      try (split; [ intros _; exact I | reflexivity ]).
    rewrite orb_true_iff, Pos.eqb_eq, negb_true_iff, Z.eqb_neq. reflexivity.
  - split.
    + intros Hall x y ty ty' btm btm' Hx Hy Htime.
      destruct (Hall (Name_var x) (ty, btm) Hx (Name_var y) (ty', btm') Hy)
        as [Heq | Hne]; [ exact Heq | exfalso; exact (Hne Htime) ].
    + intros Hall [x | s] [ty btm] Hx [y | s'] [ty' btm'] Hy; try exact I.
      destruct (Z.eq_dec (binding_time_of (ty, btm))
                  (binding_time_of (ty', btm'))) as [Heq | Hne];
        [ left; exact (Hall x y ty ty' btm btm' Hx Hy Heq)
        | right; exact Hne ].
Qed.

Lemma types_closed_b_iff (tenv : Typing_env.t) :
  types_closed_b tenv = true <-> types_closed tenv.
Proof.
  unfold types_closed_b, types_closed. eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun _ entry =>
                ∀ x : variable,
                  x ∈ free_variables (fst entry) ->
                  Keyed.is_bound name_eqb (Name_var x)
                    (Typing_env.names_to_types tenv)
                  = true)).
    intros n entry _. apply forallb_forall.
  - split.
    + intros Hall n ty btm x Hlookup. exact (Hall n (ty, btm) Hlookup x).
    + intros Hall n [ty btm] Hlookup x. exact (Hall n ty btm x Hlookup).
Qed.

Lemma with_name_mode_equal_eq
    : decides_equality Binding_time.With_name_mode.equal.
Proof.
  intros [time mode] [time' mode'].
  unfold Binding_time.With_name_mode.equal, Binding_time.equal. simpl.
  rewrite andb_true_iff, Z.eqb_eq, (name_mode_equal_eq mode mode'). split.
  - intros [-> ->]. reflexivity.
  - intros [= -> ->]. split; reflexivity.
Qed.

Lemma symbols_recorded_b_iff (tenv : Typing_env.t) :
  symbols_recorded_b tenv = true <-> symbols_recorded tenv.
Proof.
  unfold symbols_recorded_b, symbols_recorded. eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n entry =>
                match n with
                | Name_symbol _ =>
                  snd entry = Binding_time.With_name_mode.symbols
                | Name_var _ => True
                end)).
    intros [x | s] entry _; [ | apply with_name_mode_equal_eq ].
    split; [ intros _; exact I | reflexivity ].
  - split.
    + intros Hall s ty btm Hlookup.
      exact (Hall (Name_symbol s) (ty, btm) Hlookup).
    + intros Hall [x | s] [ty btm] Hlookup;
        [ exact I | exact (Hall s ty btm Hlookup) ].
Qed.

Lemma symbols_defined_b_iff (tenv : Typing_env.t) :
  symbols_defined_b tenv = true <-> symbols_defined tenv.
Proof.
  unfold symbols_defined_b, symbols_defined. eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n _ =>
                match n with
                | Name_symbol s =>
                  among s (Typing_env.defined_symbols tenv) = true
                | Name_var _ => True
                end)).
    intros [x | s] entry _; [ | reflexivity ].
    split; [ intros _; exact I | reflexivity ].
  - split.
    + intros Hall s ty btm Hlookup.
      exact (Hall (Name_symbol s) (ty, btm) Hlookup).
    + intros Hall [x | s] [ty btm] Hlookup;
        [ exact I | exact (Hall s ty btm Hlookup) ].
Qed.

(* Two maps bind no key in common when the intersection of their
   domains is empty. *)
Lemma is_empty_inter_domain {K A B : Type} (eqb : K -> K -> bool)
    (Heqb : decides_equality eqb) (m : Keyed.finite_map K A)
    (m' : Keyed.finite_map K B) :
  Keyed.is_empty
    (Keyed.inter eqb (Keyed.domain eqb m) (Keyed.domain eqb m'))
  = true
  <-> (∀ key : K,
        Keyed.is_bound eqb key m = true -> Keyed.is_bound eqb key m' = false).
Proof.
  eapply iff_trans; [ apply (keyed_is_empty_among K eqb Heqb) | ].
  split; intros Hall key.
  - intros Hm. specialize (Hall key).
    rewrite (keyed_among_inter K eqb Heqb), !(keyed_among_domain K eqb Heqb),
      Hm in Hall.
    exact Hall.
  - rewrite (keyed_among_inter K eqb Heqb), !(keyed_among_domain K eqb Heqb).
    destruct (Keyed.is_bound eqb key m) eqn:Hm;
      [ exact (Hall key Hm) | reflexivity ].
Qed.

Theorem levels_disjoint_b_iff : levels_disjoint_b_iff_statement.
Proof.
  intros earlier later. unfold levels_disjoint_b, levels_disjoint.
  rewrite andb_true_iff.
  rewrite (is_empty_inter_domain Z.eqb z_eqb_decides_equality).
  rewrite <- (is_empty_inter_domain Pos.eqb pos_eqb_decides_equality
                (Typing_env_level.defined_vars earlier)
                (Typing_env_level.defined_vars later)).
  reflexivity.
Qed.

Theorem concat_defined : concat_defined_statement.
Proof.
  intros earlier later.
  rewrite <- (levels_disjoint_b_iff earlier later).
  unfold Typing_env_level.concat, levels_disjoint_b.
  match goal with
  | |- context [if ?test then _ else _] => destruct test
  end.
  - split; [ reflexivity | discriminate ].
  - split; [ intros Hne; exfalso; apply Hne; reflexivity | discriminate ].
Qed.

Lemma levels_pairwise_disjoint_b_iff (levels : list Typing_env_level.t) :
  levels_pairwise_disjoint_b levels = true <-> levels_pairwise_disjoint levels.
Proof.
  induction levels as [ | later levels IH]; simpl.
  - split; [ intros _; exact I | reflexivity ].
  - rewrite andb_true_iff, IH, forallb_forall.
    split; intros [Hall Hrest]; (split; [ | exact Hrest ]);
      intros earlier Hin; apply levels_disjoint_b_iff, Hall, Hin.
Qed.

Lemma defined_variables_bound_b_iff (tenv : Typing_env.t) :
  defined_variables_bound_b tenv = true <-> defined_variables_bound tenv.
Proof.
  unfold defined_variables_bound_b, defined_variables_bound.
  rewrite forallb_forall.
  assert (Hlevel :
    ∀ level : Typing_env_level.t,
      for_all_bindings_b
        (fun x k =>
           match
             Keyed.lookup name_eqb (Name_var x)
               (Typing_env.names_to_types tenv)
           with
           | Some (ty, _) => kind_eqb (kind ty) k
           | None => false
           end)
        (Typing_env_level.defined_vars level)
      = true
      <-> (∀ (x : variable) (k : flambda_kind),
            lookup x (Typing_env_level.defined_vars level) = Some k ->
            ∃ (ty : flambda_type)
              (binding_time_and_mode : Binding_time.With_name_mode.t),
              Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv)
              = Some (ty, binding_time_and_mode)
              ∧ kind ty = k)).
  { intros level. eapply iff_trans.
    - apply (keyed_for_all_bindings_b_reflects Pos.eqb
               pos_eqb_decides_equality
               (fun x k =>
                  ∃ (ty : flambda_type)
                    (binding_time_and_mode : Binding_time.With_name_mode.t),
                    Keyed.lookup name_eqb (Name_var x)
                      (Typing_env.names_to_types tenv)
                    = Some (ty, binding_time_and_mode)
                    ∧ kind ty = k)).
      intros x k _.
      destruct (Keyed.lookup name_eqb (Name_var x)
                  (Typing_env.names_to_types tenv)) as [[ty btm] | ].
      + rewrite (kind_eqb_eq (kind ty) k). split.
        * intros Hkind. exists ty, btm. split; [ reflexivity | exact Hkind ].
        * intros [ty' [btm' [[= <- <-] Hkind]]]. exact Hkind.
      + split; [ discriminate | ].
        intros [ty [btm [Hnone _]]]. discriminate.
    - exact (for_all_bindings_lookup _ _ _). }
  split.
  - intros Hall level x k Hin.
    exact (proj1 (Hlevel level) (Hall level Hin) x k).
  - intros Hall level Hin. apply Hlevel. intros x k.
    exact (Hall level x k Hin).
Qed.

Lemma indexed_at_b_iff (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) :
  indexed_at_b level binding_time x = true
  <-> indexed_at level binding_time x.
Proof.
  unfold indexed_at_b, indexed_at.
  destruct (Keyed.lookup Z.eqb binding_time
              (Typing_env_level.binding_times level)) as [vars | ].
  - rewrite among_In. split.
    + intros Hin. exists vars. split; [ reflexivity | exact Hin ].
    + intros [vars' [[= <-] Hin]]. exact Hin.
  - split; [ discriminate | ]. intros [vars [Hnone _]]. discriminate.
Qed.

Theorem level_indexed_b_iff : level_indexed_b_iff_statement.
Proof.
  intros level. unfold level_indexed_b, level_indexed.
  rewrite andb_true_iff, forallb_forall.
  rewrite (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality),
    (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality).
  split.
  - intros [Hindex Hdefined] x. split.
    + intros Hbound. rewrite <- among_domain in Hbound.
      apply among_In in Hbound. specialize (Hdefined x Hbound).
      apply existsb_exists in Hdefined.
      destruct Hdefined as [binding_time [_ Hat]].
      exists binding_time. apply indexed_at_b_iff. exact Hat.
    + intros [binding_time [vars [Hlookup Hin]]].
      specialize (Hindex binding_time vars Hlookup).
      rewrite <- among_domain. apply (proj1 (all_among _ _) Hindex).
      apply among_In. exact Hin.
  - intros Hlevel. split.
    + intros binding_time vars Hlookup. apply all_among. intros x Hx.
      rewrite among_domain. apply Hlevel. exists binding_time, vars.
      split; [ exact Hlookup | apply among_In; exact Hx ].
    + intros x Hx. apply existsb_exists.
      apply among_In in Hx. rewrite among_domain in Hx.
      destruct (proj1 (Hlevel x) Hx) as [binding_time Hat].
      exists binding_time. split; [ | apply indexed_at_b_iff; exact Hat ].
      apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
      rewrite (keyed_among_domain Z Z.eqb z_eqb_decides_equality).
      destruct Hat as [vars [Hlookup _]].
      apply keyed_is_bound_lookup. exists vars. exact Hlookup.
Qed.

Lemma levels_indexed_b_iff (tenv : Typing_env.t) :
  levels_indexed_b tenv = true <-> levels_indexed tenv.
Proof.
  unfold levels_indexed_b, levels_indexed. rewrite forallb_forall.
  split; intros Hall level Hin; apply level_indexed_b_iff, Hall, Hin.
Qed.

Lemma level_binding_times_recorded_b_iff (tenv : Typing_env.t) :
  level_binding_times_recorded_b tenv = true
  <-> level_binding_times_recorded tenv.
Proof.
  unfold level_binding_times_recorded_b, level_binding_times_recorded.
  rewrite forallb_forall.
  assert (Hentry :
    ∀ (binding_time : Binding_time.t) (x : variable),
      match
        Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
      with
      | Some entry => binding_time_of entry =? binding_time
      | None => false
      end
      = true
      <-> ∃ (ty : flambda_type)
            (binding_time_and_mode : Binding_time.With_name_mode.t),
            Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)
            = Some (ty, binding_time_and_mode)
            ∧ Binding_time.With_name_mode.binding_time binding_time_and_mode
              = binding_time).
  { intros binding_time x.
    destruct (Keyed.lookup name_eqb (Name_var x)
                (Typing_env.names_to_types tenv)) as [[ty btm] | ].
    - unfold binding_time_of. simpl. rewrite Z.eqb_eq. split.
      + intros Htime. exists ty, btm. split; [ reflexivity | exact Htime ].
      + intros [ty' [btm' [[= <- <-] Htime]]]. exact Htime.
    - split; [ discriminate | ]. intros [ty [btm [Hnone _]]]. discriminate. }
  split.
  - intros Hall level binding_time x Hin [vars [Hlookup Hx]].
    specialize (Hall level Hin).
    rewrite (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality),
      (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality) in Hall.
    specialize (Hall binding_time vars Hlookup).
    rewrite forallb_forall in Hall. apply Hentry. exact (Hall x Hx).
  - intros Hall level Hin.
    rewrite (keyed_for_all_bindings_b Z Z.eqb z_eqb_decides_equality),
      (keyed_for_all_bindings_lookup Z Z.eqb z_eqb_decides_equality).
    intros binding_time vars Hlookup. apply forallb_forall. intros x Hx.
    apply Hentry. apply (Hall level binding_time x Hin).
    exists vars. split; [ exact Hlookup | exact Hx ].
Qed.

Lemma level_binding_times_in_range_b_iff (tenv : Typing_env.t) :
  level_binding_times_in_range_b tenv = true
  <-> level_binding_times_in_range tenv.
Proof.
  unfold level_binding_times_in_range_b, level_binding_times_in_range.
  rewrite forallb_forall. split.
  - intros Hall level binding_time Hin Hbound.
    specialize (Hall level Hin). rewrite forallb_forall in Hall.
    apply Z.ltb_lt, Hall.
    apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
    rewrite (keyed_among_domain Z Z.eqb z_eqb_decides_equality).
    exact Hbound.
  - intros Hall level Hin. apply forallb_forall. intros binding_time Hkey.
    apply Z.ltb_lt, (Hall level binding_time Hin).
    rewrite <- (keyed_among_domain Z Z.eqb z_eqb_decides_equality).
    apply (keyed_among_in Z Z.eqb z_eqb_decides_equality). exact Hkey.
Qed.

Lemma scopes_decreasing_b_iff (scopes : list Scope.t) :
  scopes_decreasing_b scopes = true <-> scopes_decreasing scopes.
Proof.
  induction scopes as [ | later scopes IH]; simpl.
  - split; [ intros _; exact I | reflexivity ].
  - rewrite andb_true_iff, IH.
    destruct scopes as [ | earlier scopes].
    + split; intros [_ Hrest]; (split; [ | exact Hrest ]);
        [ exact I | reflexivity ].
    + rewrite Z.ltb_lt. reflexivity.
Qed.

Lemma variables_defined_b_iff (tenv : Typing_env.t) :
  variables_defined_b tenv = true <-> variables_defined tenv.
Proof.
  unfold variables_defined_b, variables_defined. eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n _ =>
                match n with
                | Name_var x =>
                  ∃ level : Typing_env_level.t,
                    level ∈ all_levels tenv
                    ∧ is_bound x (Typing_env_level.defined_vars level) = true
                | Name_symbol _ => True
                end)).
    intros [x | s] entry _;
      [ apply existsb_exists | split; [ intros _; exact I | reflexivity ] ].
  - split.
    + intros Hall x ty btm Hlookup.
      exact (Hall (Name_var x) (ty, btm) Hlookup).
    + intros Hall [x | s] [ty btm] Hlookup;
        [ exact (Hall x ty btm Hlookup) | exact I ].
Qed.

Lemma aliases_known_b_iff (tenv : Typing_env.t) :
  aliases_known_b tenv = true <-> aliases_known tenv.
Proof.
  unfold aliases_known_b, aliases_known, demoted_to.
  apply (for_all_names_b_iff
           (fun n canonical_element =>
              Typing_env.mem None tenv n = true
              ∧ Typing_env.mem_simple None tenv canonical_element = true)).
  intros n canonical_element _. apply andb_true_iff.
Qed.

Lemma canonical_names_concrete_b_iff (tenv : Typing_env.t) :
  canonical_names_concrete_b tenv = true <-> canonical_names_concrete tenv.
Proof.
  unfold canonical_names_concrete_b, canonical_names_concrete.
  eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun n entry =>
                ∀ alias : simple,
                  get_alias_opt (fst entry) = Some alias ->
                  ~ is_canonical (Typing_env.aliases tenv)
                      (simple_of_name n))).
    intros n entry _.
    destruct (get_alias_opt (fst entry)) as [alias | ].
    + rewrite negb_true_iff. split.
      * intros Hfalse alias' _ Hcanonical.
        apply is_canonical_b_iff in Hcanonical. congruence.
      * intros Hnot.
        destruct (is_canonical_b (Typing_env.aliases tenv) (simple_of_name n))
          eqn:Htest; [ exfalso | reflexivity ].
        apply (Hnot alias eq_refl). apply is_canonical_b_iff. exact Htest.
    + split; [ intros _ alias Hnone; discriminate | reflexivity ].
  - split.
    + intros Hall n ty btm alias Hlookup.
      exact (Hall n (ty, btm) Hlookup alias).
    + intros Hall n [ty btm] Hlookup alias.
      exact (Hall n ty btm alias Hlookup).
Qed.

Theorem typing_env_well_formed_iff : typing_env_well_formed_iff_statement.
Proof.
  intros tenv. unfold typing_env_well_formed, typing_env_conditions.
  rewrite !andb_true_iff.
  rewrite binding_times_bounded_b_iff, binding_times_in_range_b_iff,
    binding_times_distinct_b_iff, types_closed_b_iff, symbols_defined_b_iff,
    symbols_recorded_b_iff,
    levels_pairwise_disjoint_b_iff, defined_variables_bound_b_iff,
    levels_indexed_b_iff, level_binding_times_recorded_b_iff,
    level_binding_times_in_range_b_iff, scopes_decreasing_b_iff,
    variables_defined_b_iff,
    (aliases_consistent_b_iff (Typing_env.aliases tenv)),
    (name_modes_recorded_b_iff _ (Typing_env.names_to_types tenv)
       (Typing_env.aliases tenv)),
    aliases_known_b_iff, canonical_names_concrete_b_iff.
  tauto.
Qed.

(* ======================================================================== *)
(* What the invariant gives                                                 *)
(* ======================================================================== *)

(* A variable is bound when it has an entry. *)
Lemma mem_var_lookup (tenv : Typing_env.t) (x : variable) :
  Typing_env.mem None tenv (Name_var x) = true
  <-> ∃ (ty : flambda_type)
        (binding_time_and_mode : Binding_time.With_name_mode.t),
        Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
        = Some (ty, binding_time_and_mode).
Proof.
  unfold Typing_env.mem.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)) as [[ty btm] | ].
  - split; [ intros _; exists ty, btm; reflexivity | reflexivity ].
  - split; [ discriminate | intros [ty [btm Hnone]]; discriminate ].
Qed.

Lemma mem_var_is_bound (tenv : Typing_env.t) (x : variable) :
  Typing_env.mem None tenv (Name_var x)
  = Keyed.is_bound name_eqb (Name_var x) (Typing_env.names_to_types tenv).
Proof.
  unfold Typing_env.mem, Keyed.is_bound.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)) as [[ty btm] | ];
    reflexivity.
Qed.

Theorem well_formed_binding_times_distinct
    : well_formed_binding_times_distinct_statement.
Proof.
  intros tenv x y Hwf Hx Hy Htime.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [[_ [_ [Hdistinct _]]] _].
  apply mem_var_lookup in Hx. destruct Hx as [ty [btm Hx]].
  apply mem_var_lookup in Hy. destruct Hy as [ty' [btm' Hy]].
  apply (Hdistinct x y ty ty' btm btm' Hx Hy).
  revert Htime. unfold binding_time_in, Aliases.binding_time_and_name_mode.
  rewrite Hx, Hy. exact (fun Htime => Htime).
Qed.

Theorem well_formed_mentioned_known : well_formed_mentioned_known_statement.
Proof.
  intros tenv n Hwf Hmention.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [_ [[_ Hlisted] [_ [Hknown _]]]]].
  assert (Hcanonical :
    ∀ n' : name,
      demoted_to (Typing_env.aliases tenv) n' (simple_of_name n) ->
      Typing_env.mem None tenv n = true).
  { intros n' Hdemoted. destruct (Hknown n' _ Hdemoted) as [_ Hmem].
    destruct n; exact Hmem. }
  destruct Hmention
    as [[c Hdemoted] | [[n' Hdemoted] | [[c Hlisted'] | [n' Hlisted']]]].
  - exact (proj1 (Hknown n c Hdemoted)).
  - exact (Hcanonical n' Hdemoted).
  - exact (proj1 (Hknown n c (proj2 (Hlisted n c) Hlisted'))).
  - exact (Hcanonical n' (proj2 (Hlisted n' (simple_of_name n)) Hlisted')).
Qed.

Lemma free_variables_bottom (k : flambda_kind) :
  free_variables (bottom k) = [].
Proof. destruct k as [ | [] | | ]; reflexivity. Qed.

Theorem find_closed : find_closed_statement.
Proof.
  intros tenv n kind ty x Hclosed Hfind Hin.
  destruct (find_cases tenv n kind ty Hfind)
    as [[binding_time_and_mode Hlookup] | [[k ->] | ->]].
  - rewrite mem_var_is_bound.
    exact (Hclosed n ty binding_time_and_mode x Hlookup Hin).
  - rewrite free_variables_bottom in Hin. destruct Hin.
  - destruct Hin.
Qed.

(* Two different simples that are known to be equal: the aliases speak
   of each that is a name. *)
Lemma aliased_mentioned (t : Aliases.t) (e e' : simple) (n : name) :
  aliased t e e' ->
  e <> e' ->
  e = simple_of_name n ->
  aliases_mention t n.
Proof.
  unfold aliased. intros Haliased Hne ->.
  destruct (canonical_cases t (simple_of_name n))
    as [[_ Hself] | [n0 [c [Heq [Hdemoted _]]]]].
  - destruct (canonical_cases t e')
      as [[_ Hself'] | [n' [c' [-> [Hdemoted' [_ Hcanonical']]]]]].
    + congruence.
    + right. left. exists n'. rewrite Hself, Hcanonical' in Haliased.
      rewrite Haliased. exact Hdemoted'.
  - apply simple_of_name_inj in Heq. subst n0. left. exists c.
    exact Hdemoted.
Qed.

Lemma name_mode_at_least_compare (name_mode min_name_mode : Name_mode.t) :
  name_mode_at_least name_mode min_name_mode ->
  match Name_mode.compare_partial_order min_name_mode name_mode with
  | None => false
  | Some c => c <=? 0
  end
  = true.
Proof.
  destruct name_mode, min_name_mode; intros [c [[= <-] Hc]];
    try reflexivity; lia.
Qed.

Theorem get_canonical_simple_bound : get_canonical_simple_bound_statement.
Proof.
  intros tenv min_name_mode simple canonical Hwf Hbound.
  pose proof (proj1 (typing_env_well_formed_iff tenv) Hwf) as Hconditions.
  destruct Hconditions as [_ [_ [Hconsistent _]]].
  unfold Typing_env.get_canonical_simple_exn.
  destruct (Typing_env.binding_time_and_mode_of_simple tenv simple)
    as [btm_simple | ] eqn:Hbtm; [ | discriminate ].
  cbn [option_map]. intros [= Hcanonical].
  (* The canonical element is bound. *)
  assert (Hknown : Typing_env.mem_simple None tenv canonical = true).
  { destruct (simple_equal canonical simple) eqn:Hequal.
    - apply simple_equal_decides_equality in Hequal. subst canonical.
      exact Hbound.
    - assert (Hne : canonical <> simple).
      { intros Heq. apply simple_equal_decides_equality in Heq.
        congruence. }
      pose proof (canonical_at_mode_aliased _ _ _ _ _ _ _ _ _ Hconsistent
                    Hcanonical) as Haliased.
      destruct canonical as [x | s | c]; [ | | reflexivity ].
      + exact (well_formed_mentioned_known tenv _ Hwf
                 (aliased_mentioned _ _ _ (Name_var x) Haliased Hne eq_refl)).
      + exact (well_formed_mentioned_known tenv _ Hwf
                 (aliased_mentioned _ _ _ (Name_symbol s) Haliased Hne
                    eq_refl)). }
  destruct canonical as [x | s | c]; [ | exact Hknown | reflexivity ].
  change (Typing_env.mem None tenv (Name_var x) = true) in Hknown.
  apply mem_var_lookup in Hknown. destruct Hknown as [ty [btm Hlookup]].
  assert (Hmode :
    name_mode_at_least
      (Binding_time.With_name_mode.scoped_name_mode btm
         (Typing_env.min_binding_time tenv))
      min_name_mode).
  { destruct (canonical_at_mode_name_mode _ _ _ _ _ _ _ _ _ Hcanonical)
      as [[Heq Hmode] | Hmode].
    - subst simple.
      unfold Typing_env.binding_time_and_mode_of_simple,
        Typing_env.binding_time_and_mode,
        Typing_env.find_with_binding_time_and_mode,
        Typing_env.find_with_binding_time_and_mode' in Hbtm.
      rewrite Hlookup in Hbtm. cbn in Hbtm.
      destruct (Typing_env.is_bottom tenv); cbn in Hbtm;
        injection Hbtm as <-; exact Hmode.
    - unfold Aliases.compute_name_mode, Aliases.binding_time_and_name_mode
        in Hmode.
      rewrite Hlookup in Hmode. exact Hmode. }
  unfold Typing_env.mem_simple, Typing_env.mem. rewrite Hlookup.
  exact (name_mode_at_least_compare _ _ Hmode).
Qed.

Theorem as_extension_without_bindings_equations
    : as_extension_without_bindings_equations_statement.
Proof.
  intros level equations [= <-]. split.
  - intros n. apply (keyed_lookup_compact name name_eqb name_eqb_eq).
  - apply (keyed_compact_distinct name name_eqb name_eqb_eq).
Qed.

(* ------------------------------------------------------------------------ *)
(* Cutting                                                                  *)
(* ------------------------------------------------------------------------ *)

Lemma is_bound_app {A : Type} (x : positive) (m m' : finite_map A) :
  is_bound x (m ++ m') = is_bound x m || is_bound x m'.
Proof.
  unfold is_bound. rewrite lookup_app. destruct (lookup x m); reflexivity.
Qed.

Lemma keyed_is_bound_app {K A : Type} (eqb : K -> K -> bool) (key : K)
    (m m' : Keyed.finite_map K A) :
  Keyed.is_bound eqb key (m ++ m')
  = Keyed.is_bound eqb key m || Keyed.is_bound eqb key m'.
Proof.
  unfold Keyed.is_bound. rewrite keyed_lookup_app.
  destruct (Keyed.lookup eqb key m); reflexivity.
Qed.

(* What [Typing_env_level.concat] answers. *)
Lemma concat_some (earlier later result : Typing_env_level.t) :
  Typing_env_level.concat earlier later = Some result ->
  result
  = Typing_env_level.create
      (Typing_env_level.defined_vars later
       ++ Typing_env_level.defined_vars earlier)
      (Typing_env_level.binding_times later
       ++ Typing_env_level.binding_times earlier)
      (Typing_env_level.equations later
       ++ Typing_env_level.equations earlier).
Proof.
  unfold Typing_env_level.concat.
  match goal with
  | |- context [if ?test then _ else _] => destruct test
  end; [ | discriminate ].
  intros [= <-]. reflexivity.
Qed.

(* A level that is disjoint from two levels is disjoint from their
   concatenation. *)
Lemma levels_disjoint_concat (level earlier later result : Typing_env_level.t)
    :
  Typing_env_level.concat earlier later = Some result ->
  levels_disjoint level earlier ->
  levels_disjoint level later ->
  levels_disjoint level result.
Proof.
  intros Hconcat [Hvars Htimes] [Hvars' Htimes'].
  apply concat_some in Hconcat. subst result. split.
  - intros x Hx. simpl. rewrite is_bound_app, (Hvars x Hx), (Hvars' x Hx).
    reflexivity.
  - intros binding_time Hbt. simpl.
    rewrite keyed_is_bound_app, (Htimes binding_time Hbt),
      (Htimes' binding_time Hbt).
    reflexivity.
Qed.

Lemma cut_loop_defined (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ result : Typing_env_level.t,
    (∀ earlier : Typing_env_level.t,
      earlier ∈ map Typing_env.One_level.level levels ->
      levels_disjoint earlier result) ->
    levels_pairwise_disjoint (map Typing_env.One_level.level levels) ->
    ∃ level : Typing_env_level.t,
      Typing_env.cut_loop cut_after result levels = Some level.
Proof.
  induction levels as [ | one_level levels IH]; intros result Hresult Hlevels.
  - exists result. reflexivity.
  - simpl. destruct (cut_after <? Typing_env.One_level.scope one_level);
      [ | exists result; reflexivity ].
    destruct Hlevels as [Hone Hlevels].
    destruct (Typing_env_level.concat (Typing_env.One_level.level one_level)
                result) as [result' | ] eqn:Hconcat.
    + simpl. apply IH; [ | exact Hlevels ].
      intros earlier Hin.
      apply (levels_disjoint_concat earlier _ _ _ Hconcat).
      * exact (Hone earlier Hin).
      * apply Hresult. right. exact Hin.
    + exfalso. apply (proj2 (concat_defined _ _)) in Hconcat;
        [ exact Hconcat | ].
      apply Hresult. left. reflexivity.
Qed.

Theorem cut_defined : cut_defined_statement.
Proof.
  intros tenv cut_after Hwf.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [[[Hcurrent Hprev] _] _]].
  unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after);
    [ exists Typing_env_level.empty; reflexivity | ].
  apply cut_loop_defined; [ exact Hcurrent | exact Hprev ].
Qed.

(* The level that [cut_loop] answers is the one it is given followed by
   the levels that it has gone through, which are the first of those
   that it is given. *)
Lemma cut_loop_parts (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ result level : Typing_env_level.t,
    Typing_env.cut_loop cut_after result levels = Some level ->
    ∃ taken rest : list Typing_env_level.t,
      map Typing_env.One_level.level levels = taken ++ rest
      ∧ Typing_env_level.equations level
        = Typing_env_level.equations result
          ++ flat_map Typing_env_level.equations taken
      ∧ Typing_env_level.defined_vars level
        = Typing_env_level.defined_vars result
          ++ flat_map Typing_env_level.defined_vars taken.
Proof.
  induction levels as [ | one_level levels IH]; intros result level.
  - intros [= <-]. exists [], []. simpl. rewrite !app_nil_r.
    split; [ reflexivity | split; reflexivity ].
  - simpl. destruct (cut_after <? Typing_env.One_level.scope one_level).
    + destruct (Typing_env_level.concat
                  (Typing_env.One_level.level one_level) result)
        as [result' | ] eqn:Hconcat; [ | discriminate ].
      simpl. intros Hloop.
      destruct (IH result' level Hloop)
        as [taken [rest [Hlevels [Hequations Hdefined]]]].
      apply concat_some in Hconcat. subst result'. simpl in *.
      exists (Typing_env.One_level.level one_level :: taken), rest.
      split; [ rewrite Hlevels; reflexivity | ]. simpl.
      rewrite Hequations, Hdefined, <- !app_assoc.
      split; reflexivity.
    + intros [= <-].
      exists [], (Typing_env.One_level.level one_level
                    :: map Typing_env.One_level.level levels).
      simpl. rewrite !app_nil_r.
      split; [ reflexivity | split; reflexivity ].
Qed.

Theorem cut_equations_found : cut_equations_found_statement.
Proof.
  intros tenv cut_after level n ty Hagree. unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after).
  - intros [= <-]. discriminate.
  - intros Hloop Hlookup.
    destruct (cut_loop_parts _ _ _ _ Hloop)
      as [taken [rest [Hlevels [Hequations _]]]].
    apply (Hagree n ty). unfold all_equations, all_levels. simpl.
    rewrite Hlevels, flat_map_app, app_assoc, <- Hequations.
    rewrite keyed_lookup_app, Hlookup. reflexivity.
Qed.

(* What is found in the bindings of several levels is found in those of
   one of them. *)
Lemma lookup_flat_map {A B : Type} (f : A -> finite_map B) (x : positive)
    (b : B) (l : list A) :
  lookup x (flat_map f l) = Some b -> ∃ a : A, a ∈ l ∧ lookup x (f a) = Some b.
Proof.
  induction l as [ | a l IH]; simpl; [ discriminate | ].
  rewrite lookup_app. destruct (lookup x (f a)) as [b' | ] eqn:Hlookup.
  - intros [= ->]. exists a. split; [ left; reflexivity | exact Hlookup ].
  - intros Hrest. destruct (IH Hrest) as [a' [Hin Ha']].
    exists a'. split; [ right; exact Hin | exact Ha' ].
Qed.

Lemma cut_loop_equations_member (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ (result level : Typing_env_level.t) (n : name) (ty : flambda_type),
    Typing_env.cut_loop cut_after result levels = Some level ->
    (n, ty) ∈ Typing_env_level.equations level ->
    (n, ty) ∈ Typing_env_level.equations result
    ∨ ∃ one_level : Typing_env.One_level.t,
        one_level ∈ levels
        ∧ cut_after < Typing_env.One_level.scope one_level
        ∧ (n, ty)
          ∈ Typing_env_level.equations
              (Typing_env.One_level.level one_level).
Proof.
  induction levels as [ | one_level levels IH]; intros result level n ty.
  - intros [= <-] Hin. left. exact Hin.
  - simpl.
    destruct (cut_after <? Typing_env.One_level.scope one_level) eqn:Hscope.
    + destruct (Typing_env_level.concat
                  (Typing_env.One_level.level one_level) result)
        as [result' | ] eqn:Hconcat; [ | discriminate ].
      simpl. intros Hloop Hin.
      destruct (IH result' level n ty Hloop Hin)
        as [Hin' | [one [Hone [Hafter Hin']]]].
      * apply concat_some in Hconcat. subst result'. simpl in Hin'.
        apply in_app_or in Hin'.
        destruct Hin' as [Hin' | Hin']; [ left; exact Hin' | right ].
        exists one_level. split; [ left; reflexivity | ].
        split; [ apply Z.ltb_lt; exact Hscope | exact Hin' ].
      * right. exists one. split; [ right; exact Hone | ].
        split; [ exact Hafter | exact Hin' ].
    + intros [= <-] Hin. left. exact Hin.
Qed.

Theorem cut_equations_member : cut_equations_member_statement.
Proof.
  intros tenv cut_after level n ty. unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after) eqn:Hscope.
  - intros [= <-] [].
  - intros Hloop Hin. apply Z.leb_gt in Hscope.
    destruct (cut_loop_equations_member _ _ _ _ _ _ Hloop Hin)
      as [Hin' | [one [Hone [Hafter Hin']]]].
    + exists (Typing_env.current_level tenv).
      split; [ left; reflexivity | ]. split; [ exact Hscope | exact Hin' ].
    + exists one. split; [ right; exact Hone | ].
      split; [ exact Hafter | exact Hin' ].
Qed.

Theorem cut_defined_variables_found : cut_defined_variables_found_statement.
Proof.
  intros tenv cut_after level x k Hwf.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [[_ [Hdefined _]] _]].
  unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after).
  - intros [= <-]. discriminate.
  - intros Hloop Hlookup.
    destruct (cut_loop_parts _ _ _ _ Hloop)
      as [taken [rest [Hlevels [_ Hvars]]]].
    rewrite Hvars, lookup_app in Hlookup.
    destruct (lookup x
                (Typing_env_level.defined_vars
                   (Typing_env.One_level.level
                      (Typing_env.current_level tenv))))
      as [k' | ] eqn:Hcurrent.
    + injection Hlookup as ->.
      apply (Hdefined _ x k (or_introl eq_refl) Hcurrent).
    + destruct (lookup_flat_map _ x k taken Hlookup) as [l [Hin Hl]].
      apply (Hdefined l x k); [ | exact Hl ].
      right. rewrite Hlevels. apply in_or_app. left. exact Hin.
Qed.

(* The levels that [cut_loop] goes through are the first of those that
   it is given, each of a scope after the one given; the next, if there
   is one, is not. *)
Lemma cut_loop_taken (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ result level : Typing_env_level.t,
    Typing_env.cut_loop cut_after result levels = Some level ->
    ∃ taken rest : list Typing_env.One_level.t,
      levels = taken ++ rest
      ∧ (∀ one_level : Typing_env.One_level.t,
          one_level ∈ taken ->
          cut_after < Typing_env.One_level.scope one_level)
      ∧ match rest with
        | [] => True
        | one_level :: _ => Typing_env.One_level.scope one_level <= cut_after
        end
      ∧ Typing_env_level.defined_vars level
        = Typing_env_level.defined_vars result
          ++ flat_map Typing_env_level.defined_vars
               (map Typing_env.One_level.level taken).
Proof.
  induction levels as [ | one_level levels IH]; intros result level.
  - intros [= <-]. exists [], []. simpl. rewrite app_nil_r.
    split; [ reflexivity | ]. split; [ intros l [] | ].
    split; [ exact I | reflexivity ].
  - simpl.
    destruct (cut_after <? Typing_env.One_level.scope one_level) eqn:Hscope.
    + destruct (Typing_env_level.concat
                  (Typing_env.One_level.level one_level) result)
        as [result' | ] eqn:Hconcat; [ | discriminate ].
      simpl. intros Hloop.
      destruct (IH result' level Hloop)
        as [taken [rest [Hlevels [Hafter [Hrest Hdefined]]]]].
      apply concat_some in Hconcat. subst result'. simpl in Hdefined.
      exists (one_level :: taken), rest.
      split; [ rewrite Hlevels; reflexivity | ].
      split;
        [ intros l [<- | Hin];
          [ apply Z.ltb_lt; exact Hscope | exact (Hafter l Hin) ]
        | ].
      split; [ exact Hrest | ].
      simpl. rewrite Hdefined, <- app_assoc. reflexivity.
    + intros [= <-]. exists [], (one_level :: levels). simpl.
      rewrite app_nil_r.
      split; [ reflexivity | ]. split; [ intros l [] | ].
      split; [ apply Z.ltb_ge; exact Hscope | reflexivity ].
Qed.

(* The first of scopes that decrease is greater than the others. *)
Lemma scopes_decreasing_head (scopes : list Scope.t) :
  ∀ later : Scope.t,
    scopes_decreasing (later :: scopes) ->
    ∀ earlier : Scope.t, earlier ∈ scopes -> earlier < later.
Proof.
  induction scopes as [ | s scopes IH];
    intros later [Hhead Hrest] earlier Hin; [ destruct Hin | ].
  destruct Hin as [<- | Hin]; [ exact Hhead | ].
  pose proof (IH s Hrest earlier Hin). lia.
Qed.

Lemma scopes_decreasing_suffix (scopes scopes' : list Scope.t) :
  scopes_decreasing (scopes ++ scopes') -> scopes_decreasing scopes'.
Proof.
  induction scopes as [ | s scopes IH]; [ exact (fun H => H) | ].
  intros [_ Hrest]. exact (IH Hrest).
Qed.

Lemma levels_pairwise_disjoint_prefix
    (levels levels' : list Typing_env_level.t) :
  levels_pairwise_disjoint (levels ++ levels') ->
  levels_pairwise_disjoint levels.
Proof.
  induction levels as [ | level levels IH]; [ intros _; exact I | ].
  intros [Hhead Hrest]. split; [ | exact (IH Hrest) ].
  intros earlier Hin. apply Hhead. apply in_or_app. left. exact Hin.
Qed.

(* What one of several disjoint levels defines is found in what they
   define together. *)
Lemma lookup_flat_map_disjoint (levels : list Typing_env_level.t)
    (level : Typing_env_level.t) (x : variable) (k : flambda_kind) :
  levels_pairwise_disjoint levels ->
  level ∈ levels ->
  lookup x (Typing_env_level.defined_vars level) = Some k ->
  lookup x (flat_map Typing_env_level.defined_vars levels) = Some k.
Proof.
  induction levels as [ | later levels IH]; intros Hdisjoint Hin Hlookup;
    [ destruct Hin | ].
  destruct Hdisjoint as [Hhead Hrest]. simpl. rewrite lookup_app.
  destruct Hin as [-> | Hin]; [ rewrite Hlookup; reflexivity | ].
  destruct (Hhead level Hin) as [Hvars _].
  assert (Hbound : is_bound x (Typing_env_level.defined_vars level) = true).
  { apply is_bound_lookup. exists k. exact Hlookup. }
  pose proof (Hvars x Hbound) as Hnot. unfold is_bound in Hnot.
  destruct (lookup x (Typing_env_level.defined_vars later));
    [ discriminate | ].
  exact (IH Hrest Hin Hlookup).
Qed.

Theorem cut_defined_variables : cut_defined_variables_statement.
Proof.
  intros tenv cut_after level x k Hwf.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [[Hdisjoint [_ [_ [_ [_ [Hscopes _]]]]]] _]].
  unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after) eqn:Hscope.
  - (* No level is after the scope. *)
    intros [= <-]. split; [ discriminate | ].
    intros [one_level [Hin [Hafter _]]]. exfalso.
    apply Z.leb_le in Hscope. unfold Typing_env.current_scope in Hscope.
    destruct Hin as [<- | Hin]; [ lia | ].
    pose proof (scopes_decreasing_head _ _ Hscopes
                  (Typing_env.One_level.scope one_level)
                  (in_map Typing_env.One_level.scope _ _ Hin)).
    lia.
  - intros Hloop. apply Z.leb_gt in Hscope.
    unfold Typing_env.current_scope in Hscope.
    destruct (cut_loop_taken _ _ _ _ Hloop)
      as [taken [rest [Hlevels [Hafter [Hrest Hvars]]]]].
    (* The current level and the levels gone through are disjoint. *)
    assert (Hprefix :
      levels_pairwise_disjoint
        (Typing_env.One_level.level (Typing_env.current_level tenv)
         :: map Typing_env.One_level.level taken)).
    { apply (levels_pairwise_disjoint_prefix
               (_ :: map Typing_env.One_level.level taken)
               (map Typing_env.One_level.level rest)).
      unfold all_levels in Hdisjoint. simpl in Hdisjoint.
      rewrite Hlevels, map_app in Hdisjoint. exact Hdisjoint. }
    change (Typing_env_level.defined_vars level
            = flat_map Typing_env_level.defined_vars
                (Typing_env.One_level.level (Typing_env.current_level tenv)
                 :: map Typing_env.One_level.level taken)) in Hvars.
    rewrite Hvars. split.
    + intros Hlookup.
      destruct (lookup_flat_map _ x k _ Hlookup) as [l [Hin Hl]].
      destruct Hin as [<- | Hin].
      * exists (Typing_env.current_level tenv).
        split; [ left; reflexivity | ]. split; [ exact Hscope | exact Hl ].
      * apply in_map_iff in Hin. destruct Hin as [one_level [<- Hin]].
        exists one_level.
        split; [ right; rewrite Hlevels; apply in_or_app; left; exact Hin | ].
        split; [ exact (Hafter one_level Hin) | exact Hl ].
    + intros [one_level [Hin [Hafter' Hl]]].
      apply (lookup_flat_map_disjoint _
               (Typing_env.One_level.level one_level) x k Hprefix);
        [ | exact Hl ].
      destruct Hin as [<- | Hin]; [ left; reflexivity | right ].
      rewrite Hlevels in Hin. apply in_app_or in Hin.
      destruct Hin as [Hin | Hin]; [ apply in_map; exact Hin | exfalso ].
      (* A level that was not gone through is not after the scope. *)
      destruct rest as [ | first rest]; [ destruct Hin | ].
      assert (Hdecreasing :
        scopes_decreasing
          (map Typing_env.One_level.scope (first :: rest))).
      { apply (scopes_decreasing_suffix
                 (Typing_env.One_level.scope (Typing_env.current_level tenv)
                  :: map Typing_env.One_level.scope taken)).
        unfold all_scopes in Hscopes. simpl in Hscopes.
        rewrite Hlevels, map_app in Hscopes. exact Hscopes. }
      destruct Hin as [<- | Hin]; [ lia | ].
      pose proof (scopes_decreasing_head _ _ Hdecreasing
                    (Typing_env.One_level.scope one_level)
                    (in_map Typing_env.One_level.scope _ _ Hin)).
      lia.
Qed.

(* What the concatenation of two disjoint levels has at a binding
   time. *)
Lemma concat_indexed_at (earlier later result : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) :
  Typing_env_level.concat earlier later = Some result ->
  indexed_at result binding_time x
  <-> indexed_at later binding_time x ∨ indexed_at earlier binding_time x.
Proof.
  intros Hconcat.
  assert (Hdisjoint : levels_disjoint earlier later).
  { apply concat_defined. rewrite Hconcat. discriminate. }
  destruct Hdisjoint as [_ Htimes].
  apply concat_some in Hconcat. subst result.
  unfold indexed_at. simpl. rewrite keyed_lookup_app.
  destruct (Keyed.lookup Z.eqb binding_time
              (Typing_env_level.binding_times later))
    as [vars | ] eqn:Hlater.
  - split; [ intros Hat; left; exact Hat | ].
    intros [Hat | [vars' [Hlookup _]]]; [ exact Hat | exfalso ].
    assert (Hbound :
      Keyed.is_bound Z.eqb binding_time
        (Typing_env_level.binding_times earlier)
      = true).
    { apply keyed_is_bound_lookup. exists vars'. exact Hlookup. }
    pose proof (Htimes binding_time Hbound) as Hfalse.
    unfold Keyed.is_bound in Hfalse. rewrite Hlater in Hfalse.
    discriminate.
  - split; [ intros Hat; right; exact Hat | ].
    intros [[vars [Hnone _]] | Hat]; [ discriminate | exact Hat ].
Qed.

Theorem concat_indexed : concat_indexed_statement.
Proof.
  intros earlier later result Hconcat Hearlier Hlater.
  pose proof (concat_indexed_at earlier later result) as Hat.
  split.
  - intros x.
    assert (Hbound :
      is_bound x (Typing_env_level.defined_vars result)
      = is_bound x (Typing_env_level.defined_vars later)
        || is_bound x (Typing_env_level.defined_vars earlier)).
    { rewrite (concat_some _ _ _ Hconcat). apply is_bound_app. }
    rewrite Hbound, orb_true_iff, (Hlater x), (Hearlier x). split.
    + intros [[binding_time Hx] | [binding_time Hx]]; exists binding_time;
        apply (Hat binding_time x Hconcat);
        [ left; exact Hx | right; exact Hx ].
    + intros [binding_time Hx]. apply (Hat binding_time x Hconcat) in Hx.
      destruct Hx as [Hx | Hx]; [ left | right ]; exists binding_time;
        exact Hx.
  - intros binding_time x Hx. apply (Hat binding_time x Hconcat) in Hx.
    destruct Hx as [Hx | Hx]; [ right | left ]; exact Hx.
Qed.

(* A level has its index, and each of its variables is at the binding
   time that the environment has for it. *)
Definition level_recorded_in (tenv : Typing_env.t)
    (level : Typing_env_level.t) : Prop :=
  level_indexed level
  ∧ ∀ (binding_time : Binding_time.t) (x : variable),
      indexed_at level binding_time x ->
      ∃ (ty : flambda_type)
        (binding_time_and_mode : Binding_time.With_name_mode.t),
        Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
        = Some (ty, binding_time_and_mode)
        ∧ Binding_time.With_name_mode.binding_time binding_time_and_mode
          = binding_time.

Lemma cut_loop_indexed (tenv : Typing_env.t) (cut_after : Scope.t)
    (levels : list Typing_env.One_level.t) :
  ∀ result level : Typing_env_level.t,
    (∀ earlier : Typing_env_level.t,
      earlier ∈ map Typing_env.One_level.level levels ->
      level_recorded_in tenv earlier) ->
    level_recorded_in tenv result ->
    Typing_env.cut_loop cut_after result levels = Some level ->
    level_recorded_in tenv level.
Proof.
  induction levels as [ | one_level levels IH];
    intros result level Hlevels Hresult.
  - intros [= <-]. exact Hresult.
  - simpl. destruct (cut_after <? Typing_env.One_level.scope one_level).
    + destruct (Typing_env_level.concat
                  (Typing_env.One_level.level one_level) result)
        as [result' | ] eqn:Hconcat; [ | discriminate ].
      simpl. apply IH.
      * intros earlier Hin. apply Hlevels. right. exact Hin.
      * destruct (Hlevels _ (or_introl eq_refl)) as [Hone Hone_times].
        destruct Hresult as [Hresult Hresult_times].
        destruct (concat_indexed _ _ _ Hconcat Hone Hresult)
          as [Hindexed Hat].
        split; [ exact Hindexed | ].
        intros binding_time x Hx.
        destruct (Hat binding_time x Hx) as [Hx' | Hx'].
        -- exact (Hone_times binding_time x Hx').
        -- exact (Hresult_times binding_time x Hx').
    + intros [= <-]. exact Hresult.
Qed.

Theorem cut_indexed : cut_indexed_statement.
Proof.
  intros tenv cut_after level Hwf.
  apply typing_env_well_formed_iff in Hwf.
  destruct Hwf as [_ [[_ [_ [Hindexed [Hrecorded _]]]] _]].
  unfold Typing_env.cut.
  destruct (Typing_env.current_scope tenv <=? cut_after).
  - intros [= <-]. split.
    + intros x. split; [ discriminate | ].
      intros [binding_time [vars [Hnone _]]]. discriminate.
    + intros binding_time x [vars [Hnone _]]. discriminate.
  - apply cut_loop_indexed.
    + intros earlier Hin. split.
      * apply Hindexed. right. exact Hin.
      * intros binding_time x Hx.
        exact (Hrecorded earlier binding_time x (or_intror Hin) Hx).
    + split.
      * apply Hindexed. left. reflexivity.
      * intros binding_time x Hx.
        exact (Hrecorded _ binding_time x (or_introl eq_refl) Hx).
Qed.

(* ------------------------------------------------------------------------ *)
(* Going through the variables of a level                                   *)
(* ------------------------------------------------------------------------ *)

Lemma insert_binding_time_in (binding_time : Binding_time.t)
    (sorted : list Binding_time.t) (t : Binding_time.t) :
  t ∈ Typing_env_level.insert_binding_time binding_time sorted
  <-> t = binding_time ∨ t ∈ sorted.
Proof.
  induction sorted as [ | first rest IH]; simpl.
  - intuition congruence.
  - destruct (binding_time <=? first); simpl;
      [ intuition congruence | rewrite IH; intuition congruence ].
Qed.

Lemma sort_binding_times_cons (first : Binding_time.t)
    (rest : list Binding_time.t) :
  Typing_env_level.sort_binding_times (first :: rest)
  = Typing_env_level.insert_binding_time first
      (Typing_env_level.sort_binding_times rest).
Proof. reflexivity. Qed.

Lemma sort_binding_times_in (times : list Binding_time.t)
    (t : Binding_time.t) :
  t ∈ Typing_env_level.sort_binding_times times <-> t ∈ times.
Proof.
  induction times as [ | first rest IH]; [ reflexivity | ].
  rewrite sort_binding_times_cons, insert_binding_time_in, IH. simpl.
  intuition congruence.
Qed.

Lemma insert_binding_time_sorted (binding_time : Binding_time.t)
    (sorted : list Binding_time.t) :
  StronglySorted Z.le sorted ->
  StronglySorted Z.le
    (Typing_env_level.insert_binding_time binding_time sorted).
Proof.
  induction sorted as [ | first rest IH]; intros Hsorted; simpl.
  - constructor; constructor.
  - apply StronglySorted_inv in Hsorted. destruct Hsorted as [Hrest Hfirst].
    destruct (Z.leb_spec binding_time first) as [Hle | Hgt].
    + constructor; [ constructor; assumption | ].
      constructor; [ exact Hle | ].
      apply Forall_forall. intros t Hin. rewrite Forall_forall in Hfirst.
      pose proof (Hfirst t Hin). lia.
    + constructor; [ exact (IH Hrest) | ].
      apply Forall_forall. intros t Hin.
      apply insert_binding_time_in in Hin.
      destruct Hin as [-> | Hin]; [ lia | ].
      rewrite Forall_forall in Hfirst. exact (Hfirst t Hin).
Qed.

Lemma sort_binding_times_sorted (times : list Binding_time.t) :
  StronglySorted Z.le (Typing_env_level.sort_binding_times times).
Proof.
  induction times as [ | first rest IH]; [ constructor | ].
  rewrite sort_binding_times_cons. apply insert_binding_time_sorted.
  exact IH.
Qed.

Lemma strongly_sorted_order (l1 l2 l3 : list Z) (a b : Z) :
  StronglySorted Z.le (l1 ++ a :: l2 ++ b :: l3) -> a <= b.
Proof.
  induction l1 as [ | c l1 IH]; simpl; intros Hsorted;
    apply StronglySorted_inv in Hsorted.
  - destruct Hsorted as [_ Hall]. rewrite Forall_forall in Hall.
    apply Hall. apply in_or_app. right. left. reflexivity.
  - exact (IH (proj1 Hsorted)).
Qed.

(* Of two elements of a list made of pieces, the first is in a piece
   that is the piece of the second or comes before it. *)
Lemma flat_map_order {A B : Type} (g : A -> list B) (l : list A) :
  ∀ (before between after : list B) (x y : B),
    flat_map g l = before ++ x :: between ++ y :: after ->
    ∃ a b : A,
      x ∈ g a
      ∧ y ∈ g b
      ∧ (a = b ∨ ∃ l1 l2 l3 : list A, l = l1 ++ a :: l2 ++ b :: l3).
Proof.
  induction l as [ | c l IH]; intros before between after x y Heq.
  - simpl in Heq. destruct before; discriminate.
  - simpl in Heq. apply app_eq_app in Heq.
    destruct Heq as [rest [[Hc Hrest] | [Hbefore Hrest]]].
    + destruct rest as [ | x' rest].
      * simpl in Hrest. symmetry in Hrest.
        destruct (IH [] between after x y Hrest)
          as [a [b [Hx [Hy Horder]]]].
        exists a, b. split; [ exact Hx | ]. split; [ exact Hy | ].
        destruct Horder as [-> | [l1 [l2 [l3 ->]]]];
          [ left; reflexivity | right ].
        exists (c :: l1), l2, l3. reflexivity.
      * simpl in Hrest. injection Hrest as <- Hrest.
        assert (Hx : x ∈ g c).
        { rewrite Hc. apply in_or_app. right. left. reflexivity. }
        assert (Hy : y ∈ rest ∨ y ∈ flat_map g l).
        { apply in_app_or. rewrite <- Hrest. apply in_or_app. right. left.
          reflexivity. }
        destruct Hy as [Hy | Hy].
        -- exists c, c. split; [ exact Hx | ].
           split; [ | left; reflexivity ].
           rewrite Hc. apply in_or_app. right. right. exact Hy.
        -- apply in_flat_map in Hy. destruct Hy as [b [Hb Hy]].
           apply in_split in Hb. destruct Hb as [l2 [l3 ->]].
           exists c, b. split; [ exact Hx | ]. split; [ exact Hy | ].
           right. exists [], l2, l3. reflexivity.
    + destruct (IH rest between after x y Hrest)
        as [a [b [Hx [Hy Horder]]]].
      exists a, b. split; [ exact Hx | ]. split; [ exact Hy | ].
      destruct Horder as [-> | [l1 [l2 [l3 ->]]]];
        [ left; reflexivity | right ].
      exists (c :: l1), l2, l3. reflexivity.
Qed.

(* The variables that a level has for a binding time are at it. *)
Lemma bucket_indexed_at (level : Typing_env_level.t)
    (binding_time : Binding_time.t) (x : variable) :
  x ∈ match
        Keyed.lookup Z.eqb binding_time
          (Typing_env_level.binding_times level)
      with
      | Some vars => dedup vars
      | None => []
      end ->
  indexed_at level binding_time x.
Proof.
  unfold indexed_at.
  destruct (Keyed.lookup Z.eqb binding_time
              (Typing_env_level.binding_times level)) as [vars | ];
    [ | intros [] ].
  intros Hin. exists vars. split; [ reflexivity | ].
  apply among_In. rewrite <- among_dedup. apply among_In. exact Hin.
Qed.

Theorem variables_in_order_indexed : variables_in_order_indexed_statement.
Proof.
  intros level x. unfold Typing_env_level.variables_in_order.
  rewrite in_flat_map. split.
  - intros [binding_time [_ Hin]]. exists binding_time.
    exact (bucket_indexed_at level binding_time x Hin).
  - intros [binding_time [vars [Hlookup Hin]]]. exists binding_time. split.
    + apply sort_binding_times_in.
      apply (keyed_among_in Z Z.eqb z_eqb_decides_equality).
      rewrite (keyed_among_domain Z Z.eqb z_eqb_decides_equality).
      apply keyed_is_bound_lookup. exists vars. exact Hlookup.
    + rewrite Hlookup. apply among_In. rewrite among_dedup.
      apply among_In. exact Hin.
Qed.

Lemma nodup_app_apart {B : Type} (l l' : list B) :
  NoDup l -> NoDup l' -> (∀ b : B, b ∈ l -> ~ b ∈ l') -> NoDup (l ++ l').
Proof.
  intros Hl Hl' Hapart. induction Hl as [ | b l Hb Hl IH]; [ exact Hl' | ].
  simpl. apply NoDup_cons.
  - intros Hin. apply in_app_or in Hin. destruct Hin as [Hin | Hin].
    + exact (Hb Hin).
    + exact (Hapart b (or_introl eq_refl) Hin).
  - apply IH. intros b' Hin. apply Hapart. right. exact Hin.
Qed.

Lemma nodup_flat_map {A B : Type} (g : A -> list B) (l : list A) :
  NoDup l ->
  (∀ a : A, a ∈ l -> NoDup (g a)) ->
  (∀ (a a' : A) (b : B), a ∈ l -> a' ∈ l -> b ∈ g a -> b ∈ g a' -> a = a') ->
  NoDup (flat_map g l).
Proof.
  intros Hl. induction Hl as [ | a l Ha Hl IH]; intros Hpieces Hapart;
    [ apply NoDup_nil | ].
  simpl. apply nodup_app_apart.
  - apply Hpieces. left. reflexivity.
  - apply IH.
    + intros a' Hin. apply Hpieces. right. exact Hin.
    + intros a' a'' b Hin' Hin''.
      exact (Hapart a' a'' b (or_intror Hin') (or_intror Hin'')).
  - intros b Hb Hin. apply in_flat_map in Hin.
    destruct Hin as [a' [Hin' Hb']].
    assert (Heq : a = a').
    { exact (Hapart a a' b (or_introl eq_refl) (or_intror Hin') Hb Hb'). }
    subst a'. exact (Ha Hin').
Qed.

Lemma insert_binding_time_nodup (binding_time : Binding_time.t)
    (sorted : list Binding_time.t) :
  ~ binding_time ∈ sorted ->
  NoDup sorted ->
  NoDup (Typing_env_level.insert_binding_time binding_time sorted).
Proof.
  induction sorted as [ | first rest IH]; intros Hnot Hsorted; simpl.
  - apply NoDup_cons; [ intros [] | apply NoDup_nil ].
  - destruct (binding_time <=? first).
    + apply NoDup_cons; [ exact Hnot | exact Hsorted ].
    + apply NoDup_cons_iff in Hsorted. destruct Hsorted as [Hfirst Hrest].
      apply NoDup_cons.
      * intros Hin. apply insert_binding_time_in in Hin.
        destruct Hin as [-> | Hin]; [ | exact (Hfirst Hin) ].
        apply Hnot. left. reflexivity.
      * apply IH; [ | exact Hrest ]. intros Hin. apply Hnot. right.
        exact Hin.
Qed.

Lemma sort_binding_times_nodup (times : list Binding_time.t) :
  NoDup times -> NoDup (Typing_env_level.sort_binding_times times).
Proof.
  induction times as [ | first rest IH]; intros Htimes;
    [ apply NoDup_nil | ].
  apply NoDup_cons_iff in Htimes. destruct Htimes as [Hfirst Hrest].
  rewrite sort_binding_times_cons. apply insert_binding_time_nodup.
  - intros Hin. apply (proj1 (sort_binding_times_in rest first)) in Hin.
    exact (Hfirst Hin).
  - exact (IH Hrest).
Qed.

Theorem variables_in_order_distinct : variables_in_order_distinct_statement.
Proof.
  intros level Hone. unfold Typing_env_level.variables_in_order.
  apply nodup_flat_map.
  - apply sort_binding_times_nodup.
    apply (keyed_distinct_NoDup Z.eqb z_eqb_decides_equality).
    apply (keyed_domain_distinct Z Z.eqb z_eqb_decides_equality).
  - intros binding_time _.
    destruct (Keyed.lookup Z.eqb binding_time
                (Typing_env_level.binding_times level)) as [vars | ];
      [ | apply NoDup_nil ].
    apply distinct_NoDup. apply dedup_distinct.
  - intros binding_time binding_time' x _ _ Hx Hx'.
    exact (Hone x binding_time binding_time'
             (bucket_indexed_at level binding_time x Hx)
             (bucket_indexed_at level binding_time' x Hx')).
Qed.

Theorem cut_variables_in_order_distinct
    : cut_variables_in_order_distinct_statement.
Proof.
  intros tenv cut_after level Hwf Hcut.
  destruct (cut_indexed tenv cut_after level Hwf Hcut) as [_ Htimes].
  apply variables_in_order_distinct.
  intros x binding_time binding_time' Hx Hx'.
  destruct (Htimes binding_time x Hx) as [ty [btm [Hlookup <-]]].
  destruct (Htimes binding_time' x Hx') as [ty' [btm' [Hlookup' <-]]].
  rewrite Hlookup in Hlookup'. injection Hlookup' as _ <-. reflexivity.
Qed.

Theorem variables_in_order_sorted : variables_in_order_sorted_statement.
Proof.
  intros level before between after x y Heq.
  unfold Typing_env_level.variables_in_order in Heq.
  destruct (flat_map_order _ _ _ _ _ _ _ Heq) as [a [b [Hx [Hy Horder]]]].
  exists a, b. split; [ exact (bucket_indexed_at level a x Hx) | ].
  split; [ exact (bucket_indexed_at level b y Hy) | ].
  destruct Horder as [-> | [l1 [l2 [l3 Hsplit]]]]; [ apply Z.le_refl | ].
  apply (strongly_sorted_order l1 l2 l3 a b). rewrite <- Hsplit.
  apply sort_binding_times_sorted.
Qed.

Theorem variables_in_order_defined : variables_in_order_defined_statement.
Proof.
  intros level x Hlevel. rewrite (variables_in_order_indexed level x).
  symmetry. exact (Hlevel x).
Qed.

Theorem fold_on_defined_vars_defined : fold_on_defined_vars_defined_statement.
Proof.
  intros A f level init Hlevel.
  unfold Typing_env_level.fold_on_defined_vars, Typing_env_level.find_kind.
  assert (Hall :
    ∀ x : variable,
      x ∈ Typing_env_level.variables_in_order level ->
      ∃ kind : flambda_kind,
        lookup x (Typing_env_level.defined_vars level) = Some kind).
  { intros x Hin. apply (variables_in_order_defined level x Hlevel) in Hin.
    apply is_bound_lookup in Hin. exact Hin. }
  revert init Hall. generalize (Typing_env_level.variables_in_order level).
  intros vars. induction vars as [ | x vars IH]; intros init Hall;
    [ reflexivity | ].
  destruct (Hall x (or_introl eq_refl)) as [kind Hkind].
  cbn [List.fold_left]. rewrite Hkind. cbn. apply IH.
  intros y Hin. apply Hall. right. exact Hin.
Qed.

(* ======================================================================== *)
(* The functions keep the invariant                                         *)
(* ======================================================================== *)

Lemma conditions_intro (tenv : Typing_env.t) :
  binding_times_bounded tenv ->
  binding_times_in_range tenv ->
  binding_times_distinct tenv ->
  types_closed tenv ->
  symbols_defined tenv ->
  symbols_recorded tenv ->
  levels_pairwise_disjoint (all_levels tenv) ->
  defined_variables_bound tenv ->
  levels_indexed tenv ->
  level_binding_times_recorded tenv ->
  level_binding_times_in_range tenv ->
  scopes_decreasing (all_scopes tenv) ->
  variables_defined tenv ->
  aliases_consistent (Typing_env.aliases tenv) ->
  name_modes_recorded (Typing_env.names_to_types tenv)
    (Typing_env.aliases tenv) ->
  aliases_known tenv ->
  canonical_names_concrete tenv ->
  typing_env_conditions tenv.
Proof. unfold typing_env_conditions. tauto. Qed.

Lemma invariant_intro (tenv : Typing_env.t) :
  typing_env_conditions tenv -> levels_agree tenv -> typing_env_invariant tenv.
Proof.
  intros Hconditions Hagree. split; [ | exact Hagree ].
  apply typing_env_well_formed_iff. exact Hconditions.
Qed.

Lemma invariant_elim (tenv : Typing_env.t) :
  typing_env_invariant tenv -> typing_env_conditions tenv ∧ levels_agree tenv.
Proof.
  intros [Hwf Hagree].
  split; [ apply typing_env_well_formed_iff; exact Hwf | exact Hagree ].
Qed.

(* The parts of the invariant, each under the name used below. *)
Ltac invariant_parts Hinvariant :=
  apply invariant_elim in Hinvariant;
  destruct Hinvariant
    as [[[Hbounded [Hrange [Hdistinct [Hclosed [Hsymbols Hsym_recorded]]]]]
         [[Hdisjoint
            [Hdefined
               [Hindexed [Hrecorded [Hlevel_range [Hscopes Hvar_defined]]]]]]
          [Hconsistent [Hmodes [Hknown Hconcrete]]]]]
        Hagree].

(* Every name that the aliases speak of is bound. *)
Lemma mentioned_known (tenv : Typing_env.t) (n : name) :
  aliases_consistent (Typing_env.aliases tenv) ->
  aliases_known tenv ->
  aliases_mention (Typing_env.aliases tenv) n ->
  Typing_env.mem None tenv n = true.
Proof.
  intros [_ Hlisted] Hknown Hmention.
  assert (Hcanonical :
    ∀ n' : name,
      demoted_to (Typing_env.aliases tenv) n' (simple_of_name n) ->
      Typing_env.mem None tenv n = true).
  { intros n' Hdemoted. destruct (Hknown n' _ Hdemoted) as [_ Hmem].
    destruct n; exact Hmem. }
  destruct Hmention
    as [[c Hdemoted] | [[n' Hdemoted] | [[c Hlisted'] | [n' Hlisted']]]].
  - exact (proj1 (Hknown n c Hdemoted)).
  - exact (Hcanonical n' Hdemoted).
  - exact (proj1 (Hknown n c (proj2 (Hlisted n c) Hlisted'))).
  - exact (Hcanonical n' (proj2 (Hlisted n' (simple_of_name n)) Hlisted')).
Qed.

Theorem create_invariant : create_invariant_statement.
Proof.
  split; [ reflexivity | ]. intros n ty Hlookup. discriminate.
Qed.

Theorem make_bottom_invariant : make_bottom_invariant_statement.
Proof. intros tenv Hinvariant. exact Hinvariant. Qed.

Theorem increment_scope_invariant : increment_scope_invariant_statement.
Proof.
  intros tenv Hinvariant. invariant_parts Hinvariant.
  apply invariant_intro; [ | exact Hagree ].
  apply conditions_intro;
    try first
      [ exact Hbounded | exact Hrange | exact Hdistinct | exact Hclosed
      | exact Hsymbols | exact Hsym_recorded | exact Hconsistent
      | exact Hmodes | exact Hknown | exact Hconcrete ].
  - (* The new level is empty. *)
    split; [ | exact Hdisjoint ].
    intros earlier _. split; intros key _; reflexivity.
  - intros level x k [<- | Hin]; [ discriminate | ].
    exact (Hdefined level x k Hin).
  - intros level [<- | Hin]; [ | exact (Hindexed level Hin) ].
    intros x. split; [ discriminate | ].
    intros [binding_time [vars [Hnone _]]]. discriminate.
  - intros level binding_time x [<- | Hin].
    + intros [vars [Hnone _]]. discriminate.
    + exact (Hrecorded level binding_time x Hin).
  - intros level binding_time [<- | Hin]; [ discriminate | ].
    exact (Hlevel_range level binding_time Hin).
  - split; [ | exact Hscopes ]. simpl.
    unfold Scope.next, Typing_env.current_scope. lia.
  - intros x ty btm Hlookup.
    destruct (Hvar_defined x ty btm Hlookup) as [level [Hin Hbound]].
    exists level. split; [ right; exact Hin | exact Hbound ].
Qed.

Theorem closure_env_invariant : closure_env_invariant_statement.
Proof.
  intros tenv Hinvariant. unfold Typing_env.closure_env.
  apply increment_scope_invariant.
  invariant_parts Hinvariant.
  apply invariant_intro; [ | exact Hagree ].
  apply conditions_intro;
    try first
      [ exact Hrange | exact Hdistinct | exact Hclosed | exact Hsymbols
      | exact Hsym_recorded | exact Hdisjoint | exact Hdefined
      | exact Hindexed
      | exact Hrecorded | exact Hlevel_range | exact Hscopes
      | exact Hvar_defined | exact Hconsistent | exact Hmodes
      | exact Hknown | exact Hconcrete ].
  destruct Hbounded as [Hearliest Hmin]. split; simpl; lia.
Qed.

(* A name that is bound stays bound when a symbol is defined. *)
Lemma mem_add_symbol_definition (tenv : Typing_env.t) (sym : symbol)
    (n : name) :
  Typing_env.mem None tenv n = true ->
  Typing_env.mem None (Typing_env.add_symbol_definition tenv sym) n = true.
Proof.
  destruct n as [x | s]; [ exact (fun Hmem => Hmem) | ].
  intros Hmem.
  change (among s (sym :: Typing_env.defined_symbols tenv) = true).
  rewrite among_cons.
  change (among s (Typing_env.defined_symbols tenv) = true) in Hmem.
  rewrite Hmem. apply orb_true_r.
Qed.

Theorem add_symbol_definition_invariant
    : add_symbol_definition_invariant_statement.
Proof.
  intros tenv sym Hinvariant. invariant_parts Hinvariant.
  apply invariant_intro; [ | exact Hagree ].
  apply conditions_intro;
    try first
      [ exact Hbounded | exact Hrange | exact Hdistinct | exact Hclosed
      | exact Hsym_recorded | exact Hdisjoint | exact Hdefined
      | exact Hindexed
      | exact Hrecorded | exact Hlevel_range | exact Hscopes
      | exact Hvar_defined | exact Hconsistent | exact Hmodes
      | exact Hconcrete ].
  - intros s ty btm Hlookup.
    exact (mem_add_symbol_definition tenv sym (Name_symbol s)
             (Hsymbols s ty btm Hlookup)).
  - intros n c Hdemoted. destruct (Hknown n c Hdemoted) as [Hn Hc].
    split; [ exact (mem_add_symbol_definition tenv sym n Hn) | ].
    destruct c as [x | s | k]; [ | | reflexivity ].
    + exact (mem_add_symbol_definition tenv sym (Name_var x) Hc).
    + exact (mem_add_symbol_definition tenv sym (Name_symbol s) Hc).
Qed.

(* ------------------------------------------------------------------------ *)
(* Defining a variable                                                      *)
(* ------------------------------------------------------------------------ *)

Lemma free_variables_unknown (kind : flambda_kind) :
  free_variables (unknown kind) = [].
Proof. destruct kind as [ | [] | | ]; reflexivity. Qed.

Lemma get_alias_opt_unknown (kind : flambda_kind) :
  get_alias_opt (unknown kind) = None.
Proof. destruct kind as [ | [] | | ]; reflexivity. Qed.

Lemma add_variable_definition_lookup (tenv : Typing_env.t) (var : variable)
    (kind : flambda_kind) (name_mode : Name_mode.t) (n : name) :
  Keyed.lookup name_eqb n
    (Typing_env.names_to_types
       (Typing_env.add_variable_definition tenv var kind name_mode))
  = if name_eqb n (Name_var var)
    then
      Some
        (unknown kind,
         Binding_time.With_name_mode.create
           (Typing_env.next_binding_time tenv) name_mode)
    else Keyed.lookup name_eqb n (Typing_env.names_to_types tenv).
Proof. reflexivity. Qed.

Lemma mem_fresh_lookup (tenv : Typing_env.t) (var : variable) :
  Typing_env.mem None tenv (Name_var var) = false ->
  Keyed.lookup name_eqb (Name_var var) (Typing_env.names_to_types tenv)
  = None.
Proof.
  rewrite mem_var_is_bound. unfold Keyed.is_bound.
  destruct (Keyed.lookup name_eqb (Name_var var)
              (Typing_env.names_to_types tenv));
    [ discriminate | reflexivity ].
Qed.

Lemma add_definition_defined_vars (level : Typing_env_level.t)
    (var : variable) (kind : flambda_kind) (binding_time : Binding_time.t) :
  Typing_env_level.defined_vars
    (Typing_env_level.add_definition level var kind binding_time)
  = (var, kind) :: Typing_env_level.defined_vars level.
Proof. reflexivity. Qed.

Lemma add_definition_is_bound_time (level : Typing_env_level.t)
    (var : variable) (kind : flambda_kind)
    (binding_time binding_time' : Binding_time.t) :
  Keyed.is_bound Z.eqb binding_time'
    (Typing_env_level.binding_times
       (Typing_env_level.add_definition level var kind binding_time))
  = (binding_time' =? binding_time)
    || Keyed.is_bound Z.eqb binding_time'
         (Typing_env_level.binding_times level).
Proof.
  unfold Keyed.is_bound. simpl.
  destruct (binding_time' =? binding_time); reflexivity.
Qed.

(* What a level has at a binding time once a variable is defined. *)
Lemma add_definition_indexed_at (level : Typing_env_level.t)
    (var : variable) (kind : flambda_kind)
    (binding_time binding_time' : Binding_time.t) (x : variable) :
  indexed_at (Typing_env_level.add_definition level var kind binding_time)
    binding_time' x
  <-> (binding_time' = binding_time ∧ x = var)
      ∨ indexed_at level binding_time' x.
Proof.
  unfold indexed_at, Typing_env_level.add_definition. simpl.
  destruct (Z.eqb_spec binding_time' binding_time) as [-> | Hne].
  - destruct (Keyed.lookup Z.eqb binding_time
                (Typing_env_level.binding_times level)) as [prev_vars | ].
    + split.
      * intros [vars [[= <-] [<- | Hin]]].
        -- left. split; reflexivity.
        -- right. exists prev_vars. split; [ reflexivity | exact Hin ].
      * intros [[_ ->] | [vars [[= <-] Hin]]].
        -- eexists. split; [ reflexivity | left; reflexivity ].
        -- eexists. split; [ reflexivity | right; exact Hin ].
    + split.
      * intros [vars [[= <-] [<- | []]]]. left. split; reflexivity.
      * intros [[_ ->] | [vars [Hnone _]]]; [ | discriminate ].
        eexists. split; [ reflexivity | left; reflexivity ].
  - split.
    + intros Hat. right. exact Hat.
    + intros [[Heq _] | Hat]; [ contradiction | exact Hat ].
Qed.

Lemma add_definition_indexed (level : Typing_env_level.t) (var : variable)
    (kind : flambda_kind) (binding_time : Binding_time.t) :
  level_indexed level ->
  level_indexed (Typing_env_level.add_definition level var kind binding_time).
Proof.
  intros Hlevel x. rewrite add_definition_defined_vars.
  unfold is_bound. rewrite lookup_cons.
  destruct (Pos.eqb_spec x var) as [-> | Hne].
  - split; [ | reflexivity ]. intros _. exists binding_time.
    apply add_definition_indexed_at. left. split; reflexivity.
  - split.
    + intros Hbound. destruct (proj1 (Hlevel x) Hbound) as [bt Hat].
      exists bt. apply add_definition_indexed_at. right. exact Hat.
    + intros [bt Hat]. apply add_definition_indexed_at in Hat.
      destruct Hat as [[_ Heq] | Hat]; [ contradiction | ].
      apply Hlevel. exists bt. exact Hat.
Qed.

Theorem add_variable_definition_invariant
    : add_variable_definition_invariant_statement.
Proof.
  intros tenv var kind name_mode Hinvariant Hfresh.
  apply mem_fresh_lookup in Hfresh.
  invariant_parts Hinvariant.
  destruct Hbounded as [Hearliest Hmin].
  set (tenv' := Typing_env.add_variable_definition tenv var kind name_mode).
  set (entry :=
         (unknown kind,
          Binding_time.With_name_mode.create
            (Typing_env.next_binding_time tenv) name_mode)).
  (* The new entry, and the entries of the other names. *)
  assert (Hnew :
    Keyed.lookup name_eqb (Name_var var) (Typing_env.names_to_types tenv')
    = Some entry).
  { unfold tenv'.
    rewrite add_variable_definition_lookup,
      (keyed_eqb_refl name_eqb name_eqb_eq).
    reflexivity. }
  assert (Hother :
    ∀ n : name,
      n <> Name_var var ->
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
      = Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)).
  { intros n Hne. unfold tenv'. rewrite add_variable_definition_lookup.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var var));
      [ contradiction | reflexivity ]. }
  assert (Hold :
    ∀ (n : name) (old : flambda_type * Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = Some old ->
      Keyed.lookup name_eqb n (Typing_env.names_to_types tenv') = Some old).
  { intros n old Hlookup. rewrite Hother; [ exact Hlookup | ].
    intros ->. congruence. }
  assert (Hmem :
    ∀ n : name,
      Typing_env.mem None tenv n = true ->
      Typing_env.mem None tenv' n = true).
  { intros [x | s] Hn; [ | exact Hn ].
    rewrite mem_var_is_bound in *.
    apply keyed_is_bound_lookup in Hn. destruct Hn as [old Hlookup].
    apply keyed_is_bound_lookup. exists old. exact (Hold _ _ Hlookup). }
  assert (Hlevels :
    all_levels tenv'
    = Typing_env_level.add_definition
        (Typing_env.One_level.level (Typing_env.current_level tenv)) var kind
        (Typing_env.next_binding_time tenv)
      :: map Typing_env.One_level.level (Typing_env.prev_levels tenv))
    by reflexivity.
  assert (Hnext :
    Typing_env.next_binding_time tenv'
    = Typing_env.next_binding_time tenv + 1) by reflexivity.
  destruct Hdisjoint as [Hcurrent Hprev].
  apply invariant_intro.
  - apply conditions_intro.
    + split; [ exact Hearliest | ].
      change (Typing_env.min_binding_time tenv
              <= Typing_env.next_binding_time tenv').
      lia.
    + intros x ty btm Hlookup. rewrite Hnext.
      destruct (Pos.eq_dec x var) as [-> | Hne].
      * rewrite Hnew in Hlookup. injection Hlookup as <- <-. simpl. lia.
      * rewrite Hother in Hlookup by congruence.
        destruct (Hrange x ty btm Hlookup). lia.
    + intros x y ty ty' btm btm' Hx Hy Htime.
      destruct (Pos.eq_dec x var) as [-> | Hx_ne],
        (Pos.eq_dec y var) as [-> | Hy_ne].
      * reflexivity.
      * exfalso. rewrite Hnew in Hx. injection Hx as <- <-.
        rewrite Hother in Hy by congruence.
        destruct (Hrange y ty' btm' Hy) as [_ Hlt]. simpl in Htime. lia.
      * exfalso. rewrite Hnew in Hy. injection Hy as <- <-.
        rewrite Hother in Hx by congruence.
        destruct (Hrange x ty btm Hx) as [_ Hlt]. simpl in Htime. lia.
      * rewrite Hother in Hx by congruence.
        rewrite Hother in Hy by congruence.
        exact (Hdistinct x y ty ty' btm btm' Hx Hy Htime).
    + intros n ty btm x Hlookup Hin.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var var))
        as [-> | Hne].
      * rewrite Hnew in Hlookup. injection Hlookup as <- <-.
        rewrite free_variables_unknown in Hin. destruct Hin.
      * rewrite Hother in Hlookup by exact Hne.
        pose proof (Hclosed n ty btm x Hlookup Hin) as Hbound.
        apply keyed_is_bound_lookup in Hbound.
        destruct Hbound as [old Hlookup'].
        apply keyed_is_bound_lookup. exists old. exact (Hold _ _ Hlookup').
    + intros s ty btm Hlookup. rewrite Hother in Hlookup by discriminate.
      exact (Hsymbols s ty btm Hlookup).
    + intros s ty btm Hlookup. rewrite Hother in Hlookup by discriminate.
      exact (Hsym_recorded s ty btm Hlookup).
    + rewrite Hlevels. split; [ | exact Hprev ].
      intros earlier Hin. destruct (Hcurrent earlier Hin) as [Hvars Htimes].
      split.
      * intros x Hx. rewrite add_definition_defined_vars.
        unfold is_bound. rewrite lookup_cons.
        destruct (Pos.eqb_spec x var) as [-> | Hne];
          [ exfalso | exact (Hvars x Hx) ].
        apply is_bound_lookup in Hx. destruct Hx as [k Hk].
        destruct (Hdefined earlier var k (or_intror Hin) Hk)
          as [ty [btm [Hlookup _]]].
        congruence.
      * intros binding_time Hbt. rewrite add_definition_is_bound_time.
        destruct (Z.eqb_spec binding_time (Typing_env.next_binding_time tenv))
          as [-> | Hne]; [ exfalso | exact (Htimes binding_time Hbt) ].
        pose proof (Hlevel_range earlier _ (or_intror Hin) Hbt). lia.
    + intros level x k Hin Hlookup. rewrite Hlevels in Hin.
      destruct Hin as [<- | Hin].
      * rewrite add_definition_defined_vars, lookup_cons in Hlookup.
        destruct (Pos.eqb_spec x var) as [-> | Hne].
        -- injection Hlookup as <-. exists (fst entry), (snd entry).
           split; [ exact Hnew | apply kind_unknown ].
        -- destruct (Hdefined _ x k (or_introl eq_refl) Hlookup)
             as [ty [btm [Hty Hkind]]].
           exists ty, btm. split; [ exact (Hold _ _ Hty) | exact Hkind ].
      * destruct (Hdefined level x k (or_intror Hin) Hlookup)
          as [ty [btm [Hty Hkind]]].
        exists ty, btm. split; [ exact (Hold _ _ Hty) | exact Hkind ].
    + intros level Hin. rewrite Hlevels in Hin. destruct Hin as [<- | Hin].
      * apply add_definition_indexed.
        exact (Hindexed _ (or_introl eq_refl)).
      * exact (Hindexed level (or_intror Hin)).
    + intros level binding_time x Hin Hat. rewrite Hlevels in Hin.
      assert (Hbefore :
        (binding_time = Typing_env.next_binding_time tenv ∧ x = var)
        ∨ ∃ level' : Typing_env_level.t,
            level' ∈ all_levels tenv ∧ indexed_at level' binding_time x).
      { destruct Hin as [<- | Hin].
        - apply add_definition_indexed_at in Hat.
          destruct Hat as [Hat | Hat]; [ left; exact Hat | right ].
          eexists. split; [ left; reflexivity | exact Hat ].
        - right. exists level. split; [ right; exact Hin | exact Hat ]. }
      destruct Hbefore as [[-> ->] | [level' [Hin' Hat']]].
      * exists (fst entry), (snd entry). split; [ exact Hnew | reflexivity ].
      * destruct (Hrecorded level' binding_time x Hin' Hat')
          as [ty [btm [Hty Htime]]].
        exists ty, btm. split; [ exact (Hold _ _ Hty) | exact Htime ].
    + intros level binding_time Hin Hbt. rewrite Hnext.
      rewrite Hlevels in Hin. destruct Hin as [<- | Hin].
      * rewrite add_definition_is_bound_time in Hbt.
        destruct (Z.eq_dec binding_time (Typing_env.next_binding_time tenv))
          as [-> | Hne]; [ lia | ].
        apply Z.eqb_neq in Hne. rewrite Hne in Hbt.
        pose proof (Hlevel_range _ binding_time (or_introl eq_refl) Hbt).
        lia.
      * pose proof (Hlevel_range level binding_time (or_intror Hin) Hbt).
        lia.
    + exact Hscopes.
    + intros x ty btm Hlookup. rewrite Hlevels.
      destruct (Pos.eq_dec x var) as [-> | Hne].
      * eexists. split; [ left; reflexivity | ].
        rewrite add_definition_defined_vars. unfold is_bound.
        rewrite lookup_cons, Pos.eqb_refl. reflexivity.
      * rewrite Hother in Hlookup by congruence.
        destruct (Hvar_defined x ty btm Hlookup) as [level [Hin Hbound]].
        destruct Hin as [<- | Hin].
        -- eexists. split; [ left; reflexivity | ].
           rewrite add_definition_defined_vars. unfold is_bound.
           rewrite lookup_cons.
           destruct (Pos.eqb_spec x var) as [Heq | _];
             [ contradiction | exact Hbound ].
        -- exists level. split; [ right; exact Hin | exact Hbound ].
    + exact Hconsistent.
    + apply (name_modes_recorded_coincide _ _
               (Typing_env.names_to_types tenv)
               (Typing_env.names_to_types tenv') (Typing_env.aliases tenv));
        [ | exact Hmodes ].
      intros n Hmention.
      pose proof (mentioned_known tenv n Hconsistent Hknown Hmention)
        as Hbound.
      destruct n as [x | s]; [ | reflexivity ].
      unfold Aliases.compute_name_mode_unscoped,
        Aliases.binding_time_and_name_mode.
      cbn [simple_of_name]. rewrite Hother; [ reflexivity | ].
      intros [= ->]. rewrite mem_var_is_bound in Hbound.
      unfold Keyed.is_bound in Hbound. rewrite Hfresh in Hbound.
      discriminate.
    + intros n c Hdemoted. destruct (Hknown n c Hdemoted) as [Hn Hc].
      split; [ exact (Hmem n Hn) | ].
      destruct c as [x | s | k];
        [ exact (Hmem (Name_var x) Hc) | exact (Hmem (Name_symbol s) Hc)
        | reflexivity ].
    + intros n ty btm alias Hlookup Halias.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq n (Name_var var))
        as [-> | Hne].
      * rewrite Hnew in Hlookup. injection Hlookup as <- <-.
        rewrite get_alias_opt_unknown in Halias. discriminate.
      * rewrite Hother in Hlookup by exact Hne.
        exact (Hconcrete n ty btm alias Hlookup Halias).
  - intros n ty Hlookup. destruct (Hagree n ty Hlookup) as [btm Hbtm].
    exists btm. exact (Hold _ _ Hbtm).
Qed.

Lemma mem_add_variable_definition (tenv : Typing_env.t) (var : variable)
    (kind : flambda_kind) (name_mode : Name_mode.t) (x : variable) :
  x <> var ->
  Typing_env.mem None
    (Typing_env.add_variable_definition tenv var kind name_mode) (Name_var x)
  = Typing_env.mem None tenv (Name_var x).
Proof.
  intros Hne. rewrite !mem_var_is_bound. unfold Keyed.is_bound.
  rewrite add_variable_definition_lookup.
  destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var x) (Name_var var))
    as [Heq | _]; [ congruence | reflexivity ].
Qed.

Theorem add_definitions_of_params_invariant
    : add_definitions_of_params_invariant_statement.
Proof.
  intros tenv params. revert tenv.
  induction params as [ | [x kind] params IH];
    intros tenv Hinvariant Hdistinct Hfresh; [ exact Hinvariant | ].
  change (negb (among x (map fst params)) && distinct (map fst params)
          = true) in Hdistinct.
  apply andb_true_iff in Hdistinct. destruct Hdistinct as [Hx Hdistinct].
  apply negb_true_iff in Hx.
  apply (IH (Typing_env.add_variable_definition tenv x kind
               Name_mode.normal)).
  - apply add_variable_definition_invariant; [ exact Hinvariant | ].
    apply Hfresh. left. reflexivity.
  - exact Hdistinct.
  - intros y Hin. rewrite mem_add_variable_definition.
    + apply Hfresh. right. exact Hin.
    + intros ->. apply among_In in Hin. congruence.
Qed.

(* ------------------------------------------------------------------------ *)
(* A new type for a name                                                    *)
(* ------------------------------------------------------------------------ *)

Lemma add_or_replace_equation_defined_vars (level : Typing_env_level.t)
    (n : name) (ty : flambda_type) :
  Typing_env_level.defined_vars
    (Typing_env_level.add_or_replace_equation level n ty)
  = Typing_env_level.defined_vars level.
Proof.
  unfold Typing_env_level.add_or_replace_equation.
  destruct (is_obviously_unknown ty); reflexivity.
Qed.

Lemma add_or_replace_equation_binding_times (level : Typing_env_level.t)
    (n : name) (ty : flambda_type) :
  Typing_env_level.binding_times
    (Typing_env_level.add_or_replace_equation level n ty)
  = Typing_env_level.binding_times level.
Proof.
  unfold Typing_env_level.add_or_replace_equation.
  destruct (is_obviously_unknown ty); reflexivity.
Qed.

Lemma add_or_replace_equation_unknown (level : Typing_env_level.t)
    (n : name) (ty : flambda_type) :
  is_obviously_unknown ty = true ->
  Typing_env_level.equations
    (Typing_env_level.add_or_replace_equation level n ty)
  = Keyed.unbind name_eqb n (Typing_env_level.equations level).
Proof.
  intros Hunknown. unfold Typing_env_level.add_or_replace_equation.
  rewrite Hunknown. reflexivity.
Qed.

Lemma add_or_replace_equation_known (level : Typing_env_level.t)
    (n : name) (ty : flambda_type) :
  is_obviously_unknown ty = false ->
  Typing_env_level.equations
    (Typing_env_level.add_or_replace_equation level n ty)
  = (n, ty) :: Typing_env_level.equations level.
Proof.
  intros Hunknown. unfold Typing_env_level.add_or_replace_equation.
  rewrite Hunknown. reflexivity.
Qed.

(* What [replace_equation] does to the types of the names: nothing for
   a variable that is not bound, and otherwise an entry for the name. *)
Lemma replace_equation_names_to_types (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  Typing_env.names_to_types (Typing_env.replace_equation tenv n ty)
  = match n with
    | Name_var _ =>
      match Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) with
      | Some (_, binding_time_and_mode) =>
        (n, (ty, binding_time_and_mode)) :: Typing_env.names_to_types tenv
      | None => Typing_env.names_to_types tenv
      end
    | Name_symbol _ =>
      (n,
       (ty,
        Binding_time.With_name_mode.create Binding_time.symbols
          Name_mode.normal))
      :: Typing_env.names_to_types tenv
    end.
Proof.
  destruct n as [x | s]; [ | reflexivity ].
  unfold Typing_env.replace_equation, Typing_env.names_to_types,
    Typing_env.with_current_level.
  cbn [Typing_env.just_after_level].
  unfold Cached_level.replace_variable_binding.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Cached_level.names_to_types (Typing_env.just_after_level tenv)))
    as [[ty0 btm0] | ]; reflexivity.
Qed.

Lemma replace_equation_aliases (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  Typing_env.aliases (Typing_env.replace_equation tenv n ty)
  = Typing_env.aliases tenv.
Proof.
  destruct n as [x | s]; [ | reflexivity ].
  unfold Typing_env.replace_equation, Typing_env.aliases,
    Typing_env.with_current_level.
  cbn [Typing_env.just_after_level].
  unfold Cached_level.replace_variable_binding.
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Cached_level.names_to_types (Typing_env.just_after_level tenv)))
    as [[ty0 btm0] | ]; reflexivity.
Qed.

(* An entry afterwards is the new one of the name, or one that was
   there. *)
Lemma replace_equation_lookup_cases (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) (m : name) (ty1 : flambda_type)
    (btm1 : Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb m
    (Typing_env.names_to_types (Typing_env.replace_equation tenv n ty))
  = Some (ty1, btm1) ->
  (m = n ∧ ty1 = ty)
  ∨ Keyed.lookup name_eqb m (Typing_env.names_to_types tenv)
    = Some (ty1, btm1).
Proof.
  rewrite replace_equation_names_to_types.
  assert (Hcons :
    ∀ btm : Binding_time.With_name_mode.t,
      Keyed.lookup name_eqb m
        ((n, (ty, btm)) :: Typing_env.names_to_types tenv)
      = Some (ty1, btm1) ->
      (m = n ∧ ty1 = ty)
      ∨ Keyed.lookup name_eqb m (Typing_env.names_to_types tenv)
        = Some (ty1, btm1)).
  { intros btm. rewrite keyed_lookup_cons.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [Heq | Hne].
    - intros [= <- _]. left. split; [ exact Heq | reflexivity ].
    - intros Hlookup. right. exact Hlookup. }
  destruct n as [x | s]; [ | apply Hcons ].
  destruct (Keyed.lookup name_eqb (Name_var x)
              (Typing_env.names_to_types tenv)) as [[ty0 btm0] | ];
    [ apply Hcons | intros Hlookup; right; exact Hlookup ].
Qed.

(* What [replace_equation] does to the types and the aliases, for a
   name that is bound: an entry is added for the name, with the binding
   time and the name mode that a variable had. *)
Lemma replace_equation_shape (tenv : Typing_env.t) (n : name)
    (ty : flambda_type) :
  Typing_env.mem None tenv n = true ->
  ∃ binding_time_and_mode : Binding_time.With_name_mode.t,
    Typing_env.names_to_types (Typing_env.replace_equation tenv n ty)
    = (n, (ty, binding_time_and_mode)) :: Typing_env.names_to_types tenv
    ∧ Typing_env.aliases (Typing_env.replace_equation tenv n ty)
      = Typing_env.aliases tenv
    ∧ (∀ (x : variable) (ty' : flambda_type)
         (binding_time_and_mode' : Binding_time.With_name_mode.t),
        n = Name_var x ->
        Keyed.lookup name_eqb n (Typing_env.names_to_types tenv)
        = Some (ty', binding_time_and_mode') ->
        binding_time_and_mode = binding_time_and_mode')
    ∧ (∀ s : symbol,
        n = Name_symbol s ->
        binding_time_and_mode = Binding_time.With_name_mode.symbols).
Proof.
  intros Hmem. destruct n as [x | s].
  - apply mem_var_lookup in Hmem. destruct Hmem as [ty0 [btm0 Hlookup]].
    exists btm0. split; [ | split; [ | split; [ | discriminate ] ] ].
    + unfold Typing_env.replace_equation, Typing_env.names_to_types,
        Typing_env.with_current_level in *.
      cbn [Typing_env.just_after_level].
      unfold Cached_level.replace_variable_binding. rewrite Hlookup.
      reflexivity.
    + unfold Typing_env.replace_equation, Typing_env.names_to_types,
        Typing_env.aliases, Typing_env.with_current_level in *.
      cbn [Typing_env.just_after_level].
      unfold Cached_level.replace_variable_binding. rewrite Hlookup.
      reflexivity.
    + intros x' ty' btm' [= <-] Hlookup'. rewrite Hlookup in Hlookup'.
      injection Hlookup' as _ <-. reflexivity.
  - exists (Binding_time.With_name_mode.create Binding_time.symbols
              Name_mode.normal).
    split; [ reflexivity | ]. split; [ reflexivity | ].
    split; [ | reflexivity ].
    intros x ty' btm' Hx. discriminate.
Qed.

Theorem replace_equation_invariant : replace_equation_invariant_statement.
Proof.
  intros tenv n ty Hinvariant Hbound Hfree Hkind Halias Hunknown.
  invariant_parts Hinvariant.
  destruct (replace_equation_shape tenv n ty Hbound)
    as [btm_new [Hmap [Haliases [Hbtm Hbtm_symbol]]]].
  set (tenv' := Typing_env.replace_equation tenv n ty) in *.
  (* The new entry, and the entries of the other names. *)
  assert (Hnew :
    Keyed.lookup name_eqb n (Typing_env.names_to_types tenv')
    = Some (ty, btm_new)).
  { rewrite Hmap. apply (keyed_lookup_head name_eqb name_eqb_eq). }
  assert (Hother :
    ∀ m : name,
      m <> n ->
      Keyed.lookup name_eqb m (Typing_env.names_to_types tenv')
      = Keyed.lookup name_eqb m (Typing_env.names_to_types tenv)).
  { intros m Hne. rewrite Hmap.
    apply (keyed_lookup_other name_eqb name_eqb_eq). exact Hne. }
  (* A variable keeps its binding time and its name mode. *)
  assert (Hvariable :
    ∀ (y : variable) (ty1 : flambda_type)
      (btm1 : Binding_time.With_name_mode.t),
      Keyed.lookup name_eqb (Name_var y) (Typing_env.names_to_types tenv')
      = Some (ty1, btm1) ->
      ∃ ty0 : flambda_type,
        Keyed.lookup name_eqb (Name_var y) (Typing_env.names_to_types tenv)
        = Some (ty0, btm1)).
  { intros y ty1 btm1 Hlookup.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var y) n)
      as [Heq | Hne].
    - subst n. rewrite Hnew in Hlookup. injection Hlookup as <- <-.
      apply mem_var_lookup in Hbound. destruct Hbound as [ty0 [btm0 Hold]].
      exists ty0. rewrite (Hbtm y ty0 btm0 eq_refl Hold). exact Hold.
    - rewrite (Hother _ Hne) in Hlookup. exists ty1. exact Hlookup. }
  (* A name that is bound stays bound. *)
  assert (Hstays :
    ∀ m : name,
      Keyed.is_bound name_eqb m (Typing_env.names_to_types tenv) = true ->
      Keyed.is_bound name_eqb m (Typing_env.names_to_types tenv') = true).
  { intros m Hm.
    destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [-> | Hne].
    - apply keyed_is_bound_lookup. eexists. exact Hnew.
    - unfold Keyed.is_bound. rewrite (Hother _ Hne). exact Hm. }
  assert (Hmem :
    ∀ m : name,
      Typing_env.mem None tenv m = true ->
      Typing_env.mem None tenv' m = true).
  { intros [x | s] Hm; [ | exact Hm ].
    rewrite mem_var_is_bound in *. exact (Hstays _ Hm). }
  assert (Hlevels :
    all_levels tenv'
    = Typing_env_level.add_or_replace_equation
        (Typing_env.One_level.level (Typing_env.current_level tenv)) n ty
      :: map Typing_env.One_level.level (Typing_env.prev_levels tenv))
    by reflexivity.
  apply invariant_intro.
  - apply conditions_intro.
    + exact Hbounded.
    + intros x ty1 btm1 Hlookup.
      destruct (Hvariable x ty1 btm1 Hlookup) as [ty0 Hold].
      exact (Hrange x ty0 btm1 Hold).
    + intros x y ty1 ty2 btm1 btm2 Hx Hy Htime.
      destruct (Hvariable x ty1 btm1 Hx) as [tyx Hx'].
      destruct (Hvariable y ty2 btm2 Hy) as [tyy Hy'].
      exact (Hdistinct x y tyx tyy btm1 btm2 Hx' Hy' Htime).
    + intros m ty1 btm1 x Hlookup Hin. apply Hstays.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [-> | Hne].
      * rewrite Hnew in Hlookup. injection Hlookup as <- <-.
        rewrite <- mem_var_is_bound. exact (Hfree x Hin).
      * rewrite (Hother _ Hne) in Hlookup.
        exact (Hclosed m ty1 btm1 x Hlookup Hin).
    + intros s ty1 btm1 Hlookup.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_symbol s) n)
        as [Heq | Hne].
      * subst n. exact Hbound.
      * rewrite (Hother _ Hne) in Hlookup.
        exact (Hsymbols s ty1 btm1 Hlookup).
    + intros s ty1 btm1 Hlookup.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_symbol s) n)
        as [Heq | Hne].
      * subst n. rewrite Hnew in Hlookup. injection Hlookup as _ <-.
        exact (Hbtm_symbol s eq_refl).
      * rewrite (Hother _ Hne) in Hlookup.
        exact (Hsym_recorded s ty1 btm1 Hlookup).
    + rewrite Hlevels. destruct Hdisjoint as [Hcurrent Hprev].
      split; [ | exact Hprev ].
      intros earlier Hin. destruct (Hcurrent earlier Hin) as [Hvars Htimes].
      split.
      * intros x Hx. rewrite add_or_replace_equation_defined_vars.
        exact (Hvars x Hx).
      * intros binding_time Hbt.
        rewrite add_or_replace_equation_binding_times.
        exact (Htimes binding_time Hbt).
    + intros level x k Hin Hlookup. rewrite Hlevels in Hin.
      assert (Hold :
        ∃ (ty1 : flambda_type) (btm1 : Binding_time.With_name_mode.t),
          Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
          = Some (ty1, btm1)
          ∧ kind ty1 = k).
      { destruct Hin as [<- | Hin].
        - rewrite add_or_replace_equation_defined_vars in Hlookup.
          exact (Hdefined _ x k (or_introl eq_refl) Hlookup).
        - exact (Hdefined level x k (or_intror Hin) Hlookup). }
      destruct Hold as [ty1 [btm1 [Hty1 Hkind1]]].
      destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var x) n)
        as [Heq | Hne].
      * subst n. exists ty, btm_new. split; [ exact Hnew | ].
        rewrite (Hkind ty1 btm1 Hty1). exact Hkind1.
      * exists ty1, btm1.
        split; [ rewrite (Hother _ Hne); exact Hty1 | exact Hkind1 ].
    + intros level Hin. rewrite Hlevels in Hin.
      destruct Hin as [<- | Hin]; [ | exact (Hindexed level (or_intror Hin)) ].
      pose proof (Hindexed _ (or_introl eq_refl)) as Hcurrent.
      intros x. unfold indexed_at.
      rewrite add_or_replace_equation_defined_vars,
        add_or_replace_equation_binding_times.
      exact (Hcurrent x).
    + intros level binding_time x Hin Hat. rewrite Hlevels in Hin.
      assert (Hold :
        ∃ (ty1 : flambda_type) (btm1 : Binding_time.With_name_mode.t),
          Keyed.lookup name_eqb (Name_var x) (Typing_env.names_to_types tenv)
          = Some (ty1, btm1)
          ∧ Binding_time.With_name_mode.binding_time btm1 = binding_time).
      { destruct Hin as [<- | Hin].
        - unfold indexed_at in Hat.
          rewrite add_or_replace_equation_binding_times in Hat.
          exact (Hrecorded _ binding_time x (or_introl eq_refl) Hat).
        - exact (Hrecorded level binding_time x (or_intror Hin) Hat). }
      destruct Hold as [ty1 [btm1 [Hty1 Htime]]].
      destruct (keyed_eqb_spec name_eqb name_eqb_eq (Name_var x) n)
        as [Heq | Hne].
      * subst n. exists ty, btm_new. split; [ exact Hnew | ].
        rewrite (Hbtm x ty1 btm1 eq_refl Hty1). exact Htime.
      * exists ty1, btm1.
        split; [ rewrite (Hother _ Hne); exact Hty1 | exact Htime ].
    + intros level binding_time Hin Hbt. rewrite Hlevels in Hin.
      destruct Hin as [<- | Hin].
      * rewrite add_or_replace_equation_binding_times in Hbt.
        exact (Hlevel_range _ binding_time (or_introl eq_refl) Hbt).
      * exact (Hlevel_range level binding_time (or_intror Hin) Hbt).
    + exact Hscopes.
    + intros y ty1 btm1 Hlookup. rewrite Hlevels.
      destruct (Hvariable y ty1 btm1 Hlookup) as [ty0 Hold].
      destruct (Hvar_defined y ty0 btm1 Hold) as [level [Hin Hbound_y]].
      destruct Hin as [<- | Hin].
      * eexists. split; [ left; reflexivity | ].
        rewrite add_or_replace_equation_defined_vars. exact Hbound_y.
      * exists level. split; [ right; exact Hin | exact Hbound_y ].
    + rewrite Haliases. exact Hconsistent.
    + rewrite Haliases.
      apply (name_modes_recorded_coincide _ _
               (Typing_env.names_to_types tenv)
               (Typing_env.names_to_types tenv') (Typing_env.aliases tenv));
        [ | exact Hmodes ].
      intros m _. destruct m as [y | s]; [ | reflexivity ].
      unfold Aliases.compute_name_mode_unscoped,
        Aliases.binding_time_and_name_mode.
      cbn [simple_of_name].
      destruct (Keyed.lookup name_eqb (Name_var y)
                  (Typing_env.names_to_types tenv'))
        as [[ty1 btm1] | ] eqn:Hlookup.
      * destruct (Hvariable y ty1 btm1 Hlookup) as [ty0 Hold].
        rewrite Hold. reflexivity.
      * destruct (Keyed.lookup name_eqb (Name_var y)
                    (Typing_env.names_to_types tenv))
          as [[ty0 btm0] | ] eqn:Hold; [ exfalso | reflexivity ].
        pose proof (Hstays (Name_var y)) as Hbound_y.
        unfold Keyed.is_bound in Hbound_y.
        rewrite Hold, Hlookup in Hbound_y.
        specialize (Hbound_y eq_refl). discriminate.
    + intros m c Hdemoted. rewrite Haliases in Hdemoted.
      destruct (Hknown m c Hdemoted) as [Hm Hc].
      split; [ exact (Hmem m Hm) | ].
      destruct c as [x | s | k];
        [ exact (Hmem (Name_var x) Hc) | exact (Hmem (Name_symbol s) Hc)
        | reflexivity ].
    + intros m ty1 btm1 alias Hlookup Halias1. rewrite Haliases.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [-> | Hne].
      * rewrite Hnew in Hlookup. injection Hlookup as <- <-.
        exact (Halias alias Halias1).
      * rewrite (Hother _ Hne) in Hlookup.
        exact (Hconcrete m ty1 btm1 alias Hlookup Halias1).
  - (* The equations of the levels. *)
    assert (Hall :
      all_equations tenv'
      = Typing_env_level.equations
          (Typing_env_level.add_or_replace_equation
             (Typing_env.One_level.level (Typing_env.current_level tenv)) n
             ty)
        ++ flat_map Typing_env_level.equations
             (map Typing_env.One_level.level (Typing_env.prev_levels tenv)))
      by reflexivity.
    assert (Hagree' :
      ∀ (m : name) (ty1 : flambda_type),
        m <> n ->
        match
          Keyed.lookup name_eqb m
            (Typing_env_level.equations
               (Typing_env.One_level.level (Typing_env.current_level tenv)))
        with
        | Some ty2 => Some ty2
        | None =>
          Keyed.lookup name_eqb m
            (flat_map Typing_env_level.equations
               (map Typing_env.One_level.level (Typing_env.prev_levels tenv)))
        end
        = Some ty1 ->
        ∃ btm : Binding_time.With_name_mode.t,
          Keyed.lookup name_eqb m (Typing_env.names_to_types tenv')
          = Some (ty1, btm)).
    { intros m ty1 Hne Hlookup. destruct (Hagree m ty1) as [btm1 Hbtm1].
      - unfold all_equations, all_levels. simpl.
        rewrite keyed_lookup_app. exact Hlookup.
      - exists btm1. rewrite (Hother _ Hne). exact Hbtm1. }
    intros m ty1 Hlookup. rewrite Hall, keyed_lookup_app in Hlookup.
    destruct (is_obviously_unknown ty) eqn:Hobvious.
    + rewrite (add_or_replace_equation_unknown _ n ty Hobvious),
        (keyed_lookup_unbind name name_eqb name_eqb_eq) in Hlookup.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [Heq | Hne].
      * subst m. destruct Hunknown as [Hfalse | Hnone]; [ discriminate | ].
        cbn in Hlookup. rewrite Hnone in Hlookup. discriminate.
      * exact (Hagree' m ty1 Hne Hlookup).
    + rewrite (add_or_replace_equation_known _ n ty Hobvious),
        keyed_lookup_cons in Hlookup.
      destruct (keyed_eqb_spec name_eqb name_eqb_eq m n) as [Heq | Hne].
      * subst m. cbn in Hlookup. injection Hlookup as <-.
        exists btm_new. exact Hnew.
      * exact (Hagree' m ty1 Hne Hlookup).
Qed.

(* ------------------------------------------------------------------------ *)
(* Two canonical elements made aliases                                      *)
(* ------------------------------------------------------------------------ *)

(* What [Typing_env.add_alias] answers. *)
Lemma add_alias_some (tenv : Typing_env.t)
    (canonical_element1 canonical_element2 : simple)
    (result : Typing_env.add_alias_result) :
  Typing_env.add_alias tenv canonical_element1 canonical_element2
  = Some (Ok result) ->
  ∃ added : Aliases.add_result,
    Aliases.add (Typing_env.binding_time_resolver tenv)
      (Typing_env.names_to_types tenv) (Typing_env.aliases tenv)
      canonical_element1 canonical_element2
    = Some (Or_bottom.Ok added)
    ∧ Typing_env.add_alias_result_t result
      = Typing_env.with_aliases tenv (Aliases.add_result_t added).
Proof.
  unfold Typing_env.add_alias, Typing_env.aliases_add.
  destruct (Aliases.add _ _ _ canonical_element1 canonical_element2)
    as [[added | ] | ]; try discriminate.
  intros [= <-]. exists added. split; reflexivity.
Qed.

(* A simple that is canonical after [Aliases.add] was canonical
   before. *)
Lemma add_canonical_before {A : Type}
    (binding_time_resolver : name -> Binding_time.With_name_mode.t)
    (binding_times_and_modes
      : Keyed.finite_map name (A * Binding_time.With_name_mode.t))
    (t : Aliases.t) (canonical_element1 canonical_element2 : simple)
    (added : Aliases.add_result) (e : simple) :
  aliases_consistent t ->
  is_canonical t canonical_element1 ->
  is_canonical t canonical_element2 ->
  Aliases.add binding_time_resolver binding_times_and_modes t
    canonical_element1 canonical_element2
  = Some (Or_bottom.Ok added) ->
  is_canonical (Aliases.add_result_t added) e ->
  is_canonical t e.
Proof.
  intros Hconsistent Hc1 Hc2 Hadd Hafter.
  destruct (canonical_cases t e)
    as [[Hcanonical _] | [n [c [-> [Hdemoted [_ Hof]]]]]];
    [ exact Hcanonical | exfalso ].
  pose proof (add_canonical_element A binding_time_resolver
                binding_times_and_modes t canonical_element1
                canonical_element2 added (simple_of_name n) Hconsistent Hc1
                Hc2 Hadd) as Hequation.
  rewrite (canonical_element_of_canonical _ _ Hafter), Hof in Hequation.
  assert (Hc : is_canonical t c) by exact (proj1 Hconsistent n c Hdemoted).
  assert (Hn : ~ is_canonical t (simple_of_name n)).
  { unfold is_canonical. rewrite canonical_of_name.
    unfold demoted_to in Hdemoted. rewrite Hdemoted. discriminate. }
  destruct (simple_equal c (simple_of_name (Aliases.demoted_name added))).
  - pose proof (add_result A binding_time_resolver binding_times_and_modes t
                  canonical_element1 canonical_element2 added Hadd)
      as Hresult.
    destruct (Aliases.defined_earlier binding_time_resolver
                binding_times_and_modes canonical_element1
                canonical_element2);
      destruct Hresult as [Hcanonical _];
      rewrite Hcanonical in Hequation; rewrite Hequation in Hn;
      [ exact (Hn Hc1) | exact (Hn Hc2) ].
  - rewrite Hequation in Hn. exact (Hn Hc).
Qed.

Theorem add_alias_invariant : add_alias_invariant_statement.
Proof.
  intros tenv c1 c2 result Hinvariant Hc1 Hc2 Hbound1 Hbound2 Hadd.
  destruct (add_alias_some tenv c1 c2 result Hadd) as [added [Hadded ->]].
  invariant_parts Hinvariant.
  apply invariant_intro; [ | exact Hagree ].
  apply conditions_intro;
    try first
      [ exact Hbounded | exact Hrange | exact Hdistinct | exact Hclosed
      | exact Hsymbols | exact Hsym_recorded | exact Hdisjoint
      | exact Hdefined | exact Hindexed | exact Hrecorded
      | exact Hlevel_range | exact Hscopes | exact Hvar_defined ].
  - exact (add_consistent _ _ _ _ c1 c2 added Hconsistent Hc1 Hc2 Hadded).
  - exact (add_name_modes_recorded _ _ _ _ c1 c2 added Hmodes Hadded).
  - assert (Hmentioned :
      ∀ n : name,
        aliases_mention (Aliases.add_result_t added) n ->
        Typing_env.mem None tenv n = true).
    { intros n Hmention.
      destruct (add_mentioned _ _ _ _ c1 c2 added n Hadded Hmention)
        as [Hold | [Heq | Heq]].
      - exact (mentioned_known tenv n Hconsistent Hknown Hold).
      - rewrite <- Heq in Hbound1. destruct n; exact Hbound1.
      - rewrite <- Heq in Hbound2. destruct n; exact Hbound2. }
    intros n c Hdemoted. split.
    + apply (Hmentioned n). left. exists c. exact Hdemoted.
    + destruct c as [x | s | k]; [ | | reflexivity ].
      * apply (Hmentioned (Name_var x)). right. left. exists n.
        exact Hdemoted.
      * apply (Hmentioned (Name_symbol s)). right. left. exists n.
        exact Hdemoted.
  - intros n ty btm alias Hlookup Halias Hcanonical.
    apply (Hconcrete n ty btm alias Hlookup Halias).
    exact (add_canonical_before _ _ _ c1 c2 added _ Hconsistent Hc1 Hc2
             Hadded Hcanonical).
Qed.

Theorem add_alias_defined : add_alias_defined_statement.
Proof.
  intros tenv c1 c2 Hinvariant Hc1 Hc2 Hne.
  invariant_parts Hinvariant.
  unfold Typing_env.add_alias, Typing_env.aliases_add.
  pose proof (add_defined _ (Typing_env.binding_time_resolver tenv)
                (Typing_env.names_to_types tenv) (Typing_env.aliases tenv)
                c1 c2 Hconsistent Hc1 Hc2 Hne) as Hanswers.
  destruct (Aliases.add _ _ _ c1 c2) as [[added | ] | ];
    [ discriminate | discriminate | contradiction ].
Qed.

(* ======================================================================== *)
(* Well-formed types                                                        *)
(* ======================================================================== *)

Theorem types_well_formed_b_iff : types_well_formed_b_iff_statement.
Proof.
  intros tenv. unfold types_well_formed_b, types_well_formed.
  eapply iff_trans.
  - apply (for_all_names_b_iff
             (fun _ entry => type_well_formed (fst entry) = true)).
    intros n entry _. reflexivity.
  - split.
    + intros Hall n ty btm Hlookup. exact (Hall n (ty, btm) Hlookup).
    + intros Hall n [ty btm] Hlookup. exact (Hall n ty btm Hlookup).
Qed.

Theorem find_well_formed : find_well_formed_statement.
Proof.
  intros tenv n kind ty Hwf Hfind.
  destruct (find_cases tenv n kind ty Hfind)
    as [[binding_time_and_mode Hlookup] | [[k ->] | ->]].
  - exact (Hwf n ty binding_time_and_mode Hlookup).
  - apply bottom_well_formed.
  - apply any_non_null_value_well_formed.
Qed.

Theorem create_types_well_formed : create_types_well_formed_statement.
Proof. intros n ty btm Hlookup. discriminate. Qed.

Theorem make_bottom_types_well_formed
    : make_bottom_types_well_formed_statement.
Proof. intros tenv Hwf. exact Hwf. Qed.

Theorem increment_scope_types_well_formed
    : increment_scope_types_well_formed_statement.
Proof. intros tenv Hwf. exact Hwf. Qed.

Theorem closure_env_types_well_formed
    : closure_env_types_well_formed_statement.
Proof. intros tenv Hwf. exact Hwf. Qed.

Theorem add_symbol_definition_types_well_formed
    : add_symbol_definition_types_well_formed_statement.
Proof. intros tenv sym Hwf. exact Hwf. Qed.

Theorem add_variable_definition_types_well_formed
    : add_variable_definition_types_well_formed_statement.
Proof.
  intros tenv var kind name_mode Hwf n ty btm Hlookup.
  rewrite add_variable_definition_lookup in Hlookup.
  destruct (name_eqb n (Name_var var)).
  - injection Hlookup as <- <-. apply unknown_well_formed.
  - exact (Hwf n ty btm Hlookup).
Qed.

Theorem replace_equation_types_well_formed
    : replace_equation_types_well_formed_statement.
Proof.
  intros tenv n ty Hty Hwf m ty1 btm1 Hlookup.
  destruct (replace_equation_lookup_cases tenv n ty m ty1 btm1 Hlookup)
    as [[_ ->] | Hold].
  - exact Hty.
  - exact (Hwf m ty1 btm1 Hold).
Qed.

Theorem add_alias_types_well_formed : add_alias_types_well_formed_statement.
Proof.
  intros tenv c1 c2 result Hadd Hwf.
  destruct (add_alias_some tenv c1 c2 result Hadd) as [added [_ ->]].
  exact Hwf.
Qed.
