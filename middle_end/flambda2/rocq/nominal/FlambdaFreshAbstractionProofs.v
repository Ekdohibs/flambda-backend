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

(* Proofs of the statements of FlambdaFreshAbstractionSpec.v. Nothing
   here needs review.

   Each equivalence has the same two halves. From the equality with
   permutations to the quantifier: the swaps with fresh names open the
   two abstractions at the same bound names, so
   [pattern_match_pair] gives equal scopes. From the quantifier to the
   equality with permutations: take fresh names outside the free names
   of both, and the swaps are the two permutations asked for. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaMapSpec FlambdaMapProofs
  FlambdaNominalSpec FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaFreshQuantifierSpec
  FlambdaFreshQuantifierProofs FlambdaFreshAbstractionSpec.
Import ListNotations.
Import Nominal.

(* ======================================================================== *)
(* One bound name                                                           *)
(* ======================================================================== *)

Section One_name.

Context {T : Type} (NT : contains_names T).

Lemma opens_swap (a : name_abstraction positive T) (fresh : positive) :
  among fresh (free_names_abstraction single_name NT a) = false ->
  opens single_name NT [(bound a, fresh)] a.
Proof.
  intros Hfresh name Hname. simpl. apply swap_other.
  - rewrite among_free_names_abstraction in Hname.
    apply andb_true_iff in Hname as [_ Hname]. simpl in Hname.
    rewrite orb_false_r in Hname. apply negb_true_iff, Pos.eqb_neq in Hname.
    exact Hname.
  - intros ->. congruence.
Qed.

Lemma pattern_match_swap (a : name_abstraction positive T)
    (fresh : positive) :
  pattern_match single_name NT [(bound a, fresh)] a
  = (fresh, concretion NT a fresh).
Proof.
  unfold pattern_match, concretion. simpl. rewrite swap_first. reflexivity.
Qed.

(* Equal concretions at one name that is free in neither abstraction
   make the abstractions equal. *)
Lemma equal_abstraction_of_concretion (a a' : name_abstraction positive T)
    (fresh : positive) :
  among fresh (free_names_abstraction single_name NT a) = false ->
  among fresh (free_names_abstraction single_name NT a') = false ->
  equal NT (concretion NT a fresh) (concretion NT a' fresh) ->
  equal_abstraction single_name NT a a'.
Proof.
  intros Hfresh Hfresh' Hequal.
  exists [(bound a, fresh)], [(bound a', fresh)].
  split; [ apply opens_swap, Hfresh | ].
  split; [ apply opens_swap, Hfresh' | ].
  rewrite !pattern_match_swap. split; [ reflexivity | exact Hequal ].
Qed.

Hypothesis HT : nominal NT.

(* Equal abstractions have equal concretions at every such name. *)
Lemma concretion_of_equal_abstraction (a a' : name_abstraction positive T)
    (fresh : positive) :
  equal_abstraction single_name NT a a' ->
  among fresh (free_names_abstraction single_name NT a) = false ->
  among fresh (free_names_abstraction single_name NT a') = false ->
  equal NT (concretion NT a fresh) (concretion NT a' fresh).
Proof.
  intros Hequal Hfresh Hfresh'.
  pose proof
    (pattern_match_pair positive T single_name NT a a' [(bound a, fresh)]
       [(bound a', fresh)] single_name_bindable HT Hequal
       (opens_swap a fresh Hfresh) (opens_swap a' fresh Hfresh'))
    as Hpair.
  rewrite !pattern_match_swap in Hpair. exact (Hpair eq_refl).
Qed.

End One_name.

Theorem concretion_pattern_match : concretion_pattern_match_statement.
Proof.
  intros T NT a fresh Hfresh.
  split; [ apply opens_swap, Hfresh | apply pattern_match_swap ].
Qed.

Theorem equal_abstraction_some_any : equal_abstraction_some_any_statement.
Proof.
  intros T NT a a' HT.
  set (names :=
    free_names_abstraction single_name NT a
    ++ free_names_abstraction single_name NT a').
  pose proof (fresh_name_fresh names) as Hfresh.
  apply among_app_false in Hfresh as [Hfresh Hfresh'].
  split; split.
  - intros Hequal fresh. apply concretion_of_equal_abstraction; assumption.
  - intros Hall.
    apply (equal_abstraction_of_concretion NT a a' (fresh_name names));
      try assumption.
    apply Hall; assumption.
  - intros Hequal. exists (fresh_name names).
    split; [ exact Hfresh | ]. split; [ exact Hfresh' | ].
    apply concretion_of_equal_abstraction; assumption.
  - intros (fresh & Hname & Hname' & Hequal).
    exact (equal_abstraction_of_concretion NT a a' fresh Hname Hname'
             Hequal).
Qed.

Theorem equal_abstraction_fresh : equal_abstraction_fresh_statement.
Proof.
  intros T NT a a' HT. split.
  - intros Hequal.
    exists
      (free_names_abstraction single_name NT a
       ++ free_names_abstraction single_name NT a').
    intros fresh Hfresh. apply among_app_false in Hfresh as [Hfresh Hfresh'].
    apply concretion_of_equal_abstraction; assumption.
  - intros Hfresh.
    destruct
      (for_fresh_elim _
         (free_names_abstraction single_name NT a
          ++ free_names_abstraction single_name NT a')
         Hfresh)
      as (fresh & Hname & Hequal).
    apply among_app_false in Hname as [Hname Hname'].
    exact (equal_abstraction_of_concretion NT a a' fresh Hname Hname'
             Hequal).
Qed.

Theorem pattern_match_pair_fresh : pattern_match_pair_fresh_statement.
Proof.
  intros T NT a a' p p' HT Hfresh.
  apply (pattern_match_pair positive T single_name NT a a' p p'
           single_name_bindable HT).
  apply (equal_abstraction_fresh T NT a a' HT). exact Hfresh.
Qed.

(* Each law is the law of [abstraction_names], carried across the
   equivalence of the two equalities. *)
Theorem fresh_abstraction_nominal : fresh_abstraction_nominal_statement.
Proof.
  intros T NT HT.
  pose proof
    (abstraction_nominal positive T single_name NT single_name_bindable HT)
    as Hlaws.
  pose proof (fun a a' => equal_abstraction_fresh T NT a a' HT) as Hsame.
  constructor.
  - intros a. apply Hsame, (equal_refl _ Hlaws).
  - intros a a' Hequal. apply Hsame, (equal_sym _ Hlaws), Hsame, Hequal.
  - intros a a' a'' Hequal Hequal'.
    apply Hsame, (equal_trans _ Hlaws) with a'; apply Hsame; assumption.
  - exact (apply_renaming_identity _ Hlaws).
  - exact (apply_renaming_compose _ Hlaws).
  - exact (apply_renaming_same _ Hlaws).
  - intros p a a' Hequal.
    apply Hsame, (equal_equivariant _ Hlaws), Hsame, Hequal.
  - exact (free_names_equivariant _ Hlaws).
  - intros a a' Hequal. apply (free_names_equal _ Hlaws), Hsame, Hequal.
  - intros p a Hfixes. apply Hsame, (apply_renaming_fixes _ Hlaws), Hfixes.
Qed.

(* ======================================================================== *)
(* A list of bound names                                                    *)
(* ======================================================================== *)

Theorem concretion_names_pattern_match
  : concretion_names_pattern_match_statement.
Proof.
  intros T NT a fresh Hdistinct Hlength Houtside.
  apply distinct_outside_app in Houtside as [Hbound [_ Hfree]]. split.
  - intros name Hname. apply permute_combine_other.
    + rewrite among_free_names_abstraction in Hname.
      apply andb_true_iff in Hname as [_ Hname].
      apply negb_true_iff in Hname. exact Hname.
    + destruct (among name fresh) eqn:Hamong; [ | reflexivity ].
      apply Hfree in Hamong. congruence.
  - unfold pattern_match, concretion_names. simpl.
    rewrite permute_combine; [ reflexivity | congruence | | ]; assumption.
Qed.

Theorem equal_abstraction_names_fresh
  : equal_abstraction_names_fresh_statement.
Proof.
  intros T NT a a' HT Hdistinct Hdistinct'.
  set (names :=
    (bound a ++ free_names_abstraction list_of_names NT a)
    ++ bound a' ++ free_names_abstraction list_of_names NT a').
  split.
  - intros Hequal.
    assert (Hlengths : length (bound a) = length (bound a')).
    { destruct Hequal as (p & p' & _ & _ & Hbound & _). simpl in Hbound.
      apply (f_equal (@length positive)) in Hbound.
      rewrite !length_map in Hbound. exact Hbound. }
    split; [ exact Hlengths | ]. exists names.
    intros fresh Hlength Houtside.
    apply distinct_outside_app in Houtside as [Houtside Houtside'].
    destruct
      (concretion_names_pattern_match T NT a fresh Hdistinct Hlength
         Houtside)
      as [Hopens Hmatch].
    destruct
      (concretion_names_pattern_match T NT a' fresh Hdistinct'
         ltac:(congruence) Houtside')
      as [Hopens' Hmatch'].
    pose proof
      (pattern_match_pair _ T list_of_names NT a a' _ _
         list_of_names_bindable HT Hequal Hopens Hopens')
      as Hpair.
    rewrite Hmatch, Hmatch' in Hpair. exact (Hpair eq_refl).
  - intros [Hlengths Hfresh].
    destruct (for_fresh_names_elim _ _ names Hfresh)
      as (fresh & Hlength & Houtside & Hequal).
    apply distinct_outside_app in Houtside as [Houtside Houtside'].
    destruct
      (concretion_names_pattern_match T NT a fresh Hdistinct Hlength
         Houtside)
      as [Hopens Hmatch].
    destruct
      (concretion_names_pattern_match T NT a' fresh Hdistinct'
         ltac:(congruence) Houtside')
      as [Hopens' Hmatch'].
    exists (combine (bound a) fresh), (combine (bound a') fresh).
    split; [ exact Hopens | ]. split; [ exact Hopens' | ].
    rewrite Hmatch, Hmatch'. split; [ reflexivity | exact Hequal ].
Qed.

(* ======================================================================== *)
(* Functions of an opened abstraction                                       *)
(* ======================================================================== *)

(* From the quantifier: exchange the name for a fresh one. The function
   is equivariant, so the name is free in its result exactly when the
   fresh one is free in its result at the renamed argument, and it is
   not. *)
Theorem freshness_condition : freshness_condition_statement.
Proof.
  intros B T X NB NT NX f HB HX Hequivariant. split.
  - intros Hfresh ab name Hbound.
    destruct (for_fresh_elim _ [] Hfresh) as (fresh & _ & Hcondition).
    set (p := [(name, fresh)]).
    assert (Hmoved : permute p name = fresh) by apply swap_first.
    assert (Hrenamed :
      among fresh (free_names NB (apply_renaming NB p (fst ab))) = true).
    { rewrite <- Hmoved, (free_names_equivariant NB HB). exact Hbound. }
    specialize
      (Hcondition (apply_renaming (pair_names NB NT) p ab) Hrenamed).
    rewrite (free_names_equal NX HX _ _ (Hequivariant p ab) fresh)
      in Hcondition.
    rewrite <- Hmoved, (free_names_equivariant NX HX) in Hcondition.
    exact Hcondition.
  - intros Hall. exists []. intros name _ ab. apply Hall.
Qed.

Theorem pattern_match_well_defined_fresh
  : pattern_match_well_defined_fresh_statement.
Proof.
  intros B T X NB NT NX f a a' p p' Hbindable HT HX Hequivariant Hrespects
    Hfresh.
  apply pattern_match_well_defined; try assumption.
  apply (freshness_condition B T X NB NT NX f
           (bindable_nominal NB Hbindable) HX Hequivariant).
  exact Hfresh.
Qed.

Theorem concretion_well_defined : concretion_well_defined_statement.
Proof.
  intros T X NT NX f a a' HT HX Hequivariant Hrespects Hcondition Hequal.
  assert (Hfresh :
    И name,
      ∀ ab : positive * T,
        among name (free_names single_name (fst ab)) = true ->
        among name (free_names NX (f ab)) = false).
  { destruct Hcondition as [avoid Hcondition]. exists avoid.
    intros name Hname [bound_name t] Hbound. simpl in Hbound.
    rewrite orb_false_r in Hbound. apply Pos.eqb_eq in Hbound.
    subst bound_name. exact (Hcondition name Hname t). }
  exists (free_names_abstraction single_name NT a). intros fresh Hname.
  exists (free_names_abstraction single_name NT a'). intros fresh' Hname'.
  pose proof
    (pattern_match_well_defined_fresh positive T X single_name NT NX f a a'
       [(bound a, fresh)] [(bound a', fresh')] single_name_bindable HT HX
       Hequivariant Hrespects Hfresh Hequal (opens_swap NT a fresh Hname)
       (opens_swap NT a' fresh' Hname'))
    as Hdefined.
  rewrite !pattern_match_swap in Hdefined. exact Hdefined.
Qed.

(* ======================================================================== *)
(* Functions that make fresh names                                          *)
(* ======================================================================== *)

(* The swaps of the first names with the second open the first
   abstraction, since an equivariant function invents no name: a free
   name of its result that is not one of the names given is free in
   the argument, and the swaps move no such name. They carry the first
   result to the second because the function is equivariant. *)
Theorem fresh_choice : fresh_choice_statement.
Proof.
  intros X Y NX NY g x count HX HY Hequivariant Hrespects.
  pose proof (pair_nominal _ X list_of_names NX list_of_names_nominal HX)
    as Hpair.
  exists (free_names NX x). intros names Hlength [Hdistinct Houtside].
  exists (free_names NX x ++ names). intros names' Hlength' Houtside'.
  apply distinct_outside_app in Houtside' as [[_ Houtside'] Hdisjoint].
  set (p := combine names names').
  assert (Hfixes : fixes p (free_names NX x))
    by (apply fixes_combine; assumption).
  assert (Hmoved : map (permute p) names = names').
  { apply permute_combine; [ congruence | exact Hdistinct | ].
    exact Hdisjoint. }
  exists p, identity_permutation.
  split; [ | split; [ intros name _; reflexivity | split ] ].
  - intros name Hname. rewrite among_free_names_abstraction in Hname.
    apply andb_true_iff in Hname as [Hresult Hnot_given].
    apply negb_true_iff in Hnot_given. simpl in Hresult, Hnot_given.
    pose proof
      (invents_no_names _ Y (pair_names list_of_names NX) NY g (names, x)
         name Hpair HY Hequivariant Hrespects Hresult)
      as Hargument.
    simpl in Hargument. rewrite among_app, Hnot_given in Hargument.
    exact (Hfixes name Hargument).
  - simpl. rewrite Hmoved. symmetry.
    exact (apply_renaming_identity list_of_names list_of_names_nominal
             names').
  - simpl. rewrite (apply_renaming_identity NY HY).
    apply (equal_trans NY HY) with (g (names', apply_renaming NX p x)).
    + apply (equal_sym NY HY).
      pose proof (Hequivariant p (names, x)) as Hrenamed.
      simpl in Hrenamed. rewrite Hmoved in Hrenamed. exact Hrenamed.
    + apply Hrespects. split; [ reflexivity | ].
      exact (apply_renaming_fixes NX HX p x Hfixes).
Qed.
