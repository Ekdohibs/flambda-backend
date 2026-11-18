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


(* The proofs of FlambdaReaperWrapperDerivedSpec.v. Nothing here needs
   review. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionProofs
  FlambdaReaperRebuild FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeast FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperWrapperSpec FlambdaReaperWrapperShapeSpec
  FlambdaReaperWrapperAutoSpec FlambdaReaperWrapperUnknownSpec
  FlambdaReaperWrapperDerivedSpec.
From OxCamlFlambda2 Require FlambdaReaperLeastProofs FlambdaReaperTypesProofs
  FlambdaReaperWrapperAutoProofs FlambdaReaperWrapperUnknownProofs.
Import ListNotations.

(* ======================================================================== *)
(* The tables of the unboxing are closed                                    *)
(* ======================================================================== *)

Lemma in_dedup_iff (l : list node) (x : node) :
  x ∈ Keyed.dedup node_eqb l <-> x ∈ l.
Proof. exact (FlambdaReaperTypesProofs.In_dedup_nodes x l). Qed.

Ltac gd t E := destruct t eqn:E; cbn [negb orb andb]; try reflexivity.

Ltac fin :=
  repeat match goal with
  | H : ?b = true |- context [?b] => rewrite H
  | H : ?b = false |- context [?b] => rewrite H
  end;
  rewrite ?node_eqb_refl, ?field_eqb_refl; cbn [negb orb andb];
  first [ left; reflexivity | right; left; reflexivity ].

Tactic Notation "pick" int_or_var(n) :=
  do n (apply in_or_app; right); try (apply in_or_app; left).

Section Closed.

Variable is_local : field -> bool.
Variable F : facts.
Variable T : tables.
Variable cannot_change : list node.
Variable max_unbox_size : nat.
Variable lambda_lifting : bool.
Variable R : unboxing_tables.

Hypothesis H0 : ∀ x,
  x ∈ derived_cannot_change_representation0 is_local F T -> x ∈ ut_crr0 R.
Hypothesis H1a : ∀ x, x ∈ ut_crr0 R -> x ∈ ut_crr1 R.
Hypothesis H1b : ∀ x, x ∈ by_function_slots F (ut_crr0 R) -> x ∈ ut_crr1 R.
Hypothesis H2a : ∀ x, x ∈ ut_crr1 R -> x ∈ ut_crr R.
Hypothesis H2b : ∀ x f y,
  (x, f, y) ∈ f_constructor F -> is_block_like_field f = true ->
  x ∈ ut_crr R.
Hypothesis H3 : ∀ x,
  x ∈ derived_cannot_unbox0_seeds F T cannot_change max_unbox_size
        lambda_lifting (ut_crr1 R) ->
  x ∈ ut_cu0 R.
Hypothesis H4 : stored_round F T (ut_crr R) (ut_cu0 R) = [].
Hypothesis H5a : ∀ x, x ∈ ut_cu0 R -> x ∈ ut_cu R.
Hypothesis H5b : ∀ x, x ∈ by_function_slots F (ut_cu0 R) -> x ∈ ut_cu R.

Lemma unboxing_closed :
  check_unboxing_closed is_local F T cannot_change max_unbox_size
    lambda_lifting R
  = true.
Proof.
  unfold check_unboxing_closed.
  repeat (apply andb_true_intro; split).
  - unfold check_crr0_any_usage. apply forallb_forall.
    intros [[x f] y] Hin. cbv beta iota.
    gd (mem_node x (t_any_usage T)) E1. gd (is_local f) E2.
    gd (is_real_field f) E3.
    apply mem_node_In, H0. pick 0. apply in_flat_map.
    exists (x, f, y). split; [ exact Hin | ]. fin.
  - unfold check_crr0_local_reading. apply forallb_forall.
    intros [[x f] y] Hin. cbv beta iota.
    gd (mem_node x (t_any_usage T)) E1. gd (is_local f) E2.
    apply forallb_forall. intros [f' z] Hr. cbv beta iota.
    gd (field_eqb f f') E3.
    apply mem_node_In, H0. pick 1. apply in_flat_map.
    exists (x, f, y). split; [ exact Hin | ].
    assert (Hex : existsb (fun '(field', _) => field_eqb f field')
                    (t_reading_field T) = true)
      by (apply existsb_exists; exists (f', z); split; [ exact Hr | ];
          exact E3).
    fin.
  - unfold check_crr0_local_sources. apply forallb_forall.
    intros [[v f] u] Hin. cbv beta iota.
    gd (mem_node v (t_has_usage T)) E1. gd (is_local f) E2.
    apply forallb_forall. intros [u1 s1] Hs1. cbv beta iota.
    gd (node_eqb u u1) E3. gd (mem_node s1 (t_has_source T)) E4.
    apply forallb_forall. intros [u2 s2] Hs2. cbv beta iota.
    gd (node_eqb u u2) E5. gd (mem_node s2 (t_has_source T)) E6.
    gd (node_eqb s1 s2) E7.
    apply node_eqb_eq in E3, E5. subst u1 u2.
    apply mem_node_In, H0. pick 2. apply in_flat_map.
    exists (v, f, u). split; [ exact Hin | ]. cbv beta iota.
    rewrite E1, E2. cbn [andb]. apply in_flat_map.
    exists (u, s1). split; [ exact Hs1 | ].
    assert (Hex : existsb (fun '(usage2, source2) =>
                             node_eqb u usage2
                             && mem_node source2 (t_has_source T)
                             && negb (node_eqb s1 source2))
                    (t_sources T) = true)
      by (apply existsb_exists; exists (u, s2); split; [ exact Hs2 | ];
          rewrite node_eqb_refl, E6, E7; reflexivity).
    fin.
  - unfold check_crr0_other_source. apply forallb_forall.
    intros [al a] Hu. apply forallb_forall. intros [a' s] Hs.
    cbv beta iota.
    gd (node_eqb a a') E1. gd (mem_node s (t_has_source T)) E2.
    gd (node_eqb s al) E3.
    apply forallb_forall. intros [[v f] a''] Hacc. cbv beta iota.
    gd (node_eqb a a'') E4. gd (is_real_field f) E5.
    gd (mem_node v (t_has_usage T)) E6.
    apply node_eqb_eq in E1, E4. subst a' a''.
    apply mem_node_In, H0. pick 3. apply in_flat_map.
    exists (al, a). split; [ exact Hu | ].
    assert (Hex1 : existsb (fun '(alias', alias_source) =>
                              node_eqb a alias'
                              && mem_node alias_source (t_has_source T)
                              && negb (node_eqb alias_source al))
                     (t_sources T) = true)
      by (apply existsb_exists; exists (a, s); split; [ exact Hs | ];
          rewrite node_eqb_refl, E2, E3; reflexivity).
    assert (Hex2 : real_field_read_with_usage F T a = true)
      by (apply existsb_exists; exists (v, f, a); split; [ exact Hacc | ];
          rewrite node_eqb_refl, E5, E6; reflexivity).
    fin.
  - unfold check_crr0_any_source_usage. apply forallb_forall.
    intros [al a] Hu. cbv beta iota.
    gd (mem_node a (t_any_source T)) E1.
    apply forallb_forall. intros [[v f] a'] Hacc. cbv beta iota.
    gd (node_eqb a a') E2. gd (is_real_field f) E3.
    gd (mem_node v (t_has_usage T)) E4.
    apply node_eqb_eq in E2. subst a'.
    apply mem_node_In, H0. pick 4. apply in_flat_map.
    exists (al, a). split; [ exact Hu | ].
    assert (Hex2 : real_field_read_with_usage F T a = true)
      by (apply existsb_exists; exists (v, f, a); split; [ exact Hacc | ];
          rewrite node_eqb_refl, E3, E4; reflexivity).
    fin.
  - unfold check_crr0_other_own_source. apply forallb_forall.
    intros [al s] Hs. cbv beta iota.
    gd (mem_node s (t_has_source T)) E1. gd (node_eqb s al) E2.
    apply mem_node_In, H0. pick 5. apply in_flat_map.
    exists (al, s). split; [ exact Hs | ]. fin.
  - unfold check_crr0_not_own_source. apply forallb_forall.
    intros [al u] Hu. cbv beta iota.
    gd (mem_node u (t_has_usage T)) E1. gd (mem_pair al al (t_sources T)) E2.
    apply mem_node_In, H0. pick 6. apply in_flat_map.
    exists (al, u). split; [ exact Hu | ]. fin.
  - unfold check_crr0_any_source. apply forallb_forall. intros x Hx.
    apply mem_node_In, H0. pick 7. exact Hx.
  - unfold check_crr0_call_witness. apply forallb_forall.
    intros [[cw f] y] Hin. cbv beta iota.
    gd (field_eqb f Field_code_id_of_call_witness) E1.
    apply mem_node_In, H0. pick 8. apply in_flat_map.
    exists (cw, f, y). split; [ exact Hin | ]. fin.
  - unfold check_crr1_crr0. apply forallb_forall. intros x Hx.
    apply mem_node_In, H1a. exact Hx.
  - unfold check_crr1_function_slot. apply forallb_forall.
    intros [[x f] y] Hin. cbv beta iota.
    gd (is_function_slot f) E1. gd (mem_node x (ut_crr0 R)) E2.
    apply mem_node_In, H1b. unfold by_function_slots. apply in_flat_map.
    exists (x, f, y). split; [ exact Hin | ]. fin.
  - unfold check_crr_crr1. apply forallb_forall. intros x Hx.
    apply mem_node_In, H2a. exact Hx.
  - unfold check_crr_block_like. apply forallb_forall.
    intros [[x f] y] Hin. cbv beta iota.
    gd (is_block_like_field f) E1.
    apply mem_node_In. exact (H2b x f y Hin E1).
  - unfold check_cu0_cannot_change_representation. apply forallb_forall.
    intros x Hx. apply mem_node_In, H3. pick 0. exact Hx.
  - unfold check_cu0_any_usage. apply forallb_forall.
    intros x Hx. apply mem_node_In, H3. pick 1. exact Hx.
  - unfold check_cu0_called_cannot_change. apply forallb_forall.
    intros [[x f] cw] Hin. cbv beta iota.
    gd (field_eqb f (Field_call_witness Known_arity_code_pointer)) E1.
    apply forallb_forall. intros [[cw' f'] c] Hin'. cbv beta iota.
    gd (node_eqb cw cw') E2. gd (field_eqb f' Field_code_id_of_call_witness) E3.
    gd (mem_node c cannot_change) E4.
    apply node_eqb_eq in E2. subst cw'.
    apply mem_node_In, H3. pick 2. apply in_flat_map.
    exists (x, f, cw). split; [ exact Hin | ].
    assert (Hex : witness_cannot_change F cannot_change cw = true)
      by (apply existsb_exists; exists (cw, f', c); split; [ exact Hin' | ];
          rewrite node_eqb_refl, E3, E4; reflexivity).
    fin.
  - unfold check_cu0_called. destruct lambda_lifting eqn:EL;
      [ reflexivity | ].
    cbn [orb]. apply forallb_forall.
    intros [[x f] cw] Hin. cbv beta iota.
    gd (field_eqb f (Field_call_witness Known_arity_code_pointer)) E1.
    gd (mem_node cw (t_has_usage T)) E2.
    apply mem_node_In, H3. pick 3. cbv iota. apply in_flat_map.
    exists (x, f, cw). split; [ exact Hin | ]. fin.
  - unfold check_cu0_result. apply forallb_forall.
    intros [a al] Hs. cbv beta iota.
    gd (mem_node al (t_has_source T)) E1.
    apply forallb_forall. intros [[cw r] a'] Hc. cbv beta iota.
    gd (node_eqb a a') E2. gd (is_return_of_call r) E3.
    apply forallb_forall. intros [[cw' f'] c] Hc'. cbv beta iota.
    gd (node_eqb cw cw') E4. gd (field_eqb f' Field_code_id_of_call_witness) E5.
    gd (mem_node c cannot_change) E6.
    apply node_eqb_eq in E2, E4. subst a' cw'.
    apply mem_node_In, H3. pick 4. apply in_flat_map.
    exists (a, al). split; [ exact Hs | ].
    assert (Hw : witness_cannot_change F cannot_change cw = true)
      by (apply existsb_exists; exists (cw, f', c); split; [ exact Hc' | ];
          rewrite node_eqb_refl, E5, E6; reflexivity).
    assert (Hex : existsb (fun '(call_witness, relation, alias') =>
                             node_eqb a alias' && is_return_of_call relation
                             && witness_cannot_change F cannot_change
                                  call_witness)
                    (f_constructor F) = true)
      by (apply existsb_exists; exists (cw, r, a); split; [ exact Hc | ];
          rewrite node_eqb_refl, E3, Hw; reflexivity).
    fin.
  - unfold check_cu0_parameter. apply forallb_forall.
    intros [a al] Hs. cbv beta iota.
    gd (mem_node al (t_has_source T)) E1.
    apply forallb_forall. intros [[cw i] a'] Hp. cbv beta iota.
    gd (node_eqb a a') E2.
    apply forallb_forall. intros [[cw' f'] c] Hc'. cbv beta iota.
    gd (node_eqb cw cw') E4. gd (field_eqb f' Field_code_id_of_call_witness) E5.
    gd (mem_node c cannot_change) E6.
    apply node_eqb_eq in E2, E4. subst a' cw'.
    apply mem_node_In, H3. pick 5. apply in_flat_map.
    exists (a, al). split; [ exact Hs | ].
    assert (Hw : witness_cannot_change F cannot_change cw = true)
      by (apply existsb_exists; exists (cw, f', c); split; [ exact Hc' | ];
          rewrite node_eqb_refl, E5, E6; reflexivity).
    assert (Hex : existsb (fun '(call_witness, _, alias') =>
                             node_eqb a alias'
                             && witness_cannot_change F cannot_change
                                  call_witness)
                    (f_parameter F) = true)
      by (apply existsb_exists; exists (cw, i, a); split; [ exact Hp | ];
          rewrite node_eqb_refl, Hw; reflexivity).
    fin.
  - unfold check_cu0_symbol. apply forallb_forall.
    intros [x y] Hs. cbv beta iota.
    gd (is_symbol_node x) E1.
    apply mem_node_In, H3. pick 6. apply in_flat_map.
    exists (x, y). split; [ exact Hs | ]. fin.
  - unfold check_cu0_stored. apply forallb_forall.
    intros [a al] Hs. apply forallb_forall. intros [[t r] a'] Hc.
    cbv beta iota.
    gd (node_eqb a a') E1. gd (mem_node_field t r (t_field_used T)) E2.
    gd (mem_node t (ut_crr R)) E3. gd (is_real_field r) E4.
    gd (mem_node t (ut_cu0 R)) E5.
    destruct (mem_node al (ut_cu0 R)) eqn:E6; [ reflexivity | exfalso ].
    apply node_eqb_eq in E1. subst a'.
    assert (Hin : al ∈ stored_round F T (ut_crr R) (ut_cu0 R)).
    { unfold stored_round. apply in_flat_map.
      exists (a, al). split; [ exact Hs | ].
      assert (Hex : existsb (fun '(to_, relation, alias') =>
                               node_eqb a alias'
                               && mem_node_field to_ relation
                                    (t_field_used T)
                               && mem_node to_ (ut_crr R)
                               && is_real_field relation
                               && mem_node to_ (ut_cu0 R))
                      (f_constructor F) = true)
        by (apply existsb_exists; exists (t, r, a); split; [ exact Hc | ];
            rewrite node_eqb_refl, E2, E3, E4, E5; reflexivity).
      fin. }
    rewrite H4 in Hin. destruct Hin.
  - unfold check_cu0_max_unbox_size. apply forallb_forall.
    intros [x f] Hin. cbv beta iota.
    gd (Nat.ltb max_unbox_size (length (used_fields T x))) E1.
    apply mem_node_In, H3. pick 7. apply filter_In. split; [ | exact E1 ].
    apply in_dedup_iff. apply in_map_iff. exists (x, f). auto.
  - unfold check_cu_cu0. apply forallb_forall. intros x Hx.
    apply mem_node_In, H5a. exact Hx.
  - unfold check_cu_function_slot. apply forallb_forall.
    intros [[x f] y] Hin. cbv beta iota.
    gd (mem_node x (ut_cu0 R)) E1. gd (is_function_slot f) E2.
    apply mem_node_In, H5b. unfold by_function_slots. apply in_flat_map.
    exists (x, f, y). split; [ exact Hin | ]. fin.
Qed.

End Closed.

(* ======================================================================== *)
(* The rounds of [cannot_unbox0] reach the fixpoint                         *)
(* ======================================================================== *)

Lemma filter_length_le {A : Type} (f : A -> bool) (l : list A) :
  length (filter f l) <= length l.
Proof.
  induction l as [ | x l IH]; cbn; [ lia | ]. destruct (f x); cbn; lia.
Qed.

(* A filter with a stronger test that drops one more element is shorter. *)
Lemma filter_length_lt {A : Type} (p q : A -> bool) (l : list A) :
  (∀ x, q x = true -> p x = true) ->
  (∃ x, x ∈ l ∧ p x = true ∧ q x = false) ->
  length (filter q l) < length (filter p l).
Proof.
  intros Hqp. induction l as [ | y l IH]; intros (x & Hx & Hp & Hq);
    [ destruct Hx | ].
  assert (Hle : length (filter q l) <= length (filter p l)).
  { clear IH Hx. induction l as [ | z l IHl]; cbn; [ lia | ].
    destruct (q z) eqn:Eq; [ rewrite (Hqp z Eq) | destruct (p z) ]; cbn;
      lia. }
  destruct Hx as [<- | Hx]; cbn.
  - rewrite Hp, Hq. cbn. lia.
  - assert (H : length (filter q l) < length (filter p l))
      by (apply IH; exists x; auto).
    destruct (q y) eqn:Eq; [ rewrite (Hqp y Eq) | destruct (p y) ]; cbn;
      lia.
Qed.

Section Rounds.

Variable F : facts.
Variable T : tables.
Variable crr : list node.

(* The allocations of the sources not in [cu0]. *)
Definition missing (cu0 : list node) : list node :=
  filter (fun y => negb (mem_node y cu0)) (map snd (t_sources T)).

(* A round that adds something adds an allocation that was missing, so
   that more rounds than there are missing allocations end in one that
   adds nothing. *)
Lemma stored_rounds_fix (fuel : nat) (cu0 : list node) :
  length (missing cu0) < fuel ->
  stored_round F T crr (stored_rounds F T crr fuel cu0) = [].
Proof.
  revert cu0. induction fuel as [ | fuel IH]; intros cu0 Hlt; [ lia | ].
  cbn [stored_rounds].
  destruct (Keyed.dedup node_eqb (stored_round F T crr cu0)) as [ | y l]
    eqn:E.
  - pose proof (fun z => proj2 (in_dedup_iff (stored_round F T crr cu0) z))
      as Hd.
    rewrite E in Hd.
    destruct (stored_round F T crr cu0) as [ | z l'] eqn:E';
      [ reflexivity | ].
    exfalso. apply (Hd z). rewrite ?E'. left. reflexivity.
  - apply IH.
    assert (Hy : y ∈ stored_round F T crr cu0)
      by (apply (in_dedup_iff _ y); rewrite E; left; reflexivity).
    unfold stored_round in Hy. apply in_flat_map in Hy.
    destruct Hy as ([al y'] & Hs & Hy).
    destruct (negb (mem_node y' cu0) && _) eqn:Ec; [ | destruct Hy ].
    destruct Hy as [-> | []].
    apply andb_true_iff in Ec as [Ec _]. apply negb_true_iff in Ec.
    enough (length (missing ((y :: l) ++ cu0)) < length (missing cu0))
      by lia.
    apply filter_length_lt.
    + intros z Hz. apply negb_true_iff in Hz. apply negb_true_iff.
      destruct (mem_node z cu0) eqn:Ez; [ | reflexivity ].
      rewrite <- Hz. symmetry. apply mem_node_In. apply in_or_app. right.
      apply mem_node_In. exact Ez.
    + exists y. split; [ apply in_map_iff; exists (al, y); auto | ].
      rewrite Ec. split; [ reflexivity | ].
      apply negb_false_iff. apply mem_node_In. left. reflexivity.
Qed.

End Rounds.

Lemma missing_lt (T : tables) (cu0 : list node) :
  length (missing T cu0) < S (length (t_sources T)).
Proof.
  unfold missing. pose proof (filter_length_le (fun y => negb (mem_node y cu0))
                                (map snd (t_sources T))) as H.
  rewrite length_map in H. lia.
Qed.

(* ======================================================================== *)
(* The derived table of [to_unbox] passes its check                         *)
(* ======================================================================== *)

Lemma relations_with_closed (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (seeds : list node) :
  check_unboxing_closed is_local F T cannot_change max_unbox_size
    lambda_lifting
    (unboxing_relations_with is_local F T cannot_change max_unbox_size
       lambda_lifting seeds)
  = true.
Proof.
  apply unboxing_closed; unfold unboxing_relations_with, unboxing_relations;
    cbv zeta; cbn [ut_crr0 ut_crr1 ut_crr ut_cu0 ut_cu].
  - intros x Hx. apply in_dedup_iff. exact Hx.
  - intros x Hx. apply in_dedup_iff. apply in_or_app. left. exact Hx.
  - intros x Hx. apply in_dedup_iff. apply in_or_app. right. exact Hx.
  - intros x Hx. apply in_dedup_iff. apply in_or_app. left. exact Hx.
  - intros x f y Hin Hf. apply in_dedup_iff. apply in_or_app. right.
    apply in_flat_map. exists (x, f, y). split; [ exact Hin | ].
    cbv beta iota. rewrite Hf. left. reflexivity.
  - intros x Hx. apply FlambdaReaperLeastProofs.stored_rounds_incl.
    apply in_dedup_iff. apply in_or_app. right.
    apply FlambdaReaperLeastProofs.stored_rounds_incl.
    apply in_dedup_iff. exact Hx.
  - apply stored_rounds_fix. apply missing_lt.
  - intros x Hx. apply in_dedup_iff. apply in_or_app. left. exact Hx.
  - intros x Hx. apply in_dedup_iff. apply in_or_app. right. exact Hx.
Qed.

Lemma derive_to_unbox_with_checked (is_local : field -> bool) (F : facts)
    (T : tables) (cannot_change : list node) (max_unbox_size : nat)
    (lambda_lifting : bool) (seeds : list node) :
  check_to_unbox_with is_local F T cannot_change max_unbox_size
    lambda_lifting seeds
    (derive_to_unbox_with is_local F T cannot_change max_unbox_size
       lambda_lifting seeds)
  = true.
Proof.
  unfold check_to_unbox_with, derive_to_unbox_with. cbv zeta.
  apply andb_true_intro; split;
    [ apply andb_true_intro; split;
      [ apply andb_true_intro; split | ] | ].
  - apply relations_with_closed.
  - apply forallb_forall. intros x Hx. apply mem_node_In.
    unfold unboxing_relations_with. cbv zeta. cbn [ut_cu0].
    apply FlambdaReaperLeastProofs.stored_rounds_incl.
    apply in_dedup_iff. apply in_or_app. left. exact Hx.
  - apply forallb_forall. intros x Hx. apply filter_In in Hx as [Hx Hcu].
    rewrite Hcu. apply (proj1 (in_dedup_iff _ _)) in Hx.
    apply mem_node_In in Hx.
    rewrite Hx. reflexivity.
  - apply forallb_forall. intros x Hx.
    destruct (mem_node x (ut_cu (unboxing_relations_with is_local F T
                                    cannot_change max_unbox_size
                                    lambda_lifting seeds))) eqn:E;
      [ rewrite orb_true_r; reflexivity | ].
    rewrite orb_false_r.
    replace (mem_node x (filter _ _)) with true; [ apply orb_true_r | ].
    symmetry. apply mem_node_In. apply filter_In. split.
    + apply in_dedup_iff. exact Hx.
    + rewrite E. reflexivity.
Qed.

Theorem derive_to_unbox_fixed_checked :
  derive_to_unbox_fixed_checked_statement.
Proof.
  intros is_local F T cannot_change max_unbox_size lambda_lifting.
  unfold check_to_unbox_fixed, derive_to_unbox_fixed.
  apply derive_to_unbox_with_checked.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem arm_needs_no_source_auto_derived :
  arm_needs_no_source_auto_derived_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting F code_deps
    cannot_change to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount Hkinds.
  assert (Hunbox :
            check_to_unbox_fixed is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox
            = true)
    by (unfold to_unbox; apply derive_to_unbox_fixed_checked).
  exact (FlambdaReaperWrapperAutoProofs.arm_needs_no_source_auto_end_to_end
           is_local u e T max_unbox_size lambda_lifting to_unbox Hloc Hslots
           Hfrag Hleast Hunbox Hcount Hkinds).
Qed.

Theorem kept_result_never_unboxed_auto_derived :
  kept_result_never_unboxed_auto_derived_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting F code_deps
    cannot_change to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount Hkinds.
  assert (Hunbox :
            check_to_unbox_fixed is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox
            = true)
    by (unfold to_unbox; apply derive_to_unbox_fixed_checked).
  exact (FlambdaReaperWrapperAutoProofs
           .kept_result_never_unboxed_auto_end_to_end
           is_local u e T max_unbox_size lambda_lifting to_unbox Hloc Hslots
           Hfrag Hleast Hunbox Hcount Hkinds).
Qed.

Theorem indirect_arm_needs_no_source_auto_derived :
  indirect_arm_needs_no_source_auto_derived_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting F code_deps
    cannot_change to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount Hkinds.
  assert (Hunbox :
            check_to_unbox_fixed is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox
            = true)
    by (unfold to_unbox; apply derive_to_unbox_fixed_checked).
  exact (FlambdaReaperWrapperUnknownProofs
           .indirect_arm_needs_no_source_auto_end_to_end
           is_local u e T max_unbox_size lambda_lifting to_unbox Hloc Hslots
           Hfrag Hleast Hunbox Hcount Hkinds).
Qed.

Theorem unboxed_callee_fails : unboxed_callee_fails_statement.
Proof.
  hnf. split.
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  repeat match goal with |- _ ∧ _ => split end;
  lazymatch goal with
  | |- behaves _ _ =>
      exists 40%nat; exists FlambdaValueSpec.no_reads; vm_compute;
      reflexivity
  | |- _ => vm_compute; reflexivity
  end.
Qed.

(* A position at which the fix stops in [make_apply_wrapper]. *)
Lemma wrapper_fails_pos (ads : list decision) :
  ∀ func : list decision,
    wrapper_fails_u ads func = true ->
    ∃ (i : nat) (fields : unboxed_fields) (fd : decision),
      nth_error ads i = Some (Unbox fields) ∧ nth_error func i = Some fd
      ∧ arm_pair (Unbox fields) fd = true.
Proof.
  induction ads as [ | ad ads IH]; intros [ | fd func] H; cbn in H;
    try discriminate.
  destruct ad as [d | fields | k], fd as [d' | fields' | k'];
    try (exists 0, fields; eexists;
         split; [ reflexivity | split; [ reflexivity | reflexivity ] ]);
    destruct (IH func H) as (i & fs & fd0 & H1 & H2 & H3);
    exists (S i), fs, fd0;
    (split; [ exact H1 | split; [ exact H2 | exact H3 ] ]).
Qed.

Theorem fix_arm_never_returns_auto_derived :
  fix_arm_never_returns_auto_derived_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting F code_deps
    cannot_change to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount Hkinds ads
    ap Hin Hkind Har Hfails.
  destruct (wrapper_fails_pos ads _ Hfails)
    as (i & fields & fd & Hads & Hfd & Harm).
  rewrite nth_error_map in Hfd.
  destruct (nth_error
              (unit_callee_decisions F T cannot_change code_deps unboxed e ap)
              i)
    as [[fd' kind] | ] eqn:E; [ | discriminate ].
  cbn in Hfd. injection Hfd as ->.
  assert (Hunbox :
            check_to_unbox_fixed is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox
            = true)
    by (unfold to_unbox; apply derive_to_unbox_fixed_checked).
  pose proof (proj1 (FlambdaReaperCallingConventionsProofs
                       .derive_cannot_change_least F T)) as Hcc.
  pose proof (FlambdaReaperWrapperAutoProofs.derive_dominated_checked T)
    as Hdom.
  pose proof (FlambdaReaperLeastProofs.least_apply_widgets is_local F T Hloc
                Hleast
                (FlambdaReaperLeastProofs.traverse_widgets_read_only u e))
    as Hwid.
  pose proof (FlambdaReaperWrapperShapeProofs.traverse_used_code_ids Auto
                is_local u e T Hloc Hleast) as Hcodes.
  pose proof (FlambdaReaperWrapperShapeProofs.traverse_argument_reads Auto u
                e) as Hargs.
  pose proof (FlambdaReaperWrapperAutoProofs.traverse_known_witnesses is_local
                u e T Hloc Hleast) as Hknown.
  destruct (ap_call_kind ap) eqn:Hk.
  1, 3:
    refine (FlambdaReaperWrapperProofs.fix_arm_never_returns Auto is_local u e
              T cannot_change max_unbox_size lambda_lifting (derive_dominated T)
              to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom Hunbox Hwid Hcodes
              Hargs Hknown Hcount Hkinds ads ap i fields fd kind Hin _ Har E
              Hads Harm);
    intros Hc; rewrite Hk in Hc; discriminate Hc.
  destruct Hkind as [H | Hlen]; [ exfalso; exact (H eq_refl) | ].
  exact (FlambdaReaperWrapperProofs.indirect_fix_arm_never_returns Auto
           is_local u e T cannot_change max_unbox_size lambda_lifting
           (derive_dominated T) to_unbox Hloc Hslots Hfrag Hleast Hcc Hdom
           Hunbox Hwid Hcodes Hargs Hknown
           (FlambdaReaperWrapperUnknownProofs.traverse_unknown_widgets Auto
              is_local u e T Hloc Hleast)
           (FlambdaReaperWrapperUnknownProofs.traverse_unknown_witnesses Auto
              is_local u e T Hloc Hleast)
           Hcount Hkinds ads ap i fields fd kind Hin Hk Hlen E Hads Harm).
Qed.
