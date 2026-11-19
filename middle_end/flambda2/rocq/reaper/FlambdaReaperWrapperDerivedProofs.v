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
  FlambdaReaperWrapperAutoProofs FlambdaReaperWrapperUnknownProofs
  FlambdaReaperWrapperProofs FlambdaReaperWrapperShapeProofs
  FlambdaReaperFlowProofs FlambdaReaperRebuildProofs FlambdaMapProofs.
From OxCamlFlambda2 Require Import FlambdaBindersSpec.
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

Theorem fix_arm_never_returns_auto_derived_full :
  fix_arm_never_returns_auto_derived_full_statement.
Proof.
  intros is_local u e T max_unbox_size lambda_lifting F code_deps
    cannot_change to_unbox unboxed Hloc Hslots Hfrag Hleast Hcount Hkinds ads
    ap Hin Hnil Har Hfails.
  destruct (ap_call_kind ap) eqn:Hk.
  1, 3:
    refine (fix_arm_never_returns_auto_derived is_local u e T max_unbox_size
              lambda_lifting Hloc Hslots Hfrag Hleast Hcount Hkinds ads ap Hin
              _ Har Hfails);
    left; rewrite Hk; discriminate.
  pose proof (Hnil eq_refl) as Hne.
  destruct (wrapper_fails_pos ads _ Hfails)
    as (i & fields & fd & Hads & Hfd & Harm).
  rewrite nth_error_map in Hfd.
  destruct (nth_error
              (unit_callee_decisions F T cannot_change code_deps unboxed e ap)
              i)
    as [[fd' kind] | ] eqn:E; [ | discriminate ].
  cbn in Hfd. injection Hfd as ->.
  rewrite <- (FlambdaReaperWrapperProofs.decision_at_unbox ads i fields kind
                Hads)
    in Harm.
  assert (Hunbox :
            check_to_unbox_fixed is_local F T cannot_change max_unbox_size
              lambda_lifting to_unbox
            = true)
    by (unfold to_unbox; apply derive_to_unbox_fixed_checked).
  exact (FlambdaReaperWrapperProofs.arm_chain_unit Auto is_local u e T
           cannot_change max_unbox_size lambda_lifting (derive_dominated T)
           to_unbox Hloc Hslots Hfrag Hleast
           (proj1 (FlambdaReaperCallingConventionsProofs
                     .derive_cannot_change_least F T))
           (FlambdaReaperWrapperAutoProofs.derive_dominated_checked T) Hunbox
           (FlambdaReaperWrapperShapeProofs.traverse_used_code_ids Auto
              is_local u e T Hloc Hleast)
           Hcount Hkinds ads ap Hin
           (FlambdaReaperWrapperUnknownProofs.traverse_unknown_witnesses Auto
              is_local u e T Hloc Hleast)
           i fd kind Hk Hne E Harm).
Qed.

(* ------------------------------------------------------------------------ *)
(* [check_never_returns]                                                    *)
(* ------------------------------------------------------------------------ *)

(* The rebuilt body of used code with a result of no source has its return
   continuation free nowhere: an [Apply_cont] to it is [Invalid], a call
   returning to it never returns, and the rest of the rebuild keeps the
   free continuations of its parts. *)
Module CheckNeverReturns.

Abbreviation fc := free_continuations_of_expr.

Lemma fc_fold_left {A : Type} (f : expr -> A -> expr) (l : list A) :
  (∀ b a, fc (f b a) = fc b) -> ∀ b : expr, fc (List.fold_left f l b) = fc b.
Proof.
  intros Hf. induction l as [ | a l IH]; intros b; [ reflexivity | ].
  cbn [List.fold_left]. rewrite IH. apply Hf.
Qed.

Lemma fold_none {A : Type} (g : option expr -> A -> option expr)
    (l : list A) :
  (∀ a, g None a = None) -> List.fold_left g l None = None.
Proof.
  intros Hg. induction l as [ | a l IH]; [ reflexivity | ].
  cbn [List.fold_left]. rewrite Hg. exact IH.
Qed.

Lemma fc_fold_opt {A : Type} (g : option expr -> A -> option expr)
    (l : list A) :
  (∀ a, g None a = None) ->
  (∀ b a b', g (Some b) a = Some b' -> fc b' = fc b) ->
  ∀ b b' : expr, List.fold_left g l (Some b) = Some b' -> fc b' = fc b.
Proof.
  intros Hn Hs. induction l as [ | a l IH]; intros b b' H.
  - cbn in H. injection H as ->. reflexivity.
  - cbn [List.fold_left] in H. destruct (g (Some b) a) as [b1 | ] eqn:E.
    + rewrite (IH b1 b' H). exact (Hs b a b1 E).
    + rewrite (fold_none g l Hn) in H. discriminate.
Qed.

Section Lets.

Variable sol : solution.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).

Lemma fc_bind_fields (u u' : unboxed_field) (body b : expr) :
  bind_fields u u' body = Some b -> fc b = fc body.
Proof.
  unfold bind_fields. destruct (unboxed_field_pairs Value u u');
    [ | discriminate ].
  intros [= <-]. apply fc_fold_left. intros b0 [[x x'] kind]. reflexivity.
Qed.

Lemma fc_bind_simple (u : unboxed_field) (kind : flambda_kind) (s : simple)
    (body b : expr) :
  bind_field_to_simple u kind s body = Some b -> fc b = fc body.
Proof.
  unfold bind_field_to_simple. destruct u; [ intros [= <-]; reflexivity | ].
  discriminate.
Qed.

Lemma fc_unboxed_binding (en : rebuild_env) (tb : unboxed_fields)
    (d : named) (body : expr) :
  incl (fc (rebuild_unboxed_binding unboxed changed en tb d body)) (fc body).
Proof.
  assert (H : ∀ o, (∀ b, o = Some b -> fc b = fc body) ->
            incl (fc (match o with Some e => e | None => Invalid end))
              (fc body)).
  { intros [b | ] Hb; [ rewrite (Hb b eq_refl); apply incl_refl | ].
    intros k []. }
  unfold rebuild_unboxed_binding. apply H. intros b Hb.
  assert (Hblock : ∀ en tag args b,
            bind_block_fields unboxed en tag args tb body = Some b ->
            fc b = fc body).
  { intros en' tag args b'. unfold bind_block_fields.
    apply fc_fold_opt; [ intros [f u]; reflexivity | ].
    intros b0 [f u] b1 E. destruct f; try discriminate;
      try (exact (fc_bind_simple _ _ _ _ _ E)).
    destruct (unboxed_of unboxed _);
      [ exact (fc_bind_fields _ _ _ _ E)
      | exact (fc_bind_simple _ _ _ _ _ E) ]. }
  assert (Hload : ∀ f arg b,
            load_unboxed_field unboxed changed f arg tb body = Some b ->
            fc b = fc body).
  { intros f arg b'. unfold load_unboxed_field.
    destruct (unboxed_of unboxed arg).
    - destruct (Keyed.lookup _ _ _); [ apply fc_bind_fields | discriminate ].
    - destruct (changed_of changed arg) as [repr | ]; [ | discriminate ].
      destruct f; try discriminate.
      destruct (Keyed.lookup _ _ _) as [u | ]; [ | discriminate ].
      destruct (unboxed_field_pairs _ _ _); [ | discriminate ].
      intros [= <-]. apply fc_fold_left. intros b0 [[x slot] kind].
      reflexivity. }
  assert (Hbox : ∀ bn contents b,
            bind_boxed_number_field bn contents tb body = Some b ->
            fc b = fc body).
  { intros bn contents b'. unfold bind_boxed_number_field.
    apply fc_fold_opt; [ intros [f u]; reflexivity | ].
    intros b0 [f u] b1 E. destruct f; try discriminate.
    destruct (Nat.eqb _ _); [ exact (fc_bind_simple _ _ _ _ _ E) | ].
    discriminate. }
  repeat match type of Hb with
  | context [match ?x with _ => _ end] => destruct x
  end;
  first [ discriminate | injection Hb as <-; reflexivity
        | exact (Hload _ _ _ Hb) | exact (Hblock _ _ _ _ Hb)
        | exact (Hbox _ _ _ Hb) | exact (fc_bind_fields _ _ _ _ Hb) ].
Qed.

Lemma fc_let_u (en : rebuild_env) (x : variable) (kind : flambda_kind)
    (d : named) (body : expr) :
  incl (fc (rebuild_let_u sol unboxed changed en x kind d body)) (fc body).
Proof.
  unfold rebuild_let_u.
  repeat match goal with
  | |- context [match ?x with _ => _ end] => destruct x
  end;
  first [ apply incl_refl | intros k [] ].
Qed.

Lemma fc_changed_binding (en : rebuild_env) (repr : closure_repr)
    (x : variable) (kind : flambda_kind) (d : named) (body : expr) :
  incl (fc (rebuild_changed_binding sol unboxed changed en repr x kind d
              body))
    (fc body).
Proof.
  unfold rebuild_changed_binding.
  destruct d; try apply fc_let_u. apply incl_refl.
Qed.

Lemma fc_unboxed_set (bound : list variable)
    (vs : list (value_slot * simple)) (body : expr) :
  incl (fc (rebuild_unboxed_set sol unboxed bound vs body)) (fc body).
Proof.
  unfold rebuild_unboxed_set.
  destruct (List.fold_left _ bound (Some body)) as [b | ] eqn:E;
    [ | intros k [] ].
  enough (Hb : fc b = fc body) by (rewrite Hb; apply incl_refl).
  revert E. apply fc_fold_opt.
  - intros x. reflexivity.
  - intros b0 x b1 E1.
    destruct (is_var_used sol x Value);
      [ | injection E1 as <-; reflexivity ].
    destruct (unboxed_of_node unboxed (Node_variable x)) as [tb | ];
      [ | discriminate ].
    unfold bind_closure_fields in E1. revert E1.
    apply fc_fold_opt; [ intros [f u]; reflexivity | ].
    intros b2 [f u] b3 E2. destruct f; try discriminate.
    destruct (lookup _ vs) as [arg | ]; [ | discriminate ].
    destruct (unboxed_of unboxed arg);
      [ exact (fc_bind_fields _ _ _ _ E2)
      | exact (fc_bind_simple _ _ _ _ _ E2) ].
Qed.

End Lets.

Lemma in_without_inv (a : positive) (bound names : list positive) :
  a ∈ without bound names -> a ∈ names ∧ a ∉ bound.
Proof.
  unfold without. rewrite filter_In, negb_true_iff. intros [Hin Hb].
  split; [ exact Hin | ]. intros Ha.
  apply FlambdaMapProofs.among_In in Ha. congruence.
Qed.

Lemma fc_wrapper (ff : positive) (en : unbox_env) (ap : apply_expr)
    (fds : list (decision * flambda_kind)) (nr : bool) (k : continuation) :
  k ∈ fc (make_apply_wrapper_u true ff en ap fds nr) ->
  k = ap_exn_continuation ap ∨ (k = ap_continuation ap ∧ nr = false).
Proof.
  unfold make_apply_wrapper_u. cbn [andb].
  destruct nr;
    [ | destruct (wrapper_args_u _ _ _) as [args | ];
        [ destruct (args_are_params _ _) | ] ];
    cbn [free_continuations_of_expr free_continuations_of_handler
         apply_cont_continuations ac_continuation ac_trap_action
         with_continuation ap_continuation ap_exn_continuation];
    intros H;
    repeat match goal with
    | H : _ ∈ _ ++ _ |- _ => apply in_app_or in H as [H | H]
    | H : _ ∈ without _ _ |- _ =>
      apply in_without_inv in H as [H H']
    | H : _ ∈ [] |- _ => destruct H
    | H : _ ∈ _ :: _ |- _ => destruct H as [H | H]
    | H : _ ∈ apply_cont_continuations _ |- _ =>
      unfold apply_cont_continuations in H;
      cbn [ac_continuation ac_trap_action] in H
    end;
    subst;
    first [ left; reflexivity | right; split; reflexivity
          | exfalso; apply H'; left; reflexivity ].
Qed.

Section Rebuilt.

Variable F : facts.
Variable sol : solution.
Variable cc : node -> bool.
Variable cds : finite_map code_dep.
Variable pdc : bool.
Variable uci : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable ff : positive.
Variable changed : list (node * closure_repr).

Lemma fc_apply (en : unbox_env) (ap : apply_expr) (k : continuation) :
  k ∈ fc (rebuild_apply_u true F sol cc cds pdc uci codes unboxed ff en ap) ->
  k = ap_exn_continuation ap
  ∨ (k = ap_continuation ap ∧ apply_never_returns F sol cds codes ap = false).
Proof.
  unfold rebuild_apply_u. intros H.
  repeat match type of H with
  | context [match ?x with _ => _ end] => destruct x
  end;
  first [ destruct H
        | apply fc_wrapper in H; cbn [ap_continuation ap_exn_continuation] in H;
          exact H ].
Qed.

Lemma fc_apply_cont (en : unbox_env) (ac : apply_cont) (k : continuation) :
  k ∈ fc (rebuild_apply_cont_u sol unboxed en ac) ->
  k ∈ apply_cont_continuations ac
  ∧ (k = ac_continuation ac ->
     existsb (simple_has_no_source sol) (ac_args ac) = false).
Proof.
  unfold rebuild_apply_cont_u, rewrite_apply_cont_u.
  destruct (existsb _ _) eqn:E; [ intros [] | ].
  intros H.
  split; [ | intros _; reflexivity ].
  assert (Hg : ∀ o : option (list simple),
             k ∈ fc (match
                       match o with
                       | Some args =>
                         Some {| ac_continuation := ac_continuation ac;
                                 ac_args := args;
                                 ac_trap_action := ac_trap_action ac |}
                       | None => None
                       end
                     with
                     | Some ac => Apply_cont ac
                     | None => Invalid
                     end) ->
             k ∈ apply_cont_continuations ac).
  { intros [args | ] H0; [ exact H0 | destruct H0 ]. }
  exact (Hg _ H).
Qed.

Lemma fc_switch (en : unbox_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (k : continuation) :
  k ∈ fc (rebuild_switch_u sol unboxed en scrutinee arms) ->
  ∃ d ac, (d, ac) ∈ Keyed.compact Z.eqb arms
    ∧ k ∈ apply_cont_continuations ac
    ∧ (k = ac_continuation ac ->
       existsb (simple_has_no_source sol) (ac_args ac) = false).
Proof.
  unfold rebuild_switch_u.
  destruct (Keyed.is_empty _); [ intros [] | ].
  cbn [free_continuations_of_expr]. intros H.
  apply in_flat_map in H as ([d ac'] & Hin & H). cbn [snd] in H.
  apply in_flat_map in Hin as ([d' ac] & Hin & Hd).
  destruct (rewrite_apply_cont_u sol unboxed en ac) as [ac0 | ] eqn:E;
    [ | destruct Hd ].
  destruct Hd as [[= <- <-] | []].
  exists d', ac. split; [ exact Hin | ].
  assert (Hr : k ∈ fc (rebuild_apply_cont_u sol unboxed en ac)).
  { unfold rebuild_apply_cont_u. rewrite E. exact H. }
  exact (fc_apply_cont en ac k Hr).
Qed.

End Rebuilt.

Lemma combine_nth_some {A B : Type} (l : list A) (l' : list B) (i : nat)
    (a : A) (b : B) :
  nth_error l i = Some a -> nth_error l' i = Some b ->
  nth_error (combine l l') i = Some (a, b).
Proof.
  revert l' i. induction l as [ | x l IH]; intros [ | y l'] [ | i]; cbn;
    try discriminate; auto.
  intros [= ->] [= ->]. reflexivity.
Qed.

Lemma call_groups_length (ap : apply_expr) :
  ap_args_arity ap <> [] ->
  length (FlambdaReaperFlowProofs.call_groups ap) = length (ap_args_arity ap).
Proof.
  intros H. unfold FlambdaReaperFlowProofs.call_groups, group_by_parameter.
  pose proof (FlambdaReaperWrapperProofs.group_by_sizes_length
                (map (@length flambda_kind)
                   (unarize_per_parameter (ap_args_arity ap)))
                (ap_args ap)) as E.
  rewrite length_map in E. unfold unarize_per_parameter in E.
  rewrite length_map in E.
  destruct (group_by_sizes _ _) as [ | g gs]; cbn [at_least_one_group];
    [ | exact E ].
  cbn in E. destruct (ap_args_arity ap); [ congruence | discriminate ].
Qed.

Section Flow.

Variable is_local : field -> bool.
Variable u : unit_interface.
Variable e0 : expr.
Variable T : tables.

Abbreviation F := (traverse_unit Auto u e0).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation cds := (unit_code_deps Auto u e0).

Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local F T.
Variable slots_read : Prop.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hnodup : NoDup (map fst cds).
Hypothesis Hids :
  ∀ id, id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e0.

Abbreviation code_edges :=
  (FlambdaReaperFlowProofs.code_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).

Hypothesis Hcode_edges : ∀ id c, (id, c) ∈ unit_codes e0 -> code_edges id c.

Lemma closedU : closed is_local G S.
Proof. exact (FlambdaReaperWrapperProofs.closedS is_local F T Hleast). Qed.

Lemma ret_nl (rk : return_kind) : is_local (Field_return_of_call rk) = false.
Proof.
  exact (FlambdaReaperWrapperProofs.return_not_local is_local rk Hloc).
Qed.

Lemma wit_nl (ep : closure_entry_point) :
  is_local (Field_call_witness ep) = false.
Proof.
  exact (FlambdaReaperWrapperProofs.witness_not_local is_local Hloc ep).
Qed.

(* An argument of an [Apply_cont] to a continuation one of whose
   parameters has no source has none. *)
Lemma ac_no_source (ce : finite_map (list node)) (ac : apply_cont)
    (ns : list node) (r : node) :
  FlambdaReaperFlowProofs.apply_cont_edges G noted ce ac ->
  count_ok (FlambdaReaperFlowProofs.counts ce) (ac_continuation ac)
    (length (ac_args ac))
  = true ->
  lookup (ac_continuation ac) ce = Some ns -> r ∈ ns ->
  s_has_source S r = false ->
  existsb (simple_has_no_source S) (ac_args ac) = true.
Proof.
  intros [Hal _] Hc Hk Hr Hno.
  unfold count_ok in Hc.
  rewrite FlambdaReaperWrapperProofs.lookup_counts, Hk in Hc.
  cbn [option_map] in Hc. apply Nat.eqb_eq in Hc.
  apply In_nth_error in Hr as [i Hi].
  destruct (nth_error (ac_args ac) i) as [s | ] eqn:Hs.
  2:{ apply nth_error_None in Hs.
      assert (i < length ns) by (apply nth_error_Some; congruence). lia. }
  assert (Hin : (r, s) ∈ combine ns (ac_args ac)).
  { apply (nth_error_In _ i). exact (combine_nth_some _ _ i r s Hi Hs). }
  pose proof (Hal ns Hk r s Hin) as Ha.
  apply existsb_exists. exists s. split; [ exact (nth_error_In _ _ Hs) | ].
  assert (Hs0 : s_has_source S (node_of_simple s) = false).
  { destruct (s_has_source S (node_of_simple s)) eqn:E; [ | reflexivity ].
    rewrite (has_source_alias _ _ _ closedU _ _ E
               (alias_input _ _ _ closedU _ _ Ha)) in Hno.
    discriminate. }
  destruct s as [x | sym | c]; cbn [simple_has_no_source node_of_simple] in *;
    try (rewrite Hs0; reflexivity).
  rewrite (has_source_any _ _ _ closedU _
             (any_source_input _ _ _ closedU _ Hconsts)) in Hs0.
  discriminate.
Qed.

(* The witnesses of a call of unknown arity whose result of rank [i] has no
   source have no source of any source. *)
Lemma chain_no_any (returns : list node) (exn called : node) (i : nat)
    (r : node) :
  nth_error returns i = Some r -> s_has_source S r = false ->
  ∀ l first w,
    FlambdaReaperFlowProofs.caller_chain_edges G returns exn called
      ((first, w) :: l) ->
    s_any_source S w = false.
Proof.
  intros Hi Hno l. induction l as [ | [f' w'] l IH]; intros first w Hch.
  - destruct Hch as (_ & _ & _ & Hres & _).
    destruct (s_any_source S w) eqn:E; [ | reflexivity ].
    pose proof (any_source_accessor _ _ _ closedU w _ r E (Hres i r Hi)
                  (ret_nl _)) as Hr.
    rewrite (has_source_any _ _ _ closedU r Hr) in Hno. discriminate.
  - destruct Hch as (_ & _ & _ & (v & Hv & Hw') & Hrest).
    pose proof (IH f' w' Hrest) as Hw'0.
    destruct (s_any_source S w) eqn:E; [ | reflexivity ].
    pose proof (any_source_accessor _ _ _ closedU w _ v E Hv (ret_nl _))
      as Hav.
    rewrite (any_source_accessor _ _ _ closedU v _ w' Hav Hw' (wit_nl _))
      in Hw'0.
    discriminate.
Qed.

Definition approx (cs : allocation_sources) (base : node) : Prop :=
  match cs with
  | Any_source => s_any_source S base = true
  | Sources m => ∀ x, m x = true -> s_sources S base x = true
  end.

(* What [never_returns_of_call] reads along the chain of witnesses of a
   call of unknown arity is among the sources of those witnesses: a
   result of rank [i] with no source makes it [true]. *)
Lemma chain_never_returns (returns : list node) (exn called : node)
    (i : nat) (r : node) (nres : nat) :
  nth_error returns i = Some r -> s_has_source S r = false -> i < nres ->
  ∀ l first w (num : list nat) (n : nat) cs base a,
    FlambdaReaperFlowProofs.caller_chain_edges G returns exn called
      ((first, w) :: l) ->
    length num = length l ->
    approx cs base ->
    g_accessor G a (Field_call_witness Unknown_arity_code_pointer) base ->
    (s_any_source S a = true -> s_any_source S w = true) ->
    (∀ s, s_sources S a s = true -> s_any_source S w = false ->
          s_sources S w s = true) ->
    never_returns_of_call F S Unknown_arity_code_pointer cs (n :: num) nres
    = true.
Proof.
  intros Hi Hno Hlt l.
  induction l as [ | [f' w'] l IH];
    intros first w num n cs base a Hch Hlen Happ Ha Hany Hsrc;
    pose proof (chain_no_any returns exn called i r Hi Hno _ first w Hch)
      as Hw;
    (assert (Hasa : s_any_source S a = false)
       by (destruct (s_any_source S a); [ | reflexivity ];
           rewrite (Hany eq_refl) in Hw; discriminate));
    (assert (Hasb : s_any_source S base = false)
       by (destruct (s_any_source S base) eqn:E; [ | reflexivity ];
           rewrite (any_source_accessor _ _ _ closedU base _ a E Ha
                      (wit_nl _)) in Hasa;
           discriminate));
    (destruct cs as [ | m]; [ cbn [approx] in Happ; congruence | ]);
    cbn [never_returns_of_call]; unfold get_field_sources;
    (match goal with
     | |- context [if existsb ?p (f_constructor F) then _ else _] =>
       destruct (existsb p (f_constructor F)) eqn:Eany
     end;
     [ exfalso; apply existsb_exists in Eany as ([[x f] y] & Hc & Ht);
       cbn beta iota in Ht;
       apply andb_true_iff in Ht as [Ht Hy];
       apply andb_true_iff in Ht as [Hx Hf];
       apply field_eqb_eq in Hf; subst f;
       pose proof (alias_accessor_source _ _ _ closedU base x _ a y Ha
                     (Happ x Hx) Hasb Hc) as Hay;
       rewrite (any_source_alias _ _ _ closedU y a Hay Hy) in Hasa;
       discriminate | ]);
    cbn beta iota;
    (* The sources read are those of [w]. *)
    (assert (Hws :
               ∀ z,
                 existsb
                   (fun '(x, f', y) =>
                      m x && field_eqb f'
                               (Field_call_witness Unknown_arity_code_pointer)
                      && s_sources S y z && s_has_source S z)
                   (f_constructor F)
                 = true ->
                 s_sources S w z = true)
     by (intros z Hz; apply existsb_exists in Hz as ([[x f] y] & Hc & Ht);
         cbn beta iota in Ht;
         apply andb_true_iff in Ht as [Ht _];
         apply andb_true_iff in Ht as [Ht Hyz];
         apply andb_true_iff in Ht as [Hx Hf];
         apply field_eqb_eq in Hf; subst f;
         pose proof (alias_accessor_source _ _ _ closedU base x _ a y Ha
                       (Happ x Hx) Hasb Hc) as Hay;
         assert (Hasy : s_any_source S y = false)
           by (destruct (s_any_source S y) eqn:E; [ | reflexivity ];
               rewrite (any_source_alias _ _ _ closedU y a Hay E) in Hasa;
               discriminate);
         exact (Hsrc z (sources_alias _ _ _ closedU y a z Hyz Hasy Hay Hasa)
                  Hw))).
  - destruct num as [ | n' num']; [ | discriminate ].
    destruct Hch as (_ & _ & _ & Hres & _).
    apply existsb_exists. exists i. split; [ apply in_seq; lia | ].
    apply negb_true_iff. unfold result_has_source_of.
    apply Bool.not_true_iff_false. intros E.
    apply existsb_exists in E as ([[z f] y] & Hc & Ht).
    cbn beta iota in Ht.
    apply andb_true_iff in Ht as [Ht Hy].
    apply andb_true_iff in Ht as [Hz Hf].
    apply field_eqb_eq in Hf. subst f.
    pose proof (alias_accessor_source _ _ _ closedU w z _ r y (Hres i r Hi)
                  (Hws z Hz) Hw Hc) as Hry.
    rewrite (has_source_alias _ _ _ closedU y r Hy Hry) in Hno.
    discriminate.
  - destruct num as [ | n' num']; [ discriminate | ].
    cbn [length] in Hlen. injection Hlen as Hlen.
    destruct Hch as (_ & _ & _ & (v & Hv & Hw') & Hrest).
    pose proof (chain_no_any returns exn called i r Hi Hno _ f' w' Hrest)
      as Hw'0.
    assert (Hasv : s_any_source S v = false).
    { destruct (s_any_source S v) eqn:E; [ | reflexivity ].
      rewrite (any_source_accessor _ _ _ closedU v _ w' E Hw' (wit_nl _))
        in Hw'0.
      discriminate. }
    refine (IH f' w' num' n' _ v w' Hrest Hlen _ Hw' (fun H => H)
              (fun s H _ => H)).
    unfold get_field_sources.
    match goal with
    | |- context [if existsb ?p (f_constructor F) then _ else _] =>
      destruct (existsb p (f_constructor F)) eqn:Eany2
    end; cbn [approx].
    + apply existsb_exists in Eany2 as ([[z f] y] & Hc & Ht).
      cbn beta iota in Ht.
      apply andb_true_iff in Ht as [Ht Hy].
      apply andb_true_iff in Ht as [Hz Hf].
      apply field_eqb_eq in Hf. subst f.
      pose proof (alias_accessor_source _ _ _ closedU w z _ v y Hv
                    (Hws z Hz) Hw Hc) as Hvy.
      rewrite (any_source_alias _ _ _ closedU y v Hvy Hy) in Hasv.
      discriminate.
    + intros x Hx. apply existsb_exists in Hx as ([[z f] y] & Hc & Ht).
      cbn beta iota in Ht.
      apply andb_true_iff in Ht as [Ht _].
      apply andb_true_iff in Ht as [Ht Hyx].
      apply andb_true_iff in Ht as [Hz Hf].
      apply field_eqb_eq in Hf. subst f.
      pose proof (alias_accessor_source _ _ _ closedU w z _ v y Hv
                    (Hws z Hz) Hw Hc) as Hvy.
      assert (Hasy : s_any_source S y = false).
      { destruct (s_any_source S y) eqn:E; [ | reflexivity ].
        rewrite (any_source_alias _ _ _ closedU y v Hvy E) in Hasv.
        discriminate. }
      exact (sources_alias _ _ _ closedU y v x Hyx Hasy Hvy Hasv).
Qed.

(* What the fix needs of a call to prove that it never returns: one of
   unknown arity has a group of arguments, and one of code that is known
   expects of it no more results than its code dep has. *)
Definition returns_ok (ap : apply_expr) : Prop :=
  (ap_call_kind ap = Indirect_unknown_arity -> ap_args_arity ap <> [])
  ∧ ∀ (id : code_id) (cd : code_dep),
      code_id_actually_called F S ap = Some id -> (id, cd) ∈ cds ->
      length (unarize (ap_return_arity ap)) <= length (cd_return cd).

(* A call that returns to a continuation one of whose parameters has no
   source never returns. *)
Lemma call_never_returns (ctx : option code_id)
    (ce : finite_map (list node)) (ap : apply_expr) (ns : list node)
    (r : node) :
  FlambdaReaperFlowProofs.apply_edges G noted (unit_codes e0) ctx ce ap ->
  conts_counted (FlambdaReaperFlowProofs.counts ce) (Apply ap) = true ->
  FlambdaReaperWrapperProofs.live T ctx -> returns_ok ap ->
  lookup (ap_continuation ap) ce = Some ns -> r ∈ ns ->
  s_has_source S r = false ->
  apply_never_returns F S cds (bound_code_metadata e0) ap = true.
Proof.
  intros (_ & _ & _ & returns & exn & Hret & _ & Hcall) Hcount Hlive
    [Hargs Har] Hk Hr Hno.
  rewrite (Hret ns Hk) in Hcall.
  cbn [conts_counted] in Hcount. apply andb_true_iff in Hcount as [Hcount _].
  unfold count_ok in Hcount.
  rewrite FlambdaReaperWrapperProofs.lookup_counts, Hk in Hcount.
  cbn [option_map] in Hcount. apply Nat.eqb_eq in Hcount.
  apply In_nth_error in Hr as [i Hi].
  assert (Hlt : i < length (unarize (ap_return_arity ap)))
    by (rewrite <- Hcount; apply nth_error_Some; congruence).
  assert (Halias : ∀ w a,
            FlambdaReaperFlowProofs.cond_alias_edge G ctx w a ->
            s_alias S w a = true).
  { intros w a Hc. destruct ctx as [c | ]; cbn in Hc.
    - exact (alias_propagate _ _ _ closedU _ _ _ (Hlive c eq_refl) Hc).
    - exact (alias_input _ _ _ closedU _ _ Hc). }
  pose proof (FlambdaReaperLeastProofs.least_apply_widgets is_local F T Hloc
                Hleast
                (FlambdaReaperLeastProofs.traverse_widgets_read_only u e0))
    as Hwid.
  pose proof (FlambdaReaperWrapperAutoProofs.traverse_known_witnesses
                is_local u e0 T Hloc Hleast) as Hknown.
  destruct (ap_call_kind ap) eqn:Hkind.
  1, 3:
    (apply FlambdaReaperFlowProofs.known_call_edges_of in Hcall;
       [ | rewrite Hkind; discriminate ]);
    destruct Hcall as (w & a & called & _ & Hres & _ & _ & _ & Hcond & Hread);
    pose proof (Halias _ _ Hcond) as Hwa;
    pose proof (Hres i r Hi) as Hrw;
    (assert (Hasw : s_any_source S w = false)
       by (destruct (s_any_source S w) eqn:E; [ | reflexivity ];
           rewrite (has_source_any _ _ _ closedU r
                      (any_source_accessor _ _ _ closedU w _ r E Hrw
                         (ret_nl _))) in Hno;
           discriminate));
    (assert (Hasa : s_any_source S a = false)
       by (destruct (s_any_source S a) eqn:E; [ | reflexivity ];
           rewrite (any_source_alias _ _ _ closedU a w Hwa E) in Hasw;
           discriminate));
    (assert (Hkey : ∀ s y,
               s_sources S a s = true ->
               (s, Field_return_of_call (Normal i), y) ∈ f_constructor F ->
               s_has_source S y = false)
       by (intros s y Hs Hc; destruct (s_has_source S y) eqn:Ey;
           [ | reflexivity ];
           pose proof (sources_alias _ _ _ closedU a w s Hs Hasa Hwa Hasw)
             as Hsw;
           pose proof (alias_accessor_source _ _ _ closedU w s _ r y Hrw Hsw
                         Hasw Hc) as Hry;
           rewrite (has_source_alias _ _ _ closedU y r Ey Hry) in Hno;
           discriminate));
    (assert (Hcallee :
               callee_result_has_source F T (node_of_simple (ap_callee ap)) i
               = false)
       by (unfold callee_result_has_source, result_has_source_at;
           apply Bool.not_true_iff_false; intros E;
           apply existsb_exists in E as ([[a' f] base] & Ha' & Ht);
           cbn beta iota in Ht;
           apply andb_true_iff in Ht as [Ht Hor];
           apply andb_true_iff in Ht as [Hf Hb];
           apply field_eqb_eq in Hf; apply node_eqb_eq in Hb; subst f base;
           destruct (FlambdaReaperWrapperProofs.widgets_same F T a' a _ Hwid
                       Ha' Hread) as [Hany Hsub];
           apply orb_true_iff in Hor as [Hm | Hm];
           [ change (s_any_source S a' = true) in Hm;
             rewrite Hany, Hasa in Hm; discriminate | ];
           apply existsb_exists in Hm as ([a'' s] & Hs & Ht2);
           cbn beta iota in Ht2;
           apply andb_true_iff in Ht2 as [Ha'' Hc];
           apply node_eqb_eq in Ha''; subst a'';
           apply existsb_exists in Hc as ([[s' f'] y] & Hc & Ht3);
           cbn beta iota in Ht3;
           apply andb_true_iff in Ht3 as [Ht3 Hy];
           apply andb_true_iff in Ht3 as [Hs' Hf'];
           apply node_eqb_eq in Hs'; apply field_eqb_eq in Hf'; subst s' f';
           assert (Hsa : s_sources S a s = true)
             by (apply Hsub; apply (mem_pair_In _ _ _); exact Hs);
           change (s_has_source S y = true) in Hy;
           rewrite (Hkey s y Hsa Hc) in Hy; discriminate));
    exact (FlambdaReaperWrapperProofs.known_never_returns Auto is_local u e0
             T Hloc Hleast Hknown slots_read noted deps Hconsts Hnodup Hids
             Hcode_edges ap a i ltac:(rewrite Hkind; discriminate) Hread
             Hcallee Hlt Har).
  (* A call of unknown arity. *)
  specialize (Hargs eq_refl).
  unfold FlambdaReaperFlowProofs.call_edges in Hcall. rewrite Hkind in Hcall.
  destruct Hcall as (ws & a & called & Hlen & Hchain & _ & Hhd & Hread).
  rewrite (call_groups_length ap Hargs) in Hlen.
  destruct ws as [ | w ws']; [ destruct (ap_args_arity ap); cbn in Hlen;
                               congruence | ].
  pose proof (Halias _ _ (Hhd w eq_refl)) as Hwa.
  unfold FlambdaReaperFlowProofs.call_groups in Hchain.
  pose proof (call_groups_length ap Hargs) as Hglen.
  unfold FlambdaReaperFlowProofs.call_groups in Hglen.
  destruct (at_least_one_group _) as [ | g gs];
    [ destruct (ap_args_arity ap); cbn in Hglen; congruence | ].
  cbn [combine] in Hchain.
  pose proof (chain_no_any _ _ _ i r Hi Hno _ g w Hchain) as Hw.
  unfold apply_never_returns, code_id_actually_called. rewrite Hkind.
  destruct (ap_callee ap) as [x | sym | c0] eqn:Hcallee; cbv zeta.
  3:{ exfalso. cbn [node_of_simple] in Hread.
      pose proof (any_source_input _ _ _ closedU _ Hconsts) as Hac.
      pose proof (any_source_accessor _ _ _ closedU _ _ a Hac Hread
                    (wit_nl _)) as Haa.
      rewrite (any_source_alias _ _ _ closedU a w Hwa Haa) in Hw.
      discriminate. }
  all: remember (map (fun c => length (unarize_component c))
                   (ap_args_arity ap)) as num eqn:Hnum;
    assert (Hnl : length num = length (ap_args_arity ap))
      by (rewrite Hnum; apply length_map);
    destruct num as [ | n num']; [ destruct (ap_args_arity ap);
                                   cbn in Hnl; congruence | ];
    refine (chain_never_returns _ _ _ i r _ Hi Hno Hlt _ g w num' n _
              _ a Hchain _ _ Hread _ _);
    [ rewrite length_combine; cbn in Hlen, Hglen, Hnl; lia
    | unfold get_direct_sources, approx;
      match goal with
      | |- context [if s_any_source S ?n then _ else _] =>
        destruct (s_any_source S n) eqn:E; [ reflexivity | ]
      end;
      intros y Hy; apply andb_true_iff in Hy as [Hy _]; exact Hy
    | intros Haa; exact (any_source_alias _ _ _ closedU a w Hwa Haa)
    | intros s Hs Hw0;
      assert (Hasa : s_any_source S a = false)
        by (destruct (s_any_source S a) eqn:E; [ | reflexivity ];
            rewrite (any_source_alias _ _ _ closedU a w Hwa E) in Hw0;
            discriminate);
      exact (sources_alias _ _ _ closedU a w s Hs Hasa Hwa Hw0) ].
Qed.

Variable cannot_change : list node.
Variable unboxed : list (node * unboxed_fields).
Variable ff : positive.
Variable changed : list (node * closure_repr).
Variable pdc : bool.

Abbreviation rebuild :=
  (rebuild_expr_u true F S (fun n => mem_node n cannot_change) cds pdc
     (static_names code_ids_bound e0) (bound_code_metadata e0) unboxed ff
     changed).
Abbreviation rebuild_handler :=
  (rebuild_handler_u true F S (fun n => mem_node n cannot_change) cds pdc
     (static_names code_ids_bound e0) (bound_code_metadata e0) unboxed ff
     changed).
Abbreviation calls :=
  (rebuilt_calls S (fun n => mem_node n cannot_change) cds
     (static_names code_ids_bound e0) unboxed).
Abbreviation handler_calls :=
  (handler_rebuilt_calls S (fun n => mem_node n cannot_change) cds
     (static_names code_ids_bound e0) unboxed).
Abbreviation edges :=
  (FlambdaReaperFlowProofs.expr_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).
Abbreviation handler_edges :=
  (FlambdaReaperFlowProofs.handler_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).

Lemma lookup_map_none {B : Type} (k : positive)
    (hs : list (positive * continuation_handler))
    (g : continuation_handler -> B) :
  k ∉ map fst hs -> lookup k (map (fun '(k', h) => (k', g h)) hs) = None.
Proof.
  induction hs as [ | [k' h] hs IH]; intros H; [ reflexivity | ].
  cbn [map lookup Keyed.lookup].
  destruct (Pos.eqb k k') eqn:E.
  - apply Pos.eqb_eq in E. subst k'. exfalso. apply H. left. reflexivity.
  - apply IH. intros H'. apply H. right. exact H'.
Qed.

Lemma traps_of (k : continuation) (ac : apply_cont) :
  k ∈ apply_cont_continuations ac -> k <> ac_continuation ac ->
  traps k ac = true.
Proof.
  unfold apply_cont_continuations, traps.
  intros [H | H] Hne; [ congruence | ].
  destruct (ac_trap_action ac); [ | destruct H ].
  destruct H as [<- | []]. apply Pos.eqb_refl.
Qed.

(* An [Apply_cont] to [k], one of whose parameters has no source, is
   rebuilt as [Invalid]; [k] is free only by a trap action. *)
Lemma ac_not_free (ctx : option code_id) (ce : finite_map (list node))
    (en : unbox_env) (ac : apply_cont) (k : continuation) (ns : list node)
    (r : node) :
  FlambdaReaperFlowProofs.apply_cont_edges G noted ce ac ->
  count_ok (FlambdaReaperFlowProofs.counts ce) (ac_continuation ac)
    (length (ac_args ac))
  = true ->
  lookup k ce = Some ns -> r ∈ ns -> s_has_source S r = false ->
  traps k ac = false ->
  k ∉ fc (rebuild_apply_cont_u S unboxed en ac).
Proof.
  intros He Hc Hk Hr Hno Ht Hin.
  apply fc_apply_cont in Hin as [Hin Hne].
  destruct (Pos.eq_dec k (ac_continuation ac)) as [-> | Hneq].
  - rewrite (ac_no_source ce ac ns r He Hc Hk Hr Hno) in Hne.
    discriminate (Hne eq_refl).
  - rewrite (traps_of k ac Hin Hneq) in Ht. discriminate.
Qed.

Ltac peel IH Hin :=
  first
    [ exact (IH _ Hin)
    | eapply IH; eapply fc_unboxed_set; exact Hin
    | eapply IH; eapply fc_unboxed_binding; exact Hin
    | eapply IH; eapply fc_changed_binding; exact Hin
    | eapply IH; eapply fc_let_u; exact Hin ].

Lemma no_return :
  (∀ e : expr, ∀ ctx ce ds k ns r,
     edges ctx ce e ->
     conts_counted (FlambdaReaperFlowProofs.counts ce) e = true ->
     binds_statics e = false ->
     FlambdaReaperWrapperProofs.live T ctx ->
     (∀ ads ap, (ads, ap) ∈ calls ds e -> returns_ok ap) ->
     lookup k ce = Some ns -> r ∈ ns -> s_has_source S r = false ->
     (k ∈ free_continuations e -> raises_to k e = false) ->
     ∀ en, k ∉ fc (rebuild en e))
  ∧ (∀ h : continuation_handler, ∀ ctx ce ds k ns r,
       handler_edges ctx ce h ->
       handler_conts_counted (FlambdaReaperFlowProofs.counts ce) h = true ->
       handler_binds_statics h = false ->
       FlambdaReaperWrapperProofs.live T ctx ->
       (∀ ads ap, (ads, ap) ∈ handler_calls ds h -> returns_ok ap) ->
       lookup k ce = Some ns -> r ∈ ns -> s_has_source S r = false ->
       (k ∈ handler_free_continuations h -> handler_raises_to k h = false) ->
       ∀ en dc decs,
         k ∉ free_continuations_of_handler (rebuild_handler en dc decs h)).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH ctx ce ds k ns r [_ He] Hc Hb Hl Hcalls Hk Hr
      Hno Hx en Hin.
    specialize (fun en => IH ctx ce ds k ns r He Hc Hb Hl Hcalls Hk Hr Hno Hx
                            en).
    cbn [rebuild_expr_u] in Hin.
    repeat match type of Hin with
    | context [if ?b then _ else _] => destruct b
    | context [match ?x with _ => _ end] => destruct x
    end;
    cbn [free_continuations_of_expr] in Hin;
    first [ destruct Hin | peel IH Hin ].
  - (* Let_static: a body of code binds nothing. *)
    intros bindings body _ IH ctx ce ds k ns r [_ He] Hc Hb Hl Hcalls Hk Hr
      Hno Hx en Hin.
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbs Hb].
    cbn [rebuild_expr_u free_continuations_of_expr] in Hin.
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b' & Hb' & Hin).
      apply in_flat_map in Hb' as (b & Hbin & Hb').
      destruct b as [id c | id | sym const].
      * assert (H : existsb binding_binds bindings = true)
          by (apply existsb_exists; exists (Bind_code id c); auto).
        congruence.
      * destruct Hb' as [<- | []]. destruct Hin.
      * assert (H : existsb binding_binds bindings = true)
          by (apply existsb_exists; exists (Bind_symbol sym const); auto).
        congruence.
    + exact (IH ctx ce ds k ns r He Hc Hb Hl
               (fun ads ap H => Hcalls ads ap (in_or_app _ _ _ (or_intror H)))
               Hk Hr Hno Hx _ Hin).
  - (* Let_cont_non_recursive *)
    intros k' h body IHh IH ctx ce ds k ns r [Hh He] Hc Hb Hl Hcalls Hk Hr
      Hno Hx en.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbh Hb].
    cbn [free_continuations raises_to] in Hx.
    assert (Hxh : k ∈ handler_free_continuations h ->
                  handler_raises_to k h = false).
    { intros H.
      assert (H' : k ∈ handler_free_continuations h
                     ++ without [k'] (free_continuations body))
        by (apply in_or_app; left; exact H).
      apply Hx in H'. apply orb_false_iff in H' as [H' _]. exact H'. }
    pose proof (IHh ctx ce ds k ns r Hh Hch Hbh Hl
                  (fun ads ap H =>
                     Hcalls ads ap (in_or_app _ _ _ (or_introl H)))
                  Hk Hr Hno Hxh) as IHh'.
    assert (Hbody : k <> k' -> ∀ en',
              k ∉ fc (rebuild en' body)).
    { intros Hne en'.
      refine (IH ctx ((k', FlambdaReaperTraversal.handler_params h) :: ce)
                _ k ns r He Hc Hb Hl
                (fun ads ap H => Hcalls ads ap (in_or_app _ _ _ (or_intror H)))
                _ Hr Hno _ en').
      - cbn [lookup Keyed.lookup].
        destruct (Pos.eqb k k') eqn:E; [ apply Pos.eqb_eq in E; congruence | ].
        exact Hk.
      - intros H.
        assert (H' : k ∈ handler_free_continuations h
                       ++ without [k'] (free_continuations body)).
        { apply in_or_app. right. unfold without. apply filter_In.
          split; [ exact H | ]. apply negb_true_iff.
          destruct (among k [k']) eqn:E; [ | reflexivity ].
          apply FlambdaMapProofs.among_In in E. destruct E as [E | []].
          congruence. }
        apply Hx in H'. apply orb_false_iff in H' as [_ H']. exact H'. }
    cbn [rebuild_expr_u]. intros Hin.
    destruct (among k' (fc (rebuild _ body))) eqn:Hk'.
    + cbn [free_continuations_of_expr] in Hin.
      apply in_app_or in Hin as [Hin | Hin]; [ exact (IHh' _ _ _ Hin) | ].
      apply in_without_inv in Hin as [Hin Hne].
      refine (Hbody _ _ Hin). intros ->. apply Hne. left. reflexivity.
    + destruct (Pos.eq_dec k k') as [-> | Hne].
      * apply FlambdaMapProofs.among_In in Hin. congruence.
      * exact (Hbody Hne _ Hin).
  - (* Let_cont_recursive *)
    intros handlers body IHh IH ctx ce ds k ns r [Hhs He] Hc Hb Hl Hcalls Hk
      Hr Hno Hx en.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    rewrite FlambdaReaperWrapperProofs.counts_rec in Hch, Hc.
    rewrite forallb_forall in Hch.
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbh Hb].
    cbn [free_continuations raises_to] in Hx.
    cbn [rebuild_expr_u free_continuations_of_expr]. intros Hin.
    apply in_without_inv in Hin as [Hin Hnot].
    assert (Hnot' : k ∉ map fst handlers).
    { intros H. apply Hnot. rewrite map_map.
      erewrite map_ext; [ exact H | ]. intros [k0 h0]. reflexivity. }
    assert (Hk' : lookup k (FlambdaReaperFlowProofs.rec_conts ce handlers)
                  = Some ns).
    { unfold FlambdaReaperFlowProofs.rec_conts.
      rewrite FlambdaMapProofs.lookup_app.
      rewrite (lookup_map_none k handlers _ Hnot'). exact Hk. }
    assert (Hin_e : ∀ k0, k0 ∉ map fst handlers ->
               k0 ∈ flat_map (fun '(_, h) => handler_free_continuations h)
                      handlers
                    ++ free_continuations body ->
               k0 ∈ without (map fst handlers)
                      (flat_map
                         (fun '(_, h) => handler_free_continuations h) handlers
                       ++ free_continuations body)).
    { intros k0 Hn H. unfold without. apply filter_In. split; [ exact H | ].
      apply negb_true_iff. destruct (among k0 (map fst handlers)) eqn:E;
        [ | reflexivity ].
      apply FlambdaMapProofs.among_In in E. contradiction. }
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k0 h'] & Hkh & Hin).
      apply in_map_iff in Hkh as ([k1 h1] & [= <- <-] & Hkh).
      refine (IHh k1 h1 Hkh ctx _ _ k ns r
                (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh)
                (Hch _ Hkh) _ Hl _ Hk' Hr Hno _ _ _ _ Hin).
      * destruct (handler_binds_statics h1) eqn:E; [ | reflexivity ].
        rewrite <- Hbh. symmetry.
        apply existsb_exists. exists (k1, h1). auto.
      * intros ads ap H. apply (Hcalls ads). apply in_or_app. left.
        apply in_flat_map. exists (k1, h1). split; [ exact Hkh | exact H ].
      * intros H.
        assert (H' := Hin_e k Hnot'
                        (in_or_app _ _ _
                           (or_introl (proj2 (in_flat_map _ _ _)
                                         (ex_intro _ (k1, h1) (conj Hkh H)))))).
        apply Hx in H'. apply orb_false_iff in H' as [H' _].
        destruct (handler_raises_to k h1) eqn:E; [ | reflexivity ].
        rewrite <- H'. symmetry. apply existsb_exists. exists (k1, h1). auto.
    + refine (IH ctx _ _ k ns r He Hc Hb Hl _ Hk' Hr Hno _ _ Hin).
      * intros ads ap H. apply (Hcalls ads). apply in_or_app. right. exact H.
      * intros H.
        assert (H' := Hin_e k Hnot' (in_or_app _ _ _ (or_intror H))).
        apply Hx in H'. apply orb_false_iff in H' as [_ H']. exact H'.
  - (* Apply: a call to [k] never returns. *)
    intros ap ctx ce ds k ns r He Hc _ Hl Hcalls Hk Hr Hno Hx en Hin.
    cbn [rebuild_expr_u] in Hin.
    apply fc_apply in Hin as [-> | [-> Hnr]].
    + cbn [free_continuations raises_to] in Hx.
      rewrite Pos.eqb_refl in Hx.
      discriminate (Hx (or_intror (or_introl eq_refl))).
    + rewrite (call_never_returns ctx ce ap ns r He Hc Hl
                 (Hcalls _ ap (or_introl eq_refl)) Hk Hr Hno) in Hnr.
      discriminate.
  - (* Apply_cont *)
    intros ac ctx ce ds k ns r He Hc _ _ _ Hk Hr Hno Hx en.
    cbn [rebuild_expr_u].
    cbn [conts_counted] in Hc.
    intros Hin. pose proof Hin as Hin'.
    apply fc_apply_cont in Hin' as [Hin' _].
    cbn [free_continuations raises_to] in Hx.
    exact (ac_not_free ctx ce en ac k ns r He Hc Hk Hr Hno (Hx Hin') Hin).
  - (* Switch *)
    intros scrutinee arms ctx ce ds k ns r He Hc _ _ _ Hk Hr Hno Hx en.
    cbn [rebuild_expr_u]. intros Hin.
    apply fc_switch in Hin as (d & ac & Hac & Hin & Hne).
    assert (Harm : (d, ac) ∈ arms).
    { apply (FlambdaMapProofs.keyed_in_compact Z.eqb Z.eqb_eq) in Hac.
      exact (FlambdaMapProofs.keyed_lookup_in Z.eqb Z.eqb_eq d ac arms Hac). }
    cbn [FlambdaReaperFlowProofs.expr_edges] in He.
    pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) He _ Hac) as Heac.
    cbn beta iota in Heac.
    cbn [conts_counted] in Hc. rewrite forallb_forall in Hc.
    pose proof (Hc _ Harm) as Hcac. cbn beta iota in Hcac.
    cbn [free_continuations raises_to] in Hx.
    assert (Ht : traps k ac = false).
    { destruct (traps k ac) eqn:E; [ | reflexivity ].
      assert (H : k ∈ flat_map (fun arm => apply_cont_continuations (snd arm))
                        arms)
        by (apply in_flat_map; exists (d, ac);
            split; [ exact Harm | exact Hin ]).
      specialize (Hx H). rewrite <- Hx. symmetry.
      apply existsb_exists. exists (d, ac). split; [ exact Harm | exact E ]. }
    destruct (Pos.eq_dec k (ac_continuation ac)) as [-> | Hneq].
    + rewrite (ac_no_source ce ac ns r Heac Hcac Hk Hr Hno) in Hne.
      discriminate (Hne eq_refl).
    + rewrite (traps_of k ac Hin Hneq) in Ht. discriminate.
  - intros ctx ce ds k ns r _ _ _ _ _ _ _ _ _ en Hin. exact Hin.
  - (* A handler *)
    intros params metadata handler IH ctx ce ds k ns r He Hc Hb Hl Hcalls Hk
      Hr Hno Hx en dc decs.
    cbn [rebuild_handler_u free_continuations_of_handler].
    destruct (dc && _); [ intros [] | ].
    exact (IH ctx ce ds k ns r He Hc Hb Hl Hcalls Hk Hr Hno Hx _).
Qed.

(* The calls of code that is used are calls of the unit. *)
Lemma code_calls_in_unit (id : code_id) (c : code) :
  is_code_id_used S id = true ->
  (∀ e : expr, ∀ ds, (id, c) ∈ unit_codes e ->
     ∀ ads ap,
       (ads, ap)
       ∈ code_rebuilt_calls S (fun n => mem_node n cannot_change) cds
           (static_names code_ids_bound e0) unboxed id c ->
       (ads, ap) ∈ calls ds e)
  ∧ (∀ h : continuation_handler, ∀ ds, (id, c) ∈ handler_unit_codes h ->
       ∀ ads ap,
         (ads, ap)
         ∈ code_rebuilt_calls S (fun n => mem_node n cannot_change) cds
             (static_names code_ids_bound e0) unboxed id c ->
         (ads, ap) ∈ handler_calls ds h).
Proof.
  intros Hused. apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [unit_codes handler_unit_codes rebuilt_calls handler_rebuilt_calls].
  - intros x kind d body IH ds Hin ads ap H. exact (IH ds Hin ads ap H).
  - intros bindings body _ IH ds Hin ads ap H.
    apply in_app_or in Hin as [Hin | Hin];
      [ | apply in_or_app; right; exact (IH ds Hin ads ap H) ].
    apply in_or_app. left.
    apply in_flat_map in Hin as (b & Hbin & Hin).
    destruct b as [id' c' | id' | sym const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= -> ->] | []].
    apply in_flat_map. exists (Bind_code id c). split; [ exact Hbin | ].
    cbn [binding_rebuilt_calls]. rewrite Hused. exact H.
  - intros k h body IHh IH ds Hin ads ap H.
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
      [ left; exact (IHh ds Hin ads ap H)
      | right; exact (IH _ Hin ads ap H) ].
  - intros handlers body IHh IH ds Hin ads ap H.
    apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
      [ left | right; exact (IH _ Hin ads ap H) ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    apply in_flat_map. exists (k, h).
    split; [ exact Hkh | exact (IHh k h Hkh _ Hin ads ap H) ].
  - intros ap0 ds [].
  - intros ac ds [].
  - intros scrutinee arms ds [].
  - intros ds [].
  - intros params metadata handler IH ds Hin ads ap H.
    exact (IH ds Hin ads ap H).
Qed.

End Flow.

End CheckNeverReturns.

Theorem check_never_returns_auto : check_never_returns_auto_statement.
Proof.
  intros is_local u e T cannot_change unboxed ff changed pdc F code_deps S
    Hloc Hslots Hfrag Hleast Hwf Hcalls id c Hc Hused.
  subst F code_deps S.
  destruct (code_never_returns _ _ id) eqn:Hnr;
    [ cbn [andb] | reflexivity ].
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e
              (∀ f : field, is_local f = false) Hfrag Hslots)
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  pose proof (proj1 FlambdaReaperRebuildProofs.unit_codes_exn_wf e Hwf id c Hc)
    as Hcwf.
  pose proof (proj1 (CheckNeverReturns.code_calls_in_unit u e T cannot_change
                       unboxed id c
                       Hused) e [] Hc) as Hcu.
  pose proof (Hcode_edges id c Hc) as Hce.
  destruct c as [params my_closure ret exn metadata body].
  cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
  destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _ & _ & Hne & Hbs & Hcc
                   & Hbe).
  pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
    as Hlookup.
  unfold code_never_returns in Hnr. rewrite Hlookup in Hnr.
  apply existsb_exists in Hnr as (r & Hr & Hno). apply negb_true_iff in Hno.
  cbn [code_exn_handlers_well_formed] in Hcwf.
  apply andb_true_iff in Hcwf as [Hcwf _].
  apply andb_true_iff in Hcwf as [Hways _].
  unfold ways_out_well_formed in Hways. rewrite forallb_forall in Hways.
  assert (Hx : ret ∈ free_continuations body -> raises_to ret body = false).
  { intros H. specialize (Hways ret H). cbv beta in Hways.
    destruct (Pos.eqb ret exn) eqn:E;
      [ apply Pos.eqb_eq in E; congruence | ].
    apply andb_true_iff in Hways as [Hw _]. apply negb_true_iff in Hw.
    exact Hw. }
  apply Bool.not_true_iff_false. intros Hin.
  apply FlambdaMapProofs.among_In in Hin.
  cbn [rebuild_code_u code_return_continuation] in Hin.
  assert (Hcounted :
            conts_counted
              (FlambdaReaperFlowProofs.counts
                 [(exn, [cd_exn cd]); (ret, cd_return cd)])
              body
            = true).
  { unfold FlambdaReaperFlowProofs.counts. cbn [map length]. rewrite Hlen.
    exact Hcc. }
  assert (Hret : lookup ret [(exn, [cd_exn cd]); (ret, cd_return cd)]
                 = Some (cd_return cd)).
  { cbn [lookup Keyed.lookup].
    destruct (Pos.eqb ret exn) eqn:E; [ apply Pos.eqb_eq in E; congruence | ].
    rewrite Pos.eqb_refl. reflexivity. }
  pose proof (FlambdaReaperWrapperProofs.used_code_live T
                (FlambdaReaperWrapperShapeProofs.traverse_used_code_ids Auto
                   is_local u e T Hloc Hleast)
                id Hused) as Hlive.
  match type of Hin with
  | context [code_body (match ?x with _ => _ end)] => destruct x
  end; cbn [code_body] in Hin;
  (destruct (existsb _ _); [ destruct Hin | ]);
  exact (proj1 (CheckNeverReturns.no_return is_local u e T Hloc Hleast _ noted
                  deps Hconsts
                  Hnodup Hids Hcode_edges cannot_change unboxed ff changed pdc)
           body (Some id) _ _ ret (cd_return cd) r Hbe Hcounted Hbs Hlive
           (fun ads ap H => Hcalls ads ap (Hcu ads ap H)) Hret Hr Hno Hx
           _ Hin).
Qed.
