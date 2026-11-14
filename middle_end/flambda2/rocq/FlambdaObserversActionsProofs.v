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

(* Proofs of the statements of FlambdaObserversActionsSpec.v. This file
   needs no review: the statements are there.

   A renaming. Each statement that is an equivalence is proved one way
   for every renaming, and the other way is that one for the inverse.
   What a continuation stands for, applied, commutes with renaming
   (FlambdaEquivarianceProofs.apply_cont_value_equivariant), and an
   observer is the renaming of the observer renamed back. The test of
   the static names in scope is renamed in
   FlambdaValidatorEquivarianceProofs.v.

   A class. The three statements over a class of continuation values
   are proved as those over the observers were, with the premise in
   the place of [observer_equivariant]: what is in the class is the
   renaming of what is in it, renamed back. The three over the
   observers are then proved as their instances at the class
   [observer slots], so that the kernel checks that each is one.

   A permutation of locations. A value moved is moved back by the
   inverse, so moving is injective; a constant and a location keep
   their form. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaNameAbstractionSpec
  FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaStaticsInScopeSpec
  FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs
  FlambdaValidatorEquivarianceProofs FlambdaHeapSpec FlambdaExitWorldSpec
  FlambdaExitWorldProofs FlambdaRunUnderSpec FlambdaObserversSpec
  FlambdaLocationsSpec FlambdaLocationsProofs
  FlambdaExitWorldActionsSpec FlambdaExitWorldActionsProofs
  FlambdaObserversActionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* A renaming                                                               *)
(* ======================================================================== *)

Lemma rename_result_inverse (r : renaming) (result : step_result) :
  rename_result (inverse_renaming r) (rename_result r result) = result
  ∧ rename_result r (rename_result (inverse_renaming r) result) = result.
Proof.
  destruct result as [s | k vs hp rd | ]; simpl; split; try reflexivity.
  - rewrite (proj1 (rename_state_inverse r s)). reflexivity.
  - rewrite (proj2 (rename_state_inverse r s)). reflexivity.
  - rewrite rename_continuation_inverse_l. reflexivity.
  - rewrite rename_continuation_inverse_r. reflexivity.
Qed.

Lemma rename_cont_value_inverse_r (r : renaming) (c : cont_value) :
  rename_cont_value r (rename_cont_value (inverse_renaming r) c) = c.
Proof.
  rewrite <- rename_cont_value_compose.
  rewrite (rename_cont_value_ext _ identity_renaming);
    [ apply rename_cont_value_id | | ].
  - apply compose_inverse_r_variable.
  - apply compose_inverse_r_continuation.
Qed.

Theorem step_result_behaves_equivariant :
  step_result_behaves_equivariant_statement.
Proof.
  intros r [s | k vs hp rd | ] b; simpl.
  - apply behaves_equivariant.
  - destruct b as [k' vs' hp' | | ]; simpl; split; intros H;
      try discriminate H.
    + injection H as Hk Hvs Hhp.
      apply rename_continuation_injective in Hk. congruence.
    + injection H as Hk Hvs Hhp. congruence.
  - destruct b as [k' vs' hp' | | ]; simpl; split; intros H;
      try discriminate H; reflexivity.
Qed.

(* A behaviour of a result renamed is the renaming of one of the
   result. *)
Lemma step_result_behaves_renamed (r : renaming) (result : step_result)
    (b : behaviour) :
  step_result_behaves (rename_result r result) b
  <-> step_result_behaves result (rename_behaviour (inverse_renaming r) b).
Proof.
  rewrite <- (rename_behaviour_inverse r b) at 1.
  apply step_result_behaves_equivariant.
Qed.

Lemma observed_refines_renamed (r : renaming) (source target : step_result) :
  observed_refines source target ->
  observed_refines (rename_result r source) (rename_result r target).
Proof.
  intros [Hexits Hends]. split.
  - intros k vs hp Hk. apply step_result_behaves_renamed in Hk.
    destruct (Hexits _ _ _ Hk) as (vs' & hp' & Hk' & Hscalars & Hdomain).
    exists vs', hp'. split; [ | split; assumption ].
    apply step_result_behaves_renamed. exact Hk'.
  - intros b Hb Hbehaves. apply step_result_behaves_renamed in Hbehaves.
    destruct (Hends (rename_behaviour (inverse_renaming r) b))
      as [Hwrong | (k & vs & hp & Hk)].
    + destruct b; [ discriminate | discriminate | exact Hb ].
    + exact Hbehaves.
    + left. apply step_result_behaves_renamed. exact Hwrong.
    + right. exists (rename_continuation r k), vs, hp.
      apply step_result_behaves_renamed. simpl.
      rewrite rename_continuation_inverse_l. exact Hk.
Qed.

Theorem observed_refines_equivariant :
  observed_refines_equivariant_statement.
Proof.
  intros r source target. split; [ | apply observed_refines_renamed ].
  intros Hrefines.
  apply (observed_refines_renamed (inverse_renaming r)) in Hrefines.
  rewrite !(proj1 (rename_result_inverse r _)) in Hrefines.
  exact Hrefines.
Qed.

Fixpoint rename_binds_no_static (r : renaming) (e : expr) :
  binds_no_static (rename_expr r e) = binds_no_static e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    simpl; try reflexivity.
  - apply rename_binds_no_static.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rename_binds_no_static. reflexivity.
  - rewrite rename_binds_no_static. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH];
      simpl; [ reflexivity | ].
    rewrite rename_binds_no_static, IH. reflexivity.
Qed.

Theorem binds_no_static_equivariant :
  binds_no_static_equivariant_statement.
Proof. exact rename_binds_no_static. Qed.

Theorem expr_statics_in_scope_equivariant :
  expr_statics_in_scope_equivariant_statement.
Proof.
  intros r syms ids e. apply expr_statics_in_scope_renamed.
Qed.

Theorem observer_equivariant : observer_equivariant_statement.
Proof.
  intros r slots c.
  destruct c
    as [[params metadata body] vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl; try reflexivity.
  destruct vars; destruct conts; destruct recursive; simpl;
    try reflexivity.
  rewrite rename_binds_no_static, expr_statics_in_scope_renamed,
    (slots_unused_equivariant r slots body).
  reflexivity.
Qed.

Theorem observers_equivariant : observers_equivariant_statement.
Proof.
  intros r slots conts. unfold observers, rename_cont_env.
  induction conts as [ | [k c] conts IH]; simpl.
  - split; intros _; constructor.
  - split; intros H; inversion H as [ | kc rest Hc Hrest]; subst;
      constructor.
    + apply (observer_equivariant r). exact Hc.
    + apply IH. exact Hrest.
    + apply (observer_equivariant r). exact Hc.
    + apply IH. exact Hrest.
Qed.

(* ---- A class of what may follow a unit --------------------------------- *)

Lemma worlds_observed_by_renamed (O : cont_value -> Prop)
    (HO : ∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c))
    (r : renaming) (w w' : exit_world) :
  worlds_observed_by O w w' ->
  worlds_observed_by O (rename_exit_world r w) (rename_exit_world r w').
Proof.
  intros [Hk Hobserved]. split.
  - simpl. rewrite Hk. reflexivity.
  - intros c Hc. simpl.
    rewrite <- (rename_cont_value_inverse_r r c).
    rewrite !apply_cont_value_equivariant.
    apply observed_refines_renamed, Hobserved.
    apply (HO (inverse_renaming r)). exact Hc.
Qed.

Theorem worlds_observed_by_equivariant :
  worlds_observed_by_equivariant_statement.
Proof.
  intros O HO r w w'. split; [ | apply worlds_observed_by_renamed, HO ].
  intros Hobserved.
  apply (worlds_observed_by_renamed O HO (inverse_renaming r)) in Hobserved.
  rewrite !(proj1 (rename_exit_world_inverse r _)) in Hobserved.
  exact Hobserved.
Qed.

Lemma refines_world_observed_by_renamed (O : cont_value -> Prop)
    (HO : ∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c))
    (r : renaming) (source target : state) :
  refines_world (worlds_observed_by O) source target ->
  refines_world (worlds_observed_by O) (rename_state r source)
    (rename_state r target).
Proof.
  intros Hrefines. apply refines_world_kept; [ | exact Hrefines ].
  intros w w'. apply worlds_observed_by_renamed, HO.
Qed.

Theorem refines_world_observed_by_equivariant :
  refines_world_observed_by_equivariant_statement.
Proof.
  intros O HO r source target.
  split; [ | apply refines_world_observed_by_renamed, HO ].
  intros Hrefines.
  apply (refines_world_observed_by_renamed O HO (inverse_renaming r))
    in Hrefines.
  rewrite !(proj1 (rename_state_inverse r _)) in Hrefines.
  exact Hrefines.
Qed.

(* A continuation environment of the class, renamed, is one. *)
Lemma class_env_renamed (O : cont_value -> Prop)
    (HO : ∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c))
    (r : renaming) (conts : cont_env) :
  Forall (fun kc => O (snd kc)) conts ->
  Forall (fun kc => O (snd kc)) (rename_cont_env r conts).
Proof.
  unfold rename_cont_env.
  induction conts as [ | [k c] conts IH]; simpl; intros H.
  - constructor.
  - inversion H as [ | kc rest Hc Hrest]; subst. constructor.
    + apply HO. exact Hc.
    + apply IH. exact Hrest.
Qed.

Lemma refines_observed_by_renamed (O : cont_value -> Prop)
    (HO : ∀ (r : renaming) (c : cont_value), O c -> O (rename_cont_value r c))
    (r : renaming) (statics : static_env) (hp : heap) (rd : read_record)
    (source target : expr) :
  refines_observed_by O statics hp rd source target ->
  refines_observed_by O (rename_static_env r statics) hp rd
    (rename_expr r source) (rename_expr r target).
Proof.
  intros Hrefines conts Hconts.
  rewrite !state_of_renamed_expr, rename_static_env_inverse_l.
  apply (observed_refines_renamed r
           (Step (State source (rename_env (inverse_renaming r) [])
                    (rename_cont_env (inverse_renaming r) conts) statics hp rd))
           (Step (State target (rename_env (inverse_renaming r) [])
                    (rename_cont_env (inverse_renaming r) conts) statics
                    hp rd))).
  apply Hrefines.
  apply class_env_renamed; [ exact HO | exact Hconts ].
Qed.

Theorem refines_observed_by_equivariant :
  refines_observed_by_equivariant_statement.
Proof.
  intros O HO r statics hp source target.
  split; [ | apply refines_observed_by_renamed, HO ].
  intros Hrefines.
  apply (refines_observed_by_renamed O HO (inverse_renaming r)) in Hrefines.
  rewrite rename_static_env_inverse_l in Hrefines.
  rewrite !rename_expr_inverse in Hrefines.
  exact Hrefines.
Qed.

(* ---- The observers: the class [observer slots] -------------------------- *)

Lemma observer_renamed (slots : list value_slot) (r : renaming)
    (c : cont_value) :
  observer slots c -> observer slots (rename_cont_value r c).
Proof. apply (proj2 (observer_equivariant r slots c)). Qed.

Lemma worlds_observed_renamed (r : renaming) (slots : list value_slot)
    (w w' : exit_world) :
  worlds_observed slots w w' ->
  worlds_observed slots (rename_exit_world r w) (rename_exit_world r w').
Proof.
  exact (worlds_observed_by_renamed (observer slots) (observer_renamed slots)
           r w w').
Qed.

Theorem worlds_observed_equivariant :
  worlds_observed_equivariant_statement.
Proof.
  intros r slots w w'.
  exact (worlds_observed_by_equivariant (observer slots)
           (observer_renamed slots) r w w').
Qed.

Lemma refines_world_observed_renamed (r : renaming)
    (slots : list value_slot) (source target : state) :
  refines_world (worlds_observed slots) source target ->
  refines_world (worlds_observed slots) (rename_state r source)
    (rename_state r target).
Proof.
  exact (refines_world_observed_by_renamed (observer slots)
           (observer_renamed slots) r source target).
Qed.

Theorem refines_world_observed_equivariant :
  refines_world_observed_equivariant_statement.
Proof.
  intros r slots source target.
  exact (refines_world_observed_by_equivariant (observer slots)
           (observer_renamed slots) r source target).
Qed.

Lemma refines_under_observers_renamed (r : renaming)
    (slots : list value_slot) (statics : static_env) (hp : heap)
      (rd : read_record)
    (source target : expr) :
  refines_under_observers slots statics hp rd source target ->
  refines_under_observers slots (rename_static_env r statics) hp rd
    (rename_expr r source) (rename_expr r target).
Proof.
  exact (refines_observed_by_renamed (observer slots) (observer_renamed slots)
           r statics hp rd source target).
Qed.

Theorem refines_under_observers_equivariant :
  refines_under_observers_equivariant_statement.
Proof.
  intros r slots statics hp rd source target.
  exact (refines_observed_by_equivariant (observer slots)
           (observer_renamed slots) r statics hp rd_ source target).
Qed.

(* ======================================================================== *)
(* A permutation of locations                                               *)
(* ======================================================================== *)

Lemma permute_value_injective (p : permutation) (v v' : value) :
  permute_value p v = permute_value p v' -> v = v'.
Proof.
  intros Heq.
  apply (f_equal (permute_value (inverse_permutation p))) in Heq.
  rewrite
    (proj1 (act_inverse permute_value locations_of_value p v value_laws)),
    (proj1 (act_inverse permute_value locations_of_value p v' value_laws))
    in Heq.
  exact Heq.
Qed.

Theorem scalar_permuted : scalar_permuted_statement.
Proof.
  intros p v. unfold scalar.
  destruct v; simpl; split; intros [[c' Hc] | [l' Hl]];
    try discriminate;
    try (left; eexists; reflexivity);
    try (right; eexists; reflexivity).
Qed.

Theorem same_scalars_permuted : same_scalars_permuted_statement.
Proof.
  intros p vs vs'. unfold same_scalars. split.
  - revert vs'.
    induction vs as [ | v vs IH]; intros [ | v' vs'] Hsame; simpl in Hsame;
      inversion Hsame as [ | a b la lb Hhead Htail]; subst; constructor.
    + intros Hscalar. apply (permute_value_injective p), Hhead.
      destruct Hscalar as [Hscalar | Hscalar]; [ left | right ];
        apply (proj2 (scalar_permuted p _)); exact Hscalar.
    + apply IH. exact Htail.
  - revert vs'.
    induction vs as [ | v vs IH]; intros [ | v' vs'] Hsame;
      inversion Hsame as [ | a b la lb Hhead Htail]; subst; simpl;
      constructor.
    + intros Hscalar. f_equal. apply Hhead.
      destruct Hscalar as [Hscalar | Hscalar]; [ left | right ];
        apply (proj1 (scalar_permuted p _)); exact Hscalar.
    + apply IH. exact Htail.
Qed.

Theorem heap_domain_permuted : heap_domain_permuted_statement.
Proof.
  intros p hp. unfold permute_heap. rewrite !map_map. reflexivity.
Qed.

Lemma map_permute_injective (p : permutation) (l l' : list positive) :
  map (permute p) l = map (permute p) l' -> l = l'.
Proof.
  revert l'.
  induction l as [ | x l IH]; intros [ | x' l'] Heq; simpl in Heq;
    try discriminate Heq; [ reflexivity | ].
  injection Heq as Hx Hl. apply permute_injective in Hx.
  rewrite Hx, (IH _ Hl). reflexivity.
Qed.

Theorem way_out_permuted : way_out_permuted_statement.
Proof.
  intros p w w'. simpl.
  split; intros (Hk & Hvs & Hhp); (split; [ exact Hk | split ]).
  - apply (proj1 (same_scalars_permuted p _ _)). exact Hvs.
  - rewrite !heap_domain_permuted in Hhp.
    apply map_permute_injective in Hhp. exact Hhp.
  - apply (proj2 (same_scalars_permuted p _ _)). exact Hvs.
  - rewrite !heap_domain_permuted, Hhp. reflexivity.
Qed.
