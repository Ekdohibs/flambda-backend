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


(* Proofs of the statements of FlambdaSimplifySwitchRewriteSpec.v.
   Nothing here needs review. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNominalSpec
  FlambdaCseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeRenameSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv FlambdaEnvRenameSpec FlambdaMeet
  FlambdaMeetSpec FlambdaMeetTermsSpec FlambdaMeetRenameSpec
  FlambdaSimplifySwitchSpec FlambdaSimplifySwitchProofs
  FlambdaSimplifySwitchArmsSpec FlambdaSimplifySwitchArmsProofs
  FlambdaSimplifySwitchMergeSpec FlambdaSimplifySwitchMergeProofs
  FlambdaSimplifyPruningSpec FlambdaSimplifyPruningRenamingSpec
  FlambdaSimplifyPruningRenamingProofs FlambdaSimplifySwitchRewriteSpec.
From OxCamlFlambda2Types Require FlambdaMeetClosuresProofs
  FlambdaMeetTermsProofs.
Import ListNotations.

(* ======================================================================== *)
(* The sets of what the slices answer are well formed                       *)
(* ======================================================================== *)

Lemma switch_of_arms_sets (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  sets_well_formed (switch_of_arms scrutinee arms) = true.
Proof. destruct arms as [ | [i ac] [ | arm arms]]; reflexivity. Qed.

Lemma bind_lets_sets
    (lets : list (variable * flambda_kind * prim * list simple)) :
  ∀ body : expr,
    sets_well_formed body = true ->
    sets_well_formed (bind_lets lets body) = true.
Proof.
  induction lets as [ | [[[x kind] p] args] lets IH]; intros body Hbody;
    [ exact Hbody | ].
  cbn [bind_lets]. apply IH. simpl. exact Hbody.
Qed.

Lemma merge_arms_sets (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) (e : expr) :
  merge_arms known fresh scrutinee arms = Some e ->
  sets_well_formed e = true.
Proof.
  unfold merge_arms.
  destruct (mergeable_arms arms) as [[cont num_args] | ]; [ | discriminate ].
  cbv beta iota zeta.
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test
  end; [ | discriminate ].
  match goal with
  | |- context [recognize_mergeable_arguments ?s ?n ?m ?rows] =>
    destruct (recognize_mergeable_arguments s n m rows) as [arguments | ]
  end; [ | discriminate ].
  cbv beta iota zeta.
  match goal with
  | |- context [rebuild_mergeable_arguments ?s ?m ?arguments] =>
    destruct (rebuild_mergeable_arguments s m arguments) as [[args m'] | ]
  end; [ | discriminate ].
  cbv beta iota zeta. intros [= <-]. apply bind_lets_sets. reflexivity.
Qed.

Lemma rebuild_switch_sets (known : computed) (fresh : list variable)
    (scrutinee : simple) (arms : list (Z * apply_cont)) :
  sets_well_formed (rebuild_switch known fresh scrutinee arms) = true.
Proof.
  unfold rebuild_switch. destruct arms as [ | arm arms]; [ reflexivity | ].
  destruct (merge_arms known fresh scrutinee (arm :: arms)) as [e | ]
    eqn:Hmerge.
  - exact (merge_arms_sets _ _ _ _ e Hmerge).
  - apply switch_of_arms_sets.
Qed.

(* ======================================================================== *)
(* The slices have what is asked of a rewrite of a [Switch]                 *)
(* ======================================================================== *)

Theorem simplify_switch_arms_rewrite_sound :
  simplify_switch_arms_rewrite_sound_statement.
Proof.
  intros meet rewrite Hmeet Hkeeps Hrewrite tenv s arms e He. split.
  - revert He. unfold simplify_switch_arms.
    destruct
      (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
      as [[[ty canonical] | ] | ]; try discriminate.
    cbv beta iota zeta.
    destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ];
      [ | discriminate ].
    intros [= <-]. apply switch_of_arms_sets.
  - intros Hfits Hmem.
    exact (simplify_switch_arms_refines meet rewrite tenv s arms e Hmeet
             Hkeeps Hrewrite Hfits Hmem He).
Qed.

Theorem simplify_switch_merging_rewrite_sound :
  simplify_switch_merging_rewrite_sound_statement.
Proof.
  intros meet rewrite fresh Hmeet Hkeeps Hrewrite tenv s arms e He. split.
  - revert He. unfold simplify_switch_merging.
    destruct
      (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
      as [[[ty canonical] | ] | ]; try discriminate.
    cbv beta iota zeta.
    destruct (simplify_arms meet rewrite tenv ty arms) as [arms' | ];
      [ | discriminate ].
    intros [= <-]. apply rebuild_switch_sets.
  - intros Hfits Hmem.
    apply (proj1 (expr_refines_under_valid_where_nothing_known _ _ _)).
    exact (simplify_switch_merging_refines meet rewrite [] fresh tenv s arms
             e Hmeet Hkeeps Hrewrite Hfits Hmem He).
Qed.

Theorem switch_arms_of_meet_to_prove_rewrite_sound :
  switch_arms_of_meet_to_prove_rewrite_sound_statement.
Proof.
  intros join fuel rewrite Hjoin Hrewrite.
  apply simplify_switch_arms_rewrite_sound; [ | | exact Hrewrite ].
  - exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound _ _ Hjoin)).
  - exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms _ _ Hjoin).
Qed.

Theorem switch_merging_of_meet_to_prove_rewrite_sound :
  switch_merging_of_meet_to_prove_rewrite_sound_statement.
Proof.
  intros join fuel rewrite fresh Hjoin Hrewrite.
  apply simplify_switch_merging_rewrite_sound; [ | | exact Hrewrite ].
  - exact (proj1 (FlambdaMeetClosuresProofs.meet_to_prove_sound _ _ Hjoin)).
  - exact (FlambdaMeetTermsProofs.meet_to_prove_keeps_terms _ _ Hjoin).
Qed.

(* ======================================================================== *)
(* Renaming: the second slice                                               *)
(* ======================================================================== *)

Theorem simplify_switch_arms_rewrite_equivariant :
  simplify_switch_arms_rewrite_equivariant_statement.
Proof.
  intros meet rewrite Hmeet Hrewrite r tenv scrutinee arms.
  exact
    (simplify_switch_arms_rename meet rewrite r tenv scrutinee arms Hmeet
       Hrewrite).
Qed.

Theorem switch_arms_of_meet_to_prove_rewrite_equivariant :
  switch_arms_of_meet_to_prove_rewrite_equivariant_statement.
Proof.
  intros join fuel rewrite Hjoin Hrewrite r tenv scrutinee arms.
  exact
    (simplify_switch_arms_of_meet_to_prove_rename join fuel rewrite r tenv
       scrutinee arms Hjoin Hrewrite).
Qed.

Theorem simplify_by_types_with_switch_arms_equivariant :
  simplify_by_types_with_switch_arms_equivariant_statement.
Proof.
  intros meet rewrite Hmeet Hrewrite.
  apply simplify_by_types_pruning_equivariant.
  exact (simplify_switch_arms_rewrite_equivariant meet rewrite Hmeet Hrewrite).
Qed.

(* ======================================================================== *)
(* Renaming, with names that are fixed                                      *)
(* ======================================================================== *)

Theorem simplify_switch_merging_fixed_names_rename :
  simplify_switch_merging_fixed_names_rename_statement.
Proof.
  intros meet rewrite r fresh tenv s arms Hmeet Hrewrite Hfixed.
  pose proof
    (simplify_switch_merging_rename meet rewrite r [] fresh tenv s arms Hmeet
       Hrewrite)
    as Hrename.
  assert (Hfresh : map (rename_variable r) fresh = fresh).
  { rewrite <- (map_id fresh) at 2. apply map_ext_in. exact Hfixed. }
  rewrite Hfresh in Hrename. exact Hrename.
Qed.
