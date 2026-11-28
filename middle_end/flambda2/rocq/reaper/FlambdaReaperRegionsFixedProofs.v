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

(* Proofs of the statements of FlambdaReaperRegionsFixedSpec.v.

   This file needs no review: what it proves is stated there. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaPrimitiveModel
  FlambdaModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNameAbstractionSpec
  FlambdaRenameApartSpec FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFieldsSpec FlambdaRegionsSpec
  FlambdaRegionsErasureSpec FlambdaRegionsCheckSpec FlambdaReaperRegionsSpec
  FlambdaReaperRegionsUnboxSpec FlambdaReaperRegionsFixedSpec.
From OxCamlFlambda2 Require FlambdaReaperSolutionProofs
  FlambdaReaperLeastProofs FlambdaReaperRegionsProofs
  FlambdaReaperRegionsCcProofs
  FlambdaReaperRegionsUnboxProofs FlambdaReaperRegionsUnboxKeepsProofs
  FlambdaReaperUnboxingRefinesCallsProofs.
Import ListNotations.

Module RUP := FlambdaReaperRegionsUnboxProofs.
Module RCP := FlambdaReaperUnboxingRefinesCallsProofs.

(* ======================================================================== *)
(* The completed tables                                                     *)
(* ======================================================================== *)

Theorem alias_consts_rebuild : alias_consts_rebuild_statement.
Proof.
  intros fixed is_local F T cannot_change code_deps dominated to_unbox tcr b
    e.
  reflexivity.
Qed.

(* ======================================================================== *)
(* The commutation, and what the rebuild keeps                              *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_erase_commutes :
  rebuild_unbox_fixed_erase_commutes_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b Hloc Hnu.
  pose proof Hloc as Hloc'. unfold regions_local in Hloc'.
  apply andb_true_iff in Hloc' as [Hpl _].
  rewrite (RUP.unboxed_fields_erase flag u e T dominated to_unbox _ Hpl).
  destruct (unboxed_fields_of_names (traverse_unit flag u e) T
              (bound_code_my_closures e) dominated to_unbox
              (fresh_name (names_of e)))
    as [ub next] eqn:Hub.
  exact (RUP.rebuild_unbox_mode_erase_commutes_with true flag u e T
           (solution_of_tables T) cannot_change code_deps dominated to_unbox
           b ub next Hloc Hnu (fun _ _ => eq_refl) eq_refl eq_refl Hub).
Qed.

Theorem rebuild_unbox_fixed_keeps_erasable :
  rebuild_unbox_fixed_keeps_erasable_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b.
  exact (FlambdaReaperRegionsUnboxKeepsProofs.rebuild_unbox_mode_erasable
           true flag u e T cannot_change code_deps dominated to_unbox b).
Qed.

Theorem rebuild_unbox_fixed_keeps_fresh :
  rebuild_unbox_fixed_keeps_fresh_statement.
Proof.
  intros flag u e T cannot_change code_deps dominated to_unbox b base.
  exact (FlambdaReaperRegionsUnboxKeepsProofs.rebuild_unbox_mode_fresh true
           flag u e T cannot_change code_deps dominated to_unbox b base).
Qed.

(* ======================================================================== *)
(* The refinement on the machine of regions                                 *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_regions_refines :
  rebuild_unbox_fixed_regions_refines_statement.
Proof.
  intros flag is_local u e T cannot_change D U b cert base Hloc Hnu Hnames
    Hslots Hfrag Hdc Hrk Hwf Hexits Hcl Hlrs Hexact Hcc Hwid Hcalls Hreads
    Hfe Hee Hcheck.
  pose proof Hloc as Hloc'. unfold regions_local in Hloc'.
  apply andb_true_iff in Hloc' as [Hpl _].
  apply (FlambdaReaperRegionsProofs.regions_refine_of_erased cert base e);
    try assumption.
  - apply rebuild_unbox_fixed_keeps_fresh; assumption.
  - apply rebuild_unbox_fixed_keeps_erasable; assumption.
  - unfold erased_initial_state.
    rewrite FlambdaReaperRegionsProofs.erase_statics_empty.
    destruct (unboxed_fields_of_names (traverse_unit flag u e) T
                (bound_code_my_closures e) D U (fresh_name (names_of e)))
      as [ub next] eqn:Hub.
    unfold rebuild_unit_unbox_fixed, rebuild_unit_unbox_repr_fixed.
    rewrite (RUP.rebuild_unbox_mode_erase_commutes_with true flag u e T
               (solution_of_tables T) cannot_change (unit_code_deps flag u e)
               D U b ub next Hloc Hnu (fun _ _ => eq_refl) eq_refl eq_refl
               Hub).
    pose proof (RCP.rebuild_unbox_fixed_refines_observably_local flag
                  is_local u (erase [] e) (alias_consts T) cannot_change D U b
                  Hslots Hfrag Hdc Hrk Hwf Hexits Hcl Hlrs Hexact Hcc Hwid
                  Hcalls Hreads)
      as H.
    unfold program_refines_observably in H.
    rewrite (FlambdaReaperRegionsCcProofs.unit_code_deps_erase flag u e Hpl)
      in H.
    unfold rebuild_unit_unbox_fixed, rebuild_unit_unbox_repr_fixed in H.
    rewrite (alias_consts_rebuild true (fun _ => false)) in H.
    unfold rebuild_unit_unbox_repr_mode, unit_unboxed_fields in H.
    rewrite Hnames, (RUP.unboxed_fields_erase flag u e T D U _ Hpl), Hub
      in H.
    exact H.
Qed.

Theorem least_rebuild_unbox_fixed_regions_refines :
  least_rebuild_unbox_fixed_regions_refines_statement.
Proof.
  intros flag is_local u e T max_unbox_size lambda_lifting b cert base Hloc
    Hnames Hslots Hfrag Hdc Hrk Hwf Hexits Hleast Hcl Hlrs Hexact Hcc Hwid
    Hcalls Hreads Hfe Hee Hcheck.
  apply (rebuild_unbox_fixed_regions_refines flag is_local u e T _ _ _ b cert
           base); try assumption.
  apply RUP.regions_not_unboxed_any_source.
  apply (RUP.closed_regions_any_source flag is_local u e T Hloc Hfrag).
  exact (FlambdaReaperSolutionProofs.check_closed_sound is_local _ T
           (proj1 Hleast)).
Qed.

(* ======================================================================== *)
(* Example                                                                  *)
(* ======================================================================== *)

Import FlambdaReaperRegionsSpec.RegionsExamples RegionsUnboxExamples.

Theorem regions_fixed_example : regions_fixed_example_statement.
Proof.
  assert (Hleast : least_solution (fun _ => false)
                     (traverse_unit Auto interface local_pair)
                     local_pair_tables)
    by (apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
        vm_compute; reflexivity).
  assert (Hreb : rebuild_unit_unbox_fixed
                   (traverse_unit Auto interface local_pair)
                   local_pair_tables
                   (derive_cannot_change
                      (traverse_unit Auto interface local_pair)
                      local_pair_tables)
                   (unit_code_deps Auto interface local_pair)
                   (derive_dominated local_pair_tables) local_pair_to_unbox
                   true local_pair
                 = local_pair_rebuilt)
    by (vm_compute; reflexivity).
  do 4 (split; [ vm_compute; reflexivity | ]).
  split; [ exact Hreb | ].
  rewrite <- Hreb. unfold local_pair_to_unbox.
  apply (least_rebuild_unbox_fixed_regions_refines Auto (fun _ => false)
           interface local_pair local_pair_tables 10 false true cert base);
    try exact Hleast; try (vm_compute; reflexivity);
    try (fold local_pair_to_unbox; rewrite Hreb; vm_compute; reflexivity).
  exact FlambdaReaperRegionsProofs.no_local_slots.
Qed.
