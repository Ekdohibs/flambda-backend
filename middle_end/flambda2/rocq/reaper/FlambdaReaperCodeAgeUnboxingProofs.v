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

(* Proofs of the statements of FlambdaReaperCodeAgeUnboxingSpec.v. Nothing
   here needs review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaReaperRebuildSpec FlambdaCodeAgeCallsSpec FlambdaReaperCodeAgeSpec
  FlambdaReaperCodeAgeProofs FlambdaReaperCodeAgeUnboxingSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesHeapProofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesNeverProofs
  FlambdaReaperUnboxingRefinesUnitFlowProofs
  FlambdaReaperUnboxingRefinesClosuresProofs
  FlambdaReaperUnboxingRefinesClosuresCallsProofs FlambdaReaperTypesProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UHP := FlambdaReaperUnboxingRefinesHeapProofs.
Module UCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module UNP := FlambdaReaperUnboxingRefinesNeverProofs.
Module UUP := FlambdaReaperUnboxingRefinesUnitFlowProofs.
Module UXP := FlambdaReaperUnboxingRefinesClosuresProofs.
Module UXCP := FlambdaReaperUnboxingRefinesClosuresCallsProofs.
Module TP := FlambdaReaperTypesProofs.

(* ======================================================================== *)
(* Units on the machine                                                     *)
(* ======================================================================== *)

Theorem refines_up_to_heap_aged_of_defined :
  refines_up_to_heap_aged_of_defined_statement.
Proof.
  intros age_of s t Href Hnw.
  assert (Htnw : ~ behaves t Goes_wrong).
  { intros Ht. destruct Href as (_ & _ & Hwrong). exact (Hnw (Hwrong Ht)). }
  pose proof (behaves_aged_defined age_of s Hnw) as Hs.
  pose proof (behaves_aged_defined age_of t Htnw) as Ht.
  destruct Href as (H1 & H2 & H3). split; [ | split ].
  - intros k vs hp Hb. apply Hs, H1 in Hb.
    destruct Hb as (vs' & hp' & Hb & Hrel). exists vs', hp'.
    split; [ | exact Hrel ]. apply Ht. exact Hb.
  - intros k vs' hp' Hb. apply Ht, H2 in Hb.
    destruct Hb as [(vs & hp & Hb & Hrel) | Hb]; [ left | right ].
    + exists vs, hp. split; [ | exact Hrel ]. apply Hs. exact Hb.
    + apply Hs. exact Hb.
  - intros Hb. apply Ht, H3 in Hb. apply Hs. exact Hb.
Qed.


(* ======================================================================== *)
(* Local unboxing                                                           *)
(* ======================================================================== *)

Theorem aged_unbox_refines_rebuild : aged_unbox_refines_rebuild_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (URP.rebuild_unbox_refines_rebuild flag is_local u e T cannot_change
         code_deps dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5).
Qed.

Theorem aged_unbox_refines_observably : aged_unbox_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (URP.rebuild_unbox_refines_observably flag is_local u e T cannot_change
         dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5 H6 H7 H8 H9 H10
         H11).
Qed.

Theorem aged_unbox_refines_observably_local :
  aged_unbox_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (URP.rebuild_unbox_refines_observably_local flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11).
Qed.


(* ======================================================================== *)
(* Local unboxing, with the heap                                            *)
(* ======================================================================== *)

Theorem aged_unbox_refines_rebuild_heap :
  aged_unbox_refines_rebuild_heap_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UHP.rebuild_unbox_refines_rebuild_heap flag is_local u e T
         cannot_change code_deps dominated to_unbox preserve_direct_calls H1 H2
         H3 H4 H5).
Qed.

Theorem aged_unbox_refines_up_to_heap : aged_unbox_refines_up_to_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 Hdef.
  apply (refines_up_to_heap_aged_of_defined age_of); [ | exact Hdef ].
  exact (UHP.rebuild_unbox_refines_up_to_heap flag is_local u e T cannot_change
         dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5 H6 H7 H8 H9 H10
         H11).
Qed.

Theorem aged_unbox_refines_up_to_heap_local :
  aged_unbox_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 Hdef.
  apply (refines_up_to_heap_aged_of_defined age_of); [ | exact Hdef ].
  exact (UHP.rebuild_unbox_refines_up_to_heap_local flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11).
Qed.


(* ======================================================================== *)
(* Unboxing across calls                                                    *)
(* ======================================================================== *)

Theorem aged_unbox_fixed_refines_cc : aged_unbox_fixed_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_refines_cc flag is_local u e T cannot_change
         dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5).
Qed.

Theorem aged_unbox_fixed_refines_observably :
  aged_unbox_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_refines_observably flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_fixed_refines_observably_local :
  aged_unbox_fixed_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_refines_observably_local flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_fixed_refines_cc_never_returns :
  aged_unbox_fixed_refines_cc_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_refines_cc_never_returns flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5).
Qed.

Theorem aged_unbox_fixed_observably_never_returns :
  aged_unbox_fixed_observably_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_observably_never_returns flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_fixed_local_never_returns :
  aged_unbox_fixed_local_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UCP.rebuild_unbox_fixed_local_never_returns flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.


(* ======================================================================== *)
(* Code that never returns, of the first rebuilt unit                       *)
(* ======================================================================== *)

Theorem aged_unbox_nr_ind_refines_cc : aged_unbox_nr_ind_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UNP.rebuild_unbox_nr_ind_refines_cc flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8).
Qed.

Theorem aged_unbox_nr_ind_refines_observably :
  aged_unbox_nr_ind_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UNP.rebuild_unbox_nr_ind_refines_observably flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16).
Qed.

Theorem aged_unbox_nr_ind_refines_observably_local :
  aged_unbox_nr_ind_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UNP.rebuild_unbox_nr_ind_refines_observably_local flag is_local u e T
         T1 cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16).
Qed.


(* ======================================================================== *)
(* The checks of the flow read off the unit's own analysis                  *)
(* ======================================================================== *)

Theorem aged_unbox_unit_flow3_refines_cc :
  aged_unbox_unit_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_flow3_refines_cc flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9).
Qed.

Theorem aged_unbox_unit_flow3_refines_observably :
  aged_unbox_unit_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 H17 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_flow3_refines_observably flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17).
Qed.

Theorem aged_unbox_unit_flow3_refines_observably_local :
  aged_unbox_unit_flow3_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 H17 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_flow3_refines_observably_local flag is_local u e
         T T1 cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17).
Qed.

Theorem aged_unbox_unit_never_refines_cc :
  aged_unbox_unit_never_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_never_refines_cc flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9).
Qed.

Theorem aged_unbox_unit_never_refines_observably :
  aged_unbox_unit_never_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14
    H15 H16 H17 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_never_refines_observably flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17).
Qed.

Theorem aged_unbox_unit_never_refines_observably_local :
  aged_unbox_unit_never_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14
    H15 H16 H17 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UUP.rebuild_unbox_unit_never_refines_observably_local flag is_local u e
         T T1 cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17).
Qed.


(* ======================================================================== *)
(* Changes of representation of sets of closures                            *)
(* ======================================================================== *)

Theorem aged_repr_fixed_refines_cc : aged_repr_fixed_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_refines_cc flag is_local u e T cannot_change
         dominated to_unbox tcr preserve_direct_calls H1 H2 H3 H4 H5).
Qed.

Theorem aged_repr_fixed_refines_observably :
  aged_repr_fixed_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_refines_observably flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_repr_fixed_refines_observably_local :
  aged_repr_fixed_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_refines_observably_local flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_repr_fixed_refines_cc_never_returns :
  aged_repr_fixed_refines_cc_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_refines_cc_never_returns flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5).
Qed.

Theorem aged_repr_fixed_observably_never_returns :
  aged_repr_fixed_observably_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_observably_never_returns flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_repr_fixed_local_never_returns :
  aged_repr_fixed_local_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXP.rebuild_repr_fixed_local_never_returns flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13).
Qed.


(* ======================================================================== *)
(* Changes of representation, on the wider fragment                         *)
(* ======================================================================== *)

Theorem aged_repr_calls_refines_cc : aged_repr_calls_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_refines_cc flag is_local u e T cannot_change
         dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3 H4 H5).
Qed.

Theorem aged_repr_calls_observably : aged_repr_calls_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_observably flag is_local u e T cannot_change
         dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3 H4 H5 H6 H7 H8
         H9 H10 H11 H12 H13).
Qed.

Theorem aged_repr_calls_local : aged_repr_calls_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_local flag is_local u e T cannot_change
         dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3 H4 H5 H6 H7 H8
         H9 H10 H11 H12 H13).
Qed.

Theorem aged_repr_calls_with_types_refines_cc :
  aged_repr_calls_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_with_types_refines_cc flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5).
Qed.

Theorem aged_repr_calls_with_types_observably :
  aged_repr_calls_with_types_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_with_types_observably flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.

Theorem aged_repr_calls_with_types_local :
  aged_repr_calls_with_types_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (UXCP.rebuild_repr_calls_with_types_local flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.


(* ======================================================================== *)
(* The rebuilds of the reaper with types                                    *)
(* ======================================================================== *)

Theorem aged_unbox_with_types_refines_rebuild :
  aged_unbox_with_types_refines_rebuild_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_rebuild flag is_local u e T
         cannot_change code_deps dominated to_unbox preserve_direct_calls H1 H2
         H3 H4 H5).
Qed.

Theorem aged_unbox_with_types_refines_observably :
  aged_unbox_with_types_refines_observably_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_observably flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_with_types_refines_observably_local :
  aged_unbox_with_types_refines_observably_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_observably_local flag is_local u e
         T cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_with_types_refines_rebuild_heap :
  aged_unbox_with_types_refines_rebuild_heap_statement.
Proof.
  intros flag is_local u e T cannot_change code_deps dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_rebuild_heap flag is_local u e T
         cannot_change code_deps dominated to_unbox preserve_direct_calls H1 H2
         H3 H4 H5).
Qed.

Theorem aged_unbox_with_types_refines_up_to_heap :
  aged_unbox_with_types_refines_up_to_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_up_to_heap_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_up_to_heap flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_with_types_refines_up_to_heap_local :
  aged_unbox_with_types_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    Hdef.
  apply (refines_up_to_heap_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_with_types_refines_up_to_heap_local flag is_local u e
         T cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13).
Qed.

Theorem aged_unbox_fixed_with_types_refines_cc_nr :
  aged_unbox_fixed_with_types_refines_cc_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_fixed_with_types_refines_cc_nr flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6).
Qed.

Theorem aged_unbox_fixed_with_types_observably_nr :
  aged_unbox_fixed_with_types_observably_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_fixed_with_types_observably_nr flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.

Theorem aged_unbox_fixed_with_types_local_nr :
  aged_unbox_fixed_with_types_local_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_fixed_with_types_local_nr flag is_local u e T
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.

Theorem aged_repr_fixed_with_types_refines_cc_nr :
  aged_repr_fixed_with_types_refines_cc_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_repr_fixed_with_types_refines_cc_nr flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5).
Qed.

Theorem aged_repr_fixed_with_types_observably_nr :
  aged_repr_fixed_with_types_observably_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_repr_fixed_with_types_observably_nr flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.

Theorem aged_repr_fixed_with_types_local_nr :
  aged_repr_fixed_with_types_local_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox tcr
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_repr_fixed_with_types_local_nr flag is_local u e T
         cannot_change dominated to_unbox tcr preserve_direct_calls nr H1 H2 H3
         H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15).
Qed.

Theorem aged_unbox_nr_ind_with_types_refines_cc :
  aged_unbox_nr_ind_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_nr_ind_with_types_refines_cc flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9).
Qed.

Theorem aged_unbox_nr_ind_with_types_observably :
  aged_unbox_nr_ind_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 H17 H18 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_nr_ind_with_types_observably flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17 H18).
Qed.

Theorem aged_unbox_nr_ind_with_types_local :
  aged_unbox_nr_ind_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls nr age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13
    H14 H15 H16 H17 H18 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_nr_ind_with_types_local flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls nr H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17 H18).
Qed.

Theorem aged_unbox_unit_never_with_types_refines_cc :
  aged_unbox_unit_never_with_types_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 Hdef.
  apply (refines_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_unit_never_with_types_refines_cc flag is_local u e T
         T1 cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10).
Qed.

Theorem aged_unbox_unit_never_with_types_observably :
  aged_unbox_unit_never_with_types_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14
    H15 H16 H17 H18 H19 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_unit_never_with_types_observably flag is_local u e T
         T1 cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4
         H5 H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17 H18 H19).
Qed.

Theorem aged_unbox_unit_never_with_types_local :
  aged_unbox_unit_never_with_types_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox
    preserve_direct_calls age_of H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13 H14
    H15 H16 H17 H18 H19 Hdef.
  apply (refines_observably_aged_of_defined age_of); [ | exact Hdef ].
  exact (TP.rebuild_unbox_unit_never_with_types_local flag is_local u e T T1
         cannot_change dominated to_unbox preserve_direct_calls H1 H2 H3 H4 H5
         H6 H7 H8 H9 H10 H11 H12 H13 H14 H15 H16 H17 H18 H19).
Qed.
