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


(* The proofs of FlambdaReaperUnboxingRefinesCallsHeapSpec.v. Nothing here
   needs review.

   The relation of FlambdaReaperUnboxingRefinesCallsSimProofs.v ([uc])
   relates the two rebuilt terms with the primitives of the heap: a [Let]
   of one, in both terms, is related by [uc_let_heap], and the states
   related have the same heap (not the empty heap). Its construction
   (FlambdaReaperUnboxingRefinesCallsProofs.rebuild_uc_h) holds for the
   edges of the units with the primitives of the heap, which
   FlambdaReaperFlowProofs.traverse_unit_facts_gen gives of a unit of
   [in_heap_fragment]; [rebuild_unbox_modes_with_refines_cc_gen] there is
   the statement against the change of calling conventions for either
   fragment and any rewriting of what the parameters state. The rest
   composes it with the statements up to the heap of the change of
   calling conventions (FlambdaReaperCallingConventionsProofs.v and
   FlambdaReaperTypesProofs.v), as
   FlambdaReaperUnboxingRefinesHeapProofs.v does for the local
   unboxing. The statements of calls whose code is not known that never
   return are those of FlambdaReaperUnboxingRefinesNeverProofs.v
   ([rebuild_unbox_nr_ind_refines_cc_gen]), with types
   FlambdaReaperTypesProofs.v ([rebuild_unbox_nr_ind_with_types_cc_gen]),
   and FlambdaReaperUnboxingRefinesUnitFlowProofs.v
   ([unit_never_covered]), on the fragment with the primitives of the
   heap. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaReaperLeastSpec FlambdaUnusedStaticsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec FlambdaReaperTypes
  FlambdaReaperUnboxingRefinesCallsHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaStaticsInScopeSpec FlambdaReaperNeverReturnsSpec
  FlambdaReaperFlowProofs FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventionsProofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesHeapProofs
  FlambdaReaperUnboxingRefinesNeverProofs
  FlambdaReaperUnboxingRefinesUnitFlowProofs FlambdaReaperTypesProofs.
Import ListNotations.

Module URCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module URHP := FlambdaReaperUnboxingRefinesHeapProofs.
Module CCP := FlambdaReaperCallingConventionsProofs.
Module TP := FlambdaReaperTypesProofs.
Module URNP := FlambdaReaperUnboxingRefinesNeverProofs.
Module UFP := FlambdaReaperUnboxingRefinesUnitFlowProofs.

(* ======================================================================== *)
(* Against the change of calling conventions                                *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_refines_cc_heap_never_returns :
  rebuild_unbox_fixed_refines_cc_heap_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hfrag Hclosed Hcalls Hcheck.
  apply (URCP.rebuild_unbox_modes_with_refines_cc_gen true None flag is_local
           u e T cannot_change dominated to_unbox pdc nr);
    [ intros t Ht; discriminate Ht
    | apply FlambdaReaperRebuildProofs.no_types_conts | exact Hslots
    | exact Hfrag | exact Hclosed | exact Hcalls | exact Hcheck ].
Qed.

Theorem rebuild_unbox_fixed_refines_cc_heap :
  rebuild_unbox_fixed_refines_cc_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_refines_cc_heap_never_returns flag is_local u e
           T cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

(* ======================================================================== *)
(* Against the unit, up to the heap                                         *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_heap_local_never_returns :
  rebuild_unbox_fixed_heap_local_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hcalls Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
                     cannot_change (unit_code_deps flag u e) pdc e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (CCP.rebuild_cc_refines_up_to_heap_local flag is_local u e T
             cannot_change pdc Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs
             Hexact Hccc Hwid).
  - exact (rebuild_unbox_fixed_refines_cc_heap_never_returns flag is_local u
             e T cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
             Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_fixed_up_to_heap_never_returns :
  rebuild_unbox_fixed_up_to_heap_never_returns_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc nr Hslots
    Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hcalls Hcheck.
  exact (rebuild_unbox_fixed_heap_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_fixed_refines_up_to_heap_local :
  rebuild_unbox_fixed_refines_up_to_heap_local_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_heap_local_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

Theorem rebuild_unbox_fixed_refines_up_to_heap :
  rebuild_unbox_fixed_refines_up_to_heap_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox pdc.
  exact (rebuild_unbox_fixed_up_to_heap_never_returns flag is_local u e T
           cannot_change dominated to_unbox pdc (fun _ => false)).
Qed.

(* ======================================================================== *)
(* The reaper with types                                                    *)
(* ======================================================================== *)

Theorem rebuild_unbox_fixed_with_types_cc_heap_nr :
  rebuild_unbox_fixed_with_types_cc_heap_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b nr Hslots
    Hfrag Hdist Hclosed Hcalls Hcheck.
  exact (URCP.rebuild_unbox_modes_with_refines_cc_gen true
           (subkinds_of_context
              (plain_context is_local (traverse_unit flag u e) T [])
              (unit_code_deps flag u e) e)
           flag is_local u e T cannot_change dominated to_unbox b nr
           (TP.subkinds_of_context_unused is_local _ T [] _ e)
           (TP.subkinds_of_context_conts _ _ _ Hdist) Hslots Hfrag Hclosed
           Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_fixed_with_types_heap_local_nr :
  rebuild_unbox_fixed_with_types_heap_local_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b nr Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hcalls
    Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (rebuild_unit_cc_with_types_from_tables is_local
                     (traverse_unit flag u e) T [] cannot_change
                     (unit_code_deps flag u e) b e)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (TP.rebuild_cc_with_types_refines_up_to_heap_local flag is_local u
             e T [] cannot_change b Hleast Hslots Hfrag Hany Hrk Hwf Hdist
             Hone Hclosed Hlrs Hexact Hccc Hwid).
  - exact (rebuild_unbox_fixed_with_types_cc_heap_nr flag is_local u e T
             cannot_change dominated to_unbox b nr Hslots Hfrag Hdist Hclosed
             Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_fixed_with_types_up_to_heap_nr :
  rebuild_unbox_fixed_with_types_up_to_heap_nr_statement.
Proof.
  intros flag is_local u e T cannot_change dominated to_unbox b nr Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hcalls Hcheck.
  exact (rebuild_unbox_fixed_with_types_heap_local_nr flag is_local u e T
           cannot_change dominated to_unbox b nr Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hcalls Hcheck).
Qed.

(* ======================================================================== *)
(* Calls whose code is not known that never return                          *)
(* ======================================================================== *)

Theorem rebuild_unbox_nr_ind_cc_heap : rebuild_unbox_nr_ind_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (URNP.rebuild_unbox_nr_ind_refines_cc_gen true flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind_heap_local :
  rebuild_unbox_nr_ind_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (first_rebuild flag u e T cannot_change pdc)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (CCP.rebuild_cc_refines_up_to_heap_local flag is_local u e T
             cannot_change pdc Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs
             Hexact Hccc Hwid).
  - exact (rebuild_unbox_nr_ind_cc_heap flag is_local u e T T1 cannot_change
             dominated to_unbox pdc nr Hslots Hfrag Hclosed Hfrag1 Hclosed1
             Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_nr_ind_up_to_heap :
  rebuild_unbox_nr_ind_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_cc_heap :
  rebuild_unbox_unit_never_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_cc_heap flag is_local u e T T1 cannot_change
           dominated to_unbox pdc _ Hslots Hfrag Hclosed Hfrag1 Hclosed1
           Hlrs1 (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov)
           Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_heap_local :
  rebuild_unbox_unit_never_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_up_to_heap :
  rebuild_unbox_unit_never_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_up_to_heap flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hproj Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

(* With types. *)

Theorem rebuild_unbox_nr_ind_with_types_cc_heap :
  rebuild_unbox_nr_ind_with_types_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (TP.rebuild_unbox_nr_ind_with_types_cc_gen true flag is_local u e T
           T1 cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind_with_types_heap_local :
  rebuild_unbox_nr_ind_with_types_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  apply (URHP.refines_up_to_heap_refines _
           (State (first_rebuild_with_types flag is_local u e T
                     cannot_change b)
              [] [] empty_static_env empty_heap no_reads)).
  - exact (TP.rebuild_cc_with_types_refines_up_to_heap_local flag is_local u
             e T [] cannot_change b Hleast Hslots Hfrag Hany Hrk Hwf Hdist
             Hone Hclosed Hlrs Hexact Hccc Hwid).
  - exact (rebuild_unbox_nr_ind_with_types_cc_heap flag is_local u e T T1
             cannot_change dominated to_unbox b nr Hslots Hfrag Hdist Hclosed
             Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_nr_ind_with_types_up_to_heap :
  rebuild_unbox_nr_ind_with_types_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b nr Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind_with_types_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox b nr Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_with_types_cc_heap :
  rebuild_unbox_unit_never_with_types_cc_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hslots
    Hfrag Hdist Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_with_types_cc_heap flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hslots Hfrag Hdist Hclosed
           Hfrag1 Hclosed1 Hlrs1
           (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_with_types_heap_local :
  rebuild_unbox_unit_never_with_types_heap_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_with_types_heap_local flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hfrag Hany Hrk
           Hwf Hdist Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_with_types_up_to_heap :
  rebuild_unbox_unit_never_with_types_up_to_heap_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox b Hleast
    Hslots Hproj Hfrag Hany Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid
    Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_with_types_up_to_heap flag is_local u e T T1
           cannot_change dominated to_unbox b _ Hleast Hslots Hproj Hfrag Hany
           Hrk Hwf Hdist Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (UFP.unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Theorem calls_heap_example : calls_heap_example_statement.
Proof.
  (* The theorems are applied to the graph and the code deps as their
     statements name them, not as the example does, so that no conversion
     computes them. *)
  assert (E1 : CallsHeapExample.facts
               = traverse_unit Auto Examples.interface
                   CallsHeapExample.unit_expr)
    by reflexivity.
  assert (E2 : CallsHeapExample.code_deps
               = unit_code_deps Auto Examples.interface
                   CallsHeapExample.unit_expr)
    by reflexivity.
  assert (Hcalls : calls_unboxing CallsHeapExample.facts
                     CallsHeapExample.tables CallsHeapExample.cannot_change
                     CallsHeapExample.code_deps CallsHeapExample.dominated
                     CallsHeapExample.to_unbox CallsHeapExample.unit_expr
                   = true)
    by (vm_compute; reflexivity).
  assert (Hreads : check_unboxed_reads CallsHeapExample.facts
                     CallsHeapExample.tables CallsHeapExample.dominated
                     CallsHeapExample.to_unbox
                   = true)
    by (vm_compute; reflexivity).
  assert (Hfixed :
    rebuild_unit_unbox_fixed CallsHeapExample.facts CallsHeapExample.tables
      CallsHeapExample.cannot_change CallsHeapExample.code_deps
      CallsHeapExample.dominated CallsHeapExample.to_unbox true
      CallsHeapExample.unit_expr
    = CallsHeapExample.rebuilt)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_heap_fragment CallsHeapExample.unit_expr = true
    ∧ any_kind_arguments CallsHeapExample.unit_expr = true
    ∧ results_kinded [] [] CallsHeapExample.unit_expr = true
    ∧ well_formed_unit Examples.interface CallsHeapExample.unit_expr = true
    ∧ exits_one_value Examples.interface CallsHeapExample.unit_expr = true
    ∧ check_closed Examples.no_local CallsHeapExample.facts
        CallsHeapExample.tables
      = true
    ∧ check_field_used_exact Examples.no_local CallsHeapExample.facts
        CallsHeapExample.tables
      = true
    ∧ check_cc_closed CallsHeapExample.facts CallsHeapExample.tables
        CallsHeapExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        CallsHeapExample.facts CallsHeapExample.tables
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid).
  assert (Hslots : is_local_slots_only Examples.no_local)
    by (intros f Hf; discriminate Hf).
  split; [ exact Hfrag | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | split; [ exact Hreads | ] ].
  split; [ exact Hfixed | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  rewrite <- Hfixed.
  rewrite E1, E2 in *.
  split.
  - exact (rebuild_unbox_fixed_refines_cc_heap Auto Examples.no_local
             Examples.interface CallsHeapExample.unit_expr
             CallsHeapExample.tables CallsHeapExample.cannot_change
             CallsHeapExample.dominated CallsHeapExample.to_unbox true Hslots
             Hfrag Hclosed Hcalls Hreads).
  - exact (rebuild_unbox_fixed_refines_up_to_heap Auto Examples.no_local
             Examples.interface CallsHeapExample.unit_expr
             CallsHeapExample.tables CallsHeapExample.cannot_change
             CallsHeapExample.dominated CallsHeapExample.to_unbox true Hslots
             (or_introl (fun f => eq_refl)) Hfrag Hany Hrk Hwf Hone Hclosed
             Hexact Hccc Hwid Hcalls Hreads).
Qed.
