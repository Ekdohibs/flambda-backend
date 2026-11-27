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

(* The rebuilds with unboxing on the machine of the core that is given a
   code age relation (FlambdaCodeAgeCallsSpec.step_aged), as
   FlambdaReaperCodeAgeSpec.least_rebuild_refines_aged_statement states
   the rebuild without unboxing on it.

   This file is to be reviewed: [program_defined], [program_refines_aged],
   [refines_up_to_heap_aged] and the statements. The proofs are in
   FlambdaReaperCodeAgeUnboxingProofs.v, which needs no review.

   Each statement is one of the statements of the rebuilds with unboxing
   (FlambdaReaperUnboxingRefines*Spec.v and FlambdaReaperTypesSpec.v),
   named after it with [aged_] in the place of [rebuild_], with the same
   premises and one more: the source of its refinement does not go wrong
   on [step] ([program_defined]). Its refinement then holds on the
   machine, whatever the relation [age_of]. Each is the statement
   composed with FlambdaReaperCodeAgeSpec.refines_aged_of_defined_statement,
   with refines_observably_aged_of_defined_statement, or, for the
   comparison up to the heap, with
   [refines_up_to_heap_aged_of_defined_statement]: a state that does not
   go wrong on [step] makes no direct call of a closure of another code
   than the code named, and the runs of the machine from it are those of
   [step]. No lifting needs more than that composition.

   The source of a refinement up to observation (or up to the heap) is
   the unit itself. That of the other statements ([_refines_cc],
   [_refines_rebuild], [_refines_rebuild_heap]) is the unit rebuilt
   without unboxing (the rebuild with changes of calling convention, the
   plain rebuild, or the first rebuilt unit of the fragments of the
   flow), and the premise is of that rebuilt unit; where it refines the
   unit up to observation, it does not go wrong where the unit does not
   (the last conjunct of FlambdaReaperRebuildSpec.refines_observably).
   The variants that keep the names of an earlier fix ([_tail_]) are
   these statements and are not repeated.

   What the premise leaves out is a unit that runs a direct call of a
   closure of a newer version of the code named: such a call goes wrong
   on [step]. FlambdaReaperCodeAgeVersionsSpec.v is of those units. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLocalFields FlambdaReaperLeastSpec
  FlambdaReaperUnboxingRefinesHeapSpec
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaCodeAgeCallsSpec
  FlambdaReaperCodeAgeSpec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec FlambdaStaticsInScopeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaCodeAgeRelation.
Import ListNotations.

(* ======================================================================== *)
(* Units on the machine                                                     *)
(* ======================================================================== *)

(* [e], run on its own, does not go wrong on [step]. *)
Definition program_defined (e : expr) : Prop :=
  ~ behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong.

(* FlambdaUnusedStaticsSpec.program_refines on the machine. *)
Definition program_refines_aged (age_of : static_env -> Code_age_relation.t)
    (source target : expr) : Prop :=
  refines_aged age_of
    (State source [] [] empty_static_env empty_heap no_reads)
    (State target [] [] empty_static_env empty_heap no_reads).

(* FlambdaReaperRebuildSpec.refines_up_to_heap, word for word, with
   [behaves_aged age_of] in the place of [behaves]. *)
Definition refines_up_to_heap_aged
    (age_of : static_env -> Code_age_relation.t) (source target : state)
    : Prop :=
  (∀ (k : continuation) (vs : list value) (hp : heap),
     behaves_aged age_of source (Exits k vs hp) ->
     ∃ (vs' : list value) (hp' : heap),
       behaves_aged age_of target (Exits k vs' hp')
       ∧ exits_observed_equal vs hp vs' hp')
  ∧ (∀ (k : continuation) (vs' : list value) (hp' : heap),
       behaves_aged age_of target (Exits k vs' hp') ->
       (∃ (vs : list value) (hp : heap),
          behaves_aged age_of source (Exits k vs hp)
          ∧ exits_observed_equal vs hp vs' hp')
       ∨ behaves_aged age_of source Goes_wrong)
  ∧ (behaves_aged age_of target Goes_wrong ->
     behaves_aged age_of source Goes_wrong).

Definition program_refines_up_to_heap_aged
    (age_of : static_env -> Code_age_relation.t) (source target : expr)
    : Prop :=
  refines_up_to_heap_aged age_of
    (State source [] [] empty_static_env empty_heap no_reads)
    (State target [] [] empty_static_env empty_heap no_reads).

(* As FlambdaReaperCodeAgeSpec.refines_observably_aged_of_defined_statement
   says of the refinement up to observation. *)
Definition refines_up_to_heap_aged_of_defined_statement : Prop :=
  ∀ (age_of : static_env -> Code_age_relation.t) (source target : state),
    refines_up_to_heap source target ->
    ~ behaves source Goes_wrong ->
    refines_up_to_heap_aged age_of source target.


(* ======================================================================== *)
(* Local unboxing                                                           *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesSpec.v: the rebuild with unboxing
   refines the rebuild without, and the unit up to observation, with no
   local field or no projection, or with local fields. *)

Definition aged_unbox_refines_rebuild_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* Local unboxing, with the heap                                            *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesHeapSpec.v: the same for the units with
   primitives of the heap, compared up to the heap. *)

Definition aged_unbox_refines_rebuild_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_refines_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_up_to_heap_aged age_of e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_refines_up_to_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_up_to_heap_aged age_of e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* Unboxing across calls                                                    *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesCallsSpec.v: the rebuild with unboxing
   in the mode of the fix, with no code taken to never return, and with
   the code [nr]. *)

Definition aged_unbox_fixed_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_refines_cc_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_observably_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_local_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* Code that never returns, of the first rebuilt unit                       *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesNeverSpec.v. *)

Definition aged_unbox_nr_ind_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (first_rebuild flag u e T cannot_change preserve_direct_calls) ->
    program_refines_aged age_of
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_nr_ind_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_nr_ind_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* The checks of the flow read off the unit's own analysis                  *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesUnitFlowSpec.v: with the reach of the
   flow, and with the code that never returns of the unit. *)

Definition aged_unbox_unit_flow3_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (first_rebuild flag u e T cannot_change preserve_direct_calls) ->
    program_refines_aged age_of
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_unit_flow3_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_unit_flow3_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_flow3_nr nr (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_unit_never_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (first_rebuild flag u e T cannot_change preserve_direct_calls) ->
    program_refines_aged age_of
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_unit_never_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_unit_never_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables T)
         (unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
      dominated to_unbox e
    = true ->
    used_reach_covered (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).


(* ======================================================================== *)
(* Changes of representation of sets of closures                            *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesClosuresSpec.v: with no code taken to
   never return, and with the code [nr]. *)

Definition aged_repr_fixed_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_fixed_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_fixed_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_fixed_refines_cc_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_unboxing_nr nr is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_fixed_observably_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing_nr nr is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_fixed_local_never_returns_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_unboxing_nr nr is_local (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).


(* ======================================================================== *)
(* Changes of representation, on the wider fragment                         *)
(* ======================================================================== *)

(* FlambdaReaperUnboxingRefinesClosuresCallsSpec.v: the rebuild without
   types, and the rebuild of the reaper with types. *)

Definition aged_repr_calls_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_calls_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_calls_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed is_local (traverse_unit flag u e) T
         cannot_change (unit_code_deps flag u e) dominated to_unbox tcr
         preserve_direct_calls e).

Definition aged_repr_calls_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined
      (FlambdaReaperTypes.rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change (unit_code_deps flag u e)
         preserve_direct_calls e) ->
    program_refines_aged age_of
      (FlambdaReaperTypes.rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change (unit_code_deps flag u e)
         preserve_direct_calls e)
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

Definition aged_repr_calls_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).

Definition aged_repr_calls_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    FlambdaReaperLeastSpec.least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    any_kind_arguments e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
      cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (FlambdaReaperTypes.rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
         dominated to_unbox tcr preserve_direct_calls e).


(* ======================================================================== *)
(* The rebuilds of the reaper with types                                    *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.v: local unboxing, without and with the heap;
   unboxing across calls and changes of representation, for the code
   [nr]; with the code that never returns of the first rebuilt unit, and
   of the unit. *)

Definition aged_unbox_with_types_refines_rebuild_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (rebuild_unit_with_types_from_tables is_local (traverse_unit flag u e)
         T [] code_deps preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_with_types_from_tables is_local (traverse_unit flag u e)
         T [] code_deps preserve_direct_calls e)
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change code_deps dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_with_types_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.static_constants (unit_codes e) e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox preserve_direct_calls e).

Definition aged_unbox_with_types_refines_observably_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.static_constants (unit_codes e) e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox preserve_direct_calls e).

Definition aged_unbox_with_types_refines_rebuild_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (rebuild_unit_with_types_from_tables is_local (traverse_unit flag u e)
         T [] code_deps preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_with_types_from_tables is_local (traverse_unit flag u e)
         T [] code_deps preserve_direct_calls e)
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change code_deps dominated to_unbox preserve_direct_calls e).

Definition aged_unbox_with_types_refines_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.static_constants (unit_codes e) e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_up_to_heap_aged age_of e
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox preserve_direct_calls e).

Definition aged_unbox_with_types_refines_up_to_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.static_constants (unit_codes e) e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_up_to_heap_aged age_of e
      (rebuild_unit_unbox_with_types is_local (traverse_unit flag u e) T
         cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox preserve_direct_calls e).

Definition aged_unbox_fixed_with_types_refines_cc_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e)
         preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e)
         preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_fixed_with_types_observably_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_fixed_with_types_local_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_repr_fixed_with_types_refines_cc_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperUnboxingRefinesClosuresSpec.repr_unboxing_nr nr is_local
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox tcr e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e)
         preserve_direct_calls e) ->
    program_refines_aged age_of
      (rebuild_unit_cc_with_types_from_tables is_local
         (traverse_unit flag u e) T [] cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e)
         preserve_direct_calls e)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox tcr preserve_direct_calls e).

Definition aged_repr_fixed_with_types_observably_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    FlambdaReaperUnboxingRefinesClosuresSpec.repr_unboxing_nr nr is_local
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox tcr e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox tcr preserve_direct_calls e).

Definition aged_repr_fixed_with_types_local_nr_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    FlambdaReaperUnboxingRefinesClosuresSpec.repr_unboxing_nr nr is_local
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox tcr e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox tcr preserve_direct_calls e).

Definition aged_unbox_nr_ind_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls) ->
    program_refines_aged age_of
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_nr_ind_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_nr_ind_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_never_nr nr
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_unit_never_with_types_refines_cc_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls) ->
    program_refines_aged age_of
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_unit_never_with_types_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).

Definition aged_unbox_unit_never_with_types_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (age_of : static_env -> Code_age_relation.t),
    least_solution is_local (traverse_unit flag u e) T ->
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    distinct (FlambdaStaticsInScopeSpec.bound_continuations e) = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    FlambdaReaperCallingConventions.check_cc_closed (traverse_unit flag u e)
      T cannot_change
    = true ->
    FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit flag u e) T
    = true ->
    in_heap_fragment
      (first_rebuild_with_types flag is_local u e T cannot_change
         preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables T)
         (FlambdaReaperCallingConventions.unit_code_deps flag u e))
      (traverse_unit flag u e) T cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
      to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesUnitFlowSpec.used_reach_covered
      (traverse_unit flag u e) T
      (traverse_unit flag u
         (first_rebuild_with_types flag is_local u e T cannot_change
            preserve_direct_calls))
      T1 e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    program_defined e ->
    program_refines_observably_aged age_of e
      (rebuild_unit_unbox_repr_fixed_with_types is_local
         (traverse_unit flag u e) T cannot_change
         (FlambdaReaperCallingConventions.unit_code_deps flag u e) dominated
         to_unbox [] preserve_direct_calls e).
