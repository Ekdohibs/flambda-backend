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

(* The states of the simulation of the changes of representation
   (FlambdaReaperUnboxingRefinesClosuresSimProofs.related, [rr]) that the
   statements of FlambdaReaperTypesCallsSpec.v apply the types to, for
   FlambdaReaperTypesCallsProofs.v. This file needs no review.

   [repr_calls_rr]: the unit rebuilt with changes of calling convention
   and the unit rebuilt with unboxing and changes of representation, of
   the fragment of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v, are
   related by [rr], where the flow may bring any code anywhere
   (FlambdaReaperUnboxingRefinesClosuresProofs
   .rebuild_repr_reach_modes_with_rr). [rr_reach]: each state that the
   run of the first comes to, and from which it steps, is related to one
   that the run of the second comes to; [rr_exit]: where the first exits,
   the second exits through the same continuation, with the same values
   and heap, from a related state. Related states bind the same static
   values ([rr_values]). The file reads the relation only through the
   lemmas of FlambdaReaperUnboxingRefinesClosuresSimProofs.v
   ([related_initial], [related_static_values], [related_answered]), not
   through its shape. The names of the parameters of the wrappers of the
   relation are any ([rr_any]); what is left of an over-application may
   call any code, as the flow may bring any code anywhere. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaRenameApartSpec FlambdaKindModel
  FlambdaModel FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaPrimitiveProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresCallsSpec
  FlambdaReaperUnboxingRefinesClosuresSimProofs
  FlambdaReaperUnboxingRefinesClosuresProofs.
From OxCamlFlambda2 Require FlambdaReaperTypesTraversalHeapExitProofs
  FlambdaReaperTypesCallsExitProofs.
Import ListNotations.

(* ======================================================================== *)
(* The two rebuilds, related by [rr]                                        *)
(* ======================================================================== *)

Lemma repr_calls_rr (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) (pdc : bool)
    (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  repr_calls_unboxing_nr nr is_local (traverse_unit flag u e) T
    cannot_change (unit_code_deps flag u e) dominated to_unbox tcr e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ (M : list (node * unboxed_fields))
    (CH : variable -> option closure_repr)
    (CS : code_id -> option code_info)
    (RCH : bool -> variable -> code_id -> Prop),
    distinct (field_variables M) = true
    ∧ (∀ (x : variable) (r : closure_repr),
         CH x = Some r -> fields_of M x = Some [])
    ∧ (∀ (x : variable) (r : closure_repr),
         CH x = Some r -> among x (field_variables M) = false)
    ∧ (∃ W : variable -> Prop,
         rr M CH CS RCH W (fun _ : code_id => true = true) []
           (rebuild_unit_cc_with types (traverse_unit flag u e)
              (solution_of_tables T) (fun n => mem_node n cannot_change)
              (unit_code_deps flag u e) pdc e)
           (rebuild_unit_unbox_repr_mode_with types true is_local
              (traverse_unit flag u e) T cannot_change
              (unit_code_deps flag u e) dominated to_unbox tcr pdc e))
    ∧ flow_run RCH (fun _ : code_id => true = true)
        (State
           (rebuild_unit_cc_with types (traverse_unit flag u e)
              (solution_of_tables T) (fun n => mem_node n cannot_change)
              (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hfragment Hclosed Hrepr Hcheck.
  destruct (rebuild_repr_reach_modes_with_rr types flag is_local u e T
              cannot_change dominated to_unbox tcr pdc nr (fun _ _ _ => true)
              Hslots Hfragment Hclosed
              (repr_calls_within_reach_unit nr is_local _ T cannot_change _
                 dominated to_unbox tcr (fun _ _ _ => true) e Hrepr)
              Hcheck)
    as (M & CH & CS & W & Hdistinct & HCM & HCf & Hrr).
  exists M, CH, CS, (fun b f id => true = true).
  split; [ exact Hdistinct | split; [ exact HCM | split; [ exact HCf | ] ] ].
  split; [ exists W; exact Hrr | ].
  apply flow_run_total; [ intros b f id | intros id ]; reflexivity.
Qed.

(* ======================================================================== *)
(* The runs                                                                 *)
(* ======================================================================== *)

Section Rr_runs.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable CH : variable -> option closure_repr.
Hypothesis CH_M : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> fields_of M x = Some [].
Hypothesis CH_fresh : ∀ (x : variable) (r : closure_repr),
  CH x = Some r -> among x (field_variables M) = false.
Variable CS : code_id -> option code_info.
Variable RCH : bool -> variable -> code_id -> Prop.

(* The relation, for some names of the parameters of the wrappers. *)
Definition rr_any (s t : state) : Prop :=
  ∃ W : variable -> Prop,
    related M CH CS RCH W (fun _ : code_id => true = true) s t.

Abbreviation R := rr_any.

(* The relation with the invariant of the flow of the first run. *)
Definition rr_rel (s t : state) : Prop :=
  R s t ∧ flow_run RCH (fun _ : code_id => true = true) s.

Lemma rr_rel_answered (s t : state) :
  rr_rel s t -> answered rr_rel eq measure s t.
Proof.
  intros [(W & Hr) Hfl0].
  pose proof (related_answered M M_distinct CH CH_M CH_fresh CS RCH W _ s t
                Hr (proj1 (Hfl0 0 s eq_refl)) (proj2 (Hfl0 0 s eq_refl)))
    as Ha.
  unfold answered in Ha |- *.
  destruct (step s) as [s1 | k vs hp rd1 | ] eqn:Hs; [ | exact Ha | exact I ].
  assert (Hfl1 : flow_run RCH (fun _ : code_id => true = true) s1).
  { intros n s2 Hn. apply (Hfl0 (S n)). rewrite run_succ, Hs. exact Hn. }
  destruct Ha
    as [H | [(t1 & Ht & Hr1) | [(Hlt & Hr1) | (t1 & n & Ht & Hr1)]]].
  - left. exact H.
  - right. left. exists t1. split; [ exact Ht | ].
    split; [ exists W; exact Hr1 | exact Hfl1 ].
  - right. right. left. split; [ exact Hlt | ].
    split; [ exists W; exact Hr1 | exact Hfl1 ].
  - right. right. right. exists t1, n. split; [ exact Ht | ].
    split; [ exists W; exact Hr1 | exact Hfl1 ].
Qed.

Lemma rr_initial (p q : expr) :
  (∃ W : variable -> Prop,
     rr M CH CS RCH W (fun _ : code_id => true = true) [] p q) ->
  flow_run RCH (fun _ : code_id => true = true)
    (State p [] [] empty_static_env empty_heap no_reads) ->
  rr_rel (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros (W & Hrr) Hfl.
  split; [ exists W; exact (related_initial M CH CS RCH W _ p q Hrr) | ].
  exact Hfl.
Qed.

(* Each state that the first run comes to, and from which it steps, is
   related to one that the second comes to. *)
Lemma rr_reach (p q : expr) :
  (∃ W : variable -> Prop,
     rr M CH CS RCH W (fun _ : code_id => true = true) [] p q) ->
  flow_run RCH (fun _ : code_id => true = true)
    (State p [] [] empty_static_env empty_heap no_reads) ->
  ∀ (n : nat) (t : state),
    run n (State p [] [] empty_static_env empty_heap no_reads) = Step t ->
    step t <> Undefined ->
    ∃ (m : nat) (t' : state),
      run m (State q [] [] empty_static_env empty_heap no_reads) = Step t'
      ∧ R t t'.
Proof.
  intros Hrr Hfl n t Hrun Ht.
  destruct (FlambdaReaperTypesCallsExitProofs.shrinking_reaches rr_rel eq
              measure rr_rel_answered n _ _ t (rr_initial p q Hrr Hfl) Hrun
              Ht)
    as (m & t' & Hm & Hr).
  exists m, t'. split; [ exact Hm | exact (proj1 Hr) ].
Qed.

(* Where the first run exits, the second exits through the same
   continuation, with the same values and heap, from a related state. *)
Lemma rr_exit (p q : expr) :
  (∃ W : variable -> Prop,
     rr M CH CS RCH W (fun _ : code_id => true = true) [] p q) ->
  flow_run RCH (fun _ : code_id => true = true)
    (State p [] [] empty_static_env empty_heap no_reads) ->
  ∀ (n : nat) (t : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State p [] [] empty_static_env empty_heap no_reads) = Step t ->
    step t = Exit k vs hp rd ->
    ∃ (m : nat) (t' : state) (rd' : read_record),
      run m (State q [] [] empty_static_env empty_heap no_reads) = Step t'
      ∧ step t' = Exit k vs hp rd' ∧ R t t'.
Proof.
  intros Hrr Hfl n t k vs hp rd Hrun Hexit.
  destruct (FlambdaReaperTypesTraversalHeapExitProofs.shrinking_exit_state
              rr_rel eq measure rr_rel_answered n _ _ t k vs hp rd
              (rr_initial p q Hrr Hfl) Hrun Hexit)
    as (m & t' & vs' & rd' & Hm & Hr & Ht' & <-).
  exists m, t', rd'.
  split; [ exact Hm | split; [ exact Ht' | exact (proj1 Hr) ] ].
Qed.

(* Related states bind the same static values. *)
Lemma rr_values (t t' : state) :
  R t t' -> static_values (state_statics t') = static_values (state_statics t).
Proof.
  intros (W & Hr). exact (related_static_values M CH CS RCH W _ t t' Hr).
Qed.

End Rr_runs.
