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

(* The states of the simulations of the rebuilds that the statements of
   FlambdaReaperTypesCallsSpec.v apply the types to, for
   FlambdaReaperTypesCallsProofs.v. This file needs no review.

   - The simulation of the rebuild with changes of calling convention
     (FlambdaReaperRebuildProofs.related): each state that the run of the
     unit comes to, and from which it steps, is related to one that the
     run of the rebuilt unit comes to, with what the relation answers
     there ([cc_reach], for any table closed under the rules, the
     closures made where the graph has only the facts of their sets, by
     FlambdaReaperRebuildProofs.rebuild_cc_with_related_checked_local,
     which discharges the hypotheses of the simulation;
     [reach_of_located], for any relation of the shape of that
     simulation; [shrinking_reaches], for any shrinking simulation). The
     plain rebuild is the case where no code changes.
   - The simulation of the unboxing across calls
     (FlambdaReaperUnboxingRefinesCallsSimProofs.related2): the same of
     the runs of the two rebuilt units ([uc_reach]), and at an exit, the
     second exits through the same continuation, with the same values and
     heap ([uc_exit]). [uc_of_calls_gen], for the fragment of
     FlambdaReaperUnboxingRefinesCallsSpec.v, with the primitives of the
     heap if [hb]: the two rebuilt terms are related by [uc]
     (FlambdaReaperUnboxingRefinesCallsProofs.rebuild_unbox_modes_uc_gen),
     with the invariants of the first run that the simulation takes. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaPrimitiveProofs FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaUnusedStaticsSpec FlambdaBindersSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaNameAbstractionSpec FlambdaNameAbstractionProofs
  FlambdaFreshQuantifierSpec FlambdaRenameApartSpec
  FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperRebuildProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesOverSpec FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec.
From OxCamlFlambda2 Require FlambdaHeapFreeSpec FlambdaHeapFreeProofs
  FlambdaReaperLocalFieldsProofs FlambdaStaticsInScopeSpec
  FlambdaReaperCallingConventionsProofs FlambdaReaperClosuresExactProofs
  FlambdaReaperPartialGroupsProofs FlambdaReaperTypesTraversalHeapExitProofs.
Import ListNotations.

Import FlambdaReaperFlowProofs.WithLocal.

Module RP := FlambdaReaperRebuildProofs.
Module URC := FlambdaReaperUnboxingRefinesCallsProofs.
Module UCS := FlambdaReaperUnboxingRefinesCallsSimProofs.
Module PG := FlambdaReaperPartialGroupsProofs.

(* ======================================================================== *)
(* The states that a simulation reaches                                     *)
(* ======================================================================== *)

Section Reach.

Variable R : state -> state -> Prop.
Variable E : list value -> list value -> Prop.
Variable measure : state -> nat.
Hypothesis R_answered : ∀ s t : state, R s t -> answered R E measure s t.

(* A state that the first run comes to, and from which it steps, is
   related to one that the second run comes to. *)
Lemma shrinking_reaches (n : nat) :
  ∀ s t s1 : state,
    R s t -> run n s = Step s1 -> step s1 <> Undefined ->
    ∃ (m : nat) (t1 : state), run m t = Step t1 ∧ R s1 t1.
Proof.
  induction n as [ | n IH]; intros s t s1 Hrel Hrun Hs1.
  - injection Hrun as <-. exists 0, t. split; [ reflexivity | exact Hrel ].
  - rewrite run_succ in Hrun. pose proof (R_answered s t Hrel) as Ha.
    unfold answered in Ha.
    destruct (step s) as [s' | k' vs' hp' rd' | ] eqn:Hs;
      [ | discriminate Hrun | discriminate Hrun ].
    destruct Ha as [Hund | [(t' & Ht & Hrel') | [(_ & Hrel')
                                                | (t' & n0 & Ht & Hrel')]]].
    + exfalso. destruct n as [ | n].
      * injection Hrun as <-. exact (Hs1 Hund).
      * rewrite run_succ, Hund in Hrun. discriminate Hrun.
    + destruct (IH s' t' s1 Hrel' Hrun Hs1) as (m & t1 & Hm & Hr1).
      exists (S m), t1. rewrite run_succ, Ht. split; [ exact Hm | exact Hr1 ].
    + exact (IH s' t s1 Hrel' Hrun Hs1).
    + destruct (IH s' t' s1 Hrel' Hrun Hs1) as (m & t1 & Hm & Hr1).
      exists (S (S n0) + m), t1. rewrite run_add, Ht.
      split; [ exact Hm | exact Hr1 ].
Qed.

End Reach.

(* A relation of the shape of that of FlambdaReaperRebuildProofs, under
   the allocations [L] of the locations of the first run, whose steps are
   answered where the first run allocates as [L] says: each state that
   the first run comes to, and from which it steps, is related to one that
   the second comes to, under some allocations. *)
Lemma reach_of_located
    (E : list value -> heap -> list value -> heap -> Prop)
    (measure : state -> nat)
    (R : (location -> option node) -> state -> state -> Prop)
    (s0 t0 : state) :
  (∀ L : location -> option node, R L s0 t0) ->
  (∀ (L : location -> option node) (i : nat) (s t : state),
     run i s0 = Step s -> FlambdaReaperFlowProofs.WithLocal.alloc_ok L s ->
     R L s t -> answered_h E measure (R L) s t) ->
  ∀ (n : nat) (s : state),
    run n s0 = Step s -> step s <> Undefined ->
    ∃ (L : location -> option node) (m : nat) (t : state),
      run m t0 = Step t ∧ R L s t ∧ answered_h E measure (R L) s t.
Proof.
  intros H0 Hans n s Hrun Hs.
  destruct (FlambdaReaperFlowProofs.located_exists (S n) s0) as (L & HL).
  set (Rn := fun (i : nat) (s1 t1 : state) => run i s0 = Step s1 ∧ R L s1 t1).
  assert (Hans' : ∀ (i : nat) (s1 t1 : state),
            i < S n -> Rn i s1 t1 -> answered_h E measure (Rn (S i)) s1 t1).
  { intros i s1 t1 Hi [Hrun1 Hrel1].
    apply (answered_h_mono E measure (R L)).
    - intros s' t' Hstep' Hrel'. split; [ | exact Hrel' ].
      exact (run_then_step i s0 s1 s' Hrun1 Hstep').
    - exact (Hans L i s1 t1 Hrun1 (HL i s1 Hi Hrun1) Hrel1). }
  destruct (bshrinking_reach E measure (S n) Rn Hans' n 0 s0 t0 s
              ltac:(lia) (conj eq_refl (H0 L)) Hrun Hs)
    as (j & m & t & Hj & Hm & [Hrj Hr]).
  exists L, m, t. split; [ exact Hm | split; [ exact Hr | ] ].
  exact (Hans L j s t Hrj (HL j s Hj Hrj) Hr).
Qed.

(* ======================================================================== *)
(* The unit and its rebuild with changes of calling convention              *)
(* ======================================================================== *)

(* The hypotheses of
   FlambdaReaperRebuildProofs.rebuild_cc_with_refines_checked_local, and
   that the closures of the unit are made where the graph has only the
   facts of their sets: each state that the run of the unit comes to,
   and from which it steps, is related to one that the run of the rebuilt
   unit comes to, under some allocations of the locations, the closures
   being so made ([sets_exact]). *)
Lemma cc_reach (hb ex : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change cannot_change_witness : node -> bool)
    (code_deps : finite_map code_dep) (preserve_direct_calls : bool)
    (types : subkinds_rewriting) :
  is_local_slots_only is_local ->
  in_fragment hb e = true ->
  checked ex (unit_codes e) e = true ->
  calls_kinded [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  types_sound_all (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) (traverse_unit flag u e) types ->
  (∀ t : subkinds_types,
     types = Some t -> st_code_deps t = unit_code_deps flag u e) ->
  conts_table_ok types e ->
  results_kinded [] [] e = true ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    cannot_change_witness cannot_change ->
  (∀ (id : code_id) (a a' n y : node),
     cannot_change (Node_code_id id) = false ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source (solution_of_tables T) a' = true ->
      s_any_source (solution_of_tables T) a = true)
     ∧ (s_sources (solution_of_tables T) a' y = true ->
        s_sources (solution_of_tables T) a y = true)) ->
  (∀ (id id' : code_id) (w : node),
     cannot_change (Node_code_id id) = false ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  (∀ id : code_id,
     cannot_change (Node_code_id id) = false ->
     lookup id code_deps = lookup id (unit_code_deps flag u e)) ->
  closures_exact (graph_of_facts (traverse_unit flag u e)) (unit_kept e) e ->
  ∃ (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep),
    ∀ (n : nat) (s : state),
      run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
      step s <> Undefined ->
      ∃ (L : location -> option node) (m : nat) (t : state),
        run m
          (State
             (rebuild_unit_cc_with types (traverse_unit flag u e)
                (solution_of_tables T) cannot_change code_deps
                preserve_direct_calls e)
             [] [] empty_static_env empty_heap no_reads)
        = Step t
        ∧ RP.related (graph_of_facts (traverse_unit flag u e))
            (solution_of_tables T) noted deps cds (unit_kept e)
            (unit_codes e) (static_names symbols_of_binding e) u
            (if hb then True else False) True (traverse_unit flag u e)
            preserve_direct_calls (static_names code_ids_bound e)
            (fresh_name (names_of e)) ex True cannot_change code_deps
            (bound_code_metadata e) types L s t
        ∧ answered_h (exits_rel (if hb then True else False) ex) measure
            (RP.related (graph_of_facts (traverse_unit flag u e))
               (solution_of_tables T) noted deps cds (unit_kept e)
               (unit_codes e) (static_names symbols_of_binding e) u
               (if hb then True else False) True (traverse_unit flag u e)
               preserve_direct_calls (static_names code_ids_bound e)
               (fresh_name (names_of e)) ex True cannot_change code_deps
               (bound_code_metadata e) types L)
            s t.
Proof.
  intros Hlocal Hfrag Hcf Hkinded Hwf Hexits Hcheck Hlrs Hcheck_exact
    Htypes Htypes_deps0 Htypes_conts Hrk Hccc Hwid' Hcid' Hcode_deps0 Hxe.
  destruct (RP.rebuild_cc_with_related_checked_local hb ex flag is_local u e
              T cannot_change cannot_change_witness code_deps
              preserve_direct_calls types True Hlocal Hfrag Hcf Hkinded Hwf
              Hexits Hcheck Hlrs Hcheck_exact Htypes Htypes_deps0
              Htypes_conts Hrk Hccc Hwid' Hcid' Hcode_deps0 (fun _ => Hxe))
    as (noted & deps & cds & H0 & Hans).
  exists noted, deps, cds.
  intros n s Hrun Hs.
  exact (reach_of_located (exits_rel (if hb then True else False) ex) measure
           (RP.related (graph_of_facts (traverse_unit flag u e))
              (solution_of_tables T) noted deps cds (unit_kept e)
              (unit_codes e) (static_names symbols_of_binding e) u
              (if hb then True else False) True (traverse_unit flag u e)
              preserve_direct_calls (static_names code_ids_bound e)
              (fresh_name (names_of e)) ex True cannot_change code_deps
              (bound_code_metadata e) types)
           _ _ H0 Hans n s Hrun Hs).
Qed.

(* ======================================================================== *)
(* The two rebuilds, related across calls                                   *)
(* ======================================================================== *)

(* The two rebuilt units of the fragment of
   FlambdaReaperUnboxingRefinesCallsSpec.v, related by [uc], with the
   invariants of the first run that the simulation takes; with the
   primitives of the heap if [hb]. *)
Lemma uc_of_calls_gen (hb : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (pdc : bool) (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_fragment hb e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ (dead : list variable) (CS : code_id -> option code_shape)
    (NR : code_id -> bool) (RCH : bool -> variable -> code_id -> Prop)
    (OV REM : code_id -> Prop),
    (∀ z : variable,
       among z dead = true ->
       s_has_usage (solution_of_tables T) (Node_variable z) = false)
    ∧ distinct
        (field_variables
           (fst (unit_unboxed_fields (traverse_unit flag u e) T dominated
                   to_unbox e)))
      = true
    ∧ uc (fst (unit_unboxed_fields (traverse_unit flag u e) T dominated
                 to_unbox e))
        dead CS NR RCH OV REM []
        (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
           (unit_code_deps flag u e) pdc e)
        (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
           (unit_code_deps flag u e) dominated to_unbox pdc e)
    ∧ flow_run RCH
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads)
    ∧ over_run OV
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads)
    ∧ rem_run REM
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Hslots Hfrag Hclosed Hcalls Hcheck.
  destruct (URC.rebuild_unbox_modes_uc_gen hb None flag is_local u e T
              cannot_change dominated to_unbox pdc nr
              (reach2 (fun _ _ => true)) (fun _ => true) (fun _ => true)
              ltac:(intros t Ht; discriminate Ht) (no_types_conts e) Hslots
              (proj2 (andb_true_iff _ _) (in_fragment_parts hb e Hfrag))
              Hclosed)
    as (dead & CS & NR & Hdead & Hdistinct & Huc).
  - exact (calls_unboxing_in_never_reach nr _ _ _ _ _ _ _ _ _ e
             (calls_unboxing_in_over3_reach nr _ _ _ _ _ _ _ _ _
                (fun _ => true) e (fun _ _ _ H => H)
                (calls_unboxing_in_over2_reach nr _ _ _ _ _ _ _
                   (fun _ => true) e
                   (calls_unboxing_in_flow2_reach nr _ _ _ _ _ _ _ _ e
                      (fun _ _ => eq_refl)
                      (calls_unboxing_in_flow_reach nr _ _ _ _ _ _
                         (fun _ _ => true) e
                         (calls_unboxing_in_over1_nr nr _ _ _ _ _ _ e
                            (calls_unboxing_in_over_nr nr _ _ _ _ _ _ e
                               Hcalls))))))).
  - exact Hcheck.
  - exists dead, CS, NR, (fun b f id => reach2 (fun _ _ => true) b f id = true),
      (fun id => true = true), (fun id => true = true).
    split; [ exact Hdead | split; [ exact Hdistinct | split; [ exact Huc | ]]].
    split; [ | split ].
    + intros n s _ ap f v id slots _ _ _. split; intros _; reflexivity.
    + intros n s _.
      exact (PG.step_over_ok_total _ s (fun _ => eq_refl)).
    + intros n s _.
      exact (PG.step_rem_ok_total _ s (fun _ => eq_refl)).
Qed.

(* The same, of the units of [in_flow_fragment]. *)
Lemma uc_of_calls (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (pdc : bool) (nr : code_id -> bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  calls_unboxing_nr nr (traverse_unit flag u e) T cannot_change
    (unit_code_deps flag u e) dominated to_unbox e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ (dead : list variable) (CS : code_id -> option code_shape)
    (NR : code_id -> bool) (RCH : bool -> variable -> code_id -> Prop)
    (OV REM : code_id -> Prop),
    (∀ z : variable,
       among z dead = true ->
       s_has_usage (solution_of_tables T) (Node_variable z) = false)
    ∧ distinct
        (field_variables
           (fst (unit_unboxed_fields (traverse_unit flag u e) T dominated
                   to_unbox e)))
      = true
    ∧ uc (fst (unit_unboxed_fields (traverse_unit flag u e) T dominated
                 to_unbox e))
        dead CS NR RCH OV REM []
        (rebuild_unit_cc_from_tables (traverse_unit flag u e) T cannot_change
           (unit_code_deps flag u e) pdc e)
        (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
           (unit_code_deps flag u e) dominated to_unbox pdc e)
    ∧ flow_run RCH
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads)
    ∧ over_run OV
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads)
    ∧ rem_run REM
        (State
           (rebuild_unit_cc_from_tables (traverse_unit flag u e) T
              cannot_change (unit_code_deps flag u e) pdc e)
           [] [] empty_static_env empty_heap no_reads).
Proof. exact (uc_of_calls_gen false flag is_local u e T cannot_change
                dominated to_unbox pdc nr). Qed.

Section Uc_runs.

Variable M : list (node * unboxed_fields).
Hypothesis M_distinct : distinct (field_variables M) = true.
Variable dead : list variable.
Variable CS : code_id -> option code_shape.
Variable NR : code_id -> bool.
Variable RCH : bool -> variable -> code_id -> Prop.
Variable OV REM : code_id -> Prop.

Abbreviation R2 := (related2 M dead CS NR RCH OV REM).

(* The relation with the invariants of the first run. *)
Definition uc_rel (s t : state) : Prop :=
  R2 s t ∧ PG.pa_inv s ∧ flow_run RCH s ∧ over_run OV s ∧ rem_run REM s.

Lemma uc_rel_answered (s t : state) :
  uc_rel s t -> answered uc_rel eq measure s t.
Proof.
  intros (Hr & Hinv & Hfl0 & Hov0 & Hrm0).
  exact (answered_pa_inv M dead CS NR RCH OV REM s t Hinv Hfl0 Hov0 Hrm0
           (related2_answered M M_distinct dead CS NR RCH OV REM s t Hr Hinv
              (Hfl0 0 s eq_refl) (Hov0 0 s eq_refl) (Hrm0 0 s eq_refl))).
Qed.

Lemma uc_rel_initial (p q : expr) :
  uc M dead CS NR RCH OV REM [] p q ->
  flow_run RCH (State p [] [] empty_static_env empty_heap no_reads) ->
  over_run OV (State p [] [] empty_static_env empty_heap no_reads) ->
  rem_run REM (State p [] [] empty_static_env empty_heap no_reads) ->
  uc_rel (State p [] [] empty_static_env empty_heap no_reads)
    (State q [] [] empty_static_env empty_heap no_reads).
Proof.
  intros Huc Hfl Hov Hrm.
  split; [ | split; [ apply PG.pa_inv_initial
                    | split; [ exact Hfl | split; [ exact Hov
                                                  | exact Hrm ] ] ] ].
  left. exists []. split; [ exact Huc | ].
  split; [ apply env_rel2_empty | split; [ constructor | ] ].
  split; [ apply statics_rel_empty | repeat split ].
Qed.

(* Each state that the first run comes to, and from which it steps, is
   related to one that the second comes to. *)
Lemma uc_reach (p q : expr) :
  uc M dead CS NR RCH OV REM [] p q ->
  flow_run RCH (State p [] [] empty_static_env empty_heap no_reads) ->
  over_run OV (State p [] [] empty_static_env empty_heap no_reads) ->
  rem_run REM (State p [] [] empty_static_env empty_heap no_reads) ->
  ∀ (n : nat) (t : state),
    run n (State p [] [] empty_static_env empty_heap no_reads) = Step t ->
    step t <> Undefined ->
    ∃ (m : nat) (t' : state),
      run m (State q [] [] empty_static_env empty_heap no_reads) = Step t'
      ∧ R2 t t'.
Proof.
  intros Huc Hfl Hov Hrm n t Hrun Ht.
  destruct (shrinking_reaches uc_rel eq measure uc_rel_answered n _ _ t
              (uc_rel_initial p q Huc Hfl Hov Hrm) Hrun Ht)
    as (m & t' & Hm & Hr).
  exists m, t'. split; [ exact Hm | exact (proj1 Hr) ].
Qed.

(* Where the first run exits, the second exits through the same
   continuation, with the same values and heap, from a related state. *)
Lemma uc_exit (p q : expr) :
  uc M dead CS NR RCH OV REM [] p q ->
  flow_run RCH (State p [] [] empty_static_env empty_heap no_reads) ->
  over_run OV (State p [] [] empty_static_env empty_heap no_reads) ->
  rem_run REM (State p [] [] empty_static_env empty_heap no_reads) ->
  ∀ (n : nat) (t : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State p [] [] empty_static_env empty_heap no_reads) = Step t ->
    step t = Exit k vs hp rd ->
    ∃ (m : nat) (t' : state) (rd' : read_record),
      run m (State q [] [] empty_static_env empty_heap no_reads) = Step t'
      ∧ step t' = Exit k vs hp rd' ∧ R2 t t'.
Proof.
  intros Huc Hfl Hov Hrm n t k vs hp rd Hrun Hexit.
  destruct (FlambdaReaperTypesTraversalHeapExitProofs.shrinking_exit_state
              uc_rel eq measure uc_rel_answered n _ _ t k vs hp rd
              (uc_rel_initial p q Huc Hfl Hov Hrm) Hrun Hexit)
    as (m & t' & vs' & rd' & Hm & Hr & Ht' & <-).
  exists m, t', rd'.
  split; [ exact Hm | split; [ exact Ht' | exact (proj1 Hr) ] ].
Qed.

End Uc_runs.
