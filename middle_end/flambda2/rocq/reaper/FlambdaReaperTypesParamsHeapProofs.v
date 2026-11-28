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

(* Proofs of FlambdaReaperTypesParamsHeapSpec.v. This file needs no
   review.

   As FlambdaReaperTypesParamsProofs.v, with the primitives of the heap:
   the check [params_aliased] of the traversal
   (FlambdaReaperTypesParamsTraversalSpec.params_aliased_traversal) gives
   that what the value of a parameter may come from may come to the node
   of the code dep (FlambdaReaperTypesParamsProofs.may_come_from_param),
   so that a value rebuilt with the heap at the node of the parameter is
   rebuilt under the metadata of the node ([param_node_facts_heap]). The
   states are those of FlambdaReaperTypesCallsHeapProofs.cc_states_heap,
   and the type is rewritten at the node by
   FlambdaReaperTypesReturnsHeapProofs.rewrite_result_type_at_sound_heap.
   The example is computed, its unit and run of the unit those of
   FlambdaReaperTypesCallsHeapSpec.calls_heap_types_example_statement. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperLocalFields FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesHeapSpec FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesReturnsSpec FlambdaReaperTypesUsedValuesSpec
  FlambdaReaperTypesCallsUsedSpec FlambdaReaperTypesParamsSpec
  FlambdaReaperTypesCallsHeapSpec FlambdaReaperTypesParamsHeapSpec
  FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperTypesProofs
  FlambdaReaperTypesCallsProofs FlambdaReaperTypesLooseProofs
  FlambdaReaperTypesParamsProofs FlambdaReaperTypesParamsTraversalProofs
  FlambdaReaperTypesCallsHeapProofs FlambdaReaperTypesReturnsHeapProofs
  FlambdaReaperUnboxing.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
From OxCamlFlambda2Types Require FlambdaTypeMeaningBoolProofs.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module LP := FlambdaReaperTypesLooseProofs.
Module PP := FlambdaReaperTypesParamsProofs.
Module CHP := FlambdaReaperTypesCallsHeapProofs.
Module RH := FlambdaReaperTypesReturnsHeapProofs.

(* ======================================================================== *)
(* The parameter and the node of the code dep, with the heap                *)
(* ======================================================================== *)

(* FlambdaReaperTypesParamsProofs.param_node_facts, of the values rebuilt
   with the heap. *)
Lemma param_node_facts_heap (local : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (β : location -> location -> Prop)
    (id : code_id) (i : nat) (p : variable) (r : node) :
  is_local_slots_only is_local ->
  least_with local is_local (traverse_unit flag u e) T e ->
  params_aliased (traverse_unit flag u e) (unit_code_deps flag u e) e
  = true ->
  at_param e id i p ->
  param_node flag u e id i = Some r ->
  s_has_usage (solution_of_tables T) r = true ->
  s_has_usage (solution_of_tables T) (Node_variable p) = true
  ∧ ∀ v v' : value,
      rebuilt_value_heap (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) β v (Node_variable p) v' ->
      rebuilt_under_heap (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) β
        (metadata_for_name
           (plain_context is_local (traverse_unit flag u e) T sets) r)
        v v'.
Proof.
  intros Hlocal Hleast Hcheck (c & Hin & Hp) Hrn Hhu.
  pose proof (PP.least_local_with local flag is_local u e T Hlocal Hleast)
    as Hll.
  unfold param_node in Hrn.
  destruct (lookup id (unit_code_deps flag u e)) as [cd | ] eqn:Hcd;
    [ | discriminate Hrn ].
  destruct (PP.params_aliased_at _ _ _ id c cd i p r Hcheck Hin Hcd Hp Hrn)
    as [HA HB].
  assert (Hhp : s_has_usage (solution_of_tables T) (Node_variable p) = true)
    by (eapply PP.param_has_usage; eassumption).
  split; [ exact Hhp | ].
  intros v v' Hrv.
  pose proof (TP.metadata_for_name_at
                (plain_context is_local (traverse_unit flag u e) T sets) r)
    as [Hat _].
  right. exists (Node_variable p). split; [ exact Hrv | split ].
  - intros a Ha Hhsa. refine (Hat a _ Hhsa).
    eapply PP.may_come_from_param; eassumption.
  - intros _. exact Hhp.
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma cc_param_heap_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (id : code_id)
    (i : nat) (p : variable) (r : node) (type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  heap_premises local flag is_local u e T sets ->
  LP.cc_hyps flag u e T cc ccw code_deps ->
  param_premises flag u e T n s id i p r tenv canon valuation valuation'
    type_var v ->
  ∃ (m : nat) (t : state) (v' : value),
    run m
      (initial_state
         (rebuild_unit_cc_with None (traverse_unit flag u e)
            (solution_of_tables T) cc code_deps b e))
    = Step t
    ∧ lookup p (state_vars t) = Some v'
    ∧ type_satisfied (state_statics t) valuation' v'
        (rewrite_result_type_at fx
           (plain_context is_local (traverse_unit flag u e) T sets) tenv
           canon r type_var).
Proof.
  intros Hfx (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits & Honce
              & Hvals & Hsets)
    (Hccc & Hwid & Hcid & Hcd)
    (Hrun & Hs & Hat & Hx & Hrn & Hhu & Henv & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck0 & Hlrs & Hcheck_exact).
  pose proof (FlambdaReaperTypesParamsTraversalProofs.params_aliased_traversal
                flag u e Hfrag Honce)
    as Hcheck.
  destruct (CHP.cc_states_heap flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck0 Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (β & m & t & Hm & Hst & Halloc & Hvars & _ & _ & _).
  destruct (param_node_facts_heap local flag is_local u e T sets β id i p r
              Hlocal Hleast Hcheck Hat Hrn Hhu)
    as [Hhp Hru].
  destruct (Hvars p v Hx Hhp) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  exact (RH.rewrite_result_type_at_sound_heap fx false Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation valuation'
           β r type_var v v' Hle Hlocal Hsets Henv Hfr Halloc Hst
           (CHP.no_static_sets _ _ _) Hcopies Hfind Hl (Hru v v' Hrv)).
Qed.

Lemma plain_param_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> param_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i p r type_var v Hprem Hres.
  destruct (cc_param_heap_gen local fx flag is_local u e T _ _ _ b sets tenv
              canon valuation valuation' n s id i p r type_var v Hfx Hprem
              (LP.cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- LP.plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem param_type_end_to_end_heap : param_type_end_to_end_heap_statement.
Proof. exact (plain_param_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem param_type_end_to_end_heap_landed :
  param_type_end_to_end_heap_landed_statement.
Proof.
  exact (plain_param_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem param_type_end_to_end_heap_local :
  param_type_end_to_end_heap_local_statement.
Proof. exact (plain_param_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem param_type_end_to_end_heap_local_landed :
  param_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (plain_param_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma cc_param_type_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_param_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i p r type_var v Hprem Hccc Hwid Hres.
  destruct (LP.cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (cc_param_heap_gen local fx flag is_local u e T _ ccw _ b sets tenv
           canon valuation valuation' n s id i p r type_var v Hfx Hprem Hh
           Hres).
Qed.

Theorem cc_param_type_end_to_end_heap :
  cc_param_type_end_to_end_heap_statement.
Proof. exact (cc_param_type_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_param_type_end_to_end_heap_landed :
  cc_param_type_end_to_end_heap_landed_statement.
Proof.
  exact (cc_param_type_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_param_type_end_to_end_heap_local :
  cc_param_type_end_to_end_heap_local_statement.
Proof. exact (cc_param_type_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_param_type_end_to_end_heap_local_landed :
  cc_param_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (cc_param_type_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Lemma param_heap_example_gen (fx : fixes) :
  rewrite_result_type_at fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       CallsHeapTypesExample.CH.facts CallsHeapTypesExample.CH.tables
       CallsHeapTypesExample.sets)
    CallsHeapTypesExample.tenv CallsHeapTypesExample.canon (Node_made 6)
    90%positive
  = CallsHeapTypesExample.param_type ->
  param_heap_example_with fx.
Proof.
  intros Hrw.
  destruct CHP.calls_heap_types_example
    as (Hprem & Hnf & _ & _ & _ & (s & Hres & _) & _).
  destruct Hres
    as (Hrun & Hs & Hx & _ & Henv & Hfr & Hcopies & Hfind & Hl).
  split; [ exact Hprem | split; [ exact Hnf | ] ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exists s.
    split; [ exact Hrun | split; [ exact Hs | ] ].
    split.
    { exists CallsHeapTypesExample.CH.callee.
      split; [ vm_compute; left; reflexivity | reflexivity ]. }
    split; [ exact Hx | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ exact Henv | split; [ exact Hfr | ] ].
    split; [ exact Hcopies | split; [ exact Hfind | exact Hl ] ]. }
  split; [ exact Hrw | ].
  exists
    (match run 7
             (initial_state
                (rebuild_unit_from_tables CallsHeapTypesExample.CH.facts
                   CallsHeapTypesExample.CH.tables true
                   CallsHeapTypesExample.CH.unit_expr))
     with
     | Step t => t
     | _ => initial_state CallsHeapTypesExample.CH.unit_expr
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
  vm_compute. reflexivity.
Qed.

Theorem param_heap_example : param_heap_example_statement.
Proof. apply param_heap_example_gen. vm_compute. reflexivity. Qed.

Theorem param_heap_example_landed : param_heap_example_landed_statement.
Proof. apply param_heap_example_gen. vm_compute. reflexivity. Qed.
