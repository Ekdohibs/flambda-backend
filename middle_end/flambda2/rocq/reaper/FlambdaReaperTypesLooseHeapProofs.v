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

(* Proofs of FlambdaReaperTypesLooseHeapSpec.v. This file needs no review.

   The states that the simulations with the heap relate keep the static
   sets whole (FlambdaReaperTypesCallsHeapProofs.cc_states_heap,
   [calls_states_heap], [calls_exit_heap], by
   FlambdaReaperTypesCcSetsProofs.v and
   FlambdaReaperTypesCallsHeapProofs.statics_rel_whole_heap); the proofs
   are those of FlambdaReaperTypesCallsHeapProofs.v and
   FlambdaReaperTypesReturnsHeapProofs.v in the fragment [loose]. *)

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
  FlambdaReaperTypesReturnsSpec FlambdaReaperTypesLooseSpec
  FlambdaReaperTypesUsedValuesSpec FlambdaReaperTypesCallsUsedSpec
  FlambdaReaperTypesCallsHeapSpec FlambdaReaperTypesLooseHeapSpec
  FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperTypesProofs
  FlambdaReaperTypesHeapProofs FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesTraversalHeapProofs FlambdaReaperTypesCallsProofs
  FlambdaReaperTypesLooseProofs FlambdaReaperTypesCallsHeapProofs
  FlambdaReaperTypesReturnsHeapProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

Module TP := FlambdaReaperTypesProofs.
Module THP := FlambdaReaperTypesHeapProofs.
Module TXP := FlambdaReaperTypesTraversalHeapProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module LP := FlambdaReaperTypesLooseProofs.
Module CHP := FlambdaReaperTypesCallsHeapProofs.
Module RH := FlambdaReaperTypesReturnsHeapProofs.

(* ======================================================================== *)
(* The plain rebuild and the rebuild with changes of calling convention     *)
(* ======================================================================== *)

Lemma cc_result_loose_heap_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  heap_premises local flag is_local u e T sets ->
  LP.cc_hyps flag u e T cc ccw code_deps ->
  unaliased_result_premises (traverse_unit flag u e) T e n s tenv canon
    valuation valuation' code_var type_var v ->
  ∃ (m : nat) (t : state) (v' : value),
    run m
      (initial_state
         (rebuild_unit_cc_with None (traverse_unit flag u e)
            (solution_of_tables T) cc code_deps b e))
    = Step t
    ∧ lookup code_var (state_vars t) = Some v'
    ∧ type_satisfied (state_statics t) valuation' v'
        (rewrite_result_type fx
           (plain_context is_local (traverse_unit flag u e) T sets) tenv
           canon code_var type_var).
Proof.
  intros Hfx (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits & Honce
              & Hvals & Hsets)
    (Hccc & Hwid & Hcid & Hcd)
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (CHP.cc_states_heap flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (β & m & t & Hm & Hst & Halloc & Hvars & _ & _ & Hwhole).
  destruct (Hvars code_var v Hx Hhu) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  exact (THP.rewrite_result_type_sound_heap_gen fx true Hfx is_local _ T
           sets tenv canon (state_statics s) (state_statics t) valuation
           valuation' β code_var type_var v v' Hle Hlocal Hsets Henv Hfr
           Halloc Hst (fun _ => Hwhole) Hcopies Hfind Hl Hrv).
Qed.

Lemma plain_result_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unaliased_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    code_var type_var v Hprem Hres.
  destruct (cc_result_loose_heap_gen local fx flag is_local u e T _ _ _ b
              sets tenv canon valuation valuation' n s code_var type_var v Hfx
              Hprem (LP.cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- LP.plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem unaliased_result_type_end_to_end_heap :
  unaliased_result_type_end_to_end_heap_statement.
Proof. exact (plain_result_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_result_type_end_to_end_heap_landed :
  unaliased_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (plain_result_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unaliased_result_type_end_to_end_heap_local :
  unaliased_result_type_end_to_end_heap_local_statement.
Proof. exact (plain_result_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_result_end_to_end_heap_local_landed :
  unaliased_result_end_to_end_heap_local_landed_statement.
Proof.
  exact (plain_result_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma cc_result_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx ->
  cc_unaliased_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s code_var type_var v Hprem Hccc Hwid Hres.
  destruct (LP.cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (cc_result_loose_heap_gen local fx flag is_local u e T _ ccw _ b sets
           tenv canon valuation valuation' n s code_var type_var v Hfx Hprem
           Hh Hres).
Qed.

Theorem cc_unaliased_result_type_end_to_end_heap :
  cc_unaliased_result_type_end_to_end_heap_statement.
Proof. exact (cc_result_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_type_end_to_end_heap_landed :
  cc_unaliased_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (cc_result_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_result_type_end_to_end_heap_local :
  cc_unaliased_result_type_end_to_end_heap_local_statement.
Proof. exact (cc_result_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_end_to_end_heap_local_landed :
  cc_unaliased_result_end_to_end_heap_local_landed_statement.
Proof.
  exact (cc_result_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The results of code                                                      *)
(* ======================================================================== *)

Lemma plain_return_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unaliased_code_result_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i arg r type_var v Hprem Hres.
  destruct (RH.cc_return_heap_gen true local fx flag is_local u e T _ _ _ b
              sets tenv canon valuation valuation' n s id i arg r type_var v
              Hfx Hprem (LP.cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- LP.plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem unaliased_code_result_end_to_end_heap :
  unaliased_code_result_end_to_end_heap_statement.
Proof. exact (plain_return_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_code_result_end_to_end_heap_landed :
  unaliased_code_result_end_to_end_heap_landed_statement.
Proof.
  exact (plain_return_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unaliased_code_result_end_to_end_heap_local :
  unaliased_code_result_end_to_end_heap_local_statement.
Proof. exact (plain_return_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_code_result_heap_local_landed :
  unaliased_code_result_heap_local_landed_statement.
Proof.
  exact (plain_return_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma cc_return_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx ->
  cc_unaliased_code_result_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i arg r type_var v Hprem Hccc Hwid Hres.
  destruct (LP.cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (RH.cc_return_heap_gen true local fx flag is_local u e T _ ccw _ b
           sets tenv canon valuation valuation' n s id i arg r type_var v Hfx
           Hprem Hh Hres).
Qed.

Theorem cc_unaliased_code_result_end_to_end_heap :
  cc_unaliased_code_result_end_to_end_heap_statement.
Proof. exact (cc_return_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_end_to_end_heap_landed :
  cc_unaliased_code_result_end_to_end_heap_landed_statement.
Proof.
  exact (cc_return_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_unaliased_code_result_end_to_end_heap_local :
  cc_unaliased_code_result_end_to_end_heap_local_statement.
Proof. exact (cc_return_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_heap_local_landed :
  cc_unaliased_code_result_heap_local_landed_statement.
Proof.
  exact (cc_return_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The unit rebuilt with unboxing across calls                              *)
(* ======================================================================== *)

Lemma calls_types_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_types_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hcalls Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (CHP.calls_exit_heap local flag is_local u e T cannot_change
              dominated to_unbox b nr sets n s k vs hp rd Hprem Hcalls Hrun
              Hexit)
    as (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & Hst & Halloc
        & Hwhole).
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  exact (THP.rewrite_symbol_type_sound_heap_gen fx true Hfx is_local _ T
           sets tenv canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' β sym allocated Hle Hlocal Hsets Henv Hfr
           Halloc Hst (fun _ => Hwhole) Hcopies Hl).
Qed.

Theorem calls_unaliased_types_end_to_end_heap :
  calls_unaliased_types_end_to_end_heap_statement.
Proof. exact (calls_types_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_types_end_to_end_heap_landed :
  calls_unaliased_types_end_to_end_heap_landed_statement.
Proof.
  exact (calls_types_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_types_end_to_end_heap_local :
  calls_unaliased_types_end_to_end_heap_local_statement.
Proof. exact (calls_types_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_types_heap_local_landed :
  calls_unaliased_types_heap_local_landed_statement.
Proof.
  exact (calls_types_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma calls_result_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_result_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s code_var type_var v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hnone.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (CHP.calls_states_heap local flag is_local u e T cannot_change
              dominated to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (β & m & t & Hm & Hst & Halloc & Hwhole & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & Hplain & _).
  exists m, t, v_mid.
  split; [ exact Hm | split; [ exact (Hplain Hin Hnone) | ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  exact (THP.rewrite_result_type_sound_heap_gen fx true Hfx is_local _ T
           sets tenv canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' β code_var type_var v v_mid Hle Hlocal Hsets
           Henv Hfr Halloc Hst (fun _ => Hwhole) Hcopies Hfind Hl Hrv).
Qed.

Theorem calls_unaliased_result_end_to_end_heap :
  calls_unaliased_result_end_to_end_heap_statement.
Proof. exact (calls_result_loose_heap false all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_result_end_to_end_heap_landed :
  calls_unaliased_result_end_to_end_heap_landed_statement.
Proof.
  exact (calls_result_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_result_end_to_end_heap_local :
  calls_unaliased_result_end_to_end_heap_local_statement.
Proof. exact (calls_result_loose_heap true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_result_heap_local_landed :
  calls_unaliased_result_heap_local_landed_statement.
Proof.
  exact (calls_result_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma calls_unboxed_loose_heap (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_unboxed_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s code_var type_var fields v Hprem
    Hcalls (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hfields.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  assert (Hleast' : if local
                    then least_solution_local is_local
                           (traverse_unit flag u e) T
                    else least_solution is_local (traverse_unit flag u e) T)
    by (destruct local; cbn [least_with] in Hleast;
        [ exact Hleast | exact (proj1 Hleast) ]).
  pose proof Hcalls as (_ & _ & Hcalls_nr & _).
  unfold FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr in Hcalls_nr.
  apply andb_true_iff in Hcalls_nr as [_ Huses].
  destruct (CHP.calls_states_heap local flag is_local u e T cannot_change
              dominated to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (β & m & t & Hm & Hst & Halloc & Hwhole & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & _ & Hunb).
  destruct (CP.fields_hold_values _ _ _ (Hunb fields Hfields))
    as (ws & Hws & Hvals).
  exists m, t, ws. split; [ exact Hm | split; [ exact Hws | ] ].
  pose proof (TXP.unboxed_result_heap_gen local true fx Hfx is_local _ T
                sets dominated to_unbox e tenv canon (state_statics s)
                (CP.values_only (state_statics t)) valuation valuation' β
                code_var type_var fields v v_mid ws Hleast' Hlocal Hsets
                (FlambdaReaperTypesTraversalProofs
                 .block_fields_of_one_kind_traversal flag u e)
                Henv Hfr Halloc Hst (fun _ => Hwhole) Hcopies Huses
                Hfields Hl Hrv Hvals)
    as Hsat.
  clear - Hsat. induction Hsat as [ | w ty ws tys Hw _ IH]; constructor;
    [ | exact IH ].
  exact (CP.type_satisfied_values_only (state_statics t) (state_statics t) _
           _ _ eq_refl Hw).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_heap :
  calls_unaliased_unboxed_end_to_end_heap_statement.
Proof.
  exact (calls_unboxed_loose_heap false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_heap_landed :
  calls_unaliased_unboxed_end_to_end_heap_landed_statement.
Proof.
  exact (calls_unboxed_loose_heap false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_heap_local :
  calls_unaliased_unboxed_end_to_end_heap_local_statement.
Proof.
  exact (calls_unboxed_loose_heap true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_heap_local_landed :
  calls_unaliased_unboxed_heap_local_landed_statement.
Proof.
  exact (calls_unboxed_loose_heap true landed_fixes TP.landed_fixes_sound).
Qed.
