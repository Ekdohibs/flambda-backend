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

(* Proofs of FlambdaReaperTypesLooseSpec.v. This file needs no review.

   The states that the simulations relate keep the static sets whole
   (FlambdaReaperTypesCallsProofs.cc_states, [calls_states],
   [calls_exit], [repr_exit], by FlambdaReaperTypesCcSetsProofs.v); the
   proofs are those of FlambdaReaperTypesCallsProofs.v and
   FlambdaReaperTypesReturnsProofs.v in the fragment [loose]. *)

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
  FlambdaReaperTypesCallsSpec FlambdaReaperTypesReturnsSpec
  FlambdaReaperTypesLooseSpec FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaNameAbstractionSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesCallsProofs FlambdaReaperTypesReturnsProofs
  FlambdaReaperTypesTraversalProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperLeastProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypingEnvProofs
  FlambdaTypeGrammarProofs FlambdaTypeMeaningBoolProofs
  FlambdaEnvAliasesProofs FlambdaBindingTime.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module TP := FlambdaReaperTypesProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module RT := FlambdaReaperTypesReturnsProofs.
Module U := FlambdaReaperUnboxing.

(* ======================================================================== *)
(* The plain rebuild and the rebuild with changes of calling convention     *)
(* ======================================================================== *)

(* The premises of [cc_states] on the table of the code that changes. *)
Definition cc_hyps (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) (T : tables) (cc ccw : node -> bool)
    (code_deps : finite_map code_dep) : Prop :=
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    ccw cc
  ∧ (∀ (id : code_id) (a a' n y : node),
       cc (Node_code_id id) = false ->
       g_accessor (graph_of_facts (traverse_unit flag u e)) a
         (Field_call_witness Known_arity_code_pointer) n ->
       g_accessor (graph_of_facts (traverse_unit flag u e)) a'
         (Field_call_witness Known_arity_code_pointer) n ->
       (s_any_source (solution_of_tables T) a' = true ->
        s_any_source (solution_of_tables T) a = true)
       ∧ (s_sources (solution_of_tables T) a' y = true ->
          s_sources (solution_of_tables T) a y = true))
  ∧ (∀ (id id' : code_id) (w : node),
       cc (Node_code_id id) = false ->
       g_constructor (graph_of_facts (traverse_unit flag u e)) w
         Field_code_id_of_call_witness (Node_code_id id) ->
       g_constructor (graph_of_facts (traverse_unit flag u e)) w
         Field_code_id_of_call_witness (Node_code_id id') ->
       id' = id)
  ∧ (∀ id : code_id,
       cc (Node_code_id id) = false ->
       lookup id code_deps = lookup id (unit_code_deps flag u e)).

(* Where no code changes. *)
Lemma cc_hyps_plain (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) :
  cc_hyps flag u e T (fun _ => true) (fun _ => true) [].
Proof.
  split; [ constructor; intros; reflexivity | ].
  split; [ intros id0 a a' n0 y H; discriminate H | ].
  split; [ intros id0 id' w H; discriminate H | ].
  intros id0 H. discriminate H.
Qed.

(* Of the tables that pass the checks. *)
Lemma cc_hyps_checked (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node) :
  check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
  check_apply_widgets (traverse_unit flag u e) T = true ->
  ∃ ccw : node -> bool,
    cc_hyps flag u e T (fun n => mem_node n cannot_change) ccw
      (unit_code_deps flag u e).
Proof.
  intros Hccc Hwid.
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  exists ccw. split; [ exact Hcl | split; [ exact Hw | split ] ];
    [ exact Hc | intros id _; reflexivity ].
Qed.

(* The plain rebuild is the rebuild with changes of calling convention
   where no code changes. *)
Lemma plain_is_cc (F : facts) (T : tables) (b : bool) (e : expr) :
  rebuild_unit_cc_with None F (solution_of_tables T) (fun _ => true) [] b e
  = rebuild_unit_from_tables F T b e.
Proof.
  unfold rebuild_unit_from_tables, rebuild_unit.
  exact (RP.rebuild_expr_cc_unchanged _ _ (fun _ => true) [] b
           (static_names code_ids_bound e) (bound_code_metadata e)
           (FlambdaNameAbstractionSpec.fresh_name (names_of e)) None
           (fun _ _ => eq_refl) empty_env e).
Qed.

Lemma cc_result_loose_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  unit_premises local flag is_local u e T sets ->
  cc_hyps flag u e T cc ccw code_deps ->
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
  destruct (CP.cc_states flag is_local u e T cc ccw code_deps b Hlocal Hfrag
              Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals Hccc
              Hwid Hcid Hcd n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & _ & _ & Hwhole).
  destruct (Hvars code_var v Hx Hhu) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  apply (TP.rewrite_result_type_sound_in fx true Hfx is_local _ T sets tenv
           canon (state_statics s) (state_statics t) valuation valuation'
           code_var type_var v v');
    try assumption.
  intros _. exact Hwhole.
Qed.

Lemma plain_result_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unaliased_result_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    code_var type_var v Hprem Hres.
  destruct (cc_result_loose_gen local fx flag is_local u e T _ _ _ b sets
              tenv canon valuation valuation' n s code_var type_var v Hfx
              Hprem (cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem unaliased_result_type_end_to_end :
  unaliased_result_type_end_to_end_statement.
Proof. exact (plain_result_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_result_type_end_to_end_landed :
  unaliased_result_type_end_to_end_landed_statement.
Proof. exact (plain_result_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem unaliased_result_type_end_to_end_local :
  unaliased_result_type_end_to_end_local_statement.
Proof. exact (plain_result_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_result_type_end_to_end_local_landed :
  unaliased_result_type_end_to_end_local_landed_statement.
Proof. exact (plain_result_loose true landed_fixes TP.landed_fixes_sound). Qed.

Lemma cc_result_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_unaliased_result_type_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s code_var type_var v Hprem Hccc Hwid Hres.
  destruct (cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (cc_result_loose_gen local fx flag is_local u e T _ ccw _ b sets tenv
           canon valuation valuation' n s code_var type_var v Hfx Hprem Hh
           Hres).
Qed.

Theorem cc_unaliased_result_type_end_to_end :
  cc_unaliased_result_type_end_to_end_statement.
Proof. exact (cc_result_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_type_end_to_end_landed :
  cc_unaliased_result_type_end_to_end_landed_statement.
Proof. exact (cc_result_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem cc_unaliased_result_type_end_to_end_local :
  cc_unaliased_result_type_end_to_end_local_statement.
Proof. exact (cc_result_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_result_type_end_to_end_local_landed :
  cc_unaliased_result_type_end_to_end_local_landed_statement.
Proof. exact (cc_result_loose true landed_fixes TP.landed_fixes_sound). Qed.

(* ======================================================================== *)
(* The results of code                                                      *)
(* ======================================================================== *)

Lemma cc_return_loose_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (id : code_id)
    (i : nat) (arg : simple) (r : node) (type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  unit_premises local flag is_local u e T sets ->
  cc_hyps flag u e T cc ccw code_deps ->
  return_premises true flag u e T n s id i arg r tenv canon valuation
    valuation' type_var v ->
  ∃ (m : nat) (t : state) (v' : value),
    run m
      (initial_state
         (rebuild_unit_cc_with None (traverse_unit flag u e)
            (solution_of_tables T) cc code_deps b e))
    = Step t
    ∧ eval_simple (state_vars t) arg = Some v'
    ∧ type_satisfied (state_statics t) valuation' v'
        (rewrite_result_type_at fx
           (plain_context is_local (traverse_unit flag u e) T sets) tenv
           canon r type_var).
Proof.
  intros Hfx (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits & Honce
              & Hvals & Hsets)
    (Hccc & Hwid & Hcid & Hcd)
    (Hrun & Hs & (c & ac & Hin & Hse & Hac & Harg) & Hev & Hrn & Hhu & Henv
     & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (CP.cc_states flag is_local u e T cc ccw code_deps b Hlocal Hfrag
              Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals Hccc
              Hwid Hcid Hcd n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & _ & _ & Hwhole).
  destruct (RT.return_rebuilt flag is_local u e T sets (state_vars s)
              (state_vars t) id c ac i arg r v Hfrag
              (check_closed_sound is_local _ _ Hcheck) Hin Hac Harg Hrn Hhu
              Hev Hvars)
    as (v' & Hv' & Hru).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hv' | ] ].
  apply (RT.rewrite_result_type_at_sound_in fx true Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation valuation'
           r type_var v v');
    try assumption.
  intros _. exact Hwhole.
Qed.

Lemma plain_return_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unaliased_code_result_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i arg r type_var v Hprem Hres.
  destruct (cc_return_loose_gen local fx flag is_local u e T _ _ _ b sets
              tenv canon valuation valuation' n s id i arg r type_var v Hfx
              Hprem (cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem unaliased_code_result_end_to_end :
  unaliased_code_result_end_to_end_statement.
Proof. exact (plain_return_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_code_result_end_to_end_landed :
  unaliased_code_result_end_to_end_landed_statement.
Proof. exact (plain_return_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem unaliased_code_result_end_to_end_local :
  unaliased_code_result_end_to_end_local_statement.
Proof. exact (plain_return_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem unaliased_code_result_end_to_end_local_landed :
  unaliased_code_result_end_to_end_local_landed_statement.
Proof. exact (plain_return_loose true landed_fixes TP.landed_fixes_sound). Qed.

Lemma cc_return_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_unaliased_code_result_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i arg r type_var v Hprem Hccc Hwid Hres.
  destruct (cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (cc_return_loose_gen local fx flag is_local u e T _ ccw _ b sets tenv
           canon valuation valuation' n s id i arg r type_var v Hfx Hprem Hh
           Hres).
Qed.

Theorem cc_unaliased_code_result_end_to_end :
  cc_unaliased_code_result_end_to_end_statement.
Proof. exact (cc_return_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_end_to_end_landed :
  cc_unaliased_code_result_end_to_end_landed_statement.
Proof. exact (cc_return_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem cc_unaliased_code_result_end_to_end_local :
  cc_unaliased_code_result_end_to_end_local_statement.
Proof. exact (cc_return_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_unaliased_code_result_end_to_end_local_landed :
  cc_unaliased_code_result_end_to_end_local_landed_statement.
Proof. exact (cc_return_loose true landed_fixes TP.landed_fixes_sound). Qed.

(* ======================================================================== *)
(* The unit rebuilt with unboxing across calls                              *)
(* ======================================================================== *)

Lemma calls_types_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_types_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hcalls Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (CP.calls_exit local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s k vs hp rd
              (CP.unit_premises_used _ _ _ _ _ _ _ Hprem) Hcalls Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Ht & Hst & Halloc & Hwhole).
  exists m, t, vs', rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_symbol_type_sound_in fx true Hfx is_local _ T sets tenv
           canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' sym allocated);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem calls_unaliased_types_end_to_end :
  calls_unaliased_types_end_to_end_statement.
Proof. exact (calls_types_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_types_end_to_end_landed :
  calls_unaliased_types_end_to_end_landed_statement.
Proof. exact (calls_types_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem calls_unaliased_types_end_to_end_local :
  calls_unaliased_types_end_to_end_local_statement.
Proof. exact (calls_types_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_types_end_to_end_local_landed :
  calls_unaliased_types_end_to_end_local_landed_statement.
Proof. exact (calls_types_loose true landed_fixes TP.landed_fixes_sound). Qed.

Lemma calls_result_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_result_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s code_var type_var v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hnone.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (CP.calls_states local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s
              (CP.unit_premises_used _ _ _ _ _ _ _ Hprem) Hcalls Hng Hrun
              Hlet)
    as (m & t & Hm & Hst & Halloc & Hwhole & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & Hplain & _).
  exists m, t, v_mid.
  split; [ exact Hm | split; [ exact (Hplain Hin Hnone) | ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_result_type_sound_in fx true Hfx is_local _ T sets tenv
           canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' code_var type_var v v_mid);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem calls_unaliased_result_end_to_end :
  calls_unaliased_result_end_to_end_statement.
Proof. exact (calls_result_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_result_end_to_end_landed :
  calls_unaliased_result_end_to_end_landed_statement.
Proof. exact (calls_result_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem calls_unaliased_result_end_to_end_local :
  calls_unaliased_result_end_to_end_local_statement.
Proof. exact (calls_result_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_result_end_to_end_local_landed :
  calls_unaliased_result_end_to_end_local_landed_statement.
Proof. exact (calls_result_loose true landed_fixes TP.landed_fixes_sound). Qed.

Lemma calls_unboxed_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unaliased_unboxed_end_to_end_with local fx.
Proof.
  intros (Hb & Hal & Hr & Hn) flag is_local u e T cannot_change dominated
    to_unbox b nr sets tenv canon valuation valuation' n s code_var type_var
    fields v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hfields.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  pose proof Hcalls as (_ & _ & Hcalls_nr & _).
  unfold FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr in Hcalls_nr.
  apply andb_true_iff in Hcalls_nr as [_ Huses].
  destruct (CP.calls_states local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s
              (CP.unit_premises_used _ _ _ _ _ _ _ Hprem) Hcalls Hng Hrun
              Hlet)
    as (m & t & Hm & Hst & Halloc & Hwhole & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & _ & Hunb).
  destruct (CP.fields_hold_values _ _ _ (Hunb fields Hfields))
    as (ws & Hws & Hvals).
  exists m, t, ws. split; [ exact Hm | split; [ exact Hws | ] ].
  assert (Hsat :
    Forall2 (type_satisfied (CP.values_only (state_statics t)) valuation') ws
      (rewrite_unboxed_result_type fx
         (plain_context is_local (traverse_unit flag u e) T sets) tenv
         (U.bound_code_my_closures e) canon code_var type_var fields)).
  { eapply (TP.rewrite_unboxed_result_type_sound_in fx Hb Hal Hr Hn is_local
              _ T sets tenv canon (state_statics s)
              (CP.values_only (state_statics t)) valuation valuation' true);
      try eassumption.
    - exact (FlambdaReaperTypesTraversalProofs
             .block_fields_of_one_kind_traversal flag u e).
    - intros _. exact Hwhole. }
  clear - Hsat. induction Hsat as [ | w ty ws tys Hw _ IH]; constructor;
    [ | exact IH ].
  exact (CP.type_satisfied_values_only (state_statics t) (state_statics t) _
           _ _ eq_refl Hw).
Qed.

Theorem calls_unaliased_unboxed_end_to_end :
  calls_unaliased_unboxed_end_to_end_statement.
Proof. exact (calls_unboxed_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_unboxed_end_to_end_landed :
  calls_unaliased_unboxed_end_to_end_landed_statement.
Proof.
  exact (calls_unboxed_loose false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unaliased_unboxed_end_to_end_local :
  calls_unaliased_unboxed_end_to_end_local_statement.
Proof. exact (calls_unboxed_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem calls_unaliased_unboxed_end_to_end_local_landed :
  calls_unaliased_unboxed_end_to_end_local_landed_statement.
Proof.
  exact (calls_unboxed_loose true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* With changes of representation                                           *)
(* ======================================================================== *)

Lemma repr_types_loose (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> repr_unaliased_types_end_to_end_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox tcr b nr
    sets tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hrepr Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (CP.repr_exit local flag is_local u e T cannot_change dominated
              to_unbox tcr b nr sets n s k vs hp rd
              (CP.unit_premises_used _ _ _ _ _ _ _ Hprem) Hrepr Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Ht & Hst & Halloc & Hwhole).
  exists m, t, vs', rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_symbol_type_sound_in fx true Hfx is_local _ T sets tenv
           canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' sym allocated);
    try assumption.
  intros _. exact Hwhole.
Qed.

Theorem repr_unaliased_types_end_to_end :
  repr_unaliased_types_end_to_end_statement.
Proof. exact (repr_types_loose false all_fixes TP.all_fixes_sound). Qed.

Theorem repr_unaliased_types_end_to_end_landed :
  repr_unaliased_types_end_to_end_landed_statement.
Proof. exact (repr_types_loose false landed_fixes TP.landed_fixes_sound). Qed.

Theorem repr_unaliased_types_end_to_end_local :
  repr_unaliased_types_end_to_end_local_statement.
Proof. exact (repr_types_loose true all_fixes TP.all_fixes_sound). Qed.

Theorem repr_unaliased_types_end_to_end_local_landed :
  repr_unaliased_types_end_to_end_local_landed_statement.
Proof. exact (repr_types_loose true landed_fixes TP.landed_fixes_sound). Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Module LooseRuns.

Import LooseExample.
Local Open Scope positive_scope.

(* The state of the run of the unit where [x60] is bound. *)
Definition state_at : state :=
  match run 3 (initial_state UnboxCallsExample.unit_expr) with
  | Step s => s
  | _ => initial_state UnboxCallsExample.unit_expr
  end.

Lemma run_at :
  run 3 (initial_state UnboxCallsExample.unit_expr) = Step state_at.
Proof. vm_compute. reflexivity. Qed.

Lemma lookup_tenv (n : name) (ty : flambda_type)
    (bt : FlambdaBindingTime.Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = Some (ty, bt) ->
  n = Name_var 93 ∧ (ty = closure_type ∨ ty = unknown Value).
Proof.
  intros Hl.
  apply (FlambdaMapProofs.keyed_lookup_in name_eqb
           FlambdaTypeGrammarProofs.name_eqb_eq) in Hl.
  vm_compute in Hl.
  destruct Hl as [[= <- <- _] | [[= <- <- _] | []]]; split; auto.
Qed.

Lemma tenv_satisfied (statics : static_env) :
  typing_env_satisfied_strictly statics valuation tenv.
Proof.
  split; [ vm_compute; reflexivity | split ].
  - intros n ty bt w Hl Hw. right.
    destruct (lookup_tenv n ty bt Hl) as [-> [-> | ->]];
      vm_compute in Hw; injection Hw as <-;
      apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff;
      vm_compute; reflexivity.
  - change (Typing_env.aliases tenv) with Aliases.empty.
    apply FlambdaEnvAliasesProofs.empty_respected.
Qed.

End LooseRuns.

Lemma loose_example_gen (fx : fixes) :
  rewrite_result_type fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       UnboxCallsExample.facts UnboxCallsExample.tables
       UnboxCallsExample.sets)
    LooseExample.tenv UnboxCallsExample.canon 60%positive 93%positive
  = LooseExample.closure_type ->
  loose_example_with fx.
Proof.
  intros Hrw.
  destruct FlambdaReaperUnboxingRefinesProofs.unboxed_call_example
    as (Hslots & Hfrag & Hstatic & Hrk & Hwf & Hone & _ & _ & _ & Hccc
        & Hwid & _).
  split.
  { split; [ exact Hslots | split ].
    { split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
               vm_compute; reflexivity | ].
      left. intros f. reflexivity. }
    split; [ exact Hfrag | split ].
    { exact (RP.static_constants_any_kind_arguments _ _ Hstatic). }
    split; [ exact Hrk | split; [ exact Hwf | split; [ exact Hone | ] ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    exact CP.UnboxCallsRuns.sets_ok. }
  split; [ exact Hccc | split; [ exact Hwid | ] ].
  split.
  { intros H.
    destruct (Keyed.lookup name_eqb (Name_var 93%positive)
                (Typing_env.names_to_types LooseExample.tenv))
      as [[ty bt] | ] eqn:Hl;
      [ | vm_compute in Hl; discriminate Hl ].
    destruct (H _ _ _ Hl) as [Hshape _].
    pose proof Hl as Hl'. vm_compute in Hl'. injection Hl' as <- _.
    vm_compute in Hshape. discriminate Hshape. }
  split.
  { exists LooseRuns.state_at.
    split; [ exact LooseRuns.run_at | ].
    split; [ vm_compute; discriminate | ].
    split; [ vm_compute; reflexivity | ].
    split; [ vm_compute; reflexivity | ].
    split; [ apply LooseRuns.tenv_satisfied | ].
    split.
    { intros n ty bt Hl.
      destruct (LooseRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
        (split; [ vm_compute; reflexivity | ]);
        intros s k w Hin; vm_compute in Hin; destruct Hin. }
    split.
    { intros s x k m Hcan. unfold value_of_name.
      exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
               _ _ _ _ _ _ _ _ (LooseRuns.tenv_satisfied
                                  (state_statics LooseRuns.state_at))
               Hcan). }
    split; [ vm_compute; discriminate | reflexivity ]. }
  split; [ exact Hrw | ].
  exists
    (match run 3 (initial_state LooseExample.cc_rebuilt) with
     | Step t => t
     | _ => initial_state LooseExample.cc_rebuilt
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
  vm_compute. reflexivity.
Qed.

Theorem loose_example : loose_example_statement.
Proof. apply loose_example_gen. vm_compute. reflexivity. Qed.

Theorem loose_example_landed : loose_example_landed_statement.
Proof. apply loose_example_gen. vm_compute. reflexivity. Qed.
