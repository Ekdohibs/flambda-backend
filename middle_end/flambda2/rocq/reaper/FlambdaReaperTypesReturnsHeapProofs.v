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

(* Proofs of FlambdaReaperTypesReturnsHeapSpec.v. This file needs no
   review.

   As FlambdaReaperTypesReturnsProofs.v, with the primitives of the heap:
   at a return, the node of the result is an alias of the argument
   (FlambdaReaperTypesReturnsProofs.return_alias_gen at [hb = true]), and
   what [rebuilt_value_heap] relates at a node it relates at every alias
   of it ([rebuilt_value_heap_alias]), so the argument has in the rebuilt
   run a value rebuilt under the metadata of the node of the result
   ([return_rebuilt_heap]). The states are those of
   FlambdaReaperTypesCallsHeapProofs.cc_states_heap, and the type is
   rewritten at the node as FlambdaReaperTypesReturnsProofs
   .rewrite_result_type_at_sound_in rewrites it, of the values with each
   location erased ([rewrite_result_type_at_sound_heap]). The lemmas of
   the states take the fragment [loose] as a parameter, for
   FlambdaReaperTypesLooseHeapProofs.v. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperLeastSpec
  FlambdaReaperLocalFields FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperCallingConventionsSpec
  FlambdaReaperTypes FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesHeapSpec FlambdaReaperTypesCallsSpec
  FlambdaReaperDeadFramesSpec FlambdaReaperTypesReturnsSpec
  FlambdaReaperTypesUsedValuesSpec FlambdaReaperTypesCallsUsedSpec
  FlambdaReaperTypesCallsHeapSpec FlambdaReaperTypesReturnsHeapSpec
  FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaNameAbstractionSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesHeapProofs FlambdaReaperTypesCallsProofs
  FlambdaReaperTypesReturnsProofs FlambdaReaperTypesLooseProofs
  FlambdaReaperTypesCallsHeapProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module TP := FlambdaReaperTypesProofs.
Module THP := FlambdaReaperTypesHeapProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module RT := FlambdaReaperTypesReturnsProofs.
Module LP := FlambdaReaperTypesLooseProofs.
Module CHP := FlambdaReaperTypesCallsHeapProofs.

(* ======================================================================== *)
(* The flow at a return, with the heap                                      *)
(* ======================================================================== *)

(* What the rebuild relates at a node, with the heap, it relates at every
   alias of it. *)
Lemma rebuilt_value_heap_alias (is_local : field -> bool) (G : graph)
    (sol : solution) (β : location -> location -> Prop) (v : value)
    (to_ from : node) (v' : value) :
  closed is_local G sol -> s_alias sol to_ from = true ->
  rebuilt_value_heap G sol β v from v' -> rebuilt_value_heap G sol β v to_ v'.
Proof.
  intros Hc Hal.
  pose proof (may_come_from_alias is_local G sol Hc) as Hm.
  destruct v; cbn [rebuilt_value_heap]; intros H; try exact H.
  - destruct H as [Has Heq].
    split; [ exact (any_source_alias _ _ _ Hc _ _ Hal Has) | exact Heq ].
  - destruct H as (sites & vs' & Heq & Ha & Hrest). exists sites, vs'.
    split; [ exact Heq | split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ] ].
  - destruct H as (Heq & Hmc & Hu).
    split; [ exact Heq | split; [ exact (Hm _ _ _ Hal Hmc) | ] ].
    intros Hhu. apply Hu. exact (has_usage_alias _ _ _ Hc _ _ Hhu Hal).
  - destruct H as (a & fields' & Ha & Hrest). exists a, fields'.
    split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ].
  - destruct H as (a & bn & from' & Ha & Hrest). exists a, bn, from'.
    split; [ exact (Hm _ _ _ Hal Ha) | exact Hrest ].
Qed.

(* At a return of a unit of [in_heap_fragment], the argument has in the
   second run a value rebuilt, with the heap, under the metadata of the
   node of the result. *)
Lemma return_rebuilt_heap (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (β : location -> location -> Prop)
    (vars vars' : env) (id : code_id) (c : code) (ac : apply_cont) (i : nat)
    (arg : simple) (r : node) (v : value) :
  in_heap_fragment e = true ->
  closed is_local (graph_of_facts (traverse_unit flag u e))
    (solution_of_tables T) ->
  (id, c) ∈ unit_codes e ->
  ac ∈ applies_to (code_return_continuation c) (code_body c) ->
  nth_error (ac_args ac) i = Some arg ->
  result_node flag u e id i = Some r ->
  s_has_usage (solution_of_tables T) r = true ->
  eval_simple vars arg = Some v ->
  (∀ (x : variable) (w : value),
     lookup x vars = Some w ->
     s_has_usage (solution_of_tables T) (Node_variable x) = true ->
     ∃ w' : value,
       lookup x vars' = Some w'
       ∧ rebuilt_value_heap (graph_of_facts (traverse_unit flag u e))
           (solution_of_tables T) β w (Node_variable x) w') ->
  ∃ v' : value,
    eval_simple vars' arg = Some v'
    ∧ rebuilt_under_heap (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) β
        (metadata_for_name
           (plain_context is_local (traverse_unit flag u e) T sets) r)
        v v'.
Proof.
  intros Hfrag Hc Hin Hac Harg Hrn Hhu Hev Hvars.
  destruct (RT.return_alias_gen true flag is_local u e T id c ac i arg r
              Hfrag Hc Hin Hac Harg Hrn)
    as (Hal & _ & Hsym).
  pose proof (alias_input _ _ _ Hc _ _ Hal) as Hsal.
  pose proof (TP.metadata_for_name_at
                (plain_context is_local (traverse_unit flag u e) T sets) r)
    as [Hat Hfu].
  destruct arg as [x | sym | c0];
    cbn [eval_simple node_of_simple] in Hev, Hsal |- *.
  - destruct (Hvars x v Hev (has_usage_alias _ _ _ Hc _ _ Hhu Hsal))
      as (v' & Hx' & Hrv).
    exists v'. split; [ exact Hx' | ]. right. exists r.
    split; [ exact (rebuilt_value_heap_alias is_local _ _ β v r _ v' Hc Hsal
                      Hrv) | ].
    split; [ exact Hat | intros _; exact Hhu ].
  - injection Hev as <-. exists (Value_symbol sym).
    split; [ reflexivity | ]. right. exists r.
    split; [ | split; [ exact Hat | intros _; exact Hhu ] ].
    cbn [rebuilt_value_heap]. split; [ reflexivity | split ].
    + exact (may_come_from_alias is_local _ _ Hc _ _ _ Hsal
               (Hsym sym eq_refl)).
    + intros Hhu'. exact (has_usage_alias _ _ _ Hc _ _ Hhu' Hsal).
  - injection Hev as <-. exists (Value_const c0).
    split; [ reflexivity | ]. left. exists c0. split; reflexivity.
Qed.

(* ======================================================================== *)
(* The type at the node of a result, with the heap                          *)
(* ======================================================================== *)

(* FlambdaReaperTypesReturnsProofs.rewrite_result_type_at_sound_in, for a
   value rebuilt with the heap under the metadata of the node: the types
   never read the heap. *)
Lemma rewrite_result_type_at_sound_heap (fx : fixes) (loose : bool) :
  TP.fixes_sound fx ->
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop) (r : node) (type_var : variable)
    (v v' : value),
    TP.least_either is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β
      (metadata_for_name (plain_context is_local F T sets) r) v v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type_at fx (plain_context is_local F T sets) tenv canon
         r type_var).
Proof.
  intros Hfx is_local F T sets tenv canon statics statics' valuation
    valuation' β r type_var v v' Hleast Hlocal Hsets Henv Hfrag Halloc Hst
    Hwhole Hcopies Hfind Hl Hv.
  apply (proj1 (THP.type_satisfied_erase statics' valuation' v' _)).
  apply (RT.rewrite_result_type_at_sound_in fx loose Hfx is_local F T sets
           tenv canon (erase_statics statics) (erase_statics statics')
           (erase_env valuation) (erase_env valuation') r type_var
           (erase_locations v) (erase_locations v')).
  - exact Hleast.
  - exact Hlocal.
  - exact Hsets.
  - exact (THP.typing_env_satisfied_strictly_erase _ _ _ Henv).
  - exact (THP.env_in_fragment_erase _ _ _ Hfrag).
  - exact (THP.statics_allocated_erase _ _ Halloc).
  - exact (THP.statics_rebuilt_heap_erase _ _ _ _ _ Hst).
  - intros Hl'. exact (THP.static_sets_whole_erase _ _ _ (Hwhole Hl')).
  - exact (THP.copies_valued_erase _ _ _ _ Hcopies).
  - exact Hfind.
  - rewrite THP.lookup_erase_env, Hl. reflexivity.
  - exact (THP.rebuilt_under_heap_erase _ _ _ _ _ _ Hv).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* At a return, in the fragment [loose] or not. *)
Lemma cc_return_heap_gen (loose local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state) (id : code_id)
    (i : nat) (arg : simple) (r : node) (type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  heap_premises local flag is_local u e T sets ->
  LP.cc_hyps flag u e T cc ccw code_deps ->
  return_premises loose flag u e T n s id i arg r tenv canon valuation
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
  destruct (CHP.cc_states_heap flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (β & m & t & Hm & Hst & Halloc & Hvars & _ & _ & Hwhole).
  destruct (return_rebuilt_heap flag is_local u e T sets β (state_vars s)
              (state_vars t) id c ac i arg r v Hfrag
              (check_closed_sound is_local _ _ Hcheck) Hin Hac Harg Hrn Hhu
              Hev Hvars)
    as (v' & Hv' & Hru).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hv' | ] ].
  exact (rewrite_result_type_at_sound_heap fx loose Hfx is_local _ T sets
           tenv canon (state_statics s) (state_statics t) valuation valuation'
           β r type_var v v' Hle Hlocal Hsets Henv Hfr Halloc Hst
           (fun _ => Hwhole) Hcopies Hfind Hl Hru).
Qed.

Lemma plain_return_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> code_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    id i arg r type_var v Hprem Hres.
  destruct (cc_return_heap_gen false local fx flag is_local u e T _ _ _ b
              sets tenv canon valuation valuation' n s id i arg r type_var v
              Hfx Hprem (LP.cc_hyps_plain flag u e T) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. rewrite <- LP.plain_is_cc.
  split; [ exact Hm | split; [ exact Hx | exact Hsat ] ].
Qed.

Theorem code_result_type_end_to_end_heap :
  code_result_type_end_to_end_heap_statement.
Proof. exact (plain_return_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_heap_landed :
  code_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (plain_return_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem code_result_type_end_to_end_heap_local :
  code_result_type_end_to_end_heap_local_statement.
Proof. exact (plain_return_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem code_result_type_end_to_end_heap_local_landed :
  code_result_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (plain_return_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma cc_return_type_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_code_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s id i arg r type_var v Hprem Hccc Hwid Hres.
  destruct (LP.cc_hyps_checked flag u e T cannot_change Hccc Hwid)
    as (ccw & Hh).
  exact (cc_return_heap_gen false local fx flag is_local u e T _ ccw _ b sets
           tenv canon valuation valuation' n s id i arg r type_var v Hfx Hprem
           Hh Hres).
Qed.

Theorem cc_code_result_type_end_to_end_heap :
  cc_code_result_type_end_to_end_heap_statement.
Proof. exact (cc_return_type_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_code_result_type_end_to_end_heap_landed :
  cc_code_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (cc_return_type_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_code_result_type_end_to_end_heap_local :
  cc_code_result_type_end_to_end_heap_local_statement.
Proof. exact (cc_return_type_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_code_result_type_end_to_end_heap_local_landed :
  cc_code_result_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (cc_return_type_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.
