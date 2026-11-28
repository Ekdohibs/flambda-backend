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

(* Proofs of FlambdaReaperTypesCallsHeapSpec.v. This file needs no review.

   As FlambdaReaperTypesCallsProofs.v, with the primitives of the heap.
   [cc_states_heap] reads off the simulation of the rebuild with changes
   of calling convention with the heap (FlambdaReaperTypesCallsExitProofs
   .cc_reach at [hb = true]) what the statements of
   FlambdaReaperTypesHeapSpec.v ask of the runs: [statics_rebuilt_heap]
   (FlambdaReaperTypesHeapProofs.statics_rel_rebuilt_heap_used, for the
   static values of the symbols with a use) and [rebuilt_value_heap]
   (FlambdaReaperTypesHeapProofs.rel_rebuilt_value_heap), for the
   correspondence of the simulation, and the exits observed equal. The
   unit rebuilt with unboxing across calls is related to the rebuild with
   changes of calling convention by [uc] with the heap
   (FlambdaReaperTypesCallsExitProofs.uc_of_calls_gen at [hb = true]),
   whose states have one heap and bind the same static values. The types
   are those of FlambdaReaperTypesHeapProofs.v and
   FlambdaReaperTypesTraversalHeapProofs.v, which erase the locations. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaBindersSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperSolutionProofs FlambdaReaperFlowSpec FlambdaReaperFlowProofs
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperCallingConventionsSpec FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesHeapSpec FlambdaReaperTypesCallsSpec
  FlambdaReaperTypesUsedValuesSpec FlambdaReaperTypesCallsUsedSpec
  FlambdaReaperTypesCallsHeapSpec FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventionsProofs FlambdaReaperLocalFieldsProofs
  FlambdaReaperUnboxing FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperUnboxingRefinesCallsHeapSpec FlambdaNameAbstractionSpec
  FlambdaReaperTypesProofs FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesHeapProofs FlambdaReaperTypesTraversalHeapProofs
  FlambdaReaperTypesCallsExitProofs FlambdaReaperTypesCallsProofs
  FlambdaReaperTypesUsedValuesProofs FlambdaReaperLeastProofs.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec
  FlambdaEnvAliases.
From OxCamlFlambda2Types Require FlambdaTypeMeaningProofs
  FlambdaTypeGrammarProofs FlambdaTypeMeaningBoolProofs FlambdaTypingEnvProofs
  FlambdaEnvAliasesProofs.
Import ListNotations.

Module RP := FlambdaReaperRebuildProofs.
Module TP := FlambdaReaperTypesProofs.
Module CX := FlambdaReaperTypesCallsExitProofs.
Module CP := FlambdaReaperTypesCallsProofs.
Module THP := FlambdaReaperTypesHeapProofs.
Module TXP := FlambdaReaperTypesTraversalHeapProofs.
Module CCP := FlambdaReaperCallingConventionsProofs.
Module U := FlambdaReaperUnboxing.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UCS := FlambdaReaperUnboxingRefinesCallsSimProofs.
Module UVP := FlambdaReaperTypesUsedValuesProofs.

(* ======================================================================== *)
(* What the simulation of the rebuild relates, with the heap                *)
(* ======================================================================== *)

(* [statics_rebuilt_heap] reads only the static values of the second
   run. *)
Lemma statics_rebuilt_heap_values (G : graph) (sol : solution)
    (β : location -> location -> Prop)
    (statics statics' statics'' : static_env) :
  static_values statics'' = static_values statics' ->
  statics_rebuilt_heap G sol β statics statics' ->
  statics_rebuilt_heap G sol β statics statics''.
Proof.
  intros Heq H s allocated Hl Hhu. rewrite Heq. exact (H s allocated Hl Hhu).
Qed.

Lemma no_static_sets (sol : solution) (statics statics' : static_env) :
  false = true -> static_sets_whole sol statics statics'.
Proof. intros H. discriminate H. Qed.

(* Two states related by the simulation of the rebuild with changes of
   calling convention, with the primitives of the heap, the static blocks
   of the symbols with a use holding values: the static environments are
   related by [statics_rebuilt_heap] for the correspondence of the
   locations of the simulation, that of the first binds allocations of
   their own, a variable with a use bound in the first is bound in the
   second to what [rebuilt_value_heap] relates to it, and the expression
   of the second is the rebuild of that of the first. *)
Lemma related_facts_heap (is_local : field -> bool) (G : graph)
    (sol : solution) (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep) (keep : value_slot -> bool)
    (codes : list (code_id * code)) (symbols : list symbol)
    (u : unit_interface) (slots_read : Prop) (F : facts) (b : bool)
    (ids : list code_id) (ff : positive) (cc : node -> bool)
    (code_deps : finite_map code_dep) (cmds : finite_map code_metadata)
    (types : subkinds_rewriting) (L : location -> option node)
    (s t : state) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  RP.related G sol noted deps cds keep codes symbols u True slots_read F b
    ids ff false True cc code_deps cmds types L s t ->
  statics_of_used_values (fun sym => s_has_usage sol (Node_symbol sym))
    (state_statics s) ->
  statics_rebuilt_heap G sol (RP.kept_beta sol True L) (state_statics s)
    (state_statics t)
  ∧ statics_allocated sol (state_statics s)
  ∧ (∀ (x : variable) (v : value),
       lookup x (state_vars s) = Some v ->
       s_has_usage sol (Node_variable x) = true ->
       ∃ v' : value,
         lookup x (state_vars t) = Some v'
         ∧ rebuilt_value_heap G sol (RP.kept_beta sol True L) v
             (Node_variable x) v')
  ∧ ∃ en : rebuild_env,
      state_expr t
      = rebuild_expr_cc F sol cc code_deps b ids cmds ff types en
          (state_expr s).
Proof.
  intros Hc Hexact
    [[(ctx0 & ce0 & _ & _ & _ & Hflows & _ & Hsf & _) _]
     (ctx & en & ce & rets & _ & Hexpr & Henv & _ & Hst & _ & _ & _)]
    Hvals.
  split; [ | split; [ | split ] ].
  - apply (THP.statics_rel_rebuilt_heap_used is_local G sol keep L _ _ Hc
             Hexact); [ | exact Hvals ].
    do 9 eexists. exact Hst.
  - intros x allocated Hl. exact (proj1 (Hsf x allocated Hl)).
  - intros x v Hx Hhu.
    destruct (Henv x v Hx) as [_ Hused].
    assert (Hu : is_var_used sol x (kind_of_value v) = true).
    { unfold is_var_used. destruct (kind_of_value v); try reflexivity;
        exact Hhu. }
    destruct (Hused Hu) as (v' & Hx' & Hrel).
    exists v'. split; [ exact Hx' | ].
    apply (THP.rel_rebuilt_value_heap is_local G sol Hc Hexact _ _ _ _ _ _ v
             (Node_variable x) v' Hrel).
    intros c ->. exact (Hflows x _ Hx).
  - exists en. exact Hexpr.
Qed.

(* The states of the run of a unit of [in_heap_fragment] and of its
   rebuild with changes of calling convention, for any table closed under
   the rules, the static blocks of the symbols with a use holding values:
   each state that the first comes to, and from which it steps, is
   related to one that the second comes to as the statements of types
   with the heap ask, for a correspondence [β] of the locations, the
   rebuild of a [Let] of a variable with a use is a [Let] or [Invalid],
   and an exit of the first is one of the second through the same
   continuation, the two observed equal up to the heap. *)
Lemma cc_states_heap (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  variables_bound_once e = true ->
  used_values T e = true ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    ccw cc ->
  (∀ (id : code_id) (a a' n y : node),
     cc (Node_code_id id) = false ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source (solution_of_tables T) a' = true ->
      s_any_source (solution_of_tables T) a = true)
     ∧ (s_sources (solution_of_tables T) a' y = true ->
        s_sources (solution_of_tables T) a y = true)) ->
  (∀ (id id' : code_id) (w : node),
     cc (Node_code_id id) = false ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  (∀ id : code_id,
     cc (Node_code_id id) = false ->
     lookup id code_deps = lookup id (unit_code_deps flag u e)) ->
  ∀ (n : nat) (s : state),
    run n (initial_state e) = Step s -> step s <> Undefined ->
    ∃ (β : location -> location -> Prop) (m : nat) (t : state),
      run m
        (initial_state
           (rebuild_unit_cc_with None (traverse_unit flag u e)
              (solution_of_tables T) cc code_deps b e))
      = Step t
      ∧ statics_rebuilt_heap (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) β (state_statics s) (state_statics t)
      ∧ statics_allocated (solution_of_tables T) (state_statics s)
      ∧ (∀ (x : variable) (v : value),
           lookup x (state_vars s) = Some v ->
           s_has_usage (solution_of_tables T) (Node_variable x) = true ->
           ∃ v' : value,
             lookup x (state_vars t) = Some v'
             ∧ rebuilt_value_heap (graph_of_facts (traverse_unit flag u e))
                 (solution_of_tables T) β v (Node_variable x) v')
      ∧ (∀ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
           state_expr s = Let y kind d body ->
           s_has_usage (solution_of_tables T) (Node_variable y) = true ->
           state_expr t = Invalid
           ∨ ∃ (d' : named) (body' : expr),
               state_expr t = Let y kind d' body')
      ∧ (∀ (k : continuation) (vs : list value) (hp : heap)
           (rd : read_record),
           step s = Exit k vs hp rd ->
           ∃ (vs' : list value) (hp' : heap) (rd' : read_record),
             step t = Exit k vs' hp' rd'
             ∧ exits_observed_equal vs hp vs' hp').
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
    Hvals Hccc Hwid Hcid Hcd n s Hrun Hs.
  pose proof (proj1 (RP.any_kind_arguments_checked (unit_codes e)) e Hdc)
    as Hcf.
  assert (Hkinded : RP.calls_kinded [] e = true).
  { apply (proj1 RP.calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  destruct (CX.cc_reach true false flag is_local u e T cc ccw code_deps b
              None Hlocal Hfrag Hcf Hkinded Hwf Hexits Hcheck Hlrs
              Hcheck_exact (fun _ _ _ _ _ _ _ _ _ _ _ _ => I)
              (RP.no_types_deps _) (RP.no_types_conts e) Hrk Hccc Hwid Hcid
              Hcd (THP.closures_exact_of_bound_once_heap flag u e Hfrag Hwf
                     Honce))
    as (noted & deps & cds & Hreach).
  destruct (Hreach n s Hrun Hs) as (L & m & t & Hm & Hrel & Hans).
  pose proof (UVP.unit_of_values_with_runs _ e n s Hvals Hrun) as Hov.
  destruct (related_facts_heap is_local _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
              _ s t (check_closed_sound is_local _ _ Hcheck)
              (check_field_used_exact_sound is_local _ _ Hcheck_exact)
              Hrel Hov)
    as (Hst & Halloc & Hvars & en & Hexpr).
  exists (RP.kept_beta (solution_of_tables T) True L), m, t.
  split; [ exact Hm | ].
  split; [ exact Hst | split; [ exact Halloc | split; [ exact Hvars | ] ] ].
  split.
  - intros y kind d body Hse Hhu. rewrite Hexpr, Hse.
    exact (CP.rebuild_let_used _ _ _ _ _ _ _ _ _ en y kind d body Hhu).
  - intros k vs hp rd Hstep. unfold RP.answered_h in Hans.
    rewrite Hstep in Hans.
    destruct Hans as (vs' & hp' & rd' & Ht & HE).
    exists vs', hp', rd'. split; [ exact Ht | exact (proj1 HE) ].
Qed.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

Lemma heap_premises_of_used (local : bool)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) :
  used_premises local flag is_local u e T sets ->
  heap_premises local flag is_local u e T sets.
Proof.
  intros (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10).
  exact (conj H1 (conj H2
           (conj (FlambdaReaperFlowProofs.in_flow_fragment_in_heap_fragment e
                    H3)
              (conj H4 (conj H5 (conj H6 (conj H7 (conj H8
                 (conj H9 H10))))))))).
Qed.

Theorem heap_premises_instance : heap_premises_instance_statement.
Proof. exact heap_premises_of_used. Qed.

(* ======================================================================== *)
(* The result types, of the rebuild with changes of calling convention      *)
(* ======================================================================== *)

Lemma cc_result_heap_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  heap_premises local flag is_local u e T sets ->
  cc_closed (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T)
    ccw cc ->
  (∀ (id : code_id) (a a' n y : node),
     cc (Node_code_id id) = false ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a
       (Field_call_witness Known_arity_code_pointer) n ->
     g_accessor (graph_of_facts (traverse_unit flag u e)) a'
       (Field_call_witness Known_arity_code_pointer) n ->
     (s_any_source (solution_of_tables T) a' = true ->
      s_any_source (solution_of_tables T) a = true)
     ∧ (s_sources (solution_of_tables T) a' y = true ->
        s_sources (solution_of_tables T) a y = true)) ->
  (∀ (id id' : code_id) (w : node),
     cc (Node_code_id id) = false ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id) ->
     g_constructor (graph_of_facts (traverse_unit flag u e)) w
       Field_code_id_of_call_witness (Node_code_id id') ->
     id' = id) ->
  (∀ id : code_id,
     cc (Node_code_id id) = false ->
     lookup id code_deps = lookup id (unit_code_deps flag u e)) ->
  result_premises (traverse_unit flag u e) T e n s tenv canon valuation
    valuation' code_var type_var v ->
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
    Hccc Hwid Hcid Hcd
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (cc_states_heap flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (β & m & t & Hm & Hst & Halloc & Hvars & _ & _).
  destruct (Hvars code_var v Hx Hhu) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  exact (THP.rewrite_result_type_sound_heap_gen fx false Hfx is_local _ T
           sets tenv canon (state_statics s) (state_statics t) valuation
           valuation' β code_var type_var v v' Hle Hlocal Hsets Henv Hfr
           Halloc Hst (no_static_sets _ _ _) Hcopies Hfind Hl Hrv).
Qed.

(* The plain rebuild is the rebuild with changes of calling convention
   where no code changes. *)
Lemma plain_result_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    code_var type_var v Hprem Hres.
  destruct (cc_result_heap_gen local fx flag is_local u e T (fun _ => true)
              (fun _ => true) [] b sets tenv canon valuation valuation' n s
              code_var type_var v Hfx Hprem
              ltac:(constructor; intros; reflexivity)
              ltac:(intros id a a' n0 y H; discriminate H)
              ltac:(intros id id' w H; discriminate H)
              ltac:(intros id H; discriminate H) Hres)
    as (m & t & v' & Hm & Hx & Hsat).
  exists m, t, v'. split; [ | split; [ exact Hx | exact Hsat ] ].
  unfold rebuild_unit_from_tables, rebuild_unit.
  rewrite <- (RP.rebuild_expr_cc_unchanged _ _ (fun _ => true) [] b
                (static_names code_ids_bound e) (bound_code_metadata e)
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)) None
                (fun _ _ => eq_refl)).
  exact Hm.
Qed.

Theorem result_type_end_to_end_heap : result_type_end_to_end_heap_statement.
Proof. exact (plain_result_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem result_type_end_to_end_heap_landed :
  result_type_end_to_end_heap_landed_statement.
Proof.
  exact (plain_result_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem result_type_end_to_end_heap_local :
  result_type_end_to_end_heap_local_statement.
Proof. exact (plain_result_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem result_type_end_to_end_heap_local_landed :
  result_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (plain_result_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma cc_result_type_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s code_var type_var v Hprem Hccc Hwid Hres.
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  exact (cc_result_heap_gen local fx flag is_local u e T _ ccw _ b sets tenv
           canon valuation valuation' n s code_var type_var v Hfx Hprem Hcl
           Hw Hc (fun _ _ => eq_refl) Hres).
Qed.

Theorem cc_result_type_end_to_end_heap :
  cc_result_type_end_to_end_heap_statement.
Proof. exact (cc_result_type_heap_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_result_type_end_to_end_heap_landed :
  cc_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (cc_result_type_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_heap_local :
  cc_result_type_end_to_end_heap_local_statement.
Proof. exact (cc_result_type_heap_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_result_type_end_to_end_heap_local_landed :
  cc_result_type_end_to_end_heap_local_landed_statement.
Proof.
  exact (cc_result_type_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The three runs                                                           *)
(* ======================================================================== *)

(* The states of the three runs, with the heap. Where the run of the unit,
   which does not go wrong, is at a [Let] of a variable with a use, the
   run of the unit rebuilt with unboxing across calls comes to a state
   whose static values are related to the unit's by
   [statics_rebuilt_heap], and in which each variable of the unit with a
   use that it does not unbox has the value that the rebuild with changes
   of calling convention has for it, related by [rebuilt_value_heap], and
   each that it unboxes has the variables of its fields at the paths of
   the fields in that value. *)
Lemma calls_states_heap (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (n : nat) (s : state) :
  heap_premises local flag is_local u e T sets ->
  calls_premises flag u e T cannot_change dominated to_unbox nr ->
  ~ behaves (initial_state e) Goes_wrong ->
  run n (initial_state e) = Step s ->
  (∃ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
     state_expr s = Let y kind d body
     ∧ s_has_usage (solution_of_tables T) (Node_variable y) = true) ->
  ∃ (β : location -> location -> Prop) (m : nat) (t : state),
    run m
      (initial_state
         (calls_rebuilt flag u e T cannot_change dominated to_unbox b))
    = Step t
    ∧ statics_rebuilt_heap (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) β (state_statics s)
        (CP.values_only (state_statics t))
    ∧ statics_allocated (solution_of_tables T) (state_statics s)
    ∧ ∀ (x : variable) (v : value),
        lookup x (state_vars s) = Some v ->
        s_has_usage (solution_of_tables T) (Node_variable x) = true ->
        ∃ v_mid : value,
          rebuilt_value_heap (graph_of_facts (traverse_unit flag u e))
            (solution_of_tables T) β v (Node_variable x) v_mid
          ∧ (x ∈ variables_of e ->
             Keyed.lookup node_eqb (Node_variable x)
               (fst (U.unit_unboxed_fields (traverse_unit flag u e) T
                       dominated to_unbox e))
             = None ->
             lookup x (state_vars t) = Some v_mid)
          ∧ ∀ fields : U.unboxed_fields,
              Keyed.lookup node_eqb (Node_variable x)
                (fst (U.unit_unboxed_fields (traverse_unit flag u e) T
                        dominated to_unbox e))
              = Some fields ->
              URP.fields_hold (state_vars t) fields v_mid.
Proof.
  intros Hprem (Hccc & Hwid & Hcalls & Hreads) Hng Hrun Hlet.
  pose proof Hprem as (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits
                       & Honce & Hvals & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  assert (Hs : step s <> Undefined).
  { intros Hu. apply Hng. exists (S n).
    replace (S n) with (n + 1) by lia.
    rewrite run_add, Hrun, run_succ, Hu. reflexivity. }
  destruct (cc_states_heap flag is_local u e T _ ccw
              (unit_code_deps flag u e) b Hlocal Hfrag Hdc Hrk Hwf Hexits
              Hcheck Hlrs Hcheck_exact Honce Hvals Hcl Hw Hc
              (fun _ _ => eq_refl) n s Hrun Hs)
    as (β & m & t & Hm & Hst & Halloc & Hvars & Hlets & _).
  assert (Ht : step t <> Undefined).
  { intros Hu. apply Hng.
    destruct (CCP.rebuild_cc_refines_up_to_heap_local flag is_local u e T
                cannot_change b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
                Hcheck_exact Hccc Hwid)
      as (_ & _ & Hgw).
    apply Hgw. exists (S m).
    replace (S m) with (m + 1) by lia.
    unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
    unfold initial_state, rebuild_unit_cc_with in Hm.
    rewrite run_add, Hm, run_succ, Hu. reflexivity. }
  destruct (CX.uc_of_calls_gen true flag is_local u e T cannot_change
              dominated to_unbox b nr Hlocal Hfrag Hcheck Hcalls Hreads)
    as (dead & CS & NR & RCH & OV & REM & Hdead & Hdist & Huc & Hfl & Hov
        & Hrm).
  destruct (CX.uc_reach _ Hdist dead CS NR RCH OV REM _ _ Huc Hfl Hov Hrm m
              t Hm Ht)
    as (m3 & t3 & Hm3 & Hrel).
  destruct Hlet as (y & kind & d & body & Hse & Hhu).
  destruct (Hlets y kind d body Hse Hhu) as [Hinv | (d' & body' & Ht_let)].
  { exfalso. apply Ht. rewrite (state_eta t), Hinv. apply step_invalid. }
  pose proof (CP.uc_let _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel Ht_let)
    as [Hplain Hunb].
  pose proof (CP.uc_values _ _ _ _ _ _ _ _ _ Hrel) as Hv.
  exists β, m3, t3. split; [ exact Hm3 | ].
  split; [ apply (statics_rebuilt_heap_values _ _ _ _ (state_statics t));
           [ exact Hv | exact Hst ] | ].
  split; [ exact Halloc | ].
  intros x v Hx Hhux.
  destruct (Hvars x v Hx Hhux) as (v_mid & Hx_t & Hrv).
  exists v_mid. split; [ exact Hrv | split ].
  - intros Hin Hnone. rewrite <- Hx_t. symmetry. apply Hplain.
    unfold UCS.plain2, URP.plain_variable.
    unfold Keyed.is_bound. rewrite Hnone.
    destruct (URP.unit_unboxed_fields_fresh (traverse_unit flag u e) T
                dominated to_unbox e)
      as [_ Hfresh].
    rewrite (Hfresh x Hin).
    destruct (among x dead) eqn:Hd; [ | reflexivity ].
    rewrite (Hdead x Hd) in Hhux. discriminate Hhux.
  - intros fields Hfields. exact (Hunb x fields v_mid Hfields Hx_t).
Qed.

(* Where the unit exits, the unit rebuilt with unboxing across calls exits
   through the same continuation, the exits observed equal up to the
   heap, from a state whose static values are related to the unit's by
   [statics_rebuilt_heap]. *)
Lemma calls_exit_heap (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  heap_premises local flag is_local u e T sets ->
  calls_premises flag u e T cannot_change dominated to_unbox nr ->
  run n (initial_state e) = Step s ->
  step s = Exit k vs hp rd ->
  ∃ (β : location -> location -> Prop) (m : nat) (t : state)
    (vs' : list value) (hp' : heap) (rd' : read_record),
    run m
      (initial_state
         (calls_rebuilt flag u e T cannot_change dominated to_unbox b))
    = Step t
    ∧ step t = Exit k vs' hp' rd'
    ∧ exits_observed_equal vs hp vs' hp'
    ∧ statics_rebuilt_heap (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) β (state_statics s)
        (CP.values_only (state_statics t))
    ∧ statics_allocated (solution_of_tables T) (state_statics s).
Proof.
  intros Hprem (Hccc & Hwid & Hcalls & Hreads) Hrun Hexit.
  pose proof Hprem as (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits
                       & Honce & Hvals & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (CP.cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  destruct (cc_states_heap flag is_local u e T _ ccw
              (unit_code_deps flag u e) b Hlocal Hfrag Hdc Hrk Hwf Hexits
              Hcheck Hlrs Hcheck_exact Honce Hvals Hcl Hw Hc
              (fun _ _ => eq_refl) n s Hrun
              ltac:(rewrite Hexit; discriminate))
    as (β & m & t & Hm & Hst & Halloc & _ & _ & Hex).
  destruct (Hex k vs hp rd Hexit) as (vs' & hp' & rd' & Ht & Hobs).
  destruct (CX.uc_of_calls_gen true flag is_local u e T cannot_change
              dominated to_unbox b nr Hlocal Hfrag Hcheck Hcalls Hreads)
    as (dead & CS & NR & RCH & OV & REM & Hdead & Hdist & Huc & Hfl & Hov
        & Hrm).
  destruct (CX.uc_exit _ Hdist dead CS NR RCH OV REM _ _ Huc Hfl Hov Hrm m
              t k vs' hp' rd' Hm Ht)
    as (m3 & t3 & rd3 & Hm3 & Ht3 & Hrel).
  exists β, m3, t3, vs', hp', rd3.
  split; [ exact Hm3 | split; [ exact Ht3 | split; [ exact Hobs | ] ] ].
  split; [ | exact Halloc ].
  apply (statics_rebuilt_heap_values _ _ _ _ (state_statics t));
    [ exact (CP.uc_values _ _ _ _ _ _ _ _ _ Hrel) | exact Hst ].
Qed.

(* ======================================================================== *)
(* The statements of the unboxing across calls                              *)
(* ======================================================================== *)

Lemma unbox_calls_types_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_calls_types_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hcalls Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (calls_exit_heap local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s k vs hp rd Hprem Hcalls Hrun Hexit)
    as (β & m & t & vs' & hp' & rd' & Hm & Ht & Hobs & Hst & Halloc).
  exists m, t, vs', hp', rd'.
  split; [ exact Hm | split; [ exact Ht | split; [ exact Hobs | ] ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  exact (THP.rewrite_symbol_type_sound_heap_gen fx false Hfx is_local _ T
           sets tenv canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' β sym allocated Hle Hlocal Hsets Henv Hfr
           Halloc Hst (no_static_sets _ _ _) Hcopies Hl).
Qed.

Theorem unbox_calls_types_end_to_end_heap :
  unbox_calls_types_end_to_end_heap_statement.
Proof.
  exact (unbox_calls_types_heap_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_heap_landed :
  unbox_calls_types_end_to_end_heap_landed_statement.
Proof.
  exact (unbox_calls_types_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_heap_local :
  unbox_calls_types_end_to_end_heap_local_statement.
Proof.
  exact (unbox_calls_types_heap_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_heap_local_landed :
  unbox_calls_types_end_to_end_heap_local_landed_statement.
Proof.
  exact (unbox_calls_types_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma unbox_calls_result_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx ->
  unbox_calls_result_type_end_to_end_heap_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s code_var type_var v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hnone.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (CP.least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (calls_states_heap local flag is_local u e T cannot_change
              dominated to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (β & m & t & Hm & Hst & Halloc & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & Hplain & _).
  exists m, t, v_mid.
  split; [ exact Hm | split; [ exact (Hplain Hin Hnone) | ] ].
  apply (CP.type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  exact (THP.rewrite_result_type_sound_heap_gen fx false Hfx is_local _ T
           sets tenv canon (state_statics s) (CP.values_only (state_statics t))
           valuation valuation' β code_var type_var v v_mid Hle Hlocal Hsets
           Henv Hfr Halloc Hst (no_static_sets _ _ _) Hcopies Hfind Hl Hrv).
Qed.

Theorem unbox_calls_result_type_end_to_end_heap :
  unbox_calls_result_type_end_to_end_heap_statement.
Proof.
  exact (unbox_calls_result_heap_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_heap_landed :
  unbox_calls_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (unbox_calls_result_heap_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_heap_local :
  unbox_calls_result_type_end_to_end_heap_local_statement.
Proof.
  exact (unbox_calls_result_heap_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem unbox_calls_result_end_to_end_heap_local_landed :
  unbox_calls_result_end_to_end_heap_local_landed_statement.
Proof.
  exact (unbox_calls_result_heap_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma calls_unboxed_result_heap_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx ->
  calls_unboxed_result_type_end_to_end_heap_with local fx.
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
  destruct (calls_states_heap local flag is_local u e T cannot_change
              dominated to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (β & m & t & Hm & Hst & Halloc & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & _ & Hunb).
  destruct (CP.fields_hold_values _ _ _ (Hunb fields Hfields))
    as (ws & Hws & Hvals).
  exists m, t, ws. split; [ exact Hm | split; [ exact Hws | ] ].
  pose proof (TXP.unboxed_result_heap_gen local false fx Hfx is_local _ T
                sets dominated to_unbox e tenv canon (state_statics s)
                (CP.values_only (state_statics t)) valuation valuation' β
                code_var type_var fields v v_mid ws Hleast' Hlocal Hsets
                (FlambdaReaperTypesTraversalProofs
                 .block_fields_of_one_kind_traversal flag u e)
                Henv Hfr Halloc Hst (no_static_sets _ _ _) Hcopies Huses
                Hfields Hl Hrv Hvals)
    as Hsat.
  clear - Hsat. induction Hsat as [ | w ty ws tys Hw _ IH]; constructor;
    [ | exact IH ].
  exact (CP.type_satisfied_values_only (state_statics t) (state_statics t) _
           _ _ eq_refl Hw).
Qed.

Theorem calls_unboxed_result_type_end_to_end_heap :
  calls_unboxed_result_type_end_to_end_heap_statement.
Proof.
  exact (calls_unboxed_result_heap_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_heap_landed :
  calls_unboxed_result_type_end_to_end_heap_landed_statement.
Proof.
  exact (calls_unboxed_result_heap_gen false landed_fixes
           TP.landed_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_heap_local :
  calls_unboxed_result_type_end_to_end_heap_local_statement.
Proof.
  exact (calls_unboxed_result_heap_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_end_to_end_heap_local_landed :
  calls_unboxed_result_end_to_end_heap_local_landed_statement.
Proof.
  exact (calls_unboxed_result_heap_gen true landed_fixes
           TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Module CallsHeapTypesRuns.

Import CallsHeapTypesExample.
Local Open Scope positive_scope.

Ltac in_cases H :=
  repeat match type of H with
         | _ ∨ _ => destruct H as [H | H]
         end;
  try discriminate H; try contradiction H.

Lemma constructors :
  f_constructor CH.facts
  = [(Node_variable 60, Field_call_witness Known_arity_code_pointer,
      Node_made 7);
     (Node_variable 60, Field_call_witness Unknown_arity_code_pointer,
      Node_made 8);
     (Node_variable 60, Field_function_slot 70, Node_variable 60);
     (Node_variable 11, Field_block 0 Value, Node_variable 10);
     (Node_variable 11, Field_is_int, Node_all_constants);
     (Node_variable 11, Field_get_tag, Node_all_constants);
     (Node_variable 10, Field_block 0 Value, Node_all_constants);
     (Node_variable 10, Field_is_int, Node_all_constants);
     (Node_variable 10, Field_get_tag, Node_all_constants);
     (Node_made 8, Field_return_of_call Exn, Node_made 4);
     (Node_made 8, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 8, Field_code_id_of_call_witness, Node_code_id 50);
     (Node_made 7, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 7, Field_return_of_call Exn, Node_made 4);
     (Node_made 7, Field_code_id_of_call_witness, Node_code_id 50)].
Proof. vm_compute. reflexivity. Qed.

Lemma sets_ok : sets_of_graph (graph_of_facts CH.facts) sets.
Proof.
  intros fdecls sites slot id Hl Hsites.
  exists (hd [] sets).
  assert (Hb0 : is_bound slot fdecls = true)
    by (unfold is_bound; rewrite Hl; reflexivity).
  pose proof (proj2 (Hsites slot id Hl slot (sites slot)) (conj Hb0 eq_refl))
    as Hself.
  apply CP.UnboxCallsRuns.cons_in in Hself. rewrite constructors in Hself.
  cbn [In] in Hself. in_cases Hself.
  injection Hself as Hs70 E70. subst slot.
  assert (Hdom : ∀ g : function_slot,
             is_bound g fdecls = true -> g = 70%positive).
  { intros g Hg. pose proof (proj2 (Hsites 70%positive id Hl g (sites g))
                               (conj Hg eq_refl)) as Hc.
    apply CP.UnboxCallsRuns.cons_in in Hc. rewrite constructors, <- Hs70 in Hc.
    cbn [In] in Hc. in_cases Hc.
    injection Hc as <- _. reflexivity. }
  split; [ left; reflexivity | split ].
  - intros g c code Hin. destruct Hin as [Hin | []].
    injection Hin as <- <- _. split; [ exact Hb0 | congruence ].
  - intros g Hg. rewrite (Hdom g Hg). reflexivity.
Qed.

(* The state of the run of the unit at the [Let] of [x43] in the code. *)
Definition state_at : state :=
  match run 6 (initial_state CH.unit_expr) with
  | Step s => s
  | _ => initial_state CH.unit_expr
  end.

Lemma run_at : run 6 (initial_state CH.unit_expr) = Step state_at.
Proof. vm_compute. reflexivity. Qed.

Lemma not_wrong : ~ behaves (initial_state CH.unit_expr) Goes_wrong.
Proof.
  intros Hw.
  assert (Hex : behaves (initial_state CH.unit_expr)
                  (Exits 1 [Value_const (Tagged_immediate 2)]
                     CH.heap_at_exit)).
  { exists 20%nat, no_reads. vm_compute. reflexivity. }
  discriminate (FlambdaSemanticsProofs.behaviour_unique _ _ _ Hw Hex).
Qed.

Lemma lookup_tenv (n : name) (ty : flambda_type)
    (bt : FlambdaBindingTime.Binding_time.With_name_mode.t) :
  Keyed.lookup name_eqb n (Typing_env.names_to_types tenv) = Some (ty, bt) ->
  n = Name_var 90 ∧ (ty = param_type ∨ ty = unknown Value).
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

End CallsHeapTypesRuns.

Lemma calls_heap_types_example_gen (fx : fixes) :
  rewrite_unboxed_result_type fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       CallsHeapTypesExample.CH.facts CallsHeapTypesExample.CH.tables
       CallsHeapTypesExample.sets)
    CallsHeapTypesExample.tenv
    (FlambdaReaperUnboxing.bound_code_my_closures
       CallsHeapTypesExample.CH.unit_expr)
    CallsHeapTypesExample.canon 40%positive 90%positive
    CallsHeapTypesExample.fields
  = [unknown Value] ->
  calls_heap_types_example_with fx.
Proof.
  intros Hrw.
  split.
  { split; [ intros f Hf; discriminate Hf | split ].
    { split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
               vm_compute; reflexivity | ].
      left. intros f. reflexivity. }
    split; [ vm_compute; reflexivity | split ].
    { vm_compute; reflexivity. }
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; [ vm_compute; reflexivity | ].
    exact CallsHeapTypesRuns.sets_ok. }
  split; [ vm_compute; reflexivity | ].
  split.
  { split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    split; vm_compute; reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exists CallsHeapTypesRuns.state_at.
    split.
    - split; [ exact CallsHeapTypesRuns.run_at | ].
      split; [ vm_compute; discriminate | ].
      split; [ vm_compute; reflexivity | ].
      split; [ vm_compute; reflexivity | ].
      split; [ apply CallsHeapTypesRuns.tenv_satisfied | ].
      split.
      { intros n ty bt Hl.
        destruct (CallsHeapTypesRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
          [ | apply TP.unknown_fragment ].
        split; [ vm_compute; reflexivity | ].
        intros s k w Hin Hw. vm_compute in Hin.
        CallsHeapTypesRuns.in_cases Hin. }
      split.
      { intros s x k m Hcan. unfold value_of_name.
        exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
                 _ _ _ _ _ _ _ _ (CallsHeapTypesRuns.tenv_satisfied
                                    (state_statics
                                       CallsHeapTypesRuns.state_at))
                 Hcan). }
      split; [ vm_compute; discriminate | reflexivity ].
    - split; [ exact CallsHeapTypesRuns.not_wrong | split ].
      + exists 43%positive, Value,
          (FlambdaReaperUnboxing.Examples.load 0 (Var 40%positive)),
          (Let 44%positive Value
             (CallsHeapTypesExample.CH.store (Var 43%positive) 2%Z)
             (FlambdaReaperUnboxing.Examples.return_to 3%positive
                [Var 44%positive])).
        split; vm_compute; reflexivity.
      + vm_compute. tauto. }
  split; [ exact Hrw | ].
  exists
    (match run 6 (initial_state CallsHeapTypesExample.CH.rebuilt) with
     | Step t => t
     | _ => initial_state CallsHeapTypesExample.CH.rebuilt
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
  vm_compute. reflexivity.
Qed.

Theorem calls_heap_types_example : calls_heap_types_example_statement.
Proof. apply calls_heap_types_example_gen. vm_compute. reflexivity. Qed.

Theorem calls_heap_types_example_landed :
  calls_heap_types_example_landed_statement.
Proof. apply calls_heap_types_example_gen. vm_compute. reflexivity. Qed.
