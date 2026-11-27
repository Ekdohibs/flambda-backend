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

(* Proofs of FlambdaReaperTypesCallsSpec.v. This file needs no review.

   The statements of types of FlambdaReaperTypesSpec.v are applied to
   the states that the simulations relate
   (FlambdaReaperTypesCallsExitProofs.v): [cc_states] reads off the
   simulation of the rebuild with changes of calling convention what the
   statements ask of the runs ([statics_rebuilt], [statics_allocated],
   [rebuilt_value] at the node of a variable, by
   FlambdaReaperTypesProofs.statics_rel_rebuilt and
   [rel_rebuilt_value]); [uc_let] and [uc_exit_values] read off that of
   the unboxing across calls the environments and the static values of
   the third run. The second and third runs bind the same static values
   and other code: a type is satisfied under the static values alone
   ([values_only], [type_satisfied_values_only]), as [statics_rebuilt]
   reads only those ([statics_rebuilt_values]).

   The proofs are those of the statements of
   FlambdaReaperTypesCallsUsedSpec.v, with the wider check of static
   blocks ([used_premises], each [_used_gen]): the static environments
   are related by [statics_rebuilt] from the static values of the symbols
   with a use alone (FlambdaReaperTypesUsedValuesProofs
   .statics_rel_rebuilt_used, in [related_facts]). The statements of
   FlambdaReaperTypesCallsSpec.v are their instances
   ([unit_premises_used]). *)

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
  FlambdaReaperTypesCallsSpec FlambdaReaperTypesUsedValuesSpec
  FlambdaReaperTypesCallsUsedSpec FlambdaRenameApartSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventionsProofs FlambdaReaperLocalFieldsProofs
  FlambdaReaperClosuresExactProofs FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaHeapFreeSpec
  FlambdaHeapFreeProofs FlambdaNameAbstractionSpec FlambdaReaperTypesProofs
  FlambdaReaperTypesTermProofs FlambdaReaperTypesTraversalProofs
  FlambdaReaperTypesCallsExitProofs FlambdaReaperTypesReprExitProofs
  FlambdaReaperTypesCcSetsProofs FlambdaClosuresProofs
  FlambdaReaperTypesUsedValuesProofs.
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
Module U := FlambdaReaperUnboxing.
Module URP := FlambdaReaperUnboxingRefinesProofs.
Module UCS := FlambdaReaperUnboxingRefinesCallsSimProofs.
Module UVP := FlambdaReaperTypesUsedValuesProofs.

(* ======================================================================== *)
(* Static values                                                            *)
(* ======================================================================== *)

(* [statics_rebuilt] reads only the static values of the second run. *)
Lemma statics_rebuilt_values (G : graph) (sol : solution)
    (statics statics' statics'' : static_env) :
  static_values statics'' = static_values statics' ->
  statics_rebuilt G sol statics statics' ->
  statics_rebuilt G sol statics statics''.
Proof.
  intros Heq H s allocated Hl Hhu. rewrite Heq. exact (H s allocated Hl Hhu).
Qed.

(* [static_sets_whole] reads only the static values of the second run. *)
Lemma static_sets_whole_values (sol : solution)
    (statics statics' statics'' : static_env) :
  static_values statics'' = static_values statics' ->
  static_sets_whole sol statics statics' ->
  static_sets_whole sol statics statics''.
Proof.
  intros Heq H s slot syms decls vs Hl Hhu.
  destruct (H s slot syms decls vs Hl Hhu) as [Hs (vs' & Hall)].
  split; [ exact Hs | ]. exists vs'. intros g s_g Hg.
  destruct (Hall g s_g Hg) as [H1 H2]. rewrite Heq.
  split; [ exact H1 | exact H2 ].
Qed.

(* The static values of a static environment, with no code. *)
Definition values_only (statics : static_env) : static_env :=
  {| static_code := []; static_values := static_values statics |}.

(* A type satisfied under the static values alone is satisfied under any
   static environment that binds them. *)
Lemma type_satisfied_values_only (statics statics' : static_env)
    (vars : env) (v : value) (t : flambda_type) :
  static_values statics' = static_values statics ->
  type_satisfied (values_only statics) vars v t ->
  type_satisfied statics' vars v t.
Proof.
  intros Heq.
  apply FlambdaTypeMeaningProofs.type_satisfied_statics_extend.
  split.
  - intros id c Hl. discriminate Hl.
  - intros s allocated Hl. cbn [values_only static_values] in Hl.
    rewrite Heq. exact Hl.
Qed.

(* ======================================================================== *)
(* No location in the flow fragment                                         *)
(* ======================================================================== *)

(* FlambdaReaperRebuildProofs.free_run_no_location, of every value of the
   state. *)
Lemma free_run_state_no_location (e : expr) (n : nat) (s : state) :
  FlambdaHeapFreeSpec.expr_free_of_heap e = true ->
  run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
  RP.state_no_location s.
Proof.
  intros He.
  assert (Hgen : ∀ (n : nat) (s0 : state),
            FlambdaHeapFreeProofs.state_free s0 -> RP.state_no_location s0 ->
            run n s0 = Step s -> RP.state_no_location s).
  { induction n0 as [ | n0 IH]; intros s0 Hfree Hnl Hrun.
    - injection Hrun as <-. exact Hnl.
    - rewrite run_succ in Hrun.
      pose proof (RP.free_step_no_location s0 Hfree Hnl) as Hs.
      destruct (step s0) as [s' | k' vs' hp' rd' | ] eqn:Hstep;
        try discriminate Hrun.
      exact (IH s' (proj1 Hs) (proj2 Hs) Hrun). }
  intros Hrun.
  apply (Hgen n (State e [] [] empty_static_env empty_heap no_reads)
           ltac:(split; [ exact He | split; [ constructor | ] ];
                        intros id c Hl; discriminate Hl)
           ltac:(split; [ | reflexivity ];
                 split; [ constructor | split; [ exact I | ] ];
                 split; [ constructor | apply heap_values_all_empty ])
           Hrun).
Qed.

(* ======================================================================== *)
(* What the simulation of the rebuild relates                               *)
(* ======================================================================== *)

(* Two states related by the simulation of the rebuild with changes of
   calling convention, with no primitive of the heap, the closures made
   where the graph has only the facts of their sets, the static blocks
   of the symbols with a use holding values: the static environments are
   related by [statics_rebuilt], that of the first binds
   allocations of their own, a variable with a use bound in the first is
   bound in the second to what [rebuilt_value] relates to it, and the
   expression of the second is the rebuild of that of the first. *)
Lemma related_facts (is_local : field -> bool) (G : graph) (sol : solution)
    (noted : symbol -> Prop) (deps : closure_dep -> Prop)
    (cds : finite_map code_dep) (keep : value_slot -> bool)
    (codes : list (code_id * code)) (symbols : list symbol)
    (u : unit_interface) (slots_read : Prop) (F : facts) (b : bool)
    (ids : list code_id) (ff : positive) (cc : node -> bool)
    (code_deps : finite_map code_dep) (cmds : finite_map code_metadata)
    (types : subkinds_rewriting) (L : location -> option node)
    (s t : state) :
  closed is_local G sol -> field_used_exact is_local G sol ->
  RP.related G sol noted deps cds keep codes symbols u False slots_read F b
    ids ff false True cc code_deps cmds types L s t ->
  RP.state_no_location s ->
  statics_of_used_values (fun sym => s_has_usage sol (Node_symbol sym))
    (state_statics s) ->
  statics_rebuilt G sol (state_statics s) (state_statics t)
  ∧ statics_allocated sol (state_statics s)
  ∧ (∀ (x : variable) (v : value),
       lookup x (state_vars s) = Some v ->
       s_has_usage sol (Node_variable x) = true ->
       ∃ v' : value,
         lookup x (state_vars t) = Some v'
         ∧ rebuilt_value G sol v (Node_variable x) v')
  ∧ ∃ en : rebuild_env,
      state_expr t
      = rebuild_expr_cc F sol cc code_deps b ids cmds ff types en
          (state_expr s).
Proof.
  intros Hc Hexact
    [[(ctx0 & ce0 & _ & _ & _ & Hflows & _ & Hsf & _) _]
     (ctx & en & ce & rets & _ & Hexpr & Henv & _ & Hst & _ & _ & _)]
    Hnl Hvals.
  destruct Hnl as [(Hvars_nl & _ & Hst_nl & _) _].
  pose proof (RP.env_no_location_of _ Hvars_nl) as Henv_nl.
  pose proof (RP.statics_no_location_of _ Hst_nl) as Hstatics_nl.
  split; [ | split; [ | split ] ].
  - apply (UVP.statics_rel_rebuilt_used is_local G sol keep _ _ Hc Hexact);
      [ | exact Hstatics_nl | exact Hvals ].
    do 10 eexists. exact Hst.
  - intros x allocated Hl. exact (proj1 (Hsf x allocated Hl)).
  - intros x v Hx Hhu.
    destruct (Henv x v Hx) as [_ Hused].
    assert (Hu : is_var_used sol x (kind_of_value v) = true).
    { unfold is_var_used. destruct (kind_of_value v); try reflexivity;
        exact Hhu. }
    destruct (Hused Hu) as (v' & Hx' & Hrel).
    exists v'. split; [ exact Hx' | ].
    apply (TP.rel_rebuilt_value is_local G sol Hc Hexact _ _ _ _ _ _ v
             (Node_variable x) v' Hrel).
    + apply RP.no_location_iff. exact (Henv_nl x v Hx).
    + intros c ->. exact (Hflows x _ Hx).
  - exists en. exact Hexpr.
Qed.

(* The rebuild keeps a [Let] of a variable with a use, or makes it
   [Invalid] where the variable has no source. *)
Lemma rebuild_let_used (F : facts) (sol : solution) (cc : node -> bool)
    (code_deps : finite_map code_dep) (b : bool) (ids : list code_id)
    (cmds : finite_map code_metadata) (ff : positive)
    (types : subkinds_rewriting) (en : rebuild_env) (y : variable)
    (kind : flambda_kind) (d : named) (body : expr) :
  s_has_usage sol (Node_variable y) = true ->
  rebuild_expr_cc F sol cc code_deps b ids cmds ff types en
    (Let y kind d body)
  = Invalid
  ∨ ∃ (d' : named) (body' : expr),
      rebuild_expr_cc F sol cc code_deps b ids cmds ff types en
        (Let y kind d body)
      = Let y kind d' body'.
Proof.
  intros Hhu.
  assert (Hu : is_var_used sol y kind = true).
  { unfold is_var_used. destruct kind; try reflexivity; exact Hhu. }
  cbn [rebuild_expr_cc].
  destruct (is_pending_projection en y d);
    [ right; do 2 eexists; reflexivity | ].
  destruct d; cbv zeta; rewrite ?Hu; cbn [orb];
    match goal with
    | |- context [if ?c then Invalid else _] =>
      destruct c; [ left; reflexivity | right; do 2 eexists; reflexivity ]
    end.
Qed.

(* The states of the run of the unit and of its rebuild with changes of
   calling convention, for any table closed under the rules, the static
   blocks of the symbols with a use holding values: each state that the
   first comes to, and from which it steps, is related to one that the
   second comes to as the statements of types ask, the rebuild of a [Let]
   of a variable with a use is a [Let] or [Invalid], and an exit of the
   first is one of the second through the same continuation with the
   same heap. *)
Lemma cc_states_used (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
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
    ∃ (m : nat) (t : state),
      run m
        (initial_state
           (rebuild_unit_cc_with None (traverse_unit flag u e)
              (solution_of_tables T) cc code_deps b e))
      = Step t
      ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) (state_statics s) (state_statics t)
      ∧ statics_allocated (solution_of_tables T) (state_statics s)
      ∧ (∀ (x : variable) (v : value),
           lookup x (state_vars s) = Some v ->
           s_has_usage (solution_of_tables T) (Node_variable x) = true ->
           ∃ v' : value,
             lookup x (state_vars t) = Some v'
             ∧ rebuilt_value (graph_of_facts (traverse_unit flag u e))
                 (solution_of_tables T) v (Node_variable x) v')
      ∧ (∀ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
           state_expr s = Let y kind d body ->
           s_has_usage (solution_of_tables T) (Node_variable y) = true ->
           state_expr t = Invalid
           ∨ ∃ (d' : named) (body' : expr),
               state_expr t = Let y kind d' body')
      ∧ (∀ (k : continuation) (vs : list value) (hp : heap)
           (rd : read_record),
           step s = Exit k vs hp rd ->
           ∃ (vs' : list value) (rd' : read_record),
             step t = Exit k vs' hp rd')
      ∧ static_sets_whole (solution_of_tables T) (state_statics s)
          (state_statics t).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
    Hvals Hccc Hwid Hcid Hcd n s Hrun Hs.
  pose proof (proj1 (RP.any_kind_arguments_checked (unit_codes e)) e Hdc)
    as Hcf.
  assert (Hkinded : RP.calls_kinded [] e = true).
  { apply (proj1 RP.calls_kinded_of_results e [] [] [] Hrk).
    intros k l Hl. discriminate Hl. }
  destruct (CX.cc_reach false false flag is_local u e T cc ccw code_deps b
              None Hlocal Hfrag Hcf Hkinded Hwf Hexits Hcheck Hlrs
              Hcheck_exact (fun _ _ _ _ _ _ _ _ _ _ _ _ => I)
              (RP.no_types_deps _) (RP.no_types_conts e) Hrk Hccc Hwid Hcid
              Hcd
              (FlambdaReaperClosuresExactProofs.closures_exact_of_bound_once
                 flag u e Hfrag Hwf Honce))
    as (noted & deps & cds & Hreach).
  destruct (Hreach n s Hrun Hs) as (L & m & t & Hm & Hrel & Hans).
  pose proof (free_run_state_no_location e n s
                (proj1 (RP.flow_fragment_free _ _) e
                   (proj1 (RP.in_fragment_parts false e Hfrag)))
                Hrun)
    as Hnl.
  pose proof (UVP.unit_of_values_with_runs _ e n s Hvals Hrun) as Hov.
  destruct (related_facts is_local _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ s t
              (check_closed_sound is_local _ _ Hcheck)
              (check_field_used_exact_sound is_local _ _ Hcheck_exact)
              Hrel Hnl Hov)
    as (Hst & Halloc & Hvars & en & Hexpr).
  exists m, t. split; [ exact Hm | ].
  split; [ exact Hst | split; [ exact Halloc | split; [ exact Hvars | ] ] ].
  split; [ | split ].
  - intros y kind d body Hse Hhu. rewrite Hexpr, Hse.
    exact (rebuild_let_used _ _ _ _ _ _ _ _ _ en y kind d body Hhu).
  - intros k vs hp rd Hstep. unfold RP.answered_h in Hans.
    rewrite Hstep in Hans.
    destruct Hans as (vs' & hp' & rd' & Ht & _ & HE).
    destruct (HE (fun H => H)) as [-> _].
    exists vs', rd'. exact Ht.
  - destruct Hrel as [_ (ctx & en' & ce & rets & _ & _ & _ & _ & Hsr & _)].
    apply (TP.statics_rel_whole _ _ _ _ _ ltac:(do 10 eexists; exact Hsr)).
    + exact (FlambdaClosuresProofs.well_formed_sets_bound u e n s Hwf Hrun).
    + exact (FlambdaReaperTypesCcSetsProofs.cc_rebuilt_sets_bound _ _ _ _ _ b
               u e m t Hwf Hm).
Qed.

(* The same, for a unit whose static blocks hold values. *)
Lemma cc_states (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  any_kind_arguments e = true ->
  results_kinded [] [] e = true ->
  well_formed_unit u e = true ->
  exits_one_value u e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  check_local_reads_used is_local (traverse_unit flag u e) T = true ->
  check_field_used_exact is_local (traverse_unit flag u e) T = true ->
  variables_bound_once e = true ->
  unit_of_values e = true ->
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
    ∃ (m : nat) (t : state),
      run m
        (initial_state
           (rebuild_unit_cc_with None (traverse_unit flag u e)
              (solution_of_tables T) cc code_deps b e))
      = Step t
      ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
          (solution_of_tables T) (state_statics s) (state_statics t)
      ∧ statics_allocated (solution_of_tables T) (state_statics s)
      ∧ (∀ (x : variable) (v : value),
           lookup x (state_vars s) = Some v ->
           s_has_usage (solution_of_tables T) (Node_variable x) = true ->
           ∃ v' : value,
             lookup x (state_vars t) = Some v'
             ∧ rebuilt_value (graph_of_facts (traverse_unit flag u e))
                 (solution_of_tables T) v (Node_variable x) v')
      ∧ (∀ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
           state_expr s = Let y kind d body ->
           s_has_usage (solution_of_tables T) (Node_variable y) = true ->
           state_expr t = Invalid
           ∨ ∃ (d' : named) (body' : expr),
               state_expr t = Let y kind d' body')
      ∧ (∀ (k : continuation) (vs : list value) (hp : heap)
           (rd : read_record),
           step s = Exit k vs hp rd ->
           ∃ (vs' : list value) (rd' : read_record),
             step t = Exit k vs' hp rd')
      ∧ static_sets_whole (solution_of_tables T) (state_statics s)
          (state_statics t).
Proof.
  intros Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
    Hvals.
  exact (cc_states_used flag is_local u e T cc ccw code_deps b Hlocal Hfrag
           Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
           (UVP.unit_of_values_with_instance _ e Hvals)).
Qed.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* What the statements of types and the simulations ask of the tables,
   from the least solution. *)
Lemma least_with_checks (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables) :
  is_local_slots_only is_local ->
  least_with local is_local (traverse_unit flag u e) T e ->
  TP.least_either is_local (traverse_unit flag u e) T
  ∧ check_closed is_local (traverse_unit flag u e) T = true
  ∧ check_local_reads_used is_local (traverse_unit flag u e) T = true
  ∧ check_field_used_exact is_local (traverse_unit flag u e) T = true.
Proof.
  intros Hlocal Hleast.
  assert (Hll : least_solution_local is_local (traverse_unit flag u e) T
                ∧ TP.least_either is_local (traverse_unit flag u e) T).
  { destruct local; cbn [least_with] in Hleast.
    - split; [ exact Hleast | right; exact Hleast ].
    - destruct Hleast as [Hleast Hslots]. split; [ | left; exact Hleast ].
      exact (FlambdaReaperLeastProofs.least_local_of_least is_local _ T
               Hleast
               (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
                  is_local u e T Hlocal Hslots)). }
  destruct Hll as [Hll Hle].
  destruct (FlambdaReaperLeastProofs.least_local_checks is_local _ T Hll)
    as [Hcheck Hlrs].
  split; [ exact Hle | split; [ exact Hcheck | split; [ exact Hlrs | ] ] ].
  exact (FlambdaReaperLeastProofs.least_field_used_exact_local is_local _ T
           Hll).
Qed.

(* The premises of FlambdaReaperTypesCallsSpec.v give those with the
   wider check of static blocks. *)
Lemma unit_premises_used (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) :
  unit_premises local flag is_local u e T sets ->
  used_premises local flag is_local u e T sets.
Proof.
  intros (H1 & H2 & H3 & H4 & H5 & H6 & H7 & H8 & H9 & H10).
  exact (conj H1 (conj H2 (conj H3 (conj H4 (conj H5 (conj H6 (conj H7
           (conj H8 (conj (UVP.unit_of_values_with_instance _ e H9)
                       H10))))))))).
Qed.

(* A statement of FlambdaReaperTypesCallsSpec.v, from the one of
   FlambdaReaperTypesCallsUsedSpec.v that [H] proves. *)
Ltac used_instance H :=
  let Hu := fresh "Hu" in
  pose proof H as Hu; red in Hu; red; intros;
  match goal with
  | Hp : unit_premises _ _ _ _ _ _ _ |- _ =>
      eapply Hu; [ exact (unit_premises_used _ _ _ _ _ _ _ Hp) | .. ];
      eassumption
  end.

(* ======================================================================== *)
(* The result types, of the rebuild with changes of calling convention      *)
(* ======================================================================== *)

Lemma cc_result_gen (local : bool) (fx : fixes)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cc ccw : node -> bool) (code_deps : finite_map code_dep) (b : bool)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (valuation valuation' : env) (n : nat) (s : state)
    (code_var type_var : variable) (v : value) :
  TP.fixes_sound fx ->
  used_premises local flag is_local u e T sets ->
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
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (cc_states_used flag is_local u e T cc ccw code_deps b Hlocal
              Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce Hvals
              Hccc Hwid Hcid Hcd n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & _ & _).
  destruct (Hvars code_var v Hx Hhu) as (v' & Hx' & Hrv).
  exists m, t, v'. split; [ exact Hm | split; [ exact Hx' | ] ].
  apply (TP.rewrite_result_type_sound_in fx false Hfx is_local _ T sets tenv
           canon (state_statics s) (state_statics t) valuation valuation'
           code_var type_var v v');
    try assumption.
  intros H. discriminate H.
Qed.

(* The plain rebuild is the rebuild with changes of calling convention
   where no code changes. *)
Lemma plain_result_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> result_type_end_to_end_used_with local fx.
Proof.
  intros Hfx flag is_local u e T b sets tenv canon valuation valuation' n s
    code_var type_var v Hprem Hres.
  destruct (cc_result_gen local fx flag is_local u e T (fun _ => true)
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

Lemma plain_result_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> result_type_end_to_end_with local fx.
Proof. intros Hfx. used_instance (plain_result_used_gen local fx Hfx). Qed.

Theorem result_type_end_to_end : result_type_end_to_end_statement.
Proof. exact (plain_result_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem result_type_end_to_end_landed :
  result_type_end_to_end_landed_statement.
Proof. exact (plain_result_gen false landed_fixes TP.landed_fixes_sound). Qed.

Theorem result_type_end_to_end_local : result_type_end_to_end_local_statement.
Proof. exact (plain_result_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem result_type_end_to_end_local_landed :
  result_type_end_to_end_local_landed_statement.
Proof. exact (plain_result_gen true landed_fixes TP.landed_fixes_sound). Qed.

(* The checks of the table give what the simulation asks of it. *)
Lemma cc_checks (flag : reaper_preserve_direct_calls) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node) :
  check_cc_closed (traverse_unit flag u e) T cannot_change = true ->
  check_apply_widgets (traverse_unit flag u e) T = true ->
  (∃ ccw : node -> bool,
     cc_closed (graph_of_facts (traverse_unit flag u e))
       (solution_of_tables T) ccw (fun n => mem_node n cannot_change))
  ∧ (∀ (id : code_id) (a a' n y : node),
       mem_node (Node_code_id id) cannot_change = false ->
       g_accessor (graph_of_facts (traverse_unit flag u e)) a
         (Field_call_witness Known_arity_code_pointer) n ->
       g_accessor (graph_of_facts (traverse_unit flag u e)) a'
         (Field_call_witness Known_arity_code_pointer) n ->
       (s_any_source (solution_of_tables T) a' = true ->
        s_any_source (solution_of_tables T) a = true)
       ∧ (s_sources (solution_of_tables T) a' y = true ->
          s_sources (solution_of_tables T) a y = true))
  ∧ (∀ (id id' : code_id) (w : node),
       mem_node (Node_code_id id) cannot_change = false ->
       g_constructor (graph_of_facts (traverse_unit flag u e)) w
         Field_code_id_of_call_witness (Node_code_id id) ->
       g_constructor (graph_of_facts (traverse_unit flag u e)) w
         Field_code_id_of_call_witness (Node_code_id id') ->
       id' = id).
Proof.
  intros Hccc Hwid.
  split; [ exact (FlambdaReaperCallingConventionsProofs.check_cc_closed_sound
                    _ _ _ Hccc) | split ].
  - intros id a a' n y _.
    exact (FlambdaReaperCallingConventionsProofs.check_apply_widgets_sound
             _ _ Hwid a a' n y).
  - intros id id' w _.
    exact (FlambdaReaperCallingConventionsProofs.check_witness_code_ids_sound
             _ (FlambdaReaperCallingConventionsProofs.traverse_witness_code_ids
                  flag u e) w id id').
Qed.

Lemma cc_result_type_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_result_type_end_to_end_used_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change b sets tenv canon valuation
    valuation' n s code_var type_var v Hprem Hccc Hwid Hres.
  destruct (cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  exact (cc_result_gen local fx flag is_local u e T _ ccw _ b sets tenv canon
           valuation valuation' n s code_var type_var v Hfx Hprem Hcl Hw Hc
           (fun _ _ => eq_refl) Hres).
Qed.

Lemma cc_result_type_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> cc_result_type_end_to_end_with local fx.
Proof. intros Hfx. used_instance (cc_result_type_used_gen local fx Hfx). Qed.

Theorem cc_result_type_end_to_end : cc_result_type_end_to_end_statement.
Proof. exact (cc_result_type_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem cc_result_type_end_to_end_landed :
  cc_result_type_end_to_end_landed_statement.
Proof.
  exact (cc_result_type_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem cc_result_type_end_to_end_local :
  cc_result_type_end_to_end_local_statement.
Proof. exact (cc_result_type_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem cc_result_type_end_to_end_local_landed :
  cc_result_type_end_to_end_local_landed_statement.
Proof.
  exact (cc_result_type_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* What the simulation of the unboxing across calls relates                 *)
(* ======================================================================== *)

(* Related states bind the same static values. *)
Lemma uc_values (M : list (node * U.unboxed_fields)) (dead : list variable)
    (CS : code_id -> option UCS.code_shape) (NR : code_id -> bool)
    (RCH : bool -> variable -> code_id -> Prop) (OV REM : code_id -> Prop)
    (t t' : state) :
  UCS.related2 M dead CS NR RCH OV REM t t' ->
  static_values (state_statics t') = static_values (state_statics t).
Proof.
  intros [(K & _ & _ & _ & [Hv _] & _)
         | [(ap & ap' & cs & _ & _ & _ & _ & _ & _ & [Hv _] & _)
           | (ac & ac' & ksh & _ & _ & _ & _ & _ & [Hv _] & _)]];
    exact Hv.
Qed.

(* At a [Let] of the first run, related states have related
   environments. *)
Lemma uc_let (M : list (node * U.unboxed_fields)) (dead : list variable)
    (CS : code_id -> option UCS.code_shape) (NR : code_id -> bool)
    (RCH : bool -> variable -> code_id -> Prop) (OV REM : code_id -> Prop)
    (t t' : state) (y : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  UCS.related2 M dead CS NR RCH OV REM t t' ->
  state_expr t = Let y kind d body ->
  UCS.env_rel2 M dead (state_vars t) (state_vars t').
Proof.
  intros [(K & _ & Henv & _) | [(ap & ap' & cs & He & _) | (ac & ac' & ksh
                                                            & He & _)]]
    Hlet; [ exact Henv | | ]; rewrite Hlet in He; discriminate He.
Qed.

(* The values of the variables of the fields of an unboxed variable. *)
Lemma fields_hold_values (vars' : env) (fields : U.unboxed_fields)
    (w : value) :
  URP.fields_hold vars' fields w ->
  ∃ ws : list value,
    Forall2 (fun (xk : variable * flambda_kind) (w' : value) =>
               lookup (fst xk) vars' = Some w')
      (U.fields_leaves fields) ws
    ∧ unboxed_values fields w ws.
Proof.
  intros Hhold. unfold unboxed_values.
  induction (UCS.fields_leaves_paths fields)
    as [ | [x k] [p k'] leaves paths [Hk Hp] _ IH].
  - exists []. split; constructor.
  - destruct IH as (ws & Hl & Hv).
    destruct (Hhold p x k Hp) as (w' & Hpw & Hx & Hkw).
    exists (w' :: ws). split; constructor; try assumption.
    split; [ exact Hpw | ]. cbn [snd] in Hk |- *. rewrite <- Hk. exact Hkw.
Qed.

(* The states of the three runs. Where the run of the unit, which does
   not go wrong, is at a [Let] of a variable with a use, the run of the
   unit rebuilt with unboxing across calls comes to a state whose static
   values are related to the unit's by [statics_rebuilt], and in which
   each variable of the unit with a use that it does not unbox has the
   value that the rebuild with changes of calling convention has for it,
   related by [rebuilt_value], and each that it unboxes has the variables
   of its fields at the paths of the fields in that value. *)
Lemma calls_states (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (n : nat) (s : state) :
  used_premises local flag is_local u e T sets ->
  calls_premises flag u e T cannot_change dominated to_unbox nr ->
  ~ behaves (initial_state e) Goes_wrong ->
  run n (initial_state e) = Step s ->
  (∃ (y : variable) (kind : flambda_kind) (d : named) (body : expr),
     state_expr s = Let y kind d body
     ∧ s_has_usage (solution_of_tables T) (Node_variable y) = true) ->
  ∃ (m : nat) (t : state),
    run m
      (initial_state
         (calls_rebuilt flag u e T cannot_change dominated to_unbox b))
    = Step t
    ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) (state_statics s)
        (values_only (state_statics t))
    ∧ statics_allocated (solution_of_tables T) (state_statics s)
    ∧ static_sets_whole (solution_of_tables T) (state_statics s)
        (values_only (state_statics t))
    ∧ ∀ (x : variable) (v : value),
        lookup x (state_vars s) = Some v ->
        s_has_usage (solution_of_tables T) (Node_variable x) = true ->
        ∃ v_mid : value,
          rebuilt_value (graph_of_facts (traverse_unit flag u e))
            (solution_of_tables T) v (Node_variable x) v_mid
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
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  assert (Hs : step s <> Undefined).
  { intros Hu. apply Hng. exists (S n).
    replace (S n) with (n + 1) by lia.
    rewrite run_add, Hrun, run_succ, Hu. reflexivity. }
  destruct (cc_states_used flag is_local u e T _ ccw (unit_code_deps flag u e) b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
              Hvals Hcl Hw Hc (fun _ _ => eq_refl) n s Hrun Hs)
    as (m & t & Hm & Hst & Halloc & Hvars & Hlets & _ & Hwhole).
  assert (Ht : step t <> Undefined).
  { intros Hu. apply Hng.
    destruct (FlambdaReaperCallingConventionsProofs
              .rebuild_cc_refines_observably_local flag is_local u e T
                cannot_change b Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs
                Hcheck_exact Hccc Hwid)
      as (_ & _ & Hgw).
    apply Hgw. exists (S m).
    replace (S m) with (m + 1) by lia.
    unfold rebuild_unit_cc_from_tables, rebuild_unit_cc.
    unfold initial_state, rebuild_unit_cc_with in Hm.
    rewrite run_add, Hm, run_succ, Hu. reflexivity. }
  destruct (CX.uc_of_calls flag is_local u e T cannot_change dominated
              to_unbox b nr Hlocal Hfrag Hcheck Hcalls Hreads)
    as (dead & CS & NR & RCH & OV & REM & Hdead & Hdist & Huc & Hfl & Hov
        & Hrm).
  destruct (CX.uc_reach _ Hdist dead CS NR RCH OV REM _ _ Huc Hfl Hov Hrm m
              t Hm Ht)
    as (m3 & t3 & Hm3 & Hrel).
  destruct Hlet as (y & kind & d & body & Hse & Hhu).
  destruct (Hlets y kind d body Hse Hhu) as [Hinv | (d' & body' & Ht_let)].
  { exfalso. apply Ht. rewrite (state_eta t), Hinv. apply step_invalid. }
  pose proof (uc_let _ _ _ _ _ _ _ _ _ _ _ _ _ Hrel Ht_let) as [Hplain Hunb].
  pose proof (uc_values _ _ _ _ _ _ _ _ _ Hrel) as Hv.
  exists m3, t3. split; [ exact Hm3 | ].
  split; [ apply (statics_rebuilt_values _ _ _ (state_statics t));
           [ exact Hv | exact Hst ] | ].
  split; [ exact Halloc | ].
  split;
    [ exact (static_sets_whole_values _ _ _ (values_only (state_statics t3))
               Hv Hwhole) | ].
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
   through the same continuation with the same heap, from a state whose
   static values are related to the unit's by [statics_rebuilt]. *)
Lemma calls_exit (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (b : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  used_premises local flag is_local u e T sets ->
  calls_premises flag u e T cannot_change dominated to_unbox nr ->
  run n (initial_state e) = Step s ->
  step s = Exit k vs hp rd ->
  ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
    run m
      (initial_state
         (calls_rebuilt flag u e T cannot_change dominated to_unbox b))
    = Step t
    ∧ step t = Exit k vs' hp rd'
    ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) (state_statics s)
        (values_only (state_statics t))
    ∧ statics_allocated (solution_of_tables T) (state_statics s)
    ∧ static_sets_whole (solution_of_tables T) (state_statics s)
        (values_only (state_statics t)).
Proof.
  intros Hprem (Hccc & Hwid & Hcalls & Hreads) Hrun Hexit.
  pose proof Hprem as (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits
                       & Honce & Hvals & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  destruct (cc_states_used flag is_local u e T _ ccw (unit_code_deps flag u e) b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
              Hvals Hcl Hw Hc (fun _ _ => eq_refl) n s Hrun
              ltac:(rewrite Hexit; discriminate))
    as (m & t & Hm & Hst & Halloc & _ & _ & Hex & Hwhole).
  destruct (Hex k vs hp rd Hexit) as (vs' & rd' & Ht).
  destruct (CX.uc_of_calls flag is_local u e T cannot_change dominated
              to_unbox b nr Hlocal Hfrag Hcheck Hcalls Hreads)
    as (dead & CS & NR & RCH & OV & REM & Hdead & Hdist & Huc & Hfl & Hov
        & Hrm).
  destruct (CX.uc_exit _ Hdist dead CS NR RCH OV REM _ _ Huc Hfl Hov Hrm m
              t k vs' hp rd' Hm Ht)
    as (m3 & t3 & rd3 & Hm3 & Ht3 & Hrel).
  exists m3, t3, vs', rd3.
  split; [ exact Hm3 | split; [ exact Ht3 | split; [ | split ] ] ].
  - apply (statics_rebuilt_values _ _ _ (state_statics t));
      [ exact (uc_values _ _ _ _ _ _ _ _ _ Hrel) | exact Hst ].
  - exact Halloc.
  - exact (static_sets_whole_values _ _ _ (values_only (state_statics t3))
             (uc_values _ _ _ _ _ _ _ _ _ Hrel) Hwhole).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Lemma unbox_calls_types_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_calls_types_end_to_end_used_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hcalls Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (calls_exit local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s k vs hp rd Hprem Hcalls Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Ht & Hst & Halloc & _).
  exists m, t, vs', rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  apply (type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_symbol_type_sound_in fx false Hfx is_local _ T sets tenv
           canon (state_statics s) (values_only (state_statics t)) valuation
           valuation' sym allocated);
    try assumption.
  intros H. discriminate H.
Qed.

Lemma unbox_calls_types_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_calls_types_end_to_end_with local fx.
Proof. intros Hfx. used_instance (unbox_calls_types_used_gen local fx Hfx). Qed.

Theorem unbox_calls_types_end_to_end :
  unbox_calls_types_end_to_end_statement.
Proof. exact (unbox_calls_types_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem unbox_calls_types_end_to_end_landed :
  unbox_calls_types_end_to_end_landed_statement.
Proof.
  exact (unbox_calls_types_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_types_end_to_end_local :
  unbox_calls_types_end_to_end_local_statement.
Proof. exact (unbox_calls_types_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem unbox_calls_types_end_to_end_local_landed :
  unbox_calls_types_end_to_end_local_landed_statement.
Proof.
  exact (unbox_calls_types_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma unbox_calls_result_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_calls_result_type_end_to_end_used_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox b nr sets
    tenv canon valuation valuation' n s code_var type_var v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hnone.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (calls_states local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (m & t & Hm & Hst & Halloc & _ & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & Hplain & _).
  exists m, t, v_mid.
  split; [ exact Hm | split; [ exact (Hplain Hin Hnone) | ] ].
  apply (type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_result_type_sound_in fx false Hfx is_local _ T sets tenv
           canon (state_statics s) (values_only (state_statics t)) valuation
           valuation' code_var type_var v v_mid);
    try assumption.
  intros H. discriminate H.
Qed.

Lemma unbox_calls_result_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> unbox_calls_result_type_end_to_end_with local fx.
Proof.
  intros Hfx. used_instance (unbox_calls_result_used_gen local fx Hfx).
Qed.

Theorem unbox_calls_result_type_end_to_end :
  unbox_calls_result_type_end_to_end_statement.
Proof. exact (unbox_calls_result_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem unbox_calls_result_type_end_to_end_landed :
  unbox_calls_result_type_end_to_end_landed_statement.
Proof.
  exact (unbox_calls_result_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem unbox_calls_result_type_end_to_end_local :
  unbox_calls_result_type_end_to_end_local_statement.
Proof. exact (unbox_calls_result_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem unbox_calls_result_end_to_end_local_landed :
  unbox_calls_result_end_to_end_local_landed_statement.
Proof.
  exact (unbox_calls_result_gen true landed_fixes TP.landed_fixes_sound).
Qed.

Lemma calls_unboxed_result_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unboxed_result_type_end_to_end_used_with local fx.
Proof.
  intros (Hb & Hal & Hr & Hn) flag is_local u e T cannot_change dominated
    to_unbox b nr sets tenv canon valuation valuation' n s code_var type_var
    fields v Hprem Hcalls
    (Hrun & Hs & Hx & Hhu & Henv & Hfr & Hcopies & Hfind & Hl)
    (Hng & Hlet & Hin) Hfields.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  pose proof Hcalls as (_ & _ & Hcalls_nr & _).
  unfold FlambdaReaperUnboxingRefinesCallsSpec.calls_unboxing_nr in Hcalls_nr.
  apply andb_true_iff in Hcalls_nr as [_ Huses].
  destruct (calls_states local flag is_local u e T cannot_change dominated
              to_unbox b nr sets n s Hprem Hcalls Hng Hrun Hlet)
    as (m & t & Hm & Hst & Halloc & _ & Hvars).
  destruct (Hvars code_var v Hx Hhu) as (v_mid & Hrv & _ & Hunb).
  destruct (fields_hold_values _ _ _ (Hunb fields Hfields))
    as (ws & Hws & Hvals).
  exists m, t, ws. split; [ exact Hm | split; [ exact Hws | ] ].
  assert (Hsat :
    Forall2 (type_satisfied (values_only (state_statics t)) valuation') ws
      (rewrite_unboxed_result_type fx
         (plain_context is_local (traverse_unit flag u e) T sets) tenv
         (U.bound_code_my_closures e) canon code_var type_var fields)).
  { eapply (TP.rewrite_unboxed_result_type_sound_in fx Hb Hal Hr Hn is_local
              _ T sets tenv canon (state_statics s)
              (values_only (state_statics t)) valuation valuation' false);
      try eassumption.
    - exact (FlambdaReaperTypesTraversalProofs
             .block_fields_of_one_kind_traversal flag u e).
    - intros H. discriminate H. }
  clear - Hsat. induction Hsat as [ | w ty ws tys Hw _ IH]; constructor;
    [ | exact IH ].
  exact (type_satisfied_values_only (state_statics t) (state_statics t) _ _ _
           eq_refl Hw).
Qed.

Lemma calls_unboxed_result_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> calls_unboxed_result_type_end_to_end_with local fx.
Proof.
  intros Hfx. used_instance (calls_unboxed_result_used_gen local fx Hfx).
Qed.

Theorem calls_unboxed_result_type_end_to_end :
  calls_unboxed_result_type_end_to_end_statement.
Proof.
  exact (calls_unboxed_result_gen false all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_landed :
  calls_unboxed_result_type_end_to_end_landed_statement.
Proof.
  exact (calls_unboxed_result_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem calls_unboxed_result_type_end_to_end_local :
  calls_unboxed_result_type_end_to_end_local_statement.
Proof.
  exact (calls_unboxed_result_gen true all_fixes TP.all_fixes_sound).
Qed.

Theorem calls_unboxed_result_end_to_end_local_landed :
  calls_unboxed_result_end_to_end_local_landed_statement.
Proof.
  exact (calls_unboxed_result_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* With changes of representation                                           *)
(* ======================================================================== *)

Module RX := FlambdaReaperTypesReprExitProofs.

(* Where the unit exits, the unit rebuilt with unboxing and changes of
   representation exits through the same continuation with the same
   heap, from a state whose static values are related to the unit's by
   [statics_rebuilt]. *)
Lemma repr_exit (local : bool) (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (b : bool) (nr : code_id -> bool)
    (sets : list set_of_closures) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  used_premises local flag is_local u e T sets ->
  repr_premises flag is_local u e T cannot_change dominated to_unbox tcr nr ->
  run n (initial_state e) = Step s ->
  step s = Exit k vs hp rd ->
  ∃ (m : nat) (t : state) (vs' : list value) (rd' : read_record),
    run m
      (initial_state
         (repr_rebuilt flag is_local u e T cannot_change dominated to_unbox
            tcr b))
    = Step t
    ∧ step t = Exit k vs' hp rd'
    ∧ statics_rebuilt (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) (state_statics s)
        (values_only (state_statics t))
    ∧ statics_allocated (solution_of_tables T) (state_statics s)
    ∧ static_sets_whole (solution_of_tables T) (state_statics s)
        (values_only (state_statics t)).
Proof.
  intros Hprem (Hccc & Hwid & Hrepr & Hreads) Hrun Hexit.
  pose proof Hprem as (Hlocal & Hleast & Hfrag & Hdc & Hrk & Hwf & Hexits
                       & Honce & Hvals & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & Hcheck & Hlrs & Hcheck_exact).
  destruct (cc_checks flag u e T cannot_change Hccc Hwid)
    as ((ccw & Hcl) & Hw & Hc).
  destruct (cc_states_used flag is_local u e T _ ccw (unit_code_deps flag u e) b
              Hlocal Hfrag Hdc Hrk Hwf Hexits Hcheck Hlrs Hcheck_exact Honce
              Hvals Hcl Hw Hc (fun _ _ => eq_refl) n s Hrun
              ltac:(rewrite Hexit; discriminate))
    as (m & t & Hm & Hst & Halloc & _ & _ & Hex & Hwhole).
  destruct (Hex k vs hp rd Hexit) as (vs' & rd' & Ht).
  destruct (RX.repr_calls_rr None flag is_local u e T cannot_change
              dominated to_unbox tcr b nr Hlocal Hfrag Hcheck Hrepr Hreads)
    as (M & CH & CS & RCH & Hdist & HCM & HCf & Hrr & Hfl).
  destruct (RX.rr_exit M Hdist CH HCM HCf CS RCH _ _ Hrr Hfl m t k vs' hp
              rd' Hm Ht)
    as (m3 & t3 & rd3 & Hm3 & Ht3 & Hrel).
  exists m3, t3, vs', rd3.
  split; [ exact Hm3 | split; [ exact Ht3 | split; [ | split ] ] ].
  - apply (statics_rebuilt_values _ _ _ (state_statics t));
      [ exact (RX.rr_values _ _ _ _ _ _ Hrel) | exact Hst ].
  - exact Halloc.
  - exact (static_sets_whole_values _ _ _ (values_only (state_statics t3))
             (RX.rr_values _ _ _ _ _ _ Hrel) Hwhole).
Qed.

Lemma repr_calls_types_used_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> repr_calls_types_end_to_end_used_with local fx.
Proof.
  intros Hfx flag is_local u e T cannot_change dominated to_unbox tcr b nr
    sets tenv canon valuation valuation' n s k vs hp rd sym allocated Hprem
    Hrepr Hrun Hexit Henv Hfr Hcopies Hl.
  pose proof Hprem as (Hlocal & Hleast & _ & _ & _ & _ & _ & _ & _ & Hsets).
  destruct (least_with_checks local flag is_local u e T Hlocal Hleast)
    as (Hle & _ & _ & _).
  destruct (repr_exit local flag is_local u e T cannot_change dominated
              to_unbox tcr b nr sets n s k vs hp rd Hprem Hrepr Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Ht & Hst & Halloc & _).
  exists m, t, vs', rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  apply (type_satisfied_values_only (state_statics t)); [ reflexivity | ].
  apply (TP.rewrite_symbol_type_sound_in fx false Hfx is_local _ T sets tenv
           canon (state_statics s) (values_only (state_statics t)) valuation
           valuation' sym allocated);
    try assumption.
  intros H. discriminate H.
Qed.

Lemma repr_calls_types_gen (local : bool) (fx : fixes) :
  TP.fixes_sound fx -> repr_calls_types_end_to_end_with local fx.
Proof. intros Hfx. used_instance (repr_calls_types_used_gen local fx Hfx). Qed.

Theorem repr_calls_types_end_to_end : repr_calls_types_end_to_end_statement.
Proof. exact (repr_calls_types_gen false all_fixes TP.all_fixes_sound). Qed.

Theorem repr_calls_types_end_to_end_landed :
  repr_calls_types_end_to_end_landed_statement.
Proof.
  exact (repr_calls_types_gen false landed_fixes TP.landed_fixes_sound).
Qed.

Theorem repr_calls_types_end_to_end_local :
  repr_calls_types_end_to_end_local_statement.
Proof. exact (repr_calls_types_gen true all_fixes TP.all_fixes_sound). Qed.

Theorem repr_calls_types_end_to_end_local_landed :
  repr_calls_types_end_to_end_local_landed_statement.
Proof.
  exact (repr_calls_types_gen true landed_fixes TP.landed_fixes_sound).
Qed.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Module UnboxCallsRuns.

Import UnboxCallsExample.
Local Open Scope positive_scope.

Ltac in_cases H :=
  repeat match type of H with
         | _ ∨ _ => destruct H as [H | H]
         end;
  try discriminate H; try contradiction H.

Lemma constructors :
  f_constructor facts
  = [(Node_variable 60, Field_call_witness Known_arity_code_pointer,
      Node_made 7);
     (Node_variable 60, Field_call_witness Unknown_arity_code_pointer,
      Node_made 8);
     (Node_variable 10, Field_block 0 Value, Node_all_constants);
     (Node_variable 10, Field_block 1 Value, Node_all_constants);
     (Node_variable 10, Field_is_int, Node_all_constants);
     (Node_variable 10, Field_get_tag, Node_all_constants);
     (Node_variable 60, Field_function_slot 70, Node_variable 60);
     (Node_variable 42, Field_block 0 Value, Node_variable 43);
     (Node_variable 42, Field_block 1 Value, Node_all_constants);
     (Node_variable 42, Field_is_int, Node_all_constants);
     (Node_variable 42, Field_get_tag, Node_all_constants);
     (Node_made 8, Field_return_of_call Exn, Node_made 4);
     (Node_made 8, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 8, Field_code_id_of_call_witness, Node_code_id 50);
     (Node_made 7, Field_return_of_call (Normal 0), Node_made 3);
     (Node_made 7, Field_return_of_call Exn, Node_made 4);
     (Node_made 7, Field_code_id_of_call_witness, Node_code_id 50)].
Proof. vm_compute. reflexivity. Qed.

Lemma cons_in (F : FlambdaReaperGraphSpec.facts) (a : node) (f : field)
    (b : node) :
  g_constructor (graph_of_facts F) a f b -> (a, f, b) ∈ f_constructor F.
Proof. exact (fun H => H). Qed.

Lemma sets_ok : sets_of_graph (graph_of_facts facts) sets.
Proof.
  intros fdecls sites slot id Hl Hsites.
  exists (hd [] sets).
  assert (Hb0 : is_bound slot fdecls = true)
    by (unfold is_bound; rewrite Hl; reflexivity).
  pose proof (proj2 (Hsites slot id Hl slot (sites slot)) (conj Hb0 eq_refl))
    as Hself.
  apply cons_in in Hself. rewrite constructors in Hself.
  cbn [In] in Hself. in_cases Hself.
  injection Hself as Hs70 E70. subst slot.
  assert (Hdom : ∀ g : function_slot,
             is_bound g fdecls = true -> g = 70%positive).
  { intros g Hg. pose proof (proj2 (Hsites 70%positive id Hl g (sites g))
                               (conj Hg eq_refl)) as Hc.
    apply cons_in in Hc. rewrite constructors, <- Hs70 in Hc.
    cbn [In] in Hc. in_cases Hc.
    injection Hc as <- _. reflexivity. }
  split; [ left; reflexivity | split ].
  - intros g c code Hin. destruct Hin as [Hin | []].
    injection Hin as <- <- _. split; [ exact Hb0 | congruence ].
  - intros g Hg. rewrite (Hdom g Hg). reflexivity.
Qed.

(* The state of the run of the unit at the [Let] of [x43] in the code. *)
Definition state_at : state :=
  match run 5 (initial_state unit_expr) with
  | Step s => s
  | _ => initial_state unit_expr
  end.

Lemma run_at : run 5 (initial_state unit_expr) = Step state_at.
Proof. vm_compute. reflexivity. Qed.

Lemma not_wrong : ~ behaves (initial_state unit_expr) Goes_wrong.
Proof.
  intros Hw.
  assert (Hex : behaves (initial_state unit_expr)
                  (Exits 1 [Value_const (Tagged_immediate 8)] empty_heap)).
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

End UnboxCallsRuns.

Lemma unbox_calls_example_gen (fx : fixes) :
  rewrite_unboxed_result_type fx
    (plain_context FlambdaReaperUnboxing.Examples.no_local
       UnboxCallsExample.facts UnboxCallsExample.tables
       UnboxCallsExample.sets)
    UnboxCallsExample.tenv
    (FlambdaReaperUnboxing.bound_code_my_closures UnboxCallsExample.unit_expr)
    UnboxCallsExample.canon 40%positive 90%positive UnboxCallsExample.fields
  = [this_tagged_immediate 8%Z] ->
  unbox_calls_example_with fx.
Proof.
  intros Hrw.
  destruct FlambdaReaperUnboxingRefinesProofs.unboxed_call_example
    as (Hslots & Hfrag & Hstatic & Hrk & Hwf & Hone & Hclosed & Hexact & _
        & Hccc & Hwid & _).
  split.
  { split; [ exact Hslots | split ].
    { split; [ apply (FlambdaReaperLeastProofs.check_least_sound _ _ 100);
               vm_compute; reflexivity | ].
      left. intros f. reflexivity. }
    split; [ exact Hfrag | split ].
    { exact (RP.static_constants_any_kind_arguments _ _ Hstatic). }
    split; [ exact Hrk | split; [ exact Hwf | split; [ exact Hone | ] ] ].
    split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
    exact UnboxCallsRuns.sets_ok. }
  split.
  { split; [ exact Hccc | split; [ exact Hwid | ] ].
    split; vm_compute; reflexivity. }
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { exists UnboxCallsRuns.state_at.
    split.
    - split; [ exact UnboxCallsRuns.run_at | ].
      split; [ vm_compute; discriminate | ].
      split; [ vm_compute; reflexivity | ].
      split; [ vm_compute; reflexivity | ].
      split; [ apply UnboxCallsRuns.tenv_satisfied | ].
      split.
      { intros n ty bt Hl.
        destruct (UnboxCallsRuns.lookup_tenv n ty bt Hl) as [_ [-> | ->]];
          [ | apply TP.unknown_fragment ].
        split; [ vm_compute; reflexivity | ].
        intros s k w Hin Hw. vm_compute in Hin.
        UnboxCallsRuns.in_cases Hin. injection Hin as <- <-.
        cbn in Hw. injection Hw as <-. left. eexists. reflexivity. }
      split.
      { intros s x k m Hcan. unfold value_of_name.
        exact (FlambdaTypingEnvProofs.get_canonical_simple_satisfied
                 _ _ _ _ _ _ _ _ (UnboxCallsRuns.tenv_satisfied
                                    (state_statics UnboxCallsRuns.state_at))
                 Hcan). }
      split; [ vm_compute; discriminate | reflexivity ].
    - split; [ exact UnboxCallsRuns.not_wrong | split ].
      + exists 43%positive, Value,
          (FlambdaReaperUnboxing.Examples.load 1 (Var 40%positive)),
          (Let 42%positive Value
             (FlambdaReaperUnboxing.Examples.make_block
                [Var 43%positive; FlambdaReaperUnboxing.Examples.int 9%Z])
             (FlambdaReaperUnboxing.Examples.return_to 3%positive
                [Var 42%positive])).
        split; vm_compute; reflexivity.
      + vm_compute. tauto. }
  split; [ exact Hrw | ].
  exists
    (match run 5 (initial_state UnboxCallsExample.rebuilt) with
     | Step t => t
     | _ => initial_state UnboxCallsExample.rebuilt
     end).
  split; [ vm_compute; reflexivity | split; [ vm_compute; reflexivity | ] ].
  apply FlambdaTypeMeaningBoolProofs.type_satisfied_b_iff.
  vm_compute. reflexivity.
Qed.

Theorem unbox_calls_example : unbox_calls_example_statement.
Proof. apply unbox_calls_example_gen. vm_compute. reflexivity. Qed.

Theorem unbox_calls_example_landed : unbox_calls_example_landed_statement.
Proof. apply unbox_calls_example_gen. vm_compute. reflexivity. Qed.
