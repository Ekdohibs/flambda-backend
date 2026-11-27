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
     heap ([uc_exit]). [rebuild_unbox_fixed_uc] is the proof of
     FlambdaReaperUnboxingRefinesCallsProofs
     .rebuild_unbox_modes_with_refines_cc_never up to its last step,
     copied (about 200 lines) so as not to edit that file: the two rebuilt
     terms are related by [uc]; [uc_of_calls], for the fragment of
     FlambdaReaperUnboxingRefinesCallsSpec.v, with the invariants of the
     first run that the simulation takes. *)

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

(* The proof of
   FlambdaReaperUnboxingRefinesCallsProofs
   .rebuild_unbox_modes_with_refines_cc_never up to its last step: the two
   rebuilt terms are related by [uc], for some parameters that the second
   run may give anything ([dead], none with a use). *)
Lemma rebuild_unbox_fixed_uc (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node) (pdc : bool)
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) :
  (∀ t : subkinds_types, types = Some t ->
   ∀ (n : node) (ps : param_subkind) (S : static_env) (v : value),
     s_has_usage (solution_of_tables T) n = false ->
     value_at S (st_rewrite t n ps) v = true) ->
  conts_table_ok types e ->
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  FlambdaReaperUnboxingRefinesNeverSpec.calls_unboxing_reach_never_nr nr
    (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
    dominated to_unbox reach ovr rem e
  = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ (dead : list variable) (CS : code_id -> option code_shape)
    (NR : code_id -> bool),
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
        dead CS NR (fun b f id => reach b f id = true)
        (fun id => ovr id = true) (fun id => rem id = true) []
        (rebuild_unit_cc_with types (traverse_unit flag u e)
           (solution_of_tables T) (fun n => mem_node n cannot_change)
           (unit_code_deps flag u e) pdc e)
        (rebuild_unit_unbox_repr_mode_with types true (fun _ => false)
           (traverse_unit flag u e) T cannot_change (unit_code_deps flag u e)
           dominated to_unbox [] pdc e).
Proof.
  intros Hunused Htable Hslots Hfragment Hclosed Hcalls Hcheck.
  assert (Htabled : ∀ t : subkinds_types, types = Some t ->
            ∀ (k : continuation) (params : list (node * param_subkind)),
              (k, params) ∈ handlers_params e ->
              lookup k (st_conts t) = Some params).
  { intros t Ht k params Hin. destruct (Htable t Ht) as [Hc Hd].
    rewrite Hc.
    exact (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _
             (conts_table_nodup e Hd) Hin). }
  set (F := traverse_unit flag u e) in *.
  unfold calls_unboxing_reach_never_nr in Hcalls.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  pose proof (unboxed_fields_of_names_fresh F T (bound_code_my_closures e)
                dominated to_unbox
                (FlambdaNameAbstractionSpec.fresh_name (names_of e)))
    as Hnext.
  cbv zeta in Hnext. destruct Hnext as (Hnext & _ & _).
  unfold rebuild_unit_unbox_repr_mode_with, rebuild_unit_cc_with.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hcalls, Huses, Hdistinct, Hfresh.
  assert (Hnext' : (FlambdaNameAbstractionSpec.fresh_name (names_of e)
                    <= next)%positive).
  { unfold unit_unboxed_fields in Hunboxed. rewrite Hunboxed in Hnext.
    exact Hnext. }
  clear Hnext.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts flag u e True
              Hfragment (or_introl I))
    as (noted & deps & cds & Hedges & _).
  assert (Hfields_read :
    ∀ (x : variable) (fields : unboxed_fields) (f : field)
      (v : unboxed_field),
      fields_of unboxed x = Some fields -> (f, v) ∈ fields ->
      ∃ w base : node,
        g_accessor (graph_of_facts F) w f base
        ∧ s_usages (solution_of_tables T) (Node_variable x) base = true
        ∧ s_has_usage (solution_of_tables T) w = true).
  { intros x fields f v Hx Hin. apply (fields_of_in unboxed) in Hx.
    pose proof Hx as Hxu.
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_unboxed F T _ unboxed _ _ true Huses Hxu) in Hbase.
    apply mem_node_In in Hbase.
    destruct (get_direct_usages_in T _ _ Hbase) as [Hus _].
    exists w, base. split; [ exact Hacc | split; [ exact Hus | ] ].
    destruct Hw as [Hw | Hw]; [ | exact Hw ].
    exact (has_usage_any _ _ _ Hcl _ Hw). }
  assert (Hreads_found :
    ∀ (x y : variable) (kind : flambda_kind) (f : field)
      (fields : unboxed_fields),
      fields_of unboxed x = None -> fields_of unboxed y = Some fields ->
      g_accessor (graph_of_facts F) (Node_variable x) f (Node_variable y) ->
      is_block_like_field f = true ->
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hblock Hdead Hused Hkind.
    assert (Hsx : mem_node (Node_variable x) (t_has_source T) = true).
    { unfold is_dead_var in Hdead.
      destruct Hkind as [-> | (nk & ->)]; simpl in Hdead;
        apply negb_false_iff in Hdead; exact Hdead. }
    assert (Hux : mem_node (Node_variable x) (t_has_usage T) = true).
    { unfold is_var_used in Hused.
      destruct Hkind as [-> | (nk & ->)]; exact Hused. }
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hhx : has_to_be_unboxed dominated to_unbox (Node_variable x)
                  = false).
    { destruct (has_to_be_unboxed dominated to_unbox (Node_variable x))
        eqn:Hh; [ | reflexivity ].
      unfold has_to_be_unboxed in Hh.
      apply existsb_exists in Hh as ([a x'] & Hin & Htest).
      apply andb_true_iff in Htest as [Hx' Ha].
      apply node_eqb_eq in Hx'. subst x'.
      pose proof (unboxed_fields_of_names_bound F T
                    (bound_code_my_closures e) dominated to_unbox
                    (FlambdaNameAbstractionSpec.fresh_name (names_of e)) a
                    (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (FlambdaNameAbstractionSpec.fresh_name
                              (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & Hfields).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses' : mem_node (Node_variable y)
                       (add_usages_through_function_slots F T
                          (bound_code_my_closures e) true
                          (get_direct_usages T [Node_variable y]))
                     = true).
    { rewrite (add_usages_unboxed F T _ unboxed _ _ true Huses Hy).
      apply mem_node_In. apply get_direct_usages_self; assumption. }
    subst fields.
    destruct (get_fields_found F T _ _ _ _ Hacc Huses' Hfs Hux)
      as [Hex Hall].
    apply (mk_unboxed_fields_lookup F T _ dominated to_unbox _ a f Hdata).
    - intros Hno. rewrite Hno in Hsrc. discriminate Hsrc.
    - exact Hex.
    - intros use Hin.
      destruct (Hall use Hin) as [-> | (flow_to & -> & Hw)];
        [ left; reflexivity | right ].
      exists flow_to. split; [ reflexivity | ].
      destruct (forallb _ flow_to) eqn:Hft; [ | reflexivity ].
      rewrite forallb_forall in Hft. rewrite (Hft _ Hw) in Hhx.
      discriminate Hhx. }
  set (dead := filter (fun z =>
                         negb (s_has_usage (solution_of_tables T)
                                 (Node_variable z))
                         && is_none (fields_of unboxed z))
                 (variables_of e)).
  assert (Hdead : ∀ z, among z dead = true ->
            s_has_usage (solution_of_tables T) (Node_variable z) = false).
  { intros z Hz. apply among_In in Hz. unfold dead in Hz.
    apply filter_In in Hz as [_ Hz]. apply andb_true_iff in Hz as [Hz _].
    apply negb_true_iff in Hz. exact Hz. }
  destruct (proj1 (rebuild_uc F (solution_of_tables T)
                     (fun n => mem_node n cannot_change)
                     (unit_code_deps flag u e) pdc
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     unboxed
                     (FlambdaNameAbstractionSpec.fresh_name (names_of e))
                     next dead noted deps cds (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local nr reach
                     ovr rem types Hcl Hslots
                     Hfields_read Hreads_found Hdead Hunused)
              e empty_env empty_unbox_env [] None (unit_conts u))
    as (Huc & _).
  - split.
    + intros k ps h Hl. discriminate Hl.
    + reflexivity.
    + intros x. reflexivity.
    + reflexivity.
    + reflexivity.
    + reflexivity.
    + intros k. reflexivity.
    + intros k. left. reflexivity.
    + intros k ps h Hl. discriminate Hl.
  - exact Hcalls.
  - split; [ | split ].
    + intros z Hz. split; [ exact (Hfresh z Hz) | ].
      intros Hu Hn. apply among_In. unfold dead. apply filter_In.
      split; [ exact Hz | ]. rewrite Hu, Hn. reflexivity.
    + intros k Hk.
      assert (Hle : (k <= fold_right Pos.max 1 (names_of e))%positive)
        by (apply fold_max_le; unfold names_of; apply in_or_app; right;
            exact Hk).
      unfold FlambdaNameAbstractionSpec.fresh_name in Hnext' |- *.
      split; intros ->; lia.
    + exact Htabled.
  - exact Hedges.
  - exists dead. do 2 eexists.
    split; [ exact Hdead | split; [ exact Hdistinct | exact Huc ] ].
Qed.

(* The two rebuilt units of the fragment of
   FlambdaReaperUnboxingRefinesCallsSpec.v, related by [uc], with the
   invariants of the first run that the simulation takes. *)
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
Proof.
  intros Hslots Hfrag Hclosed Hcalls Hcheck.
  destruct (rebuild_unbox_fixed_uc None flag is_local u e T cannot_change
              dominated to_unbox pdc nr (reach2 (fun _ _ => true))
              (fun _ => true) (fun _ => true)
              ltac:(intros t Ht; discriminate Ht) (no_types_conts e) Hslots
              Hfrag Hclosed)
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
