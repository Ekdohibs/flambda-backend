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

(* The exit of the simulation of the local unboxing with the primitives
   of the heap (FlambdaReaperUnboxingRefinesHeapProofs.relatedh): where
   the run of the unit rebuilt without unboxing exits, the run of the
   unit rebuilt with it exits through the same continuation, with the
   same values and the same heap, from a state with the same static
   environment. For FlambdaReaperTypesTraversalHeapProofs.v. This file
   needs no review.

   [rebuild_unbox_ubh_heap] is the proof of
   [rebuild_unbox_with_refines_rebuild_heap]
   (FlambdaReaperUnboxingRefinesHeapProofs.v) up to its last step: the
   two rebuilt terms are related by [ubh], whose simulation relates the
   states of the two runs ([relatedh]) with one static environment.
   [shrinking_exit_state]: in a simulation of
   FlambdaReaperRebuildProofs ([answered]), the state from which the
   first run exits is related to one from which the second exits. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaBindersSpec FlambdaWellFormedSpec
  FlambdaNameAbstractionSpec FlambdaRenameApartSpec FlambdaMapSpec
  FlambdaMapProofs FlambdaValueSpec FlambdaHeapSpec
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaUnusedStaticsSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperSolutionProofs FlambdaReaperRebuild
  FlambdaReaperFlowSpec FlambdaReaperRebuildSpec FlambdaReaperRebuildProofs
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesProofs
  FlambdaReaperUnboxingRefinesHeapSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsProofs FlambdaFreshQuantifierSpec
  FlambdaPrimitiveProofs FlambdaNameAbstractionSpec FlambdaRenameApartSpec
  FlambdaHeapProofs FlambdaReaperUnboxingRefinesHeapProofs.
Import ListNotations.

Module URP := FlambdaReaperUnboxingRefinesProofs.
Module URH := FlambdaReaperUnboxingRefinesHeapProofs.

(* ======================================================================== *)
(* The state of an exit, in a simulation                                    *)
(* ======================================================================== *)

Section Exit_state.

Variable R : state -> state -> Prop.
Variable E : list value -> list value -> Prop.
Variable measure : state -> nat.
Hypothesis R_answered : ∀ s t : state, R s t -> answered R E measure s t.

(* A state that the first run comes to, and from which it exits, is
   related to one that the second run comes to. *)
Lemma shrinking_reaches_exit (n : nat) :
  ∀ (s t s1 : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    R s t -> run n s = Step s1 -> step s1 = Exit k vs hp rd ->
    ∃ (m : nat) (t1 : state), run m t = Step t1 ∧ R s1 t1.
Proof.
  induction n as [ | n IH]; intros s t s1 k vs hp rd Hrel Hrun Hexit.
  - injection Hrun as <-. exists 0, t. split; [ reflexivity | exact Hrel ].
  - rewrite run_succ in Hrun. pose proof (R_answered s t Hrel) as Ha.
    unfold answered in Ha.
    destruct (step s) as [s' | k' vs' hp' rd' | ] eqn:Hs;
      [ | discriminate Hrun | discriminate Hrun ].
    destruct Ha as [Hund | [(t' & Ht & Hrel') | [(_ & Hrel')
                                                | (t' & n0 & Ht & Hrel')]]].
    + exfalso. destruct n as [ | n].
      * injection Hrun as <-. rewrite Hund in Hexit. discriminate Hexit.
      * rewrite run_succ, Hund in Hrun. discriminate Hrun.
    + destruct (IH s' t' s1 k vs hp rd Hrel' Hrun Hexit)
        as (m & t1 & Hm & Hr1).
      exists (S m), t1. rewrite run_succ, Ht. split; [ exact Hm | exact Hr1 ].
    + exact (IH s' t s1 k vs hp rd Hrel' Hrun Hexit).
    + destruct (IH s' t' s1 k vs hp rd Hrel' Hrun Hexit)
        as (m & t1 & Hm & Hr1).
      exists (S (S n0) + m), t1. rewrite run_add, Ht.
      split; [ exact Hm | exact Hr1 ].
Qed.

(* The state from which the first run exits is related to one from which
   the second exits, through the same continuation, with the same heap
   and values related by [E]. *)
Lemma shrinking_exit_state (n : nat) (s t s1 : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  R s t -> run n s = Step s1 -> step s1 = Exit k vs hp rd ->
  ∃ (m : nat) (t1 : state) (vs' : list value) (rd' : read_record),
    run m t = Step t1 ∧ R s1 t1 ∧ step t1 = Exit k vs' hp rd' ∧ E vs vs'.
Proof.
  intros Hrel Hrun Hexit.
  destruct (shrinking_reaches_exit n s t s1 k vs hp rd Hrel Hrun Hexit)
    as (m & t1 & Hm & Hr1).
  pose proof (R_answered s1 t1 Hr1) as Ha. unfold answered in Ha.
  rewrite Hexit in Ha. destruct Ha as (vs' & rd' & Ht1 & HE).
  exists m, t1, vs', rd'.
  split; [ exact Hm | split; [ exact Hr1 | split; [ exact Ht1 | exact HE ] ] ].
Qed.

End Exit_state.

(* ======================================================================== *)
(* The two rebuilds of a unit of the fragment, related                      *)
(* ======================================================================== *)

Lemma rebuild_unbox_ubh_heap (types : subkinds_rewriting)
    (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  ∃ M : list (node * unboxed_fields),
    distinct (URP.field_variables M) = true
    ∧ URH.ubh M []
        (rebuild_unit_with types (traverse_unit flag u e)
           (solution_of_tables T) preserve_direct_calls e)
        (rebuild_unit_unbox_with types (traverse_unit flag u e) T
           cannot_change code_deps dominated to_unbox preserve_direct_calls
           e).
Proof.
  intros Hslots Hfragment Hclosed Hlocal Hcheck.
  set (F := traverse_unit flag u e) in *.
  unfold local_unboxing in Hlocal.
  pose proof (unit_unboxed_fields_fresh F T dominated to_unbox e)
    as [Hdistinct Hfresh].
  unfold rebuild_unit_unbox_with, rebuild_unit_unbox_repr_mode_with.
  destruct (unit_unboxed_fields F T dominated to_unbox e) as [unboxed next]
    eqn:Hunboxed.
  cbn [fst] in Hlocal, Hdistinct, Hfresh.
  rename Hlocal into Hloc.
  pose proof (closure_free_reads_blocks flag u e
                (proj1 (local_closure_free unboxed (solution_of_tables T))
                   e [] [] Hloc))
    as Hreads.
  apply (proj1 (strong_local unboxed (solution_of_tables T)) e [] [] Hloc)
    in Hfresh.
  rename Hfresh into Hloc'.
  assert (Hchanged :
    unit_changed_representations (fun _ => false) F T dominated to_unbox [] e
    = []) by reflexivity.
  rewrite Hchanged.
  pose proof (check_closed_sound is_local F T Hclosed) as Hcl.
  destruct (FlambdaReaperFlowProofs.traverse_unit_facts_gen flag u e True true
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
    unfold unit_unboxed_fields in Hunboxed.
    assert (Hx' : (Node_variable x, fields)
                  ∈ fst (unboxed_fields_of_names F T (bound_code_my_closures e)
                           dominated to_unbox (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hx).
    destruct (unboxed_fields_of_names_in F T _ dominated to_unbox _ _ _ Hx')
      as (a & next' & ->).
    destruct (mk_unboxed_fields_in F T _ dominated to_unbox _ a _ next' f _
                Hin)
      as (use & Huse).
    destruct (get_fields_in F T _ f use Huse)
      as (w & base & Hacc & Hbase & Hw).
    rewrite (add_usages_none F T _ Hreads) in Hbase.
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
      is_dead_var (solution_of_tables T) x kind = false ->
      is_var_used (solution_of_tables T) x kind = true ->
      kind = Value ∨ (∃ nk : naked_number_kind, kind = Naked_number nk) ->
      ∃ v : variable,
        Keyed.lookup field_eqb f fields = Some (Not_unboxed v)).
  { intros x y kind f fields Hx Hy Hacc Hdead Hused Hkind.
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
                    (fresh_name (names_of e)) a (Node_variable x) Hin Ha)
        as Hb.
      rewrite Hunboxed in Hb. cbn [fst] in Hb. unfold fields_of in Hx.
      exfalso.
      exact (keyed_lookup_none_not_in node_eqb node_eqb_decides _ _ Hx Hb). }
    apply (fields_of_in unboxed) in Hy.
    assert (Hy' : (Node_variable y, fields)
                  ∈ fst (unboxed_fields_of_names F T
                           (bound_code_my_closures e) dominated to_unbox
                           (fresh_name (names_of e))))
      by (rewrite Hunboxed; exact Hy).
    destruct (unboxed_fields_of_names_dom F T _ dominated to_unbox _ _ _ Hy')
      as (a & next' & Hdom & Ha & ->).
    unfold check_unboxed_reads in Hcheck. rewrite forallb_forall in Hcheck.
    specialize (Hcheck _ Hacc). cbn beta iota in Hcheck.
    rewrite Hsx, Hux, Hhx in Hcheck. simpl in Hcheck.
    rewrite forallb_forall in Hcheck. specialize (Hcheck _ Hdom).
    cbn beta iota in Hcheck. rewrite node_eqb_refl, Ha in Hcheck.
    simpl in Hcheck. apply andb_true_iff in Hcheck as [Hnau Hsrc].
    apply negb_true_iff in Hnau.
    assert (Hblock : is_block_like_field f = true).
    { rewrite forallb_forall in Hreads. exact (Hreads _ Hacc). }
    assert (Hfs : is_function_slot f = false)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hdata : is_data_field f = true)
      by (destruct f; try discriminate Hblock; reflexivity).
    assert (Hus : mem_pair (Node_variable y) (Node_variable y) (t_usages T)
                  = true)
      by exact (usages_accessor _ _ _ Hcl _ _ _ Hacc Hnau).
    assert (Hhu : mem_node (Node_variable y) (t_has_usage T) = true)
      by exact (has_usage_accessor _ _ _ Hcl _ _ _ Hux Hacc).
    assert (Huses : mem_node (Node_variable y)
                      (add_usages_through_function_slots F T
                         (bound_code_my_closures e) true
                         (get_direct_usages T [Node_variable y]))
                    = true).
    { rewrite (add_usages_none F T _ Hreads). apply mem_node_In.
      apply get_direct_usages_self; assumption. }
    destruct (get_fields_found F T _ _ _ _ Hacc Huses Hfs Hux)
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
  destruct (proj1 (URH.rebuild_ubh F (solution_of_tables T)
                     (fun n => mem_node n cannot_change) code_deps
                     preserve_direct_calls (static_names code_ids_bound e)
                     (bound_code_metadata e) unboxed
                     (fresh_name (names_of e)) next types noted deps cds
                     (unit_kept e) (unit_codes e)
                     (static_names symbols_of_binding e) is_local Hcl Hslots
                     Hfields_read Hreads_found)
              e empty_unbox_env [] (unit_conts u) uinv_empty (Forall2_nil _)
              Hloc' Hedges)
    as (Hub & _ & _).
  exists unboxed. split; [ exact Hdistinct | exact Hub ].
Qed.

(* ======================================================================== *)
(* The exit                                                                 *)
(* ======================================================================== *)

(* Where the run of the unit rebuilt without unboxing exits, the run of
   the unit rebuilt with unboxing exits through the same continuation,
   with the same values and heap, from a state with the same static
   environment. *)
Lemma unbox_exit_heap (flag : reaper_preserve_direct_calls)
    (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (n : nat) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record) :
  is_local_slots_only is_local ->
  in_heap_fragment e = true ->
  check_closed is_local (traverse_unit flag u e) T = true ->
  local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
  check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
  run n
    (State
       (rebuild_unit_from_tables (traverse_unit flag u e) T
          preserve_direct_calls e)
       [] [] empty_static_env empty_heap no_reads)
  = Step s ->
  step s = Exit k vs hp rd ->
  ∃ (m : nat) (t : state) (rd' : read_record),
    run m
      (State
         (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
            code_deps dominated to_unbox preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)
    = Step t
    ∧ step t = Exit k vs hp rd' ∧ state_statics t = state_statics s.
Proof.
  intros Hslots Hfragment Hclosed Hlocal Hcheck Hrun Hexit.
  destruct (rebuild_unbox_ubh_heap None flag is_local u e T cannot_change
              code_deps dominated to_unbox preserve_direct_calls Hslots
              Hfragment Hclosed Hlocal Hcheck)
    as (M & Hdistinct & Hub).
  assert (Hrel :
    URH.relatedh M
      (State
         (rebuild_unit_from_tables (traverse_unit flag u e) T
            preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)
      (State
         (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
            code_deps dominated to_unbox preserve_direct_calls e)
         [] [] empty_static_env empty_heap no_reads)).
  { exists []. split; [ exact Hub | split; [ apply URP.env_rel_empty | ] ].
    split; [ constructor | repeat split ]. }
  destruct (shrinking_exit_state (URH.relatedh M) eq measure
              (URH.relatedh_answered M Hdistinct) n _ _ s k vs hp rd Hrel
              Hrun Hexit)
    as (m & t & vs' & rd' & Hm & Hr & Ht & <-).
  exists m, t, rd'. split; [ exact Hm | split; [ exact Ht | ] ].
  destruct Hr as (K & _ & _ & _ & Hst & _). symmetry. exact Hst.
Qed.
