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
(* The proofs of FlambdaReaperUnboxingRefinesNever2Spec.v. Nothing here
   needs review.

   FlambdaReaperUnboxingRefinesCallsProofs.v relates the two rebuilt units
   on the fragment [calls_unboxing_reach_never2_nr], given [flow_run],
   [over_run] and [rem_run] of the first
   ([rebuild_unbox_modes_refines_cc_never2_gen]): its construction is that
   of [calls_unboxing_reach_never_nr] made generic in the check of a call
   ([rebuild_uc_hw]), with [call_rebuilt2]. A call whose code is not known
   that never returns is related by
   FlambdaReaperUnboxingRefinesCallsSimProofs.uc_call_ind_never2
   ([call_rebuilt_never2], [never_code_shape2]), whose step
   ([never_call_answered2_hp]) reads the bound of a partial application off
   [pa_inv]: a partial application holds a group of arguments at least
   (FlambdaReaperPartialGroupsProofs.pa_made). The three runs of the
   first rebuilt unit are proved as in
   FlambdaReaperUnboxingRefinesNeverProofs.v, whose proofs this repeats.
   The checks are monotone in the check of a call ([calls_w_mono]), which
   gives the inclusion of the fragment of
   FlambdaReaperUnboxingRefinesNeverSpec.v; and in the reach, where
   [never_code_ok2] reads [reach true] negated, so that a smaller reach
   makes it weaker, as the others ([call_mono_n2]), which gives the forms
   read off the unit's own analysis. *)

From Stdlib Require Import ZArith List Bool PeanoNat Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaBindersSpec FlambdaRenameApartSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesProofs FlambdaReaperLeast
  FlambdaReaperLocalFields FlambdaReaperLocalFieldsSpec
  FlambdaReaperSolutionSpec FlambdaReaperSolutionProofs
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesCallsSimProofs
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesNever2Spec
  FlambdaReaperUnboxingRefinesReachProofs
  FlambdaReaperUnboxingRefinesReach2Proofs
  FlambdaReaperUnboxingRefinesOver2Proofs
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesNeverProofs
  FlambdaReaperUnboxingRefinesUnitFlowProofs
  FlambdaReaperUnboxingRefinesCallsProofs FlambdaReaperPartialGroupsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperLocalFieldsProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperCallingConventionsProofs
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

Import FlambdaReaperFlowProofs.

(* ======================================================================== *)
(* The checks are monotone                                                  *)
(* ======================================================================== *)

Section MonoW.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
(* The check of a call, and a weaker one. *)
Variable ok ok' : finite_map flambda_kind -> finite_map (list pdesc * bool)
  -> apply_expr -> bool.
(* The variables at which the two are compared. *)
Variable vars : list variable.
Hypothesis ok_le : ∀ kinds C ap,
  (∀ f : variable, ap_callee ap = Var f -> f ∈ vars) ->
  ok kinds C ap = true -> ok' kinds C ap = true.

Lemma calls_w_mono :
  (∀ e : expr, ∀ kinds C,
     incl (variables_of e) vars ->
     calls_expr_w F sol cannot_change code_deps unit_code_ids codes unboxed
       nr ok kinds C e
     = true ->
     calls_expr_w F sol cannot_change code_deps unit_code_ids codes unboxed
       nr ok' kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       incl (variables_in_handler h) vars ->
       calls_handler_w F sol cannot_change code_deps unit_code_ids codes
         unboxed nr ok kinds C ps h
       = true ->
       calls_handler_w F sol cannot_change code_deps unit_code_ids codes
         unboxed nr ok' kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds C Hv H. cbn [calls_expr_w] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_let vars x kind d body Hv) H2).
  - intros bindings body IHc IH kinds C Hv H. cbn [calls_expr_w] in H |- *.
    apply andb_true_iff in H as [Hbs H].
    rewrite (IH _ _ (vars_static_body vars bindings body Hv) H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    pose proof (vars_static_code vars bindings body id c Hv Hin) as Hvc.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_w calls_code_w code_body] in Hbs, IHb, Hvc |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hvc Hb).
  - intros k h body IHh IH kinds C Hv H. cbn [calls_expr_w] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont vars k h body Hv) as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff.
    split; [ exact (IHh _ _ _ Hvh Hh) | exact (IH _ _ Hvb H) ].
  - intros handlers body IHh IH kinds C Hv H. cbn [calls_expr_w] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont_rec vars handlers body Hv)
      as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ Hvb H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ (Hvh k h Hin) Hh).
  - intros ap kinds C Hv H. cbn [calls_expr_w] in H |- *.
    exact (ok_le kinds C ap (vars_callee vars ap Hv) H).
  - intros ac kinds C _ H. exact H.
  - intros scrutinee arms kinds C _ H. exact H.
  - intros kinds C _ H. exact H.
  - intros params metadata handler IH kinds C ps Hv H.
    cbn [calls_handler_w] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_handler vars params metadata handler Hv) H2).
Qed.

End MonoW.

Section Mono2.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.
(* The larger reach, and the smaller one. *)
Variable reach reach' : bool -> variable -> code_id -> bool.
Variable ovr ovr' rem rem' : code_id -> bool.
Variable vars : list variable.
Hypothesis reach_le : ∀ (b : bool) (f : variable) (id : code_id),
  f ∈ vars -> is_var_used sol f Value = true -> id ∈ map fst codes ->
  reach' b f id = true -> reach b f id = true.
Hypothesis ovr_le : ∀ id : code_id,
  id ∈ map fst codes -> ovr' id = true -> ovr id = true.
Hypothesis rem_le : ∀ id : code_id,
  id ∈ map fst codes -> rem' id = true -> rem id = true.

(* [never_code_ok2] reads [reach true] negated: a smaller reach makes it
   weaker. *)
Lemma ind_never_mono2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  f ∈ vars ->
  ind_never_ok2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach kinds C ap f
  = true ->
  ind_never_ok2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' kinds C ap f
  = true.
Proof.
  unfold ind_never_ok2. intros Hf H.
  apply andb_true_iff in H as [Hpre Hall].
  assert (Hu : is_var_used sol f Value = true).
  { pose proof Hpre as Hu.
    do 4 (apply andb_true_iff in Hu as [Hu _]).
    apply andb_true_iff in Hu as [_ Hu]. exact Hu. }
  rewrite Hpre. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  assert (Hid : id ∈ map fst codes) by exact (in_map fst codes (id, md) Hin).
  apply orb_true_iff in Hall as [Hn | Ho].
  - apply negb_true_iff in Hn.
    rewrite (reach_false sol codes reach reach' vars reach_le _ f id Hf Hu
               Hid Hn).
    reflexivity.
  - apply orb_true_iff. right.
    unfold never_code_ok2 in Ho |- *.
    apply orb_true_iff in Ho as [Ho | Ho]; [ rewrite Ho; reflexivity | ].
    apply andb_true_iff in Ho as [Ho Hlen].
    apply andb_true_iff in Ho as [Ho Hn].
    apply orb_true_iff. right. rewrite Ho, Hlen. cbn [andb].
    rewrite andb_true_r. apply negb_true_iff in Hn. apply negb_true_iff.
    exact (reach_false sol codes reach reach' vars reach_le true f id Hf Hu
             Hid Hn).
Qed.

Lemma call_mono_n2 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  (∀ f : variable, ap_callee ap = Var f -> f ∈ vars) ->
  call_ok_n2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr rem kinds C ap
  = true ->
  call_ok_n2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' ovr' rem' kinds C ap
  = true.
Proof.
  intros Hv. unfold call_ok_n2.
  assert (Hf3 : call_ok_f3 F sol cannot_change code_deps unit_code_ids codes
                  unboxed nr reach ovr rem kinds C ap
                = true ->
                call_ok_f3 F sol cannot_change code_deps unit_code_ids codes
                  unboxed nr reach' ovr' rem' kinds C ap
                = true)
    by exact (call_mono_f3 F sol cannot_change code_deps unit_code_ids codes
                unboxed nr reach reach' ovr ovr' rem rem' vars reach_le
                ovr_le rem_le kinds C ap Hv).
  destruct (ap_callee ap) as [f | sym | cst] eqn:Hc;
    [ | exact Hf3 | exact Hf3 ].
  destruct (code_id_actually_called F sol ap) eqn:Hid; [ exact Hf3 | ].
  unfold ind_call_ok_n2.
  destruct (apply_never_returns F sol code_deps codes ap).
  - apply ind_never_mono2. exact (Hv f eq_refl).
  - apply (ind_call_mono_f3 F sol cannot_change code_deps unit_code_ids codes
             unboxed reach reach' ovr ovr' rem rem' vars reach_le ovr_le
             rem_le).
    exact (Hv f eq_refl).
Qed.

End Mono2.

(* [call_ok_n] is within [call_ok_n2]. *)
Lemma call_ok_n_n2 (F : facts) (sol : solution) (cannot_change : node -> bool)
    (code_deps : finite_map code_dep) (unit_code_ids : list code_id)
    (codes : finite_map code_metadata) (unboxed : list (node * unboxed_fields))
    (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  call_ok_n F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr rem kinds C ap
  = true ->
  call_ok_n2 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr rem kinds C ap
  = true.
Proof.
  unfold call_ok_n, call_ok_n2.
  destruct (ap_callee ap) as [f | sym | cst]; try exact (fun H => H).
  destruct (code_id_actually_called F sol ap); [ exact (fun H => H) | ].
  unfold ind_call_ok_n, ind_call_ok_n2.
  destruct (apply_never_returns F sol code_deps codes ap);
    [ | exact (fun H => H) ].
  unfold ind_never_ok, ind_never_ok2. intros H.
  apply andb_true_iff in H as [Hpre Hall]. rewrite Hpre. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  apply orb_true_iff in Hall as [Hn | Ho]; [ rewrite Hn; reflexivity | ].
  unfold never_code_ok2. rewrite Ho. cbn [orb]. apply orb_true_r.
Qed.

(* ======================================================================== *)
(* The fragments                                                            *)
(* ======================================================================== *)

Lemma calls_unboxing_in_never2_reach (nr : code_id -> bool) (F : facts)
    (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (reach : bool -> variable -> code_id -> bool)
    (ovr rem : code_id -> bool) (e : expr) :
  calls_unboxing_reach_never_nr nr F T cannot_change code_deps dominated
    to_unbox reach ovr rem e
  = true ->
  calls_unboxing_reach_never2_nr nr F T cannot_change code_deps dominated
    to_unbox reach ovr rem e
  = true.
Proof.
  unfold calls_unboxing_reach_never_nr, calls_unboxing_reach_never2_nr.
  cbv zeta. intros H.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  rewrite (proj1 (calls_n_w F (solution_of_tables T)
                    (fun n => mem_node n cannot_change) code_deps
                    (static_names code_ids_bound e) (bound_code_metadata e)
                    (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                    reach ovr rem))
    in H1.
  exact (proj1 (calls_w_mono F (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps
                  (static_names code_ids_bound e) (bound_code_metadata e)
                  (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                  _ _ (variables_of e)
                  (fun kinds C ap _ => call_ok_n_n2 _ _ _ _ _ _ _ _ _ _ _
                                         kinds C ap))
           e [] [] (incl_refl _) H1).
Qed.

Theorem calls_unboxing_never_in_never2 :
  calls_unboxing_never_in_never2_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox F1 T1 e H.
  exact (calls_unboxing_in_never2_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach2 F1 T1) (over_reach F1 T1)
           (rem_reach F1 T1) e H).
Qed.

Theorem calls_unboxing_unit_never_in_never2 :
  calls_unboxing_unit_never_in_never2_statement.
Proof.
  intros nr F T cannot_change code_deps dominated to_unbox e H.
  exact (calls_unboxing_in_never2_reach nr F T cannot_change code_deps
           dominated to_unbox (flow_reach2 F T) (over_reach F T)
           (rem_reach F T) e H).
Qed.

(* The fragment read off the unit's own analysis is in that read off the
   first rebuilt unit's, where the reach of the latter is within that of
   the former. *)
Lemma unit_never2_covered (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) :
  calls_unboxing_unit_never2_nr nr F T cannot_change code_deps dominated
    to_unbox e
  = true ->
  used_reach_covered F T F1 T1 e = true ->
  calls_unboxing_never2_nr nr F T cannot_change code_deps dominated to_unbox
    F1 T1 e
  = true.
Proof.
  unfold calls_unboxing_unit_never2_nr, calls_unboxing_never2_nr,
    calls_unboxing_reach_never2_nr.
  cbv zeta. intros H Hcov.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  destruct (covered_parts F T F1 T1 e Hcov) as (Hr & Ho & Hm).
  exact (proj1 (calls_w_mono F (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps
                  (static_names code_ids_bound e) (bound_code_metadata e)
                  (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                  _ _ (variables_of e)
                  (call_mono_n2 F (solution_of_tables T)
                     (fun n => mem_node n cannot_change) code_deps
                     (static_names code_ids_bound e) (bound_code_metadata e)
                     (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                     (flow_reach2 F T) (flow_reach2 F1 T1) (over_reach F T)
                     (over_reach F1 T1) (rem_reach F T) (rem_reach F1 T1)
                     (variables_of e) Hr Ho Hm))
           e [] [] (incl_refl _) H1).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The statement, of the units of the fragment with the primitives of the
   heap if [hb]. *)
Lemma rebuild_unbox_nr_ind2_refines_cc_gen (hb : bool) :
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T T1 : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool),
    is_local_slots_only is_local ->
    flow_fragment hb (unit_codes e) (static_names symbols_of_binding e) e
    && distinct (static_names code_ids_of_binding e) = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    in_heap_fragment
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
    = true ->
    check_closed is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    check_local_reads_used is_local
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1
    = true ->
    calls_unboxing_never2_nr nr (traverse_unit flag u e) T cannot_change
      (unit_code_deps flag u e) dominated to_unbox
      (traverse_unit flag u
         (first_rebuild flag u e T cannot_change preserve_direct_calls))
      T1 e
    = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (first_rebuild flag u e T cannot_change preserve_direct_calls)
      (rebuild_unit_unbox_fixed (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck.
  unfold first_rebuild in *.
  set (e1 := rebuild_unit_cc_from_tables (traverse_unit flag u e) T
               cannot_change (unit_code_deps flag u e) pdc e) in *.
  set (F1 := traverse_unit flag u e1) in *.
  pose proof (check_closed_sound is_local _ _ Hclosed1) as Hc1.
  pose proof (FlambdaReaperLocalFieldsProofs.check_local_reads_used_sound
                is_local _ _ Hlrs1) as Hlr1.
  apply (rebuild_unbox_modes_refines_cc_never2_gen hb flag is_local u e T
           cannot_change dominated to_unbox pdc nr (flow_reach2 F1 T1)
           (over_reach F1 T1) (rem_reach F1 T1) Hslots Hfrag Hclosed Hcalls
           Hcheck).
  - (* The flow, as in FlambdaReaperUnboxingRefinesReach2Proofs.v. *)
    intros n s Hrun ap f v id slots _ _ Hf.
    destruct (run_flows_called2 flag is_local u e1 T1 n Hslots Hfrag1 Hc1
                Hlr1)
      as (located & cds & Hcds & Hnd & Hflows).
    fold F1 in Hcds, Hflows. unfold e1 in Hflows. rewrite Hrun in Hflows.
    destruct Hflows as ((Hvars & _ & Hst & _) & Hchains & Hpa).
    assert (Hclosure :
      ∀ (v0 : value) (id0 : code_id) (slots0 : list (value_slot * value)),
        lookup f (state_vars s) = Some v0 ->
        closure_of (state_statics s) v0 = Some (id0, slots0) ->
        flow_reach F1 T1 f id0 = true)
      by (intros v0 id0 slots0 Hf0 Hclo0;
          exact (closure_reach F1 T1 cds (unit_kept e1) located
                   (state_vars s) (state_statics s) f v0 id0 slots0 Hcds
                   Hvars Hst Hf0 Hclo0)).
    split; intros Hclo; cbn [flow_reach2].
    + exact (Hclosure v id slots Hf Hclo).
    + destruct v as [c | code vs0 | slot decls vs | sym | tag fields
                     | elements | contents | l | cl applied aa];
        try (unfold flow_reach_u; rewrite (Hclosure _ id slots Hf Hclo);
             reflexivity).
      exact (partial_reach is_local F1 T1 cds (unit_kept e1) located
               (state_statics s) f cl applied aa id slots Hc1 Hcds Hnd Hst
               Hchains (Hpa f _ Hf) (Hvars f _ Hf) Hclo).
  - (* What a step over-applies. *)
    intros n s Hrun.
    pose proof (run_over_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hover.
    fold F1 in Hover. unfold e1 in Hover. rewrite Hrun in Hover.
    exact (step_over_ok_mono _ _ s (over_used_reach F1 T1) Hover).
  - (* What is left of an over-application calls. *)
    intros n s Hrun.
    pose proof (run_rem_used flag is_local u e1 T1 n Hslots Hfrag1 Hc1 Hlr1)
      as Hrem.
    fold F1 in Hrem. unfold e1 in Hrem. rewrite Hrun in Hrem.
    exact (step_rem_ok_mono _ _ s (rem_used_reach F1 T1) Hrem).
Qed.

Theorem rebuild_unbox_nr_ind2_refines_cc :
  rebuild_unbox_nr_ind2_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr.
  exact (rebuild_unbox_nr_ind2_refines_cc_gen false flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr).
Qed.

Theorem rebuild_unbox_nr_ind2_refines_observably_local :
  rebuild_unbox_nr_ind2_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (refines_observably_refines _ _ _
           (FlambdaReaperCallingConventionsProofs
            .rebuild_cc_refines_observably_local
              flag is_local u e T cannot_change pdc Hslots Hfrag Hany
              Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid)
           (rebuild_unbox_nr_ind2_refines_cc flag is_local u e T T1
              cannot_change dominated to_unbox pdc nr Hslots Hfrag Hclosed
              Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck)).
Qed.

Theorem rebuild_unbox_nr_ind2_refines_observably :
  rebuild_unbox_nr_ind2_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcheck.
  exact (rebuild_unbox_nr_ind2_refines_observably_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed
           (FlambdaReaperFlowProofs.old_premise_local_reads_used flag
              is_local u e T Hslots Hproj)
           Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcheck).
Qed.


Theorem rebuild_unbox_unit_never2_refines_cc :
  rebuild_unbox_unit_never2_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_refines_cc flag is_local u e T T1 cannot_change
           dominated to_unbox pdc _ Hslots Hfrag Hclosed Hfrag1 Hclosed1
           Hlrs1 (unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov)
           Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_observably_local :
  rebuild_unbox_unit_never2_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_refines_observably_local flag is_local u e T
           T1 cannot_change dominated to_unbox pdc _ Hslots Hfrag Hany Hrk
           Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never2_refines_observably :
  rebuild_unbox_unit_never2_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind2_refines_observably flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hproj Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_never2_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.
