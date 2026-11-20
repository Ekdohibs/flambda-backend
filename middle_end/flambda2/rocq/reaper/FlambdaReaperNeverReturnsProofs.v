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

(* The proofs of FlambdaReaperNeverReturnsSpec.v. Nothing here needs review.

   The fragment without the check gives the fragment where every code of
   the unit passes the check and binds no statics in its body
   ([unchecked_checked]). The check is [no_live_return], the analogue for
   the original body of
   FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.no_return, which is
   of the rebuilt body; a call that returns to the return continuation
   never returns by [call_never_returns] there. The row of [call_ok] on the
   return arity gives the premise of that lemma on calls of known code. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeastSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperNeverReturnsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperRebuildProofs
  FlambdaReaperWrapperProofs FlambdaReaperWrapperDerivedProofs
  FlambdaReaperLeastProofs FlambdaReaperUnboxingRefinesCallsProofs.
Import ListNotations.

Module CNR := FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.
Module RCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module UE := FlambdaReaperUnboxing.Examples.
Module RS := FlambdaReaperRebuildSpec.

(* ======================================================================== *)
(* The fragment with the check                                              *)
(* ======================================================================== *)

Lemma forallb_ext_in {A : Type} (f g : A -> bool) (l : list A) :
  (∀ x, x ∈ l -> f x = g x) -> forallb f l = forallb g l.
Proof.
  induction l as [ | x l IH]; intros H; [ reflexivity | ].
  cbn [forallb]. rewrite (H x (or_introl eq_refl)).
  rewrite IH; [ reflexivity | ]. intros y Hy. apply H. right. exact Hy.
Qed.

(* The calls of a term, not in the code it binds. *)
Fixpoint term_calls (e : expr) : list apply_expr :=
  match e with
  | Let _ _ _ body | Let_static _ body => term_calls body
  | Let_cont_non_recursive _ h body =>
    handler_term_calls h ++ term_calls body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_term_calls h) handlers
    ++ term_calls body
  | Apply ap => [ap]
  | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_term_calls (h : continuation_handler) : list apply_expr :=
  match h with
  | Handler _ _ handler => term_calls handler
  end.

Section Equiv.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable nr : code_id -> bool.

Abbreviation calls_expr :=
  (FlambdaReaperUnboxingRefinesCallsSpec.calls_expr F sol cannot_change
     code_deps unit_code_ids codes unboxed nr).
Abbreviation calls_handler :=
  (FlambdaReaperUnboxingRefinesCallsSpec.calls_handler F sol cannot_change
     code_deps unit_code_ids codes unboxed nr).
Abbreviation un_expr :=
  (calls_unchecked_expr F sol cannot_change code_deps unit_code_ids codes
     unboxed nr).
Abbreviation un_handler :=
  (calls_unchecked_handler F sol cannot_change code_deps unit_code_ids codes
     unboxed nr).
Abbreviation un_code :=
  (calls_unchecked_code F sol cannot_change code_deps unit_code_ids codes
     unboxed nr).
Abbreviation call_ok :=
  (FlambdaReaperUnboxingRefinesCallsSpec.call_ok F sol cannot_change
     code_deps unit_code_ids codes unboxed nr).

(* Where no code is bound, the two fragments agree. *)
Lemma unchecked_no_statics :
  (∀ e : expr, ∀ kinds C,
     binds_statics e = false -> un_expr kinds C e = calls_expr kinds C e)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       handler_binds_statics h = false ->
       un_handler kinds C ps h = calls_handler kinds C ps h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [binds_statics handler_binds_statics calls_unchecked_expr
         calls_unchecked_handler
         FlambdaReaperUnboxingRefinesCallsSpec.calls_expr
         FlambdaReaperUnboxingRefinesCallsSpec.calls_handler].
  - intros x kind d body IH kinds C Hb. rewrite (IH _ _ Hb). reflexivity.
  - intros bindings body _ IH kinds C Hb.
    apply orb_false_iff in Hb as [Hbs Hb]. rewrite (IH _ _ Hb).
    f_equal. apply forallb_ext_in. intros b Hin.
    destruct b as [id c | id | s const]; [ | reflexivity | ];
      exfalso; assert (H : existsb binding_binds bindings = true)
        by (apply existsb_exists; eexists; split; [ exact Hin | ];
            reflexivity);
      congruence.
  - intros k h body IHh IH kinds C Hb.
    apply orb_false_iff in Hb as [Hbh Hb].
    rewrite (IHh _ _ _ Hbh), (IH _ _ Hb). reflexivity.
  - intros handlers body IHh IH kinds C Hb.
    apply orb_false_iff in Hb as [Hbh Hb]. rewrite (IH _ _ Hb). f_equal.
    apply forallb_ext_in. intros [k h] Hin. apply (IHh k h Hin).
    destruct (handler_binds_statics h) eqn:E; [ | reflexivity ].
    rewrite <- Hbh. symmetry. apply existsb_exists. exists (k, h).
    split; [ exact Hin | exact E ].
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - reflexivity.
  - intros params metadata handler IH kinds C ps Hb.
    rewrite (IH _ _ Hb). reflexivity.
Qed.

(* The check of [calls_code] for code [id]. *)
Definition code_checked (id : code_id) (c : code) : Prop :=
  binds_statics (code_body c) = false
  ∧ (negb (nr id)
     || negb
          (among (code_return_continuation c)
             (live_conts F sol code_deps codes (code_body c))))
    = true.

Lemma unchecked_checked :
  (∀ e : expr, ∀ kinds C,
     un_expr kinds C e = true ->
     (∀ id c, (id, c) ∈ unit_codes e -> code_checked id c) ->
     calls_expr kinds C e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       un_handler kinds C ps h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h -> code_checked id c) ->
       calls_handler kinds C ps h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [unit_codes handler_unit_codes calls_unchecked_expr
         calls_unchecked_handler
         FlambdaReaperUnboxingRefinesCallsSpec.calls_expr
         FlambdaReaperUnboxingRefinesCallsSpec.calls_handler].
  - intros x kind d body IH kinds C H Hc.
    apply andb_true_iff in H as [H1 H2].
    rewrite H1, (IH _ _ H2 Hc). reflexivity.
  - intros bindings body _ IH kinds C H Hc.
    apply andb_true_iff in H as [Hbs H].
    rewrite (IH _ _ H
               (fun id c Hin => Hc id c (in_or_app _ _ _ (or_intror Hin)))).
    rewrite andb_true_r. rewrite forallb_forall in Hbs |- *.
    intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    cbn [calls_unchecked_binding
         FlambdaReaperUnboxingRefinesCallsSpec.calls_binding] in Hbs |- *.
    assert (Hin' : (id, c) ∈ unit_codes (Let_static bindings body)).
    { cbn [unit_codes]. apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id c). split; [ exact Hin | left; reflexivity ]. }
    destruct (Hc id c Hin') as [Hb Hchk].
    destruct c as [params mc ret exn md body'].
    cbn [code_body code_return_continuation] in Hb, Hchk.
    cbn [calls_unchecked_code
         FlambdaReaperUnboxingRefinesCallsSpec.calls_code] in Hbs |- *.
    rewrite Hchk, andb_true_r.
    apply andb_true_iff in Hbs as [Hm Hbs]. rewrite Hm. cbn [andb].
    destruct (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs _ _ _ _ _ _ _)
      as [[pds rds] | ]; [ | discriminate Hbs ].
    destruct (FlambdaReaperUnboxingRefinesCallsSpec.code_exn_decision _ _ _ _)
      as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    rewrite <- (proj1 unchecked_no_statics body' _ _ Hb). exact Hbs.
  - intros k h body IHh IH kinds C H Hc.
    cbv zeta in H |- *. apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split.
    + exact (IHh _ _ _ Hh
               (fun id c Hin => Hc id c (in_or_app _ _ _ (or_introl Hin)))).
    + exact (IH _ _ H
               (fun id c Hin => Hc id c (in_or_app _ _ _ (or_intror Hin)))).
  - intros handlers body IHh IH kinds C H Hc.
    cbv zeta in H |- *. apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split;
      [ | exact (IH _ _ H
                   (fun id c Hin =>
                      Hc id c (in_or_app _ _ _ (or_intror Hin)))) ].
    rewrite forallb_forall in Hh |- *.
    intros [k h] Hin. specialize (Hh _ Hin). cbv beta iota in Hh |- *.
    refine (IHh k h Hin _ _ _ Hh _). intros id c Hc'. apply Hc.
    apply in_or_app. left. apply in_flat_map. exists (k, h). auto.
  - intros ap kinds C H _. exact H.
  - intros ac kinds C H _. exact H.
  - intros scrutinee arms kinds C H _. exact H.
  - intros kinds C _ _. reflexivity.
  - intros params metadata handler IH kinds C ps H Hc.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2 Hc).
Qed.

(* The fragment without the check finds each code of the unit and each
   call of a term. *)
Lemma unchecked_parts :
  (∀ e : expr, ∀ kinds C,
     un_expr kinds C e = true ->
     (∀ id c, (id, c) ∈ unit_codes e -> un_code id c = true)
     ∧ ∀ ap, ap ∈ term_calls e -> ∃ kinds' C', call_ok kinds' C' ap = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       un_handler kinds C ps h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h -> un_code id c = true)
       ∧ ∀ ap, ap ∈ handler_term_calls h ->
           ∃ kinds' C', call_ok kinds' C' ap = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [unit_codes handler_unit_codes term_calls handler_term_calls
         calls_unchecked_expr calls_unchecked_handler].
  - intros x kind d body IH kinds C H.
    apply andb_true_iff in H as [_ H]. exact (IH _ _ H).
  - intros bindings body _ IH kinds C H.
    apply andb_true_iff in H as [Hbs H].
    destruct (IH _ _ H) as [Hcodes Hcalls].
    split; [ | exact Hcalls ].
    intros id c Hin. apply in_app_or in Hin as [Hin | Hin];
      [ | exact (Hcodes id c Hin) ].
    apply in_flat_map in Hin as (b & Hb & Hin).
    destruct b as [id' c' | id' | s const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    rewrite forallb_forall in Hbs. exact (Hbs _ Hb).
  - intros k h body IHh IH kinds C H.
    cbv zeta in H. apply andb_true_iff in H as [Hh H].
    destruct (IHh _ _ _ Hh) as [Hc1 Hk1]. destruct (IH _ _ H) as [Hc2 Hk2].
    split; [ intros ? ? Hin | intros ? Hin ];
      apply in_app_or in Hin as [Hin | Hin]; eauto.
  - intros handlers body IHh IH kinds C H.
    cbv zeta in H. apply andb_true_iff in H as [Hh H].
    destruct (IH _ _ H) as [Hc2 Hk2]. rewrite forallb_forall in Hh.
    split.
    + intros id c Hin. apply in_app_or in Hin as [Hin | Hin]; eauto.
      apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      specialize (Hh _ Hkh). cbv beta iota in Hh.
      exact (proj1 (IHh k h Hkh _ _ _ Hh) id c Hin).
    + intros ap Hin. apply in_app_or in Hin as [Hin | Hin]; eauto.
      apply in_flat_map in Hin as ([k h] & Hkh & Hin).
      specialize (Hh _ Hkh). cbv beta iota in Hh.
      exact (proj2 (IHh k h Hkh _ _ _ Hh) ap Hin).
  - intros ap kinds C H. split; [ intros ? ? [] | ].
    intros ap' [<- | []]. eauto.
  - intros ac kinds C H. split; [ intros ? ? [] | intros ? [] ].
  - intros scrutinee arms kinds C H. split; [ intros ? ? [] | intros ? [] ].
  - intros kinds C H. split; [ intros ? ? [] | intros ? [] ].
  - intros params metadata handler IH kinds C ps H.
    apply andb_true_iff in H as [_ H]. exact (IH _ _ H).
Qed.

(* The body of code of the fragment without the check is of it. *)
Lemma unchecked_code_body (id : code_id) (c : code) :
  un_code id c = true -> ∃ kinds C, un_expr kinds C (code_body c) = true.
Proof.
  destruct c as [params mc ret exn md body].
  cbn [calls_unchecked_code code_body]. intros H.
  apply andb_true_iff in H as [_ H].
  destruct (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs _ _ _ _ _ _ _)
    as [[pds rds] | ]; [ | discriminate H ].
  destruct (FlambdaReaperUnboxingRefinesCallsSpec.code_exn_decision _ _ _ _)
    as [d | ]; [ | rewrite !andb_false_r in H; discriminate H ].
  apply andb_true_iff in H as [_ H]. eauto.
Qed.

(* A call of code that is known, of the fragment, expects of it the
   results of its metadata. *)
Lemma call_ok_known (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (id : code_id) :
  call_ok kinds C ap = true ->
  code_id_actually_called F sol ap = Some id ->
  ∃ md, lookup id codes = Some md ∧ ap_return_arity ap = cm_result_arity md.
Proof.
  unfold FlambdaReaperUnboxingRefinesCallsSpec.call_ok. intros H Hid.
  rewrite Hid in H. destruct (ap_callee ap) as [f | | ]; try discriminate H.
  repeat (apply andb_true_iff in H as [_ H]).
  destruct (lookup id codes) as [md | ]; [ | discriminate H ].
  exists md. split; [ reflexivity | ].
  destruct (FlambdaReaperUnboxingRefinesCallsSpec.code_pdescs _ _ _ _ _ _ _)
    as [[pds rds] | ]; [ | discriminate H ].
  repeat (apply andb_true_iff in H as [H ?]).
  match goal with
  | Hr : arity_eqb (ap_return_arity ap) _ = true |- _ =>
    apply arity_eqb_eq; exact Hr
  end.
Qed.

End Equiv.

(* ======================================================================== *)
(* The return continuation is not live                                      *)
(* ======================================================================== *)

(* An arm of a switch whose keys are distinct is kept. *)
Lemma distinct_in_compact (arms : list (Z * apply_cont)) (d : Z)
    (ac : apply_cont) :
  Keyed.distinct Z.eqb (map fst arms) = true -> (d, ac) ∈ arms ->
  (d, ac) ∈ Keyed.compact Z.eqb arms.
Proof.
  intros Hd Hin. apply (keyed_in_compact Z.eqb Z.eqb_eq).
  induction arms as [ | [d' ac'] arms IH]; [ destruct Hin | ].
  cbn [map fst Keyed.distinct] in Hd. apply andb_true_iff in Hd as [Hn Hd].
  cbn [Keyed.lookup]. destruct Hin as [[= -> ->] | Hin].
  - rewrite Z.eqb_refl. reflexivity.
  - destruct (Z.eqb_spec d d') as [-> | Hne]; [ | exact (IH Hd Hin) ].
    exfalso. apply negb_true_iff in Hn. apply Bool.not_true_iff_false in Hn.
    apply Hn. apply existsb_exists. exists d'.
    split; [ exact (in_map fst _ _ Hin) | apply Z.eqb_refl ].
Qed.

(* The check of the term holds of the body of each code of the unit. *)
Lemma keys_and_groups_codes :
  (∀ e : expr,
     keys_and_groups_ok e = true ->
     ∀ id c, (id, c) ∈ unit_codes e -> keys_and_groups_ok (code_body c) = true)
  ∧ (∀ h : continuation_handler,
       handler_keys_and_groups_ok h = true ->
       ∀ id c, (id, c) ∈ handler_unit_codes h ->
         keys_and_groups_ok (code_body c) = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [keys_and_groups_ok handler_keys_and_groups_ok unit_codes
         handler_unit_codes].
  - intros x kind d body IH H. exact (IH H).
  - intros bindings body _ IH H id c Hin.
    apply andb_true_iff in H as [Hbs H].
    apply in_app_or in Hin as [Hin | Hin]; [ | exact (IH H id c Hin) ].
    apply in_flat_map in Hin as (b & Hb & Hin).
    destruct b as [id' c' | id' | s const]; cbn in Hin;
      [ | destruct Hin | destruct Hin ].
    destruct Hin as [[= <- <-] | []].
    rewrite forallb_forall in Hbs. specialize (Hbs _ Hb).
    destruct c' as [params mc ret exn md body']. exact Hbs.
  - intros k h body IHh IH H id c Hin.
    apply andb_true_iff in H as [Hh H].
    apply in_app_or in Hin as [Hin | Hin]; eauto.
  - intros handlers body IHh IH H id c Hin.
    apply andb_true_iff in H as [Hh H]. rewrite forallb_forall in Hh.
    apply in_app_or in Hin as [Hin | Hin]; [ | eauto ].
    apply in_flat_map in Hin as ([k h] & Hkh & Hin).
    exact (IHh k h Hkh (Hh _ Hkh) id c Hin).
  - intros ap _ id c [].
  - intros ac _ id c [].
  - intros scrutinee arms _ id c [].
  - intros _ id c [].
  - intros params metadata handler IH H. exact (IH H).
Qed.

Section Live.

Variable is_local : field -> bool.
Variable u : unit_interface.
Variable e0 : expr.
Variable T : tables.

Abbreviation F := (traverse_unit Auto u e0).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation cds := (unit_code_deps Auto u e0).
Abbreviation codes := (bound_code_metadata e0).

Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast : least_solution is_local F T.
Variable slots_read : Prop.
Variable noted : symbol -> Prop.
Variable deps : closure_dep -> Prop.
Hypothesis Hconsts : g_any_source G Node_all_constants.
Hypothesis Hnodup : NoDup (map fst cds).
Hypothesis Hids :
  ∀ id, id ∈ map fst cds -> id ∈ static_names code_ids_of_binding e0.

Abbreviation code_edges :=
  (FlambdaReaperFlowProofs.code_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).

Hypothesis Hcode_edges : ∀ id c, (id, c) ∈ unit_codes e0 -> code_edges id c.

Abbreviation edges :=
  (FlambdaReaperFlowProofs.expr_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).
Abbreviation handler_edges :=
  (FlambdaReaperFlowProofs.handler_edges G noted deps cds (unit_kept e0)
     (unit_codes e0) (static_names symbols_of_binding e0) slots_read False).
Abbreviation lc := (live_conts F S cds codes).
Abbreviation lch := (live_conts_handler F S cds codes).

(* A call of code that is known expects of it no more results than its
   code dep has. *)
Definition known_ok (ap : apply_expr) : Prop :=
  ∀ (id : code_id) (cd : code_dep),
    code_id_actually_called F S ap = Some id -> (id, cd) ∈ cds ->
    length (unarize (ap_return_arity ap)) <= length (cd_return cd).

(* As [CNR.no_return], of the term itself: [k], whose parameters include
   [r], which has no source, is not live. *)
Lemma no_live_return :
  (∀ e : expr, ∀ ctx ce k ns r,
     edges ctx ce e ->
     conts_counted (FlambdaReaperFlowProofs.counts ce) e = true ->
     binds_statics e = false ->
     FlambdaReaperWrapperProofs.live T ctx ->
     keys_and_groups_ok e = true ->
     (∀ ap, ap ∈ term_calls e -> known_ok ap) ->
     lookup k ce = Some ns -> r ∈ ns -> s_has_source S r = false ->
     (k ∈ free_continuations e -> raises_to k e = false) ->
     k ∉ lc e)
  ∧ (∀ h : continuation_handler, ∀ ctx ce k ns r,
       handler_edges ctx ce h ->
       handler_conts_counted (FlambdaReaperFlowProofs.counts ce) h = true ->
       handler_binds_statics h = false ->
       FlambdaReaperWrapperProofs.live T ctx ->
       handler_keys_and_groups_ok h = true ->
       (∀ ap, ap ∈ handler_term_calls h -> known_ok ap) ->
       lookup k ce = Some ns -> r ∈ ns -> s_has_source S r = false ->
       (k ∈ handler_free_continuations h -> handler_raises_to k h = false) ->
       k ∉ lch h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - (* Let *)
    intros x kind d body IH ctx ce k ns r [_ He] Hc Hb Hl Hkg Hcalls Hk Hr
      Hno Hx.
    exact (IH ctx ce k ns r He Hc Hb Hl Hkg Hcalls Hk Hr Hno Hx).
  - (* Let_static: a body of code binds nothing. *)
    intros bindings body _ IH ctx ce k ns r [_ He] Hc Hb Hl Hkg Hcalls Hk Hr
      Hno Hx Hin.
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbs Hb].
    cbn [keys_and_groups_ok] in Hkg. apply andb_true_iff in Hkg as [_ Hkg].
    cbn [live_conts] in Hin. apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as (b & Hbin & Hin).
      destruct b as [id c | id | sym const];
        cbn [live_conts_binding] in Hin; [ | destruct Hin | destruct Hin ].
      assert (H : existsb binding_binds bindings = true)
        by (apply existsb_exists; exists (Bind_code id c); auto).
      congruence.
    + exact (IH ctx ce k ns r He Hc Hb Hl Hkg Hcalls Hk Hr Hno Hx Hin).
  - (* Let_cont_non_recursive *)
    intros k' h body IHh IH ctx ce k ns r [Hh He] Hc Hb Hl Hkg Hcalls Hk Hr
      Hno Hx.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbh Hb].
    cbn [keys_and_groups_ok] in Hkg. apply andb_true_iff in Hkg as [Hkgh Hkg].
    cbn [free_continuations raises_to] in Hx.
    cbn [live_conts]. intros Hin. apply in_app_or in Hin as [Hin | Hin].
    + refine (IHh ctx ce k ns r Hh Hch Hbh Hl Hkgh
                (fun ap H => Hcalls ap (in_or_app _ _ _ (or_introl H)))
                Hk Hr Hno _ Hin).
      intros H.
      assert (H' : k ∈ handler_free_continuations h
                     ++ without [k'] (free_continuations body))
        by (apply in_or_app; left; exact H).
      apply Hx in H'. apply orb_false_iff in H' as [H' _]. exact H'.
    + apply CNR.in_without_inv in Hin as [Hin Hne].
      assert (Hkk : k <> k') by (intros ->; apply Hne; left; reflexivity).
      refine (IH ctx ((k', FlambdaReaperTraversal.handler_params h) :: ce)
                k ns r He Hc Hb Hl Hkg
                (fun ap H => Hcalls ap (in_or_app _ _ _ (or_intror H)))
                _ Hr Hno _ Hin).
      * cbn [lookup Keyed.lookup].
        destruct (Pos.eqb k k') eqn:E; [ apply Pos.eqb_eq in E; congruence | ].
        exact Hk.
      * intros H.
        assert (H' : k ∈ handler_free_continuations h
                       ++ without [k'] (free_continuations body)).
        { apply in_or_app. right. unfold without. apply filter_In.
          split; [ exact H | ]. apply negb_true_iff.
          destruct (among k [k']) eqn:E; [ | reflexivity ].
          apply FlambdaMapProofs.among_In in E. destruct E as [E | []].
          congruence. }
        apply Hx in H'. apply orb_false_iff in H' as [_ H']. exact H'.
  - (* Let_cont_recursive *)
    intros handlers body IHh IH ctx ce k ns r [Hhs He] Hc Hb Hl Hkg Hcalls Hk
      Hr Hno Hx.
    cbn [conts_counted] in Hc. apply andb_true_iff in Hc as [Hch Hc].
    rewrite FlambdaReaperWrapperProofs.counts_rec in Hch, Hc.
    rewrite forallb_forall in Hch.
    cbn [binds_statics] in Hb. apply orb_false_iff in Hb as [Hbh Hb].
    cbn [keys_and_groups_ok] in Hkg. apply andb_true_iff in Hkg as [Hkgh Hkg].
    rewrite forallb_forall in Hkgh.
    cbn [free_continuations raises_to] in Hx.
    cbn [live_conts]. intros Hin.
    apply CNR.in_without_inv in Hin as [Hin Hnot].
    assert (Hk' : lookup k (FlambdaReaperFlowProofs.rec_conts ce handlers)
                  = Some ns).
    { unfold FlambdaReaperFlowProofs.rec_conts.
      rewrite FlambdaMapProofs.lookup_app.
      rewrite (CNR.lookup_map_none k handlers _ Hnot). exact Hk. }
    assert (Hin_e : ∀ k0, k0 ∉ map fst handlers ->
               k0 ∈ flat_map (fun '(_, h) => handler_free_continuations h)
                      handlers
                    ++ free_continuations body ->
               k0 ∈ without (map fst handlers)
                      (flat_map
                         (fun '(_, h) => handler_free_continuations h) handlers
                       ++ free_continuations body)).
    { intros k0 Hn H. unfold without. apply filter_In. split; [ exact H | ].
      apply negb_true_iff. destruct (among k0 (map fst handlers)) eqn:E;
        [ | reflexivity ].
      apply FlambdaMapProofs.among_In in E. contradiction. }
    apply in_app_or in Hin as [Hin | Hin].
    + apply in_flat_map in Hin as ([k1 h1] & Hkh & Hin).
      refine (IHh k1 h1 Hkh ctx _ k ns r
                (proj1 (FlambdaMapProofs.every_in _ _ _) Hhs _ Hkh)
                (Hch _ Hkh) _ Hl (Hkgh _ Hkh) _ Hk' Hr Hno _ Hin).
      * destruct (handler_binds_statics h1) eqn:E; [ | reflexivity ].
        rewrite <- Hbh. symmetry.
        apply existsb_exists. exists (k1, h1). auto.
      * intros ap H. apply Hcalls. apply in_or_app. left.
        apply in_flat_map. exists (k1, h1). split; [ exact Hkh | exact H ].
      * intros H.
        assert (H' := Hin_e k Hnot
                        (in_or_app _ _ _
                           (or_introl (proj2 (in_flat_map _ _ _)
                                         (ex_intro _ (k1, h1) (conj Hkh H)))))).
        apply Hx in H'. apply orb_false_iff in H' as [H' _].
        destruct (handler_raises_to k h1) eqn:E; [ | reflexivity ].
        rewrite <- H'. symmetry. apply existsb_exists. exists (k1, h1). auto.
    + refine (IH ctx _ k ns r He Hc Hb Hl Hkg _ Hk' Hr Hno _ Hin).
      * intros ap H. apply Hcalls. apply in_or_app. right. exact H.
      * intros H.
        assert (H' := Hin_e k Hnot (in_or_app _ _ _ (or_intror H))).
        apply Hx in H'. apply orb_false_iff in H' as [_ H']. exact H'.
  - (* Apply: a call to [k] never returns, and none raises to it. *)
    intros ap ctx ce k ns r He Hc _ Hl Hkg Hcalls Hk Hr Hno Hx.
    cbn [free_continuations raises_to] in Hx.
    cbn [live_conts]. intros Hin.
    destruct (apply_never_returns F S cds codes ap) eqn:Hnr.
    + destruct Hin as [<- | []]. rewrite Pos.eqb_refl in Hx.
      discriminate (Hx (or_intror (or_introl eq_refl))).
    + destruct Hin as [<- | [<- | []]].
      * assert (Hok : CNR.returns_ok u e0 T ap).
        { split; [ | exact (Hcalls ap (or_introl eq_refl)) ].
          intros Hkind Hnil. cbn [keys_and_groups_ok] in Hkg.
          rewrite Hkind, Hnil in Hkg. discriminate Hkg. }
        rewrite (CNR.call_never_returns is_local u e0 T Hloc Hleast slots_read
                   noted deps Hconsts Hnodup Hids Hcode_edges ctx ce ap ns r
                   He Hc Hl Hok Hk Hr Hno) in Hnr.
        discriminate Hnr.
      * rewrite Pos.eqb_refl in Hx.
        discriminate (Hx (or_intror (or_introl eq_refl))).
  - (* Apply_cont *)
    intros ac ctx ce k ns r He Hc _ _ _ _ Hk Hr Hno Hx.
    cbn [free_continuations raises_to] in Hx.
    cbn [live_conts]. unfold live_apply_cont. intros Hin.
    destruct (existsb (simple_has_no_source S) (ac_args ac)) eqn:Hex;
      [ destruct Hin | ].
    destruct (Pos.eq_dec k (ac_continuation ac)) as [-> | Hneq].
    + cbn [conts_counted] in Hc.
      rewrite (CNR.ac_no_source is_local u e0 T Hleast noted Hconsts ce ac ns
                 r He Hc Hk Hr Hno) in Hex.
      discriminate Hex.
    + rewrite (CNR.traps_of k ac Hin Hneq) in Hx. discriminate (Hx Hin).
  - (* Switch: every arm is kept. *)
    intros scrutinee arms ctx ce k ns r He Hc _ _ Hkg _ Hk Hr Hno Hx.
    cbn [keys_and_groups_ok] in Hkg.
    cbn [free_continuations raises_to] in Hx.
    cbn [live_conts]. intros Hin.
    apply in_flat_map in Hin as ([d ac] & Harm & Hin). cbn [snd] in Hin.
    unfold live_apply_cont in Hin.
    destruct (existsb (simple_has_no_source S) (ac_args ac)) eqn:Hex;
      [ destruct Hin | ].
    pose proof (distinct_in_compact arms d ac Hkg Harm) as Hac.
    cbn [FlambdaReaperFlowProofs.expr_edges] in He.
    pose proof (proj1 (FlambdaMapProofs.every_in _ _ _) He _ Hac) as Heac.
    cbn beta iota in Heac.
    cbn [conts_counted] in Hc. rewrite forallb_forall in Hc.
    pose proof (Hc _ Harm) as Hcac. cbn beta iota in Hcac.
    assert (Ht : traps k ac = false).
    { destruct (traps k ac) eqn:E; [ | reflexivity ].
      assert (H : k ∈ flat_map (fun arm => apply_cont_continuations (snd arm))
                        arms)
        by (apply in_flat_map; exists (d, ac);
            split; [ exact Harm | exact Hin ]).
      specialize (Hx H). rewrite <- Hx. symmetry.
      apply existsb_exists. exists (d, ac). split; [ exact Harm | exact E ]. }
    destruct (Pos.eq_dec k (ac_continuation ac)) as [-> | Hneq].
    + rewrite (CNR.ac_no_source is_local u e0 T Hleast noted Hconsts ce ac ns
                 r Heac Hcac Hk Hr Hno) in Hex.
      discriminate Hex.
    + rewrite (CNR.traps_of k ac Hin Hneq) in Ht. discriminate Ht.
  - intros ctx ce k ns r _ _ _ _ _ _ _ _ _ _ Hin. exact Hin.
  - (* A handler *)
    intros params metadata handler IH ctx ce k ns r He Hc Hb Hl Hkg Hcalls
      Hk Hr Hno Hx.
    exact (IH ctx ce k ns r He Hc Hb Hl Hkg Hcalls Hk Hr Hno Hx).
Qed.

(* The return continuation of code of the unit that is used and some
   result of which has no source is not live in its body. *)
Lemma code_live_return (id : code_id) (c : code) :
  (id, c) ∈ unit_codes e0 ->
  code_exn_handlers_well_formed c = true ->
  keys_and_groups_ok (code_body c) = true ->
  (∀ ap, ap ∈ term_calls (code_body c) -> known_ok ap) ->
  used_never_returns S cds id = true ->
  among (code_return_continuation c) (lc (code_body c)) = false.
Proof.
  intros Hin Hcwf Hkg Hcalls Hnr.
  unfold used_never_returns in Hnr. apply andb_true_iff in Hnr as [Hnr Hused].
  pose proof (Hcode_edges id c Hin) as Hce.
  destruct c as [params my_closure ret exn metadata body].
  cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
  destruct Hce as (cd & Hcd & _ & _ & _ & _ & Hlen & _ & _ & Hne & Hbs & Hcc
                   & Hbe).
  pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
    as Hlookup.
  unfold code_never_returns in Hnr. rewrite Hlookup in Hnr.
  apply existsb_exists in Hnr as (r & Hr & Hno). apply negb_true_iff in Hno.
  cbn [code_exn_handlers_well_formed] in Hcwf.
  apply andb_true_iff in Hcwf as [Hcwf _].
  apply andb_true_iff in Hcwf as [Hways _].
  unfold ways_out_well_formed in Hways. rewrite forallb_forall in Hways.
  assert (Hx : ret ∈ free_continuations body -> raises_to ret body = false).
  { intros H. specialize (Hways ret H). cbv beta in Hways.
    destruct (Pos.eqb ret exn) eqn:E;
      [ apply Pos.eqb_eq in E; congruence | ].
    apply andb_true_iff in Hways as [Hw _]. apply negb_true_iff in Hw.
    exact Hw. }
  assert (Hcounted :
            conts_counted
              (FlambdaReaperFlowProofs.counts
                 [(exn, [cd_exn cd]); (ret, cd_return cd)])
              body
            = true).
  { unfold FlambdaReaperFlowProofs.counts. cbn [map length]. rewrite Hlen.
    exact Hcc. }
  assert (Hret : lookup ret [(exn, [cd_exn cd]); (ret, cd_return cd)]
                 = Some (cd_return cd)).
  { cbn [lookup Keyed.lookup].
    destruct (Pos.eqb ret exn) eqn:E; [ apply Pos.eqb_eq in E; congruence | ].
    rewrite Pos.eqb_refl. reflexivity. }
  cbn [code_return_continuation code_body] in Hkg, Hcalls |- *.
  apply Bool.not_true_iff_false. intros Hin'.
  apply FlambdaMapProofs.among_In in Hin'.
  refine (proj1 no_live_return body (Some id) _ ret _ r Hbe Hcounted Hbs _
            Hkg Hcalls Hret Hr Hno Hx Hin').
  intros c0 [= <-]. exact Hused.
Qed.

End Live.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

(* The fragment without the check gives the fragment with it, for the code
   that is used and some result of which has no source. *)
Lemma unit_checked (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cc : node -> bool)
    (unboxed : list (node * unboxed_fields)) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  exn_handlers_well_formed e = true ->
  keys_and_groups_ok e = true ->
  least_solution is_local (traverse_unit Auto u e) T ->
  calls_unchecked_expr (traverse_unit Auto u e) (solution_of_tables T) cc
    (unit_code_deps Auto u e) (static_names code_ids_bound e)
    (bound_code_metadata e) unboxed
    (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
    [] [] e
  = true ->
  FlambdaReaperUnboxingRefinesCallsSpec.calls_expr (traverse_unit Auto u e)
    (solution_of_tables T) cc (unit_code_deps Auto u e)
    (static_names code_ids_bound e) (bound_code_metadata e) unboxed
    (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
    [] [] e
  = true.
Proof.
  intros Hloc Hfrag Hewf Hkg Hleast Hun.
  set (F := traverse_unit Auto u e) in *.
  set (S := solution_of_tables T) in *.
  set (cds := unit_code_deps Auto u e) in *.
  set (ids := static_names code_ids_bound e) in *.
  set (codes := bound_code_metadata e) in *.
  set (nr := used_never_returns S cds) in *.
  destruct (FlambdaReaperWrapperProofs.unit_facts Auto u e True Hfrag
              (or_introl I))
    as (noted & deps & Hedges & Hconsts & _ & Hnodup & Hids).
  pose proof (fun id c Hc =>
                proj1 (FlambdaReaperWrapperProofs.unit_codes_edges
                         _ _ _ _ _ _ _ _ _)
                  e None (unit_conts u) Hedges id c Hc) as Hcode_edges.
  assert (Hnd : NoDup (map fst (unit_codes e))).
  { rewrite (proj1 FlambdaReaperRebuildProofs.unit_codes_ids e).
    apply distinct_NoDup. unfold in_flow_fragment in Hfrag.
    apply andb_true_iff in Hfrag as [_ Hd]. exact Hd. }
  destruct (proj1 (unchecked_parts F S cc cds ids codes unboxed nr) e [] []
              Hun)
    as [Hcodes _].
  refine (proj1 (unchecked_checked F S cc cds ids codes unboxed nr) e [] []
            Hun _).
  intros id c Hin.
  pose proof (Hcode_edges id c Hin) as Hce.
  split.
  { destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hbs & _).
    exact Hbs. }
  destruct (nr id) eqn:Hnr; [ cbn [negb orb] | reflexivity ].
  apply negb_true_iff.
  refine (code_live_return is_local u e T Hloc Hleast True noted deps Hconsts
            Hnodup Hids Hcode_edges id c Hin _ _ _ Hnr).
  - exact (proj1 FlambdaReaperRebuildProofs.unit_codes_exn_wf e Hewf id c
             Hin).
  - exact (proj1 keys_and_groups_codes e Hkg id c Hin).
  - (* A call of known code expects the results of that code. *)
    intros ap Hap id' cd Hcall Hcd.
    destruct (unchecked_code_body F S cc cds ids codes unboxed nr id c
                (Hcodes id c Hin))
      as (kinds & C & Hbody).
    destruct (proj2 (proj1 (unchecked_parts F S cc cds ids codes unboxed nr)
                       _ _ _ Hbody) ap Hap)
      as (kinds' & C' & Hok).
    destruct (call_ok_known F S cc cds ids codes unboxed nr kinds' C' ap id'
                Hok Hcall)
      as (md & Hmd & Hra).
    destruct (proj1 FlambdaReaperWrapperProofs.unit_codes_names e id'
                (Hids id' (in_map fst _ _ Hcd)))
      as [(c' & Hin') _].
    unfold codes in Hmd.
    rewrite (FlambdaReaperRebuildProofs.lookup_bound_code_metadata e id' c'
               Hnd Hin') in Hmd.
    injection Hmd as <-.
    pose proof (Hcode_edges id' c' Hin') as Hce'.
    destruct c' as [p' mc' ret' exn' md' body'].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce'.
    destruct Hce' as (cd' & Hcd' & _ & _ & _ & _ & Hlen' & _).
    pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd)
      as L1.
    pose proof (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd')
      as L2.
    rewrite L1 in L2. injection L2 as <-.
    cbn [metadata_of_code] in Hra. rewrite Hra, Hlen'. apply le_n.
Qed.

Theorem never_returns_derived_observably :
  never_returns_derived_observably_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox pdc Hloc Hslots
    Hfrag Hany Hres Hwf Hexits Hkg Hleast Hcheck Hfue Hcc Hwid Hcalls Hreads.
  assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
  { unfold well_formed_unit in Hwf.
    destruct (unit_exn_handlers_well_formed (unit_exn u) e);
      [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  unfold unit_exn_handlers_well_formed in Hue.
  apply andb_true_iff in Hue as [_ Hewf].
  apply (RCP.rebuild_unbox_fixed_observably_never_returns
           Auto is_local u e T cannot_change dominated to_unbox pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  unfold calls_unboxing_unchecked in Hcalls. unfold calls_unboxing_nr.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  rewrite Huses, andb_true_r.
  exact (unit_checked is_local u e T _ _ Hloc Hfrag Hewf Hkg Hleast Hcalls).
Qed.

Theorem never_returns_derived_refines_cc :
  never_returns_derived_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox pdc Hloc Hfrag Hewf
    Hkg Hleast Hcheck Hcalls Hreads.
  apply (RCP.rebuild_unbox_fixed_refines_cc_never_returns
           Auto is_local u e T cannot_change dominated to_unbox pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  unfold calls_unboxing_unchecked in Hcalls. unfold calls_unboxing_nr.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  rewrite Huses, andb_true_r.
  exact (unit_checked is_local u e T _ _ Hloc Hfrag Hewf Hkg Hleast Hcalls).
Qed.

Theorem never_returns_derived_example :
  never_returns_derived_example_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution UE.no_local NeverExample.facts
                     NeverExample.tables).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (Hcalls :
    calls_unboxing_unchecked
      (used_never_returns (solution_of_tables NeverExample.tables)
         NeverExample.code_deps)
      NeverExample.facts NeverExample.tables NeverExample.cannot_change
      NeverExample.code_deps [] [] NeverExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  assert (Hkg : keys_and_groups_ok NeverExample.unit_expr = true)
    by (vm_compute; reflexivity).
  assert (Hprem :
    in_flow_fragment NeverExample.unit_expr = true
    ∧ RS.any_kind_arguments NeverExample.unit_expr = true
    ∧ RS.results_kinded [] [] NeverExample.unit_expr = true
    ∧ well_formed_unit UE.interface NeverExample.unit_expr = true
    ∧ RS.exits_one_value UE.interface NeverExample.unit_expr = true
    ∧ check_closed UE.no_local NeverExample.facts NeverExample.tables
      = true
    ∧ check_field_used_exact UE.no_local NeverExample.facts
        NeverExample.tables
      = true
    ∧ check_cc_closed NeverExample.facts NeverExample.tables
        NeverExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        NeverExample.facts NeverExample.tables
      = true
    ∧ FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
        NeverExample.facts NeverExample.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hreads).
  split; [ exact Hleast | ].
  split; [ exact Hkg | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  apply (never_returns_derived_observably
           UE.no_local UE.interface NeverExample.unit_expr
           NeverExample.tables NeverExample.cannot_change [] [] true);
    try assumption.
  - intros f Hf. discriminate Hf.
  - left. intros f. reflexivity.
Qed.
