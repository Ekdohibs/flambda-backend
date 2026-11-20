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
   ([unchecked_checked]; [repr_unchecked_checked] for the closures). The
   check is [no_live_return], the analogue for the original body of
   FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.no_return, which is
   of the rebuilt body; a call that returns to the return continuation
   never returns by [call_never_returns], copied from there for tables that
   are closed and pass [check_apply_widgets] and [check_known_witnesses].
   A least solution passes them with the fix of local fields or without
   ([least_either_checks]). The row of [call_ok] (of [repr_call]) on the
   return arity gives the premise of that lemma on calls of known code. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperRebuild
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaReaperLeastSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperSolutionProofs FlambdaReaperNeverReturnsSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperCallingConventionsSpec.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs FlambdaReaperRebuildProofs
  FlambdaReaperWrapperProofs FlambdaReaperWrapperDerivedProofs
  FlambdaReaperLeastProofs FlambdaReaperUnboxingRefinesCallsProofs
  FlambdaReaperWrapperSpec FlambdaReaperWrapperAutoProofs
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesClosuresProofs.
Import ListNotations.

Module CNR := FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.
Module RCP := FlambdaReaperUnboxingRefinesCallsProofs.
Module UE := FlambdaReaperUnboxing.Examples.
Module RS := FlambdaReaperRebuildSpec.
Module WS := FlambdaReaperWrapperSpec.
Module WP := FlambdaReaperWrapperProofs.
Module WA := FlambdaReaperWrapperAutoProofs.
Module LP := FlambdaReaperLeastProofs.
Module RCL := FlambdaReaperUnboxingRefinesClosuresProofs.

Abbreviation known := (Field_call_witness Known_arity_code_pointer).

Ltac unfold_rels :=
  cbn [graph_of_facts solution_of_tables holds g_alias g_use g_accessor
       g_constructor g_argument g_parameter g_propagate
       g_alias_if_any_source g_any_usage g_any_source g_any_source_if_used
       g_zero_alloc_source s_alias s_any_usage s_any_source s_has_usage
       s_has_source s_usages s_sources s_reading_field s_escaping_field
       s_field_used] in *.

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
(* The closures fragment with the check                                     *)
(* ======================================================================== *)

Section ReprEquiv.

Variable F : facts.
Variable sol : solution.
Variable cannot_change : node -> bool.
Variable code_deps : finite_map code_dep.
Variable unit_code_ids : list code_id.
Variable codes : finite_map code_metadata.
Variable unboxed : list (node * unboxed_fields).
Variable changed : list (node * closure_repr).
Variable my_closures : finite_map variable.
Variable nr : code_id -> bool.

Abbreviation repr_expr :=
  (RCS.repr_expr F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr).
Abbreviation repr_handler :=
  (RCS.repr_handler F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr).
Abbreviation ru_expr :=
  (repr_unchecked_expr F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr).
Abbreviation ru_handler :=
  (repr_unchecked_handler F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr).
Abbreviation ru_code :=
  (repr_unchecked_code F sol cannot_change code_deps unit_code_ids codes
     unboxed changed my_closures nr).
Abbreviation repr_call :=
  (RCS.repr_call F sol cannot_change code_deps unit_code_ids codes unboxed
     changed my_closures nr).

(* Where no code is bound, the two fragments agree. *)
Lemma repr_unchecked_no_statics :
  (∀ e : expr, ∀ kinds C,
     binds_statics e = false -> ru_expr kinds C e = repr_expr kinds C e)
  ∧ (∀ h : continuation_handler, ∀ kinds C ds,
       handler_binds_statics h = false ->
       ru_handler kinds C ds h = repr_handler kinds C ds h).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [binds_statics handler_binds_statics repr_unchecked_expr
         repr_unchecked_handler RCS.repr_expr RCS.repr_handler].
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
  - intros params metadata handler IH kinds C ds Hb.
    rewrite (IH _ _ Hb). reflexivity.
Qed.

Lemma repr_unchecked_checked :
  (∀ e : expr, ∀ kinds C,
     ru_expr kinds C e = true ->
     (∀ id c, (id, c) ∈ unit_codes e ->
        code_checked F sol code_deps codes nr id c) ->
     repr_expr kinds C e = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ds,
       ru_handler kinds C ds h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h ->
          code_checked F sol code_deps codes nr id c) ->
       repr_handler kinds C ds h = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [unit_codes handler_unit_codes repr_unchecked_expr
         repr_unchecked_handler RCS.repr_expr RCS.repr_handler].
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
    cbn [repr_unchecked_binding RCS.repr_binding] in Hbs |- *.
    assert (Hin' : (id, c) ∈ unit_codes (Let_static bindings body)).
    { cbn [unit_codes]. apply in_or_app. left. apply in_flat_map.
      exists (Bind_code id c). split; [ exact Hin | left; reflexivity ]. }
    destruct (Hc id c Hin') as [Hb Hchk].
    destruct c as [params mc ret exn md body'].
    cbn [code_body code_return_continuation] in Hb, Hchk.
    cbn [repr_unchecked_code RCS.repr_code] in Hbs |- *.
    rewrite Hchk, andb_true_r.
    destruct (code_change_u _ _ _ _ _ _ _) as [ | pd rd ];
      [ rewrite !andb_false_r in Hbs; discriminate Hbs | ].
    rewrite <- (proj1 repr_unchecked_no_statics body' _ _ Hb). exact Hbs.
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
  - intros params metadata handler IH kinds C ds H Hc.
    apply andb_true_iff in H as [H1 H2]. rewrite H1. exact (IH _ _ H2 Hc).
Qed.

(* The closures fragment without the check finds each code of the unit and
   each call of a term. *)
Lemma repr_unchecked_parts :
  (∀ e : expr, ∀ kinds C,
     ru_expr kinds C e = true ->
     (∀ id c, (id, c) ∈ unit_codes e -> ru_code id c = true)
     ∧ ∀ ap, ap ∈ term_calls e -> ∃ C', repr_call C' ap = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ds,
       ru_handler kinds C ds h = true ->
       (∀ id c, (id, c) ∈ handler_unit_codes h -> ru_code id c = true)
       ∧ ∀ ap, ap ∈ handler_term_calls h -> ∃ C', repr_call C' ap = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind;
    cbn [unit_codes handler_unit_codes term_calls handler_term_calls
         repr_unchecked_expr repr_unchecked_handler].
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
  - intros params metadata handler IH kinds C ds H.
    apply andb_true_iff in H as [_ H]. exact (IH _ _ H).
Qed.

(* The body of code of the closures fragment without the check is of it. *)
Lemma repr_unchecked_code_body (id : code_id) (c : code) :
  ru_code id c = true -> ∃ kinds C, ru_expr kinds C (code_body c) = true.
Proof.
  destruct c as [params mc ret exn md body].
  cbn [repr_unchecked_code code_body]. intros H.
  apply andb_true_iff in H as [_ H].
  destruct (code_change_u _ _ _ _ _ _ _) as [ | pd rd ]; [ discriminate H | ].
  apply andb_true_iff in H as [_ H]. eauto.
Qed.

(* A call of the closures fragment expects of its code the results of its
   metadata. *)
Lemma repr_call_known (C : finite_map (list decision * bool))
    (ap : apply_expr) (id : code_id) :
  repr_call C ap = true ->
  code_id_actually_called F sol ap = Some id ->
  ∃ md, lookup id codes = Some md ∧ ap_return_arity ap = cm_result_arity md.
Proof.
  unfold RCS.repr_call. intros H Hid.
  rewrite Hid in H. destruct (ap_callee ap) as [f | | ]; try discriminate H.
  apply andb_true_iff in H as [_ H].
  destruct (lookup id codes) as [md | ]; [ | discriminate H ].
  destruct (lookup id my_closures) as [mc | ]; [ | discriminate H ].
  exists md. split; [ reflexivity | ].
  destruct (code_change_u _ _ _ _ _ _ _) as [ | pd rd ]; [ discriminate H | ].
  repeat (apply andb_true_iff in H as [H ?]).
  match goal with
  | Hr : arity_eqb (ap_return_arity ap) _ = true |- _ =>
    apply arity_eqb_eq; exact Hr
  end.
Qed.

End ReprEquiv.

(* ======================================================================== *)
(* What the proofs read of a least solution                                 *)
(* ======================================================================== *)

(* FlambdaReaperWrapperAutoProofs.traverse_known_witnesses of a least
   solution of the rules with the fix of local fields: the inversions are
   of tuples that the rule of the fix does not conclude. *)
Section KnownLocal.

Variable a : acc.
Variable is_local : field -> bool.
Variable T : tables.
Hypothesis Hloc : is_local_slots_only is_local.
Hypothesis Hleast :
  least_solution_local is_local (facts_of_list (acc_facts a)) T.
Hypothesis Hkf : ∀ id cd fct,
  (id, cd) ∈ acc_code_deps a -> fct ∈ acc_facts a ->
  cd_known_arity_call_witness cd ∈ LP.AutoWidgets.into_nodes fct ->
  WA.KnownWitnesses.known_fact id cd fct.
Hypothesis Hcons : ∀ b from,
  Fact_constructor b known from ∈ acc_facts a ->
  (∃ id cd, (id, cd) ∈ acc_code_deps a
            ∧ from = cd_known_arity_call_witness cd)
  ∨ Fact_any_source from ∈ acc_facts a.

Abbreviation F := (facts_of_list (acc_facts a)).
Abbreviation G := (graph_of_facts F).
Abbreviation S := (solution_of_tables T).
Abbreviation D := (derivable_local is_local G S).

(* Nothing is aliased into the witness of known arity of a code dep. *)
Lemma no_alias_into_known_local (id : code_id) (cd : code_dep)
    (from : node) :
  (id, cd) ∈ acc_code_deps a ->
  ~ D (Atom_alias (cd_known_arity_call_witness cd) from).
Proof.
  intros Hin Dal. pose proof (fun fct => Hkf id cd fct Hin) as Hk.
  inversion Dal; subst;
    repeat match goal with
    | H : derivable_local _ _ _ (Atom_reading_field _ _) |- _ =>
      inversion H; subst; clear H
    end;
    unfold_rels;
    match goal with
    | H : (_, _) ∈ f_alias _ |- _ =>
      exact (Hk _ (WA.alias_in _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_propagate _ |- _ =>
      exact (Hk _ (WA.propagate_in _ _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_alias_if_any_source _ |- _ =>
      exact (Hk _ (WA.if_any_source_in _ _ _ _ H) (or_introl eq_refl))
    | H : (_, _, _) ∈ f_parameter _ |- _ =>
      exact (Hk _ (WA.parameter_in _ _ _ _ H) (or_intror (or_introl eq_refl))
               eq_refl)
    | H : (_, _, _) ∈ f_accessor _ |- _ =>
      exact (Hk _ (WA.accessor_in _ _ _ _ H) (or_introl eq_refl))
    end.
Qed.

(* The sources of a node only read off [n] with the field of known arity
   are witnesses of known arity of code deps. *)
Lemma known_source_local (x n s : node) :
  (x, known, n) ∈ f_accessor F ->
  LP.read_only F x known n true ->
  (x, s) ∈ t_sources T ->
  ∃ id cd, (id, cd) ∈ acc_code_deps a ∧ s = cd_known_arity_call_witness cd.
Proof.
  intros Hacc Hro Hs.
  pose proof (LP.least_local_closed is_local F T Hleast) as Hc.
  pose proof (WP.known_not_local is_local Hloc) as Hknown.
  apply (proj2 (mem_pair_In _ _ _)) in Hs.
  apply (LP.least_local_derivable is_local F T Hleast (Atom_sources x s))
    in Hs.
  destruct (LP.sources_into_local is_local F T x known n true Hro s Hs)
    as (from & Dfs & Hgf & Dal & Hgx).
  assert (Hfrom : ∃ b, (b, known, from) ∈ f_constructor F).
  { destruct (LP.alias_into_local is_local F T x known n true Hro Hknown
                from Dal)
      as [(b & Hb & _) | [(bs & _ & _ & Hb) | [_ Dn]]].
    - exists b. exact Hb.
    - exists bs. exact Hb.
    - exfalso.
      apply (LP.least_local_holds is_local F T Hleast) in Dn.
      pose proof (any_source_accessor _ _ _ Hc n _ x Dn Hacc Hknown).
      unfold_rels. congruence. }
  destruct Hfrom as (b & Hb).
  destruct (Hcons _ _ (WA.constructor_in _ _ _ _ Hb))
    as [(id & cd & Hin & ->) | Hany].
  - exists id, cd. split; [ exact Hin | ].
    inversion Dfs; subst; [ reflexivity | reflexivity | ].
    exfalso. eapply no_alias_into_known_local; eassumption.
  - exfalso.
    pose proof (any_source_input _ _ _ Hc from (WA.any_source_of _ _ Hany))
      as H.
    unfold_rels. congruence.
Qed.

Lemma known_witnesses_local :
  check_widgets_read_only F = true ->
  WS.check_known_witnesses F T (acc_code_deps a) = true.
Proof.
  intros Hro. unfold check_widgets_read_only in Hro.
  rewrite forallb_forall in Hro.
  unfold WS.check_known_witnesses. apply andb_true_intro. split.
  - apply forallb_forall. intros [[x f] n] Hacc. cbv beta iota.
    destruct (field_eqb f known) eqn:Ef; [ | reflexivity ].
    apply field_eqb_eq in Ef. subst f. cbn [negb orb].
    pose proof (Hro _ Hacc) as Hrx. cbv beta iota in Hrx.
    rewrite field_eqb_refl in Hrx. cbn [negb orb] in Hrx.
    apply LP.only_read_into_read_only in Hrx.
    apply forallb_forall. intros [x' s] Hs. cbv beta iota.
    destruct (node_eqb x' x) eqn:Ex; [ | reflexivity ].
    apply node_eqb_eq in Ex. subst x'.
    destruct (mem_node s (t_has_source T)); [ | reflexivity ].
    cbn [negb orb].
    destruct (known_source_local x n s Hacc Hrx Hs) as (id & cd & Hin & ->).
    apply existsb_exists. exists (id, cd). split; [ exact Hin | ].
    apply node_eqb_refl.
  - apply forallb_forall. intros [id cd] Hin. apply forallb_forall.
    intros [[w f] r] Hcw. cbv beta iota.
    destruct (node_eqb w (cd_known_arity_call_witness cd)) eqn:Ew;
      [ | reflexivity ].
    apply node_eqb_eq in Ew. subst w. cbn [negb orb].
    pose proof (Hkf id cd _ Hin (WA.constructor_in _ _ _ _ Hcw)
                  (or_introl eq_refl)) as Hk.
    destruct f as [ | | | | | | | [ i | ] | ]; try reflexivity;
      cbn [WA.KnownWitnesses.known_fact] in Hk.
    + rewrite Hk. apply node_eqb_refl.
    + rewrite Hk. apply node_eqb_refl.
Qed.

End KnownLocal.

Lemma traverse_known_witnesses_local (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) :
  is_local_slots_only is_local ->
  least_solution_local is_local (traverse_unit Auto u e) T ->
  WS.check_known_witnesses (traverse_unit Auto u e) T (unit_code_deps Auto u e)
  = true.
Proof.
  intros Hloc Hleast.
  destruct (WA.KnownWitnesses.unit_known Auto u e) as [Hkf Hcons].
  pose proof (LP.traverse_widgets_read_only u e) as Hro.
  unfold unit_code_deps.
  rewrite traverse_unit_acc_facts in Hleast, Hro |- *.
  exact (known_witnesses_local _ is_local T Hloc Hleast Hkf Hcons Hro).
Qed.

(* A least solution of the rules, with the fix of local fields or without,
   is closed and passes [check_apply_widgets] and [check_known_witnesses]. *)
Lemma least_either_checks (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) :
  is_local_slots_only is_local ->
  least_solution is_local (traverse_unit Auto u e) T
  ∨ least_solution_local is_local (traverse_unit Auto u e) T ->
  closed is_local (graph_of_facts (traverse_unit Auto u e))
    (solution_of_tables T)
  ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
      (traverse_unit Auto u e) T
    = true
  ∧ WS.check_known_witnesses (traverse_unit Auto u e) T
      (unit_code_deps Auto u e)
    = true.
Proof.
  intros Hloc [Hl | Hl]; (split; [ | split ]).
  - exact (LP.least_closed is_local _ T Hl).
  - exact (LP.least_apply_widgets is_local _ T Hloc Hl
             (LP.traverse_widgets_read_only u e)).
  - exact (WA.traverse_known_witnesses is_local u e T Hloc Hl).
  - exact (LP.least_local_closed is_local _ T Hl).
  - exact (LP.least_apply_widgets_local is_local _ T Hloc Hl
             (LP.traverse_widgets_read_only u e)).
  - exact (traverse_known_witnesses_local is_local u e T Hloc Hl).
Qed.

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
Hypothesis Hclosed : closed is_local G S.
Hypothesis Hwid :
  FlambdaReaperCallingConventionsSpec.check_apply_widgets F T = true.
Hypothesis Hknown : WS.check_known_witnesses F T cds = true.
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

(* FlambdaReaperWrapperDerivedProofs.CheckNeverReturns.call_never_returns,
   with what it uses, of tables with the three checks: the lemmas there
   read only those of a least solution. *)

Lemma ret_nl (rk : return_kind) : is_local (Field_return_of_call rk) = false.
Proof.
  exact (FlambdaReaperWrapperProofs.return_not_local is_local rk Hloc).
Qed.

Lemma wit_nl (ep : closure_entry_point) :
  is_local (Field_call_witness ep) = false.
Proof.
  exact (FlambdaReaperWrapperProofs.witness_not_local is_local Hloc ep).
Qed.

(* An argument of an [Apply_cont] to a continuation one of whose
   parameters has no source has none. *)
Lemma ac_no_source (ce : finite_map (list node)) (ac : apply_cont)
    (ns : list node) (r : node) :
  FlambdaReaperFlowProofs.apply_cont_edges G noted ce ac ->
  count_ok (FlambdaReaperFlowProofs.counts ce) (ac_continuation ac)
    (length (ac_args ac))
  = true ->
  lookup (ac_continuation ac) ce = Some ns -> r ∈ ns ->
  s_has_source S r = false ->
  existsb (simple_has_no_source S) (ac_args ac) = true.
Proof.
  intros [Hal _] Hc Hk Hr Hno.
  unfold count_ok in Hc.
  rewrite FlambdaReaperWrapperProofs.lookup_counts, Hk in Hc.
  cbn [option_map] in Hc. apply Nat.eqb_eq in Hc.
  apply In_nth_error in Hr as [i Hi].
  destruct (nth_error (ac_args ac) i) as [s | ] eqn:Hs.
  2:{ apply nth_error_None in Hs.
      assert (i < length ns) by (apply nth_error_Some; congruence). lia. }
  assert (Hin : (r, s) ∈ combine ns (ac_args ac)).
  { apply (nth_error_In _ i). exact (CNR.combine_nth_some _ _ i r s Hi Hs). }
  pose proof (Hal ns Hk r s Hin) as Ha.
  apply existsb_exists. exists s. split; [ exact (nth_error_In _ _ Hs) | ].
  assert (Hs0 : s_has_source S (node_of_simple s) = false).
  { destruct (s_has_source S (node_of_simple s)) eqn:E; [ | reflexivity ].
    rewrite (has_source_alias _ _ _ Hclosed _ _ E
               (alias_input _ _ _ Hclosed _ _ Ha)) in Hno.
    discriminate. }
  destruct s as [x | sym | c]; cbn [simple_has_no_source node_of_simple] in *;
    try (rewrite Hs0; reflexivity).
  rewrite (has_source_any _ _ _ Hclosed _
             (any_source_input _ _ _ Hclosed _ Hconsts)) in Hs0.
  discriminate.
Qed.

(* The witnesses of a call of unknown arity whose result of rank [i] has no
   source have no source of any source. *)
Lemma chain_no_any (returns : list node) (exn called : node) (i : nat)
    (r : node) :
  nth_error returns i = Some r -> s_has_source S r = false ->
  ∀ l first w,
    FlambdaReaperFlowProofs.caller_chain_edges G returns exn called
      ((first, w) :: l) ->
    s_any_source S w = false.
Proof.
  intros Hi Hno l. induction l as [ | [f' w'] l IH]; intros first w Hch.
  - destruct Hch as (_ & _ & _ & Hres & _).
    destruct (s_any_source S w) eqn:E; [ | reflexivity ].
    pose proof (any_source_accessor _ _ _ Hclosed w _ r E (Hres i r Hi)
                  (ret_nl _)) as Hr.
    rewrite (has_source_any _ _ _ Hclosed r Hr) in Hno. discriminate.
  - destruct Hch as (_ & _ & _ & (v & Hv & Hw') & Hrest).
    pose proof (IH f' w' Hrest) as Hw'0.
    destruct (s_any_source S w) eqn:E; [ | reflexivity ].
    pose proof (any_source_accessor _ _ _ Hclosed w _ v E Hv (ret_nl _))
      as Hav.
    rewrite (any_source_accessor _ _ _ Hclosed v _ w' Hav Hw' (wit_nl _))
      in Hw'0.
    discriminate.
Qed.

(* What [never_returns_of_call] reads along the chain of witnesses of a
   call of unknown arity is among the sources of those witnesses: a
   result of rank [i] with no source makes it [true]. *)
Lemma chain_never_returns (returns : list node) (exn called : node)
    (i : nat) (r : node) (nres : nat) :
  nth_error returns i = Some r -> s_has_source S r = false -> i < nres ->
  ∀ l first w (num : list nat) (n : nat) cs base a,
    FlambdaReaperFlowProofs.caller_chain_edges G returns exn called
      ((first, w) :: l) ->
    length num = length l ->
    CNR.approx T cs base ->
    g_accessor G a (Field_call_witness Unknown_arity_code_pointer) base ->
    (s_any_source S a = true -> s_any_source S w = true) ->
    (∀ s, s_sources S a s = true -> s_any_source S w = false ->
          s_sources S w s = true) ->
    never_returns_of_call F S Unknown_arity_code_pointer cs (n :: num) nres
    = true.
Proof.
  intros Hi Hno Hlt l.
  induction l as [ | [f' w'] l IH];
    intros first w num n cs base a Hch Hlen Happ Ha Hany Hsrc;
    pose proof (chain_no_any returns exn called i r Hi Hno _ first w Hch)
      as Hw;
    (assert (Hasa : s_any_source S a = false)
       by (destruct (s_any_source S a); [ | reflexivity ];
           rewrite (Hany eq_refl) in Hw; discriminate));
    (assert (Hasb : s_any_source S base = false)
       by (destruct (s_any_source S base) eqn:E; [ | reflexivity ];
           rewrite (any_source_accessor _ _ _ Hclosed base _ a E Ha
                      (wit_nl _)) in Hasa;
           discriminate));
    (destruct cs as [ | m]; [ cbn [CNR.approx] in Happ; congruence | ]);
    cbn [never_returns_of_call]; unfold get_field_sources;
    (match goal with
     | |- context [if existsb ?p (f_constructor F) then _ else _] =>
       destruct (existsb p (f_constructor F)) eqn:Eany
     end;
     [ exfalso; apply existsb_exists in Eany as ([[x f] y] & Hc & Ht);
       cbn beta iota in Ht;
       apply andb_true_iff in Ht as [Ht Hy];
       apply andb_true_iff in Ht as [Hx Hf];
       apply field_eqb_eq in Hf; subst f;
       pose proof (alias_accessor_source _ _ _ Hclosed base x _ a y Ha
                     (Happ x Hx) Hasb Hc) as Hay;
       rewrite (any_source_alias _ _ _ Hclosed y a Hay Hy) in Hasa;
       discriminate | ]);
    cbn beta iota;
    (* The sources read are those of [w]. *)
    (assert (Hws :
               ∀ z,
                 existsb
                   (fun '(x, f', y) =>
                      m x && field_eqb f'
                               (Field_call_witness Unknown_arity_code_pointer)
                      && s_sources S y z && s_has_source S z)
                   (f_constructor F)
                 = true ->
                 s_sources S w z = true)
     by (intros z Hz; apply existsb_exists in Hz as ([[x f] y] & Hc & Ht);
         cbn beta iota in Ht;
         apply andb_true_iff in Ht as [Ht _];
         apply andb_true_iff in Ht as [Ht Hyz];
         apply andb_true_iff in Ht as [Hx Hf];
         apply field_eqb_eq in Hf; subst f;
         pose proof (alias_accessor_source _ _ _ Hclosed base x _ a y Ha
                       (Happ x Hx) Hasb Hc) as Hay;
         assert (Hasy : s_any_source S y = false)
           by (destruct (s_any_source S y) eqn:E; [ | reflexivity ];
               rewrite (any_source_alias _ _ _ Hclosed y a Hay E) in Hasa;
               discriminate);
         exact (Hsrc z (sources_alias _ _ _ Hclosed y a z Hyz Hasy Hay Hasa)
                  Hw))).
  - destruct num as [ | n' num']; [ | discriminate ].
    destruct Hch as (_ & _ & _ & Hres & _).
    apply existsb_exists. exists i. split; [ apply in_seq; lia | ].
    apply negb_true_iff. unfold result_has_source_of.
    apply Bool.not_true_iff_false. intros E.
    apply existsb_exists in E as ([[z f] y] & Hc & Ht).
    cbn beta iota in Ht.
    apply andb_true_iff in Ht as [Ht Hy].
    apply andb_true_iff in Ht as [Hz Hf].
    apply field_eqb_eq in Hf. subst f.
    pose proof (alias_accessor_source _ _ _ Hclosed w z _ r y (Hres i r Hi)
                  (Hws z Hz) Hw Hc) as Hry.
    rewrite (has_source_alias _ _ _ Hclosed y r Hy Hry) in Hno.
    discriminate.
  - destruct num as [ | n' num']; [ discriminate | ].
    cbn [length] in Hlen. injection Hlen as Hlen.
    destruct Hch as (_ & _ & _ & (v & Hv & Hw') & Hrest).
    pose proof (chain_no_any returns exn called i r Hi Hno _ f' w' Hrest)
      as Hw'0.
    assert (Hasv : s_any_source S v = false).
    { destruct (s_any_source S v) eqn:E; [ | reflexivity ].
      rewrite (any_source_accessor _ _ _ Hclosed v _ w' E Hw' (wit_nl _))
        in Hw'0.
      discriminate. }
    refine (IH f' w' num' n' _ v w' Hrest Hlen _ Hw' (fun H => H)
              (fun s H _ => H)).
    unfold get_field_sources.
    match goal with
    | |- context [if existsb ?p (f_constructor F) then _ else _] =>
      destruct (existsb p (f_constructor F)) eqn:Eany2
    end; cbn [CNR.approx].
    + apply existsb_exists in Eany2 as ([[z f] y] & Hc & Ht).
      cbn beta iota in Ht.
      apply andb_true_iff in Ht as [Ht Hy].
      apply andb_true_iff in Ht as [Hz Hf].
      apply field_eqb_eq in Hf. subst f.
      pose proof (alias_accessor_source _ _ _ Hclosed w z _ v y Hv
                    (Hws z Hz) Hw Hc) as Hvy.
      rewrite (any_source_alias _ _ _ Hclosed y v Hvy Hy) in Hasv.
      discriminate.
    + intros x Hx. apply existsb_exists in Hx as ([[z f] y] & Hc & Ht).
      cbn beta iota in Ht.
      apply andb_true_iff in Ht as [Ht _].
      apply andb_true_iff in Ht as [Ht Hyx].
      apply andb_true_iff in Ht as [Hz Hf].
      apply field_eqb_eq in Hf. subst f.
      pose proof (alias_accessor_source _ _ _ Hclosed w z _ v y Hv
                    (Hws z Hz) Hw Hc) as Hvy.
      assert (Hasy : s_any_source S y = false).
      { destruct (s_any_source S y) eqn:E; [ | reflexivity ].
        rewrite (any_source_alias _ _ _ Hclosed y v Hvy E) in Hasv.
        discriminate. }
      exact (sources_alias _ _ _ Hclosed y v x Hyx Hasy Hvy Hasv).
Qed.

(* A call of one group of arguments whose code is not known, read at the
   field of its arity, of whose result of rank [i] no source of the node
   read off its callee has a source: it never returns
   ([never_returns_of_call]). *)
Lemma one_group_never_returns (ep : closure_entry_point) (f a : node)
    (n nres i : nat) :
  (a, Field_call_witness ep, f) ∈ f_accessor F ->
  WS.result_has_source_at F T (Field_call_witness ep) f i = false ->
  i < nres ->
  never_returns_of_call F S ep (get_direct_sources S f) [n] nres = true.
Proof.
  intros Ha Hno Hi.
  destruct (WP.no_source_at Auto u e0 T _ f a a a i Hno Ha) as [Hasa _].
  pose proof (wit_nl ep) as Hnl.
  cbn [never_returns_of_call]. unfold get_direct_sources.
  destruct (s_any_source S f) eqn:Ef.
  { rewrite (any_source_accessor _ _ _ Hclosed f _ a Ef Ha Hnl) in Hasa.
    discriminate. }
  unfold get_field_sources.
  match goal with
  | |- context [if existsb ?p (f_constructor F) then _ else _] =>
    destruct (existsb p (f_constructor F)) eqn:Eany
  end.
  { exfalso. apply existsb_exists in Eany as ([[x f'] y] & Hc & Ht).
    cbn beta iota in Ht.
    apply andb_true_iff in Ht as [Ht Hy].
    apply andb_true_iff in Ht as [Hx Hf'].
    apply andb_true_iff in Hx as [Hfx _].
    apply field_eqb_eq in Hf'. subst f'.
    pose proof (alias_accessor_source _ _ _ Hclosed f x _ a y Ha Hfx Ef Hc)
      as Hay.
    rewrite (any_source_alias _ _ _ Hclosed y a Hay Hy) in Hasa.
    discriminate. }
  cbn beta iota.
  apply existsb_exists. exists i.
  split; [ apply in_seq; lia | ].
  apply negb_true_iff. unfold result_has_source_of.
  apply Bool.not_true_iff_false. intros E.
  apply existsb_exists in E as ([[z f'] r] & Hc & Ht).
  cbn beta iota in Ht.
  apply andb_true_iff in Ht as [Ht Hr].
  apply andb_true_iff in Ht as [Hz Hf'].
  apply field_eqb_eq in Hf'. subst f'.
  apply existsb_exists in Hz as ([[x f'] y] & Hcx & Ht).
  cbn beta iota in Ht.
  apply andb_true_iff in Ht as [Ht _].
  apply andb_true_iff in Ht as [Ht Hyz].
  apply andb_true_iff in Ht as [Hx Hf'].
  apply andb_true_iff in Hx as [Hfx _].
  apply field_eqb_eq in Hf'. subst f'.
  pose proof (alias_accessor_source _ _ _ Hclosed f x _ a y Ha Hfx Ef Hcx)
    as Hay.
  assert (Hasy : s_any_source S y = false).
  { destruct (s_any_source S y) eqn:E; [ | reflexivity ].
    rewrite (any_source_alias _ _ _ Hclosed y a Hay E) in Hasa.
    discriminate. }
  pose proof (sources_alias _ _ _ Hclosed y a z Hyz Hasy Hay Hasa) as Haz.
  destruct (WP.no_source_at Auto u e0 T _ f a z r i Hno Ha) as [_ Hnr].
  rewrite (Hnr Haz Hc) in Hr. discriminate.
Qed.

(* The call of the arm, direct or of known arity, never returns: its code,
   if known, has a result with no source (that of rank [i], which the
   call expects of it), and otherwise the witness that it reads has
   none. *)
Lemma known_never_returns (ap : apply_expr) (a : node) (i : nat) :
  ap_call_kind ap <> Indirect_unknown_arity ->
  (a, known, node_of_simple (ap_callee ap)) ∈ f_accessor F ->
  WS.callee_result_has_source F T (node_of_simple (ap_callee ap)) i = false ->
  i < length (unarize (ap_return_arity ap)) ->
  (∀ (id : code_id) (cd : code_dep),
     code_id_actually_called F S ap = Some id -> (id, cd) ∈ cds ->
     length (unarize (ap_return_arity ap)) <= length (cd_return cd)) ->
  apply_never_returns F S cds (bound_code_metadata e0) ap = true.
Proof.
  intros Hk Hread Hno Hi Har.
  unfold WS.callee_result_has_source in Hno.
  destruct (WP.no_source_at Auto u e0 T _ _ a a a i Hno Hread) as [Hasa _].
  unfold apply_never_returns.
  destruct (code_id_actually_called F S ap) as [id | ] eqn:Ecall.
  - destruct (WP.called_some F S ap id Ecall) as (ids & Eids & Esing).
    assert (Hid : id ∈ ids).
    { pose proof (proj1 (FlambdaMapProofs.keyed_get_singleton _ Pos.eqb
                           WP.pos_eqb_decides id ids) Esing id) as H.
      rewrite Pos.eqb_refl in H.
      exact (proj1 (FlambdaMapProofs.keyed_among_in _ Pos.eqb
                      WP.pos_eqb_decides id ids) H). }
    destruct (WP.directly_inv F S _ ids id Eids Hid)
      as (a1 & s & Ha1 & Hs1 & Hhs & Hcid).
    destruct (WP.kw_source F T cds Hknown a1 _ s Ha1 Hs1 Hhs)
      as (id' & cd & Hcd & ->).
    pose proof (proj2 (WP.kw_constructor F T cds Hknown id' cd _ _ Hcd Hcid)
                  eq_refl) as Heq.
    injection Heq as <-.
    destruct (WP.cds_code Auto u e0 slots_read noted deps Hnodup Hids
                         Hcode_edges id cd Hcd) as (Hkw & (c & Hc) & _).
    pose proof (Har id cd eq_refl Hcd) as Hlen.
    destruct (nth_error (cd_return cd) i) as [r | ] eqn:Hr.
    2:{ apply nth_error_None in Hr. lia. }
    destruct Hkw as (_ & Hkw_res & _).
    pose proof (Hkw_res i r Hr) as Hcr.
    destruct (WP.no_source_at Auto u e0 T _ _ a1
                (cd_known_arity_call_witness cd) r i Hno Ha1)
      as [_ Hnr].
    pose proof (Hnr Hs1 Hcr) as Hnr'.
    destruct (lookup id (bound_code_metadata e0)) as [md | ] eqn:Emd.
    + unfold code_never_returns.
      rewrite (FlambdaReaperFlowProofs.lookup_nodup_in _ _ _ Hnodup Hcd).
      apply existsb_exists. exists r.
      split; [ exact (nth_error_In _ _ Hr) | ].
      rewrite Hnr'. reflexivity.
    + exfalso. apply FlambdaMapProofs.lookup_none_not_in in Emd. apply Emd.
      rewrite (proj1 WP.bound_code_metadata_codes e0), map_map.
      apply in_map_iff. exists (id, c). split; [ reflexivity | exact Hc ].
  - destruct (ap_callee ap) as [x | sym | c0] eqn:Hcallee;
      cbv zeta.
    3:{ cbn [node_of_simple] in Hread.
        pose proof (any_source_input _ _ _ Hclosed _ Hconsts) as Hac.
        rewrite (any_source_accessor _ _ _ Hclosed _ _ a Hac Hread
                   (WP.known_not_local is_local Hloc))
          in Hasa.
        discriminate. }
    all: destruct (ap_call_kind ap) eqn:Hkind;
      try (exfalso; apply Hk; first [ reflexivity | exact Hkind ]);
      exact (one_group_never_returns Known_arity_code_pointer _ a _ _ i
               Hread Hno Hi).
Qed.

(* A call that returns to a continuation one of whose parameters has no
   source never returns. *)
Lemma call_never_returns (ctx : option code_id)
    (ce : finite_map (list node)) (ap : apply_expr) (ns : list node)
    (r : node) :
  FlambdaReaperFlowProofs.apply_edges G noted (unit_codes e0) ctx ce ap ->
  conts_counted (FlambdaReaperFlowProofs.counts ce) (Apply ap) = true ->
  FlambdaReaperWrapperProofs.live T ctx -> CNR.returns_ok u e0 T ap ->
  lookup (ap_continuation ap) ce = Some ns -> r ∈ ns ->
  s_has_source S r = false ->
  apply_never_returns F S cds (bound_code_metadata e0) ap = true.
Proof.
  intros (_ & _ & _ & returns & exn & Hret & _ & Hcall) Hcount Hlive
    [Hargs Har] Hk Hr Hno.
  rewrite (Hret ns Hk) in Hcall.
  cbn [conts_counted] in Hcount. apply andb_true_iff in Hcount as [Hcount _].
  unfold count_ok in Hcount.
  rewrite FlambdaReaperWrapperProofs.lookup_counts, Hk in Hcount.
  cbn [option_map] in Hcount. apply Nat.eqb_eq in Hcount.
  apply In_nth_error in Hr as [i Hi].
  assert (Hlt : i < length (unarize (ap_return_arity ap)))
    by (rewrite <- Hcount; apply nth_error_Some; congruence).
  assert (Halias : ∀ w a,
            FlambdaReaperFlowProofs.cond_alias_edge G ctx w a ->
            s_alias S w a = true).
  { intros w a Hc. destruct ctx as [c | ]; cbn in Hc.
    - exact (alias_propagate _ _ _ Hclosed _ _ _ (Hlive c eq_refl) Hc).
    - exact (alias_input _ _ _ Hclosed _ _ Hc). }
  destruct (ap_call_kind ap) eqn:Hkind.
  1, 3:
    (apply FlambdaReaperFlowProofs.known_call_edges_of in Hcall;
       [ | rewrite Hkind; discriminate ]);
    destruct Hcall as (w & a & called & _ & Hres & _ & _ & _ & Hcond & Hread);
    pose proof (Halias _ _ Hcond) as Hwa;
    pose proof (Hres i r Hi) as Hrw;
    (assert (Hasw : s_any_source S w = false)
       by (destruct (s_any_source S w) eqn:E; [ | reflexivity ];
           rewrite (has_source_any _ _ _ Hclosed r
                      (any_source_accessor _ _ _ Hclosed w _ r E Hrw
                         (ret_nl _))) in Hno;
           discriminate));
    (assert (Hasa : s_any_source S a = false)
       by (destruct (s_any_source S a) eqn:E; [ | reflexivity ];
           rewrite (any_source_alias _ _ _ Hclosed a w Hwa E) in Hasw;
           discriminate));
    (assert (Hkey : ∀ s y,
               s_sources S a s = true ->
               (s, Field_return_of_call (Normal i), y) ∈ f_constructor F ->
               s_has_source S y = false)
       by (intros s y Hs Hc; destruct (s_has_source S y) eqn:Ey;
           [ | reflexivity ];
           pose proof (sources_alias _ _ _ Hclosed a w s Hs Hasa Hwa Hasw)
             as Hsw;
           pose proof (alias_accessor_source _ _ _ Hclosed w s _ r y Hrw Hsw
                         Hasw Hc) as Hry;
           rewrite (has_source_alias _ _ _ Hclosed y r Ey Hry) in Hno;
           discriminate));
    (assert (Hcallee :
               WS.callee_result_has_source F T (node_of_simple (ap_callee ap)) i
               = false)
       by (unfold WS.callee_result_has_source, WS.result_has_source_at;
           apply Bool.not_true_iff_false; intros E;
           apply existsb_exists in E as ([[a' f] base] & Ha' & Ht);
           cbn beta iota in Ht;
           apply andb_true_iff in Ht as [Ht Hor];
           apply andb_true_iff in Ht as [Hf Hb];
           apply field_eqb_eq in Hf; apply node_eqb_eq in Hb; subst f base;
           destruct (FlambdaReaperWrapperProofs.widgets_same F T a' a _ Hwid
                       Ha' Hread) as [Hany Hsub];
           apply orb_true_iff in Hor as [Hm | Hm];
           [ change (s_any_source S a' = true) in Hm;
             rewrite Hany, Hasa in Hm; discriminate | ];
           apply existsb_exists in Hm as ([a'' s] & Hs & Ht2);
           cbn beta iota in Ht2;
           apply andb_true_iff in Ht2 as [Ha'' Hc];
           apply node_eqb_eq in Ha''; subst a'';
           apply existsb_exists in Hc as ([[s' f'] y] & Hc & Ht3);
           cbn beta iota in Ht3;
           apply andb_true_iff in Ht3 as [Ht3 Hy];
           apply andb_true_iff in Ht3 as [Hs' Hf'];
           apply node_eqb_eq in Hs'; apply field_eqb_eq in Hf'; subst s' f';
           assert (Hsa : s_sources S a s = true)
             by (apply Hsub; apply (mem_pair_In _ _ _); exact Hs);
           change (s_has_source S y = true) in Hy;
           rewrite (Hkey s y Hsa Hc) in Hy; discriminate));
    exact (known_never_returns ap a i ltac:(rewrite Hkind; discriminate)
             Hread Hcallee Hlt Har).
  (* A call of unknown arity. *)
  specialize (Hargs eq_refl).
  unfold FlambdaReaperFlowProofs.call_edges in Hcall. rewrite Hkind in Hcall.
  destruct Hcall as (ws & a & called & Hlen & Hchain & _ & Hhd & Hread).
  rewrite (CNR.call_groups_length ap Hargs) in Hlen.
  destruct ws as [ | w ws']; [ destruct (ap_args_arity ap); cbn in Hlen;
                               congruence | ].
  pose proof (Halias _ _ (Hhd w eq_refl)) as Hwa.
  unfold FlambdaReaperFlowProofs.call_groups in Hchain.
  pose proof (CNR.call_groups_length ap Hargs) as Hglen.
  unfold FlambdaReaperFlowProofs.call_groups in Hglen.
  destruct (at_least_one_group _) as [ | g gs];
    [ destruct (ap_args_arity ap); cbn in Hglen; congruence | ].
  cbn [combine] in Hchain.
  pose proof (chain_no_any _ _ _ i r Hi Hno _ g w Hchain) as Hw.
  unfold apply_never_returns, code_id_actually_called. rewrite Hkind.
  destruct (ap_callee ap) as [x | sym | c0] eqn:Hcallee; cbv zeta.
  3:{ exfalso. cbn [node_of_simple] in Hread.
      pose proof (any_source_input _ _ _ Hclosed _ Hconsts) as Hac.
      pose proof (any_source_accessor _ _ _ Hclosed _ _ a Hac Hread
                    (wit_nl _)) as Haa.
      rewrite (any_source_alias _ _ _ Hclosed a w Hwa Haa) in Hw.
      discriminate. }
  all: remember (map (fun c => length (unarize_component c))
                   (ap_args_arity ap)) as num eqn:Hnum;
    assert (Hnl : length num = length (ap_args_arity ap))
      by (rewrite Hnum; apply length_map);
    destruct num as [ | n num']; [ destruct (ap_args_arity ap);
                                   cbn in Hnl; congruence | ];
    refine (chain_never_returns _ _ _ i r _ Hi Hno Hlt _ g w num' n _
              _ a Hchain _ _ Hread _ _);
    [ rewrite length_combine; cbn in Hlen, Hglen, Hnl; lia
    | unfold get_direct_sources, CNR.approx;
      match goal with
      | |- context [if s_any_source S ?n then _ else _] =>
        destruct (s_any_source S n) eqn:E; [ reflexivity | ]
      end;
      intros y Hy; apply andb_true_iff in Hy as [Hy _]; exact Hy
    | intros Haa; exact (any_source_alias _ _ _ Hclosed a w Hwa Haa)
    | intros s Hs Hw0;
      assert (Hasa : s_any_source S a = false)
        by (destruct (s_any_source S a) eqn:E; [ | reflexivity ];
            rewrite (any_source_alias _ _ _ Hclosed a w Hwa E) in Hw0;
            discriminate);
      exact (sources_alias _ _ _ Hclosed a w s Hs Hasa Hwa Hw0) ].
Qed.

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
        rewrite (call_never_returns ctx ce ap ns r He Hc Hl Hok Hk Hr Hno)
          in Hnr.
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
      rewrite (ac_no_source ce ac ns r He Hc Hk Hr Hno) in Hex.
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
    + rewrite (ac_no_source ce ac ns r Heac Hcac Hk Hr Hno) in Hex.
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

(* Each code of the unit passes the check of [calls_code] for the code that
   is used and some result of which has no source, if the calls of its body
   to known code expect the results of that code. *)
Lemma unit_codes_checked (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  exn_handlers_well_formed e = true ->
  keys_and_groups_ok e = true ->
  least_solution is_local (traverse_unit Auto u e) T
  ∨ least_solution_local is_local (traverse_unit Auto u e) T ->
  (∀ id c, (id, c) ∈ unit_codes e ->
     ∀ ap, ap ∈ term_calls (code_body c) ->
     ∀ id', code_id_actually_called (traverse_unit Auto u e)
              (solution_of_tables T) ap
            = Some id' ->
     ∃ md, lookup id' (bound_code_metadata e) = Some md
           ∧ ap_return_arity ap = cm_result_arity md) ->
  ∀ id c, (id, c) ∈ unit_codes e ->
    code_checked (traverse_unit Auto u e) (solution_of_tables T)
      (unit_code_deps Auto u e) (bound_code_metadata e)
      (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
      id c.
Proof.
  intros Hloc Hfrag Hewf Hkg Hleast Hknown_calls.
  destruct (least_either_checks is_local u e T Hloc Hleast)
    as (Hclosed & Hwid & Hknown).
  set (F := traverse_unit Auto u e) in *.
  set (S := solution_of_tables T) in *.
  set (cds := unit_code_deps Auto u e) in *.
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
  intros id c Hin.
  pose proof (Hcode_edges id c Hin) as Hce.
  split.
  { destruct c as [params mc ret exn md body].
    cbn [FlambdaReaperFlowProofs.code_edges] in Hce.
    destruct Hce as (_ & _ & _ & _ & _ & _ & _ & _ & _ & _ & Hbs & _).
    exact Hbs. }
  destruct (nr id) eqn:Hnr; [ cbn [negb orb] | reflexivity ].
  apply negb_true_iff.
  refine (code_live_return is_local u e T Hloc Hclosed Hwid Hknown True
            noted deps Hconsts Hnodup Hids Hcode_edges id c Hin _ _ _ Hnr).
  - exact (proj1 FlambdaReaperRebuildProofs.unit_codes_exn_wf e Hewf id c
             Hin).
  - exact (proj1 keys_and_groups_codes e Hkg id c Hin).
  - (* A call of known code expects the results of that code. *)
    intros ap Hap id' cd Hcall Hcd.
    destruct (Hknown_calls id c Hin ap Hap id' Hcall) as (md & Hmd & Hra).
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

(* The fragment without the check gives the fragment with it, for the code
   that is used and some result of which has no source, of a least
   solution with the fix of local fields or without. *)
Lemma unit_checked_either (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cc : node -> bool)
    (unboxed : list (node * unboxed_fields)) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  exn_handlers_well_formed e = true ->
  keys_and_groups_ok e = true ->
  least_solution is_local (traverse_unit Auto u e) T
  ∨ least_solution_local is_local (traverse_unit Auto u e) T ->
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
  destruct (proj1 (unchecked_parts F S cc cds ids codes unboxed nr) e [] []
              Hun)
    as [Hcodes _].
  refine (proj1 (unchecked_checked F S cc cds ids codes unboxed nr) e [] []
            Hun _).
  refine (unit_codes_checked is_local u e T Hloc Hfrag Hewf Hkg Hleast _).
  intros id c Hin ap Hap id' Hcall.
  destruct (unchecked_code_body F S cc cds ids codes unboxed nr id c
              (Hcodes id c Hin))
    as (kinds & C & Hbody).
  destruct (proj2 (proj1 (unchecked_parts F S cc cds ids codes unboxed nr)
                     _ _ _ Hbody) ap Hap)
    as (kinds' & C' & Hok).
  exact (call_ok_known F S cc cds ids codes unboxed nr kinds' C' ap id' Hok
           Hcall).
Qed.

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
  intros Hloc Hfrag Hewf Hkg Hleast.
  exact (unit_checked_either is_local u e T cc unboxed Hloc Hfrag Hewf Hkg
           (or_introl Hleast)).
Qed.

(* The same of the closures fragment. *)
Lemma unit_repr_checked (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cc : node -> bool)
    (unboxed : list (node * unboxed_fields))
    (changed : list (node * closure_repr)) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  exn_handlers_well_formed e = true ->
  keys_and_groups_ok e = true ->
  least_solution is_local (traverse_unit Auto u e) T
  ∨ least_solution_local is_local (traverse_unit Auto u e) T ->
  repr_unchecked_expr (traverse_unit Auto u e) (solution_of_tables T) cc
    (unit_code_deps Auto u e) (static_names code_ids_bound e)
    (bound_code_metadata e) unboxed changed (bound_code_my_closures e)
    (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
    [] [] e
  = true ->
  RCS.repr_expr (traverse_unit Auto u e) (solution_of_tables T) cc
    (unit_code_deps Auto u e) (static_names code_ids_bound e)
    (bound_code_metadata e) unboxed changed (bound_code_my_closures e)
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
  set (mcs := bound_code_my_closures e) in *.
  set (nr := used_never_returns S cds) in *.
  destruct (proj1 (repr_unchecked_parts F S cc cds ids codes unboxed changed
                     mcs nr) e [] [] Hun)
    as [Hcodes _].
  refine (proj1 (repr_unchecked_checked F S cc cds ids codes unboxed changed
                   mcs nr) e [] [] Hun _).
  refine (unit_codes_checked is_local u e T Hloc Hfrag Hewf Hkg Hleast _).
  intros id c Hin ap Hap id' Hcall.
  destruct (repr_unchecked_code_body F S cc cds ids codes unboxed changed mcs
              nr id c (Hcodes id c Hin))
    as (kinds & C & Hbody).
  destruct (proj2 (proj1 (repr_unchecked_parts F S cc cds ids codes unboxed
                            changed mcs nr) _ _ _ Hbody) ap Hap)
    as (C' & Hok).
  exact (repr_call_known F S cc cds ids codes unboxed changed mcs nr C' ap id'
           Hok Hcall).
Qed.

(* [exn_handlers_well_formed] of [well_formed_unit]. *)
Lemma well_formed_exn (u : unit_interface) (e : expr) :
  well_formed_unit u e = true -> exn_handlers_well_formed e = true.
Proof.
  intros Hwf.
  assert (Hue : unit_exn_handlers_well_formed (unit_exn u) e = true).
  { unfold well_formed_unit in Hwf.
    destruct (unit_exn_handlers_well_formed (unit_exn u) e);
      [ reflexivity | ].
    rewrite andb_false_r in Hwf. discriminate Hwf. }
  unfold unit_exn_handlers_well_formed in Hue.
  apply andb_true_iff in Hue as [_ Hewf]. exact Hewf.
Qed.

Theorem never_returns_derived_observably :
  never_returns_derived_observably_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox pdc Hloc Hslots
    Hfrag Hany Hres Hwf Hexits Hkg Hleast Hcheck Hfue Hcc Hwid Hcalls Hreads.
  pose proof (well_formed_exn u e Hwf) as Hewf.
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

Theorem never_returns_derived_local : never_returns_derived_local_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox pdc Hloc Hfrag Hany
    Hres Hwf Hexits Hkg Hleast Hcc Hcalls Hreads.
  pose proof (well_formed_exn u e Hwf) as Hewf.
  destruct (LP.least_local_checks is_local _ T Hleast) as [Hclosed Hlrs].
  apply (RCP.rebuild_unbox_fixed_local_never_returns
           Auto is_local u e T cannot_change dominated to_unbox pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  - exact (LP.least_field_used_exact_local is_local _ T Hleast).
  - exact (proj1 (proj2 (least_either_checks is_local u e T Hloc
                           (or_intror Hleast)))).
  - unfold calls_unboxing_unchecked in Hcalls. unfold calls_unboxing_nr.
    apply andb_true_iff in Hcalls as [Hcalls Huses].
    rewrite Huses, andb_true_r.
    exact (unit_checked_either is_local u e T _ _ Hloc Hfrag Hewf Hkg
             (or_intror Hleast) Hcalls).
Qed.

(* [repr_unboxing_unchecked] gives [repr_unboxing_nr] for the code of
   [used_never_returns]. *)
Lemma repr_unboxing_checked (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node) :
  is_local_slots_only is_local ->
  in_flow_fragment e = true ->
  exn_handlers_well_formed e = true ->
  keys_and_groups_ok e = true ->
  least_solution is_local (traverse_unit Auto u e) T
  ∨ least_solution_local is_local (traverse_unit Auto u e) T ->
  repr_unboxing_unchecked
    (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
    is_local (traverse_unit Auto u e) T cannot_change
    (unit_code_deps Auto u e) dominated to_unbox tcr e
  = true ->
  RCS.repr_unboxing_nr
    (used_never_returns (solution_of_tables T) (unit_code_deps Auto u e))
    is_local (traverse_unit Auto u e) T cannot_change
    (unit_code_deps Auto u e) dominated to_unbox tcr e
  = true.
Proof.
  intros Hloc Hfrag Hewf Hkg Hleast Hcalls.
  unfold repr_unboxing_unchecked in Hcalls. unfold RCS.repr_unboxing_nr.
  apply andb_true_iff in Hcalls as [Hcalls Huses].
  rewrite Huses, andb_true_r.
  exact (unit_repr_checked is_local u e T _ _ _ Hloc Hfrag Hewf Hkg Hleast
           Hcalls).
Qed.

Theorem repr_never_returns_derived_observably :
  repr_never_returns_derived_observably_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc Hloc Hslots
    Hfrag Hany Hres Hwf Hexits Hkg Hleast Hcc Hcalls Hreads.
  pose proof (well_formed_exn u e Hwf) as Hewf.
  apply (RCL.rebuild_repr_fixed_observably_never_returns
           Auto is_local u e T cannot_change dominated to_unbox tcr pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  - exact (proj1 Hleast).
  - exact (LP.least_field_used_exact is_local _ T Hleast).
  - exact (proj1 (proj2 (least_either_checks is_local u e T Hloc
                           (or_introl Hleast)))).
  - exact (repr_unboxing_checked is_local u e T cannot_change dominated
             to_unbox tcr Hloc Hfrag Hewf Hkg (or_introl Hleast) Hcalls).
Qed.

Theorem repr_never_returns_derived_local :
  repr_never_returns_derived_local_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc Hloc Hfrag
    Hany Hres Hwf Hexits Hkg Hleast Hcc Hcalls Hreads.
  pose proof (well_formed_exn u e Hwf) as Hewf.
  destruct (LP.least_local_checks is_local _ T Hleast) as [Hclosed Hlrs].
  apply (RCL.rebuild_repr_fixed_local_never_returns
           Auto is_local u e T cannot_change dominated to_unbox tcr pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  - exact (LP.least_field_used_exact_local is_local _ T Hleast).
  - exact (proj1 (proj2 (least_either_checks is_local u e T Hloc
                           (or_intror Hleast)))).
  - exact (repr_unboxing_checked is_local u e T cannot_change dominated
             to_unbox tcr Hloc Hfrag Hewf Hkg (or_intror Hleast) Hcalls).
Qed.

Theorem repr_never_returns_derived_refines_cc :
  repr_never_returns_derived_refines_cc_statement.
Proof.
  intros is_local u e T cannot_change dominated to_unbox tcr pdc Hloc Hfrag
    Hewf Hkg Hleast Hcalls Hreads.
  apply (RCL.rebuild_repr_fixed_refines_cc_never_returns
           Auto is_local u e T cannot_change dominated to_unbox tcr pdc
           (used_never_returns (solution_of_tables T)
              (unit_code_deps Auto u e)));
    try assumption.
  - exact (proj1 Hleast).
  - exact (repr_unboxing_checked is_local u e T cannot_change dominated
             to_unbox tcr Hloc Hfrag Hewf Hkg (or_introl Hleast) Hcalls).
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

Theorem repr_never_returns_derived_example :
  repr_never_returns_derived_example_statement.
Proof.
  cbv zeta.
  assert (Hleast : least_solution UE.no_local RCS.ReprNeverExample.facts
                     RCS.ReprNeverExample.tables).
  { apply (FlambdaReaperLeastProofs.compute_least_sound _ _ 40).
    vm_compute. reflexivity. }
  assert (Hcalls :
    repr_unboxing_unchecked
      (used_never_returns (solution_of_tables RCS.ReprNeverExample.tables)
         RCS.ReprNeverExample.code_deps)
      UE.no_local RCS.ReprNeverExample.facts RCS.ReprNeverExample.tables []
      RCS.ReprNeverExample.code_deps RCS.ReprNeverExample.dominated
      RCS.ReprNeverExample.to_unbox RCS.ReprNeverExample.tcr
      RCS.ReprNeverExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  assert (Hkg : keys_and_groups_ok RCS.ReprNeverExample.unit_expr = true)
    by (vm_compute; reflexivity).
  split; [ exact Hleast | ].
  split; [ exact Hkg | ].
  split; [ vm_compute; reflexivity | ].
  split; [ exact Hcalls | ].
  apply (repr_never_returns_derived_observably
           UE.no_local UE.interface RCS.ReprNeverExample.unit_expr
           RCS.ReprNeverExample.tables [] RCS.ReprNeverExample.dominated
           RCS.ReprNeverExample.to_unbox RCS.ReprNeverExample.tcr true);
    try assumption; try (vm_compute; reflexivity).
  - intros f Hf. discriminate Hf.
  - left. intros f. reflexivity.
Qed.
