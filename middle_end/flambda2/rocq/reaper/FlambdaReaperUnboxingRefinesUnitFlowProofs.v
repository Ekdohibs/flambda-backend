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

(* The proofs of FlambdaReaperUnboxingRefinesUnitFlowSpec.v. Nothing here
   needs review.

   The checks of the reach fragments are monotone in what they read
   ([calls_mono_f3], [calls_mono_n]): each code that [reach] gives a
   callee, [ovr] or [rem] is asked something, and the rest nothing, so
   that a check with a larger [reach], [ovr] and [rem] implies the check
   with smaller ones. [reach] is compared only at the variables of the
   unit with a use, which is all that a check reads it at (a callee of a
   call whose code is not known is used, by [ind_call_ok_f3] and
   [ind_never_ok]), and at the code of the unit, which is all that the
   checks range over. [used_reach_covered] gives the comparison of the
   reach of the first rebuilt unit's analysis with that of the unit's own
   ([covered_parts]), so that the fragment read off the unit's own
   analysis is in that read off the first rebuilt unit's
   ([unit_flow3_covered], [unit_never_covered]), and the statements of
   FlambdaReaperUnboxingRefinesReach3Spec.v and
   FlambdaReaperUnboxingRefinesNeverSpec.v give the new ones. *)

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
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesOverSpec
  FlambdaReaperUnboxingRefinesOver1Spec
  FlambdaReaperUnboxingRefinesReachSpec FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesOver2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesReach3Proofs
  FlambdaReaperUnboxingRefinesNeverProofs
  FlambdaReaperUnboxingRefinesCallsProofs.
From OxCamlFlambda2 Require FlambdaReaperFlowProofs
  FlambdaReaperCallingConventionsSpec FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The checks are monotone in what they read                               *)
(* ======================================================================== *)

Section Mono.

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
(* The variables at which the two reaches are compared. *)
Variable vars : list variable.
Hypothesis reach_le : ∀ (b : bool) (f : variable) (id : code_id),
  f ∈ vars -> is_var_used sol f Value = true -> id ∈ map fst codes ->
  reach' b f id = true -> reach b f id = true.
Hypothesis ovr_le : ∀ id : code_id,
  id ∈ map fst codes -> ovr' id = true -> ovr id = true.
Hypothesis rem_le : ∀ id : code_id,
  id ∈ map fst codes -> rem' id = true -> rem id = true.

Lemma reach_false (b : bool) (f : variable) (id : code_id) :
  f ∈ vars -> is_var_used sol f Value = true -> id ∈ map fst codes ->
  reach b f id = false -> reach' b f id = false.
Proof.
  intros Hf Hu Hid Hr. destruct (reach' b f id) eqn:E; [ | reflexivity ].
  rewrite (reach_le b f id Hf Hu Hid E) in Hr. exact Hr.
Qed.

Lemma ovr_false (id : code_id) :
  id ∈ map fst codes -> ovr id = false -> ovr' id = false.
Proof.
  intros Hid Ho. destruct (ovr' id) eqn:E; [ | reflexivity ].
  rewrite (ovr_le id Hid E) in Ho. exact Ho.
Qed.

Lemma rem_false (id : code_id) :
  id ∈ map fst codes -> rem id = false -> rem' id = false.
Proof.
  intros Hid Hr. destruct (rem' id) eqn:E; [ | reflexivity ].
  rewrite (rem_le id Hid E) in Hr. exact Hr.
Qed.

Lemma ind_call_mono_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  f ∈ vars ->
  ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach ovr rem kinds C ap f
  = true ->
  ind_call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
    reach' ovr' rem' kinds C ap f
  = true.
Proof.
  unfold ind_call_ok_f3. intros Hf H.
  apply andb_true_iff in H as [Hpre Hm].
  assert (Hu : is_var_used sol f Value = true).
  { pose proof Hpre as Hu.
    do 4 (apply andb_true_iff in Hu as [Hu _]).
    apply andb_true_iff in Hu as [_ Hu]. exact Hu. }
  rewrite Hpre. cbn [andb].
  destruct (lookup (ap_continuation ap) C) as [[kps [ | ]] | ];
    try discriminate Hm.
  apply andb_true_iff in Hm as [Hm Hall]. rewrite Hm. cbn [andb].
  rewrite forallb_forall in Hall |- *. intros [id md] Hin.
  specialize (Hall _ Hin). cbn beta iota in Hall |- *.
  assert (Hid : id ∈ map fst codes) by exact (in_map fst codes (id, md) Hin).
  destruct (never_over codes ap); cbn [andb orb negb] in Hall |- *.
  - destruct (ind_code_ok_o1 sol cannot_change code_deps unit_code_ids
                unboxed true kps (ap_return_arity ap) id md);
      [ apply orb_true_r | ].
    rewrite orb_false_r in Hall |- *. rewrite orb_false_r in Hall |- *.
    apply negb_true_iff in Hall. apply negb_true_iff.
    exact (reach_false _ f id Hf Hu Hid Hall).
  - apply orb_true_iff in Hall as [Hn | Ho].
    + apply negb_true_iff, orb_false_iff in Hn as [Hr Hrm].
      rewrite (reach_false false f id Hf Hu Hid Hr),
        (rem_false id Hid Hrm).
      reflexivity.
    + apply orb_true_iff. right.
      destruct (ovr id) eqn:Eo.
      * exact (ind_code_ok_o1_or _ _ _ _ _ false (negb (ovr' id)) kps
                 (ap_return_arity ap) id md Ho).
      * rewrite (ovr_false id Hid Eo). exact Ho.
Qed.

Lemma call_mono_f3 (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  (∀ f : variable, ap_callee ap = Var f -> f ∈ vars) ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr rem kinds C ap
  = true ->
  call_ok_f3 F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' ovr' rem' kinds C ap
  = true.
Proof.
  intros Hv. unfold call_ok_f3.
  destruct (ap_callee ap) as [f | sym | cst] eqn:Hc;
    [ | exact (fun H => H) | exact (fun H => H) ].
  destruct (code_id_actually_called F sol ap); [ exact (fun H => H) | ].
  apply ind_call_mono_f3. exact (Hv f eq_refl).
Qed.

Lemma ind_never_mono (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) (f : variable) :
  f ∈ vars ->
  ind_never_ok F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach kinds C ap f
  = true ->
  ind_never_ok F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' kinds C ap f
  = true.
Proof.
  unfold ind_never_ok. intros Hf H.
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
    rewrite (reach_false _ f id Hf Hu Hid Hn). reflexivity.
  - rewrite Ho. apply orb_true_r.
Qed.

Lemma call_mono_n (kinds : finite_map flambda_kind)
    (C : finite_map (list pdesc * bool)) (ap : apply_expr) :
  (∀ f : variable, ap_callee ap = Var f -> f ∈ vars) ->
  call_ok_n F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach ovr rem kinds C ap
  = true ->
  call_ok_n F sol cannot_change code_deps unit_code_ids codes unboxed nr
    reach' ovr' rem' kinds C ap
  = true.
Proof.
  intros Hv. unfold call_ok_n.
  destruct (ap_callee ap) as [f | sym | cst] eqn:Hc;
    [ | apply call_mono_f3; intros f0 Hf0; rewrite Hc in Hf0;
        discriminate Hf0
      | apply call_mono_f3; intros f0 Hf0; rewrite Hc in Hf0;
        discriminate Hf0 ].
  destruct (code_id_actually_called F sol ap) eqn:Hid;
    [ apply call_mono_f3; intros f0 Hf0; rewrite Hc in Hf0;
      injection Hf0 as <-; exact (Hv f eq_refl) | ].
  unfold ind_call_ok_n.
  destruct (apply_never_returns F sol code_deps codes ap).
  - apply ind_never_mono. exact (Hv f eq_refl).
  - apply ind_call_mono_f3. exact (Hv f eq_refl).
Qed.

(* The variables of the parts of a term are among those of the term. *)
Lemma vars_let (x : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  incl (variables_of (Let x kind d body)) vars ->
  incl (variables_of body) vars.
Proof.
  intros Hv z Hz. apply Hv. cbn [variables_of]. right. apply in_or_app.
  right. exact Hz.
Qed.

Lemma vars_static_body (bindings : list static_binding) (body : expr) :
  incl (variables_of (Let_static bindings body)) vars ->
  incl (variables_of body) vars.
Proof.
  intros Hv z Hz. apply Hv. cbn [variables_of]. apply in_or_app.
  right. exact Hz.
Qed.

Lemma vars_static_code (bindings : list static_binding) (body : expr)
    (id : code_id) (c : code) :
  incl (variables_of (Let_static bindings body)) vars ->
  Bind_code id c ∈ bindings ->
  incl (variables_of (code_body c)) vars.
Proof.
  intros Hv Hin z Hz. apply Hv. cbn [variables_of]. apply in_or_app. left.
  apply in_flat_map. exists (Bind_code id c). split; [ exact Hin | ].
  destruct c as [params mc ret exn md body']. cbn [variables_in_binding
    variables_in_code code_body] in Hz |- *.
  right. apply in_or_app. right. exact Hz.
Qed.

Lemma vars_cont (k : continuation) (h : continuation_handler) (body : expr) :
  incl (variables_of (Let_cont_non_recursive k h body)) vars ->
  incl (variables_in_handler h) vars ∧ incl (variables_of body) vars.
Proof.
  intros Hv. split; intros z Hz; apply Hv; cbn [variables_of];
    apply in_or_app; [ left | right ]; exact Hz.
Qed.

Lemma vars_cont_rec (handlers : list (continuation * continuation_handler))
    (body : expr) :
  incl (variables_of (Let_cont_recursive handlers body)) vars ->
  (∀ k h, (k, h) ∈ handlers -> incl (variables_in_handler h) vars)
  ∧ incl (variables_of body) vars.
Proof.
  intros Hv. split.
  - intros k h Hin z Hz. apply Hv. cbn [variables_of]. apply in_or_app.
    left. apply in_flat_map. exists (k, h). split; [ exact Hin | exact Hz ].
  - intros z Hz. apply Hv. cbn [variables_of]. apply in_or_app. right.
    exact Hz.
Qed.

Lemma vars_callee (ap : apply_expr) :
  incl (variables_of (Apply ap)) vars ->
  ∀ f : variable, ap_callee ap = Var f -> f ∈ vars.
Proof.
  intros Hv f Hf. apply Hv. cbn [variables_of]. unfold variables_in_apply.
  rewrite Hf. left. reflexivity.
Qed.

Lemma vars_handler (params : list variable) (md : handler_metadata)
    (handler : expr) :
  incl (variables_in_handler (Handler params md handler)) vars ->
  incl (variables_of handler) vars.
Proof.
  intros Hv z Hz. apply Hv. cbn [variables_in_handler]. apply in_or_app.
  right. exact Hz.
Qed.

Lemma calls_mono_f3 :
  (∀ e : expr, ∀ kinds C,
     incl (variables_of e) vars ->
     calls_expr_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach ovr rem kinds C e
     = true ->
     calls_expr_f3 F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach' ovr' rem' kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       incl (variables_in_handler h) vars ->
       calls_handler_f3 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach ovr rem kinds C ps h
       = true ->
       calls_handler_f3 F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach' ovr' rem' kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds C Hv H. cbn [calls_expr_f3] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_let x kind d body Hv) H2).
  - intros bindings body IHc IH kinds C Hv H. cbn [calls_expr_f3] in H |- *.
    apply andb_true_iff in H as [Hbs H].
    rewrite (IH _ _ (vars_static_body bindings body Hv) H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    pose proof (vars_static_code bindings body id c Hv Hin) as Hvc.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_f3 calls_code_f3 code_body] in Hbs, IHb, Hvc |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hvc Hb).
  - intros k h body IHh IH kinds C Hv H. cbn [calls_expr_f3] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont k h body Hv) as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff.
    split; [ exact (IHh _ _ _ Hvh Hh) | exact (IH _ _ Hvb H) ].
  - intros handlers body IHh IH kinds C Hv H. cbn [calls_expr_f3] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont_rec handlers body Hv)
      as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ Hvb H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ (Hvh k h Hin) Hh).
  - intros ap kinds C Hv H. cbn [calls_expr_f3] in H |- *.
    exact (call_mono_f3 kinds C ap (vars_callee ap Hv) H).
  - intros ac kinds C _ H. exact H.
  - intros scrutinee arms kinds C _ H. exact H.
  - intros kinds C _ H. exact H.
  - intros params metadata handler IH kinds C ps Hv H.
    cbn [calls_handler_f3] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_handler params metadata handler Hv) H2).
Qed.

Lemma calls_mono_n :
  (∀ e : expr, ∀ kinds C,
     incl (variables_of e) vars ->
     calls_expr_n F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach ovr rem kinds C e
     = true ->
     calls_expr_n F sol cannot_change code_deps unit_code_ids codes unboxed
       nr reach' ovr' rem' kinds C e
     = true)
  ∧ (∀ h : continuation_handler, ∀ kinds C ps,
       incl (variables_in_handler h) vars ->
       calls_handler_n F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach ovr rem kinds C ps h
       = true ->
       calls_handler_n F sol cannot_change code_deps unit_code_ids codes
         unboxed nr reach' ovr' rem' kinds C ps h
       = true).
Proof.
  apply FlambdaReaperFlowProofs.expr_handler_ind.
  - intros x kind d body IH kinds C Hv H. cbn [calls_expr_n] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_let x kind d body Hv) H2).
  - intros bindings body IHc IH kinds C Hv H. cbn [calls_expr_n] in H |- *.
    apply andb_true_iff in H as [Hbs H].
    rewrite (IH _ _ (vars_static_body bindings body Hv) H), andb_true_r.
    rewrite forallb_forall in Hbs |- *. intros b Hin. specialize (Hbs b Hin).
    destruct b as [id c | id | s const]; [ | exact Hbs | exact Hbs ].
    pose proof (IHc id c Hin) as IHb.
    pose proof (vars_static_code bindings body id c Hv Hin) as Hvc.
    destruct c as [params mc ret exn md body'].
    cbn [calls_binding_n calls_code_n code_body] in Hbs, IHb, Hvc |- *.
    apply andb_true_iff in Hbs as [Hbs Hlive]. rewrite Hlive, andb_true_r.
    apply andb_true_iff in Hbs as [Hmd Hbs]. rewrite Hmd. cbn [andb].
    destruct (code_pdescs _ _ _ _ _ _ _) as [[pds rds] | ];
      [ | discriminate Hbs ].
    destruct (code_exn_decision _ _ _ _) as [d | ];
      [ | rewrite !andb_false_r in Hbs; discriminate Hbs ].
    apply andb_true_iff in Hbs as [Hpre Hb]. rewrite Hpre.
    exact (IHb _ _ Hvc Hb).
  - intros k h body IHh IH kinds C Hv H. cbn [calls_expr_n] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont k h body Hv) as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff.
    split; [ exact (IHh _ _ _ Hvh Hh) | exact (IH _ _ Hvb H) ].
  - intros handlers body IHh IH kinds C Hv H. cbn [calls_expr_n] in H |- *.
    cbv zeta in H |- *. destruct (vars_cont_rec handlers body Hv)
      as [Hvh Hvb].
    apply andb_true_iff in H as [Hh H].
    apply andb_true_iff. split; [ | exact (IH _ _ Hvb H) ].
    rewrite forallb_forall in Hh |- *. intros [k h] Hin.
    specialize (Hh _ Hin). cbn beta iota in Hh |- *.
    exact (IHh k h Hin _ _ _ (Hvh k h Hin) Hh).
  - intros ap kinds C Hv H. cbn [calls_expr_n] in H |- *.
    exact (call_mono_n kinds C ap (vars_callee ap Hv) H).
  - intros ac kinds C _ H. exact H.
  - intros scrutinee arms kinds C _ H. exact H.
  - intros kinds C _ H. exact H.
  - intros params metadata handler IH kinds C ps Hv H.
    cbn [calls_handler_n] in H |- *.
    apply andb_true_iff in H as [H1 H2]. rewrite H1.
    exact (IH _ _ (vars_handler params metadata handler Hv) H2).
Qed.

End Mono.

(* ======================================================================== *)
(* The fragments read off the unit's own analysis                          *)
(* ======================================================================== *)

(* What [used_reach_covered] gives, for the hypotheses of [Mono]. *)
Lemma covered_parts (F : facts) (T : tables) (F1 : facts) (T1 : tables)
    (e : expr) :
  used_reach_covered F T F1 T1 e = true ->
  (∀ (b : bool) (f : variable) (id : code_id),
     f ∈ variables_of e ->
     is_var_used (solution_of_tables T) f Value = true ->
     id ∈ map fst (bound_code_metadata e) ->
     flow_reach2 F1 T1 b f id = true -> flow_reach2 F T b f id = true)
  ∧ (∀ id : code_id,
       id ∈ map fst (bound_code_metadata e) ->
       over_reach F1 T1 id = true -> over_reach F T id = true)
  ∧ (∀ id : code_id,
       id ∈ map fst (bound_code_metadata e) ->
       rem_reach F1 T1 id = true -> rem_reach F T id = true).
Proof.
  unfold used_reach_covered. cbv zeta. intros H.
  apply andb_true_iff in H as [Hv Hids].
  rewrite forallb_forall in Hv, Hids.
  split; [ | split ].
  - intros b f id Hf Hu Hid Hr1.
    specialize (Hv f Hf). cbn [is_var_used] in Hu. rewrite Hu in Hv.
    cbn [negb orb] in Hv. rewrite forallb_forall in Hv.
    specialize (Hv id Hid). cbn [forallb] in Hv. rewrite andb_true_r in Hv.
    apply andb_true_iff in Hv as [Ht Hf'].
    destruct b; [ rewrite Hr1 in Ht; exact Ht | rewrite Hr1 in Hf';
                                                exact Hf' ].
  - intros id Hid Ho1. specialize (Hids id Hid).
    apply andb_true_iff in Hids as [Ho _]. rewrite Ho1 in Ho. exact Ho.
  - intros id Hid Hm1. specialize (Hids id Hid).
    apply andb_true_iff in Hids as [_ Hm]. rewrite Hm1 in Hm. exact Hm.
Qed.

(* The fragment of FlambdaReaperUnboxingRefinesReach3Spec.v read off the
   unit's own analysis is in that read off the first rebuilt unit's, where
   the reach of the latter is within that of the former. *)
Lemma unit_flow3_covered (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) :
  calls_unboxing_unit_flow3_nr nr F T cannot_change code_deps dominated
    to_unbox e
  = true ->
  used_reach_covered F T F1 T1 e = true ->
  calls_unboxing_flow3_nr nr F T cannot_change code_deps dominated to_unbox
    F1 T1 e
  = true.
Proof.
  unfold calls_unboxing_unit_flow3_nr, calls_unboxing_flow3_nr,
    calls_unboxing_reach_rem_nr.
  cbv zeta. intros H Hcov.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  destruct (covered_parts F T F1 T1 e Hcov) as (Hr & Ho & Hm).
  exact (proj1 (calls_mono_f3 F (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps
                  (static_names code_ids_bound e) (bound_code_metadata e)
                  (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                  (flow_reach2 F T) (flow_reach2 F1 T1) (over_reach F T)
                  (over_reach F1 T1) (rem_reach F T) (rem_reach F1 T1)
                  (variables_of e) Hr Ho Hm)
           e [] [] (incl_refl _) H1).
Qed.

(* The same for the fragment of FlambdaReaperUnboxingRefinesNeverSpec.v. *)
Lemma unit_never_covered (nr : code_id -> bool) (F : facts) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node) (F1 : facts)
    (T1 : tables) (e : expr) :
  calls_unboxing_unit_never_nr nr F T cannot_change code_deps dominated
    to_unbox e
  = true ->
  used_reach_covered F T F1 T1 e = true ->
  calls_unboxing_never_nr nr F T cannot_change code_deps dominated to_unbox
    F1 T1 e
  = true.
Proof.
  unfold calls_unboxing_unit_never_nr, calls_unboxing_never_nr,
    calls_unboxing_reach_never_nr.
  cbv zeta. intros H Hcov.
  apply andb_true_iff in H as [H1 H2]. rewrite H2, andb_true_r.
  destruct (covered_parts F T F1 T1 e Hcov) as (Hr & Ho & Hm).
  exact (proj1 (calls_mono_n F (solution_of_tables T)
                  (fun n => mem_node n cannot_change) code_deps
                  (static_names code_ids_bound e) (bound_code_metadata e)
                  (fst (unit_unboxed_fields F T dominated to_unbox e)) nr
                  (flow_reach2 F T) (flow_reach2 F1 T1) (over_reach F T)
                  (over_reach F1 T1) (rem_reach F T) (rem_reach F1 T1)
                  (variables_of e) Hr Ho Hm)
           e [] [] (incl_refl _) H1).
Qed.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Theorem rebuild_unbox_unit_flow3_refines_cc :
  rebuild_unbox_unit_flow3_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_flow3_refines_cc flag is_local u e T T1 cannot_change
           dominated to_unbox pdc nr Hslots Hfrag Hclosed Hfrag1 Hclosed1
           Hlrs1 (unit_flow3_covered nr _ _ _ _ _ _ _ _ e Hcalls Hcov)
           Hcheck).
Qed.

Theorem rebuild_unbox_unit_flow3_refines_observably_local :
  rebuild_unbox_unit_flow3_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_flow3_refines_observably_local flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hfrag Hany Hrk Hwf
           Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_flow3_covered nr _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_flow3_refines_observably :
  rebuild_unbox_unit_flow3_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc nr
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_flow3_refines_observably flag is_local u e T T1
           cannot_change dominated to_unbox pdc nr Hslots Hproj Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_flow3_covered nr _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_refines_cc :
  rebuild_unbox_unit_never_refines_cc_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hclosed Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_refines_cc flag is_local u e T T1 cannot_change
           dominated to_unbox pdc _ Hslots Hfrag Hclosed Hfrag1 Hclosed1
           Hlrs1 (unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov)
           Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_refines_observably_local :
  rebuild_unbox_unit_never_refines_observably_local_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hfrag Hany Hrk Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_refines_observably_local flag is_local u e T
           T1 cannot_change dominated to_unbox pdc _ Hslots Hfrag Hany Hrk
           Hwf Hone Hclosed Hlrs Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

Theorem rebuild_unbox_unit_never_refines_observably :
  rebuild_unbox_unit_never_refines_observably_statement.
Proof.
  intros flag is_local u e T T1 cannot_change dominated to_unbox pdc
    Hslots Hproj Hfrag Hany Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1
    Hclosed1 Hlrs1 Hcalls Hcov Hcheck.
  exact (rebuild_unbox_nr_ind_refines_observably flag is_local u e T T1
           cannot_change dominated to_unbox pdc _ Hslots Hproj Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1
           (unit_never_covered _ _ _ _ _ _ _ _ _ e Hcalls Hcov) Hcheck).
Qed.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module UE := FlambdaReaperUnboxing.Examples.

Theorem unit_flow_examples : unit_flow_examples_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  vm_compute; reflexivity.
Qed.

Theorem unit_flow3_example : unit_flow3_example_statement.
Proof.
  assert (Hprem :
    in_flow_fragment Flow3Example.unit_expr = true
    ∧ any_kind_arguments Flow3Example.unit_expr = true
    ∧ results_kinded [] [] Flow3Example.unit_expr = true
    ∧ well_formed_unit UE.interface Flow3Example.unit_expr = true
    ∧ exits_one_value UE.interface Flow3Example.unit_expr = true
    ∧ check_closed UE.no_local Flow3Example.facts Flow3Example.tables = true
    ∧ check_field_used_exact UE.no_local Flow3Example.facts
        Flow3Example.tables
      = true
    ∧ check_cc_closed Flow3Example.facts Flow3Example.tables
        Flow3Example.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        Flow3Example.facts Flow3Example.tables
      = true
    ∧ in_heap_fragment Flow3Example.first = true
    ∧ check_closed UE.no_local Flow3Example.facts1 Flow3Example.tables1
      = true
    ∧ check_local_reads_used UE.no_local Flow3Example.facts1
        Flow3Example.tables1
      = true
    ∧ check_unboxed_reads Flow3Example.facts Flow3Example.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  (* The checks as the statement has them. *)
  assert (Hcalls :
    calls_unboxing_unit_flow3_nr (fun _ => false)
      (traverse_unit Auto UE.interface Flow3Example.unit_expr)
      Flow3Example.tables Flow3Example.cannot_change
      (unit_code_deps Auto UE.interface Flow3Example.unit_expr) [] []
      Flow3Example.unit_expr
    = true)
    by (vm_compute; reflexivity).
  assert (Hcov :
    used_reach_covered
      (traverse_unit Auto UE.interface Flow3Example.unit_expr)
      Flow3Example.tables
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface Flow3Example.unit_expr
            Flow3Example.tables Flow3Example.cannot_change true))
      Flow3Example.tables1 Flow3Example.unit_expr
    = true)
    by (vm_compute; reflexivity).
  exact (rebuild_unbox_unit_flow3_refines_observably Auto UE.no_local
           UE.interface Flow3Example.unit_expr Flow3Example.tables
           Flow3Example.tables1 Flow3Example.cannot_change [] [] true
           (fun _ => false) Hslots (or_introl (fun _ => eq_refl)) Hfrag Hany
           Hrk Hwf Hone Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls
           Hcov Hreads).
Qed.

Theorem unit_never_example : unit_never_example_statement.
Proof.
  assert (Hprem :
    in_flow_fragment NeverIndExample.unit_expr = true
    ∧ any_kind_arguments NeverIndExample.unit_expr = true
    ∧ results_kinded [] [] NeverIndExample.unit_expr = true
    ∧ well_formed_unit UE.interface NeverIndExample.unit_expr = true
    ∧ exits_one_value UE.interface NeverIndExample.unit_expr = true
    ∧ check_closed UE.no_local NeverIndExample.facts NeverIndExample.tables
      = true
    ∧ check_field_used_exact UE.no_local NeverIndExample.facts
        NeverIndExample.tables
      = true
    ∧ check_cc_closed NeverIndExample.facts NeverIndExample.tables
        NeverIndExample.cannot_change
      = true
    ∧ FlambdaReaperCallingConventionsSpec.check_apply_widgets
        NeverIndExample.facts NeverIndExample.tables
      = true
    ∧ in_heap_fragment NeverIndExample.first = true
    ∧ check_closed UE.no_local NeverIndExample.facts1 NeverIndExample.tables1
      = true
    ∧ check_local_reads_used UE.no_local NeverIndExample.facts1
        NeverIndExample.tables1
      = true
    ∧ check_unboxed_reads NeverIndExample.facts NeverIndExample.tables [] []
      = true)
    by (vm_compute; repeat split).
  destruct Hprem as (Hfrag & Hany & Hrk & Hwf & Hone & Hclosed & Hexact
                     & Hccc & Hwid & Hfrag1 & Hclosed1 & Hlrs1 & Hreads).
  assert (Hslots : is_local_slots_only UE.no_local)
    by (intros f Hf; discriminate Hf).
  (* The checks as the statement has them. *)
  assert (Hcalls :
    calls_unboxing_unit_never_nr
      (FlambdaReaperNeverReturnsSpec.used_never_returns
         (solution_of_tables NeverIndExample.tables)
         (unit_code_deps Auto UE.interface NeverIndExample.unit_expr))
      (traverse_unit Auto UE.interface NeverIndExample.unit_expr)
      NeverIndExample.tables NeverIndExample.cannot_change
      (unit_code_deps Auto UE.interface NeverIndExample.unit_expr) [] []
      NeverIndExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  assert (Hcov :
    used_reach_covered
      (traverse_unit Auto UE.interface NeverIndExample.unit_expr)
      NeverIndExample.tables
      (traverse_unit Auto UE.interface
         (first_rebuild Auto UE.interface NeverIndExample.unit_expr
            NeverIndExample.tables NeverIndExample.cannot_change true))
      NeverIndExample.tables1 NeverIndExample.unit_expr
    = true)
    by (vm_compute; reflexivity).
  exact (rebuild_unbox_unit_never_refines_observably Auto UE.no_local
           UE.interface NeverIndExample.unit_expr NeverIndExample.tables
           NeverIndExample.tables1 NeverIndExample.cannot_change [] [] true
           Hslots (or_introl (fun _ => eq_refl)) Hfrag Hany Hrk Hwf Hone
           Hclosed Hexact Hccc Hwid Hfrag1 Hclosed1 Hlrs1 Hcalls Hcov Hreads).
Qed.

Theorem poison_reach : poison_reach_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  vm_compute; reflexivity.
Qed.
