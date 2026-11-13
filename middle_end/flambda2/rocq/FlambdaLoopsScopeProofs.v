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

(* Proofs of the statements of FlambdaLoopsScopeSpec.v. Nothing here
   needs review. The relation between a term and one that keeps its
   scope, with its lemmas, is in FlambdaScopedProofs.v; what is known
   of the following of aliases is in FlambdaKeepsScopeProofs.v. *)

From Stdlib Require Import ZArith List Bool Arith Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaUnusedContsSpec
  FlambdaUnusedContsProofs FlambdaDeadParamsSpec FlambdaForwardSpec
  FlambdaForwardArmSpec
  FlambdaLoopAliasesSpec FlambdaLoopLetsSpec FlambdaLoopifySpec
  FlambdaParamAliasesSpec FlambdaRecHandlersSpec FlambdaShortcutSpec
  FlambdaSameHandlersSpec
  FlambdaSinkContsSpec FlambdaSinkUsesSpec FlambdaUnusedParamsSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeProofs FlambdaLoopsScopeSpec.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
Import ListNotations.

(* Moving a continuation to the handler that alone uses it. *)

(* A [Let_cont] moved past a [Let], past a group of static bindings,
   into the handler of the next [Let_cont], and past that one. *)
Lemma scoped_let_cont_past_let (k : continuation) (h : continuation_handler)
    (x : variable) (kind : flambda_kind) (d : named) (body sunk : expr) :
  scoped (Let_cont_non_recursive k h body) sunk ->
  scoped (Let_cont_non_recursive k h (Let x kind d body))
    (Let x kind d sunk).
Proof.
  intros Hsunk Hclosed. scope_in Hclosed. scope.
  destruct (Hsunk Hclosed) as (Hclosed' & Hv & Hk). scope_in Hv. scope_in Hk.
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ right; apply in_or_app; left; exact Hy | ].
  apply in_without in Hy. destruct Hy as [Hy Hx].
  apply Hv, in_app_or in Hy.
  destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
  apply in_or_app. right. apply in_without. split; assumption.
Qed.

Lemma scoped_let_cont_past_let_static (k : continuation)
    (h : continuation_handler) (bindings : list static_binding)
    (body sunk : expr) :
  scoped (Let_cont_non_recursive k h body) sunk ->
  scoped (Let_cont_non_recursive k h (Let_static bindings body))
    (Let_static bindings sunk).
Proof.
  intros Hsunk Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hclosed].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hbody].
  assert (Hboth : code_closed (Let_cont_non_recursive k h body) = true)
    by (scope; rewrite Hh, Hbody; reflexivity).
  destruct (Hsunk Hboth) as (Hclosed' & Hv & Hk). scope_in Hv. scope_in Hk.
  rewrite Hgroup, Hclosed'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ right; apply in_or_app; left; exact Hy | ].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_or_app. right. exact Hy.
  - rewrite (bindings_closed_no_continuations bindings Hgroup). exact Hk.
Qed.

Lemma scoped_let_cont_into_handler (k k' : continuation)
    (h : continuation_handler) (params : list variable)
    (metadata : handler_metadata) (handler body sunk : expr) :
  cont_free_b k body = false ->
  scoped (Let_cont_non_recursive k h handler) sunk ->
  scoped
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k' (Handler params metadata handler) body))
    (Let_cont_non_recursive k' (Handler params metadata sunk) body).
Proof.
  intros Hunused Hsunk Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hclosed].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hhandler Hbody].
  assert (Hboth : code_closed (Let_cont_non_recursive k h handler) = true)
    by (scope; rewrite Hh, Hhandler; reflexivity).
  destruct (Hsunk Hboth) as (Hclosed' & Hv & Hk). scope_in Hv. scope_in Hk.
  rewrite Hclosed', Hbody. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy];
      [ | right; apply in_or_app; right; exact Hy ].
    apply in_without in Hy. destruct Hy as [Hy Hparams].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_or_app. left. apply in_without. split; assumption.
  - intros j Hj. apply in_app_or in Hj. apply in_or_app.
    destruct Hj as [Hj | Hj].
    + apply Hk, in_app_or in Hj.
      destruct Hj as [Hj | Hj]; [ left; exact Hj | right ].
      apply in_without in Hj. destruct Hj as [Hj Hout].
      apply in_without.
      split; [ apply in_or_app; left; exact Hj | exact Hout ].
    + right. apply in_without.
      split; [ apply in_or_app; right; exact Hj | ].
      apply in_without in Hj. destruct Hj as [Hj _].
      intros [<- | []].
      rewrite (cont_free_b_free k body Hbody Hj) in Hunused. discriminate.
Qed.

Lemma scoped_let_cont_past_let_cont (k k' : continuation)
    (h : continuation_handler) (params : list variable)
    (metadata : handler_metadata) (handler body sunk : expr) :
  cont_free_b k handler = false ->
  scoped (Let_cont_non_recursive k h body) sunk ->
  scoped
    (Let_cont_non_recursive k h
       (Let_cont_non_recursive k' (Handler params metadata handler) body))
    (Let_cont_non_recursive k' (Handler params metadata handler) sunk).
Proof.
  intros Hunused Hsunk Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hh Hclosed].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hhandler Hbody].
  assert (Hboth : code_closed (Let_cont_non_recursive k h body) = true)
    by (scope; rewrite Hh, Hbody; reflexivity).
  destruct (Hsunk Hboth) as (Hclosed' & Hv & Hk). scope_in Hv. scope_in Hk.
  rewrite Hhandler, Hclosed'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy];
      [ right; apply in_or_app; left; exact Hy | ].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_or_app. right. exact Hy.
  - intros j Hj. apply in_app_or in Hj. apply in_or_app.
    destruct Hj as [Hj | Hj].
    + right. apply in_without.
      split; [ apply in_or_app; left; exact Hj | ].
      intros [<- | []].
      rewrite (cont_free_b_free k handler Hhandler Hj) in Hunused.
      discriminate.
    + apply in_without in Hj. destruct Hj as [Hj Hout'].
      apply Hk, in_app_or in Hj.
      destruct Hj as [Hj | Hj]; [ left; exact Hj | right ].
      apply in_without in Hj. destruct Hj as [Hj Hout].
      apply in_without. split; [ | exact Hout ].
      apply in_or_app. right. apply in_without. split; assumption.
Qed.

Fixpoint sink_into_handler_scoped (k : continuation)
    (h : continuation_handler) (body : expr) {struct body} :
  ∀ sunk : expr,
    sink_into_handler k h body = Some sunk ->
    scoped (Let_cont_non_recursive k h body) sunk.
Proof.
  destruct body
    as [x kind d body | bindings body | k' [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [sink_into_handler]; intros sunk;
    try (intros Hsunk; discriminate Hsunk).
  - destruct (var_free_in_handler_b x h);
      [ intros Hsunk; discriminate Hsunk | ].
    destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hbody;
      cbn [option_map]; intros Hsunk; [ | discriminate Hsunk ].
    injection Hsunk as <-.
    apply scoped_let_cont_past_let, sink_into_handler_scoped, Hbody.
  - destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hbody;
      cbn [option_map]; intros Hsunk; [ | discriminate Hsunk ].
    injection Hsunk as <-.
    apply scoped_let_cont_past_let_static, sink_into_handler_scoped, Hbody.
  - match goal with
    | |- (if ?test then _ else _) = _ -> _ => destruct test eqn:Hinto
    end.
    + apply andb_true_iff in Hinto. destruct Hinto as [Hinto _].
      apply andb_true_iff in Hinto. destruct Hinto as [Hunused _].
      apply negb_true_iff in Hunused.
      destruct (sink_into_handler k h handler) as [sunk' | ] eqn:Hhandler.
      * intros Hsunk. injection Hsunk as <-.
        apply scoped_let_cont_into_handler;
          [ exact Hunused | apply sink_into_handler_scoped, Hhandler ].
      * destruct (applied_at_tail k h handler); intros Hsunk;
          [ | discriminate Hsunk ].
        injection Hsunk as <-.
        apply scoped_let_cont_into_handler;
          [ exact Hunused | apply scoped_refl ].
    + match goal with
      | |- (if ?test then _ else _) = _ -> _ =>
        destruct test eqn:Hpast; [ | intros Hsunk; discriminate Hsunk ]
      end.
      apply andb_true_iff in Hpast. destruct Hpast as [Hpast _].
      apply andb_true_iff in Hpast. destruct Hpast as [_ Hunused].
      apply negb_true_iff in Hunused.
      destruct (sink_into_handler k h body) as [sunk' | ] eqn:Hbody;
        cbn [option_map]; intros Hsunk; [ | discriminate Hsunk ].
      injection Hsunk as <-.
      apply scoped_let_cont_past_let_cont;
        [ exact Hunused | apply sink_into_handler_scoped, Hbody ].
Qed.

Fixpoint sink_let_conts_scoped (e : expr) : scoped e (sink_let_conts e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [sink_let_conts]; try (scoped_under_binders sink_let_conts_scoped).
  cbv zeta.
  destruct
    (sink_into_handler k (Handler params metadata (sink_let_conts handler))
       (sink_let_conts body))
    as [sunk | ] eqn:Hsunk.
  - eapply scoped_trans;
      [ | exact (sink_into_handler_scoped _ _ _ _ Hsunk) ].
    scoped_under_binders sink_let_conts_scoped.
  - scoped_under_binders sink_let_conts_scoped.
Qed.

Theorem sink_let_conts_keeps_scope : sink_let_conts_keeps_scope_statement.
Proof. apply keeps_scope_intro, sink_let_conts_scoped. Qed.

(* A continuation moved down past the bindings under it, or into the
   handler under it: the moves are those above. *)
Fixpoint sink_to_use_scoped (k : continuation)
    (h : continuation_handler) (body : expr) {struct body} :
  scoped (Let_cont_non_recursive k h body) (sink_to_use k h body).
Proof.
  destruct body
    as [x kind d body | bindings body | k' [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [sink_to_use]; try apply scoped_refl.
  - destruct (var_free_in_handler_b x h); [ apply scoped_refl | ].
    apply scoped_let_cont_past_let, sink_to_use_scoped.
  - apply scoped_let_cont_past_let_static, sink_to_use_scoped.
  - match goal with
    | |- scoped _ (if ?test then _ else _) => destruct test eqn:Hpast
    end.
    + apply andb_true_iff in Hpast. destruct Hpast as [Hpast _].
      apply andb_true_iff in Hpast. destruct Hpast as [_ Hunused].
      apply negb_true_iff in Hunused.
      apply scoped_let_cont_past_let_cont;
        [ exact Hunused | apply sink_to_use_scoped ].
    + match goal with
      | |- scoped _ (if ?test then _ else _) =>
        destruct test eqn:Hinto; [ | apply scoped_refl ]
      end.
      apply andb_true_iff in Hinto. destruct Hinto as [Hinto _].
      apply andb_true_iff in Hinto. destruct Hinto as [Hunused _].
      apply negb_true_iff in Hunused.
      apply scoped_let_cont_into_handler;
        [ exact Hunused | apply sink_to_use_scoped ].
Qed.

Fixpoint sink_conts_to_uses_scoped (e : expr) :
  scoped e (sink_conts_to_uses e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [sink_conts_to_uses];
    try (scoped_under_binders sink_conts_to_uses_scoped).
  eapply scoped_trans; [ | apply sink_to_use_scoped ].
  scoped_under_binders sink_conts_to_uses_scoped.
Qed.

Theorem sink_conts_to_uses_keeps_scope :
  sink_conts_to_uses_keeps_scope_statement.
Proof. apply keeps_scope_intro, sink_conts_to_uses_scoped. Qed.

(* An arm of a switch that goes where its continuation goes. *)

(* An arm as it was, or the application of the handler. *)
Lemma shortcut_arm_from (k : continuation) (target : apply_cont)
    (arm : Z * apply_cont) :
  snd (shortcut_arm k target arm) = snd arm
  ∨ snd (shortcut_arm k target arm) = target.
Proof.
  unfold shortcut_arm. destruct (ac_args (snd arm)); [ | left; reflexivity ].
  destruct
    (Pos.eqb (ac_continuation (snd arm)) k && no_trap_action (snd arm));
    [ right | left ]; reflexivity.
Qed.

Lemma shortcut_arms_names (f : apply_cont -> list positive) (k : continuation)
    (target : apply_cont) (arms : list (Z * apply_cont)) :
  incl (flat_map (fun arm => f (snd arm)) (map (shortcut_arm k target) arms))
    (f target ++ flat_map (fun arm => f (snd arm)) arms).
Proof.
  intros a Ha. apply in_flat_map in Ha. destruct Ha as (arm' & Harm' & Ha).
  apply in_map_iff in Harm'. destruct Harm' as (arm & <- & Harm).
  apply in_or_app.
  destruct (shortcut_arm_from k target arm) as [Hsame | Htarget].
  - right. apply in_flat_map. exists arm. rewrite <- Hsame.
    split; [ exact Harm | exact Ha ].
  - left. rewrite <- Htarget. exact Ha.
Qed.

Lemma shortcut_switch_scoped (k : continuation) (h : continuation_handler)
    (body : expr) :
  scoped (Let_cont_non_recursive k h body) (shortcut_switch k h body).
Proof.
  unfold shortcut_switch.
  destruct h as [[ | param params] metadata handler]; destruct handler;
    destruct body; try apply scoped_refl.
  destruct (hm_params_kinds metadata); [ | apply scoped_refl ].
  match goal with
  | |- context [Pos.eqb (ac_continuation ?target) k || ?exn || ?trap] =>
    destruct (Pos.eqb (ac_continuation target) k || exn || trap);
      [ apply scoped_refl | ]
  end.
  cbv zeta. intros _.
  match goal with
  | |- context [map (shortcut_arm k ?target) ?arms] =>
    pose proof
      (shortcut_arms_names free_variables_of_apply_cont k target arms) as Hv;
    pose proof (shortcut_arms_names apply_cont_continuations k target arms)
      as Hk;
    destruct (arms_apply k (map (shortcut_arm k target) arms)) eqn:Happly
  end;
    scope; (split; [ reflexivity | split ]).
  - intros x Hx. apply in_app_or in Hx. apply in_or_app.
    destruct Hx as [Hx | Hx]; [ left; exact Hx | ].
    apply in_app_or in Hx.
    destruct Hx as [Hx | Hx]; [ right; apply in_or_app; left; exact Hx | ].
    apply Hv, in_app_or in Hx.
    destruct Hx as [Hx | Hx];
      [ left | right; apply in_or_app; right; exact Hx ].
    apply in_without. split; [ exact Hx | intros [] ].
  - intros j Hj. apply in_app_or in Hj. apply in_or_app.
    destruct Hj as [Hj | Hj]; [ left; exact Hj | ].
    apply in_without in Hj. destruct Hj as [Hj Hout].
    apply Hk, in_app_or in Hj.
    destruct Hj as [Hj | Hj];
      [ left; exact Hj | right; apply in_without; split; assumption ].
  - intros x Hx. apply in_app_or in Hx. apply in_or_app.
    destruct Hx as [Hx | Hx]; [ right; apply in_or_app; left; exact Hx | ].
    apply Hv, in_app_or in Hx.
    destruct Hx as [Hx | Hx];
      [ left | right; apply in_or_app; right; exact Hx ].
    apply in_without. split; [ exact Hx | intros [] ].
  - intros j Hj. apply in_or_app.
    pose proof (arms_apply_none k _ Happly) as Hnone.
    assert (Hout : ~ In j [k]) by (intros [<- | []]; exact (Hnone Hj)).
    apply Hk, in_app_or in Hj.
    destruct Hj as [Hj | Hj];
      [ left; exact Hj | right; apply in_without; split; assumption ].
Qed.

Fixpoint shortcut_switches_scoped (e : expr) :
  scoped e (shortcut_switches e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [shortcut_switches];
    try (scoped_under_binders shortcut_switches_scoped).
  eapply scoped_trans; [ | apply shortcut_switch_scoped ].
  scoped_under_binders shortcut_switches_scoped.
Qed.

Theorem shortcut_switches_keeps_scope :
  shortcut_switches_keeps_scope_statement.
Proof. apply keeps_scope_intro, shortcut_switches_scoped. Qed.

(* A rewrite taken into the handlers of recursive continuations. *)

Section Rewrite_rec_handlers.

Variable f : expr -> expr.

Hypothesis f_keeps_scope : keeps_scope f.

Fixpoint rewrite_rec_handlers_scoped (e : expr) :
  scoped e (rewrite_rec_handlers f e)
with rewrite_rec_handlers_in_handler_scoped (h : continuation_handler) :
  scoped_handler h (rewrite_rec_handlers_in_handler f h)
with rewrite_rec_handler_scoped (h : continuation_handler) :
  scoped_handler h (rewrite_rec_handler f h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rewrite_rec_handlers rewrite_rec_handlers_in_handler
           rewrite_rec_handler].
    + apply scoped_let;
        [ apply incl_refl | apply rewrite_rec_handlers_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply rewrite_rec_handlers_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_rec_handlers_in_handler_scoped
        | apply rewrite_rec_handlers_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_rec_handlers_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_rec_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler].
    cbn [rewrite_rec_handlers rewrite_rec_handlers_in_handler
         rewrite_rec_handler].
    apply scoped_handler_intro, rewrite_rec_handlers_scoped.
  - destruct h as [params metadata handler].
    cbn [rewrite_rec_handlers rewrite_rec_handlers_in_handler
         rewrite_rec_handler].
    apply scoped_handler_intro.
    eapply scoped_trans;
      [ apply rewrite_rec_handlers_scoped
      | apply rewrite_code_body_scoped, f_keeps_scope ].
Qed.

End Rewrite_rec_handlers.

Theorem rewrite_rec_handlers_keeps_scope :
  rewrite_rec_handlers_keeps_scope_statement.
Proof.
  intros f Hf. apply keeps_scope_intro. intros e.
  exact (rewrite_rec_handlers_scoped f Hf e).
Qed.

(* Following aliases into the handlers of recursive continuations. *)

(* [scoped_under] for a handler and for a group of handlers. *)
Definition scoped_handler_under (names : list variable)
    (h h' : continuation_handler) : Prop :=
  handler_code_closed h = true ->
  handler_code_closed h' = true
  ∧ incl (free_variables_of_handler h') (free_variables_of_handler h ++ names)
  ∧ incl (free_continuations_of_handler h')
      (free_continuations_of_handler h).

Definition scoped_handlers_under (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_code_closed h) handlers' = true
  ∧ incl (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_variables_of_handler h) handlers
       ++ names)
  ∧ incl
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers).

Lemma scoped_under_trans (names : list variable) (e e' e'' : expr) :
  scoped_under names e e' ->
  scoped_under names e' e'' ->
  scoped_under names e e''.
Proof.
  intros Hfirst Hsecond Hclosed.
  destruct (Hfirst Hclosed) as (Hclosed' & Hv & Hk).
  destruct (Hsecond Hclosed') as (Hclosed'' & Hv' & Hk').
  split; [ exact Hclosed'' | split; [ | eapply incl_tran; eassumption ] ].
  intros y Hy. apply Hv', in_app_or in Hy.
  destruct Hy as [Hy | Hy]; [ exact (Hv y Hy) | ].
  apply in_or_app. right. exact Hy.
Qed.

Lemma scoped_handler_under_intro (names : list variable)
    (params : list variable) (metadata : handler_metadata)
    (handler handler' : expr) :
  scoped_under names handler handler' ->
  scoped_handler_under names (Handler params metadata handler)
    (Handler params metadata handler').
Proof.
  intros Hhandler Hclosed. scope_in Hclosed. scope.
  destruct (Hhandler Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
  apply Hv, in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
  apply in_without. split; assumption.
Qed.

Lemma scoped_handlers_under_nil (names : list variable) :
  scoped_handlers_under names [] [].
Proof.
  intros Hclosed.
  split; [ exact Hclosed | split; [ apply incl_appl | ]; apply incl_refl ].
Qed.

Lemma scoped_handlers_under_cons (names : list variable)
    (k k' : continuation) (h h' : continuation_handler)
    (handlers handlers' : list (continuation * continuation_handler)) :
  scoped_handler_under names h h' ->
  scoped_handlers_under names handlers handlers' ->
  scoped_handlers_under names ((k, h) :: handlers) ((k', h') :: handlers').
Proof.
  intros Hh Hhandlers Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hhandlers Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'.
  split; [ reflexivity | split; [ | apply incl_app_app; assumption ] ].
  apply incl_app_under; assumption.
Qed.

(* A recursive [Let_cont] whose handlers may change. *)
Lemma scoped_under_let_cont_recursive_handlers (names : list variable)
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers_under names handlers handlers' ->
  scoped_under names body body' ->
  scoped_under names (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandlers Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - apply incl_app_under; assumption.
  - rewrite Hnames. apply incl_without, incl_app_app; assumption.
Qed.

Fixpoint follow_aliases_in_loops_scoped_under (e : expr) :
  ∀ (names : list variable) (tenv : typing_env),
    tenv_within names tenv ->
    scoped_under names e (follow_aliases_in_loops tenv e)
with follow_aliases_in_loop_scoped_under (h : continuation_handler) :
  ∀ (names : list variable) (tenv : typing_env),
    tenv_within names tenv ->
    scoped_handler_under names h (follow_aliases_in_loop tenv h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros names tenv Htenv;
      cbn [follow_aliases_in_loops follow_aliases_in_loops_in_handler
           follow_aliases_in_loop].
    + apply scoped_under_let; [ apply incl_appl, incl_refl | ].
      apply follow_aliases_in_loops_scoped_under.
      eapply tenv_within_weaken;
        [ | apply typing_env_under_let_within, Htenv ].
      apply incl_tl, incl_refl.
    + apply scoped_under_let_static.
      apply follow_aliases_in_loops_scoped_under.
      eapply tenv_within_weaken; [ | exact Htenv ]. apply incl_appl, incl_refl.
    + apply scoped_under_let_cont;
        apply follow_aliases_in_loops_scoped_under;
        [ apply drop_vars_within, Htenv | exact Htenv ].
    + apply scoped_under_let_cont_recursive_handlers.
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_under_nil | ].
        apply scoped_handlers_under_cons;
          [ apply follow_aliases_in_loop_scoped_under, Htenv | exact IH ].
      * apply follow_aliases_in_loops_scoped_under, Htenv.
    + apply scoped_under_refl.
    + apply scoped_under_refl.
    + apply scoped_under_refl.
    + apply scoped_under_refl.
  - destruct h as [params metadata handler]. intros names tenv Htenv.
    cbn [follow_aliases_in_loops follow_aliases_in_loops_in_handler
         follow_aliases_in_loop].
    apply scoped_handler_under_intro.
    eapply scoped_under_trans;
      [ apply follow_aliases_in_loops_scoped_under, drop_vars_within, Htenv
      | ].
    unfold rewrite_code_body. cbv zeta.
    match goal with
    | |- context [Bool.eqb ?b ?b'] => destruct (Bool.eqb b b')
    end;
      [ apply follow_aliases_scoped_under, drop_vars_within, Htenv
      | apply scoped_under_refl ].
Qed.

Lemma follow_aliases_in_loops_scoped (e : expr) :
  scoped e (follow_aliases_in_loops [] e).
Proof.
  apply scoped_under_none, follow_aliases_in_loops_scoped_under.
  intros entry [].
Qed.

Theorem follow_aliases_in_loops_keeps_scope :
  follow_aliases_in_loops_keeps_scope_statement.
Proof. apply keeps_scope_intro, follow_aliases_in_loops_scoped. Qed.

(* Removing the parameters that a handler does not use. *)

Lemma keep_incl {A : Type} (mask : list bool) :
  ∀ l : list A, incl (keep mask l) l.
Proof.
  induction mask as [ | used mask IH]; intros [ | a l]; cbn [keep].
  - intros x [].
  - intros x [].
  - destruct used; intros x [].
  - destruct used.
    + intros x [<- | Hx]; [ left; reflexivity | right; exact (IH l x Hx) ].
    + intros x Hx. right. exact (IH l x Hx).
Qed.

Lemma keep_all_or_incl {A : Type} (mask : option (list bool)) (l : list A) :
  incl (keep_all_or mask l) l.
Proof.
  destruct mask as [mask | ]; cbn [keep_all_or];
    [ apply keep_incl | apply incl_refl ].
Qed.

Lemma drop_args_free (known : masks) (ac : apply_cont) :
  incl (free_variables_of_apply_cont (drop_args known ac))
    (free_variables_of_apply_cont ac).
Proof.
  unfold free_variables_of_apply_cont, drop_args, free_variables_of_simples.
  cbn [ac_args]. intros x Hx. apply in_flat_map in Hx.
  destruct Hx as (s & Hs & Hx). apply in_flat_map. exists s.
  split; [ exact (keep_all_or_incl _ _ s Hs) | exact Hx ].
Qed.

(* A parameter that the test says is used is kept. *)
Lemma keep_used (used : variable -> bool) (x : variable)
    (params : list variable) :
  In x params -> used x = true -> In x (keep (map used params) params).
Proof.
  induction params as [ | param params IH]; cbn [map keep]; [ intros [] | ].
  intros [-> | Hx] Hused.
  - rewrite Hused. left. reflexivity.
  - destruct (used param); [ right | ]; exact (IH Hx Hused).
Qed.

Lemma used_params_kept (is_exn_handler : bool) (params : list variable)
    (handler : expr) (x : variable) :
  In x params ->
  var_free_b x handler = true ->
  In x (keep_all_or (used_params is_exn_handler params handler) params).
Proof.
  intros Hx Hused. unfold used_params. cbv zeta.
  pose proof
    (keep_used (fun param => var_free_b param handler) x params Hx Hused)
    as Hkept.
  destruct is_exn_handler;
    [ destruct params as [ | param params]; cbn [map] | ];
    match goal with
    | |- context [forallb ?test ?mask] => destruct (forallb test mask)
    end;
    cbn [keep_all_or keep]; try exact Hx; try exact Hkept.
  destruct Hx as [<- | Hx]; [ left; reflexivity | right ].
  exact (keep_used (fun param => var_free_b param handler) x params Hx Hused).
Qed.

(* A handler less the parameters that it does not use, as the test
   says of the handler before or of the handler after. *)
Lemma scoped_handler_unused_params (is_exn_handler : bool)
    (params : list variable) (metadata metadata' : handler_metadata)
    (handler handler' tested : expr) (mask : option (list bool)) :
  mask = None ∨ mask = used_params is_exn_handler params tested ->
  tested = handler ∨ tested = handler' ->
  scoped handler handler' ->
  scoped_handler (Handler params metadata handler)
    (Handler (keep_all_or mask params) metadata' handler').
Proof.
  intros Hmask Htested Hhandler Hclosed. scope_in Hclosed. scope.
  destruct (Hhandler Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hkept].
  apply in_without. split; [ apply Hv, Hy | ].
  intros Hparam. apply Hkept.
  destruct Hmask as [-> | ->]; [ exact Hparam | ].
  apply used_params_kept; [ exact Hparam | ].
  destruct Htested as [-> | ->]; apply var_free_b_free;
    solve [ assumption | apply Hv, Hy ].
Qed.

Fixpoint drop_params_scoped (e : expr) :
  ∀ known : masks, scoped e (drop_params known e)
with drop_params_in_loop_scoped (h : continuation_handler) :
  ∀ (known : masks) (k : continuation) (whole : expr),
    scoped_handler h
      (drop_params_in_loop known (used_params_in_loop k h whole) h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros known; cbn [drop_params drop_params_in_loop]; cbv zeta.
    + apply scoped_let; [ apply incl_refl | apply drop_params_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply drop_params_scoped ].
    + apply scoped_let_cont; [ | apply drop_params_scoped ].
      apply
        (scoped_handler_unused_params (hm_is_exn_handler metadata) _ _ _ _ _
           (drop_params known handler));
        [ destruct (returned_to k body); [ left | right ]; reflexivity
        | right; reflexivity | apply drop_params_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply drop_params_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * set (known' := _ ++ known). clearbody known'.
        set (whole := Let_cont_recursive handlers body). clearbody whole.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply drop_params_in_loop_scoped | exact IH ].
    + apply scoped_refl.
    + intros _. scope.
      split; [ reflexivity | split; [ apply drop_args_free | ] ].
      apply incl_refl.
    + intros _. scope. split; [ reflexivity | split ].
      * apply incl_app_app; [ apply incl_refl | ].
        intros x Hx. apply in_flat_map in Hx.
        destruct Hx as (arm' & Harm' & Hx). apply in_map_iff in Harm'.
        destruct Harm' as (arm & <- & Harm). apply in_flat_map. exists arm.
        split; [ exact Harm | exact (drop_args_free known (snd arm) x Hx) ].
      * rewrite
          (flat_map_map_same (fun arm => apply_cont_continuations (snd arm))
             (fun arm => apply_cont_continuations (snd arm)));
          [ apply incl_refl | intros arm; reflexivity ].
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros known k whole.
    unfold used_params_in_loop. cbn [drop_params drop_params_in_loop].
    apply
      (scoped_handler_unused_params (hm_is_exn_handler metadata) _ _ _ _ _
         handler);
      [ destruct (returned_to k whole); [ left | right ]; reflexivity
      | left; reflexivity | apply drop_params_scoped ].
Qed.

Theorem drop_unused_params_keeps_scope :
  drop_unused_params_keeps_scope_statement.
Proof.
  apply keeps_scope_intro. intros e. unfold drop_unused_params.
  apply drop_params_scoped.
Qed.

(* The parameters that a loop only passes round removed, with the
   [Let]s that are left unused. Whatever the masks of a loop are, a
   handler loses only parameters that it does not use once it is
   rewritten, so how the masks are found is not looked at. *)

Fixpoint drop_dead_scoped (e : expr) :
  ∀ known : masks, scoped e (drop_dead known e)
with drop_dead_in_loop_scoped (h : continuation_handler) :
  ∀ (inside : masks) (kept : bool),
    scoped_handler h (drop_dead_in_loop inside kept h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros known; cbn [drop_dead drop_dead_in_loop]; cbv zeta.
    + destruct (var_free_b x (drop_dead known body)) eqn:Hfree; cbn [orb];
        [ apply scoped_let; [ apply incl_refl | apply drop_dead_scoped ] | ].
      destruct (named_may_go d); cbn [negb];
        [ | apply scoped_let; [ apply incl_refl | apply drop_dead_scoped ] ].
      apply scoped_let_unused; [ apply drop_dead_scoped | exact Hfree ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply drop_dead_scoped ].
    + apply scoped_let_cont; [ | apply drop_dead_scoped ].
      apply
        (scoped_handler_unused_params (hm_is_exn_handler metadata) _ _ _ _ _
           (drop_dead known handler));
        [ destruct (returned_to k body); [ left | right ]; reflexivity
        | right; reflexivity | apply drop_dead_scoped ].
    + assert
        (Hhandlers :
          ∀ (inside : masks) (kept : continuation -> bool),
            scoped_handlers handlers
              (map
                 (fun '(k, h) => (k, drop_dead_in_loop inside (kept k) h))
                 handlers)).
      { intros inside kept.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply drop_dead_in_loop_scoped | exact IH ]. }
      apply scoped_let_cont_recursive;
        [ | apply Hhandlers | apply drop_dead_scoped ].
      rewrite map_map. apply map_ext. intros [k h]. reflexivity.
    + apply scoped_refl.
    + intros _. scope.
      split; [ reflexivity | split; [ apply drop_args_free | ] ].
      apply incl_refl.
    + intros _. scope. split; [ reflexivity | split ].
      * apply incl_app_app; [ apply incl_refl | ].
        intros x Hx. apply in_flat_map in Hx.
        destruct Hx as (arm' & Harm' & Hx). apply in_map_iff in Harm'.
        destruct Harm' as (arm & <- & Harm). apply in_flat_map. exists arm.
        split; [ exact Harm | exact (drop_args_free known (snd arm) x Hx) ].
      * rewrite
          (flat_map_map_same (fun arm => apply_cont_continuations (snd arm))
             (fun arm => apply_cont_continuations (snd arm)));
          [ apply incl_refl | intros arm; reflexivity ].
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros inside kept.
    cbn [drop_dead drop_dead_in_loop]. cbv zeta.
    apply
      (scoped_handler_unused_params (hm_is_exn_handler metadata) _ _ _ _ _
         (drop_dead inside handler));
      [ destruct kept; [ left | right ]; reflexivity
      | right; reflexivity | apply drop_dead_scoped ].
Qed.

Lemma drop_dead_params_scoped (e : expr) : scoped e (drop_dead_params e).
Proof. unfold drop_dead_params. apply drop_dead_scoped. Qed.

Theorem drop_dead_params_keeps_scope :
  drop_dead_params_keeps_scope_statement.
Proof. apply keeps_scope_intro, drop_dead_params_scoped. Qed.

(* The unused [Let]s that are sure to have a value removed, in the
   handlers of recursive continuations too. *)

Fixpoint remove_unused_defined_lets_in_loops_scoped (e : expr) :
  ∀ kinds : kinds_in_scope,
    scoped e (remove_unused_defined_lets_in_loops kinds e)
with remove_unused_defined_lets_in_handler_scoped
    (h : continuation_handler) :
  ∀ kinds : kinds_in_scope,
    scoped_handler h (remove_unused_defined_lets_in_handler kinds h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros kinds;
      cbn [remove_unused_defined_lets_in_loops
           remove_unused_defined_lets_in_handler].
    + cbv zeta. destruct (named_defined kinds kind d); cbn [andb];
        [ | apply scoped_let;
            [ apply incl_refl
            | apply remove_unused_defined_lets_in_loops_scoped ] ].
      destruct
        (var_free_b x
           (remove_unused_defined_lets_in_loops ((x, kind) :: kinds) body))
        eqn:Hfree; cbn [negb];
        [ apply scoped_let;
          [ apply incl_refl
          | apply remove_unused_defined_lets_in_loops_scoped ]
        | ].
      apply scoped_let_unused;
        [ apply remove_unused_defined_lets_in_loops_scoped | exact Hfree ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl
        | apply remove_unused_defined_lets_in_loops_scoped ].
    + apply scoped_let_cont;
        [ apply remove_unused_defined_lets_in_handler_scoped
        | apply remove_unused_defined_lets_in_loops_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply remove_unused_defined_lets_in_loops_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply remove_unused_defined_lets_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros kinds.
    cbn [remove_unused_defined_lets_in_loops
         remove_unused_defined_lets_in_handler].
    apply scoped_handler_intro, remove_unused_defined_lets_in_loops_scoped.
Qed.

Theorem remove_unused_defined_lets_in_loops_keeps_scope :
  remove_unused_defined_lets_in_loops_keeps_scope_statement.
Proof.
  intros kinds. apply keeps_scope_intro. intros e.
  apply remove_unused_defined_lets_in_loops_scoped.
Qed.

(* A parameter that every application gives the same thing, bound to it
   at the start of the handler. *)

Definition alias_variables (alias : option simple) : list variable :=
  match alias with
  | Some s => free_variables_of_simple s
  | None => []
  end.

(* The [Let]s that are put around a handler bring the variables of what
   they bind, and nothing else. *)
Lemma bind_aliases_scope (params : list variable) :
  ∀ (kinds : list flambda_kind) (aliases : list (option simple))
    (handler : expr),
    code_closed (bind_aliases params kinds aliases handler)
    = code_closed handler
    ∧ incl (free_variables_of_expr (bind_aliases params kinds aliases handler))
        (free_variables_of_expr handler ++ flat_map alias_variables aliases)
    ∧ free_continuations_of_expr (bind_aliases params kinds aliases handler)
      = free_continuations_of_expr handler.
Proof.
  induction params as [ | param params IH];
    intros [ | kind kinds] [ | alias aliases] handler; cbn [bind_aliases];
    try
      (split; [ reflexivity | split; [ apply incl_appl, incl_refl | ] ];
       reflexivity).
  destruct (IH kinds aliases handler) as (Hc & Hv & Hk).
  cbv zeta. destruct alias as [s | ]; cbn [flat_map alias_variables]; scope.
  - split; [ exact Hc | split; [ | exact Hk ] ].
    cbn [free_variables_of_named].
    intros y Hy. apply in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy]; [ right; apply in_or_app; left; exact Hy | ].
    apply in_without in Hy. destruct Hy as [Hy _].
    apply Hv, in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ left; exact Hy | right; apply in_or_app; right; exact Hy ].
  - split; [ exact Hc | split; [ exact Hv | exact Hk ] ].
Qed.

Lemma aliases_from_In (params : list variable) :
  ∀ (i : nat) (alias_at : nat -> variable -> option simple) (s : simple),
    In (Some s) (aliases_from i params alias_at) ->
    ∃ (j : nat) (param : variable), alias_at j param = Some s.
Proof.
  induction params as [ | param params IH]; intros i alias_at s;
    cbn [aliases_from]; [ intros [] | ].
  intros [Hfirst | Hrest];
    [ exists i, param; exact Hfirst | exact (IH (S i) alias_at s Hrest) ].
Qed.

(* What a parameter is bound to is an argument of an application in the
   body of the [Let_cont], and no binder of the [Let_cont] binds it. *)
Lemma aliases_of_from (k : continuation) (params : list variable)
    (metadata : handler_metadata) (loop : option expr) (body whole : expr)
    (s : simple) :
  In (Some s) (aliases_of k params metadata loop body whole) ->
  ∃ args : list simple,
    some_args k body = Some args ∧ In s args ∧ stable s whole = true.
Proof.
  unfold aliases_of.
  match goal with
  | |- In _ (if ?test then _ else _) -> _ => destruct test; [ intros [] | ]
  end.
  destruct (some_args k body) as [args | ]; [ | intros [] ].
  intros Hin. apply aliases_from_In in Hin.
  destruct Hin as (j & param & Halias). exists args.
  split; [ reflexivity | ].
  cbv beta in Halias. revert Halias.
  destruct (nth_error args j) as [s' | ] eqn:Hnth;
    [ | intros Halias; discriminate Halias ].
  match goal with
  | |- (if ?test then _ else _) = _ -> _ =>
    destruct test eqn:Htest; [ | intros Halias; discriminate Halias ]
  end.
  intros Halias. injection Halias as ->.
  apply andb_true_iff in Htest. destruct Htest as [Htest _].
  apply andb_true_iff in Htest. destruct Htest as [Hstable _].
  split; [ exact (nth_error_In _ _ Hnth) | exact Hstable ].
Qed.

Lemma args_if_applies_args (k : continuation) (ac : apply_cont)
    (args : list simple) :
  args_if_applies k ac = Some args -> args = ac_args ac.
Proof.
  unfold args_if_applies.
  destruct (Pos.eqb (ac_continuation ac) k); intros Hargs;
    [ injection Hargs as <-; reflexivity | discriminate Hargs ].
Qed.

Lemma first_some_from {A : Type} (a b : option A) (x : A) :
  first_some a b = Some x -> a = Some x ∨ b = Some x.
Proof.
  destruct a; cbn [first_some]; intros Hsome; [ left | right ]; exact Hsome.
Qed.

(* A variable of an argument of an application in an expression is free
   in the expression, if the expression does not bind it. *)
Fixpoint some_args_free (e : expr) :
  ∀ (k : continuation) (args : list simple) (s : simple) (x : variable),
    some_args k e = Some args ->
    In s args ->
    In x (free_variables_of_simple s) ->
    binds_var x e = false ->
    In x (free_variables_of_expr e)
with some_args_in_handler_free (h : continuation_handler) :
  ∀ (k : continuation) (args : list simple) (s : simple) (x : variable),
    some_args_in_handler k h = Some args ->
    In s args ->
    In x (free_variables_of_simple s) ->
    binds_var_in_handler x h = false ->
    In x (free_variables_of_handler h).
Proof.
  - destruct e
      as [y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros k args s x;
      cbn [some_args some_args_in_handler binds_var binds_var_in_handler];
      scope; intros Hargs Hs Hx Hbinds; try discriminate Hargs.
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hne Hbinds].
      apply in_or_app. right. apply in_without. split.
      * exact (some_args_free body k args s x Hargs Hs Hx Hbinds).
      * intros [<- | []]. rewrite Pos.eqb_refl in Hne. discriminate Hne.
    + apply in_or_app. right.
      exact (some_args_free body k args s x Hargs Hs Hx Hbinds).
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hh Hbody].
      apply in_or_app. apply first_some_from in Hargs.
      destruct Hargs as [Hargs | Hargs];
        [ right; exact (some_args_free body k args s x Hargs Hs Hx Hbody)
        | left;
          exact (some_args_in_handler_free h k args s x Hargs Hs Hx Hh) ].
    + apply orb_false_iff in Hbinds. destruct Hbinds as [Hgroup Hbody].
      apply in_or_app. apply first_some_from in Hargs.
      destruct Hargs as [Hargs | Hargs];
        [ right; exact (some_args_free body k args s x Hargs Hs Hx Hbody)
        | left ].
      revert Hargs Hgroup.
      induction handlers as [ | [k' h] handlers IH];
        cbn [fold_right existsb flat_map]; intros Hargs Hgroup;
        [ discriminate Hargs | ].
      apply orb_false_iff in Hgroup. destruct Hgroup as [Hh Hrest].
      apply in_or_app. apply first_some_from in Hargs.
      destruct Hargs as [Hargs | Hargs];
        [ left;
          exact (some_args_in_handler_free h k args s x Hargs Hs Hx Hh)
        | right; exact (IH Hargs Hrest) ].
    + apply args_if_applies_args in Hargs. subst args.
      unfold free_variables_of_apply_cont, free_variables_of_simples.
      apply in_flat_map. exists s. split; assumption.
    + apply in_or_app. right. revert Hargs.
      induction arms as [ | arm arms IH]; cbn [fold_right flat_map];
        intros Hargs; [ discriminate Hargs | ].
      apply in_or_app. apply first_some_from in Hargs.
      destruct Hargs as [Hargs | Hargs]; [ left | right; exact (IH Hargs) ].
      apply args_if_applies_args in Hargs. subst args.
      unfold free_variables_of_apply_cont, free_variables_of_simples.
      apply in_flat_map. exists s. split; assumption.
  - destruct h as [params metadata handler]. intros k args s x.
    cbn [some_args some_args_in_handler binds_var binds_var_in_handler].
    scope. intros Hargs Hs Hx Hbinds.
    apply orb_false_iff in Hbinds. destruct Hbinds as [Hparams Hbinds].
    apply in_without.
    split; [ exact (some_args_free handler k args s x Hargs Hs Hx Hbinds) | ].
    intros Hin. apply among_In in Hin. congruence.
Qed.

Lemma aliases_of_free (k : continuation) (params : list variable)
    (metadata : handler_metadata) (loop : option expr) (body whole : expr)
    (s : simple) :
  (∀ x : variable, binds_var x whole = false -> binds_var x body = false) ->
  In (Some s) (aliases_of k params metadata loop body whole) ->
  incl (free_variables_of_simple s) (free_variables_of_expr body).
Proof.
  intros Hwhole Hin.
  destruct (aliases_of_from _ _ _ _ _ _ _ Hin)
    as (args & Hargs & Hs & Hstable).
  intros x Hx. apply (some_args_free body k args s x Hargs Hs Hx), Hwhole.
  destruct s as [y | sym | c]; cbn [free_variables_of_simple] in Hx;
    [ | destruct Hx | destruct Hx ].
  destruct Hx as [<- | []]. cbn [stable] in Hstable.
  apply negb_true_iff in Hstable. exact Hstable.
Qed.

(* A handler whose parameters are bound again to variables among the
   names given. *)
Lemma scoped_handler_under_bound (names : list variable)
    (params : list variable) (metadata : handler_metadata)
    (kinds : list flambda_kind) (aliases : list (option simple))
    (handler handler' : expr) :
  (∀ s : simple,
    In (Some s) aliases -> incl (free_variables_of_simple s) names) ->
  scoped handler handler' ->
  scoped_handler_under names (Handler params metadata handler)
    (Handler params metadata (bind_aliases params kinds aliases handler')).
Proof.
  intros Haliases Hhandler Hclosed. scope_in Hclosed. scope.
  destruct (Hhandler Hclosed) as (Hclosed' & Hv & Hk).
  destruct (bind_aliases_scope params kinds aliases handler')
    as (Hc & Hv' & Hk').
  rewrite Hc, Hk'. split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
  apply Hv', in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy].
  - left. apply in_without. split; [ apply Hv, Hy | exact Hparams ].
  - right. apply in_flat_map in Hy. destruct Hy as (alias & Halias & Hy).
    destruct alias as [s | ]; [ | destruct Hy ].
    exact (Haliases s Halias y Hy).
Qed.

(* A [Let_cont] whose handlers gain free variables of its body only. *)
Lemma scoped_let_cont_handler_under (k : continuation)
    (h h' : continuation_handler) (body body' : expr) :
  scoped_handler_under (free_variables_of_expr body) h h' ->
  scoped body body' ->
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  intros Hh Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hh Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ exact (Hv y Hy) | ].
    apply in_or_app. right. exact (Hv' y Hy).
  - apply incl_app_app; [ exact Hk | apply incl_without, Hk' ].
Qed.

Lemma scoped_let_cont_recursive_handlers_under
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers_under (free_variables_of_expr body) handlers handlers' ->
  scoped body body' ->
  scoped (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandlers Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - intros y Hy. apply in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ exact (Hv y Hy) | ].
    apply in_or_app. right. exact (Hv' y Hy).
  - rewrite Hnames. apply incl_without, incl_app_app; assumption.
Qed.

Fixpoint bind_aliased_params_scoped (e : expr) :
  scoped e (bind_aliased_params e)
with bind_aliased_params_in_loop_scoped (h : continuation_handler) :
  ∀ (lone : bool) (k : continuation) (body whole : expr),
    (∀ x : variable, binds_var x whole = false -> binds_var x body = false) ->
    scoped_handler_under (free_variables_of_expr body) h
      (bind_aliased_params_in_loop lone k body whole h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      cbn [bind_aliased_params bind_aliased_params_in_loop].
    + apply scoped_let; [ apply incl_refl | apply bind_aliased_params_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply bind_aliased_params_scoped ].
    + apply scoped_let_cont_handler_under;
        [ | apply bind_aliased_params_scoped ].
      apply scoped_handler_under_bound;
        [ | apply bind_aliased_params_scoped ].
      intros s. apply aliases_of_free. intros x Hbinds.
      cbn [binds_var binds_var_in_handler] in Hbinds.
      apply orb_false_iff in Hbinds. exact (proj2 Hbinds).
    + apply scoped_let_cont_recursive_handlers_under;
        [ | | apply bind_aliased_params_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * assert (Hwhole :
          ∀ x : variable,
            binds_var x (Let_cont_recursive handlers body) = false ->
            binds_var x body = false).
        { intros x Hbinds. cbn [binds_var] in Hbinds.
          apply orb_false_iff in Hbinds. exact (proj2 Hbinds). }
        set (lone := alone handlers).
        set (whole := Let_cont_recursive handlers body) in *.
        clearbody lone whole.
        induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_under_nil | ].
        apply scoped_handlers_under_cons;
          [ apply bind_aliased_params_in_loop_scoped, Hwhole | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros lone k body whole Hwhole.
    cbn [bind_aliased_params bind_aliased_params_in_loop]. cbv zeta.
    apply scoped_handler_under_bound; [ | apply bind_aliased_params_scoped ].
    intros s Hs.
    destruct lone;
      [ exact (aliases_of_free _ _ _ _ _ _ _ Hwhole Hs) | destruct Hs ].
Qed.

Theorem bind_aliased_params_keeps_scope :
  bind_aliased_params_keeps_scope_statement.
Proof. apply keeps_scope_intro, bind_aliased_params_scoped. Qed.

(* An application of a continuation whose handler only applies another,
   made an application of that other. *)

(* A term that keeps the scope of another, but for the variables and
   the continuations given. *)
Definition scoped_among (vs : list variable) (ks : list continuation)
    (e e' : expr) : Prop :=
  code_closed e = true ->
  code_closed e' = true
  ∧ incl (free_variables_of_expr e') (free_variables_of_expr e ++ vs)
  ∧ incl (free_continuations_of_expr e')
      (free_continuations_of_expr e ++ ks).

Definition scoped_handlers_among (vs : list variable)
    (ks : list continuation)
    (handlers handlers' : list (continuation * continuation_handler))
    : Prop :=
  forallb (fun '(_, h) => handler_code_closed h) handlers = true ->
  forallb (fun '(_, h) => handler_code_closed h) handlers' = true
  ∧ incl (flat_map (fun '(_, h) => free_variables_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_variables_of_handler h) handlers ++ vs)
  ∧ incl
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers')
      (flat_map (fun '(_, h) => free_continuations_of_handler h) handlers
       ++ ks).

Lemma scoped_among_none (e e' : expr) :
  scoped_among [] [] e e' -> scoped e e'.
Proof.
  intros Hamong Hclosed. destruct (Hamong Hclosed) as (Hclosed' & Hv & Hk).
  rewrite app_nil_r in Hv, Hk. split; [ exact Hclosed' | split; assumption ].
Qed.

Lemma scoped_among_refl (vs : list variable) (ks : list continuation)
    (e : expr) :
  scoped_among vs ks e e.
Proof.
  intros Hclosed. split; [ exact Hclosed | split; apply incl_appl, incl_refl ].
Qed.

Lemma scoped_among_let (vs : list variable) (ks : list continuation)
    (x : variable) (kind : flambda_kind) (d : named) (body body' : expr) :
  scoped_among vs ks body body' ->
  scoped_among vs ks (Let x kind d body) (Let x kind d body').
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  split; [ exact Hclosed' | split; [ | exact Hk ] ].
  intros y Hy. apply in_app_or in Hy. apply in_or_app.
  destruct Hy as [Hy | Hy]; [ left; apply in_or_app; left; exact Hy | ].
  apply in_without in Hy. destruct Hy as [Hy Hx].
  apply Hv, in_app_or in Hy.
  destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
  apply in_or_app. right. apply in_without. split; assumption.
Qed.

Lemma scoped_among_let_static (vs : list variable) (ks : list continuation)
    (bindings : list static_binding) (body body' : expr) :
  scoped_among vs ks body body' ->
  scoped_among vs ks (Let_static bindings body) (Let_static bindings body').
Proof.
  intros Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hgroup Hclosed].
  destruct (Hbody Hclosed) as (Hclosed' & Hv & Hk).
  rewrite Hgroup, Hclosed', <- !app_assoc.
  split; [ reflexivity | ].
  split; (apply incl_app_app; [ apply incl_refl | assumption ]).
Qed.

(* A [Let_cont] in whose body the variables and continuations of the
   handler may come to be named. *)
Lemma scoped_among_let_cont (vs vs' : list variable)
    (ks ks' : list continuation) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' body body' : expr) :
  incl vs' (without params (free_variables_of_expr handler')) ->
  incl ks' (free_continuations_of_expr handler') ->
  scoped_among vs ks handler handler' ->
  scoped_among (vs' ++ vs) (ks' ++ ks) body body' ->
  scoped_among vs ks
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k (Handler params metadata handler') body').
Proof.
  intros Hvs' Hks' Hhandler Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandler Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - assert (Hh :
      incl (without params (free_variables_of_expr handler'))
        ((without params (free_variables_of_expr handler)
          ++ free_variables_of_expr body)
         ++ vs)).
    { intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply Hv, in_app_or in Hy. apply in_or_app.
      destruct Hy as [Hy | Hy]; [ left | right; exact Hy ].
      apply in_or_app. left. apply in_without. split; assumption. }
    intros y Hy. apply in_app_or in Hy.
    destruct Hy as [Hy | Hy]; [ exact (Hh y Hy) | ].
    apply Hv', in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ apply in_or_app; left; apply in_or_app; right; exact Hy | ].
    apply in_app_or in Hy.
    destruct Hy as [Hy | Hy];
      [ exact (Hh y (Hvs' y Hy)) | apply in_or_app; right; exact Hy ].
  - assert (Hh :
      incl (free_continuations_of_expr handler')
        ((free_continuations_of_expr handler
          ++ without [k] (free_continuations_of_expr body))
         ++ ks)).
    { intros j Hj. apply Hk, in_app_or in Hj. apply in_or_app.
      destruct Hj as [Hj | Hj];
        [ left; apply in_or_app; left; exact Hj | right; exact Hj ]. }
    intros j Hj. apply in_app_or in Hj.
    destruct Hj as [Hj | Hj]; [ exact (Hh j Hj) | ].
    apply in_without in Hj. destruct Hj as [Hj Hout].
    apply Hk', in_app_or in Hj. destruct Hj as [Hj | Hj].
    + apply in_or_app. left. apply in_or_app. right. apply in_without.
      split; assumption.
    + apply in_app_or in Hj.
      destruct Hj as [Hj | Hj];
        [ exact (Hh j (Hks' j Hj)) | apply in_or_app; right; exact Hj ].
Qed.

Lemma scoped_handlers_among_nil (vs : list variable)
    (ks : list continuation) :
  scoped_handlers_among vs ks [] [].
Proof.
  intros Hclosed. split; [ exact Hclosed | split; apply incl_appl, incl_refl ].
Qed.

Lemma scoped_handlers_among_cons (vs : list variable)
    (ks : list continuation) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler handler' : expr)
    (handlers handlers' : list (continuation * continuation_handler)) :
  scoped_among vs ks handler handler' ->
  scoped_handlers_among vs ks handlers handlers' ->
  scoped_handlers_among vs ks
    ((k, Handler params metadata handler) :: handlers)
    ((k, Handler params metadata handler') :: handlers').
Proof.
  intros Hhandler Hhandlers Hclosed. cbn [forallb flat_map] in Hclosed |- *.
  scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandler Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hhandlers Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - apply incl_app_under; [ | exact Hv' ].
    intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
    apply Hv, in_app_or in Hy. apply in_or_app.
    destruct Hy as [Hy | Hy];
      [ left; apply in_without; split; assumption | right; exact Hy ].
  - apply incl_app_under; assumption.
Qed.

Lemma scoped_among_let_cont_recursive (vs : list variable)
    (ks : list continuation)
    (handlers handlers' : list (continuation * continuation_handler))
    (body body' : expr) :
  map fst handlers' = map fst handlers ->
  scoped_handlers_among vs ks handlers handlers' ->
  scoped_among vs ks body body' ->
  scoped_among vs ks (Let_cont_recursive handlers body)
    (Let_cont_recursive handlers' body').
Proof.
  intros Hnames Hhandlers Hbody Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hhandlers Hfirst) as (Hfirst' & Hv & Hk).
  destruct (Hbody Hrest) as (Hrest' & Hv' & Hk').
  rewrite Hfirst', Hrest'. split; [ reflexivity | split ].
  - apply incl_app_under; assumption.
  - rewrite Hnames. intros j Hj. apply in_without in Hj.
    destruct Hj as [Hj Hout].
    apply (incl_app_under _ _ _ _ _ Hk Hk'), in_app_or in Hj.
    apply in_or_app.
    destruct Hj as [Hj | Hj];
      [ left; apply in_without; split; assumption | right; exact Hj ].
Qed.

(* The variables that an application may come to name where it is made
   to go where a handler goes: those of the handler's application that
   are not its parameters. And the continuations. *)
Definition forward_variables
    (entry : continuation * option (list variable * apply_cont))
    : list variable :=
  match snd entry with
  | Some (params, target) =>
    without params (free_variables_of_apply_cont target)
  | None => []
  end.

Definition forward_continuations
    (entry : continuation * option (list variable * apply_cont))
    : list continuation :=
  match snd entry with
  | Some (_, target) => apply_cont_continuations target
  | None => []
  end.

Lemma forward_of_In (F : forwards) (k : continuation)
    (params : list variable) (target : apply_cont) :
  forward_of F k = Some (params, target) ->
  In (k, Some (params, target)) F.
Proof.
  unfold forward_of.
  destruct (lookup k F) as [entry | ] eqn:Hlookup; [ | discriminate ].
  intros ->. exact (lookup_in _ _ _ Hlookup).
Qed.

Lemma in_fst_combine (params : list variable) :
  ∀ (args : list simple) (x : variable),
    length args = length params ->
    In x params ->
    In x (map fst (combine params args)).
Proof.
  induction params as [ | param params IH]; intros [ | arg args] x Hlength;
    try discriminate Hlength; cbn [combine map fst]; [ intros [] | ].
  injection Hlength as Hlength.
  intros [<- | Hx]; [ left; reflexivity | right; exact (IH args x Hlength Hx) ].
Qed.

Lemma subst_simple_free (params : list variable) (args : list simple)
    (s : simple) :
  length args = length params ->
  incl (free_variables_of_simple (subst_simple params args s))
    (free_variables_of_simples args
     ++ without params (free_variables_of_simple s)).
Proof.
  intros Hlength. destruct s as [x | sym | c]; cbn [subst_simple];
    try (intros y []).
  destruct (lookup x (combine params args)) as [arg | ] eqn:Hlookup.
  - intros y Hy. apply in_or_app. left.
    unfold free_variables_of_simples. apply in_flat_map. exists arg.
    split; [ | exact Hy ]. apply lookup_in in Hlookup.
    exact (in_combine_r _ _ _ _ Hlookup).
  - intros y Hy. apply in_or_app. right. apply in_without.
    split; [ exact Hy | ].
    cbn [free_variables_of_simple] in Hy. destruct Hy as [<- | []].
    intros Hin. apply (lookup_none_not_in x (combine params args) Hlookup).
    exact (in_fst_combine params args x Hlength Hin).
Qed.

Lemma forwarded_free (params : list variable) (args : list simple)
    (target : apply_cont) :
  length args = length params ->
  incl (free_variables_of_apply_cont (forwarded params args target))
    (free_variables_of_simples args
     ++ without params (free_variables_of_apply_cont target)).
Proof.
  intros Hlength. unfold forwarded, free_variables_of_apply_cont.
  cbn [ac_args]. intros y Hy.
  unfold free_variables_of_simples in Hy. apply in_flat_map in Hy.
  destruct Hy as (s' & Hs' & Hy). apply in_map_iff in Hs'.
  destruct Hs' as (s & <- & Hs).
  apply (subst_simple_free params args s Hlength), in_app_or in Hy.
  apply in_or_app. destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
  apply in_without in Hy. destruct Hy as [Hy Hparams].
  apply in_without. split; [ | exact Hparams ].
  unfold free_variables_of_simples. apply in_flat_map. exists s.
  split; assumption.
Qed.

Lemma forward_apply_cont_keeping_scope (F : forwards) (ac : apply_cont) :
  incl (free_variables_of_apply_cont (forward_apply_cont_keeping F ac))
    (free_variables_of_apply_cont ac ++ flat_map forward_variables F)
  ∧ incl (apply_cont_continuations (forward_apply_cont_keeping F ac))
      (apply_cont_continuations ac ++ flat_map forward_continuations F).
Proof.
  unfold forward_apply_cont_keeping.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ]
    eqn:Hforward;
    [ | split; apply incl_appl, incl_refl ].
  match goal with
  | |- context [if ?test then _ else _] => destruct test eqn:Htest
  end;
    [ | split; apply incl_appl, incl_refl ].
  apply andb_true_iff in Htest. destruct Htest as [Hlength _].
  apply Nat.eqb_eq in Hlength. apply forward_of_In in Hforward. split.
  - intros y Hy.
    change
      (In y
         (free_variables_of_apply_cont (forwarded params (ac_args ac) target)))
      in Hy.
    apply (forwarded_free params (ac_args ac) target Hlength), in_app_or
      in Hy.
    apply in_or_app. destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_flat_map. exists (ac_continuation ac, Some (params, target)).
    split; [ exact Hforward | exact Hy ].
  - (* the continuation of the handler, and the one of the trap action
       that the application keeps *)
    intros j Hj. unfold forwarded_keeping, apply_cont_continuations in Hj.
    cbn [ac_continuation ac_trap_action] in Hj. apply in_or_app.
    destruct Hj as [<- | Hj].
    + right. apply in_flat_map.
      exists (ac_continuation ac, Some (params, target)).
      split; [ exact Hforward | left; reflexivity ].
    + left. unfold apply_cont_continuations. right. exact Hj.
Qed.

Lemma forward_arm_scope (F : forwards) (ac : apply_cont) :
  incl (free_variables_of_apply_cont (forward_arm F ac))
    (free_variables_of_apply_cont ac ++ flat_map forward_variables F)
  ∧ incl (apply_cont_continuations (forward_arm F ac))
      (apply_cont_continuations ac ++ flat_map forward_continuations F).
Proof.
  unfold forward_arm.
  destruct (forward_of F (ac_continuation ac))
    as [[[ | param params] target] | ] eqn:Hforward;
    try apply forward_apply_cont_keeping_scope;
    destruct (ac_args ac) as [ | arg args];
    try apply forward_apply_cont_keeping_scope.
  destruct (no_trap_action ac); [ | apply forward_apply_cont_keeping_scope ].
  apply forward_of_In in Hforward. split.
  - intros y Hy. apply (forwarded_free [] [] target eq_refl) in Hy.
    apply in_or_app. right. apply in_flat_map.
    exists (ac_continuation ac, Some ([], target)).
    split; [ exact Hforward | exact Hy ].
  - intros j Hj. apply in_or_app. right. apply in_flat_map.
    exists (ac_continuation ac, Some ([], target)).
    split; [ exact Hforward | exact Hj ].
Qed.

(* A call that returns further on has the variables of the call, and
   returns to a continuation of the entry. *)
Lemma forward_apply_scope (F : forwards) (ap : apply_expr) :
  free_variables_of_apply (forward_apply F ap) = free_variables_of_apply ap
  ∧ incl
      [ap_continuation (forward_apply F ap);
       ap_exn_continuation (forward_apply F ap)]
      ([ap_continuation ap; ap_exn_continuation ap]
       ++ flat_map forward_continuations F).
Proof.
  unfold forward_apply, alias_of.
  destruct (forward_of F (ap_continuation ap)) as [[params target] | ]
    eqn:Hforward;
    [ | split; [ reflexivity | apply incl_appl, incl_refl ] ].
  destruct (no_trap_action target && passes_on params (ac_args target));
    [ | split; [ reflexivity | apply incl_appl, incl_refl ] ].
  apply forward_of_In in Hforward. split; [ reflexivity | ].
  cbn [ap_continuation ap_exn_continuation].
  intros j [<- | [<- | []]]; apply in_or_app.
  - right. apply in_flat_map.
    exists (ap_continuation ap, Some (params, target)).
    split; [ exact Hforward | left; reflexivity ].
  - left. right. left. reflexivity.
Qed.

(* A continuation whose applications are forwarded has a handler that
   is one application. *)
Lemma forwarding_some (k : continuation) (params params' : list variable)
    (metadata : handler_metadata) (handler body : expr)
    (target : apply_cont) :
  forwarding k params metadata handler body = Some (params', target) ->
  params' = params ∧ handler = Apply_cont target.
Proof.
  unfold forwarding. destruct handler; try discriminate.
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test; [ discriminate | ]
  end.
  intros Hsome. injection Hsome as <- <-. split; reflexivity.
Qed.

Lemma forwards_none_variables
    (handlers : list (continuation * continuation_handler)) (F : forwards) :
  flat_map forward_variables (map (fun '(k, _) => (k, None)) handlers ++ F)
  = flat_map forward_variables F.
Proof.
  rewrite flat_map_app.
  induction handlers as [ | [k h] handlers IH];
    cbn [map flat_map forward_variables snd app];
    [ reflexivity | exact IH ].
Qed.

Lemma forwards_none_continuations
    (handlers : list (continuation * continuation_handler)) (F : forwards) :
  flat_map forward_continuations
    (map (fun '(k, _) => (k, None)) handlers ++ F)
  = flat_map forward_continuations F.
Proof.
  rewrite flat_map_app.
  induction handlers as [ | [k h] handlers IH];
    cbn [map flat_map forward_continuations snd app];
    [ reflexivity | exact IH ].
Qed.

Fixpoint forward_applications_scoped_among (e : expr) :
  ∀ (calls : bool) (F : forwards),
    scoped_among (flat_map forward_variables F)
      (flat_map forward_continuations F) e
      (forward_applications_under calls F e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros calls F;
    cbn [forward_applications_under forward_applications_in_handler];
    cbv zeta.
  - apply scoped_among_let, forward_applications_scoped_among.
  - apply scoped_among_let_static, forward_applications_scoped_among.
  - pose proof (forward_applications_scoped_among handler calls F)
      as Hhandler.
    pose proof
      (forward_applications_scoped_among body calls
         ((k,
           forwarding k params metadata
             (forward_applications_under calls F handler) body)
          :: F))
      as Hbody.
    cbn [flat_map] in Hbody.
    refine
      (scoped_among_let_cont _ _ _ _ k params metadata _ _ _ _ _ _ Hhandler
         Hbody).
    + unfold forward_variables. cbn [snd].
      destruct
        (forwarding k params metadata
           (forward_applications_under calls F handler) body)
        as [[params' target] | ] eqn:Hforwarding;
        [ | intros y [] ].
      apply forwarding_some in Hforwarding.
      destruct Hforwarding as [-> Hhandler']. rewrite Hhandler'. scope.
      apply incl_refl.
    + unfold forward_continuations. cbn [snd].
      destruct
        (forwarding k params metadata
           (forward_applications_under calls F handler) body)
        as [[params' target] | ] eqn:Hforwarding;
        [ | intros j [] ].
      apply forwarding_some in Hforwarding.
      destruct Hforwarding as [_ Hhandler']. rewrite Hhandler'. scope.
      apply incl_refl.
  - apply scoped_among_let_cont_recursive.
    + rewrite map_map. apply map_ext. intros [k h]. reflexivity.
    + rewrite <- (forwards_none_variables handlers F),
        <- (forwards_none_continuations handlers F).
      set (F' := _ ++ F). clearbody F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        cbn [map]; [ apply scoped_handlers_among_nil | ].
      cbn [forward_applications_under forward_applications_in_handler].
      apply scoped_handlers_among_cons;
        [ apply forward_applications_scoped_among | exact IH ].
    + rewrite <- (forwards_none_variables handlers F),
        <- (forwards_none_continuations handlers F).
      apply forward_applications_scoped_among.
  - destruct calls; [ | apply scoped_among_refl ].
    intros _. scope. destruct (forward_apply_scope F ap) as [Hv Hk].
    split; [ reflexivity | split; [ | exact Hk ] ].
    rewrite Hv. apply incl_appl, incl_refl.
  - intros _. scope.
    destruct (forward_apply_cont_keeping_scope F ac) as [Hv Hk].
    split; [ reflexivity | split; assumption ].
  - intros _. scope. split; [ reflexivity | split ].
    + rewrite <- app_assoc. apply incl_app_app; [ apply incl_refl | ].
      apply
        (incl_flat_map_under
           (fun arm => free_variables_of_apply_cont (snd arm))
           (fun arm => free_variables_of_apply_cont (snd arm))).
      intros arm. cbn [snd]. exact (proj1 (forward_arm_scope F (snd arm))).
    + apply
        (incl_flat_map_under (fun arm => apply_cont_continuations (snd arm))
           (fun arm => apply_cont_continuations (snd arm))).
      intros arm. cbn [snd]. exact (proj2 (forward_arm_scope F (snd arm))).
  - apply scoped_among_refl.
Qed.

Lemma forward_applications_with_scoped (calls : bool) (e : expr) :
  scoped e (forward_applications_with calls e).
Proof.
  apply scoped_among_none.
  exact (forward_applications_scoped_among e calls []).
Qed.

Lemma forward_applications_scoped (e : expr) :
  scoped e (forward_applications e).
Proof. exact (forward_applications_with_scoped false e). Qed.

Theorem forward_applications_keeps_scope :
  forward_applications_keeps_scope_statement.
Proof. apply keeps_scope_intro, forward_applications_scoped. Qed.

Theorem forward_applications_with_keeps_scope :
  forward_applications_with_keeps_scope_statement.
Proof.
  intros calls. apply keeps_scope_intro, forward_applications_with_scoped.
Qed.

(* An application made to go where an arm of the handler of its
   continuation goes (FlambdaForwardArmSpec.v). The variables that an
   application may come to name are those of the arms that are not
   parameters of the handler, and the continuations those of the
   arms. *)
Definition arm_forward_variables (entry : continuation * option arm_entry)
    : list variable :=
  match snd entry with
  | Some (params, _, _, arms, _) =>
    without params
      (flat_map (fun arm => free_variables_of_apply_cont (snd arm)) arms)
  | None => []
  end.

Definition arm_forward_continuations
    (entry : continuation * option arm_entry) : list continuation :=
  match snd entry with
  | Some (_, _, _, arms, _) =>
    flat_map (fun arm => apply_cont_continuations (snd arm)) arms
  | None => []
  end.

(* A continuation that has an entry has a handler that is a [Switch]
   with the arms of the entry. *)
Lemma arm_entry_of_some (kinded : bool) (params params' : list variable)
    (metadata metadata' : handler_metadata) (handler body body' : expr)
    (i : nat) (arms : list (Z * apply_cont)) :
  arm_entry_of kinded params metadata handler body
  = Some (params', metadata', i, arms, body') ->
  params' = params ∧ ∃ p : variable, handler = Switch (Var p) arms.
Proof.
  unfold arm_entry_of. destruct handler; try discriminate.
  match goal with
  | |- match ?s with _ => _ end = _ -> _ => destruct s; try discriminate
  end.
  match goal with
  | |- (if ?test then _ else _) = _ -> _ => destruct test; [ discriminate | ]
  end.
  match goal with
  | |- match ?position with _ => _ end = _ -> _ =>
    destruct position; [ | discriminate ]
  end.
  intros Hsome. injection Hsome as <- _ _ <- _. eauto.
Qed.

Lemma forward_to_arm_apply_cont_scope (kinded : bool) (G : kinds_in_scope)
    (F : arm_forwards) (ac : apply_cont) :
  incl
    (free_variables_of_apply_cont (forward_to_arm_apply_cont kinded G F ac))
    (free_variables_of_apply_cont ac ++ flat_map arm_forward_variables F)
  ∧ incl
      (apply_cont_continuations (forward_to_arm_apply_cont kinded G F ac))
      (apply_cont_continuations ac ++ flat_map arm_forward_continuations F).
Proof.
  unfold forward_to_arm_apply_cont, arm_of_apply_cont.
  destruct (lookup (ac_continuation ac) F)
    as [[[[[[params metadata] i] arms] body] | ] | ] eqn:Hlookup;
    try (split; apply incl_appl, incl_refl).
  destruct (nth_error (ac_args ac) i) as [[x | sym | c] | ];
    try (split; apply incl_appl, incl_refl).
  destruct c as [n | | | | | | | | | | | | | | ];
    try (split; apply incl_appl, incl_refl).
  destruct (find_arm n arms) as [target | ] eqn:Hfind;
    [ | split; apply incl_appl, incl_refl ].
  destruct
    (forwarding (ac_continuation ac) params metadata (Apply_cont target) body)
    as [[params' target'] | ] eqn:Hforwarding;
    [ | split; apply incl_appl, incl_refl ].
  apply forwarding_some in Hforwarding.
  destruct Hforwarding as [-> Htarget]. injection Htarget as <-.
  match goal with
  | |- context [if ?test then _ else _] => destruct test eqn:Htest
  end;
    [ | split; apply incl_appl, incl_refl ].
  apply andb_true_iff in Htest. destruct Htest as [Htest _].
  apply andb_true_iff in Htest. destruct Htest as [Htest _].
  apply andb_true_iff in Htest. destruct Htest as [Hlength _].
  apply Nat.eqb_eq in Hlength. apply lookup_in in Hlookup.
  apply find_arm_In in Hfind. destruct Hfind as [j Hfind]. split.
  - intros y Hy.
    apply (forwarded_free params (ac_args ac) target Hlength), in_app_or
      in Hy.
    apply in_or_app. destruct Hy as [Hy | Hy]; [ left; exact Hy | right ].
    apply in_flat_map.
    exists (ac_continuation ac, Some (params, metadata, i, arms, body)).
    split; [ exact Hlookup | ].
    unfold arm_forward_variables. cbn [snd].
    apply in_without in Hy. destruct Hy as [Hy Hparams].
    apply in_without. split; [ | exact Hparams ].
    apply in_flat_map. exists (j, target). split; [ exact Hfind | exact Hy ].
  - intros j' Hj. apply in_or_app. right. apply in_flat_map.
    exists (ac_continuation ac, Some (params, metadata, i, arms, body)).
    split; [ exact Hlookup | ].
    unfold arm_forward_continuations. cbn [snd].
    apply in_flat_map. exists (j, target). split; [ exact Hfind | exact Hj ].
Qed.

Lemma arm_forwards_none_variables
    (handlers : list (continuation * continuation_handler))
    (F : arm_forwards) :
  flat_map arm_forward_variables
    (map (fun '(k, _) => (k, None)) handlers ++ F)
  = flat_map arm_forward_variables F.
Proof.
  rewrite flat_map_app.
  induction handlers as [ | [k h] handlers IH];
    cbn [map flat_map arm_forward_variables snd app];
    [ reflexivity | exact IH ].
Qed.

Lemma arm_forwards_none_continuations
    (handlers : list (continuation * continuation_handler))
    (F : arm_forwards) :
  flat_map arm_forward_continuations
    (map (fun '(k, _) => (k, None)) handlers ++ F)
  = flat_map arm_forward_continuations F.
Proof.
  rewrite flat_map_app.
  induction handlers as [ | [k h] handlers IH];
    cbn [map flat_map arm_forward_continuations snd app];
    [ reflexivity | exact IH ].
Qed.

Fixpoint forward_to_arms_scoped_among (e : expr) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    scoped_among (flat_map arm_forward_variables F)
      (flat_map arm_forward_continuations F) e
      (forward_to_arms_under kinded G F e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinded G F;
    cbn [forward_to_arms_under forward_to_arms_in_handler];
    cbv zeta.
  - apply scoped_among_let, forward_to_arms_scoped_among.
  - apply scoped_among_let_static, forward_to_arms_scoped_among.
  - pose proof
      (forward_to_arms_scoped_among handler kinded
         (params_in_scope params (hm_params_kinds metadata) G) F)
      as Hhandler.
    pose proof
      (forward_to_arms_scoped_among body kinded G
         ((k,
           arm_entry_of kinded params metadata
             (forward_to_arms_under kinded
                (params_in_scope params (hm_params_kinds metadata) G) F
                handler)
             body)
          :: F))
      as Hbody.
    cbn [flat_map] in Hbody.
    refine
      (scoped_among_let_cont _ _ _ _ k params metadata _ _ _ _ _ _ Hhandler
         Hbody).
    + unfold arm_forward_variables. cbn [snd].
      destruct
        (arm_entry_of kinded params metadata
           (forward_to_arms_under kinded
              (params_in_scope params (hm_params_kinds metadata) G) F
              handler)
           body)
        as [[[[[params' metadata'] i] arms] body'] | ] eqn:Hentry;
        [ | intros y [] ].
      apply arm_entry_of_some in Hentry.
      destruct Hentry as (-> & p & Hhandler'). rewrite Hhandler'. scope.
      intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hparams].
      apply in_without. split; [ | exact Hparams ].
      apply in_or_app. right. exact Hy.
    + unfold arm_forward_continuations. cbn [snd].
      destruct
        (arm_entry_of kinded params metadata
           (forward_to_arms_under kinded
              (params_in_scope params (hm_params_kinds metadata) G) F
              handler)
           body)
        as [[[[[params' metadata'] i] arms] body'] | ] eqn:Hentry;
        [ | intros j [] ].
      apply arm_entry_of_some in Hentry.
      destruct Hentry as (_ & p & Hhandler'). rewrite Hhandler'. scope.
      apply incl_refl.
  - apply scoped_among_let_cont_recursive.
    + rewrite map_map. apply map_ext. intros [k h]. reflexivity.
    + rewrite <- (arm_forwards_none_variables handlers F),
        <- (arm_forwards_none_continuations handlers F).
      set (F' := _ ++ F). clearbody F'.
      induction handlers as [ | [k [params metadata handler]] handlers IH];
        cbn [map]; [ apply scoped_handlers_among_nil | ].
      cbn [forward_to_arms_under forward_to_arms_in_handler].
      apply scoped_handlers_among_cons;
        [ apply forward_to_arms_scoped_among | exact IH ].
    + rewrite <- (arm_forwards_none_variables handlers F),
        <- (arm_forwards_none_continuations handlers F).
      apply forward_to_arms_scoped_among.
  - apply scoped_among_refl.
  - intros _. scope.
    destruct (forward_to_arm_apply_cont_scope kinded G F ac) as [Hv Hk].
    split; [ reflexivity | split; assumption ].
  - intros _. scope. split; [ reflexivity | split ].
    + rewrite <- app_assoc. apply incl_app_app; [ apply incl_refl | ].
      apply
        (incl_flat_map_under
           (fun arm => free_variables_of_apply_cont (snd arm))
           (fun arm => free_variables_of_apply_cont (snd arm))).
      intros arm. cbn [snd].
      exact (proj1 (forward_to_arm_apply_cont_scope kinded G F (snd arm))).
    + apply
        (incl_flat_map_under (fun arm => apply_cont_continuations (snd arm))
           (fun arm => apply_cont_continuations (snd arm))).
      intros arm. cbn [snd].
      exact (proj2 (forward_to_arm_apply_cont_scope kinded G F (snd arm))).
  - apply scoped_among_refl.
Qed.

Lemma forward_to_arms_scoped (e : expr) : scoped e (forward_to_arms e).
Proof.
  apply scoped_among_none.
  exact (forward_to_arms_scoped_among e false [] []).
Qed.

Theorem forward_to_arms_keeps_scope : forward_to_arms_keeps_scope_statement.
Proof. apply keeps_scope_intro, forward_to_arms_scoped. Qed.

(* A continuation whose handler is that of a continuation bound around
   it applies that one. The handler that is put names that continuation
   and its own parameters. *)
Lemma free_variables_of_params (params : list variable) :
  free_variables_of_simples (map Var params) = params.
Proof.
  induction params as [ | param params IH]; [ reflexivity | ].
  unfold free_variables_of_simples in IH |- *.
  cbn [map flat_map free_variables_of_simple app]. rewrite IH. reflexivity.
Qed.

Fixpoint share_with_scoped (k : continuation) (h : continuation_handler)
    (e : expr) {struct e} :
  ∀ (xs : list variable) (ks : list continuation),
    scoped_among [] [k] e (share_with k h xs ks e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k' [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros xs ks; cbn [share_with]; try apply scoped_among_refl.
  - apply scoped_among_let, share_with_scoped.
  - apply scoped_among_let_static, share_with_scoped.
  - destruct (Pos.eqb k' k); [ apply scoped_among_refl | ].
    destruct (same_handler h xs ks (Handler params metadata handler)).
    + pose proof (share_with_scoped k h body xs (k' :: ks)) as Hbody.
      intros Hclosed. scope_in Hclosed. scope.
      apply andb_true_iff in Hclosed. destruct Hclosed as [_ Hrest].
      destruct (Hbody Hrest) as (Hrest' & Hv & Hk).
      rewrite Hrest'. split; [ reflexivity | split ].
      * intros y Hy. apply in_app_or in Hy. destruct Hy as [Hy | Hy].
        -- apply in_without in Hy. destruct Hy as [Hy Hparams].
           unfold free_variables_of_apply_cont in Hy. cbn [ac_args] in Hy.
           rewrite free_variables_of_params in Hy. contradiction.
        -- apply Hv in Hy. rewrite app_nil_r in Hy |- *.
           apply in_or_app. right. exact Hy.
      * intros j Hj. apply in_app_or in Hj. destruct Hj as [Hj | Hj].
        -- cbn in Hj. destruct Hj as [<- | []].
           apply in_or_app. right. left. reflexivity.
        -- apply in_without in Hj. destruct Hj as [Hj Hout].
           apply Hk, in_app_or in Hj. apply in_or_app.
           destruct Hj as [Hj | Hj]; [ left | right; exact Hj ].
           apply in_or_app. right. apply in_without. split; assumption.
    + apply (scoped_among_let_cont [] [] [k] []);
        [ intros y [] | intros j [] | apply scoped_among_refl
        | apply share_with_scoped ].
Qed.

Lemma share_under_scoped (k : continuation) (h : continuation_handler)
    (body : expr) :
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h (share_with k h [] [k] body)).
Proof.
  pose proof (share_with_scoped k h body [] [k]) as Hbody.
  intros Hclosed. scope_in Hclosed. scope.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hfirst Hrest].
  destruct (Hbody Hrest) as (Hrest' & Hv & Hk).
  rewrite Hfirst, Hrest'. split; [ reflexivity | split ].
  - apply incl_app_app; [ apply incl_refl | ].
    rewrite app_nil_r in Hv. exact Hv.
  - apply incl_app_app; [ apply incl_refl | ].
    intros j Hj. apply in_without in Hj. destruct Hj as [Hj Hout].
    apply Hk, in_app_or in Hj.
    destruct Hj as [Hj | Hj]; [ | contradiction ].
    apply in_without. split; assumption.
Qed.

Fixpoint share_handlers_scoped (e : expr) : scoped e (share_handlers e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [share_handlers]; cbv zeta;
    try (scoped_under_binders share_handlers_scoped).
  destruct (sharable (Handler params metadata (share_handlers handler))).
  - eapply scoped_trans; [ | apply share_under_scoped ].
    scoped_under_binders share_handlers_scoped.
  - scoped_under_binders share_handlers_scoped.
Qed.

Theorem share_handlers_keeps_scope : share_handlers_keeps_scope_statement.
Proof. apply keeps_scope_intro, share_handlers_scoped. Qed.

(* Loopify. *)

Lemma without_nil_incl (bound names : list positive) :
  without bound names = [] -> incl names bound.
Proof.
  intros Hnone a Ha. destruct (among a bound) eqn:Hamong;
    [ apply among_In; exact Hamong | ].
  assert (Hin : In a (without bound names)).
  { apply in_without. split; [ exact Ha | ].
    intros Hbound. apply among_In in Hbound. congruence. }
  rewrite Hnone in Hin. destruct Hin.
Qed.

Lemma free_variables_of_vars (xs : list variable) :
  free_variables_of_simples (map Var xs) = xs.
Proof.
  unfold free_variables_of_simples.
  induction xs as [ | x xs IH]; cbn [map flat_map free_variables_of_simple];
    [ reflexivity | rewrite IH; reflexivity ].
Qed.

(* The body of code with its self tail calls made jumps: its code is
   closed as that of the body is, its free variables are free in the
   body, and its free continuations are free in the body or the name of
   the loop. *)
Fixpoint loop_calls_scope (my_closure : variable) (a : arity)
    (ret exn self : continuation) (e : expr) :
  code_closed (loop_calls my_closure a ret exn self e) = code_closed e
  ∧ incl (free_variables_of_expr (loop_calls my_closure a ret exn self e))
      (free_variables_of_expr e)
  ∧ incl (free_continuations_of_expr (loop_calls my_closure a ret exn self e))
      (self :: free_continuations_of_expr e)
with loop_calls_in_handler_scope (my_closure : variable)
    (a : arity) (ret exn self : continuation)
    (h : continuation_handler) :
  handler_code_closed (loop_calls_in_handler my_closure a ret exn self h)
  = handler_code_closed h
  ∧ incl
      (free_variables_of_handler
         (loop_calls_in_handler my_closure a ret exn self h))
      (free_variables_of_handler h)
  ∧ incl
      (free_continuations_of_handler
         (loop_calls_in_handler my_closure a ret exn self h))
      (self :: free_continuations_of_handler h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [loop_calls loop_calls_in_handler].
    + destruct (loop_calls_scope my_closure a ret exn self body)
        as (Hc & Hv & Hk).
      scope. split; [ exact Hc | split; [ | exact Hk ] ].
      apply incl_app_app; [ apply incl_refl | apply incl_without, Hv ].
    + destruct (loop_calls_scope my_closure a ret exn self body)
        as (Hc & Hv & Hk).
      scope. rewrite Hc. split; [ reflexivity | split ].
      * apply incl_app_app; [ apply incl_refl | exact Hv ].
      * intros j Hj. apply in_app_or in Hj.
        destruct Hj as [Hj | Hj]; [ right; apply in_or_app; left; exact Hj | ].
        destruct (Hk j Hj) as [<- | Hj'];
          [ left; reflexivity | right; apply in_or_app; right; exact Hj' ].
    + destruct (loop_calls_in_handler_scope my_closure a ret exn self h)
        as (Hc & Hv & Hk).
      destruct (loop_calls_scope my_closure a ret exn self body)
        as (Hc' & Hv' & Hk').
      scope. rewrite Hc, Hc'. split; [ reflexivity | split ].
      * apply incl_app_app; assumption.
      * intros j Hj. apply in_app_or in Hj. destruct Hj as [Hj | Hj].
        -- destruct (Hk j Hj) as [<- | Hj'];
             [ left; reflexivity | right; apply in_or_app; left; exact Hj' ].
        -- apply in_without in Hj. destruct Hj as [Hj Hout].
           destruct (Hk' j Hj) as [<- | Hj']; [ left; reflexivity | right ].
           apply in_or_app. right. apply in_without. split; assumption.
    + destruct (loop_calls_scope my_closure a ret exn self body)
        as (Hc' & Hv' & Hk').
      assert (Hgroup :
        map fst
          (map
             (fun '(k, h) =>
                (k, loop_calls_in_handler my_closure a ret exn self h))
             handlers)
        = map fst handlers
        ∧ forallb (fun '(_, h) => handler_code_closed h)
            (map
               (fun '(k, h) =>
                  (k, loop_calls_in_handler my_closure a ret exn self h))
               handlers)
          = forallb (fun '(_, h) => handler_code_closed h) handlers
        ∧ incl
            (flat_map (fun '(_, h) => free_variables_of_handler h)
               (map
                  (fun '(k, h) =>
                     (k, loop_calls_in_handler my_closure a ret exn self h))
                  handlers))
            (flat_map (fun '(_, h) => free_variables_of_handler h) handlers)
        ∧ incl
            (flat_map (fun '(_, h) => free_continuations_of_handler h)
               (map
                  (fun '(k, h) =>
                     (k, loop_calls_in_handler my_closure a ret exn self h))
                  handlers))
            (self
             :: flat_map (fun '(_, h) => free_continuations_of_handler h)
                  handlers)).
      { induction handlers as [ | [k h] handlers IH];
          cbn [map fst forallb flat_map].
        - split; [ reflexivity | split; [ reflexivity | split ] ];
            intros j [].
        - destruct IH as (Hnames & Hc & Hv & Hk).
          destruct (loop_calls_in_handler_scope my_closure a ret exn self h)
            as (Hc0 & Hv0 & Hk0).
          rewrite Hnames, Hc, Hc0.
          split; [ reflexivity | split; [ reflexivity | split ] ].
          + apply incl_app_app; assumption.
          + intros j Hj. apply in_app_or in Hj. destruct Hj as [Hj | Hj].
            * destruct (Hk0 j Hj) as [<- | Hj'];
                [ left; reflexivity
                | right; apply in_or_app; left; exact Hj' ].
            * destruct (Hk j Hj) as [<- | Hj'];
                [ left; reflexivity
                | right; apply in_or_app; right; exact Hj' ]. }
      destruct Hgroup as (Hnames & Hc & Hv & Hk).
      scope. rewrite Hnames, Hc, Hc'. split; [ reflexivity | split ].
      * apply incl_app_app; assumption.
      * intros j Hj. apply in_without in Hj. destruct Hj as [Hj Hout].
        apply in_app_or in Hj. destruct Hj as [Hj | Hj].
        -- destruct (Hk j Hj) as [<- | Hj']; [ left; reflexivity | right ].
           apply in_without.
           split; [ apply in_or_app; left; exact Hj' | exact Hout ].
        -- destruct (Hk' j Hj) as [<- | Hj']; [ left; reflexivity | right ].
           apply in_without.
           split; [ apply in_or_app; right; exact Hj' | exact Hout ].
    + destruct (is_self_tail_call my_closure a ret exn ap).
      * scope. split; [ reflexivity | split ].
        -- unfold free_variables_of_apply_cont, free_variables_of_apply.
           cbn [ac_args]. apply incl_appr, incl_appl, incl_refl.
        -- cbn [apply_cont_continuations ac_continuation ac_trap_action].
           intros j [<- | []]. left. reflexivity.
      * split; [ reflexivity | split; [ apply incl_refl | ] ].
        apply incl_tl, incl_refl.
    + split; [ reflexivity | split; [ apply incl_refl | ] ].
      apply incl_tl, incl_refl.
    + split; [ reflexivity | split; [ apply incl_refl | ] ].
      apply incl_tl, incl_refl.
    + split; [ reflexivity | split; [ apply incl_refl | ] ].
      apply incl_tl, incl_refl.
  - destruct h as [params metadata handler].
    cbn [loop_calls loop_calls_in_handler].
    destruct (loop_calls_scope my_closure a ret exn self handler)
      as (Hc & Hv & Hk).
    scope. split; [ exact Hc | split; [ apply incl_without, Hv | exact Hk ] ].
Qed.

(* Code whose body is made the handler of a loop that is entered with
   the parameters. *)
Lemma scoped_code_looped (params : list variable) (my_closure : variable)
    (ret exn self : continuation) (metadata : code_metadata)
    (loop_metadata : handler_metadata) (body looped : expr) :
  code_closed looped = code_closed body ->
  incl (free_variables_of_expr looped) (free_variables_of_expr body) ->
  incl (free_continuations_of_expr looped)
    (self :: free_continuations_of_expr body) ->
  scoped_code (Code params my_closure ret exn metadata body)
    (Code params my_closure ret exn metadata
       (Let_cont_recursive [(self, Handler params loop_metadata looped)]
          (Apply_cont (Apply_cont_expr self (map Var params) None)))).
Proof.
  intros Hc Hv Hk Hclosed. scope_in Hclosed.
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hclosed Hbody].
  apply andb_true_iff in Hclosed. destruct Hclosed as [Hvars Hconts].
  apply is_empty_nil, without_nil_incl in Hvars.
  apply is_empty_nil, without_nil_incl in Hconts.
  scope. cbn [forallb flat_map map fst]. scope.
  cbn [apply_cont_continuations ac_continuation ac_trap_action].
  unfold free_variables_of_apply_cont. cbn [ac_args].
  rewrite free_variables_of_vars, Hc, Hbody, !app_nil_r.
  apply andb_true_iff. split; [ apply andb_true_iff; split | reflexivity ].
  - apply (incl_is_empty _ []); [ | reflexivity ].
    intros y Hy. apply in_without in Hy. destruct Hy as [Hy Hbound].
    destruct Hbound. apply in_app_or in Hy. destruct Hy as [Hy | Hy].
    + apply in_without in Hy. apply Hvars, Hv, (proj1 Hy).
    + right. exact Hy.
  - apply (incl_is_empty _ []); [ | reflexivity ].
    intros j Hj. apply in_without in Hj. destruct Hj as [Hj Hbound].
    destruct Hbound. apply in_without in Hj. destruct Hj as [Hj Hself].
    apply in_app_or in Hj.
    destruct Hj as [Hj | [<- | []]];
      [ | destruct Hself; left; reflexivity ].
    destruct (Hk j Hj) as [<- | Hj'];
      [ destruct Hself; left; reflexivity | exact (Hconts j Hj') ].
Qed.

Lemma loopify_code_scoped (self : continuation) (c : code) :
  scoped_code c (loopify_code self c).
Proof.
  unfold loopify_code.
  destruct (loopable self c); [ | intros Hclosed; exact Hclosed ].
  destruct c as [params my_closure ret exn metadata body].
  unfold looped_code, loop_handler.
  cbn [code_params code_my_closure code_return_continuation
       code_exn_continuation code_body].
  destruct
    (loop_calls_scope my_closure (cm_params_arity metadata) ret exn self
       body)
    as (Hc & Hv & Hk).
  apply scoped_code_looped; assumption.
Qed.

Lemma loopify_bindings_scoped (G : loopified)
    (bindings : list static_binding) :
  scoped_bindings bindings (map (loopify_binding G) bindings).
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map];
    [ apply scoped_bindings_nil | ].
  destruct binding as [id c | id | sym const]; cbn [loopify_binding];
    [ | apply scoped_bindings_cons_same, IH
    | apply scoped_bindings_cons_same, IH ].
  destruct (lookup id G) as [self | ];
    [ | apply scoped_bindings_cons_same, IH ].
  apply scoped_bindings_cons_code; [ apply loopify_code_scoped | exact IH ].
Qed.

Fixpoint loopify_unit_scoped (G : loopified) (e : expr) :
  scoped e (loopify_unit G e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [loopify_unit];
    first
      [ apply scoped_let_static;
        [ apply loopify_bindings_scoped | apply loopify_unit_scoped ]
      | scoped_under_binders (loopify_unit_scoped G) ].
Qed.

Theorem loopify_unit_keeps_scope : loopify_unit_keeps_scope_statement.
Proof.
  intros G. apply keeps_scope_intro. intros e. apply loopify_unit_scoped.
Qed.
