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

(* Proofs of the statements of FlambdaValidateScopeSpec.v. Nothing here
   needs review. A round is a chain of rewrites, each of which keeps
   scope by a lemma of FlambdaKeepsScopeProofs.v or of
   FlambdaLoopsScopeProofs.v. The chain is taken apart by a tactic that
   finds those lemmas among its hints, so a rewrite that is added to a
   round, or moved in it, needs its lemma among the hints and nothing
   else. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaBindersSpec
  FlambdaBindersProofs FlambdaNominalSpec FlambdaNominalProofs
  FlambdaAlphaSpec FlambdaAlphaProofs FlambdaEquivarianceProofs
  FlambdaFreeNamesSpec FlambdaFreeNamesProofs FlambdaApplySpec
  FlambdaTupledSpec FlambdaInlineSiblingsSpec FlambdaGuardScopeSpec
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec
  FlambdaCseSpec FlambdaIdentitiesSpec FlambdaRenameApartSpec
  FlambdaUnboxSpec FlambdaUnusedContsSpec FlambdaForwardSpec
  FlambdaLoopAliasesSpec FlambdaLoopLetsSpec FlambdaParamAliasesSpec
  FlambdaRecHandlersSpec FlambdaShortcutSpec FlambdaSameHandlersSpec
  FlambdaSinkContsSpec
  FlambdaUnusedParamsSpec FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaKeepsScopeProofs FlambdaLoopsScopeSpec FlambdaLoopsScopeProofs
  FlambdaUnboxVariantsSpec FlambdaValidateScopeSpec.
From OxCamlFlambda2 Require FlambdaUnboxParamsNamesProofs.
From OxCamlFlambda2 Require Import FlambdaUnboxParamsSpec
  FlambdaUnboxInScopeSpec FlambdaUnboxInScopeProofs
  FlambdaUnboxInScopeRenameSpec FlambdaUnboxInScopeRenameProofs
  FlambdaUnboxLoopsSpec FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require Import FlambdaDemoteSpec.
From OxCamlFlambda2 Require FlambdaDemoteScopeProofs.
From OxCamlFlambda2 Require Import FlambdaShareProjectionsSpec
  FlambdaShareProjectionsScopeProofs.
From OxCamlFlambda2 Require Import FlambdaWrapReturnsSpec
  FlambdaWrapReturnsNamesProofs.
From OxCamlFlambda2 Require Import FlambdaUntagParamsSpec
  FlambdaUntagParamsNamesProofs.
From OxCamlFlambda2 Require Import FlambdaRecoverComparisonSpec
  FlambdaRecoverComparisonScopeProofs.
From OxCamlFlambda2 Require Import FlambdaSameComparisonSpec
  FlambdaSameComparisonScopeProofs.
From OxCamlFlambda2 Require Import FlambdaMergeTablesSpec
  FlambdaMergeTablesScopeProofs.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
From OxCamlFlambda2 Require FlambdaShadowParamsNamesProofs
  FlambdaTwinParamsNamesProofs.
Import ListNotations.

Lemma rename_apart_scoped (e : expr) : scoped e (rename_apart e).
Proof. apply keeps_scope_scoped, rename_apart_keeps_scope. Qed.

Lemma rename_apart_avoiding_scoped (avoid : list continuation) (e : expr) :
  scoped e (rename_apart_avoiding avoid e).
Proof. apply keeps_scope_scoped, rename_apart_avoiding_keeps_scope. Qed.

Lemma drop_unused_params_scoped (e : expr) : scoped e (drop_unused_params e).
Proof. apply keeps_scope_scoped, drop_unused_params_keeps_scope. Qed.

Lemma unbox_variant_params_scoped (e : expr) :
  scoped e (unbox_variant_params e).
Proof.
  apply keeps_scope_scoped.
  exact FlambdaUnboxParamsNamesProofs.unbox_variant_params_keeps_scope.
Qed.

(* The lemmas that say of a rewrite that its result keeps the scope of
   the term. A rewrite that takes another asks that the other keeps
   scope, which [keeps_scope_intro] makes of a lemma of this form. The
   hints go to the files that import this one, with the tactic. *)
Create HintDb scoped.

#[export] Hint Resolve keeps_scope_intro rename_apart_scoped
  rename_apart_avoiding_scoped
  fold_constants_scoped collapse_switches_scoped merge_switches_scoped
  merge_tables_scoped
  remove_unused_conts_scoped
  remove_unused_lets_scoped remove_unused_defined_lets_scoped
  apply_identities_scoped inline_tail_conts_scoped thread_switches_scoped
  drop_deleted_code_scoped merge_statics_scoped rewrite_code_scoped
  load_known_blocks_scoped unbox_known_numbers_scoped
  share_known_primitives_scoped follow_aliases_scoped sink_let_conts_scoped
  shortcut_switches_scoped share_handlers_scoped
  follow_aliases_in_loops_scoped
  drop_unused_params_scoped drop_dead_params_scoped
  rewrite_rec_handlers_scoped bind_aliased_params_scoped
  forward_applications_scoped remove_unused_defined_lets_in_loops_scoped
  unbox_variant_params_scoped share_known_projections_scoped
  recover_known_comparisons_scoped fold_same_comparisons_scoped
  wrap_returns_scoped untag_params_scoped
  : scoped.

(* A chain of rewrites, taken apart from the outside. *)
Ltac scoped_chain :=
  repeat
    first
      [ apply scoped_refl
      | eapply scoped_trans; [ | solve [ eauto with scoped ] ] ].

Lemma normalise_expr_before_scoped (e : expr) :
  scoped e (normalise_expr_before e).
Proof. unfold normalise_expr_before. cbv zeta. scoped_chain. Qed.

#[export] Hint Resolve normalise_expr_before_scoped : scoped.

Theorem normalise_expr_before_keeps_scope :
  normalise_expr_before_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_expr_before_scoped. Qed.

Lemma normalise_code_before_scoped (e : expr) :
  scoped e (normalise_code_before e).
Proof. unfold normalise_code_before. scoped_chain. Qed.

#[export] Hint Resolve normalise_code_before_scoped : scoped.

Theorem normalise_code_before_keeps_scope :
  normalise_code_before_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_code_before_scoped. Qed.

(* The unboxing given what is known of symbols alone, and the round of
   the first phase given it. *)
Lemma unbox_variants_knowing_symbols_scoped (known : facts) (e : expr) :
  symbols_only known -> scoped e (unbox_variants_knowing known e).
Proof.
  intros Honly. apply keeps_scope_scoped.
  exact (unbox_variants_knowing_symbols_keeps_scope known Honly).
Qed.

#[export] Hint Resolve unbox_variants_knowing_symbols_scoped : scoped.

Lemma unbox_loops_knowing_symbols_scoped (known : facts) (e : expr) :
  symbols_only known -> scoped e (unbox_loops_knowing known e).
Proof.
  intros Honly. apply keeps_scope_scoped.
  exact (unbox_loops_knowing_symbols_keeps_scope known Honly).
Qed.

#[export] Hint Resolve unbox_loops_knowing_symbols_scoped : scoped.

Lemma unbox_together_knowing_symbols_scoped (known : facts) (e : expr) :
  symbols_only known -> scoped e (unbox_together_knowing known e).
Proof.
  intros Honly. apply keeps_scope_scoped.
  exact (unbox_together_knowing_symbols_keeps_scope known Honly).
Qed.

#[export] Hint Resolve unbox_together_knowing_symbols_scoped : scoped.

Lemma normalise_expr_params_knowing_scoped (known : facts) (e : expr) :
  symbols_only known -> scoped e (normalise_expr_params_knowing known e).
Proof.
  intros Honly. unfold normalise_expr_params_knowing. cbv zeta.
  scoped_chain.
Qed.

Lemma normalise_expr_params_scoped (e : expr) :
  scoped e (normalise_expr_params e).
Proof. apply normalise_expr_params_knowing_scoped, no_facts_only. Qed.

#[export] Hint Resolve normalise_expr_params_scoped : scoped.

Theorem normalise_expr_params_keeps_scope :
  normalise_expr_params_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_expr_params_scoped. Qed.

Lemma normalise_code_params_knowing_keeps_scope :
  keeps_scope_where_known normalise_code_params_knowing.
Proof.
  intros known Honly. apply keeps_scope_intro. intros e.
  unfold normalise_code_params_knowing.
  eapply scoped_trans;
    [ | apply normalise_expr_params_knowing_scoped, Honly ].
  apply keeps_scope_scoped.
  apply (rewrite_rec_handlers_in_scope_keeps_scope
    normalise_expr_params_knowing); [ | exact Honly ].
  intros known' Honly'. apply keeps_scope_intro. intros e'.
  apply normalise_expr_params_knowing_scoped, Honly'.
Qed.

Lemma normalise_code_params_scoped (e : expr) :
  scoped e (normalise_code_params e).
Proof.
  apply keeps_scope_scoped.
  exact (normalise_code_params_knowing_keeps_scope no_facts no_facts_only).
Qed.

#[export] Hint Resolve normalise_code_params_scoped : scoped.

Theorem normalise_code_params_keeps_scope :
  normalise_code_params_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_code_params_scoped. Qed.

Lemma normalise_params_round_scoped (e : expr) :
  scoped e (normalise_params_round e).
Proof.
  unfold normalise_params_round.
  eapply scoped_trans; [ apply merge_tables_scoped | ].
  eapply scoped_trans; [ apply normalise_code_params_scoped | ].
  apply keeps_scope_scoped.
  exact (rewrite_code_in_scope_keeps_scope _
    normalise_code_params_knowing_keeps_scope no_facts no_facts_only).
Qed.

#[export] Hint Resolve normalise_params_round_scoped : scoped.

Theorem normalise_params_round_keeps_scope :
  normalise_params_round_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_params_round_scoped. Qed.

(* The demotion of exception handlers, with whatever names in scope
   (FlambdaDemoteScopeProofs), and as the validation applies it. *)
Lemma demote_handlers_scoped (scope : list continuation) (e : expr) :
  scoped e (demote_handlers scope e).
Proof. apply FlambdaDemoteScopeProofs.scoped_demote_handlers. Qed.

#[export] Hint Resolve demote_handlers_scoped : scoped.

Lemma demote_handlers_closed_keeps_scope : keeps_scope (demote_handlers []).
Proof. apply keeps_scope_intro, demote_handlers_scoped. Qed.

(* ======================================================================== *)
(* A round of the term before                                               *)
(* ======================================================================== *)

(* Inlining the calls of known code. *)

(* The conjuncts of the tests among the hypotheses, each a hypothesis
   of its own. *)
Ltac split_tests :=
  repeat
    match goal with
    | H : _ && _ = true |- _ =>
      let H' := fresh "Htest" in
      apply andb_true_iff in H; destruct H as [H H']
    end.

(* The body of tupled code, its parameters bound to the fields of the
   tuple, in place of a call that gives it the tuple, keeps the scope of
   the call. *)
Lemma inlined_tupled_call_scoped (ap : apply_expr) (c : code) :
  tupled_inlinable_b ap c = true ->
  scoped (Apply ap)
    (inlined_tupled_call c
       (inline_renaming (code_return_continuation c)
          (code_exn_continuation c) (ap_continuation ap)
          (ap_exn_continuation ap))
       ap).
Proof.
  intros Hinline. unfold inlined_tupled_call.
  destruct (ap_args ap) as [ | tuple [ | arg args]] eqn:Hargs;
    try apply scoped_refl.
  unfold tupled_inlinable_b in Hinline. split_tests.
  match goal with
  | H : expr_in_scope _ _ _ = true |- _ => rename H into Hscope
  end.
  match goal with
  | H : negb (binds_statics _) = true |- _ => rename H into Hstatics
  end.
  match goal with
  | H : negb (Pos.eqb (code_return_continuation _) _) = true |- _ =>
    rename H into Hcode
  end.
  match goal with
  | H : negb (Pos.eqb (ap_continuation _) _) = true |- _ =>
    rename H into Hcall
  end.
  apply negb_true_iff in Hstatics, Hcall, Hcode.
  apply Pos.eqb_neq in Hcall, Hcode.
  pose proof (binds_statics_code_closed (code_body c) Hstatics) as Hclosed.
  rewrite (expr_in_scope_agrees _ _ _ Hclosed) in Hscope.
  apply andb_true_iff in Hscope. destruct Hscope as [Hscope_v Hscope_k].
  destruct
    (inline_renaming_spec (code_return_continuation c)
       (code_exn_continuation c) (ap_continuation ap)
       (ap_exn_continuation ap) Hcode Hcall)
    as (Hfixes & Hret & Hexn).
  set (r :=
    inline_renaming (code_return_continuation c) (code_exn_continuation c)
      (ap_continuation ap) (ap_exn_continuation ap))
    in *.
  intros _. unfold inlined_tupled_body. scope.
  rewrite let_tuple_fields_code_closed, code_closed_renamed.
  split; [ exact Hclosed | split ].
  - intros x Hx. cbn [free_variables_of_named] in Hx.
    unfold free_variables_of_apply. apply in_or_app.
    apply in_app_or in Hx. destruct Hx as [Hx | Hx]; [ left; exact Hx | ].
    right. apply in_or_app. left. rewrite Hargs.
    apply in_without in Hx. destruct Hx as [Hx Hclosure].
    apply let_tuple_fields_free_variables in Hx.
    destruct Hx as [Hx | [Hx Hparams]]; [ exact Hx | ].
    rewrite free_variables_of_expr_renamed in Hx. apply in_map_iff in Hx.
    destruct Hx as (y & Hy & Hin). rewrite Hfixes in Hy. subst y.
    destruct (all_among_In _ _ x Hscope_v Hin) as [Heq | Hin'];
      [ destruct Hclosure; left; exact Heq | contradiction ].
  - intros k Hk. rewrite let_tuple_fields_free_continuations in Hk.
    rewrite free_continuations_of_expr_renamed in Hk. apply in_map_iff in Hk.
    destruct Hk as (j & Hj & Hin).
    destruct (all_among_In _ _ j Hscope_k Hin) as [Heq | [Heq | []]];
      subst j; [ left | right; left ]; congruence.
Qed.

(* A call split into the full call and the call of its result keeps
   the scope of the call: the two calls have its callee, its
   arguments, its extra arguments and its two continuations between
   them, and the continuation and the variable that the term binds
   are bound. *)
Lemma split_over_application_scoped (called : code_id) (c : code)
    (ap : apply_expr) :
  scoped (Apply ap) (split_over_application called c ap).
Proof.
  intros _. unfold split_over_application. scope.
  split; [ reflexivity | split ].
  - intros x Hx. unfold free_variables_of_apply, exn_extra_simples in *.
    cbn [ap_callee ap_args ap_exn_extra_args] in Hx.
    rewrite <- (firstn_skipn (cardinal_unarized (code_params_arity c))
                  (ap_args ap)) at 1.
    unfold free_variables_of_simples in *. rewrite flat_map_app.
    apply in_app_or in Hx. destruct Hx as [Hx | Hx].
    + apply in_without in Hx. destruct Hx as [Hx Hbound].
      cbn [flat_map free_variables_of_simple app] in Hx.
      destruct Hx as [Hx | Hx].
      * destruct Hbound. left. exact Hx.
      * apply in_app_or in Hx. destruct Hx as [Hx | Hx].
        -- apply in_or_app. right. apply in_or_app. left.
           apply in_or_app. right. exact Hx.
        -- apply in_or_app. right. apply in_or_app. right. exact Hx.
    + apply in_app_or in Hx. destruct Hx as [Hx | Hx];
        [ apply in_or_app; left; exact Hx | ].
      apply in_app_or in Hx. destruct Hx as [Hx | Hx].
      * apply in_or_app. right. apply in_or_app. left.
        apply in_or_app. left. exact Hx.
      * apply in_or_app. right. apply in_or_app. right. exact Hx.
  - intros k Hk. cbn [ap_continuation ap_exn_continuation] in Hk.
    apply in_app_or in Hk. destruct Hk as [Hk | Hk]; [ exact Hk | ].
    apply in_without in Hk. destruct Hk as [Hk Hbound].
    destruct Hk as [Hk | [Hk | []]].
    + destruct Hbound. left. exact Hk.
    + right. left. exact Hk.
Qed.

(* The body of code in place of a call of it keeps the scope of the
   call: the test says that the free variables and continuations of the
   body are bound by the code. *)
Lemma inline_apply_scoped (kept : list code_id)
    (known : list (code_id * code)) (ap : apply_expr) :
  scoped (Apply ap) (inline_apply kept known ap).
Proof.
  unfold inline_apply.
  destruct (ap_call_kind ap) as [called | ]; [ | apply scoped_refl ].
  destruct (lookup called known) as [c | ]; [ | apply scoped_refl ].
  destruct (negb (among called kept) && inlinable_b ap c) eqn:Hinline;
    [ | destruct (negb (among called kept) && tupled_inlinable_b ap c)
          eqn:Htupled;
        [ | destruct (over_applied_b ap c);
            [ apply split_over_application_scoped | apply scoped_refl ] ];
      apply andb_true_iff in Htupled; destruct Htupled as [_ Htupled];
      exact (inlined_tupled_call_scoped ap c Htupled) ].
  unfold inlinable_b in Hinline. split_tests.
  match goal with
  | H : expr_in_scope _ _ _ = true |- _ => rename H into Hscope
  end.
  match goal with
  | H : negb (binds_statics _) = true |- _ => rename H into Hstatics
  end.
  match goal with
  | H : Nat.eqb (length (code_params _)) _ = true |- _ =>
    rename H into Hlength
  end.
  match goal with
  | H : negb (Pos.eqb (code_return_continuation _) _) = true |- _ =>
    rename H into Hcode
  end.
  match goal with
  | H : negb (Pos.eqb (ap_continuation _) _) = true |- _ =>
    rename H into Hcall
  end.
  apply negb_true_iff in Hstatics, Hcall, Hcode.
  apply Pos.eqb_neq in Hcall, Hcode. apply Nat.eqb_eq in Hlength.
  pose proof (binds_statics_code_closed (code_body c) Hstatics) as Hclosed.
  rewrite (expr_in_scope_agrees _ _ _ Hclosed) in Hscope.
  apply andb_true_iff in Hscope. destruct Hscope as [Hscope_v Hscope_k].
  destruct
    (inline_renaming_spec (code_return_continuation c)
       (code_exn_continuation c) (ap_continuation ap)
       (ap_exn_continuation ap) Hcode Hcall)
    as (Hfixes & Hret & Hexn).
  set (r :=
    inline_renaming (code_return_continuation c) (code_exn_continuation c)
      (ap_continuation ap) (ap_exn_continuation ap))
    in *.
  intros _. unfold inlined_body. scope.
  rewrite let_params_code_closed, code_closed_renamed.
  split; [ exact Hclosed | split ].
  - intros x Hx. cbn [free_variables_of_named] in Hx.
    unfold free_variables_of_apply. apply in_or_app.
    apply in_app_or in Hx. destruct Hx as [Hx | Hx]; [ left; exact Hx | ].
    right. apply in_or_app. left.
    apply in_without in Hx. destruct Hx as [Hx Hclosure].
    apply let_params_free_variables in Hx; [ | exact Hlength ].
    destruct Hx as [Hx | [Hx Hparams]]; [ exact Hx | ].
    rewrite free_variables_of_expr_renamed in Hx. apply in_map_iff in Hx.
    destruct Hx as (y & Hy & Hin). rewrite Hfixes in Hy. subst y.
    destruct (all_among_In _ _ x Hscope_v Hin) as [Heq | Hin'];
      [ destruct Hclosure; left; exact Heq | contradiction ].
  - intros k Hk. rewrite let_params_free_continuations in Hk.
    rewrite free_continuations_of_expr_renamed in Hk. apply in_map_iff in Hk.
    destruct Hk as (j & Hj & Hin).
    destruct (all_among_In _ _ j Hscope_k Hin) as [Heq | [Heq | []]];
      subst j; [ left | right; left ]; congruence.
Qed.

Fixpoint inline_calls_scoped (e : expr) :
  ∀ (kept : list code_id) (known : list (code_id * code)),
    scoped e (inline_calls kept known e)
with inline_calls_in_handler_scoped (h : continuation_handler) :
  ∀ (kept : list code_id) (known : list (code_id * code)),
    scoped_handler h (inline_calls_in_handler kept known h)
with inline_calls_in_code_scoped (c : code) :
  ∀ (kept : list code_id) (known : list (code_id * code)),
    scoped_code c (inline_calls_in_code kept known c).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros kept known;
      cbn [inline_calls inline_calls_in_handler inline_calls_in_binding
           inline_calls_in_code].
    + apply scoped_let; [ apply incl_refl | apply inline_calls_scoped ].
    + apply scoped_let_static; [ | apply inline_calls_scoped ].
      induction bindings as [ | binding bindings IH]; cbn [map];
        [ apply scoped_bindings_nil | ].
      destruct binding as [id c | id | sym const];
        cbn [inline_calls inline_calls_in_handler inline_calls_in_binding
             inline_calls_in_code];
        [ apply scoped_bindings_cons_code;
          [ apply inline_calls_in_code_scoped | exact IH ]
        | apply scoped_bindings_cons_same, IH
        | apply scoped_bindings_cons_same, IH ].
    + apply scoped_let_cont;
        [ apply inline_calls_in_handler_scoped | apply inline_calls_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply inline_calls_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply inline_calls_in_handler_scoped | exact IH ].
    + apply inline_apply_scoped.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros kept known.
    cbn [inline_calls inline_calls_in_handler inline_calls_in_binding
         inline_calls_in_code].
    apply scoped_handler_intro, inline_calls_scoped.
  - destruct c as [params my_closure ret exn metadata body].
    intros kept known.
    cbn [inline_calls inline_calls_in_handler inline_calls_in_binding
         inline_calls_in_code].
    apply scoped_code_intro, inline_calls_scoped.
Qed.

Theorem inline_calls_keeps_scope : inline_calls_keeps_scope_statement.
Proof.
  intros kept. apply keeps_scope_intro. intros e. apply inline_calls_scoped.
Qed.

(* Inlining the calls of the other code of a group. *)

(* The bindings of a group after others that are the same. *)
Lemma scoped_bindings_app_same (done : list static_binding) :
  ∀ bindings bindings' : list static_binding,
    scoped_bindings bindings bindings' ->
    scoped_bindings (done ++ bindings) (done ++ bindings').
Proof.
  induction done as [ | binding done IH]; intros bindings bindings' Hscoped;
    [ exact Hscoped | ].
  cbn [app]. apply scoped_bindings_cons_same, IH, Hscoped.
Qed.

(* One binding with calls inlined in the body of its code. *)
Lemma inline_calls_in_binding_scoped (kept : list code_id)
    (known : list (code_id * code)) (binding : static_binding)
    (bindings : list static_binding) :
  scoped_bindings (binding :: bindings)
    (inline_calls_in_binding kept known binding :: bindings).
Proof.
  destruct binding as [id c | id | sym const]; cbn [inline_calls_in_binding];
    [ | apply scoped_bindings_refl | apply scoped_bindings_refl ].
  apply scoped_bindings_cons_code;
    [ apply inline_calls_in_code_scoped | apply scoped_bindings_refl ].
Qed.

Lemma inline_in_group_scoped (kept : list code_id)
    (todo : list static_binding) :
  ∀ done : list static_binding,
    scoped_bindings (done ++ todo) (inline_in_group kept done todo).
Proof.
  induction todo as [ | binding todo IH]; intros done; cbn [inline_in_group].
  - rewrite app_nil_r. apply scoped_bindings_refl.
  - eapply scoped_bindings_trans; [ | apply IH ].
    rewrite <- app_assoc. cbn [app].
    apply scoped_bindings_app_same, inline_calls_in_binding_scoped.
Qed.

Fixpoint inline_sibling_calls_scoped (kept : list code_id) (e : expr) :
  scoped e (inline_sibling_calls kept e).
Proof.
  destruct e
    as [x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [inline_sibling_calls]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply inline_sibling_calls_scoped ].
  - apply scoped_let_static; [ | apply inline_sibling_calls_scoped ].
    exact (inline_in_group_scoped kept bindings []).
  - apply scoped_let_cont; [ | apply inline_sibling_calls_scoped ].
    apply scoped_handler_intro, inline_sibling_calls_scoped.
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply inline_sibling_calls_scoped ].
Qed.

Theorem inline_sibling_calls_keeps_scope :
  inline_sibling_calls_keeps_scope_statement.
Proof.
  intros kept. apply keeps_scope_intro. intros e.
  apply inline_sibling_calls_scoped.
Qed.

#[export] Hint Resolve inline_calls_scoped inline_sibling_calls_scoped
  : scoped.

(* The round once the calls are inlined, and the round as a whole. *)

Lemma normalise_before_inlined_scoped (e : expr) :
  scoped e (normalise_before_inlined e).
Proof. unfold normalise_before_inlined. scoped_chain. Qed.

#[export] Hint Resolve normalise_before_inlined_scoped : scoped.

Theorem normalise_before_inlined_keeps_scope :
  normalise_before_inlined_keeps_scope_statement.
Proof. apply keeps_scope_intro, normalise_before_inlined_scoped. Qed.

(* The twins, where a round has settled the term. *)
#[export] Hint Resolve FlambdaShadowParamsNamesProofs.shadow_params_scoped
  FlambdaTwinParamsNamesProofs.twin_params_scoped : scoped.

Lemma with_twins_scoped (avoid : list continuation) (e : expr) :
  scoped e (with_twins avoid e).
Proof.
  unfold with_twins. cbv zeta.
  destruct (FlambdaCompareSpec.expr_eqb [] [] _ _);
    [ apply scoped_refl | scoped_chain ].
Qed.

#[export] Hint Resolve with_twins_scoped : scoped.

Lemma with_twins_keeps_scope (avoid : list continuation) :
  keeps_scope (with_twins avoid).
Proof. apply keeps_scope_intro, with_twins_scoped. Qed.

Lemma normalise_before_round_avoiding_scoped (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  scoped e (normalise_before_round_avoiding avoid kept e).
Proof.
  unfold normalise_before_round_avoiding. cbv zeta.
  destruct (FlambdaCompareSpec.expr_eqb [] [] e _); scoped_chain.
Qed.

Lemma normalise_before_round_avoiding_keeps_scope
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope (normalise_before_round_avoiding avoid kept).
Proof. apply keeps_scope_intro, normalise_before_round_avoiding_scoped. Qed.

Lemma normalise_before_round_scoped (kept : list code_id) (e : expr) :
  scoped e (normalise_before_round kept e).
Proof. apply normalise_before_round_avoiding_scoped. Qed.

#[export] Hint Resolve normalise_before_round_scoped : scoped.

(* ======================================================================== *)
(* A rewrite made to keep scope by a test                                   *)
(* ======================================================================== *)

Lemma scope_kept_scoped (e e' : expr) :
  scope_kept e e' = true -> scoped e e'.
Proof.
  unfold scope_kept. intros Hkept _.
  apply andb_true_iff in Hkept. destruct Hkept as [Hkept Hk].
  apply andb_true_iff in Hkept. destruct Hkept as [Hclosed Hv].
  split; [ exact Hclosed | ].
  split; intros name Hname;
    [ exact (all_among_In _ _ name Hv Hname)
    | exact (all_among_In _ _ name Hk Hname) ].
Qed.

Lemma scoped_scope_kept (e e' : expr) :
  code_closed e = true -> scoped e e' -> scope_kept e e' = true.
Proof.
  intros Hclosed Hscoped. destruct (Hscoped Hclosed) as (Hclosed' & Hv & Hk).
  unfold scope_kept. rewrite Hclosed'. cbn [andb].
  apply andb_true_iff.
  split; apply all_among_iff; intros name Hname; apply among_In;
    [ apply Hv | apply Hk ]; apply among_In; exact Hname.
Qed.

Lemma guard_scope_scoped (f : expr -> expr) (e : expr) :
  scoped e (guard_scope f e).
Proof.
  unfold guard_scope. cbv zeta. intros Hclosed. rewrite Hclosed.
  cbn [negb orb].
  destruct (scope_kept e (f e)) eqn:Hkept;
    [ exact (scope_kept_scoped e (f e) Hkept Hclosed)
    | exact (scoped_refl e Hclosed) ].
Qed.

#[export] Hint Resolve guard_scope_scoped : scoped.

Theorem guard_scope_keeps_scope : guard_scope_keeps_scope_statement.
Proof. intros f. apply keeps_scope_intro, guard_scope_scoped. Qed.

Theorem guard_scope_unchanged : guard_scope_unchanged_statement.
Proof.
  intros f Hf e. unfold guard_scope. cbv zeta.
  destruct (code_closed e) eqn:Hclosed; cbn [negb orb]; [ | reflexivity ].
  rewrite (scoped_scope_kept e (f e) Hclosed (keeps_scope_scoped f Hf e)).
  reflexivity.
Qed.

Lemma all_among_rename_variable (r : renaming) (xs ys : list variable) :
  FlambdaMapSpec.all_among (map (rename_variable r) xs)
    (map (rename_variable r) ys)
  = FlambdaMapSpec.all_among xs ys.
Proof.
  unfold FlambdaMapSpec.all_among.
  induction xs as [ | x xs IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite among_rename_variable, IH. reflexivity.
Qed.

Lemma all_among_rename_continuation (r : renaming)
    (ks js : list continuation) :
  FlambdaMapSpec.all_among (map (rename_continuation r) ks)
    (map (rename_continuation r) js)
  = FlambdaMapSpec.all_among ks js.
Proof.
  unfold FlambdaMapSpec.all_among.
  induction ks as [ | k ks IH]; [ reflexivity | ].
  cbn [map forallb]. rewrite among_rename_continuation, IH. reflexivity.
Qed.

Lemma scope_kept_renamed (r : renaming) (e e' : expr) :
  scope_kept (rename_expr r e) (rename_expr r e') = scope_kept e e'.
Proof.
  unfold scope_kept.
  rewrite code_closed_renamed, !free_variables_of_expr_renamed,
    !free_continuations_of_expr_renamed, all_among_rename_variable,
    all_among_rename_continuation.
  reflexivity.
Qed.

Theorem guard_scope_equivariant : guard_scope_equivariant_statement.
Proof.
  intros f Hf r e. unfold guard_scope. cbv zeta.
  rewrite Hf, code_closed_renamed, scope_kept_renamed.
  destruct (negb (code_closed e) || scope_kept e (f e)); reflexivity.
Qed.
