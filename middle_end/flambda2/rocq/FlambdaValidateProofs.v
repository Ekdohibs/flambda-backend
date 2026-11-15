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

(* Proofs of the statements of FlambdaValidateSpec.v. Nothing here needs
   review: each step of the check is a theorem of the file it comes from,
   and these are put end to end. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaCompareSpec FlambdaCompareProofs
  FlambdaNormaliseSpec FlambdaNormaliseProofs FlambdaBlockLoadSpec
  FlambdaBlockLoadProofs FlambdaCodeLetsSpec FlambdaCodeLetsProofs
  FlambdaCseSpec FlambdaCseProofs
  FlambdaCseEquivalentSpec FlambdaCseEquivalentProofs
  FlambdaMadeLengthsSpec FlambdaMadeLengthsProofs
  FlambdaDeadParamsSpec FlambdaDeadParamsProofs
  FlambdaForwardSpec FlambdaForwardProofs
  FlambdaForwardKindedSpec FlambdaForwardKindedProofs
  FlambdaForwardArmSpec FlambdaForwardArmProofs
  FlambdaLiftContsSpec FlambdaLiftContsProofs
  FlambdaIdentitiesSpec FlambdaIdentitiesProofs
  FlambdaMergeSwitchesSpec FlambdaMergeSwitchesProofs
  FlambdaMergeTablesSpec FlambdaMergeTablesProofs
  FlambdaInlineSiblingsSpec FlambdaInlineSiblingsProofs
  FlambdaInlineTailSpec FlambdaInlineTailProofs
  FlambdaRenameApartSpec
  FlambdaRenameApartProofs FlambdaStaticsOrderSpec
  FlambdaStaticsOrderProofs FlambdaLoopAliasesSpec FlambdaLoopAliasesProofs
  FlambdaLoopLetsSpec FlambdaLoopLetsProofs FlambdaParamAliasesSpec
  FlambdaParamAliasesProofs FlambdaRecHandlersSpec FlambdaRecHandlersProofs
  FlambdaRecoverComparisonSpec FlambdaRecoverComparisonProofs
  FlambdaSameComparisonSpec FlambdaSameComparisonProofs
  FlambdaShareProjectionsSpec FlambdaShareProjectionsProofs
  FlambdaShortcutSpec FlambdaShortcutProofs FlambdaSameHandlersSpec
  FlambdaSameHandlersProofs FlambdaSinkContsSpec
  FlambdaSinkContsProofs FlambdaSinkUsesSpec FlambdaSinkUsesProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaGiveScrutineeSpec FlambdaGiveScrutineeProofs
  FlambdaArmNumbersSpec FlambdaArmNumbersProofs
  FlambdaAliasLetsSpec FlambdaAliasLetsProofs
  FlambdaWrapReturnsSpec FlambdaWrapReturnsProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs
  FlambdaUnboxSpec FlambdaUnboxProofs
  FlambdaUnboxVariantsSpec FlambdaUnboxVariantsProofs
  FlambdaUnboxParamsSpec FlambdaUnboxInScopeSpec FlambdaUnboxInScopeProofs
  FlambdaUnboxLoopsSpec FlambdaUnboxLoopsProofs FlambdaUnboxTogetherSpec
  FlambdaUnusedContsSpec FlambdaUnusedContsProofs FlambdaUnusedParamsSpec
  FlambdaUnusedParamsProofs FlambdaUnusedProjectionsSpec
  FlambdaUnusedProjectionsProofs FlambdaReadsAfterSpec
  FlambdaReadsAfterProofs FlambdaUnusedStaticsSpec
  FlambdaUnusedStaticsProofs FlambdaLiftSpec FlambdaLiftProofs
  FlambdaLiftFromCodeSpec FlambdaLiftFromCodeProofs
  FlambdaLiftFromArmsSpec FlambdaLiftFromArmsProofs
  FlambdaLiftAroundSpec FlambdaLiftAroundProofs
  FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs FlambdaKnownSymbolsSpec
  FlambdaKnownSymbolsProofs FlambdaKnownClosuresSpec FlambdaKnownClosuresProofs
  FlambdaFloatCodeSpec FlambdaFloatCodeProofs FlambdaGroupsUpSpec
  FlambdaGroupsUpProofs FlambdaHoistCodeSpec
  FlambdaHoistCodeProofs FlambdaLoopifySpec FlambdaLoopifyProofs
  FlambdaHoistStaticsSpec FlambdaGuardScopeSpec FlambdaGuardScopeProofs
  FlambdaValidateSpec.
From OxCamlFlambda2 Require FlambdaHoistStaticsProofs.
From OxCamlFlambda2 Require FlambdaBindConstParamsSpec
  FlambdaBindConstParamsProofs.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require FlambdaUnaliasRecParamsProofs.
From OxCamlFlambda2 Require FlambdaShadowParamsProofs FlambdaTwinParamsProofs.
From OxCamlFlambda2 Require Import FlambdaDemoteSpec.
From OxCamlFlambda2 Require Import FlambdaUnusedHeadStaticsSpec
  FlambdaUnusedHeadStaticsProofs.
From OxCamlFlambda2 Require Import FlambdaCopyCodeSpec FlambdaCopyCodeProofs.
From OxCamlFlambda2 Require Import FlambdaClosureCallsSpec
  FlambdaClosureCallsProofs.
From OxCamlFlambda2 Require Import FlambdaClosureOfCodeSpec.
From OxCamlFlambda2 Require FlambdaClosureOfCodeProofs.
From OxCamlFlambda2 Require FlambdaDemoteProofs.
From OxCamlFlambda2 Require Import FlambdaKnownSiblingsSpec
  FlambdaKnownSiblingsProofs.
From OxCamlFlambda2 Require Import FlambdaKnownSlotsSpec
  FlambdaKnownSlotsProofs.
From OxCamlFlambda2 Require FlambdaSharedReadsProofs.
From OxCamlFlambda2 Require Import FlambdaStubCallsSpec
  FlambdaStubCallsProofs.
From OxCamlFlambda2 Require Import FlambdaNeverReturnsSpec
  FlambdaNeverReturnsProofs.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec
  FlambdaAliasesProofs.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
From OxCamlFlambda2 Require FlambdaUnboxTogetherProofs.
Import ListNotations.

Lemma expr_equivalent_refl (e : expr) : expr_equivalent e e.
Proof. intros vars conts statics hp rd. apply equivalent_refl. Qed.

Lemma expr_equivalent_sym (e e' : expr) :
  expr_equivalent e e' -> expr_equivalent e' e.
Proof. intros H vars conts statics hp rd. apply equivalent_sym, H. Qed.

(* One rewrite that is an equivalence, at the head of the right-hand
   side. *)
Ltac equivalence_step :=
  first
    [ apply fold_constants_equivalent
    | apply follow_aliases_equivalent
    | apply remove_unused_defined_lets_equivalent
    | apply remove_unused_projections_equivalent
    | apply load_known_blocks_surely_equivalent
    | apply fold_made_lengths_equivalent
    | apply remove_constant_allocations_equivalent
    | apply drop_deleted_code_equivalent
    | apply merge_statics_equivalent
    | apply inline_known_calls_equivalent
    | apply inline_known_sibling_calls_equivalent
    | apply thread_switches_equivalent
    | apply remove_unused_conts_equivalent
    | apply sink_let_conts_equivalent
    | apply follow_aliases_in_loops_equivalent
    | apply shortcut_switches_equivalent
    | apply share_handlers_equivalent
    | apply remove_unused_defined_lets_in_loops_equivalent
    | apply inline_tail_conts_kinded_equivalent
    | apply forward_kinded_equivalent
    | apply forward_to_arms_kinded_equivalent
    | apply FlambdaBindConstParamsProofs.bind_const_params_equivalent
    | apply lift_handler_conts_equivalent
    | apply unalias_params_equivalent
    | apply FlambdaUnaliasRecParamsProofs.unalias_rec_params_equivalent
    | apply give_scrutinees_equivalent
    | apply numbers_in_arms_equivalent
    | apply merge_alias_lets_equivalent
    | apply share_known_primitives_equivalent ].

Ltac refinement_step :=
  first
    [ apply inline_calls_refines
    | apply collapse_switches_refines
    | apply merge_switches_refines
    | apply inline_tail_conts_refines
    | apply remove_unused_lets_refines
    | apply load_known_blocks_refines
    | apply unbox_known_numbers_refines
    | apply drop_unused_params_refines
    | apply wrap_returns_refines
    | apply untag_params_refines
    | apply drop_dead_params_refines
    | apply bind_aliased_params_refines
    | apply forward_applications_refines
    | apply unbox_variant_params_refines
    | apply recover_known_comparisons_refines
    | apply fold_same_comparisons_refines
    | apply share_known_primitives_refines
    | apply share_known_projections_refines
    | apply apply_identities_refines
    | apply expr_equivalent_refines; equivalence_step ].

Lemma normalise_expr_before_refines (e : expr) :
  expr_refines e (normalise_expr_before e).
Proof.
  unfold normalise_expr_before. cbv zeta.
  repeat (eapply expr_refines_trans; [ | solve [ refinement_step ] ]).
  apply expr_refines_refl.
Qed.

Lemma normalise_code_before_refines (body : expr) :
  expr_refines body (normalise_code_before body).
Proof.
  unfold normalise_code_before.
  exact (expr_refines_trans _ _ _ (normalise_expr_before_refines body)
    (rewrite_rec_handlers_refines _ normalise_expr_before_refines _)).
Qed.

Lemma normalise_before_inlined_refines (e : expr) :
  expr_refines e (normalise_before_inlined e).
Proof.
  unfold normalise_before_inlined.
  eapply expr_refines_trans;
    [ | apply rewrite_code_refines, normalise_code_before_refines ].
  eapply expr_refines_trans;
    [ | apply expr_equivalent_refines, merge_statics_equivalent ].
  eapply expr_refines_trans;
    [ | apply expr_equivalent_refines, drop_deleted_code_equivalent ].
  apply normalise_code_before_refines.
Qed.

(* The twins: each rewrite refines, and the term is given back where
   no twin was made. *)
Lemma with_twins_refines (avoid : list continuation) (e : expr) :
  expr_refines e (with_twins avoid e).
Proof.
  unfold with_twins. cbv zeta.
  destruct (expr_eqb [] [] _ _); [ apply expr_refines_refl | ].
  eapply expr_refines_trans; [ | apply normalise_before_inlined_refines ].
  eapply expr_refines_trans;
    [ | apply rewrite_code_refines,
          FlambdaTwinParamsProofs.twin_params_refines ].
  eapply expr_refines_trans;
    [ | apply FlambdaTwinParamsProofs.twin_params_refines ].
  eapply expr_refines_trans;
    [ | apply expr_equivalent_refines, rename_apart_avoiding_equivalent ].
  eapply expr_refines_trans;
    [ | apply rewrite_code_refines,
          FlambdaShadowParamsProofs.shadow_params_refines ].
  apply FlambdaShadowParamsProofs.shadow_params_refines.
Qed.

Lemma normalise_before_round_avoiding_refines (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  expr_refines e (normalise_before_round_avoiding avoid kept e).
Proof.
  unfold normalise_before_round_avoiding. cbv zeta.
  assert (Hround :
    expr_refines e
      (normalise_before_inlined
         (rename_apart_avoiding avoid
            (inline_sibling_calls kept (inline_calls kept [] e))))).
  { eapply expr_refines_trans; [ | apply normalise_before_inlined_refines ].
    eapply expr_refines_trans;
      [ | apply expr_equivalent_refines, rename_apart_avoiding_equivalent ].
    eapply expr_refines_trans; [ | apply inline_sibling_calls_refines ].
    apply inline_calls_refines. }
  destruct (expr_eqb [] [] e _); [ | exact Hround ].
  exact (expr_refines_trans _ _ _ Hround (with_twins_refines avoid _)).
Qed.

Lemma normalise_before_round_refines (kept : list code_id) (e : expr) :
  expr_refines e (normalise_before_round kept e).
Proof. apply normalise_before_round_avoiding_refines. Qed.

(* The first phase of the term before. The round given what is known of
   symbols refines wherever that holds: the rewrites before the
   unboxing and those after it refine under all environments, and the
   unboxing where what it is given holds. *)
Lemma expr_refines_at (e e' : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) :
  expr_refines e e' ->
  refines (State e vars conts statics hp rd)
    (State e' vars conts statics hp rd).
Proof. intros Hrefines. apply Hrefines. Qed.

Lemma normalise_expr_params_knowing_refines :
  refines_where_known normalise_expr_params_knowing.
Proof.
  intros known e vars conts statics hp rd Honly Hbound.
  unfold normalise_expr_params_knowing. cbv zeta.
  match goal with
  | |- context [unbox_variants_knowing known ?x] =>
    apply (refines_trans _
      (State
         (unbox_together_knowing known
            (untag_params
               (unbox_loops_knowing known (unbox_variants_knowing known x))))
         vars conts statics hp rd));
      [ apply (refines_trans _
          (State
             (untag_params
                (unbox_loops_knowing known (unbox_variants_knowing known x)))
             vars conts statics hp rd));
          [ apply (refines_trans _
              (State
                 (unbox_loops_knowing known (unbox_variants_knowing known x))
                 vars conts statics hp rd));
              [ apply (refines_trans _
                  (State (unbox_variants_knowing known x) vars conts statics
                     hp rd));
                  [ apply (refines_trans _ (State x vars conts statics hp rd));
                      [ | exact (unbox_variants_knowing_symbols_refines known
                            x vars conts statics hp rd_ Honly Hbound) ]
                  | exact (unbox_loops_knowing_symbols_refines known _ vars
                      conts statics hp rd_ Honly Hbound) ]
              | apply expr_refines_at, untag_params_refines ]
          | exact
              (FlambdaUnboxTogetherProofs
               .unbox_together_knowing_symbols_refines known _ vars conts
                 statics hp rd_ Honly Hbound) ]
      | ]
  end; apply expr_refines_at;
    repeat (eapply expr_refines_trans; [ | solve [ refinement_step ] ]);
    apply expr_refines_refl.
Qed.

Lemma normalise_expr_params_refines (e : expr) :
  expr_refines e (normalise_expr_params e).
Proof.
  intros vars conts statics hp rd.
  exact (normalise_expr_params_knowing_refines no_facts e vars conts statics
    hp rd_ no_facts_only (no_symbols_bound statics)).
Qed.

Lemma normalise_code_params_knowing_refines :
  refines_where_known normalise_code_params_knowing.
Proof.
  intros known body vars conts statics hp rd Honly Hbound.
  unfold normalise_code_params_knowing.
  eapply refines_trans;
    [ exact (rewrite_rec_handlers_in_scope_refines _
        normalise_expr_params_knowing_refines known body vars conts statics
        hp rd_ Honly Hbound)
    | exact (normalise_expr_params_knowing_refines known _ vars conts
        statics hp rd_ Honly Hbound) ].
Qed.

Lemma normalise_code_params_refines (body : expr) :
  expr_refines body (normalise_code_params body).
Proof.
  intros vars conts statics hp rd.
  exact (normalise_code_params_knowing_refines no_facts body vars conts
    statics hp rd_ no_facts_only (no_symbols_bound statics)).
Qed.

Lemma normalise_params_round_refines (e : expr) :
  expr_refines e (normalise_params_round e).
Proof.
  unfold normalise_params_round.
  eapply expr_refines_trans;
    [ | apply rewrite_code_in_scope_expr_refines,
          normalise_code_params_knowing_refines ].
  eapply expr_refines_trans;
    [ apply merge_tables_refines | apply normalise_code_params_refines ].
Qed.

(* What follows a round of the first phase: the parameters of loops
   that every turn passes on taken away, the continuations of a
   handler bound outside it, and the applications made to go to the
   arms. *)
Lemma arms_expr_before_refines (e : expr) :
  expr_refines e (arms_expr_before e).
Proof.
  unfold arms_expr_before.
  eapply expr_refines_trans;
    [ | apply expr_equivalent_refines, remove_unused_conts_equivalent ].
  eapply expr_refines_trans; [ | apply forward_applications_refines ].
  eapply expr_refines_trans; [ | apply forward_to_arms_refines ].
  eapply expr_refines_trans;
    [ | apply expr_equivalent_refines, lift_handler_conts_equivalent ].
  apply expr_equivalent_refines,
    FlambdaUnaliasRecParamsProofs.unalias_rec_params_equivalent.
Qed.

Lemma arms_code_before_refines (body : expr) :
  expr_refines body (arms_code_before body).
Proof.
  unfold arms_code_before.
  exact (expr_refines_trans _ _ _
    (rewrite_rec_handlers_refines _ arms_expr_before_refines body)
    (arms_expr_before_refines _)).
Qed.

Lemma first_phase_round_refines (e : expr) :
  expr_refines e (first_phase_round e).
Proof.
  unfold first_phase_round.
  eapply expr_refines_trans;
    [ | apply rewrite_code_refines, arms_code_before_refines ].
  eapply expr_refines_trans;
    [ apply normalise_params_round_refines | apply arms_code_before_refines ].
Qed.

Lemma iterate_refines (f : expr -> expr) :
  (∀ e : expr, expr_refines e (f e)) ->
  ∀ (rounds : nat) (e : expr), expr_refines e (iterate rounds f e).
Proof.
  intros Hf rounds.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply expr_refines_refl | ].
  cbv zeta. destruct (expr_eqb [] [] e (f e)); [ apply Hf | ].
  exact (expr_refines_trans _ _ _ (Hf e) (IH _)).
Qed.

(* The demotion of exception handlers and the rounds after it. *)
Lemma demote_and_tidy_refines (rounds : nat) (e : expr) :
  expr_refines e (demote_and_tidy rounds e).
Proof.
  unfold demote_and_tidy.
  eapply expr_refines_trans;
    [ | apply iterate_refines, normalise_before_inlined_refines ].
  eapply expr_refines_trans;
    [ | apply iterate_refines, first_phase_round_refines ].
  apply expr_equivalent_refines,
    FlambdaDemoteProofs.demote_handlers_equivalent.
Qed.

Lemma normalise_expr_after_equivalent (e : expr) :
  expr_equivalent e (normalise_expr_after e).
Proof.
  unfold normalise_expr_after, inline_once_applied_after.
  repeat (eapply expr_equivalent_trans; [ | solve [ equivalence_step ] ]).
  apply expr_equivalent_refl.
Qed.

Lemma normalise_code_after_equivalent (body : expr) :
  expr_equivalent body (normalise_code_after body).
Proof.
  unfold normalise_code_after.
  exact (expr_equivalent_trans _ _ _ (normalise_expr_after_equivalent body)
    (rewrite_rec_handlers_equivalent _ normalise_expr_after_equivalent _)).
Qed.

Lemma normalise_after_round_equivalent (kept : list code_id) (e : expr) :
  expr_equivalent e (normalise_after_round kept e).
Proof.
  unfold normalise_after_round, inline_once_applied_after.
  eapply expr_equivalent_trans;
    [ | apply remove_unused_defined_lets_in_code_equivalent ].
  eapply expr_equivalent_trans;
    [ | apply forward_to_arms_in_code_equivalent ].
  eapply expr_equivalent_trans;
    [ | apply forward_kinded_in_code_equivalent ].
  eapply expr_equivalent_trans;
    [ | apply rewrite_code_equivalent, normalise_code_after_equivalent ].
  repeat (eapply expr_equivalent_trans; [ | solve [ equivalence_step ] ]).
  eapply expr_equivalent_trans;
    [ | apply rewrite_rec_handlers_equivalent,
          normalise_expr_after_equivalent ].
  repeat (eapply expr_equivalent_trans; [ | solve [ equivalence_step ] ]).
  apply expr_equivalent_refl.
Qed.

Lemma normalise_before_refines (rounds : nat) (kept : list code_id) :
  ∀ e : expr, expr_refines e (normalise_before rounds kept e).
Proof.
  intros e. unfold normalise_before.
  eapply expr_refines_trans; [ | apply demote_and_tidy_refines ].
  eapply expr_refines_trans;
    [ | exact (iterate_refines _ (normalise_before_round_refines kept)
                 rounds _) ].
  eapply expr_refines_trans;
    [ | apply guard_scope_refines,
          FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines ].
  exact (expr_refines_trans _ _ _
    (iterate_refines _ first_phase_round_refines rounds e)
    (iterate_refines _ (normalise_before_round_refines kept) rounds _)).
Qed.

Lemma normalise_after_equivalent (rounds : nat) (kept : list code_id) :
  ∀ e : expr, expr_equivalent e (normalise_after rounds kept e).
Proof.
  unfold normalise_after.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply expr_equivalent_refl | ].
  cbv zeta.
  destruct (expr_eqb [] [] e (normalise_after_round kept e));
    [ apply normalise_after_round_equivalent | ].
  exact (expr_equivalent_trans _ _ _ (normalise_after_round_equivalent kept e)
    (IH _)).
Qed.

(* The placing of the continuations of a term is an equivalence. *)
Lemma place_conts_equivalent (e : expr) : expr_equivalent e (place_conts e).
Proof.
  unfold place_conts.
  exact (expr_equivalent_trans _ _ _ (sink_conts_to_uses_equivalent e)
    (rewrite_rec_handlers_equivalent _ sink_conts_to_uses_equivalent _)).
Qed.

Lemma placed_equivalent (e : expr) : expr_equivalent e (placed e).
Proof.
  unfold placed.
  exact (expr_equivalent_trans _ _ _ (place_conts_equivalent e)
    (rewrite_code_equivalent _ place_conts_equivalent _)).
Qed.

(* The tidying of the term before, once its continuations are placed,
   refines. *)
Lemma tidy_expr_before_refines (e : expr) :
  expr_refines e (tidy_expr_before e).
Proof.
  unfold tidy_expr_before.
  repeat (eapply expr_refines_trans; [ | solve [ refinement_step ] ]).
  apply expr_refines_refl.
Qed.

Lemma tidy_code_before_refines (body : expr) :
  expr_refines body (tidy_code_before body).
Proof.
  unfold tidy_code_before.
  exact (expr_refines_trans _ _ _ (tidy_expr_before_refines body)
    (rewrite_rec_handlers_refines _ tidy_expr_before_refines _)).
Qed.

Lemma tidy_before_round_refines (e : expr) :
  expr_refines e (tidy_before_round e).
Proof.
  unfold tidy_before_round.
  exact (expr_refines_trans _ _ _ (tidy_code_before_refines e)
    (rewrite_code_refines _ tidy_code_before_refines _)).
Qed.

Lemma tidied_before_refines (rounds : nat) (e : expr) :
  expr_refines e (tidied_before rounds e).
Proof.
  unfold tidied_before.
  exact (expr_refines_trans _ _ _
    (expr_refines_trans _ _ _
       (expr_equivalent_refines _ _ (placed_equivalent e))
       (iterate_refines _ tidy_before_round_refines rounds _))
    (expr_equivalent_refines _ _ (groups_up_equivalent _))).
Qed.

(* That of the term after is an equivalence. *)
Lemma tidy_expr_after_equivalent (e : expr) :
  expr_equivalent e (tidy_expr_after e).
Proof.
  unfold tidy_expr_after.
  repeat (eapply expr_equivalent_trans; [ | solve [ equivalence_step ] ]).
  apply expr_equivalent_refl.
Qed.

Lemma tidy_code_after_equivalent (body : expr) :
  expr_equivalent body (tidy_code_after body).
Proof.
  unfold tidy_code_after.
  exact (expr_equivalent_trans _ _ _ (tidy_expr_after_equivalent body)
    (rewrite_rec_handlers_equivalent _ tidy_expr_after_equivalent _)).
Qed.

Lemma tidy_after_round_equivalent (e : expr) :
  expr_equivalent e (tidy_after_round e).
Proof.
  unfold tidy_after_round.
  eapply expr_equivalent_trans;
    [ | apply remove_unused_defined_lets_in_code_equivalent ].
  exact (expr_equivalent_trans _ _ _ (tidy_code_after_equivalent e)
    (rewrite_code_equivalent _ tidy_code_after_equivalent _)).
Qed.

Lemma tidied_after_equivalent (rounds : nat) :
  ∀ e : expr, expr_equivalent e (tidied_after rounds e).
Proof.
  intros e. unfold tidied_after.
  refine (expr_equivalent_trans _ _ _ _ (groups_up_equivalent _)).
  apply (expr_equivalent_trans _ _ _ (placed_equivalent e)).
  generalize (placed e). clear e.
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply expr_equivalent_refl | ].
  cbv zeta.
  destruct (expr_eqb [] [] e (tidy_after_round e));
    [ apply tidy_after_round_equivalent | ].
  exact (expr_equivalent_trans _ _ _ (tidy_after_round_equivalent e) (IH _)).
Qed.

(* The term after, normalised, with its continuations placed and
   tidied, is equivalent to the term after. *)
Lemma tidied_after_normalised_equivalent (rounds : nat)
    (kept : list code_id) (after : expr) :
  expr_equivalent after
    (tidied_after rounds (normalise_after rounds kept after)).
Proof.
  exact (expr_equivalent_trans _ _ _
    (normalise_after_equivalent rounds kept after)
    (tidied_after_equivalent rounds _)).
Qed.

(* A term that is, once its static bindings are in the order of the
   term after normalised, the same as that term up to bound names: it
   is equivalent to the term after. *)
Lemma ordered_same_equivalent (rounds : nat) (kept : list code_id)
    (e after : expr) :
  expr_eqb [] []
    (order_statics_as (tidied_after rounds (normalise_after rounds kept after))
       e)
    (tidied_after rounds (normalise_after rounds kept after))
  = true ->
  expr_equivalent e after.
Proof.
  intros Heq.
  apply (expr_equivalent_trans _ _ _
    (order_statics_as_equivalent
       (tidied_after rounds (normalise_after rounds kept after)) e)).
  apply (expr_equivalent_trans _ _ _ (expr_eqb_sound _ _ Heq)).
  apply expr_equivalent_sym, tidied_after_normalised_equivalent.
Qed.

(* The calls of the term before taken on, or not: it is refined. *)
Lemma take_returns_on_refines (e : expr) :
  expr_refines e (take_returns_on e).
Proof.
  unfold take_returns_on.
  apply (expr_refines_trans _ _ _ (forward_applications_with_refines true e)).
  apply expr_equivalent_refines.
  exact (expr_equivalent_trans _ _ _ (remove_unused_conts_equivalent _)
    (rewrite_rec_handlers_equivalent _ remove_unused_conts_equivalent _)).
Qed.

Lemma returns_taken_on_refines (e : expr) :
  expr_refines e (returns_taken_on e).
Proof.
  unfold returns_taken_on.
  exact (expr_refines_trans _ _ _ (take_returns_on_refines e)
    (rewrite_code_refines _ take_returns_on_refines _)).
Qed.

Lemma with_returns_refines (taken : bool) (e : expr) :
  expr_refines e (with_returns taken e).
Proof.
  destruct taken; [ apply returns_taken_on_refines | apply expr_refines_refl ].
Qed.

(* The two normalised terms being the same up to bound names, once the
   static bindings of the first are in the order of the second, the term
   after refines the term before. *)
Lemma normalised_refines (rounds : nat) (kept : list code_id) (taken : bool)
    (before after : expr) :
  expr_eqb [] []
    (order_statics_as (tidied_after rounds (normalise_after rounds kept after))
       (tidied_before rounds
          (with_returns taken (normalise_before rounds kept before))))
    (tidied_after rounds (normalise_after rounds kept after))
  = true ->
  expr_refines before after.
Proof.
  intros Heq.
  apply (expr_refines_trans _ _ _
    (normalise_before_refines rounds kept before)).
  apply (expr_refines_trans _ _ _ (with_returns_refines taken _)).
  apply (expr_refines_trans _ _ _ (tidied_before_refines rounds _)).
  apply expr_equivalent_refines.
  exact (ordered_same_equivalent rounds kept _ after Heq).
Qed.

Lemma expr_refines_program (source target : expr) :
  expr_refines source target -> program_refines source target.
Proof. intros H. apply H. Qed.

Lemma without_statics_program_refines (D : dropped) (e : expr) :
  program_refines e (without_statics D e).
Proof.
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  apply (refines_trans _ _ _ (drop_statics_program_refines D e e' Hdrop)).
  apply expr_refines_program, expr_equivalent_refines.
  exact (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
    (merge_statics_equivalent _)).
Qed.

(* The term after refines, run on its own, the term after without the
   unused static bindings of its head. *)
Lemma without_head_statics_program_refines (D : dropped) (e : expr) :
  program_refines (without_head_statics D e) e.
Proof.
  unfold without_head_statics.
  destruct (head_statics_removable D e) eqn:Hchecks; [ | apply refines_refl ].
  unfold without_statics.
  destruct (drop_statics D e) as [e' | ] eqn:Hdrop; [ | apply refines_refl ].
  assert (Hhead : drop_head_statics D e = Some e')
    by (unfold drop_head_statics; rewrite Hchecks; exact Hdrop).
  apply (refines_trans _ (State e' [] [] empty_static_env empty_heap no_reads));
    [ | exact (drop_head_statics_program_refines D e e' Hhead) ].
  apply expr_refines_program, expr_equivalent_refines, expr_equivalent_sym.
  exact (expr_equivalent_trans _ _ _ (drop_deleted_code_equivalent e')
    (merge_statics_equivalent _)).
Qed.

Lemma with_lifting_program_refines (L : lifting) (e : expr) :
  program_refines e (with_lifting L e).
Proof.
  unfold with_lifting. cbv zeta.
  set (at_head :=
    match lift_to_head_unit (lifting_unit L) (lifted_to_head L) e with
    | Some e' => e'
    | None => e
    end).
  set (out_of_code :=
    match
      lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) at_head
    with
    | Some e' => e'
    | None => at_head
    end).
  set (lifted :=
    match lift_unit L out_of_code with
    | Some e' => e'
    | None => out_of_code
    end).
  set (around :=
    match
      lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L)
        lifted
    with
    | Some e' => e'
    | None => lifted
    end).
  apply
    (refines_trans _
       (State (add_head_tables (tables_put L) around) [] [] empty_static_env
          empty_heap no_reads));
    [ | exact (with_copies_program_refines (code_copied L) _) ].
  apply
    (refines_trans _ (State around [] [] empty_static_env empty_heap no_reads));
    [ | exact (add_head_tables_program_refines (tables_put L) around) ].
  apply
    (refines_trans _
       (State at_head [] [] empty_static_env empty_heap no_reads)).
  - subst at_head.
    destruct (lift_to_head_unit (lifting_unit L) (lifted_to_head L) e)
      as [e' | ] eqn:Hhead;
      [ | apply refines_refl ].
    exact (lift_to_head_unit_program_refines _ _ e e' Hhead).
  - apply
    (refines_trans _
       (State out_of_code [] [] empty_static_env empty_heap no_reads)).
    + subst out_of_code.
      destruct
        (lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) at_head)
        as [e' | ] eqn:Hout;
        [ | apply refines_refl ].
      exact (lift_from_code_unit_program_refines _ _ at_head e' Hout).
    + apply
        (refines_trans _
           (State lifted [] [] empty_static_env empty_heap no_reads)).
      * subst lifted.
        destruct (lift_unit L out_of_code) as [e' | ] eqn:Hlift;
          [ | apply refines_refl ].
        exact (lift_unit_program_refines L out_of_code e' Hlift).
      * subst around.
        destruct
          (lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L)
             lifted)
          as [e' | ] eqn:Haround;
          [ | apply refines_refl ].
        exact (lift_around_unit_program_refines _ _ _ lifted e' Haround).
Qed.

(* With no [Let] to make an alias of a symbol bound around it, no
   group to put, no table to bind and no code to copy, [with_lifting]
   is its first three liftings. *)
Lemma with_lifting_nothing_around (L : lifting) (e : expr) :
  lifted_around L = [] ->
  groups_put L = [] ->
  tables_put L = [] ->
  code_copied L = [] ->
  with_lifting L e
  = (let at_head :=
       match lift_to_head_unit (lifting_unit L) (lifted_to_head L) e with
       | Some e' => e'
       | None => e
       end
     in
     let out_of_code :=
       match
         lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) at_head
       with
       | Some e' => e'
       | None => at_head
       end
     in
     match lift_unit L out_of_code with
     | Some e' => e'
     | None => out_of_code
     end).
Proof.
  intros Haround Hgroups Htables Hcopied. unfold with_lifting. cbv zeta.
  rewrite Haround, Hgroups, Htables, Hcopied.
  cbn [add_head_tables with_copies].
  apply FlambdaLiftAroundProofs.lift_around_unit_nothing.
Qed.

Lemma with_known_program_refines (e : expr) :
  program_refines e (with_known e).
Proof.
  unfold with_known.
  destruct (fold_known_unit e) as [e' | ] eqn:Hfold; [ | apply refines_refl ].
  exact (fold_known_unit_program_refines e e' Hfold).
Qed.

Lemma iterate_lifted_program_refines (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    program_refines e (iterate rounds (normalise_lifted_round avoid kept) e).
Proof.
  assert (Hround :
    ∀ e : expr, program_refines e (normalise_lifted_round avoid kept e)).
  { intros e. unfold normalise_lifted_round.
    apply (refines_trans _ _ _
      (expr_refines_program _ _
         (normalise_before_round_avoiding_refines avoid kept e))).
    apply (refines_trans _ _ _
      (expr_refines_program _ _ (use_known_closures_refines _))).
    apply (refines_trans _ _ _ (with_known_program_refines _)).
    apply (refines_trans _ _ _ (read_known_slots_program_refines _)).
    apply FlambdaSharedReadsProofs.share_static_reads_program_refines. }
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply refines_refl | ].
  cbv zeta.
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    [ apply Hround | ].
  exact (refines_trans _ _ _ (Hround e) (IH _)).
Qed.

Lemma normalise_lifted_program_refines (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr, program_refines e (normalise_lifted rounds avoid kept e).
Proof.
  intros e. unfold normalise_lifted.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (iterate_refines _ first_phase_round_refines rounds e))).
  apply (refines_trans _ _ _
    (iterate_lifted_program_refines rounds avoid kept _)).
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (guard_scope_refines _ _
          (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines
             _)))).
  apply iterate_lifted_program_refines.
Qed.

(* The calls of the closures that a [Let] makes made direct, in the
   term and in the bodies of its code. *)
Lemma with_made_closures_called_refines (e : expr) :
  expr_refines e (with_made_closures_called e).
Proof.
  unfold with_made_closures_called.
  eapply expr_refines_trans; [ apply call_made_closures_refines | ].
  apply rewrite_code_refines, call_made_closures_refines.
Qed.

(* One round of [normalise_lifted], from what is proved of its rounds
   in turn: [iterate] of one round is that round. *)
Lemma normalise_lifted_round_program_refines (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  program_refines e (normalise_lifted_round avoid kept e).
Proof.
  generalize (iterate_lifted_program_refines 1 avoid kept e).
  cbn [iterate]. cbv zeta.
  destruct (expr_eqb [] [] e (normalise_lifted_round avoid kept e));
    exact (fun H => H).
Qed.

(* The rounds that are followed by those calls made direct, where
   they are asked to. *)
Lemma iterate_lifted_calling_program_refines (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    program_refines e
      (iterate rounds (normalise_lifted_round_calling calling avoid kept) e).
Proof.
  assert (Hround :
    ∀ e : expr,
      program_refines e (normalise_lifted_round_calling calling avoid kept e)).
  { intros e. unfold normalise_lifted_round_calling. cbv zeta.
    apply (refines_trans _ _ _
      (normalise_lifted_round_program_refines avoid kept e)).
    destruct calling; [ | apply refines_refl ].
    exact (expr_refines_program _ _ (with_made_closures_called_refines _)). }
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply refines_refl | ].
  cbv zeta.
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_calling calling avoid kept e));
    [ apply Hround | ].
  exact (refines_trans _ _ _ (Hround e) (IH _)).
Qed.

Lemma normalise_lifted_calling_program_refines (calling : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    program_refines e (normalise_lifted_calling calling rounds avoid kept e).
Proof.
  intros e. unfold normalise_lifted_calling.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (iterate_refines _ first_phase_round_refines rounds e))).
  apply (refines_trans _ _ _
    (iterate_lifted_calling_program_refines calling rounds avoid kept _)).
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (guard_scope_refines _ _
          (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines
             _)))).
  apply iterate_lifted_calling_program_refines.
Qed.

Theorem normalise_lifted_calling_unset :
  normalise_lifted_calling_unset_statement.
Proof. intros rounds avoid kept e. reflexivity. Qed.

(* One round of [normalise_lifted_calling], from what is proved of its
   rounds in turn: [iterate] of one round is that round. *)
Lemma normalise_lifted_round_calling_program_refines (calling : bool)
    (avoid : list continuation) (kept : list code_id) (e : expr) :
  program_refines e (normalise_lifted_round_calling calling avoid kept e).
Proof.
  generalize (iterate_lifted_calling_program_refines calling 1 avoid kept e).
  cbn [iterate]. cbv zeta.
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_calling calling avoid kept e));
    exact (fun H => H).
Qed.

(* The rounds that end with what the closures of the code of the unit
   are known to be used in the bodies of that code, where they are
   asked to: of a unit run on its own, as the rounds are. *)
Lemma iterate_lifted_own_program_refines (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    program_refines e
      (iterate rounds (normalise_lifted_round_own calling own avoid kept) e).
Proof.
  assert (Hround :
    ∀ e : expr,
      program_refines e
        (normalise_lifted_round_own calling own avoid kept e)).
  { intros e. unfold normalise_lifted_round_own. cbv zeta.
    apply (refines_trans _ _ _
      (normalise_lifted_round_calling_program_refines calling avoid kept e)).
    destruct own; [ | apply refines_refl ].
    exact (FlambdaClosureOfCodeProofs.know_own_closures_program_refines _). }
  induction rounds as [ | rounds IH]; intros e; cbn [iterate];
    [ apply refines_refl | ].
  cbv zeta.
  destruct
    (expr_eqb [] [] e (normalise_lifted_round_own calling own avoid kept e));
    [ apply Hround | ].
  exact (refines_trans _ _ _ (Hround e) (IH _)).
Qed.

Lemma normalise_lifted_own_program_refines (calling own : bool)
    (rounds : nat) (avoid : list continuation) (kept : list code_id) :
  ∀ e : expr,
    program_refines e (normalise_lifted_own calling own rounds avoid kept e).
Proof.
  intros e. unfold normalise_lifted_own.
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (iterate_refines _ first_phase_round_refines rounds e))).
  apply (refines_trans _ _ _
    (iterate_lifted_own_program_refines calling own rounds avoid kept _)).
  apply (refines_trans _ _ _
    (expr_refines_program _ _
       (guard_scope_refines _ _
          (FlambdaSimplifyCarriedProofs.simplify_by_types_carried_refines
             _)))).
  apply iterate_lifted_own_program_refines.
Qed.

Theorem normalise_lifted_own_unset : normalise_lifted_own_unset_statement.
Proof. intros calling rounds avoid kept e. reflexivity. Qed.

(* What ends the normalisation of a term in which something was
   lifted: the demotion with its rounds, and then the rounds that use
   what is known, each followed by the demotion with its rounds. *)
Lemma demote_and_tidy_known_round_program_refines (inner : nat) (e : expr) :
  program_refines e (demote_and_tidy_known_round inner e).
Proof.
  unfold demote_and_tidy_known_round.
  apply (refines_trans _ _ _
    (expr_refines_program _ _ (use_known_closures_refines _))).
  apply (refines_trans _ _ _ (with_known_program_refines _)).
  apply (refines_trans _ _ _ (read_known_slots_program_refines _)).
  apply (refines_trans _ _ _
    (FlambdaSharedReadsProofs.share_static_reads_program_refines _)).
  apply expr_refines_program, demote_and_tidy_refines.
Qed.

Lemma demote_and_tidy_known_program_refines (rounds : nat) (e : expr) :
  program_refines e (demote_and_tidy_known rounds e).
Proof.
  unfold demote_and_tidy_known.
  apply (refines_trans _ _ _
    (expr_refines_program _ _ (demote_and_tidy_refines rounds e))).
  generalize (demote_and_tidy rounds e). generalize rounds at 1 as n.
  induction n as [ | n IH]; intros e0; cbn [iterate];
    [ apply refines_refl | ].
  cbv zeta.
  destruct (expr_eqb [] [] e0 (demote_and_tidy_known_round rounds e0));
    [ apply demote_and_tidy_known_round_program_refines | ].
  exact
    (refines_trans _ _ _
       (demote_and_tidy_known_round_program_refines rounds e0) (IH _)).
Qed.

Lemma lifted_in_turn_program_refines (rounds : nat)
    (avoid : list continuation) (Ls : list lifting) (kept : list code_id) :
  ∀ e : expr, program_refines e (lifted_in_turn rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros e; cbn [lifted_in_turn].
  { apply (refines_trans _ _ _
      (read_known_slots_anywhere_program_refines e)).
    apply demote_and_tidy_known_program_refines. }
  apply (refines_trans _ _ _ (with_lifting_program_refines L _)).
  apply (refines_trans _ _ _
    (normalise_lifted_own_program_refines (made_closures_called L)
       (own_closures_known L) rounds avoid kept _)).
  apply IH.
Qed.

Lemma without_slots_program_refines (slots : list value_slot) (e : expr) :
  program_refines e (without_slots slots e).
Proof.
  unfold without_slots.
  destruct (drop_unused_slots slots e) as [e' | ] eqn:Hdrop;
    [ | apply refines_refl ].
  apply (refines_trans _ _ _
    (drop_unused_slots_program_refines slots e e' Hdrop)).
  apply expr_refines_program.
  eapply expr_refines_trans; [ apply remove_unused_lets_refines | ].
  apply rewrite_code_refines, remove_unused_lets_refines.
Qed.

(* The term with its code first, run on its own, behaves as the
   term. *)
Lemma with_code_first_program_equivalent (e : expr) :
  equivalent (State e [] [] empty_static_env empty_heap no_reads)
    (State (with_code_first e) [] [] empty_static_env empty_heap no_reads).
Proof.
  unfold with_code_first.
  destruct (hoist_code e) as [e' | ] eqn:Hhoist; [ | apply equivalent_refl ].
  apply (equivalent_trans _ _ _ (hoist_code_program_equivalent e e' Hhoist)).
  apply
    (equivalent_trans _ _ _ (drop_deleted_code_equivalent e' [] [] _ _ rd_)).
  apply merge_statics_equivalent.
Qed.

(* The term with its static bindings that need nothing first, run on
   its own, behaves as the term. *)
Lemma with_statics_first_program_equivalent (Ls : list lifting) (e : expr) :
  equivalent (State e [] [] empty_static_env empty_heap no_reads)
    (State (with_statics_first Ls e) [] [] empty_static_env empty_heap
       no_reads).
Proof.
  unfold with_statics_first.
  destruct Ls as [ | L Ls]; [ apply equivalent_refl | ].
  destruct (hoist_statics (lifting_unit L) e) as [e' | ] eqn:Hhoist;
    [ | apply equivalent_refl ].
  apply (equivalent_trans _ _ _
    (FlambdaHoistStaticsProofs.hoist_statics_program_equivalent _ e e'
       Hhoist)).
  apply
    (equivalent_trans _ _ _ (drop_deleted_code_equivalent e' [] [] _ _ rd_)).
  apply merge_statics_equivalent.
Qed.

Theorem validate_unit_sound_everywhere :
  validate_unit_sound_everywhere_statement.
Proof.
  intros rounds unused slots G Ls kept before after. unfold validate_unit.
  destruct (expr_eqb [] [] before after) eqn:Hsame.
  - intros _. apply expr_equivalent_refines, expr_eqb_sound, Hsame.
  - cbv zeta. intros Hvalidated.
    apply (expr_refines_trans _ _ _ (demote_stub_calls_refines before)).
    apply (expr_refines_trans _ _ _
      (cut_returns_refines (demote_stub_calls before))).
    apply (expr_refines_trans _ _ _
      (loopify_unit_refines G (cut_returns [] (demote_stub_calls before)))).
    revert Hvalidated. clear Hsame.
    generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
    clear before. intros before.
    set (a := tidied_after rounds (normalise_after rounds kept after)).
    set (taken := negb (keeps_a_return a)).
    destruct (expr_eqb [] []
                (order_statics_as a
                   (tidied_before rounds
                      (with_returns taken
                         (normalise_before rounds kept before))))
                a)
      eqn:Hnormalised.
    + intros _.
      exact (normalised_refines rounds kept taken before after Hnormalised).
    + destruct (expr_eqb [] []
                  (order_statics_as a
                     (without_statics unused
                        (tidied_before rounds
                           (with_returns taken
                              (normalise_before rounds kept before)))))
                  a);
        [ intros [H | H]; discriminate H | ].
      set (c := without_head_statics unused a).
      destruct (expr_eqb [] []
                  (order_statics_as c
                     (without_statics unused
                        (tidied_before rounds
                           (with_returns taken
                              (normalise_before rounds kept before)))))
                  c);
        [ intros [H | H]; discriminate H | ].
      destruct
        (expr_eqb [] []
           (order_statics_as (float_code (with_code_first c))
              (float_code
                 (without_statics unused
                    (without_slots slots
                       (tidied_before rounds
                          (with_returns taken
                             (lifted_in_turn rounds (unit_continuations Ls)
                                Ls kept before)))))))
           (float_code (with_code_first c)));
        [ intros [H | H]; discriminate H | ].
      destruct
        (expr_eqb [] []
           (order_statics_as (with_statics_first Ls c)
              (with_statics_first Ls
                 (without_statics unused
                    (without_slots slots
                       (tidied_before rounds
                          (with_returns taken
                             (lifted_in_turn rounds (unit_continuations Ls)
                                Ls kept before)))))))
           (with_statics_first Ls c));
        intros [H | H]; discriminate H.
Qed.

Theorem validate_unit_sound : validate_unit_sound_statement.
Proof.
  intros rounds unused slots G Ls kept before after. unfold validate_unit.
  destruct (expr_eqb [] [] before after) eqn:Hsame.
  - intros _.
    apply expr_refines_program, expr_equivalent_refines, expr_eqb_sound,
      Hsame.
  - cbv zeta. intros Hvalidated.
    apply (refines_trans _ _ _
      (expr_refines_program _ _ (demote_stub_calls_refines before))).
    apply (refines_trans _ _ _
      (expr_refines_program _ _
         (cut_returns_refines (demote_stub_calls before)))).
    apply (refines_trans _ _ _
      (expr_refines_program _ _
         (loopify_unit_refines G
            (cut_returns [] (demote_stub_calls before))))).
    revert Hvalidated. clear Hsame.
    generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
    clear before. intros before.
    set (a := tidied_after rounds (normalise_after rounds kept after)).
    set (taken := negb (keeps_a_return a)).
    destruct (expr_eqb [] []
                (order_statics_as a
                   (tidied_before rounds
                      (with_returns taken
                         (normalise_before rounds kept before))))
                a)
      eqn:Hnormalised.
    + intros _. apply expr_refines_program.
      exact (normalised_refines rounds kept taken before after Hnormalised).
    + assert (Hbefore :
        program_refines before
          (tidied_before rounds
             (with_returns taken (normalise_before rounds kept before)))).
      { apply expr_refines_program.
        apply (expr_refines_trans _ _ _
          (normalise_before_refines rounds kept before)).
        apply (expr_refines_trans _ _ _ (with_returns_refines taken _)).
        apply tidied_before_refines. }
      destruct (expr_eqb [] []
                  (order_statics_as a
                     (without_statics unused
                        (tidied_before rounds
                           (with_returns taken
                              (normalise_before rounds kept before)))))
                  a) eqn:Hremoved.
      * intros _. apply (refines_trans _ _ _ Hbefore).
        apply (refines_trans _ _ _
          (without_statics_program_refines unused _)).
        apply expr_refines_program, expr_equivalent_refines.
        exact (ordered_same_equivalent rounds kept _ after Hremoved).
      * assert (Hafter :
          equivalent (State after [] [] empty_static_env empty_heap no_reads)
            (State a [] [] empty_static_env empty_heap no_reads))
          by exact (tidied_after_normalised_equivalent rounds kept after []
                      [] empty_static_env _ rd_).
        set (c := without_head_statics unused a).
        assert (Hcleaned : program_refines c after).
        { apply (refines_trans _ _ _
            (without_head_statics_program_refines unused a)).
          apply equivalent_refines, equivalent_sym, Hafter. }
        destruct (expr_eqb [] []
                    (order_statics_as c
                       (without_statics unused
                          (tidied_before rounds
                             (with_returns taken
                                (normalise_before rounds kept before)))))
                    c) eqn:Hremoved_head;
          [ intros _; apply (refines_trans _ _ _ Hbefore);
            apply (refines_trans _ _ _
              (without_statics_program_refines unused _));
            apply (refines_trans _
              (State c [] [] empty_static_env empty_heap no_reads));
              [ | exact Hcleaned ];
            apply expr_refines_program, expr_equivalent_refines;
            apply (expr_equivalent_trans _ _ _
              (order_statics_as_equivalent c _));
            exact (expr_eqb_sound _ _ Hremoved_head)
          | ].
        set (lifted :=
          without_statics unused
            (without_slots slots
               (tidied_before rounds
                  (with_returns taken
                     (lifted_in_turn rounds (unit_continuations Ls)
                        Ls kept before))))).
        assert (Hlifting : program_refines before lifted).
        { apply (refines_trans _ _ _
            (lifted_in_turn_program_refines rounds (unit_continuations Ls) Ls
               kept _)).
          apply (refines_trans _ _ _
            (expr_refines_program _ _ (with_returns_refines taken _))).
          apply (refines_trans _ _ _
            (expr_refines_program _ _ (tidied_before_refines rounds _))).
          apply (refines_trans _ _ _ (without_slots_program_refines slots _)).
          apply without_statics_program_refines. }
        destruct
          (expr_eqb [] []
             (order_statics_as (float_code (with_code_first c))
                (float_code lifted))
             (float_code (with_code_first c))) eqn:Hlifted.
        -- intros _. apply (refines_trans _ _ _ Hlifting).
           apply (refines_trans _
             (State (with_code_first c) [] [] empty_static_env empty_heap
                no_reads)).
           { apply expr_refines_program, expr_equivalent_refines.
             apply (expr_equivalent_trans _ _ _ (float_code_equivalent _)).
             apply (expr_equivalent_trans _ _ _
               (order_statics_as_equivalent
                  (float_code (with_code_first c)) _)).
             apply (expr_equivalent_trans _ _ _ (expr_eqb_sound _ _ Hlifted)).
             apply expr_equivalent_sym, float_code_equivalent. }
           apply (refines_trans _
             (State c [] [] empty_static_env empty_heap no_reads));
             [ | exact Hcleaned ].
           apply equivalent_refines, equivalent_sym.
           apply with_code_first_program_equivalent.
        -- destruct
             (expr_eqb [] []
                (order_statics_as (with_statics_first Ls c)
                   (with_statics_first Ls lifted))
                (with_statics_first Ls c)) eqn:Hfirst;
             [ intros _ | intros H; contradiction H; reflexivity ].
           apply (refines_trans _ _ _ Hlifting).
           apply (refines_trans _ _ _
             (equivalent_refines _ _
                (with_statics_first_program_equivalent Ls lifted))).
           apply (refines_trans _
             (State (with_statics_first Ls c) [] [] empty_static_env
                empty_heap no_reads)).
           { apply expr_refines_program, expr_equivalent_refines.
             apply (expr_equivalent_trans _ _ _
               (order_statics_as_equivalent (with_statics_first Ls c) _)).
             exact (expr_eqb_sound _ _ Hfirst). }
           apply (refines_trans _
             (State c [] [] empty_static_env empty_heap no_reads));
             [ | exact Hcleaned ].
           apply equivalent_refines, equivalent_sym.
           apply with_statics_first_program_equivalent.
Qed.
