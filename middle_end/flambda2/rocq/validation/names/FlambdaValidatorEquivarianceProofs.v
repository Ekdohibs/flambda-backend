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


(* Proofs of the statements of FlambdaValidatorEquivarianceSpec.v. Nothing
   here needs review.

   Each rewrite is taken as the two of FlambdaEquivarianceProofs.v are:
   one lemma for each function the rewrite is made of, saying that the
   function commutes with the action, and one recursion over expressions
   (and over handlers, bindings and code where the rewrite goes into
   them). A test that compares names for equality answers the same on
   the renamed names because a renaming is injective. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaSemanticsProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec FlambdaAlphaProofs
  FlambdaApplySpec FlambdaNormaliseSpec FlambdaEquivarianceSpec
  FlambdaEquivarianceProofs FlambdaUnusedContsSpec FlambdaUnusedStaticsSpec
  FlambdaUnusedHeadStaticsSpec
  FlambdaPrimitiveModel FlambdaPrimitiveSpec FlambdaBlockLoadSpec
  FlambdaBindersSpec FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaUnboxSpec FlambdaWellFormedSpec FlambdaWellKindedSpec
  FlambdaCompareSpec FlambdaCompareAlphaSpec FlambdaCompareAlphaProofs
  FlambdaMyClosureSpec FlambdaMyClosureProofs FlambdaLiftSpec
  FlambdaLiftProofs FlambdaLiftFromCodeSpec FlambdaLiftFromCodeProofs
  FlambdaLiftAroundSpec
  FlambdaLiftFromArmsSpec FlambdaStaticsInScopeSpec FlambdaHoistStaticsSpec
  FlambdaUnusedSlotsSpec FlambdaUnusedSlotsProofs
  FlambdaKnownSymbolsSpec FlambdaKnownSymbolsProofs
  FlambdaKnownClosuresSpec FlambdaKnownClosuresProofs FlambdaFloatCodeSpec
  FlambdaFloatCodeProofs FlambdaGroupsUpSpec FlambdaGroupsUpProofs
  FlambdaHoistCodeSpec FlambdaHoistCodeProofs
  FlambdaInlineFreshProofs FlambdaRecHandlersSpec
  FlambdaSinkContsSpec FlambdaSinkUsesSpec FlambdaShortcutSpec
  FlambdaSameHandlersSpec FlambdaUnusedParamsSpec
  FlambdaLoopAliasesSpec FlambdaLoopLetsSpec FlambdaParamAliasesSpec
  FlambdaForwardSpec FlambdaForwardJumpsSpec FlambdaForwardKindedSpec
  FlambdaForwardArmSpec
  FlambdaInlineTailSpec FlambdaDeadParamsSpec
  FlambdaCodeLetsSpec
  FlambdaLoopifySpec
  FlambdaCseSpec FlambdaCseProofs
  FlambdaMadeLengthsSpec FlambdaMadeLengthsProofs
  FlambdaRecoverComparisonSpec FlambdaRecoverComparisonProofs
  FlambdaSameComparisonSpec FlambdaSameComparisonProofs
  FlambdaShareProjectionsSpec FlambdaShareProjectionsProofs
  FlambdaIdentitiesSpec FlambdaIdentitiesProofs
  FlambdaMergeSwitchesSpec FlambdaMergeSwitchesProofs
  FlambdaMergeTablesSpec FlambdaMergeTablesProofs FlambdaStaticsOrderSpec
  FlambdaStaticsOrderProofs FlambdaRenameApartSpec FlambdaRenameApartProofs
  FlambdaRenameApartCanonicalProofs
  FlambdaInlineSiblingsSpec FlambdaInlineSiblingsProofs
  FlambdaUnusedProjectionsSpec FlambdaReadsAfterSpec
  FlambdaUnboxVariantsSpec
  FlambdaUnaliasParamsSpec FlambdaWrapReturnsSpec FlambdaUntagParamsSpec
  FlambdaGiveScrutineeSpec FlambdaArmNumbersSpec FlambdaValidateSpec
  FlambdaValidatorEquivarianceSpec FlambdaValidatorEquivarianceBaseProofs.
From OxCamlFlambda2 Require FlambdaKeepsScopeProofs FlambdaValidateScopeProofs
  FlambdaLoopsScopeProofs
  FlambdaStaticsScopeProofs FlambdaLiftFromArmsProofs
  FlambdaLiftAroundProofs
  FlambdaHoistStaticsProofs FlambdaUnboxParamsNamesProofs
  FlambdaUnaliasParamsNamesProofs FlambdaWrapReturnsNamesProofs
  FlambdaUntagParamsNamesProofs FlambdaGiveScrutineeNamesProofs
  FlambdaArmNumbersNamesProofs FlambdaShadowParamsNamesProofs
  FlambdaTwinParamsNamesProofs FlambdaBindConstParamsNamesProofs
  FlambdaAliasLetsProofs.
From OxCamlFlambda2 Require FlambdaLiftContsNamesProofs.
From OxCamlFlambda2 Require Import FlambdaUnaliasRecParamsSpec.
From OxCamlFlambda2 Require FlambdaUnaliasRecParamsNamesProofs.
From OxCamlFlambda2 Require Import FlambdaDemoteSpec.
From OxCamlFlambda2 Require FlambdaDemoteProofs.
From OxCamlFlambda2 Require Import FlambdaLiftObservedSpec.
From OxCamlFlambda2 Require Import FlambdaKnownSlotsSpec.
From OxCamlFlambda2 Require FlambdaKnownSiblingsProofs.
From OxCamlFlambda2 Require FlambdaKnownSlotsProofs.
From OxCamlFlambda2 Require FlambdaReadsAfterProofs.
From OxCamlFlambda2 Require FlambdaSharedReadsSpec FlambdaSharedReadsProofs.
From OxCamlFlambda2 Require Import FlambdaStubCallsSpec.
From OxCamlFlambda2 Require FlambdaStubCallsProofs.
From OxCamlFlambda2 Require FlambdaCopyCodeProofs.
From OxCamlFlambda2 Require Import FlambdaClosureCallsSpec.
From OxCamlFlambda2 Require FlambdaClosureCallsProofs.
From OxCamlFlambda2 Require Import FlambdaClosureOfCodeSpec.
From OxCamlFlambda2 Require FlambdaClosureOfCodeProofs.
From OxCamlFlambda2 Require Import FlambdaNeverReturnsSpec.
From OxCamlFlambda2 Require FlambdaNeverReturnsProofs
  FlambdaNeverReturnsScopeProofs.
From OxCamlFlambda2 Require Import FlambdaUnboxParamsSpec
  FlambdaUnboxVariantsSpec FlambdaUnboxInScopeSpec FlambdaUnboxInScopeProofs
  FlambdaUnboxInScopeRenameSpec FlambdaUnboxInScopeRenameProofs
  FlambdaUnboxLoopsSpec FlambdaUnboxTogetherSpec.
From OxCamlFlambda2 Require FlambdaUnboxTogetherNamesProofs.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
From OxCamlFlambda2 Require FlambdaGuardScopeSpec.
From OxCamlFlambda2Types Require FlambdaSimplifyCarriedSpec
  FlambdaSimplifyCarriedProofs.
Import ListNotations.

(* ======================================================================== *)
(* Lists under a map                                                        *)
(* ======================================================================== *)

Lemma forallb_map_with {A B : Type} (f : B -> bool) (g : A -> B)
    (h : A -> bool) (l : list A) :
  (∀ a : A, f (g a) = h a) -> forallb f (map g l) = forallb h l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

Lemma filter_map_with {A B : Type} (keep' : B -> bool) (g : A -> B)
    (keep : A -> bool) (l : list A) :
  (∀ a : A, keep' (g a) = keep a) ->
  filter keep' (map g l) = map g (filter keep l).
Proof.
  intros Hkeep. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hkeep, IH. destruct (keep a); reflexivity.
Qed.

(* A test of equality answers the same on the images of an injective
   function. *)
Lemma dec_injective {A B : Type} (dec : ∀ a a' : A, {a = a'} + {a <> a'})
    (g : A -> A) (a a' : A) (yes no : B) :
  (∀ a a' : A, g a = g a' -> a = a') ->
  (if dec (g a) (g a') then yes else no) = (if dec a a' then yes else no).
Proof.
  intros Hinjective.
  destruct (dec a a') as [-> | Hne], (dec (g _) (g _)) as [Heq | Hne'];
    try reflexivity; [ contradiction | ].
  elim Hne. exact (Hinjective _ _ Heq).
Qed.

(* ======================================================================== *)
(* A renaming is injective                                                  *)
(* ======================================================================== *)

Lemma rename_simple_injective (r : renaming) (s s' : simple) :
  rename_simple r s = rename_simple r s' -> s = s'.
Proof.
  destruct s, s'; simpl; intros Heq; try discriminate; try exact Heq.
  injection Heq as Heq. apply rename_variable_injective in Heq. congruence.
Qed.

Lemma rename_simples_injective (r : renaming) (args args' : list simple) :
  map (rename_simple r) args = map (rename_simple r) args' -> args = args'.
Proof.
  revert args'.
  induction args as [ | s args IH]; intros [ | s' args']; simpl; intros Heq;
    try discriminate; [ reflexivity | ].
  injection Heq as Hs Hargs.
  apply rename_simple_injective in Hs. apply IH in Hargs. congruence.
Qed.

Lemma rename_trap_action_injective (r : renaming)
    (action action' : option trap_action) :
  option_map (rename_trap_action r) action
  = option_map (rename_trap_action r) action' ->
  action = action'.
Proof.
  destruct action as [[k | k kind] | ], action' as [[k' | k' kind'] | ];
    simpl; intros Heq; try discriminate; try reflexivity;
    injection Heq as Hk; apply rename_continuation_injective in Hk;
    congruence.
Qed.

Lemma no_trap_action_renamed (r : renaming) (ac : apply_cont) :
  no_trap_action (rename_apply_cont r ac) = no_trap_action ac.
Proof.
  unfold no_trap_action. simpl. destruct (ac_trap_action ac); reflexivity.
Qed.

Lemma no_exn_extra_args_renamed (r : renaming) (ap : apply_expr) :
  no_exn_extra_args (rename_apply r ap) = no_exn_extra_args ap.
Proof.
  unfold no_exn_extra_args. simpl.
  destruct (ap_exn_extra_args ap); reflexivity.
Qed.

Lemma rename_apply_cont_injective (r : renaming) (ac ac' : apply_cont) :
  rename_apply_cont r ac = rename_apply_cont r ac' -> ac = ac'.
Proof.
  destruct ac as [k args trap], ac' as [k' args' trap'].
  unfold rename_apply_cont. simpl. intros Heq.
  injection Heq as Hk Hargs Htrap.
  apply rename_continuation_injective in Hk.
  apply rename_simples_injective in Hargs.
  apply rename_trap_action_injective in Htrap. congruence.
Qed.

(* ======================================================================== *)
(* The checks of FlambdaNormaliseSpec.v                                     *)
(* ======================================================================== *)

Theorem expr_in_scope_equivariant : expr_in_scope_equivariant_statement.
Proof.
  intros r vs ks e. apply FlambdaEquivarianceProofs.expr_in_scope_equivariant.
Qed.

Theorem inlinable_b_equivariant : inlinable_b_equivariant_statement.
Proof. exact FlambdaEquivarianceProofs.inlinable_b_equivariant. Qed.

Theorem tupled_inlinable_b_equivariant :
  tupled_inlinable_b_equivariant_statement.
Proof. exact FlambdaEquivarianceProofs.tupled_inlinable_b_equivariant. Qed.

Theorem var_free_b_equivariant : var_free_b_equivariant_statement.
Proof. exact var_free_b_equivariant_aux. Qed.

(* ======================================================================== *)
(* Removing unused [Let]s                                                   *)
(* ======================================================================== *)

(* The laws of renaming of the four rewrites that remove unused
   [Let]s are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states and proves by itself the
   laws of the tests that the first two make of a named expression. It
   is given the laws of the free names of
   FlambdaValidatorEquivarianceBaseProofs.v, which are proved by hand
   there: the tactic derives them where they are not given, and is
   slower. In this section the tactic rewrites with its database in
   one pass over the goal, which is quicker and is enough for every
   proof here. *)
From OxCamlFlambda2 Require Import FlambdaEquivarianceTactic.

Section Unused_lets.

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Hint Rewrite var_in_simple_equivariant var_in_named_equivariant
  var_free_b_equivariant_aux cont_free_b_equivariant_aux : equivariance.

Lemma remove_unused_lets_equivariant_aux (r : renaming) (e : expr) :
  remove_unused_lets (rename_expr r e) = rename_expr r (remove_unused_lets e).
Proof. equivariance. Qed.

Theorem remove_unused_lets_equivariant :
  remove_unused_lets_equivariant_statement.
Proof. exact remove_unused_lets_equivariant_aux. Qed.

Theorem remove_unused_total_lets_equivariant :
  remove_unused_total_lets_equivariant_statement.
Proof. equivariance. Qed.

(* The variables in scope with their kinds are a list of pairs, whose
   type does not say how it is renamed: the action is
   [rename_kinds_in_scope]. The laws of [named_defined] and of
   [params_in_scope] are given to the tactic once proved; other parts
   of this file use the three laws by name.

   The law of [params_in_scope], which adds the parameters of a
   handler one by one, is proved by hand. The tactic does not prove
   it: the hypothesis of its induction is of the variables in scope
   renamed, and the variables in scope of the next step are a renamed
   pair put before them. *)
#[local] Instance kinds_in_scope_renaming : Renaming_of kinds_in_scope :=
  rename_kinds_in_scope.

Lemma kind_of_simple_in_scope_equivariant (r : renaming)
    (kinds : kinds_in_scope) (s : simple) :
  kind_of_simple_in_scope (rename_kinds_in_scope r kinds) (rename_simple r s)
  = kind_of_simple_in_scope kinds s.
Proof. equivariance. Qed.

Lemma named_defined_equivariant (r : renaming) (kinds : kinds_in_scope)
    (kind : flambda_kind) (d : named) :
  named_defined (rename_kinds_in_scope r kinds) kind (rename_named r d)
  = named_defined kinds kind d.
Proof. equivariance. Qed.

Lemma params_in_scope_equivariant (r : renaming) (params : list variable) :
  ∀ (params_kinds : list flambda_kind) (kinds : kinds_in_scope),
    params_in_scope (map (rename_variable r) params) params_kinds
      (rename_kinds_in_scope r kinds)
    = rename_kinds_in_scope r (params_in_scope params params_kinds kinds).
Proof.
  induction params as [ | param params IH];
    intros [ | kind params_kinds] kinds; simpl; try reflexivity.
  exact (IH params_kinds ((param, kind) :: kinds)).
Qed.

#[local] Hint Rewrite named_defined_equivariant params_in_scope_equivariant
  : equivariance.

(* At a [Let] the variables in scope are one pair more. That a renamed
   pair before the renamed variables in scope is the renaming of the
   pair before them holds by computation; it is given to the tactic,
   and computation is kept from undoing it. *)
Lemma kinds_in_scope_cons_renamed (r : renaming) (x : variable)
    (kind : flambda_kind) (kinds : kinds_in_scope) :
  (rename_variable r x, kind) :: rename_kinds_in_scope r kinds
  = rename_kinds_in_scope r ((x, kind) :: kinds).
Proof. reflexivity. Qed.

Section Defined_lets.

#[local] Arguments rename_kinds_in_scope : simpl never.
#[local] Hint Rewrite kinds_in_scope_cons_renamed : equivariance.

Lemma remove_unused_defined_lets_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ kinds : kinds_in_scope,
    remove_unused_defined_lets (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets kinds e).
Proof. equivariance. Qed.

End Defined_lets.

Theorem remove_unused_defined_lets_equivariant :
  remove_unused_defined_lets_equivariant_statement.
Proof.
  intros r kinds e. apply remove_unused_defined_lets_equivariant_aux.
Qed.

(* The projections of function slots that nothing uses. What is known
   of the sets of closures is renamed by [rename_known_sets]. The laws
   of [sets_under_let] and of [projection_defined] are stated so that
   the rewrite does not derive them at each [Let].

   The law of [forget_sets], which forgets the variables of a list one
   by one, is proved by hand, for the reason given of
   [params_in_scope]. *)
#[local] Instance known_sets_renaming : Renaming_of known_sets :=
  rename_known_sets.

Lemma forget_set_renamed (r : renaming) (x : variable) (known : known_sets) :
  forget_set (rename_variable r x) (rename_known_sets r known)
  = rename_known_sets r (forget_set x known).
Proof. equivariance. Qed.

Lemma forget_sets_renamed (r : renaming) (xs : list variable) :
  ∀ known : known_sets,
    forget_sets (map (rename_variable r) xs) (rename_known_sets r known)
    = rename_known_sets r (forget_sets xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_set_renamed. apply IH.
Qed.

#[local] Hint Rewrite forget_set_renamed forget_sets_renamed : equivariance.

Lemma sets_under_let_renamed (r : renaming) (known : known_sets)
    (x : variable) (d : named) :
  sets_under_let (rename_known_sets r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_sets r (sets_under_let known x d).
Proof. equivariance. Qed.

Lemma projection_defined_renamed (r : renaming) (known : known_sets)
    (d : named) :
  projection_defined (rename_known_sets r known) (rename_named r d)
  = projection_defined known d.
Proof. equivariance. Qed.

#[local] Hint Rewrite sets_under_let_renamed projection_defined_renamed
  : equivariance.

Theorem remove_projections_knowing_equivariant :
  remove_projections_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem remove_unused_projections_equivariant :
  remove_unused_projections_equivariant_statement.
Proof.
  intros r e. exact (remove_projections_knowing_equivariant r [] e).
Qed.

End Unused_lets.

(* ======================================================================== *)
(* A continuation applied once, at the end of its scope                     *)
(* ======================================================================== *)

(* The laws of renaming of [inline_cont_at_tail] and of the two
   rewrites that are made of it, FlambdaNormaliseSpec.inline_tail_conts
   and FlambdaInlineTailSpec.inline_tail_conts_kinded, in two ways side
   by side, to be compared.

   The first way: each law is proved from the definitions by the
   tactic of FlambdaEquivarianceTactic.v. The tactic is given one law,
   that of [var_free_b], which is proved by hand in
   FlambdaValidatorEquivarianceBaseProofs.v: where it is not given the
   tactic derives it, and the first proof is slower. The law of
   [inline_cont_at_tail], once proved, is given to the tactic for the
   two rewrites. *)
From OxCamlFlambda2 Require Import FlambdaEquivarianceTactic
  FlambdaTermInductionProofs FlambdaSpineWalkSpec FlambdaSpineWalkNamesProofs.

Section Inline_tail.

#[local] Hint Rewrite var_free_b_equivariant_aux : equivariance.

Lemma inline_cont_at_tail_equivariant_aux (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata) (handler : expr) :
  ∀ body : expr,
    inline_cont_at_tail (rename_continuation r k)
      (map (rename_variable r) params) metadata (rename_expr r handler)
      (rename_expr r body)
    = option_map (rename_expr r)
        (inline_cont_at_tail k params metadata handler body).
Proof. equivariance. Qed.
#[local] Hint Rewrite inline_cont_at_tail_equivariant_aux : equivariance.

Theorem inline_cont_at_tail_equivariant :
  inline_cont_at_tail_equivariant_statement.
Proof.
  intros r k params metadata handler body.
  apply inline_cont_at_tail_equivariant_aux.
Qed.

Lemma inline_tail_conts_equivariant_aux (r : renaming) (e : expr) :
  inline_tail_conts (rename_expr r e) = rename_expr r (inline_tail_conts e).
Proof. equivariance. Qed.

Theorem inline_tail_conts_equivariant :
  inline_tail_conts_equivariant_statement.
Proof. exact inline_tail_conts_equivariant_aux. Qed.

(* Inlining a continuation at the end of its scope, where its handler
   gives one kind for each parameter. *)
Lemma inline_tail_conts_kinded_equivariant_aux (r : renaming) (e : expr) :
  inline_tail_conts_kinded (rename_expr r e)
  = rename_expr r (inline_tail_conts_kinded e).
Proof. equivariance. Qed.

Lemma inline_tail_conts_kinded_equivariant :
  rewrite_equivariant inline_tail_conts_kinded.
Proof. exact inline_tail_conts_kinded_equivariant_aux. Qed.

(* The second way: each of the two rewrites is the walk over the spine
   of a term (FlambdaSpineWalkSpec.walk_spine) of a rewrite of one
   node, which is not recursive. At a non-recursive [Let_cont] whose
   handler and body the walk has rewritten, the step puts the handler
   in the place of the application that ends the body, where
   [inline_cont_at_tail] gives it; it leaves every other node as it
   is. The law of renaming of a rewrite is then that of its step, by
   FlambdaSpineWalkNamesProofs.walk_spine_equivariant, and the tactic
   proves the law of a step.

   The definitions of the two rewrites are not written as walks: they
   are as they were, in FlambdaNormaliseSpec.v and
   FlambdaInlineTailSpec.v, and that each is the walk of its step is a
   lemma here. To write them as walks would change those two files
   and what is extracted from them, and the other laws of the two
   rewrites would be had through the equation, or proved again of the
   step. *)
Definition inline_tail_step (e : expr) : expr :=
  match e with
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    match inline_cont_at_tail k params metadata handler body with
    | Some inlined => inlined
    | None => e
    end
  | _ => e
  end.

Definition inline_tail_kinded_step (e : expr) : expr :=
  match e with
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    if Nat.eqb (length params) (length (hm_params_kinds metadata))
    then
      match inline_cont_at_tail k params metadata handler body with
      | Some inlined => inlined
      | None => e
      end
    else e
  | _ => e
  end.

Lemma inline_tail_conts_as_walk (e : expr) :
  inline_tail_conts e = walk_spine inline_tail_step e.
Proof.
  induction e using expr_deep_ind; cbn [inline_tail_conts walk_spine];
    rewrite <- ?IHe, <- ?IHe1, <- ?IHe2; reflexivity.
Qed.

Lemma inline_tail_conts_kinded_as_walk (e : expr) :
  inline_tail_conts_kinded e = walk_spine inline_tail_kinded_step e.
Proof.
  induction e using expr_deep_ind; cbn [inline_tail_conts_kinded walk_spine];
    rewrite <- ?IHe, <- ?IHe1, <- ?IHe2; reflexivity.
Qed.

Lemma inline_tail_step_equivariant : rewrite_equivariant inline_tail_step.
Proof. equivariance. Qed.

Lemma inline_tail_kinded_step_equivariant :
  rewrite_equivariant inline_tail_kinded_step.
Proof. equivariance. Qed.

Lemma inline_tail_conts_renamed_by_walk :
  rewrite_equivariant inline_tail_conts.
Proof.
  intros r e. rewrite !inline_tail_conts_as_walk.
  apply (walk_spine_equivariant _ inline_tail_step_equivariant).
Qed.

Lemma inline_tail_conts_kinded_renamed_by_walk :
  rewrite_equivariant inline_tail_conts_kinded.
Proof.
  intros r e. rewrite !inline_tail_conts_kinded_as_walk.
  apply (walk_spine_equivariant _ inline_tail_kinded_step_equivariant).
Qed.

End Inline_tail.

(* ======================================================================== *)
(* Static bindings: deleted code, and groups                                *)
(* ======================================================================== *)

Lemma binds_something_renamed (r : renaming) (binding : static_binding) :
  binds_something (rename_static_binding r binding)
  = binds_something binding.
Proof. destruct binding; reflexivity. Qed.

Lemma let_static_if_any_equivariant (r : renaming)
    (bindings : list static_binding) (body : expr) :
  let_static_if_any (map (rename_static_binding r) bindings)
    (rename_expr r body)
  = rename_expr r (let_static_if_any bindings body).
Proof. destruct bindings; reflexivity. Qed.

Fixpoint drop_deleted_code_equivariant_aux (r : renaming) (e : expr) :
  drop_deleted_code (rename_expr r e) = rename_expr r (drop_deleted_code e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite drop_deleted_code_equivariant_aux. reflexivity.
  - rewrite (filter_map_with _ _ _ _ (binds_something_renamed r)),
      drop_deleted_code_equivariant_aux.
    apply let_static_if_any_equivariant.
  - rewrite !drop_deleted_code_equivariant_aux. reflexivity.
  - rewrite drop_deleted_code_equivariant_aux. reflexivity.
Qed.

Theorem drop_deleted_code_equivariant :
  drop_deleted_code_equivariant_statement.
Proof. exact drop_deleted_code_equivariant_aux. Qed.

(* The law of [merge_statics] is proved from the definitions by the
   tactic of FlambdaEquivarianceTactic.v, which states and proves the
   laws of [binds_code] and of [code_first] as it goes. The law of
   [drop_deleted_code] above stays by hand: the tactic leaves the case
   of [Let_static], where the bindings are filtered. *)
Lemma merge_statics_equivariant_aux (r : renaming) (e : expr) :
  merge_statics (rename_expr r e) = rename_expr r (merge_statics e).
Proof. equivariance. Qed.

Theorem merge_statics_equivariant : merge_statics_equivariant_statement.
Proof. exact merge_statics_equivariant_aux. Qed.

(* ======================================================================== *)
(* Inlining calls of symbols known to be closures                           *)
(* ======================================================================== *)

Theorem call_agrees_b_equivariant : call_agrees_b_equivariant_statement.
Proof. exact call_agrees_b_renamed. Qed.

Theorem inline_known_calls_equivariant :
  inline_known_calls_equivariant_statement.
Proof.
  intros r kept known_code known_symbols e.
  apply inline_known_calls_equivariant_aux.
Qed.

(* ======================================================================== *)
(* Switches                                                                 *)
(* ======================================================================== *)

(* The laws of the two rewrites, and that of [arms_apply], are proved
   from the definitions by the tactic of FlambdaEquivarianceTactic.v.
   Each rewrite is given the law of what it does at one node,
   [collapse_switch] and [thread_switch], which stay by hand: the
   first compares applications of continuations for equality, and at
   the second the tactic does not end. In a section, so that what is
   given to the tactic here is given to it here only. *)
Section Switches.

Lemma collapse_switch_equivariant (r : renaming) (scrutinee : simple)
    (arms : list (Z * apply_cont)) :
  collapse_switch (rename_simple r scrutinee)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = rename_expr r (collapse_switch scrutinee arms).
Proof.
  destruct arms as [ | [i ac] others]; [ reflexivity | ].
  unfold collapse_switch. cbn [map fst snd].
  rewrite
    (forallb_map_with _ _
       (fun arm => if apply_cont_eq_dec (snd arm) ac then true else false)).
  - destruct
      (forallb
         (fun arm => if apply_cont_eq_dec (snd arm) ac then true else false)
         others);
      reflexivity.
  - intros arm. cbn [snd].
    apply (dec_injective apply_cont_eq_dec (rename_apply_cont r)).
    apply rename_apply_cont_injective.
Qed.

#[local] Hint Rewrite collapse_switch_equivariant : equivariance.
#[local] Arguments collapse_switch : simpl never.

Lemma collapse_switches_equivariant_aux (r : renaming) (e : expr) :
  collapse_switches (rename_expr r e) = rename_expr r (collapse_switches e).
Proof. equivariance. Qed.

Theorem collapse_switches_equivariant :
  collapse_switches_equivariant_statement.
Proof. exact collapse_switches_equivariant_aux. Qed.

Lemma thread_arm_equivariant (r : renaming) (k : continuation)
    (inner : list (Z * apply_cont)) (arm : Z * apply_cont) :
  thread_arm (rename_continuation r k)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) inner)
    (fst arm, rename_apply_cont r (snd arm))
  = (fst (thread_arm k inner arm),
     rename_apply_cont r (snd (thread_arm k inner arm))).
Proof.
  destruct arm as [i ac]. unfold thread_arm. cbn [fst snd].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  destruct (ac_args ac) as [ | arg args]; [ | reflexivity ]. cbn [map].
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  rewrite rename_continuation_eqb, no_trap_action_renamed.
  destruct (Pos.eqb (ac_continuation ac) k && no_trap_action ac);
    [ | reflexivity ].
  rewrite find_arm_renamed.
  destruct (find_arm i inner) as [ac' | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite among_apply_cont_continuations_renamed.
  destruct (among k (apply_cont_continuations ac')); reflexivity.
Qed.

Lemma arms_apply_equivariant (r : renaming) (k : continuation)
    (arms : list (Z * apply_cont)) :
  arms_apply (rename_continuation r k)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = arms_apply k arms.
Proof. equivariance. Qed.

Lemma thread_switch_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  thread_switch (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_expr r (thread_switch k h body).
Proof.
  destruct h as [[ | param params] metadata handler]; [ | reflexivity ].
  destruct handler as [ | | | | | | inner_scrutinee inner | ];
    try reflexivity.
  destruct body as [ | | | | | | scrutinee arms | ]; try reflexivity.
  cbn [thread_switch rename_handler rename_expr map].
  destruct (hm_params_kinds metadata); [ | reflexivity ].
  rewrite (dec_injective simple_eq_dec (rename_simple r))
    by apply rename_simple_injective.
  destruct (simple_eq_dec scrutinee inner_scrutinee); [ | reflexivity ].
  rewrite !map_map.
  rewrite
    (map_ext _
       (fun arm =>
          (fst (thread_arm k inner arm),
           rename_apply_cont r (snd (thread_arm k inner arm))))
       (thread_arm_equivariant r k inner)).
  rewrite <- (map_map (thread_arm k inner)
                (fun arm => (fst arm, rename_apply_cont r (snd arm)))).
  rewrite arms_apply_equivariant.
  destruct (arms_apply k (map (thread_arm k inner) arms)); reflexivity.
Qed.

#[local] Hint Rewrite thread_switch_equivariant : equivariance.
#[local] Arguments thread_switch : simpl never.

Lemma thread_switches_equivariant_aux (r : renaming) (e : expr) :
  thread_switches (rename_expr r e) = rename_expr r (thread_switches e).
Proof. equivariance. Qed.

Theorem thread_switches_equivariant :
  thread_switches_equivariant_statement.
Proof. exact thread_switches_equivariant_aux. Qed.

End Switches.

(* ======================================================================== *)
(* Rewriting the bodies of code                                             *)
(* ======================================================================== *)

(* The two laws are proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, for a rewrite [f] that the section
   takes with its law. The law of [rewrite_code_body] is stated, and
   is then a hint for the law of [rewrite_code]: the tactic cannot
   read it from the type of [rewrite_code_body], which says nothing
   of [f]. *)
Section Rewrite_code.

Variable f : expr -> expr.
Hypothesis f_equivariant : rewrite_equivariant f.

Lemma rewrite_code_body_equivariant (r : renaming) (body : expr) :
  rewrite_code_body f (rename_expr r body)
  = rename_expr r (rewrite_code_body f body).
Proof. equivariance. Qed.

#[local] Hint Rewrite rewrite_code_body_equivariant : equivariance.

Lemma rewrite_code_equivariant_aux (r : renaming) (e : expr) :
  rewrite_code f (rename_expr r e) = rename_expr r (rewrite_code f e).
Proof. equivariance. Qed.

End Rewrite_code.

Theorem rewrite_code_equivariant : rewrite_code_equivariant_statement.
Proof.
  intros f f_equivariant r e.
  apply rewrite_code_equivariant_aux, f_equivariant.
Qed.

(* ======================================================================== *)
(* FlambdaUnusedContsSpec.v                                                 *)
(* ======================================================================== *)

(* The law of [remove_unused_conts] is proved from the definitions by
   the tactic of FlambdaEquivarianceTactic.v. It is given the law of
   [cont_free_b], which is proved by hand in
   FlambdaValidatorEquivarianceBaseProofs.v; in a section, so that it
   is given here only. *)
Section Unused_conts.

#[local] Hint Rewrite cont_free_b_equivariant_aux : equivariance.

Theorem cont_free_b_equivariant : cont_free_b_equivariant_statement.
Proof. exact cont_free_b_equivariant_aux. Qed.

Lemma remove_unused_conts_equivariant_aux (r : renaming) (e : expr) :
  remove_unused_conts (rename_expr r e)
  = rename_expr r (remove_unused_conts e).
Proof. equivariance. Qed.

Theorem remove_unused_conts_equivariant :
  remove_unused_conts_equivariant_statement.
Proof. exact remove_unused_conts_equivariant_aux. Qed.

End Unused_conts.

(* ======================================================================== *)
(* FlambdaUnusedStaticsSpec.v                                               *)
(* ======================================================================== *)

(* The laws of this section are proved from the definitions by the
   tactic of FlambdaEquivarianceTactic.v, but for two that are of what
   the machine holds and not of a term. The names removed are symbols
   and code ids, which no renaming moves, so the argument that holds
   them is as it was on both sides of a law. *)
Section Dropped.

(* [keep_map] as a function of lists that the tactic goes under. *)
Lemma keep_map_flat_map {A : Type} (keep : A -> bool) (f : A -> A)
    (l : list A) :
  FlambdaUnusedStaticsSpec.keep_map keep f l
  = flat_map (fun a => if keep a then [f a] else []) l.
Proof.
  induction l as [ | a l IH]; [ reflexivity | ]. cbn. rewrite IH.
  destruct (keep a); reflexivity.
Qed.
#[local] Hint Rewrite @keep_map_flat_map : equivariance.

(* Three laws that the tactic would otherwise derive many times. *)
Lemma simple_avoids_renamed : equivariance_of simple_avoids.
Proof. equivariance. Qed.
#[local] Hint Rewrite simple_avoids_renamed : equivariance.

Lemma simples_avoid_renamed : equivariance_of simples_avoid.
Proof. equivariance. Qed.
#[local] Hint Rewrite simples_avoid_renamed : equivariance.

Lemma value_slots_avoid_renamed : equivariance_of value_slots_avoid.
Proof. equivariance. Qed.
#[local] Hint Rewrite value_slots_avoid_renamed : equivariance.

Theorem expr_avoids_equivariant : expr_avoids_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite expr_avoids_equivariant : equivariance.

Theorem binds_only_kept_equivariant : binds_only_kept_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite binds_only_kept_equivariant : equivariance.

Theorem erase_expr_equivariant : erase_expr_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite erase_expr_equivariant : equivariance.

Theorem drop_statics_equivariant : drop_statics_equivariant_statement.
Proof. equivariance. Qed.

(* The condition on the environments. A handler and code are terms; a
   continuation of the machine and the static environment are not, and
   their laws are by hand. *)
Lemma handler_free_of_renamed : equivariance_of handler_free_of.
Proof. equivariance. Qed.
#[local] Hint Rewrite handler_free_of_renamed : equivariance.

Lemma code_free_of_renamed : equivariance_of code_free_of.
Proof. equivariance. Qed.
#[local] Hint Rewrite code_free_of_renamed : equivariance.

Lemma env_free_of_renamed (D : dropped) (r : renaming) (vars : env) :
  env_free_of D (rename_env r vars) = env_free_of D vars.
Proof. equivariance. Qed.
#[local] Hint Rewrite env_free_of_renamed : equivariance.

Lemma handlers_free_of_renamed (D : dropped) (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_free_of D h) (rename_handlers r handlers)
  = forallb (fun '(_, h) => handler_free_of D h) handlers.
Proof. equivariance. Qed.

Fixpoint cont_free_of_renamed (D : dropped) (r : renaming) (c : cont_value) :
  cont_free_of D (rename_cont_value r c) = cont_free_of D c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | reflexivity | rewrite (cont_free_of_renamed D r c); reflexivity | ].
  - rewrite handler_free_of_renamed, env_free_of_renamed,
      handlers_free_of_renamed.
    do 2 f_equal.
    induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
    rewrite (cont_free_of_renamed D r c), IH. reflexivity.
  - rewrite (cont_free_of_renamed D r return_to),
      (cont_free_of_renamed D r exn_to).
    reflexivity.
Qed.
#[local] Hint Rewrite cont_free_of_renamed : equivariance.

Lemma statics_free_of_renamed (D : dropped) (r : renaming)
    (statics : static_env) :
  statics_free_of D (rename_static_env r statics) = statics_free_of D statics.
Proof.
  unfold statics_free_of, rename_static_env. simpl. f_equal.
  unfold map_values. apply forallb_map_with. intros [id c].
  apply code_free_of_renamed.
Qed.
#[local] Hint Rewrite statics_free_of_renamed : equivariance.

Theorem envs_free_of_equivariant : envs_free_of_equivariant_statement.
Proof. equivariance. Qed.

(* FlambdaUnusedHeadStaticsSpec.v: the checks of the removal from the
   head of a unit read the same of a term renamed. *)
Lemma head_bindings_renamed (r : renaming) (e : expr) :
  head_bindings (rename_expr r e)
  = map (rename_static_binding r) (head_bindings e).
Proof. equivariance. Qed.
#[local] Hint Rewrite head_bindings_renamed : equivariance.

Lemma under_head_renamed : equivariance_of under_head.
Proof. equivariance. Qed.
#[local] Hint Rewrite under_head_renamed : equivariance.

Lemma symbols_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map symbols_of_binding (map (rename_static_binding r) bindings)
  = flat_map symbols_of_binding bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite symbols_of_bindings_renamed : equivariance.

Lemma code_ids_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map code_ids_of_binding (map (rename_static_binding r) bindings)
  = flat_map code_ids_of_binding bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite code_ids_of_bindings_renamed : equivariance.

Lemma plain_simple_renamed_same : equivariance_of plain_simple.
Proof. equivariance. Qed.
#[local] Hint Rewrite plain_simple_renamed_same : equivariance.

Lemma head_statics_removable_renamed (D : dropped) (r : renaming) (e : expr) :
  head_statics_removable D (rename_expr r e) = head_statics_removable D e.
Proof. equivariance. Qed.

End Dropped.

(* ======================================================================== *)
(* FlambdaBlockLoadSpec.v                                                   *)
(* ======================================================================== *)

(* The laws of renaming of this section are proved from the definitions
   by the tactic of FlambdaEquivarianceTactic.v. It is told how what is
   known is renamed, since the types do not say it (the blocks known of
   variables are a map keyed by variables, and a known block is a pair
   of a tag and a list): the actions are [rename_known_blocks] and
   [rename_known_block]. The laws that other files use by name keep
   their names and statements, and each law is given to the tactic for
   the proofs after it. A law that no other file uses is stated where
   the tactic would otherwise derive it in two proofs or more. One law
   is stated in the form that computation gives it,
   [static_blocks_made_renamed]: the blocks that the bindings of a
   [Let_static] make, under [flat_map]. *)
Section Block_load.

(* The database of laws is tried in one pass (rewrite_strat) in this
   section: every proof of it passes so, and takes a third of the
   time. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance known_blocks_renaming : Renaming_of known_blocks :=
  rename_known_blocks.
#[local] Instance known_block_renaming : Renaming_of known_block :=
  rename_known_block.
#[local] Hint Rewrite var_in_simple_equivariant
  FlambdaReadsAfterProofs.surely_value_renamed : equivariance.

Lemma unary_applied_equivariant (r : renaming) (p : prim)
    (args : list simple) :
  unary_applied p (map (rename_simple r) args)
  = option_map (fun applied => (fst applied, rename_simple r (snd applied)))
      (unary_applied p args).
Proof. equivariance. Qed.
#[local] Hint Rewrite unary_applied_equivariant : equivariance.

Lemma forget_in_block_equivariant (r : renaming) (x : variable)
    (block : known_block) :
  forget_in_block (rename_variable r x) (rename_known_block r block)
  = rename_known_block r (forget_in_block x block).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_in_block_equivariant : equivariance.

Lemma forget_var_equivariant (r : renaming) (x : variable)
    (known : known_blocks) :
  forget_var (rename_variable r x) (rename_known_blocks r known)
  = rename_known_blocks r (forget_var x known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_var_equivariant : equivariance.

Lemma forget_vars_equivariant (r : renaming) (xs : list variable) :
  ∀ known : known_blocks,
    forget_vars (map (rename_variable r) xs) (rename_known_blocks r known)
    = rename_known_blocks r (forget_vars xs known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_vars_equivariant : equivariance.

Lemma block_made_equivariant (r : renaming) (d : named) :
  block_made (rename_named r d)
  = option_map (rename_known_block r) (block_made d).
Proof. equivariance. Qed.
#[local] Hint Rewrite block_made_equivariant : equivariance.

Lemma known_under_let_equivariant (r : renaming) (known : known_blocks)
    (x : variable) (d : named) :
  known_under_let (rename_known_blocks r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_blocks r (known_under_let known x d).
Proof. equivariance. Qed.
#[local] Hint Rewrite known_under_let_equivariant : equivariance.

Lemma static_blocks_made_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map (fun binding => static_block_made (rename_static_binding r binding))
    bindings
  = flat_map
      (fun binding =>
         map (fun made : positive * known_block =>
                (fst made, rename_known_block r (snd made)))
           (static_block_made binding))
      bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite static_blocks_made_renamed : equivariance.

Lemma known_under_statics_equivariant (r : renaming) (known : known_blocks)
    (bindings : list static_binding) :
  known_under_statics (rename_known_blocks r known)
    (map (rename_static_binding r) bindings)
  = rename_known_blocks r (known_under_statics known bindings).
Proof. equivariance. Qed.
#[local] Hint Rewrite known_under_statics_equivariant : equivariance.

Lemma known_block_of_equivariant (r : renaming) (known : known_blocks)
    (s : simple) :
  known_block_of (rename_known_blocks r known) (rename_simple r s)
  = option_map (rename_known_block r) (known_block_of known s).
Proof. equivariance. Qed.
#[local] Hint Rewrite known_block_of_equivariant : equivariance.

Lemma field_loaded_equivariant (r : renaming) (kind : block_access_kind)
    (field : Z) (block : known_block) :
  field_loaded kind field (rename_known_block r block)
  = option_map (rename_simple r) (field_loaded kind field block).
Proof. equivariance. Qed.
#[local] Hint Rewrite field_loaded_equivariant : equivariance.

Lemma of_known_block_equivariant (r : renaming) (p : unary_primitive)
    (block : known_block) :
  of_known_block p (rename_known_block r block)
  = option_map (rename_simple r) (of_known_block p block).
Proof. equivariance. Qed.
#[local] Hint Rewrite of_known_block_equivariant : equivariance.

Lemma load_known_named_equivariant (r : renaming) (known : known_blocks)
    (d : named) :
  load_known_named (rename_known_blocks r known) (rename_named r d)
  = rename_named r (load_known_named known d).
Proof. equivariance. Qed.
#[local] Hint Rewrite load_known_named_equivariant : equivariance.

Theorem load_blocks_knowing_equivariant :
  load_blocks_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem load_known_blocks_equivariant :
  load_known_blocks_equivariant_statement.
Proof.
  intros r e. exact (load_blocks_knowing_equivariant r no_blocks e).
Qed.

(* The read of a known block that the term after is given. *)
Lemma load_known_named_surely_equivariant (r : renaming) (known : known_blocks)
    (d : named) :
  load_known_named_surely (rename_known_blocks r known) (rename_named r d)
  = rename_named r (load_known_named_surely known d).
Proof. equivariance. Qed.
#[local] Hint Rewrite load_known_named_surely_equivariant : equivariance.

Theorem load_blocks_knowing_surely_equivariant :
  load_blocks_knowing_surely_equivariant_statement.
Proof. equivariance. Qed.

Theorem load_known_blocks_surely_equivariant :
  load_known_blocks_surely_equivariant_statement.
Proof.
  intros r e. exact (load_blocks_knowing_surely_equivariant r no_blocks e).
Qed.

End Block_load.

(* ======================================================================== *)
(* FlambdaUnboxSpec.v                                                       *)
(* ======================================================================== *)

(* The laws of renaming of this section are proved from the definitions
   by the tactic of FlambdaEquivarianceTactic.v. It is told two things.
   The type of what is known does not say how it is renamed (its first
   field is a map keyed by variables): the action is
   [rename_known_numbers]. And the law of [lookup] in that map, at a
   renamed variable, is stated here; the tactic proves it. The laws
   that other files use by name keep their names and statements, and
   each is given to the tactic for the proofs after it, with two laws
   stated before this section and the law of [forget_wrapped_var],
   which the tactic would otherwise derive in two proofs. *)
Section Unbox_numbers.

(* The database of laws is tried in one pass (rewrite_strat) in this
   section: every proof of it passes so, and takes a third of the
   time. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance known_numbers_renaming : Renaming_of known_numbers :=
  rename_known_numbers.
#[local] Hint Rewrite var_in_simple_equivariant
  unary_applied_equivariant : equivariance.

Lemma forget_wrapped_var_equivariant (r : renaming) (x : variable)
    (known : known_numbers) :
  forget_wrapped_var (rename_variable r x) (rename_known_numbers r known)
  = rename_known_numbers r (forget_wrapped_var x known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_wrapped_var_equivariant : equivariance.

Lemma lookup_wrapped_var_renamed (r : renaming) (x : variable)
    (wrapped_vars : list (variable * (wrapping * simple))) :
  lookup (rename_variable r x)
    (map
       (fun entry =>
          (rename_variable r (fst entry),
           (fst (snd entry), rename_simple r (snd (snd entry)))))
       wrapped_vars)
  = option_map (fun wrapped => (fst wrapped, rename_simple r (snd wrapped)))
      (lookup x wrapped_vars).
Proof. equivariance. Qed.
#[local] Hint Rewrite lookup_wrapped_var_renamed : equivariance.

Lemma forget_wrapped_vars_equivariant (r : renaming) (xs : list variable) :
  ∀ known : known_numbers,
    forget_wrapped_vars (map (rename_variable r) xs)
      (rename_known_numbers r known)
    = rename_known_numbers r (forget_wrapped_vars xs known).
Proof. equivariance. Qed.
#[local] Hint Rewrite forget_wrapped_vars_equivariant : equivariance.

Lemma numbers_under_let_equivariant (r : renaming) (known : known_numbers)
    (x : variable) (d : named) :
  numbers_under_let (rename_known_numbers r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_numbers r (numbers_under_let known x d).
Proof. equivariance. Qed.
#[local] Hint Rewrite numbers_under_let_equivariant : equivariance.

Lemma numbers_under_statics_equivariant (r : renaming)
    (known : known_numbers) (bindings : list static_binding) :
  numbers_under_statics (rename_known_numbers r known)
    (map (rename_static_binding r) bindings)
  = rename_known_numbers r (numbers_under_statics known bindings).
Proof. equivariance. Qed.
#[local] Hint Rewrite numbers_under_statics_equivariant : equivariance.

Lemma unwrapped_equivariant (r : renaming) (known : known_numbers)
    (how : wrapping) (wrapped : simple) :
  unwrapped (rename_known_numbers r known) how (rename_simple r wrapped)
  = option_map (rename_simple r) (unwrapped known how wrapped).
Proof. equivariance. Qed.
#[local] Hint Rewrite unwrapped_equivariant : equivariance.

Lemma unbox_known_named_equivariant (r : renaming) (known : known_numbers)
    (d : named) :
  unbox_known_named (rename_known_numbers r known) (rename_named r d)
  = rename_named r (unbox_known_named known d).
Proof. equivariance. Qed.
#[local] Hint Rewrite unbox_known_named_equivariant : equivariance.

Theorem unbox_numbers_knowing_equivariant :
  unbox_numbers_knowing_equivariant_statement.
Proof. equivariance. Qed.

Theorem unbox_known_numbers_equivariant :
  unbox_known_numbers_equivariant_statement.
Proof.
  intros r e. exact (unbox_numbers_knowing_equivariant r no_numbers e).
Qed.

End Unbox_numbers.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

(* The laws of renaming of this section are proved from the definitions
   by the tactic of FlambdaEquivarianceTactic.v. It is told how what is
   known is renamed, since the types do not say it (a typing environment
   is a list of pairs of a positive and a type): the actions are
   [rename_naive_typing_env] and [rename_naive_type]. The laws of the
   functions that [follow_aliases] is made of are stated, and each is
   given to the tactic for the proofs after it: a law that is not
   stated is derived again by every proof that needs it. The two laws
   of lists are the forms that the terms have where [canonical_simple]
   is under a [map]. *)
Section Follow_aliases.

(* The database of laws is tried in one pass (rewrite_strat) in this
   section: every proof of it passes so, and takes a third of the
   time. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance naive_typing_env_renaming : Renaming_of typing_env :=
  rename_naive_typing_env.
#[local] Instance naive_type_renaming : Renaming_of flambda_type :=
  rename_naive_type.
#[local] Hint Rewrite var_in_simple_equivariant
  var_in_value_slots_equivariant : equivariance.

Lemma canonical_simple_equivariant (r : renaming) (tenv : typing_env)
    (s : simple) :
  canonical_simple (rename_naive_typing_env r tenv) (rename_simple r s)
  = rename_simple r (canonical_simple tenv s).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_simple_equivariant : equivariance.

Lemma canonical_simples_equivariant (r : renaming) (tenv : typing_env)
    (args : list simple) :
  map (canonical_simple (rename_naive_typing_env r tenv))
    (map (rename_simple r) args)
  = map (rename_simple r) (map (canonical_simple tenv) args).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_simples_equivariant : equivariance.

Lemma canonical_exn_extra_args_equivariant (r : renaming) (tenv : typing_env)
    (extra : list (simple * flambda_kind)) :
  map
    (fun arg =>
       (canonical_simple (rename_naive_typing_env r tenv) (fst arg), snd arg))
    (map (fun arg => (rename_simple r (fst arg), snd arg)) extra)
  = map (fun arg => (rename_simple r (fst arg), snd arg))
      (map (fun arg => (canonical_simple tenv (fst arg), snd arg)) extra).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_exn_extra_args_equivariant : equivariance.

Lemma canonical_value_slots_equivariant (r : renaming) (tenv : typing_env)
    (slots : list (value_slot * simple)) :
  canonical_value_slots (rename_naive_typing_env r tenv)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = map (fun slot => (fst slot, rename_simple r (snd slot)))
      (canonical_value_slots tenv slots).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_value_slots_equivariant : equivariance.

Lemma closure_of_simple_equivariant (r : renaming) (tenv : typing_env)
    (s : simple) :
  closure_of_simple (rename_naive_typing_env r tenv) (rename_simple r s)
  = option_map
      (fun closure : code_id * list (value_slot * simple) =>
         (fst closure,
          map (fun slot => (fst slot, rename_simple r (snd slot)))
            (snd closure)))
      (closure_of_simple tenv s).
Proof. equivariance. Qed.
#[local] Hint Rewrite closure_of_simple_equivariant : equivariance.

Lemma canonical_call_kind_equivariant (r : renaming) (tenv : typing_env)
    (callee : simple) (kind : call_kind) :
  canonical_call_kind (rename_naive_typing_env r tenv)
    (rename_simple r callee) kind
  = canonical_call_kind tenv callee kind.
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_call_kind_equivariant : equivariance.

Lemma canonical_apply_equivariant (r : renaming) (tenv : typing_env)
    (ap : apply_expr) :
  canonical_apply (rename_naive_typing_env r tenv) (rename_apply r ap)
  = rename_apply r (canonical_apply tenv ap).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_apply_equivariant : equivariance.

Lemma canonical_apply_cont_equivariant (r : renaming) (tenv : typing_env)
    (ac : apply_cont) :
  canonical_apply_cont (rename_naive_typing_env r tenv)
    (rename_apply_cont r ac)
  = rename_apply_cont r (canonical_apply_cont tenv ac).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_apply_cont_equivariant : equivariance.

Lemma canonical_named_equivariant (r : renaming) (tenv : typing_env)
    (d : named) :
  canonical_named (rename_naive_typing_env r tenv) (rename_named r d)
  = rename_named r (canonical_named tenv d).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_named_equivariant : equivariance.

Lemma canonical_static_binding_equivariant (r : renaming)
    (tenv : typing_env) (binding : static_binding) :
  canonical_static_binding (rename_naive_typing_env r tenv)
    (rename_static_binding r binding)
  = rename_static_binding r (canonical_static_binding tenv binding).
Proof. equivariance. Qed.
#[local] Hint Rewrite canonical_static_binding_equivariant : equivariance.

Lemma drop_var_equivariant (r : renaming) (x : variable) (tenv : typing_env) :
  drop_var (rename_variable r x) (rename_naive_typing_env r tenv)
  = rename_naive_typing_env r (drop_var x tenv).
Proof. equivariance. Qed.
#[local] Hint Rewrite drop_var_equivariant : equivariance.

Lemma drop_vars_equivariant (r : renaming) (xs : list variable)
    (tenv : typing_env) :
  drop_vars (map (rename_variable r) xs) (rename_naive_typing_env r tenv)
  = rename_naive_typing_env r (drop_vars xs tenv).
Proof. equivariance. Qed.
#[local] Hint Rewrite drop_vars_equivariant : equivariance.

Lemma typing_env_under_let_equivariant (r : renaming) (tenv : typing_env)
    (x : variable) (d : named) :
  typing_env_under_let (rename_naive_typing_env r tenv) (rename_variable r x)
    (rename_named r d)
  = rename_naive_typing_env r (typing_env_under_let tenv x d).
Proof. equivariance. Qed.
#[local] Hint Rewrite typing_env_under_let_equivariant : equivariance.

Lemma follow_aliases_equivariant_aux (r : renaming) (e : expr) :
  ∀ tenv : typing_env,
    follow_aliases (rename_naive_typing_env r tenv) (rename_expr r e)
    = rename_expr r (follow_aliases tenv e).
Proof. equivariance. Qed.

Theorem follow_aliases_equivariant : follow_aliases_equivariant_statement.
Proof. intros r tenv e. apply follow_aliases_equivariant_aux. Qed.

Theorem follow_aliases_unknowing_equivariant :
  follow_aliases_unknowing_equivariant_statement.
Proof. intros r e. exact (follow_aliases_equivariant_aux r e []). Qed.

End Follow_aliases.

(* ======================================================================== *)
(* The checks of FlambdaWellFormedSpec.v                                    *)
(* ======================================================================== *)

(* The laws of the checks, up to the exception handlers, are proved by
   the tactic of FlambdaEquivarianceTactic.v, with the hints of this
   section, which do not leave it. By hand there: the free continuations
   (what a binder takes away, at renamed names, is by the injectivity of
   a renaming) and the test that a static set is bound whole (the
   equality of two static constants, by injectivity). The laws of the
   trap actions are by hand. The section stays open over them for the
   law of a whole unit, which is by the tactic and is the last of the
   section. *)
Section Well_formed_checks.

Lemma forallb_same {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = g a) -> forallb f l = forallb g l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

Fixpoint well_formed_free_continuations_renamed (r : renaming) (e : expr) :
  free_continuations (rename_expr r e)
  = map (rename_continuation r) (free_continuations e)
with handler_free_continuations_renamed (r : renaming)
    (h : continuation_handler) :
  handler_free_continuations (rename_handler r h)
  = map (rename_continuation r) (handler_free_continuations h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr free_continuations].
    + apply well_formed_free_continuations_renamed.
    + apply well_formed_free_continuations_renamed.
    + rewrite map_app, handler_free_continuations_renamed,
        well_formed_free_continuations_renamed.
      rewrite <- (without_map_injective (rename_continuation r)
                    (rename_continuation_eqb r) [k]).
      reflexivity.
    + fold (rename_handlers r handlers).
      rewrite map_fst_rename_handlers.
      rewrite <- (without_map_injective (rename_continuation r)
                    (rename_continuation_eqb r)).
      rewrite map_app, well_formed_free_continuations_renamed.
      do 2 f_equal. unfold rename_handlers.
      induction handlers as [ | [k h] handlers IH]; [ reflexivity | ].
      cbn [map flat_map].
      rewrite map_app, (handler_free_continuations_renamed r h), IH.
      reflexivity.
    + reflexivity.
    + apply apply_cont_continuations_renamed.
    + induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
      cbn [map flat_map fst snd].
      rewrite map_app, apply_cont_continuations_renamed, IH. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_free_continuations].
    apply well_formed_free_continuations_renamed.
Qed.

Theorem well_formed_free_continuations_equivariant :
  well_formed_free_continuations_equivariant_statement.
Proof. exact well_formed_free_continuations_renamed. Qed.

#[local] Hint Rewrite well_formed_free_continuations_renamed
  handler_free_continuations_renamed : equivariance.

Theorem handler_at_toplevel_equivariant :
  handler_at_toplevel_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite handler_at_toplevel_equivariant : equivariance.

Theorem statics_at_toplevel_equivariant :
  statics_at_toplevel_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite statics_at_toplevel_equivariant : equivariance.

(* From here the tactic rewrites with its database in one pass, which
   is faster. A proof that the one pass leaves open is between a line
   that gives the step back to autorewrite and a line that takes the
   one pass again. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Section Static_names.

Variable r : renaming.
Variable names_of : static_binding -> list positive.
Hypothesis names_of_renamed :
  ∀ binding : static_binding,
    names_of (rename_static_binding r binding) = names_of binding.

#[local] Hint Rewrite names_of_renamed : equivariance.

Lemma flat_map_names_renamed (bindings : list static_binding) :
  flat_map names_of (map (rename_static_binding r) bindings)
  = flat_map names_of bindings.
Proof. equivariance. Qed.
#[local] Hint Rewrite flat_map_names_renamed : equivariance.

Lemma static_names_renamed (e : expr) :
  static_names names_of (rename_expr r e) = static_names names_of e.
Proof. equivariance. Qed.

(* A group binds a name twice, or one that is bound, as the renamed
   group does. *)
Lemma rebinds_renamed (is_bound is_bound' : positive -> bool)
    (bindings : list static_binding) :
  (∀ name : positive, is_bound' name = is_bound name) ->
  rebinds names_of is_bound' (map (rename_static_binding r) bindings)
  = rebinds names_of is_bound bindings.
Proof. equivariance. Qed.

End Static_names.

Lemma code_ids_of_binding_renamed (r : renaming) (binding : static_binding) :
  code_ids_of_binding (rename_static_binding r binding)
  = code_ids_of_binding binding.
Proof. equivariance. Qed.

Lemma symbols_of_binding_renamed (r : renaming) (binding : static_binding) :
  symbols_of_binding (rename_static_binding r binding)
  = symbols_of_binding binding.
Proof. equivariance. Qed.
#[local] Hint Rewrite code_ids_of_binding_renamed symbols_of_binding_renamed
  : equivariance.

Theorem static_names_equivariant : static_names_equivariant_statement.
Proof. intros r names_of e Hnames. apply static_names_renamed, Hnames. Qed.

#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
Theorem bound_static_names_equivariant :
  bound_static_names_equivariant_statement.
Proof. equivariance. Qed.
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

(* Sets of closures. A static set is bound whole when the group has, for
   each symbol of the set, a binding of that symbol to the very set: the
   test of equality of static constants answers the same on renamed
   ones. *)
Lemma rename_value_slots_injective (r : renaming)
    (slots slots' : list (value_slot * simple)) :
  map (fun slot => (fst slot, rename_simple r (snd slot))) slots
  = map (fun slot => (fst slot, rename_simple r (snd slot))) slots' ->
  slots = slots'.
Proof.
  revert slots'.
  induction slots as [ | [slot s] slots IH]; intros [ | [slot' s'] slots'];
    simpl; intros Heq; try discriminate; [ reflexivity | ].
  injection Heq as Hslot Hs Hslots.
  apply rename_simple_injective in Hs. apply IH in Hslots. congruence.
Qed.

Lemma rename_static_const_injective (r : renaming)
    (const const' : static_const) :
  rename_static_const r const = rename_static_const r const' ->
  const = const'.
Proof.
  destruct const, const'; simpl; intros Heq; try discriminate;
    injection Heq; intros; subst; try reflexivity; f_equal;
    eauto using rename_value_slots_injective, rename_simples_injective,
      rename_simple_injective.
Qed.

Lemma binds_symbol_to_renamed (r : renaming) (s : symbol)
    (const : static_const) (binding : static_binding) :
  binds_symbol_to s (rename_static_const r const)
    (rename_static_binding r binding)
  = binds_symbol_to s const binding.
Proof.
  destruct binding as [id c | id | s' const']; try reflexivity.
  cbn [rename_static_binding binds_symbol_to]. f_equal.
  apply (dec_injective static_const_eq_dec (rename_static_const r)).
  apply rename_static_const_injective.
Qed.

Lemma static_set_whole_renamed (r : renaming)
    (bindings : list static_binding) (binding : static_binding) :
  static_set_whole (map (rename_static_binding r) bindings)
    (rename_static_binding r binding)
  = static_set_whole bindings binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  destruct const
    as [ | slot closure_symbols function_decls value_slots | | | | ];
    try reflexivity.
  cbn [rename_static_binding rename_static_const static_set_whole]. f_equal.
  apply forallb_same. intros [slot' s'].
  apply existsb_map_with. intros binding'.
  exact
    (binds_symbol_to_renamed r s'
       (Static_const_closure_of_set slot' closure_symbols function_decls
          value_slots)
       binding').
Qed.

(* The law of [static_set_whole] over the bindings of a group, in the
   form that the group has it in once the list is entered. *)
Lemma static_sets_whole_renamed (r : renaming)
    (bindings : list static_binding) :
  forallb
    (fun a =>
       static_set_whole (map (rename_static_binding r) bindings)
         (rename_static_binding r a))
    bindings
  = forallb (static_set_whole bindings) bindings.
Proof. apply forallb_same, static_set_whole_renamed. Qed.
#[local] Hint Rewrite static_sets_whole_renamed : equivariance.

Theorem sets_well_formed_equivariant : sets_well_formed_equivariant_statement.
Proof. equivariance. Qed.

(* Numbers: a renaming changes no constant, no tag and no discriminant. *)
Lemma simple_in_range_renamed (r : renaming) (s : simple) :
  simple_in_range (rename_simple r s) = simple_in_range s.
Proof. equivariance. Qed.
#[local] Hint Rewrite simple_in_range_renamed : equivariance.

Lemma value_slots_in_range_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  value_slots_in_range
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_in_range slots.
Proof. equivariance. Qed.
#[local] Hint Rewrite value_slots_in_range_renamed : equivariance.

Lemma arm_in_range_renamed (r : renaming) (i : Z) (ac : apply_cont) :
  arm_in_range (i, rename_apply_cont r ac) = arm_in_range (i, ac).
Proof. equivariance. Qed.
#[local] Hint Rewrite arm_in_range_renamed : equivariance.
#[local] Arguments arm_in_range : simpl never.

#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
Theorem numbers_in_range_equivariant : numbers_in_range_equivariant_statement.
Proof. equivariance. Qed.
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

(* Exception handlers. *)

Lemma traps_renamed (r : renaming) (k : continuation) (ac : apply_cont) :
  traps (rename_continuation r k) (rename_apply_cont r ac) = traps k ac.
Proof. equivariance. Qed.
#[local] Hint Rewrite traps_renamed : equivariance.

Lemma is_raise_renamed (r : renaming) (ac : apply_cont) :
  is_raise (rename_apply_cont r ac) = is_raise ac.
Proof. equivariance. Qed.
#[local] Hint Rewrite is_raise_renamed : equivariance.

Lemma applies_renamed (r : renaming) (k : continuation) (ac : apply_cont) :
  applies (rename_continuation r k) (rename_apply_cont r ac) = applies k ac.
Proof. equivariance. Qed.
#[local] Hint Rewrite applies_renamed : equivariance.

Lemma traps_apart_renamed (r : renaming) (k : continuation)
    (ac : apply_cont) :
  traps_apart (rename_continuation r k) (rename_apply_cont r ac)
  = traps_apart k ac.
Proof. equivariance. Qed.
#[local] Hint Rewrite traps_apart_renamed : equivariance.

Theorem raises_to_equivariant : raises_to_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite raises_to_equivariant : equivariance.

Theorem applies_cont_equivariant : applies_cont_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite applies_cont_equivariant : equivariance.

Theorem gives_extra_args_equivariant :
  gives_extra_args_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite gives_extra_args_equivariant : equivariance.

Theorem installs_or_removes_equivariant :
  installs_or_removes_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite installs_or_removes_equivariant : equivariance.

(* By autorewrite from here to the end of the section. *)
#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
Theorem ways_out_well_formed_equivariant :
  ways_out_well_formed_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite ways_out_well_formed_equivariant : equivariance.

Theorem exn_handlers_well_formed_equivariant :
  exn_handlers_well_formed_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite exn_handlers_well_formed_equivariant : equivariance.

Theorem unit_exn_handlers_well_formed_equivariant :
  unit_exn_handlers_well_formed_equivariant_statement.
Proof. equivariance. Qed.

(* Trap actions. *)

Lemma among_trap_stack_renamed (r : renaming) (k : continuation)
    (ts : trap_stack) :
  among (rename_continuation r k) (rename_trap_stack r ts) = among k ts.
Proof. apply among_rename_continuation. Qed.

Lemma trap_stack_eqb_renamed (r : renaming) (ts : trap_stack) :
  ∀ ts' : trap_stack,
    trap_stack_eqb (rename_trap_stack r ts) (rename_trap_stack r ts')
    = trap_stack_eqb ts ts'.
Proof.
  unfold rename_trap_stack.
  induction ts as [ | k ts IH]; intros [ | k' ts']; simpl; try reflexivity.
  rewrite rename_continuation_eqb, IH. reflexivity.
Qed.

Definition rename_entry (r : renaming) (entered : entry) : entry :=
  match entered with
  | Not_entered => Not_entered
  | Entered_with ts => Entered_with (rename_trap_stack r ts)
  | Entered_with_two_stacks => Entered_with_two_stacks
  end.

Lemma map_snd_rename_demands (r : renaming) (ds : demands) :
  map snd (rename_demands r ds) = map (rename_trap_stack r) (map snd ds).
Proof. unfold rename_demands. rewrite !map_map. reflexivity. Qed.

Lemma rename_demands_app (r : renaming) (ds ds' : demands) :
  rename_demands r (ds ++ ds') = rename_demands r ds ++ rename_demands r ds'.
Proof. apply map_app. Qed.

Lemma filter_demands_renamed (r : renaming) (k : continuation)
    (ds : demands) :
  filter (fun d => Pos.eqb (fst d) (rename_continuation r k))
    (rename_demands r ds)
  = rename_demands r (filter (fun d => Pos.eqb (fst d) k) ds).
Proof.
  unfold rename_demands. apply filter_map_with. intros d. cbn [fst].
  apply rename_continuation_eqb.
Qed.

Definition entry_of_stacks (stacks : list trap_stack) : entry :=
  match stacks with
  | [] => Not_entered
  | ts :: others =>
    if forallb (trap_stack_eqb ts) others then Entered_with ts
    else Entered_with_two_stacks
  end.

Lemma entry_of_stacks_renamed (r : renaming) (stacks : list trap_stack) :
  entry_of_stacks (map (rename_trap_stack r) stacks)
  = rename_entry r (entry_of_stacks stacks).
Proof.
  destruct stacks as [ | ts others]; [ reflexivity | ].
  cbn [map entry_of_stacks].
  rewrite (forallb_map_with _ _ (trap_stack_eqb ts))
    by (intros ts'; apply trap_stack_eqb_renamed).
  destruct (forallb (trap_stack_eqb ts) others); reflexivity.
Qed.

Lemma entry_of_renamed (r : renaming) (k : continuation) (ds : demands) :
  entry_of (rename_continuation r k) (rename_demands r ds)
  = rename_entry r (entry_of k ds).
Proof.
  change
    (entry_of_stacks
       (map snd
          (filter (fun d => Pos.eqb (fst d) (rename_continuation r k))
             (rename_demands r ds)))
     = rename_entry r
         (entry_of_stacks
            (map snd (filter (fun d => Pos.eqb (fst d) k) ds)))).
  rewrite filter_demands_renamed, map_snd_rename_demands.
  apply entry_of_stacks_renamed.
Qed.

Definition leave_scope_with (keep clear : continuation * trap_stack -> bool)
    (ds : demands) : option demands :=
  if forallb clear (filter keep ds) then Some (filter keep ds) else None.

Lemma leave_scope_with_renamed
    (g : continuation * trap_stack -> continuation * trap_stack)
    (keep keep' clear clear' : continuation * trap_stack -> bool)
    (ds : demands) :
  (∀ d : continuation * trap_stack, keep' (g d) = keep d) ->
  (∀ d : continuation * trap_stack, clear' (g d) = clear d) ->
  leave_scope_with keep' clear' (map g ds)
  = option_map (map g) (leave_scope_with keep clear ds).
Proof.
  intros Hkeep Hclear. unfold leave_scope_with.
  rewrite (filter_map_with keep' g keep ds Hkeep),
    (forallb_map_with clear' g clear _ Hclear).
  destruct (forallb clear (filter keep ds)); reflexivity.
Qed.

Lemma leave_scope_renamed (r : renaming) (ks : list continuation)
    (ds : demands) :
  leave_scope (map (rename_continuation r) ks) (rename_demands r ds)
  = option_map (rename_demands r) (leave_scope ks ds).
Proof.
  apply
    (leave_scope_with_renamed
       (fun d => (rename_continuation r (fst d), rename_trap_stack r (snd d)))
       (fun d => negb (among (fst d) ks))
       (fun d => negb (among (fst d) (map (rename_continuation r) ks)))
       (fun d => forallb (fun k => negb (among k (snd d))) ks)
       (fun d =>
          forallb (fun k => negb (among k (snd d)))
            (map (rename_continuation r) ks))
       ds).
  - intros d. cbn [fst]. rewrite among_rename_continuation. reflexivity.
  - intros d. cbn [snd]. apply forallb_map_with. intros k.
    rewrite among_trap_stack_renamed. reflexivity.
Qed.

Lemma apply_cont_demands_renamed (r : renaming) (ts : trap_stack)
    (ac : apply_cont) :
  apply_cont_demands (rename_trap_stack r ts) (rename_apply_cont r ac)
  = option_map (rename_demands r) (apply_cont_demands ts ac).
Proof.
  unfold apply_cont_demands. simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl; try reflexivity.
  destruct ts as [ | top rest]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb. destruct (Pos.eqb top k); reflexivity.
Qed.

Lemma apply_demands_renamed (r : renaming) (ts : trap_stack)
    (ap : apply_expr) :
  apply_demands (rename_trap_stack r ts) (rename_apply r ap)
  = option_map (rename_demands r) (apply_demands ts ap).
Proof.
  unfold apply_demands. destruct ts as [ | top rest]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb top (ap_exn_continuation ap)); reflexivity.
Qed.

Lemma arms_demands_renamed (r : renaming) (ts : trap_stack)
    (arms : list (Z * apply_cont)) :
  arms_demands (rename_trap_stack r ts)
    (map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms)
  = option_map (rename_demands r) (arms_demands ts arms).
Proof.
  induction arms as [ | [i ac] arms IH]; [ reflexivity | ].
  cbn [map arms_demands fst snd]. rewrite apply_cont_demands_renamed, IH.
  destruct (apply_cont_demands ts ac) as [ds | ]; [ | reflexivity ].
  destruct (arms_demands ts arms) as [ds' | ]; [ | reflexivity ].
  cbn [option_map]. rewrite rename_demands_app. reflexivity.
Qed.

Lemma code_demands_met_renamed (r : renaming) (ret exn : continuation)
    (ds : demands) :
  code_demands_met (rename_continuation r ret) (rename_continuation r exn)
    (rename_demands r ds)
  = code_demands_met ret exn ds.
Proof.
  unfold code_demands_met, rename_demands. apply forallb_map_with. intros d.
  cbn [fst snd]. rewrite !rename_continuation_eqb.
  match goal with
  | |- (if ?c then _ else _) = (if ?c' then _ else _) =>
    change c' with c; destruct c
  end.
  - exact (trap_stack_eqb_renamed r (snd d) []).
  - apply (f_equal2 andb); [ reflexivity | ].
    exact (trap_stack_eqb_renamed r (snd d) [exn]).
Qed.

Lemma fold_left_renamed {S B B' : Type} (g : S -> S) (gb : B -> B')
    (step : S -> B -> S) (step' : S -> B' -> S) (l : list B) :
  Forall (fun b => ∀ s : S, step' (g s) (gb b) = g (step s b)) l ->
  ∀ s : S,
    List.fold_left step' (map gb l) (g s) = g (List.fold_left step l s).
Proof.
  induction 1 as [ | b l Hb Hl IH]; intros s; simpl; [ reflexivity | ].
  rewrite Hb. apply IH.
Qed.

(* A recursive group: the passes over its handlers, with the function
   that gives the demands of a handler as an argument. *)
Definition group_pass
    (demands_of : trap_stack -> continuation_handler -> option demands)
    (handlers : list (continuation * continuation_handler))
    (state : group_state) : option group_state :=
  fold_left
    (fun state '(k, h) =>
       let* '(checked, ds) := state in
       if among k checked then Some (checked, ds)
       else
         match entry_of k ds with
         | Not_entered => Some (checked, ds)
         | Entered_with entered =>
           let* ds' := demands_of entered h in
           Some (k :: checked, ds ++ ds')
         | Entered_with_two_stacks => None
         end)
    (Some state) handlers.

Definition group_demands
    (demands_of : trap_stack -> continuation_handler -> option demands)
    (handlers : list (continuation * continuation_handler)) (ds : demands)
    : option demands :=
  let* '(checked, ds) :=
    FlambdaWellFormedSpec.iterate (length handlers)
      (group_pass demands_of handlers) ([], ds)
  in
  if forallb
       (fun k =>
          match entry_of k ds with
          | Not_entered => true
          | Entered_with _ => among k checked
          | Entered_with_two_stacks => false
          end)
       (map fst handlers)
  then leave_scope (map fst handlers) ds
  else None.

Lemma trap_demands_recursive (ts : trap_stack)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  trap_demands ts (Let_cont_recursive handlers body)
  = if existsb (fun k => among k ts) (map fst handlers) then None
    else
      let* ds := trap_demands ts body in
      group_demands handler_trap_demands handlers ds.
Proof. reflexivity. Qed.

Definition rename_group_state (r : renaming) (state : group_state)
    : group_state :=
  (map (rename_continuation r) (fst state), rename_demands r (snd state)).

Section Group.

Variable r : renaming.
Variables demands_of demands_of'
  : trap_stack -> continuation_handler -> option demands.

Definition demands_of_renamed (h : continuation_handler) : Prop :=
  ∀ ts : trap_stack,
    demands_of' (rename_trap_stack r ts) (rename_handler r h)
    = option_map (rename_demands r) (demands_of ts h).

Lemma group_step_renamed (k : continuation) (h : continuation_handler)
    (state : option group_state) :
  demands_of_renamed h ->
  (let* '(checked, ds) := option_map (rename_group_state r) state in
   if among (rename_continuation r k) checked then Some (checked, ds)
   else
     match entry_of (rename_continuation r k) ds with
     | Not_entered => Some (checked, ds)
     | Entered_with entered =>
       let* ds' := demands_of' entered (rename_handler r h) in
       Some (rename_continuation r k :: checked, ds ++ ds')
     | Entered_with_two_stacks => None
     end)
  = option_map (rename_group_state r)
      (let* '(checked, ds) := state in
       if among k checked then Some (checked, ds)
       else
         match entry_of k ds with
         | Not_entered => Some (checked, ds)
         | Entered_with entered =>
           let* ds' := demands_of entered h in
           Some (k :: checked, ds ++ ds')
         | Entered_with_two_stacks => None
         end).
Proof.
  intros Hh. destruct state as [[checked ds] | ]; [ | reflexivity ].
  cbn [option_map rename_group_state fst snd].
  rewrite among_rename_continuation.
  destruct (among k checked); [ reflexivity | ].
  rewrite entry_of_renamed.
  destruct (entry_of k ds) as [ | entered | ]; try reflexivity.
  cbn [rename_entry]. rewrite Hh.
  destruct (demands_of entered h) as [ds' | ]; [ | reflexivity ].
  cbn [option_map rename_group_state fst snd map].
  apply f_equal. apply (f_equal2 pair); [ reflexivity | ].
  symmetry. apply map_app.
Qed.

Lemma group_fold_renamed
    (handlers : list (continuation * continuation_handler)) :
  Forall (fun entry => demands_of_renamed (snd entry)) handlers ->
  ∀ state : option group_state,
    fold_left
      (fun state '(k, h) =>
         let* '(checked, ds) := state in
         if among k checked then Some (checked, ds)
         else
           match entry_of k ds with
           | Not_entered => Some (checked, ds)
           | Entered_with entered =>
             let* ds' := demands_of' entered h in
             Some (k :: checked, ds ++ ds')
           | Entered_with_two_stacks => None
           end)
      (option_map (rename_group_state r) state) (rename_handlers r handlers)
    = option_map (rename_group_state r)
        (fold_left
           (fun state '(k, h) =>
              let* '(checked, ds) := state in
              if among k checked then Some (checked, ds)
              else
                match entry_of k ds with
                | Not_entered => Some (checked, ds)
                | Entered_with entered =>
                  let* ds' := demands_of entered h in
                  Some (k :: checked, ds ++ ds')
                | Entered_with_two_stacks => None
                end)
           state handlers).
Proof.
  intros Hhandlers state. unfold fold_left, rename_handlers.
  apply (fold_left_renamed (option_map (rename_group_state r))).
  refine (Forall_impl _ _ Hhandlers).
  intros [k h] Hh s. exact (group_step_renamed k h s Hh).
Qed.

Lemma group_pass_renamed
    (handlers : list (continuation * continuation_handler))
    (state : group_state) :
  Forall (fun entry => demands_of_renamed (snd entry)) handlers ->
  group_pass demands_of' (rename_handlers r handlers)
    (rename_group_state r state)
  = option_map (rename_group_state r) (group_pass demands_of handlers state).
Proof.
  intros Hhandlers. exact (group_fold_renamed handlers Hhandlers (Some state)).
Qed.

Lemma iterate_renamed {A : Type} (g : A -> A) (f f' : A -> option A)
    (n : nat) :
  (∀ x : A, f' (g x) = option_map g (f x)) ->
  ∀ x : A,
    FlambdaWellFormedSpec.iterate n f' (g x)
    = option_map g (FlambdaWellFormedSpec.iterate n f x).
Proof.
  intros Hf. induction n as [ | n IH]; intros x; [ reflexivity | ].
  cbn [FlambdaWellFormedSpec.iterate]. rewrite Hf.
  destruct (f x) as [y | ]; [ | reflexivity ].
  cbn [option_map]. apply IH.
Qed.

Lemma group_demands_renamed
    (handlers : list (continuation * continuation_handler)) (ds : demands) :
  Forall (fun entry => demands_of_renamed (snd entry)) handlers ->
  group_demands demands_of' (rename_handlers r handlers)
    (rename_demands r ds)
  = option_map (rename_demands r) (group_demands demands_of handlers ds).
Proof.
  intros Hhandlers. unfold group_demands.
  replace (length (rename_handlers r handlers)) with (length handlers)
    by (unfold rename_handlers; rewrite length_map; reflexivity).
  change ([], rename_demands r ds) with (rename_group_state r ([], ds)).
  rewrite
    (iterate_renamed (rename_group_state r) (group_pass demands_of handlers)
       (group_pass demands_of' (rename_handlers r handlers)) (length handlers)
       (fun state => group_pass_renamed handlers state Hhandlers)).
  destruct
    (FlambdaWellFormedSpec.iterate (length handlers)
       (group_pass demands_of handlers) ([], ds))
    as [[checked ds'] | ];
    [ | reflexivity ].
  cbn [option_map rename_group_state fst snd].
  rewrite map_fst_rename_handlers.
  rewrite
    (forallb_map_with _ _
       (fun k =>
          match entry_of k ds' with
          | Not_entered => true
          | Entered_with _ => among k checked
          | Entered_with_two_stacks => false
          end)).
  - rewrite leave_scope_renamed.
    destruct
      (forallb
         (fun k =>
            match entry_of k ds' with
            | Not_entered => true
            | Entered_with _ => among k checked
            | Entered_with_two_stacks => false
            end)
         (map fst handlers));
      reflexivity.
  - intros k. rewrite entry_of_renamed.
    destruct (entry_of k ds') as [ | entered | ]; cbn [rename_entry];
      try reflexivity.
    apply among_rename_continuation.
Qed.

End Group.

Fixpoint trap_demands_renamed (r : renaming) (e : expr) {struct e} :
  ∀ ts : trap_stack,
    trap_demands (rename_trap_stack r ts) (rename_expr r e)
    = option_map (rename_demands r) (trap_demands ts e)
with handler_trap_demands_renamed (r : renaming) (h : continuation_handler)
    {struct h} :
  ∀ ts : trap_stack,
    handler_trap_demands (rename_trap_stack r ts) (rename_handler r h)
    = option_map (rename_demands r) (handler_trap_demands ts h)
with binding_traps_well_formed_renamed (r : renaming)
    (binding : static_binding) {struct binding} :
  binding_traps_well_formed (rename_static_binding r binding)
  = binding_traps_well_formed binding
with code_traps_well_formed_renamed (r : renaming) (c : code) {struct c} :
  code_traps_well_formed (rename_code r c) = code_traps_well_formed c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros ts.
    + cbn [rename_expr trap_demands]. apply trap_demands_renamed.
    + cbn [rename_expr trap_demands].
      replace
        (forallb binding_traps_well_formed
           (map (rename_static_binding r) bindings))
        with (forallb binding_traps_well_formed bindings).
      * destruct (forallb binding_traps_well_formed bindings);
          [ apply trap_demands_renamed | reflexivity ].
      * induction bindings as [ | binding bindings IH]; cbn [map forallb];
          [ reflexivity | ].
        rewrite (binding_traps_well_formed_renamed r binding), <- IH.
        reflexivity.
    + cbn [rename_expr trap_demands]. rewrite among_trap_stack_renamed.
      destruct (among k ts); [ reflexivity | ].
      rewrite (trap_demands_renamed r body ts).
      destruct (trap_demands ts body) as [ds | ]; [ | reflexivity ].
      cbn [option_map].
      change [rename_continuation r k]
        with (map (rename_continuation r) [k]).
      rewrite leave_scope_renamed, entry_of_renamed.
      destruct (leave_scope [k] ds) as [outer | ]; [ | reflexivity ].
      destruct (entry_of k ds) as [ | entered | ]; try reflexivity.
      cbn [option_map rename_entry].
      rewrite among_trap_stack_renamed.
      destruct (among k entered); [ reflexivity | ].
      rewrite (handler_trap_demands_renamed r h entered).
      destruct (handler_trap_demands entered h) as [ds' | ];
        [ | reflexivity ].
      cbn [option_map]. rewrite rename_demands_app.
      reflexivity.
    + assert
        (Hhandlers :
          Forall
            (fun entry =>
               demands_of_renamed r handler_trap_demands
                 handler_trap_demands (snd entry))
            handlers).
      { induction handlers as [ | [k h] handlers IH]; constructor;
          [ | exact IH ].
        intros ts'. apply handler_trap_demands_renamed. }
      cbn [rename_expr].
      change
        (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
           handlers)
        with (rename_handlers r handlers).
      rewrite !trap_demands_recursive, map_fst_rename_handlers.
      rewrite
        (existsb_map_with _ _ (fun k => among k ts))
        by (intros k; apply among_trap_stack_renamed).
      destruct (existsb (fun k => among k ts) (map fst handlers));
        [ reflexivity | ].
      rewrite (trap_demands_renamed r body ts).
      destruct (trap_demands ts body) as [ds | ]; [ | reflexivity ].
      cbn [option_map].
      exact
        (group_demands_renamed r handler_trap_demands handler_trap_demands
           handlers ds Hhandlers).
    + cbn [rename_expr trap_demands]. apply apply_demands_renamed.
    + cbn [rename_expr trap_demands]. apply apply_cont_demands_renamed.
    + cbn [rename_expr trap_demands]. apply arms_demands_renamed.
    + reflexivity.
  - destruct h as [params metadata handler]. intros ts.
    cbn [rename_handler handler_trap_demands]. apply trap_demands_renamed.
  - destruct binding as [id c | id | sym const];
      cbn [rename_static_binding binding_traps_well_formed];
      try reflexivity.
    apply code_traps_well_formed_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_traps_well_formed].
    change [rename_continuation r exn] with (rename_trap_stack r [exn]).
    rewrite (trap_demands_renamed r body [exn]).
    destruct (trap_demands [exn] body) as [ds | ]; [ | reflexivity ].
    cbn [option_map]. apply code_demands_met_renamed.
Qed.

Lemma traps_well_formed_renamed (r : renaming) (unit_exn : continuation)
    (e : expr) :
  traps_well_formed (rename_continuation r unit_exn) (rename_expr r e)
  = traps_well_formed unit_exn e.
Proof.
  unfold traps_well_formed.
  change [rename_continuation r unit_exn]
    with (rename_trap_stack r [unit_exn]).
  rewrite (trap_demands_renamed r e [unit_exn]).
  destruct (trap_demands [unit_exn] e) as [ds | ]; [ | reflexivity ].
  cbn [option_map]. unfold rename_demands. apply forallb_map_with. intros d.
  cbn [fst snd]. rewrite rename_continuation_eqb.
  match goal with
  | |- trap_stack_eqb _ (if ?c then _ else _)
       = trap_stack_eqb _ (if ?c' then _ else _) =>
    change c' with c; destruct c
  end.
  - exact (trap_stack_eqb_renamed r (snd d) []).
  - exact (trap_stack_eqb_renamed r (snd d) [unit_exn]).
Qed.

Theorem trap_demands_equivariant : trap_demands_equivariant_statement.
Proof. intros r ts e. apply trap_demands_renamed. Qed.

Theorem traps_well_formed_equivariant :
  traps_well_formed_equivariant_statement.
Proof. exact traps_well_formed_renamed. Qed.

(* The check of a whole unit, by the tactic: the laws above are its
   hints. *)
#[local] Hint Rewrite sets_well_formed_equivariant
  numbers_in_range_equivariant unit_exn_handlers_well_formed_equivariant
  traps_well_formed_renamed : equivariance.
#[local] Hint Rewrite static_names_renamed
  using (intros; solve [ eqv_core ]) : equivariance.
#[local] Instance wf_unit_interface_renaming : Renaming_of unit_interface :=
  rename_unit_interface.

Theorem well_formed_unit_equivariant : well_formed_unit_equivariant_statement.
Proof. equivariance. Qed.

End Well_formed_checks.

(* The states that the machine makes undefined. *)
Lemma code_id_bound_renamed (r : renaming) (statics : static_env)
    (id : code_id) :
  code_id_bound (rename_static_env r statics) id = code_id_bound statics id.
Proof.
  unfold code_id_bound. rewrite lookup_rename_code.
  destruct (lookup id (static_code statics)); reflexivity.
Qed.

Lemma calls_code_binding_statics_equivariant (r : renaming) (ap : apply_expr)
    (vars : env) (statics : static_env) :
  calls_code_binding_statics (rename_apply r ap) (rename_env r vars)
    (rename_static_env r statics)
  = calls_code_binding_statics ap vars statics.
Proof.
  unfold calls_code_binding_statics, rename_apply. cbn [ap_callee].
  rewrite eval_simple_equivariant.
  destruct (eval_simple vars (ap_callee ap)) as [callee | ];
    [ | reflexivity ].
  (* the closure called, which a partial application keeps *)
  match goal with
  | |- context [closure_of statics ?called] => generalize called
  end.
  intros closure.
  replace (closure_of (rename_static_env r statics) closure)
    with (closure_of statics closure)
    by (destruct closure; reflexivity).
  destruct (closure_of statics closure) as [[called slots] | ];
    [ | reflexivity ].
  rewrite lookup_rename_code.
  destruct (lookup called (static_code statics)) as [c | ]; cbn [option_map];
    [ | reflexivity ].
  destruct c as [params my_closure ret exn metadata body]. simpl.
  apply rename_binds_statics.
Qed.

Theorem static_fault_equivariant : static_fault_equivariant_statement.
Proof.
  intros r s. unfold static_fault, rename_state.
  cbn [state_expr state_vars state_statics].
  destruct (state_expr s)
    as [ x kind d body | bindings body | k h body | handlers body | ap | ac
       | scrutinee arms | ];
    cbn [rename_expr]; try reflexivity.
  - rewrite
      (rebinds_renamed r _ (code_ids_of_binding_renamed r)
         (code_id_bound (state_statics s)))
      by apply code_id_bound_renamed.
    rewrite
      (rebinds_renamed r _ (symbols_of_binding_renamed r)
         (symbol_bound (state_statics s)))
      by reflexivity.
    reflexivity.
  - exact (rename_handlers_bind_statics r handlers).
  - apply calls_code_binding_statics_equivariant.
Qed.

(* ======================================================================== *)
(* The passes for loops                                                     *)
(* ======================================================================== *)

(* The rewrites of FlambdaRecHandlersSpec.v, FlambdaSinkContsSpec.v,
   FlambdaShortcutSpec.v, FlambdaUnusedParamsSpec.v and
   FlambdaLoopAliasesSpec.v commute with renaming. None makes a name;
   each compares names by equality only. The statements are in
   FlambdaValidatorEquivarianceSpec.v; the lemma here that has the name
   of a statement proves it.

   The proofs are by the tactic of FlambdaEquivarianceTactic.v where it
   does them. The passes are in one section, so that what is given to
   the tactic here is given to it here only: the law of a function, once
   proved, is a hint for the proofs after it, as the proofs by hand
   rewrote with the lemmas before them. *)
Section Passes_for_loops.

(* Laws of the free names that are proved by hand in
   FlambdaValidatorEquivarianceBaseProofs.v. The tactic derives them
   where it meets those functions, in every case of every induction
   again; given as hints they are not derived. *)
#[local] Hint Rewrite var_in_simple_equivariant var_in_value_slots_equivariant
  var_in_named_equivariant var_in_static_binding_equivariant
  var_free_b_equivariant_aux var_free_in_handler_b_equivariant
  cont_free_b_equivariant_aux cont_free_in_handler_b_equivariant
  : equivariance.

(* Two laws that are proved further up in this file, for the same
   reason. *)
#[local] Hint Rewrite no_trap_action_renamed
  inline_cont_at_tail_equivariant_aux : equivariance.

(* The tactic looks for the laws of the database in one pass over the
   goal in the proofs below, which is several times faster than its
   usual way. The one pass does not see a law inside a [match] or
   under a binder; the two proofs that need the usual way take it
   back, each for itself. *)
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

(* A rewrite applied to the handlers of recursive continuations. *)
Section Rewrite_rec_handlers.

Variable f : expr -> expr.
Hypothesis f_equivariant : rewrite_equivariant f.

Lemma rewrite_rec_handlers_equivariant :
  rewrite_equivariant (rewrite_rec_handlers f).
Proof. equivariance. Qed.

End Rewrite_rec_handlers.

(* Moving a continuation to the handler that alone uses it. *)
Lemma params_not_free_equivariant (r : renaming) (params : list variable)
    (h : continuation_handler) :
  params_not_free (map (rename_variable r) params) (rename_handler r h)
  = params_not_free params h.
Proof. equivariance. Qed.
#[local] Hint Rewrite params_not_free_equivariant : equivariance.

Lemma sink_let_conts_equivariant_aux (r : renaming) (e : expr) :
  sink_let_conts (rename_expr r e) = rename_expr r (sink_let_conts e).
Proof. equivariance. Qed.

Lemma sink_let_conts_equivariant : rewrite_equivariant sink_let_conts.
Proof. exact sink_let_conts_equivariant_aux. Qed.

(* Moving a continuation down past the bindings under it. *)
Lemma sink_conts_to_uses_equivariant_aux (r : renaming) (e : expr) :
  sink_conts_to_uses (rename_expr r e)
  = rename_expr r (sink_conts_to_uses e).
Proof. equivariance. Qed.

Lemma sink_conts_to_uses_equivariant :
  rewrite_equivariant sink_conts_to_uses.
Proof. exact sink_conts_to_uses_equivariant_aux. Qed.

(* An arm of a switch that applies a continuation which only applies
   another. *)
Lemma shortcut_arm_equivariant (r : renaming) (k : continuation)
    (target : apply_cont) (arm : Z * apply_cont) :
  shortcut_arm (rename_continuation r k) (rename_apply_cont r target)
    (fst arm, rename_apply_cont r (snd arm))
  = (fst (shortcut_arm k target arm),
     rename_apply_cont r (snd (shortcut_arm k target arm))).
Proof.
  destruct arm as [i ac]. unfold shortcut_arm. cbn [fst snd].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  destruct (ac_args ac) as [ | arg args]; [ | reflexivity ]. cbn [map].
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  rewrite rename_continuation_eqb, no_trap_action_renamed.
  destruct (Pos.eqb (ac_continuation ac) k && no_trap_action ac);
    reflexivity.
Qed.

Lemma shortcut_switch_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  shortcut_switch (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_expr r (shortcut_switch k h body).
Proof.
  destruct h as [[ | param params] metadata handler]; [ | reflexivity ].
  destruct handler as [ | | | | | target | | ]; try reflexivity.
  destruct body as [ | | | | | | scrutinee arms | ]; try reflexivity.
  cbn [shortcut_switch rename_handler rename_expr map].
  destruct (hm_params_kinds metadata); [ | reflexivity ].
  change (ac_continuation (rename_apply_cont r target))
    with (rename_continuation r (ac_continuation target)).
  rewrite rename_continuation_eqb, no_trap_action_renamed.
  destruct
    (Pos.eqb (ac_continuation target) k || hm_is_exn_handler metadata
     || negb (no_trap_action target));
    [ reflexivity | ].
  rewrite !map_map.
  rewrite
    (map_ext _
       (fun arm =>
          (fst (shortcut_arm k target arm),
           rename_apply_cont r (snd (shortcut_arm k target arm))))
       (shortcut_arm_equivariant r k target)).
  rewrite <- (map_map (shortcut_arm k target)
                (fun arm => (fst arm, rename_apply_cont r (snd arm)))).
  rewrite arms_apply_equivariant.
  destruct (arms_apply k (map (shortcut_arm k target) arms)); reflexivity.
Qed.

Fixpoint shortcut_switches_equivariant_aux (r : renaming) (e : expr) :
  shortcut_switches (rename_expr r e) = rename_expr r (shortcut_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite shortcut_switches_equivariant_aux. reflexivity.
  - rewrite shortcut_switches_equivariant_aux. reflexivity.
  - rewrite !shortcut_switches_equivariant_aux.
    exact
      (shortcut_switch_equivariant r k
         (Handler params metadata (shortcut_switches handler))
         (shortcut_switches body)).
  - rewrite shortcut_switches_equivariant_aux. reflexivity.
Qed.

Lemma shortcut_switches_equivariant : rewrite_equivariant shortcut_switches.
Proof. exact shortcut_switches_equivariant_aux. Qed.

(* A continuation whose handler is that of a continuation bound around
   it applies that one. *)

(* The comparison of two handlers with no names paired. *)
Lemma handler_eqb_closed_equivariant (r : renaming)
    (h h' : continuation_handler) :
  handler_eqb [] [] (rename_handler r h) (rename_handler r h')
  = handler_eqb [] [] h h'.
Proof.
  pose proof (expr_eqb_equivariant r [] []
    (Let_cont_non_recursive 1%positive h Invalid)
    (Let_cont_non_recursive 1%positive h' Invalid)) as Heq.
  cbn [rename_name_map map rename_expr expr_eqb] in Heq.
  rewrite !andb_true_r in Heq. exact Heq.
Qed.

Lemma same_handler_renamed (r : renaming) (h : continuation_handler)
    (xs : list variable) (ks : list continuation)
    (h' : continuation_handler) :
  same_handler (rename_handler r h) (map (rename_variable r) xs)
    (map (rename_continuation r) ks) (rename_handler r h')
  = same_handler h xs ks h'.
Proof.
  unfold same_handler. rewrite handler_eqb_closed_equivariant.
  rewrite
    (existsb_map_with
       (fun x => var_free_in_handler_b x (rename_handler r h'))
       (rename_variable r) (fun x => var_free_in_handler_b x h') xs
       (fun x => var_free_in_handler_b_equivariant r x h')),
    (existsb_map_with
       (fun c => cont_free_in_handler_b c (rename_handler r h'))
       (rename_continuation r) (fun c => cont_free_in_handler_b c h') ks
       (fun c => cont_free_in_handler_b_equivariant r c h')).
  destruct h' as [params' metadata' handler'].
  cbn [rename_handler handler_params].
  rewrite distinct_names_renamed. reflexivity.
Qed.
#[local] Hint Rewrite same_handler_renamed : equivariance.
#[local] Arguments same_handler : simpl never.

Lemma share_handlers_equivariant_aux (r : renaming) (e : expr) :
  share_handlers (rename_expr r e) = rename_expr r (share_handlers e).
Proof. equivariance. Qed.

Lemma share_handlers_equivariant : rewrite_equivariant share_handlers.
Proof. exact share_handlers_equivariant_aux. Qed.

(* A pass that carries a state down the term and adds to it: in the
   renamed term the state is then a list made of renamed parts, and the
   tactic does not use the hypothesis of the induction there. This
   rule, for the function [f] of the pass and the renaming [rn] of its
   state, takes the state from the other side of the equation and
   shows that the list is that state renamed. *)
#[local] Ltac eqv_state_back f rn :=
  match goal with
  | r : renaming,
    H : context [f _ (rename_expr _ ?e) = rename_expr _ (f _ ?e)]
    |- context [f ?m' (rename_expr ?r ?e)] =>
    match goal with
    | |- context [f ?m e] =>
      eqv_no_renaming m;
      let m'' := eval cbv beta in (rn r m) in
      tryif constr_eq m' m'' then fail
      else (replace m' with m''; [ rewrite H | solve [ eqv_core ] ])
    end
  end.

(* Removing the parameters of a continuation that its handler does not
   use. The masks are keyed by continuations, which are renamed. *)
Lemma keep_map {A B : Type} (g : A -> B) (mask : list bool) :
  ∀ l : list A, keep mask (map g l) = map g (keep mask l).
Proof.
  induction mask as [ | b mask IH]; intros [ | a l]; simpl; try reflexivity.
  - destruct b; reflexivity.
  - destruct b; simpl; rewrite IH; reflexivity.
Qed.

Lemma keep_all_or_map {A B : Type} (g : A -> B) (mask : option (list bool))
    (l : list A) :
  keep_all_or mask (map g l) = map g (keep_all_or mask l).
Proof. destruct mask; [ apply keep_map | reflexivity ]. Qed.

Lemma returned_to_equivariant (r : renaming) (k : continuation)
    (e : expr) :
  returned_to (rename_continuation r k) (rename_expr r e) = returned_to k e.
Proof. equivariance. Qed.

Lemma used_params_equivariant (r : renaming) (is_exn_handler : bool)
    (params : list variable) (handler : expr) :
  used_params is_exn_handler (map (rename_variable r) params)
    (rename_expr r handler)
  = used_params is_exn_handler params handler.
Proof.
  unfold used_params. rewrite map_map.
  rewrite (map_ext _ (fun param => var_free_b param handler))
    by (intros param; apply var_free_b_equivariant_aux).
  reflexivity.
Qed.

(* In a loop [returned_to] is asked of the whole term of the case, which
   computation has taken apart: its law at that form. *)
Lemma returned_to_rec_renamed (r : renaming) (k : continuation)
    (handlers : list (continuation * continuation_handler)) (body : expr) :
  returned_to (rename_continuation r k)
    (Let_cont_recursive
       (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
          handlers)
       (rename_expr r body))
  = returned_to k (Let_cont_recursive handlers body).
Proof.
  exact (returned_to_equivariant r k (Let_cont_recursive handlers body)).
Qed.

(* This pass and the next carry masks down the term. The law of each is
   proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states the law of the function of
   masks and a term from its type and proves it as it goes: the masks
   are renamed as their type says, the name of each continuation and
   nothing of what a mask holds. Both are given the four laws above.
   The tactic does not derive those of [keep_all_or] and of
   [used_params], and without those of [returned_to] it leaves the
   goal; [returned_to] is kept from unfolding, so that its law at the
   term of a loop applies. *)
Section Masks.

#[local] Hint Rewrite @keep_all_or_map returned_to_equivariant
  used_params_equivariant returned_to_rec_renamed : equivariance.

#[local] Arguments returned_to : simpl never.

#[local] Abbreviation rename_masks r :=
  (map (fun p : continuation * option (list bool) =>
          (rename_continuation r (fst p), snd p))).

(* The laws of [mask_of] and of [drop_args], which both passes call at
   every application of a continuation: stated, they are not derived in
   each proof. *)
Lemma mask_of_renamed (r : renaming) (known : masks) (k : continuation) :
  mask_of (rename_masks r known) (rename_continuation r k) = mask_of known k.
Proof.
  unfold mask_of.
  induction known as [ | [k' mask] known IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma drop_args_renamed (r : renaming) (known : masks) (ac : apply_cont) :
  drop_args (rename_masks r known) (rename_apply_cont r ac)
  = rename_apply_cont r (drop_args known ac).
Proof.
  unfold drop_args, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite mask_of_renamed, keep_all_or_map. reflexivity.
Qed.
#[local] Hint Rewrite mask_of_renamed drop_args_renamed : equivariance.

(* The masks are added to under a continuation and in a loop: the rule
   above. *)
#[local] Ltac eqv_extra ::=
  eqv_state_back drop_params (fun r : renaming => rename_masks r).

Lemma drop_params_renamed : equivariance_of drop_params.
Proof. equivariance. Qed.

Lemma drop_unused_params_equivariant : rewrite_equivariant drop_unused_params.
Proof. intros r e. exact (drop_params_renamed r [] e). Qed.

(* Removing the parameters of a loop that are only passed round it.
   What is stated here is what the tactic does not do by itself.
   - The search for the masks of a loop ([settle]) takes a function, so
     its law is a lemma about two functions that agree up to renaming,
     and a rule gives the lemma to the tactic. Computation is kept from
     unfolding [settle], so that the rule finds it.
   - The hypothesis of the induction is about renamed masks. In a loop
     the masks are a list made of renamed parts, and where the call is
     under a function whose answer has no names the tactic does not use
     the hypothesis from right to left. A second rule, as for the pass
     above, takes the masks from the other side of the equation and
     shows that the list is those masks renamed. *)
Lemma count_params_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  count_params
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
  = count_params handlers.
Proof. equivariance. Qed.
#[local] Hint Rewrite count_params_renamed : equivariance.

Lemma masks_eqb_equivariant : equivariance_of masks_eqb.
Proof. equivariance. Qed.

(* The search goes the same way on the renamed term, through masks
   whose names are renamed. *)
Lemma settle_renamed (r : renaming) (fuel : nat)
    (next next' : masks -> masks) (first first' : masks) :
  first' = rename_masks r first ->
  (∀ loop : masks, next' (rename_masks r loop) = rename_masks r (next loop)) ->
  settle fuel next' first'
  = option_map (rename_masks r) (settle fuel next first).
Proof.
  intros -> Hnext. revert first.
  induction fuel as [ | fuel IH]; intros loop; cbn [settle];
    [ reflexivity | ].
  rewrite Hnext, masks_eqb_equivariant.
  destruct (masks_eqb (next loop) loop); [ reflexivity | apply IH ].
Qed.

#[local] Arguments settle : simpl never.

(* The rules. The law of [count_params], which the one pass over the
   database does not see where the handlers are renamed under a
   binder, so that the two searches have the same fuel. Then: [settle]
   at a function and at first masks that mention the renaming, where
   the other side has it at ones that do not. Last, the rule above for
   the masks that the pass adds to. *)
#[local] Ltac eqv_extra ::=
  first
    [ rewrite count_params_renamed
    | match goal with
      | r : renaming |- context [settle ?fuel ?next' ?first'] =>
        eqv_has_renaming next';
        match goal with
        | |- context [settle fuel ?next ?first] =>
          eqv_no_renaming next;
          rewrite (settle_renamed r fuel next next' first first')
        end
      end
    | eqv_state_back drop_dead (fun r : renaming => rename_masks r) ].

Lemma drop_dead_params_equivariant : rewrite_equivariant drop_dead_params.
Proof. equivariance. Qed.

End Masks.

(* Removing the unused [Let]s whose defining expression has a value, in
   the handlers of recursive continuations too. The pass adds to the
   kinds in scope at a [Let]; in the renamed term they are then a
   renamed binding put before renamed kinds, and the tactic does not
   use the hypothesis of the induction there. The rule writes them as
   the kinds of the other side of the equation renamed, which they are
   by computation, and uses the hypothesis. *)
Section Loop_lets.

#[local] Hint Rewrite named_defined_equivariant params_in_scope_equivariant
  : equivariance.

#[local] Ltac eqv_extra ::=
  eqv_state_back remove_unused_defined_lets_in_loops rename_kinds_in_scope.

Lemma remove_unused_defined_lets_in_loops_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ kinds : kinds_in_scope,
    remove_unused_defined_lets_in_loops (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets_in_loops kinds e).
Proof. equivariance. Qed.

End Loop_lets.

(* The same in the bodies of code, with the parameters of the code in
   scope. *)
Lemma code_in_scope_renamed (r : renaming) (params : list variable)
    (my_closure : variable) (metadata : code_metadata) :
  code_in_scope (map (rename_variable r) params) (rename_variable r my_closure)
    metadata
  = rename_kinds_in_scope r (code_in_scope params my_closure metadata).
Proof.
  exact (params_in_scope_equivariant r params _ [(my_closure, Value)]).
Qed.
#[local] Hint Rewrite code_in_scope_renamed
  remove_unused_defined_lets_in_loops_equivariant_aux : equivariance.

Lemma remove_unused_defined_lets_in_code_equivariant :
  rewrite_equivariant remove_unused_defined_lets_in_code.
Proof. equivariance. Qed.

(* Binding the parameters that every application gives the same thing. *)
#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
Lemma simple_eqb_renamed (r : renaming) (s s' : simple) :
  simple_eqb (rename_simple r s) (rename_simple r s') = simple_eqb s s'.
Proof. equivariance. Qed.
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Lemma binds_var_equivariant (r : renaming) (x : variable) (e : expr) :
  binds_var (rename_variable r x) (rename_expr r e) = binds_var x e.
Proof. equivariance. Qed.
#[local] Hint Rewrite binds_var_equivariant : equivariance.

Lemma stable_equivariant (r : renaming) (s : simple) (e : expr) :
  stable (rename_simple r s) (rename_expr r e) = stable s e.
Proof. equivariance. Qed.

Lemma args_if_applies_equivariant (r : renaming) (k : continuation)
    (ac : apply_cont) :
  args_if_applies (rename_continuation r k) (rename_apply_cont r ac)
  = option_map (map (rename_simple r)) (args_if_applies k ac).
Proof. equivariance. Qed.

Lemma first_some_map {A B : Type} (g : A -> B) (a b : option A) :
  first_some (option_map g a) (option_map g b)
  = option_map g (first_some a b).
Proof. destruct a; reflexivity. Qed.

Fixpoint some_args_equivariant (r : renaming) (k : continuation) (e : expr) :
  some_args (rename_continuation r k) (rename_expr r e)
  = option_map (map (rename_simple r)) (some_args k e)
with some_args_in_handler_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) :
  some_args_in_handler (rename_continuation r k) (rename_handler r h)
  = option_map (map (rename_simple r)) (some_args_in_handler k h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try reflexivity.
    + apply some_args_equivariant.
    + apply some_args_equivariant.
    + rewrite some_args_equivariant, some_args_in_handler_equivariant.
      apply first_some_map.
    + rewrite some_args_equivariant, <- first_some_map. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (some_args_in_handler_equivariant r k h), IH.
      apply first_some_map.
    + apply args_if_applies_equivariant.
    + induction arms as [ | [discriminant ac] arms IH]; simpl;
        [ reflexivity | ].
      rewrite args_if_applies_equivariant, IH. apply first_some_map.
  - destruct h as [params metadata handler]. simpl.
    apply some_args_equivariant.
Qed.

Lemma given_equivariant (r : renaming) (k : continuation) (i : nat)
    (ok ok' : simple -> bool) (ac : apply_cont) :
  (∀ arg : simple, ok' (rename_simple r arg) = ok arg) ->
  given (rename_continuation r k) i ok' (rename_apply_cont r ac)
  = given k i ok ac.
Proof.
  intros Hok. unfold given, rename_apply_cont.
  cbn [ac_continuation ac_args].
  rewrite rename_continuation_eqb, nth_error_map.
  destruct (nth_error (ac_args ac) i); simpl; [ rewrite Hok | ]; reflexivity.
Qed.

Fixpoint always_given_equivariant (r : renaming) (k : continuation)
    (i : nat) (ok ok' : simple -> bool) (e : expr) :
  (∀ arg : simple, ok' (rename_simple r arg) = ok arg) ->
  always_given (rename_continuation r k) i ok' (rename_expr r e)
  = always_given k i ok e
with always_given_in_handler_equivariant (r : renaming) (k : continuation)
    (i : nat) (ok ok' : simple -> bool) (h : continuation_handler) :
  (∀ arg : simple, ok' (rename_simple r arg) = ok arg) ->
  always_given_in_handler (rename_continuation r k) i ok' (rename_handler r h)
  = always_given_in_handler k i ok h.
Proof.
  - intros Hok.
    destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try reflexivity.
    + apply always_given_equivariant, Hok.
    + apply always_given_equivariant, Hok.
    + rewrite (always_given_in_handler_equivariant r k i ok ok' h Hok),
        (always_given_equivariant r k i ok ok' body Hok).
      reflexivity.
    + rewrite (always_given_equivariant r k i ok ok' body Hok). f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (always_given_in_handler_equivariant r k i ok ok' h Hok), IH.
      reflexivity.
    + apply given_equivariant, Hok.
    + induction arms as [ | [discriminant ac] arms IH]; simpl;
        [ reflexivity | ].
      rewrite (given_equivariant r k i ok ok' ac Hok), IH. reflexivity.
  - intros Hok. destruct h as [params metadata handler]. simpl.
    apply always_given_equivariant, Hok.
Qed.

Lemma aliases_from_equivariant (r : renaming) (params : list variable) :
  ∀ (j : nat) (alias_at alias_at' : nat -> variable -> option simple),
    (∀ (i : nat) (param : variable),
      alias_at' i (rename_variable r param)
      = option_map (rename_simple r) (alias_at i param)) ->
    aliases_from j (map (rename_variable r) params) alias_at'
    = map (option_map (rename_simple r)) (aliases_from j params alias_at).
Proof.
  induction params as [ | param params IH]; intros j alias_at alias_at' Hat;
    simpl; [ reflexivity | ].
  rewrite Hat, (IH (S j) alias_at alias_at' Hat). reflexivity.
Qed.

Lemma aliases_of_equivariant (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (loop loop' : option expr) (body whole : expr) :
  loop' = option_map (rename_expr r) loop ->
  aliases_of (rename_continuation r k) (map (rename_variable r) params)
    metadata loop' (rename_expr r body) (rename_expr r whole)
  = map (option_map (rename_simple r))
      (aliases_of k params metadata loop body whole).
Proof.
  intros ->. unfold aliases_of.
  rewrite returned_to_equivariant,
    (distinct_map_injective _ (rename_variable_eqb r)), length_map,
    some_args_equivariant.
  destruct
    (returned_to k whole || hm_is_exn_handler metadata
     || negb (distinct params)
     || negb (Nat.eqb (length (hm_params_kinds metadata)) (length params)));
    [ reflexivity | ].
  destruct (some_args k body) as [args | ]; simpl; [ | reflexivity ].
  apply aliases_from_equivariant. intros i param.
  rewrite nth_error_map.
  destruct (nth_error args i) as [s | ]; simpl; [ | reflexivity ].
  rewrite stable_equivariant,
    (always_given_equivariant r k i (fun arg => simple_eqb arg s))
    by (intros arg; apply simple_eqb_renamed).
  destruct loop as [handler | ]; simpl.
  - rewrite
      (always_given_equivariant r k i
         (fun arg => simple_eqb arg s || simple_eqb arg (Var param))),
      binds_var_equivariant
      by (intros arg; rewrite simple_eqb_renamed; f_equal;
          exact (simple_eqb_renamed r arg (Var param))).
    destruct
      (stable s whole && always_given k i (fun arg => simple_eqb arg s) body
       && (always_given k i
             (fun arg => simple_eqb arg s || simple_eqb arg (Var param))
             handler
           && negb (binds_var param handler)));
      reflexivity.
  - destruct
      (stable s whole && always_given k i (fun arg => simple_eqb arg s) body
       && true);
      reflexivity.
Qed.

Lemma bind_aliases_equivariant (r : renaming) (params : list variable) :
  ∀ (kinds : list flambda_kind) (aliases : list (option simple))
    (handler : expr),
    bind_aliases (map (rename_variable r) params) kinds
      (map (option_map (rename_simple r)) aliases) (rename_expr r handler)
    = rename_expr r (bind_aliases params kinds aliases handler).
Proof. equivariance. Qed.

Lemma alone_map {A B : Type} (g : A -> B) (l : list A) :
  alone (map g l) = alone l.
Proof. destruct l as [ | a [ | b l]]; reflexivity. Qed.

(* The law of the pass is by the tactic. The aliases of a continuation
   are asked of the whole term of the case, which computation has taken
   apart. [aliases_of] is kept from unfolding, and the rule writes the
   term that it is asked of as the term of the other side of the
   equation renamed, which it is by computation, and uses the law
   above. *)
Section Param_aliases.

#[local] Arguments aliases_of : simpl never.

#[local] Hint Rewrite bind_aliases_equivariant @alone_map : equivariance.

#[local] Ltac eqv_extra ::=
  match goal with
  | |- context [aliases_of (rename_continuation ?r ?k) _ ?metadata ?loop' _
                  ?whole'] =>
    match goal with
    | |- context [aliases_of k ?params metadata ?loop ?body ?whole] =>
      eqv_no_renaming whole;
      change whole' with (rename_expr r whole);
      rewrite
        (aliases_of_equivariant r k params metadata loop loop' body whole
           eq_refl)
    end
  end.

Lemma bind_aliased_params_equivariant :
  rewrite_equivariant bind_aliased_params.
Proof. equivariance. Qed.

End Param_aliases.

(* An application made to go where the handler of its continuation
   goes. The entries are keyed by continuations, which are renamed, and
   keep parameters and an [Apply_cont], which are renamed too. *)
#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.
Lemma binds_cont_equivariant (r : renaming) (k : continuation) (e : expr) :
  binds_cont (rename_continuation r k) (rename_expr r e) = binds_cont k e.
Proof. equivariance. Qed.
#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

Definition rename_forward_entry (r : renaming)
    (entry : option (list variable * apply_cont))
    : option (list variable * apply_cont) :=
  option_map
    (fun entry =>
       (map (rename_variable r) (fst entry), rename_apply_cont r (snd entry)))
    entry.

Definition rename_forwards (r : renaming) (F : forwards) : forwards :=
  map
    (fun binding =>
       (rename_continuation r (fst binding),
        rename_forward_entry r (snd binding)))
    F.

Lemma forward_of_renamed (r : renaming) (F : forwards) (k : continuation) :
  forward_of (rename_forwards r F) (rename_continuation r k)
  = rename_forward_entry r (forward_of F k).
Proof.
  unfold forward_of, rename_forwards.
  induction F as [ | [k' entry] F IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma subst_simple_equivariant (r : renaming) (params : list variable)
    (args : list simple) (s : simple) :
  subst_simple (map (rename_variable r) params) (map (rename_simple r) args)
    (rename_simple r s)
  = rename_simple r (subst_simple params args s).
Proof. equivariance. Qed.

Lemma forwarded_equivariant (r : renaming) (params : list variable)
    (args : list simple) (target : apply_cont) :
  forwarded (map (rename_variable r) params) (map (rename_simple r) args)
    (rename_apply_cont r target)
  = rename_apply_cont r (forwarded params args target).
Proof.
  unfold forwarded, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action]. rewrite !map_map.
  f_equal. apply map_ext. intros s. apply subst_simple_equivariant.
Qed.

Lemma forward_apply_cont_equivariant (r : renaming) (F : forwards)
    (ac : apply_cont) :
  forward_apply_cont (rename_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_apply_cont F ac).
Proof.
  unfold forward_apply_cont.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  rewrite forward_of_renamed.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ];
    [ | reflexivity ].
  cbn [rename_forward_entry option_map fst snd].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite !length_map, !no_trap_action_renamed.
  destruct
    (Nat.eqb (length (ac_args ac)) (length params)
     && no_trap_action ac && no_trap_action target);
    [ apply forwarded_equivariant | reflexivity ].
Qed.

Lemma forwarded_keeping_equivariant (r : renaming)
    (action : option trap_action) (params : list variable)
    (args : list simple) (target : apply_cont) :
  forwarded_keeping (option_map (rename_trap_action r) action)
    (map (rename_variable r) params) (map (rename_simple r) args)
    (rename_apply_cont r target)
  = rename_apply_cont r (forwarded_keeping action params args target).
Proof.
  unfold forwarded_keeping, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action]. rewrite !map_map.
  f_equal. apply map_ext. intros s. apply subst_simple_equivariant.
Qed.

Lemma forward_apply_cont_keeping_equivariant (r : renaming) (F : forwards)
    (ac : apply_cont) :
  forward_apply_cont_keeping (rename_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_apply_cont_keeping F ac).
Proof.
  unfold forward_apply_cont_keeping.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  rewrite forward_of_renamed.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ];
    [ | reflexivity ].
  cbn [rename_forward_entry option_map fst snd].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite !length_map, !no_trap_action_renamed.
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target);
    [ | reflexivity ].
  change (ac_trap_action (rename_apply_cont r ac))
    with (option_map (rename_trap_action r) (ac_trap_action ac)).
  apply forwarded_keeping_equivariant.
Qed.

Lemma forward_arm_equivariant (r : renaming) (F : forwards)
    (ac : apply_cont) :
  forward_arm (rename_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_arm F ac).
Proof.
  pose proof (forward_apply_cont_keeping_equivariant r F ac) as Hgeneral.
  unfold forward_arm.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite forward_of_renamed.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hof;
    cbn [rename_forward_entry option_map fst snd].
  - destruct params as [ | p params], (ac_args ac) as [ | arg args] eqn:Hargs;
      cbn [map]; try exact Hgeneral.
    rewrite no_trap_action_renamed.
    destruct (no_trap_action ac);
      [ exact (forwarded_equivariant r [] [] target) | exact Hgeneral ].
  - destruct (ac_args ac); exact Hgeneral.
Qed.

Lemma passes_on_renamed (r : renaming) (params : list variable) :
  ∀ args : list simple,
    passes_on (map (rename_variable r) params) (map (rename_simple r) args)
    = passes_on params args.
Proof.
  induction params as [ | p params IH]; intros [ | [x | sym | c] args];
    simpl; try reflexivity.
  rewrite rename_variable_eqb, IH. reflexivity.
Qed.

Lemma forward_apply_equivariant (r : renaming) (F : forwards)
    (ap : apply_expr) :
  forward_apply (rename_forwards r F) (rename_apply r ap)
  = rename_apply r (forward_apply F ap).
Proof.
  unfold forward_apply, alias_of.
  change (ap_continuation (rename_apply r ap))
    with (rename_continuation r (ap_continuation ap)).
  rewrite forward_of_renamed.
  destruct (forward_of F (ap_continuation ap)) as [[params target] | ];
    [ | reflexivity ].
  cbn [rename_forward_entry option_map fst snd].
  change (ac_args (rename_apply_cont r target))
    with (map (rename_simple r) (ac_args target)).
  rewrite no_trap_action_renamed, passes_on_renamed.
  destruct (no_trap_action target && passes_on params (ac_args target));
    reflexivity.
Qed.

Lemma forwarding_equivariant (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  forwarding (rename_continuation r k) (map (rename_variable r) params)
    metadata (rename_expr r handler) (rename_expr r body)
  = rename_forward_entry r (forwarding k params metadata handler body).
Proof.
  destruct handler as [ | | | | | target | | ]; try reflexivity.
  cbn [rename_expr forwarding].
  rewrite among_apply_cont_continuations_renamed,
    (distinct_map_injective _ (rename_variable_eqb r)),
    apply_cont_continuations_renamed.
  rewrite
    (existsb_map_with (fun j => binds_cont j (rename_expr r body))
       (rename_continuation r) (fun j => binds_cont j body)
       (apply_cont_continuations target)
       (fun j => binds_cont_equivariant r j body)).
  change (ac_args (rename_apply_cont r target))
    with (map (rename_simple r) (ac_args target)).
  rewrite
    (existsb_map_with _ (rename_simple r)
       (fun s =>
          match s with
          | Var x => negb (among x params) && binds_var x body
          | Symbol _ | Const _ => false
          end)
       (ac_args target)).
  - match goal with
    | |- (if ?check then _ else _) = _ => destruct check; reflexivity
    end.
  - intros [x | sym | c]; cbn [rename_simple]; try reflexivity.
    rewrite among_rename_variable, binds_var_equivariant. reflexivity.
Qed.

(* The law of the pass is by the tactic, which is given the laws above
   of what the pass does at an application, and a rule. The pass adds
   an entry at a continuation and entries at a loop; in the renamed
   term the entries are then a list made of renamed parts, and the
   tactic does not use the hypothesis of the induction there. The rule
   takes the entries from the other side of the equation and shows that
   the list is those entries renamed. *)
Section Forward_applications.

#[local] Hint Rewrite forward_apply_equivariant
  forward_apply_cont_keeping_equivariant forward_arm_equivariant
  forwarding_equivariant : equivariance.

#[local] Ltac eqv_extra ::=
  match goal with
  | r : renaming,
    H : context [forward_applications_under _ _ (rename_expr _ ?e)
                 = rename_expr _ (forward_applications_under _ _ ?e)]
    |- context [forward_applications_under ?c ?m' (rename_expr ?r ?e)] =>
    lazymatch m' with
    | rename_forwards _ _ => fail
    | _ => idtac
    end;
    match goal with
    | |- context [forward_applications_under c ?m e] =>
      eqv_no_renaming m;
      replace m' with (rename_forwards r m);
      [ rewrite H | solve [ eqv_core ] ]
    end
  end.

Lemma forward_applications_under_equivariant (r : renaming) (e : expr) :
  ∀ (calls : bool) (F : forwards),
    forward_applications_under calls (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_applications_under calls F e).
Proof. equivariance. Qed.

End Forward_applications.

Lemma forward_applications_with_equivariant (calls : bool) :
  rewrite_equivariant (forward_applications_with calls).
Proof.
  intros r e. exact (forward_applications_under_equivariant r e calls []).
Qed.

Lemma forward_applications_equivariant :
  rewrite_equivariant forward_applications.
Proof. exact (forward_applications_with_equivariant false). Qed.

(* The same for the continuations of no parameters
   (FlambdaForwardJumpsSpec.v): the entry is that of [forwarding] or
   none, by tests that a renaming does not change. *)
Lemma jump_entry_equivariant (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  jump_entry (rename_continuation r k) (map (rename_variable r) params)
    metadata (rename_expr r handler) (rename_expr r body)
  = rename_forward_entry r (jump_entry k params metadata handler body).
Proof.
  unfold jump_entry.
  destruct params as [ | p params]; [ | reflexivity ].
  destruct (hm_params_kinds metadata); [ | reflexivity ].
  exact (forwarding_equivariant r k [] metadata handler body).
Qed.

(* The law of the pass by the tactic, with the same rule as for the pass
   above. *)
Section Forward_jumps.

#[local] Hint Rewrite forward_apply_cont_equivariant jump_entry_equivariant
  : equivariance.

#[local] Ltac eqv_extra ::=
  eqv_state_back forward_jumps_under rename_forwards.

Lemma forward_jumps_under_equivariant (r : renaming) (e : expr) :
  ∀ F : forwards,
    forward_jumps_under (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_jumps_under F e).
Proof. equivariance. Qed.

End Forward_jumps.

Lemma forward_jumps_equivariant : rewrite_equivariant forward_jumps.
Proof.
  intros r e. exact (forward_jumps_under_equivariant r e []).
Qed.

(* The passes below take the database in the tactic's usual way. *)
#[local] Ltac eqv_database ::= eqv_database_by_autorewrite.

(* The same where the arguments are known to have the kinds of the
   parameters (FlambdaForwardKindedSpec.v). The kinds of the variables
   in scope are renamed with the variables; those that the handlers
   of the continuations in scope give their parameters are kept by
   continuation, and a renaming acts on the continuations. *)
Definition rename_cont_kinds (r : renaming) (K : cont_kinds) : cont_kinds :=
  map (fun entry => (rename_continuation r (fst entry), snd entry)) K.

(* The tactic is told how the three states of the pass are renamed. It
   is given the laws of what the pass does at an application of a
   continuation and at an arm, and of the entry kept of a continuation,
   which are by hand for the time of compiling. *)
Section Forward_kinded.

#[local] Existing Instance kinds_in_scope_renaming.
#[local] Instance cont_kinds_renaming : Renaming_of cont_kinds :=
  rename_cont_kinds.
#[local] Instance forwards_renaming : Renaming_of forwards :=
  rename_forwards.

#[local] Hint Rewrite kind_of_simple_in_scope_equivariant
  params_in_scope_equivariant : equivariance.

Lemma lookup_rename_cont_kinds (r : renaming) (k : continuation)
    (K : cont_kinds) :
  lookup (rename_continuation r k) (rename_cont_kinds r K) = lookup k K.
Proof. equivariance. Qed.

Lemma have_kinds_equivariant (r : renaming) (G : kinds_in_scope)
    (args : list simple) :
  ∀ kinds : list flambda_kind,
    have_kinds (rename_kinds_in_scope r G) (map (rename_simple r) args) kinds
    = have_kinds G args kinds.
Proof. equivariance. Qed.

Lemma kinded_entry_equivariant (r : renaming) (k : continuation)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  kinded_entry (rename_continuation r k) (map (rename_variable r) params)
    metadata (rename_expr r handler) (rename_expr r body)
  = rename_forward_entry r (kinded_entry k params metadata handler body).
Proof.
  unfold kinded_entry. rewrite length_map.
  destruct (Nat.eqb (length params) (length (hm_params_kinds metadata)));
    [ apply forwarding_equivariant | reflexivity ].
Qed.

Lemma forward_kinded_apply_cont_equivariant (r : renaming)
    (G : kinds_in_scope) (K : cont_kinds) (F : forwards) (ac : apply_cont) :
  forward_kinded_apply_cont (rename_kinds_in_scope r G)
    (rename_cont_kinds r K) (rename_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_kinded_apply_cont G K F ac).
Proof.
  unfold forward_kinded_apply_cont.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  rewrite forward_of_renamed, lookup_rename_cont_kinds.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ];
    [ | reflexivity ].
  cbn [rename_forward_entry option_map fst snd].
  destruct (lookup (ac_continuation ac) K) as [kinds | ]; [ | reflexivity ].
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite !length_map, !no_trap_action_renamed, have_kinds_equivariant.
  destruct
    (Nat.eqb (length (ac_args ac)) (length params) && no_trap_action target
     && have_kinds G (ac_args ac) kinds);
    [ | reflexivity ].
  change (ac_trap_action (rename_apply_cont r ac))
    with (option_map (rename_trap_action r) (ac_trap_action ac)).
  apply forwarded_keeping_equivariant.
Qed.

Lemma forward_kinded_arm_equivariant (r : renaming) (G : kinds_in_scope)
    (K : cont_kinds) (F : forwards) (ac : apply_cont) :
  forward_kinded_arm (rename_kinds_in_scope r G) (rename_cont_kinds r K)
    (rename_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_kinded_arm G K F ac).
Proof.
  pose proof (forward_kinded_apply_cont_equivariant r G K F ac) as Hgeneral.
  unfold forward_kinded_arm.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite forward_of_renamed, lookup_rename_cont_kinds.
  destruct (forward_of F (ac_continuation ac)) as [[params target] | ] eqn:Hof;
    cbn [rename_forward_entry option_map fst snd].
  - destruct params as [ | p params],
      (lookup (ac_continuation ac) K) as [[ | kind kinds] | ],
      (ac_args ac) as [ | arg args] eqn:Hargs;
      cbn [map]; try exact Hgeneral.
    rewrite no_trap_action_renamed.
    destruct (no_trap_action ac);
      [ exact (forwarded_equivariant r [] [] target) | exact Hgeneral ].
  - destruct (lookup (ac_continuation ac) K) as [[ | kind kinds] | ],
      (ac_args ac);
      exact Hgeneral.
Qed.

#[local] Hint Rewrite kinded_entry_equivariant
  forward_kinded_apply_cont_equivariant forward_kinded_arm_equivariant
  : equivariance.
#[local] Arguments kinded_entry : simpl never.
#[local] Arguments forward_kinded_apply_cont : simpl never.
#[local] Arguments forward_kinded_arm : simpl never.

Lemma forward_kinded_under_equivariant (r : renaming) (e : expr) :
  ∀ (G : kinds_in_scope) (K : cont_kinds) (F : forwards),
    forward_kinded_under (rename_kinds_in_scope r G) (rename_cont_kinds r K)
      (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_kinded_under G K F e).
Proof. equivariance. Qed.

End Forward_kinded.

Lemma forward_kinded_equivariant : rewrite_equivariant forward_kinded.
Proof.
  intros r e. exact (forward_kinded_under_equivariant r e [] [] []).
Qed.

(* The same in the bodies of code, with the kinds of what a call
   binds. *)
Lemma forward_kinded_in_binding_equivariant (r : renaming)
    (binding : static_binding) :
  forward_kinded_in_binding (rename_static_binding r binding)
  = rename_static_binding r (forward_kinded_in_binding binding).
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code forward_kinded_in_binding].
  unfold code_in_scope.
  change [(rename_variable r my_closure, Value)]
    with (rename_kinds_in_scope r [(my_closure, Value)]).
  rewrite params_in_scope_equivariant.
  pose proof
    (forward_kinded_under_equivariant r body
       (params_in_scope params (unarize (cm_params_arity metadata))
          [(my_closure, Value)])
       [] [])
    as Hbody.
  cbn [rename_cont_kinds rename_forwards map] in Hbody.
  rewrite Hbody. reflexivity.
Qed.

(* The law of a binding, where the pass begins with those kinds and
   nothing else, is given to the tactic. *)
Section Forward_kinded_in_code.

#[local] Hint Rewrite forward_kinded_in_binding_equivariant : equivariance.
#[local] Arguments forward_kinded_in_binding : simpl never.

Lemma forward_kinded_in_code_equivariant :
  rewrite_equivariant forward_kinded_in_code.
Proof. equivariance. Qed.

End Forward_kinded_in_code.

(* An application that goes to an arm of the handler of its
   continuation (FlambdaForwardArmSpec.v). What is kept of a
   continuation is renamed with the term: its parameters, its arms and
   the body of its [Let_cont]. *)
Definition rename_arm_entry (r : renaming) (entry : option arm_entry)
    : option arm_entry :=
  option_map
    (fun '(params, metadata, i, arms, body) =>
       (map (rename_variable r) params, metadata, i,
        map (fun arm => (fst arm, rename_apply_cont r (snd arm))) arms,
        rename_expr r body))
    entry.

Definition rename_arm_forwards (r : renaming) (F : arm_forwards)
    : arm_forwards :=
  map
    (fun binding =>
       (rename_continuation r (fst binding), rename_arm_entry r (snd binding)))
    F.

(* The tactic is told how the two states of the pass are renamed. It
   is given the laws of the entry kept of a continuation and of what
   the pass does at an application of a continuation, which are by
   hand: the tactic does not come to the end of the second. *)
Section Forward_to_arms.

#[local] Instance arm_kinds_in_scope_renaming : Renaming_of kinds_in_scope :=
  rename_kinds_in_scope.
#[local] Instance arm_forwards_renaming : Renaming_of arm_forwards :=
  rename_arm_forwards.

#[local] Hint Rewrite params_in_scope_equivariant : equivariance.

Lemma lookup_rename_arm_forwards (r : renaming) (F : arm_forwards)
    (k : continuation) :
  lookup (rename_continuation r k) (rename_arm_forwards r F)
  = option_map (rename_arm_entry r) (lookup k F).
Proof. equivariance. Qed.

Lemma position_of_renamed (r : renaming) (x : variable)
    (params : list variable) :
  position_of (rename_variable r x) (map (rename_variable r) params)
  = position_of x params.
Proof. equivariance. Qed.

Lemma arm_entry_of_equivariant (r : renaming) (kinded : bool)
    (params : list variable) (metadata : handler_metadata)
    (handler body : expr) :
  arm_entry_of kinded (map (rename_variable r) params) metadata
    (rename_expr r handler) (rename_expr r body)
  = rename_arm_entry r (arm_entry_of kinded params metadata handler body).
Proof.
  destruct handler as [ | | | | | | scrutinee arms | ]; try reflexivity.
  destruct scrutinee as [p | sym | c]; try reflexivity.
  cbn [rename_expr rename_simple arm_entry_of].
  rewrite length_map, position_of_renamed.
  destruct
    (kinded
     && negb (Nat.eqb (length params) (length (hm_params_kinds metadata))));
    [ reflexivity | ].
  destruct (position_of p params); reflexivity.
Qed.

Lemma forward_to_arm_apply_cont_equivariant (r : renaming) (kinded : bool)
    (G : kinds_in_scope) (F : arm_forwards) (ac : apply_cont) :
  forward_to_arm_apply_cont kinded (rename_kinds_in_scope r G)
    (rename_arm_forwards r F) (rename_apply_cont r ac)
  = rename_apply_cont r (forward_to_arm_apply_cont kinded G F ac).
Proof.
  unfold forward_to_arm_apply_cont, arm_of_apply_cont.
  change (ac_continuation (rename_apply_cont r ac))
    with (rename_continuation r (ac_continuation ac)).
  change (ac_args (rename_apply_cont r ac))
    with (map (rename_simple r) (ac_args ac)).
  rewrite lookup_rename_arm_forwards.
  destruct (lookup (ac_continuation ac) F)
    as [[[[[[params metadata] i] arms] body] | ] | ];
    try reflexivity.
  cbn [option_map rename_arm_entry].
  rewrite nth_error_map.
  destruct (nth_error (ac_args ac) i) as [[x | sym | c] | ];
    try reflexivity.
  destruct c as [n | | | | | | | | | | | | | | ]; try reflexivity.
  cbn [option_map rename_simple].
  rewrite find_arm_renamed.
  destruct (find_arm n arms) as [target | ]; [ | reflexivity ].
  cbn [option_map].
  pose proof
    (forwarding_equivariant r (ac_continuation ac) params metadata
       (Apply_cont target) body)
    as Hforwarding.
  cbn [rename_expr] in Hforwarding. rewrite Hforwarding.
  destruct
    (forwarding (ac_continuation ac) params metadata (Apply_cont target) body)
    as [[params' target'] | ];
    [ | reflexivity ].
  cbn [rename_forward_entry option_map fst snd].
  rewrite !length_map, !no_trap_action_renamed, have_kinds_equivariant.
  destruct
    (Nat.eqb (length (ac_args ac)) (length params') && no_trap_action ac
     && no_trap_action target'
     && (negb kinded
         || have_kinds G (ac_args ac) (hm_params_kinds metadata)));
    [ apply forwarded_equivariant | reflexivity ].
Qed.

#[local] Hint Rewrite arm_entry_of_equivariant
  forward_to_arm_apply_cont_equivariant : equivariance.
#[local] Arguments arm_entry_of : simpl never.
#[local] Arguments forward_to_arm_apply_cont : simpl never.

Lemma forward_to_arms_under_equivariant (r : renaming) (e : expr) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    forward_to_arms_under kinded (rename_kinds_in_scope r G)
      (rename_arm_forwards r F) (rename_expr r e)
    = rename_expr r (forward_to_arms_under kinded G F e).
Proof. equivariance. Qed.

End Forward_to_arms.

Lemma forward_to_arms_equivariant : rewrite_equivariant forward_to_arms.
Proof.
  intros r e. exact (forward_to_arms_under_equivariant r e false [] []).
Qed.

Lemma forward_to_arms_kinded_equivariant :
  rewrite_equivariant forward_to_arms_kinded.
Proof.
  intros r e. exact (forward_to_arms_under_equivariant r e true [] []).
Qed.

(* The kinded pass in the bodies of code, with the kinds of what a call
   binds. *)
Lemma forward_to_arms_in_binding_equivariant (r : renaming)
    (binding : static_binding) :
  forward_to_arms_in_binding (rename_static_binding r binding)
  = rename_static_binding r (forward_to_arms_in_binding binding).
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code forward_to_arms_in_binding].
  unfold code_in_scope.
  change [(rename_variable r my_closure, Value)]
    with (rename_kinds_in_scope r [(my_closure, Value)]).
  rewrite params_in_scope_equivariant.
  pose proof
    (forward_to_arms_under_equivariant r body true
       (params_in_scope params (unarize (cm_params_arity metadata))
          [(my_closure, Value)])
       [])
    as Hbody.
  cbn [rename_arm_forwards map] in Hbody.
  rewrite Hbody. reflexivity.
Qed.

(* The law of a binding, where the pass begins with those kinds and
   nothing else, is given to the tactic. *)
Section Forward_to_arms_in_code.

#[local] Hint Rewrite forward_to_arms_in_binding_equivariant : equivariance.
#[local] Arguments forward_to_arms_in_binding : simpl never.

Lemma forward_to_arms_in_code_equivariant :
  rewrite_equivariant forward_to_arms_in_code.
Proof. equivariance. Qed.

End Forward_to_arms_in_code.

(* Taking away a parameter that every application gives one variable:
   the lemmas are in FlambdaUnaliasParamsNamesProofs.v. *)
Theorem unalias_params_equivariant : unalias_params_equivariant_statement.
Proof. exact FlambdaUnaliasParamsNamesProofs.unalias_params_renamed. Qed.

(* Giving a parameter the scrutinee where every application gives it
   the number of its arm: the lemmas are in
   FlambdaGiveScrutineeNamesProofs.v. *)
Theorem give_scrutinees_equivariant : give_scrutinees_equivariant_statement.
Proof. exact FlambdaGiveScrutineeNamesProofs.give_scrutinees_renamed. Qed.

(* Putting the number of an arm for the scrutinee in the arguments of
   the arm: the lemmas are in FlambdaArmNumbersNamesProofs.v. *)
Theorem numbers_in_arms_equivariant : numbers_in_arms_equivariant_statement.
Proof. exact FlambdaArmNumbersNamesProofs.numbers_in_arms_renamed. Qed.

(* Removing an unused parameter of a continuation that a call returns
   to: the lemmas are in FlambdaWrapReturnsNamesProofs.v. *)
Theorem wrap_returns_equivariant : wrap_returns_equivariant_statement.
Proof. exact FlambdaWrapReturnsNamesProofs.wrap_returns_renamed. Qed.

(* Untagging the parameters that are only untagged or passed on: the
   lemmas are in FlambdaUntagParamsNamesProofs.v. *)
Theorem untag_params_equivariant : untag_params_equivariant_statement.
Proof. exact FlambdaUntagParamsNamesProofs.untag_params_renamed. Qed.

(* Binding parameters again at the head of their handlers, and giving
   them twins: the lemmas are in FlambdaShadowParamsNamesProofs.v and
   FlambdaTwinParamsNamesProofs.v. *)
Theorem shadow_params_equivariant : shadow_params_equivariant_statement.
Proof. exact FlambdaShadowParamsNamesProofs.shadow_params_renamed. Qed.

Theorem twin_params_equivariant : twin_params_equivariant_statement.
Proof. exact FlambdaTwinParamsNamesProofs.twin_params_renamed. Qed.

(* Taking away a parameter that every application gives one symbol or
   one constant: the lemmas are in FlambdaBindConstParamsNamesProofs.v. *)
Theorem bind_const_params_equivariant :
  bind_const_params_equivariant_statement.
Proof.
  exact FlambdaBindConstParamsNamesProofs.bind_const_params_renamed.
Qed.

(* Taking away a parameter of a loop that every turn passes on: the
   lemmas are in FlambdaUnaliasRecParamsNamesProofs.v. *)
Theorem unalias_rec_params_equivariant :
  unalias_rec_params_equivariant_statement.
Proof.
  exact FlambdaUnaliasRecParamsNamesProofs.unalias_rec_params_renamed.
Qed.

(* Following aliases into the handlers of recursive continuations. The
   tactic is told how a typing environment is renamed, and is given the
   laws of the environment under a [Let] and without the parameters of
   a handler, and that of the pass that follows aliases, which the
   handler of a loop is given to. *)
Section Follow_aliases_in_loops.

#[local] Instance loops_typing_env_renaming : Renaming_of typing_env :=
  rename_naive_typing_env.

#[local] Hint Rewrite typing_env_under_let_equivariant drop_vars_equivariant
  follow_aliases_equivariant_aux : equivariance.

Lemma follow_aliases_in_loops_equivariant (r : renaming) (tenv : typing_env)
    (e : expr) :
  follow_aliases_in_loops (rename_naive_typing_env r tenv) (rename_expr r e)
  = rename_expr r (follow_aliases_in_loops tenv e).
Proof. equivariance. Qed.

End Follow_aliases_in_loops.

Lemma follow_aliases_in_loops_unknowing_equivariant :
  rewrite_equivariant (follow_aliases_in_loops []).
Proof.
  intros r e. exact (follow_aliases_in_loops_equivariant r [] e).
Qed.

(* Loopify. The name given for the loop of each piece of code is renamed
   with the term; code ids are not renamed ([rename_loopified]). The
   laws of [loop_calls] and of [loopify_unit] are by the tactic, which
   is told how the names given are renamed. Three laws that it is given
   are by hand: that of [loopable], because the test that the
   parameters are distinct answers the same of renamed parameters by
   the renaming being injective, and those of [loop_safe] and of
   [looped_code] for the time of compiling. *)
Section Loopify.

#[local] Instance loopified_renaming : Renaming_of loopified :=
  rename_loopified.

#[local] Hint Rewrite no_exn_extra_args_renamed : equivariance.

Lemma loop_calls_equivariant (r : renaming) (my_closure : variable)
    (params_arity : arity) (ret exn self : continuation) (e : expr) :
  loop_calls (rename_variable r my_closure) params_arity
    (rename_continuation r ret) (rename_continuation r exn)
    (rename_continuation r self) (rename_expr r e)
  = rename_expr r (loop_calls my_closure params_arity ret exn self e).
Proof. equivariance. Qed.

Fixpoint loop_safe_equivariant (r : renaming) (my_closure : variable)
    (ret exn self : continuation) (e : expr) :
  loop_safe (rename_variable r my_closure) (rename_continuation r ret)
    (rename_continuation r exn) (rename_continuation r self)
    (rename_expr r e)
  = loop_safe my_closure ret exn self e
with loop_safe_handler_equivariant (r : renaming) (my_closure : variable)
    (ret exn self : continuation) (h : continuation_handler) :
  loop_safe_handler (rename_variable r my_closure)
    (rename_continuation r ret) (rename_continuation r exn)
    (rename_continuation r self) (rename_handler r h)
  = loop_safe_handler my_closure ret exn self h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr loop_safe]; try reflexivity.
    + rewrite rename_variable_eqb, loop_safe_equivariant. reflexivity.
    + apply loop_safe_equivariant.
    + rewrite !rename_continuation_eqb, loop_safe_handler_equivariant,
        loop_safe_equivariant.
      reflexivity.
    + rewrite loop_safe_equivariant. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite !rename_continuation_eqb,
        (loop_safe_handler_equivariant r my_closure ret exn self h), IH.
      reflexivity.
    + cbn [rename_apply ap_continuation ap_exn_continuation].
      rewrite !rename_continuation_eqb. reflexivity.
    + rewrite among_apply_cont_continuations_renamed. reflexivity.
    + apply forallb_map_with. intros arm. cbn [snd].
      rewrite among_apply_cont_continuations_renamed. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite among_rename_variable, loop_safe_equivariant. reflexivity.
Qed.

Lemma looped_code_equivariant (r : renaming) (self : continuation)
    (c : code) :
  looped_code (rename_continuation r self) (rename_code r c)
  = rename_code r (looped_code self c).
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold looped_code, loop_handler, code_params_kinds, code_params_arity.
  simpl. rewrite loop_calls_equivariant. unfold rename_apply_cont. simpl.
  rewrite !map_map. reflexivity.
Qed.

Lemma loopable_equivariant (r : renaming) (self : continuation) (c : code) :
  loopable (rename_continuation r self) (rename_code r c) = loopable self c.
Proof.
  destruct c as [params my_closure ret exn metadata body].
  unfold loopable.
  cbn [rename_code code_params code_my_closure code_return_continuation
       code_exn_continuation code_body code_params_arity code_is_tupled].
  rewrite among_rename_variable,
    (distinct_map_injective _ (rename_variable_eqb r)),
    !rename_continuation_eqb, loop_safe_equivariant, loop_calls_equivariant,
    cont_free_b_equivariant_aux.
  reflexivity.
Qed.

Lemma lookup_rename_loopified (r : renaming) (G : loopified) (id : code_id) :
  lookup id (rename_loopified r G)
  = option_map (rename_continuation r) (lookup id G).
Proof. equivariance. Qed.

#[local] Hint Rewrite looped_code_equivariant loopable_equivariant
  lookup_rename_loopified : equivariance.
#[local] Arguments looped_code : simpl never.
#[local] Arguments loopable : simpl never.

Lemma loopify_unit_equivariant (r : renaming) (G : loopified) (e : expr) :
  loopify_unit (rename_loopified r G) (rename_expr r e)
  = rename_expr r (loopify_unit G e).
Proof. equivariance. Qed.

End Loopify.

End Passes_for_loops.

(* ======================================================================== *)
(* The normalisations of FlambdaValidateSpec.v                              *)
(* ======================================================================== *)

(* The comparison answers the same when both terms are renamed
   (FlambdaCompareAlphaProofs.expr_eqb_equivariant), here with no names
   paired. *)
Lemma expr_eqb_renamed (r : renaming) (e e' : expr) :
  expr_eqb [] [] (rename_expr r e) (rename_expr r e') = expr_eqb [] [] e e'.
Proof. exact (expr_eqb_equivariant r [] [] e e'). Qed.

(* The comparison answers the same of two terms in either order. *)
Lemma expr_eqb_swap (e e' : expr) :
  expr_eqb [] [] e e' = expr_eqb [] [] e' e.
Proof.
  destruct (expr_eqb [] [] e e') eqn:He.
  - symmetry. exact (expr_eqb_symmetric _ _ He).
  - destruct (expr_eqb [] [] e' e) eqn:He'; [ | reflexivity ].
    rewrite (expr_eqb_symmetric _ _ He') in He. discriminate.
Qed.

(* The comparison after which [iterate] stops answers the same of the
   term renamed, so the same number of applications is made. *)
Theorem iterate_equivariant : iterate_equivariant_statement.
Proof.
  intros rounds f f_equivariant.
  induction rounds as [ | rounds IH]; intros r e; cbn [iterate];
    [ reflexivity | ].
  cbv zeta. rewrite f_equivariant, expr_eqb_renamed.
  destruct (expr_eqb [] [] e (f e)); [ reflexivity | apply IH ].
Qed.

(* The rewrites as the validation applies them: with nothing in scope and
   nothing known. *)
Lemma remove_unused_defined_lets_closed_equivariant :
  rewrite_equivariant (remove_unused_defined_lets []).
Proof.
  intros r e. exact (remove_unused_defined_lets_equivariant_aux r e []).
Qed.

Lemma remove_unused_defined_lets_in_loops_closed_equivariant :
  rewrite_equivariant (remove_unused_defined_lets_in_loops []).
Proof.
  intros r e.
  exact (remove_unused_defined_lets_in_loops_equivariant_aux r e []).
Qed.

Theorem remove_unused_defined_lets_in_loops_equivariant :
  remove_unused_defined_lets_in_loops_equivariant_statement.
Proof.
  intros r kinds e. apply remove_unused_defined_lets_in_loops_equivariant_aux.
Qed.

Lemma inline_calls_unknowing_equivariant (kept : list code_id) :
  rewrite_equivariant (inline_calls kept []).
Proof. intros r e. exact (inline_calls_equivariant r kept [] e). Qed.

Lemma inline_known_calls_unknowing_equivariant (kept : list code_id) :
  rewrite_equivariant (inline_known_calls kept [] []).
Proof.
  intros r e. exact (inline_known_calls_equivariant_aux r kept [] e []).
Qed.

(* The unboxing of the parameters of continuations commutes with
   renaming: FlambdaUnboxParamsNamesProofs.v proves it from the
   equations of this file for what is known of blocks and of wrapped
   numbers, which it takes as premises, as this file is to use the
   result. *)
Lemma unbox_variant_params_renamed :
  rewrite_equivariant unbox_variant_params.
Proof.
  apply FlambdaUnboxParamsNamesProofs.unbox_variant_params_equivariant;
    first
      [ exact var_free_b_equivariant
      | exact remove_unused_conts_equivariant
      | exact forget_vars_equivariant
      | exact known_under_let_equivariant
      | exact known_under_statics_equivariant
      | exact known_block_of_equivariant
      | exact field_loaded_equivariant
      | exact load_known_named_equivariant
      | exact forget_wrapped_vars_equivariant
      | exact numbers_under_let_equivariant
      | exact numbers_under_statics_equivariant
      | exact unwrapped_equivariant
      | exact unbox_known_named_equivariant ].
Qed.

(* The renaming is taken out of each rewrite in turn, from the
   innermost. *)
Ltac rename_outwards :=
  repeat
    first
      [ rewrite fold_constants_equivariant
      | rewrite collapse_switches_equivariant_aux
      | rewrite merge_switches_equivariant
      | rewrite remove_unused_conts_equivariant_aux
      | rewrite inline_tail_conts_equivariant_aux
      | rewrite thread_switches_equivariant_aux
      | rewrite load_known_blocks_equivariant
      | rewrite unbox_known_numbers_equivariant
      | rewrite recover_known_comparisons_equivariant
      | rewrite fold_same_comparisons_equivariant
      | rewrite share_known_primitives_equivariant
      | rewrite share_known_projections_equivariant
      | rewrite apply_identities_equivariant
      | rewrite follow_aliases_unknowing_equivariant
      | rewrite remove_unused_lets_equivariant_aux
      | rewrite remove_unused_defined_lets_closed_equivariant
      | rewrite remove_unused_projections_equivariant
      | rewrite load_known_blocks_surely_equivariant
      | rewrite fold_made_lengths_equivariant
      | rewrite FlambdaReadsAfterProofs.remove_constant_allocations_equivariant
      | rewrite inline_calls_unknowing_equivariant
      | rewrite inline_known_calls_unknowing_equivariant
      | rewrite FlambdaKnownSiblingsProofs.inline_known_sibling_calls_renamed
      | rewrite drop_deleted_code_equivariant_aux
      | rewrite merge_statics_equivariant_aux
      | rewrite sink_let_conts_equivariant_aux
      | rewrite shortcut_switches_equivariant_aux
      | rewrite share_handlers_equivariant_aux
      | rewrite follow_aliases_in_loops_unknowing_equivariant
      | rewrite drop_unused_params_equivariant
      | rewrite wrap_returns_equivariant
      | rewrite untag_params_equivariant
      | rewrite drop_dead_params_equivariant
      | rewrite bind_aliased_params_equivariant
      | rewrite forward_applications_equivariant
      | rewrite forward_jumps_equivariant
      | rewrite forward_kinded_equivariant
      | rewrite forward_to_arms_equivariant
      | rewrite forward_to_arms_kinded_equivariant
      | rewrite FlambdaBindConstParamsNamesProofs.bind_const_params_renamed
      | rewrite FlambdaLiftContsNamesProofs.lift_handler_conts_renamed
      | rewrite unbox_variant_params_renamed
      | rewrite remove_unused_defined_lets_in_loops_closed_equivariant
      | rewrite inline_tail_conts_kinded_equivariant_aux
      | rewrite FlambdaUnaliasParamsNamesProofs.unalias_params_renamed
      | rewrite FlambdaUnaliasRecParamsNamesProofs.unalias_rec_params_renamed
      | rewrite FlambdaGiveScrutineeNamesProofs.give_scrutinees_renamed
      | rewrite FlambdaArmNumbersNamesProofs.numbers_in_arms_renamed
      | rewrite FlambdaAliasLetsProofs.merge_alias_lets_renamed ].

Lemma normalise_expr_before_equivariant :
  rewrite_equivariant normalise_expr_before.
Proof.
  intros r e. unfold normalise_expr_before. cbv zeta. rename_outwards.
  reflexivity.
Qed.

Lemma normalise_code_before_equivariant :
  rewrite_equivariant normalise_code_before.
Proof.
  intros r e. unfold normalise_code_before.
  rewrite normalise_expr_before_equivariant.
  apply rewrite_rec_handlers_equivariant, normalise_expr_before_equivariant.
Qed.

Theorem normalise_before_inlined_equivariant :
  normalise_before_inlined_equivariant_statement.
Proof.
  intros r e. unfold normalise_before_inlined.
  rewrite normalise_code_before_equivariant. rename_outwards.
  apply rewrite_code_equivariant, normalise_code_before_equivariant.
Qed.

(* The first phase of the term before. The unboxing given what is
   known of symbols alone commutes with renaming, as a renaming does
   not change what it is given; so does the round given it. *)
Lemma unbox_variants_knowing_symbols_renamed :
  equivariant_where_known unbox_variants_knowing.
Proof.
  intros known Honly r e.
  rewrite <- (symbols_only_renamed r known Honly) at 1.
  apply FlambdaUnboxParamsNamesProofs.unbox_variants_knowing_equivariant;
    first
      [ exact var_free_b_equivariant
      | exact remove_unused_conts_equivariant
      | exact forget_vars_equivariant
      | exact known_under_let_equivariant
      | exact known_under_statics_equivariant
      | exact known_block_of_equivariant
      | exact field_loaded_equivariant
      | exact load_known_named_equivariant
      | exact forget_wrapped_vars_equivariant
      | exact numbers_under_let_equivariant
      | exact numbers_under_statics_equivariant
      | exact unwrapped_equivariant
      | exact unbox_known_named_equivariant ].
Qed.

Lemma unbox_loops_knowing_symbols_renamed :
  equivariant_where_known unbox_loops_knowing.
Proof.
  intros known Honly r e.
  rewrite <- (symbols_only_renamed r known Honly) at 1.
  apply FlambdaUnboxParamsNamesProofs.unbox_loops_knowing_equivariant;
    first
      [ exact var_free_b_equivariant
      | exact remove_unused_conts_equivariant
      | exact forget_vars_equivariant
      | exact known_under_let_equivariant
      | exact known_under_statics_equivariant
      | exact known_block_of_equivariant
      | exact field_loaded_equivariant
      | exact load_known_named_equivariant
      | exact forget_wrapped_vars_equivariant
      | exact numbers_under_let_equivariant
      | exact numbers_under_statics_equivariant
      | exact unwrapped_equivariant
      | exact unbox_known_named_equivariant ].
Qed.

Lemma unbox_together_knowing_symbols_renamed :
  equivariant_where_known unbox_together_knowing.
Proof.
  intros known Honly r e.
  rewrite <- (symbols_only_renamed r known Honly) at 1.
  apply FlambdaUnboxTogetherNamesProofs.unbox_together_knowing_renamed;
    first
      [ exact var_free_b_equivariant
      | exact remove_unused_conts_equivariant
      | exact forget_vars_equivariant
      | exact known_under_let_equivariant
      | exact known_under_statics_equivariant
      | exact known_block_of_equivariant
      | exact field_loaded_equivariant
      | exact load_known_named_equivariant
      | exact forget_wrapped_vars_equivariant
      | exact numbers_under_let_equivariant
      | exact numbers_under_statics_equivariant
      | exact unwrapped_equivariant
      | exact unbox_known_named_equivariant ].
Qed.

Lemma normalise_expr_params_knowing_equivariant :
  equivariant_where_known normalise_expr_params_knowing.
Proof.
  intros known Honly r e. unfold normalise_expr_params_knowing. cbv zeta.
  rename_outwards.
  rewrite (unbox_variants_knowing_symbols_renamed known Honly),
    (unbox_loops_knowing_symbols_renamed known Honly).
  rename_outwards.
  rewrite (unbox_together_knowing_symbols_renamed known Honly).
  rename_outwards. reflexivity.
Qed.

Lemma normalise_expr_params_equivariant :
  rewrite_equivariant normalise_expr_params.
Proof.
  exact (normalise_expr_params_knowing_equivariant no_facts no_facts_only).
Qed.

Lemma normalise_code_params_knowing_equivariant :
  equivariant_where_known normalise_code_params_knowing.
Proof.
  intros known Honly r e. unfold normalise_code_params_knowing.
  rewrite
    (rewrite_rec_handlers_in_scope_equivariant _
       normalise_expr_params_knowing_equivariant known Honly).
  apply normalise_expr_params_knowing_equivariant, Honly.
Qed.

Lemma normalise_code_params_equivariant :
  rewrite_equivariant normalise_code_params.
Proof.
  exact (normalise_code_params_knowing_equivariant no_facts no_facts_only).
Qed.

Lemma normalise_params_round_equivariant :
  rewrite_equivariant normalise_params_round.
Proof.
  intros r e. unfold normalise_params_round.
  rewrite merge_tables_equivariant, normalise_code_params_equivariant.
  exact (rewrite_code_in_scope_equivariant _
    normalise_code_params_knowing_equivariant no_facts no_facts_only r _).
Qed.

(* What follows that round in the first phase, and the two together. *)
Lemma arms_expr_before_equivariant : rewrite_equivariant arms_expr_before.
Proof.
  intros r e. unfold arms_expr_before. rename_outwards. reflexivity.
Qed.

Lemma arms_code_before_equivariant : rewrite_equivariant arms_code_before.
Proof.
  intros r e. unfold arms_code_before.
  rewrite
    (rewrite_rec_handlers_equivariant arms_expr_before
       arms_expr_before_equivariant).
  apply arms_expr_before_equivariant.
Qed.

Theorem first_phase_round_equivariant :
  first_phase_round_equivariant_statement.
Proof.
  intros r e. unfold first_phase_round.
  rewrite normalise_params_round_equivariant, arms_code_before_equivariant.
  apply rewrite_code_equivariant, arms_code_before_equivariant.
Qed.

(* The demotion of exception handlers, as the validation applies it:
   with no name in scope; and with the rounds that follow it. *)
Lemma demote_handlers_closed_equivariant :
  rewrite_equivariant (demote_handlers []).
Proof.
  intros r e.
  exact (FlambdaDemoteProofs.demote_handlers_equivariant r [] e).
Qed.

Theorem demote_and_tidy_equivariant : demote_and_tidy_equivariant_statement.
Proof.
  intros rounds r e. unfold demote_and_tidy.
  rewrite demote_handlers_closed_equivariant.
  rewrite
    (iterate_equivariant rounds first_phase_round
       first_phase_round_equivariant).
  exact
    (iterate_equivariant rounds normalise_before_inlined
       normalise_before_inlined_equivariant r _).
Qed.

Lemma normalise_expr_after_equivariant :
  rewrite_equivariant normalise_expr_after.
Proof.
  intros r e. unfold normalise_expr_after, inline_once_applied_after.
  rename_outwards. reflexivity.
Qed.

Lemma normalise_code_after_equivariant :
  rewrite_equivariant normalise_code_after.
Proof.
  intros r e. unfold normalise_code_after.
  rewrite normalise_expr_after_equivariant.
  apply rewrite_rec_handlers_equivariant, normalise_expr_after_equivariant.
Qed.

Lemma normalise_after_round_equivariant (kept : list code_id) :
  rewrite_equivariant (normalise_after_round kept).
Proof.
  intros r e. unfold normalise_after_round, inline_once_applied_after.
  rename_outwards.
  rewrite
    (rewrite_rec_handlers_equivariant normalise_expr_after
       normalise_expr_after_equivariant).
  rename_outwards.
  rewrite (rewrite_code_equivariant _ normalise_code_after_equivariant),
    forward_kinded_in_code_equivariant, forward_to_arms_in_code_equivariant.
  apply remove_unused_defined_lets_in_code_equivariant.
Qed.

Theorem normalise_after_equivariant : normalise_after_equivariant_statement.
Proof.
  intros rounds kept.
  apply iterate_equivariant, normalise_after_round_equivariant.
Qed.

(* The placing of continuations: the pass in the term, in the handlers
   of its recursive continuations and in its code. *)
Lemma place_conts_equivariant : rewrite_equivariant place_conts.
Proof.
  intros r e. unfold place_conts.
  rewrite sink_conts_to_uses_equivariant_aux.
  apply
    (rewrite_rec_handlers_equivariant sink_conts_to_uses
       sink_conts_to_uses_equivariant).
Qed.

Theorem placed_equivariant : placed_equivariant_statement.
Proof.
  intros r e. unfold placed. rewrite place_conts_equivariant.
  apply (rewrite_code_equivariant _ place_conts_equivariant).
Qed.

(* The tidying that follows the placing: a round for an expression,
   for the body of code and for a term, of the term before and of the
   term after. *)
Lemma tidy_expr_before_equivariant : rewrite_equivariant tidy_expr_before.
Proof.
  intros r e. unfold tidy_expr_before. rename_outwards. reflexivity.
Qed.

Lemma tidy_code_before_equivariant : rewrite_equivariant tidy_code_before.
Proof.
  intros r e. unfold tidy_code_before.
  rewrite tidy_expr_before_equivariant.
  apply rewrite_rec_handlers_equivariant, tidy_expr_before_equivariant.
Qed.

Lemma tidy_before_round_equivariant : rewrite_equivariant tidy_before_round.
Proof.
  intros r e. unfold tidy_before_round.
  rewrite tidy_code_before_equivariant.
  apply rewrite_code_equivariant, tidy_code_before_equivariant.
Qed.

Theorem tidied_before_equivariant : tidied_before_equivariant_statement.
Proof.
  intros rounds r e. unfold tidied_before. rewrite placed_equivariant.
  rewrite
    (iterate_equivariant rounds tidy_before_round
       tidy_before_round_equivariant r _).
  symmetry. apply groups_up_equivariant.
Qed.

Lemma tidy_expr_after_equivariant : rewrite_equivariant tidy_expr_after.
Proof.
  intros r e. unfold tidy_expr_after. rename_outwards. reflexivity.
Qed.

Lemma tidy_code_after_equivariant : rewrite_equivariant tidy_code_after.
Proof.
  intros r e. unfold tidy_code_after.
  rewrite tidy_expr_after_equivariant.
  apply rewrite_rec_handlers_equivariant, tidy_expr_after_equivariant.
Qed.

Lemma tidy_after_round_equivariant : rewrite_equivariant tidy_after_round.
Proof.
  intros r e. unfold tidy_after_round.
  rewrite tidy_code_after_equivariant.
  rewrite (rewrite_code_equivariant _ tidy_code_after_equivariant).
  apply remove_unused_defined_lets_in_code_equivariant.
Qed.

Theorem tidied_after_equivariant : tidied_after_equivariant_statement.
Proof.
  intros rounds r e. unfold tidied_after. rewrite placed_equivariant.
  rewrite
    (iterate_equivariant rounds tidy_after_round
       tidy_after_round_equivariant r _).
  symmetry. apply groups_up_equivariant.
Qed.

Theorem without_statics_equivariant : without_statics_equivariant_statement.
Proof.
  intros D r e. unfold without_statics. rewrite drop_statics_equivariant.
  destruct (drop_statics D e) as [e' | ]; cbn [option_map]; [ | reflexivity ].
  rewrite drop_deleted_code_equivariant_aux, merge_statics_equivariant_aux.
  reflexivity.
Qed.

(* ======================================================================== *)
(* FlambdaWellKindedSpec.v                                                  *)
(* ======================================================================== *)

(* The laws of renaming of the check of kinds are proved from the
   definitions by the tactic of FlambdaEquivarianceTactic.v. It is told
   how what is in scope is renamed, since the types do not say it (the
   kinds of the variables and the arities of the continuations are maps
   keyed by positives): the actions are [rename_kind_env] and
   [rename_arity_env]. What is stated for it besides is what a lookup
   answers at a renamed name in a map whose keys are renamed, which is
   proved by hand; and each law of the section is given to the tactic
   for the proofs after it. *)
Section Well_kinded.

#[local] Instance kind_env_renaming : Renaming_of kind_env :=
  rename_kind_env.
#[local] Instance arity_env_renaming : Renaming_of arity_env :=
  rename_arity_env.

(* A map whose keys are renamed, and what they are bound to kept. *)
Lemma lookup_map_keys_injective {A : Type} (f : positive -> positive)
    (x : positive) (l : finite_map A) :
  (∀ a b : positive, Pos.eqb (f a) (f b) = Pos.eqb a b) ->
  lookup (f x) (Keyed.map_keys f l) = lookup x l.
Proof.
  intros Hf. unfold Keyed.map_keys, Keyed.map_bindings.
  induction l as [ | [y a] l IH]; simpl; [ reflexivity | ].
  rewrite Hf. destruct (Pos.eqb x y); [ reflexivity | exact IH ].
Qed.

Lemma is_bound_map_keys_injective {A : Type} (f : positive -> positive)
    (x : positive) (l : finite_map A) :
  (∀ a b : positive, Pos.eqb (f a) (f b) = Pos.eqb a b) ->
  is_bound (f x) (Keyed.map_keys f l) = is_bound x l.
Proof.
  intros Hf. unfold is_bound. rewrite (lookup_map_keys_injective _ _ _ Hf).
  reflexivity.
Qed.

Lemma lookup_rename_kind_env (r : renaming) (x : variable)
    (kinds : kind_env) :
  lookup (rename_variable r x) (rename_kind_env r kinds) = lookup x kinds.
Proof. apply lookup_map_keys_injective, rename_variable_eqb. Qed.

Lemma lookup_rename_arity_env (r : renaming) (k : continuation)
    (conts : arity_env) :
  lookup (rename_continuation r k) (rename_arity_env r conts)
  = lookup k conts.
Proof. apply lookup_map_keys_injective, rename_continuation_eqb. Qed.

Lemma is_bound_rename_arity_env (r : renaming) (k : continuation)
    (conts : arity_env) :
  is_bound (rename_continuation r k) (rename_arity_env r conts)
  = is_bound k conts.
Proof. apply is_bound_map_keys_injective, rename_continuation_eqb. Qed.
#[local] Hint Rewrite lookup_rename_kind_env lookup_rename_arity_env
  is_bound_rename_arity_env : equivariance.

Theorem handler_kinds_equivariant : handler_kinds_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite handler_kinds_equivariant : equivariance.

Lemma kind_of_simple_renamed (r : renaming) (kinds : kind_env) (s : simple) :
  kind_of_simple (rename_kind_env r kinds) (rename_simple r s)
  = kind_of_simple kinds s.
Proof. equivariance. Qed.
#[local] Hint Rewrite kind_of_simple_renamed : equivariance.

Theorem kind_of_simple_equivariant : kind_of_simple_equivariant_statement.
Proof. exact kind_of_simple_renamed. Qed.

Theorem kinds_of_simples_equivariant : kinds_of_simples_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite kinds_of_simples_equivariant : equivariance.

Theorem kind_of_named_equivariant : kind_of_named_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite kind_of_named_equivariant : equivariance.

Theorem bind_param_kinds_equivariant : bind_param_kinds_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite bind_param_kinds_equivariant : equivariance.

Theorem static_const_well_kinded_equivariant :
  static_const_well_kinded_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite static_const_well_kinded_equivariant : equivariance.

Theorem apply_cont_well_kinded_equivariant :
  apply_cont_well_kinded_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite apply_cont_well_kinded_equivariant : equivariance.

Theorem apply_well_kinded_equivariant :
  apply_well_kinded_equivariant_statement.
Proof. equivariance. Qed.
#[local] Hint Rewrite apply_well_kinded_equivariant : equivariance.

(* For the check of a term: the parameters are distinct, the arities of
   the handlers of a recursive [Let_cont] are put before those in
   scope, and the parameters of code are bound in a map that holds
   [my_closure] alone. The three laws are of the forms that the check
   has there. *)
Lemma distinct_rename_variables (r : renaming) (params : list variable) :
  distinct (map (rename_variable r) params) = distinct params.
Proof. exact (distinct_map_injective _ (rename_variable_eqb r) params). Qed.

Lemma handler_arities_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler))
    (conts : arity_env) :
  handler_arities
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
  ++ rename_arity_env r conts
  = rename_arity_env r (handler_arities handlers ++ conts).
Proof.
  unfold handler_arities, rename_arity_env, Keyed.map_keys,
    Keyed.map_bindings, map_values.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite handler_kinds_equivariant, IH. reflexivity.
Qed.

Lemma bind_param_kinds_of_code (r : renaming) (params : list variable)
    (params_kinds : list flambda_kind) (x : variable) (kind : flambda_kind) :
  bind_param_kinds (map (rename_variable r) params) params_kinds
    [(rename_variable r x, kind)]
  = option_map (rename_kind_env r)
      (bind_param_kinds params params_kinds [(x, kind)]).
Proof.
  exact (bind_param_kinds_equivariant r params params_kinds [(x, kind)]).
Qed.
#[local] Hint Rewrite distinct_rename_variables handler_arities_renamed
  bind_param_kinds_of_code : equivariance.
#[local] Arguments handler_arities : simpl never.

Theorem well_kinded_equivariant : well_kinded_equivariant_statement.
Proof. equivariance. Qed.

Theorem well_kinded_unit_equivariant : well_kinded_unit_equivariant_statement.
Proof.
  intros r slots unit_return unit_exn e. unfold well_kinded_unit.
  exact
    (well_kinded_equivariant r slots [] (unit_arities unit_return unit_exn) e).
Qed.

End Well_kinded.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The lifting: its rewrites and its checks commute with renaming by
   the statements of their own files. *)
Lemma only_closures_of_renamed (r : renaming)
    (known : list (code_id * symbol)) (e : expr) :
  only_closures_of known (rename_expr r e) = only_closures_of known e.
Proof.
  unfold only_closures_of.
  induction known as [ | [id s] known IH]; simpl; [ reflexivity | ].
  rewrite only_closure_of_equivariant, IH. reflexivity.
Qed.

Lemma lift_unit_renamed (r : renaming) (L : lifting) (e : expr) :
  lift_unit (rename_lifting r L) (rename_expr r e)
  = option_map (rename_expr r) (lift_unit L e).
Proof.
  unfold lift_unit, rename_lifting.
  cbn [lifting_unit lifted_lets lifted_my_closures lifted_out_of_code
       lifted_to_head].
  cbv zeta.
  rewrite <- lift_expr_equivariant, <- bind_my_closures_equivariant.
  rewrite well_formed_unit_equivariant, only_closures_of_renamed.
  pose proof (exits_plain_knowing_equivariant r [] [] e) as Hexits.
  simpl in Hexits. rewrite Hexits.
  rewrite (FlambdaHeapFreeProofs.expr_free_of_heap_rename r e).
  destruct
    (exits_plain_knowing [] [] e
     && FlambdaHeapFreeSpec.expr_free_of_heap e
     && well_formed_unit (lifting_unit L)
          (bind_my_closures (lifted_my_closures L)
             (lift_expr (lifted_lets L) root e))
     && only_closures_of (lifted_my_closures L)
          (bind_my_closures (lifted_my_closures L)
             (lift_expr (lifted_lets L) root e)));
    reflexivity.
Qed.

(* The lifting out of code: the checks answer the same on the renamed
   unit, and the rewrite commutes with the renaming. *)
Lemma lift_from_code_unit_renamed (r : renaming) (u : unit_interface)
    (K : lets_in_code) (e : expr) :
  lift_from_code_unit (rename_unit_interface r u) K (rename_expr r e)
  = option_map (rename_expr r) (lift_from_code_unit u K e).
Proof.
  unfold lift_from_code_unit. cbv zeta.
  rewrite <- lift_from_code_equivariant, well_formed_unit_equivariant.
  pose proof (exits_plain_knowing_equivariant r [] [] e) as Hexits.
  simpl in Hexits. rewrite Hexits.
  rewrite (FlambdaHeapFreeProofs.expr_free_of_heap_rename r e).
  destruct
    (exits_plain_knowing [] [] e
     && FlambdaHeapFreeSpec.expr_free_of_heap e
     && well_formed_unit u (lift_from_code K [] root e));
    reflexivity.
Qed.

(* The lifting to the head of the unit, in the same way. *)
Lemma lift_to_head_unit_renamed (r : renaming) (u : unit_interface)
    (lets : list (path * lift_to)) (e : expr) :
  lift_to_head_unit (rename_unit_interface r u) lets (rename_expr r e)
  = option_map (rename_expr r) (lift_to_head_unit u lets e).
Proof.
  unfold lift_to_head_unit. cbv zeta.
  rewrite <- FlambdaLiftFromArmsProofs.lift_to_head_equivariant,
    well_formed_unit_equivariant.
  pose proof (exits_plain_knowing_equivariant r [] [] e) as Hexits.
  simpl in Hexits. rewrite Hexits.
  rewrite (FlambdaHeapFreeProofs.expr_free_of_heap_rename r e).
  destruct
    (exits_plain_knowing [] [] e
     && FlambdaHeapFreeSpec.expr_free_of_heap e
     && well_formed_unit u (lift_to_head lets e));
    reflexivity.
Qed.

(* The lifting to a group around: the groups are read off the unit, so
   those of the renamed unit are those of the unit, renamed; the check
   answers the same, and the rewrite commutes with the renaming. *)
Lemma lift_around_unit_renamed (r : renaming) (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e : expr) :
  lift_around_unit (rename_unit_interface r u) lets places (rename_expr r e)
  = option_map (rename_expr r) (lift_around_unit u lets places e).
Proof.
  unfold lift_around_unit. cbv zeta.
  rewrite FlambdaLiftAroundProofs.groups_of_equivariant,
    <- FlambdaLiftAroundProofs.lift_around_equivariant,
    well_formed_unit_equivariant.
  pose proof
    (FlambdaLiftAroundProofs.around_ok_equivariant r lets
       (groups_of lets places e) [] root e)
    as Hok.
  cbn [rename_known_constants map] in Hok. rewrite Hok.
  pose proof (exits_plain_knowing_equivariant r [] [] e) as Hexits.
  simpl in Hexits. rewrite Hexits.
  rewrite (FlambdaHeapFreeProofs.expr_free_of_heap_rename r e).
  destruct
    (exits_plain_knowing [] [] e
     && FlambdaHeapFreeSpec.expr_free_of_heap e
     && around_ok lets (groups_of lets places e) [] root e
     && well_formed_unit u
          (lift_around lets (groups_of lets places e) root e));
    reflexivity.
Qed.

(* The tables bound at the head of a unit
   (FlambdaUnusedHeadStaticsSpec.add_head_tables): a table whose
   elements are constants and symbols is not changed by a renaming,
   and the checks are of names that no renaming moves. *)
Lemma plain_simples_renamed (r : renaming) (elements : list simple) :
  forallb plain_simple elements = true ->
  map (rename_simple r) elements = elements.
Proof.
  induction elements as [ | s elements IH]; [ reflexivity | ].
  cbn [forallb map]. intros Hplain.
  apply andb_true_iff in Hplain. destruct Hplain as [Hs Hplain].
  rewrite (IH Hplain). destruct s; [ discriminate | reflexivity | reflexivity ].
Qed.

Lemma tables_renamed (r : renaming) (tables : list (symbol * list simple)) :
  tables_plain tables = true ->
  map (rename_static_binding r) (map table_binding tables)
  = map table_binding tables.
Proof.
  unfold tables_plain.
  induction tables as [ | [s elements] tables IH]; [ reflexivity | ].
  cbn [forallb snd map]. intros Hplain.
  apply andb_true_iff in Hplain. destruct Hplain as [Helements Hplain].
  rewrite (IH Hplain). unfold table_binding at 1 3. cbn [fst snd].
  cbn [rename_static_binding rename_static_const].
  rewrite (plain_simples_renamed r elements Helements). reflexivity.
Qed.

Lemma add_head_tables_renamed (r : renaming)
    (tables : list (symbol * list simple)) (e : expr) :
  add_head_tables tables (rename_expr r e)
  = rename_expr r (add_head_tables tables e).
Proof.
  unfold add_head_tables.
  destruct tables as [ | table tables]; [ reflexivity | ].
  cbv zeta.
  destruct (tables_plain (table :: tables)) eqn:Hplain; cbn [andb];
    [ | reflexivity ].
  assert
    (Hrenamed :
      Let_static (map table_binding (table :: tables)) (rename_expr r e)
      = rename_expr r (Let_static (map table_binding (table :: tables)) e)).
  { cbn [rename_expr]. rewrite (tables_renamed r _ Hplain). reflexivity. }
  rewrite Hrenamed, head_statics_removable_renamed.
  destruct
    (head_statics_removable (tables_dropped (table :: tables))
       (Let_static (map table_binding (table :: tables)) e));
    reflexivity.
Qed.

Lemma with_lifting_renamed (r : renaming) (L : lifting) (e : expr) :
  with_lifting (rename_lifting r L) (rename_expr r e)
  = rename_expr r (with_lifting L e).
Proof.
  unfold with_lifting. cbv zeta.
  change (code_copied (rename_lifting r L)) with (code_copied L).
  rewrite FlambdaCopyCodeProofs.with_copies_equivariant. f_equal.
  change (tables_put (rename_lifting r L)) with (tables_put L).
  rewrite <- add_head_tables_renamed. f_equal.
  change (lifting_unit (rename_lifting r L))
    with (rename_unit_interface r (lifting_unit L)).
  change (lifted_out_of_code (rename_lifting r L))
    with (lifted_out_of_code L).
  change (lifted_to_head (rename_lifting r L)) with (lifted_to_head L).
  change (lifted_around (rename_lifting r L)) with (lifted_around L).
  change (groups_put (rename_lifting r L)) with (groups_put L).
  rewrite lift_to_head_unit_renamed.
  destruct (lift_to_head_unit (lifting_unit L) (lifted_to_head L) e)
    as [at_head | ];
    cbn [option_map]; rewrite lift_from_code_unit_renamed;
    destruct (lift_from_code_unit (lifting_unit L) (lifted_out_of_code L) _)
      as [e' | ];
    cbn [option_map]; rewrite lift_unit_renamed;
    destruct (lift_unit L _) as [lifted | ];
    cbn [option_map]; rewrite lift_around_unit_renamed;
    destruct
      (lift_around_unit (lifting_unit L) (lifted_around L) (groups_put L) _);
    reflexivity.
Qed.

Theorem with_lifting_equivariant : with_lifting_equivariant_statement.
Proof. exact with_lifting_renamed. Qed.

(* The four liftings without the two tests of the unit
   (FlambdaLiftObservedSpec.v), as the four with them: the other checks
   answer the same on the renamed unit, and each rewrite commutes with
   the renaming. *)
Lemma lift_unit_observed_renamed (r : renaming) (L : lifting) (e : expr) :
  lift_unit_observed (rename_lifting r L) (rename_expr r e)
  = option_map (rename_expr r) (lift_unit_observed L e).
Proof.
  unfold lift_unit_observed, rename_lifting.
  cbn [lifting_unit lifted_lets lifted_my_closures lifted_out_of_code
       lifted_to_head].
  cbv zeta.
  rewrite <- lift_expr_equivariant, <- bind_my_closures_equivariant.
  rewrite well_formed_unit_equivariant, only_closures_of_renamed.
  match goal with
  | |- (if ?c then _ else _) = _ => destruct c
  end;
    reflexivity.
Qed.

Lemma lift_from_code_unit_observed_renamed (r : renaming)
    (u : unit_interface) (K : lets_in_code) (e : expr) :
  lift_from_code_unit_observed (rename_unit_interface r u) K
    (rename_expr r e)
  = option_map (rename_expr r) (lift_from_code_unit_observed u K e).
Proof.
  unfold lift_from_code_unit_observed. cbv zeta.
  rewrite <- lift_from_code_equivariant, well_formed_unit_equivariant.
  destruct (well_formed_unit u (lift_from_code K [] root e)); reflexivity.
Qed.

Lemma lift_to_head_unit_observed_renamed (r : renaming) (u : unit_interface)
    (lets : list (path * lift_to)) (e : expr) :
  lift_to_head_unit_observed (rename_unit_interface r u) lets
    (rename_expr r e)
  = option_map (rename_expr r) (lift_to_head_unit_observed u lets e).
Proof.
  unfold lift_to_head_unit_observed. cbv zeta.
  rewrite <- FlambdaLiftFromArmsProofs.lift_to_head_equivariant,
    well_formed_unit_equivariant.
  destruct (well_formed_unit u (lift_to_head lets e)); reflexivity.
Qed.

Lemma lift_around_unit_observed_renamed (r : renaming) (u : unit_interface)
    (lets : list (path * lift_to)) (places : list (path * list path))
    (e : expr) :
  lift_around_unit_observed (rename_unit_interface r u) lets places
    (rename_expr r e)
  = option_map (rename_expr r) (lift_around_unit_observed u lets places e).
Proof.
  unfold lift_around_unit_observed. cbv zeta.
  rewrite FlambdaLiftAroundProofs.groups_of_equivariant,
    <- FlambdaLiftAroundProofs.lift_around_equivariant,
    well_formed_unit_equivariant.
  pose proof
    (FlambdaLiftAroundProofs.around_ok_equivariant r lets
       (groups_of lets places e) [] root e)
    as Hok.
  cbn [rename_known_constants map] in Hok. rewrite Hok.
  destruct
    (around_ok lets (groups_of lets places e) [] root e
     && well_formed_unit u
          (lift_around lets (groups_of lets places e) root e));
    reflexivity.
Qed.

Lemma with_lifting_observed_renamed (r : renaming) (L : lifting) (e : expr) :
  with_lifting_observed (rename_lifting r L) (rename_expr r e)
  = rename_expr r (with_lifting_observed L e).
Proof.
  unfold with_lifting_observed, liftings_observed, around_observed,
    lets_observed, out_of_code_observed, to_head_observed.
  change (code_copied (rename_lifting r L)) with (code_copied L).
  rewrite FlambdaCopyCodeProofs.with_copies_equivariant. f_equal.
  change (tables_put (rename_lifting r L)) with (tables_put L).
  rewrite <- add_head_tables_renamed. f_equal.
  change (lifting_unit (rename_lifting r L))
    with (rename_unit_interface r (lifting_unit L)).
  change (lifted_out_of_code (rename_lifting r L))
    with (lifted_out_of_code L).
  change (lifted_to_head (rename_lifting r L)) with (lifted_to_head L).
  change (lifted_around (rename_lifting r L)) with (lifted_around L).
  change (groups_put (rename_lifting r L)) with (groups_put L).
  rewrite lift_to_head_unit_observed_renamed.
  destruct
    (lift_to_head_unit_observed (lifting_unit L) (lifted_to_head L) e)
    as [at_head | ];
    cbn [option_map]; rewrite lift_from_code_unit_observed_renamed;
    destruct
      (lift_from_code_unit_observed (lifting_unit L) (lifted_out_of_code L)
         _)
      as [e' | ];
    cbn [option_map]; rewrite lift_unit_observed_renamed;
    destruct (lift_unit_observed L _) as [lifted | ];
    cbn [option_map]; rewrite lift_around_unit_observed_renamed;
    destruct
      (lift_around_unit_observed (lifting_unit L) (lifted_around L)
         (groups_put L) _);
    reflexivity.
Qed.

Theorem with_lifting_observed_equivariant :
  with_lifting_observed_equivariant_statement.
Proof. exact with_lifting_observed_renamed. Qed.

(* The removal of unused value slots. *)
Lemma without_slots_equivariant (slots : list value_slot) :
  rewrite_equivariant (without_slots slots).
Proof.
  intros r e. unfold without_slots, drop_unused_slots.
  rewrite slots_unused_equivariant, <- drop_slots_equivariant.
  destruct (slots_unused slots e); [ | reflexivity ].
  rewrite remove_unused_lets_equivariant.
  exact (rewrite_code_equivariant remove_unused_lets
           remove_unused_lets_equivariant r _).
Qed.

(* The folding on known symbols. What is known has no variable in it,
   so a renaming leaves it as it is. *)
Lemma rename_closed_simple (r : renaming) (s : simple) :
  closed_simple s = true -> rename_simple r s = s.
Proof. destruct s; simpl; [ discriminate | reflexivity | reflexivity ]. Qed.

Lemma closed_simple_renamed (r : renaming) (s : simple) :
  closed_simple (rename_simple r s) = closed_simple s.
Proof. destruct s; reflexivity. Qed.

Lemma rename_closed_value_slots (r : renaming)
    (slots : list (value_slot * simple)) :
  forallb (fun slot => closed_simple (snd slot)) slots = true ->
  map (fun slot => (fst slot, rename_simple r (snd slot))) slots = slots.
Proof.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs Hslots].
  rewrite (rename_closed_simple r s Hs), (IH Hslots). reflexivity.
Qed.

Lemma closed_value_slots_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  forallb (fun slot => closed_simple (snd slot))
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = forallb (fun slot => closed_simple (snd slot)) slots.
Proof.
  induction slots as [ | [slot s] slots IH]; simpl; [ reflexivity | ].
  rewrite closed_simple_renamed, IH. reflexivity.
Qed.

Lemma rename_closed_simples (r : renaming) (args : list simple) :
  forallb closed_simple args = true -> map (rename_simple r) args = args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  intros H. apply andb_true_iff in H as [Hs Hargs].
  rewrite (rename_closed_simple r s Hs), (IH Hargs). reflexivity.
Qed.

Lemma closed_simples_renamed (r : renaming) (args : list simple) :
  forallb closed_simple (map (rename_simple r) args)
  = forallb closed_simple args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite closed_simple_renamed, IH. reflexivity.
Qed.

Lemma rename_closed_static_const (r : renaming) (const : static_const) :
  closed_static_const const = true -> rename_static_const r const = const.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl; intros Hclosed.
  - rewrite (rename_closed_value_slots r slots Hclosed). reflexivity.
  - rewrite (rename_closed_value_slots r slots Hclosed). reflexivity.
  - rewrite (rename_closed_simples r fields Hclosed). reflexivity.
  - rewrite (rename_closed_simple r contents Hclosed). reflexivity.
  - reflexivity.
  - rewrite (rename_closed_simples r elements Hclosed). reflexivity.
Qed.

Lemma closed_static_const_renamed (r : renaming) (const : static_const) :
  closed_static_const (rename_static_const r const)
  = closed_static_const const.
Proof.
  destruct const as [code slots | fslot syms decls slots | tag fields
    | contents | bytes | elements];
    simpl.
  - apply closed_value_slots_renamed.
  - apply closed_value_slots_renamed.
  - apply closed_simples_renamed.
  - apply closed_simple_renamed.
  - reflexivity.
  - apply closed_simples_renamed.
Qed.

Lemma known_of_binding_renamed (r : renaming) (binding : static_binding) :
  known_of_binding (rename_static_binding r binding)
  = known_of_binding binding.
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  cbn [rename_static_binding known_of_binding].
  rewrite closed_static_const_renamed.
  destruct (closed_static_const const) eqn:Hclosed; [ | reflexivity ].
  rewrite (rename_closed_static_const r const Hclosed). reflexivity.
Qed.

Fixpoint known_bindings_renamed (r : renaming) (e : expr) {struct e} :
  known_bindings (rename_expr r e) = known_bindings e
with known_bindings_of_handler_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  known_bindings_of_handler (rename_handler r h)
  = known_bindings_of_handler h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply known_bindings_renamed.
    + rewrite known_bindings_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite known_of_binding_renamed, IH. reflexivity.
    + rewrite known_bindings_of_handler_renamed, known_bindings_renamed.
      reflexivity.
    + rewrite known_bindings_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite known_bindings_of_handler_renamed, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply known_bindings_renamed.
Qed.

Lemma set_of_binding_renamed (r : renaming) (binding : static_binding) :
  set_of_binding (rename_static_binding r binding) = set_of_binding binding.
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  destruct const
    as [code slots | fslot syms decls slots | tag fields | contents | bytes
        | elements];
    try reflexivity.
  simpl. rewrite map_map.
  rewrite
    (map_ext _
       (fun field =>
          if closed_simple field then field else Const unknown_field));
    [ reflexivity | ].
  intros [x | s | c]; reflexivity.
Qed.

Fixpoint known_sets_renamed (r : renaming) (e : expr) {struct e} :
  FlambdaKnownSymbolsSpec.known_sets (rename_expr r e)
  = FlambdaKnownSymbolsSpec.known_sets e
with known_sets_of_handler_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  known_sets_of_handler (rename_handler r h) = known_sets_of_handler h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply known_sets_renamed.
    + rewrite known_sets_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
      rewrite set_of_binding_renamed, IH. reflexivity.
    + rewrite known_sets_of_handler_renamed, known_sets_renamed.
      reflexivity.
    + rewrite known_sets_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite known_sets_of_handler_renamed, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply known_sets_renamed.
Qed.

Lemma binding_bound_as_renamed (r : renaming)
    (known sets : list (symbol * static_const)) (binding : static_binding) :
  binding_bound_as known sets (rename_static_binding r binding)
  = binding_bound_as known sets binding.
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  cbn [rename_static_binding binding_bound_as]. f_equal.
  - destruct (lookup sym known) as [expected | ]; [ | reflexivity ].
    rewrite closed_static_const_renamed.
    destruct (closed_static_const const) eqn:Hclosed; [ | reflexivity ].
    rewrite (rename_closed_static_const r const Hclosed). reflexivity.
  - destruct (lookup sym sets) as [expected | ]; [ | reflexivity ].
    change (Bind_symbol sym (rename_static_const r const))
      with (rename_static_binding r (Bind_symbol sym const)).
    rewrite set_of_binding_renamed. reflexivity.
Qed.

Fixpoint bound_as_renamed (r : renaming)
    (known sets : list (symbol * static_const)) (e : expr) {struct e} :
  bound_as known sets (rename_expr r e) = bound_as known sets e
with handler_bound_as_renamed (r : renaming)
    (known sets : list (symbol * static_const)) (h : continuation_handler)
    {struct h} :
  handler_bound_as known sets (rename_handler r h)
  = handler_bound_as known sets h
with code_of_binding_bound_as_renamed (r : renaming)
    (known sets : list (symbol * static_const)) (binding : static_binding)
    {struct binding} :
  code_of_binding_bound_as known sets (rename_static_binding r binding)
  = code_of_binding_bound_as known sets binding
with code_bound_as_renamed (r : renaming)
    (known sets : list (symbol * static_const)) (c : code) {struct c} :
  code_bound_as known sets (rename_code r c) = code_bound_as known sets c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply bound_as_renamed.
    + rewrite bound_as_renamed. f_equal. f_equal.
      * induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite binding_bound_as_renamed, IH. reflexivity.
      * induction bindings as [ | binding bindings IH]; simpl;
          [ reflexivity | ].
        rewrite code_of_binding_bound_as_renamed, IH. reflexivity.
    + rewrite handler_bound_as_renamed, bound_as_renamed. reflexivity.
    + rewrite bound_as_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite handler_bound_as_renamed, IH. reflexivity.
  - destruct h as [params metadata handler]. simpl. apply bound_as_renamed.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    apply code_bound_as_renamed.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply bound_as_renamed.
Qed.

Lemma with_known_equivariant : rewrite_equivariant with_known.
Proof.
  intros r e. unfold with_known, fold_known_unit. cbv zeta.
  rewrite known_bindings_renamed, known_sets_renamed, bound_as_renamed.
  destruct (bound_as (known_bindings e) _ e); [ | reflexivity ].
  symmetry. apply fold_known_equivariant.
Qed.

(* In this part the laws of the tests of one name or constant, of the
   continuations that a unit binds and of the interface continuations
   are proved from the definitions by [equivariance]
   (FlambdaEquivarianceTactic.v); six of them are given to the tactic,
   as hints of the Section, for the laws after them. The rest is by
   hand: the two laws that turn round the theorem of a pass; the laws
   of the unit with its code first and with its static bindings first,
   which rewrite with laws of other files; and the check of the static
   names in scope, of which the tactic leaves goals. *)
Section Code_and_statics_first.

(* The use of the static closures in sight, and the moving up of the
   bindings of code. *)
Lemma use_known_closures_rewrite_equivariant :
  rewrite_equivariant use_known_closures.
Proof. intros r e. symmetry. apply use_known_closures_equivariant. Qed.

(* The unit with its code first: the checks answer the same on the
   renamed unit, for the free continuation renamed, and the two parts
   commute with the renaming. *)
Lemma hoist_code_renamed (r : renaming) (e : expr) :
  hoist_code (rename_expr r e) = option_map (rename_expr r) (hoist_code e).
Proof.
  unfold hoist_code. cbv zeta.
  assert (Htoplevel :
    statics_at_toplevel_of_unit (rename_expr r e)
    = statics_at_toplevel_of_unit e).
  { unfold statics_at_toplevel_of_unit.
    rewrite well_formed_free_continuations_renamed.
    induction (free_continuations e) as [ | k ks IH]; [ reflexivity | ].
    cbn [map existsb].
    rewrite (statics_at_toplevel_equivariant r k true e), IH. reflexivity. }
  rewrite Htoplevel, (proj1 (bound_static_names_equivariant r e)),
    code_bindings_renamed, code_ids_of_bindings_renamed,
    closures_in_scope_renamed, <- without_code_renamed.
  destruct
    (statics_at_toplevel_of_unit e
     && distinct (static_names code_ids_of_binding e)
     && distinct (flat_map code_ids_of_binding (code_bindings e))
     && closures_in_scope (fun _ => false) e);
    reflexivity.
Qed.

Lemma with_code_first_equivariant : rewrite_equivariant with_code_first.
Proof.
  intros r e. unfold with_code_first. rewrite hoist_code_renamed.
  destruct (hoist_code e) as [e' | ]; cbn [option_map]; [ | reflexivity ].
  rewrite drop_deleted_code_equivariant_aux, merge_statics_equivariant_aux.
  reflexivity.
Qed.

Lemma float_code_rewrite_equivariant : rewrite_equivariant float_code.
Proof. intros r e. symmetry. apply float_code_equivariant. Qed.

(* The unit with its static bindings that need nothing first. The check
   that every static name is in scope where it is mentioned reads no
   variable, and reads the continuations only to tell which handlers
   are at toplevel: it answers the same on the renamed unit, for the
   exception continuation renamed. *)
Lemma simple_statics_in_scope_renamed (r : renaming) (syms : list symbol)
    (s : simple) :
  simple_statics_in_scope syms (rename_simple r s)
  = simple_statics_in_scope syms s.
Proof. equivariance. Qed.
#[local] Hint Rewrite simple_statics_in_scope_renamed : equivariance.

Lemma simples_statics_in_scope_renamed (r : renaming) (syms : list symbol)
    (args : list simple) :
  simples_statics_in_scope syms (map (rename_simple r) args)
  = simples_statics_in_scope syms args.
Proof. equivariance. Qed.
#[local] Hint Rewrite simples_statics_in_scope_renamed : equivariance.

Lemma value_slots_statics_in_scope_renamed (r : renaming)
    (syms : list symbol) (slots : list (value_slot * simple)) :
  value_slots_statics_in_scope syms
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_statics_in_scope syms slots.
Proof. equivariance. Qed.
#[local] Hint Rewrite value_slots_statics_in_scope_renamed : equivariance.

Lemma named_statics_in_scope_renamed (r : renaming) (syms : list symbol)
    (ids : list code_id) (d : named) :
  named_statics_in_scope syms ids (rename_named r d)
  = named_statics_in_scope syms ids d.
Proof.
  destruct d; cbn [rename_named named_statics_in_scope];
    rewrite ?simple_statics_in_scope_renamed,
      ?simples_statics_in_scope_renamed,
      ?value_slots_statics_in_scope_renamed;
    reflexivity.
Qed.

Lemma static_const_statics_in_scope_renamed (r : renaming)
    (syms : list symbol) (ids : list code_id) (const : static_const) :
  static_const_statics_in_scope syms ids (rename_static_const r const)
  = static_const_statics_in_scope syms ids const.
Proof.
  destruct const; cbn [rename_static_const static_const_statics_in_scope];
    rewrite ?simple_statics_in_scope_renamed,
      ?simples_statics_in_scope_renamed,
      ?value_slots_statics_in_scope_renamed;
    reflexivity.
Qed.
#[local] Hint Rewrite static_const_statics_in_scope_renamed : equivariance.

Lemma apply_cont_statics_in_scope_renamed (r : renaming)
    (syms : list symbol) (ac : apply_cont) :
  apply_cont_statics_in_scope syms (rename_apply_cont r ac)
  = apply_cont_statics_in_scope syms ac.
Proof. equivariance. Qed.

Lemma apply_statics_in_scope_renamed (r : renaming) (syms : list symbol)
    (ids : list code_id) (ap : apply_expr) :
  apply_statics_in_scope syms ids (rename_apply r ap)
  = apply_statics_in_scope syms ids ap.
Proof.
  unfold apply_statics_in_scope.
  rewrite exn_extra_simples_rename. unfold rename_apply.
  cbn [ap_callee ap_call_kind ap_args].
  rewrite simple_statics_in_scope_renamed, !simples_statics_in_scope_renamed.
  reflexivity.
Qed.

Lemma is_closure_const_renamed (r : renaming) (const : static_const) :
  is_closure_const (rename_static_const r const) = is_closure_const const.
Proof. equivariance. Qed.
#[local] Hint Rewrite is_closure_const_renamed : equivariance.

Lemma closure_symbols_of_binding_renamed (r : renaming)
    (binding : static_binding) :
  closure_symbols_of_binding (rename_static_binding r binding)
  = closure_symbols_of_binding binding.
Proof. equivariance. Qed.

Lemma declared_code_ids_of_binding_renamed (r : renaming)
    (binding : static_binding) :
  declared_code_ids_of_binding (rename_static_binding r binding)
  = declared_code_ids_of_binding binding.
Proof. equivariance. Qed.

Lemma group_consts_in_scope_renamed (r : renaming) (all : list symbol)
    (ids : list code_id) (bindings : list static_binding) :
  ∀ before : list symbol,
    group_consts_in_scope before all ids
      (map (rename_static_binding r) bindings)
    = group_consts_in_scope before all ids bindings.
Proof. equivariance. Qed.

Fixpoint expr_statics_in_scope_renamed (r : renaming) (e : expr) {struct e} :
  ∀ (syms : list symbol) (ids : list code_id),
    expr_statics_in_scope syms ids (rename_expr r e)
    = expr_statics_in_scope syms ids e
with handler_statics_in_scope_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  ∀ (syms : list symbol) (ids : list code_id),
    handler_statics_in_scope syms ids (rename_handler r h)
    = handler_statics_in_scope syms ids h
with binding_statics_in_scope_renamed (r : renaming)
    (binding : static_binding) {struct binding} :
  ∀ (syms : list symbol) (ids : list code_id),
    binding_statics_in_scope syms ids (rename_static_binding r binding)
    = binding_statics_in_scope syms ids binding
with code_statics_in_scope_renamed (r : renaming) (c : code) {struct c} :
  ∀ (syms : list symbol) (ids : list code_id),
    code_statics_in_scope syms ids (rename_code r c)
    = code_statics_in_scope syms ids c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros syms ids; cbn [rename_expr expr_statics_in_scope];
      try reflexivity.
    + rewrite named_statics_in_scope_renamed, expr_statics_in_scope_renamed.
      reflexivity.
    + cbv zeta.
      rewrite
        (flat_map_names_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (flat_map_names_renamed r declared_code_ids_of_binding
           (declared_code_ids_of_binding_renamed r)),
        (flat_map_names_renamed r closure_symbols_of_binding
           (closure_symbols_of_binding_renamed r)),
        group_consts_in_scope_renamed, expr_statics_in_scope_renamed.
      f_equal. f_equal.
      match goal with
      | |- forallb (binding_statics_in_scope ?all ?ids') _
           = forallb (binding_statics_in_scope ?all2 ?ids2) _ =>
        change all2 with all; change ids2 with ids'; generalize all ids'
      end.
      intros all ids'.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_statics_in_scope_renamed r binding), IH.
      reflexivity.
    + rewrite
        (static_names_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (static_names_renamed r declared_code_ids_of_binding
           (declared_code_ids_of_binding_renamed r)),
        handler_statics_in_scope_renamed, expr_statics_in_scope_renamed.
      reflexivity.
    + rewrite expr_statics_in_scope_renamed,
        (static_names_renamed r symbols_of_binding
           (symbols_of_binding_renamed r)),
        (static_names_renamed r declared_code_ids_of_binding
           (declared_code_ids_of_binding_renamed r)).
      f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_statics_in_scope_renamed r h), IH.
      reflexivity.
    + apply apply_statics_in_scope_renamed.
    + apply apply_cont_statics_in_scope_renamed.
    + rewrite simple_statics_in_scope_renamed. f_equal.
      apply forallb_map_with. intros arm. cbn [snd].
      apply apply_cont_statics_in_scope_renamed.
  - destruct h as [params metadata handler]. intros syms ids.
    cbn [rename_handler handler_statics_in_scope].
    apply expr_statics_in_scope_renamed.
  - destruct binding as [id c | id | s const]; intros syms ids;
      cbn [rename_static_binding binding_statics_in_scope];
      try reflexivity.
    apply code_statics_in_scope_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    intros syms ids. cbn [rename_code code_statics_in_scope].
    apply expr_statics_in_scope_renamed.
Qed.

(* The continuations that the renamed unit binds are those that the
   unit binds, renamed. *)
Lemma bound_continuations_renamed (r : renaming) (e : expr) :
  bound_continuations (rename_expr r e)
  = map (rename_continuation r) (bound_continuations e).
Proof. equivariance. Qed.
#[local] Hint Rewrite bound_continuations_renamed : equivariance.

Lemma interface_conts_not_rebound_renamed (r : renaming)
    (u : unit_interface) (e : expr) :
  interface_conts_not_rebound (rename_unit_interface r u) (rename_expr r e)
  = interface_conts_not_rebound u e.
Proof. equivariance. Qed.

Lemma static_names_apart_renamed (r : renaming) (imports : list symbol)
    (e : expr) :
  static_names_apart imports (rename_expr r e) = static_names_apart imports e.
Proof.
  unfold static_names_apart.
  rewrite
    (static_names_renamed r symbols_of_binding
       (symbols_of_binding_renamed r)),
    (static_names_renamed r declared_code_ids_of_binding
       (declared_code_ids_of_binding_renamed r)).
  reflexivity.
Qed.

Lemma hoist_statics_renamed (r : renaming) (u : unit_interface) (e : expr) :
  hoist_statics (rename_unit_interface r u) (rename_expr r e)
  = option_map (rename_expr r) (hoist_statics u e).
Proof.
  unfold hoist_statics, statics_in_scope.
  rewrite well_formed_unit_equivariant, expr_statics_in_scope_renamed,
    interface_conts_not_rebound_renamed, static_names_apart_renamed,
    FlambdaHoistStaticsProofs.hoisted_bindings_renamed,
    <- FlambdaHoistStaticsProofs.without_hoisted_renamed.
  destruct
    (well_formed_unit u e
     && expr_statics_in_scope [] [] e
     && interface_conts_not_rebound u e
     && static_names_apart [] e);
    reflexivity.
Qed.

End Code_and_statics_first.

(* The liftings may be under another renaming than the term, if the
   two rename the continuations of the units of the liftings alike:
   the checks go by those of the first. *)
Lemma with_statics_first_renamed_alike (r r' : renaming) (Ls : list lifting)
    (e : expr) :
  rename_conts_alike r r' (unit_continuations Ls) ->
  with_statics_first (map (rename_lifting r) Ls) (rename_expr r' e)
  = rename_expr r' (with_statics_first Ls e).
Proof.
  intros Halike. destruct Ls as [ | L Ls]; [ reflexivity | ].
  cbn [map with_statics_first].
  assert (Hunit :
    lifting_unit (rename_lifting r L)
    = rename_unit_interface r' (lifting_unit L)).
  { unfold rename_lifting, rename_unit_interface. cbn [lifting_unit].
    rewrite <- (Halike (unit_return (lifting_unit L))),
      <- (Halike (unit_exn (lifting_unit L)));
      [ reflexivity | | ];
      unfold unit_continuations; cbn [flat_map]; rewrite among_app;
      apply orb_true_iff; left; rewrite !among_cons, Pos.eqb_refl;
      [ apply orb_true_r | reflexivity ]. }
  rewrite Hunit, hoist_statics_renamed.
  destruct (hoist_statics (lifting_unit L) e) as [e' | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite drop_deleted_code_equivariant_aux, merge_statics_equivariant_aux.
  reflexivity.
Qed.

(* ======================================================================== *)
(* Rewrites that make names                                                 *)
(* ======================================================================== *)

Theorem rewrite_equivariant_up_to : rewrite_equivariant_up_to_statement.
Proof.
  intros f f_equivariant r e _. exists r. split; [ | apply f_equivariant ].
  split; intros; reflexivity.
Qed.

Theorem keeps_scope_compose : keeps_scope_compose_statement.
Proof.
  intros f g Kf Kg e Hclosed.
  destruct (Kf e Hclosed) as (Hclosed_f & Hvars_f & Hconts_f).
  destruct (Kg (f e) Hclosed_f) as (Hclosed_g & Hvars_g & Hconts_g).
  split; [ exact Hclosed_g | ]. split.
  - intros x Hx. apply Hvars_f, Hvars_g, Hx.
  - intros k Hk. apply Hconts_f, Hconts_g, Hk.
Qed.

Theorem equivariant_up_to_compose : equivariant_up_to_compose_statement.
Proof.
  intros f g Kf Kg Hf Hg r e Hclosed.
  destruct (Hf r e Hclosed) as (r1 & [Hvars1 Hconts1] & Heq1).
  destruct (Kf e Hclosed) as (Hclosed_f & _ & _).
  destruct (Hg r1 (f e) Hclosed_f) as (r2 & [Hvars2 Hconts2] & Heq2).
  destruct (Kg (f e) Hclosed_f) as (_ & Hvars_g & Hconts_g).
  exists r2. split; [ split | ].
  - intros x Hx. rewrite (Hvars2 x Hx). apply Hvars1, Hvars_g, Hx.
  - intros k Hk. rewrite (Hconts2 k Hk). apply Hconts1, Hconts_g, Hk.
  - rewrite Heq1. exact Heq2.
Qed.

Theorem keeps_scope_iterate : keeps_scope_iterate_statement.
Proof.
  intros rounds f Kf. induction rounds as [ | rounds IH]; intros e Hclosed.
  - split; [ exact Hclosed | ]. split; intros; assumption.
  - cbn [iterate]. cbv zeta.
    destruct (expr_eqb [] [] e (f e)); [ exact (Kf e Hclosed) | ].
    destruct (Kf e Hclosed) as (Hclosed_f & Hvars_f & Hconts_f).
    destruct (IH (f e) Hclosed_f) as (Hclosed' & Hvars' & Hconts').
    split; [ exact Hclosed' | ]. split.
    + intros x Hx. apply Hvars_f, Hvars', Hx.
    + intros k Hk. apply Hconts_f, Hconts', Hk.
Qed.

(* Two renamings of a term with closed code that rename its free names
   alike give alpha-equivalent terms (a name that is free by the
   relation is among the free names), which the comparison accepts; the
   comparison is symmetric and transitive, and answers the same of two
   terms renamed by one renaming. *)
Theorem expr_eqb_renamed_alike : expr_eqb_renamed_alike_statement.
Proof.
  intros r r' e e' Hclosed [Hvars Hconts].
  assert (Halpha : alpha_equivalent (rename_expr r' e) (rename_expr r e)).
  { apply alpha_renamings_agree; [ exact Hclosed | | ].
    - intros x Hx. apply Hvars, var_free_among, Hx.
    - intros k Hk. apply Hconts, cont_free_among, Hk. }
  pose proof (alpha_equivalent_accepted _ _ Halpha) as Hforth.
  pose proof (expr_eqb_symmetric _ _ Hforth) as Hback.
  pose proof (expr_eqb_renamed r e e') as Hsame.
  destruct (expr_eqb [] [] e e') eqn:He.
  - exact (expr_eqb_transitive _ _ _ Hforth Hsame).
  - destruct (expr_eqb [] [] (rename_expr r' e) (rename_expr r e')) eqn:He';
      [ | reflexivity ].
    pose proof (expr_eqb_transitive _ _ _ Hback He') as Habsurd.
    rewrite Hsame in Habsurd. discriminate.
Qed.

Theorem expr_eqb_up_to : expr_eqb_up_to_statement.
Proof.
  intros f r e e' Kf Hf Hclosed.
  destruct (Hf r e Hclosed) as (r' & Halike & Heq). rewrite Heq.
  apply expr_eqb_renamed_alike; [ | exact Halike ].
  exact (proj1 (Kf e Hclosed)).
Qed.

(* Two renamings that rename the free names of a term alike rename
   alike those of what a rewrite that keeps scope makes of it. *)
Lemma renames_alike_kept (f : expr -> expr) (r r' : renaming) (e : expr) :
  keeps_scope f ->
  code_closed e = true ->
  renames_alike r r' e ->
  code_closed (f e) = true ∧ renames_alike r r' (f e).
Proof.
  intros Kf Hclosed [Hvars Hconts].
  destruct (Kf e Hclosed) as (Hclosed_f & Hvars_f & Hconts_f).
  split; [ exact Hclosed_f | ]. split.
  - intros x Hx. apply Hvars, Hvars_f, Hx.
  - intros k Hk. apply Hconts, Hconts_f, Hk.
Qed.

(* Where a rewrite commutes with renaming up to the names it makes, the
   comparison of a term with what the rewrite makes of it answers the
   same of the term renamed: the renaming of the term and the one that
   the rewrite's result is under rename alike the free names of that
   result. *)
Lemma same_after_renamed (r r1 : renaming) (e e' : expr) :
  code_closed e' = true ->
  renames_alike r r1 e' ->
  expr_eqb [] [] (rename_expr r e) (rename_expr r1 e') = expr_eqb [] [] e e'.
Proof.
  intros Hclosed Halike.
  rewrite (expr_eqb_swap (rename_expr r e)),
    (expr_eqb_renamed_alike r r1 e' e Hclosed Halike).
  apply expr_eqb_swap.
Qed.

Theorem equivariant_up_to_iterate : equivariant_up_to_iterate_statement.
Proof.
  intros rounds f Kf Hf.
  induction rounds as [ | rounds IH]; intros r e Hclosed.
  - exists r. split; [ split; intros; reflexivity | reflexivity ].
  - cbn [iterate]. cbv zeta.
    destruct (Hf r e Hclosed) as (r1 & Halike1 & Heq1).
    destruct (Kf e Hclosed) as (Hclosed_f & _ & _).
    rewrite Heq1, (same_after_renamed r r1 e (f e) Hclosed_f Halike1).
    destruct (expr_eqb [] [] e (f e));
      [ exists r1; split; [ exact Halike1 | reflexivity ] | ].
    destruct Halike1 as [Hvars1 Hconts1].
    destruct (IH r1 (f e) Hclosed_f) as (r2 & [Hvars2 Hconts2] & Heq2).
    destruct (keeps_scope_iterate rounds f Kf (f e) Hclosed_f)
      as (_ & Hvars' & Hconts').
    exists r2. split; [ split | ].
    + intros x Hx. rewrite (Hvars2 x Hx). apply Hvars1, Hvars', Hx.
    + intros k Hk. rewrite (Hconts2 k Hk). apply Hconts1, Hconts', Hk.
    + exact Heq2.
Qed.

(* The second of three renamings does to the free names of a term what
   the first does, and the third does to those of what a rewrite that
   keeps scope makes of the term what the second does: the third does
   to them what the first does. *)
Lemma renames_alike_then (f : expr -> expr) (r r1 r2 : renaming) (e : expr) :
  keeps_scope f ->
  code_closed e = true ->
  renames_alike r r1 e ->
  renames_alike r1 r2 (f e) ->
  renames_alike r r2 (f e).
Proof.
  intros Kf Hclosed [Hvars Hconts] [Hvars' Hconts'].
  destruct (Kf e Hclosed) as (_ & Hv & Hk).
  split.
  - intros x Hx. rewrite (Hvars' x Hx). apply Hvars, Hv, Hx.
  - intros k Hk'. rewrite (Hconts' k Hk'). apply Hconts, Hk, Hk'.
Qed.

Lemma map_rename_conts_alike (r r' : renaming) (ks : list continuation) :
  rename_conts_alike r r' ks ->
  map (rename_continuation r') ks = map (rename_continuation r) ks.
Proof.
  intros Halike. apply map_ext_in. intros k Hk.
  apply Halike, among_In, Hk.
Qed.

Theorem rewrite_equivariant_up_to_avoiding :
  rewrite_equivariant_up_to_avoiding_statement.
Proof.
  intros f Hf r avoid e _. exists r. split; [ | split ].
  - split; intros; reflexivity.
  - intros k _. reflexivity.
  - apply Hf.
Qed.

Theorem equivariant_up_to_avoiding_before :
  equivariant_up_to_avoiding_before_statement.
Proof.
  intros f g Kg Hg Hf r avoid e Hclosed.
  destruct (Kg e Hclosed) as (Hclosed_g & _ & _).
  destruct (Hf r avoid (g e) Hclosed_g) as (r' & Halike & Havoid & Heq).
  exists r'. split; [ exact Halike | split; [ exact Havoid | ] ].
  rewrite Hg. exact Heq.
Qed.

Theorem equivariant_up_to_avoiding_after :
  equivariant_up_to_avoiding_after_statement.
Proof.
  intros f g Kf Kg Hg Hf r avoid e Hclosed.
  destruct (Hf r avoid e Hclosed) as (r' & Halike & Havoid & Heq).
  destruct (Kf avoid e Hclosed) as (Hclosed_f & _ & _).
  destruct (renames_alike_kept g r r' (f avoid e) Kg Hclosed_f Halike)
    as (_ & Halike_g).
  exists r'. split; [ exact Halike_g | split; [ exact Havoid | ] ].
  rewrite Heq. apply Hg.
Qed.

Theorem equivariant_up_to_avoiding_iterate :
  equivariant_up_to_avoiding_iterate_statement.
Proof.
  intros rounds f Kf Hf.
  induction rounds as [ | rounds IH]; intros r avoid e Hclosed.
  - exists r. split; [ split; intros; reflexivity | ].
    split; [ intros k _; reflexivity | reflexivity ].
  - cbn [iterate]. cbv zeta.
    destruct (Hf r avoid e Hclosed) as (r1 & Halike1 & Havoid1 & Heq1).
    destruct (Kf avoid e Hclosed) as (Hclosed_f & _ & _).
    rewrite Heq1,
      (same_after_renamed r r1 e (f avoid e) Hclosed_f Halike1).
    destruct (expr_eqb [] [] e (f avoid e)).
    { exists r1. split; [ exact Halike1 | ].
      split; [ exact Havoid1 | reflexivity ]. }
    destruct (IH r1 avoid (f avoid e) Hclosed_f)
      as (r2 & Halike2 & Havoid2 & Heq2).
    rewrite (map_rename_conts_alike r r1 avoid Havoid1) in Heq2.
    exists r2. split; [ | split ].
    + exact
        (renames_alike_then (iterate rounds (f avoid)) r r1 r2 (f avoid e)
           (keeps_scope_iterate rounds (f avoid) (Kf avoid)) Hclosed_f
           Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid1, Hk.
    + exact Heq2.
Qed.

Theorem equivariant_up_to_avoiding_nothing :
  equivariant_up_to_avoiding_nothing_statement.
Proof.
  intros f Hf r e Hclosed.
  destruct (Hf r [] e Hclosed) as (r' & Halike & _ & Heq).
  exists r'. split; [ exact Halike | exact Heq ].
Qed.

(* Two rewrites that are given the same continuations to avoid, one
   after the other: the second renaming of the first is the one that
   the second is asked about, and it does to the continuations what
   the first renaming does, so the second is given the same list. *)
Lemma equivariant_up_to_avoiding_then
    (f g : list continuation -> expr -> expr) :
  (∀ avoid : list continuation, keeps_scope (f avoid)) ->
  (∀ avoid : list continuation, keeps_scope (g avoid)) ->
  equivariant_up_to_avoiding f ->
  equivariant_up_to_avoiding g ->
  equivariant_up_to_avoiding (fun avoid e => g avoid (f avoid e)).
Proof.
  intros Kf Kg Hf Hg r avoid e Hclosed.
  destruct (Hf r avoid e Hclosed) as (r1 & Halike1 & Havoid1 & Heq1).
  destruct (Kf avoid e Hclosed) as (Hclosed_f & _ & _).
  destruct (Hg r1 avoid (f avoid e) Hclosed_f)
    as (r2 & Halike2 & Havoid2 & Heq2).
  rewrite (map_rename_conts_alike r r1 avoid Havoid1) in Heq2.
  exists r2. split; [ | split ].
  - exact
      (renames_alike_then (g avoid) r r1 r2 (f avoid e) (Kg avoid) Hclosed_f
         Halike1 Halike2).
  - intros k Hk. rewrite (Havoid2 k Hk). apply Havoid1, Hk.
  - rewrite Heq1. exact Heq2.
Qed.

Theorem keeps_scope_unless_same : keeps_scope_unless_same_statement.
Proof.
  intros f g h Kf Kg Kh e Hclosed. cbv beta.
  destruct (expr_eqb [] [] (f e) (g (f e))).
  - exact (Kf e Hclosed).
  - exact
      (keeps_scope_compose (fun e => g (f e)) h (keeps_scope_compose f g Kf Kg)
         Kh e Hclosed).
Qed.

(* The second renaming of the first rewrite and that of the second
   rename alike the free names of what the second gives, so the
   comparison takes the same branch for the renamed term. *)
Theorem equivariant_up_to_unless_same :
  equivariant_up_to_unless_same_statement.
Proof.
  intros f g h Kf Kg Kh Hf Hg Hh r e Hclosed. cbv beta.
  destruct (Hf r e Hclosed) as (r1 & Halike1 & Heq1).
  destruct (Kf e Hclosed) as (Hclosed_f & _ & _).
  destruct (Hg r1 (f e) Hclosed_f) as (r2 & Halike2 & Heq2).
  destruct (Kg (f e) Hclosed_f) as (Hclosed_g & _ & _).
  rewrite Heq1, Heq2.
  rewrite (expr_eqb_swap (rename_expr r1 (f e))),
    (expr_eqb_renamed_alike r1 r2 (g (f e)) (f e) Hclosed_g Halike2),
    (expr_eqb_swap (g (f e))).
  destruct (expr_eqb [] [] (f e) (g (f e))).
  - exists r1. split; [ exact Halike1 | reflexivity ].
  - destruct (Hh r2 (g (f e)) Hclosed_g) as (r3 & Halike3 & Heq3).
    exists r3. split; [ | exact Heq3 ].
    exact
      (renames_alike_then h r r2 r3 (g (f e)) Kh Hclosed_g
         (renames_alike_then g r r1 r2 (f e) Kg Hclosed_f Halike1 Halike2)
         Halike3).
Qed.

Theorem equivariant_up_to_avoiding_unless_same :
  equivariant_up_to_avoiding_unless_same_statement.
Proof.
  intros f h g Kf Kg Kh Hf Hg Hh r avoid e Hclosed. cbv beta.
  destruct (Hf r avoid e Hclosed) as (r1 & Halike1 & Havoid1 & Heq1).
  destruct (Kf avoid e Hclosed) as (Hclosed_f & _ & _).
  destruct (Kg (f avoid e) Hclosed_f) as (Hclosed_g & _ & _).
  rewrite Heq1, (Hg r1 (f avoid e)), expr_eqb_renamed.
  destruct (expr_eqb [] [] (f avoid e) (g (f avoid e))).
  - exists r1. split; [ exact Halike1 | split; [ exact Havoid1 | ] ].
    reflexivity.
  - destruct (Hh r1 avoid (g (f avoid e)) Hclosed_g)
      as (r2 & Halike2 & Havoid2 & Heq2).
    rewrite (map_rename_conts_alike r r1 avoid Havoid1) in Heq2.
    exists r2. split; [ | split ].
    + exact
        (renames_alike_then (h avoid) r r1 r2 (g (f avoid e)) (Kh avoid)
           Hclosed_g
           (proj2
              (renames_alike_kept g r r1 (f avoid e) Kg Hclosed_f Halike1))
           Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid1, Hk.
    + exact Heq2.
Qed.

(* ======================================================================== *)
(* The validation, one of whose rewrites makes names                        *)
(* ======================================================================== *)

(* Each rewrite of the validation keeps scope: the statements of
   FlambdaKeepsScopeSpec.v, of FlambdaValidateScopeSpec.v and of
   FlambdaStaticsScopeSpec.v. *)
Lemma inline_calls_scope (kept : list code_id) :
  keeps_scope (inline_calls kept []).
Proof. exact (FlambdaValidateScopeProofs.inline_calls_keeps_scope kept). Qed.

Lemma inline_sibling_calls_scope (kept : list code_id) :
  keeps_scope (inline_sibling_calls kept).
Proof.
  exact (FlambdaValidateScopeProofs.inline_sibling_calls_keeps_scope kept).
Qed.

Lemma rename_apart_avoiding_scope (avoid : list continuation) :
  keeps_scope (rename_apart_avoiding avoid).
Proof.
  exact (FlambdaKeepsScopeProofs.rename_apart_avoiding_keeps_scope avoid).
Qed.

Lemma normalise_before_inlined_scope : keeps_scope normalise_before_inlined.
Proof.
  exact FlambdaValidateScopeProofs.normalise_before_inlined_keeps_scope.
Qed.

Lemma with_known_scope : keeps_scope with_known.
Proof. exact FlambdaStaticsScopeProofs.with_known_keeps_scope. Qed.

Lemma with_lifting_scope (L : lifting) : keeps_scope (with_lifting L).
Proof. exact (FlambdaStaticsScopeProofs.with_lifting_keeps_scope L). Qed.

Lemma without_slots_scope (slots : list value_slot) :
  keeps_scope (without_slots slots).
Proof. exact (FlambdaStaticsScopeProofs.without_slots_keeps_scope slots). Qed.

Theorem without_head_statics_equivariant :
  without_head_statics_equivariant_statement.
Proof.
  intros unused r e. unfold without_head_statics.
  rewrite head_statics_removable_renamed.
  destruct (head_statics_removable unused e); [ | reflexivity ].
  apply without_statics_equivariant.
Qed.

Lemma without_statics_scope (unused : dropped) :
  keeps_scope (without_statics unused).
Proof.
  exact (FlambdaStaticsScopeProofs.without_statics_keeps_scope unused).
Qed.

Lemma order_statics_as_scope (pattern : expr) :
  keeps_scope (order_statics_as pattern).
Proof.
  exact (FlambdaStaticsScopeProofs.order_statics_as_keeps_scope pattern).
Qed.

Lemma with_statics_first_scope (Ls : list lifting) :
  keeps_scope (with_statics_first Ls).
Proof.
  exact (FlambdaStaticsScopeProofs.with_statics_first_keeps_scope Ls).
Qed.

(* The renaming that the file of [rename_apart] gives agrees with the
   first on the names that are free in the term, and on the
   continuations to avoid. A name that is free in the result is free
   in the term, as the rewrite keeps scope. *)
Theorem rename_apart_avoiding_commutes_up_to :
  rename_apart_avoiding_commutes_up_to_statement.
Proof.
  intros r avoid e Hclosed.
  destruct (rename_apart_avoiding_equivariant_up_to r avoid e Hclosed)
    as (r' & Hvars & Hconts & Havoid & Heq).
  destruct (rename_apart_avoiding_scope avoid e Hclosed)
    as (_ & Hvars_kept & Hconts_kept).
  exists r'. split; [ split | split; [ exact Havoid | exact Heq ] ].
  - intros x Hx. apply Hvars, Hvars_kept, Hx.
  - intros k Hk. apply Hconts, Hconts_kept, Hk.
Qed.

Theorem rename_apart_commutes_up_to : rename_apart_commutes_up_to_statement.
Proof.
  exact
    (equivariant_up_to_avoiding_nothing rename_apart_avoiding
       rename_apart_avoiding_commutes_up_to).
Qed.

(* A round is four rewrites, one after the other. The first two, the
   inlinings, as one rewrite: it commutes with renaming and keeps
   scope. *)
Definition inline_both (kept : list code_id) (e : expr) : expr :=
  inline_sibling_calls kept (inline_calls kept [] e).

Lemma inline_both_scope (kept : list code_id) : keeps_scope (inline_both kept).
Proof.
  exact
    (keeps_scope_compose (inline_calls kept []) (inline_sibling_calls kept)
       (inline_calls_scope kept) (inline_sibling_calls_scope kept)).
Qed.

Lemma inline_both_equivariant (kept : list code_id) :
  rewrite_equivariant (inline_both kept).
Proof.
  intros r e. unfold inline_both.
  rewrite (inline_calls_unknowing_equivariant kept r e).
  apply inline_sibling_calls_equivariant.
Qed.

(* What a round does before it looks for twins, in which the names
   made avoid the continuations given: the inlinings before the
   rewrite that makes names, and the rest after it, commute with
   renaming and keep scope. *)
Definition settle_round (kept : list code_id) (avoid : list continuation)
    (e : expr) : expr :=
  normalise_before_inlined (rename_apart_avoiding avoid (inline_both kept e)).

Lemma settle_round_scope (avoid : list continuation) (kept : list code_id) :
  keeps_scope (settle_round kept avoid).
Proof.
  exact
    (keeps_scope_compose
       (fun e => rename_apart_avoiding avoid (inline_both kept e))
       normalise_before_inlined
       (keeps_scope_compose (inline_both kept) (rename_apart_avoiding avoid)
          (inline_both_scope kept) (rename_apart_avoiding_scope avoid))
       normalise_before_inlined_scope).
Qed.

Lemma settle_round_up_to (kept : list code_id) :
  equivariant_up_to_avoiding (settle_round kept).
Proof.
  exact
    (equivariant_up_to_avoiding_after
       (fun avoid e => rename_apart_avoiding avoid (inline_both kept e))
       normalise_before_inlined
       (fun avoid =>
          keeps_scope_compose (inline_both kept) (rename_apart_avoiding avoid)
            (inline_both_scope kept) (rename_apart_avoiding_scope avoid))
       normalise_before_inlined_scope normalise_before_inlined_equivariant
       (equivariant_up_to_avoiding_before rename_apart_avoiding
          (inline_both kept) (inline_both_scope kept)
          (inline_both_equivariant kept)
          rename_apart_avoiding_commutes_up_to)).
Qed.

(* The twins, where a round has settled the term: the parameters
   bound again and the twins made, each in the term and then in the
   bodies of code, commute with renaming and keep scope; the rewrite
   that makes names is between them; and the comparison that says
   whether a twin was made answers the same of the two terms under one
   renaming. *)
Definition shadow_all (e : expr) : expr :=
  rewrite_code FlambdaShadowParamsSpec.shadow_params
    (FlambdaShadowParamsSpec.shadow_params e).

Definition twin_all (e : expr) : expr :=
  rewrite_code FlambdaTwinParamsSpec.twin_params
    (FlambdaTwinParamsSpec.twin_params e).

Lemma shadow_all_scope : keeps_scope shadow_all.
Proof.
  exact
    (keeps_scope_compose FlambdaShadowParamsSpec.shadow_params
       (rewrite_code FlambdaShadowParamsSpec.shadow_params)
       FlambdaStaticsScopeProofs.shadow_params_keeps_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          FlambdaStaticsScopeProofs.shadow_params_keeps_scope)).
Qed.

Lemma shadow_all_equivariant : rewrite_equivariant shadow_all.
Proof.
  intros r e. unfold shadow_all. rewrite (shadow_params_equivariant r e).
  apply rewrite_code_equivariant, shadow_params_equivariant.
Qed.

Lemma twin_all_scope : keeps_scope twin_all.
Proof.
  exact
    (keeps_scope_compose FlambdaTwinParamsSpec.twin_params
       (rewrite_code FlambdaTwinParamsSpec.twin_params)
       FlambdaStaticsScopeProofs.twin_params_keeps_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          FlambdaStaticsScopeProofs.twin_params_keeps_scope)).
Qed.

Lemma twin_all_equivariant : rewrite_equivariant twin_all.
Proof.
  intros r e. unfold twin_all. rewrite (twin_params_equivariant r e).
  apply rewrite_code_equivariant, twin_params_equivariant.
Qed.

Lemma with_twins_unfold (avoid : list continuation) (e : expr) :
  with_twins avoid e
  = if expr_eqb [] [] (rename_apart_avoiding avoid (shadow_all e))
         (twin_all (rename_apart_avoiding avoid (shadow_all e)))
    then e
    else
      normalise_before_inlined
        (twin_all (rename_apart_avoiding avoid (shadow_all e))).
Proof. reflexivity. Qed.

Lemma with_twins_scope (avoid : list continuation) :
  keeps_scope (with_twins avoid).
Proof. exact (FlambdaValidateScopeProofs.with_twins_keeps_scope avoid). Qed.

Lemma with_twins_up_to : equivariant_up_to_avoiding with_twins.
Proof.
  intros r avoid e Hclosed.
  destruct
    (equivariant_up_to_avoiding_before rename_apart_avoiding shadow_all
       shadow_all_scope shadow_all_equivariant
       rename_apart_avoiding_commutes_up_to r avoid e Hclosed)
    as (r1 & Halike1 & Havoid1 & Heq1).
  destruct
    (keeps_scope_compose shadow_all (rename_apart_avoiding avoid)
       shadow_all_scope (rename_apart_avoiding_scope avoid) e Hclosed)
    as (Hclosed1 & _ & _).
  cbv beta in Halike1, Heq1, Hclosed1.
  rewrite !with_twins_unfold, Heq1, (twin_all_equivariant r1 _).
  set (shadowed := rename_apart_avoiding avoid (shadow_all e)) in *.
  rewrite
    (expr_eqb_renamed_alike r1 r1 shadowed (twin_all shadowed) Hclosed1
       (conj (fun _ _ => eq_refl) (fun _ _ => eq_refl))).
  destruct (expr_eqb [] [] shadowed (twin_all shadowed)).
  - exists r. split; [ split; intros; reflexivity | ].
    split; [ intros k _; reflexivity | reflexivity ].
  - exists r1. split; [ | split; [ exact Havoid1 | ] ].
    + exact
        (proj2
           (renames_alike_kept
              (fun x => normalise_before_inlined (twin_all x)) r r1 shadowed
              (keeps_scope_compose twin_all normalise_before_inlined
                 twin_all_scope normalise_before_inlined_scope)
              Hclosed1 Halike1)).
    + apply normalise_before_inlined_equivariant.
Qed.

(* A round: what it does before it looks for twins and, where that
   has left the term as it was, the twins. The comparison answers the
   same of the term renamed. *)
Lemma normalise_before_round_avoiding_unfold (avoid : list continuation)
    (kept : list code_id) (e : expr) :
  normalise_before_round_avoiding avoid kept e
  = if expr_eqb [] [] e (settle_round kept avoid e)
    then with_twins avoid (settle_round kept avoid e)
    else settle_round kept avoid e.
Proof. reflexivity. Qed.

Lemma normalise_before_round_avoiding_scope (avoid : list continuation)
    (kept : list code_id) :
  keeps_scope (normalise_before_round_avoiding avoid kept).
Proof.
  exact
    (FlambdaValidateScopeProofs.normalise_before_round_avoiding_keeps_scope
       avoid kept).
Qed.

Lemma normalise_before_round_avoiding_up_to (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => normalise_before_round_avoiding avoid kept).
Proof.
  intros r avoid e Hclosed. cbv beta.
  rewrite !normalise_before_round_avoiding_unfold.
  destruct (settle_round_up_to kept r avoid e Hclosed)
    as (r1 & Halike1 & Havoid1 & Heq1).
  destruct (settle_round_scope avoid kept e Hclosed) as (Hclosed1 & _ & _).
  rewrite Heq1,
    (same_after_renamed r r1 e (settle_round kept avoid e) Hclosed1 Halike1).
  destruct (expr_eqb [] [] e (settle_round kept avoid e)).
  - destruct (with_twins_up_to r1 avoid (settle_round kept avoid e) Hclosed1)
      as (r2 & Halike2 & Havoid2 & Heq2).
    rewrite (map_rename_conts_alike r r1 avoid Havoid1) in Heq2.
    exists r2. split; [ | split ].
    + exact
        (renames_alike_then (with_twins avoid) r r1 r2
           (settle_round kept avoid e) (with_twins_scope avoid) Hclosed1
           Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid1, Hk.
    + exact Heq2.
  - exists r1. split; [ exact Halike1 | ].
    split; [ exact Havoid1 | reflexivity ].
Qed.

(* A round of the term before is that round with nothing to avoid. *)
Theorem normalise_before_round_keeps_scope :
  normalise_before_round_keeps_scope_statement.
Proof.
  intros kept. exact (normalise_before_round_avoiding_scope [] kept).
Qed.

Theorem normalise_before_round_equivariant_up_to :
  normalise_before_round_equivariant_up_to_statement.
Proof.
  intros kept.
  exact
    (equivariant_up_to_avoiding_nothing
       (fun avoid => normalise_before_round_avoiding avoid kept)
       (normalise_before_round_avoiding_up_to kept)).
Qed.

(* The first phase of the term before, however many times it is
   applied: it commutes with renaming and keeps scope. *)
Lemma params_phase_scope (rounds : nat) :
  keeps_scope (iterate rounds first_phase_round).
Proof.
  exact
    (keeps_scope_iterate rounds first_phase_round
       FlambdaValidateScopeProofs.first_phase_round_keeps_scope).
Qed.

Lemma params_phase_equivariant (rounds : nat) :
  rewrite_equivariant (iterate rounds first_phase_round).
Proof.
  exact
    (iterate_equivariant rounds first_phase_round
       first_phase_round_equivariant).
Qed.

(* The demotion of exception handlers and the rounds after it, which
   end the normalisation of the term before: they keep scope, by the
   scope law of the demotion (FlambdaDemoteScopeProofs, through
   FlambdaValidateScopeProofs) and those of the rounds, and they
   commute with renaming. *)
Lemma demote_and_tidy_scope (rounds : nat) :
  keeps_scope (demote_and_tidy rounds).
Proof.
  exact
    (keeps_scope_compose
       (fun e => iterate rounds first_phase_round (demote_handlers [] e))
       (iterate rounds normalise_before_inlined)
       (keeps_scope_compose (demote_handlers [])
          (iterate rounds first_phase_round)
          FlambdaValidateScopeProofs.demote_handlers_closed_keeps_scope
          (params_phase_scope rounds))
       (keeps_scope_iterate rounds normalise_before_inlined
          normalise_before_inlined_scope)).
Qed.

Lemma demote_and_tidy_up_to (rounds : nat) :
  equivariant_up_to (demote_and_tidy rounds).
Proof.
  exact
    (rewrite_equivariant_up_to (demote_and_tidy rounds)
       (demote_and_tidy_equivariant rounds)).
Qed.

(* The rewrites from the types, where what they give keeps scope, as
   one rewrite: it commutes with renaming, since the function does and
   the test reads the same of a renamed term. *)
Definition by_types (e : expr) : expr :=
  FlambdaGuardScopeSpec.guard_scope
    FlambdaSimplifyCarriedSpec.simplify_by_types_carried e.

Lemma by_types_scope : keeps_scope by_types.
Proof.
  exact
    (FlambdaValidateScopeProofs.guard_scope_keeps_scope
       FlambdaSimplifyCarriedSpec.simplify_by_types_carried).
Qed.

Lemma by_types_equivariant : rewrite_equivariant by_types.
Proof.
  exact
    (FlambdaValidateScopeProofs.guard_scope_equivariant
       FlambdaSimplifyCarriedSpec.simplify_by_types_carried
       FlambdaSimplifyCarriedProofs.simplify_by_types_carried_equivariant).
Qed.

(* The first phase, rounds of a rewrite, the rewrites from the types,
   and the rounds of that rewrite once more: what each normalisation
   of the term before is, or begins with, for the round that is its
   own. *)
Definition phases_with (rounds : nat) (round : expr -> expr) (e : expr)
    : expr :=
  iterate rounds round
    (by_types (iterate rounds round (iterate rounds first_phase_round e))).

(* They keep scope where the round does. *)
Lemma phases_with_scope (rounds : nat) (round : expr -> expr) :
  keeps_scope round -> keeps_scope (phases_with rounds round).
Proof.
  intros Kround.
  pose proof (keeps_scope_iterate rounds round Kround) as Krounds.
  exact
    (keeps_scope_compose
       (fun e =>
          by_types
            (iterate rounds round (iterate rounds first_phase_round e)))
       (iterate rounds round)
       (keeps_scope_compose
          (fun e => iterate rounds round (iterate rounds first_phase_round e))
          by_types
          (keeps_scope_compose (iterate rounds first_phase_round)
             (iterate rounds round) (params_phase_scope rounds) Krounds)
          by_types_scope)
       Krounds).
Qed.

(* For a round that is given continuations to avoid, that keeps scope
   and that commutes with renaming up to the names it makes, they
   commute with renaming up to the names they make: the first phase
   and the rewrites from the types commute with renaming, and both
   runs of the rounds are given the same continuations. *)
Lemma phases_with_up_to (rounds : nat)
    (round : list continuation -> expr -> expr) :
  (∀ avoid : list continuation, keeps_scope (round avoid)) ->
  equivariant_up_to_avoiding round ->
  equivariant_up_to_avoiding (fun avoid => phases_with rounds (round avoid)).
Proof.
  intros Kround Hround.
  pose proof
    (fun avoid => keeps_scope_iterate rounds (round avoid) (Kround avoid))
    as Krounds.
  pose proof (equivariant_up_to_avoiding_iterate rounds round Kround Hround)
    as Hrounds.
  pose proof
    (fun avoid =>
       keeps_scope_compose (iterate rounds first_phase_round)
         (iterate rounds (round avoid)) (params_phase_scope rounds)
         (Krounds avoid))
    as Kfirst.
  exact
    (equivariant_up_to_avoiding_then
       (fun avoid e =>
          by_types
            (iterate rounds (round avoid)
               (iterate rounds first_phase_round e)))
       (fun avoid => iterate rounds (round avoid))
       (fun avoid =>
          keeps_scope_compose
            (fun e =>
               iterate rounds (round avoid)
                 (iterate rounds first_phase_round e))
            by_types (Kfirst avoid) by_types_scope)
       Krounds
       (equivariant_up_to_avoiding_after
          (fun avoid e =>
             iterate rounds (round avoid)
               (iterate rounds first_phase_round e))
          by_types Kfirst by_types_scope by_types_equivariant
          (equivariant_up_to_avoiding_before
             (fun avoid => iterate rounds (round avoid))
             (iterate rounds first_phase_round) (params_phase_scope rounds)
             (params_phase_equivariant rounds) Hrounds))
       Hrounds).
Qed.

Theorem normalise_before_keeps_scope : normalise_before_keeps_scope_statement.
Proof.
  intros rounds kept e.
  exact
    (keeps_scope_compose (phases_with rounds (normalise_before_round kept))
       (demote_and_tidy rounds)
       (phases_with_scope rounds (normalise_before_round kept)
          (normalise_before_round_keeps_scope kept))
       (demote_and_tidy_scope rounds) e).
Qed.

Theorem normalise_before_equivariant_up_to :
  normalise_before_equivariant_up_to_statement.
Proof.
  intros rounds kept r e.
  exact
    (equivariant_up_to_compose
       (phases_with rounds (normalise_before_round kept))
       (demote_and_tidy rounds)
       (phases_with_scope rounds (normalise_before_round kept)
          (normalise_before_round_keeps_scope kept))
       (demote_and_tidy_scope rounds)
       (equivariant_up_to_avoiding_nothing
          (fun avoid =>
             phases_with rounds (normalise_before_round_avoiding avoid kept))
          (phases_with_up_to rounds
             (fun avoid => normalise_before_round_avoiding avoid kept)
             (fun avoid => normalise_before_round_avoiding_scope avoid kept)
             (normalise_before_round_avoiding_up_to kept)))
       (demote_and_tidy_up_to rounds) r e).
Qed.

(* The use of what is known of the static closures in sight and of
   what the unit binds its symbols to, as one rewrite. *)
Lemma use_known_closures_scope : keeps_scope use_known_closures.
Proof. exact FlambdaStaticsScopeProofs.use_known_closures_keeps_scope. Qed.

Lemma float_code_scope : keeps_scope float_code.
Proof. exact FlambdaStaticsScopeProofs.float_code_keeps_scope. Qed.

Lemma read_known_slots_scope : keeps_scope read_known_slots.
Proof. exact FlambdaStaticsScopeProofs.read_known_slots_keeps_scope. Qed.

Lemma read_known_slots_rewrite_equivariant :
  rewrite_equivariant read_known_slots.
Proof. exact FlambdaKnownSlotsProofs.read_known_slots_equivariant. Qed.

Lemma read_known_slots_anywhere_scope : keeps_scope read_known_slots_anywhere.
Proof.
  exact FlambdaStaticsScopeProofs.read_known_slots_anywhere_keeps_scope.
Qed.

Lemma share_static_reads_scope :
  keeps_scope FlambdaSharedReadsSpec.share_static_reads.
Proof. exact FlambdaSharedReadsProofs.share_static_reads_keeps_scope. Qed.

Lemma share_static_reads_rewrite_equivariant :
  rewrite_equivariant FlambdaSharedReadsSpec.share_static_reads.
Proof. exact FlambdaSharedReadsProofs.share_static_reads_equivariant. Qed.

Definition use_known (e : expr) : expr :=
  FlambdaSharedReadsSpec.share_static_reads
    (read_known_slots (with_known (use_known_closures e))).

Lemma use_known_scope : keeps_scope use_known.
Proof.
  exact
    (keeps_scope_compose
       (fun e => read_known_slots (with_known (use_known_closures e)))
       FlambdaSharedReadsSpec.share_static_reads
       (keeps_scope_compose (fun e => with_known (use_known_closures e))
          read_known_slots
          (keeps_scope_compose use_known_closures with_known
             use_known_closures_scope with_known_scope)
          read_known_slots_scope)
       share_static_reads_scope).
Qed.

Lemma use_known_equivariant : rewrite_equivariant use_known.
Proof.
  intros r e. unfold use_known.
  rewrite use_known_closures_rewrite_equivariant, with_known_equivariant,
    read_known_slots_rewrite_equivariant.
  apply share_static_reads_rewrite_equivariant.
Qed.

(* What ends the normalisation of a term in which something was
   lifted: the demotion with its rounds, and then the rounds that use
   what is known, each followed by the demotion with its rounds. None
   of them makes a name. *)
Lemma demote_and_tidy_known_round_scope (rounds : nat) :
  keeps_scope (demote_and_tidy_known_round rounds).
Proof.
  exact
    (keeps_scope_compose use_known (demote_and_tidy rounds) use_known_scope
       (demote_and_tidy_scope rounds)).
Qed.

Lemma demote_and_tidy_known_round_equivariant (rounds : nat) :
  rewrite_equivariant (demote_and_tidy_known_round rounds).
Proof.
  intros r e. unfold demote_and_tidy_known_round.
  rewrite use_known_closures_rewrite_equivariant, with_known_equivariant,
    read_known_slots_rewrite_equivariant,
    share_static_reads_rewrite_equivariant.
  apply demote_and_tidy_equivariant.
Qed.

Lemma demote_and_tidy_known_scope (rounds : nat) :
  keeps_scope (demote_and_tidy_known rounds).
Proof.
  exact
    (keeps_scope_compose (demote_and_tidy rounds)
       (iterate rounds (demote_and_tidy_known_round rounds))
       (demote_and_tidy_scope rounds)
       (keeps_scope_iterate rounds (demote_and_tidy_known_round rounds)
          (demote_and_tidy_known_round_scope rounds))).
Qed.

Theorem demote_and_tidy_known_equivariant :
  demote_and_tidy_known_equivariant_statement.
Proof.
  intros rounds r e. unfold demote_and_tidy_known.
  rewrite demote_and_tidy_equivariant.
  exact
    (iterate_equivariant rounds (demote_and_tidy_known_round rounds)
       (demote_and_tidy_known_round_equivariant rounds) r _).
Qed.

(* A round of the normalisation of a term in which something was
   lifted: a round that avoids the continuations given, then that
   rewrite. *)
Lemma normalise_lifted_round_scope (avoid : list continuation)
    (kept : list code_id) :
  keeps_scope (normalise_lifted_round avoid kept).
Proof.
  exact
    (keeps_scope_compose (normalise_before_round_avoiding avoid kept)
       use_known (normalise_before_round_avoiding_scope avoid kept)
       use_known_scope).
Qed.

Lemma normalise_lifted_round_up_to (kept : list code_id) :
  equivariant_up_to_avoiding (fun avoid => normalise_lifted_round avoid kept).
Proof.
  exact
    (equivariant_up_to_avoiding_after
       (fun avoid => normalise_before_round_avoiding avoid kept) use_known
       (fun avoid => normalise_before_round_avoiding_scope avoid kept)
       use_known_scope use_known_equivariant
       (normalise_before_round_avoiding_up_to kept)).
Qed.

(* The rounds of a term in which something was lifted. *)
Lemma lifted_rounds_scope (rounds : nat) (avoid : list continuation)
    (kept : list code_id) :
  keeps_scope (iterate rounds (normalise_lifted_round avoid kept)).
Proof.
  exact
    (keeps_scope_iterate rounds (normalise_lifted_round avoid kept)
       (normalise_lifted_round_scope avoid kept)).
Qed.

Lemma lifted_rounds_up_to (rounds : nat) (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => iterate rounds (normalise_lifted_round avoid kept)).
Proof.
  exact
    (equivariant_up_to_avoiding_iterate rounds
       (fun avoid => normalise_lifted_round avoid kept)
       (fun avoid => normalise_lifted_round_scope avoid kept)
       (normalise_lifted_round_up_to kept)).
Qed.

Theorem normalise_lifted_keeps_scope : normalise_lifted_keeps_scope_statement.
Proof.
  intros rounds avoid kept.
  exact
    (phases_with_scope rounds (normalise_lifted_round avoid kept)
       (normalise_lifted_round_scope avoid kept)).
Qed.

Theorem normalise_lifted_equivariant_up_to :
  normalise_lifted_equivariant_up_to_statement.
Proof.
  intros rounds kept.
  exact
    (phases_with_up_to rounds (fun avoid => normalise_lifted_round avoid kept)
       (fun avoid => normalise_lifted_round_scope avoid kept)
       (normalise_lifted_round_up_to kept)).
Qed.

(* The calls of the closures that a [Let] makes made direct, in the
   term and in the bodies of its code: the kind of a call has no name
   in it. *)
Lemma call_made_closures_scope : keeps_scope call_made_closures.
Proof.
  intros e Hclosed. unfold call_made_closures.
  destruct (FlambdaClosureCallsProofs.call_closures_knowing_names [] e)
    as (Hvariables & Hcontinuations & Hcode).
  rewrite Hvariables, Hcontinuations, Hcode.
  split; [ exact Hclosed | split; intros name Hname; exact Hname ].
Qed.

Lemma call_made_closures_rewrite_equivariant :
  rewrite_equivariant call_made_closures.
Proof.
  intros r e. symmetry.
  apply FlambdaClosureCallsProofs.call_made_closures_equivariant.
Qed.

Lemma with_made_closures_called_scope : keeps_scope with_made_closures_called.
Proof.
  exact
    (keeps_scope_compose call_made_closures (rewrite_code call_made_closures)
       call_made_closures_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          call_made_closures_scope)).
Qed.

Lemma with_made_closures_called_equivariant :
  rewrite_equivariant with_made_closures_called.
Proof.
  intros r e. unfold with_made_closures_called.
  rewrite call_made_closures_rewrite_equivariant.
  apply (rewrite_code_equivariant _ call_made_closures_rewrite_equivariant).
Qed.

(* A round of FlambdaValidateSpec.normalise_lifted_calling: the round
   of [normalise_lifted] and then, where [calling] is set, those calls
   made direct. *)
Lemma normalise_lifted_round_calling_scope (calling : bool)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope (normalise_lifted_round_calling calling avoid kept).
Proof.
  destruct calling; [ | exact (normalise_lifted_round_scope avoid kept) ].
  exact
    (keeps_scope_compose (normalise_lifted_round avoid kept)
       with_made_closures_called (normalise_lifted_round_scope avoid kept)
       with_made_closures_called_scope).
Qed.

Lemma normalise_lifted_round_calling_up_to (calling : bool)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => normalise_lifted_round_calling calling avoid kept).
Proof.
  destruct calling; [ | exact (normalise_lifted_round_up_to kept) ].
  exact
    (equivariant_up_to_avoiding_after
       (fun avoid => normalise_lifted_round avoid kept)
       with_made_closures_called
       (fun avoid => normalise_lifted_round_scope avoid kept)
       with_made_closures_called_scope with_made_closures_called_equivariant
       (normalise_lifted_round_up_to kept)).
Qed.

(* Those rounds, iterated. *)
Lemma lifted_calling_rounds_scope (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope
    (iterate rounds (normalise_lifted_round_calling calling avoid kept)).
Proof.
  exact
    (keeps_scope_iterate rounds
       (normalise_lifted_round_calling calling avoid kept)
       (normalise_lifted_round_calling_scope calling avoid kept)).
Qed.

Lemma lifted_calling_rounds_up_to (calling : bool) (rounds : nat)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid =>
       iterate rounds (normalise_lifted_round_calling calling avoid kept)).
Proof.
  exact
    (equivariant_up_to_avoiding_iterate rounds
       (fun avoid => normalise_lifted_round_calling calling avoid kept)
       (fun avoid => normalise_lifted_round_calling_scope calling avoid kept)
       (normalise_lifted_round_calling_up_to calling kept)).
Qed.

(* The normalisation with those rounds: the phases of
   [normalise_lifted], with that round ([phases_with]). *)
Lemma normalise_lifted_calling_scope (calling : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope (normalise_lifted_calling calling rounds avoid kept).
Proof.
  exact
    (phases_with_scope rounds
       (normalise_lifted_round_calling calling avoid kept)
       (normalise_lifted_round_calling_scope calling avoid kept)).
Qed.

Lemma normalise_lifted_calling_up_to (calling : bool) (rounds : nat)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => normalise_lifted_calling calling rounds avoid kept).
Proof.
  exact
    (phases_with_up_to rounds
       (fun avoid => normalise_lifted_round_calling calling avoid kept)
       (fun avoid => normalise_lifted_round_calling_scope calling avoid kept)
       (normalise_lifted_round_calling_up_to calling kept)).
Qed.

(* What the closures of the code of a unit are known to be, used in
   the bodies of that code: a call keeps its callee, its arguments and
   its continuations, and what is put in the place of a projection is
   a symbol or a constant. *)
Lemma know_own_closures_scope : keeps_scope know_own_closures.
Proof.
  exact
    (FlambdaScopedProofs.keeps_scope_intro know_own_closures
       FlambdaClosureOfCodeProofs.know_own_closures_names).
Qed.

Lemma know_own_closures_rewrite_equivariant :
  rewrite_equivariant know_own_closures.
Proof. exact FlambdaClosureOfCodeProofs.know_own_closures_equivariant. Qed.

(* A round of FlambdaValidateSpec.normalise_lifted_own: the round of
   [normalise_lifted_calling] and then, where [own] is set, that. *)
Lemma normalise_lifted_round_own_scope (calling own : bool)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope (normalise_lifted_round_own calling own avoid kept).
Proof.
  destruct own;
    [ | exact (normalise_lifted_round_calling_scope calling avoid kept) ].
  exact
    (keeps_scope_compose (normalise_lifted_round_calling calling avoid kept)
       know_own_closures
       (normalise_lifted_round_calling_scope calling avoid kept)
       know_own_closures_scope).
Qed.

Lemma normalise_lifted_round_own_up_to (calling own : bool)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => normalise_lifted_round_own calling own avoid kept).
Proof.
  destruct own;
    [ | exact (normalise_lifted_round_calling_up_to calling kept) ].
  exact
    (equivariant_up_to_avoiding_after
       (fun avoid => normalise_lifted_round_calling calling avoid kept)
       know_own_closures
       (fun avoid => normalise_lifted_round_calling_scope calling avoid kept)
       know_own_closures_scope know_own_closures_rewrite_equivariant
       (normalise_lifted_round_calling_up_to calling kept)).
Qed.

(* Those rounds, iterated. *)
Lemma lifted_own_rounds_scope (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope
    (iterate rounds (normalise_lifted_round_own calling own avoid kept)).
Proof.
  exact
    (keeps_scope_iterate rounds
       (normalise_lifted_round_own calling own avoid kept)
       (normalise_lifted_round_own_scope calling own avoid kept)).
Qed.

Lemma lifted_own_rounds_up_to (calling own : bool) (rounds : nat)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid =>
       iterate rounds (normalise_lifted_round_own calling own avoid kept)).
Proof.
  exact
    (equivariant_up_to_avoiding_iterate rounds
       (fun avoid => normalise_lifted_round_own calling own avoid kept)
       (fun avoid => normalise_lifted_round_own_scope calling own avoid kept)
       (normalise_lifted_round_own_up_to calling own kept)).
Qed.

(* The normalisation with those rounds: the phases of
   [normalise_lifted], with that round ([phases_with]). *)
Lemma normalise_lifted_own_scope (calling own : bool) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) :
  keeps_scope (normalise_lifted_own calling own rounds avoid kept).
Proof.
  exact
    (phases_with_scope rounds
       (normalise_lifted_round_own calling own avoid kept)
       (normalise_lifted_round_own_scope calling own avoid kept)).
Qed.

Lemma normalise_lifted_own_up_to (calling own : bool) (rounds : nat)
    (kept : list code_id) :
  equivariant_up_to_avoiding
    (fun avoid => normalise_lifted_own calling own rounds avoid kept).
Proof.
  exact
    (phases_with_up_to rounds
       (fun avoid => normalise_lifted_round_own calling own avoid kept)
       (fun avoid => normalise_lifted_round_own_scope calling own avoid kept)
       (normalise_lifted_round_own_up_to calling own kept)).
Qed.

(* The liftings in turn keep scope. *)
Lemma lifted_in_turn_scope (rounds : nat) (avoid : list continuation)
    (kept : list code_id) (Ls : list lifting) :
  keeps_scope (lifted_in_turn rounds avoid Ls kept).
Proof.
  induction Ls as [ | L Ls IH]; intros e Hclosed; cbn [lifted_in_turn].
  - destruct (read_known_slots_anywhere_scope e Hclosed)
      as (Hclosed1 & Hv1 & Hk1).
    destruct (demote_and_tidy_known_scope rounds _ Hclosed1)
      as (Hclosed2 & Hv2 & Hk2).
    split; [ exact Hclosed2 | split ].
    + intros x Hx. apply Hv1, Hv2, Hx.
    + intros k Hk. apply Hk1, Hk2, Hk.
  - destruct (with_lifting_scope L e Hclosed) as (Hclosed1 & Hv1 & Hk1).
    destruct
      (normalise_lifted_own_scope (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
         Hclosed1)
      as (Hclosed2 & Hv2 & Hk2).
    destruct (IH _ Hclosed2) as (Hclosed3 & Hv3 & Hk3).
    split; [ exact Hclosed3 | split ].
    + intros x Hx. apply Hv1, Hv2, Hv3, Hx.
    + intros k Hk. apply Hk1, Hk2, Hk3, Hk.
Qed.

Theorem lifted_in_turn_keeps_scope : lifted_in_turn_keeps_scope_statement.
Proof. exact lifted_in_turn_scope. Qed.

(* A lifting renamed by two renamings that rename alike the
   continuations of a list in which the two of its unit are. *)
Lemma rename_lifting_alike (r r' : renaming) (L : lifting)
    (avoid : list continuation) :
  rename_conts_alike r r' avoid ->
  among (unit_return (lifting_unit L)) avoid = true ->
  among (unit_exn (lifting_unit L)) avoid = true ->
  rename_lifting r L = rename_lifting r' L.
Proof.
  intros Halike Hret Hexn. unfold rename_lifting, rename_unit_interface.
  rewrite (Halike _ Hret), (Halike _ Hexn). reflexivity.
Qed.

(* The liftings in turn, of a term under a renaming that does to its
   free names, and to the continuations to avoid, what the renaming of
   the liftings does; the continuations of the unit of each lifting
   are among those to avoid. *)
Lemma lifted_in_turn_up_to (r : renaming) (rounds : nat)
    (avoid : list continuation) (kept : list code_id) (Ls : list lifting) :
  (∀ L : lifting,
    In L Ls ->
    among (unit_return (lifting_unit L)) avoid = true
    ∧ among (unit_exn (lifting_unit L)) avoid = true) ->
  ∀ (r1 : renaming) (e : expr),
    code_closed e = true ->
    renames_alike r r1 e ->
    rename_conts_alike r r1 avoid ->
    ∃ r' : renaming,
      renames_alike r r' (lifted_in_turn rounds avoid Ls kept e)
      ∧ rename_conts_alike r r' avoid
      ∧ lifted_in_turn rounds (map (rename_continuation r) avoid)
          (map (rename_lifting r) Ls) kept (rename_expr r1 e)
        = rename_expr r' (lifted_in_turn rounds avoid Ls kept e).
Proof.
  induction Ls as [ | L Ls IH]; intros Hunits r1 e Hclosed Halike Havoid;
    cbn [map lifted_in_turn].
  - exists r1. split; [ | split; [ exact Havoid | ] ].
    + destruct
        (renames_alike_kept read_known_slots_anywhere r r1 e
           read_known_slots_anywhere_scope Hclosed Halike)
        as (Hclosed1 & Halike1).
      exact
        (proj2
           (renames_alike_kept (demote_and_tidy_known rounds) r r1 _
              (demote_and_tidy_known_scope rounds) Hclosed1 Halike1)).
    + rewrite FlambdaKnownSlotsProofs.read_known_slots_anywhere_equivariant.
      apply demote_and_tidy_known_equivariant.
  - destruct (Hunits L (or_introl eq_refl)) as [Hret Hexn].
    rewrite (rename_lifting_alike r r1 L avoid Havoid Hret Hexn),
      with_lifting_renamed.
    destruct
      (renames_alike_kept (with_lifting L) r r1 e (with_lifting_scope L)
         Hclosed Halike)
      as (Hclosed1 & Halike1).
    change (made_closures_called (rename_lifting r1 L))
      with (made_closures_called L).
    change (own_closures_known (rename_lifting r1 L))
      with (own_closures_known L).
    destruct
      (normalise_lifted_own_up_to (made_closures_called L)
         (own_closures_known L) rounds kept r1 avoid (with_lifting L e)
         Hclosed1)
      as (r2 & Halike2 & Havoid2 & Heq).
    cbv beta in Heq, Halike2.
    rewrite (map_rename_conts_alike r r1 avoid Havoid) in Heq.
    rewrite Heq.
    destruct
      (normalise_lifted_own_scope (made_closures_called L)
         (own_closures_known L) rounds avoid kept (with_lifting L e)
         Hclosed1)
      as (Hclosed2 & _ & _).
    apply (IH (fun L' HL' => Hunits L' (or_intror HL')) r2).
    + exact Hclosed2.
    + exact
        (renames_alike_then
           (normalise_lifted_own (made_closures_called L)
              (own_closures_known L) rounds avoid kept)
           r r1 r2 (with_lifting L e)
           (normalise_lifted_own_scope (made_closures_called L)
              (own_closures_known L) rounds avoid kept)
           Hclosed1 Halike1 Halike2).
    + intros k Hk. rewrite (Havoid2 k Hk). apply Havoid, Hk.
Qed.

Theorem lifted_in_turn_equivariant_up_to :
  lifted_in_turn_equivariant_up_to_statement.
Proof. exact lifted_in_turn_up_to. Qed.

(* The continuations given for the loops, renamed by two renamings
   that rename them alike. *)
Lemma rename_loopified_alike (r r' : renaming) (G : loopified) :
  rename_conts_alike r r' (map snd G) ->
  rename_loopified r G = rename_loopified r' G.
Proof.
  intros Halike. unfold rename_loopified. apply map_ext_in.
  intros [id k] Hin. cbn [fst snd]. f_equal. symmetry.
  apply Halike, among_In, in_map_iff. exists (id, k).
  split; [ reflexivity | exact Hin ].
Qed.

(* The continuations of the renamed loops are those of the loops,
   renamed. *)
Lemma loop_continuations_renamed (r : renaming) (G : loopified) :
  map snd (rename_loopified r G) = map (rename_continuation r) (map snd G).
Proof. unfold rename_loopified. rewrite !map_map. reflexivity. Qed.

(* The continuations of the units of the liftings: those of the renamed
   liftings are those of the liftings, renamed; and the two of the unit
   of each lifting are among them. *)
Lemma unit_continuations_renamed (r : renaming) (Ls : list lifting) :
  unit_continuations (map (rename_lifting r) Ls)
  = map (rename_continuation r) (unit_continuations Ls).
Proof.
  unfold unit_continuations.
  induction Ls as [ | L Ls IH]; [ reflexivity | ].
  cbn [map flat_map app]. rewrite IH. reflexivity.
Qed.

Lemma unit_continuations_has (Ls : list lifting) :
  ∀ L : lifting,
    In L Ls ->
    among (unit_return (lifting_unit L)) (unit_continuations Ls) = true
    ∧ among (unit_exn (lifting_unit L)) (unit_continuations Ls) = true.
Proof.
  intros L HL.
  split; apply among_In; unfold unit_continuations; apply in_flat_map;
    exists L; (split; [ exact HL | ]).
  - left. reflexivity.
  - right. left. reflexivity.
Qed.

(* The placing of continuations keeps scope, as its pass does. *)
Lemma place_conts_keeps_scope : keeps_scope place_conts.
Proof.
  exact
    (keeps_scope_compose sink_conts_to_uses
       (rewrite_rec_handlers sink_conts_to_uses)
       FlambdaLoopsScopeProofs.sink_conts_to_uses_keeps_scope
       (FlambdaLoopsScopeProofs.rewrite_rec_handlers_keeps_scope _
          FlambdaLoopsScopeProofs.sink_conts_to_uses_keeps_scope)).
Qed.

Theorem placed_keeps_scope : placed_keeps_scope_statement.
Proof.
  exact
    (keeps_scope_compose place_conts (rewrite_code place_conts)
       place_conts_keeps_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          place_conts_keeps_scope)).
Qed.

(* The tidying of the term before that follows the placing keeps
   scope: each rewrite of its rounds does. *)
Lemma give_then_unalias_keeps_scope :
  keeps_scope
    (fun e => unalias_rec_params (unalias_params (give_scrutinees e))).
Proof.
  exact
    (keeps_scope_compose (fun e => unalias_params (give_scrutinees e))
       unalias_rec_params
       (keeps_scope_compose give_scrutinees unalias_params
          FlambdaStaticsScopeProofs.give_scrutinees_keeps_scope
          FlambdaStaticsScopeProofs.unalias_params_keeps_scope)
       FlambdaStaticsScopeProofs.unalias_rec_params_keeps_scope).
Qed.

Lemma tidy_expr_before_keeps_scope : keeps_scope tidy_expr_before.
Proof.
  unfold tidy_expr_before.
  refine
    (keeps_scope_compose _ (remove_unused_defined_lets []) _
       (FlambdaKeepsScopeProofs.remove_unused_defined_lets_keeps_scope [])).
  refine
    (keeps_scope_compose _ remove_unused_lets _
       FlambdaKeepsScopeProofs.remove_unused_lets_keeps_scope).
  refine
    (keeps_scope_compose _ (follow_aliases_in_loops []) _
       FlambdaLoopsScopeProofs.follow_aliases_in_loops_keeps_scope).
  refine
    (keeps_scope_compose _ (follow_aliases []) _
       FlambdaKeepsScopeProofs.follow_aliases_keeps_scope).
  refine
    (keeps_scope_compose _ share_known_primitives _
       FlambdaKeepsScopeProofs.share_known_primitives_keeps_scope).
  refine
    (keeps_scope_compose _ inline_tail_conts _
       FlambdaKeepsScopeProofs.inline_tail_conts_keeps_scope).
  refine
    (keeps_scope_compose _ sink_let_conts _
       FlambdaLoopsScopeProofs.sink_let_conts_keeps_scope).
  refine
    (keeps_scope_compose _ remove_unused_conts _
       FlambdaKeepsScopeProofs.remove_unused_conts_keeps_scope).
  refine
    (keeps_scope_compose _ merge_switches _
       FlambdaKeepsScopeProofs.merge_switches_keeps_scope).
  refine
    (keeps_scope_compose _ collapse_switches _
       FlambdaKeepsScopeProofs.collapse_switches_keeps_scope).
  refine
    (keeps_scope_compose _ forward_applications _
       FlambdaLoopsScopeProofs.forward_applications_keeps_scope).
  refine
    (keeps_scope_compose _ fold_constants _
       FlambdaKeepsScopeProofs.fold_constants_keeps_scope).
  exact
    (keeps_scope_compose
       (fun e => unalias_rec_params (unalias_params (give_scrutinees e)))
       numbers_in_arms give_then_unalias_keeps_scope
       FlambdaStaticsScopeProofs.numbers_in_arms_keeps_scope).
Qed.

Lemma tidy_code_before_keeps_scope : keeps_scope tidy_code_before.
Proof.
  exact
    (keeps_scope_compose tidy_expr_before
       (rewrite_rec_handlers tidy_expr_before) tidy_expr_before_keeps_scope
       (FlambdaLoopsScopeProofs.rewrite_rec_handlers_keeps_scope _
          tidy_expr_before_keeps_scope)).
Qed.

Lemma tidy_before_round_keeps_scope : keeps_scope tidy_before_round.
Proof.
  exact
    (keeps_scope_compose tidy_code_before (rewrite_code tidy_code_before)
       tidy_code_before_keeps_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          tidy_code_before_keeps_scope)).
Qed.

Theorem tidied_before_keeps_scope : tidied_before_keeps_scope_statement.
Proof.
  intros rounds.
  exact
    (keeps_scope_compose
       (fun e => iterate rounds tidy_before_round (placed e)) groups_up
       (keeps_scope_compose placed (iterate rounds tidy_before_round)
          placed_keeps_scope
          (keeps_scope_iterate rounds tidy_before_round
             tidy_before_round_keeps_scope))
       FlambdaStaticsScopeProofs.groups_up_keeps_scope).
Qed.

(* The calls of the term before taken on when its rounds are over
   (FlambdaValidateSpec.with_returns): taken or not, it commutes with
   renaming and keeps scope; and whether a term has a call that would
   be taken on is the same of the term renamed. *)
Lemma take_returns_on_equivariant : rewrite_equivariant take_returns_on.
Proof.
  intros r e. unfold take_returns_on.
  rewrite (forward_applications_with_equivariant true r e),
    remove_unused_conts_equivariant.
  apply rewrite_rec_handlers_equivariant, remove_unused_conts_equivariant.
Qed.

Lemma returns_taken_on_equivariant : rewrite_equivariant returns_taken_on.
Proof.
  intros r e. unfold returns_taken_on. rewrite take_returns_on_equivariant.
  apply rewrite_code_equivariant, take_returns_on_equivariant.
Qed.

Theorem with_returns_equivariant : with_returns_equivariant_statement.
Proof.
  intros taken r e.
  destruct taken; [ apply returns_taken_on_equivariant | reflexivity ].
Qed.

Lemma forwarded_in_unit_equivariant (calls : bool) :
  rewrite_equivariant (forwarded_in_unit calls).
Proof.
  intros r e. unfold forwarded_in_unit.
  rewrite (forward_applications_with_equivariant calls r e).
  apply rewrite_code_equivariant, forward_applications_with_equivariant.
Qed.

Theorem keeps_a_return_equivariant : keeps_a_return_equivariant_statement.
Proof.
  intros r e. unfold keeps_a_return.
  rewrite !forwarded_in_unit_equivariant, expr_eqb_renamed. reflexivity.
Qed.

Lemma take_returns_on_keeps_scope : keeps_scope take_returns_on.
Proof.
  exact
    (keeps_scope_compose
       (fun e => remove_unused_conts (forward_applications_with true e))
       (rewrite_rec_handlers remove_unused_conts)
       (keeps_scope_compose (forward_applications_with true)
          remove_unused_conts
          (FlambdaLoopsScopeProofs.forward_applications_with_keeps_scope
             true)
          FlambdaKeepsScopeProofs.remove_unused_conts_keeps_scope)
       (FlambdaLoopsScopeProofs.rewrite_rec_handlers_keeps_scope _
          FlambdaKeepsScopeProofs.remove_unused_conts_keeps_scope)).
Qed.

Lemma returns_taken_on_keeps_scope : keeps_scope returns_taken_on.
Proof.
  exact
    (keeps_scope_compose take_returns_on (rewrite_code take_returns_on)
       take_returns_on_keeps_scope
       (FlambdaKeepsScopeProofs.rewrite_code_keeps_scope _
          take_returns_on_keeps_scope)).
Qed.

Theorem with_returns_keeps_scope : with_returns_keeps_scope_statement.
Proof.
  intros taken.
  destruct taken; [ exact returns_taken_on_keeps_scope | ].
  intros e Hclosed.
  split; [ exact Hclosed | split; intros name Hname; exact Hname ].
Qed.

(* The term before normalised is, under the renaming, that term under a
   second renaming; the term before with the liftings made in turn is
   that term under a third, since each normalisation after a lifting
   avoids the continuations of the units of the liftings. Whether
   the calls of the term before are taken on is the same for the
   pair renamed, the term after being under the first renaming. The
   taking on of those calls, the placing
   of continuations and the tidying, which come next, the removal of
   static bindings,
   the removal of value slots, the moving up of the bindings of code
   and the ordering of static bindings commute with each of them,
   whatever renames the pattern, and keep scope; so the
   two renamings rename the free names of each term that is compared
   alike. The term after is under the first renaming throughout: its
   normalisation and the putting of its code first commute with it.
   The term before may be under a renaming of its own from the start,
   if that renaming does to its free names, and to the continuations
   of the units and of the loops, what the first does. *)
Theorem validate_unit_renamed_alike : validate_unit_renamed_alike_statement.
Proof.
  intros r r1 rounds unused slots G Ls kept before after Hclosed Halike1
    Hheld.
  unfold validate_unit.
  rewrite (expr_eqb_renamed_alike r r1 before after Hclosed Halike1).
  destruct (expr_eqb [] [] before after); [ reflexivity | ].
  cbv zeta.
  assert (Hunits : rename_conts_alike r r1 (unit_continuations Ls)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. reflexivity. }
  assert (Hloops : rename_conts_alike r r1 (map snd G)).
  { intros k Hk. apply Hheld. rewrite among_app, Hk. apply orb_true_r. }
  rewrite FlambdaStubCallsProofs.demote_stub_calls_equivariant,
    FlambdaNeverReturnsProofs.cut_returns_unit_equivariant,
    (rename_loopified_alike r r1 G Hloops), loopify_unit_equivariant.
  destruct
    (renames_alike_kept demote_stub_calls r r1 before
       FlambdaStaticsScopeProofs.demote_stub_calls_keeps_scope Hclosed
       Halike1)
    as (Hclosed0 & Halike0).
  destruct
    (renames_alike_kept (cut_returns []) r r1 (demote_stub_calls before)
       (FlambdaNeverReturnsScopeProofs.cut_returns_keeps_scope [])
       Hclosed0 Halike0)
    as (Hclosed00 & Halike00).
  destruct
    (renames_alike_kept (loopify_unit G) r r1
       (cut_returns [] (demote_stub_calls before))
       (FlambdaLoopsScopeProofs.loopify_unit_keeps_scope G) Hclosed00
       Halike00)
    as (Hclosed' & Halike1').
  clear Hclosed Halike1 Hheld Hloops Hclosed0 Halike0 Hclosed00 Halike00.
  revert Hclosed' Halike1'.
  generalize (loopify_unit G (cut_returns [] (demote_stub_calls before))).
  clear before.
  intros before Hclosed Halike1.
  destruct (normalise_before_equivariant_up_to rounds kept r1 before Hclosed)
    as (r' & Halike_b & Heq).
  pose proof
    (renames_alike_then (normalise_before rounds kept) r r1 r' before
       (normalise_before_keeps_scope rounds kept) Hclosed Halike1 Halike_b)
    as Halike0.
  destruct (normalise_before_keeps_scope rounds kept before Hclosed)
    as (Hclosed_b0 & _ & _).
  destruct
    (lifted_in_turn_up_to r rounds (unit_continuations Ls) kept Ls
       (unit_continuations_has Ls) r1 before Hclosed Halike1 Hunits)
    as (r'' & Halike0' & Havoid' & Heq').
  destruct
    (lifted_in_turn_scope rounds (unit_continuations Ls) kept Ls before
       Hclosed)
    as (Hclosed_l0 & _ & _).
  rewrite unit_continuations_renamed.
  rewrite Heq, Heq', normalise_after_equivariant, tidied_after_equivariant,
    keeps_a_return_equivariant, !with_returns_equivariant,
    !tidied_before_equivariant,
    without_head_statics_equivariant,
    without_slots_equivariant, !without_statics_equivariant,
    with_code_first_equivariant, !float_code_rewrite_equivariant,
    (with_statics_first_renamed_alike r r Ls _ (fun k _ => eq_refl)),
    (with_statics_first_renamed_alike r r'' Ls _ Havoid'),
    !order_statics_as_equivariant.
  set (a := tidied_after rounds (normalise_after rounds kept after)).
  generalize (negb (keeps_a_return a)). intros taken.
  destruct
    (renames_alike_kept (with_returns taken) r r'
       (normalise_before rounds kept before)
       (with_returns_keeps_scope taken) Hclosed_b0 Halike0)
    as (Hclosed_bt & Halike_bt).
  destruct
    (renames_alike_kept (tidied_before rounds) r r'
       (with_returns taken (normalise_before rounds kept before))
       (tidied_before_keeps_scope rounds) Hclosed_bt Halike_bt)
    as (Hclosed_b & Halike).
  destruct
    (renames_alike_kept (with_returns taken) r r''
       (lifted_in_turn rounds (unit_continuations Ls) Ls kept before)
       (with_returns_keeps_scope taken) Hclosed_l0 Halike0')
    as (Hclosed_lt & Halike_lt).
  destruct
    (renames_alike_kept (tidied_before rounds) r r''
       (with_returns taken
          (lifted_in_turn rounds (unit_continuations Ls) Ls kept before))
       (tidied_before_keeps_scope rounds) Hclosed_lt Halike_lt)
    as (Hclosed_l & Halike').
  set
    (b :=
      tidied_before rounds
        (with_returns taken (normalise_before rounds kept before)))
    in *.
  set
    (l :=
      tidied_before rounds
        (with_returns taken
           (lifted_in_turn rounds (unit_continuations Ls) Ls kept before)))
    in *.
  set (a' := without_head_statics unused a).
  set (c := float_code (with_code_first a')).
  set (c' := with_statics_first Ls a').
  destruct
    (renames_alike_kept (order_statics_as a) r r' b (order_statics_as_scope a)
       Hclosed_b Halike)
    as (Hclosed_ordered & Halike_ordered).
  destruct
    (renames_alike_kept (without_statics unused) r r' b
       (without_statics_scope unused) Hclosed_b Halike)
    as (Hclosed_without & Halike_without).
  destruct
    (renames_alike_kept (order_statics_as a) r r' (without_statics unused b)
       (order_statics_as_scope a) Hclosed_without Halike_without)
    as (Hclosed_both & Halike_both).
  destruct
    (renames_alike_kept (order_statics_as a') r r' (without_statics unused b)
       (order_statics_as_scope a') Hclosed_without Halike_without)
    as (Hclosed_both' & Halike_both').
  destruct
    (renames_alike_kept (without_slots slots) r r'' l
       (without_slots_scope slots) Hclosed_l Halike')
    as (Hclosed_slots & Halike_slots).
  destruct
    (renames_alike_kept (without_statics unused) r r''
       (without_slots slots l) (without_statics_scope unused) Hclosed_slots
       Halike_slots)
    as (Hclosed_lifted_without & Halike_lifted_without).
  destruct
    (renames_alike_kept float_code r r''
       (without_statics unused (without_slots slots l)) float_code_scope
       Hclosed_lifted_without Halike_lifted_without)
    as (Hclosed_floated & Halike_floated).
  destruct
    (renames_alike_kept (order_statics_as c) r r''
       (float_code (without_statics unused (without_slots slots l)))
       (order_statics_as_scope c) Hclosed_floated Halike_floated)
    as (Hclosed_last & Halike_last).
  destruct
    (renames_alike_kept (with_statics_first Ls) r r''
       (without_statics unused (without_slots slots l))
       (with_statics_first_scope Ls) Hclosed_lifted_without
       Halike_lifted_without)
    as (Hclosed_first & Halike_first).
  destruct
    (renames_alike_kept (order_statics_as c') r r''
       (with_statics_first Ls
          (without_statics unused (without_slots slots l)))
       (order_statics_as_scope c') Hclosed_first Halike_first)
    as (Hclosed_last' & Halike_last').
  rewrite
    (expr_eqb_renamed_alike r r' (order_statics_as a b) a Hclosed_ordered
       Halike_ordered),
    (expr_eqb_renamed_alike r r'
       (order_statics_as a (without_statics unused b)) a Hclosed_both
       Halike_both),
    (expr_eqb_renamed_alike r r'
       (order_statics_as a' (without_statics unused b)) a' Hclosed_both'
       Halike_both'),
    (expr_eqb_renamed_alike r r''
       (order_statics_as c
          (float_code (without_statics unused (without_slots slots l))))
       c Hclosed_last Halike_last),
    (expr_eqb_renamed_alike r r''
       (order_statics_as c'
          (with_statics_first Ls
             (without_statics unused (without_slots slots l))))
       c' Hclosed_last' Halike_last').
  reflexivity.
Qed.

Theorem validate_unit_equivariant : validate_unit_equivariant_statement.
Proof.
  intros r rounds unused slots G Ls kept before after Hclosed.
  apply validate_unit_renamed_alike; [ exact Hclosed | | ].
  - split; intros name _; reflexivity.
  - intros k _. reflexivity.
Qed.
