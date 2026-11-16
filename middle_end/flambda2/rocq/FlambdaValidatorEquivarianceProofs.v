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

Lemma named_may_go_renamed (r : renaming) (d : named) :
  named_may_go (rename_named r d) = named_may_go d.
Proof. destruct d; reflexivity. Qed.

Fixpoint remove_unused_lets_equivariant_aux (r : renaming) (e : expr) :
  remove_unused_lets (rename_expr r e) = rename_expr r (remove_unused_lets e)
with remove_unused_lets_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  remove_unused_lets_in_handler (rename_handler r h)
  = rename_handler r (remove_unused_lets_in_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite remove_unused_lets_equivariant_aux, var_free_b_equivariant_aux,
        named_may_go_renamed.
      destruct
        (var_free_b x (remove_unused_lets body) || negb (named_may_go d));
        reflexivity.
    + rewrite remove_unused_lets_equivariant_aux. reflexivity.
    + rewrite remove_unused_lets_in_handler_equivariant,
        remove_unused_lets_equivariant_aux.
      reflexivity.
    + rewrite remove_unused_lets_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite remove_unused_lets_equivariant_aux. reflexivity.
Qed.

Theorem remove_unused_lets_equivariant :
  remove_unused_lets_equivariant_statement.
Proof. exact remove_unused_lets_equivariant_aux. Qed.

Lemma named_is_total_renamed (r : renaming) (kind : flambda_kind)
    (d : named) :
  named_is_total kind (rename_named r d) = named_is_total kind d.
Proof. destruct d as [[x | sym | c] | | | | | | ]; reflexivity. Qed.

Fixpoint remove_unused_total_lets_equivariant_aux (r : renaming) (e : expr) :
  remove_unused_total_lets (rename_expr r e)
  = rename_expr r (remove_unused_total_lets e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite remove_unused_total_lets_equivariant_aux,
      var_free_b_equivariant_aux, named_is_total_renamed.
    destruct
      (named_is_total kind d
       && negb (var_free_b x (remove_unused_total_lets body)));
      reflexivity.
  - rewrite remove_unused_total_lets_equivariant_aux. reflexivity.
  - rewrite !remove_unused_total_lets_equivariant_aux. reflexivity.
  - rewrite remove_unused_total_lets_equivariant_aux. reflexivity.
Qed.

Theorem remove_unused_total_lets_equivariant :
  remove_unused_total_lets_equivariant_statement.
Proof. exact remove_unused_total_lets_equivariant_aux. Qed.

(* A variable is found among the renamed variables in scope where it was
   found, at the same kind: a renaming is injective. *)
Lemma lookup_rename_kinds_in_scope (r : renaming) (x : variable)
    (kinds : kinds_in_scope) :
  lookup (rename_variable r x) (rename_kinds_in_scope r kinds)
  = lookup x kinds.
Proof.
  induction kinds as [ | [y kind] kinds IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma map_fst_rename_kinds_in_scope (r : renaming) (kinds : kinds_in_scope) :
  map fst (rename_kinds_in_scope r kinds)
  = map (rename_variable r) (map fst kinds).
Proof.
  unfold rename_kinds_in_scope. rewrite !map_map. reflexivity.
Qed.

Lemma kind_of_simple_in_scope_equivariant (r : renaming)
    (kinds : kinds_in_scope) (s : simple) :
  kind_of_simple_in_scope (rename_kinds_in_scope r kinds) (rename_simple r s)
  = kind_of_simple_in_scope kinds s.
Proof.
  destruct s; simpl; [ apply lookup_rename_kinds_in_scope | | ];
    reflexivity.
Qed.

Lemma named_defined_equivariant (r : renaming) (kinds : kinds_in_scope)
    (kind : flambda_kind) (d : named) :
  named_defined (rename_kinds_in_scope r kinds) kind (rename_named r d)
  = named_defined kinds kind d.
Proof.
  destruct d; simpl; try reflexivity;
    [ rewrite kind_of_simple_in_scope_equivariant; reflexivity
    | rewrite map_fst_rename_kinds_in_scope,
        value_slots_in_scope_equivariant;
      reflexivity
    | rewrite map_fst_rename_kinds_in_scope,
        value_slots_in_scope_equivariant;
      reflexivity ].
Qed.

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

Fixpoint remove_unused_defined_lets_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ kinds : kinds_in_scope,
    remove_unused_defined_lets (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets kinds e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros kinds; simpl; try reflexivity.
  - change ((rename_variable r x, kind) :: rename_kinds_in_scope r kinds)
      with (rename_kinds_in_scope r ((x, kind) :: kinds)).
    rewrite remove_unused_defined_lets_equivariant_aux,
      named_defined_equivariant, var_free_b_equivariant_aux.
    destruct
      (named_defined kinds kind d
       && negb
            (var_free_b x
               (remove_unused_defined_lets ((x, kind) :: kinds) body)));
      reflexivity.
  - rewrite remove_unused_defined_lets_equivariant_aux. reflexivity.
  - rewrite params_in_scope_equivariant,
      !remove_unused_defined_lets_equivariant_aux.
    reflexivity.
  - rewrite remove_unused_defined_lets_equivariant_aux. reflexivity.
Qed.

Theorem remove_unused_defined_lets_equivariant :
  remove_unused_defined_lets_equivariant_statement.
Proof.
  intros r kinds e. apply remove_unused_defined_lets_equivariant_aux.
Qed.

(* The projections of function slots that nothing uses. *)

Lemma lookup_rename_known_sets (r : renaming) (x : variable)
    (known : known_sets) :
  lookup (rename_variable r x) (rename_known_sets r known) = lookup x known.
Proof.
  induction known as [ | [y entry] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb, IH. reflexivity.
Qed.

Lemma forget_set_renamed (r : renaming) (x : variable) (known : known_sets) :
  forget_set (rename_variable r x) (rename_known_sets r known)
  = rename_known_sets r (forget_set x known).
Proof.
  induction known as [ | [y entry] known IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); simpl; rewrite IH;
    reflexivity.
Qed.

Lemma forget_sets_renamed (r : renaming) (xs : list variable) :
  ∀ known : known_sets,
    forget_sets (map (rename_variable r) xs) (rename_known_sets r known)
    = rename_known_sets r (forget_sets xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_set_renamed. apply IH.
Qed.

Lemma sets_under_let_renamed (r : renaming) (known : known_sets)
    (x : variable) (d : named) :
  sets_under_let (rename_known_sets r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_sets r (sets_under_let known x d).
Proof.
  unfold sets_under_let. rewrite forget_set_renamed.
  destruct d; reflexivity.
Qed.

Lemma projection_defined_renamed (r : renaming) (known : known_sets)
    (d : named) :
  projection_defined (rename_known_sets r known) (rename_named r d)
  = projection_defined known d.
Proof.
  destruct d as [s | p args | code slots | fslot decls slots | slot closure
    | pfrom slot closure | move_from move_to closure];
    try reflexivity.
  destruct closure as [x | sym | c]; try reflexivity.
  simpl. rewrite lookup_rename_known_sets. reflexivity.
Qed.

Fixpoint remove_projections_knowing_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ known : known_sets,
    remove_projections_knowing (rename_known_sets r known) (rename_expr r e)
    = rename_expr r (remove_projections_knowing known e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros known; simpl; try reflexivity.
  - rewrite sets_under_let_renamed,
      remove_projections_knowing_equivariant_aux,
      projection_defined_renamed, var_free_b_equivariant_aux.
    destruct
      (kind_eqb kind Value
       && projection_defined known d
       && negb
            (var_free_b x
               (remove_projections_knowing (sets_under_let known x d) body)));
      reflexivity.
  - rewrite remove_projections_knowing_equivariant_aux. reflexivity.
  - rewrite forget_sets_renamed, !remove_projections_knowing_equivariant_aux.
    reflexivity.
  - rewrite remove_projections_knowing_equivariant_aux. reflexivity.
Qed.

Theorem remove_projections_knowing_equivariant :
  remove_projections_knowing_equivariant_statement.
Proof.
  intros r known e. apply remove_projections_knowing_equivariant_aux.
Qed.

Theorem remove_unused_projections_equivariant :
  remove_unused_projections_equivariant_statement.
Proof.
  intros r e. exact (remove_projections_knowing_equivariant_aux r e []).
Qed.

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

Lemma binds_code_renamed (r : renaming) (binding : static_binding) :
  binds_code (rename_static_binding r binding) = binds_code binding.
Proof. destruct binding; reflexivity. Qed.

Lemma code_first_equivariant (r : renaming) (bindings : list static_binding) :
  code_first (map (rename_static_binding r) bindings)
  = map (rename_static_binding r) (code_first bindings).
Proof.
  unfold code_first. rewrite map_app.
  rewrite (filter_map_with _ _ binds_code) by apply binds_code_renamed.
  rewrite (filter_map_with _ _ (fun binding => negb (binds_code binding)))
    by (intros binding; rewrite binds_code_renamed; reflexivity).
  reflexivity.
Qed.

Fixpoint merge_statics_equivariant_aux (r : renaming) (e : expr) :
  merge_statics (rename_expr r e) = rename_expr r (merge_statics e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite merge_statics_equivariant_aux. reflexivity.
  - rewrite merge_statics_equivariant_aux.
    destruct (merge_statics body); simpl;
      rewrite <- ?map_app, code_first_equivariant; reflexivity.
  - rewrite !merge_statics_equivariant_aux. reflexivity.
  - rewrite merge_statics_equivariant_aux. reflexivity.
Qed.

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

Fixpoint collapse_switches_equivariant_aux (r : renaming) (e : expr) :
  collapse_switches (rename_expr r e) = rename_expr r (collapse_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite collapse_switches_equivariant_aux. reflexivity.
  - rewrite collapse_switches_equivariant_aux. reflexivity.
  - rewrite !collapse_switches_equivariant_aux. reflexivity.
  - rewrite collapse_switches_equivariant_aux. reflexivity.
  - apply collapse_switch_equivariant.
Qed.

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
Proof.
  unfold arms_apply. apply existsb_map_with. intros arm. cbn [snd].
  apply among_apply_cont_continuations_renamed.
Qed.

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

Fixpoint thread_switches_equivariant_aux (r : renaming) (e : expr) :
  thread_switches (rename_expr r e) = rename_expr r (thread_switches e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    simpl; try reflexivity.
  - rewrite thread_switches_equivariant_aux. reflexivity.
  - rewrite thread_switches_equivariant_aux. reflexivity.
  - rewrite !thread_switches_equivariant_aux.
    exact
      (thread_switch_equivariant r k
         (Handler params metadata (thread_switches handler))
         (thread_switches body)).
  - rewrite thread_switches_equivariant_aux. reflexivity.
Qed.

Theorem thread_switches_equivariant :
  thread_switches_equivariant_statement.
Proof. exact thread_switches_equivariant_aux. Qed.

(* ======================================================================== *)
(* Rewriting the bodies of code                                             *)
(* ======================================================================== *)

Section Rewrite_code.

Variable f : expr -> expr.
Hypothesis f_equivariant : rewrite_equivariant f.

Lemma rewrite_code_body_equivariant (r : renaming) (body : expr) :
  rewrite_code_body f (rename_expr r body)
  = rename_expr r (rewrite_code_body f body).
Proof.
  unfold rewrite_code_body. rewrite f_equivariant, !rename_binds_statics.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f body)));
    reflexivity.
Qed.

Lemma rewrite_code_in_binding_equivariant (r : renaming)
    (binding : static_binding) :
  rewrite_code_in_binding f (rename_static_binding r binding)
  = rename_static_binding r (rewrite_code_in_binding f binding).
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code rewrite_code_in_binding].
  rewrite rewrite_code_body_equivariant. reflexivity.
Qed.

Fixpoint rewrite_code_equivariant_aux (r : renaming) (e : expr) :
  rewrite_code f (rename_expr r e) = rename_expr r (rewrite_code f e)
with rewrite_code_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  rewrite_code_in_handler f (rename_handler r h)
  = rename_handler r (rewrite_code_in_handler f h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite rewrite_code_equivariant_aux. reflexivity.
    + rewrite rewrite_code_equivariant_aux, !map_map. f_equal.
      apply map_ext. exact (rewrite_code_in_binding_equivariant r).
    + rewrite rewrite_code_in_handler_equivariant,
        rewrite_code_equivariant_aux.
      reflexivity.
    + rewrite rewrite_code_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (rewrite_code_in_handler_equivariant r h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite rewrite_code_equivariant_aux. reflexivity.
Qed.

End Rewrite_code.

Theorem rewrite_code_equivariant : rewrite_code_equivariant_statement.
Proof.
  intros f f_equivariant r e.
  apply rewrite_code_equivariant_aux, f_equivariant.
Qed.

(* ======================================================================== *)
(* FlambdaUnusedContsSpec.v                                                 *)
(* ======================================================================== *)

Theorem cont_free_b_equivariant : cont_free_b_equivariant_statement.
Proof. exact cont_free_b_equivariant_aux. Qed.

Fixpoint remove_unused_conts_equivariant_aux (r : renaming) (e : expr) :
  remove_unused_conts (rename_expr r e)
  = rename_expr r (remove_unused_conts e)
with remove_unused_conts_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  remove_unused_conts_in_handler (rename_handler r h)
  = rename_handler r (remove_unused_conts_in_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite remove_unused_conts_equivariant_aux. reflexivity.
    + rewrite remove_unused_conts_equivariant_aux. reflexivity.
    + rewrite remove_unused_conts_equivariant_aux,
        cont_free_b_equivariant_aux,
        remove_unused_conts_in_handler_equivariant.
      destruct (cont_free_b k (remove_unused_conts body)); reflexivity.
    + rewrite remove_unused_conts_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite remove_unused_conts_equivariant_aux. reflexivity.
Qed.

Theorem remove_unused_conts_equivariant :
  remove_unused_conts_equivariant_statement.
Proof. exact remove_unused_conts_equivariant_aux. Qed.

(* ======================================================================== *)
(* FlambdaUnusedStaticsSpec.v                                               *)
(* ======================================================================== *)

Section Dropped.

Variable D : dropped.

Lemma simple_avoids_renamed (r : renaming) (s : simple) :
  simple_avoids D (rename_simple r s) = simple_avoids D s.
Proof. destruct s; reflexivity. Qed.

Lemma simples_avoid_renamed (r : renaming) (args : list simple) :
  simples_avoid D (map (rename_simple r) args) = simples_avoid D args.
Proof.
  unfold simples_avoid. apply forallb_map_with. apply simple_avoids_renamed.
Qed.

Lemma value_slots_avoid_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  value_slots_avoid D
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_avoid D slots.
Proof.
  unfold value_slots_avoid. apply forallb_map_with. intros slot.
  apply simple_avoids_renamed.
Qed.

Lemma named_avoids_renamed (r : renaming) (d : named) :
  named_avoids D (rename_named r d) = named_avoids D d.
Proof.
  destruct d; simpl;
    rewrite ?value_slots_avoid_renamed, ?simples_avoid_renamed,
      ?simple_avoids_renamed;
    reflexivity.
Qed.

Lemma static_const_avoids_renamed (r : renaming) (const : static_const) :
  static_const_avoids D (rename_static_const r const)
  = static_const_avoids D const.
Proof.
  destruct const; simpl;
    rewrite ?value_slots_avoid_renamed, ?simples_avoid_renamed,
      ?simple_avoids_renamed;
    reflexivity.
Qed.

Lemma apply_cont_avoids_renamed (r : renaming) (ac : apply_cont) :
  apply_cont_avoids D (rename_apply_cont r ac) = apply_cont_avoids D ac.
Proof. apply simples_avoid_renamed. Qed.

Lemma apply_avoids_renamed (r : renaming) (ap : apply_expr) :
  apply_avoids D (rename_apply r ap) = apply_avoids D ap.
Proof.
  unfold apply_avoids. simpl.
  rewrite exn_extra_simples_rename, simple_avoids_renamed,
    !simples_avoid_renamed.
  reflexivity.
Qed.

Fixpoint expr_avoids_renamed (r : renaming) (e : expr) :
  expr_avoids D (rename_expr r e) = expr_avoids D e
with handler_avoids_renamed (r : renaming) (h : continuation_handler) :
  handler_avoids D (rename_handler r h) = handler_avoids D h
with binding_avoids_renamed (r : renaming) (binding : static_binding) :
  binding_avoids D (rename_static_binding r binding)
  = binding_avoids D binding
with code_avoids_renamed (r : renaming) (c : code) :
  code_avoids D (rename_code r c) = code_avoids D c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl.
    + rewrite named_avoids_renamed, expr_avoids_renamed. reflexivity.
    + rewrite expr_avoids_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (binding_avoids_renamed r binding), IH. reflexivity.
    + rewrite handler_avoids_renamed, expr_avoids_renamed. reflexivity.
    + rewrite expr_avoids_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_avoids_renamed r h), IH. reflexivity.
    + apply apply_avoids_renamed.
    + apply apply_cont_avoids_renamed.
    + rewrite simple_avoids_renamed. f_equal.
      apply forallb_map_with. intros arm. apply apply_cont_avoids_renamed.
    + reflexivity.
  - destruct h as [params metadata handler]. simpl. apply expr_avoids_renamed.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    + rewrite code_avoids_renamed. reflexivity.
    + rewrite static_const_avoids_renamed. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply expr_avoids_renamed.
Qed.

Lemma binding_kept_renamed (r : renaming) (binding : static_binding) :
  binding_kept D (rename_static_binding r binding) = binding_kept D binding.
Proof. destruct binding; reflexivity. Qed.

Fixpoint erase_expr_equivariant_aux (r : renaming) (e : expr) :
  erase_expr D (rename_expr r e) = rename_expr r (erase_expr D e)
with erase_handler_equivariant (r : renaming) (h : continuation_handler) :
  erase_handler D (rename_handler r h) = rename_handler r (erase_handler D h)
with erase_binding_equivariant (r : renaming) (binding : static_binding) :
  erase_binding D (rename_static_binding r binding)
  = rename_static_binding r (erase_binding D binding)
with erase_code_equivariant (r : renaming) (c : code) :
  erase_code D (rename_code r c) = rename_code r (erase_code D c).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite erase_expr_equivariant_aux. reflexivity.
    + rewrite erase_expr_equivariant_aux. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite binding_kept_renamed, IH.
      destruct (binding_kept D binding); [ | reflexivity ].
      simpl. rewrite (erase_binding_equivariant r binding). reflexivity.
    + rewrite erase_handler_equivariant, erase_expr_equivariant_aux.
      reflexivity.
    + rewrite erase_expr_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (erase_handler_equivariant r h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite erase_expr_equivariant_aux. reflexivity.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    rewrite erase_code_equivariant. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    rewrite erase_expr_equivariant_aux. reflexivity.
Qed.

Fixpoint binds_only_kept_renamed (r : renaming) (e : expr) :
  binds_only_kept D (rename_expr r e) = binds_only_kept D e
with handler_binds_only_kept_renamed (r : renaming)
    (h : continuation_handler) :
  handler_binds_only_kept D (rename_handler r h)
  = handler_binds_only_kept D h
with binding_binds_only_kept_renamed (r : renaming)
    (binding : static_binding) :
  binding_binds_only_kept D (rename_static_binding r binding)
  = binding_binds_only_kept D binding
with code_binds_only_kept_renamed (r : renaming) (c : code) :
  code_binds_only_kept D (rename_code r c) = code_binds_only_kept D c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_only_kept_renamed.
    + rewrite binds_only_kept_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; simpl;
        [ reflexivity | ].
      rewrite (binding_binds_only_kept_renamed r binding), IH. reflexivity.
    + rewrite handler_binds_only_kept_renamed, binds_only_kept_renamed.
      reflexivity.
    + rewrite binds_only_kept_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_binds_only_kept_renamed r h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_only_kept_renamed.
  - destruct binding as [id c | id | sym const]; simpl; try reflexivity.
    rewrite code_binds_only_kept_renamed. reflexivity.
  - destruct c as [params my_closure ret exn metadata body]. simpl.
    apply binds_only_kept_renamed.
Qed.

Lemma handler_free_of_renamed (r : renaming) (h : continuation_handler) :
  handler_free_of D (rename_handler r h) = handler_free_of D h.
Proof.
  unfold handler_free_of.
  rewrite handler_avoids_renamed, handler_binds_only_kept_renamed.
  reflexivity.
Qed.

Lemma code_free_of_renamed (r : renaming) (c : code) :
  code_free_of D (rename_code r c) = code_free_of D c.
Proof.
  unfold code_free_of.
  rewrite code_avoids_renamed, code_binds_only_kept_renamed. reflexivity.
Qed.

Lemma env_free_of_renamed (r : renaming) (vars : env) :
  env_free_of D (rename_env r vars) = env_free_of D vars.
Proof.
  unfold env_free_of, rename_env. apply forallb_map_with.
  intros [x v]. reflexivity.
Qed.

Lemma handlers_free_of_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  forallb (fun '(_, h) => handler_free_of D h) (rename_handlers r handlers)
  = forallb (fun '(_, h) => handler_free_of D h) handlers.
Proof.
  unfold rename_handlers. apply forallb_map_with. intros [k h].
  apply handler_free_of_renamed.
Qed.

Fixpoint cont_free_of_renamed (r : renaming) (c : cont_value) :
  cont_free_of D (rename_cont_value r c) = cont_free_of D c.
Proof.
  destruct c
    as [h vars conts recursive | k | c extra
       | remaining remaining_arity return_arity return_to exn_to];
    simpl;
    [ | reflexivity | rewrite (cont_free_of_renamed r c); reflexivity | ].
  - rewrite handler_free_of_renamed, env_free_of_renamed,
      handlers_free_of_renamed.
    do 2 f_equal.
    induction conts as [ | [k c] conts IH]; simpl; [ reflexivity | ].
    rewrite (cont_free_of_renamed r c), IH. reflexivity.
  - rewrite (cont_free_of_renamed r return_to),
      (cont_free_of_renamed r exn_to).
    reflexivity.
Qed.

Lemma statics_free_of_renamed (r : renaming) (statics : static_env) :
  statics_free_of D (rename_static_env r statics) = statics_free_of D statics.
Proof.
  unfold statics_free_of, rename_static_env. simpl. f_equal.
  unfold map_values. apply forallb_map_with. intros [id c].
  apply code_free_of_renamed.
Qed.

(* FlambdaUnusedHeadStaticsSpec.v: the checks of the removal from the
   head of a unit read the same of a term renamed. *)
Lemma head_bindings_renamed (r : renaming) (e : expr) :
  head_bindings (rename_expr r e)
  = map (rename_static_binding r) (head_bindings e).
Proof.
  induction e; simpl; try reflexivity.
  rewrite map_app. f_equal. assumption.
Qed.

Lemma under_head_renamed (r : renaming) (e : expr) :
  under_head (rename_expr r e) = rename_expr r (under_head e).
Proof. induction e; simpl; try reflexivity. assumption. Qed.

Lemma symbols_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map symbols_of_binding (map (rename_static_binding r) bindings)
  = flat_map symbols_of_binding bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct binding; reflexivity.
Qed.

Lemma code_ids_of_bindings_renamed (r : renaming)
    (bindings : list static_binding) :
  flat_map code_ids_of_binding (map (rename_static_binding r) bindings)
  = flat_map code_ids_of_binding bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite IH. destruct binding; reflexivity.
Qed.

Lemma plain_simple_renamed_same (r : renaming) (s : simple) :
  plain_simple (rename_simple r s) = plain_simple s.
Proof. destruct s; reflexivity. Qed.

Lemma value_slots_plain_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  value_slots_plain
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_plain slots.
Proof.
  unfold value_slots_plain. apply forallb_map_with. intros slot.
  apply plain_simple_renamed_same.
Qed.

Lemma static_const_total_renamed (r : renaming) (const : static_const) :
  static_const_total (rename_static_const r const)
  = static_const_total const.
Proof.
  destruct const; simpl;
    rewrite ?value_slots_plain_renamed, ?plain_simple_renamed_same;
    try reflexivity.
  all: apply forallb_map_with; apply plain_simple_renamed_same.
Qed.

Lemma removed_binding_total_renamed (r : renaming)
    (binding : static_binding) :
  removed_binding_total D (rename_static_binding r binding)
  = removed_binding_total D binding.
Proof.
  destruct binding; simpl; try reflexivity.
  rewrite static_const_total_renamed. reflexivity.
Qed.

Lemma kept_code_binds_only_kept_renamed (r : renaming)
    (binding : static_binding) :
  kept_code_binds_only_kept D (rename_static_binding r binding)
  = kept_code_binds_only_kept D binding.
Proof.
  destruct binding; simpl; try reflexivity.
  rewrite code_binds_only_kept_renamed. reflexivity.
Qed.

Lemma head_statics_removable_renamed (r : renaming) (e : expr) :
  head_statics_removable D (rename_expr r e) = head_statics_removable D e.
Proof.
  unfold head_statics_removable.
  rewrite expr_avoids_renamed, under_head_renamed, binds_only_kept_renamed,
    head_bindings_renamed, symbols_of_bindings_renamed,
    code_ids_of_bindings_renamed,
    (forallb_map_with _ _ _ _ (kept_code_binds_only_kept_renamed r)),
    (forallb_map_with _ _ _ _ (removed_binding_total_renamed r)).
  reflexivity.
Qed.

End Dropped.

Theorem expr_avoids_equivariant : expr_avoids_equivariant_statement.
Proof. intros r D e. apply expr_avoids_renamed. Qed.

Theorem erase_expr_equivariant : erase_expr_equivariant_statement.
Proof. intros D r e. apply erase_expr_equivariant_aux. Qed.

Theorem drop_statics_equivariant : drop_statics_equivariant_statement.
Proof.
  intros r D e. unfold drop_statics. rewrite expr_avoids_renamed.
  destruct (expr_avoids D e); [ | reflexivity ].
  simpl. rewrite erase_expr_equivariant_aux. reflexivity.
Qed.

Theorem binds_only_kept_equivariant : binds_only_kept_equivariant_statement.
Proof. intros r D e. apply binds_only_kept_renamed. Qed.

Theorem envs_free_of_equivariant : envs_free_of_equivariant_statement.
Proof.
  intros r D vars conts statics. unfold envs_free_of.
  rewrite env_free_of_renamed, statics_free_of_renamed. do 2 f_equal.
  unfold rename_cont_env. apply forallb_map_with. intros [k c].
  apply cont_free_of_renamed.
Qed.

(* ======================================================================== *)
(* FlambdaBlockLoadSpec.v                                                   *)
(* ======================================================================== *)

(* A map whose keys are renamed: a renaming is injective, so a name is
   found where its image is. *)
Lemma lookup_renamed_keys {A B : Type} (p : permutation) (g : A -> B)
    (x : positive) (l : list (positive * A)) :
  lookup (permute p x)
    (map (fun entry => (permute p (fst entry), g (snd entry))) l)
  = option_map g (lookup x l).
Proof.
  induction l as [ | [y a] l IH]; simpl; [ reflexivity | ].
  rewrite permute_eqb. destruct (Pos.eqb x y); [ reflexivity | exact IH ].
Qed.

Lemma lookup_value_slots_renamed (r : renaming) (slot : value_slot)
    (slots : list (value_slot * simple)) :
  lookup slot (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = option_map (rename_simple r) (lookup slot slots).
Proof.
  induction slots as [ | [slot' s] slots IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb slot slot'); [ reflexivity | exact IH ].
Qed.

Lemma forget_field_equivariant (r : renaming) (x : variable)
    (field : option simple) :
  forget_field (rename_variable r x) (option_map (rename_simple r) field)
  = option_map (rename_simple r) (forget_field x field).
Proof.
  destruct field as [s | ]; simpl; [ | reflexivity ].
  rewrite var_in_simple_equivariant. destruct (var_in_simple x s); reflexivity.
Qed.

Lemma forget_in_block_equivariant (r : renaming) (x : variable)
    (block : known_block) :
  forget_in_block (rename_variable r x) (rename_known_block r block)
  = rename_known_block r (forget_in_block x block).
Proof.
  unfold forget_in_block, rename_known_block. simpl. rewrite !map_map.
  f_equal. apply map_ext. intros field. apply forget_field_equivariant.
Qed.

Lemma forget_var_equivariant (r : renaming) (x : variable)
    (known : known_blocks) :
  forget_var (rename_variable r x) (rename_known_blocks r known)
  = rename_known_blocks r (forget_var x known).
Proof.
  unfold forget_var, rename_known_blocks, map_values. simpl. f_equal.
  - rewrite
      (filter_map_with _ _
         (fun entry => negb (Pos.eqb x (fst entry))))
      by (intros entry; simpl; rewrite rename_variable_eqb; reflexivity).
    rewrite !map_map. apply map_ext. intros entry. simpl.
    rewrite forget_in_block_equivariant. reflexivity.
  - rewrite !map_map. apply map_ext. intros [sym block]. simpl.
    rewrite forget_in_block_equivariant. reflexivity.
Qed.

Lemma forget_vars_equivariant (r : renaming) (xs : list variable) :
  ∀ known : known_blocks,
    forget_vars (map (rename_variable r) xs) (rename_known_blocks r known)
    = rename_known_blocks r (forget_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_var_equivariant. apply IH.
Qed.

Lemma block_made_equivariant (r : renaming) (d : named) :
  block_made (rename_named r d)
  = option_map (rename_known_block r) (block_made d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  destruct p as [p | p | [[tag]| ] | p | rp]; try reflexivity.
  simpl. unfold rename_known_block. simpl. rewrite !map_map. reflexivity.
Qed.

Lemma known_under_let_equivariant (r : renaming) (known : known_blocks)
    (x : variable) (d : named) :
  known_under_let (rename_known_blocks r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_blocks r (known_under_let known x d).
Proof.
  unfold known_under_let.
  rewrite forget_var_equivariant, block_made_equivariant.
  destruct (block_made d) as [block | ]; cbn [option_map]; [ | reflexivity ].
  rewrite forget_in_block_equivariant. reflexivity.
Qed.

Lemma static_block_made_renamed (r : renaming) (binding : static_binding) :
  static_block_made (rename_static_binding r binding)
  = map_values (rename_known_block r) (static_block_made binding).
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  destruct const; try reflexivity.
  simpl. unfold rename_known_block. simpl. rewrite !map_map. reflexivity.
Qed.

Lemma known_under_statics_equivariant (r : renaming) (known : known_blocks)
    (bindings : list static_binding) :
  known_under_statics (rename_known_blocks r known)
    (map (rename_static_binding r) bindings)
  = rename_known_blocks r (known_under_statics known bindings).
Proof.
  unfold known_under_statics, rename_known_blocks, map_values. simpl.
  f_equal. rewrite map_app. f_equal.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite static_block_made_renamed, map_app, IH. reflexivity.
Qed.

Lemma known_block_of_equivariant (r : renaming) (known : known_blocks)
    (s : simple) :
  known_block_of (rename_known_blocks r known) (rename_simple r s)
  = option_map (rename_known_block r) (known_block_of known s).
Proof.
  destruct s as [x | sym | c]; simpl;
    [ apply lookup_renamed_keys | apply lookup_map_values | reflexivity ].
Qed.

Lemma field_loaded_equivariant (r : renaming) (kind : block_access_kind)
    (field : Z) (block : known_block) :
  field_loaded kind field (rename_known_block r block)
  = option_map (rename_simple r) (field_loaded kind field block).
Proof.
  destruct kind as [tag size [ | ]]; [ | reflexivity ].
  unfold field_loaded, rename_known_block. simpl. rewrite length_map.
  destruct
    (known_is tag (fst block)
     && known_is size (Z.of_nat (length (snd block)))
     && (0 <=? field)%Z);
    [ | reflexivity ].
  rewrite nth_error_map.
  destruct (nth_error (snd block) (Z.to_nat field)) as [[contents | ] | ];
    reflexivity.
Qed.

Lemma of_known_block_equivariant (r : renaming) (p : unary_primitive)
    (block : known_block) :
  of_known_block p (rename_known_block r block)
  = option_map (rename_simple r) (of_known_block p block).
Proof.
  destruct p as [kind mut field | | | | | | | | | | | | | | ];
    try reflexivity.
  destruct mut; try reflexivity; apply field_loaded_equivariant.
Qed.

Lemma unary_applied_equivariant (r : renaming) (p : prim)
    (args : list simple) :
  unary_applied p (map (rename_simple r) args)
  = option_map (fun applied => (fst applied, rename_simple r (snd applied)))
      (unary_applied p args).
Proof.
  destruct p; try reflexivity.
  destruct args as [ | arg [ | arg' args]]; reflexivity.
Qed.

Lemma load_known_named_equivariant (r : renaming) (known : known_blocks)
    (d : named) :
  load_known_named (rename_known_blocks r known) (rename_named r d)
  = rename_named r (load_known_named known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  cbn [rename_named load_known_named]. rewrite unary_applied_equivariant.
  destruct (unary_applied p args) as [[p' block] | ]; cbn [option_map fst snd];
    [ | reflexivity ].
  rewrite known_block_of_equivariant.
  destruct (known_block_of known block) as [made | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite of_known_block_equivariant.
  destruct (of_known_block p' made); reflexivity.
Qed.

Fixpoint load_blocks_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : known_blocks,
    load_blocks_knowing (rename_known_blocks r known) (rename_expr r e)
    = rename_expr r (load_blocks_knowing known e)
with load_blocks_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : known_blocks,
    load_blocks_in_handler (rename_known_blocks r known) (rename_handler r h)
    = rename_handler r (load_blocks_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; try reflexivity;
      cbn [rename_expr load_blocks_knowing].
    + rewrite load_known_named_equivariant, known_under_let_equivariant,
        load_blocks_knowing_equivariant_aux.
      reflexivity.
    + rewrite known_under_statics_equivariant,
        load_blocks_knowing_equivariant_aux.
      reflexivity.
    + rewrite load_blocks_in_handler_equivariant,
        load_blocks_knowing_equivariant_aux.
      reflexivity.
    + rewrite load_blocks_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [rename_handler load_blocks_in_handler].
    rewrite forget_vars_equivariant, load_blocks_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem load_blocks_knowing_equivariant :
  load_blocks_knowing_equivariant_statement.
Proof. intros r known e. apply load_blocks_knowing_equivariant_aux. Qed.

Theorem load_known_blocks_equivariant :
  load_known_blocks_equivariant_statement.
Proof.
  intros r e. exact (load_blocks_knowing_equivariant_aux r e no_blocks).
Qed.

(* The read of a known block that the term after is given. *)
Lemma of_known_block_surely_equivariant (r : renaming)
    (p : unary_primitive) (block : known_block) :
  of_known_block_surely p (rename_known_block r block)
  = option_map (rename_simple r) (of_known_block_surely p block).
Proof.
  unfold of_known_block_surely.
  destruct p; try reflexivity.
  rewrite of_known_block_equivariant.
  match goal with
  | |- context [of_known_block ?p block] =>
    destruct (of_known_block p block) as [contents | ]
  end;
    cbn [option_map]; [ | reflexivity ].
  rewrite FlambdaReadsAfterProofs.surely_value_renamed.
  destruct (surely_value contents); reflexivity.
Qed.

Lemma load_known_named_surely_equivariant (r : renaming) (known : known_blocks)
    (d : named) :
  load_known_named_surely (rename_known_blocks r known) (rename_named r d)
  = rename_named r (load_known_named_surely known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  cbn [rename_named load_known_named_surely]. rewrite unary_applied_equivariant.
  destruct (unary_applied p args) as [[p' block] | ]; cbn [option_map fst snd];
    [ | reflexivity ].
  rewrite known_block_of_equivariant.
  destruct (known_block_of known block) as [made | ]; cbn [option_map];
    [ | reflexivity ].
  rewrite of_known_block_surely_equivariant.
  destruct (of_known_block_surely p' made); reflexivity.
Qed.

Fixpoint load_blocks_knowing_surely_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : known_blocks,
    load_blocks_knowing_surely (rename_known_blocks r known) (rename_expr r e)
    = rename_expr r (load_blocks_knowing_surely known e)
with load_blocks_in_handler_surely_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : known_blocks,
    load_blocks_in_handler_surely (rename_known_blocks r known)
      (rename_handler r h)
    = rename_handler r (load_blocks_in_handler_surely known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; try reflexivity;
      cbn [rename_expr load_blocks_knowing_surely].
    + rewrite load_known_named_surely_equivariant, known_under_let_equivariant,
        load_blocks_knowing_surely_equivariant_aux.
      reflexivity.
    + rewrite known_under_statics_equivariant,
        load_blocks_knowing_surely_equivariant_aux.
      reflexivity.
    + rewrite load_blocks_in_handler_surely_equivariant,
        load_blocks_knowing_surely_equivariant_aux.
      reflexivity.
    + rewrite load_blocks_knowing_surely_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [rename_handler load_blocks_in_handler_surely].
    rewrite forget_vars_equivariant, load_blocks_knowing_surely_equivariant_aux.
    reflexivity.
Qed.

Theorem load_blocks_knowing_surely_equivariant :
  load_blocks_knowing_surely_equivariant_statement.
Proof. intros r known e. apply load_blocks_knowing_surely_equivariant_aux. Qed.

Theorem load_known_blocks_surely_equivariant :
  load_known_blocks_surely_equivariant_statement.
Proof.
  intros r e. exact (load_blocks_knowing_surely_equivariant_aux r e no_blocks).
Qed.

(* ======================================================================== *)
(* FlambdaUnboxSpec.v                                                       *)
(* ======================================================================== *)

Lemma forget_wrapped_var_equivariant (r : renaming) (x : variable)
    (known : known_numbers) :
  forget_wrapped_var (rename_variable r x) (rename_known_numbers r known)
  = rename_known_numbers r (forget_wrapped_var x known).
Proof.
  unfold forget_wrapped_var, rename_known_numbers. simpl. f_equal.
  apply filter_map_with. intros entry. simpl.
  rewrite rename_variable_eqb, var_in_simple_equivariant. reflexivity.
Qed.

Lemma forget_wrapped_vars_equivariant (r : renaming) (xs : list variable) :
  ∀ known : known_numbers,
    forget_wrapped_vars (map (rename_variable r) xs)
      (rename_known_numbers r known)
    = rename_known_numbers r (forget_wrapped_vars xs known).
Proof.
  induction xs as [ | x xs IH]; intros known; simpl; [ reflexivity | ].
  rewrite forget_wrapped_var_equivariant. apply IH.
Qed.

Lemma number_wrapped_equivariant (r : renaming) (d : named) :
  number_wrapped (rename_named r d)
  = option_map (fun wrapped => (fst wrapped, rename_simple r (snd wrapped)))
      (number_wrapped d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  cbn [rename_named number_wrapped]. rewrite unary_applied_equivariant.
  destruct (unary_applied p args) as [[p' contents] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  destruct p'; reflexivity.
Qed.

Lemma numbers_under_let_equivariant (r : renaming) (known : known_numbers)
    (x : variable) (d : named) :
  numbers_under_let (rename_known_numbers r known) (rename_variable r x)
    (rename_named r d)
  = rename_known_numbers r (numbers_under_let known x d).
Proof.
  unfold numbers_under_let.
  rewrite forget_wrapped_var_equivariant, number_wrapped_equivariant.
  destruct (number_wrapped d) as [[how contents] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  rewrite var_in_simple_equivariant.
  destruct (var_in_simple x contents); reflexivity.
Qed.

Lemma static_number_boxed_renamed (r : renaming) (binding : static_binding) :
  static_number_boxed (rename_static_binding r binding)
  = static_number_boxed binding.
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  destruct const as [ | | | [x | sym' | c] | | ]; reflexivity.
Qed.

Lemma numbers_under_statics_equivariant (r : renaming)
    (known : known_numbers) (bindings : list static_binding) :
  numbers_under_statics (rename_known_numbers r known)
    (map (rename_static_binding r) bindings)
  = rename_known_numbers r (numbers_under_statics known bindings).
Proof.
  unfold numbers_under_statics, rename_known_numbers. simpl. do 2 f_equal.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite static_number_boxed_renamed, IH. reflexivity.
Qed.

Lemma unwrapped_equivariant (r : renaming) (known : known_numbers)
    (how : wrapping) (wrapped : simple) :
  unwrapped (rename_known_numbers r known) how (rename_simple r wrapped)
  = option_map (rename_simple r) (unwrapped known how wrapped).
Proof.
  destruct wrapped as [x | sym | c]; simpl; [ | | reflexivity ].
  - unfold rename_variable.
    rewrite
      (lookup_renamed_keys (variable_permutation r)
         (fun wrapped => (fst wrapped, rename_simple r (snd wrapped)))).
    destruct (lookup x (known_wrapped_vars known)) as [[how' contents] | ];
      cbn [option_map fst snd]; [ | reflexivity ].
    destruct (wrapping_eqb how how'); reflexivity.
  - destruct how as [kind | ]; [ | reflexivity ].
    destruct (lookup sym (known_boxed_symbols known)) as [c | ];
      [ | reflexivity ].
    destruct (is_number_of_kind kind c); reflexivity.
Qed.

Lemma unwrapping_of_known_equivariant (r : renaming) (known : known_numbers)
    (p : prim) (args : list simple) :
  unwrapping_of_known (rename_known_numbers r known) p
    (map (rename_simple r) args)
  = option_map (rename_simple r) (unwrapping_of_known known p args).
Proof.
  unfold unwrapping_of_known. rewrite unary_applied_equivariant.
  destruct (unary_applied p args) as [[p' wrapped] | ];
    cbn [option_map fst snd]; [ | reflexivity ].
  destruct p'; try reflexivity; apply unwrapped_equivariant.
Qed.

Lemma unbox_known_named_equivariant (r : renaming) (known : known_numbers)
    (d : named) :
  unbox_known_named (rename_known_numbers r known) (rename_named r d)
  = rename_named r (unbox_known_named known d).
Proof.
  destruct d as [s | p args | | | | | ]; try reflexivity.
  cbn [rename_named unbox_known_named].
  rewrite unwrapping_of_known_equivariant.
  destruct (unwrapping_of_known known p args); reflexivity.
Qed.

Fixpoint unbox_numbers_knowing_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : known_numbers,
    unbox_numbers_knowing (rename_known_numbers r known) (rename_expr r e)
    = rename_expr r (unbox_numbers_knowing known e)
with unbox_numbers_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ known : known_numbers,
    unbox_numbers_in_handler (rename_known_numbers r known)
      (rename_handler r h)
    = rename_handler r (unbox_numbers_in_handler known h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros known; try reflexivity;
      cbn [rename_expr unbox_numbers_knowing].
    + rewrite unbox_known_named_equivariant, numbers_under_let_equivariant,
        unbox_numbers_knowing_equivariant_aux.
      reflexivity.
    + rewrite numbers_under_statics_equivariant,
        unbox_numbers_knowing_equivariant_aux.
      reflexivity.
    + rewrite unbox_numbers_in_handler_equivariant,
        unbox_numbers_knowing_equivariant_aux.
      reflexivity.
    + rewrite unbox_numbers_knowing_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. intros known.
    cbn [rename_handler unbox_numbers_in_handler].
    rewrite forget_wrapped_vars_equivariant,
      unbox_numbers_knowing_equivariant_aux.
    reflexivity.
Qed.

Theorem unbox_numbers_knowing_equivariant :
  unbox_numbers_knowing_equivariant_statement.
Proof. intros r known e. apply unbox_numbers_knowing_equivariant_aux. Qed.

Theorem unbox_known_numbers_equivariant :
  unbox_known_numbers_equivariant_statement.
Proof.
  intros r e. exact (unbox_numbers_knowing_equivariant_aux r e no_numbers).
Qed.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

Lemma find_alias_equivariant (r : renaming) (tenv : typing_env)
    (x : variable) :
  find_alias (rename_naive_typing_env r tenv) (rename_variable r x)
  = option_map (rename_simple r) (find_alias tenv x).
Proof.
  unfold rename_naive_typing_env.
  induction tenv as [ | [y t] tenv IH]; simpl; [ reflexivity | ].
  destruct t as [ | | s | code slots]; simpl; try exact IH.
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma canonical_simple_equivariant (r : renaming) (tenv : typing_env)
    (s : simple) :
  canonical_simple (rename_naive_typing_env r tenv) (rename_simple r s)
  = rename_simple r (canonical_simple tenv s).
Proof.
  destruct s as [x | sym | c]; simpl; try reflexivity.
  rewrite find_alias_equivariant. destruct (find_alias tenv x); reflexivity.
Qed.

Lemma canonical_simples_equivariant (r : renaming) (tenv : typing_env)
    (args : list simple) :
  map (canonical_simple (rename_naive_typing_env r tenv))
    (map (rename_simple r) args)
  = map (rename_simple r) (map (canonical_simple tenv) args).
Proof.
  rewrite !map_map. apply map_ext. intros s.
  apply canonical_simple_equivariant.
Qed.

Lemma canonical_value_slots_equivariant (r : renaming) (tenv : typing_env)
    (slots : list (value_slot * simple)) :
  canonical_value_slots (rename_naive_typing_env r tenv)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = map (fun slot => (fst slot, rename_simple r (snd slot)))
      (canonical_value_slots tenv slots).
Proof.
  unfold canonical_value_slots. rewrite !map_map. apply map_ext.
  intros slot. simpl. rewrite canonical_simple_equivariant. reflexivity.
Qed.

(* The action on the closure that a variable is known to be. *)
Definition rename_known_closure (r : renaming)
    (closure : code_id * list (value_slot * simple))
    : code_id * list (value_slot * simple) :=
  (fst closure,
   map (fun slot => (fst slot, rename_simple r (snd slot))) (snd closure)).

Lemma find_closure_equivariant (r : renaming) (tenv : typing_env)
    (x : variable) :
  find_closure (rename_naive_typing_env r tenv) (rename_variable r x)
  = option_map (rename_known_closure r) (find_closure tenv x).
Proof.
  unfold rename_naive_typing_env.
  induction tenv as [ | [y t] tenv IH]; simpl; [ reflexivity | ].
  destruct t as [ | | s | code slots]; simpl; try exact IH.
  rewrite rename_variable_eqb. destruct (Pos.eqb x y); [ reflexivity | ].
  exact IH.
Qed.

Lemma closure_of_simple_equivariant (r : renaming) (tenv : typing_env)
    (s : simple) :
  closure_of_simple (rename_naive_typing_env r tenv) (rename_simple r s)
  = option_map (rename_known_closure r) (closure_of_simple tenv s).
Proof.
  destruct s; simpl; [ apply find_closure_equivariant | | ]; reflexivity.
Qed.

Lemma canonical_project_equivariant (r : renaming) (tenv : typing_env)
    (slot : value_slot) (closure : simple) :
  canonical_project (rename_naive_typing_env r tenv) slot
    (rename_simple r closure)
  = rename_named r (canonical_project tenv slot closure).
Proof.
  unfold canonical_project. rewrite closure_of_simple_equivariant.
  destruct (closure_of_simple tenv closure) as [[code slots] | ];
    cbn [option_map rename_known_closure fst snd]; [ | reflexivity ].
  rewrite lookup_value_slots_renamed. destruct (lookup slot slots); reflexivity.
Qed.

Lemma canonical_named_equivariant (r : renaming) (tenv : typing_env)
    (d : named) :
  canonical_named (rename_naive_typing_env r tenv) (rename_named r d)
  = rename_named r (canonical_named tenv d).
Proof.
  destruct d; cbn [rename_named canonical_named];
    rewrite ?canonical_simple_equivariant, ?canonical_simples_equivariant,
      ?canonical_value_slots_equivariant;
    try reflexivity.
  apply canonical_project_equivariant.
Qed.

Lemma canonical_static_binding_equivariant (r : renaming)
    (tenv : typing_env) (binding : static_binding) :
  canonical_static_binding (rename_naive_typing_env r tenv)
    (rename_static_binding r binding)
  = rename_static_binding r (canonical_static_binding tenv binding).
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  destruct const;
    cbn [rename_static_binding rename_static_const canonical_static_binding
         canonical_static_const];
    rewrite ?canonical_simple_equivariant, ?canonical_simples_equivariant,
      ?canonical_value_slots_equivariant;
    reflexivity.
Qed.

Lemma canonical_apply_cont_equivariant (r : renaming) (tenv : typing_env)
    (ac : apply_cont) :
  canonical_apply_cont (rename_naive_typing_env r tenv)
    (rename_apply_cont r ac)
  = rename_apply_cont r (canonical_apply_cont tenv ac).
Proof.
  unfold canonical_apply_cont, rename_apply_cont.
  cbn [ac_continuation ac_args]. rewrite canonical_simples_equivariant.
  reflexivity.
Qed.

Lemma canonical_call_kind_equivariant (r : renaming) (tenv : typing_env)
    (callee : simple) (kind : call_kind) :
  canonical_call_kind (rename_naive_typing_env r tenv)
    (rename_simple r callee) kind
  = canonical_call_kind tenv callee kind.
Proof.
  unfold canonical_call_kind. rewrite closure_of_simple_equivariant.
  destruct kind as [called | ],
    (closure_of_simple tenv callee) as [[code slots] | ];
    reflexivity.
Qed.

Lemma canonical_exn_extra_args_equivariant (r : renaming) (tenv : typing_env)
    (extra : list (simple * flambda_kind)) :
  map
    (fun arg =>
       (canonical_simple (rename_naive_typing_env r tenv) (fst arg), snd arg))
    (map (fun arg => (rename_simple r (fst arg), snd arg)) extra)
  = map (fun arg => (rename_simple r (fst arg), snd arg))
      (map (fun arg => (canonical_simple tenv (fst arg), snd arg)) extra).
Proof.
  rewrite !map_map. apply map_ext. intros [s kind]. cbn [fst snd].
  rewrite canonical_simple_equivariant. reflexivity.
Qed.

Lemma canonical_apply_equivariant (r : renaming) (tenv : typing_env)
    (ap : apply_expr) :
  canonical_apply (rename_naive_typing_env r tenv) (rename_apply r ap)
  = rename_apply r (canonical_apply tenv ap).
Proof.
  unfold canonical_apply, rename_apply.
  cbn [ap_callee ap_call_kind ap_args ap_continuation ap_exn_continuation
       ap_exn_extra_args ap_args_arity ap_return_arity].
  rewrite canonical_simple_equivariant, canonical_call_kind_equivariant,
    canonical_simples_equivariant, canonical_exn_extra_args_equivariant.
  reflexivity.
Qed.

Lemma var_in_type_equivariant (r : renaming) (x : variable)
    (t : flambda_type) :
  var_in_type (rename_variable r x) (rename_naive_type r t) = var_in_type x t.
Proof.
  destruct t; simpl;
    [ reflexivity | reflexivity | apply var_in_simple_equivariant
    | apply var_in_value_slots_equivariant ].
Qed.

Lemma drop_var_equivariant (r : renaming) (x : variable) (tenv : typing_env) :
  drop_var (rename_variable r x) (rename_naive_typing_env r tenv)
  = rename_naive_typing_env r (drop_var x tenv).
Proof.
  unfold drop_var, rename_naive_typing_env. apply filter_map_with.
  intros [y t]. simpl.
  rewrite rename_variable_eqb, var_in_type_equivariant. reflexivity.
Qed.

Lemma drop_vars_equivariant (r : renaming) (xs : list variable)
    (tenv : typing_env) :
  drop_vars (map (rename_variable r) xs) (rename_naive_typing_env r tenv)
  = rename_naive_typing_env r (drop_vars xs tenv).
Proof.
  unfold drop_vars. induction xs as [ | x xs IH]; simpl; [ reflexivity | ].
  rewrite IH. apply drop_var_equivariant.
Qed.

Lemma typing_env_under_let_equivariant (r : renaming) (tenv : typing_env)
    (x : variable) (d : named) :
  typing_env_under_let (rename_naive_typing_env r tenv) (rename_variable r x)
    (rename_named r d)
  = rename_naive_typing_env r (typing_env_under_let tenv x d).
Proof.
  destruct d as [s | p args | code slots | | | | ];
    cbn [rename_named typing_env_under_let];
    try apply drop_var_equivariant.
  - rewrite var_in_simple_equivariant, drop_var_equivariant.
    destruct (var_in_simple x s); reflexivity.
  - (* What is known of the result of a store is a constant, which a
       renaming leaves as it is. *)
    destruct p as [u | b | vr | q | r0]; try apply drop_var_equivariant.
    destruct q as [mut tag | kind init field | kind | kind set_kind];
      try apply drop_var_equivariant;
      rewrite drop_var_equivariant; reflexivity.
  - rewrite var_in_value_slots_equivariant, drop_var_equivariant.
    destruct (var_in_value_slots x slots); reflexivity.
Qed.

Fixpoint follow_aliases_equivariant_aux (r : renaming) (e : expr) :
  ∀ tenv : typing_env,
    follow_aliases (rename_naive_typing_env r tenv) (rename_expr r e)
    = rename_expr r (follow_aliases tenv e)
with follow_aliases_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ tenv : typing_env,
    follow_aliases_in_handler (rename_naive_typing_env r tenv)
      (rename_handler r h)
    = rename_handler r (follow_aliases_in_handler tenv h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros tenv; cbn [rename_expr follow_aliases].
    + rewrite canonical_named_equivariant, typing_env_under_let_equivariant,
        follow_aliases_equivariant_aux.
      reflexivity.
    + rewrite follow_aliases_equivariant_aux, !map_map. f_equal.
      apply map_ext. intros binding.
      apply canonical_static_binding_equivariant.
    + rewrite follow_aliases_in_handler_equivariant,
        follow_aliases_equivariant_aux.
      reflexivity.
    + rewrite follow_aliases_equivariant_aux. reflexivity.
    + rewrite canonical_apply_equivariant. reflexivity.
    + rewrite canonical_apply_cont_equivariant. reflexivity.
    + rewrite canonical_simple_equivariant, !map_map. f_equal.
      apply map_ext. intros arm. cbn [fst snd].
      rewrite canonical_apply_cont_equivariant. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler]. intros tenv.
    cbn [rename_handler follow_aliases_in_handler].
    rewrite drop_vars_equivariant, follow_aliases_equivariant_aux.
    reflexivity.
Qed.

Theorem follow_aliases_equivariant : follow_aliases_equivariant_statement.
Proof. intros r tenv e. apply follow_aliases_equivariant_aux. Qed.

Theorem follow_aliases_unknowing_equivariant :
  follow_aliases_unknowing_equivariant_statement.
Proof. intros r e. exact (follow_aliases_equivariant_aux r e []). Qed.

(* ======================================================================== *)
(* The checks of FlambdaWellFormedSpec.v                                    *)
(* ======================================================================== *)

Lemma forallb_same {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = g a) -> forallb f l = forallb g l.
Proof.
  intros Hfg. induction l as [ | a l IH]; simpl; [ reflexivity | ].
  rewrite Hfg, IH. reflexivity.
Qed.

Lemma existsb_same {A : Type} (f g : A -> bool) (l : list A) :
  (∀ a : A, f a = g a) -> existsb f l = existsb g l.
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

Lemma traps_renamed (r : renaming) (k : continuation) (ac : apply_cont) :
  traps (rename_continuation r k) (rename_apply_cont r ac) = traps k ac.
Proof.
  unfold traps. simpl.
  destruct (ac_trap_action ac) as [[k' | k' kind] | ]; simpl;
    [ apply rename_continuation_eqb | apply rename_continuation_eqb
    | reflexivity ].
Qed.

Fixpoint raises_to_equivariant_aux (r : renaming) (k : continuation)
    (e : expr) :
  raises_to (rename_continuation r k) (rename_expr r e) = raises_to k e
with handler_raises_to_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) :
  handler_raises_to (rename_continuation r k) (rename_handler r h)
  = handler_raises_to k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply raises_to_equivariant_aux.
    + apply raises_to_equivariant_aux.
    + rewrite handler_raises_to_equivariant, raises_to_equivariant_aux.
      reflexivity.
    + rewrite raises_to_equivariant_aux. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_raises_to_equivariant r k h), IH. reflexivity.
    + apply rename_continuation_eqb.
    + exact (traps_renamed r k ac).
    + apply existsb_map_with. intros arm.
      exact (traps_renamed r k (snd arm)).
  - destruct h as [params metadata handler]. simpl.
    apply raises_to_equivariant_aux.
Qed.

Theorem raises_to_equivariant : raises_to_equivariant_statement.
Proof. exact raises_to_equivariant_aux. Qed.

Lemma handler_is_exn_handler_renamed (r : renaming)
    (h : continuation_handler) :
  handler_is_exn_handler (rename_handler r h) = handler_is_exn_handler h.
Proof. destruct h; reflexivity. Qed.

Theorem handler_at_toplevel_equivariant :
  handler_at_toplevel_equivariant_statement.
Proof.
  intros r unit_exn k h body. unfold handler_at_toplevel.
  rewrite handler_is_exn_handler_renamed,
    well_formed_free_continuations_renamed.
  f_equal.
  exact
    (all_among_map_injective (rename_continuation r)
       (rename_continuation_eqb r) (free_continuations body) [k; unit_exn]).
Qed.

Fixpoint statics_at_toplevel_equivariant_aux (r : renaming)
    (unit_exn : continuation) (e : expr) :
  ∀ toplevel : bool,
    statics_at_toplevel (rename_continuation r unit_exn) toplevel
      (rename_expr r e)
    = statics_at_toplevel unit_exn toplevel e
with handler_statics_at_toplevel_equivariant (r : renaming)
    (unit_exn : continuation) (h : continuation_handler) :
  ∀ toplevel : bool,
    handler_statics_at_toplevel (rename_continuation r unit_exn) toplevel
      (rename_handler r h)
    = handler_statics_at_toplevel unit_exn toplevel h
with binding_statics_at_toplevel_equivariant (r : renaming)
    (unit_exn : continuation) (binding : static_binding) :
  binding_statics_at_toplevel (rename_continuation r unit_exn)
    (rename_static_binding r binding)
  = binding_statics_at_toplevel unit_exn binding
with code_statics_at_toplevel_equivariant (r : renaming)
    (unit_exn : continuation) (c : code) :
  code_statics_at_toplevel (rename_continuation r unit_exn) (rename_code r c)
  = code_statics_at_toplevel unit_exn c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros toplevel; cbn [rename_expr statics_at_toplevel];
      try reflexivity.
    + apply statics_at_toplevel_equivariant_aux.
    + rewrite statics_at_toplevel_equivariant_aux. do 2 f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_statics_at_toplevel_equivariant r unit_exn binding),
        IH.
      reflexivity.
    + rewrite handler_at_toplevel_equivariant,
        handler_statics_at_toplevel_equivariant,
        statics_at_toplevel_equivariant_aux.
      reflexivity.
    + rewrite statics_at_toplevel_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_statics_at_toplevel_equivariant r unit_exn h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros toplevel.
    cbn [rename_handler handler_statics_at_toplevel].
    apply statics_at_toplevel_equivariant_aux.
  - destruct binding as [id c | id | sym const];
      cbn [rename_static_binding binding_statics_at_toplevel];
      try reflexivity.
    apply code_statics_at_toplevel_equivariant.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_statics_at_toplevel].
    apply statics_at_toplevel_equivariant_aux.
Qed.

Theorem statics_at_toplevel_equivariant :
  statics_at_toplevel_equivariant_statement.
Proof.
  intros r unit_exn toplevel e. apply statics_at_toplevel_equivariant_aux.
Qed.

Section Static_names.

Variable r : renaming.
Variable names_of : static_binding -> list positive.
Hypothesis names_of_renamed :
  ∀ binding : static_binding,
    names_of (rename_static_binding r binding) = names_of binding.

Lemma flat_map_names_renamed (bindings : list static_binding) :
  flat_map names_of (map (rename_static_binding r) bindings)
  = flat_map names_of bindings.
Proof.
  induction bindings as [ | binding bindings IH]; simpl; [ reflexivity | ].
  rewrite names_of_renamed, IH. reflexivity.
Qed.

Fixpoint static_names_renamed (e : expr) :
  static_names names_of (rename_expr r e) = static_names names_of e
with handler_static_names_renamed (h : continuation_handler) :
  handler_static_names names_of (rename_handler r h)
  = handler_static_names names_of h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply static_names_renamed.
    + rewrite static_names_renamed, flat_map_names_renamed. reflexivity.
    + rewrite handler_static_names_renamed, static_names_renamed.
      reflexivity.
    + rewrite static_names_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_static_names_renamed h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl. apply static_names_renamed.
Qed.

(* A group binds a name twice, or one that is bound, as the renamed
   group does. *)
Lemma rebinds_renamed (is_bound is_bound' : positive -> bool)
    (bindings : list static_binding) :
  (∀ name : positive, is_bound' name = is_bound name) ->
  rebinds names_of is_bound' (map (rename_static_binding r) bindings)
  = rebinds names_of is_bound bindings.
Proof.
  intros Hbound. unfold rebinds. rewrite flat_map_names_renamed.
  f_equal. apply existsb_same, Hbound.
Qed.

End Static_names.

Lemma code_ids_of_binding_renamed (r : renaming) (binding : static_binding) :
  code_ids_of_binding (rename_static_binding r binding)
  = code_ids_of_binding binding.
Proof. destruct binding; reflexivity. Qed.

Lemma symbols_of_binding_renamed (r : renaming) (binding : static_binding) :
  symbols_of_binding (rename_static_binding r binding)
  = symbols_of_binding binding.
Proof. destruct binding; reflexivity. Qed.

Theorem static_names_equivariant : static_names_equivariant_statement.
Proof. intros r names_of e Hnames. apply static_names_renamed, Hnames. Qed.

Theorem bound_static_names_equivariant :
  bound_static_names_equivariant_statement.
Proof.
  intros r e. split; apply static_names_renamed;
    [ apply code_ids_of_binding_renamed | apply symbols_of_binding_renamed ].
Qed.

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

Lemma named_sets_well_formed_renamed (r : renaming) (d : named) :
  named_sets_well_formed (rename_named r d) = named_sets_well_formed d.
Proof. destruct d; reflexivity. Qed.

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

Lemma static_set_code_ids_renamed (r : renaming) (binding : static_binding) :
  static_set_code_ids (rename_static_binding r binding)
  = static_set_code_ids binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  destruct const; reflexivity.
Qed.

Lemma binds_symbol_of_set_renamed (r : renaming)
    (binding binding' : static_binding) :
  binds_symbol_of_set (rename_static_binding r binding)
    (rename_static_binding r binding')
  = binds_symbol_of_set binding binding'.
Proof.
  destruct binding as [id c | id | s const], binding' as [id' c' | id' | s' c'];
    try reflexivity;
    destruct const; reflexivity.
Qed.

Lemma static_sets_code_disjoint_renamed (r : renaming)
    (bindings : list static_binding) :
  static_sets_code_disjoint (map (rename_static_binding r) bindings)
  = static_sets_code_disjoint bindings.
Proof.
  induction bindings as [ | binding bindings IH];
    cbn [map static_sets_code_disjoint]; [ reflexivity | ].
  rewrite IH. f_equal. apply forallb_map_with. intros binding'.
  rewrite binds_symbol_of_set_renamed, !static_set_code_ids_renamed.
  reflexivity.
Qed.

Fixpoint sets_well_formed_renamed (r : renaming) (e : expr) :
  sets_well_formed (rename_expr r e) = sets_well_formed e
with handler_sets_well_formed_renamed (r : renaming)
    (h : continuation_handler) :
  handler_sets_well_formed (rename_handler r h) = handler_sets_well_formed h
with binding_sets_well_formed_renamed (r : renaming)
    (binding : static_binding) :
  binding_sets_well_formed (rename_static_binding r binding)
  = binding_sets_well_formed binding
with code_sets_well_formed_renamed (r : renaming) (c : code) :
  code_sets_well_formed (rename_code r c) = code_sets_well_formed c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr sets_well_formed]; try reflexivity.
    + rewrite named_sets_well_formed_renamed, sets_well_formed_renamed.
      reflexivity.
    + rewrite static_sets_code_disjoint_renamed, sets_well_formed_renamed.
      rewrite (forallb_map_with _ _ (static_set_whole bindings))
        by apply static_set_whole_renamed.
      do 2 f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_sets_well_formed_renamed r binding), IH. reflexivity.
    + rewrite handler_sets_well_formed_renamed, sets_well_formed_renamed.
      reflexivity.
    + rewrite sets_well_formed_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_sets_well_formed_renamed r h), IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_sets_well_formed].
    apply sets_well_formed_renamed.
  - destruct binding as [id c | id | sym const];
      cbn [rename_static_binding binding_sets_well_formed]; try reflexivity.
    apply code_sets_well_formed_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_sets_well_formed]. apply sets_well_formed_renamed.
Qed.

Theorem sets_well_formed_equivariant : sets_well_formed_equivariant_statement.
Proof. exact sets_well_formed_renamed. Qed.

(* Numbers: a renaming changes no constant, no tag and no discriminant. *)
Lemma simple_in_range_renamed (r : renaming) (s : simple) :
  simple_in_range (rename_simple r s) = simple_in_range s.
Proof. destruct s; reflexivity. Qed.

Lemma simples_in_range_renamed (r : renaming) (args : list simple) :
  forallb simple_in_range (map (rename_simple r) args)
  = forallb simple_in_range args.
Proof. apply forallb_map_with, simple_in_range_renamed. Qed.

Lemma value_slots_in_range_renamed (r : renaming)
    (slots : list (value_slot * simple)) :
  value_slots_in_range
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_in_range slots.
Proof.
  unfold value_slots_in_range. apply forallb_map_with. intros slot.
  apply simple_in_range_renamed.
Qed.

Lemma named_in_range_renamed (r : renaming) (d : named) :
  named_in_range (rename_named r d) = named_in_range d.
Proof.
  destruct d; cbn [rename_named named_in_range];
    rewrite ?simples_in_range_renamed, ?value_slots_in_range_renamed,
      ?simple_in_range_renamed;
    reflexivity.
Qed.

Lemma static_const_in_range_renamed (r : renaming) (const : static_const) :
  static_const_in_range (rename_static_const r const)
  = static_const_in_range const.
Proof.
  destruct const; cbn [rename_static_const static_const_in_range];
    rewrite ?simples_in_range_renamed, ?value_slots_in_range_renamed,
      ?simple_in_range_renamed;
    reflexivity.
Qed.

Lemma apply_cont_in_range_renamed (r : renaming) (ac : apply_cont) :
  apply_cont_in_range (rename_apply_cont r ac) = apply_cont_in_range ac.
Proof. apply simples_in_range_renamed. Qed.

Lemma apply_in_range_renamed (r : renaming) (ap : apply_expr) :
  apply_in_range (rename_apply r ap) = apply_in_range ap.
Proof.
  unfold apply_in_range.
  rewrite exn_extra_simples_rename. unfold rename_apply.
  cbn [ap_callee ap_args].
  rewrite simple_in_range_renamed, !simples_in_range_renamed. reflexivity.
Qed.

Fixpoint numbers_in_range_renamed (r : renaming) (e : expr) :
  numbers_in_range (rename_expr r e) = numbers_in_range e
with handler_numbers_in_range_renamed (r : renaming)
    (h : continuation_handler) :
  handler_numbers_in_range (rename_handler r h) = handler_numbers_in_range h
with binding_numbers_in_range_renamed (r : renaming)
    (binding : static_binding) :
  binding_numbers_in_range (rename_static_binding r binding)
  = binding_numbers_in_range binding
with code_numbers_in_range_renamed (r : renaming) (c : code) :
  code_numbers_in_range (rename_code r c) = code_numbers_in_range c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr numbers_in_range].
    + rewrite named_in_range_renamed, numbers_in_range_renamed. reflexivity.
    + rewrite numbers_in_range_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_numbers_in_range_renamed r binding), IH. reflexivity.
    + rewrite handler_numbers_in_range_renamed, numbers_in_range_renamed.
      reflexivity.
    + rewrite numbers_in_range_renamed. f_equal.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_numbers_in_range_renamed r h), IH. reflexivity.
    + apply apply_in_range_renamed.
    + apply apply_cont_in_range_renamed.
    + rewrite simple_in_range_renamed. f_equal.
      apply forallb_map_with. intros arm. unfold arm_in_range. cbn [fst snd].
      rewrite apply_cont_in_range_renamed. reflexivity.
    + reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_numbers_in_range].
    apply numbers_in_range_renamed.
  - destruct binding as [id c | id | sym const];
      cbn [rename_static_binding binding_numbers_in_range]; try reflexivity.
    + apply code_numbers_in_range_renamed.
    + apply static_const_in_range_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_numbers_in_range]. apply numbers_in_range_renamed.
Qed.

Theorem numbers_in_range_equivariant : numbers_in_range_equivariant_statement.
Proof. exact numbers_in_range_renamed. Qed.

(* Exception handlers. *)

Lemma is_raise_renamed (r : renaming) (ac : apply_cont) :
  is_raise (rename_apply_cont r ac) = is_raise ac.
Proof.
  unfold is_raise. simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl; try reflexivity.
  apply rename_continuation_eqb.
Qed.

Lemma applies_renamed (r : renaming) (k : continuation) (ac : apply_cont) :
  applies (rename_continuation r k) (rename_apply_cont r ac) = applies k ac.
Proof.
  unfold applies. rewrite is_raise_renamed. simpl.
  rewrite rename_continuation_eqb. reflexivity.
Qed.

Lemma traps_apart_renamed (r : renaming) (k : continuation)
    (ac : apply_cont) :
  traps_apart (rename_continuation r k) (rename_apply_cont r ac)
  = traps_apart k ac.
Proof.
  unfold traps_apart. rewrite traps_renamed, is_raise_renamed. reflexivity.
Qed.

Lemma gives_extra_args_apply_renamed (r : renaming) (k : continuation)
    (ap : apply_expr) :
  Pos.eqb (rename_continuation r k) (ap_exn_continuation (rename_apply r ap))
  && negb (no_exn_extra_args (rename_apply r ap))
  = Pos.eqb k (ap_exn_continuation ap) && negb (no_exn_extra_args ap).
Proof.
  rewrite no_exn_extra_args_renamed. simpl.
  rewrite rename_continuation_eqb. reflexivity.
Qed.

Fixpoint applies_cont_renamed (r : renaming) (k : continuation) (e : expr) :
  applies_cont (rename_continuation r k) (rename_expr r e) = applies_cont k e
with handler_applies_cont_renamed (r : renaming) (k : continuation)
    (h : continuation_handler) :
  handler_applies_cont (rename_continuation r k) (rename_handler r h)
  = handler_applies_cont k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply applies_cont_renamed.
    + apply applies_cont_renamed.
    + rewrite handler_applies_cont_renamed, applies_cont_renamed.
      reflexivity.
    + rewrite applies_cont_renamed. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (handler_applies_cont_renamed r k h), IH. reflexivity.
    + apply rename_continuation_eqb.
    + exact (applies_renamed r k ac).
    + apply existsb_map_with. intros arm.
      exact (applies_renamed r k (snd arm)).
  - destruct h as [params metadata handler]. simpl.
    apply applies_cont_renamed.
Qed.

Fixpoint gives_extra_args_renamed (r : renaming) (k : continuation)
    (e : expr) :
  gives_extra_args (rename_continuation r k) (rename_expr r e)
  = gives_extra_args k e
with handler_gives_extra_args_renamed (r : renaming) (k : continuation)
    (h : continuation_handler) :
  handler_gives_extra_args (rename_continuation r k) (rename_handler r h)
  = handler_gives_extra_args k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr gives_extra_args]; try reflexivity.
    + apply gives_extra_args_renamed.
    + apply gives_extra_args_renamed.
    + rewrite handler_gives_extra_args_renamed, gives_extra_args_renamed.
      reflexivity.
    + rewrite gives_extra_args_renamed. f_equal.
      induction handlers as [ | [k' h] handlers IH]; cbn [map existsb];
        [ reflexivity | ].
      rewrite (handler_gives_extra_args_renamed r k h), IH. reflexivity.
    + apply gives_extra_args_apply_renamed.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_gives_extra_args].
    apply gives_extra_args_renamed.
Qed.

Fixpoint installs_or_removes_renamed (r : renaming) (k : continuation)
    (e : expr) :
  installs_or_removes (rename_continuation r k) (rename_expr r e)
  = installs_or_removes k e
with handler_installs_or_removes_renamed (r : renaming) (k : continuation)
    (h : continuation_handler) :
  handler_installs_or_removes (rename_continuation r k) (rename_handler r h)
  = handler_installs_or_removes k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr installs_or_removes]; try reflexivity.
    + apply installs_or_removes_renamed.
    + apply installs_or_removes_renamed.
    + rewrite handler_installs_or_removes_renamed,
        installs_or_removes_renamed.
      reflexivity.
    + rewrite installs_or_removes_renamed. f_equal.
      induction handlers as [ | [k' h] handlers IH]; cbn [map existsb];
        [ reflexivity | ].
      rewrite (handler_installs_or_removes_renamed r k h), IH. reflexivity.
    + apply traps_apart_renamed.
    + apply existsb_map_with. intros arm. cbn [snd].
      apply traps_apart_renamed.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_installs_or_removes].
    apply installs_or_removes_renamed.
Qed.

Lemma ways_out_well_formed_renamed (r : renaming) (exn : continuation)
    (e : expr) :
  ways_out_well_formed (rename_continuation r exn) (rename_expr r e)
  = ways_out_well_formed exn e.
Proof.
  unfold ways_out_well_formed. rewrite well_formed_free_continuations_renamed.
  apply forallb_map_with. intros k.
  rewrite rename_continuation_eqb, installs_or_removes_renamed,
    raises_to_equivariant_aux, gives_extra_args_renamed.
  reflexivity.
Qed.

Lemma takes_exception_renamed (r : renaming) (h : continuation_handler) :
  takes_exception (rename_handler r h) = takes_exception h.
Proof. destruct h as [[ | param params] metadata handler]; reflexivity. Qed.

Lemma handlers_raise_to_renamed (r : renaming) (k : continuation)
    (handlers : list (continuation * continuation_handler)) :
  existsb (fun '(_, h) => handler_raises_to (rename_continuation r k) h)
    (rename_handlers r handlers)
  = existsb (fun '(_, h) => handler_raises_to k h) handlers.
Proof.
  unfold rename_handlers. apply existsb_map_with. intros [k' h].
  apply handler_raises_to_equivariant.
Qed.

Fixpoint exn_handlers_well_formed_renamed (r : renaming) (e : expr)
    {struct e} :
  exn_handlers_well_formed (rename_expr r e) = exn_handlers_well_formed e
with handler_exn_handlers_well_formed_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  handler_exn_handlers_well_formed (rename_handler r h)
  = handler_exn_handlers_well_formed h
with binding_exn_handlers_well_formed_renamed (r : renaming)
    (binding : static_binding) {struct binding} :
  binding_exn_handlers_well_formed (rename_static_binding r binding)
  = binding_exn_handlers_well_formed binding
with code_exn_handlers_well_formed_renamed (r : renaming) (c : code)
    {struct c} :
  code_exn_handlers_well_formed (rename_code r c)
  = code_exn_handlers_well_formed c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr exn_handlers_well_formed]; try reflexivity.
    + apply exn_handlers_well_formed_renamed.
    + rewrite exn_handlers_well_formed_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_exn_handlers_well_formed_renamed r binding), IH.
      reflexivity.
    + rewrite handler_is_exn_handler_renamed, applies_cont_renamed,
        takes_exception_renamed, raises_to_equivariant_aux,
        handler_exn_handlers_well_formed_renamed,
        exn_handlers_well_formed_renamed.
      reflexivity.
    + rewrite exn_handlers_well_formed_renamed. f_equal.
      assert
        (Hgroup :
          ∀ group : list (continuation * continuation_handler),
            forallb
              (fun '(k, h) =>
                 (negb (handler_is_exn_handler h)
                  && negb (raises_to k (rename_expr r body))
                  && negb
                       (existsb (fun '(_, h') => handler_raises_to k h')
                          (rename_handlers r group))
                  && handler_exn_handlers_well_formed h)%bool)
              (map
                 (fun '(k, h) =>
                    (rename_continuation r k, rename_handler r h))
                 handlers)
            = forallb
                (fun '(k, h) =>
                   (negb (handler_is_exn_handler h)
                    && negb (raises_to k body)
                    && negb
                         (existsb (fun '(_, h') => handler_raises_to k h')
                            group)
                    && handler_exn_handlers_well_formed h)%bool)
                handlers).
      { intros group.
        induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
          [ reflexivity | ].
        rewrite IH, handler_is_exn_handler_renamed, raises_to_equivariant_aux,
          handlers_raise_to_renamed,
          (handler_exn_handlers_well_formed_renamed r h).
        reflexivity. }
      exact (Hgroup handlers).
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_exn_handlers_well_formed].
    apply exn_handlers_well_formed_renamed.
  - destruct binding as [id c | id | sym const];
      cbn [rename_static_binding binding_exn_handlers_well_formed];
      try reflexivity.
    apply code_exn_handlers_well_formed_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_exn_handlers_well_formed].
    rewrite ways_out_well_formed_renamed, applies_cont_renamed,
      exn_handlers_well_formed_renamed.
    reflexivity.
Qed.

Lemma unit_exn_handlers_well_formed_renamed (r : renaming)
    (unit_exn : continuation) (e : expr) :
  unit_exn_handlers_well_formed (rename_continuation r unit_exn)
    (rename_expr r e)
  = unit_exn_handlers_well_formed unit_exn e.
Proof.
  unfold unit_exn_handlers_well_formed.
  rewrite ways_out_well_formed_renamed, applies_cont_renamed,
    exn_handlers_well_formed_renamed.
  reflexivity.
Qed.

Theorem applies_cont_equivariant : applies_cont_equivariant_statement.
Proof. exact applies_cont_renamed. Qed.

Theorem gives_extra_args_equivariant :
  gives_extra_args_equivariant_statement.
Proof. exact gives_extra_args_renamed. Qed.

Theorem installs_or_removes_equivariant :
  installs_or_removes_equivariant_statement.
Proof. exact installs_or_removes_renamed. Qed.

Theorem ways_out_well_formed_equivariant :
  ways_out_well_formed_equivariant_statement.
Proof. exact ways_out_well_formed_renamed. Qed.

Theorem exn_handlers_well_formed_equivariant :
  exn_handlers_well_formed_equivariant_statement.
Proof. exact exn_handlers_well_formed_renamed. Qed.

Theorem unit_exn_handlers_well_formed_equivariant :
  unit_exn_handlers_well_formed_equivariant_statement.
Proof. exact unit_exn_handlers_well_formed_renamed. Qed.

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

Theorem well_formed_unit_equivariant : well_formed_unit_equivariant_statement.
Proof.
  intros r u e. unfold well_formed_unit, rename_unit_interface.
  cbn [unit_return unit_exn].
  rewrite statics_at_toplevel_equivariant_aux,
    (static_names_renamed r _ (code_ids_of_binding_renamed r)),
    (static_names_renamed r _ (symbols_of_binding_renamed r)),
    names_in_scope_equivariant, binders_distinct_equivariant,
    code_closed_equivariant, sets_well_formed_renamed,
    numbers_in_range_renamed, unit_exn_handlers_well_formed_renamed,
    traps_well_formed_renamed.
  reflexivity.
Qed.

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
   of a statement proves it. *)

(* A rewrite applied to the handlers of recursive continuations. *)
Section Rewrite_rec_handlers.

Variable f : expr -> expr.
Hypothesis f_equivariant : rewrite_equivariant f.

Fixpoint rewrite_rec_handlers_equivariant_aux (r : renaming) (e : expr) :
  rewrite_rec_handlers f (rename_expr r e)
  = rename_expr r (rewrite_rec_handlers f e)
with rewrite_rec_handlers_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  rewrite_rec_handlers_in_handler f (rename_handler r h)
  = rename_handler r (rewrite_rec_handlers_in_handler f h)
with rewrite_rec_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  rewrite_rec_handler f (rename_handler r h)
  = rename_handler r (rewrite_rec_handler f h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite rewrite_rec_handlers_equivariant_aux. reflexivity.
    + rewrite rewrite_rec_handlers_equivariant_aux. reflexivity.
    + rewrite rewrite_rec_handlers_in_handler_equivariant,
        rewrite_rec_handlers_equivariant_aux.
      reflexivity.
    + rewrite rewrite_rec_handlers_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (rewrite_rec_handler_equivariant r h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite rewrite_rec_handlers_equivariant_aux. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite rewrite_rec_handlers_equivariant_aux,
      (rewrite_code_body_equivariant f f_equivariant).
    reflexivity.
Qed.

Lemma rewrite_rec_handlers_equivariant :
  rewrite_equivariant (rewrite_rec_handlers f).
Proof. exact rewrite_rec_handlers_equivariant_aux. Qed.

End Rewrite_rec_handlers.

(* Moving a continuation to the handler that alone uses it. *)
Lemma applied_at_tail_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) :
  applied_at_tail (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = applied_at_tail k h body.
Proof.
  destruct h as [params metadata handler]. unfold applied_at_tail.
  cbn [rename_handler]. rewrite inline_cont_at_tail_equivariant_aux.
  destruct (inline_cont_at_tail k params metadata handler body); reflexivity.
Qed.

Lemma params_not_free_equivariant (r : renaming) (params : list variable)
    (h : continuation_handler) :
  params_not_free (map (rename_variable r) params) (rename_handler r h)
  = params_not_free params h.
Proof.
  unfold params_not_free. apply forallb_map_with. intros param.
  rewrite var_free_in_handler_b_equivariant. reflexivity.
Qed.

Fixpoint sink_into_handler_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) {struct body} :
  sink_into_handler (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = option_map (rename_expr r) (sink_into_handler k h body).
Proof.
  destruct body
    as [ x kind d body | bindings body | k' [params' metadata' handler'] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler sink_into_handler]; try reflexivity.
  - rewrite var_free_in_handler_b_equivariant,
      (sink_into_handler_equivariant r k h body).
    destruct (var_free_in_handler_b x h); [ reflexivity | ].
    destruct (sink_into_handler k h body); reflexivity.
  - rewrite (sink_into_handler_equivariant r k h body).
    destruct (sink_into_handler k h body); reflexivity.
  - rewrite !cont_free_b_equivariant_aux,
      !cont_free_in_handler_b_equivariant, params_not_free_equivariant,
      applied_at_tail_equivariant, rename_continuation_eqb,
      (sink_into_handler_equivariant r k h handler'),
      (sink_into_handler_equivariant r k h body).
    destruct
      (negb (cont_free_b k body) && negb (cont_free_in_handler_b k h)
       && params_not_free params' h).
    + destruct (sink_into_handler k h handler'); [ reflexivity | ].
      cbn [option_map].
      destruct (applied_at_tail k h handler'); reflexivity.
    + destruct
        (negb (Pos.eqb k k') && negb (cont_free_b k handler')
         && negb (cont_free_in_handler_b k' h));
        [ | reflexivity ].
      destruct (sink_into_handler k h body); reflexivity.
Qed.

Fixpoint sink_let_conts_equivariant_aux (r : renaming) (e : expr) :
  sink_let_conts (rename_expr r e) = rename_expr r (sink_let_conts e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler sink_let_conts]; try reflexivity.
  - rewrite sink_let_conts_equivariant_aux. reflexivity.
  - rewrite sink_let_conts_equivariant_aux. reflexivity.
  - rewrite !sink_let_conts_equivariant_aux.
    pose proof
      (sink_into_handler_equivariant r k
         (Handler params metadata (sink_let_conts handler))
         (sink_let_conts body))
      as Hsink.
    cbn [rename_handler] in Hsink. rewrite Hsink.
    destruct
      (sink_into_handler k (Handler params metadata (sink_let_conts handler))
         (sink_let_conts body));
      reflexivity.
  - rewrite sink_let_conts_equivariant_aux. reflexivity.
Qed.

Lemma sink_let_conts_equivariant : rewrite_equivariant sink_let_conts.
Proof. exact sink_let_conts_equivariant_aux. Qed.

(* Moving a continuation down past the bindings under it. *)
Fixpoint sink_to_use_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (body : expr) {struct body} :
  sink_to_use (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_expr r (sink_to_use k h body).
Proof.
  destruct body
    as [ x kind d body | bindings body | k' [params' metadata' handler'] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler sink_to_use]; try reflexivity.
  - rewrite var_free_in_handler_b_equivariant,
      (sink_to_use_equivariant r k h body).
    destruct (var_free_in_handler_b x h); reflexivity.
  - rewrite (sink_to_use_equivariant r k h body). reflexivity.
  - rewrite !cont_free_b_equivariant_aux,
      !cont_free_in_handler_b_equivariant, params_not_free_equivariant,
      rename_continuation_eqb, (sink_to_use_equivariant r k h body),
      (sink_to_use_equivariant r k h handler').
    destruct
      (negb (Pos.eqb k k') && negb (cont_free_b k handler')
       && negb (cont_free_in_handler_b k' h));
      [ reflexivity | ].
    destruct
      (negb (cont_free_b k body) && negb (cont_free_in_handler_b k h)
       && params_not_free params' h);
      reflexivity.
Qed.

Fixpoint sink_conts_to_uses_equivariant_aux (r : renaming) (e : expr) :
  sink_conts_to_uses (rename_expr r e)
  = rename_expr r (sink_conts_to_uses e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler sink_conts_to_uses]; try reflexivity.
  - rewrite sink_conts_to_uses_equivariant_aux. reflexivity.
  - rewrite sink_conts_to_uses_equivariant_aux. reflexivity.
  - rewrite !sink_conts_to_uses_equivariant_aux.
    exact
      (sink_to_use_equivariant r k
         (Handler params metadata (sink_conts_to_uses handler))
         (sink_conts_to_uses body)).
  - rewrite sink_conts_to_uses_equivariant_aux. reflexivity.
Qed.

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
Lemma may_be_static_renamed (r : renaming) (d : named) :
  may_be_static (rename_named r d) = may_be_static d.
Proof. destruct d; reflexivity. Qed.

Fixpoint straight_renamed (r : renaming) (e : expr) {struct e} :
  ∀ lets : nat, straight lets (rename_expr r e) = straight lets e.
Proof.
  destruct e as [ x kind d body | | | | | ac | | ]; intros lets;
    cbn [rename_expr straight]; try reflexivity.
  - rewrite may_be_static_renamed.
    destruct (may_be_static d), lets; try reflexivity.
    apply straight_renamed.
  - apply no_trap_action_renamed.
Qed.

Lemma sharable_renamed (r : renaming) (h : continuation_handler) :
  sharable (rename_handler r h) = sharable h.
Proof.
  destruct h as [params metadata handler].
  destruct handler as [ x kind d body | | | | | | | ]; try reflexivity.
  cbn [rename_handler].
  change (sharable
            (Handler (map (rename_variable r) params) metadata
               (rename_expr r (Let x kind d body))))
    with (negb (hm_is_exn_handler metadata)
          && straight 31 (rename_expr r (Let x kind d body))).
  rewrite straight_renamed. reflexivity.
Qed.

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

Fixpoint share_with_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (e : expr) {struct e} :
  ∀ (xs : list variable) (ks : list continuation),
    share_with (rename_continuation r k) (rename_handler r h)
      (map (rename_variable r) xs) (map (rename_continuation r) ks)
      (rename_expr r e)
    = rename_expr r (share_with k h xs ks e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k' [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    intros xs ks; cbn [rename_expr rename_handler share_with];
    try reflexivity.
  - rewrite <- (share_with_equivariant r k h body (x :: xs) ks).
    reflexivity.
  - rewrite (share_with_equivariant r k h body xs ks). reflexivity.
  - rewrite rename_continuation_eqb.
    destruct (Pos.eqb k' k); [ reflexivity | ].
    cbn [rename_expr].
    change (Handler (map (rename_variable r) params) metadata
              (rename_expr r handler))
      with (rename_handler r (Handler params metadata handler)).
    rewrite same_handler_renamed.
    rewrite <- (share_with_equivariant r k h body xs (k' :: ks)).
    destruct (same_handler h xs ks (Handler params metadata handler));
      cbn [rename_expr rename_handler map]; [ | reflexivity ].
    unfold rename_apply_cont. cbn [ac_continuation ac_args ac_trap_action].
    rewrite !map_map. reflexivity.
Qed.

Fixpoint share_handlers_equivariant_aux (r : renaming) (e : expr) :
  share_handlers (rename_expr r e) = rename_expr r (share_handlers e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler share_handlers]; cbv zeta;
    try reflexivity.
  - rewrite share_handlers_equivariant_aux. reflexivity.
  - rewrite share_handlers_equivariant_aux. reflexivity.
  - rewrite !share_handlers_equivariant_aux.
    change (Handler (map (rename_variable r) params) metadata
              (rename_expr r (share_handlers handler)))
      with (rename_handler r
              (Handler params metadata (share_handlers handler))).
    rewrite sharable_renamed.
    destruct (sharable (Handler params metadata (share_handlers handler)));
      [ | reflexivity ].
    rewrite <- (share_with_equivariant r k
                  (Handler params metadata (share_handlers handler))
                  (share_handlers body) [] [k]).
    reflexivity.
  - rewrite share_handlers_equivariant_aux. reflexivity.
Qed.

Lemma share_handlers_equivariant : rewrite_equivariant share_handlers.
Proof. exact share_handlers_equivariant_aux. Qed.

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

Lemma mask_of_renamed (r : renaming) (known : masks) (k : continuation) :
  mask_of (Keyed.map_keys (rename_continuation r) known)
    (rename_continuation r k)
  = mask_of known k.
Proof.
  unfold mask_of, Keyed.map_keys, Keyed.map_bindings.
  assert (Hlookup :
    lookup (rename_continuation r k)
      (map (fun binding => (rename_continuation r (fst binding), snd binding))
         known)
    = lookup k known).
  { induction known as [ | [k' mask] known IH]; simpl; [ reflexivity | ].
    rewrite rename_continuation_eqb.
    destruct (Pos.eqb k k'); [ reflexivity | exact IH ]. }
  rewrite Hlookup. reflexivity.
Qed.

Lemma drop_args_equivariant (r : renaming) (known : masks) (ac : apply_cont) :
  drop_args (Keyed.map_keys (rename_continuation r) known)
    (rename_apply_cont r ac)
  = rename_apply_cont r (drop_args known ac).
Proof.
  unfold drop_args, rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action].
  rewrite mask_of_renamed, keep_all_or_map. reflexivity.
Qed.

Fixpoint returned_to_equivariant (r : renaming) (k : continuation)
    (e : expr) :
  returned_to (rename_continuation r k) (rename_expr r e) = returned_to k e
with returned_to_in_handler_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) :
  returned_to_in_handler (rename_continuation r k) (rename_handler r h)
  = returned_to_in_handler k h.
Proof.
  - destruct e
      as [ x kind d body | bindings body | bound h body | handlers body | ap
         | ac | scrutinee arms | ];
      simpl; try reflexivity.
    + apply returned_to_equivariant.
    + apply returned_to_equivariant.
    + rewrite returned_to_in_handler_equivariant, returned_to_equivariant.
      reflexivity.
    + rewrite returned_to_equivariant. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (returned_to_in_handler_equivariant r k h), IH. reflexivity.
    + rewrite !rename_continuation_eqb. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply returned_to_equivariant.
Qed.

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

Lemma used_params_in_loop_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) (whole : expr) :
  used_params_in_loop (rename_continuation r k) (rename_handler r h)
    (rename_expr r whole)
  = used_params_in_loop k h whole.
Proof.
  destruct h as [params metadata handler]. simpl.
  rewrite returned_to_equivariant, used_params_equivariant. reflexivity.
Qed.

Lemma loop_masks_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler)) (whole : expr)
    (known : masks) :
  map (fun '(k, h) => (k, used_params_in_loop k h (rename_expr r whole)))
    (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
       handlers)
  ++ Keyed.map_keys (rename_continuation r) known
  = Keyed.map_keys (rename_continuation r)
      (map (fun '(k, h) => (k, used_params_in_loop k h whole)) handlers
       ++ known).
Proof.
  unfold Keyed.map_keys, Keyed.map_bindings. rewrite map_app. f_equal.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite used_params_in_loop_equivariant, IH. reflexivity.
Qed.

Fixpoint drop_params_equivariant_aux (r : renaming) (e : expr) :
  ∀ known : masks,
    drop_params (Keyed.map_keys (rename_continuation r) known)
      (rename_expr r e)
    = rename_expr r (drop_params known e)
with drop_params_in_loop_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (known : masks) (mask : option (list bool)),
    drop_params_in_loop (Keyed.map_keys (rename_continuation r) known) mask
      (rename_handler r h)
    = rename_handler r (drop_params_in_loop known mask h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros known; cbn [rename_expr rename_handler drop_params];
      try reflexivity.
    + rewrite drop_params_equivariant_aux. reflexivity.
    + rewrite drop_params_equivariant_aux. reflexivity.
    + rewrite (drop_params_equivariant_aux r handler known),
        returned_to_equivariant, used_params_equivariant, keep_all_or_map.
      change
        ((rename_continuation r k,
          if returned_to k body then None
          else
            used_params (hm_is_exn_handler metadata) params
              (drop_params known handler))
         :: Keyed.map_keys (rename_continuation r) known)
        with
          (Keyed.map_keys (rename_continuation r)
             ((k,
               if returned_to k body then None
               else
                 used_params (hm_is_exn_handler metadata) params
                   (drop_params known handler))
              :: known)).
      rewrite drop_params_equivariant_aux. reflexivity.
    + change
        (Let_cont_recursive
           (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
              handlers)
           (rename_expr r body))
        with (rename_expr r (Let_cont_recursive handlers body)).
      rewrite loop_masks_renamed, drop_params_equivariant_aux. f_equal.
      generalize (Let_cont_recursive handlers body). intros whole.
      generalize
        (map (fun '(k, h) => (k, used_params_in_loop k h whole)) handlers
         ++ known).
      intros known'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite used_params_in_loop_equivariant,
        (drop_params_in_loop_equivariant r h known'), IH.
      reflexivity.
    + rewrite drop_args_equivariant. reflexivity.
    + f_equal. rewrite !map_map. apply map_ext. intros arm. cbn [fst snd].
      rewrite drop_args_equivariant. reflexivity.
  - destruct h as [params metadata handler]. intros known mask.
    cbn [rename_handler drop_params_in_loop].
    rewrite drop_params_equivariant_aux, keep_all_or_map. reflexivity.
Qed.

Lemma drop_unused_params_equivariant : rewrite_equivariant drop_unused_params.
Proof. intros r e. exact (drop_params_equivariant_aux r e []). Qed.

(* Removing the parameters of a loop that are only passed round it. The
   law of the pass is proved from the definitions by the tactic of
   FlambdaEquivarianceTactic.v, which states the law of [drop_dead] from
   its type and proves it as it goes: the masks are renamed as their type
   says, the name of each continuation and nothing of what a mask holds.

   What is stated here is what the tactic does not do by itself.
   - The search for the masks of a loop ([settle]) takes a function, so
     its law is a lemma about two functions that agree up to renaming,
     and a rule gives the lemma to the tactic. Computation is kept from
     unfolding [settle], so that the rule finds it.
   - The hypothesis of the induction is about renamed masks. In a loop
     the masks are a list made of renamed parts, and where the call is
     under a function whose answer has no names the tactic does not use
     the hypothesis from right to left. A second rule takes the masks
     from the other side of the equation and shows that the list is
     those masks renamed.
   - [returned_to] is asked of the whole term of the case, which
     computation has taken apart: it is kept from unfolding, and its law
     is stated at that form.
   Four laws that are proved by hand are given to the tactic: three
   that are above, and that of [var_free_b], which is in
   FlambdaValidatorEquivarianceBaseProofs.v. The tactic does not derive
   those of [used_params] and of [keep_all_or]; with the law of
   [var_free_b] that it derives, the proof is not complete; and without
   that of [returned_to] it leaves the goal. *)
From OxCamlFlambda2 Require Import FlambdaEquivarianceTactic.

Section Dead_params.

#[local] Hint Rewrite @keep_all_or_map var_free_b_equivariant_aux
  returned_to_equivariant used_params_equivariant : equivariance.

#[local] Arguments returned_to : simpl never.

#[local] Abbreviation rename_masks r :=
  (map (fun p : continuation * option (list bool) =>
          (rename_continuation r (fst p), snd p))).

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
#[local] Hint Rewrite returned_to_rec_renamed : equivariance.

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

(* The two rules. The first: [settle] at a function and at first masks
   that mention the renaming, where the other side has it at ones that
   do not. The second: [drop_dead] at masks that are not written as a
   renamed list, of a term that the induction has a hypothesis for,
   where the other side has it at masks that do not mention the
   renaming. *)
#[local] Ltac eqv_extra ::=
  first
    [ match goal with
      | r : renaming |- context [settle ?fuel ?next' ?first'] =>
        eqv_has_renaming next';
        match goal with
        | |- context [settle fuel ?next ?first] =>
          eqv_no_renaming next;
          rewrite (settle_renamed r fuel next next' first first')
        end
      end
    | match goal with
      | r : renaming,
        H : context [drop_dead _ (rename_expr _ ?e)
                     = rename_expr _ (drop_dead _ ?e)]
        |- context [drop_dead ?m' (rename_expr ?r ?e)] =>
        lazymatch m' with
        | map _ _ => fail
        | _ => idtac
        end;
        match goal with
        | |- context [drop_dead ?m e] =>
          eqv_no_renaming m;
          replace m' with (rename_masks r m);
          [ rewrite H | solve [ eqv_core ] ]
        end
      end ].

Lemma drop_dead_params_equivariant : rewrite_equivariant drop_dead_params.
Proof. equivariance. Qed.

End Dead_params.

(* Removing the unused [Let]s whose defining expression has a value, in
   the handlers of recursive continuations too. *)
Fixpoint remove_unused_defined_lets_in_loops_equivariant_aux (r : renaming)
    (e : expr) :
  ∀ kinds : kinds_in_scope,
    remove_unused_defined_lets_in_loops (rename_kinds_in_scope r kinds)
      (rename_expr r e)
    = rename_expr r (remove_unused_defined_lets_in_loops kinds e)
with remove_unused_defined_lets_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ kinds : kinds_in_scope,
    remove_unused_defined_lets_in_handler (rename_kinds_in_scope r kinds)
      (rename_handler r h)
    = rename_handler r (remove_unused_defined_lets_in_handler kinds h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros kinds;
      cbn [rename_expr remove_unused_defined_lets_in_loops];
      try reflexivity.
    + change ((rename_variable r x, kind) :: rename_kinds_in_scope r kinds)
        with (rename_kinds_in_scope r ((x, kind) :: kinds)).
      rewrite remove_unused_defined_lets_in_loops_equivariant_aux,
        named_defined_equivariant, var_free_b_equivariant_aux.
      destruct
        (named_defined kinds kind d
         && negb
              (var_free_b x
                 (remove_unused_defined_lets_in_loops ((x, kind) :: kinds)
                    body)));
        reflexivity.
    + rewrite remove_unused_defined_lets_in_loops_equivariant_aux.
      reflexivity.
    + rewrite remove_unused_defined_lets_in_handler_equivariant,
        remove_unused_defined_lets_in_loops_equivariant_aux.
      reflexivity.
    + rewrite remove_unused_defined_lets_in_loops_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (remove_unused_defined_lets_in_handler_equivariant r h kinds),
        IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros kinds.
    cbn [rename_handler remove_unused_defined_lets_in_handler].
    rewrite params_in_scope_equivariant,
      remove_unused_defined_lets_in_loops_equivariant_aux.
    reflexivity.
Qed.

(* The same in the bodies of code, with the parameters of the code in
   scope. *)
Lemma remove_unused_defined_lets_in_binding_equivariant (r : renaming)
    (binding : static_binding) :
  remove_unused_defined_lets_in_binding (rename_static_binding r binding)
  = rename_static_binding r (remove_unused_defined_lets_in_binding binding).
Proof.
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    try reflexivity.
  cbn [rename_static_binding rename_code
       remove_unused_defined_lets_in_binding].
  unfold code_in_scope.
  change [(rename_variable r my_closure, Value)]
    with (rename_kinds_in_scope r [(my_closure, Value)]).
  rewrite params_in_scope_equivariant,
    remove_unused_defined_lets_in_loops_equivariant_aux.
  reflexivity.
Qed.

Fixpoint remove_unused_defined_lets_in_code_equivariant_aux (r : renaming)
    (e : expr) :
  remove_unused_defined_lets_in_code (rename_expr r e)
  = rename_expr r (remove_unused_defined_lets_in_code e)
with remove_unused_defined_lets_in_code_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  remove_unused_defined_lets_in_code_handler (rename_handler r h)
  = rename_handler r (remove_unused_defined_lets_in_code_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite remove_unused_defined_lets_in_code_equivariant_aux.
      reflexivity.
    + rewrite remove_unused_defined_lets_in_code_equivariant_aux, !map_map.
      f_equal. apply map_ext.
      exact (remove_unused_defined_lets_in_binding_equivariant r).
    + rewrite remove_unused_defined_lets_in_code_handler_equivariant,
        remove_unused_defined_lets_in_code_equivariant_aux.
      reflexivity.
    + rewrite remove_unused_defined_lets_in_code_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (remove_unused_defined_lets_in_code_handler_equivariant r h),
        IH.
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite remove_unused_defined_lets_in_code_equivariant_aux. reflexivity.
Qed.

Lemma remove_unused_defined_lets_in_code_equivariant :
  rewrite_equivariant remove_unused_defined_lets_in_code.
Proof. exact remove_unused_defined_lets_in_code_equivariant_aux. Qed.

(* Binding the parameters that every application gives the same thing. *)
Lemma simple_eqb_renamed (r : renaming) (s s' : simple) :
  simple_eqb (rename_simple r s) (rename_simple r s') = simple_eqb s s'.
Proof.
  unfold simple_eqb.
  destruct (simple_eq_dec (rename_simple r s) (rename_simple r s'))
    as [Heq | Hne'],
    (simple_eq_dec s s') as [Heq' | Hne];
    try reflexivity.
  - exfalso. apply Hne, (rename_simple_injective r), Heq.
  - exfalso. apply Hne'. rewrite Heq'. reflexivity.
Qed.

Fixpoint binds_var_equivariant (r : renaming) (x : variable) (e : expr) :
  binds_var (rename_variable r x) (rename_expr r e) = binds_var x e
with binds_var_in_handler_equivariant (r : renaming) (x : variable)
    (h : continuation_handler) :
  binds_var_in_handler (rename_variable r x) (rename_handler r h)
  = binds_var_in_handler x h.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite rename_variable_eqb, binds_var_equivariant. reflexivity.
    + apply binds_var_equivariant.
    + rewrite binds_var_in_handler_equivariant, binds_var_equivariant.
      reflexivity.
    + rewrite binds_var_equivariant. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (binds_var_in_handler_equivariant r x h), IH. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite among_rename_variable, binds_var_equivariant. reflexivity.
Qed.

Lemma stable_equivariant (r : renaming) (s : simple) (e : expr) :
  stable (rename_simple r s) (rename_expr r e) = stable s e.
Proof.
  destruct s; simpl; [ rewrite binds_var_equivariant | | ]; reflexivity.
Qed.

Lemma args_if_applies_equivariant (r : renaming) (k : continuation)
    (ac : apply_cont) :
  args_if_applies (rename_continuation r k) (rename_apply_cont r ac)
  = option_map (map (rename_simple r)) (args_if_applies k ac).
Proof.
  unfold args_if_applies, rename_apply_cont.
  cbn [ac_continuation ac_args]. rewrite rename_continuation_eqb.
  destruct (Pos.eqb (ac_continuation ac) k); reflexivity.
Qed.

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
Proof.
  induction params as [ | p params IH]; intros kinds aliases handler;
    [ reflexivity | ].
  destruct kinds as [ | kind kinds]; [ reflexivity | ].
  destruct aliases as [ | [s | ] aliases]; [ reflexivity | | ]; simpl;
    rewrite IH; reflexivity.
Qed.

Lemma alone_map {A B : Type} (g : A -> B) (l : list A) :
  alone (map g l) = alone l.
Proof. destruct l as [ | a [ | b l]]; reflexivity. Qed.

Fixpoint bind_aliased_params_equivariant_aux (r : renaming) (e : expr) :
  bind_aliased_params (rename_expr r e) = rename_expr r (bind_aliased_params e)
with bind_aliased_params_in_loop_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (flag : bool) (k : continuation) (body whole : expr),
    bind_aliased_params_in_loop flag (rename_continuation r k)
      (rename_expr r body) (rename_expr r whole) (rename_handler r h)
    = rename_handler r (bind_aliased_params_in_loop flag k body whole h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      try reflexivity.
    + cbn [rename_expr bind_aliased_params].
      rewrite bind_aliased_params_equivariant_aux. reflexivity.
    + cbn [rename_expr bind_aliased_params].
      rewrite bind_aliased_params_equivariant_aux. reflexivity.
    + cbn [rename_expr rename_handler bind_aliased_params].
      change
        (Let_cont_non_recursive (rename_continuation r k)
           (Handler (map (rename_variable r) params) metadata
              (rename_expr r handler))
           (rename_expr r body))
        with
          (rename_expr r
             (Let_cont_non_recursive k (Handler params metadata handler)
                body)).
      rewrite
        (aliases_of_equivariant r k params metadata None None body _ eq_refl),
        (bind_aliased_params_equivariant_aux r handler),
        bind_aliases_equivariant,
        (bind_aliased_params_equivariant_aux r body).
      reflexivity.
    + cbn [rename_expr bind_aliased_params].
      rewrite (bind_aliased_params_equivariant_aux r body), alone_map.
      change
        (Let_cont_recursive
           (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
              handlers)
           (rename_expr r body))
        with (rename_expr r (Let_cont_recursive handlers body)).
      f_equal.
      generalize (alone handlers) (Let_cont_recursive handlers body).
      intros flag whole.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (bind_aliased_params_in_loop_equivariant r h flag k body whole),
        IH.
      reflexivity.
  - destruct h as [params metadata handler]. intros flag k body whole.
    cbn [rename_handler bind_aliased_params_in_loop].
    rewrite (bind_aliased_params_equivariant_aux r handler).
    destruct flag.
    + rewrite
        (aliases_of_equivariant r k params metadata (Some handler)
           (Some (rename_expr r handler)) body whole eq_refl),
        bind_aliases_equivariant.
      reflexivity.
    + change (@nil (option simple))
        with (map (option_map (rename_simple r)) []).
      rewrite bind_aliases_equivariant. reflexivity.
Qed.

Lemma bind_aliased_params_equivariant :
  rewrite_equivariant bind_aliased_params.
Proof. exact bind_aliased_params_equivariant_aux. Qed.

(* An application made to go where the handler of its continuation
   goes. The entries are keyed by continuations, which are renamed, and
   keep parameters and an [Apply_cont], which are renamed too. *)
Fixpoint binds_cont_equivariant (r : renaming) (k : continuation) (e : expr) :
  binds_cont (rename_continuation r k) (rename_expr r e) = binds_cont k e
with binds_cont_in_handler_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) :
  binds_cont_in_handler (rename_continuation r k) (rename_handler r h)
  = binds_cont_in_handler k h.
Proof.
  - destruct e
      as [ y kind d body | bindings body | k' h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + apply binds_cont_equivariant.
    + apply binds_cont_equivariant.
    + rewrite rename_continuation_eqb, binds_cont_in_handler_equivariant,
        binds_cont_equivariant.
      reflexivity.
    + rewrite binds_cont_equivariant. f_equal.
      induction handlers as [ | [k' h] handlers IH]; simpl; [ reflexivity | ].
      rewrite rename_continuation_eqb,
        (binds_cont_in_handler_equivariant r k h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    apply binds_cont_equivariant.
Qed.

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
Proof.
  destruct s as [x | sym | c]; try reflexivity.
  cbn [rename_simple subst_simple].
  assert (Hlookup :
    lookup (rename_variable r x)
      (combine (map (rename_variable r) params) (map (rename_simple r) args))
    = option_map (rename_simple r) (lookup x (combine params args))).
  { revert args.
    induction params as [ | p params IH]; intros [ | arg args]; simpl;
      try reflexivity.
    rewrite rename_variable_eqb.
    destruct (Pos.eqb x p); [ reflexivity | apply IH ]. }
  rewrite Hlookup.
  destruct (lookup x (combine params args)); reflexivity.
Qed.

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

Fixpoint forward_applications_under_equivariant (r : renaming) (e : expr) :
  ∀ (calls : bool) (F : forwards),
    forward_applications_under calls (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_applications_under calls F e)
with forward_applications_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (calls : bool) (F : forwards),
    forward_applications_in_handler calls (rename_forwards r F)
      (rename_handler r h)
    = rename_handler r (forward_applications_in_handler calls F h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros calls F; try reflexivity.
    + cbn [rename_expr forward_applications_under].
      rewrite forward_applications_under_equivariant. reflexivity.
    + cbn [rename_expr forward_applications_under].
      rewrite forward_applications_under_equivariant. reflexivity.
    + cbn [rename_expr rename_handler forward_applications_under]. cbv zeta.
      rewrite (forward_applications_under_equivariant r handler calls F),
        forwarding_equivariant.
      change
        ((rename_continuation r k,
          rename_forward_entry r
            (forwarding k params metadata
               (forward_applications_under calls F handler) body))
         :: rename_forwards r F)
        with
          (rename_forwards r
             ((k,
               forwarding k params metadata
                 (forward_applications_under calls F handler) body)
              :: F)).
      rewrite (forward_applications_under_equivariant r body). reflexivity.
    + cbn [rename_expr forward_applications_under]. cbv zeta.
      assert (Hscope :
        map (fun '(k, _) => (k, None))
          (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
             handlers)
        ++ rename_forwards r F
        = rename_forwards r
            (map (fun '(k, _) => (k, None)) handlers ++ F)).
      { unfold rename_forwards. rewrite map_app, !map_map. f_equal.
        apply map_ext. intros [k h]. reflexivity. }
      rewrite Hscope. clear Hscope.
      rewrite (forward_applications_under_equivariant r body). f_equal.
      generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                    handlers ++ F).
      intros F'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_applications_in_handler_equivariant r h calls F'), IH.
      reflexivity.
    + cbn [rename_expr forward_applications_under].
      destruct calls; [ | reflexivity ].
      rewrite forward_apply_equivariant. reflexivity.
    + cbn [rename_expr forward_applications_under].
      rewrite forward_apply_cont_keeping_equivariant. reflexivity.
    + cbn [rename_expr forward_applications_under]. f_equal.
      rewrite !map_map. apply map_ext. intros [i ac]. cbn [fst snd].
      rewrite forward_arm_equivariant. reflexivity.
  - destruct h as [params metadata handler]. intros calls F.
    cbn [rename_handler forward_applications_in_handler].
    rewrite forward_applications_under_equivariant. reflexivity.
Qed.

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

Fixpoint forward_jumps_under_equivariant (r : renaming) (e : expr) :
  ∀ F : forwards,
    forward_jumps_under (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_jumps_under F e)
with forward_jumps_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ F : forwards,
    forward_jumps_in_handler (rename_forwards r F)
      (rename_handler r h)
    = rename_handler r (forward_jumps_in_handler F h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros F; try reflexivity.
    + cbn [rename_expr forward_jumps_under].
      rewrite forward_jumps_under_equivariant. reflexivity.
    + cbn [rename_expr forward_jumps_under].
      rewrite forward_jumps_under_equivariant. reflexivity.
    + cbn [rename_expr rename_handler forward_jumps_under]. cbv zeta.
      rewrite (forward_jumps_under_equivariant r handler F),
        jump_entry_equivariant.
      change
        ((rename_continuation r k,
          rename_forward_entry r
            (jump_entry k params metadata
               (forward_jumps_under F handler) body))
         :: rename_forwards r F)
        with
          (rename_forwards r
             ((k,
               jump_entry k params metadata
                 (forward_jumps_under F handler) body)
              :: F)).
      rewrite (forward_jumps_under_equivariant r body). reflexivity.
    + cbn [rename_expr forward_jumps_under]. cbv zeta.
      assert (Hscope :
        map (fun '(k, _) => (k, None))
          (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
             handlers)
        ++ rename_forwards r F
        = rename_forwards r
            (map (fun '(k, _) => (k, None)) handlers ++ F)).
      { unfold rename_forwards. rewrite map_app, !map_map. f_equal.
        apply map_ext. intros [k h]. reflexivity. }
      rewrite Hscope. clear Hscope.
      rewrite (forward_jumps_under_equivariant r body). f_equal.
      generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                    handlers ++ F).
      intros F'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_jumps_in_handler_equivariant r h F'), IH.
      reflexivity.
    + cbn [rename_expr forward_jumps_under].
      rewrite forward_apply_cont_equivariant. reflexivity.
    + cbn [rename_expr forward_jumps_under]. f_equal.
      rewrite !map_map. apply map_ext. intros [i ac]. cbn [fst snd].
      rewrite forward_apply_cont_equivariant. reflexivity.
  - destruct h as [params metadata handler]. intros F.
    cbn [rename_handler forward_jumps_in_handler].
    rewrite forward_jumps_under_equivariant. reflexivity.
Qed.

Lemma forward_jumps_equivariant : rewrite_equivariant forward_jumps.
Proof.
  intros r e. exact (forward_jumps_under_equivariant r e []).
Qed.

(* The same where the arguments are known to have the kinds of the
   parameters (FlambdaForwardKindedSpec.v). The kinds of the variables
   in scope are renamed with the variables; those that the handlers
   of the continuations in scope give their parameters are kept by
   continuation, and a renaming acts on the continuations. *)
Definition rename_cont_kinds (r : renaming) (K : cont_kinds) : cont_kinds :=
  map (fun entry => (rename_continuation r (fst entry), snd entry)) K.

Lemma lookup_rename_cont_kinds (r : renaming) (k : continuation)
    (K : cont_kinds) :
  lookup (rename_continuation r k) (rename_cont_kinds r K) = lookup k K.
Proof.
  induction K as [ | [k' kinds] K IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb. destruct (Pos.eqb k k'); [ reflexivity | ].
  exact IH.
Qed.

Lemma have_kinds_equivariant (r : renaming) (G : kinds_in_scope)
    (args : list simple) :
  ∀ kinds : list flambda_kind,
    have_kinds (rename_kinds_in_scope r G) (map (rename_simple r) args) kinds
    = have_kinds G args kinds.
Proof.
  induction args as [ | s args IH]; intros [ | kind kinds];
    cbn [map have_kinds]; try reflexivity.
  rewrite kind_of_simple_in_scope_equivariant, IH. reflexivity.
Qed.

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

Fixpoint forward_kinded_under_equivariant (r : renaming) (e : expr) :
  ∀ (G : kinds_in_scope) (K : cont_kinds) (F : forwards),
    forward_kinded_under (rename_kinds_in_scope r G) (rename_cont_kinds r K)
      (rename_forwards r F) (rename_expr r e)
    = rename_expr r (forward_kinded_under G K F e)
with forward_kinded_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (G : kinds_in_scope) (K : cont_kinds) (F : forwards),
    forward_kinded_in_handler (rename_kinds_in_scope r G)
      (rename_cont_kinds r K) (rename_forwards r F) (rename_handler r h)
    = rename_handler r (forward_kinded_in_handler G K F h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros G K F; try reflexivity.
    + cbn [rename_expr forward_kinded_under].
      change ((rename_variable r x, kind) :: rename_kinds_in_scope r G)
        with (rename_kinds_in_scope r ((x, kind) :: G)).
      rewrite forward_kinded_under_equivariant. reflexivity.
    + cbn [rename_expr forward_kinded_under].
      rewrite forward_kinded_under_equivariant. reflexivity.
    + cbn [rename_expr rename_handler forward_kinded_under]. cbv zeta.
      rewrite params_in_scope_equivariant,
        (forward_kinded_under_equivariant r handler),
        kinded_entry_equivariant.
      change
        ((rename_continuation r k, hm_params_kinds metadata)
         :: rename_cont_kinds r K)
        with (rename_cont_kinds r ((k, hm_params_kinds metadata) :: K)).
      change
        ((rename_continuation r k,
          rename_forward_entry r
            (kinded_entry k params metadata
               (forward_kinded_under
                  (params_in_scope params (hm_params_kinds metadata) G) K F
                  handler)
               body))
         :: rename_forwards r F)
        with
          (rename_forwards r
             ((k,
               kinded_entry k params metadata
                 (forward_kinded_under
                    (params_in_scope params (hm_params_kinds metadata) G) K
                    F handler)
                 body)
              :: F)).
      rewrite (forward_kinded_under_equivariant r body). reflexivity.
    + cbn [rename_expr forward_kinded_under]. cbv zeta.
      assert (Hscope :
        map (fun '(k, _) => (k, None))
          (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
             handlers)
        ++ rename_forwards r F
        = rename_forwards r
            (map (fun '(k, _) => (k, None)) handlers ++ F)).
      { unfold rename_forwards. rewrite map_app, !map_map. f_equal.
        apply map_ext. intros [k h]. reflexivity. }
      rewrite Hscope. clear Hscope.
      rewrite (forward_kinded_under_equivariant r body). f_equal.
      generalize (map (fun '(k, _) => (k, @None (list variable * apply_cont)))
                    handlers ++ F).
      intros F'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_kinded_in_handler_equivariant r h G K F'), IH.
      reflexivity.
    + cbn [rename_expr forward_kinded_under].
      rewrite forward_kinded_apply_cont_equivariant. reflexivity.
    + cbn [rename_expr forward_kinded_under]. f_equal.
      rewrite !map_map. apply map_ext. intros [i ac]. cbn [fst snd].
      rewrite forward_kinded_arm_equivariant. reflexivity.
  - destruct h as [params metadata handler]. intros G K F.
    cbn [rename_handler forward_kinded_in_handler].
    rewrite params_in_scope_equivariant, forward_kinded_under_equivariant.
    reflexivity.
Qed.

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

Fixpoint forward_kinded_in_code_equivariant_aux (r : renaming) (e : expr) :
  forward_kinded_in_code (rename_expr r e)
  = rename_expr r (forward_kinded_in_code e)
with forward_kinded_in_code_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  forward_kinded_in_code_handler (rename_handler r h)
  = rename_handler r (forward_kinded_in_code_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite forward_kinded_in_code_equivariant_aux. reflexivity.
    + rewrite forward_kinded_in_code_equivariant_aux, !map_map.
      f_equal. apply map_ext.
      exact (forward_kinded_in_binding_equivariant r).
    + rewrite forward_kinded_in_code_handler_equivariant,
        forward_kinded_in_code_equivariant_aux.
      reflexivity.
    + rewrite forward_kinded_in_code_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_kinded_in_code_handler_equivariant r h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite forward_kinded_in_code_equivariant_aux. reflexivity.
Qed.

Lemma forward_kinded_in_code_equivariant :
  rewrite_equivariant forward_kinded_in_code.
Proof. exact forward_kinded_in_code_equivariant_aux. Qed.

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

Lemma lookup_rename_arm_forwards (r : renaming) (F : arm_forwards)
    (k : continuation) :
  lookup (rename_continuation r k) (rename_arm_forwards r F)
  = option_map (rename_arm_entry r) (lookup k F).
Proof.
  unfold rename_arm_forwards.
  induction F as [ | [k' entry] F IH]; simpl; [ reflexivity | ].
  rewrite rename_continuation_eqb.
  destruct (Pos.eqb k k'); [ reflexivity | exact IH ].
Qed.

Lemma position_of_renamed (r : renaming) (x : variable)
    (params : list variable) :
  position_of (rename_variable r x) (map (rename_variable r) params)
  = position_of x params.
Proof.
  induction params as [ | y params IH]; simpl; [ reflexivity | ].
  rewrite rename_variable_eqb, IH. reflexivity.
Qed.

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

Fixpoint forward_to_arms_under_equivariant (r : renaming) (e : expr) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    forward_to_arms_under kinded (rename_kinds_in_scope r G)
      (rename_arm_forwards r F) (rename_expr r e)
    = rename_expr r (forward_to_arms_under kinded G F e)
with forward_to_arms_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ (kinded : bool) (G : kinds_in_scope) (F : arm_forwards),
    forward_to_arms_in_handler kinded (rename_kinds_in_scope r G)
      (rename_arm_forwards r F) (rename_handler r h)
    = rename_handler r (forward_to_arms_in_handler kinded G F h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k [params metadata handler] body
         | handlers body | ap | ac | scrutinee arms | ];
      intros kinded G F; try reflexivity.
    + cbn [rename_expr forward_to_arms_under].
      change ((rename_variable r x, kind) :: rename_kinds_in_scope r G)
        with (rename_kinds_in_scope r ((x, kind) :: G)).
      rewrite forward_to_arms_under_equivariant. reflexivity.
    + cbn [rename_expr forward_to_arms_under].
      rewrite forward_to_arms_under_equivariant. reflexivity.
    + cbn [rename_expr rename_handler forward_to_arms_under]. cbv zeta.
      rewrite params_in_scope_equivariant,
        (forward_to_arms_under_equivariant r handler),
        arm_entry_of_equivariant.
      change
        ((rename_continuation r k,
          rename_arm_entry r
            (arm_entry_of kinded params metadata
               (forward_to_arms_under kinded
                  (params_in_scope params (hm_params_kinds metadata) G) F
                  handler)
               body))
         :: rename_arm_forwards r F)
        with
          (rename_arm_forwards r
             ((k,
               arm_entry_of kinded params metadata
                 (forward_to_arms_under kinded
                    (params_in_scope params (hm_params_kinds metadata) G) F
                    handler)
                 body)
              :: F)).
      rewrite (forward_to_arms_under_equivariant r body). reflexivity.
    + cbn [rename_expr forward_to_arms_under]. cbv zeta.
      assert (Hscope :
        map (fun '(k, _) => (k, None))
          (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
             handlers)
        ++ rename_arm_forwards r F
        = rename_arm_forwards r
            (map (fun '(k, _) => (k, None)) handlers ++ F)).
      { unfold rename_arm_forwards. rewrite map_app, !map_map. f_equal.
        apply map_ext. intros [k h]. reflexivity. }
      rewrite Hscope. clear Hscope.
      rewrite (forward_to_arms_under_equivariant r body). f_equal.
      generalize (map (fun '(k, _) => (k, @None arm_entry)) handlers ++ F).
      intros F'.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_to_arms_in_handler_equivariant r h kinded G F'), IH.
      reflexivity.
    + cbn [rename_expr forward_to_arms_under].
      rewrite forward_to_arm_apply_cont_equivariant. reflexivity.
    + cbn [rename_expr forward_to_arms_under]. f_equal.
      rewrite !map_map. apply map_ext. intros [i ac]. cbn [fst snd].
      rewrite forward_to_arm_apply_cont_equivariant. reflexivity.
  - destruct h as [params metadata handler]. intros kinded G F.
    cbn [rename_handler forward_to_arms_in_handler].
    rewrite params_in_scope_equivariant, forward_to_arms_under_equivariant.
    reflexivity.
Qed.

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

Fixpoint forward_to_arms_in_code_equivariant_aux (r : renaming) (e : expr) :
  forward_to_arms_in_code (rename_expr r e)
  = rename_expr r (forward_to_arms_in_code e)
with forward_to_arms_in_code_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  forward_to_arms_in_code_handler (rename_handler r h)
  = rename_handler r (forward_to_arms_in_code_handler h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      simpl; try reflexivity.
    + rewrite forward_to_arms_in_code_equivariant_aux. reflexivity.
    + rewrite forward_to_arms_in_code_equivariant_aux, !map_map.
      f_equal. apply map_ext.
      exact (forward_to_arms_in_binding_equivariant r).
    + rewrite forward_to_arms_in_code_handler_equivariant,
        forward_to_arms_in_code_equivariant_aux.
      reflexivity.
    + rewrite forward_to_arms_in_code_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (forward_to_arms_in_code_handler_equivariant r h), IH.
      reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite forward_to_arms_in_code_equivariant_aux. reflexivity.
Qed.

Lemma forward_to_arms_in_code_equivariant :
  rewrite_equivariant forward_to_arms_in_code.
Proof. exact forward_to_arms_in_code_equivariant_aux. Qed.

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

(* Following aliases into the handlers of recursive continuations. *)
Fixpoint follow_aliases_in_loops_equivariant_aux (r : renaming) (e : expr) :
  ∀ tenv : typing_env,
    follow_aliases_in_loops (rename_naive_typing_env r tenv) (rename_expr r e)
    = rename_expr r (follow_aliases_in_loops tenv e)
with follow_aliases_in_loops_in_handler_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ tenv : typing_env,
    follow_aliases_in_loops_in_handler (rename_naive_typing_env r tenv)
      (rename_handler r h)
    = rename_handler r (follow_aliases_in_loops_in_handler tenv h)
with follow_aliases_in_loop_equivariant (r : renaming)
    (h : continuation_handler) :
  ∀ tenv : typing_env,
    follow_aliases_in_loop (rename_naive_typing_env r tenv)
      (rename_handler r h)
    = rename_handler r (follow_aliases_in_loop tenv h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros tenv; cbn [rename_expr follow_aliases_in_loops];
      try reflexivity.
    + rewrite typing_env_under_let_equivariant,
        follow_aliases_in_loops_equivariant_aux.
      reflexivity.
    + rewrite follow_aliases_in_loops_equivariant_aux. reflexivity.
    + rewrite follow_aliases_in_loops_in_handler_equivariant,
        follow_aliases_in_loops_equivariant_aux.
      reflexivity.
    + rewrite follow_aliases_in_loops_equivariant_aux. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite (follow_aliases_in_loop_equivariant r h tenv), IH. reflexivity.
  - destruct h as [params metadata handler]. intros tenv.
    cbn [rename_handler follow_aliases_in_loops_in_handler].
    rewrite drop_vars_equivariant, follow_aliases_in_loops_equivariant_aux.
    reflexivity.
  - destruct h as [params metadata handler]. intros tenv.
    cbn [rename_handler follow_aliases_in_loop].
    rewrite drop_vars_equivariant, follow_aliases_in_loops_equivariant_aux.
    unfold rewrite_code_body. cbv zeta.
    rewrite follow_aliases_equivariant_aux, !rename_binds_statics.
    destruct
      (Bool.eqb
         (binds_statics
            (follow_aliases_in_loops (drop_vars params tenv) handler))
         (binds_statics
            (follow_aliases (drop_vars params tenv)
               (follow_aliases_in_loops (drop_vars params tenv) handler))));
      reflexivity.
Qed.

Lemma follow_aliases_in_loops_equivariant (r : renaming) (tenv : typing_env)
    (e : expr) :
  follow_aliases_in_loops (rename_naive_typing_env r tenv) (rename_expr r e)
  = rename_expr r (follow_aliases_in_loops tenv e).
Proof. apply follow_aliases_in_loops_equivariant_aux. Qed.

Lemma follow_aliases_in_loops_unknowing_equivariant :
  rewrite_equivariant (follow_aliases_in_loops []).
Proof.
  intros r e. exact (follow_aliases_in_loops_equivariant_aux r e []).
Qed.

(* Loopify. The name given for the loop of each piece of code is renamed
   with the term; code ids are not renamed ([rename_loopified]). *)
Lemma is_self_tail_call_equivariant (r : renaming) (my_closure : variable)
    (params_arity : arity) (ret exn : continuation) (ap : apply_expr) :
  is_self_tail_call (rename_variable r my_closure) params_arity
    (rename_continuation r ret) (rename_continuation r exn)
    (rename_apply r ap)
  = is_self_tail_call my_closure params_arity ret exn ap.
Proof.
  unfold is_self_tail_call. rewrite no_exn_extra_args_renamed.
  cbn [rename_apply ap_callee ap_continuation ap_exn_continuation
       ap_args_arity].
  rewrite !rename_continuation_eqb.
  destruct (ap_callee ap) as [x | sym | c]; cbn [rename_simple];
    [ rewrite rename_variable_eqb | | ]; reflexivity.
Qed.

Fixpoint loop_calls_equivariant (r : renaming) (my_closure : variable)
    (params_arity : arity) (ret exn self : continuation) (e : expr) :
  loop_calls (rename_variable r my_closure) params_arity
    (rename_continuation r ret) (rename_continuation r exn)
    (rename_continuation r self) (rename_expr r e)
  = rename_expr r (loop_calls my_closure params_arity ret exn self e)
with loop_calls_in_handler_equivariant (r : renaming) (my_closure : variable)
    (params_arity : arity) (ret exn self : continuation)
    (h : continuation_handler) :
  loop_calls_in_handler (rename_variable r my_closure) params_arity
    (rename_continuation r ret) (rename_continuation r exn)
    (rename_continuation r self) (rename_handler r h)
  = rename_handler r
      (loop_calls_in_handler my_closure params_arity ret exn self h).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr loop_calls]; try reflexivity.
    + rewrite loop_calls_equivariant. reflexivity.
    + rewrite loop_calls_equivariant. reflexivity.
    + rewrite loop_calls_in_handler_equivariant, loop_calls_equivariant.
      reflexivity.
    + rewrite loop_calls_equivariant. f_equal.
      induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
      rewrite
        (loop_calls_in_handler_equivariant r my_closure params_arity ret exn
           self h),
        IH.
      reflexivity.
    + rewrite is_self_tail_call_equivariant.
      destruct (is_self_tail_call my_closure params_arity ret exn ap);
        reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite loop_calls_equivariant. reflexivity.
Qed.

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

Lemma loopify_code_equivariant (r : renaming) (self : continuation)
    (c : code) :
  loopify_code (rename_continuation r self) (rename_code r c)
  = rename_code r (loopify_code self c).
Proof.
  unfold loopify_code. rewrite loopable_equivariant, looped_code_equivariant.
  destruct (loopable self c); reflexivity.
Qed.

Lemma lookup_rename_loopified (r : renaming) (G : loopified) (id : code_id) :
  lookup id (rename_loopified r G)
  = option_map (rename_continuation r) (lookup id G).
Proof.
  unfold rename_loopified.
  induction G as [ | [id' self] G IH]; simpl; [ reflexivity | ].
  destruct (Pos.eqb id id'); [ reflexivity | exact IH ].
Qed.

Lemma loopify_binding_equivariant (r : renaming) (G : loopified)
    (binding : static_binding) :
  loopify_binding (rename_loopified r G) (rename_static_binding r binding)
  = rename_static_binding r (loopify_binding G binding).
Proof.
  destruct binding as [id c | id | sym const]; try reflexivity.
  cbn [rename_static_binding loopify_binding].
  rewrite lookup_rename_loopified.
  destruct (lookup id G) as [self | ]; cbn [option_map];
    [ | reflexivity ].
  cbn [rename_static_binding]. rewrite loopify_code_equivariant.
  reflexivity.
Qed.

Fixpoint loopify_unit_equivariant (r : renaming) (G : loopified) (e : expr) :
  loopify_unit (rename_loopified r G) (rename_expr r e)
  = rename_expr r (loopify_unit G e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler loopify_unit]; try reflexivity.
  - rewrite loopify_unit_equivariant. reflexivity.
  - rewrite loopify_unit_equivariant, !map_map. f_equal.
    apply map_ext. intros binding. apply loopify_binding_equivariant.
  - rewrite !loopify_unit_equivariant. reflexivity.
  - rewrite loopify_unit_equivariant. reflexivity.
Qed.

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

Lemma handler_kinds_renamed (r : renaming) (h : continuation_handler) :
  handler_kinds (rename_handler r h) = handler_kinds h.
Proof. destruct h; reflexivity. Qed.

Lemma kind_of_simple_renamed (r : renaming) (kinds : kind_env) (s : simple) :
  kind_of_simple (rename_kind_env r kinds) (rename_simple r s)
  = kind_of_simple kinds s.
Proof.
  destruct s as [x | sym | c]; simpl; try reflexivity.
  apply lookup_map_keys_injective, rename_variable_eqb.
Qed.

Lemma kinds_of_simples_renamed (r : renaming) (kinds : kind_env)
    (args : list simple) :
  kinds_of_simples (rename_kind_env r kinds) (map (rename_simple r) args)
  = kinds_of_simples kinds args.
Proof.
  induction args as [ | s args IH]; simpl; [ reflexivity | ].
  rewrite kind_of_simple_renamed, IH. reflexivity.
Qed.

Lemma bind_param_kinds_renamed (r : renaming) (params : list variable) :
  ∀ (params_kinds : list flambda_kind) (kinds : kind_env),
    bind_param_kinds (map (rename_variable r) params) params_kinds
      (rename_kind_env r kinds)
    = option_map (rename_kind_env r)
        (bind_param_kinds params params_kinds kinds).
Proof.
  induction params as [ | x params IH]; intros [ | k params_kinds] kinds;
    simpl; try reflexivity.
  exact (IH params_kinds ((x, k) :: kinds)).
Qed.

Lemma handler_arities_renamed (r : renaming)
    (handlers : list (continuation * continuation_handler)) :
  handler_arities (rename_handlers r handlers)
  = rename_arity_env r (handler_arities handlers).
Proof.
  unfold handler_arities, rename_handlers, rename_arity_env, Keyed.map_keys,
    Keyed.map_bindings, map_values.
  induction handlers as [ | [k h] handlers IH]; simpl; [ reflexivity | ].
  rewrite handler_kinds_renamed, IH. reflexivity.
Qed.

Lemma value_slots_well_kinded_renamed (r : renaming) (slots : slot_kinds)
    (kinds : kind_env) (value_slots : list (value_slot * simple)) :
  value_slots_well_kinded slots (rename_kind_env r kinds)
    (map (fun slot => (fst slot, rename_simple r (snd slot))) value_slots)
  = value_slots_well_kinded slots kinds value_slots.
Proof.
  unfold value_slots_well_kinded. apply forallb_map_with. intros [slot s].
  simpl. rewrite kind_of_simple_renamed. reflexivity.
Qed.

Lemma kind_of_named_renamed (r : renaming) (slots : slot_kinds)
    (kinds : kind_env) (d : named) :
  kind_of_named slots (rename_kind_env r kinds) (rename_named r d)
  = kind_of_named slots kinds d.
Proof.
  destruct d; cbn [rename_named kind_of_named];
    rewrite ?kind_of_simple_renamed, ?kinds_of_simples_renamed,
      ?value_slots_well_kinded_renamed;
    reflexivity.
Qed.

Lemma static_const_well_kinded_renamed (r : renaming) (slots : slot_kinds)
    (kinds : kind_env) (const : static_const) :
  static_const_well_kinded slots (rename_kind_env r kinds)
    (rename_static_const r const)
  = static_const_well_kinded slots kinds const.
Proof.
  destruct const; cbn [rename_static_const static_const_well_kinded];
    rewrite ?kind_of_simple_renamed, ?value_slots_well_kinded_renamed;
    try reflexivity.
  all: apply forallb_map_with; intros s; rewrite kind_of_simple_renamed;
    reflexivity.
Qed.

Lemma is_bound_map_keys_injective {A : Type} (f : positive -> positive)
    (x : positive) (l : finite_map A) :
  (∀ a b : positive, Pos.eqb (f a) (f b) = Pos.eqb a b) ->
  is_bound (f x) (Keyed.map_keys f l) = is_bound x l.
Proof.
  intros Hf. unfold is_bound. rewrite (lookup_map_keys_injective _ _ _ Hf).
  reflexivity.
Qed.

Lemma trap_action_in_scope_renamed (r : renaming) (conts : arity_env)
    (ac : apply_cont) :
  trap_action_in_scope (rename_arity_env r conts) (rename_apply_cont r ac)
  = trap_action_in_scope conts ac.
Proof.
  unfold trap_action_in_scope, rename_arity_env. simpl.
  destruct (ac_trap_action ac) as [[k | k kind] | ]; simpl;
    [ | | reflexivity ];
    apply is_bound_map_keys_injective, rename_continuation_eqb.
Qed.

Lemma exn_extra_kinds_renamed (r : renaming) (ap : apply_expr) :
  map snd (ap_exn_extra_args (rename_apply r ap))
  = map snd (ap_exn_extra_args ap).
Proof. simpl. rewrite map_map. reflexivity. Qed.

Lemma apply_cont_well_kinded_renamed (r : renaming) (kinds : kind_env)
    (conts : arity_env) (ac : apply_cont) :
  apply_cont_well_kinded (rename_kind_env r kinds) (rename_arity_env r conts)
    (rename_apply_cont r ac)
  = apply_cont_well_kinded kinds conts ac.
Proof.
  unfold apply_cont_well_kinded. rewrite trap_action_in_scope_renamed.
  unfold rename_arity_env. simpl.
  rewrite (lookup_map_keys_injective _ _ _ (rename_continuation_eqb r)),
    kinds_of_simples_renamed.
  reflexivity.
Qed.

Lemma apply_well_kinded_renamed (r : renaming) (kinds : kind_env)
    (conts : arity_env) (ap : apply_expr) :
  apply_well_kinded (rename_kind_env r kinds) (rename_arity_env r conts)
    (rename_apply r ap)
  = apply_well_kinded kinds conts ap.
Proof.
  unfold apply_well_kinded.
  rewrite exn_extra_simples_rename, !exn_extra_kinds_renamed.
  unfold rename_arity_env. simpl.
  rewrite !(lookup_map_keys_injective _ _ _ (rename_continuation_eqb r)),
    kind_of_simple_renamed, !kinds_of_simples_renamed.
  reflexivity.
Qed.

Fixpoint well_kinded_renamed (r : renaming) (slots : slot_kinds) (e : expr)
    {struct e} :
  ∀ (kinds : kind_env) (conts : arity_env),
    well_kinded slots (rename_kind_env r kinds) (rename_arity_env r conts)
      (rename_expr r e)
    = well_kinded slots kinds conts e
with handler_well_kinded_renamed (r : renaming) (slots : slot_kinds)
    (h : continuation_handler) {struct h} :
  ∀ (kinds : kind_env) (conts : arity_env),
    handler_well_kinded slots (rename_kind_env r kinds)
      (rename_arity_env r conts) (rename_handler r h)
    = handler_well_kinded slots kinds conts h
with binding_well_kinded_renamed (r : renaming) (slots : slot_kinds)
    (binding : static_binding) {struct binding} :
  ∀ kinds : kind_env,
    binding_well_kinded slots (rename_kind_env r kinds)
      (rename_static_binding r binding)
    = binding_well_kinded slots kinds binding
with code_well_kinded_renamed (r : renaming) (slots : slot_kinds) (c : code)
    {struct c} :
  code_well_kinded slots (rename_code r c) = code_well_kinded slots c.
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros kinds conts; cbn [rename_expr well_kinded].
    + rewrite kind_of_named_renamed. f_equal.
      exact (well_kinded_renamed r slots body ((x, kind) :: kinds) conts).
    + rewrite well_kinded_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (binding_well_kinded_renamed r slots binding), IH. reflexivity.
    + rewrite handler_well_kinded_renamed, handler_kinds_renamed. f_equal.
      exact
        (well_kinded_renamed r slots body kinds
           ((k, handler_kinds h) :: conts)).
    + change
        (map (fun '(k, h) => (rename_continuation r k, rename_handler r h))
           handlers)
        with (rename_handlers r handlers).
      rewrite handler_arities_renamed.
      replace
        (rename_arity_env r (handler_arities handlers)
         ++ rename_arity_env r conts)
        with (rename_arity_env r (handler_arities handlers ++ conts))
        by apply map_app.
      rewrite well_kinded_renamed. f_equal.
      generalize (handler_arities handlers ++ conts). intros conts'.
      unfold rename_handlers.
      induction handlers as [ | [k h] handlers IH]; cbn [map forallb];
        [ reflexivity | ].
      rewrite (handler_well_kinded_renamed r slots h), IH. reflexivity.
    + apply apply_well_kinded_renamed.
    + apply apply_cont_well_kinded_renamed.
    + rewrite kind_of_simple_renamed. f_equal. apply forallb_map_with.
      intros arm. apply apply_cont_well_kinded_renamed.
    + reflexivity.
  - destruct h as [params metadata handler]. intros kinds conts.
    cbn [rename_handler handler_well_kinded].
    rewrite (distinct_map_injective _ (rename_variable_eqb r)),
      bind_param_kinds_renamed.
    destruct (bind_param_kinds params (hm_params_kinds metadata) kinds)
      as [handler_kinds | ];
      cbn [option_map]; [ | reflexivity ].
    rewrite well_kinded_renamed. reflexivity.
  - destruct binding as [id c | id | sym const]; intros kinds;
      cbn [rename_static_binding binding_well_kinded].
    + apply code_well_kinded_renamed.
    + reflexivity.
    + apply static_const_well_kinded_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_well_kinded].
    change (rename_variable r my_closure :: map (rename_variable r) params)
      with (map (rename_variable r) (my_closure :: params)).
    rewrite (distinct_map_injective _ (rename_variable_eqb r)).
    change [(rename_variable r my_closure, Value)]
      with (rename_kind_env r [(my_closure, Value)]).
    rewrite bind_param_kinds_renamed.
    destruct
      (bind_param_kinds params (unarize (cm_params_arity metadata))
         [(my_closure, Value)])
      as [body_kinds | ];
      cbn [option_map]; [ | reflexivity ].
    f_equal.
    exact
      (well_kinded_renamed r slots body body_kinds
         [(ret, unarize (cm_result_arity metadata)); (exn, [Value])]).
Qed.

Theorem handler_kinds_equivariant : handler_kinds_equivariant_statement.
Proof. exact handler_kinds_renamed. Qed.

Theorem kind_of_simple_equivariant : kind_of_simple_equivariant_statement.
Proof. exact kind_of_simple_renamed. Qed.

Theorem kinds_of_simples_equivariant : kinds_of_simples_equivariant_statement.
Proof. exact kinds_of_simples_renamed. Qed.

Theorem kind_of_named_equivariant : kind_of_named_equivariant_statement.
Proof. exact kind_of_named_renamed. Qed.

Theorem bind_param_kinds_equivariant : bind_param_kinds_equivariant_statement.
Proof. exact bind_param_kinds_renamed. Qed.

Theorem static_const_well_kinded_equivariant :
  static_const_well_kinded_equivariant_statement.
Proof. exact static_const_well_kinded_renamed. Qed.

Theorem apply_cont_well_kinded_equivariant :
  apply_cont_well_kinded_equivariant_statement.
Proof. exact apply_cont_well_kinded_renamed. Qed.

Theorem apply_well_kinded_equivariant :
  apply_well_kinded_equivariant_statement.
Proof. exact apply_well_kinded_renamed. Qed.

Theorem well_kinded_equivariant : well_kinded_equivariant_statement.
Proof.
  intros r slots kinds conts e. apply well_kinded_renamed.
Qed.

Theorem well_kinded_unit_equivariant : well_kinded_unit_equivariant_statement.
Proof.
  intros r slots unit_return unit_exn e. unfold well_kinded_unit.
  exact
    (well_kinded_renamed r slots e [] (unit_arities unit_return unit_exn)).
Qed.

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
Proof. destruct s; reflexivity. Qed.

Lemma simples_statics_in_scope_renamed (r : renaming) (syms : list symbol)
    (args : list simple) :
  simples_statics_in_scope syms (map (rename_simple r) args)
  = simples_statics_in_scope syms args.
Proof.
  unfold simples_statics_in_scope. apply forallb_map_with. intros s.
  apply simple_statics_in_scope_renamed.
Qed.

Lemma value_slots_statics_in_scope_renamed (r : renaming)
    (syms : list symbol) (slots : list (value_slot * simple)) :
  value_slots_statics_in_scope syms
    (map (fun slot => (fst slot, rename_simple r (snd slot))) slots)
  = value_slots_statics_in_scope syms slots.
Proof.
  unfold value_slots_statics_in_scope. apply forallb_map_with. intros slot.
  cbn [snd]. apply simple_statics_in_scope_renamed.
Qed.

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

Lemma apply_cont_statics_in_scope_renamed (r : renaming)
    (syms : list symbol) (ac : apply_cont) :
  apply_cont_statics_in_scope syms (rename_apply_cont r ac)
  = apply_cont_statics_in_scope syms ac.
Proof.
  unfold apply_cont_statics_in_scope, rename_apply_cont. cbn [ac_args].
  apply simples_statics_in_scope_renamed.
Qed.

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
Proof. destruct const; reflexivity. Qed.

Lemma closure_symbols_of_binding_renamed (r : renaming)
    (binding : static_binding) :
  closure_symbols_of_binding (rename_static_binding r binding)
  = closure_symbols_of_binding binding.
Proof.
  destruct binding as [id c | id | s const]; try reflexivity.
  cbn [rename_static_binding closure_symbols_of_binding].
  rewrite is_closure_const_renamed. reflexivity.
Qed.

Lemma declared_code_ids_of_binding_renamed (r : renaming)
    (binding : static_binding) :
  declared_code_ids_of_binding (rename_static_binding r binding)
  = declared_code_ids_of_binding binding.
Proof. destruct binding; reflexivity. Qed.

Lemma group_consts_in_scope_renamed (r : renaming) (all : list symbol)
    (ids : list code_id) (bindings : list static_binding) :
  ∀ before : list symbol,
    group_consts_in_scope before all ids
      (map (rename_static_binding r) bindings)
    = group_consts_in_scope before all ids bindings.
Proof.
  induction bindings as [ | [id c | id | s const] bindings IH];
    intros before; cbn [map rename_static_binding group_consts_in_scope];
    [ reflexivity | apply IH | apply IH | ].
  rewrite is_closure_const_renamed, !static_const_statics_in_scope_renamed,
    !IH.
  reflexivity.
Qed.

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
Fixpoint bound_continuations_renamed (r : renaming) (e : expr) {struct e} :
  bound_continuations (rename_expr r e)
  = map (rename_continuation r) (bound_continuations e)
with handler_bound_continuations_renamed (r : renaming)
    (h : continuation_handler) {struct h} :
  handler_bound_continuations (rename_handler r h)
  = map (rename_continuation r) (handler_bound_continuations h)
with binding_bound_continuations_renamed (r : renaming)
    (binding : static_binding) {struct binding} :
  binding_bound_continuations (rename_static_binding r binding)
  = map (rename_continuation r) (binding_bound_continuations binding)
with code_bound_continuations_renamed (r : renaming) (c : code) {struct c} :
  code_bound_continuations (rename_code r c)
  = map (rename_continuation r) (code_bound_continuations c).
Proof.
  - destruct e
      as [ x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rename_expr bound_continuations]; try reflexivity.
    + apply bound_continuations_renamed.
    + rewrite map_app, bound_continuations_renamed. f_equal.
      induction bindings as [ | binding bindings IH]; cbn [map flat_map];
        [ reflexivity | ].
      rewrite map_app, (binding_bound_continuations_renamed r binding), IH.
      reflexivity.
    + cbn [map].
      rewrite map_app, handler_bound_continuations_renamed,
        bound_continuations_renamed.
      reflexivity.
    + rewrite !map_app, bound_continuations_renamed. f_equal; [ | f_equal ].
      * rewrite !map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map flat_map];
          [ reflexivity | ].
        rewrite map_app, (handler_bound_continuations_renamed r h), IH.
        reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_bound_continuations].
    apply bound_continuations_renamed.
  - destruct binding as [id c | id | s const];
      cbn [rename_static_binding binding_bound_continuations];
      try reflexivity.
    apply code_bound_continuations_renamed.
  - destruct c as [params my_closure ret exn metadata body].
    cbn [rename_code code_bound_continuations map].
    rewrite bound_continuations_renamed. reflexivity.
Qed.

Lemma interface_conts_not_rebound_renamed (r : renaming)
    (u : unit_interface) (e : expr) :
  interface_conts_not_rebound (rename_unit_interface r u) (rename_expr r e)
  = interface_conts_not_rebound u e.
Proof.
  unfold interface_conts_not_rebound, rename_unit_interface.
  cbn [unit_return unit_exn].
  rewrite bound_continuations_renamed, !among_rename_continuation.
  reflexivity.
Qed.

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
