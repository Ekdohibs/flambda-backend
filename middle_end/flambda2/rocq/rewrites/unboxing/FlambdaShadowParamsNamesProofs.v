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


(* FlambdaShadowParamsSpec.shadow_params commutes with renaming and
   keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and
   FlambdaStaticsScopeProofs.v prove the two statements. There is no
   statement here, and nothing here needs review.

   The pass compares names by equality only and makes none. So each of
   its functions gives, on a renamed term, the renamed result: the
   rewrite for renamed positions, and the positions that the
   validation uses, which are computed from the term. For the scope:
   a [Let] that the pass puts mentions a parameter of the handler
   whose head it is at, and binds that parameter. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceBaseProofs FlambdaScopedProofs
  FlambdaUntagParamsSpec FlambdaUntagParamsProofs
  FlambdaUntagParamsNamesProofs FlambdaEquivarianceTactic
  FlambdaShadowParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

(* The laws of renaming are proved from the definitions by the tactic
   of FlambdaEquivarianceTactic.v ([equivariance]), which states and
   proves by itself the laws of the functions that the pass is made
   of, those of the rewrite ([shadow_lets], [shadow_expr]) among
   them. It is told how the positions are renamed (a map keyed by
   continuations is a list of pairs: the action is [rename_keyed]),
   and it is given two laws of functions of lists that other files
   prove. *)
#[local] Instance positions_renaming : Renaming_of positions := rename_keyed.
#[local] Hint Rewrite @moved_from_map distinct_renamed : equivariance.

(* ======================================================================== *)
(* The positions that the validation uses commute with renaming             *)
(* ======================================================================== *)

(* Five laws are stated. The tactic proves four of them. The law of
   [gives_needed] is given back to it and the function is kept folded,
   since computation would unfold it in an arm of a [Switch] before
   the law applies. The laws of [handlers_of], [needed_round] and
   [needed_positions] are those from which the law of the positions
   of the pass is proved, by hand: the pass starts its rounds at no
   position, which is not something renamed as it is written. That
   law is given to the tactic for the law of the pass, with the
   positions kept folded, so that the six rounds are not computed. *)
Lemma gives_needed_renamed (r : renaming) (needed : positions) (p : variable)
    (ac : apply_cont) :
  gives_needed (rename_keyed r needed) (rename_variable r p)
    (rename_apply_cont r ac)
  = gives_needed needed p ac.
Proof. equivariance. Qed.
#[local] Hint Rewrite gives_needed_renamed : equivariance.
#[local] Arguments gives_needed : simpl never.

Lemma handlers_of_renamed (r : renaming) (e : expr) :
  handlers_of (rename_expr r e) = rename_handlers r (handlers_of e).
Proof. equivariance. Qed.

Lemma needed_round_renamed (r : renaming)
    (hs : list (continuation * continuation_handler)) (needed : positions) :
  needed_round (rename_handlers r hs) (rename_keyed r needed)
  = rename_keyed r (needed_round hs needed).
Proof. equivariance. Qed.
#[local] Hint Rewrite needed_round_renamed : equivariance.

Lemma needed_positions_renamed (r : renaming)
    (hs : list (continuation * continuation_handler)) (rounds : nat) :
  ∀ needed : positions,
    needed_positions rounds (rename_handlers r hs) (rename_keyed r needed)
    = rename_keyed r (needed_positions rounds hs needed).
Proof. equivariance. Qed.

Lemma shadow_positions_renamed (r : renaming) (e : expr) :
  shadow_positions (rename_expr r e) = rename_keyed r (shadow_positions e).
Proof.
  unfold shadow_positions. rewrite handlers_of_renamed.
  exact (needed_positions_renamed r (handlers_of e) 6 []).
Qed.
#[local] Hint Rewrite shadow_positions_renamed : equivariance.
#[local] Arguments shadow_positions : simpl never.

Lemma shadow_params_renamed : rewrite_equivariant shadow_params.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma shadow_lets_code_closed (us : list nat) (e : expr) :
  ∀ (params : list variable) (kinds : list flambda_kind) (i : nat),
    code_closed (shadow_lets us i params kinds e) = code_closed e.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i;
    cbn [shadow_lets]; try reflexivity.
  destruct (at_position us i); cbn [code_closed]; apply IH.
Qed.

Lemma shadow_lets_free_continuations (us : list nat) (e : expr) :
  ∀ (params : list variable) (kinds : list flambda_kind) (i : nat),
    free_continuations_of_expr (shadow_lets us i params kinds e)
    = free_continuations_of_expr e.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i;
    cbn [shadow_lets]; try reflexivity.
  destruct (at_position us i); cbn [free_continuations_of_expr]; apply IH.
Qed.

(* A free variable of an expression under the [Let]s is one of the
   expression, or one of the parameters. *)
Lemma shadow_lets_free_variables (us : list nat) (a : variable) (e : expr) :
  ∀ (params : list variable) (kinds : list flambda_kind) (i : nat),
    In a (free_variables_of_expr (shadow_lets us i params kinds e)) ->
    In a (free_variables_of_expr e) ∨ In a params.
Proof.
  induction params as [ | p params IH]; intros [ | kind kinds] i;
    cbn [shadow_lets]; try (intros Hin; left; exact Hin).
  assert (Hrest :
    In a
      (free_variables_of_expr (shadow_lets us (S i) params kinds e)) ->
    In a (free_variables_of_expr e) ∨ In a (p :: params)).
  { intros Hin. destruct (IH kinds (S i) Hin) as [Hin' | Hin'];
      [ left; exact Hin' | right; right; exact Hin' ]. }
  destruct (at_position us i); [ | exact Hrest ].
  cbn [free_variables_of_expr]. intros Hin.
  apply in_app_or in Hin as [Hin | Hin].
  - right. left. cbn in Hin. destruct Hin as [<- | []]. reflexivity.
  - apply in_without in Hin as [Hin _]. exact (Hrest Hin).
Qed.

Fixpoint shadow_expr_code_closed (sites : positions) (e : expr) {struct e} :
  code_closed (shadow_expr sites e) = code_closed e
with shadow_handler_code_closed (sites : positions) (us : list nat)
    (h : continuation_handler) {struct h} :
  handler_code_closed (shadow_handler sites us h) = handler_code_closed h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [shadow_expr code_closed]; try reflexivity.
    + apply shadow_expr_code_closed.
    + rewrite shadow_expr_code_closed. reflexivity.
    + rewrite shadow_handler_code_closed, shadow_expr_code_closed.
      reflexivity.
    + rewrite shadow_expr_code_closed. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite shadow_handler_code_closed, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [shadow_handler handler_code_closed].
    destruct (distinct params); [ rewrite shadow_lets_code_closed | ];
      apply shadow_expr_code_closed.
Qed.

Fixpoint shadow_expr_free_continuations (sites : positions) (e : expr)
    {struct e} :
  free_continuations_of_expr (shadow_expr sites e)
  = free_continuations_of_expr e
with shadow_handler_free_continuations (sites : positions) (us : list nat)
    (h : continuation_handler) {struct h} :
  free_continuations_of_handler (shadow_handler sites us h)
  = free_continuations_of_handler h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [shadow_expr free_continuations_of_expr]; try reflexivity.
    + apply shadow_expr_free_continuations.
    + rewrite shadow_expr_free_continuations. reflexivity.
    + rewrite shadow_handler_free_continuations,
        shadow_expr_free_continuations.
      reflexivity.
    + rewrite shadow_expr_free_continuations. f_equal.
      * induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map fst]. rewrite IH. reflexivity.
      * f_equal.
        induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
        cbn [map flat_map].
        rewrite shadow_handler_free_continuations, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [shadow_handler free_continuations_of_handler].
    destruct (distinct params);
      [ rewrite shadow_lets_free_continuations | ];
      apply shadow_expr_free_continuations.
Qed.

Fixpoint shadow_expr_free_variables (sites : positions) (a : variable)
    (e : expr) {struct e} :
  In a (free_variables_of_expr (shadow_expr sites e)) ->
  In a (free_variables_of_expr e)
with shadow_handler_free_variables (sites : positions) (a : variable)
    (us : list nat) (h : continuation_handler) {struct h} :
  In a (free_variables_of_handler (shadow_handler sites us h)) ->
  In a (free_variables_of_handler h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [shadow_expr free_variables_of_expr];
      try (intros Hin; exact Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        apply in_or_app; [ left; exact Hin | right ].
      apply in_without in Hin as [Hin Hy]. apply in_without.
      split; [ exact (shadow_expr_free_variables sites a body Hin) | exact Hy ].
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        apply in_or_app; [ left; exact Hin | right ].
      exact (shadow_expr_free_variables sites a body Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        apply in_or_app; [ left | right ].
      * exact (shadow_handler_free_variables sites a _ hh Hin).
      * exact (shadow_expr_free_variables sites a body Hin).
    + intros Hin. apply in_app_or in Hin as [Hin | Hin];
        apply in_or_app; [ left | right ].
      * induction handlers as [ | [k' hh] handlers IH];
          cbn [map flat_map] in Hin |- *; [ contradiction | ].
        apply in_app_or in Hin as [Hin | Hin]; apply in_or_app;
          [ left | right; exact (IH Hin) ].
        exact (shadow_handler_free_variables sites a _ hh Hin).
      * exact (shadow_expr_free_variables sites a body Hin).
  - destruct h as [params metadata handler].
    cbn [shadow_handler free_variables_of_handler]. intros Hin.
    apply in_without in Hin as [Hin Hparams]. apply in_without.
    split; [ | exact Hparams ].
    destruct (distinct params);
      [ | exact (shadow_expr_free_variables sites a handler Hin) ].
    destruct
      (shadow_lets_free_variables us a _ params (hm_params_kinds metadata) 0
         Hin)
      as [Hin' | Hin'];
      [ exact (shadow_expr_free_variables sites a handler Hin')
      | contradiction ].
Qed.

Lemma shadow_expr_scoped (sites : positions) (e : expr) :
  scoped e (shadow_expr sites e).
Proof.
  intros Hclosed. rewrite shadow_expr_code_closed.
  split; [ exact Hclosed | split ].
  - intros a Ha. exact (shadow_expr_free_variables sites a e Ha).
  - rewrite shadow_expr_free_continuations. apply incl_refl.
Qed.

Lemma shadow_params_scoped (e : expr) : scoped e (shadow_params e).
Proof.
  unfold shadow_params.
  destruct (count_positions (shadow_positions e)); [ apply scoped_refl | ].
  apply shadow_expr_scoped.
Qed.

Lemma shadow_params_scope : keeps_scope shadow_params.
Proof. apply keeps_scope_intro, shadow_params_scoped. Qed.
