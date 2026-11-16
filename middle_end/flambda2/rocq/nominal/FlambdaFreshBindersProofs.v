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

(* Proofs of the statements of FlambdaFreshBindersSpec.v. Nothing here
   needs review.

   The free names of an expression whose code is closed are a list
   (FlambdaFreeNamesSpec.v), so a name outside the list is not free. The
   statements that open a binder are those of FlambdaNominalSpec.v at a
   swap. A rule that compares two binders at one name is two uses of the
   rule of [alpha_equivalent] for that binder, at the two swaps, with the
   rule for a context between them. *)

From Stdlib Require Import ZArith List Bool Lia.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaContextSpec
  FlambdaBindersSpec FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaCompareSpec FlambdaNameAbstractionSpec
  FlambdaNameAbstractionProofs FlambdaFreshQuantifierSpec
  FlambdaFreshQuantifierProofs FlambdaFreshBindersSpec.
Import ListNotations.

(* ======================================================================== *)
(* The swaps                                                                *)
(* ======================================================================== *)

Lemma rename_variable_swap (x fresh : variable) :
  rename_variable (swap_variable x fresh) x = fresh.
Proof. unfold rename_variable. simpl. apply swap_first. Qed.

Lemma rename_variable_swap_other (x fresh y : variable) :
  y <> x -> y <> fresh -> rename_variable (swap_variable x fresh) y = y.
Proof.
  intros Hx Hfresh. unfold rename_variable. simpl.
  apply swap_other; assumption.
Qed.

Lemma rename_continuation_swap (k fresh : continuation) :
  rename_continuation (swap_continuation k fresh) k = fresh.
Proof. unfold rename_continuation. simpl. apply swap_first. Qed.

Lemma rename_continuation_swap_other (k fresh k' : continuation) :
  k' <> k -> k' <> fresh ->
  rename_continuation (swap_continuation k fresh) k' = k'.
Proof.
  intros Hk Hfresh. unfold rename_continuation. simpl.
  apply swap_other; assumption.
Qed.

Lemma rename_variables_swap (xs fresh : list variable) :
  length fresh = length xs ->
  distinct xs = true ->
  distinct_outside xs fresh ->
  map (rename_variable (swap_variables xs fresh)) xs = fresh.
Proof.
  intros Hlength Hdistinct Houtside.
  apply (permute_combine xs fresh);
    [ symmetry; exact Hlength | exact Hdistinct | exact Houtside ].
Qed.

Lemma rename_variables_swap_other (xs fresh : list variable) (y : variable) :
  among y xs = false -> among y fresh = false ->
  rename_variable (swap_variables xs fresh) y = y.
Proof. exact (permute_combine_other xs fresh y). Qed.

(* ======================================================================== *)
(* A fresh name is not free                                                 *)
(* ======================================================================== *)

Lemma not_free_variable (x : variable) (e : expr) :
  code_closed e = true ->
  among x (free_variables_of_expr e) = false ->
  ~ var_free_in_expr x e.
Proof.
  intros Hclosed Hx Hfree. apply (var_free_agrees x e Hclosed) in Hfree.
  congruence.
Qed.

Lemma not_free_continuation (k : continuation) (e : expr) :
  code_closed e = true ->
  among k (free_continuations_of_expr e) = false ->
  ~ cont_free_in_expr k e.
Proof.
  intros Hclosed Hk Hfree. apply (cont_free_agrees k e Hclosed) in Hfree.
  congruence.
Qed.

(* A fresh name is not free by the relations. This was a statement of
   FlambdaFreshBindersSpec.v; of the lists of free names it is
   FlambdaFreshQuantifierSpec.for_fresh_avoids_statement. It keeps its
   name and its text because the proofs below use it. *)
Definition fresh_not_free_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    (И x, ~ var_free_in_expr x e) ∧ (И k, ~ cont_free_in_expr k e).

Theorem fresh_not_free : fresh_not_free_statement.
Proof.
  intros e Hclosed. split.
  - exists (free_variables_of_expr e). intros x.
    apply not_free_variable, Hclosed.
  - exists (free_continuations_of_expr e). intros k.
    apply not_free_continuation, Hclosed.
Qed.

(* ======================================================================== *)
(* Opening a binder at a fresh name                                         *)
(* ======================================================================== *)

Theorem freshen_let_fresh : freshen_let_fresh_statement.
Proof.
  intros x kind d body Hclosed.
  apply (for_fresh_weaken (fun fresh => ~ var_free_in_expr fresh body)).
  - intros fresh Hfresh.
    exact (freshen_let x fresh kind d body Hclosed Hfresh).
  - apply (fresh_not_free body Hclosed).
Qed.

Theorem freshen_let_cont_fresh : freshen_let_cont_fresh_statement.
Proof.
  intros k handler body Hclosed.
  exists (free_continuations_of_expr body). intros fresh Hfresh.
  pose proof
    (rename_let_cont (swap_continuation k fresh) k handler body Hclosed)
    as Hrename.
  rewrite rename_continuation_swap in Hrename. apply Hrename.
  - intros y _. reflexivity.
  - intros k' Hdifferent Hfree.
    apply rename_continuation_swap_other; [ exact Hdifferent | ].
    intros ->. exact (not_free_continuation fresh body Hclosed Hfresh Hfree).
Qed.

(* The swaps of the parameters with fresh names fix the other free
   variables of the handler, and every continuation. *)
Lemma swap_variables_fixes (params fresh : list variable) (handler : expr) :
  code_closed handler = true ->
  distinct_outside (params ++ free_variables_of_expr handler) fresh ->
  (∀ y : variable,
    y ∉ params -> var_free_in_expr y handler ->
    rename_variable (swap_variables params fresh) y = y)
  ∧ (∀ k : continuation,
      cont_free_in_expr k handler ->
      rename_continuation (swap_variables params fresh) k = k).
Proof.
  intros Hclosed Houtside.
  apply distinct_outside_app in Houtside as [_ [_ Hfree]]. split.
  - intros y Hnot_param Hy. apply rename_variables_swap_other.
    + destruct (among y params) eqn:Hamong; [ | reflexivity ].
      apply among_In in Hamong. contradiction.
    + destruct (among y fresh) eqn:Hamong; [ | reflexivity ].
      apply Hfree in Hamong.
      apply (var_free_agrees y handler Hclosed) in Hy. congruence.
  - intros k _. reflexivity.
Qed.

Theorem freshen_handler_params_fresh : freshen_handler_params_fresh_statement.
Proof.
  intros k params metadata handler body Hclosed Hdistinct.
  exists (params ++ free_variables_of_expr handler).
  intros fresh Hlength Houtside.
  destruct (swap_variables_fixes params fresh handler Hclosed Houtside)
    as [Hvariables Hcontinuations].
  apply distinct_outside_app in Houtside as [Hparams _].
  pose proof
    (rename_handler_params (swap_variables params fresh) k params metadata
       handler body Hclosed Hvariables Hcontinuations)
    as Hrename.
  cbn [rename_handler] in Hrename.
  rewrite (rename_variables_swap params fresh Hlength Hdistinct Hparams)
    in Hrename.
  exact Hrename.
Qed.

(* ======================================================================== *)
(* Two binders compared at one fresh name                                   *)
(* ======================================================================== *)

Lemma alpha_let_swap (x fresh : variable) (kind : flambda_kind) (d : named)
    (body : expr) :
  code_closed body = true ->
  among fresh (free_variables_of_expr body) = false ->
  alpha_equivalent (Let x kind d body)
    (Let fresh kind d (rename_expr (swap_variable x fresh) body)).
Proof.
  intros Hclosed Hfresh.
  pose proof
    (Alpha_let_by_relation (swap_variable x fresh) x kind d body Hclosed)
    as Hrule.
  rewrite rename_variable_swap in Hrule. apply Hrule.
  - intros y Hdifferent Hy.
    apply rename_variable_swap_other; [ exact Hdifferent | ].
    intros ->. exact (not_free_variable fresh body Hclosed Hfresh Hy).
  - intros k _. reflexivity.
Qed.

Theorem alpha_let_at : alpha_let_at_statement.
Proof.
  intros x x' fresh kind d body body' Hclosed Hclosed' Hfresh Hfresh'
    Hbodies.
  apply Alpha_trans
    with (Let fresh kind d (rename_expr (swap_variable x fresh) body));
    [ apply alpha_let_swap; assumption | ].
  apply Alpha_trans
    with (Let fresh kind d (rename_expr (swap_variable x' fresh) body'));
    [ | apply Alpha_sym, alpha_let_swap; assumption ].
  exact (Alpha_context (Let_body fresh kind d Hole) _ _ Hbodies).
Qed.

Theorem alpha_let_fresh : alpha_let_fresh_statement.
Proof.
  intros x x' kind d body body' Hclosed Hclosed' Hfresh.
  destruct
    (for_fresh_elim _
       (free_variables_of_expr body ++ free_variables_of_expr body') Hfresh)
    as (fresh & Hname & Hbodies).
  apply among_app_false in Hname as [Hname Hname'].
  exact (alpha_let_at x x' fresh kind d body body' Hclosed Hclosed' Hname
           Hname' Hbodies).
Qed.

Lemma alpha_let_cont_swap (k fresh : continuation)
    (handler : continuation_handler) (body : expr) :
  code_closed body = true ->
  among fresh (free_continuations_of_expr body) = false ->
  alpha_equivalent (Let_cont_non_recursive k handler body)
    (Let_cont_non_recursive fresh handler
       (rename_expr (swap_continuation k fresh) body)).
Proof.
  intros Hclosed Hfresh.
  pose proof
    (Alpha_let_cont_by_relation (swap_continuation k fresh) k handler body
       Hclosed)
    as Hrule.
  rewrite rename_continuation_swap in Hrule. apply Hrule.
  - intros y _. reflexivity.
  - intros k' Hdifferent Hk'.
    apply rename_continuation_swap_other; [ exact Hdifferent | ].
    intros ->. exact (not_free_continuation fresh body Hclosed Hfresh Hk').
Qed.

Theorem alpha_let_cont_at : alpha_let_cont_at_statement.
Proof.
  intros k k' fresh handler body body' Hclosed Hclosed' Hfresh Hfresh'
    Hbodies.
  apply Alpha_trans
    with (Let_cont_non_recursive fresh handler
            (rename_expr (swap_continuation k fresh) body));
    [ apply alpha_let_cont_swap; assumption | ].
  apply Alpha_trans
    with (Let_cont_non_recursive fresh handler
            (rename_expr (swap_continuation k' fresh) body'));
    [ | apply Alpha_sym, alpha_let_cont_swap; assumption ].
  exact
    (Alpha_context (Let_cont_non_recursive_body fresh handler Hole) _ _
       Hbodies).
Qed.

Theorem alpha_let_cont_fresh : alpha_let_cont_fresh_statement.
Proof.
  intros k k' handler body body' Hclosed Hclosed' Hfresh.
  destruct
    (for_fresh_elim _
       (free_continuations_of_expr body ++ free_continuations_of_expr body')
       Hfresh)
    as (fresh & Hname & Hbodies).
  apply among_app_false in Hname as [Hname Hname'].
  exact (alpha_let_cont_at k k' fresh handler body body' Hclosed Hclosed'
           Hname Hname' Hbodies).
Qed.

Lemma alpha_handler_swap (k : continuation) (params fresh : list variable)
    (metadata : handler_metadata) (handler body : expr) :
  code_closed handler = true ->
  distinct params = true ->
  length fresh = length params ->
  distinct_outside (params ++ free_variables_of_expr handler) fresh ->
  alpha_equivalent
    (Let_cont_non_recursive k (Handler params metadata handler) body)
    (Let_cont_non_recursive k
       (Handler fresh metadata
          (rename_expr (swap_variables params fresh) handler))
       body).
Proof.
  intros Hclosed Hdistinct Hlength Houtside.
  destruct (swap_variables_fixes params fresh handler Hclosed Houtside)
    as [Hvariables Hcontinuations].
  apply distinct_outside_app in Houtside as [Hparams _].
  pose proof
    (Alpha_handler_by_relation (swap_variables params fresh) k params
       metadata handler body Hclosed Hvariables Hcontinuations)
    as Hrule.
  cbn [rename_handler] in Hrule.
  rewrite (rename_variables_swap params fresh Hlength Hdistinct Hparams)
    in Hrule.
  exact Hrule.
Qed.

Theorem alpha_handler_at : alpha_handler_at_statement.
Proof.
  intros k params params' fresh metadata handler handler' body Hclosed
    Hclosed' Hdistinct Hdistinct' Hlength Hlength' Houtside Houtside'
    Hhandlers.
  apply Alpha_trans
    with (Let_cont_non_recursive k
            (Handler fresh metadata
               (rename_expr (swap_variables params fresh) handler))
            body);
    [ apply alpha_handler_swap; assumption | ].
  apply Alpha_trans
    with (Let_cont_non_recursive k
            (Handler fresh metadata
               (rename_expr (swap_variables params' fresh) handler'))
            body);
    [ | apply Alpha_sym, alpha_handler_swap; assumption ].
  exact
    (Alpha_context
       (Let_cont_non_recursive_handler k fresh metadata Hole body) _ _
       Hhandlers).
Qed.

Theorem alpha_handler_fresh : alpha_handler_fresh_statement.
Proof.
  intros k params params' metadata handler handler' body Hclosed Hclosed'
    Hdistinct Hdistinct' Hlength Hfresh.
  destruct
    (for_fresh_names_elim _ _
       ((params ++ free_variables_of_expr handler)
        ++ params' ++ free_variables_of_expr handler')
       Hfresh)
    as (fresh & Hlength_fresh & Houtside & Hhandlers).
  apply distinct_outside_app in Houtside as [Houtside Houtside'].
  apply (alpha_handler_at k params params' fresh metadata handler handler'
           body);
    solve [ assumption | exact (eq_trans Hlength_fresh Hlength) ].
Qed.

(* ======================================================================== *)
(* The pairs of the comparison                                              *)
(* ======================================================================== *)

(* A name that the map pairs with nothing matches itself and no other
   name. *)
Lemma names_match_fresh_left (m : name_map) (fresh a' : positive) :
  among fresh (map fst m) = false -> a' <> fresh ->
  names_match m fresh a' = false.
Proof.
  intros Hfresh Hdifferent. induction m as [ | [b b'] m IH]; simpl.
  - apply Pos.eqb_neq. congruence.
  - simpl in Hfresh. apply among_cons_false in Hfresh as [Hb Hfresh].
    destruct (Pos.eqb_spec fresh b); [ contradiction | ].
    destruct (Pos.eqb a' b'); [ reflexivity | exact (IH Hfresh) ].
Qed.

Lemma names_match_fresh_right (m : name_map) (fresh a : positive) :
  among fresh (map snd m) = false -> a <> fresh ->
  names_match m a fresh = false.
Proof.
  intros Hfresh Hdifferent. induction m as [ | [b b'] m IH]; simpl.
  - apply Pos.eqb_neq. exact Hdifferent.
  - simpl in Hfresh. apply among_cons_false in Hfresh as [Hb Hfresh].
    destruct (Pos.eqb_spec fresh b'); [ contradiction | ].
    destruct (Pos.eqb a b); [ reflexivity | exact (IH Hfresh) ].
Qed.

Lemma names_match_fresh_both (m : name_map) (fresh : positive) :
  among fresh (map fst m) = false -> among fresh (map snd m) = false ->
  names_match m fresh fresh = true.
Proof.
  intros Hfresh Hfresh'. induction m as [ | [b b'] m IH]; simpl.
  - apply Pos.eqb_refl.
  - simpl in Hfresh, Hfresh'.
    apply among_cons_false in Hfresh as [Hb Hfresh].
    apply among_cons_false in Hfresh' as [Hb' Hfresh'].
    destruct (Pos.eqb_spec fresh b); [ contradiction | ].
    destruct (Pos.eqb_spec fresh b'); [ contradiction | ].
    exact (IH Hfresh Hfresh').
Qed.

Theorem names_match_fresh : names_match_fresh_statement.
Proof.
  intros m x x' a a'. exists (a :: a' :: map fst m ++ map snd m).
  intros fresh Hfresh.
  apply among_cons_false in Hfresh as [Ha Hfresh].
  apply among_cons_false in Hfresh as [Ha' Hfresh].
  apply among_app_false in Hfresh as [Hleft Hright]. simpl.
  destruct (Pos.eqb_spec a x) as [-> | Hx];
    destruct (Pos.eqb_spec a' x') as [-> | Hx'].
  - rewrite !swap_first. symmetry.
    apply names_match_fresh_both; assumption.
  - rewrite swap_first, (swap_other x' fresh a') by congruence. symmetry.
    apply names_match_fresh_left; congruence.
  - rewrite swap_first, (swap_other x fresh a) by congruence. symmetry.
    apply names_match_fresh_right; congruence.
  - rewrite (swap_other x fresh a), (swap_other x' fresh a') by congruence.
    reflexivity.
Qed.
