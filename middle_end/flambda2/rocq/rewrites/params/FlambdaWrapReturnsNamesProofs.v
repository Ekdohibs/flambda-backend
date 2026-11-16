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

(* FlambdaWrapReturnsSpec.wrap_returns commutes with renaming and keeps
   scope: the lemmas from which FlambdaValidatorEquivarianceProofs.v
   and FlambdaStaticsScopeProofs.v prove the two statements. There is
   no statement here, and nothing here needs review.

   The pass compares names by equality only and makes none: the
   wrapper that it binds has the name of the continuation and the
   parameters of its handler. So each of its functions gives, on a
   renamed term, the renamed result. For the scope: an application
   that loses an argument loses free variables and no continuation;
   the wrapper has no free variable, and its one free continuation is
   the return continuation of the call that it is put around; and the
   parameter that the handler loses is not free in the handler. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec
  FlambdaValidatorEquivarianceBaseProofs FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaEquivarianceTactic
  FlambdaWrapReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The law of the pass is proved from the definitions alone
   ([equivariance], of FlambdaEquivarianceTactic.v): the tactic states
   and proves, as it goes, the laws of the functions that the pass is
   made of. Two laws are given to it besides. That of [distinct] at
   renamed variables is stated here, because the type of [distinct]
   does not say that its list is of variables. That of [var_free_b] is
   proved in FlambdaValidatorEquivarianceBaseProofs.v, and the tactic
   would otherwise derive it again at each use. *)

Lemma distinct_rename_variables (r : renaming) (params : list variable) :
  distinct (map (rename_variable r) params) = distinct params.
Proof. equivariance. Qed.

#[local] Hint Rewrite distinct_rename_variables var_free_b_equivariant_aux
  : equivariance.

Lemma wrap_returns_renamed : rewrite_equivariant wrap_returns.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

(* The wrapper has no free variable: what it applies [k] to are
   parameters of its own. *)
Lemma wrapper_free_variables (k : continuation) (i : nat)
    (params : list variable) (metadata : handler_metadata) (a : variable) :
  ~ In a (free_variables_of_handler (wrapper k i params metadata)).
Proof.
  unfold wrapper.
  cbn [free_variables_of_handler free_variables_of_expr].
  unfold free_variables_of_apply_cont, free_variables_of_simples.
  cbn [ac_args]. intros Ha. apply in_without in Ha as [Ha Hnot].
  apply in_flat_map in Ha as (s & Hs & Ha).
  apply in_map_iff in Hs as (x & <- & Hx).
  destruct Ha as [<- | []].
  exact (Hnot (in_drop_nth i params x Hx)).
Qed.

(* Its one free continuation is [k]. *)
Lemma wrapper_free_continuations (k : continuation) (i : nat)
    (params : list variable) (metadata : handler_metadata) :
  free_continuations_of_handler (wrapper k i params metadata) = [k].
Proof. reflexivity. Qed.

Fixpoint wrap_uses_scoped (k : continuation) (i : nat)
    (params : list variable) (metadata : handler_metadata) (e : expr)
    {struct e} :
  scoped e (wrap_uses k i (wrapper k i params metadata) e)
with wrap_uses_in_handler_scoped (k : continuation) (i : nat)
    (params : list variable) (metadata : handler_metadata)
    (h : continuation_handler) {struct h} :
  scoped_handler h
    (wrap_uses_in_handler k i (wrapper k i params metadata) h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [wrap_uses]; try apply scoped_refl.
    + apply scoped_let; [ apply incl_refl | apply wrap_uses_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl | apply wrap_uses_scoped ].
    + apply scoped_let_cont;
        [ apply wrap_uses_in_handler_scoped | apply wrap_uses_scoped ].
    + apply scoped_let_cont_recursive; [ | | apply wrap_uses_scoped ].
      * rewrite map_map. apply map_ext. intros [k' hh]. reflexivity.
      * induction handlers as [ | [k' hh] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply wrap_uses_in_handler_scoped | exact IH ].
    + destruct (Pos.eqb_spec (ap_continuation ap) k) as [Hreturn | _];
        [ | apply scoped_refl ].
      intros _.
      cbn [code_closed free_variables_of_expr free_continuations_of_expr].
      split; [ reflexivity | split ].
      * intros a Ha. apply in_app_or in Ha as [Ha | Ha]; [ | exact Ha ].
        exfalso. exact (wrapper_free_variables k i params metadata a Ha).
      * rewrite wrapper_free_continuations. intros a [<- | Ha].
        -- left. exact Hreturn.
        -- apply in_without in Ha as [Ha _]. exact Ha.
    + intros _.
      cbn [code_closed free_variables_of_expr free_continuations_of_expr].
      split; [ reflexivity | split ].
      * apply drop_arg_of_free_variables.
      * rewrite drop_arg_of_continuations. apply incl_refl.
    + intros _.
      cbn [code_closed free_variables_of_expr free_continuations_of_expr].
      split; [ reflexivity | split ].
      * apply incl_app_app; [ apply incl_refl | ].
        induction arms as [ | [n ac] arms IH]; [ apply incl_refl | ].
        cbn [map flat_map snd].
        apply incl_app_app; [ apply drop_arg_of_free_variables | exact IH ].
      * induction arms as [ | [n ac] arms IH]; [ apply incl_refl | ].
        cbn [map flat_map snd]. rewrite drop_arg_of_continuations.
        apply incl_app_app; [ apply incl_refl | exact IH ].
  - destruct h as [ps m he]. cbn [wrap_uses_in_handler].
    apply scoped_handler_intro, wrap_uses_scoped.
Qed.

Lemma wrap_return_scoped (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  wrap_return i k h body = Some (h', body') ->
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold wrap_return.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && negb (var_free_b p handler) && only_applied_or_returned_to k body
     && (2 <=? uses_of k body))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [_ Hunused].
  apply negb_true_iff in Hunused.
  intros Hclosed.
  cbn [code_closed handler_code_closed] in Hclosed |- *.
  apply andb_true_iff in Hclosed as [Hhandler Hbody].
  destruct (wrap_uses_scoped k i params metadata body Hbody)
    as (Hbody' & Hv & Hk).
  split; [ rewrite Hhandler, Hbody'; reflexivity | ].
  cbn [free_variables_of_expr free_variables_of_handler
       free_continuations_of_expr free_continuations_of_handler].
  split.
  - intros a Ha. apply in_or_app. apply in_app_or in Ha as [Ha | Ha];
      [ | right; exact (Hv a Ha) ].
    apply in_without in Ha as [Ha Hnot].
    left. apply in_without. split; [ exact Ha | ].
    intros Hin.
    destruct (in_nth_or_drop_nth i params p a Hparam Hin) as [-> | Hin'];
      [ | exact (Hnot Hin') ].
    rewrite (var_free_b_free p handler Hhandler Ha) in Hunused.
    discriminate.
  - apply incl_app_app; [ apply incl_refl | apply incl_without, Hk ].
Qed.

Lemma wrap_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (wrap_from i k h body))
         (snd (wrap_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [wrap_from];
    [ apply scoped_refl | ].
  destruct (wrap_return i k h body) as [[h' body'] | ] eqn:Hreturn;
    [ | apply IH ].
  exact (wrap_return_scoped i k h body h' body' Hreturn).
Qed.

Fixpoint wrap_returns_scoped (e : expr) : scoped e (wrap_returns e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [wrap_returns]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply wrap_returns_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply wrap_returns_scoped ].
  - pose proof
      (wrap_from_scoped (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as Hfrom.
    destruct
      (wrap_from (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as [h' body'].
    refine (scoped_trans _ _ _ _ Hfrom).
    apply scoped_let_cont;
      [ apply scoped_handler_intro, wrap_returns_scoped
      | apply wrap_returns_scoped ].
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply wrap_returns_scoped ].
Qed.

Lemma wrap_returns_scope : keeps_scope wrap_returns.
Proof. apply keeps_scope_intro, wrap_returns_scoped. Qed.
