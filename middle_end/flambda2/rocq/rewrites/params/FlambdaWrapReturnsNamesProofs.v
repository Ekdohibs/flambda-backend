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
  FlambdaUnaliasParamsNamesProofs FlambdaWrapReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma trap_free_of_equivariant (r : renaming) (k : continuation)
    (ac : apply_cont) :
  trap_free_of (rename_continuation r k) (rename_apply_cont r ac)
  = trap_free_of k ac.
Proof.
  unfold trap_free_of, rename_apply_cont. cbn [ac_trap_action].
  destruct (ac_trap_action ac) as [action | ]; [ | reflexivity ].
  cbn [option_map].
  rewrite trap_action_handler_renamed, rename_continuation_eqb. reflexivity.
Qed.

Fixpoint only_applied_or_returned_to_equivariant (r : renaming)
    (k : continuation) (e : expr) {struct e} :
  only_applied_or_returned_to (rename_continuation r k) (rename_expr r e)
  = only_applied_or_returned_to k e
with handler_only_applied_or_returned_to_equivariant (r : renaming)
    (k : continuation) (h : continuation_handler) {struct h} :
  handler_only_applied_or_returned_to (rename_continuation r k)
    (rename_handler r h)
  = handler_only_applied_or_returned_to k h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr only_applied_or_returned_to]; try reflexivity.
    + apply only_applied_or_returned_to_equivariant.
    + apply only_applied_or_returned_to_equivariant.
    + rewrite rename_continuation_eqb,
        handler_only_applied_or_returned_to_equivariant,
        only_applied_or_returned_to_equivariant.
      reflexivity.
    + rewrite only_applied_or_returned_to_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map forallb].
      rewrite rename_continuation_eqb,
        handler_only_applied_or_returned_to_equivariant, IH.
      reflexivity.
    + cbn [rename_apply ap_exn_continuation].
      rewrite rename_continuation_eqb. reflexivity.
    + apply trap_free_of_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map forallb fst snd].
      rewrite trap_free_of_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler handler_only_applied_or_returned_to].
    apply only_applied_or_returned_to_equivariant.
Qed.

Lemma one_if_same_equivariant (r : renaming) (k k' : continuation) :
  one_if_same (rename_continuation r k) (rename_continuation r k')
  = one_if_same k k'.
Proof. unfold one_if_same. rewrite rename_continuation_eqb. reflexivity. Qed.

Fixpoint uses_of_equivariant (r : renaming) (k : continuation) (e : expr)
    {struct e} :
  uses_of (rename_continuation r k) (rename_expr r e) = uses_of k e
with uses_in_handler_equivariant (r : renaming) (k : continuation)
    (h : continuation_handler) {struct h} :
  uses_in_handler (rename_continuation r k) (rename_handler r h)
  = uses_in_handler k h.
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr uses_of]; try reflexivity.
    + apply uses_of_equivariant.
    + apply uses_of_equivariant.
    + rewrite uses_in_handler_equivariant, uses_of_equivariant. reflexivity.
    + rewrite uses_of_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map list_sum fold_right].
      rewrite uses_in_handler_equivariant. f_equal. exact IH.
    + cbn [rename_apply ap_continuation]. apply one_if_same_equivariant.
    + unfold rename_apply_cont. cbn [ac_continuation].
      apply one_if_same_equivariant.
    + induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map list_sum fold_right fst snd].
      unfold rename_apply_cont at 1. cbn [ac_continuation].
      rewrite one_if_same_equivariant. f_equal. exact IH.
  - destruct h as [params metadata handler].
    cbn [rename_handler uses_in_handler]. apply uses_of_equivariant.
Qed.

Lemma distinct_rename_variables (r : renaming) (params : list variable) :
  distinct (map (rename_variable r) params) = distinct params.
Proof.
  induction params as [ | x params IH]; [ reflexivity | ].
  cbn [map distinct]. rewrite among_rename_variable, IH. reflexivity.
Qed.

Lemma wrapper_equivariant (r : renaming) (k : continuation) (i : nat)
    (params : list variable) (metadata : handler_metadata) :
  wrapper (rename_continuation r k) i (map (rename_variable r) params)
    metadata
  = rename_handler r (wrapper k i params metadata).
Proof.
  unfold wrapper. cbn [rename_handler rename_expr].
  unfold rename_apply_cont.
  cbn [ac_continuation ac_args ac_trap_action option_map].
  rewrite drop_nth_map, !map_map. reflexivity.
Qed.

Fixpoint wrap_uses_equivariant (r : renaming) (k : continuation) (i : nat)
    (w : continuation_handler) (e : expr) {struct e} :
  wrap_uses (rename_continuation r k) i (rename_handler r w)
    (rename_expr r e)
  = rename_expr r (wrap_uses k i w e)
with wrap_uses_in_handler_equivariant (r : renaming) (k : continuation)
    (i : nat) (w : continuation_handler) (h : continuation_handler)
    {struct h} :
  wrap_uses_in_handler (rename_continuation r k) i (rename_handler r w)
    (rename_handler r h)
  = rename_handler r (wrap_uses_in_handler k i w h).
Proof.
  - destruct e as [y kind d body | bindings body | k' hh body | handlers body
                  | ap | ac | scrutinee arms | ];
      cbn [rename_expr wrap_uses]; try reflexivity.
    + rewrite wrap_uses_equivariant. reflexivity.
    + rewrite wrap_uses_equivariant. reflexivity.
    + rewrite wrap_uses_in_handler_equivariant, wrap_uses_equivariant.
      reflexivity.
    + rewrite wrap_uses_equivariant. f_equal.
      induction handlers as [ | [k' hh] handlers IH]; [ reflexivity | ].
      cbn [map]. rewrite wrap_uses_in_handler_equivariant, IH. reflexivity.
    + cbn [rename_apply ap_continuation].
      rewrite rename_continuation_eqb.
      destruct (Pos.eqb (ap_continuation ap) k); reflexivity.
    + rewrite drop_arg_of_equivariant. reflexivity.
    + f_equal. induction arms as [ | [n ac] arms IH]; [ reflexivity | ].
      cbn [map fst snd]. rewrite drop_arg_of_equivariant, IH. reflexivity.
  - destruct h as [params metadata handler].
    cbn [rename_handler wrap_uses_in_handler].
    rewrite wrap_uses_equivariant. reflexivity.
Qed.

Lemma wrap_return_equivariant (r : renaming) (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) :
  wrap_return i (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_unaliased r (wrap_return i k h body).
Proof.
  destruct h as [params metadata handler].
  cbn [rename_handler wrap_return].
  rewrite nth_error_map.
  destruct (nth_error params i) as [p | ]; [ | reflexivity ].
  cbn [option_map].
  rewrite distinct_rename_variables, var_free_b_equivariant_aux,
    only_applied_or_returned_to_equivariant, uses_of_equivariant,
    wrapper_equivariant, wrap_uses_equivariant, drop_nth_map.
  destruct
    (negb (hm_is_exn_handler metadata) && distinct params
     && negb (var_free_b p handler) && only_applied_or_returned_to k body
     && (2 <=? uses_of k body));
    reflexivity.
Qed.

Lemma wrap_from_equivariant (r : renaming) (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    wrap_from i (rename_continuation r k) (rename_handler r h)
      (rename_expr r body)
    = (rename_handler r (fst (wrap_from i k h body)),
       rename_expr r (snd (wrap_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [wrap_from];
    [ reflexivity | ].
  rewrite wrap_return_equivariant.
  destruct (wrap_return i k h body) as [[h' body'] | ];
    cbn [rename_unaliased option_map]; [ reflexivity | apply IH ].
Qed.

Fixpoint wrap_returns_equivariant_aux (r : renaming) (e : expr) :
  wrap_returns (rename_expr r e) = rename_expr r (wrap_returns e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler wrap_returns]; try reflexivity.
  - rewrite wrap_returns_equivariant_aux. reflexivity.
  - rewrite wrap_returns_equivariant_aux. reflexivity.
  - rewrite !wrap_returns_equivariant_aux, length_map.
    pose proof
      (wrap_from_equivariant r (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as Hfrom.
    cbn [rename_handler] in Hfrom. rewrite Hfrom.
    destruct
      (wrap_from (length params) k
         (Handler params metadata (wrap_returns handler))
         (wrap_returns body))
      as [h' body'].
    reflexivity.
  - rewrite wrap_returns_equivariant_aux. reflexivity.
Qed.

Lemma wrap_returns_renamed : rewrite_equivariant wrap_returns.
Proof. exact wrap_returns_equivariant_aux. Qed.

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
