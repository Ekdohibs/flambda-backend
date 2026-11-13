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

(* FlambdaUnaliasRecParamsSpec.unalias_rec_params commutes with
   renaming and keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, so each of
   its functions gives, on a renamed term, the renamed result. For the
   scope: an application that loses an argument loses free variables
   and no continuation, in the body of the [Let_cont] and in the
   handler; the [Let] that the handler gains mentions the variable
   that the applications in the body gave, which is free in the body
   where an application gives it and no binder on the way binds it;
   and the parameter that the handler loses is bound by that [Let].
   The lemmas about the check and about the argument dropped are those
   of FlambdaUnaliasParamsNamesProofs.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaUnaliasRecParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

Lemma unalias_rec_param_equivariant (r : renaming) (i : nat)
    (k : continuation) (h : continuation_handler) (body : expr) :
  unalias_rec_param i (rename_continuation r k) (rename_handler r h)
    (rename_expr r body)
  = rename_unaliased r (unalias_rec_param i k h body).
Proof.
  destruct h as [params metadata handler].
  cbn [rename_handler unalias_rec_param].
  rewrite nth_error_map, first_given_var_equivariant.
  destruct (nth_error params i) as [p | ]; [ | reflexivity ].
  cbn [option_map].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ];
    [ | reflexivity ].
  destruct (first_given_var k i body) as [[x | ] | ]; try reflexivity.
  cbn [rename_given option_map].
  rewrite !all_give_var_equivariant, drop_nth_map, !among_rename_variable,
    !drop_arg_equivariant.
  destruct
    (negb (hm_is_exn_handler metadata) && all_give_var k i x true body
     && all_give_var k i p true handler
     && negb (among x (drop_nth i params))
     && negb (among p (drop_nth i params)));
    reflexivity.
Qed.

Lemma unalias_rec_from_equivariant (r : renaming) (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    unalias_rec_from i (rename_continuation r k) (rename_handler r h)
      (rename_expr r body)
    = (rename_handler r (fst (unalias_rec_from i k h body)),
       rename_expr r (snd (unalias_rec_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [unalias_rec_from];
    [ reflexivity | ].
  rewrite unalias_rec_param_equivariant.
  destruct (unalias_rec_param i k h body) as [[h' body'] | ];
    cbn [rename_unaliased option_map]; apply IH.
Qed.

Fixpoint unalias_rec_params_equivariant_aux (r : renaming) (e : expr) :
  unalias_rec_params (rename_expr r e) = rename_expr r (unalias_rec_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [rename_expr rename_handler unalias_rec_params]; try reflexivity.
  - rewrite unalias_rec_params_equivariant_aux. reflexivity.
  - rewrite unalias_rec_params_equivariant_aux. reflexivity.
  - rewrite !unalias_rec_params_equivariant_aux. reflexivity.
  - rewrite unalias_rec_params_equivariant_aux.
    destruct handlers
      as [ | [k [params metadata handler]] [ | second handlers]];
      try reflexivity.
    cbn [map rename_handler]. rewrite length_map.
    pose proof
      (unalias_rec_from_equivariant r (length params) k
         (Handler params metadata handler) (unalias_rec_params body))
      as Hfrom.
    cbn [rename_handler] in Hfrom. rewrite Hfrom.
    destruct
      (unalias_rec_from (length params) k (Handler params metadata handler)
         (unalias_rec_params body))
      as [h' body'].
    reflexivity.
Qed.

Lemma unalias_rec_params_renamed : rewrite_equivariant unalias_rec_params.
Proof. exact unalias_rec_params_equivariant_aux. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma unalias_rec_param_scoped (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  unalias_rec_param i k h body = Some (h', body') ->
  scoped (Let_cont_recursive [(k, h)] body)
    (Let_cont_recursive [(k, h')] body').
Proof.
  destruct h as [params metadata handler]. unfold unalias_rec_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ];
    [ | discriminate ].
  destruct (first_given_var k i body) as [[x | ] | ] eqn:Hfirst;
    try discriminate.
  destruct
    (negb (hm_is_exn_handler metadata) && all_give_var k i x true body
     && all_give_var k i p true handler
     && negb (among x (drop_nth i params))
     && negb (among p (drop_nth i params)))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [_ Hgive].
  destruct (given_free k i x body true (Some x) Hfirst Hgive) as [_ Hx].
  intros Hclosed.
  cbn [code_closed handler_code_closed forallb] in Hclosed |- *.
  apply andb_true_iff in Hclosed as [Hhandler Hbody].
  apply andb_true_iff in Hhandler as [Hhandler _].
  destruct (drop_arg_scoped k i handler Hhandler) as (Hhandler' & Hvh & Hkh).
  destruct (drop_arg_scoped k i body Hbody) as (Hbody' & Hv & Hk).
  split; [ rewrite Hhandler', Hbody'; reflexivity | ].
  cbn [free_variables_of_expr free_variables_of_handler
       free_variables_of_named free_variables_of_simple
       free_continuations_of_expr free_continuations_of_handler
       flat_map map fst].
  rewrite !app_nil_r.
  split.
  - intros a Ha. apply in_or_app. apply in_app_or in Ha as [Ha | Ha];
      [ | right; exact (Hv a Ha) ].
    apply in_without in Ha as [Ha Hnot].
    apply in_app_or in Ha as [[<- | []] | Ha]; [ right; exact Hx | ].
    apply in_without in Ha as [Ha Hne].
    left. apply in_without. split; [ exact (Hvh a Ha) | ].
    intros Hin.
    destruct (in_nth_or_drop_nth i params p a Hparam Hin) as [-> | Hin'];
      [ apply Hne; left; reflexivity | exact (Hnot Hin') ].
  - apply incl_without, incl_app_app; assumption.
Qed.

Lemma unalias_rec_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_recursive [(k, h)] body)
      (Let_cont_recursive [(k, fst (unalias_rec_from i k h body))]
         (snd (unalias_rec_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [unalias_rec_from];
    [ apply scoped_refl | ].
  destruct (unalias_rec_param i k h body) as [[h' body'] | ] eqn:Hparam;
    [ | apply IH ].
  exact
    (scoped_trans _ _ _ (unalias_rec_param_scoped i k h body h' body' Hparam)
       (IH k h' body')).
Qed.

Fixpoint unalias_rec_params_scoped (e : expr) :
  scoped e (unalias_rec_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [unalias_rec_params]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply unalias_rec_params_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply unalias_rec_params_scoped ].
  - apply scoped_let_cont;
      [ apply scoped_handler_intro, unalias_rec_params_scoped
      | apply unalias_rec_params_scoped ].
  - assert (Hbody :
      scoped (Let_cont_recursive handlers body)
        (Let_cont_recursive handlers (unalias_rec_params body))).
    { apply scoped_let_cont_recursive;
        [ reflexivity | apply scoped_handlers_refl
        | apply unalias_rec_params_scoped ]. }
    destruct handlers
      as [ | [k [params metadata handler]] [ | second handlers]];
      try exact Hbody.
    pose proof
      (unalias_rec_from_scoped (length params) k
         (Handler params metadata handler) (unalias_rec_params body))
      as Hfrom.
    destruct
      (unalias_rec_from (length params) k (Handler params metadata handler)
         (unalias_rec_params body))
      as [h' body'].
    exact (scoped_trans _ _ _ Hbody Hfrom).
Qed.

Lemma unalias_rec_params_scope : keeps_scope unalias_rec_params.
Proof. apply keeps_scope_intro, unalias_rec_params_scoped. Qed.
