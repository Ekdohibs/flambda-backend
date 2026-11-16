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

(* FlambdaBindConstParamsSpec.bind_const_params commutes with renaming
   and keeps scope: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaStaticsScopeProofs.v
   prove the two statements. There is no statement here, and nothing
   here needs review.

   The pass compares names by equality only and makes none, and a
   renaming moves no symbol and no constant, so each of its functions
   gives, on a renamed term, the renamed result. For the scope: an
   application that loses an argument loses free variables and no
   continuation; the [Let] that the handler gains mentions no variable;
   and the parameter that the handler loses is bound by that [Let]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaUnaliasParamsSpec FlambdaUnaliasParamsProofs
  FlambdaUnaliasParamsNamesProofs FlambdaEquivarianceTactic
  FlambdaBindConstParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The pass commutes with renaming                                          *)
(* ======================================================================== *)

(* The law of the pass is proved from the definitions alone
   ([equivariance], of FlambdaEquivarianceTactic.v): the tactic states
   and proves, as it goes, the laws of the functions that the pass is
   made of, the law of [same_simple] among them (a renaming is a
   permutation: two simples are the same once renamed if and only if
   they are). It is given one step: how to go under [first_answer], a
   function that takes a function, with the lemma of
   FlambdaUnaliasParamsNamesProofs.v, as that file does. *)

#[local] Ltac eqv_extra ::=
  match goal with
  | |- context [first_answer ?f' ?l] =>
    match goal with
    | |- context [first_answer ?f l] =>
      tryif constr_eq f f' then fail
      else (eqv_no_renaming f;
            erewrite (first_answer_ext_among f' f _ l);
            [ | eqv_enter l; solve [ eqv_core ] ])
    end
  end.

Lemma bind_const_params_renamed : rewrite_equivariant bind_const_params.
Proof. equivariance. Qed.

(* ======================================================================== *)
(* The pass keeps scope                                                     *)
(* ======================================================================== *)

Lemma bind_const_param_scoped (i : nat) (k : continuation)
    (h : continuation_handler) (body : expr) (h' : continuation_handler)
    (body' : expr) :
  bind_const_param i k h body = Some (h', body') ->
  scoped (Let_cont_non_recursive k h body)
    (Let_cont_non_recursive k h' body').
Proof.
  destruct h as [params metadata handler]. unfold bind_const_param.
  destruct (nth_error params i) as [p | ] eqn:Hparam; [ | discriminate ].
  destruct (nth_error (hm_params_kinds metadata) i) as [kind | ];
    [ | discriminate ].
  destruct (first_given k i body) as [[s | ] | ]; try discriminate.
  destruct
    (is_plain s && negb (hm_is_exn_handler metadata) && all_give k i s body
     && negb (among p (drop_nth i params)))
    eqn:Hcond; [ | discriminate ].
  intros [= <- <-].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hcond _].
  apply andb_true_iff in Hcond as [Hplain _].
  assert (Hnone : free_variables_of_simple s = [])
    by (destruct s; [ discriminate | reflexivity | reflexivity ]).
  intros Hclosed.
  cbn [code_closed handler_code_closed] in Hclosed |- *.
  apply andb_true_iff in Hclosed as [Hhandler Hbody].
  destruct (drop_arg_scoped k i body Hbody) as (Hbody' & Hv & Hk).
  split; [ rewrite Hhandler, Hbody'; reflexivity | ].
  cbn [free_variables_of_expr free_variables_of_handler
       free_variables_of_named free_continuations_of_expr
       free_continuations_of_handler].
  rewrite Hnone. cbn [app].
  split.
  - intros a Ha. apply in_or_app. apply in_app_or in Ha as [Ha | Ha];
      [ | right; exact (Hv a Ha) ].
    apply in_without in Ha as [Ha Hnot].
    apply in_without in Ha as [Ha Hne].
    left. apply in_without. split; [ exact Ha | ].
    intros Hin.
    destruct (in_nth_or_drop_nth i params p a Hparam Hin) as [-> | Hin'];
      [ apply Hne; left; reflexivity | exact (Hnot Hin') ].
  - apply incl_app_app; [ apply incl_refl | apply incl_without, Hk ].
Qed.

Lemma bind_const_from_scoped (i : nat) :
  ∀ (k : continuation) (h : continuation_handler) (body : expr),
    scoped (Let_cont_non_recursive k h body)
      (Let_cont_non_recursive k (fst (bind_const_from i k h body))
         (snd (bind_const_from i k h body))).
Proof.
  induction i as [ | i IH]; intros k h body; cbn [bind_const_from];
    [ apply scoped_refl | ].
  destruct (bind_const_param i k h body) as [[h' body'] | ] eqn:Hparam;
    [ | apply IH ].
  exact (scoped_trans _ _ _
    (bind_const_param_scoped i k h body h' body' Hparam) (IH k h' body')).
Qed.

Fixpoint bind_const_params_scoped (e : expr) :
  scoped e (bind_const_params e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k [params metadata handler] body
       | handlers body | ap | ac | scrutinee arms | ];
    cbn [bind_const_params]; try apply scoped_refl.
  - apply scoped_let; [ apply incl_refl | apply bind_const_params_scoped ].
  - apply scoped_let_static;
      [ apply scoped_bindings_refl | apply bind_const_params_scoped ].
  - pose proof
      (bind_const_from_scoped (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as Hfrom.
    destruct
      (bind_const_from (length params) k
         (Handler params metadata (bind_const_params handler))
         (bind_const_params body))
      as [h' body'].
    refine (scoped_trans _ _ _ _ Hfrom).
    apply scoped_let_cont;
      [ apply scoped_handler_intro, bind_const_params_scoped
      | apply bind_const_params_scoped ].
  - apply scoped_let_cont_recursive;
      [ reflexivity | apply scoped_handlers_refl
      | apply bind_const_params_scoped ].
Qed.

Lemma bind_const_params_scope : keeps_scope bind_const_params.
Proof. apply keeps_scope_intro, bind_const_params_scoped. Qed.
