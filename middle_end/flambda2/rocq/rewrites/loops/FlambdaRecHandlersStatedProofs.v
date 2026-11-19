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

(* Proofs for FlambdaRecHandlersStatedSpec.v. Nothing here needs
   review.

   The statement is proved layer by layer with the laws of
   FlambdaStatedContextProofs.v, which take an equivalence under what
   is known through each binder: the handlers of a recursive
   [Let_cont] one at a time
   (FlambdaStatedContextProofs.let_cont_recursive_handler_equivalent_stated),
   then its body. The statement for a rewrite that refines is proved
   by the same script with the laws of the refinement under what is
   known. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaSemanticsSpec FlambdaSemanticsProofs FlambdaNormaliseSpec
  FlambdaNormaliseProofs FlambdaStatedScopeSpec FlambdaStatedScopeProofs
  FlambdaStatedContextProofs FlambdaRecHandlersSpec FlambdaRecHandlersProofs
  FlambdaRecHandlersStatedSpec.
Import ListNotations.

(* The rewrite keeps whether a term binds something static. *)
Fixpoint rewrite_rec_handlers_stated_binds_statics
    (f : stated_in_scope -> expr -> expr) (e : expr) {struct e} :
  ∀ S : stated_in_scope,
    binds_statics (rewrite_rec_handlers_stated f S e) = binds_statics e.
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros S; simpl; try reflexivity.
  - apply rewrite_rec_handlers_stated_binds_statics.
  - rewrite rewrite_rec_handlers_stated_binds_statics. reflexivity.
  - destruct h as [params metadata handler]. simpl.
    rewrite !rewrite_rec_handlers_stated_binds_statics. reflexivity.
  - rewrite rewrite_rec_handlers_stated_binds_statics. f_equal.
    induction handlers as [ | [k [params metadata handler]] handlers IH]; simpl;
      [ reflexivity | ].
    rewrite rewrite_code_body_binds_statics,
      rewrite_rec_handlers_stated_binds_statics, IH.
    reflexivity.
Qed.

(* [rewrite_code_body] is an equivalence under what is known where the
   rewrite is one at the body. *)
Lemma rewrite_code_body_equivalent_stated (g : expr -> expr)
    (S : stated_in_scope) (body : expr) :
  expr_equivalent_stated S body (g body) ->
  expr_equivalent_stated S body (rewrite_code_body g body).
Proof.
  intros Hg. unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (g body)));
    [ exact Hg | apply expr_equivalent_stated_refl ].
Qed.

Fixpoint rewrite_rec_handlers_stated_equivalent_aux
    (f : stated_in_scope -> expr -> expr)
    (Hf : ∀ (S : stated_in_scope) (e : expr),
        expr_equivalent_stated S e (f S e))
    (e : expr) {struct e} :
  ∀ S : stated_in_scope,
    expr_equivalent_stated S e (rewrite_rec_handlers_stated f S e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros S; simpl; try apply expr_equivalent_stated_refl.
  - apply let_equivalent_stated,
      (rewrite_rec_handlers_stated_equivalent_aux f Hf).
  - apply let_static_equivalent_stated,
      (rewrite_rec_handlers_stated_equivalent_aux f Hf).
  - destruct h as [params metadata handler]. simpl.
    apply (expr_equivalent_stated_trans S _
      (Let_cont_non_recursive k
         (Handler params metadata
            (rewrite_rec_handlers_stated f
               (params_stated params (hm_params_subkinds metadata) S)
               handler))
         body)).
    + apply let_cont_handler_equivalent_stated,
        (rewrite_rec_handlers_stated_equivalent_aux f Hf).
    + apply let_cont_body_equivalent_stated,
        (rewrite_rec_handlers_stated_equivalent_aux f Hf).
  - apply (expr_equivalent_stated_trans S _
      (Let_cont_recursive
         (map (fun '(k, h) => (k, rewrite_rec_handler_stated f S h)) handlers)
         body));
      [ | apply let_cont_recursive_body_equivalent_stated,
            (rewrite_rec_handlers_stated_equivalent_aux f Hf) ].
    assert (Hhandlers :
      ∀ before : list (continuation * continuation_handler),
        expr_equivalent_stated S
          (Let_cont_recursive (before ++ handlers) body)
          (Let_cont_recursive
             (before
              ++ map (fun '(k, h) => (k, rewrite_rec_handler_stated f S h))
                   handlers)
             body)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros before; simpl.
      - apply expr_equivalent_stated_refl.
      - eapply expr_equivalent_stated_trans.
        + apply
            (let_cont_recursive_handler_equivalent_stated S before k params
               metadata handler
               (rewrite_code_body
                  (f (params_stated params (hm_params_subkinds metadata) S))
                  (rewrite_rec_handlers_stated f
                     (params_stated params (hm_params_subkinds metadata) S)
                     handler))
               handlers body).
          * rewrite rewrite_code_body_binds_statics,
              rewrite_rec_handlers_stated_binds_statics.
            reflexivity.
          * eapply expr_equivalent_stated_trans;
              [ apply (rewrite_rec_handlers_stated_equivalent_aux f Hf)
              | apply rewrite_code_body_equivalent_stated, Hf ].
        + specialize
            (IH
               (before
                ++ [(k,
                     Handler params metadata
                       (rewrite_code_body
                          (f (params_stated params
                                (hm_params_subkinds metadata) S))
                          (rewrite_rec_handlers_stated f
                             (params_stated params
                                (hm_params_subkinds metadata) S)
                             handler)))])).
          rewrite <- !app_assoc in IH. exact IH. }
    exact (Hhandlers []).
Qed.

Theorem rewrite_rec_handlers_stated_equivalent :
  rewrite_rec_handlers_stated_equivalent_statement.
Proof.
  intros f Hf S e.
  exact (rewrite_rec_handlers_stated_equivalent_aux f Hf e S).
Qed.

(* ======================================================================== *)
(* A rewrite that refines                                                   *)
(* ======================================================================== *)

Lemma rewrite_code_body_refines_stated (g : expr -> expr)
    (S : stated_in_scope) (body : expr) :
  expr_refines_stated S body (g body) ->
  expr_refines_stated S body (rewrite_code_body g body).
Proof.
  intros Hg. unfold rewrite_code_body.
  destruct (Bool.eqb (binds_statics body) (binds_statics (g body)));
    [ exact Hg | apply expr_refines_stated_refl ].
Qed.

Fixpoint rewrite_rec_handlers_stated_refines_aux
    (f : stated_in_scope -> expr -> expr)
    (Hf : ∀ (S : stated_in_scope) (e : expr),
        expr_refines_stated S e (f S e))
    (e : expr) {struct e} :
  ∀ S : stated_in_scope,
    expr_refines_stated S e (rewrite_rec_handlers_stated f S e).
Proof.
  destruct e
    as [ x kind d body | bindings body | k h body | handlers body
       | ap | ac | scrutinee arms | ];
    intros S; simpl; try apply expr_refines_stated_refl.
  - apply let_stated_refines,
      (rewrite_rec_handlers_stated_refines_aux f Hf).
  - apply let_static_stated_refines,
      (rewrite_rec_handlers_stated_refines_aux f Hf).
  - destruct h as [params metadata handler]. simpl.
    apply (expr_refines_stated_trans S _
      (Let_cont_non_recursive k
         (Handler params metadata
            (rewrite_rec_handlers_stated f
               (params_stated params (hm_params_subkinds metadata) S)
               handler))
         body)).
    + apply let_cont_handler_stated_refines,
        (rewrite_rec_handlers_stated_refines_aux f Hf).
    + apply let_cont_body_stated_refines,
        (rewrite_rec_handlers_stated_refines_aux f Hf).
  - apply (expr_refines_stated_trans S _
      (Let_cont_recursive
         (map (fun '(k, h) => (k, rewrite_rec_handler_stated f S h)) handlers)
         body));
      [ | apply let_cont_recursive_body_stated_refines,
            (rewrite_rec_handlers_stated_refines_aux f Hf) ].
    assert (Hhandlers :
      ∀ before : list (continuation * continuation_handler),
        expr_refines_stated S
          (Let_cont_recursive (before ++ handlers) body)
          (Let_cont_recursive
             (before
              ++ map (fun '(k, h) => (k, rewrite_rec_handler_stated f S h))
                   handlers)
             body)).
    { induction handlers as [ | [k [params metadata handler]] handlers IH];
        intros before; simpl.
      - apply expr_refines_stated_refl.
      - eapply expr_refines_stated_trans.
        + apply
            (let_cont_recursive_handler_stated_refines S before k params
               metadata handler
               (rewrite_code_body
                  (f (params_stated params (hm_params_subkinds metadata) S))
                  (rewrite_rec_handlers_stated f
                     (params_stated params (hm_params_subkinds metadata) S)
                     handler))
               handlers body).
          * rewrite rewrite_code_body_binds_statics,
              rewrite_rec_handlers_stated_binds_statics.
            reflexivity.
          * eapply expr_refines_stated_trans;
              [ apply (rewrite_rec_handlers_stated_refines_aux f Hf)
              | apply rewrite_code_body_refines_stated, Hf ].
        + specialize
            (IH
               (before
                ++ [(k,
                     Handler params metadata
                       (rewrite_code_body
                          (f (params_stated params
                                (hm_params_subkinds metadata) S))
                          (rewrite_rec_handlers_stated f
                             (params_stated params
                                (hm_params_subkinds metadata) S)
                             handler)))])).
          rewrite <- !app_assoc in IH. exact IH. }
    exact (Hhandlers []).
Qed.

Theorem rewrite_rec_handlers_stated_refines :
  rewrite_rec_handlers_stated_refines_statement.
Proof.
  intros f Hf S e.
  exact (rewrite_rec_handlers_stated_refines_aux f Hf e S).
Qed.
