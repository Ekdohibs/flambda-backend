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

(* FlambdaRecHandlersStatedSpec.rewrite_rec_handlers_stated commutes
   with renaming where the rewrite that it is given does, and keeps
   scope where that rewrite does: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaValidateScopeProofs.v
   prove that the rounds of the validation commute with renaming and
   keep scope. There is no statement here, and nothing here needs
   review.

   The functions act on names only through the rewrite and through
   what is known of the variables in scope, which is renamed by
   FlambdaStatedScopeNamesProofs.rename_stated_in_scope. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaStatedScopeSpec FlambdaStatedScopeNamesProofs
  FlambdaRecHandlersStatedSpec FlambdaEquivarianceTactic.
Import ListNotations.

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance stated_in_scope_renaming : Renaming_of stated_in_scope :=
  rename_stated_in_scope.

#[local] Hint Rewrite stated_of_named_equivariant params_stated_equivariant
  : equivariance.

Section Rewrite_rec_handlers_stated.

Variable f : stated_in_scope -> expr -> expr.
Hypothesis f_equivariant :
  ∀ (r : renaming) (S : stated_in_scope) (e : expr),
    f (rename_stated_in_scope r S) (rename_expr r e) = rename_expr r (f S e).

#[local] Arguments rename_stated_in_scope : simpl never.
#[local] Hint Rewrite stated_in_scope_cons_renamed : equivariance.

(* The law of [rewrite_code_body] at the rewrite given what is known:
   stated, as the tactic cannot read it from the type of
   [rewrite_code_body]. *)
Lemma rewrite_code_body_stated_equivariant (r : renaming)
    (S : stated_in_scope) (body : expr) :
  rewrite_code_body (f (rename_stated_in_scope r S)) (rename_expr r body)
  = rename_expr r (rewrite_code_body (f S) body).
Proof. equivariance. Qed.

#[local] Hint Rewrite rewrite_code_body_stated_equivariant : equivariance.

Lemma rewrite_rec_handlers_stated_equivariant (r : renaming) (e : expr) :
  ∀ S : stated_in_scope,
    rewrite_rec_handlers_stated f (rename_stated_in_scope r S)
      (rename_expr r e)
    = rename_expr r (rewrite_rec_handlers_stated f S e).
Proof. equivariance. Qed.

End Rewrite_rec_handlers_stated.

(* ======================================================================== *)
(* The rewrite keeps scope                                                  *)
(* ======================================================================== *)

Section Rewrite_rec_handlers_stated_scope.

Variable f : stated_in_scope -> expr -> expr.
Hypothesis f_scoped :
  ∀ (S : stated_in_scope) (e : expr), scoped e (f S e).

Lemma rewrite_code_body_stated_scoped (S : stated_in_scope) (body : expr) :
  scoped body (rewrite_code_body (f S) body).
Proof.
  unfold rewrite_code_body. cbv zeta.
  destruct (Bool.eqb (binds_statics body) (binds_statics (f S body)));
    [ apply f_scoped | apply scoped_refl ].
Qed.

Fixpoint rewrite_rec_handlers_stated_scoped (e : expr) :
  ∀ S : stated_in_scope, scoped e (rewrite_rec_handlers_stated f S e)
with rewrite_rec_handlers_stated_in_handler_scoped
    (h : continuation_handler) :
  ∀ S : stated_in_scope,
    scoped_handler h (rewrite_rec_handlers_stated_in_handler f S h)
with rewrite_rec_handler_stated_scoped (h : continuation_handler) :
  ∀ S : stated_in_scope,
    scoped_handler h (rewrite_rec_handler_stated f S h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      intros S;
      cbn [rewrite_rec_handlers_stated rewrite_rec_handlers_stated_in_handler
           rewrite_rec_handler_stated].
    + apply scoped_let;
        [ apply incl_refl | apply rewrite_rec_handlers_stated_scoped ].
    + apply scoped_let_static;
        [ apply scoped_bindings_refl
        | apply rewrite_rec_handlers_stated_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_rec_handlers_stated_in_handler_scoped
        | apply rewrite_rec_handlers_stated_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_rec_handlers_stated_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_rec_handler_stated_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler]. intros S.
    cbn [rewrite_rec_handlers_stated rewrite_rec_handlers_stated_in_handler
         rewrite_rec_handler_stated].
    apply scoped_handler_intro, rewrite_rec_handlers_stated_scoped.
  - destruct h as [params metadata handler]. intros S.
    cbn [rewrite_rec_handlers_stated rewrite_rec_handlers_stated_in_handler
         rewrite_rec_handler_stated].
    apply scoped_handler_intro.
    eapply scoped_trans;
      [ apply rewrite_rec_handlers_stated_scoped
      | apply rewrite_code_body_stated_scoped ].
Qed.

End Rewrite_rec_handlers_stated_scope.
