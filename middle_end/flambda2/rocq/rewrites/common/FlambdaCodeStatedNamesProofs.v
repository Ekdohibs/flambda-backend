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

(* FlambdaCodeStatedSpec.rewrite_code_stated commutes with renaming
   where the rewrite that it is given does, and keeps scope where that
   rewrite does: the lemmas from which
   FlambdaValidatorEquivarianceProofs.v and FlambdaValidateScopeProofs.v
   prove that the rounds of the validation commute with renaming and
   keep scope. There is no statement here, and nothing here needs
   review.

   The functions act on names only through the rewrite and through
   what the code states of its parameters, which is renamed with them
   (FlambdaStatedScopeNamesProofs.code_stated_equivariant). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNominalProofs
  FlambdaEquivarianceSpec FlambdaEquivarianceProofs FlambdaBindersSpec
  FlambdaNormaliseSpec FlambdaValidatorEquivarianceSpec FlambdaScopedProofs
  FlambdaStatedScopeSpec FlambdaStatedScopeNamesProofs
  FlambdaRecHandlersStatedNamesProofs FlambdaCodeStatedSpec
  FlambdaEquivarianceTactic.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite commutes with renaming                                       *)
(* ======================================================================== *)

#[local] Ltac eqv_database ::= eqv_database_by_one_pass.

#[local] Instance stated_in_scope_renaming : Renaming_of stated_in_scope :=
  rename_stated_in_scope.

Section Rewrite_code_stated.

Variable g : stated_in_scope -> expr -> expr.
Hypothesis g_equivariant :
  ∀ (r : renaming) (S : stated_in_scope) (e : expr),
    g (rename_stated_in_scope r S) (rename_expr r e) = rename_expr r (g S e).

#[local] Arguments rename_stated_in_scope : simpl never.

(* The body of code under what the code states, renamed. *)
Lemma rewrite_code_body_code_stated_equivariant (r : renaming)
    (params : list variable) (my_closure : variable)
    (metadata : code_metadata) (body : expr) :
  rewrite_code_body
    (g (code_stated (map (rename_variable r) params)
          (rename_variable r my_closure) metadata))
    (rename_expr r body)
  = rename_expr r
      (rewrite_code_body (g (code_stated params my_closure metadata)) body).
Proof.
  rewrite code_stated_equivariant.
  exact
    (rewrite_code_body_stated_equivariant g g_equivariant r
       (code_stated params my_closure metadata) body).
Qed.

#[local] Hint Rewrite rewrite_code_body_code_stated_equivariant
  : equivariance.

Lemma rewrite_code_stated_equivariant (r : renaming) (e : expr) :
  rewrite_code_stated g (rename_expr r e)
  = rename_expr r (rewrite_code_stated g e).
Proof. equivariance. Qed.

End Rewrite_code_stated.

(* ======================================================================== *)
(* The rewrite keeps scope                                                  *)
(* ======================================================================== *)

Section Rewrite_code_stated_scope.

Variable g : stated_in_scope -> expr -> expr.
Hypothesis g_scoped :
  ∀ (S : stated_in_scope) (e : expr), scoped e (g S e).

Lemma rewrite_code_stated_in_bindings_scoped
    (bindings : list static_binding) :
  scoped_bindings bindings (map (rewrite_code_stated_in_binding g) bindings).
Proof.
  induction bindings as [ | binding bindings IH]; cbn [map];
    [ apply scoped_bindings_nil | ].
  destruct binding
    as [id [params my_closure ret exn metadata body] | id | sym const];
    cbn [rewrite_code_stated_in_binding];
    [ apply scoped_bindings_cons_code;
      [ apply scoped_code_intro,
          (rewrite_code_body_stated_scoped g g_scoped)
      | exact IH ]
    | apply scoped_bindings_cons_same, IH
    | apply scoped_bindings_cons_same, IH ].
Qed.

Fixpoint rewrite_code_stated_scoped (e : expr) :
  scoped e (rewrite_code_stated g e)
with rewrite_code_stated_in_handler_scoped (h : continuation_handler) :
  scoped_handler h (rewrite_code_stated_in_handler g h).
Proof.
  - destruct e
      as [x kind d body | bindings body | k h body | handlers body | ap | ac
         | scrutinee arms | ];
      cbn [rewrite_code_stated rewrite_code_stated_in_handler].
    + apply scoped_let;
        [ apply incl_refl | apply rewrite_code_stated_scoped ].
    + apply scoped_let_static;
        [ apply rewrite_code_stated_in_bindings_scoped
        | apply rewrite_code_stated_scoped ].
    + apply scoped_let_cont;
        [ apply rewrite_code_stated_in_handler_scoped
        | apply rewrite_code_stated_scoped ].
    + apply scoped_let_cont_recursive;
        [ | | apply rewrite_code_stated_scoped ].
      * rewrite map_map. apply map_ext. intros [k h]. reflexivity.
      * induction handlers as [ | [k h] handlers IH]; cbn [map];
          [ apply scoped_handlers_nil | ].
        apply scoped_handlers_cons;
          [ apply rewrite_code_stated_in_handler_scoped | exact IH ].
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
    + apply scoped_refl.
  - destruct h as [params metadata handler].
    cbn [rewrite_code_stated rewrite_code_stated_in_handler].
    apply scoped_handler_intro, rewrite_code_stated_scoped.
Qed.

End Rewrite_code_stated_scope.
