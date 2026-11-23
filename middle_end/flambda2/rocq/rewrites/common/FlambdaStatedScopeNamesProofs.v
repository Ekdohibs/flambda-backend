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

(* FlambdaStatedScopeSpec.v under a renaming: the lemmas from which the
   laws of renaming of the rewrites that read what is known of the
   variables in scope are proved. There is no statement here, and
   nothing here needs review.

   What is known of the variables in scope
   (FlambdaStatedScopeSpec.stated_in_scope) is a list of pairs, whose
   type does not say how it is renamed: the action is
   [rename_stated_in_scope], which renames the variables as
   FlambdaNormaliseSpec.rename_kinds_in_scope does; what is known of a
   variable has no names in it. A file that gives these laws to the
   tactic of FlambdaEquivarianceTactic.v declares the action
   ([Renaming_of stated_in_scope]) and the laws that it wants, as this
   file does for itself. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaMapProofs FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNominalProofs FlambdaEquivarianceSpec FlambdaEquivarianceProofs
  FlambdaEquivarianceTactic FlambdaStatedScopeSpec.
Import ListNotations.

Definition rename_stated_in_scope (r : renaming) (S : stated_in_scope)
    : stated_in_scope :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) S.

#[local] Instance stated_in_scope_renaming : Renaming_of stated_in_scope :=
  rename_stated_in_scope.

(* A renamed pair before the renamed scope is the renaming of the pair
   before the scope, by computation. Where it is given to the tactic,
   computation is to be kept from undoing it
   ([Arguments rename_stated_in_scope : simpl never]). *)
Lemma stated_in_scope_cons_renamed (r : renaming) (x : variable)
    (ps : param_subkind) (S : stated_in_scope) :
  (rename_variable r x, ps) :: rename_stated_in_scope r S
  = rename_stated_in_scope r ((x, ps) :: S).
Proof. reflexivity. Qed.

Lemma stated_of_simple_equivariant (r : renaming) (S : stated_in_scope)
    (s : simple) :
  stated_of_simple (rename_stated_in_scope r S) (rename_simple r s)
  = stated_of_simple S s.
Proof. equivariance. Qed.

Lemma stated_of_named_equivariant (r : renaming) (S : stated_in_scope)
    (d : named) :
  stated_of_named (rename_stated_in_scope r S) (rename_named r d)
  = stated_of_named S d.
Proof. equivariance. Qed.

Lemma is_stated_equivariant (r : renaming) (S : stated_in_scope)
    (ps : param_subkind) (s : simple) :
  is_stated (rename_stated_in_scope r S) ps (rename_simple r s)
  = is_stated S ps s.
Proof. equivariance. Qed.

Lemma are_stated_equivariant (r : renaming) (S : stated_in_scope)
    (sks : list param_subkind) :
  ∀ args : list simple,
    are_stated (rename_stated_in_scope r S) (map (rename_simple r) args) sks
    = are_stated S args sks.
Proof. equivariance. Qed.

Lemma is_stated_at_equivariant (r : renaming) (S : stated_in_scope)
    (sks : list param_subkind) (i : nat) (s : simple) :
  is_stated_at (rename_stated_in_scope r S) sks i (rename_simple r s)
  = is_stated_at S sks i s.
Proof. equivariance. Qed.

(* The parameters of a binder are added one by one: by hand, as the
   law of FlambdaNormaliseSpec.params_in_scope is. *)
Lemma params_stated_equivariant (r : renaming) (params : list variable) :
  ∀ (sks : list param_subkind) (S : stated_in_scope),
    params_stated (map (rename_variable r) params) sks
      (rename_stated_in_scope r S)
    = rename_stated_in_scope r (params_stated params sks S).
Proof.
  induction params as [ | param params IH]; intros [ | ps sks] S; simpl;
    try reflexivity.
  - exact (IH [] ((param, any_value) :: S)).
  - exact (IH sks ((param, ps) :: S)).
Qed.

Lemma code_stated_equivariant (r : renaming) (params : list variable)
    (my_closure : variable) (metadata : code_metadata) :
  code_stated (map (rename_variable r) params) (rename_variable r my_closure)
    metadata
  = rename_stated_in_scope r (code_stated params my_closure metadata).
Proof.
  unfold code_stated.
  exact
    (params_stated_equivariant r params (cm_params_subkinds metadata)
       [(my_closure, not_null)]).
Qed.
