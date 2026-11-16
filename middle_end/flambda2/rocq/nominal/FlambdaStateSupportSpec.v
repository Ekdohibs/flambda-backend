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

(* Every expression and every state has a finite support: with the
   action of FlambdaEquivarianceSpec.v, the states of the machine are a
   nominal set.

   This file is to be reviewed: the three statements.

   A nominal set is a set with an action of the permutations of names in
   which every element has a finite support: finitely many names such
   that a permutation that fixes them fixes the element.
   FlambdaAlphaSpec.v and FlambdaEquivarianceSpec.v have the action of
   the renamings on expressions and on states, that it is an action
   (identity, composition, inverses), and that a step, a run and a
   behaviour commute with it. Here is the support, of an expression and
   of a state as they are written (the equality is [=]): there are
   finitely many variables and continuations such that a renaming that
   fixes them leaves the expression, or the state, as it is. They are
   the names that occur in it, those it binds included; the statements
   say only that there are such names.

   So the expressions and the states are nominal sets, and [step], [run]
   and [behaves] are equivariant functions and relations between them.

   The machine of FlambdaTrapSpec.v runs a state under a stack of
   exception handlers and of calls, and commutes with the action too
   (FlambdaTrapSpec.stack_step_equivariant_statement). A state with a
   stack has a finite support as well
   ([stack_configuration_finitely_supported_statement]), so what that
   machine runs is a nominal set.

   A state is not made an instance of
   FlambdaNameAbstractionSpec.contains_names, and the equivariance of
   execution stays as it is stated in FlambdaEquivarianceSpec.v, for
   three reasons. The interface is for one sort of name at a time, and
   the machine is renamed in both at once. The interface asks for the
   free names and for an equality under which a permutation that fixes
   them changes nothing; with [=] those names are all the names that
   occur, which is the statement here and is not what [free_names]
   means for a term. And with the equality that [free_names] is meant
   for, the same up to bound names, a state would need that equality
   of every handler in its continuations and of the code in its static
   environment, and, as for expressions, only the states whose code is
   closed would obey the laws.

   Nothing is trusted here. The proofs are in
   FlambdaStateSupportProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNameAbstractionSpec
  FlambdaEquivarianceSpec FlambdaTrapSpec.
Import ListNotations.

Definition expr_finitely_supported_statement : Prop :=
  ∀ e : expr,
    ∃ (vs : list variable) (ks : list continuation),
      ∀ r : renaming,
        fixes (variable_permutation r) vs ->
        fixes (continuation_permutation r) ks ->
        rename_expr r e = e.

Definition state_finitely_supported_statement : Prop :=
  ∀ s : state,
    ∃ (vs : list variable) (ks : list continuation),
      ∀ r : renaming,
        fixes (variable_permutation r) vs ->
        fixes (continuation_permutation r) ks ->
        rename_state r s = s.

(* A state and a stack, which are what the machine of FlambdaTrapSpec.v
   runs, are supported together by finitely many names. *)
Definition stack_configuration_finitely_supported_statement : Prop :=
  ∀ (s : state) (st : stack),
    ∃ (vs : list variable) (ks : list continuation),
      ∀ r : renaming,
        fixes (variable_permutation r) vs ->
        fixes (continuation_permutation r) ks ->
        rename_state r s = s ∧ rename_stack r st = st.
