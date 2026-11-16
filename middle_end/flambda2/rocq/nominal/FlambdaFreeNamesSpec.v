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

(* The older definitions of free names and of scope say the same as the
   functions of FlambdaBindersSpec.v, on a term whose code is closed.

   This file is to be reviewed: the six statements.

   The model has four older accounts of which names a term uses:
   - the relations [var_free_in_expr] and [cont_free_in_expr] of
     FlambdaNominalSpec.v, in which the statements about binders and
     about alpha-equivalence were written, and in which the three
     statements about a binding that is not used,
     FlambdaApplySpec.inlinable and many proofs still are;
   - [free_continuations] of FlambdaWellFormedSpec.v, by which a handler
     is found to be at toplevel;
   - [expr_in_scope] of FlambdaNormaliseSpec.v, the check of scope that
     the test of inlining uses.
   None of them looks into code: each treats a binding of code as
   mentioning no variable and no continuation. The functions of
   FlambdaBindersSpec.v do look into code. So the two sides can differ
   only on a term that binds code which is not closed (they need not:
   a binder around the code may bind what the code leaves free), and
   they agree where [code_closed] holds, which is every well-formed
   term. On every term, a name that is free by a relation is among the
   free names given by the function ([var_free_among_statement],
   [cont_free_among_statement]): the functions look at more.

   These statements are what lets the older definitions be restated as
   the one function, a statement at a time, without changing what any
   of them says of a well-formed term. The statements that a bound
   name may be renamed and the rules of alpha-equivalence have been
   (FlambdaCanonicalFreeNamesSpec.v, FlambdaAlphaSpec.v), and their
   older forms are lemmas, proved as they were. When the two checks
   and [inlinable] have been too, and no proof needs the relations
   but those of the three statements about a binding that is not used,
   which keep them, this file is to say that alone.

   The proofs are in FlambdaFreeNamesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaNominalSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaNormaliseSpec.
Import ListNotations.

(* A variable is free by the relation exactly when it is among the free
   variables. *)
Definition var_free_agrees_statement : Prop :=
  ∀ (x : variable) (e : expr),
    code_closed e = true ->
    (var_free_in_expr x e <-> among x (free_variables_of_expr e) = true).

Definition cont_free_agrees_statement : Prop :=
  ∀ (k : continuation) (e : expr),
    code_closed e = true ->
    (cont_free_in_expr k e
     <-> among k (free_continuations_of_expr e) = true).

(* Whatever the code, a name that is free by the relation is among the
   free names: the function looks into code and the relation does not.
   A statement whose premise is that a name is not free may so be
   restated by the function with no premise on code. The restated
   statement is not the same statement: where code is not closed its
   premise asks more (a name that only code mentions is among the free
   names and is not free by the relation). Where code is closed the two
   premises are the same. *)
Definition var_free_among_statement : Prop :=
  ∀ (x : variable) (e : expr),
    var_free_in_expr x e -> among x (free_variables_of_expr e) = true.

Definition cont_free_among_statement : Prop :=
  ∀ (k : continuation) (e : expr),
    cont_free_in_expr k e ->
    among k (free_continuations_of_expr e) = true.

(* The two functions for the free continuations give the same list. *)
Definition free_continuations_agree_statement : Prop :=
  ∀ e : expr,
    code_closed e = true ->
    free_continuations_of_expr e = free_continuations e.

(* The check of scope is that the free names are among those given. *)
Definition expr_in_scope_agrees_statement : Prop :=
  ∀ (vs : list variable) (ks : list continuation) (e : expr),
    code_closed e = true ->
    expr_in_scope vs ks e = names_among vs ks e.
