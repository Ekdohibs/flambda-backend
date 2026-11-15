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

(* The rewrites of the validation keep scope.

   This file is to be reviewed: the statements.

   A rewrite keeps scope (FlambdaValidatorEquivarianceSpec.keeps_scope)
   if, on a term whose code is closed, the code of its result is closed
   and every variable or continuation that is free in the result is free
   in the term; the free variables and continuations are those of the
   one function for each sort of FlambdaBindersSpec.v, which goes into
   code. FlambdaValidatorEquivarianceSpec.v says why it matters: a
   rewrite that makes names commutes with a renaming of the term only up
   to those names, and that is enough for the validation as long as the
   rewrites that follow it keep scope. The statements here say that a
   rewrite of the validation does, one statement to a rewrite: the
   comparison, the pass that gives every binder a name of its own, and
   the rewrites that a round of the validation is made of.

   What is not here: the inlining of the calls of known code and the
   rounds as wholes, which are in FlambdaValidateScopeSpec.v and
   FlambdaValidatorEquivarianceSpec.v; the rewrites for loops, in
   FlambdaLoopsScopeSpec.v; and the rewrites for lifting and for the
   static bindings, in FlambdaStaticsScopeSpec.v.

   The comparison up to bound names comes first
   ([expr_eqb_keeps_scope_statement]): of two terms that it accepts,
   if the code of the first is closed, the code of the second is, and
   every free variable or continuation of the second is free in the
   first. A normalisation that keeps its candidate only where the
   comparison accepts it (FlambdaRenameApartSpec.rename_apart_avoiding,
   and so FlambdaRenameApartSpec.rename_apart) keeps scope for that
   reason alone.

   Nothing is asked of a term but that its code is closed, which
   [keeps_scope] itself asks. Without it the comparison accepts two
   terms of which the second has a free variable that the first has
   not (FlambdaCompareAlphaSpec.expr_eqb_open_code_statement is such a
   pair).

   Nothing is trusted here. The proofs are in
   FlambdaKeepsScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaCompareSpec
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaCseSpec
  FlambdaMadeLengthsSpec
  FlambdaIdentitiesSpec FlambdaMergeSwitchesSpec FlambdaRenameApartSpec
  FlambdaStaticsOrderSpec
  FlambdaUnboxSpec FlambdaUnusedContsSpec FlambdaUnusedStaticsSpec
  FlambdaValidateSpec FlambdaValidatorEquivarianceSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The comparison                                                           *)
(* ======================================================================== *)

Definition expr_eqb_keeps_scope_statement : Prop :=
  ∀ e e' : expr,
    code_closed e = true ->
    expr_eqb [] [] e e' = true ->
    code_closed e' = true
    ∧ (∀ x : variable,
        among x (free_variables_of_expr e') = true ->
        among x (free_variables_of_expr e) = true)
    ∧ (∀ k : continuation,
        among k (free_continuations_of_expr e') = true ->
        among k (free_continuations_of_expr e) = true).

(* ======================================================================== *)
(* The rewrites of a round of the term before                               *)
(* ======================================================================== *)

(* Every binder given a name of its own, whatever the continuations
   that the new names are to avoid, and with none to avoid. *)
Definition rename_apart_avoiding_keeps_scope_statement : Prop :=
  ∀ avoid : list continuation, keeps_scope (rename_apart_avoiding avoid).

Definition rename_apart_keeps_scope_statement : Prop :=
  keeps_scope rename_apart.

(* The rewrites of a round for the body of code, which need to know
   nothing of where they are applied. *)
Definition fold_constants_keeps_scope_statement : Prop :=
  keeps_scope fold_constants.

Definition collapse_switches_keeps_scope_statement : Prop :=
  keeps_scope collapse_switches.

Definition merge_switches_keeps_scope_statement : Prop :=
  keeps_scope merge_switches.

Definition remove_unused_conts_keeps_scope_statement : Prop :=
  keeps_scope remove_unused_conts.

Definition remove_unused_lets_keeps_scope_statement : Prop :=
  keeps_scope remove_unused_lets.

(* Whatever the variables that are taken to be in scope, and the kinds
   they are taken to have. *)
Definition remove_unused_defined_lets_keeps_scope_statement : Prop :=
  ∀ kinds : kinds_in_scope, keeps_scope (remove_unused_defined_lets kinds).

Definition apply_identities_keeps_scope_statement : Prop :=
  keeps_scope apply_identities.

Definition inline_tail_conts_keeps_scope_statement : Prop :=
  keeps_scope inline_tail_conts.

Definition thread_switches_keeps_scope_statement : Prop :=
  keeps_scope thread_switches.

(* The rewrites that tidy the static bindings. *)
Definition drop_deleted_code_keeps_scope_statement : Prop :=
  keeps_scope drop_deleted_code.

Definition merge_statics_keeps_scope_statement : Prop :=
  keeps_scope merge_statics.

(* A rewrite that keeps scope, taken into the bodies of code. *)
Definition rewrite_code_keeps_scope_statement : Prop :=
  ∀ f : expr -> expr, keeps_scope f -> keeps_scope (rewrite_code f).

(* The rewrites that go by what is known of the variables and the
   symbols, as the validation runs them: with nothing known to begin
   with. *)
Definition load_known_blocks_keeps_scope_statement : Prop :=
  keeps_scope load_known_blocks.

Definition unbox_known_numbers_keeps_scope_statement : Prop :=
  keeps_scope unbox_known_numbers.

Definition share_known_primitives_keeps_scope_statement : Prop :=
  keeps_scope share_known_primitives.

Definition fold_made_lengths_keeps_scope_statement : Prop :=
  keeps_scope fold_made_lengths.

Definition follow_aliases_keeps_scope_statement : Prop :=
  keeps_scope (follow_aliases []).
