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

(* The binders of Flambda 2 and the freshness quantifier.

   This file is to be reviewed: the statements, against those of
   FlambdaCanonicalFreeNamesSpec.v and FlambdaAlphaSpec.v that they
   restate, and
   against the compiler's [Name_abstraction.pattern_match] and
   [pattern_match_pair] (nominal/name_abstraction.ml), which open a
   binder at a fresh name and two binders at one fresh name.

   FlambdaCanonicalFreeNamesSpec.v says of each binder that a renaming
   may be applied to the bound names and their scope if it fixes the other
   free names of the scope. That is the general form: any permutation,
   with the names it must not move said by a premise. The compiler uses
   one case of it, the swap of the bound name with a fresh one. Here
   that case is said with the quantifier of
   FlambdaFreshQuantifierSpec.v, and the rules of alpha-equivalence at
   a binder are given in the form in which two terms are compared.

   - A fresh name is not free: the free names of an expression are a
     list (FlambdaBindersSpec.v), and all but finitely many names are
     outside a list (FlambdaFreshQuantifierSpec.for_fresh_avoids_statement).
     This is what lets a premise "not free in the body" be discharged
     by the quantifier.
   - Opening a binder at a fresh name: the variable of a [Let], the
     continuation of a [Let_cont], the parameters of a handler
     ([freshen_let_fresh_statement] and the two after it). Each is the
     statement of FlambdaCanonicalFreeNamesSpec.v for a swap, with the
     premise that the new name is not free replaced by the quantifier.
     Nothing is gained in strength, and something is hidden: the
     statement of FlambdaCanonicalFreeNamesSpec.v says which names will
     do (those not free in the scope), and a rewrite that has a name in
     hand needs that.
   - Two binders compared at one fresh name. Two [Let]s with the same
     kind and the same defining expression are alpha-equivalent if
     their bodies are, once the two bound variables are exchanged for
     one name that is free in neither body ([alpha_let_at_statement]);
     so they are if that holds for a fresh name
     ([alpha_let_fresh_statement]). The same for the continuation of a
     [Let_cont] and for the parameters of a handler, a list of names
     that repeats none, the two handlers having the same metadata (the
     kinds of their parameters). The first form asks for one
     name and is the one to use when a proof has a fresh name in hand;
     the second is the first with the side conditions hidden. These are
     rules that [alpha_equivalent] obeys, not new constructors of it.
   - The comparison of FlambdaCompareSpec.v pairs the two bound names
     of a binder where the compiler's comparison opens both binders at
     one fresh name. On names the two do the same
     ([names_match_fresh_statement]).

   The code that an expression binds is closed in every statement, as
   in the files this one restates.

   Nothing is trusted here. The proofs are in
   FlambdaFreshBindersProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaCompareSpec FlambdaFreshQuantifierSpec.
Import ListNotations.

(* ======================================================================== *)
(* The swaps                                                                *)
(* ======================================================================== *)

(* The renaming that exchanges a variable with another, or a
   continuation; and the one that exchanges each name of a list with the
   name at its position in another. *)
Definition swap_variable (x fresh : variable) : renaming :=
  Renaming [(x, fresh)] [].

Definition swap_continuation (k fresh : continuation) : renaming :=
  Renaming [] [(k, fresh)].

Definition swap_variables (xs fresh : list variable) : renaming :=
  Renaming (combine xs fresh) [].

Definition swap_continuations (ks fresh : list continuation) : renaming :=
  Renaming [] (combine ks fresh).

(* ======================================================================== *)
(* Opening a binder at a fresh name                                         *)
(* ======================================================================== *)

(* FlambdaCanonicalFreeNamesSpec.freshen_let_canonical_statement. *)
Definition freshen_let_fresh_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    code_closed body = true ->
    И fresh,
      expr_equivalent (Let x kind defining_expr body)
        (Let fresh kind defining_expr
           (rename_expr (swap_variable x fresh) body)).

(* FlambdaCanonicalFreeNamesSpec.rename_let_cont_canonical_statement,
   for a swap. *)
Definition freshen_let_cont_fresh_statement : Prop :=
  ∀ (k : continuation) (handler : continuation_handler) (body : expr),
    code_closed body = true ->
    И fresh,
      expr_equivalent (Let_cont_non_recursive k handler body)
        (Let_cont_non_recursive fresh handler
           (rename_expr (swap_continuation k fresh) body)).

(* FlambdaCanonicalFreeNamesSpec.rename_handler_params_canonical_statement,
   for the swaps with as many fresh names. *)
Definition freshen_handler_params_fresh_statement : Prop :=
  ∀ (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    code_closed handler = true ->
    distinct params = true ->
    И[length params] fresh,
      expr_equivalent
        (Let_cont_non_recursive k (Handler params metadata handler) body)
        (Let_cont_non_recursive k
           (Handler fresh metadata
              (rename_expr (swap_variables params fresh) handler))
           body).

(* ======================================================================== *)
(* Two binders compared at one fresh name                                   *)
(* ======================================================================== *)

(* The variable of a [Let]. The two have the same kind and the same
   defining expression. *)
Definition alpha_let_at_statement : Prop :=
  ∀ (x x' fresh : variable) (kind : flambda_kind) (defining_expr : named)
    (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    among fresh (free_variables_of_expr body) = false ->
    among fresh (free_variables_of_expr body') = false ->
    alpha_equivalent (rename_expr (swap_variable x fresh) body)
      (rename_expr (swap_variable x' fresh) body') ->
    alpha_equivalent (Let x kind defining_expr body)
      (Let x' kind defining_expr body').

Definition alpha_let_fresh_statement : Prop :=
  ∀ (x x' : variable) (kind : flambda_kind) (defining_expr : named)
    (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    (И fresh,
      alpha_equivalent (rename_expr (swap_variable x fresh) body)
        (rename_expr (swap_variable x' fresh) body')) ->
    alpha_equivalent (Let x kind defining_expr body)
      (Let x' kind defining_expr body').

(* The continuation of a non-recursive [Let_cont]. *)
Definition alpha_let_cont_at_statement : Prop :=
  ∀ (k k' fresh : continuation) (handler : continuation_handler)
    (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    among fresh (free_continuations_of_expr body) = false ->
    among fresh (free_continuations_of_expr body') = false ->
    alpha_equivalent (rename_expr (swap_continuation k fresh) body)
      (rename_expr (swap_continuation k' fresh) body') ->
    alpha_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive k' handler body').

Definition alpha_let_cont_fresh_statement : Prop :=
  ∀ (k k' : continuation) (handler : continuation_handler)
    (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    (И fresh,
      alpha_equivalent (rename_expr (swap_continuation k fresh) body)
        (rename_expr (swap_continuation k' fresh) body')) ->
    alpha_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive k' handler body').

(* The parameters of a handler: lists that repeat no name, as the
   binders of a well-formed term are
   ([FlambdaBindersSpec.binders_distinct]). The two handlers have the
   same metadata, which is the kinds of their parameters. *)
Definition alpha_handler_at_statement : Prop :=
  ∀ (k : continuation) (params params' fresh : list variable)
    (metadata : handler_metadata) (handler handler' body : expr),
    code_closed handler = true ->
    code_closed handler' = true ->
    distinct params = true ->
    distinct params' = true ->
    length fresh = length params ->
    length fresh = length params' ->
    distinct_outside (params ++ free_variables_of_expr handler) fresh ->
    distinct_outside (params' ++ free_variables_of_expr handler') fresh ->
    alpha_equivalent (rename_expr (swap_variables params fresh) handler)
      (rename_expr (swap_variables params' fresh) handler') ->
    alpha_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params' metadata handler') body).

Definition alpha_handler_fresh_statement : Prop :=
  ∀ (k : continuation) (params params' : list variable)
    (metadata : handler_metadata) (handler handler' body : expr),
    code_closed handler = true ->
    code_closed handler' = true ->
    distinct params = true ->
    distinct params' = true ->
    length params = length params' ->
    (И[length params] fresh,
      alpha_equivalent (rename_expr (swap_variables params fresh) handler)
        (rename_expr (swap_variables params' fresh) handler')) ->
    alpha_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k (Handler params' metadata handler') body).

(* ======================================================================== *)
(* The pairs of the comparison                                              *)
(* ======================================================================== *)

(* To pair two names is to exchange each for one fresh name. *)
Definition names_match_fresh_statement : Prop :=
  ∀ (m : name_map) (x x' a a' : positive),
    И fresh,
      names_match ((x, x') :: m) a a'
      = names_match m (swap x fresh a) (swap x' fresh a').
