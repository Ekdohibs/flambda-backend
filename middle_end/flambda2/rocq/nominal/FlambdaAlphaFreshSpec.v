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

(* Alpha-equivalence at each binder, by the freshness quantifier.

   This file is to be reviewed: [swap_binders], [alpha_handlers],
   [alpha_code], [alpha_binding] and the statements; and against the
   compiler's [pattern_match_pair] of [Non_recursive_let_cont_handler],
   [Continuation_handler] and [Function_params_and_body]
   (terms/flambda.ml), by which compare/compare.ml compares two
   binders: each opens the two at the same fresh names.

   FlambdaFreshBindersSpec.v gives the rule by which two binders are
   compared: they are alpha-equivalent if their scopes are, once the
   bound names of both are exchanged for the same fresh names.
   FlambdaCompareConverseSpec.v says that for [Let] the rule can be
   read both ways ([alpha_let_iff_statement]). Here the same is said of
   the other binders that are not recursive:
   - the continuation of a non-recursive [Let_cont], whose scope is the
     body ([alpha_let_cont_body_iff_statement]);
   - the parameters of its handler, whose scope is the handler
     ([alpha_handler_iff_statement]);
   - the two together ([alpha_let_cont_iff_statement]);
   - the binders of code, which are [my_closure], the parameters and
     the return and exception continuations, and whose scope is the
     body of the code ([alpha_code_iff_statement]).
   A [Let_static] binds no name that a renaming moves. Two are
   alpha-equivalent exactly when their bindings are, one by one, and
   their bodies are ([alpha_let_static_iff_statement]).

   So at each of these binders, of the terms said below,
   alpha-equivalence is the equality of name abstractions of
   FlambdaFreshAbstractionSpec.v: the scopes agree at fresh names.
   [alpha_handlers], [alpha_code] and [alpha_binding] are that test of
   agreement, and are defined for all terms; they say that two binders
   are the same name abstraction only of the terms said below. The
   continuations of a recursive [Let_cont], whose scope is the
   handlers as well as the body, are in FlambdaAlphaGroupSpec.v.

   Which terms. Every statement is for terms whose code is closed and
   whose binders repeat no name: the two conditions of well-formedness
   (FlambdaBindersSpec.v, [code_closed] and [binders_distinct]) for
   which the converse of the comparison is stated
   (FlambdaCompareConverseSpec.v). The two directions do not use them
   alike.
   - From right to left, which is the rule of FlambdaFreshBindersSpec.v,
     the proof uses that the code is closed, as every rule of
     [alpha_equivalent] that renames does, and that the lists of bound
     names that are compared repeat no name. It does not use that the
     binders within the scopes are distinct.
   - From left to right the proof uses both conditions on both terms.
     It goes through the comparison (FlambdaCompareSpec.v), which
     accepts alpha-equivalent terms: it accepts the scopes under the
     pairs of the bound names, so it accepts them with no names paired
     once the bound names of both are exchanged for fresh ones, and of
     terms that have the two conditions what it accepts is
     alpha-equivalent.
   Of terms outside the two conditions one thing is said: the premise
   on the lists cannot be left out. Two handlers of which the first
   repeats a parameter and the second does not can agree at fresh
   names, and the two [Let_cont]s that have them are not
   alpha-equivalent ([alpha_handlers_repeated_parameter_statement]):
   there the right hand side of [alpha_handler_iff_statement] holds and
   the left does not.

   Nothing is trusted here. The proofs are in
   FlambdaAlphaFreshProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaBindersSpec
  FlambdaNominalSpec FlambdaAlphaSpec FlambdaFreshQuantifierSpec
  FlambdaFreshBindersSpec.
Import ListNotations.

(* ======================================================================== *)
(* Two scopes compared at fresh names                                       *)
(* ======================================================================== *)

(* The renaming that exchanges each variable of a list with the variable
   at its position in another, and each continuation of a list with the
   continuation at its position in another. With one of the two pairs
   of lists empty it is [swap_variables] or [swap_continuations] of
   FlambdaFreshBindersSpec.v. *)
Definition swap_binders (xs fresh : list variable)
    (ks fresh_continuations : list continuation) : renaming :=
  Renaming (combine xs fresh) (combine ks fresh_continuations).

(* The test of two handlers opened at fresh variables: they have as
   many parameters and the same metadata (the kinds of the parameters),
   and the handlers agree once the parameters of both are exchanged for
   the same fresh variables. That two handlers which pass it are the
   same name abstraction is [alpha_handler_iff_statement], for handlers
   whose code is closed and whose binders repeat no name; it is not so
   of all handlers ([alpha_handlers_repeated_parameter_statement]). *)
Definition alpha_handlers (h h' : continuation_handler) : Prop :=
  match h, h' with
  | Handler params metadata handler, Handler params' metadata' handler' =>
    length params = length params'
    ∧ metadata = metadata'
    ∧ И[length params] fresh,
        alpha_equivalent (rename_expr (swap_variables params fresh) handler)
          (rename_expr (swap_variables params' fresh) handler')
  end.

(* The same test of two pieces of code: as many parameters, the same
   metadata (the arities of the parameters and of the results), and the
   bodies agree once [my_closure] and the parameters of both are
   exchanged for the same fresh variables, and the return and exception
   continuations of both for the same two fresh continuations.
   Variables and continuations are two sorts of names, so nothing is
   asked of a fresh variable and a fresh continuation together. That
   two pieces of code which pass it are the same name abstraction is
   [alpha_code_iff_statement], for code that is closed and whose
   binders repeat no name. *)
Definition alpha_code (c c' : code) : Prop :=
  match c, c' with
  | Code params my_closure return_continuation exn_continuation metadata
      body,
    Code params' my_closure' return_continuation' exn_continuation' metadata'
      body' =>
    length params = length params'
    ∧ metadata = metadata'
    ∧ И[S (length params)] fresh,
        И[2] fresh_continuations,
          alpha_equivalent
            (rename_expr
               (swap_binders (my_closure :: params) fresh
                  [return_continuation; exn_continuation]
                  fresh_continuations)
               body)
            (rename_expr
               (swap_binders (my_closure' :: params') fresh
                  [return_continuation'; exn_continuation']
                  fresh_continuations)
               body')
  end.

(* The test of two static bindings: the same code id with code that
   passes the test above, the same deleted code id, or the same symbol
   with the same constant. *)
Definition alpha_binding (binding binding' : static_binding) : Prop :=
  match binding, binding' with
  | Bind_code id c, Bind_code id' c' => id = id' ∧ alpha_code c c'
  | Bind_deleted_code id, Bind_deleted_code id' => id = id'
  | Bind_symbol s const, Bind_symbol s' const' => s = s' ∧ const = const'
  | _, _ => False
  end.

(* ======================================================================== *)
(* A non-recursive [Let_cont]                                               *)
(* ======================================================================== *)

(* The continuation is a name abstraction over the body. *)
Definition alpha_let_cont_body_iff_statement : Prop :=
  ∀ (k k' : continuation) (h : continuation_handler) (body body' : expr),
    code_closed body = true ->
    code_closed body' = true ->
    binders_distinct body = true ->
    binders_distinct body' = true ->
    (alpha_equivalent (Let_cont_non_recursive k h body)
       (Let_cont_non_recursive k' h body')
     <-> И fresh,
           alpha_equivalent (rename_expr (swap_continuation k fresh) body)
             (rename_expr (swap_continuation k' fresh) body')).

(* The parameters are a name abstraction over the handler. *)
Definition alpha_handler_iff_statement : Prop :=
  ∀ (k : continuation) (h h' : continuation_handler) (body : expr),
    handler_code_closed h = true ->
    handler_code_closed h' = true ->
    handler_binders_distinct h = true ->
    handler_binders_distinct h' = true ->
    (alpha_equivalent (Let_cont_non_recursive k h body)
       (Let_cont_non_recursive k h' body)
     <-> alpha_handlers h h').

(* Where a list repeats a name, the swaps send the name to the last of
   its fresh names; so a handler that repeats a parameter agrees at
   fresh names with one that does not, and uses its last parameter. The
   two are not alpha-equivalent where they stand in a [Let_cont], since
   alpha-equivalent terms have their binders distinct together
   (FlambdaAlphaSpec.alpha_equivalent_binders_distinct_statement). *)
Definition alpha_handlers_repeated_parameter_statement : Prop :=
  ∃ h h' : continuation_handler,
    alpha_handlers h h'
    ∧ handler_binders_distinct h = false
    ∧ handler_binders_distinct h' = true
    ∧ ∀ (k : continuation) (body : expr),
        binders_distinct body = true ->
        ~ alpha_equivalent (Let_cont_non_recursive k h body)
            (Let_cont_non_recursive k h' body).

(* The two together. *)
Definition alpha_let_cont_iff_statement : Prop :=
  ∀ (k k' : continuation) (h h' : continuation_handler) (body body' : expr),
    code_closed (Let_cont_non_recursive k h body) = true ->
    code_closed (Let_cont_non_recursive k' h' body') = true ->
    binders_distinct (Let_cont_non_recursive k h body) = true ->
    binders_distinct (Let_cont_non_recursive k' h' body') = true ->
    (alpha_equivalent (Let_cont_non_recursive k h body)
       (Let_cont_non_recursive k' h' body')
     <-> alpha_handlers h h'
         ∧ И fresh,
             alpha_equivalent (rename_expr (swap_continuation k fresh) body)
               (rename_expr (swap_continuation k' fresh) body')).

(* ======================================================================== *)
(* Code and static bindings                                                 *)
(* ======================================================================== *)

(* Code is a name abstraction of its binders over its body, wherever it
   is bound. *)
Definition alpha_code_iff_statement : Prop :=
  ∀ (before : list static_binding) (id : code_id) (c c' : code)
    (after : list static_binding) (body : expr),
    code_is_closed c = true ->
    code_is_closed c' = true ->
    code_binders_distinct c = true ->
    code_binders_distinct c' = true ->
    (alpha_equivalent (Let_static (before ++ Bind_code id c :: after) body)
       (Let_static (before ++ Bind_code id c' :: after) body)
     <-> alpha_code c c').

Definition alpha_let_static_iff_statement : Prop :=
  ∀ (bindings bindings' : list static_binding) (body body' : expr),
    code_closed (Let_static bindings body) = true ->
    code_closed (Let_static bindings' body') = true ->
    binders_distinct (Let_static bindings body) = true ->
    binders_distinct (Let_static bindings' body') = true ->
    (alpha_equivalent (Let_static bindings body)
       (Let_static bindings' body')
     <-> Forall2 alpha_binding bindings bindings'
         ∧ alpha_equivalent body body').
