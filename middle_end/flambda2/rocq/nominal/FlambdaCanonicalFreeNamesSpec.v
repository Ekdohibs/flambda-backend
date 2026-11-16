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

(* The statements about binders: that the names a binder binds may be
   renamed, and that a binding that is not used may be removed.

   This file is to be reviewed: the statements.

   The free names of a term are given by one function for each sort
   (FlambdaBindersSpec.free_variables_of_expr and
   free_continuations_of_expr), which goes into code. Each statement
   here is written with those:
   - a renaming that fixes the free names of the scope of a binder,
     other than the names bound (FlambdaAlphaSpec.fixes_names), may be
     applied to the bound names and the scope
     ([rename_let_canonical_statement] and the five after it);
   - a binding whose name is not among the free names of its scope may
     be removed ([unused_let_canonical_statement] and the two after
     it). These three are corollaries of the statements of
     FlambdaNominalSpec.v, which say more;
   - a renaming of the variables of a term and of its environment that
     fixes its free continuations does not change its behaviour
     ([rename_variables_equivalent_canonical_statement]).
   The rules of FlambdaAlphaSpec.alpha_equivalent are the first of
   these, with [alpha_equivalent] for [expr_equivalent]. The binders of
   code are in FlambdaNominalSpec.v ([rename_code_binders_statement]).

   The statements that rename were in FlambdaNominalSpec.v and
   FlambdaAlphaSpec.v, written with the relations [var_free_in_expr]
   and [cont_free_in_expr] of FlambdaNominalSpec.v, which do not go
   into code. Those older forms are no longer statements of a Spec
   file. They are lemmas of FlambdaNominalProofs.v and
   FlambdaAlphaProofs.v, under the names they had ([rename_let] and so
   on, which is why the names here end in [_canonical]), because proofs
   use them. Each statement here is proved from the older one and from
   FlambdaFreeNamesSpec.var_free_among_statement and
   cont_free_among_statement: what is free by the relation is among the
   list, whatever the code. So the premise about names here asks at
   least as much as the older one did, and the two are the same where
   code is closed (FlambdaFreeNamesSpec.var_free_agrees_statement).

   What that means for each sort of statement:
   - those that rename ask that the code is closed, so they say what
     the older forms said;
   - the three about a binding that is not used ask nothing of the
     code, and here the premise asks more than that of
     FlambdaNominalSpec.unused_let_statement and the two after it,
     which are written with the relations and are still statements. Of
     a term whose code is not closed those say more: a name that occurs
     only in code is not free by the relation, and the binding may
     still be removed, since the machine never lets code read a name
     that it does not bind. The rewrites of the validator that remove
     bindings are proved from them. Of a well-formed term, whose code
     is closed, the two forms say the same.

   Nothing is trusted here. The proofs are in
   FlambdaCanonicalFreeNamesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaHeapSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaNameAbstractionSpec.
Import ListNotations.

(* ======================================================================== *)
(* The names that a binder binds may be renamed                             *)
(* ======================================================================== *)

(* The variable of a [Let]: a renaming that fixes the free names of the
   body other than the bound variable may be applied to the bound
   variable and the body. The code that the body binds is closed, here
   and in the statements below. *)
Definition rename_let_canonical_statement : Prop :=
  ∀ (r : renaming) (x : variable) (kind : flambda_kind)
    (defining_expr : named) (body : expr),
    code_closed body = true ->
    fixes_names r (without [x] (free_variables_of_expr body))
      (free_continuations_of_expr body) ->
    expr_equivalent (Let x kind defining_expr body)
      (Let (rename_variable r x) kind defining_expr (rename_expr r body)).

(* [Name_abstraction.pattern_match] on a [Let]: the bound variable
   becomes one that is not free in the body. *)
Definition freshen_let_canonical_statement : Prop :=
  ∀ (x fresh : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    code_closed body = true ->
    among fresh (free_variables_of_expr body) = false ->
    expr_equivalent (Let x kind defining_expr body)
      (Let fresh kind defining_expr
         (rename_expr (Renaming [(x, fresh)] []) body)).

(* The continuation of a non-recursive [Let_cont]. *)
Definition rename_let_cont_canonical_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (handler : continuation_handler)
    (body : expr),
    code_closed body = true ->
    fixes_names r (free_variables_of_expr body)
      (without [k] (free_continuations_of_expr body)) ->
    expr_equivalent (Let_cont_non_recursive k handler body)
      (Let_cont_non_recursive (rename_continuation r k) handler
         (rename_expr r body)).

(* The parameters of a continuation handler. The free variables of a
   handler are those of its body that are not its parameters. *)
Definition rename_handler_params_canonical_statement : Prop :=
  ∀ (r : renaming) (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler body : expr),
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    expr_equivalent
      (Let_cont_non_recursive k (Handler params metadata handler) body)
      (Let_cont_non_recursive k
         (rename_handler r (Handler params metadata handler)) body).

(* The parameters of one handler of a recursive [Let_cont]. *)
Definition rename_recursive_handler_params_canonical_statement : Prop :=
  ∀ (r : renaming) (before : list (continuation * continuation_handler))
    (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr)
    (after : list (continuation * continuation_handler)) (body : expr),
    code_closed handler = true ->
    fixes_names r
      (free_variables_of_handler (Handler params metadata handler))
      (free_continuations_of_expr handler) ->
    expr_equivalent
      (Let_cont_recursive
         (before ++ (k, Handler params metadata handler) :: after) body)
      (Let_cont_recursive
         (before
          ++ (k, rename_handler r (Handler params metadata handler)) :: after)
         body).

(* The continuations of a recursive [Let_cont], and the parameters of
   its handlers: a renaming that fixes the free names of the whole may
   be applied throughout. *)
Definition rename_let_cont_recursive_canonical_statement : Prop :=
  ∀ (r : renaming) (handlers : list (continuation * continuation_handler))
    (body : expr),
    code_closed (Let_cont_recursive handlers body) = true ->
    fixes_names r
      (free_variables_of_expr (Let_cont_recursive handlers body))
      (free_continuations_of_expr (Let_cont_recursive handlers body)) ->
    expr_equivalent (Let_cont_recursive handlers body)
      (rename_expr r (Let_cont_recursive handlers body)).

(* ======================================================================== *)
(* Bindings that are not used                                               *)
(* ======================================================================== *)

(* A [Let] whose variable is not among the free variables of its body
   may be removed, if its defining expression is not a primitive of the
   heap. This and the two statements below are corollaries of
   FlambdaNominalSpec.unused_let_statement and the two after it, which
   ask less: nothing is asked of the code, and those do not count a
   name that occurs only in code. *)
Definition unused_let_canonical_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
    (body : expr),
    named_changes_heap defining_expr = false ->
    among x (free_variables_of_expr body) = false ->
    expr_refines (Let x kind defining_expr body) body.

Definition unused_let_cont_canonical_statement : Prop :=
  ∀ (k : continuation) (handler : continuation_handler) (body : expr),
    among k (free_continuations_of_expr body) = false ->
    expr_equivalent (Let_cont_non_recursive k handler body) body.

Definition unused_let_cont_recursive_canonical_statement : Prop :=
  ∀ (handlers : list (continuation * continuation_handler)) (body : expr),
    handlers_bind_statics handlers = false ->
    (∀ k : continuation,
      among k (free_continuations_of_expr body) = true ->
      lookup k handlers = None) ->
    expr_equivalent (Let_cont_recursive handlers body) body.

(* ======================================================================== *)
(* Execution is equivariant in the variables                                *)
(* ======================================================================== *)

(* Renaming an expression and the environment of its variables together
   does not change its behaviour, for a renaming that fixes the free
   continuations, which are observed. The code of the expression is
   closed. *)
Definition rename_variables_equivalent_canonical_statement : Prop :=
  ∀ (r : renaming) (e : expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    code_closed e = true ->
    fixes (continuation_permutation r) (free_continuations_of_expr e) ->
    equivalent (State e vars conts statics hp rd)
      (State (rename_expr r e) (rename_env r vars) conts statics hp rd).
