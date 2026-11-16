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

(* Rewriting inside a term: refinement and equivalence of expressions are
   preserved by every context.

   This file is to be reviewed: [context] and [plug], against the terms of
   FlambdaModel.v (a context is a term with one hole, which may be under
   any binder and in the handler of a continuation), and the two
   statements. They are what lets a rewrite proved of two expressions on
   their own, such as those of FlambdaControlFlowSpec.v, be applied to a
   subterm of a program, the body of a function included.

   Where the hole is in the body of code or in the handler of a recursive
   continuation ([hole_runs_many_times]), nothing static may be bound, and
   the machine makes the call or the [Let_cont] undefined if something is
   (FlambdaSemanticsSpec.binds_statics). So there the two expressions
   must agree on whether they bind something static.

   The premise is a refinement under all environments ([expr_refines]). A
   rewrite that is sound only under what is known of the variables in
   scope, as most of the simplifier's are, is not covered: that needs a
   refinement relative to a property of the environments.

   The proofs are in FlambdaContextProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec.
Import ListNotations.

(* An expression with a hole: in the body of a [Let], of either sort, or of
   a [Let_cont], in a handler of a [Let_cont], or in the body of code
   bound by a [Let_static]. *)
Inductive context : Type :=
  | Hole
  | Let_body (x : variable) (kind : flambda_kind) (defining_expr : named)
      (body : context)
  | Let_static_body (bindings : list static_binding) (body : context)
  | Let_static_code (before : list static_binding) (id : code_id)
      (params : list variable) (my_closure : variable)
      (return_continuation exn_continuation : continuation)
      (metadata : code_metadata) (code_body : context)
      (after : list static_binding) (body : expr)
  | Let_cont_non_recursive_body (k : continuation)
      (handler : continuation_handler) (body : context)
  | Let_cont_non_recursive_handler (k : continuation)
      (params : list variable) (metadata : handler_metadata)
      (handler : context) (body : expr)
  | Let_cont_recursive_body
      (handlers : list (continuation * continuation_handler))
      (body : context)
  | Let_cont_recursive_handler
      (before : list (continuation * continuation_handler))
      (k : continuation) (params : list variable)
      (metadata : handler_metadata) (handler : context)
      (after : list (continuation * continuation_handler)) (body : expr).

Fixpoint plug (c : context) (e : expr) : expr :=
  match c with
  | Hole => e
  | Let_body x kind defining_expr body =>
    Let x kind defining_expr (plug body e)
  | Let_static_body bindings body => Let_static bindings (plug body e)
  | Let_static_code before id params my_closure return_continuation
      exn_continuation metadata code_body after body =>
    Let_static
      (before
       ++ Bind_code id
            (Code params my_closure return_continuation exn_continuation
               metadata (plug code_body e))
       :: after)
      body
  | Let_cont_non_recursive_body k handler body =>
    Let_cont_non_recursive k handler (plug body e)
  | Let_cont_non_recursive_handler k params metadata handler body =>
    Let_cont_non_recursive k (Handler params metadata (plug handler e))
      body
  | Let_cont_recursive_body handlers body =>
    Let_cont_recursive handlers (plug body e)
  | Let_cont_recursive_handler before k params metadata handler after body
    =>
    Let_cont_recursive
      (before ++ (k, Handler params metadata (plug handler e)) :: after)
      body
  end.

(* Whether the hole is in the body of code or in the handler of a
   recursive continuation. *)
Fixpoint hole_runs_many_times (c : context) : bool :=
  match c with
  | Hole => false
  | Let_body _ _ _ c | Let_static_body _ c
  | Let_cont_non_recursive_body _ _ c
  | Let_cont_non_recursive_handler _ _ _ c _ | Let_cont_recursive_body _ c =>
    hole_runs_many_times c
  | Let_static_code _ _ _ _ _ _ _ _ _ _
  | Let_cont_recursive_handler _ _ _ _ _ _ _ => true
  end.

Definition plug_refines_statement : Prop :=
  ∀ (c : context) (source target : expr),
    (hole_runs_many_times c = true ->
     binds_statics source = binds_statics target) ->
    expr_refines source target ->
    expr_refines (plug c source) (plug c target).

Definition plug_equivalent_statement : Prop :=
  ∀ (c : context) (e e' : expr),
    (hole_runs_many_times c = true -> binds_statics e = binds_statics e') ->
    expr_equivalent e e' -> expr_equivalent (plug c e) (plug c e').
