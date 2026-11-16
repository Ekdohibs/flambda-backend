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

(* Following aliases into the handlers of recursive continuations: a
   normalisation that is an equivalence.

   This file is to be reviewed: [follow_aliases_in_loops], which is
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   FlambdaAliasesSpec.follow_aliases replaces a variable
   bound by a [Let] to a [Simple] by what it is bound to, in the body of
   the [Let] and in the handlers of the non-recursive continuations
   there; it leaves the handlers of recursive continuations as they are.
   FlambdaRecHandlersSpec.rewrite_rec_handlers takes it into such a
   handler, but with nothing known: an alias bound outside the loop is
   not followed inside it. Simplify follows it: the handler of a
   recursive continuation is simplified under the environment of its
   [Let_cont], to which the parameters are added at unknown types
   (simplify/simplify_let_cont_expr.ml,
   [simplify_single_recursive_handler]). The front end binds the bound
   of a [for] loop before the loop, so the term after has the constant
   in the test of the loop and the term before a variable.

   [follow_aliases_in_loops] goes through a term keeping what
   [follow_aliases] keeps (the typing environment of
   FlambdaAliasesSpec.typing_env_under_let), and changes only the
   handlers of the recursive continuations: each is given to
   [follow_aliases] with what is known where the [Let_cont] is, less
   what mentions the parameters of the handler
   (FlambdaAliasesSpec.drop_vars), after the loops inside it have been
   done in the same way. A handler is left alone if that would change
   whether it binds something static
   (FlambdaNormaliseSpec.rewrite_code_body); [follow_aliases] never
   does, but that is not needed here.

   Why this is right although the handler runs many times: what is
   known where the [Let_cont] is holds of the variables of its closure,
   which do not change; and the parameters, which do, are forgotten.

   The [Let]s are not changed here, so what is known under one is what
   it says as it stands; [follow_aliases] itself is to be run before,
   and again after.

   Renaming: the pass compares names by equality only and binds none; it
   commutes with a renaming where [follow_aliases] does. That statement
   is owed to the nominal work and is not made here.

   The proofs are in FlambdaLoopAliasesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec FlambdaAliasesSpec.
Import ListNotations.

Fixpoint follow_aliases_in_loops (tenv : typing_env) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d
      (follow_aliases_in_loops (typing_env_under_let tenv x d) body)
  | Let_static bindings body =>
    Let_static bindings (follow_aliases_in_loops tenv body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (follow_aliases_in_loops_in_handler tenv h)
      (follow_aliases_in_loops tenv body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, follow_aliases_in_loop tenv h)) handlers)
      (follow_aliases_in_loops tenv body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with follow_aliases_in_loops_in_handler (tenv : typing_env)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (follow_aliases_in_loops (drop_vars params tenv) handler)
  end
with follow_aliases_in_loop (tenv : typing_env) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (rewrite_code_body (follow_aliases (drop_vars params tenv))
         (follow_aliases_in_loops (drop_vars params tenv) handler))
  end.

Definition follow_aliases_in_loops_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (follow_aliases_in_loops [] e).
