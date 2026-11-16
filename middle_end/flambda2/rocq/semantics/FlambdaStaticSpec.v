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

(* "Let symbol": first facts about [Let_static].

   This file is to be reviewed: the statements, against the meaning given
   to [Let_static] in FlambdaSemanticsSpec.v ([eval_static_bindings]).

   - A group of bindings may be split in two, or two merged
     ([let_static_split_statement]). The static environment is global and
     a symbol is a pointer whether or not it is bound yet, so nothing
     requires a recursive group to be bound at once.
   - The projection of a value slot from a symbol bound to a closure is
     the value of the slot ([project_static_closure_statement]).
   - Calls of code bound by a [Let_static] are in FlambdaApplySpec.v
     ([inline_let_code_statement]).

   - The body of code bound by a [Let_static] may be replaced by one that
     refines it and binds something static if and only if it does
     ([let_static_code_refines_statement]): simplifying the body of a
     function is sound. It is the case of FlambdaContextSpec.v
     of a hole in code, stated on its own. The premise is a refinement
     under all environments, the static one included, so a rewrite of the
     body that relies on what other code or symbols are bound to is not
     covered yet.

   Not yet here: removing a binding that is not used, which needs the free
   symbols and code ids of terms, of code and of values.

   The proofs are in FlambdaStaticProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec.
Import ListNotations.

Definition let_static_split_statement : Prop :=
  ∀ (bindings bindings' : list static_binding) (body : expr),
    expr_equivalent (Let_static (bindings ++ bindings') body)
      (Let_static bindings (Let_static bindings' body)).

Definition project_static_closure_statement : Prop :=
  ∀ (s : symbol) (code : code_id) (slot : value_slot) (contents : simple)
    (x : variable) (kind : flambda_kind) (body : expr),
    expr_equivalent
      (Let_static
         [Bind_symbol s (Static_const_set_of_closures code [(slot, contents)])]
         (Let x kind (Project_value_slot slot (Symbol s)) body))
      (Let_static
         [Bind_symbol s (Static_const_set_of_closures code [(slot, contents)])]
         (Let x kind (Simple contents) body)).

Definition let_static_code_refines_statement : Prop :=
  ∀ (before : list static_binding) (id : code_id) (params : list variable)
    (my_closure : variable) (ret exn : continuation)
    (metadata : code_metadata) (code_body code_body' : expr)
    (after : list static_binding) (body : expr),
    binds_statics code_body = binds_statics code_body' ->
    expr_refines code_body code_body' ->
    expr_refines
      (Let_static
         (before
          ++ Bind_code id
               (Code params my_closure ret exn metadata code_body)
          :: after)
         body)
      (Let_static
         (before
          ++ Bind_code id
               (Code params my_closure ret exn metadata code_body')
          :: after)
         body).
