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

(* Removing the unused [Let]s whose defining expression has a value, in
   the bodies of code, with the parameters of the code in scope: a
   normalisation that is an equivalence.

   This file is to be reviewed: [code_in_scope] and
   [remove_unused_defined_lets_in_code], which is extracted and run by
   the validation of middle_end/flambda2/tests/semantics/, and the
   statement.

   FlambdaLoopLetsSpec.remove_unused_defined_lets_in_loops removes a
   [Let] that nothing uses where its defining expression is sure to have
   a value of the kind of the [Let]: for a variable, where the variable
   is in scope at that kind. FlambdaNormaliseSpec.rewrite_code takes a
   rewrite into the body of code with no variable in scope, so a [Let]
   of a parameter of the code stays. The term after Simplify has such
   [Let]s. Where Simplify makes a loop of a function (loopify), every
   parameter of the function becomes a parameter of the loop; one that
   every self tail call passes on as it is, is then a parameter of the
   loop that every application gives the same thing, the parameter of
   the code: Simplify removes it from the loop and binds its name to the
   parameter of the code at the start of the handler (as
   FlambdaLoopLetsSpec.v describes), and the name is often not used
   again.

   [remove_unused_defined_lets_in_code] applies
   [remove_unused_defined_lets_in_loops] to the body of every piece of
   code that a term binds, wherever the binding is, as
   FlambdaNormaliseSpec.rewrite_code does, with the variables that a
   call binds in scope ([code_in_scope]): [my_closure] at the kind
   [Value], and the parameters at the kinds of the arity of the code,
   unarized.

   Why those kinds: the body of code runs at a call only
   (FlambdaSemanticsSpec.call_code), under [my_closure] bound to what is
   called, a closure, which is a value of kind [Value], and the
   parameters bound to the arguments; and a call is undefined unless
   the values of its arguments have the kinds of the arity of the code,
   unarized. The body of code that binds something static does not run
   at all; the pass does not change whether a body does.

   Renaming: the pass compares names by equality only and binds none; it
   commutes with a renaming as [remove_unused_defined_lets_in_loops]
   does (FlambdaValidatorEquivarianceProofs.v).

   The proofs are in FlambdaCodeLetsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaLoopLetsSpec.
Import ListNotations.

(* The variables in scope at the start of the body of code, each with
   the kind of the value that a call binds it to. *)
Definition code_in_scope (params : list variable) (my_closure : variable)
    (metadata : code_metadata) : kinds_in_scope :=
  params_in_scope params (unarize (cm_params_arity metadata))
    [(my_closure, Value)].

Definition remove_unused_defined_lets_in_binding (binding : static_binding)
    : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata
         (remove_unused_defined_lets_in_loops
            (code_in_scope params my_closure metadata) body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

Fixpoint remove_unused_defined_lets_in_code (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (remove_unused_defined_lets_in_code body)
  | Let_static bindings body =>
    Let_static (map remove_unused_defined_lets_in_binding bindings)
      (remove_unused_defined_lets_in_code body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (remove_unused_defined_lets_in_code_handler h)
      (remove_unused_defined_lets_in_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, remove_unused_defined_lets_in_code_handler h))
         handlers)
      (remove_unused_defined_lets_in_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with remove_unused_defined_lets_in_code_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (remove_unused_defined_lets_in_code handler)
  end.

Definition remove_unused_defined_lets_in_code_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_defined_lets_in_code e).
