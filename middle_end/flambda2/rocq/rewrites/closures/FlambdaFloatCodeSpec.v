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

(* The bindings of code moved up to the head of what they are in: a
   normalisation that is an equivalence.

   This file is to be reviewed: [float_code], which is extracted and run
   by the validation of middle_end/flambda2/tests/semantics/, and the
   statements.

   The two terms of a unit do not bind code in the same places. The
   conversion from Lambda binds all the code of a unit first, around
   the rest (from_lambda/closure_conversion.ml,
   [bind_static_consts_and_code], which folds over [Acc.code];
   [wrap_final_module_block] calls it on the body of the unit).
   Simplify binds the new code of a set of closures with the constants
   that it lifts where it simplifies the set, and places those around
   the [Let] that they come from, once that is at the toplevel of the
   unit (simplify/simplify_let_expr.ml, [rebuild_let];
   simplify/expr_builder.ml, [place_lifted_constants]): so after the
   [Let]s that come before that one. A unit whose exported function
   uses a value computed when the unit runs has, after Simplify, the
   code of the function bound after the [Let] of that value.

   [float_code] moves the bindings of code up. A group of static
   bindings that follows a [Let], or that is at the head of the body of
   a [Let_cont], has its bindings of code taken out and bound before
   the [Let] or the [Let_cont] ([float_over]); groups that then follow
   one another are made one, the code first, as
   FlambdaNormaliseSpec.merge_statics makes them. So code goes up to
   the head of the unit, or of the handler that it is in. It does not
   leave a handler: the handler may never run, or run after a call of
   the code. The bodies of code and the handlers of recursive
   continuations are left as they are. Bindings of deleted code, which
   bind nothing, go with the code.

   It is an equivalence under all environments
   ([float_code_equivalent_statement]), so it may be applied to both
   terms: binding code reads no variable and no continuation, a [Let]
   or a [Let_cont] does not read the code that is bound, and where one
   of the two is undefined the term is undefined in either order.

   The proofs are in FlambdaFloatCodeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNormaliseSpec.
Import ListNotations.

(* [inner] under the binder [wrap], with the code that a group of
   bindings at the head of [inner] binds taken out and bound before the
   binder. *)
Definition float_over (wrap : expr -> expr) (inner : expr) : expr :=
  match inner with
  | Let_static bindings body =>
    let_static_if_any (filter binds_code bindings)
      (wrap
         (let_static_if_any
            (filter (fun binding => negb (binds_code binding)) bindings)
            body))
  | _ => wrap inner
  end.

Fixpoint float_code (e : expr) : expr :=
  match e with
  | Let x kind d body => float_over (Let x kind d) (float_code body)
  | Let_static bindings body =>
    match float_code body with
    | Let_static bindings' body' =>
      Let_static (code_first (bindings ++ bindings')) body'
    | body' => Let_static (code_first bindings) body'
    end
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    float_over
      (Let_cont_non_recursive k
         (Handler params metadata (float_code handler)))
      (float_code body)
  | Let_cont_recursive handlers body =>
    float_over (Let_cont_recursive handlers) (float_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition float_code_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (float_code e).

(* The law that it rests on, for a [Let]: a binding of code and a [Let]
   may change places. *)
Definition float_code_over_let_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (d : named)
    (codes : list static_binding) (body : expr),
    forallb binds_code codes = true ->
    expr_equivalent (Let x kind d (Let_static codes body))
      (Let_static codes (Let x kind d body)).

(* It commutes with renaming. *)
Definition float_code_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (float_code e) = float_code (rename_expr r e).
