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

(* The handler of a continuation put where the continuation is applied,
   at the end of its scope, as an equivalence.

   This file is to be reviewed: [inline_tail_conts_kinded] and the
   statement.

   FlambdaNormaliseSpec.inline_tail_conts puts the handler of a
   non-recursive continuation in place of the application of the
   continuation, where that application ends a run of [Let]s that is the
   body of its [Let_cont], with the parameters bound by [Let]s to the
   arguments. It is stated there to refine the term, which is what the
   normalisation of the term before Simplify needs
   ([inline_tail_conts_refines_statement]).

   The normalisation of the term after Simplify needs an equivalence,
   and that pass is not one. The application has a step only where the
   values of the arguments are of the kinds that the handler gives its
   parameters (FlambdaSemanticsSpec.apply_cont_value); the [Let]s test
   each value against the kind that FlambdaApplySpec.let_params gives
   it, which is the kind of the handler at that position, or [Value]
   where the handler gives too few. So where a handler gives fewer or
   more kinds than it has parameters, the application is undefined and
   the [Let]s may run.

   [inline_tail_conts_kinded] is the same pass where the handler gives
   one kind for each parameter, and leaves the [Let_cont] alone
   otherwise. It is an equivalence
   ([inline_tail_conts_kinded_equivalent_statement]), under all
   environments: the application binds the parameters to the values of
   the arguments and runs the handler, and the [Let]s do the same, one
   after the other, with the same tests of kinds. Where an argument has
   no value, or a value is not of its kind, both are undefined: the
   application at once, the [Let]s when they reach it.
   [inline_cont_at_tail] asks that there are as many arguments as
   parameters, so the application is not undefined for their number
   where the [Let]s would run.

   The compiler gives every parameter a kind, so on its terms the two
   passes do the same.

   Renaming: the pass compares names by equality only and binds none
   that the term has not got, as [inline_tail_conts]. That statement is
   owed to the nominal work and is not made here.

   The proofs are in FlambdaInlineTailProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec.
Import ListNotations.

Fixpoint inline_tail_conts_kinded (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (inline_tail_conts_kinded body)
  | Let_static bindings body =>
    Let_static bindings (inline_tail_conts_kinded body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler' := inline_tail_conts_kinded handler in
    let body' := inline_tail_conts_kinded body in
    let kept :=
      Let_cont_non_recursive k (Handler params metadata handler') body'
    in
    if Nat.eqb (length params) (length (hm_params_kinds metadata))
    then
      match inline_cont_at_tail k params metadata handler' body' with
      | Some inlined => inlined
      | None => kept
      end
    else kept
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (inline_tail_conts_kinded body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition inline_tail_conts_kinded_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (inline_tail_conts_kinded e).
