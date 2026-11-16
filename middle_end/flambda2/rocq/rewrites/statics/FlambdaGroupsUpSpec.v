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


(* A group of static bindings under a [Let_cont] moved above it: a
   normalisation that is an equivalence.

   This file is to be reviewed: [groups_up], which is extracted and run
   by the validation of middle_end/flambda2/tests/semantics/, and the
   statements.

   The two terms of a unit do not bind a symbol in the same place among
   the [Let_cont]s of a test. Simplify binds a block that it lifts out
   of the arm of a test under the last [Let] before the test, below the
   [Let_cont]s of the arms; a lifting of the validation that binds it
   before the [Let_cont] of the arm, which is not in this tree yet,
   has it above. The handler of a [Let_cont] is kept with
   the variables and the continuations that are bound where it stands,
   and not with the static environment: it sees what is bound when it
   runs. So a group of static bindings at the head of the body of a
   [Let_cont], recursive or not, may be bound before the [Let_cont]
   instead.

   [groups_up] does that everywhere outside the bodies of code and the
   handlers of recursive continuations, and makes one group of groups
   that then follow one another, the code first, as
   FlambdaNormaliseSpec.merge_statics makes them. A group does not go
   above a [Let]: the [Let] may bind a variable that it mentions. It
   does not leave a handler.

   It is an equivalence under all environments
   ([groups_up_equivalent_statement]), so it may be applied to both
   terms: binding a group reads no continuation, and a [Let_cont] reads
   nothing; where the group is undefined the term is undefined in
   either order (FlambdaFloatCodeProofs.v has the two laws, for code,
   stated for any group).

   The proofs are in FlambdaGroupsUpProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNormaliseSpec.
Import ListNotations.

(* [inner] under the binder [wrap], with the group of bindings at the
   head of [inner], if there is one, bound before the binder. *)
Definition group_over (wrap : expr -> expr) (inner : expr) : expr :=
  match inner with
  | Let_static bindings body => Let_static bindings (wrap body)
  | _ => wrap inner
  end.

Fixpoint groups_up (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (groups_up body)
  | Let_static bindings body =>
    match groups_up body with
    | Let_static bindings' body' =>
      Let_static (code_first (bindings ++ bindings')) body'
    | body' => Let_static (code_first bindings) body'
    end
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    group_over
      (Let_cont_non_recursive k
         (Handler params metadata (groups_up handler)))
      (groups_up body)
  | Let_cont_recursive handlers body =>
    group_over (Let_cont_recursive handlers) (groups_up body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition groups_up_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (groups_up e).

(* It commutes with renaming. *)
Definition groups_up_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (groups_up e) = groups_up (rename_expr r e).
