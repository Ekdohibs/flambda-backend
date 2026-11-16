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

(* The bindings of a group of static bindings in another order: a
   normalisation that is an equivalence.

   This file is to be reviewed: [order_statics_as] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   Why it is needed. Simplify builds the group of static bindings of a
   set of closures again, and the code of the functions comes out in an
   order of its own: for

     let rec even n = ... odd (n - 1) and odd n = ... even (n - 1)

   the term before binds the code of [even] and then that of [odd], and
   the term after the other way round. The comparison of the validation
   (FlambdaCompareSpec.expr_eqb) takes the bindings of a group one by
   one, in order, and says that the two terms differ.

   The order of the bindings of a group does not matter. A group binds
   each of its names once: binding one twice is undefined, whichever
   comes first. A constant is evaluated under the variables, and not
   under what the group has bound so far
   (FlambdaSemanticsSpec.eval_static_bindings). And the machine only
   looks names up in the static environment, so two static environments
   in which every code id and every symbol is bound to the same thing
   are the same to it, whatever the order in which they were bound.

   [order_statics_as pattern e] goes through [e] and [pattern] together.
   Where both have a group it moves to the front of the group of [e],
   one at a time and the last first, the binding of each name that the
   group of [pattern] binds; so, where the group of [pattern] binds no
   name twice, the group of [e] begins with those bindings in the order
   of [pattern], and what [pattern] does not bind follows in the order
   it had. Where the two terms are not of the same shape, [e] is left as
   it is. Handlers of recursive continuations and code are not gone
   into: a group there makes the term undefined
   (FlambdaSemanticsSpec.binds_statics).

   Nothing is asked of [pattern]: whatever it is, the result is [e] with
   the bindings of some of its groups permuted. The validation gives the
   term after Simplify, normalised, as the pattern for the term before.

   The pass reads symbols and code ids, which no renaming changes, and
   the shape of the two terms: it commutes with renaming
   ([order_statics_as_equivariant_statement]).

   The proofs are in FlambdaStaticsOrderProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* A group in the order of another                                          *)
(* ======================================================================== *)

(* The two bindings bind the same name: the same code id to code, or the
   same symbol. *)
Definition binds_same_name (binding binding' : static_binding) : bool :=
  match binding, binding' with
  | Bind_code id _, Bind_code id' _ => Pos.eqb id id'
  | Bind_symbol s _, Bind_symbol s' _ => Pos.eqb s s'
  | _, _ => false
  end.

(* The first binding of the list of which [is_it] holds, moved to the
   front; the list as it is if there is none. *)
Fixpoint to_front (is_it : static_binding -> bool)
    (bindings : list static_binding) : list static_binding :=
  match bindings with
  | [] => []
  | binding :: rest =>
    if is_it binding then bindings
    else
      match to_front is_it rest with
      | found :: rest' =>
        if is_it found then found :: binding :: rest' else bindings
      | [] => bindings
      end
  end.

(* The bindings with those of the names that [pattern] binds first, in
   the order of [pattern]. *)
Definition order_as (pattern bindings : list static_binding)
    : list static_binding :=
  fold_right (fun named bindings => to_front (binds_same_name named) bindings)
    bindings pattern.

(* ======================================================================== *)
(* The normalisation                                                        *)
(* ======================================================================== *)

Fixpoint order_statics_as (pattern e : expr) {struct e} : expr :=
  match e, pattern with
  | Let x kind d body, Let _ _ _ pattern =>
    Let x kind d (order_statics_as pattern body)
  | Let_static bindings body, Let_static pattern_bindings pattern =>
    Let_static (order_as pattern_bindings bindings)
      (order_statics_as pattern body)
  | Let_cont_non_recursive k (Handler params metadata handler) body,
    Let_cont_non_recursive _ (Handler _ _ pattern_handler) pattern =>
    Let_cont_non_recursive k
      (Handler params metadata (order_statics_as pattern_handler handler))
      (order_statics_as pattern body)
  | Let_cont_recursive handlers body, Let_cont_recursive _ pattern =>
    Let_cont_recursive handlers (order_statics_as pattern body)
  | _, _ => e
  end.

Definition order_statics_as_equivalent_statement : Prop :=
  ∀ pattern e : expr, expr_equivalent e (order_statics_as pattern e).

(* The pattern being renamed or not: only its shape, its symbols and its
   code ids are read. *)
Definition order_statics_as_equivariant_statement : Prop :=
  ∀ (r r' : renaming) (pattern e : expr),
    order_statics_as (rename_expr r' pattern) (rename_expr r e)
    = rename_expr r (order_statics_as pattern e).
