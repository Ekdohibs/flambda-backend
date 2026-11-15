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

(* Not computing again a primitive that has been computed: the result
   is equivalent to the term.

   This file is to be reviewed: the statement.

   FlambdaCseSpec.v defines [share_known_primitives] and states that
   its result refines the term. That the result is equivalent to the
   term is stated here
   ([share_known_primitives_equivalent_statement]), and that file
   names this statement for it. A refinement is what a rewrite of the
   term before Simplify owes. A rewrite of the term after owes the
   converse, that the term refines its result; so a rewrite that the
   validation is to apply to both of the terms that it compares owes
   the equivalence.

   Why it holds is in the definition of what the pass knows
   (FlambdaCseSpec.all_computed): a variable known to hold the result
   of a primitive is bound, where the term runs, to the value that the
   primitive gives when the machine runs it there on its arguments,
   at the heap and the read record of that place, and that run leaves
   the heap and the record as they are. That says that the primitive
   runs there, to which value, and that it changes nothing. So the
   [Let] of a primitive and the [Let] of the variable that the pass
   puts in its place bind the same value and go on with the same heap
   and the same record: the pass does not put a variable that has a
   value where a primitive had none, which a refinement would allow.
   The statement is about the pass as FlambdaCseSpec.v defines it,
   whatever [shareable] lists among the primitives whose value stays
   (FlambdaHeapSpec.stable_prim) and whatever
   FlambdaCseSpec.computed_under_let adds, as long as [all_computed]
   is kept; that it is kept is proved in FlambdaCseProofs.v.

   The proofs are in FlambdaCseEquivalentProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaCseSpec.
Import ListNotations.

(* The result behaves as the term does, under all environments. *)
Definition share_known_primitives_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (share_known_primitives e).
