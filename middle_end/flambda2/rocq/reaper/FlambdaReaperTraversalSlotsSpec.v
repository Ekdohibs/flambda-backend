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

(* What the traversal puts into the variable of a projection of a value
   slot, and where it reads value slots.

   This file is to be reviewed: [slot_projection] and the statements. The
   proofs are in FlambdaReaperTraversalSlotsProofs.v, which needs no
   review.

   - [slot_read_only_statement]: in the facts of the traversal (any flag)
     of a unit that binds each variable once, the variable of a [Let] of
     a projection of a value slot is put into by its read only
     ([only_read_into] of FlambdaReaperLeastSpec.v, with no alias that
     holds where the closure has any source): no other fact aliases into
     it, reads into it, gives it a source, makes it a parameter or holds
     something in it. A step of the traversal puts something into a
     variable only where the variable is bound: the [Let] of a projection
     makes its read; the other steps put things into the parameters of
     handlers and of code, into the variables of other [Let]s (a block, a
     closure, the read of a function slot, ...), into symbols, and into
     the nodes that they make.
   - [slot_reads_projected_statement]: each read of a value slot in
     those facts is that of the [Let] of a projection: of the value slot
     that it projects, off the node of the simple that it projects it
     from, into its variable. No other step reads a value slot. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperLeastSpec.
Import ListNotations.

(* The value slot that a defining expression projects and the simple it
   projects it from, if it is a projection of a value slot. *)
Definition slot_projection (d : named) : option (value_slot * simple) :=
  match d with
  | Project_value_slot w c | Project_value_slot_from _ w c => Some (w, c)
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ => None
  end.

Definition slot_read_only_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (v : variable) (d : named) (w : value_slot) (c : simple),
    variables_bound_once e = true ->
    (v, d) ∈ let_bindings e -> slot_projection d = Some (w, c) ->
    only_read_into (traverse_unit flag u e) (Node_variable v)
      (Field_value_slot w) (node_of_simple c) false
    = true.

Definition slot_reads_projected_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (to_ : node) (w : value_slot) (b : node),
    variables_bound_once e = true ->
    (to_, Field_value_slot w, b) ∈ f_accessor (traverse_unit flag u e) ->
    ∃ (v : variable) (d : named) (c : simple),
      to_ = Node_variable v ∧ (v, d) ∈ let_bindings e
      ∧ slot_projection d = Some (w, c) ∧ b = node_of_simple c.
