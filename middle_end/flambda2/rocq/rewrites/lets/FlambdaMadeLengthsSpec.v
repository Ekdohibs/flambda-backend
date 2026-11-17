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

(* The length of a mutable array that a [Let] in scope has made: a
   normalisation that is an equivalence.

   This file is to be reviewed: [fold_made_lengths] and
   [lengths_knowing], which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   The sharing of primitives of FlambdaCseSpec.v knows, under a [Let]
   of a variable to [Make_mutable_array], that the variable holds the
   array which that allocation made of its arguments
   (FlambdaCseSpec.array_made_under_let), and makes a later [Let] of the length
   of that variable, at any length kind, a [Let] of the number of
   those arguments (FlambdaCseSpec.made_length_named). The elements of
   such an array may be written; its length does not change
   (FlambdaHeapSpec.array_length_known_statement).

   [fold_made_lengths] is that rule alone. It is the traversal of
   FlambdaCseSpec.share_primitives_knowing, but for the one case said
   below, with only the arrays that [Let]s in scope have made kept as
   known, and only a [Let] of the length of one rewritten. No
   primitive is shared and no allocation is touched. What is known of
   a variable, and what mentions it in an argument, is forgotten
   under a binder of it (FlambdaCseSpec.forget_computed); a handler
   of a non-recursive continuation is gone into, with what is known
   where it is defined; the handlers of recursive ones and the bodies
   of code are left as they are. So nothing is known of an array that
   no [Let] in scope makes: of a parameter, of what a call returns,
   of what is read from a block or from a closure.

   The one case is the handlers of recursive continuations. The
   sharing of primitives goes into each of them with what is known
   where the [Let_cont] is, less what mentions the parameters of the
   handler; this rule leaves them. So it does not make a constant of
   the length that a loop takes of an array which a [Let] made before
   the loop.

   It is for the rounds of the term after Simplify
   (FlambdaValidateSpec.normalise_expr_after), which do not share
   primitives; those of the term before do. Where the validation
   inlines a call in the two terms, the rounds of the term before make
   the length of an array made at the call its constant, and then fold
   the comparisons on it and the arms that those decide; the rounds of
   the term after must do the same, or the two terms no longer meet.
   Nothing else of the sharing is applied to the term after in its
   rounds.

   The result is equivalent to the term
   ([fold_made_lengths_equivalent_statement]), which is what a rewrite
   of the term after Simplify owes. Why it holds is in what it means
   for an entry of an allocation to hold
   (FlambdaCseSpec.array_made_holds): the variable is bound, where the
   term runs, to what the allocation gave on as many values as it had
   arguments, in a heap that the one it left has evolved to. The
   length of that array has a value there, at any length kind,
   exactly where that number is a tagged immediate, and the value is
   that number (FlambdaPrimitiveSpec.array_length_of makes the test
   that FlambdaCseSpec.made_length_named makes). So the [Let] of the
   length and the [Let] of the constant bind the same value and go on
   with the same heap and the same read record: the constant is not
   put where the length had no value.

   The rule commutes with the renamings of FlambdaNominalSpec.v, what
   it knows being renamed with the term
   ([lengths_knowing_equivariant_statement],
   [fold_made_lengths_equivariant_statement]): it compares variables
   for equality and makes no name. It keeps scope
   (FlambdaKeepsScopeSpec.fold_made_lengths_keeps_scope_statement).

   The proofs are in FlambdaMadeLengthsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaNominalSpec FlambdaCseSpec.
Import ListNotations.

(* What is known is a list of FlambdaCseSpec.computed in which every
   entry is of the allocation of a mutable array: the traversal starts
   from nothing and adds what FlambdaCseSpec.array_made_under_let
   gives. *)
Fixpoint lengths_knowing (known : computed) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind
      match made_length_named known d with
      | Some d' => d'
      | None => d
      end
      (lengths_knowing
         (array_made_under_let x d ++ forget_computed x known) body)
  | Let_static bindings body =>
    Let_static bindings (lengths_knowing known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (lengths_in_handler known h)
      (lengths_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (lengths_knowing known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with lengths_in_handler (known : computed) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (lengths_knowing (forget_computed_vars params known) handler)
  end.

Definition fold_made_lengths (e : expr) : expr := lengths_knowing [] e.

(* The result behaves as the term does, under all environments. *)
Definition fold_made_lengths_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (fold_made_lengths e).

(* What is known is renamed with the term
   (FlambdaCseSpec.rename_computed). *)
Definition lengths_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : computed) (e : expr),
    lengths_knowing (rename_computed r known) (rename_expr r e)
    = rename_expr r (lengths_knowing known e).

(* The rule as the validation runs it, with nothing known. *)
Definition fold_made_lengths_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    fold_made_lengths (rename_expr r e) = rename_expr r (fold_made_lengths e).
