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


(* The unboxing of continuation parameters commutes with renaming, and
   keeps scope.

   This file is to be reviewed: [rename_facts] and the statements.

   FlambdaUnboxParamsSpec.unbox_known_params goes through a term
   carrying what is known of its variables (the blocks and the wrapped
   numbers of FlambdaBlockLoadSpec.v and FlambdaUnboxSpec.v, and the
   reads of boxes in the form of FlambdaCseSpec.v), compares
   names by equality only, and binds no name that the term has not got:
   the variable that becomes a parameter was bound by the [Let] that
   goes. So it commutes with the renamings of FlambdaNominalSpec.v, in
   the forms of FlambdaValidatorEquivarianceSpec.v, which this file
   follows:
   - what a primitive that reads a box is known to give of an argument
     ([projection_of]) is renamed when what is known and the argument
     are;
   - an expression with its applications extended
     ([extend_applications]) is renamed when what is known, the
     continuation and the expression are, and the one fails where the
     other does;
   - the pass given what is known ([unbox_params_knowing]) gives the
     renamed result when what is known is renamed with the term;
   - the pass as the validation would run it, with nothing known, is a
     rewrite that takes a term and nothing else
     ([unbox_known_params_equivariant_statement], which is
     [rewrite_equivariant unbox_known_params]).
   The position of a parameter and the primitive are not names and are
   as they were.

   Each of these statements is an equation and holds of every term.

   The pass also keeps scope
   (FlambdaValidatorEquivarianceSpec.keeps_scope): on a term whose code
   is closed, the code of the result is closed and every variable or
   continuation that is free in the result is free in the term. What an
   application is given is a constant, or what a [Let] or a static
   binding in scope was given, and that is forgotten under a binder of
   it; and the variable that becomes a parameter was bound in the
   handler by the [Let] that goes.

   The proofs are in FlambdaUnboxParamsRenameProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaPrimitiveModel
  FlambdaNominalSpec FlambdaBlockLoadSpec FlambdaUnboxSpec FlambdaCseSpec
  FlambdaValidatorEquivarianceSpec FlambdaUnboxParamsSpec.
Import ListNotations.

(* What is known is renamed as the passes that keep it rename it. *)
Definition rename_facts (r : renaming) (known : facts) : facts :=
  Facts (rename_known_blocks r (facts_blocks known))
    (rename_known_numbers r (facts_numbers known))
    (rename_computed r (facts_reads known)).

Definition projection_of_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (p : unary_primitive) (arg : simple),
    projection_of (rename_facts r known) p (rename_simple r arg)
    = option_map (rename_simple r) (projection_of known p arg).

Definition extend_applications_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (k : continuation) (i : nat)
    (p : unary_primitive) (e : expr),
    extend_applications (rename_facts r known) (rename_continuation r k) i p
      (rename_expr r e)
    = option_map (rename_expr r) (extend_applications known k i p e).

Definition unbox_params_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : facts) (e : expr),
    unbox_params_knowing (rename_facts r known) (rename_expr r e)
    = rename_expr r (unbox_params_knowing known e).

Definition unbox_known_params_equivariant_statement : Prop :=
  rewrite_equivariant unbox_known_params.

Definition unbox_known_params_keeps_scope_statement : Prop :=
  keeps_scope unbox_known_params.
