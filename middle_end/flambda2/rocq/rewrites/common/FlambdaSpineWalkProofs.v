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

(* Proofs of the statements of FlambdaSpineWalkSpec.v. Nothing here
   needs review.

   Each is one induction on the term (FlambdaTermInductionProofs.v):
   what the induction gives for what is inside a node, under the
   context of the node (FlambdaContextProofs.v), and then the
   hypothesis on the step. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaSemanticsSpec
  FlambdaSemanticsProofs FlambdaContextSpec FlambdaContextProofs
  FlambdaNormaliseProofs FlambdaTermInductionProofs FlambdaSpineWalkSpec.
Import ListNotations.

Theorem walk_spine_refines : walk_spine_refines_statement.
Proof.
  intros step Hstep e.
  induction e using expr_deep_ind; cbn [walk_spine];
    refine (expr_refines_trans _ _ _ _ (Hstep _));
    try apply expr_refines_refl.
  - exact (plug_refines (Let_body x kind d Hole) _ _ (not_many _) IHe).
  - exact (plug_refines (Let_static_body bindings Hole) _ _ (not_many _) IHe).
  - apply (expr_refines_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (walk_spine step e1)) e2)).
    + exact (plug_refines
        (Let_cont_non_recursive_handler k params metadata Hole e2) _ _
        (not_many _) IHe1).
    + exact (plug_refines
        (Let_cont_non_recursive_body k
           (Handler params metadata (walk_spine step e1)) Hole)
        _ _ (not_many _) IHe2).
  - exact (plug_refines (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) IHe).
Qed.

Theorem walk_spine_equivalent : walk_spine_equivalent_statement.
Proof.
  intros step Hstep e.
  induction e using expr_deep_ind; cbn [walk_spine];
    refine (expr_equivalent_trans _ _ _ _ (Hstep _));
    try (intros vars conts statics hp rd; apply equivalent_refl).
  - exact (plug_equivalent (Let_body x kind d Hole) _ _ (not_many _) IHe).
  - exact (plug_equivalent (Let_static_body bindings Hole) _ _ (not_many _)
      IHe).
  - apply (expr_equivalent_trans _
      (Let_cont_non_recursive k
         (Handler params metadata (walk_spine step e1)) e2)).
    + exact (plug_equivalent
        (Let_cont_non_recursive_handler k params metadata Hole e2) _ _
        (not_many _) IHe1).
    + exact (plug_equivalent
        (Let_cont_non_recursive_body k
           (Handler params metadata (walk_spine step e1)) Hole)
        _ _ (not_many _) IHe2).
  - exact (plug_equivalent (Let_cont_recursive_body handlers Hole) _ _
      (not_many _) IHe).
Qed.
