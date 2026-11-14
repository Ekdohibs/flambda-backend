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

(* Proofs of the statements of FlambdaCanonicalFreeNamesSpec.v. This file
   needs no review.

   Each statement is proved from its older form, a lemma of
   FlambdaNominalProofs.v or of FlambdaAlphaProofs.v, with the premise
   about names turned by [var_free_among] and [cont_free_among]: what is
   free by the relation is among the list, so a renaming that fixes the
   list fixes it ([variables_fixed] and the lemmas beside it, in
   FlambdaAlphaProofs.v), and a name that is not among the list is not
   free. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaMapProofs FlambdaBindersSpec FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaNominalProofs FlambdaAlphaSpec
  FlambdaAlphaProofs FlambdaFreeNamesSpec FlambdaFreeNamesProofs
  FlambdaNameAbstractionSpec FlambdaCanonicalFreeNamesSpec.
Import ListNotations.

(* ======================================================================== *)
(* The names that a binder binds may be renamed; bindings not used          *)
(* ======================================================================== *)

Theorem rename_let_canonical : rename_let_canonical_statement.
Proof.
  intros r x kind d body Hclosed [Hvars Hconts].
  apply rename_let;
    [ exact Hclosed | apply variables_fixed_but_one, Hvars
    | apply continuations_fixed, Hconts ].
Qed.

Theorem freshen_let_canonical : freshen_let_canonical_statement.
Proof.
  intros x fresh kind d body Hclosed Hfresh.
  apply freshen_let; [ exact Hclosed | ].
  intros Hfree. apply var_free_among in Hfree. congruence.
Qed.

Theorem rename_let_cont_canonical : rename_let_cont_canonical_statement.
Proof.
  intros r k h body Hclosed [Hvars Hconts].
  apply rename_let_cont;
    [ exact Hclosed | apply variables_fixed, Hvars
    | apply continuations_fixed_but_one, Hconts ].
Qed.

Theorem rename_handler_params_canonical :
  rename_handler_params_canonical_statement.
Proof.
  intros r k params metadata handler body Hclosed [Hvars Hconts].
  apply rename_handler_params;
    [ exact Hclosed | apply variables_fixed_but_params, Hvars
    | apply continuations_fixed, Hconts ].
Qed.

Theorem rename_recursive_handler_params_canonical :
  rename_recursive_handler_params_canonical_statement.
Proof.
  intros r before k params metadata handler after body Hclosed
    [Hvars Hconts].
  apply rename_recursive_handler_params;
    [ exact Hclosed | apply variables_fixed_but_params, Hvars
    | apply continuations_fixed, Hconts ].
Qed.

Theorem rename_let_cont_recursive_canonical :
  rename_let_cont_recursive_canonical_statement.
Proof.
  intros r handlers body Hclosed [Hvars Hconts].
  apply rename_let_cont_recursive;
    [ exact Hclosed | apply variables_fixed, Hvars
    | apply continuations_fixed, Hconts ].
Qed.

Theorem unused_let_canonical : unused_let_canonical_statement.
Proof.
  intros x kind d body Hheap Hunused. apply unused_let; [ exact Hheap | ].
  intros Hfree. apply var_free_among in Hfree. congruence.
Qed.

Theorem unused_let_cont_canonical : unused_let_cont_canonical_statement.
Proof.
  intros k h body Hunused. apply unused_let_cont.
  intros Hfree. apply cont_free_among in Hfree. congruence.
Qed.

Theorem unused_let_cont_recursive_canonical :
  unused_let_cont_recursive_canonical_statement.
Proof.
  intros handlers body Hstatics Hunbound.
  apply unused_let_cont_recursive; [ exact Hstatics | ].
  intros k Hk. apply Hunbound, cont_free_among, Hk.
Qed.

(* ======================================================================== *)
(* Execution is equivariant in the variables                                *)
(* ======================================================================== *)

Theorem rename_variables_equivalent_canonical :
  rename_variables_equivalent_canonical_statement.
Proof.
  intros r e vars conts statics hp rd Hclosed Hconts.
  apply rename_variables_equivalent;
    [ exact Hclosed | apply continuations_fixed, Hconts ].
Qed.
