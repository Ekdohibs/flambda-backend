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

(* The proofs of FlambdaReaperClosuresTablesAlikeSpec.v. Nothing here
   needs review.

   [checked_within]: a chain of nested calls goes through a field read
   into names all to be unboxed with one source, where [unboxing_checked]
   asks one level less; so no chain is longer than its bound. The
   counterexample is computed. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaReaperGraphSpec FlambdaReaperUnboxing FlambdaReaperLeastSpec
  FlambdaReaperClosuresTablesAlikeSpec.
From OxCamlFlambda2 Require FlambdaReaperLeastProofs
  FlambdaReaperUnboxingFuelSpec FlambdaReaperUnboxingFuelProofs.
Import ListNotations.

Module UF := FlambdaReaperUnboxingFuelSpec.
Module UFP := FlambdaReaperUnboxingFuelProofs.

Section Checked.

Variable F : facts.
Variable T : tables.
Variable cm : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

Lemma checked_within (n : nat) :
  ∀ (block : node) (fields : list (field * field_use)),
    unboxing_checked F T cm dominated to_unbox n block fields = true ->
    UF.calls_within F T cm dominated to_unbox n block fields.
Proof.
  induction n as [ | n IH]; intros block fields Hc source fields' Hn;
    inversion Hn as [f flow_to source' Hin Hd Hs Ha]; subst;
    cbn [unboxing_checked] in Hc; rewrite forallb_forall in Hc;
    specialize (Hc _ Hin); cbv beta iota in Hc;
    rewrite Hd, Hs in Hc; cbn [negb orb] in Hc; rewrite Ha in Hc;
    [ discriminate Hc | exact (IH _ _ Hc) ].
Qed.

End Checked.

Theorem unboxing_checked_ends : unboxing_checked_ends_statement.
Proof.
  intros F T cm dominated to_unbox n block fields Hc.
  exact (UFP.within_ends F T cm dominated to_unbox n block fields
           (checked_within F T cm dominated to_unbox n block fields Hc)).
Qed.

Theorem unboxed_alike_counterexample :
  unboxed_alike_counterexample_statement.
Proof.
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split; [ vm_compute; reflexivity | ].
  split.
  { apply (FlambdaReaperLeastProofs.check_least_sound _ _ 200).
    vm_compute. reflexivity. }
  do 10 (split; [ vm_compute; reflexivity | ]).
  vm_compute. discriminate.
Qed.
