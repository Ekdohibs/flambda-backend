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


(* Proofs of the statements of FlambdaSimplifyPrimitivesRenamingSpec.v.
   This file needs no review: each statement is closed by a [Theorem]
   whose type is the statement. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaBindingTime FlambdaTypingEnv FlambdaTypingEnvInvariantSpec
  FlambdaTypeRenameSpec FlambdaTypeRenameProofs
  FlambdaEnvRenameSpec FlambdaEnvRenameProofs FlambdaSimplifyPrimitivesSpec
  FlambdaSimplifyPrimitivesRenamingSpec.
Import ListNotations.

Theorem simplify_simple_equivariant : simplify_simple_equivariant_statement.
Proof.
  intros r tenv s. unfold simplify_simple.
  rewrite mem_simple_equivariant.
  destruct (Typing_env.mem_simple (Some Name_mode.normal) tenv s);
    [ | reflexivity ].
  rewrite type_simple_in_term_exn_equivariant.
  destruct (Typing_env.type_simple_in_term_exn tenv (Some Name_mode.normal) s)
    as [[[ty canonical] | ] | ];
    reflexivity.
Qed.

Theorem simplify_simples_equivariant
    : simplify_simples_equivariant_statement.
Proof.
  intros r tenv args.
  induction args as [ | arg args IH]; cbn [map simplify_simples];
    [ reflexivity | ].
  rewrite simplify_simple_equivariant, IH.
  destruct (simplify_simple tenv arg) as [answer | ]; [ | reflexivity ].
  destruct (simplify_simples tenv args) as [answers | ]; reflexivity.
Qed.

Theorem well_formed_input_b_rename : well_formed_input_b_rename_statement.
Proof.
  intros r tenv ty. unfold well_formed_input_b.
  rewrite types_well_formed_b_rename, (proj1 type_well_formed_rename).
  reflexivity.
Qed.

(* The types and the canonical elements of renamed answers. *)
Lemma map_fst_rename_answer (r : renaming)
    (answers : list (flambda_type * simple)) :
  map fst (map (rename_answer r) answers)
  = map (rename_type r) (map fst answers).
Proof. rewrite !map_map. reflexivity. Qed.

Lemma map_snd_rename_answer (r : renaming)
    (answers : list (flambda_type * simple)) :
  map snd (map (rename_answer r) answers)
  = map (rename_simple r) (map snd answers).
Proof. rewrite !map_map. reflexivity. Qed.
