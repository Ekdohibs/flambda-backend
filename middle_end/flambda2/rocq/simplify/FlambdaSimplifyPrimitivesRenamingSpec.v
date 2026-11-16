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


(* The type of an argument, computed, commutes with renaming.

   This file is to be reviewed: [rename_answer] and the three
   statements.

   [simplify_simple] and [simplify_simples]
   (FlambdaSimplifyPrimitivesSpec.v) read a typing environment and the
   [Simple]s of a term, and answer types and canonical elements, which
   hold variables. A renaming (FlambdaNominalSpec.renaming) acts on
   [Simple]s, on types ([rename_type], FlambdaTypeRenameSpec.v) and on
   typing environments ([rename_typing_env], FlambdaEnvRenameSpec.v).
   Asked about the renamed [Simple] in the renamed environment, the
   two functions answer the renamed answer, and stop where they
   stopped. The test [well_formed_input_b] answers the same of the
   renamed environment and type.

   The statements are equations and have no premise. They are here,
   and not with those of the function that is built on these
   (FlambdaSimplifyPassRenamingSpec.v), so that another function on
   terms that looks up its [Simple]s can state its own equation.

   Nothing is trusted here. The proofs are in
   FlambdaSimplifyPrimitivesRenamingProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaNominalSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar FlambdaTypingEnv
  FlambdaTypeRenameSpec FlambdaEnvRenameSpec FlambdaSimplifyPrimitivesSpec.
Import ListNotations.

(* What [simplify_simple] answers: the type and the canonical element
   are renamed. *)
Definition rename_answer (r : renaming) (answer : flambda_type * simple)
    : flambda_type * simple :=
  (rename_type r (fst answer), rename_simple r (snd answer)).

Definition simplify_simple_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (s : simple),
    simplify_simple (rename_typing_env r tenv) (rename_simple r s)
    = option_map (rename_answer r) (simplify_simple tenv s).

Definition simplify_simples_equivariant_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (args : list simple),
    simplify_simples (rename_typing_env r tenv) (map (rename_simple r) args)
    = option_map (map (rename_answer r)) (simplify_simples tenv args).

Definition well_formed_input_b_rename_statement : Prop :=
  ∀ (r : renaming) (tenv : Typing_env.t) (ty : flambda_type),
    well_formed_input_b (rename_typing_env r tenv) (rename_type r ty)
    = well_formed_input_b tenv ty.
