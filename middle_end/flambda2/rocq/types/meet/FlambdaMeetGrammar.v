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



(* What meet uses of its neighbours, and that FlambdaTypingEnv.v and
   FlambdaMapSpec.v do not have.

   This file is to be reviewed, each definition against the function of
   the compiler that it names. It holds definitions only and it
   extracts. They are:
   - of types/env/typing_env.ml, for FlambdaTypingEnv.v, which is not
     this stream's to edit, and where it is to move under the same
     name: [kind_of_simple];
   - [sort_numbers], a sort of integers, which the module of maps
     has not and is not to have.
   What was here of the grammar of types ([must_be_singleton] and the
   others) is in FlambdaTypeGrammar.v.

   The difference from the compiler: [kind_of_simple] is
   [Simple.kind]. A variable of the compiler carries its kind; a
   variable of the model does not, and its kind is that of the type
   that the environment has for it (FlambdaTypingEnv.v). [None] where
   the environment has none. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlOption.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningBoolSpec FlambdaBindingTime FlambdaEnvAliases
  FlambdaTypingEnv.
Import ListNotations.

(* ======================================================================== *)
(* Of the module of maps                                                    *)
(* ======================================================================== *)

(* Integers in increasing order. *)
Fixpoint insert_number (i : Z) (sorted : list Z) : list Z :=
  match sorted with
  | [] => [i]
  | j :: rest => if Z.leb i j then i :: sorted else j :: insert_number i rest
  end.

Definition sort_numbers (numbers : list Z) : list Z :=
  fold_right insert_number [] numbers.

(* ======================================================================== *)
(* Of the typing environment                                                *)
(* ======================================================================== *)

(* [Simple.kind], read from the environment for a variable. *)
Definition kind_of_simple (tenv : Typing_env.t) (simple : simple)
    : option flambda_kind :=
  match simple with
  | Const c => Some (kind_of_const c)
  | Symbol _ => Some Value
  | Var x => option_map kind (Typing_env.find tenv (Name_var x) None)
  end.
