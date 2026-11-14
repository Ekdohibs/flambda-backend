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


(* The closed static bindings of a unit taken to its head, of a unit
   that something follows.

   This file is to be reviewed: the statement.

   FlambdaHoistStaticsSpec.hoist_statics is stated of a unit run on its
   own and from its imports (FlambdaHoistStaticsSpec.v). Here the two
   units are followed by something: they are run from one continuation
   environment of what may follow a unit
   (FlambdaObserversSpec.observers: a way out, or a handler with no
   environment of its own that binds nothing static and names no symbol
   and no code id), from the static environment of the imports and the
   empty heap, and they behave alike
   ([hoist_statics_equivalent_observed_statement]). The premise is that
   of the statement from the imports: the unit binds no name that the
   imports bind. With no continuation it is that statement.

   What is new with something after. Where the unit leaves before it
   has executed all its static bindings, as by its exception
   continuation, the unit with its bindings first has bound symbols
   that the unit has not. What follows names no symbol,
   and the values that it is given hold only symbols that the unit had
   bound when it left
   (FlambdaStaticsInScopeSpec.statics_bound_in_run_statement), which
   the two static environments bind to the same; so it reads the same
   in the two runs. A handler that named a symbol could tell the two
   apart: one that the unit had not bound yet is bound in the second
   run only.

   Whether this is what the verdict on two units one of which has its
   bindings first is to mean has not been decided
   (FlambdaObserversSpec.v). No function of the validation changes.

   The proofs are in FlambdaHoistStaticsObservedProofs.v, which needs
   no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaImportsSpec
  FlambdaHoistStaticsSpec FlambdaObserversSpec.
Import ListNotations.

(* A unit that passes and the unit with its closed bindings first, each
   run from one continuation environment of what may follow a unit and
   from the static environment of its imports, behave alike, if the
   unit binds no name that the imports bind. *)
Definition hoist_statics_equivalent_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env)
    (u : unit_interface) (e e' : expr),
    hoist_statics u e = Some e' ->
    static_names_free_in (imports_statics imports) e = true ->
    observers [] conts ->
    equivalent (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State e' [] conts (imports_statics imports) empty_heap no_reads).
