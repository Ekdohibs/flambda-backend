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

(* Lifting out of the arms of a test at the toplevel of a unit: a
   [Let], outside code, of a closure, an immutable block, a boxed
   number or an immutable array whose parts are symbols and constants,
   made a static binding of a symbol at the head of the unit.

   This file is to be reviewed: [lift_to_head], against what Simplify
   does (below); [lift_to_head_unit], which is extracted and run by the
   validation of middle_end/flambda2/tests/semantics/; and the
   statements.

   What Simplify does. A static binding stands only at the toplevel of
   a unit: in the unit itself, and in a handler that all of its body
   goes to (FlambdaWellFormedSpec.handler_at_unit_toplevel). The arms of a
   test that is not resolved are handlers that are not at toplevel:

     let f = if c then (fun x -> x + 1) else (fun x -> x + 2)

   has one handler for each arm, each making a closure, and
   FlambdaLiftSpec.lift_expr, which makes a [Let] a static binding
   where it stands, cannot be used there. Simplify lifts the two
   closures, which need nothing from around them, as it lifts out of
   the code of a function (FlambdaLiftFromCodeSpec.v), and places
   their bindings, in one group, at the nearest place at toplevel that
   it comes back through: under the last [Let] before the test, in the
   unit or in the handler at toplevel that the test is in.

   What is here. [lift_to_head] makes the [Let]s at the paths given
   aliases of symbols (FlambdaLiftFromCodeSpec.lift_from_code, with no
   code named), and binds the symbols in one group around the unit
   (FlambdaLiftFromCodeSpec.lifted_out_bindings):

     e   becomes   Let_static bindings e'

   where [e'] is [e] with [Let x = Symbol s in ...] for each [Let]
   lifted. This is not where Simplify binds them. The validation
   brings the two terms together otherwise: it compares them with
   their static bindings that mention no variable bound first. As for
   the lifting out of code, only a [Let] that mentions no variable is
   lifted (FlambdaLiftFromCodeSpec.closed_named), the variable stays,
   bound to the symbol, a [Let] is named by its path in the unit, the
   handlers of recursive continuations are gone into and the bodies
   of code are not, two [Let]s that are given one symbol and make the
   same constant have one binding
   (FlambdaLiftFromCodeSpec.once_each, and the header of that file
   for what is refused), and which [Let]s, to which symbols, is a
   guess of the caller's, of which nothing is believed. The guess may
   name any such [Let] of the unit, in a handler that is at toplevel
   or in the unit itself as well as in the arm of a test or in a
   loop.

   Why it is right. As for the lifting out of code: the first run makes
   a closure, a block, a boxed number or an array when the [Let] is
   executed, and the second has the symbol, bound before anything else
   of the unit runs, to the same thing; the two values are related by
   FlambdaLiftSpec.value_lifted. The checks are the same
   ([lift_to_head_unit]): the unit exits with symbols and constants
   only; it has no primitive of the heap
   (FlambdaHeapFreeSpec.expr_free_of_heap; a limit, the CR at
   FlambdaLiftSpec.lift_unit); and the result is a well-formed unit,
   so that the symbols given are new and distinct. The binding of a
   symbol cannot itself be undefined, for the reasons given in
   FlambdaLiftFromCodeSpec.v; so a [Let] that is never executed brings
   no undefinedness to the unit lifted.

   The proofs are in FlambdaLiftFromArmsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaWellFormedSpec
  FlambdaUnusedStaticsSpec FlambdaImportsSpec FlambdaHeapFreeSpec
  FlambdaLiftSpec FlambdaLiftFromCodeSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The unit with the [Let]s at the paths [lets] made aliases of the
   symbols that stand for them, and those symbols bound, once each, in
   one group around it. The unit itself if nothing is lifted. *)
Definition lift_to_head (lets : list (path * lift_to)) (e : expr) : expr :=
  let bindings := once_each [] (lifted_out_bindings lets root e) in
  match bindings with
  | [] => e
  | _ :: _ => Let_static bindings (lift_from_code [] lets root e)
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The unit with the [Let]s of the guess lifted to its head: if the
   unit exits with symbols and constants only and has no primitive of
   the heap, and the result is a well-formed unit. The second is a
   limit, and not a condition of the lifting: the CR at
   FlambdaLiftSpec.lift_unit. *)
Definition lift_to_head_unit (u : unit_interface)
    (lets : list (path * lift_to)) (e : expr) : option expr :=
  let lifted := lift_to_head lets e in
  if
    exits_plain_knowing [] [] e
    && expr_free_of_heap e
    && well_formed_unit u lifted
  then Some lifted
  else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [lift_to_head_unit] gives refines the unit, run on its own. *)
Definition lift_to_head_unit_program_refines_statement : Prop :=
  ∀ (u : unit_interface) (lets : list (path * lift_to)) (e e' : expr),
    lift_to_head_unit u lets e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports: if the result binds no name that the
   imports bind, it refines the unit. A symbol that the guess gives to
   a [Let] and that is an import would be bound twice, and the group
   at the head is executed whether the arm of the [Let] is taken or
   not. *)
Definition lift_to_head_unit_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (u : unit_interface)
    (lets : list (path * lift_to)) (e e' : expr),
    lift_to_head_unit u lets e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    unit_refines imports e e'.

(* The rewrite commutes with renaming: a [Let] is named by where it is,
   and what stands for one that is lifted mentions no variable and no
   continuation. *)
Definition lift_to_head_equivariant_statement : Prop :=
  ∀ (r : renaming) (lets : list (path * lift_to)) (e : expr),
    rename_expr r (lift_to_head lets e) = lift_to_head lets (rename_expr r e).
