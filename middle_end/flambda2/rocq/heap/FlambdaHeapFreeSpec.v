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

(* A term with no primitive of the heap in it. What is stated of one
   is of its run on its own from the empty heap, under a static
   environment whose code has none either: if that run exits, it
   exits with the empty heap. Nothing is stated of a run from another
   heap, under other code, or under a continuation environment.

   This file is to be reviewed: [expr_free_of_heap], the test;
   [statics_free_of_heap]; and the two statements.

   What it is for. The four liftings to symbols (FlambdaLiftSpec.v and
   the three files like it) are stated as refinements, and
   FlambdaSemanticsSpec.refines compares the heaps of two exits by
   equality. A cell of the heap holds values (FlambdaHeapSpec.v). So a
   unit that stores in a cell a block that a lifting makes static
   exits with the block in its heap, where the unit lifted exits with
   the symbol: the two heaps are alike up to
   FlambdaLiftSpec.value_lifted, and they are not equal. What two
   exits are to be compared by is decision D40, which is not taken.
   For now, each of the four liftings lifts nothing in the unit that
   it is given if that unit has a primitive of the heap anywhere:
   [expr_free_of_heap] is their guard for the time being. A unit that
   passes it, run as said above, exits with the empty heap if it
   exits ([free_of_heap_exits_empty_statement]), and a heap alike to
   the empty heap is empty. The copy of code for a [Let] asks the
   same test of the unit that it is given, for the same reason
   (FlambdaCopyLetsSpec.copy_let_code).

   The test is of the text: no [Let] of a primitive of the heap
   (FlambdaModel.Prim of FlambdaPrimitiveModel.Heap) in the unit, in a
   handler or in code. It refuses a unit one of whose functions has a
   primitive of the heap that no run of the unit reaches. It does not
   refuse the other primitives that may read a cell (those of
   FlambdaHeapSpec.prim_reads_heap that are not of the heap: a load,
   the length of an array and a comparison of two locations are among
   them), so it does not say that what a term computes is independent
   of the heap: run from a heap that has cells, a term that passes may
   read them.

   It is the guard that the liftings have for now: a limit of them
   pending that decision, and not a property that a unit is meant to
   have. Nothing here justifies removing it or weakening it while the
   heaps of two exits are compared by equality: a finer guard would
   need a proof of its own and a measure of the units that it turns.
   That the heaps come to be compared by a relation that holds of a
   unit and the unit lifted is one way for it to go, and is not said
   to be the only one.

   The proofs are in FlambdaHeapFreeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaHeapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* The test                                                                 *)
(* ======================================================================== *)

(* No [Let] of the expression has a primitive of the heap for its
   defining expression (FlambdaHeapSpec.named_changes_heap). Handlers,
   those of a recursive group too, and the code that a [Let_static]
   binds are looked into. *)
Fixpoint expr_free_of_heap (e : expr) : bool :=
  match e with
  | Let _ _ d body => negb (named_changes_heap d) && expr_free_of_heap body
  | Let_static bindings body =>
    forallb binding_free_of_heap bindings && expr_free_of_heap body
  | Let_cont_non_recursive _ h body =>
    handler_free_of_heap h && expr_free_of_heap body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_free_of_heap h) handlers
    && expr_free_of_heap body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_free_of_heap (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => expr_free_of_heap handler
  end
with binding_free_of_heap (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_free_of_heap c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_free_of_heap (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => expr_free_of_heap body
  end.

(* The same of the code that a static environment binds, every entry.
   Nothing is asked of what it binds its symbols to. *)
Definition statics_free_of_heap (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_free_of_heap c) (static_code statics).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit that passes the test, run on its own from the empty heap and
   from a static environment whose code passes it, exits with the
   empty heap. The empty static environment and that of the imports of
   a unit (FlambdaImportsSpec.imports_statics) bind no code. *)
Definition free_of_heap_exits_empty_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (k : continuation) (vs : list value)
    (hp : heap),
    expr_free_of_heap e = true ->
    statics_free_of_heap statics = true ->
    behaves (State e [] [] statics empty_heap no_reads) (Exits k vs hp) ->
    hp = empty_heap.

(* The test commutes with renaming: no name is looked at. *)
Definition expr_free_of_heap_rename_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    expr_free_of_heap (rename_expr r e) = expr_free_of_heap e.
