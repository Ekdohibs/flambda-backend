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
   heap, under other code, or under a continuation environment. The
   same is stated of a term that has none outside the bodies of its
   code and that calls nothing outside them ([free_of_heap_fine]),
   under any static environment: its run enters no code.

   This file is to be reviewed: [expr_free_of_heap], the test;
   [statics_free_of_heap]; [calls_nothing],
   [expr_free_of_heap_outside_code] and [free_of_heap_fine], the guard
   that the liftings ask, with the CR at the last; and the statements.

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
   it is given unless that unit passes [free_of_heap_fine], their
   guard for the time being: the unit has no primitive of the heap
   anywhere ([expr_free_of_heap]), or it has none outside the bodies
   of its code and no [Apply] outside them. A unit that passes it,
   run as said above, exits with the empty heap if it exits
   ([free_of_heap_fine_exits_empty_statement]), and a heap alike to
   the empty heap is empty. The copy of code for a [Let] asks the
   same guard of the unit that it is given, for the same reason
   (FlambdaCopyLetsSpec.copy_let_code).

   [expr_free_of_heap] is of the text: no [Let] of a primitive of the
   heap (FlambdaModel.Prim of FlambdaPrimitiveModel.Heap) in the unit,
   in a handler or in code. It refuses a unit one of whose functions
   has a primitive of the heap that no run of the unit reaches. It
   does not refuse the other primitives that may read a cell (those of
   FlambdaHeapSpec.prim_reads_heap that are not of the heap: a load,
   the length of an array and a comparison of two locations are among
   them), so it does not say that what a term computes is independent
   of the heap: run from a heap that has cells, a term that passes may
   read them.

   [free_of_heap_fine] accepts as well a unit whose functions have
   primitives of the heap, where the unit has no [Apply] outside the
   bodies of its code ([calls_nothing]). That test is of the text: an
   [Apply] in a handler that no run reaches makes it fail. From the
   first state of the unit, with no continuation given, the first
   entry into code needs an [Apply]. (An [Apply_cont] enters code
   where its continuation is what is left of an over-application,
   FlambdaSemanticsSpec.apply_cont_value; a call makes such a
   continuation, so a run that has made no call has none.) So the
   run of such a unit on its own enters no code and executes no
   primitive that is in a body, whatever code the static environment
   binds ([calls_nothing_exits_empty_statement]).

   What the finer guard gives away. The statements of the liftings
   are of the unit's own run (FlambdaUnusedStaticsSpec.program_refines,
   FlambdaImportsSpec.unit_refines). For a unit whose own run enters
   no code they say nothing of the bodies of its functions, with a
   guard or without one: no state of that run is in a body.
   [expr_free_of_heap] is of the text of all the code of the unit,
   entered or not: while it was the guard, a lifting was made, and a
   verdict could rest on it, only in a unit whose code has no
   primitive of the heap. The finer guard asks that only of a unit
   that has an [Apply] outside the bodies of its code. A module of a
   library often has none: its own run makes closures and a block.
   Where it has none, and no primitive of the heap outside those
   bodies, it passes the finer guard though its functions store in
   cells; where the other checks of a lifting pass it may be lifted,
   and a verdict may rest on that lifting, on a run that calls none
   of those functions. What the validation then has of them is that
   the two terms, bodies included, are the same once its rewrites
   are made, each of which has its own statement; that is no theorem
   about a call of them. That a verdict may rest on that is decision
   D65: yes, provisionally, by the human on 18 November 2026; to be
   come back to. The CR at [free_of_heap_fine] says what such a
   verdict does not say.

   It is the guard that the liftings have for now: a limit of them
   pending decision D40, and not a property that a unit is meant to
   have. Nothing here justifies removing it while the heaps of two
   exits are compared by equality. [free_of_heap_fine] is a finer
   guard with a proof of its own; the measure of the units that it
   turns is in requests/C3-finer-guards.md. That the heaps come to be
   compared by a relation that holds of a unit and the unit lifted is
   one way for the guard to go, and is not said to be the only one.

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
(* A unit whose own run enters no code                                      *)
(* ======================================================================== *)

(* The expression has no [Apply]. Handlers, those of a recursive group
   too, are looked into; the code that a [Let_static] binds is not.
   Run from no continuation, an expression first enters code at an
   [Apply]: an [Apply_cont] enters code only through what is left of
   an over-application, which a call makes. So the run of such an
   expression on its own enters no code. *)
Fixpoint calls_nothing (e : expr) : bool :=
  match e with
  | Let _ _ _ body => calls_nothing body
  | Let_static _ body => calls_nothing body
  | Let_cont_non_recursive _ h body =>
    handler_calls_nothing h && calls_nothing body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_calls_nothing h) handlers
    && calls_nothing body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_calls_nothing (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => calls_nothing handler
  end.

(* [expr_free_of_heap] with every static binding counted as passing:
   the code that a [Let_static] binds is not looked into. *)
Fixpoint expr_free_of_heap_outside_code (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    negb (named_changes_heap d) && expr_free_of_heap_outside_code body
  | Let_static _ body => expr_free_of_heap_outside_code body
  | Let_cont_non_recursive _ h body =>
    handler_free_of_heap_outside_code h
    && expr_free_of_heap_outside_code body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_free_of_heap_outside_code h) handlers
    && expr_free_of_heap_outside_code body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_free_of_heap_outside_code (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => expr_free_of_heap_outside_code handler
  end.

(* CR mshinwell: [free_of_heap_fine] accepts a unit that has a
   primitive of the heap in the body of a function, where the unit has
   no [Apply] outside the bodies of its code. Its statement is true
   ([free_of_heap_fine_exits_empty_statement]): the run of such a unit
   enters no code, so it executes no such primitive. What is given
   away is not in the statement. The statements of the liftings are of
   the unit's own run, and for a unit whose run enters no code they
   say nothing of the bodies of its functions, with this guard or
   with the one before it. [expr_free_of_heap] is of the text of all
   the code of the unit, entered or not: with it a lifting is made,
   and a verdict may rest on it, only in a unit whose code has no
   primitive of the heap. The finer guard asks that only of a unit
   that has an [Apply] outside the bodies of its code. So a module
   of a library that has none, and no primitive of the heap outside
   those bodies, passes the guard though its functions store in
   cells; where the other checks of a lifting pass it may be lifted,
   and a verdict may rest on that lifting, on a run that calls none
   of those functions. The other answer is to keep
   [expr_free_of_heap] as the guard: no such lifting is then made,
   and a pair that needs it stays not validated unless another
   comparison succeeds, with the second check of
   FlambdaValidateRelatedSpec.v, which makes the liftings without the
   guard, for what it accepts. Decision D65: yes, provisionally, by
   the human on 18 November 2026; to be come back to. *)

(* The guard that the liftings ask: the unit has no primitive of the
   heap, or it calls nothing outside the bodies of its code and has no
   primitive of the heap outside them. *)
Definition free_of_heap_fine (e : expr) : bool :=
  expr_free_of_heap e
  || (calls_nothing e && expr_free_of_heap_outside_code e).

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

(* A unit that calls nothing outside the bodies of its code and has no
   primitive of the heap outside them, run on its own from the empty
   heap and from any static environment, exits with the empty heap:
   nothing is asked of the code that the unit or the static
   environment binds. *)
Definition calls_nothing_exits_empty_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (k : continuation) (vs : list value)
    (hp : heap),
    calls_nothing e = true ->
    expr_free_of_heap_outside_code e = true ->
    behaves (State e [] [] statics empty_heap no_reads) (Exits k vs hp) ->
    hp = empty_heap.

(* [free_of_heap_exits_empty_statement] of the guard that the liftings
   ask. The code of the static environment is asked to pass the test
   only where the unit passes by having no primitive of the heap
   anywhere; a unit that passes only by calling nothing needs nothing
   of it. *)
Definition free_of_heap_fine_exits_empty_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (k : continuation) (vs : list value)
    (hp : heap),
    free_of_heap_fine e = true ->
    (expr_free_of_heap e = true -> statics_free_of_heap statics = true) ->
    behaves (State e [] [] statics empty_heap no_reads) (Exits k vs hp) ->
    hp = empty_heap.

(* The test commutes with renaming: no name is looked at. *)
Definition expr_free_of_heap_rename_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    expr_free_of_heap (rename_expr r e) = expr_free_of_heap e.

(* So do the test of the calls, the test outside the bodies of code
   and the guard that the liftings ask. *)
Definition calls_nothing_rename_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    calls_nothing (rename_expr r e) = calls_nothing e.

Definition expr_free_of_heap_outside_code_rename_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    expr_free_of_heap_outside_code (rename_expr r e)
    = expr_free_of_heap_outside_code e.

Definition free_of_heap_fine_rename_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    free_of_heap_fine (rename_expr r e) = free_of_heap_fine e.
