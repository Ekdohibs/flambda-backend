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


(* The removal of unused value slots, of a unit that something follows.

   This file is to be reviewed: the statement.

   FlambdaUnusedSlotsSpec.drop_unused_slots removes value slots from
   the static sets of closures of a unit that projects none of them,
   and is stated of the unit run on its own and from its imports
   (FlambdaUnusedSlotsSpec.v). Here the unit is followed by something:
   the unit and the unit without the slots are run from one
   continuation environment of what may follow a unit
   (FlambdaObserversSpec.observers), from the static environment of the
   imports and the empty heap. What may follow is taken at the slots
   removed: FlambdaObserversSpec.observer asks of a handler that it
   project none of the slots that it is given, so no slot removed is
   read after the unit either. A closure that the unit hands on may be
   a symbol bound to a static set, and what follows may project a slot
   of it that is kept.

   The second run refines the first, with the same exits
   ([drop_unused_slots_refines_observed_statement]): what follows is
   the same term in the two runs, it has no static set of closures to
   remove a slot from, and no value changes, as a value does not hold
   what a symbol is bound to. With no continuation it is
   FlambdaUnusedSlotsSpec.drop_unused_slots_refines_imports_statement.
   The two runs then come to results that differ in nothing that an
   exit is compared by (FlambdaObserversSpec.refines_under_observers,
   at the slots removed): FlambdaUnusedSlotsObservedProofs.v has that
   as a lemma.

   The statement is in a file of its own because
   FlambdaObserversSpec.v says what an observer is with the check of
   FlambdaUnusedSlotsSpec.v ([slots_unused]).

   Whether this is what the verdict on a unit whose value slots are
   removed is to mean has not been decided (FlambdaObserversSpec.v). No
   function of the validation changes.

   The proofs are in FlambdaUnusedSlotsObservedProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaImportsSpec FlambdaUnusedSlotsSpec
  FlambdaObserversSpec.
Import ListNotations.

(* What [drop_unused_slots] gives refines the unit, the two run from one
   continuation environment of what may follow a unit, none of which
   projects a slot removed, and from the static environment of the
   imports. *)
Definition drop_unused_slots_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (dropped : list value_slot)
    (conts : cont_env) (e e' : expr),
    drop_unused_slots dropped e = Some e' ->
    observers dropped conts ->
    refines (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State e' [] conts (imports_statics imports) empty_heap no_reads).
