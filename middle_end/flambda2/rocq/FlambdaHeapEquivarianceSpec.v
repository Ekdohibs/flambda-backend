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

(* Renamings act on the states of the prototype machine of the heap
   (FlambdaHeapMachineSpec.v), and its execution is equivariant.

   This file is to be reviewed: the action ([rename_heap_state],
   [rename_outcome], [rename_heap_result], [rename_heap_behaviour]) and
   the statements.

   FlambdaHeapMachineSpec.v has the prototype of the heap: a machine
   of its own beside that of FlambdaSemanticsSpec.v, with states of
   its own ([heap_state]) and exits that carry the memory. The states
   of FlambdaSemanticsSpec.v have a heap too, and a read record, and
   its machine runs the primitives of the heap in a [Let];
   FlambdaEquivarianceSpec.v says that the renamings of variables and
   of continuations act on those states and that a step, a run and a
   behaviour commute with the action. Here is the same of the
   prototype.

   What the action does to memory: nothing. A location is a name of a
   sort of its own (requests/C4-heap.md, sections 2.2 and 5): no
   renaming of variables or of continuations moves one. A value holds
   locations, symbols and code ids, and no variable and no continuation;
   a cell of the heap holds values, and so do the extra arguments that a
   continuation keeps. So the heap of a state is left as it is, and so
   are the values and the heap of an outcome. The code of the static
   environment is renamed, as in a state of FlambdaSemanticsSpec.v,
   and an outcome carries a static environment, so its code is renamed
   too;
   what is observed of an outcome is read from its values, its
   statically allocated values and its heap
   ([outcome_related_equivariant_statement]: renaming the code of two
   outcomes, by any two renamings, does not change whether they are
   related).

   Two runs are compared in FlambdaHeapMachineSpec.v up to a
   correspondence between their locations ([outcome_related]). That is
   another relation than the renaming of bound names, on another sort
   of thing, and the two do not meet: the statements here are
   equations, for one run and its renaming, with the same locations on
   both sides. The action of the permutations of locations is in
   FlambdaLocationsSpec.v, and what the primitives and a step of
   FlambdaSemanticsSpec.v do with it is in
   FlambdaLocationsPrimitiveSpec.v and FlambdaLocationsMachineSpec.v;
   nothing is said there of a step of the machine of this file.

   The statements:
   - the action is that of a group, and it commutes with forgetting the
     heap;
   - a step, a run and a behaviour commute with it, with no premise;
   - so refinement of states, and of expressions under closed memories,
     is preserved by renaming;
   - a state of the prototype has a finite support, as a state of
     FlambdaSemanticsSpec.v has (FlambdaStateSupportSpec.v): with the
     action, the states of the prototype are a nominal set, and the
     heap adds nothing to the support.

   Nothing is trusted here. The proofs are in
   FlambdaHeapEquivarianceProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaAlphaSpec
  FlambdaNameAbstractionSpec FlambdaEquivarianceSpec FlambdaHeapSpec
  FlambdaHeapMachineSpec.
Import ListNotations.

(* ======================================================================== *)
(* The action                                                               *)
(* ======================================================================== *)

(* What is not the heap is renamed as in FlambdaEquivarianceSpec.v. The
   heap holds values, which hold no variables or continuations. *)
Definition rename_heap_state (r : renaming) (s : heap_state) : heap_state :=
  Heap_state (rename_expr r (hs_expr s)) (rename_env r (hs_vars s))
    (rename_cont_env r (hs_conts s)) (rename_static_env r (hs_statics s))
    (hs_heap s).

(* The values and the heap of an outcome hold no variables or
   continuations. Its static environment has code, which is renamed. *)
Definition rename_outcome (r : renaming) (o : outcome) : outcome :=
  Outcome (outcome_values o) (rename_static_env r (outcome_statics o))
    (outcome_heap o).

Definition rename_heap_result (r : renaming) (result : heap_step_result)
    : heap_step_result :=
  match result with
  | Heap_step s => Heap_step (rename_heap_state r s)
  | Heap_exit k o => Heap_exit (rename_continuation r k) (rename_outcome r o)
  | Heap_undefined => Heap_undefined
  end.

Definition rename_heap_behaviour (r : renaming) (b : heap_behaviour)
    : heap_behaviour :=
  match b with
  | Heap_exits k o => Heap_exits (rename_continuation r k) (rename_outcome r o)
  | Heap_goes_wrong => Heap_goes_wrong
  | Heap_diverges => Heap_diverges
  end.

Definition rename_heap_state_identity_statement : Prop :=
  ∀ s : heap_state, rename_heap_state identity_renaming s = s.

Definition rename_heap_state_compose_statement : Prop :=
  ∀ (second first : renaming) (s : heap_state),
    rename_heap_state (compose_renaming second first) s
    = rename_heap_state second (rename_heap_state first s).

Definition rename_heap_state_inverse_statement : Prop :=
  ∀ (r : renaming) (s : heap_state),
    rename_heap_state (inverse_renaming r) (rename_heap_state r s) = s
    ∧ rename_heap_state r (rename_heap_state (inverse_renaming r) s) = s.

(* The action is that on the states of FlambdaSemanticsSpec.v, with
   the heap carried along: [forget_heap] gives such a state, with the
   heap of the state and no reads. *)
Definition forget_heap_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : heap_state),
    forget_heap (rename_heap_state r s) = rename_state r (forget_heap s).

(* Finitely many variables and continuations support a state: a renaming
   that fixes them leaves the state as it is. They are those that
   support the state that [forget_heap] gives. *)
Definition heap_state_finitely_supported_statement : Prop :=
  ∀ s : heap_state,
    ∃ (vs : list variable) (ks : list continuation),
      ∀ r : renaming,
        fixes (variable_permutation r) vs ->
        fixes (continuation_permutation r) ks ->
        rename_heap_state r s = s.

(* ======================================================================== *)
(* Execution is equivariant                                                 *)
(* ======================================================================== *)

Definition heap_step_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : heap_state),
    heap_step (rename_heap_state r s) = rename_heap_result r (heap_step s).

Definition heap_run_equivariant_statement : Prop :=
  ∀ (r : renaming) (fuel : nat) (s : heap_state),
    heap_run fuel (rename_heap_state r s)
    = rename_heap_result r (heap_run fuel s).

Definition heap_behaves_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : heap_state) (b : heap_behaviour),
    heap_behaves (rename_heap_state r s) (rename_heap_behaviour r b)
    <-> heap_behaves s b.

(* ======================================================================== *)
(* Refinement is preserved                                                  *)
(* ======================================================================== *)

(* Whether two outcomes are related is read from their values, their
   statically allocated values and their heaps: the code that they carry
   may be renamed, on each side as one likes. *)
Definition outcome_related_equivariant_statement : Prop :=
  ∀ (poison_refines : bool) (r r' : renaming) (o o' : outcome),
    outcome_related poison_refines (rename_outcome r o)
      (rename_outcome r' o')
    <-> outcome_related poison_refines o o'.

Definition heap_refines_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : heap_state),
    heap_refines source target ->
    heap_refines (rename_heap_state r source) (rename_heap_state r target).

(* A rewrite that is correct with one choice of names is correct with
   every other. *)
Definition heap_expr_refines_equivariant_statement : Prop :=
  ∀ (r : renaming) (source target : expr),
    heap_expr_refines source target ->
    heap_expr_refines (rename_expr r source) (rename_expr r target).
