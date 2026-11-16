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

(* The value slots of a static set of closures that nothing projects are
   removed.

   This file is to be reviewed: [drop_slots], against what Simplify does
   (below); [slots_unused], the check; [drop_unused_slots], which is
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; and the statements.

   What Simplify does. Where it binds the symbols of a set of closures,
   one that was static or one that it has just lifted
   (FlambdaLiftSpec.v), it leaves out of the set every value slot that
   no projection uses: simplify/expr_builder.ml,
   [remove_unused_value_slots], called by [create_let_symbols], which
   keeps a slot if [Name_occurrences.value_slot_is_used_or_imported] says
   so of the value slots used in projections ([UA.used_value_slots]). A
   projection is counted where it is simplified and stays a projection
   (simplify/simplify_unary_primitive.ml,
   [simplify_project_value_slot], [DA.add_use_of_value_slot]), in the
   whole unit, on the way down; so a slot may be kept that no projection
   of the term rebuilt uses. A set of closures that is not static keeps
   its value slots here. This happens to every module that exports a
   function which calls another function of the module: the first is a
   value slot of the closure of the second, read by its code, and once
   the call is made direct, or inlined, nothing reads it.

   What is here. [drop_slots dropped e] removes the value slots
   [dropped] from every static set of closures that [e] binds, wherever
   the binding is. It is right only if no projection of such a slot is
   executed. For a unit run on its own it is enough that the unit has no
   projection of one at all, in code and in handlers too, which is what
   [slots_unused dropped e] checks: every closure that the run reads is
   then one that the unit made, and every projection one of the unit's.
   Under other environments the term may be given a closure whose code,
   bound elsewhere, projects the slot; so this is no rewrite under all
   environments, and it is said of a unit run on its own
   ([drop_unused_slots_program_refines_statement]).

   No value changes: a value does not hold what a symbol is bound to,
   and a closure that is not static keeps its slots. So nothing is asked
   of what the unit exits with.

   The model has no imported value slots (a unit refers to no other),
   so [dropped] is any list of value slots. Which slots is a guess of
   the caller's, read off the term after Simplify; nothing is believed
   of it. A slot that Simplify kept, though nothing projects it, is not
   in the guess, and stays.

   It refines and is not an equivalence: a slot whose contents have no
   value (a variable that is not bound) makes the binding undefined
   before and not after.

   The proofs are in FlambdaUnusedSlotsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaUnusedStaticsSpec
  FlambdaImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The value slots that are not among [dropped], in their order. *)
Definition keep_slots {A : Type} (dropped : list value_slot)
    (slots : list (value_slot * A)) : list (value_slot * A) :=
  filter (fun slot => negb (among (fst slot) dropped)) slots.

(* A static set of closures without the value slots [dropped]. *)
Definition drop_slots_static_const (dropped : list value_slot)
    (const : static_const) : static_const :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    Static_const_set_of_closures code (keep_slots dropped value_slots)
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    Static_const_closure_of_set slot closure_symbols function_decls
      (keep_slots dropped value_slots)
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    const
  end.

(* Every static set of closures that the expression binds, in handlers
   and in code too, without the value slots [dropped]. *)
Fixpoint drop_slots (dropped : list value_slot) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (drop_slots dropped body)
  | Let_static bindings body =>
    Let_static (map (drop_slots_binding dropped) bindings)
      (drop_slots dropped body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (drop_slots_handler dropped h)
      (drop_slots dropped body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, drop_slots_handler dropped h)) handlers)
      (drop_slots dropped body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with drop_slots_handler (dropped : list value_slot)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (drop_slots dropped handler)
  end
with drop_slots_binding (dropped : list value_slot)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (drop_slots_code dropped c)
  | Bind_deleted_code _ => binding
  | Bind_symbol s const =>
    Bind_symbol s (drop_slots_static_const dropped const)
  end
with drop_slots_code (dropped : list value_slot) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (drop_slots dropped body)
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The defining expression projects no value slot of [dropped]. *)
Definition named_projects_none (dropped : list value_slot) (d : named)
    : bool :=
  match d with
  | Project_value_slot slot _ | Project_value_slot_from _ slot _ =>
    negb (among slot dropped)
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

(* The expression has no projection of a value slot of [dropped].
   Handlers and code are looked into. *)
Fixpoint slots_unused (dropped : list value_slot) (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    named_projects_none dropped d && slots_unused dropped body
  | Let_static bindings body =>
    forallb (binding_slots_unused dropped) bindings
    && slots_unused dropped body
  | Let_cont_non_recursive _ h body =>
    handler_slots_unused dropped h && slots_unused dropped body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_slots_unused dropped h) handlers
    && slots_unused dropped body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_slots_unused (dropped : list value_slot)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => slots_unused dropped handler
  end
with binding_slots_unused (dropped : list value_slot)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_slots_unused dropped c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_slots_unused (dropped : list value_slot) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => slots_unused dropped body
  end.

(* The unit without the value slots [dropped] in its static sets of
   closures, if it projects none of them. *)
Definition drop_unused_slots (dropped : list value_slot) (e : expr)
    : option expr :=
  if slots_unused dropped e then Some (drop_slots dropped e) else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [drop_unused_slots] gives refines the unit, run on its own. *)
Definition drop_unused_slots_program_refines_statement : Prop :=
  ∀ (dropped : list value_slot) (e e' : expr),
    drop_unused_slots dropped e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports. Nothing more is asked: an import is no
   closure, so none has a slot to remove, and the rewrite moves no
   binding. *)
Definition drop_unused_slots_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (dropped : list value_slot)
    (e e' : expr),
    drop_unused_slots dropped e = Some e' -> unit_refines imports e e'.

(* The rewrite makes no projection and removes none: a term passes the
   check before it if and only if it does after. *)
Definition drop_slots_slots_unused_statement : Prop :=
  ∀ (dropped dropped' : list value_slot) (e : expr),
    slots_unused dropped' (drop_slots dropped e) = slots_unused dropped' e.

(* The rewrite and the check commute with renaming, which renames no
   value slot. *)
Definition drop_slots_equivariant_statement : Prop :=
  ∀ (r : renaming) (dropped : list value_slot) (e : expr),
    rename_expr r (drop_slots dropped e) = drop_slots dropped (rename_expr r e).

Definition slots_unused_equivariant_statement : Prop :=
  ∀ (r : renaming) (dropped : list value_slot) (e : expr),
    slots_unused dropped (rename_expr r e) = slots_unused dropped e.
