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

(* Failure sites of the fix that the model meets with a fallback, never
   taken: a function slot of a set whose representation changes that the
   representation has no new slot for. rebuild.ml stops there
   ([rewrite_set_of_closures], "Could not find rewritten function slot";
   slot_offsets_analysis.ml likewise; [Function_slot.Map.find] of the
   current slot in [rewrite_field_access_chg_repr] and
   [load_field_from_value_which_is_being_unboxed] raises [Not_found]);
   the model keeps the slot ([rename_function_slot], [current_slot]).

   This file is to be reviewed: the statement. The proofs are in
   FlambdaReaperFixFailsSlotsProofs.v.

   [changed_decls_all_renamed_statement]: of every [Let] of a closure of
   a unit of the flow fragment, well formed and binding each variable
   once, with a least solution and a table [dominated] that passes its
   check, if the changes of representation give its variable a
   representation, that representation has a new slot for every
   function slot of the declarations of the set. The premises are those
   of [changed_slot_current_statement]
   (FlambdaReaperClosuresTablesSpec.v), whose proof this one follows:
   the representation is that of the set of a closure of [tcr], whose
   function slots are those of the [Let].

   [changed_current_renamed_statement]: every representation that
   [unit_changed_representations] makes has a new slot for its current
   function slot, for any tables: the current slot is one of the set
   whose function slots it renames. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperRebuild FlambdaReaperUnboxing.
Import ListNotations.

Definition changed_decls_all_renamed_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (x : variable) (slot : function_slot)
    (decls : list (function_slot * code_id))
    (vs : list (value_slot * simple)) (r : closure_repr),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    check_dominated T dominated = true ->
    (x, Closure_of_set slot decls vs) ∈ let_bindings e ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_changed_representations is_local (traverse_unit Auto u e) T
         dominated to_unbox tcr e)
    = Some r ->
    ∀ g : function_slot,
      is_bound g decls = true -> is_bound g (cr_function_slots r) = true.

Definition changed_current_renamed_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr)
    (n : node) (r : closure_repr),
    Keyed.lookup node_eqb n
      (unit_changed_representations is_local F T dominated to_unbox tcr e)
    = Some r ->
    is_bound (cr_current r) (cr_function_slots r) = true.
