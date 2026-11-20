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

(* The premises of the statement of the closure types of a set whose
   representation changes
   (FlambdaReaperTypesSpec.rewrite_changed_type_sound_slots_statement), of
   the graph of the traversal and of the representations of the unit.

   This file is to be reviewed: [function_slots_of_sets],
   [changed_closures_one_allocation] and the statements, against
   FlambdaReaperTypesSpec.v, FlambdaReaperTraversal.v (the traversal) and
   FlambdaReaperUnboxing.v ([changed_representations]). The proofs are in
   FlambdaReaperTypesTraversalProofs.v, which needs no review.

   The statements:
   - [function_slots_mutual_traversal_statement]: the graph of the
     traversal of a unit of the flow fragment, well formed and binding
     each variable once, has [function_slots_mutual]. The allocations of
     the set of a [Let] of a closure are those it holds at its function
     slots ([record_set_of_closures_deps]). A static set is bound whole:
     the symbol that a binding holds at a function slot is bound to a
     closure of the same set, whose symbols include the first
     (FlambdaWellFormedSpec.static_set_whole).
   - [function_slots_of_sets_traversal_statement]: of the same units, the
     function slots of the graph are those of sets
     ([function_slots_of_sets]): an allocation holds one closure at a
     function slot, and a closure at one function slot only, and the
     closures it holds hold the same function slots as it does.
   - [reprs_of_sets_derived_statement]: [reprs_of_sets] of the
     representations of the unit, derived from their construction, for a
     graph whose function slots are of sets and mutual, a least solution
     whose table of [dominated_by_allocation_point] is checked, and the
     closures of the sets of [tcr] each of one allocation
     ([changed_closures_one_allocation]). Without the last,
     [reprs_of_sets] can fail: a closure with a source other than itself
     is dominated by nothing, and so has no representation, where the
     other closures of its set have one.
   - [rewrite_changed_type_sound_traversal_statement]:
     [rewrite_changed_type_sound_slots_statement] for the graph of the
     traversal of such a unit, with [function_slots_mutual] and
     [reprs_of_sets] discharged; [changed_closures_one_allocation] is
     asked in place of [reprs_of_sets].

   What is left out: [changed_closures_one_allocation] of the least
   solution of the graph of the traversal (a closure of the flow fragment
   that binds each variable once should have no other source: nothing
   aliases into the variable of its [Let] or into its symbol, and the
   projection that binds another closure of its set reads only that
   closure from the first), [block_fields_of_one_kind] of the graph of
   the traversal (its fields of blocks are all of kind value), and
   [changed_unstored]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing FlambdaReaperUnboxingSpec
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaBindingTime FlambdaTypingEnv
  FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* The function slots of a graph are those of sets: an allocation holds
   one closure at a function slot, a closure at one function slot only,
   and the closures it holds at its function slots hold the same function
   slots as it does, as the allocations of a set of closures do. *)
Definition function_slots_of_sets (G : graph) : Prop :=
  (∀ (a b b' : node) (g : function_slot),
     g_constructor G a (Field_function_slot g) b ->
     g_constructor G a (Field_function_slot g) b' -> b = b')
  ∧ (∀ (a b : node) (g g' : function_slot),
       g_constructor G a (Field_function_slot g) b ->
       g_constructor G a (Field_function_slot g') b -> g = g')
  ∧ ∀ (a b c : node) (g h : function_slot),
      g_constructor G a (Field_function_slot g) b ->
      g_constructor G a (Field_function_slot h) c
      <-> g_constructor G b (Field_function_slot h) c.

(* The closures of the sets of the allocations of the table [tcr] of
   [to_change_representation] each come from one allocation, which can
   only be itself: [multiple_allocation_points] holds of none. *)
Definition changed_closures_one_allocation (G : graph) (sol : solution)
    (tcr : list node) : Prop :=
  ∀ (x b : node) (g : function_slot),
    x ∈ tcr -> g_constructor G x (Field_function_slot g) b ->
    ¬ FlambdaReaperUnboxingSpec.multiple_allocation_points_of sol b.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition function_slots_mutual_traversal_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    function_slots_mutual (graph_of_facts (traverse_unit Auto u e)).

Definition function_slots_of_sets_traversal_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    function_slots_of_sets (graph_of_facts (traverse_unit Auto u e)).

Definition reprs_of_sets_derived_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) (e : expr),
    least_solution is_local F T ->
    function_slots_of_sets (graph_of_facts F) ->
    function_slots_mutual (graph_of_facts F) ->
    FlambdaReaperUnboxing.check_dominated T dominated = true ->
    changed_closures_one_allocation (graph_of_facts F) (solution_of_tables T)
      tcr ->
    reprs_of_sets (graph_of_facts F)
      (FlambdaReaperUnboxing.unit_changed_representations is_local F T
         dominated to_unbox tcr e).

(* [rewrite_changed_type_sound_slots_statement] for the graph of the
   traversal of a unit [e] of the flow fragment, well formed and binding
   each variable once: [function_slots_mutual] holds, and
   [reprs_of_sets] follows from [changed_closures_one_allocation]. *)
Definition rewrite_changed_type_sound_traversal_statement : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (fuel : nat) (m : rewrite_metadata)
    (t : flambda_type) (n : node) (v v_mid v' : value),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts (traverse_unit Auto u e)) sets ->
    block_fields_of_one_kind (graph_of_facts (traverse_unit Auto u e)) ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) statics statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    FlambdaReaperUnboxing.check_dominated T dominated = true ->
    FlambdaReaperUnboxingRefinesCallsSpec.check_unboxed_uses
      (traverse_unit Auto u e) T
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields
              (traverse_unit Auto u e) T dominated to_unbox e))
    = true ->
    changed_unstored (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local
         (traverse_unit Auto u e) T dominated to_unbox tcr e) ->
    changed_closures_one_allocation (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) tcr ->
    type_in_fragment_with true valuation t ->
    closure_in_set statics v <> None ->
    rebuilt_value (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) v n v_mid ->
    metadata_at (solution_of_tables T) m n ->
    changed_at (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local
         (traverse_unit Auto u e) T dominated to_unbox tcr e)
      n v_mid v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary_u all_fixes
         (Rewrite_context (traverse_unit Auto u e) T is_local sets
            (map fst
               (FlambdaReaperUnboxing.unit_changed_representations is_local
                  (traverse_unit Auto u e) T dominated to_unbox tcr e)))
         tenv canon (FlambdaReaperUnboxing.bound_code_my_closures e)
         (FlambdaReaperUnboxing.unit_changed_representations is_local
            (traverse_unit Auto u e) T dominated to_unbox tcr e)
         fuel m t).
