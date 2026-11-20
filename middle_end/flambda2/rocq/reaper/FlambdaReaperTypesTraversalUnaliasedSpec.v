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

(* The graph of the traversal has [closures_unaliased]
   (FlambdaReaperTypesTraversalSpec.v), and with it the statement of the
   closure types of a set whose representation changes needs no condition
   on the graph.

   This file is to be reviewed: the statements, against
   FlambdaReaperTypesTraversalSpec.v. The proofs are in
   FlambdaReaperTypesTraversalUnaliasedProofs.v, which needs no review.

   - [closures_unaliased_traversal_statement]: the graph of the traversal
     of a unit of the flow fragment, well formed and binding each variable
     once, has [closures_unaliased]. The nodes held at a function slot are
     the variable of a [Let] of a closure, the variable of a projection
     that [set_closures] records, the symbol of a static closure and the
     nodes that [closures_of_set] makes. Into the variable of a [Let]
     only its binding puts something: nothing for a closure, the read of
     its slot for a projection. Into a symbol only the binding of a
     constant other than a closure puts something, or the delayed deps
     if it is imported. Into a node that [closures_of_set] makes nothing.
   - [changed_type_sound_full_statement]:
     FlambdaReaperTypesTraversalSpec.changed_type_sound_traversal_statement
     without [closures_unaliased]: the closure types of a set whose
     representation changes are rewritten soundly, for the graph of the
     traversal of such a unit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTraversalSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesCallsSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition closures_unaliased_traversal_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_flow_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    closures_unaliased (graph_of_facts (traverse_unit Auto u e)).

(* FlambdaReaperTypesTraversalSpec.changed_type_sound_traversal_statement
   without [closures_unaliased] of the graph. *)
Definition changed_type_sound_full_statement : Prop :=
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
