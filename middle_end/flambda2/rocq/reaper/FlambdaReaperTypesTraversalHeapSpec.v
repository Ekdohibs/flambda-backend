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

(* The rewriting of types with the unboxing and the changes of
   representation of closures, for the units with primitives of the heap
   (FlambdaReaperFlowSpec.in_heap_fragment), which
   FlambdaReaperTypesSpec.v, FlambdaReaperTypesTraversalSpec.v and
   FlambdaReaperTypesTraversalUnaliasedSpec.v leave out: their relations
   of the values of the runs have none for a location, and the
   statements of the traversal are of [in_flow_fragment].

   This file is to be reviewed: the statements, against those of the
   files above and of FlambdaReaperTypesHeapSpec.v, whose relation of
   the values of two runs with the heap ([rebuilt_value_heap],
   [statics_rebuilt_heap]) they take. The proofs are in
   FlambdaReaperTypesTraversalHeapProofs.v, which needs no review.

   The runs. The statements of the unboxing and of the changes of
   representation compare three runs: that of the unit, that of the unit
   rebuilt without unboxing, and that of the unit rebuilt with it. The
   first two are related as in FlambdaReaperTypesHeapSpec.v: a location
   of the first is rebuilt as the location that a correspondence [β] of
   the locations of the two runs gives it. The last two are related as
   without the heap: the values of the fields of an unboxed variable are
   those at the paths of the fields in the value of the variable
   ([unboxed_values]), and the value slots of a closure in its new
   representation are those at the paths of the new slots
   ([changed_at]), locations included: the two rebuilds are taken to
   have one heap. The simulation of the local unboxing with the heap has
   them so (FlambdaReaperUnboxingRefinesHeapProofs.relatedh: the same
   primitives of the heap run on the same values, and what the unboxing
   removes is an immutable block or a boxed number, not a cell). That of
   the changes of representation
   (FlambdaReaperUnboxingRefinesClosuresSimProofs.v) is of the flow
   fragment; a change of representation changes a closure, not a cell.

   What the proofs without the heap need of [in_flow_fragment]: of the
   traversal, only that the unit has no older form of set of closures
   (FlambdaReaperClosuresExactProofs.fragment_no_old_sets, of either
   fragment); of the meaning of types, nothing, as it reads no heap
   (FlambdaReaperTypesHeapSpec.type_satisfied_erase_statement).

   The statements:
   - [function_slots_mutual_traversal_heap_statement],
     [function_slots_of_sets_traversal_heap_statement] and
     [closures_unaliased_traversal_heap_statement]: the properties of
     the graph of the traversal that FlambdaReaperTypesTraversalSpec.v
     and FlambdaReaperTypesTraversalUnaliasedSpec.v give a unit of the
     flow fragment, well formed and binding each variable once, for the
     units of [in_heap_fragment] so: the traversal of a primitive of the
     heap makes no constructor at a function slot and puts nothing into
     a closure, as the invariants of the traversal in their proofs hold
     of every step. ([block_fields_of_one_kind_traversal_statement] is of
     any unit.)
   - [rewrite_changed_type_sound_slots_heap_statement]:
     FlambdaReaperTypesSpec.rewrite_changed_type_sound_slots_statement,
     the closure types of a set whose representation changes, with the
     first two runs related by [rebuilt_value_heap] and
     [statics_rebuilt_heap] for some [β].
   - [rewrite_unboxed_result_type_sound_heap_statement] and
     [rewrite_unaliased_unboxed_result_type_sound_heap_statement]: the
     statements of FlambdaReaperTypesSpec.v of the types of the new
     parameters and results of an unboxed variable of code, with the
     first two runs so related; with [_local] in their names, for a
     least solution of the rules with the fix of local fields.
   - [rewrite_changed_type_sound_traversal_heap_statement],
     [changed_type_sound_traversal_heap_statement] and
     [changed_type_sound_full_heap_statement]: the statements of
     FlambdaReaperTypesTraversalSpec.v and
     FlambdaReaperTypesTraversalUnaliasedSpec.v of the closure types of a
     set whose representation changes, for the graph of the traversal of
     a unit of [in_heap_fragment], well formed and binding each variable
     once, with the first two runs so related. The last asks of the
     graph only [changed_unstored], as without the heap.
   - [unbox_types_end_to_end_heap_statement]: the statement of the runs
     of FlambdaReaperTypesHeapSpec.types_end_to_end_term_heap_statement,
     of the unit rebuilt with local unboxing
     (FlambdaReaperUnboxing.rebuild_unit_unbox, of the fragment of
     FlambdaReaperUnboxingRefinesSpec.v, [local_unboxing] and
     [check_unboxed_reads]) in place of the plain rebuild: where the unit
     exits, the unit rebuilt with unboxing exits through the same
     continuation, the exits observed equal up to a correspondence of
     their locations, and the type of each symbol bound, rewritten, holds
     of the symbol there. The local unboxing changes no representation
     of a closure, so the types are those of the plain rewriting
     ([plain_context]). [unaliased_unbox_types_end_to_end_heap_statement]:
     the same for the fragment [loose]; with [_local] in their names, for
     a least solution of the rules with the fix of local fields.
   - [unbox_types_heap_example_statement]: a unit of [in_heap_fragment],
     not of [in_flow_fragment], of the premises of the last statements,
     whose least solution unboxes a block holding a cell
     (FlambdaReaperUnboxingRefinesHeapSpec.HeapExample under the binding
     of a static block, which it returns): the premises are met with an
     unboxing.
   - Each statement of types has a form for the fixes to land
     (FlambdaReaperTypes.landed_fixes), with [_landed]; four names are
     shortened: [unaliased_unboxed_result_type_sound_heap_landed_statement],
     [unboxed_result_type_sound_heap_local_landed_statement],
     [unaliased_unboxed_result_sound_heap_local_landed_statement] and
     [unaliased_unbox_end_to_end_heap_local_landed_statement].
   No counterexample: the statements hold.

   What is left out: as without the heap, the relation of the values of
   the runs at a return of code, which the statements of result types
   take as a premise, and [changed_unstored]; the simulations of the
   unboxing across calls and closures and of the changes of
   representation with the heap, whose runs the statements of result
   types and of closure types are to be applied to, as
   [unbox_types_end_to_end_heap_statement] applies the statement of
   symbols to the runs of the local unboxing; the meaning of
   [Mutable_block] (FlambdaReaperTypesHeapSpec.v); and cells that the
   rules unbox (FlambdaReaperUnboxingRefinesHeapSpec.cell_example), of
   which the rebuild with unboxing binds the fields to the values the
   cell was made with: [unboxed_values] has no path into a location, so
   the statements of result types say nothing of such a variable. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec
  FlambdaReaperTypesHeapSpec FlambdaReaperTypesTraversalSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec FlambdaReaperUnboxing
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesHeapSpec FlambdaReaperLeast
  FlambdaReaperCallingConventions.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* The graph of the traversal                                               *)
(* ======================================================================== *)

Definition function_slots_mutual_traversal_heap_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    function_slots_mutual (graph_of_facts (traverse_unit Auto u e)).

Definition function_slots_of_sets_traversal_heap_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    function_slots_of_sets (graph_of_facts (traverse_unit Auto u e)).

Definition closures_unaliased_traversal_heap_statement : Prop :=
  ∀ (u : unit_interface) (e : expr),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    closures_unaliased (graph_of_facts (traverse_unit Auto u e)).

(* ======================================================================== *)
(* The statements of a graph and its tables                                 *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.rewrite_changed_type_sound_slots_with, with the
   values and the static environments of the first two runs related by
   [rebuilt_value_heap] and [statics_rebuilt_heap]. *)
Definition rewrite_changed_type_sound_slots_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (n : node)
    (v v_mid v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    block_fields_of_one_kind (graph_of_facts F) ->
    function_slots_mutual (graph_of_facts F) ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    FlambdaReaperUnboxing.check_dominated T dominated = true ->
    FlambdaReaperUnboxingRefinesCallsSpec.check_unboxed_uses F T
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields F T dominated to_unbox
              e))
    = true ->
    changed_unstored (graph_of_facts F) (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local F T
         dominated to_unbox tcr e) ->
    reprs_of_sets (graph_of_facts F)
      (FlambdaReaperUnboxing.unit_changed_representations is_local F T
         dominated to_unbox tcr e) ->
    type_in_fragment_with true valuation t ->
    closure_in_set statics v <> None ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v n
      v_mid ->
    metadata_at (solution_of_tables T) m n ->
    changed_at (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local F T
         dominated to_unbox tcr e)
      n v_mid v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary_u fx
         (Rewrite_context F T is_local sets
            (map fst
               (FlambdaReaperUnboxing.unit_changed_representations is_local F
                  T dominated to_unbox tcr e)))
         tenv canon (FlambdaReaperUnboxing.bound_code_my_closures e)
         (FlambdaReaperUnboxing.unit_changed_representations is_local F T
            dominated to_unbox tcr e)
         fuel m t).

Definition rewrite_changed_type_sound_slots_heap_statement : Prop :=
  rewrite_changed_type_sound_slots_heap_with all_fixes.

Definition rewrite_changed_type_sound_slots_heap_landed_statement : Prop :=
  rewrite_changed_type_sound_slots_heap_with landed_fixes.

(* FlambdaReaperTypesSpec.rewrite_unboxed_result_type_sound_with, with the
   first two runs so related, for a least solution of the rules with or
   without the fix of local fields ([local]), of the fragment [loose] or
   not ([loose], the static sets being kept whole). *)
Definition unboxed_result_type_sound_heap_with (local loose : bool)
    (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (β : location -> location -> Prop)
    (code_var type_var : variable)
    (fields : FlambdaReaperUnboxing.unboxed_fields) (v v_mid : value)
    (ws : list value),
    (if local then least_solution_local is_local F T
     else least_solution is_local F T) ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    block_fields_of_one_kind (graph_of_facts F) ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    (loose = true ->
     static_sets_whole (solution_of_tables T) statics statics') ->
    copies_valued tenv canon valuation valuation' ->
    FlambdaReaperUnboxingRefinesCallsSpec.check_unboxed_uses F T
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields F T dominated to_unbox
              e))
    = true ->
    Keyed.lookup node_eqb (Node_variable code_var)
      (fst (FlambdaReaperUnboxing.unit_unboxed_fields F T dominated to_unbox
              e))
    = Some fields ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v_mid ->
    unboxed_values fields v_mid ws ->
    Forall2 (type_satisfied statics' valuation') ws
      (rewrite_unboxed_result_type fx (plain_context is_local F T sets) tenv
         (FlambdaReaperUnboxing.bound_code_my_closures e) canon code_var
         type_var fields).

Definition rewrite_unboxed_result_type_sound_heap_statement : Prop :=
  unboxed_result_type_sound_heap_with false false all_fixes.

Definition rewrite_unboxed_result_type_sound_heap_landed_statement : Prop :=
  unboxed_result_type_sound_heap_with false false landed_fixes.

Definition rewrite_unaliased_unboxed_result_type_sound_heap_statement : Prop :=
  unboxed_result_type_sound_heap_with false true all_fixes.

Definition unaliased_unboxed_result_type_sound_heap_landed_statement : Prop :=
  unboxed_result_type_sound_heap_with false true landed_fixes.

Definition rewrite_unboxed_result_type_sound_heap_local_statement : Prop :=
  unboxed_result_type_sound_heap_with true false all_fixes.

Definition unboxed_result_type_sound_heap_local_landed_statement : Prop :=
  unboxed_result_type_sound_heap_with true false landed_fixes.

Definition unaliased_unboxed_result_type_sound_heap_local_statement : Prop :=
  unboxed_result_type_sound_heap_with true true all_fixes.

Definition unaliased_unboxed_result_sound_heap_local_landed_statement : Prop :=
  unboxed_result_type_sound_heap_with true true landed_fixes.

(* ======================================================================== *)
(* The statements of the traversal of a unit                                *)
(* ======================================================================== *)

(* FlambdaReaperTypesTraversalSpec.rewrite_changed_type_sound_traversal_with
   of the units of [in_heap_fragment], with the first two runs related by
   [rebuilt_value_heap] and [statics_rebuilt_heap]. *)
Definition rewrite_changed_type_sound_traversal_heap_with (fx : fixes)
    : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (n : node)
    (v v_mid v' : value),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts (traverse_unit Auto u e)) sets ->
    block_fields_of_one_kind (graph_of_facts (traverse_unit Auto u e)) ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β statics statics' ->
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
    rebuilt_value_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β v n v_mid ->
    metadata_at (solution_of_tables T) m n ->
    changed_at (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local
         (traverse_unit Auto u e) T dominated to_unbox tcr e)
      n v_mid v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary_u fx
         (Rewrite_context (traverse_unit Auto u e) T is_local sets
            (map fst
               (FlambdaReaperUnboxing.unit_changed_representations is_local
                  (traverse_unit Auto u e) T dominated to_unbox tcr e)))
         tenv canon (FlambdaReaperUnboxing.bound_code_my_closures e)
         (FlambdaReaperUnboxing.unit_changed_representations is_local
            (traverse_unit Auto u e) T dominated to_unbox tcr e)
         fuel m t).

Definition rewrite_changed_type_sound_traversal_heap_statement : Prop :=
  rewrite_changed_type_sound_traversal_heap_with all_fixes.

Definition rewrite_changed_type_sound_traversal_heap_landed_statement : Prop :=
  rewrite_changed_type_sound_traversal_heap_with landed_fixes.

(* FlambdaReaperTypesTraversalSpec.changed_type_sound_traversal_with of
   the units of [in_heap_fragment], the first two runs so related. *)
Definition changed_type_sound_traversal_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (n : node)
    (v v_mid v' : value),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts (traverse_unit Auto u e)) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β statics statics' ->
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
    closures_unaliased (graph_of_facts (traverse_unit Auto u e)) ->
    type_in_fragment_with true valuation t ->
    closure_in_set statics v <> None ->
    rebuilt_value_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β v n v_mid ->
    metadata_at (solution_of_tables T) m n ->
    changed_at (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local
         (traverse_unit Auto u e) T dominated to_unbox tcr e)
      n v_mid v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary_u fx
         (Rewrite_context (traverse_unit Auto u e) T is_local sets
            (map fst
               (FlambdaReaperUnboxing.unit_changed_representations is_local
                  (traverse_unit Auto u e) T dominated to_unbox tcr e)))
         tenv canon (FlambdaReaperUnboxing.bound_code_my_closures e)
         (FlambdaReaperUnboxing.unit_changed_representations is_local
            (traverse_unit Auto u e) T dominated to_unbox tcr e)
         fuel m t).

Definition changed_type_sound_traversal_heap_statement : Prop :=
  changed_type_sound_traversal_heap_with all_fixes.

Definition changed_type_sound_traversal_heap_landed_statement : Prop :=
  changed_type_sound_traversal_heap_with landed_fixes.

(* FlambdaReaperTypesTraversalUnaliasedSpec.changed_type_sound_full_with
   of the units of [in_heap_fragment], the first two runs so related:
   no condition on the graph but [changed_unstored]. *)
Definition changed_type_sound_full_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (e : expr) (tenv : Typing_env.t)
    (canon : canon_fn) (statics statics' : static_env)
    (valuation valuation' : env) (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (n : node)
    (v v_mid v' : value),
    in_heap_fragment e = true -> well_formed_unit u e = true ->
    variables_bound_once e = true ->
    least_solution is_local (traverse_unit Auto u e) T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts (traverse_unit Auto u e)) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β statics statics' ->
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
    rebuilt_value_heap (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T) β v n v_mid ->
    metadata_at (solution_of_tables T) m n ->
    changed_at (solution_of_tables T)
      (FlambdaReaperUnboxing.unit_changed_representations is_local
         (traverse_unit Auto u e) T dominated to_unbox tcr e)
      n v_mid v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary_u fx
         (Rewrite_context (traverse_unit Auto u e) T is_local sets
            (map fst
               (FlambdaReaperUnboxing.unit_changed_representations is_local
                  (traverse_unit Auto u e) T dominated to_unbox tcr e)))
         tenv canon (FlambdaReaperUnboxing.bound_code_my_closures e)
         (FlambdaReaperUnboxing.unit_changed_representations is_local
            (traverse_unit Auto u e) T dominated to_unbox tcr e)
         fuel m t).

Definition changed_type_sound_full_heap_statement : Prop :=
  changed_type_sound_full_heap_with all_fixes.

Definition changed_type_sound_full_heap_landed_statement : Prop :=
  changed_type_sound_full_heap_with landed_fixes.

(* ======================================================================== *)
(* The exported types of the unit rebuilt with local unboxing               *)
(* ======================================================================== *)

(* FlambdaReaperTypesHeapSpec.types_end_to_end_term_heap_with, of the unit
   rebuilt with local unboxing in place of the plain rebuild, for a least
   solution of the rules with the fix of local fields if [local] (whose
   check replaces the premise that no field is local or that no slot is
   projected), of the fragment [loose] if [loose]. *)
Definition unbox_types_end_to_end_heap_with (local loose : bool)
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (code_deps : finite_map code_dep)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (if local then least_solution_local is_local (traverse_unit flag u e) T
     else
       least_solution is_local (traverse_unit flag u e) T
       ∧ ((∀ f : field, is_local f = false) ∨ projection_free e = true)) ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    FlambdaReaperUnboxingRefinesSpec.local_unboxing (traverse_unit flag u e)
      T dominated to_unbox e
    = true ->
    FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      (traverse_unit flag u e) T dominated to_unbox
    = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with loose valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (FlambdaReaperUnboxing.rebuild_unit_unbox (traverse_unit flag u e)
              T cannot_change code_deps dominated to_unbox
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unbox_types_end_to_end_heap_statement : Prop :=
  unbox_types_end_to_end_heap_with false false all_fixes.

Definition unbox_types_end_to_end_heap_landed_statement : Prop :=
  unbox_types_end_to_end_heap_with false false landed_fixes.

Definition unaliased_unbox_types_end_to_end_heap_statement : Prop :=
  unbox_types_end_to_end_heap_with false true all_fixes.

Definition unaliased_unbox_types_end_to_end_heap_landed_statement : Prop :=
  unbox_types_end_to_end_heap_with false true landed_fixes.

Definition unbox_types_end_to_end_heap_local_statement : Prop :=
  unbox_types_end_to_end_heap_with true false all_fixes.

Definition unbox_types_end_to_end_heap_local_landed_statement : Prop :=
  unbox_types_end_to_end_heap_with true false landed_fixes.

Definition unaliased_unbox_types_end_to_end_heap_local_statement : Prop :=
  unbox_types_end_to_end_heap_with true true all_fixes.

Definition unaliased_unbox_end_to_end_heap_local_landed_statement : Prop :=
  unbox_types_end_to_end_heap_with true true landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlambdaReaperUnboxingRefinesHeapSpec.HeapExample under the
   binding of a static block [s90], which it returns:

     let s90 = [3] in
     let x10 = mutable { 1 } in
     let x11 = [x10] in
     let x12 = x11.(0) in
     let x13 = x12.(0) <- 2 in
     let x14 = x10.(0) in              (a mutable load, with no use)
     return s90

   The least solution, as [compute_least] gives it, unboxes [x11], whose
   field the rebuild binds to the variable 15. *)
Module UnboxTypesExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition access : block_access_kind :=
  FlambdaReaperUnboxingRefinesHeapSpec.HeapExample.access.

Definition unit_expr : expr :=
  Let_static [Bind_symbol 90 (Static_const_block 0 [int 3])]
    (Let 10 Value
       (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
       (Let 11 Value (make_block [Var 10])
          (Let 12 Value (load 0 (Var 11))
             (Let 13 Value
                (Prim (Heap (Block_set access Assignment 0)) [Var 12; int 2])
                (Let 14 Value
                   (Prim (Unary (Block_load access Mutable 0)) [Var 10])
                   (return_to 1 [Symbol 90])))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 100 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition dominated : list (node * node) :=
  FlambdaReaperUnboxing.derive_dominated tables.

Definition to_unbox : list node :=
  FlambdaReaperUnboxing.derive_to_unbox no_local facts tables cannot_change
    max_unbox_size false.

Definition rebuilt : expr :=
  Let_static [Bind_symbol 90 (Static_const_block 0 [int 3])]
    (Let 10 Value
       (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
       (Let 15 Value (Simple (Var 10))
          (Let 12 Value (Simple (Var 15))
             (Let 13 Value
                (Prim (Heap (Block_set access Assignment 0)) [Var 12; int 2])
                (return_to 1 [Symbol 90]))))).

Definition s90 : statically_allocated_value :=
  Static_block 0 [Value_const (Tagged_immediate 3)].

Definition heap_at_exit : heap :=
  FlambdaReaperUnboxingRefinesHeapSpec.HeapExample.heap_at_exit.

(* The state from which the unit exits. *)
Definition exit_state : state :=
  State (return_to 1 [Symbol 90])
    [(14, Value_const (Tagged_immediate 2));
     (13, Value_const (Tagged_immediate 0)); (12, Value_location 2);
     (11, Value_block 0 [Value_location 2]); (10, Value_location 2)]
    [] {| static_code := []; static_values := [(90, s90)] |} heap_at_exit
    no_reads.

End UnboxTypesExample.

(* The unit meets the premises of [unbox_types_end_to_end_heap_statement]
   on the unit (all but those on the typing environment), with its least
   solution and no local field; the rules unbox [x11], and the unit
   rebuilt with unboxing is [rebuilt]. The unit exits from [exit_state],
   where [s90] is bound, and so does [rebuilt], with the same heap. *)
Definition unbox_types_heap_example_statement : Prop :=
  in_heap_fragment UnboxTypesExample.unit_expr = true
  ∧ in_flow_fragment UnboxTypesExample.unit_expr = false
  ∧ FlambdaReaperRebuildSpec.any_kind_arguments UnboxTypesExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.results_kinded [] [] UnboxTypesExample.unit_expr
    = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      UnboxTypesExample.unit_expr
    = true
  ∧ FlambdaReaperRebuildSpec.exits_one_value
      FlambdaReaperUnboxing.Examples.interface UnboxTypesExample.unit_expr
    = true
  ∧ variables_bound_once UnboxTypesExample.unit_expr = true
  ∧ unit_of_values UnboxTypesExample.unit_expr = true
  ∧ is_local_slots_only FlambdaReaperUnboxing.Examples.no_local
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local
      UnboxTypesExample.facts UnboxTypesExample.tables
  ∧ (∀ f : field, FlambdaReaperUnboxing.Examples.no_local f = false)
  ∧ sets_of_graph (graph_of_facts UnboxTypesExample.facts) []
  ∧ UnboxTypesExample.to_unbox = [Node_variable 11%positive]
  ∧ FlambdaReaperUnboxingRefinesSpec.local_unboxing UnboxTypesExample.facts
      UnboxTypesExample.tables UnboxTypesExample.dominated
      UnboxTypesExample.to_unbox UnboxTypesExample.unit_expr
    = true
  ∧ FlambdaReaperUnboxingRefinesSpec.check_unboxed_reads
      UnboxTypesExample.facts UnboxTypesExample.tables
      UnboxTypesExample.dominated UnboxTypesExample.to_unbox
    = true
  ∧ FlambdaReaperUnboxing.rebuild_unit_unbox UnboxTypesExample.facts
      UnboxTypesExample.tables UnboxTypesExample.cannot_change
      (FlambdaReaperCallingConventions.unit_code_deps Auto
         FlambdaReaperUnboxing.Examples.interface UnboxTypesExample.unit_expr)
      UnboxTypesExample.dominated UnboxTypesExample.to_unbox true
      UnboxTypesExample.unit_expr
    = UnboxTypesExample.rebuilt
  ∧ run 6 (State UnboxTypesExample.unit_expr [] [] empty_static_env
             empty_heap no_reads)
    = Step UnboxTypesExample.exit_state
  ∧ step UnboxTypesExample.exit_state
    = Exit 1%positive [Value_symbol 90%positive]
        UnboxTypesExample.heap_at_exit no_reads
  ∧ lookup 90%positive
      (static_values (state_statics UnboxTypesExample.exit_state))
    = Some UnboxTypesExample.s90
  ∧ run 10 (State UnboxTypesExample.rebuilt [] [] empty_static_env
              empty_heap no_reads)
    = Exit 1%positive [Value_symbol 90%positive]
        UnboxTypesExample.heap_at_exit no_reads.
