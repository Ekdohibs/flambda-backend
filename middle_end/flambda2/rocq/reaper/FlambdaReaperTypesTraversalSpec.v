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
   [changed_closures_one_allocation], [closures_unaliased],
   [check_closures_unaliased], [StoredExample] and the statements, against
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
   - [block_fields_of_one_kind_traversal_statement]: the graph of the
     traversal of any unit has [block_fields_of_one_kind]: only
     [block_deps] makes a constructor at a field of a block, of kind
     value.
   - [changed_closures_one_allocation_unaliased_statement]:
     [changed_closures_one_allocation] of a least solution, for a graph
     whose function slots are of sets and mutual, and whose closures are
     put into only by the constructors of their sets and by reads of
     their slot off a closure that holds them there
     ([closures_unaliased]). What the rules derive then aliases into a
     closure only itself, gives it no [any_source] and no
     [reading_field] and itself as its one source.
   - [check_closures_unaliased_sound_statement]: a test of
     [closures_unaliased] on the facts; it passes on the examples of
     FlambdaReaperUnboxing.v with a set of closures
     ([closures_unaliased_examples_statement]).
   - [changed_type_sound_traversal_statement]:
     [rewrite_changed_type_sound_traversal_statement] with
     [block_fields_of_one_kind] discharged, and [closures_unaliased] of
     the graph asked in place of [changed_closures_one_allocation].
   - [changed_unstored_counterexample_statement]: [changed_unstored] does
     not follow from the tables. In [StoredExample], a unit of the
     premises of the statements, the closure [x60] is stored in a block,
     and [to_change_representation] holds of it: the rules of
     unboxing_analysis.ml keep from a change of representation an
     allocation that escapes or whose uses may have other sources, not
     one that is stored. No fault is found: the rebuilt unit stores the
     closure in its new representation, and the type of the field is
     rewritten under the metadata of the field, which has that
     representation (not checked with the compiler). The premise is one
     of the third run of the statements, whose fields hold what those of
     the second run hold ([changed_at], [changed_closure_rebuilt]);
     without it, that run would change the closures held in fields too.

   What is left out: [closures_unaliased] of the graph of the traversal.
   It should hold of a unit of the flow fragment that binds each
   variable once: nothing but a constructor puts anything into the
   variable of a [Let] of a closure, a symbol of a static closure or a
   node made for a closure of a set ([closures_of_set]), and into the
   variable of a projection that [set_closures] records nothing but its
   read; the check passes on the examples. A proof needs an invariant of
   the traversal on the variables of the closures and on the nodes made
   for them, as that of FlambdaReaperUnusedCodeTraversalProofs.v on the
   [my_closure] nodes of the code deps. And [changed_unstored]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec.
From OxCamlFlambda2 Require FlambdaReaperUnboxing FlambdaReaperUnboxingSpec
  FlambdaReaperUnboxingRefinesCallsSpec FlambdaReaperLeast
  FlambdaReaperCallingConventions.
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

(* The closures of a graph, the nodes held at a function slot, are put
   into only by the constructors of their sets and by reads of their
   function slot off a closure that holds them there: no fact aliases into
   one, gives it a source or makes it a parameter. *)
Definition closures_unaliased (G : graph) : Prop :=
  ∀ (x b : node) (g : function_slot),
    g_constructor G x (Field_function_slot g) b ->
    (∀ from : node, ¬ g_alias G b from)
    ∧ (∀ c from : node, ¬ g_propagate G c b from)
    ∧ (∀ c from : node, ¬ g_alias_if_any_source G c b from)
    ∧ (∀ (base : node) (i : nat), ¬ g_parameter G base i b)
    ∧ ¬ g_any_source G b ∧ ¬ g_zero_alloc_source G b
    ∧ (∀ c : node, ¬ g_any_source_if_used G c b)
    ∧ (∀ from : node, ¬ g_use G b from)
    ∧ ∀ (f : field) (base : node),
        g_accessor G b f base ->
        ∃ h : function_slot,
          f = Field_function_slot h
          ∧ g_constructor G base (Field_function_slot h) b.

(* [closures_unaliased] of the graph of facts, as a test: for each node
   held at a function slot, no fact but a constructor and a read of a
   function slot off a node that holds it there puts something into it. *)
Definition check_closures_unaliased (F : facts) : bool :=
  forallb
    (fun '(_, fs, b) =>
       negb (is_function_slot fs)
       || (forallb (fun '(t, _) => negb (node_eqb t b)) (f_alias F)
           && forallb (fun '(_, t, _) => negb (node_eqb t b)) (f_propagate F)
           && forallb (fun '(_, t, _) => negb (node_eqb t b))
                (f_alias_if_any_source F)
           && forallb (fun '(_, _, t) => negb (node_eqb t b)) (f_parameter F)
           && negb (mem_node b (f_any_source F))
           && negb (mem_node b (f_zero_alloc_source F))
           && forallb (fun '(_, t) => negb (node_eqb t b))
                (f_any_source_if_used F)
           && forallb (fun '(t, _) => negb (node_eqb t b)) (f_use F)
           && forallb
                (fun '(t, f, base) =>
                   negb (node_eqb t b)
                   || (is_function_slot f
                       && existsb
                            (fun '(base', f', from) =>
                               node_eqb base' base && field_eqb f' f
                               && node_eqb from b)
                            (f_constructor F)))
                (f_accessor F)))
    (f_constructor F).

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

(* The fields of blocks of the graph of the traversal are all of kind
   value ([block_deps]), for any unit. *)
Definition block_fields_of_one_kind_traversal_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr),
    block_fields_of_one_kind (graph_of_facts (traverse_unit flag u e)).

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

Definition changed_closures_one_allocation_unaliased_statement : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables) (tcr : list node),
    least_solution is_local F T ->
    function_slots_of_sets (graph_of_facts F) ->
    function_slots_mutual (graph_of_facts F) ->
    closures_unaliased (graph_of_facts F) ->
    changed_closures_one_allocation (graph_of_facts F) (solution_of_tables T)
      tcr.

Definition check_closures_unaliased_sound_statement : Prop :=
  ∀ F : facts,
    check_closures_unaliased F = true -> closures_unaliased (graph_of_facts F).

(* The check passes on the units of FlambdaReaperUnboxing.Examples with a
   set of closures. *)
Definition closures_unaliased_examples_statement : Prop :=
  check_closures_unaliased FlambdaReaperUnboxing.Examples.block_in_slot_facts
  = true
  ∧ check_closures_unaliased
      FlambdaReaperUnboxing.Examples.unboxed_closure_facts
    = true
  ∧ check_closures_unaliased FlambdaReaperUnboxing.Examples.static_set_facts
    = true.

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

(* [rewrite_changed_type_sound_traversal_statement] with
   [block_fields_of_one_kind] discharged, and [closures_unaliased], a
   condition on the graph, in place of [changed_closures_one_allocation],
   a condition on the solution. *)
Definition changed_type_sound_traversal_statement : Prop :=
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
    closures_unaliased (graph_of_facts (traverse_unit Auto u e)) ->
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

(* ======================================================================== *)
(* A changed closure stored in a block                                      *)
(* ======================================================================== *)

(* [changed_unstored] does not follow from the tables of the rules: the
   closure [x60] of a set whose code 53 returns its value slot 80 is
   stored in field 0 of the block [x10], read back into [x11] and called.
   The block is unboxed; the closure, called and not lambda-lifted, is not
   unboxed, and nothing stops the change of its representation, which
   the derived table of [to_change_representation] has. *)
Module StoredExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static [Bind_code 53 code_53]
    (Let 60 Value (Closure_of_set 70 [(70, 53)] [(80, int 7)])
       (Let 10 Value (make_block [Var 60])
          (Let 11 Value (load 0 (Var 10)) (Apply (call 53 (Var 11)))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

(* The least solution, as the model computes it. *)
Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 100 with
  | Some T => T
  | None => Tables [] [] [] [] [] [] [] [] [] []
  end.

Definition dominated : list (node * node) :=
  FlambdaReaperUnboxing.derive_dominated tables.

Definition reprs : list (node * FlambdaReaperUnboxing.closure_repr) :=
  FlambdaReaperUnboxing.unit_changed_representations no_local facts tables
    dominated [Node_variable 10] [Node_variable 60] unit_expr.

End StoredExample.

(* The unit is of the premises of the statements, its graph passes
   [check_closures_unaliased], its tables are least, [x10] is the one
   allocation to unbox, the table [[x60]] of [to_change_representation]
   passes the checks, and [changed_unstored] is false. *)
Definition changed_unstored_counterexample_statement : Prop :=
  in_flow_fragment StoredExample.unit_expr = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      StoredExample.unit_expr
    = true
  ∧ variables_bound_once StoredExample.unit_expr = true
  ∧ check_closures_unaliased StoredExample.facts = true
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local
      StoredExample.facts StoredExample.tables
  ∧ FlambdaReaperCallingConventions.derive_cannot_change StoredExample.facts
      StoredExample.tables
    = []
  ∧ FlambdaReaperUnboxing.derive_to_unbox
      FlambdaReaperUnboxing.Examples.no_local StoredExample.facts
      StoredExample.tables [] FlambdaReaperUnboxing.Examples.max_unbox_size
      false
    = [Node_variable 10%positive]
  ∧ FlambdaReaperUnboxing.check_to_change_representation
      FlambdaReaperUnboxing.Examples.no_local StoredExample.facts
      StoredExample.tables [] FlambdaReaperUnboxing.Examples.max_unbox_size
      false [Node_variable 10%positive] [Node_variable 60%positive]
    = true
  ∧ FlambdaReaperUnboxing.check_sets_whole StoredExample.facts
      StoredExample.tables [Node_variable 10%positive]
      [Node_variable 60%positive]
    = true
  ∧ ¬ changed_unstored (graph_of_facts StoredExample.facts)
        (solution_of_tables StoredExample.tables) StoredExample.reprs.
