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


(* The rewriting of types for the units with primitives of the heap
   (FlambdaReaperFlowSpec.in_heap_fragment), which
   FlambdaReaperTypesSpec.v leaves out: its relation of the values of
   the two runs ([rebuilt_value]) has none for a location, and its
   statements of the two runs are of [in_flow_fragment].

   This file is to be reviewed: [erase_locations], [rebuilt_value_heap],
   [static_rebuilt_heap], [statics_rebuilt_heap], [rebuilt_under_heap],
   [locations_one_to_one], [rebuild_exit_relation_heap] and
   [rebuild_exit_relation_whole_heap], and the statements, against
   FlambdaReaperTypesSpec.v, FlambdaReaperTypesTermSpec.v and
   FlambdaReaperRebuildSpec.v ([exits_observed_equal]). The proofs are in
   FlambdaReaperTypesHeapProofs.v, which needs no review.

   How the types speak of a value of the heap. The meaning of types
   (FlambdaTypeMeaningSpec.type_satisfied) reads a value through
   [block_of], [boxed_number_of], [closure_in_set],
   [project_function_slot] and [string_of] of the static environment,
   none of which reads the heap; a location is none of these, and
   [Mutable_block], the head of a mutable block, holds of no value. So
   a type holds of a location as of a value of kind value that is not
   null and has nothing to show, but for an alias, which names it. The
   immutable array of the one location, which nothing that types read
   tells apart, is such a value: [type_satisfied_erase_statement] says
   that a type holds of a value exactly as it holds of the value with
   each location so replaced ([erase_locations]), in the static
   environment and the valuation so replaced. The rewriting keeps the
   head of a mutable block ([Rule.identity] keeps [Mutable_block] as it
   is) or forgets it, and rewrites an alias to the copy of its name. So
   the rewritten types speak of a location only through aliases, and the
   fragment of FlambdaReaperTypesSpec.v ([type_in_fragment]) has no
   alias to a name that holds one.

   What the rebuild makes of a location ([rebuilt_value_heap]): the
   location that a correspondence [β] of the locations of the two runs
   gives it; otherwise as [rebuilt_value]. The correspondence of the
   simulation (FlambdaReaperRebuildProofs, [kept_beta]) relates the
   locations of the allocations that the rebuild keeps, one to one
   ([locations_one_to_one]); a location in a field that a use reads has
   its allocation kept, its cell related to that of the second run.

   The statements:
   - [type_satisfied_erase_statement]: above.
   - [rewrite_type_sound_heap_statement],
     [rewrite_symbol_type_sound_heap_statement],
     [rewrite_result_type_sound_heap_statement] and those of the
     fragment [loose] ([rewrite_unaliased_type_sound_heap_statement],
     [rewrite_unaliased_symbol_type_sound_heap_statement],
     [rewrite_unaliased_result_type_sound_heap_statement]): those of
     FlambdaReaperTypesSpec.v without [_heap], with the values and the
     static environments related by [rebuilt_value_heap] for some [β].
   - [rebuild_exit_relation_heap_holds_statement]: what the statement
     of the two runs asks of the simulation, for the units of
     [in_heap_fragment]: where the unit exits, the rebuilt unit exits
     through the same continuation, the two exits observed equal up to
     a correspondence of their locations
     (FlambdaReaperRebuildSpec.exits_observed_equal, as
     [rebuild_refines_up_to_heap_statement] has them), and the static
     environments related by [statics_rebuilt_heap] for a correspondence
     of locations one to one. Its premises are those of
     FlambdaReaperTypesSpec.rebuild_exit_relation_holds_statement, with
     [in_heap_fragment] in place of [in_flow_fragment].
     [rebuild_exit_relation_whole_heap_holds_statement]: the same with
     the static sets kept whole.
   - [exported_types_sound_heap_statement] and
     [exported_unaliased_types_sound_heap_statement]: the statements of
     the two runs given those relations.
   - [exported_types_sound_end_to_end_heap_statement] and
     [unaliased_types_end_to_end_heap_statement]: with no premise on the
     relation of the runs; [types_end_to_end_term_heap_statement] and
     [unaliased_types_end_to_end_term_heap_statement]: with the check of
     the term (FlambdaReaperTypesTermSpec.unit_of_values) in place of
     [exits_of_values].
   - Each of these has a form for the fixes to land, with [_landed], and
     one with local fields, with [_local], as in FlambdaReaperTypesSpec.v:
     for a least solution of the rules with the fix
     ([least_solution_local]) where the statement is of a graph, with the
     check of the fix where it is of a unit; two names are shortened:
     [unaliased_symbol_type_sound_heap_local_statement] and
     [unaliased_result_type_sound_heap_local_statement], with their
     [_landed].
   No counterexample: the statements hold.

   What is left out: a meaning of [Mutable_block], which the meaning of
   types is to have with a heap (FlambdaTypeMeaningSpec.v, workstream
   C4): no valuation satisfies a typing environment that gives it to a
   name with a value, and the statements say nothing of such an
   environment. With a meaning that reads the cell of a location, the
   argument is to go through: [β] relates only locations whose allocation
   the rebuild keeps, to cells of the same mutability and tag. Also left
   out: the unboxing and the changes of representation with the heap
   (FlambdaReaperTypesTraversalSpec.v), whose simulation is of
   [in_flow_fragment]. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveSpec FlambdaSemanticsSpec
  FlambdaWellFormedSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperRebuild FlambdaReaperFlowSpec
  FlambdaReaperLeastSpec FlambdaReaperLocalFields FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesTermSpec.
From OxCamlFlambda2 Require FlambdaReaperRebuildSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

(* ======================================================================== *)
(* Locations seen by the types                                              *)
(* ======================================================================== *)

(* A value with each location [l] replaced by the immutable array of
   [l]: a value of kind value that is no block, boxed number, closure or
   string, as a location is, and two values are one exactly when the
   values they replace are. *)
Fixpoint erase_locations (v : value) : value :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ => v
  | Value_closure code slots =>
    Value_closure code (map (fun '(w, x) => (w, erase_locations x)) slots)
  | Value_closure_of_set slot decls slots =>
    Value_closure_of_set slot decls
      (map (fun '(w, x) => (w, erase_locations x)) slots)
  | Value_block tag fields => Value_block tag (map erase_locations fields)
  | Value_immutable_array elements =>
    Value_immutable_array (map erase_locations elements)
  | Value_location l => Value_immutable_array [Value_location l]
  | Value_partial_application callee args arity =>
    Value_partial_application (erase_locations callee)
      (map erase_locations args) arity
  end.

Definition erase_slots (slots : list (value_slot * value))
    : list (value_slot * value) :=
  map (fun '(w, x) => (w, erase_locations x)) slots.

Definition erase_static (a : statically_allocated_value)
    : statically_allocated_value :=
  match a with
  | Static_closure code slots => Static_closure code (erase_slots slots)
  | Static_closure_of_set slot syms decls slots =>
    Static_closure_of_set slot syms decls (erase_slots slots)
  | Static_block tag fields => Static_block tag (map erase_locations fields)
  | Static_boxed_number contents =>
    Static_boxed_number (erase_locations contents)
  | Static_immutable_array elements =>
    Static_immutable_array (map erase_locations elements)
  | Static_string _ | Static_imported _ => a
  end.

Definition erase_statics (statics : static_env) : static_env :=
  Static_env (static_code statics)
    (map (fun '(s, a) => (s, erase_static a)) (static_values statics)).

Definition erase_env (vars : env) : env :=
  map (fun '(x, v) => (x, erase_locations v)) vars.

(* ======================================================================== *)
(* What the rebuild makes of a value, with the heap                         *)
(* ======================================================================== *)

Section Rebuilt_heap.

Variable G : graph.
Variable sol : solution.
Variable β : location -> location -> Prop.

(* [rebuilt_value], with a location rebuilt as the location that [β]
   gives it. *)
Fixpoint rebuilt_value_heap (v : value) (n : node) (v' : value) {struct v}
    : Prop :=
  match v with
  | Value_const _ => s_any_source sol n = true ∧ v' = v
  | Value_symbol s =>
    v' = v ∧ may_come_from sol (Node_symbol s) n
    ∧ (s_has_usage sol n = true -> s_has_usage sol (Node_symbol s) = true)
  | Value_block tag fields =>
    ∃ (a : node) (fields' : list value),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ v' = Value_block tag fields'
      ∧ fields_rebuilt
          (fun i field field' =>
             ∃ from : node,
               g_constructor G a (Field_block i (kind_of_value field)) from
               ∧ (s_field_used sol a (Field_block i (kind_of_value field))
                  = true ->
                  s_has_usage sol from = true
                  ∧ rebuilt_value_heap field from field')
               ∧ (s_field_used sol a (Field_block i (kind_of_value field))
                  = false ->
                  field' = Value_const (Poison (kind_of_value field))))
          0 fields fields'
  | Value_boxed_number c =>
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from sol a n ∧ s_has_source sol a = true
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ v' = (if s_field_used sol a (Field_boxed_number bn) then v
              else Value_const (Poison Value))
  | Value_immutable_array _ =>
    ∃ elements' : list value, v' = Value_immutable_array elements'
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      v' = Value_closure_of_set slot function_decls value_slots'
      ∧ may_come_from sol (sites slot) n
      ∧ set_sites G function_decls sites
      ∧ for_all_bindings
          (fun w x =>
             ∀ (f : function_slot) (id : code_id),
               lookup f function_decls = Some id ->
               s_field_used sol (sites f) (Field_value_slot w) = true ->
               ∃ (from : node) (x' : value),
                 slot_held G function_decls sites w from
                 ∧ s_has_source sol from = true
                 ∧ s_has_usage sol from = true
                 ∧ lookup w value_slots' = Some x'
                 ∧ rebuilt_value_heap x from x')
          value_slots
  | Value_partial_application _ _ args_arity =>
    ∃ (closure' : value) (args' : list value),
      v' = Value_partial_application closure' args' args_arity
  | Value_location l =>
    ∃ l' : location, v' = Value_location l' ∧ β l l'
  | Value_closure _ _ => False
  end.

(* [static_rebuilt], with [rebuilt_value_heap]. *)
Definition static_rebuilt_heap (s : symbol)
    (allocated : statically_allocated_value)
    (found : option statically_allocated_value) : Prop :=
  match allocated with
  | Static_block tag fields =>
    ∃ fields' : list value,
      found = Some (Static_block tag fields')
      ∧ fields_rebuilt
          (field_rebuilt G sol rebuilt_value_heap (Node_symbol s)) 0 fields
          fields'
  | Static_boxed_number _ | Static_string _ | Static_imported _ =>
    found = Some allocated
  | Static_immutable_array _ =>
    ∃ elements' : list value, found = Some (Static_immutable_array elements')
  | Static_closure_of_set slot closure_symbols function_decls value_slots =>
    ∃ (sites : function_slot -> node)
      (value_slots' : list (value_slot * value)),
      found
      = Some
          (Static_closure_of_set slot closure_symbols function_decls
             value_slots')
      ∧ sites slot = Node_symbol s
      ∧ set_sites G function_decls sites
      ∧ for_all_bindings
          (value_slot_rebuilt G sol rebuilt_value_heap function_decls sites
             value_slots')
          value_slots
  | Static_closure _ _ => False
  end.

Definition statics_rebuilt_heap (statics statics' : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    s_has_usage sol (Node_symbol s) = true ->
    static_rebuilt_heap s allocated (lookup s (static_values statics')).

(* [rebuilt_under], with [rebuilt_value_heap]. *)
Definition rebuilt_under_heap (m : rewrite_metadata) (v v' : value) : Prop :=
  (∃ c : const, v = Value_const c ∧ v' = v)
  ∨ ∃ n : node,
      rebuilt_value_heap v n v' ∧ metadata_at sol m n
      ∧ (forgets m = false -> s_has_usage sol n = true).

End Rebuilt_heap.

(* [β] relates a location to one at most, and one to a location at
   most. *)
Definition locations_one_to_one (β : location -> location -> Prop) : Prop :=
  (∀ l l1 l2 : location, β l l1 -> β l l2 -> l1 = l2)
  ∧ (∀ l1 l2 l' : location, β l1 l' -> β l2 l' -> l1 = l2).

(* ======================================================================== *)
(* The meaning of types and the locations                                   *)
(* ======================================================================== *)

Definition type_satisfied_erase_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (v : value) (t : flambda_type),
    type_satisfied (erase_statics statics) (erase_env vars)
      (erase_locations v) t
    <-> type_satisfied statics vars v t.

(* ======================================================================== *)
(* The statements of a graph and its tables                                 *)
(* ======================================================================== *)

Definition rewrite_type_sound_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment valuation t ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).

Definition rewrite_type_sound_heap_statement : Prop :=
  rewrite_type_sound_heap_with all_fixes.

Definition rewrite_type_sound_heap_landed_statement : Prop :=
  rewrite_type_sound_heap_with landed_fixes.

Definition rewrite_symbol_type_sound_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (s : symbol) (allocated : statically_allocated_value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).

Definition rewrite_symbol_type_sound_heap_statement : Prop :=
  rewrite_symbol_type_sound_heap_with all_fixes.

Definition rewrite_symbol_type_sound_heap_landed_statement : Prop :=
  rewrite_symbol_type_sound_heap_with landed_fixes.

Definition rewrite_result_type_sound_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (code_var type_var : variable) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).

Definition rewrite_result_type_sound_heap_statement : Prop :=
  rewrite_result_type_sound_heap_with all_fixes.

Definition rewrite_result_type_sound_heap_landed_statement : Prop :=
  rewrite_result_type_sound_heap_with landed_fixes.

(* The fragment [loose], the static sets of closures kept whole. *)
Definition rewrite_unaliased_type_sound_heap_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment_with true valuation t ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).

Definition rewrite_unaliased_type_sound_heap_statement : Prop :=
  rewrite_unaliased_type_sound_heap_with all_fixes.

Definition rewrite_unaliased_type_sound_heap_landed_statement : Prop :=
  rewrite_unaliased_type_sound_heap_with landed_fixes.

Definition rewrite_unaliased_symbol_type_sound_heap_with (fx : fixes)
    : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (s : symbol) (allocated : statically_allocated_value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).

Definition rewrite_unaliased_symbol_type_sound_heap_statement : Prop :=
  rewrite_unaliased_symbol_type_sound_heap_with all_fixes.

Definition rewrite_unaliased_symbol_type_sound_heap_landed_statement : Prop :=
  rewrite_unaliased_symbol_type_sound_heap_with landed_fixes.

Definition rewrite_unaliased_result_type_sound_heap_with (fx : fixes)
    : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (code_var type_var : variable) (v v' : value),
    least_solution is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).

Definition rewrite_unaliased_result_type_sound_heap_statement : Prop :=
  rewrite_unaliased_result_type_sound_heap_with all_fixes.

Definition rewrite_unaliased_result_type_sound_heap_landed_statement : Prop :=
  rewrite_unaliased_result_type_sound_heap_with landed_fixes.

(* With local fields: the statements above for a least solution of the
   rules with the fix of local fields. *)
Definition rewrite_type_sound_heap_local_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment valuation t ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).

Definition rewrite_type_sound_heap_local_statement : Prop :=
  rewrite_type_sound_heap_local_with all_fixes.

Definition rewrite_type_sound_heap_local_landed_statement : Prop :=
  rewrite_type_sound_heap_local_with landed_fixes.

Definition rewrite_symbol_type_sound_heap_local_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (s : symbol) (allocated : statically_allocated_value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).

Definition rewrite_symbol_type_sound_heap_local_statement : Prop :=
  rewrite_symbol_type_sound_heap_local_with all_fixes.

Definition rewrite_symbol_type_sound_heap_local_landed_statement : Prop :=
  rewrite_symbol_type_sound_heap_local_with landed_fixes.

Definition rewrite_result_type_sound_heap_local_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (code_var type_var : variable) (v v' : value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).

Definition rewrite_result_type_sound_heap_local_statement : Prop :=
  rewrite_result_type_sound_heap_local_with all_fixes.

Definition rewrite_result_type_sound_heap_local_landed_statement : Prop :=
  rewrite_result_type_sound_heap_local_with landed_fixes.

Definition rewrite_unaliased_type_sound_heap_local_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) (v v' : value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    type_in_fragment_with true valuation t ->
    rebuilt_under_heap (graph_of_facts F) (solution_of_tables T) β m v v' ->
    type_satisfied statics valuation v t ->
    type_satisfied statics' valuation' v'
      (rewrite_arbitrary fx (plain_context is_local F T sets) tenv canon fuel
         m t).

Definition rewrite_unaliased_type_sound_heap_local_statement : Prop :=
  rewrite_unaliased_type_sound_heap_local_with all_fixes.

Definition rewrite_unaliased_type_sound_heap_local_landed_statement : Prop :=
  rewrite_unaliased_type_sound_heap_local_with landed_fixes.

Definition rewrite_unaliased_symbol_type_sound_heap_local_with (fx : fixes)
    : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (s : symbol) (allocated : statically_allocated_value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    lookup s (static_values statics) = Some allocated ->
    type_satisfied statics' valuation' (Value_symbol s)
      (rewrite_symbol_type fx (plain_context is_local F T sets) tenv canon s).

Definition unaliased_symbol_type_sound_heap_local_statement : Prop :=
  rewrite_unaliased_symbol_type_sound_heap_local_with all_fixes.

Definition unaliased_symbol_type_sound_heap_local_landed_statement : Prop :=
  rewrite_unaliased_symbol_type_sound_heap_local_with landed_fixes.

Definition rewrite_unaliased_result_type_sound_heap_local_with (fx : fixes)
    : Prop :=
  ∀ (is_local : field -> bool) (F : facts) (T : tables)
    (sets : list set_of_closures) (tenv : Typing_env.t) (canon : canon_fn)
    (statics statics' : static_env) (valuation valuation' : env)
    (β : location -> location -> Prop)
    (code_var type_var : variable) (v v' : value),
    least_solution_local is_local F T ->
    is_local_slots_only is_local ->
    sets_of_graph (graph_of_facts F) sets ->
    typing_env_satisfied_strictly statics valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    statics_allocated (solution_of_tables T) statics ->
    statics_rebuilt_heap (graph_of_facts F) (solution_of_tables T) β statics
      statics' ->
    static_sets_whole (solution_of_tables T) statics statics' ->
    copies_valued tenv canon valuation valuation' ->
    Typing_env.find tenv (Name_var type_var) None <> None ->
    lookup type_var valuation = Some v ->
    rebuilt_value_heap (graph_of_facts F) (solution_of_tables T) β v
      (Node_variable code_var) v' ->
    type_satisfied statics' valuation' v'
      (rewrite_result_type fx (plain_context is_local F T sets) tenv canon
         code_var type_var).

Definition unaliased_result_type_sound_heap_local_statement : Prop :=
  rewrite_unaliased_result_type_sound_heap_local_with all_fixes.

Definition unaliased_result_type_sound_heap_local_landed_statement : Prop :=
  rewrite_unaliased_result_type_sound_heap_local_with landed_fixes.

(* ======================================================================== *)
(* The two runs                                                             *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.rebuild_exit_relation, with the heap: where the
   unit [e] exits, every symbol bound is an allocation of its own, and
   the rebuilt unit [e'] exits through the same continuation, the exits
   observed equal up to a correspondence of their locations, from a
   state whose static environment [statics_rebuilt_heap] relates to
   that of the unit for a correspondence [β] one to one. *)
Definition rebuild_exit_relation_heap (G : graph) (sol : solution)
    (e e' : expr) : Prop :=
  ∀ (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    statics_allocated sol (state_statics s)
    ∧ ∃ (β : location -> location -> Prop) (m : nat) (t : state)
        (vs' : list value) (hp' : heap) (rd' : read_record),
        run m (State e' [] [] empty_static_env empty_heap no_reads) = Step t
        ∧ step t = Exit k vs' hp' rd'
        ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
        ∧ locations_one_to_one β
        ∧ statics_rebuilt_heap G sol β (state_statics s) (state_statics t).

(* The same, with the static sets of closures kept whole. *)
Definition rebuild_exit_relation_whole_heap (G : graph) (sol : solution)
    (e e' : expr) : Prop :=
  ∀ (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    statics_allocated sol (state_statics s)
    ∧ ∃ (β : location -> location -> Prop) (m : nat) (t : state)
        (vs' : list value) (hp' : heap) (rd' : read_record),
        run m (State e' [] [] empty_static_env empty_heap no_reads) = Step t
        ∧ step t = Exit k vs' hp' rd'
        ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
        ∧ locations_one_to_one β
        ∧ statics_rebuilt_heap G sol β (state_statics s) (state_statics t)
        ∧ static_sets_whole sol (state_statics s) (state_statics t).

(* FlambdaReaperTypesSpec.exported_types_sound_with, given
   [rebuild_exit_relation_heap]. *)
Definition exported_types_sound_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    rebuild_exit_relation_heap (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e) ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition exported_types_sound_heap_statement : Prop :=
  exported_types_sound_heap_with all_fixes.

Definition exported_types_sound_heap_landed_statement : Prop :=
  exported_types_sound_heap_with landed_fixes.

(* The same for the fragment [loose], given
   [rebuild_exit_relation_whole_heap]. *)
Definition exported_unaliased_types_sound_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    least_solution is_local (traverse_unit flag u e) T ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    rebuild_exit_relation_whole_heap
      (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e) ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition exported_unaliased_types_sound_heap_statement : Prop :=
  exported_unaliased_types_sound_heap_with all_fixes.

Definition exported_unaliased_types_sound_heap_landed_statement : Prop :=
  exported_unaliased_types_sound_heap_with landed_fixes.

(* ======================================================================== *)
(* The exit relation and the rebuild                                        *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.rebuild_exit_relation_holds_statement, of the
   units of [in_heap_fragment]. *)
Definition rebuild_exit_relation_heap_holds_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    exits_of_values e ->
    rebuild_exit_relation_heap (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition rebuild_exit_relation_whole_heap_holds_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    exits_of_values e ->
    rebuild_exit_relation_whole_heap
      (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* With local fields, the tables passing the check of the fix. *)
Definition rebuild_exit_relation_heap_holds_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    exits_of_values e ->
    rebuild_exit_relation_heap (graph_of_facts (traverse_unit flag u e))
      (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

Definition rebuild_exit_relation_whole_heap_holds_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    variables_bound_once e = true ->
    exits_of_values e ->
    rebuild_exit_relation_whole_heap
      (graph_of_facts (traverse_unit flag u e)) (solution_of_tables T) e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* ======================================================================== *)
(* With no premise on the runs                                              *)
(* ======================================================================== *)

(* FlambdaReaperTypesSpec.exported_types_sound_end_to_end_with, of the
   units of [in_heap_fragment]: the rebuilt unit exits through the same
   continuation, the exits observed equal up to a correspondence of
   their locations, and the type of each symbol bound, rewritten, holds
   of the symbol there. *)
Definition exported_types_sound_end_to_end_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    exits_of_values e ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition exported_types_sound_end_to_end_heap_statement : Prop :=
  exported_types_sound_end_to_end_heap_with all_fixes.

Definition exported_types_sound_end_to_end_heap_landed_statement : Prop :=
  exported_types_sound_end_to_end_heap_with landed_fixes.

(* The fragment [loose]. *)
Definition unaliased_types_end_to_end_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    exits_of_values e ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_heap_statement : Prop :=
  unaliased_types_end_to_end_heap_with all_fixes.

Definition unaliased_types_end_to_end_heap_landed_statement : Prop :=
  unaliased_types_end_to_end_heap_with landed_fixes.

(* With local fields, for the least solution of the rules with the
   fix. *)
Definition exported_types_sound_end_to_end_heap_local_with
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    exits_of_values e ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition exported_types_sound_end_to_end_heap_local_statement : Prop :=
  exported_types_sound_end_to_end_heap_local_with all_fixes.

Definition exported_types_sound_end_to_end_heap_local_landed_statement : Prop :=
  exported_types_sound_end_to_end_heap_local_with landed_fixes.

Definition unaliased_types_end_to_end_heap_local_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    exits_of_values e ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_heap_local_statement : Prop :=
  unaliased_types_end_to_end_heap_local_with all_fixes.

Definition unaliased_types_end_to_end_heap_local_landed_statement : Prop :=
  unaliased_types_end_to_end_heap_local_with landed_fixes.

(* ======================================================================== *)
(* With the check of the term                                               *)
(* ======================================================================== *)

(* The statements above with FlambdaReaperTypesTermSpec.unit_of_values in
   place of [exits_of_values]. *)
Definition types_end_to_end_term_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition types_end_to_end_term_heap_statement : Prop :=
  types_end_to_end_term_heap_with all_fixes.

Definition types_end_to_end_term_heap_landed_statement : Prop :=
  types_end_to_end_term_heap_with landed_fixes.

Definition unaliased_types_end_to_end_term_heap_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_term_heap_statement : Prop :=
  unaliased_types_end_to_end_term_heap_with all_fixes.

Definition unaliased_types_end_to_end_term_heap_landed_statement : Prop :=
  unaliased_types_end_to_end_term_heap_with landed_fixes.

Definition types_end_to_end_term_heap_local_with (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition types_end_to_end_term_heap_local_statement : Prop :=
  types_end_to_end_term_heap_local_with all_fixes.

Definition types_end_to_end_term_heap_local_landed_statement : Prop :=
  types_end_to_end_term_heap_local_with landed_fixes.

Definition unaliased_types_end_to_end_term_heap_local_with
    (fx : fixes) : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool) (sets : list set_of_closures)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (n : nat) (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record) (sym : symbol)
    (allocated : statically_allocated_value),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    FlambdaReaperRebuildSpec.any_kind_arguments e = true ->
    FlambdaReaperRebuildSpec.results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    FlambdaReaperRebuildSpec.exits_one_value u e = true ->
    least_solution_local is_local (traverse_unit flag u e) T ->
    variables_bound_once e = true ->
    unit_of_values e = true ->
    sets_of_graph (graph_of_facts (traverse_unit flag u e)) sets ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    step s = Exit k vs hp rd ->
    typing_env_satisfied_strictly (state_statics s) valuation tenv ->
    env_in_fragment_with true valuation tenv ->
    copies_valued tenv canon valuation valuation' ->
    lookup sym (static_values (state_statics s)) = Some allocated ->
    ∃ (m : nat) (t : state) (vs' : list value) (hp' : heap)
      (rd' : read_record),
      run m
        (State
           (rebuild_unit_from_tables (traverse_unit flag u e) T
              preserve_direct_calls e)
           [] [] empty_static_env empty_heap no_reads)
      = Step t
      ∧ step t = Exit k vs' hp' rd'
      ∧ FlambdaReaperRebuildSpec.exits_observed_equal vs hp vs' hp'
      ∧ type_satisfied (state_statics t) valuation' (Value_symbol sym)
          (rewrite_symbol_type fx
             (plain_context is_local (traverse_unit flag u e) T sets) tenv
             canon sym).

Definition unaliased_types_end_to_end_term_heap_local_statement : Prop :=
  unaliased_types_end_to_end_term_heap_local_with all_fixes.

Definition unaliased_types_end_to_end_term_heap_local_landed_statement : Prop :=
  unaliased_types_end_to_end_term_heap_local_with landed_fixes.
