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

(* The closure types of the sets whose representation changes, applied end
   to end to the runs of the simulation of the changes of representation
   ([rr], FlambdaReaperUnboxingRefinesClosuresSimProofs.v), on the fragment
   of FlambdaReaperUnboxingRefinesClosuresCallsSpec.v
   ([repr_calls_unboxing_nr]).

   This file is to be reviewed: the premises and the statements, against
   FlambdaReaperTypesSpec.v (the closure types: [changed_closure_rebuilt],
   [rewrite_changed_type_sound_slots_statement],
   [rewrite_stored_type_sound_slots_statement]),
   FlambdaReaperTypesTraversalUnaliasedSpec.v (their forms for the graph of
   the traversal, [changed_type_sound_full_statement] and
   [stored_type_sound_full_statement], which the proofs apply) and
   FlambdaReaperTypesCallsSpec.v (the end-to-end statements of the symbols,
   whose premises these take). The proofs are in
   FlambdaReaperTypesReprClosuresProofs.v, which needs no review.

   The runs. The run of the unit, that of the unit rebuilt with changes of
   calling convention, and that of the unit rebuilt with unboxing and
   changes of representation in the mode of the fix, as in
   FlambdaReaperTypesCallsSpec.v. The last two are related by [rr], whose
   relation of the closures of a set whose representation changes now
   carries the value slots of the closure of the third run exactly: the
   new slots of the representation, in their order, each once
   ([crel_exact]), as the rebuild makes them
   (FlambdaReaperUnboxing.changed_value_slots). With the value of each new
   slot at its path in the closure of the second run, which [crel] has,
   that is [changed_closure_rebuilt], but for the kind of what a new slot
   holds (below).

   The statements, each with [_landed] for the fixes to land
   (FlambdaReaperTypes.landed_fixes):
   - [repr_changed_types_end_to_end_statement]: at a state of the run of
     the unit that binds a variable [x] whose representation changes (a
     closure of a set that changes, or a name that such a closure
     dominates, as the [my_closure] of its code), the run of the unit
     rebuilt with changes of calling convention comes to a state that
     binds [x] to [v_mid], which [rebuilt_value] relates to the value of
     the first run, and the run of the unit rebuilt with unboxing and
     changes of representation to one that binds [x] to [v'] with
     [changed_closure_rebuilt r v_mid v'] for the representation [r] of
     [x]; and the closure type of the fragment [loose] that holds of the
     value of [x] in the first run, rewritten with the changes of
     representation of the unit under a metadata that holds at [x], holds
     of [v'] (FlambdaReaperTypesTraversalUnaliasedSpec
     .changed_type_sound_full_statement).
   - [repr_stored_types_end_to_end_statement]: the same for any variable
     [x] of the unit with a use, which changes, or which the rebuild does
     not unbox and which no allocation that changes may come to: the type
     of the fragment [loose] that holds of its value in the first run,
     rewritten so, holds of its value in the third run
     (FlambdaReaperTypesTraversalUnaliasedSpec
     .stored_type_sound_full_statement, the third run related to the first
     by [rebuilt_changed]).
   - [repr_types_example_statement]: the unit
     FlambdaReaperUnboxing.Examples.block_in_slot (the closure [x60] of
     the code 52 holds in its slot 80 the block [x10] of 7 and 8, whose
     field 0 the code reads; the set changes its representation: the
     slot 80 unboxed, its field 0 the new slot 81, the new function slot
     82) meets the premises of the statements at the state of its run
     that binds [x60]; the closure type of [x60]
     (FlambdaReaperTypesSpec.ChangedExample.closure_type), rewritten, is
     the closure at 82 of the code 52 with [= 7] at 81; the run of the
     unit rebuilt with unboxing and changes of representation binds [x60]
     to the closure at 82 holding 7 at 81, which meets
     [changed_closure_rebuilt] with the closure of the run rebuilt
     without, and of which the rewritten type holds.

   Of the premises: those of FlambdaReaperTypesCallsSpec
   .repr_calls_types_end_to_end_statement on the unit and the tables, for
   the least solution of the rules ([unit_premises], with no field local
   or no slot projected, as the statements of the closure types ask
   [least_solution]) and the flag [Auto] (the graph of the traversal of
   FlambdaReaperTypesTraversalUnaliasedSpec.v); those of the closure types
   on the tables and the graph: [check_dominated] (which the
   representations are made from), and that no closure whose
   representation changes is held in a field other than a function slot
   ([changed_unstored]: the simulation relates no closure that changes
   held in a field, a block or a value slot); and those of the types on
   the state, as in FlambdaReaperTypesCallsSpec.v. Two premises are added,
   and why:
   - The kinds of the new slots ([repr_kinds_premises]):
     [changed_closure_rebuilt] asks each new slot to hold a value of the
     kind of its leaf, which for a value slot not unboxed in turn is
     [Value] (FlambdaReaperUnboxing.field_kind). The model's value slots
     have no kind and may hold a naked number, which the slot of the
     compiler would hold at its kind (Value_slot.kind). The premise: the
     unit rebuilt with unboxing and changes of representation is
     well-kinded (FlambdaWellKindedSpec.well_kinded_unit) for a table of
     slot kinds that gives the kind [Value] to each new slot of a
     representation that is not unboxed in turn, so that its run holds in
     those slots values of kind [Value] (FlambdaWellKindedSpec
     .run_well_kinded_statement). For a slot unboxed in turn, [crel] gives
     the kinds.
   - The run of the unit does not go wrong, so that the run of its rebuild
     with changes of calling convention, which refines it, does not
     either, and [rr] relates the state it comes to with one of the third
     run (as for the third run of FlambdaReaperTypesCallsSpec.v).
   [static_sets_whole], which the closure types ask of the static sets of
   the second run, holds with nothing asked: [rr] binds static blocks
   only, so the second run binds no static set of closures, and the first
   no static set with a use. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperRebuild FlambdaReaperFlowSpec FlambdaReaperTypes
  FlambdaReaperTypesSpec FlambdaReaperTypesCallsSpec
  FlambdaReaperCallingConventions.
From OxCamlFlambda2 Require FlambdaReaperUnboxing FlambdaRenameApartSpec
  FlambdaWellKindedSpec.
From OxCamlFlambda2Types Require Import FlambdaTypeGrammar
  FlambdaTypeMeaningSpec FlambdaTypingEnv FlambdaTypingEnvSpec.
Import ListNotations.

Module U := FlambdaReaperUnboxing.

(* ======================================================================== *)
(* The premises                                                             *)
(* ======================================================================== *)

(* The changes of representation of the unit, for the flag [Auto]. *)
Definition unit_reprs (is_local : field -> bool) (u : unit_interface)
    (e : expr) (T : tables) (dominated : list (node * node))
    (to_unbox tcr : list node) : list (node * U.closure_repr) :=
  U.unit_changed_representations is_local (traverse_unit Auto u e) T
    dominated to_unbox tcr e.

(* No allocation with a source that may come to [n] changes its
   representation. *)
Definition no_changed_allocation (sol : solution)
    (reprs : list (node * U.closure_repr)) (n : node) : Prop :=
  ∀ a : node,
    may_come_from sol a n -> s_has_source sol a = true ->
    Keyed.lookup node_eqb a reprs = None.

(* The table [slots] gives the kind [Value] to each new slot of a
   representation that is not unboxed in turn. *)
Definition new_slots_of_values (slots : FlambdaWellKindedSpec.slot_kinds)
    (reprs : list (node * U.closure_repr)) : Prop :=
  ∀ (a : node) (r : U.closure_repr) (w s : value_slot),
    Keyed.lookup node_eqb a reprs = Some r ->
    (Field_value_slot w, U.Not_unboxed s) ∈ U.cr_fields r ->
    lookup s slots = Some Value.

(* The kinds of the new slots: the unit rebuilt with unboxing and changes
   of representation is well-kinded for [slots], which gives the new slots
   not unboxed in turn the kind [Value]. *)
Definition repr_kinds_premises (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox tcr : list node)
    (preserve_direct_calls : bool) (slots : FlambdaWellKindedSpec.slot_kinds)
    : Prop :=
  FlambdaWellKindedSpec.well_kinded_unit slots (unit_return u) (unit_exn u)
    (repr_rebuilt Auto is_local u e T cannot_change dominated to_unbox tcr
       preserve_direct_calls)
  = true
  ∧ new_slots_of_values slots
      (unit_reprs is_local u e T dominated to_unbox tcr).

(* What the statements of the closure types ask of the tables and the
   graph: the table of [dominated_by_allocation_point] checked, and no
   closure that changes held in a field other than a function slot. *)
Definition repr_tables_premises (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (dominated : list (node * node)) (to_unbox tcr : list node) : Prop :=
  U.check_dominated T dominated = true
  ∧ changed_unstored (graph_of_facts (traverse_unit Auto u e))
      (solution_of_tables T)
      (unit_reprs is_local u e T dominated to_unbox tcr).

(* The state [s] of the run of the unit, which does not go wrong, binds
   the variable [x] of the unit, which has a use, to [v], of which the
   type [t] of the fragment [loose] holds under a valuation of the names
   of [tenv] that satisfies it; the metadata [m] holds at [x]. *)
Definition repr_state_premises (T : tables) (e : expr) (n : nat) (s : state)
    (tenv : Typing_env.t) (canon : canon_fn) (valuation valuation' : env)
    (x : variable) (v : value) (t : flambda_type) (m : rewrite_metadata)
    : Prop :=
  run n (initial_state e) = Step s
  ∧ ~ behaves (initial_state e) Goes_wrong
  ∧ lookup x (state_vars s) = Some v
  ∧ s_has_usage (solution_of_tables T) (Node_variable x) = true
  ∧ x ∈ FlambdaRenameApartSpec.variables_of e
  ∧ typing_env_satisfied_strictly (state_statics s) valuation tenv
  ∧ env_in_fragment_with true valuation tenv
  ∧ copies_valued tenv canon valuation valuation'
  ∧ type_in_fragment_with true valuation t
  ∧ metadata_at (solution_of_tables T) m (Node_variable x)
  ∧ type_satisfied (state_statics s) valuation v t.

(* The type [t] rewritten with the changes of representation of the unit
   ([rewrite_arbitrary_u]). *)
Definition repr_rewritten (fx : fixes) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (sets : list set_of_closures) (dominated : list (node * node))
    (to_unbox tcr : list node) (tenv : Typing_env.t) (canon : canon_fn)
    (fuel : nat) (m : rewrite_metadata) (t : flambda_type) : flambda_type :=
  rewrite_arbitrary_u fx
    (Rewrite_context (traverse_unit Auto u e) T is_local sets
       (map fst (unit_reprs is_local u e T dominated to_unbox tcr)))
    tenv canon (U.bound_code_my_closures e)
    (unit_reprs is_local u e T dominated to_unbox tcr) fuel m t.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The closure types of a variable whose representation changes, of the
   three runs: the second binds it to [v_mid], related to its value in the
   first by [rebuilt_value]; the third to [v'], the closure of [v_mid] in
   the representation [r] of the variable ([changed_closure_rebuilt]), of
   which the type rewritten holds. *)
Definition repr_changed_types_end_to_end_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool) (sets : list set_of_closures)
    (slots : FlambdaWellKindedSpec.slot_kinds) (tenv : Typing_env.t)
    (canon : canon_fn) (valuation valuation' : env) (fuel : nat)
    (m : rewrite_metadata) (t : flambda_type) (n : nat) (s : state)
    (x : variable) (v : value) (r : U.closure_repr),
    unit_premises false Auto is_local u e T sets ->
    repr_premises Auto is_local u e T cannot_change dominated to_unbox tcr nr ->
    repr_tables_premises is_local u e T dominated to_unbox tcr ->
    repr_kinds_premises is_local u e T cannot_change dominated to_unbox tcr
      preserve_direct_calls slots ->
    repr_state_premises T e n s tenv canon valuation valuation' x v t m ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_reprs is_local u e T dominated to_unbox tcr)
    = Some r ->
    ∃ (m2 : nat) (t2 : state) (v_mid : value) (m3 : nat) (t3 : state)
      (v' : value),
      run m2
        (initial_state
           (rebuild_unit_cc_from_tables (traverse_unit Auto u e) T
              cannot_change (unit_code_deps Auto u e) preserve_direct_calls
              e))
      = Step t2
      ∧ lookup x (state_vars t2) = Some v_mid
      ∧ rebuilt_value (graph_of_facts (traverse_unit Auto u e))
          (solution_of_tables T) v (Node_variable x) v_mid
      ∧ run m3
          (initial_state
             (repr_rebuilt Auto is_local u e T cannot_change dominated
                to_unbox tcr preserve_direct_calls))
        = Step t3
      ∧ lookup x (state_vars t3) = Some v'
      ∧ changed_closure_rebuilt r v_mid v'
      ∧ type_satisfied (state_statics t3) valuation' v'
          (repr_rewritten fx is_local u e T sets dominated to_unbox tcr tenv
             canon fuel m t).

Definition repr_changed_types_end_to_end_statement : Prop :=
  repr_changed_types_end_to_end_with all_fixes.

Definition repr_changed_types_end_to_end_landed_statement : Prop :=
  repr_changed_types_end_to_end_with landed_fixes.

(* The types of any variable of the unit with a use, which changes, or
   which the rebuild does not unbox and to which no allocation that
   changes may come: the type rewritten holds of its value in the third
   run. *)
Definition repr_stored_types_end_to_end_with (fx : fixes) : Prop :=
  ∀ (is_local : field -> bool) (u : unit_interface) (e : expr) (T : tables)
    (cannot_change : list node) (dominated : list (node * node))
    (to_unbox tcr : list node) (preserve_direct_calls : bool)
    (nr : code_id -> bool) (sets : list set_of_closures)
    (slots : FlambdaWellKindedSpec.slot_kinds) (tenv : Typing_env.t)
    (canon : canon_fn) (valuation valuation' : env) (fuel : nat)
    (m : rewrite_metadata) (t : flambda_type) (n : nat) (s : state)
    (x : variable) (v : value),
    unit_premises false Auto is_local u e T sets ->
    repr_premises Auto is_local u e T cannot_change dominated to_unbox tcr nr ->
    repr_tables_premises is_local u e T dominated to_unbox tcr ->
    repr_kinds_premises is_local u e T cannot_change dominated to_unbox tcr
      preserve_direct_calls slots ->
    repr_state_premises T e n s tenv canon valuation valuation' x v t m ->
    Keyed.lookup node_eqb (Node_variable x)
      (unit_reprs is_local u e T dominated to_unbox tcr)
    <> None
    ∨ Keyed.lookup node_eqb (Node_variable x)
        (fst (U.unit_unboxed_fields (traverse_unit Auto u e) T dominated
                to_unbox e))
      = None
      ∧ no_changed_allocation (solution_of_tables T)
          (unit_reprs is_local u e T dominated to_unbox tcr)
          (Node_variable x) ->
    ∃ (m3 : nat) (t3 : state) (v' : value),
      run m3
        (initial_state
           (repr_rebuilt Auto is_local u e T cannot_change dominated to_unbox
              tcr preserve_direct_calls))
      = Step t3
      ∧ lookup x (state_vars t3) = Some v'
      ∧ type_satisfied (state_statics t3) valuation' v'
          (repr_rewritten fx is_local u e T sets dominated to_unbox tcr tenv
             canon fuel m t).

Definition repr_stored_types_end_to_end_statement : Prop :=
  repr_stored_types_end_to_end_with all_fixes.

Definition repr_stored_types_end_to_end_landed_statement : Prop :=
  repr_stored_types_end_to_end_with landed_fixes.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit FlambdaReaperUnboxing.Examples.block_in_slot:

     code 52 (x40) =                  (my_closure 41, return 3)
       let x45 = x41.(80) in
       let x46 = x45.(0) in
       return x46
     let x10 = [7; 8] in
     let x60 = closure 70 of the code 52 with x10 at 80 in
     apply x60 9 -> k1

   with the tables of the solver; the set of [x60] changes its
   representation ([tcr]), the block [x10] in its slot unboxed
   ([to_unbox]). The state [state_at] is that of the run of the unit at the
   call, which binds [x60] to the closure at 70 holding [7; 8] at 80. *)
Module ReprTypesExample.

Local Open Scope positive_scope.

Definition unit_expr : expr := U.Examples.block_in_slot.

Definition tables : tables := U.Examples.block_in_slot_tables.

Definition dominated : list (node * node) := U.Examples.block_in_slot_dominated.

Definition to_unbox : list node := [Node_variable 10].

Definition tcr : list node := [Node_variable 60].

(* The kinds of the value slots of the rebuilt unit. *)
Definition slots : FlambdaWellKindedSpec.slot_kinds :=
  [(80, Value); (81, Value)].

(* The unit rebuilt with unboxing and changes of representation. *)
Definition rebuilt : expr :=
  repr_rebuilt Auto U.Examples.no_local U.Examples.interface unit_expr tables
    [] dominated to_unbox tcr true.

Definition state_at : state :=
  match run 3 (initial_state unit_expr) with
  | Step s => s
  | _ => initial_state unit_expr
  end.

Definition canon : canon_fn := fun n _ _ => n.

End ReprTypesExample.

(* The unit meets the premises of the statements, with the set of closures
   of [x60] (FlambdaReaperTypesSpec.ChangedExample.sets), the code taken to
   never return by none, the table of slot kinds [slots], and, at
   [state_at], [x60] bound to FlambdaReaperTypesSpec.ChangedExample.closure,
   of the type ChangedExample.closure_type, under the metadata
   [Single_source x60], with no name in the typing environment. Its
   representation is FlambdaReaperUnboxing.Examples.block_in_slot_repr; the
   type rewritten is ChangedExample.rewritten (the closure at 82 of the
   code 52 with [= 7] at 81). The run of the rebuilt unit binds [x60] to
   ChangedExample.closure' (the closure at 82 holding 7 at 81), which meets
   [changed_closure_rebuilt] with the closure of the run of the unit
   rebuilt without unboxing (ChangedExample.closure_mid), and of which the
   type rewritten holds. *)
Definition repr_types_example_with (fx : fixes) : Prop :=
  unit_premises false Auto U.Examples.no_local U.Examples.interface
    ReprTypesExample.unit_expr ReprTypesExample.tables ChangedExample.sets
  ∧ repr_premises Auto U.Examples.no_local U.Examples.interface
      ReprTypesExample.unit_expr ReprTypesExample.tables []
      ReprTypesExample.dominated ReprTypesExample.to_unbox
      ReprTypesExample.tcr (fun _ => false)
  ∧ repr_tables_premises U.Examples.no_local U.Examples.interface
      ReprTypesExample.unit_expr ReprTypesExample.tables
      ReprTypesExample.dominated ReprTypesExample.to_unbox
      ReprTypesExample.tcr
  ∧ repr_kinds_premises U.Examples.no_local U.Examples.interface
      ReprTypesExample.unit_expr ReprTypesExample.tables []
      ReprTypesExample.dominated ReprTypesExample.to_unbox
      ReprTypesExample.tcr true ReprTypesExample.slots
  ∧ repr_state_premises ReprTypesExample.tables ReprTypesExample.unit_expr 3
      ReprTypesExample.state_at Typing_env.create ReprTypesExample.canon []
      [] 60%positive ChangedExample.closure ChangedExample.closure_type
      (Single_source ChangedExample.x60)
  ∧ Keyed.lookup node_eqb ChangedExample.x60
      (unit_reprs U.Examples.no_local U.Examples.interface
         ReprTypesExample.unit_expr ReprTypesExample.tables
         ReprTypesExample.dominated ReprTypesExample.to_unbox
         ReprTypesExample.tcr)
    = Some U.Examples.block_in_slot_repr
  ∧ repr_rewritten fx U.Examples.no_local U.Examples.interface
      ReprTypesExample.unit_expr ReprTypesExample.tables ChangedExample.sets
      ReprTypesExample.dominated ReprTypesExample.to_unbox
      ReprTypesExample.tcr Typing_env.create ReprTypesExample.canon 6
      (Single_source ChangedExample.x60) ChangedExample.closure_type
    = ChangedExample.rewritten
  ∧ ∃ t3 : state,
      run 3 (initial_state ReprTypesExample.rebuilt) = Step t3
      ∧ lookup 60%positive (state_vars t3) = Some ChangedExample.closure'
      ∧ changed_closure_rebuilt U.Examples.block_in_slot_repr
          ChangedExample.closure_mid ChangedExample.closure'
      ∧ type_satisfied (state_statics t3) [] ChangedExample.closure'
          ChangedExample.rewritten.

Definition repr_types_example_statement : Prop :=
  repr_types_example_with all_fixes.

Definition repr_types_example_landed_statement : Prop :=
  repr_types_example_with landed_fixes.
