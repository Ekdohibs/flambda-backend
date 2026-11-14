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

(* Erasure: the run of a term on the machine of regions
   (FlambdaRegionsSpec.v), where the machine does not stop, is the run
   of the term with its regions dropped on the machine of
   FlambdaSemanticsSpec.v.

   This file is to be reviewed: the relation [value_erased] and the two
   statements. The term with its regions dropped is
   FlambdaRegionsSpec.erase, which is reviewed there. This file adds
   nothing to either machine. FlambdaRegionsErasureProofs.v proves
   the statements.

   WHY

   The validation is given terms with their regions dropped (decision
   D37), and what it proves is about their runs on the machine of
   FlambdaSemanticsSpec.v. The check of regions runs the terms with
   their regions kept, on another machine. Nothing so far says that
   the two machines agree on what a term computes. The statements here
   say it: the machine of regions adds ways of having no step
   (decisions D45 to D47), names for local allocations, and the
   regions themselves, which [erase] drops; and nothing else.

   THE TERM WITH ITS REGIONS DROPPED

   FlambdaRegionsSpec.erase: the [Let] of a begin is dropped, that of
   an end binds the unit value, a [Local_allocation] becomes an alias
   or nothing, and a parameter of the kind [Region] is dropped with
   the argument at its place in every [Apply_cont] of its
   continuation. The test tool compares, for each unit, [erase] of the
   term that it runs on the machine of regions with the term that it
   validates; nothing here states that they are the same.

   [erase] cannot follow a region that code gives to its return
   continuation, or to its exception continuation: those are not bound
   by a [Let_cont] of the body. And a poison constant of the kind
   [Region] is a value that a region parameter can be given and that
   its handler may go on to evaluate, where the erased handler has no
   parameter for it. So both statements are of a term, and of a
   static environment, in which no call returns or raises to a
   handler that has a region parameter and there is no such constant
   (FlambdaRegionsSpec.erasable), which the tool tests too.

   THE VALUES ([value_erased])

   The machine of regions has a symbol where the other machine has a
   block, a boxed number, an array or a closure: the symbol that the
   local allocation was bound to. [value_erased rs v v'] says that
   [v'] is [v] with each such symbol replaced by what it is bound to
   in the state [rs], in a frame or in the static environment, and so
   on inside. It is FlambdaLiftSpec.value_lifted read the other way,
   and of the symbols that the machine drew only: a symbol of the term
   or of the imports is itself.

   Of a symbol that the machine drew and that is bound to nothing, a
   local allocation that is gone or a mark, nothing is related to it.
   That is right for a value that may be evaluated, by the invariant
   of FlambdaRegionsInvariantSpec.v: it holds no such symbol, except
   in a partial application. So the relation is asked of a value only
   where it may be evaluated ([value_erased_if_usable]) in the two
   places where a value that is gone can be: the callee and the
   arguments of a partial application, and the values at an exit, as
   an exit may be given extra arguments that a continuation has held
   since a call (the header of FlambdaRegionsInvariantSpec.v, "what
   follows"). Of an argument that a partial application keeps, and of
   a value at an exit, the kind is asked whether or not the value may
   be evaluated: a call tests the kinds of the arguments kept with
   those of the new ones, on either machine, and a symbol that is gone
   is of the kind [Value].

   So of a value at an exit that may not be evaluated, a symbol whose
   allocation is gone, the statements say its kind and nothing else:
   the run of the erased term may have any value of the kind [Value]
   at its place. They are not the agreement of the values at the exit
   for every run that the check passes. For a run all of whose values
   at the exit may be evaluated they give [value_erased] of each,
   with what that relation leaves out of a partial application
   unchanged. That every run from the state that the tool starts in is
   such a run is not stated here: it is
   FlambdaRegionsCallsSpec.exit_values_usable_statement, from the fact
   that code cannot end a region of its caller, and the two statements
   below are said again there with [value_erased] of every value at
   the exit.

   THE STATEMENTS

   Both are of a run from the state that the test tool starts in, with
   the symbols that the machine draws apart from those of the term,
   and of a term that [erase] is meant for
   (FlambdaRegionsSpec.erasable).

   - [erasure_exits_statement]. If the run on the machine of regions
     exits, the run of the erased term exits through the same
     continuation, with as many values; each value of the first is of
     the kind of the value of the second at its place, and each that
     may be evaluated is erased to it.
   - [erasure_adds_no_exit_statement]. If the run of the erased term
     exits, the run on the machine of regions exits through the same
     continuation with values related in the same way, or stops
     somewhere with no step: for one of the rules of regions, or
     because the machine of FlambdaSemanticsSpec.v has none there on
     the symbols of local allocations.

   Together: a run on the machine of regions that does not stop
   without a step exits if and only if the run of the erased term
   does, and alike.

   WHAT IS NOT STATED

   - Anything of a run that goes on for ever.
   - That where the machine of regions stops for another reason than
     its rules, the run of the erased term goes wrong. It is false: a
     partial application whose callee is a local closure that is gone
     is called on one machine and has no step on the other
     (FlambdaRegionsSpec.v, "what this stage leaves out").
   - Anything of the counts, or of the frames that are left at the
     exit.
   - Anything of the heap at the exit. Neither machine makes a cell
     in a run of a term that the conversion makes ([step] does not run
     the primitives of the heap), so the two heaps are those that the
     runs began with; a value at the exit that is a location is related
     to itself, and nothing is said of what the heaps hold at it.
   - That [erase] of what the test tool converts with regions kept is
     what it converts with regions dropped: the tool tests it of each
     unit.
   - That a run from the state that the tool starts in exits with
     values that may all be evaluated: that is in
     FlambdaRegionsCallsSpec.v (see "the values", above). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaClosuresSpec
  FlambdaRegionsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The values                                                               *)
(* ======================================================================== *)

Definition slot_erased (R : value -> value -> Prop)
    (slot slot' : value_slot * value) : Prop :=
  fst slot = fst slot' ∧ R (snd slot) (snd slot').

(* [value_erased rs v v']: [v'] is the value [v] of the state [rs] of
   the machine of regions, with what each local allocation is bound to
   in place of its symbol.

   The first nine rules are of values that have the same head: a
   constant, a boxed number, a location, a symbol that the machine did
   not draw, and a block, an array, a closure, a closure of a set or a
   partial application with parts erased. In a partial application a
   part is asked to be erased only if it may be evaluated, and an
   argument is asked to be of the same kind in any case.

   The last five are of a symbol that the machine drew and that
   [live_statics] binds: it is erased to a block, a boxed number, an
   array or a closure whose parts are those of what it is bound to,
   erased. The symbol of a set of closures is the one that the set
   gives the slot, and the set is bound whole
   (FlambdaClosuresSpec.static_set_bound). *)
#[local] Set Warnings "-register-all".
Inductive value_erased (rs : region_state) : value -> value -> Prop :=
  | Erased_const (c : const) :
    value_erased rs (Value_const c) (Value_const c)
  | Erased_boxed_number (c : const) :
    value_erased rs (Value_boxed_number c) (Value_boxed_number c)
  | Erased_location (l : location) :
    value_erased rs (Value_location l) (Value_location l)
  | Erased_symbol (s : symbol) :
    drawn rs s = false ->
    value_erased rs (Value_symbol s) (Value_symbol s)
  | Erased_block (tag : Z) (fields fields' : list value) :
    Forall2 (value_erased rs) fields fields' ->
    value_erased rs (Value_block tag fields) (Value_block tag fields')
  | Erased_array (elements elements' : list value) :
    Forall2 (value_erased rs) elements elements' ->
    value_erased rs (Value_immutable_array elements)
      (Value_immutable_array elements')
  | Erased_closure (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_erased (value_erased rs)) value_slots value_slots' ->
    value_erased rs (Value_closure code value_slots)
      (Value_closure code value_slots')
  | Erased_closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_erased (value_erased rs)) value_slots value_slots' ->
    value_erased rs (Value_closure_of_set slot function_decls value_slots)
      (Value_closure_of_set slot function_decls value_slots')
  | Erased_partial_application (callee callee' : value)
      (args args' : list value) (args_arity : arity) :
    (value_usable rs callee = true -> value_erased rs callee callee') ->
    Forall2
      (fun arg arg' =>
         kind_of_value arg' = kind_of_value arg
         ∧ (value_usable rs arg = true -> value_erased rs arg arg'))
      args args' ->
    value_erased rs (Value_partial_application callee args args_arity)
      (Value_partial_application callee' args' args_arity)
  | Erased_local_block (s : symbol) (tag : Z) (fields fields' : list value) :
    drawn rs s = true ->
    lookup s (static_values (live_statics rs))
    = Some (Static_block tag fields) ->
    Forall2 (value_erased rs) fields fields' ->
    value_erased rs (Value_symbol s) (Value_block tag fields')
  | Erased_local_boxed_number (s : symbol) (c : const) :
    drawn rs s = true ->
    lookup s (static_values (live_statics rs))
    = Some (Static_boxed_number (Value_const c)) ->
    value_erased rs (Value_symbol s) (Value_boxed_number c)
  | Erased_local_array (s : symbol) (elements elements' : list value) :
    drawn rs s = true ->
    lookup s (static_values (live_statics rs))
    = Some (Static_immutable_array elements) ->
    Forall2 (value_erased rs) elements elements' ->
    value_erased rs (Value_symbol s) (Value_immutable_array elements')
  | Erased_local_closure (s : symbol) (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) :
    drawn rs s = true ->
    lookup s (static_values (live_statics rs))
    = Some (Static_closure code value_slots) ->
    Forall2 (slot_erased (value_erased rs)) value_slots value_slots' ->
    value_erased rs (Value_symbol s) (Value_closure code value_slots')
  | Erased_local_closure_of_set (s : symbol) (slot : function_slot)
      (closure_symbols : list (function_slot * symbol))
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) :
    drawn rs s = true ->
    lookup slot closure_symbols = Some s ->
    map fst closure_symbols = map fst function_decls ->
    static_set_bound (live_statics rs) closure_symbols function_decls
      value_slots ->
    Forall2 (slot_erased (value_erased rs)) value_slots value_slots' ->
    value_erased rs (Value_symbol s)
      (Value_closure_of_set slot function_decls value_slots').

(* The two values are of one kind, and the relation holds if the first
   may be evaluated. *)
Definition value_erased_if_usable (rs : region_state) (v v' : value)
    : Prop :=
  kind_of_value v' = kind_of_value v
  ∧ (value_usable rs v = true -> value_erased rs v v').

(* ======================================================================== *)
(* Statements                                                               *)
(* ======================================================================== *)

(* The state in which the erased term is run: that of
   FlambdaRegionsSpec.initial_region_state, with the term and the code
   of the static environment erased. *)
Definition erased_initial_state (e : expr) (statics : static_env) : state :=
  State (erase [] e) [] [] (erase_statics statics) empty_heap.

(* Where the run on the machine of regions exits, the run of the erased
   term exits through the same continuation, with as many values; each
   value of the first run is of the kind of the value of the second at
   its place, and each that may be evaluated is erased to it, under the
   state that the first run left by. *)
Definition erasure_exits_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (fuel : nat)
    (k : continuation) (vs : list value) (rs : region_state),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    region_run fuel (initial_region_state e statics base)
    = Region_exit k vs rs ->
    ∃ (vs' : list value) (hp : heap),
      behaves (erased_initial_state e statics) (Exits k vs' hp)
      ∧ Forall2 (value_erased_if_usable rs) vs vs'.

(* Where the run of the erased term exits, the run on the machine of
   regions exits alike, or stops with no step. *)
Definition erasure_adds_no_exit_statement : Prop :=
  ∀ (e : expr) (statics : static_env) (base : symbol) (k : continuation)
    (vs' : list value) (hp : heap),
    region_names_fresh base e statics = true ->
    erasable e statics = true ->
    behaves (erased_initial_state e statics) (Exits k vs' hp) ->
    ∃ fuel : nat,
      (∃ fault : region_fault,
        region_run fuel (initial_region_state e statics base)
        = Region_undefined fault)
      ∨ (∃ (vs : list value) (rs : region_state),
          region_run fuel (initial_region_state e statics base)
          = Region_exit k vs rs
          ∧ Forall2 (value_erased_if_usable rs) vs vs').
