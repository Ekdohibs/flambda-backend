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

(* What the rebuild of the reaper is worth: the unit rebuilt from a
   solution that passes the checks refines the unit.

   This file is to be reviewed: the checks [closure_free],
   [no_code_no_slots], [no_code], [no_call], [exits_one_value] and
   [results_kinded], the statements, and the examples, against
   FlambdaReaperRebuild.v (the rebuild, which needs no review) and
   FlambdaReaperFlowSpec.v (the meaning of a solution). The proofs are
   in FlambdaReaperRebuildProofs.v, which needs no review.

   The statement ([rebuild_refines_statement]) is for a unit [e] of the
   fragment of FlambdaReaperFlowSpec.v ([in_flow_fragment]) that has no
   closure, no projection, no call and no code ([closure_free]: [Let]s
   of simples and of primitives, [Let_cont]s, [Apply_cont]s,
   [Switch]es, [Invalid], and [Let_static]s of blocks), with the
   interface [u]; any mode of the traversal ([flag]) and any of the
   rebuild ([preserve_direct_calls], which no term without calls
   reads); and tables [T] that pass the two checks of
   FlambdaReaperSolution.v on the graph that the traversal makes of the
   unit. The unit rebuilt from them ([rebuild_unit_from_tables])
   refines the unit run on its own ([program_refines]): an exit of the
   unit is one of the rebuilt unit, with the same values, and the
   rebuilt unit goes wrong only where the unit does. The premises, and
   why each is there:
   - [is_local_slots_only]: what the compiler guarantees of the local
     fields. The fields of a block and of a boxed number are never
     local, so the rules carry what they hold.
   - [(∀ f, is_local f = false) ∨ projection_free e = true]: the
     premise of FlambdaReaperFlowSpec.flow_sound_statement. With local
     value slots and projections the rebuild goes wrong
     (FlambdaReaperFlowSpec.local_slot_counterexample_statement). A
     unit of the fragment here has no projection and meets it; it is
     stated for the fragment to grow.
   - [in_flow_fragment]: the fragment in which FlambdaReaperFlowSpec.v
     says what a solution means.
   - [closure_free]: the part of that fragment for which the statement
     is proved. The rebuild of a set of closures (value slots removed,
     [Invalid] where a closure has no source) and of a call (masks of
     the arguments, wrappers, the callee kept) is not covered yet.
   - [well_formed_unit]: of it only [names_in_scope] is used: every
     continuation that the unit applies is bound in it or is one of the
     two of the unit, through which the run exits. The traversal gives
     the parameters of these two nodes with any usage, at which
     FlambdaReaperFlowSpec.flow_sound_statement places the values of an
     exit; it says nothing of an exit through another continuation,
     which the machine allows.
   - [exits_one_value]: every [Apply_cont] of a continuation of the unit
     gives it one value. The traversal gives each of the two
     continuations of the unit one parameter, as traverse.ml does
     ([dummy_toplevel_return]: the module block; the exception), and
     aliases the arguments of an [Apply_cont] to the parameters,
     cutting to the shorter list where the [List.iter2] of
     [apply_cont_deps] in traverse.ml raises. A second value given to
     the return continuation of the unit then has no use, and the
     rebuild makes it poison: the rebuilt unit exits with other values
     ([exits_one_value_needed_statement]). It asks the same of a call
     that returns to a continuation of the unit: one result, and no
     extra argument with an exception that goes to one; otherwise the
     rebuild makes the second result poison
     ([call_exits_one_value_needed_statement]). [well_formed_unit] does
     not ask it; it is a condition of well-formedness that the
     compiler's units meet (the return arity of a call is that of its
     continuation, and the continuation of the unit takes one value), not
     a fault of the rebuild, and a row for requests/C0-well-formed.md.
   - [check_closed]: the solution is closed under the rules
     (FlambdaReaperSolutionSpec.check_closed_sound_statement).
   - [check_field_used_exact]: a field said to be used is one that a
     rule of the post-processing gives, so what the field holds has a
     use and the rebuild keeps it. Without it a solution may say a
     field used that nothing reads, and the rebuild then boxes poison,
     which [Box_number] refuses: a unit that boxes a number that it
     only ever passes to a continuation which reads its tag, in a
     branch that does not run, exits, and its rebuild goes wrong at the
     boxing ([field_used_exact_needed_statement]). For the fields of a
     block it would not matter: a block of poison is a block.

   Exits compare values exactly. What reaches a continuation of the unit
   is at a node with any usage; a block there has all its fields used
   (the fields of a block are not local) and what they hold has any
   usage, down to the constants, so the rebuild keeps it whole; a boxed
   number there keeps its contents; and a symbol is compared as a
   symbol, whatever its static block holds after the rebuild.

   The second slice of FlambdaReaperFlowSpec.v (closures, projections,
   static sets, code, direct calls) is to be covered by another
   statement, [rebuild_refines_observably_statement], whose premises
   are those of [rebuild_refines_statement] and [results_kinded], and
   which compares exits up to what the world outside the unit observes
   ([program_refines_observably], [observed_equal]): equal values,
   except that closures are compared without their value slots, inside
   blocks, arrays, boxed numbers and partial applications too; symbols
   are compared by name. Both choices answer a counterexample:
   - A closure that reaches an exit loses the value slots that no
     projection of the unit names ([closure_exit_counterexample_statement]:
     exits are not equal). Nothing outside the unit can read them, and
     the exits are equal up to them
     ([closure_exit_observed_statement]). A closure may reach an exit
     inside an exception raised by code, so no check of the
     [Apply_cont]s of the unit could keep closures from exits.
   - The wrapper that the rebuild gives a call ([make_apply_wrapper])
     tests the kinds of the return arity of the call, which the machine
     does not test of what code returns
     ([wrapper_kinds_counterexample_statement]). [results_kinded] asks
     that each code apply its return continuation to simples of the
     kinds of its result arity, and that the return continuation of
     each call, where the term binds it, take the kinds of the return
     arity of the call; the counterexample fails it, and is not refined
     even observably ([results_kinded_needed_statement]). The
     compiler's units, which are well-kinded, meet it; the model asks
     nothing of kinds otherwise (FlambdaWellKindedSpec.v).
   [rebuild_refines_observably_statement] is proved for the units that
   meet [no_call] in place of [closure_free]: closures and static sets of
   closures with value slots, projections of function slots and of
   value slots, code (rebuilt, never called) and deleted code, but no
   call. A unit without code meets it ([no_code_no_call_statement]), a
   unit without value slots meets [no_code]
   ([no_code_no_slots_no_code_statement]), and so does a unit without
   closures ([closure_free_no_code_no_slots_statement]). The rebuild keeps
   a value slot of a set where one of the closures of the set has it
   used, with the value it holds, which then has a use; it keeps a set
   with the projections that follow it ([projections_of_set]) if one of
   them has a use, and a static set if one of its symbols has one; it
   keeps code whose code id has a use, rebuilt, and deletes the rest,
   which no run reads as no code is called. [no_code] asks
   that a projection of a value slot be into a variable of a kind other
   than [Region] and [Rec_info]: the rebuild keeps such a variable whether
   or not it has a use ([is_var_used]), and the proof needs the use to
   know that the value slot read is kept. The premise [results_kinded] is
   not used yet. What remains is to drop [no_call]: calls (the masks of
   the arguments, the wrappers, the callee, the rebuilt bodies of code
   run).
   The statement holds of more than nothing ([example_statement]): a
   unit with a static block, a block one field of which no load reads, a
   [Let] with no use, and a continuation one parameter of which has no
   use. Its least solution passes the checks; its rebuild makes the
   field poison, removes the [Let], the parameter and the argument
   given to it; and the unit exits with 7, as the rebuilt unit does.
   [closures_example_statement] does the same for a unit with closures
   and a static set of closures, without value slots or code; and
   [value_slots_example_statement], up to observation, for a unit whose
   closures have value slots that the rebuild removes, and which exits
   with a closure; [code_example_statement] for a unit with code. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaMapSpec FlambdaSemanticsSpec
  FlambdaUnusedStaticsSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperFlowSpec FlambdaReaperRebuild.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Definition named_closure_free (defining_expr : named) : bool :=
  match defining_expr with
  | Simple _ | Prim _ _ => true
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ => false
  end.

Definition binding_closure_free (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ _) => true
  | _ => false
  end.

(* No closure, no projection, no call and no code: [Let]s of simples and
   of primitives, [Let_static]s of blocks, [Let_cont]s, [Apply_cont]s,
   [Switch]es and [Invalid]. *)
Fixpoint closure_free (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_closure_free defining_expr && closure_free body
  | Let_static bindings body =>
    forallb binding_closure_free bindings && closure_free body
  | Let_cont_non_recursive _ h body =>
    handler_closure_free h && closure_free body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_closure_free h) handlers
    && closure_free body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_closure_free (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => closure_free handler
  end.

(* No call, no code, and closures without value slots: [Let]s of
   simples, of primitives, of closures of sets with no value slot and of
   projections of function slots, [Let_static]s of blocks, of deleted
   code and of closures of static sets with no value slot, each bound to
   the symbol that its set names for its slot, [Let_cont]s,
   [Apply_cont]s, [Switch]es and [Invalid]. *)
Definition named_no_code_no_slots (defining_expr : named) : bool :=
  match defining_expr with
  | Simple _ | Prim _ _ | Project_function_slot _ _ _ => true
  | Closure_of_set _ _ [] => true
  | Closure_of_set _ _ (_ :: _) | Set_of_closures _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _ => false
  end.

Definition binding_no_code_no_slots (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ _) | Bind_deleted_code _ => true
  | Bind_symbol s (Static_const_closure_of_set slot closure_symbols
                     function_decls []) =>
    is_bound slot function_decls
    && match lookup slot closure_symbols with
       | Some s' => Pos.eqb s' s
       | None => false
       end
  | _ => false
  end.

Fixpoint no_code_no_slots (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_no_code_no_slots defining_expr && no_code_no_slots body
  | Let_static bindings body =>
    forallb binding_no_code_no_slots bindings && no_code_no_slots body
  | Let_cont_non_recursive _ h body =>
    handler_no_code_no_slots h && no_code_no_slots body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_no_code_no_slots h) handlers
    && no_code_no_slots body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_no_code_no_slots (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => no_code_no_slots handler
  end.

(* A unit without closures meets it. *)
Definition closure_free_no_code_no_slots_statement : Prop :=
  ∀ e : expr, closure_free e = true -> no_code_no_slots e = true.

(* No call and no code: [Let]s of simples, of primitives, of closures of
   sets, of projections of function slots and, into a variable of a kind
   other than [Region] and [Rec_info], of value slots; [Let_static]s of
   blocks, of deleted code and of closures of static sets, each bound to
   the symbol that its set names for its slot; [Let_cont]s,
   [Apply_cont]s, [Switch]es and [Invalid]. *)
Definition named_no_code (kind : flambda_kind) (defining_expr : named)
    : bool :=
  match defining_expr with
  | Simple _ | Prim _ _ | Closure_of_set _ _ _ | Project_function_slot _ _ _ =>
    true
  | Project_value_slot_from _ _ _ =>
    match kind with
    | Region | Rec_info => false
    | Value | Naked_number _ => true
    end
  | Set_of_closures _ _ | Project_value_slot _ _ => false
  end.

Definition binding_no_code (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ _) | Bind_deleted_code _ => true
  | Bind_symbol s (Static_const_closure_of_set slot closure_symbols
                     function_decls _) =>
    is_bound slot function_decls
    && match lookup slot closure_symbols with
       | Some s' => Pos.eqb s' s
       | None => false
       end
  | _ => false
  end.

Fixpoint no_code (e : expr) : bool :=
  match e with
  | Let _ kind defining_expr body =>
    named_no_code kind defining_expr && no_code body
  | Let_static bindings body =>
    forallb binding_no_code bindings && no_code body
  | Let_cont_non_recursive _ h body => handler_no_code h && no_code body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_no_code h) handlers && no_code body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_no_code (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => no_code handler
  end.

(* A unit without value slots meets it. *)
Definition no_code_no_slots_no_code_statement : Prop :=
  ∀ e : expr, no_code_no_slots e = true -> no_code e = true.

(* No call: [no_code], and [Let_static]s of code too, whose bodies meet
   [no_code]. *)
Definition binding_no_call (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => no_code (code_body c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding_no_code binding
  end.

Fixpoint no_call (e : expr) : bool :=
  match e with
  | Let _ kind defining_expr body =>
    named_no_code kind defining_expr && no_call body
  | Let_static bindings body =>
    forallb binding_no_call bindings && no_call body
  | Let_cont_non_recursive _ h body => handler_no_call h && no_call body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_no_call h) handlers && no_call body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_no_call (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => no_call handler
  end.

(* A unit without code meets it. *)
Definition no_code_no_call_statement : Prop :=
  ∀ e : expr, no_code e = true -> no_call e = true.

(* ======================================================================== *)
(* Exits                                                                    *)
(* ======================================================================== *)

(* An [Apply_cont] of a continuation of the unit gives it one value. *)
Definition apply_cont_exits_one_value (u : unit_interface) (ac : apply_cont)
    : bool :=
  if among (ac_continuation ac) [unit_return u; unit_exn u]
  then Nat.eqb (length (ac_args ac)) 1
  else true.

(* A call that returns to a continuation of the unit returns one value,
   and one whose exception goes to a continuation of the unit passes no
   extra argument with it. *)
Definition apply_exits_one_value (u : unit_interface) (ap : apply_expr)
    : bool :=
  (if among (ap_continuation ap) [unit_return u; unit_exn u]
   then Nat.eqb (length (unarize (ap_return_arity ap))) 1
   else true)
  && (if among (ap_exn_continuation ap) [unit_return u; unit_exn u]
      then Nat.eqb (length (ap_exn_extra_args ap)) 0
      else true).

(* Every [Apply_cont] of a continuation of the unit, and every call that
   returns to one, in the expression or in a handler of it, gives it one
   value. *)
Fixpoint exits_one_value (u : unit_interface) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => exits_one_value u body
  | Let_cont_non_recursive _ h body =>
    handler_exits_one_value u h && exits_one_value u body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_exits_one_value u h) handlers
    && exits_one_value u body
  | Apply ap => apply_exits_one_value u ap
  | Invalid => true
  | Apply_cont ac => apply_cont_exits_one_value u ac
  | Switch _ arms =>
    forallb (fun '(_, ac) => apply_cont_exits_one_value u ac) arms
  end
with handler_exits_one_value (u : unit_interface) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => exits_one_value u handler
  end.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition rebuild_refines_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    closure_free e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    program_refines e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit of FlowExample with a [Let] that has no use and a parameter
   that has no use:

     let symbol s50 = { 7 } in
     let x10 = { s50; 2 } in
     let x14 = 3 in              (no use)
     let x11 = x10.(0) in
     let x12 = x11.(0) in
     let_cont k5 x13 x15 = return x13 in     ([x15] has no use)
     k5 x12 x14

   Its least solution is that of FlowExample with the constants carried
   to [x14] and to [x15] ([alias], [any_source]); neither has a use. *)
Module RebuildExample.

Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [Const (Tagged_immediate 7)])]
    (Let 10 Value
       (FlowExample.make_block [Symbol 50; Const (Tagged_immediate 2)])
       (Let 14 Value (Simple (Const (Tagged_immediate 3)))
          (Let 11 Value (FlowExample.load 0 (Var 10))
             (Let 12 Value (FlowExample.load 0 (Var 11))
                (Let_cont_non_recursive 5
                   (Handler [13; 15] (make_handler_metadata [Value; Value])
                      (FlowExample.return_to 1 [Var 13]))
                   (FlowExample.return_to 5 [Var 12; Var 14])))))).

Definition x14 : node := Node_variable 14.
Definition x15 : node := Node_variable 15.

Definition solution : tables :=
  let '(Tables a u s hu hs us so rf ef fu) := FlowExample.solution in
  Tables
    ((x14, FlowExample.constants) :: (x15, x14) :: a) u (x14 :: x15 :: s) hu
    (x14 :: x15 :: hs) us so rf ef fu.

(* The rebuilt unit. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [Const (Tagged_immediate 7)])]
    (Let 10 Value (FlowExample.make_block [Symbol 50; Const (Poison Value)])
       (Let 11 Value (FlowExample.load 0 (Var 10))
          (Let 12 Value (FlowExample.load 0 (Var 11))
             (Let_cont_non_recursive 5
                (Handler [13] (make_handler_metadata [Value])
                   (FlowExample.return_to 1 [Var 13]))
                (FlowExample.return_to 5 [Var 12]))))).

End RebuildExample.

(* The example meets the premises of [rebuild_refines_statement], with
   its least solution; the rebuild removes a field, a [Let], a parameter
   and an argument; and the unit exits with 7, as the rebuilt unit
   does. *)
Definition example_statement : Prop :=
  is_local_slots_only FlowExample.is_local
  ∧ (∀ f : field, FlowExample.is_local f = false)
  ∧ in_flow_fragment RebuildExample.unit_expr = true
  ∧ closure_free RebuildExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface RebuildExample.unit_expr = true
  ∧ exits_one_value FlowExample.interface RebuildExample.unit_expr = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface RebuildExample.unit_expr)
      RebuildExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface RebuildExample.unit_expr)
      RebuildExample.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables RebuildExample.solution ->
       check_closed FlowExample.is_local
         (traverse_unit Never FlowExample.interface RebuildExample.unit_expr)
         T
       = false)
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface RebuildExample.unit_expr)
      RebuildExample.solution false RebuildExample.unit_expr
    = RebuildExample.rebuilt
  ∧ run 8 (State RebuildExample.unit_expr [] [] empty_static_env empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap
  ∧ run 7 (State RebuildExample.rebuilt [] [] empty_static_env empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap.

(* ======================================================================== *)
(* What the premises are for                                                *)
(* ======================================================================== *)

(* [let x10 = 5 in return 1 x10]: two values given to the return
   continuation of the unit, which the traversal gives one parameter.
   [x10] has no use; the rebuild removes its [Let] and gives poison in
   its place. The tables are the least solution. *)
Module TwoValuesExample.

Local Open Scope positive_scope.

Definition unit_expr : expr :=
  Let 10 Value (Simple (Const (Tagged_immediate 5)))
    (FlowExample.return_to 1 [Const (Tagged_immediate 1); Var 10]).

Definition solution : tables :=
  {| t_alias :=
       [(FlowExample.x10, FlowExample.constants);
        (FlowExample.returned, FlowExample.constants)];
     t_any_usage :=
       [FlowExample.returned; FlowExample.raised; FlowExample.constants];
     t_any_source :=
       [FlowExample.constants; FlowExample.x10; FlowExample.returned];
     t_has_usage :=
       [FlowExample.returned; FlowExample.raised; FlowExample.constants];
     t_has_source :=
       [FlowExample.constants; FlowExample.x10; FlowExample.returned];
     t_usages := []; t_sources := []; t_reading_field := [];
     t_escaping_field := []; t_field_used := [] |}.

Definition rebuilt : expr :=
  FlowExample.return_to 1 [Const (Tagged_immediate 1); Const (Poison Value)].

End TwoValuesExample.

Definition exits_one_value_needed_statement : Prop :=
  in_flow_fragment TwoValuesExample.unit_expr = true
  ∧ closure_free TwoValuesExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface TwoValuesExample.unit_expr = true
  ∧ exits_one_value FlowExample.interface TwoValuesExample.unit_expr = false
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface TwoValuesExample.unit_expr)
      TwoValuesExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface TwoValuesExample.unit_expr)
      TwoValuesExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface TwoValuesExample.unit_expr)
      TwoValuesExample.solution false TwoValuesExample.unit_expr
    = TwoValuesExample.rebuilt
  ∧ ~ program_refines TwoValuesExample.unit_expr TwoValuesExample.rebuilt.

(* The unit

     let x20 = 1.0 in                    (a naked float)
     let x21 = box x20 in
     let_cont k6 x22 =
       let x23 = get_tag x22 in          (never run)
       return x23
     in
     switch 0 with 0 -> return 0 | 1 -> k6 x21

   [x21] has a use, by the [Get_tag] of [x22], but nothing reads the
   number it boxes, which has no use. The tables are the least solution
   with the field of [x21] said to be used: they are closed, but no rule
   gives that tuple. The rebuild keeps the boxing and gives it poison in
   place of [x20], whose [Let] it removes; the boxing of poison has no
   value. *)
Module UnreadFieldExample.

Local Open Scope positive_scope.

Definition handler_k6 : continuation_handler :=
  Handler [22] (make_handler_metadata [Value])
    (Let 23 (Naked_number Kind_naked_immediate) (Prim (Unary Get_tag) [Var 22])
       (FlowExample.return_to 1 [Var 23])).

Definition box_float (s : simple) : named :=
  Prim (Unary (Box_number Boxable_naked_float)) [s].

Definition arms : list (Z * apply_cont) :=
  [(0%Z,
    {| ac_continuation := 1; ac_args := [Const (Tagged_immediate 0)];
       ac_trap_action := None |});
   (1%Z,
    {| ac_continuation := 6; ac_args := [Var 21]; ac_trap_action := None |})].

Definition unit_expr : expr :=
  Let 20 (Naked_number Kind_naked_float) (Simple (Const (Naked_float 1)))
    (Let 21 Value (box_float (Var 20))
       (Let_cont_non_recursive 6 handler_k6
          (Switch (Const (Naked_immediate 0)) arms))).

Definition x20 : node := Node_variable 20.
Definition x21 : node := Node_variable 21.
Definition x22 : node := Node_variable 22.
Definition x23 : node := Node_variable 23.
Definition constants : node := Node_all_constants.
Definition returned : node := Node_made 1.
Definition raised : node := Node_made 2.

Definition solution : tables :=
  {| t_alias :=
       [(x20, constants); (returned, x23); (returned, constants); (x22, x21)];
     t_any_usage := [returned; raised; constants; x23];
     t_any_source := [constants; x20; returned];
     t_has_usage := [returned; raised; constants; x23; x22; x21];
     t_has_source := [constants; x20; returned; x21; x22];
     t_usages := [(x22, x22); (x21, x22)];
     t_sources := [(x21, x21); (x22, x21)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used := [(x21, Field_boxed_number Boxable_naked_float)] |}.

Definition rebuilt : expr :=
  Let 21 Value (box_float (Const (Poison (Naked_number Kind_naked_float))))
    (Let_cont_non_recursive 6
       (Handler [22] (make_handler_metadata [Value]) Invalid)
       (Switch (Const (Naked_immediate 0)) arms)).

End UnreadFieldExample.

Definition field_used_exact_needed_statement : Prop :=
  in_flow_fragment UnreadFieldExample.unit_expr = true
  ∧ closure_free UnreadFieldExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface UnreadFieldExample.unit_expr
    = true
  ∧ exits_one_value FlowExample.interface UnreadFieldExample.unit_expr
    = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface UnreadFieldExample.unit_expr)
      UnreadFieldExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface UnreadFieldExample.unit_expr)
      UnreadFieldExample.solution
    = false
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface UnreadFieldExample.unit_expr)
      UnreadFieldExample.solution false UnreadFieldExample.unit_expr
    = UnreadFieldExample.rebuilt
  ∧ ~ program_refines UnreadFieldExample.unit_expr
        UnreadFieldExample.rebuilt.

(* ======================================================================== *)
(* What the second slice of the fragment needs                              *)
(* ======================================================================== *)

(* The statement does not carry over as it stands to the units with
   closures, code and calls ([in_flow_fragment] without
   [closure_free]): each of the two units below meets every premise of
   [rebuild_refines_statement] but [closure_free], no field is local and
   it has no projection, its tables (worked out by iterating the rules
   on its graph) pass the two checks, and its rebuild does not refine
   it. *)

(* A closure that reaches an exit through a variable:

     let code f50 x40 = return x40 in       (bound by a [Let_static])
     let x10 = 5 in
     let x11 = closure f50 with { w60 = x10 } in
     return x11

   No projection names [w60], so the traversal gives the closure no fact
   of its value slot ([unit_kept]), no rule says it used, and the
   rebuild removes it ([rebuild_value_slots]). The unit exits with a
   closure that holds 5 in [w60], the rebuilt unit with one that holds
   nothing. Nothing in the unit can read the slot, so no observation of
   the closure tells the two apart; but exits are compared as values. A
   closure may also leave the unit inside an exception raised by code,
   through the exception continuation of a call, which no check of the
   [Apply_cont]s of the unit sees. *)
Module ClosureExitExample.

Local Open Scope positive_scope.

Definition identity_code : code :=
  Code [40] 41 3 4
    (make_code_metadata [Singleton Value] [Singleton Value])
    (FlowExample.return_to 3 [Var 40]).

Definition unit_expr : expr :=
  Let_static [Bind_code 50 identity_code]
    (Let 10 Value (Simple (Const (Tagged_immediate 5)))
       (Let 11 Value (Closure_of_set 70 [(70, 50)] [(60, Var 10)])
          (FlowExample.return_to 1 [Var 11]))).

Definition solution : tables :=
  {| t_alias :=
       [(Node_made 1, Node_variable 11); (Node_made 3, Node_variable 40);
        (Node_made 5, Node_variable 11); (Node_variable 10, Node_all_constants);
        (Node_variable 40, Node_made 6); (Node_variable 41, Node_made 5)];
     t_any_usage :=
       [Node_code_id 50; Node_made 1; Node_made 2; Node_made 3; Node_made 4;
        Node_made 6; Node_made 7; Node_made 8; Node_variable 11;
        Node_variable 40];
     t_any_source :=
       [Node_all_constants; Node_code_id 50; Node_made 3; Node_made 6;
        Node_variable 10; Node_variable 40];
     t_has_usage :=
       [Node_code_id 50; Node_made 1; Node_made 2; Node_made 3; Node_made 4;
        Node_made 6; Node_made 7; Node_made 8; Node_variable 11;
        Node_variable 40];
     t_has_source :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 3;
        Node_made 5; Node_made 6; Node_made 7; Node_made 8; Node_variable 10;
        Node_variable 11; Node_variable 40; Node_variable 41];
     t_usages := [];
     t_sources :=
       [(Node_made 1, Node_variable 11); (Node_made 5, Node_variable 11);
        (Node_made 7, Node_made 7); (Node_made 8, Node_made 8);
        (Node_variable 11, Node_variable 11);
        (Node_variable 41, Node_variable 11)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(Node_made 7, Field_code_id_of_call_witness);
        (Node_made 7, Field_return_of_call (Normal 0%nat));
        (Node_made 7, Field_return_of_call Exn);
        (Node_made 8, Field_code_id_of_call_witness);
        (Node_made 8, Field_return_of_call (Normal 0%nat));
        (Node_made 8, Field_return_of_call Exn);
        (Node_variable 11, Field_call_witness Known_arity_code_pointer);
        (Node_variable 11, Field_call_witness Unknown_arity_code_pointer);
        (Node_variable 11, Field_function_slot 70)] |}.

Definition rebuilt : expr :=
  Let_static [Bind_code 50 identity_code]
    (Let 11 Value (Closure_of_set 70 [(70, 50)] [])
       (FlowExample.return_to 1 [Var 11])).

End ClosureExitExample.

Definition closure_exit_counterexample_statement : Prop :=
  in_flow_fragment ClosureExitExample.unit_expr = true
  ∧ projection_free ClosureExitExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface ClosureExitExample.unit_expr
    = true
  ∧ exits_one_value FlowExample.interface ClosureExitExample.unit_expr
    = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution false ClosureExitExample.unit_expr
    = ClosureExitExample.rebuilt
  ∧ run 4
      (State ClosureExitExample.unit_expr [] [] empty_static_env empty_heap)
    = Exit 1%positive
        [Value_closure_of_set 70%positive [(70%positive, 50%positive)]
           [(60%positive, Value_const (Tagged_immediate 5))]]
        empty_heap
  ∧ run 3
      (State ClosureExitExample.rebuilt [] [] empty_static_env empty_heap)
    = Exit 1%positive
        [Value_closure_of_set 70%positive [(70%positive, 50%positive)] []]
        empty_heap
  ∧ ~ program_refines ClosureExitExample.unit_expr ClosureExitExample.rebuilt.

(* A call whose return continuation takes a value of another kind than
   the return arity of the call:

     let code f50 x40 = return (naked 5) in   (result arity: one value)
     let x60 = closure f50 in
     let_cont k20 (x32 : naked immediate) = return 0 in
     f50 x60 9 -> k20

   The code returns a naked immediate though its result arity is a
   value: the machine compares the arity of the call with that of the
   code, not with what is returned ([call_code]), and the handler of
   [k20] takes it. [x32] has no use, so the rebuild has the call return
   to a wrapper ([make_apply_wrapper], the continuation 61), whose
   parameter has the kind of the return arity of the call: the wrapper
   refuses the naked immediate, and the rebuilt unit goes wrong where
   the unit returns 0. The compiler's units are well-kinded and do not
   have such a call; the model asks nothing of the kinds of a unit. *)
Module WrapperKindsExample.

Local Open Scope positive_scope.

Definition naked_code : code :=
  Code [40] 41 3 4
    (make_code_metadata [Singleton Value] [Singleton Value])
    (FlowExample.return_to 3 [Const (Naked_immediate 5)]).

Definition call_to (k : continuation) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 50;
     ap_args := [Const (Tagged_immediate 9)]; ap_continuation := k;
     ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 50 naked_code]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [32]
             (make_handler_metadata [Naked_number Kind_naked_immediate])
             (FlowExample.return_to 1 [Const (Tagged_immediate 0)]))
          (Apply (call_to 20)))).

Definition solution : tables :=
  {| t_alias :=
       [(Node_made 1, Node_all_constants); (Node_made 10, Node_code_id 50);
        (Node_made 11, Node_made 7); (Node_made 2, Node_made 4);
        (Node_made 3, Node_all_constants); (Node_made 5, Node_variable 60);
        (Node_made 6, Node_all_constants); (Node_made 9, Node_made 11);
        (Node_variable 32, Node_made 3); (Node_variable 40, Node_made 6);
        (Node_variable 41, Node_made 5)];
     t_any_usage :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 2; Node_made 4];
     t_any_source :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 3; Node_made 6; Node_variable 32; Node_variable 40];
     t_has_usage :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 11; Node_made 2; Node_made 4; Node_made 7; Node_made 9;
        Node_variable 60];
     t_has_source :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 11; Node_made 3; Node_made 5; Node_made 6; Node_made 7;
        Node_made 8; Node_made 9; Node_variable 32; Node_variable 40;
        Node_variable 41; Node_variable 60];
     t_usages :=
       [(Node_made 11, Node_made 9); (Node_made 7, Node_made 9);
        (Node_made 9, Node_made 9); (Node_variable 60, Node_variable 60)];
     t_sources :=
       [(Node_made 11, Node_made 7); (Node_made 5, Node_variable 60);
        (Node_made 7, Node_made 7); (Node_made 8, Node_made 8);
        (Node_made 9, Node_made 7); (Node_variable 41, Node_variable 60);
        (Node_variable 60, Node_variable 60)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(Node_made 7, Field_code_id_of_call_witness);
        (Node_made 7, Field_return_of_call Exn);
        (Node_variable 60, Field_call_witness Known_arity_code_pointer)] |}.

Definition rebuilt : expr :=
  Let_static [Bind_code 50 naked_code]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let_cont_non_recursive 20
          (Handler [] (make_handler_metadata [])
             (FlowExample.return_to 1 [Const (Tagged_immediate 0)]))
          (Let_cont_non_recursive 61
             (Handler [61] (make_handler_metadata [Value])
                (FlowExample.return_to 20 []))
             (Apply (call_to 61))))).

End WrapperKindsExample.

Definition wrapper_kinds_counterexample_statement : Prop :=
  in_flow_fragment WrapperKindsExample.unit_expr = true
  ∧ projection_free WrapperKindsExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface WrapperKindsExample.unit_expr
    = true
  ∧ exits_one_value FlowExample.interface WrapperKindsExample.unit_expr
    = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface WrapperKindsExample.unit_expr)
      WrapperKindsExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface WrapperKindsExample.unit_expr)
      WrapperKindsExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface WrapperKindsExample.unit_expr)
      WrapperKindsExample.solution false WrapperKindsExample.unit_expr
    = WrapperKindsExample.rebuilt
  ∧ run 6
      (State WrapperKindsExample.unit_expr [] [] empty_static_env empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap
  ∧ run 6
      (State WrapperKindsExample.rebuilt [] [] empty_static_env empty_heap)
    = Undefined
  ∧ ~ program_refines WrapperKindsExample.unit_expr
        WrapperKindsExample.rebuilt.

(* ======================================================================== *)
(* Refinement up to observation                                             *)
(* ======================================================================== *)

(* [R] of the elements at each position of two lists of one length. *)
Definition values_related (R : value -> value -> Prop)
    : list value -> list value -> Prop :=
  fix values_related (l l' : list value) {struct l} : Prop :=
    match l, l' with
    | [], [] => True
    | v :: l, v' :: l' => R v v' ∧ values_related l l'
    | _, _ => False
    end.

(* What the world outside the unit can tell of two values is the same:
   they are equal, but for the value slots of closures, which only the
   projections of the unit read. *)
Fixpoint observed_equal (v v' : value) {struct v} : Prop :=
  match v, v' with
  | Value_const c, Value_const c' => c = c'
  | Value_closure code _, Value_closure code' _ => code = code'
  | Value_closure_of_set slot decls _, Value_closure_of_set slot' decls' _ =>
    slot = slot' ∧ decls = decls'
  | Value_symbol s, Value_symbol s' => s = s'
  | Value_block tag fields, Value_block tag' fields' =>
    tag = tag' ∧ values_related (fun w w' => observed_equal w w') fields fields'
  | Value_immutable_array elements, Value_immutable_array elements' =>
    values_related (fun w w' => observed_equal w w') elements elements'
  | Value_boxed_number c, Value_boxed_number c' => c = c'
  | Value_location l, Value_location l' => l = l'
  | Value_partial_application callee args arity,
    Value_partial_application callee' args' arity' =>
    observed_equal callee callee'
    ∧ values_related (fun w w' => observed_equal w w') args args'
    ∧ arity = arity'
  | _, _ => False
  end.

(* [refines], with the exits of the two states compared by
   [observed_equal]: an exit of the source is one of the target through
   the same continuation, with values observed equal and the same heap;
   the target ends only as the source does, or where the source goes
   wrong. *)
Definition refines_observably (source target : state) : Prop :=
  (∀ (k : continuation) (vs : list value) (hp : heap),
     behaves source (Exits k vs hp) ->
     ∃ vs' : list value,
       behaves target (Exits k vs' hp)
       ∧ values_related observed_equal vs vs')
  ∧ (∀ (k : continuation) (vs' : list value) (hp : heap),
       behaves target (Exits k vs' hp) ->
       (∃ vs : list value,
          behaves source (Exits k vs hp)
          ∧ values_related observed_equal vs vs')
       ∨ behaves source Goes_wrong)
  ∧ (behaves target Goes_wrong -> behaves source Goes_wrong).

Definition program_refines_observably (source target : expr) : Prop :=
  refines_observably (State source [] [] empty_static_env empty_heap)
    (State target [] [] empty_static_env empty_heap).

Definition refines_observably_of_refines_statement : Prop :=
  ∀ source target : state,
    refines source target -> refines_observably source target.

(* ======================================================================== *)
(* What code returns, and what calls return to                              *)
(* ======================================================================== *)

Definition simple_kind (kinds : finite_map flambda_kind) (s : simple)
    : option flambda_kind :=
  match s with
  | Var x => lookup x kinds
  | Symbol _ => Some Value
  | Const c => Some (kind_of_const c)
  end.

(* The simples have, one for one, the kinds [expected]. *)
Fixpoint simples_of_kinds (kinds : finite_map flambda_kind)
    (args : list simple) (expected : list flambda_kind) : bool :=
  match args, expected with
  | [], [] => true
  | s :: args, k :: expected =>
    match simple_kind kinds s with
    | Some k' => kind_eqb k' k && simples_of_kinds kinds args expected
    | None => false
    end
  | _, _ => false
  end.

Definition handler_kinds (h : continuation_handler) : list flambda_kind :=
  match h with
  | Handler _ metadata _ => hm_params_kinds metadata
  end.

(* An [Apply_cont] of the return continuation of the code it is in
   gives it simples of the kinds of the result arity. *)
Definition returns_kinded (kinds : finite_map flambda_kind)
    (ks : finite_map (bool * list flambda_kind)) (ac : apply_cont) : bool :=
  match lookup (ac_continuation ac) ks with
  | Some (true, expected) => simples_of_kinds kinds (ac_args ac) expected
  | _ => true
  end.

(* [kinds] has the kinds of the variables in scope, [ks] the kinds of the
   parameters of the continuations in scope that the term binds, each
   with whether it is the return continuation of the code: each code
   applies its return continuation to simples of the kinds of its result
   arity, and the return continuation of each call, where the term binds
   it, takes the kinds of the return arity of the call. *)
Fixpoint results_kinded (kinds : finite_map flambda_kind)
    (ks : finite_map (bool * list flambda_kind)) (e : expr) : bool :=
  match e with
  | Let x kind _ body => results_kinded ((x, kind) :: kinds) ks body
  | Let_static bindings body =>
    forallb binding_results_kinded bindings && results_kinded kinds ks body
  | Let_cont_non_recursive k h body =>
    handler_results_kinded kinds ks h
    && results_kinded kinds ((k, (false, handler_kinds h)) :: ks) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_results_kinded kinds
           (map (fun '(k, h) => (k, (false, handler_kinds h))) handlers ++ ks)
           h)
      handlers
    && results_kinded kinds
         (map (fun '(k, h) => (k, (false, handler_kinds h))) handlers ++ ks)
         body
  | Apply ap =>
    match lookup (ap_continuation ap) ks with
    | Some (_, expected) => kinds_eqb expected (unarize (ap_return_arity ap))
    | None => true
    end
  | Apply_cont ac => returns_kinded kinds ks ac
  | Switch _ arms => forallb (fun '(_, ac) => returns_kinded kinds ks ac) arms
  | Invalid => true
  end
with handler_results_kinded (kinds : finite_map flambda_kind)
    (ks : finite_map (bool * list flambda_kind)) (h : continuation_handler)
    : bool :=
  match h with
  | Handler params metadata handler =>
    results_kinded (rev (combine params (hm_params_kinds metadata)) ++ kinds)
      ks handler
  end
with binding_results_kinded (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_results_kinded c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_results_kinded (c : code) : bool :=
  match c with
  | Code params my_closure return_continuation _ metadata body =>
    results_kinded
      (rev (combine params (unarize (cm_params_arity metadata)))
       ++ [(my_closure, Value)])
      [(return_continuation, (true, unarize (cm_result_arity metadata)))]
      body
  end.

(* ======================================================================== *)
(* The statement up to observation                                          *)
(* ======================================================================== *)

Definition rebuild_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    no_call e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    program_refines_observably e
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e).

(* The fragment of [no_code_no_slots] holds of more than nothing: a unit
   with a set of two closures without value slots, and a static set of
   the same functions, both of deleted code:

     let deleted_code f50 and f51 in
     let symbol s80 = closure at 70 of { 70 = f50; 71 = f51 }
     and symbol s81 = closure at 71 of { 70 = f50; 71 = f51 } in
     let x10 = closure at 70 of { 70 = f50; 71 = f51 } in
     let x11 = x10 moved from 70 to 71 in    (a projection of the set)
     let x12 = closure at 70 of { 70 = f50; 71 = f51 } in    (no use)
     let x13 = s80 moved from 70 to 71 in
     let x14 = { x11; x13 } in
     return x14

   Its least solution passes the checks. The rebuild keeps the first set
   with the projection that follows it, and the static set, whose
   symbols have a use, and removes [x12]; the two units exit with the
   same block, of the closure of the set at 71 and of [s81]. *)
Module ClosuresExample.

Local Open Scope positive_scope.

Definition function_decls : list (function_slot * code_id) :=
  [(70, 50); (71, 51)].

Definition closure_symbols : list (function_slot * symbol) :=
  [(70, 80); (71, 81)].

Definition static_set : list static_binding :=
  [Bind_deleted_code 50; Bind_deleted_code 51;
   Bind_symbol 80
     (Static_const_closure_of_set 70 closure_symbols function_decls []);
   Bind_symbol 81
     (Static_const_closure_of_set 71 closure_symbols function_decls [])].

Definition unit_expr : expr :=
  Let_static static_set
    (Let 10 Value (Closure_of_set 70 function_decls [])
       (Let 11 Value (Project_function_slot 70 71 (Var 10))
          (Let 12 Value (Closure_of_set 70 function_decls [])
             (Let 13 Value (Project_function_slot 70 71 (Symbol 80))
                (Let 14 Value (FlowExample.make_block [Var 11; Var 13])
                   (FlowExample.return_to 1 [Var 14])))))).

Definition solution : tables :=
  {| t_alias :=
       [(Node_made 1, Node_variable 14); (Node_variable 11, Node_made 3);
        (Node_variable 13, Node_symbol 81)];
     t_any_usage :=
       [Node_all_constants; Node_made 1; Node_made 10; Node_made 2; Node_made 3;
        Node_made 7; Node_made 8; Node_made 9; Node_symbol 80; Node_symbol 81;
        Node_variable 10; Node_variable 11; Node_variable 13; Node_variable 14];
     t_any_source :=
       [Node_all_constants; Node_made 10; Node_made 5; Node_made 6; Node_made 7;
        Node_made 8; Node_made 9];
     t_has_usage :=
       [Node_all_constants; Node_made 1; Node_made 10; Node_made 2; Node_made 3;
        Node_made 7; Node_made 8; Node_made 9; Node_symbol 80; Node_symbol 81;
        Node_variable 10; Node_variable 11; Node_variable 13; Node_variable 14];
     t_has_source :=
       [Node_all_constants; Node_made 1; Node_made 10; Node_made 3; Node_made 4;
        Node_made 5; Node_made 6; Node_made 7; Node_made 8; Node_made 9;
        Node_symbol 80; Node_symbol 81; Node_variable 10; Node_variable 11;
        Node_variable 12; Node_variable 13; Node_variable 14];
     t_usages :=
       [(Node_symbol 80, Node_symbol 80); (Node_variable 10, Node_variable 10)];
     t_sources :=
       [(Node_made 1, Node_variable 14); (Node_made 3, Node_made 3);
        (Node_made 4, Node_made 4); (Node_symbol 80, Node_symbol 80);
        (Node_symbol 81, Node_symbol 81); (Node_variable 10, Node_variable 10);
        (Node_variable 11, Node_made 3); (Node_variable 12, Node_variable 12);
        (Node_variable 13, Node_symbol 81);
        (Node_variable 14, Node_variable 14)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(Node_made 10, Field_code_id_of_call_witness);
        (Node_made 3, Field_call_witness Known_arity_code_pointer);
        (Node_made 3, Field_call_witness Unknown_arity_code_pointer);
        (Node_made 3, Field_function_slot 70);
        (Node_made 3, Field_function_slot 71);
        (Node_made 7, Field_code_id_of_call_witness);
        (Node_made 8, Field_code_id_of_call_witness);
        (Node_made 9, Field_code_id_of_call_witness);
        (Node_symbol 80, Field_call_witness Known_arity_code_pointer);
        (Node_symbol 80, Field_call_witness Unknown_arity_code_pointer);
        (Node_symbol 80, Field_function_slot 70);
        (Node_symbol 80, Field_function_slot 71);
        (Node_symbol 81, Field_call_witness Known_arity_code_pointer);
        (Node_symbol 81, Field_call_witness Unknown_arity_code_pointer);
        (Node_symbol 81, Field_function_slot 70);
        (Node_symbol 81, Field_function_slot 71);
        (Node_variable 10, Field_call_witness Known_arity_code_pointer);
        (Node_variable 10, Field_call_witness Unknown_arity_code_pointer);
        (Node_variable 10, Field_function_slot 70);
        (Node_variable 10, Field_function_slot 71);
        (Node_variable 14, Field_block 0%nat Value);
        (Node_variable 14, Field_block 1%nat Value);
        (Node_variable 14, Field_get_tag); (Node_variable 14, Field_is_int)] |}.

Definition rebuilt : expr :=
  Let_static static_set
    (Let 10 Value (Closure_of_set 70 function_decls [])
       (Let 11 Value (Project_function_slot 70 71 (Var 10))
          (Let 13 Value (Project_function_slot 70 71 (Symbol 80))
             (Let 14 Value (FlowExample.make_block [Var 11; Var 13])
                (FlowExample.return_to 1 [Var 14]))))).

Definition result : value :=
  Value_block 0 [Value_closure_of_set 71 function_decls []; Value_symbol 81].

End ClosuresExample.

Definition closures_example_statement : Prop :=
  in_flow_fragment ClosuresExample.unit_expr = true
  ∧ no_code_no_slots ClosuresExample.unit_expr = true
  ∧ results_kinded [] [] ClosuresExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface ClosuresExample.unit_expr = true
  ∧ exits_one_value FlowExample.interface ClosuresExample.unit_expr = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosuresExample.unit_expr)
      ClosuresExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosuresExample.unit_expr)
      ClosuresExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface ClosuresExample.unit_expr)
      ClosuresExample.solution false ClosuresExample.unit_expr
    = ClosuresExample.rebuilt
  ∧ behaves
      (State ClosuresExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [ClosuresExample.result] empty_heap)
  ∧ behaves
      (State ClosuresExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive [ClosuresExample.result] empty_heap).

(* The statement up to observation holds of more than nothing, and
   compares exits up to the value slots: a unit with a set of closures
   with two value slots, of which a projection reads one, and a static
   set with two, of which a projection reads one; the unit exits with a
   block of the two values read and of the closure:

     let deleted_code f50 and f51 in
     let symbol s80 = closure at 70 of { 70 = f50; 71 = f51 }
                      with { w62 = 7; w63 = 8 }
     and symbol s81 = closure at 71 of the same in
     let x10 = 5 in
     let x9 = 6 in
     let x11 = closure at 70 of { 70 = f50; 71 = f51 }
               with { w60 = x10; w61 = x9 } in
     let x12 = x11 moved from 70 to 71 in
     let x13 = w60 of x12 at 71 in
     let x14 = w62 of s81 at 71 in
     let x15 = { x13; x14; x11 } in
     return x15

   Its least solution passes the checks. The rebuild removes the value
   slots that no projection reads ([w61], [w63]) and [x9]; the unit exits
   with a closure that holds 5 and 6, the rebuilt unit with one that
   holds 5, observed equal. *)
Module ValueSlotsExample.

Local Open Scope positive_scope.

Definition function_decls : list (function_slot * code_id) :=
  [(70, 50); (71, 51)].

Definition closure_symbols : list (function_slot * symbol) :=
  [(70, 80); (71, 81)].

Definition tagged (n : Z) : simple := Const (Tagged_immediate n).

Definition static_set (value_slots : list (value_slot * simple))
    : list static_binding :=
  [Bind_deleted_code 50; Bind_deleted_code 51;
   Bind_symbol 80
     (Static_const_closure_of_set 70 closure_symbols function_decls
        value_slots);
   Bind_symbol 81
     (Static_const_closure_of_set 71 closure_symbols function_decls
        value_slots)].

Definition reads (value_slots : list (value_slot * simple)) : expr :=
  Let 11 Value (Closure_of_set 70 function_decls value_slots)
    (Let 12 Value (Project_function_slot 70 71 (Var 11))
       (Let 13 Value (Project_value_slot_from 71 60 (Var 12))
          (Let 14 Value (Project_value_slot_from 71 62 (Symbol 81))
             (Let 15 Value
                (FlowExample.make_block [Var 13; Var 14; Var 11])
                (FlowExample.return_to 1 [Var 15]))))).

Definition unit_expr : expr :=
  Let_static (static_set [(62, tagged 7); (63, tagged 8)])
    (Let 10 Value (Simple (tagged 5))
       (Let 9 Value (Simple (tagged 6))
          (reads [(60, Var 10); (61, Var 9)]))).

Definition solution : tables :=
  {| t_alias :=
       [(Node_made 1, Node_variable 15); (Node_variable 10, Node_all_constants);
        (Node_variable 12, Node_made 3); (Node_variable 13, Node_variable 10);
        (Node_variable 14, Node_all_constants);
        (Node_variable 9, Node_all_constants)];
     t_any_usage :=
       [Node_all_constants; Node_made 1; Node_made 2; Node_made 3; Node_made 4;
        Node_made 5; Node_variable 10; Node_variable 11; Node_variable 13;
        Node_variable 14; Node_variable 15];
     t_any_source :=
       [Node_all_constants; Node_made 4; Node_made 5; Node_made 6; Node_made 7;
        Node_variable 10; Node_variable 13; Node_variable 14; Node_variable 9];
     t_has_usage :=
       [Node_all_constants; Node_made 1; Node_made 2; Node_made 3; Node_made 4;
        Node_made 5; Node_symbol 81; Node_variable 10; Node_variable 11;
        Node_variable 12; Node_variable 13; Node_variable 14; Node_variable 15];
     t_has_source :=
       [Node_all_constants; Node_made 1; Node_made 3; Node_made 4; Node_made 5;
        Node_made 6; Node_made 7; Node_symbol 80; Node_symbol 81;
        Node_variable 10; Node_variable 11; Node_variable 12; Node_variable 13;
        Node_variable 14; Node_variable 15; Node_variable 9];
     t_usages :=
       [(Node_symbol 81, Node_symbol 81); (Node_variable 12, Node_variable 12)];
     t_sources :=
       [(Node_made 1, Node_variable 15); (Node_made 3, Node_made 3);
        (Node_symbol 80, Node_symbol 80); (Node_symbol 81, Node_symbol 81);
        (Node_variable 11, Node_variable 11); (Node_variable 12, Node_made 3);
        (Node_variable 15, Node_variable 15)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(Node_made 3, Field_call_witness Known_arity_code_pointer);
        (Node_made 3, Field_call_witness Unknown_arity_code_pointer);
        (Node_made 3, Field_function_slot 70);
        (Node_made 3, Field_function_slot 71);
        (Node_made 3, Field_value_slot 60);
        (Node_made 4, Field_code_id_of_call_witness);
        (Node_made 5, Field_code_id_of_call_witness);
        (Node_symbol 81, Field_value_slot 62);
        (Node_variable 11, Field_call_witness Known_arity_code_pointer);
        (Node_variable 11, Field_call_witness Unknown_arity_code_pointer);
        (Node_variable 11, Field_function_slot 70);
        (Node_variable 11, Field_function_slot 71);
        (Node_variable 11, Field_value_slot 60);
        (Node_variable 15, Field_block 0%nat Value);
        (Node_variable 15, Field_block 1%nat Value);
        (Node_variable 15, Field_block 2%nat Value);
        (Node_variable 15, Field_get_tag); (Node_variable 15, Field_is_int)] |}.

Definition rebuilt : expr :=
  Let_static (static_set [(62, tagged 7)])
    (Let 10 Value (Simple (tagged 5)) (reads [(60, Var 10)])).

Definition result (value_slots : list (value_slot * value)) : value :=
  Value_block 0
    [Value_const (Tagged_immediate 5); Value_const (Tagged_immediate 7);
     Value_closure_of_set 70 function_decls value_slots].

Definition source_result : value :=
  result [(60, Value_const (Tagged_immediate 5));
          (61, Value_const (Tagged_immediate 6))].

Definition rebuilt_result : value :=
  result [(60, Value_const (Tagged_immediate 5))].

End ValueSlotsExample.

Definition value_slots_example_statement : Prop :=
  in_flow_fragment ValueSlotsExample.unit_expr = true
  ∧ no_code ValueSlotsExample.unit_expr = true
  ∧ results_kinded [] [] ValueSlotsExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface ValueSlotsExample.unit_expr
    = true
  ∧ exits_one_value FlowExample.interface ValueSlotsExample.unit_expr = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface ValueSlotsExample.unit_expr)
      ValueSlotsExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface ValueSlotsExample.unit_expr)
      ValueSlotsExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface ValueSlotsExample.unit_expr)
      ValueSlotsExample.solution false ValueSlotsExample.unit_expr
    = ValueSlotsExample.rebuilt
  ∧ behaves
      (State ValueSlotsExample.unit_expr [] [] empty_static_env empty_heap)
      (Exits 1%positive [ValueSlotsExample.source_result] empty_heap)
  ∧ behaves
      (State ValueSlotsExample.rebuilt [] [] empty_static_env empty_heap)
      (Exits 1%positive [ValueSlotsExample.rebuilt_result] empty_heap)
  ∧ observed_equal ValueSlotsExample.source_result
      ValueSlotsExample.rebuilt_result
  ∧ ValueSlotsExample.source_result <> ValueSlotsExample.rebuilt_result.

(* The unit

     let code f50 x40 =
       let x42 = 8 in
       return x40 x42                    (two values)
     in
     let x11 = closure f50 in
     x11 7 -> return                     (a direct call to the unit)

   meets the premises of the statements but [exits_one_value], which a
   call that returns two values to the continuation of the unit fails.
   The traversal gives that continuation one parameter, and aliases the
   results of the call to its parameters as far as they go
   ([apply_deps], as traverse.ml); the second result, [x42], has no use,
   and the rebuild makes it poison. The unit exits with 7 and 8, the
   rebuilt unit with 7 and poison: not even observed equal. The
   compiler's units do not do this: the return arity of a call is that
   of its continuation, and the continuation of the unit takes one value
   (the module block), so [exits_one_value] is a condition of
   well-formedness that they meet, as for [Apply_cont]. *)
Module CallExitsExample.

Local Open Scope positive_scope.

Definition metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value; Singleton Value].

Definition two_results (second : simple) : expr :=
  FlowExample.return_to 3 [Var 40; second].

Definition call : apply_expr :=
  {| ap_callee := Var 11; ap_call_kind := Direct 50;
     ap_args := [Const (Tagged_immediate 7)];
     ap_continuation := 1; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value; Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4 metadata
          (Let 42 Value (Simple (Const (Tagged_immediate 8)))
             (two_results (Var 42))))]
    (Let 11 Value (Closure_of_set 70 [(70, 50)] []) (Apply call)).

Definition solution : tables :=
  {| t_alias :=
       [(Node_made 1, Node_made 3); (Node_made 10, Node_made 12);
        (Node_made 11, Node_code_id 50); (Node_made 12, Node_made 8);
        (Node_made 2, Node_made 5); (Node_made 3, Node_variable 40);
        (Node_made 4, Node_variable 42); (Node_made 6, Node_variable 11);
        (Node_made 7, Node_all_constants); (Node_variable 40, Node_made 7);
        (Node_variable 41, Node_made 6);
        (Node_variable 42, Node_all_constants)];
     t_any_usage :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 11;
        Node_made 2; Node_made 3; Node_made 5; Node_made 7; Node_variable 40];
     t_any_source :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 11;
        Node_made 3; Node_made 4; Node_made 7; Node_variable 40;
        Node_variable 42];
     t_has_usage :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 11; Node_made 12; Node_made 2; Node_made 3; Node_made 5;
        Node_made 7; Node_made 8; Node_variable 11; Node_variable 40];
     t_has_source :=
       [Node_all_constants; Node_code_id 50; Node_made 1; Node_made 10;
        Node_made 11; Node_made 12; Node_made 3; Node_made 4; Node_made 6;
        Node_made 7; Node_made 8; Node_made 9; Node_variable 11;
        Node_variable 40; Node_variable 41; Node_variable 42];
     t_usages :=
       [(Node_made 10, Node_made 10); (Node_made 12, Node_made 10);
        (Node_made 8, Node_made 10); (Node_variable 11, Node_variable 11)];
     t_sources :=
       [(Node_made 10, Node_made 8); (Node_made 12, Node_made 8);
        (Node_made 6, Node_variable 11); (Node_made 8, Node_made 8);
        (Node_made 9, Node_made 9); (Node_variable 11, Node_variable 11);
        (Node_variable 41, Node_variable 11)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(Node_made 8, Field_code_id_of_call_witness);
        (Node_made 8, Field_return_of_call (Normal 0%nat));
        (Node_made 8, Field_return_of_call Exn);
        (Node_variable 11, Field_call_witness Known_arity_code_pointer)] |}.

Definition rebuilt : expr :=
  Let_static
    [Bind_code 50
       (Code [40] 41 3 4 metadata (two_results (Const (Poison Value))))]
    (Let 11 Value (Closure_of_set 70 [(70, 50)] []) (Apply call)).

End CallExitsExample.

Definition call_exits_one_value_needed_statement : Prop :=
  in_flow_fragment CallExitsExample.unit_expr = true
  ∧ results_kinded [] [] CallExitsExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface CallExitsExample.unit_expr = true
  ∧ exits_one_value FlowExample.interface CallExitsExample.unit_expr = false
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface CallExitsExample.unit_expr)
      CallExitsExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface CallExitsExample.unit_expr)
      CallExitsExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface CallExitsExample.unit_expr)
      CallExitsExample.solution false CallExitsExample.unit_expr
    = CallExitsExample.rebuilt
  ∧ ~ program_refines_observably CallExitsExample.unit_expr
        CallExitsExample.rebuilt.

(* The unit of [closure_exit_counterexample_statement] meets
   [results_kinded], and its rebuild refines it up to observation. *)
Definition closure_exit_observed_statement : Prop :=
  results_kinded [] [] ClosureExitExample.unit_expr = true
  ∧ program_refines_observably ClosureExitExample.unit_expr
      ClosureExitExample.rebuilt.

(* The statement up to observation holds of a unit with code: the unit
   of [closure_exit_counterexample_statement], which binds code that it
   does not call, meets its premises with its least solution; the
   rebuild keeps the code, rebuilt, and removes the value slot. *)
Definition code_example_statement : Prop :=
  in_flow_fragment ClosureExitExample.unit_expr = true
  ∧ no_call ClosureExitExample.unit_expr = true
  ∧ results_kinded [] [] ClosureExitExample.unit_expr = true
  ∧ well_formed_unit FlowExample.interface ClosureExitExample.unit_expr
    = true
  ∧ exits_one_value FlowExample.interface ClosureExitExample.unit_expr
    = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution
    = true
  ∧ check_field_used_exact FlowExample.is_local
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution
    = true
  ∧ rebuild_unit_from_tables
      (traverse_unit Never FlowExample.interface ClosureExitExample.unit_expr)
      ClosureExitExample.solution false ClosureExitExample.unit_expr
    = ClosureExitExample.rebuilt.

(* The unit of [wrapper_kinds_counterexample_statement] fails
   [results_kinded], and its rebuild does not refine it even up to
   observation. *)
Definition results_kinded_needed_statement : Prop :=
  results_kinded [] [] WrapperKindsExample.unit_expr = false
  ∧ ~ program_refines_observably WrapperKindsExample.unit_expr
        WrapperKindsExample.rebuilt.
