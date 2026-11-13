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

   This file is to be reviewed: the checks [closure_free] and
   [exits_one_value], the statement, and the examples, against
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
     ([exits_one_value_needed_statement]). [well_formed_unit] does not
     ask it; it is a condition that the compiler's units meet, and a
     row for requests/C0-well-formed.md.
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
   static sets, code, direct calls) is not covered: two units of it,
   which meet every premise but [closure_free], have rebuilds that do not
   refine them. A closure that reaches an exit loses the value slots
   that no projection names ([closure_exit_counterexample_statement]):
   an exact comparison of exits needs a premise that no closure reaches
   one, or the exits are to be compared up to the value slots that the
   unit cannot read. And the wrapper that the rebuild gives a call tests
   the kinds of the return arity of the call, which the machine does not
   test of what code returns ([wrapper_kinds_counterexample_statement]):
   a premise that the return arity of a call agrees with the parameters
   of its return continuation is needed.

   The statement holds of more than nothing ([example_statement]): a
   unit with a static block, a block one field of which no load reads, a
   [Let] with no use, and a continuation one parameter of which has no
   use. Its least solution passes the checks; its rebuild makes the
   field poison, removes the [Let], the parameter and the argument
   given to it; and the unit exits with 7, as the rebuilt unit does. *)

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

(* ======================================================================== *)
(* Exits                                                                    *)
(* ======================================================================== *)

(* An [Apply_cont] of a continuation of the unit gives it one value. *)
Definition apply_cont_exits_one_value (u : unit_interface) (ac : apply_cont)
    : bool :=
  if among (ac_continuation ac) [unit_return u; unit_exn u]
  then Nat.eqb (length (ac_args ac)) 1
  else true.

(* Every [Apply_cont] of a continuation of the unit, in the expression or
   in a handler of it, gives it one value. *)
Fixpoint exits_one_value (u : unit_interface) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => exits_one_value u body
  | Let_cont_non_recursive _ h body =>
    handler_exits_one_value u h && exits_one_value u body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_exits_one_value u h) handlers
    && exits_one_value u body
  | Apply _ | Invalid => true
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
