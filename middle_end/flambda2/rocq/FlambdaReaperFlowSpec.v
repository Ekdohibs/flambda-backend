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

(* What a solution of the reaper's rules means on the machine: where the
   values of a run may be.

   This file is to be reviewed: [may_come_from], [flows] and what is
   made of them ([state_flows], [exit_flows]), the fragment
   ([in_flow_fragment]), and the statements. The proofs are in
   FlambdaReaperFlowProofs.v, which needs no review.

   The rules of the reaper (FlambdaReaperGraphSpec.closed) speak of
   allocations: a node has [sources], the nodes of the allocations its
   value may come from, or [any_source]. The machine has no identity of
   allocations: a block or a boxed number is its contents, and two made
   alike are one value (FlambdaValueSpec.v). So what a value is at a
   node cannot be read off the value. [flows v n], "the value [v] may be
   at the node [n]", says instead that there is an allocation that [v]
   may come from, and is defined by recursion on [v]:
   - a constant may be at a node that has any source: the traversal
     gives every constant the node [Node_all_constants], which has any
     source ([traverse_unit]), and the rules carry it along aliases and
     reads; a primitive that the traversal reads as no other gives its
     variable any source;
   - a block may be at [n] if there is a node [a] that [n] may come from
     ([may_come_from]), which has the two constructor facts of a block
     ([Is_int], [Get_tag]), and each field [i] of which may be at some
     node [from] with the fact that [a] holds [from] in its field [i];
   - a boxed number likewise, with its one field, whose contents is a
     constant;
   - a symbol is an allocation of its own: it may be at [n] if the node
     of the symbol is one that [n] may come from. What it points to is
     said once for the static environment ([statics_flow]), as for a
     block whose node is that of the symbol;
   - an immutable array has no allocation in the graph (the traversal
     reads [Make_immutable_array] and [Array_load] as other primitives:
     the variable bound has any source and a use of each argument), so
     it may be at a node with any source whose elements are constants or
     may each be at a node that has any usage if the array's node has a
     usage;
   - nothing else is in the fragment yet: closures, partial applications
     and locations are not at any node.
   The allocation is existential, and need not be the same for the
   value at two nodes; but at each node there is one, and what the rules
   give of the node's sources is given of it. [flows] does not depend on
   the static environment, which only grows.

   [may_come_from a n] has the five things that the rules carry from an
   allocation to the nodes its value reaches by aliases: [n] has a
   source; [a] is a source of [n], or [n] has any source; if [a] has any
   source so has [n]; if [n] has any usage so has [a]; and, if [n] has a
   usage, every use of [n] is one of [a] or [a] has any usage. The last
   is guarded by the usage of [n] because a value read out of an
   immutable array, or given to the identity, reaches its variable by a
   use and not by an alias: that variable has the uses of its reads,
   which are not those of the allocation, until it is itself used, when
   the rules give every argument of the primitive any usage.

   What the rebuild needs of the solution can be read off: a value at a
   node gives the node a source ([flows_has_source_statement]), so an
   [Apply_cont] whose argument has none is never run; and a read of a
   field that has a usage gives the field [field_used] in every
   allocation that the value read may come from
   ([read_field_used_statement]), so a field made poison is never read
   by a use.

   The statement is about a fragment of the terms
   ([in_flow_fragment]): a unit made of [Let]s of simples and of
   primitives, [Let_cont]s, recursive or not, [Apply_cont]s, [Switch]es
   and [Invalid], and [Let_static]s of blocks only. It has no call, no
   closure, no projection, and no code. Every primitive is in it: those
   that the traversal reads as a block made, a load, [Is_int], [Get_tag]
   and the boxing and unboxing of a number, and the others, of which
   the traversal knows only the arguments.

   For a unit in the fragment, and tables that pass the check of
   FlambdaReaperSolution.v ([check_closed], which is proved to imply
   [closed]), on the graph that the traversal makes of the unit
   ([traverse_unit]), every state of the run of the unit has every
   variable at its node, the variables of the environments that its
   continuations keep at theirs, and its static environment as the
   graph says ([state_flows]); and an exit through a continuation of
   the unit has each of its values at the node that the traversal gives
   the parameter of that continuation ([exit_flows],
   [unit_cont_nodes]: the made nodes 1 and 2, which have any usage,
   [unit_cont_nodes_used_statement]) ([flow_sound_statement]). The unit
   need not be well-formed: a continuation applied where it is not in
   scope is a way out of the term, for which the statement says
   nothing, and nothing else of well-formedness is used.

   The statement holds of more than nothing: a unit of the fragment,
   with a static block, a block made, two loads and a continuation, has
   its least solution pass the check, and its run exits
   ([example_statement]). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Definition named_in_flow_fragment (defining_expr : named) : bool :=
  match defining_expr with
  | Simple _ | Prim _ _ => true
  | Set_of_closures _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ => false
  end.

Definition binding_in_flow_fragment (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ _) => true
  | _ => false
  end.

Fixpoint in_flow_fragment (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_in_flow_fragment defining_expr && in_flow_fragment body
  | Let_static bindings body =>
    forallb binding_in_flow_fragment bindings && in_flow_fragment body
  | Let_cont_non_recursive _ h body =>
    handler_in_flow_fragment h && in_flow_fragment body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_in_flow_fragment h) handlers
    && in_flow_fragment body
  | Apply _ => false
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_in_flow_fragment (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => in_flow_fragment handler
  end.

(* ======================================================================== *)
(* Where a value may be                                                     *)
(* ======================================================================== *)

Section Flow.

Variable G : graph.
Variable sol : solution.

(* The value at [n] may come from the allocation [a]. *)
Definition may_come_from (a n : node) : Prop :=
  s_has_source sol n = true
  ∧ (s_any_source sol n = true ∨ s_sources sol n a = true)
  ∧ (s_any_source sol a = true -> s_any_source sol n = true)
  ∧ (s_any_usage sol n = true -> s_any_usage sol a = true)
  ∧ (s_has_usage sol n = true ->
     ∀ u : node,
       s_usages sol n u = true ->
       s_usages sol a u = true ∨ s_any_usage sol a = true).

(* [P i field] of each field of a list, the first at [i]. *)
Definition fields_from (P : nat -> value -> Prop)
    : nat -> list value -> Prop :=
  fix fields_from (i : nat) (fields : list value) : Prop :=
    match fields with
    | [] => True
    | field :: fields => P i field ∧ fields_from (S i) fields
    end.

(* The value [v] may be at the node [n]. *)
Fixpoint flows (v : value) (n : node) {struct v} : Prop :=
  match v with
  | Value_const _ => s_any_source sol n = true
  | Value_block _ fields =>
    ∃ a : node,
      may_come_from a n
      ∧ g_constructor G a Field_is_int Node_all_constants
      ∧ g_constructor G a Field_get_tag Node_all_constants
      ∧ fields_from
          (fun i field =>
             ∃ from : node,
               g_constructor G a (Field_block i Value) from
               ∧ flows field from)
          0 fields
  | Value_boxed_number c =>
    ∃ (a : node) (bn : boxable_number) (from : node),
      may_come_from a n
      ∧ is_number_of_kind bn c = true
      ∧ g_constructor G a (Field_boxed_number bn) from
      ∧ s_any_source sol from = true
  | Value_symbol s => may_come_from (Node_symbol s) n
  | Value_immutable_array elements =>
    s_any_source sol n = true
    ∧ every
        (fun element =>
           (∃ c : const, element = Value_const c)
           ∨ ∃ m : node,
               flows element m
               ∧ (s_has_usage sol n = true -> s_any_usage sol m = true))
        elements
  | Value_closure _ _ | Value_closure_of_set _ _ _ | Value_location _
  | Value_partial_application _ _ _ => False
  end.

(* What a symbol points to, as for a block whose allocation is the node
   of the symbol; and the symbol is an allocation of its own. Only
   blocks are in the fragment. *)
Definition static_value_flows (s : symbol)
    (allocated : statically_allocated_value) : Prop :=
  may_come_from (Node_symbol s) (Node_symbol s)
  ∧ match allocated with
    | Static_block _ fields =>
      g_constructor G (Node_symbol s) Field_is_int Node_all_constants
      ∧ g_constructor G (Node_symbol s) Field_get_tag Node_all_constants
      ∧ fields_from
          (fun i field =>
             ∃ from : node,
               g_constructor G (Node_symbol s) (Field_block i Value) from
               ∧ flows field from)
          0 fields
    | _ => False
    end.

Definition statics_flow (statics : static_env) : Prop :=
  ∀ (s : symbol) (allocated : statically_allocated_value),
    lookup s (static_values statics) = Some allocated ->
    static_value_flows s allocated.

(* Every variable is at its node. *)
Definition env_flows (vars : env) : Prop :=
  ∀ (x : variable) (v : value),
    lookup x vars = Some v -> flows v (Node_variable x).

(* The environments that a continuation keeps. Only handlers and ways
   out are in the fragment. *)
Inductive cont_flows : cont_value -> Prop :=
  | Closure_flows (h : continuation_handler) (vars : env) (conts : cont_env)
      (recursive : list (continuation * continuation_handler)) :
    env_flows vars ->
    (∀ (k : continuation) (c : cont_value), (k, c) ∈ conts -> cont_flows c) ->
    cont_flows (Closure h vars conts recursive)
  | Outer_flows (k : continuation) : cont_flows (Outer k).

Definition state_flows (s : state) : Prop :=
  env_flows (state_vars s)
  ∧ (∀ (k : continuation) (c : cont_value),
       (k, c) ∈ state_conts s -> cont_flows c)
  ∧ statics_flow (state_statics s).

(* The nodes of the parameters of the continuations of the unit, as
   [traverse_unit] starts the traversal with them: the made node 1 is
   the parameter of the continuation that the unit returns to, and 2
   that of its exception continuation. *)
Definition unit_conts (u : unit_interface) : finite_map (list node) :=
  [(unit_exn u, [Node_made 2]); (unit_return u, [Node_made 1])].

Definition unit_cont_nodes (u : unit_interface) (k : continuation)
    : list node :=
  match lookup k (unit_conts u) with
  | Some nodes => nodes
  | None => []
  end.

(* Each value of an exit is at the node of the parameter of the
   continuation of the unit at its position. *)
Definition exit_flows (u : unit_interface) (k : continuation)
    (vs : list value) : Prop :=
  ∀ (i : nat) (v : value) (p : node),
    nth_error vs i = Some v ->
    nth_error (unit_cont_nodes u k) i = Some p ->
    flows v p.

End Flow.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition flow_sound_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    match run n (State e [] [] empty_static_env empty_heap) with
    | Step s =>
      state_flows (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) s
    | Exit k vs _ =>
      exit_flows (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) u k vs
    | Undefined => True
    end.

(* The nodes of the continuations of the unit are those the traversal
   gives any usage. *)
Definition unit_cont_nodes_used_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (u : unit_interface) (e : expr)
    (k : continuation) (p : node),
    in_flow_fragment e = true ->
    p ∈ unit_cont_nodes u k ->
    g_any_usage (graph_of_facts (traverse_unit flag u e)) p.

(* A node at which a value may be has a source. *)
Definition flows_has_source_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (sol : solution) (v : value)
    (n : node),
    closed is_local G sol -> flows G sol v n -> s_has_source sol n = true.

(* A field that is read into a node that has a usage is used in every
   allocation that the value read may come from. *)
Definition read_field_used_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (sol : solution)
    (a base to_ from : node) (f : field),
    closed is_local G sol ->
    is_local f = false ->
    g_constructor G a f from ->
    g_accessor G to_ f base ->
    may_come_from sol a base ->
    s_has_usage sol to_ = true ->
    s_field_used sol a f = true.

(* And what the field holds may be where it is read into. *)
Definition read_flows_statement : Prop :=
  ∀ (is_local : field -> bool) (G : graph) (sol : solution)
    (a base to_ from : node) (f : field) (v : value),
    closed is_local G sol ->
    is_local f = false ->
    g_constructor G a f from ->
    g_accessor G to_ f base ->
    may_come_from sol a base ->
    flows G sol v from ->
    flows G sol v to_.

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The unit

     let symbol s50 = { 7 } in
     let x10 = { s50; 2 } in
     let x11 = x10.(0) in        (the symbol)
     let x12 = x11.(0) in        (7, read through the symbol)
     let_cont k5 x13 = return x13 in
     k5 x12

   returns to 1 and raises to 2; no field is local. Its least solution
   (worked out by hand from the rules): the read of [x10] gives [x11] the
   alias [s50] and the read of [x11] gives [x12] the alias of the
   constants, both by the sources of the block read
   ([alias_accessor_source]) and by its uses ([alias_constructor_usage]),
   the uses of [x11] being those of [s50]. Field 0 of [x10] and of [s50]
   is used; field 1 of [x10] is not. *)
Module FlowExample.

Local Open Scope positive_scope.

Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition load (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load (Block_access_values None None Any_value)
                 Immutable field)) [block].

Definition make_block (fields : list simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) fields.

Definition unit_expr : expr :=
  Let_static
    [Bind_symbol 50 (Static_const_block 0 [Const (Tagged_immediate 7)])]
    (Let 10 Value (make_block [Symbol 50; Const (Tagged_immediate 2)])
       (Let 11 Value (load 0 (Var 10))
          (Let 12 Value (load 0 (Var 11))
             (Let_cont_non_recursive 5
                (Handler [13] (make_handler_metadata [Value])
                   (return_to 1 [Var 13]))
                (return_to 5 [Var 12]))))).

Definition is_local (f : field) : bool := false.

Definition x10 : node := Node_variable 10.
Definition x11 : node := Node_variable 11.
Definition x12 : node := Node_variable 12.
Definition x13 : node := Node_variable 13.
Definition s50 : node := Node_symbol 50.
Definition constants : node := Node_all_constants.
Definition returned : node := Node_made 1.
Definition raised : node := Node_made 2.

Definition solution : tables :=
  {| t_alias := [(x13, x12); (returned, x13); (x11, s50); (x12, constants)];
     t_any_usage := [returned; raised; x13; x12; constants];
     t_any_source := [constants; x12; x13; returned];
     t_has_usage := [returned; raised; x13; x12; constants; x11; x10; s50];
     t_has_source := [constants; x12; x13; returned; s50; x10; x11];
     t_usages := [(x11, x11); (x10, x10); (s50, x11)];
     t_sources := [(x10, x10); (s50, s50); (x11, s50)];
     t_reading_field := []; t_escaping_field := [];
     t_field_used :=
       [(x10, Field_block 0 Value); (s50, Field_block 0 Value)] |}.

End FlowExample.

(* The example meets the premises of [flow_sound_statement], with its
   least solution, and its run exits with 7. *)
Definition example_statement : Prop :=
  is_local_slots_only FlowExample.is_local
  ∧ in_flow_fragment FlowExample.unit_expr = true
  ∧ check_closed FlowExample.is_local
      (traverse_unit Never FlowExample.interface FlowExample.unit_expr)
      FlowExample.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables FlowExample.solution ->
       check_closed FlowExample.is_local
         (traverse_unit Never FlowExample.interface FlowExample.unit_expr) T
       = false)
  ∧ run 7 (State FlowExample.unit_expr [] [] empty_static_env empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 7)] empty_heap.
