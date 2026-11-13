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
   ([in_flow_fragment] and what it is made of, [projection_free]), and
   the statements. The proofs are in FlambdaReaperFlowProofs.v, which
   needs no review.

   The rules of the reaper (FlambdaReaperGraphSpec.closed) speak of
   allocations: a node has [sources], the nodes of the allocations its
   value may come from, or [any_source]. The machine has no identity of
   allocations: a block, a boxed number or a closure is its contents,
   and two made alike are one value (FlambdaValueSpec.v). So what a
   value is at a node cannot be read off the value. [flows v n], "the
   value [v] may be at the node [n]", says instead that there is an
   allocation that [v] may come from, and is defined by recursion on
   [v]:
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
   - a closure of a set may be at [n] if there are nodes for the
     closures of the set ([sites]), that of its slot being one that [n]
     may come from; each is an allocation of its own, has what [called]
     asks for the code of its slot, holds the node of each closure of the
     set in its function slot, and holds, in each value slot that the
     traversal keeps ([kept]: those that a projection of the unit names,
     [unit_kept]), a node at which the value of the slot may be. [called]
     is a parameter: the statement takes [has_call_witness], that the
     allocation has a call witness of known arity; the proof uses more,
     the witness and [my_closure] of the code;
   - a symbol is an allocation of its own: it may be at [n] if the node
     of the symbol is one that [n] may come from. What it points to is
     said once for the static environment ([statics_flow]), as for a
     block or a closure whose node is that of the symbol;
   - an immutable array has no allocation in the graph (the traversal
     reads [Make_immutable_array] and [Array_load] as other primitives:
     the variable bound has any source and a use of each argument), so
     it may be at a node with any source whose elements are constants or
     may each be at a node that has any usage if the array's node has a
     usage;
   - partial applications and locations are not in the fragment, and are
     at no node; nor are the closures of the older form
     ([Value_closure]).
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

   The statement is about a fragment of the terms ([in_flow_fragment]):
   a unit made of [Let]s of simples, of primitives, of closures of sets
   ([Closure_of_set]) and of projections from them
   ([Project_value_slot_from], [Project_function_slot]), [Let_cont]s,
   recursive or not, [Apply_cont]s, [Switch]es, [Invalid], direct calls,
   and [Let_static]s of code, of blocks and of closures of static sets.
   Every primitive is in it. A direct call has no extra argument for its
   exception continuation, and the code it names, if the unit binds it,
   is not tupled and has as many parameters as the call has arguments
   (S7's hypothesis). The body of code binds nothing static, its two
   continuations differ, and it applies only continuations in scope,
   each to as many values as it has parameters, the return continuation
   of the code to as many as the code has results and its exception
   continuation to one ([conts_counted]); the closure symbols of a static
   set are symbols that the unit binds, and its slot is among its
   function slots; the code ids that the unit binds are distinct. Not in
   it: indirect calls, partial applications and over-applications,
   extra arguments of exception continuations, and the older forms of
   sets of closures.

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
   need not be well-formed: a continuation applied at toplevel where it
   is not in scope is a way out of the term, for which the statement
   says nothing, and the conditions that the proof needs are those of
   the fragment.

   Local fields. The statement asks that no field is local, unless the
   unit has no projection ([projection_free]). With local value slots it
   does not hold, and the reaper is unsound: a closure stored in an
   immutable array and loaded back has any source at its new node and
   no alias to its allocation, so the read of a local value slot of it
   gives what is read no source when nothing uses it, and the rebuild
   replaces by [Invalid] an [Apply_cont] that passes it on
   ([local_slot_counterexample_statement]: the least solution passes
   the check, the unit exits, and the rebuilt unit goes wrong).

   The statement holds of more than nothing: a unit with a static
   block, a block made, two loads and a continuation
   ([example_statement]), and a unit with code, a closure of it whose
   value slot is read through [my_closure], and a direct call
   ([call_example_statement]), each have their least solution pass the
   check, and their runs exit. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperSolution FlambdaReaperSolutionSpec
  FlambdaReaperRebuild.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

(* The code that a unit binds where its [Let_static]s are, as
   [static_names] has the names. *)
Fixpoint unit_codes (e : expr) : list (code_id * code) :=
  match e with
  | Let _ _ _ body => unit_codes body
  | Let_static bindings body =>
    flat_map
      (fun binding =>
         match binding with
         | Bind_code id c => [(id, c)]
         | Bind_deleted_code _ | Bind_symbol _ _ => []
         end)
      bindings
    ++ unit_codes body
  | Let_cont_non_recursive _ h body => handler_unit_codes h ++ unit_codes body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_unit_codes h) handlers ++ unit_codes body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_unit_codes (h : continuation_handler) : list (code_id * code) :=
  match h with
  | Handler _ _ handler => unit_codes handler
  end.

(* In the body of code, every continuation applied, or given to a call,
   is bound, and is given as many values as the traversal has nodes for
   its parameters: [ks] has the number of each continuation in scope. *)
Definition count_ok (ks : finite_map nat) (k : continuation) (n : nat)
    : bool :=
  match lookup k ks with
  | Some m => Nat.eqb m n
  | None => false
  end.

Fixpoint conts_counted (ks : finite_map nat) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => conts_counted ks body
  | Let_cont_non_recursive k h body =>
    handler_conts_counted ks h
    && conts_counted ((k, length (handler_params h)) :: ks) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_conts_counted
           (map (fun '(k, h) => (k, length (handler_params h))) handlers
            ++ ks)
           h)
      handlers
    && conts_counted
         (map (fun '(k, h) => (k, length (handler_params h))) handlers ++ ks)
         body
  | Apply ap =>
    count_ok ks (ap_continuation ap) (length (unarize (ap_return_arity ap)))
    && count_ok ks (ap_exn_continuation ap) 1
  | Apply_cont ac => count_ok ks (ac_continuation ac) (length (ac_args ac))
  | Switch _ arms =>
    forallb
      (fun '(_, ac) => count_ok ks (ac_continuation ac) (length (ac_args ac)))
      arms
  | Invalid => true
  end
with handler_conts_counted (ks : finite_map nat) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => conts_counted ks handler
  end.

Definition named_in_flow_fragment (defining_expr : named) : bool :=
  match defining_expr with
  | Simple _ | Prim _ _ | Closure_of_set _ _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ => true
  | Set_of_closures _ _ | Project_value_slot _ _ => false
  end.

(* A direct call, with no extra argument for its exception continuation,
   of code that, if the unit binds it, is not tupled and has as many
   parameters as the call has arguments. *)
Definition apply_in_flow_fragment (codes : list (code_id * code))
    (ap : apply_expr) : bool :=
  match ap_call_kind ap with
  | Direct id =>
    no_exn_extra_args ap
    && forallb
         (fun '(id', c) =>
            negb (Pos.eqb id id')
            || (negb (code_is_tupled c)
                && Nat.eqb (length (ap_args_arity ap))
                     (length (code_params_arity c))))
         codes
  | Indirect => false
  end.

(* The terms of the fragment, given the code and the symbols that the
   unit binds. *)
Fixpoint flow_fragment (codes : list (code_id * code)) (symbols : list symbol)
    (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_in_flow_fragment defining_expr && flow_fragment codes symbols body
  | Let_static bindings body =>
    forallb (binding_flow_fragment codes symbols) bindings
    && flow_fragment codes symbols body
  | Let_cont_non_recursive _ h body =>
    handler_flow_fragment codes symbols h && flow_fragment codes symbols body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_flow_fragment codes symbols h) handlers
    && flow_fragment codes symbols body
  | Apply ap => apply_in_flow_fragment codes ap
  | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_flow_fragment (codes : list (code_id * code))
    (symbols : list symbol) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => flow_fragment codes symbols handler
  end
with binding_flow_fragment (codes : list (code_id * code))
    (symbols : list symbol) (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ _) => true
  | Bind_symbol _
      (Static_const_closure_of_set slot closure_symbols function_decls _) =>
    is_bound slot function_decls
    && forallb (fun '(_, s) => among s symbols) closure_symbols
  | Bind_symbol _ _ => false
  | Bind_code _ c => code_flow_fragment codes symbols c
  | Bind_deleted_code _ => true
  end
with code_flow_fragment (codes : list (code_id * code))
    (symbols : list symbol) (c : code) : bool :=
  match c with
  | Code _ _ return_continuation exn_continuation metadata body =>
    negb (Pos.eqb return_continuation exn_continuation)
    && negb (binds_statics body)
    && conts_counted
         [(exn_continuation, 1);
          (return_continuation, length (unarize (cm_result_arity metadata)))]
         body
    && flow_fragment codes symbols body
  end.

(* The unit: its terms are of the fragment, and the code ids it binds are
   distinct. *)
Definition in_flow_fragment (e : expr) : bool :=
  flow_fragment (unit_codes e) (static_names symbols_of_binding e) e
  && distinct (static_names code_ids_of_binding e).

(* No projection from a closure, in the unit or in its code. *)
Definition named_projection_free (defining_expr : named) : bool :=
  match defining_expr with
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ => false
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _ => true
  end.

Fixpoint projection_free (e : expr) : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_projection_free defining_expr && projection_free body
  | Let_static bindings body =>
    forallb binding_projection_free bindings && projection_free body
  | Let_cont_non_recursive _ h body =>
    handler_projection_free h && projection_free body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_projection_free h) handlers
    && projection_free body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_projection_free (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => projection_free handler
  end
with binding_projection_free (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_projection_free c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_projection_free (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => projection_free body
  end.

(* ======================================================================== *)
(* Where a value may be                                                     *)
(* ======================================================================== *)

Section Flow.

Variable G : graph.
Variable sol : solution.
(* The value slots that the traversal gives constructor facts. *)
Variable kept : value_slot -> bool.
(* What the allocation of a closure has for the code of its slot. *)
Variable called : node -> code_id -> Prop.

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

(* The value [v] may be at the node [n]. A closure of a set may be at
   [n] if there are nodes for the closures of the set ([sites]), each an
   allocation of its own with what [called] asks for the code of its
   slot, the constructor facts of every function slot of the set, and
   the values of the value slots kept. *)
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
  | Value_closure_of_set slot function_decls value_slots =>
    ∃ sites : function_slot -> node,
      may_come_from (sites slot) n
      ∧ ∀ (f : function_slot) (id : code_id),
          lookup f function_decls = Some id ->
          may_come_from (sites f) (sites f)
          ∧ called (sites f) id
          ∧ (∀ (g : function_slot) (id' : code_id),
               lookup g function_decls = Some id' ->
               g_constructor G (sites f) (Field_function_slot g) (sites g))
          ∧ for_all_bindings
              (fun w field =>
                 kept w = true ->
                 ∃ from : node,
                   g_constructor G (sites f) (Field_value_slot w) from
                   ∧ flows field from)
              value_slots
  | Value_closure _ _ | Value_location _ | Value_partial_application _ _ _ =>
    False
  end.

(* What a symbol points to, as for a block or a closure whose allocation
   is the node of the symbol; and the symbol is an allocation of its
   own. The function slots of a static set lead to the symbols of its
   closures, each an allocation of its own. *)
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
    | Static_closure_of_set slot closure_symbols function_decls value_slots =>
      (∀ id : code_id,
         lookup slot function_decls = Some id -> called (Node_symbol s) id)
      ∧ (∀ (g : function_slot) (s' : symbol),
           lookup g closure_symbols = Some s' ->
           g_constructor G (Node_symbol s) (Field_function_slot g)
             (Node_symbol s')
           ∧ may_come_from (Node_symbol s') (Node_symbol s'))
      ∧ for_all_bindings
          (fun w field =>
             kept w = true ->
             ∃ from : node,
               g_constructor G (Node_symbol s) (Field_value_slot w) from
               ∧ flows field from)
          value_slots
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

(* The value slots that the traversal of a unit keeps: those that a
   projection names ([should_keep_value_slot]). *)
Definition unit_kept (e : expr) (w : value_slot) : bool :=
  among w (projected_value_slots e).

(* The allocation of a closure has a call witness of known arity. *)
Definition has_call_witness (G : graph) (a : node) (id : code_id) : Prop :=
  ∃ witness : node,
    g_constructor G a (Field_call_witness Known_arity_code_pointer) witness.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition flow_sound_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (n : nat),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    match run n (State e [] [] empty_static_env empty_heap) with
    | Step s =>
      state_flows (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) (unit_kept e)
        (has_call_witness (graph_of_facts (traverse_unit flag u e))) s
    | Exit k vs _ =>
      exit_flows (graph_of_facts (traverse_unit flag u e))
        (solution_of_tables T) (unit_kept e)
        (has_call_witness (graph_of_facts (traverse_unit flag u e))) u k vs
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
  ∀ (is_local : field -> bool) (G : graph) (sol : solution)
    (kept : value_slot -> bool) (called : node -> code_id -> Prop)
    (v : value) (n : node),
    closed is_local G sol -> flows G sol kept called v n ->
    s_has_source sol n = true.

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
    (kept : value_slot -> bool) (called : node -> code_id -> Prop)
    (a base to_ from : node) (f : field) (v : value),
    closed is_local G sol ->
    is_local f = false ->
    g_constructor G a f from ->
    g_accessor G to_ f base ->
    may_come_from sol a base ->
    flows G sol kept called v from ->
    flows G sol kept called v to_.

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

(* A call. The code 20 returns the value slot 60 of its closure, read
   through [my_closure] ([x31]); its parameter [x30] has no use. The
   unit makes the closure [x10] of it, with the value slots 60 and 61,
   of which no projection names 61, and calls it directly with 7,
   returning through the continuation 5 to 1:

     let code 20 (x30) my_closure x31 = return (x31.60) in
     let x10 = closure 20 { 60 = 5; 61 = 6 } in
     let_cont k5 x12 = return x12 in
     x10 7 -> k5

   The made nodes 1 and 2 are the parameters of the continuations of the
   unit; 3 to 8 are the code dep of 20 (its result 3, its exception 4,
   [my_closure] 5, its parameter 6, its witnesses 7 and 8); 9 to 11 are
   the witness of the call, [called] and [apply]. In the least solution
   (worked out from the rules, and checked to be least), the read of the
   witness of known arity of [x10] makes the witness of the call an
   alias of that of the code, and the code id used; [my_closure] is then
   an alias of [x10], and the read of the slot 60 an alias of the
   constant. The witness of the call is used, and the slot 60 of [x10];
   the slot 61 has no constructor fact, as no projection names it. *)
Module FlowCallExample.

Local Open Scope positive_scope.

Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition code20 : code :=
  Code [30] 31 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 32 Value (Project_value_slot_from 40 60 (Var 31))
       (return_to 3 [Var 32])).

Definition call : apply_expr :=
  {| ap_callee := Var 10; ap_call_kind := Direct 20;
     ap_args := [Const (Tagged_immediate 7)];
     ap_continuation := 5; ap_exn_continuation := 2; ap_exn_extra_args := [];
     ap_args_arity := [Singleton Value];
     ap_return_arity := [Singleton Value] |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 20 code20]
    (Let 10 Value
       (Closure_of_set 40 [(40, 20)]
          [(60, Const (Tagged_immediate 5)); (61, Const (Tagged_immediate 6))])
       (Let_cont_non_recursive 5
          (Handler [12] (make_handler_metadata [Value]) (return_to 1 [Var 12]))
          (Apply call))).

Definition is_local (f : field) : bool := false.

Definition x10 : node := Node_variable 10.
Definition x12 : node := Node_variable 12.
Definition x30 : node := Node_variable 30.
Definition x31 : node := Node_variable 31.
Definition x32 : node := Node_variable 32.
Definition code20_node : node := Node_code_id 20.
Definition constants : node := Node_all_constants.
Definition m (n : positive) : node := Node_made n.

Definition solution : tables :=
  {| t_alias :=
       [(m 1, x12); (m 10, code20_node); (m 11, m 7); (m 2, m 4);
        (m 3, x32); (m 5, x10); (m 6, constants); (m 9, m 11); (x12, m 3);
        (x30, m 6); (x31, m 5); (x32, constants)];
     t_any_usage :=
       [constants; code20_node; m 1; m 10; m 2; m 3; m 4; x12; x32];
     t_any_source :=
       [constants; code20_node; m 1; m 10; m 3; m 6; x12; x30; x32];
     t_has_usage :=
       [constants; code20_node; m 1; m 10; m 11; m 2; m 3; m 4; m 5; m 7;
        m 9; x10; x12; x31; x32];
     t_has_source :=
       [constants; code20_node; m 1; m 10; m 11; m 3; m 5; m 6; m 7; m 8;
        m 9; x10; x12; x30; x31; x32];
     t_usages :=
       [(m 11, m 9); (m 5, x31); (m 7, m 9); (m 9, m 9); (x10, x10);
        (x10, x31); (x31, x31)];
     t_sources :=
       [(m 11, m 7); (m 5, x10); (m 7, m 7); (m 8, m 8); (m 9, m 7);
        (x10, x10); (x31, x10)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(m 7, Field_code_id_of_call_witness);
        (m 7, Field_return_of_call (Normal 0));
        (m 7, Field_return_of_call Exn);
        (x10, Field_call_witness Known_arity_code_pointer);
        (x10, Field_value_slot 60)] |}.

End FlowCallExample.

(* The call example meets the premises of [flow_sound_statement], with
   its least solution, and its run exits with 5. *)
Definition call_example_statement : Prop :=
  is_local_slots_only FlowCallExample.is_local
  ∧ (∀ f : field, FlowCallExample.is_local f = false)
  ∧ in_flow_fragment FlowCallExample.unit_expr = true
  ∧ check_closed FlowCallExample.is_local
      (traverse_unit Never FlowCallExample.interface FlowCallExample.unit_expr)
      FlowCallExample.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables FlowCallExample.solution ->
       check_closed FlowCallExample.is_local
         (traverse_unit Never FlowCallExample.interface
            FlowCallExample.unit_expr) T
       = false)
  ∧ run 7 (State FlowCallExample.unit_expr [] [] empty_static_env empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 5)] empty_heap.

(* Why local fields are kept out where a closure is projected from. The
   value slot 60 is local. The closure [x10] goes into an immutable array
   [x11] and is loaded back out of it into [x12], whose node has any
   source and a use of [x11], but is not an alias of [x10]; the slot is
   read into [x13], which is passed to the continuation 5, whose
   parameter is not used:

     let code 20 (x30) my_closure x31 = return x30 in
     let x10 = closure 20 { 60 = 5 } in
     let x11 = [| x10 |] in
     let x12 = x11.(0) in
     let x13 = x12.60 in
     let_cont k5 x14 = return 0 in
     k5 x13

   Nothing reads [x13] in a way that is used, so [x10] has no usage, and
   the slot 60 does not escape; [x12] has any source, so the read of a
   local field gives [x13] only [reading_field]. In the least solution
   [x13] has no source, and the rebuild (FlambdaReaperRebuild.v, which
   transcribes rebuild.ml) replaces [k5 x13] by [Invalid]: the unit
   exits, and what the reaper makes of it goes wrong. This is a case
   where the reaper, with [-reaper-local-fields], is unsound. *)
Module LocalSlotCounterexample.

Local Open Scope positive_scope.

Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

Definition code20 : code :=
  Code [30] 31 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 30]).

Definition unit_expr : expr :=
  Let_static [Bind_code 20 code20]
    (Let 10 Value (Closure_of_set 40 [(40, 20)] [(60, Const (Tagged_immediate
      5))])
       (Let 11 Value (Prim (Variadic (Make_immutable_array Array_values))
                        [Var 10])
          (Let 12 Value
             (Prim (Binary (Array_load Array_values Load_values Immutable))
                [Var 11; Const (Tagged_immediate 0)])
             (Let 13 Value (Project_value_slot_from 40 60 (Var 12))
                (Let_cont_non_recursive 5
                   (Handler [14] (make_handler_metadata [Value])
                      (return_to 1 [Const (Tagged_immediate 0)]))
                   (return_to 5 [Var 13])))))).

Definition is_local (f : field) : bool := field_eqb f (Field_value_slot 60).

Definition v (n : positive) : node := Node_variable n.
Definition m (n : positive) : node := Node_made n.

Definition solution : tables :=
  {| t_alias :=
       [(v 14, v 13); (m 1, Node_all_constants); (m 3, v 30); (v 31, m 5);
        (v 30, m 6)];
     t_any_usage := [m 1; m 2; Node_all_constants];
     t_any_source := [v 12; v 11; Node_code_id 20; Node_all_constants; m 1];
     t_has_usage := [m 1; m 2; Node_all_constants];
     t_has_source :=
       [v 12; v 11; Node_code_id 20; Node_all_constants; m 1; v 10; m 7;
        m 8];
     t_usages := [(v 12, v 12)];
     t_sources := [(v 10, v 10); (m 7, m 7); (m 8, m 8)];
     t_reading_field := [(Field_value_slot 60, v 13)];
     t_escaping_field := [];
     t_field_used := [] |}.

End LocalSlotCounterexample.

(* The unit is of the fragment, its slot 60 is local, its least solution
   passes the check, the unit exits, and the rebuilt unit goes wrong. *)
Definition local_slot_counterexample_statement : Prop :=
  is_local_slots_only LocalSlotCounterexample.is_local
  ∧ in_flow_fragment LocalSlotCounterexample.unit_expr = true
  ∧ check_closed LocalSlotCounterexample.is_local
      (traverse_unit Never LocalSlotCounterexample.interface
         LocalSlotCounterexample.unit_expr)
      LocalSlotCounterexample.solution
    = true
  ∧ (∀ T : tables,
       T ∈ one_fewer_tables LocalSlotCounterexample.solution ->
       check_closed LocalSlotCounterexample.is_local
         (traverse_unit Never LocalSlotCounterexample.interface
            LocalSlotCounterexample.unit_expr) T
       = false)
  ∧ run 10
      (State LocalSlotCounterexample.unit_expr [] [] empty_static_env
         empty_heap)
    = Exit 1%positive [Value_const (Tagged_immediate 0)] empty_heap
  ∧ run 10
      (State
         (rebuild_unit_from_tables
            (traverse_unit Never LocalSlotCounterexample.interface
               LocalSlotCounterexample.unit_expr)
            LocalSlotCounterexample.solution false
            LocalSlotCounterexample.unit_expr)
         [] [] empty_static_env empty_heap)
    = Undefined.
