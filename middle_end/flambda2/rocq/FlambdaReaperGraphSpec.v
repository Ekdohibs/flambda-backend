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

(* The flow graph of the reaper, and what its solution is.

   This file is to be reviewed, against middle_end/flambda2/reaper/
   (global_flow_graph.ml, points_to_analysis.ml) and
   middle_end/flambda2/identifiers/field.mli. It holds definitions only.

   The reaper is a whole-unit pass (reaper/reaper.ml). Its first part,
   the traversal (reaper/traverse.ml), reads the unit and builds a graph:
   relations between names, saying that the value of one name may be
   that of another ([alias]), a field of another ([accessor]), that the
   value of a name holds another in a field ([constructor]), and so on.
   Its second part solves Datalog rules over the graph
   (points_to_analysis.ml): which names have a use, which have a
   source, the uses of each name ([usages]), the allocations each may
   come from ([sources]). Its third part rebuilds the unit from the
   solution (rebuild.ml). This file is about the first two: the graph,
   and what it is for a solution to be closed under the rules. What the
   traversal must put into the graph for a given term is in
   FlambdaReaperTraversalSpec.v.

   The nodes. The compiler's nodes are [Code_id_or_name.t]: code ids,
   variables and symbols. The traversal also makes variables of its own,
   which no term binds (the return and exception variables of each
   code, the variables of each call, the variable standing for the
   result of the toplevel: [Variable.create] in traverse.ml and
   traverse_acc.ml). Here they are [Node_made n], for a name [n] chosen
   by whoever builds the graph. Constants are one node,
   [Node_all_constants], the symbol [all_constants] that traverse.ml
   makes for the unit (traverse.ml, [run]).

   The fields. [field] is [Field.view]: the real fields, which a
   primitive reads at run time (the fields of a block, value and
   function slots, [Is_int], [Get_tag], the contents of a boxed number),
   and the virtual ones, which link a closure to the code it calls and a
   call to what it returns. [Cofield.t] is a parameter number.

   Which fields are local ([Analysis_scope.is_local_field]) depends on
   the flag [-reaper-local-fields] and on the compilation unit of a
   slot. It is a parameter of the rules, [is_local], with what the
   compiler guarantees of it: only value and function slots are ever
   local ([is_local_slots_only]).

   The solution. The Datalog engine computes tables; here they are
   functions to [bool], one per relation of points_to_analysis.ml that
   the rebuild or the rules read. That a solution is closed under the
   rules ([closed]) is all that the statements ask of it: the least
   solution, which the engine computes, is closed, and so is any larger
   one. A rule whose conclusion is guarded by [nontop_usages] or
   [nontop_sources] ([let$$] in points_to_analysis.ml) has the guard as
   a premise: at the fixpoint, if the premises and the guard hold, the
   conclusion was derived.

   [field_of_constructor_is_used] is computed by the post-processing
   rules, after the fixpoint, and only by them
   ([field_used_exact]): the rebuild keeps a field because one of these
   rules says it is used, so what the proofs need of a kept field is
   that one of them applied.

   Not here: the rules of the unboxing analysis
   (unboxing_analysis.ml), the zero-alloc check, and the relations
   [allocation_point_dominator], [multiple_allocation_points],
   [field_of_constructor_is_used_top] and
   [field_of_constructor_is_used_as], which only the unboxing and the
   type rewriting read. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel.
Import ListNotations.

(* ======================================================================== *)
(* Nodes and fields                                                         *)
(* ======================================================================== *)

(* [Code_id_or_name.t], and the variables that the traversal makes. *)
Inductive node : Type :=
  | Node_variable (x : variable)
  | Node_symbol (s : symbol)
  | Node_code_id (c : code_id)
  | Node_all_constants
  | Node_made (n : positive).

Definition node_eq_dec : ∀ n n' : node, {n = n'} + {n <> n'}.
Proof. repeat decide equality. Defined.

Definition node_of_simple (s : simple) : node :=
  match s with
  | Var x => Node_variable x
  | Symbol sym => Node_symbol sym
  | Const _ => Node_all_constants
  end.

(* [Field.closure_entry_point]. *)
Inductive closure_entry_point : Type :=
  | Unknown_arity_code_pointer
  | Known_arity_code_pointer.

(* [Field.return_kind]. *)
Inductive return_kind : Type :=
  | Normal (i : nat)
  | Exn.

(* [Field.view]. The kind of a field of a block is the compiler's, which
   the model reads off [Block_load] and [Make_block]. *)
Inductive field : Type :=
  | Field_block (i : nat) (kind : flambda_kind)
  | Field_value_slot (w : value_slot)
  | Field_function_slot (f : function_slot)
  | Field_call_witness (entry : closure_entry_point)
  | Field_is_int
  | Field_get_tag
  | Field_boxed_number (kind : boxable_number)
  | Field_return_of_call (r : return_kind)
  | Field_code_id_of_call_witness.

Definition field_eq_dec : ∀ f f' : field, {f = f'} + {f <> f'}.
Proof. repeat decide equality. Defined.

(* [Field.is_real_field]: the fields that a primitive reads. *)
Definition is_real_field (f : field) : bool :=
  match f with
  | Field_block _ _ | Field_value_slot _ | Field_function_slot _
  | Field_is_int | Field_get_tag | Field_boxed_number _ => true
  | Field_call_witness _ | Field_return_of_call _
  | Field_code_id_of_call_witness => false
  end.

(* [Field.is_function_slot]. *)
Definition is_function_slot (f : field) : bool :=
  match f with
  | Field_function_slot _ => true
  | _ => false
  end.

(* What the compiler guarantees of [Analysis_scope.is_local_field]: a
   field that is not a value slot or a function slot is never local. *)
Definition is_local_slots_only (is_local : field -> bool) : Prop :=
  ∀ f : field,
    is_local f = true ->
    (∃ w : value_slot, f = Field_value_slot w)
    ∨ (∃ fs : function_slot, f = Field_function_slot fs).

(* ======================================================================== *)
(* The graph                                                                *)
(* ======================================================================== *)

(* [Global_flow_graph.graph]. The relations are those that the
   traversal adds to; a graph may hold more facts than the traversal of a
   unit gives, and the statements hold of any such graph. The order of
   the arguments of each relation is that of the Datalog atom
   ([Global_flow_graph.Relations]). *)
Record graph : Type := Graph {
  (* [alias ~to_ ~from]: [to_] may be [from]. *)
  g_alias : node -> node -> Prop;
  (* [use ~to_ ~from]: [to_] is computed from [from], which is used in a
     way that is not tracked. *)
  g_use : node -> node -> Prop;
  (* [accessor ~to_ relation ~base]: [to_] is a field of [base]. *)
  g_accessor : node -> field -> node -> Prop;
  (* [constructor ~base relation ~from]: [base] holds [from] in a field. *)
  g_constructor : node -> field -> node -> Prop;
  (* [argument ~from relation ~base]: [from] is given as the parameter
     number [relation] to what [base] calls. *)
  g_argument : node -> nat -> node -> Prop;
  (* [parameter ~base relation ~to_]: [to_] is the parameter number
     [relation] of the code that [base] stands for. *)
  g_parameter : node -> nat -> node -> Prop;
  (* [propagate ~if_used ~to_ ~from]: an alias if [if_used] has any
     usage. *)
  g_propagate : node -> node -> node -> Prop;
  (* [alias_if_any_source ~if_any_source ~to_ ~from]: an alias if
     [if_any_source] has any source. *)
  g_alias_if_any_source : node -> node -> node -> Prop;
  g_any_usage : node -> Prop;
  g_any_source : node -> Prop;
  (* [any_source_if_used ~if_used x]. *)
  g_any_source_if_used : node -> node -> Prop;
  g_zero_alloc_source : node -> Prop
}.

(* ======================================================================== *)
(* The solution                                                             *)
(* ======================================================================== *)

(* The tables of points_to_analysis.ml that the rules or the rebuild
   read. [s_alias] is the relation [alias] after the rules, which add to
   the input. *)
Record solution : Type := Solution {
  s_alias : node -> node -> bool;
  s_any_usage : node -> bool;
  s_any_source : node -> bool;
  s_has_usage : node -> bool;
  s_has_source : node -> bool;
  (* [usages x y]: [y] is a use of [x]. *)
  s_usages : node -> node -> bool;
  (* [sources x y]: [y] is an allocation that [x] may come from. *)
  s_sources : node -> node -> bool;
  (* [reading_field relation to_], [escaping_field relation from]. *)
  s_reading_field : field -> node -> bool;
  s_escaping_field : field -> node -> bool;
  (* [field_of_constructor_is_used base relation]. *)
  s_field_used : node -> field -> bool
}.

(* ======================================================================== *)
(* The rules                                                                *)
(* ======================================================================== *)

Section Rules.

Variable is_local : field -> bool.
Variable G : graph.
Variable S : solution.

Notation alias := (s_alias S).
Notation any_usage := (s_any_usage S).
Notation any_source := (s_any_source S).
Notation has_usage := (s_has_usage S).
Notation has_source := (s_has_source S).
Notation usages := (s_usages S).
Notation sources := (s_sources S).

(* The rules of points_to_analysis.ml, each a field named after the
   group of rules it is in ([alias_rules], [has_usage_rules], ...) and
   its place in the group. Each says that the solution has the
   conclusion of the rule wherever it has the premises. *)
Record closed : Prop := {
  (* The input [alias] is part of the relation. *)
  alias_input : ∀ to_ from, g_alias G to_ from -> alias to_ from = true;
  (* alias_rules *)
  alias_propagate : ∀ if_used to_ from,
    any_usage if_used = true -> g_propagate G if_used to_ from ->
    alias to_ from = true;
  alias_if_any_source : ∀ if_any_source to_ from,
    any_source if_any_source = true ->
    g_alias_if_any_source G if_any_source to_ from ->
    alias to_ from = true;
  alias_constructor_usage : ∀ base base_use relation to_ from,
    g_constructor G base relation from ->
    usages base base_use = true -> any_usage base = false ->
    g_accessor G to_ relation base_use ->
    alias to_ from = true;
  alias_local_constructor_usage : ∀ base base_use relation to_ from,
    is_local relation = true ->
    g_constructor G base relation from ->
    usages base base_use = true ->
    g_accessor G to_ relation base_use ->
    alias to_ from = true;
  alias_parameter_usage : ∀ base base_use relation to_ from,
    g_parameter G base relation to_ ->
    usages base base_use = true -> any_usage base = false ->
    g_argument G from relation base_use ->
    alias to_ from = true;
  alias_accessor_source : ∀ base base_source relation to_ from,
    g_accessor G to_ relation base ->
    sources base base_source = true -> any_source base = false ->
    g_constructor G base_source relation from ->
    alias to_ from = true;
  alias_local_accessor_source : ∀ base base_source relation to_ from,
    is_local relation = true ->
    g_accessor G to_ relation base ->
    sources base base_source = true ->
    g_constructor G base_source relation from ->
    alias to_ from = true;
  alias_escaping_reading : ∀ relation to_ from,
    s_escaping_field S relation from = true ->
    s_reading_field S relation to_ = true ->
    alias to_ from = true;
  alias_argument_source : ∀ base base_source relation to_ from,
    g_argument G from relation base ->
    sources base base_source = true -> any_source base = false ->
    g_parameter G base_source relation to_ ->
    alias to_ from = true;
  (* has_usage_rules *)
  has_usage_any : ∀ x, any_usage x = true -> has_usage x = true;
  has_usage_alias : ∀ to_ from,
    has_usage to_ = true -> alias to_ from = true -> has_usage from = true;
  has_usage_accessor : ∀ to_ relation base,
    has_usage to_ = true -> g_accessor G to_ relation base ->
    has_usage base = true;
  has_usage_argument : ∀ from relation base,
    has_source from = true -> g_argument G from relation base ->
    has_usage base = true;
  (* has_source_rules *)
  has_source_any : ∀ x, any_source x = true -> has_source x = true;
  has_source_alias : ∀ from to_,
    has_source from = true -> alias to_ from = true -> has_source to_ = true;
  has_source_constructor : ∀ from relation base,
    has_source from = true -> g_constructor G base relation from ->
    has_source base = true;
  has_source_function_slot : ∀ from relation base,
    g_constructor G base relation from -> is_function_slot relation = true ->
    has_source base = true;
  has_source_parameter : ∀ to_ relation base,
    has_usage to_ = true -> g_parameter G base relation to_ ->
    has_source base = true;
  (* any_usage_rules *)
  any_usage_use : ∀ to_ from,
    has_usage to_ = true -> g_use G to_ from -> any_usage from = true;
  any_usage_alias : ∀ to_ from,
    any_usage to_ = true -> alias to_ from = true -> any_usage from = true;
  any_usage_constructor : ∀ base relation from,
    any_usage base = true -> g_constructor G base relation from ->
    is_local relation = false -> any_usage from = true;
  any_usage_argument : ∀ base relation from,
    any_source base = true -> g_argument G from relation base ->
    any_usage from = true;
  (* any_source_rules *)
  any_source_input : ∀ x, g_any_source G x -> any_source x = true;
  any_source_zero_alloc : ∀ x,
    g_zero_alloc_source G x -> any_source x = true;
  any_source_if_used : ∀ if_used x,
    any_usage if_used = true -> g_any_source_if_used G if_used x ->
    any_source x = true;
  any_source_alias : ∀ from to_,
    alias to_ from = true -> any_source from = true ->
    any_source to_ = true;
  any_source_parameter : ∀ base relation to_,
    any_usage base = true -> g_parameter G base relation to_ ->
    any_source to_ = true;
  any_source_accessor : ∀ base relation to_,
    any_source base = true -> g_accessor G to_ relation base ->
    is_local relation = false -> any_source to_ = true;
  any_source_use : ∀ from to_,
    has_source from = true -> g_use G to_ from -> any_source to_ = true;
  (* The input [any_usage] is part of the relation. *)
  any_usage_input : ∀ x, g_any_usage G x -> any_usage x = true;
  (* usages_rules, with the guard of [nontop_usages] *)
  usages_accessor : ∀ to_ relation base,
    g_accessor G to_ relation base -> any_usage base = false ->
    usages base base = true;
  usages_argument : ∀ from relation base,
    g_argument G from relation base -> any_usage base = false ->
    usages base base = true;
  usages_alias : ∀ to_ from usage,
    usages to_ usage = true -> any_usage to_ = false ->
    alias to_ from = true -> any_usage from = false ->
    usages from usage = true;
  (* sources_rules, with the guard of [nontop_sources] *)
  sources_constructor : ∀ from relation base,
    g_constructor G base relation from -> any_source base = false ->
    sources base base = true;
  sources_parameter : ∀ to_ relation base,
    g_parameter G base relation to_ -> any_source base = false ->
    sources base base = true;
  sources_alias : ∀ from to_ source,
    sources from source = true -> any_source from = false ->
    alias to_ from = true -> any_source to_ = false ->
    sources to_ source = true;
  (* local_rules *)
  escaping_field_rule : ∀ base relation from,
    any_usage base = true -> g_constructor G base relation from ->
    is_local relation = true -> s_escaping_field S relation from = true;
  reading_field_rule : ∀ base relation to_,
    any_source base = true -> g_accessor G to_ relation base ->
    is_local relation = true -> s_reading_field S relation to_ = true;
  (* post_processing_rules: [field_of_constructor_is_used] *)
  field_used_any_usage : ∀ base relation from,
    g_constructor G base relation from -> any_usage base = true ->
    is_local relation = false -> s_field_used S base relation = true;
  field_used_usage : ∀ base relation from usage v,
    g_constructor G base relation from -> usages base usage = true ->
    g_accessor G v relation usage -> has_usage v = true ->
    s_field_used S base relation = true;
  field_used_reading : ∀ base relation from x,
    g_constructor G base relation from -> any_usage base = true ->
    s_reading_field S relation x = true -> has_usage x = true ->
    s_field_used S base relation = true;
  field_used_local_source : ∀ usage base relation from v,
    g_constructor G base relation from -> sources usage base = true ->
    is_local relation = true -> any_usage base = true ->
    g_accessor G v relation usage -> has_usage v = true ->
    s_field_used S base relation = true
}.

(* [field_of_constructor_is_used] holds only where a rule of the
   post-processing gives it. The rules whose conclusion is
   [any_usage v] rather than [has_usage v] are subsumed: [any_usage]
   implies [has_usage]. *)
Definition field_used_exact : Prop :=
  ∀ (base : node) (relation : field),
    s_field_used S base relation = true ->
    ∃ from : node,
      g_constructor G base relation from
      ∧ ((any_usage base = true ∧ is_local relation = false)
         ∨ (∃ usage v : node,
              usages base usage = true ∧ g_accessor G v relation usage
              ∧ has_usage v = true)
         ∨ (∃ x : node,
              any_usage base = true ∧ s_reading_field S relation x = true
              ∧ has_usage x = true)
         ∨ (∃ usage v : node,
              sources usage base = true ∧ is_local relation = true
              ∧ any_usage base = true ∧ g_accessor G v relation usage
              ∧ has_usage v = true)).

End Rules.
