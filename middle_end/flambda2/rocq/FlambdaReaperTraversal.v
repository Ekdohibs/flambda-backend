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

(* The traversal of the reaper, transcribed: the facts of the flow graph
   (FlambdaReaperGraphSpec.facts) that the compiler's reaper builds for a
   unit.

   This file transcribes middle_end/flambda2/reaper/traverse.ml (the
   functions named below), the parts of traverse_acc.ml that add to the
   graph, traverse_env.ml, and the two places where the graph is
   completed before the rules are solved: analysis.ml ([solve], which
   gives the toplevel return [any_usage]) and traverse_acc.ml
   ([resolve_delayed_deps]). It needs no review: what the facts mean is
   FlambdaReaperGraphSpec.v, and the test tool compares them with the
   graph the compiler builds. It is extracted ([traverse_unit]).

   The shape follows the OCaml: [traverse] is [Traverse.traverse], and
   each construct has the function of the same name. The accumulator
   ([acc], [Traverse_acc.t]) is threaded where the OCaml mutates it. It
   holds the facts as a list ([fact], one constructor to a relation of
   [Global_flow_graph.graph]), the code deps ([Code_dep.t], made once per
   code id by [prepare_code] and looked up after), the delayed deps
   ([Delayed_deps.t]: calls of known code, closures of sets, and the
   symbols that went through [simple_to_node]) and a counter, from which
   the variables that the OCaml makes with [Variable.create] are made, as
   [Node_made n]. Which number a made node has is of no meaning; nor is
   the order of the facts, nor whether one is listed twice: the graph is
   [graph_of_facts], which asks only whether a fact is in a list.

   The environment ([env], [Traverse_env.t]) has the parameters of the
   continuations in scope, the code id of the code being traversed
   ([current_code_id]: [None] at toplevel), the mode of
   [should_preserve_direct_calls], and the value slots to keep. The facts
   that depend on the code id are those of [Acc.add_cond_any_usage],
   [add_cond_any_source] and [cond_alias] ([cond_any_usage],
   [cond_any_source], [cond_alias] here): unconditional at toplevel, and
   inside code conditional on the code id having a use ([use],
   [any_source_if_used], [propagate]).

   The primitives. [traverse_prim] goes by the classification of
   [Traverse.traverse_prim], and so looks inside the primitives of
   FlambdaPrimitiveModel.v, as the OCaml looks inside
   [Flambda_primitive.t]: a block made ([Make_block], and
   [Make_block_with_identity], which is the compiler's [Make_block] at
   another mutability) has constructor facts, a load from one, [Is_int]
   of a variant, [Get_tag] and the unboxing of a number have accessor
   facts, the boxing of a number a constructor fact; every other
   primitive is the default case, with use facts from the free names of
   its arguments, [any_source], and for a primitive with arbitrary
   effects ([prim_has_arbitrary_effects], from
   terms/flambda_primitive.ml, [effects_and_coeffects]) the conditional
   [any_usage] of the variable bound.

   The deliberate differences from the OCaml, each where the model
   cannot follow it or does not have what it reads:
   - Closures of a set. The model's [Let] binds one variable, to the
     closure at [slot] of a [Closure_of_set]; the compiler's binds one to
     each closure of the set. The other closures of the set have a made
     node each, made where the [Let] is traversed, which stands for the
     variable the compiler binds to it; the facts of
     [record_set_of_closures_deps] are given to these nodes as to that
     variable. In the model the other closures are then bound by
     [Project_function_slot] from the first (FlambdaModel.let_closures),
     which gives accessor facts where the compiler binds them directly.
   - Static sets. A static set of closures is one [Bind_symbol] for each
     of its symbols (FlambdaModel.bind_closures). Each gives the facts of
     [record_set_of_closures_deps] whose base is its own symbol (the
     value slots, the function slots, the closure dep with the code of
     its slot). For a set bound whole these are those of the compiler,
     which goes through the set once.
   - The older forms [Set_of_closures] and
     [Static_const_set_of_closures], a set of one function whose slot
     has no name, have the facts of a set of one closure but the
     constructor fact of its function slot, which has no name; the
     projection [Project_value_slot] is read as
     [Project_value_slot_from].
   - Code ids, symbols and value slots have no compilation unit in the
     model. A code id that no [Let_static] of the unit binds (code or
     deleted code) is taken for one of another unit: it is the
     [is_external] of [traverse_call_kind]. A symbol that no
     [Let_static] of the unit binds is taken for an imported one: if it
     went through [simple_to_node] it is given [any_source], as
     [resolve_delayed_deps] gives it to the imported symbols outside the
     analysis scope ([Current_unit] here, as in reaper.ml). (A symbol or
     a code id that is bound only in the body of code, which a
     well-formed unit does not have, is taken for one of another unit
     too.) Every value slot
     is of the unit, so [should_keep_value_slot] keeps exactly those that
     a projection of the unit names ([value_slots_to_keep],
     [Name_occurrences.value_slots_in_normal_projections] of the free
     names of the unit, the bodies of code included); every projection
     of the model is normal. [should_keep_function_slot] keeps every
     function slot, as in the OCaml.
   - No zero-alloc attribute and no [[@opaque]]: [prepare_code] never
     takes [never_delete], so there is no [zero_alloc_source] fact,
     and the body of code is traversed as in the non-opaque case of
     [traverse_function_params_and_body]. The mode of
     [should_preserve_direct_calls] is the flag
     ([reaper_preserve_direct_calls], a parameter of [traverse_unit]),
     read as in [run0] and [traverse_function_params_and_body]:
     [Zero_alloc] is [No] everywhere, as no code is checked.
   - The relation [code_id_my_closure] ([add_code_id_my_closure]) is
     not among the facts of FlambdaReaperGraphSpec.v and is not made.
     Nor is anything that only the rebuild reads ([continuation_info],
     [fixed_arity_continuation], [record_apply_for_rebuild],
     [add_set_of_closures], [add_code]).
   - Calls. The model has [Direct] and [Indirect]. A direct call is read
     as [Direct], whose callee is always given (the model has no direct
     call without a callee). An indirect call is read as
     [Indirect_unknown_arity], with [make_unknown_arity_apply_widget];
     the compiler's [Indirect_known_arity] is not in the model (the
     status of S7 says so). The model has no [Never_returns], method, C
     call or effect, so the [default_acc] of [traverse_apply] is never
     used.
   - Where the OCaml stops with a fatal error, the model does something
     defined, which no term that the compiler builds reaches:
     a continuation not in scope ([Env.find_cont]) has fresh parameters
     with [any_usage], as the toplevel's are, since the machine exits
     through it with the values; an exception continuation with no
     parameter is given a fresh node for the exception; lists that
     [List.iter2] would refuse for their lengths are cut to the shorter
     ([combine]); [group_by_parameter] cuts or drops arguments that do
     not fit the arity; code with no parameter, and a call of unknown
     arity with no argument, have one group of none, so one witness
     ([create_unknown_arity_non_tupled_call_witnesses],
     [make_unknown_arity_apply_widget]); a call of code that the unit
     does not have gets the facts of external code from
     [add_apply_dep] even where that code is deleted code of the unit
     (an assertion fails in the compiler); a [Block_load] at a negative
     field is a load of field 0 ([Z.to_nat]); a primitive applied to
     another number of arguments than it takes is the default case; a
     closure of a set at a slot that the set does not have has none of
     the facts of the set; a static closure whose slot has no code has
     no closure dep.
   - The maps of the compiler are lists here: the value slots and the
     function declarations of a set, the closure symbols of a static
     set and the arms of a [Switch] are gone through by their bindings
     that count ([compact]), as the compiler goes through a map. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec.
Import ListNotations.

(* ======================================================================== *)
(* Facts and the accumulator                                                *)
(* ======================================================================== *)

(* One fact of [Global_flow_graph.graph], with the arguments in the order
   of the Datalog atom, as in [facts]. *)
Inductive fact : Type :=
  | Fact_alias (to_ from : node)
  | Fact_use (to_ from : node)
  | Fact_accessor (to_ : node) (f : field) (base : node)
  | Fact_constructor (base : node) (f : field) (from : node)
  | Fact_argument (from : node) (i : nat) (base : node)
  | Fact_parameter (base : node) (i : nat) (to_ : node)
  | Fact_propagate (if_used to_ from : node)
  | Fact_alias_if_any_source (if_any_source to_ from : node)
  | Fact_any_usage (x : node)
  | Fact_any_source (x : node)
  | Fact_any_source_if_used (if_used x : node).

(* The facts, sorted by relation. There is no [zero_alloc_source]. *)
Definition facts_of_list (l : list fact) : facts :=
  {| f_alias := flat_map
       (fun fct => match fct with
                   | Fact_alias to_ from => [(to_, from)]
                   | _ => []
                   end) l;
     f_use := flat_map
       (fun fct => match fct with
                   | Fact_use to_ from => [(to_, from)]
                   | _ => []
                   end) l;
     f_accessor := flat_map
       (fun fct => match fct with
                   | Fact_accessor to_ f base => [(to_, f, base)]
                   | _ => []
                   end) l;
     f_constructor := flat_map
       (fun fct => match fct with
                   | Fact_constructor base f from => [(base, f, from)]
                   | _ => []
                   end) l;
     f_argument := flat_map
       (fun fct => match fct with
                   | Fact_argument from i base => [(from, i, base)]
                   | _ => []
                   end) l;
     f_parameter := flat_map
       (fun fct => match fct with
                   | Fact_parameter base i to_ => [(base, i, to_)]
                   | _ => []
                   end) l;
     f_propagate := flat_map
       (fun fct => match fct with
                   | Fact_propagate if_used to_ from => [(if_used, to_, from)]
                   | _ => []
                   end) l;
     f_alias_if_any_source := flat_map
       (fun fct => match fct with
                   | Fact_alias_if_any_source if_any_source to_ from =>
                     [(if_any_source, to_, from)]
                   | _ => []
                   end) l;
     f_any_usage := flat_map
       (fun fct => match fct with
                   | Fact_any_usage x => [x]
                   | _ => []
                   end) l;
     f_any_source := flat_map
       (fun fct => match fct with
                   | Fact_any_source x => [x]
                   | _ => []
                   end) l;
     f_any_source_if_used := flat_map
       (fun fct => match fct with
                   | Fact_any_source_if_used if_used x => [(if_used, x)]
                   | _ => []
                   end) l;
     f_zero_alloc_source := [] |}.

(* [Traverse_acc.Code_dep.t], without the metadata: the made nodes that
   stand for the parameters, [my_closure], the results and the exception
   of code, and its call witnesses, one for a call of known arity and
   one for each parameter for a call of unknown arity. *)
Record code_dep : Type := Code_dep {
  cd_params : list node;
  cd_my_closure : node;
  cd_return : list node;
  cd_exn : node;
  cd_known_arity_call_witness : node;
  cd_unknown_arity_call_witnesses : list node
}.

(* [Delayed_deps.apply_dep]. The closure is always given: the model's
   calls always have a callee. *)
Record apply_dep : Type := Apply_dep {
  ad_function_containing_apply_expr : option code_id;
  ad_apply_code_id : code_id;
  ad_apply_closure : node;
  ad_apply_call_witness : node
}.

(* [Delayed_deps.closure_dep]. *)
Record closure_dep : Type := Closure_dep {
  cld_let_bound_name_of_the_closure : node;
  cld_closure_code_id : code_id
}.

(* [Traverse_acc.t], with what the graph needs of it, and the counter
   from which made nodes are made. [acc_symbols] are the symbols that
   went through [simple_to_node] ([imported_symbols] of [Delayed_deps.t]
   before it is known which are imported). *)
Record acc : Type := Acc {
  acc_next : positive;
  acc_facts : list fact;
  acc_code_deps : finite_map code_dep;
  acc_apply_deps : list apply_dep;
  acc_closure_deps : list closure_dep;
  acc_symbols : list symbol
}.

Definition empty_acc : acc :=
  {| acc_next := 1; acc_facts := []; acc_code_deps := [];
     acc_apply_deps := []; acc_closure_deps := []; acc_symbols := [] |}.

Definition add_facts (fcts : list fact) (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := fcts ++ acc_facts a;
     acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
     acc_closure_deps := acc_closure_deps a; acc_symbols := acc_symbols a |}.

Definition add_fact (fct : fact) (a : acc) : acc := add_facts [fct] a.

(* [Variable.create]: a node that nothing else is. *)
Definition fresh (a : acc) : node * acc :=
  (Node_made (acc_next a),
   {| acc_next := Pos.succ (acc_next a); acc_facts := acc_facts a;
      acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
      acc_closure_deps := acc_closure_deps a;
      acc_symbols := acc_symbols a |}).

Fixpoint fresh_list (n : nat) (a : acc) : list node * acc :=
  match n with
  | O => ([], a)
  | S n =>
    let '(x, a) := fresh a in
    let '(xs, a) := fresh_list n a in
    (x :: xs, a)
  end.

(* [Acc.add_code_dep]. *)
Definition add_code_dep (id : code_id) (cd : code_dep) (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := acc_facts a;
     acc_code_deps := (id, cd) :: acc_code_deps a;
     acc_apply_deps := acc_apply_deps a;
     acc_closure_deps := acc_closure_deps a; acc_symbols := acc_symbols a |}.

(* [Acc.add_apply]. *)
Definition push_apply_dep (d : apply_dep) (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := acc_facts a;
     acc_code_deps := acc_code_deps a; acc_apply_deps := d :: acc_apply_deps a;
     acc_closure_deps := acc_closure_deps a; acc_symbols := acc_symbols a |}.

(* [Acc.add_set_of_closures_dep], for the graph. *)
Definition push_closure_dep (d : closure_dep) (a : acc) : acc :=
  {| acc_next := acc_next a; acc_facts := acc_facts a;
     acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
     acc_closure_deps := d :: acc_closure_deps a;
     acc_symbols := acc_symbols a |}.

(* What [Acc.simple_to_node] does besides giving the node of a simple
   ([node_of_simple]): a symbol is recorded, to be given [any_source] at
   the end if it is imported. *)
Definition note_simple (s : simple) (a : acc) : acc :=
  match s with
  | Symbol sym =>
    {| acc_next := acc_next a; acc_facts := acc_facts a;
       acc_code_deps := acc_code_deps a; acc_apply_deps := acc_apply_deps a;
       acc_closure_deps := acc_closure_deps a;
       acc_symbols := sym :: acc_symbols a |}
  | Var _ | Const _ => a
  end.

Definition note_simples (l : list simple) (a : acc) : acc :=
  fold_left (fun a s => note_simple s a) l a.

(* The elements of a list with their positions, from 0 ([List.iteri]). *)
Definition indexed {A : Type} (l : list A) : list (nat * A) :=
  combine (seq 0 (length l)) l.

(* ======================================================================== *)
(* The environment                                                          *)
(* ======================================================================== *)

(* [Flambda_features.reaper_preserve_direct_calls]. *)
Inductive reaper_preserve_direct_calls : Type :=
  | Never
  | Always
  | Zero_alloc
  | Auto.

(* [Traverse_env.should_preserve_direct_calls]. *)
Inductive should_preserve_direct_calls : Type :=
  | Preserve_yes
  | Preserve_no
  | Preserve_auto.

(* The mode, from the flag, as [run0] and
   [traverse_function_params_and_body] read it. The latter gives
   [Zero_alloc] the mode [Yes] in code that the zero-alloc checker
   checks; the model has no such code. *)
Definition should_preserve_of_flag (flag : reaper_preserve_direct_calls)
    : should_preserve_direct_calls :=
  match flag with
  | Never | Zero_alloc => Preserve_no
  | Always => Preserve_yes
  | Auto => Preserve_auto
  end.

(* [Traverse_env.t], without what only the rebuild reads, and with the
   code ids that the unit binds, which stand for those of the current
   unit. *)
Record traverse_env : Type := Traverse_env {
  env_conts : finite_map (list node);
  env_current_code_id : option code_id;
  env_should_preserve_direct_calls : should_preserve_direct_calls;
  env_value_slots_to_keep : list value_slot;
  env_unit_code_ids : list code_id
}.

(* [Env.add_cont]. *)
Definition add_cont (env : traverse_env) (k : continuation) (params : list node)
    : traverse_env :=
  {| env_conts := (k, params) :: env_conts env;
     env_current_code_id := env_current_code_id env;
     env_should_preserve_direct_calls := env_should_preserve_direct_calls env;
     env_value_slots_to_keep := env_value_slots_to_keep env;
     env_unit_code_ids := env_unit_code_ids env |}.

(* [Traverse_env.should_keep_value_slot], for value slots of the unit. *)
Definition should_keep_value_slot (env : traverse_env) (w : value_slot)
    : bool :=
  among w (env_value_slots_to_keep env).

(* [Env.find_cont], with the number of parameters expected. A
   continuation not in scope (a fatal error in the OCaml) is a way out of
   the term for the machine: it gets fresh parameters with [any_usage]. *)
Definition find_cont (env : traverse_env) (k : continuation) (count : nat)
    (a : acc) : list node * acc :=
  match lookup k (env_conts env) with
  | Some params => (params, a)
  | None =>
    let '(params, a) := fresh_list count a in
    (params, add_facts (map Fact_any_usage params) a)
  end.

(* [Acc.add_cond_any_usage]. *)
Definition cond_any_usage (env : traverse_env) (s : simple) (a : acc) : acc :=
  let a := note_simple s a in
  match env_current_code_id env with
  | None => add_fact (Fact_any_usage (node_of_simple s)) a
  | Some id => add_fact (Fact_use (Node_code_id id) (node_of_simple s)) a
  end.

(* [Acc.cond_alias]. *)
Definition cond_alias (env : traverse_env) (from to_ : node) (a : acc) : acc :=
  match env_current_code_id env with
  | None => add_fact (Fact_alias to_ from) a
  | Some id => add_fact (Fact_propagate (Node_code_id id) to_ from) a
  end.

(* ======================================================================== *)
(* Arities                                                                  *)
(* ======================================================================== *)

(* The unarized arguments cut by the sizes of the parameters. *)
Fixpoint group_by_sizes {A : Type} (sizes : list nat) (l : list A)
    : list (list A) :=
  match sizes with
  | [] => []
  | n :: sizes => firstn n l :: group_by_sizes sizes (skipn n l)
  end.

(* [Flambda_arity.group_by_parameter]. Arguments that do not fit the
   arity (a fatal error in the OCaml) are cut or dropped. *)
Definition group_by_parameter {A : Type} (a : arity) (l : list A)
    : list (list A) :=
  group_by_sizes (map (@length flambda_kind) (unarize_per_parameter a)) l.

(* No group (no parameter, a fatal error in the OCaml) is one group of
   none. *)
Definition at_least_one_group {A : Type} (groups : list (list A))
    : list (list A) :=
  match groups with
  | [] => [[]]
  | _ :: _ => groups
  end.

(* ======================================================================== *)
(* Code                                                                     *)
(* ======================================================================== *)

(* [Acc.create_known_arity_call_witness]. *)
Definition create_known_arity_call_witness (id : code_id)
    (params returns : list node) (exn : node) (a : acc) : node * acc :=
  let '(witness, a) := fresh a in
  let a :=
    add_facts
      (map (fun '(i, p) => Fact_parameter witness i p) (indexed params)
       ++ map (fun '(i, r) =>
                 Fact_constructor witness (Field_return_of_call (Normal i)) r)
            (indexed returns)
       ++ [Fact_constructor witness (Field_return_of_call Exn) exn;
           Fact_constructor witness Field_code_id_of_call_witness
             (Node_code_id id)])
      a
  in
  (witness, a).

(* [Acc.create_unknown_arity_tupled_call_witnesses]: one witness, whose
   one parameter is the tuple, the parameters being read from its fields;
   [Is_int] of the tuple is read and used if the code is, so that the
   tuple, which the [caml_tuplify] functions read, stays alive. *)
Definition create_unknown_arity_tupled_call_witnesses (id : code_id)
    (params returns : list node) (exn : node) (a : acc) : list node * acc :=
  let '(witness, a) := fresh a in
  let '(untuple_var, a) := fresh a in
  let '(keep_tuple_alive_var, a) := fresh a in
  let a :=
    add_facts
      (map (fun '(i, r) =>
              Fact_constructor witness (Field_return_of_call (Normal i)) r)
         (indexed returns)
       ++ [Fact_constructor witness (Field_return_of_call Exn) exn;
           Fact_constructor witness Field_code_id_of_call_witness
             (Node_code_id id);
           Fact_parameter witness 0 untuple_var]
       ++ map (fun '(i, p) => Fact_accessor p (Field_block i Value) untuple_var)
            (indexed params)
       ++ [Fact_accessor keep_tuple_alive_var Field_is_int untuple_var;
           Fact_use (Node_code_id id) keep_tuple_alive_var])
      a
  in
  ([witness], a).

(* [add_deps] of [Acc.create_unknown_arity_non_tupled_call_witnesses]: a
   witness for each parameter, the last one returning the results, the
   others a partial application whose call witness is the next one. *)
Fixpoint unknown_arity_witness_deps (id : code_id) (returns : list node)
    (exn : node) (params_and_witnesses : list (list node * node)) (a : acc)
    : acc :=
  match params_and_witnesses with
  | [] => a
  | (first, witness) :: rest =>
    let a :=
      add_facts
        (map (fun '(i, p) => Fact_parameter witness i p) (indexed first)
         ++ [Fact_constructor witness Field_code_id_of_call_witness
               (Node_code_id id)])
        a
    in
    match rest with
    | [] =>
      add_facts
        (Fact_constructor witness (Field_return_of_call Exn) exn
         :: map (fun '(i, r) =>
                   Fact_constructor witness (Field_return_of_call (Normal i))
                     r)
              (indexed returns))
        a
    | (_, next_witness) :: _ =>
      let '(v, a) := fresh a in
      let a :=
        add_facts
          [Fact_constructor witness (Field_return_of_call (Normal 0)) v;
           Fact_constructor v (Field_call_witness Unknown_arity_code_pointer)
             next_witness]
          a
      in
      unknown_arity_witness_deps id returns exn rest a
    end
  end.

(* [Acc.create_unknown_arity_non_tupled_call_witnesses]. *)
Definition create_unknown_arity_non_tupled_call_witnesses (id : code_id)
    (params_arity : arity) (params returns : list node) (exn : node)
    (a : acc) : list node * acc :=
  let groups := at_least_one_group (group_by_parameter params_arity params) in
  let '(witnesses, a) := fresh_list (length groups) a in
  (witnesses,
   unknown_arity_witness_deps id returns exn (combine groups witnesses) a).

(* [Acc.create_unknown_arity_call_witnesses]. *)
Definition create_unknown_arity_call_witnesses (id : code_id)
    (is_tupled : bool) (params_arity : arity) (params returns : list node)
    (exn : node) (a : acc) : list node * acc :=
  if is_tupled
  then create_unknown_arity_tupled_call_witnesses id params returns exn a
  else
    create_unknown_arity_non_tupled_call_witnesses id params_arity params
      returns exn a.

(* [Traverse.prepare_code], for code that is not [never_delete] (the
   model has no zero-alloc attribute). *)
Definition prepare_code (id : code_id) (c : code) (a : acc) : acc :=
  let '(returns, a) := fresh_list (length (code_result_kinds c)) a in
  let '(exn, a) := fresh a in
  let '(my_closure, a) := fresh a in
  let '(params, a) := fresh_list (length (code_params_kinds c)) a in
  let '(known, a) :=
    create_known_arity_call_witness id params returns exn a in
  let '(unknown, a) :=
    create_unknown_arity_call_witnesses id (code_is_tupled c)
      (code_params_arity c) params returns exn a
  in
  let a := add_fact (Fact_any_source (Node_code_id id)) a in
  add_code_dep id
    {| cd_params := params; cd_my_closure := my_closure;
       cd_return := returns; cd_exn := exn;
       cd_known_arity_call_witness := known;
       cd_unknown_arity_call_witnesses := unknown |}
    a.

Definition prepare_binding (binding : static_binding) (a : acc) : acc :=
  match binding with
  | Bind_code id c => prepare_code id c a
  | Bind_deleted_code _ | Bind_symbol _ _ => a
  end.

(* The environment of the body of code
   ([traverse_function_params_and_body]): its two continuations only. *)
Definition code_env (env : traverse_env) (id : code_id)
    (return_continuation exn_continuation : continuation) (cd : code_dep)
    : traverse_env :=
  {| env_conts :=
       [(exn_continuation, [cd_exn cd]); (return_continuation, cd_return cd)];
     env_current_code_id := Some id;
     env_should_preserve_direct_calls := env_should_preserve_direct_calls env;
     env_value_slots_to_keep := env_value_slots_to_keep env;
     env_unit_code_ids := env_unit_code_ids env |}.

(* ======================================================================== *)
(* Sets of closures                                                         *)
(* ======================================================================== *)

(* [Traverse.record_set_of_closures_deps]: for each closure, given by its
   node and the code of its slot ([closures]), the closure dep, and the
   constructor facts of the value slots kept and of the function slots
   ([function_slots]: each slot with the node of its closure). *)
Definition record_set_of_closures_deps (env : traverse_env)
    (closures : list (node * option code_id))
    (function_slots : list (function_slot * node))
    (value_slots : list (value_slot * simple)) (a : acc) : acc :=
  let value_slots :=
    filter (fun '(w, _) => should_keep_value_slot env w) (compact value_slots)
  in
  fold_left
    (fun a '(base, code) =>
       let a :=
         match code with
         | Some id =>
           push_closure_dep
             {| cld_let_bound_name_of_the_closure := base;
                cld_closure_code_id := id |} a
         | None => a
         end
       in
       let a := note_simples (map snd value_slots) a in
       add_facts
         (map (fun '(w, s) =>
                 Fact_constructor base (Field_value_slot w) (node_of_simple s))
            value_slots
          ++ map (fun '(f, closure) =>
                    Fact_constructor base (Field_function_slot f) closure)
               function_slots)
         a)
    closures a.

(* The closures of a [Closure_of_set] bound to [x] at [slot]: [x] for that
   slot, a made node for each other. *)
Fixpoint closures_of_set (x : variable) (slot : function_slot)
    (function_decls : list (function_slot * code_id)) (a : acc)
    : list (function_slot * node * code_id) * acc :=
  match function_decls with
  | [] => ([], a)
  | (f, id) :: function_decls =>
    let '(name, a) :=
      if Pos.eqb f slot then (Node_variable x, a) else fresh a in
    let '(rest, a) := closures_of_set x slot function_decls a in
    ((f, name, id) :: rest, a)
  end.

(* [Traverse.traverse_set_of_closures]. *)
Definition traverse_closure_of_set (env : traverse_env) (x : variable)
    (slot : function_slot) (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (a : acc) : acc :=
  let function_decls := compact function_decls in
  if among slot (map fst function_decls)
  then
    let '(closures, a) := closures_of_set x slot function_decls a in
    record_set_of_closures_deps env
      (map (fun '(_, name, id) => (name, Some id)) closures)
      (map (fun '(f, name, _) => (f, name)) closures)
      value_slots a
  else a.

(* ======================================================================== *)
(* Primitives                                                               *)
(* ======================================================================== *)

(* [Flambda_primitive.effects_and_coeffects] is [Arbitrary_effects]. *)
Definition prim_has_arbitrary_effects (p : prim) : bool :=
  match p with
  | Unary u =>
    match u with
    (* [Opaque_identity]: arbitrary effects. *)
    | Opaque_identity _ => true
    (* [reading_from_a_block]: no effects. *)
    | Block_load _ _ _
    | Is_int _ | Is_null | Get_tag | Array_length _ | String_length
    | Unary_int_arith _ _ | Num_conv _ _ | Boolean_not
    | Unbox_number _ | Untag_immediate | Tag_immediate
    | Reinterpret_64_bit_word _ => false
    (* [Only_generative_effects]. *)
    | Box_number _ => false
    end
  | Binary b =>
    match b with
    (* [reading_from_an_array], [reading_from_a_string_or_bigstring]: no
       effects. *)
    | Array_load _ _ _ | String_or_bigstring_load _
    | Phys_equal _ | Int_arith _ _ | Int_shift _ _ | Int_comp _ _ => false
    end
  | Variadic v =>
    match v with
    (* [Only_generative_effects]. *)
    | Make_block _ | Make_immutable_array _ => false
    end
  | Heap h =>
    match h with
    (* [Make_block] and [Make_array]: [Only_generative_effects]. *)
    | Make_block_with_identity _ _ | Make_mutable_array _ => false
    (* [writing_to_a_block], [writing_to_an_array]: arbitrary effects. *)
    | Block_set _ _ _ | Array_set _ _ => true
    end
  end.

(* The free names of the arguments of a primitive
   ([Named.free_names]). *)
Definition free_names_of_simples (args : list simple) : list node :=
  flat_map
    (fun s =>
       match s with
       | Var x => [Node_variable x]
       | Symbol sym => [Node_symbol sym]
       | Const _ => []
       end)
    args.

(* The constructor facts of a block made, as [traverse_prim] gives them
   to [Make_block]: its fields, and [Is_int] and [Get_tag] from the
   constants. Every field of a block of the model is a value. *)
Definition block_deps (base : node) (fields : list simple) (a : acc) : acc :=
  let a := note_simples fields a in
  add_facts
    (map (fun '(i, s) =>
            Fact_constructor base (Field_block i Value) (node_of_simple s))
       (indexed fields)
     ++ [Fact_constructor base Field_is_int Node_all_constants;
         Fact_constructor base Field_get_tag Node_all_constants])
    a.

(* [Traverse.traverse_prim], for the [Let] of [x]. *)
Definition traverse_prim (env : traverse_env) (x : variable) (p : prim)
    (args : list simple) (a : acc) : acc :=
  let bound := Node_variable x in
  let accessor f s a :=
    add_fact (Fact_accessor bound f (node_of_simple s)) (note_simple s a) in
  match p, args with
  | Variadic (Make_block _), fields
  | Heap (Make_block_with_identity _ _), fields => block_deps bound fields a
  | Unary (Block_load _ mut field), [block] =>
    let a := accessor (Field_block (Z.to_nat field) Value) block a in
    match mut with
    | Mutable => add_fact (Fact_any_source bound) a
    | Immutable | Immutable_unique => a
    end
  | Unary (Is_int true), [arg] => accessor Field_is_int arg a
  | Unary Get_tag, [arg] => accessor Field_get_tag arg a
  | Unary (Box_number bn), [contents] =>
    add_fact (Fact_constructor bound (Field_boxed_number bn)
           (node_of_simple contents))
      (note_simple contents a)
  | Unary (Unbox_number bn), [arg] => accessor (Field_boxed_number bn) arg a
  | _, _ =>
    let a :=
      if prim_has_arbitrary_effects p then cond_any_usage env (Var x) a else a
    in
    add_facts
      (Fact_any_source bound
       :: map (fun from => Fact_use bound from) (free_names_of_simples args))
      a
  end.

(* [Traverse.traverse_let], for the defining expression. *)
Definition traverse_named (env : traverse_env) (x : variable)
    (defining_expr : named) (a : acc) : acc :=
  let bound := Node_variable x in
  let accessor f s a :=
    add_fact (Fact_accessor bound f (node_of_simple s)) (note_simple s a) in
  match defining_expr with
  | Simple s => add_fact (Fact_alias bound (node_of_simple s)) (note_simple s a)
  | Prim p args => traverse_prim env x p args a
  | Set_of_closures id value_slots =>
    record_set_of_closures_deps env [(bound, Some id)] [] value_slots a
  | Closure_of_set slot function_decls value_slots =>
    traverse_closure_of_set env x slot function_decls value_slots a
  | Project_value_slot w closure
  | Project_value_slot_from _ w closure =>
    accessor (Field_value_slot w) closure a
  | Project_function_slot _ move_to closure =>
    accessor (Field_function_slot move_to) closure a
  end.

(* ======================================================================== *)
(* Static constants                                                         *)
(* ======================================================================== *)

(* [traverse_static_set_of_closures] and
   [traverse_block_like_static_const], for one binding. *)
Definition traverse_static_binding (env : traverse_env)
    (binding : static_binding) (a : acc) : acc :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => a
  | Bind_symbol s const =>
    let name := Node_symbol s in
    match const with
    | Static_const_set_of_closures id value_slots =>
      record_set_of_closures_deps env [(name, Some id)] [] value_slots a
    | Static_const_closure_of_set slot closure_symbols function_decls
        value_slots =>
      record_set_of_closures_deps env [(name, lookup slot function_decls)]
        (map (fun '(f, sym) => (f, Node_symbol sym)) (compact closure_symbols))
        value_slots a
    | Static_const_block _ fields
    | Static_const_immutable_value_array fields => block_deps name fields a
    | Static_const_boxed_number _ | Static_const_immutable_string _ =>
      add_fact (Fact_alias name Node_all_constants) a
    end
  end.

(* ======================================================================== *)
(* Calls                                                                    *)
(* ======================================================================== *)

(* [Acc.make_known_arity_apply_widget]. *)
Definition make_known_arity_apply_widget (env : traverse_env) (ap : apply_expr)
    (returns : list node) (exn : node) (a : acc) : node * acc :=
  let '(witness, a) := fresh a in
  let '(called, a) := fresh a in
  let '(apply_node, a) := fresh a in
  let a := note_simples (ap_args ap) a in
  let a :=
    add_facts
      (map (fun '(i, s) => Fact_argument (node_of_simple s) i witness)
         (indexed (ap_args ap))
       ++ map (fun '(i, r) =>
                 Fact_accessor r (Field_return_of_call (Normal i)) witness)
            (indexed returns)
       ++ [Fact_accessor exn (Field_return_of_call Exn) witness;
           Fact_accessor called Field_code_id_of_call_witness witness;
           Fact_any_usage called])
      a
  in
  (apply_node, cond_alias env apply_node witness a).

(* [add_deps] of [Acc.make_unknown_arity_apply_widget]. *)
Fixpoint unknown_arity_apply_deps (returns : list node) (exn called : node)
    (args_and_witnesses : list (list simple * node)) (a : acc) : acc :=
  match args_and_witnesses with
  | [] => a
  | (first, witness) :: rest =>
    let a :=
      add_facts
        (map (fun '(i, s) => Fact_argument (node_of_simple s) i witness)
           (indexed first)
         ++ [Fact_accessor exn (Field_return_of_call Exn) witness;
             Fact_accessor called Field_code_id_of_call_witness witness])
        a
    in
    match rest with
    | [] =>
      add_facts
        (map (fun '(i, r) =>
                Fact_accessor r (Field_return_of_call (Normal i)) witness)
           (indexed returns))
        a
    | (_, next_witness) :: _ =>
      let '(v, a) := fresh a in
      let a :=
        add_facts
          [Fact_accessor v (Field_return_of_call (Normal 0)) witness;
           Fact_accessor next_witness
             (Field_call_witness Unknown_arity_code_pointer) v]
          a
      in
      unknown_arity_apply_deps returns exn called rest a
    end
  end.

(* [Acc.make_unknown_arity_apply_widget]. *)
Definition make_unknown_arity_apply_widget (env : traverse_env)
    (ap : apply_expr) (returns : list node) (exn : node) (a : acc)
    : node * acc :=
  let '(called, a) := fresh a in
  let a := add_fact (Fact_any_usage called) a in
  let groups :=
    at_least_one_group (group_by_parameter (ap_args_arity ap) (ap_args ap)) in
  let '(witnesses, a) := fresh_list (length groups) a in
  let a := note_simples (ap_args ap) a in
  let a :=
    unknown_arity_apply_deps returns exn called (combine groups witnesses) a
  in
  let '(apply_node, a) := fresh a in
  match witnesses with
  | first :: _ => (apply_node, cond_alias env apply_node first a)
  | [] => (apply_node, a)
  end.

(* [add_apply] of [traverse_call_kind]. *)
Definition add_apply (env : traverse_env) (id : code_id)
    (callee call_widget : node) (only_if_closure_any_source : bool)
    (a : acc) : acc :=
  if only_if_closure_any_source
  then
    let '(callee_if_any_source, a) := fresh a in
    let '(widget_if_any_source, a) := fresh a in
    let a :=
      add_facts
        [Fact_alias_if_any_source callee callee_if_any_source callee;
         Fact_alias_if_any_source callee call_widget widget_if_any_source]
        a
    in
    push_apply_dep
      {| ad_function_containing_apply_expr := env_current_code_id env;
         ad_apply_code_id := id;
         ad_apply_closure := callee_if_any_source;
         ad_apply_call_witness := widget_if_any_source |}
      a
  else
    push_apply_dep
      {| ad_function_containing_apply_expr := env_current_code_id env;
         ad_apply_code_id := id;
         ad_apply_closure := callee;
         ad_apply_call_witness := call_widget |}
      a.

(* [Traverse.traverse_call_kind], for a call that has a callee. *)
Definition traverse_call_kind (env : traverse_env) (ap : apply_expr)
    (exn_arg : node) (return_args : list node) (a : acc) : acc :=
  let closure := node_of_simple (ap_callee ap) in
  let a := note_simple (ap_callee ap) a in
  match ap_call_kind ap with
  | Direct id =>
    let '(call_widget, a) :=
      make_known_arity_apply_widget env ap return_args exn_arg a in
    let a :=
      add_fact (Fact_accessor call_widget
             (Field_call_witness Known_arity_code_pointer) closure) a in
    let is_external := negb (among id (env_unit_code_ids env)) in
    match env_should_preserve_direct_calls env with
    | Preserve_yes => add_apply env id closure call_widget false a
    | Preserve_auto => add_apply env id closure call_widget true a
    | Preserve_no =>
      if is_external then add_apply env id closure call_widget false a else a
    end
  | Indirect =>
    let '(call_widget, a) :=
      make_unknown_arity_apply_widget env ap return_args exn_arg a in
    add_fact (Fact_accessor call_widget
           (Field_call_witness Unknown_arity_code_pointer) closure) a
  end.

(* [Traverse.traverse_apply]: the extra arguments of the exception
   continuation are aliased to its parameters after the first, which
   stands for the exception. *)
Definition traverse_apply (env : traverse_env) (ap : apply_expr) (a : acc)
    : acc :=
  let '(return_args, a) :=
    find_cont env (ap_continuation ap)
      (length (unarize (ap_return_arity ap))) a in
  let '(exn_params, a) :=
    find_cont env (ap_exn_continuation ap)
      (S (length (ap_exn_extra_args ap))) a in
  let '(exn_arg, a) :=
    match exn_params with
    | [] => fresh a
    | exn_param :: extra_params =>
      let extra_args := exn_extra_simples ap in
      let a := note_simples extra_args a in
      (exn_param,
       add_facts
         (map (fun '(p, s) => Fact_alias p (node_of_simple s))
            (combine extra_params extra_args))
         a)
    end
  in
  traverse_call_kind env ap exn_arg return_args a.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* [Traverse.apply_cont_deps]. *)
Definition apply_cont_deps (env : traverse_env) (ac : apply_cont) (a : acc)
    : acc :=
  let args := ac_args ac in
  let '(params, a) := find_cont env (ac_continuation ac) (length args) a in
  let a := note_simples args a in
  add_facts
    (map (fun '(p, s) => Fact_alias p (node_of_simple s)) (combine params args))
    a.

(* [Traverse.traverse_switch]. *)
Definition traverse_switch (env : traverse_env) (scrutinee : simple)
    (arms : list (Z * apply_cont)) (a : acc) : acc :=
  let a := cond_any_usage env scrutinee a in
  fold_left (fun a '(_, ac) => apply_cont_deps env ac a)
    (Keyed.compact Z.eqb arms) a.

Definition handler_params (h : continuation_handler) : list node :=
  match h with Handler params _ _ => map Node_variable params end.

(* The facts of [traverse_let_cont_non_recursive] about an exception
   handler: its first parameter has any source, and, conditionally, any
   usage. *)
Definition exn_handler_deps (env : traverse_env) (h : continuation_handler)
    (a : acc) : acc :=
  match h with
  | Handler (p :: _) metadata _ =>
    if hm_is_exn_handler metadata
    then
      cond_any_usage env (Var p)
        (add_fact (Fact_any_source (Node_variable p)) a)
    else a
  | Handler [] _ _ => a
  end.

(* The continuations of a recursive [Let_cont] in scope, in its handlers
   and in its body; the first handler of a continuation counts. *)
Definition add_rec_conts (env : traverse_env)
    (handlers : list (continuation * continuation_handler)) : traverse_env :=
  fold_right (fun '(k, h) env => add_cont env k (handler_params h)) env
    handlers.

(* ======================================================================== *)
(* The traversal                                                            *)
(* ======================================================================== *)

(* [Traverse.traverse]. A [Let_static] is [traverse_static_consts] (the
   code deps of all its code first, then its static constants), then the
   traversal of its code ([traverse_code], from the computation of
   [named] in [traverse_let]), then its body. *)
Fixpoint traverse (env : traverse_env) (e : expr) (a : acc) {struct e} : acc :=
  match e with
  | Let x _ defining_expr body =>
    traverse env body (traverse_named env x defining_expr a)
  | Let_static bindings body =>
    let a := fold_left (fun a b => prepare_binding b a) bindings a in
    let a := fold_left (fun a b => traverse_static_binding env b a) bindings a
    in
    let a := fold_left (fun a b => traverse_binding env b a) bindings a in
    traverse env body a
  | Let_cont_non_recursive k h body =>
    let a := traverse_handler env h a in
    let a := exn_handler_deps env h a in
    traverse (add_cont env k (handler_params h)) body a
  | Let_cont_recursive handlers body =>
    let env := add_rec_conts env handlers in
    let a := fold_left (fun a '(_, h) => traverse_handler env h a) handlers a in
    traverse env body a
  | Apply ap => traverse_apply env ap a
  | Apply_cont ac => apply_cont_deps env ac a
  | Switch scrutinee arms => traverse_switch env scrutinee arms a
  | Invalid => a
  end
with traverse_handler (env : traverse_env) (h : continuation_handler) (a : acc)
    {struct h} : acc :=
  match h with
  | Handler _ _ handler => traverse env handler a
  end
with traverse_binding (env : traverse_env) (binding : static_binding) (a : acc)
    {struct binding} : acc :=
  match binding with
  | Bind_code id c => traverse_code env id c a
  | Bind_deleted_code _ | Bind_symbol _ _ => a
  end
(* [Traverse.traverse_function_params_and_body], not opaque: the
   parameters and [my_closure] are aliased to those of the code dep. *)
with traverse_code (env : traverse_env) (id : code_id) (c : code) (a : acc)
    {struct c} : acc :=
  match c with
  | Code params my_closure return_continuation exn_continuation _ body =>
    match lookup id (acc_code_deps a) with
    | Some cd =>
      let a :=
        add_facts
          (Fact_alias (Node_variable my_closure) (cd_my_closure cd)
           :: map (fun '(p, arg) => Fact_alias (Node_variable p) arg)
                (combine params (cd_params cd)))
          a
      in
      traverse
        (code_env env id return_continuation exn_continuation cd) body a
    | None => a
    end
  end.

(* ======================================================================== *)
(* The delayed deps                                                         *)
(* ======================================================================== *)

(* [Traverse_acc.add_alias_for_caller]. *)
Definition alias_for_caller (caller : option code_id) (from to_ : node)
    : fact :=
  match caller with
  | None => Fact_alias to_ from
  | Some id => Fact_propagate (Node_code_id id) to_ from
  end.

(* [Traverse_acc.add_apply_dep]. *)
Definition apply_dep_facts (code_deps : finite_map code_dep) (d : apply_dep)
    : list fact :=
  let caller := ad_function_containing_apply_expr d in
  let closure := ad_apply_closure d in
  let call := ad_apply_call_witness d in
  match lookup (ad_apply_code_id d) code_deps with
  | Some cd =>
    [alias_for_caller caller (cd_known_arity_call_witness cd) call;
     alias_for_caller caller closure (cd_my_closure cd)]
  | None =>
    match caller with
    | None => [Fact_any_source call; Fact_any_usage closure]
    | Some id =>
      [Fact_any_source_if_used (Node_code_id id) call;
       Fact_use (Node_code_id id) closure]
    end
  end.

(* [Traverse_acc.add_closure_dep]. *)
Definition add_closure_dep (code_deps : finite_map code_dep)
    (d : closure_dep) (a : acc) : acc :=
  let closure := cld_let_bound_name_of_the_closure d in
  let id := cld_closure_code_id d in
  match lookup id code_deps with
  | Some cd =>
    add_facts
      (Fact_propagate (Node_code_id id) (cd_my_closure cd) closure
       :: Fact_constructor closure (Field_call_witness Known_arity_code_pointer)
            (cd_known_arity_call_witness cd)
       :: match cd_unknown_arity_call_witnesses cd with
          | witness :: _ =>
            [Fact_constructor closure
               (Field_call_witness Unknown_arity_code_pointer) witness]
          | [] => []
          end)
      a
  | None =>
    let '(witness, a) := fresh a in
    add_facts
      [Fact_any_source witness;
       Fact_constructor closure (Field_call_witness Known_arity_code_pointer)
         witness;
       Fact_constructor closure
         (Field_call_witness Unknown_arity_code_pointer) witness;
       Fact_constructor witness Field_code_id_of_call_witness closure]
      a
  end.

(* [Traverse_acc.resolve_delayed_deps]: [unit_symbols] are the symbols
   the unit binds, the others are imported. *)
Definition resolve_delayed_deps (unit_symbols : list symbol) (a : acc)
    : acc :=
  let code_deps := acc_code_deps a in
  let a := add_facts (flat_map (apply_dep_facts code_deps) (acc_apply_deps a)) a
  in
  let a :=
    fold_left (fun a d => add_closure_dep code_deps d a) (acc_closure_deps a) a
  in
  add_facts
    (map (fun s => Fact_any_source (Node_symbol s))
       (without unit_symbols (dedup (acc_symbols a))))
    a.

(* ======================================================================== *)
(* The unit                                                                 *)
(* ======================================================================== *)

(* The value slots that the projections of an expression name, the
   bodies of code included ([value_slots_in_normal_projections]). *)
Definition projected_value_slots_of_named (defining_expr : named)
    : list value_slot :=
  match defining_expr with
  | Project_value_slot w _ | Project_value_slot_from _ w _ => [w]
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ => []
  end.

Fixpoint projected_value_slots (e : expr) : list value_slot :=
  match e with
  | Let _ _ defining_expr body =>
    projected_value_slots_of_named defining_expr ++ projected_value_slots body
  | Let_static bindings body =>
    flat_map binding_projected_value_slots bindings
    ++ projected_value_slots body
  | Let_cont_non_recursive _ h body =>
    handler_projected_value_slots h ++ projected_value_slots body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_projected_value_slots h) handlers
    ++ projected_value_slots body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_projected_value_slots (h : continuation_handler)
    : list value_slot :=
  match h with
  | Handler _ _ handler => projected_value_slots handler
  end
with binding_projected_value_slots (binding : static_binding)
    : list value_slot :=
  match binding with
  | Bind_code _ c => code_projected_value_slots c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_projected_value_slots (c : code) : list value_slot :=
  match c with
  | Code _ _ _ _ _ body => projected_value_slots body
  end.

(* The code ids that a binding binds, deleted code included: those of the
   current unit. *)
Definition all_code_ids_of_binding (binding : static_binding)
    : list code_id :=
  match binding with
  | Bind_code id _ | Bind_deleted_code id => [id]
  | Bind_symbol _ _ => []
  end.

(* [Traverse.run] and [run0], then what [Analysis.solve] adds before it
   solves ([any_usage] of the toplevel return, for the analysis scope
   [Current_unit]) and [resolve_delayed_deps]: the facts of the graph of
   the unit [e] with the interface [u]. *)
Definition traverse_unit (flag : reaper_preserve_direct_calls)
    (u : unit_interface) (e : expr) : facts :=
  let a := add_fact (Fact_any_source Node_all_constants) empty_acc in
  let '(dummy_toplevel_return, a) := fresh a in
  let '(dummy_toplevel_exn, a) := fresh a in
  let a := add_fact (Fact_any_usage dummy_toplevel_exn) a in
  let env :=
    {| env_conts :=
         [(unit_exn u, [dummy_toplevel_exn]);
          (unit_return u, [dummy_toplevel_return])];
       env_current_code_id := None;
       env_should_preserve_direct_calls := should_preserve_of_flag flag;
       env_value_slots_to_keep := projected_value_slots e;
       env_unit_code_ids := static_names all_code_ids_of_binding e |}
  in
  let a := traverse env e a in
  let a := add_fact (Fact_any_usage dummy_toplevel_return) a in
  let a := resolve_delayed_deps (static_names symbols_of_binding e) a in
  facts_of_list (acc_facts a).

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Module Examples.

Local Open Scope positive_scope.

(* A unit that returns to 1 and raises to 2. Made node 1 is the
   parameter of the first (the toplevel return), 2 that of the second. *)
Definition interface : unit_interface :=
  {| unit_return := 1; unit_exn := 2 |}.

Definition return_to (k : continuation) (args : list simple) : expr :=
  Apply_cont {| ac_continuation := k; ac_args := args;
                ac_trap_action := None |}.

(* [let x10 = Make_block 0 [1; 2] in let x11 = x10.(1) in return x11]:
   the block has its fields and [Is_int] and [Get_tag], the load reads
   field 1, and the toplevel return is aliased to the result. *)
Definition block_and_load : expr :=
  Let 10 Value
    (Prim (Variadic (Make_block (Block_kind_values 0)))
       [Const (Tagged_immediate 1); Const (Tagged_immediate 2)])
    (Let 11 Value
       (Prim (Unary (Block_load (Block_access_values None None Any_value)
                       Immutable 1)) [Var 10])
       (return_to 1 [Var 11])).

Example traverse_block_and_load :
  traverse_unit Never interface block_and_load =
  {| f_alias := [(Node_made 1, Node_variable 11)];
     f_use := [];
     f_accessor := [(Node_variable 11, Field_block 1 Value, Node_variable 10)];
     f_constructor :=
       [(Node_variable 10, Field_block 0 Value, Node_all_constants);
        (Node_variable 10, Field_block 1 Value, Node_all_constants);
        (Node_variable 10, Field_is_int, Node_all_constants);
        (Node_variable 10, Field_get_tag, Node_all_constants)];
     f_argument := [];
     f_parameter := [];
     f_propagate := [];
     f_alias_if_any_source := [];
     f_any_usage := [Node_made 1; Node_made 2];
     f_any_source := [Node_all_constants];
     f_any_source_if_used := [];
     f_zero_alloc_source := [] |}.
Proof. vm_compute. reflexivity. Qed.

(* The code 20 returns its parameter (30; its [my_closure] is 31, its
   continuations 3 and 4). The unit makes the closure 13 of it, at the
   function slot 40, calls it directly with 42 and returns the result,
   through the continuation 5 whose parameter is 12. *)
Definition identity_code : code :=
  Code [30] 31 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (return_to 3 [Var 30]).

Definition call_identity (kind : call_kind) : expr :=
  Let_static [Bind_code 20 identity_code]
    (Let 13 Value (Closure_of_set 40 [(40, 20)] [])
       (Let_cont_non_recursive 5
          (Handler [12] (make_handler_metadata [Value]) (return_to 1 [Var 12]))
          (Apply {| ap_callee := Var 13; ap_call_kind := kind;
                    ap_args := [Const (Tagged_immediate 42)];
                    ap_continuation := 5; ap_exn_continuation := 2;
                    ap_exn_extra_args := [];
                    ap_args_arity := [Singleton Value];
                    ap_return_arity := [Singleton Value] |}))).

(* With [Auto]. The code dep of 20 is made first: its result 3, its
   exception 4, its [my_closure] 5, its parameter 6, its witnesses 7 (of
   known arity) and 8 (of unknown arity). The body of the code aliases
   its parameter and [my_closure] to these, and its result to its
   parameter. The call makes the witness 9, [called] 10 and [apply] 11,
   which reads the known-arity witness of the closure; as the callee may
   be any closure, the call is resolved through 12 and 13, which are
   aliases of the callee and of the call only if the callee has any
   source; the delayed dep of the call aliases 13 to the witness of the
   code, and [my_closure] to 12. The delayed dep of the closure gives it
   the two witnesses of the code, and [my_closure] if the code is
   used. *)
Example traverse_direct_call :
  traverse_unit Auto interface (call_identity (Direct 20)) =
  {| f_alias :=
       [(Node_made 13, Node_made 7); (Node_made 5, Node_made 12);
        (Node_made 9, Node_made 11); (Node_made 1, Node_variable 12);
        (Node_made 3, Node_variable 30); (Node_variable 31, Node_made 5);
        (Node_variable 30, Node_made 6)];
     f_use := [];
     f_accessor :=
       [(Node_made 11, Field_call_witness Known_arity_code_pointer,
         Node_variable 13);
        (Node_variable 12, Field_return_of_call (Normal 0), Node_made 9);
        (Node_made 2, Field_return_of_call Exn, Node_made 9);
        (Node_made 10, Field_code_id_of_call_witness, Node_made 9)];
     f_constructor :=
       [(Node_variable 13, Field_call_witness Known_arity_code_pointer,
         Node_made 7);
        (Node_variable 13, Field_call_witness Unknown_arity_code_pointer,
         Node_made 8);
        (Node_variable 13, Field_function_slot 40, Node_variable 13);
        (Node_made 8, Field_return_of_call Exn, Node_made 4);
        (Node_made 8, Field_return_of_call (Normal 0), Node_made 3);
        (Node_made 8, Field_code_id_of_call_witness, Node_code_id 20);
        (Node_made 7, Field_return_of_call (Normal 0), Node_made 3);
        (Node_made 7, Field_return_of_call Exn, Node_made 4);
        (Node_made 7, Field_code_id_of_call_witness, Node_code_id 20)];
     f_argument := [(Node_all_constants, 0%nat, Node_made 9)];
     f_parameter :=
       [(Node_made 8, 0%nat, Node_made 6); (Node_made 7, 0%nat, Node_made 6)];
     f_propagate := [(Node_code_id 20, Node_made 5, Node_variable 13)];
     f_alias_if_any_source :=
       [(Node_variable 13, Node_made 12, Node_variable 13);
        (Node_variable 13, Node_made 11, Node_made 13)];
     f_any_usage := [Node_made 1; Node_made 10; Node_made 2];
     f_any_source := [Node_code_id 20; Node_all_constants];
     f_any_source_if_used := [];
     f_zero_alloc_source := [] |}.
Proof. vm_compute. reflexivity. Qed.

(* The same call made indirect: [called] is 9, the one witness of the
   call (one parameter) is 10, and [apply] 11 reads the unknown-arity
   witness of the closure. There is no delayed dep of the call. *)
Example traverse_indirect_call :
  f_accessor (traverse_unit Never interface (call_identity Indirect)) =
  [(Node_made 11, Field_call_witness Unknown_arity_code_pointer,
    Node_variable 13);
   (Node_variable 12, Field_return_of_call (Normal 0), Node_made 10);
   (Node_made 2, Field_return_of_call Exn, Node_made 10);
   (Node_made 9, Field_code_id_of_call_witness, Node_made 10)]
  ∧ f_alias (traverse_unit Never interface (call_identity Indirect)) =
  [(Node_made 10, Node_made 11); (Node_made 1, Node_variable 12);
   (Node_made 3, Node_variable 30); (Node_variable 31, Node_made 5);
   (Node_variable 30, Node_made 6)].
Proof. vm_compute. split; reflexivity. Qed.

(* Tupled code of two parameters (30 and 32). The witness of unknown
   arity, 9, takes one parameter, the tuple 10, whose fields are the
   parameters 6 and 7 of the code dep; 11 reads [Is_int] of the tuple
   and is used if the code is. *)
Definition tupled_code : code :=
  Code [30; 32] 31 3 4
    (make_tupled_code_metadata [Singleton Value; Singleton Value]
       [Singleton Value])
    (return_to 3 [Var 32]).

Definition bind_tupled : expr :=
  Let_static [Bind_code 21 tupled_code]
    (return_to 1 [Const (Tagged_immediate 0)]).

Example traverse_tupled_code :
  f_use (traverse_unit Never interface bind_tupled)
  = [(Node_code_id 21, Node_made 11)]
  ∧ f_accessor (traverse_unit Never interface bind_tupled)
    = [(Node_made 6, Field_block 0 Value, Node_made 10);
       (Node_made 7, Field_block 1 Value, Node_made 10);
       (Node_made 11, Field_is_int, Node_made 10)]
  ∧ f_parameter (traverse_unit Never interface bind_tupled)
    = [(Node_made 9, 0%nat, Node_made 10);
       (Node_made 8, 0%nat, Node_made 6);
       (Node_made 8, 1%nat, Node_made 7)].
Proof. vm_compute. repeat split. Qed.

(* An exception handler 6, whose parameter 14 has any source and any
   usage; [Opaque_identity], which has arbitrary effects, of the symbol
   50, which the unit does not bind; and an alias of the symbol 51,
   which it does not bind either. The symbol 51 went through
   [simple_to_node] and is given [any_source], as an imported symbol;
   50 is only a free name of a primitive and is not, as in the
   compiler. *)
Definition exn_handler_and_effects : expr :=
  Let_cont_non_recursive 6
    (Handler [14] (make_exn_handler_metadata [Value]) (return_to 1 [Var 14]))
    (Let 15 Value (Prim (Unary (Opaque_identity Value)) [Symbol 50])
       (Let 16 Value (Simple (Symbol 51)) (return_to 1 [Var 15]))).

Example traverse_exn_handler_and_effects :
  traverse_unit Never interface exn_handler_and_effects =
  {| f_alias :=
       [(Node_made 1, Node_variable 15); (Node_variable 16, Node_symbol 51);
        (Node_made 1, Node_variable 14)];
     f_use := [(Node_variable 15, Node_symbol 50)];
     f_accessor := [];
     f_constructor := [];
     f_argument := [];
     f_parameter := [];
     f_propagate := [];
     f_alias_if_any_source := [];
     f_any_usage :=
       [Node_made 1; Node_variable 15; Node_variable 14; Node_made 2];
     f_any_source :=
       [Node_symbol 51; Node_variable 15; Node_variable 14;
        Node_all_constants];
     f_any_source_if_used := [];
     f_zero_alloc_source := [] |}.
Proof. vm_compute. reflexivity. Qed.

End Examples.
