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

(* Calls of known arity: what a call that the compiler marks
   [Indirect_known_arity] comes to.

   This file is to be reviewed: the statements, against
   FlambdaSemanticsSpec.code_ids_allow, [known_arity_full] and
   [shape_of_closure_call], which they restate, and against what the
   compiler does with such a call (simplify/simplify_apply_expr.ml,
   [simplify_function_call] and
   [simplify_function_call_where_callee's_type_unavailable];
   to_cmm/to_cmm_expr.ml, the case [Indirect_known_arity];
   backend/cmm_helpers.ml, [indirect_full_call]). The two review
   comments at those definitions say where the compiler and the model
   part (decisions D58 and D59).

   The compiler has three kinds of call of a function
   ([Call_kind.Function_call.t]). A direct call names the code. A call
   of unknown arity is the generic application: it may give the callee
   fewer or more arguments than its code has parameters. A call of
   known arity promises that the callee is a closure of code that has
   the arity of the call, so that the backend calls the code with the
   arguments and tests nothing; it may list the code ids that the code
   of the callee is among.

   - A call of known arity that has a shape is a full call, of code
     that is not tupled, and it is what the call of unknown arity of the
     same callee and arguments is ([known_arity_shape_statement]); and
     it has that shape wherever its tests pass
     ([known_arity_defined_statement]). So it is the call of unknown arity
     kept where that is such a full call, and nowhere else
     ([known_arity_of_unknown_statement]).
   - So a step of such a call, in each of the three machines, is the
     step of the call of unknown arity ([known_arity_call_statement],
     [known_arity_step_statement], [known_arity_heap_step_statement],
     [known_arity_stack_step_statement]). Not the other way: the call of
     unknown arity is defined where this one is not.
   - It has no shape: with no group of arguments
     ([known_arity_no_group_statement]); of a partial application, also
     one that is given exactly what is left
     ([known_arity_partial_application_statement]); of tupled code
     ([known_arity_tupled_statement]); with another number of groups of
     arguments than the code has parameters
     ([known_arity_other_arity_statement]); of code that its list has
     not ([known_arity_not_listed_statement]).
   - The list is read as a set: two lists with the same members allow
     the same code ([code_ids_allow_members_statement]), and the empty
     list allows none ([code_ids_allow_none_statement]). A call that
     has a shape keeps it when its list is forgotten
     ([known_arity_forget_statement]).
   - The examples: calls that are computed, of the code of
     FlambdaPartialApplicationSpec.v and FlambdaTupledSpec.v
     ([known_arity_examples_run_statement],
     [known_arity_examples_undefined_statement],
     [unknown_arity_examples_run_statement],
     [unknown_arity_examples_undefined_statement]).

   The proofs are in FlambdaKnownArityProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaPartialApplicationSpec
  FlambdaTupledSpec FlambdaHeapMachineSpec FlambdaTrapSpec.
Import ListNotations.

(* ======================================================================== *)
(* The shape of a call of known arity                                       *)
(* ======================================================================== *)

(* A call of known arity that has a shape: the call of unknown arity of
   the same callee and arguments has that shape; the callee is a
   closure of code that is bound, is not tupled, has as many parameters
   as the call has groups of arguments, of which there is one at least,
   and is allowed by the list; and the shape is the full call of that
   code with the arguments as they are. *)
Definition known_arity_shape_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) (shape : call_shape),
    shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity
    = Some shape ->
    shape_of_call statics callee Indirect_unknown_arity vs args_arity
      return_arity
    = Some shape
    ∧ ∃ (called : code_id) (slots : list (value_slot * value)) (c : code),
        closure_of statics callee = Some (called, slots)
        ∧ lookup called (static_code statics) = Some c
        ∧ code_is_tupled c = false
        ∧ args_arity <> []
        ∧ length args_arity = length (code_params_arity c)
        ∧ code_ids_allow code_ids called = true
        ∧ shape = Full_call callee c vs args_arity.

(* Where those tests pass, the call has that shape. *)
Definition known_arity_defined_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = false ->
    args_arity <> [] ->
    length args_arity = length (code_params_arity c) ->
    code_ids_allow code_ids called = true ->
    shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity
    = Some (Full_call callee c vs args_arity).

(* The same in one equation: a call of known arity is the call of
   unknown arity of the same callee and arguments, kept where the call
   has a group of arguments, the callee is a closure and no partial
   application, and the call of unknown arity is a full call of code
   that is not tupled and that the list allows. *)
Definition known_arity_of_unknown_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity),
    shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity
    = match
        args_arity, partial_application_of callee,
        closure_of statics callee,
        shape_of_call statics callee Indirect_unknown_arity vs args_arity
          return_arity
      with
      | _ :: _, None, Some (called, _), Some (Full_call closure c vs' arity')
        =>
        if code_ids_allow code_ids called && negb (code_is_tupled c)
        then Some (Full_call closure c vs' arity')
        else None
      | _, _, _, _ => None
      end.

(* ======================================================================== *)
(* A step of a call of known arity                                          *)
(* ======================================================================== *)

(* A call of known arity that is not undefined is the call of unknown
   arity. *)
Definition known_arity_call_statement : Prop :=
  ∀ (callee : value) (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record),
    call_function callee (Indirect_known_arity code_ids) vs args_arity
      return_arity return_to exn_to statics hp rd
    <> Undefined ->
    call_function callee Indirect_unknown_arity vs args_arity return_arity
      return_to exn_to statics hp rd
    = call_function callee (Indirect_known_arity code_ids) vs args_arity
        return_arity return_to exn_to statics hp rd.

(* A call with the kind [kind] in place of its own. *)
Definition apply_with_kind (kind : call_kind) (ap : apply_expr)
    : apply_expr :=
  Apply_expr (ap_callee ap) kind (ap_args ap) (ap_continuation ap)
    (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
    (ap_return_arity ap).

(* A call of known arity that has a step: the same call made of unknown
   arity has that step. *)
Definition known_arity_step_statement : Prop :=
  ∀ (ap : apply_expr) (code_ids : option (list code_id)) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    ap_call_kind ap = Indirect_known_arity code_ids ->
    step (State (Apply ap) vars conts statics hp rd) <> Undefined ->
    step
      (State (Apply (apply_with_kind Indirect_unknown_arity ap)) vars conts
         statics hp rd)
    = step (State (Apply ap) vars conts statics hp rd).

(* The same of the machine with a heap (FlambdaHeapMachineSpec.v). *)
Definition known_arity_heap_step_statement : Prop :=
  ∀ (ap : apply_expr) (code_ids : option (list code_id)) (vars : env)
    (conts : cont_env) (statics : static_env) (h : heap),
    ap_call_kind ap = Indirect_known_arity code_ids ->
    heap_step (Heap_state (Apply ap) vars conts statics h) <> Heap_undefined ->
    heap_step
      (Heap_state (Apply (apply_with_kind Indirect_unknown_arity ap)) vars
         conts statics h)
    = heap_step (Heap_state (Apply ap) vars conts statics h).

(* The same of the machine with a stack of handlers (FlambdaTrapSpec.v). *)
Definition known_arity_stack_step_statement : Prop :=
  ∀ (ap : apply_expr) (code_ids : option (list code_id)) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record) (st : stack),
    ap_call_kind ap = Indirect_known_arity code_ids ->
    stack_step (State (Apply ap) vars conts statics hp rd) st
    <> Stack_undefined ->
    stack_step
      (State (Apply (apply_with_kind Indirect_unknown_arity ap)) vars conts
         statics hp rd)
      st
    = stack_step (State (Apply ap) vars conts statics hp rd) st.

(* ======================================================================== *)
(* Where a call of known arity is undefined                                 *)
(* ======================================================================== *)

(* With no group of arguments. (A group may unarize to no value: the
   arity is what is tested, not the values.) *)
Definition known_arity_no_group_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (return_arity : arity),
    shape_of_call statics callee (Indirect_known_arity code_ids) vs []
      return_arity
    = None.

(* Of a partial application, whatever it is given. *)
Definition known_arity_partial_application_statement : Prop :=
  ∀ (statics : static_env) (closure : value) (applied : list value)
    (applied_arity : arity) (code_ids : option (list code_id))
    (vs : list value) (args_arity return_arity : arity),
    shape_of_call statics
      (Value_partial_application closure applied applied_arity)
      (Indirect_known_arity code_ids) vs args_arity return_arity
    = None.

(* Of a closure of tupled code, whatever it is given. *)
Definition known_arity_tupled_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = true ->
    shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity
    = None.

(* With another number of groups of arguments than the code of the
   callee has parameters: fewer or more, and also where the call has
   one. *)
Definition known_arity_other_arity_statement : Prop :=
  ∀ (statics : static_env) (callee : value)
    (code_ids : option (list code_id)) (vs : list value)
    (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    length args_arity <> length (code_params_arity c) ->
    shape_of_call statics callee (Indirect_known_arity code_ids) vs
      args_arity return_arity
    = None.

(* Of a closure of code that the list has not. *)
Definition known_arity_not_listed_statement : Prop :=
  ∀ (statics : static_env) (callee : value) (ids : list code_id)
    (vs : list value) (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)),
    closure_of statics callee = Some (called, slots) ->
    called ∉ ids ->
    shape_of_call statics callee (Indirect_known_arity (Some ids)) vs
      args_arity return_arity
    = None.

(* ======================================================================== *)
(* The list of code ids                                                     *)
(* ======================================================================== *)

(* A list allows its members and nothing else; no list allows every
   code. So two lists with the same members allow the same code,
   whatever their order and however often they name an id. *)
Definition code_ids_allow_members_statement : Prop :=
  ∀ (ids : list code_id) (called : code_id),
    code_ids_allow (Some ids) called = true <-> called ∈ ids.

Definition code_ids_allow_none_statement : Prop :=
  ∀ called : code_id,
    code_ids_allow (Some []) called = false
    ∧ code_ids_allow None called = true.

(* A call of known arity that has a shape has it with no list. *)
Definition known_arity_forget_statement : Prop :=
  ∀ (statics : static_env) (callee : value) (ids : list code_id)
    (vs : list value) (args_arity return_arity : arity)
    (shape : call_shape),
    shape_of_call statics callee (Indirect_known_arity (Some ids)) vs
      args_arity return_arity
    = Some shape ->
    shape_of_call statics callee (Indirect_known_arity None) vs args_arity
      return_arity
    = Some shape.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

(* Calls of the code of FlambdaPartialApplicationSpec.v ([first x y] is
   [x], [identity x] is [x]) and of FlambdaTupledSpec.v ([first (x, y)]
   is [x], tupled), and of five more pieces of code: [constant], which
   has no parameter and returns 1; [of_void], which returns 1 and has
   one parameter that stands for no variable; [newer_first], another
   code id bound to the code of [first], as a newer version of it would
   be; [tupled_one], tupled code of one parameter, which returns it; and
   [tupled_none], tupled code of no parameter, which returns 1. *)
Section Examples.

Let x : variable := 1%positive.
Let my_closure : variable := 3%positive.
Let partial : variable := 4%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 1%positive.
Let error : continuation := 2%positive.
Let apply_rest : continuation := 3%positive.

Definition two_values : arity := [Singleton Value; Singleton Value].
Definition three_values : arity :=
  [Singleton Value; Singleton Value; Singleton Value].
Definition one_void : arity := [Unboxed_product []].
Definition one_naked_immediate : arity :=
  [Singleton (Naked_number Kind_naked_immediate)].

(* The kinds of [two_values], grouped otherwise: one group of two values
   and one of none. *)
Definition two_values_regrouped : arity :=
  [Unboxed_product [Singleton Value; Singleton Value]; Unboxed_product []].

Definition returns_one (metadata : code_metadata) : code :=
  Code [] my_closure ret exn metadata
    (Apply_cont (Apply_cont_expr ret [Const (Tagged_immediate 1)] None)).

Definition tupled_one_code : code :=
  Code [x] my_closure ret exn (make_tupled_code_metadata one_value one_value)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

Definition constant_code_id : code_id := 3%positive.
Definition of_void_code_id : code_id := 4%positive.
Definition newer_first_code_id : code_id := 5%positive.
Definition tupled_one_code_id : code_id := 6%positive.
Definition tupled_none_code_id : code_id := 7%positive.
Definition constant_symbol : symbol := 3%positive.
Definition of_void_symbol : symbol := 4%positive.
Definition newer_first_symbol : symbol := 5%positive.
Definition tupled_one_symbol : symbol := 6%positive.
Definition tupled_none_symbol : symbol := 7%positive.
Definition single_symbol : symbol := 8%positive.

Definition known_arity_bindings : list static_binding :=
  example_bindings
  ++ [Bind_code constant_code_id
        (returns_one (make_code_metadata [] one_value));
      Bind_code of_void_code_id
        (returns_one (make_code_metadata one_void one_value));
      Bind_code newer_first_code_id first_code;
      Bind_code tupled_one_code_id tupled_one_code;
      Bind_code tupled_none_code_id
        (returns_one (make_tupled_code_metadata [] one_value));
      Bind_symbol constant_symbol
        (Static_const_set_of_closures constant_code_id []);
      Bind_symbol of_void_symbol
        (Static_const_set_of_closures of_void_code_id []);
      Bind_symbol newer_first_symbol
        (Static_const_set_of_closures newer_first_code_id []);
      Bind_symbol tupled_one_symbol
        (Static_const_set_of_closures tupled_one_code_id []);
      Bind_symbol tupled_none_symbol
        (Static_const_set_of_closures tupled_none_code_id []);
      Bind_symbol single_symbol
        (Static_const_block 0 [Const (Tagged_immediate 1)])].

(* A call of the closure that a symbol is bound to. *)
Definition call_symbol (bindings : list static_binding) (s : symbol)
    (kind : call_kind) (args : list simple) (args_arity return_arity : arity)
    : expr :=
  Let_static bindings
    (Apply
       (Apply_expr (Symbol s) kind args done error [] args_arity
          return_arity)).

Definition one_and_two : list simple :=
  [Const (Tagged_immediate 1); Const (Tagged_immediate 2)].

(* [first 1 2]: a full call, of two groups of arguments. *)
Definition call_first (kind : call_kind) : expr :=
  call_symbol known_arity_bindings first_symbol kind one_and_two two_values
    one_value.

(* The same call of the closure of [newer_first]. *)
Definition call_newer_first (kind : call_kind) : expr :=
  call_symbol known_arity_bindings newer_first_symbol kind one_and_two
    two_values one_value.

(* [identity 1]: a full call, of one group. *)
Definition call_identity (kind : call_kind) : expr :=
  call_symbol known_arity_bindings identity_symbol kind
    [Const (Tagged_immediate 1)] one_value one_value.

(* [first 1]: one group of arguments, of code with two parameters. *)
Definition call_first_with_one (kind : call_kind) : expr :=
  call_symbol known_arity_bindings first_symbol kind
    [Const (Tagged_immediate 1)] one_value one_value.

(* [identity 1 2 3]: three groups, of code with one parameter. *)
Definition call_identity_with_three (kind : call_kind) : expr :=
  call_symbol known_arity_bindings identity_symbol kind
    (one_and_two ++ [Const (Tagged_immediate 3)]) three_values one_value.

(* [constant], given no group of arguments. *)
Definition call_constant (kind : call_kind) : expr :=
  call_symbol known_arity_bindings constant_symbol kind [] [] one_value.

(* [of_void], given one group that stands for no value. *)
Definition call_of_void (kind : call_kind) : expr :=
  call_symbol known_arity_bindings of_void_symbol kind [] one_void one_value.

(* [first 1 2] with the two arguments in one group, followed by a group
   of none: as many groups as [first] has parameters, and the same
   kinds, grouped otherwise. *)
Definition call_first_regrouped (kind : call_kind) : expr :=
  call_symbol known_arity_bindings first_symbol kind one_and_two
    two_values_regrouped one_value.

(* [first] given a naked immediate where its arity says a value. *)
Definition call_first_of_naked (kind : call_kind) : expr :=
  call_symbol known_arity_bindings first_symbol kind
    [Const (Naked_immediate 1); Const (Tagged_immediate 2)] two_values
    one_value.

(* [first 1 2], said to return a naked immediate. *)
Definition call_first_to_naked (kind : call_kind) : expr :=
  call_symbol known_arity_bindings first_symbol kind one_and_two two_values
    one_naked_immediate.

(* [first 1], of unknown arity, and then its result, a partial
   application, called with [2]: exactly what is left. *)
Definition call_partial (kind : call_kind) : expr :=
  Let_static known_arity_bindings
    (Let_cont_non_recursive apply_rest
       (Handler [partial] (make_handler_metadata [Value])
          (Apply
             (Apply_expr (Var partial) kind [Const (Tagged_immediate 2)] done
                error [] one_value one_value)))
       (Apply
          (Apply_expr (Symbol first_symbol) Indirect_unknown_arity
             [Const (Tagged_immediate 1)] apply_rest error [] one_value
             one_value))).

(* The tupled [first], given the pair [(1, 2)]. *)
Definition call_tupled_with_tuple (kind : call_kind) : expr :=
  call_symbol tupled_bindings tupled_first_symbol kind [Symbol pair_symbol]
    one_value one_value.

(* The tupled [first], given the components [1] and [2], as the
   compiler's call of known arity gives them. *)
Definition call_tupled_with_components (kind : call_kind) : expr :=
  call_symbol tupled_bindings tupled_first_symbol kind one_and_two
    two_values one_value.

(* [tupled_one], given the tuple of the one field [1]: one group, of
   tupled code that has one parameter. *)
Definition call_tupled_one (kind : call_kind) : expr :=
  call_symbol known_arity_bindings tupled_one_symbol kind
    [Symbol single_symbol] one_value one_value.

(* [tupled_none], given one group. *)
Definition call_tupled_none (kind : call_kind) : expr :=
  call_symbol known_arity_bindings tupled_none_symbol kind
    [Symbol single_symbol] one_value one_value.

(* Calls of known arity that run, and return 1: the full call of
   [first], with no list, with a list that has its code, with one that
   has it twice among other code, and with the same members in another
   order; the full call of [identity], which has one group; the call
   of [of_void], whose one group stands for no value; the call of
   [newer_first] with a list that has its code id. *)
Definition known_arity_examples_run_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [call_first (Indirect_known_arity None);
       call_first (Indirect_known_arity (Some [first_code_id]));
       call_first
         (Indirect_known_arity
            (Some [identity_code_id; first_code_id; first_code_id]));
       call_first
         (Indirect_known_arity (Some [first_code_id; identity_code_id]));
       call_identity (Indirect_known_arity None);
       call_of_void (Indirect_known_arity None);
       call_newer_first
         (Indirect_known_arity (Some [newer_first_code_id]))] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 1)] empty_heap).

(* Calls of known arity that are undefined.
   - By the list: of [first] with the empty list, and with a list that
     has other code only; of [newer_first] with a list that has the
     code id of [first] and not its own, though the two are bound to
     the same code (the model has no relation of code to its newer
     versions: decision D59).
   - By the arity: with one group of arguments, of [first]; with three,
     of [identity]; with no group, of [constant].
   - Of a partial application, given exactly what is left.
   - Of tupled code: of two parameters, given the tuple, and given the
     components; of one parameter, given one group; of no parameter.
   - As the call of unknown arity of the same data is, by the tests of
     the full call ([call_code]): the kinds of [first] grouped
     otherwise; an argument of another kind than its arity says; a
     return arity that is not the result arity of the code. *)
Definition known_arity_examples_undefined_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [call_first (Indirect_known_arity (Some []));
       call_first (Indirect_known_arity (Some [identity_code_id]));
       call_newer_first (Indirect_known_arity (Some [first_code_id]));
       call_first_with_one (Indirect_known_arity None);
       call_identity_with_three (Indirect_known_arity None);
       call_constant (Indirect_known_arity None);
       call_partial (Indirect_known_arity None);
       call_tupled_with_tuple (Indirect_known_arity None);
       call_tupled_with_components (Indirect_known_arity None);
       call_tupled_one (Indirect_known_arity None);
       call_tupled_none (Indirect_known_arity None);
       call_first_regrouped (Indirect_known_arity None);
       call_first_of_naked (Indirect_known_arity None);
       call_first_to_naked (Indirect_known_arity None)] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong.

(* The full call of [first], and four of the calls that are undefined
   of known arity, made of unknown arity: they run and return 1. The
   four: with no group, of [constant]; of the partial application; of
   tupled code of two parameters, given the tuple; of tupled code of
   one parameter, given the tuple. *)
Definition unknown_arity_examples_run_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [call_first Indirect_unknown_arity;
       call_constant Indirect_unknown_arity;
       call_partial Indirect_unknown_arity;
       call_tupled_with_tuple Indirect_unknown_arity;
       call_tupled_one Indirect_unknown_arity] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 1)] empty_heap).

(* The three calls that fail the tests of the full call are undefined
   of unknown arity too. *)
Definition unknown_arity_examples_undefined_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [call_first_regrouped Indirect_unknown_arity;
       call_first_of_naked Indirect_unknown_arity;
       call_first_to_naked Indirect_unknown_arity] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong.

End Examples.
