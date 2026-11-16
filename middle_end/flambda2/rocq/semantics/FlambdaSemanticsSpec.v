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

(* Execution of the terms of FlambdaModel.v, and what it means for two
   terms to behave alike.

   This file is to be reviewed: [closure_of], [closure_in_set],
   [project_value_slot], [project_function_slot], [eval_named],
   [eval_static_bindings], [call_code], [tuple_fields], [detuple],
   [arguments_for_code], [shape_of_code_call], [shape_of_closure_call],
   [shape_of_call], [run_call], [apply_cont_value], [with_extra_args],
   [call_function] and the rules of [step], then [behaves],
   [equivalent] and [refines]. The values, the static
   environment and the reading of allocated values are in
   FlambdaValueSpec.v, and what the primitives compute is in
   FlambdaPrimitiveSpec.v; this file exports both.

   The semantics is an abstract machine. A state is an expression with an
   environment giving values to variables, one giving closures to
   continuations, and the static environment; [step] is a function, so
   execution is deterministic by construction. A continuation closure keeps
   the environments of its definition, so no substitution is involved and
   binders may shadow.

   The static environment is what [Let_static] has bound so far: the code
   of code ids, and the values that symbols point to. It is global: it is
   not kept in closures, and only grows. A symbol is a value by itself, a
   pointer ([Value_symbol]), whether or not it is bound; what it points to
   is looked up where it is used, as a closure to call or to project
   from. So a group of bindings may be recursive, and static data cyclic.
   A [Let_static] is executed where it stands; nothing lifts it.

   Flambda permits a [Let_static] at the toplevel of a compilation unit
   only: FlambdaWellFormedSpec.v has the condition. The machine keeps
   static bindings out of the two places where one could be executed more
   than once: calling code whose body binds something static is
   undefined, and so is a [Let_cont_recursive] one of whose handlers does
   ([binds_statics]). It also makes undefined binding a code id or a
   symbol that is bound already. So what a name is bound to never changes
   in a run, and what is known of the static environment where a handler
   is defined still holds where it runs. The rest of the condition, that
   the handler of a non-recursive continuation with static bindings
   postdominates its body, is not the machine's to check.

   A continuation with no binding stands for a way out of the term: the
   return or exception continuation of the enclosing unit. Applying one
   ends execution: the result [Exit] carries the continuation, its
   arguments, and the heap and the read record of the state. What is
   observed of an exit ([Exits]) is the continuation, the values and
   the heap, and not the read record.

   A state has a heap (FlambdaHeapSpec.v: locations bound to cells, a
   cell being a block with an identity or a mutable array) and a read
   record (FlambdaValueSpec.read_record). A [Let] of a primitive is
   run on the heap ([exec_named], by FlambdaHeapSpec.exec_prim): a
   primitive of the heap makes a cell or writes to one, and a load,
   [Get_tag], [Is_int], [Is_null], [Opaque_identity], the length of an
   array and [Phys_equal] read a cell, or answer of it, where they are
   given a location. A step changes the heap only at a [Let]
   ([step_heap_statement]). The read record is carried: nothing adds
   to it ([step_keeps_reads_statement]) and nothing reads it
   ([step_ignores_reads_statement]; decisions D49 and D52). Two exits
   are compared with their heaps, which must
   be equal ([refines]; decision D40).

   A function call ([call_function]) runs the body of the code of the
   closure called, with the code's return and exception continuations
   standing for the caller's. There is no stack.

   A call need not give a function as many arguments as its code has
   parameters: it is the generic application of the compiler
   ([shape_of_call] says what a call is). With fewer arguments it makes
   a partial application, a value of its own that keeps the closure and
   the arguments so far ([Value_partial_application]), and gives it to
   the return continuation of the call; an indirect call of that value
   with more arguments is the call of the closure with all of them. With
   more arguments it is a full call with the first of them, whose result
   is then called with the rest: the full call returns to what is left
   of the application ([Over_application]), which is a continuation that
   no term names. A partial application is not a closure: nothing can be
   projected from it, and a direct call of it is undefined. The compiler
   has no such value: Simplify turns the partial application of a
   function it knows into a closure of new code, and at run time one of
   a function it does not know is a closure of [caml_curryN]. The
   machine cannot make code, and its one value stands for both.

   Tupled code (FlambdaModel.code_is_tupled) has a parameter for each
   component of a tuple, and a call gives its closure one argument, the
   tuple, in place of them ([detuple]): the machine reads the fields of
   the tuple as Simplify's projections do ([tuple_fields]) and goes on
   with them as the arguments, followed by any others that the call
   gives. Every call of this fragment does so, a direct one too; the
   compiler's direct call of tupled code, which gives the code the
   components themselves, is not in the fragment.

   Exceptions. An exception handler is a continuation; a raise is an
   [Apply_cont] of it, the exception being the first argument; and a call
   names the continuation that its exceptions go to, with extra
   arguments, which the call evaluates: the callee raises to the caller's
   handler applied to the exception followed by them ([with_extra_args]).
   The two calls that an over-application makes, the full call and the
   call of its result, raise to that one continuation with those extra
   arguments.
   So where an exception goes is said by the term, as for any
   continuation, and the machine has no stack of handlers: a trap action
   ([Push], [Pop]) is not executed. The code that the backend makes does
   have such a stack and goes by it, not by the names. The front end
   builds terms in which the two cannot differ: a raise, and a call
   that raises, names the handler installed last.
   FlambdaWellFormedSpec.v has that discipline as a check of a term
   ([traps_well_formed]). FlambdaTrapSpec.v has a second machine, which
   goes by a stack of handlers as the backend's code does, and the
   statement that the two machines of the model behave alike on a unit
   that passes the check and whose binders are distinct. That the
   backend's code behaves as that second machine is not stated, and
   nothing proves it. There are no backtraces, so the kind of a raise
   changes nothing. The exception is any value: nothing here says what
   an exception is made of.

   A set of closures has functions, each the code of a function slot, and
   value slots, which the functions share. Each closure of the set is a
   value of its own: the set and the function slot the closure is at
   ([Value_closure_of_set]). The closures of one set differ in that slot
   only; a closure does not hold the others. Calling a closure runs the
   code of its slot with [my_closure] bound to that closure, so the
   functions of a set have each their own [my_closure], and reach one
   another by projecting a function slot from it
   ([project_function_slot]): that is how a set is recursive, and no value
   is cyclic. The closures of a static set are symbols, each bound to its
   closure with the symbols of the others ([Static_closure_of_set]);
   projecting a function slot from a symbol gives the symbol of that
   slot. A closure has no identity: two sets made alike are equal.

   The closures of a set of one function whose function slot is not named
   ([Value_closure], [Static_closure]) are those of the fragment before
   function slots. They are read as the closure at
   [unnamed_function_slot] of a set with that one function.

   These definitions can express sets of closures that the compiler does
   not make. In a set the compiler makes, every closure is at a function
   slot of the set, no function slot and no value slot is given twice,
   the functions have each their own code, and the symbols of a static
   set are bound together, each to the closure at its slot. What the
   machine does with the other sets is a choice, not a model of the
   compiler:
   - a closure at a function slot that is not among those of its set is
     not made: making one is undefined
     ([closure_of_set_made_statement],
     [static_closure_of_set_made_statement]). A value of that shape may
     still be in an environment given to a statement; it is not a closure
     ([closure_in_set]), so calling it or projecting from it is undefined;
   - where a function slot or a value slot is given twice, the first of
     the two is the one found, as everywhere in FlambdaMapSpec.v. The
     second is not removed: every value slot is evaluated where the set
     is made, so a second one that has no value makes that undefined,
     and the closure holds both lists as they were given;
   - a set in which two function slots have the same code is made, and
     each of its closures runs that code; so are two static sets of one
     group that have the same code;
   - a closure of code that is not bound is made, and calling it is
     undefined, as for the closures of the fragment before;
   - nothing checks that the symbols a static closure names as those of
     the closures of its set are bound to them. A projection from it
     gives the symbol named, bound or not, as a symbol is a value whether
     or not it is bound. FlambdaClosuresSpec.v says what holds of a set
     whose symbols are bound as [bind_closures] binds them
     ([static_set_bound]).
   A unit that makes only sets such as the compiler's is well-formed
   (FlambdaWellFormedSpec.sets_well_formed, a check, which does not look
   at value slots). The statements of this file are about every term and
   every environment, those that no well-formed unit reaches included;
   FlambdaClosuresSpec.v says what more holds of the run of a
   well-formed unit.

   Execution is undefined ([Undefined]) at [Invalid], at an unbound variable,
   at a continuation applied to the wrong number of arguments, at a
   function called with as many arguments as it has parameters at
   arities that are not those of its code, or with fewer or more
   arguments at arities that the compiler does not take for a partial
   application or an over-application, or with arguments that are not
   of the kinds of the call's arity, at a call of a partial application
   that is direct or gives it no argument, at a call of tupled code
   whose first argument is not a tuple with one field of kind [Value]
   for each parameter of the code,
   at a [Let] whose defining expression has a value of another kind than
   the [Let] gives its variable, at a handler applied to values of other
   kinds than it gives its parameters,
   at a [Switch] whose scrutinee is not a naked immediate with
   an arm, where a primitive has no value, at a call of what is not a
   closure, of a closure whose code is not bound, or directly of a closure
   of other code than the one named, at the projection of a value slot
   from what is not a closure with that slot, at the projection of a
   value slot or of a function slot that names the function slot it
   projects from, from what is not a closure at that function slot of a
   set with the slot asked for, and where a closure is made, by a [Let] or
   statically, at a function slot that is not among those of its set.
   A transformation may do anything to a term whose
   execution is undefined: that is [refines], the statement of correctness
   to use for a transformation; [equivalent] is the stronger, symmetric one.
   [refines_defined_statement] and [refines_antisymmetric_statement] say how
   the two are related.

   Kinds. The kinds and arities that terms carry (FlambdaModel.v) are
   tested where a variable is bound, and nowhere else. A call
   ([call_function]) is undefined unless it agrees with the code called,
   in the three ways given at [call_code] for a full call and in those
   given at [shape_of_closure_call] for the others. Simplify assumes
   each of them, and a well-typed program whose types are refined by a
   GADT can break the second and the third in a branch that is never
   run. A [Let] is
   undefined unless the value of its defining expression is of its kind
   ([step]), and the application of a handler unless the values are of
   the kinds of its parameters ([apply_cont_value]). So, in every run, a
   variable is bound only to a value of the kind that its binder gives
   (FlambdaWellKindedSpec.v has the statements). The CR at [step] says
   why the last two are less certain than the first.

   The primitives are those of FlambdaPrimitiveModel.v, and
   [FlambdaPrimitiveSpec.eval_prim] says what each computes. A law of a
   primitive (say, that an addition commutes) is a lemma about it.

   The lemmas and the proofs are in FlambdaSemanticsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec.
From OxCamlFlambda2 Require Export FlambdaValueSpec FlambdaPrimitiveSpec
  FlambdaHeapSpec.
Import ListNotations.

(* ======================================================================== *)
(* Environments                                                             *)
(* ======================================================================== *)

(* An environment is a finite map of FlambdaMapSpec.v: an association
   list, of which [lookup] finds the first binding of a name. *)
Definition env : Type := list (variable * value).

(* What a continuation is. [Closure] is a continuation handler with the
   environments of its definition. For a handler of a recursive
   [Let_cont], [recursive] is all the handlers of that [Let_cont], which
   are in scope in the handler; it is empty for a non-recursive one.
   [Outer] is a way out of the whole term, by the continuation named.
   [Extra_args] is a continuation that is to be given, after the arguments
   it is applied to, the values kept here: it is what a call with extra
   arguments for its exception continuation gives its callee to raise
   to.
   [Over_application] is where the full call inside a call with more
   arguments than the function has parameters returns to: its result is
   to be called with the arguments that remain ([remaining], given at
   [remaining_arity]), for a result of [return_arity], with the two
   continuations of the whole call. *)
#[local] Set Warnings "-register-all".
Inductive cont_value : Type :=
  | Closure (h : continuation_handler) (vars : env)
      (conts : list (continuation * cont_value))
      (recursive : list (continuation * continuation_handler))
  | Outer (k : continuation)
  | Extra_args (c : cont_value) (extra : list value)
  | Over_application (remaining : list value)
      (remaining_arity return_arity : arity) (return_to exn_to : cont_value).

Definition cont_env : Type := list (continuation * cont_value).

(* What a continuation name stands for: a continuation with no binding is
   a way out of the term. *)
Definition resolve (k : continuation) (conts : cont_env) : cont_value :=
  match lookup k conts with
  | Some c => c
  | None => Outer k
  end.

(* The closures of the handlers of a recursive [Let_cont] defined under
   [vars] and [conts]. *)
Definition rec_closures
    (handlers : list (continuation * continuation_handler)) (vars : env)
    (conts : cont_env) : cont_env :=
  map (fun '(k, h) => (k, Closure h vars conts handlers)) handlers.

(* ======================================================================== *)
(* Evaluation                                                               *)
(* ======================================================================== *)

Definition eval_simple (vars : env) (s : simple) : option value :=
  match s with
  | Var x => lookup x vars
  | Symbol sym => Some (Value_symbol sym)
  | Const c => Some (Value_const c)
  end.

Fixpoint eval_simples (vars : env) (args : list simple)
    : option (list value) :=
  match args with
  | [] => Some []
  | s :: args =>
    let* v := eval_simple vars s in
    let* vs := eval_simples vars args in
    Some (v :: vs)
  end.

Fixpoint eval_value_slots (vars : env)
    (value_slots : list (value_slot * simple))
    : option (list (value_slot * value)) :=
  match value_slots with
  | [] => Some []
  | (slot, s) :: value_slots =>
    let* v := eval_simple vars s in
    let* vs := eval_value_slots vars value_slots in
    Some ((slot, v) :: vs)
  end.

(* [closure_of] and [closure_in_set] see the same closures: the code of a
   closure is the code of the function slot it is at. *)
Definition closure_in_set_of_statement : Prop :=
  ∀ (statics : static_env) (v : value) (code : code_id)
    (value_slots : list (value_slot * value)),
    closure_of statics v = Some (code, value_slots)
    <-> ∃ (slot : function_slot)
          (function_decls : list (function_slot * code_id)),
          closure_in_set statics v = Some (slot, function_decls, value_slots)
          ∧ lookup slot function_decls = Some code.

(* A closure is at one of the function slots of its set. *)
Definition closure_in_set_slot_statement : Prop :=
  ∀ (statics : static_env) (v : value) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)),
    closure_in_set statics v = Some (slot, function_decls, value_slots) ->
    is_bound slot function_decls = true.

(* A value slot is projected only from a closure at [project_from], and is
   then the value of the slot. *)
Definition project_value_slot_defined_statement : Prop :=
  ∀ (statics : static_env) (project_from : function_slot)
    (slot : value_slot) (v w : value),
    project_value_slot statics project_from slot v = Some w
    <-> ∃ (function_decls : list (function_slot * code_id))
          (value_slots : list (value_slot * value)),
          closure_in_set statics v
          = Some (project_from, function_decls, value_slots)
          ∧ lookup slot value_slots = Some w.

(* A function slot is projected only from a closure at [move_from] of a
   set that has [move_to]. These are necessary conditions: from a symbol
   the projection also needs the set to name a symbol for [move_to]. *)
Definition project_function_slot_defined_statement : Prop :=
  ∀ (statics : static_env) (move_from move_to : function_slot)
    (v w : value),
    project_function_slot statics move_from move_to v = Some w ->
    ∃ (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * value)) (code code' : code_id),
      closure_in_set statics v = Some (move_from, function_decls, value_slots)
      ∧ lookup move_from function_decls = Some code
      ∧ lookup move_to function_decls = Some code'.

(* From a closure made by a [Let] it gives the closure of the same set at
   the other slot. *)
Definition project_function_slot_of_set_statement : Prop :=
  ∀ (statics : static_env) (move_from move_to : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * value)) (code code' : code_id),
    lookup move_from function_decls = Some code ->
    lookup move_to function_decls = Some code' ->
    project_function_slot statics move_from move_to
      (Value_closure_of_set move_from function_decls value_slots)
    = Some (Value_closure_of_set move_to function_decls value_slots).

(* CR mshinwell: Making a closure at a function slot that is not among
   those of its set is undefined: the first line of the case
   [Closure_of_set] here, and of [Static_const_closure_of_set] in
   [eval_static_const]. No term of the compiler makes one, and
   FlambdaWellFormedSpec.well_formed_unit refuses a unit that would. The
   other choice is to make the value, which nothing then reads as a
   closure ([closure_of] and [closure_in_set] have no value on it), and
   to leave it to well-formedness. Decision D20. *)

(* CR mshinwell: the projection of a value slot from what is not a closure,
   or from a closure that has not got the slot, is undefined (the cases
   [Project_value_slot] and [Project_value_slot_from] of [eval_named]).
   Simplify makes such a projection [Invalid] where the type of the
   closure shows it. Not certain: whether a term that the compiler builds
   can project, in code that runs, a slot that the closure has not got.
   Decision D3, which is about function slots; this is the same question
   for value slots. *)
Definition eval_named (statics : static_env) (vars : env)
    (defining_expr : named) : option value :=
  match defining_expr with
  | Simple s => eval_simple vars s
  | Prim p args =>
    let* vs := eval_simples vars args in
    eval_prim statics p vs
  | Set_of_closures code value_slots =>
    let* vs := eval_value_slots vars value_slots in
    Some (Value_closure code vs)
  | Closure_of_set slot function_decls value_slots =>
    let* _ := lookup slot function_decls in
    let* vs := eval_value_slots vars value_slots in
    Some (Value_closure_of_set slot function_decls vs)
  | Project_value_slot slot closure =>
    let* v := eval_simple vars closure in
    let* called := closure_of statics v in
    lookup slot (snd called)
  | Project_value_slot_from project_from slot closure =>
    let* v := eval_simple vars closure in
    project_value_slot statics project_from slot v
  | Project_function_slot move_from move_to closure =>
    let* v := eval_simple vars closure in
    project_function_slot statics move_from move_to v
  end.

(* What a named expression does on a heap: its value, and the heap
   after. A primitive is run on the values of its arguments by
   FlambdaHeapSpec.exec_prim: it may read the heap, and a primitive of
   the heap changes it. Anything else has the value that [eval_named]
   gives it, and the heap is as it came. *)
Definition exec_named (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env) (defining_expr : named)
    : option (value * heap * read_record) :=
  match defining_expr with
  | Prim p args =>
    let* vs := eval_simples vars args in
    let* '(v, hp') := exec_prim statics hp p vs in
    Some (v, hp', rd)
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    let* v := eval_named statics vars defining_expr in
    Some (v, hp, rd)
  end.

(* CR mshinwell: As for [Closure_of_set] in [eval_named]: the constant of
   a closure at a function slot that is not among those of its set has
   no value (the first line of the case [Static_const_closure_of_set]),
   so the group that binds it is undefined. Decision D20. *)
Definition eval_static_const (vars : env) (const : static_const)
    : option statically_allocated_value :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    let* vs := eval_value_slots vars value_slots in
    Some (Static_closure code vs)
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    let* _ := lookup slot function_decls in
    let* vs := eval_value_slots vars value_slots in
    Some (Static_closure_of_set slot closure_symbols function_decls vs)
  | Static_const_block tag fields =>
    let* vs := eval_simples vars fields in
    Some (Static_block tag vs)
  | Static_const_boxed_number contents =>
    let* v := eval_simple vars contents in
    Some (Static_boxed_number v)
  | Static_const_immutable_string bytes => Some (Static_string bytes)
  | Static_const_immutable_value_array elements =>
    let* vs := eval_simples vars elements in
    Some (Static_immutable_array vs)
  end.

(* What is made at a function slot of a set, by a [Let] or statically, is
   a closure at that slot of that set: a closure at a slot that its set
   has not got is not made. *)
Definition closure_of_set_made_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (slot : function_slot)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (v : value),
    eval_named statics vars (Closure_of_set slot function_decls value_slots)
    = Some v ->
    ∃ slot_values : list (value_slot * value),
      eval_value_slots vars value_slots = Some slot_values
      ∧ closure_in_set statics v = Some (slot, function_decls, slot_values).

Definition static_closure_of_set_made_statement : Prop :=
  ∀ (statics : static_env) (vars : env) (slot : function_slot)
    (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (s : symbol)
    (allocated : statically_allocated_value),
    eval_static_const vars
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots)
    = Some allocated ->
    lookup s (static_values statics) = Some allocated ->
    ∃ slot_values : list (value_slot * value),
      eval_value_slots vars value_slots = Some slot_values
      ∧ closure_in_set statics (Value_symbol s)
        = Some (slot, function_decls, slot_values).

(* CR mshinwell: binding a code id or a symbol that is bound already is
   undefined (the two cases of [eval_static_bindings] that find the name),
   so that what a name is bound to never changes. A well-formed unit never
   does it (FlambdaWellFormedSpec.v). Not certain that every unit the
   compiler builds is such: nothing there checks, across a unit, that a
   symbol or a code id is bound once. Discrepancy C0-4; no decision is
   asked yet. *)

(* The static environment with a group of bindings added. Deleted code is
   not bound. A code id or a symbol is bound once: binding one that is
   bound already, by this group or by an earlier one, is undefined. So
   what a name is bound to never changes (FlambdaWellFormedSpec.v). *)
Fixpoint eval_static_bindings (vars : env) (bindings : list static_binding)
    (statics : static_env) : option static_env :=
  match bindings with
  | [] => Some statics
  | Bind_code id c :: bindings =>
    match lookup id (static_code statics) with
    | Some _ => None
    | None =>
      eval_static_bindings vars bindings
        (Static_env ((id, c) :: static_code statics) (static_values statics))
    end
  | Bind_deleted_code _ :: bindings =>
    eval_static_bindings vars bindings statics
  | Bind_symbol s const :: bindings =>
    match lookup s (static_values statics) with
    | Some _ => None
    | None =>
      let* allocated := eval_static_const vars const in
      eval_static_bindings vars bindings
        (Static_env (static_code statics)
           ((s, allocated) :: static_values statics))
    end
  end.

(* CR mshinwell: a call of code whose body binds something static is
   undefined, and so is a recursive [Let_cont] one of whose handlers does
   ([call_code] and [step], by [binds_statics]). For a binding of a
   symbol this follows the compiler, which stops on one that is not at
   toplevel. For a group that binds only code the compiler has no such
   check (simplify/simplify_named.ml, the case [Static_consts]). So if a
   pass leaves code bound inside a function or a loop, the model calls
   undefined a term that the compiler runs. Discrepancy C0-3; no decision
   yet. *)

(* Whether an expression has a [Let_static] that binds something, in its
   body or in a handler. The code that a [Let_static] binds is not looked
   into: it is checked when it is called. *)
Definition binding_binds (binding : static_binding) : bool :=
  match binding with
  | Bind_deleted_code _ => false
  | Bind_code _ _ | Bind_symbol _ _ => true
  end.

Fixpoint binds_statics (e : expr) : bool :=
  match e with
  | Let _ _ _ body => binds_statics body
  | Let_static bindings body =>
    existsb binding_binds bindings || binds_statics body
  | Let_cont_non_recursive _ h body =>
    handler_binds_statics h || binds_statics body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_binds_statics h) handlers
    || binds_statics body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_binds_statics (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => binds_statics handler
  end.

Definition handlers_bind_statics
    (handlers : list (continuation * continuation_handler)) : bool :=
  existsb (fun '(_, h) => handler_binds_statics h) handlers.

(* Binds parameters to arguments; there must be as many of each. *)
Fixpoint bind_params (params : list variable) (vs : list value) (vars : env)
    : option env :=
  match params, vs with
  | [], [] => Some vars
  | x :: params, v :: vs => bind_params params vs ((x, v) :: vars)
  | _, _ => None
  end.

Fixpoint find_arm (i : Z) (arms : list (Z * apply_cont)) : option apply_cont :=
  match arms with
  | [] => None
  | (discriminant, ac) :: arms =>
    if Z.eqb i discriminant then Some ac else find_arm i arms
  end.

(* What a [Switch] may be on. *)
Definition naked_immediate_of_value (v : value) : option Z :=
  match v with
  | Value_const (Naked_immediate i) => Some i
  | _ => None
  end.

(* ======================================================================== *)
(* Steps                                                                    *)
(* ======================================================================== *)

(* A state of the machine: the expression to run, the values of its
   variables, what its continuations stand for, and the static
   environment. It is a record, so that a property of a state can name the
   components it is about ([state_statics s]) and say nothing of the
   others. *)
Record state : Type := State {
  state_expr : expr;
  state_vars : env;
  state_conts : cont_env;
  state_statics : static_env;
  state_heap : heap;
  state_reads : read_record
}.

(* What one step of a state, or a bounded run, comes to: another state, the
   application of an unbound continuation, or undefined behaviour. *)
Inductive step_result : Type :=
  | Step (s : state)
  | Exit (k : continuation) (vs : list value) (hp : heap)
      (rd : read_record)
  | Undefined.

(* The exception continuation that a call gives its callee: the caller's,
   to be applied to the exception followed by the extra arguments of the
   call. With no extra argument it is the caller's as it is. *)
(* CR mshinwell: Extra arguments given to a way out of the term are not
   undefined: the exit has the exception followed by them, as a way out
   takes any number of values, whatever applies it. The exception
   continuation of a unit, or of code, takes the exception alone, and
   Simplify stops on a call that gives it more. Well-formedness refuses
   such a call ([FlambdaWellFormedSpec.ways_out_well_formed]); the
   machine has no test, as a way out has no arity to test against.
   Discrepancy C5-5. *)
Definition with_extra_args (c : cont_value) (extra : list value)
    : cont_value :=
  match extra with
  | [] => c
  | _ :: _ => Extra_args c extra
  end.

(* A full call of a closure of the code [c] runs the body of the code.
   The only variables are the parameters, bound to the arguments, and
   [my_closure], bound to the closure; the only continuations are the
   return and exception continuations of the code, which stand for what
   the caller's stand for. So there is no stack: to return is to apply
   the continuation the caller gave. The body must bind nothing static:
   it would do so at each call.

   The call must agree with the code, in three ways. Simplify assumes
   each: it replaces by [Invalid] a call that fails one
   (simplify/simplify_apply_expr.ml).
   - The values of the arguments are of the kinds that the call's own
     arity gives, unarized ([simplify_apply_shared], which compares the
     kinds of the arguments with [Flambda_arity.unarize] of
     [Apply.args_arity]; the reason is
     [Application_argument_kind_mismatch]). The machine looks at the
     kind of a value here, at a [Let] and where a handler is applied
     ([step], [apply_cont_value]). A poison constant has the kind it
     carries, and an allocated value or a symbol, bound or not, has kind
     [Value]: so poison of the right kind passes, and that is
     intended.
   - The arity of the arguments is that of the parameters of the code,
     each parameter unarized ([arity_mismatch], on
     [Flambda_arity.unarize_per_parameter];
     [Direct_application_parameter_kind_mismatch]). One parameter that
     is an unboxed pair is not two parameters; how products are nested
     inside one parameter does not count.
   - The arity of the results that the call expects is that of the
     results of the code, unboxed products and all
     ([simplify_direct_function_call], by
     [Flambda_arity.equal_ignoring_subkinds];
     [Application_result_kind_mismatch]). *)
Definition call_code (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) : step_result :=
  if binds_statics (code_body c) then Undefined
  else if
    negb
      (kinds_eqb (map kind_of_value vs) (unarize args_arity)
       && kinds_per_parameter_eqb (unarize_per_parameter args_arity)
            (unarize_per_parameter (code_params_arity c))
       && arity_eqb return_arity (code_result_arity c))
  then Undefined
  else
    match bind_params (code_params c) vs [(code_my_closure c, closure)] with
    | Some callee_vars =>
      Step (State (code_body c) callee_vars
        [(code_return_continuation c, return_to);
         (code_exn_continuation c, exn_to)]
        statics hp rd)
    | None => Undefined
    end.

(* What a call is, before anything runs: a full call of code; a full
   call with the first arguments, whose result is then called with the
   arguments that remain; or a partial application, the value that
   stands for a closure given some of its arguments. *)
Inductive call_shape : Type :=
  | Full_call (closure : value) (c : code) (vs : list value)
      (args_arity : arity)
  | Over_call (closure : value) (c : code) (vs remaining : list value)
      (args_arity remaining_arity : arity)
  | Partial_call (partial : value).

(* The access kind at which Simplify reads the fields of the tuple that
   a call gives tupled code ([Simplify_common.project_tuple]): a block
   of values of tag 0 with as many fields as the code has parameters. *)
Definition tuple_access (size : nat) : block_access_kind :=
  Block_access_values (Some 0%Z) (Some (Z.of_nat size)) Any_value.

(* The fields [field], [field + 1], ... of a tuple of [size]
   components, [count] of them: what the projections that
   [simplify_direct_tuple_application] makes read, each an immutable
   [Block_load] at [tuple_access]. Nothing if one of them has no
   value. *)
Fixpoint tuple_fields_from (statics : static_env) (size : nat)
    (tuple : value) (field count : nat) : option (list value) :=
  match count with
  | O => Some []
  | S count =>
    let* v :=
      eval_prim statics
        (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
        [tuple]
    in
    let* vs := tuple_fields_from statics size tuple (S field) count in
    Some (v :: vs)
  end.

(* The fields of a tuple of [size] components. So the tuple is an
   immutable block of tag 0 that has exactly [size] fields, all of kind
   [Value], or a symbol bound to one
   (FlambdaTupledSpec.tuple_is_block_statement). A tuple of no component
   is not looked at. *)
(* CR mshinwell: a tuple given to tupled code must have the tag 0 and
   exactly as many fields as the code has parameters, each of kind
   [Value]: otherwise the call is undefined. That is what the access
   kind of Simplify's projections says, as this fragment reads an
   access kind (the review comment at
   FlambdaPrimitiveSpec.eval_block_load). At run time [caml_tuplifyN]
   (backend/cmm_helpers.ml, [tuplify_function]) reads the first fields
   of whatever it is given and tests nothing. It would be wrong if the
   compiler gives a tupled function a block of another tag or with more
   fields. Decisions D15 and D22. *)
Definition tuple_fields (statics : static_env) (size : nat) (tuple : value)
    : option (list value) :=
  tuple_fields_from statics size tuple 0 size.

(* The arguments of a call of tupled code as the code takes them, with
   their arity: the fields of the first argument, the tuple, then the
   arguments that remain. [size] is the number of parameters of the
   code, unarized. The arguments must be of the kinds of the arity of
   the call, as for every call, and the first parameter of that arity
   must be one [Value]. The arity that results is that of
   [simplify_direct_tuple_application]: [size] values, then the arity of
   the call less its first parameter. With no argument the call is
   undefined: the compiler stops on one. *)
Definition detuple (statics : static_env) (size : nat) (vs : list value)
    (args_arity : arity) : option (list value * arity) :=
  match vs, args_arity with
  | tuple :: rest, first :: rest_arity =>
    if
      negb
        (kinds_eqb (map kind_of_value vs) (unarize args_arity)
         && kinds_eqb (unarize_component first) [Value])
    then None
    else
      let* fields := tuple_fields statics size tuple in
      Some (fields ++ rest, arity_of_kinds (repeat Value size) ++ rest_arity)
  | _, _ => None
  end.

(* The arguments of a call of a closure of the code [c] as the code
   takes them, with their arity: as the call gives them, unless the code
   is tupled. *)
Definition arguments_for_code (statics : static_env) (c : code)
    (vs : list value) (args_arity : arity) : option (list value * arity) :=
  if code_is_tupled c
  then detuple statics (cardinal_unarized (code_params_arity c)) vs args_arity
  else Some (vs, args_arity).

(* What the call of a closure of the code [c] is, given the arguments
   as the code takes them; [None] where it is undefined.

   A call is the generic application of the compiler, which tells the
   three shapes apart by the number of arguments, counted by parameter
   and not unarized ([simplify_direct_function_call]; at run time, for a
   call of unknown arity, [caml_applyN]). With as many arguments as the
   code has parameters it is a full call, and [call_code] makes the
   three tests. Otherwise the arguments must be of the kinds of the
   arity that the call is made at, and the arities must pass the test by
   which the compiler takes the call for a partial application or an
   over-application (FlambdaModel.partial_or_over_application).
   - With more arguments than parameters it is a full call with the
     first of them, at the result arity of the code, which is one
     [Value]; the result is then called with the rest.
   - With fewer, but not none, it is a partial application of the
     closure to the arguments, and nothing is called.
   What is not tested: at a partial application, nothing is compared
   with the result arity of the code; at an over-application, nothing
   with the return arity of the call, until the result is called. A call
   with no argument of code that has parameters is undefined: the
   compiler stops on one. *)
(* CR mshinwell: fewer arguments than parameters, of tupled code, is
   undefined. A partial application keeps the closure and the
   arguments, and is called later as its closure is with all the
   arguments ([shape_of_call]); for a closure of tupled code that call
   would take the first of them for a tuple. It cannot come about in
   this fragment for code whose parameters are single values, as the
   compiler's tupled code is: a tuple has a field for each. Simplify
   would make a wrapper of a direct call of tupled code with too few
   arguments ([simplify_direct_partial_application]); the compiler is
   not known to make such a call. *)
Definition shape_of_code_call (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) : option call_shape :=
  let provided := length args_arity in
  let params := length (code_params_arity c) in
  if Nat.eqb provided params
  then Some (Full_call closure c vs args_arity)
  else if
    negb
      (kinds_eqb (map kind_of_value vs) (unarize args_arity)
       && partial_or_over_application args_arity return_arity
            (code_params_arity c) (code_result_arity c))
  then None
  else if Nat.ltb params provided
  then
    let size := cardinal_unarized (code_params_arity c) in
    Some
      (Over_call closure c (firstn size vs) (skipn size vs)
         (firstn params args_arity) (partially_apply args_arity params))
  else if code_is_tupled c
  then None
  else Some (Partial_call (Value_partial_application closure vs args_arity)).

(* Whether the code ids that a call of known arity lists allow the code
   id [called]: a list allows its members, and no list allows every
   code ([Or_unknown.Unknown]). The empty list allows none. *)
(* CR mshinwell: the code ids of a call of known arity. The compiler
   says that "only the provided functions (or a simplified version
   thereof) can be called" (terms/call_kind.mli). Here the call is
   undefined unless the code id of the callee is one of those listed.
   That is a choice of the model and stricter than the compiler: the
   model has not got what code is a newer version of (the code age
   relation; discrepancy C0-1). The compiler relies on the list.
   Simplify meets it, in the code age relation, with the code id that
   the type of the callee gives, keeps what is related or of unknown
   relation, and makes the call [Invalid] if nothing is left
   (simplify/simplify_apply_expr.ml, [simplify_direct_function_call];
   types/env/code_age_relation.ml, [meet_set]); the zero_alloc
   checker takes the list for the callees (backend/cmm.mli,
   [Capply]). Nothing tests at run time that the callee is one of
   them. Asked of the developers: is the list a promise that every
   pass must keep true, as it is read here, and is "a simplified
   version" any newer version in the code age relation?
   Decision D59. *)
Definition code_ids_allow (code_ids : option (list code_id))
    (called : code_id) : bool :=
  match code_ids with
  | Some ids => among called ids
  | None => true
  end.

(* The tests of a call of known arity on its arity and on the code of
   its callee: the call has one group of arguments at least; the code
   is bound, is not tupled, and has as many parameters as the call has
   groups of arguments. Arguments and parameters are counted by group,
   as the generic application counts them ([shape_of_code_call]), and
   not by the values that they unarize to: one group may stand for no
   value. *)
Definition known_arity_full (statics : static_env) (called : code_id)
    (args_arity : arity) : bool :=
  match args_arity, lookup called (static_code statics) with
  | _ :: _, Some c =>
    negb (code_is_tupled c)
    && Nat.eqb (length args_arity) (length (code_params_arity c))
  | _, _ => false
  end.

(* What the call of a closure is; [None] where it is undefined. A direct
   call must be to a closure of the code named. A call of known arity
   must be to a closure of code that its list allows ([code_ids_allow])
   and must pass the tests of [known_arity_full]; it is then what the
   call of unknown arity of the same closure is, which is a full call
   (FlambdaKnownAritySpec.known_arity_shape_statement). The code takes
   the arguments as they are, unless it is tupled: then the first
   argument is a tuple, and the code takes its fields followed by the
   other arguments ([arguments_for_code]). That is the rewrite of
   [simplify_direct_tuple_application], made by the machine: the
   compiler gives a tupled function a tuple wherever the call does not
   go by the code (a call of unknown arity), and at run time the
   closure's own entry point takes the tuple apart ([caml_tuplifyN]). *)

(* CR mshinwell: two calls are undefined in [shape_of_closure_call] on
   trust.
   - A call of a closure whose code is not bound, or is deleted (no code
     is found), while the closure itself is made, by every form of
     closure. It cannot be undefined where the closure is made, as a
     group may bind a symbol to a closure before it binds the code. It
     is right if code is deleted only when it is never called, which
     nothing checks; for the term after Simplify it is the safe side. It
     is not decided whether well-formedness is to ask that the code of
     every closure is bound by the time it is called, as it is to ask of
     the symbols a term mentions (decision D8). Decision D20.
   - A direct call of a closure of other code than the one named. The
     backend would run the code named, with that closure. Not certain
     where the two code ids are versions of one function: the compiler
     relates them by its code age relation, which the model has not got.
     Discrepancy C0-1; no decision yet. *)

(* CR mshinwell: a direct call is a generic application here: with
   fewer or more arguments than the code has parameters it is a partial
   application or an over-application. That is how Simplify treats a
   call of any kind once it knows the callee ([simplify_function_call]
   sends all three to [simplify_direct_function_call], which goes by
   the number of arguments, subject to its tests of kinds and to the
   case where it makes no stub for a partial application: the comment
   on calls of known arity below), and it is what makes a call of unknown
   arity of a known closure the same as a direct call of its code
   whatever the number of arguments
   (FlambdaApplySpec.apply_indirect_known_statement). The backend reads
   a direct call as a full call: it stops on one that has not as many
   arguments as the code has parameters (to_cmm/to_cmm_expr.ml, "Wrong
   arity for direct call"). So the machine gives a meaning to direct
   calls that the backend refuses. No term that the compiler makes is
   known to have one: closure conversion makes a call direct only where
   it is full (from_lambda/closure_conversion.ml, [close_apply]). A
   call of known arity, which the backend also reads as a full call, is
   one here too (the next comment; decision D58). Whether a direct call
   that is not full should be undefined as well, or ruled out by
   well-formedness where the code is named, is not decided.
   Discrepancy C3-9, which is now of direct calls alone. *)
(* CR mshinwell: a call of known arity
   ([Call_kind.Function_call.Indirect_known_arity]) has a meaning here
   only where it is a full call: it has one group of arguments at
   least, and its callee is a closure of code that is bound, is not
   tupled, and has as many parameters as the call has groups of
   arguments ([code_ids_allow] and [known_arity_full] above). It is
   then the call of unknown arity of the same callee, which is that
   full call (FlambdaKnownAritySpec.known_arity_shape_statement).
   Every other call of known arity is undefined: with no group of
   arguments; of a partial application made at run time by a call of
   unknown arity; with fewer or more groups of arguments than the
   code has parameters, also where the call has one group; of tupled
   code.
   What the compiler does. Simplify makes a call of known arity in
   two places: from a call of unknown arity, where the type of the
   callee gives candidate code that is not tupled and has the arity
   of the call ([simplify_function_call]); and from a direct call
   whose code id it gives up
   ([simplify_function_call_where_callee's_type_unavailable]), which
   is a full call only if the direct call was. There is no general
   check that a call of known arity keeps the promise of its kind:
   Simplify tests what it learns of the callee, and may treat a
   mismatch of the number of groups as a partial application or an
   over-application (below); the backend emits no test at run time
   of the callee. The backend loads a code pointer from the closure,
   the first field for one group of arguments and the third for
   more, and calls it with the arguments given; it stops on a call
   with no group
   (to_cmm/to_cmm_expr.ml, the case [Indirect_known_arity];
   backend/cmm_helpers.ml, [indirect_full_call]). With two groups or
   more: a closure of code that is not tupled and has one parameter
   or none has no third field with a code pointer; of a closure of
   other code that has not the number of parameters of the call,
   the third field is the code of the full application, which is
   then given another number of arguments than it takes. With one
   group the backend's call is that of the generic application; it
   is undefined here where the code has not one parameter, as the
   kind promises the arity and the reaper goes by that promise
   (identifiers/field.mli,
   [Known_arity_code_pointer]; to_cmm/to_cmm_set_of_closures.ml puts
   a function that fails in the first field where a set is marked
   [only_full_applications]). Simplify, once it knows the callee,
   treats a call of any kind by the number of its arguments
   ([simplify_direct_function_call]): with fewer it may make a
   closure, and with more it may split the call. Each has a test of
   its own, without which the call is made [Invalid] or the compiler
   stops: with more arguments, that the code of the callee returns
   one value, which the arguments that remain are then given to;
   with fewer, that the arity which the application itself gives for
   its result is one value. Where no stub is to be made for a
   partial application it treats the call as one whose callee is not
   known; and it stops on a call with no argument of code that has
   parameters.
   Where the model is stricter than the compiler, by choice. Three
   sorts of call of known arity are undefined here although the
   backend's call has the result of the generic application: of a
   partial application made at run time, given exactly the parameters
   that are left (it is a closure of what is left to apply; the
   compiler is not known to make such a call, and the value has no
   code id here); with one group of arguments, of code that is not
   tupled and has more parameters, which goes by the first field of
   the closure and returns a partial application; and with one group,
   of tupled code given its tuple, which goes by the same field to
   the function that takes the tuple apart. The last two are so only
   where that field has not been given the function that fails (a
   set marked [only_full_applications], above). Nothing of the kind
   is said of another mismatch: with two groups or more the call
   goes by a field that may hold no code, or gives code another
   number of arguments than it takes (above). Tupled code otherwise:
   Simplify takes a call of known arity, like a direct call, to give
   tupled code its components ([simplify_function_call],
   [call_must_be_detupled]), and the backend's call with two groups
   or more loads the code that takes them. Every call of known arity
   of tupled code is undefined here, so that every call of known
   arity that has a step is the call of unknown arity of the same
   callee, and the test tool refuses a unit that has a call of known
   arity and binds tupled code (the next comment).
   Asked of the developers: is a call of known arity whose callee
   has not the arity of the call a broken promise, which a check
   could catch under [-flambda2-invariants] or in [to_cmm]; or has
   it the meaning of the generic application, which the backend
   would then have to keep? A direct call is still a generic
   application here (discrepancy C3-9, which is now of direct calls
   alone).
   Decision D58. *)
(* CR mshinwell: a direct call of tupled code gives it a tuple here, as
   a call of unknown arity does. In the compiler a direct call and a
   call of known arity are in the convention of the code: Simplify
   takes them to give tupled code the components
   ([simplify_function_call], the comment at [call_must_be_detupled]).
   So the compiler's direct calls and calls of known arity of tupled
   code are not in this fragment: a call of known arity of tupled code
   is undefined (the comment above), the test tool refuses both, and
   the [Direct] of this fragment says of tupled code what the compiler
   has no call kind for: the code is known and the argument is still
   the tuple. It is so in order that a call of unknown arity of a known
   closure is a direct call of its code whatever the code
   (FlambdaApplySpec.apply_indirect_known_statement), which the
   canonicalisation of a term by aliases relies on under every static
   environment. The other choice is a [Direct] in the convention of the
   code, as in the compiler, and a premise that the code is not tupled
   on that statement and on what is built on it. *)
Definition shape_of_closure_call (statics : static_env) (closure : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    : option call_shape :=
  match closure_of statics closure with
  | Some (called, _) =>
    let expected :=
      match kind with
      | Direct named => Pos.eqb called named
      | Indirect_unknown_arity => true
      | Indirect_known_arity code_ids =>
        (code_ids_allow code_ids called
         && known_arity_full statics called args_arity)%bool
      end
    in
    match expected, lookup called (static_code statics) with
    | true, Some c =>
      let* '(vs', args_arity') := arguments_for_code statics c vs args_arity in
      shape_of_code_call closure c vs' args_arity' return_arity
    | _, _ => None
    end
  | None => None
  end.

(* What a call is. A call of unknown arity of a partial application is
   the call of its closure with the arguments that it has been given
   already, followed by those of the call, of which there must be one at
   least: the compiler makes a partial application a closure of code
   with the parameters that are left, and stops on a call of it with no
   argument. The arguments of the call must be of the kinds of the arity
   that the call is made at, and that is tested here, of the call's own
   arguments, as Simplify tests it of every call before it goes by the
   code of the callee ([simplify_apply_shared]; where the kinds
   disagree, [Application_argument_kind_mismatch]). The call of the
   closure tests the arguments kept and the new ones together, and in
   that test a partial application that no call made, one that keeps
   arguments that are not of the kinds of the arity it keeps, could
   hide a mismatch. A direct call of a partial application is
   undefined: it is a closure of no code of the term. So is a call of
   known arity (decision D58). *)
Definition shape_of_call (statics : static_env) (callee : value)
    (kind : call_kind) (vs : list value) (args_arity return_arity : arity)
    : option call_shape :=
  match kind, partial_application_of callee with
  | Indirect_unknown_arity, Some (closure, applied, applied_arity) =>
    match args_arity with
    | [] => None
    | _ :: _ =>
      if kinds_eqb (map kind_of_value vs) (unarize args_arity)
      then
        shape_of_closure_call statics closure Indirect_unknown_arity
          (applied ++ vs) (applied_arity ++ args_arity) return_arity
      else None
    end
  | _, _ => shape_of_closure_call statics callee kind vs args_arity return_arity
  end.

(* A call of a given shape runs. A full call is made at the return arity
   of the call and returns to its return continuation. The full call
   inside an over-application is made at the result arity of the code,
   and returns to what is left of the application
   ([Over_application]); its exception continuation is that of the
   call. [return_partial] is what is done with a partial application: it
   is given to the return continuation of the call ([call_function]
   below). *)
Definition run_call (return_partial : value -> step_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env)
      (hp : heap) (rd : read_record) : step_result :=
  match shape with
  | Some (Full_call closure c vs args_arity) =>
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
  | Some (Over_call closure c vs remaining args_arity remaining_arity) =>
    call_code closure c vs args_arity (code_result_arity c)
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      exn_to statics hp rd
  | Some (Partial_call partial) => return_partial partial
  | None => Undefined
  end.

(* Applying a handler runs it under the environments of its definition,
   with its parameters bound and, for a recursive one, the handlers defined
   with it in scope again. The values must be of the kinds that the
   handler gives its parameters: otherwise there is no step. Applying a
   way out ends execution, whatever the values: a way out is the
   interface of the term, which says nothing of kinds. Applying a
   continuation that has extra arguments applies it to the arguments
   followed by those: so the extra arguments are tested with the others,
   by the handler that they reach. Applying what is left of an
   over-application, to the one value that the full call returned, is
   an indirect call of that value with the arguments that remain; its
   exception continuation is that of the over-application, extra
   arguments included. No variable is bound there, and the kind of the
   value is not tested: a value that can be called is of kind
   [Value]. *)
Fixpoint apply_cont_value (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record)
      {struct c} : step_result :=
  match c with
  | Outer k => Exit k vs hp rd
  | Closure (Handler params metadata handler) def_vars def_conts recursive =>
    if kinds_eqb (map kind_of_value vs) (hm_params_kinds metadata)
    then
      match bind_params params vs def_vars with
      | Some handler_vars =>
        Step (State handler handler_vars
          (rec_closures recursive def_vars def_conts ++ def_conts) statics hp
            rd)
      | None => Undefined
      end
    else Undefined
  | Extra_args c extra => apply_cont_value c (vs ++ extra) statics hp rd
  | Over_application remaining remaining_arity return_arity return_to exn_to =>
    match vs with
    | [v] =>
      run_call
        (fun partial => apply_cont_value return_to [partial] statics hp rd)
        (shape_of_call statics v Indirect_unknown_arity remaining
           remaining_arity return_arity)
        return_arity return_to exn_to statics hp rd
    | _ => Undefined
    end
  end.

Definition apply_continuation (k : continuation) (vs : list value)
    (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record) : step_result :=
  apply_cont_value (resolve k conts) vs statics hp rd.

(* A call: what [shape_of_call] says it is runs, and a partial
   application is given to the return continuation of the call. *)
Definition call_function (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity)
    (return_to exn_to : cont_value) (statics : static_env)
      (hp : heap) (rd : read_record) : step_result :=
  run_call (fun partial => apply_cont_value return_to [partial] statics hp rd)
    (shape_of_call statics callee kind vs args_arity return_arity)
    return_arity return_to exn_to statics hp rd.

(* CR mshinwell: a [Let] of a value of another kind than the one it
   gives its variable is undefined, and so is a handler applied to
   values of other kinds than those of its parameters
   ([apply_cont_value]). It is not certain that this is right. The
   compiler stops on a term in which it sees such a mismatch, so
   Simplify never turns one into [Invalid], and no well-typed program is
   known to produce one: nothing in the compiler says what such a term
   means. The other reading is that the kind a binder gives has no
   effect on execution; then nothing says that a variable holds a value
   of its kind, and the rewrites that remove a binding need not test
   kinds (FlambdaNormaliseSpec.v).
   Not tested under either reading: the kind of a field of a static
   block or of a value slot, where it is stored (the fields of a block
   that a primitive makes are tested, as arguments of the primitive:
   FlambdaPrimitiveSpec.eval_variadic_primitive); the kinds of the
   values given to a way out of the term; a value that a call returns,
   against the result arity of the code; and the extra arguments of the
   exception continuation of a call, against the kinds that the call
   gives them. The handler that a value returns to, or that an
   exception reaches with the extra arguments, tests them against its
   own kinds; and the result arity of the code is compared with that of
   the call, as two annotations ([call_code]). Decision D10. *)

Definition step (s : state) : step_result :=
  match s with
  | State (Let x kind defining_expr body) vars conts statics hp rd =>
    match exec_named statics hp rd vars defining_expr with
    | Some (v, hp', rd') =>
      if kind_eqb (kind_of_value v) kind
      then Step (State body ((x, v) :: vars) conts statics hp' rd')
      else Undefined
    | None => Undefined
    end
  | State (Let_static bindings body) vars conts statics hp rd =>
    match eval_static_bindings vars bindings statics with
    | Some statics' => Step (State body vars conts statics' hp rd)
    | None => Undefined
    end
  | State (Let_cont_non_recursive k h body) vars conts statics hp rd =>
    Step (State body vars ((k, Closure h vars conts []) :: conts) statics hp rd)
  | State (Let_cont_recursive handlers body) vars conts statics hp rd =>
    if handlers_bind_statics handlers then Undefined
    else
      Step
        (State body vars (rec_closures handlers vars conts ++ conts) statics hp
           rd)
  | State (Apply ap) vars conts statics hp rd =>
    match
      eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
      eval_simples vars (exn_extra_simples ap)
    with
    | Some callee, Some vs, Some extra =>
      call_function callee (ap_call_kind ap) vs (ap_args_arity ap)
        (ap_return_arity ap) (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd
    | _, _, _ => Undefined
    end
  | State (Apply_cont ac) vars conts statics hp rd =>
    match eval_simples vars (ac_args ac) with
    | Some vs => apply_continuation (ac_continuation ac) vs conts statics hp rd
    | None => Undefined
    end
  | State (Switch scrutinee arms) vars conts statics hp rd =>
    match eval_simple vars scrutinee with
    | Some v =>
      match naked_immediate_of_value v with
      | Some i =>
        match find_arm i arms with
        | Some ac => Step (State (Apply_cont ac) vars conts statics hp rd)
        | None => Undefined
        end
      | None => Undefined
      end
    | None => Undefined
    end
  | State Invalid _ _ _ _ _ => Undefined
  end.

(* At most [fuel] steps: the state reached, or how execution ended. *)
Fixpoint run (fuel : nat) (s : state) : step_result :=
  match fuel with
  | O => Step s
  | S fuel =>
    match step s with
    | Step s' => run fuel s'
    | result => result
    end
  end.

(* ======================================================================== *)
(* A [Let] and the heap                                                     *)
(* ======================================================================== *)

(* What [exec_named] does, said without saying what the primitives of
   the heap do. A primitive is run by FlambdaHeapSpec.exec_prim, and
   these statements are proved of [exec_named] so defined, from
   statements of FlambdaHeapSpec.v about [exec_prim]:
   [exec_prim_heap_statement], [exec_prim_eval_prim_statement],
   [eval_prim_exec_prim_statement], [exec_prim_no_location_statement],
   [exec_prim_evolves_statement] and [no_coeffects_statement]. They
   are how a proof reasons about the step of a [Let].

   A named expression that is not a primitive does not look at the heap:
   it has the value that [eval_named] gives it, and the heap is as it
   came. *)
Definition exec_named_not_prim_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (d : named),
    (∀ (p : prim) (args : list simple), d <> Prim p args) ->
    exec_named statics hp rd vars d
    = let* v := eval_named statics vars d in
      Some (v, hp, rd).

(* A primitive is run on the values of its arguments: two lists of
   arguments that have the same values, each under its environment, do
   the same. *)
Definition exec_named_args_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars vars' : env)
    (p : prim)
    (args args' : list simple),
    eval_simples vars' args' = eval_simples vars args ->
    exec_named statics hp rd vars' (Prim p args')
    = exec_named statics hp rd vars (Prim p args).

(* And a primitive has no value when one of its arguments has none. *)
Definition exec_named_no_args_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (p : prim)
    (args : list simple),
    eval_simples vars args = None ->
    exec_named statics hp rd vars (Prim p args) = None.

(* A primitive depends on the static environment only through what the
   symbols among the values of its arguments are bound to; and one that
   has a value keeps it, and returns the same heap, when the static
   environment grows: when each of those symbols that is bound stays
   bound to the same thing (FlambdaHeapSpec.exec_prim_static_values_statement
   and exec_prim_statics_extend_statement, which are the two statements
   of FlambdaPrimitiveSpec.v on a heap). *)
Definition exec_named_static_values_statement : Prop :=
  ∀ (statics statics' : static_env) (hp : heap) (rd : read_record) (vars : env)
    (p : prim)
    (args : list simple),
    (∀ (vs : list value) (s : symbol),
      eval_simples vars args = Some vs ->
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = lookup s (static_values statics')) ->
    exec_named statics hp rd vars (Prim p args)
    = exec_named statics' hp rd vars (Prim p args).

Definition exec_named_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (p : prim)
    (args : list simple) (v : value),
    (∀ (vs : list value) (s : symbol)
       (allocated : statically_allocated_value),
      eval_simples vars args = Some vs ->
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values statics') = Some allocated) ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp', rd') ->
    exec_named statics' hp rd vars (Prim p args) = Some (v, hp', rd').

(* Only a primitive of the heap changes the heap. *)
Definition exec_named_heap_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named)
    (v : value),
    (∀ (q : heap_primitive) (args : list simple), d <> Prim (Heap q) args) ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    hp' = hp.

(* A primitive that does not read the heap
   (FlambdaHeapSpec.prim_reads_heap) and that has a value has it by
   [eval_named], and leaves the heap and the read record as they came.
   The last is not the record doing nothing: such a primitive reads no
   cell, so there is nothing for it to record. *)
Definition exec_named_eval_named_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (p : prim)
    (args : list simple) (v : value),
    prim_reads_heap p = false ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp', rd') ->
    eval_named statics vars (Prim p args) = Some v ∧ hp' = hp ∧ rd' = rd.

(* And it has the value that [eval_named] gives it, when the locations
   among the values of its arguments are bound in the heap. *)
Definition eval_named_exec_named_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (p : prim)
    (args : list simple) (vs : list value) (v : value),
    prim_reads_heap p = false ->
    eval_simples vars args = Some vs ->
    (∀ l : location, Value_location l ∈ vs -> is_bound l hp = true) ->
    eval_named statics vars (Prim p args) = Some v ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp, rd).

(* A primitive that is not of the heap, applied to arguments none of
   whose values is a location, is as [eval_named] has it: a read of a
   block that is a value or that a symbol is bound to, for one. *)
Definition exec_named_no_location_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd : read_record) (vars : env)
    (p : prim)
    (args : list simple) (vs : list value),
    (∀ q : heap_primitive, p <> Heap q) ->
    eval_simples vars args = Some vs ->
    (∀ l : location, Value_location l ∉ vs) ->
    exec_named statics hp rd vars (Prim p args)
    = let* v := eval_named statics vars (Prim p args) in
      Some (v, hp, rd).

(* The heap after is one that the heap before evolves to
   (FlambdaHeapSpec.heap_evolves): what is bound stays bound, and what
   is not mutable stays as it is. So does the heap of a state at each
   step. *)
Definition exec_named_evolves_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record) (vars : env)
    (d : named)
    (v : value),
    exec_named statics hp rd vars d = Some (v, hp', rd') -> heap_evolves hp hp'.

Definition step_evolves_statement : Prop :=
  ∀ s s' : state,
    step s = Step s' -> heap_evolves (state_heap s) (state_heap s').

(* A step changes the heap only at a [Let]: there the heap after is the
   one that the defining expression leaves, and at every other
   construct it is the heap before. *)
Definition step_heap_statement : Prop :=
  ∀ s s' : state,
    step s = Step s' ->
    state_heap s' = state_heap s
    ∨ ∃ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
        (v : value),
        state_expr s = Let x kind d body
        ∧ exec_named (state_statics s) (state_heap s) (state_reads s)
          (state_vars s) d
          = Some (v, state_heap s', state_reads s').

(* An exit leaves the heap of the state that exits: the steps that can
   exit are a call and the application of a continuation, and neither
   changes the heap. *)
Definition step_exit_heap_statement : Prop :=
  ∀ (s : state) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    step s = Exit k vs hp rd -> hp = state_heap s.

(* A named expression that is stable (FlambdaHeapSpec.named_stable: not a
   primitive, or a primitive with no effects and no coeffects) and that
   has a value has the same value, and changes nothing, in every heap
   that the heap evolves to. This is what keeps a fact about the value
   of a variable across the steps that follow, stores among them. *)
Definition exec_named_stable_statement : Prop :=
  ∀ (statics : static_env) (hp hp' later : heap) (rd rd' rd_later : read_record)
    (vars : env) (d : named)
    (v : value),
    named_stable d = true ->
    heap_evolves hp later ->
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    exec_named statics later rd_later vars d = Some (v, later, rd_later).

(* A primitive makes its value, and what it puts in the heap, out of
   the values of its arguments and of what the heap holds
   (FlambdaHeapSpec.exec_prim_preserves_statement, which is
   FlambdaPrimitiveSpec.eval_prim_preserves_statement on a heap). *)
Definition exec_named_preserves_statement : Prop :=
  ∀ (P : value -> Prop) (statics : static_env) (hp hp' : heap)
    (rd rd' : read_record) (vars : env)
    (p : prim) (args : list simple) (v : value),
    holds_of_numbers P ->
    holds_of_blocks P ->
    holds_of_arrays P ->
    holds_in_static_blocks statics P ->
    holds_in_static_arrays statics P ->
    (∀ l : location, P (Value_location l)) ->
    (∀ vs : list value,
      eval_simples vars args = Some vs -> ∀ arg : value, arg ∈ vs -> P arg) ->
    heap_values_all P hp ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp', rd') ->
    P v ∧ heap_values_all P hp'.

(* ======================================================================== *)
(* The read record does nothing yet                                         *)
(* ======================================================================== *)

(* The record is carried by the machine and is not yet used: nothing
   looks at it and nothing adds to it. That is said here, once. Decisions
   D49 and D52 change [exec_named], and these statements with it. *)

(* A state with another record, and what a step or a run comes to with
   another record. *)
Definition with_reads (rd : read_record) (s : state) : state :=
  State (state_expr s) (state_vars s) (state_conts s) (state_statics s)
    (state_heap s) rd.

Definition result_with_reads (rd : read_record) (result : step_result)
    : step_result :=
  match result with
  | Step s => Step (with_reads rd s)
  | Exit k vs hp _ => Exit k vs hp rd
  | Undefined => Undefined
  end.

(* Nothing looks at the record: with another record the executor of a
   named expression and a step do what they do with this one, where
   they fail and where a step is undefined as well. *)
Definition exec_named_ignores_reads_statement : Prop :=
  ∀ (statics : static_env) (hp : heap) (rd1 rd2 : read_record) (vars : env)
    (d : named),
    exec_named statics hp rd2 vars d
    = match exec_named statics hp rd1 vars d with
      | Some (v, hp', _) => Some (v, hp', rd2)
      | None => None
      end.

Definition step_ignores_reads_statement : Prop :=
  ∀ (rd : read_record) (s : state),
    step (with_reads rd s) = result_with_reads rd (step s).

(* Nothing adds to the record: the executor gives back the record it was
   given, and a step leaves the record of the state. *)
Definition exec_named_keeps_reads_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (d : named) (v : value),
    exec_named statics hp rd vars d = Some (v, hp', rd') -> rd' = rd.

Definition step_keeps_reads_statement : Prop :=
  ∀ s : state, result_with_reads (state_reads s) (step s) = step s.

(* What the heap and the record of a state may become. While the record
   does nothing: the heap evolves (FlambdaHeapSpec.heap_evolves) and the
   record stays. Decisions D49 and D52 change this definition, not the
   two statements under it. *)
Definition memory_evolves (hp : heap) (rd : read_record) (hp' : heap)
    (rd' : read_record) : Prop :=
  heap_evolves hp hp' ∧ rd' = rd.

Definition memory_evolves_refl_statement : Prop :=
  ∀ (hp : heap) (rd : read_record), memory_evolves hp rd hp rd.

Definition memory_evolves_trans_statement : Prop :=
  ∀ (hp1 hp2 hp3 : heap) (rd1 rd2 rd3 : read_record),
    memory_evolves hp1 rd1 hp2 rd2 ->
    memory_evolves hp2 rd2 hp3 rd3 ->
    memory_evolves hp1 rd1 hp3 rd3.

(* The executor of a named expression and a step only do that. *)
Definition exec_named_memory_evolves_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (d : named) (v : value),
    exec_named statics hp rd vars d = Some (v, hp', rd') ->
    memory_evolves hp rd hp' rd'.

Definition step_memory_evolves_statement : Prop :=
  ∀ s s' : state,
    step s = Step s' ->
    memory_evolves (state_heap s) (state_reads s) (state_heap s')
      (state_reads s').

(* A stable primitive (FlambdaHeapSpec.stable_prim) that has run runs
   again where it left the heap and the record, with the same value, and
   changes neither. *)
Definition stable_prim_again_statement : Prop :=
  ∀ (statics : static_env) (hp hp' : heap) (rd rd' : read_record)
    (vars : env) (p : prim) (args : list simple) (v : value),
    stable_prim p = true ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp', rd') ->
    exec_named statics hp' rd' vars (Prim p args) = Some (v, hp', rd').

(* It stays: a stable primitive that runs and changes nothing does the
   same wherever the heap and the record evolve to. *)
Definition stable_prim_stays_statement : Prop :=
  ∀ (statics : static_env) (hp hp2 : heap) (rd rd2 : read_record)
    (vars : env) (p : prim) (args : list simple) (v : value),
    stable_prim p = true ->
    exec_named statics hp rd vars (Prim p args) = Some (v, hp, rd) ->
    memory_evolves hp rd hp2 rd2 ->
    exec_named statics hp2 rd2 vars (Prim p args) = Some (v, hp2, rd2).

(* ======================================================================== *)
(* Where a state holds values                                               *)
(* ======================================================================== *)

(* An invariant of a run that is about values (they are well formed, the
   symbols in them are bound) has to hold of every value that a state
   holds. This is where they are, said once: a new place for values, such
   as a heap, is added here.

   [P] is asked of every value that an environment binds, shadowed
   bindings included. *)
Definition env_values_all (P : value -> Prop) (vars : env) : Prop :=
  Forall (fun binding => P (snd binding)) vars.

(* A continuation holds the values of the environment that its closure
   keeps, and those that the continuations it keeps hold. A way out holds
   none. A continuation with extra arguments holds those, and what the
   continuation inside holds. *)
Fixpoint cont_values_all (P : value -> Prop) (c : cont_value) : Prop :=
  match c with
  | Outer _ => True
  | Closure _ vars conts _ =>
    env_values_all P vars
    ∧ fold_right
        (fun binding rest => cont_values_all P (snd binding) ∧ rest) True
        conts
  | Extra_args c extra => Forall P extra ∧ cont_values_all P c
  | Over_application remaining _ _ return_to exn_to =>
    Forall P remaining
    ∧ cont_values_all P return_to
    ∧ cont_values_all P exn_to
  end.

Definition conts_values_all (P : value -> Prop) (conts : cont_env) : Prop :=
  fold_right (fun binding rest => cont_values_all P (snd binding) ∧ rest)
    True conts.

(* A state holds values in its variables, in the environments that its
   continuations keep, in its static environment and in the cells of its
   heap. [P] is asked of the values of the first two and of the values
   of every entry of the heap (FlambdaHeapSpec.heap_values_all), and [Q]
   of what each symbol is bound to; what is to hold of the values inside
   a statically allocated value is for [Q] to say. Both may depend on
   the static environment of the state. The arguments of an exit are
   not in a state: the statement covers them. *)
Definition state_values_all (P : static_env -> value -> Prop)
    (Q : static_env -> statically_allocated_value -> Prop) (s : state)
    : Prop :=
  env_values_all (P (state_statics s)) (state_vars s)
  ∧ conts_values_all (P (state_statics s)) (state_conts s)
  ∧ Forall (fun binding => Q (state_statics s) (snd binding))
      (static_values (state_statics s))
  ∧ heap_values_all (P (state_statics s)) (state_heap s).

(* The same without the cells of the heap: the variables, the
   environments that the continuations keep and the static environment.
   It is what [state_values_all] asked before the cells were a place
   where a state holds values. *)
Definition state_values_all_but_cells (P : static_env -> value -> Prop)
    (Q : static_env -> statically_allocated_value -> Prop) (s : state)
    : Prop :=
  env_values_all (P (state_statics s)) (state_vars s)
  ∧ conts_values_all (P (state_statics s)) (state_conts s)
  ∧ Forall (fun binding => Q (state_statics s) (snd binding))
      (static_values (state_statics s)).

(* Of a state whose heap is empty the two say the same: the empty heap
   holds no value. The state that a unit starts in, run on its own or
   from its imports, is such a state. *)
Definition state_values_all_empty_heap_statement : Prop :=
  ∀ (P : static_env -> value -> Prop)
    (Q : static_env -> statically_allocated_value -> Prop) (s : state),
    state_heap s = empty_heap ->
    (state_values_all P Q s <-> state_values_all_but_cells P Q s).

(* A step keeps it, and the arguments of an exit satisfy [P], given of
   the state at hand:
   - at a [Let], [P] of the value of the defining expression and of the
     values of the heap after ([exec_named_preserves_statement] gives
     both for a primitive);
   - at a [Let_static] whose bindings evaluate, that [P] is kept by the
     static environment that results, and [Q] of everything bound in it;
   - at an [Apply] and at an [Apply_cont], [P] of the values of the
     simples that the step reads and passes on: the callee, the
     arguments, and the extra arguments of the exception continuation
     of a call, which the continuation given to the callee keeps. The
     value of a variable satisfies [P] already; this is for the
     constants and the symbols among them, of which [P] need not hold
     in general (a symbol that is not bound).
   What a call or the application of a continuation binds are those
   values, so nothing more is asked for them. A call may also make a
   partial application out of its callee and its arguments, or call one
   that an argument or a continuation holds: so [P] of a partial
   application is to be [P] of the value it calls and of its
   arguments. And a call of tupled code binds the fields of the tuple
   that it is given, by an argument or by a continuation: so [P] is to
   hold of the fields of a tuple of which it holds, under the static
   environment of the state. *)
Definition step_values_all_statement : Prop :=
  ∀ (P : static_env -> value -> Prop)
    (Q : static_env -> statically_allocated_value -> Prop) (s : state),
    (∀ (statics : static_env) (callee : value) (args : list value)
       (args_arity : arity),
      P statics (Value_partial_application callee args args_arity)
      <-> P statics callee ∧ Forall (P statics) args) ->
    state_values_all P Q s ->
    (∀ (size : nat) (tuple : value) (fields : list value),
      P (state_statics s) tuple ->
      tuple_fields (state_statics s) size tuple = Some fields ->
      Forall (P (state_statics s)) fields) ->
    (∀ (x : variable) (kind : flambda_kind) (defining_expr : named)
       (body : expr) (v : value) (hp' : heap) (rd' : read_record),
      state_expr s = Let x kind defining_expr body ->
      exec_named (state_statics s) (state_heap s) (state_reads s) (state_vars s)
        defining_expr
      = Some (v, hp', rd') ->
      P (state_statics s) v ∧ heap_values_all (P (state_statics s)) hp') ->
    (∀ (bindings : list static_binding) (body : expr)
       (statics' : static_env),
      state_expr s = Let_static bindings body ->
      eval_static_bindings (state_vars s) bindings (state_statics s)
      = Some statics' ->
      (∀ v : value, P (state_statics s) v -> P statics' v)
      ∧ Forall (fun binding => Q statics' (snd binding))
          (static_values statics')) ->
    (∀ (ap : apply_expr) (callee : value) (vs extra : list value),
      state_expr s = Apply ap ->
      eval_simple (state_vars s) (ap_callee ap) = Some callee ->
      eval_simples (state_vars s) (ap_args ap) = Some vs ->
      eval_simples (state_vars s) (exn_extra_simples ap) = Some extra ->
      P (state_statics s) callee
      ∧ Forall (P (state_statics s)) vs
      ∧ Forall (P (state_statics s)) extra) ->
    (∀ (ac : apply_cont) (vs : list value),
      state_expr s = Apply_cont ac ->
      eval_simples (state_vars s) (ac_args ac) = Some vs ->
      Forall (P (state_statics s)) vs) ->
    match step s with
    | Step s' => state_values_all P Q s'
    | Exit _ vs _ _ => Forall (P (state_statics s)) vs
    | Undefined => True
    end.

(* ======================================================================== *)
(* Behaviours                                                               *)
(* ======================================================================== *)

Inductive behaviour : Type :=
  | Exits (k : continuation) (vs : list value) (hp : heap)
  | Goes_wrong
  | Diverges.

Definition behaves (s : state) (b : behaviour) : Prop :=
  match b with
  | Exits k vs hp =>
    ∃ (fuel : nat) (rd : read_record), run fuel s = Exit k vs hp rd
  | Goes_wrong => ∃ fuel : nat, run fuel s = Undefined
  | Diverges => ∀ fuel : nat, ∃ s' : state, run fuel s = Step s'
  end.

(* A state has at most one behaviour. (That it has one is not provable
   without the excluded middle.) *)
Definition behaviour_unique_statement : Prop :=
  ∀ (s : state) (b b' : behaviour), behaves s b -> behaves s b' -> b = b'.

Definition equivalent (s s' : state) : Prop :=
  ∀ b : behaviour, behaves s b <-> behaves s' b.

(* The target behaves as the source does, unless the source goes wrong: an
   exit of the source is one of the target, and the way the target ends, if
   it does, is the way the source ends, unless the source goes wrong. With
   the excluded middle this is to say that every behaviour of the target is
   one of the source unless the source goes wrong; it is stated of the
   behaviours that end so that it can be proved without. *)
(* CR mshinwell: An exit is compared with its heap, and the two heaps are
   to be equal: the same locations, bound to the same cells, entry for
   entry. This is the interim choice, and the strict one. Different
   heaps are refused even when the difference is only garbage that is
   unreachable from the exit, or a consistent correspondence of
   locations: so a term after that does not allocate a block which the
   term before allocates and which is unreachable from the exit, and
   one that allocates two blocks in the other order, each then having
   the location of the other. (A change of order that leaves the same
   heap is not refused: a block allocated after a load in place of
   before it.) The comparison that is intended is up to a
   correspondence between the locations of the two runs; the
   prototype's is FlambdaHeapMachineSpec.outcome_related, which relates
   outcomes that carry their static environments, as an exit here does
   not. Decision D40. *)
(* CR mshinwell: An exit does not carry the static environment of the
   state that exits, and a symbol is a value whether it is bound or
   not. So a target may exit with a symbol that it never binds, and
   refine a source that binds it:
   FlambdaExitBoundSpec.refined_exit_unbound_statement has such a
   pair. Of a unit run from its imports and the empty heap, what says
   that the symbols at its exit are bound is three tests of the unit
   (FlambdaExitBoundSpec.unit_refines_exits_bound_statement joins them
   to a refinement of two such units), and they do not say what a
   symbol is bound to. The comparison that is intended is of exits
   that carry their static environments, as that of decision D40 is.
   Decision D57. *)
Definition refines (source target : state) : Prop :=
  (∀ (k : continuation) (vs : list value) (hp : heap),
    behaves source (Exits k vs hp) -> behaves target (Exits k vs hp))
  ∧ (∀ b : behaviour,
      b <> Diverges ->
      behaves target b -> behaves source b ∨ behaves source Goes_wrong).

(* What [refines] comes to: a source that does not go wrong and its target
   are equivalent, divergence included; and two states that refine each
   other are equivalent. *)
Definition refines_defined_statement : Prop :=
  ∀ source target : state,
    refines source target ->
    ~ behaves source Goes_wrong ->
    equivalent source target.

Definition refines_antisymmetric_statement : Prop :=
  ∀ s s' : state, refines s s' -> refines s' s -> equivalent s s'.

(* The same for expressions: under all environments. *)
Definition expr_equivalent (e e' : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    equivalent (State e vars conts statics hp rd)
      (State e' vars conts statics hp rd).

Definition expr_refines (source target : expr) : Prop :=
  ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    refines (State source vars conts statics hp rd)
      (State target vars conts statics hp rd).

(* A step does not change the behaviour. *)
Definition step_equivalent_statement : Prop :=
  ∀ s s' : state, step s = Step s' -> equivalent s s'.
