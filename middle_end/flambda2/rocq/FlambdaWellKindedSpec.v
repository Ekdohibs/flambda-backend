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

(* Well-kinded terms: what the machine makes undefined on account of
   kinds, and a check of the kinds that a term carries.

   This file is to be reviewed: the check ([well_kinded] and what it is
   made of, [well_kinded_unit]) against the places where the compiler
   checks kinds, listed below; what it is for a state to be well-kinded
   ([state_well_kinded]); [kind_fault]; and the statements.

   A variable of Flambda 2 has a kind ([Variable.kind]), and the terms of
   FlambdaModel.v give it where the variable is bound. Two things are
   said here about kinds, and they are to be kept apart.

   What is undefined, with no premise. A call that does not agree with
   the code it calls has no step
   ([call_at_wrong_kinds_undefined_statement];
   FlambdaSemanticsSpec.call_code makes the three tests of a full call,
   and FlambdaSemanticsSpec.shape_of_closure_call those of a partial
   application and of an over-application), and a
   primitive has no value on an argument of another kind than it takes
   (FlambdaPrimitiveSpec.eval_prim_kinds_statement, of the value that a
   primitive has with no heap; of a primitive run on a heap, as a
   [Let] runs it, FlambdaHeapSpec.exec_prim_kinds_statement, where the
   primitive is not of the heap). These are the
   mismatches of kinds that Simplify replaces by [Invalid], and the ones
   that well-typed programs are known to produce: OxCaml accepts
   programs (for one, a match on a GADT whose index has layout [any])
   from which Simplify, having inlined, gets a call that does not agree
   with its code, in a branch that is never run. The machine has to be
   undefined there, or Simplify would be wrong on such programs. The
   request (requests/C2-kinds.md) lists the forms found. Of the kinds
   that a primitive of the heap takes no statement is made:
   FlambdaPrimitiveSpec.prim_args_well_kinded is false of every one of
   them, and each makes its own tests
   (FlambdaHeapSpec.exec_heap_primitive). What a primitive gives, run
   on a heap, is of the kind of its result, and that is said of every
   primitive (FlambdaHeapSpec.exec_prim_result_kind_statement).

   What a binder gives, with no premise either. The machine also has no
   step at a [Let] of a value of another kind than its own, nor where a
   handler is applied to values of other kinds than those of its
   parameters (FlambdaSemanticsSpec.step, [apply_cont_value]; the CR at
   [step] says how certain that is). With the first test of a call, it
   follows that the step that binds a variable binds it to a value of
   the kind of its binder, whatever the term and whatever the state
   ([let_binds_at_kind_statement], [handler_binds_at_kinds_statement],
   [call_binds_at_kinds_statement],
   [called_closure_at_kind_statement]). These are statements about the
   step that binds. That a variable, where it is read, holds a value of the
   kind of the binder that the term has in scope there follows from
   them where no other binder of the same name comes between: that is
   the scoping of the term, and nothing here says it. What the tests do
   not reach: a value given to a way out of the term; and what is put
   in a field of a static block or in a value slot, which is not tested
   where it is stored. (A block that a primitive makes has its fields
   tested, as arguments of the primitive.) What is read back is tested
   where it is used: a load asks of the field the kind it is given
   (FlambdaPrimitiveSpec.v), and the [Let] that binds what was read
   tests its kind.

   The check. [well_kinded] is a boolean function: under the kinds of
   the variables in scope and the kinds of the parameters of the
   continuations in scope, the kinds that the term carries agree with one
   another. It is a sufficient condition and a diagnostic. A unit that
   passes it starts in a well-kinded state, a well-kinded state steps to
   a well-kinded state, and a well-kinded state is never undefined for
   one of the reasons of [kind_fault], the two tests of the binders
   among them. It is not a condition that the
   terms of the compiler are known to meet, and no statement about those
   terms may take it as a premise:
   - each of its conditions is one that the compiler checks where it
     visits the node, and stops on. It does not visit everything: a
     [Switch] on a constant, whose arm that is not taken applies a
     continuation to an argument of another kind than its parameter,
     compiles, since Simplify removes the arm before it looks at it, and
     fails the check here, which looks at every branch;
   - that what closure conversion builds passes it is known by
     experiment only: in the programs tried, a well-typed program gives
     a mismatch at a call, or in what a load reads, and never where a
     variable is bound (the request, as above).
   So it is not proposed as a condition of
   FlambdaWellFormedSpec.well_formed_unit. The test tool prints what it
   answers for the unit before Simplify and for the unit after, and goes
   on whatever the answer.

   What it checks:
   - every variable and every continuation used is in scope;
   - the kind a [Let] gives its variable is that of its defining
     expression ([kind_of_named]; the compiler's
     [Typing_env.add_variable_definition] fails otherwise). That of a
     primitive application is the kind of the primitive's result
     (FlambdaPrimitiveSpec.prim_result_kind), whatever its arguments;
   - the arguments of an [Apply_cont] have the kinds of the parameters of
     its continuation, in number and in order (the compiler fails
     otherwise: [Continuation_uses_env.add_continuation_use],
     [Continuation_uses.add_use]);
   - the scrutinee of a [Switch] is a naked immediate;
   - the return continuation of an [Apply] takes parameters of the kinds
     of the return arity the call is made at, unarized, and its exception
     continuation takes one argument of kind [Value] followed by the
     extra arguments of the call, each of which is of the kind it is
     given with;
   - a handler has as many kinds as parameters and its parameters are
     distinct ([Bound_parameters.create]), as are the parameters of code
     and its [my_closure] ([Bound_for_function.create]);
   - the body of code is well-kinded on its own: under its parameters,
     whose kinds are those of its parameter arity, unarized, and
     [my_closure], which has kind [Value], with its return continuation
     at its result arity, unarized, and its exception continuation at one
     [Value];
   - a value slot holds, where a closure is made (by a [Let] or
     statically, of a set of one function or of several), a [Simple] of
     the kind of the slot, and a variable bound to its projection has
     that kind. The compiler's value slots have their kinds
     ([Value_slot.kind]); here the kinds are a table given to the check
     ([slot_kinds]);
   - a variable bound to a closure, or to the projection of a function
     slot, has kind [Value];
   - the fields of a static block have kind [Value] (the blocks of this
     fragment are blocks of values; the compiler checks a field against
     the kind that the shape of the block gives it) and the contents of
     a static boxed number are a number of a kind that can be boxed.

   What it does not look at, each being undefined where it is executed:
   - the kinds of the arguments of a primitive
     ([Simplify_primitive.arg_kind_mismatch] gives [Invalid]; a
     primitive has no value on such arguments);
   - the kind of the callee of an [Apply] ([Provers.wrong_kind] answers
     "invalid", and the call becomes [Invalid]), and of what a value slot
     or a function slot is projected from (in the compiler a primitive,
     so as above). What is not a closure can be neither called nor
     projected from. Nor does it look at the function slot that a
     projection names (FlambdaValueSpec.project_value_slot,
     [project_function_slot]);
   - whether a call is made at the arities of the code called, even where
     the code is named, and whether its arguments are of the kinds of the
     arity it is made at: the tests of FlambdaSemanticsSpec.call_code
     and of FlambdaSemanticsSpec.shape_of_closure_call.

   What else it does not check:
   - that symbols and code ids are in scope: a symbol is a value whether
     or not it is bound, and deleted code is in scope and not bound;
   - that a bound name is not already in scope (Simplify's freshness);
   - where static bindings are: that is FlambdaWellFormedSpec.v.

   The dynamic counterpart. Every value has a kind
   (FlambdaValueSpec.kind_of_value). A state is well-kinded
   ([state_well_kinded], a boolean function too) when its expression is
   well-kinded under the kinds of the values of its variables
   ([kinds_of_env]) and the kinds of the parameters of what its
   continuations stand for ([arities_of_conts]), and everything it holds
   is well-kinded: the values, down the value slots of closures and the
   fields of blocks; the handlers of continuation closures, under the
   environments they keep; the code and the statically allocated values
   of the static environment; and the values that the cells of the heap
   hold ([heap_well_kinded]), which a load gives to a variable. A
   location is well-kinded as a value, whatever its cell holds: what the
   cell holds is asked of the heap. A continuation that is a way out of the
   term has the parameters given by [outer], which is the interface of
   the term: for a unit, its return and exception continuations, each
   taking one [Value] ([unit_arities]). The kinds are those of the values
   held now: the predicate does not remember what a binder said. What a
   binder gives is in the step, and needs no well-kinded state (above).

   The statements:
   - a call that does not agree with the code called is undefined,
     whatever the state ([call_at_wrong_kinds_undefined_statement]);
   - the step that binds a variable binds a value of the kind of the
     binder, whatever the state ([let_binds_at_kind_statement],
     [handler_binds_at_kinds_statement],
     [call_binds_at_kinds_statement],
     [called_closure_at_kind_statement]);
   - a unit that passes the check starts in a well-kinded state
     ([well_kinded_unit_state_statement]);
   - a well-kinded state steps to a well-kinded state
     ([step_well_kinded_statement], [run_well_kinded_statement]);
   - a well-kinded state is not undefined for a reason that the check
     rules out ([well_kinded_no_kind_fault_statement]): [kind_fault] is
     those reasons, as a test on a state. It reads no unbound variable;
     its [Let] is of a value of the kind of the [Let]; it applies no
     handler to the wrong number of arguments, nor to values of other
     kinds than those of its parameters; its [Switch] is on a value of
     kind naked immediate;
   - a well-kinded state exits only through a continuation of [outer],
     with values of the kinds of its parameters
     ([well_kinded_exit_statement]);
   - the three together for a unit run on its own
     ([well_kinded_unit_never_faults_statement],
     [well_kinded_unit_exits_statement]).

   What a well-kinded state may still do that is undefined: reach
   [Invalid]; apply a primitive that has no value, as it has none on
   arguments of the wrong kinds; [Switch] on a value with no arm, or on
   a poison constant (which has the kind it is given); call what is not
   a closure, a closure whose code is not bound, directly a closure of
   other code than the one named, a closure of code with which the call
   does not agree, or code whose body binds something static; project a
   value slot from what is not a closure with that slot; bind a static
   name twice, or a recursive handler that binds something static. The
   last three are what FlambdaWellFormedSpec.v rules out for a unit.

   Partial applications and over-applications. A call with fewer or
   more arguments than the code has parameters is one, where its
   arities pass the compiler's test
   (FlambdaModel.partial_or_over_application) and its arguments are of
   the kinds of its arity; otherwise it is undefined, as Simplify makes
   it [Invalid]. A well-kinded state stays well-kinded through both: a
   partial application is a well-kinded value of kind [Value] when the
   value it calls and its arguments are well-kinded, and the test makes
   the call that returns it expect one [Value]; what is left of an
   over-application ([Over_application]) is a well-kinded continuation
   of one parameter of kind [Value], which the test makes the result of
   the code, when the arguments that remain are well-kinded and its two
   continuations are, the one it returns to at the return arity of the
   call. The kinds of the arguments that a partial application keeps
   are not compared with anything until it is called with the rest.

   Tupled functions are left out too. A call of unknown arity gives a
   tupled function one argument, the tuple, whatever the arity of its
   code ([Code_metadata.is_tupled]; [simplify_direct_tuple_application]
   turns such a call into a direct one of the code, on the fields); a
   direct call and a call of known arity go by the arity of the code.
   The code of this fragment has no such mark, and its [Indirect] is
   both sorts of indirect call. So the call with the tuple would be
   a partial application here, or undefined, or, where the code has one
   parameter, run with the parameter bound to the tuple and not to its
   field, which is a wrong result and not an undefined one. The test
   tool refuses a unit that has tupled code and a call that is not
   direct.

   The primitives. The check takes the kind of the result of a primitive
   from FlambdaPrimitiveSpec.prim_result_kind, and the proofs use two of
   the laws there: a primitive that has a value has one of its result
   kind ([eval_prim_kinds_statement]), and a primitive makes its value
   out of its arguments ([eval_prim_preserves_statement]).

   The proofs are in FlambdaWellKindedProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaHeapSpec FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is in scope                                                         *)
(* ======================================================================== *)

(* The variables in scope with their kinds, the continuations in scope
   with the kinds of their parameters, and the kinds of the value slots
   ([Value_slot.kind]). *)
Definition kind_env : Type := finite_map flambda_kind.
Definition arity_env : Type := finite_map (list flambda_kind).
Definition slot_kinds : Type := finite_map flambda_kind.

Definition has_kind (found : option flambda_kind) (kind : flambda_kind)
    : bool :=
  match found with
  | Some kind' => kind_eqb kind' kind
  | None => false
  end.

Definition has_arity (found : option (list flambda_kind))
    (expected : list flambda_kind) : bool :=
  match found with
  | Some found => kinds_eqb found expected
  | None => false
  end.

(* ======================================================================== *)
(* The check of a term                                                      *)
(* ======================================================================== *)

(* [Simple.kind], for a variable in scope. *)
Definition kind_of_simple (kinds : kind_env) (s : simple)
    : option flambda_kind :=
  match s with
  | Var x => lookup x kinds
  | Symbol _ => Some Value
  | Const c => Some (kind_of_const c)
  end.

Fixpoint kinds_of_simples (kinds : kind_env) (args : list simple)
    : option (list flambda_kind) :=
  match args with
  | [] => Some []
  | s :: args =>
    let* k := kind_of_simple kinds s in
    let* ks := kinds_of_simples kinds args in
    Some (k :: ks)
  end.

(* The kinds of parameters, bound as FlambdaSemanticsSpec.bind_params
   binds their values; there must be as many kinds as parameters. *)
Fixpoint bind_param_kinds (params : list variable)
    (params_kinds : list flambda_kind) (kinds : kind_env)
    : option kind_env :=
  match params, params_kinds with
  | [], [] => Some kinds
  | x :: params, k :: params_kinds =>
    bind_param_kinds params params_kinds ((x, k) :: kinds)
  | _, _ => None
  end.

(* The kinds of the parameters of the handlers of a recursive
   [Let_cont]. *)
Definition handler_arities
    (handlers : list (continuation * continuation_handler)) : arity_env :=
  map_values handler_kinds handlers.

Section Checker.

Variable slots : slot_kinds.

(* Each value slot holds a [Simple] in scope, of the kind of the slot. *)
Definition value_slots_well_kinded (kinds : kind_env)
    (value_slots : list (value_slot * simple)) : bool :=
  forallb
    (fun '(slot, s) =>
       match lookup slot slots with
       | Some kind => has_kind (kind_of_simple kinds s) kind
       | None => false
       end)
    value_slots.

(* [Flambda.Named.kind]: the kind of a defining expression, if what it
   reads is in scope and, for a closure, its value slots are well-kinded.
   A closure, of either form, and the projection of a function slot have
   kind [Value]; the projection of a value slot has the kind of the
   slot. The kinds of the arguments of a primitive, and of what a slot
   is projected from, are not looked at, nor is the function slot that a
   projection names. *)
Definition kind_of_named (kinds : kind_env) (defining_expr : named)
    : option flambda_kind :=
  match defining_expr with
  | Simple s => kind_of_simple kinds s
  | Prim p args =>
    let* _ := kinds_of_simples kinds args in
    Some (prim_result_kind p)
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    if value_slots_well_kinded kinds value_slots then Some Value else None
  | Project_value_slot slot closure
  | Project_value_slot_from _ slot closure =>
    let* _ := kind_of_simple kinds closure in
    lookup slot slots
  | Project_function_slot _ _ closure =>
    let* _ := kind_of_simple kinds closure in
    Some Value
  end.

Definition static_const_well_kinded (kinds : kind_env)
    (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    value_slots_well_kinded kinds value_slots
  | Static_const_block _ fields =>
    forallb (fun s => has_kind (kind_of_simple kinds s) Value) fields
  | Static_const_immutable_value_array elements =>
    forallb (fun s => has_kind (kind_of_simple kinds s) Value) elements
  | Static_const_boxed_number contents =>
    match kind_of_simple kinds contents with
    | Some kind => is_some (boxable_number_of_kind kind)
    | None => false
    end
  | Static_const_immutable_string _ => true
  end.

(* The continuation that a trap action names is in scope. *)
Definition trap_action_in_scope (conts : arity_env) (ac : apply_cont)
    : bool :=
  match ac_trap_action ac with
  | Some action => is_bound (trap_action_handler action) conts
  | None => true
  end.

(* The arguments are in scope and have the kinds of the parameters of the
   continuation, which is in scope, as is the continuation of the trap
   action. *)
Definition apply_cont_well_kinded (kinds : kind_env) (conts : arity_env)
    (ac : apply_cont) : bool :=
  trap_action_in_scope conts ac
  && match lookup (ac_continuation ac) conts with
     | Some params_kinds =>
       has_arity (kinds_of_simples kinds (ac_args ac)) params_kinds
     | None => false
     end.

(* The callee and the arguments are in scope, whatever their kinds. The
   extra arguments of the exception continuation are in scope and of the
   kinds they are given with. The return continuation takes what the
   call says it returns, and the exception continuation one [Value]
   followed by the extra arguments
   ([Exn_continuation.arity]). *)
Definition apply_well_kinded (kinds : kind_env) (conts : arity_env)
    (ap : apply_expr) : bool :=
  is_some (kind_of_simple kinds (ap_callee ap))
  && is_some (kinds_of_simples kinds (ap_args ap))
  && has_arity (kinds_of_simples kinds (exn_extra_simples ap))
       (map snd (ap_exn_extra_args ap))
  && has_arity (lookup (ap_continuation ap) conts)
       (unarize (ap_return_arity ap))
  && has_arity (lookup (ap_exn_continuation ap) conts)
       (Value :: map snd (ap_exn_extra_args ap)).

(* CR mshinwell: the check answers no where the kind of a [Let] is not
   that of its defining expression, an argument of an [Apply_cont] is
   not of the kind of the parameter, the scrutinee of a [Switch] is not
   a naked immediate, the return continuation of a call does not take
   what the call says it returns, a value slot holds a [Simple] of
   another kind than the slot's, or a field of a static block is not a
   [Value]. The compiler has a check of its own for each, which stops it
   where Simplify visits the node, so Simplify never makes one of these
   [Invalid]; and no well-typed program is known to produce one. The
   machine is undefined where it executes one of the first three, and
   where a call returns a value of another kind than the handler it
   returns to takes (the CR in FlambdaSemanticsSpec.v); it does not
   test the last two. What is not decided is what they should be: an
   error of the term, as the check has it, which is then a condition
   that some terms the compiler accepts do not meet (it does not visit
   a branch it has removed); or undefined where executed, as the
   machine has the first four. Decision D10. *)

(* The check: the expression is well-kinded under the kinds of the
   variables and the kinds of the parameters of the continuations in
   scope. *)
Fixpoint well_kinded (kinds : kind_env) (conts : arity_env) (e : expr)
    : bool :=
  match e with
  | Let x kind defining_expr body =>
    has_kind (kind_of_named kinds defining_expr) kind
    && well_kinded ((x, kind) :: kinds) conts body
  | Let_static bindings body =>
    forallb (binding_well_kinded kinds) bindings
    && well_kinded kinds conts body
  | Let_cont_non_recursive k h body =>
    handler_well_kinded kinds conts h
    && well_kinded kinds ((k, handler_kinds h) :: conts) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_well_kinded kinds (handler_arities handlers ++ conts) h)
      handlers
    && well_kinded kinds (handler_arities handlers ++ conts) body
  | Apply ap => apply_well_kinded kinds conts ap
  | Apply_cont ac => apply_cont_well_kinded kinds conts ac
  | Switch scrutinee arms =>
    has_kind (kind_of_simple kinds scrutinee)
      (Naked_number Kind_naked_immediate)
    && forallb (fun arm => apply_cont_well_kinded kinds conts (snd arm)) arms
  | Invalid => true
  end
with handler_well_kinded (kinds : kind_env) (conts : arity_env)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    distinct params
    && match bind_param_kinds params (hm_params_kinds metadata) kinds with
       | Some handler_kinds => well_kinded handler_kinds conts handler
       | None => false
       end
  end
with binding_well_kinded (kinds : kind_env) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ c => code_well_kinded c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const => static_const_well_kinded kinds const
  end
with code_well_kinded (c : code) : bool :=
  match c with
  | Code params my_closure return_continuation exn_continuation metadata
      body =>
    distinct (my_closure :: params)
    && match
         bind_param_kinds params (unarize (cm_params_arity metadata))
           [(my_closure, Value)]
       with
       | Some body_kinds =>
         well_kinded body_kinds
           [(return_continuation, unarize (cm_result_arity metadata));
            (exn_continuation, [Value])]
           body
       | None => false
       end
  end.

(* ======================================================================== *)
(* Well-kinded units                                                        *)
(* ======================================================================== *)

(* The interface of a unit: its return continuation takes the module
   block and its exception continuation the exception, each one argument
   of kind [Value]. *)
Definition unit_arities (unit_return unit_exn : continuation) : arity_env :=
  [(unit_return, [Value]); (unit_exn, [Value])].

(* The test tool runs the check of a unit, below, on the unit before
   Simplify and on the unit after, prints the two answers, and goes on
   whatever they are (decision D11). It is not part of
   FlambdaValidateSpec.validate_unit, whose statement has no premise
   about kinds. *)

(* A unit, with its return and exception continuations: it is
   well-kinded with no variable in scope. *)
Definition well_kinded_unit (unit_return unit_exn : continuation) (e : expr)
    : bool :=
  well_kinded [] (unit_arities unit_return unit_exn) e.

(* ======================================================================== *)
(* Well-kinded states                                                       *)
(* ======================================================================== *)

(* Every closure in the value holds, in each value slot, a well-kinded
   value of the kind of the slot. The fields of a block are looked into:
   a closure may be among them. *)
Fixpoint value_well_kinded (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _
  | Value_location _ =>
    true
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb
      (fun '(slot, v) =>
         has_kind (lookup slot slots) (kind_of_value v)
         && value_well_kinded v)
      value_slots
  | Value_block _ fields | Value_immutable_array fields =>
    forallb value_well_kinded fields
  | Value_partial_application callee args _ =>
    value_well_kinded callee && forallb value_well_kinded args
  end.

Definition env_well_kinded (vars : env) : bool :=
  forallb (fun '(_, v) => value_well_kinded v) vars.

(* The kinds of the variables of an environment: those of their values. *)
Definition kinds_of_env (vars : env) : kind_env :=
  map_values kind_of_value vars.

(* The kinds of the parameters of what a continuation stands for: those
   of a handler, or the ones [outer] gives a way out of the term. A way
   out that [outer] does not bind is not well-kinded
   ([cont_well_kinded]); the answer for it is only there to make the
   function total. A continuation that has extra arguments takes what
   the continuation inside does but for the last ones, which the extra
   arguments are. The continuation of the full call inside an
   over-application takes its one result, a value. *)
Fixpoint arity_of_cont (outer : arity_env) (c : cont_value)
    : list flambda_kind :=
  match c with
  | Closure h _ _ _ => handler_kinds h
  | Outer k =>
    match lookup k outer with
    | Some params_kinds => params_kinds
    | None => []
    end
  | Extra_args c extra =>
    let params_kinds := arity_of_cont outer c in
    firstn (length params_kinds - length extra) params_kinds
  | Over_application _ _ _ _ _ => [Value]
  end.

(* The kinds of the parameters of the continuations of an environment;
   one that is not bound is a way out of the term
   (FlambdaSemanticsSpec.resolve). *)
Definition arities_of_conts (outer : arity_env) (conts : cont_env)
    : arity_env :=
  map_values (arity_of_cont outer) conts ++ outer.

(* A way out of the term is one [outer] has. A handler closure keeps
   well-kinded environments, under which its handler is well-kinded, and
   so are the handlers defined with it, which are in scope in it. The
   extra arguments of a continuation are well-kinded values, of the
   kinds of the last parameters of the continuation inside. The
   continuation of the full call inside an over-application keeps
   well-kinded arguments and two well-kinded continuations: the one it
   returns to takes the results of the whole call, and the one for
   exceptions a value. *)
Fixpoint cont_well_kinded (outer : arity_env) (c : cont_value) : bool :=
  match c with
  | Outer k => is_bound k outer
  | Extra_args c extra =>
    cont_well_kinded outer c
    && forallb value_well_kinded extra
    && kinds_eqb (map kind_of_value extra)
         (skipn (length (arity_of_cont outer c) - length extra)
            (arity_of_cont outer c))
  | Closure h vars conts recursive =>
    env_well_kinded vars
    && forallb (fun '(_, c) => cont_well_kinded outer c) conts
    && handler_well_kinded (kinds_of_env vars)
         (handler_arities recursive ++ arities_of_conts outer conts) h
    && forallb
         (fun '(_, h) =>
            handler_well_kinded (kinds_of_env vars)
              (handler_arities recursive ++ arities_of_conts outer conts) h)
         recursive
  | Over_application remaining _ return_arity return_to exn_to =>
    forallb value_well_kinded remaining
    && cont_well_kinded outer return_to
    && cont_well_kinded outer exn_to
    && kinds_eqb (arity_of_cont outer return_to) (unarize return_arity)
    && kinds_eqb (arity_of_cont outer exn_to) [Value]
  end.

Definition conts_well_kinded (outer : arity_env) (conts : cont_env) : bool :=
  forallb (fun '(_, c) => cont_well_kinded outer c) conts.

Definition allocated_well_kinded (allocated : statically_allocated_value)
    : bool :=
  match allocated with
  | Static_closure code value_slots =>
    value_well_kinded (Value_closure code value_slots)
  | Static_closure_of_set slot _ function_decls value_slots =>
    value_well_kinded (Value_closure_of_set slot function_decls value_slots)
  | Static_block _ fields =>
    forallb
      (fun v => kind_eqb (kind_of_value v) Value && value_well_kinded v)
      fields
  | Static_immutable_array elements =>
    forallb
      (fun v => kind_eqb (kind_of_value v) Value && value_well_kinded v)
      elements
  | Static_boxed_number contents =>
    is_some (boxable_number_of_kind (kind_of_value contents))
  | Static_string _ | Static_imported _ => true
  end.

Definition statics_well_kinded (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_well_kinded c) (static_code statics)
  && forallb (fun '(_, allocated) => allocated_well_kinded allocated)
       (static_values statics).

(* Every value that a cell of the heap holds is well-kinded: of every
   binding of the list, those that a later write hides too, as
   FlambdaHeapSpec.heap_values_all reads a heap. Nothing is asked of
   the shape of a cell, nor of the kinds of what it holds: a load
   tests the kind of what it reads (FlambdaHeapSpec.load_field), and a
   [Let] the kind of what it binds. *)
Definition heap_well_kinded (hp : heap) : bool :=
  forallb
    (fun binding => forallb value_well_kinded (values_of_cell (snd binding)))
    hp.

Definition state_well_kinded (outer : arity_env) (s : state) : bool :=
  well_kinded (kinds_of_env (state_vars s))
    (arities_of_conts outer (state_conts s)) (state_expr s)
  && env_well_kinded (state_vars s)
  && conts_well_kinded outer (state_conts s)
  && statics_well_kinded (state_statics s)
  && heap_well_kinded (state_heap s).

(* ======================================================================== *)
(* What a well-kinded state cannot do                                       *)
(* ======================================================================== *)

(* The [Simple]s that a step of the expression evaluates. *)
Definition simples_of_static_const (const : static_const) : list simple :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    map snd value_slots
  | Static_const_block _ fields => fields
  | Static_const_immutable_value_array elements => elements
  | Static_const_boxed_number contents => [contents]
  | Static_const_immutable_string _ => []
  end.

Definition simples_of_binding (binding : static_binding) : list simple :=
  match binding with
  | Bind_symbol _ const => simples_of_static_const const
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

Definition simples_read (e : expr) : list simple :=
  match e with
  | Let _ _ (Simple s) _ => [s]
  | Let _ _ (Prim _ args) _ => args
  | Let _ _ (Set_of_closures _ value_slots) _
  | Let _ _ (Closure_of_set _ _ value_slots) _ =>
    map snd value_slots
  | Let _ _ (Project_value_slot _ closure) _
  | Let _ _ (Project_value_slot_from _ _ closure) _
  | Let _ _ (Project_function_slot _ _ closure) _ =>
    [closure]
  | Let_static bindings _ => flat_map simples_of_binding bindings
  | Apply ap => ap_callee ap :: ap_args ap ++ exn_extra_simples ap
  | Apply_cont ac => ac_args ac
  | Switch scrutinee _ => [scrutinee]
  | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _ | Invalid => []
  end.

Definition unbound (vars : env) (s : simple) : bool :=
  negb (is_some (eval_simple vars s)).

(* The continuation stands for a handler that has not as many parameters
   as the values it is given: [given] of them, and the extra arguments
   that the continuation has. The continuation of the full call inside
   an over-application takes one. *)
Fixpoint wrong_number_given (c : cont_value) (given : nat) : bool :=
  match c with
  | Closure (Handler params _ _) _ _ _ =>
    negb (Nat.eqb (length params) given)
  | Outer _ => false
  | Extra_args c extra => wrong_number_given c (given + length extra)
  | Over_application _ _ _ _ _ => negb (Nat.eqb 1 given)
  end.

Definition wrong_number_of_arguments (ac : apply_cont) (conts : cont_env)
    : bool :=
  wrong_number_given (resolve (ac_continuation ac) conts)
    (length (ac_args ac)).

(* The value that the defining expression of a [Let] gives, run as the
   step runs it, on the heap and with the read record of the state
   (FlambdaSemanticsSpec.exec_named), is not of the kind that the [Let]
   gives its variable. *)
Definition wrong_kind_bound (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env) (kind : flambda_kind) (d : named)
    : bool :=
  match exec_named statics hp rd vars d with
  | Some (v, _, _) => negb (kind_eqb (kind_of_value v) kind)
  | None => false
  end.

(* The continuation stands for a handler that gives its parameters other
   kinds than those of the values it is given: [given], followed by the
   kinds of the extra arguments that the continuation has. What is left
   of an over-application takes one [Value], the function to call with
   the arguments that remain. *)
Fixpoint wrong_kinds_given (c : cont_value) (given : list flambda_kind)
    : bool :=
  match c with
  | Closure (Handler _ metadata _) _ _ _ =>
    negb (kinds_eqb given (hm_params_kinds metadata))
  | Outer _ => false
  | Extra_args c extra =>
    wrong_kinds_given c (given ++ map kind_of_value extra)
  | Over_application _ _ _ _ _ => negb (kinds_eqb given [Value])
  end.

Definition wrong_kinds_of_arguments (ac : apply_cont) (vars : env)
    (conts : cont_env) : bool :=
  match eval_simples vars (ac_args ac) with
  | Some vs =>
    wrong_kinds_given (resolve (ac_continuation ac) conts)
      (map kind_of_value vs)
  | None => false
  end.

(* The state is undefined for a reason that kinds rule out: it reads an
   unbound variable; it is a [Let] of a value of another kind than its
   own; it applies a handler to the wrong number of arguments, or to
   values of other kinds than those of its parameters; or it is a
   [Switch] on a value that is not of kind naked immediate. *)
Definition kind_fault (s : state) : bool :=
  existsb (unbound (state_vars s)) (simples_read (state_expr s))
  || match state_expr s with
     | Let _ kind d _ =>
       wrong_kind_bound (state_statics s) (state_heap s) (state_reads s)
         (state_vars s) kind d
     | Apply_cont ac =>
       wrong_number_of_arguments ac (state_conts s)
       || wrong_kinds_of_arguments ac (state_vars s) (state_conts s)
     | Switch scrutinee _ =>
       match eval_simple (state_vars s) scrutinee with
       | Some v =>
         negb
           (kind_eqb (kind_of_value v) (Naked_number Kind_naked_immediate))
       | None => false
       end
     | Let_static _ _ | Let_cont_non_recursive _ _ _
     | Let_cont_recursive _ _ | Apply _ | Invalid =>
       false
     end.

End Checker.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What Simplify replaces by [Invalid] at a call of a closure has no
   step: a call whose arguments are not of the kinds of the arity it is
   made at; a call with as many arguments as the code has parameters
   that is not made at the arities of the code; and a call with fewer or
   more arguments whose arities fail the test for a partial application
   or an over-application. The last two are of code that is not tupled,
   which the call gives its arguments as they are: a call of tupled code
   is compared with the code once the tuple that it gives has been
   taken apart (FlambdaTupledSpec.v). Nothing is asked of the state. *)
Definition call_at_wrong_kinds_undefined_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee : value) (vs : list value) (called : code_id)
    (value_slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some vs ->
    closure_of statics callee = Some (called, value_slots) ->
    lookup called (static_code statics) = Some c ->
    map kind_of_value vs <> unarize (ap_args_arity ap)
    ∨ code_is_tupled c = false
      ∧ (length (ap_args_arity ap) = length (code_params_arity c)
         ∧ (unarize_per_parameter (ap_args_arity ap)
            <> unarize_per_parameter (code_params_arity c)
            ∨ ap_return_arity ap <> code_result_arity c)
         ∨ length (ap_args_arity ap) <> length (code_params_arity c)
           ∧ partial_or_over_application (ap_args_arity ap)
               (ap_return_arity ap) (code_params_arity c)
               (code_result_arity c)
             = false) ->
    step (State (Apply ap) vars conts statics hp rd) = Undefined.

(* What a binder gives, whatever the state: the step that binds a
   variable binds it to a value of the kind of its binder. A [Let] that
   has a step binds a value of its kind. A handler that runs has its
   parameters bound to values of the kinds it gives them, one for each.
   The body of code is entered in one place, a full call that has a
   step (FlambdaSemanticsSpec.call_code, which is what every call that
   runs code comes to: a call with as many arguments as the code has
   parameters, the full call inside an over-application, and the call
   of a partial application): it runs with the parameters of the code
   bound to values of the kinds of its parameters
   ([call_binds_at_kinds_statement]), and [my_closure] to the closure
   called, which is of kind [Value]
   ([called_closure_at_kind_statement], about the closure that a call
   gives [call_code]: FlambdaSemanticsSpec.shape_of_call says what a
   call is, and [call_function] and [apply_cont_value] run nothing
   else). The third statement is about that place, and not about an
   [Apply], which may do more than enter a body, or less: a partial
   application enters none. *)
Definition let_binds_at_kind_statement : Prop :=
  ∀ (x : variable) (kind : flambda_kind) (d : named) (body : expr)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record)
      (s : state),
    step (State (Let x kind d body) vars conts statics hp rd) = Step s ->
    ∃ (v : value) (hp' : heap) (rd' : read_record),
      exec_named statics hp rd vars d = Some (v, hp', rd')
      ∧ kind_of_value v = kind
      ∧ s = State body ((x, v) :: vars) conts statics hp' rd'.

Definition handler_binds_at_kinds_statement : Prop :=
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
      (s : state),
    apply_cont_value
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      vs statics hp rd
    = Step s ->
    map kind_of_value vs = hm_params_kinds metadata
    ∧ ∃ handler_vars : env,
        bind_params params vs def_vars = Some handler_vars
        ∧ s
          = State handler handler_vars
              (rec_closures recursive def_vars def_conts ++ def_conts)
              statics hp rd.

Definition call_binds_at_kinds_statement : Prop :=
  ∀ (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s : state),
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
    = Step s ->
    map kind_of_value vs = code_params_kinds c
    ∧ bind_params (code_params c) vs [(code_my_closure c, closure)]
      = Some (state_vars s)
    ∧ state_expr s = code_body c.

Definition called_closure_at_kind_statement : Prop :=
  ∀ (statics : static_env) (callee : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity),
    match shape_of_call statics callee kind vs args_arity return_arity with
    | Some (Full_call closure _ _ _) | Some (Over_call closure _ _ _ _ _) =>
      kind_of_value closure = Value
    | Some (Partial_call _) | None => True
    end.

Definition well_kinded_unit_state_statement : Prop :=
  ∀ (slots : slot_kinds) (unit_return unit_exn : continuation) (e : expr),
    well_kinded_unit slots unit_return unit_exn e = true ->
    state_well_kinded slots (unit_arities unit_return unit_exn)
      (State e [] [] empty_static_env empty_heap no_reads)
    = true.

Definition step_well_kinded_statement : Prop :=
  ∀ (slots : slot_kinds) (outer : arity_env) (s s' : state),
    state_well_kinded slots outer s = true ->
    step s = Step s' ->
    state_well_kinded slots outer s' = true.

Definition run_well_kinded_statement : Prop :=
  ∀ (slots : slot_kinds) (outer : arity_env) (n : nat) (s s' : state),
    state_well_kinded slots outer s = true ->
    run n s = Step s' ->
    state_well_kinded slots outer s' = true.

Definition well_kinded_no_kind_fault_statement : Prop :=
  ∀ (slots : slot_kinds) (outer : arity_env) (s : state),
    state_well_kinded slots outer s = true -> kind_fault s = false.

Definition well_kinded_exit_statement : Prop :=
  ∀ (slots : slot_kinds) (outer : arity_env) (s : state) (k : continuation)
    (vs : list value) (hp : heap) (rd : read_record),
    state_well_kinded slots outer s = true ->
    step s = Exit k vs hp rd ->
    lookup k outer = Some (map kind_of_value vs)
    ∧ forallb (value_well_kinded slots) vs = true.

(* A well-kinded unit, run on its own, never meets a kind fault, and
   exits, if it does, through its return or its exception continuation
   with one value of kind [Value]. *)
Definition well_kinded_unit_never_faults_statement : Prop :=
  ∀ (slots : slot_kinds) (unit_return unit_exn : continuation) (e : expr)
    (n : nat) (s : state),
    well_kinded_unit slots unit_return unit_exn e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    kind_fault s = false.

Definition well_kinded_unit_exits_statement : Prop :=
  ∀ (slots : slot_kinds) (unit_return unit_exn : continuation) (e : expr)
    (n : nat) (k : continuation) (vs : list value) (hp : heap)
      (rd : read_record),
    well_kinded_unit slots unit_return unit_exn e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Exit k vs hp rd
      ->
    (k = unit_return ∨ k = unit_exn)
    ∧ ∃ v : value, vs = [v] ∧ kind_of_value v = Value.
