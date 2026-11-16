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

(* A machine with a stack of exception handlers and of calls, and what the
   discipline of trap actions buys: on a unit that follows it, and whose
   binders are distinct, this machine and the machine of
   FlambdaSemanticsSpec.v behave alike.

   This file is to be reviewed: [stack_entry], [stack_run_call],
   [unwind], [stack_enter], [stack_call], [stack_apply_cont],
   [stack_step], [stack_run],
   [stack_behaves], [initial_stack] and the three statements about the
   two machines, against what the backend does with the terms it is
   given (middle_end/flambda2/to_cmm/to_cmm_expr.ml: translate_raise,
   translate_apply, translate_jump_to_continuation;
   backend/select_utils.ml: combine_traps, set_traps,
   Stack_offset_and_exn.process_basic; backend/ssa/ssa.ml:
   compute_block_end_trap_stack);
   [raises_as_marked] against to_cmm's test of what is a raise
   (to_cmm_expr.ml: apply_cont, let_cont, let_cont_not_inlined;
   to_cmm_env.ml: create, add_exn_handler, is_exn_handler), with its
   two statements; and the action of renamings on a stack
   ([rename_stack_entry]), with its four statements.

   The machine of FlambdaSemanticsSpec.v goes by the names in a term: a
   raise goes to the continuation it names, and an exception of a call
   to the exception continuation the call names; it does not execute
   trap actions. The code that the backend makes goes by a stack. The
   machine here is that reading of a term, written from to_cmm and from
   the selection of instructions; that the backend's code behaves as
   this machine does is not stated here, and nothing here proves it:
   - a [Push] installs a handler: it puts on the stack what the
     continuation it names stands for. A [Pop] removes the entry on
     top, whatever it names. (The backend stops, when it compiles, on
     a [Pop] of a jump or of a return that names another handler than
     the one on top: backend/select_utils.ml,
     [Stack_offset_and_exn.process_basic]; backend/ssa/ssa.ml,
     [compute_block_end_trap_stack]. The machine here does not check
     the handler that a [Pop] names; [traps_well_formed] checks it
     against the top);
   - a raise (FlambdaModel.is_raise) removes the entry on top and goes
     to it, whatever continuation the raise names;
   - a call of code with as many arguments as it has parameters
     (FlambdaSemanticsSpec.shape_of_call says what a call is) puts on
     the stack the return continuation of the caller,
     with the extra arguments of the call, and above it a handler that
     stands for the exception continuation of the code called, which is
     how the body of code starts: with one handler installed. The body
     runs with its two continuations bound to nothing, so that applying
     either leaves the body ([Exit]): by the return continuation, the
     call returns to what the stack holds; by the exception
     continuation, the exception is raised again in the caller, with
     the extra arguments of the call after it, to the handler that is
     then on top. The exception continuation that the call names is
     not looked at (To_cmm_expr.translate_apply does not look at it
     either, and wraps the call in a handler that adds the extra
     arguments);
   - a call that makes a partial application puts nothing on the stack:
     the partial application is given to the return continuation of the
     call. A call with more arguments than the code has parameters is
     the call of the code with the first of them, as above, but the
     stack keeps the arguments that remain ([Call_then]); when the body
     returns, its result is called with those, under the stack as it
     was at the call, for the return continuation and with the extra
     arguments of the whole call. So a raise in either of the two calls
     goes to the caller's handler with the extra arguments. That is the
     backend's path for an indirect call of unknown arity: the compiled
     program makes the two calls in one generic application
     (caml_applyN), inside the handler that To_cmm_expr.translate_apply
     wraps it in. The machine reads a direct call and a call of known
     arity in the same way, which the backend does not: it refuses the
     one and compiles the other as a full call (discrepancy C3-9), and
     nothing is said here of what the compiled program does with
     those;
   - a unit starts with one handler installed, which stands for its
     exception continuation, and ends when it applies a continuation it
     does not bind.
   Everything else is the step of FlambdaSemanticsSpec.v.

   Choices, where the backend's code would be in no state that a term
   describes; none matters to a term that follows the discipline:
   - a [Pop] or a raise with no handler on top of the stack is
     undefined, and so is a return with a handler still installed above
     the one the body started with;
   - which [Apply_cont] is a raise is said by the term
     (FlambdaModel.is_raise: it pops the continuation that it applies).
     to_cmm takes an [Apply_cont] for a raise when the continuation it
     applies is an exception handler, and stops if it has no [Pop]
     (discrepancy C5-2). The two agree on a unit whose exception
     handlers are apart from its other continuations
     ([raises_as_marked_statement]).

   [stack_agrees_statement]: a unit whose binders are distinct
   (FlambdaBindersSpec.binders_distinct) and that follows the discipline
   of trap actions (FlambdaWellFormedSpec.traps_well_formed) has the
   same behaviours on the two machines of the model. Both are conditions
   of a well-formed unit. The first is needed: the check of trap
   actions is for such a term, and without it lets through code whose
   return and exception continuations are one name, on which the two
   machines differ ([stack_needs_distinct_binders_statement]).
   [stack_agrees_from_statement] is the same of a unit run from a
   static environment that holds no code, as that of the symbols that
   a unit imports is.
   [stack_differs_statement]: without the discipline they differ; the
   term is one where a raise names a handler other than the one
   installed last (discrepancy C5-1), and its binders are distinct.

   What the statements do not say. They are about the two machines of
   the model, not about the backend. One part of the step from here to
   the backend's code is made: this machine takes for a raise what the
   term says is one, where to_cmm goes by the mark on the handler
   applied, and in a unit that passes the other check
   (FlambdaWellFormedSpec.unit_exn_handlers_well_formed) the two take
   the same [Apply_cont]s for raises ([raises_as_marked_statement]; not
   without the check, [raises_not_as_marked_statement]). The rest is
   not made: what to_cmm does with a raise, a call or a trap action is
   not modelled.

   This machine commutes with renaming, as the other does
   ([stack_step_equivariant_statement] and the three after it;
   FlambdaEquivarianceSpec.v has the other's). A stack holds names (the
   two continuations of the code of a call in progress) and
   continuation values, and the action renames those.

   The proofs are in FlambdaTrapProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaEquivarianceSpec FlambdaWellFormedSpec.
Import ListNotations.

(* ======================================================================== *)
(* The stack                                                                *)
(* ======================================================================== *)

(* What the stack holds, the last put on it first. [Trap] is an
   exception handler installed. [Call] is a call in progress: the return
   and exception continuations of the code called, what the return
   continuation of the call stands for in the caller, and the values of
   the extra arguments of the call. [Call_then] is the first call of an
   over-application in progress: the same, with the arguments that
   remain, the arity that they are given at and the arity of the result
   of the whole call, as FlambdaSemanticsSpec.Over_application keeps
   them. *)
Inductive stack_entry : Type :=
  | Trap (c : cont_value)
  | Call (ret exn : continuation) (return_to : cont_value)
      (extra : list value)
  | Call_then (ret exn : continuation) (remaining : list value)
      (remaining_arity return_arity : arity) (return_to : cont_value)
      (extra : list value).

Definition stack : Type := list stack_entry.

(* What one step comes to, as FlambdaSemanticsSpec.step_result, with the
   stack. *)
Inductive stack_result : Type :=
  | Stack_step (s : state) (st : stack)
  | Stack_exit (k : continuation) (vs : list value) (hp : heap)
  | Stack_undefined.

(* How control leaves what is running: by applying the continuation [k],
   which stands for nothing there, or by a raise. *)
Inductive leaving : Type :=
  | Leaving_by (k : continuation)
  | Raising.

(* A call of the shape given, made under the stack [st]; [return_to] is
   what its return continuation stands for and [extra] the values of its
   extra arguments. The body of the code runs as
   FlambdaSemanticsSpec.call_code has it, but with its return and
   exception continuations standing for nothing; the stack has the
   call, and the handler that the body starts with. For an
   over-application the call is that of the first arguments, made at
   the result arity of the code as FlambdaSemanticsSpec.run_call makes
   it, and the stack keeps what remains. [return_partial] is what is
   done with a partial application. *)
Definition stack_run_call (return_partial : value -> stack_result)
    (shape : option call_shape) (return_arity : arity)
    (return_to : cont_value) (extra : list value) (statics : static_env)
      (hp : heap) (rd : read_record)
    (st : stack) : stack_result :=
  match shape with
  | Some (Full_call closure c vs args_arity) =>
    let ret := code_return_continuation c in
    let exn := code_exn_continuation c in
    match
      call_code closure c vs args_arity return_arity (Outer ret) (Outer exn)
        statics hp rd
    with
    | Step s =>
      Stack_step s (Trap (Outer exn) :: Call ret exn return_to extra :: st)
    | Exit _ _ _ _ | Undefined => Stack_undefined
    end
  | Some (Over_call closure c vs remaining args_arity remaining_arity) =>
    let ret := code_return_continuation c in
    let exn := code_exn_continuation c in
    match
      call_code closure c vs args_arity (code_result_arity c) (Outer ret)
        (Outer exn) statics hp rd
    with
    | Step s =>
      Stack_step s
        (Trap (Outer exn)
         :: Call_then ret exn remaining remaining_arity return_arity
              return_to extra
         :: st)
    | Exit _ _ _ _ | Undefined => Stack_undefined
    end
  | Some (Partial_call partial) => return_partial partial
  | None => Stack_undefined
  end.

(* Leaving with the values [vs], under the stack [st].
   - A raise goes to the handler on top. If that stands for nothing, as
     the handler that the body of code starts with does, the raise
     leaves by it.
   - Leaving the unit, which is when no call is in progress, ends
     execution, if at most the handler that it started with is
     installed.
   - Leaving the body of code by its return continuation, with the
     handler it started with on top, returns to the caller.
   - Leaving the body of the first call of an over-application by its
     return continuation, with that handler on top and one value, calls
     the value with the arguments that remain, under the stack below
     the call; a partial application that this makes is given to the
     return continuation of the whole call.
   - Leaving the body by its exception continuation, that handler
     removed, raises in the caller, with the extra arguments of the
     call. *)
(* CR mshinwell: Where the stack is not what a term that follows the
   discipline would have, this machine is undefined: a [Pop] or a raise
   with no handler on top, a return or the end of the unit with a
   handler still installed, the body of code left by a continuation
   that is neither of its two, the first call of an over-application
   that returns other than one value. The code that the backend makes
   would do
   something there that no term describes (a [Pop] removes whatever is
   on top, a raise goes to whatever handler is installed, of whatever
   call), and to_cmm stops on some of these terms when it visits them.
   So undefined is a choice. Nothing proved depends on it:
   [stack_agrees_statement] is about units on which none of this
   happens. Discrepancies C5-1 and C5-2. *)
Fixpoint unwind (st : stack) (how : leaving) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record) : stack_result :=
  match how, st with
  | Raising, Trap c :: rest =>
    match apply_cont_value c vs statics hp rd with
    | Step s => Stack_step s rest
    | Exit k vs hp rd => unwind rest (Leaving_by k) vs statics hp rd
    | Undefined => Stack_undefined
    end
  | Raising, ([] | Call _ _ _ _ :: _ | Call_then _ _ _ _ _ _ _ :: _) =>
    Stack_undefined
  | Leaving_by k, ([] | [Trap _]) => Stack_exit k vs hp
  | Leaving_by k, Trap _ :: Call ret _ return_to _ :: rest =>
    if Pos.eqb k ret
    then
      match apply_cont_value return_to vs statics hp rd with
      | Step s => Stack_step s rest
      | Exit k vs hp rd => unwind rest (Leaving_by k) vs statics hp rd
      | Undefined => Stack_undefined
      end
    else Stack_undefined
  | Leaving_by k,
    Trap _
    :: Call_then ret _ remaining remaining_arity return_arity return_to extra
    :: rest =>
    if Pos.eqb k ret
    then
      match vs with
      | [v] =>
        stack_run_call
          (fun partial =>
             match apply_cont_value return_to [partial] statics hp rd with
             | Step s => Stack_step s rest
             | Exit k vs hp rd => unwind rest (Leaving_by k) vs statics hp rd
             | Undefined => Stack_undefined
             end)
          (shape_of_call statics v Indirect remaining remaining_arity
             return_arity)
          return_arity return_to extra statics hp rd rest
      | _ => Stack_undefined
      end
    else Stack_undefined
  | Leaving_by k, Trap _ :: Trap _ :: _ => Stack_undefined
  | Leaving_by k,
    (Call _ exn _ extra | Call_then _ exn _ _ _ _ extra) :: rest =>
    if Pos.eqb k exn
    then unwind rest Raising (vs ++ extra) statics hp rd
    else Stack_undefined
  end.

(* Going to what a continuation stands for, under the stack [st]. *)
Definition stack_enter (c : cont_value) (vs : list value)
    (statics : static_env) (hp : heap) (rd : read_record)
      (st : stack) : stack_result :=
  match apply_cont_value c vs statics hp rd with
  | Step s => Stack_step s st
  | Exit k vs hp rd => unwind st (Leaving_by k) vs statics hp rd
  | Undefined => Stack_undefined
  end.

(* ======================================================================== *)
(* Steps                                                                    *)
(* ======================================================================== *)

(* A call: what FlambdaSemanticsSpec.shape_of_call says it is, made
   under the stack. A partial application is given to the return
   continuation of the call, and nothing is put on the stack. *)
Definition stack_call (ap : apply_expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record)
      (st : stack) : stack_result :=
  match
    eval_simple vars (ap_callee ap), eval_simples vars (ap_args ap),
    eval_simples vars (exn_extra_simples ap)
  with
  | Some callee, Some vs, Some extra =>
    let return_to := resolve (ap_continuation ap) conts in
    stack_run_call
      (fun partial => stack_enter return_to [partial] statics hp rd st)
      (shape_of_call statics callee (ap_call_kind ap) vs (ap_args_arity ap)
         (ap_return_arity ap))
      (ap_return_arity ap) return_to extra statics hp rd st
  | _, _, _ => Stack_undefined
  end.

(* An [Apply_cont]: its trap action is performed, and then it goes to
   its continuation; or it is a raise, which goes to the handler on top
   of the stack. *)
Definition stack_apply_cont (ac : apply_cont) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (st : stack) : stack_result :=
  match eval_simples vars (ac_args ac) with
  | Some vs =>
    let c := resolve (ac_continuation ac) conts in
    match ac_trap_action ac with
    | None => stack_enter c vs statics hp rd st
    | Some (Push exn_handler) =>
      stack_enter c vs statics hp rd (Trap (resolve exn_handler conts) :: st)
    | Some (Pop _ _) =>
      if is_raise ac then unwind st Raising vs statics hp rd
      else
        match st with
        | Trap _ :: rest => stack_enter c vs statics hp rd rest
        | [] | Call _ _ _ _ :: _ | Call_then _ _ _ _ _ _ _ :: _ =>
          Stack_undefined
        end
    end
  | None => Stack_undefined
  end.

(* One step. The constructs other than [Apply] and [Apply_cont] step as
   in FlambdaSemanticsSpec.v; none of them ends execution. *)
Definition stack_step (s : state) (st : stack) : stack_result :=
  match state_expr s with
  | Apply ap =>
    stack_call ap (state_vars s) (state_conts s) (state_statics s)
      (state_heap s) (state_reads s) st
  | Apply_cont ac =>
    stack_apply_cont ac (state_vars s) (state_conts s) (state_statics s)
      (state_heap s) (state_reads s) st
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Switch _ _ | Invalid =>
    match step s with
    | Step s' => Stack_step s' st
    | Exit _ _ _ _ | Undefined => Stack_undefined
    end
  end.

(* At most [fuel] steps. *)
Fixpoint stack_run (fuel : nat) (s : state) (st : stack) : stack_result :=
  match fuel with
  | O => Stack_step s st
  | S fuel =>
    match stack_step s st with
    | Stack_step s' st' => stack_run fuel s' st'
    | result => result
    end
  end.

(* The behaviours, as FlambdaSemanticsSpec.behaves. *)
Definition stack_behaves (s : state) (st : stack) (b : behaviour) : Prop :=
  match b with
  | Exits k vs hp => ∃ fuel : nat, stack_run fuel s st = Stack_exit k vs hp
  | Goes_wrong => ∃ fuel : nat, stack_run fuel s st = Stack_undefined
  | Diverges =>
    ∀ fuel : nat,
      ∃ (s' : state) (st' : stack), stack_run fuel s st = Stack_step s' st'
  end.

(* A unit starts with one handler installed: its exception
   continuation. *)
Definition initial_stack (unit_exn : continuation) : stack :=
  [Trap (Outer unit_exn)].

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit whose binders are distinct and that follows the discipline of
   trap actions, run on its own, behaves on the machine with the stack
   as it does on the machine that goes by names. *)
Definition stack_agrees_statement : Prop :=
  ∀ (unit_exn : continuation) (e : expr) (b : behaviour),
    binders_distinct e = true ->
    traps_well_formed unit_exn e = true ->
    stack_behaves (State e [] [] empty_static_env empty_heap no_reads)
      (initial_stack unit_exn)
      b
    <-> behaves (State e [] [] empty_static_env empty_heap no_reads) b.

(* The same of a unit run from a static environment that holds no
   code, whatever it binds symbols to: that of the symbols that a unit
   imports is one (FlambdaImportsSpec.imports_statics). With the empty
   static environment it is the statement above. Code that is bound
   before the unit starts is left out: nothing is asked here of its
   binders or of its trap actions. *)
Definition stack_agrees_from_statement : Prop :=
  ∀ (statics : static_env) (unit_exn : continuation) (e : expr)
    (b : behaviour),
    static_code statics = [] ->
    binders_distinct e = true ->
    traps_well_formed unit_exn e = true ->
    stack_behaves (State e [] [] statics empty_heap no_reads)
      (initial_stack unit_exn)
      b
    <-> behaves (State e [] [] statics empty_heap no_reads) b.

(* Two handlers are installed, [outer] then [inner], and the body raises
   to [outer]. By the names the raise goes to [outer], which leaves with
   2. By the stack it goes to [inner], which removes [outer] and leaves
   with 1. *)
Definition raise_over_handler : expr :=
  let outer : continuation := 3%positive in
  let inner : continuation := 4%positive in
  let body : continuation := 5%positive in
  let leave : continuation := 2%positive in
  let x : variable := 1%positive in
  let leave_with (i : Z) (action : option trap_action) : expr :=
    Apply_cont
      (Apply_cont_expr leave [Const (Tagged_immediate i)] action)
  in
  let catches (e : expr) : continuation_handler :=
    Handler [x] (make_exn_handler_metadata [Value]) e
  in
  let enters (action : trap_action) : expr :=
    Apply_cont (Apply_cont_expr body [] (Some action))
  in
  Let_cont_non_recursive outer (catches (leave_with 2%Z None))
    (Let_cont_non_recursive body
       (Handler [] (make_handler_metadata [])
          (Let_cont_non_recursive inner
             (catches (leave_with 1%Z (Some (Pop outer None))))
             (Let_cont_non_recursive body
                (Handler [] (make_handler_metadata [])
                   (Apply_cont
                      (Apply_cont_expr outer [Const (Tagged_immediate 0%Z)]
                         (Some (Pop outer (Some No_trace))))))
                (enters (Push inner)))))
       (enters (Push outer))).

(* Without the discipline the two machines differ, on a term whose
   binders are distinct. *)
Definition stack_differs_statement : Prop :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let s := State raise_over_handler [] [] empty_static_env empty_heap no_reads
    in
  binders_distinct raise_over_handler = true
  ∧ traps_well_formed unit_exn raise_over_handler = false
  ∧ behaves s (Exits leave [Value_const (Tagged_immediate 2%Z)] empty_heap)
  ∧ stack_behaves s (initial_stack unit_exn)
      (Exits leave [Value_const (Tagged_immediate 1%Z)] empty_heap).

(* Code whose return and exception continuations are one name, called by
   a unit. The body of the code raises. By the names the raise goes to
   what the name stands for in the body, which is the return
   continuation of the call: the unit leaves with 1. By the stack it
   goes to the handler installed, which is the caller's: the unit
   leaves by its exception continuation, with the exception. *)
Definition raise_in_code_of_one_continuation : expr :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let after : continuation := 3%positive in
  let both : continuation := 4%positive in
  let f : variable := 1%positive in
  let x : variable := 2%positive in
  let id : code_id := 1%positive in
  Let_static
    [Bind_code id
       (Code [] f both both (make_code_metadata [] [Singleton Value])
          (Apply_cont
             (Apply_cont_expr both [Const (Tagged_immediate 0%Z)]
                (Some (Pop both (Some No_trace))))))]
    (Let f Value (Set_of_closures id [])
       (Let_cont_non_recursive after
          (Handler [x] (make_handler_metadata [Value])
             (Apply_cont
                (Apply_cont_expr leave [Const (Tagged_immediate 1%Z)] None)))
          (Apply
             (Apply_expr (Var f) (Direct id) [] after unit_exn [] []
                [Singleton Value])))).

(* The discipline alone is not enough: this unit follows it, its binders
   are not distinct, and the two machines differ on it. *)
Definition stack_needs_distinct_binders_statement : Prop :=
  let unit_exn : continuation := 1%positive in
  let leave : continuation := 2%positive in
  let s := State raise_in_code_of_one_continuation [] [] empty_static_env
    empty_heap no_reads in
  binders_distinct raise_in_code_of_one_continuation = false
  ∧ traps_well_formed unit_exn raise_in_code_of_one_continuation = true
  ∧ behaves s (Exits leave [Value_const (Tagged_immediate 1%Z)] empty_heap)
  ∧ stack_behaves s (initial_stack unit_exn)
      (Exits unit_exn [Value_const (Tagged_immediate 0%Z)] empty_heap).
(* ======================================================================== *)
(* Which [Apply_cont] is a raise                                            *)
(* ======================================================================== *)

(* The two machines take an [Apply_cont] for a raise when the term says
   that it is one: it pops the continuation that it applies
   (FlambdaModel.is_raise, the compiler's [Apply_cont_expr.is_raise],
   which is what Simplify goes by). to_cmm goes by the continuation
   applied: an [Apply_cont], alone or as the arm of a [Switch], is a
   raise when that continuation is one of the exception handlers in
   scope (To_cmm_expr.apply_cont, by To_cmm_env.is_exn_handler). Those
   are the exception continuation of the function, or of the unit, that
   is being translated (To_cmm_env.create) and the continuations bound
   since by a non-recursive [Let_cont] whose handler says that it is an
   exception handler, in the body of that [Let_cont]
   (To_cmm_expr.let_cont_not_inlined, by To_cmm_env.add_exn_handler).
   to_cmm stops on a recursive [Let_cont] with such a handler, and never
   takes a name out of the set: the compiler binds a name once.

   [raises_as_marked exn_handlers e]: with [exn_handlers] the exception
   handlers in scope, the two readings take the same [Apply_cont]s of
   [e] for raises. *)
Definition raise_as_marked (exn_handlers : list continuation)
    (ac : apply_cont) : bool :=
  Bool.eqb (is_raise ac) (among (ac_continuation ac) exn_handlers).

Fixpoint raises_as_marked (exn_handlers : list continuation) (e : expr)
    : bool :=
  match e with
  | Let _ _ _ body => raises_as_marked exn_handlers body
  | Let_static bindings body =>
    forallb binding_raises_as_marked bindings
    && raises_as_marked exn_handlers body
  | Let_cont_non_recursive k h body =>
    handler_raises_as_marked exn_handlers h
    && raises_as_marked
         (if handler_is_exn_handler h then k :: exn_handlers
          else exn_handlers)
         body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_raises_as_marked exn_handlers h)
      handlers
    && raises_as_marked exn_handlers body
  | Apply_cont ac => raise_as_marked exn_handlers ac
  | Switch _ arms =>
    forallb (fun arm => raise_as_marked exn_handlers (snd arm)) arms
  | Apply _ | Invalid => true
  end
with handler_raises_as_marked (exn_handlers : list continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => raises_as_marked exn_handlers handler
  end
with binding_raises_as_marked (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_raises_as_marked c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_raises_as_marked (c : code) : bool :=
  match c with
  | Code _ _ _ exn _ body => raises_as_marked [exn] body
  end.

(* In a unit whose exception handlers are apart from its other
   continuations, an [Apply_cont] is a raise for to_cmm exactly when the
   term says that it is one. *)
Definition raises_as_marked_statement : Prop :=
  ∀ (unit_exn : continuation) (e : expr),
    unit_exn_handlers_well_formed unit_exn e = true ->
    raises_as_marked [unit_exn] e = true.

(* An exception handler that is applied with no trap action: to_cmm takes
   the [Apply_cont] for a raise, and the term says that it is a jump. *)
Definition handler_applied_as_jump : expr :=
  let leave : continuation := 2%positive in
  let handler : continuation := 3%positive in
  let x : variable := 1%positive in
  Let_cont_non_recursive handler
    (Handler [x] (make_exn_handler_metadata [Value])
       (Apply_cont (Apply_cont_expr leave [] None)))
    (Apply_cont
       (Apply_cont_expr handler [Const (Tagged_immediate 0%Z)] None)).

(* Without the check the two readings differ. *)
Definition raises_not_as_marked_statement : Prop :=
  let unit_exn : continuation := 1%positive in
  unit_exn_handlers_well_formed unit_exn handler_applied_as_jump = false
  ∧ raises_as_marked [unit_exn] handler_applied_as_jump = false.

(* ======================================================================== *)
(* The machine with the stack is equivariant                                *)
(* ======================================================================== *)

(* The action of renamings on a stack; FlambdaEquivarianceSpec.v has it
   on states. A handler installed, and the continuation that a call
   returns to, are continuation values, and are renamed as such. The two
   continuations of the code of a call in progress are names, and are
   renamed. The extra arguments of a call, and the arguments that
   remain of an over-application, are values, which hold no variables
   or continuations. *)
Definition rename_stack_entry (r : renaming) (entry : stack_entry)
    : stack_entry :=
  match entry with
  | Trap c => Trap (rename_cont_value r c)
  | Call ret exn return_to extra =>
    Call (rename_continuation r ret) (rename_continuation r exn)
      (rename_cont_value r return_to) extra
  | Call_then ret exn remaining remaining_arity return_arity return_to
      extra =>
    Call_then (rename_continuation r ret) (rename_continuation r exn)
      remaining remaining_arity return_arity (rename_cont_value r return_to)
      extra
  end.

Definition rename_stack (r : renaming) (st : stack) : stack :=
  map (rename_stack_entry r) st.

Definition rename_stack_result (r : renaming) (result : stack_result)
    : stack_result :=
  match result with
  | Stack_step s st => Stack_step (rename_state r s) (rename_stack r st)
  | Stack_exit k vs hp => Stack_exit (rename_continuation r k) vs hp
  | Stack_undefined => Stack_undefined
  end.

(* A step, a run and the behaviours of this machine commute with
   renaming, as those of the other machine do
   (FlambdaEquivarianceSpec.step_equivariant_statement and the two after
   it). As there, nothing is asked of the state. *)
Definition stack_step_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state) (st : stack),
    stack_step (rename_state r s) (rename_stack r st)
    = rename_stack_result r (stack_step s st).

Definition stack_run_equivariant_statement : Prop :=
  ∀ (r : renaming) (fuel : nat) (s : state) (st : stack),
    stack_run fuel (rename_state r s) (rename_stack r st)
    = rename_stack_result r (stack_run fuel s st).

Definition stack_behaves_equivariant_statement : Prop :=
  ∀ (r : renaming) (s : state) (st : stack) (b : behaviour),
    stack_behaves (rename_state r s) (rename_stack r st)
      (rename_behaviour r b)
    <-> stack_behaves s st b.

(* The stack that a unit starts with, renamed, is the stack that the
   unit starts with when its exception continuation is renamed. *)
Definition initial_stack_equivariant_statement : Prop :=
  ∀ (r : renaming) (unit_exn : continuation),
    rename_stack r (initial_stack unit_exn)
    = initial_stack (rename_continuation r unit_exn).
