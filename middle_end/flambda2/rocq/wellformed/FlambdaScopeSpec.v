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

(* What a run keeps of scope: a state in which every variable and every
   continuation that can be used is bound, and all code is closed.

   This file is to be reviewed: [cont_in_scope] and [state_in_scope]
   against [step] (FlambdaSemanticsSpec.v), [variables_read] against the
   evaluations that [step] makes, and the statements.

   A unit that passes rows 3 and 6 of well-formedness
   (FlambdaBindersSpec.names_in_scope and [code_closed]) uses no
   variable and no continuation that is not bound around the use. The
   machine does not run a unit as a whole: it runs a state, in which
   part of the unit is the expression, part is in the handlers that the
   continuations keep, and part is the code of the static environment.
   [state_in_scope] is the condition on a state that a step keeps
   ([step_in_scope_statement]) and that the first state of such a unit
   meets ([unit_in_scope_statement]). It says, with [outer] the
   continuations by which the run may leave:
   - of the expression: its free variables have values, its free
     continuations are bound or are among [outer], and its code is
     closed;
   - of each handler that a continuation keeps, and of the handlers of
     its recursive group, the same under the environments kept: the
     free variables are bound by the environment kept, the free
     continuations by the group, by the continuation environment kept
     or by [outer]; and so on through the continuations kept;
   - of a way out ([Outer k]) that a continuation environment holds: [k]
     is among [outer];
   - of the static environment: the code it binds is closed.

   What follows of a state in scope:
   - its step reads no variable that has no value
     ([in_scope_reads_bound_statement]). This is where closed code is
     needed: a call runs the body of code under its parameters and
     [my_closure] and nothing of where the code was bound;
   - if its step is an exit, the exit is by one of [outer]
     ([step_in_scope_statement]). A continuation that is not bound is a
     way out in the machine and not a fault, so this is what "no
     continuation is used unbound" comes to.
   [unit_run_statement] is the two for the run of a unit.

   What this is not about:
   - symbols and code ids. A state in scope may mention a symbol that
     is not bound, and a closure in it may be of code that is not
     bound; that is row 4 of well-formedness;
   - values. No value holds a variable or a continuation, so the
     condition asks nothing of the values that the environments bind,
     only which names they bind;
   - that the names of a binder are distinct (row 5). The machine
     shadows, and nothing here needs it.

   The proofs are in FlambdaScopeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* States in scope                                                          *)
(* ======================================================================== *)

(* The free variables of the expression are among [vs], its free
   continuations among [ks], and the code bound in it is closed. *)
Definition expr_scoped (vs : list variable) (ks : list continuation)
    (e : expr) : Prop :=
  names_among vs ks e = true ∧ code_closed e = true.

Definition handler_scoped (vs : list variable) (ks : list continuation)
    (h : continuation_handler) : Prop :=
  handler_names_among vs ks h = true ∧ handler_code_closed h = true.

(* A way out is one of [outer]. A closure runs its handler under the
   environment it keeps, with the parameters bound, and with the
   continuations of its recursive group and those it keeps: the handler
   is scoped by these, and so is every handler of the group, since each
   becomes such a closure when this one is applied. The continuations
   kept are in scope in turn. A continuation with extra arguments is in
   scope when the one inside is: the extra arguments are values. What is
   left of an over-application names nothing; the two continuations it
   keeps are in scope. *)
Fixpoint cont_in_scope (outer : list continuation) (c : cont_value) : Prop :=
  match c with
  | Outer k => among k outer = true
  | Extra_args c _ => cont_in_scope outer c
  | Closure h vars conts recursive =>
    let vs := map fst vars in
    let ks := map fst recursive ++ map fst conts ++ outer in
    handler_scoped vs ks h
    ∧ Forall (fun binding => handler_scoped vs ks (snd binding)) recursive
    ∧ fold_right
        (fun binding rest => cont_in_scope outer (snd binding) ∧ rest) True
        conts
  | Over_application _ _ _ return_to exn_to =>
    cont_in_scope outer return_to ∧ cont_in_scope outer exn_to
  end.

Definition conts_in_scope (outer : list continuation) (conts : cont_env)
    : Prop :=
  fold_right (fun binding rest => cont_in_scope outer (snd binding) ∧ rest)
    True conts.

(* A continuation that the expression uses and the continuation
   environment does not bind stands for a way out by that name
   ([resolve]); so it is to be among [outer]. *)
Definition state_in_scope (outer : list continuation) (s : state) : Prop :=
  expr_scoped (map fst (state_vars s)) (map fst (state_conts s) ++ outer)
    (state_expr s)
  ∧ conts_in_scope outer (state_conts s)
  ∧ Forall (fun binding => code_is_closed (snd binding) = true)
      (static_code (state_statics s)).

(* ======================================================================== *)
(* A run stays in scope                                                     *)
(* ======================================================================== *)

(* A unit whose names are in scope and whose code is closed starts in
   scope, with its return and exception continuations as the ways out. *)
Definition unit_in_scope_statement : Prop :=
  ∀ (unit_return unit_exn : continuation) (e : expr),
    names_in_scope unit_return unit_exn e = true ->
    code_closed e = true ->
    state_in_scope [unit_return; unit_exn]
      (State e [] [] empty_static_env empty_heap no_reads).

(* A step keeps a state in scope, and an exit is by one of [outer]. *)
Definition step_in_scope_statement : Prop :=
  ∀ (outer : list continuation) (s : state),
    state_in_scope outer s ->
    match step s with
    | Step s' => state_in_scope outer s'
    | Exit k _ _ _ => among k outer = true
    | Undefined => True
    end.

(* ======================================================================== *)
(* A state in scope reads no variable that has no value                     *)
(* ======================================================================== *)

(* The variables whose values the step of an expression may ask for:
   those of the simples that it evaluates. They are the free variables
   of the defining expression of a [Let], of the static constants of a
   [Let_static] (code is bound, not evaluated), of the callee and the
   arguments of an [Apply], of the arguments of an [Apply_cont], and of
   the scrutinee of a [Switch]. A [Let_cont] reads none. *)
Definition variables_read (e : expr) : list variable :=
  match e with
  | Let _ _ defining_expr _ => free_variables_of_named defining_expr
  | Let_static bindings _ =>
    flat_map
      (fun binding =>
         match binding with
         | Bind_symbol _ const => free_variables_of_static_const const
         | Bind_code _ _ | Bind_deleted_code _ => []
         end)
      bindings
  | Apply ap => free_variables_of_apply ap
  | Apply_cont ac => free_variables_of_apply_cont ac
  | Switch scrutinee _ => free_variables_of_simple scrutinee
  | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _ | Invalid => []
  end.

(* The step of the state may ask for the value of a variable that has
   none: it would then be undefined ([eval_simple] has no value on a
   variable that is not bound, and on nothing else). *)
Definition reads_unbound_variable (s : state) : bool :=
  existsb (fun x => negb (is_bound x (state_vars s)))
    (variables_read (state_expr s)).

Definition in_scope_reads_bound_statement : Prop :=
  ∀ (outer : list continuation) (s : state),
    state_in_scope outer s -> reads_unbound_variable s = false.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

(* A unit whose names are in scope and whose code is closed, run on its
   own: no state of its run reads a variable that has no value, and if
   it exits, that is by its return or its exception continuation. *)
Definition unit_run_statement : Prop :=
  ∀ (unit_return unit_exn : continuation) (e : expr) (n : nat),
    names_in_scope unit_return unit_exn e = true ->
    code_closed e = true ->
    match run n (State e [] [] empty_static_env empty_heap no_reads) with
    | Step s => reads_unbound_variable s = false
    | Exit k _ _ _ => k = unit_return ∨ k = unit_exn
    | Undefined => True
    end.
