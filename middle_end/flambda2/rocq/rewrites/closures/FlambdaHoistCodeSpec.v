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

(* The code of a unit bound first: a normalisation of a unit that is run
   on its own.

   This file is to be reviewed: [code_bindings], [without_code],
   [closures_in_scope] and [hoist_code], which are extracted and run by
   the validation of middle_end/flambda2/tests/semantics/, and the
   statements.

   The two terms of a unit do not bind code in the same places
   (FlambdaFloatCodeSpec.v says where each does). A function defined
   after a call that is not inlined, or after a branch that is not
   resolved, has its closure lifted by Simplify, and its new code bound,
   in the handler of the continuation that the call or the branch goes
   to; the conversion from Lambda has bound the code of the unit at its
   head. FlambdaFloatCodeSpec.float_code takes code to the head of the
   handler that it is in and no further.

   [hoist_code] binds all the code of the toplevel of a unit first: the
   bindings of code of its groups of static bindings, in handlers too
   ([code_bindings]), in one group around the unit with those bindings
   removed ([without_code]). Bindings of deleted code, which bind
   nothing, stay where they are, for
   FlambdaNormaliseSpec.drop_deleted_code to remove. The code of
   functions and the handlers of recursive continuations are not looked
   into for code to take out: the machine has no step for a call of
   code, or for a recursive [Let_cont], that binds something static.

   This is no equivalence under all environments, and not one of every
   unit run on its own. With its code bound first a unit may be defined
   where it was not: where a closure is called before its code is
   bound. And it may be undefined where it was defined: a code id that
   is bound a second time in a handler that is never entered is bound
   twice in the one group. So three things are checked of the unit,
   and no code is moved if one fails ([hoist_code] answers [None]).
   - Its static bindings are at toplevel
     (FlambdaWellFormedSpec.statics_at_unit_toplevel, for one of its free
     continuations as the exception continuation of the unit), and bind
     each code id once (FlambdaWellFormedSpec.static_names, of the code
     ids). These are the first condition of
     FlambdaWellFormedSpec.well_formed_unit and, of the second, the part
     about code ids: that each symbol is bound once is not asked, as no
     binding of a symbol is moved. They are what the invariant of
     FlambdaWellFormedProofs.v asks, taken for code ids, to say that no
     code id is bound twice.
   - The code that is bound first has distinct code ids. This follows
     from the condition before; it is asked again, of the code in the
     order in which it is taken out, so that the proof need not relate
     the two orders.
   - Every closure is made of code that is bound by then
     ([closures_in_scope]): a [Let] or a static binding that makes a
     closure is under the binding of the code of each function of its
     set, or in the same group. Then no closure of code that is not yet
     bound exists, and none is called.

   A unit that passes is equivalent, run on its own, to the unit with
   its code first ([hoist_code_program_equivalent_statement]): the two
   runs are in step, with the same values, and differ only in when the
   code is bound.

   The proofs are in FlambdaHoistCodeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaImportsSpec FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The code of the toplevel                                                 *)
(* ======================================================================== *)

Definition is_code (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ _ => true
  | Bind_deleted_code _ | Bind_symbol _ _ => false
  end.

(* The bindings of code of the groups of static bindings of an
   expression, those of the handler of a [Let_cont] after those of its
   body, as a run comes to them. *)
Fixpoint code_bindings (e : expr) : list static_binding :=
  match e with
  | Let _ _ _ body => code_bindings body
  | Let_static bindings body => filter is_code bindings ++ code_bindings body
  | Let_cont_non_recursive _ h body =>
    code_bindings body ++ handler_code_bindings h
  | Let_cont_recursive _ body => code_bindings body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_code_bindings (h : continuation_handler)
    : list static_binding :=
  match h with
  | Handler _ _ handler => code_bindings handler
  end.

(* The expression with those bindings removed. A group left with no
   binding stays. *)
Fixpoint without_code (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (without_code body)
  | Let_static bindings body =>
    Let_static (filter (fun binding => negb (is_code binding)) bindings)
      (without_code body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (handler_without_code h) (without_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (without_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with handler_without_code (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (without_code handler)
  end.

(* ======================================================================== *)
(* Closures are made of code that is bound                                  *)
(* ======================================================================== *)

(* CR mshinwell: this check is here, and not a premise taken from
   well-formedness, because FlambdaWellFormedSpec.well_formed_unit does
   not yet ask that the symbols and the code ids a term mentions are
   bound where they are mentioned (row 4 of the table of
   requests/C0-well-formed.md). When it does, this check and
   [closures_in_scope_run_statement] may give way to that condition and
   its invariant of states, but only if those give what is used here:
   that the code of every closure of a state is bound, so that a call
   finds the same code in both runs. The invariant as it is planned
   (every code id that a closure of a state holds is bound, or was
   declared deleted) is not enough by itself: if a code id that is
   declared deleted could be bound later, then with that code bound
   first a call of a closure made in between, which was undefined,
   would run. So the replacement needs that condition together with
   the one planned beside it, that the static names of a unit are
   distinct with the declarations of deleted code counted (a code id
   declared deleted is then not bound by the unit); this check stays
   until both are there. The check is not
   that condition for code ids. It asks less: of the closures that are
   made, and not of a direct call, which names a code id too (the
   machine has no step for a direct call unless its callee is a closure
   of the code named, so closures are enough here). And it asks more in
   two places: a handler is given only the code that is bound where its
   [Let_cont] is, where that condition also gives a toplevel handler
   the code that the body of its [Let_cont] binds at toplevel; and the
   code of a closure is to be bound, where that condition accepts code
   that is declared deleted. For a unit that fails the check in either
   place no code is moved; if it needed that to be validated, it stays
   not validated. *)

Definition decls_bound (bound : code_id -> bool)
    (decls : list (function_slot * code_id)) : bool :=
  forallb (fun decl => bound (snd decl)) decls.

Definition named_closures_bound (bound : code_id -> bool) (d : named)
    : bool :=
  match d with
  | Set_of_closures code _ => bound code
  | Closure_of_set _ decls _ => decls_bound bound decls
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    true
  end.

Definition const_closures_bound (bound : code_id -> bool)
    (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures code _ => bound code
  | Static_const_closure_of_set _ _ decls _ => decls_bound bound decls
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    true
  end.

(* The code ids given are bound as well. *)
Definition bound_with (ids : list code_id) (bound : code_id -> bool)
    : code_id -> bool :=
  fun id => among id ids || bound id.

(* Every closure that the expression makes, by a [Let] or by a static
   binding, in handlers and in code too, is of code that [bound] has, or
   that a group of static bindings around it binds, or the group that it
   is in. A handler is asked for no more than is bound where its
   [Let_cont] is. *)
Fixpoint closures_in_scope (bound : code_id -> bool) (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    named_closures_bound bound d && closures_in_scope bound body
  | Let_static bindings body =>
    let bound := bound_with (flat_map code_ids_of_binding bindings) bound in
    forallb (binding_closures_in_scope bound) bindings
    && closures_in_scope bound body
  | Let_cont_non_recursive _ h body =>
    handler_closures_in_scope bound h && closures_in_scope bound body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_closures_in_scope bound h) handlers
    && closures_in_scope bound body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_closures_in_scope (bound : code_id -> bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => closures_in_scope bound handler
  end
with binding_closures_in_scope (bound : code_id -> bool)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_closures_in_scope bound c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const => const_closures_bound bound const
  end
with code_closures_in_scope (bound : code_id -> bool) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => closures_in_scope bound body
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The static bindings of the unit are at toplevel, for one of its free
   continuations as its exception continuation. *)
Definition statics_at_unit_toplevel_of_unit (e : expr) : bool :=
  existsb (fun unit_exn => statics_at_unit_toplevel unit_exn true e)
    (free_continuations e).

(* The unit with its code bound first, if it passes the checks. *)
Definition hoist_code (e : expr) : option expr :=
  let codes := code_bindings e in
  if
    statics_at_unit_toplevel_of_unit e
    && distinct (static_names code_ids_of_binding e)
    && distinct (flat_map code_ids_of_binding codes)
    && closures_in_scope (fun _ => false) e
  then Some (Let_static codes (without_code e))
  else None.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit that passes and the unit with its code first, each run on its
   own from empty environments, behave alike. *)
Definition hoist_code_program_equivalent_statement : Prop :=
  ∀ e e' : expr,
    hoist_code e = Some e' ->
    equivalent (State e [] [] empty_static_env empty_heap no_reads)
      (State e' [] [] empty_static_env empty_heap no_reads).

(* The same of a unit that imports symbols: the two, each run from the
   static environment of its imports, behave alike. Nothing more is
   asked: that environment holds no code, so no code id of the unit is
   bound in it, and the pass moves code only. *)
Definition hoist_code_equivalent_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (e e' : expr),
    hoist_code e = Some e' ->
    equivalent (State e [] [] (imports_statics imports) empty_heap no_reads)
      (State e' [] [] (imports_statics imports) empty_heap no_reads).

(* The same of the two units followed by something: run from one
   continuation environment of what may follow a unit
   (FlambdaObserversSpec.observers: a way out, or a handler with no
   environment of its own that binds nothing static and names no
   symbol and no code id), the two behave alike. With no continuation
   it is the statement above. What may follow names no code id, so it
   makes no closure of code that the unit has not bound yet; it may
   call a closure that the unit gave it, whose code the unit has
   bound. *)
Definition hoist_code_equivalent_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (conts : cont_env)
    (e e' : expr),
    hoist_code e = Some e' ->
    observers [] conts ->
    equivalent (State e [] conts (imports_statics imports) empty_heap no_reads)
      (State e' [] conts (imports_statics imports) empty_heap no_reads).

(* What the check of closures is for. The closures in a value, or in
   what a symbol is bound to, are of code that [bound] has: every
   function of the set of each. *)
Fixpoint value_closures_bound (bound : code_id -> bool) (v : value) : bool :=
  match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _
  | Value_location _ =>
    true
  | Value_closure code slots =>
    bound code
    && forallb (fun slot => value_closures_bound bound (snd slot)) slots
  | Value_closure_of_set _ decls slots =>
    decls_bound bound decls
    && forallb (fun slot => value_closures_bound bound (snd slot)) slots
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_closures_bound bound) fields
  | Value_partial_application callee args _ =>
    value_closures_bound bound callee
    && forallb (value_closures_bound bound) args
  end.

Definition allocated_closures_bound (bound : code_id -> bool)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code slots =>
    bound code
    && forallb (fun slot => value_closures_bound bound (snd slot)) slots
  | Static_closure_of_set _ _ decls slots =>
    decls_bound bound decls
    && forallb (fun slot => value_closures_bound bound (snd slot)) slots
  | Static_block _ fields => forallb (value_closures_bound bound) fields
  | Static_immutable_array elements =>
    forallb (value_closures_bound bound) elements
  | Static_boxed_number contents => value_closures_bound bound contents
  | Static_string _ | Static_imported _ => true
  end.

(* Every closure that the state holds, in a variable, in what a
   continuation keeps, in a cell of its heap or in what a symbol is
   bound to, is of code that is bound. *)
Definition closures_bound (s : state) : Prop :=
  state_values_all
    (fun statics v => value_closures_bound (code_id_bound statics) v = true)
    (fun statics allocated =>
       allocated_closures_bound (code_id_bound statics) allocated = true)
    s.

(* No state of the run of a unit that passes the check holds a closure
   of code that is not bound: so none is called. *)
Definition closures_in_scope_run_statement : Prop :=
  ∀ (e : expr) (n : nat) (s : state),
    closures_in_scope (fun _ => false) e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    closures_bound s.

(* The parts of the pass commute with renaming, which renames no code
   id; and the check of closures reads no variable and no
   continuation. *)
Definition code_bindings_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    code_bindings (rename_expr r e)
    = map (rename_static_binding r) (code_bindings e).

Definition without_code_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (without_code e) = without_code (rename_expr r e).

Definition closures_in_scope_equivariant_statement : Prop :=
  ∀ (r : renaming) (bound : code_id -> bool) (e : expr),
    closures_in_scope bound (rename_expr r e) = closures_in_scope bound e.
