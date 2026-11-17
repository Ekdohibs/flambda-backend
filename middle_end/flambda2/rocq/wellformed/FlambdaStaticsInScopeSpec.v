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


(* Every symbol and every code id that a unit mentions is in scope where
   it is mentioned: row 4 of the table of requests/C0-well-formed.md.

   This file is to be reviewed: [statics_in_scope] against the rule of
   scope of section 3 of that note and against the order in which
   Simplify processes a group of static bindings
   (simplify/simplify_static_const.ml), and [static_names_apart].

   The scope of a static name is not that of a variable, and it is not
   read off the text alone. A unit is a sequence of groups of static
   bindings ([Let_static]), each at toplevel (row 1), and the rule is:
   - a symbol that the unit imports is in scope everywhere;
   - in a group, the symbols that the group binds to closures are in
     scope in every definition of the group. Its other symbols are in
     scope in the definitions that follow them, and in the closures and
     the code of the group. The code ids of the group, bound or declared
     deleted, are in scope in every definition of the group;
   - after a group, all it binds or declares deleted is in scope in the
     body;
   - what the body of a [Let_cont] binds is in scope in its handlers,
     recursive or not, at toplevel or not (see below);
   - the body of code sees what is in scope where the code is bound;
   - a code id that is declared deleted is in scope, and is not code:
     [Bind_deleted_code] binds nothing, and a call of a closure of it
     stays undefined where it is executed.

   The clause for the handlers of a [Let_cont] is right only of a unit
   that passes row 1 and [interface_conts_not_rebound], and the check
   is meant with those two. In such a unit a handler binds something
   static only if the body of its [Let_cont] can do nothing but apply
   its continuation or leave the unit by its exception continuation
   (FlambdaWellFormedSpec.handler_at_unit_toplevel). So the run of the body
   of any [Let_cont] binds all that the body binds, down its spine and
   then in each such handler in turn, before it enters a handler of
   that [Let_cont], which it does by applying its continuation or by
   the return of a call to it. The run may also leave the unit by its
   exception continuation before it has bound all of that: row 1
   allows it, and then no handler of the [Let_cont] is entered. What
   the body binds through [Let]s, [Let_static]s and the bodies of
   [Let_cont]s, in no handler, is bound before the body applies
   anything, whatever the unit; the rest needs the two conditions.
   Without row 1 a branch of the body can go round a handler that
   binds a symbol ([branch_round_binding_reads_unbound_statement]);
   without the other a raise that was to leave the unit can
   ([rebound_unit_exn_reads_unbound_statement]).

   Simplify makes such terms, and this is its own discipline: it
   simplifies the body of a [Let_cont] first, and gives each handler an
   environment with the constants that it has lifted meanwhile
   (simplify/join_points.ml, [compute_handler_env], and
   simplify/simplify_let_cont_expr.ml,
   [simplify_single_recursive_handler]: the constants lifted after the
   fork), whether or not the handler is at toplevel. It places those
   constants in the body, under the last [Let] before a test or in the
   handler of a continuation of the body. So a handler may mention a
   symbol that is bound below it in the text: a closure lifted out of
   an arm of a test, in the handler of a test inside that one; or a
   block that the body builds, in the handler of the continuation that
   the body applies last.

   What mentions a static name: a simple that is a symbol; the code of a
   set of closures, made by a [Let] or statically; the symbols that a
   static set gives its closures; and the code that a direct call names.
   A primitive names none.

   [statics_in_scope] is this as a check, with the symbols that the
   unit imports as an argument. [static_names_apart] is what goes with
   it of row 2: no static binding binds an import, and a code id is not
   both bound and declared deleted, nor declared deleted twice. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaScopeSpec
  FlambdaWellFormedSpec.
Import ListNotations.

(* ======================================================================== *)
(* What mentions a static name                                              *)
(* ======================================================================== *)

(* In all of these, [syms] are the symbols in scope and [ids] the code
   ids in scope, bound or declared deleted. *)

Definition simple_statics_in_scope (syms : list symbol) (s : simple) : bool :=
  match s with
  | Symbol sym => among sym syms
  | Var _ | Const _ => true
  end.

Definition simples_statics_in_scope (syms : list symbol) (args : list simple)
    : bool :=
  forallb (simple_statics_in_scope syms) args.

Definition value_slots_statics_in_scope (syms : list symbol)
    (value_slots : list (value_slot * simple)) : bool :=
  forallb (fun slot => simple_statics_in_scope syms (snd slot)) value_slots.

Definition named_statics_in_scope (syms : list symbol) (ids : list code_id)
    (defining_expr : named) : bool :=
  match defining_expr with
  | Simple s => simple_statics_in_scope syms s
  | Prim _ args => simples_statics_in_scope syms args
  | Set_of_closures code value_slots =>
    among code ids && value_slots_statics_in_scope syms value_slots
  | Closure_of_set _ function_decls value_slots =>
    all_among (map snd function_decls) ids
    && value_slots_statics_in_scope syms value_slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    simple_statics_in_scope syms closure
  end.

Definition static_const_statics_in_scope (syms : list symbol)
    (ids : list code_id) (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    among code ids && value_slots_statics_in_scope syms value_slots
  | Static_const_closure_of_set _ closure_symbols function_decls value_slots =>
    all_among (map snd closure_symbols) syms
    && all_among (map snd function_decls) ids
    && value_slots_statics_in_scope syms value_slots
  | Static_const_block _ fields => simples_statics_in_scope syms fields
  | Static_const_immutable_value_array elements =>
    simples_statics_in_scope syms elements
  | Static_const_boxed_number contents =>
    simple_statics_in_scope syms contents
  | Static_const_immutable_string _ => true
  end.

Definition apply_statics_in_scope (syms : list symbol) (ids : list code_id)
    (ap : apply_expr) : bool :=
  simple_statics_in_scope syms (ap_callee ap)
  && match ap_call_kind ap with
     | Direct code => among code ids
     | Indirect_unknown_arity | Indirect_known_arity None => true
     | Indirect_known_arity (Some code_ids) => all_among code_ids ids
     end
  && simples_statics_in_scope syms (ap_args ap)
  && simples_statics_in_scope syms (exn_extra_simples ap).

Definition apply_cont_statics_in_scope (syms : list symbol) (ac : apply_cont)
    : bool :=
  simples_statics_in_scope syms (ac_args ac).

(* ======================================================================== *)
(* A group of static bindings                                               *)
(* ======================================================================== *)

(* Whether a static constant is a closure. *)
Definition is_closure_const (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ _ | Static_const_closure_of_set _ _ _ _ =>
    true
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    false
  end.

(* The symbols that a binding binds to a closure. *)
Definition closure_symbols_of_binding (binding : static_binding)
    : list symbol :=
  match binding with
  | Bind_symbol s const => if is_closure_const const then [s] else []
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The code ids that a binding brings into scope: the one it binds, or
   the one it declares deleted. *)
Definition declared_code_ids_of_binding (binding : static_binding)
    : list code_id :=
  match binding with
  | Bind_code id _ | Bind_deleted_code id => [id]
  | Bind_symbol _ _ => []
  end.

(* The constants of a group, in order. [before] are the symbols in scope
   in a constant that is not a closure: those in scope around the group,
   the closure symbols of the group, and the other symbols that the
   group has bound so far. [all] are those in scope in a closure: those
   in scope around the group and all that the group binds. Simplify
   processes a group in that order: the closure symbols are declared
   first, the other constants are taken in order, and the closures last
   (simplify/simplify_static_const.ml). *)
Fixpoint group_consts_in_scope (before all : list symbol)
    (ids : list code_id) (bindings : list static_binding) : bool :=
  match bindings with
  | [] => true
  | Bind_symbol s const :: bindings =>
    if is_closure_const const
    then
      static_const_statics_in_scope all ids const
      && group_consts_in_scope before all ids bindings
    else
      static_const_statics_in_scope before ids const
      && group_consts_in_scope (s :: before) all ids bindings
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    group_consts_in_scope before all ids bindings
  end.

(* ======================================================================== *)
(* A unit                                                                   *)
(* ======================================================================== *)

(* Every symbol and every code id that the expression mentions is in
   scope, given those in scope around it. A group of bindings brings its
   names into scope as its comment above says, and the code it binds is
   checked under all that the group binds. A handler of a [Let_cont] is
   checked with, in scope, what the body of the [Let_cont] binds: the
   static names of the body, which in a unit that passes row 1 are all
   at toplevel in it. *)
Fixpoint expr_statics_in_scope (syms : list symbol) (ids : list code_id)
    (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_statics_in_scope syms ids defining_expr
    && expr_statics_in_scope syms ids body
  | Let_static bindings body =>
    let all := flat_map symbols_of_binding bindings ++ syms in
    let ids' := flat_map declared_code_ids_of_binding bindings ++ ids in
    group_consts_in_scope
      (flat_map closure_symbols_of_binding bindings ++ syms) all ids' bindings
    && forallb (binding_statics_in_scope all ids') bindings
    && expr_statics_in_scope all ids' body
  | Let_cont_non_recursive _ h body =>
    handler_statics_in_scope
      (static_names symbols_of_binding body ++ syms)
      (static_names declared_code_ids_of_binding body ++ ids) h
    && expr_statics_in_scope syms ids body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_statics_in_scope
           (static_names symbols_of_binding body ++ syms)
           (static_names declared_code_ids_of_binding body ++ ids) h)
      handlers
    && expr_statics_in_scope syms ids body
  | Apply ap => apply_statics_in_scope syms ids ap
  | Apply_cont ac => apply_cont_statics_in_scope syms ac
  | Switch scrutinee arms =>
    simple_statics_in_scope syms scrutinee
    && forallb (fun arm => apply_cont_statics_in_scope syms (snd arm)) arms
  | Invalid => true
  end
with handler_statics_in_scope (syms : list symbol) (ids : list code_id)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler => expr_statics_in_scope syms ids handler
  end
with binding_statics_in_scope (syms : list symbol) (ids : list code_id)
    (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code _ c => code_statics_in_scope syms ids c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_statics_in_scope (syms : list symbol) (ids : list code_id)
    (c : code) {struct c} : bool :=
  match c with
  | Code _ _ _ _ _ body => expr_statics_in_scope syms ids body
  end.

(* Row 4 of well-formedness. A unit, with the symbols it imports: every
   symbol that it mentions is one of those or is in scope where it is
   mentioned, and every code id that it mentions is in scope there, bound
   or declared deleted. *)
Definition statics_in_scope (imports : list symbol) (e : expr) : bool :=
  expr_statics_in_scope imports [] e.

(* What goes with it of row 2: no static binding of the unit binds a
   symbol that it imports, and no code id is both bound and declared
   deleted, nor declared deleted twice (nor bound twice, which
   [well_formed_unit] asks already). *)
Definition static_names_apart (imports : list symbol) (e : expr) : bool :=
  negb (existsb (fun s => among s imports) (static_names symbols_of_binding e))
  && distinct (static_names declared_code_ids_of_binding e).

(* ======================================================================== *)
(* The continuations of the interface are not bound again                   *)
(* ======================================================================== *)

(* The continuations that an expression binds: by a [Let_cont], in its
   handlers too, and as the two continuations of the code it binds, with
   those that the code binds in turn. *)
Fixpoint bound_continuations (e : expr) : list continuation :=
  match e with
  | Let _ _ _ body => bound_continuations body
  | Let_static bindings body =>
    flat_map binding_bound_continuations bindings ++ bound_continuations body
  | Let_cont_non_recursive k h body =>
    k :: handler_bound_continuations h ++ bound_continuations body
  | Let_cont_recursive handlers body =>
    map fst handlers
    ++ flat_map (fun '(_, h) => handler_bound_continuations h) handlers
    ++ bound_continuations body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_bound_continuations (h : continuation_handler)
    : list continuation :=
  match h with
  | Handler _ _ handler => bound_continuations handler
  end
with binding_bound_continuations (binding : static_binding)
    : list continuation :=
  match binding with
  | Bind_code _ c => code_bound_continuations c
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end
with code_bound_continuations (c : code) : list continuation :=
  match c with
  | Code _ _ return_continuation exn_continuation _ body =>
    return_continuation :: exn_continuation :: bound_continuations body
  end.

(* No [Let_cont] of the unit, and no code, binds a continuation with the
   name of the return continuation of the unit or of its exception
   continuation.

   Here the model is stricter than the compiler (decision D32). The
   compiler checks nothing of the sort; its binders are fresh names, so
   no term it builds binds one of the two again.

   Why it is asked. Row 1 reads "the handler runs after the toplevel
   bindings of its body" off names, and row 4 relies on it: a handler
   is at toplevel when its body can only apply its continuation or use
   the exception continuation of the unit
   (FlambdaWellFormedSpec.handler_at_unit_toplevel). If a [Let_cont] of the
   unit binds that name again, a body that uses it does not leave the
   unit: it goes to a handler of the unit, which may go on to an outer
   handler that reads what the handler skipped was to bind. The unit of
   [rebound_unit_exn_reads_unbound_statement] below does so: it passes
   [statics_in_scope] and every condition of [well_formed_unit] but the
   last, and its run reads a symbol that is not bound.

   The last condition of [well_formed_unit], the discipline of trap
   actions (FlambdaWellFormedSpec.traps_well_formed), refuses that
   unit: it refuses a [Let_cont] that binds a name on the stack of the
   handlers installed, and at toplevel the exception continuation of
   the unit is on it. So in a unit that passes [well_formed_unit] no
   [Let_cont] that runs at toplevel should bind that name, and the
   half of this condition for the exception continuation may be more
   than is needed. That is not proved: it goes through the stack of
   handlers, the trap check does not visit a handler that nothing
   applies, and it asks nothing of the names that code binds. Until it
   is, [statics_bound_in_run_statement] takes this condition as a
   premise, and its proof uses the half for the exception continuation
   only. The half for the return continuation is asked so that the two
   continuations of the interface are alike: nothing here reads the
   name of the return continuation. *)
Definition interface_conts_not_rebound (u : unit_interface) (e : expr)
    : bool :=
  negb (among (unit_return u) (bound_continuations e))
  && negb (among (unit_exn u) (bound_continuations e)).

(* ======================================================================== *)
(* What a state holds                                                       *)
(* ======================================================================== *)

(* The static names that a static environment binds. *)
Definition bound_symbols (statics : static_env) : list symbol :=
  map fst (static_values statics).

Definition bound_code_ids (statics : static_env) : list code_id :=
  map fst (static_code statics).

(* Every symbol that the value is or holds is among [syms], and the code
   of every closure that it is or holds is among [ids]: down the fields
   of a block, the value slots of a closure, and the callee and the
   arguments of a partial application. *)
Fixpoint value_statics_among (syms : list symbol) (ids : list code_id)
    (v : value) : bool :=
  match v with
  | Value_const _ | Value_boxed_number _ | Value_location _ => true
  | Value_symbol s => among s syms
  | Value_closure code value_slots =>
    among code ids
    && forallb (fun '(_, v) => value_statics_among syms ids v) value_slots
  | Value_closure_of_set _ function_decls value_slots =>
    all_among (map snd function_decls) ids
    && forallb (fun '(_, v) => value_statics_among syms ids v) value_slots
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_statics_among syms ids) fields
  | Value_partial_application callee args _ =>
    value_statics_among syms ids callee
    && forallb (value_statics_among syms ids) args
  end.

(* The same of what a symbol is bound to. The symbols that a static set
   gives its closures count. *)
Definition allocated_statics_among (syms : list symbol) (ids : list code_id)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code value_slots =>
    among code ids
    && forallb (fun '(_, v) => value_statics_among syms ids v) value_slots
  | Static_closure_of_set _ closure_symbols function_decls value_slots =>
    all_among (map snd closure_symbols) syms
    && all_among (map snd function_decls) ids
    && forallb (fun '(_, v) => value_statics_among syms ids v) value_slots
  | Static_block _ fields => forallb (value_statics_among syms ids) fields
  | Static_immutable_array elements =>
    forallb (value_statics_among syms ids) elements
  | Static_boxed_number contents => value_statics_among syms ids contents
  | Static_string _ | Static_imported _ => true
  end.

(* The code ids that a unit declares deleted. *)
Definition deleted_code_ids_of_binding (binding : static_binding)
    : list code_id :=
  match binding with
  | Bind_deleted_code id => [id]
  | Bind_code _ _ | Bind_symbol _ _ => []
  end.

Definition deleted_code_ids (e : expr) : list code_id :=
  static_names deleted_code_ids_of_binding e.

(* A value mentions only what the static environment binds: every
   symbol that it is or holds is bound, and the code of every closure
   that it is or holds is bound, or is among [deleted], the code ids
   that the unit declares deleted. *)
Definition value_statics_bound (deleted : list code_id)
    (statics : static_env) (v : value) : Prop :=
  value_statics_among (bound_symbols statics)
    (bound_code_ids statics ++ deleted) v
  = true.

Definition allocated_statics_bound (deleted : list code_id)
    (statics : static_env) (allocated : statically_allocated_value) : Prop :=
  allocated_statics_among (bound_symbols statics)
    (bound_code_ids statics ++ deleted) allocated
  = true.

(* Everything that a state holds mentions only what its static
   environment binds: the values of its variables, those that its
   continuations keep, those of the cells of its heap, and what its
   symbols are bound to (FlambdaSemanticsSpec.state_values_all says
   where a state holds values). *)
Definition state_statics_bound (deleted : list code_id) (s : state) : Prop :=
  state_values_all (value_statics_bound deleted)
    (allocated_statics_bound deleted) s.

(* ======================================================================== *)
(* The fault                                                                *)
(* ======================================================================== *)

(* A symbol that is bound to nothing. *)
Definition unbound_symbol (statics : static_env) (v : value) : bool :=
  match v with
  | Value_symbol s => negb (symbol_bound statics s)
  | _ => false
  end.

(* The closure that a call of a value looks up: what a partial
   application was made of, or the value itself. *)
Definition closure_called (callee : value) : value :=
  match partial_application_of callee with
  | Some (closure, _, _) => closure
  | None => callee
  end.

(* A call of the value looks up a symbol that is not bound, or the code
   of a closure, which is neither bound nor declared deleted by the
   unit. A call of a closure of deleted code is undefined
   (FlambdaSemanticsSpec.shape_of_closure_call) and is not this fault. *)
Definition calls_unbound (deleted : list code_id) (statics : static_env)
    (callee : value) : bool :=
  unbound_symbol statics (closure_called callee)
  || match closure_of statics (closure_called callee) with
     | Some (called, _) =>
       negb (code_id_bound statics called || among called deleted)
     | None => false
     end.

(* The arguments that the closure called is given by a call of the
   value with the arguments [vs]: those that a partial application
   kept, then [vs]. *)
Definition arguments_called (callee : value) (vs : list value)
    : list value :=
  match partial_application_of callee with
  | Some (_, applied, _) => applied ++ vs
  | None => vs
  end.

(* The tuple that such a call reads: the first of those arguments, where
   the code of the closure called is bound and is tupled
   (FlambdaSemanticsSpec.arguments_for_code). No call makes a partial
   application of a closure of tupled code, but a state may hold one
   that no call made, and a call of it reads the first argument that it
   kept. *)
Definition tuple_read (statics : static_env) (callee : value)
    (vs : list value) : option value :=
  match closure_of statics (closure_called callee) with
  | Some (called, _) =>
    match lookup called (static_code statics) with
    | Some c =>
      if code_is_tupled c then hd_error (arguments_called callee vs)
      else None
    | None => None
    end
  | None => None
  end.

(* A call of the value with the arguments reads, as the tuple that it
   gives tupled code, a symbol that is not bound. *)
Definition reads_unbound_tuple (statics : static_env) (callee : value)
    (vs : list value) : bool :=
  match tuple_read statics callee vs with
  | Some tuple => unbound_symbol statics tuple
  | None => false
  end.

(* The value that the application of a continuation calls, and the
   arguments that it calls it with: the one value given to what is left
   of an over-application, and the arguments that remain. *)
Fixpoint called_by_cont (c : cont_value) (vs : list value)
    : option (value * list value) :=
  match c with
  | Over_application remaining _ _ _ _ =>
    match vs with
    | [v] => Some (v, remaining)
    | _ => None
    end
  | Extra_args c extra => called_by_cont c (vs ++ extra)
  | Outer _ | Closure _ _ _ _ => None
  end.

(* The step of the state looks up a symbol that is not bound, or code
   that is neither bound nor declared deleted. The machine looks a
   symbol up where it reads an allocated value: an argument of a
   primitive, the closure that a projection is from, the closure that a
   call is of, and the tuple that a call gives tupled code; and it looks
   code up at a call, which an [Apply] makes, and the application of
   what is left of an over-application.
   A [Let_static] looks its own names up to see that they are not bound
   yet; that is [FlambdaWellFormedSpec.static_fault].

   The test is of each place that the step would look at, and not of
   the first that it reaches: it is true of a call whose callee is a
   symbol that is not bound even where an argument of the call does not
   evaluate, so that the step is undefined before any lookup. It says
   more often than the machine that something unbound is looked up,
   never less; the statement below says that it is false. *)
Definition statics_scope_fault (deleted : list code_id) (s : state) : bool :=
  match state_expr s with
  | Let _ _ (Prim _ args) _ =>
    match eval_simples (state_vars s) args with
    | Some vs => existsb (unbound_symbol (state_statics s)) vs
    | None => false
    end
  | Let _ _ (Project_value_slot _ closure) _
  | Let _ _ (Project_value_slot_from _ _ closure) _
  | Let _ _ (Project_function_slot _ _ closure) _ =>
    match eval_simple (state_vars s) closure with
    | Some v => unbound_symbol (state_statics s) v
    | None => false
    end
  | Apply ap =>
    match eval_simple (state_vars s) (ap_callee ap) with
    | Some callee =>
      calls_unbound deleted (state_statics s) callee
      || match eval_simples (state_vars s) (ap_args ap) with
         | Some vs => reads_unbound_tuple (state_statics s) callee vs
         | None => false
         end
    | None => false
    end
  | Apply_cont ac =>
    match eval_simples (state_vars s) (ac_args ac) with
    | Some vs =>
      match
        called_by_cont (resolve (ac_continuation ac) (state_conts s)) vs
      with
      | Some (v, remaining) =>
        calls_unbound deleted (state_statics s) v
        || reads_unbound_tuple (state_statics s) v remaining
      | None => false
      end
    | None => false
    end
  | Let _ _ _ _ | Let_static _ _ | Let_cont_non_recursive _ _ _
  | Let_cont_recursive _ _ | Switch _ _ | Invalid =>
    false
  end.

(* ======================================================================== *)
(* The run of a unit                                                        *)
(* ======================================================================== *)

(* The static environment that a unit is run from: it binds the symbols
   that the unit imports and nothing else, and no code; and what it
   binds them to mentions only those symbols.

   The predicate is about static names only; it asks nothing else of
   what the imports are bound to. Ordinary imports that are constants
   are covered, such as the symbols of the predefined exceptions. No
   import that is a closure that can be called is: its code is code of
   another unit, which this environment does not bind, and
   [allocated_statics_among imports []] refuses a closure of any code.
   A unit that imports such closures needs that code in the static
   environment that it is run from, and a statement of what that code
   may mention; that goes with the work on the imports of a unit, and
   is not here. *)
Definition imports_bound (imports : list symbol) (statics : static_env)
    : Prop :=
  static_code statics = []
  ∧ (∀ s : symbol, symbol_bound statics s = among s imports)
  ∧ Forall
      (fun binding => allocated_statics_among imports [] (snd binding) = true)
      (static_values statics).

(* No state of the run of a well-formed unit, started with its imports
   bound and from a heap whose cells hold values that mention no symbol
   but the imports and no code id, holds a symbol that is not bound or a
   closure of code that is neither bound nor declared deleted, anywhere,
   the cells of its heap included; nor does its step look one up; and
   the values that it exits with hold none. The premise on the heap is
   sufficient and not the weakest: it refuses, in a cell of the heap
   that the unit starts from, any closure that mentions a code id, one
   of code that the unit declares deleted included, which the
   conclusion would allow. The heap is empty wherever the statement is
   used. *)
Definition statics_bound_in_run_statement : Prop :=
  ∀ (imports : list symbol) (u : unit_interface) (e : expr)
    (statics : static_env) (hp : heap) (rd : read_record) (n : nat) (s : state),
    well_formed_unit u e = true ->
    statics_in_scope imports e = true ->
    interface_conts_not_rebound u e = true ->
    imports_bound imports statics ->
    heap_values_all (fun v => value_statics_among imports [] v = true) hp ->
    run n (State e [] [] statics hp rd) = Step s ->
    state_statics_bound (deleted_code_ids e) s
    ∧ statics_scope_fault (deleted_code_ids e) s = false
    ∧ match step s with
      | Exit _ vs _ _ =>
        Forall
          (value_statics_bound (deleted_code_ids e) (state_statics s)) vs
      | Step _ | Undefined => True
      end.

(* Why [interface_conts_not_rebound], or the discipline of trap actions,
   is asked. There is a unit that passes [statics_in_scope],
   [static_names_apart] and every condition of [well_formed_unit] but
   the last, which it fails, and whose run reads a symbol that is not
   bound. So it is not a unit that [well_formed_unit] accepts: the
   statement says what goes wrong when neither of the two is asked. It
   binds the name of its exception continuation again, as an exception
   handler:

     let_cont k_outer = (x = the first field of s; return x) in
     let_cont exn e = (apply k_outer) in
     let_cont k = (let_static s = a block; apply k_outer) in
     raise to exn

   Row 1 lets the handler of [k] bind [s]: the body of its [Let_cont]
   only raises to the name of the exception continuation of the unit.
   The check lets the handler of [k_outer] mention [s], which the body
   of its [Let_cont] binds. The raise goes to the handler bound in the
   unit, which goes to [k_outer], and [s] was never bound. *)
Definition rebound_unit_exn_reads_unbound_statement : Prop :=
  ∃ (u : unit_interface) (e : expr) (n : nat) (s : state),
    statics_at_unit_toplevel (unit_exn u) true e = true
    ∧ distinct (static_names code_ids_of_binding e) = true
    ∧ distinct (static_names symbols_of_binding e) = true
    ∧ names_in_scope (unit_return u) (unit_exn u) e = true
    ∧ binders_distinct e = true
    ∧ code_closed e = true
    ∧ sets_well_formed e = true
    ∧ numbers_in_range e = true
    ∧ unit_exn_handlers_well_formed (unit_exn u) e = true
    ∧ traps_well_formed (unit_exn u) e = false
    ∧ statics_in_scope [] e = true
    ∧ static_names_apart [] e = true
    ∧ interface_conts_not_rebound u e = false
    ∧ run n (State e [] [] (Static_env [] []) empty_heap no_reads) = Step s
    ∧ statics_scope_fault (deleted_code_ids e) s = true.

(* Why row 1 is asked with the check. There is a unit that passes
   [statics_in_scope], [static_names_apart],
   [interface_conts_not_rebound] and every condition of
   [well_formed_unit] but the first, which it fails, and whose run reads
   a symbol that is not bound. A handler of it binds a symbol, and the
   body of its [Let_cont] can go round it:

     let v = 1 in
     let_cont k_a = (x = the first field of s; return x) in
     let_cont k_b = (let_static s = a block; apply k_a) in
     switch v | 0 -> apply k_b | 1 -> apply k_a

   The check lets the handler of [k_a] mention [s], which the body of
   its [Let_cont] binds. Row 1 refuses the unit: the handler of [k_b]
   binds a symbol, and the body of its [Let_cont] can also apply
   [k_a]. *)
Definition branch_round_binding_reads_unbound_statement : Prop :=
  ∃ (u : unit_interface) (e : expr) (n : nat) (s : state),
    statics_at_unit_toplevel (unit_exn u) true e = false
    ∧ distinct (static_names code_ids_of_binding e) = true
    ∧ distinct (static_names symbols_of_binding e) = true
    ∧ names_in_scope (unit_return u) (unit_exn u) e = true
    ∧ binders_distinct e = true
    ∧ code_closed e = true
    ∧ sets_well_formed e = true
    ∧ numbers_in_range e = true
    ∧ unit_exn_handlers_well_formed (unit_exn u) e = true
    ∧ traps_well_formed (unit_exn u) e = true
    ∧ statics_in_scope [] e = true
    ∧ static_names_apart [] e = true
    ∧ interface_conts_not_rebound u e = true
    ∧ run n (State e [] [] (Static_env [] []) empty_heap no_reads) = Step s
    ∧ statics_scope_fault (deleted_code_ids e) s = true.
