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
   bindings ([Let_static]), each outside loops and code (row 1), and
   the rule is:
   - a symbol that the unit imports is in scope everywhere;
   - in a group, the symbols that the group binds to closures are in
     scope in every definition of the group. Its other symbols are in
     scope in the definitions that follow them, and in the closures and
     the code of the group. The code ids of the group, bound or declared
     deleted, are in scope in every definition of the group;
   - after a group, all it binds or declares deleted is in scope in the
     body;
   - in a handler of a [Let_cont], recursive or not, names that the
     body of the [Let_cont] binds on every way to the handler are in
     scope: those that [exits] finds, which is a lower bound (see
     below);
   - the body of code sees what is in scope where the code is bound;
   - a code id that is declared deleted is in scope, and is not code:
     [Bind_deleted_code] binds nothing, and a call of a closure of it
     stays undefined where it is executed.

   The clause for the handlers of a [Let_cont]. The run enters a
   handler from the body of its [Let_cont]: the body applies the
   continuation, or gives it to a call that returns or raises to it.
   What is in scope in the handler is what the body has bound by then,
   on every way from its start to such a place: down its spine, and in
   the handlers of the continuations bound inside it that it goes
   through on the way. [exits] reads those names off the text, for
   every continuation at once; what it gives is a lower bound (its
   comment says where it gives less than a run would have). A handler
   of a recursive continuation is entered from the handlers of its
   group too, and those are entered with no less bound.

   The clause does not rest on Simplify's rule for the term it is given,
   nor on a condition on the names of the continuations of the
   interface. [statics_bound_in_run_statement] still takes
   [well_formed_unit] as a premise, and its proof uses of it that no
   static binding is in a loop or in code, which is row 1 as a unit is
   asked it (FlambdaWellFormedSpec.statics_outside_loops_and_code). A
   handler was once given all that the body of its [Let_cont] binds.
   That is right only of a unit in which every handler that binds
   something is entered on every way through the body of its [Let_cont],
   which is what the model's reading of Simplify's rule for the term it
   is given asks (FlambdaWellFormedSpec.statics_at_unit_toplevel) and is
   more than row 1: where a branch of the body can go round a handler
   that binds a symbol, the handlers after it must not mention the
   symbol ([round_unit_refused_statement]); nor where a raise that looks
   as if it left the unit goes to a handler of the unit
   ([rebound_unit_refused_statement]).

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

   What the rule is asked of: the terms that are given to Simplify, and
   those that it returns. It is not a rule of every term that the
   compiler makes. In classic mode Simplify is not run, and closure
   conversion may bind a block of symbols at the head of the unit
   before the symbols that the block mentions; such a unit fails this
   check, and there is no run of Simplify to validate there.

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
(* What is bound where a continuation is applied                            *)
(* ======================================================================== *)

(* The ways out of an expression: for each continuation that the
   expression applies or gives to a call, and does not bind, static
   names that are bound on every way from the start of the expression to
   a place where it does so. It is read by [lookup]. *)
Definition exit_names : Type := finite_map (list positive).

(* [names] are bound first: they are bound on every way out. *)
Definition with_bound (names : list positive) (l : exit_names)
    : exit_names :=
  map_values (fun entry => names ++ entry) l.

(* One way more, to [k] with [names] bound. If [l] has a way to [k]
   already, only the names of both ways are bound on every way to it. *)
Fixpoint add_exit (k : continuation) (names : list positive)
    (l : exit_names) : exit_names :=
  match l with
  | [] => [(k, names)]
  | (k', names') :: l =>
    if Pos.eqb k k'
    then (k', inter names names') :: l
    else (k', names') :: add_exit k names l
  end.

(* The ways of both. *)
Definition meet_exits (l l' : exit_names) : exit_names :=
  fold_right (fun exit l' => add_exit (fst exit) (snd exit) l') l' l.

(* Without the ways to the continuations [bound]. *)
Definition remove_keys (bound : list continuation) (l : exit_names)
    : exit_names :=
  filter (fun exit => negb (among (fst exit) bound)) l.

(* What is bound on every way to a continuation of [group]: the names
   that [l] has for every continuation of the group that it has.
   [None] if it has none of them. *)
Fixpoint group_entry (group : list continuation) (l : exit_names)
    : option (list positive) :=
  match group with
  | [] => None
  | k :: group =>
    match lookup k l, group_entry group l with
    | Some names, Some names' => Some (inter names names')
    | Some names, None => Some names
    | None, entry => entry
    end
  end.

(* The ways out of an expression, for the static names that [names_of]
   gives of a binding: the symbols it binds, or the code ids it binds
   or declares deleted.

   What a way has is a lower bound, read off the text, and not the
   exact set. Every arm of a [Switch] counts, whatever the scrutinee
   is. A call counts for its return continuation and for its exception
   continuation, both with what is bound before the call. A trap action
   names a continuation and applies none: the machine executes no trap
   action (FlambdaSemanticsSpec.v).

   A [Let_cont] of a continuation [k] that its body applies: the ways
   of the body to other continuations stay, and the handler adds its
   own, each with what is bound on every way of the body to [k] and
   then what the handler binds on the way. The ways of the body to [k]
   itself go; those of the handler to that number stay, as in the
   handler the number is of a continuation bound further out. If the
   body has no way to [k] the handler is never entered, and has no way
   out.

   A recursive group: what is bound on every way of the body to a
   continuation of the group is bound in all its handlers, however
   they are entered, since a handler is entered from the body or from a
   handler. That one entry serves the whole group: a name bound on
   every way of the body to one handler and not to another is not
   counted for either. The ways of the handlers to the group itself
   go. *)
Fixpoint exits (names_of : static_binding -> list positive) (e : expr)
    {struct e} : exit_names :=
  match e with
  | Let _ _ _ body => exits names_of body
  | Let_static bindings body =>
    with_bound (flat_map names_of bindings) (exits names_of body)
  | Let_cont_non_recursive k h body =>
    let of_body := exits names_of body in
    match lookup k of_body with
    | Some entry =>
      meet_exits (remove_keys [k] of_body)
        (with_bound entry (handler_exits names_of h))
    | None => of_body
    end
  | Let_cont_recursive handlers body =>
    let of_body := exits names_of body in
    let group := map fst handlers in
    match group_entry group of_body with
    | Some entry =>
      meet_exits (remove_keys group of_body)
        (with_bound entry
           (remove_keys group
              (fold_right
                 (fun '(_, h) l => meet_exits (handler_exits names_of h) l)
                 [] handlers)))
    | None => of_body
    end
  | Apply ap =>
    add_exit (ap_continuation ap) [] [(ap_exn_continuation ap, [])]
  | Apply_cont ac => [(ac_continuation ac, [])]
  | Switch _ arms =>
    fold_right (fun arm l => add_exit (ac_continuation (snd arm)) [] l) []
      arms
  | Invalid => []
  end
with handler_exits (names_of : static_binding -> list positive)
    (h : continuation_handler) {struct h} : exit_names :=
  match h with
  | Handler _ _ handler => exits names_of handler
  end.

(* The static names in scope in the handler of the non-recursive
   continuation [k] bound around [body], beside those in scope around
   the [Let_cont]: those bound on every way of [body] to [k]. A handler
   that [body] has no way to is never entered; it is given all that
   [body] binds. *)
Definition handler_scope (names_of : static_binding -> list positive)
    (k : continuation) (body : expr) : list positive :=
  match lookup k (exits names_of body) with
  | Some names => names
  | None => static_names names_of body
  end.

(* The same for the handlers of a recursive group. *)
Definition recursive_scope (names_of : static_binding -> list positive)
    (group : list continuation) (body : expr) : list positive :=
  match group_entry group (exits names_of body) with
  | Some names => names
  | None => static_names names_of body
  end.

(* A way out has only names that the expression binds. *)
Definition exits_within_static_names_statement : Prop :=
  ∀ (names_of : static_binding -> list positive) (e : expr)
    (k : continuation) (names : list positive),
    lookup k (exits names_of e) = Some names ->
    all_among names (static_names names_of e) = true.

(* Four expressions and their ways out, for the symbols. The symbols
   are [1] and [2], bound to a block; the continuations are numbers
   from [1] to [5]. *)
Definition exits_example_block : static_const :=
  Static_const_block 0 [Const (Tagged_immediate 0)].

Definition exits_example_jump (k : continuation) : expr :=
  Apply_cont (Apply_cont_expr k [] None).

(* A continuation bound again inside. The body binds the symbol and
   applies [3]; the handler of [3] applies the number [3], which there
   is a continuation bound further out. The way to that one has the
   symbol bound. *)
Definition exits_rebound_term : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata []) (exits_example_jump 3%positive))
    (Let_static [Bind_symbol 1%positive exits_example_block]
       (exits_example_jump 3%positive)).

Definition exits_rebound_statement : Prop :=
  exits symbols_of_binding exits_rebound_term
  = [(3%positive, [1%positive])].

(* A handler that nothing applies. Its way out, to [2], is not one of
   the expression, and it is given all that the body binds. *)
Definition exits_never_applied_body : expr :=
  Let_static [Bind_symbol 1%positive exits_example_block]
    (exits_example_jump 1%positive).

Definition exits_never_applied_term : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata []) (exits_example_jump 2%positive))
    exits_never_applied_body.

Definition exits_never_applied_statement : Prop :=
  exits symbols_of_binding exits_never_applied_term
  = [(1%positive, [1%positive])]
  ∧ handler_scope symbols_of_binding 3%positive exits_never_applied_body
    = [1%positive].

(* The body of a recursive group of [3] and [4]. It binds the symbol
   [1]; then a test applies [3], or goes on to the handler of [5], which
   binds the symbol [2] and applies [4]. On every way into the group
   only [1] is bound. *)
Definition exits_group_body : expr :=
  Let_static [Bind_symbol 1%positive exits_example_block]
    (Let_cont_non_recursive 5%positive
       (Handler [] (make_handler_metadata [])
          (Let_static [Bind_symbol 2%positive exits_example_block]
             (exits_example_jump 4%positive)))
       (Switch (Var 1%positive)
          [(0%Z, Apply_cont_expr 3%positive [] None);
           (1%Z, Apply_cont_expr 5%positive [] None)])).

Definition exits_group_statement : Prop :=
  lookup 3%positive (exits symbols_of_binding exits_group_body)
  = Some [1%positive]
  ∧ lookup 4%positive (exits symbols_of_binding exits_group_body)
    = Some [1%positive; 2%positive]
  ∧ recursive_scope symbols_of_binding [3%positive; 4%positive] exits_group_body
    = [1%positive].

(* A call that returns to [3] and raises to [4], bound around it. The
   handler of [3] binds the symbol and applies [1]. The symbol is bound
   on the way out to [1], and not on the way of the exception: the
   handler of [4] is given nothing. *)
Definition exits_example_call : expr :=
  Apply
    (Apply_expr (Var 1%positive) Indirect_unknown_arity [] 3%positive
       4%positive [] [] [Singleton Value]).

Definition exits_call_returns : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [1%positive] (make_handler_metadata [Value])
       (Let_static [Bind_symbol 1%positive exits_example_block]
          (exits_example_jump 1%positive)))
    exits_example_call.

Definition exits_call_statement : Prop :=
  lookup 1%positive (exits symbols_of_binding exits_call_returns)
  = Some [1%positive]
  ∧ lookup 4%positive (exits symbols_of_binding exits_call_returns)
    = Some []
  ∧ handler_scope symbols_of_binding 4%positive exits_call_returns = [].

(* ======================================================================== *)
(* A unit                                                                   *)
(* ======================================================================== *)

(* Every symbol and every code id that the expression mentions is in
   scope, given those in scope around it. A group of bindings brings its
   names into scope as its comment above says, and the code it binds is
   checked under all that the group binds. A handler of a [Let_cont] is
   checked with, in scope, what the body of the [Let_cont] binds on
   every way to the handler ([handler_scope], [recursive_scope]). *)
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
  | Let_cont_non_recursive k h body =>
    handler_statics_in_scope
      (handler_scope symbols_of_binding k body ++ syms)
      (handler_scope declared_code_ids_of_binding k body ++ ids) h
    && expr_statics_in_scope syms ids body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_statics_in_scope
           (recursive_scope symbols_of_binding (map fst handlers) body ++ syms)
           (recursive_scope declared_code_ids_of_binding (map fst handlers) body
            ++ ids)
           h)
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

   Why it was asked. The model's reading of Simplify's rule of place
   takes "the handler runs after the bindings of its body" from names: a
   handler is at toplevel when its body can only apply its continuation
   or use the exception continuation of the unit
   (FlambdaWellFormedSpec.handler_at_unit_toplevel). If a [Let_cont] of
   the unit binds that name again, a body that uses it does not leave
   the unit: it goes to a handler of the unit, which may go on to an
   outer handler that reads what the handler skipped was to bind
   ([rebound_unit]). While a handler was given all that the body of its
   [Let_cont] binds, the check of this file was right only with this
   condition, and [statics_bound_in_run_statement] took it as a premise.

   The check no longer needs it: it goes by the continuations that the
   text applies and refuses that unit by itself
   ([rebound_unit_refused_statement]), and the statement has lost the
   premise. The condition stays for those who ask it: the hoisting of
   static bindings (FlambdaHoistStaticsSpec.hoist_statics) tests it. *)
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

(* No state of the run of a well-formed unit that passes the check,
   started with its imports bound and from a heap whose cells hold
   values that mention no symbol but the imports and no code id, holds
   a symbol that is not bound or a closure of code that is neither
   bound nor declared deleted, anywhere, the cells of its heap
   included; nor does its step look one up; and the values that it
   exits with hold none. The premise on the heap is
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

(* ======================================================================== *)
(* The check that gave a handler all that the body binds                    *)
(* ======================================================================== *)

(* The check as it was before a handler was given what is bound on
   every way to it: a handler of a [Let_cont] has in scope all that the
   body of the [Let_cont] binds. It is no part of the check that a unit
   is asked; it is here for the statement below. *)
Fixpoint expr_statics_in_scope_of_whole_bodies (syms : list symbol)
    (ids : list code_id) (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ defining_expr body =>
    named_statics_in_scope syms ids defining_expr
    && expr_statics_in_scope_of_whole_bodies syms ids body
  | Let_static bindings body =>
    let all := flat_map symbols_of_binding bindings ++ syms in
    let ids' := flat_map declared_code_ids_of_binding bindings ++ ids in
    group_consts_in_scope
      (flat_map closure_symbols_of_binding bindings ++ syms) all ids' bindings
    && forallb (binding_statics_in_scope_of_whole_bodies all ids') bindings
    && expr_statics_in_scope_of_whole_bodies all ids' body
  | Let_cont_non_recursive _ h body =>
    handler_statics_in_scope_of_whole_bodies
      (static_names symbols_of_binding body ++ syms)
      (static_names declared_code_ids_of_binding body ++ ids) h
    && expr_statics_in_scope_of_whole_bodies syms ids body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_statics_in_scope_of_whole_bodies
           (static_names symbols_of_binding body ++ syms)
           (static_names declared_code_ids_of_binding body ++ ids) h)
      handlers
    && expr_statics_in_scope_of_whole_bodies syms ids body
  | Apply ap => apply_statics_in_scope syms ids ap
  | Apply_cont ac => apply_cont_statics_in_scope syms ac
  | Switch scrutinee arms =>
    simple_statics_in_scope syms scrutinee
    && forallb (fun arm => apply_cont_statics_in_scope syms (snd arm)) arms
  | Invalid => true
  end
with handler_statics_in_scope_of_whole_bodies (syms : list symbol)
    (ids : list code_id) (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler =>
    expr_statics_in_scope_of_whole_bodies syms ids handler
  end
with binding_statics_in_scope_of_whole_bodies (syms : list symbol)
    (ids : list code_id) (binding : static_binding) {struct binding} : bool :=
  match binding with
  | Bind_code _ c => code_statics_in_scope_of_whole_bodies syms ids c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_statics_in_scope_of_whole_bodies (syms : list symbol)
    (ids : list code_id) (c : code) {struct c} : bool :=
  match c with
  | Code _ _ _ _ _ body => expr_statics_in_scope_of_whole_bodies syms ids body
  end.

Definition statics_in_scope_of_whole_bodies (imports : list symbol)
    (e : expr) : bool :=
  expr_statics_in_scope_of_whole_bodies imports [] e.

(* A unit that passed the three checks of before passes the two of now.
   Its static bindings were at toplevel by the model's reading of
   Simplify's rule, which gives that they are outside loops and code
   (FlambdaWellFormedSpec.at_unit_toplevel_outside_loops_and_code_statement);
   and with that rule, and no continuation of the interface bound
   again ([interface_conts_not_rebound]: the premise is of both
   continuations, and the proof uses it of the exception continuation
   alone), a unit that passed the check which gave a handler all that
   the body binds passes the check of this file. The other conditions
   of [well_formed_unit] are those that they were. *)
Definition whole_bodies_in_scope_statement : Prop :=
  ∀ (imports : list symbol) (u : unit_interface) (e : expr),
    statics_at_unit_toplevel (unit_exn u) true e = true ->
    interface_conts_not_rebound u e = true ->
    statics_in_scope_of_whole_bodies imports e = true ->
    statics_in_scope imports e = true.

(* ======================================================================== *)
(* Two units that the check refuses                                         *)
(* ======================================================================== *)

(* The interface of both: the return continuation is [1] and the
   exception continuation is [2]. *)
Definition refused_interface : unit_interface :=
  Unit_interface 1%positive 2%positive.

(* A unit whose body goes round a handler that binds a symbol:

     let v = 1 in
     let_cont k_a = (x = the first field of s; return x) in
     let_cont k_b = (let_static s = a block; apply k_a) in
     switch v | 0 -> apply k_b | 1 -> apply k_a

   The variable of the test is [2], and [1] is the variable that the
   handler of [k_a] binds to the field that it reads of the symbol [1];
   [k_a] and [k_b] are [3] and [4]. *)
Definition round_unit : expr :=
  Let 2%positive (Naked_number Kind_naked_immediate)
    (Simple (Const (Naked_immediate 1)))
    (Let_cont_non_recursive 3%positive
       (Handler [] (make_handler_metadata [])
          (Let 1%positive Value
             (Prim
                (Unary
                   (Block_load (Block_access_values None None Any_value)
                      Immutable 0))
                [Symbol 1%positive])
             (Apply_cont (Apply_cont_expr 1%positive [Var 1%positive] None))))
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static [Bind_symbol 1%positive exits_example_block]
                (Apply_cont (Apply_cont_expr 3%positive [] None))))
          (Switch (Var 2%positive)
             [(0%Z, Apply_cont_expr 4%positive [] None);
              (1%Z, Apply_cont_expr 3%positive [] None)]))).

(* Why a handler is given what is bound on every way to it, and not all
   that the body of its [Let_cont] binds. The unit above is well-formed:
   its one static binding is in no loop and in no code. The symbol is
   bound on one of the two ways to [k_a] and not on the other, so the
   check refuses the unit; and its run, which takes the other way, reads
   the symbol when it is not bound. Nothing but this check refuses the
   unit. The model's reading of Simplify's rule for the term it is given
   does (FlambdaWellFormedSpec.statics_at_unit_toplevel: the handler of
   [k_b] binds a symbol, and the body of its [Let_cont] can also apply
   [k_a]), and with that rule a handler could be given all that the body
   binds; a unit need not keep that rule. *)
Definition round_unit_refused_statement : Prop :=
  well_formed_unit refused_interface round_unit = true
  ∧ static_names_apart [] round_unit = true
  ∧ interface_conts_not_rebound refused_interface round_unit = true
  ∧ statics_at_unit_toplevel (unit_exn refused_interface) true round_unit
    = false
  ∧ statics_in_scope [] round_unit = false
  ∧ (∃ (n : nat) (s : state),
      run n (State round_unit [] [] (Static_env [] []) empty_heap no_reads)
      = Step s
      ∧ statics_scope_fault (deleted_code_ids round_unit) s = true).

(* A unit that binds the name of its exception continuation again, as an
   exception handler:

     let_cont k_outer = (x = the first field of s; return x) in
     let_cont exn e = (apply k_outer) in
     let_cont k = (let_static s = a block; apply k_outer) in
     raise to exn

   The continuations [k_outer] and [k] are [3] and [4]. The variable,
   the symbol and the parameter of the exception handler are [1], [1]
   and [2]. *)
Definition rebound_apply_k_outer : expr :=
  Apply_cont (Apply_cont_expr 3%positive [] None).

Definition rebound_unit : expr :=
  Let_cont_non_recursive 3%positive
    (Handler [] (make_handler_metadata [])
       (Let 1%positive Value
          (Prim
             (Unary
                (Block_load (Block_access_values None None Any_value)
                   Immutable 0))
             [Symbol 1%positive])
          (Apply_cont (Apply_cont_expr 1%positive [Var 1%positive] None))))
    (Let_cont_non_recursive 2%positive
       (Handler [2%positive] (make_exn_handler_metadata [Value])
          rebound_apply_k_outer)
       (Let_cont_non_recursive 4%positive
          (Handler [] (make_handler_metadata [])
             (Let_static [Bind_symbol 1%positive exits_example_block]
                rebound_apply_k_outer))
          (Apply_cont
             (Apply_cont_expr 2%positive [Const (Tagged_immediate 0)]
                (Some (Pop 2%positive None)))))).

(* The raise goes to the handler bound in the unit, which goes to
   [k_outer], and [s] was never bound: the run reads a symbol that is
   not bound. The check refuses the unit: nothing applies [k], and on
   the way to [k_outer] through the handler of [exn] nothing is bound.
   It goes by the continuations that the text applies, whatever their
   names, so it needs no condition on the names of the continuations of
   the interface; the unit fails [interface_conts_not_rebound], and the
   model's reading of Simplify's rule of place lets the handler of [k]
   bind [s], as the body of its [Let_cont] only raises to the name of
   the exception continuation of the unit. *)
Definition rebound_unit_refused_statement : Prop :=
  statics_at_unit_toplevel (unit_exn refused_interface) true rebound_unit
  = true
  ∧ static_names_apart [] rebound_unit = true
  ∧ interface_conts_not_rebound refused_interface rebound_unit = false
  ∧ statics_in_scope [] rebound_unit = false
  ∧ (∃ (n : nat) (s : state),
      run n (State rebound_unit [] [] (Static_env [] []) empty_heap no_reads)
      = Step s
      ∧ statics_scope_fault (deleted_code_ids rebound_unit) s = true).
