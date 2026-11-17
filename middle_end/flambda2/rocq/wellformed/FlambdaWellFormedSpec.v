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

(* Where static bindings may be: the well-formedness of a compilation unit.

   This file is to be reviewed: [statics_outside_loops_and_code]
   against where Simplify puts the static bindings of the term it makes
   (simplify/simplify_let_expr.ml and simplify_let_cont_expr.ml, where
   the lifted constants are placed; simplify_apply_cont_expr.ml, a
   handler put at its one use) and against what To_cmm needs of a
   static binding (to_cmm/to_cmm_shared.ml, make_update),
   [statics_at_unit_toplevel] against the definition of toplevel in
   Simplify (simplify/simplify_let_cont_expr.ml, at_unit_toplevel, and
   Downwards_env.at_unit_toplevel), [sets_well_formed] against what the
   compiler checks of a set of closures (terms/static_const.ml,
   match_against_bound_static_pattern) and what it assumes
   (terms/function_declarations.ml, create), [numbers_in_range] against
   the types of the compiler's numbers (the comment of its section says
   which), [unit_exn_handlers_well_formed] and [traps_well_formed]
   against what the backend requires of the terms it is given
   (to_cmm/to_cmm_expr.ml: translate_raise, translate_apply,
   let_cont_exn_handler; backend/select_utils.ml: combine_traps,
   set_traps, Stack_offset_and_exn.process_basic; backend/ssa/ssa.ml:
   compute_block_end_trap_stack) and what the front end builds
   (from_lambda/lambda_to_flambda.ml, the case Ltrywith;
   closure_conversion.ml, close_raise0), [well_formed_unit], and the
   statements. Three of the conditions of [well_formed_unit] are
   defined in FlambdaBindersSpec.v and are reviewed there, against the
   compiler's binders: [names_in_scope], [binders_distinct] and
   [code_closed].

   A [Let_static] is executed once at most. Two rules say where one
   may stand, and a unit is asked the second.

   Simplify accepts a [Let_static] at the toplevel of the compilation
   unit that it is given, and nowhere else. Toplevel is:
   - the unit itself;
   - the body of a [Let], of a [Let_static] or of a [Let_cont] that is at
     toplevel;
   - the handler of a non-recursive continuation bound at toplevel, which
     is not an exception handler, when all its body can do is to apply
     that continuation or raise to the exception continuation of the
     unit: the handler then postdominates its body.
   Nothing else is: not the handler of a recursive continuation, which may
   run many times, and never the body of code, which runs once for each
   call.

   [statics_at_unit_toplevel] is the model's reading of this, as a check
   of the text: it carries down the term
   whether it is at toplevel, as Simplify does.

   The term that Simplify returns need not keep that rule: it can have
   static bindings in handlers that are not at toplevel. The rule that
   a unit is asked is that no [Let_static] is in the handler of a
   recursive continuation or in the body of code, at any depth:
   [statics_outside_loops_and_code], which is row 1 of the table of
   requests/C0-well-formed.md as a unit is asked it (decision D60; the
   CR at the definition says what Simplify and To_cmm do). It is asked
   of the terms that are given to Simplify and of those that it
   returns; that every term the compiler makes keeps it is not proved,
   and is not said. A term that passes [statics_at_unit_toplevel]
   passes this one
   ([at_unit_toplevel_outside_loops_and_code_statement]).
   Whether a static name is bound where it is mentioned is not part of
   either rule: FlambdaStaticsInScopeSpec.v has that check, and a
   verdict on a unit needs both.

   [well_formed_unit] asks [statics_outside_loops_and_code], and
   also that no code id or symbol is bound twice, that the sets of
   closures of the unit are whole ([sets_well_formed]), that its
   numbers are in range ([numbers_in_range]), and that its exception
   handlers and trap actions are in order
   ([unit_exn_handlers_well_formed], [traps_well_formed]).

   It asks three things of the variables and the continuations of the
   unit as well, which are rows 3, 5 and 6 of the table of
   requests/C0-well-formed.md:
   - every variable and every continuation that the unit uses is bound
     around the use, but for the two continuations of its interface
     ([names_in_scope]);
   - the names that one binder binds are distinct: the parameters of a
     handler; the parameters of code and its [my_closure], which is so
     not one of them; the two continuations of code; the continuations
     of a recursive group, which is not empty ([binders_distinct]);
   - code is closed: its body has no free variable but its parameters
     and [my_closure], and no free continuation but its two; what the
     body binds itself it may use ([code_closed]).
   The first the compiler checks where a name is visited (Simplify
   stops on a name that its environment has not got), not where a term
   is built. In the last two the model asks more than the compiler
   checks, by the ruling that it may where the condition ought to be an
   invariant; FlambdaBindersSpec.v says at each definition what the
   compiler does.
   What they give is that the run of a well-formed unit reads no
   variable that has no value and leaves only by a continuation of its
   interface ([well_formed_names_bound_statement], from
   FlambdaScopeSpec.v).

   The model can express sets of closures that the compiler does not
   make (FlambdaSemanticsSpec.v says what the machine does with them).
   [sets_well_formed] is what the compiler's sets have that the model's
   may lack:
   - a closure is at a function slot of its set, and the function slots
     of a set are distinct;
   - the functions of a set have each their own code id. The compiler
     builds a set in which two have the same, and Simplify fails on it;
     such a term is not well-formed Flambda 2 (the human's ruling);
   - the closures of a static set are bound together. The compiler binds
     them with one pattern; here each symbol has a binding of its own,
     which repeats the set and the symbols of its closures, and the
     check is that one group has the binding of each symbol of the set,
     all with the same set;
   - the static sets that one group binds have no code in common, be
     they of one function or of several. Simplify fails on a group two
     of whose sets have the same code.
   It does not ask that the value slots of a set are distinct (the
   compiler's are a map), nor that a code id is named by one set only:
   two sets of a unit may have the same code, unless they are static
   and one group binds them both.
   FlambdaClosuresSpec.v says what follows for a well-formed unit: it
   never makes a closure at a slot its set has not got, and the symbols
   of a static set are bound to its closures in every state of its run.

   Exception handlers and trap actions. The machine does not execute
   trap actions and goes by the names in a term: a raise goes to the
   continuation it names, and an exception of a call to the exception
   continuation the call names (FlambdaSemanticsSpec.v). The code that
   the backend makes has a stack of handlers and goes by that: a [Push]
   installs a handler on it and a [Pop] removes the one installed last,
   whatever continuation the trap action names; a raise goes to the
   handler installed last, whatever continuation it names; and so does
   an exception raised by a call, whatever exception continuation the
   call names. Two checks are of what the front end keeps to, under
   which the names and such a stack cannot part, and the handlers of a
   term are what the backend takes them for:
   - [unit_exn_handlers_well_formed]: exception handlers are apart from
     the other continuations. A handler says whether it is one
     ([handler_is_exn_handler]). One that does is not applied as an
     ordinary continuation and takes the exception; one that does not
     is not named by a trap action nor as the exception continuation of
     a call; none is recursive. Of the continuations that a unit, or
     code, does not bind, only the exception continuation is used as an
     exception handler, and that by calls and raises only; and no call
     gives one extra arguments ([ways_out_well_formed]);
   - [traps_well_formed]: the discipline of trap actions. The check
     follows the term with the stack of the handlers installed, as a
     list of continuation names, the last installed first. The body of
     code, and a unit, start with one handler installed: their
     exception continuation, which stands for the handler of the
     caller. A [Pop], a raise and the exception continuation of a call
     name the handler installed last; a continuation is entered with
     one stack, however it is reached; code returns, and a unit ends,
     with the stack they started with.
   Both are stricter than anything the compiler checks; the comment at
   each definition says what the compiler does check. Both are
   conditions of [well_formed_unit]. FlambdaTrapSpec.v has a machine
   with a stack of handlers and the statement of what the discipline
   buys: a unit that follows it, and whose binders are distinct,
   behaves on that machine as it does on this one. Of the first check
   it has the statement that, in a unit that passes it, the
   [Apply_cont]s that to_cmm takes for raises are those that the term
   says are raises. FlambdaTrapProofs.v has examples of terms that each
   check accepts and refuses.

   The machine enforces the part of this that keeps a static binding from
   being executed more than once. Calling code whose body binds something
   static is undefined, as is a [Let_cont_recursive] one of whose handlers
   does (FlambdaSemanticsSpec.binds_statics); and binding a name that is
   bound already is undefined
   (FlambdaSemanticsSpec.eval_static_bindings). It follows, for any term,
   that what a code id or a symbol is bound to never changes during a run
   ([step_extends_statics_statement], [run_extends_statics_statement]):
   this is what a rewrite needs, to rely in a handler on a binding made
   before the handler was defined. That the handler of a non-recursive
   continuation with static bindings postdominates its body is left to
   the check here.

   A well-formed unit, run on its own, is never undefined for one of
   these reasons ([well_formed_never_faults_statement]): [static_fault]
   is the three of them, as a test on a state. The proof does not use
   that a handler with static bindings postdominates its body: a
   non-recursive continuation is out of scope in its own handler, which
   is enough for the handler to run once at most.

   A handler says whether it is an exception handler
   (FlambdaModel.handler_is_exn_handler), and toplevel goes by that.
   [raises_to] says whether an expression uses a continuation as one: a
   call names it as its exception continuation, or a trap action names
   it.

   The proofs are in FlambdaWellFormedProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaBindersSpec FlambdaSemanticsSpec FlambdaScopeSpec.
Import ListNotations.

(* ======================================================================== *)
(* Where control may go                                                     *)
(* ======================================================================== *)

(* The continuations an expression may apply, or give to a call, that it
   does not bind itself. Code is closed, so bindings of code have none. *)
Fixpoint free_continuations (e : expr) : list continuation :=
  match e with
  | Let _ _ _ body | Let_static _ body => free_continuations body
  | Let_cont_non_recursive k h body =>
    handler_free_continuations h ++ without [k] (free_continuations body)
  | Let_cont_recursive handlers body =>
    without (map fst handlers)
      (flat_map (fun '(_, h) => handler_free_continuations h) handlers
       ++ free_continuations body)
  | Apply ap => [ap_continuation ap; ap_exn_continuation ap]
  | Apply_cont ac => apply_cont_continuations ac
  | Switch _ arms =>
    flat_map (fun arm => apply_cont_continuations (snd arm)) arms
  | Invalid => []
  end
with handler_free_continuations (h : continuation_handler)
    : list continuation :=
  match h with
  | Handler _ _ handler => free_continuations handler
  end.

(* Whether the expression uses [k] as an exception handler: some call
   has it as its exception continuation, or some trap action names it. *)
Definition traps (k : continuation) (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some action => Pos.eqb k (trap_action_handler action)
  | None => false
  end.

Fixpoint raises_to (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => raises_to k body
  | Let_cont_non_recursive _ h body =>
    handler_raises_to k h || raises_to k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_raises_to k h) handlers
    || raises_to k body
  | Apply ap => Pos.eqb k (ap_exn_continuation ap)
  | Apply_cont ac => traps k ac
  | Switch _ arms => existsb (fun arm => traps k (snd arm)) arms
  | Invalid => false
  end
with handler_raises_to (k : continuation) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => raises_to k handler
  end.

(* ======================================================================== *)
(* Static bindings are at toplevel                                          *)
(* ======================================================================== *)

(* Whether the handler [h] of the non-recursive continuation [k], bound
   at toplevel around [body], is at toplevel too: it is not an exception
   handler, and all [body] can do is to apply [k] or raise to the
   exception continuation of the unit. In a well-formed unit a handler
   that is not an exception handler is not used as one
   ([exn_handlers_well_formed]). *)
Definition handler_at_unit_toplevel (unit_exn k : continuation)
    (h : continuation_handler) (body : expr) : bool :=
  negb (handler_is_exn_handler h)
  && all_among (free_continuations body) [k; unit_exn].

(* CR mshinwell: this refuses a [Let_static] that is not at toplevel even
   when it binds only code. The compiler stops on one that binds a symbol
   and lets one of code alone through (simplify/simplify_named.ml, the case
   [Static_consts]). So the condition is stricter than the compiler, and a
   unit that the compiler builds could fail it; none of the examples does.
   Discrepancy C0-3; no decision yet. *)

(* Every [Let_static] of the expression is at toplevel, given whether the
   expression itself is ([toplevel]) and the exception continuation of the
   unit. *)
Fixpoint statics_at_unit_toplevel (unit_exn : continuation) (toplevel : bool)
    (e : expr) : bool :=
  match e with
  | Let _ _ _ body => statics_at_unit_toplevel unit_exn toplevel body
  | Let_static bindings body =>
    toplevel
    && forallb (binding_statics_at_unit_toplevel unit_exn) bindings
    && statics_at_unit_toplevel unit_exn toplevel body
  | Let_cont_non_recursive k h body =>
    handler_statics_at_unit_toplevel unit_exn
      (toplevel && handler_at_unit_toplevel unit_exn k h body) h
    && statics_at_unit_toplevel unit_exn toplevel body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_statics_at_unit_toplevel unit_exn false h)
      handlers
    && statics_at_unit_toplevel unit_exn toplevel body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_statics_at_unit_toplevel (unit_exn : continuation)
    (toplevel : bool) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => statics_at_unit_toplevel unit_exn toplevel handler
  end
with binding_statics_at_unit_toplevel (unit_exn : continuation)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_statics_at_unit_toplevel unit_exn c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_statics_at_unit_toplevel (unit_exn : continuation) (c : code)
    : bool :=
  match c with
  | Code _ _ _ _ _ body => statics_at_unit_toplevel unit_exn false body
  end.

(* ======================================================================== *)
(* Static bindings are outside loops and code                               *)
(* ======================================================================== *)

(* CR mshinwell: which rule of place is a unit to keep? Simplify keeps
   its own account of what is at the toplevel of the unit that it is
   given, and stops on a [Let] of static constants that binds a symbol
   and is not there (simplify/simplify_named.ml, the case
   [Static_consts]; a group that binds only code passes).
   [statics_at_unit_toplevel] is the model's reading of that rule and
   is not the same test: the CR at its definition has one way in which
   the two differ. Simplify's own output does not keep its rule: a
   handler that has one inlinable use, which is an [Apply_cont] without
   a trap action, can be put in the place of the use with the static
   bindings at its head, and nothing asks where it lands
   (simplify/simplify_let_cont_expr.ml, [is_single_inlinable_use];
   simplify/simplify_apply_cont_expr.ml). The uses that are left when
   the handler of a recursive continuation has been simplified, which
   are of continuations bound outside it, are marked as not inlinable
   (simplify/simplify_let_cont_expr.ml, [simplify_handler], with
   [CUE.mark_non_inlinable]), so a handler bound outside a recursive
   handler is not put in the place of a use inside it; a continuation
   bound inside a loop may still be inlined there. So a static binding
   at the toplevel of the unit can end, in the output, in the handler
   of a non-recursive continuation that is not under a recursive
   handler and not in code, and Simplify does not take such a term
   back. The rule below is a restriction of the text that is enough
   for the model, with the other conditions of a well-formed unit:
   the run of a well-formed unit from the empty static environment
   reaches none of the states that the machine makes undefined to keep
   static bindings from being executed twice
   ([well_formed_never_faults_statement]). It is not said to be the
   set of the terms that Simplify returns, nor of those that To_cmm
   compiles rightly.
   To_cmm does not check it whole. After its own substitution, an
   update of a static whose value is a Cmm variable is emitted where
   that variable is bound, and any other update is sequenced at the
   binding of the static (to_cmm/to_cmm_shared.ml, [make_update]).
   Symbol initialisations that are left over at the end of the body
   of code are a fatal error (to_cmm/to_cmm_set_of_closures.ml); those
   left over in the handler of a loop are reported and dropped
   (to_cmm/to_cmm_expr.ml). The other course is to keep Simplify's
   rule and except the handlers that can only leave the unit. This one
   is taken provisionally. Decision D60. *)

(* Every [Let_static] of the expression is outside the handlers of
   recursive continuations and outside the body of code, given whether
   the expression itself is ([outside]). The handler of a non-recursive
   continuation is where its [Let_cont] is; the handlers of a recursive
   one are in a loop, and so is all that is in them, the handler of a
   non-recursive continuation too; the body of code is in code, and so
   is all that is in it.

   This is a test of the text. It does not say that a handler runs once
   at most: a handler in a loop that the run enters once is refused.
   What it gives, with the other conditions of a well-formed unit, is
   that the run of such a unit from the empty static environment
   reaches none of the states that the machine makes undefined to keep
   static bindings from being executed twice
   ([well_formed_never_faults_statement]). It says nothing of whether a
   static name is bound where it is mentioned: that is the check of
   FlambdaStaticsInScopeSpec.v.

   A group that binds only code is refused in a loop and in code as a
   group that binds a symbol is. *)
Fixpoint statics_outside_loops_and_code (outside : bool) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => statics_outside_loops_and_code outside body
  | Let_static bindings body =>
    outside
    && forallb binding_statics_outside_loops_and_code bindings
    && statics_outside_loops_and_code outside body
  | Let_cont_non_recursive _ h body =>
    handler_statics_outside_loops_and_code outside h
    && statics_outside_loops_and_code outside body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_statics_outside_loops_and_code false h)
      handlers
    && statics_outside_loops_and_code outside body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_statics_outside_loops_and_code (outside : bool)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => statics_outside_loops_and_code outside handler
  end
with binding_statics_outside_loops_and_code (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ c => code_statics_outside_loops_and_code c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_statics_outside_loops_and_code (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => statics_outside_loops_and_code false body
  end.

(* A term that passes the model's reading of Simplify's rule passes this
   one. *)
Definition at_unit_toplevel_outside_loops_and_code_statement : Prop :=
  ∀ (unit_exn : continuation) (toplevel : bool) (e : expr),
    statics_at_unit_toplevel unit_exn toplevel e = true ->
    statics_outside_loops_and_code toplevel e = true.

(* ======================================================================== *)
(* Static names are bound once                                              *)
(* ======================================================================== *)

Definition code_ids_of_binding (binding : static_binding) : list code_id :=
  match binding with
  | Bind_code id _ => [id]
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end.

Definition symbols_of_binding (binding : static_binding) : list symbol :=
  match binding with
  | Bind_symbol s _ => [s]
  | Bind_code _ _ | Bind_deleted_code _ => []
  end.

(* The names the [Let_static]s of an expression bind, by [names_of] for
   one binding. Deleted code is not bound, as in the semantics. Code is
   not looked into: in a well-formed unit it binds nothing. *)
Fixpoint static_names (names_of : static_binding -> list positive)
    (e : expr) : list positive :=
  match e with
  | Let _ _ _ body => static_names names_of body
  | Let_static bindings body =>
    flat_map names_of bindings ++ static_names names_of body
  | Let_cont_non_recursive _ h body =>
    handler_static_names names_of h ++ static_names names_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_static_names names_of h) handlers
    ++ static_names names_of body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_static_names (names_of : static_binding -> list positive)
    (h : continuation_handler) : list positive :=
  match h with
  | Handler _ _ handler => static_names names_of handler
  end.

(* ======================================================================== *)
(* Sets of closures are whole                                               *)
(* ======================================================================== *)

(* A closure of a set: the function slots of the set are distinct, the
   closure is at one of them, and the functions of the set have each
   their own code. *)
Definition set_well_formed (slot : function_slot)
    (function_decls : list (function_slot * code_id)) : bool :=
  is_bound slot function_decls
  && distinct (map fst function_decls)
  && distinct (map snd function_decls).

Definition named_sets_well_formed (d : named) : bool :=
  match d with
  | Closure_of_set slot function_decls _ =>
    set_well_formed slot function_decls
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    true
  end.

(* Whether a binding binds the symbol to the constant. *)
Definition binds_symbol_to (s : symbol) (const : static_const)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s' const' =>
    Pos.eqb s s' && if static_const_eq_dec const const' then true else false
  | Bind_code _ _ | Bind_deleted_code _ => false
  end.

(* A binding of a group which binds a symbol to a closure of a static
   set: the set has a symbol for each of its functions, in their order;
   the symbol bound is the one the set has for the slot; and the group
   binds every symbol of the set to the closure of the same set at that
   symbol's slot. This is what the one pattern of the compiler binds
   ([Bound_static.Pattern.Set_of_closures], against which
   [Static_const.match_against_bound_static_pattern] checks the function
   slots of the set, in order). *)
Definition static_set_whole (bindings : list static_binding)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol s
      (Static_const_closure_of_set slot closure_symbols function_decls
         value_slots) =>
    set_well_formed slot function_decls
    && (if
          list_eq_dec Pos.eq_dec (map fst closure_symbols)
            (map fst function_decls)
        then true
        else false)
    && match lookup slot closure_symbols with
       | Some s' => Pos.eqb s s'
       | None => false
       end
    && forallb
         (fun '(slot', s') =>
            existsb
              (binds_symbol_to s'
                 (Static_const_closure_of_set slot' closure_symbols
                    function_decls value_slots))
              bindings)
         closure_symbols
  | Bind_symbol _
      (Static_const_set_of_closures _ _ | Static_const_block _ _
      | Static_const_boxed_number _ | Static_const_immutable_string _
      | Static_const_immutable_value_array _)
  | Bind_code _ _ | Bind_deleted_code _ =>
    true
  end.

(* The code ids of the functions of the static set of which a binding
   binds a closure; none for another binding. *)
Definition static_set_code_ids (binding : static_binding) : list code_id :=
  match binding with
  | Bind_symbol _ (Static_const_set_of_closures code _) => [code]
  | Bind_symbol _ (Static_const_closure_of_set _ _ function_decls _) =>
    map snd function_decls
  | Bind_symbol _
      (Static_const_block _ _ | Static_const_boxed_number _
      | Static_const_immutable_string _
      | Static_const_immutable_value_array _)
  | Bind_code _ _ | Bind_deleted_code _ =>
    []
  end.

(* Whether the second binding binds one of the symbols of the static set
   of which the first binds a closure. In a group that binds its sets
   whole, the two are then bindings of one set. *)
Definition binds_symbol_of_set (binding binding' : static_binding) : bool :=
  match binding, binding' with
  | Bind_symbol _ (Static_const_closure_of_set _ closure_symbols _ _),
    Bind_symbol s' _ =>
    among s' (map snd closure_symbols)
  | _, _ => false
  end.

(* The static sets that a group binds have no code in common. A set is
   counted once, though each of its symbols has a binding of its own.
   Simplify gives each code id of the sets of one group one new code id
   (Simplify_set_of_closures_context, the old-to-new code ids of all the
   sets), and fails on a group two of whose sets have the same code.
   Two sets bound by two groups may have the same. *)
Fixpoint static_sets_code_disjoint (bindings : list static_binding) : bool :=
  match bindings with
  | [] => true
  | binding :: bindings =>
    forallb
      (fun binding' =>
         orb (binds_symbol_of_set binding binding')
           match
             inter (static_set_code_ids binding)
               (static_set_code_ids binding')
           with
           | [] => true
           | _ :: _ => false
           end)
      bindings
    && static_sets_code_disjoint bindings
  end.

(* Every closure of a set that the expression makes is at a function slot
   of its set, whose function slots are distinct and whose functions
   have each their own code; every static set is bound whole by the
   group that binds one of its symbols; and the static sets of one group
   have no code in common. Code and handlers are looked into. *)
Fixpoint sets_well_formed (e : expr) : bool :=
  match e with
  | Let _ _ d body => named_sets_well_formed d && sets_well_formed body
  | Let_static bindings body =>
    forallb (static_set_whole bindings) bindings
    && static_sets_code_disjoint bindings
    && forallb binding_sets_well_formed bindings
    && sets_well_formed body
  | Let_cont_non_recursive _ h body =>
    handler_sets_well_formed h && sets_well_formed body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_sets_well_formed h) handlers
    && sets_well_formed body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_sets_well_formed (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => sets_well_formed handler
  end
with binding_sets_well_formed (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_sets_well_formed c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_sets_well_formed (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => sets_well_formed body
  end.

(* ======================================================================== *)
(* Numbers are in range                                                     *)
(* ======================================================================== *)

(* The numbers of a term are among those that the compiler's can be.
   - The tag of every block made, by a [Make_block] or statically, and
     the tag that a [Block_load] names, is at least 0 and below 251.
   - Every constant is in the range of its kind, wherever it is: an
     argument, a value slot, a field of a static block, the contents of
     a static boxed number, the scrutinee of a [Switch].
   - The field and the size of a [Block_load] and the discriminants of
     a [Switch] are integers of the target's OCaml, and the bytes of a
     string are bytes.
   FlambdaPrimitiveSpec.v says what the ranges are (the section "Numbers
   in range").

   This is not stricter than the compiler: there each of these numbers
   is of a type whose values are in range ([Tag.Scannable.t],
   [Reg_width_const.t], [Target_ocaml_int.t], [string]), so every term
   it builds passes. The model's numbers are integers, and without the
   check it has terms that the compiler has not: a block with tag 300,
   an int8 that is 200. The machine does not test the tag of a block,
   and the arithmetic has no value on a constant out of range.

   FlambdaRangeSpec.v says what the check gives: in the run of a unit
   that passes, and so of a well-formed unit, every number that a state
   holds is in range. *)

Definition simple_in_range (s : simple) : bool :=
  match s with
  | Const c => const_in_range c
  | Var _ | Symbol _ => true
  end.

Definition value_slots_in_range (value_slots : list (value_slot * simple))
    : bool :=
  forallb (fun slot => simple_in_range (snd slot)) value_slots.

Definition named_in_range (d : named) : bool :=
  match d with
  | Simple s => simple_in_range s
  | Prim p args => prim_in_range p && forallb simple_in_range args
  | Set_of_closures _ value_slots | Closure_of_set _ _ value_slots =>
    value_slots_in_range value_slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    simple_in_range closure
  end.

Definition static_const_in_range (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures _ value_slots
  | Static_const_closure_of_set _ _ _ value_slots =>
    value_slots_in_range value_slots
  | Static_const_block tag fields =>
    tag_in_range tag && forallb simple_in_range fields
  | Static_const_boxed_number contents => simple_in_range contents
  | Static_const_immutable_string bytes => forallb byte_in_range bytes
  | Static_const_immutable_value_array elements =>
    forallb simple_in_range elements
  end.

Definition apply_cont_in_range (ac : apply_cont) : bool :=
  forallb simple_in_range (ac_args ac).

Definition apply_in_range (ap : apply_expr) : bool :=
  simple_in_range (ap_callee ap)
  && forallb simple_in_range (ap_args ap)
  && forallb simple_in_range (exn_extra_simples ap).

Definition arm_in_range (arm : Z * apply_cont) : bool :=
  ocaml_int_in_range (fst arm) && apply_cont_in_range (snd arm).

(* Every number of the expression is in range. Code and handlers are
   looked into. *)
Fixpoint numbers_in_range (e : expr) : bool :=
  match e with
  | Let _ _ d body => named_in_range d && numbers_in_range body
  | Let_static bindings body =>
    forallb binding_numbers_in_range bindings && numbers_in_range body
  | Let_cont_non_recursive _ h body =>
    handler_numbers_in_range h && numbers_in_range body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_numbers_in_range h) handlers
    && numbers_in_range body
  | Apply ap => apply_in_range ap
  | Apply_cont ac => apply_cont_in_range ac
  | Switch scrutinee arms =>
    simple_in_range scrutinee && forallb arm_in_range arms
  | Invalid => true
  end
with handler_numbers_in_range (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => numbers_in_range handler
  end
with binding_numbers_in_range (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_numbers_in_range c
  | Bind_deleted_code _ => true
  | Bind_symbol _ const => static_const_in_range const
  end
with code_numbers_in_range (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => numbers_in_range body
  end.

(* ======================================================================== *)
(* Exception handlers                                                       *)
(* ======================================================================== *)

(* Whether the [Apply_cont] applies [k] as an ordinary continuation: it
   applies it, and is not a raise. *)
Definition applies (k : continuation) (ac : apply_cont) : bool :=
  Pos.eqb k (ac_continuation ac) && negb (is_raise ac).

(* Whether the expression applies [k] as an ordinary continuation: by an
   [Apply_cont] that is not a raise, or as the return continuation of a
   call. *)
Fixpoint applies_cont (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => applies_cont k body
  | Let_cont_non_recursive _ h body =>
    handler_applies_cont k h || applies_cont k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_applies_cont k h) handlers
    || applies_cont k body
  | Apply ap => Pos.eqb k (ap_continuation ap)
  | Apply_cont ac => applies k ac
  | Switch _ arms => existsb (fun arm => applies k (snd arm)) arms
  | Invalid => false
  end
with handler_applies_cont (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => applies_cont k handler
  end.

(* Whether some call in the expression gives extra arguments to [k], its
   exception continuation. *)
Fixpoint gives_extra_args (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => gives_extra_args k body
  | Let_cont_non_recursive _ h body =>
    handler_gives_extra_args k h || gives_extra_args k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_gives_extra_args k h) handlers
    || gives_extra_args k body
  | Apply ap =>
    Pos.eqb k (ap_exn_continuation ap) && negb (no_exn_extra_args ap)
  | Apply_cont _ | Switch _ _ | Invalid => false
  end
with handler_gives_extra_args (k : continuation) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => gives_extra_args k handler
  end.

(* Whether some [Apply_cont] of the expression that is not a raise has a
   trap action that names [k]: it installs [k], or removes it on the way
   to another continuation. *)
Definition traps_apart (k : continuation) (ac : apply_cont) : bool :=
  traps k ac && negb (is_raise ac).

Fixpoint installs_or_removes (k : continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => installs_or_removes k body
  | Let_cont_non_recursive _ h body =>
    handler_installs_or_removes k h || installs_or_removes k body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => handler_installs_or_removes k h) handlers
    || installs_or_removes k body
  | Apply_cont ac => traps_apart k ac
  | Switch _ arms => existsb (fun arm => traps_apart k (snd arm)) arms
  | Apply _ | Invalid => false
  end
with handler_installs_or_removes (k : continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => installs_or_removes k handler
  end.

(* The continuations that the expression does not bind, where it is the
   body of code or a unit and [exn] its exception continuation: none but
   [exn] is used as an exception handler; [exn] is named by calls, as
   their exception continuation, and by the trap actions of raises, and
   is not installed or removed by another trap action; and no call gives
   one of them extra arguments. For code that is closed the others are
   its return continuation alone.

   The model is stricter than the compiler here.
   - That the return continuation is not used as an exception handler:
     for code, to_cmm fails on a trap action that names the return
     continuation of a function
     (To_cmm_expr.translate_jump_to_continuation, by
     To_cmm_env.get_cmm_continuation). For a unit nothing checks it:
     to_cmm gives the return continuation of the unit a handler of its
     own, as for an ordinary continuation (To_cmm.unit0), and lets
     through a trap action that names it. Discrepancy R1-22.
   - That no call gives extra arguments to one of them: Simplify stops
     on such a call when it visits it, and only by accident (an
     [Invalid_argument] in Continuation_uses_env.add_continuation_use).
     Without this the machine would leave the term with the exception
     followed by the extra arguments, where the exception continuation
     of code and of a unit takes the exception alone. It is an invariant
     of what the front end and the transformations build, which give
     extra arguments only to the handlers of a [try]. It is not an
     agreement of kinds, and it says nothing of a way out that is
     applied to the wrong number of values in another way, which stays
     defined. Discrepancy C5-5.
   For the second thing asked, that [exn] is not installed or removed
   apart from a raise, the compiler has a check when it visits the term:
   to_cmm stops on a trap action that names the exception continuation
   of a function, or of a unit, unless it is that of a raise
   (To_cmm_expr.translate_jump_to_continuation, by
   To_cmm_env.get_cmm_continuation, which does not find it). Discrepancy
   C5-9. *)
Definition ways_out_well_formed (exn : continuation) (e : expr) : bool :=
  forallb
    (fun k =>
       ((if Pos.eqb k exn then negb (installs_or_removes k e)
         else negb (raises_to k e))
        && negb (gives_extra_args k e))%bool)
    (free_continuations e).

(* An exception handler has a first parameter, for the exception. *)
Definition takes_exception (h : continuation_handler) : bool :=
  match h with
  | Handler [] _ _ => false
  | Handler (_ :: _) _ _ => true
  end.

(* Exception handlers are kept apart from the other continuations.
   - A handler that says it is an exception handler
     ([handler_is_exn_handler]) is not applied as an ordinary
     continuation in the body of its [Let_cont], and has a parameter for
     the exception. ([Trap_action]: "continuations cannot be used both
     as an exception handler and as a normal continuation". When it
     visits them, to_cmm fails on an [Apply_cont] of an exception handler
     that has no [Pop] (To_cmm_expr.translate_raise) and on an exception
     handler with no parameter (To_cmm_env.add_exn_handler).)
   - A handler that does not say so is not used as an exception handler
     there: no trap action names its continuation and no call has it as
     its exception continuation.
   - No handler of a recursive [Let_cont] is an exception handler
     ([Let_cont_expr.create_recursive] fails on one), and none is used
     as one.
   - In code, [ways_out_well_formed], and the exception continuation is
     not applied as an ordinary continuation.

   The model is stricter than the compiler here. Nothing in Flambda 2
   checks that a continuation used as an exception handler says that it
   is one, nor that an exception handler is not the return continuation
   of a call: to_cmm treats an [Apply_cont] by whether the continuation
   applied is an exception handler (To_cmm_expr.apply_cont, by
   To_cmm_env.is_exn_handler) and does not look at the handler that a
   trap action or a call names. The front end marks the handlers it
   installs, and Simplify removes the trap actions that name a handler
   when it clears the mark (Simplify_common.clear_demoted_trap_action).
   Discrepancy C5-8.

   The check goes by names and does not follow rebinding, so it may
   refuse a term that binds one name twice, once as an exception
   handler. *)
Fixpoint exn_handlers_well_formed (e : expr) : bool :=
  match e with
  | Let _ _ _ body => exn_handlers_well_formed body
  | Let_static bindings body =>
    forallb binding_exn_handlers_well_formed bindings
    && exn_handlers_well_formed body
  | Let_cont_non_recursive k h body =>
    (if handler_is_exn_handler h
     then negb (applies_cont k body) && takes_exception h
     else negb (raises_to k body))
    && handler_exn_handlers_well_formed h
    && exn_handlers_well_formed body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k, h) =>
         (negb (handler_is_exn_handler h)
          && negb (raises_to k body)
          && negb (existsb (fun '(_, h') => handler_raises_to k h') handlers)
          && handler_exn_handlers_well_formed h)%bool)
      handlers
    && exn_handlers_well_formed body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_exn_handlers_well_formed (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => exn_handlers_well_formed handler
  end
with binding_exn_handlers_well_formed (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_exn_handlers_well_formed c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_exn_handlers_well_formed (c : code) : bool :=
  match c with
  | Code _ _ _ exn _ body =>
    ways_out_well_formed exn body
    && negb (applies_cont exn body)
    && exn_handlers_well_formed body
  end.

(* A unit, as code: of the continuations it does not bind, only its
   exception continuation is used as an exception handler, so its return
   continuation is not; no call gives one of them extra arguments; and
   its exception continuation is not applied as an ordinary
   continuation. *)
Definition unit_exn_handlers_well_formed (unit_exn : continuation)
    (e : expr) : bool :=
  ways_out_well_formed unit_exn e
  && negb (applies_cont unit_exn e)
  && exn_handlers_well_formed e.

(* ======================================================================== *)
(* Trap actions                                                             *)
(* ======================================================================== *)

(* The exception handlers installed, the last installed first. *)
Definition trap_stack : Type := list continuation.

Fixpoint trap_stack_eqb (ts ts' : trap_stack) : bool :=
  match ts, ts' with
  | [], [] => true
  | k :: ts, k' :: ts' => Pos.eqb k k' && trap_stack_eqb ts ts'
  | _, _ => false
  end.

(* What an expression asks of the continuations it does not bind: each
   one it may go to, with the handlers installed when it gets there. *)
Definition demands : Type := list (continuation * trap_stack).

(* How the demands say a continuation is entered. *)
Inductive entry : Type :=
  | Not_entered
  | Entered_with (ts : trap_stack)
  | Entered_with_two_stacks.

Definition entry_of (k : continuation) (ds : demands) : entry :=
  match map snd (filter (fun d => Pos.eqb (fst d) k) ds) with
  | [] => Not_entered
  | ts :: others =>
    if forallb (trap_stack_eqb ts) others then Entered_with ts
    else Entered_with_two_stacks
  end.

(* The demands that are not on the continuations [ks], which go out of
   scope. None may be for a stack that holds one of [ks]. *)
Definition leave_scope (ks : list continuation) (ds : demands)
    : option demands :=
  let outer := filter (fun d => negb (among (fst d) ks)) ds in
  if forallb (fun d => forallb (fun k => negb (among k (snd d))) ks) outer
  then Some outer
  else None.

(* An [Apply_cont] under the stack [ts]. A [Push] of [k] asks that the
   handler [k] run under [ts], and goes on with [k] installed. A [Pop]
   must be of the handler installed last. *)
Definition apply_cont_demands (ts : trap_stack) (ac : apply_cont)
    : option demands :=
  match ac_trap_action ac with
  | None => Some [(ac_continuation ac, ts)]
  | Some (Push k) => Some [(k, ts); (ac_continuation ac, k :: ts)]
  | Some (Pop k _) =>
    match ts with
    | top :: rest =>
      if Pos.eqb top k then Some [(ac_continuation ac, rest)] else None
    | [] => None
    end
  end.

(* A call under the stack [ts]: its exception continuation is the
   handler installed last, and runs under what is below it; the call
   returns under [ts]. *)
Definition apply_demands (ts : trap_stack) (ap : apply_expr)
    : option demands :=
  match ts with
  | top :: rest =>
    if Pos.eqb top (ap_exn_continuation ap)
    then Some [(ap_continuation ap, ts); (ap_exn_continuation ap, rest)]
    else None
  | [] => None
  end.

Fixpoint arms_demands (ts : trap_stack) (arms : list (Z * apply_cont))
    : option demands :=
  match arms with
  | [] => Some []
  | (_, ac) :: arms =>
    let* ds := apply_cont_demands ts ac in
    let* ds' := arms_demands ts arms in
    Some (ds ++ ds')
  end.

(* The body of code starts with its exception continuation as the one
   handler installed, returns so, and raises to it so. (Its return and
   exception continuations are two names, by
   FlambdaBindersSpec.binders_distinct: were they one, it would stand
   for the return continuation, and a raise would return.) *)
Definition code_demands_met (ret exn : continuation) (ds : demands) : bool :=
  forallb
    (fun d =>
       if Pos.eqb (fst d) exn then trap_stack_eqb (snd d) []
       else (Pos.eqb (fst d) ret && trap_stack_eqb (snd d) [exn])%bool)
    ds.

(* What has been found of the handlers of a recursive group: those
   checked so far, and the demands of the body and of those. *)
Definition group_state : Type := list continuation * demands.

Fixpoint iterate {A : Type} (n : nat) (f : A -> option A) (x : A)
    : option A :=
  match n with
  | O => Some x
  | S n =>
    let* x := f x in
    iterate n f x
  end.

(* The demands of an expression under the stack [ts], if it follows the
   discipline of trap actions:
   - a [Pop] names the handler installed last. A raise is a [Pop] of the
     handler that it then applies;
   - the exception continuation of a call is the handler installed last;
   - a continuation is entered with one stack, however it is reached:
     all that apply it do so, after their trap actions, with the same
     handlers installed, and its handler is checked under those. For an
     exception handler, that stack is the one under it where it is
     installed, which is the same at each [Push], and is what is left
     when a raise or a call comes to it;
   - a name that is on the stack is not bound again, and is not on the
     stack of a continuation that is entered from outside its scope. So
     a name on the stack means one handler.
   The check is for a term whose binders are distinct
   (FlambdaBindersSpec.binders_distinct, which a well-formed unit is
   asked before this): it does not ask again that the continuations of
   a recursive [Let_cont] have distinct names, nor that the return and
   exception continuations of code are two names.
   The stack with which a continuation is entered is not written in the
   term: it is found from the applications of the continuation, in the
   body of its [Let_cont] and, for a recursive one, in the handlers of
   its group. It cannot be the stack where the [Let_cont] stands: the
   front end defines the continuation of the body of a [try] outside
   the [try] and enters it with a [Push], and Simplify lifts
   continuations out of handlers. A handler that nothing applies is not
   checked: it never runs.

   The model is stricter than the compiler here, in part. Flambda 2
   does not check handler names against the installed trap stack (a
   review comment in terms/flambda.ml asks for a check of an
   [Apply_cont] that has a trap action). The selection of
   instructions asks that a continuation be reached with one stack
   and that no [Pop] find the stack empty (backend/select_utils.ml,
   [set_traps] and [combine_traps], which do not look at the handler
   that a [Pop] names). A later pass stops the compiler where a [Pop]
   of a jump or of a return names another handler than the one on top
   ([Stack_offset_and_exn.process_basic] of the same file, run by
   [update_cfg] from backend/cfg_selectgen.ml; on the other path of
   the backend, [compute_block_end_trap_stack] of
   backend/ssa/ssa.ml). So of the name of such a [Pop] the model asks
   what the backend asks. Of a raise and of the exception continuation
   of a call it asks more: to_cmm does not keep the handler that a
   raise names (to_cmm/to_cmm_expr.ml, [translate_raise]) and does not
   look at the one that a call names ([translate_apply]), and the
   backend goes by the handler on top. The front end builds terms
   that pass (from_lambda/lambda_to_flambda.ml, the case [Ltrywith];
   closure_conversion.ml, [close_raise0]). Discrepancy C5-1. *)
Fixpoint trap_demands (ts : trap_stack) (e : expr) : option demands :=
  match e with
  | Let _ _ _ body => trap_demands ts body
  | Let_static bindings body =>
    if forallb binding_traps_well_formed bindings
    then trap_demands ts body
    else None
  | Let_cont_non_recursive k h body =>
    if among k ts then None
    else
      let* ds := trap_demands ts body in
      let* outer := leave_scope [k] ds in
      match entry_of k ds with
      | Not_entered => Some outer
      | Entered_with entered =>
        (* [k] is not in scope in its handler: what the handler asks is
           asked of the continuations around the [Let_cont]. *)
        if among k entered then None
        else
          let* ds' := handler_trap_demands entered h in
          Some (outer ++ ds')
      | Entered_with_two_stacks => None
      end
  | Let_cont_recursive handlers body =>
    let ks := map fst handlers in
    if existsb (fun k => among k ts) ks then None
    else
      let* ds := trap_demands ts body in
      (* Each pass checks the handlers that are known to be entered and
         have not been checked; as many passes as handlers reach them
         all, which the test after the passes confirms. *)
      let pass (state : group_state) : option group_state :=
        fold_left
          (fun state '(k, h) =>
             let* '(checked, ds) := state in
             if among k checked then Some (checked, ds)
             else
               match entry_of k ds with
               | Not_entered => Some (checked, ds)
               | Entered_with entered =>
                 let* ds' := handler_trap_demands entered h in
                 Some (k :: checked, ds ++ ds')
               | Entered_with_two_stacks => None
               end)
          (Some state) handlers
      in
      let* '(checked, ds) := iterate (length handlers) pass ([], ds) in
      if forallb
           (fun k =>
              match entry_of k ds with
              | Not_entered => true
              | Entered_with _ => among k checked
              | Entered_with_two_stacks => false
              end)
           ks
      then leave_scope ks ds
      else None
  | Apply ap => apply_demands ts ap
  | Apply_cont ac => apply_cont_demands ts ac
  | Switch _ arms => arms_demands ts arms
  | Invalid => Some []
  end
with handler_trap_demands (ts : trap_stack) (h : continuation_handler)
    : option demands :=
  match h with
  | Handler _ _ handler => trap_demands ts handler
  end
with binding_traps_well_formed (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_traps_well_formed c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_traps_well_formed (c : code) : bool :=
  match c with
  | Code _ _ ret exn _ body =>
    match trap_demands [exn] body with
    | Some ds => code_demands_met ret exn ds
    | None => false
    end
  end.

(* A unit follows the discipline: it starts with its exception
   continuation as the one handler installed, and leaves as code does,
   by its exception continuation or by any other continuation that it
   does not bind. *)
Definition traps_well_formed (unit_exn : continuation) (e : expr) : bool :=
  match trap_demands [unit_exn] e with
  | Some ds =>
    forallb
      (fun d =>
         trap_stack_eqb (snd d)
           (if Pos.eqb (fst d) unit_exn then [] else [unit_exn]))
      ds
  | None => false
  end.

(* ======================================================================== *)
(* Well-formed units                                                        *)
(* ======================================================================== *)

(* What a unit is given by where it is run: the continuation that it
   returns to and its exception continuation. These are the parts of
   the compiler's [Flambda_unit.t] that the conditions need
   (return_continuation and exn_continuation there). It has more: the
   module symbol and the region of the unit, for two. What else of it a
   condition comes to need is to be a field here, so that no use of
   [well_formed_unit] changes. *)
Record unit_interface : Type := Unit_interface {
  unit_return : continuation;
  unit_exn : continuation
}.

(* A unit, with its interface: its static bindings are outside loops
   and code, and bind each code id and each symbol once; its variables
   and its continuations are in scope, the names of each binder are
   distinct and its code has no free name but its own binders; its sets
   of closures are whole; its numbers are in range; its exception
   handlers are apart from its other continuations; and it follows the
   discipline of trap actions. The conditions are in the order of the
   rows of the table of requests/C0-well-formed.md (1, 2, 3, 5, 6, 7 and
   8), then rows 11 and 12 of requests/C5-exceptions.md. *)
Definition well_formed_unit (u : unit_interface) (e : expr) : bool :=
  statics_outside_loops_and_code true e
  && distinct (static_names code_ids_of_binding e)
  && distinct (static_names symbols_of_binding e)
  && names_in_scope (unit_return u) (unit_exn u) e
  && binders_distinct e
  && code_closed e
  && sets_well_formed e
  && numbers_in_range e
  && unit_exn_handlers_well_formed (unit_exn u) e
  && traps_well_formed (unit_exn u) e.

(* ======================================================================== *)
(* What is bound stays bound                                                *)
(* ======================================================================== *)

(* Everything the first static environment binds, the second binds to the
   same thing. *)
Definition statics_extend (statics statics' : static_env) : Prop :=
  (∀ (id : code_id) (c : code),
    lookup id (static_code statics) = Some c ->
    lookup id (static_code statics') = Some c)
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values statics') = Some allocated).

Definition step_extends_statics_statement : Prop :=
  ∀ s s' : state,
    step s = Step s' ->
    statics_extend (state_statics s) (state_statics s').

Definition run_extends_statics_statement : Prop :=
  ∀ (n : nat) (s s' : state),
    run n s = Step s' ->
    statics_extend (state_statics s) (state_statics s').

(* ======================================================================== *)
(* A well-formed unit does nothing that is forbidden                        *)
(* ======================================================================== *)

Definition code_id_bound (statics : static_env) (id : code_id) : bool :=
  match lookup id (static_code statics) with
  | Some _ => true
  | None => false
  end.

Definition symbol_bound (statics : static_env) (s : symbol) : bool :=
  match lookup s (static_values statics) with
  | Some _ => true
  | None => false
  end.

(* A group binds a name twice, or a name that is bound already. *)
Definition rebinds (names_of : static_binding -> list positive)
    (is_bound : positive -> bool) (bindings : list static_binding) : bool :=
  negb (distinct (flat_map names_of bindings))
  || existsb is_bound (flat_map names_of bindings).

(* The callee is a closure of code whose body binds something static,
   or a partial application of one. (The call is undefined where it runs
   that code; with fewer arguments than the code has parameters it runs
   none. A call made where a continuation is applied, of the result of
   the full call inside an over-application, is not looked at here.) *)
Definition calls_code_binding_statics (ap : apply_expr) (vars : env)
    (statics : static_env) : bool :=
  match eval_simple vars (ap_callee ap) with
  | Some callee =>
    let closure :=
      match partial_application_of callee with
      | Some (closure, _, _) => closure
      | None => callee
      end
    in
    match closure_of statics closure with
    | Some (called, _) =>
      match lookup called (static_code statics) with
      | Some c => binds_statics (code_body c)
      | None => false
      end
    | None => false
    end
  | None => false
  end.

(* The state is one the machine makes undefined to keep static bindings
   from being executed twice, or, at an [Apply], would if the call ran
   the code of its callee. *)
Definition static_fault (s : state) : bool :=
  match state_expr s with
  | Let_static bindings _ =>
    rebinds code_ids_of_binding (code_id_bound (state_statics s)) bindings
    || rebinds symbols_of_binding (symbol_bound (state_statics s)) bindings
  | Let_cont_recursive handlers _ => handlers_bind_statics handlers
  | Apply ap =>
    calls_code_binding_statics ap (state_vars s) (state_statics s)
  | Let _ _ _ _ | Let_cont_non_recursive _ _ _ | Apply_cont _ | Switch _ _
  | Invalid =>
    false
  end.

Definition well_formed_never_faults_statement : Prop :=
  ∀ (u : unit_interface) (e : expr) (n : nat) (s : state),
    well_formed_unit u e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    static_fault s = false.

(* ======================================================================== *)
(* A well-formed unit reads no name that is not bound                       *)
(* ======================================================================== *)

(* No state of the run of a well-formed unit reads a variable that has no
   value, and if the run ends, it is by the continuation that the unit
   returns to or by its exception continuation. This is
   FlambdaScopeSpec.unit_run_statement of a well-formed unit; the test
   [reads_unbound_variable] is defined there. It says nothing yet of
   symbols and code ids: that is row 4 of the table. *)
Definition well_formed_names_bound_statement : Prop :=
  ∀ (u : unit_interface) (e : expr) (n : nat),
    well_formed_unit u e = true ->
    match run n (State e [] [] empty_static_env empty_heap no_reads) with
    | Step s => reads_unbound_variable s = false
    | Exit k _ _ _ => k = unit_return u ∨ k = unit_exn u
    | Undefined => True
    end.
