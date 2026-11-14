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

(* Lifting: a [Let] of a closure, an immutable block, a boxed number or
   an immutable array of values made a static binding of a symbol.

   This file is to be reviewed: [lift_named] and [lift_expr], against
   what Simplify does (below); [exits_plain] and [exits_plain_knowing];
   [lift_unit], which is extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; the relation [value_lifted];
   and the statements.

   What Simplify does. Where it simplifies a [Let] it may bind a symbol
   in place of the variable and replace the variable by the symbol:
   - a set of closures, when its variables are bound at normal name
     mode and each of its value slots is a constant, a symbol, or a
     variable defined at the toplevel of the unit or known to be a
     projection from a symbol, with a coercion that has no free
     variable (simplify/simplify_set_of_closures.ml,
     [type_value_slots_and_make_lifting_decision_for_one_set], the
     definition of [can_lift], and [simplify_and_lift_set_of_closures]);
     a variable defined at toplevel, in a set that may be allocated
     locally, must also be proved never to hold a local value
     ([T.never_holds_locally_allocated_values]; a projection from a
     symbol is exempt); the symbols are named after the function slots;
   - the result of a primitive that only allocates, bound at normal name
     mode, for which the simplification of the primitive asks for it
     ([try_reify]), when the type that Simplify has of the result can be
     rebuilt as a static constant: an immutable block of values, a boxed
     number, an immutable array (simplify/simplify_named.ml,
     [allow_lifting]; simplify/lifting/reification.ml, [try_to_reify],
     [create_static_const] and [lift]; types/reify.ml, [reify]); the
     symbol is named after the variable, unless an equal constant has
     been lifted already, when the symbol of that one is used
     ([DA.find_shareable_constant] in [lift]).
   Neither asks that the [Let] is at the toplevel of the unit: a
   constant whose parts are constants and symbols is lifted out of a
   function or a loop, its binding being placed at toplevel. The model
   has no name modes, no coercions and no local allocations, so none of
   these conditions is one of [lift_named]: what it lifts that Simplify
   would not is compared with a term in which it is not lifted, and
   fails there.

   What is here. [lift_expr] makes the [Let]s that it is told to into
   static bindings, each where it stands:

     Let x = Closure_of_set slot function_decls value_slots in body
       becomes
     Let_static (bind_closures closure_symbols function_decls value_slots)
       (Let x = Symbol s in body)        with s the symbol of [slot]

     Let x = Make_block tag fields in body
       becomes
     Let_static [Bind_symbol s (Static_const_block tag fields)]
       (Let x = Symbol s in body)

     Let x = Box_number kind contents in body
       becomes
     Let_static [Bind_symbol s (Static_const_boxed_number contents)]
       (Let x = Symbol s in body)

     Let x = Make_immutable_array kind elements in body
       becomes
     Let_static
       [Bind_symbol s (Static_const_immutable_value_array elements)]
       (Let x = Symbol s in body)

   The kind of the array is dropped: the static constant has none, and
   every array kind of the model is one of values
   (FlambdaPrimitiveModel.array_kind). The [Let] has no value where an
   element is not one that its kind holds
   (FlambdaPrimitiveSpec.array_made), and the binding has one wherever
   its elements have: the term lifted may be defined where the term
   was not, which the statements here, of refinements, allow.

   Likewise the closure of a set of one function whose function slot
   is not named ([Set_of_closures], to [Static_const_set_of_closures]):
   the test tool still converts a unit that has only such sets to these
   forms.

   The variable stays, bound to the symbol; following the alias and
   removing the [Let] are other passes
   (../types/rocq/FlambdaAliasesSpec.follow_aliases,
   FlambdaNormaliseSpec.remove_unused_lets). The fields and the value
   slots are not touched: where one is a variable, the static constant
   is computed where the [Let] was, from the same variable. The other
   closures of a set are not looked for: a projection of a function slot
   from [x] stays a projection, now from a symbol.

   In the code of a lifted closure, [my_closure] is then the symbol:
   Simplify simplifies the code where it lifts the set, with the type
   of [my_closure] an alias of the symbol, so that a recursive function
   calls itself through its symbol. [lift_unit] does it too, by
   FlambdaMyClosureSpec.bind_my_closures, for the code ids and symbols
   that the guess names, and checks what that needs: the unit lifted
   makes no other closure of that code
   (FlambdaMyClosureSpec.only_closures_of).

   Which [Let]s, to which symbols, and which code is that of which
   symbol, is a guess of the caller's ([lifting]), read off the term
   after Simplify; nothing is believed of it. A [Let] is named by where
   it is in the term ([path]: the handler of a non-recursive
   continuation is to one side and what follows a binding to the
   other), not by its variable, so that two [Let]s of one variable are
   told apart and renaming the term changes nothing
   ([lift_expr_equivariant_statement]). The bodies of code and the
   handlers of recursive continuations are not gone into.

   Why it is not a rewrite like the others. The variable was a closure,
   and is now a symbol bound to that closure: different values. No
   primitive and no rule of the machine tells them apart
   (FlambdaValueSpec.v: every reading is an observation), but the
   continuation that the term leaves through is given the value itself
   ([Exits]). So the term lifted does not refine the term under all
   environments, nor even run on its own, if a lifted value reaches the
   exit. What is true is stated twice:

   - [lift_simulates_statement]: run on their own, the term and the term
     lifted end alike, with values at the exit that are the same up to
     [value_lifted]: where the first has a closure, a block, a boxed
     number or an immutable array, the second may have a symbol bound to
     it, here and inside closures, blocks and arrays. The first premise
     is that the term lifted is a well-formed unit
     (FlambdaWellFormedSpec.well_formed_unit), which is what keeps each
     new binding from being executed twice and each new symbol from
     being bound already: a [Let] in a loop or in code, or two [Let]s
     given one symbol, are refused by it. The second is that the term
     lifted makes one closure only, the symbol, of each code in which
     [my_closure] was bound to a symbol.
   - [lift_unit_program_refines_statement]: if moreover the term gives
     only symbols and constants to the continuations it may leave
     through ([exits_plain_knowing]), the values at the exit are equal,
     and the term lifted refines the term, run on its own
     (FlambdaUnusedStaticsSpec.program_refines). [lift_unit] checks
     both, and this is the statement that
     FlambdaValidateSpec.validate_unit uses. A unit made from an OCaml
     source leaves with the symbol of its module block, and passes.

   [exits_plain] is a check on a term, with a statement of its own
   ([exits_plain_sound_statement]): a unit that passes it exits with symbols
   and constants only. It follows the continuations in scope: one bound
   by a [Let_cont], and the return continuation of code, never stand for
   a way out, provided that every call returns to such a one; so a
   variable may be given to them. Any other continuation, the exception
   continuation of code included, may be the way out of the unit, and
   is given symbols and constants only.

   [exits_plain_knowing] is the check that [lift_unit] makes. It is
   [exits_plain] with one more thing counted as a symbol or a constant:
   a variable known to hold one, which is a parameter of a
   non-recursive handler that every way of reaching the handler gives
   a symbol, a constant or such a variable ([params_known], [spoils]).
   A handler that raises again what it caught, in code that is kept,
   passes so. It has the same statement
   ([exits_plain_knowing_sound_statement], and from a static
   environment whose code passes the check
   [exits_plain_knowing_sound_from_statement]), and it accepts whatever
   [exits_plain] accepts ([exits_plain_knowing_accepts_statement]).

   What neither check says. The machine has no step for a call of code
   that neither the term nor the static environment it is run from
   binds: an import is an object of unknown contents, and a call of a
   C function is not in the model. So of a run in which code that the
   model has not got raises a block, "validated" says nothing: with a
   handler around the call under [exits_plain_knowing], exactly as with
   none around it under [exits_plain].

   What is not here, and makes a unit "not validated" rather than
   wrongly accepted:
   - lifting out of a function, where the binding moves to toplevel:
     that is FlambdaLiftFromCodeSpec.v, for what mentions no variable;
   - lifting out of a loop: that is FlambdaLiftFromCodeSpec.v in code
     and FlambdaLiftFromArmsSpec.v outside it, for what mentions no
     variable;
   - the sharing of equal constants: here a symbol is bound once, so
     two [Let]s cannot be given the same symbol; the two files just
     named put one binding for two [Let]s of one constant that are
     lifted out of one body (FlambdaLiftFromCodeSpec.once_each);
   - a unit that gives its return continuation, or raises in code, a
     variable that is not known to hold a symbol or a constant: see
     the CR at [exits_plain].

   Identity. The model's [Make_block] is the immutable one
   (FlambdaPrimitiveModel.v) and its values have no identity, so nothing
   lifted here has one. The array of [Make_immutable_array] has none
   either: the physical comparison of two array values has no value
   (FlambdaPrimitiveSpec.eval_phys_equal; decision D19 is open), so a
   unit that could tell one array from two is undefined before it is
   lifted. A block that is mutable or unique, and a mutable array, are
   made by primitives of the heap
   (FlambdaPrimitiveModel.heap_primitive), which [lift_named] does not
   lift. Simplify also lifts a block that is [Immutable_unique]
   (reification.ml, [create_static_const]), out of functions too, which
   changes how many blocks there are: no lifting here follows it.

   The proofs are in FlambdaLiftProofs.v, which needs no review. They
   need one law of the primitives that the others do not give: a
   primitive applied to arguments related by [value_lifted] has a value
   related to the value it had
   (FlambdaPrimitiveSpec.eval_prim_related_statement). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaWellFormedSpec FlambdaClosuresSpec FlambdaUnusedStaticsSpec
  FlambdaMyClosureSpec FlambdaImportsSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is to be lifted                                                     *)
(* ======================================================================== *)

(* What a [Let] is to be bound to in place of what it makes: for a block
   or a boxed number, a symbol; for a closure of a set, the symbols of
   all the closures of the set, by function slot. *)
Inductive lift_to : Type :=
  | Lift_to_symbol (s : symbol)
  | Lift_to_closure_symbols (closure_symbols : list (function_slot * symbol)).

(* Where a [Let] is in a term: [root] for the term itself, [p~0] for what
   follows the binding at [p], and [p~1] for the handler of the
   non-recursive continuation bound at [p]. The handlers of the
   recursive continuations bound at [p] are at [p~1~0], [p~1~1~0],
   [p~1~1~1~0] and so on, in their order: [q~0] for a handler and
   [q~1] in the place of [q] for the next, from [p~1]
   ([rec_handlers_map]). A path under one of them is under no other
   and not under [p~0], where their body is. *)
Definition path : Type := positive.

(* The path of the term itself. *)
Definition root : path := 1%positive.

(* The handlers of a group of recursive continuations each rewritten by
   [f], which is given the path of the handler; [q] is [p~1] for the
   group bound at [p]. *)
Fixpoint rec_handlers_map
    (f : path -> continuation_handler -> continuation_handler) (q : path)
    (handlers : list (continuation * continuation_handler))
    : list (continuation * continuation_handler) :=
  match handlers with
  | [] => []
  | (k, h) :: rest => (k, f (xO q) h) :: rec_handlers_map f (xI q) rest
  end.

(* Whether [f] holds of each of the handlers. *)
Fixpoint rec_handlers_forall (f : path -> continuation_handler -> bool)
    (q : path) (handlers : list (continuation * continuation_handler))
    : bool :=
  match handlers with
  | [] => true
  | (_, h) :: rest => f (xO q) h && rec_handlers_forall f (xI q) rest
  end.

(* What [f] gives for each of the handlers, in their order. *)
Fixpoint rec_handlers_bindings
    (f : path -> continuation_handler -> list static_binding) (q : path)
    (handlers : list (continuation * continuation_handler))
    : list static_binding :=
  match handlers with
  | [] => []
  | (_, h) :: rest => f (xO q) h ++ rec_handlers_bindings f (xI q) rest
  end.

(* The guess: the [Let]s to lift, by their paths; the code ids whose
   only closure is, once those are lifted, a symbol, with that symbol;
   the interface of the unit (the continuations that it returns and
   raises to), which the check of well-formedness takes; the [Let]s
   to lift out of the bodies of code, by code id and by their paths in
   the body; and the [Let]s to lift to the head of the unit, by their
   paths; the [Let]s to make aliases of symbols that a group of
   static bindings around them binds, by their paths; and the
   [Let_cont]s before which such a group is put, by their paths, each
   with the paths of the [Let]s whose bindings make the group; and
   the tables to bind at the head of the unit, each a symbol with the
   elements of an immutable array, which lift no [Let]; and the
   copies of code to make, each a list of code ids paired with new
   ones and the symbols of the static sets of closures that get the
   new ones, which lift no [Let] either; and whether the rounds that
   follow the lifting also make direct the calls of the closures that
   a [Let] makes. The last seven are not read here: the first two are
   for FlambdaLiftFromCodeSpec.lift_from_code_unit and
   FlambdaLiftFromArmsSpec.lift_to_head_unit, which the validation
   applies before [lift_unit], the next two for
   FlambdaLiftAroundSpec.lift_around_unit, which it applies after,
   the fifth for FlambdaUnusedHeadStaticsSpec.add_head_tables, which
   it applies after that, the sixth for
   FlambdaCopyCodeSpec.with_copies, which it applies last of all, and
   the last for the rounds that the validation goes through once the
   lifting is made (FlambdaValidateSpec.normalise_lifted_calling). *)
Record lifting : Type := Lifting {
  lifting_unit : unit_interface;
  lifted_lets : list (path * lift_to);
  lifted_my_closures : list (code_id * symbol);
  lifted_out_of_code : list (code_id * list (path * lift_to));
  lifted_to_head : list (path * lift_to);
  lifted_around : list (path * lift_to);
  groups_put : list (path * list path);
  tables_put : list (symbol * list simple);
  code_copied : list (list (code_id * code_id) * list symbol);
  made_closures_called : bool
}.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The static bindings that stand for a defining expression, and the
   symbol that stands for its value; none if the expression is not one
   that is lifted, or not to this. For a closure, the set must have a
   symbol for each of its function slots, in their order. *)
Definition lift_named (target : lift_to) (d : named)
    : option (list static_binding * symbol) :=
  match target, d with
  | Lift_to_symbol s,
    Prim (Variadic (Make_block (Block_kind_values tag))) fields =>
    Some ([Bind_symbol s (Static_const_block tag fields)], s)
  | Lift_to_symbol s, Prim (Variadic (Make_immutable_array _)) elements =>
    Some ([Bind_symbol s (Static_const_immutable_value_array elements)], s)
  | Lift_to_symbol s, Prim (Unary (Box_number _)) [contents] =>
    Some ([Bind_symbol s (Static_const_boxed_number contents)], s)
  | Lift_to_symbol s, Set_of_closures code value_slots =>
    Some ([Bind_symbol s (Static_const_set_of_closures code value_slots)], s)
  | Lift_to_closure_symbols closure_symbols,
    Closure_of_set slot function_decls value_slots =>
    match lookup slot closure_symbols with
    | Some s =>
      if
        list_eq_dec Pos.eq_dec (map fst closure_symbols)
          (map fst function_decls)
      then
        Some (bind_closures closure_symbols function_decls value_slots, s)
      else None
    | None => None
    end
  | _, _ => None
  end.

(* The term with the [Let]s at the paths given lifted. [at_path] is the
   path of the term. *)
Fixpoint lift_expr (lets : list (path * lift_to)) (at_path : path) (e : expr)
    : expr :=
  match e with
  | Let x kind d body =>
    let body := lift_expr lets (xO at_path) body in
    match lookup at_path lets with
    | Some target =>
      match lift_named target d with
      | Some (bindings, s) =>
        Let_static bindings (Let x kind (Simple (Symbol s)) body)
      | None => Let x kind d body
      end
    | None => Let x kind d body
    end
  | Let_static bindings body =>
    Let_static bindings (lift_expr lets (xO at_path) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (lift_handler lets (xI at_path) h)
      (lift_expr lets (xO at_path) body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (lift_expr lets (xO at_path) body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with lift_handler (lets : list (path * lift_to)) (at_path : path)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (lift_expr lets at_path handler)
  end.

(* ======================================================================== *)
(* What a unit may exit with                                                *)
(* ======================================================================== *)

(* A symbol or a constant: what has the same value in every environment. *)
Definition plain_simple (s : simple) : bool :=
  match s with
  | Symbol _ | Const _ => true
  | Var _ => false
  end.

(* An application of a continuation that is in scope, or of symbols and
   constants only. *)
Definition apply_cont_exits_plain (scope : list continuation)
    (ac : apply_cont) : bool :=
  among (ac_continuation ac) scope || forallb plain_simple (ac_args ac).

(* CR mshinwell: [exits_plain] refuses a unit that gives a variable to a
   continuation it may leave through: one that returns a variable, one
   in which code applies its exception continuation to a variable (a
   raise of an exception that is not a constant), and one in which a
   call gives a variable as an extra argument to an exception
   continuation that is not bound around the call. A lifted value
   would not always reach the exit in such a unit, so this is a choice,
   made so that what is compared at the exit can stay equality. If an
   exit compared values through symbols (decisions D1 and D6, with the
   design of the heap), the check would go and
   [lift_simulates_statement] would be the whole statement.
   [exits_plain_knowing] below refuses fewer: it accepts such a variable
   where it is known to hold a symbol or a constant. *)

(* The expression gives a variable only to continuations of [scope], and
   every call in it returns to one. The exceptions of a call go to its
   exception continuation with the extra arguments of the call after
   them: that continuation is one of [scope], or the extra arguments
   are symbols and constants. [scope] are continuations that do not
   stand for a way out of the unit: those bound by a [Let_cont] around
   the expression and, in code, the return continuation. *)
Fixpoint exits_plain (scope : list continuation) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => exits_plain scope body
  | Let_static bindings body =>
    forallb binding_exits_plain bindings && exits_plain scope body
  | Let_cont_non_recursive k h body =>
    handler_exits_plain scope h && exits_plain (k :: scope) body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) => handler_exits_plain (map fst handlers ++ scope) h)
      handlers
    && exits_plain (map fst handlers ++ scope) body
  | Apply ap =>
    among (ap_continuation ap) scope
    && (among (ap_exn_continuation ap) scope
        || forallb plain_simple (exn_extra_simples ap))
  | Apply_cont ac => apply_cont_exits_plain scope ac
  | Switch _ arms =>
    forallb (fun arm => apply_cont_exits_plain scope (snd arm)) arms
  | Invalid => true
  end
with handler_exits_plain (scope : list continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => exits_plain scope handler
  end
with binding_exits_plain (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_exits_plain c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_exits_plain (c : code) : bool :=
  match c with
  | Code _ _ ret _ _ body => exits_plain [ret] body
  end.

(* A value that is a constant or a symbol. *)
Definition plain_value (v : value) : Prop :=
  match v with
  | Value_const _ | Value_symbol _ => True
  | _ => False
  end.

(* A unit that passes the check exits with constants and symbols only. *)
Definition exits_plain_sound_statement : Prop :=
  ∀ (e : expr) (n : nat) (k : continuation) (vs : list value) (hp : heap),
    exits_plain [] e = true ->
    run n (State e [] [] empty_static_env empty_heap) = Exit k vs hp ->
    ∀ v : value, v ∈ vs -> plain_value v.

(* ======================================================================== *)
(* What a unit may exit with, where some variables are known                *)
(* ======================================================================== *)

(* [exits_plain] refuses a unit that gives a variable to a continuation
   it may leave through. The test below is the same test, with one more
   thing counted as a symbol or a constant: a variable known to hold
   one. Such a variable is a parameter of a non-recursive handler that
   every way of reaching the handler gives a symbol, a constant or such
   a variable ([params_known]).

   What a body may give the parameters of the continuations that it
   does not bind is computed in one pass ([spoils]); a parameter of a
   non-recursive handler is known in the handler if the scope of the
   handler spoils neither its position nor the whole handler.

   The ways of reaching a handler [k] bound by a [Let_cont], in the
   machine of FlambdaSemanticsSpec.v:
   1. an [Apply_cont] of [k], or an arm of a [Switch] that applies
      [k], in the scope of [k]: the arguments are those written;
   2. a call that returns to [k]: the parameters are given what the
      call returns, of which nothing is known. A handler that a call
      returns to has no parameter counted;
   3. a call whose exception continuation is [k]: the parameters are
      given what the code called, or code that it calls with its own
      exception continuation, gives that continuation, then the extra
      arguments of each call on the way, the innermost first
      (FlambdaSemanticsSpec.with_extra_args). What code gives its
      exception continuation is tested where the code is bound
      ([code_exits_plain_knowing]: that continuation is not in scope
      there, so every argument and every extra argument given to it is
      a symbol, a constant or a variable known to hold one; where the
      code gives its return and exception continuations one name, the
      name is in scope, and it stands for the return continuation:
      FlambdaSemanticsSpec.call_code binds that one first, and its
      entry hides that of the exception continuation in the
      environment of the body). The extra arguments of the calls in
      the scope of [k] are tested here, all of them and not by
      position: nested extra arguments move the positions.
   There is no other way: a continuation is not a value, a trap action
   applies nothing, and a partial application is given to the return
   continuation of its call.

   What the test does not say. The machine has no step for a call of
   code that neither the term nor the static environment it is run
   from binds: an import is an object of unknown contents, and a call
   of a C function is not in the model. So of a run in which code that
   the model has not got raises a block, "validated" says nothing,
   exactly as for a call with no handler around it under
   [exits_plain].

   What is NOT counted, on purpose:
   - a parameter of a recursive handler;
   - a parameter given a symbol on one path and anything else on
     another: one application that gives it a variable not known to
     hold a symbol or a constant is enough to refuse it;
   - a variable bound again: under a [Let] of [x], [x] is forgotten
     (FlambdaMapSpec.without); the parameters of a handler are taken
     one after the other as the machine binds them
     (FlambdaSemanticsSpec.bind_params: of two parameters of one name
     the later is the one that the handler reads), each added or
     forgotten ([params_known]). The body of code starts with nothing
     known;
   - a variable that a [Let] binds to a symbol or to a constant
     (it would be sound; no unit measured needs it). *)

(* A symbol, a constant, or a variable known to hold one. *)
Definition plain_or (plain : list variable) (s : simple) : bool :=
  match s with
  | Symbol _ | Const _ => true
  | Var x => among x plain
  end.

(* What an expression may do to the parameters of the continuations
   that it does not bind: [(k, Some i)] says that the parameter of [k]
   at position [i] may be given something that is not a symbol, a
   constant or a variable known to hold one; [(k, None)] says so of
   every parameter of [k]. *)
Definition spoilt : Type := list (continuation * option nat).

Definition spoils_at (k : continuation) (i : nat) (sp : spoilt) : bool :=
  existsb
    (fun '(k', at_) =>
       Pos.eqb k k'
       && match at_ with
          | None => true
          | Some j => Nat.eqb i j
          end)
    sp.

(* The continuations [ks] are bound: what is said of them is of other
   continuations of those names no longer. *)
Definition forget_conts (ks : list continuation) (sp : spoilt) : spoilt :=
  filter (fun '(k, _) => negb (among k ks)) sp.

(* The arguments from position [i] on that are not known to be symbols
   or constants. An application with fewer or more arguments than the
   handler has parameters has no step
   (FlambdaSemanticsSpec.apply_cont_value, by [bind_params]), so a
   position without an argument needs nothing said. *)
Fixpoint args_spoil (k : continuation) (plain : list variable) (i : nat)
    (args : list simple) : spoilt :=
  match args with
  | [] => []
  | s :: args =>
    (if plain_or plain s then [] else [(k, Some i)])
    ++ args_spoil k plain (S i) args
  end.

Definition apply_cont_spoils (plain : list variable) (ac : apply_cont)
    : spoilt :=
  args_spoil (ac_continuation ac) plain 0 (ac_args ac).

(* What is known in the handler of [k], given what its scope does to
   its parameters ([sp]): what is known where the [Let_cont] stands,
   [known] at first, then each parameter in turn, from position [i]
   on: added if the scope gives it only symbols, constants and
   variables known to hold one, forgotten if not. The order is that of
   FlambdaSemanticsSpec.bind_params, so that of two parameters of one
   name the later decides. *)
Fixpoint params_known (k : continuation) (sp : spoilt) (i : nat)
    (params known : list variable) : list variable :=
  match params with
  | [] => known
  | x :: params =>
    params_known k sp (S i) params
      (if spoils_at k i sp then without [x] known else x :: known)
  end.

(* [spoils plain e]: what [e] may do to the parameters of the
   continuations that it does not bind, [plain] being the variables
   known to hold a symbol or a constant where [e] stands.
   - An application of [k], and an arm of a [Switch] that applies it,
     spoils the positions at which its argument is not known.
   - A call spoils every parameter of the continuation that it returns
     to, and, if one of its extra arguments is not known, every
     parameter of its exception continuation: all of them and not the
     one at a position, as what the code called gives that continuation
     comes first and may be of any length, then the extra arguments of
     the calls on the way, the innermost first.
   - Under a [Let_cont] of [k], the handler is read with what its scope
     leaves known of its parameters, and what the scope says of [k] is
     forgotten: the handler is not in the scope of [k], so what it says
     of a continuation of that name is kept.
   - The parameters of a recursive handler are never known.
   Static bindings are not looked into: code does not see the
   continuations of the term around it. *)
Fixpoint spoils (plain : list variable) (e : expr) {struct e} : spoilt :=
  match e with
  | Let x _ _ body => spoils (without [x] plain) body
  | Let_static _ body => spoils plain body
  | Let_cont_non_recursive k (Handler params _ handler) body =>
    let sp := spoils plain body in
    spoils (params_known k sp 0 params plain) handler ++ forget_conts [k] sp
  | Let_cont_recursive handlers body =>
    forget_conts (map fst handlers)
      (flat_map (fun '(_, h) => handler_spoils plain h) handlers
       ++ spoils plain body)
  | Apply ap =>
    (ap_continuation ap, None)
    :: (if forallb (plain_or plain) (exn_extra_simples ap)
        then []
        else [(ap_exn_continuation ap, None)])
  | Apply_cont ac => apply_cont_spoils plain ac
  | Switch _ arms =>
    flat_map (fun arm => apply_cont_spoils plain (snd arm)) arms
  | Invalid => []
  end
with handler_spoils (plain : list variable) (h : continuation_handler)
    {struct h} : spoilt :=
  match h with
  | Handler params _ handler => spoils (without params plain) handler
  end.

Definition apply_cont_exits_plain_knowing (scope : list continuation)
    (plain : list variable) (ac : apply_cont) : bool :=
  among (ac_continuation ac) scope || forallb (plain_or plain) (ac_args ac).

(* [exits_plain] with [plain]: the expression gives variables that are
   not known to hold a symbol or a constant only to continuations of
   [scope], and every call in it returns to one. *)
Fixpoint exits_plain_knowing (scope : list continuation)
    (plain : list variable) (e : expr) {struct e} : bool :=
  match e with
  | Let x _ _ body => exits_plain_knowing scope (without [x] plain) body
  | Let_static bindings body =>
    forallb binding_exits_plain_knowing bindings
    && exits_plain_knowing scope plain body
  | Let_cont_non_recursive k (Handler params _ handler) body =>
    exits_plain_knowing scope
      (params_known k (spoils plain body) 0 params plain) handler
    && exits_plain_knowing (k :: scope) plain body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(_, h) =>
         handler_exits_plain_knowing (map fst handlers ++ scope) plain h)
      handlers
    && exits_plain_knowing (map fst handlers ++ scope) plain body
  | Apply ap =>
    among (ap_continuation ap) scope
    && (among (ap_exn_continuation ap) scope
        || forallb (plain_or plain) (exn_extra_simples ap))
  | Apply_cont ac => apply_cont_exits_plain_knowing scope plain ac
  | Switch _ arms =>
    forallb
      (fun arm => apply_cont_exits_plain_knowing scope plain (snd arm))
      arms
  | Invalid => true
  end
with handler_exits_plain_knowing (scope : list continuation)
    (plain : list variable) (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler params _ handler =>
    exits_plain_knowing scope (without params plain) handler
  end
with binding_exits_plain_knowing (binding : static_binding) {struct binding}
    : bool :=
  match binding with
  | Bind_code _ c => code_exits_plain_knowing c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_exits_plain_knowing (c : code) {struct c} : bool :=
  match c with
  | Code _ _ ret _ _ body => exits_plain_knowing [ret] [] body
  end.

(* [exits_plain_sound_statement], word for word but for the test: a
   unit that passes the test, run on its own, exits with symbols and
   constants only. *)
Definition exits_plain_knowing_sound_statement : Prop :=
  ∀ (e : expr) (n : nat) (k : continuation) (vs : list value) (hp : heap),
    exits_plain_knowing [] [] e = true ->
    run n (State e [] [] empty_static_env empty_heap) = Exit k vs hp ->
    ∀ v : value, v ∈ vs -> plain_value v.

(* The same from a static environment, with what it is assumed of
   written out: every code that the environment binds passes the test.
   The empty one and that of the imports
   (FlambdaImportsSpec.imports_statics) bind no code, so the premise
   holds of them with nothing to check; a unit run after a group of
   static bindings that binds code (FlambdaImportsSpec
   .prefix_refines_statement) needs it of that code. *)
Definition exits_plain_knowing_sound_from_statement : Prop :=
  ∀ (statics : static_env) (e : expr) (n : nat) (k : continuation)
    (vs : list value) (hp : heap),
    (∀ (id : code_id) (c : code),
       lookup id (static_code statics) = Some c ->
       code_exits_plain_knowing c = true) ->
    exits_plain_knowing [] [] e = true ->
    run n (State e [] [] statics empty_heap) = Exit k vs hp ->
    ∀ v : value, v ∈ vs -> plain_value v.

(* Nothing that [exits_plain] accepts is refused, whatever is known. *)
Definition exits_plain_knowing_accepts_statement : Prop :=
  ∀ (scope : list continuation) (plain : list variable) (e : expr),
    exits_plain scope e = true -> exits_plain_knowing scope plain e = true.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The unit with the [Let]s of the guess lifted and, in the code that
   the guess names, [my_closure] bound to the symbol given: if the unit
   exits with symbols and constants only, the result is a well-formed
   unit, and it makes no closure of that code but the symbol. *)
Definition lift_unit (L : lifting) (e : expr) : option expr :=
  let lifted :=
    bind_my_closures (lifted_my_closures L)
      (lift_expr (lifted_lets L) root e)
  in
  if
    exits_plain_knowing [] [] e
    && well_formed_unit (lifting_unit L) lifted
    && only_closures_of (lifted_my_closures L) lifted
  then Some lifted
  else None.

(* ======================================================================== *)
(* A value, and the same value with symbols for some of its parts           *)
(* ======================================================================== *)

(* [value_lifted statics' v v']: [v'] is [v], but that where [v] has a
   closure, a block, a boxed number or an immutable array, here or
   inside a closure, a block or an array, [v'] may have a symbol that
   [statics'] binds to it. For a closure of a set with named function
   slots, [statics'] binds the symbols of the whole set, each to the
   closure at its slot ([static_set_bound]), and the set has a symbol
   for each function slot. A constant is lifted to itself only, and so
   is a symbol. A partial application is lifted to a partial
   application, of the value it calls lifted, to its arguments lifted,
   at the same arity: none is made a symbol. An immutable array that a
   primitive made is lifted to an array of its elements lifted
   ([Lifted_array]), or to a symbol that [statics'] binds to a static
   immutable array of its elements lifted ([Lifted_to_array]). A
   location of the heap is lifted to itself only ([Lifted_location]):
   the relation is the identity on locations, and says nothing of what
   the heap holds at one. The rule is used under the others: a block, an
   array, a closure or a partial application that holds a location, in a
   field, in an element, in a value slot, or in what it calls or an
   argument, is lifted to one that holds the same location there; and a
   block, an array or a closure that holds one may be lifted to a symbol
   bound to a block, an array or a closure that holds the same, as a
   partial application may not. It is there so that every value is
   lifted to itself.

   A term run on its own, from no variable, no continuation, the empty
   static environment and the empty heap, with the steps that the
   machine has, reaches no value that holds a location, at any depth:
   its first state holds none, the heap of a state binds every
   location of the state
   (FlambdaLocationsMachineSpec.run_keeps_locations_bound_statement),
   and no step changes the heap. So the rule relates no pair that such
   a run reaches. A run from an environment that holds a location may
   reach one.

   The relation speaks of the second static environment only. A symbol
   is related to itself whatever it is bound to, in either run
   ([Lifted_symbol]): by itself that says nothing of what the two runs
   read through it. It is one part of the simulation of
   FlambdaLiftProofs.v, which has beside it that every symbol the first
   run binds, the second binds to the same thing, lifted. A symbol that
   only the second run binds, one of those given to a lifted [Let], may
   be mentioned by the first term; nothing can be read through it
   there, as every operation on a symbol bound to nothing is undefined
   (FlambdaPrimitiveSpec.eval_prim_symbols_bound_statement, decision
   D8), so the first run is undefined where the two would differ. *)
Definition slot_lifted (R : value -> value -> Prop)
    (slot slot' : value_slot * value) : Prop :=
  fst slot = fst slot' ∧ R (snd slot) (snd slot').

(* (The warning silenced is about induction schemes through the lists.) *)
#[local] Set Warnings "-register-all".
Inductive value_lifted (statics' : static_env) : value -> value -> Prop :=
  | Lifted_const (c : const) :
    value_lifted statics' (Value_const c) (Value_const c)
  | Lifted_symbol (s : symbol) :
    value_lifted statics' (Value_symbol s) (Value_symbol s)
  | Lifted_boxed_number (c : const) :
    value_lifted statics' (Value_boxed_number c) (Value_boxed_number c)
  | Lifted_block (tag : Z) (fields fields' : list value) :
    Forall2 (value_lifted statics') fields fields' ->
    value_lifted statics' (Value_block tag fields) (Value_block tag fields')
  | Lifted_array (elements elements' : list value) :
    Forall2 (value_lifted statics') elements elements' ->
    value_lifted statics' (Value_immutable_array elements)
      (Value_immutable_array elements')
  | Lifted_closure (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots' ->
    value_lifted statics' (Value_closure code value_slots)
      (Value_closure code value_slots')
  | Lifted_closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) :
    Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots' ->
    value_lifted statics'
      (Value_closure_of_set slot function_decls value_slots)
      (Value_closure_of_set slot function_decls value_slots')
  | Lifted_partial_application (callee callee' : value)
      (args args' : list value) (args_arity : arity) :
    value_lifted statics' callee callee' ->
    Forall2 (value_lifted statics') args args' ->
    value_lifted statics'
      (Value_partial_application callee args args_arity)
      (Value_partial_application callee' args' args_arity)
  | Lifted_to_boxed_number (c : const) (s : symbol) :
    lookup s (static_values statics')
    = Some (Static_boxed_number (Value_const c)) ->
    value_lifted statics' (Value_boxed_number c) (Value_symbol s)
  | Lifted_to_block (tag : Z) (fields fields' : list value) (s : symbol) :
    lookup s (static_values statics') = Some (Static_block tag fields') ->
    Forall2 (value_lifted statics') fields fields' ->
    value_lifted statics' (Value_block tag fields) (Value_symbol s)
  | Lifted_to_array (elements elements' : list value) (s : symbol) :
    lookup s (static_values statics')
    = Some (Static_immutable_array elements') ->
    Forall2 (value_lifted statics') elements elements' ->
    value_lifted statics' (Value_immutable_array elements) (Value_symbol s)
  | Lifted_to_closure (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots value_slots' : list (value_slot * value)) (s : symbol)
      (closure_symbols : list (function_slot * symbol)) :
    lookup slot closure_symbols = Some s ->
    map fst closure_symbols = map fst function_decls ->
    static_set_bound statics' closure_symbols function_decls value_slots' ->
    Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots' ->
    value_lifted statics'
      (Value_closure_of_set slot function_decls value_slots)
      (Value_symbol s)
  | Lifted_to_unnamed_closure (code : code_id)
      (value_slots value_slots' : list (value_slot * value)) (s : symbol) :
    lookup s (static_values statics')
    = Some (Static_closure code value_slots') ->
    Forall2 (slot_lifted (value_lifted statics')) value_slots value_slots' ->
    value_lifted statics' (Value_closure code value_slots) (Value_symbol s)
  | Lifted_location (l : location) :
    value_lifted statics' (Value_location l) (Value_location l).

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The term lifted, with [my_closure] bound to its symbol in the code
   that [known] names, run on its own, ends as the term does, up to
   [value_lifted] at the exit: an exit of the term is one of the term
   lifted, through the same continuation, with values lifted under the
   static environment that the second run has reached; and if the term
   lifted ends, so does the term, or it goes wrong. (By the first part
   the term then ends alike, as a run has one behaviour.) *)
Definition lift_simulates_statement : Prop :=
  ∀ (u : unit_interface) (lets : list (path * lift_to))
    (known : list (code_id * symbol)) (e : expr),
    let lifted := bind_my_closures known (lift_expr lets root e) in
    well_formed_unit u lifted = true ->
    only_closures_of known lifted = true ->
    (∀ (k : continuation) (vs : list value) (hp : heap),
      behaves (State e [] [] empty_static_env empty_heap) (Exits k vs hp) ->
      ∃ (n : nat) (last : state) (vs' : list value),
        run n (State lifted [] [] empty_static_env empty_heap) = Step last
        ∧ step last = Exit k vs' hp
        ∧ Forall2 (value_lifted (state_statics last)) vs vs')
    ∧ (∀ b : behaviour,
        b <> Diverges ->
        behaves (State lifted [] [] empty_static_env empty_heap) b ->
        behaves (State e [] [] empty_static_env empty_heap) Goes_wrong
        ∨ ∃ (k : continuation) (vs : list value) (hp : heap),
            behaves (State e [] [] empty_static_env empty_heap)
              (Exits k vs hp)).

(* What [lift_unit] gives refines the unit, run on its own. *)
Definition lift_unit_program_refines_statement : Prop :=
  ∀ (L : lifting) (e e' : expr),
    lift_unit L e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports (FlambdaImportsSpec.imports_statics): if
   the unit lifted binds no name that the imports bind
   (FlambdaImportsSpec.static_names_free_in), it refines the unit. A
   symbol that the guess gives to a [Let] and that is an import would
   be bound twice. *)
Definition lift_unit_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (L : lifting) (e e' : expr),
    lift_unit L e = Some e' ->
    static_names_free_in (imports_statics imports) e' = true ->
    unit_refines imports e e'.

(* Lifting commutes with renaming: a [Let] is named by where it is. The
   renaming is of variables and continuations, in code too: symbols,
   code ids and slots are not renamed. *)
Definition lift_expr_equivariant_statement : Prop :=
  ∀ (r : renaming) (lets : list (path * lift_to)) (at_path : path)
    (e : expr),
    rename_expr r (lift_expr lets at_path e)
    = lift_expr lets at_path (rename_expr r e).

(* So does the check of the exits, the scope being renamed with the
   term. *)
Definition exits_plain_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope : list continuation) (e : expr),
    exits_plain (map (rename_continuation r) scope) (rename_expr r e)
    = exits_plain scope e.

(* And [exits_plain_knowing]: the variables known are renamed with the
   term. A renaming is a permutation of the names of each sort, so no
   two variables become one; the statement needs that. *)
Definition exits_plain_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (scope : list continuation) (plain : list variable)
    (e : expr),
    exits_plain_knowing (map (rename_continuation r) scope)
      (map (rename_variable r) plain) (rename_expr r e)
    = exits_plain_knowing scope plain e.
