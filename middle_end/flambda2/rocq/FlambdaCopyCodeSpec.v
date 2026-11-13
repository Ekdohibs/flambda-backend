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

(* A copy of code for static sets of closures: where Simplify has given
   a static set of closures new code, the term before is given a copy
   of the old code under the new code ids, for that set. A
   normalisation that refines.

   This file is to be reviewed: [copy_static_code] and what it is made
   of, against what Simplify does (below); the check; and the
   statements, the last two of which are the rewrite computed on a
   small unit whose code is copied twice.

   What Simplify does. It gives new code to every set of closures that
   it simplifies, wherever the set stands: a new code id for each
   function of the set, whose code is the old code simplified where
   the set is, with what is known there of the value slots
   (simplify/simplify_set_of_closures_context.ml,
   [compute_old_to_new_code_ids_all_sets], by [Code_id.rename];
   simplify/simplify_set_of_closures.ml, [simplify_function]). The new
   code says that it is a newer version of the old
   ([Code.newer_version_of]). A set that the unit makes once has one
   newer version; the old code is then deleted, and the conversion of
   the test tool gives the new code the number of the old. A set that
   Simplify meets twice has two: once in the body of a function that
   stays, and once in a copy of that body that it inlines. In the
   copy the value slots may hold constants or symbols, and the code
   made there reads them, where the code made for the function that
   stays projects the slots. So the term after has, for one code id
   of the term before, the code of the function that stays and other
   code, under a new id, for the closure of the copy.

   What is here. [copy_static_code ids symbols e] gives the code of
   [e] a copy under a new id, for each pair of [ids], and gives the
   copies to the static sets of closures that [e] binds to [symbols]:
   - a binding of code whose id [ids] pairs with a new one is followed
     by a binding of the same code under the new id;
   - in the static sets of closures bound to the symbols of [symbols],
     each code id that [ids] pairs with a new one is replaced by it;
   - a direct call whose callee is one of those symbols names the
     copy, if [ids] gives one to the code that it names;
   - a direct call that names copied code and whose callee is no
     symbol is made indirect, everywhere, in code and in the copies
     too.
   The last is what lets a copy be the same text as the code that it
   copies. The body of a function may call itself, or a function of
   its set, through [my_closure], by a direct call that names the old
   code; in a closure of the copy that call would name code that the
   closure is not of. The rewrites that make a call direct where the
   callee is known make these direct again, with the right code:
   FlambdaMyClosureSpec.bind_my_closures binds [my_closure] to the
   symbol where the code has one closure, and
   FlambdaKnownClosuresSpec.use_known_closures reads the code of a
   symbol.

   It is right whatever [ids] and [symbols] are, if the unit binds
   none of the new code ids and they are distinct, and the old ones
   are distinct ([copy_ok]): which code and which symbols is a guess
   of the caller's, read off the term after, and nothing is believed
   of it. Only static sets are given the copies. A closure that a
   [Let] makes is a value that holds its code ids
   (FlambdaValueSpec.Value_closure_of_set), and the two runs would
   differ in it; a symbol is the same value in both, as a value does
   not hold what a symbol is bound to. So no value changes, and the
   rewrite refines the unit with the same exits
   ([copy_static_code_program_refines_statement], and from the
   imports and followed by something). It is the argument of
   FlambdaUnusedSlotsSpec.drop_unused_slots, which removes value slots
   from static sets only for the same reason. A set made by a [Let]
   that Simplify gives new code is left as it is: this rewrite does
   not change the code ids that such a closure holds, and does nothing
   for a unit whose two terms differ in them. The comparisons that
   come before it and the other normalisations may still validate
   such a unit.

   The proofs are in FlambdaCopyCodeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaUnusedStaticsSpec
  FlambdaImportsSpec FlambdaObserversSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The copy of a code id, where [ids] gives it one; the id itself where
   it gives none. *)
Definition copy_of (ids : list (code_id * code_id)) (id : code_id)
    : code_id :=
  match lookup id ids with
  | Some id' => id'
  | None => id
  end.

(* The functions of a set, each with the copy of its code. *)
Definition copy_decls (ids : list (code_id * code_id))
    (function_decls : list (function_slot * code_id))
    : list (function_slot * code_id) :=
  map (fun decl => (fst decl, copy_of ids (snd decl))) function_decls.

(* A static set of closures with the copies of its code. *)
Definition copy_static_const (ids : list (code_id * code_id))
    (const : static_const) : static_const :=
  match const with
  | Static_const_set_of_closures code value_slots =>
    Static_const_set_of_closures (copy_of ids code) value_slots
  | Static_const_closure_of_set slot closure_symbols function_decls
      value_slots =>
    Static_const_closure_of_set slot closure_symbols
      (copy_decls ids function_decls) value_slots
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    const
  end.

(* The call with another call kind. *)
Definition with_call_kind (kind : call_kind) (ap : apply_expr)
    : apply_expr :=
  Apply_expr (ap_callee ap) kind (ap_args ap) (ap_continuation ap)
    (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
    (ap_return_arity ap).

(* A direct call whose callee is one of the symbols names the copy of
   the code that it names; one that names copied code and whose callee
   is no symbol is made indirect. *)
Definition copy_call (ids : list (code_id * code_id))
    (symbols : list symbol) (ap : apply_expr) : apply_expr :=
  match ap_call_kind ap with
  | Indirect => ap
  | Direct id =>
    match ap_callee ap with
    | Symbol s =>
      if among s symbols
      then with_call_kind (Direct (copy_of ids id)) ap
      else ap
    | Var _ | Const _ =>
      match lookup id ids with
      | Some _ => with_call_kind Indirect ap
      | None => ap
      end
    end
  end.

(* The rewrite, of an expression, a handler, a static binding (which
   becomes one binding or two) and code. *)
Fixpoint copy_expr (ids : list (code_id * code_id)) (symbols : list symbol)
    (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (copy_expr ids symbols body)
  | Let_static bindings body =>
    Let_static (flat_map (copy_binding ids symbols) bindings)
      (copy_expr ids symbols body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (copy_handler ids symbols h)
      (copy_expr ids symbols body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, copy_handler ids symbols h)) handlers)
      (copy_expr ids symbols body)
  | Apply ap => Apply (copy_call ids symbols ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with copy_handler (ids : list (code_id * code_id)) (symbols : list symbol)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (copy_expr ids symbols handler)
  end
with copy_binding (ids : list (code_id * code_id)) (symbols : list symbol)
    (binding : static_binding) : list static_binding :=
  match binding with
  | Bind_code id c =>
    let c' := copy_code ids symbols c in
    Bind_code id c'
    :: match lookup id ids with
       | Some id' => [Bind_code id' c']
       | None => []
       end
  | Bind_deleted_code _ => [binding]
  | Bind_symbol s const =>
    [Bind_symbol s
       (if among s symbols then copy_static_const ids const else const)]
  end
with copy_code (ids : list (code_id * code_id)) (symbols : list symbol)
    (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (copy_expr ids symbols body)
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The expression binds no code id of [new], in handlers and in code
   too. A declaration of deleted code binds nothing
   (FlambdaSemanticsSpec.eval_static_bindings). *)
Fixpoint binds_no_code (new : list code_id) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => binds_no_code new body
  | Let_static bindings body =>
    forallb (binding_binds_no_code new) bindings && binds_no_code new body
  | Let_cont_non_recursive _ h body =>
    handler_binds_no_code new h && binds_no_code new body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_binds_no_code new h) handlers
    && binds_no_code new body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_binds_no_code (new : list code_id) (h : continuation_handler)
    : bool :=
  match h with
  | Handler _ _ handler => binds_no_code new handler
  end
with binding_binds_no_code (new : list code_id) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code id c => negb (among id new) && code_binds_no_code new c
  | Bind_deleted_code _ | Bind_symbol _ _ => true
  end
with code_binds_no_code (new : list code_id) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => binds_no_code new body
  end.

(* The old code ids are distinct, the new ones are distinct, and the
   unit binds none of the new. *)
Definition copy_ok (ids : list (code_id * code_id)) (e : expr) : bool :=
  distinct (map fst ids) && distinct (map snd ids)
  && binds_no_code (map snd ids) e.

(* The unit with copies of its code, by [ids], for the static sets of
   closures that it binds to [symbols], if the check passes. *)
Definition copy_static_code (ids : list (code_id * code_id))
    (symbols : list symbol) (e : expr) : option expr :=
  if copy_ok ids e then Some (copy_expr ids symbols e) else None.

(* Each of the copies given, in turn; one whose check fails is left
   out. *)
Fixpoint with_copies
    (copies : list (list (code_id * code_id) * list symbol)) (e : expr)
    : expr :=
  match copies with
  | [] => e
  | (ids, symbols) :: copies =>
    with_copies copies
      match copy_static_code ids symbols e with
      | Some e' => e'
      | None => e
      end
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What [copy_static_code] gives refines the unit, run on its own. *)
Definition copy_static_code_program_refines_statement : Prop :=
  ∀ (ids : list (code_id * code_id)) (symbols : list symbol) (e e' : expr),
    copy_static_code ids symbols e = Some e' -> program_refines e e'.

(* The same of a unit that imports symbols, run from the static
   environment of its imports. Nothing more is asked: the imports bind
   no code, and an import is no closure. *)
Definition copy_static_code_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (ids : list (code_id * code_id))
    (symbols : list symbol) (e e' : expr),
    copy_static_code ids symbols e = Some e' -> unit_refines imports e e'.

(* The same of a unit that something follows: the unit and the unit
   with the copies, run from one continuation environment of what may
   follow a unit (FlambdaObserversSpec.observers) and from the static
   environment of the imports. What follows is the same term in the two
   runs: an observer binds nothing static, and names no symbol and no
   code id, so the rewrite leaves it as it is. *)
Definition copy_static_code_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (slots : list value_slot)
    (ids : list (code_id * code_id)) (symbols : list symbol)
    (conts : cont_env) (e e' : expr),
    copy_static_code ids symbols e = Some e' ->
    observers slots conts ->
    refines (State e [] conts (imports_statics imports) empty_heap)
      (State e' [] conts (imports_statics imports) empty_heap).

(* With no copy asked for, the term is as it was. *)
Definition with_copies_none_statement : Prop :=
  ∀ e : expr, with_copies [] e = e.

(* The three, of each of the copies in turn. Nothing is asked: a copy
   whose check fails is left out. *)
Definition with_copies_program_refines_statement : Prop :=
  ∀ (copies : list (list (code_id * code_id) * list symbol)) (e : expr),
    program_refines e (with_copies copies e).

Definition with_copies_refines_imports_statement : Prop :=
  ∀ (imports : list (symbol * import_class))
    (copies : list (list (code_id * code_id) * list symbol)) (e : expr),
    unit_refines imports e (with_copies copies e).

Definition with_copies_refines_observed_statement : Prop :=
  ∀ (imports : list (symbol * import_class)) (slots : list value_slot)
    (copies : list (list (code_id * code_id) * list symbol))
    (conts : cont_env) (e : expr),
    observers slots conts ->
    refines (State e [] conts (imports_statics imports) empty_heap)
      (State (with_copies copies e) [] conts (imports_statics imports)
         empty_heap).

(* The rewrite commutes with renaming, which renames no code id and no
   symbol, and so does the check. *)
Definition copy_expr_equivariant_statement : Prop :=
  ∀ (r : renaming) (ids : list (code_id * code_id)) (symbols : list symbol)
    (e : expr),
    rename_expr r (copy_expr ids symbols e)
    = copy_expr ids symbols (rename_expr r e).

Definition binds_no_code_equivariant_statement : Prop :=
  ∀ (r : renaming) (new : list code_id) (e : expr),
    binds_no_code new (rename_expr r e) = binds_no_code new e.

Definition with_copies_equivariant_statement : Prop :=
  ∀ (r : renaming) (copies : list (list (code_id * code_id) * list symbol))
    (e : expr),
    rename_expr r (with_copies copies e)
    = with_copies copies (rename_expr r e).

(* ======================================================================== *)
(* One code copied twice                                                    *)
(* ======================================================================== *)

(* What the rewrite makes of a small unit, computed. One piece of
   code, which calls itself through [my_closure] by a direct call, is
   the code of two static closures, and the unit calls the second.
   Simplify may give each of the two closures new code of its own,
   where a function that makes the closure is inlined at two places.
   A copy gives a code id one new id; two copies of one code are two
   copies given in turn. *)
Section Examples.

Let x : variable := 1%positive.
Let my_closure : variable := 2%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let old : code_id := 1%positive.
Let first_new : code_id := 2%positive.
Let second_new : code_id := 3%positive.
Let first_closure : symbol := 1%positive.
Let second_closure : symbol := 2%positive.

Definition copied_arity : arity := [Singleton Value].

(* The code, with the kind given for the call of itself. *)
Definition copied_code (kind : call_kind) : code :=
  Code [x] my_closure ret exn
    (make_code_metadata copied_arity copied_arity)
    (Apply
       (Apply_expr (Var my_closure) kind [Var x] ret exn [] copied_arity
          copied_arity)).

(* The unit, with the code given for each of the two closures and for
   the call of the second, and the bindings of code given. *)
Definition copied_unit (code_bindings : list static_binding)
    (first second : code_id) : expr :=
  Let_static
    (code_bindings
     ++ [Bind_symbol first_closure (Static_const_set_of_closures first []);
         Bind_symbol second_closure (Static_const_set_of_closures second [])])
    (Apply
       (Apply_expr (Symbol second_closure) (Direct second)
          [Const (Tagged_immediate 0)] done exn [] copied_arity
          copied_arity)).

Definition copied_twice_before : expr :=
  copied_unit [Bind_code old (copied_code (Direct old))] old old.

(* Each closure has a copy of its own, and the call of the second
   names the second copy. The call of the code by itself is indirect
   in the code and in the two copies. *)
Definition copied_twice_after : expr :=
  copied_unit
    [Bind_code old (copied_code Indirect);
     Bind_code second_new (copied_code Indirect);
     Bind_code first_new (copied_code Indirect)]
    first_new second_new.

(* Two copies given in turn, each for one of the closures. *)
Definition with_copies_twice_statement : Prop :=
  with_copies
    [([(old, first_new)], [first_closure]);
     ([(old, second_new)], [second_closure])]
    copied_twice_before
  = copied_twice_after.

(* One copy that pairs the code with two new ids is not made: its
   check fails, and the unit is left as it is. *)
Definition with_copies_two_in_one_statement : Prop :=
  copy_static_code [(old, first_new); (old, second_new)]
    [first_closure; second_closure] copied_twice_before
  = None
  ∧ with_copies
      [([(old, first_new); (old, second_new)],
        [first_closure; second_closure])]
      copied_twice_before
    = copied_twice_before.

End Examples.
