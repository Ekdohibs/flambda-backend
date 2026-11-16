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

(* In the code of a closure that is a symbol, [my_closure] is the symbol.

   This file is to be reviewed: [bind_my_closures], against what Simplify
   does (below); [only_closure_of], the check; and the statements.

   What Simplify does. It simplifies the code of a function where it
   simplifies the set of closures that the function is of, and gives
   [my_closure], in that code, the type of an alias of the name that the
   closure is bound to (simplify/simplify_set_of_closures.ml,
   [dacc_inside_function], the case [Some function_slot]:
   [T.alias_type_of K.value (Simple.name name)]). Where the set is
   static, or has just been lifted (FlambdaLiftSpec.v), that name is the
   symbol of the closure, and every use of [my_closure] in the code
   becomes the symbol: a recursive function calls itself through its
   symbol. That is right only where the code so simplified runs for
   that closure and for no other. It is not inferred here from the new
   code ids that Simplify gives: there is one new id for each old one
   across the sets simplified together
   (simplify/simplify_set_of_closures_context.ml,
   [compute_old_to_new_code_ids_all_sets]), so sets that had the same
   code have the same new code. It is a condition, checked
   ([only_closure_of], below).

   What is here. [bind_my_closures known e] puts, at the head of the
   code of each code id that [known] gives a symbol,

     Let my_closure = Symbol s in body

   so that [my_closure] is from there on an alias of the symbol;
   following the alias and removing the [Let] are other passes
   (FlambdaAliasesSpec.follow_aliases,
   FlambdaNormaliseSpec.remove_unused_lets, taken into code by
   FlambdaNormaliseSpec.rewrite_code). Code one of whose parameters is
   named as its [my_closure] is left alone: the parameter is what the
   name means there.

   It is right only if the closure called is the symbol whenever that
   code runs. Nothing in the model makes that so: a code id may be the
   code of several closures (FlambdaWellFormedSpec.v: two sets of a unit
   may have the same code), and under other environments the code may be
   called through a closure that was there already. So this is no
   rewrite under all environments. For a unit run on its own it is
   enough that the unit makes one closure only of the code, the static
   one bound to the symbol, which is what [only_closure_of id s e]
   checks: no [Let] of [e] makes a closure of a set that has the code
   [id], in code and in handlers too, and a static binding of a closure
   whose code is [id] binds [s]. A unit that passes calls the code [id]
   through [s] only ([only_closure_sound_statement]).

   Which code ids, and which symbols, is a guess of the caller's;
   nothing is believed of it. The rewrite is used by
   FlambdaLiftSpec.lift_unit, with the lifting of [Let]s, and it is
   there that it is said to refine the unit
   (FlambdaLiftSpec.lift_unit_program_refines_statement): after the
   [Let] of a closure is lifted, its code is that of one closure, the
   symbol.

   The proofs are in FlambdaMyClosureProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* Every binding of code, wherever it is, has [my_closure] bound to the
   symbol that [known] gives its code id, if it gives one. *)
Fixpoint bind_my_closures (known : list (code_id * symbol)) (e : expr)
    : expr :=
  match e with
  | Let x kind d body => Let x kind d (bind_my_closures known body)
  | Let_static bindings body =>
    Let_static (map (bind_my_closures_in_binding known) bindings)
      (bind_my_closures known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (bind_my_closures_in_handler known h)
      (bind_my_closures known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, bind_my_closures_in_handler known h)) handlers)
      (bind_my_closures known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with bind_my_closures_in_handler (known : list (code_id * symbol))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (bind_my_closures known handler)
  end
with bind_my_closures_in_binding (known : list (code_id * symbol))
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (bind_my_closure known id c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with bind_my_closure (known : list (code_id * symbol)) (id : code_id)
    (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    let body := bind_my_closures known body in
    match lookup id known with
    | Some s =>
      if among my_closure params
      then Code params my_closure ret exn metadata body
      else
        Code params my_closure ret exn metadata
          (Let my_closure Value (Simple (Symbol s)) body)
    | None => Code params my_closure ret exn metadata body
    end
  end.

(* ======================================================================== *)
(* The check                                                                *)
(* ======================================================================== *)

(* The defining expression makes no closure of a set that has the code
   [id]. *)
Definition named_makes_no_closure_of (id : code_id) (d : named) : bool :=
  match d with
  | Set_of_closures code _ => negb (Pos.eqb code id)
  | Closure_of_set _ function_decls _ =>
    negb (among id (map snd function_decls))
  | Simple _ | Prim _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    true
  end.

(* If the constant is a closure whose code is [id], the symbol bound to
   it is [s]. The code of a closure of a set is that of its slot. *)
Definition static_closure_of_is (id : code_id) (s sym : symbol)
    (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures code _ =>
    negb (Pos.eqb code id) || Pos.eqb sym s
  | Static_const_closure_of_set slot _ function_decls _ =>
    match lookup slot function_decls with
    | Some code => negb (Pos.eqb code id) || Pos.eqb sym s
    | None => true
    end
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    true
  end.

(* The only closure of the code [id] that the expression makes is the
   static one bound to [s]. Handlers and code are looked into. *)
Fixpoint only_closure_of (id : code_id) (s : symbol) (e : expr) : bool :=
  match e with
  | Let _ _ d body =>
    named_makes_no_closure_of id d && only_closure_of id s body
  | Let_static bindings body =>
    forallb (binding_only_closure_of id s) bindings
    && only_closure_of id s body
  | Let_cont_non_recursive _ h body =>
    handler_only_closure_of id s h && only_closure_of id s body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_only_closure_of id s h) handlers
    && only_closure_of id s body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_only_closure_of (id : code_id) (s : symbol)
    (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => only_closure_of id s handler
  end
with binding_only_closure_of (id : code_id) (s : symbol)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ c => code_only_closure_of id s c
  | Bind_deleted_code _ => true
  | Bind_symbol sym const => static_closure_of_is id s sym const
  end
with code_only_closure_of (id : code_id) (s : symbol) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => only_closure_of id s body
  end.

(* The check for every code id that is given a symbol. *)
Definition only_closures_of (known : list (code_id * symbol)) (e : expr)
    : bool :=
  forallb (fun '(id, s) => only_closure_of id s e) known.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A unit that passes the check, run on its own, calls the code [id]
   through the symbol [s] only: in every state of its run that is at a
   call, if the callee is a closure of the code [id], it is [s]. *)
Definition only_closure_sound_statement : Prop :=
  ∀ (id : code_id) (s : symbol) (e : expr) (n : nat) (st : state)
    (ap : apply_expr) (callee : value)
    (value_slots : list (value_slot * value)),
    only_closure_of id s e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step st ->
    state_expr st = Apply ap ->
    eval_simple (state_vars st) (ap_callee ap) = Some callee ->
    closure_of (state_statics st) callee = Some (id, value_slots) ->
    callee = Value_symbol s.

(* The rewrite makes no closure: a term passes the check before it if
   and only if it does after. *)
Definition bind_my_closures_only_closure_statement : Prop :=
  ∀ (known : list (code_id * symbol)) (id : code_id) (s : symbol)
    (e : expr),
    only_closure_of id s (bind_my_closures known e) = only_closure_of id s e.

(* The rewrite and the check commute with renaming, which goes into
   code and renames neither code ids nor symbols. *)
Definition bind_my_closures_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : list (code_id * symbol)) (e : expr),
    rename_expr r (bind_my_closures known e)
    = bind_my_closures known (rename_expr r e).

Definition only_closure_of_equivariant_statement : Prop :=
  ∀ (r : renaming) (id : code_id) (s : symbol) (e : expr),
    only_closure_of id s (rename_expr r e) = only_closure_of id s e.
