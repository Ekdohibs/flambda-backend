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

(* Following aliases through a term: a normalisation that is an
   equivalence.

   This file is to be reviewed: [follow_aliases] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   [follow_aliases] goes through a term with a typing environment
   (FlambdaTypesSpec.v) holding, for the variables bound by a [Let] to a
   [Simple], that [Simple]. Every use of such a variable is replaced by
   what it is bound to ([canonical_simple]). It holds too, for the
   variables bound to a closure, the code and the value slots of the
   closure: a call of such a variable becomes a direct call of the code,
   and the projection of a value slot from it becomes what the slot
   holds. This is what the simplifier
   does with its aliases: there is no substitution, and the [Let] stays,
   to be removed if it is then unused
   (FlambdaNormaliseSpec.remove_unused_lets).

   It holds too, for a variable bound to a store ([Block_set],
   [Array_set]), the constant unit, and the uses of the variable that
   the pass visits are replaced by that constant, under the rules of
   scope and shadowing below. Simplify gives the result of a store the
   type of the constant (simplify/simplify_primitive_result.ml,
   [create_unit], for a [Block_set];
   simplify/simplify_ternary_primitive.ml, [simplify_array_set]), so
   the term after has the constant where the term before has the
   variable. The pass keeps the [Let] of the store. That its result is
   then unused does not let the rewrites that remove an unused [Let]
   remove it: their tests refuse a primitive of the heap
   (FlambdaNormaliseSpec.prim_may_fold, [named_defined]).

   Going under a binder, what is known of its name, and of anything said
   to be an alias of it, is dropped ([drop_var]): the name stands for
   something else from there on. So nothing is asked of the names of the
   term. Handlers of non-recursive continuations are gone into, with what
   is known where they are defined; those of recursive ones and code are
   left as they are.

   The result is equivalent to the term, not only a refinement of it
   ([follow_aliases_equivalent_statement]), so it may be applied on both
   sides of the comparison.

   The proofs are in FlambdaAliasesProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNormaliseSpec.
From OxCamlFlambda2Types Require Import FlambdaTypesSpec.
Import ListNotations.

(* ======================================================================== *)
(* Forgetting a name                                                        *)
(* ======================================================================== *)

Definition var_in_type (x : variable) (t : flambda_type) : bool :=
  match t with
  | Equals s => var_in_simple x s
  | Closure_of _ slots => var_in_value_slots x slots
  | Unknown | Bottom => false
  end.

(* What is known that does not mention [x]. *)
Definition drop_var (x : variable) (tenv : typing_env) : typing_env :=
  filter (fun '(y, t) => negb (Pos.eqb x y) && negb (var_in_type x t)) tenv.

Definition drop_vars (xs : list variable) (tenv : typing_env) : typing_env :=
  fold_right drop_var tenv xs.

(* ======================================================================== *)
(* Following aliases                                                        *)
(* ======================================================================== *)

Definition canonical_value_slots (tenv : typing_env)
    (slots : list (value_slot * simple)) : list (value_slot * simple) :=
  map (fun slot => (fst slot, canonical_simple tenv (snd slot))) slots.

(* The closure that a variable is known to be. *)
Fixpoint find_closure (tenv : typing_env) (x : variable)
    : option (code_id * list (value_slot * simple)) :=
  match tenv with
  | [] => None
  | (y, Closure_of code slots) :: tenv =>
    if Pos.eqb x y then Some (code, slots) else find_closure tenv x
  | _ :: tenv => find_closure tenv x
  end.

Definition closure_of_simple (tenv : typing_env) (s : simple)
    : option (code_id * list (value_slot * simple)) :=
  match s with
  | Var x => find_closure tenv x
  | Symbol _ | Const _ => None
  end.

(* The projection of a value slot from a known closure is what the slot
   holds. *)
Definition canonical_project (tenv : typing_env) (slot : value_slot)
    (closure : simple) : named :=
  match closure_of_simple tenv closure with
  | Some (_, slots) =>
    match lookup slot slots with
    | Some contents => Simple contents
    | None => Project_value_slot slot closure
    end
  | None => Project_value_slot slot closure
  end.

Definition canonical_named (tenv : typing_env) (d : named) : named :=
  match d with
  | Simple s => Simple (canonical_simple tenv s)
  | Prim p args => Prim p (map (canonical_simple tenv) args)
  | Set_of_closures code slots =>
    Set_of_closures code (canonical_value_slots tenv slots)
  | Closure_of_set slot decls slots =>
    Closure_of_set slot decls (canonical_value_slots tenv slots)
  | Project_value_slot slot closure =>
    canonical_project tenv slot (canonical_simple tenv closure)
  | Project_value_slot_from project_from slot closure =>
    Project_value_slot_from project_from slot (canonical_simple tenv closure)
  | Project_function_slot move_from move_to closure =>
    Project_function_slot move_from move_to (canonical_simple tenv closure)
  end.

Definition canonical_static_const (tenv : typing_env) (const : static_const)
    : static_const :=
  match const with
  | Static_const_set_of_closures code slots =>
    Static_const_set_of_closures code (canonical_value_slots tenv slots)
  | Static_const_closure_of_set slot symbols decls slots =>
    Static_const_closure_of_set slot symbols decls
      (canonical_value_slots tenv slots)
  | Static_const_block tag fields =>
    Static_const_block tag (map (canonical_simple tenv) fields)
  | Static_const_boxed_number contents =>
    Static_const_boxed_number (canonical_simple tenv contents)
  | Static_const_immutable_string bytes => Static_const_immutable_string bytes
  | Static_const_immutable_value_array elements =>
    Static_const_immutable_value_array (map (canonical_simple tenv) elements)
  end.

Definition canonical_static_binding (tenv : typing_env)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id c
  | Bind_deleted_code id => Bind_deleted_code id
  | Bind_symbol s const => Bind_symbol s (canonical_static_const tenv const)
  end.

Definition canonical_apply_cont (tenv : typing_env) (ac : apply_cont)
    : apply_cont :=
  Apply_cont_expr (ac_continuation ac)
    (map (canonical_simple tenv) (ac_args ac)) (ac_trap_action ac).

(* A call of a known closure is a direct call of its code. *)
Definition canonical_call_kind (tenv : typing_env) (callee : simple)
    (kind : call_kind) : call_kind :=
  match kind, closure_of_simple tenv callee with
  | Indirect, Some (code, _) => Direct code
  | Indirect, None | Direct _, _ => kind
  end.

Definition canonical_apply (tenv : typing_env) (ap : apply_expr) : apply_expr :=
  let callee := canonical_simple tenv (ap_callee ap) in
  Apply_expr callee (canonical_call_kind tenv callee (ap_call_kind ap))
    (map (canonical_simple tenv) (ap_args ap)) (ap_continuation ap)
    (ap_exn_continuation ap)
    (map (fun arg => (canonical_simple tenv (fst arg), snd arg))
       (ap_exn_extra_args ap))
    (ap_args_arity ap) (ap_return_arity ap).

(* What is known in the body of a [Let] of [x] to [d]: nothing of the old
   [x], and that the new one is [d] if that is a [Simple] or a closure not
   mentioning it. If [d] is a store, the new [x] is the constant unit:
   a store has a value only where it returns that constant
   (FlambdaHeapSpec.exec_heap_primitive, whose two cases of a
   store end with [unit_value]). *)
Definition typing_env_under_let (tenv : typing_env) (x : variable) (d : named)
    : typing_env :=
  match d with
  | Simple s =>
    if var_in_simple x s
    then drop_var x tenv
    else (x, Equals s) :: drop_var x tenv
  | Set_of_closures code slots =>
    if var_in_value_slots x slots
    then drop_var x tenv
    else (x, Closure_of code slots) :: drop_var x tenv
  | Prim (Heap (Block_set _ _ _ | Array_set _ _)) _ =>
    (x, Equals (Const (Tagged_immediate 0))) :: drop_var x tenv
  | Prim _ _ | Closure_of_set _ _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ | Project_function_slot _ _ _ =>
    drop_var x tenv
  end.

Fixpoint follow_aliases (tenv : typing_env) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let d' := canonical_named tenv d in
    Let x kind d' (follow_aliases (typing_env_under_let tenv x d') body)
  | Let_static bindings body =>
    Let_static (map (canonical_static_binding tenv) bindings)
      (follow_aliases tenv body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (follow_aliases_in_handler tenv h)
      (follow_aliases tenv body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (follow_aliases tenv body)
  | Apply ap => Apply (canonical_apply tenv ap)
  | Apply_cont ac => Apply_cont (canonical_apply_cont tenv ac)
  | Switch scrutinee arms =>
    Switch (canonical_simple tenv scrutinee)
      (map (fun arm => (fst arm, canonical_apply_cont tenv (snd arm))) arms)
  | Invalid => Invalid
  end
with follow_aliases_in_handler (tenv : typing_env) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (follow_aliases (drop_vars params tenv) handler)
  end.

Definition follow_aliases_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (follow_aliases [] e).
