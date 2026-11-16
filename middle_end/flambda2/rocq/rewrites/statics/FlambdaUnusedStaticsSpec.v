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

(* Removing static bindings that are not used.

   This file is to be reviewed: [expr_avoids] and [erase_expr], which are
   extracted and run by the validation of
   middle_end/flambda2/tests/semantics/; [envs_free_of]; and the
   statements.

   Simplify removes the code and the symbols that nothing refers to any
   more. Removing a static binding is not sound under every environment:
   a value already there may be a closure of the code, or the symbol, and
   then the term with the binding can call it or look into it and the term
   without cannot. It is sound under every environment in which the names
   removed do not occur ([envs_free_of]), which is what is stated: the
   names a term binds are fresh, in Flambda. A term run on its own, from
   empty environments, is a case
   ([drop_statics_program_refines_statement]).

   [drop_statics] takes the names to remove ([dropped]). It checks that
   nothing in the term refers to any of them, outside the bindings that go
   ([expr_avoids]), and removes those bindings everywhere, in handlers and
   in code too ([erase_expr]). Which names to remove is for the caller to
   find; a wrong guess is refused, not believed. A [Let_static] left with
   no binding stays, to be removed by
   FlambdaNormaliseSpec.drop_deleted_code.

   It is a refinement and not an equivalence: a binding removed may have
   been of a static constant with no value.

   The primitives introduce no symbols and no code ids: a value they
   return refers only to what their arguments refer to
   (FlambdaPrimitiveSpec.eval_prim_preserves_statement). Otherwise a
   primitive could return a closure of code that has been removed.

   The proofs are in FlambdaUnusedStaticsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaHeapSpec FlambdaSemanticsSpec FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* The names removed, and what refers to none of them                       *)
(* ======================================================================== *)

Record dropped : Type := Dropped {
  dropped_symbols : list symbol;
  dropped_code : list code_id
}.

Definition symbol_kept (D : dropped) (s : symbol) : bool :=
  negb (among s (dropped_symbols D)).

Definition code_kept (D : dropped) (id : code_id) : bool :=
  negb (among id (dropped_code D)).

(* The code of every function of a set is kept, and so is the symbol of
   every closure of a static set. *)
Definition function_decls_avoid (D : dropped)
    (decls : list (function_slot * code_id)) : bool :=
  forallb (fun decl => code_kept D (snd decl)) decls.

Definition closure_symbols_avoid (D : dropped)
    (symbols : list (function_slot * symbol)) : bool :=
  forallb (fun sym => symbol_kept D (snd sym)) symbols.

Fixpoint value_avoids (D : dropped) (v : value) : bool :=
  match v with
  | Value_const _ => true
  | Value_symbol s => symbol_kept D s
  | Value_closure code slots =>
    code_kept D code && forallb (fun '(_, v) => value_avoids D v) slots
  | Value_closure_of_set _ decls slots =>
    function_decls_avoid D decls
    && forallb (fun '(_, v) => value_avoids D v) slots
  | Value_block _ fields | Value_immutable_array fields =>
    forallb (value_avoids D) fields
  | Value_boxed_number _ => true
  | Value_location _ => true
  | Value_partial_application callee args _ =>
    value_avoids D callee && forallb (value_avoids D) args
  end.

Definition simple_avoids (D : dropped) (s : simple) : bool :=
  match s with
  | Symbol sym => symbol_kept D sym
  | Var _ | Const _ => true
  end.

Definition simples_avoid (D : dropped) (args : list simple) : bool :=
  forallb (simple_avoids D) args.

Definition value_slots_avoid (D : dropped)
    (slots : list (value_slot * simple)) : bool :=
  forallb (fun slot => simple_avoids D (snd slot)) slots.

Definition named_avoids (D : dropped) (d : named) : bool :=
  match d with
  | Simple s => simple_avoids D s
  | Prim _ args => simples_avoid D args
  | Set_of_closures code slots => code_kept D code && value_slots_avoid D slots
  | Closure_of_set _ decls slots =>
    function_decls_avoid D decls && value_slots_avoid D slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    simple_avoids D closure
  end.

Definition static_const_avoids (D : dropped) (const : static_const) : bool :=
  match const with
  | Static_const_set_of_closures code slots =>
    code_kept D code && value_slots_avoid D slots
  | Static_const_closure_of_set _ symbols decls slots =>
    closure_symbols_avoid D symbols && function_decls_avoid D decls
    && value_slots_avoid D slots
  | Static_const_block _ fields => simples_avoid D fields
  | Static_const_immutable_value_array elements => simples_avoid D elements
  | Static_const_boxed_number contents => simple_avoids D contents
  | Static_const_immutable_string _ => true
  end.

Definition apply_cont_avoids (D : dropped) (ac : apply_cont) : bool :=
  simples_avoid D (ac_args ac).

Definition apply_avoids (D : dropped) (ap : apply_expr) : bool :=
  simple_avoids D (ap_callee ap)
  && match ap_call_kind ap with
     | Direct code => code_kept D code
     | Indirect_unknown_arity => true
     end
  && simples_avoid D (ap_args ap)
  && simples_avoid D (exn_extra_simples ap).

(* Nothing in the expression refers to a name removed, outside the
   bindings of those names. *)
Fixpoint expr_avoids (D : dropped) (e : expr) : bool :=
  match e with
  | Let _ _ d body => named_avoids D d && expr_avoids D body
  | Let_static bindings body =>
    forallb (binding_avoids D) bindings && expr_avoids D body
  | Let_cont_non_recursive _ h body =>
    handler_avoids D h && expr_avoids D body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_avoids D h) handlers
    && expr_avoids D body
  | Apply ap => apply_avoids D ap
  | Apply_cont ac => apply_cont_avoids D ac
  | Switch scrutinee arms =>
    simple_avoids D scrutinee
    && forallb (fun arm => apply_cont_avoids D (snd arm)) arms
  | Invalid => true
  end
with handler_avoids (D : dropped) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => expr_avoids D handler
  end
with binding_avoids (D : dropped) (binding : static_binding) : bool :=
  match binding with
  | Bind_code id c => if code_kept D id then code_avoids D c else true
  | Bind_deleted_code _ => true
  | Bind_symbol s const =>
    if symbol_kept D s then static_const_avoids D const else true
  end
with code_avoids (D : dropped) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => expr_avoids D body
  end.

(* ======================================================================== *)
(* Removing the bindings                                                    *)
(* ======================================================================== *)

Definition binding_kept (D : dropped) (binding : static_binding) : bool :=
  match binding with
  | Bind_code id _ => code_kept D id
  | Bind_deleted_code _ => true
  | Bind_symbol s _ => symbol_kept D s
  end.

(* The elements kept, each transformed. *)
Fixpoint keep_map {A : Type} (keep : A -> bool) (f : A -> A) (l : list A)
    : list A :=
  match l with
  | [] => []
  | a :: l => if keep a then f a :: keep_map keep f l else keep_map keep f l
  end.

Fixpoint erase_expr (D : dropped) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (erase_expr D body)
  | Let_static bindings body =>
    Let_static (keep_map (binding_kept D) (erase_binding D) bindings)
      (erase_expr D body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (erase_handler D h) (erase_expr D body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive (map (fun '(k, h) => (k, erase_handler D h)) handlers)
      (erase_expr D body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with erase_handler (D : dropped) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (erase_expr D handler)
  end
with erase_binding (D : dropped) (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (erase_code D c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with erase_code (D : dropped) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (erase_expr D body)
  end.

Definition drop_statics (D : dropped) (e : expr) : option expr :=
  if expr_avoids D e then Some (erase_expr D e) else None.

(* ======================================================================== *)
(* Environments in which the names removed do not occur                     *)
(* ======================================================================== *)

(* Every binding of the expression, in its handlers and its code too, is
   one that is kept: the expression binds none of the names removed. *)
Fixpoint binds_only_kept (D : dropped) (e : expr) : bool :=
  match e with
  | Let _ _ _ body => binds_only_kept D body
  | Let_static bindings body =>
    forallb (binding_binds_only_kept D) bindings && binds_only_kept D body
  | Let_cont_non_recursive _ h body =>
    handler_binds_only_kept D h && binds_only_kept D body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => handler_binds_only_kept D h) handlers
    && binds_only_kept D body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => true
  end
with handler_binds_only_kept (D : dropped) (h : continuation_handler) : bool :=
  match h with
  | Handler _ _ handler => binds_only_kept D handler
  end
with binding_binds_only_kept (D : dropped) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code id c => code_kept D id && code_binds_only_kept D c
  | Bind_deleted_code _ => true
  | Bind_symbol s _ => symbol_kept D s
  end
with code_binds_only_kept (D : dropped) (c : code) : bool :=
  match c with
  | Code _ _ _ _ _ body => binds_only_kept D body
  end.

(* A handler, or code, that neither refers to a name removed nor binds
   one. *)
Definition handler_free_of (D : dropped) (h : continuation_handler) : bool :=
  handler_avoids D h && handler_binds_only_kept D h.

Definition code_free_of (D : dropped) (c : code) : bool :=
  code_avoids D c && code_binds_only_kept D c.

Definition env_free_of (D : dropped) (vars : env) : bool :=
  forallb (fun '(_, v) => value_avoids D v) vars.

Fixpoint cont_free_of (D : dropped) (c : cont_value) : bool :=
  match c with
  | Outer _ => true
  | Closure h vars conts recursive =>
    handler_free_of D h
    && env_free_of D vars
    && forallb (fun '(_, c) => cont_free_of D c) conts
    && forallb (fun '(_, h) => handler_free_of D h) recursive
  | Extra_args c extra => cont_free_of D c && forallb (value_avoids D) extra
  | Over_application remaining _ _ return_to exn_to =>
    forallb (value_avoids D) remaining
    && cont_free_of D return_to
    && cont_free_of D exn_to
  end.

Definition allocated_free_of (D : dropped)
    (allocated : statically_allocated_value) : bool :=
  match allocated with
  | Static_closure code slots =>
    code_kept D code && forallb (fun '(_, v) => value_avoids D v) slots
  | Static_closure_of_set _ symbols decls slots =>
    closure_symbols_avoid D symbols && function_decls_avoid D decls
    && forallb (fun '(_, v) => value_avoids D v) slots
  | Static_block _ fields => forallb (value_avoids D) fields
  | Static_immutable_array elements => forallb (value_avoids D) elements
  | Static_boxed_number contents => value_avoids D contents
  | Static_string _ | Static_imported _ => true
  end.

Definition statics_free_of (D : dropped) (statics : static_env) : bool :=
  forallb (fun '(_, c) => code_free_of D c) (static_code statics)
  && forallb (fun '(_, allocated) => allocated_free_of D allocated)
       (static_values statics).

(* The names removed occur nowhere in the environments: no value refers
   to one, and no handler or code refers to one or binds one. *)
Definition envs_free_of (D : dropped) (vars : env) (conts : cont_env)
    (statics : static_env) : bool :=
  env_free_of D vars
  && forallb (fun '(_, c) => cont_free_of D c) conts
  && statics_free_of D statics.

(* And in no value of a cell of the heap, the cells that a later write
   hides included. *)
Definition heap_free_of (D : dropped) (h : heap) : bool :=
  forallb
    (fun binding => forallb (value_avoids D) (values_of_cell (snd binding)))
    h.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* The heap is asked what the environments are asked: a cell may hold a
   symbol, and a load from the cell gives it to the term, which could
   then read through a name that the term after does not bind. The
   empty heap holds no value, so the statement of a term run on its own
   has no premise on the heap. *)
Definition drop_statics_refines_statement : Prop :=
  ∀ (D : dropped) (e e' : expr),
    drop_statics D e = Some e' ->
    ∀ (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
      envs_free_of D vars conts statics = true ->
      heap_free_of D hp = true ->
      refines (State e vars conts statics hp rd)
        (State e' vars conts statics hp rd).

(* Refinement of terms run on their own. Refinement under all environments
   ([expr_refines]) implies it. *)
Definition program_refines (source target : expr) : Prop :=
  refines (State source [] [] empty_static_env empty_heap no_reads)
    (State target [] [] empty_static_env empty_heap no_reads).

Definition drop_statics_program_refines_statement : Prop :=
  ∀ (D : dropped) (e e' : expr),
    drop_statics D e = Some e' -> program_refines e e'.
