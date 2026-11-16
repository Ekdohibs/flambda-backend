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

(* What is known of a static closure whose binding is in sight is used:
   a value slot is read, and a call is made direct. A normalisation that
   refines.

   This file is to be reviewed: [use_known_closures] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   Simplify replaces the projection of a value slot from a closure that
   it knows by what the slot holds
   (simplify/simplify_unary_primitive.ml, [simplify_project_value_slot],
   from the type of the closure). A symbol has the type of the closure
   that it is bound to from its binding on. So where an exported
   function uses a value that is computed when its module runs, the
   function's closure becomes a symbol bound to a closure one of whose
   value slots is the variable of that value, and where the function is
   then inlined, further down the unit, the projection of the slot from
   the symbol becomes the variable. Likewise it makes a call of a
   closure that it knows a direct call of the code of the closure
   (simplify/simplify_apply_expr.ml, [simplify_function_call], from the
   type of the callee): where an exported function is given to another
   and that one is inlined, the call of its argument is a call of a
   symbol.

   FlambdaAliasesSpec.follow_aliases does both for a
   variable that a [Let] binds to a closure. [use_known_closures] does
   them for a symbol. It goes through a term keeping, for the symbols
   that the [Let_static]s passed bind to closures, what they are bound
   to ([known_closures]). A [Let] of [Project_value_slot slot (Symbol s)]
   or [Project_value_slot_from project_from slot (Symbol s)], with [s]
   known to be a closure that has the slot and, for the second, that is
   at [project_from], becomes a [Let] of what the slot holds
   ([read_slot_named]); the [Let] stays: its variable is then an alias,
   which [follow_aliases] follows. An indirect call whose callee is
   [Symbol s], with [s] known to be a closure, becomes a direct call of
   the code of that closure ([call_known_closure]); inlining it is
   another pass (FlambdaNormaliseSpec.inline_calls). Nothing else is
   rewritten.

   What a slot holds is a variable, a symbol or a constant of the term
   where the symbol is bound. Going under a binder of a variable, the
   closures that have that variable in a value slot are forgotten
   ([forget_slot_var]): the name stands for something else from there
   on. So nothing is asked of the names of the term. A handler of a
   non-recursive continuation is gone into, with what is known where it
   is defined; the handlers of recursive ones and the bodies of code are
   left as they are. In code no variable of the unit is in scope; what
   the unit binds its symbols to with no variable in it is used there
   by FlambdaKnownSymbolsSpec.fold_known_unit.

   The result refines the term, under all environments
   ([use_known_closures_refines_statement]): a symbol is bound once, so
   once the [Let_static] has run the symbol is bound to that closure,
   whose slot has the value that the variable had, and still has, and
   whose code is the code named, so that the direct call does what the
   indirect one did. The pass commutes with renaming, what it knows
   being renamed with the term ([rename_known_closures], which is to be
   reviewed).

   The proofs are in FlambdaKnownClosuresProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNormaliseSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* The symbols known to be closures, each with the constant that it is
   bound to. A symbol may be listed more than once; the first is
   used. *)
Definition known_closures : Type := list (symbol * static_const).

(* The closure that a static binding binds a symbol to. *)
Definition static_closure_made (binding : static_binding)
    : list (symbol * static_const) :=
  match binding with
  | Bind_symbol s
      ((Static_const_set_of_closures _ _
       | Static_const_closure_of_set _ _ _ _) as const) =>
    [(s, const)]
  | Bind_symbol _
      (Static_const_block _ _ | Static_const_boxed_number _
      | Static_const_immutable_string _
      | Static_const_immutable_value_array _)
  | Bind_code _ _ | Bind_deleted_code _ =>
    []
  end.

(* What is known in the body of a [Let_static]: its closures too. *)
Definition closures_under_statics (known : known_closures)
    (bindings : list static_binding) : known_closures :=
  flat_map static_closure_made bindings ++ known.

(* What is known once [x] is bound again: nothing of the closures that
   have the old [x] in a value slot. *)
Definition forget_slot_var (x : variable) (known : known_closures)
    : known_closures :=
  filter
    (fun entry =>
       negb (var_in_static_binding x (Bind_symbol (fst entry) (snd entry))))
    known.

(* The same of parameters, bound one after the other. *)
Fixpoint forget_slot_vars (xs : list variable) (known : known_closures)
    : known_closures :=
  match xs with
  | [] => known
  | x :: xs => forget_slot_vars xs (forget_slot_var x known)
  end.

(* ======================================================================== *)
(* Reading a slot                                                           *)
(* ======================================================================== *)

(* The function slot that a static closure is at, and its value slots.
   A closure of a set of one function, with no function slot named, is
   at FlambdaValueSpec.unnamed_function_slot, as
   FlambdaValueSpec.closure_in_set has it. *)
Definition static_closure_slots (const : static_const)
    : option (function_slot * list (value_slot * simple)) :=
  match const with
  | Static_const_set_of_closures _ slots =>
    Some (unnamed_function_slot, slots)
  | Static_const_closure_of_set slot _ _ slots => Some (slot, slots)
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    None
  end.

(* What the value slot [slot] of the closure that [closure] is known to
   be holds; if [project_from] is given, only where the closure is at
   that function slot. *)
Definition slot_read (known : known_closures)
    (project_from : option function_slot) (slot : value_slot)
    (closure : simple) : option simple :=
  match closure with
  | Symbol s =>
    match lookup s known with
    | Some const =>
      match static_closure_slots const with
      | Some (at_slot, slots) =>
        match project_from with
        | Some from => if Pos.eqb at_slot from then lookup slot slots else None
        | None => lookup slot slots
        end
      | None => None
      end
    | None => None
    end
  | Var _ | Const _ => None
  end.

(* The projection of a value slot from a known static closure is what
   the slot holds. *)
Definition read_slot_named (known : known_closures) (d : named) : named :=
  match d with
  | Project_value_slot slot closure =>
    match slot_read known None slot closure with
    | Some contents => Simple contents
    | None => d
    end
  | Project_value_slot_from project_from slot closure =>
    match slot_read known (Some project_from) slot closure with
    | Some contents => Simple contents
    | None => d
    end
  | Simple _ | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_function_slot _ _ _ =>
    d
  end.

(* The code of a static closure: for a closure of a set, that of the
   function slot it is at. *)
Definition static_closure_code (const : static_const) : option code_id :=
  match const with
  | Static_const_set_of_closures code _ => Some code
  | Static_const_closure_of_set slot _ function_decls _ =>
    lookup slot function_decls
  | Static_const_block _ _ | Static_const_boxed_number _
  | Static_const_immutable_string _
  | Static_const_immutable_value_array _ =>
    None
  end.

(* The code of the closure that the callee is known to be. *)
Definition callee_code (known : known_closures) (callee : simple)
    : option code_id :=
  match callee with
  | Symbol s =>
    match lookup s known with
    | Some const => static_closure_code const
    | None => None
    end
  | Var _ | Const _ => None
  end.

(* A call of unknown arity of a known static closure is a direct call of
   its code. A call of known arity is left: it is undefined wherever
   the code of its callee has not as many parameters as the call has
   groups of arguments
   (FlambdaKnownAritySpec.known_arity_other_arity_statement), and
   nothing here says that it is the direct call. *)
Definition call_known_closure (known : known_closures) (ap : apply_expr)
    : apply_expr :=
  match ap_call_kind ap, callee_code known (ap_callee ap) with
  | Indirect_unknown_arity, Some code =>
    Apply_expr (ap_callee ap) (Direct code) (ap_args ap) (ap_continuation ap)
      (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
      (ap_return_arity ap)
  | Indirect_unknown_arity, None | Direct _, _ | Indirect_known_arity _, _ =>
    ap
  end.

Fixpoint use_closures_knowing (known : known_closures) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind (read_slot_named known d)
      (use_closures_knowing (forget_slot_var x known) body)
  | Let_static bindings body =>
    Let_static bindings
      (use_closures_knowing (closures_under_statics known bindings) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (use_closures_in_handler known h)
      (use_closures_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (use_closures_knowing known body)
  | Apply ap => Apply (call_known_closure known ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with use_closures_in_handler (known : known_closures)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (use_closures_knowing (forget_slot_vars params known) handler)
  end.

Definition use_known_closures (e : expr) : expr := use_closures_knowing [] e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition use_known_closures_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (use_known_closures e).

(* One projection: where the symbol is bound to what the constant makes
   in the environment at hand, the projection has the value of what the
   slot is known to hold. *)
Definition read_slot_named_sound_statement : Prop :=
  ∀ (known : known_closures) (statics : static_env) (vars : env)
    (d : named) (v : value),
    (∀ (s : symbol) (const : static_const),
      (s, const) ∈ known ->
      ∃ allocated : statically_allocated_value,
        eval_static_const vars const = Some allocated
        ∧ lookup s (static_values statics) = Some allocated) ->
    eval_named statics vars d = Some v ->
    eval_named statics vars (read_slot_named known d) = Some v.

(* One call: under the same condition, the call made direct steps as the
   call does. *)
Definition call_known_closure_sound_statement : Prop :=
  ∀ (known : known_closures) (statics : static_env) (hp : heap)
    (rd : read_record) (vars : env)
    (conts : cont_env) (ap : apply_expr),
    (∀ (s : symbol) (const : static_const),
      (s, const) ∈ known ->
      ∃ allocated : statically_allocated_value,
        eval_static_const vars const = Some allocated
        ∧ lookup s (static_values statics) = Some allocated) ->
    step (State (Apply (call_known_closure known ap)) vars conts statics hp rd)
    = step (State (Apply ap) vars conts statics hp rd).

(* The action of a renaming on what is known: the constants are renamed,
   the symbols are not. *)
Definition rename_known_closures (r : renaming) (known : known_closures)
    : known_closures :=
  map_values (rename_static_const r) known.

(* The pass commutes with renaming, what is known being renamed with the
   term. *)
Definition use_closures_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : known_closures) (e : expr),
    rename_expr r (use_closures_knowing known e)
    = use_closures_knowing (rename_known_closures r known) (rename_expr r e).

Definition use_known_closures_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (use_known_closures e) = use_known_closures (rename_expr r e).
