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

(* A call of a closure that a [Let] in sight makes is made direct: a
   normalisation that refines.

   This file is to be reviewed: [call_made_closures] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statements.

   Simplify makes a call of a closure that it knows a direct call of the
   code of the closure (simplify/simplify_apply_expr.ml,
   [simplify_function_call], from the type of the callee). A variable
   that a [Let] binds to a closure of a set has the type of that closure
   from the [Let] on, and so has a variable bound to the projection of a
   function slot from it. So where a function that applies its argument
   is inlined at a call that gives it a closure made just before, the
   call of the argument is a call of the variable of that closure, and
   it is direct in the term after.

   FlambdaKnownClosuresSpec.use_known_closures does this for a symbol
   that a [Let_static] in sight binds to a closure, and
   ../types/rocq/FlambdaAliasesSpec.follow_aliases for a variable that a
   [Let] binds to the closure of a set of one function with no function
   slot named ([Set_of_closures]). [call_made_closures] does it for a
   variable that a [Let] binds to a closure at a function slot of a set
   ([Closure_of_set]). It goes through a term keeping, for the variables
   in scope that are known to be such closures, the function slot that
   each is at and the code of each function slot of its set
   ([made_closures]). A variable is known to be one where a [Let] binds
   it ([closure_made]):

   - to [Closure_of_set slot function_decls value_slots]: the closure
     at [slot] of the set whose code is [function_decls];
   - to [Project_function_slot move_from move_to (Var y)], with [y]
     known to be the closure at [move_from] of a set: the closure at
     [move_to] of that set.

   An indirect call whose callee is a variable known to be the closure
   at a function slot of a set that has code for that slot becomes the
   direct call of that code ([call_made_closure]); inlining it is
   another pass (FlambdaNormaliseSpec.inline_calls). Nothing else is
   rewritten: a call that is direct already is left as it is, whatever
   code it names.

   What is known of a variable mentions no other variable name: it
   has function slots and code ids. Going under a binder of a
   variable, what is known of that name is forgotten: the name stands
   for something else from there on. So nothing is asked of the names
   of the term. The handlers of continuations, recursive or not, are
   gone into, with what is known where they are defined less their
   parameters: a handler runs under the variables of its definition.
   The bodies of code are left as they are
   (FlambdaNormaliseSpec.rewrite_code takes the pass into code, with
   nothing known there).

   The result refines the term, under all environments
   ([call_made_closures_refines_statement]): once the [Let] has run, its
   variable is bound to the closure that it made, and the code of a
   closure of a set is the code of the function slot that it is at
   (FlambdaValueSpec.closure_of), so the direct call does what the
   indirect one did (FlambdaApplySpec.apply_indirect_known_statement).
   Only the refinement is stated and proved; that the two are
   equivalent is not claimed.

   The pass changes the kind of calls and nothing else, so the result
   has the free variables and the free continuations of the term, and
   its code is closed where that of the term is
   ([call_closures_knowing_names_statement]); and it commutes with
   renaming, what is known being renamed with the term
   ([rename_made_closures], which is to be reviewed).

   The proofs are in FlambdaClosureCallsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* What is known                                                            *)
(* ======================================================================== *)

(* What is known of a closure that a [Let] made: the function slot that
   it is at, and the code of each function slot of its set. *)
Definition made_closure : Type :=
  (function_slot * list (function_slot * code_id))%type.

(* The variables known to be closures that a [Let] made. Of a variable
   listed more than once, the first entry is the one used. *)
Definition made_closures : Type := finite_map made_closure.

(* The closure that a named expression makes, where the term alone says
   it: a closure at a function slot of a set, or the projection of a
   function slot from a variable known to be the closure at the slot
   projected from. *)
Definition closure_made (known : made_closures) (d : named)
    : option made_closure :=
  match d with
  | Closure_of_set slot function_decls _ => Some (slot, function_decls)
  | Project_function_slot move_from move_to (Var closure) =>
    match lookup closure known with
    | Some (slot, function_decls) =>
      if Pos.eqb slot move_from then Some (move_to, function_decls)
      else None
    | None => None
    end
  | Project_function_slot _ _ (Symbol _ | Const _) | Simple _ | Prim _ _
  | Set_of_closures _ _ | Project_value_slot _ _
  | Project_value_slot_from _ _ _ =>
    None
  end.

(* What is known in the body of a [Let] of [x] to [d]: nothing of the
   old [x], and that the new one is the closure that [d] makes, if the
   term says that it makes one. *)
Definition made_under_let (known : made_closures) (x : variable) (d : named)
    : made_closures :=
  match closure_made known d with
  | Some closure => (x, closure) :: unbind x known
  | None => unbind x known
  end.

(* What is known once the parameters are bound: nothing of them. *)
Fixpoint unbind_params (params : list variable) (known : made_closures)
    : made_closures :=
  match params with
  | [] => known
  | param :: params => unbind_params params (unbind param known)
  end.

(* ======================================================================== *)
(* Calling a closure that a [Let] made                                      *)
(* ======================================================================== *)

(* The code of the closure that the callee is known to be: that of the
   function slot it is at. *)
Definition made_callee_code (known : made_closures) (callee : simple)
    : option code_id :=
  match callee with
  | Var x =>
    match lookup x known with
    | Some (slot, function_decls) => lookup slot function_decls
    | None => None
    end
  | Symbol _ | Const _ => None
  end.

(* An indirect call of a variable known to be a closure is a direct call
   of the code of the closure. *)
Definition call_made_closure (known : made_closures) (ap : apply_expr)
    : apply_expr :=
  match ap_call_kind ap, made_callee_code known (ap_callee ap) with
  | Indirect, Some code =>
    Apply_expr (ap_callee ap) (Direct code) (ap_args ap) (ap_continuation ap)
      (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
      (ap_return_arity ap)
  | Indirect, None | Direct _, _ => ap
  end.

Fixpoint call_closures_knowing (known : made_closures) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (call_closures_knowing (made_under_let known x d) body)
  | Let_static bindings body =>
    Let_static bindings (call_closures_knowing known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (call_closures_in_handler known h)
      (call_closures_knowing known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, call_closures_in_handler known h)) handlers)
      (call_closures_knowing known body)
  | Apply ap => Apply (call_made_closure known ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with call_closures_in_handler (known : made_closures)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (call_closures_knowing (unbind_params params known) handler)
  end.

Definition call_made_closures (e : expr) : expr := call_closures_knowing [] e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition call_made_closures_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (call_made_closures e).

(* One call: where each variable that is known is bound to a closure at
   the function slot known, of a set with the code known, the call made
   direct steps as the call does. *)
Definition call_made_closure_sound_statement : Prop :=
  ∀ (known : made_closures) (statics : static_env) (hp : heap) (vars : env)
    (conts : cont_env) (ap : apply_expr),
    (∀ (x : variable) (slot : function_slot)
       (function_decls : list (function_slot * code_id)),
      lookup x known = Some (slot, function_decls) ->
      ∃ value_slots : list (value_slot * value),
        lookup x vars
        = Some (Value_closure_of_set slot function_decls value_slots)) ->
    step (State (Apply (call_made_closure known ap)) vars conts statics hp)
    = step (State (Apply ap) vars conts statics hp).

(* The pass changes the kind of calls and nothing else: the result has
   the free variables and the free continuations of the term, and its
   code is closed where that of the term is. *)
Definition call_closures_knowing_names_statement : Prop :=
  ∀ (known : made_closures) (e : expr),
    free_variables_of_expr (call_closures_knowing known e)
    = free_variables_of_expr e
    ∧ free_continuations_of_expr (call_closures_knowing known e)
      = free_continuations_of_expr e
    ∧ code_closed (call_closures_knowing known e) = code_closed e.

(* The action of a renaming on what is known: the variables are renamed;
   function slots and code ids are not. *)
Definition rename_made_closures (r : renaming) (known : made_closures)
    : made_closures :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) known.

(* The pass commutes with renaming, what is known being renamed with the
   term. *)
Definition call_closures_knowing_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : made_closures) (e : expr),
    rename_expr r (call_closures_knowing known e)
    = call_closures_knowing (rename_made_closures r known) (rename_expr r e).

Definition call_made_closures_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    rename_expr r (call_made_closures e) = call_made_closures (rename_expr r e).
