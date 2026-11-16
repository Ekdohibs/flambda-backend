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

(* Function calls: what is known of a call, and inlining.

   This file is to be reviewed: [let_params] and [inlined_body], against
   what the simplifier builds when it inlines a call
   (simplify/inlining/inlining_transforms.ml), and the statements.

   - A call whose callee is a closure of known code may be made direct
     ([apply_indirect_known_statement]), whatever the number of its
     arguments: a direct call is the generic application that an
     indirect one is.
   - Inlining a direct call is a refinement ([inline_statement]): the call
     becomes the body of the code, with the closure and the arguments
     bound by [Let] to [my_closure] and the parameters, and with the
     code's return and exception continuations renamed to those of the
     call. It is a refinement and not an equivalence because the call is
     undefined when the callee is not a closure of the code named, and
     when the call does not agree with the code
     (FlambdaSemanticsSpec.call_function): the inlined body tests
     neither.

   The code must be what the code id of the call is bound to: in the
   static environment ([inline_statement]), or by a [Let_static] around
   the call ([inline_let_code_statement]). The other premises
   ([inlinable]) are the conditions for this to be the same computation:
   - the code is closed: the free variables of its body are its parameters
     and [my_closure], and its free continuations are its return and
     exception continuations. In the call the body runs under these only;
     inlined, it runs under everything in scope at the call;
   - the bindings do not capture: [my_closure] and the parameters are not
     free in the arguments. The simplifier ensures it by taking fresh
     names for them, which FlambdaNominalSpec.v allows;
   - there are as many arguments as parameters, and the call is a full
     call: it gives the code as many arguments as it has parameters,
     counted by parameter. With fewer or more it would be a partial
     application or an over-application
     (FlambdaSemanticsSpec.shape_of_closure_call);
   - the body binds nothing static. The call of code that does is
     undefined, wherever it is; inlined at toplevel, the body would have
     a meaning.
   The renaming is a permutation, so a call whose return and exception
   continuations are one continuation, of code in which they are two, is
   not covered; otherwise there is such a renaming
   ([inline_renaming_exists_statement]).

   Nothing here decides whether to inline.

   The proofs are in FlambdaApplyProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* Calls of known code                                                      *)
(* ======================================================================== *)

Definition apply_indirect_known_statement : Prop :=
  ∀ (callee : simple) (args : list simple)
    (extra : list (simple * flambda_kind)) (k exn : continuation)
    (args_arity return_arity : arity) (v : value) (called : code_id)
    (slots : list (value_slot * value)) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    eval_simple vars callee = Some v ->
    closure_of statics v = Some (called, slots) ->
    equivalent
      (State
         (Apply
            (Apply_expr callee Indirect args k exn extra args_arity
               return_arity))
         vars conts statics hp rd)
      (State
         (Apply
            (Apply_expr callee (Direct called) args k exn extra args_arity
               return_arity))
         vars conts statics hp rd).

(* ======================================================================== *)
(* Inlining                                                                 *)
(* ======================================================================== *)

(* The parameters bound to the arguments, one [Let] each, around a body.
   The [Let] of a parameter gives it the kind that the list of kinds has
   for it. Every parameter that has an argument is bound, also where the
   list of kinds is too short (the kind is then [Value]): the bindings
   are those the call makes, whatever the term is annotated with. *)
Fixpoint let_params (params : list variable) (kinds : list flambda_kind)
    (args : list simple) (body : expr) : expr :=
  match params, args with
  | param :: params, arg :: args =>
    Let param (hd Value kinds) (Simple arg)
      (let_params params (tl kinds) args body)
  | _, _ => body
  end.

(* A call of code [c] inlined: [r] renames the continuations of the code
   to those of the call. The parameters have the kinds that the code
   gives them, and [my_closure] the kind [Value]. *)
Definition inlined_body (c : code) (r : renaming) (ap : apply_expr) : expr :=
  Let (code_my_closure c) Value (Simple (ap_callee ap))
    (let_params (code_params c) (code_params_kinds c) (ap_args ap)
       (rename_expr r (code_body c))).

(* What makes inlining a call [ap] of code [c] sound, [r] being the
   renaming of the code's continuations. *)
Definition inlinable (ap : apply_expr) (c : code) (r : renaming) : Prop :=
  (* the code is closed *)
  (∀ x : variable,
    var_free_in_expr x (code_body c) ->
    x = code_my_closure c ∨ x ∈ code_params c)
  ∧ (∀ k : continuation,
      cont_free_in_expr k (code_body c) ->
      k = code_return_continuation c ∨ k = code_exn_continuation c)
  (* the renaming *)
  ∧ (∀ x : variable, rename_variable r x = x)
  ∧ rename_continuation r (code_return_continuation c) = ap_continuation ap
  ∧ rename_continuation r (code_exn_continuation c) = ap_exn_continuation ap
  (* the exception continuation of the call takes the exception alone: the
     exception continuation of the code does *)
  ∧ ap_exn_extra_args ap = []
  (* the bindings do not capture *)
  ∧ ~ var_free_in_simples (code_my_closure c) (ap_args ap)
  ∧ (∀ param : variable,
      param ∈ code_params c -> ~ var_free_in_simples param (ap_args ap))
  ∧ length (code_params c) = length (ap_args ap)
  (* the call is a full call: it has as many arguments as the code has
     parameters, counted by parameter. With fewer or more it is a
     partial application or an over-application, which does more, or
     less, than run the body; and it may be one with as many values as
     the code has variables, where a parameter is an unboxed product of
     nothing *)
  ∧ length (ap_args_arity ap) = length (code_params_arity c)
  (* the body binds nothing static: inlined, it could come to be at
     toplevel, where it would be allowed to *)
  ∧ binds_statics (code_body c) = false
  (* the code is not tupled: a call gives tupled code a tuple, and the
     bindings of the inlined body bind the parameters to the arguments
     as they are *)
  ∧ code_is_tupled c = false.

(* Where the code id of a direct call is bound to code [c], the call
   refines [c] inlined. *)
Definition inline_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    inlinable ap c r ->
    refines (State (Apply ap) vars conts statics hp rd)
      (State (inlined_body c r ap) vars conts statics hp rd).

(* The same as a rewrite of terms, with the binding of the code in sight. *)
Definition inline_let_code_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming),
    ap_call_kind ap = Direct called ->
    inlinable ap c r ->
    expr_refines (Let_static [Bind_code called c] (Apply ap))
      (Let_static [Bind_code called c] (inlined_body c r ap)).

Definition inline_renaming_exists_statement : Prop :=
  ∀ ret exn k k_exn : continuation,
    ret <> exn ->
    k <> k_exn ->
    ∃ r : renaming,
      (∀ x : variable, rename_variable r x = x)
      ∧ rename_continuation r ret = k
      ∧ rename_continuation r exn = k_exn.
