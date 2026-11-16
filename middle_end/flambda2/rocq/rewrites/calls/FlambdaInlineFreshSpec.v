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

(* Inlining a call with the variables of the code exchanged for fresh
   ones.

   This file is to be reviewed: [code_variables], [open_code],
   [inlined_at], [inlined_fresh] and the statements; and against the
   simplifier's [Inlining_transforms.inline]
   (simplify/inlining/inlining_transforms.ml), which opens the code by
   [Function_params_and_body.pattern_match], and so at fresh names,
   before it binds the closure and the arguments.

   FlambdaApplySpec.v says that a direct call refines the body of its
   code inlined ([inline_let_code_statement]), with [my_closure] and
   the parameters of the code as the variables of the [Let]s that bind
   the closure and the arguments. Two of its premises
   (FlambdaApplySpec.inlinable) say that those bindings capture nothing:
   [my_closure] and the parameters are not free in the arguments. The
   simplifier has no such condition, because it takes fresh names.
   Here is that:

   - [open_code fresh c] is [c] with [my_closure] and its parameters
     exchanged for the variables [fresh], in its body too. It is
     [rename_code] at the swaps, and the binders of closed code may be
     renamed (FlambdaNominalSpec.rename_code_binders_statement).
   - A call refines the body of the code opened at [fresh], inlined
     ([inline_at_statement]), for every list [fresh] of as many
     different variables that are not variables of the code and not
     free in the arguments. This is the explicit form: it says which
     names will do. The two premises about capture are gone; in their
     place the code is asked to be closed in the sense of
     FlambdaBindersSpec.v and its variables to be distinct, as they
     are in a well-formed term.
   - [inlined_fresh avoid c r ap] is the function: it takes the names to
     avoid, makes the fresh variables by
     FlambdaFreshQuantifierSpec.fresh_names, and inlines at them. A
     call refines it when the names to avoid include the free
     variables of the arguments ([inline_fresh_statement]).
   - The choice does not matter ([inline_at_choice_statement],
     [inline_fresh_choice_statement]): two lists of fresh variables,
     or two lists of names to avoid, give alpha-equivalent terms. The
     variables made are bound in the result, by the [Let]s; so the two
     results are related by alpha-equivalence, which the permutation
     that exchanges the two lists gives
     (FlambdaAlphaSpec.Alpha_rename), and not by equality of the
     second with the first renamed: a fresh variable is outside the
     free variables of the body and may be one that the body binds.
     For this the fresh variables are also outside the callee.
   - With the quantifier ([inline_at_fresh_statement]): for all but
     finitely many lists of variables the call refines the code
     inlined at them. It follows from the explicit form, and says
     less.

   The refinement is given in the two forms of FlambdaApplySpec.v: of
   a call with the binding of its code in sight, as
   [inline_let_code_statement] is ([inline_at_statement],
   [inline_fresh_statement]); and of a state in whose static
   environment the code id of the call is bound to the code, as
   [inline_statement] is ([inline_at_state_statement],
   [inline_fresh_state_statement]). The second is the form for a pass
   that inlines the calls of the code it has seen bound.

   Nothing is trusted here. The proofs are in
   FlambdaInlineFreshProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaBindersSpec FlambdaNominalSpec
  FlambdaAlphaSpec FlambdaApplySpec FlambdaFreshQuantifierSpec
  FlambdaFreshBindersSpec.
Import ListNotations.

(* ======================================================================== *)
(* Code opened at fresh variables                                           *)
(* ======================================================================== *)

(* The variables that code binds. *)
Definition code_variables (c : code) : list variable :=
  code_my_closure c :: code_params c.

(* [Function_params_and_body.pattern_match], for the variables: the
   code with [my_closure] and its parameters exchanged for [fresh], a
   variable for [my_closure] and then one for each parameter. *)
Definition open_code (fresh : list variable) (c : code) : code :=
  rename_code (swap_variables (code_variables c) fresh) c.

(* The call inlined at those variables. *)
Definition inlined_at (fresh : list variable) (c : code) (r : renaming)
    (ap : apply_expr) : expr :=
  inlined_body (open_code fresh c) r ap.

(* The same with the variables made: as many as the code binds, outside
   the names to avoid and the variables of the code. *)
Definition inlined_fresh (avoid : list variable) (c : code) (r : renaming)
    (ap : apply_expr) : expr :=
  inlined_at
    (fresh_names (length (code_variables c)) (avoid ++ code_variables c)) c r
    ap.

(* ======================================================================== *)
(* A call refines its code inlined at fresh variables                       *)
(* ======================================================================== *)

(* What is asked of a call [ap] of code [c] and of the renaming [r] of
   the continuations of the code, whatever the variables: the premises
   of FlambdaApplySpec.inlinable other than those about capture, with
   closed code said as FlambdaBindersSpec.v says it. *)
Definition inlinable_at_fresh (ap : apply_expr) (c : code) (r : renaming)
    : Prop :=
  code_is_closed c = true
  ∧ distinct (code_variables c) = true
  ∧ (∀ x : variable, rename_variable r x = x)
  ∧ rename_continuation r (code_return_continuation c) = ap_continuation ap
  ∧ rename_continuation r (code_exn_continuation c) = ap_exn_continuation ap
  ∧ ap_exn_extra_args ap = []
  ∧ length (code_params c) = length (ap_args ap)
  ∧ length (ap_args_arity ap) = length (code_params_arity c)
  ∧ binds_statics (code_body c) = false
  ∧ code_is_tupled c = false.

(* The explicit form: any variables that are different from one
   another, are not variables of the code and are not free in the
   arguments. *)
Definition inline_at_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (fresh : list variable),
    ap_call_kind ap = Direct called ->
    inlinable_at_fresh ap c r ->
    length fresh = length (code_variables c) ->
    distinct_outside
      (code_variables c ++ free_variables_of_simples (ap_args ap)) fresh ->
    expr_refines (Let_static [Bind_code called c] (Apply ap))
      (Let_static [Bind_code called c] (inlined_at fresh c r ap)).

(* The function: the names to avoid include the free variables of the
   arguments. *)
Definition inline_fresh_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (avoid : list variable),
    ap_call_kind ap = Direct called ->
    inlinable_at_fresh ap c r ->
    all_among (free_variables_of_simples (ap_args ap)) avoid = true ->
    expr_refines (Let_static [Bind_code called c] (Apply ap))
      (Let_static [Bind_code called c] (inlined_fresh avoid c r ap)).

(* The same of a state, where the code id of the call is bound to the
   code in the static environment: FlambdaApplySpec.inline_statement
   with the code opened at fresh variables. *)
Definition inline_at_state_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (fresh : list variable) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    inlinable_at_fresh ap c r ->
    length fresh = length (code_variables c) ->
    distinct_outside
      (code_variables c ++ free_variables_of_simples (ap_args ap)) fresh ->
    refines (State (Apply ap) vars conts statics hp rd)
      (State (inlined_at fresh c r ap) vars conts statics hp rd).

Definition inline_fresh_state_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (avoid : list variable) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    inlinable_at_fresh ap c r ->
    all_among (free_variables_of_simples (ap_args ap)) avoid = true ->
    refines (State (Apply ap) vars conts statics hp rd)
      (State (inlined_fresh avoid c r ap) vars conts statics hp rd).

(* With the quantifier. *)
Definition inline_at_fresh_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming),
    ap_call_kind ap = Direct called ->
    inlinable_at_fresh ap c r ->
    И[length (code_variables c)] fresh,
      expr_refines (Let_static [Bind_code called c] (Apply ap))
        (Let_static [Bind_code called c] (inlined_at fresh c r ap)).

(* ======================================================================== *)
(* The choice of the fresh variables does not matter                        *)
(* ======================================================================== *)

(* Two lists of fresh variables, each outside the variables of the
   code and the free variables of the call, the callee included. *)
Definition inline_at_choice_statement : Prop :=
  ∀ (ap : apply_expr) (c : code) (r : renaming)
    (fresh fresh' : list variable),
    code_is_closed c = true ->
    distinct (code_variables c) = true ->
    (∀ x : variable, rename_variable r x = x) ->
    length (code_params c) = length (ap_args ap) ->
    length fresh = length (code_variables c) ->
    length fresh' = length (code_variables c) ->
    distinct_outside (code_variables c ++ free_variables_of_apply ap) fresh ->
    distinct_outside (code_variables c ++ free_variables_of_apply ap) fresh'
    ->
    alpha_equivalent (inlined_at fresh c r ap) (inlined_at fresh' c r ap).

(* Two lists of names to avoid, each with the free variables of the
   call. *)
Definition inline_fresh_choice_statement : Prop :=
  ∀ (ap : apply_expr) (c : code) (r : renaming) (avoid avoid' : list variable),
    code_is_closed c = true ->
    distinct (code_variables c) = true ->
    (∀ x : variable, rename_variable r x = x) ->
    length (code_params c) = length (ap_args ap) ->
    all_among (free_variables_of_apply ap) avoid = true ->
    all_among (free_variables_of_apply ap) avoid' = true ->
    alpha_equivalent (inlined_fresh avoid c r ap)
      (inlined_fresh avoid' c r ap).
