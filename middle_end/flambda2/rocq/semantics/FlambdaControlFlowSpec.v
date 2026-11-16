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

(* Rewrites of control flow that the simplifier performs, stated against the
   semantics of FlambdaSemanticsSpec.v.

   This file is to be reviewed: each statement says that a rewrite preserves
   behaviour ([equivalent], [expr_equivalent]) or may only remove undefined
   behaviour ([expr_refines]). They are first checks that the semantics says
   what is meant, and the first entries of a library of sound rewrites; none
   is yet tied to the code of the simplifier. They are local: each relates
   two terms, or two states, as they stand; FlambdaContextSpec.v lets those
   between terms be applied inside a larger term.

   The proofs are in FlambdaControlFlowProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Continuations                                                            *)
(* ======================================================================== *)

(* Inlining a non-recursive continuation at its application: binding [k] and
   applying it at once is running its handler with the parameters bound to
   the arguments. The handler runs under the continuations of the
   [Let_cont], in which [k] is not bound, as it is not in scope there. The
   premises say when the application has a step: the arguments have
   values, of the kinds that the handler gives its parameters, and as
   many as there are parameters. *)
Definition let_cont_apply_cont_statement : Prop :=
  ∀ (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler : expr) (args : list simple) (vars handler_vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (vs : list value),
    eval_simples vars args = Some vs ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    equivalent
      (State
         (Let_cont_non_recursive k (Handler params metadata handler)
            (Apply_cont (Apply_cont_expr k args None)))
         vars conts statics hp rd)
      (State handler handler_vars conts statics hp rd).

(* The same with one parameter, as a rewrite of terms: the parameter becomes
   a [Let], at the kind that the handler gives it. The application and the
   [Let] then test the same of the value of the argument. *)
Definition let_cont_apply_cont_one_param_statement : Prop :=
  ∀ (k : continuation) (x : variable) (metadata : handler_metadata)
    (kind : flambda_kind) (handler : expr) (arg : simple),
    hm_params_kinds metadata = [kind] ->
    expr_equivalent
      (Let_cont_non_recursive k (Handler [x] metadata handler)
         (Apply_cont (Apply_cont_expr k [arg] None)))
      (Let x kind (Simple arg) handler).

(* Entering a recursive continuation: the handler runs with all the handlers
   of the [Let_cont] in scope. The handlers must bind nothing static. *)
Definition let_cont_recursive_apply_cont_statement : Prop :=
  ∀ (k : continuation) (params : list variable) (metadata : handler_metadata)
    (handler : expr) (handlers : list (continuation * continuation_handler))
    (args : list simple) (vars handler_vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record) (vs : list value),
    handlers_bind_statics handlers = false ->
    lookup k handlers = Some (Handler params metadata handler) ->
    eval_simples vars args = Some vs ->
    map kind_of_value vs = hm_params_kinds metadata ->
    bind_params params vs vars = Some handler_vars ->
    equivalent
      (State
         (Let_cont_recursive handlers
            (Apply_cont (Apply_cont_expr k args None)))
         vars conts statics hp rd)
      (State handler handler_vars
         (rec_closures handlers vars conts ++ conts) statics hp rd).

(* A continuation that only applies itself, applied. *)
Definition loop_forever (k : continuation) : expr :=
  Let_cont_recursive
    [(k,
      Handler [] (make_handler_metadata [])
        (Apply_cont (Apply_cont_expr k [] None)))]
    (Apply_cont (Apply_cont_expr k [] None)).

Definition loop_forever_diverges_statement : Prop :=
  ∀ (k : continuation) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    behaves (State (loop_forever k) vars conts statics hp rd) Diverges.

(* ======================================================================== *)
(* Switches                                                                 *)
(* ======================================================================== *)

(* A [Switch] on a constant is its arm for that constant. *)
Definition switch_const_statement : Prop :=
  ∀ (i : Z) (arms : list (Z * apply_cont)) (ac : apply_cont),
    find_arm i arms = Some ac ->
    expr_equivalent (Switch (Const (Naked_immediate i)) arms) (Apply_cont ac).

(* A [Switch] whose arms are all the same [Apply_cont] may become it. This
   is a refinement and not an equivalence: the [Switch] is undefined when
   its scrutinee has no arm, and the [Apply_cont] is not. *)
Definition switch_same_arms_statement : Prop :=
  ∀ (scrutinee : simple) (arms : list (Z * apply_cont)) (ac : apply_cont),
    (∀ (i : Z) (arm : apply_cont), (i, arm) ∈ arms -> arm = ac) ->
    expr_refines (Switch scrutinee arms) (Apply_cont ac).
