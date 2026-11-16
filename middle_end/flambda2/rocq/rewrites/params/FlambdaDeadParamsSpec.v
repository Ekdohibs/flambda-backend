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

(* Removing the parameters of a loop that are only passed round it: a
   normalisation that refines.

   This file is to be reviewed: [drop_dead_params] and what it is made
   of, which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement.

   The front end makes every mutable variable that a loop assigns a
   parameter of the recursive continuation of the loop. Where the
   variable is not read after the loop, and inside it only to compute
   its next value ([steps := !steps + 2]), the handler uses the
   parameter, but only in a [Let] whose variable is the argument that
   the next application of the continuation gives for the same
   parameter. Simplify removes the parameter, its arguments and the
   [Let]: its data flow analysis (simplify/flow/data_flow_graph.ml,
   [required_names]) finds the names that something other than
   themselves needs, and a parameter of a recursive continuation that is
   not among them goes (simplify/simplify_let_cont_expr.ml,
   [decide_param_usage_recursive]).
   FlambdaUnusedParamsSpec.drop_unused_params does not: it removes from a
   loop the parameters that are not free in the handler as it stands.

   [drop_dead_params] is that pass with two things more. It removes
   every [Let] whose variable is left unused, where its defining
   expression may go (FlambdaNormaliseSpec.named_may_go, as
   FlambdaNormaliseSpec.remove_unused_lets asks), going up the term, so
   that a [Let] that only an argument that goes used goes too. And at a
   recursive [Let_cont] it looks for masks, one for each handler, such
   that the parameters a mask removes are not free in the handler once
   it is rewritten under those very masks ([settle]). It starts from the
   masks that remove every parameter and, while some parameter that is
   removed is free in its rewritten handler, takes for the next masks
   those that keep what the handlers so rewritten use ([used_params] of
   each). Each round keeps more than the one before, so the search ends
   within as many rounds as there are parameters; it is given two more.
   If it has not ended, every parameter of the loop is kept. What it
   finds are the masks that keep the fewest parameters.

   What stops it is what stops [drop_unused_params], but for its third
   point: a continuation that an [Apply] returns to keeps its
   parameters ([returned_to]), the first parameter of an exception
   handler is kept, and the bodies of code are not gone into. A
   parameter that the handler passes to another continuation whose
   handler uses it is kept, as is one that is used to compute anything
   that is needed: a test of the loop, an argument for another
   parameter that is kept.

   The result refines the term ([drop_dead_params_refines_statement]),
   under all environments. It is not equivalent to it: an argument or a
   [Let] that is removed is no longer evaluated, so a term that was
   undefined there may become defined. So the pass is for the term
   before Simplify.

   What is checked and what is not: nothing is asked of the names of
   the term, and nothing of the arities, as for [drop_unused_params].
   The statement does not depend on how the masks of a loop are found:
   the search may stop anywhere, and what is used of its answer is only
   that one more round leaves it as it is ([masks_eqb]).

   Renaming: the functions here compare names by equality only and bind
   none, so the pass commutes with a renaming
   (FlambdaValidatorEquivarianceProofs.drop_dead_params_equivariant).

   The proofs are in FlambdaDeadParamsProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNormaliseSpec FlambdaUnusedParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The search for the masks of a loop                                       *)
(* ======================================================================== *)

Fixpoint bools_eqb (l l' : list bool) : bool :=
  match l, l' with
  | [], [] => true
  | b :: l, b' :: l' => Bool.eqb b b' && bools_eqb l l'
  | _, _ => false
  end.

Definition mask_eqb (mask mask' : option (list bool)) : bool :=
  match mask, mask' with
  | None, None => true
  | Some mask, Some mask' => bools_eqb mask mask'
  | _, _ => false
  end.

(* The same names with the same masks, in the same order. *)
Fixpoint masks_eqb (known known' : masks) : bool :=
  match known, known' with
  | [], [] => true
  | (k, mask) :: known, (k', mask') :: known' =>
    Pos.eqb k k' && mask_eqb mask mask' && masks_eqb known known'
  | _, _ => false
  end.

(* The first masks, going from [loop] by [next], that [next] leaves as
   they are, if they are found within [fuel] rounds. *)
Fixpoint settle (fuel : nat) (next : masks -> masks) (loop : masks)
    : option masks :=
  match fuel with
  | O => None
  | S fuel =>
    let loop' := next loop in
    if masks_eqb loop' loop then Some loop else settle fuel next loop'
  end.

(* How many parameters the handlers have, all together. *)
Definition count_params
    (handlers : list (continuation * continuation_handler)) : nat :=
  fold_right (fun '(_, Handler params _ _) n => length params + n) 0 handlers.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* [drop_dead known e] is [e] with the arguments removed that [known]
   says, the unused [Let]s removed, and the unused parameters of the
   continuations it binds removed with their arguments.
   [live_params inside kept h] is the mask of the handler [h] of a loop,
   rewritten under [inside]: every parameter if [kept] says so, and
   otherwise those that the rewritten handler uses; and
   [drop_dead_in_loop inside kept h] is the handler rewritten, with the
   parameters of that mask. *)
Fixpoint drop_dead (known : masks) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body := drop_dead known body in
    if var_free_b x body || negb (named_may_go d) then Let x kind d body
    else body
  | Let_static bindings body => Let_static bindings (drop_dead known body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler := drop_dead known handler in
    let mask :=
      if returned_to k body then None
      else used_params (hm_is_exn_handler metadata) params handler
    in
    Let_cont_non_recursive k
      (Handler (keep_all_or mask params) (keep_kinds mask metadata) handler)
      (drop_dead ((k, mask) :: known) body)
  | Let_cont_recursive handlers body =>
    (* The masks of the handlers rewritten under [loop]. *)
    let next (loop : masks) : masks :=
      map (fun '(k, h) => (k, live_params (loop ++ known) (returned_to k e) h))
        handlers
    in
    let settled :=
      settle (S (S (count_params handlers))) next
        (map (fun '(k, _) => (k, Some [])) handlers)
    in
    (* Whether the handler of [k] keeps all its parameters. *)
    let kept (k : continuation) : bool :=
      match settled with
      | Some _ => returned_to k e
      | None => true
      end
    in
    let loop :=
      match settled with
      | Some loop => loop
      | None => map (fun '(k, _) => (k, None)) handlers
      end
    in
    Let_cont_recursive
      (map (fun '(k, h) => (k, drop_dead_in_loop (loop ++ known) (kept k) h))
         handlers)
      (drop_dead (loop ++ known) body)
  | Apply _ | Invalid => e
  | Apply_cont ac => Apply_cont (drop_args known ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun arm => (fst arm, drop_args known (snd arm))) arms)
  end
with live_params (inside : masks) (kept : bool) (h : continuation_handler)
    : option (list bool) :=
  match h with
  | Handler params metadata handler =>
    if kept then None
    else
      used_params (hm_is_exn_handler metadata) params
        (drop_dead inside handler)
  end
with drop_dead_in_loop (inside : masks) (kept : bool)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    let handler := drop_dead inside handler in
    let mask :=
      if kept then None
      else used_params (hm_is_exn_handler metadata) params handler
    in
    Handler (keep_all_or mask params) (keep_kinds mask metadata) handler
  end.

Definition drop_dead_params (e : expr) : expr := drop_dead [] e.

Definition drop_dead_params_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (drop_dead_params e).
