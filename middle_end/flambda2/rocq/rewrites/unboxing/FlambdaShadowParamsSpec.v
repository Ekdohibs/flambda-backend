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


(* The parameters of continuations that hold a tagged immediate and
   are untagged, or passed on to where one is, are bound again to
   themselves at the head of their handlers: a normalisation that is
   an equivalence, and that prepares FlambdaTwinParamsSpec.twin_params.

   This file is to be reviewed: [shadow_expr] and what it is made of,
   which are extracted and run by the validation of
   middle_end/flambda2/tests/semantics/, and the statement. A round
   of the second phase of the validation applies the pass to the term
   before Simplify once it has settled the term, just before
   FlambdaRenameApartSpec.rename_apart_avoiding
   (FlambdaValidateSpec.with_twins). The
   positions that the pass is run with ([shadow_positions]) are
   extracted and run too, and nothing is believed of them: the
   statement is about any positions.

   Why. Where Simplify unboxes a parameter of a continuation that
   holds a tagged immediate (simplify/unboxing/), the continuation
   gets a parameter more for the number, and the tagged parameter
   stays if something else uses it. So for a flag that a loop both
   tests and keeps,

     let any = ref false in
     for i = 0 to n do any := !any || i > limit done;
     !any

   the continuation where the arms of [||] meet has two parameters
   after Simplify, the flag as a value and the flag as a number,
   where it had one before. A pass on the term before that makes the
   second parameter needs a name for it, and the passes of the
   validation make no names but
   FlambdaRenameApartSpec.rename_apart_avoiding. This pass gives that
   one a binder to name:

     k (p) = body        becomes        k (p) = let p = p in body

   and after the renaming the handler is [k (p) = let p' = p in body]
   with [p'] for [p] in the body, which
   FlambdaTwinParamsSpec.twin_params makes [k (p', p) = body].

   [shadow_expr sites] puts, at the head of the handler of every
   continuation, a [Let] of each parameter at a position that [sites]
   has for the name of the continuation, to itself, with the kind
   that the handler gives that parameter ([shadow_lets]); the
   parameters are taken in their order. At the head of a handler
   whose parameters are not distinct no [Let] is put; its body is
   gone into as any other. The handlers of recursive
   continuations are gone into; the bodies of code are not:
   FlambdaNormaliseSpec.rewrite_code takes a pass there.

   The result is equivalent to the term
   ([shadow_expr_equivalent_statement]), under all environments,
   whatever the positions. Why it holds: a handler runs when it is
   given values of the kinds of its parameters, so each such [Let]
   has a value of its kind, and binds the variable to the value that
   it had.

   The positions that the pass is run with ([shadow_positions]): the
   parameters of kind [Value] of the handlers that are not exception
   handlers, outside code, that their handler untags somewhere, or
   gives to a continuation at such a position ([needed_positions],
   by rounds from nothing, six of them). Names are compared as they
   are, with no regard for scope. Nothing is claimed of these
   positions. Where there is none the pass gives the term back.

   What it is not: it does not say which parameters Simplify unboxes;
   a parameter bound again that FlambdaTwinParamsSpec.twin_params
   then leaves is followed and removed by the rounds, as any alias
   is.

   Renaming: the functions here compare names by equality only and
   make none. FlambdaShadowParamsNamesProofs.v has the lemmas that
   the pass commutes with renaming and keeps scope; the statements
   are in FlambdaValidatorEquivarianceSpec.v and
   FlambdaStaticsScopeSpec.v.

   The proofs are in FlambdaShadowParamsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaPrimitiveModel FlambdaSemanticsSpec
  FlambdaNormaliseSpec FlambdaUntagParamsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The expression under a [Let] of each parameter at a position of
   [us] to itself, with its kind; [i] is the position of the first
   parameter. *)
Fixpoint shadow_lets (us : list nat) (i : nat) (params : list variable)
    (kinds : list flambda_kind) (e : expr) : expr :=
  match params, kinds with
  | p :: params, kind :: kinds =>
    if at_position us i
    then Let p kind (Simple (Var p)) (shadow_lets us (S i) params kinds e)
    else shadow_lets us (S i) params kinds e
  | _, _ => e
  end.

Fixpoint shadow_expr (sites : positions) (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (shadow_expr sites body)
  | Let_static bindings body => Let_static bindings (shadow_expr sites body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (shadow_handler sites (positions_of sites k) h)
      (shadow_expr sites body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) => (k, shadow_handler sites (positions_of sites k) h))
         handlers)
      (shadow_expr sites body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with shadow_handler (sites : positions) (us : list nat)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (if distinct params
       then
         shadow_lets us 0 params (hm_params_kinds metadata)
           (shadow_expr sites handler)
       else shadow_expr sites handler)
  end.

(* The result behaves as the term does, under all environments,
   whatever the positions. *)
Definition shadow_expr_equivalent_statement : Prop :=
  ∀ (sites : positions) (e : expr),
    expr_equivalent e (shadow_expr sites e).

(* ======================================================================== *)
(* The positions that the pass is run with                                  *)
(* ======================================================================== *)

(* The handlers that a term binds outside code, each with its name. *)
Fixpoint handlers_of (e : expr) {struct e}
    : list (continuation * continuation_handler) :=
  match e with
  | Let _ _ _ body | Let_static _ body => handlers_of body
  | Let_cont_non_recursive k h body =>
    (k, h) :: handlers_in_handler h ++ handlers_of body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(k, h) => (k, h) :: handlers_in_handler h) handlers
    ++ handlers_of body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handlers_in_handler (h : continuation_handler) {struct h}
    : list (continuation * continuation_handler) :=
  match h with
  | Handler _ _ handler => handlers_of handler
  end.

(* The variable is untagged somewhere in the expression, outside
   code. *)
Fixpoint untagged_in (p : variable) (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ d body =>
    match untag_of d with
    | Some q => Pos.eqb q p
    | None => false
    end
    || untagged_in p body
  | Let_static _ body => untagged_in p body
  | Let_cont_non_recursive _ h body =>
    untagged_in_handler p h || untagged_in p body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => untagged_in_handler p h) handlers
    || untagged_in p body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => false
  end
with untagged_in_handler (p : variable) (h : continuation_handler)
    {struct h} : bool :=
  match h with
  | Handler _ _ handler => untagged_in p handler
  end.

(* The application gives the variable at a position of [needed]. *)
Definition gives_needed (needed : positions) (p : variable)
    (ac : apply_cont) : bool :=
  var_in_simples p
    (moved_from (positions_of needed (ac_continuation ac)) 0 (ac_args ac)).

(* Some application of the expression does, outside code. *)
Fixpoint given_needed (needed : positions) (p : variable) (e : expr)
    {struct e} : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => given_needed needed p body
  | Let_cont_non_recursive _ h body =>
    given_needed_in_handler needed p h || given_needed needed p body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => given_needed_in_handler needed p h) handlers
    || given_needed needed p body
  | Apply _ | Invalid => false
  | Apply_cont ac => gives_needed needed p ac
  | Switch _ arms => existsb (fun '(_, ac) => gives_needed needed p ac) arms
  end
with given_needed_in_handler (needed : positions) (p : variable)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler => given_needed needed p handler
  end.

(* The positions of the parameters of kind [Value] that the handler
   untags or gives at a position of [needed]; [i] is the position of
   the first parameter. *)
Fixpoint needed_from (needed : positions) (i : nat) (params : list variable)
    (kinds : list flambda_kind) (handler : expr) : list nat :=
  match params, kinds with
  | p :: params, kind :: kinds =>
    if kind_eqb kind Value
       && (untagged_in p handler || given_needed needed p handler)
    then i :: needed_from needed (S i) params kinds handler
    else needed_from needed (S i) params kinds handler
  | _, _ => []
  end.

Definition needed_of (needed : positions) (h : continuation_handler)
    : list nat :=
  match h with
  | Handler params metadata handler =>
    if hm_is_exn_handler metadata then []
    else needed_from needed 0 params (hm_params_kinds metadata) handler
  end.

(* One round: the positions of each handler, given those of the round
   before. *)
Definition needed_round (hs : list (continuation * continuation_handler))
    (needed : positions) : positions :=
  map (fun '(k, h) => (k, needed_of needed h)) hs.

Fixpoint needed_positions (rounds : nat)
    (hs : list (continuation * continuation_handler)) (needed : positions)
    : positions :=
  match rounds with
  | O => needed
  | S rounds => needed_positions rounds hs (needed_round hs needed)
  end.

Definition shadow_positions (e : expr) : positions :=
  needed_positions 6 (handlers_of e) [].

(* The pass, with the positions that it finds; the term as it is where
   it finds none. *)
Definition shadow_params (e : expr) : expr :=
  match count_positions (shadow_positions e) with
  | O => e
  | S _ => shadow_expr (shadow_positions e) e
  end.
