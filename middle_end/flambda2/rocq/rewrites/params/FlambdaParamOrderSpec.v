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

(* The parameters of a continuation put in the order in which the text of
   its handler first names them: a normalisation that is an equivalence.

   This file is to be reviewed: [order_params] and what it is made of,
   and the statement.

   Simplify gives a continuation new parameters and its applications
   new arguments, and puts each parameter that it makes in front of
   those that it has made
   (simplify/continuation_extra_params_and_args.ml, [add]). The
   rewrites of the validation that give a continuation a parameter put
   it where their own rule says. So the two terms that the validation
   compares may have one handler with its parameters, and the
   applications of its continuation with their arguments, in two
   orders, and be the same otherwise. No other rewrite changes the
   order of the parameters of a continuation.

   [order_params] gives both terms one rule for the order. At a
   non-recursive [Let_cont] of [k] ([order_node]) the parameters of
   the handler are put in the order of their first occurrence in the
   list of the free variables of the handler
   (FlambdaBindersSpec.free_variables_of_expr); those that are not in
   the list come last, in the order they had ([use_order]). That list
   follows the text of the handler, through every handler inside it
   and every arm of a [Switch], taken or not: the order is one of the
   text, and not of what a run reads first. It is the one place where
   the order of that list is asked, and not only what is in it. The
   kinds of the parameters go with them, and so do the arguments of
   every application of [k] in the body of the [Let_cont], in the
   handlers of the continuations bound there too ([order_args]). No
   name is made. Two terms that behave alike need not get one form by
   it: two parameters that the handler does not name keep their
   order.

   What the rewrite asks at a node:
   - that the handler is not an exception handler: the first parameter
     of one is the exception, and what is raised gives it;
   - that the parameters are distinct, and as many as the kinds;
   - that every mention of [k] in the body of the [Let_cont] is an
     application of [k] to as many arguments as there are parameters
     ([all_apply_with]): no call returns or raises to [k], whose
     results come in the order of the call; no trap action names [k];
     and no [Let_cont] inside binds [k] again. The bodies of code are
     not looked into: code runs with its own continuations;
   - that the order computed is one of the positions of the
     parameters, each once ([is_order]). It is by how it is made; the
     test is there so that, for its choice of order, the proof of the
     equivalence uses only that the test succeeds, and not how the
     order is computed. The law of renaming
     (FlambdaParamOrderNamesSpec.v) also uses how the order is
     computed.
   Where one of these fails the node is left as it is.

   The rewrite is made at each non-recursive [Let_cont] that the walk
   of FlambdaSpineWalkSpec.walk_spine reaches, its handler and its
   body first. The walk does not go into the handlers of recursive
   continuations nor into the bodies of code: a [Let_cont] there is
   not a node of this pass (FlambdaRecHandlersSpec.rewrite_rec_handlers
   and FlambdaNormaliseSpec.rewrite_code take a pass there). The
   rewrite of a node that the walk reaches is another traversal:
   [all_apply_with] and [order_args] do go into the handlers of the
   recursive continuations bound in the body of the node, where they
   test and order the applications of the continuation of the node.
   They do not order the parameters of those recursive continuations:
   the parameters of a recursive continuation keep their order.

   One walk may leave another ordering to do. The inside of a node
   is rewritten first; the rewrite of a [Let_cont] around it then
   changes the order of the arguments of applications that the
   handlers inside name, and with it the order in which those handlers
   first name their parameters. A second walk orders those. Nothing
   that is stated here depends on the order having settled.

   The result is equivalent to the term
   ([order_params_equivalent_statement]), under all environments:
   where a closure is applied the machine tests the kind of each value
   against the kind of its parameter and binds the two, and with the
   parameters, the kinds and the arguments in one other order the same
   values meet the same kinds and the same names. The statement is of
   the machine of FlambdaSemanticsSpec.v, which does not execute trap
   actions.

   Renaming: the functions here compare names by equality only and
   make none; the order is read off the list of the free variables of
   the handler, which a renaming renames in place.

   The proofs are in FlambdaParamOrderProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool Arith.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaSemanticsSpec FlambdaBindersSpec FlambdaSpineWalkSpec
  FlambdaForwardArmSpec.
Import ListNotations.

(* ======================================================================== *)
(* The order                                                                *)
(* ======================================================================== *)

(* Where a handler first names a parameter: its position among the
   free variables of the handler, in the order of their list. A
   parameter that is not among them comes after all those that are. *)
Definition param_use_key (free : list variable) (x : variable) : nat :=
  match position_of x free with
  | Some i => i
  | None => length free
  end.

(* An entry (key, position) put in a list that is sorted by key, after
   the entries of the same key. *)
Fixpoint insert_by_use (entry : nat * nat) (l : list (nat * nat))
    : list (nat * nat) :=
  match l with
  | [] => [entry]
  | e :: rest =>
    if Nat.leb (fst e) (fst entry) then e :: insert_by_use entry rest
    else entry :: l
  end.

(* The entries of the parameters from position [i] on, put one after
   the other in a sorted list. *)
Fixpoint use_entries (free : list variable) (i : nat)
    (params : list variable) (sorted : list (nat * nat))
    : list (nat * nat) :=
  match params with
  | [] => sorted
  | x :: params =>
    use_entries free (S i) params
      (insert_by_use (param_use_key free x, i) sorted)
  end.

(* The positions of the parameters, in the order in which the handler
   first names them. *)
Definition use_order (params : list variable) (handler : expr) : list nat :=
  map snd (use_entries (free_variables_of_expr handler) 0 params []).

(* The elements of a list at the positions of an order, if it has an
   element at each. *)
Fixpoint in_order {A : Type} (order : list nat) (l : list A)
    : option (list A) :=
  match order with
  | [] => Some []
  | i :: order =>
    match nth_error l i, in_order order l with
    | Some a, Some rest => Some (a :: rest)
    | _, _ => None
    end
  end.

(* An order of [n] positions: as many, each below [n], and each
   position below [n] among them. *)
Definition is_order (n : nat) (order : list nat) : bool :=
  Nat.eqb (length order) n
  && forallb (fun j => Nat.ltb j n) order
  && forallb (fun i => existsb (Nat.eqb i) order) (seq 0 n).

(* ======================================================================== *)
(* The applications of the continuation                                     *)
(* ======================================================================== *)

(* No trap action of an [Apply_cont] names [k]; and if the [Apply_cont]
   applies [k], it has [n] arguments. *)
Definition apply_cont_with (k : continuation) (n : nat) (ac : apply_cont)
    : bool :=
  match ac_trap_action ac with
  | Some action => negb (Pos.eqb (trap_action_handler action) k)
  | None => true
  end
  && (if Pos.eqb (ac_continuation ac) k
      then Nat.eqb (length (ac_args ac)) n
      else true).

(* Every mention of [k] in a term is such an application, and [k] is
   not bound again. *)
Fixpoint all_apply_with (k : continuation) (n : nat) (e : expr) {struct e}
    : bool :=
  match e with
  | Let _ _ _ body => all_apply_with k n body
  | Let_static _ body => all_apply_with k n body
  | Let_cont_non_recursive k' h body =>
    negb (Pos.eqb k' k)
    && handler_all_apply_with k n h
    && all_apply_with k n body
  | Let_cont_recursive handlers body =>
    forallb
      (fun '(k', h) => negb (Pos.eqb k' k) && handler_all_apply_with k n h)
      handlers
    && all_apply_with k n body
  | Apply ap =>
    negb (Pos.eqb (ap_continuation ap) k)
    && negb (Pos.eqb (ap_exn_continuation ap) k)
  | Apply_cont ac => apply_cont_with k n ac
  | Switch _ arms => forallb (fun '(_, ac) => apply_cont_with k n ac) arms
  | Invalid => true
  end
with handler_all_apply_with (k : continuation) (n : nat)
    (h : continuation_handler) {struct h} : bool :=
  match h with
  | Handler _ _ handler => all_apply_with k n handler
  end.

(* An application of [k] with its arguments in an order. *)
Definition order_args_of (k : continuation) (order : list nat)
    (ac : apply_cont) : apply_cont :=
  if Pos.eqb (ac_continuation ac) k
  then
    match in_order order (ac_args ac) with
    | Some args =>
      {| ac_continuation := ac_continuation ac;
         ac_args := args;
         ac_trap_action := ac_trap_action ac |}
    | None => ac
    end
  else ac.

(* Every application of [k] in a term so. *)
Fixpoint order_args (k : continuation) (order : list nat) (e : expr)
    {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (order_args k order body)
  | Let_static bindings body =>
    Let_static bindings (order_args k order body)
  | Let_cont_non_recursive k' h body =>
    Let_cont_non_recursive k' (order_args_in_handler k order h)
      (order_args k order body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k', h) => (k', order_args_in_handler k order h)) handlers)
      (order_args k order body)
  | Apply_cont ac => Apply_cont (order_args_of k order ac)
  | Switch scrutinee arms =>
    Switch scrutinee
      (map (fun '(n, ac) => (n, order_args_of k order ac)) arms)
  | Apply _ | Invalid => e
  end
with order_args_in_handler (k : continuation) (order : list nat)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (order_args k order handler)
  end.

(* ======================================================================== *)
(* The pass                                                                 *)
(* ======================================================================== *)

(* The pass at one node: the parameters of a non-recursive [Let_cont]
   in the order in which its handler first names them, their kinds and
   the arguments of the applications of its continuation with them.
   Any other term is left as it is, and so is a [Let_cont] of which
   one of the things asked above fails. *)
Definition order_node (e : expr) : expr :=
  match e with
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let n := length params in
    let order := use_order params handler in
    if negb (hm_is_exn_handler metadata)
       && distinct params
       && Nat.eqb (length (hm_params_kinds metadata)) n
       && is_order n order
       && all_apply_with k n body
    then
      match
        in_order order params, in_order order (hm_params_kinds metadata)
      with
      | Some params', Some kinds' =>
        Let_cont_non_recursive k
          (Handler params'
             {| hm_params_kinds := kinds'; hm_is_exn_handler := false |}
             handler)
          (order_args k order body)
      | _, _ => e
      end
    else e
  | Let _ _ _ _ | Let_static _ _ | Let_cont_recursive _ _ | Apply _
  | Apply_cont _ | Switch _ _ | Invalid => e
  end.

(* The pass: that step at each non-recursive [Let_cont] that
   FlambdaSpineWalkSpec.walk_spine reaches, its handler and its body
   first. The walk does not go into the handlers of recursive
   continuations, nor into the bodies of code. *)
Definition order_params (e : expr) : expr := walk_spine order_node e.

(* The result behaves as the term does, under all environments. *)
Definition order_params_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (order_params e).
