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

(* The untagging of a parameter that states a tagged immediate, of a
   continuation that a call returns to, is bound at the head of the
   handler: a normalisation that is an equivalence.

   The statement of the rewrite,
   [bind_untaggings_equivalent_statement], is of the machine that has
   no step where a handler is applied to values that are not of what
   its parameters state (FlambdaSemanticsSpec.apply_cont_value,
   through FlambdaSubkindSpec.values_of_subkinds; read back in
   FlambdaSubkindBinderSpec.handler_binds_at_subkinds_statement). That
   test is decision D25, at parameters, and without it the statement is
   false: this header says why. This file is to be reviewed:
   [bind_untaggings] and what it is made of, and the two statements.
   The proofs are in FlambdaUntagAtBinderProofs.v, which needs no
   review.

   Simplify can unbox the parameters of a continuation
   (simplify/unboxing/). Once its decision for a parameter is to
   unbox a naked immediate, the continuation gets a parameter more for
   the number, and that decision is kept whether or not the uses gain
   by it ([Is_unboxing_beneficial.filter_non_beneficial_decisions]:
   "At worst, this unboxing untags an integer"). The other conditions
   of unboxing are not passed over. The type that the decision is made
   from may be no more than what the binder states: the kind of a
   parameter has a subkind ([Flambda_kind.With_subkind.t]), and
   Simplify reads it into the type
   ([More_type_creators.unknown_with_subkind]). At a use that is an
   application the number is given there. A call cannot give it: where
   a call returns to the continuation, Simplify makes a wrapper that
   the call returns to, which untags what the call gives and applies
   the continuation ([Unboxing_epa.unbox_arg], in its case
   [Added_by_wrapper_at_rewrite_use], with
   [New_let_binding_with_named_args]):

     k_wrapper (param : imm tagged) =
       let unboxed = %untag_imm (param) in cont k (unboxed)

   So for a boolean that the source keeps across another call,

     let b = p x in
     let n = count p rest in
     if b then n + 1 else n

   the term before Simplify untags [b] after the second call, where it
   is tested, and the term after untags it where the first call
   returns, before the second call.

   FlambdaUntagParamsSpec.untag_params and
   FlambdaTwinParamsSpec.twin_params follow that unboxing where every
   application of the continuation is in the term, and both ask that
   no call returns to the continuation: a call gives a continuation
   values that the term does not show. This pass is for that case.

   What it does ([bind_untaggings]). At the handler of a non-recursive
   continuation that a call in the body of its [Let_cont] returns to
   ([returns_to]), for each parameter [x] that states at least "not
   null, a tagged immediate" ([states_tagged]: a tagged immediate, or
   a variant with no constructor that is a block, and not null), if
   the handler untags [x] somewhere:

     k (x) = H[ let u = Untag_immediate (x) in body ]
     becomes
     k (x) = let u = Untag_immediate (x) in H[ body ]

   - [u] is the variable of the first [Let] of the handler that
     untags [x] at the kind of a naked immediate ([first_untagging]):
     the first in one walk, through [Let]s and [Let_static]s, into
     the handler of a continuation before the body of its [Let_cont],
     into the handlers of recursive continuations; not into the
     bodies of code; and not under a binder of [x].
   - A [Let] of [u] to that untagging is put at the head of the
     handler, and every [Let] of [u] to the untagging of [x] that is
     reached from the head with no other binder of [u] and no binder
     of [x] on the way is taken away ([drop_untagging]). Any other
     untagging of [x], bound to another variable, stays.
     FlambdaCseSpec.share_known_primitives, applied after this pass,
     can make it a [Let] of a variable that is known to hold the same
     untagging; what that pass knows is forgotten under a binder of
     that variable or of [x].
   - The first untagging gives the name [u], and no more: it need not
     be one of the [Let]s that are taken away. If it is under another
     binder of [u] (a [Let] of [u] to something else, a handler with
     the parameter [u]), it stays in its place, and the [Let] at the
     head may then be read by nothing. A use of [u] under another
     binder of [u] keeps the value of that binder.
   - Nothing is done for [x] where [u] has a free use in the handler
     ([untagging_at_head]): the [Let] at the head would capture that
     use. Nothing is done either where [u] is a parameter of the
     handler. That is a caution, and the equivalence does not need it
     (a parameter with no free use in the handler would be hidden with
     no effect): with it the pass never hides a parameter of the
     handler, and [u] is not [x]. Nothing is done at an exception
     handler, nor at a handler whose parameters are not distinct.
   [bind_untaggings] leaves the handlers of recursive continuations as
   they are, and rewrites no handler inside them:
   FlambdaRecHandlersSpec.rewrite_rec_handlers takes a pass there. The
   search for the untaggings of a parameter of an enclosing handler,
   and their removal, do go into them, and stop at their binders as at
   any other. The bodies of code are not gone into:
   FlambdaNormaliseSpec.rewrite_code takes a pass there. The pass
   makes no name.

   Why it is an equivalence, and of which machine. In a
   machine that enters a handler only with values that its
   parameters' subkinds accept, [x] holds a tagged immediate in range
   wherever the handler runs ([untagging_of_stated_statement]: what
   the test accepts at such a binder, the untagging has a value on).
   The [Let] at the head then has a value, and changes nothing but
   [u]. No use of [u] was free in the handler, so the only uses that
   read the [u] of the head are those that were in the scope of a
   [Let] that is taken away; and there [u] holds what that [Let] gave
   it, as neither [x] nor [u] is bound again between the head and it.
   A call's result enters the handler of the continuation that the
   call returns to as an application does, so the test is made there
   too.

   In a machine that makes no such test the statement is false. Under
   an environment in which the first call returns a block and the
   second call raises or does not end, the term runs to the second
   call and the result is undefined at the head of the handler; and
   the result does not refine the term. No second call is needed:
   where the handler untags [x] only in the handler of a continuation
   that it never applies, and the call returns null, the term exits
   and the result is undefined at the head. So the statement stands
   or falls with the test of FlambdaSemanticsSpec.apply_cont_value,
   and it says nothing of a machine that has no such test.

   What it is not.
   - It does not decide as Simplify decides, and it is not Simplify's
     wrapper: it moves an untagging that the handler has, and makes no
     parameter. Where a call returns to a continuation that is applied
     elsewhere too, Simplify gives the continuation a parameter for
     the number and each call a wrapper; this pass does not make that
     shape.
   - It does nothing for a parameter of code or of a recursive
     continuation, whatever it states.
   - It reads what the handler's own metadata state
     ([hm_params_subkinds]): a parameter of which nothing is stated is
     left, whatever is known of the values that it is given.
   - It is intended for both terms; nothing is said here of which
     pairs of terms a validation that applies it accepts.

   Renaming and scope: the functions compare names by equality only
   and make none, and "first" is by the structure of the term. The
   statements that the pass commutes with renaming and keeps scope are
   in FlambdaUntagAtBinderNamesSpec.v. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaUntagParamsSpec FlambdaSubkindSpec FlambdaSubkindOrderSpec.
Import ListNotations.

(* ======================================================================== *)
(* The parameters that state a tagged immediate                             *)
(* ======================================================================== *)

(* What a parameter states is at least "not null, a tagged immediate":
   it may not be null, and its subkind is a tagged immediate or a
   variant with no constructor that is a block
   (FlambdaSubkindOrderSpec.param_says_at_least). *)
Definition states_tagged (ps : param_subkind) : bool :=
  param_says_at_least ps (false, Subkind_tagged_immediate).

(* The parameters of a handler that state it, in their order. What a
   parameter states is the entry at its position; a parameter beyond
   the end of the list states nothing. *)
Fixpoint tagged_params (params : list variable) (sks : list param_subkind)
    : list variable :=
  match params, sks with
  | p :: params, ps :: sks =>
    if states_tagged ps then p :: tagged_params params sks
    else tagged_params params sks
  | _, _ => []
  end.

(* ======================================================================== *)
(* The untaggings of a parameter in its handler                             *)
(* ======================================================================== *)

(* A [Let] at this kind of this defining expression untags [x] to a
   naked immediate. At another kind such a [Let] has no step. *)
Definition untags (x : variable) (kind : flambda_kind) (d : named) : bool :=
  kind_eqb kind untagged_kind
  && match untag_of d with
     | Some p => Pos.eqb p x
     | None => false
     end.

(* The variables that the [Let]s of an expression bind to the
   untagging of [x], outside code, in the order of one walk: a [Let]
   before its body, the handler of a continuation before the body of
   its [Let_cont]. Nothing under a binder of [x]. *)
Fixpoint untaggings (x : variable) (e : expr) {struct e} : list variable :=
  match e with
  | Let y kind d body =>
    (if untags x kind d then [y] else [])
    ++ (if Pos.eqb y x then [] else untaggings x body)
  | Let_static _ body => untaggings x body
  | Let_cont_non_recursive _ h body =>
    untaggings_in_handler x h ++ untaggings x body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => untaggings_in_handler x h) handlers
    ++ untaggings x body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with untaggings_in_handler (x : variable) (h : continuation_handler)
    {struct h} : list variable :=
  match h with
  | Handler params _ handler =>
    if among x params then [] else untaggings x handler
  end.

(* The first of them. A [match], and not [hd_error] of Rocq's library:
   the extraction of this file then adds no function to the files
   that are made from that library. *)
Definition first_untagging (x : variable) (e : expr) : option variable :=
  match untaggings x e with
  | [] => None
  | u :: _ => Some u
  end.

(* The expression without its [Let]s of [u] to the untagging of [x],
   outside code. It is left as it is under any other binder of [u] and
   under a binder of [x]: there one of the two no longer holds what it
   held at the head. *)
Fixpoint drop_untagging (x u : variable) (e : expr) {struct e} : expr :=
  match e with
  | Let y kind d body =>
    if Pos.eqb y u
    then if untags x kind d then drop_untagging x u body else e
    else if Pos.eqb y x then e
    else Let y kind d (drop_untagging x u body)
  | Let_static bindings body => Let_static bindings (drop_untagging x u body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (drop_untagging_in_handler x u h)
      (drop_untagging x u body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, drop_untagging_in_handler x u h)) handlers)
      (drop_untagging x u body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with drop_untagging_in_handler (x u : variable) (h : continuation_handler)
    {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    if among x params || among u params then h
    else Handler params metadata (drop_untagging x u handler)
  end.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The body of a handler with the untagging of its parameter [x] bound
   at its head, where the body untags [x] and the variable of its
   first untagging is not a parameter and is not free in the body. *)
Definition untagging_at_head (params : list variable) (x : variable)
    (handler : expr) : expr :=
  match first_untagging x handler with
  | Some u =>
    if negb (among u params) && negb (var_free_b u handler)
    then
      Let u untagged_kind (Prim (Unary Untag_immediate) [Var x])
        (drop_untagging x u handler)
    else handler
  | None => handler
  end.

(* A call of the expression, outside code, returns to [k]. A
   [Let_cont] of the name [k] hides it in its body, and a recursive
   one in its handlers too. *)
Fixpoint returns_to (k : continuation) (e : expr) {struct e} : bool :=
  match e with
  | Let _ _ _ body | Let_static _ body => returns_to k body
  | Let_cont_non_recursive k' h body =>
    returns_to_in_handler k h
    || (negb (Pos.eqb k' k) && returns_to k body)
  | Let_cont_recursive handlers body =>
    negb (among k (map fst handlers))
    && (existsb (fun '(_, h) => returns_to_in_handler k h) handlers
        || returns_to k body)
  | Apply ap => Pos.eqb (ap_continuation ap) k
  | Apply_cont _ | Switch _ _ | Invalid => false
  end
with returns_to_in_handler (k : continuation) (h : continuation_handler)
    {struct h} : bool :=
  match h with
  | Handler _ _ handler => returns_to k handler
  end.

(* The handler of a non-recursive continuation is rewritten where
   [returned_to] says that a call returns to it, it is not an
   exception handler, and its parameters are distinct: the untaggings
   of the parameters that state a tagged immediate are bound at its
   head, that of the first such parameter outermost. The handlers of
   recursive continuations are left as they are. *)
Fixpoint bind_untaggings (e : expr) {struct e} : expr :=
  match e with
  | Let x kind d body => Let x kind d (bind_untaggings body)
  | Let_static bindings body => Let_static bindings (bind_untaggings body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (bind_untaggings_in_handler (returns_to k body) h)
      (bind_untaggings body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (bind_untaggings body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with bind_untaggings_in_handler (returned_to : bool)
    (h : continuation_handler) {struct h} : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (if returned_to && negb (hm_is_exn_handler metadata) && distinct params
       then
         fold_right (untagging_at_head params) (bind_untaggings handler)
           (tagged_params params (hm_params_subkinds metadata))
       else bind_untaggings handler)
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* What the test of a binder accepts, where the binder states at least
   "not null, a tagged immediate", is a value that the untagging has a
   value on. *)
Definition untagging_of_stated_statement : Prop :=
  ∀ (statics : static_env) (ps : param_subkind) (v : value),
    states_tagged ps = true ->
    value_at statics ps v = true ->
    ∃ i : Z,
      eval_unary_primitive statics Untag_immediate v
      = Some (Value_const (Naked_immediate i)).

(* The result behaves as the term does, under all environments.

   Decision D25, at parameters. It rests on the test that
   FlambdaSemanticsSpec.apply_cont_value makes where a handler is
   entered, and it does not hold of a machine that binds a parameter
   to a value that its subkind does not accept: the header has the
   term that shows it. *)
Definition bind_untaggings_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (bind_untaggings e).
