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

(* Normalising a term by rewrites that are proved: the first of them,
   inlining direct calls of known code.

   This file is to be reviewed: the functions, which are extracted and run
   by the validation of middle_end/flambda2/tests/semantics/, and the
   statement.

   The validation compares the term before Simplify with the term after
   (FlambdaCompareSpec.v). They are seldom the same as they stand, so each
   is first brought to a normal form by rewrites whose correctness is a
   theorem; what the comparison then accepts is correct by those theorems
   and the soundness of the comparison. The rewrites applied to the term
   before may be refinements; those applied to the term after must be
   equivalences, as the chain runs from the one to the other.

   [inline_calls] is for the term before. It goes down the term, into the
   handlers of continuations and the bodies of code too, keeping the code
   that the [Let_static]s passed have bound ([known]), and replaces a
   direct call of known code by the code inlined
   (FlambdaApplySpec.inlined_body) where that is sound ([inlinable_b],
   a check that is enough for FlambdaApplySpec.inlinable). That is for
   code that is not tupled. A direct call that gives known tupled code
   one argument, the tuple, is replaced by the code inlined with its
   parameters bound to the fields of the tuple
   (FlambdaTupledSpec.inlined_tupled_call) where that is sound
   ([tupled_inlinable_b], a check that is enough for
   FlambdaTupledSpec.tupled_inlinable). What is known where a
   handler or code is defined holds where it runs, however much later: a
   code id is bound once (FlambdaWellFormedSpec.v). The body of code is
   rewritten with what was known before its group of bindings, so nothing
   is inlined into itself; and what a group makes known is its code as
   rewritten. A call of code whose id is among those given ([kept]) is
   not inlined; one that gives the code more arguments than it has
   parameters may still be split (below). The caller chooses them, and
   they are not trusted: not inlining a call is always sound. They are
   there for the code that Simplify does not inline, recursive
   functions above all: a call of one, inlined, leaves a call of the
   same code to inline in the next round, and so on for as many rounds
   as there are, where Simplify leaves the first call alone.

   A direct call that gives known code more arguments than it has
   parameters is not inlined as it stands. Simplify splits it
   (simplify/simplify_common.ml, [split_direct_over_application]):
   the full call, with the first arguments, returns to a new
   continuation whose handler calls the result with the arguments
   that remain. [inline_calls] does the same
   ([split_over_application], where [over_applied_b] says that it
   may), whether or not the code is among those whose calls are left:
   the full call is then one that a later round may inline, and
   where it is left it is the call that the term after has. The
   machine does this with an over-application already
   (FlambdaPartialApplicationSpec.over_application_split_statement),
   so the term split refines the call
   ([split_over_application_refines_statement]). The continuation
   takes the name of the return continuation of the code and its
   parameter the name of the closure variable of the code: names of
   the term, as the names that an inlined body brings are, so no
   name is made.

   [expr_in_scope] is the check that code is closed. It is a sufficient
   condition, by its soundness (in the Proofs file); nothing is claimed
   where it answers no.

   [inline_tail_conts], for the term before, puts the handler of a
   non-recursive continuation where the continuation is applied, when that
   is at the end of a run of [Let]s that is the body of its [Let_cont]:
   the continuation is then applied exactly once. The parameters become
   [Let]s. It is what inlining a call leaves to be done, the continuation
   of the call being applied at the end of the inlined body.

   [drop_deleted_code], for either term, removes the bindings of deleted
   code, which bind nothing, and a [Let_static] left with no binding.

   [remove_unused_total_lets], for either term, is the part of
   [remove_unused_lets] that is an equivalence: it removes an unused [Let]
   only where the defining expression is a constant or a symbol, which
   always has a value, and that value is of the kind of the [Let]. A
   [Let] has no step where the value is of another kind
   (FlambdaSemanticsSpec.step), so removing one that does not pass the
   test would give a meaning to a term that has none.

   [remove_unused_defined_lets] goes further, also as an equivalence: it
   keeps the variables in scope as it goes, each with the kind of its
   binder (of the [Let], or of the handler for a parameter), and removes
   an unused [Let] whose defining expression is a [Simple] or a closure
   made of constants, symbols and variables in scope, which has a value
   where those variables are bound, and that value is of the kind of the
   [Let]. For a variable that is known from its binder: the machine binds
   a variable only to a value of the kind of its binder.

   [inline_known_calls] is inlining as an equivalence, for either term:
   it inlines a direct call only where the callee is a symbol that a
   [Let_static] passed has bound to a closure of the code called, so that
   the call cannot be undefined for being of something else; and only
   where the arities of the call are those of the code as
   FlambdaSemanticsSpec.call_function asks ([call_agrees_b]), so that it
   cannot be undefined for that either. The kinds of the values of the
   arguments are not asked for: the call tests them, and so does the
   inlined body, where each parameter is bound by a [Let] at its kind.
   For the two to test the same, the code must give as many kinds as it
   has parameters. It goes where [inline_calls] goes, and leaves the
   same calls alone.

   [merge_statics], for either term, makes one group of [Let_static]s that
   follow one another, and puts the code of a group before its symbols.
   Neither changes the static environment that results, so groups that
   differ only in how they are cut up and ordered become the same.

   [collapse_switches], for the term before, replaces a [Switch] whose
   arms are all the same by that arm. It is a refinement only: the
   [Switch] is undefined where its scrutinee has no arm.

   [fold_constants], for either term, replaces a primitive applied to
   constants by its value, when that is a constant, and a [Switch] on a
   constant by its arm. The value of the primitive is what
   FlambdaPrimitiveSpec.eval_prim says, which the semantics uses too. It
   is asked under the empty static environment: on constants a primitive
   does not look at it. Where it visits them, the pass also writes in
   one form two kinds that the machine does not read, the array kind
   of a store into an array and the kind of a length of an array (the
   comment at [fold_named] says why that is done, and why here).

   [remove_unused_lets], also for the term before, removes each [Let] whose
   variable is not free in its body, in bodies and in the handlers of
   non-recursive continuations. [var_free_b] is the test; it may say that
   a variable is free when it is not, never the reverse.

   The proofs are in FlambdaNormaliseProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaApplySpec FlambdaCompareSpec
  FlambdaTupledSpec.
Import ListNotations.

(* ======================================================================== *)
(* Names in scope                                                           *)
(* ======================================================================== *)

Definition simple_in_scope (vs : list variable) (s : simple) : bool :=
  match s with
  | Var x => among x vs
  | Symbol _ | Const _ => true
  end.

Definition simples_in_scope (vs : list variable) (args : list simple) : bool :=
  forallb (simple_in_scope vs) args.

Definition named_in_scope (vs : list variable) (d : named) : bool :=
  match d with
  | Simple s => simple_in_scope vs s
  | Prim _ args => simples_in_scope vs args
  | Set_of_closures _ slots | Closure_of_set _ _ slots =>
    forallb (fun slot => simple_in_scope vs (snd slot)) slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    simple_in_scope vs closure
  end.

Definition static_binding_in_scope (vs : list variable)
    (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => true
  | Bind_symbol _
      (Static_const_set_of_closures _ slots
      | Static_const_closure_of_set _ _ _ slots) =>
    forallb (fun slot => simple_in_scope vs (snd slot)) slots
  | Bind_symbol _ (Static_const_block _ fields) => simples_in_scope vs fields
  | Bind_symbol _ (Static_const_immutable_value_array elements) =>
    simples_in_scope vs elements
  | Bind_symbol _ (Static_const_boxed_number contents) =>
    simple_in_scope vs contents
  | Bind_symbol _ (Static_const_immutable_string _) => true
  end.

Definition apply_cont_in_scope (vs : list variable) (ks : list continuation)
    (ac : apply_cont) : bool :=
  all_among (apply_cont_continuations ac) ks
  && simples_in_scope vs (ac_args ac).

(* Every variable and continuation the expression uses is bound in it or is
   one of those given. *)
Fixpoint expr_in_scope (vs : list variable) (ks : list continuation)
    (e : expr) : bool :=
  match e with
  | Let x _ d body =>
    named_in_scope vs d && expr_in_scope (x :: vs) ks body
  | Let_static bindings body =>
    forallb (static_binding_in_scope vs) bindings
    && expr_in_scope vs ks body
  | Let_cont_non_recursive k h body =>
    handler_in_scope vs ks h && expr_in_scope vs (k :: ks) body
  | Let_cont_recursive handlers body =>
    let ks' := map fst handlers ++ ks in
    forallb (fun '(_, h) => handler_in_scope vs ks' h) handlers
    && expr_in_scope vs ks' body
  | Apply ap =>
    simple_in_scope vs (ap_callee ap)
    && simples_in_scope vs (ap_args ap)
    && among (ap_continuation ap) ks
    && among (ap_exn_continuation ap) ks
    && simples_in_scope vs (exn_extra_simples ap)
  | Apply_cont ac => apply_cont_in_scope vs ks ac
  | Switch scrutinee arms =>
    simple_in_scope vs scrutinee
    && forallb (fun arm => apply_cont_in_scope vs ks (snd arm)) arms
  | Invalid => true
  end
with handler_in_scope (vs : list variable) (ks : list continuation)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler => expr_in_scope (params ++ vs) ks handler
  end.

(* ======================================================================== *)
(* Inlining direct calls of known code                                      *)
(* ======================================================================== *)

(* The renaming of FlambdaApplySpec.inline_renaming_exists_statement. *)
Definition inline_renaming (ret exn k k_exn : continuation) : renaming :=
  Renaming [] [(swap ret k exn, k_exn); (ret, k)].

Definition var_in_simples (x : variable) (args : list simple) : bool :=
  existsb
    (fun s =>
       match s with
       | Var y => Pos.eqb x y
       | Symbol _ | Const _ => false
       end)
    args.

(* Enough for FlambdaApplySpec.inlinable at [inline_renaming]
   (FlambdaNormaliseProofs.inlinable_b_sound); the converse is not
   claimed. It asks more than the predicate: that the two continuations
   of the code differ, and that those of the call differ. *)
Definition inlinable_b (ap : apply_expr) (c : code) : bool :=
  negb (code_is_tupled c)
  && expr_in_scope (code_my_closure c :: code_params c)
    [code_return_continuation c; code_exn_continuation c] (code_body c)
  && negb (Pos.eqb (code_return_continuation c) (code_exn_continuation c))
  && negb (Pos.eqb (ap_continuation ap) (ap_exn_continuation ap))
  && no_exn_extra_args ap
  && negb (var_in_simples (code_my_closure c) (ap_args ap))
  && forallb (fun param => negb (var_in_simples param (ap_args ap)))
       (code_params c)
  && Nat.eqb (length (code_params c)) (length (ap_args ap))
  && Nat.eqb (length (ap_args_arity ap)) (length (code_params_arity c))
  && negb (binds_statics (code_body c)).

(* Enough for FlambdaTupledSpec.tupled_inlinable at [inline_renaming]
   and the one argument of the call
   (FlambdaNormaliseProofs.tupled_inlinable_b_sound); the converse is
   not claimed. It asks more than the predicate: that the two
   continuations of the code differ, and that those of the call
   differ. *)
Definition tupled_inlinable_b (ap : apply_expr) (c : code) : bool :=
  code_is_tupled c
  && expr_in_scope (code_my_closure c :: code_params c)
    [code_return_continuation c; code_exn_continuation c] (code_body c)
  && negb (Pos.eqb (code_return_continuation c) (code_exn_continuation c))
  && negb (Pos.eqb (ap_continuation ap) (ap_exn_continuation ap))
  && no_exn_extra_args ap
  && negb (var_in_simples (code_my_closure c) (ap_args ap))
  && forallb (fun param => negb (var_in_simples param (ap_args ap)))
       (code_params c)
  && Nat.eqb (length (ap_args ap)) 1
  && arity_eqb (ap_args_arity ap) [Singleton Value]
  && arity_eqb (code_params_arity c)
       (arity_of_kinds (repeat Value (length (code_params c))))
  && negb (Nat.eqb (length (code_params c)) 0)
  && negb (binds_statics (code_body c)).

(* A direct call of the code [c] that may be split: the code is not
   tupled; the call gives more arguments than the code has parameters,
   by the arities, and as many values at least as the parameters are
   variables; the return continuation of the code, which the new
   continuation is named after, is not the exception continuation of
   the call; and the closure variable of the code, which the parameter
   of its handler is named after, is not among the arguments that
   remain nor among the extra arguments of the exception continuation.
   Enough for the premises of
   FlambdaPartialApplicationSpec.over_application_split_statement that
   are about the term; those about the run (the callee is a closure of
   the code called, which is bound) hold wherever the call is
   defined. *)
Definition over_applied_b (ap : apply_expr) (c : code) : bool :=
  negb (code_is_tupled c)
  && Nat.ltb (length (code_params_arity c)) (length (ap_args_arity ap))
  && Nat.leb (cardinal_unarized (code_params_arity c)) (length (ap_args ap))
  && negb (Pos.eqb (code_return_continuation c) (ap_exn_continuation ap))
  && negb
       (var_in_simples (code_my_closure c)
          (skipn (cardinal_unarized (code_params_arity c)) (ap_args ap)))
  && negb (var_in_simples (code_my_closure c) (exn_extra_simples ap)).

(* The over-application split: the full call of the code [called],
   direct, with the first arguments, at the two arities of the code,
   returns to a continuation whose handler takes its result, one
   [Value], and calls it, indirectly, with the arguments that remain,
   at the arity that remains and the return arity of the call. Both
   calls have the exception continuation of the call, with its extra
   arguments. The arguments are split where the machine splits them
   (FlambdaSemanticsSpec.shape_of_code_call). *)
Definition split_over_application (called : code_id) (c : code)
    (ap : apply_expr) : expr :=
  let size := cardinal_unarized (code_params_arity c) in
  let params := length (code_params_arity c) in
  let k := code_return_continuation c in
  let p := code_my_closure c in
  Let_cont_non_recursive k
    (Handler [p] (make_handler_metadata [Value])
       (Apply
          (Apply_expr (Var p) Indirect_unknown_arity (skipn size (ap_args ap))
             (ap_continuation ap) (ap_exn_continuation ap)
             (ap_exn_extra_args ap)
             (skipn params (ap_args_arity ap)) (ap_return_arity ap))))
    (Apply
       (Apply_expr (ap_callee ap) (Direct called) (firstn size (ap_args ap))
          k (ap_exn_continuation ap) (ap_exn_extra_args ap)
          (code_params_arity c) (code_result_arity c))).

(* The code that a group of bindings adds to what is known, the last
   bound first, as FlambdaSemanticsSpec.eval_static_bindings adds it to
   the static environment. *)
Fixpoint codes_of (bindings : list static_binding)
    (known : list (code_id * code)) : list (code_id * code) :=
  match bindings with
  | [] => known
  | Bind_code id c :: bindings => codes_of bindings ((id, c) :: known)
  | (Bind_deleted_code _ | Bind_symbol _ _) :: bindings =>
    codes_of bindings known
  end.

Definition inline_apply (kept : list code_id)
    (known : list (code_id * code)) (ap : apply_expr) : expr :=
  match ap_call_kind ap with
  | Direct called =>
    match lookup called known with
    | Some c =>
      if negb (among called kept) && inlinable_b ap c
      then
        inlined_body c
          (inline_renaming (code_return_continuation c)
             (code_exn_continuation c) (ap_continuation ap)
             (ap_exn_continuation ap))
          ap
      else if negb (among called kept) && tupled_inlinable_b ap c
      then
        inlined_tupled_call c
          (inline_renaming (code_return_continuation c)
             (code_exn_continuation c) (ap_continuation ap)
             (ap_exn_continuation ap))
          ap
      else if over_applied_b ap c
      then split_over_application called c ap
      else Apply ap
    | None => Apply ap
    end
  | Indirect_unknown_arity => Apply ap
  end.

Fixpoint inline_calls (kept : list code_id) (known : list (code_id * code))
    (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (inline_calls kept known body)
  | Let_static bindings body =>
    let bindings := map (inline_calls_in_binding kept known) bindings in
    Let_static bindings (inline_calls kept (codes_of bindings known) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (inline_calls_in_handler kept known h)
      (inline_calls kept known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, inline_calls_in_handler kept known h)) handlers)
      (inline_calls kept known body)
  | Apply ap => inline_apply kept known ap
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with inline_calls_in_handler (kept : list code_id)
    (known : list (code_id * code))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (inline_calls kept known handler)
  end
with inline_calls_in_binding (kept : list code_id)
    (known : list (code_id * code))
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (inline_calls_in_code kept known c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with inline_calls_in_code (kept : list code_id)
    (known : list (code_id * code)) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (inline_calls kept known body)
  end.

(* A direct call that passes [over_applied_b] for the code that the
   static environment binds the code id called to is refined by the
   call split. *)
Definition split_over_application_refines_statement : Prop :=
  ∀ (called : code_id) (c : code) (ap : apply_expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record),
    ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    over_applied_b ap c = true ->
    refines (State (Apply ap) vars conts statics hp rd)
      (State (split_over_application called c ap) vars conts statics hp rd).

Definition inline_calls_refines_statement : Prop :=
  ∀ (kept : list code_id) (e : expr),
    expr_refines e (inline_calls kept [] e).

(* ======================================================================== *)
(* Removing unused [Let]s                                                   *)
(* ======================================================================== *)

Definition var_in_simple (x : variable) (s : simple) : bool :=
  match s with
  | Var y => Pos.eqb x y
  | Symbol _ | Const _ => false
  end.

Definition var_in_value_slots (x : variable)
    (slots : list (value_slot * simple)) : bool :=
  existsb (fun slot => var_in_simple x (snd slot)) slots.

Definition var_in_named (x : variable) (d : named) : bool :=
  match d with
  | Simple s => var_in_simple x s
  | Prim _ args => var_in_simples x args
  | Set_of_closures _ slots | Closure_of_set _ _ slots =>
    var_in_value_slots x slots
  | Project_value_slot _ closure | Project_value_slot_from _ _ closure
  | Project_function_slot _ _ closure =>
    var_in_simple x closure
  end.

Definition var_in_static_binding (x : variable) (binding : static_binding)
    : bool :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => false
  | Bind_symbol _
      (Static_const_set_of_closures _ slots
      | Static_const_closure_of_set _ _ _ slots) =>
    var_in_value_slots x slots
  | Bind_symbol _ (Static_const_block _ fields) => var_in_simples x fields
  | Bind_symbol _ (Static_const_immutable_value_array elements) =>
    var_in_simples x elements
  | Bind_symbol _ (Static_const_boxed_number contents) =>
    var_in_simple x contents
  | Bind_symbol _ (Static_const_immutable_string _) => false
  end.

(* Whether a variable may be free in an expression. *)
Fixpoint var_free_b (x : variable) (e : expr) : bool :=
  match e with
  | Let y _ d body =>
    var_in_named x d || (negb (Pos.eqb x y) && var_free_b x body)
  | Let_static bindings body =>
    existsb (var_in_static_binding x) bindings || var_free_b x body
  | Let_cont_non_recursive _ h body =>
    var_free_in_handler_b x h || var_free_b x body
  | Let_cont_recursive handlers body =>
    existsb (fun '(_, h) => var_free_in_handler_b x h) handlers
    || var_free_b x body
  | Apply ap =>
    var_in_simple x (ap_callee ap) || var_in_simples x (ap_args ap)
    || var_in_simples x (exn_extra_simples ap)
  | Apply_cont ac => var_in_simples x (ac_args ac)
  | Switch scrutinee arms =>
    var_in_simple x scrutinee
    || existsb (fun arm => var_in_simples x (ac_args (snd arm))) arms
  | Invalid => false
  end
with var_free_in_handler_b (x : variable) (h : continuation_handler) : bool :=
  match h with
  | Handler params _ handler =>
    negb (among x params) && var_free_b x handler
  end.

(* Which primitives the normalisation may fold, and remove when unused:
   all but [Opaque_identity], which is meant to hide a value from the
   simplifier and which Simplify leaves alone, and the primitives of the
   heap, which make an object with an identity or write to one: the
   machine runs them, an exit is compared with its heap, and so no
   allocation or store of a cell of the heap is among what may be
   removed. The primitives of regions are left out too: the machine of
   FlambdaSemanticsSpec.v gives none of them a value. *)
Definition prim_may_fold (p : prim) : bool :=
  match p with
  | Unary (Opaque_identity _) | Heap _ | Region_primitive _ => false
  | Unary _ | Binary _ | Variadic _ => true
  end.

Definition named_may_go (d : named) : bool :=
  match d with
  | Prim p _ => prim_may_fold p
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    true
  end.

Fixpoint remove_unused_lets (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body' := remove_unused_lets body in
    if var_free_b x body' || negb (named_may_go d)
    then Let x kind d body'
    else body'
  | Let_static bindings body => Let_static bindings (remove_unused_lets body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (remove_unused_lets_in_handler h)
      (remove_unused_lets body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_unused_lets body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with remove_unused_lets_in_handler (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (remove_unused_lets handler)
  end.

Definition remove_unused_lets_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (remove_unused_lets e).

(* ======================================================================== *)
(* A continuation applied once, at the end of its scope                     *)
(* ======================================================================== *)

(* [body] with the handler in place of the application of [k] that ends
   it, if it is a run of [Let]s ending so, none of which binds a variable
   of the handler. The application must have no trap action: the handler
   in its place would not perform it (Simplify does not inline a
   continuation that is applied with one). *)
Fixpoint inline_cont_at_tail (k : continuation) (params : list variable)
    (metadata : handler_metadata) (handler : expr) (body : expr)
    : option expr :=
  match body with
  | Let x kind d body =>
    if var_free_in_handler_b x (Handler params metadata handler)
    then None
    else
      option_map (Let x kind d)
        (inline_cont_at_tail k params metadata handler body)
  | Let_static bindings body =>
    option_map (Let_static bindings)
      (inline_cont_at_tail k params metadata handler body)
  | Apply_cont ac =>
    if Pos.eqb (ac_continuation ac) k
       && no_trap_action ac
       && Nat.eqb (length params) (length (ac_args ac))
       && forallb (fun param => negb (var_in_simples param (ac_args ac)))
            params
    then
      Some
        (let_params params (hm_params_kinds metadata) (ac_args ac) handler)
    else None
  | Let_cont_non_recursive _ _ _ | Let_cont_recursive _ _ | Apply _
  | Switch _ _ | Invalid =>
    None
  end.

Fixpoint inline_tail_conts (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (inline_tail_conts body)
  | Let_static bindings body => Let_static bindings (inline_tail_conts body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    let handler' := inline_tail_conts handler in
    let body' := inline_tail_conts body in
    match inline_cont_at_tail k params metadata handler' body' with
    | Some inlined => inlined
    | None =>
      Let_cont_non_recursive k (Handler params metadata handler') body'
    end
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (inline_tail_conts body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition inline_tail_conts_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (inline_tail_conts e).

(* ======================================================================== *)
(* Bindings of deleted code                                                 *)
(* ======================================================================== *)

Definition binds_something (binding : static_binding) : bool :=
  match binding with
  | Bind_deleted_code _ => false
  | Bind_code _ _ | Bind_symbol _ _ => true
  end.

Definition let_static_if_any (bindings : list static_binding) (body : expr)
    : expr :=
  match bindings with
  | [] => body
  | _ :: _ => Let_static bindings body
  end.

Fixpoint drop_deleted_code (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (drop_deleted_code body)
  | Let_static bindings body =>
    let_static_if_any (filter binds_something bindings)
      (drop_deleted_code body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (drop_deleted_code handler))
      (drop_deleted_code body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (drop_deleted_code body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition drop_deleted_code_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (drop_deleted_code e).

(* ======================================================================== *)
(* Folding constants                                                        *)
(* ======================================================================== *)

Fixpoint consts_of_simples (args : list simple) : option (list const) :=
  match args with
  | [] => Some []
  | Const c :: args => option_map (cons c) (consts_of_simples args)
  | (Var _ | Symbol _) :: _ => None
  end.

(* Two cases of [fold_named] fold no constant. They write in one form
   a kind that an access to an array carries and that the machine does
   not read.

   The array kind of a store, an [Array_set], becomes the one that its
   set kind determines (FlambdaPrimitiveSpec.array_kind_of_set_kind);
   the set kind is kept. Simplify changes the array kind of a store
   where the type of the array says that its elements are immediates,
   and keeps the set kind ([Simplify_common.specialise_array_kind],
   called by [Simplify_ternary_primitive.simplify_array_set]): the
   store of the term after then carries another array kind than that
   of the term before.

   The kind of a length, an [Array_length], becomes that of an array
   of values. The front end can take the length of one array at two
   kinds, one for [Array.length] and another for the test of the
   bounds of an access, and Simplify makes the second of two lengths
   of one array the first whatever their kinds
   ([Simplify_unary_primitive.simplify_array_length] does not look at
   the kind); the sharing of primitives
   (FlambdaCseSpec.share_known_primitives) compares the primitives as
   they are written.

   Both are equivalences: a length and a store do the same at every
   array kind (FlambdaHeapSpec.array_kind_of_access_statement). The
   array kind of a load is left as it is.

   They are here because [fold_constants] is a pass that the
   validation applies to both terms, in the rounds of each and in the
   tidying that follows the rounds (FlambdaValidateSpec.v). Where
   [fold_named] visits a store or a length it writes that kind in one
   form; a store or a length that no application of the pass visits
   keeps the kind it has. *)
(* CR mshinwell: with these two cases, two terms that differ in the
   array kind of a store, or in the kind of a length, can be found
   the same where the pass has visited those: a verdict of the
   validation does not say that these kinds agree in the two terms
   that it was given. The machine reads neither. The
   compiler chooses the code of a store into an array of values by
   the set kind alone ([To_cmm_primitive.array_set0]: the store of an
   integer for the set kinds of immediates and of values that the
   collector may ignore, and for the set kind of values the store
   that this set kind's own argument asks, an assignment or an
   initialisation), and the set kind is kept here and compared, with
   what it carries; and it compiles a length in the same way at the
   four kinds that the model has ([To_cmm_primitive.array_length],
   which is given the array kind, or [Values] where the kind is
   [Float_array_opt_dynamic]). The other answer would be that the
   validation tells these kinds apart: a unit in which Simplify has
   changed the array kind of a store can then stay not validated for
   that difference. Decision D56. *)
Definition fold_named (d : named) : named :=
  match d with
  | Prim (Heap (Array_set _ set_kind)) args =>
    Prim (Heap (Array_set (array_kind_of_set_kind set_kind) set_kind)) args
  | Prim (Unary (Array_length _)) args =>
    Prim (Unary (Array_length (Array_kind Array_values))) args
  | Prim p args =>
    match (if prim_may_fold p then consts_of_simples args else None) with
    | Some cs =>
      match eval_prim empty_static_env p (map Value_const cs) with
      | Some (Value_const c) => Simple (Const c)
      | _ => d
      end
    | None => d
    end
  | Simple _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    d
  end.

Definition fold_switch (scrutinee : simple) (arms : list (Z * apply_cont))
    : expr :=
  match scrutinee with
  | Const (Naked_immediate i) =>
    match find_arm i arms with
    | Some ac => Apply_cont ac
    | None => Switch scrutinee arms
    end
  | _ => Switch scrutinee arms
  end.

Fixpoint fold_constants (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind (fold_named d) (fold_constants body)
  | Let_static bindings body => Let_static bindings (fold_constants body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k (Handler params metadata (fold_constants handler))
      (fold_constants body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (fold_constants body)
  | Switch scrutinee arms => fold_switch scrutinee arms
  | Apply _ | Apply_cont _ | Invalid => e
  end.

Definition fold_constants_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (fold_constants e).

(* ======================================================================== *)
(* Removing unused [Let]s whose defining expression always has a value     *)
(* ======================================================================== *)

(* The defining expression always has a value, and that value is of the
   kind given: a [Let] of it at that kind always has a step. *)
Definition named_is_total (kind : flambda_kind) (d : named) : bool :=
  match d with
  | Simple (Const c) => kind_eqb (kind_of_const c) kind
  | Simple (Symbol _) => kind_eqb Value kind
  | Simple (Var _) | Prim _ _ | Set_of_closures _ _ | Closure_of_set _ _ _
  | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    false
  end.

Fixpoint remove_unused_total_lets (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    let body' := remove_unused_total_lets body in
    if named_is_total kind d && negb (var_free_b x body')
    then body'
    else Let x kind d body'
  | Let_static bindings body =>
    Let_static bindings (remove_unused_total_lets body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (remove_unused_total_lets handler))
      (remove_unused_total_lets body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_unused_total_lets body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition remove_unused_total_lets_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_total_lets e).

(* ======================================================================== *)
(* Removing unused [Let]s whose defining expression has a value in scope   *)
(* ======================================================================== *)

(* The variables in scope, each with the kind of its binder; the binder
   passed last comes first, and is the one that counts. *)
Definition kinds_in_scope : Type := list (variable * flambda_kind).

(* A renaming acts on the variables; the kinds have no names in them.
   The variables of [kinds] are [map fst kinds]. *)
Definition rename_kinds_in_scope (r : renaming) (kinds : kinds_in_scope)
    : kinds_in_scope :=
  map (fun entry => (rename_variable r (fst entry), snd entry)) kinds.

(* The kind of the value of a [Simple], wherever the variables given are
   bound to values of their kinds: of a constant its own, of a symbol
   [Value], of a variable the kind given. *)
Definition kind_of_simple_in_scope (kinds : kinds_in_scope) (s : simple)
    : option flambda_kind :=
  match s with
  | Var x => lookup x kinds
  | Symbol _ => Some Value
  | Const c => Some (kind_of_const c)
  end.

(* The defining expression has a value of the kind given wherever the
   variables given are bound to values of their kinds. A closure is of
   kind [Value]. *)
Definition named_defined (kinds : kinds_in_scope) (kind : flambda_kind)
    (d : named) : bool :=
  match d with
  | Simple s =>
    match kind_of_simple_in_scope kinds s with
    | Some found => kind_eqb found kind
    | None => false
    end
  | Set_of_closures _ slots =>
    kind_eqb Value kind
    && forallb (fun slot => simple_in_scope (map fst kinds) (snd slot)) slots
  | Closure_of_set slot decls slots =>
    kind_eqb Value kind
    && is_bound slot decls
    && forallb (fun slot => simple_in_scope (map fst kinds) (snd slot)) slots
  | Prim _ _ | Project_value_slot _ _ | Project_value_slot_from _ _ _
  | Project_function_slot _ _ _ =>
    false
  end.

(* The parameters of a handler added to the variables in scope, each
   with the kind that the handler gives it, in the order in which
   FlambdaSemanticsSpec.bind_params binds them. Where the handler does
   not give as many kinds as it has parameters it never runs. *)
Fixpoint params_in_scope (params : list variable)
    (params_kinds : list flambda_kind) (kinds : kinds_in_scope)
    : kinds_in_scope :=
  match params, params_kinds with
  | param :: params, kind :: params_kinds =>
    params_in_scope params params_kinds ((param, kind) :: kinds)
  | _, _ => kinds
  end.

Fixpoint remove_unused_defined_lets (kinds : kinds_in_scope) (e : expr)
    : expr :=
  match e with
  | Let x kind d body =>
    let body' := remove_unused_defined_lets ((x, kind) :: kinds) body in
    if named_defined kinds kind d && negb (var_free_b x body')
    then body'
    else Let x kind d body'
  | Let_static bindings body =>
    Let_static bindings (remove_unused_defined_lets kinds body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata
         (remove_unused_defined_lets
            (params_in_scope params (hm_params_kinds metadata) kinds)
            handler))
      (remove_unused_defined_lets kinds body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (remove_unused_defined_lets kinds body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition remove_unused_defined_lets_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (remove_unused_defined_lets [] e).

(* ======================================================================== *)
(* Inlining calls of symbols known to be closures                           *)
(* ======================================================================== *)

(* What a group of bindings adds to what is known of the symbols: the code
   of those bound to closures, and that the others are not known to be
   closures, the last bound first. *)
Fixpoint symbols_of (bindings : list static_binding)
    (known : list (symbol * option code_id))
    : list (symbol * option code_id) :=
  match bindings with
  | [] => known
  | Bind_symbol s (Static_const_set_of_closures code _) :: bindings =>
    symbols_of bindings ((s, Some code) :: known)
  | Bind_symbol s (Static_const_closure_of_set slot _ function_decls _)
    :: bindings =>
    symbols_of bindings ((s, lookup slot function_decls) :: known)
  | Bind_symbol s
      (Static_const_block _ _ | Static_const_boxed_number _
      | Static_const_immutable_string _
      | Static_const_immutable_value_array _)
    :: bindings =>
    symbols_of bindings ((s, None) :: known)
  | (Bind_code _ _ | Bind_deleted_code _) :: bindings =>
    symbols_of bindings known
  end.

(* Whether a call agrees with code [c] in what
   FlambdaSemanticsSpec.call_function asks of the terms alone: its
   arities are those of the code. What it asks of the values of the
   arguments, that they be of the kinds of the arity, the inlined body
   asks too, provided the code gives one kind for each parameter. *)
Definition call_agrees_b (ap : apply_expr) (c : code) : bool :=
  kinds_per_parameter_eqb (unarize_per_parameter (ap_args_arity ap))
    (unarize_per_parameter (code_params_arity c))
  && arity_eqb (ap_return_arity ap) (code_result_arity c)
  && Nat.eqb (length (code_params c)) (length (code_params_kinds c)).

Definition inline_known_apply (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (ap : apply_expr)
    : expr :=
  match ap_callee ap, ap_call_kind ap with
  | Symbol s, Direct called =>
    match lookup s known_symbols, lookup called known_code with
    | Some (Some code), Some c =>
      if negb (among called kept)
         && (Pos.eqb code called && inlinable_b ap c && call_agrees_b ap c)
      then
        inlined_body c
          (inline_renaming (code_return_continuation c)
             (code_exn_continuation c) (ap_continuation ap)
             (ap_exn_continuation ap))
          ap
      else Apply ap
    | _, _ => Apply ap
    end
  | _, _ => Apply ap
  end.

Fixpoint inline_known_calls (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (e : expr) : expr :=
  match e with
  | Let x kind d body =>
    Let x kind d (inline_known_calls kept known_code known_symbols body)
  | Let_static bindings body =>
    let bindings :=
      map (inline_known_calls_in_binding kept known_code known_symbols) bindings
    in
    Let_static bindings
      (inline_known_calls kept (codes_of bindings known_code)
         (symbols_of bindings known_symbols) body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k
      (inline_known_calls_in_handler kept known_code known_symbols h)
      (inline_known_calls kept known_code known_symbols body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map
         (fun '(k, h) =>
            (k, inline_known_calls_in_handler kept known_code known_symbols h))
         handlers)
      (inline_known_calls kept known_code known_symbols body)
  | Apply ap => inline_known_apply kept known_code known_symbols ap
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with inline_known_calls_in_handler (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata
      (inline_known_calls kept known_code known_symbols handler)
  end
with inline_known_calls_in_binding (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id))
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c =>
    Bind_code id (inline_known_calls_in_code kept known_code known_symbols c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with inline_known_calls_in_code (kept : list code_id)
    (known_code : list (code_id * code))
    (known_symbols : list (symbol * option code_id)) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata
      (inline_known_calls kept known_code known_symbols body)
  end.

Definition inline_known_calls_equivalent_statement : Prop :=
  ∀ (kept : list code_id) (e : expr),
    expr_equivalent e (inline_known_calls kept [] [] e).

(* ======================================================================== *)
(* Groups of static bindings                                                *)
(* ======================================================================== *)

Definition binds_code (binding : static_binding) : bool :=
  match binding with
  | Bind_code _ _ | Bind_deleted_code _ => true
  | Bind_symbol _ _ => false
  end.

(* The bindings of code, then those of symbols, each in their order. *)
Definition code_first (bindings : list static_binding) : list static_binding :=
  filter binds_code bindings
  ++ filter (fun binding => negb (binds_code binding)) bindings.

Fixpoint merge_statics (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (merge_statics body)
  | Let_static bindings body =>
    match merge_statics body with
    | Let_static bindings' body' =>
      Let_static (code_first (bindings ++ bindings')) body'
    | body' => Let_static (code_first bindings) body'
    end
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k (Handler params metadata (merge_statics handler))
      (merge_statics body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (merge_statics body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition merge_statics_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (merge_statics e).

(* ======================================================================== *)
(* Switches whose arms are all the same                                     *)
(* ======================================================================== *)

Definition simple_eq_dec : ∀ s s' : simple, {s = s'} + {s <> s'}.
Proof. decide equality; try apply Pos.eq_dec; apply const_eq_dec. Defined.

Definition trap_action_eq_dec :
  ∀ action action' : trap_action, {action = action'} + {action <> action'}.
Proof. repeat decide equality. Defined.

Definition apply_cont_eq_dec :
  ∀ ac ac' : apply_cont, {ac = ac'} + {ac <> ac'}.
Proof.
  decide equality;
    [ decide equality; apply trap_action_eq_dec
    | apply (list_eq_dec simple_eq_dec)
    | apply Pos.eq_dec ].
Defined.

Definition collapse_switch (scrutinee : simple)
    (arms : list (Z * apply_cont)) : expr :=
  match arms with
  | (_, ac) :: others =>
    if forallb
         (fun arm => if apply_cont_eq_dec (snd arm) ac then true else false)
         others
    then Apply_cont ac
    else Switch scrutinee arms
  | [] => Switch scrutinee arms
  end.

Fixpoint collapse_switches (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (collapse_switches body)
  | Let_static bindings body => Let_static bindings (collapse_switches body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    Let_cont_non_recursive k
      (Handler params metadata (collapse_switches handler))
      (collapse_switches body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (collapse_switches body)
  | Switch scrutinee arms => collapse_switch scrutinee arms
  | Apply _ | Apply_cont _ | Invalid => e
  end.

Definition collapse_switches_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (collapse_switches e).

(* ======================================================================== *)
(* Rewriting the bodies of code                                             *)
(* ======================================================================== *)

(* The rewrites above that need to know nothing of where they are applied
   do not go into the bodies of code. [rewrite_code] applies one to the
   body of every piece of code a term binds, wherever the binding is. The
   body is left alone if the rewrite would change whether it binds
   something static, as the body of code may bind nothing static
   (FlambdaSemanticsSpec.binds_statics). *)
Definition rewrite_code_body (f : expr -> expr) (body : expr) : expr :=
  let body' := f body in
  if Bool.eqb (binds_statics body) (binds_statics body') then body'
  else body.

Definition rewrite_code_in_binding (f : expr -> expr)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id (Code params my_closure ret exn metadata body) =>
    Bind_code id
      (Code params my_closure ret exn metadata (rewrite_code_body f body))
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

Fixpoint rewrite_code (f : expr -> expr) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (rewrite_code f body)
  | Let_static bindings body =>
    Let_static (map (rewrite_code_in_binding f) bindings)
      (rewrite_code f body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (rewrite_code_in_handler f h)
      (rewrite_code f body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, rewrite_code_in_handler f h)) handlers)
      (rewrite_code f body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with rewrite_code_in_handler (f : expr -> expr) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (rewrite_code f handler)
  end.

(* A rewrite that refines, or is an equivalence, under all environments
   is one when applied to the bodies of code. *)
Definition rewrite_code_refines_statement : Prop :=
  ∀ f : expr -> expr,
    (∀ e : expr, expr_refines e (f e)) ->
    ∀ e : expr, expr_refines e (rewrite_code f e).

Definition rewrite_code_equivalent_statement : Prop :=
  ∀ f : expr -> expr,
    (∀ e : expr, expr_equivalent e (f e)) ->
    ∀ e : expr, expr_equivalent e (rewrite_code f e).

(* ======================================================================== *)
(* A switch whose arm goes to a switch on the same scrutinee                *)
(* ======================================================================== *)

(* Where the body of a [Let_cont] is a [Switch], and the handler takes no
   parameters and is a [Switch] on the same scrutinee, an arm of the body
   that applies the continuation may go at once where the arm of the
   handler for the same value goes: the handler runs under the variables
   of the [Let_cont], so the scrutinee has the same value in it. The arm
   of the handler must not name the continuation being bound, which in
   the handler is another one: neither as the continuation that it
   applies nor in its trap action. An arm of the body that has a trap
   action stays as it is. The [Let_cont] goes if no arm names its
   continuation any more. Simplify does as much from what an arm tells
   it of the scrutinee. The handler must give no kinds of parameters
   either: one that has no parameters and gives a kind never runs
   (FlambdaSemanticsSpec.apply_cont_value), and an arm that applies it
   is undefined. *)
(* CR mshinwell: an arm of the handler whose trap action names [k] is
   left where it is. Until this was repaired only the continuation that
   the arm applies was compared with [k], so an arm of the handler that
   installed or removed an outer exception handler of the name [k] was
   taken into the body, where [k] is the continuation being bound: its
   trap action was captured. The machine of FlambdaSemanticsSpec.v has
   no stack of handlers and does not read a trap action, so
   [thread_switches_equivalent_statement] held of the rewrite as it
   was; it says nothing of the machine of FlambdaTrapSpec.v, which has
   one. Simplify does not meet the case: the names it binds are
   fresh. *)
Definition thread_arm (k : continuation) (inner : list (Z * apply_cont))
    (arm : Z * apply_cont) : Z * apply_cont :=
  let (i, ac) := arm in
  match ac_args ac with
  | [] =>
    if Pos.eqb (ac_continuation ac) k && no_trap_action ac
    then
      match find_arm i inner with
      | Some ac' =>
        if among k (apply_cont_continuations ac') then arm else (i, ac')
      | None => arm
      end
    else arm
  | _ :: _ => arm
  end.

Definition arms_apply (k : continuation) (arms : list (Z * apply_cont))
    : bool :=
  existsb (fun arm => among k (apply_cont_continuations (snd arm))) arms.

Definition thread_switch (k : continuation) (h : continuation_handler)
    (body : expr) : expr :=
  match h, body with
  | Handler [] metadata (Switch inner_scrutinee inner),
    Switch scrutinee arms =>
    match hm_params_kinds metadata with
    | [] =>
      if simple_eq_dec scrutinee inner_scrutinee
      then
        let arms := map (thread_arm k inner) arms in
        if arms_apply k arms
        then Let_cont_non_recursive k h (Switch scrutinee arms)
        else Switch scrutinee arms
      else Let_cont_non_recursive k h body
    | _ :: _ => Let_cont_non_recursive k h body
    end
  | _, _ => Let_cont_non_recursive k h body
  end.

Fixpoint thread_switches (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (thread_switches body)
  | Let_static bindings body => Let_static bindings (thread_switches body)
  | Let_cont_non_recursive k (Handler params metadata handler) body =>
    thread_switch k (Handler params metadata (thread_switches handler))
      (thread_switches body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive handlers (thread_switches body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end.

Definition thread_switches_equivalent_statement : Prop :=
  ∀ e : expr, expr_equivalent e (thread_switches e).
