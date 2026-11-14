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

(* Calls of code that never returns: a normalisation of the term before
   Simplify that is a refinement.

   This file is to be reviewed: [never_returns_b], [invalid_params],
   [cut_return], [returns_to_invalid] and [cut_returns], which the
   validation applies to the term before
   (FlambdaValidateSpec.validate_unit, with no code known at the head
   of the unit), and the statements.

   Simplify gives a call no return continuation where the type of the
   callee's result is bottom ([Never_returns]; simplify_apply_expr.ml).
   The model has no such call. The test tool converts one to a call
   that returns to a continuation bound around it and used nowhere
   else, whose handler is [Invalid] (the trust ledger, "Calls that
   never return"; requests/C5-never-returns.md, form B). The term
   before has, for the same call, the return continuation that the
   front end gave it. So the two terms differ, and the validation
   needs a rewrite of the term before that gives such a call the same
   shape.

   [cut_returns] is that rewrite. It goes down the term with the code
   that the groups of static bindings above have bound, as
   FlambdaNormaliseSpec.inline_calls does. A direct call of such code
   whose body does not mention its return continuation, once the
   bindings of the continuations that nothing uses are removed from
   it, is put under a binding of the call's own return continuation to
   a handler that is [Invalid]:

       apply f (x) -> k * e
     becomes
       apply f (x) -> k * e   where k (r) = invalid

   The tests ([never_returns_b]):
   - the body of the code, once the bindings of the continuations that
     nothing in their scope uses are removed
     (FlambdaUnusedContsSpec.remove_unused_conts, which gives a term
     that runs as the body does), does not mention the return
     continuation of the code (FlambdaUnusedContsSpec.cont_free_b,
     which may say that a continuation is free when it is not, never
     the reverse). Then no run of the body applies it, whatever it is
     bound to: the lemma is [call_code_never_returns_statement];
   - the code is not tupled, and the call gives at least as many
     arguments as the code has parameters, so it is a full call or an
     over-application and not a partial application: a partial
     application is given to the return continuation at once
     (FlambdaSemanticsSpec.run_call). The arguments and the parameters
     are counted as FlambdaSemanticsSpec.shape_of_code_call counts
     them, by the components of the two arities and not unarized. In
     an over-application the full
     call inside returns to what is left of the application, and it is
     that full call which never returns;
   - the return and exception continuations of the call are two
     names: the binding is of the first, and must not capture the
     second.

   Names. The rewrite makes none. The continuation that it binds is
   the call's own, bound again around the call: inside the binding the
   call returns to the new handler, and outside nothing changes. The
   parameters of the handler are names that the code binds, its
   closure variable and then its parameters, as many as the call has
   results ([invalid_params]); where the code has too few the call is
   left as it is. They are bound in a handler that is [Invalid], so
   they are read by nothing. So the rewrite commutes with renaming
   exactly, the code known being renamed with the term
   ([cut_returns_equivariant_statement]), and keeps scope
   (FlambdaNeverReturnsScopeSpec.v). It does not keep the binders of a
   term distinct: the comparison of the validation is up to bound
   names, and its rounds rename binders apart themselves
   (FlambdaRenameApartSpec.v).

   A call that is already under such a binding is left
   ([returns_to_invalid]), so the rewrite applied again changes
   nothing more there, and the form that the tool's conversion makes
   is left as it is.

   What is known is the code of the groups above, not that of the
   group that binds the body being rewritten: a call, in code, of code
   of its own group is left. The test is on the body as the rewrite
   has left it. So code that never returns only because of a call in
   it that never returns passes the test where its code is bound by a
   later group than the code that it calls. Where that call is a tail
   call ([let fail2 () = fail ()]) the binding that the rewrite makes
   around it is of the return continuation of the code. Where it is
   not ([let fail2 () = let r = fail () in r]) the handler that the
   call returned to is still bound, around the new binding; it is what
   mentions the return continuation of the code, and nothing uses it.
   That is why the test is of the body with such bindings removed; the
   rewrite does not remove them from the term (the third example).

   Where it is asked. FlambdaValidateSpec.validate_unit asks the
   rewrite once, where its first comparison has failed: of the term
   before after the demotion of the direct calls in the code of stubs,
   and before the loops are made. The rounds of its normalisation
   (FlambdaValidateSpec.normalise_before, [normalise_lifted]) do not
   ask it again: a call of code that never returns which one of those
   rounds makes, by inlining a call or by making a call direct, is
   left as it is. The reading of known arrays does ask it again: each
   round of FlambdaValidateReadingSpec.reading_round begins as
   [validate_unit] does, so the rewrite is asked there of what the
   round before gave, where that round read an array.

   [cut_returns_refines_statement]: the term refines its result, under
   all environments. It is a rewrite of the term before only: the
   converse is not claimed (it holds of the calls rewritten, but the
   statement that the validation uses is the refinement).

   The proofs are in FlambdaNeverReturnsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaNormaliseSpec
  FlambdaUnusedContsSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* Whether a call, of code [c], may be given a return continuation that
   is undefined. *)
Definition never_returns_b (ap : apply_expr) (c : code) : bool :=
  negb
    (cont_free_b (code_return_continuation c)
       (remove_unused_conts (code_body c)))
  && negb (code_is_tupled c)
  && Nat.leb (length (code_params_arity c)) (length (ap_args_arity ap))
  && negb (Pos.eqb (ap_continuation ap) (ap_exn_continuation ap)).

(* The parameters of the handler: the first [count] of the closure
   variable of the code followed by its parameters, if there are as
   many. *)
Definition invalid_params (c : code) (count : nat) : option (list variable) :=
  let names := code_my_closure c :: code_params c in
  if Nat.leb count (length names) then Some (firstn count names) else None.

Definition cut_return (known : list (code_id * code)) (ap : apply_expr)
    : expr :=
  match ap_call_kind ap with
  | Direct called =>
    match lookup called known with
    | Some c =>
      let kinds := unarize (ap_return_arity ap) in
      match invalid_params c (length kinds) with
      | Some params =>
        if never_returns_b ap c
        then
          Let_cont_non_recursive (ap_continuation ap)
            (Handler params (make_handler_metadata kinds) Invalid) (Apply ap)
        else Apply ap
      | None => Apply ap
      end
    | None => Apply ap
    end
  | Indirect => Apply ap
  end.

(* Whether a binding of [k] is of a handler that is [Invalid] around a
   call that returns to [k]. *)
Definition returns_to_invalid (k : continuation) (h : continuation_handler)
    (body : expr) : bool :=
  match h, body with
  | Handler _ _ Invalid, Apply ap => Pos.eqb (ap_continuation ap) k
  | _, _ => false
  end.

Fixpoint cut_returns (known : list (code_id * code)) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (cut_returns known body)
  | Let_static bindings body =>
    let bindings := map (cut_returns_in_binding known) bindings in
    Let_static bindings (cut_returns (codes_of bindings known) body)
  | Let_cont_non_recursive k h body =>
    if returns_to_invalid k h body then e
    else
      Let_cont_non_recursive k (cut_returns_in_handler known h)
        (cut_returns known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, cut_returns_in_handler known h)) handlers)
      (cut_returns known body)
  | Apply ap => cut_return known ap
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with cut_returns_in_handler (known : list (code_id * code))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (cut_returns known handler)
  end
with cut_returns_in_binding (known : list (code_id * code))
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c => Bind_code id (cut_returns_in_code known c)
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end
with cut_returns_in_code (known : list (code_id * code)) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (cut_returns known body)
  end.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* Two calls of code whose body does not mention its return
   continuation once the bindings of the continuations that nothing
   uses are removed from it, and that differ only in what they return
   to, run alike: both are undefined, or both take a step, to states
   that behave alike. *)
Definition call_code_never_returns_statement : Prop :=
  ∀ (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity)
    (return_to return_to' exn_to : cont_value) (statics : static_env)
    (hp : heap),
    cont_free_b (code_return_continuation c)
      (remove_unused_conts (code_body c))
    = false ->
    match
      call_code closure c vs args_arity return_arity return_to exn_to
        statics hp,
      call_code closure c vs args_arity return_arity return_to' exn_to
        statics hp
    with
    | Step s, Step s' => equivalent s s'
    | Undefined, Undefined => True
    | _, _ => False
    end.

(* The term refines its result, under all environments. *)
Definition cut_returns_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (cut_returns [] e).

(* The rewrite commutes with renaming, the code known being renamed
   with the term. *)
Definition cut_returns_equivariant_statement : Prop :=
  ∀ (r : renaming) (known : list (code_id * code)) (e : expr),
    cut_returns (map_values (rename_code r) known) (rename_expr r e)
    = rename_expr r (cut_returns known e).

(* ======================================================================== *)
(* Three examples                                                           *)
(* ======================================================================== *)

(* Code that raises what it is given, and never returns; the names are
   those of no other example. *)
Definition example_fail : code_id := 1%positive.
Definition example_ret : continuation := 1%positive.
Definition example_exn : continuation := 2%positive.
Definition example_return : continuation := 3%positive.
Definition example_error : continuation := 4%positive.
Definition example_closure : variable := 1%positive.
Definition example_param : variable := 2%positive.

Definition example_fail_code : code :=
  Code [example_param] example_closure example_ret example_exn
    (make_code_metadata [Singleton Value] [Singleton Value])
    (Apply_cont
       (Apply_cont_expr example_exn [Var example_param]
          (Some (Pop example_exn (Some Regular))))).

Definition example_call (callee : variable) : apply_expr :=
  Apply_expr (Var callee) (Direct example_fail)
    [Const (Tagged_immediate 5)] example_return example_error []
    [Singleton Value] [Singleton Value].

(* A call of the code, with the code in sight: it is put under a
   binding of its return continuation to a handler that is [Invalid],
   whose parameter is the closure variable of the code. *)
Definition cut_returns_example_statement : Prop :=
  ∀ callee : variable,
    cut_returns []
      (Let_static [Bind_code example_fail example_fail_code]
         (Apply (example_call callee)))
    = Let_static [Bind_code example_fail example_fail_code]
        (Let_cont_non_recursive example_return
           (Handler [example_closure] (make_handler_metadata [Value])
              Invalid)
           (Apply (example_call callee))).

(* The rewrite applied to its result changes nothing. *)
Definition cut_returns_example_again_statement : Prop :=
  ∀ callee : variable,
    let once :=
      cut_returns []
        (Let_static [Bind_code example_fail example_fail_code]
           (Apply (example_call callee)))
    in
    cut_returns [] once = once.

(* Code that calls [example_fail] and gives back what the call returns:
   [let through f = let r = fail 5 in r]. It never returns, and its
   body mentions its return continuation, in the handler that the call
   returns to. The names declared here are those of no other example;
   [example_return], [example_error] and [example_call] are those of
   the first. *)
Definition example_through : code_id := 2%positive.
Definition example_through_ret : continuation := 5%positive.
Definition example_outer_return : continuation := 6%positive.
Definition example_outer_error : continuation := 7%positive.
Definition example_through_closure : variable := 3%positive.
Definition example_through_param : variable := 4%positive.
Definition example_result : variable := 5%positive.

Definition example_through_code (body : expr) : code :=
  Code [example_through_param] example_through_closure example_through_ret
    example_error
    (make_code_metadata [Singleton Value] [Singleton Value])
    (Let_cont_non_recursive example_return
       (Handler [example_result] (make_handler_metadata [Value])
          (Apply_cont
             (Apply_cont_expr example_through_ret [Var example_result]
                None)))
       body).

Definition example_outer_call (callee : variable) : apply_expr :=
  Apply_expr (Var callee) (Direct example_through)
    [Const (Tagged_immediate 5)] example_outer_return example_outer_error
    [] [Singleton Value] [Singleton Value].

(* A call of that code, with both codes in sight, each bound by a
   group of its own. The call inside the code is put under a binding
   of its return continuation; the handler that it returned to stays
   bound around that, used by nothing. The call of the code is put
   under such a binding too: the test is of the body without the
   handler that nothing uses.

   This is an equation of the rewrite, as the two before it are. The
   rewrite looks at the code that a call names, and not at what is
   called or at what is given: nothing is said of a run of the term.
   A run of this one that reaches the call inside is undefined there,
   as the code is given a constant and calls its parameter. *)
Definition cut_returns_example_through_statement : Prop :=
  ∀ callee : variable,
    cut_returns []
      (Let_static [Bind_code example_fail example_fail_code]
         (Let_static
            [Bind_code example_through
               (example_through_code
                  (Apply (example_call example_through_param)))]
            (Apply (example_outer_call callee))))
    = Let_static [Bind_code example_fail example_fail_code]
        (Let_static
           [Bind_code example_through
              (example_through_code
                 (Let_cont_non_recursive example_return
                    (Handler [example_closure] (make_handler_metadata [Value])
                       Invalid)
                    (Apply (example_call example_through_param))))]
           (Let_cont_non_recursive example_outer_return
              (Handler [example_through_closure]
                 (make_handler_metadata [Value]) Invalid)
              (Apply (example_outer_call callee)))).
