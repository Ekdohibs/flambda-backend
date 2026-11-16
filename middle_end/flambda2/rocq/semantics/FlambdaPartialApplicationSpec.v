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

(* Partial applications and over-applications: what a call comes to
   that does not give a function as many arguments as its code has
   parameters.

   This file is to be reviewed: the statements, against
   FlambdaSemanticsSpec.shape_of_closure_call, [shape_of_call] and
   [apply_cont_value], which they restate as steps, and against what the
   compiler does with such calls (simplify/simplify_apply_expr.ml,
   [simplify_direct_partial_application] and
   [simplify_direct_over_application]; at run time, [caml_applyN] and
   [caml_curryN]).

   - A call that gives a closure fewer arguments than its code has
     parameters calls nothing: it gives its return continuation a
     partial application, the value that keeps the closure and the
     arguments ([apply_partial_statement]).
   - An indirect call of a partial application, with one argument at
     least and arguments of the kinds of its arity, is the call of the
     closure that it keeps with the arguments that it keeps followed by
     those of the call ([apply_partial_application_statement]). A
     direct call of one, and a call that gives it no argument, are
     undefined ([apply_partial_application_undefined_statement]). So is
     a call whose own arguments are not of the kinds of its own arity,
     whatever the partial application keeps
     ([apply_partial_application_kinds_statement]): the test is not
     left to the call of the closure, which looks at the arguments kept
     and the new ones together, where a partial application that no
     call made could hide the mismatch
     ([malformed_partial_application_statement] is an example).
   - So a partial application whose result is at once called with more
     arguments refines the one call with all the arguments
     ([partial_then_apply_statement]). It is a refinement and not an
     equivalence because the first call is undefined where the
     compiler's test of arities fails for it, or its arguments are not
     of the kinds of its arity, and the one call need not be.
   - A call that gives a closure more arguments than its code has
     parameters is a full call with the first of them, at the result
     arity of the code, that returns to what is left of the application
     ([apply_over_statement]); and what is left, applied to the one
     value that the full call returns, is an indirect call of that value
     with the arguments that remain
     ([over_application_applied_statement]).
   - So an over-application is refined by the full call followed, in a
     continuation, by the call of its result with the arguments that
     remain ([over_application_split_statement]): the term that Simplify
     makes of an over-application of a function it knows
     ([Simplify_common.split_direct_over_application]), but for the
     regions that it opens around the two calls when the function may
     allocate locally, which the fragment has not got. The two runs are
     not step for step the same: where the machine applies what is left
     of the application, the term applies a handler, which takes one
     step more.
   - These statements are about code that is not tupled. A call of
     tupled code takes its first argument apart before it counts its
     arguments (FlambdaTupledSpec.v), so that it is never a partial
     application, and is an over-application where it has a second
     argument.
   - Two units run as they should, one with a partial application that
     is then applied and one with an over-application; they are
     well-formed, and pass the check of kinds
     ([examples_run_statement], [examples_well_formed_statement]).

   Exceptions. A call evaluates the extra arguments of its exception
   continuation whatever it turns out to be, so each statement about a
   step supposes that they have values. A partial application raises
   nothing. The full call inside an over-application and the call of
   its result raise to the one exception continuation of the
   over-application, with its extra arguments: what is left of the
   application keeps that continuation with the values of the extra
   arguments ([apply_over_statement]), and the term that splits an
   over-application gives the two calls the same exception continuation
   and the same extra arguments. Two units show it, one that raises in
   the full call and one that raises in the call of its result
   ([over_raises_statement]). A call whose extra arguments have no value
   is undefined even where it would make a partial application and not
   use them ([apply_extra_undefined_statement]). The machine does not
   execute trap actions, so no statement here depends on one.

   The proofs are in FlambdaPartialApplicationProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaNominalSpec FlambdaWellFormedSpec
  FlambdaWellKindedSpec.
Import ListNotations.

(* ======================================================================== *)
(* Partial applications                                                     *)
(* ======================================================================== *)

(* A call that gives a closure fewer arguments than its code has
   parameters, of the kinds of its arity and at arities that pass the
   compiler's test (which asks that there is one argument at least, and
   that the call expects one [Value]), runs no code: it applies its
   return continuation to the partial application of the closure to the
   arguments. The extra arguments of its exception continuation have
   values, and are not used. *)
Definition apply_partial_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee : value) (vs extra : list value) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some vs ->
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    closure_of statics callee = Some (called, slots) ->
    ap_call_kind ap = Indirect_unknown_arity
    ∨ ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = false ->
    length (ap_args_arity ap) < length (code_params_arity c) ->
    map kind_of_value vs = unarize (ap_args_arity ap) ->
    partial_or_over_application (ap_args_arity ap) (ap_return_arity ap)
      (code_params_arity c) (code_result_arity c)
    = true ->
    step (State (Apply ap) vars conts statics hp rd)
    = apply_continuation (ap_continuation ap)
        [Value_partial_application callee vs (ap_args_arity ap)] conts
        statics hp rd.

(* An indirect call of a partial application, with one argument at
   least and arguments of the kinds of its arity, is the call of the
   closure that the partial application keeps with the arguments that it
   keeps, followed by those of the call. (The closure kept is not itself
   a partial application: a call makes a partial application of a
   closure only.) *)
Definition apply_partial_application_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (closure : value) (applied vs extra : list value)
    (applied_arity : arity),
    eval_simple vars (ap_callee ap)
    = Some (Value_partial_application closure applied applied_arity) ->
    eval_simples vars (ap_args ap) = Some vs ->
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    ap_call_kind ap = Indirect_unknown_arity ->
    ap_args_arity ap <> [] ->
    map kind_of_value vs = unarize (ap_args_arity ap) ->
    partial_application_of closure = None ->
    step (State (Apply ap) vars conts statics hp rd)
    = call_function closure Indirect_unknown_arity (applied ++ vs)
        (applied_arity ++ ap_args_arity ap) (ap_return_arity ap)
        (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd.

(* A direct call of a partial application is undefined, as it is a
   closure of no code; and so is a call that gives it no argument. *)
Definition apply_partial_application_undefined_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (closure : value) (applied : list value) (applied_arity : arity),
    eval_simple vars (ap_callee ap)
    = Some (Value_partial_application closure applied applied_arity) ->
    ap_call_kind ap <> Indirect_unknown_arity ∨ ap_args_arity ap = [] ->
    step (State (Apply ap) vars conts statics hp rd) = Undefined.

(* A call of a partial application whose own arguments are not of the
   kinds of its own arity is undefined, whatever the partial application
   keeps. (Simplify tests the arguments of every call against the arity
   of the call before it goes by the code of the callee:
   [simplify_apply_shared].) The call of the closure tests the arguments
   kept and the new ones together, against the two arities together:
   that test alone would let a partial application that keeps arguments
   that are not of the arity it keeps, which no call makes, make up for
   a mismatch in the call. *)
Definition apply_partial_application_kinds_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (closure : value) (applied vs : list value) (applied_arity : arity),
    eval_simple vars (ap_callee ap)
    = Some (Value_partial_application closure applied applied_arity) ->
    eval_simples vars (ap_args ap) = Some vs ->
    map kind_of_value vs <> unarize (ap_args_arity ap) ->
    step (State (Apply ap) vars conts statics hp rd) = Undefined.

(* A call of a closure with fewer arguments than its code has
   parameters, whose result is at once called with more arguments,
   refines the one call with all the arguments. The continuation [k]
   that takes the partial application is bound around the first call;
   its parameter [p] is not free in the arguments of the second, nor in
   the extra arguments of its exception continuation. The exception
   continuation of the first call is not used: a partial application
   raises nothing. *)
Definition partial_then_apply_statement : Prop :=
  ∀ (f : simple) (kind : call_kind) (args args' : list simple)
    (k k' exn exn' : continuation)
    (extra extra' : list (simple * flambda_kind)) (p : variable)
    (metadata : handler_metadata)
    (args_arity args_arity' partial_arity return_arity : arity) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (callee : value)
    (called : code_id) (slots : list (value_slot * value)) (c : code),
    eval_simple vars f = Some callee ->
    closure_of statics callee = Some (called, slots) ->
    kind = Indirect_unknown_arity ∨ kind = Direct called ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = false ->
    length args_arity < length (code_params_arity c) ->
    args_arity' <> [] ->
    ~ var_free_in_simples p args' ->
    ~ var_free_in_simples p (map fst extra') ->
    refines
      (State
         (Let_cont_non_recursive k
            (Handler [p] metadata
               (Apply
                  (Apply_expr (Var p) Indirect_unknown_arity args' k' exn'
                     extra' args_arity' return_arity)))
            (Apply
               (Apply_expr f kind args k exn extra args_arity
                  partial_arity)))
         vars conts statics hp rd)
      (State
         (Apply
            (Apply_expr f kind (args ++ args') k' exn' extra'
               (args_arity ++ args_arity') return_arity))
         vars conts statics hp rd).

(* ======================================================================== *)
(* Over-applications                                                        *)
(* ======================================================================== *)

(* A call that gives a closure more arguments than its code has
   parameters, of the kinds of its arity and at arities that pass the
   compiler's test (which asks that the code returns one [Value]), is a
   full call with the first arguments, as many as the code has
   variables for its parameters, made at the result arity of the code.
   It returns to what is left of the application, which keeps the
   arguments that remain and the two continuations of the call; its
   exception continuation is that of the call, with the extra arguments
   of the call. *)
Definition apply_over_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee : value) (vs extra : list value) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some vs ->
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    closure_of statics callee = Some (called, slots) ->
    ap_call_kind ap = Indirect_unknown_arity
    ∨ ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = false ->
    length (code_params_arity c) < length (ap_args_arity ap) ->
    map kind_of_value vs = unarize (ap_args_arity ap) ->
    partial_or_over_application (ap_args_arity ap) (ap_return_arity ap)
      (code_params_arity c) (code_result_arity c)
    = true ->
    step (State (Apply ap) vars conts statics hp rd)
    = call_code callee c
        (firstn (cardinal_unarized (code_params_arity c)) vs)
        (firstn (length (code_params_arity c)) (ap_args_arity ap))
        (code_result_arity c)
        (Over_application
           (skipn (cardinal_unarized (code_params_arity c)) vs)
           (partially_apply (ap_args_arity ap)
              (length (code_params_arity c)))
           (ap_return_arity ap) (resolve (ap_continuation ap) conts)
           (with_extra_args (resolve (ap_exn_continuation ap) conts) extra))
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd.

(* What is left of an over-application, applied to one value, is an
   indirect call of that value with the arguments that remain; applied
   to none or to several, it is undefined. *)
Definition over_application_applied_statement : Prop :=
  ∀ (remaining : list value) (remaining_arity return_arity : arity)
    (return_to exn_to : cont_value) (vs : list value) (statics : static_env)
      (hp : heap) (rd : read_record),
    apply_cont_value
      (Over_application remaining remaining_arity return_arity return_to
         exn_to)
      vs statics hp rd
    = match vs with
      | [v] =>
        call_function v Indirect_unknown_arity remaining remaining_arity
          return_arity return_to exn_to statics hp rd
      | _ => Undefined
      end.

(* A call that gives a closure more arguments than its code has
   parameters is refined by the term that splits it: the full call,
   direct, with the first arguments, at the two arities of the code,
   whose result a continuation [k] takes and calls, indirectly, with
   the arguments that remain, at the arity that remains and the return
   arity of the call. The arguments are split where the arity is: the
   first are as many as the code has variables for its parameters. The
   continuation [k] is bound around the full call and is not the
   exception continuation of the call, which both calls of the term
   have, with its extra arguments; its handler takes one [Value] and is
   not an exception handler; its parameter [p] is not free in the
   arguments that remain, nor in the extra arguments.
   The call may be of any kind: a direct call of other code than the
   closure's is undefined. It is a refinement and not an equivalence:
   the over-application is undefined where the compiler's test of
   arities fails for it, or an argument that remains is not of the kind
   of its arity, even when the full call never returns. (Where the code
   does not return one [Value], the handler that the compiler makes is
   [Invalid]; there the test of arities fails, the over-application is
   undefined, and any term refines it.) *)
Definition over_application_split_statement : Prop :=
  ∀ (f : simple) (kind : call_kind) (args args' : list simple)
    (k k' exn' : continuation) (extra : list (simple * flambda_kind))
    (p : variable)
    (args_arity args_arity' return_arity : arity) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap) (rd : read_record)
      (callee : value)
    (called : code_id) (slots : list (value_slot * value)) (c : code),
    eval_simple vars f = Some callee ->
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = false ->
    length args_arity = length (code_params_arity c) ->
    length args = cardinal_unarized (code_params_arity c) ->
    args_arity' <> [] ->
    k <> exn' ->
    ~ var_free_in_simples p args' ->
    ~ var_free_in_simples p (map fst extra) ->
    refines
      (State
         (Apply
            (Apply_expr f kind (args ++ args') k' exn' extra
               (args_arity ++ args_arity') return_arity))
         vars conts statics hp rd)
      (State
         (Let_cont_non_recursive k
            (Handler [p] (make_handler_metadata [Value])
               (Apply
                  (Apply_expr (Var p) Indirect_unknown_arity args' k' exn' extra
                     args_arity' return_arity)))
            (Apply
               (Apply_expr f (Direct called) args k exn' extra
                  (code_params_arity c) (code_result_arity c))))
         vars conts statics hp rd).

(* ======================================================================== *)
(* Two examples                                                             *)
(* ======================================================================== *)

(* [first x y] is [x], and [identity x] is [x]; both are bound
   statically. *)
Section Examples.

Let x : variable := 1%positive.
Let y : variable := 2%positive.
Let my_closure : variable := 3%positive.
Let partial : variable := 4%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 1%positive.
Let error : continuation := 2%positive.
Let apply_rest : continuation := 3%positive.

Definition one_value : arity := [Singleton Value].

Definition first_code : code :=
  Code [x; y] my_closure ret exn
    (make_code_metadata [Singleton Value; Singleton Value] one_value)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

Definition identity_code : code :=
  Code [x] my_closure ret exn (make_code_metadata one_value one_value)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

Definition first_code_id : code_id := 1%positive.
Definition identity_code_id : code_id := 2%positive.
Definition first_symbol : symbol := 1%positive.
Definition identity_symbol : symbol := 2%positive.

Definition example_bindings : list static_binding :=
  [Bind_code first_code_id first_code;
   Bind_code identity_code_id identity_code;
   Bind_symbol first_symbol (Static_const_set_of_closures first_code_id []);
   Bind_symbol identity_symbol
     (Static_const_set_of_closures identity_code_id [])].

(* [first 1], and then its result applied to [2]. *)
Definition partial_example : expr :=
  Let_static example_bindings
    (Let_cont_non_recursive apply_rest
       (Handler [partial] (make_handler_metadata [Value])
          (Apply
             (Apply_expr (Var partial) Indirect_unknown_arity
                [Const (Tagged_immediate 2)] done error [] one_value
                one_value)))
       (Apply
          (Apply_expr (Symbol first_symbol) Indirect_unknown_arity
             [Const (Tagged_immediate 1)] apply_rest error [] one_value
             one_value))).

(* [identity first 1 2]: [identity] is given three arguments and has one
   parameter, so it is called with [first], and its result with the two
   that remain. *)
Definition over_example : expr :=
  Let_static example_bindings
    (Apply
       (Apply_expr (Symbol identity_symbol) Indirect_unknown_arity
          [Symbol first_symbol; Const (Tagged_immediate 1);
           Const (Tagged_immediate 2)]
          done error []
          [Singleton Value; Singleton Value; Singleton Value] one_value)).

(* Run on their own, both return 1. *)
Definition examples_run_statement : Prop :=
  ∀ e : expr,
    e ∈ [partial_example; over_example] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 1)] empty_heap).

(* Both are well-formed units, and pass the check of kinds. *)
Definition examples_well_formed_statement : Prop :=
  ∀ e : expr,
    e ∈ [partial_example; over_example] ->
    well_formed_unit (Unit_interface done error) e = true
    ∧ well_kinded_unit [] done error e = true.

(* A partial application that no call makes: it keeps one argument, at
   an arity of no parameter. Called with no argument at an arity of one
   [Value], it would give [identity] one [Value] at one parameter, the
   arguments kept and the new ones taken together. The call is
   undefined: its own arguments are not of the kinds of its own
   arity. *)
Definition malformed_partial_application_statement : Prop :=
  step
    (State
       (Apply
          (Apply_expr (Var partial) Indirect_unknown_arity [] done error []
             one_value one_value))
       [(partial,
         Value_partial_application (Value_closure identity_code_id [])
           [Value_const (Tagged_immediate 5)] [])]
       []
       (Static_env [(identity_code_id, identity_code)] []) empty_heap no_reads)
  = Undefined.

(* [raise x] raises [x], and [raiser x] returns [raise]. *)
Definition raise_code : code :=
  Code [x] my_closure ret exn (make_code_metadata one_value one_value)
    (Apply_cont
       (Apply_cont_expr exn [Var x] (Some (Pop exn (Some Regular))))).

Definition raise_code_id : code_id := 3%positive.
Definition raiser_code_id : code_id := 4%positive.
Definition raise_symbol : symbol := 3%positive.
Definition raiser_symbol : symbol := 4%positive.

Definition raiser_code : code :=
  Code [x] my_closure ret exn (make_code_metadata one_value one_value)
    (Apply_cont (Apply_cont_expr ret [Symbol raise_symbol] None)).

Definition raising_bindings : list static_binding :=
  [Bind_code raise_code_id raise_code;
   Bind_code raiser_code_id raiser_code;
   Bind_symbol raise_symbol (Static_const_set_of_closures raise_code_id []);
   Bind_symbol raiser_symbol
     (Static_const_set_of_closures raiser_code_id [])].

(* The callee given two arguments, of which its code takes one; the
   exception continuation of the call has one extra argument, 9. *)
Definition over_raising (callee : symbol) (first second : Z) : expr :=
  Let_static raising_bindings
    (Apply
       (Apply_expr (Symbol callee) Indirect_unknown_arity
          [Const (Tagged_immediate first); Const (Tagged_immediate second)]
          done error
          [(Const (Tagged_immediate 9), Value)]
          [Singleton Value; Singleton Value] one_value)).

(* [raise 7 2] raises 7 in the full call, and [raiser 1 5] raises 5 in
   the call of its result: both leave by the exception continuation,
   with the exception followed by the extra argument. *)
Definition over_raises_statement : Prop :=
  behaves
    (State (over_raising raise_symbol 7 2) [] [] empty_static_env empty_heap
       no_reads)
    (Exits error
       [Value_const (Tagged_immediate 7); Value_const (Tagged_immediate 9)]
         empty_heap)
  ∧ behaves
    (State (over_raising raiser_symbol 1 5) [] [] empty_static_env empty_heap
       no_reads)
      (Exits error
         [Value_const (Tagged_immediate 5);
          Value_const (Tagged_immediate 9)] empty_heap).

End Examples.

(* A call whose extra arguments have no value is undefined, whatever
   the call would be: a call that would make a partial application, and
   so would not use them, too. *)
Definition apply_extra_undefined_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record),
    eval_simples vars (exn_extra_simples ap) = None ->
    step (State (Apply ap) vars conts statics hp rd) = Undefined.
