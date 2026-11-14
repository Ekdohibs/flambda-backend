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


(* Tupled functions: what a call of code that the compiler marks as
   tupled comes to.

   This file is to be reviewed: the statements, against
   FlambdaSemanticsSpec.tuple_fields, [detuple], [arguments_for_code]
   and [shape_of_closure_call], which they restate, and against what
   the compiler does with a tupled function
   (simplify/simplify_apply_expr.ml,
   [simplify_direct_tuple_application]; at run time [caml_tuplifyN],
   backend/cmm_helpers.ml, [tuplify_function]).

   A function of the source that takes a tuple, [fun (x, y) -> e], is
   compiled to code with a parameter for each component, marked as
   tupled ([Code_metadata.is_tupled]). A caller that does not know the
   code gives the closure one argument, the tuple, and the entry point
   of the closure takes the tuple apart; where Simplify knows the code,
   it makes the projections itself and calls the code with the
   components. In the model every call of the fragment gives tupled
   code the tuple, and the machine takes it apart, with the projections
   that Simplify makes.

   - The fields that the machine reads of a tuple of a given size are
     those of an immutable block of tag 0 that has exactly that many
     fields, all of kind [Value]: the tuple is that block, or a symbol
     bound to it ([tuple_is_block_statement]).
   - A call that gives tupled code one argument, a tuple with a field
     for each parameter, is the full call of the code with the fields
     ([apply_tupled_statement]).
   - A call that gives it more arguments is that full call, at the
     result arity of the code, returning to what is left of the
     application: an over-application, whose first argument is the
     tuple ([apply_tupled_over_statement]).
   - A call of tupled code is undefined where it has no argument, where
     the first parameter of its arity is not one [Value], or where its
     first argument is not such a tuple
     ([apply_tupled_undefined_statement]); and it is never a partial
     application ([tupled_never_partial_statement]).
   - Three units run as they should: tupled code given a tuple that is
     a symbol, given one that a primitive makes, and given a tuple and
     one more argument. They are well-formed, and pass the check of
     kinds ([examples_run_statement],
     [examples_well_formed_statement]). Given a triple, or a block of
     another tag, in place of the pair, the call is undefined
     ([wrong_tuple_undefined_statement]).

   The statements about calls suppose that the parameters of the code
   are single values, as many as it has variables for them: so are
   those of the compiler's tupled code, each the component of a tuple.

   - A direct call that gives tupled code a tuple is refined by the body
     of the code with its parameters bound to the fields of the tuple,
     each by the projection that Simplify makes
     ([inline_tupled_statement]), and the two are equivalent where the
     callee is a closure of that code, the call returns at the result
     arity of the code and the value of the tuple is no location
     ([inline_tupled_equivalent_statement]).
     FlambdaApplySpec.inlinable excludes tupled code, as its bindings
     are of the parameters to the arguments; these are the bindings for
     tupled code. The validation inlines such a call
     (FlambdaNormaliseSpec.inline_apply, by [inlined_tupled_call]).

   What is not here: the direct call of tupled code as the compiler has
   it, which gives the code the components (the review comment at
   FlambdaSemanticsSpec.shape_of_closure_call), and so the statement
   that such a call is the call with the tuple; and the inlining of a
   call that gives tupled code a tuple and more arguments.

   The proofs are in FlambdaTupledProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaPrimitiveModel FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaNominalSpec FlambdaWellFormedSpec FlambdaWellKindedSpec.
Import ListNotations.

(* ======================================================================== *)
(* Tuples                                                                   *)
(* ======================================================================== *)

(* The fields of a tuple of [size] components, for a size that is not
   zero: the value is read as a block of tag 0 with exactly [size]
   fields, each of kind [Value], and they are its fields. (A value read
   as a block is a block or a symbol bound to one:
   FlambdaValueSpec.block_of.) *)
Definition tuple_is_block_statement : Prop :=
  ∀ (statics : static_env) (size : nat) (tuple : value)
    (fields : list value),
    size <> 0 ->
    tuple_fields statics size tuple = Some fields
    <-> block_of statics tuple = Some (0%Z, fields)
        ∧ length fields = size
        ∧ Forall (fun v => kind_of_value v = Value) fields.

(* ======================================================================== *)
(* Calls of tupled code                                                     *)
(* ======================================================================== *)

(* The parameters of the code are single values, one for each of its
   variables. *)
Definition params_are_values (c : code) : Prop :=
  code_params_arity c = arity_of_kinds (repeat Value (length (code_params c))).

(* A call that gives a closure of tupled code one argument, at an arity
   of one [Value], where the argument is a tuple with a field for each
   parameter: it is the full call of the code with the fields, at the
   arity of its parameters. *)
Definition apply_tupled_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee tuple : value) (fields extra : list value) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some [tuple] ->
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    closure_of statics callee = Some (called, slots) ->
    ap_call_kind ap = Indirect ∨ ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = true ->
    params_are_values c ->
    code_params c <> [] ->
    ap_args_arity ap = [Singleton Value] ->
    tuple_fields statics (length (code_params c)) tuple = Some fields ->
    step (State (Apply ap) vars conts statics hp rd)
    = call_code callee c fields (code_params_arity c) (ap_return_arity ap)
        (resolve (ap_continuation ap) conts)
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd.

(* A call that gives it more arguments, of the kinds of its arity and
   at arities that pass the compiler's test (which looks at the arity
   of the call with the tuple taken apart: as many values as the code
   has parameters, then the arity of the other arguments): it is that
   full call, at the result arity of the code, and returns to what is
   left of the application, which keeps the other arguments. *)
Definition apply_tupled_over_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee tuple : value) (fields rest extra : list value)
    (rest_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some (tuple :: rest) ->
    eval_simples vars (exn_extra_simples ap) = Some extra ->
    closure_of statics callee = Some (called, slots) ->
    ap_call_kind ap = Indirect ∨ ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = true ->
    params_are_values c ->
    code_params c <> [] ->
    ap_args_arity ap = Singleton Value :: rest_arity ->
    rest_arity <> [] ->
    map kind_of_value rest = unarize rest_arity ->
    partial_or_over_application (code_params_arity c ++ rest_arity)
      (ap_return_arity ap) (code_params_arity c) (code_result_arity c)
    = true ->
    tuple_fields statics (length (code_params c)) tuple = Some fields ->
    step (State (Apply ap) vars conts statics hp rd)
    = call_code callee c fields (code_params_arity c) (code_result_arity c)
        (Over_application rest rest_arity (ap_return_arity ap)
           (resolve (ap_continuation ap) conts)
           (with_extra_args (resolve (ap_exn_continuation ap) conts) extra))
        (with_extra_args (resolve (ap_exn_continuation ap) conts) extra)
        statics hp rd.

(* A call of a closure of tupled code is undefined where it has no
   argument; where its arity has no parameter, or the first is not one
   [Value]; and where its first argument is not a tuple with a field
   for each parameter of the code. *)
Definition apply_tupled_undefined_statement : Prop :=
  ∀ (ap : apply_expr) (vars : env) (conts : cont_env) (statics : static_env)
    (hp : heap) (rd : read_record)
    (callee : value) (vs : list value) (called : code_id)
    (slots : list (value_slot * value)) (c : code),
    eval_simple vars (ap_callee ap) = Some callee ->
    eval_simples vars (ap_args ap) = Some vs ->
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = true ->
    vs = []
    ∨ (∀ (first : arity_component) (rest_arity : arity),
        ap_args_arity ap = first :: rest_arity ->
        unarize_component first <> [Value])
    ∨ (∃ (tuple : value) (rest : list value),
        vs = tuple :: rest
        ∧ tuple_fields statics (cardinal_unarized (code_params_arity c))
            tuple
          = None) ->
    step (State (Apply ap) vars conts statics hp rd) = Undefined.

(* No call of a closure of tupled code is a partial application. *)
Definition tupled_never_partial_statement : Prop :=
  ∀ (statics : static_env) (closure : value) (kind : call_kind)
    (vs : list value) (args_arity return_arity : arity) (called : code_id)
    (slots : list (value_slot * value)) (c : code) (partial : value),
    closure_of statics closure = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    code_is_tupled c = true ->
    shape_of_closure_call statics closure kind vs args_arity return_arity
    <> Some (Partial_call partial).

(* ======================================================================== *)
(* Three examples                                                           *)
(* ======================================================================== *)

(* [first (x, y)] is [x], tupled code of two parameters, and
   [identity x] is [x]; both are bound statically, with the pair
   [(1, 2)] and the pair [(identity, 0)]. *)
Section Examples.

Let x : variable := 1%positive.
Let y : variable := 2%positive.
Let my_closure : variable := 3%positive.
Let pair : variable := 4%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 1%positive.
Let error : continuation := 2%positive.

Definition a_value : arity := [Singleton Value].

Definition tupled_first_code : code :=
  Code [x; y] my_closure ret exn
    (make_tupled_code_metadata [Singleton Value; Singleton Value] a_value)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

Definition untupled_identity_code : code :=
  Code [x] my_closure ret exn (make_code_metadata a_value a_value)
    (Apply_cont (Apply_cont_expr ret [Var x] None)).

Definition tupled_first_code_id : code_id := 1%positive.
Definition untupled_identity_code_id : code_id := 2%positive.
Definition tupled_first_symbol : symbol := 1%positive.
Definition untupled_identity_symbol : symbol := 2%positive.
Definition pair_symbol : symbol := 3%positive.
Definition pair_of_identity_symbol : symbol := 4%positive.

Definition tupled_bindings : list static_binding :=
  [Bind_code tupled_first_code_id tupled_first_code;
   Bind_code untupled_identity_code_id untupled_identity_code;
   Bind_symbol tupled_first_symbol
     (Static_const_set_of_closures tupled_first_code_id []);
   Bind_symbol untupled_identity_symbol
     (Static_const_set_of_closures untupled_identity_code_id []);
   Bind_symbol pair_symbol
     (Static_const_block 0
        [Const (Tagged_immediate 1); Const (Tagged_immediate 2)]);
   Bind_symbol pair_of_identity_symbol
     (Static_const_block 0
        [Symbol untupled_identity_symbol; Const (Tagged_immediate 0)])].

(* [first pair], where the pair is a symbol. *)
Definition tupled_static_example : expr :=
  Let_static tupled_bindings
    (Apply
       (Apply_expr (Symbol tupled_first_symbol) Indirect [Symbol pair_symbol]
          done error [] a_value a_value)).

(* [first (1, 2)], where a primitive makes the pair, and the call names
   the code. *)
Definition tupled_made_example : expr :=
  Let_static tupled_bindings
    (Let pair Value
       (Prim (Variadic (Make_block (Block_kind_values 0)))
          [Const (Tagged_immediate 1); Const (Tagged_immediate 2)])
       (Apply
          (Apply_expr (Symbol tupled_first_symbol)
             (Direct tupled_first_code_id) [Var pair] done error [] a_value
             a_value))).

(* [first (identity, 0) 1]: [first] is given the pair, and its result,
   [identity], the argument that remains. *)
Definition tupled_over_example : expr :=
  Let_static tupled_bindings
    (Apply
       (Apply_expr (Symbol tupled_first_symbol) Indirect
          [Symbol pair_of_identity_symbol; Const (Tagged_immediate 1)] done
          error [] [Singleton Value; Singleton Value] a_value)).

(* Run on their own, the three return 1. *)
Definition examples_run_statement : Prop :=
  ∀ e : expr,
    e ∈ [tupled_static_example; tupled_made_example; tupled_over_example] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_const (Tagged_immediate 1)] empty_heap).

(* They are well-formed units, and pass the check of kinds. *)
Definition examples_well_formed_statement : Prop :=
  ∀ e : expr,
    e ∈ [tupled_static_example; tupled_made_example; tupled_over_example] ->
    well_formed_unit (Unit_interface done error) e = true
    ∧ well_kinded_unit [] done error e = true.

(* [first] given a triple, and given a block of tag 1 with two fields:
   both calls are undefined. *)
Definition tupled_wrong_tuple (tag : Z) (fields : list simple) : expr :=
  Let_static tupled_bindings
    (Let pair Value
       (Prim (Variadic (Make_block (Block_kind_values tag))) fields)
       (Apply
          (Apply_expr (Symbol tupled_first_symbol) Indirect [Var pair] done
             error [] a_value a_value))).

Definition wrong_tuple_undefined_statement : Prop :=
  ∀ e : expr,
    e
    ∈ [tupled_wrong_tuple 0
         [Const (Tagged_immediate 1); Const (Tagged_immediate 2);
          Const (Tagged_immediate 3)];
       tupled_wrong_tuple 1
         [Const (Tagged_immediate 1); Const (Tagged_immediate 2)]] ->
    behaves (State e [] [] empty_static_env empty_heap no_reads) Goes_wrong.

End Examples.


(* ======================================================================== *)
(* Inlining a call of tupled code                                           *)
(* ======================================================================== *)

(* The parameters bound to the fields of a tuple, one [Let] each, around
   a body: the projections that Simplify makes of a call that gives
   tupled code a tuple ([Simplify_common.project_tuple]), with the
   parameters of the code for the variables that it binds. *)
Fixpoint let_tuple_fields (params : list variable) (size field : nat)
    (tuple : simple) (body : expr) : expr :=
  match params with
  | [] => body
  | param :: params =>
    Let param Value
      (Prim
         (Unary (Block_load (tuple_access size) Immutable (Z.of_nat field)))
         [tuple])
      (let_tuple_fields params size (S field) tuple body)
  end.

(* A call that gives the tupled code [c] the tuple [tuple], inlined: [r]
   renames the continuations of the code to those of the call. *)
Definition inlined_tupled_body (c : code) (r : renaming) (ap : apply_expr)
    (tuple : simple) : expr :=
  Let (code_my_closure c) Value (Simple (ap_callee ap))
    (let_tuple_fields (code_params c) (length (code_params c)) 0 tuple
       (rename_expr r (code_body c))).

(* The same for a call as it stands, whose one argument is the tuple. A
   call of another number of arguments is left as it is. *)
Definition inlined_tupled_call (c : code) (r : renaming) (ap : apply_expr)
    : expr :=
  match ap_args ap with
  | [tuple] => inlined_tupled_body c r ap tuple
  | _ => Apply ap
  end.

(* What makes that sound. The conditions are those of
   FlambdaApplySpec.inlinable, but for the arguments: the call gives one
   argument, the tuple, at the arity of one [Value]; the code is tupled,
   and its parameters are single values, at least one. *)
Definition tupled_inlinable (ap : apply_expr) (c : code) (r : renaming)
    (tuple : simple) : Prop :=
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
  (* the exception continuation of the call takes the exception alone *)
  ∧ ap_exn_extra_args ap = []
  (* the call gives the tuple and nothing else *)
  ∧ ap_args ap = [tuple]
  ∧ ap_args_arity ap = [Singleton Value]
  (* the bindings do not capture the tuple *)
  ∧ ~ var_free_in_simple (code_my_closure c) tuple
  ∧ (∀ param : variable,
      param ∈ code_params c -> ~ var_free_in_simple param tuple)
  (* the code *)
  ∧ code_is_tupled c = true
  ∧ params_are_values c
  ∧ code_params c <> []
  ∧ binds_statics (code_body c) = false.

(* Where the code id of a direct call is bound to tupled code [c], the
   call refines [c] inlined with its parameters bound to the fields of
   the tuple. Nothing is asked of the tuple: where it is not one of as
   many fields as the code has parameters, the call is undefined. *)
Definition inline_tupled_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (tuple : simple) (vars : env) (conts : cont_env) (statics : static_env)
      (hp : heap) (rd : read_record),
    ap_call_kind ap = Direct called ->
    lookup called (static_code statics) = Some c ->
    tupled_inlinable ap c r tuple ->
    refines (State (Apply ap) vars conts statics hp rd)
      (State (inlined_tupled_body c r ap tuple) vars conts statics hp rd).

(* Where the callee is a closure of the code called, the call returns
   at the result arity of the code, and the value of the tuple is no
   location, the two are equivalent, which is what a rewrite of the
   term after Simplify needs. If the tuple has the fields, the two meet
   as for the refinement. If it has not, the call is undefined and so
   is one of the projections. If the argument has no value both go
   wrong.
   Of a location the two need not be equivalent, hence the last
   premise. The code has a parameter ([tupled_inlinable]), so the call
   reads the tuple, and is undefined: FlambdaSemanticsSpec.tuple_fields
   reads no cell. A projection reads the cell that the heap binds the
   location to (FlambdaHeapSpec.exec_unary_on_cell). Where that is a
   block made [Immutable_unique], of tag 0, with as many fields as the
   code has parameters, each of kind [Value], the projections have
   their values and the inlined body comes to the body of the code,
   which the call does not. (Where the location is bound to nothing,
   or one of the projections has no value on its cell, the inlined
   body goes wrong, as the call does.) The refinement above asks
   nothing of the tuple: an undefined call is refined by anything. *)
Definition inline_tupled_equivalent_statement : Prop :=
  ∀ (ap : apply_expr) (called : code_id) (c : code) (r : renaming)
    (tuple : simple) (callee : value) (slots : list (value_slot * value))
    (vars : env) (conts : cont_env) (statics : static_env) (hp : heap)
      (rd : read_record),
    ap_call_kind ap = Direct called ->
    eval_simple vars (ap_callee ap) = Some callee ->
    closure_of statics callee = Some (called, slots) ->
    lookup called (static_code statics) = Some c ->
    tupled_inlinable ap c r tuple ->
    ap_return_arity ap = code_result_arity c ->
    (∀ l : location, eval_simple vars tuple <> Some (Value_location l)) ->
    equivalent (State (Apply ap) vars conts statics hp rd)
      (State (inlined_tupled_body c r ap tuple) vars conts statics hp rd).
