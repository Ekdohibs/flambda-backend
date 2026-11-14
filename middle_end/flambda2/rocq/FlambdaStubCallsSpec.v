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

(* In the code of a stub, the direct calls of the unit's other code are
   made indirect, as Simplify makes them. A normalisation that refines.

   This file is to be reviewed: [demote_stub_calls] and what it is made
   of, against what Simplify does (below), and the statements, the last
   three of which are computed: the rewrite on a small unit, with the
   code of its wrapper a stub and not, and a call that the rewrite
   gives a step that it had not. [demote_stub_calls] is extracted and
   run by the validation of
   middle_end/flambda2/tests/semantics/, on the term before.

   What Simplify does. Code says whether it is a stub
   (FlambdaModel.code_is_stub, the compiler's [Code.stub]). The
   function that a partial application of a known function becomes is
   one (from_lambda/closure_conversion.ml, [wrap_partial_application]):
   its closure keeps the function in a value slot, and its code
   projects the slot from [my_closure] and calls what it gets, by a
   direct call of the code of the function. Unless
   [-flambda2-simplify-stubs] is given, and by default it is not
   (driver/oxcaml_flags.ml), Simplify does not simplify the code of a
   stub where it simplifies the set of closures that the code is of,
   as it does other code (simplify/simplify_set_of_closures.ml,
   [simplify_function]). It simplifies it once, where the code is
   bound (simplify/simplify_static_const.ml, by
   [Simplify_set_of_closures.simplify_static_stub_function]). There
   [my_closure] has an unknown type
   (simplify/simplify_set_of_closures.ml, [dacc_inside_function], the
   case of no function slot), and nothing is inlined
   ([DE.enter_stub_function]). So, in the code of a stub (discrepancy
   C3-11):
   - the projection of a value slot from [my_closure] stays as it is:
     Simplify does not know there what the slot holds;
   - a direct call whose callee is such a projection has a callee
     whose type is not available, and where it names code of the unit
     that is not a stub it becomes an indirect call of known arity
     (simplify/simplify_apply_expr.ml,
     [simplify_function_call_where_callee's_type_unavailable], the
     case [Direct code_id]). The code that a unit binds, its stubs
     excepted, is never simplified as it stands: each set of closures
     is given a new version of it, and the old is not to be kept alive
     by a call that names it ([DA.code_ids_never_simplified]; the
     reason is at its definition, in
     simplify/simplify_static_const.ml). A direct call that names the
     code of another stub, or code that the unit does not bind, keeps
     its code id. So does one that names code which says that it is a
     newer version of other code ([Code.newer_version_of]): Simplify
     leaves that code out of the set too. The code that the front end
     makes does not say it (from_lambda/closure_conversion.ml).

   What is here. [demote_stub_calls e] makes indirect, in the code of
   each stub that the unit [e] binds, every direct call that names
   code which [e] binds and which is not a stub ([non_stub_code]). The
   [Indirect] of the model is both sorts of indirect call of the
   compiler (FlambdaSemanticsSpec.v, the comment at
   [shape_of_closure_call]). Every such call of that code is made
   indirect, whatever its callee. Simplify does it where the type of
   the callee is not available. Where the type says which closure the
   callee is, Simplify keeps the call direct, and the validation makes
   direct again the call that this has made indirect: for a variable
   that a [Let] of the stub binds to a [Set_of_closures] whose value
   slots do not hold the variable
   (../types/rocq/FlambdaAliasesSpec.follow_aliases, by
   [typing_env_under_let]), and for a symbol that the unit binds,
   outside code, to a closure
   (FlambdaKnownSymbolsSpec.fold_known_unit). [follow_aliases] does
   not do it for a variable that a [Let] binds to a [Closure_of_set],
   however many functions the set has, and a unit that has such a
   call in the code of a stub may not be validated. The model has not
   got what code is a newer version of, so that code is not left out
   of [non_stub_code]: in a unit written by hand that has it, the
   rewrite makes indirect a call that Simplify leaves direct, and the
   unit may not be validated. Everything else is left: code that is
   not a stub, and what is outside code.

   It refines, under all environments, and rests on no invariant
   ([demote_stub_calls_refines_statement]): a direct call is the
   indirect call of the same callee with the same arguments, made
   undefined where the callee is not a closure of the code named
   (FlambdaSemanticsSpec.shape_of_closure_call). So a direct call that
   has a step has that step once made indirect
   ([demote_call_step_statement]). It is not an equivalence: a direct
   call that names other code than that of its callee is undefined,
   and the indirect call runs ([demote_call_defines_statement]).
   Nothing is believed of the flag: the rewrite refines whatever code
   the flag is set on.

   This file does not say that the other rewrites of the validation
   leave the code of a stub as Simplify does; each says so of itself.
   FlambdaKnownSlotsSpec.read_slots reads no value slot there.

   The proofs are in FlambdaStubCallsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* The rewrite                                                              *)
(* ======================================================================== *)

(* The id of the code that a static binding binds, if it is not a
   stub. *)
Definition binding_non_stub_code (binding : static_binding) : list code_id :=
  match binding with
  | Bind_code id c => if code_is_stub c then [] else [id]
  | Bind_deleted_code _ | Bind_symbol _ _ => []
  end.

(* The ids of the code that an expression binds and that is not a stub,
   in handlers too. The body of code binds nothing static
   (FlambdaSemanticsSpec.call_code), and is not looked into. *)
Fixpoint non_stub_code (e : expr) : list code_id :=
  match e with
  | Let _ _ _ body => non_stub_code body
  | Let_static bindings body =>
    flat_map binding_non_stub_code bindings ++ non_stub_code body
  | Let_cont_non_recursive _ h body =>
    handler_non_stub_code h ++ non_stub_code body
  | Let_cont_recursive handlers body =>
    flat_map (fun '(_, h) => handler_non_stub_code h) handlers
    ++ non_stub_code body
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => []
  end
with handler_non_stub_code (h : continuation_handler) : list code_id :=
  match h with
  | Handler _ _ handler => non_stub_code handler
  end.

(* A direct call that names one of the code ids given is made
   indirect; any other call is left. *)
Definition demote_call (ids : list code_id) (ap : apply_expr) : apply_expr :=
  match ap_call_kind ap with
  | Direct code =>
    if among code ids
    then
      Apply_expr (ap_callee ap) Indirect (ap_args ap) (ap_continuation ap)
        (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
        (ap_return_arity ap)
    else ap
  | Indirect => ap
  end.

(* Every call of an expression, in handlers too. Static bindings are
   left: the body of code binds nothing static
   (FlambdaSemanticsSpec.call_code). *)
Fixpoint demote_calls (ids : list code_id) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (demote_calls ids body)
  | Let_static bindings body => Let_static bindings (demote_calls ids body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (demote_calls_in_handler ids h)
      (demote_calls ids body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, demote_calls_in_handler ids h)) handlers)
      (demote_calls ids body)
  | Apply ap => Apply (demote_call ids ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with demote_calls_in_handler (ids : list code_id) (h : continuation_handler)
    : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (demote_calls ids handler)
  end.

Definition demote_calls_in_code (ids : list code_id) (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (demote_calls ids body)
  end.

(* The same in the code of a stub that a static binding binds. Code
   that is not a stub is left. *)
Definition demote_in_stub_binding (ids : list code_id)
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c =>
    if code_is_stub c then Bind_code id (demote_calls_in_code ids c)
    else binding
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* The same in the code of each stub that an expression binds, in
   handlers too. Everything that is not the code of a stub is left. *)
Fixpoint demote_in_stubs (ids : list code_id) (e : expr) : expr :=
  match e with
  | Let x kind d body => Let x kind d (demote_in_stubs ids body)
  | Let_static bindings body =>
    Let_static (map (demote_in_stub_binding ids) bindings)
      (demote_in_stubs ids body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (demote_in_stubs_of_handler ids h)
      (demote_in_stubs ids body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, demote_in_stubs_of_handler ids h)) handlers)
      (demote_in_stubs ids body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with demote_in_stubs_of_handler (ids : list code_id)
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (demote_in_stubs ids handler)
  end.

(* In the code of each stub of a unit, the direct calls that name code
   which the unit binds and which is not a stub are made indirect. *)
Definition demote_stub_calls (e : expr) : expr :=
  demote_in_stubs (non_stub_code e) e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

(* A call that has a step has the same step once demoted, whatever the
   code ids given and the environments. *)
Definition demote_call_step_statement : Prop :=
  ∀ (ids : list code_id) (ap : apply_expr) (vars : env) (conts : cont_env)
    (statics : static_env) (hp : heap) (rd : read_record),
    step (State (Apply ap) vars conts statics hp rd) <> Undefined ->
    step (State (Apply (demote_call ids ap)) vars conts statics hp rd)
    = step (State (Apply ap) vars conts statics hp rd).

(* An expression with its calls demoted refines the expression, under
   all environments, whatever the code ids given. *)
Definition demote_calls_refines_statement : Prop :=
  ∀ (ids : list code_id) (e : expr), expr_refines e (demote_calls ids e).

(* So does an expression with the calls of the code of its stubs
   demoted. *)
Definition demote_in_stubs_refines_statement : Prop :=
  ∀ (ids : list code_id) (e : expr), expr_refines e (demote_in_stubs ids e).

Definition demote_stub_calls_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (demote_stub_calls e).

(* The rewrite commutes with renaming, which renames no code id and
   keeps what code is annotated with. *)
Definition demote_stub_calls_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    demote_stub_calls (rename_expr r e) = rename_expr r (demote_stub_calls e).

(* ======================================================================== *)
(* A unit, and a call                                                       *)
(* ======================================================================== *)

(* The rewrite computed on a small unit, and a call that it gives a
   step. The unit is what a partial application of a function of two
   parameters to one constant becomes: the code [function]; the code
   [wrapper], a stub, of a closure that keeps the function in a value
   slot, projects it from [my_closure] and calls it with the constant
   and its own parameter; and the two closures, static. *)
Section Examples.

Let x : variable := 1%positive.
Let y : variable := 2%positive.
Let my_closure : variable := 3%positive.
Let callee : variable := 4%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let function : code_id := 1%positive.
Let wrapper : code_id := 2%positive.
Let function_symbol : symbol := 1%positive.
Let wrapper_symbol : symbol := 2%positive.
Let function_slot_kept : value_slot := 1%positive.

Definition stub_example_function : code :=
  Code [x; y] my_closure ret exn
    (make_code_metadata [Singleton Value; Singleton Value] [Singleton Value])
    (Apply_cont (Apply_cont_expr ret [Var y] None)).

(* The call of the wrapper, of the kind given. *)
Definition stub_example_call (kind : call_kind) : apply_expr :=
  Apply_expr (Var callee) kind [Const (Tagged_immediate 1); Var x] ret exn []
    [Singleton Value; Singleton Value] [Singleton Value].

(* The code of the wrapper, with the metadata and the kind of call
   given. *)
Definition stub_example_wrapper (metadata : code_metadata) (kind : call_kind)
    : code :=
  Code [x] my_closure ret exn metadata
    (Let callee Value (Project_value_slot function_slot_kept (Var my_closure))
       (Apply (stub_example_call kind))).

Definition stub_example_metadata : code_metadata :=
  make_code_metadata [Singleton Value] [Singleton Value].

Definition stub_example (metadata : code_metadata) (kind : call_kind) : expr :=
  Let_static
    [Bind_code function stub_example_function;
     Bind_code wrapper (stub_example_wrapper metadata kind);
     Bind_symbol function_symbol (Static_const_set_of_closures function []);
     Bind_symbol wrapper_symbol
       (Static_const_set_of_closures wrapper
          [(function_slot_kept, Symbol function_symbol)])]
    (Apply_cont (Apply_cont_expr done [Symbol wrapper_symbol] None)).

(* Where the wrapper is a stub, its direct call of the code of the
   function is made indirect. *)
Definition demote_stub_calls_demotes_statement : Prop :=
  demote_stub_calls
    (stub_example (stub_code_metadata stub_example_metadata) (Direct function))
  = stub_example (stub_code_metadata stub_example_metadata) Indirect.

(* Where the wrapper is not marked as a stub, nothing changes. *)
Definition demote_stub_calls_not_a_stub_statement : Prop :=
  demote_stub_calls (stub_example stub_example_metadata (Direct function))
  = stub_example stub_example_metadata (Direct function).

(* A direct call that names the code of the wrapper, of a closure of the
   code of the function: the call has no step, and has one once made
   indirect. So the rewrite is no equivalence. *)
Definition demote_call_defines_statement : Prop :=
  let vars :=
    [(callee, Value_closure function []);
     (x, Value_const (Tagged_immediate 0))]
  in
  let conts := [(ret, Outer ret); (exn, Outer exn)] in
  let statics :=
    {| static_values := [];
       static_code := [(function, stub_example_function)] |}
  in
  step
    (State (Apply (stub_example_call (Direct wrapper))) vars conts statics
       empty_heap no_reads)
  = Undefined
  ∧ step
      (State
         (Apply (demote_call [wrapper] (stub_example_call (Direct wrapper))))
         vars conts statics empty_heap no_reads)
    <> Undefined.

End Examples.
