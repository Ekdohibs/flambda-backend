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
   made indirect, as Simplify makes them. A normalisation that refines,
   in two forms.

   [demote_stub_calls] makes them calls of unknown arity. No function
   of the validation applies it now: it is here with its laws and its
   computed examples, as the form that demotes every such call and
   rests on no invariant. [demote_full_stub_calls] makes them calls of
   known arity, which is what Simplify makes. It is extracted and run
   by the validation of middle_end/flambda2/tests/semantics/, on the
   term before.

   This file is to be reviewed: the two rewrites and what they are made
   of, against what Simplify does (below), and the statements. Of the
   second rewrite also: the test of a call ([full_call_of_b]), what is
   known in the code of a stub ([demote_full_in_stubs]), and what its
   laws suppose of the static environment ([statics_know_signatures]).
   The last six statements are computed: the first rewrite on a small
   unit, with the code of its wrapper a stub and not, and a call that
   it gives a step that it had not; the second rewrite by where the
   code that a call names is bound, and by what the code and the call
   are; and a call that would lose its step if a call of known arity
   were made of it without knowing its code.

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

   The first form. [demote_stub_calls e] makes indirect, in the code of
   each stub that the unit [e] binds, every direct call that names
   code which [e] binds and which is not a stub ([non_stub_code]),
   whatever its callee and wherever [e] binds the code. It makes them
   calls of unknown arity, where Simplify makes calls of known arity
   (above). Everything else is left: code that is not a stub, and what
   is outside code.

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

   The second form. A call of known arity has a meaning only where it
   is a full call (FlambdaSemanticsSpec.shape_of_closure_call and the
   comments there): it has one group of arguments at least, and its
   callee is a closure of code that is bound, is not tupled, and has as
   many parameters as the call has groups of arguments. A direct call
   is defined more often: with fewer or more groups than the code has
   parameters it is a partial application or an over-application. So
   [demote_full_stub_calls e] makes a call of known arity that lists no
   code ([Indirect_known_arity None], the compiler's [Unknown]) only of
   a direct call that it knows to be full:
   - the code named is not a stub, as in the first form; it is not
     tupled; and it has as many parameters as the call has groups of
     arguments, of which there is one at least ([full_call_of_b]; the
     arity of the call is counted, by group, and not its arguments);
   - [e] binds that code in the group of bindings of the stub, before
     the stub or after it, or in a group around that group
     ([demote_full_in_stubs], by FlambdaNormaliseSpec.codes_of).
   Merely being bound elsewhere in the unit is not enough. A term is
   run under any static environment. Were it enough that [e] binds the
   code somewhere, in a handler beside the stub or in a group that
   follows, the environment could bind the code id to code of another
   arity when the code of the stub runs; the direct call would then be
   a partial application or an over-application, with a step, and the
   call of known arity would have none
   ([demote_full_call_needs_scope_statement]). Bound in
   the group of the stub or around it, the code id is bound to that
   code whenever the code of the stub runs: no code of a group runs
   before the group is bound, and a code id is bound once
   (FlambdaSemanticsSpec.eval_static_bindings).

   Among the calls that the first form demotes, these are left direct:
   a call of tupled code (the test tool refuses a unit that has one;
   FlambdaSemanticsSpec.v, the comments at [shape_of_closure_call]); a
   call with no group of arguments (the backend stops on a call of
   known arity that has none; the same comments); a call with fewer or
   more groups than the code has parameters (closure conversion makes
   a call direct only where it is full); a call of code that [e] binds
   elsewhere. Where Simplify makes a call of known arity of such a
   call, the unit may not be validated.

   The model has not got what code is a newer version of, so neither
   form leaves that code out, as Simplify does (above): in a unit
   written by hand that has it, the rewrite demotes a call that
   Simplify leaves direct, and the unit may not be validated.

   It refines, under all environments
   ([demote_full_stub_calls_refines_statement]). Unlike the first form
   it rests on an invariant, which the machine provides. The laws of
   its parts ([demote_full_call_step_statement],
   [demote_full_calls_refines_statement],
   [demote_full_in_stubs_refines_statement]) hold where the static
   environment binds each code id that is known to code with the arity
   of parameters of the code known, tupled or not as that code is
   ([statics_know_signatures]). A group of bindings, as it is in the
   term or as the rewrite leaves it, gives that of its own code
   ([group_knows_signatures_statement]), and a step keeps it
   ([statics_know_signatures_step_statement]). The rewrite of a unit
   starts with nothing known, so it supposes nothing. It is not an
   equivalence, as the first form is not, and nothing is believed of
   the flag.

   What the validation makes direct again, and what it does not. The
   second form demotes every call that passes its test, whatever its
   callee. Simplify does it where the type of the callee is not
   available; where the type says which closure the callee is, it
   keeps the call direct. The validation makes direct again a call of
   known arity whose callee is a symbol that the unit binds, outside
   code, to a closure (FlambdaKnownSymbolsSpec.fold_known_unit, by
   [fold_known_apply]). It does not where the callee is a variable:
   FlambdaAliasesSpec.follow_aliases makes direct a call of unknown
   arity of a variable that a [Let] binds to a [Set_of_closures] whose
   value slots do not hold the variable (by [typing_env_under_let]),
   and leaves a call of known arity; for a variable that a [Let] binds
   to a [Closure_of_set] it makes no call direct, of either kind. So a
   unit in which Simplify keeps such a call direct, in the code of the
   stub or in a copy of it where the stub is inlined, has a call of
   known arity before and a direct call after, unless the callee is
   such a symbol by the time [fold_known_unit] comes to the call, and
   may not be validated.

   This file does not say that the other rewrites of the validation
   leave the code of a stub as Simplify does; each says so of itself.
   FlambdaKnownSlotsSpec.read_slots reads no value slot there.

   The proofs are in FlambdaStubCallsProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec FlambdaSemanticsSpec FlambdaNominalSpec
  FlambdaNormaliseSpec.
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
      Apply_expr (ap_callee ap) Indirect_unknown_arity (ap_args ap)
        (ap_continuation ap)
        (ap_exn_continuation ap) (ap_exn_extra_args ap) (ap_args_arity ap)
        (ap_return_arity ap)
    else ap
  | Indirect_unknown_arity | Indirect_known_arity _ => ap
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
(* The rewrite that makes a call of known arity                             *)
(* ======================================================================== *)

(* Whether a direct call of the code [c] may be made a call of known
   arity: the code is not a stub and is not tupled, and the call has one
   group of arguments at least and as many groups as the code has
   parameters. It is the arity of the call that is counted, by group,
   and not its arguments: one group may unarize to no value. *)
Definition full_call_of_b (ap : apply_expr) (c : code) : bool :=
  negb (code_is_stub c)
  && negb (code_is_tupled c)
  && match ap_args_arity ap with [] => false | _ :: _ => true end
  && Nat.eqb (length (ap_args_arity ap)) (length (code_params_arity c)).

(* A direct call that names code which is known, and of which the call
   passes that test, is made a call of known arity that lists no code;
   any other call is left. *)
Definition demote_full_call (known : list (code_id * code))
    (ap : apply_expr) : apply_expr :=
  match ap_call_kind ap with
  | Direct called =>
    match lookup called known with
    | Some c =>
      if full_call_of_b ap c
      then
        Apply_expr (ap_callee ap) (Indirect_known_arity None) (ap_args ap)
          (ap_continuation ap) (ap_exn_continuation ap)
          (ap_exn_extra_args ap) (ap_args_arity ap) (ap_return_arity ap)
      else ap
    | None => ap
    end
  | Indirect_unknown_arity | Indirect_known_arity _ => ap
  end.

(* Every call of an expression, in handlers too. Static bindings are
   left: the body of code binds nothing static
   (FlambdaSemanticsSpec.call_code). *)
Fixpoint demote_full_calls (known : list (code_id * code)) (e : expr)
    : expr :=
  match e with
  | Let x kind d body => Let x kind d (demote_full_calls known body)
  | Let_static bindings body =>
    Let_static bindings (demote_full_calls known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (demote_full_calls_in_handler known h)
      (demote_full_calls known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, demote_full_calls_in_handler known h))
         handlers)
      (demote_full_calls known body)
  | Apply ap => Apply (demote_full_call known ap)
  | Apply_cont _ | Switch _ _ | Invalid => e
  end
with demote_full_calls_in_handler (known : list (code_id * code))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (demote_full_calls known handler)
  end.

Definition demote_full_calls_in_code (known : list (code_id * code))
    (c : code) : code :=
  match c with
  | Code params my_closure ret exn metadata body =>
    Code params my_closure ret exn metadata (demote_full_calls known body)
  end.

(* The same in the code of a stub that a static binding binds. Code
   that is not a stub is left. *)
Definition demote_full_in_stub_binding (known : list (code_id * code))
    (binding : static_binding) : static_binding :=
  match binding with
  | Bind_code id c =>
    if code_is_stub c
    then Bind_code id (demote_full_calls_in_code known c)
    else binding
  | Bind_deleted_code _ | Bind_symbol _ _ => binding
  end.

(* The same in the code of each stub that an expression binds, in
   handlers too. What is known in the code of a stub is the code of its
   own group of bindings and of the groups around that group
   (FlambdaNormaliseSpec.codes_of), and nothing else: not code that the
   expression binds elsewhere, in a handler beside the group or after
   it. Everything that is not the code of a stub is left. *)
Fixpoint demote_full_in_stubs (known : list (code_id * code)) (e : expr)
    : expr :=
  match e with
  | Let x kind d body => Let x kind d (demote_full_in_stubs known body)
  | Let_static bindings body =>
    let known := codes_of bindings known in
    Let_static (map (demote_full_in_stub_binding known) bindings)
      (demote_full_in_stubs known body)
  | Let_cont_non_recursive k h body =>
    Let_cont_non_recursive k (demote_full_in_stubs_of_handler known h)
      (demote_full_in_stubs known body)
  | Let_cont_recursive handlers body =>
    Let_cont_recursive
      (map (fun '(k, h) => (k, demote_full_in_stubs_of_handler known h))
         handlers)
      (demote_full_in_stubs known body)
  | Apply _ | Apply_cont _ | Switch _ _ | Invalid => e
  end
with demote_full_in_stubs_of_handler (known : list (code_id * code))
    (h : continuation_handler) : continuation_handler :=
  match h with
  | Handler params metadata handler =>
    Handler params metadata (demote_full_in_stubs known handler)
  end.

(* In the code of each stub of a unit, the direct calls that the test
   passes, of code which the unit binds in the group of the stub or in a
   group around it, are made calls of known arity. *)
Definition demote_full_stub_calls (e : expr) : expr :=
  demote_full_in_stubs [] e.

(* ======================================================================== *)
(* What that rewrite supposes of the static environment                     *)
(* ======================================================================== *)

(* What the meaning of a call of known arity asks of the code of its
   callee: the arity of its parameters and whether it is tupled. The
   rewrite also reads whether code is a stub, which no call does. *)
Definition same_signature (c c' : code) : Prop :=
  code_params_arity c' = code_params_arity c
  ∧ code_is_tupled c' = code_is_tupled c.

(* The static environment binds each code id that is known to code with
   the signature of the code known. *)
Definition statics_know_signatures (known : list (code_id * code))
    (statics : static_env) : Prop :=
  ∀ (id : code_id) (c : code),
    lookup id known = Some c ->
    ∃ c' : code,
      lookup id (static_code statics) = Some c' ∧ same_signature c c'.

(* A step keeps it: a step only adds to the static environment, and a
   code id is bound once. *)
Definition statics_know_signatures_step_statement : Prop :=
  ∀ (known : list (code_id * code)) (s s' : state),
    statics_know_signatures known (state_statics s) ->
    step s = Step s' ->
    statics_know_signatures known (state_statics s').

(* A group of bindings provides it of its own code: evaluated as it is
   in the term, or as the rewrite leaves it with whatever is known, in a
   static environment that knows the signatures of [known], it gives one
   that knows those of the group too. *)
Definition group_knows_signatures_statement : Prop :=
  ∀ (known known' : list (code_id * code)) (vars : env)
    (bindings : list static_binding) (statics extended : static_env),
    statics_know_signatures known statics ->
    eval_static_bindings vars bindings statics = Some extended
    ∨ eval_static_bindings vars
        (map (demote_full_in_stub_binding known') bindings) statics
      = Some extended ->
    statics_know_signatures (codes_of bindings known) extended.

(* ======================================================================== *)
(* The statements of that rewrite                                           *)
(* ======================================================================== *)

(* A call that has a step has the same step once demoted, where the
   static environment knows the signatures of the code known. *)
Definition demote_full_call_step_statement : Prop :=
  ∀ (known : list (code_id * code)) (ap : apply_expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_know_signatures known statics ->
    step (State (Apply ap) vars conts statics hp rd) <> Undefined ->
    step (State (Apply (demote_full_call known ap)) vars conts statics hp rd)
    = step (State (Apply ap) vars conts statics hp rd).

(* An expression with its calls demoted refines the expression, in a
   state whose static environment knows the signatures of the code
   known. *)
Definition demote_full_calls_refines_statement : Prop :=
  ∀ (known : list (code_id * code)) (e : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_know_signatures known statics ->
    refines (State e vars conts statics hp rd)
      (State (demote_full_calls known e) vars conts statics hp rd).

(* So does an expression with the calls of the code of its stubs
   demoted. *)
Definition demote_full_in_stubs_refines_statement : Prop :=
  ∀ (known : list (code_id * code)) (e : expr) (vars : env)
    (conts : cont_env) (statics : static_env) (hp : heap)
    (rd : read_record),
    statics_know_signatures known statics ->
    refines (State e vars conts statics hp rd)
      (State (demote_full_in_stubs known e) vars conts statics hp rd).

(* With nothing known at the start nothing is supposed: the rewrite of a
   unit refines it under all environments. *)
Definition demote_full_stub_calls_refines_statement : Prop :=
  ∀ e : expr, expr_refines e (demote_full_stub_calls e).

(* The rewrite commutes with renaming, which renames no code id and
   keeps what code is annotated with. *)
Definition demote_full_stub_calls_equivariant_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    demote_full_stub_calls (rename_expr r e)
    = rename_expr r (demote_full_stub_calls e).

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
Let z : variable := 5%positive.

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
  = stub_example (stub_code_metadata stub_example_metadata)
    Indirect_unknown_arity.

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

(* The rewrite that makes a call of known arity, computed. The wrapper
   is a stub that makes the call given; the code of the function has the
   metadata given. *)
Definition stub_example_wrapper_calling (ap : apply_expr) : static_binding :=
  Bind_code wrapper
    (Code [x] my_closure ret exn (stub_code_metadata stub_example_metadata)
       (Let callee Value
          (Project_value_slot function_slot_kept (Var my_closure))
          (Apply ap))).

Definition stub_example_function_with (metadata : code_metadata)
    : static_binding :=
  Bind_code function
    (Code [x; y] my_closure ret exn metadata
       (Apply_cont (Apply_cont_expr ret [Var y] None))).

Definition stub_example_end : expr :=
  Apply_cont (Apply_cont_expr done [] None).

(* Where the code that the call names is bound. In the group of the
   stub, before the stub or after it, and in a group around that group,
   the call is demoted. In a group that follows, in a handler beside,
   and where the unit does not bind the code, the call is left. *)
Definition demote_full_stub_calls_scope_statement : Prop :=
  let f := stub_example_function_with
             (make_code_metadata [Singleton Value; Singleton Value]
                [Singleton Value]) in
  let w kind := stub_example_wrapper_calling (stub_example_call kind) in
  let direct := Direct function in
  let known := Indirect_known_arity None in
  demote_full_stub_calls (Let_static [f; w direct] stub_example_end)
  = Let_static [f; w known] stub_example_end
  ∧ demote_full_stub_calls (Let_static [w direct; f] stub_example_end)
    = Let_static [w known; f] stub_example_end
  ∧ demote_full_stub_calls
      (Let_static [f] (Let_static [w direct] stub_example_end))
    = Let_static [f] (Let_static [w known] stub_example_end)
  ∧ (let e := Let_static [w direct] (Let_static [f] stub_example_end) in
     demote_full_stub_calls e = e)
  ∧ (let e :=
       Let_cont_non_recursive done
         (Handler [] (make_handler_metadata [])
            (Let_static [f] stub_example_end))
         (Let_static [w direct] stub_example_end)
     in
     demote_full_stub_calls e = e)
  ∧ (let e := Let_static [w direct] stub_example_end in
     demote_full_stub_calls e = e).

(* What the code named and the call must be. The call is left where the
   code is a stub; where it is tupled; where the call has fewer groups of
   arguments than the code has parameters; and where it has none, the
   arity of parameters of the code being empty too (the test reads the
   arity that the code is annotated with, and not its parameters). *)
Definition demote_full_stub_calls_test_statement : Prop :=
  let two := [Singleton Value; Singleton Value] in
  let unit (metadata : code_metadata) (ap : apply_expr) :=
    Let_static
      [stub_example_function_with metadata; stub_example_wrapper_calling ap]
      stub_example_end
  in
  let call (args : list simple) (args_arity : arity) :=
    Apply_expr (Var callee) (Direct function) args ret exn [] args_arity
      [Singleton Value]
  in
  let full := call [Const (Tagged_immediate 1); Var x] two in
  (let e :=
     unit (stub_code_metadata (make_code_metadata two [Singleton Value])) full
   in
   demote_full_stub_calls e = e)
  ∧ (let e := unit (make_tupled_code_metadata two [Singleton Value]) full in
     demote_full_stub_calls e = e)
  ∧ (let e :=
       unit (make_code_metadata two [Singleton Value])
         (call [Var x] [Singleton Value])
     in
     demote_full_stub_calls e = e)
  ∧ (let e := unit (make_code_metadata [] [Singleton Value]) (call [] []) in
     demote_full_stub_calls e = e).

(* Why the rewrite must know the code. A direct call with two groups of
   arguments, where the static environment binds the code id to code
   of three parameters: the call is a partial application and has a
   step; the call of known arity of the same callee has none. *)
Definition demote_full_call_needs_scope_statement : Prop :=
  let vars :=
    [(callee, Value_closure function []);
     (x, Value_const (Tagged_immediate 0))]
  in
  let conts := [(ret, Outer ret); (exn, Outer exn)] in
  let three := [Singleton Value; Singleton Value; Singleton Value] in
  let statics :=
    {| static_values := [];
       static_code :=
         [(function,
           Code [x; y; z] my_closure ret exn
             (make_code_metadata three [Singleton Value])
             (Apply_cont (Apply_cont_expr ret [Var z] None)))] |}
  in
  step
    (State (Apply (stub_example_call (Direct function))) vars conts statics
       empty_heap no_reads)
  <> Undefined
  ∧ step
      (State (Apply (stub_example_call (Indirect_known_arity None))) vars
         conts statics empty_heap no_reads)
    = Undefined.

End Examples.
