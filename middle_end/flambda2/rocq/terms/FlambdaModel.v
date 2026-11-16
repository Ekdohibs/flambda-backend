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

(* The terms of Flambda 2 as the semantics sees them: a fragment.

   This file is to be reviewed, against middle_end/flambda2/terms/flambda.mli
   and the modules it refers to. It holds definitions only. The fragment has
   - the constants ([Reg_width_const.t]), in full but for the string of
     [Poison]; the kinds ([Flambda_kind.t]) are in FlambdaKindModel.v;
   - [Let] of one variable, bound to a [Simple], a primitive application, a
     closure of a new set of closures, or the projection of a value slot
     or of a function slot; the primitives ([Flambda_primitive.t]) are in
     FlambdaPrimitiveModel.v, a fragment of their own;
   - [Let] with the bound pattern [Static] ("let symbol"), here
     [Let_static]: a group of bindings of code ids to code and of symbols
     to static constants;
   - [Let_cont], non-recursive and recursive;
   - [Apply] of a function, direct or indirect, with a return continuation
     and an exception continuation, which may have extra arguments
     ([Exn_continuation.t]);
   - [Apply_cont], with its trap action ([Trap_action.t]);
   - [Switch];
   - [Invalid].

   What is left out, and is to be added in later steps:
   - of a set of closures: a function slot whose code is deleted
     ([Function_declarations.Deleted]), the allocation mode (a set on the
     stack is not told from one on the heap), [only_full_applications],
     and everything about its layout in memory;
   - the static constants that are arrays of naked numbers or of
     floats, the empty array or float blocks, and mutable blocks and
     strings (an immutable array of values is here:
     [Static_const_immutable_value_array]);
   - [Rec_info];
   - the other call kinds (methods, C calls, effects), [Never_returns],
     partial and over-application, and the modes and inlining attributes
     of an [Apply];
   - the [is_cold] flag of continuation handlers, and the
     [is_applied_with_traps] of a non-recursive [Let_cont], which says
     of its body what can be read there;
   - subkinds and nullability ([Flambda_kind.With_subkind]), debuginfo,
     and every annotation that has no effect on execution.

   Kinds. A variable has a kind, given where it is bound: the compiler's
   [Variable.kind]. A [Let] gives the kind of its variable; a
   continuation handler gives the kinds of its parameters; code gives the
   arity of its parameters and that of its results, and an [Apply] the
   two arities it calls at ([Apply_expr.args_arity], [return_arity]) and
   the kind of each extra argument of its exception continuation. Code
   also says whether it is tupled ([cm_is_tupled], the compiler's
   [Code_metadata.is_tupled]): whether a call that does not know the
   code gives it one argument, a tuple, for all its parameters. It
   says whether it is a stub too ([cm_is_stub], the compiler's
   [Code.stub]), which is no kind and which execution does not read. An
   arity ([arity]) is the compiler's [Flambda_arity.t]: it keeps how the
   parameters are grouped into unboxed products, which the compiler
   compares at a call, while the parameters and the arguments themselves
   are unarized, here as there. Execution (FlambdaSemanticsSpec.v) looks
   at none of these but the arities of a call and of the code called.

   What a handler and code are annotated with is a record each
   ([handler_metadata], [code_metadata]), so that a later annotation is
   one more field and no term changes shape for it. Make one with
   [make_handler_metadata] or [make_code_metadata], read it with its
   projections, and bind it whole in a pattern; do not apply the
   constructor of the record or take one apart by position. A field
   added later is then given, by the function, the value that leaves a
   term written before with its meaning.

   Exception handlers. A handler says whether it is an exception
   handler ([hm_is_exn_handler], the compiler's
   [Continuation_handler.is_exn_handler]): one that trap actions install
   and remove, and that raises and the exceptions of calls go to.
   Execution does not look at the flag. [make_handler_metadata] makes a
   handler that is not one, and [make_exn_handler_metadata] one that
   is.

   Kinds, arities and the two records of metadata have no names in them.
   Renaming a term leaves them as they are, and two terms that are the
   same up to the names they bind have the same ones.

   Differences of presentation from the compiler:
   - Variables and continuations are plain names. The compiler works up to
     alpha-equivalence ([Name_abstraction]); here a binder shadows, and the
     semantics (FlambdaSemanticsSpec.v) uses environments and closures, so
     no substitution and no freshness condition is needed to define
     execution. Symbols and code ids are plain names in the compiler too.
   - The two forms of [Let_cont] are two constructors of [expr], and the
     [Let] of static constants a third, [Let_static].
   - A [Let_static] pairs each name with what it is bound to
     ([static_binding]), where the compiler has a [Bound_static.t] and a
     [Static_const_group.t] side by side.
   - A recursive [Let_cont] in the compiler has [invariant_params] shared
     by all its handlers; here every handler lists all its parameters, the
     invariant ones first.
   - The kinds of the parameters of a handler are a list beside the list
     of parameters ([hm_params_kinds]), where the compiler pairs each
     parameter with its kind ([Bound_parameter.t]). Code has no such
     list: the kinds of its parameters are those of its parameter arity,
     unarized ([code_params_kinds]); in the compiler the two are kept
     equal by construction. The compiler has no term in which the list
     of kinds and the list of parameters differ in length; here there
     are such terms, and nothing yet says that they are not well-formed.
   - A primitive application is a primitive with a list of arguments, where
     [Flambda_primitive.t] holds the arguments by arity. A primitive
     applied to the wrong number of arguments has no value
     (FlambdaPrimitiveSpec.v).
   - [Project_value_slot] and [Project_function_slot], unary primitives in
     the compiler, are forms of [named] of their own, as their meaning is
     fixed by that of closures.
   - A set of closures has two presentations. [Set_of_closures] and
     [Static_const_set_of_closures] are a set of one function, whose
     function slot is not named; [Project_value_slot] projects from a
     closure whatever function slot it is at. [Closure_of_set] and
     [Static_const_closure_of_set] are a set of any number of functions,
     each at a named function slot; [Project_value_slot_from] is the
     compiler's projection of a value slot, which names the function slot
     of the closure it projects from ([project_from]).
   - A [Let] binds one variable. The compiler's [Let] of a set of closures
     binds one variable to each closure of the set
     ([Bound_pattern.Set_of_closures]); here it is [let_closures]: a [Let]
     of the first closure, and for each other closure a [Let] of its
     projection from the first. Closures have no identity, so this is the
     same.
   - Likewise a [Let_static] binds one symbol to each closure of a static
     set by a [Bind_symbol] of its own, which repeats the set and says
     which symbols its other closures are bound to; the compiler binds
     them all with one pattern ([Bound_static.Pattern.Set_of_closures]).
   - Of the names in the forms of a set, a renaming acts on the variables
     only: those of the value slots and of the closure projected from,
     and the variables that [let_closures] binds, each by a [Let].
     Function slots and value slots are not renamed, here or in the
     compiler. Nor are the code ids of the functions and the symbols of
     the closures of a static set, like every code id and symbol of the
     model: the compiler's [Bound_pattern.Static] does not permute the
     names it binds either.
   - The callee of an [Apply] is always given, where the compiler may omit
     it for a direct call.
   - The exception continuation of an [Apply] is two fields, the handler
     and the extra arguments with their kinds, where the compiler has an
     [Exn_continuation.t].
   - The numbers in constants are integers: those of the floats and
     vectors are bit patterns, as in the compiler. No type here
     restricts one to its range, nor a tag, nor any other number of a
     term: FlambdaPrimitiveSpec.v says what the ranges are, and
     FlambdaWellFormedSpec.numbers_in_range is the check that every
     number of a term is in its range. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Export FlambdaKindModel FlambdaPrimitiveModel.
Import ListNotations.

(* [Variable.t], [Continuation.t], [Symbol.t], [Code_id.t], [Value_slot.t]
   and [Function_slot.t]. *)
Definition variable : Type := positive.
Definition continuation : Type := positive.
Definition symbol : Type := positive.
Definition code_id : Type := positive.
Definition value_slot : Type := positive.
Definition function_slot : Type := positive.

(* ======================================================================== *)
(* Constants                                                                *)
(* ======================================================================== *)

(* [Reg_width_const.t]. *)
Inductive const : Type :=
  | Naked_immediate (i : Z)
  | Tagged_immediate (i : Z)
  | Naked_float32 (bits : Z)
  | Naked_float (bits : Z)
  | Naked_int8 (i : Z)
  | Naked_int16 (i : Z)
  | Naked_int32 (i : Z)
  | Naked_int64 (i : Z)
  | Naked_nativeint (i : Z)
  | Naked_vec128 (bits : Z)
  | Naked_vec256 (bits : Z)
  | Naked_vec512 (bits : Z)
  | Naked_mask (bits : Z)
  | Null
  | Poison (kind : flambda_kind).

(* [Reg_width_const.kind]. *)
Definition kind_of_const (c : const) : flambda_kind :=
  match c with
  | Naked_immediate _ => Naked_number Kind_naked_immediate
  | Tagged_immediate _ => Value
  | Naked_float32 _ => Naked_number Kind_naked_float32
  | Naked_float _ => Naked_number Kind_naked_float
  | Naked_int8 _ => Naked_number Kind_naked_int8
  | Naked_int16 _ => Naked_number Kind_naked_int16
  | Naked_int32 _ => Naked_number Kind_naked_int32
  | Naked_int64 _ => Naked_number Kind_naked_int64
  | Naked_nativeint _ => Naked_number Kind_naked_nativeint
  | Naked_vec128 _ => Naked_number Kind_naked_vec128
  | Naked_vec256 _ => Naked_number Kind_naked_vec256
  | Naked_vec512 _ => Naked_number Kind_naked_vec512
  | Naked_mask _ => Naked_number Kind_naked_mask
  | Null => Value
  | Poison kind => kind
  end.

(* ======================================================================== *)
(* Arities                                                                  *)
(* ======================================================================== *)

(* [Flambda_arity.Component.t], without subkinds: what one parameter of a
   function, or its result, is made of. An unboxed product stands for as
   many variables as it has kinds in it, and the unboxed product of
   nothing ("void") for none. (The warning silenced is about induction
   schemes through the list, which nothing uses.) *)
#[local] Set Warnings "-register-all".
Inductive arity_component : Type :=
  | Singleton (k : flambda_kind)
  | Unboxed_product (components : list arity_component).

(* [Flambda_arity.t] (at [`Complex]): one component to a parameter. *)
Definition arity : Type := list arity_component.

(* [Flambda_arity.Component.unarize]: the kinds of the variables that a
   component stands for, left to right. *)
Fixpoint unarize_component (c : arity_component) : list flambda_kind :=
  match c with
  | Singleton k => [k]
  | Unboxed_product components => flat_map unarize_component components
  end.

(* [Flambda_arity.unarize]. *)
Definition unarize (a : arity) : list flambda_kind :=
  flat_map unarize_component a.

(* [Flambda_arity.unarize_per_parameter]. *)
Definition unarize_per_parameter (a : arity) : list (list flambda_kind) :=
  map unarize_component a.

(* [Flambda_arity.create_singletons]: the arity of parameters none of
   which is an unboxed product. *)
Definition arity_of_kinds (kinds : list flambda_kind) : arity :=
  map Singleton kinds.

Definition arity_component_eq_dec :
  ∀ c c' : arity_component, {c = c'} + {c <> c'}.
Proof.
  fix eq_dec 1. intros [k | components] [k' | components'];
    try (right; discriminate).
  - destruct (kind_eq_dec k k') as [-> | Hne];
      [ left; reflexivity | right; congruence ].
  - destruct (list_eq_dec eq_dec components components') as [-> | Hne];
      [ left; reflexivity | right; congruence ].
Defined.

(* [Flambda_arity.equal_ignoring_subkinds]: the same components, with the
   same unboxed products in them. *)
Definition arity_eqb (a a' : arity) : bool :=
  if list_eq_dec arity_component_eq_dec a a' then true else false.

(* Equality of lists of kinds, and of lists of those. *)
Definition kinds_eqb (kinds kinds' : list flambda_kind) : bool :=
  if list_eq_dec kind_eq_dec kinds kinds' then true else false.

Definition kinds_per_parameter_eqb
    (kinds kinds' : list (list flambda_kind)) : bool :=
  if list_eq_dec (list_eq_dec kind_eq_dec) kinds kinds' then true else false.

(* [Flambda_arity.cardinal_unarized]: how many variables the parameters
   of an arity are, unarized. *)
Definition cardinal_unarized (a : arity) : nat := length (unarize a).

(* [Flambda_arity.partially_apply]: the arity that is left once the
   first [provided] parameters are given. *)
Definition partially_apply (a : arity) (provided : nat) : arity :=
  skipn provided a.

(* [arity_mismatch] of simplify/simplify_apply_expr.ml, negated: the
   kinds of the parameters and of the arguments, each unarized, agree as
   far as the shorter of the two goes. *)
Fixpoint kinds_agree_on_prefix (params args : list (list flambda_kind))
    : bool :=
  match params, args with
  | [], _ | _, [] => true
  | param :: params, arg :: args =>
    kinds_eqb param arg && kinds_agree_on_prefix params args
  end.

(* [Flambda_arity.is_one_param_of_kind_value]. *)
Definition is_one_param_of_kind_value (a : arity) : bool :=
  match a with
  | [Singleton Value] => true
  | _ => false
  end.

(* The test of arities by which the compiler takes a call made at the
   arities [args_arity] and [return_arity], of code whose arities are
   [params_arity] and [result_arity], for a partial application or an
   over-application ([simplify_direct_function_call]): the arguments
   agree with the parameters as far as they go, and there are more
   arguments than parameters and the code returns one [Value], or fewer,
   but not none, and the call expects one [Value]. The machine makes
   this test of a call that has not as many arguments as the code has
   parameters (FlambdaSemanticsSpec.shape_of_closure_call).

   It is a test of the arities and of nothing else. The compiler comes
   to it with a call whose arguments have the kinds of [args_arity]: one
   whose arguments have not is [Invalid] already
   ([simplify_apply_shared]), whatever this test says of its arities;
   the machine asks that too. And it comes to it with the arguments as
   the code takes them: for tupled code, after the tuple that a call of
   unknown arity gives has been taken apart
   ([simplify_direct_tuple_application]), as the machine does.

   Of the calls with more or fewer arguments than parameters that fail
   the test, Simplify replaces by [Invalid] those whose arguments do not
   agree with the parameters as far as they go, and those whose results
   are not one [Value] as above. The one that is left is a call with no
   argument of code that has parameters: there the compiler stops with a
   fatal error, so no term it accepts has one. *)
Definition partial_or_over_application
    (args_arity return_arity params_arity result_arity : arity) : bool :=
  let provided := length args_arity in
  let params := length params_arity in
  kinds_agree_on_prefix (unarize_per_parameter params_arity)
    (unarize_per_parameter args_arity)
  && (if Nat.ltb params provided
      then is_one_param_of_kind_value result_arity
      else
        Nat.ltb 0 provided && Nat.ltb provided params
        && is_one_param_of_kind_value return_arity).

(* ======================================================================== *)
(* Terms                                                                    *)
(* ======================================================================== *)

(* [Simple.t], without coercions. *)
Inductive simple : Type :=
  | Var (x : variable)
  | Symbol (s : symbol)
  | Const (c : const).

(* [Flambda.named]. A [Set_of_closures] has one function, given by its
   code, and the values of its value slots.

   [Closure_of_set] is the closure at [slot] of a new set of closures
   ([Set_of_closures.t]): [function_decls] are its functions, the code of
   each function slot in the order of [Function_declarations.funs_in_order],
   and [value_slots] the values of its value slots, which its functions
   share. [Project_value_slot_from] is the value of a value slot of the
   set of which [closure] is the closure at [project_from], and
   [Project_function_slot] the closure at [move_to] of the set of which
   [closure] is the closure at [move_from]. *)
Inductive named : Type :=
  | Simple (s : simple)
  | Prim (p : prim) (args : list simple)
  | Set_of_closures (code : code_id)
      (value_slots : list (value_slot * simple))
  | Closure_of_set (slot : function_slot)
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * simple))
  | Project_value_slot (slot : value_slot) (closure : simple)
  | Project_value_slot_from (project_from : function_slot)
      (slot : value_slot) (closure : simple)
  | Project_function_slot (move_from move_to : function_slot)
      (closure : simple).

(* [Static_const.t]: what a symbol may be bound to. Its fields are
   [Simple]s, where the compiler has, for a boxed number, a constant or a
   variable. A block is immutable. [Static_const_immutable_value_array]
   is the compiler's [Immutable_value_array], of its elements: the
   constant has no array kind.

   [Static_const_closure_of_set] is the closure at [slot] of a static set
   of closures; [closure_symbols] are the symbols that the closures of the
   set are bound to ([Bound_static.Pattern.Set_of_closures]), this one
   included. *)
Inductive static_const : Type :=
  | Static_const_set_of_closures (code : code_id)
      (value_slots : list (value_slot * simple))
  | Static_const_closure_of_set (slot : function_slot)
      (closure_symbols : list (function_slot * symbol))
      (function_decls : list (function_slot * code_id))
      (value_slots : list (value_slot * simple))
  | Static_const_block (tag : Z) (fields : list simple)
  | Static_const_boxed_number (contents : simple)
  | Static_const_immutable_string (bytes : list Z)
  | Static_const_immutable_value_array (elements : list simple).

(* Whether two simples, or two static constants, are the same. *)
Definition simple_eq_dec : forall s s' : simple, {s = s'} + {s <> s'}.
Proof.
  decide equality; try apply Pos.eq_dec.
  decide equality; try apply Z.eq_dec. repeat decide equality.
Defined.

Definition static_const_eq_dec :
  forall const const' : static_const, {const = const'} + {const <> const'}.
Proof.
  decide equality;
    try apply Pos.eq_dec; try apply Z.eq_dec; try apply simple_eq_dec;
    apply list_eq_dec; try apply Z.eq_dec; try apply simple_eq_dec;
    decide equality; try apply Pos.eq_dec; apply simple_eq_dec.
Defined.

(* [Trap_action.Raise_kind.t]: whether a raise records a backtrace. *)
Inductive raise_kind : Type :=
  | Regular
  | Reraise
  | No_trace.

(* [Trap_action.t]: what an [Apply_cont] does to the stack of exception
   handlers installed, before it applies its continuation. [Push] installs
   a handler. [Pop] removes the one installed last, which the compiler
   expects to be the one named. *)
Inductive trap_action : Type :=
  | Push (exn_handler : continuation)
  | Pop (exn_handler : continuation) (kind : option raise_kind).

(* [Apply_cont_expr.t]. An [Apply_cont] whose trap action pops the
   continuation that it applies is a raise ([Apply_cont_expr.is_raise]):
   the exception is its first argument. *)
Record apply_cont : Type := Apply_cont_expr {
  ac_continuation : continuation;
  ac_args : list simple;
  ac_trap_action : option trap_action
}.

(* The continuation that a trap action names. *)
Definition trap_action_handler (action : trap_action) : continuation :=
  match action with
  | Push exn_handler | Pop exn_handler _ => exn_handler
  end.

(* The continuations that an [Apply_cont] names: the one it applies, and
   the one of its trap action if it has one. *)
Definition apply_cont_continuations (ac : apply_cont) : list continuation :=
  ac_continuation ac
  :: match ac_trap_action ac with
     | Some action => [trap_action_handler action]
     | None => []
     end.

Definition no_trap_action (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | None => true
  | Some _ => false
  end.

(* [Apply_cont_expr.is_raise]. *)
Definition is_raise (ac : apply_cont) : bool :=
  match ac_trap_action ac with
  | Some (Pop exn_handler _) => Pos.eqb (ac_continuation ac) exn_handler
  | Some (Push _) | None => false
  end.

(* [Call_kind.Function_call.t]: the code called is known, or it is that of
   whatever closure the callee turns out to be. *)
Inductive call_kind : Type :=
  | Direct (code : code_id)
  | Indirect.

(* [Apply_expr.t]. An exception raised by the call goes to the exception
   continuation, which is applied to the exception followed by the extra
   arguments, each given with its kind ([Exn_continuation.t]). The
   arities are those the call is made at ([Apply_expr.args_arity],
   [return_arity]): the kinds of its arguments, which are unarized,
   grouped by parameter, and those of the results it expects. *)
Record apply_expr : Type := Apply_expr {
  ap_callee : simple;
  ap_call_kind : call_kind;
  ap_args : list simple;
  ap_continuation : continuation;
  ap_exn_continuation : continuation;
  ap_exn_extra_args : list (simple * flambda_kind);
  ap_args_arity : arity;
  ap_return_arity : arity
}.

(* The extra arguments of the exception continuation of a call, without
   their kinds. *)
Definition exn_extra_simples (ap : apply_expr) : list simple :=
  map fst (ap_exn_extra_args ap).

Definition no_exn_extra_args (ap : apply_expr) : bool :=
  match ap_exn_extra_args ap with
  | [] => true
  | _ :: _ => false
  end.

(* What a continuation handler is annotated with: the kinds of its
   parameters (those of its [Bound_parameters.t]), one to a parameter,
   and whether it is an exception handler
   ([Continuation_handler.is_exn_handler]). *)
Record handler_metadata : Type := Handler_metadata {
  hm_params_kinds : list flambda_kind;
  hm_is_exn_handler : bool
}.

(* The metadata of a handler that is not an exception handler. *)
Definition make_handler_metadata (params_kinds : list flambda_kind)
    : handler_metadata :=
  {| hm_params_kinds := params_kinds; hm_is_exn_handler := false |}.

(* The metadata of an exception handler. *)
Definition make_exn_handler_metadata (params_kinds : list flambda_kind)
    : handler_metadata :=
  {| hm_params_kinds := params_kinds; hm_is_exn_handler := true |}.

Definition handler_metadata_eqb (m m' : handler_metadata) : bool :=
  kinds_eqb (hm_params_kinds m) (hm_params_kinds m')
  && Bool.eqb (hm_is_exn_handler m) (hm_is_exn_handler m').

(* What code is annotated with, of [Code_metadata.t]: the arity of its
   parameters ([params_arity]; unarized, it gives the kinds of the
   parameters themselves) and that of its results ([result_arity];
   unarized, it gives the kinds of the parameters of its return
   continuation), whether it is tupled ([is_tupled]) and whether it is
   a stub ([stub]).

   Tupled code is what the compiler makes of a function whose one
   parameter is a tuple pattern, as in [let f (x, y) = ...]
   (lambda/translcore.ml, [transl_tupled_function]): the code has one
   parameter for each component of the tuple. A call that goes by the
   code gives it the components. A call that does not know the code
   gives the function one argument, the tuple
   (FlambdaSemanticsSpec.shape_of_closure_call).

   A stub is code that the compiler makes and the program does not
   write, such as the function that a partial application of a known
   function becomes, which keeps the function and the arguments given
   and calls the one with the others and its own
   (from_lambda/closure_conversion.ml, [wrap_partial_application]).
   [cm_is_stub] is the compiler's [Code.stub], as it is in the term.
   The machine (FlambdaSemanticsSpec.v) does not read it, and neither
   does FlambdaWellFormedSpec.well_formed_unit. Nothing is believed of
   it: it may be set on any code, and what a term means is the same
   whatever it says. The comparison of two terms reads it
   ([code_metadata_eqb]): two terms that differ in it are not the
   same. Two rewrites of the validation read it, to leave the code of
   a stub as Simplify leaves it: FlambdaKnownSlotsSpec.read_slots reads
   no value slot there, and FlambdaStubCallsSpec.demote_stub_calls
   makes its direct calls indirect. Each refines whatever the flag
   says. *)
Record code_metadata : Type := Code_metadata {
  cm_params_arity : arity;
  cm_result_arity : arity;
  cm_is_tupled : bool;
  cm_is_stub : bool
}.

(* The metadata of code that is not tupled, and is not a stub. *)
Definition make_code_metadata (params_arity result_arity : arity)
    : code_metadata :=
  {| cm_params_arity := params_arity;
     cm_result_arity := result_arity;
     cm_is_tupled := false;
     cm_is_stub := false |}.

(* The metadata of tupled code, which is not a stub. *)
Definition make_tupled_code_metadata (params_arity result_arity : arity)
    : code_metadata :=
  {| cm_params_arity := params_arity;
     cm_result_arity := result_arity;
     cm_is_tupled := true;
     cm_is_stub := false |}.

(* The same metadata, of code that is a stub. *)
Definition stub_code_metadata (m : code_metadata) : code_metadata :=
  {| cm_params_arity := cm_params_arity m;
     cm_result_arity := cm_result_arity m;
     cm_is_tupled := cm_is_tupled m;
     cm_is_stub := true |}.

Definition code_metadata_eqb (m m' : code_metadata) : bool :=
  arity_eqb (cm_params_arity m) (cm_params_arity m')
  && arity_eqb (cm_result_arity m) (cm_result_arity m')
  && Bool.eqb (cm_is_tupled m) (cm_is_tupled m')
  && Bool.eqb (cm_is_stub m) (cm_is_stub m').

(* [Flambda.expr], [Flambda.continuation_handler],
   [Flambda.static_const_or_code] with the name it is bound to, and
   [Function_params_and_body], the code of a function.

   The arms of a [Switch] are a [Target_ocaml_int.Map.t] in the compiler;
   here they are an association list, of which the first arm with a given
   discriminant counts.

   The body of code runs with the parameters and the closure called
   ([my_closure]) as its only variables, and the return and exception
   continuations as its only continuations.

   A [Let] gives the kind of its variable. A handler and code give the
   kinds of their parameters, and code those of its results, in their
   metadata. [my_closure] has kind [Value], and an exception continuation
   takes one argument of kind [Value]. *)
Inductive expr : Type :=
  | Let (x : variable) (kind : flambda_kind) (defining_expr : named)
      (body : expr)
  | Let_static (bindings : list static_binding) (body : expr)
  | Let_cont_non_recursive (k : continuation)
      (handler : continuation_handler) (body : expr)
  | Let_cont_recursive (handlers : list (continuation * continuation_handler))
      (body : expr)
  | Apply (ap : apply_expr)
  | Apply_cont (ac : apply_cont)
  | Switch (scrutinee : simple) (arms : list (Z * apply_cont))
  | Invalid
with continuation_handler : Type :=
  | Handler (params : list variable) (metadata : handler_metadata)
      (handler : expr)
with static_binding : Type :=
  | Bind_code (id : code_id) (c : code)
  | Bind_deleted_code (id : code_id)
  | Bind_symbol (s : symbol) (const : static_const)
with code : Type :=
  | Code (params : list variable) (my_closure : variable)
      (return_continuation exn_continuation : continuation)
      (metadata : code_metadata) (body : expr).

(* The kinds of the parameters of a handler. *)
Definition handler_kinds (h : continuation_handler) : list flambda_kind :=
  match h with Handler _ metadata _ => hm_params_kinds metadata end.

(* Whether a handler is an exception handler. *)
Definition handler_is_exn_handler (h : continuation_handler) : bool :=
  match h with Handler _ metadata _ => hm_is_exn_handler metadata end.

Definition code_params (c : code) : list variable :=
  match c with Code params _ _ _ _ _ => params end.

Definition code_my_closure (c : code) : variable :=
  match c with Code _ my_closure _ _ _ _ => my_closure end.

Definition code_return_continuation (c : code) : continuation :=
  match c with Code _ _ return_continuation _ _ _ => return_continuation end.

Definition code_exn_continuation (c : code) : continuation :=
  match c with Code _ _ _ exn_continuation _ _ => exn_continuation end.

(* The arities of the parameters of code, and of its results. *)
Definition code_params_arity (c : code) : arity :=
  match c with Code _ _ _ _ metadata _ => cm_params_arity metadata end.

Definition code_result_arity (c : code) : arity :=
  match c with Code _ _ _ _ metadata _ => cm_result_arity metadata end.

(* Whether code is tupled. *)
Definition code_is_tupled (c : code) : bool :=
  match c with Code _ _ _ _ metadata _ => cm_is_tupled metadata end.

(* Whether code is a stub. *)
Definition code_is_stub (c : code) : bool :=
  match c with Code _ _ _ _ metadata _ => cm_is_stub metadata end.

(* The kinds of the parameters of code, and of the parameters of its
   return continuation. *)
Definition code_params_kinds (c : code) : list flambda_kind :=
  unarize (code_params_arity c).

Definition code_result_kinds (c : code) : list flambda_kind :=
  unarize (code_result_arity c).

Definition code_body (c : code) : expr :=
  match c with Code _ _ _ _ _ body => body end.

(* ======================================================================== *)
(* Sets of closures of several functions                                    *)
(* ======================================================================== *)

(* Each variable bound to the projection of its function slot from the
   closure [base], which is at [base_slot]. A closure has kind [Value]. *)
Fixpoint let_projections (base : variable) (base_slot : function_slot)
    (bound : list (variable * function_slot)) (body : expr) : expr :=
  match bound with
  | [] => body
  | (x, slot) :: bound =>
    Let x Value (Project_function_slot base_slot slot (Var base))
      (let_projections base base_slot bound body)
  end.

(* The compiler's [Let] of a set of closures: the variables [bound_vars]
   ([Bound_pattern.Set_of_closures]) are bound, in the order of the
   functions of the set, to its closures. The first closure is made and
   the others are projected from it. There must be as many variables as
   functions, and at least one; otherwise there is no such term. *)
Definition let_closures (bound_vars : list variable)
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) (body : expr)
    : option expr :=
  match bound_vars, function_decls with
  | x :: bound_vars, (slot, _) :: other_decls =>
    if Nat.eqb (length bound_vars) (length other_decls)
    then
      Some
        (Let x Value (Closure_of_set slot function_decls value_slots)
           (let_projections x slot
              (combine bound_vars (map fst other_decls)) body))
    else None
  | _, _ => None
  end.

(* The static set of closures whose closures are bound to
   [closure_symbols], as the bindings of a [Let_static]: one for each
   symbol. *)
Definition bind_closures (closure_symbols : list (function_slot * symbol))
    (function_decls : list (function_slot * code_id))
    (value_slots : list (value_slot * simple)) : list static_binding :=
  map
    (fun '(slot, s) =>
       Bind_symbol s
         (Static_const_closure_of_set slot closure_symbols function_decls
            value_slots))
    closure_symbols.
