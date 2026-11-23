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

(* Controls of FlambdaUntagAtBinderSpec.bind_untaggings: small terms,
   what the pass makes of each, and what the machine does at some of
   their handlers.

   This file is to be reviewed: the terms and the statements. Each
   statement is a closed Rocq equality, or a conjunction of such
   equalities, about the displayed terms or machine values. The
   Flambda terms may have free names. None uses the equivalence
   theorem of the pass. The proofs, in
   FlambdaUntagAtBinderExamplesProofs.v, are computations and need no
   review.

   The terms. The variable 1 is a function and the variable 2 an
   argument for it. The continuation 1 is where a term returns and
   the continuation 2 where it raises. [call k] calls the function
   and returns to [k]. In [returned handler] a first call returns to
   the continuation 3, whose one parameter, the variable 10, states
   "not null, a tagged immediate" ([tagged]), and [handler] is the
   body of that handler. Most controls have that shape and say what
   the pass makes of the handler. The variable of the untagging is
   12. Where a second call is made it returns to the continuation 4,
   whose parameter, the variable 11, states nothing.

   The sections:
   - the pass acts: an untagging that is after a second call is bound
     at the head of the handler, and one that is at the head stays;
   - other binders of the two variables: every [Let] of the variable
     to the untagging that is reached with no binder of either on the
     way is taken away, and one under such a binder stays, whether
     the binder is a [Let] or a parameter; the first untagging gives
     the name even where it is under another binder of that name, and
     a use under that binder keeps what the binder gave it, which the
     machine shows;
   - recursive continuations: an untagging in the handler of one is
     taken away for the parameter of an enclosing handler, unless a
     parameter of the recursive handler is one of the two variables;
     nothing is bound at the head of a recursive handler, nor inside
     one, by the pass itself, and
     FlambdaRecHandlersSpec.rewrite_rec_handlers takes the pass there;
   - where the pass does nothing: the variable has a free use in the
     handler; the untagging is at another kind; the variable is a
     parameter of the handler; the parameters are not distinct; the
     handler is an exception handler; no call returns to it; the
     parameter may be null, or is of a variant that has a block;
   - what the machine accepts at the binder: a tagged immediate in
     range, and neither null, nor poison, nor a tagged immediate out
     of range, nor a block; the untagging has a value on what is
     accepted. The range is that of
     FlambdaPrimitiveSpec.int_width;
   - which calls return to a handler: a non-recursive continuation of
     the same name between the handler and a call hides the handler
     from a call in the body of its [Let_cont] and not from one in
     its own handler; a recursive one hides it from both; one of
     another name hides nothing;
   - two parameters that state a tagged immediate: both untaggings
     are bound at the head, that of the first parameter outermost;
     where the two are untagged to one variable in two places, the
     [Let] at the head for the first parameter is read by nothing and
     its untagging stays, the machine does the same with both
     handlers, and the pass applied again adds two more such
     [Let]s. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec FlambdaSemanticsSpec FlambdaNormaliseSpec
  FlambdaRecHandlersSpec FlambdaUntagParamsSpec FlambdaSubkindSpec
  FlambdaUntagAtBinderSpec.
Import ListNotations.
Open Scope positive_scope.

(* ======================================================================== *)
(* What the terms are made of                                               *)
(* ======================================================================== *)

(* An application with no trap action. *)
Definition jump (k : continuation) (args : list simple) : apply_cont :=
  Apply_cont_expr k args None.

Definition untag (x : variable) : named :=
  Prim (Unary Untag_immediate) [Var x].

Definition tag (x : variable) : named := Prim (Unary Tag_immediate) [Var x].

(* The function 1 called on the argument 2. The call returns to [k]
   and raises to the continuation 2. *)
Definition call (k : continuation) : expr :=
  Apply
    (Apply_expr (Var 1) Indirect_unknown_arity [Var 2] k 2 []
       (arity_of_kinds [Value]) (arity_of_kinds [Value])).

(* What a parameter states: not null, a tagged immediate. *)
Definition tagged : param_subkind := (false, Subkind_tagged_immediate).

(* The handler that the first call returns to, and the term. *)
Definition returned_handler (handler : expr) : continuation_handler :=
  Handler [10] (Handler_metadata [Value] [tagged] false) handler.

Definition returned (handler : expr) : expr :=
  Let_cont_non_recursive 3 (returned_handler handler) (call 3).

(* A second call. *)
Definition second_call (handler : expr) : expr :=
  Let_cont_non_recursive 4
    (Handler [11] (make_handler_metadata [Value]) handler) (call 4).

(* The term returns [a] where the number tested is 0, and [b] where it
   is 1. *)
Definition exits (a b : simple) : list (Z * apply_cont) :=
  [(0%Z, jump 1 [a]); (1%Z, jump 1 [b])].

(* What the machine does where a handler is applied to one value under
   no environment: at most [fuel] steps of its body, or [Undefined]
   where the value is not of the kind or of what the parameter
   states. *)
Definition applied (fuel : nat) (h : continuation_handler) (v : value)
    : step_result :=
  match
    apply_cont_value (Closure h [] [] []) [v] empty_static_env empty_heap
      no_reads
  with
  | Step s => run fuel s
  | result => result
  end.

(* Whether the machine enters the handler with that value. *)
Definition enters (h : continuation_handler) (v : value) : bool :=
  match applied 0 h v with
  | Step _ => true
  | Exit _ _ _ _ | Undefined => false
  end.

(* ======================================================================== *)
(* The pass acts                                                            *)
(* ======================================================================== *)

(* The parameter is untagged after the second call, where it is
   tested. *)
Definition late_handler : expr :=
  second_call
    (Let 12 untagged_kind (untag 10)
       (Switch (Var 12) (exits (Var 11) (Var 10)))).

Definition late : expr := returned late_handler.

(* It is untagged where the first call returns. *)
Definition late_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (second_call (Switch (Var 12) (exits (Var 11) (Var 10))))).

Definition late_bound_statement : Prop :=
  bind_untaggings late = late_result.

(* An untagging at the head stays there. *)
Definition at_head_kept_statement : Prop :=
  bind_untaggings late_result = late_result.

(* ======================================================================== *)
(* Other binders of the two variables                                       *)
(* ======================================================================== *)

(* Two [Let]s of the variable to the untagging, one in the handler of
   a continuation and one in the body of its [Let_cont]: both are
   taken away. *)
Definition repeated : expr :=
  returned
    (Let_cont_non_recursive 5
       (Handler [] (make_handler_metadata [])
          (Let 12 untagged_kind (untag 10)
             (Let 13 Value (tag 12) (Apply_cont (jump 1 [Var 13])))))
       (second_call
          (Let 12 untagged_kind (untag 10)
             (Switch (Var 12) [(0%Z, jump 5 []); (1%Z, jump 1 [Var 11])])))).

Definition repeated_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Let 13 Value (tag 12) (Apply_cont (jump 1 [Var 13]))))
          (second_call
             (Switch (Var 12) [(0%Z, jump 5 []); (1%Z, jump 1 [Var 11])])))).

Definition repeated_bound_statement : Prop :=
  bind_untaggings repeated = repeated_result.

(* A [Let] of the variable to something else: the untagging before it
   is taken away, and the one after it stays. *)
Definition rebound_u : expr :=
  returned
    (Let 14 Value (Simple (Var 10))
       (Let 12 untagged_kind (untag 10)
          (Let 12 untagged_kind (Simple (Const (Naked_immediate 7)))
             (Let 12 untagged_kind (untag 10)
                (Switch (Var 12) (exits (Var 14) (Var 10))))))).

Definition rebound_u_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (Let 14 Value (Simple (Var 10))
          (Let 12 untagged_kind (Simple (Const (Naked_immediate 7)))
             (Let 12 untagged_kind (untag 10)
                (Switch (Var 12) (exits (Var 14) (Var 10))))))).

Definition rebound_u_bound_statement : Prop :=
  bind_untaggings rebound_u = rebound_u_result.

(* A [Let] of the parameter: the untagging after it is of another
   value, and stays. *)
Definition rebound_x : expr :=
  returned
    (Let 14 Value (Simple (Var 10))
       (Let 12 untagged_kind (untag 10)
          (Let 10 Value (Simple (Const (Tagged_immediate 5)))
             (Let 12 untagged_kind (untag 10)
                (Switch (Var 12) (exits (Var 14) (Var 10))))))).

Definition rebound_x_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (Let 14 Value (Simple (Var 10))
          (Let 10 Value (Simple (Const (Tagged_immediate 5)))
             (Let 12 untagged_kind (untag 10)
                (Switch (Var 12) (exits (Var 14) (Var 10))))))).

Definition rebound_x_bound_statement : Prop :=
  bind_untaggings rebound_x = rebound_x_result.

(* The first untagging is under a [Let] of its variable to the number
   7, which the variable 13 reads. It stays, and the [Let] at the head
   is read by nothing. *)
Definition below_binder_handler : expr :=
  Let 12 untagged_kind (Simple (Const (Naked_immediate 7)))
    (Let 13 Value (tag 12)
       (Let 12 untagged_kind (untag 10)
          (Switch (Var 12) (exits (Var 13) (Var 10))))).

Definition below_binder_result_handler : expr :=
  Let 12 untagged_kind (untag 10) below_binder_handler.

Definition below_binder_bound_statement : Prop :=
  bind_untaggings (returned below_binder_handler)
  = returned below_binder_result_handler.

(* Applied to the tagged immediate 0, both handlers return what the
   variable 13 read, the tagged immediate 7; applied to 1, both return
   the parameter. *)
Definition below_binder_runs_statement : Prop :=
  applied 10 (returned_handler below_binder_handler)
    (Value_const (Tagged_immediate 0))
  = Exit 1 [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ applied 10 (returned_handler below_binder_result_handler)
      (Value_const (Tagged_immediate 0))
    = Exit 1 [Value_const (Tagged_immediate 7)] empty_heap no_reads
  ∧ applied 10 (returned_handler below_binder_handler)
      (Value_const (Tagged_immediate 1))
    = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads
  ∧ applied 10 (returned_handler below_binder_result_handler)
      (Value_const (Tagged_immediate 1))
    = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads.

(* The same where the other binder is a parameter of a handler inside
   the handler. *)
Definition below_param_handler : expr :=
  Let_cont_non_recursive 5
    (Handler [12] (make_handler_metadata [untagged_kind])
       (Let 13 Value (tag 12)
          (Let 12 untagged_kind (untag 10)
             (Switch (Var 12) (exits (Var 13) (Var 10))))))
    (Apply_cont (jump 5 [Const (Naked_immediate 7)])).

Definition below_param_bound_statement : Prop :=
  bind_untaggings (returned below_param_handler)
  = returned (Let 12 untagged_kind (untag 10) below_param_handler).

(* ======================================================================== *)
(* Recursive continuations                                                  *)
(* ======================================================================== *)

(* The untagging is in the handler of a recursive continuation, whose
   parameter is neither of the two variables: it is taken away. *)
Definition in_loop : expr :=
  returned
    (Let_cont_recursive
       [(5,
         Handler [15] (make_handler_metadata [Value])
           (Let 12 untagged_kind (untag 10)
              (Switch (Var 12)
                 [(0%Z, jump 1 [Var 15]); (1%Z, jump 5 [Var 15])])))]
       (Apply_cont (jump 5 [Var 2]))).

Definition in_loop_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (Let_cont_recursive
          [(5,
            Handler [15] (make_handler_metadata [Value])
              (Switch (Var 12)
                 [(0%Z, jump 1 [Var 15]); (1%Z, jump 5 [Var 15])]))]
          (Apply_cont (jump 5 [Var 2])))).

Definition in_loop_bound_statement : Prop :=
  bind_untaggings in_loop = in_loop_result.

(* The parameter of the recursive handler is the parameter of the
   enclosing one: the untagging is of another value, and nothing is
   done. *)
Definition in_loop_of_x : expr :=
  returned
    (Let_cont_recursive
       [(5,
         Handler [10] (make_handler_metadata [Value])
           (Let 12 untagged_kind (untag 10)
              (Switch (Var 12)
                 [(0%Z, jump 1 [Var 10]); (1%Z, jump 5 [Var 10])])))]
       (Apply_cont (jump 5 [Var 2]))).

Definition in_loop_of_x_kept_statement : Prop :=
  bind_untaggings in_loop_of_x = in_loop_of_x.

(* Two recursive handlers. The parameter of the second is the variable
   of the untagging: its [Let] stays, and that of the first is taken
   away. *)
Definition in_loops : expr :=
  returned
    (Let_cont_recursive
       [(5,
         Handler [15] (make_handler_metadata [Value])
           (Let 12 untagged_kind (untag 10)
              (Switch (Var 12)
                 [(0%Z, jump 1 [Var 15]); (1%Z, jump 6 [Var 12])])));
        (6,
         Handler [12] (make_handler_metadata [untagged_kind])
           (Let 12 untagged_kind (untag 10)
              (Switch (Var 12)
                 [(0%Z, jump 5 [Var 10]); (1%Z, jump 1 [Var 10])])))]
       (Apply_cont (jump 5 [Var 2]))).

Definition in_loops_result : expr :=
  returned
    (Let 12 untagged_kind (untag 10)
       (Let_cont_recursive
          [(5,
            Handler [15] (make_handler_metadata [Value])
              (Switch (Var 12)
                 [(0%Z, jump 1 [Var 15]); (1%Z, jump 6 [Var 12])]));
           (6,
            Handler [12] (make_handler_metadata [untagged_kind])
              (Let 12 untagged_kind (untag 10)
                 (Switch (Var 12)
                    [(0%Z, jump 5 [Var 10]); (1%Z, jump 1 [Var 10])])))]
          (Apply_cont (jump 5 [Var 2])))).

Definition in_loops_bound_statement : Prop :=
  bind_untaggings in_loops = in_loops_result.

(* The handler that the call returns to is recursive: nothing is bound
   at its head. *)
Definition loop_returned_to : expr :=
  Let_cont_recursive [(3, returned_handler late_handler)] (call 3).

Definition loop_returned_to_kept_statement : Prop :=
  bind_untaggings loop_returned_to = loop_returned_to.

(* A term inside the handler of a recursive continuation: the pass
   leaves it, and rewrite_rec_handlers takes the pass to it. *)
Definition inside_loop (e : expr) : expr :=
  Let_cont_recursive [(5, Handler [] (make_handler_metadata []) e)]
    (Apply_cont (jump 5 [])).

Definition inside_loop_kept_statement : Prop :=
  bind_untaggings (inside_loop late) = inside_loop late.

Definition inside_loop_bound_statement : Prop :=
  rewrite_rec_handlers bind_untaggings (inside_loop late)
  = inside_loop late_result.

(* ======================================================================== *)
(* Where the pass does nothing                                              *)
(* ======================================================================== *)

(* The variable of the untagging has a free use in the handler: a
   [Let] at the head would capture it. *)
Definition free_u : expr :=
  returned
    (Let 13 Value (tag 12)
       (second_call
          (Let 12 untagged_kind (untag 10)
             (Switch (Var 12) (exits (Var 13) (Var 11)))))).

Definition free_u_kept_statement : Prop := bind_untaggings free_u = free_u.

(* The untagging is bound at the kind of values, where it has no
   step: it is not one of those that the pass looks for. *)
Definition wrong_kind : expr :=
  returned
    (second_call
       (Let 12 Value (untag 10) (Apply_cont (jump 1 [Var 12])))).

Definition wrong_kind_kept_statement : Prop :=
  bind_untaggings wrong_kind = wrong_kind.

(* It stays where it is when another untagging, at the kind of naked
   immediates, is bound at the head. *)
Definition wrong_kind_first : expr :=
  returned
    (second_call
       (Let 12 Value (untag 10)
          (Let 13 untagged_kind (untag 10)
             (Switch (Var 13) (exits (Var 11) (Var 12)))))).

Definition wrong_kind_first_result : expr :=
  returned
    (Let 13 untagged_kind (untag 10)
       (second_call
          (Let 12 Value (untag 10)
             (Switch (Var 13) (exits (Var 11) (Var 12)))))).

Definition wrong_kind_first_bound_statement : Prop :=
  bind_untaggings wrong_kind_first = wrong_kind_first_result.

(* The term [late] with other parameters and metadata for the handler
   that the first call returns to. *)
Definition late_with (params : list variable) (metadata : handler_metadata)
    : expr :=
  Let_cont_non_recursive 3 (Handler params metadata late_handler) (call 3).

(* The variable of the untagging is a parameter of the handler. *)
Definition u_is_param_kept_statement : Prop :=
  let e :=
    late_with [10; 12]
      (Handler_metadata [Value; untagged_kind] [tagged] false)
  in
  bind_untaggings e = e.

(* The parameters are not distinct. *)
Definition twice_param_kept_statement : Prop :=
  let e :=
    late_with [10; 10] (Handler_metadata [Value; Value] [tagged; tagged] false)
  in
  bind_untaggings e = e.

(* The handler is an exception handler. *)
Definition exn_handler_kept_statement : Prop :=
  let e := late_with [10] (Handler_metadata [Value] [tagged] true) in
  bind_untaggings e = e.

(* No call returns to the handler: it is applied by a jump. *)
Definition jumped_to : expr :=
  Let_cont_non_recursive 3 (returned_handler late_handler)
    (Apply_cont (jump 3 [Var 2])).

Definition jumped_to_kept_statement : Prop :=
  bind_untaggings jumped_to = jumped_to.

(* The parameter may be null. *)
Definition nullable_kept_statement : Prop :=
  let e :=
    late_with [10]
      (Handler_metadata [Value] [(true, Subkind_tagged_immediate)] false)
  in
  bind_untaggings e = e.

(* The parameter is of a variant that has a block. *)
Definition variant_with_block : param_subkind :=
  (false, Subkind_variant [0%Z; 1%Z] (Shape_undetermined 0 Shapes_nil)).

Definition variant_with_block_kept_statement : Prop :=
  let e :=
    late_with [10] (Handler_metadata [Value] [variant_with_block] false)
  in
  bind_untaggings e = e.

(* Nothing is stated of the parameter. *)
Definition nothing_stated_kept_statement : Prop :=
  let e := late_with [10] (make_handler_metadata [Value]) in
  bind_untaggings e = e.

(* The parameter is of a variant with constant constructors only: the
   pass acts as for a tagged immediate. *)
Definition constants_only : param_subkind :=
  (false, Subkind_variant [0%Z; 1%Z] Shapes_nil).

Definition constants_only_bound_statement : Prop :=
  bind_untaggings
    (late_with [10] (Handler_metadata [Value] [constants_only] false))
  = Let_cont_non_recursive 3
      (Handler [10] (Handler_metadata [Value] [constants_only] false)
         (Let 12 untagged_kind (untag 10)
            (second_call (Switch (Var 12) (exits (Var 11) (Var 10))))))
      (call 3).

(* ======================================================================== *)
(* What the machine accepts at the binder                                   *)
(* ======================================================================== *)

(* The least integer above the range of a tagged immediate. *)
Definition above_range : Z := 2 ^ (int_width Standard_tagged_immediate - 1).

(* The source handler is entered with the tagged immediate 1 and
   with the largest in range; the rewritten handler is also
   entered with the largest in range. *)
Definition entered_in_range_statement : Prop :=
  enters (returned_handler late_handler) (Value_const (Tagged_immediate 1))
  = true
  ∧ enters (returned_handler late_handler)
      (Value_const (Tagged_immediate (above_range - 1)))
    = true
  ∧ enters
      (returned_handler
         (Let 12 untagged_kind (untag 10)
            (second_call (Switch (Var 12) (exits (Var 11) (Var 10))))))
      (Value_const (Tagged_immediate (above_range - 1)))
    = true.

(* The source handler is not entered with null, poison, the tagged
   immediate just above the range, or the displayed block. *)
Definition not_entered_statement : Prop :=
  let h := returned_handler late_handler in
  enters h (Value_const Null) = false
  ∧ enters h (Value_const (Poison Value)) = false
  ∧ enters h (Value_const (Tagged_immediate above_range)) = false
  ∧ enters h (Value_block 0 []) = false.

(* The untagging has a value on what is accepted, and none on the
   others. *)
Definition untagging_defined_statement : Prop :=
  let untagging := eval_unary_primitive empty_static_env Untag_immediate in
  untagging (Value_const (Tagged_immediate (above_range - 1)))
  = Some (Value_const (Naked_immediate (above_range - 1)))
  ∧ untagging (Value_const Null) = None
  ∧ untagging (Value_const (Poison Value)) = None
  ∧ untagging (Value_const (Tagged_immediate above_range)) = None
  ∧ untagging (Value_block 0 []) = None.

(* A variant with constant constructors only: one of its constructors
   is accepted, and another tagged immediate is not. *)
Definition constants_only_accepted_statement : Prop :=
  let accepted := value_at empty_static_env constants_only in
  accepted (Value_const (Tagged_immediate 1)) = true
  ∧ accepted (Value_const (Tagged_immediate 2)) = false.

(* ======================================================================== *)
(* Which calls return to a handler                                          *)
(* ======================================================================== *)

(* The body of the handler of [late_result]. *)
Definition late_result_handler : expr :=
  Let 12 untagged_kind (untag 10)
    (second_call (Switch (Var 12) (exits (Var 11) (Var 10)))).

(* Between the handler and the first call is a non-recursive
   continuation of the same name, 3, that is applied by a jump. The
   call is in its handler, where the name is still that of the outer
   continuation: the pass rewrites the outer handler. *)
Definition seen_in_handler (handler : expr) : expr :=
  Let_cont_non_recursive 3 (returned_handler handler)
    (Let_cont_non_recursive 3
       (Handler [15] (make_handler_metadata [Value]) (call 3))
       (Apply_cont (jump 3 [Var 2]))).

Definition seen_in_handler_bound_statement : Prop :=
  bind_untaggings (seen_in_handler late_handler)
  = seen_in_handler late_result_handler.

(* The call is in the body of the inner [Let_cont], where the name is
   that of the inner continuation: the pass rewrites the inner handler
   and leaves the outer one. *)
Definition hidden_in_body_bound_statement : Prop :=
  bind_untaggings
    (Let_cont_non_recursive 3 (returned_handler late_handler) late)
  = Let_cont_non_recursive 3 (returned_handler late_handler) late_result.

(* The inner continuation is recursive. Its name is its own in its
   handler and in the body of its [Let_cont]: the pass leaves the
   outer handler wherever the call is. *)
Definition hidden_in_recursive_handler : expr :=
  Let_cont_non_recursive 3 (returned_handler late_handler)
    (Let_cont_recursive
       [(3, Handler [15] (make_handler_metadata [Value]) (call 3))]
       (Apply_cont (jump 3 [Var 2]))).

Definition hidden_in_recursive_handler_kept_statement : Prop :=
  bind_untaggings hidden_in_recursive_handler = hidden_in_recursive_handler.

Definition hidden_in_recursive_body : expr :=
  Let_cont_non_recursive 3 (returned_handler late_handler)
    (Let_cont_recursive
       [(3,
         Handler [15] (make_handler_metadata [Value])
           (Apply_cont (jump 1 [Var 15])))]
       (call 3)).

Definition hidden_in_recursive_body_kept_statement : Prop :=
  bind_untaggings hidden_in_recursive_body = hidden_in_recursive_body.

(* A recursive continuation of another name, 5, hides nothing: the
   call in its handler returns to the continuation 3, and the pass
   rewrites the handler of that one. *)
Definition seen_in_recursive_handler (handler : expr) : expr :=
  Let_cont_non_recursive 3 (returned_handler handler)
    (Let_cont_recursive
       [(5, Handler [15] (make_handler_metadata [Value]) (call 3))]
       (Apply_cont (jump 5 [Var 2]))).

Definition seen_in_recursive_handler_bound_statement : Prop :=
  bind_untaggings (seen_in_recursive_handler late_handler)
  = seen_in_recursive_handler late_result_handler.

(* ======================================================================== *)
(* Two parameters that state a tagged immediate                             *)
(* ======================================================================== *)

(* The handler that the first call returns to has the parameters 10
   and 15, and each states a tagged immediate. *)
Definition two_tagged_handler (handler : expr) : continuation_handler :=
  Handler [10; 15] (Handler_metadata [Value; Value] [tagged; tagged] false)
    handler.

Definition returned_two (handler : expr) : expr :=
  Let_cont_non_recursive 3 (two_tagged_handler handler) (call 3).

(* The second parameter is untagged before the first, to two
   variables: both untaggings are bound at the head, that of the first
   parameter outermost. *)
Definition two_params : expr :=
  returned_two
    (second_call
       (Let 16 untagged_kind (untag 15)
          (Let 12 untagged_kind (untag 10)
             (Let 13 Value (tag 16)
                (Switch (Var 12) (exits (Var 13) (Var 11))))))).

Definition two_params_result : expr :=
  returned_two
    (Let 12 untagged_kind (untag 10)
       (Let 16 untagged_kind (untag 15)
          (second_call
             (Let 13 Value (tag 16)
                (Switch (Var 12) (exits (Var 13) (Var 11))))))).

Definition two_params_bound_statement : Prop :=
  bind_untaggings two_params = two_params_result.

(* The two parameters are untagged to one variable, 12, in two places
   of which neither is in the scope of the other: the first parameter
   in the body of a [Let_cont], and the second in its handler. *)
Definition reused_handler : expr :=
  Let_cont_non_recursive 5
    (Handler [] (make_handler_metadata [])
       (Let 12 untagged_kind (untag 15)
          (Switch (Var 12) (exits (Var 10) (Var 15)))))
    (Let 12 untagged_kind (untag 10)
       (Switch (Var 12) [(0%Z, jump 5 []); (1%Z, jump 1 [Var 10])])).

(* The untagging of the second parameter is bound at the head, and
   its [Let] in the handler of the continuation 5 is taken away. That
   of the first parameter is bound above it, where nothing reads it:
   the other hides it at once, and the untagging of the first
   parameter stays where it was. *)
Definition reused_result_handler : expr :=
  Let 12 untagged_kind (untag 10)
    (Let 12 untagged_kind (untag 15)
       (Let_cont_non_recursive 5
          (Handler [] (make_handler_metadata [])
             (Switch (Var 12) (exits (Var 10) (Var 15))))
          (Let 12 untagged_kind (untag 10)
             (Switch (Var 12) [(0%Z, jump 5 []); (1%Z, jump 1 [Var 10])])))).

Definition reused_bound_statement : Prop :=
  bind_untaggings (returned_two reused_handler)
  = returned_two reused_result_handler.

(* What the machine does where a handler is applied to two values
   under no environment, as [applied] for one. *)
Definition applied_to_two (fuel : nat) (h : continuation_handler)
    (v w : value) : step_result :=
  match
    apply_cont_value (Closure h [] [] []) [v; w] empty_static_env empty_heap
      no_reads
  with
  | Step s => run fuel s
  | result => result
  end.

(* Applied to the tagged immediates 0 and 1, both handlers return the
   second, which is what the handler of the continuation 5 does where
   the variable 12 is the untagging of the second parameter; applied
   to 1 and 0, both return the first. *)
Definition reused_runs_statement : Prop :=
  applied_to_two 12 (two_tagged_handler reused_handler)
    (Value_const (Tagged_immediate 0)) (Value_const (Tagged_immediate 1))
  = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads
  ∧ applied_to_two 12 (two_tagged_handler reused_result_handler)
      (Value_const (Tagged_immediate 0)) (Value_const (Tagged_immediate 1))
    = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads
  ∧ applied_to_two 12 (two_tagged_handler reused_handler)
      (Value_const (Tagged_immediate 1)) (Value_const (Tagged_immediate 0))
    = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads
  ∧ applied_to_two 12 (two_tagged_handler reused_result_handler)
      (Value_const (Tagged_immediate 1)) (Value_const (Tagged_immediate 0))
    = Exit 1 [Value_const (Tagged_immediate 1)] empty_heap no_reads.

(* The pass applied to its result binds both untaggings at the head
   once more, and takes nothing away: on this term each application
   adds two [Let]s that nothing reads. *)
Definition reused_again_statement : Prop :=
  bind_untaggings (returned_two reused_result_handler)
  = returned_two
      (Let 12 untagged_kind (untag 10)
         (Let 12 untagged_kind (untag 15) reused_result_handler)).
