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

(* The two shapes of calls whose code is not known and which never return
   that the fragment of FlambdaReaperUnboxingRefinesNeverSpec.v refuses,
   each on a unit: the analysis of the fix finds the call one that never
   returns, the rebuild of the fix makes it return to a handler [Invalid],
   and the rebuilt unit refines the unit, the runs computed.

   This file is to be reviewed: the two units and the statements. The
   proofs are in FlambdaReaperUnboxingRefinesNeverShapesProofs.v, which
   needs no review.

   The shapes:
   - A call of unknown arity that completes a partial application of code
     with more parameters than the call has groups of arguments
     ([never_partial_example_statement]). The codes 50 and 51 take two
     parameters and raise the first; the unit applies the closure of one
     of them (chosen by a [Switch]) to 5, which makes a partial
     application [x32], and applies [x32] to 6, which completes it: the
     code is entered and raises 5. [never_code_ok] asks the code that the
     flow brings to [x32] (50 and 51, as the code of the closure of a
     partial application) to have no more parameters than the call has
     groups (one), and refuses the call.
   - An over-application whose first code returns and whose remainder
     never returns ([never_over_example_statement]). The codes 52 and 53
     take one parameter and return a closure of 51, which raises its
     parameter; the unit applies the closure of 52 or of 53 to 5 and 6:
     the code returns the closure of 51, which the remainder applies to 6,
     and 51 raises 6. [never_code_ok] asks the code that the flow brings
     to the callee (52 and 53) to be code that never returns, and refuses
     the call.
   In both, [apply_never_returns] of the fix holds of the call (it follows
   the witnesses of unknown arity group by group, through the chain of a
   partial application and through the first result of the code
   over-applied, to the results of code with no source), the code taken
   to never return ([used_never_returns]) is 50 and 51, resp. 51, and the
   fragment refuses the unit, read off the first rebuilt unit or off the
   unit's own analysis. The rebuild of the fix gives the call a handler
   [Invalid] and drops the handler 20 of the unit, which the change of
   calling conventions alone keeps, [Invalid]. The unit, its first
   rebuild and its rebuild in the mode of the fix exit alike (with the
   exception 5, resp. 6), so that the rebuilt unit refines both.

   Both shapes are sound in the model: the call enters code that never
   returns, by completing the partial application or by the remainder of
   the over-application, and the frame that the first run gives it is
   never applied. They are not admitted in the fragment here, as the case
   of such a call in the relation of the proofs is in
   FlambdaReaperUnboxingRefinesCallsSimProofs.v and its construction in
   FlambdaReaperUnboxingRefinesCallsProofs.v, which this change leaves as
   they are:
   - the first shape needs [never_targets] to compare the parameters of
     the code of the closure of a partial application with the groups the
     call gives and those the partial application holds, and a run
     invariant that gives a bound on the latter (the flow puts the
     partial application at the allocation of the first result of the
     [k]-th witness of the code; its depth in the chain of witnesses is
     the bound);
   - the second needs a case of [cont_rel] for an over-application whose
     two continuations are not related, applied only where what is left of
     it enters code that never returns, and a run invariant that ties
     what is left of each over-application to the code over-applied
     ([step_rem_ok] ties it to [REM], the code that what is left of any
     over-application may call, which is too coarse for a check per
     call).

   FlambdaReaperUnboxingRefinesNever2Spec.v admits the first shape where
   the closure of the code is never brought to the callee itself (only
   partial applications of it, which hold a group of arguments at least),
   and the code has one parameter more than the call has groups: the unit
   of [never_partial_example_statement] is of its fragments, and its
   rebuilt unit refines the unit by its statements
   ([never_partial2_example_statement], below). The second shape is not
   admitted. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperLeast FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesReach2Spec
  FlambdaReaperUnboxingRefinesReach3Spec
  FlambdaReaperUnboxingRefinesNeverSpec
  FlambdaReaperUnboxingRefinesUnitFlowSpec
  FlambdaReaperUnboxingRefinesNever2Spec.
From OxCamlFlambda2 Require FlambdaReaperCallingConventionsSpec
  FlambdaReaperNeverReturnsSpec.
Import ListNotations.

(* ======================================================================== *)
(* Terms of both units                                                      *)
(* ======================================================================== *)

Module NeverShapes.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

(* A call of [callee] of unknown arity, one group for each argument, which
   raises to the unit's exception continuation. *)
Definition call (callee : variable) (args : list simple) (k : continuation)
    : apply_expr :=
  {| ap_callee := Var callee; ap_call_kind := Indirect_unknown_arity;
     ap_args := args; ap_continuation := k; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := map (fun _ => Singleton Value) args;
     ap_return_arity := [Singleton Value] |}.

(* Code of [n] parameters of kind [Value] and one result. *)
Definition metadata (n : nat) : code_metadata :=
  make_code_metadata (repeat (Singleton Value) n) [Singleton Value].

Definition jump (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

(* [x] raised to the exception continuation [exn] of the code. *)
Definition raise (exn : continuation) (x : variable) : expr :=
  Apply_cont {| ac_continuation := exn; ac_args := [Var x];
                ac_trap_action := Some (Pop exn None) |}.

(* The handler [k] of the parameter [x], running [handler], over [body]. *)
Definition handler (k : continuation) (x : variable) (handler : expr)
    (body : expr) : expr :=
  Let_cont_non_recursive k
    (Handler [x] (make_handler_metadata [Value]) handler) body.

(* The closure [x] of the set [slot] of the code [id], returned to [k]. *)
Definition return_closure (id : code_id) (x : variable)
    (slot : function_slot) (k : continuation) : expr :=
  Let x Value (Closure_of_set slot [(slot, id)] []) (return_to k [Var x]).

(* The call [ap], returning to a handler [Invalid] [w] of the parameter
   [w]. *)
Definition invalid_call (w : continuation) (ap : apply_expr) : expr :=
  handler w w Invalid (Apply (with_continuation ap w)).

(* The handler 20 that returns its parameter, and the one that the change
   of calling conventions makes of it, its parameter having no source. *)
Definition returns_20 (body : expr) : expr :=
  handler 20 30 (return_to 1 [Var 30]) body.

Definition invalid_20 (body : expr) : expr := handler 20 30 Invalid body.

(* The closure [x60] of 50 or [x61] of 51 given to the handler 21. *)
Definition choose_21 : expr :=
  Switch (Const (Naked_immediate 0))
    [(0%Z, jump 21 [Var 60]); (1%Z, jump 21 [Var 61])].

End NeverShapes.

(* ======================================================================== *)
(* A call that completes a partial application                              *)
(* ======================================================================== *)

(* The codes 50 and 51 take two parameters and raise the first. In the
   handler 21, [x31] (the closure of 50 or of 51) is applied to 5, which
   makes the partial application [x32] given to the handler 22; there
   [x32] is applied to 6, which completes it, returning to 20. *)
Module PartialNeverExample.

Import FlambdaReaperUnboxing.Examples NeverShapes.
Local Open Scope positive_scope.

Definition partial_call : apply_expr := call 31 [int 5] 22.

Definition completing_call : apply_expr := call 32 [int 6] 20.

(* The unit, with the handler 20 [around] the rest and the call in 22
   [call_22]. *)
Definition unit_with (around : expr -> expr) (call_22 : expr) : expr :=
  Let_static
    [Bind_code 50 (Code [40; 44] 41 3 4 (metadata 2) (raise 4 40));
     Bind_code 51 (Code [42; 45] 43 5 6 (metadata 2) (raise 6 42))]
    (Let 60 Value (Closure_of_set 70 [(70, 50)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 51)] [])
          (around
             (handler 22 32 call_22
                (handler 21 31 (Apply partial_call) choose_21))))).

Definition unit_expr : expr := unit_with returns_20 (Apply completing_call).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition nr : code_id -> bool :=
  FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables tables)
    code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

(* The change of calling conventions alone: the handler 20 is kept,
   [Invalid]. *)
Definition cc_rebuilt : expr := unit_with invalid_20 (Apply completing_call).

(* The rebuild in the mode of the fix: the completing call returns to the
   handler [Invalid] 62, and the handler 20 is dropped. *)
Definition rebuilt : expr :=
  unit_with (fun body => body) (invalid_call 62 completing_call).

End PartialNeverExample.

Definition never_partial_example_statement : Prop :=
  map
    (apply_never_returns PartialNeverExample.facts
       (solution_of_tables PartialNeverExample.tables)
       PartialNeverExample.code_deps
       (bound_code_metadata PartialNeverExample.unit_expr))
    [PartialNeverExample.completing_call; PartialNeverExample.partial_call]
  = [true; false]
  ∧ map PartialNeverExample.nr [50; 51]%positive = [true; true]
  ∧ calls_unboxing_never_nr PartialNeverExample.nr PartialNeverExample.facts
      PartialNeverExample.tables PartialNeverExample.cannot_change
      PartialNeverExample.code_deps [] [] PartialNeverExample.facts1
      PartialNeverExample.tables1 PartialNeverExample.unit_expr
    = false
  ∧ calls_unboxing_unit_never_nr PartialNeverExample.nr
      PartialNeverExample.facts PartialNeverExample.tables
      PartialNeverExample.cannot_change PartialNeverExample.code_deps [] []
      PartialNeverExample.unit_expr
    = false
  ∧ PartialNeverExample.first = PartialNeverExample.cc_rebuilt
  ∧ rebuild_unit_unbox_fixed PartialNeverExample.facts
      PartialNeverExample.tables PartialNeverExample.cannot_change
      PartialNeverExample.code_deps [] [] true PartialNeverExample.unit_expr
    = PartialNeverExample.rebuilt
  ∧ program_refines PartialNeverExample.first PartialNeverExample.rebuilt
  ∧ program_refines_observably PartialNeverExample.unit_expr
      PartialNeverExample.rebuilt.

(* ======================================================================== *)
(* An over-application whose remainder never returns                        *)
(* ======================================================================== *)

(* The code 51 raises its parameter; 52 and 53 return a closure of 51. In
   the handler 21, [x31] (the closure of 52 or of 53) is applied to 5 and
   6, returning to 20: the code returns the closure of 51, which what is
   left of the over-application applies to 6. *)
Module OverNeverExample.

Import FlambdaReaperUnboxing.Examples NeverShapes.
Local Open Scope positive_scope.

Definition over_call : apply_expr := call 31 [int 5; int 6] 20.

(* The unit, with the handler 20 [around] the rest and the call in 21
   [call_21]. *)
Definition unit_with (around : expr -> expr) (call_21 : expr) : expr :=
  Let_static
    [Bind_code 51 (Code [42] 43 5 6 (metadata 1) (raise 6 42));
     Bind_code 52 (Code [44] 45 7 8 (metadata 1) (return_closure 51 63 73 7));
     Bind_code 53
       (Code [46] 47 9 10 (metadata 1) (return_closure 51 64 74 9))]
    (Let 60 Value (Closure_of_set 70 [(70, 52)] [])
       (Let 61 Value (Closure_of_set 71 [(71, 53)] [])
          (around (handler 21 31 call_21 choose_21)))).

Definition unit_expr : expr := unit_with returns_20 (Apply over_call).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node :=
  FlambdaReaperCallingConventions.derive_cannot_change facts tables.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

Definition nr : code_id -> bool :=
  FlambdaReaperNeverReturnsSpec.used_never_returns (solution_of_tables tables)
    code_deps.

Definition first : expr :=
  first_rebuild Auto interface unit_expr tables cannot_change true.

Definition facts1 : FlambdaReaperGraphSpec.facts :=
  traverse_unit Auto interface first.

Definition tables1 : FlambdaReaperGraphSpec.tables :=
  match compute_least no_local facts1 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cc_rebuilt : expr := unit_with invalid_20 (Apply over_call).

(* The over-application returns to the handler [Invalid] 65, and the
   handler 20 is dropped. *)
Definition rebuilt : expr :=
  unit_with (fun body => body) (invalid_call 65 over_call).

End OverNeverExample.

Definition never_over_example_statement : Prop :=
  apply_never_returns OverNeverExample.facts
    (solution_of_tables OverNeverExample.tables) OverNeverExample.code_deps
    (bound_code_metadata OverNeverExample.unit_expr) OverNeverExample.over_call
  = true
  ∧ map OverNeverExample.nr [51; 52; 53]%positive = [true; false; false]
  ∧ calls_unboxing_never_nr OverNeverExample.nr OverNeverExample.facts
      OverNeverExample.tables OverNeverExample.cannot_change
      OverNeverExample.code_deps [] [] OverNeverExample.facts1
      OverNeverExample.tables1 OverNeverExample.unit_expr
    = false
  ∧ calls_unboxing_unit_never_nr OverNeverExample.nr OverNeverExample.facts
      OverNeverExample.tables OverNeverExample.cannot_change
      OverNeverExample.code_deps [] [] OverNeverExample.unit_expr
    = false
  ∧ OverNeverExample.first = OverNeverExample.cc_rebuilt
  ∧ rebuild_unit_unbox_fixed OverNeverExample.facts OverNeverExample.tables
      OverNeverExample.cannot_change OverNeverExample.code_deps [] [] true
      OverNeverExample.unit_expr
    = OverNeverExample.rebuilt
  ∧ program_refines OverNeverExample.first OverNeverExample.rebuilt
  ∧ program_refines_observably OverNeverExample.unit_expr
      OverNeverExample.rebuilt.

(* ======================================================================== *)
(* The first shape, admitted                                                *)
(* ======================================================================== *)

(* The unit of [never_partial_example_statement] is of the fragments of
   FlambdaReaperUnboxingRefinesNever2Spec.v, read off the first rebuilt
   unit and off the unit's own analysis, with the least solutions and the
   code of [used_never_returns]: its rebuilt unit refines its first rebuild
   and the unit up to observation, by the statements of that file. *)
Definition never_partial2_example_statement : Prop :=
  calls_unboxing_never2_nr PartialNeverExample.nr PartialNeverExample.facts
    PartialNeverExample.tables PartialNeverExample.cannot_change
    PartialNeverExample.code_deps [] [] PartialNeverExample.facts1
    PartialNeverExample.tables1 PartialNeverExample.unit_expr
  = true
  ∧ calls_unboxing_unit_never2_nr PartialNeverExample.nr
      PartialNeverExample.facts PartialNeverExample.tables
      PartialNeverExample.cannot_change PartialNeverExample.code_deps [] []
      PartialNeverExample.unit_expr
    = true
  ∧ program_refines PartialNeverExample.first
      (rebuild_unit_unbox_fixed PartialNeverExample.facts
         PartialNeverExample.tables PartialNeverExample.cannot_change
         PartialNeverExample.code_deps [] [] true
         PartialNeverExample.unit_expr)
  ∧ program_refines_observably PartialNeverExample.unit_expr
      (rebuild_unit_unbox_fixed PartialNeverExample.facts
         PartialNeverExample.tables PartialNeverExample.cannot_change
         PartialNeverExample.code_deps [] [] true
         PartialNeverExample.unit_expr).
