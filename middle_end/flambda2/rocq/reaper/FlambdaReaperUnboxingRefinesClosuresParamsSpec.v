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

(* A unit whose code takes a parameter that the rebuild unboxes, beside a
   set of closures whose representation changes, and the relation of the
   proofs of FlambdaReaperUnboxingRefinesClosuresSpec.v
   (FlambdaReaperUnboxingRefinesClosuresSimProofs.rr) relating its two
   rebuilds.

   This file is to be reviewed: the example and its statement. The proof
   is in FlambdaReaperUnboxingRefinesClosuresParamsProofs.v, which needs
   no review.

   The relation describes the parameters of code by their shapes in the
   second run ([ci_params]: the same, or the variables of the fields of
   an unboxed one) and relates the arguments of a direct call at those
   shapes ([args_frame]): the second run gives the variables of the fields
   of an unboxed name for an unboxed parameter, and the code enters with
   those variables bound to the fields of what the first run gives
   ([call_code_entered_u]). The fragments of
   FlambdaReaperUnboxingRefinesClosuresSpec.v and of the files that widen
   it have no unboxed parameter; the construction of the relation for
   them gives every code plain shapes. A fragment with unboxed parameters,
   and the construction of the relation for it, are left; the example
   relates the two rebuilds of one unit by the relation built by hand.

   [repr_params_example_statement] ([ParamsExample]): the unit of
   FlambdaReaperUnboxingRefinesClosuresSpec.TwoFieldsExample, whose code
   54 also reads the field 0 of its parameter [x40], called with the
   block [x11] of 32 and 5. With the least tables, the rules unbox [x10]
   (stored in the slot of the set of [x60], which changes its
   representation) and [x11], which dominates the parameter [x40]: the
   code 54 changes its calling convention, its parameter unboxed into
   the variable 63 of its field 0. The unit is not of the widest fragment
   ([repr_over_unboxing_nr], for any code taken to never return, any flow
   and any code of what is left of an over-application). The unit
   rebuilt with the change of calling conventions alone
   ([first_rebuild]) is [first_rebuilt]; the rebuild in the mode of the
   fix is [rebuilt], which binds the variable 67 of the field 0 of [x11]
   to 32 and gives it to the call; the unit and the rebuilt unit exit
   with 47; the rebuilt unit refines the first rebuild. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec FlambdaReaperUnboxingRefinesSpec
  FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperUnboxingRefinesReachSpec
  FlambdaReaperUnboxingRefinesClosuresOverSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast.
Import ListNotations.

(* ======================================================================== *)
(* The example                                                              *)
(* ======================================================================== *)

Module ParamsExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition add (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate Add)) [a; b].

(* The code 54 of TwoFieldsExample, which also reads the field 0 of its
   parameter. *)
Definition code_54 : code :=
  Code [40] 41 3 4 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 45 Value (Project_value_slot_from 70 80 (Var 41))
       (Let 46 Value (load 0 (Var 45))
          (Let 47 Value (load 1 (Var 45))
             (Let 48 Value (add (Var 46) (Var 47))
                (Let 50 Value (load 0 (Var 40))
                   (Let 49 Value (add (Var 48) (Var 50))
                      (return_to 3 [Var 49]))))))).

Definition call (args : list simple) (a : arity) : apply_expr :=
  {| ap_callee := Var 60; ap_call_kind := Direct 54; ap_args := args;
     ap_continuation := 1; ap_exn_continuation := 2;
     ap_exn_extra_args := [];
     ap_args_arity := a;
     ap_return_arity := a |}.

Definition unit_expr : expr :=
  Let_static [Bind_code 54 code_54]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 11 Value (make_block [int 32; int 5])
          (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
             (Apply (call [Var 11] [Singleton Value]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 40 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

Definition code_deps : finite_map code_dep :=
  unit_code_deps Auto interface unit_expr.

(* The metadata of the code in both rebuilds, and the arity of the
   call. *)
Definition arity1 : arity := [Unboxed_product [Singleton Value]].

Definition metadata1 : code_metadata :=
  {| cm_params_arity := arity1;
     cm_params_subkinds := [];
     cm_result_arity := arity1;
     cm_result_subkinds := [];
     cm_is_tupled := false;
     cm_is_stub := false |}.

(* The unit rebuilt with the change of calling conventions alone: the
   field 1 of [x11], which nothing reads, is poison. *)
Definition first_rebuilt : expr :=
  Let_static
    [Bind_code 54
       (Code [40] 41 3 4 metadata1
          (Let 45 Value (Project_value_slot_from 70 80 (Var 41))
             (Let 46 Value (load 0 (Var 45))
                (Let 47 Value (load 1 (Var 45))
                   (Let 48 Value (add (Var 46) (Var 47))
                      (Let 50 Value (load 0 (Var 40))
                         (Let 49 Value (add (Var 48) (Var 50))
                            (return_to 3 [Var 49]))))))))]
    (Let 10 Value (make_block [int 7; int 8])
       (Let 11 Value (make_block [int 32; Const (Poison Value)])
          (Let 60 Value (Closure_of_set 70 [(70, 54)] [(80, Var 10)])
             (Apply (call [Var 11] arity1))))).

(* The rebuild in the mode of the fix: the parameter of the code is the
   variable 63 of the field 0 of [x40], which it reads in place of the
   load; [x10] is the variables 65 and 66, the new slots 81 and 82 of the
   set, which the code reads into the variables 61 and 62 of the fields
   of [x45]; [x11] is the variable 67, given to the call. *)
Definition rebuilt : expr :=
  Let_static
    [Bind_code 54
       (Code [63] 41 3 4 metadata1
          (Let 62 Value (Project_value_slot_from 83 82 (Var 41))
             (Let 61 Value (Project_value_slot_from 83 81 (Var 41))
                (Let 46 Value (Simple (Var 61))
                   (Let 47 Value (Simple (Var 62))
                      (Let 48 Value (add (Var 46) (Var 47))
                         (Let 50 Value (Simple (Var 63))
                            (Let 49 Value (add (Var 48) (Var 50))
                               (return_to 3 [Var 49])))))))))]
    (Let 66 Value (Simple (int 8))
       (Let 65 Value (Simple (int 7))
          (Let 67 Value (Simple (int 32))
             (Let 60 Value
                (Closure_of_set 83 [(83, 54)] [(81, Var 65); (82, Var 66)])
                (Apply (call [Var 67] arity1)))))).

End ParamsExample.

(* ======================================================================== *)
(* The statement                                                            *)
(* ======================================================================== *)

Definition repr_params_example_statement : Prop :=
  (∀ (nr : code_id -> bool) (reach : bool -> variable -> code_id -> bool)
     (rem : code_id -> bool),
     repr_over_unboxing_nr nr Examples.no_local ParamsExample.facts
       ParamsExample.tables [] ParamsExample.code_deps
       ParamsExample.dominated ParamsExample.to_unbox ParamsExample.tcr reach
       rem ParamsExample.unit_expr
     = false)
  ∧ check_unboxed_reads ParamsExample.facts ParamsExample.tables
      ParamsExample.dominated ParamsExample.to_unbox
    = true
  ∧ first_rebuild Auto Examples.interface ParamsExample.unit_expr
      ParamsExample.tables [] true
    = ParamsExample.first_rebuilt
  ∧ rebuild_unit_unbox_repr_fixed Examples.no_local ParamsExample.facts
      ParamsExample.tables [] ParamsExample.code_deps ParamsExample.dominated
      ParamsExample.to_unbox ParamsExample.tcr true ParamsExample.unit_expr
    = ParamsExample.rebuilt
  ∧ run 40
      (State ParamsExample.unit_expr [] [] empty_static_env empty_heap
         no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 47%Z)] [] []
  ∧ run 40
      (State ParamsExample.rebuilt [] [] empty_static_env empty_heap no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 47%Z)] [] []
  ∧ program_refines ParamsExample.first_rebuilt ParamsExample.rebuilt.
