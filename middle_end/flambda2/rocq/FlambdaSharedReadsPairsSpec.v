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

(* Two pairs of units given to the validation, computed: one that the
   sharing of reads (FlambdaSharedReadsSpec.share_static_reads) makes
   it accept, and one that it must refuse.

   This file is to be reviewed: the two units and the two statements.
   It is apart from FlambdaSharedReadsSpec.v because the validation
   (FlambdaValidateSpec.validate_unit) comes after that file.

   In each pair the unit binds [x] and [y] to two values that are not
   known (the identity that nothing folds, of 1 and of 2), then one
   group: the code [reader], a closure of that code at the symbol
   [closure] whose value slot holds [x], and a block at the symbol
   [block]. The code reads the value slot of the closure and the first
   field of the block, from the two symbols, and returns the two
   values; the unit calls it and exits with what it returns. The term
   after is the same unit with the second read made the variable of
   the first, which is what Simplify does where the two places hold
   one variable. The validation is given, as the test tool gives it
   where the term after still calls the code, a lifting that lifts
   nothing and the code to be left where it is called.

   - Where the field holds [x] too, the pair is validated, by the
     comparison of the lifted terms ([shared_reads_validated_statement]).
     That is the sharing of reads at work: the term before keeps its
     two reads through every other rewrite of the validation.
   - Where the field holds [y], the same term after is another
     program: the unit before exits with 1 and 2, and the term after
     with 1 and 1. The two places are of two classes, the term before
     keeps its two reads, and the pair is not validated
     ([wrong_sharing_refused_statement]).

   Nothing here is used by the validation.

   The proofs are in FlambdaSharedReadsPairsProofs.v, which needs no
   review: each is a computation. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaValueSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaUnusedStaticsSpec
  FlambdaLiftSpec FlambdaValidateSpec.
Import ListNotations.

Section Pairs.

Let x : variable := 1%positive.
Let y : variable := 2%positive.
Let arg : variable := 3%positive.
Let my_closure : variable := 4%positive.
Let first_read : variable := 5%positive.
Let second_read : variable := 6%positive.
Let ret : continuation := 1%positive.
Let exn : continuation := 2%positive.
Let done : continuation := 3%positive.
Let failed : continuation := 4%positive.
Let reader : code_id := 1%positive.
Let closure : symbol := 1%positive.
Let block : symbol := 2%positive.
Let slot : value_slot := 1%positive.

(* The load of the first field of the block, from its symbol. *)
Definition pair_field_read : named :=
  Prim
    (Unary (Block_load (Block_access_values None None Any_value) Immutable 0))
    [Symbol block].

(* The code: the projection of the value slot of the closure, from its
   symbol, then the second read as given; it returns what the two
   bound. *)
Definition pair_code (second : named) : code :=
  Code [arg] my_closure ret exn
    (make_code_metadata [Singleton Value] [Singleton Value; Singleton Value])
    (Let first_read Value (Project_value_slot slot (Symbol closure))
       (Let second_read Value second
          (Apply_cont
             (Apply_cont_expr ret [Var first_read; Var second_read] None)))).

(* The unit: the code with the second read given, the closure with [x]
   in its value slot, the block with the field given; then the call of
   the code, whose results are those of the unit. *)
Definition pair_unit (second : named) (in_field : simple) : expr :=
  Let x Value
    (Prim (Unary (Opaque_identity Value)) [Const (Tagged_immediate 1)])
    (Let y Value
       (Prim (Unary (Opaque_identity Value)) [Const (Tagged_immediate 2)])
       (Let_static
          [Bind_code reader (pair_code second);
           Bind_symbol closure
             (Static_const_set_of_closures reader [(slot, Var x)]);
           Bind_symbol block
             (Static_const_block 0 [in_field; Symbol closure])]
          (Apply
             (Apply_expr (Symbol closure) (Direct reader)
                [Const (Tagged_immediate 0)] done failed []
                [Singleton Value] [Singleton Value; Singleton Value])))).

(* The second read made the variable of the first. *)
Definition pair_shared : named := Simple (Var first_read).

(* The validation of a pair: ten rounds, nothing removed, a lifting
   that lifts nothing, and the code left where it is called. *)
Definition validate_pair (before after : expr) : validation :=
  validate_unit 10 (Dropped [] []) [] []
    [Lifting (Unit_interface done failed) [] [] [] [] [] [] [] [] false
       false]
    [reader]
    before after.

(* Where the slot and the field hold one variable, the pair is
   validated, by the comparison of the lifted terms; the two units
   exit with the same two values. *)
Definition shared_reads_validated_statement : Prop :=
  validate_pair (pair_unit pair_field_read (Var x))
    (pair_unit pair_shared (Var x))
  = Same_once_lifted
  ∧ run 7
      (State (pair_unit pair_field_read (Var x)) [] [] empty_static_env
         empty_heap)
    = Exit done
        [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 1)]
        empty_heap
  ∧ run 7
      (State (pair_unit pair_shared (Var x)) [] [] empty_static_env
         empty_heap)
    = Exit done
        [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 1)]
        empty_heap.

(* Where they hold two variables, the term after is another program,
   and the pair is not validated. *)
Definition wrong_sharing_refused_statement : Prop :=
  validate_pair (pair_unit pair_field_read (Var y))
    (pair_unit pair_shared (Var y))
  = Not_validated
  ∧ run 7
      (State (pair_unit pair_field_read (Var y)) [] [] empty_static_env
         empty_heap)
    = Exit done
        [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 2)]
        empty_heap
  ∧ run 7
      (State (pair_unit pair_shared (Var y)) [] [] empty_static_env
         empty_heap)
    = Exit done
        [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 1)]
        empty_heap.

End Pairs.
