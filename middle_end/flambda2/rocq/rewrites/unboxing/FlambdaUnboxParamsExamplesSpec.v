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


(* Three bodies of code in which Simplify unboxed the parameter of a
   join, and the validation of each once [unbox_known_params] is in the
   round of the term before.

   This file is to be reviewed: the bodies, against what the test tool
   prints of the sources below; the round; and the statements.

   The bodies are those of the function of each source, as the
   validation has them when it compares: the body before Simplify once
   the rounds of today have been applied to it, and the body after.
   They are written by hand from what the test tool prints of the two
   terms it compares (middle_end/flambda2/tests/tools/flsemantics.ml,
   with FLSEMANTICS_PRINT_TERMS set); the names are numbered here in
   the order in which they are bound.

   [pair_computed] is

     let[@inline never] sum : bool -> int -> int =
      fun c x ->
       let p = if c then x + 1, 1 else 2, x + 2 in
       fst p + snd p

   where [fst] and [snd] are %field0_immut and %field1_immut. Each
   branch makes a pair and jumps to the continuation of the [let],
   which loads the two fields. Simplify gives that continuation the two
   fields as parameters and not the pair, and no block is made.

   [pair_built] is the same with [x, 1] and [2, x] for the two pairs.
   Each branch is then left with a jump and nothing else, and Simplify
   makes the arm of the [Switch] jump to the join itself.

   [int64_built] is

     let[@inline never] f : bool -> int64 -> int64 -> int64 =
      fun c a b ->
       let x = if c then add a b else sub a b in
       add x 1L

   with the arithmetic of %int64_add and %int64_sub. Each branch boxes
   its result, and the continuation unboxes it. Simplify gives the
   continuation the number unboxed, and neither branch boxes.

   [round_with_unboxing] is FlambdaValidateSpec.normalise_expr_before
   with [unbox_known_params] where it would go: after what follows the
   values, so that what an application gives is known from the term,
   and before the unused bindings and parameters are removed, so that
   the boxed parameter goes in the same round. It is here for the
   examples only; the validation does not run it.

   What is said of [pair_computed] and of [int64_built]: the body
   before, after five such rounds, is the body after up to bound names
   ([..._validated_statement]); after five rounds of
   FlambdaValidateSpec.normalise_expr_before, which has no unboxing of
   parameters, it is not ([..._not_validated_today_statement]: the
   statements were written before the validation ran such a pass, and
   their names are kept), so the pass is what was missing; and the
   body after refines the body before, under all
   environments ([..._refines_statement]), which follows from the first
   by the theorems of the passes and of the comparison.

   What is said of [pair_built]: after five rounds the body before has
   the parameters and the arguments of the body after, and no block,
   but each arm of its [Switch] still goes through the continuation of
   its branch, whose handler is now a single jump
   ([pair_built_unboxed_statement], with the term in full). The round
   makes that shortcut only where the [Switch] is directly under the
   [Let_cont] (FlambdaShortcutSpec.v), and here a [Let] and another
   [Let_cont] stand between. So that unit needs one more rewrite, which
   is not this file's.

   The proofs are in FlambdaUnboxParamsExamplesProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaPrimitiveModel FlambdaSemanticsSpec FlambdaCompareSpec
  FlambdaNormaliseSpec FlambdaBlockLoadSpec FlambdaLoopAliasesSpec
  FlambdaShortcutSpec FlambdaSinkContsSpec FlambdaUnboxSpec
  FlambdaUnusedContsSpec FlambdaUnusedParamsSpec FlambdaValidateSpec
  FlambdaUnboxParamsSpec.
From OxCamlFlambda2Types Require Import FlambdaAliasesSpec.
Import ListNotations.
Open Scope positive_scope.

(* ======================================================================== *)
(* The round                                                                *)
(* ======================================================================== *)

Definition round_with_unboxing (e : expr) : expr :=
  let e :=
    shortcut_switches
      (thread_switches
         (inline_tail_conts
            (sink_let_conts
               (remove_unused_conts (collapse_switches (fold_constants e))))))
  in
  let e :=
    unbox_known_params
      (follow_aliases_in_loops []
         (follow_aliases [] (unbox_known_numbers (load_known_blocks e))))
  in
  drop_unused_params
    (remove_unused_defined_lets [] (remove_unused_lets (fold_constants e))).

(* ======================================================================== *)
(* What the bodies are made of                                              *)
(* ======================================================================== *)

(* An application with no trap action. *)
Definition jump (k : continuation) (args : list simple) : apply_cont :=
  Apply_cont_expr k args None.

Definition tagged (i : Z) : simple := Const (Tagged_immediate i).

Definition naked_immediate : flambda_kind :=
  Naked_number Kind_naked_immediate.

Definition naked_int64 : flambda_kind := Naked_number Kind_naked_int64.

(* The load that the front end makes of [fst] and [snd]: marked
   [Immutable], with nothing said of the tag or the size. *)
Definition field (i : Z) (block : variable) : named :=
  Prim (Unary (Block_load (Block_access_values None None Any_value)
                 Immutable i))
    [Var block].

Definition pair (first second : simple) : named :=
  Prim (Variadic (Make_block (Block_kind_values 0))) [first; second].

Definition add_tagged (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_tagged_immediate Add)) [a; b].

Definition untag (s : simple) : named := Prim (Unary Untag_immediate) [s].

Definition unbox_int64 (s : simple) : named :=
  Prim (Unary (Unbox_number Boxable_naked_int64)) [s].

Definition box_int64 (s : simple) : named :=
  Prim (Unary (Box_number Boxable_naked_int64)) [s].

Definition int64_arith (op : binary_int_arith_op) (a b : simple) : named :=
  Prim (Binary (Int_arith Standard_naked_int64 op)) [a; b].

(* The return continuation of the code is 1. The continuation of the
   join is 2, and those of the two branches are 3 and 4. *)

(* ======================================================================== *)
(* A pair computed on both paths                                            *)
(* ======================================================================== *)

(* The parameters of the code are [c], 1, and [x], 2. *)
Definition pair_computed_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 4 Value (field 1 3)
          (Let 5 Value (field 0 3)
             (Let 6 Value (add_tagged (Var 5) (Var 4))
                (Apply_cont (jump 1 [Var 6]))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 7 Value (add_tagged (Var 2) (tagged 2))
             (Let 8 Value (pair (tagged 2) (Var 7))
                (Apply_cont (jump 2 [Var 8])))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 2) (tagged 1))
                (Let 10 Value (pair (Var 9) (tagged 1))
                   (Apply_cont (jump 2 [Var 10])))))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

Definition pair_computed_after : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 5] (make_handler_metadata [Value; Value])
       (Let 6 Value (add_tagged (Var 5) (Var 4))
          (Apply_cont (jump 1 [Var 6]))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 7 Value (add_tagged (Var 2) (tagged 2))
             (Apply_cont (jump 2 [Var 7; tagged 2]))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 9 Value (add_tagged (Var 2) (tagged 1))
                (Apply_cont (jump 2 [tagged 1; Var 9]))))
          (Let 11 naked_immediate (untag (Var 1))
             (Switch (Var 11) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

Definition pair_computed_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_unboxing pair_computed_before)
    pair_computed_after
  = true.

Definition pair_computed_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before pair_computed_before)
    pair_computed_after
  = false.

Definition pair_computed_refines_statement : Prop :=
  expr_refines pair_computed_before pair_computed_after.

(* ======================================================================== *)
(* A pair built on both paths                                               *)
(* ======================================================================== *)

Definition pair_built_before : expr :=
  Let_cont_non_recursive 2
    (Handler [3] (make_handler_metadata [Value])
       (Let 4 Value (field 1 3)
          (Let 5 Value (field 0 3)
             (Let 6 Value (add_tagged (Var 5) (Var 4))
                (Apply_cont (jump 1 [Var 6]))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 7 Value (pair (tagged 2) (Var 2))
             (Apply_cont (jump 2 [Var 7]))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 8 Value (pair (Var 2) (tagged 1))
                (Apply_cont (jump 2 [Var 8]))))
          (Let 9 naked_immediate (untag (Var 1))
             (Switch (Var 9) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

(* The body after Simplify. *)
Definition pair_built_after : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 5] (make_handler_metadata [Value; Value])
       (Let 6 Value (add_tagged (Var 5) (Var 4))
          (Apply_cont (jump 1 [Var 6]))))
    (Let 9 naked_immediate (untag (Var 1))
       (Switch (Var 9)
          [(0%Z, jump 2 [Var 2; tagged 2]); (1%Z, jump 2 [tagged 1; Var 2])])).

(* The same with each arm going through the continuation of its
   branch. *)
Definition pair_built_unboxed : expr :=
  Let_cont_non_recursive 2
    (Handler [4; 5] (make_handler_metadata [Value; Value])
       (Let 6 Value (add_tagged (Var 5) (Var 4))
          (Apply_cont (jump 1 [Var 6]))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Apply_cont (jump 2 [Var 2; tagged 2])))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Apply_cont (jump 2 [tagged 1; Var 2])))
          (Let 9 naked_immediate (untag (Var 1))
             (Switch (Var 9) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

Definition pair_built_unboxed_statement : Prop :=
  iterate 5 round_with_unboxing pair_built_before = pair_built_unboxed.

Definition pair_built_not_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_unboxing pair_built_before)
    pair_built_after
  = false.

(* ======================================================================== *)
(* A boxed number boxed on both paths                                       *)
(* ======================================================================== *)

(* The parameters of the code are [c], 1, [a], 2, and [b], 3. *)
Definition int64_built_before : expr :=
  Let_cont_non_recursive 2
    (Handler [4] (make_handler_metadata [Value])
       (Let 5 naked_int64 (unbox_int64 (Var 4))
          (Let 6 naked_int64 (int64_arith Add (Var 5) (Const (Naked_int64 1)))
             (Let 7 Value (box_int64 (Var 6))
                (Apply_cont (jump 1 [Var 7]))))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 8 naked_int64 (unbox_int64 (Var 3))
             (Let 9 naked_int64 (unbox_int64 (Var 2))
                (Let 10 naked_int64 (int64_arith Sub (Var 9) (Var 8))
                   (Let 11 Value (box_int64 (Var 10))
                      (Apply_cont (jump 2 [Var 11])))))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 12 naked_int64 (unbox_int64 (Var 3))
                (Let 13 naked_int64 (unbox_int64 (Var 2))
                   (Let 14 naked_int64 (int64_arith Add (Var 13) (Var 12))
                      (Let 15 Value (box_int64 (Var 14))
                         (Apply_cont (jump 2 [Var 15])))))))
          (Let 16 naked_immediate (untag (Var 1))
             (Switch (Var 16) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

Definition int64_built_after : expr :=
  Let_cont_non_recursive 2
    (Handler [5] (make_handler_metadata [naked_int64])
       (Let 6 naked_int64 (int64_arith Add (Var 5) (Const (Naked_int64 1)))
          (Let 7 Value (box_int64 (Var 6))
             (Apply_cont (jump 1 [Var 7])))))
    (Let_cont_non_recursive 3
       (Handler [] (make_handler_metadata [])
          (Let 8 naked_int64 (unbox_int64 (Var 3))
             (Let 9 naked_int64 (unbox_int64 (Var 2))
                (Let 10 naked_int64 (int64_arith Sub (Var 9) (Var 8))
                   (Apply_cont (jump 2 [Var 10]))))))
       (Let_cont_non_recursive 4
          (Handler [] (make_handler_metadata [])
             (Let 12 naked_int64 (unbox_int64 (Var 3))
                (Let 13 naked_int64 (unbox_int64 (Var 2))
                   (Let 14 naked_int64 (int64_arith Add (Var 13) (Var 12))
                      (Apply_cont (jump 2 [Var 14]))))))
          (Let 16 naked_immediate (untag (Var 1))
             (Switch (Var 16) [(0%Z, jump 3 []); (1%Z, jump 4 [])])))).

Definition int64_built_validated_statement : Prop :=
  expr_eqb [] [] (iterate 5 round_with_unboxing int64_built_before)
    int64_built_after
  = true.

Definition int64_built_not_validated_today_statement : Prop :=
  expr_eqb [] [] (iterate 5 normalise_expr_before int64_built_before)
    int64_built_after
  = false.

Definition int64_built_refines_statement : Prop :=
  expr_refines int64_built_before int64_built_after.
