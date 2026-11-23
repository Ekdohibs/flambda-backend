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


(* The rebuild with local unboxing of FlambdaReaperUnboxingRefinesSpec.v,
   for the units with primitives of the heap
   (FlambdaReaperFlowSpec.in_heap_fragment).

   This file is to be reviewed: the statements, against
   FlambdaReaperUnboxingRefinesSpec.v and FlambdaReaperRebuildSpec.v. The
   proofs are in FlambdaReaperUnboxingRefinesHeapProofs.v, which needs no
   review.

   How the unboxing meets the heap. What the unboxing removes is the
   allocation of an immutable block of values or of a boxed number
   ([unboxed_binding_ok]), which the machine makes as a value, not as a
   cell: no location is allocated or freed by it. A variable unboxed is,
   in the fragment ([local_unboxing]), only read, aliased, held in a
   field of a block unboxed in turn or given to a handler; it is no
   argument of a primitive of the heap, so no block unboxed is stored
   into a cell (where the compiler would give it any usage, and not
   unbox it), nor compared physically. A field of a block unboxed may
   hold a location: the unboxing binds the variable of the field to it,
   and a read of the field becomes that variable, the same location, so
   that a store into the cell is seen alike through either. A read of an
   unboxed variable is not of a mutable field ([read_shape_ok]). The
   [Let]s of the primitives of the heap are those of the plain rebuild,
   on arguments that are not unboxed.

   The statements:
   - [rebuild_unbox_refines_rebuild_heap_statement]:
     FlambdaReaperUnboxingRefinesSpec.rebuild_unbox_refines_rebuild_statement
     of the units of [in_heap_fragment]. The two rebuilds run the same
     primitives of the heap on the same values, so the comparison is
     [program_refines], heaps included: the same cells at the same
     locations.
   - [rebuild_unbox_refines_up_to_heap_statement] and
     [rebuild_unbox_refines_up_to_heap_local_statement]:
     [rebuild_unbox_refines_observably_statement] and its form with local
     fields, of the units of [in_heap_fragment], with the exits compared
     up to a correspondence of the locations of the two runs
     ([program_refines_up_to_heap], as
     FlambdaReaperRebuildSpec.rebuild_refines_up_to_heap_statement has
     them): the first statement composed with that one.
   - [heap_example_statement]: a unit of [in_heap_fragment] and not of
     [in_flow_fragment] whose unboxed block holds a location, stored
     into through the field read; the unboxing unboxes it, and the unit
     and its rebuild with unboxing exit alike, with the same heap.
   - [cell_example_statement]: what the fragment leaves out. The rules
     unbox a cell that is only read (FlambdaReaperUnboxing.derive_to_unbox
     takes it, as the compiler's unboxing would a mutable block that no
     store reaches: its traversal gives such a block the facts of an
     immutable one, and a store gives the block any usage); the rebuild
     binds the fields to the values the cell was made with, which no
     store changes. The rebuilt unit exits with the same values but one
     cell fewer, so that it refines the plain rebuild up to the heap
     ([program_refines_up_to_heap]) and not with heaps compared by
     equality ([program_refines]): such units are to be compared up to a
     correspondence of locations, which the fragment
     ([unboxed_binding_ok], blocks of values and boxed numbers) does not
     need, and leaves them out.
   No counterexample. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing FlambdaReaperLeast
  FlambdaUnusedStaticsSpec FlambdaReaperLocalFields
  FlambdaReaperUnboxingRefinesSpec.
Import ListNotations.

Definition rebuild_unbox_refines_rebuild_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_refines_up_to_heap_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_heap_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* The same with local fields: the tables pass the check of the fix
   too. *)
Definition rebuild_unbox_refines_up_to_heap_local_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_heap_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_local_reads_used is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    check_unboxed_reads (traverse_unit flag u e) T dominated to_unbox = true ->
    program_refines_up_to_heap e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* An immutable block that holds a cell, read, and the cell stored into
   through the field read:

     let x10 = mutable { 1 } in
     let x11 = [x10] in
     let x12 = x11.(0) in
     let x13 = x12.(0) <- 2 in
     let x14 = x10.(0) in              (a mutable load: 2)
     return x14

   Its least solution, as [compute_least] gives it, passes the checks of
   the statements; the rules unbox [x11], whose field, the location of
   the cell, the rebuild binds to the variable 15, which the read
   becomes. The unit and the rebuilt unit exit with 2 and the same
   heap. *)
Module HeapExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition access : block_access_kind :=
  Block_access_values None None Any_value.

Definition unit_expr : expr :=
  Let 10 Value (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
    (Let 11 Value (make_block [Var 10])
       (Let 12 Value (load 0 (Var 11))
          (Let 13 Value
             (Prim (Heap (Block_set access Assignment 0)) [Var 12; int 2])
             (Let 14 Value (Prim (Unary (Block_load access Mutable 0)) [Var 10])
                (return_to 1 [Var 14]))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 10 Value (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1])
    (Let 15 Value (Simple (Var 10))
       (Let 12 Value (Simple (Var 15))
          (Let 13 Value
             (Prim (Heap (Block_set access Assignment 0)) [Var 12; int 2])
             (Let 14 Value (Prim (Unary (Block_load access Mutable 0)) [Var 10])
                (return_to 1 [Var 14]))))).

Definition cell (n : Z) : cell :=
  Cell_block Cell_mutable 0 [Value_const (Tagged_immediate n)].

Definition heap_at_exit : heap := [(2, cell 2%Z); (2, cell 1%Z)].

End HeapExample.

Definition heap_example_statement : Prop :=
  is_local_slots_only Examples.no_local
  ∧ in_heap_fragment HeapExample.unit_expr = true
  ∧ in_flow_fragment HeapExample.unit_expr = false
  ∧ static_constants (unit_codes HeapExample.unit_expr) HeapExample.unit_expr
    = true
  ∧ results_kinded [] [] HeapExample.unit_expr = true
  ∧ well_formed_unit Examples.interface HeapExample.unit_expr = true
  ∧ exits_one_value Examples.interface HeapExample.unit_expr = true
  ∧ check_closed Examples.no_local HeapExample.facts HeapExample.tables = true
  ∧ check_field_used_exact Examples.no_local HeapExample.facts
      HeapExample.tables
    = true
  ∧ check_dominated HeapExample.tables HeapExample.dominated = true
  ∧ check_to_unbox Examples.no_local HeapExample.facts HeapExample.tables
      HeapExample.cannot_change Examples.max_unbox_size false
      HeapExample.to_unbox
    = true
  ∧ HeapExample.to_unbox = [Node_variable 11%positive]
  ∧ local_unboxing HeapExample.facts HeapExample.tables
      HeapExample.dominated HeapExample.to_unbox HeapExample.unit_expr
    = true
  ∧ check_unboxed_reads HeapExample.facts HeapExample.tables
      HeapExample.dominated HeapExample.to_unbox
    = true
  ∧ rebuild_unit_unbox HeapExample.facts HeapExample.tables
      HeapExample.cannot_change
      (unit_code_deps Auto Examples.interface HeapExample.unit_expr)
      HeapExample.dominated HeapExample.to_unbox true HeapExample.unit_expr
    = HeapExample.rebuilt
  ∧ run 10 (State HeapExample.unit_expr [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 2)]
        HeapExample.heap_at_exit no_reads
  ∧ run 10 (State HeapExample.rebuilt [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [Value_const (Tagged_immediate 2)]
        HeapExample.heap_at_exit no_reads.

(* A cell that is only read:

     let x10 = mutable { 1; 5 } in
     let x11 = x10.(0) in              (a mutable load)
     let x12 = x10.(1) in              (a mutable load)
     let x14 = [x11; x12] in
     return x14

   The rules unbox [x10], which the fragment leaves out; the rebuild with
   unboxing binds the variables 15 and 16 of its fields to 1 and 5. *)
Module CellExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition mutable_load (field : Z) (block : simple) : named :=
  Prim (Unary (Block_load HeapExample.access Mutable field)) [block].

Definition unit_expr : expr :=
  Let 10 Value
    (Prim (Heap (Make_block_with_identity Cell_mutable 0)) [int 1; int 5])
    (Let 11 Value (mutable_load 0 (Var 10))
       (Let 12 Value (mutable_load 1 (Var 10))
          (Let 14 Value (make_block [Var 11; Var 12])
             (return_to 1 [Var 14])))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match compute_least no_local facts 40 with
  | Some T => T
  | None => empty_tables
  end.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition plain : expr := rebuild_unit_from_tables facts tables true unit_expr.

Definition rebuilt : expr :=
  Let 16 Value (Simple (int 5))
    (Let 15 Value (Simple (int 1))
       (Let 11 Value (Simple (Var 15))
          (Let 12 Value (Simple (Var 16))
             (Let 14 Value (make_block [Var 11; Var 12])
                (return_to 1 [Var 14]))))).

Definition result : value :=
  Value_block 0
    [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 5)].

Definition heap_at_exit : heap :=
  [(2, Cell_block Cell_mutable 0
         [Value_const (Tagged_immediate 1); Value_const (Tagged_immediate 5)])].

End CellExample.

Definition cell_example_statement : Prop :=
  in_heap_fragment CellExample.unit_expr = true
  ∧ check_closed Examples.no_local CellExample.facts CellExample.tables = true
  ∧ check_field_used_exact Examples.no_local CellExample.facts
      CellExample.tables
    = true
  ∧ CellExample.to_unbox = [Node_variable 10%positive]
  ∧ local_unboxing CellExample.facts CellExample.tables
      CellExample.dominated CellExample.to_unbox CellExample.unit_expr
    = false
  ∧ rebuild_unit_unbox CellExample.facts CellExample.tables
      CellExample.cannot_change
      (unit_code_deps Auto Examples.interface CellExample.unit_expr)
      CellExample.dominated CellExample.to_unbox true CellExample.unit_expr
    = CellExample.rebuilt
  ∧ run 10 (State CellExample.unit_expr [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [CellExample.result] CellExample.heap_at_exit no_reads
  ∧ run 10 (State CellExample.plain [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [CellExample.result] CellExample.heap_at_exit no_reads
  ∧ run 10 (State CellExample.rebuilt [] [] empty_static_env empty_heap
              no_reads)
    = Exit 1%positive [CellExample.result] empty_heap no_reads
  ∧ ~ program_refines CellExample.plain CellExample.rebuilt
  ∧ program_refines_up_to_heap CellExample.plain CellExample.rebuilt.
