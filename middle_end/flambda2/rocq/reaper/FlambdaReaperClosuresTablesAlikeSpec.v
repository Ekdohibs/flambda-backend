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

(* "Unboxed alike": the value slot of a field of a representation that is
   unboxed in turn, and the variable that the set holds there, as
   FlambdaReaperClosuresTablesFieldsSpec.repr_field_frag_ok asks of them
   ([unboxed_field_pairs]), for the tables of the unboxing analysis. It
   does not follow from the premises of the statements, nor from the
   compiler's recursion ending with no fatal error.

   This file is to be reviewed: [unboxing_checked], [SlotOfBlock] and the
   statements. The proofs are in FlambdaReaperClosuresTablesAlikeProofs.v,
   which needs no review.

   [unboxing_checked n block fields]: the compiler's recursion
   ([mk_unboxed_fields] of unboxing_analysis.ml) on the allocation
   [block] and the fields [fields] makes no chain of more than [n] nested
   calls, and stops with none of two of its fatal errors: a field read
   into names some to be unboxed and some not, or into names all to be
   unboxed with several sources. A field that is not data is left out
   (the compiler stops on some of them, which this does not ask).
   [unboxing_checked_ends_statement]: such a recursion ends
   (FlambdaReaperUnboxingFuelSpec.unboxing_ends).

   [unboxed_alike_counterexample_statement]: the unit [SlotOfBlock] is of
   the premises of the statements of
   FlambdaReaperClosuresTablesFieldsSpec.v; the recursion of the change of
   representation of the closure [x61] and that of the unboxing of the
   block [x10] end, with no such fatal error; the slot 80 of the
   representation of [x61], which holds [x10], is unboxed two levels
   deep, field 0 of field 0, while the field 0 of [x10] is a variable:
   the two are not unboxed alike, and the slot fails
   [repr_field_frag_ok]. The reads of the slot ([x44], in the code 51)
   are unboxed as the slot is. What differs: the fields of [x10] are read
   from its uses and from the uses of the names read off them at a
   function slot ([add_usages_through_function_slots]), whatever was
   read. A handler that is never applied projects a function slot off
   [x10], a block, into [x90], reads field 0 of [x90] into [x91] and
   returns it: [x91] has any usage, so the field 0 of [x10] is
   [Use_unknown]. The reads of the slot see no such projection, and
   unbox the field 0, read into [x45], which [x11] dominates, as [x11] is
   to be unboxed: [x90] has no source, so that the any usage of [x91]
   reaches no allocation. The compiler, given the same unit, would stop
   with a fatal error in the rebuild of the set of [x61]
   ([fold2_subset_u] of rebuild.ml, "[fold2_unboxed_subset]"); it never
   projects a function slot off a block. So "unboxed alike" needs,
   besides the recursions ending with no fatal error, that every node
   read off a use at a function slot be a closure of a set that holds
   that function slot. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaBindersSpec FlambdaReaperGraphSpec FlambdaReaperTraversal
  FlambdaReaperSolution FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperRebuild
  FlambdaReaperRebuildSpec FlambdaReaperCallingConventions
  FlambdaReaperUnboxing FlambdaUnusedStaticsSpec
  FlambdaReaperUnboxingRefinesSpec FlambdaReaperUnboxingRefinesCallsSpec
  FlambdaReaperLocalFields FlambdaReaperUnboxingRefinesClosuresSpec
  FlambdaReaperClosuresTablesSpec FlambdaReaperClosuresTablesFieldsSpec.
From OxCamlFlambda2 Require FlambdaReaperLeast FlambdaReaperUnboxingFuelSpec.
Import ListNotations.

(* ======================================================================== *)
(* The compiler's recursion, checked                                        *)
(* ======================================================================== *)

Section Checked.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

Abbreviation fields_read :=
  (FlambdaReaperUnboxingFuelSpec.fields_read F T code_my_closures).

Fixpoint unboxing_checked (n : nat) (block : node)
    (fields : list (field * field_use)) {struct n} : bool :=
  forallb
    (fun '(f, use) =>
       negb (is_data_field f)
       || match get_single_field_source F T block f, use with
          | No_source, _ | _, Use_unknown => true
          | field_source, Use_known flow_to =>
            if forallb (has_to_be_unboxed dominated to_unbox) flow_to
            then
              match field_source, n with
              | One_source source, S n =>
                unboxing_checked n source (fields_read flow_to)
              | _, _ => false
              end
            else
              negb (existsb (has_to_be_unboxed dominated to_unbox) flow_to)
          end)
    fields.

End Checked.

(* ======================================================================== *)
(* A function slot projected off a block                                    *)
(* ======================================================================== *)

(* The code 51 of the closure [x61] reads its slot 80 into [x44], field 0
   of it into [x45] and field 0 of that into [x46], which it switches on.
   The unit makes [x11 = [7]], [x10 = [x11]], and [x61] with [x10] in the
   slot 80, and calls [x61]; the handler 20, never applied, projects the
   function slot 72 off [x10] into [x90], reads field 0 of [x90] into
   [x91] and returns [x91]. [x11] and [x10] are to be unboxed, and [x61]
   changes its representation. *)
Module SlotOfBlock.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition goto (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition code_51 : code :=
  Code [42] 43 5 6 (make_code_metadata [Singleton Value] [Singleton Value])
    (Let 44 Value (Project_value_slot_from 71 80 (Var 43))
       (Let 45 Value (load 0 (Var 44))
          (Let 46 Value (load 0 (Var 45))
             (Let 48 (Naked_number Kind_naked_immediate)
                (Prim (Unary Untag_immediate) [Var 46])
                (Switch (Var 48) [(0%Z, goto 5 [Var 42])]))))).

Definition unit_expr : expr :=
  Let_static [Bind_code 51 code_51]
    (Let 11 Value (make_block [int 7])
       (Let 10 Value (make_block [Var 11])
          (Let 61 Value (Closure_of_set 71 [(71, 51)] [(80, Var 10)])
             (Let_cont_non_recursive 20
                (Handler [] (make_handler_metadata [])
                   (Let 90 Value (Project_function_slot 71 72 (Var 10))
                      (Let 91 Value (load 0 (Var 90))
                         (return_to 1 [Var 91]))))
                (Apply (ChangedInSlot.call 51 61 [int 5] 1 2)))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 200 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables [] max_unbox_size false.

Definition tcr : list node :=
  derive_to_change_representation no_local facts tables [] max_unbox_size
    false to_unbox.

Definition my_closures : finite_map variable :=
  bound_code_my_closures unit_expr.

Definition unboxed : list (node * unboxed_fields) :=
  fst (unit_unboxed_fields facts tables dominated to_unbox unit_expr).

Definition reprs : list (node * closure_repr) :=
  unit_changed_representations no_local facts tables dominated to_unbox tcr
    unit_expr.

(* The fields of the closures of the set of [x61], from which its
   representation is made, and the fields of [x10] read from its uses,
   from which its unboxing is made. *)
Definition set_fields : list (field * field_use) :=
  get_fields_of_constructors facts tables no_local
    (map snd (set_function_slots facts (Node_variable 61))).

Definition fields_read_10 : list (field * field_use) :=
  FlambdaReaperUnboxingFuelSpec.fields_read facts tables my_closures
    [Node_variable 10].

(* The slot 80 in the representation of [x61], the unboxed fields of
   [x10], and those of [x44], the read of the slot. *)
Definition slot_80 : unboxed_field :=
  Unboxed
    [(Field_block 0 Value, Unboxed [(Field_block 0 Value, Not_unboxed 81)])].

Definition fields_10 : unboxed_fields :=
  [(Field_block 0 Value, Not_unboxed 95)].

Definition fields_44 : unboxed_fields :=
  [(Field_block 0 Value, Unboxed [(Field_block 0 Value, Not_unboxed 93)])].

End SlotOfBlock.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition unboxing_checked_ends_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n : nat)
    (block : node) (fields : list (field * field_use)),
    unboxing_checked F T code_my_closures dominated to_unbox n block fields
    = true ->
    FlambdaReaperUnboxingFuelSpec.unboxing_ends F T code_my_closures
      dominated to_unbox block fields.

(* The unit is of the premises of the statements of
   FlambdaReaperClosuresTablesFieldsSpec.v, the recursions of the
   representation of [x61] and of the unboxing of [x10] end with no fatal
   error, and the slot 80 of the representation of [x61], which holds
   [x10], is not unboxed alike [x10], but alike [x44]. *)
Definition unboxed_alike_counterexample_statement : Prop :=
  in_flow_fragment SlotOfBlock.unit_expr = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      SlotOfBlock.unit_expr
    = true
  ∧ variables_bound_once SlotOfBlock.unit_expr = true
  ∧ least_solution FlambdaReaperUnboxing.Examples.no_local SlotOfBlock.facts
      SlotOfBlock.tables
  ∧ derive_cannot_change SlotOfBlock.facts SlotOfBlock.tables = []
  ∧ analysis_tables FlambdaReaperUnboxing.Examples.no_local
      SlotOfBlock.facts SlotOfBlock.tables []
      FlambdaReaperUnboxing.Examples.max_unbox_size false
      SlotOfBlock.dominated SlotOfBlock.to_unbox SlotOfBlock.tcr
    = true
  ∧ SlotOfBlock.to_unbox
    = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ unboxing_checked SlotOfBlock.facts SlotOfBlock.tables
      SlotOfBlock.my_closures SlotOfBlock.dominated SlotOfBlock.to_unbox 2
      (Node_variable 61%positive) SlotOfBlock.set_fields
    = true
  ∧ unboxing_checked SlotOfBlock.facts SlotOfBlock.tables
      SlotOfBlock.my_closures SlotOfBlock.dominated SlotOfBlock.to_unbox 2
      (Node_variable 10%positive) SlotOfBlock.fields_read_10
    = true
  ∧ Keyed.lookup node_eqb (Node_variable 61%positive) SlotOfBlock.reprs
    = Some (Closure_repr [(Field_value_slot 80, SlotOfBlock.slot_80)]
              [(71, 82)] 71)%positive
  ∧ fields_of SlotOfBlock.unboxed 10%positive = Some SlotOfBlock.fields_10
  ∧ unboxed_field_pairs Value SlotOfBlock.slot_80
      (Unboxed SlotOfBlock.fields_10)
    = None
  ∧ repr_field_frag_ok SlotOfBlock.unboxed SlotOfBlock.reprs
      [(80%positive, Var 10%positive)]
      (Field_value_slot 80%positive, SlotOfBlock.slot_80)
    = false
  ∧ fields_of SlotOfBlock.unboxed 44%positive = Some SlotOfBlock.fields_44
  ∧ unboxed_field_pairs Value SlotOfBlock.slot_80
      (Unboxed SlotOfBlock.fields_44)
    <> None.
