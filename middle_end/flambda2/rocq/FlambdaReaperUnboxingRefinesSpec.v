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

(* What the rebuild with unboxing is worth where the unboxing is local:
   the unit rebuilt with unboxing from a solution that passes the checks
   refines the unit, as the plain rebuild does, if the allocations that
   are unboxed are blocks and boxed numbers that are only read in the
   scope where they are bound.

   This file is to be reviewed: the fragment ([local_unboxing]), the
   statements, and the example, against FlambdaReaperUnboxing.v (the
   rebuild with unboxing, which needs no review) and
   FlambdaReaperRebuildSpec.v (the plain statement). The proofs are in
   FlambdaReaperUnboxingRefinesProofs.v, which needs no review.

   [rebuild_unbox_refines_rebuild_statement]: for a unit [e] of the
   fragment of FlambdaReaperFlowSpec.v ([in_flow_fragment]), with the
   interface [u], any mode of the traversal ([flag]) and of the rebuild
   ([preserve_direct_calls]), and tables [T] that are closed under the
   rules ([check_closed]), the unit rebuilt with unboxing
   ([rebuild_unit_unbox]: no change of representation) refines the unit
   rebuilt without ([FlambdaReaperRebuild.rebuild_unit_from_tables]),
   run on its own ([program_refines]), if the unboxing that the tables
   of [dominated_by_allocation_point] and [to_unbox] give is local
   ([local_unboxing]). The two rebuilds make the same term but for the
   [Let] of an unboxed variable, which the unboxing makes [Let]s of the
   variables of its fields, each bound to what the allocation holds there,
   and a read of a field of an unboxed variable, which it makes the
   variable of the field. A field unboxed is read, with a use, from a use
   of the variable ([get_fields] of [mk_unboxed_fields]; the graph of a
   unit of the fragment reads no function slot and makes no call, so
   that no use is through one), so the rules give it
   [field_of_constructor_is_used] and give what it holds a use: the plain
   rebuild keeps the field, and binds what it holds. That is what
   [check_closed] is asked for; [is_local_slots_only] says that the
   fields of a block and of a boxed number are not local, and
   [in_flow_fragment] is the fragment in which FlambdaReaperFlowProofs.v
   says what facts the traversal gives.

   [rebuild_unbox_refines_observably_statement]: with the premises of
   FlambdaReaperRebuildSpec.rebuild_refines_observably_statement, the
   unit rebuilt with unboxing refines the unit up to observation
   ([program_refines_observably]); it is the plain statement composed
   with the first.

   The fragment ([local_unboxing]), for the unboxed fields of the names
   that the tables give ([unit_unboxed_fields]) and the solution:
   - [local_expr]: no closure, no projection, no call and no code (the
     terms of FlambdaReaperRebuildSpec.closure_free); a variable that is
     unboxed is bound by a [Let] of a block of values or of a boxed
     number ([unboxed_binding_ok]), and is otherwise only read, by a
     [Block_load] or an [Unbox_number] ([unboxed_read]): it is not a
     parameter, nor an argument of a primitive, of an [Apply_cont], of a
     [Switch] or of a static block. A symbol that is unboxed is not used.
   - [unboxed_binding_ok]: the unboxed fields of a block are fields of
     the block, of kind [Value], each a variable, and the argument held
     there is of kind [Value] where the rebuild looks it up; the unboxed
     field of a boxed number is its contents, a variable. Otherwise the
     OCaml stops with a fatal error, or the model makes a field poison.
   - [plain_binding_ok]: a read that the rebuild keeps (its variable is
     used) reads a field that is unboxed, a variable. Otherwise the OCaml
     stops with a fatal error ([rewrite_field_access]) and the model
     makes the read [Invalid]. This one is about the solution and does
     not follow from [check_closed]: a field read whose variable is used
     has a source in the least solution, and is then unboxed, but a
     solution that is closed and not least may say that it has none.
   - A handler has as many kinds as parameters.
   Nothing is asked of the variables that stand for the fields: the
   model names them after every name of the unit, each once, so that
   they are fresh (FlambdaReaperUnboxingRefinesProofs.v proves it).

   Of the checks of FlambdaReaperUnboxing.v none is asked: in the
   fragment, unboxing any allocation is sound. They are needed where an
   unboxed name is passed to a continuation, a call or code, as are the
   two checks of the fixes of FlambdaReaperUnboxingSpec.v
   ([check_called_my_closure] and [check_result_receivers]), which are
   about calls and code. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaSemanticsSpec FlambdaWellFormedSpec
  FlambdaReaperGraphSpec FlambdaReaperTraversal FlambdaReaperSolution
  FlambdaReaperFlowSpec FlambdaReaperRebuild FlambdaReaperRebuildSpec
  FlambdaReaperCallingConventions FlambdaReaperUnboxing
  FlambdaUnusedStaticsSpec.
From OxCamlFlambda2 Require FlambdaReaperTupledSpec.
Import ListNotations.

(* ======================================================================== *)
(* The fragment                                                             *)
(* ======================================================================== *)

Section Fragment.

(* The unboxed fields of the names, and the solution. *)
Variable unboxed : list (node * unboxed_fields).
Variable sol : solution.

Definition fields_of (x : variable) : option unboxed_fields :=
  Keyed.lookup node_eqb (Node_variable x) unboxed.

(* A variable that is not unboxed. *)
Definition plain_variable (x : variable) : bool :=
  negb (Keyed.is_bound node_eqb (Node_variable x) unboxed).

(* A simple that is not an unboxed name. *)
Definition plain_simple (s : simple) : bool :=
  match s with
  | Var x => plain_variable x
  | Symbol sym => negb (Keyed.is_bound node_eqb (Node_symbol sym) unboxed)
  | Const _ => true
  end.

Definition kinds_env (kinds : finite_map flambda_kind) : rebuild_env :=
  {| env_conts := []; env_kinds := kinds; env_projections := [] |}.

(* The [Let] of an unboxed variable, of kind [kind], to [d], the unboxed
   fields of the variable being [fields]: a block of values, of which each
   unboxed field is a field, holding an argument of kind [Value] where the
   rebuild looks it up; or a boxed number, whose unboxed field is its
   contents. Each field is a variable. *)
Definition unboxed_binding_ok (kinds : finite_map flambda_kind)
    (kind : flambda_kind) (d : named) (fields : unboxed_fields) : bool :=
  match kind, d with
  | Value, Prim (Variadic (Make_block (Block_kind_values _))) args =>
    forallb plain_simple args
    && forallb
         (fun '(f, u) =>
            match f, u with
            | Field_block i Value, Not_unboxed _ =>
              match nth_error args i with
              | Some arg => kind_eqb (simple_kind (kinds_env kinds) arg) Value
              | None => false
              end
            | _, _ => false
            end)
         fields
  | Value, Prim (Unary (Box_number bn)) [contents] =>
    plain_simple contents
    && forallb
         (fun '(f, u) =>
            match u with
            | Not_unboxed _ => field_eqb f (Field_boxed_number bn)
            | Unboxed _ => false
            end)
         fields
  | _, _ => false
  end.

(* The read of a field of an unboxed variable: the field, and the unboxed
   fields of the variable. *)
Definition unboxed_read (d : named) : option (field * unboxed_fields) :=
  match d with
  | Prim (Unary (Block_load _ _ _)) [Var x]
  | Prim (Unary (Unbox_number _)) [Var x] =>
    match field_read d, fields_of x with
    | Some (f, _), Some fields => Some (f, fields)
    | _, _ => None
    end
  | _ => None
  end.

(* The [Let] of [x], which is not unboxed: a read of an unboxed variable
   that, if the rebuild keeps it, reads a field unboxed as a variable; or
   a simple or a primitive of simples of [plain_simple]. *)
Definition plain_binding_ok (x : variable) (kind : flambda_kind) (d : named)
    : bool :=
  match unboxed_read d with
  | Some (f, fields) =>
    negb (is_var_used sol x kind)
    || match Keyed.lookup field_eqb f fields with
       | Some (Not_unboxed _) => true
       | _ => false
       end
  | None =>
    match d with
    | Simple s => plain_simple s
    | Prim _ args => forallb plain_simple args
    | _ => false
    end
  end.

Definition plain_apply_cont (ac : apply_cont) : bool :=
  forallb plain_simple (ac_args ac).

Definition local_binding (binding : static_binding) : bool :=
  match binding with
  | Bind_symbol _ (Static_const_block _ fields) =>
    forallb plain_simple fields
  | _ => false
  end.

(* [kinds] are those of the variables in scope, as the rebuild has them
   ([env_kinds]). *)
Fixpoint local_expr (kinds : finite_map flambda_kind) (e : expr) : bool :=
  match e with
  | Let x kind d body =>
    match fields_of x with
    | Some fields => unboxed_binding_ok kinds kind d fields
    | None => plain_binding_ok x kind d
    end
    && local_expr ((x, kind) :: kinds) body
  | Let_static bindings body =>
    forallb local_binding bindings && local_expr kinds body
  | Let_cont_non_recursive _ h body =>
    local_handler kinds h && local_expr kinds body
  | Let_cont_recursive handlers body =>
    forallb (fun '(_, h) => local_handler kinds h) handlers
    && local_expr kinds body
  | Apply _ => false
  | Apply_cont ac => plain_apply_cont ac
  | Switch scrutinee arms =>
    plain_simple scrutinee && forallb (fun '(_, ac) => plain_apply_cont ac) arms
  | Invalid => true
  end
with local_handler (kinds : finite_map flambda_kind)
    (h : continuation_handler) : bool :=
  match h with
  | Handler params metadata handler =>
    Nat.eqb (length params) (length (hm_params_kinds metadata))
    && forallb plain_variable params
    && local_expr (rev (combine params (hm_params_kinds metadata)) ++ kinds)
         handler
  end.

End Fragment.

(* The unit [e] is of the fragment, for the unboxing that the tables
   give from its graph [F]. *)
Definition local_unboxing (F : facts) (T : tables)
    (dominated : list (node * node)) (to_unbox : list node) (e : expr)
    : bool :=
  let unboxed := fst (unit_unboxed_fields F T dominated to_unbox e) in
  local_expr unboxed (solution_of_tables T) [] e.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition rebuild_unbox_refines_rebuild_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (code_deps : finite_map code_dep) (dominated : list (node * node))
    (to_unbox : list node) (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    in_flow_fragment e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    program_refines
      (rebuild_unit_from_tables (traverse_unit flag u e) T
         preserve_direct_calls e)
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change code_deps
         dominated to_unbox preserve_direct_calls e).

Definition rebuild_unbox_refines_observably_statement : Prop :=
  ∀ (flag : reaper_preserve_direct_calls) (is_local : field -> bool)
    (u : unit_interface) (e : expr) (T : tables) (cannot_change : list node)
    (dominated : list (node * node)) (to_unbox : list node)
    (preserve_direct_calls : bool),
    is_local_slots_only is_local ->
    (∀ f : field, is_local f = false) ∨ projection_free e = true ->
    in_flow_fragment e = true ->
    static_constants (unit_codes e) e = true ->
    results_kinded [] [] e = true ->
    well_formed_unit u e = true ->
    exits_one_value u e = true ->
    check_closed is_local (traverse_unit flag u e) T = true ->
    check_field_used_exact is_local (traverse_unit flag u e) T = true ->
    local_unboxing (traverse_unit flag u e) T dominated to_unbox e = true ->
    program_refines_observably e
      (rebuild_unit_unbox (traverse_unit flag u e) T cannot_change
         (unit_code_deps flag u e) dominated to_unbox preserve_direct_calls e).

(* ======================================================================== *)
(* An example                                                               *)
(* ======================================================================== *)

(* The fragment holds of more than nothing: a unit that unboxes a block,
   read in its body and in a handler, and a boxed number:

     let x10 = [7; 8] in
     let x11 = box_int64 5 in
     let k20 x30 = (let x31 = x10.(0) in let x32 = [x30; x31] in
                    return x32) in
     let x12 = x10.(1) in
     let x13 = unbox_int64 x11 in
     let x14 = box_int64 x13 in
     let x15 = [x12; x14] in
     k20 x15

   Its solution, as the solver of the test tool computes it, passes the
   checks of the plain statement, and those of the unboxing with the
   tables of [dominated_by_allocation_point] and [to_unbox] that the rules
   give, which unbox [x10] and [x11] ([x14] escapes); the unit meets the
   premises of [rebuild_unbox_refines_observably_statement]. The rebuild
   binds the variables of the fields, 34 and 35 for [x10] and 33 for
   [x11], to 7, 8 and 5, and reads them in place of the loads; the unit and
   the rebuilt unit exit with the same block. *)
Module LocalExample.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition box64 (s : simple) : named :=
  Prim (Unary (Box_number Boxable_naked_int64)) [s].

Definition unbox64 (s : simple) : named :=
  Prim (Unary (Unbox_number Boxable_naked_int64)) [s].

Definition unit_expr : expr :=
  Let 10 Value (make_block [int 7; int 8])
    (Let 11 Value (box64 (Const (Naked_int64 5)))
       (Let_cont_non_recursive 20
          (Handler [30] (make_handler_metadata [Value])
             (Let 31 Value (load 0 (Var 10))
                (Let 32 Value (make_block [Var 30; Var 31])
                   (return_to 1 [Var 32]))))
          (Let 12 Value (load 1 (Var 10))
             (Let 13 (Naked_number Kind_naked_int64) (unbox64 (Var 11))
                (Let 14 Value (box64 (Var 13))
                   (Let 15 Value (make_block [Var 12; Var 14])
                      (return_to 20 [Var 15]))))))).

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition int64 : field := Field_boxed_number Boxable_naked_int64.

Definition tables : tables :=
  {| t_alias :=
       [(v 30, v 15); (m 1, v 32); (v 13, consts); (v 31, consts);
        (v 12, consts)];
     t_any_usage :=
       [m 1; m 2; v 32; v 30; v 31; consts; v 15; v 12; v 14; v 13];
     t_any_source := [consts; v 13; v 31; v 12];
     t_has_usage :=
       [m 1; m 2; v 32; v 30; v 31; consts; v 15; v 12; v 14; v 13; v 11;
        v 10];
     t_has_source :=
       [consts; v 15; v 32; v 11; v 10; v 30; m 1; v 13; v 31; v 12; v 14];
     t_usages := [(v 11, v 11); (v 10, v 10)];
     t_sources :=
       [(v 15, v 15); (v 14, v 14); (v 32, v 32); (v 11, v 11); (v 10, v 10);
        (v 30, v 15); (m 1, v 32)];
     t_reading_field := [];
     t_escaping_field := [];
     t_field_used :=
       [(v 15, Field_block 0 Value); (v 15, Field_block 1 Value);
        (v 15, Field_is_int); (v 15, Field_get_tag); (v 14, int64);
        (v 32, Field_block 0 Value); (v 32, Field_block 1 Value);
        (v 32, Field_is_int); (v 32, Field_get_tag); (v 11, int64);
        (v 10, Field_block 0 Value); (v 10, Field_block 1 Value)] |}.

Definition cannot_change : list node := derive_cannot_change facts tables.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox no_local facts tables cannot_change max_unbox_size false.

Definition rebuilt : expr :=
  Let 35 Value (Simple (int 8))
    (Let 34 Value (Simple (int 7))
       (Let 33 (Naked_number Kind_naked_int64)
          (Simple (Const (Naked_int64 5)))
          (Let_cont_non_recursive 20
             (Handler [30] (make_handler_metadata [Value])
                (Let 31 Value (Simple (Var 34))
                   (Let 32 Value (make_block [Var 30; Var 31])
                      (return_to 1 [Var 32]))))
             (Let 12 Value (Simple (Var 35))
                (Let 13 (Naked_number Kind_naked_int64) (Simple (Var 33))
                   (Let 14 Value (box64 (Var 13))
                      (Let 15 Value (make_block [Var 12; Var 14])
                         (return_to 20 [Var 15])))))))).

Definition result : value :=
  Value_block 0
    [Value_block 0
       [Value_const (Tagged_immediate 8); Value_boxed_number (Naked_int64 5)];
     Value_const (Tagged_immediate 7)].

End LocalExample.

Definition local_example_statement : Prop :=
  is_local_slots_only Examples.no_local
  ∧ in_flow_fragment LocalExample.unit_expr = true
  ∧ static_constants (unit_codes LocalExample.unit_expr)
      LocalExample.unit_expr
    = true
  ∧ results_kinded [] [] LocalExample.unit_expr = true
  ∧ well_formed_unit Examples.interface LocalExample.unit_expr = true
  ∧ exits_one_value Examples.interface LocalExample.unit_expr = true
  ∧ check_closed Examples.no_local LocalExample.facts LocalExample.tables
    = true
  ∧ check_field_used_exact Examples.no_local LocalExample.facts
      LocalExample.tables
    = true
  ∧ check_dominated LocalExample.tables LocalExample.dominated = true
  ∧ check_to_unbox Examples.no_local LocalExample.facts LocalExample.tables
      LocalExample.cannot_change Examples.max_unbox_size false
      LocalExample.to_unbox
    = true
  ∧ LocalExample.to_unbox
    = [Node_variable 11%positive; Node_variable 10%positive]
  ∧ local_unboxing LocalExample.facts LocalExample.tables
      LocalExample.dominated LocalExample.to_unbox LocalExample.unit_expr
    = true
  ∧ rebuild_unit_unbox LocalExample.facts LocalExample.tables
      LocalExample.cannot_change
      (unit_code_deps Auto Examples.interface LocalExample.unit_expr)
      LocalExample.dominated LocalExample.to_unbox true
      LocalExample.unit_expr
    = LocalExample.rebuilt
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LocalExample.unit_expr)
      (Exits 1%positive [LocalExample.result] empty_heap)
  ∧ behaves (FlambdaReaperTupledSpec.on_its_own LocalExample.rebuilt)
      (Exits 1%positive [LocalExample.result] empty_heap).

(* Of the four units of blocks of FlambdaReaperUnboxing.v, with the tables
   there, the fragment holds of the first, whose block is only read, and
   not of the others: the block is given to a continuation, returned by
   code, or held in a field of another block that is unboxed. *)
Definition fragment_examples_statement : Prop :=
  let v (x : positive) : node := Node_variable x in
  local_unboxing Examples.read_block_facts Examples.read_block_tables
    [(v 10%positive, v 10%positive)] [v 10%positive] Examples.read_block
  = true
  ∧ local_unboxing Examples.block_to_continuation_facts
      Examples.block_to_continuation_tables
      [(v 10%positive, v 10%positive); (v 10%positive, v 30%positive)]
      [v 10%positive] Examples.block_to_continuation
    = false
  ∧ local_unboxing Examples.block_from_code_facts
      Examples.block_from_code_tables Examples.block_from_code_dominated
      [v 42%positive] Examples.block_from_code
    = false
  ∧ local_unboxing Examples.nested_block_facts Examples.nested_block_tables
      [(v 11%positive, v 11%positive); (v 10%positive, v 10%positive);
       (v 10%positive, v 12%positive)]
      [v 11%positive; v 10%positive] Examples.nested_block
    = false.
