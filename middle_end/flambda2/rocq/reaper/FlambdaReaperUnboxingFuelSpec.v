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

(* The depth of the model's unboxing of a block
   (FlambdaReaperUnboxing.mk_unboxed_fields_full) against the compiler's
   [mk_unboxed_fields] (unboxing_analysis.ml), whose recursion has no
   bound.

   This file is to be reviewed: [fields_read], [nested_call],
   [unboxing_ends], [calls_within], [unboxes], [unboxing_depth],
   [nesting], [LoopField] and the statements. The proofs are in
   FlambdaReaperUnboxingFuelProofs.v, which needs no review.

   The compiler's recursion. [unboxes block fields next r]: the
   compiler's [mk_unboxed_fields] of the allocation [block] and of the
   fields [fields] gives [r], its variables named from [next] on as the
   model names them. A field is left out, a variable, or unboxed in turn:
   the fields of its one source read from the uses of the names that it
   is read into ([fields_read]), with no bound on the depth. Where the
   compiler stops with a fatal error (a field of an unexpected kind, a
   field read into names all to be unboxed with several sources, or into
   names some to be unboxed and some not) the relation has what the
   model has, as FlambdaReaperUnboxing.mk_unboxed_fields_with says;
   elsewhere it is the compiler's recursion. [nested_call block fields
   source fields']: the recursion on [block] and [fields] calls itself on
   [source] and [fields'], for a field that it unboxes in turn.
   [unboxing_ends block fields]: no chain of such calls from [block] and
   [fields] is infinite, which is when the compiler's recursion ends (it
   has no other loop). [calls_within n block fields]: no chain has more
   than [n] calls.

   The model. [unboxing_depth T] is
   [length (t_sources T) * 2 ^ length (t_usages T)], the depth of
   [mk_unboxed_fields_full], which does not compute it.

   What is stated.
   - [full_depth_statement]: [mk_unboxed_fields_full] is
     [mk_unboxed_fields] at the depth [unboxing_depth T], for any facts
     and tables.
   - [ends_within_statement]: where the compiler's recursion ends, no
     chain of calls is longer than [unboxing_depth T]. The call that a
     field makes is given by its source, one of [t_sources T], and by
     which of the first names of the pairs of [t_usages T] are among the
     names that it is read into ([get_direct_usages] reads nothing
     else); so at most [unboxing_depth T] calls differ. A call that comes
     back in a chain of calls makes the chain infinite; so where the
     recursion ends, the calls of a chain are distinct.
     [within_ends_statement]: the converse, for any depth. The
     compiler's recursion ends if and only if its chains are no longer
     than [unboxing_depth T].
   - [unboxes_ends_statement]: the relation gives a result, for every
     [next], where the recursion ends, and only there.
   - [fuel_enough_statement]: where the relation gives a result, the
     model gives that one: the fuel is enough. So the relation gives at
     most one.
   - [more_fuel_statement]: where the recursion ends, every depth from
     [unboxing_depth T] on gives the model's result: more fuel changes
     nothing. [fuel_changes_statement]: where it does not end, one more
     level changes the unboxing, at every depth. [ends_stable_statement]:
     so the recursion ends if and only if the depths [unboxing_depth T]
     and [S (unboxing_depth T)] give the same unboxing.
   - [name_unboxed_fields_relation_statement]: the unboxed fields of a
     name are those that the relation gives, where it gives some. The
     fields of a changed representation are [mk_unboxed_fields_full] of
     [get_fields_of_constructors] in the same way.
   - [loop_field_statement]: a unit where the compiler's recursion does
     not end ([LoopField]). A loop makes a block [x10] of its parameter
     [x30] and 7, reads [x11 = x30.(0)] and [x12 = x30.(1)], and goes
     round again with [x10] or [x11], or returns [x12]; it is entered
     from a handler never applied, so that [x10] is the one source of
     [x30]. [x10] is to be unboxed, and its field 0 is read into [x11],
     which is to be unboxed too, as are the names it flows to: the call
     for that field is on [x10] and the fields read from the uses of
     [x11], which are those read from the uses of [x10]. The call is the
     one it is made from, and the recursion does not end. Each depth
     gives a different unboxing, the depth [n] one whose fields nest [n]
     levels ([nesting]), and the model, at the depth
     [unboxing_depth T] = 48, one of 48 levels. No fuel makes the model
     agree with the compiler, which has no result. Nor are two depths
     [n] and [S n] unboxed alike ([unboxed_field_pairs] of the two is
     [None]: where the first has a variable, the second unboxes), which
     is how a slot holding such a block, unboxed one level below its
     closure, and the name of the block, unboxed from the top, would
     differ at any fuel (not shown on a unit: the model's unboxing there
     is too deep to compute). *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaWellFormedSpec FlambdaReaperGraphSpec
  FlambdaReaperTraversal FlambdaReaperFlowSpec FlambdaReaperLeastSpec
  FlambdaReaperUnboxing.
From OxCamlFlambda2 Require FlambdaReaperLeast.
Import ListNotations.

(* ======================================================================== *)
(* The compiler's recursion                                                 *)
(* ======================================================================== *)

Section Recursion.

Variable F : facts.
Variable T : tables.
Variable code_my_closures : finite_map variable.
Variable dominated : list (node * node).
Variable to_unbox : list node.

(* The fields read from the uses of the names [flow_to]: those of the
   recursive call for a field read into them. *)
Definition fields_read (flow_to : list node) : list (field * field_use) :=
  get_fields F T
    (add_usages_through_function_slots F T code_my_closures true
       (get_direct_usages T flow_to)).

(* The recursion on [block] and [fields] calls itself on [source] and
   [fields']. *)
Inductive nested_call (block : node) (fields : list (field * field_use))
    : node -> list (field * field_use) -> Prop :=
  | Nested_call (f : field) (flow_to : list node) (source : node) :
      (f, Use_known flow_to) ∈ fields ->
      is_data_field f = true ->
      get_single_field_source F T block f = One_source source ->
      forallb (has_to_be_unboxed dominated to_unbox) flow_to = true ->
      nested_call block fields source (fields_read flow_to).

(* The recursion on [block] and [fields] ends. *)
Inductive unboxing_ends (block : node) (fields : list (field * field_use))
    : Prop :=
  | Unboxing_ends :
      (∀ (source : node) (fields' : list (field * field_use)),
         nested_call block fields source fields' ->
         unboxing_ends source fields') ->
      unboxing_ends block fields.

(* No chain of calls from [block] and [fields] has more than [n]. *)
Fixpoint calls_within (n : nat) (block : node)
    (fields : list (field * field_use)) : Prop :=
  ∀ (source : node) (fields' : list (field * field_use)),
    nested_call block fields source fields' ->
    match n with
    | O => False
    | S n => calls_within n source fields'
    end.

(* The compiler's [mk_unboxed_fields], field by field. A field of a kind
   that is not data, or with no source, is left out; one read into a
   name of any usage, or into names not all to be unboxed, is a variable,
   and so is one read into names all to be unboxed with several sources;
   one read into names all to be unboxed with one source is unboxed in
   turn, from the next variable on. *)
Inductive unboxes
    : node -> list (field * field_use) -> positive
      -> unboxed_fields * positive -> Prop :=
  | Unboxes_nil (block : node) (next : positive) :
      unboxes block [] next ([], next)
  | Unboxes_left_out (block : node) (f : field) (use : field_use)
      (fields : list (field * field_use)) (next : positive)
      (r : unboxed_fields * positive) :
      is_data_field f = false
      ∨ get_single_field_source F T block f = No_source ->
      unboxes block fields next r ->
      unboxes block ((f, use) :: fields) next r
  | Unboxes_unknown (block : node) (f : field)
      (fields : list (field * field_use)) (next : positive)
      (rest : unboxed_fields) (next' : positive) :
      is_data_field f = true ->
      get_single_field_source F T block f <> No_source ->
      unboxes block fields (Pos.succ next) (rest, next') ->
      unboxes block ((f, Use_unknown) :: fields) next
        ((f, Not_unboxed next) :: rest, next')
  | Unboxes_kept (block : node) (f : field) (flow_to : list node)
      (fields : list (field * field_use)) (next : positive)
      (rest : unboxed_fields) (next' : positive) :
      is_data_field f = true ->
      get_single_field_source F T block f <> No_source ->
      forallb (has_to_be_unboxed dominated to_unbox) flow_to = false ->
      unboxes block fields (Pos.succ next) (rest, next') ->
      unboxes block ((f, Use_known flow_to) :: fields) next
        ((f, Not_unboxed next) :: rest, next')
  | Unboxes_many (block : node) (f : field) (flow_to : list node)
      (fields : list (field * field_use)) (next : positive)
      (rest : unboxed_fields) (next' : positive) :
      is_data_field f = true ->
      get_single_field_source F T block f = Many_sources ->
      forallb (has_to_be_unboxed dominated to_unbox) flow_to = true ->
      unboxes block fields (Pos.succ next) (rest, next') ->
      unboxes block ((f, Use_known flow_to) :: fields) next
        ((f, Not_unboxed next) :: rest, next')
  | Unboxes_nested (block : node) (f : field) (flow_to : list node)
      (source : node) (fields : list (field * field_use)) (next : positive)
      (nested : unboxed_fields) (next1 : positive) (rest : unboxed_fields)
      (next2 : positive) :
      is_data_field f = true ->
      get_single_field_source F T block f = One_source source ->
      forallb (has_to_be_unboxed dominated to_unbox) flow_to = true ->
      unboxes source (fields_read flow_to) next (nested, next1) ->
      unboxes block fields next1 (rest, next2) ->
      unboxes block ((f, Use_known flow_to) :: fields) next
        ((f, Unboxed nested) :: rest, next2).

End Recursion.

(* The depth of [mk_unboxed_fields_full]. *)
Definition unboxing_depth (T : tables) : nat :=
  length (t_sources T) * 2 ^ length (t_usages T).

(* How many levels the fields of an unboxing nest: 0 if none is unboxed
   in turn. *)
Fixpoint unboxed_nesting (u : unboxed_field) : nat :=
  match u with
  | Not_unboxed _ => 0
  | Unboxed fields =>
    S ((fix deepest (fields : list (field * unboxed_field)) : nat :=
          match fields with
          | [] => 0
          | (_, u) :: fields => Nat.max (unboxed_nesting u) (deepest fields)
          end) fields)
  end.

Definition nesting (fields : unboxed_fields) : nat :=
  Nat.pred (unboxed_nesting (Unboxed fields)).

(* ======================================================================== *)
(* A recursion that does not end                                            *)
(* ======================================================================== *)

(* The loop 21 of parameter [x30] makes [x10 = [x30; 7]], reads
   [x11 = x30.(0)] and [x12 = x30.(1)], and goes round with [x11] (arm 0)
   or [x10] (arm 1), or returns [x12] (arm 2). It is entered from the
   handler 20, never applied, whose parameter has no source. *)
Module LoopField.

Import FlambdaReaperUnboxing.Examples.
Local Open Scope positive_scope.

Definition goto (k : continuation) (args : list simple) : apply_cont :=
  {| ac_continuation := k; ac_args := args; ac_trap_action := None |}.

Definition loop : expr :=
  Let 10 Value (make_block [Var 30; int 7])
    (Let 11 Value (load 0 (Var 30))
       (Let 12 Value (load 1 (Var 30))
          (Let_cont_non_recursive 24
             (Handler [] (make_handler_metadata []) (return_to 1 [Var 12]))
             (Let_cont_non_recursive 22
                (Handler [] (make_handler_metadata [])
                   (Apply_cont (goto 21 [Var 11])))
                (Let_cont_non_recursive 23
                   (Handler [] (make_handler_metadata [])
                      (Apply_cont (goto 21 [Var 10])))
                   (Switch (Const (Naked_immediate 0))
                      [(0%Z, goto 22 []); (1%Z, goto 23 []);
                       (2%Z, goto 24 [])])))))).

Definition unit_expr : expr :=
  Let_cont_non_recursive 20
    (Handler [31] (make_handler_metadata [Value])
       (Let_cont_recursive
          [(21, Handler [30] (make_handler_metadata [Value]) loop)]
          (Apply_cont (goto 21 [Var 31]))))
    Invalid.

Definition facts : facts := traverse_unit Auto interface unit_expr.

Definition tables : tables :=
  match FlambdaReaperLeast.compute_least no_local facts 100 with
  | Some T => T
  | None => FlambdaReaperLeast.empty_tables
  end.

Definition dominated : list (node * node) := derive_dominated tables.

Definition to_unbox : list node :=
  derive_to_unbox_fixed no_local facts tables [] max_unbox_size false.

Definition my_closures : finite_map variable :=
  bound_code_my_closures unit_expr.

(* The fields of [x10] read from its uses, as [name_unboxed_fields] reads
   them. *)
Definition fields_10 : list (field * field_use) :=
  fields_read facts tables my_closures [Node_variable 10].

Definition unboxed : list (node * unboxed_fields) :=
  fst (unit_unboxed_fields facts tables dominated to_unbox unit_expr).

End LoopField.

(* ======================================================================== *)
(* The statements                                                           *)
(* ======================================================================== *)

Definition full_depth_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)) (next : positive),
    mk_unboxed_fields_full F T code_my_closures dominated to_unbox block
      fields next
    = mk_unboxed_fields F T code_my_closures dominated to_unbox
        (unboxing_depth T) block fields next.

Definition ends_within_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)),
    unboxing_ends F T code_my_closures dominated to_unbox block fields ->
    calls_within F T code_my_closures dominated to_unbox (unboxing_depth T)
      block fields.

Definition within_ends_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (n : nat)
    (block : node) (fields : list (field * field_use)),
    calls_within F T code_my_closures dominated to_unbox n block fields ->
    unboxing_ends F T code_my_closures dominated to_unbox block fields.

Definition unboxes_ends_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)),
    (unboxing_ends F T code_my_closures dominated to_unbox block fields ->
     ∀ next : positive,
       ∃ r : unboxed_fields * positive,
         unboxes F T code_my_closures dominated to_unbox block fields next r)
    ∧ (∀ (next : positive) (r : unboxed_fields * positive),
         unboxes F T code_my_closures dominated to_unbox block fields next r ->
         unboxing_ends F T code_my_closures dominated to_unbox block fields).

Definition fuel_enough_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)) (next : positive)
    (r : unboxed_fields * positive),
    unboxes F T code_my_closures dominated to_unbox block fields next r ->
    mk_unboxed_fields_full F T code_my_closures dominated to_unbox block
      fields next
    = r.

Definition more_fuel_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)) (next : positive) (n : nat),
    unboxing_ends F T code_my_closures dominated to_unbox block fields ->
    unboxing_depth T <= n ->
    mk_unboxed_fields F T code_my_closures dominated to_unbox n block fields
      next
    = mk_unboxed_fields_full F T code_my_closures dominated to_unbox block
        fields next.

Definition fuel_changes_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)) (next : positive) (n : nat),
    ¬ unboxing_ends F T code_my_closures dominated to_unbox block fields ->
    fst (mk_unboxed_fields F T code_my_closures dominated to_unbox n block
           fields next)
    <> fst (mk_unboxed_fields F T code_my_closures dominated to_unbox (S n)
              block fields next).

Definition ends_stable_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (block : node)
    (fields : list (field * field_use)) (next : positive),
    unboxing_ends F T code_my_closures dominated to_unbox block fields
    <-> fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
               (unboxing_depth T) block fields next)
        = fst (mk_unboxed_fields F T code_my_closures dominated to_unbox
                 (S (unboxing_depth T)) block fields next).

Definition name_unboxed_fields_relation_statement : Prop :=
  ∀ (F : facts) (T : tables) (code_my_closures : finite_map variable)
    (dominated : list (node * node)) (to_unbox : list node) (a x : node)
    (next : positive) (r : unboxed_fields * positive),
    unboxes F T code_my_closures dominated to_unbox a
      (fields_read F T code_my_closures [x]) next r ->
    name_unboxed_fields F T code_my_closures dominated to_unbox a x next = r.

(* The unit is of the fragment; its tables pass [check_least], and are
   so the least solution (FlambdaReaperLeastSpec.v,
   [check_least_sound_statement]); [dominated] passes its check, and
   [x10] is the one allocation to unbox; the recursion on [x10] and its
   fields does not end; the depth [n] gives fields that nest [n] levels,
   and are not unboxed alike with those of the depth [S n]; and the
   model's fields of [x10] nest 48. *)
Definition loop_field_statement : Prop :=
  in_flow_fragment LoopField.unit_expr = true
  ∧ well_formed_unit FlambdaReaperUnboxing.Examples.interface
      LoopField.unit_expr
    = true
  ∧ variables_bound_once LoopField.unit_expr = true
  ∧ FlambdaReaperLeast.check_least FlambdaReaperUnboxing.Examples.no_local
      LoopField.facts 100 LoopField.tables
    = true
  ∧ check_dominated LoopField.tables LoopField.dominated = true
  ∧ LoopField.to_unbox = [Node_variable 10%positive]
  ∧ ¬ unboxing_ends LoopField.facts LoopField.tables LoopField.my_closures
        LoopField.dominated LoopField.to_unbox (Node_variable 10%positive)
        LoopField.fields_10
  ∧ (∀ (n : nat) (next : positive),
       nesting
         (fst (mk_unboxed_fields LoopField.facts LoopField.tables
                 LoopField.my_closures LoopField.dominated LoopField.to_unbox
                 n (Node_variable 10%positive) LoopField.fields_10 next))
       = n)
  ∧ (∀ (n : nat) (next next' : positive),
       unboxed_field_pairs Value
         (Unboxed
            (fst (mk_unboxed_fields LoopField.facts LoopField.tables
                    LoopField.my_closures LoopField.dominated
                    LoopField.to_unbox n (Node_variable 10%positive)
                    LoopField.fields_10 next)))
         (Unboxed
            (fst (mk_unboxed_fields LoopField.facts LoopField.tables
                    LoopField.my_closures LoopField.dominated
                    LoopField.to_unbox (S n) (Node_variable 10%positive)
                    LoopField.fields_10 next')))
       = None)
  ∧ unboxing_depth LoopField.tables = 48
  ∧ option_map nesting
      (Keyed.lookup node_eqb (Node_variable 10%positive) LoopField.unboxed)
    = Some 48.
