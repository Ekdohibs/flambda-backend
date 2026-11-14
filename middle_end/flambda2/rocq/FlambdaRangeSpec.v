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

(* What a unit whose numbers are in range gains: no state of its run
   holds a number out of range.

   This file is to be reviewed: [state_in_range] and what it is made of,
   and the statements.

   The check is FlambdaWellFormedSpec.numbers_in_range, a condition on
   the term alone: every tag of a block, every constant and every other
   number of the term is one that the compiler's can be.
   FlambdaPrimitiveSpec.v says which those are (the section "Numbers in
   range") and what it is for a value to be in range
   ([value_in_range]): its constants are in the range of their kind and
   its blocks have tags in range, all the way down.

   A state is in range ([state_in_range]) when everything it holds is:
   the expression left to run; the values of its variables; the handlers
   and the environments that its continuations keep; what its symbols
   are bound to, and its code; and the cells of its heap
   ([heap_in_range]). The cells are there because a step reads them: a
   load from a location brings to a variable a field or an element of
   its cell, and [Get_tag] the tag of a block that has an identity.
   Where a primitive has a value on a heap, the value and the heap
   after are in range, given that the primitive's own numbers are in
   range, that its arguments are, that what each symbol among them is
   bound to is, and that the heap before is
   ([exec_prim_in_range_statement]). A unit that passes the check
   starts in such a state, with no cell; a step keeps a state so
   ([step_in_range_statement]), and what the run leaves with is in
   range too ([run_in_range_statement],
   [run_exit_in_range_statement]).

   The check is a condition of well-formedness
   (FlambdaWellFormedSpec.well_formed_unit), so every state of the run
   of a well-formed unit is in range
   ([well_formed_in_range_statement]). Two things follow for such a
   state.
   - Every block that a variable holds, made by a primitive or static,
     has a tag that the compiler's blocks can have, and so have the
     blocks in its fields ([blocks_in_range_statement]). The machine
     does not test the tag where a block is made; this is why it need
     not.
   - Every constant of an integer kind that a variable holds is an
     integer of that kind ([integers_in_range_statement]). The
     arithmetic, the comparisons, tagging and untagging have no value
     on a constant out of the range of its kind
     (FlambdaPrimitiveSpec.int_of_value); in such a run that case does
     not arise, for a variable or for a constant written in the term.

   The check looks at no name, so a renaming of the term leaves it as
   it is ([numbers_in_range_renamed_statement]).

   Three units are run, so that the statements are not about units that
   all go wrong, and to show what the check keeps out. One passes, and
   leaves with a block of the greatest tag and an int8 that has wrapped
   ([in_range_runs_statement]). One makes a block with tag 300: the
   check refuses it, and its run leaves with that block
   ([tag_out_of_range_statement]). One adds to the int8 200: the check
   refuses it, and its run goes wrong
   ([constant_out_of_range_statement]).

   The proofs are in FlambdaRangeProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List.
From OxCamlStdlib Require Import OxCamlLogic OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec FlambdaNominalSpec.
Import ListNotations.

(* ======================================================================== *)
(* States in range                                                          *)
(* ======================================================================== *)

(* Every value of the environment is in range, those that are shadowed
   included. *)
Definition env_in_range (vars : env) : Prop :=
  ∀ (x : variable) (v : value), (x, v) ∈ vars -> value_in_range v = true.

(* What a continuation stands for is in range: a way out is; a handler
   with the environments of its definition is when the handler, the
   values of its variables, what its continuations stand for, and the
   handlers defined with it are; a continuation with extra arguments is
   when the continuation inside and the extra arguments are; what is
   left of an over-application is when the arguments that remain and its
   two continuations are. *)
Inductive cont_in_range : cont_value -> Prop :=
  | Outer_in_range (k : continuation) : cont_in_range (Outer k)
  | Extra_args_in_range (c : cont_value) (extra : list value) :
    cont_in_range c ->
    forallb value_in_range extra = true ->
    cont_in_range (Extra_args c extra)
  | Closure_in_range (h : continuation_handler) (vars : env)
      (conts : list (continuation * cont_value))
      (recursive : list (continuation * continuation_handler)) :
    handler_numbers_in_range h = true ->
    env_in_range vars ->
    (∀ (k : continuation) (c : cont_value),
      (k, c) ∈ conts -> cont_in_range c) ->
    (∀ (k : continuation) (h' : continuation_handler),
      (k, h') ∈ recursive -> handler_numbers_in_range h' = true) ->
    cont_in_range (Closure h vars conts recursive)
  | Over_application_in_range (remaining : list value)
      (remaining_arity return_arity : arity)
      (return_to exn_to : cont_value) :
    forallb value_in_range remaining = true ->
    cont_in_range return_to ->
    cont_in_range exn_to ->
    cont_in_range
      (Over_application remaining remaining_arity return_arity return_to
         exn_to).

Definition conts_in_range (conts : cont_env) : Prop :=
  ∀ (k : continuation) (c : cont_value), (k, c) ∈ conts -> cont_in_range c.

(* The code that is bound, and what the symbols are bound to. *)
Definition statics_in_range (statics : static_env) : Prop :=
  (∀ (id : code_id) (c : code),
    (id, c) ∈ static_code statics -> code_numbers_in_range c = true)
  ∧ (∀ (s : symbol) (allocated : statically_allocated_value),
      (s, allocated) ∈ static_values statics ->
      allocated_in_range allocated = true).

(* A cell is in range: the tag of a block with an identity is one
   that the compiler's blocks can have, and its fields are in range;
   the elements of an array are in range. *)
Definition cell_in_range (c : cell) : bool :=
  match c with
  | Cell_block _ tag fields =>
    tag_in_range tag && forallb value_in_range fields
  | Cell_array _ elements => forallb value_in_range elements
  end.

(* Every cell of the heap is in range, those that a later write hides
   included. *)
Definition heap_in_range (h : heap) : Prop :=
  ∀ (l : location) (c : cell), (l, c) ∈ h -> cell_in_range c = true.

Definition state_in_range (s : state) : Prop :=
  numbers_in_range (state_expr s) = true
  ∧ env_in_range (state_vars s)
  ∧ conts_in_range (state_conts s)
  ∧ statics_in_range (state_statics s)
  ∧ heap_in_range (state_heap s).

(* ======================================================================== *)
(* A primitive on a heap in range                                           *)
(* ======================================================================== *)

(* A primitive on a heap brings no number out of range:
   FlambdaPrimitiveSpec.eval_prim_in_range_statement on a heap whose
   cells are in range. Where the primitive has a value, that value and
   the heap after are in range, given that the primitive's own numbers
   are in range, that its arguments are, that what each symbol among
   the arguments is bound to is, and that the heap before is. So what
   it reads of a cell is in range, the tag that [Get_tag] reads
   included, and the cells that it makes or changes are in range.
   Nothing is said of a primitive that has no value. *)
Definition exec_prim_in_range_statement : Prop :=
  ∀ (statics : static_env) (h h' : heap) (p : prim) (vs : list value)
    (v : value),
    prim_in_range p = true ->
    (∀ (s : symbol) (allocated : statically_allocated_value),
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      allocated_in_range allocated = true) ->
    (∀ arg : value, arg ∈ vs -> value_in_range arg = true) ->
    heap_in_range h ->
    exec_prim statics h p vs = Some (v, h') ->
    value_in_range v = true ∧ heap_in_range h'.

(* ======================================================================== *)
(* A run keeps a state in range                                             *)
(* ======================================================================== *)

Definition step_in_range_statement : Prop :=
  ∀ s s' : state, state_in_range s -> step s = Step s' -> state_in_range s'.

(* A unit whose numbers are in range, run on its own, is in range in
   every state, and leaves with values in range. *)
Definition run_in_range_statement : Prop :=
  ∀ (e : expr) (n : nat) (s : state),
    numbers_in_range e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_in_range s.

Definition run_exit_in_range_statement : Prop :=
  ∀ (e : expr) (n : nat) (k : continuation) (vs : list value) (hp : heap)
    (rd : read_record),
    numbers_in_range e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Exit k vs hp rd
      ->
    ∀ v : value, v ∈ vs -> value_in_range v = true.

(* So is every state of the run of a well-formed unit. *)
Definition well_formed_in_range_statement : Prop :=
  ∀ (u : unit_interface) (e : expr) (n : nat) (s : state),
    well_formed_unit u e = true ->
    run n (State e [] [] empty_static_env empty_heap no_reads) = Step s ->
    state_in_range s.

(* ======================================================================== *)
(* What a state in range holds                                              *)
(* ======================================================================== *)

(* A block that a variable holds, made by a primitive or static, has a
   tag in range, and its fields are in range. *)
Definition blocks_in_range_statement : Prop :=
  ∀ (s : state) (x : variable) (v : value) (tag : Z) (fields : list value),
    state_in_range s ->
    lookup x (state_vars s) = Some v ->
    block_of (state_statics s) v = Some (tag, fields) ->
    tag_in_range tag = true
    ∧ ∀ field : value, field ∈ fields -> value_in_range field = true.

(* A constant of an integer kind that a variable holds is an integer of
   that kind. *)
Definition integers_in_range_statement : Prop :=
  ∀ (s : state) (x : variable) (kind : standard_int) (i : Z),
    state_in_range s ->
    lookup x (state_vars s) = Some (Value_const (const_of_int kind i)) ->
    int_of_value kind (Value_const (const_of_int kind i)) = Some i.

(* ======================================================================== *)
(* Renaming                                                                 *)
(* ======================================================================== *)

Definition numbers_in_range_renamed_statement : Prop :=
  ∀ (r : renaming) (e : expr),
    numbers_in_range (rename_expr r e) = numbers_in_range e.

(* ======================================================================== *)
(* Examples                                                                 *)
(* ======================================================================== *)

Section Examples.

Local Open Scope positive_scope.

Definition done : continuation := 1.
Definition x : variable := 1.
Definition y : variable := 2.

Definition int8_add (a b : Z) : named :=
  Prim (Binary (Int_arith Standard_naked_int8 Add))
    [Const (Naked_int8 a); Const (Naked_int8 b)].

(* In range: the greatest int8 and 1 are added, and a block with the
   greatest tag is made. *)
Definition in_range_unit : expr :=
  Let x (Naked_number Kind_naked_int8) (int8_add 127 1)
    (Let y Value
       (Prim (Variadic (Make_block (Block_kind_values 250)))
          [Const (Tagged_immediate 1)])
       (Apply_cont (Apply_cont_expr done [Var y; Var x] None))).

Definition in_range_runs_statement : Prop :=
  numbers_in_range in_range_unit = true
  ∧ behaves (State in_range_unit [] [] empty_static_env empty_heap no_reads)
      (Exits done
         [Value_block 250 [Value_const (Tagged_immediate 1)];
          Value_const (Naked_int8 (-128))] empty_heap).

(* A block with a tag that no block of the compiler has. The machine
   makes it. *)
Definition tag_out_of_range_unit : expr :=
  Let x Value (Prim (Variadic (Make_block (Block_kind_values 300))) [])
    (Apply_cont (Apply_cont_expr done [Var x] None)).

Definition tag_out_of_range_statement : Prop :=
  numbers_in_range tag_out_of_range_unit = false
  ∧ behaves
    (State tag_out_of_range_unit [] [] empty_static_env empty_heap no_reads)
      (Exits done [Value_block 300 []] empty_heap).

(* An int8 that is 200. The addition has no value on it. *)
Definition constant_out_of_range_unit : expr :=
  Let x (Naked_number Kind_naked_int8) (int8_add 200 0)
    (Apply_cont (Apply_cont_expr done [Var x] None)).

Definition constant_out_of_range_statement : Prop :=
  numbers_in_range constant_out_of_range_unit = false
  ∧ behaves
    (State constant_out_of_range_unit [] [] empty_static_env empty_heap
       no_reads)
      Goes_wrong.

End Examples.
