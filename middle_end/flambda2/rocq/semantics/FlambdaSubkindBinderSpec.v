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

(* Subkinds at binders: what the machine tests of a value where a
   parameter binds it, and what follows.

   This file is to be reviewed: the statements.

   The machine has no step where a handler is applied, or the body of
   code entered, with a value that is not of what the parameter states
   of it (FlambdaSemanticsSpec.apply_cont_value and call_code, through
   FlambdaSubkindSpec.values_of_subkinds). A [Let] states nothing in
   the compiler ([Bound_var]) and has no test of a subkind here.
   Decision D25, at parameters.

   The statements:
   - [handler_binds_at_subkinds_statement],
     [call_binds_at_subkinds_statement]: the step that binds the
     parameters of a handler, or of code, binds values of what they
     state. No premise: this is the test of the machine, read back.
   - [value_of_subkind_extend_statement],
     [values_of_subkinds_extend_statement]: what a subkind says of a
     value stays true as the static environment grows, so it is still
     true where the parameter is read.
   - [values_of_subkinds_related_statement]: the test reads a value
     through the observations only.
   - [value_of_block_subkind_statement]: a value of the subkind of a
     block of one tag is a block of that tag with as many fields as the
     subkind gives, each of kind [Value] and of its subkind.
   - [load_at_block_subkind_statement]: so a load of a field of it has
     the same value whether or not its access kind names the tag and
     the size.
   - [load_of_block_subkind_statement]: and a load, with the field
     kind [Any_value], of a position for which the subkind gives an
     entry has a value, of kind [Value] and of what the entry says.
     The statement before it equates two loads, of any field kind,
     which could both have no value; this one says that with
     [Any_value] they have one.

   The proofs are in FlambdaSubkindBinderProofs.v, which needs no
   review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaSemanticsSpec FlambdaWellFormedSpec.
Import ListNotations.

(* ======================================================================== *)
(* A parameter holds a value of what it states                              *)
(* ======================================================================== *)

Definition handler_binds_at_subkinds_statement : Prop :=
  ∀ (params : list variable) (metadata : handler_metadata) (handler : expr)
    (def_vars : env) (def_conts : cont_env)
    (recursive : list (continuation * continuation_handler))
    (vs : list value) (statics : static_env) (hp : heap) (rd : read_record)
    (s : state),
    apply_cont_value
      (Closure (Handler params metadata handler) def_vars def_conts recursive)
      vs statics hp rd
    = Step s ->
    values_of_subkinds statics vs (hm_params_subkinds metadata) = true.

Definition call_binds_at_subkinds_statement : Prop :=
  ∀ (closure : value) (c : code) (vs : list value)
    (args_arity return_arity : arity) (return_to exn_to : cont_value)
    (statics : static_env) (hp : heap) (rd : read_record) (s : state),
    call_code closure c vs args_arity return_arity return_to exn_to statics hp
      rd
    = Step s ->
    values_of_subkinds statics vs (code_params_subkinds c) = true.

(* ======================================================================== *)
(* What a subkind says stays true                                           *)
(* ======================================================================== *)

Definition value_of_subkind_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (sk : subkind) (v : value),
    statics_extend statics statics' ->
    value_of_subkind statics sk v = true ->
    value_of_subkind statics' sk v = true.

Definition values_of_subkinds_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (vs : list value)
    (sks : list param_subkind),
    statics_extend statics statics' ->
    values_of_subkinds statics vs sks = true ->
    values_of_subkinds statics' vs sks = true.

(* ======================================================================== *)
(* The test reads a value through the observations only                     *)
(* ======================================================================== *)

(* Values that the observations read alike (FlambdaPrimitiveSpec.v,
   [relates_constants] and [relates_observations]) are of the same
   subkinds: where the first are, the second are. As for the
   primitives, the law goes one way. *)
Definition values_of_subkinds_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (statics statics' : static_env)
    (vs vs' : list value) (sks : list param_subkind),
    relates_constants R ->
    relates_observations R statics statics' ->
    Forall2 R vs vs' ->
    values_of_subkinds statics vs sks = true ->
    values_of_subkinds statics' vs' sks = true.

(* ======================================================================== *)
(* A block of one tag                                                       *)
(* ======================================================================== *)

(* A block of one tag with these fields: a tuple, or a record. *)
Definition block_subkind (tag : Z) (fields : field_subkinds) : subkind :=
  Subkind_variant [] (Shape_determined tag fields Shapes_nil).

Fixpoint number_of_fields (sks : field_subkinds) : nat :=
  match sks with
  | Fields_nil => 0
  | Fields_cons _ _ rest => S (number_of_fields rest)
  end.

Definition value_of_block_subkind_statement : Prop :=
  ∀ (statics : static_env) (tag : Z) (sks : field_subkinds) (v : value),
    value_of_subkind statics (block_subkind tag sks) v = true ->
    ∃ fields : list value,
      block_of statics v = Some (tag, fields)
      ∧ length fields = number_of_fields sks
      ∧ fields_of_subkinds statics sks fields = true
      ∧ Forall (fun field => kind_of_value field = Value) fields.

Definition load_at_block_subkind_statement : Prop :=
  ∀ (statics : static_env) (tag : Z) (sks : field_subkinds) (v : value)
    (field_kind : block_access_field_kind) (field : Z),
    value_of_subkind statics (block_subkind tag sks) v = true ->
    eval_block_load statics
      (Block_access_values (Some tag)
         (Some (Z.of_nat (number_of_fields sks))) field_kind)
      field v
    = eval_block_load statics (Block_access_values None None field_kind)
        field v.

(* What is said of the field at a position: whether it may be null,
   and its subkind. *)
Fixpoint field_subkind_at (sks : field_subkinds) (field : nat)
    : option param_subkind :=
  match sks, field with
  | Fields_nil, _ => None
  | Fields_cons nullable sk _, O => Some (nullable, sk)
  | Fields_cons _ _ rest, S field => field_subkind_at rest field
  end.

Definition load_of_block_subkind_statement : Prop :=
  ∀ (statics : static_env) (tag : Z) (sks : field_subkinds) (v : value)
    (field : nat) (ps : param_subkind),
    value_of_subkind statics (block_subkind tag sks) v = true ->
    field_subkind_at sks field = Some ps ->
    ∃ loaded : value,
      eval_block_load statics (Block_access_values None None Any_value)
        (Z.of_nat field) v
      = Some loaded
      ∧ kind_of_value loaded = Value
      ∧ value_at statics ps loaded = true.
