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

(* The kinds of Flambda 2.

   This file is to be reviewed, against
   middle_end/flambda2/kinds/flambda_kind.mli. It holds definitions only.
   It has [Flambda_kind.t] and [Naked_number_kind.t], and the two
   restrictions of the naked number kinds that the primitives are given in
   (FlambdaPrimitiveModel.v): [Standard_int.t], the integers that
   arithmetic is done on, and [Boxable_number.t], the numbers that have a
   boxed form.

   It has, last, the subkinds of the kind [Value]
   ([With_subkind.Non_null_value_subkind.t]), for decision D25, at
   parameters: see [subkind].

   What is left out: the shapes of mixed blocks, and
   [Standard_int_or_float]. Arities ([Flambda_arity.t]) are in
   FlambdaModel.v.

   A constructor is prefixed where the compiler's name is that of a
   constant (FlambdaModel.v) or of another kind here. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic.
Import ListNotations.

(* [Flambda_kind.Naked_number_kind.t]. *)
Inductive naked_number_kind : Type :=
  | Kind_naked_immediate
  | Kind_naked_float32
  | Kind_naked_float
  | Kind_naked_int8
  | Kind_naked_int16
  | Kind_naked_int32
  | Kind_naked_int64
  | Kind_naked_nativeint
  | Kind_naked_vec128
  | Kind_naked_vec256
  | Kind_naked_vec512
  | Kind_naked_mask.

(* [Flambda_kind.t]. *)
Inductive flambda_kind : Type :=
  | Value
  | Naked_number (k : naked_number_kind)
  | Region
  | Rec_info.

Definition kind_eq_dec : ∀ k k' : flambda_kind, {k = k'} + {k <> k'}.
Proof. repeat decide equality. Defined.

(* [Flambda_kind.equal]. *)
Definition kind_eqb (k k' : flambda_kind) : bool :=
  if kind_eq_dec k k' then true else false.

(* [Flambda_kind.Standard_int.t]. *)
Inductive standard_int : Type :=
  | Standard_tagged_immediate
  | Standard_naked_immediate
  | Standard_naked_int8
  | Standard_naked_int16
  | Standard_naked_int32
  | Standard_naked_int64
  | Standard_naked_nativeint.

(* [Flambda_kind.Standard_int.to_kind]. *)
Definition kind_of_standard_int (k : standard_int) : flambda_kind :=
  match k with
  | Standard_tagged_immediate => Value
  | Standard_naked_immediate => Naked_number Kind_naked_immediate
  | Standard_naked_int8 => Naked_number Kind_naked_int8
  | Standard_naked_int16 => Naked_number Kind_naked_int16
  | Standard_naked_int32 => Naked_number Kind_naked_int32
  | Standard_naked_int64 => Naked_number Kind_naked_int64
  | Standard_naked_nativeint => Naked_number Kind_naked_nativeint
  end.

(* [Flambda_kind.Boxable_number.t]. *)
Inductive boxable_number : Type :=
  | Boxable_naked_float32
  | Boxable_naked_float
  | Boxable_naked_int32
  | Boxable_naked_int64
  | Boxable_naked_nativeint
  | Boxable_naked_vec128
  | Boxable_naked_vec256
  | Boxable_naked_vec512
  | Boxable_naked_mask.

(* [Flambda_kind.Boxable_number.unboxed_kind]. *)
Definition unboxed_kind (k : boxable_number) : flambda_kind :=
  match k with
  | Boxable_naked_float32 => Naked_number Kind_naked_float32
  | Boxable_naked_float => Naked_number Kind_naked_float
  | Boxable_naked_int32 => Naked_number Kind_naked_int32
  | Boxable_naked_int64 => Naked_number Kind_naked_int64
  | Boxable_naked_nativeint => Naked_number Kind_naked_nativeint
  | Boxable_naked_vec128 => Naked_number Kind_naked_vec128
  | Boxable_naked_vec256 => Naked_number Kind_naked_vec256
  | Boxable_naked_vec512 => Naked_number Kind_naked_vec512
  | Boxable_naked_mask => Naked_number Kind_naked_mask
  end.

(* The boxable number that a kind is, if it is a naked number kind that
   has a boxed form: the converse of [unboxed_kind]. *)
Definition boxable_number_of_kind (k : flambda_kind)
    : option boxable_number :=
  match k with
  | Naked_number Kind_naked_float32 => Some Boxable_naked_float32
  | Naked_number Kind_naked_float => Some Boxable_naked_float
  | Naked_number Kind_naked_int32 => Some Boxable_naked_int32
  | Naked_number Kind_naked_int64 => Some Boxable_naked_int64
  | Naked_number Kind_naked_nativeint => Some Boxable_naked_nativeint
  | Naked_number Kind_naked_vec128 => Some Boxable_naked_vec128
  | Naked_number Kind_naked_vec256 => Some Boxable_naked_vec256
  | Naked_number Kind_naked_vec512 => Some Boxable_naked_vec512
  | Naked_number Kind_naked_mask => Some Boxable_naked_mask
  | Naked_number (Kind_naked_immediate | Kind_naked_int8 | Kind_naked_int16)
  | Value | Region | Rec_info =>
    None
  end.

Definition boxable_number_eq_dec
    : ∀ n n' : boxable_number, {n = n'} + {n <> n'}.
Proof. decide equality. Defined.

Definition boxable_number_eqb (n n' : boxable_number) : bool :=
  if boxable_number_eq_dec n n' then true else false.

(* [Flambda_kind.With_subkind.Non_null_value_subkind.t]: what is known
   of a value of kind [Value] that is not null, beyond its kind. A
   variant is a tagged immediate among [consts], or a block whose tag is
   among those of [non_consts]; for a tag that is determined, the block
   has exactly the fields listed, each of its subkind and nullable or
   not ([constructor_shape], with [full_kind list]). The compiler keeps
   the tags in a map; here they are a list, of which the first entry
   with a given tag counts. Left out: the subkinds of arrays and of a
   block of floats, which are read as [Subkind_anything]; and the
   fields of a constructor whose fields are not all values (a mixed
   block), whose tag is kept and read as undetermined. Decision D25, at
   parameters. *)
Inductive subkind : Type :=
  | Subkind_anything
  | Subkind_tagged_immediate
  | Subkind_boxed_number (n : boxable_number)
  | Subkind_variant (consts : list Z) (non_consts : constructor_shapes)
with constructor_shapes : Type :=
  | Shapes_nil
  | Shape_undetermined (tag : Z) (rest : constructor_shapes)
  | Shape_determined (tag : Z) (fields : field_subkinds)
      (rest : constructor_shapes)
with field_subkinds : Type :=
  | Fields_nil
  | Fields_cons (nullable : bool) (sk : subkind) (rest : field_subkinds).

Fixpoint subkind_eq_dec (sk sk' : subkind) {struct sk}
    : {sk = sk'} + {sk <> sk'}
with constructor_shapes_eq_dec (shapes shapes' : constructor_shapes)
    {struct shapes} : {shapes = shapes'} + {shapes <> shapes'}
with field_subkinds_eq_dec (sks sks' : field_subkinds) {struct sks}
    : {sks = sks'} + {sks <> sks'}.
Proof.
  - decide equality;
      [ apply boxable_number_eq_dec | apply (list_eq_dec Z.eq_dec) ].
  - decide equality; apply Z.eq_dec.
  - decide equality. decide equality.
Defined.

(* [Flambda_kind.With_subkind.t] without its kind: whether the value may
   be null, and its subkind if it is not. *)
Definition param_subkind : Type := (bool * subkind)%type.

Definition param_subkind_eq_dec
    : ∀ ps ps' : param_subkind, {ps = ps'} + {ps <> ps'}.
Proof.
  decide equality; [ apply subkind_eq_dec | decide equality ].
Defined.

Definition param_subkinds_eqb (sks sks' : list param_subkind) : bool :=
  if list_eq_dec param_subkind_eq_dec sks sks' then true else false.

(* [Flambda_kind.With_subkind.any_value], which says nothing. *)
Definition any_value : param_subkind := (true, Subkind_anything).

(* Whether every value of the kind is of the subkind. *)
Definition says_nothing (nullable : bool) (sk : subkind) : bool :=
  nullable && match sk with Subkind_anything => true | _ => false end.

(* Whether nothing is stated of any parameter: each entry is one that
   every value of the kind is of. The empty list states nothing. *)
Definition subkinds_say_nothing (sks : list param_subkind) : bool :=
  forallb (fun ps => says_nothing (fst ps) (snd ps)) sks.
