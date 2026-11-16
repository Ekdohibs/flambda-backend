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

   What is left out: subkinds ([With_subkind]), the shapes of mixed blocks,
   and [Standard_int_or_float]. Arities ([Flambda_arity.t]) are in
   FlambdaModel.v.

   A constructor is prefixed where the compiler's name is that of a
   constant (FlambdaModel.v) or of another kind here. *)

From OxCamlStdlib Require Import OxCamlLogic.

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
