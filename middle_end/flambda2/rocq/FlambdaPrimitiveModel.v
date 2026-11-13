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

(* The primitives of Flambda 2 as the semantics sees them: a fragment.

   This file is to be reviewed, against
   middle_end/flambda2/terms/flambda_primitive.mli. It holds definitions
   only: the types of the primitives without their arguments. What each
   computes is FlambdaPrimitiveSpec.v.

   The fragment has, with the compiler's names:
   - of [unary_primitive]: [Block_load] from an immutable block of
     values, whatever the load says of mutability, [Is_int], [Is_null],
     [Get_tag], [Array_length] of an array of values, [String_length] of
     a string, [Opaque_identity], the
     unary [Int_arith] (here [Unary_int_arith]), [Num_conv] between the
     integer kinds, [Boolean_not], [Unbox_number],
     [Box_number], [Untag_immediate], [Tag_immediate] and
     [Reinterpret_64_bit_word] between a tagged immediate and an int64;
   - of [binary_primitive]: [Array_load] from an array of values,
     [String_or_bigstring_load] of an integer of one, two, four or
     eight bytes from a string, [Phys_equal], and [Int_arith],
     [Int_shift] and [Int_comp], at every [Standard_int.t];
   - of [variadic_primitive]: [Make_block] of an immutable block of
     values, and [Make_array] of an immutable array of values (here
     [Make_immutable_array]);
   - under [Heap], with the primitives of C4's prototype: [Make_array]
     of a mutable array of values (here [Make_mutable_array]), and the
     ternary [Array_set] into one;
   - under [Region_primitive], the primitives of regions that are not
     ghosts, [Begin_region], [Begin_try_region], [End_region] and
     [End_try_region], and [Local_allocation], which is no primitive of
     the compiler. The machine of FlambdaSemanticsSpec.v gives none of
     them a value; FlambdaRegionsSpec.v says what they do.
   Every other primitive is left out. The conversion from the compiler's
   terms must refuse a primitive that is not here, and a primitive whose
   fields have a value that is not here; it must never approximate one.

   Differences from the compiler:
   - [Opaque_identity] has no [middle_end_only], which has no effect on
     execution.
   - [Box_number], [Make_block] and [Make_immutable_array] have no
     allocation mode. The values they make are immutable and have no
     identity, so nothing in the model tells an allocation in a region
     from one on the heap, and the conversion of the test tool, which
     drops the regions, reads a local one as an ordinary one where it
     converts the primitive (decision D37, which is open;
     rocq/TRUST-LEDGER.md, "Regions"). The machine of
     FlambdaRegionsSpec.v is told that an allocation is local by the
     primitive [Local_allocation] after it.
   - [Make_block] has no mutability: it is that of immutable blocks
     ([Immutable], not [Immutable_unique]), and the conversion must
     refuse the others. So every block of the model is immutable.
     [Block_load] has the mutability of the compiler's, and is a load
     from such a block whatever that is: the front end marks [Mutable]
     a load that the source declares as one of a field that may be
     written ([Reads_vary]: [%field0] and [%field1], and the [!] of a
     reference), and nothing stops a source from declaring such a load
     of a tuple.
     The only [Block_access_kind.t] and [Block_kind.t] are those of
     blocks of values, [Values]; an [Or_unknown.t] is an [option],
     [Unknown] being [None].
   - [Block_kind.Values] has no list of the kinds of the fields. In the
     compiler each has a subkind, which Simplify takes an argument that
     is a variable to be of ([Simplify_variadic_primitive]). The model
     has no subkinds, so the conversion must refuse a list that says
     more of a field than [any_value].
   - A tag is any integer, where the compiler's [Tag.Scannable.t] is at
     least 0 and below 251. The field and the size of a [Block_load]
     are any integers too, where the compiler's are signed integers of
     63 bits ([Target_ocaml_int.t]). A primitive with a number out of
     its range is one the compiler cannot build. No type here refuses
     it: FlambdaPrimitiveSpec.prim_in_range is the test, and a unit
     that passes the check FlambdaWellFormedSpec.numbers_in_range has
     no such primitive.
   - The comparisons are those of integers only: the compiler's
     [comparison] and [comparison_behaviour] are polymorphic in the
     signedness so as to serve the floats too, which are not here. Their
     constructors are prefixed with [Int_], as [Eq], [Lt] and [Gt] are
     Rocq's.
   - [signedness] is both [Flambda_primitive.signed_or_unsigned] and
     [Scalar.Signedness.t], which have the same constructors.
   - The constructors of [equality_comparison] are prefixed with
     [Phys_], for the same reason: the compiler's are [Eq] and [Neq].
   - [String_length] has no [string_or_bytes]: it is the length of a
     string ([String]). The model has no bytes, which are mutable, and
     the conversion must refuse [Bytes].
   - [Num_conv] is between two [Standard_int.t], where the compiler's is
     between two [Standard_int_or_float.t], a record of [src] and
     [dst]: the conversions to and from the two kinds of float are not
     here, and the conversion must refuse them.
   - The unary [Int_arith] of the compiler is [Unary_int_arith] here, as
     [Int_arith] is the binary one.
   - [reinterpret_64_bit_word] has the two of the compiler's
     [Reinterpret_64_bit_word.t] that are between integers. The two
     that are between an int64 and a float are not here, and the
     conversion must refuse them.
   - [String_or_bigstring_load] has no [string_like_value]: it is the
     load from a string ([String]), and the conversion must refuse
     [Bytes] and [Bigstring]. The only [string_accessor_width]s are
     those of an integer of one, two, four or eight bytes; the
     conversion must refuse the others ([Single], which loads a float,
     the three of vectors, and [Mask]).
   - The only arrays are those whose elements are of kind [Value]:
     [array_kind] has [Immediates], [Gc_ignorable_values] and [Values]
     of the compiler's [Array_kind.t], and the conversion must refuse
     the others (the arrays of floats, of naked numbers, of vectors and
     of masks, and [Unboxed_product]). [array_load_kind] and
     [array_set_kind] have the same three of [Array_load_kind.t] and of
     [Array_set_kind.t]. The constructors are prefixed with [Array_],
     [Load_] and [Set_]: the three types of the compiler have the same
     names in three modules. [Float_array_opt_dynamic], the kind of an
     [Array_length] of an array that may be one of floats, is here: the
     model has no array of floats, and the conversion must refuse the
     load and the store that the front end makes for the case of
     floats.
   - [Make_array] is two primitives, by its mutability, and neither
     carries it. At [Mutable] it is [Make_mutable_array], under [Heap],
     with no allocation mode: the allocation on the heap of an array
     that has an identity. At [Immutable] it is [Make_immutable_array],
     a variadic primitive, which makes a value with no identity, as
     [Make_block] makes an immutable block. [Immutable_unique] is not a
     mutability of arrays: Simplify stops on a [Make_array] that has
     it, and the conversion must refuse it.
   - The conversion must refuse an [Array_set] whose array kind is
     [Values] and whose set kind is not. The front end makes none, and
     Simplify stops on one unless the type of the array lets it change
     the array kind to [Immediates] first
     ([Simplify_ternary_primitive.simplify_array_set], after
     [Simplify_common.specialise_array_kind]).

   To add a primitive: add its constructor here, its case to the
   evaluation, to the kinds and to [prim_in_range] of
   FlambdaPrimitiveSpec.v, and its case to the proofs of the laws of
   that file, in FlambdaPrimitiveProofs.v. Nothing else in the model
   looks inside a primitive. *)

From Stdlib Require Import ZArith.
From OxCamlFlambda2 Require Import FlambdaKindModel.

(* [Flambda_primitive.signed_or_unsigned]. *)
Inductive signedness : Type :=
  | Signed
  | Unsigned.

(* [Flambda_primitive.Block_access_field_kind.t]. *)
Inductive block_access_field_kind : Type :=
  | Any_value
  | Immediate.

(* [Flambda_primitive.Block_access_kind.t]. *)
Inductive block_access_kind : Type :=
  | Block_access_values (tag : option Z) (size : option Z)
      (field_kind : block_access_field_kind).

(* [Mutability.t]. *)
Inductive mutability : Type :=
  | Mutable
  | Immutable
  | Immutable_unique.

(* [Flambda_primitive.unary_int_arith_op]. *)
Inductive unary_int_arith_op : Type :=
  | Not
  | Swap_byte_endianness.

(* [Flambda_primitive.Reinterpret_64_bit_word.t], between integers. *)
Inductive reinterpret_64_bit_word : Type :=
  | Tagged_int63_as_unboxed_int64
  | Unboxed_int64_as_tagged_int63.

(* [Flambda_primitive.Array_kind.t], of the arrays whose elements are of
   kind [Value]. *)
Inductive array_kind : Type :=
  | Array_immediates
  | Array_gc_ignorable_values
  | Array_values.

(* [Flambda_primitive.Array_kind_for_length.t]. *)
Inductive array_kind_for_length : Type :=
  | Array_kind (kind : array_kind)
  | Float_array_opt_dynamic.

(* [Flambda_primitive.Array_load_kind.t], of the same arrays. *)
Inductive array_load_kind : Type :=
  | Load_immediates
  | Load_gc_ignorable_values
  | Load_values.

(* [Flambda_primitive.unary_primitive]. *)
Inductive unary_primitive : Type :=
  | Block_load (kind : block_access_kind) (mut : mutability) (field : Z)
  | Is_int (variant_only : bool)
  | Is_null
  | Get_tag
  | Array_length (kind : array_kind_for_length)
  | String_length
  | Opaque_identity (kind : flambda_kind)
  | Unary_int_arith (kind : standard_int) (op : unary_int_arith_op)
  | Num_conv (src dst : standard_int)
  | Boolean_not
  | Unbox_number (kind : boxable_number)
  | Box_number (kind : boxable_number)
  | Untag_immediate
  | Tag_immediate
  | Reinterpret_64_bit_word (reinterpret : reinterpret_64_bit_word).

(* [Flambda_primitive.binary_int_arith_op]. *)
Inductive binary_int_arith_op : Type :=
  | Add
  | Sub
  | Mul
  | Div (s : signedness)
  | Mod (s : signedness)
  | And
  | Or
  | Xor.

(* [Flambda_primitive.int_shift_op]. *)
Inductive int_shift_op : Type :=
  | Lsl
  | Lsr
  | Asr.

(* [Flambda_primitive.comparison], of integers. *)
Inductive int_comparison : Type :=
  | Int_eq
  | Int_neq
  | Int_lt (s : signedness)
  | Int_gt (s : signedness)
  | Int_le (s : signedness)
  | Int_ge (s : signedness).

(* [Flambda_primitive.comparison_behaviour], of integers. *)
Inductive int_comparison_behaviour : Type :=
  | Yielding_bool (c : int_comparison)
  | Yielding_int_like_compare_functions (s : signedness).

(* [Flambda_primitive.string_accessor_width], of the integers. *)
Inductive string_accessor_width : Type :=
  | Eight
  | Eight_signed
  | Sixteen
  | Sixteen_signed
  | Thirty_two
  | Sixty_four.

(* [Flambda_primitive.equality_comparison]. *)
Inductive equality_comparison : Type :=
  | Phys_eq
  | Phys_neq.

(* [Flambda_primitive.binary_primitive]. *)
Inductive binary_primitive : Type :=
  | Array_load (kind : array_kind) (load_kind : array_load_kind)
      (mut : mutability)
  | String_or_bigstring_load (width : string_accessor_width)
  | Phys_equal (op : equality_comparison)
  | Int_arith (kind : standard_int) (op : binary_int_arith_op)
  | Int_shift (kind : standard_int) (op : int_shift_op)
  | Int_comp (kind : standard_int) (behaviour : int_comparison_behaviour).

(* [Flambda_primitive.Block_kind.t]. *)
Inductive block_kind : Type :=
  | Block_kind_values (tag : Z).

(* [Flambda_primitive.variadic_primitive]. *)
Inductive variadic_primitive : Type :=
  | Make_block (kind : block_kind)
  | Make_immutable_array (kind : array_kind).

(* C4's prototype. The primitives of the heap that are not above, under
   one constructor of [prim] so that the prototype changes no
   constructor that exists. In the change itself they are where the
   compiler has them: [Make_block] gains its [Mutability.t], and
   [Block_set] is a binary primitive. The load from a block that has an
   identity is [Block_load] above, with its mutability, and the
   comparison of two such blocks is [Phys_equal] above: the prototype
   says what they do on a location (FlambdaHeapSpec.v). The two
   primitives of arrays that are here are with them for the same
   reason: in the change itself [Make_array] is a variadic primitive
   with its [Mutability.t], and [Array_set] a ternary one. The length
   and the load of an array are [Array_length] and [Array_load]
   above. *)

(* [Mutability.t] without [Immutable]: the mutability of a block that
   has an identity. *)
Inductive cell_mutability : Type :=
  | Cell_mutable
  | Cell_immutable_unique.

(* [Flambda_primitive.Init_or_assign.t], without the mode of an
   assignment. *)
Inductive init_or_assign : Type :=
  | Initialization
  | Assignment.

(* [Flambda_primitive.Array_set_kind.t], of the arrays whose elements are
   of kind [Value]. *)
Inductive array_set_kind : Type :=
  | Set_immediates
  | Set_gc_ignorable_values
  | Set_values (init : init_or_assign).

Inductive heap_primitive : Type :=
  (* [Make_block (Values (tag, _), Mutable | Immutable_unique, Heap)]:
     variadic. *)
  | Make_block_with_identity (mut : cell_mutability) (tag : Z)
  (* [Block_set { kind; init; field }]: binary. *)
  | Block_set (kind : block_access_kind) (init : init_or_assign) (field : Z)
  (* [Make_array (kind, Mutable, Heap)]: variadic. *)
  | Make_mutable_array (kind : array_kind)
  (* [Array_set (kind, set_kind)]: ternary. *)
  | Array_set (kind : array_kind) (set_kind : array_set_kind).

(* The primitives of regions, under one constructor of [prim]. The
   machine of FlambdaSemanticsSpec.v gives none of them a value: a term
   that has one is for the machine of FlambdaRegionsSpec.v, which says
   what each does, and the conversion of the test tool makes them for
   the run on that machine only.

   [Begin_region] and [Begin_try_region] are the compiler's variadic
   primitives of those names with [ghost = false], and take no
   argument: the parent region that the compiler's may be given is not
   here, as nothing runs for it. [End_region] and [End_try_region] are
   the compiler's unary primitives with [ghost = false], on the
   variable of the region. The compiler's four with [ghost = true] are
   not here: nothing runs for them either.

   [Local_allocation] is no primitive of the compiler. The allocation
   primitives and the sets of closures of the model have no allocation
   mode, so the mode [Local] of an allocation is said by a second
   [Let], put directly after that of the allocation:
   [Local_allocation] applied to the variable of the allocation, and
   then to the variable of the region that the mode names, if the
   conversion has one for it (it has none for the region of the code
   that the allocation is in, for which nothing is passed when the
   program runs). The conversion has the second [Let] bind the same
   variable again. *)
Inductive region_primitive : Type :=
  | Begin_region
  | Begin_try_region
  | End_region
  | End_try_region
  | Local_allocation.

(* [Flambda_primitive.Without_args.t]: a primitive without its
   arguments. *)
Inductive prim : Type :=
  | Unary (p : unary_primitive)
  | Binary (p : binary_primitive)
  | Variadic (p : variadic_primitive)
  | Heap (p : heap_primitive)
  | Region_primitive (p : region_primitive).
