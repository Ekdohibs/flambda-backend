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

(* What the primitives of FlambdaPrimitiveModel.v compute, their kinds,
   and the laws that every primitive obeys.

   This file is to be reviewed: [eval_prim] and what it is made of, against
   middle_end/flambda2/terms/flambda_primitive.mli for what each primitive
   is, and against middle_end/flambda2/simplify/simplify_unary_primitive.ml,
   simplify_binary_primitive.ml, simplify_variadic_primitive.ml and
   number_adjuncts.ml for what Simplify takes each to compute; the kinds,
   against [arg_kind_of_unary_primitive], [args_kind_of_binary_primitive],
   [args_kind_of_variadic_primitive] and the [result_kind_of_] functions
   of flambda_primitive.ml; and the statements at the end.

   [eval_prim statics p vs] is the value of the primitive [p] applied to
   the values [vs], if it has one. Where it has none the term is
   undefined (FlambdaSemanticsSpec.v), and a transformation may do
   anything to it. A primitive has no value
   - when it is not given as many arguments as it takes;
   - when an argument is not of the kind it takes: this is what Simplify
     assumes, which replaces a primitive applied to an argument of the
     wrong kind by [Invalid] ([Simplify_primitive.arg_kind_mismatch]);
   - when an argument is not what the primitive is for, even if of the
     right kind: the cases are listed at each primitive below. Most are
     there because Simplify assumes that the case does not arise, where
     the machine would compute something;
   - when it computes with an integer that is out of the range of its
     kind;
   - when an argument is a symbol that is bound to nothing
     ([eval_prim_symbols_bound_statement]): any operation on such a
     symbol is undefined, and a well-formed term has none.

   The integers. The model is of a 64-bit target: the tagged and the
   naked immediates are OCaml's [int], of 63 bits, and the other kinds
   have 8, 16, 32 and 64 bits ([int_width]). A constant holds any integer
   (FlambdaModel.v); the arithmetic, the comparisons, tagging and
   untagging take only integers in the range of their kind
   ([int_of_value]), and their results are in range. Arithmetic is on
   mathematical integers and wraps ([wrap]), which is two's complement.

   Numbers in range. The compiler's numbers are of types that hold a
   range only, and the model's are integers. The section "Numbers in
   range" says which integers the compiler's are: a constant of each
   kind ([const_in_range]), the tag of a block ([tag_in_range]), the
   numbers that a primitive carries ([prim_in_range]), and a value all
   of whose numbers are such ([value_in_range]). A unit that passes the
   check FlambdaWellFormedSpec.numbers_in_range has no other, and no
   primitive makes one ([eval_prim_in_range_statement]), so the run of
   such a unit never meets one (FlambdaRangeSpec.v).
   FlambdaPrimitiveProofs.v shows that the results of arithmetic are in
   range, that at 64 bits [wrap], [unsigned] and [in_range] are the
   [wrap64], [unsigned64] and [fits_int64] of rocq/OxCamlInt64.v, and
   that at 63 bits addition, subtraction and multiplication are those of
   OCaml's [int] as rocq/OxCamlInt.v models it.

   Allocated values are read through the observations of
   FlambdaValueSpec.v ([block_of], ...), which look through a symbol to
   what it is bound to. So the value of a primitive may depend on the
   static environment, but only on what the symbols among its arguments
   are bound to ([eval_prim_static_values_statement]).

   Arrays. An array that has an identity is a cell of the heap
   (FlambdaHeapSpec.v), and [eval_prim] has no value on an access to
   one, as it has none on the primitives of the heap. An immutable
   array of values is a value that [Make_immutable_array] makes of its
   arguments, or what a symbol may be bound to
   (FlambdaValueSpec.array_of): the table that Simplify makes of a
   [Switch] whose arms give one continuation constants. Either is
   immutable data and not a location, and [eval_prim] has a value on
   [Array_length] and on [Array_load] of one. The section
   "Arrays" says what the four primitives of arrays compute on the
   elements of an array, which is what FlambdaHeapSpec.v applies to a
   cell and what is applied here to the elements of an immutable
   array;
   it is to be reviewed against
   [Simplify_common.specialise_array_kind] and the functions for
   arrays of the four files of Simplify above, and against
   [Lambda_to_flambda_primitives] for the checks of bounds that the
   front end puts before an access.

   The laws. The rest of the model does not look inside a primitive: it
   uses the statements at the end, which hold of every primitive.
   [eval_prim] is opaque after FlambdaPrimitiveProofs.v, as [step] is, so
   that no proof elsewhere comes to depend on which primitives there are.
   A proof about one primitive rewrites with the equation of its arity
   there ([eval_prim_unary], [eval_prim_binary], [eval_prim_variadic]).
   The last law says that the value of a primitive on its arguments is
   its value on arguments that the observations read alike
   ([eval_prim_related_statement]): a block made by a primitive and a
   symbol bound to an equal block, for one.

   The proofs are in FlambdaPrimitiveProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaModel FlambdaMapSpec
  FlambdaValueSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* Integers of a width                                                      *)
(* ======================================================================== *)

(* The integers of [width] bits: the signed reading of a machine register
   of that width. *)
Definition in_range (width z : Z) : bool :=
  (- 2 ^ (width - 1) <=? z) && (z <? 2 ^ (width - 1)).

(* Two's-complement wrap-around: the integer of [width] bits that differs
   from [z] by a multiple of [2 ^ width]. *)
Definition wrap (width z : Z) : Z :=
  (z + 2 ^ (width - 1)) mod 2 ^ width - 2 ^ (width - 1).

(* The unsigned reading of an integer of [width] bits. *)
Definition unsigned (width z : Z) : Z :=
  if z <? 0 then z + 2 ^ width else z.

Definition reading (width : Z) (s : signedness) (z : Z) : Z :=
  match s with
  | Signed => z
  | Unsigned => unsigned width z
  end.

(* The number of bits of the integers of a kind, on a 64-bit target. *)
Definition int_width (kind : standard_int) : Z :=
  match kind with
  | Standard_tagged_immediate | Standard_naked_immediate => 63
  | Standard_naked_int8 => 8
  | Standard_naked_int16 => 16
  | Standard_naked_int32 => 32
  | Standard_naked_int64 | Standard_naked_nativeint => 64
  end.

(* The constant that is an integer of a kind. *)
Definition const_of_int (kind : standard_int) (i : Z) : const :=
  match kind with
  | Standard_tagged_immediate => Tagged_immediate i
  | Standard_naked_immediate => Naked_immediate i
  | Standard_naked_int8 => Naked_int8 i
  | Standard_naked_int16 => Naked_int16 i
  | Standard_naked_int32 => Naked_int32 i
  | Standard_naked_int64 => Naked_int64 i
  | Standard_naked_nativeint => Naked_nativeint i
  end.

(* The integer that a value is, if it is a constant of the kind and in
   its range. A constant that is out of the range of its kind is not an
   integer of that kind, so the arithmetic, the comparisons, tagging and
   untagging have no value on it. The compiler cannot build such a
   constant, a unit that passes FlambdaWellFormedSpec.numbers_in_range
   has none, and no primitive makes one (the section "Numbers in
   range"): so in the run of such a unit the test of the range here
   never fails ([int_of_value_in_range_statement]). *)
Definition int_of_value (kind : standard_int) (v : value) : option Z :=
  match kind, v with
  | Standard_tagged_immediate, Value_const (Tagged_immediate i)
  | Standard_naked_immediate, Value_const (Naked_immediate i)
  | Standard_naked_int8, Value_const (Naked_int8 i)
  | Standard_naked_int16, Value_const (Naked_int16 i)
  | Standard_naked_int32, Value_const (Naked_int32 i)
  | Standard_naked_int64, Value_const (Naked_int64 i)
  | Standard_naked_nativeint, Value_const (Naked_nativeint i) =>
    if in_range (int_width kind) i then Some i else None
  | _, _ => None
  end.

Definition is_tagged_immediate (v : value) : bool :=
  is_some (int_of_value Standard_tagged_immediate v).

(* ======================================================================== *)
(* Integer arithmetic and comparison                                        *)
(* ======================================================================== *)

(* [Int_arith]. Addition, subtraction and multiplication wrap. Division
   truncates towards zero and [Mod] is its remainder, with the sign of
   the dividend; the unsigned ones are those of the unsigned readings.
   Neither has a value on a zero divisor: the compiler tests for zero
   before a division, and Simplify takes one by zero to be unreachable
   ([Simplify_binary_primitive], [op_lhs_unknown]). The least integer
   divided by minus one wraps to itself. The bitwise operations are those
   of two's complement. *)
(* CR mshinwell: [Div] and [Mod] have no value on a zero divisor. It is
   what Simplify assumes, which makes such a division [Invalid]. It is
   right for a division of the source, which follows a test of the
   divisor. Nothing checks it for the unsafe divisions that the front
   end makes without a test (those of comprehensions), where the
   machine's division traps on x86-64. It would be wrong if a division
   by zero that is reached has to mean something, or if a pass may move
   a division to where its divisor has not been tested. Decision D14. *)
Definition eval_int_arith (width : Z) (op : binary_int_arith_op) (a b : Z)
    : option Z :=
  match op with
  | Add => Some (wrap width (a + b))
  | Sub => Some (wrap width (a - b))
  | Mul => Some (wrap width (a * b))
  | Div s =>
    if b =? 0 then None
    else Some (wrap width (Z.quot (reading width s a) (reading width s b)))
  | Mod s =>
    if b =? 0 then None
    else Some (wrap width (Z.rem (reading width s a) (reading width s b)))
  | And => Some (wrap width (Z.land a b))
  | Or => Some (wrap width (Z.lor a b))
  | Xor => Some (wrap width (Z.lxor a b))
  end.

(* The integer whose bytes are those of a list, the least significant
   first. Of each element of the list only the low eight bits are
   taken, which changes nothing of a byte. It is at least 0. *)
Fixpoint little_endian (bytes : list Z) : Z :=
  match bytes with
  | [] => 0
  | byte :: rest => byte mod 256 + 256 * little_endian rest
  end.

(* The low [count] bytes of an integer in two's complement, the least
   significant first. *)
Fixpoint bytes_of (count : nat) (z : Z) : list Z :=
  match count with
  | O => []
  | S count => z mod 256 :: bytes_of count (z / 256)
  end.

(* The integer, at least 0, whose [count] bytes are the low [count]
   bytes of [z] in the opposite order. *)
Definition swap_bytes (count : nat) (z : Z) : Z :=
  little_endian (rev (bytes_of count z)).

(* The unary [Int_arith].
   [Not] is the complement of every bit, which for an integer in two's
   complement is its opposite less one. The result is in range as it
   is; it is wrapped so that this is evident (FlambdaPrimitiveProofs.v,
   [eval_unary_int_arith_not]).
   [Swap_byte_endianness] puts the bytes of an integer in the opposite
   order, and is not the same thing at every kind:
   - at the two kinds of immediates it is the swap of the two low bytes
     and nothing of the others: the result is from 0 to 65535, whatever
     the bits of the argument above the sixteenth. It is what [%bswap16]
     of the source becomes, which is on an [int];
   - at the naked int8 it is the integer itself (wrapped here, as for
     [Not]);
   - at the naked int16, int32, int64 and nativeint it is the integer
     of the kind whose bytes are those of the argument in the opposite
     order, so that the last byte of the argument gives the sign.
   This is [To_cmm_primitive.unary_int_arith_primitive], and what
   Simplify computes of a constant: [swap_byte_endianness] of each
   [Int_number_kind] of [Number_adjuncts], which for the immediates is
   [Target_ocaml_int.get_least_significant_16_bits_then_byte_swap]. *)
Definition eval_unary_int_arith (kind : standard_int)
    (op : unary_int_arith_op) (a : Z) : Z :=
  match op with
  | Not => wrap (int_width kind) (Z.lnot a)
  | Swap_byte_endianness =>
    match kind with
    | Standard_tagged_immediate | Standard_naked_immediate =>
      swap_bytes 2 a
    | Standard_naked_int8 => wrap 8 a
    | Standard_naked_int16 => wrap 16 (swap_bytes 2 a)
    | Standard_naked_int32 => wrap 32 (swap_bytes 4 a)
    | Standard_naked_int64 | Standard_naked_nativeint =>
      wrap 64 (swap_bytes 8 a)
    end
  end.

(* The amounts by which an integer of a kind is shifted: those from 0 up
   to, and not including, this bound. It is the width of the kind, but
   64 for the tagged immediates, which have 63 bits: OCaml's [lsl],
   [lsr] and [asr] are specified at the amount 63. These are the
   amounts at which Simplify computes the shift ([Number_adjuncts], at
   [integer_bit_width] and [with_shift]). *)
Definition shift_bound (kind : standard_int) : Z :=
  match kind with
  | Standard_tagged_immediate => 64
  | _ => int_width kind
  end.

(* [Int_shift], by an amount that is not negative. [Lsl] shifts left
   and wraps. [Lsr] shifts the unsigned reading right, and [Asr] the
   integer itself, which keeps its sign. The result of a right shift is
   wrapped too, so that every result is in range by its form; that
   changes nothing but an [Lsr] by 0, which is the integer shifted and
   not its unsigned reading (FlambdaPrimitiveProofs.v,
   [eval_int_shift_asr] and [eval_int_shift_lsr]). *)
(* CR mshinwell: [Int_shift] has no value on an amount that is negative,
   or at or above [shift_bound]. Simplify says that such a shift is
   undefined, and that it cannot make it [Invalid]
   ([Simplify_binary_primitive], at [op_lhs_unknown] of the shifts); it
   folds one to zero ([Number_adjuncts.with_shift]), where the machine
   shifts by the amount modulo the size of the register on x86-64, so
   the two differ: on a shift by 64, which the machine does not do at
   all, and on an arithmetic shift right of a negative number.
   For the naked immediates the bound is 63, where it is 64 for the
   tagged ones, which are the same integers: Simplify folds a shift
   right by 63 of a negative naked immediate to 0 and the machine gives
   -1 (C1-9). The library documents that amount as unspecified for the
   untagged [int] and as specified for [int]. No value is what the
   documentation of the shifts says of these amounts; it would be
   wrong if a program is entitled to some answer there. Decision
   D18. *)
Definition eval_int_shift (width : Z) (op : int_shift_op) (a amount : Z)
    : Z :=
  match op with
  | Lsl => wrap width (Z.shiftl a amount)
  | Lsr => wrap width (Z.shiftr (unsigned width a) amount)
  | Asr => wrap width (Z.shiftr a amount)
  end.

Definition eval_int_comparison (width : Z) (c : int_comparison) (a b : Z)
    : bool :=
  match c with
  | Int_eq => a =? b
  | Int_neq => negb (a =? b)
  | Int_lt s => reading width s a <? reading width s b
  | Int_gt s => reading width s b <? reading width s a
  | Int_le s => reading width s a <=? reading width s b
  | Int_ge s => reading width s b <=? reading width s a
  end.

(* [Int_comp]: 1 or 0 for a comparison yielding a boolean; -1, 0 or 1 for
   one that behaves as the [compare] functions do. *)
Definition eval_int_comp (width : Z) (behaviour : int_comparison_behaviour)
    (a b : Z) : Z :=
  match behaviour with
  | Yielding_bool c => if eval_int_comparison width c a b then 1 else 0
  | Yielding_int_like_compare_functions s =>
    match reading width s a ?= reading width s b with
    | Lt => -1
    | Eq => 0
    | Gt => 1
    end
  end.

(* ======================================================================== *)
(* Arrays                                                                   *)
(* ======================================================================== *)

(* What the primitives of arrays compute on the elements of an array,
   given as a list, and on the kind that the array was made at. The
   array itself is a cell of the heap (FlambdaHeapSpec.v), or, for an
   immutable array, a value or what a symbol is bound to
   (FlambdaValueSpec.array_of), which [eval_prim] reads with the same
   functions; an immutable array has no kind, and only its length and
   its elements are read.

   An array keeps the kind it was made at, and each of its elements is
   a value that this kind holds ([array_element_ok]): a tagged
   immediate for [Array_immediates], anything of kind [Value] for the
   two others. The kind is kept because Simplify keeps it: the type
   that it gives a mutable array has the kind of the elements, and
   stays the type of the array across stores
   ([Flambda2_types.mutable_array], in
   [Simplify_variadic_primitive.simplify_make_array]); and
   [Simplify_common.specialise_array_kind] makes a load or a store at
   [Values] or at [Gc_ignorable_values] one at [Immediates] where that
   type says that the elements are immediates. So a store of a value
   that the kind of the array does not hold has no value here: it
   would make that type wrong.

   The array kind that an access carries ([Array_length],
   [Array_load], [Array_set]) is not looked at: every array kind of
   the model is one of values, and among those Simplify changes the
   kind of an access as just said. It is to matter when there are
   arrays of naked numbers, where Simplify gives [Invalid] for a load
   or a store at a kind of naked numbers of an array whose type has
   another kind of element.

   A load says what it takes the element to be ([array_load_kind]) and
   has no value on another: at [Load_immediates], on an element that
   is not a tagged immediate. A store says the same of the new value
   ([array_set_kind]). Whether a store is an initialisation or an
   assignment changes nothing. *)

(* The kind of array whose elements a load kind, and a set kind, take
   the element to be one of. *)
Definition array_kind_of_load_kind (load_kind : array_load_kind)
    : array_kind :=
  match load_kind with
  | Load_immediates => Array_immediates
  | Load_gc_ignorable_values => Array_gc_ignorable_values
  | Load_values => Array_values
  end.

Definition array_kind_of_set_kind (set_kind : array_set_kind) : array_kind :=
  match set_kind with
  | Set_immediates => Array_immediates
  | Set_gc_ignorable_values => Array_gc_ignorable_values
  | Set_values _ => Array_values
  end.

(* Whether an array of a kind holds a value:
   [Flambda_primitive.Array_kind.element_kinds], with the subkind of
   the immediates. *)
(* CR mshinwell: An array made at [Gc_ignorable_values] holds any value
   of kind [Value], as [Array_kind.element_kinds] says, and a load or a
   store at that kind takes any. The model has no notion of what the
   collector may ignore, so it gives a meaning to a store at this kind
   of a value that the collector must not ignore, which the machine
   does without telling the collector. The front end uses the kind for
   the stores of integers into an array of them ([Pintarray_set]) and
   for the arrays of values that are not in the heap. If such a store
   is to be undefined, the test here is to refuse more. Decided while
   the owner of the branch was away
   (middle_end/flambda2/rocq/status/P1.md). *)
Definition array_element_ok (kind : array_kind) (v : value) : bool :=
  match kind with
  | Array_immediates => is_tagged_immediate v
  | Array_gc_ignorable_values | Array_values =>
    kind_eqb (kind_of_value v) Value
  end.

(* [Make_array]: the elements of the array made are the arguments, each
   of which is a value that the kind holds. *)
(* CR mshinwell: A [Make_array] says nothing of its elements but what
   its array kind says. In the compiler the kind of the elements has a
   subkind, and Simplify meets the type of each argument with it and
   gives [Invalid] where they cannot agree
   ([Simplify_variadic_primitive.simplify_make_array]); the only
   subkind that the array kinds of the model give is that of the
   immediates, which is tested. Decision D25 is the same question for
   [Make_block]. *)
Definition array_made (kind : array_kind) (vs : list value)
    : option (list value) :=
  if forallb (array_element_ok kind) vs then Some vs else None.

(* [Array_length]: the number of elements, a tagged immediate. It does
   not depend on the kind of the array nor on the kind that the
   primitive carries: [Simplify_unary_primitive.simplify_array_length]
   looks at neither. With [Float_array_opt_dynamic] the array may be
   one of floats or one of values; the compiled code reads the size in
   words either way, a float being a word on a 64-bit target, and the
   model has no array of floats. *)
Definition array_length_of (elements : list value) : option value :=
  let length := Z.of_nat (length elements) in
  if in_range (int_width Standard_tagged_immediate) length
  then Some (Value_const (Tagged_immediate length))
  else None.

(* The position in an array that an index is: the index is a tagged
   immediate, at least 0 and below the number of elements. *)
(* CR mshinwell: A load or a store at an index that the array has not
   got has no value. [Array_load] and [Array_set] are the accesses
   that test nothing: the front end puts a test of the bounds before
   one, which raises, unless the source asks for none
   ([Array.unsafe_get], [-unsafe]), and the machine reads or writes
   the word at the index. Simplify replaces a load at a known index
   out of the bounds of an immutable array of known length by
   [Invalid] ([Simplify_binary_primitive.simplify_array_load]); it
   says nothing of a mutable array. It would be wrong if a program
   that has a meaning reads out of the bounds of an array and does
   not use what it reads. Decided while the owner of the branch was
   away (middle_end/flambda2/rocq/status/P1.md). *)
Definition array_index (elements : list value) (index : value)
    : option nat :=
  let* i := int_of_value Standard_tagged_immediate index in
  if (0 <=? i) && (i <? Z.of_nat (length elements))
  then Some (Z.to_nat i)
  else None.

(* [Array_load]: the element at the index, if it is one that the load
   kind takes. *)
Definition array_loaded (load_kind : array_load_kind)
    (elements : list value) (index : value) : option value :=
  let* i := array_index elements index in
  let* v := nth_error elements i in
  if array_element_ok (array_kind_of_load_kind load_kind) v
  then Some v
  else None.

(* A list with the element at a position replaced. There is none if the
   list has no such position. *)
Fixpoint replace_nth {A : Type} (n : nat) (x : A) (l : list A)
    : option (list A) :=
  match n, l with
  | O, _ :: l => Some (x :: l)
  | S n, y :: l =>
    let* l' := replace_nth n x l in
    Some (y :: l')
  | _, [] => None
  end.

(* [Array_set]: the elements with the one at the index replaced by the
   new value, which must be a value that the set kind takes and one
   that the kind of the array holds. *)
Definition array_stored (set_kind : array_set_kind) (kind : array_kind)
    (elements : list value) (index new : value) : option (list value) :=
  if array_element_ok (array_kind_of_set_kind set_kind) new
     && array_element_ok kind new
  then
    let* i := array_index elements index in
    replace_nth i new elements
  else None.

(* ======================================================================== *)
(* The primitives                                                           *)
(* ======================================================================== *)

(* A value that is a symbol is bound. A primitive that reads its
   argument has no value on a symbol that is bound to nothing, because
   no observation has an answer there. This is the test for one that
   passes its argument on without reading it. *)
Definition bound_if_symbol (statics : static_env) (v : value) : bool :=
  match v with
  | Value_symbol s => is_some (lookup s (static_values statics))
  | _ => true
  end.

(* A boolean result is a naked immediate, 1 or 0. *)
Definition naked_bool (b : bool) : value :=
  Value_const (Naked_immediate (if b then 1 else 0)).

(* [Is_int]. With [variant_only], the argument is taken to be a variant:
   a tagged immediate or a block, and the primitive has no value on
   anything else. That is what Simplify assumes: it takes the argument
   to be a variant that is not null
   ([Flambda2_types.add_is_int_relation]), and
   [Provers.prove_is_int_generic_value] answers [Invalid] on closures,
   boxed numbers, strings and arrays. The machine computes the low bit
   of anything. Without [variant_only], the primitive also answers no
   on every other allocated value. It has no value on null, on poison,
   or on a symbol bound to nothing; on a poison constant Simplify folds
   it to 1. *)
(* CR mshinwell: With [variant_only], [Is_int] has no value on a closure,
   a boxed number, a string or an immutable array. The machine computes
   the low bit of the word and answers 0; Simplify answers [Invalid].
   That the front end applies the variant-only form to variants only is
   nowhere checked or written down. It would be wrong if code that the
   machine runs applies it to something else that is not ill-kinded:
   then Simplify removes code that runs, and the model, by making it
   undefined, does not say so. Decisions D5 and D22. *)
(* CR mshinwell: [Is_int] has no value on null, with or without
   [variant_only], where the machine answers 0. With [variant_only]
   Simplify makes it [Invalid]; without, Simplify claims nothing, and
   the model could as well answer 0. It has no value on poison either,
   where Simplify folds it to 1 and the backend makes poison a tagged
   immediate: that is right only if a poison is never the argument of a
   primitive that is executed. Decision D8. *)
Definition eval_is_int (statics : static_env) (variant_only : bool)
    (v : value) : option bool :=
  if is_tagged_immediate v then Some true
  else if
    if variant_only then is_some (block_of statics v)
    else is_allocated statics v
  then Some false
  else None.

(* [Is_null]. It has no value on poison or on a symbol bound to
   nothing; on a poison constant Simplify folds it to 0. *)
(* CR mshinwell: [Is_null] has no value on poison. Simplify folds it to 0
   on a poison constant ([Reg_width_const.is_null]) and the backend
   makes a poison of kind [Value] a tagged immediate, so the compiler
   takes it to be 0. No value is sound for that folding and is right if
   a poison is never the argument of a primitive that is executed; it
   would be wrong if one may be, when the model should rather say that
   poison is some tagged immediate. Decision D8 (which rules on the
   symbol bound to nothing, not on poison). *)
Definition eval_is_null (statics : static_env) (v : value) : option bool :=
  match v with
  | Value_const Null => Some true
  | _ =>
    if is_tagged_immediate v || is_allocated statics v then Some false
    else None
  end.

(* Whether a constant is a naked number of a boxable kind. *)
Definition is_number_of_kind (kind : boxable_number) (c : const) : bool :=
  match kind, c with
  | Boxable_naked_float32, Naked_float32 _
  | Boxable_naked_float, Naked_float _
  | Boxable_naked_int32, Naked_int32 _
  | Boxable_naked_int64, Naked_int64 _
  | Boxable_naked_nativeint, Naked_nativeint _
  | Boxable_naked_vec128, Naked_vec128 _
  | Boxable_naked_vec256, Naked_vec256 _
  | Boxable_naked_vec512, Naked_vec512 _
  | Boxable_naked_mask, Naked_mask _ =>
    true
  | _, _ => false
  end.

(* What is known of a tag or a size is right. *)
Definition known_is (known : option Z) (actual : Z) : bool :=
  match known with
  | Some expected => expected =? actual
  | None => true
  end.

(* [Block_load]: the field of an immutable block of values, the first
   being field 0. It has no value on what is not a block; on a block
   whose tag or whose number of fields is not the one the access kind
   gives, where it gives one; on a field that the block has not got; on
   a field that is not of kind [Value]; or, where the access kind says
   that the field is an immediate, on one that is not a tagged
   immediate. Simplify
   ([Simplify_unary_primitive.simplify_immutable_block_load0]) takes the
   block to have that tag and that field only where it does not already
   know the field, when it reads it whatever the tag; it does not use
   the size.

   The mutability of the load is not looked at. It says what the
   compiler may take of the load: one marked [Immutable] or
   [Immutable_unique] may be shared with another of the same field and
   replaced by what is known of the field, and one marked [Mutable] may
   not ([Simplify_unary_primitive.simplify_mutable_block_load] keeps
   one whose argument is not a constant, and makes one of a constant
   [Invalid], where the load has no value here either). It is not a
   test of the block: the machine reads the field
   ([To_cmm_primitive.block_load]), and a block of the model is
   immutable, so that a load marked [Mutable] from one, which is what
   [%field0] of the source makes when it is applied to a tuple, reads
   what the block was made with. ([fst] and [snd] of the standard
   library are [%field0_immut] and [%field1_immut], and make loads
   marked [Immutable].) When the model has blocks that can be written,
   the effects of a load will go by its mark and not by the block, as
   they do in the compiler ([Flambda_primitive.effects_and_coeffects]:
   a load marked [Mutable] has a coeffect); today no primitive of the
   model has an effect. *)
(* CR mshinwell: [Block_load] has no value on a block whose tag or whose
   number of fields is not the one the access kind gives, nor, with
   [Immediate], on a field that is not a tagged immediate. The machine
   reads the field whatever the three are. Simplify uses the tag on one
   of its two paths only, never the size, and reads a field that is not
   an immediate under [Immediate]; nothing in the compiler says what an
   access kind promises of the block, or checks it. It would be wrong
   if the compiler makes a load whose access kind disagrees with the
   block that reaches it: the model then calls undefined a load that
   the compiler means. Decisions D15 and D22. *)
Definition eval_block_load (statics : static_env) (kind : block_access_kind)
    (field : Z) (v : value) : option value :=
  match kind with
  | Block_access_values tag size field_kind =>
    let* '(block_tag, fields) := block_of statics v in
    if known_is tag block_tag
       && known_is size (Z.of_nat (length fields))
       && (0 <=? field)
    then
      let* loaded := nth_error fields (Z.to_nat field) in
      if
        match field_kind with
        | Any_value => kind_eqb (kind_of_value loaded) Value
        | Immediate => is_tagged_immediate loaded
        end
      then Some loaded
      else None
    else None
  end.

(* [Reinterpret_64_bit_word], between a tagged immediate and an int64.
   The machine word of the tagged immediate [i] is [2 * i + 1].
   [Tagged_int63_as_unboxed_int64] is that word as an int64.
   [Unboxed_int64_as_tagged_int63] is the tagged immediate whose word
   is the int64 with its lowest bit set: the int64 halved, towards
   minus infinity. The backend leaves the word as it is for the first
   and sets the lowest bit for the second
   ([To_cmm_primitive.unary_primitive]); Simplify computes [2 * i + 1],
   and a logical shift to the right by one bit, read as an integer of
   63 bits, which is the same half ([Simplify_unary_primitive], the
   two modules [Simplify_reinterpret_] between these integers).
   The first has no value on a value that is not a tagged immediate,
   where the machine would give the address: Simplify makes it
   [Invalid] there, as it asks the type of the argument to be that of
   a tagged immediate. *)
Definition eval_reinterpret_64_bit_word
    (reinterpret : reinterpret_64_bit_word) (v : value) : option value :=
  match reinterpret with
  | Tagged_int63_as_unboxed_int64 =>
    let* i := int_of_value Standard_tagged_immediate v in
    Some (Value_const (Naked_int64 (2 * i + 1)))
  | Unboxed_int64_as_tagged_int63 =>
    let* word := int_of_value Standard_naked_int64 v in
    Some (Value_const (Tagged_immediate (word / 2)))
  end.

(* A unary primitive applied to a value.
   - [Get_tag] is the tag of a block, and has no value on anything else,
     a closure, a boxed number, a string and an immutable array
     included: Simplify takes its argument to be a variant that is not
     null ([Flambda2_types.add_get_tag_relation]).
   - [Array_length] is the number of elements of an immutable array,
     one that a primitive made or one that a symbol is bound to
     ([array_length_of], above), and has no value on anything else. An
     array of the heap is a location, which FlambdaHeapSpec.v reads.
   - [String_length] is the number of bytes of an immutable string, and
     has no value on anything else: Simplify makes it [Invalid] where
     its argument cannot be a string
     ([Simplify_unary_primitive.simplify_string_length]). A string so
     long that its length is not a naked immediate has none; the
     compiler has no such string.
   - [Opaque_identity] is the identity, on a value of the kind it
     carries that is not a symbol bound to nothing.
   - [Box_number] makes a boxed number of a naked number of the kind it
     carries, and [Unbox_number] is the naked number inside a boxed
     number of that kind. Neither looks at the number: it is not
     restricted to a range. Neither has a value on poison: a poison
     constant is not boxed, and a boxed poison, which is a boxed number
     (FlambdaValueSpec.v), is not unboxed, as [is_number_of_kind] goes
     by the constructor of the constant.
   - [Unary_int_arith] is on an integer of the kind it carries, and
     gives one.
   - [Num_conv] changes an integer of the kind [src] into the integer
     of the kind [dst] that is the same modulo two to the width of
     [dst]: the same integer where [dst] holds it, as it does where it
     is at least as wide as [src], and the integer wrapped where it
     does not. This is what Simplify folds it to
     ([Simplify_unary_primitive.Make_simplify_int_conv], with the
     [to_immediate], [to_naked_int8], ... of [Number_adjuncts], each of
     which keeps the low bits of the number and extends their sign),
     and what the backend computes
     ([Cmm_helpers.Scalar_type.Integral.static_cast]: nothing where
     the destination holds every number of the source, a sign
     extension of the low bits where it does not). Between the tagged
     and the naked immediates it is [Untag_immediate] and
     [Tag_immediate].
   - [Boolean_not] is the negation of a boolean of the source, which is
     the tagged immediate 0 or 1: 1 for 0 and 0 for 1. It has no value
     on another tagged immediate, nor on anything else.
   - [Untag_immediate] and [Tag_immediate] change a tagged immediate into
     the naked immediate that is the same integer, and back.
   - [Reinterpret_64_bit_word] reads the 64 bits of one integer as
     another ([eval_reinterpret_64_bit_word], above). *)
(* CR mshinwell: [Get_tag] has no value on what is not a block: a tagged
   immediate, null, a closure, a boxed number, a string, an immutable
   array. On anything allocated the machine reads the tag in the
   header. Simplify makes it [Invalid] on null and on a boxed number,
   and leaves it alone on a tagged immediate; the unboxing of variants
   records the tag of a scrutinee that may be an immediate. It would be
   wrong if the compiler makes a [Get_tag] that is executed on a
   closure, a boxed number, a string or an array and uses the result.
   Decisions D5 and D22. *)
(* CR mshinwell: [Box_number] has no value on a poison constant, and
   [Unbox_number] none on a boxed number whose contents are poison,
   although such a box is a boxed number for [Is_null] and [Is_int]
   (FlambdaValueSpec.v). Nothing in the compiler says what boxing a
   poison means when it is executed; the backend makes a poison some
   number of its kind, which it would box and unbox like another. It
   would be wrong if a poison may be boxed where the code is executed
   and the box is then only passed on, not read. Decision D21. *)
(* CR mshinwell: [Boolean_not] has no value on a tagged immediate other
   than 0 and 1. Simplify makes it [Invalid] where its argument cannot
   be 0 or 1 ([Simplify_unary_primitive.simplify_boolean_not]), so the
   model cannot give it a value there. The machine flips one bit of
   the word ([Cmm_helpers.mk_not]), which gives the integer with its
   lowest bit flipped. The front end makes the primitive from [not] of
   the source, whose argument has the type [bool]; nothing checks that
   it is given a boolean. It would be wrong if code that is executed
   applies it to another integer and uses the result: then Simplify
   removes code that runs, and the model, by making it undefined, does
   not say so. Decision D38. *)
Definition eval_unary_primitive (statics : static_env) (p : unary_primitive)
    (v : value) : option value :=
  match p with
  | Block_load kind _ field => eval_block_load statics kind field v
  | Is_int variant_only =>
    option_map naked_bool (eval_is_int statics variant_only v)
  | Is_null => option_map naked_bool (eval_is_null statics v)
  | Get_tag =>
    let* '(tag, _) := block_of statics v in
    Some (Value_const (Naked_immediate tag))
  (* The length of an immutable array, a value or what a symbol is
     bound to. An array that can be written is a cell of the heap, and
     has no value here (the section "Arrays"). *)
  | Array_length _ =>
    let* elements := array_of statics v in
    array_length_of elements
  | String_length =>
    let* bytes := string_of statics v in
    let length := Z.of_nat (length bytes) in
    if in_range (int_width Standard_naked_immediate) length
    then Some (Value_const (Naked_immediate length))
    else None
  | Opaque_identity kind =>
    if kind_eqb (kind_of_value v) kind && bound_if_symbol statics v
    then Some v
    else None
  | Unary_int_arith kind op =>
    let* a := int_of_value kind v in
    Some
      (Value_const
         (const_of_int kind (eval_unary_int_arith kind op a)))
  | Num_conv src dst =>
    let* i := int_of_value src v in
    Some (Value_const (const_of_int dst (wrap (int_width dst) i)))
  | Boolean_not =>
    let* i := int_of_value Standard_tagged_immediate v in
    if i =? 0 then Some (Value_const (Tagged_immediate 1))
    else if i =? 1 then Some (Value_const (Tagged_immediate 0))
    else None
  | Unbox_number kind =>
    let* c := boxed_number_of statics v in
    if is_number_of_kind kind c then Some (Value_const c) else None
  | Box_number kind =>
    match v with
    | Value_const c =>
      if is_number_of_kind kind c then Some (Value_boxed_number c) else None
    | _ => None
    end
  | Untag_immediate =>
    let* i := int_of_value Standard_tagged_immediate v in
    Some (Value_const (Naked_immediate i))
  | Tag_immediate =>
    let* i := int_of_value Standard_naked_immediate v in
    Some (Value_const (Tagged_immediate i))
  | Reinterpret_64_bit_word reinterpret =>
    eval_reinterpret_64_bit_word reinterpret v
  end.

(* [Flambda_primitive.byte_width_of_string_accessor_width]: how many
   bytes a load reads. *)
Definition byte_width_of_string_accessor_width
    (width : string_accessor_width) : nat :=
  match width with
  | Eight | Eight_signed => 1%nat
  | Sixteen | Sixteen_signed => 2%nat
  | Thirty_two => 4%nat
  | Sixty_four => 8%nat
  end.

(* [String_or_bigstring_load] of an integer from an immutable string.
   The index is a naked nativeint, and is that of a byte, the first
   being at index 0, whatever the width. The load reads as many bytes as
   the width has, from the one at the index on, the least significant
   first: the model is of a little-endian target, as the two 64-bit
   targets of the compiler are ([Arch.big_endian] is [false] for both).
   With [Eight] and [Sixteen] the result is a naked immediate, of what
   was read without a sign, from 0 to 255 and to 65535. With the others
   it is the integer of the width that has the bits read: a naked int8
   for [Eight_signed], a naked int16 for [Sixteen_signed], a naked int32
   for [Thirty_two] and a naked int64 for [Sixty_four]. This is
   [To_cmm_primitive.string_like_load], with the loads of
   [Cmm_helpers.unaligned_load_16] and the like, and the kinds are those
   of [Flambda_primitive.result_kind_of_binary_primitive'].
   It has no value on what is not a string, or unless the string has
   every byte that is read: the machine would read whatever is there,
   and the front end makes such a load after a test of the index
   ([Lambda_to_flambda_primitives.string_like_load]), or without one
   for [String.unsafe_get] and the like. Simplify does not compute it
   ([Simplify_binary_primitive.simplify_string_or_bigstring_load]). *)
(* CR mshinwell: the byte after the last of a string is one of the
   string's block in the compiler, and is 0, so that a load of two bytes
   from the last index has a value on the machine. It has none here:
   the front end's test refuses it, and an unsafe load that reads it is
   taken to be a mistake of the program. Should it be 0? *)
(* CR mshinwell: a string of the model is an immutable value, so this
   load gives the byte the string was made with, whenever it is done.
   The compiler erases the conversions between bytes and strings before
   Flambda 2 ([Lambda_to_lambda_transforms]: [Pbytes_to_string] and
   [Pbytes_of_string] applied to an argument become the argument), so
   that a string of the compiler's can be the block of bytes that are
   written later; and [To_cmm] moves a load from a string, which has no
   effect and no coeffect ([Flambda_primitive]), past such a write
   (X5-3). A string made from bytes by an erased conversion is outside
   the model, which has no bytes: the conversion to the model refuses
   every primitive on bytes, but cannot see an erased conversion. *)
Definition eval_string_load (statics : static_env)
    (width : string_accessor_width) (v1 v2 : value) : option value :=
  let* bytes := string_of statics v1 in
  let* index := int_of_value Standard_naked_nativeint v2 in
  let count := byte_width_of_string_accessor_width width in
  if (0 <=? index) && (index + Z.of_nat count <=? Z.of_nat (length bytes))
  then
    let read :=
      little_endian (firstn count (skipn (Z.to_nat index) bytes))
    in
    Some
      (Value_const
         match width with
         | Eight | Sixteen => Naked_immediate read
         | Eight_signed => Naked_int8 (wrap 8 read)
         | Sixteen_signed => Naked_int16 (wrap 16 read)
         | Thirty_two => Naked_int32 (wrap 32 read)
         | Sixty_four => Naked_int64 (wrap 64 read)
         end)
  else None.

(* [Phys_equal]: whether two values of kind [Value] are the same value.
   The compiler compares the two machine words
   ([To_cmm_primitive.phys_equal]), which for two tagged immediates is
   whether they are the same integer. It is what [=] and [<>] at type
   [int] of the source become ([Lambda_to_flambda_primitives], at
   [Pphys_equal]), as well as [==] and [!=]. Simplify folds it by
   [Provers.prove_physical_equality], which on two constants answers
   whether they are the same constant.

   It has a value on two tagged immediates: whether they are the same
   integer. And on a tagged immediate against an allocated value (a
   block, a closure, a boxed number, a string, an immutable array, or
   a symbol bound to one): they are not the same, as the word of the
   one is odd and that of the other a pointer, and as
   [prove_physical_equality] answers.

   An imported symbol ([FlambdaValueSpec.imported_of]) of the class of
   identities or of data is not a tagged immediate either: it is a
   pointer ([FlambdaValueSpec.is_imported_pointer]). Of one of the
   class [Import_unknown] nothing is known, and the comparison has no
   value on it against anything.

   It has a value where one of the two is an identity: an imported
   symbol of the class [Import_identity]
   ([FlambdaValueSpec.identity_of]). Two identities are the same when
   they are one symbol: [prove_physical_equality] answers so of any two
   symbols. An identity is not an allocated value (a block, a closure,
   a boxed number, a string, an immutable array, or a symbol bound to
   one), and is not an imported symbol of the class [Import_data]; of
   one of the class [Import_unknown] it is not known whether it is. This
   is what a handler does to tell which exception it has caught: it
   compares the exception, or its first field, with the symbol of a
   predefined exception (lambda/matching.ml,
   [combine_extension_constructor]), and what it caught may be a block:
   an exception with arguments.

   It has no value on anything else. So an imported symbol of the class
   [Import_data] is as an immutable allocated value is: the comparison
   has no value on it against an allocated value, against another
   imported symbol of that class, or against itself. And an identity
   against null, poison, a number that is not a value or a symbol that
   is not bound has no value, as a tagged immediate against them has
   none. *)
(* CR mshinwell: [Phys_equal] has no value on two allocated values, on
   null or on poison, where the machine compares any two words. This is
   a first version, and not what the primitive is to mean in the end.
   - On two allocated values, what it means is part of the design of
     the heap, which is to extend this definition. The allocated values
     of the model have no identity, and the compiler shares equal
     immutable values (it makes one of two equal [Make_block]s or
     [Box_number]s, and lifts constants), which turns an answer of 0
     into 1. Simplify answers 1 on the same variable or symbol twice,
     and 0 on two different symbols and on allocated values of
     different sorts.
   - On null, the machine has the word 0, equal to itself and to no
     tagged immediate or pointer, and Simplify folds two nulls to 1 and
     a null against another constant or a symbol to 0. No value here
     as for [Is_int] on null, until the model says what null is.
   - On poison, Simplify folds two poisons of the same kind and name to
     1 and a poison against any other constant or a symbol to 0
     ([Reg_width_const.equal]), and the backend makes a poison of kind
     [Value] a tagged immediate, which may be the other argument. No
     value is sound for both, and is right only if a poison is never
     the argument of a [Phys_equal] that is executed.
   No value is conservative: it would be wrong only in that nothing is
   then proved of a program that compares such values, as library code
   does with [==]. Decision D19. *)
(* CR mshinwell: [Phys_equal] answers 0 of an identity against an
   allocated value of the unit and against an import that is data, and
   of two identities with different symbols. The machine code does;
   Simplify proves it only of two symbols, or where the shapes of the
   two types differ. The reason it is right: an identity is made once
   (by the runtime for a predefined exception, by the unit that
   declares an exception otherwise) and is never copied or shared, so
   nothing that a unit allocates, and nothing that Simplify shares or
   lifts, can be it. That the compiler shares equal immutable values
   (decision D30) is about immutable structure, which an identity is
   not. It rests on two things that are not checked here. One is that
   the class is given to nothing but an exception: the conversion from
   the compiler's terms decides it, from the symbol alone, and is to
   give it to the predefined exceptions only until something can check
   more. The other is that two symbols of that class are two objects:
   [exception E = Not_found] in another unit makes no second symbol,
   only another name for the first. For an import that is data there
   is no such reason, since Simplify may know its contents from the
   other unit and share it with a block of this one, and the comparison
   has no value. Decision 2 of requests/C5-imported-symbols.md. *)
Definition eval_phys_equal (statics : static_env) (v1 v2 : value)
    : option bool :=
  match
    int_of_value Standard_tagged_immediate v1,
    int_of_value Standard_tagged_immediate v2
  with
  | Some a, Some b => Some (a =? b)
  | Some _, None =>
    if is_allocated statics v2 || is_imported_pointer statics v2
    then Some false
    else None
  | None, Some _ =>
    if is_allocated statics v1 || is_imported_pointer statics v1
    then Some false
    else None
  | None, None =>
    match identity_of statics v1, identity_of statics v2 with
    | Some s1, Some s2 => Some (Pos.eqb s1 s2)
    | Some _, None =>
      if is_allocated statics v2 || is_imported_pointer statics v2
      then Some false
      else None
    | None, Some _ =>
      if is_allocated statics v1 || is_imported_pointer statics v1
      then Some false
      else None
    | None, None => None
    end
  end.

(* A binary primitive applied to two values. For [Int_arith] and
   [Int_comp], both are integers of the kind the primitive carries. For
   [Int_shift], the first is one, and the second, the amount, a naked
   immediate. The result of a comparison, and of [Phys_equal], is a
   naked immediate: for [Phys_equal], 1 or 0 as the answer of [Phys_eq]
   is yes or no, and the other way round for [Phys_neq]. *)
Definition eval_binary_primitive (statics : static_env)
    (p : binary_primitive) (v1 v2 : value) : option value :=
  match p with
  (* A load from an immutable array, a value or what a symbol is bound
     to: the element at the index, where the index is one that the
     array has and the element one that the load kind takes
     ([array_loaded]). The array kind and the mark that the load
     carries are not looked at. An array that can be written is a cell
     of the heap, and has no value here (the section "Arrays"). *)
  (* CR mshinwell: An immutable array is read at any load kind that
     its elements fit, whatever kind it was made at: a table of tagged
     immediates has a value read at [Load_values], and so has an array
     value made at [Array_immediates]. The compiler's constant has no
     kind either ([Static_const.Immutable_value_array]); the symbol has
     the
     type of an array of the kind that Simplify made it for
     (simplify/simplify_switch_expr.ml,
     [rebuild_mergeable_argument]), and Simplify makes no load from it
     at another kind. Reading it at a wider kind is what a load at
     [Values] of an [int array] does, and what is done here for the
     arrays of the heap. It would be wrong if a load at a kind other
     than the one of the symbol's type had to be [Invalid]. Section 3
     of requests/P1-immutable-arrays.md. *)
  | Array_load _ load_kind _ =>
    let* elements := array_of statics v1 in
    array_loaded load_kind elements v2
  | String_or_bigstring_load width => eval_string_load statics width v1 v2
  | Phys_equal op =>
    let* same := eval_phys_equal statics v1 v2 in
    Some
      (naked_bool
         match op with
         | Phys_eq => same
         | Phys_neq => negb same
         end)
  | Int_arith kind op =>
    let* a := int_of_value kind v1 in
    let* b := int_of_value kind v2 in
    let* result := eval_int_arith (int_width kind) op a b in
    Some (Value_const (const_of_int kind result))
  | Int_shift kind op =>
    let* a := int_of_value kind v1 in
    let* amount := int_of_value Standard_naked_immediate v2 in
    if (0 <=? amount) && (amount <? shift_bound kind)
    then
      Some
        (Value_const
           (const_of_int kind (eval_int_shift (int_width kind) op a amount)))
    else None
  | Int_comp kind behaviour =>
    let* a := int_of_value kind v1 in
    let* b := int_of_value kind v2 in
    Some
      (Value_const
         (Naked_immediate (eval_int_comp (int_width kind) behaviour a b)))
  end.

(* A variadic primitive applied to values. [Make_block] makes an
   immutable block of its arguments, with the tag it carries. It has no
   value if an argument is not of kind [Value], or is a symbol that is
   bound to nothing. A block of no fields is
   made like another: the compiler's backend refuses to allocate one
   ([To_cmm_primitive.check_alloc_fields]) and relies on Simplify's
   having made it a symbol, which is an obligation of the compiler and
   not a reason for the term to be undefined.
   [Make_immutable_array] makes an immutable array of its arguments
   ([Value_immutable_array]): the compiler's [Make_array] at the
   mutability [Immutable]. It has no value if an argument is not a
   value that its array kind holds ([array_made]), or is a symbol that
   is bound to nothing. The array made does not keep the kind. An
   array of no elements is made like another, and is told from no
   other: no array value has an identity. *)
(* CR mshinwell: A [Make_block] says nothing of its fields but that each
   is of kind [Value]: the block kind of the model has not the subkind
   that the compiler gives each field, and the conversion refuses a
   block that has one. So nothing is undefined here by a field's
   subkind. In the compiler Simplify meets each argument that is a
   variable with the subkind and gives [Invalid] where they cannot
   agree, the subkinds come from the types of the source, and the
   backend ignores them. When they come into the model, whether a
   field that is not of its subkind makes the block undefined is to be
   settled with the kinds. Decision D25. *)
Definition eval_variadic_primitive (statics : static_env)
    (p : variadic_primitive) (vs : list value) : option value :=
  match p with
  | Make_block (Block_kind_values tag) =>
    if forallb (fun v => kind_eqb (kind_of_value v) Value) vs
       && forallb (bound_if_symbol statics) vs
    then Some (Value_block tag vs)
    else None
  | Make_immutable_array kind =>
    let* elements := array_made kind vs in
    if forallb (bound_if_symbol statics) vs
    then Some (Value_immutable_array elements)
    else None
  end.

(* The value of a primitive applied to the values of its arguments, if it
   has one. *)
Definition eval_prim (statics : static_env) (p : prim) (vs : list value)
    : option value :=
  match p, vs with
  | Unary p, [v] => eval_unary_primitive statics p v
  | Binary p, [v1; v2] => eval_binary_primitive statics p v1 v2
  | Variadic p, vs => eval_variadic_primitive statics p vs
  | _, _ => None
  end.

(* ======================================================================== *)
(* Arrays: the statements                                                   *)
(* ======================================================================== *)

(* What follows is what FlambdaHeapSpec.v needs of these functions. *)

(* An element that an array holds is of kind [Value]. *)
Definition array_element_kind_statement : Prop :=
  ∀ (kind : array_kind) (v : value),
    array_element_ok kind v = true -> kind_of_value v = Value.

(* The elements of an array made are its arguments, and each is one
   that the kind holds. *)
Definition array_made_of_arguments_statement : Prop :=
  ∀ (kind : array_kind) (vs elements : list value),
    array_made kind vs = Some elements ->
    elements = vs ∧ (∀ v : value, v ∈ vs -> array_element_ok kind v = true).

(* The length is that of the list: it is the same for two arrays of as
   many elements. *)
Definition array_length_of_same_length_statement : Prop :=
  ∀ elements elements' : list value,
    length elements' = length elements ->
    array_length_of elements' = array_length_of elements.

(* What a load answers is an element of the array, and one that its
   load kind takes. *)
Definition array_loaded_element_statement : Prop :=
  ∀ (load_kind : array_load_kind) (elements : list value) (index v : value),
    array_loaded load_kind elements index = Some v ->
    v ∈ elements
    ∧ array_element_ok (array_kind_of_load_kind load_kind) v = true.

(* A store keeps the number of elements; the new value is one that the
   kind of the array holds; and each element after the store is the new
   value or an element before it. So an array whose elements are values
   that its kind holds stays one. *)
Definition array_stored_elements_statement : Prop :=
  ∀ (set_kind : array_set_kind) (kind : array_kind) (elements : list value)
    (index new : value) (stored : list value),
    array_stored set_kind kind elements index new = Some stored ->
    length stored = length elements
    ∧ array_element_ok kind new = true
    ∧ (∀ v : value, v ∈ stored -> v = new ∨ v ∈ elements).

(* A load at the index of a store answers the value stored, if it is
   one that the load kind takes. *)
Definition array_load_after_store_statement : Prop :=
  ∀ (set_kind : array_set_kind) (kind : array_kind)
    (load_kind : array_load_kind) (elements : list value)
    (index new : value) (stored : list value),
    array_stored set_kind kind elements index new = Some stored ->
    array_loaded load_kind stored index
    = if array_element_ok (array_kind_of_load_kind load_kind) new
      then Some new
      else None.

(* A load at another index answers what it did before the store. *)
Definition array_load_after_other_store_statement : Prop :=
  ∀ (set_kind : array_set_kind) (kind : array_kind)
    (load_kind : array_load_kind) (elements : list value)
    (index new other : value) (stored : list value) (i j : Z),
    array_stored set_kind kind elements index new = Some stored ->
    int_of_value Standard_tagged_immediate index = Some i ->
    int_of_value Standard_tagged_immediate other = Some j ->
    i <> j ->
    array_loaded load_kind stored other
    = array_loaded load_kind elements other.

(* What the functions compute on some elements they compute on related
   ones, for a relation between values under which the second of two
   related values is of the kind of the first, and is every integer
   that the first is. A relation that [relates_constants]
   (the section "The laws of every primitive") is one, and so is the
   relation of FlambdaHeapMachineSpec.v between the values of two
   runs, which relates a poison to any value of its kind. As for
   [eval_prim_related_statement], the statements go one way. *)
Definition relates_numbers (R : value -> value -> Prop) : Prop :=
  ∀ v v' : value,
    R v v' ->
    kind_of_value v' = kind_of_value v
    ∧ (∀ (kind : standard_int) (i : Z),
        int_of_value kind v = Some i -> int_of_value kind v' = Some i).

Definition array_made_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (kind : array_kind)
    (vs vs' elements : list value),
    relates_numbers R ->
    Forall2 R vs vs' ->
    array_made kind vs = Some elements ->
    ∃ elements' : list value,
      array_made kind vs' = Some elements' ∧ Forall2 R elements elements'.

Definition array_loaded_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (load_kind : array_load_kind)
    (elements elements' : list value) (index index' v : value),
    relates_numbers R ->
    Forall2 R elements elements' ->
    R index index' ->
    array_loaded load_kind elements index = Some v ->
    ∃ v' : value,
      array_loaded load_kind elements' index' = Some v' ∧ R v v'.

Definition array_stored_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (set_kind : array_set_kind)
    (kind : array_kind) (elements elements' : list value)
    (index index' new new' : value) (stored : list value),
    relates_numbers R ->
    Forall2 R elements elements' ->
    R index index' ->
    R new new' ->
    array_stored set_kind kind elements index new = Some stored ->
    ∃ stored' : list value,
      array_stored set_kind kind elements' index' new' = Some stored'
      ∧ Forall2 R stored stored'.

(* ======================================================================== *)
(* Numbers in range                                                         *)
(* ======================================================================== *)

(* The numbers of the compiler's terms are of types that hold a range
   only; here they are integers. What follows says which integers the
   compiler's are, for a 64-bit target. A term, a value or a primitive
   is in range when every number in it is. *)

(* The tag of a block of values: the compiler's [Tag.Scannable.t], at
   least 0 and below [Obj.no_scan_tag] ([Tag.Scannable.create]). *)
Definition tag_in_range (tag : Z) : bool := (0 <=? tag) && (tag <? 251).

(* A byte of a string. *)
Definition byte_in_range (byte : Z) : bool := (0 <=? byte) && (byte <? 256).

(* An integer of the target's OCaml, of 63 bits: the compiler's
   [Target_ocaml_int.t], which the fields and the sizes of blocks and
   the discriminants of a [Switch] are, as well as the immediates. *)
Definition ocaml_int_in_range (i : Z) : bool :=
  in_range (int_width Standard_tagged_immediate) i.

(* A constant is in the range of its kind. An integer is one of the
   width of its kind ([int_width]). The number of a float or of a vector
   is its bit pattern, read as an integer in two's complement as the
   integers are: of 32 bits for a float32, 64 for a float and for a
   mask, and 128, 256 and 512 for the vectors. Null and poison have no
   number.

   In the compiler a constant is in range by its type
   ([Int_ids.Const_data.t]): an immediate is a [Target_ocaml_int.t],
   which every operation extends from the sign bit of its width
   ([One_bit_fewer]); an int8 and an int16 are a [Numeric_types.Int8.t]
   and an [Int16.t], of a private type whose values are made by
   functions that wrap or fail; an int32 and an int64 are OCaml's, and
   a nativeint a [Targetint_32_64.t]; a float and a vector are their
   bit patterns, of one or several machine words. *)
Definition const_in_range (c : const) : bool :=
  match c with
  | Naked_immediate i | Tagged_immediate i => ocaml_int_in_range i
  | Naked_int8 i => in_range 8 i
  | Naked_int16 i => in_range 16 i
  | Naked_float32 i | Naked_int32 i => in_range 32 i
  | Naked_float i | Naked_int64 i | Naked_nativeint i | Naked_mask i =>
    in_range 64 i
  | Naked_vec128 bits => in_range 128 bits
  | Naked_vec256 bits => in_range 256 bits
  | Naked_vec512 bits => in_range 512 bits
  | Null | Poison _ => true
  end.

(* What is known of a tag or a size is in range. *)
Definition known_in_range (test : Z -> bool) (known : option Z) : bool :=
  match known with
  | Some z => test z
  | None => true
  end.

(* The numbers that a primitive carries are in range: the tag of the
   block that a [Make_block] makes, and the tag, the size and the field
   of a [Block_load]. The tags are the compiler's [Tag.Scannable.t], and
   the size and the field its [Target_ocaml_int.t]. A primitive that is
   added has its case here, which is [true] if it carries no number. *)
Definition unary_primitive_in_range (p : unary_primitive) : bool :=
  match p with
  | Block_load (Block_access_values tag size _) _ field =>
    known_in_range tag_in_range tag
    && known_in_range ocaml_int_in_range size
    && ocaml_int_in_range field
  | Is_int _ | Is_null | Get_tag | Array_length _ | String_length
  | Opaque_identity _ | Unary_int_arith _ _ | Num_conv _ _ | Boolean_not
  | Unbox_number _ | Box_number _ | Untag_immediate | Tag_immediate
  | Reinterpret_64_bit_word _ =>
    true
  end.

Definition binary_primitive_in_range (p : binary_primitive) : bool :=
  match p with
  | Array_load _ _ _ | String_or_bigstring_load _ | Phys_equal _
  | Int_arith _ _ | Int_shift _ _ | Int_comp _ _ =>
    true
  end.

Definition variadic_primitive_in_range (p : variadic_primitive) : bool :=
  match p with
  | Make_block (Block_kind_values tag) => tag_in_range tag
  | Make_immutable_array _ => true
  end.

(* C4's prototype: the primitives of the heap carry the numbers of the
   [Make_block] and the access that they stand for. *)
Definition access_in_range (kind : block_access_kind) (field : Z) : bool :=
  match kind with
  | Block_access_values tag size _ =>
    known_in_range tag_in_range tag
    && known_in_range ocaml_int_in_range size
    && ocaml_int_in_range field
  end.

Definition heap_primitive_in_range (p : heap_primitive) : bool :=
  match p with
  | Make_block_with_identity _ tag => tag_in_range tag
  | Block_set kind _ field => access_in_range kind field
  | Make_mutable_array _ | Array_set _ _ => true
  end.

Definition prim_in_range (p : prim) : bool :=
  match p with
  | Unary p => unary_primitive_in_range p
  | Binary p => binary_primitive_in_range p
  | Variadic p => variadic_primitive_in_range p
  | Heap p => heap_primitive_in_range p
  | Region_primitive _ => true
  end.

(* A value is in range: its constants are, and the tags of its blocks,
   all the way down the fields of blocks, the value slots of closures,
   and the value called and the arguments of a partial application. A
   symbol is: what it is bound to is not looked at here. *)
Fixpoint value_in_range (v : value) : bool :=
  match v with
  | Value_const c | Value_boxed_number c => const_in_range c
  | Value_closure _ value_slots | Value_closure_of_set _ _ value_slots =>
    forallb (fun '(_, v) => value_in_range v) value_slots
  | Value_symbol _ | Value_location _ => true
  | Value_block tag fields =>
    tag_in_range tag && forallb value_in_range fields
  | Value_immutable_array elements => forallb value_in_range elements
  | Value_partial_application callee args _ =>
    value_in_range callee && forallb value_in_range args
  end.

(* What a symbol is bound to is in range. *)
Definition allocated_in_range (allocated : statically_allocated_value)
    : bool :=
  match allocated with
  | Static_closure _ value_slots
  | Static_closure_of_set _ _ _ value_slots =>
    forallb (fun '(_, v) => value_in_range v) value_slots
  | Static_block tag fields =>
    tag_in_range tag && forallb value_in_range fields
  | Static_boxed_number contents => value_in_range contents
  | Static_string bytes => forallb byte_in_range bytes
  | Static_immutable_array elements => forallb value_in_range elements
  | Static_imported _ => true
  end.

(* ======================================================================== *)
(* Kinds                                                                    *)
(* ======================================================================== *)

Definition naked_immediate_kind : flambda_kind :=
  Naked_number Kind_naked_immediate.

(* [Flambda_primitive.arg_kind_of_unary_primitive]. *)
Definition arg_kind_of_unary_primitive (p : unary_primitive) : flambda_kind :=
  match p with
  | Block_load _ _ _ | Is_int _ | Is_null | Get_tag | Array_length _
  | String_length | Boolean_not | Unbox_number _ | Untag_immediate
  | Reinterpret_64_bit_word Tagged_int63_as_unboxed_int64 =>
    Value
  | Opaque_identity kind => kind
  | Unary_int_arith kind _ => kind_of_standard_int kind
  | Num_conv src _ => kind_of_standard_int src
  | Box_number kind => unboxed_kind kind
  | Tag_immediate => naked_immediate_kind
  | Reinterpret_64_bit_word Unboxed_int64_as_tagged_int63 =>
    Naked_number Kind_naked_int64
  end.

(* [Flambda_primitive.result_kind_of_unary_primitive']. *)
Definition result_kind_of_unary_primitive (p : unary_primitive)
    : flambda_kind :=
  match p with
  | Is_int _ | Is_null | Get_tag | String_length | Untag_immediate =>
    naked_immediate_kind
  | Opaque_identity kind => kind
  | Unary_int_arith kind _ => kind_of_standard_int kind
  | Num_conv _ dst => kind_of_standard_int dst
  | Unbox_number kind => unboxed_kind kind
  | Block_load _ _ _ | Array_length _ | Boolean_not | Box_number _
  | Tag_immediate
  | Reinterpret_64_bit_word Unboxed_int64_as_tagged_int63 =>
    Value
  | Reinterpret_64_bit_word Tagged_int63_as_unboxed_int64 =>
    Naked_number Kind_naked_int64
  end.

(* [Flambda_primitive.args_kind_of_binary_primitive]. *)
Definition args_kind_of_binary_primitive (p : binary_primitive)
    : flambda_kind * flambda_kind :=
  match p with
  | String_or_bigstring_load _ =>
    (Value, Naked_number Kind_naked_nativeint)
  | Array_load _ _ _ | Phys_equal _ => (Value, Value)
  | Int_arith kind _ | Int_comp kind _ =>
    (kind_of_standard_int kind, kind_of_standard_int kind)
  | Int_shift kind _ => (kind_of_standard_int kind, naked_immediate_kind)
  end.

(* [Flambda_primitive.result_kind_of_binary_primitive']. *)
Definition result_kind_of_binary_primitive (p : binary_primitive)
    : flambda_kind :=
  match p with
  (* [Array_load_kind.kind_of_loaded_value], which is [Value] for the
     load kinds of the model. *)
  | Array_load _ _ _ => Value
  | Int_arith kind _ | Int_shift kind _ => kind_of_standard_int kind
  | String_or_bigstring_load (Eight | Sixteen) | Phys_equal _
  | Int_comp _ _ =>
    naked_immediate_kind
  | String_or_bigstring_load Eight_signed => Naked_number Kind_naked_int8
  | String_or_bigstring_load Sixteen_signed =>
    Naked_number Kind_naked_int16
  | String_or_bigstring_load Thirty_two => Naked_number Kind_naked_int32
  | String_or_bigstring_load Sixty_four => Naked_number Kind_naked_int64
  end.

(* [Flambda_primitive.args_kind_of_variadic_primitive]: the kind that
   every argument has (the compiler's [Variadic_all_of_kind]). *)
Definition args_kind_of_variadic_primitive (p : variadic_primitive)
    : flambda_kind :=
  match p with
  | Make_block _ | Make_immutable_array _ => Value
  end.

(* [Flambda_primitive.result_kind_of_variadic_primitive']. *)
Definition result_kind_of_variadic_primitive (p : variadic_primitive)
    : flambda_kind :=
  match p with
  | Make_block _ | Make_immutable_array _ => Value
  end.

(* C4's prototype: the kinds of the results of the primitives of the
   heap. *)
Definition result_kind_of_heap_primitive (p : heap_primitive)
    : flambda_kind :=
  match p with
  | Make_block_with_identity _ _ | Block_set _ _ _
  | Make_mutable_array _ | Array_set _ _ =>
    Value
  end.

(* The kinds of the results of the primitives of regions: a region for
   the two that begin one, and a value for the others (the unit value
   for the two that end one, the allocation itself for
   [Local_allocation]). *)
Definition result_kind_of_region_primitive (p : region_primitive)
    : flambda_kind :=
  match p with
  | Begin_region | Begin_try_region => Region
  | End_region | End_try_region | Local_allocation => Value
  end.

(* [Flambda_primitive.result_kind']: the kind of the variable that a
   [Let] binds to the primitive. *)
Definition prim_result_kind (p : prim) : flambda_kind :=
  match p with
  | Unary p => result_kind_of_unary_primitive p
  | Binary p => result_kind_of_binary_primitive p
  | Variadic p => result_kind_of_variadic_primitive p
  | Heap p => result_kind_of_heap_primitive p
  | Region_primitive p => result_kind_of_region_primitive p
  end.

(* Whether arguments of these kinds are as many as the primitive takes,
   and of the kinds it takes. *)
Definition prim_args_well_kinded (p : prim) (kinds : list flambda_kind)
    : bool :=
  match p, kinds with
  | Unary p, [kind] => kind_eqb kind (arg_kind_of_unary_primitive p)
  | Binary p, [kind1; kind2] =>
    kind_eqb kind1 (fst (args_kind_of_binary_primitive p))
    && kind_eqb kind2 (snd (args_kind_of_binary_primitive p))
  | Variadic p, kinds =>
    forallb (fun kind => kind_eqb kind (args_kind_of_variadic_primitive p))
      kinds
  | _, _ => false
  end.

(* ======================================================================== *)
(* Equality, and which primitives to fold                                   *)
(* ======================================================================== *)

Definition prim_eq_dec : ∀ p p' : prim, {p = p'} + {p <> p'}.
Proof. repeat decide equality. Defined.

(* [Flambda_primitive.Without_args.equal]. *)
Definition prim_eqb (p p' : prim) : bool :=
  if prim_eq_dec p p' then true else false.

(* ======================================================================== *)
(* The laws of every primitive                                              *)
(* ======================================================================== *)

(* The value of a primitive depends on the static environment only
   through what the symbols among its arguments are bound to. In
   particular it does not depend on the code, and on constants it does
   not depend on the static environment at all. *)
Definition eval_prim_static_values_statement : Prop :=
  ∀ (statics statics' : static_env) (p : prim) (vs : list value),
    (∀ s : symbol,
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = lookup s (static_values statics')) ->
    eval_prim statics p vs = eval_prim statics' p vs.

(* A primitive that has a value keeps it when the static environment
   grows: when each symbol among its arguments that is bound stays bound
   to the same thing, whatever else comes to be bound. The statement
   above does not give this, as it asks of a symbol that is not bound
   that it stay so. It holds because no primitive has a value by a
   symbol's being bound to nothing
   ([eval_prim_symbols_bound_statement]). The premise is that of
   [FlambdaWellFormedSpec.statics_extend], which a run of the machine
   keeps, asked of the symbols among the arguments only. *)
Definition eval_prim_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (p : prim) (vs : list value)
    (v : value),
    (∀ (s : symbol) (allocated : statically_allocated_value),
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values statics') = Some allocated) ->
    eval_prim statics p vs = Some v ->
    eval_prim statics' p vs = Some v.

(* A primitive has no value on a symbol that is bound to nothing: any
   operation on such a symbol is undefined. (The symbols that are in
   the fields of an argument are not looked at: the primitive does not
   operate on them. In a well-formed unit there is no such symbol
   anywhere.) *)
Definition eval_prim_symbols_bound_statement : Prop :=
  ∀ (statics : static_env) (p : prim) (vs : list value) (v : value)
    (s : symbol),
    eval_prim statics p vs = Some v ->
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) <> None.

(* A primitive that has a value was given arguments of its kinds, as many
   as it takes, and its value is of its result kind. *)
Definition eval_prim_kinds_statement : Prop :=
  ∀ (statics : static_env) (p : prim) (vs : list value) (v : value),
    eval_prim statics p vs = Some v ->
    prim_args_well_kinded p (map kind_of_value vs) = true
    ∧ kind_of_value v = prim_result_kind p.

(* A primitive makes its value out of its arguments: it returns a number,
   one of its arguments, a block or an immutable array of them, a field
   of a block that is one of them, or an element of an immutable array
   that is one of them. So a property of values that holds of numbers,
   that holds of a block exactly when it holds of its fields and of an
   immutable array exactly when it holds of its elements
   ([holds_of_arrays]), and that holds of the fields of the static
   block, and of the elements of the static array, that a symbol with
   the property is bound to, holds of the value of a primitive whose
   arguments have it. That no
   primitive brings in a symbol or a code id is a case
   (FlambdaUnusedStaticsProofs.eval_prim_avoids). *)
Definition holds_of_numbers (P : value -> Prop) : Prop :=
  ∀ c : const, P (Value_const c) ∧ P (Value_boxed_number c).

Definition holds_of_blocks (P : value -> Prop) : Prop :=
  ∀ (tag : Z) (fields : list value),
    P (Value_block tag fields) <-> (∀ v : value, v ∈ fields -> P v).

Definition holds_of_arrays (P : value -> Prop) : Prop :=
  ∀ elements : list value,
    P (Value_immutable_array elements)
    <-> (∀ v : value, v ∈ elements -> P v).

Definition holds_in_static_blocks (statics : static_env)
    (P : value -> Prop) : Prop :=
  ∀ (s : symbol) (tag : Z) (fields : list value),
    P (Value_symbol s) ->
    lookup s (static_values statics) = Some (Static_block tag fields) ->
    ∀ v : value, v ∈ fields -> P v.

Definition holds_in_static_arrays (statics : static_env)
    (P : value -> Prop) : Prop :=
  ∀ (s : symbol) (elements : list value),
    P (Value_symbol s) ->
    lookup s (static_values statics)
    = Some (Static_immutable_array elements) ->
    ∀ v : value, v ∈ elements -> P v.

Definition eval_prim_preserves_statement : Prop :=
  ∀ (P : value -> Prop) (statics : static_env) (p : prim) (vs : list value)
    (v : value),
    holds_of_numbers P ->
    holds_of_blocks P ->
    holds_of_arrays P ->
    holds_in_static_blocks statics P ->
    holds_in_static_arrays statics P ->
    (∀ arg : value, arg ∈ vs -> P arg) ->
    eval_prim statics p vs = Some v ->
    P v.

(* A primitive brings no number out of range. One whose own numbers are
   in range ([prim_in_range]), applied to values in range, has a value
   in range, where each symbol among its arguments is bound to
   something in range. The statement above does not give this: being
   in range does not hold of every number, and holds of a block by its
   tag as well as by its fields. So the constants that the arithmetic
   makes are in the range of their kind, and the tag that [Get_tag]
   reads is an immediate. *)
Definition eval_prim_in_range_statement : Prop :=
  ∀ (statics : static_env) (p : prim) (vs : list value) (v : value),
    prim_in_range p = true ->
    (∀ (s : symbol) (allocated : statically_allocated_value),
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      allocated_in_range allocated = true) ->
    (∀ arg : value, arg ∈ vs -> value_in_range arg = true) ->
    eval_prim statics p vs = Some v ->
    value_in_range v = true.

(* A constant of an integer kind that is in range is an integer of that
   kind: among values in range, the arithmetic, the comparisons, tagging
   and untagging have no value only on what is not a constant of their
   kind. *)
Definition int_of_value_in_range_statement : Prop :=
  ∀ (kind : standard_int) (i : Z),
    const_in_range (const_of_int kind i) = true ->
    int_of_value kind (Value_const (const_of_int kind i)) = Some i.

(* [Phys_equal] of a value against itself: where it has an answer, the
   answer is that the two are the same. Simplify folds the comparison
   of two arguments that are one constant, one symbol or one variable
   to that answer (types/provers.ml, [prove_physical_equality], its
   last match). Here there is an answer on a tagged immediate and on
   an identity only. So a term in which the comparison is folded
   refines the term that has it: it is defined wherever that one is,
   with the same answer. The converse fails, as the folded term is
   defined where the comparison has no answer, and the fold is no
   equivalence. *)
Definition phys_equal_same_statement : Prop :=
  ∀ (statics : static_env) (op : equality_comparison)
    (v result : value),
    eval_binary_primitive statics (Phys_equal op) v v = Some result ->
    result
    = naked_bool
        match op with
        | Phys_eq => true
        | Phys_neq => false
        end.

(* [Phys_equal] of two different symbols: where it has an answer, the
   answer is that the two differ. Simplify folds the comparison of two
   symbols that are not the same to that answer (types/provers.ml,
   [prove_physical_equality], its last match). Here there is an answer
   only where one of the two is an identity
   (FlambdaValueSpec.identity_of: an imported symbol of the class of
   exceptions), and the other is an identity, an allocated value or an
   import that is data. So, as for a value against itself, a term in
   which the comparison is folded refines the term that has it, and
   the fold is no equivalence. *)
Definition phys_equal_symbols_statement : Prop :=
  ∀ (statics : static_env) (op : equality_comparison) (s1 s2 : symbol)
    (result : value),
    s1 <> s2 ->
    eval_binary_primitive statics (Phys_equal op) (Value_symbol s1)
      (Value_symbol s2)
    = Some result ->
    result
    = naked_bool
        match op with
        | Phys_eq => false
        | Phys_neq => true
        end.

(* What a primitive computes on its arguments it computes on values
   that the observations read alike. The premise is a relation between
   values, the first read under [statics] and the second under
   [statics'], in three parts:
   - related values are the same constant, or neither is a constant
     ([relates_constants]);
   - a constant and a boxed number are related to themselves, and blocks
     made alike of related fields, and immutable arrays of related
     elements, are related ([relates_made]): what a primitive makes of
     related arguments is related;
   - what an observation answers of the first value it answers of the
     second, with related parts; if the first is a closure or a partial
     application so is the second; if the first is not a symbol bound
     to nothing nor is the second; if the first is an imported symbol
     the second is the same imported symbol, of the same class; and if
     the first is an immutable array so is the second, with related
     elements ([relates_observations]).
   Equality is such a relation when the two static environments are the
   same. So is the relation between a value and the same value with a
   symbol in place of some of the blocks, boxed numbers and closures in
   it, each symbol bound to what it stands for
   (FlambdaLiftSpec.value_lifted): that is what lets a [Let] of a block
   be made a static binding of a symbol.

   The law goes one way: where the primitive has a value on the first
   arguments it has a related one on the second. Nothing is said where
   it has none on the first; the second may then have a value. So it is
   not that related values cannot be told apart, only that an answer
   given of the first is given of the second.

   It holds because every primitive reads an allocated argument through
   the observations only: by its tag, its fields, its contents, and
   never by which allocation it is. It constrains what a primitive may
   ever do: none may have a value that depends on the identity of an
   immutable allocated value. In particular [Phys_equal] can have no
   value on two of them, or it would tell a block from a symbol bound to
   an equal block. What it answers of an imported symbol does depend on
   its class, and of an identity on which symbol it is, and the part of
   the premise on imported symbols is for that.
   With a heap the premise needs one more part, that a location is
   related to a location only. *)
Definition relates_constants (R : value -> value -> Prop) : Prop :=
  (∀ (c : const) (v' : value), R (Value_const c) v' -> v' = Value_const c)
  ∧ (∀ (v : value) (c : const), R v (Value_const c) -> v = Value_const c).

Definition relates_made (R : value -> value -> Prop) : Prop :=
  (∀ c : const,
    R (Value_const c) (Value_const c)
    ∧ R (Value_boxed_number c) (Value_boxed_number c))
  ∧ (∀ (tag : Z) (fields fields' : list value),
      Forall2 R fields fields' ->
      R (Value_block tag fields) (Value_block tag fields'))
  ∧ (∀ elements elements' : list value,
      Forall2 R elements elements' ->
      R (Value_immutable_array elements)
        (Value_immutable_array elements')).

Definition relates_observations (R : value -> value -> Prop)
    (statics statics' : static_env) : Prop :=
  ∀ v v' : value,
    R v v' ->
    (∀ (tag : Z) (fields : list value),
      block_of statics v = Some (tag, fields) ->
      ∃ fields' : list value,
        block_of statics' v' = Some (tag, fields') ∧ Forall2 R fields fields')
    ∧ (∀ c : const,
        boxed_number_of statics v = Some c ->
        boxed_number_of statics' v' = Some c)
    ∧ (∀ bytes : list Z,
        string_of statics v = Some bytes ->
        string_of statics' v' = Some bytes)
    ∧ (closure_of statics v <> None -> closure_of statics' v' <> None)
    ∧ (partial_application_of v <> None ->
       partial_application_of v' <> None)
    ∧ (bound_if_symbol statics v = true ->
       bound_if_symbol statics' v' = true)
    ∧ (∀ i : symbol * import_class,
        imported_of statics v = Some i ->
        imported_of statics' v' = Some i)
    ∧ (∀ elements : list value,
        array_of statics v = Some elements ->
        ∃ elements' : list value,
          array_of statics' v' = Some elements'
          ∧ Forall2 R elements elements').

Definition respects_observations (R : value -> value -> Prop)
    (statics statics' : static_env) : Prop :=
  relates_constants R ∧ relates_made R
  ∧ relates_observations R statics statics'.

Definition eval_prim_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (statics statics' : static_env) (p : prim)
    (vs vs' : list value) (v : value),
    respects_observations R statics statics' ->
    Forall2 R vs vs' ->
    eval_prim statics p vs = Some v ->
    ∃ v' : value, eval_prim statics' p vs' = Some v' ∧ R v v'.
