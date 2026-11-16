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


(* What some C functions of the runtime compute: a table of functions on
   the values of the model, for the day a call of a C function has a
   meaning. Nothing uses it yet.

   This file is to be reviewed: [eval_extcall] and what it is made of,
   against runtime/str.c ([caml_string_equal], [caml_string_notequal],
   [caml_string_compare], [caml_string_lessthan] and the three like it)
   and runtime/compare.c ([do_compare_val], and [caml_compare],
   [caml_equal] and the five like them); the rewrites of
   middle_end/flambda2/simplify/simplify_extcall.ml, against
   [specialised_for_tagged_immediates]; and the statements.

   The machine has no step for a call of a C function, and the conversion
   from the compiler's terms refuses one. How such a call would step is
   not decided, and is not this file's to decide
   (requests/P1-c-calls.md). What is here is the part that does not
   depend on it: for fourteen functions that the front end calls for
   plain programs on strings and integers, the value of the function on
   the values of its arguments, where it is a function of them alone.

   The functions, each of two arguments, by their names in the runtime:
   - [Caml_string_comparison c]: [caml_string_compare],
     [caml_string_equal], [caml_string_notequal], [caml_string_lessthan],
     [caml_string_lessequal], [caml_string_greaterthan],
     [caml_string_greaterequal], for [c] from [Cmp_compare] on. They are
     what [compare], [=], [<>], [<], [<=], [>] and [>=] become at the
     type [string], and what a [match] on a string is made of.
   - [Caml_comparison c]: [caml_compare], [caml_equal], [caml_notequal],
     [caml_lessthan], [caml_lessequal], [caml_greaterthan],
     [caml_greaterequal]: the same seven at a type that the front end
     does not know.

   Where each has a value, and what the C function does elsewhere. "No
   value" below is a decision to be reviewed, not a fact about C: in
   each case the C function does something.
   - The comparisons of strings have a value on two strings. On them
     they read the two strings and nothing else, allocate nothing and
     raise nothing. On anything else they have no value here. The C
     functions do not look at what they are given. Each answers at
     once, "equal", when its two arguments are the same word, be it a
     pointer or a tagged immediate. Otherwise [caml_string_equal]
     compares the words of two blocks of any tag, and
     [caml_string_compare] takes the last byte of a block for the
     padding of a string and compares the bytes that this leaves; given
     a tagged immediate that is not its other argument, each reads the
     memory before an address that is no block. None raises or
     allocates.
   - The polymorphic comparisons have a value on two tagged immediates,
     which they compare as integers, and on two strings, which they
     compare as the comparisons of strings do. On these they return at
     once: they read nothing else, allocate nothing, raise nothing and
     do not poll. On anything else they have no value here. The C
     function does far more, and on a pair of values it may return,
     raise or not end. A tagged immediate is less than any block that
     is not a custom block; two blocks of different tags are ordered by
     their tags, and two of one tag by their sizes, then their fields
     in order, strings by their bytes, boxed integers by their numbers,
     floats as numbers with a rule for NaN that differs between
     [compare] and the others. It raises [Invalid_argument] when it
     comes to compare two closures, two abstract blocks, two
     continuations, or two blocks of one tag of which one is a mixed
     block; it does not come to that where an earlier field or a
     difference of tags has decided, nor, for [compare], where the two
     values are the same pointer. On a long
     comparison it runs pending actions (signal handlers and
     finalisers, which may raise); it allocates its own stack outside
     the heap for deep values; and it does not end on some cyclic
     values.

   The model's strings are immutable. A string that is the block of
   bytes written later, which an erased conversion from bytes allows
   (X5-3, at FlambdaPrimitiveSpec.eval_string_load), is outside the
   model for these functions as it is for the loads.

   The proofs are in FlambdaExtcallProofs.v, which needs no review. *)

From Stdlib Require Import ZArith List Bool.
From OxCamlStdlib Require Import OxCamlLogic OxCamlOption OxCamlList.
From OxCamlFlambda2 Require Import FlambdaKindModel FlambdaModel
  FlambdaMapSpec FlambdaValueSpec FlambdaPrimitiveModel
  FlambdaPrimitiveSpec.
Import ListNotations.

Local Open Scope Z_scope.

(* ======================================================================== *)
(* The functions                                                            *)
(* ======================================================================== *)

(* What a comparison answers: an integer as [compare] does, or a boolean
   for one of the six relations. *)
Inductive extcall_comparison : Type :=
  | Cmp_compare
  | Cmp_equal
  | Cmp_notequal
  | Cmp_lessthan
  | Cmp_lessequal
  | Cmp_greaterthan
  | Cmp_greaterequal.

(* The C functions of the table. *)
Inductive extcall : Type :=
  | Caml_string_comparison (c : extcall_comparison)
  | Caml_comparison (c : extcall_comparison).

(* ======================================================================== *)
(* The order of strings                                                     *)
(* ======================================================================== *)

(* The order of two strings by their bytes: that of the first byte in
   which they differ, the bytes read without a sign, and that of their
   lengths if one is the beginning of the other. It is what
   [caml_string_compare] computes with [memcmp] on the shorter length
   and then the lengths. Of each element of a list only the low eight
   bits are taken, as the loads do, which changes nothing of a byte. *)
Fixpoint compare_bytes (bytes1 bytes2 : list Z) : comparison :=
  match bytes1, bytes2 with
  | [], [] => Eq
  | [], _ :: _ => Lt
  | _ :: _, [] => Gt
  | byte1 :: rest1, byte2 :: rest2 =>
    match byte1 mod 256 ?= byte2 mod 256 with
    | Eq => compare_bytes rest1 rest2
    | Lt => Lt
    | Gt => Gt
    end
  end.

(* ======================================================================== *)
(* The answers                                                              *)
(* ======================================================================== *)

(* An order as the integer that [compare] returns. *)
Definition sign_of (order : comparison) : Z :=
  match order with
  | Lt => -1
  | Eq => 0
  | Gt => 1
  end.

(* What a comparison answers of an order: the integer of [compare], or
   1 or 0 for a relation that holds or not. Each relation is read off
   the integer, as in the C: [caml_string_lessthan] is
   [caml_string_compare (s1, s2) < Val_int (0)]. *)
Definition answer_of (c : extcall_comparison) (order : comparison) : Z :=
  let sign := sign_of order in
  match c with
  | Cmp_compare => sign
  | Cmp_equal => Z.b2z (sign =? 0)
  | Cmp_notequal => Z.b2z (negb (sign =? 0))
  | Cmp_lessthan => Z.b2z (sign <? 0)
  | Cmp_lessequal => Z.b2z (sign <=? 0)
  | Cmp_greaterthan => Z.b2z (0 <? sign)
  | Cmp_greaterequal => Z.b2z (0 <=? sign)
  end.

(* An answer as a value: a tagged immediate, which is how the integer
   of [compare] and a boolean of the source are represented. *)
Definition value_of_answer (c : extcall_comparison) (order : comparison)
    : value :=
  Value_const (Tagged_immediate (answer_of c order)).

(* ======================================================================== *)
(* The values of the functions                                              *)
(* ======================================================================== *)

(* The order that the polymorphic comparison finds between two values:
   that of the integers for two tagged immediates, that of the bytes for
   two strings. *)
(* CR mshinwell: the polymorphic comparison has no value here but on two
   tagged immediates and on two strings. The C function returns, raises
   or does not end on every pair of values (the header of this file
   says how). So this table has no result for the comparison of two
   tuples, two lists or two boxed integers with [=], which OCaml
   defines. What "no result" is to mean for a program is not settled,
   and this file does not say: it has no machine. It is to be decided
   with what a call of a C function is: either this function is
   extended (to immutable blocks by structure, to boxed integers, with
   a raise on a closure), or a call on other values is a case of its
   own, outside the model, that stops the check of a unit. It would be
   wrong to read "no result" as "undefined" and to let a pass rewrite
   such a call freely. *)
Definition compare_values (statics : static_env) (v1 v2 : value)
    : option comparison :=
  match
    int_of_value Standard_tagged_immediate v1,
    int_of_value Standard_tagged_immediate v2
  with
  | Some i1, Some i2 => Some (i1 ?= i2)
  | _, _ =>
    let* bytes1 := string_of statics v1 in
    let* bytes2 := string_of statics v2 in
    Some (compare_bytes bytes1 bytes2)
  end.

(* [eval_extcall statics f vs] is what the C function [f] returns on the
   values [vs], if it is given two and they are in the part of its
   domain that the table has. *)
(* CR mshinwell: a comparison of strings has no value on what is not a
   string. The C functions compare the memory of whatever two blocks
   they are given, so [caml_string_equal] of two tuples answers by
   their words on the machine. The front end calls them at the type
   [string] only (and at [bytes], which the model has not got), so it
   takes values that types do not vouch for (those of a GADT, or of
   [Obj.magic]) to reach them otherwise. *)
Definition eval_extcall (statics : static_env) (f : extcall)
    (vs : list value) : option value :=
  match f, vs with
  | Caml_string_comparison c, [v1; v2] =>
    let* bytes1 := string_of statics v1 in
    let* bytes2 := string_of statics v2 in
    Some (value_of_answer c (compare_bytes bytes1 bytes2))
  | Caml_comparison c, [v1; v2] =>
    let* order := compare_values statics v1 v2 in
    Some (value_of_answer c order)
  | _, _ => None
  end.

(* ======================================================================== *)
(* What Simplify makes of a polymorphic comparison                          *)
(* ======================================================================== *)

(* Where the types prove that both arguments of a polymorphic comparison
   are tagged immediates, Simplify replaces the call by this primitive
   on the two arguments, followed by [Tag_immediate]
   ([Simplify_extcall.simplify_comparison_of_tagged_immediates], with
   the primitives that [simplify_returning_extcall] gives it). *)
Definition specialised_for_tagged_immediates (c : extcall_comparison)
    : binary_primitive :=
  match c with
  | Cmp_compare =>
    Int_comp Standard_tagged_immediate
      (Yielding_int_like_compare_functions Signed)
  | Cmp_equal => Phys_equal Phys_eq
  | Cmp_notequal => Phys_equal Phys_neq
  | Cmp_lessthan =>
    Int_comp Standard_tagged_immediate (Yielding_bool (Int_lt Signed))
  | Cmp_lessequal =>
    Int_comp Standard_tagged_immediate (Yielding_bool (Int_le Signed))
  | Cmp_greaterthan =>
    Int_comp Standard_tagged_immediate (Yielding_bool (Int_gt Signed))
  | Cmp_greaterequal =>
    Int_comp Standard_tagged_immediate (Yielding_bool (Int_ge Signed))
  end.

(* ======================================================================== *)
(* Statements: the order of strings                                         *)
(* ======================================================================== *)

(* The order of strings is a total order on strings read by their
   bytes: a string is equal to itself, two strings are equal exactly
   when they have the same bytes, the order one way is the opposite of
   the order the other way, and each of "less", "equal" and "greater"
   is transitive. *)
Definition compare_bytes_refl_statement : Prop :=
  ∀ bytes : list Z, compare_bytes bytes bytes = Eq.

Definition compare_bytes_eq_statement : Prop :=
  ∀ bytes1 bytes2 : list Z,
    compare_bytes bytes1 bytes2 = Eq
    <-> map (fun byte => byte mod 256) bytes1
        = map (fun byte => byte mod 256) bytes2.

(* Of strings whose bytes are bytes, as those of a unit that passes
   FlambdaWellFormedSpec.numbers_in_range are: equal exactly when they
   are the same string. *)
Definition compare_bytes_eq_in_range_statement : Prop :=
  ∀ bytes1 bytes2 : list Z,
    forallb byte_in_range bytes1 = true ->
    forallb byte_in_range bytes2 = true ->
    (compare_bytes bytes1 bytes2 = Eq <-> bytes1 = bytes2).

Definition compare_bytes_antisym_statement : Prop :=
  ∀ bytes1 bytes2 : list Z,
    compare_bytes bytes2 bytes1 = CompOpp (compare_bytes bytes1 bytes2).

Definition compare_bytes_trans_statement : Prop :=
  ∀ (order : comparison) (bytes1 bytes2 bytes3 : list Z),
    compare_bytes bytes1 bytes2 = order ->
    compare_bytes bytes2 bytes3 = order ->
    compare_bytes bytes1 bytes3 = order.

(* Equal strings are ordered alike against any other. *)
Definition compare_bytes_eq_compat_statement : Prop :=
  ∀ bytes1 bytes2 bytes3 : list Z,
    compare_bytes bytes1 bytes2 = Eq ->
    compare_bytes bytes1 bytes3 = compare_bytes bytes2 bytes3.

(* ======================================================================== *)
(* Statements: the functions among themselves                               *)
(* ======================================================================== *)

(* On two strings the polymorphic comparison is the comparison of
   strings: the front end calls the second where it knows the type and
   the first where it does not, and they answer alike. *)
Definition polymorphic_comparison_of_strings_statement : Prop :=
  ∀ (statics : static_env) (c : extcall_comparison) (v1 v2 : value)
    (bytes1 bytes2 : list Z),
    string_of statics v1 = Some bytes1 ->
    string_of statics v2 = Some bytes2 ->
    eval_extcall statics (Caml_comparison c) [v1; v2]
    = eval_extcall statics (Caml_string_comparison c) [v1; v2].

(* On two tagged immediates a polymorphic comparison has the value of
   the primitive that Simplify replaces it by, tagged. This is equality
   of the two results and no more. If a call of a C function came to be
   something that a run shows (an event), the rewrite would need a law
   besides, that the event of such a call may be dropped. *)
Definition specialised_for_tagged_immediates_statement : Prop :=
  ∀ (statics : static_env) (c : extcall_comparison) (v1 v2 : value)
    (i1 i2 : Z),
    int_of_value Standard_tagged_immediate v1 = Some i1 ->
    int_of_value Standard_tagged_immediate v2 = Some i2 ->
    ∃ compared : value,
      eval_prim statics (Binary (specialised_for_tagged_immediates c))
        [v1; v2]
      = Some compared
      ∧ eval_prim statics (Unary Tag_immediate) [compared]
        = eval_extcall statics (Caml_comparison c) [v1; v2].

(* ======================================================================== *)
(* Statements: the laws that the primitives obey                            *)
(* ======================================================================== *)

(* The laws of FlambdaPrimitiveSpec.v, of the functions of the table: a
   call of one of them could be evaluated as a primitive is, by a
   machine that goes by those laws. *)

(* The value depends on the static environment only through what the
   symbols among the arguments are bound to. *)
Definition eval_extcall_static_values_statement : Prop :=
  ∀ (statics statics' : static_env) (f : extcall) (vs : list value),
    (∀ s : symbol,
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = lookup s (static_values statics')) ->
    eval_extcall statics f vs = eval_extcall statics' f vs.

(* A value is kept when the static environment grows. *)
Definition eval_extcall_statics_extend_statement : Prop :=
  ∀ (statics statics' : static_env) (f : extcall) (vs : list value)
    (v : value),
    (∀ (s : symbol) (allocated : statically_allocated_value),
      Value_symbol s ∈ vs ->
      lookup s (static_values statics) = Some allocated ->
      lookup s (static_values statics') = Some allocated) ->
    eval_extcall statics f vs = Some v ->
    eval_extcall statics' f vs = Some v.

(* No value on a symbol that is bound to nothing. *)
Definition eval_extcall_symbols_bound_statement : Prop :=
  ∀ (statics : static_env) (f : extcall) (vs : list value) (v : value)
    (s : symbol),
    eval_extcall statics f vs = Some v ->
    Value_symbol s ∈ vs ->
    lookup s (static_values statics) <> None.

(* A function that has a value was given two arguments of kind [Value],
   and its value is of kind [Value]: the signature that the front end
   declares for each of them, [val * val -> val]. *)
Definition eval_extcall_kinds_statement : Prop :=
  ∀ (statics : static_env) (f : extcall) (vs : list value) (v : value),
    eval_extcall statics f vs = Some v ->
    map kind_of_value vs = [Value; Value] ∧ kind_of_value v = Value.

(* The value is the tagged immediate -1, 0 or 1 for a function like
   [compare], and 0 or 1 for the others. So it is in range, and it is a
   constant: the functions allocate nothing and return none of their
   arguments. *)
Definition eval_extcall_answer_statement : Prop :=
  ∀ (statics : static_env) (f : extcall) (vs : list value) (v : value),
    eval_extcall statics f vs = Some v ->
    ∃ i : Z,
      v = Value_const (Tagged_immediate i)
      ∧ match f with
        | Caml_string_comparison Cmp_compare | Caml_comparison Cmp_compare =>
          i = -1 ∨ i = 0 ∨ i = 1
        | _ => i = 0 ∨ i = 1
        end.

Definition eval_extcall_in_range_statement : Prop :=
  ∀ (statics : static_env) (f : extcall) (vs : list value) (v : value),
    eval_extcall statics f vs = Some v -> value_in_range v = true.

(* What a function computes on its arguments it computes on values that
   the observations read alike, as a primitive does
   (FlambdaPrimitiveSpec.eval_prim_related_statement): a function reads
   a string by its bytes, and never by which allocation it is. The C
   functions do answer at once on two pointers that are the same, but
   to say what they would have found by reading. *)
Definition eval_extcall_related_statement : Prop :=
  ∀ (R : value -> value -> Prop) (statics statics' : static_env)
    (f : extcall) (vs vs' : list value) (v : value),
    respects_observations R statics statics' ->
    Forall2 R vs vs' ->
    eval_extcall statics f vs = Some v ->
    ∃ v' : value, eval_extcall statics' f vs' = Some v' ∧ R v v'.
