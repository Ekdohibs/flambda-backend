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

(* The answers of the provers and of reification, for the tests.

   A prover of the compiler (middle_end/flambda2/types/provers.ml) and its
   transcription in the Rocq model (FlambdaProvers.v, extracted) do not answer
   in one type: the compiler has its sets, its maps and its numbers, the model
   lists and its own numbers. Each answer is made here into an [answer], one
   type for both sides, which is printed in one way. [same] says when two
   answers say the same thing: it is structural equality ([Stdlib.( = )]) after
   these conversions, except that any two [Stops] are the same whatever their
   messages. The same is done of what reification answers (reify.ml;
   FlambdaReify.v).

   This file is trusted like the converter of types (type_converter.ml), and is
   unverified: it is to be read against the types of the answers on the two
   sides (provers.mli, reify.mli; FlambdaProvers.v, FlambdaReify.v). The rules
   it keeps are these.

   [Of_compiler] takes an answer of the compiler and [Of_model] an answer of the
   model. For each type of answer the two have a function of one name.

   A number, a name, a type, a shape, a mode and a kind of the compiler are
   converted by the functions of type_converter.ml. Three things are converted
   here, in the representations of that file: the kind of number that is boxed,
   constructor for constructor ([boxable_number_to_model]; that file converts it
   only inside a type); a string, as the list of its bytes; and a size (below).

   A set is a list in order without repeats, on both sides: of a set of the
   compiler the elements are converted and then put in order; a list of the
   model, which may repeat an element and is in no order that means anything
   (FlambdaProvers.v says so), is put in order.

   A map is the list of its bindings in the order of its keys, without the
   bindings that an earlier one shadows ([Model_type.map]).

   A type inside an answer is in the normal form of model_type.ml, and so are
   the types of a closures entry.

   A size is a natural number in the model and a target integer in the compiler:
   both are an [int] here, by [N.int_of_nat] and by [Target_ocaml_int.to_int].

   Nothing is dropped but what the converter of types drops: the [rec_info] of a
   function type, the name of a poison constant.

   Where the model has no answer ([None]: FlambdaProvers.v says that it is where
   the compiler stops) the answer is [Stops]; the tests make a [Stops] of a
   fatal error of the compiler too. *)

open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_term_basics
open Flambda2_bound_identifiers
module K = Flambda_kind
module TG = Flambda2_types__Type_grammar
module P = Flambda2_types__Provers
module Reify = Flambda2_types__Reify
module M = Flambda2_rocq_semantics.FlambdaModel
module MK = Flambda2_rocq_semantics.FlambdaKindModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module MP = Flambda2_rocq_semantics.FlambdaProvers
module MR = Flambda2_rocq_semantics.FlambdaReify
module N = Model_numbers
module C = Type_converter
module Float32 = Numeric_types.Float32_by_bit_pattern
module Float = Numeric_types.Float_by_bit_pattern
module Int8 = Numeric_types.Int8
module Int16 = Numeric_types.Int16
module Int32 = Numeric_types.Int32
module Int64 = Numeric_types.Int64
module Vec128 = Vector_types.Vec128.Bit_pattern
module Vec256 = Vector_types.Vec256.Bit_pattern
module Vec512 = Vector_types.Vec512.Bit_pattern
module Mask = Vector_types.Mask.Bit_pattern

(* What a prover proves, or the result that it knows. *)
type value =
  | Unit
  | Bool of bool
  | Number of N.z
  | Numbers of N.z list  (** A set: in order, without repeats. *)
  | Simple of M.simple
  | Alloc_mode of MT.Alloc_mode.t
  | Variant_like of
      { const_ctors : N.z list MT.Or_unknown.t;  (** A set. *)
        non_const_ctors_with_sizes : (N.z * (int * MT.block_shape)) list
            (** A map from tags: in the order of the tags. *)
      }
  | Boxed of MT.Alloc_mode.t * MK.boxable_number * MT.flambda_type
  | Tagged_immediate
  | Tag_and_size of N.z * MT.block_shape * int
  | Block of N.z * MT.block_shape * int * M.simple list
      (** The fields are a list: in the order of the fields. *)
  | Closures_entry of
      M.function_slot * MT.Alloc_mode.t * MT.closures_entry * MT.function_type
  | Code_ids of M.code_id list  (** A set. *)
  | Strings of N.z list list
      (** A set of strings, each the list of its bytes. *)

(* What reification lifts: [Reify.to_lift], without its constructors of arrays,
   which the model has not got. The nine constructors of the boxed numbers are
   one, with the kind of number that is boxed, as in the model. *)
type lifted =
  | Immutable_block of N.z * bool * MT.scannable_block_shape * M.simple list
      (** The tag, [is_unique], the shape and the fields. *)
  | Boxed_number of MK.boxable_number * N.z

(* The answer of a prover, or of reification: a [proof_of_property] is [Proved]
   or [Unknown]; a [meet_shortcut] is [Known_result], [Need_meet] or [Invalid];
   a [reification_result] is [Lift], [Simple], [Cannot_reify] or [Invalid]. *)
type answer =
  | Stops of string
      (** No answer: a fatal error of the compiler, [None] of the model. *)
  | Proved of value
  | Unknown
  | Known_result of value
  | Need_meet
  | Invalid
  | Lift of lifted
  | Simple of M.simple
  | Cannot_reify

(* Two answers say the same. Two that stop are the same whatever is said of why:
   the model does not say why. *)
let same (answer : answer) (answer' : answer) : bool =
  match answer, answer' with
  | Stops _, Stops _ -> true
  | ( ( Stops _ | Proved _ | Unknown | Known_result _ | Need_meet | Invalid
      | Lift _ | Simple _ | Cannot_reify ),
      _ ) ->
    Stdlib.( = ) answer answer'

let numbers (numbers : N.z list) : value =
  Numbers (Model_type.set N.compare_z numbers)

(* The elements of a set of target integers. *)
let elements_of_target_ints (set : Target_ocaml_int.Set.t) : N.z list =
  List.map C.z_of_target_int (Target_ocaml_int.Set.elements set)

(* [Flambda_kind.Boxable_number.t]: constructor for constructor. *)
let boxable_number_to_model (number : K.Boxable_number.t) : MK.boxable_number =
  match number with
  | Naked_float32 -> Boxable_naked_float32
  | Naked_float -> Boxable_naked_float
  | Naked_int32 -> Boxable_naked_int32
  | Naked_int64 -> Boxable_naked_int64
  | Naked_nativeint -> Boxable_naked_nativeint
  | Naked_vec128 -> Boxable_naked_vec128
  | Naked_vec256 -> Boxable_naked_vec256
  | Naked_vec512 -> Boxable_naked_vec512
  | Naked_mask -> Boxable_naked_mask

(* ======================================================================== *)
(* The answers of the compiler                                              *)
(* ======================================================================== *)

module Of_compiler = struct
  let unit () : value = Unit

  let bool (b : bool) : value = Bool b

  (* [Target_ocaml_int.t], and a set of them. *)
  let number (i : Target_ocaml_int.t) : value = Number (C.z_of_target_int i)

  let target_ints (set : Target_ocaml_int.Set.t) : value =
    numbers (elements_of_target_ints set)

  (* The sets of the naked numbers of the eleven other kinds: each number as the
     converter of types has it in a type of that kind. *)
  let float32s (set : Float32.Set.t) : value =
    numbers (List.map C.z_of_float32 (Float32.Set.elements set))

  let floats (set : Float.Set.t) : value =
    numbers (List.map C.z_of_float (Float.Set.elements set))

  let int8s (set : Int8.Set.t) : value =
    numbers (List.map C.z_of_int8 (Int8.Set.elements set))

  let int16s (set : Int16.Set.t) : value =
    numbers (List.map C.z_of_int16 (Int16.Set.elements set))

  let int32s (set : Int32.Set.t) : value =
    numbers (List.map C.z_of_int32 (Int32.Set.elements set))

  let int64s (set : Int64.Set.t) : value =
    numbers (List.map C.z_of_int64 (Int64.Set.elements set))

  let nativeints (set : Targetint_32_64.Set.t) : value =
    numbers (List.map C.z_of_nativeint (Targetint_32_64.Set.elements set))

  let vec128s (set : Vec128.Set.t) : value =
    numbers (List.map C.z_of_vec128 (Vec128.Set.elements set))

  let vec256s (set : Vec256.Set.t) : value =
    numbers (List.map C.z_of_vec256 (Vec256.Set.elements set))

  let vec512s (set : Vec512.Set.t) : value =
    numbers (List.map C.z_of_vec512 (Vec512.Set.elements set))

  let masks (set : Mask.Set.t) : value =
    numbers (List.map C.z_of_mask (Mask.Set.elements set))

  (* A set of tags: the numbers of the tags. *)
  let tags (set : Tag.Set.t) : value =
    numbers (List.map C.tag_to_model (Tag.Set.elements set))

  let simple (names : Model_names.t) (simple : Simple.t) : value =
    Simple (C.simple_to_model names simple)

  let alloc_mode (mode : Alloc_mode.For_types.t) : value =
    Alloc_mode (C.alloc_mode_to_model mode)

  (* [Provers.variant_like_proof]: its two fields. A scannable tag is its
     tag. *)
  let variant_like
      ({ const_ctors; non_const_ctors_with_sizes } : P.variant_like_proof) :
      value =
    Variant_like
      { const_ctors =
          C.or_unknown_to_model
            (fun set ->
              Model_type.set N.compare_z (elements_of_target_ints set))
            const_ctors;
        non_const_ctors_with_sizes =
          Model_type.map N.compare_z Fun.id
            (List.map
               (fun (tag, (size, shape)) ->
                 ( C.tag_to_model (Tag.Scannable.to_tag tag),
                   (Target_ocaml_int.to_int size, C.block_shape_to_model shape)
                 ))
               (Tag.Scannable.Map.bindings non_const_ctors_with_sizes))
      }

  (* [Provers.boxed_or_tagged_number]: constructor for constructor. *)
  let boxed_or_tagged_number (names : Model_names.t)
      (number : P.boxed_or_tagged_number) : value =
    match number with
    | Boxed (alloc_mode, number, contents) ->
      Boxed
        ( C.alloc_mode_to_model alloc_mode,
          boxable_number_to_model number,
          Model_type.normalise (C.type_to_model names contents) )
    | Tagged_immediate -> Tagged_immediate

  let tag_and_size
      ((tag, shape, size) : Tag.t * K.Block_shape.t * Target_ocaml_int.t) :
      value =
    Tag_and_size
      ( C.tag_to_model tag,
        C.block_shape_to_model shape,
        Target_ocaml_int.to_int size )

  let block (names : Model_names.t)
      ((tag, shape, size, fields) :
        Tag.t * K.Block_shape.t * Target_ocaml_int.t * Simple.t list) : value =
    Block
      ( C.tag_to_model tag,
        C.block_shape_to_model shape,
        Target_ocaml_int.to_int size,
        List.map (C.simple_to_model names) fields )

  (* A function type is its code id, as in the converter of types. *)
  let closures_entry (names : Model_names.t)
      ((slot, alloc_mode, entry, function_type) :
        Function_slot.t
        * Alloc_mode.For_types.t
        * TG.Closures_entry.t
        * TG.Function_type.t) : value =
    Closures_entry
      ( C.function_slot_to_model names slot,
        C.alloc_mode_to_model alloc_mode,
        Model_type.closures_entry N.compare_positive
          (C.closures_entry_to_model names entry),
        { function_type_code_id =
            C.code_id_to_model names (TG.Function_type.code_id function_type)
        } )

  let code_ids (names : Model_names.t) (set : Code_id.Set.t) : value =
    Code_ids
      (Model_type.set N.compare_positive
         (List.map (C.code_id_to_model names) (Code_id.Set.elements set)))

  (* A string is the list of its bytes, as in the converter of types. *)
  let strings (set : String_info.Set.t) : value =
    Strings
      (Model_type.set Model_type.compare_bytes
         (List.map
            (fun str ->
              List.init (String.length str) (fun i ->
                  N.z_of_int (Char.code str.[i])))
            (String_info.Set.elements set)))

  (* [Provers.proof_of_property] and [Provers.meet_shortcut]: constructor for
     constructor. *)
  let proof (value : 'a -> value) (proof : 'a P.proof_of_property) : answer =
    match proof with Proved a -> Proved (value a) | Unknown -> Unknown

  let shortcut (value : 'a -> value) (shortcut : 'a P.meet_shortcut) : answer =
    match shortcut with
    | Known_result a -> Known_result (value a)
    | Need_meet -> Need_meet
    | Invalid -> Invalid

  (* [Reify.reification_result]: constructor for constructor, but for [Lift],
     where the nine constructors of the boxed numbers are the one
     [Boxed_number], each number as the converter of types has it in a type of
     its kind, and where an array is not in the model. *)
  let reification (names : Model_names.t) (result : Reify.reification_result) :
      answer =
    let boxed number n : answer = Lift (Boxed_number (number, n)) in
    match result with
    | Lift (Immutable_block { tag; is_unique; shape; fields }) -> (
      match C.block_shape_to_model (Scannable shape) with
      | Scannable shape ->
        Lift
          (Immutable_block
             ( C.tag_to_model (Tag.Scannable.to_tag tag),
               is_unique,
               shape,
               List.map (C.simple_to_model names) fields ))
      | Float_record -> assert false)
    | Lift (Boxed_float32 n) -> boxed Boxable_naked_float32 (C.z_of_float32 n)
    | Lift (Boxed_float n) -> boxed Boxable_naked_float (C.z_of_float n)
    | Lift (Boxed_int32 n) -> boxed Boxable_naked_int32 (C.z_of_int32 n)
    | Lift (Boxed_int64 n) -> boxed Boxable_naked_int64 (C.z_of_int64 n)
    | Lift (Boxed_nativeint n) ->
      boxed Boxable_naked_nativeint (C.z_of_nativeint n)
    | Lift (Boxed_vec128 n) -> boxed Boxable_naked_vec128 (C.z_of_vec128 n)
    | Lift (Boxed_vec256 n) -> boxed Boxable_naked_vec256 (C.z_of_vec256 n)
    | Lift (Boxed_vec512 n) -> boxed Boxable_naked_vec512 (C.z_of_vec512 n)
    | Lift (Boxed_mask n) -> boxed Boxable_naked_mask (C.z_of_mask n)
    | Lift
        ( Immutable_float32_array _ | Immutable_float_array _
        | Immutable_int_array _ | Immutable_int8_array _
        | Immutable_int16_array _ | Immutable_int32_array _
        | Immutable_int64_array _ | Immutable_nativeint_array _
        | Immutable_vec128_array _ | Immutable_vec256_array _
        | Immutable_vec512_array _ | Immutable_mask_array _
        | Immutable_value_array _ | Empty_array _ ) ->
      C.not_in_the_model "an array to lift"
    | Simple simple -> Simple (C.simple_to_model names simple)
    | Cannot_reify -> Cannot_reify
    | Invalid -> Invalid
end

(* ======================================================================== *)
(* The answers of the model                                                 *)
(* ======================================================================== *)

module Of_model = struct
  let unit () : value = Unit

  let bool (b : bool) : value = Bool b

  let number (z : N.z) : value = Number z

  (* A set of numbers of any kind, and a set of tags: a list. *)
  let numbers : N.z list -> value = numbers

  let simple (simple : M.simple) : value = Simple simple

  let alloc_mode (mode : MT.Alloc_mode.t) : value = Alloc_mode mode

  (* [variant_like_proof]: its two fields. *)
  let variant_like
      ({ const_ctors; non_const_ctors_with_sizes } : MP.variant_like_proof) :
      value =
    Variant_like
      { const_ctors =
          (match const_ctors with
          | Known numbers -> Known (Model_type.set N.compare_z numbers)
          | Unknown -> Unknown);
        non_const_ctors_with_sizes =
          Model_type.map N.compare_z
            (fun (size, shape) -> N.int_of_nat size, shape)
            non_const_ctors_with_sizes
      }

  (* [Boxed_or_tagged_number.t]: constructor for constructor. *)
  let boxed_or_tagged_number (number : MP.Boxed_or_tagged_number.t) : value =
    match number with
    | Boxed (alloc_mode, number, contents) ->
      Boxed (alloc_mode, number, Model_type.normalise contents)
    | Tagged_immediate -> Tagged_immediate

  let tag_and_size (((tag, shape), size) : (N.z * MT.block_shape) * N.nat) :
      value =
    Tag_and_size (tag, shape, N.int_of_nat size)

  let block
      ((((tag, shape), size), fields) :
        ((N.z * MT.block_shape) * N.nat) * M.simple list) : value =
    Block (tag, shape, N.int_of_nat size, fields)

  let closures_entry
      ((((slot, alloc_mode), entry), function_type) :
        ((M.function_slot * MT.Alloc_mode.t) * MT.closures_entry)
        * MT.function_type) : value =
    Closures_entry
      ( slot,
        alloc_mode,
        Model_type.closures_entry N.compare_positive entry,
        function_type )

  let code_ids (code_ids : M.code_id list) : value =
    Code_ids (Model_type.set N.compare_positive code_ids)

  let strings (strings : N.z list list) : value =
    Strings (Model_type.set Model_type.compare_bytes strings)

  (* [Proof_of_property.t] and [Meet_shortcut.t]: constructor for constructor;
     [None] is no answer. *)
  let no_answer : answer = Stops "the model has no answer"

  let proof (value : 'a -> value) (proof : 'a MP.Proof_of_property.t option) :
      answer =
    match proof with
    | None -> no_answer
    | Some (Proved a) -> Proved (value a)
    | Some Unknown -> Unknown

  let shortcut (value : 'a -> value) (shortcut : 'a MP.Meet_shortcut.t option) :
      answer =
    match shortcut with
    | None -> no_answer
    | Some (Known_result a) -> Known_result (value a)
    | Some Need_meet -> Need_meet
    | Some Invalid -> Invalid

  (* [Reification_result.t]: constructor for constructor; [None] is no
     answer. *)
  let reification (result : MR.Reification_result.t option) : answer =
    match result with
    | None -> no_answer
    | Some (Lift (Immutable_block (tag, is_unique, shape, fields))) ->
      Lift (Immutable_block (tag, is_unique, shape, fields))
    | Some (Lift (Boxed_number (number, n))) -> Lift (Boxed_number (number, n))
    | Some (Simple simple) -> Simple simple
    | Some Cannot_reify -> Cannot_reify
    | Some Invalid -> Invalid
end

(* ======================================================================== *)
(* Printing                                                                 *)
(* ======================================================================== *)

let print_value ppf (value : value) =
  let numbers ppf numbers = Model_type.print_list N.print_z ppf numbers in
  match value with
  | Unit -> Format.pp_print_string ppf "()"
  | Bool b -> Format.pp_print_bool ppf b
  | Number z -> N.print_z ppf z
  | Numbers zs -> numbers ppf zs
  | Simple simple -> Model_type.print_simple ppf simple
  | Alloc_mode mode -> Model_type.print_alloc_mode ppf mode
  | Variant_like { const_ctors; non_const_ctors_with_sizes } ->
    Format.fprintf ppf
      "@[<hov 1>((const_ctors %a)@ (non_const_ctors_with_sizes %a))@]"
      (Model_type.print_or_unknown numbers)
      const_ctors
      (Model_type.print_map N.print_z (fun ppf (size, shape) ->
           Format.fprintf ppf "%d, %a" size Model_type.print_block_shape shape))
      non_const_ctors_with_sizes
  | Boxed (alloc_mode, number, contents) ->
    (* The kind of number that is boxed is shown as the kind of what is in the
       box. *)
    Format.fprintf ppf "@[<hov 1>(Boxed %a@ %a@ %a)@]"
      Model_type.print_alloc_mode alloc_mode Model_type.print_kind
      (MK.unboxed_kind number) Model_type.print contents
  | Tagged_immediate -> Format.pp_print_string ppf "Tagged_immediate"
  | Tag_and_size (tag, shape, size) ->
    Format.fprintf ppf "@[<hov 1>(tag %a,@ %a,@ size %d)@]" N.print_z tag
      Model_type.print_block_shape shape size
  | Block (tag, shape, size, fields) ->
    Format.fprintf ppf "@[<hov 1>(tag %a,@ %a,@ size %d,@ fields %a)@]"
      N.print_z tag Model_type.print_block_shape shape size
      (Model_type.print_list Model_type.print_simple)
      fields
  | Closures_entry (slot, alloc_mode, entry, { function_type_code_id }) ->
    Format.fprintf ppf "@[<hov 1>(slot %a,@ %a,@ %a,@ code%a)@]"
      N.print_positive slot Model_type.print_alloc_mode alloc_mode
      Model_type.print_closures_entry entry N.print_positive
      function_type_code_id
  | Code_ids code_ids ->
    Model_type.print_list
      (fun ppf code_id -> Format.fprintf ppf "code%a" N.print_positive code_id)
      ppf code_ids
  | Strings strings -> Model_type.print_list numbers ppf strings

let print ppf (answer : answer) =
  match answer with
  | Stops why -> Format.fprintf ppf "no answer (%s)" why
  | Proved value ->
    Format.fprintf ppf "@[<hov 1>(Proved@ %a)@]" print_value value
  | Unknown -> Format.pp_print_string ppf "Unknown"
  | Known_result value ->
    Format.fprintf ppf "@[<hov 1>(Known_result@ %a)@]" print_value value
  | Need_meet -> Format.pp_print_string ppf "Need_meet"
  | Invalid -> Format.pp_print_string ppf "Invalid"
  | Lift (Immutable_block (tag, is_unique, shape, fields)) ->
    Format.fprintf ppf
      "@[<hov 1>(Lift@ (Immutable_block@ (tag %a)@ (is_unique %b)@ (shape %a)@ \
       (fields %a)))@]"
      N.print_z tag is_unique Model_type.print_block_shape
      (Scannable shape : MT.block_shape)
      (Model_type.print_list Model_type.print_simple)
      fields
  | Lift (Boxed_number (number, n)) ->
    Format.fprintf ppf "@[<hov 1>(Lift@ (Boxed_number@ %a@ %a))@]"
      Model_type.print_kind (MK.unboxed_kind number) N.print_z n
  | Simple simple ->
    Format.fprintf ppf "@[<hov 1>(Simple@ %a)@]" Model_type.print_simple simple
  | Cannot_reify -> Format.pp_print_string ppf "Cannot_reify"
