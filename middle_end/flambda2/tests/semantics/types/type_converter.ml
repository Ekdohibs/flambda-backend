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

(* The converter between the types of the compiler and those of the Rocq model.

   [Type_grammar.t] (middle_end/flambda2/types/grammar/type_grammar.ml) on one
   side, [flambda_type] (FlambdaTypeGrammar.v, extracted) on the other. The
   tests that compare a function of the model with the function of the compiler
   that it transcribes go through this file, so it is trusted, and it is
   unverified: it is to be read, case by case, against the two grammars. The
   header of FlambdaTypeGrammar.v says, constructor by constructor, what the
   model has; a case below that is not a constructor for a constructor says what
   it does and refers to it.

   The rules it keeps: - Nothing is approximated. What the model has not got
   ([Array], a [Simple] with a coercion) is refused, by [Not_in_the_model]. What
   the compiler cannot build from a type of the model (a number outside the
   range of its kind, a kind of naked number that has no type, an empty set of
   numbers as a head, a name that has no number) is refused, by
   [Not_in_the_compiler]. - What the model leaves out on purpose is dropped on
   the way to the model, and said so here: the [rec_info] of a function type,
   the expression of a [Rec_info] head, the string of a [Poison] constant, the
   cache of free names of a type. On the way back a function type gets the
   unknown [rec_info]; a [Rec_info] head is refused, as the model does not say
   which expression it was. (For the tests that compare a type of the compiler
   with what comes back, two conversions at the end of this file say what was
   dropped of [Rec_info] and put the [rec_info] of function types back.) - The
   maps of the model are association lists of which the first binding of a key
   counts, and its sets are lists. A map of the compiler becomes the list of its
   bindings and a set the list of its elements, in the order of the compiler's
   maps and sets, which means nothing. A list of the model becomes a map by
   adding its bindings from the last to the first, so that the first binding of
   a key is the one kept ([Map.of_list] would keep the last). - The names are
   numbered by the [Model_names.t] given (model_names.ml), and the numbers
   converted by model_numbers.ml; both are trusted with this file.

   The types of the compiler are abstract outside its types library, and
   [Type_grammar] is not exported by it. This file reaches the modules of the
   library under the names the build gives them ([Flambda2_types__...]), which
   needs no change to the compiler. One type is abstract even there: the size of
   a case of blocks ([Type_grammar.Block_size.t]). [int_of_block_size] and
   [block_size_of_int] read and make one through functions of [Type_grammar]
   that are exported. *)

open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_term_basics
open Flambda2_bound_identifiers
open Flambda2_lattices
open Flambda2_nominal
module K = Flambda_kind
module TG = Flambda2_types__Type_grammar
module TD = Flambda2_types__Type_descr
module MTC = Flambda2_types__More_type_creators
module Contents = Flambda2_types__Set_of_closures_contents
module M = Flambda2_rocq_semantics.FlambdaModel
module MK = Flambda2_rocq_semantics.FlambdaKindModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module N = Model_numbers
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
module I64 = Stdlib.Int64

exception Not_in_the_model of string

exception Not_in_the_compiler of string

let not_in_the_model fmt =
  Format.kasprintf (fun what -> raise (Not_in_the_model what)) fmt

let not_in_the_compiler fmt =
  Format.kasprintf (fun what -> raise (Not_in_the_compiler what)) fmt

type machine_width = Target_system.Machine_width.t

(* ======================================================================== *)
(* What is dropped of [Rec_info]                                            *)
(* ======================================================================== *)

(* The model has not got the [rec_info] of a function type nor the expression of
   a [Rec_info] head, and the conversion to the model drops them. A test that
   compares a type of the compiler with what comes back from the model needs to
   know what was dropped, and to have it put back:
   [type_to_model_with_what_is_dropped] and [type_of_model_with_rec_infos], at
   the end of this file, are the conversions with that. They work through the
   two references below, which the conversions consult and which are as they are
   here but during a call of one of those two functions: nothing is recorded,
   and a function type gets the unknown [rec_info]. *)
type dropped =
  { rec_infos : (Code_id.t * TG.t) list;
        (** The [rec_info] of each function type, the unknown one too, with the
            code id of the function type, in the order of the conversion. *)
    variables : Variable.t list
        (** The variables that those [rec_info]s mention, and those that the
            expressions of the [Rec_info] heads dropped mention. *)
  }

let recording : dropped option ref = ref None

let record_rec_info code_id rec_info =
  match !recording with
  | None -> ()
  | Some dropped ->
    recording
      := Some
           { rec_infos = (code_id, rec_info) :: dropped.rec_infos;
             variables =
               Name_occurrences.fold_variables (TG.free_names rec_info)
                 ~init:dropped.variables ~f:(fun variables variable ->
                   variable :: variables)
           }

let record_rec_info_expr (expr : Rec_info_expr.t) =
  match !recording with
  | None -> ()
  | Some dropped ->
    recording
      := Some
           { dropped with
             variables =
               Name_occurrences.fold_variables (Rec_info_expr.free_names expr)
                 ~init:dropped.variables ~f:(fun variables variable ->
                   variable :: variables)
           }

(* The [rec_info] that a function type made from the model gets. *)
let rec_info_of_code : (Code_id.t -> TG.t) ref =
  ref (fun (_ : Code_id.t) -> TG.any_rec_info)

(* ======================================================================== *)
(* Numbers                                                                  *)
(* ======================================================================== *)

(* Each sort of number of the compiler, to the model and back. An integer is its
   signed value, and a bit pattern is read as an integer in two's complement of
   as many bits as it has (model_numbers.ml). Back, [None] is a number outside
   the range. *)

let z_of_target_int (i : Target_ocaml_int.t) : N.z =
  N.z_of_int64 (Target_ocaml_int.to_int64 i)

(* [Target_ocaml_int.of_int64] reduces its argument to the width of the target's
   [int], so a number is in range if it comes back the same. *)
let target_int_of_z (machine_width : machine_width) (z : N.z) :
    Target_ocaml_int.t option =
  Option.bind (N.int64_of_z z) (fun n ->
      let i = Target_ocaml_int.of_int64 machine_width n in
      if I64.equal (Target_ocaml_int.to_int64 i) n then Some i else None)

let z_of_int8 (i : Int8.t) : N.z = N.z_of_int (Int8.to_int i)

let int8_of_z (z : N.z) : Int8.t option =
  match N.int_of_z z with
  | Some n when n >= -128 && n <= 127 -> Some (Int8.of_int n)
  | Some _ | None -> None

let z_of_int16 (i : Int16.t) : N.z = N.z_of_int (Int16.to_int i)

let int16_of_z (z : N.z) : Int16.t option =
  match N.int_of_z z with
  | Some n when n >= -32768 && n <= 32767 -> Some (Int16.of_int n)
  | Some _ | None -> None

let z_of_int32 (i : int32) : N.z = N.z_of_int64 (I64.of_int32 i)

let int32_of_z (z : N.z) : int32 option =
  match N.int64_of_z z with
  | Some n when I64.equal (I64.of_int32 (I64.to_int32 n)) n ->
    Some (I64.to_int32 n)
  | Some _ | None -> None

let z_of_int64 (i : int64) : N.z = N.z_of_int64 i

let int64_of_z (z : N.z) : int64 option = N.int64_of_z z

(* The 32 bits of a float32 and the 64 of a float, as the integers of those
   widths. *)
let z_of_float32 (f : Float32.t) : N.z = z_of_int32 (Float32.to_bits f)

let float32_of_z (z : N.z) : Float32.t option =
  Option.map Float32.of_bits (int32_of_z z)

let z_of_float (f : Float.t) : N.z = z_of_int64 (Float.to_bits f)

let float_of_z (z : N.z) : Float.t option =
  Option.map Float.of_bits (int64_of_z z)

let z_of_nativeint (i : Targetint_32_64.t) : N.z =
  N.z_of_int64 (Targetint_32_64.to_int64 i)

(* As for [target_int_of_z]. *)
let nativeint_of_z (machine_width : machine_width) (z : N.z) :
    Targetint_32_64.t option =
  Option.bind (N.int64_of_z z) (fun n ->
      let i = Targetint_32_64.of_int64 machine_width n in
      if I64.equal (Targetint_32_64.to_int64 i) n then Some i else None)

let z_of_vec128 (v : Vec128.t) : N.z =
  let ({ word0; word1 } : Vec128.bits) = Vec128.to_bits v in
  N.z_of_signed_words [word0; word1]

let vec128_of_z (z : N.z) : Vec128.t option =
  match N.signed_words_of_z ~count:2 z with
  | Some [word0; word1] -> Some (Vec128.of_bits { word0; word1 })
  | Some _ | None -> None

let z_of_vec256 (v : Vec256.t) : N.z =
  let ({ word0; word1; word2; word3 } : Vec256.bits) = Vec256.to_bits v in
  N.z_of_signed_words [word0; word1; word2; word3]

let vec256_of_z (z : N.z) : Vec256.t option =
  match N.signed_words_of_z ~count:4 z with
  | Some [word0; word1; word2; word3] ->
    Some (Vec256.of_bits { word0; word1; word2; word3 })
  | Some _ | None -> None

let z_of_vec512 (v : Vec512.t) : N.z =
  let ({ word0; word1; word2; word3; word4; word5; word6; word7 } : Vec512.bits)
      =
    Vec512.to_bits v
  in
  N.z_of_signed_words [word0; word1; word2; word3; word4; word5; word6; word7]

let vec512_of_z (z : N.z) : Vec512.t option =
  match N.signed_words_of_z ~count:8 z with
  | Some [word0; word1; word2; word3; word4; word5; word6; word7] ->
    Some
      (Vec512.of_bits
         { word0; word1; word2; word3; word4; word5; word6; word7 })
  | Some _ | None -> None

let z_of_mask (m : Mask.t) : N.z =
  let ({ word0 } : Mask.bits) = Mask.to_bits m in
  z_of_int64 word0

let mask_of_z (z : N.z) : Mask.t option =
  Option.map (fun word0 -> Mask.of_bits { word0 }) (int64_of_z z)

(* A number of the model as a number of the compiler, or a refusal. *)
let in_range what (of_z : N.z -> 'a option) (z : N.z) : 'a =
  match of_z z with
  | Some n -> n
  | None -> not_in_the_compiler "the number %a is not %s" N.print_z z what

(* ======================================================================== *)
(* Kinds, names, constants, [Simple]s                                       *)
(* ======================================================================== *)

(* [Flambda_kind.Naked_number_kind.t]: constructor for constructor. *)
let naked_number_kind_to_model (k : K.Naked_number_kind.t) :
    MK.naked_number_kind =
  match k with
  | Naked_immediate -> Kind_naked_immediate
  | Naked_float32 -> Kind_naked_float32
  | Naked_float -> Kind_naked_float
  | Naked_int8 -> Kind_naked_int8
  | Naked_int16 -> Kind_naked_int16
  | Naked_int32 -> Kind_naked_int32
  | Naked_int64 -> Kind_naked_int64
  | Naked_nativeint -> Kind_naked_nativeint
  | Naked_vec128 -> Kind_naked_vec128
  | Naked_vec256 -> Kind_naked_vec256
  | Naked_vec512 -> Kind_naked_vec512
  | Naked_mask -> Kind_naked_mask

let naked_number_kind_of_model (k : MK.naked_number_kind) :
    K.Naked_number_kind.t =
  match k with
  | Kind_naked_immediate -> Naked_immediate
  | Kind_naked_float32 -> Naked_float32
  | Kind_naked_float -> Naked_float
  | Kind_naked_int8 -> Naked_int8
  | Kind_naked_int16 -> Naked_int16
  | Kind_naked_int32 -> Naked_int32
  | Kind_naked_int64 -> Naked_int64
  | Kind_naked_nativeint -> Naked_nativeint
  | Kind_naked_vec128 -> Naked_vec128
  | Kind_naked_vec256 -> Naked_vec256
  | Kind_naked_vec512 -> Naked_vec512
  | Kind_naked_mask -> Naked_mask

(* [Flambda_kind.t]: constructor for constructor. *)
let kind_to_model (k : K.t) : MK.flambda_kind =
  match k with
  | Value -> Value
  | Naked_number k -> Naked_number (naked_number_kind_to_model k)
  | Region -> Region
  | Rec_info -> Rec_info

let kind_of_model (k : MK.flambda_kind) : K.t =
  match k with
  | Value -> K.value
  | Naked_number k -> K.naked_number (naked_number_kind_of_model k)
  | Region -> K.region
  | Rec_info -> K.rec_info

(* Names. A name of the model that the numbering has no name for is refused. *)
let named what (sort : _ Model_names.sort) (p : N.positive) =
  match sort.named p with
  | Some name -> name
  | None -> not_in_the_compiler "no %s is numbered %a" what N.print_positive p

let variable_to_model (names : Model_names.t) (var : Variable.t) : M.variable =
  names.variables.number var

let variable_of_model (names : Model_names.t) (var : M.variable) : Variable.t =
  named "variable" names.variables var

let symbol_to_model (names : Model_names.t) (symbol : Symbol.t) : M.symbol =
  names.symbols.number symbol

let symbol_of_model (names : Model_names.t) (symbol : M.symbol) : Symbol.t =
  named "symbol" names.symbols symbol

let code_id_to_model (names : Model_names.t) (code_id : Code_id.t) : M.code_id =
  names.code_ids.number code_id

let code_id_of_model (names : Model_names.t) (code_id : M.code_id) : Code_id.t =
  named "code id" names.code_ids code_id

let value_slot_to_model (names : Model_names.t) (slot : Value_slot.t) :
    M.value_slot =
  names.value_slots.number slot

let value_slot_of_model (names : Model_names.t) (slot : M.value_slot) :
    Value_slot.t =
  named "value slot" names.value_slots slot

let function_slot_to_model (names : Model_names.t) (slot : Function_slot.t) :
    M.function_slot =
  names.function_slots.number slot

let function_slot_of_model (names : Model_names.t) (slot : M.function_slot) :
    Function_slot.t =
  named "function slot" names.function_slots slot

(* [Name.t]: a variable or a symbol. *)
let name_to_model (names : Model_names.t) (name : Name.t) : MT.name =
  Name.pattern_match name
    ~var:(fun var : MT.name -> Name_var (variable_to_model names var))
    ~symbol:(fun symbol : MT.name -> Name_symbol (symbol_to_model names symbol))

let name_of_model (names : Model_names.t) (name : MT.name) : Name.t =
  match name with
  | Name_var var -> Name.var (variable_of_model names var)
  | Name_symbol symbol -> Name.symbol (symbol_of_model names symbol)

let option_to_model to_model = Option.map to_model

(* [Reg_width_const.t]: constructor for constructor. The model's [Poison] has
   the kind and not the string, which says where the poison came from. *)
let const_to_model (const : Reg_width_const.t) : M.const =
  match Reg_width_const.descr const with
  | Naked_immediate i -> Naked_immediate (z_of_target_int i)
  | Tagged_immediate i -> Tagged_immediate (z_of_target_int i)
  | Naked_float32 f -> Naked_float32 (z_of_float32 f)
  | Naked_float f -> Naked_float (z_of_float f)
  | Naked_int8 i -> Naked_int8 (z_of_int8 i)
  | Naked_int16 i -> Naked_int16 (z_of_int16 i)
  | Naked_int32 i -> Naked_int32 (z_of_int32 i)
  | Naked_int64 i -> Naked_int64 (z_of_int64 i)
  | Naked_nativeint i -> Naked_nativeint (z_of_nativeint i)
  | Naked_vec128 v -> Naked_vec128 (z_of_vec128 v)
  | Naked_vec256 v -> Naked_vec256 (z_of_vec256 v)
  | Naked_vec512 v -> Naked_vec512 (z_of_vec512 v)
  | Naked_mask m -> Naked_mask (z_of_mask m)
  | Null -> Null
  | Poison (kind, (_ : string)) -> Poison (kind_to_model kind)

(* The string of a [Poison] made from the model is this one. *)
let poison_from_the_model = "from the Rocq model"

let const_of_model (machine_width : machine_width) (const : M.const) :
    Reg_width_const.t =
  let module RWC = Reg_width_const in
  match const with
  | Naked_immediate i ->
    RWC.naked_immediate
      (in_range "an immediate" (target_int_of_z machine_width) i)
  | Tagged_immediate i ->
    RWC.tagged_immediate
      (in_range "an immediate" (target_int_of_z machine_width) i)
  | Naked_float32 f ->
    RWC.naked_float32 (in_range "the bits of a float32" float32_of_z f)
  | Naked_float f ->
    RWC.naked_float (in_range "the bits of a float" float_of_z f)
  | Naked_int8 i -> RWC.naked_int8 (in_range "an int8" int8_of_z i)
  | Naked_int16 i -> RWC.naked_int16 (in_range "an int16" int16_of_z i)
  | Naked_int32 i -> RWC.naked_int32 (in_range "an int32" int32_of_z i)
  | Naked_int64 i -> RWC.naked_int64 (in_range "an int64" int64_of_z i)
  | Naked_nativeint i ->
    RWC.naked_nativeint
      (in_range "a nativeint" (nativeint_of_z machine_width) i)
  | Naked_vec128 v ->
    RWC.naked_vec128 (in_range "the bits of a vec128" vec128_of_z v)
  | Naked_vec256 v ->
    RWC.naked_vec256 (in_range "the bits of a vec256" vec256_of_z v)
  | Naked_vec512 v ->
    RWC.naked_vec512 (in_range "the bits of a vec512" vec512_of_z v)
  | Naked_mask m -> RWC.naked_mask (in_range "the bits of a mask" mask_of_z m)
  | Null -> RWC.const_null
  | Poison kind -> RWC.const_poison (kind_of_model kind) poison_from_the_model

(* [Simple.t]. The model has no coercions (FlambdaModel.v): a [Simple] that has
   one that is not the identity is refused. *)
let simple_to_model (names : Model_names.t) (simple : Simple.t) : M.simple =
  let no_coercion coercion =
    if not (Coercion.is_id coercion)
    then not_in_the_model "the coercion of %a" Simple.print simple
  in
  Simple.pattern_match' simple
    ~var:(fun var ~coercion : M.simple ->
      no_coercion coercion;
      Var (variable_to_model names var))
    ~symbol:(fun symbol ~coercion : M.simple ->
      no_coercion coercion;
      Symbol (symbol_to_model names symbol))
    ~const:(fun const : M.simple -> Const (const_to_model const))

let simple_of_model (names : Model_names.t) (machine_width : machine_width)
    (simple : M.simple) : Simple.t =
  match simple with
  | Var var -> Simple.var (variable_of_model names var)
  | Symbol symbol -> Simple.symbol (symbol_of_model names symbol)
  | Const const -> Simple.const (const_of_model machine_width const)

(* ======================================================================== *)
(* What the grammar is built with                                           *)
(* ======================================================================== *)

(* [Or_unknown.t], [Or_bottom.t], [Or_unknown_or_bottom.t]: constructor for
   constructor. *)
let or_unknown_to_model to_model (x : _ Or_unknown.t) : _ MT.Or_unknown.t =
  match x with Known x -> Known (to_model x) | Unknown -> Unknown

let or_unknown_of_model of_model (x : _ MT.Or_unknown.t) : _ Or_unknown.t =
  match x with Known x -> Known (of_model x) | Unknown -> Unknown

let or_bottom_to_model to_model (x : _ Or_bottom.t) : _ MT.Or_bottom.t =
  match x with Ok x -> Ok (to_model x) | Bottom -> Bottom

let or_bottom_of_model of_model (x : _ MT.Or_bottom.t) : _ Or_bottom.t =
  match x with Ok x -> Ok (of_model x) | Bottom -> Bottom

let or_unknown_or_bottom_to_model to_model (x : _ Or_unknown_or_bottom.t) :
    _ MT.or_unknown_or_bottom =
  match x with Unknown -> Unknown | Ok x -> Ok (to_model x) | Bottom -> Bottom

let or_unknown_or_bottom_of_model of_model (x : _ MT.or_unknown_or_bottom) :
    _ Or_unknown_or_bottom.t =
  match x with Unknown -> Unknown | Ok x -> Ok (of_model x) | Bottom -> Bottom

(* [Alloc_mode.For_types.t]: constructor for constructor. The compiler cannot
   build [Local]: the type is private, [Alloc_mode.For_types.local ()] gives
   [Heap_or_local] (or [Heap], with stack allocation disabled), and nothing else
   gives it. With stack allocation disabled it cannot build [Heap_or_local]
   either. *)
let alloc_mode_to_model (mode : Alloc_mode.For_types.t) : MT.Alloc_mode.t =
  match mode with
  | Heap -> Heap
  | Local -> Local
  | Heap_or_local -> Heap_or_local

let alloc_mode_of_model (mode : MT.Alloc_mode.t) : Alloc_mode.For_types.t =
  match mode with
  | Heap -> Alloc_mode.For_types.heap
  | Local ->
    not_in_the_compiler
      "the allocation mode Local, which no function of the compiler makes"
  | Heap_or_local -> (
    match Alloc_mode.For_types.unknown () with
    | Heap_or_local as mode -> mode
    | Heap | Local ->
      not_in_the_compiler
        "the allocation mode Heap_or_local, with stack allocation disabled")

(* [Type_grammar.is_null]: constructor for constructor. *)
let is_null_to_model (names : Model_names.t) (is_null : TG.is_null) : MT.is_null
    =
  match is_null with
  | Not_null -> Not_null
  | Maybe_null { is_null } ->
    Maybe_null (option_to_model (variable_to_model names) is_null)

let is_null_of_model (names : Model_names.t) (is_null : MT.is_null) : TG.is_null
    =
  match is_null with
  | Not_null -> Not_null
  | Maybe_null is_null ->
    Maybe_null { is_null = Option.map (variable_of_model names) is_null }

(* [Type_grammar.Relation.descr]: constructor for constructor. *)
let relation_to_model (relation : TG.Relation.t) : MT.Relation.t =
  match TG.Relation.descr relation with
  | Is_null -> Is_null
  | Is_int -> Is_int
  | Get_tag -> Get_tag

let relation_of_model (relation : MT.Relation.t) : TG.Relation.t =
  match relation with
  | Is_null -> TG.Relation.is_null
  | Is_int -> TG.Relation.is_int
  | Get_tag -> TG.Relation.get_tag

(* [head_of_kind_naked_immediate], through [Head_of_kind_naked_immediate.descr]:
   the set of numbers is the list of its elements, and the map from relations to
   sets of names is the list of the pairs of a relation and a name in it. A
   relation that the map binds to the empty set leaves nothing in the list.

   Back, the names of one relation are gathered into one set. The compiler has
   no head whose set of numbers is empty ([from_descr] answers [Bottom]): it is
   refused. *)
let head_of_kind_naked_immediate_to_model (names : Model_names.t)
    (head : TG.head_of_kind_naked_immediate) : MT.head_of_kind_naked_immediate =
  let ({ naked_immediates; inverse_relations }
        : TG.Head_of_kind_naked_immediate.descr) =
    TG.Head_of_kind_naked_immediate.descr head
  in
  { naked_immediates =
      or_unknown_to_model
        (fun imms ->
          List.map z_of_target_int (Target_ocaml_int.Set.elements imms))
        naked_immediates;
    inverse_relations =
      List.concat_map
        (fun (relation, relation_names) ->
          List.map
            (fun name -> relation_to_model relation, name_to_model names name)
            (Name.Set.elements relation_names))
        (TG.Relation.Map.bindings inverse_relations)
  }

let head_of_kind_naked_immediate_of_model (names : Model_names.t)
    (machine_width : machine_width) (head : MT.head_of_kind_naked_immediate) :
    TG.head_of_kind_naked_immediate =
  let naked_immediates =
    or_unknown_of_model
      (fun imms ->
        Target_ocaml_int.Set.of_list
          (List.map
             (in_range "an immediate" (target_int_of_z machine_width))
             imms))
      head.naked_immediates
  in
  let inverse_relations =
    List.fold_left
      (fun relations (relation, name) ->
        TG.Relation.Map.update
          (relation_of_model relation)
          (fun relation_names ->
            Some
              (Name.Set.add (name_of_model names name)
                 (Option.value relation_names ~default:Name.Set.empty)))
          relations)
      TG.Relation.Map.empty head.inverse_relations
  in
  match
    TG.Head_of_kind_naked_immediate.from_descr
      { naked_immediates; inverse_relations }
  with
  | Ok head -> head
  | Bottom -> not_in_the_compiler "a head of naked immediates with no number"

(* [Flambda_kind.flat_suffix_element], which is the naked number kinds under
   other names: constructor for constructor. *)
let flat_suffix_element_to_model (element : K.flat_suffix_element) :
    MK.naked_number_kind =
  match element with
  | Naked_float -> Kind_naked_float
  | Naked_float32 -> Kind_naked_float32
  | Naked_int8 -> Kind_naked_int8
  | Naked_int16 -> Kind_naked_int16
  | Naked_int32 -> Kind_naked_int32
  | Naked_int64 -> Kind_naked_int64
  | Naked_nativeint -> Kind_naked_nativeint
  | Naked_immediate -> Kind_naked_immediate
  | Naked_vec128 -> Kind_naked_vec128
  | Naked_vec256 -> Kind_naked_vec256
  | Naked_vec512 -> Kind_naked_vec512
  | Naked_mask -> Kind_naked_mask

let flat_suffix_element_of_model (k : MK.naked_number_kind) :
    K.flat_suffix_element =
  match k with
  | Kind_naked_float -> Naked_float
  | Kind_naked_float32 -> Naked_float32
  | Kind_naked_int8 -> Naked_int8
  | Kind_naked_int16 -> Naked_int16
  | Kind_naked_int32 -> Naked_int32
  | Kind_naked_int64 -> Naked_int64
  | Kind_naked_nativeint -> Naked_nativeint
  | Kind_naked_immediate -> Naked_immediate
  | Kind_naked_vec128 -> Naked_vec128
  | Kind_naked_vec256 -> Naked_vec256
  | Kind_naked_vec512 -> Naked_vec512
  | Kind_naked_mask -> Naked_mask

(* [Flambda_kind.Block_shape.t]. Of a [Mixed_block_shape.t] the model keeps the
   size of the prefix of values and the flat suffix, which determine it
   ([from_prefix_size_and_suffix_elements]). *)
let block_shape_to_model (shape : K.Block_shape.t) : MT.block_shape =
  match shape with
  | Scannable Value_only -> Scannable Value_only
  | Scannable (Mixed_record shape) ->
    Scannable
      (Mixed_record
         ( N.nat_of_int (K.Mixed_block_shape.value_prefix_size shape),
           List.map flat_suffix_element_to_model
             (Array.to_list (K.Mixed_block_shape.flat_suffix shape)) ))
  | Float_record -> Float_record

let block_shape_of_model (shape : MT.block_shape) : K.Block_shape.t =
  match shape with
  | Scannable Value_only -> Scannable Value_only
  | Scannable (Mixed_record (value_prefix_size, flat_suffix)) ->
    Scannable
      (Mixed_record
         (K.Mixed_block_shape.from_prefix_size_and_suffix_elements
            (N.int_of_nat value_prefix_size)
            (List.map flat_suffix_element_of_model flat_suffix)))
  | Float_record -> Float_record

(* [Tag.t]: its number. Back, a number that is not a tag is refused. *)
let tag_to_model (tag : Tag.t) : N.z = N.z_of_int (Tag.to_int tag)

let tag_of_model (machine_width : machine_width) (tag : N.z) : Tag.t =
  match
    Option.bind
      (target_int_of_z machine_width tag)
      (Tag.create_from_targetint machine_width)
  with
  | Some tag -> tag
  | None -> not_in_the_compiler "the number %a is not a tag" N.print_z tag

(* [Type_grammar.Block_size.t], the size of a case of blocks, is abstract.
   [Row_like_for_blocks.get_singleton] gives the size of the one case of a
   structure with one tag and a size known exactly, so the size is read by
   putting it in such a structure. *)
let int_of_block_size (size : TG.Block_size.t) : int =
  let case =
    TG.Row_like_case.create ~maps_to:[||]
      ~index:
        (TG.Row_like_index.create
           ~domain:(TG.Row_like_index_domain.known size)
           ~shape:(K.Block_shape.Scannable Value_only))
      ~env_extension:TG.Env_extension.empty
  in
  let blocks =
    TG.Row_like_for_blocks.create_raw
      ~known_tags:(Tag.Map.singleton Tag.zero (Or_unknown.Known case))
      ~other_tags:Bottom ~alloc_mode:Alloc_mode.For_types.heap
  in
  match TG.Row_like_for_blocks.get_singleton blocks with
  | Some (_, _, size, _, _) -> Target_ocaml_int.to_int size
  | None ->
    not_in_the_model
      "the size of a case of blocks, which the compiler did not give back"

(* And [Row_like_for_blocks.create] makes a case whose size is the number of the
   fields it is given, so a size is made by giving it that many. *)
let block_size_of_int (machine_width : machine_width) (size : int) :
    TG.Block_size.t =
  let blocks =
    TG.Row_like_for_blocks.create ~machine_width
      ~shape:(K.Block_shape.Scannable Value_only)
      ~field_tys:(List.init size (fun _ -> TG.any_value))
      (Open Unknown) Alloc_mode.For_types.heap
  in
  match blocks.other_tags with
  | Ok { index = { domain = At_least size | Known size; _ }; _ } -> size
  | Bottom ->
    not_in_the_compiler "a size of %d, which the compiler did not make" size

(* [Type_grammar.row_like_index_domain]: constructor for constructor. *)
let index_domain_to_model to_model (domain : _ TG.row_like_index_domain) :
    _ MT.Row_like_index_domain.t =
  match domain with
  | Known index -> Known (to_model index)
  | At_least index -> At_least (to_model index)

let index_domain_of_model of_model (domain : _ MT.Row_like_index_domain.t) :
    _ TG.row_like_index_domain =
  match domain with
  | Known index -> TG.Row_like_index_domain.known (of_model index)
  | At_least index -> TG.Row_like_index_domain.at_least (of_model index)

(* [Set_of_closures_contents.t]: its two sets, as lists. *)
let contents_to_model (names : Model_names.t) (contents : Contents.t) :
    MT.set_of_closures_contents =
  { contents_closures =
      List.map
        (function_slot_to_model names)
        (Function_slot.Set.elements (Contents.closures contents));
    contents_value_slots =
      List.map
        (value_slot_to_model names)
        (Value_slot.Set.elements (Contents.value_slots contents))
  }

let contents_of_model (names : Model_names.t)
    (contents : MT.set_of_closures_contents) : Contents.t =
  Contents.create
    (Function_slot.Set.of_list
       (List.map (function_slot_of_model names) contents.contents_closures))
    (Value_slot.Set.of_list
       (List.map (value_slot_of_model names) contents.contents_value_slots))

(* [Type_descr.t], as [Type_descr.descr] shows it: constructor for constructor.
   The cache of free names is not seen. *)
let descr_to_model (names : Model_names.t) head_to_model (ty : _ TD.t) :
    _ MT.type_descr =
  match TD.descr ty with
  | Unknown -> Unknown
  | Bottom -> Bottom
  | Ok (No_alias head) -> Ok (No_alias (head_to_model head))
  | Ok (Equals simple) -> Ok (Equals (simple_to_model names simple))

(* A set of numbers of one kind, as the list of its elements; and back, where an
   empty set is no head of the compiler ([create_set] answers [Bottom]). *)
let numbers_of_model what of_z of_list create_set (numbers : N.z list) =
  match
    (create_set (of_list (List.map (in_range what of_z) numbers))
      : _ Or_bottom.t)
  with
  | Ok head -> head
  | Bottom -> not_in_the_compiler "a head with no number (%s)" what

(* A map of the compiler from the list that a map of the model is. The first
   binding of a key is the one kept, and a binding that an earlier one shadows
   is not looked at. *)
let map_of_model ~empty ~mem ~add key_of_model value_of_model bindings =
  List.fold_left
    (fun map (key, value) ->
      let key = key_of_model key in
      if mem key map then map else add key (value_of_model value) map)
    empty bindings

(* ======================================================================== *)
(* From the compiler to the model                                           *)
(* ======================================================================== *)

(* [Type_grammar.t]. The eleven constructors of the naked numbers other than the
   immediates are the one [Type_naked_number] with the kind; the head of a
   [Rec_info] type, an expression, is dropped. *)
let rec type_to_model (names : Model_names.t) (t : TG.t) : MT.flambda_type =
  let numbers kind elements z_of_number ty : MT.flambda_type =
    Type_naked_number
      ( kind,
        descr_to_model names
          (fun head -> List.map z_of_number (elements head))
          ty )
  in
  match t with
  | Value ty ->
    Type_value (descr_to_model names (head_of_kind_value_to_model names) ty)
  | Naked_immediate ty ->
    Type_naked_immediate
      (descr_to_model names (head_of_kind_naked_immediate_to_model names) ty)
  | Naked_float32 ty ->
    numbers Kind_naked_float32
      (fun (head : TG.head_of_kind_naked_float32) ->
        Float32.Set.elements (head :> Float32.Set.t))
      z_of_float32 ty
  | Naked_float ty ->
    numbers Kind_naked_float
      (fun (head : TG.head_of_kind_naked_float) ->
        Float.Set.elements (head :> Float.Set.t))
      z_of_float ty
  | Naked_int8 ty ->
    numbers Kind_naked_int8
      (fun (head : TG.head_of_kind_naked_int8) ->
        Int8.Set.elements (head :> Int8.Set.t))
      z_of_int8 ty
  | Naked_int16 ty ->
    numbers Kind_naked_int16
      (fun (head : TG.head_of_kind_naked_int16) ->
        Int16.Set.elements (head :> Int16.Set.t))
      z_of_int16 ty
  | Naked_int32 ty ->
    numbers Kind_naked_int32
      (fun (head : TG.head_of_kind_naked_int32) ->
        Int32.Set.elements (head :> Int32.Set.t))
      z_of_int32 ty
  | Naked_int64 ty ->
    numbers Kind_naked_int64
      (fun (head : TG.head_of_kind_naked_int64) ->
        Int64.Set.elements (head :> Int64.Set.t))
      z_of_int64 ty
  | Naked_nativeint ty ->
    numbers Kind_naked_nativeint
      (fun (head : TG.head_of_kind_naked_nativeint) ->
        Targetint_32_64.Set.elements (head :> Targetint_32_64.Set.t))
      z_of_nativeint ty
  | Naked_vec128 ty ->
    numbers Kind_naked_vec128
      (fun (head : TG.head_of_kind_naked_vec128) ->
        Vec128.Set.elements (head :> Vec128.Set.t))
      z_of_vec128 ty
  | Naked_vec256 ty ->
    numbers Kind_naked_vec256
      (fun (head : TG.head_of_kind_naked_vec256) ->
        Vec256.Set.elements (head :> Vec256.Set.t))
      z_of_vec256 ty
  | Naked_vec512 ty ->
    numbers Kind_naked_vec512
      (fun (head : TG.head_of_kind_naked_vec512) ->
        Vec512.Set.elements (head :> Vec512.Set.t))
      z_of_vec512 ty
  | Naked_mask ty ->
    numbers Kind_naked_mask
      (fun (head : TG.head_of_kind_naked_mask) ->
        Mask.Set.elements (head :> Mask.Set.t))
      z_of_mask ty
  | Rec_info ty ->
    Type_rec_info
      (descr_to_model names
         (fun (expr : Rec_info_expr.t) -> record_rec_info_expr expr)
         ty)
  | Region ty -> Type_region (descr_to_model names (fun () -> ()) ty)

(* [head_of_kind_value]: its two fields. *)
and head_of_kind_value_to_model (names : Model_names.t)
    ({ non_null; is_null } : TG.head_of_kind_value) : MT.head_of_kind_value =
  Head_of_kind_value
    ( or_unknown_or_bottom_to_model
        (head_of_kind_value_non_null_to_model names)
        non_null,
      is_null_to_model names is_null )

(* [head_of_kind_value_non_null]. The nine constructors of the boxed numbers are
   the one [Boxed_number] with the number that is boxed, one of the nine of
   [Flambda_kind.Boxable_number.t]. [Array] is not in the model. *)
and head_of_kind_value_non_null_to_model (names : Model_names.t)
    (head : TG.head_of_kind_value_non_null) : MT.head_of_kind_value_non_null =
  let boxed (number : MK.boxable_number) ty mode :
      MT.head_of_kind_value_non_null =
    Boxed_number (number, type_to_model names ty, alloc_mode_to_model mode)
  in
  match head with
  | Variant { is_int; immediates; get_tag; blocks; extensions; is_unique } ->
    Variant
      ( option_to_model (variable_to_model names) is_int,
        or_unknown_to_model (type_to_model names) immediates,
        option_to_model (variable_to_model names) get_tag,
        or_unknown_to_model (row_like_for_blocks_to_model names) blocks,
        variant_extensions_to_model names extensions,
        is_unique )
  | Mutable_block { alloc_mode } ->
    Mutable_block (alloc_mode_to_model alloc_mode)
  | Boxed_float32 (ty, alloc_mode) -> boxed Boxable_naked_float32 ty alloc_mode
  | Boxed_float (ty, alloc_mode) -> boxed Boxable_naked_float ty alloc_mode
  | Boxed_int32 (ty, alloc_mode) -> boxed Boxable_naked_int32 ty alloc_mode
  | Boxed_int64 (ty, alloc_mode) -> boxed Boxable_naked_int64 ty alloc_mode
  | Boxed_nativeint (ty, alloc_mode) ->
    boxed Boxable_naked_nativeint ty alloc_mode
  | Boxed_vec128 (ty, alloc_mode) -> boxed Boxable_naked_vec128 ty alloc_mode
  | Boxed_vec256 (ty, alloc_mode) -> boxed Boxable_naked_vec256 ty alloc_mode
  | Boxed_vec512 (ty, alloc_mode) -> boxed Boxable_naked_vec512 ty alloc_mode
  | Boxed_mask (ty, alloc_mode) -> boxed Boxable_naked_mask ty alloc_mode
  | Closures { by_function_slot; alloc_mode } ->
    Closures
      ( row_like_for_closures_to_model names by_function_slot,
        alloc_mode_to_model alloc_mode )
  | String strs ->
    (* A string is the list of its bytes. *)
    String
      (List.map
         (fun str ->
           List.init (String.length str) (fun i ->
               N.z_of_int (Char.code str.[i])))
         (String_info.Set.elements strs))
  | Array { element_kind = _; length = _; contents = _; alloc_mode = _ } ->
    not_in_the_model "a type of arrays"

(* [row_like_block_case]: its three fields. The product of the types of the
   fields, an array, is their list; the size is a number. *)
and row_like_block_case_to_model (names : Model_names.t)
    ({ maps_to; index = { domain; shape }; env_extension } :
      TG.row_like_block_case) : MT.row_like_block_case =
  { case_maps_to = List.map (type_to_model names) (Array.to_list maps_to);
    case_index =
      { index_domain =
          index_domain_to_model
            (fun size -> N.nat_of_int (int_of_block_size size))
            domain;
        index_shape = block_shape_to_model shape
      };
    case_env_extension = env_extension_to_model names env_extension
  }

(* [row_like_for_blocks]: its three fields; the map from tags is the list of its
   bindings. *)
and row_like_for_blocks_to_model (names : Model_names.t)
    ({ known_tags; other_tags; alloc_mode } : TG.row_like_for_blocks) :
    MT.row_like_for_blocks =
  Row_like_for_blocks
    ( List.map
        (fun (tag, case) ->
          ( tag_to_model tag,
            or_unknown_to_model (row_like_block_case_to_model names) case ))
        (Tag.Map.bindings known_tags),
      or_bottom_to_model (row_like_block_case_to_model names) other_tags,
      alloc_mode_to_model alloc_mode )

(* [row_like_for_closures]: the map from function slots is the list of its
   bindings; the shape of the index of a case is [unit]. *)
and row_like_for_closures_to_model (names : Model_names.t)
    ({ known_closures } : TG.row_like_for_closures) : MT.row_like_for_closures =
  Row_like_for_closures
    (List.map
       (fun ( slot,
              ({ maps_to; index = { domain; shape = () }; env_extension } :
                (_, unit, TG.closures_entry) TG.row_like_case) ) :
            (_ * (_, _, _, _) MT.row_like_case) ->
         ( function_slot_to_model names slot,
           { case_maps_to = closures_entry_to_model names maps_to;
             case_index =
               { index_domain =
                   index_domain_to_model (contents_to_model names) domain;
                 index_shape = ()
               };
             case_env_extension = env_extension_to_model names env_extension
           } ))
       (Function_slot.Map.bindings known_closures))

(* [closures_entry]: its three fields, each map the list of its bindings (the
   two products are read with the accessors of [Closures_entry]). A function
   type is its code id: the model leaves its [rec_info] out, and it is dropped
   here. *)
and closures_entry_to_model (names : Model_names.t) (entry : TG.closures_entry)
    : MT.closures_entry =
  Closures_entry
    ( List.map
        (fun (slot, function_type) ->
          ( function_slot_to_model names slot,
            or_unknown_to_model
              (fun (function_type : TG.function_type) : MT.function_type ->
                let code_id = TG.Function_type.code_id function_type in
                record_rec_info code_id
                  (TG.Function_type.rec_info function_type);
                { function_type_code_id = code_id_to_model names code_id })
              function_type ))
        (Function_slot.Map.bindings entry.function_types),
      List.map
        (fun (slot, ty) ->
          function_slot_to_model names slot, type_to_model names ty)
        (Function_slot.Map.bindings
           (TG.Closures_entry.function_slot_types entry)),
      List.map
        (fun (slot, ty) ->
          value_slot_to_model names slot, type_to_model names ty)
        (Value_slot.Map.bindings (TG.Closures_entry.value_slot_types entry)) )

(* [env_extension]: the map from names is the list of its bindings. *)
and env_extension_to_model (names : Model_names.t)
    ({ equations } : TG.env_extension) : MT.env_extension =
  Env_extension
    (List.map
       (fun (name, ty) -> name_to_model names name, type_to_model names ty)
       (Name.Map.bindings equations))

(* [variant_extensions]: constructor for constructor. *)
and variant_extensions_to_model (names : Model_names.t)
    (extensions : TG.variant_extensions) : MT.variant_extensions =
  match extensions with
  | No_extensions -> No_extensions
  | Ext { when_immediate; when_block } ->
    Ext
      ( env_extension_to_model names when_immediate,
        env_extension_to_model names when_block )

(* ======================================================================== *)
(* From the model to the compiler                                           *)
(* ======================================================================== *)

(* The types of the compiler are private: they are built with the functions of
   [Type_grammar] that check nothing (the raw constructors), but for the
   products of function slots, whose creation function checks that the types are
   of kind value; a product of the model that is not is refused. *)

(* A type of the model that is unknown, bottom or an alias, at a kind. *)
let type_of_model_descr (names : Model_names.t) (machine_width : machine_width)
    (kind : K.t) create_from_head (ty : _ MT.type_descr) : TG.t =
  match ty with
  | Unknown -> MTC.unknown kind
  | Bottom -> MTC.bottom kind
  | Ok (Equals simple) ->
    TG.alias_type_of kind (simple_of_model names machine_width simple)
  | Ok (No_alias head) -> create_from_head head

let rec type_of_model (names : Model_names.t) (machine_width : machine_width)
    (t : MT.flambda_type) : TG.t =
  let descr kind create_from_head ty =
    type_of_model_descr names machine_width kind create_from_head ty
  in
  match t with
  | Type_value ty ->
    descr K.value
      (fun head ->
        TG.create_from_head_value
          (head_of_kind_value_of_model names machine_width head))
      ty
  | Type_naked_immediate ty ->
    descr K.naked_immediate
      (fun head ->
        TG.create_from_head_naked_immediate
          (head_of_kind_naked_immediate_of_model names machine_width head))
      ty
  | Type_naked_number (Kind_naked_immediate, _) ->
    (* The header of FlambdaTypeGrammar.v: this "corresponds to no type of the
       compiler". *)
    not_in_the_compiler
      "a type of naked numbers of kind naked immediate (the type of naked \
       immediates is another constructor)"
  | Type_naked_number (Kind_naked_float32, ty) ->
    descr K.naked_float32
      (fun numbers ->
        TG.create_from_head_naked_float32
          (numbers_of_model "the bits of a float32" float32_of_z
             Float32.Set.of_list TG.Head_of_kind_naked_float32.create_set
             numbers))
      ty
  | Type_naked_number (Kind_naked_float, ty) ->
    descr K.naked_float
      (fun numbers ->
        TG.create_from_head_naked_float
          (numbers_of_model "the bits of a float" float_of_z Float.Set.of_list
             TG.Head_of_kind_naked_float.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_int8, ty) ->
    descr K.naked_int8
      (fun numbers ->
        TG.create_from_head_naked_int8
          (numbers_of_model "an int8" int8_of_z Int8.Set.of_list
             TG.Head_of_kind_naked_int8.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_int16, ty) ->
    descr K.naked_int16
      (fun numbers ->
        TG.create_from_head_naked_int16
          (numbers_of_model "an int16" int16_of_z Int16.Set.of_list
             TG.Head_of_kind_naked_int16.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_int32, ty) ->
    descr K.naked_int32
      (fun numbers ->
        TG.create_from_head_naked_int32
          (numbers_of_model "an int32" int32_of_z Int32.Set.of_list
             TG.Head_of_kind_naked_int32.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_int64, ty) ->
    descr K.naked_int64
      (fun numbers ->
        TG.create_from_head_naked_int64
          (numbers_of_model "an int64" int64_of_z Int64.Set.of_list
             TG.Head_of_kind_naked_int64.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_nativeint, ty) ->
    descr K.naked_nativeint
      (fun numbers ->
        TG.create_from_head_naked_nativeint
          (numbers_of_model "a nativeint"
             (nativeint_of_z machine_width)
             Targetint_32_64.Set.of_list
             TG.Head_of_kind_naked_nativeint.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_vec128, ty) ->
    descr K.naked_vec128
      (fun numbers ->
        TG.create_from_head_naked_vec128
          (let module Head = TG.Head_of_kind_naked_vec128 in
           numbers_of_model "the bits of a vec128" vec128_of_z
             Vec128.Set.of_list Head.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_vec256, ty) ->
    descr K.naked_vec256
      (fun numbers ->
        TG.create_from_head_naked_vec256
          (let module Head = TG.Head_of_kind_naked_vec256 in
           numbers_of_model "the bits of a vec256" vec256_of_z
             Vec256.Set.of_list Head.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_vec512, ty) ->
    descr K.naked_vec512
      (fun numbers ->
        TG.create_from_head_naked_vec512
          (let module Head = TG.Head_of_kind_naked_vec512 in
           numbers_of_model "the bits of a vec512" vec512_of_z
             Vec512.Set.of_list Head.create_set numbers))
      ty
  | Type_naked_number (Kind_naked_mask, ty) ->
    descr K.naked_mask
      (fun numbers ->
        TG.create_from_head_naked_mask
          (numbers_of_model "the bits of a mask" mask_of_z Mask.Set.of_list
             TG.Head_of_kind_naked_mask.create_set numbers))
      ty
  | Type_rec_info ty ->
    descr K.rec_info
      (fun () ->
        not_in_the_compiler
          "a Rec_info type given by its head: the model has not got the \
           expression")
      ty
  | Type_region ty -> descr K.region TG.create_from_head_region ty

and head_of_kind_value_of_model (names : Model_names.t)
    (machine_width : machine_width)
    (Head_of_kind_value (non_null, is_null) : MT.head_of_kind_value) :
    TG.head_of_kind_value =
  { non_null =
      or_unknown_or_bottom_of_model
        (head_of_kind_value_non_null_of_model names machine_width)
        non_null;
    is_null = is_null_of_model names is_null
  }

and head_of_kind_value_non_null_of_model (names : Model_names.t)
    (machine_width : machine_width) (head : MT.head_of_kind_value_non_null) :
    TG.head_of_kind_value_non_null =
  let module H = TG.Head_of_kind_value_non_null in
  let ty_of_model = type_of_model names machine_width in
  match head with
  | Variant (is_int, immediates, get_tag, blocks, extensions, is_unique) ->
    H.create_variant ~is_unique
      ~blocks:
        (or_unknown_of_model
           (row_like_for_blocks_of_model names machine_width)
           blocks)
      ~immediates:(or_unknown_of_model ty_of_model immediates)
      ~extensions:(variant_extensions_of_model names machine_width extensions)
      ~is_int:(Option.map (variable_of_model names) is_int)
      ~get_tag:(Option.map (variable_of_model names) get_tag)
  | Mutable_block alloc_mode ->
    H.create_mutable_block (alloc_mode_of_model alloc_mode)
  | Boxed_number (number, contents, alloc_mode) -> (
    let contents = ty_of_model contents in
    let alloc_mode = alloc_mode_of_model alloc_mode in
    match number with
    | Boxable_naked_float32 -> H.create_boxed_float32 contents alloc_mode
    | Boxable_naked_float -> H.create_boxed_float contents alloc_mode
    | Boxable_naked_int32 -> H.create_boxed_int32 contents alloc_mode
    | Boxable_naked_int64 -> H.create_boxed_int64 contents alloc_mode
    | Boxable_naked_nativeint -> H.create_boxed_nativeint contents alloc_mode
    | Boxable_naked_vec128 -> H.create_boxed_vec128 contents alloc_mode
    | Boxable_naked_vec256 -> H.create_boxed_vec256 contents alloc_mode
    | Boxable_naked_vec512 -> H.create_boxed_vec512 contents alloc_mode
    | Boxable_naked_mask -> H.create_boxed_mask contents alloc_mode)
  | Closures (by_function_slot, alloc_mode) ->
    H.create_closures
      (row_like_for_closures_of_model names machine_width by_function_slot)
      (alloc_mode_of_model alloc_mode)
  | String strs ->
    H.create_string
      (String_info.Set.of_list
         (List.map
            (fun bytes ->
              String.of_seq
                (Seq.map
                   (fun byte ->
                     match N.int_of_z byte with
                     | Some byte when byte >= 0 && byte <= 255 -> Char.chr byte
                     | Some _ | None ->
                       not_in_the_compiler "the number %a is not a byte"
                         N.print_z byte)
                   (List.to_seq bytes)))
            strs))

and row_like_block_case_of_model (names : Model_names.t)
    (machine_width : machine_width)
    ({ case_maps_to; case_index; case_env_extension } : MT.row_like_block_case)
    : TG.row_like_block_case =
  TG.Row_like_case.create
    ~maps_to:
      (Array.of_list
         (List.map (type_of_model names machine_width) case_maps_to))
    ~index:
      (TG.Row_like_index.create
         ~domain:
           (index_domain_of_model
              (fun size -> block_size_of_int machine_width (N.int_of_nat size))
              case_index.index_domain)
         ~shape:(block_shape_of_model case_index.index_shape))
    ~env_extension:
      (env_extension_of_model names machine_width case_env_extension)

and row_like_for_blocks_of_model (names : Model_names.t)
    (machine_width : machine_width)
    (Row_like_for_blocks (known_tags, other_tags, alloc_mode) :
      MT.row_like_for_blocks) : TG.row_like_for_blocks =
  TG.Row_like_for_blocks.create_raw
    ~known_tags:
      (map_of_model ~empty:Tag.Map.empty ~mem:Tag.Map.mem ~add:Tag.Map.add
         (tag_of_model machine_width)
         (or_unknown_of_model
            (row_like_block_case_of_model names machine_width))
         known_tags)
    ~other_tags:
      (or_bottom_of_model
         (row_like_block_case_of_model names machine_width)
         other_tags)
    ~alloc_mode:(alloc_mode_of_model alloc_mode)

and row_like_for_closures_of_model (names : Model_names.t)
    (machine_width : machine_width)
    (Row_like_for_closures known_closures : MT.row_like_for_closures) :
    TG.row_like_for_closures =
  TG.Row_like_for_closures.create_raw
    ~known_closures:
      (map_of_model ~empty:Function_slot.Map.empty ~mem:Function_slot.Map.mem
         ~add:Function_slot.Map.add
         (function_slot_of_model names)
         (fun ({ case_maps_to; case_index; case_env_extension } :
                (_, unit, _, _) MT.row_like_case) ->
           TG.Row_like_case.create
             ~maps_to:(closures_entry_of_model names machine_width case_maps_to)
             ~index:
               (TG.Row_like_index.create
                  ~domain:
                    (index_domain_of_model (contents_of_model names)
                       case_index.index_domain)
                  ~shape:case_index.index_shape)
             ~env_extension:
               (env_extension_of_model names machine_width case_env_extension))
         known_closures)

(* A function type of the model is a code id; the compiler's gets the unknown
   [rec_info], of which the model says nothing ([rec_info_of_code], which is
   another function only during [type_of_model_with_rec_infos]). *)
and closures_entry_of_model (names : Model_names.t)
    (machine_width : machine_width)
    (Closures_entry (function_types, closure_types, value_slot_types) :
      MT.closures_entry) : TG.closures_entry =
  let ty_of_model = type_of_model names machine_width in
  let closure_types =
    map_of_model ~empty:Function_slot.Map.empty ~mem:Function_slot.Map.mem
      ~add:Function_slot.Map.add
      (function_slot_of_model names)
      (fun ty ->
        let ty = ty_of_model ty in
        (* [Product.Function_slot_indexed.create] is a fatal error on a type of
           another kind. *)
        if not (K.equal (TG.kind ty) K.value)
        then
          not_in_the_compiler
            "a closure type that is not of kind value, in a closures entry";
        ty)
      closure_types
  in
  TG.Closures_entry.create
    ~function_types:
      (map_of_model ~empty:Function_slot.Map.empty ~mem:Function_slot.Map.mem
         ~add:Function_slot.Map.add
         (function_slot_of_model names)
         (or_unknown_of_model
            (fun ({ function_type_code_id = code_id } : MT.function_type) ->
              let code_id = code_id_of_model names code_id in
              TG.Function_type.create code_id
                ~rec_info:(!rec_info_of_code code_id)))
         function_types)
    ~closure_types:(TG.Product.Function_slot_indexed.create closure_types)
    ~value_slot_types:
      (TG.Product.Value_slot_indexed.create
         (map_of_model ~empty:Value_slot.Map.empty ~mem:Value_slot.Map.mem
            ~add:Value_slot.Map.add
            (value_slot_of_model names)
            ty_of_model value_slot_types))

and env_extension_of_model (names : Model_names.t)
    (machine_width : machine_width) (Env_extension equations : MT.env_extension)
    : TG.env_extension =
  TG.Env_extension.create
    ~equations:
      (map_of_model ~empty:Name.Map.empty ~mem:Name.Map.mem ~add:Name.Map.add
         (name_of_model names)
         (type_of_model names machine_width)
         equations)

and variant_extensions_of_model (names : Model_names.t)
    (machine_width : machine_width) (extensions : MT.variant_extensions) :
    TG.variant_extensions =
  match extensions with
  | No_extensions -> No_extensions
  | Ext (when_immediate, when_block) ->
    Ext
      { when_immediate =
          env_extension_of_model names machine_width when_immediate;
        when_block = env_extension_of_model names machine_width when_block
      }

(* ======================================================================== *)
(* The two conversions, with their refusals as results                      *)
(* ======================================================================== *)

type 'a converted =
  | Converted of 'a
  | Refused of string

let to_model (names : Model_names.t) (t : TG.t) : MT.flambda_type converted =
  match type_to_model names t with
  | t -> Converted t
  | exception Not_in_the_model what -> Refused what

let of_model (names : Model_names.t) (machine_width : machine_width)
    (t : MT.flambda_type) : TG.t converted =
  match type_of_model names machine_width t with
  | t -> Converted t
  | exception Not_in_the_compiler what -> Refused what

(* ======================================================================== *)
(* The conversions with what is dropped of [Rec_info]                       *)
(* ======================================================================== *)

(* [type_to_model], with what it dropped of the [rec_info] of function types and
   of the expressions of [Rec_info] heads. *)
let type_to_model_with_what_is_dropped (names : Model_names.t) (t : TG.t) :
    MT.flambda_type * dropped =
  let before = !recording in
  recording := Some { rec_infos = []; variables = [] };
  Fun.protect
    ~finally:(fun () -> recording := before)
    (fun () ->
      let m = type_to_model names t in
      match !recording with
      | Some { rec_infos; variables } ->
        m, { rec_infos = List.rev rec_infos; variables }
      | None -> assert false)

(* [type_of_model], where a function type whose code id is in the list gets the
   [rec_info] given with it (the first, if there are several), and not the
   unknown one. *)
let type_of_model_with_rec_infos (names : Model_names.t)
    (machine_width : machine_width) ~(rec_infos : (Code_id.t * TG.t) list)
    (m : MT.flambda_type) : TG.t =
  let before = !rec_info_of_code in
  (rec_info_of_code
     := fun code_id ->
          match
            List.find_opt
              (fun (code_id', (_ : TG.t)) -> Code_id.equal code_id code_id')
              rec_infos
          with
          | Some ((_ : Code_id.t), rec_info) -> rec_info
          | None -> TG.any_rec_info);
  Fun.protect
    ~finally:(fun () -> rec_info_of_code := before)
    (fun () -> type_of_model names machine_width m)
