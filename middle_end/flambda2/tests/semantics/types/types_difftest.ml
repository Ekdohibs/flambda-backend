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

(* The types library of the compiler against its Rocq model.

   Each function that the model has (FlambdaTypeGrammar.v, extracted) is run on
   generated inputs beside the function of the compiler that it transcribes, and
   the two answers are compared through the converter (type_converter.ml). What
   is printed is, for each comparison, how many cases were run, how many agreed,
   how many could not be run and why, and every disagreement with the seed of
   its case and a shrunk input.

   Everything is seeded and nothing depends on time or on the machine: the
   output is the same on every run, and is kept in types_difftest.expected. The
   number of cases of a comparison is its [~count] below.

   The functions of the compiler are those of the modules of its types library,
   which the build names [Flambda2_types__...]; the library does not export
   them. *)

open Flambda2_identifiers
open Flambda2_kinds
open Flambda2_numbers
open Flambda2_term_basics
open Flambda2_bound_identifiers
open Flambda2_lattices
open Flambda2_nominal
open Type_comparisons
module K = Flambda_kind
module TG = Flambda2_types__Type_grammar
module TE = Flambda2_types__Typing_env
module MTC = Flambda2_types__More_type_creators
module Binary = Flambda2_types__Meet_and_join
module N_way = Flambda2_types__Meet_and_n_way_join
module Binary_join = Flambda2_types__Join_levels_old
module N_way_join = Flambda2_types__Join_env
module ME = Flambda2_types__Meet_env
module M = Flambda2_rocq_semantics.FlambdaModel
module MK = Flambda2_rocq_semantics.FlambdaKindModel
module MT = Flambda2_rocq_semantics.FlambdaTypeGrammar
module V = Flambda2_rocq_semantics.FlambdaValueSpec
module Meaning = Flambda2_rocq_semantics.FlambdaTypeMeaningBoolSpec
module Primitive_spec = Flambda2_rocq_semantics.FlambdaPrimitiveSpec
module Type_equal = Flambda2_rocq_semantics.FlambdaTypeEqualSpec
module Z = Flambda2_rocq_library.BinInt.Z
module Converter = Flambda2_rocq_types_converter.Type_converter
module Model_names = Flambda2_rocq_types_converter.Model_names
module Model_type = Flambda2_rocq_types_converter.Model_type
module N = Flambda2_rocq_types_converter.Model_numbers
module G = Type_generator
module Random = Minicheck.Splittable_random

(* How a join decides that the types it is given are one type, which it then
   keeps instead of joining: the compiler's flag [-flambda2-join-sharing]. With
   the default, physical equality, the result of a join depends on how its
   arguments are shared in memory (T7-7). The model has no physical equality:
   the tests run with the structural setting, and the comparisons of shared and
   unshared arguments are run with the default as well. *)
let with_join_sharing (setting : Oxcaml_flags.join_sharing) (f : unit -> 'a) :
    'a =
  let before = !Oxcaml_flags.Flambda2.join_sharing in
  Oxcaml_flags.Flambda2.join_sharing := Oxcaml_flags.Set setting;
  Fun.protect f ~finally:(fun () ->
      Oxcaml_flags.Flambda2.join_sharing := before)

(* ======================================================================== *)
(* The names of the tests                                                   *)
(* ======================================================================== *)

(* A few names of each sort, made once, numbered in this order. The variables
   and the symbols are defined in a typing environment, with nothing known of
   them, for the functions of the compiler that want one. *)
type names =
  { numbering : Model_names.t;
    pool : G.pool;
    env : TE.t;
    (* The variables and the symbols that [env] defines, with their kinds. *)
    defined : (Name.t * K.t) list;
    (* A parameter of a continuation of each kind, for the joins. They are not
       in the pool: no generated type mentions them. *)
    parameters : (MK.flambda_kind * Variable.t) list;
    (* A variable of kind value that only one use of a continuation binds, for
       the chosen joins. It is numbered after the variables of the tests: what a
       join says of it is read like what it says of a variable of its own. *)
    local : Variable.t;
    (* A naked immediate that a join may have its source define for types only,
       for the chosen joins. It is numbered after [local], and read like it. *)
    for_types_only : Variable.t;
    (* The variables of the tests are numbered from 1 to this. *)
    variables_of_the_tests : int;
    (* The variables numbered up to this are printed by their numbers: those of
       the tests, [local] and [for_types_only]. *)
    variables_with_a_name : int
  }

let create_names () : names =
  let numbering = Model_names.create () in
  let unit = Current_unit.get_cu_exn () in
  let variables kind names =
    List.map (fun name -> Variable.create name kind) names
  in
  let value_variables = variables K.value ["x"; "y"; "z"] in
  let naked_immediate_variables = variables K.naked_immediate ["i"; "j"] in
  let naked_float_variables = variables K.naked_float ["f"] in
  let region_variables = variables K.region ["r"] in
  let rec_info_variables = variables K.rec_info ["d"] in
  let symbols =
    List.map
      (fun name -> Symbol.create unit (Linkage_name.of_string name))
      ["s"; "t"]
  in
  let code_ids =
    List.map
      (fun name -> Code_id.create ~name ~debug:Debuginfo.none unit)
      ["c"; "c'"]
  in
  let function_slots =
    List.map (fun name -> Function_slot.create unit ~name ~size:2) ["g"; "h"]
  in
  let value_slots =
    List.map
      (fun name ->
        Value_slot.create unit ~name ~is_always_immediate:false K.value)
      ["w"; "w'"]
  in
  let number (sort : _ Model_names.sort) = List.map sort.number in
  let all_variables =
    value_variables @ naked_immediate_variables @ naked_float_variables
    @ region_variables @ rec_info_variables
  in
  let env =
    List.fold_left
      (fun env var ->
        TE.add_definition env
          (Bound_name.create_var
             (Bound_var.create var Flambda_debug_uid.none Name_mode.normal))
          (Variable.kind var))
      (TE.create ~machine_width ~resolver:(fun _ -> None))
      all_variables
  in
  let env =
    List.fold_left
      (fun env symbol ->
        TE.add_definition env (Bound_name.create_symbol symbol) K.value)
      env symbols
  in
  let defined =
    List.map (fun var -> Name.var var, Variable.kind var) all_variables
    @ List.map (fun symbol -> Name.symbol symbol, K.value) symbols
  in
  let parameters =
    List.map
      (fun kind -> kind, Variable.create "p" (Converter.kind_of_model kind))
      G.all_kinds
  in
  let local = Variable.create "u" K.value in
  let for_types_only = Variable.create "n" K.naked_immediate in
  let value_variables = number numbering.variables value_variables in
  let naked_immediate_variables =
    number numbering.variables naked_immediate_variables
  in
  let naked_float_variables =
    number numbering.variables naked_float_variables
  in
  let region_variables = number numbering.variables region_variables in
  let rec_info_variables = number numbering.variables rec_info_variables in
  (* The parameters are numbered after the variables of the pool. *)
  List.iter
    (fun (_, parameter) ->
      ignore (numbering.variables.number parameter : N.positive))
    parameters;
  ignore (numbering.variables.number local : N.positive);
  ignore (numbering.variables.number for_types_only : N.positive);
  { numbering;
    pool =
      { value_variables;
        naked_immediate_variables;
        naked_float_variables;
        region_variables;
        rec_info_variables;
        symbols = number numbering.symbols symbols;
        code_ids = number numbering.code_ids code_ids;
        function_slots = number numbering.function_slots function_slots;
        value_slots = number numbering.value_slots value_slots;
        unknown_name = N.positive_of_int 1000
      };
    env;
    defined;
    parameters;
    local;
    for_types_only;
    variables_of_the_tests = List.length all_variables + List.length parameters;
    variables_with_a_name =
      List.length all_variables + List.length parameters + 2
  }

let parameter (names : names) (kind : MK.flambda_kind) : Variable.t =
  List.assoc kind names.parameters

(* ======================================================================== *)
(* The comparisons of one type                                              *)
(* ======================================================================== *)

(* The comparisons themselves are in type_comparisons.ml, which the other tools
   of this directory use too. *)

let to_compiler (names : names) (t : MT.flambda_type) : TG.t =
  Converter.type_of_model names.numbering machine_width t

let to_model (names : names) (t : TG.t) : MT.flambda_type =
  Converter.type_to_model names.numbering t

let both (names : names) (m : MT.flambda_type) : both =
  { model = m; compiler = to_compiler names m }

let comparisons_of_one_type (names : names) : both comparison list =
  Type_comparisons.comparisons_of_one_type names.numbering

let well_formed_types (names : names) : both comparison list =
  Type_comparisons.well_formed_types names.numbering

let round_trip_from_the_compiler (names : names) : both comparison list =
  Type_comparisons.round_trip_from_the_compiler names.numbering names.env

(* ======================================================================== *)
(* The creation functions                                                   *)
(* ======================================================================== *)

(* A call of a creation function that the model has, with its arguments as the
   model has them. It is run in the model as it is, and in the compiler on the
   arguments converted; the two types are then compared. *)
type slots_and_types =
  { this_function_slot : M.function_slot;
    function_types : (M.function_slot * MT.function_type MT.Or_unknown.t) list;
    closure_types : (M.function_slot * MT.flambda_type) list;
    value_slot_types : (M.value_slot * MT.flambda_type) list;
    closure_alloc_mode : MT.Alloc_mode.t
  }

type block =
  { tag : N.z;
    shape : MT.block_shape;
    fields : MT.flambda_type list;
    block_alloc_mode : MT.Alloc_mode.t
  }

type call =
  | Unknown of MK.flambda_kind
  | Bottom of MK.flambda_kind
  | Alias_type_of of MK.flambda_kind * M.simple
  | Type_for_const of M.const
  | Null
  | Any_non_null_value
  | This_tagged_immediate of N.z
  | This_naked_immediate of N.z
  | These_naked_immediates of N.z list
  | These_tagged_immediates of N.z list
  | Any_tagged_immediate
  | Any_block
  | Tag_immediate of MT.flambda_type
  | Box_number of MK.boxable_number * MT.flambda_type * MT.Alloc_mode.t
  | Mutable_block of MT.Alloc_mode.t
  | This_immutable_string of N.z list
  | Create_variant of
      bool
      * MT.flambda_type MT.Or_unknown.t
      * MT.row_like_for_blocks MT.Or_unknown.t
      * MT.variant_extensions
  | Create_closures of MT.Alloc_mode.t * MT.row_like_for_closures
  | Blocks_with_these_tags of N.z list * MT.Alloc_mode.t
  | Immutable_block of bool * block
  | Row_like_create_exactly of block
  | Row_like_create_at_least of block
  | Row_like_create_at_least_unknown_tag of block
  | Is_int_for_scrutinee of MT.name
  | Get_tag_for_block of MT.name
  | Is_null_for_scrutinee of MT.name
  | Exactly_this_closure of slots_and_types
  | At_least_this_closure of slots_and_types
  | Unknown_like of MT.flambda_type
  | Bottom_like of MT.flambda_type
  | Unknown_from_shape of MT.block_shape * int
  | Unknown_with_subkind of MK.flambda_kind
  | Kind_for_const of M.const
  | These_naked_numbers of MK.naked_number_kind * N.z list
  | Any_tagged_bool
  | Any_naked_bool
  | Any_tagged_immediate_or_null
  | Tagged_immediate_alias_to of M.variable
  | Boxed_number_alias_to of MK.boxable_number * M.variable * MT.Alloc_mode.t
  | This_boxed_number of MK.boxable_number * N.z * MT.Alloc_mode.t
  | These_boxed_numbers of MK.boxable_number * N.z list * MT.Alloc_mode.t
  | Any_boxed_number of MK.boxable_number
  | Row_like_create of block * MT.Open_or_closed.t
      (** The tag is that of the second argument, and not the block's. *)
  | Immutable_block_with_size_at_least of
      N.z MT.Or_unknown.t * int * MT.block_shape * M.variable
  | Closure_with_at_least_these_function_slots of
      M.function_slot * (M.function_slot * M.simple) list
  | Static_closure_with_this_code of
      M.function_slot * M.symbol option * M.code_id
  | Closure_with_at_least_these_value_slots of
      M.function_slot * (M.value_slot * (M.variable * MK.flambda_kind)) list
  | Closure_with_at_least_this_value_slot of
      M.function_slot * M.value_slot * M.variable * MK.flambda_kind

let name_of_call (call : call) =
  match call with
  | Unknown _ -> "unknown"
  | Bottom _ -> "bottom"
  | Alias_type_of _ -> "alias_type_of"
  | Type_for_const _ -> "type_for_const"
  | Null -> "null"
  | Any_non_null_value -> "any_non_null_value"
  | This_tagged_immediate _ -> "this_tagged_immediate"
  | This_naked_immediate _ -> "this_naked_immediate"
  | These_naked_immediates _ -> "these_naked_immediates"
  | These_tagged_immediates _ -> "these_tagged_immediates"
  | Any_tagged_immediate -> "any_tagged_immediate"
  | Any_block -> "any_block"
  | Tag_immediate _ -> "tag_immediate"
  | Box_number _ -> "box_number"
  | Mutable_block _ -> "mutable_block"
  | This_immutable_string _ -> "this_immutable_string"
  | Create_variant _ -> "create_variant"
  | Create_closures _ -> "create_closures"
  | Blocks_with_these_tags _ -> "blocks_with_these_tags"
  | Immutable_block _ -> "immutable_block"
  | Row_like_create_exactly _ -> "row_like_for_blocks_create_exactly"
  | Row_like_create_at_least _ -> "row_like_for_blocks_create_at_least"
  | Row_like_create_at_least_unknown_tag _ ->
    "row_like_for_blocks_create_at_least_unknown_tag"
  | Is_int_for_scrutinee _ -> "is_int_for_scrutinee"
  | Get_tag_for_block _ -> "get_tag_for_block"
  | Is_null_for_scrutinee _ -> "is_null_for_scrutinee"
  | Exactly_this_closure _ -> "exactly_this_closure"
  | At_least_this_closure _ -> "at_least_this_closure"
  | Unknown_like _ -> "unknown_like"
  | Bottom_like _ -> "bottom_like"
  | Unknown_from_shape _ -> "unknown_from_shape"
  | Unknown_with_subkind _ -> "unknown_with_subkind, of a kind"
  | Kind_for_const _ -> "kind_for_const, as the unknown type of the kind"
  | These_naked_numbers _ -> "these_naked_numbers"
  | Any_tagged_bool -> "any_tagged_bool"
  | Any_naked_bool -> "any_naked_bool"
  | Any_tagged_immediate_or_null -> "any_tagged_immediate_or_null"
  | Tagged_immediate_alias_to _ -> "tagged_immediate_alias_to"
  | Boxed_number_alias_to _ -> "boxed_number_alias_to"
  | This_boxed_number _ -> "this_boxed_number"
  | These_boxed_numbers _ -> "these_boxed_numbers"
  | Any_boxed_number _ -> "any_boxed_number"
  | Row_like_create _ -> "row_like_for_blocks_create"
  | Immutable_block_with_size_at_least _ -> "immutable_block_with_size_at_least"
  | Closure_with_at_least_these_function_slots _ ->
    "closure_with_at_least_these_function_slots"
  | Static_closure_with_this_code _ -> "static_closure_with_this_code"
  | Closure_with_at_least_these_value_slots _ ->
    "closure_with_at_least_these_value_slots"
  | Closure_with_at_least_this_value_slot _ ->
    "closure_with_at_least_this_value_slot"

let names_of_calls =
  [ "unknown";
    "bottom";
    "alias_type_of";
    "type_for_const";
    "null";
    "any_non_null_value";
    "this_tagged_immediate";
    "this_naked_immediate";
    "these_naked_immediates";
    "these_tagged_immediates";
    "any_tagged_immediate";
    "any_block";
    "tag_immediate";
    "box_number";
    "mutable_block";
    "this_immutable_string";
    "create_variant";
    "create_closures";
    "blocks_with_these_tags";
    "immutable_block";
    "row_like_for_blocks_create_exactly";
    "row_like_for_blocks_create_at_least";
    "row_like_for_blocks_create_at_least_unknown_tag";
    "is_int_for_scrutinee";
    "get_tag_for_block";
    "is_null_for_scrutinee";
    "exactly_this_closure";
    "at_least_this_closure";
    "unknown_like";
    "bottom_like";
    "unknown_from_shape";
    "unknown_with_subkind, of a kind";
    "kind_for_const, as the unknown type of the kind";
    "these_naked_numbers";
    "any_tagged_bool";
    "any_naked_bool";
    "any_tagged_immediate_or_null";
    "tagged_immediate_alias_to";
    "boxed_number_alias_to";
    "this_boxed_number";
    "these_boxed_numbers";
    "any_boxed_number";
    "row_like_for_blocks_create";
    "immutable_block_with_size_at_least";
    "closure_with_at_least_these_function_slots";
    "static_closure_with_this_code";
    "closure_with_at_least_these_value_slots";
    "closure_with_at_least_this_value_slot" ]

(* A row-like structure of blocks as a type, to compare it as one. *)
let blocks_as_type_of_model (blocks : MT.row_like_for_blocks) =
  MT.create_variant false MT.Or_unknown.Unknown (MT.Or_unknown.Known blocks)
    MT.No_extensions

let blocks_as_type_of_compiler (blocks : TG.row_like_for_blocks) =
  TG.create_variant ~is_unique:false ~immediates:Unknown ~blocks:(Known blocks)
    ~extensions:No_extensions

(* A function of the model that answers [None] where the compiler's function
   stops (a fatal error, an index outside an array) has no type to compare: the
   compiler's function is then to stop too. *)
exception No_answer

let call_in_the_model (call : call) : MT.flambda_type =
  let answer (result : _ option) =
    match result with Some result -> result | None -> raise No_answer
  in
  match call with
  | Unknown kind -> MT.unknown kind
  | Bottom kind -> MT.bottom kind
  | Alias_type_of (kind, simple) -> MT.alias_type_of kind simple
  | Type_for_const const -> MT.type_for_const const
  | Null -> MT.null
  | Any_non_null_value -> MT.any_non_null_value
  | This_tagged_immediate i -> MT.this_tagged_immediate i
  | This_naked_immediate i -> MT.this_naked_immediate i
  | These_naked_immediates imms -> MT.these_naked_immediates imms
  | These_tagged_immediates imms -> MT.these_tagged_immediates imms
  | Any_tagged_immediate -> MT.any_tagged_immediate
  | Any_block -> MT.any_block
  | Tag_immediate t -> MT.tag_immediate t
  | Box_number (number, contents, alloc_mode) ->
    MT.box_number number contents alloc_mode
  | Mutable_block alloc_mode -> MT.mutable_block alloc_mode
  | This_immutable_string bytes -> MT.this_immutable_string bytes
  | Create_variant (is_unique, immediates, blocks, extensions) ->
    MT.create_variant is_unique immediates blocks extensions
  | Create_closures (alloc_mode, closures) ->
    MT.create_closures alloc_mode closures
  | Blocks_with_these_tags (tags, alloc_mode) ->
    MT.blocks_with_these_tags tags alloc_mode
  | Immutable_block (is_unique, { tag; shape; fields; block_alloc_mode }) ->
    MT.immutable_block is_unique tag shape block_alloc_mode fields
  | Row_like_create_exactly { tag; shape; fields; block_alloc_mode } ->
    blocks_as_type_of_model
      (MT.row_like_for_blocks_create_exactly tag
         (N.nat_of_int (List.length fields))
         shape fields block_alloc_mode)
  | Row_like_create_at_least { tag; shape; fields; block_alloc_mode } ->
    blocks_as_type_of_model
      (MT.row_like_for_blocks_create_at_least tag
         (N.nat_of_int (List.length fields))
         shape fields block_alloc_mode)
  | Row_like_create_at_least_unknown_tag
      { tag = _; shape; fields; block_alloc_mode } ->
    blocks_as_type_of_model
      (MT.row_like_for_blocks_create_at_least_unknown_tag
         (N.nat_of_int (List.length fields))
         shape fields block_alloc_mode)
  | Is_int_for_scrutinee name -> MT.is_int_for_scrutinee name
  | Get_tag_for_block name -> MT.get_tag_for_block name
  | Is_null_for_scrutinee name -> MT.is_null_for_scrutinee name
  | Exactly_this_closure
      { this_function_slot;
        function_types;
        closure_types;
        value_slot_types;
        closure_alloc_mode
      } ->
    MT.exactly_this_closure this_function_slot function_types closure_types
      value_slot_types closure_alloc_mode
  | At_least_this_closure
      { this_function_slot;
        function_types;
        closure_types;
        value_slot_types;
        closure_alloc_mode
      } ->
    MT.at_least_this_closure this_function_slot function_types closure_types
      value_slot_types closure_alloc_mode
  | Unknown_like t -> MT.unknown_like t
  | Bottom_like t -> MT.bottom_like t
  | Unknown_from_shape (shape, index) ->
    answer (MT.unknown_from_shape shape (N.nat_of_int index))
  | Unknown_with_subkind kind -> MT.unknown_with_subkind kind
  | Kind_for_const const -> MT.unknown (MT.kind_for_const const)
  | These_naked_numbers (kind, numbers) -> MT.these_naked_numbers kind numbers
  | Any_tagged_bool -> MT.any_tagged_bool
  | Any_naked_bool -> MT.any_naked_bool
  | Any_tagged_immediate_or_null -> MT.any_tagged_immediate_or_null
  | Tagged_immediate_alias_to variable -> MT.tagged_immediate_alias_to variable
  | Boxed_number_alias_to (number, variable, alloc_mode) ->
    MT.boxed_number_alias_to number variable alloc_mode
  | This_boxed_number (number, i, alloc_mode) ->
    MT.this_boxed_number number i alloc_mode
  | These_boxed_numbers (number, numbers, alloc_mode) ->
    MT.these_boxed_numbers number numbers alloc_mode
  | Any_boxed_number number -> MT.any_boxed_number number
  | Row_like_create
      ({ tag = _; shape; fields; block_alloc_mode }, open_or_closed) ->
    blocks_as_type_of_model
      (answer
         (MT.row_like_for_blocks_create shape fields open_or_closed
            block_alloc_mode))
  | Immutable_block_with_size_at_least (tag, n, shape, x) ->
    answer (MT.immutable_block_with_size_at_least tag (N.nat_of_int n) shape x)
  | Closure_with_at_least_these_function_slots (function_slot, bindings) ->
    MT.closure_with_at_least_these_function_slots function_slot bindings
  | Static_closure_with_this_code (function_slot, symbol, code) ->
    MT.static_closure_with_this_code function_slot symbol code
  | Closure_with_at_least_these_value_slots (function_slot, value_slots) ->
    MT.closure_with_at_least_these_value_slots function_slot value_slots
  | Closure_with_at_least_this_value_slot
      (function_slot, value_slot, variable, kind) ->
    MT.closure_with_at_least_this_value_slot function_slot value_slot variable
      kind

exception No_such_function of string

let call_in_the_compiler (names : names) (call : call) : TG.t =
  let module C = Converter in
  let numbering = names.numbering in
  let ty = to_compiler names in
  let immediate = C.in_range "an immediate" (C.target_int_of_z machine_width) in
  let immediates imms =
    Target_ocaml_int.Set.of_list (List.map immediate imms)
  in
  let scrutinee name = Simple.name (C.name_of_model numbering name) in
  let variable = C.variable_of_model numbering in
  let function_slot = C.function_slot_of_model numbering in
  let numbers what of_z of_list numbers =
    of_list (List.map (C.in_range what of_z) numbers)
  in
  let no_function_for_vectors () =
    raise
      (No_such_function
         "the compiler has no such function for the vectors and the masks")
  in
  let create_blocks { tag; shape; fields; block_alloc_mode } open_or_closed =
    TG.Row_like_for_blocks.create ~machine_width
      ~shape:(C.block_shape_of_model shape)
      ~field_tys:(List.map ty fields)
      (open_or_closed (C.tag_of_model machine_width tag))
      (C.alloc_mode_of_model block_alloc_mode)
  in
  let closure_arguments
      { this_function_slot;
        function_types;
        closure_types;
        value_slot_types;
        closure_alloc_mode
      } =
    let function_slot = C.function_slot_of_model numbering in
    let slot_map of_model bindings =
      C.map_of_model ~empty:Function_slot.Map.empty ~mem:Function_slot.Map.mem
        ~add:Function_slot.Map.add function_slot of_model bindings
    in
    ( function_slot this_function_slot,
      slot_map
        (C.or_unknown_of_model
           (fun ({ function_type_code_id = code_id } : MT.function_type) ->
             TG.Function_type.create
               (C.code_id_of_model numbering code_id)
               ~rec_info:TG.any_rec_info))
        function_types,
      slot_map ty closure_types,
      C.map_of_model ~empty:Value_slot.Map.empty ~mem:Value_slot.Map.mem
        ~add:Value_slot.Map.add
        (C.value_slot_of_model numbering)
        ty value_slot_types,
      C.alloc_mode_of_model closure_alloc_mode )
  in
  match call with
  | Unknown kind -> MTC.unknown (C.kind_of_model kind)
  | Bottom kind -> MTC.bottom (C.kind_of_model kind)
  | Alias_type_of (kind, simple) ->
    TG.alias_type_of (C.kind_of_model kind)
      (C.simple_of_model numbering machine_width simple)
  | Type_for_const const ->
    MTC.type_for_const (C.const_of_model machine_width const)
  | Null -> TG.null
  | Any_non_null_value -> TG.any_non_null_value
  | This_tagged_immediate i -> TG.this_tagged_immediate (immediate i)
  | This_naked_immediate i -> TG.this_naked_immediate (immediate i)
  | These_naked_immediates imms -> TG.these_naked_immediates (immediates imms)
  | These_tagged_immediates imms ->
    MTC.these_tagged_immediates (immediates imms)
  | Any_tagged_immediate -> MTC.any_tagged_immediate
  | Any_block -> MTC.any_block
  | Tag_immediate t -> TG.tag_immediate (ty t)
  | Box_number (number, contents, alloc_mode) -> (
    let contents = ty contents in
    let alloc_mode = C.alloc_mode_of_model alloc_mode in
    match number with
    | Boxable_naked_float32 -> TG.box_float32 contents alloc_mode
    | Boxable_naked_float -> TG.box_float contents alloc_mode
    | Boxable_naked_int32 -> TG.box_int32 contents alloc_mode
    | Boxable_naked_int64 -> TG.box_int64 contents alloc_mode
    | Boxable_naked_nativeint -> TG.box_nativeint contents alloc_mode
    | Boxable_naked_vec128 -> TG.box_vec128 contents alloc_mode
    | Boxable_naked_vec256 -> TG.box_vec256 contents alloc_mode
    | Boxable_naked_vec512 -> TG.box_vec512 contents alloc_mode
    | Boxable_naked_mask -> TG.box_mask contents alloc_mode)
  | Mutable_block alloc_mode ->
    TG.mutable_block (C.alloc_mode_of_model alloc_mode)
  | This_immutable_string bytes ->
    TG.this_immutable_string
      (String.of_seq
         (Seq.map
            (fun byte ->
              Char.chr (C.in_range "a byte" (fun z -> N.int_of_z z) byte))
            (List.to_seq bytes)))
  | Create_variant (is_unique, immediates, blocks, extensions) ->
    TG.create_variant ~is_unique
      ~immediates:(C.or_unknown_of_model ty immediates)
      ~blocks:
        (C.or_unknown_of_model
           (C.row_like_for_blocks_of_model numbering machine_width)
           blocks)
      ~extensions:
        (C.variant_extensions_of_model numbering machine_width extensions)
  | Create_closures (alloc_mode, closures) ->
    TG.create_closures
      (C.alloc_mode_of_model alloc_mode)
      (C.row_like_for_closures_of_model numbering machine_width closures)
  | Blocks_with_these_tags (tags, alloc_mode) -> (
    match
      MTC.blocks_with_these_tags ~machine_width
        (Tag.Set.of_list (List.map (C.tag_of_model machine_width) tags))
        (C.alloc_mode_of_model alloc_mode)
    with
    | Known t -> t
    | Unknown ->
      raise (No_such_function "the compiler has no answer for these tags"))
  | Immutable_block (is_unique, { tag; shape; fields; block_alloc_mode }) ->
    MTC.immutable_block ~machine_width ~is_unique
      (C.tag_of_model machine_width tag)
      ~shape:(C.block_shape_of_model shape)
      (C.alloc_mode_of_model block_alloc_mode)
      ~fields:(List.map ty fields)
  | Row_like_create_exactly block ->
    blocks_as_type_of_compiler (create_blocks block (fun tag -> Closed tag))
  | Row_like_create_at_least block ->
    blocks_as_type_of_compiler
      (create_blocks block (fun tag -> Open (Known tag)))
  | Row_like_create_at_least_unknown_tag block ->
    blocks_as_type_of_compiler (create_blocks block (fun _ -> Open Unknown))
  | Is_int_for_scrutinee name ->
    TG.is_int_for_scrutinee ~machine_width ~scrutinee:(scrutinee name)
  | Get_tag_for_block name -> TG.get_tag_for_block ~block:(scrutinee name)
  | Is_null_for_scrutinee name ->
    TG.is_null ~machine_width ~scrutinee:(scrutinee name)
  | Exactly_this_closure arguments ->
    let ( this_function_slot,
          all_function_slots_in_set,
          all_closure_types_in_set,
          all_value_slots_in_set,
          alloc_mode ) =
      closure_arguments arguments
    in
    MTC.exactly_this_closure this_function_slot ~all_function_slots_in_set
      ~all_closure_types_in_set ~all_value_slots_in_set alloc_mode
  | At_least_this_closure arguments ->
    let ( this_function_slot,
          at_least_these_function_slots,
          at_least_these_closure_types,
          at_least_these_value_slots,
          alloc_mode ) =
      closure_arguments arguments
    in
    MTC.at_least_this_closure this_function_slot ~at_least_these_function_slots
      ~at_least_these_closure_types ~at_least_these_value_slots alloc_mode
  | Unknown_like t -> MTC.unknown_like (ty t)
  | Bottom_like t -> MTC.bottom_like (ty t)
  | Unknown_from_shape (shape, index) ->
    MTC.unknown_from_shape (C.block_shape_of_model shape) index
  | Unknown_with_subkind kind ->
    MTC.unknown_with_subkind ~machine_width
      (K.With_subkind.anything (C.kind_of_model kind))
  | Kind_for_const const ->
    MTC.unknown (MTC.kind_for_const (C.const_of_model machine_width const))
  | These_naked_numbers (kind, ns) -> (
    match kind with
    | Kind_naked_immediate -> TG.these_naked_immediates (immediates ns)
    | Kind_naked_float32 ->
      TG.these_naked_float32s
        (numbers "the bits of a float32" C.float32_of_z
           Numeric_types.Float32_by_bit_pattern.Set.of_list ns)
    | Kind_naked_float ->
      TG.these_naked_floats
        (numbers "the bits of a float" C.float_of_z
           Numeric_types.Float_by_bit_pattern.Set.of_list ns)
    | Kind_naked_int8 ->
      TG.these_naked_int8s
        (numbers "an int8" C.int8_of_z Numeric_types.Int8.Set.of_list ns)
    | Kind_naked_int16 ->
      TG.these_naked_int16s
        (numbers "an int16" C.int16_of_z Numeric_types.Int16.Set.of_list ns)
    | Kind_naked_int32 ->
      TG.these_naked_int32s
        (numbers "an int32" C.int32_of_z Numeric_types.Int32.Set.of_list ns)
    | Kind_naked_int64 ->
      TG.these_naked_int64s
        (numbers "an int64" C.int64_of_z Numeric_types.Int64.Set.of_list ns)
    | Kind_naked_nativeint ->
      TG.these_naked_nativeints
        (numbers "a nativeint"
           (C.nativeint_of_z machine_width)
           Targetint_32_64.Set.of_list ns)
    | Kind_naked_vec128 ->
      TG.these_naked_vec128s
        (numbers "the bits of a vec128" C.vec128_of_z
           Vector_types.Vec128.Bit_pattern.Set.of_list ns)
    | Kind_naked_vec256 ->
      TG.these_naked_vec256s
        (numbers "the bits of a vec256" C.vec256_of_z
           Vector_types.Vec256.Bit_pattern.Set.of_list ns)
    | Kind_naked_vec512 ->
      TG.these_naked_vec512s
        (numbers "the bits of a vec512" C.vec512_of_z
           Vector_types.Vec512.Bit_pattern.Set.of_list ns)
    | Kind_naked_mask ->
      TG.these_naked_masks
        (numbers "the bits of a mask" C.mask_of_z
           Vector_types.Mask.Bit_pattern.Set.of_list ns))
  | Any_tagged_bool -> MTC.any_tagged_bool ~machine_width
  | Any_naked_bool -> MTC.any_naked_bool ~machine_width
  | Any_tagged_immediate_or_null -> MTC.any_tagged_immediate_or_null
  | Tagged_immediate_alias_to x ->
    TG.tagged_immediate_alias_to ~naked_immediate:(variable x)
  | Boxed_number_alias_to (number, x, alloc_mode) -> (
    let x = variable x in
    let alloc_mode = C.alloc_mode_of_model alloc_mode in
    match number with
    | Boxable_naked_float32 ->
      TG.boxed_float32_alias_to ~naked_float32:x alloc_mode
    | Boxable_naked_float -> TG.boxed_float_alias_to ~naked_float:x alloc_mode
    | Boxable_naked_int32 -> TG.boxed_int32_alias_to ~naked_int32:x alloc_mode
    | Boxable_naked_int64 -> TG.boxed_int64_alias_to ~naked_int64:x alloc_mode
    | Boxable_naked_nativeint ->
      TG.boxed_nativeint_alias_to ~naked_nativeint:x alloc_mode
    | Boxable_naked_vec128 ->
      TG.boxed_vec128_alias_to ~naked_vec128:x alloc_mode
    | Boxable_naked_vec256 ->
      TG.boxed_vec256_alias_to ~naked_vec256:x alloc_mode
    | Boxable_naked_vec512 ->
      TG.boxed_vec512_alias_to ~naked_vec512:x alloc_mode
    | Boxable_naked_mask -> TG.boxed_mask_alias_to ~naked_mask:x alloc_mode)
  | This_boxed_number (number, i, alloc_mode) -> (
    let alloc_mode = C.alloc_mode_of_model alloc_mode in
    match number with
    | Boxable_naked_float32 ->
      MTC.this_boxed_float32
        (C.in_range "the bits of a float32" C.float32_of_z i)
        alloc_mode
    | Boxable_naked_float ->
      MTC.this_boxed_float
        (C.in_range "the bits of a float" C.float_of_z i)
        alloc_mode
    | Boxable_naked_int32 ->
      MTC.this_boxed_int32 (C.in_range "an int32" C.int32_of_z i) alloc_mode
    | Boxable_naked_int64 ->
      MTC.this_boxed_int64 (C.in_range "an int64" C.int64_of_z i) alloc_mode
    | Boxable_naked_nativeint ->
      MTC.this_boxed_nativeint
        (C.in_range "a nativeint" (C.nativeint_of_z machine_width) i)
        alloc_mode
    | Boxable_naked_vec128 ->
      MTC.this_boxed_vec128
        (C.in_range "the bits of a vec128" C.vec128_of_z i)
        alloc_mode
    | Boxable_naked_vec256 ->
      MTC.this_boxed_vec256
        (C.in_range "the bits of a vec256" C.vec256_of_z i)
        alloc_mode
    | Boxable_naked_vec512 ->
      MTC.this_boxed_vec512
        (C.in_range "the bits of a vec512" C.vec512_of_z i)
        alloc_mode
    | Boxable_naked_mask ->
      MTC.this_boxed_mask
        (C.in_range "the bits of a mask" C.mask_of_z i)
        alloc_mode)
  | These_boxed_numbers (number, ns, alloc_mode) -> (
    let alloc_mode = C.alloc_mode_of_model alloc_mode in
    match number with
    | Boxable_naked_float32 ->
      MTC.these_boxed_float32s
        (numbers "the bits of a float32" C.float32_of_z
           Numeric_types.Float32_by_bit_pattern.Set.of_list ns)
        alloc_mode
    | Boxable_naked_float ->
      MTC.these_boxed_floats
        (numbers "the bits of a float" C.float_of_z
           Numeric_types.Float_by_bit_pattern.Set.of_list ns)
        alloc_mode
    | Boxable_naked_int32 ->
      MTC.these_boxed_int32s
        (numbers "an int32" C.int32_of_z Numeric_types.Int32.Set.of_list ns)
        alloc_mode
    | Boxable_naked_int64 ->
      MTC.these_boxed_int64s
        (numbers "an int64" C.int64_of_z Numeric_types.Int64.Set.of_list ns)
        alloc_mode
    | Boxable_naked_nativeint ->
      MTC.these_boxed_nativeints
        (numbers "a nativeint"
           (C.nativeint_of_z machine_width)
           Targetint_32_64.Set.of_list ns)
        alloc_mode
    | Boxable_naked_vec128 | Boxable_naked_vec256 | Boxable_naked_vec512
    | Boxable_naked_mask ->
      no_function_for_vectors ())
  | Any_boxed_number number -> (
    match number with
    | Boxable_naked_float32 -> MTC.any_boxed_float32
    | Boxable_naked_float -> MTC.any_boxed_float
    | Boxable_naked_int32 -> MTC.any_boxed_int32
    | Boxable_naked_int64 -> MTC.any_boxed_int64
    | Boxable_naked_nativeint -> MTC.any_boxed_nativeint
    | Boxable_naked_vec128 | Boxable_naked_vec256 | Boxable_naked_vec512
    | Boxable_naked_mask ->
      no_function_for_vectors ())
  | Row_like_create (block, open_or_closed) ->
    let tag = C.tag_of_model machine_width in
    blocks_as_type_of_compiler
      (create_blocks block (fun (_ : Tag.t) ->
           match open_or_closed with
           | Open (Known t) -> Open (Known (tag t))
           | Open Unknown -> Open Unknown
           | Closed t -> Closed (tag t)))
  | Immutable_block_with_size_at_least (tag, n, shape, x) ->
    MTC.immutable_block_with_size_at_least ~machine_width
      ~tag:(C.or_unknown_of_model (C.tag_of_model machine_width) tag)
      ~n:(Target_ocaml_int.of_int machine_width n)
      ~shape:(C.block_shape_of_model shape)
      ~field_n_minus_one:(variable x)
  | Closure_with_at_least_these_function_slots (this_function_slot, bindings) ->
    MTC.closure_with_at_least_these_function_slots
      ~this_function_slot:(function_slot this_function_slot)
      (C.map_of_model ~empty:Function_slot.Map.empty ~mem:Function_slot.Map.mem
         ~add:Function_slot.Map.add function_slot
         (C.simple_of_model numbering machine_width)
         bindings)
  | Static_closure_with_this_code (this_function_slot, symbol, code) ->
    MTC.static_closure_with_this_code
      ~this_function_slot:(function_slot this_function_slot)
      ~closure_symbol:(Option.map (C.symbol_of_model numbering) symbol)
      ~code_id:(C.code_id_of_model numbering code)
  | Closure_with_at_least_these_value_slots (this_function_slot, value_slots) ->
    MTC.closure_with_at_least_these_value_slots
      ~this_function_slot:(function_slot this_function_slot)
      (C.map_of_model ~empty:Value_slot.Map.empty ~mem:Value_slot.Map.mem
         ~add:Value_slot.Map.add
         (C.value_slot_of_model numbering)
         (fun (x, kind) -> variable x, C.kind_of_model kind)
         value_slots)
  | Closure_with_at_least_this_value_slot
      (this_function_slot, value_slot, x, kind) ->
    MTC.closure_with_at_least_this_value_slot
      ~this_function_slot:(function_slot this_function_slot)
      (C.value_slot_of_model numbering value_slot)
      ~value_slot_var:(variable x) ~value_slot_kind:(C.kind_of_model kind)

(* A call within what the compiler's function asks of its arguments: the kinds
   that it checks, a tag that the shape permits, fields of the kinds of the
   shape, sets given without repeats, maps without shadowed bindings. *)
let generate_call (names : names) r : call =
  let pool = names.pool in
  let config : G.config = { well_formed = true; depth = 2 } in
  let ty kind = G.type_of_kind config r pool kind in
  let alloc_mode () = G.alloc_mode config r in
  let numbers () =
    Model_type.set N.compare_z (G.list r ~max:4 (fun () -> G.small_number r))
  in
  let block () : block =
    let shape = G.block_shape r in
    let tag =
      match shape with
      | Scannable (Value_only | Mixed_record _) ->
        N.z_of_int (G.pick r [0; 1; 2; 250])
      | Float_record -> MT.double_array_tag
    in
    let size =
      match shape with
      | Scannable (Mixed_record (value_prefix_size, flat_suffix)) ->
        N.int_of_nat value_prefix_size + List.length flat_suffix
      | Scannable Value_only | Float_record -> G.below r 4
    in
    { tag;
      shape;
      fields = List.map ty (MT.field_kinds shape (N.nat_of_int size));
      block_alloc_mode = alloc_mode ()
    }
  in
  (* [Row_like_for_blocks.create] gives a block of floats its one tag, known or
     not: no exported function of the compiler makes a structure of float blocks
     of an unknown tag, so that case of the model's function is compared with
     nothing. *)
  let rec scannable_block () =
    let block = block () in
    match block.shape with
    | Scannable (Value_only | Mixed_record _) -> block
    | Float_record -> scannable_block ()
  in
  let distinct_keys bindings =
    Model_type.map N.compare_positive Fun.id bindings
  in
  let slots_and_types () : slots_and_types =
    { this_function_slot = G.pick r pool.function_slots;
      function_types =
        distinct_keys
          (G.list r ~max:2 (fun () ->
               ( G.pick r pool.function_slots,
                 if G.chance r ~percent:25
                 then MT.Or_unknown.Unknown
                 else
                   MT.Or_unknown.Known
                     { MT.function_type_code_id = G.pick r pool.code_ids } )));
      closure_types =
        distinct_keys
          (G.list r ~max:2 (fun () ->
               let slot = G.pick r pool.function_slots in
               slot, ty MK.Value));
      value_slot_types =
        distinct_keys
          (G.list r ~max:2 (fun () -> G.pick r pool.value_slots, ty MK.Value));
      closure_alloc_mode = alloc_mode ()
    }
  in
  let kind () = G.kind r in
  let name () = G.name config r pool in
  let naked_immediate = MK.Naked_number Kind_naked_immediate in
  (* A variable of the pool, of whatever kind: the functions that are given one
     do not look at its kind. *)
  let variable () =
    G.pick r
      (pool.value_variables @ pool.naked_immediate_variables
     @ pool.naked_float_variables)
  in
  let boxable_number () = G.pick r G.all_boxable_numbers in
  let naked_number_kind () = G.pick r G.all_naked_number_kinds in
  (* A tag that a block of values may have, one that it may not, and the tag of
     the blocks of floats. *)
  let any_tag () = N.z_of_int (G.pick r [0; 1; 2; 250; 251; 254]) in
  let open_or_closed () : MT.Open_or_closed.t =
    match G.below r 3 with
    | 0 -> Closed (any_tag ())
    | 1 -> Open (Known (any_tag ()))
    | _ -> Open Unknown
  in
  let generators : (unit -> call) list =
    [ (fun () -> Unknown (kind ()));
      (fun () -> Bottom (kind ()));
      (fun () ->
        let kind = kind () in
        match G.simple config r pool kind with
        | Some simple -> Alias_type_of (kind, simple)
        | None -> Unknown kind);
      (fun () ->
        let kind = kind () in
        match G.const_of_kind config r kind with
        | Some const -> Type_for_const const
        | None -> Type_for_const (Poison kind));
      (fun () -> Null);
      (fun () -> Any_non_null_value);
      (fun () -> This_tagged_immediate (G.small_number r));
      (fun () -> This_naked_immediate (G.small_number r));
      (fun () -> These_naked_immediates (numbers ()));
      (fun () -> These_tagged_immediates (numbers ()));
      (fun () -> Any_tagged_immediate);
      (fun () -> Any_block);
      (fun () -> Tag_immediate (ty naked_immediate));
      (fun () ->
        let number = G.pick r G.all_boxable_numbers in
        Box_number (number, ty (MK.unboxed_kind number), alloc_mode ()));
      (fun () -> Mutable_block (alloc_mode ()));
      (fun () ->
        This_immutable_string
          (G.list r ~max:4 (fun () -> N.z_of_int (G.below r 256))));
      (fun () ->
        Create_variant
          ( G.chance r ~percent:20,
            (if G.chance r ~percent:30
             then MT.Or_unknown.Unknown
             else MT.Or_unknown.Known (ty naked_immediate)),
            (if G.chance r ~percent:30
             then MT.Or_unknown.Unknown
             else MT.Or_unknown.Known (G.row_like_for_blocks config r pool)),
            if G.chance r ~percent:70
            then MT.No_extensions
            else
              MT.Ext
                (G.env_extension config r pool, G.env_extension config r pool)
          ));
      (fun () ->
        Create_closures (alloc_mode (), G.row_like_for_closures config r pool));
      (fun () ->
        Blocks_with_these_tags
          ( Model_type.set N.compare_z
              (G.list r ~max:3 (fun () ->
                   let tag = G.pick r [0; 1; 2; 250] in
                   N.z_of_int tag)),
            alloc_mode () ));
      (fun () -> Immutable_block (G.chance r ~percent:20, block ()));
      (fun () -> Row_like_create_exactly (block ()));
      (fun () -> Row_like_create_at_least (block ()));
      (fun () -> Row_like_create_at_least_unknown_tag (scannable_block ()));
      (fun () -> Is_int_for_scrutinee (name ()));
      (fun () -> Get_tag_for_block (name ()));
      (fun () -> Is_null_for_scrutinee (name ()));
      (fun () -> Exactly_this_closure (slots_and_types ()));
      (fun () -> At_least_this_closure (slots_and_types ()));
      (fun () -> Unknown_like (ty (kind ())));
      (fun () -> Bottom_like (ty (kind ())));
      (* An index that the shape has, or one past its fields. *)
      (fun () -> Unknown_from_shape (G.block_shape r, G.below r 5));
      (fun () -> Unknown_with_subkind (kind ()));
      (fun () ->
        let kind = kind () in
        match G.const_of_kind config r kind with
        | Some const -> Kind_for_const const
        | None -> Kind_for_const (Poison kind));
      (fun () -> These_naked_numbers (naked_number_kind (), numbers ()));
      (fun () -> Any_tagged_bool);
      (fun () -> Any_naked_bool);
      (fun () -> Any_tagged_immediate_or_null);
      (fun () -> Tagged_immediate_alias_to (variable ()));
      (fun () ->
        Boxed_number_alias_to (boxable_number (), variable (), alloc_mode ()));
      (fun () ->
        This_boxed_number (boxable_number (), G.small_number r, alloc_mode ()));
      (fun () ->
        These_boxed_numbers (boxable_number (), numbers (), alloc_mode ()));
      (fun () -> Any_boxed_number (boxable_number ()));
      (fun () -> Row_like_create (block (), open_or_closed ()));
      (fun () ->
        Immutable_block_with_size_at_least
          ( (if G.chance r ~percent:30
             then MT.Or_unknown.Unknown
             else MT.Or_unknown.Known (any_tag ())),
            G.below r 5,
            G.block_shape r,
            variable () ));
      (fun () ->
        Closure_with_at_least_these_function_slots
          ( G.pick r pool.function_slots,
            distinct_keys
              (G.list r ~max:2 (fun () ->
                   ( G.pick r pool.function_slots,
                     match G.simple config r pool MK.Value with
                     | Some simple -> simple
                     | None -> M.Var (variable ()) ))) ));
      (fun () ->
        Static_closure_with_this_code
          ( G.pick r pool.function_slots,
            (if G.chance r ~percent:50
             then Some (G.pick r pool.symbols)
             else None),
            G.pick r pool.code_ids ));
      (fun () ->
        Closure_with_at_least_these_value_slots
          ( G.pick r pool.function_slots,
            distinct_keys
              (G.list r ~max:2 (fun () ->
                   G.pick r pool.value_slots, (variable (), kind ()))) ));
      (fun () ->
        Closure_with_at_least_this_value_slot
          ( G.pick r pool.function_slots,
            G.pick r pool.value_slots,
            variable (),
            kind () )) ]
  in
  (G.pick r generators) ()

(* A call is shown by the type that the model makes of it. *)
let model_on_call (call : call) : MT.flambda_type option =
  match call_in_the_model call with
  | of_model -> Some of_model
  | exception No_answer -> None

let print_call ppf (call : call) =
  match model_on_call call with
  | Some of_model ->
    Format.fprintf ppf "@[<v>a call of %s, which in the model gives@ %a@]"
      (name_of_call call) Model_type.print of_model
  | None ->
    Format.fprintf ppf "a call of %s, which in the model has no answer"
      (name_of_call call)

(* What the compiler's function does on a call: a type, as the model has it, or
   it stops (a fatal error, or the exception of an index outside an array). *)
let compiler_on_call (names : names) (call : call) :
    (MT.flambda_type, string) result =
  match
    or_fatal_error (fun () ->
        match call_in_the_compiler names call with
        | of_compiler -> Ok (to_model names of_compiler)
        | exception Invalid_argument what -> Error ("an exception: " ^ what))
  with
  | Ok result -> result
  | Error line -> Error ("a fatal error: " ^ line)

let creation_functions (names : names) : call comparison list =
  List.map
    (fun name ->
      { name;
        compare =
          (fun call ->
            if not (String.equal (name_of_call call) name)
            then Not_this_comparison
            else
              match compiler_on_call names call, model_on_call call with
              | exception No_such_function why -> Not_run why
              | Ok of_compiler, Some of_model ->
                agree_if
                  (Model_type.same of_model of_compiler)
                  (Format.asprintf "the compiler gives %a" Model_type.print
                     of_compiler)
              | Ok of_compiler, None ->
                Disagree
                  (Format.asprintf
                     "the model has no answer and the compiler gives %a"
                     Model_type.print of_compiler)
              | Error _, None -> Agree
              | Error why, Some (_ : MT.flambda_type) ->
                Disagree ("the compiler stops: " ^ why))
      })
    names_of_calls
  @ [ { name = "a creation function that is not in the list above";
        compare =
          (fun call ->
            if List.mem (name_of_call call) names_of_calls
            then Not_this_comparison
            else Disagree "it is compared with nothing")
      };
      (* These calls are counted above with those that agree. *)
      { name = "the calls on which the model has no answer: the compiler stops";
        compare =
          (fun call ->
            match model_on_call call with
            | Some (_ : MT.flambda_type) -> Not_this_comparison
            | None -> (
              match compiler_on_call names call with
              | exception No_such_function why -> Not_run why
              | Error (_ : string) -> Agree
              | Ok of_compiler ->
                Disagree
                  (Format.asprintf "the compiler gives %a" Model_type.print
                     of_compiler)))
      };
      { name =
          "type_well_formed of what a creation function gives, in the model \
           and in the compiler";
        compare =
          (fun call ->
            match compiler_on_call names call, model_on_call call with
            | exception No_such_function why -> Not_run why
            | (Error _ | Ok _), None -> Not_run "the model has no answer"
            | Error why, Some (_ : MT.flambda_type) ->
              Not_run ("the compiler stops: " ^ why)
            | Ok of_compiler, Some of_model ->
              (* The lists given to a call may bind a key twice. *)
              let of_model = is_well_formed (Model_type.normalise of_model) in
              let of_compiler = is_well_formed of_compiler in
              agree_if
                (Bool.equal of_model of_compiler)
                (Printf.sprintf
                   "it is %b of what the model gives and %b of what the \
                    compiler gives"
                   of_model of_compiler))
      };
      { name = "the calls that give a type of which type_well_formed is true";
        compare =
          (fun call ->
            match model_on_call call with
            | None -> Not_run "the model has no answer"
            | Some of_model ->
              if is_well_formed (Model_type.normalise of_model)
              then Agree
              else Not_run "it is false of the type")
      } ]

(* ======================================================================== *)
(* A relation on a constant, and the bottom test of a row-like structure    *)
(* ======================================================================== *)

(* [Relation.of_const]: each relation on a constant of each kind. *)
let relation_of_const (names : names) :
    (MT.Relation.t * M.const) comparison list =
  ignore (names : names);
  let print ppf (x : N.z MT.Or_bottom.t) =
    match x with
    | Ok n -> N.print_z ppf n
    | Bottom -> Format.pp_print_string ppf "Bottom"
  in
  [ { name = "Relation.of_const";
      compare =
        (fun (relation, const) ->
          let of_model = MT.Relation.of_const relation const in
          let of_compiler =
            Converter.or_bottom_to_model Converter.z_of_target_int
              (TG.Relation.of_const ~machine_width
                 (Converter.relation_of_model relation)
                 (Converter.const_of_model machine_width const))
          in
          agree_if
            (Stdlib.( = ) of_model of_compiler)
            (Format.asprintf "the model answers %a, the compiler %a" print
               of_model print of_compiler))
    } ]

(* [Row_like_for_blocks.is_bottom], on raw structures. *)
let row_like_bottom (names : names) : MT.row_like_for_blocks comparison list =
  [ { name = "Row_like_for_blocks.is_bottom";
      compare =
        (fun blocks ->
          let of_model = MT.row_like_for_blocks_is_bottom blocks in
          let of_compiler =
            TG.Row_like_for_blocks.is_bottom
              (Converter.row_like_for_blocks_of_model names.numbering
                 machine_width blocks)
          in
          agree_if
            (Bool.equal of_model of_compiler)
            (Printf.sprintf "the model answers %b, the compiler %b" of_model
               of_compiler))
    } ]

(* ======================================================================== *)
(* The two implementations of meet and join                                 *)
(* ======================================================================== *)

(* The compiler has two: [Meet_and_join], with [Join_levels_old], which is the
   one it runs unless told otherwise, and [Meet_and_n_way_join], with
   [Join_env], which it runs for the n-way join algorithm
   ([Flambda_features.join_algorithm], from the flag [-flambda2-join-algorithm]
   of driver/oxcaml_flags.ml; [Meet.meet] and [Join_levels.cut_and_n_way_join]
   choose by it). Here each is called by its own module, so both run in one
   process whatever the flag.

   Until the model has a meet and a join, the two are compared with each other:
   on the same two types, the same answer, as types of the model. *)

type implementation =
  { implementation : string;
    meet : TE.t -> TG.t -> TG.t -> (TG.t * TE.t) Or_bottom.t;
    (* The environment at a continuation with one parameter, from the
       environments at its uses: the definition environment, the environments at
       the uses, the parameter, and the scope to cut after. *)
    join :
      TE.t ->
      TE.t list ->
      params:Bound_parameters.t ->
      cut_after:Scope.t ->
      TE.t;
    meet_expanded_head : ME.meet_expanded_head
  }

let binary : implementation =
  { implementation = "binary";
    meet = Binary.meet;
    join =
      (fun env use_envs ~params ~cut_after ->
        Binary_join.cut_and_n_way_join env
          (List.map
             (fun use_env ->
               ( use_env,
                 Apply_cont_rewrite_id.create (),
                 Continuation_use_kind.Inlinable ))
             use_envs)
          ~params ~cut_after ~extra_lifted_consts_in_use_envs:Symbol.Set.empty
          ~extra_allowed_names:Name_occurrences.empty);
    meet_expanded_head = Binary.meet_expanded_head
  }

let n_way : implementation =
  { implementation = "n-way";
    meet = N_way.meet;
    join =
      (fun env use_envs ~params:_ ~cut_after ->
        fst
          (N_way_join.cut_and_n_way_join_with_analysis
             ~meet_expanded_head:N_way.meet_expanded_head
             ~n_way_join_type:N_way.n_way_join env ~cut_after
             (List.map (fun use_env -> (), use_env) use_envs)));
    meet_expanded_head = N_way.meet_expanded_head
  }

(* What an environment knows of the names of the tests, as types of the
   model. *)
let types_of_names (names : names) (env : TE.t) : MT.flambda_type list =
  List.map
    (fun (name, kind) -> to_model names (TE.find env name (Some kind)))
    names.defined

let print_types ppf types = Model_type.print_list Model_type.print ppf types

(* The meet of two types, and what it teaches of the names. *)
let meet_as_model (names : names) (implementation : implementation) t1 t2 :
    (MT.flambda_type * MT.flambda_type list) option =
  match implementation.meet names.env t1 t2 with
  | Bottom -> None
  | Ok (t, env) -> Some (to_model names t, types_of_names names env)

(* What a join gives: the type of the parameter, and the variables that the join
   made of its own (those that are not of the tests) that this type mentions, or
   the type of one of them, each with what the joined environment says of it.
   Where other types are read from the joined environment besides, the variables
   that they mention are among these too ([reached_from] below). *)
type joined =
  { joined : MT.flambda_type;
    own : (M.variable * MT.flambda_type) list;
    (* Those of [own] that the joined environment lets terms mention (their name
       mode is normal); the others are for types only. *)
    own_in_terms : M.variable list
  }

let is_own (names : names) (x : M.variable) =
  match N.int_of_positive x with
  | Some number -> number > names.variables_of_the_tests
  | None -> true

(* The variables of a type that are not of the tests, once each, in the order in
   which the type as written in one way mentions them. *)
let own_variables_of (names : names) (t : MT.flambda_type) : M.variable list =
  List.fold_left
    (fun found x ->
      if is_own names x && not (List.mem x found) then found @ [x] else found)
    []
    (MT.free_variables (Model_type.normalise t))

let joined_of_env ?(reached_from = []) (names : names) (env : TE.t)
    (parameter : Variable.t) : joined =
  let type_of variable =
    to_model names
      (TE.find env (Name.var variable) (Some (Variable.kind variable)))
  in
  let joined = type_of parameter in
  let rec close own (pending : M.variable list) =
    match pending with
    | [] -> List.rev own
    | x :: pending ->
      if List.mem_assoc x own
      then close own pending
      else
        let variable = Converter.variable_of_model names.numbering x in
        if not (TE.mem env (Name.var variable))
        then
          raise (Not_defined (Format.asprintf "%a" Model_type.print_variable x))
        else
          let t = type_of variable in
          close ((x, t) :: own) (pending @ own_variables_of names t)
  in
  let own =
    close [] (List.concat_map (own_variables_of names) (joined :: reached_from))
  in
  let own_in_terms =
    List.filter
      (fun x ->
        TE.mem ~min_name_mode:Name_mode.normal env
          (Name.var (Converter.variable_of_model names.numbering x)))
      (List.map fst own)
  in
  { joined; own; own_in_terms }

(* The join at a continuation with one parameter of a kind, which is applied to
   a value of each of the types given, one at each use: the joined environment
   and the parameter. For the use of a position in the list, [known] gives what
   that use knows of the names of the tests besides, and [locals] the variables
   that only that use binds, with their types. [in_types] are variables that the
   environment of the continuation's definition defines for types only.
   [observe] is given the environments of the uses before they are joined. *)
let join_env (names : names) (implementation : implementation)
    ?(known = fun (_ : int) -> []) ?(locals = fun (_ : int) -> [])
    ?(in_types = []) ?(observe = fun (_ : TE.t list) -> ())
    (kind : MK.flambda_kind) (types : TG.t list) : TE.t * Variable.t =
  let variable = parameter names kind in
  let compiler_kind = Converter.kind_of_model kind in
  let meet_expanded_head = implementation.meet_expanded_head in
  let define mode env variable =
    TE.add_definition env
      (Bound_name.create_var
         (Bound_var.create variable Flambda_debug_uid.none mode))
      (Variable.kind variable)
  in
  let env =
    List.fold_left
      (define Name_mode.in_types)
      (define Name_mode.normal names.env variable)
      in_types
  in
  let use_envs =
    List.mapi
      (fun i t ->
        let locals = locals i in
        let env =
          List.fold_left
            (fun env (local, _) -> define Name_mode.normal env local)
            (TE.increment_scope env) locals
        in
        ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
            List.fold_left
              (fun env (name, t) ->
                ME.add_equation env name t ~meet_expanded_head)
              env
              (known i
              @ List.map (fun (local, t) -> Name.var local, t) locals
              @ [Name.var variable, t])))
      types
  in
  observe use_envs;
  let joined =
    implementation.join env use_envs
      ~params:
        (Bound_parameters.create
           [ Bound_parameter.create variable
               (K.With_subkind.anything compiler_kind)
               Flambda_debug_uid.none ])
      ~cut_after:(TE.current_scope env)
  in
  joined, variable

let join (names : names) (implementation : implementation) ?known ?locals
    (kind : MK.flambda_kind) (types : TG.t list) : joined =
  let env, parameter =
    join_env names implementation ?known ?locals kind types
  in
  joined_of_env names env parameter

(* What is printed, with the variables numbered above a threshold named [e0],
   [e1], ... in the order they come. [own] is the table of those names: given
   again, it goes on with a second text where the first left it. *)
let rename_variables_above ?(own = Hashtbl.create 4) (threshold : int)
    (printed : string) : string =
  let buffer = Buffer.create (String.length printed) in
  let is_digit c = c >= '0' && c <= '9' in
  let length = String.length printed in
  let rec copy i =
    if i < length
    then
      if
        Char.equal printed.[i] 'v'
        && i + 1 < length
        && is_digit printed.[i + 1]
        && (i = 0 || not (is_digit printed.[i - 1]))
      then (
        let rec end_of_number j =
          if j < length && is_digit printed.[j]
          then end_of_number (j + 1)
          else j
        in
        let j = end_of_number (i + 1) in
        let number = int_of_string (String.sub printed (i + 1) (j - i - 1)) in
        if number <= threshold
        then Buffer.add_string buffer (String.sub printed i (j - i))
        else (
          if not (Hashtbl.mem own number)
          then Hashtbl.add own number (Hashtbl.length own);
          Buffer.add_string buffer
            (Printf.sprintf "e%d" (Hashtbl.find own number)));
        copy j)
      else (
        Buffer.add_char buffer printed.[i];
        copy (i + 1))
  in
  copy 0;
  Buffer.contents buffer

(* The same text with every variable numbered above the threshold written
   alike. *)
let without_the_names_above (threshold : int) (printed : string) : string =
  let own = Hashtbl.create 4 in
  let renamed = rename_variables_above ~own threshold printed in
  (* The new names are [e] and a number: the number is left out. *)
  let buffer = Buffer.create (String.length renamed) in
  let is_digit c = c >= '0' && c <= '9' in
  let is_letter c = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') in
  let length = String.length renamed in
  let rec copy i =
    if i < length
    then
      let starts_a_word =
        i = 0
        ||
        let before = renamed.[i - 1] in
        not (is_letter before || is_digit before)
      in
      if
        Char.equal renamed.[i] 'e'
        && i + 1 < length
        && is_digit renamed.[i + 1]
        && starts_a_word
      then (
        Buffer.add_char buffer 'e';
        let rec after j =
          if j < length && is_digit renamed.[j] then after (j + 1) else j
        in
        copy (after (i + 1)))
      else (
        Buffer.add_char buffer renamed.[i];
        copy (i + 1))
  in
  copy 0;
  Buffer.contents buffer

(* The same for the variables that are not those of the tests. *)
let rename_own_variables (names : names) (printed : string) : string =
  rename_variables_above names.variables_with_a_name printed

(* Whether two texts are the same but for how they are laid out. The lines of a
   text are broken before its variables are renamed, so where they are broken
   depends on how long the numbers of the variables were: two texts that differ
   only there are one answer. Nothing else is forgiven: the printer brackets
   every part of a type, so the words of a text in their order say what the type
   is. *)
let same_text (text : string) (text' : string) : bool =
  let words text =
    List.filter
      (fun word -> not (String.equal word ""))
      (String.split_on_char ' '
         (String.map (fun c -> if Char.equal c '\n' then ' ' else c) text))
  in
  List.equal String.equal (words text) (words text')

(* Types as printed, each written in one way. The variables that are not of the
   tests are renamed once, over the whole text: a variable that two of the types
   mention has one name in both, so two texts are equal only if one renaming of
   those variables takes the types of the one to the types of the other. *)
let print_with_own_variables (names : names) (types : MT.flambda_type list) :
    string =
  rename_own_variables names
    (String.concat "\n"
       (List.map
          (fun m ->
            Format.asprintf "%a" Model_type.print (Model_type.normalise m))
          types))

(* What a join gives, as printed: the type, then what is said of each of its own
   variables. *)
let print_joined_as_numbered ({ joined; own; own_in_terms } : joined) : string =
  let print ppf t = Model_type.print ppf (Model_type.normalise t) in
  Format.asprintf "@[<v>%a%a@]" print joined
    (fun ppf own ->
      List.iter
        (fun (x, t) ->
          Format.fprintf ppf "@ where %a%s : %a" Model_type.print_variable x
            (if List.mem x own_in_terms
             then ", which terms may mention,"
             else "")
            print t)
        own)
    own

let print_joined (names : names) (joined : joined) : string =
  rename_own_variables names (print_joined_as_numbered joined)

(* What a joined environment says, as printed: the type of the parameter, the
   types of the names of the tests, and what is said of each variable that is
   not of the tests and that any of these types reaches. *)
let print_joined_with_names (names : names) (env : TE.t)
    (parameter : Variable.t) : string =
  let learnt = types_of_names names env in
  rename_own_variables names
    (Format.asprintf "@[<v>%s@ and for the names %a@]"
       (print_joined_as_numbered
          (joined_of_env ~reached_from:learnt names env parameter))
       print_types
       (List.map Model_type.normalise learnt))

let two_implementations (names : names) :
    (MT.flambda_type * MT.flambda_type) comparison list =
  let print_meet ppf result =
    match result with
    | None -> Format.pp_print_string ppf "Bottom"
    | Some (t, types) ->
      Format.fprintf ppf "@[<v>%a@ and for the names %a@]" Model_type.print t
        print_types types
  in
  (* A meet may answer a type that mentions a variable made on the way. The
     environment answered with it is to define every such variable. *)
  let defined_in_the_answer (implementation : implementation) =
    { name =
        Printf.sprintf
          "meet (%s): the variables of the answer are defined in the \
           environment answered"
          implementation.implementation;
      compare =
        (fun (m1, m2) ->
          let t1 = to_compiler names m1 and t2 = to_compiler names m2 in
          match implementation.meet names.env t1 t2 with
          | Bottom -> Not_run "the meet is bottom"
          | Ok (t, env) -> (
            let of_the_names =
              List.map
                (fun (name, kind) -> TE.find env name (Some kind))
                names.defined
            in
            let not_defined =
              List.concat_map
                (fun t ->
                  Name_occurrences.fold_names (TG.free_names t) ~init:[]
                    ~f:(fun not_defined name ->
                      if TE.mem env name
                      then not_defined
                      else name :: not_defined))
                (t :: of_the_names)
            in
            match not_defined with
            | [] -> Agree
            | name :: _ ->
              Known_to_disagree
                ( "T4-9",
                  Format.asprintf
                    "@[<v>the meet answers@ %a@ and the environment answered \
                     does not define %a@]"
                    Model_type.print (to_model names t) Model_type.print_name
                    (Converter.name_to_model names.numbering name) )))
    }
  in
  let well_formed_meet (implementation : implementation) =
    { name =
        Printf.sprintf
          "meet (%s) of two types of which type_well_formed is true: it is \
           true of the answer and of what it teaches"
          implementation.implementation;
      compare =
        (fun (m1, m2) ->
          if not (is_well_formed m1 && is_well_formed m2)
          then Not_run "it is false of one of the two types"
          else
            let t1 = to_compiler names m1 and t2 = to_compiler names m2 in
            match meet_as_model names implementation t1 t2 with
            | None -> Not_run "the meet is bottom"
            | Some (meet, types) -> well_formed (meet :: types))
    }
  in
  let well_formed_join (implementation : implementation) =
    { name =
        Printf.sprintf
          "join (%s) of two types of which type_well_formed is true: it is \
           true of the answer and of what it says of its own variables"
          implementation.implementation;
      compare =
        (fun (m1, m2) ->
          if not (is_well_formed m1 && is_well_formed m2)
          then Not_run "it is false of one of the two types"
          else
            let types = [to_compiler names m1; to_compiler names m2] in
            let joined = join names implementation Value types in
            well_formed (joined.joined :: List.map snd joined.own))
    }
  in
  [ defined_in_the_answer binary;
    defined_in_the_answer n_way;
    well_formed_meet binary;
    well_formed_meet n_way;
    well_formed_join binary;
    well_formed_join n_way;
    { name = "meet: the binary and the n-way implementation";
      compare =
        (fun (m1, m2) ->
          let t1 = to_compiler names m1 and t2 = to_compiler names m2 in
          let of_binary = meet_as_model names binary t1 t2 in
          let of_n_way = meet_as_model names n_way t1 t2 in
          let same =
            match of_binary, of_n_way with
            | None, None -> true
            | Some (t, types), Some (t', types') ->
              Model_type.same t t' && List.equal Model_type.same types types'
            | None, Some _ | Some _, None -> false
          in
          agree_if same
            (Format.asprintf
               "@[<v>the binary meet gives@ %a@ the n-way meet@ %a@]" print_meet
               of_binary print_meet of_n_way))
    };
    { name = "join: the binary and the n-way implementation";
      compare =
        (fun (m1, m2) ->
          let types = [to_compiler names m1; to_compiler names m2] in
          let of_binary = join names binary Value types in
          let of_n_way = join names n_way Value types in
          (* A join may make variables of its own, which are not the same
             variables in the two: the two results are compared as they are
             printed, with those variables named in the order they come. *)
          let printed joined = print_joined names joined in
          if same_text (printed of_binary) (printed of_n_way)
          then Agree
          else
            (* The two may differ in how much they keep: that is recorded and
               not a failure. Whether each is above the types joined is for the
               comparison with the meaning of types. *)
            Known_to_disagree
              ( (match
                   ( MT.is_obviously_unknown of_binary.joined,
                     MT.is_obviously_unknown of_n_way.joined )
                 with
                | true, false ->
                  "T7-3, the binary join is unknown and the n-way join is not"
                | false, true ->
                  "T7-3, the n-way join is unknown and the binary join is not"
                | false, false | true, true ->
                  "T7-3, the two joins know different things"),
                Format.asprintf
                  "@[<v>the binary join gives@ %s@ the n-way join@ %s@]"
                  (printed of_binary) (printed of_n_way) ))
    } ]

(* ======================================================================== *)
(* The equality of types                                                    *)
(* ======================================================================== *)

(* [type_eqb] of FlambdaTypeEqualSpec.v against the compiler's
   [Type_grammar.equal], which the joins ask with the structural setting of the
   flag for sharing, and against the sameness that these tests use
   ([Model_type.same]: two types of the model written in one way are equal). The
   pairs are a type twice (converted twice: two values), a type and one of the
   smaller types that the shrinker gives for it, and two types generated
   apart. *)
let equality_of_types (names : names) :
    (MT.flambda_type * MT.flambda_type) comparison list =
  [ { name = "type_eqb against Type_grammar.equal";
      compare =
        (fun (m1, m2) ->
          let of_model = Type_equal.type_eqb m1 m2 in
          let of_compiler =
            TG.equal (to_compiler names m1) (to_compiler names m2)
          in
          agree_if
            (Bool.equal of_model of_compiler)
            (Format.asprintf "the model answers %b, the compiler %b" of_model
               of_compiler))
    };
    { name = "type_eqb against the sameness of the tests";
      compare =
        (fun (m1, m2) ->
          let of_model = Type_equal.type_eqb m1 m2 in
          let of_tests = Model_type.same m1 m2 in
          agree_if
            (Bool.equal of_model of_tests)
            (Format.asprintf "type_eqb answers %b, the tests %b" of_model
               of_tests))
    };
    { name = "the pairs that type_eqb answers to be the same";
      compare =
        (fun (m1, m2) ->
          if Type_equal.type_eqb m1 m2 then Agree else Not_run "not the same")
    } ]

let generate_pair_for_equality (names : names) (config : G.config) r :
    MT.flambda_type * MT.flambda_type =
  let first = G.flambda_type config r names.pool in
  match G.below r 3 with
  | 0 -> first, first
  | 1 -> (
    match G.shrink first with
    | [] -> first, first
    | _ :: _ as smaller -> first, G.pick r smaller)
  | _ -> first, G.flambda_type config r names.pool

(* ======================================================================== *)
(* Meet and join against the meaning of types                               *)
(* ======================================================================== *)

(* What a type means is [type_satisfied] of FlambdaTypeMeaningSpec.v, which
   [type_satisfied_b] decides (FlambdaTypeMeaningBoolSpec.v, extracted). So what
   a meet and a join of the compiler owe to the meaning can be tested on values,
   before the model has a meet or a join of its own: - a value that has one of
   two types has their join; - a value that has two types has their meet, and
   what the meet teaches of the names is true; and a value that has the meet,
   where what it teaches is true, has the two types. A world is a static
   environment and the values of the variables of the tests. No value of kind
   value in it is poison: typing facts are about values that are not. *)
type world =
  { statics : V.static_env;
    vars : (M.variable * V.value) list
  }

let tagged i : V.value = Value_const (Tagged_immediate (N.z_of_int i))

let naked_immediate i : V.value = Value_const (Naked_immediate (N.z_of_int i))

let const_of_number (k : MK.naked_number_kind) (n : N.z) : M.const =
  match k with
  | Kind_naked_immediate -> Naked_immediate n
  | Kind_naked_float32 -> Naked_float32 n
  | Kind_naked_float -> Naked_float n
  | Kind_naked_int8 -> Naked_int8 n
  | Kind_naked_int16 -> Naked_int16 n
  | Kind_naked_int32 -> Naked_int32 n
  | Kind_naked_int64 -> Naked_int64 n
  | Kind_naked_nativeint -> Naked_nativeint n
  | Kind_naked_vec128 -> Naked_vec128 n
  | Kind_naked_vec256 -> Naked_vec256 n
  | Kind_naked_vec512 -> Naked_vec512 n
  | Kind_naked_mask -> Naked_mask n

let naked_immediate_of_const (c : M.const) : N.z option =
  match c with
  | Naked_immediate i -> Some i
  | Tagged_immediate _ | Naked_float32 _ | Naked_float _ | Naked_int8 _
  | Naked_int16 _ | Naked_int32 _ | Naked_int64 _ | Naked_nativeint _
  | Naked_vec128 _ | Naked_vec256 _ | Naked_vec512 _ | Naked_mask _ | Null
  | Poison _ ->
    None

(* Some value of a kind. *)
let default_of_kind (k : MK.flambda_kind) : V.value =
  match k with
  | Value -> tagged 0
  | Naked_number k -> Value_const (const_of_number k (N.z_of_int 0))
  | Region | Rec_info -> Value_const (Poison k)

(* A value of kind value that is not poison. *)
let rec generate_value (names : names) r depth : V.value =
  let pool = names.pool in
  match if depth <= 0 then 0 else G.below r 10 with
  | 0 | 1 | 2 | 3 -> tagged (G.below r 4)
  | 4 -> Value_const Null
  | 5 | 6 ->
    if G.chance r ~percent:15
    then
      Value_block
        ( MT.double_array_tag,
          G.list r ~max:3 (fun () : V.value ->
              Value_const (Naked_float (N.z_of_int (G.below r 5)))) )
    else
      Value_block
        ( N.z_of_int (G.pick r [0; 1; 2; 250]),
          G.list r ~max:3 (fun () -> generate_value names r (depth - 1)) )
  | 7 ->
    let kind = G.pick r G.boxable_kinds in
    Value_boxed_number (const_of_number kind (N.z_of_int (G.below r 5)))
  | 8 -> (
    let value_slots =
      Model_type.map N.compare_positive Fun.id
        (G.list r ~max:2 (fun () ->
             G.pick r pool.value_slots, generate_value names r (depth - 1)))
    in
    (* The closure of a set of one function with no slot named, or the closure
       at one of the function slots of a set. *)
    let functions =
      if G.chance r ~percent:50
      then []
      else
        List.filter_map
          (fun slot ->
            if G.chance r ~percent:70
            then Some (slot, G.pick r pool.code_ids)
            else None)
          pool.function_slots
    in
    match functions with
    | [] -> Value_closure (G.pick r pool.code_ids, value_slots)
    | _ :: _ ->
      Value_closure_of_set (fst (G.pick r functions), functions, value_slots))
  | _ -> Value_symbol (G.pick r pool.symbols)

let generate_world (names : names) r : world =
  let pool = names.pool in
  let bind variables value = List.map (fun x -> x, value ()) variables in
  (* A static set of closures with a function at each function slot of the pool,
     whose closures are the symbols of the pool in their order. A symbol may be
     bound to its closure of this set; where the others are not, the projection
     of their slots from it gives a symbol that is something else, which the
     machine allows. *)
  let functions =
    List.map (fun slot -> slot, G.pick r pool.code_ids) pool.function_slots
  in
  let closure_symbols =
    List.filteri
      (fun i _ -> i < List.length pool.symbols)
      (List.mapi
         (fun i slot ->
           slot, List.nth pool.symbols (i mod List.length pool.symbols))
         pool.function_slots)
  in
  let value_slots_of_the_set =
    Model_type.map N.compare_positive Fun.id
      (G.list r ~max:2 (fun () ->
           G.pick r pool.value_slots, generate_value names r 1))
  in
  { statics =
      { static_code = [];
        static_values =
          List.map
            (fun symbol : (_ * V.statically_allocated_value) ->
              ( symbol,
                match G.below r 5 with
                | 4 -> (
                  match
                    List.find_opt
                      (fun (_, symbol') -> Stdlib.( = ) symbol symbol')
                      closure_symbols
                  with
                  | Some (slot, _) ->
                    Static_closure_of_set
                      (slot, closure_symbols, functions, value_slots_of_the_set)
                  | None -> Static_block (N.z_of_int 0, []))
                | 0 ->
                  Static_closure
                    ( G.pick r pool.code_ids,
                      [G.pick r pool.value_slots, generate_value names r 1] )
                | 1 ->
                  Static_string
                    (G.list r ~max:3 (fun () -> N.z_of_int (97 + G.below r 3)))
                | 2 ->
                  Static_boxed_number
                    (Value_const (Naked_float (N.z_of_int (G.below r 5))))
                | _ ->
                  Static_block
                    ( N.z_of_int (G.pick r [0; 1; 2]),
                      G.list r ~max:3 (fun () -> generate_value names r 1) ) ))
            pool.symbols
      };
    vars =
      bind pool.value_variables (fun () -> generate_value names r 2)
      @ bind pool.naked_immediate_variables (fun () ->
          naked_immediate (G.below r 3))
      @ bind pool.naked_float_variables (fun () : V.value ->
          Value_const (Naked_float (N.z_of_int (G.below r 5))))
      @ bind pool.region_variables (fun () -> default_of_kind Region)
      @ bind pool.rec_info_variables (fun () -> default_of_kind Rec_info)
  }

let value_of_simple (world : world) (simple : M.simple) : V.value list =
  match simple with
  | Var x -> Option.to_list (List.assoc_opt x world.vars)
  | Symbol s -> [V.Value_symbol s]
  | Const c -> [V.Value_const c]

(* Values that may well have a type: made from the type, to be tried. They are a
   help to the tests and are trusted for nothing: every use of a value asks
   [type_satisfied_b] whether it has the type. *)
let rec values_for (names : names) (world : world) (t : MT.flambda_type) :
    V.value list =
  let descr kind head (ty : _ MT.type_descr) =
    match ty with
    | Unknown -> [default_of_kind kind]
    | Bottom -> []
    | Ok (Equals simple) -> value_of_simple world simple
    | Ok (No_alias h) -> head h
  in
  match t with
  | Type_value ty ->
    descr Value
      (fun (Head_of_kind_value (non_null, is_null) : MT.head_of_kind_value) ->
        (match is_null with
          | Not_null -> []
          | Maybe_null _ -> [V.Value_const Null])
        @
        match non_null with
        | Unknown -> [tagged 0; V.Value_block (N.z_of_int 0, [])]
        | Bottom -> []
        | Ok head -> values_for_head names world head)
      ty
  | Type_naked_immediate ty ->
    descr (Naked_number Kind_naked_immediate)
      (fun (head : MT.head_of_kind_naked_immediate) ->
        match head.naked_immediates with
        | Known numbers ->
          List.map (fun n : V.value -> Value_const (Naked_immediate n)) numbers
        | Unknown -> List.init 4 naked_immediate)
      ty
  | Type_naked_number (k, ty) ->
    descr (Naked_number k)
      (List.map (fun n : V.value -> Value_const (const_of_number k n)))
      ty
  | Type_rec_info _ -> [default_of_kind Rec_info]
  | Type_region _ -> [default_of_kind Region]

and values_for_head names world (head : MT.head_of_kind_value_non_null) :
    V.value list =
  let first_or_default kind t =
    match values_for names world t with
    | v :: _ -> v
    | [] -> default_of_kind kind
  in
  let block tag (case : MT.row_like_block_case) : V.value =
    let size =
      match case.case_index.index_domain with
      | Known size | At_least size -> size
    in
    let kinds = MT.field_kinds case.case_index.index_shape size in
    Value_block
      ( tag,
        List.mapi
          (fun i kind ->
            match List.nth_opt case.case_maps_to i with
            | Some t -> first_or_default kind t
            | None -> default_of_kind kind)
          kinds )
  in
  match head with
  | Variant (_, immediates, _, blocks, _, _) -> (
    (match immediates with
      | Unknown -> [tagged 0; tagged 1]
      | Known t ->
        List.filter_map
          (fun (v : V.value) : V.value option ->
            match v with
            | Value_const c ->
              Option.map
                (fun i : V.value -> Value_const (Tagged_immediate i))
                (naked_immediate_of_const c)
            | Value_closure _ | Value_closure_of_set _ | Value_symbol _
            | Value_block _ | Value_immutable_array _ | Value_boxed_number _
            | Value_location _ | Value_partial_application _ ->
              None)
          (values_for names world t))
    @
    match blocks with
    | Unknown -> [V.Value_block (N.z_of_int 0, [])]
    | Known (Row_like_for_blocks (known_tags, other_tags, _)) -> (
      List.map
        (fun (tag, (case : _ MT.Or_unknown.t)) ->
          match case with
          | Unknown -> V.Value_block (tag, [])
          | Known case -> block tag case)
        known_tags
      @
      match other_tags with
      | Bottom -> []
      | Ok case -> [block (N.z_of_int 3) case]))
  | Mutable_block _ | String _ -> []
  | Boxed_number (_, contents, _) ->
    List.filter_map
      (fun (v : V.value) : V.value option ->
        match v with
        | Value_const c -> Some (Value_boxed_number c)
        | Value_closure _ | Value_closure_of_set _ | Value_symbol _
        | Value_block _ | Value_immutable_array _ | Value_boxed_number _
        | Value_location _ | Value_partial_application _ ->
          None)
      (values_for names world contents)
  | Closures (Row_like_for_closures cases, _) ->
    (* For each case, the closure at its function slot of a set with the
       functions and the value slots of the index: each function has the code
       that the case says, or else one code of the pool, tried in turn. Where
       the set is of the one function with no slot named, also the closure that
       names no slot. *)
    List.concat_map
      (fun (slot, (case : (_, _, _, _) MT.row_like_case)) ->
        let (Closures_entry (function_types, _, value_slot_types)
              : MT.closures_entry) =
          case.case_maps_to
        in
        let (contents : MT.set_of_closures_contents) =
          match case.case_index.index_domain with
          | Known contents | At_least contents -> contents
        in
        let value_slots =
          List.map
            (fun slot ->
              ( slot,
                match List.assoc_opt slot value_slot_types with
                | Some t -> first_or_default Value t
                | None -> tagged 0 ))
            (Model_type.set N.compare_positive contents.contents_value_slots)
        in
        let function_slots =
          Model_type.set N.compare_positive (slot :: contents.contents_closures)
        in
        List.concat_map
          (fun code : V.value list ->
            let functions =
              List.map
                (fun slot ->
                  ( slot,
                    match List.assoc_opt slot function_types with
                    | Some (Known { function_type_code_id }) ->
                      function_type_code_id
                    | Some Unknown | None -> code ))
                function_slots
            in
            Value_closure_of_set (slot, functions, value_slots)
            ::
            (match functions with
            | [(slot, code)] when Stdlib.( = ) slot V.unnamed_function_slot ->
              [V.Value_closure (code, value_slots)]
            | [] | _ :: _ -> []))
          names.pool.code_ids)
      cases

(* The values tried against two types: those made from the types, the values of
   the variables and the symbols, and a few more. *)
let candidates (names : names) (world : world) r types : V.value list =
  let not_poison v = not (Meaning.is_poison_b v) in
  List.filter not_poison
    (List.concat_map (values_for names world) types
    @ List.map snd world.vars
    @ List.map (fun symbol -> V.Value_symbol symbol) names.pool.symbols
    @ List.init 8 (fun _ -> generate_value names r 2))

let has (world : world) (v : V.value) (t : MT.flambda_type) : bool =
  Meaning.type_satisfied_b world.statics world.vars v t

let rec parts_of_value (v : V.value) : V.value list =
  v
  ::
  (match v with
  | Value_const _ | Value_symbol _ | Value_boxed_number _ | Value_location _ ->
    []
  | Value_block (_, fields) | Value_immutable_array fields ->
    List.concat_map parts_of_value fields
  | Value_closure (_, value_slots) | Value_closure_of_set (_, _, value_slots) ->
    List.concat_map (fun (_, v) -> parts_of_value v) value_slots
  | Value_partial_application (callee, args, _) ->
    parts_of_value callee @ List.concat_map parts_of_value args)

(* A join may make variables of its own, which stand for some value. A value has
   what a join gives if, with the parameter holding the value and for some
   values of those variables, it has the type of the parameter and each of those
   variables has the type that the joined environment gives it. (These types may
   mention the parameter: a relation between it and a variable.) A variable may
   also have no value, and then nothing is asked of its type: it stands for a
   variable that only some of the uses bind, and what satisfies a typing
   environment need not give every variable a value. The values tried for a
   variable are those made from its type, small numbers and the parts of the
   value, of its kind. *)
type own_variables_answer =
  | Has  (** Values were found. *)
  | Has_not
      (** There is no variable to find a value for, and the test fails. *)
  | Unresolved
      (** No values were found for the variables. The search is finite, so this
          decides nothing: it is neither that the value has what the join gives
          nor that it has not. *)

let has_joined (names : names) (world : world) (v : V.value)
    ({ joined; own; own_in_terms = _ } : joined) : own_variables_answer =
  let values_of (x_type : MT.flambda_type) =
    let kind = MT.kind x_type in
    List.fold_left
      (fun values value ->
        if
          Stdlib.( = ) (V.kind_of_value value) kind
          && not (List.exists (Meaning.value_eqb value) values)
        then values @ [value]
        else values)
      []
      (values_for names world x_type
      @ List.map naked_immediate [0; 1; 2; 3; 4; 5; 250; 254]
      @ parts_of_value v)
  in
  let rec for_some vars (rest : (M.variable * MT.flambda_type) list) =
    match rest with
    | [] ->
      Meaning.type_satisfied_b world.statics vars v joined
      && List.for_all
           (fun (x, x_type) ->
             match List.assoc_opt x vars with
             | Some value ->
               Meaning.type_satisfied_b world.statics vars value x_type
             | None -> true)
           own
    | (x, x_type) :: rest ->
      for_some vars rest
      || List.exists
           (fun value -> for_some ((x, value) :: vars) rest)
           (values_of x_type)
  in
  let vars =
    ( Converter.variable_to_model names.numbering
        (parameter names (MT.kind joined)),
      v )
    :: world.vars
  in
  match own with
  | [] ->
    (* No variable of its own: the test decides. *)
    if for_some vars [] then Has else Has_not
  | _ :: _ ->
    (* The search is finite: finding values shows that the value has what the
       join gives; not finding any shows nothing. *)
    if List.compare_length_with own 3 <= 0 && for_some vars own
    then Has
    else Unresolved

let print_value ppf (v : V.value) =
  let rec print ppf (v : V.value) =
    match v with
    | Value_const c -> Model_type.print_const ppf c
    | Value_symbol s -> Model_type.print_symbol ppf s
    | Value_block (tag, fields) ->
      Format.fprintf ppf "@[<hov 1>(block %a@ %a)@]" N.print_z tag
        (Model_type.print_list print)
        fields
    | Value_immutable_array elements ->
      Format.fprintf ppf "@[<hov 1>(array@ %a)@]"
        (Model_type.print_list print)
        elements
    | Value_boxed_number c ->
      Format.fprintf ppf "(boxed %a)" Model_type.print_const c
    | Value_location l -> Format.fprintf ppf "<location %a>" N.print_positive l
    | Value_partial_application (callee, args, _) ->
      Format.fprintf ppf "@[<hov 1>(partial %a@ %a)@]" print callee
        (Model_type.print_list print)
        args
    | Value_closure (code, value_slots) ->
      Format.fprintf ppf "@[<hov 1>(closure code%a@ %a)@]" N.print_positive code
        print_value_slots value_slots
    | Value_closure_of_set (slot, functions, value_slots) ->
      Format.fprintf ppf "@[<hov 1>(closure at %a of@ %a@ %a)@]"
        N.print_positive slot
        (Model_type.print_list (fun ppf (slot, code) ->
             Format.fprintf ppf "%a = code%a" N.print_positive slot
               N.print_positive code))
        functions print_value_slots value_slots
  and print_value_slots ppf value_slots =
    Model_type.print_list
      (fun ppf (slot, v) ->
        Format.fprintf ppf "%a = %a" N.print_positive slot print v)
      ppf value_slots
  in
  print ppf v

type semantic_case =
  { world_seed : int;
    first : MT.flambda_type;
    second : MT.flambda_type
  }

let semantic_comparisons (names : names) : semantic_case comparison list =
  let setting { world_seed; first; second } =
    let r = Random.of_int world_seed in
    let world = generate_world names r in
    let values = candidates names world r [first; second] in
    world, values, to_compiler names first, to_compiler names second
  in
  let join_is_above (implementation : implementation) =
    { name =
        Printf.sprintf "join (%s): a value of one of the types has the join"
          implementation.implementation;
      compare =
        (fun ({ world_seed = _; first; second } as case) ->
          let world, values, t1, t2 = setting case in
          let kind = MT.kind first in
          if not (Stdlib.( = ) kind (MT.kind second))
          then Not_run "the two types are not of one kind"
          else
            let join = join names implementation kind [t1; t2] in
            let of_an_input =
              List.filter
                (fun v -> has world v first || has world v second)
                values
            in
            let outside =
              List.find_opt
                (fun v ->
                  match has_joined names world v join with
                  | Has_not -> true
                  | Has | Unresolved -> false)
                of_an_input
            in
            match outside, of_an_input with
            | Some v, _ ->
              let what =
                Format.asprintf
                  "@[<v>the value %a has one of the types and not their join,@ \
                   %s@]"
                  print_value v (print_joined names join)
              in
              (* Before its repair in the compiler the n-way join kept a
                 function type that only one of the types has (T1-4), and such a
                 case was shown as recorded. It is a failure like any other
                 now. *)
              Disagree what
            | None, [] -> Not_run "no value tried has either type"
            | None, _ :: _ ->
              if
                List.exists
                  (fun v ->
                    match has_joined names world v join with
                    | Unresolved -> true
                    | Has | Has_not -> false)
                  of_an_input
              then
                Not_run
                  "the join has variables of its own, and no values were found \
                   for them"
              else Agree)
    }
  in
  let types_of_names_in world env =
    List.filter_map
      (fun (name, kind) ->
        let value : V.value option =
          Name.pattern_match name
            ~var:(fun var ->
              List.assoc_opt
                (Converter.variable_to_model names.numbering var)
                world.vars)
            ~symbol:(fun symbol ->
              Some
                (V.Value_symbol
                   (Converter.symbol_to_model names.numbering symbol)))
        in
        match value with
        | Some value when not (Meaning.is_poison_b value) ->
          Some (name, value, to_model names (TE.find env name (Some kind)))
        | Some _ | None -> None)
      names.defined
  in
  let contains ~part whole =
    let length = String.length part in
    let rec at i =
      i + length <= String.length whole
      && (String.equal (String.sub whole i length) part || at (i + 1))
    in
    at 0
  in
  let meet_is_the_values_of_both (implementation : implementation) =
    { name =
        Printf.sprintf "meet (%s): a value of both types has the meet"
          implementation.implementation;
      compare =
        (fun ({ world_seed = _; first; second } as case) ->
          let world, values, t1, t2 = setting case in
          let of_both =
            List.filter
              (fun v -> has world v first && has world v second)
              values
          in
          match implementation.meet names.env t1 t2 with
          | Bottom -> (
            match of_both with
            | v :: _ ->
              Disagree
                (Format.asprintf
                   "the meet is bottom, and the value %a has both types"
                   print_value v)
            | [] -> Not_run "no value tried has both types")
          | Ok (meet, env) -> (
            let meet = to_model names meet in
            let learnt = types_of_names_in world env in
            let untrue =
              List.find_opt
                (fun (_, value, t) -> not (has world value t))
                learnt
            in
            match
              List.find_opt (fun v -> not (has world v meet)) of_both, untrue
            with
            | Some v, _ ->
              Disagree
                (Format.asprintf
                   "@[<v>the value %a has both types and not their meet,@ %a@]"
                   print_value v Model_type.print meet)
            | None, Some (name, value, t) when of_both <> [] ->
              let what =
                Format.asprintf
                  "@[<v>a value has both types, and the meet says of %a, which \
                   is %a, that it has the type@ %a@]"
                  Model_type.print_name
                  (Converter.name_to_model names.numbering name)
                  print_value value Model_type.print t
              in
              (* What [reduce_inverse_relations] deduces from [Get_tag]: one of
                 the types has that relation somewhere, at the top or in the
                 immediates of a variant. *)
              let printed =
                Format.asprintf "%a %a" Model_type.print first Model_type.print
                  second
              in
              if contains ~part:"Get_tag " printed
              then Known_to_disagree ("T1-1", what)
              else Disagree what
            | None, (Some _ | None) ->
              if of_both = []
              then Not_run "no value tried has both types"
              else Agree))
    }
  in
  (* The other way, which is about how much a meet keeps and not about whether
     it is right: where what the meet teaches of the names is true, a value of
     the meet has both types. Where it has not, the meet is larger than the two
     types together. *)
  let meet_is_no_larger (implementation : implementation) =
    { name =
        Printf.sprintf "meet (%s): a value of the meet has both types"
          implementation.implementation;
      compare =
        (fun ({ world_seed = _; first; second } as case) ->
          let world, values, t1, t2 = setting case in
          match implementation.meet names.env t1 t2 with
          | Bottom -> Not_run "the meet is bottom: no value has it"
          | Ok (meet, env) -> (
            let meet = to_model names meet in
            let learnt = types_of_names_in world env in
            if
              not (List.for_all (fun (_, value, t) -> has world value t) learnt)
            then Not_run "what the meet teaches is not true in the world"
            else
              match
                List.find_opt
                  (fun v ->
                    has world v meet
                    && not (has world v first && has world v second))
                  values
              with
              | None ->
                if List.exists (fun v -> has world v meet) values
                then Agree
                else Not_run "no value tried has the meet"
              | Some v ->
                let what =
                  Format.asprintf
                    "@[<v>the value %a has the meet,@ %a@ and not both types@]"
                    print_value v Model_type.print meet
                in
                let printed =
                  Format.asprintf "%a %a" Model_type.print first
                    Model_type.print second
                in
                if contains ~part:"Mutable_block" printed
                then
                  Known_to_disagree
                    ("T7-5, the meet with a mutable block keeps too much", what)
                else if
                  contains ~part:"(Maybe_null (Some" printed
                  || contains ~part:"(is_int (Some" printed
                  || contains ~part:"(get_tag (Some" printed
                then
                  Known_to_disagree
                    ( "T7-4, the meet forgets what a variable of a type holds",
                      what )
                else if
                  contains ~part:"Is_null " printed
                  || contains ~part:"Is_int " printed
                  || contains ~part:"Get_tag " printed
                then
                  Known_to_disagree
                    ( "T7-6, the meet forgets a relation of a naked immediate",
                      what )
                else Disagree what))
    }
  in
  [ join_is_above binary;
    join_is_above n_way;
    meet_is_the_values_of_both binary;
    meet_is_the_values_of_both n_way;
    meet_is_no_larger binary;
    meet_is_no_larger n_way ]

let print_semantic_case ppf { world_seed; first; second } =
  Format.fprintf ppf "@[<v>%a@ and@ %a@ in the world of seed %d@]"
    Model_type.print first Model_type.print second world_seed

let shrink_semantic_case ({ world_seed = _; first; second } as case) =
  List.map (fun first -> { case with first }) (G.shrink first)
  @ List.map (fun second -> { case with second }) (G.shrink second)

(* ======================================================================== *)
(* Shared and not                                                           *)
(* ======================================================================== *)

(* The results of meet and join are to be functions of the structure of their
   inputs: a type built once and given twice (shared: the two arguments are one
   value) and the same type built twice (not shared) are to give the same
   result. Where they do not, the result depends on physical equality. *)
let shared_and_not (names : names) (sharing : sharing) :
    MT.flambda_type comparison list =
  let compare_results name of_types =
    { name;
      compare =
        (fun m ->
          let t = to_compiler names m in
          let again = to_compiler names m in
          if t == again
          then Not_run "the type built twice is one value"
          else
            let shared = of_types t t in
            let not_shared = of_types t again in
            if same_text shared not_shared
            then Agree
            else
              depends_on_sharing sharing
                (Format.asprintf
                   "@[<v>with the two arguments one value:@ %s@ with the type \
                    built twice:@ %s@]"
                   shared not_shared))
    }
  in
  let join implementation t t' =
    print_joined names (join names implementation Value [t; t'])
  in
  let meet implementation t t' =
    match meet_as_model names implementation t t' with
    | None -> "Bottom"
    | Some (meet, types) -> print_with_own_variables names (meet :: types)
  in
  [ compare_results "join (binary) of a type with itself" (join binary);
    compare_results "join (n-way) of a type with itself" (join n_way);
    compare_results "meet (binary) of a type with itself" (meet binary);
    compare_results "meet (n-way) of a type with itself" (meet n_way) ]

(* ======================================================================== *)
(* Shared parts                                                             *)
(* ======================================================================== *)

(* The same for two different types that have a part in common: the part is
   converted once and put in both (the two types share one value), or converted
   twice. The types are made around the parts by the compiler's own functions: -
   two blocks of tag 0 whose first field is the common part; - two variants of
   immediates whose immediate case says of a variable that it has the common
   part as its type; - two uses of a continuation that pass the same variable,
   each knowing that the variable has the common part as its type. What is
   compared is the type of the parameter with what the join says of the names of
   the tests and of each variable of its own that any of these types reaches. *)
type parts =
  { common : MT.flambda_type;
    left : MT.flambda_type;
    right : MT.flambda_type
  }

let shared_parts (names : names) (sharing : sharing) : parts comparison list =
  let variable =
    Converter.variable_of_model names.numbering
      (List.hd names.pool.value_variables)
  in
  let block fields =
    MTC.immutable_block ~machine_width ~is_unique:false
      (Converter.tag_of_model machine_width (N.z_of_int 0))
      ~shape:(Converter.block_shape_of_model (Scannable Value_only))
      (Converter.alloc_mode_of_model Heap)
      ~fields
  in
  let variant numbers said =
    TG.create_variant ~is_unique:false
      ~immediates:
        (Known
           (to_compiler names
              (MT.these_naked_immediates (List.map N.z_of_int numbers))))
      ~blocks:
        (Known
           (Converter.row_like_for_blocks_of_model names.numbering machine_width
              MT.row_like_for_blocks_bottom))
      ~extensions:
        (Ext
           { when_immediate =
               TG.Env_extension.create
                 ~equations:(Name.Map.singleton (Name.var variable) said);
             when_block = TG.Env_extension.empty
           })
  in
  let join implementation ?known types =
    let env, parameter = join_env names implementation ?known Value types in
    print_joined_with_names names env parameter
  in
  let meet implementation t t' =
    match meet_as_model names implementation t t' with
    | None -> "Bottom"
    | Some (meet, types) -> print_with_own_variables names (meet :: types)
  in
  let compare_results name result =
    { name;
      compare =
        (fun { common; left; right } ->
          let once = to_compiler names common in
          let again = to_compiler names common in
          if once == again
          then Not_run "the common part built twice is one value"
          else
            let left = to_compiler names left in
            let right = to_compiler names right in
            let shared = result ~common:once ~common':once ~left ~right in
            let not_shared = result ~common:once ~common':again ~left ~right in
            if same_text shared not_shared
            then Agree
            else
              depends_on_sharing sharing
                (Format.asprintf
                   "@[<v>with the common part one value:@ %s@ with the common \
                    part built twice:@ %s@]"
                   shared not_shared))
    }
  in
  let of_blocks operation ~common ~common' ~left ~right =
    operation (block [common; left]) (block [common'; right])
  in
  let of_variants (implementation : implementation) ~common ~common' ~left:_
      ~right:_ =
    join implementation [variant [5; 6] common; variant [6; 7] common']
  in
  let of_what_is_known (implementation : implementation) ~common ~common'
      ~left:_ ~right:_ =
    let passed = TG.alias_type_of K.value (Simple.var variable) in
    join implementation
      ~known:(fun i -> [(Name.var variable, if i = 0 then common else common')])
      [passed; passed]
  in
  List.concat_map
    (fun (implementation : implementation) ->
      let name what =
        Printf.sprintf "join (%s) of %s" implementation.implementation what
      in
      [ compare_results
          (name "two blocks with a common field")
          (of_blocks (fun t t' -> join implementation [t; t']));
        compare_results
          (name "two variants whose immediate case says the same")
          (of_variants implementation);
        compare_results
          (name "a variable of which the two uses know the same")
          (of_what_is_known implementation) ])
    [binary; n_way]
  @ List.map
      (fun (implementation : implementation) ->
        compare_results
          (Printf.sprintf "meet (%s) of two blocks with a common field"
             implementation.implementation)
          (of_blocks (meet implementation)))
      [binary; n_way]

(* ======================================================================== *)
(* Chosen joins                                                             *)
(* ======================================================================== *)

(* Joins chosen by hand, where the two implementations are known or thought to
   part. For each: the types joined, what each implementation gives, and for a
   few values, each in a world where the variables named hold the values given,
   which of the types joined the value has and whether it has what each join
   gives. A value that has one of the types and not a join is a disagreement,
   unless it is recorded. *)
type use =
  { (* The variables that only this use binds, with their types. *)
    binds : (M.variable * MT.flambda_type) list;
    (* What this use knows of names that the source defines. *)
    learns : (MT.name * MT.flambda_type) list;
    (* The type of what the continuation is applied to. *)
    passes : MT.flambda_type
  }

type chosen_join =
  { title : string;
    kind : MK.flambda_kind;
    (* Variables that the source defines for types only. *)
    in_types : M.variable list;
    (* What every use knows of the names of the tests. *)
    known : (MT.name * MT.flambda_type) list;
    uses : use list;
    probes : ((M.variable * V.value) list * V.value) list;
    (* The entry that records that the n-way join is not above the types. *)
    recorded : string option
  }

(* A world where every variable of the tests holds the first value of its kind
   and every symbol is a block of tag 0 with no field. *)
let plain_world (names : names) : world =
  let pool = names.pool in
  let bind kind = List.map (fun x -> x, default_of_kind kind) in
  { statics =
      { static_code = [];
        static_values =
          List.map
            (fun symbol : (_ * V.statically_allocated_value) ->
              symbol, Static_block (N.z_of_int 0, []))
            pool.symbols
      };
    vars =
      bind Value pool.value_variables
      @ bind (Naked_number Kind_naked_immediate) pool.naked_immediate_variables
      @ bind (Naked_number Kind_naked_float) pool.naked_float_variables
      @ bind Region pool.region_variables
      @ bind Rec_info pool.rec_info_variables
  }

let print_lines ppf text =
  Format.pp_print_list ~pp_sep:Format.pp_print_cut Format.pp_print_string ppf
    (String.split_on_char '\n' text)

let chosen_joins (names : names) (cases : chosen_join list) =
  Format.printf "@.Chosen joins@.";
  let print_type ppf t = Model_type.print ppf (Model_type.normalise t) in
  let plain = plain_world names in
  List.iter
    (fun { title; kind; in_types; known; uses; probes; recorded } ->
      let lines = ref [] in
      let line print = lines := print :: !lines in
      List.iter
        (fun x ->
          line (fun ppf ->
              Format.fprintf ppf "the source defines %a for types only"
                Model_type.print_variable x))
        in_types;
      List.iter
        (fun { binds; learns; passes } ->
          line (fun ppf ->
              Format.fprintf ppf "@[<v 2>at one use:@ @[%a@]" print_type passes;
              List.iter
                (fun (x, t) ->
                  Format.fprintf ppf "@ where that use binds %a :@ @[%a@]"
                    Model_type.print_variable x print_type t)
                binds;
              List.iter
                (fun (name, t) ->
                  Format.fprintf ppf "@ where that use knows of %a:@ @[%a@]"
                    Model_type.print_name name print_type t)
                learns;
              Format.fprintf ppf "@]"))
        uses;
      List.iter
        (fun (name, t) ->
          line (fun ppf ->
              Format.fprintf ppf "@[<v 2>every use knows of %a:@ @[%a@]@]"
                Model_type.print_name name print_type t))
        known;
      let compiler_name = Converter.name_of_model names.numbering in
      let compiler_variable = Converter.variable_of_model names.numbering in
      let compiler_types =
        List.map (fun { passes; _ } -> to_compiler names passes) uses
      in
      let equations =
        List.map (fun (name, t) -> compiler_name name, to_compiler names t)
      in
      let compiler_known i =
        equations known @ equations (List.nth uses i).learns
      in
      let compiler_locals i =
        List.map
          (fun (x, t) -> compiler_variable x, to_compiler names t)
          (List.nth uses i).binds
      in
      (* The names of the source of which one use knows something that the
         others do not, or that are for types only: what each use and each join
         says of them is shown. *)
      let watched =
        List.sort_uniq Stdlib.compare
          (List.map (fun x -> MT.Name_var x) in_types
          @ List.concat_map (fun { learns; _ } -> List.map fst learns) uses)
      in
      let says_of env ppf (name : MT.name) =
        let compiler = compiler_name name in
        let simple = Simple.name compiler in
        Format.fprintf ppf "says of %a:@ @[%a@]" Model_type.print_name name
          print_type
          (to_model names (TE.find env compiler None));
        let canonical = TE.get_canonical_simple_ignoring_name_mode env simple in
        if not (Simple.equal canonical simple)
        then
          Format.fprintf ppf "@ which is %a" Model_type.print_simple
            (Converter.simple_to_model names.numbering canonical)
      in
      let joins =
        List.map
          (fun (implementation : implementation) ->
            let observe use_envs =
              if watched <> []
              then
                List.iteri
                  (fun i use_env ->
                    let text =
                      Format.asprintf "@[<v 2>use %d of the %s join%s%a@]"
                        (i + 1) implementation.implementation
                        (if TE.is_bottom use_env
                         then ", whose environment is bottom,"
                         else "")
                        (fun ppf ->
                          List.iter (fun name ->
                              Format.fprintf ppf " %a" (says_of use_env) name))
                        watched
                    in
                    line (fun ppf -> print_lines ppf text))
                  use_envs
            in
            let joined =
              or_fatal_error (fun () ->
                  let env, parameter =
                    join_env names implementation ~known:compiler_known
                      ~locals:compiler_locals
                      ~in_types:(List.map compiler_variable in_types)
                      ~observe kind compiler_types
                  in
                  ( joined_of_env names env parameter,
                    List.map
                      (fun name ->
                        Format.asprintf "@[<v 2>%a@]" (says_of env) name)
                      watched ))
            in
            let given =
              match joined with
              | Ok (joined, _) -> print_joined names joined
              | Error what ->
                "nothing: the compiler stops with the fatal error " ^ what
            in
            line (fun ppf ->
                Format.fprintf ppf "@[<v 2>the %s join gives@ %a@]"
                  implementation.implementation print_lines given);
            (match joined with
            | Ok (_, of_the_watched) ->
              List.iter
                (fun text ->
                  line (fun ppf ->
                      Format.fprintf ppf "@[<v 2>the %s join %a@]"
                        implementation.implementation print_lines text))
                of_the_watched
            | Error _ -> ());
            implementation, Option.map fst (Result.to_option joined))
          [binary; n_way]
      in
      List.iter
        (fun (assignments, v) ->
          let world = { plain with vars = assignments @ plain.vars } in
          (* A value has the type of a use for some values of the variables that
             the use binds. Where the use binds some and no values are found for
             them, it is not known whether the value has the type. *)
          let of_each_use =
            List.mapi
              (fun i { binds; passes; learns = _ } ->
                ( string_of_int (i + 1),
                  has_joined names world v
                    { joined = passes; own = binds; own_in_terms = [] } ))
              uses
          in
          let uses_that answer =
            List.filter_map
              (fun (use, answer') ->
                if Stdlib.( = ) answer answer' then Some use else None)
              of_each_use
          in
          let of_the_types = uses_that Has in
          let of_a_join ((implementation : implementation), joined) =
            Option.map
              (fun joined ->
                Printf.sprintf "the %s join: %s" implementation.implementation
                  (match has_joined names world v joined with
                  | Has -> "yes"
                  | Unresolved ->
                    "not for the values tried of its own variables"
                  | Has_not -> (
                    match of_the_types, recorded with
                    | [], _ -> "no"
                    | _ :: _, Some entry -> "no (recorded as " ^ entry ^ ")"
                    | _ :: _, None ->
                      incr failures;
                      "no: A DISAGREEMENT")))
              joined
          in
          let of_the_uses =
            (match of_the_types with
              | [] -> []
              | [use] -> ["use " ^ use]
              | _ :: _ :: _ -> ["uses " ^ String.concat " and " of_the_types])
            @ List.map
                (fun use ->
                  "use " ^ use
                  ^ " not for the values tried of the variables it binds")
                (uses_that Unresolved)
          in
          let answers =
            (match of_the_uses with
            | [] -> "no use"
            | _ :: _ -> String.concat ", " of_the_uses)
            :: List.filter_map of_a_join joins
          in
          line (fun ppf ->
              List.iter
                (fun (x, value) ->
                  Format.fprintf ppf "with %a = %a, " Model_type.print_variable
                    x print_value value)
                assignments;
              Format.fprintf ppf "%a: %s" print_value v
                (String.concat "; " answers)))
        probes;
      Format.printf "  @[<v 2>@[%a@]@ %a@]@." Format.pp_print_text title
        (Format.pp_print_list ~pp_sep:Format.pp_print_cut (fun ppf print ->
             print ppf))
        (List.rev !lines))
    cases

let the_chosen_joins (names : names) : chosen_join list =
  let pool = names.pool in
  let z = N.z_of_int in
  let x = List.hd pool.value_variables in
  let i = List.hd pool.naked_immediate_variables in
  let local = Converter.variable_to_model names.numbering names.local in
  let for_types_only =
    Converter.variable_to_model names.numbering names.for_types_only
  in
  let use ?(binds = []) ?(learns = []) passes = { binds; learns; passes } in
  let constant n = MT.this_tagged_immediate (z n) in
  let block fields =
    MT.immutable_block false (z 0) (Scannable Value_only) Heap fields
  in
  let block_value fields : V.value =
    Value_block (z 0, List.map tagged fields)
  in
  let no_blocks = MT.Or_unknown.Known MT.row_like_for_blocks_bottom in
  let variant_5_or_6 numbers =
    MT.create_variant false
      (Known (MT.these_naked_immediates [z 5; z 6]))
      no_blocks
      (Ext
         ( Env_extension
             [Name_var i, MT.these_naked_immediates (List.map z numbers)],
           MT.empty_env_extension ))
  in
  let is_int_of_x numbers : MT.flambda_type =
    Type_naked_immediate
      (Ok
         (No_alias
            { naked_immediates = Known (List.map z numbers);
              inverse_relations = [Is_int, Name_var x]
            }))
  in
  let slot =
    List.find
      (fun slot -> Stdlib.( = ) slot V.unnamed_function_slot)
      pool.function_slots
  in
  let closure function_types =
    MT.at_least_this_closure slot function_types [] [] MT.Alloc_mode.Heap
  in
  let code, other_code =
    match pool.code_ids with
    | code :: other_code :: _ -> code, other_code
    | [] | [_] -> assert false
  in
  [ { title =
        "Two blocks, (0, 0) and (1, 1), each field an alias of its constant";
      kind = Value;
      in_types = [];
      known = [];
      uses =
        (let zero = constant 0 and one = constant 1 in
         [use (block [zero; zero]); use (block [one; one])]);
      probes =
        [ [], block_value [0; 0];
          [], block_value [1; 1];
          [], block_value [0; 1];
          [], block_value [2; 2] ];
      recorded = None
    };
    { title =
        "The immediate 0, and a block whose field is a variable that only that \
         use binds, known to be 4 or 5";
      kind = Value;
      in_types = [];
      known = [];
      uses =
        [ use (constant 0);
          use
            ~binds:[local, MT.these_tagged_immediates [z 4; z 5]]
            (block [MT.alias_type_of Value (Var local)]) ];
      probes =
        [ [], tagged 0;
          [], block_value [4];
          [], block_value [5];
          [], block_value [6];
          [], tagged 1 ];
      recorded = None
    };
    { title =
        "The same, the variable known to be 4: its type is the alias of the \
         constant";
      kind = Value;
      in_types = [];
      known = [];
      uses =
        [ use (constant 0);
          use
            ~binds:[local, MT.these_tagged_immediates [z 4]]
            (block [MT.alias_type_of Value (Var local)]) ];
      probes = [[], tagged 0; [], block_value [4]; [], block_value [5]];
      recorded = None
    };
    { title =
        "The immediates 5 and 6 as a variant whose immediate case says that a \
         variable is 1, and that it is 2 (R1-15)";
      kind = Value;
      in_types = [];
      known = [];
      uses = [use (variant_5_or_6 [1]); use (variant_5_or_6 [2])];
      probes =
        [ [i, naked_immediate 1], tagged 5;
          [i, naked_immediate 2], tagged 5;
          [i, naked_immediate 3], tagged 5;
          [i, naked_immediate 1], tagged 7 ];
      recorded = None
    };
    { title =
        "A number that is 0 or 2, and one that is 2 or 4, each what is_int \
         answers on a variable that every use knows to be the immediate 3: no \
         number is either (R1-16)";
      kind = Naked_number Kind_naked_immediate;
      in_types = [];
      known = [Name_var x, constant 3];
      uses = [use (is_int_of_x [0; 2]); use (is_int_of_x [2; 4])];
      probes =
        [ [x, tagged 3], naked_immediate 0;
          [x, tagged 3], naked_immediate 1;
          [x, tagged 3], naked_immediate 2 ];
      recorded = None
    };
    { title =
        "The same two numbers as the immediates of two variants with no block \
         (R1-16)";
      kind = Value;
      in_types = [];
      known = [Name_var x, constant 3];
      uses =
        [ use
            (MT.create_variant false
               (Known (is_int_of_x [0; 2]))
               no_blocks No_extensions);
          use
            (MT.create_variant false
               (Known (is_int_of_x [2; 4]))
               no_blocks No_extensions) ];
      probes = [[x, tagged 3], tagged 0; [x, tagged 3], tagged 2];
      recorded = None
    };
    { title =
        "A variable of the source, for types only, that one use knows to be 0 \
         or 2 and what is_int answers on a variable known to be the immediate \
         3 (no number is that), and the other use knows to be 1";
      kind = Value;
      in_types = [for_types_only];
      known = [Name_var x, constant 3];
      uses =
        [ use
            ~learns:[Name_var for_types_only, is_int_of_x [0; 2]]
            (MT.this_tagged_immediate (z 0));
          use
            ~learns:[Name_var for_types_only, MT.this_naked_immediate (z 1)]
            (constant 0) ];
      probes = [];
      recorded = None
    };
    { title =
        "A closure whose own function has a known code, and one of which \
         nothing is said of the code (T1-4)";
      kind = Value;
      in_types = [];
      known = [];
      uses =
        [ use
            (closure
               [slot, MT.Or_unknown.Known { function_type_code_id = code }]);
          use (closure []) ];
      probes =
        [[], V.Value_closure (code, []); [], V.Value_closure (other_code, [])];
      recorded = None
    } ]

(* ======================================================================== *)
(* Examples of the join of types                                            *)
(* ======================================================================== *)

(* Small joins whose answers were written down from a transcription of the n-way
   join of types into the model, without running the compiler
   (FlambdaJoinTypesExamplesSpec.v). Each is run here: the types are the types
   of the argument at the uses of a continuation, and the answer is the type of
   the parameter in the joined environment, which is compared with the answer
   written down.

   Where the transcription makes a variable, the compiler makes one of its own
   and gives it a type in the joined environment. The join of environments then
   projects out a variable that is mentioned once and is not what [Is_int],
   [Get_tag] or [Is_null] answers: its type is put in its place
   ([Join_env.define_or_eliminate_variables_and_add_equations]). So: - where the
   variable is for [Is_int] or [Get_tag], the answer written down has a variable
   that is not of the tests there, the two are compared as text with those
   variables renamed, and what the compiler says of its variable is shown; -
   where it is another variable, mentioned once, the answer written here is not
   the statement's: it has in the place of the variable the type that the notes
   of the transcription say the variable has, the join of the types of what it
   stands for.

   Each example says where it is from: a statement of that file, by its name
   without [example_join_] and [_statement], with its inputs and its answer; a
   statement with its inputs and another answer, for the reason above; or not
   that file.

   A difference does not fail the test: it is shown, for the owner of the
   transcription to say which side is wrong. *)
type source =
  | Statement of string
  | Statement_with_another_answer of string
  | Not_from_the_file

type written_down =
  | Answer of MT.flambda_type
  | Thought_to_stop
      (** The transcription has no answer, where the compiler is thought to
          stop. *)
  | Nothing_to_compare of string
      (** What the compiler answers is shown, and why it is not compared. *)

type example_of_join =
  { example : string;
    source : source;
    (* What every use knows of variables of the tests. *)
    knowing : (M.variable * MT.flambda_type) list;
    inputs : MT.flambda_type list;
    answer : written_down
  }

let examples_of_joins (names : names) : example_of_join list =
  let z = N.z_of_int in
  let numbers = List.map z in
  let own i = N.positive_of_int (names.variables_with_a_name + i) in
  let x, y =
    match names.pool.value_variables with
    | x :: y :: _ -> x, y
    | [] | [_] -> assert false
  in
  let function_slot = List.hd names.pool.function_slots in
  let value_slot = List.hd names.pool.value_slots in
  let code, other_code =
    match names.pool.code_ids with
    | code :: other_code :: _ -> code, other_code
    | [] | [_] -> assert false
  in
  let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate in
  let naked_float : MK.flambda_kind = Naked_number Kind_naked_float in
  let tagged = MT.these_tagged_immediates in
  let constant i = MT.this_tagged_immediate (z i) in
  let alias variable = MT.alias_type_of Value (Var variable) in
  let floats numbers : MT.flambda_type =
    Type_naked_number (Kind_naked_float, Ok (No_alias numbers))
  in
  (* An alias of kind value to a naked immediate. *)
  let alias_to_a_naked_immediate =
    MT.alias_type_of Value (Const (Naked_immediate (z 1)))
  in
  (* A variant with the variables given for [Is_int] and [Get_tag], the
     immediates given, and for each tag listed the blocks on the heap of exactly
     the fields given, which are values. *)
  let variant is_int immediates get_tag blocks : MT.flambda_type =
    MT.non_null_value
      (Variant
         ( is_int,
           Known immediates,
           get_tag,
           Known
             (Row_like_for_blocks
                ( List.map
                    (fun (tag, fields) ->
                      ( z tag,
                        MT.Or_unknown.Known
                          ({ case_maps_to = fields;
                             case_index =
                               { index_domain =
                                   Known (N.nat_of_int (List.length fields));
                                 index_shape = Scannable Value_only
                               };
                             case_env_extension = MT.empty_env_extension
                           }
                            : MT.row_like_block_case) ))
                    blocks,
                  Bottom,
                  Heap )),
           No_extensions,
           false ))
  in
  let block tag fields =
    MT.immutable_block false (z tag) (Scannable Value_only) Heap fields
  in
  let function_types codes =
    List.map
      (fun code_id ->
        ( function_slot,
          MT.Or_unknown.Known
            ({ function_type_code_id = code_id } : MT.function_type) ))
      codes
  in
  (* The closure at the function slot of a set of that one function, with or
     without a function type for it, and one value slot. *)
  let closure codes value_slot_type =
    MT.exactly_this_closure function_slot (function_types codes) []
      [value_slot, value_slot_type]
      Heap
  in
  let statement ?(knowing = []) name example inputs answer =
    { example; source = Statement name; knowing; inputs; answer }
  in
  let with_another_answer ?(knowing = []) name example inputs answer =
    { example;
      source = Statement_with_another_answer name;
      knowing;
      inputs;
      answer = Answer answer
    }
  in
  let not_from_the_file ?(knowing = []) example inputs answer =
    { example;
      source = Not_from_the_file;
      knowing;
      inputs;
      answer = Answer answer
    }
  in
  [ statement "naked_immediates" "two sets of naked immediates"
      [ MT.these_naked_immediates (numbers [1; 2]);
        MT.these_naked_immediates (numbers [2; 3]) ]
      (Answer (MT.these_naked_immediates (numbers [1; 2; 3])));
    statement "tagged_immediates" "two sets of tagged immediates"
      [tagged (numbers [1; 2]); tagged (numbers [3; 4])]
      (Answer (tagged (numbers [3; 4; 1; 2])));
    statement "boxed_floats" "two boxed floats"
      [ MT.box_number Boxable_naked_float (floats (numbers [1; 2])) Heap;
        MT.box_number Boxable_naked_float (floats (numbers [3])) Heap ]
      (Answer
         (MT.box_number Boxable_naked_float (floats (numbers [1; 2; 3])) Heap));
    statement "strings" "two strings"
      [ MT.this_immutable_string (numbers [104; 105]);
        MT.this_immutable_string (numbers [111]) ]
      (Answer (MT.non_null_value (String [numbers [104; 105]; numbers [111]])));
    statement "one_constant" "one constant twice"
      [constant 5; constant 5]
      (Answer (constant 5));
    statement "two_constants" "two constants"
      [constant 5; constant 6]
      (Nothing_to_compare
         "the transcription makes a variable and answers an alias to it; here \
          the parameter is that variable, and the answer is its type");
    not_from_the_file
      "two constants in a field (a variable is made, mentioned once: its type \
       is in its place)"
      [block 0 [constant 5]; block 0 [constant 6]]
      (block 0 [tagged (numbers [5; 6])]);
    not_from_the_file
      "two constants in two fields (one variable is made, mentioned twice)"
      [block 0 [constant 5; constant 5]; block 0 [constant 6; constant 6]]
      (block 0 [alias (own 1); alias (own 1)]);
    statement "constant_and_set" "a constant and a set"
      [constant 5; tagged (numbers [1; 2])]
      (Answer (tagged (numbers [1; 2; 5])));
    statement "unknown" "a set and the unknown type"
      [MT.these_naked_immediates (numbers [1; 2]); MT.unknown naked_immediate]
      (Answer (MT.unknown naked_immediate));
    statement "bottom" "a set and bottom"
      [MT.these_naked_immediates (numbers [1; 2]); MT.bottom naked_immediate]
      (Answer (MT.these_naked_immediates (numbers [1; 2])));
    statement "all_bottom" "bottom twice"
      [MT.bottom Value; MT.bottom Value]
      (Answer (MT.bottom Value));
    statement "string_and_immediate" "a string and a tagged immediate"
      [MT.this_immutable_string (numbers [104; 105]); MT.any_tagged_immediate]
      (Answer MT.any_non_null_value);
    statement "null_and_immediate" "null and a tagged immediate"
      [MT.null; MT.any_tagged_immediate]
      (Answer
         (Type_value
            (Ok
               (No_alias
                  (Head_of_kind_value
                     ( Ok
                         (Variant
                            ( None,
                              Unknown,
                              None,
                              Known MT.row_like_for_blocks_bottom,
                              No_extensions,
                              false )),
                       Maybe_null None ))))));
    statement "mutable_blocks" "two mutable blocks, on the heap and local"
      [MT.mutable_block Heap; MT.mutable_block Local]
      (Answer (MT.mutable_block Heap_or_local));
    not_from_the_file "two mutable blocks, on the heap and on the heap or local"
      [MT.mutable_block Heap; MT.mutable_block Heap_or_local]
      (MT.mutable_block Heap_or_local);
    statement "fields" "blocks of one tag and one size"
      [block 0 [tagged (numbers [1; 2])]; block 0 [tagged (numbers [3; 4])]]
      (Answer (block 0 [tagged (numbers [1; 2; 3; 4])]));
    statement "same_fields" "blocks whose field is one variable"
      [block 0 [alias x]; block 0 [alias x]]
      (Answer (block 0 [alias x]));
    with_another_answer "fields_of_two_variables"
      "blocks whose fields are two variables of which nothing is known (a \
       variable is made, mentioned once: its type is in its place)"
      [block 0 [alias x]; block 0 [alias y]]
      (block 0 [MT.unknown Value]);
    not_from_the_file
      "blocks whose fields are two variables known to be numbers (the same)"
      ~knowing:[x, tagged (numbers [1; 2]); y, tagged (numbers [3; 4])]
      [block 0 [alias x]; block 0 [alias y]]
      (block 0 [tagged (numbers [1; 2; 3; 4])]);
    not_from_the_file
      "blocks with two fields that are two variables of which nothing is known \
       (one variable is made, mentioned twice)"
      [block 0 [alias x; alias x]; block 0 [alias y; alias y]]
      (block 0 [alias (own 1); alias (own 1)]);
    not_from_the_file
      "blocks with two fields that are two variables known to be numbers (the \
       same)"
      ~knowing:[x, tagged (numbers [1; 2]); y, tagged (numbers [3; 4])]
      [block 0 [alias x; alias x]; block 0 [alias y; alias y]]
      (block 0 [alias (own 1); alias (own 1)]);
    statement "two_sizes" "blocks of one tag and two sizes"
      [ block 0 [tagged (numbers [1; 2])];
        block 0 [tagged (numbers [3; 4]); MT.any_tagged_immediate] ]
      (Answer
         (MT.create_variant false
            (Known (MT.bottom naked_immediate))
            (Known
               (MT.row_like_for_blocks_create_at_least (z 0) (N.nat_of_int 1)
                  (Scannable Value_only)
                  [tagged (numbers [1; 2; 3; 4])]
                  Heap))
            No_extensions));
    statement "two_shapes"
      "blocks of one tag and two shapes (two values; a value and a naked float)"
      [ block 0 [MT.any_tagged_immediate; MT.any_tagged_immediate];
        MT.immutable_block false (z 0)
          (Scannable (Mixed_record (N.nat_of_int 1, [Kind_naked_float])))
          Heap
          [MT.any_tagged_immediate; MT.unknown naked_float] ]
      (Answer (MT.blocks_with_these_tags [z 0] Heap));
    statement "two_tags" "blocks of two tags (a variable is made for Get_tag)"
      [ block 0 [MT.any_tagged_immediate];
        block 1 [tagged (numbers [1; 2]); MT.any_tagged_immediate] ]
      (Answer
         (variant None
            (MT.bottom naked_immediate)
            (Some (own 1))
            [ 0, [MT.any_tagged_immediate];
              1, [tagged (numbers [1; 2]); MT.any_tagged_immediate] ]));
    statement "immediate_and_block"
      "an immediate and a block (a variable is made for Is_int)"
      [constant 0; block 0 [MT.any_tagged_immediate]]
      (Answer
         (variant
            (Some (own 1))
            (MT.this_naked_immediate (z 0))
            None
            [0, [MT.any_tagged_immediate]]));
    statement "variants" "two variants"
      [ variant None
          (MT.this_naked_immediate (z 0))
          None
          [ 0, [tagged (numbers [1; 2])];
            1, [tagged (numbers [3; 4]); MT.any_tagged_immediate] ];
        variant None
          (MT.these_naked_immediates (numbers [0; 1]))
          None
          [1, [tagged (numbers [5; 6]); MT.any_tagged_immediate]] ]
      (Answer
         (variant None
            (MT.these_naked_immediates (numbers [0; 1]))
            None
            [ 0, [tagged (numbers [1; 2])];
              1, [tagged (numbers [3; 4; 5; 6]); MT.any_tagged_immediate] ]));
    statement "closures" "two closures of one code"
      [ closure [code] (tagged (numbers [1; 2]));
        closure [code] (tagged (numbers [3; 4])) ]
      (Answer (closure [code] (tagged (numbers [3; 4; 1; 2]))));
    statement "closures_of_two_codes" "two closures of two codes"
      [ closure [code] (tagged (numbers [1; 2]));
        closure [other_code] (tagged (numbers [3; 4])) ]
      (Answer
         (MT.exactly_this_closure function_slot
            [function_slot, MT.Or_unknown.Unknown]
            []
            [value_slot, tagged (numbers [3; 4; 1; 2])]
            Heap));
    statement "closures_one_without_function_type"
      "a closure with a function type, and the closures of at least a value \
       slot"
      [ closure [code] (tagged (numbers [1; 2]));
        MT.at_least_this_closure function_slot [] []
          [value_slot, tagged (numbers [3; 4])]
          Heap ]
      (Answer
         (MT.at_least_this_closure function_slot [] []
            [value_slot, tagged (numbers [3; 4; 1; 2])]
            Heap));
    statement "alias_of_another_kind"
      "a tagged immediate and an alias of kind value to a naked immediate \
       (T5b-3)"
      [MT.any_tagged_immediate; alias_to_a_naked_immediate]
      Thought_to_stop;
    statement "closure_types_of_another_kind"
      "two closures whose closure types join to a type of another kind"
      [ MT.exactly_this_closure function_slot (function_types [code])
          [function_slot, alias_to_a_naked_immediate]
          [] Heap;
        MT.exactly_this_closure function_slot (function_types [code])
          [function_slot, MT.unknown Value]
          [] Heap ]
      Thought_to_stop ]

(* What the compiler does on the inputs of an example. *)
type the_compiler =
  | Answers of joined
  | Stops of string
  | Is_not_given of string  (** The inputs are not types of the compiler. *)
  | Is_not_read of string  (** The answer is not read: a failure of the test. *)

let run_examples_of_joins (names : names) =
  Format.printf
    "@.Examples of the join of types, written down from a transcription: the \
     n-way join of the compiler on the same types@.";
  let text t = print_with_own_variables names [t] in
  let examples = examples_of_joins names in
  let as_written = ref 0 in
  let as_written_with_variables = ref 0 in
  let stops_as_thought = ref 0 in
  let shown = ref 0 in
  let not_run = ref 0 in
  let different = ref 0 in
  List.iter
    (fun { example; source; knowing; inputs; answer } ->
      let example =
        match source with
        | Statement name -> Printf.sprintf "%s [%s]" example name
        | Statement_with_another_answer name ->
          Printf.sprintf "%s [the inputs of %s, another answer]" example name
        | Not_from_the_file -> Printf.sprintf "%s [not from the file]" example
      in
      (* What the compiler does: it answers, it stops, or the types are not the
         compiler's. *)
      let result =
        match
          ( List.map (to_compiler names) inputs,
            List.map
              (fun (variable, t) ->
                let variable =
                  Converter.variable_of_model names.numbering variable
                in
                Name.var variable, to_compiler names t)
              knowing )
        with
        | exception Converter.Not_in_the_compiler what -> Is_not_given what
        | types, known -> (
          let kind = MT.kind (List.hd inputs) in
          match
            or_fatal_error (fun () ->
                match join names n_way ~known:(fun _ -> known) kind types with
                | joined -> Answers joined
                | exception Assert_failure (file, _, _) ->
                  Stops ("an assertion of " ^ Filename.basename file ^ " fails")
                | exception Converter.Not_in_the_model what ->
                  Is_not_read ("the answer is not in the model: " ^ what)
                | exception Not_defined what ->
                  Is_not_read
                    (what
                   ^ " is in the answer and the environment does not define it"
                    ))
          with
          | Ok result -> result
          | Error line -> Stops ("a fatal error: " ^ general line))
      in
      match result, answer with
      | Is_not_read why, (Answer _ | Thought_to_stop | Nothing_to_compare _) ->
        incr different;
        Format.printf "  %s: DIFFERENT. %s@." example why
      | Is_not_given what, (Answer _ | Thought_to_stop | Nothing_to_compare _)
        ->
        incr not_run;
        Format.printf "  %s: not run, not in the compiler: %s@." example what
      | Stops why, Thought_to_stop ->
        incr stops_as_thought;
        Format.printf
          "  %s: the compiler stops, where the transcription has no answer: \
           %s@."
          example why
      | Stops why, (Answer _ | Nothing_to_compare _) ->
        incr different;
        Format.printf "  %s: DIFFERENT. The compiler stops: %s@." example why
      | Answers joined, Thought_to_stop ->
        incr different;
        Format.printf
          "  @[<v>%s: DIFFERENT. The compiler was thought to stop, and \
           answers:@ %s@]@."
          example
          (print_joined names joined)
      | Answers joined, Nothing_to_compare why ->
        incr shown;
        Format.printf
          "  @[<v>%s: not compared; the compiler answers@ %s@ (%s)@]@." example
          (print_joined names joined)
          why
      | Answers joined, Answer answer ->
        if
          Model_type.same joined.joined answer
          && match joined.own with [] -> true | _ :: _ -> false
        then (
          incr as_written;
          Format.printf "  %s: as written down@." example)
        else if same_text (text joined.joined) (text answer)
        then (
          incr as_written_with_variables;
          Format.printf
            "  @[<v>%s: as written down, with the compiler's variables:@ %s@]@."
            example
            (print_joined names joined))
        else (
          incr different;
          Format.printf
            "  @[<v>%s: DIFFERENT. Written down:@ %s@ the compiler answers:@ \
             %s@]@."
            example (text answer)
            (print_joined names joined)))
    examples;
  Format.printf
    "  Of %d: %d as written down, and %d more with the compiler's variables \
     for those written down; %d where the compiler stops and the transcription \
     has no answer; %d not compared; %d not run; %d different@."
    (List.length examples) !as_written !as_written_with_variables
    !stops_as_thought !shown !not_run !different

(* ======================================================================== *)
(* Examples of the meet                                                     *)
(* ======================================================================== *)

(* Meets whose answers were written down from a transcription of the meet into
   the model, without running the compiler: scenarios made by hand from the
   calls of the types library that Simplify makes for a match on a variant, a
   match on the tag of a block, a match on a value that may be null, two loads
   from a pair, the boxing and unboxing of a float, and two projections of value
   slots (FlambdaMeetMatchExamplesSpec.v, FlambdaMeetTagExamplesSpec.v,
   FlambdaMeetProjectionExamplesSpec.v and three statements of
   FlambdaMeetExamplesSpec.v, in middle_end/flambda2/rocq/types/). Each is run
   here, with each of the two meets of the compiler: the environment is built by
   the same calls, the meet is called on the two types of the statement, and
   what is compared with the statement is whether the meet is bottom, the type
   answered, the type that the environment answered has for the variables that
   the statement names ([Typing_env.find]), and their canonical elements.

   An example is named as its statement is, without [_statement]. The statements
   are about two meets of the model, the one that transcribes the compiler's and
   one that is to be proved, which answer alike but at one step: the two
   examples that are marked say what the statement says of the first. Where the
   meet of the model calls a join, the statement is with a join that forgets;
   what is compared here is what such a statement says of the type answered and
   of the variables.

   The variables of a scenario are numbered from 1 in the order given, as in the
   statements, and so are its two function slots and its two value slots; the
   shapes are made by the model's creation functions and converted. The
   relations of [Is_int], [Get_tag] and [Is_null] are added with the functions
   of the library that Simplify calls ([Flambda2_types.Typing_env]), which are
   of the interface that the library exports and take the meet from the flag of
   the compiler: the environment is built through that interface as far as the
   last relation, and read through the observer of typing environments from
   there.

   A difference does not fail the test: it is shown, for the owner of the
   transcription to say which side is wrong. *)
type step =
  | Define of int  (** The variable of that number is defined. *)
  | Relation of
      { relation : MT.Relation.t;
        result : int;
        scrutinee : int
      }
  | Equation of int * MT.flambda_type
  | Meet_of of MT.flambda_type * MT.flambda_type
      (** The environment that the meet answers is the one to go on in. *)

type stated =
  | Is_bottom
  | Answers_type of MT.flambda_type

type example_of_meet =
  { statement : string;
    (* The kinds of the variables 1, 2, ... *)
    variables : MK.flambda_kind list;
    steps : step list;
    (* The meet that the statement is about, in the environment of the steps;
       [None] where the statement is about the steps alone. *)
    meet : (MT.flambda_type * MT.flambda_type * stated) option;
    (* What the statement says of the environment answered. *)
    finds : (int * MK.flambda_kind * MT.flambda_type) list;
    canonicals : (int * M.simple) list
  }

let examples_of_meets : example_of_meet list =
  let v = N.positive_of_int in
  let z = N.z_of_int in
  let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate in
  let naked_float : MK.flambda_kind = Naked_number Kind_naked_float in
  let alias kind i = MT.alias_type_of kind (Var (v i)) in
  let number i = MT.this_naked_immediate (z i) in
  let constant i : M.simple = Const (Naked_immediate (z i)) in
  let no_blocks = MT.Or_unknown.Known MT.row_like_for_blocks_bottom in
  let no_immediates = MT.Or_unknown.Known (MT.bottom naked_immediate) in
  (* The blocks of tag 0 that have at least the fields given. *)
  let blocks fields : MT.row_like_for_blocks MT.Or_unknown.t =
    Known
      (Row_like_for_blocks
         ( [ ( z 0,
               Known
                 { case_maps_to = fields;
                   case_index =
                     { index_domain =
                         At_least (N.nat_of_int (List.length fields));
                       index_shape = Scannable Value_only
                     };
                   case_env_extension = MT.empty_env_extension
                 } ) ],
           Bottom,
           Heap_or_local ))
  in
  (* The blocks of tag 0, of which nothing more is known. *)
  let blocks_of_tag_0 : MT.row_like_for_blocks MT.Or_unknown.t =
    Known (Row_like_for_blocks ([z 0, Unknown], Bottom, Heap_or_local))
  in
  let value non_null is_null : MT.flambda_type =
    Type_value (Ok (No_alias (Head_of_kind_value (non_null, is_null))))
  in
  let variant ?is_int ?get_tag immediates blocks : MT.flambda_type =
    value
      (Ok (Variant (is_int, immediates, get_tag, blocks, No_extensions, false)))
      Not_null
  in
  (* The shape of a load of the field at [n - 1] of a block of tag 0 into a
     variable. *)
  let load_shape n result =
    match
      MT.immutable_block_with_size_at_least
        (Known (z 0))
        (N.nat_of_int n) (Scannable Value_only) (v result)
    with
    | Some shape -> shape
    | None -> MT.bottom Value
  in
  let example statement variables steps ?meet ?(finds = []) ?(canonicals = [])
      () =
    { statement; variables; steps; meet; finds; canonicals }
  in
  (* A scrutinee (1) and the result of a relation to it (2). *)
  let related relation =
    [Define 1; Define 2; Relation { relation; result = 2; scrutinee = 1 }]
  in
  let result = alias naked_immediate 2 in
  let x = alias Value 1 in
  (* The match: the parameter (1), the result of [Is_int] (2), the result of
     [Untag_immediate] (3) and the result of the load (4). *)
  let of_the_match = [MK.Value; naked_immediate; naked_immediate; MK.Value] in
  let match_variant = variant ~is_int:(v 2) in
  let start = related Is_int in
  let untagged = alias naked_immediate 3 in
  let in_the_immediates = start @ [Meet_of (result, number 1)] in
  let in_the_blocks = start @ [Meet_of (result, number 0)] in
  let untag = MT.tagged_immediate_alias_to (v 3) in
  let after_untag = in_the_immediates @ [Define 3; Meet_of (x, untag)] in
  let x_untagged = match_variant (Known untagged) no_blocks in
  (* The tag: the parameter (1), the result of [Get_tag] (2) and the result of
     the load (3). *)
  let of_the_tag = [MK.Value; naked_immediate; MK.Value] in
  let tags_variant = variant ~get_tag:(v 2) in
  (* Null: the parameter (1) and the result of [Is_null] (2). *)
  let of_null = [MK.Value; naked_immediate] in
  let maybe_null : MT.is_null = Maybe_null (Some (v 2)) in
  (* The pair: the parameter (1) and the results of the two loads (2, 3). *)
  let of_the_pair = [MK.Value; MK.Value; MK.Value] in
  let pair fields = variant no_immediates (blocks fields) in
  (* The float: the sum (1), its box (2), the result of the unboxing (3). *)
  let of_the_float = [naked_float; MK.Value; naked_float] in
  let box_type = MT.box_number Boxable_naked_float (alias naked_float 1) Heap in
  let boxed = [Define 1; Define 2; Equation (2, box_type)] in
  (* The closure (1) and the results of the two projections (2, 3). *)
  let of_the_closure = [MK.Value; MK.Value; MK.Value] in
  let slot_shape value_slot result =
    MT.closure_with_at_least_this_value_slot (v 1) (v value_slot) (v result)
      Value
  in
  (* A closure at the first function slot of a set that has at least the two, in
     which the closure at the second is a number. *)
  let closures_type n =
    MT.at_least_this_closure (v 1)
      [v 1, Unknown; v 2, Unknown]
      [v 2, MT.this_tagged_immediate (z n)]
      [] Heap
  in
  [ example "match_is_int" of_the_match start
      ~finds:
        [ 2, naked_immediate, MT.is_int_for_scrutinee (Name_var (v 1));
          1, Value, match_variant Unknown Unknown ]
      ();
    example "match_arm_immediate" of_the_match start
      ~meet:(result, number 1, Answers_type result)
      ~canonicals:[2, constant 1]
      ~finds:[1, Value, match_variant Unknown no_blocks]
      ();
    example "match_arm_block" of_the_match start
      ~meet:(result, number 0, Answers_type result)
      ~canonicals:[2, constant 0]
      ~finds:[1, Value, match_variant no_immediates Unknown]
      ();
    example "match_no_other_arm" of_the_match start
      ~meet:(result, number 2, Is_bottom)
      ();
    example "match_untag" of_the_match
      (in_the_immediates @ [Define 3])
      ~meet:(x, untag, Answers_type x)
      ~finds:
        [1, Value, x_untagged; 3, naked_immediate, MT.unknown naked_immediate]
      ();
    example "match_arm_constructor" of_the_match after_untag
      ~meet:(untagged, number 0, Answers_type untagged)
      ~canonicals:[3, constant 0]
      ~finds:[1, Value, x_untagged]
      ();
    example "match_inner_arm" of_the_match
      (after_untag @ [Meet_of (untagged, number 0)])
      ~meet:(untagged, number 1, Is_bottom)
      ();
    example "match_load" of_the_match
      (in_the_blocks @ [Define 4])
      ~meet:(x, load_shape 1 4, Answers_type x)
      ~finds:
        [ 1, Value, match_variant no_immediates (blocks [alias Value 4]);
          4, Value, MT.unknown Value ]
      ();
    example "match_load_of_immediate" of_the_match
      (in_the_immediates @ [Define 4])
      ~meet:(x, load_shape 1 4, Is_bottom)
      ();
    example "example_get_tag_env_made" of_the_tag (related Get_tag)
      ~finds:
        [ 2, naked_immediate, MT.get_tag_for_block (Name_var (v 1));
          1, Value, tags_variant Unknown Unknown ]
      ();
    example "tags_arm, of the meet that transcribes the compiler's" of_the_tag
      (related Get_tag)
      ~meet:(result, number 0, Answers_type result)
      ~finds:[1, Value, tags_variant no_immediates blocks_of_tag_0]
      ();
    example "tags_load, of the meet that transcribes the compiler's" of_the_tag
      (related Get_tag @ [Meet_of (result, number 0); Define 3])
      ~meet:(x, load_shape 1 3, Answers_type x)
      ~finds:[1, Value, tags_variant no_immediates (blocks [alias Value 3])]
      ();
    example "null_relation" of_null (related Is_null)
      ~finds:
        [ 2, naked_immediate, MT.is_null_for_scrutinee (Name_var (v 1));
          1, Value, value Unknown maybe_null ]
      ();
    example "null_arm_null" of_null (related Is_null)
      ~meet:(result, number 1, Answers_type result)
      ~canonicals:[2, constant 1]
      ~finds:[1, Value, value Bottom maybe_null]
      ();
    example "null_arm_value" of_null (related Is_null)
      ~meet:(result, number 0, Answers_type result)
      ~canonicals:[2, constant 0]
      ~finds:[1, Value, MT.any_non_null_value]
      ();
    example "meet_through_poison" [MK.Value]
      [Define 1; Equation (1, MT.null)]
      ~meet:(x, MT.any_non_null_value, Is_bottom)
      ();
    example "meet_closures_invalid_statics" [] []
      ~meet:(closures_type 0, closures_type 1, Is_bottom)
      ();
    example "pair_first_load" of_the_pair [Define 1; Define 2]
      ~meet:(x, load_shape 1 2, Answers_type x)
      ~finds:[1, Value, pair [alias Value 2]]
      ();
    example "pair_second_load" of_the_pair
      [Define 1; Define 2; Meet_of (x, load_shape 1 2); Define 3]
      ~meet:(x, load_shape 2 3, Answers_type x)
      ~finds:[1, Value, pair [alias Value 2; alias Value 3]]
      ();
    example "float_boxed" of_the_float boxed ~finds:[2, Value, box_type] ();
    example "float_unboxed" of_the_float (boxed @ [Define 3])
      ~meet:
        ( alias Value 2,
          MT.boxed_number_alias_to Boxable_naked_float (v 3) Heap_or_local,
          Answers_type (alias Value 2) )
      ~canonicals:[3, Var (v 1)]
      ~finds:[2, Value, box_type]
      ();
    example "closure_first_slot" of_the_closure [Define 1; Define 2]
      ~meet:(x, slot_shape 1 2, Answers_type x)
      ~finds:[1, Value, slot_shape 1 2]
      ();
    example "closure_second_slot" of_the_closure
      [Define 1; Define 2; Meet_of (x, slot_shape 1 2); Define 3]
      ~meet:(x, slot_shape 2 3, Answers_type x)
      ~finds:
        [ ( 1,
            Value,
            MT.closure_with_at_least_these_value_slots (v 1)
              [v 2, (v 3, Value); v 1, (v 2, Value)] ) ]
      () ]

(* The names of a scenario, and the environment that its steps make with one of
   the two meets. *)
type scenario =
  { numbering : Model_names.t;
    variable : int -> Variable.t;
    after_the_steps : (TE.t, string) result
  }

let scenario ?(made = fun (_ : int -> Variable.t) -> ())
    (implementation : implementation) (kinds : MK.flambda_kind list)
    (steps : step list) : scenario =
  let numbering = Model_names.create () in
  let unit = Current_unit.get_cu_exn () in
  let variables =
    List.map
      (fun kind ->
        let variable = Variable.create "v" (Converter.kind_of_model kind) in
        ignore (numbering.variables.number variable : N.positive);
        variable)
      kinds
  in
  List.iter
    (fun name ->
      ignore
        (numbering.function_slots.number
           (Function_slot.create unit ~name ~size:2)
          : N.positive))
    ["f"; "g"];
  List.iter
    (fun name ->
      ignore
        (numbering.value_slots.number
           (Value_slot.create unit ~name ~is_always_immediate:false K.value)
          : N.positive))
    ["a"; "b"];
  let variable i = List.nth variables (i - 1) in
  (* The variables are told before any step is run: a step may stop. *)
  made variable;
  let bound i =
    Bound_name.create_var
      (Bound_var.create (variable i) Flambda_debug_uid.none Name_mode.normal)
  in
  let to_compiler = Converter.type_of_model numbering machine_width in
  let meet_expanded_head = implementation.meet_expanded_head in
  (* The steps as far as the last relation, through the interface that the
     library exports, with the meet of the implementation chosen by the flag
     that the library reads. *)
  let module Exported = Flambda2_types.Typing_env in
  let rec exported env steps =
    match steps with
    | Define i :: steps ->
      exported
        (Exported.add_definition env (bound i) (Variable.kind (variable i)))
        steps
    | Relation { relation; result; scrutinee } :: steps ->
      let add =
        match relation with
        | Is_int -> Exported.add_is_int_relation
        | Get_tag -> Exported.add_get_tag_relation
        | Is_null -> Exported.add_is_null_relation
      in
      exported
        (add env
           (Name.var (variable result))
           ~scrutinee:(Simple.var (variable scrutinee)))
        steps
    | (Equation _ | Meet_of _) :: _ | [] -> env, steps
  in
  let algorithm : Oxcaml_flags.join_algorithm =
    if String.equal implementation.implementation n_way.implementation
    then N_way
    else Binary
  in
  let before = !Oxcaml_flags.Flambda2.join_algorithm in
  Oxcaml_flags.Flambda2.join_algorithm := Oxcaml_flags.Set algorithm;
  let env, steps =
    Fun.protect
      ~finally:(fun () -> Oxcaml_flags.Flambda2.join_algorithm := before)
      (fun () ->
        exported
          (Exported.create ~machine_width ~resolver:(fun _ -> None))
          steps)
  in
  (* The same environment, as the modules of the library have it. *)
  let observed = ref None in
  TE.For_testing.set_observer
    (Some (fun env (_ : Continuation.t) -> observed := Some env));
  Exported.notify_test_observer env ~at_handler:(Continuation.create ());
  TE.For_testing.set_observer None;
  let env = match !observed with Some env -> env | None -> assert false in
  let rec go env steps : (TE.t, string) result =
    match steps with
    | [] -> Ok env
    | Define i :: steps ->
      go (TE.add_definition env (bound i) (Variable.kind (variable i))) steps
    | Equation (i, t) :: steps ->
      go
        (ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
             ME.add_equation env
               (Name.var (variable i))
               (to_compiler t) ~meet_expanded_head))
        steps
    | Meet_of (t1, t2) :: steps -> (
      match implementation.meet env (to_compiler t1) (to_compiler t2) with
      | Bottom -> Error "a meet on the way to the environment is bottom"
      | Ok ((_ : TG.t), env) -> go env steps)
    | Relation _ :: _ ->
      Error "a relation after a step that is not a definition"
  in
  { numbering; variable; after_the_steps = go env steps }

(* What differs from the statement when an example is run with one of the two
   meets; nothing if the list is empty. *)
let run_example_of_meet (implementation : implementation)
    (example : example_of_meet) : string list =
  let { numbering; variable; after_the_steps } =
    scenario implementation example.variables example.steps
  in
  let to_compiler = Converter.type_of_model numbering machine_width in
  let to_model = Converter.type_to_model numbering in
  let differences = ref [] in
  let differs fmt =
    Format.kasprintf (fun what -> differences := what :: !differences) fmt
  in
  let print ppf t = Model_type.print ppf (Model_type.normalise t) in
  let check env =
    List.iter
      (fun (i, kind, stated) ->
        let found =
          to_model
            (TE.find env
               (Name.var (variable i))
               (Some (Converter.kind_of_model kind)))
        in
        if not (Model_type.same found stated)
        then
          differs "@[<v>the type of v%d is@ %a@ and is stated to be@ %a@]" i
            print found print stated)
      example.finds;
    List.iter
      (fun (i, stated) ->
        match
          TE.get_canonical_simple_exn env ~min_name_mode:Name_mode.in_types
            (Simple.var (variable i))
        with
        | canonical ->
          let found = Converter.simple_to_model numbering canonical in
          if not (Stdlib.( = ) found stated)
          then
            differs "the canonical element of v%d is %a and is stated to be %a"
              i Model_type.print_simple found Model_type.print_simple stated
        | exception Not_found -> differs "v%d has no canonical element" i)
      example.canonicals
  in
  (match after_the_steps with
  | Error why -> differs "%s" why
  | Ok env -> (
    match example.meet with
    | None -> check env
    | Some (t1, t2, stated) -> (
      match
        implementation.meet env (to_compiler t1) (to_compiler t2), stated
      with
      | Bottom, Is_bottom -> ()
      | Bottom, Answers_type _ -> differs "the meet is bottom"
      | Ok (t, (_ : TE.t)), Is_bottom ->
        differs "@[<v>the meet is stated to be bottom, and answers@ %a@]" print
          (to_model t)
      | Ok (t, env), Answers_type stated ->
        let answered = to_model t in
        if not (Model_type.same answered stated)
        then
          differs "@[<v>the meet answers@ %a@ and is stated to answer@ %a@]"
            print answered print stated;
        check env)));
  List.rev !differences

(* What running an example with one of the two meets comes to. *)
type run_of_meet =
  | As_stated
  | Differs of string list
  | Stops_at of string
  | Is_not_run of string

let run_examples_of_meets () =
  Format.printf
    "@.Examples of the meet, written down from a transcription: the two meets \
     of the compiler on the same calls@.";
  let as_stated = ref 0 and different = ref 0 and not_run = ref 0 in
  let implementations = [binary; n_way] in
  List.iter
    (fun (example : example_of_meet) ->
      let runs =
        List.map
          (fun (implementation : implementation) ->
            let run =
              match
                or_fatal_error (fun () ->
                    match run_example_of_meet implementation example with
                    | [] -> As_stated
                    | _ :: _ as differences -> Differs differences
                    | exception Assert_failure (file, _, _) ->
                      Stops_at
                        ("an assertion of " ^ Filename.basename file ^ " fails")
                    | exception Converter.Not_in_the_compiler what ->
                      Is_not_run ("not in the compiler: " ^ what)
                    | exception Converter.Not_in_the_model what ->
                      Differs ["an answer is not in the model: " ^ what])
              with
              | Ok run -> run
              | Error line -> Stops_at ("a fatal error: " ^ general line)
            in
            (match run with
            | As_stated -> incr as_stated
            | Differs _ | Stops_at _ -> incr different
            | Is_not_run _ -> incr not_run);
            implementation.implementation, run)
          implementations
      in
      if
        List.for_all
          (fun (_, run) ->
            match run with
            | As_stated -> true
            | Differs _ | Stops_at _ | Is_not_run _ -> false)
          runs
      then
        Format.printf "  %s: as stated, with each of the two meets@."
          example.statement
      else
        List.iter
          (fun (implementation, run) ->
            let name =
              Printf.sprintf "%s (%s)" example.statement implementation
            in
            match run with
            | As_stated -> Format.printf "  %s: as stated@." name
            | Differs differences ->
              Format.printf "  @[<v>%s: DIFFERENT.%a@]@." name
                (fun ppf ->
                  List.iter (fun what ->
                      Format.fprintf ppf "@ %a" print_lines what))
                differences
            | Stops_at why ->
              Format.printf "  %s: DIFFERENT. The compiler stops: %s@." name why
            | Is_not_run why -> Format.printf "  %s: not run, %s@." name why)
          runs)
    examples_of_meets;
  Format.printf
    "  Of %d statements, each with the two meets: %d runs as stated, %d \
     different, %d not run@."
    (List.length examples_of_meets)
    !as_stated !different !not_run

(* ======================================================================== *)
(* Chosen meets                                                             *)
(* ======================================================================== *)

(* Two meets of types written by hand, each in an environment made as those of
   the examples above are, which answer what the meets of generated types above
   were not seen to answer. What each of the two meets of the compiler does is
   shown; nothing is compared, and the expected output is the record.

   - T7-2. The variable 1 is a value, the variable 3 is the tagging of the
   variable 2, and the variable 2 is [Get_tag] of the variable 1. The types met
   are the blocks of tag 0 or 1 whose one field is the variable 3, and the
   blocks of tag 0 whose field is the number 0 or of tag 1 whose field is the
   number 1. Neither has an extension. In each case of the answer the field is a
   number, so the variable 2 is, and the variable 1 is a block of that tag: the
   extension of the case says so. Given to the variable 1, the answer is then a
   type of a name that has an extension on that name, on which the compiler's
   [Equal_types_for_debug.equal_type] does not end; that function is not called
   here. - T4-9. The three variables are values of which nothing is known. The
   types met are blocks of one field: a variant whose blocks, of tag 0 or 1,
   have for their field the variable 1; and a variant whose blocks have for
   their field a pair, of the variable 2 twice for tag 0 and of the variable 3
   twice for tag 1. The answer of the n-way meet mentions a variable that the
   environment answered with it does not define. *)
type chosen_meet =
  { label : string;
    kinds : MK.flambda_kind list;
    before : step list;
    met : MT.flambda_type * MT.flambda_type;
    (* The variable that is then given the answer, by an equation, and whose
       type is read. *)
    given_to : int option
  }

let the_chosen_meets : chosen_meet list =
  let v = N.positive_of_int in
  let z = N.z_of_int in
  let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate in
  let alias i = MT.alias_type_of Value (Var (v i)) in
  let variant immediates cases : MT.flambda_type =
    Type_value
      (Ok
         (No_alias
            (Head_of_kind_value
               ( Ok
                   (Variant
                      ( None,
                        immediates,
                        None,
                        Known
                          (Row_like_for_blocks
                             ( List.map
                                 (fun (tag, fields) :
                                      (_
                                      * MT.row_like_block_case MT.Or_unknown.t)
                                    ->
                                   ( z tag,
                                     Known
                                       { case_maps_to = fields;
                                         case_index =
                                           { index_domain =
                                               Known
                                                 (N.nat_of_int
                                                    (List.length fields));
                                             index_shape = Scannable Value_only
                                           };
                                         case_env_extension =
                                           MT.empty_env_extension
                                       } ))
                                 cases,
                               Bottom,
                               Heap )),
                        No_extensions,
                        false )),
                 Not_null ))))
  in
  let blocks = variant (Known (MT.bottom naked_immediate)) in
  let immediates_or_blocks = variant Unknown in
  let number i = MT.this_tagged_immediate (z i) in
  [ { label = "T7-2";
      kinds = [Value; naked_immediate; Value];
      before =
        [ Define 1;
          Define 2;
          Define 3;
          Equation (3, MT.tagged_immediate_alias_to (v 2));
          Equation (2, MT.get_tag_for_block (Name_var (v 1))) ];
      met =
        ( blocks [0, [alias 3]; 1, [alias 3]],
          blocks [0, [number 0]; 1, [number 1]] );
      given_to = Some 1
    };
    { label = "T4-9";
      kinds = [Value; Value; Value];
      before = [Define 1; Define 2; Define 3];
      met =
        ( blocks [0, [immediates_or_blocks [0, [alias 1]; 1, [alias 1]]]],
          blocks
            [ ( 0,
                [ immediates_or_blocks
                    [ 0, [blocks [0, [alias 2; alias 2]]];
                      1, [blocks [0, [alias 3; alias 3]]] ] ] ) ] );
      given_to = None
    } ]

(* Whether a type is, or has within it, a type that [part] accepts, or has, in
   an extension, an equation on a name that [equation_on] accepts. *)
let rec has_a_part ~(part : MT.flambda_type -> bool)
    ~(equation_on : MT.name -> bool) (t : MT.flambda_type) : bool =
  let within = has_a_part ~part ~equation_on in
  let in_extension (Env_extension equations : MT.env_extension) =
    List.exists (fun (name, t) -> equation_on name || within t) equations
  in
  let in_block_case (case : MT.row_like_block_case) =
    in_extension case.case_env_extension || List.exists within case.case_maps_to
  in
  part t
  ||
  match t with
  | Type_value (Ok (No_alias (Head_of_kind_value (Ok head, _)))) -> (
    match head with
    | Variant (_, immediates, _, blocks, extensions, _) -> (
      (match immediates with Known t -> within t | Unknown -> false)
      || (match blocks with
        | Known (Row_like_for_blocks (known, other, _)) -> (
          List.exists
            (fun ((_ : N.z), (case : _ MT.Or_unknown.t)) ->
              match case with
              | Known case -> in_block_case case
              | Unknown -> false)
            known
          || match other with Ok case -> in_block_case case | Bottom -> false)
        | Unknown -> false)
      ||
      match extensions with
      | No_extensions -> false
      | Ext (when_immediate, when_block) ->
        in_extension when_immediate || in_extension when_block)
    | Boxed_number (_, contents, _) -> within contents
    | Closures (Row_like_for_closures known, _) ->
      let in_case (case : MT.row_like_closures_case) =
        in_extension case.case_env_extension
        ||
        let (Closures_entry (_, closure_types, value_slot_types)) =
          case.case_maps_to
        in
        List.exists
          (fun ((_ : N.positive), t) -> within t)
          (closure_types @ value_slot_types)
      in
      List.exists (fun ((_ : N.positive), case) -> in_case case) known
    | Mutable_block _ | String _ -> false)
  | Type_value
      ( Unknown | Bottom
      | Ok (Equals _)
      | Ok (No_alias (Head_of_kind_value ((Unknown | Bottom), _))) )
  | Type_naked_immediate _ | Type_naked_number _ | Type_rec_info _
  | Type_region _ ->
    false

(* Whether a type has, in an extension, an equation on a name. *)
let has_an_extension_on (name : MT.name) (t : MT.flambda_type) : bool =
  has_a_part
    ~part:(fun (_ : MT.flambda_type) -> false)
    ~equation_on:(MT.name_eqb name) t

(* Whether a type is, or has within it, an alias of a symbol. *)
let has_an_alias_of_a_symbol (t : MT.flambda_type) : bool =
  has_a_part
    ~part:(fun t ->
      match MT.get_alias_opt t with
      | Some (Symbol _) -> true
      | Some (Var _ | Const _) | None -> false)
    ~equation_on:(fun (_ : MT.name) -> false)
    t

let chosen_meets () =
  Format.printf "@.Chosen meets@.";
  List.iter
    (fun { label; kinds; before; met = first, second; given_to } ->
      List.iter
        (fun (implementation : implementation) ->
          let name =
            Printf.sprintf "%s (%s)" label implementation.implementation
          in
          let observed =
            or_fatal_error (fun () ->
                let { numbering; variable; after_the_steps } =
                  scenario implementation kinds before
                in
                let to_compiler =
                  Converter.type_of_model numbering machine_width
                in
                let to_model = Converter.type_to_model numbering in
                let print_names ppf names =
                  Format.pp_print_list
                    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf ", ")
                    Model_type.print_name ppf names
                in
                match after_the_steps with
                | Error why -> ["not run: " ^ why]
                | Ok env -> (
                  match
                    implementation.meet env (to_compiler first)
                      (to_compiler second)
                  with
                  | Bottom -> ["the meet is bottom"]
                  | Ok (t, env) ->
                    let not_defined =
                      Name_occurrences.fold_names (TG.free_names t) ~init:[]
                        ~f:(fun not_defined name ->
                          if TE.mem env name
                          then not_defined
                          else
                            Converter.name_to_model numbering name
                            :: not_defined)
                    in
                    let answered = to_model t in
                    let name_of i : MT.name =
                      Name_var (numbering.variables.number (variable i))
                    in
                    let with_equations =
                      List.filter
                        (fun name -> has_an_extension_on name answered)
                        (List.mapi (fun i _ -> name_of (i + 1)) kinds)
                    in
                    let given =
                      match given_to with
                      | None -> []
                      | Some i ->
                        let x = variable i in
                        let meet_expanded_head =
                          implementation.meet_expanded_head
                        in
                        let env =
                          ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
                              ME.add_equation env (Name.var x) t
                                ~meet_expanded_head)
                        in
                        let of_x =
                          to_model
                            (TE.find env (Name.var x) (Some (Variable.kind x)))
                        in
                        [ Format.asprintf
                            "given the answer by an equation, %a has a type \
                             that has %s extension with an equation on %a"
                            Model_type.print_name (name_of i)
                            (if has_an_extension_on (name_of i) of_x
                             then "an"
                             else "no")
                            Model_type.print_name (name_of i) ]
                    in
                    [ (match with_equations with
                      | [] ->
                        "no extension of the answer has an equation on a \
                         variable of the environment"
                      | _ :: _ ->
                        Format.asprintf
                          "extensions of the answer have equations on %a"
                          print_names with_equations);
                      (match not_defined with
                      | [] ->
                        "the environment answered defines every name of the \
                         answer"
                      | _ :: _ ->
                        Format.asprintf
                          "the answer mentions %a, which the environment \
                           answered does not define"
                          print_names (List.rev not_defined));
                      Printf.sprintf "type_well_formed is %b of the answer"
                        (is_well_formed answered) ]
                    @ given))
          in
          match observed with
          | Ok observed ->
            Format.printf "  @[<v>%s:%a@]@." name
              (fun ppf ->
                List.iter (fun what -> Format.fprintf ppf "@ %s" what))
              observed
          | Error line ->
            Format.printf "  %s: the compiler stops: %s@." name (general line))
        [binary; n_way])
    the_chosen_meets

(* ======================================================================== *)
(* The meet of the model beside the compiler's                              *)
(* ======================================================================== *)

(* The transcription of the meet (FlambdaMeet.v), given the transcription of the
   n-way join for the join that it calls (FlambdaJoinMeet.v, in
   middle_end/flambda2/rocq/types/), is extracted, and is run here beside the
   n-way meet of the compiler: on the scenarios of the examples of the meet and
   of the chosen meets, where it is the function that the statements are about
   that is run and not an answer copied from them, and on generated pairs of
   types.

   The model is run with the choices that are the compiler's, with the depth of
   joins that the compiler reads from its flag, and with fuel. It answers
   nothing where the compiler stops with a fatal error, where the fuel is too
   little, and at one state where the compiler goes on (T4-10).

   What is compared is what can be seen of the two answers: whether the meet is
   bottom, the type answered, and of the environment answered the type and the
   canonical element of every name that the scenario defines, and the type of
   every other variable that those types mention (the variables that the meet
   and the joins inside it made). The two sides do not name such variables
   alike: the two are compared as printed, with those variables renamed in the
   order in which they come. *)
module Model_env = Flambda2_rocq_semantics.FlambdaTypingEnv.Typing_env
module Model_meet = Flambda2_rocq_semantics.FlambdaMeet
module Model_join_meet = Flambda2_rocq_semantics.FlambdaJoinMeet
module Model_join_env = Flambda2_rocq_semantics.FlambdaJoinEnv.Join_env
module Model_name_mode = Flambda2_rocq_semantics.FlambdaBindingTime.Name_mode
module Model_join_spec = Flambda2_rocq_semantics.FlambdaJoinSpec
module Model_analysis = Flambda2_rocq_semantics.FlambdaJoinEnv.Analysis

type model =
  { meet_of_model :
      Model_env.t ->
      MT.flambda_type ->
      MT.flambda_type ->
      (MT.flambda_type * Model_env.t) MT.Or_bottom.t option;
    add_equation_of_model :
      Model_env.t -> MT.name -> MT.flambda_type -> Model_env.t option;
    add_relation_of_model :
      MT.Relation.t -> Model_env.t -> MT.name -> M.simple -> Model_env.t option;
    (* The join of environments, for the tests of the join below. *)
    (* The uses are given their positions for names. *)
    join_of_model : int Model_join_spec.join_of_environments
  }

(* The choices of the model: the compiler's, or those of the meet and of the
   join of which the contracts are to be proved. *)
type choices =
  | Of_the_compiler
  | To_be_proved
  | With_the_repaired_test_of_the_join

let the_model (choices : choices) : model =
  let fuel = N.nat_of_int 200 in
  let fuels : Model_join_meet.fuels =
    { fuel_of_env = fuel; fuel_of_types = fuel; fuel_of_meet = fuel }
  in
  let join_depth = N.nat_of_int (Flambda2_ui.Flambda_features.join_depth ()) in
  let join_choices, meet_choices =
    match choices with
    | Of_the_compiler ->
      Model_join_env.compiler_choices, Model_meet.compiler_choices
    | To_be_proved ->
      Model_join_env.repaired_choices, Model_meet.repaired_choices
    | With_the_repaired_test_of_the_join ->
      Model_join_env.repaired_choices, Model_meet.compiler_choices
  in
  let join =
    Model_join_meet.model_join_scoped_envs join_choices join_depth fuels
  in
  { meet_of_model =
      Model_join_meet.model_meet join_choices meet_choices join_depth fuels;
    add_equation_of_model =
      Model_join_meet.model_add_equation join_choices meet_choices join_depth
        fuels;
    add_relation_of_model =
      (fun relation ->
        let add =
          match relation with
          | Is_int -> Model_meet.Flambda2_types.add_is_int_relation
          | Get_tag -> Model_meet.Flambda2_types.add_get_tag_relation
          | Is_null -> Model_meet.Flambda2_types.add_is_null_relation
        in
        add join meet_choices fuel);
    join_of_model =
      Model_join_meet.model_join join_choices meet_choices join_depth fuels
  }

(* The environment that the steps of a scenario make in the model. *)
let scenario_in_the_model (model : model) (kinds : MK.flambda_kind list)
    (steps : step list) : (Model_env.t, string) result =
  let v = N.positive_of_int in
  let rec go env steps : (Model_env.t, string) result =
    match steps with
    | [] -> Ok env
    | Define i :: steps ->
      go
        (Model_env.add_variable_definition env (v i)
           (List.nth kinds (i - 1))
           Model_name_mode.normal)
        steps
    | Relation { relation; result; scrutinee } :: steps -> (
      match
        model.add_relation_of_model relation env
          (Name_var (v result))
          (Var (v scrutinee))
      with
      | Some env -> go env steps
      | None -> Error "no answer where a relation is added")
    | Equation (i, t) :: steps -> (
      match model.add_equation_of_model env (Name_var (v i)) t with
      | Some env -> go env steps
      | None -> Error "no answer where an equation is added")
    | Meet_of (t1, t2) :: steps -> (
      match model.meet_of_model env t1 t2 with
      | Some (Ok ((_ : MT.flambda_type), env)) -> go env steps
      | Some Bottom -> Error "a meet on the way to the environment is bottom"
      | None -> Error "no answer of a meet on the way to the environment")
  in
  go Model_env.create steps

(* What is seen of a meet, or of an environment where no meet is asked. *)
type seen =
  | Nothing_seen of string
  | Bottom_seen
  | Seen of
      { answer : MT.flambda_type option;
        (* Whether the environment is marked as one that no run reaches. *)
        env_is_bottom : bool;
        (* The type and the canonical element of each name of the scenario. *)
        of_names : (MT.name * MT.flambda_type option * M.simple option) list;
        (* The other variables that these types reach, with their types. *)
        of_others : (M.variable * MT.flambda_type option) list
      }

(* The variables numbered above [threshold] that the types reach, each with the
   type that [type_of] gives it, in the order in which they are reached. *)
let reached ~(threshold : int) ~(type_of : M.variable -> MT.flambda_type option)
    (roots : MT.flambda_type list) : (M.variable * MT.flambda_type option) list
    =
  let is_other x =
    match N.int_of_positive x with
    | Some number -> number > threshold
    | None -> true
  in
  let others_of t =
    List.fold_left
      (fun found x ->
        if is_other x && not (List.mem x found) then found @ [x] else found)
      []
      (MT.free_variables (Model_type.normalise t))
  in
  let rec close found (pending : M.variable list) =
    match pending with
    | [] -> List.rev found
    | x :: pending ->
      if List.mem_assoc x found
      then close found pending
      else
        let t = type_of x in
        close ((x, t) :: found)
          (pending @ match t with Some t -> others_of t | None -> [])
  in
  close [] (List.concat_map others_of roots)

(* What is seen of the names given, where a name that the environment does not
   define is asked nothing of. *)
let sees ~threshold ~(env_is_bottom : bool) ~(defined : MT.name -> bool)
    ~(type_of : MT.name -> MT.flambda_type option)
    ~(canonical : MT.name -> M.simple option) (names : MT.name list)
    (answer : MT.flambda_type option) : seen =
  let of_names =
    List.map
      (fun (name : MT.name) ->
        if defined name
        then name, type_of name, canonical name
        else name, None, None)
      names
  in
  let roots =
    Option.to_list answer @ List.filter_map (fun (_, t, _) -> t) of_names
  in
  Seen
    { answer;
      env_is_bottom;
      of_names;
      of_others =
        reached ~threshold
          ~type_of:(fun x ->
            let name : MT.name = Name_var x in
            if defined name then type_of name else None)
          roots
    }

exception Model_has_no_answer of string

(* What is seen in an environment of the model. *)
let seen_in_the_model ~threshold (names : MT.name list)
    (answer : MT.flambda_type option) (env : Model_env.t) : seen =
  match
    sees ~threshold ~env_is_bottom:(Model_env.is_bottom env)
      ~defined:(Model_env.mem None env)
      ~type_of:(fun name ->
        match Model_env.find env name None with
        | Some t -> Some t
        | None -> raise (Model_has_no_answer "no type for a name"))
      ~canonical:(fun name ->
        match
          Model_env.get_canonical_simple_exn env (Some Model_name_mode.in_types)
            None (MT.simple_of_name name)
        with
        | Some canonical -> canonical
        | None ->
          raise (Model_has_no_answer "no answer for a canonical element"))
      names answer
  with
  | seen -> seen
  | exception Model_has_no_answer why -> Nothing_seen why

(* What is seen in an environment of the compiler. *)
let seen_in_the_compiler (numbering : Model_names.t) ~threshold
    (names : MT.name list) (answer : TG.t option) (env : TE.t) : seen =
  let to_model = Converter.type_to_model numbering in
  let of_model = Converter.name_of_model numbering in
  sees ~threshold ~env_is_bottom:(TE.is_bottom env)
    ~defined:(fun name -> TE.mem env (of_model name))
    ~type_of:(fun name ->
      let name = of_model name in
      let kind =
        Name.pattern_match name ~var:Variable.kind ~symbol:(fun _ -> K.value)
      in
      Some (to_model (TE.find env name (Some kind))))
    ~canonical:(fun name ->
      match
        TE.get_canonical_simple_exn env ~min_name_mode:Name_mode.in_types
          (Simple.name (of_model name))
      with
      | canonical -> Some (Converter.simple_to_model numbering canonical)
      | exception Not_found -> None)
    names
    (Option.map to_model answer)

(* What is seen, as printed, with the variables numbered above the threshold
   renamed in the order in which they come. *)
let print_seen ?(own = Hashtbl.create 4) ~threshold (seen : seen) : string =
  match seen with
  | Nothing_seen why -> "no answer: " ^ why
  | Bottom_seen -> "the meet is bottom"
  | Seen { answer; env_is_bottom; of_names; of_others } ->
    (* The text, with the types in a normal form and the other variables in the
       order given. *)
    let text normalise of_others =
      let print ppf t = Model_type.print ppf (normalise t) in
      let print_type ppf (t : MT.flambda_type option) =
        match t with
        | Some t -> print ppf t
        | None -> Format.pp_print_string ppf "not defined"
      in
      Format.asprintf "@[<v>%a%s%a%a@]"
        (fun ppf (answer : MT.flambda_type option) ->
          match answer with
          | Some t -> Format.fprintf ppf "the meet answers %a" print t
          | None -> Format.pp_print_string ppf "the environment")
        answer
        (if env_is_bottom then ", in an environment that is bottom" else "")
        (fun ppf ->
          List.iter (fun (name, t, (canonical : M.simple option)) ->
              Format.fprintf ppf
                "@ %a : %a, of which the canonical element is %a"
                Model_type.print_name name print_type t
                (fun ppf (canonical : M.simple option) ->
                  match canonical with
                  | Some simple -> Model_type.print_simple ppf simple
                  | None -> Format.pp_print_string ppf "not found")
                canonical))
        of_names
        (fun ppf ->
          List.iter (fun (x, t) ->
              Format.fprintf ppf "@ where %a : %a" Model_type.print_variable x
                print_type t))
        of_others
    in
    (* The variables numbered above the threshold are named in the order in
       which a first text comes to them. In that text a set of names, or a map
       keyed by names, is written in the order of the numbers of its variables,
       which for those variables mean nothing and are not the same on the two
       sides of a comparison. So the text is written again with those variables
       in the order of their new names, which is the text answered: a set that
       has two of them is then written in one way, provided that the first text
       came to both before it came to the set.

       Where the first text comes to two of them inside one set, it has them in
       the order of what each is by itself: its type, as printed, with every
       such variable written alike. Two that are alike in that are in the order
       of their numbers, and the two sides may then name them differently: a
       difference that is none, never an agreement that is none. *)
    let by_itself : (int, string) Hashtbl.t = Hashtbl.create 4 in
    List.iter
      (fun (x, (t : MT.flambda_type option)) ->
        match N.int_of_positive x with
        | Some number ->
          Hashtbl.replace by_itself number
            (match t with
            | Some t ->
              without_the_names_above threshold
                (Format.asprintf "%a" Model_type.print (Model_type.normalise t))
            | None -> "not defined")
        | None -> ())
      of_others;
    let first_order x y =
      match N.int_of_positive x, N.int_of_positive y with
      | Some number, Some number' ->
        let made = number > threshold and made' = number' > threshold in
        if made && made'
        then
          let of_itself number =
            Option.value (Hashtbl.find_opt by_itself number) ~default:""
          in
          let c = String.compare (of_itself number) (of_itself number') in
          if c <> 0 then c else Int.compare number number'
        else if made
        then 1
        else if made'
        then -1
        else Int.compare number number'
      | Some _, None -> -1
      | None, Some _ -> 1
      | None, None -> 0
    in
    ignore
      (rename_variables_above ~own threshold
         (text (Model_type.normalise_by first_order) of_others)
        : string);
    let rank (x : M.variable) : int * int =
      match N.int_of_positive x with
      | Some number when number <= threshold -> 0, number
      | Some number -> (
        match Hashtbl.find_opt own number with
        | Some name -> 1, name
        | None -> 2, number)
      | None -> 3, 0
    in
    let order x y =
      let kind, position = rank x in
      let kind', position' = rank y in
      let c = Int.compare kind kind' in
      if c <> 0 then c else Int.compare position position'
    in
    rename_variables_above ~own threshold
      (text
         (Model_type.normalise_by order)
         (List.stable_sort (fun (x, _) (y, _) -> order x y) of_others))

(* The two say the same: both stop, both are bottom, or both see the same. *)
let same_seen ~threshold (seen : seen) (seen' : seen) : bool =
  match seen, seen' with
  | Nothing_seen _, Nothing_seen _ | Bottom_seen, Bottom_seen -> true
  | Seen _, Seen _ ->
    same_text (print_seen ~threshold seen) (print_seen ~threshold seen')
  | (Nothing_seen _ | Bottom_seen | Seen _), _ -> false

(* What is seen, with the type of a name that is an alias read as the alias of
   its canonical element. Two names of one class may each keep an alias to the
   other member or to the canonical element: where a meet adds again the
   equations of a level that it cut, the compiler adds them in the order of the
   names and the model in the order in which they were made, and an alias to a
   name that has since become an alias of a constant is kept by one and not by
   the other (T7-10). The canonical elements are compared as they are.

   An alias is read so only where what is seen says that what it is an alias to
   is of the class of the name: a constant is its own canonical element; a name
   has the canonical element of its own line, if it has one. Where that is not
   the canonical element of the name, or is not seen, the type is left as it is,
   and a difference stays one. *)
let up_to_the_alias_kept (seen : seen) : seen =
  match seen with
  | Nothing_seen _ | Bottom_seen -> seen
  | Seen { answer; env_is_bottom; of_names; of_others } ->
    let canonical_of (target : M.simple) : M.simple option =
      let of_name (name : MT.name) =
        List.find_map
          (fun (name', (_ : MT.flambda_type option), canonical) ->
            if MT.name_eqb name name' then canonical else None)
          of_names
      in
      match target with
      | Const _ -> Some target
      | Var x -> of_name (Name_var x)
      | Symbol symbol -> of_name (Name_symbol symbol)
    in
    let of_names =
      List.map
        (fun ((name, t, canonical) as entry) ->
          match t, canonical with
          | Some t, Some canonical -> (
            match Option.bind (MT.get_alias_opt t) canonical_of with
            | Some of_the_target when Stdlib.( = ) of_the_target canonical ->
              ( name,
                Some (MT.alias_type_of (MT.kind t) canonical),
                Some canonical )
            | Some (_ : M.simple) | None -> entry)
          | Some _, None | None, (Some _ | None) -> entry)
        of_names
    in
    Seen { answer; env_is_bottom; of_names; of_others }

(* What [up_to_the_alias_kept] takes for the same and what it does not, on lines
   written by hand: two naked immediates, of which the first is 2, and a third
   in two of the cases. *)
let control_of_the_alias_kept () =
  let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate in
  let v i = N.positive_of_int i in
  let variable i : M.simple = Var (v i) in
  let two : M.simple = Const (Naked_immediate (N.z_of_int 2)) in
  let three : M.simple = Const (Naked_immediate (N.z_of_int 3)) in
  let alias simple = Some (MT.alias_type_of naked_immediate simple) in
  let unknown = Some (MT.unknown naked_immediate) in
  let line i t canonical : MT.name * MT.flambda_type option * M.simple option =
    Name_var (v i), t, Some canonical
  in
  let seen ?(answer = MT.unknown Value) ?(of_others = []) of_names : seen =
    Seen { answer = Some answer; env_is_bottom = false; of_names; of_others }
  in
  let threshold = 3 in
  let same one other =
    same_seen ~threshold (up_to_the_alias_kept one) (up_to_the_alias_kept other)
  in
  let first = line 1 (alias two) two in
  let reference = seen [first; line 2 (alias two) two] in
  let made = v 4 in
  let cases =
    [ ( "an alias to the constant, and an alias to a name of which the \
         constant is the canonical element",
        same reference (seen [first; line 2 (alias (variable 1)) two]) );
      ( "the same through two names",
        same
          (seen [first; line 2 (alias two) two; line 3 (alias two) two])
          (seen
             [ first;
               line 2 (alias (variable 1)) two;
               line 3 (alias (variable 2)) two ]) );
      ( "an alias to another constant than the canonical element",
        not (same reference (seen [first; line 2 (alias three) two])) );
      ( "an alias to a name of another class",
        not
          (same
             (seen [first; line 2 (alias two) two; line 3 unknown (variable 3)])
             (seen
                [ first;
                  line 2 (alias (variable 3)) two;
                  line 3 unknown (variable 3) ])) );
      ( "an alias to a name of which nothing is seen",
        not (same reference (seen [first; line 2 (alias (Var made)) two])) );
      ( "another type answered",
        not
          (same reference
             (seen
                ~answer:(MT.this_tagged_immediate (N.z_of_int 0))
                [first; line 2 (alias two) two])) );
      ( "a type that is no alias, and another",
        not
          (same
             (seen [first; line 2 unknown (variable 2)])
             (seen
                [ first;
                  line 2
                    (Some (MT.these_naked_immediates [N.z_of_int 0]))
                    (variable 2) ])) );
      ( "a name that is defined, and not",
        not (same reference (seen [first; Name_var (v 2), None, None])) );
      ( "another canonical element",
        not
          (same
             (seen [first; line 2 (alias (variable 1)) two])
             (seen [first; line 2 (alias (variable 1)) (variable 1)])) );
      ( "another type of a variable that was made",
        let lines = [first; line 2 (alias two) two] in
        not
          (same
             (seen ~of_others:[made, unknown] lines)
             (seen ~of_others:[made, alias two] lines)) ) ]
  in
  Format.printf
    "@.What is the same up to the alias that a name keeps (%d cases): %d as \
     expected@."
    (List.length cases)
    (List.length (List.filter snd cases));
  List.iter
    (fun (name, as_expected) ->
      if not as_expected
      then (
        incr failures;
        Format.printf "  DISAGREEMENT: %s@." name))
    cases

(* A scenario and a meet in the model and in the compiler. *)
let scenario_in_both (model : model) (kinds : MK.flambda_kind list)
    (steps : step list) (meet : (MT.flambda_type * MT.flambda_type) option) :
    seen * seen =
  let threshold = List.length kinds in
  let names : MT.name list =
    List.mapi (fun i _ : MT.name -> Name_var (N.positive_of_int (i + 1))) kinds
  in
  let in_the_model =
    match scenario_in_the_model model kinds steps with
    | Error why -> Nothing_seen why
    | Ok env -> (
      match meet with
      | None -> seen_in_the_model ~threshold names None env
      | Some (t1, t2) -> (
        match model.meet_of_model env t1 t2 with
        | None -> Nothing_seen "no answer of the meet"
        | Some Bottom -> Bottom_seen
        | Some (Ok (t, env)) -> seen_in_the_model ~threshold names (Some t) env)
      )
  in
  let in_the_compiler =
    match
      or_fatal_error (fun () ->
          let { numbering; variable = _; after_the_steps } =
            scenario n_way kinds steps
          in
          let to_compiler = Converter.type_of_model numbering machine_width in
          match after_the_steps with
          | Error why -> Nothing_seen why
          | Ok env -> (
            match meet with
            | None -> seen_in_the_compiler numbering ~threshold names None env
            | Some (t1, t2) -> (
              match n_way.meet env (to_compiler t1) (to_compiler t2) with
              | Bottom -> Bottom_seen
              | Ok (t, env) ->
                seen_in_the_compiler numbering ~threshold names (Some t) env)))
    with
    | Ok seen -> seen
    | Error line -> Nothing_seen ("a fatal error: " ^ general line)
  in
  in_the_model, in_the_compiler

let scenarios_in_the_model () =
  Format.printf
    "@.The meet of the model beside the n-way meet of the compiler, on the \
     scenarios of the examples of the meet and of the chosen meets@.";
  let model = the_model Of_the_compiler in
  let same = ref 0 and different = ref 0 in
  let scenarios =
    List.map
      (fun (example : example_of_meet) ->
        ( example.statement,
          example.variables,
          example.steps,
          Option.map (fun (t1, t2, (_ : stated)) -> t1, t2) example.meet ))
      examples_of_meets
    @ List.map
        (fun (chosen : chosen_meet) ->
          chosen.label, chosen.kinds, chosen.before, Some chosen.met)
        the_chosen_meets
  in
  List.iter
    (fun (name, kinds, steps, meet) ->
      let threshold = List.length kinds in
      let in_the_model, in_the_compiler =
        scenario_in_both model kinds steps meet
      in
      if same_seen ~threshold in_the_model in_the_compiler
      then (
        incr same;
        Format.printf "  %s: the same%s@." name
          (match in_the_model with
          | Nothing_seen why ->
            Printf.sprintf " (neither answers: the model, %s; the compiler, %s)"
              why
              (print_seen ~threshold in_the_compiler)
          | Bottom_seen -> " (bottom)"
          | Seen _ -> ""))
      else (
        incr different;
        Format.printf
          "  @[<v>%s: DIFFERENT. The model:@ %a@ the compiler:@ %a@]@." name
          print_lines
          (print_seen ~threshold in_the_model)
          print_lines
          (print_seen ~threshold in_the_compiler)))
    scenarios;
  Format.printf "  Of %d scenarios: %d the same, %d different@."
    (List.length scenarios) !same !different;
  (* The same with the choices that are not the compiler's: where they make no
     difference, and where they do. *)
  let to_be_proved = the_model To_be_proved in
  let differing =
    List.filter_map
      (fun (name, kinds, steps, meet) ->
        let threshold = List.length kinds in
        let in_the_model, in_the_compiler =
          scenario_in_both to_be_proved kinds steps meet
        in
        if same_seen ~threshold in_the_model in_the_compiler
        then None
        else Some name)
      scenarios
  in
  Format.printf
    "  @[<v>With the choices of the meet and of the join that are to be \
     proved, in place of the compiler's, the model differs from the compiler \
     on %d of them:%a@]@."
    (List.length differing)
    (fun ppf -> List.iter (fun name -> Format.fprintf ppf "@ %s" name))
    differing;
  (* The same scenarios with their variables defined in the reverse order, and
     before every other step: the variables have the numbers that they had and
     other binding times, so that the canonical element of a class of aliases
     may be another. It is also counted on how many the compiler then answers
     otherwise than in the order given. *)
  let reversed steps =
    let defines, others =
      List.partition
        (fun (step : step) ->
          match step with
          | Define _ -> true
          | Relation _ | Equation _ | Meet_of _ -> false)
        steps
    in
    List.rev defines @ others
  in
  let same = ref 0 and different = ref 0 and moved = ref 0 in
  List.iter
    (fun (name, kinds, steps, meet) ->
      let threshold = List.length kinds in
      let in_the_model, in_the_compiler =
        scenario_in_both model kinds (reversed steps) meet
      in
      let (_ : seen), in_the_order_given =
        scenario_in_both model kinds steps meet
      in
      if not (same_seen ~threshold in_the_compiler in_the_order_given)
      then incr moved;
      if same_seen ~threshold in_the_model in_the_compiler
      then incr same
      else (
        incr different;
        Format.printf
          "  @[<v>%s, the variables defined in the reverse order: DIFFERENT. \
           The model:@ %a@ the compiler:@ %a@]@."
          name print_lines
          (print_seen ~threshold in_the_model)
          print_lines
          (print_seen ~threshold in_the_compiler)))
    scenarios;
  Format.printf
    "  With the variables of each scenario defined in the reverse order: %d \
     the same, %d different (the compiler answers otherwise than in the order \
     given on %d)@."
    !same !different !moved

(* ======================================================================== *)
(* A meet whose answer depends on the identifiers of its variables          *)
(* ======================================================================== *)

(* A variant that is never an immediate, with blocks of the tags given, of
   values, and no extension. *)
let variant_of_blocks (cases : (int * MT.flambda_type list) list) :
    MT.flambda_type =
  MT.create_variant false
    (Known (MT.bottom (Naked_number Kind_naked_immediate)))
    (Known
       (Row_like_for_blocks
          ( List.map
              (fun (tag, fields) : (_ * MT.row_like_block_case MT.Or_unknown.t)
                 ->
                ( N.z_of_int tag,
                  Known
                    { case_maps_to = fields;
                      case_index =
                        { index_domain =
                            Known (N.nat_of_int (List.length fields));
                          index_shape = Scannable Value_only
                        };
                      case_env_extension = MT.empty_env_extension
                    } ))
              cases,
            Bottom,
            Heap )))
    No_extensions

(* T7-13, which is T4-10 reached by a run. Three values, defined in the order of
   their numbers. The variable 1 is a variant whose blocks, of tag 0 or 1, hold
   the variable 3; and then one whose blocks hold a block of three fields, for
   tag 0, or of one, for tag 1, that are the variable 2: each case of the
   variable 1 then has an extension that gives the variable 3 that many fields.
   The variable 2 is a variant whose blocks hold the variable 3. The meet is of
   the variable 2 and a variant whose blocks, of tag 0, hold a block of two
   fields that are the variable 1.

   The meet of the field makes the variables 2 and 1 aliases while the equation
   on the variable 2 is being added, and gives the variable 3 two fields, which
   neither case of the variable 1 allows. The compiler then gives the variable 2
   a type that is no alias, though it is no longer canonical
   ([Meet_env.add_concrete_equation_on_canonical_name]), which its check for
   concrete types on canonical names only refuses
   ([Typing_env.replace_equation], under a debug flag that is off by default);
   and whether it finds the meet bottom depends on the order of the identifiers
   of the three variables, which are hashes.

   So this is not one of [the_chosen_meets]: what a meet of the compiler answers
   on it in one run is not what it answers in the next. It is run many times,
   with new variables each time, with each meet of the compiler, without and
   with that check, and what is answered is given for each order of the
   identifiers. The model answers the same whatever its variables: once is
   enough.

   The number of fields of the last block is a parameter, for a control: with
   three fields, which the case of tag 0 of the variable 1 allows, the meet is
   bottom in no order. *)
let the_meet_of_t7_13 ~(fields : int) : chosen_meet =
  let alias i = MT.alias_type_of Value (Var (N.positive_of_int i)) in
  let blocks = variant_of_blocks in
  let last = blocks [0, List.init fields (fun _ -> alias 1)] in
  { label = "T7-13";
    kinds = [Value; Value; Value];
    before =
      [ Define 1;
        Define 2;
        Define 3;
        Equation (1, blocks [0, [alias 3]; 1, [alias 3]]);
        Equation
          ( 1,
            blocks
              [ 0, [blocks [0, [alias 2; alias 2; alias 2]]];
                1, [blocks [0, [alias 2]]] ] );
        Equation (2, blocks [0, [alias 3]; 1, [alias 3]]) ];
    met = alias 2, blocks [0, [last]];
    given_to = None
  }

(* What a meet of the compiler answers on that scenario in one run. *)
type answer_of_t7_13 =
  | Is_bottom
  | Is_not_bottom of
      { (* Whether the variable 2 has, in the environment answered, a type that
           is no alias, and the variable 1 for its canonical element. *)
        concrete_and_an_alias_of_v1 : bool
      }
  | Is_refused_by_the_check
  | Stops_otherwise of string
  | Is_not_run of string

let print_answer_of_t7_13 (answer : answer_of_t7_13) : string =
  match answer with
  | Is_bottom -> "bottom"
  | Is_not_bottom { concrete_and_an_alias_of_v1 = true } ->
    "not bottom; v2 has a type that is no alias, and v1 for its canonical \
     element"
  | Is_not_bottom { concrete_and_an_alias_of_v1 = false } -> "not bottom"
  | Is_refused_by_the_check ->
    "the compiler stops: a concrete type is given to a name that is not \
     canonical"
  | Stops_otherwise line -> "the compiler stops: " ^ general line
  | Is_not_run why -> "not run: " ^ why

(* The six orders of the identifiers of the three variables, least first. *)
let the_orders_of_t7_13 : string list =
  [ "v1 < v2 < v3";
    "v1 < v3 < v2";
    "v2 < v1 < v3";
    "v2 < v3 < v1";
    "v3 < v1 < v2";
    "v3 < v2 < v1" ]

(* One run, with new variables: the order of their identifiers, least first, and
   the answer. The scenario tells its variables before it runs a step, so that
   the order is known of a run that a step stops; where it is not told them, the
   run has no order, and is in none of the six. *)
let run_the_meet_of_t7_13 ~fields (implementation : implementation) :
    string option * answer_of_t7_13 =
  let { label = _; kinds; before; met = first, second; given_to = _ } =
    the_meet_of_t7_13 ~fields
  in
  let order = ref None in
  let made (variable : int -> Variable.t) =
    order
      := Some
           (String.concat " < "
              (List.map
                 (fun (i, (_ : Variable.t)) -> Printf.sprintf "v%d" i)
                 (List.sort
                    (fun ((_ : int), x) ((_ : int), y) -> Variable.compare x y)
                    (List.map (fun i -> i, variable i) [1; 2; 3]))))
  in
  let answer =
    or_fatal_error (fun () ->
        let { numbering; variable; after_the_steps } =
          scenario ~made implementation kinds before
        in
        let to_compiler = Converter.type_of_model numbering machine_width in
        match after_the_steps with
        | Error why -> Is_not_run why
        | Ok env -> (
          match
            implementation.meet env (to_compiler first) (to_compiler second)
          with
          | Bottom -> Is_bottom
          | Ok ((_ : TG.t), env) ->
            let v1 = variable 1 and v2 = variable 2 in
            let concrete =
              Option.is_none
                (TG.get_alias_opt (TE.find env (Name.var v2) (Some K.value)))
            in
            let an_alias_of_v1 =
              match
                TE.get_canonical_simple_exn env
                  ~min_name_mode:Name_mode.in_types (Simple.var v2)
              with
              | canonical -> Simple.equal canonical (Simple.var v1)
              | exception Not_found -> false
            in
            Is_not_bottom
              { concrete_and_an_alias_of_v1 = concrete && an_alias_of_v1 }))
  in
  ( !order,
    match answer with
    | Ok answer -> answer
    | Error line ->
      let refused = "Trying to add equation giving concrete type on" in
      if
        String.length line >= String.length refused
        && String.equal (String.sub line 0 (String.length refused)) refused
      then Is_refused_by_the_check
      else Stops_otherwise line )

let the_meet_of_t7_13_in_every_order () =
  let runs = 240 in
  Format.printf
    "@.A meet whose answer depends on the identifiers of its variables (T7-13, \
     T4-10), run %d times with new variables with each meet of the compiler, \
     without and with its check for concrete types on canonical names only: \
     what is answered, for each order of the identifiers@."
    runs;
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  let the_order_of_bottom = "v2 < v3 < v1" in
  let overwritten : answer_of_t7_13 =
    Is_not_bottom { concrete_and_an_alias_of_v1 = true }
  in
  (* For each order seen, the answers seen in it. A run of which the order is
     not known is under a name that is none of the six. The check of the
     compiler is put back whatever happens. *)
  let answers_by_order ~fields (implementation : implementation)
      ~(with_the_check : bool) : (string * answer_of_t7_13 list) list =
    let flag = Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals in
    let before = !flag in
    flag := with_the_check;
    let seen = ref [] in
    Fun.protect
      ~finally:(fun () -> flag := before)
      (fun () ->
        for _ = 1 to runs do
          let order, answer = run_the_meet_of_t7_13 ~fields implementation in
          let order =
            Option.value order ~default:"the variables were not made"
          in
          let answers = Option.value (List.assoc_opt order !seen) ~default:[] in
          if not (List.mem answer answers)
          then
            seen := (order, answer :: answers) :: List.remove_assoc order !seen
        done);
    List.sort
      (fun (order, (_ : answer_of_t7_13 list)) (order', _) ->
        String.compare order order')
      !seen
  in
  let the_six_orders_and_no_other
      (by_order : (string * answer_of_t7_13 list) list) =
    List.equal String.equal
      (List.map (fun (order, (_ : answer_of_t7_13 list)) -> order) by_order)
      the_orders_of_t7_13
  in
  List.iter
    (fun ((implementation : implementation), with_the_check) ->
      let by_order =
        answers_by_order ~fields:2 implementation ~with_the_check
      in
      Format.printf "  the %s meet, %s the check:@."
        implementation.implementation
        (if with_the_check then "with" else "without");
      List.iter
        (fun (order, answers) ->
          Format.printf "    %s: %s@." order
            (String.concat " / " (List.map print_answer_of_t7_13 answers)))
        by_order;
      check "  each of the six orders is seen, and no run is without one"
        (the_six_orders_and_no_other by_order);
      check "  one answer in each order"
        (List.for_all
           (fun ((_ : string), answers) -> List.length answers = 1)
           by_order);
      let in_every_other_order (expected : answer_of_t7_13) =
        List.for_all
          (fun (order, answers) ->
            String.equal order the_order_of_bottom
            || List.equal ( = ) answers [expected])
          by_order
      in
      check
        (Printf.sprintf "  bottom in the order %s" the_order_of_bottom)
        (Option.equal (List.equal ( = ))
           (List.assoc_opt the_order_of_bottom by_order)
           (Some [Is_bottom]));
      if with_the_check
      then
        check "  in every other order, refused by the check"
          (in_every_other_order Is_refused_by_the_check)
      else
        check
          "  in every other order, not bottom, with a type that is no alias \
           for a name that is not canonical"
          (in_every_other_order overwritten))
    [binary, false; binary, true; n_way, false; n_way, true];
  List.iter
    (fun (choices, name) ->
      let { label = _; kinds; before; met = first, second; given_to = _ } =
        the_meet_of_t7_13 ~fields:2
      in
      let model = the_model choices in
      check
        (Printf.sprintf "the model, with %s: bottom" name)
        (match scenario_in_the_model model kinds before with
        | Error (_ : string) -> false
        | Ok env -> (
          match model.meet_of_model env first second with
          | Some Bottom -> true
          | Some (Ok _) | None -> false)))
    [ Of_the_compiler, "the compiler's choices";
      To_be_proved, "the choices that are to be proved" ];
  Format.printf
    "  a control: the last block with three fields, which one case of v1 \
     allows@.";
  List.iter
    (fun ((implementation : implementation), with_the_check) ->
      let by_order =
        answers_by_order ~fields:3 implementation ~with_the_check
      in
      Format.printf "  the %s meet, %s the check:@."
        implementation.implementation
        (if with_the_check then "with" else "without");
      List.iter
        (fun (order, answers) ->
          Format.printf "    %s: %s@." order
            (String.concat " / " (List.map print_answer_of_t7_13 answers)))
        by_order;
      check "  each of the six orders is seen, and no run is without one"
        (the_six_orders_and_no_other by_order);
      let in_every_order (expected : answer_of_t7_13) =
        List.for_all
          (fun ((_ : string), answers) -> List.equal ( = ) answers [expected])
          by_order
      in
      if with_the_check
      then
        check "  in every order, refused by the check"
          (in_every_order Is_refused_by_the_check)
      else
        check
          "  in every order, not bottom, with a type that is no alias for a \
           name that is not canonical"
          (in_every_order overwritten))
    [binary, false; binary, true; n_way, false; n_way, true];
  let { label = _; kinds; before; met = first, second; given_to = _ } =
    the_meet_of_t7_13 ~fields:3
  in
  let model = the_model Of_the_compiler in
  check "the model, with the compiler's choices, on the control: not bottom"
    (match scenario_in_the_model model kinds before with
    | Error (_ : string) -> false
    | Ok env -> (
      match model.meet_of_model env first second with
      | Some (Ok _) -> true
      | Some Bottom | None -> false))

(* ======================================================================== *)
(* A line of equations that leaves a type that is no alias on a name that   *)
(* is not canonical, in every order of the identifiers                      *)
(* ======================================================================== *)

(* T7-13 in a second shape. Three variables, defined in the order of their
   numbers: [m], the variable 1, a value; and two naked immediates, [jv] and
   [t].

   First, [t] is the tag of [m]: its type is that relation and nothing else,
   which is what the unboxing of a parameter that is a variant gives the
   variable of the tag. Then [jv] is 0 or 1. Then [m] is the immediate 0, or a
   block of tag 0 or 1: its type names [jv] as its tag and says, for when it is
   a block, that [t] is [jv] and that [jv] is the tag of [m]. Last, [t] is 0 or
   1.

   After the third equation [t] and [jv] are both canonical: that they are
   aliases is said only in the extension of the variant for a block. The last
   equation is added on [t], which has the relation. The meet of the two reduces
   the relation ([reduce_inverse_relations], the step that T1-1 is about): [m]
   is a block of tag 0 or 1. What the type of [m] says for a block is then added
   to the environment, and [t] becomes an alias of [jv], which was defined
   before it, while its own equation is being added. The compiler then gives [t]
   the type that the meet answered, which is no alias
   ([Meet_env.add_concrete_equation_on_canonical_name]).

   The first three equations are a reduced form of what was observed of the
   environment at a handler whose parameter, a variant, is unboxed, in a
   compilation with the n-way join of a unit that is not in this tree. Kept: the
   variable of the tag has the relation and nothing else; the type of the
   parameter may be an immediate, names as its tag a variable that an earlier
   join made, and says for when it is a block that the variable of the tag is
   that variable and that that variable is its tag; that variable is 0 or 1. Not
   kept: the variable that the type names for whether it is an immediate; its
   immediates, an alias of a variable there, the number 0 here; what it says for
   when it is an immediate; the fields of its blocks of tag 1, three there, two
   here. By reading, the last equation is what a later join adds where it names
   by [t] what its uses know of the tag.

   There is no join here, and what is answered is the same in every order of the
   identifiers, unlike the scenario above. The line is run many times, with new
   variables each time, with each meet of the compiler, without and with its
   check for concrete types on canonical names only; the model is run once.

   The type of [m] is given in three ways: written whole; by two equations, of
   which the first names [jv] as the tag and the second, after the relation on
   [t], names [t], as the shape of an unboxing does; and by the relation that
   the library adds for a tag ([Typing_env.add_get_tag_relation], which gives
   [t] the relation and [m] a type that names [t]), followed by the type that
   names [jv].

   Two controls: with [t] defined before [jv], [jv] is the one that becomes an
   alias; with [t] given one number in the place of "0 or 1", [t] becomes an
   alias of that number.

   What is asked: of the model with the compiler's choices, that [t] and no
   other name has a type that is no alias and a canonical element that is not
   itself, which is [jv] (in the controls, that no name has); of each meet of
   the compiler without the check, in each order of the identifiers, that it
   sees what that model sees; and with the check, that it stops with the fatal
   error of the check (in the controls, that it sees what the model sees). The
   model with the choices that are to be proved is run too, and what it has of
   such names is printed: nothing is asked of it. *)

(* A type of naked immediates: the numbers given, or any number where none is
   given; with the relation "the tag of the variable [tag_of]" where one is
   given. *)
let immediates_of_the_line (numbers : int list) ~(tag_of : int option) :
    MT.flambda_type =
  Type_naked_immediate
    (Ok
       (No_alias
          { naked_immediates =
              (match numbers with
              | [] -> Unknown
              | _ :: _ -> Known (List.map N.z_of_int numbers));
            inverse_relations =
              (match tag_of with
              | None -> []
              | Some m ->
                [(Get_tag : MT.Relation.t), Name_var (N.positive_of_int m)])
          }))

(* The variant of the line: the immediate 0, or a block of tag 0, of one field,
   or of tag 1, of two. Its type names the variable [get_tag] as its tag, and
   has the equations [when_block] in its extension for a block. *)
let variant_of_the_line ~(get_tag : int)
    ~(when_block : (int * MT.flambda_type) list) : MT.flambda_type =
  let unknown = MT.unknown Value in
  let head : MT.head_of_kind_value_non_null =
    Variant
      ( None,
        Known (immediates_of_the_line [0] ~tag_of:None),
        Some (N.positive_of_int get_tag),
        Known
          (Row_like_for_blocks
             ( List.map
                 (fun (tag, fields) :
                      (_ * MT.row_like_block_case MT.Or_unknown.t) ->
                   ( N.z_of_int tag,
                     Known
                       { case_maps_to = fields;
                         case_index =
                           { index_domain =
                               Known (N.nat_of_int (List.length fields));
                             index_shape = Scannable Value_only
                           };
                         case_env_extension = MT.empty_env_extension
                       } ))
                 [0, [unknown]; 1, [unknown; unknown]],
               Bottom,
               Heap )),
        (match when_block with
        | [] -> No_extensions
        | _ :: _ ->
          Ext
            ( MT.empty_env_extension,
              Env_extension
                (List.map
                   (fun (i, t) : (MT.name * MT.flambda_type) ->
                     Name_var (N.positive_of_int i), t)
                   when_block) )),
        false )
  in
  Type_value (Ok (No_alias (Head_of_kind_value (Ok head, Not_null))))

type how_the_line_gives_m =
  | Written_whole
  | By_two_equations
      (** A type that names [jv] as the tag; then the relation on [t] and a type
          that names [t]. *)
  | By_the_relation_of_the_library
      (** The relation on [t], added as the library adds it, which gives [m] a
          type that names [t]; then a type that names [jv]. *)

(* The steps of the line: [jv] and [t] are the variables 2 and 3, one way or the
   other; [last] is the type that [t] is given last. *)
let the_line_of_t7_13 ~(jv : int) ~(t : int) (how : how_the_line_gives_m)
    ~(last : MT.flambda_type) : step list =
  let m = 1 in
  let alias_of_jv =
    MT.alias_type_of (Naked_number Kind_naked_immediate)
      (Var (N.positive_of_int jv))
  in
  let the_tag_of_m = immediates_of_the_line [] ~tag_of:(Some m) in
  let zero_or_one = immediates_of_the_line [0; 1] ~tag_of:None in
  let zero_or_one_the_tag_of_m =
    immediates_of_the_line [0; 1] ~tag_of:(Some m)
  in
  [Define 1; Define 2; Define 3]
  @ (match how with
    | Written_whole ->
      [ Equation (t, the_tag_of_m);
        Equation (jv, zero_or_one);
        Equation
          ( m,
            variant_of_the_line ~get_tag:jv
              ~when_block:[t, alias_of_jv; jv, zero_or_one_the_tag_of_m] ) ]
    | By_two_equations ->
      [ Equation (jv, zero_or_one);
        Equation
          ( m,
            variant_of_the_line ~get_tag:jv
              ~when_block:[jv, zero_or_one_the_tag_of_m] );
        Equation (t, the_tag_of_m);
        Equation (m, variant_of_the_line ~get_tag:t ~when_block:[]) ]
    | By_the_relation_of_the_library ->
      [ Relation { relation = Get_tag; result = t; scrutinee = m };
        Equation (jv, zero_or_one);
        Equation
          ( m,
            variant_of_the_line ~get_tag:jv
              ~when_block:[jv, zero_or_one_the_tag_of_m] ) ])
  @ [Equation (t, last)]

let the_kinds_of_the_line : MK.flambda_kind list =
  [Value; Naked_number Kind_naked_immediate; Naked_number Kind_naked_immediate]

let the_names_of_the_line : MT.name list =
  List.map (fun i : MT.name -> Name_var (N.positive_of_int i)) [1; 2; 3]

(* What one of the two meets of the compiler answers on the line in one run. *)
type answer_of_the_line =
  | Sees_after_the_line of seen
  | Line_is_refused_by_the_check
  | Line_stops_otherwise of string

let same_answer_of_the_line (answer : answer_of_the_line)
    (answer' : answer_of_the_line) : bool =
  match answer, answer' with
  | Sees_after_the_line seen, Sees_after_the_line seen' ->
    same_seen ~threshold:3 seen seen'
  | Line_is_refused_by_the_check, Line_is_refused_by_the_check -> true
  | Line_stops_otherwise why, Line_stops_otherwise why' -> String.equal why why'
  | ( ( Sees_after_the_line _ | Line_is_refused_by_the_check
      | Line_stops_otherwise _ ),
      _ ) ->
    false

let print_answer_of_the_line (answer : answer_of_the_line) : string =
  match answer with
  | Sees_after_the_line seen -> print_seen ~threshold:3 seen
  | Line_is_refused_by_the_check ->
    "the compiler stops: a concrete type is given to a name that is not \
     canonical"
  | Line_stops_otherwise why -> "the compiler stops: " ^ why

(* The six orders of the identifiers of the three variables, least first. *)
let the_orders_of_the_line : string list =
  [ "jv < m < t";
    "jv < t < m";
    "m < jv < t";
    "m < t < jv";
    "t < jv < m";
    "t < m < jv" ]

(* One run, with new variables: the order of their identifiers, least first, and
   the answer. As for the scenario above, the variables are told before a step
   is run. *)
let run_the_line ~(jv : int) ~(t : int) (implementation : implementation)
    (steps : step list) : string option * answer_of_the_line =
  let order = ref None in
  let made (variable : int -> Variable.t) =
    let role i =
      if Int.equal i jv then "jv" else if Int.equal i t then "t" else "m"
    in
    order
      := Some
           (String.concat " < "
              (List.map
                 (fun (i, (_ : Variable.t)) -> role i)
                 (List.sort
                    (fun ((_ : int), x) ((_ : int), y) -> Variable.compare x y)
                    (List.map (fun i -> i, variable i) [1; 2; 3]))))
  in
  let answer =
    or_fatal_error (fun () ->
        let { numbering; variable = _; after_the_steps } =
          scenario ~made implementation the_kinds_of_the_line steps
        in
        match after_the_steps with
        | Error why -> Line_stops_otherwise why
        | Ok env ->
          Sees_after_the_line
            (seen_in_the_compiler numbering ~threshold:3 the_names_of_the_line
               None env))
  in
  ( !order,
    match answer with
    | Ok answer -> answer
    | Error line ->
      if
        String.starts_with
          ~prefix:"Trying to add equation giving concrete type on" line
      then Line_is_refused_by_the_check
      else Line_stops_otherwise (general line) )

(* The names that have, in what is seen, a type that is no alias and a canonical
   element that is not themselves, each with that element. No answer where no
   environment is seen, or where a name is seen without its type or without its
   canonical element. *)
let concrete_and_not_canonical (seen : seen) : (MT.name * M.simple) list option
    =
  match seen with
  | Nothing_seen _ | Bottom_seen -> None
  | Seen { answer = _; env_is_bottom = _; of_names; of_others = _ } ->
    List.fold_right
      (fun ((name : MT.name), (t : MT.flambda_type option), canonical) found ->
        match found, t, canonical with
        | None, (Some _ | None), (Some _ | None)
        | Some _, None, (Some _ | None)
        | Some _, Some _, None ->
          None
        | Some found, Some t, Some (canonical : M.simple) ->
          if
            Option.is_none (MT.get_alias_opt t)
            && not (Stdlib.( = ) canonical (MT.simple_of_name name))
          then Some ((name, canonical) :: found)
          else Some found)
      of_names (Some [])

let print_concrete_and_not_canonical (found : (MT.name * M.simple) list option)
    : string =
  match found with
  | None -> "not seen"
  | Some [] -> "none"
  | Some (_ :: _ as found) ->
    String.concat ", "
      (List.map
         (fun (name, canonical) ->
           Format.asprintf "%a, of which the canonical element is %a"
             Model_type.print_name name Model_type.print_simple canonical)
         found)

let the_line_of_t7_13_in_every_order () =
  let runs = 240 in
  Format.printf
    "@.A line of equations that leaves a type that is no alias on a name that \
     is not canonical (T7-13, T4-10; through the reduction of T1-1), run %d \
     times with new variables with each meet of the compiler, without and with \
     its check for concrete types on canonical names only. m, the variable 1, \
     is a value; jv and t are naked immediates. t is the tag of m; jv is 0 or \
     1; m is the immediate 0 or a block of tag 0 or 1, names jv as its tag, \
     and says for when it is a block that t is jv and that jv is its tag; \
     last, t is given \"0 or 1\"@."
    runs;
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  (* For each order seen, the answers seen in it; the check of the compiler is
     put back whatever happens. *)
  let answers_by_order ~jv ~t (implementation : implementation) steps
      ~(with_the_check : bool) : (string * answer_of_the_line list) list =
    let flag = Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals in
    let before = !flag in
    flag := with_the_check;
    let seen = ref [] in
    Fun.protect
      ~finally:(fun () -> flag := before)
      (fun () ->
        for _ = 1 to runs do
          let order, answer = run_the_line ~jv ~t implementation steps in
          let order =
            Option.value order ~default:"the variables were not made"
          in
          let answers = Option.value (List.assoc_opt order !seen) ~default:[] in
          if not (List.exists (same_answer_of_the_line answer) answers)
          then
            seen := (order, answer :: answers) :: List.remove_assoc order !seen
        done);
    List.sort
      (fun (order, (_ : answer_of_the_line list)) (order', _) ->
        String.compare order order')
      !seen
  in
  let zero_or_one = immediates_of_the_line [0; 1] ~tag_of:None in
  let one = MT.this_naked_immediate (N.z_of_int 1) in
  let has (seen : seen) (expected : (MT.name * M.simple) list) : bool =
    Option.equal (List.equal Stdlib.( = ))
      (concrete_and_not_canonical seen)
      (Some expected)
  in
  let such_names =
    "with a type that is no alias and a canonical element that is not itself"
  in
  List.iter
    (fun (title, jv, t, how, last, (is_the_line : bool)) ->
      Format.printf "  %s (jv is v%d and t is v%d)@." title jv t;
      let steps = the_line_of_t7_13 ~jv ~t how ~last in
      (* What is to be seen of such names: [t], with [jv] for its canonical
         element; in a control, none. *)
      let expected : (MT.name * M.simple) list =
        if is_the_line
        then [Name_var (N.positive_of_int t), Var (N.positive_of_int jv)]
        else []
      in
      let in_the_model choices : seen =
        match
          scenario_in_the_model (the_model choices) the_kinds_of_the_line steps
        with
        | Error why -> Nothing_seen why
        | Ok env ->
          seen_in_the_model ~threshold:3 the_names_of_the_line None env
      in
      let model = in_the_model Of_the_compiler in
      Format.printf
        "    @[<v>the model, with the compiler's choices, sees:@ %a@]@."
        print_lines
        (print_seen ~threshold:3 model);
      Format.printf "    there, %s: %s@." such_names
        (print_concrete_and_not_canonical (concrete_and_not_canonical model));
      check
        (if is_the_line
         then "    that is t, with jv for its canonical element, and no other"
         else "    no name is")
        (has model expected);
      Format.printf
        "    the model, with the choices that are to be proved: %s: %s@."
        such_names
        (print_concrete_and_not_canonical
           (concrete_and_not_canonical (in_the_model To_be_proved)));
      List.iter
        (fun ((implementation : implementation), with_the_check) ->
          let by_order =
            answers_by_order ~jv ~t implementation steps ~with_the_check
          in
          let in_every_order (holds : answer_of_the_line -> bool) =
            List.for_all
              (fun ((_ : string), answers) ->
                match answers with
                | [answer] -> holds answer
                | [] | _ :: _ :: _ -> false)
              by_order
          in
          let stops = with_the_check && is_the_line in
          let as_expected =
            List.equal String.equal
              (List.map
                 (fun (order, (_ : answer_of_the_line list)) -> order)
                 by_order)
              the_orders_of_the_line
            &&
            if stops
            then
              in_every_order (fun answer ->
                  match answer with
                  | Line_is_refused_by_the_check -> true
                  | Sees_after_the_line _ | Line_stops_otherwise _ -> false)
            else
              in_every_order (fun answer ->
                  match answer with
                  | Sees_after_the_line seen ->
                    same_seen ~threshold:3 model seen && has seen expected
                  | Line_is_refused_by_the_check | Line_stops_otherwise _ ->
                    false)
          in
          check
            (Printf.sprintf
               "  the %s meet, %s the check: each of the six orders is seen, \
                and in each %s"
               implementation.implementation
               (if with_the_check then "with" else "without")
               (if stops
                then
                  "the compiler stops: a concrete type is given to a name that \
                   is not canonical"
                else "the compiler sees what the model sees"))
            as_expected;
          if not as_expected
          then
            List.iter
              (fun (order, answers) ->
                Format.printf "      @[<v>%s:%a@]@." order
                  (fun ppf ->
                    List.iter (fun answer ->
                        Format.fprintf ppf "@ %a" print_lines
                          (print_answer_of_the_line answer)))
                  answers)
              by_order)
        [binary, false; binary, true; n_way, false; n_way, true])
    [ "the type of m written whole", 2, 3, Written_whole, zero_or_one, true;
      ( "the type of m by two equations: one that names jv as the tag; then, \
         after the relation on t, one that names t",
        2,
        3,
        By_two_equations,
        zero_or_one,
        true );
      ( "the relation on t added as the library adds it, which gives m a type \
         that names t; then a type of m that names jv",
        2,
        3,
        By_the_relation_of_the_library,
        zero_or_one,
        true );
      ( "a control: t defined before jv (the type of m written whole)",
        3,
        2,
        Written_whole,
        zero_or_one,
        false );
      ( "a control: t given the number 1 in the place of \"0 or 1\" (the type \
         of m written whole)",
        2,
        3,
        Written_whole,
        one,
        false ) ];
  (* Where the names of that sort are not all to be read, none is said to be
     there, and none is said not to be. *)
  let a_name : MT.name = Name_var (N.positive_of_int 1) in
  let a_type = MT.unknown Value in
  let not_read (seen : seen) =
    Option.is_none (concrete_and_not_canonical seen)
  in
  check
    "a control: of such names nothing is said where nothing is seen, where the \
     meet is bottom, and where a name is seen without its type or without its \
     canonical element"
    (not_read (Nothing_seen "taken away")
    && not_read Bottom_seen
    && not_read
         (Seen
            { answer = None;
              env_is_bottom = false;
              of_names = [a_name, None, Some (MT.simple_of_name a_name)];
              of_others = []
            })
    && not_read
         (Seen
            { answer = None;
              env_is_bottom = false;
              of_names = [a_name, Some a_type, None];
              of_others = []
            })
    && not
         (not_read
            (Seen
               { answer = None;
                 env_is_bottom = false;
                 of_names =
                   [a_name, Some a_type, Some (MT.simple_of_name a_name)];
                 of_others = []
               })))

(* ======================================================================== *)
(* Two lines of equations on which the model and the two meets of the       *)
(* compiler do not answer alike, in every order of the identifiers          *)
(* ======================================================================== *)

(* T7-14 and T7-15. Four values, defined in the order of their numbers; the
   variants are never immediates ([variant_of_blocks]).

   The line of T7-14. The variable 1 is a variant whose blocks, of tag 0 or 1,
   hold the variable 3; and then one whose blocks hold a block of two fields
   that are the variable 2, for tag 0, or a block of one field that is the
   variable 3, for tag 1: each case of the variable 1 then has an extension that
   says which block the variable 3 is. The variable 2 is the variable 1. The
   variable 4 is a variant whose blocks, of tag 0, hold the variable 2; and then
   one whose blocks hold a block of tag 0 and one field that is the variable 2.

   The last equation meets the variable 2, which is the variable 1, with a block
   of tag 0 and one field: the variable 1 has tag 0, and its field, the variable
   3, is the variable 2, which is the variable 1; and the extension of tag 0
   gives the variable 3 two fields, where the variable 1 has one. No value has
   both. The model answers bottom. Neither meet of the compiler does, in each of
   the twenty-four orders of the identifiers of the four variables.

   Between the model and the n-way meet the difference is T4-21, which the owner
   of the meet found by a trace of the model on this line; the compiler was not
   traced. What a meet adds in a scope is added again when the scope is left: by
   the join that the meet of the cases calls, and then by the meet of the
   variant around it when it returns. The n-way meet, as its source reads, goes
   through a map of names each time, so each time in the order of the names. The
   model goes through lists, and here it adds the equations on the variables 3
   and 1 in one order and then in the other, which no order of names gives; that
   is how it reaches the contradiction. Run outside this tree with either adding
   reversed, the model is not bottom, and sees what both meets see in the twelve
   orders in which the identifier of the variable 1 is less than that of the
   variable 3, or what they see in the twelve others; in both runs, by that
   trace, it ignores the extension of tag 0 because an equation is being added
   to the variable 1
   ([Meet_env.add_or_delay_concrete_equation_on_canonical_name]), and that is
   where the model then loses the contradiction. That the compiler loses it at
   that rule is what its source and those runs of the model suggest; no trace of
   the compiler establishes it. Why the binary meet is not bottom was not looked
   at.

   The line of T7-15 is the same with one case in the place of two: the variable
   1 is a variant whose blocks, of tag 0, hold the variable 3, and then one
   whose blocks hold a block of two fields that are the variable 2, so that the
   variable 3 is that block in the environment. The binary meet answers bottom.
   The n-way meet and the model see after the last equation what they see before
   it. By a trace of the model, which the owner of the meet ran, and by the
   source of the n-way meet, which was read and not traced: the join that the
   meet of the cases ends with finds the contradiction when it adds the
   equations of the case to its target, and answers the target marked bottom,
   with none of them ([Meet_env.add_env_extension], called by
   [Join_env.define_or_eliminate_variables_and_add_equations]); the meet of the
   cases answers that environment; and the meet of the variant, when it returns,
   cuts the level of that environment, which has no equation, and adds it to the
   environment before the scope, which is not bottom ([direct_return] of
   [meet_disjunction]). The mark is lost there, and nothing is learnt.

   Each line is run many times, with new variables each time, with each meet of
   the compiler, and what is answered is asked in each of the twenty-four orders
   of the identifiers; the model is run once. A control: with the last two
   equations of the line of T7-14 replaced by one that says of the variable 2
   itself that it is a block of tag 0 and one field that is the variable 2, the
   model and both meets see the same, and it is not bottom. No value has those
   equations either: the variable 1 then has tag 0, its field, the variable 3,
   is the variable 1, and the extension of tag 0 gives the variable 3 two
   fields. So the control is not a line that has a value: on it the same
   contradiction is reached by another route, and the model misses it as both
   meets do. *)
type line_of_values =
  | Line_of_t7_14
  | Line_of_t7_15
  | Line_of_t7_14_on_v2

(* The steps of a line; without its last equation where [whole] is not set. *)
let the_line_of_values ?(whole = true) (line : line_of_values) : step list =
  let alias i = MT.alias_type_of Value (Var (N.positive_of_int i)) in
  let blocks = variant_of_blocks in
  let two_cases : step list =
    [ Equation (1, blocks [0, [alias 3]; 1, [alias 3]]);
      Equation
        ( 1,
          blocks
            [0, [blocks [0, [alias 2; alias 2]]]; 1, [blocks [0, [alias 3]]]] );
      Equation (2, alias 1) ]
  in
  let one_case : step list =
    [ Equation (1, blocks [0, [alias 3]]);
      Equation (1, blocks [0, [blocks [0, [alias 2; alias 2]]]]);
      Equation (2, alias 1) ]
  in
  let through_v4 : step list =
    [Equation (4, blocks [0, [alias 2]])]
    @ if whole then [Equation (4, blocks [0, [blocks [0, [alias 2]]]])] else []
  in
  let on_v2 : step list =
    if whole then [Equation (2, blocks [0, [alias 2]])] else []
  in
  [Define 1; Define 2; Define 3; Define 4]
  @
  match line with
  | Line_of_t7_14 -> two_cases @ through_v4
  | Line_of_t7_15 -> one_case @ through_v4
  | Line_of_t7_14_on_v2 -> two_cases @ on_v2

let the_kinds_of_the_lines_of_values : MK.flambda_kind list =
  [Value; Value; Value; Value]

let the_names_of_the_lines_of_values : MT.name list =
  List.map (fun i : MT.name -> Name_var (N.positive_of_int i)) [1; 2; 3; 4]

(* One run of a line of values, with new variables: the order of their
   identifiers, least first, taken before the first step, and what is seen after
   the steps, or the fatal error. *)
let run_a_line_of_values (implementation : implementation) (steps : step list) :
    string option * (seen, string) result =
  let order = ref None in
  let made (variable : int -> Variable.t) =
    order
      := Some
           (String.concat " < "
              (List.map
                 (fun (i, (_ : Variable.t)) -> Printf.sprintf "v%d" i)
                 (List.sort
                    (fun ((_ : int), x) ((_ : int), y) -> Variable.compare x y)
                    (List.map (fun i -> i, variable i) [1; 2; 3; 4]))))
  in
  let answer =
    or_fatal_error (fun () ->
        let { numbering; variable = _; after_the_steps } =
          scenario ~made implementation the_kinds_of_the_lines_of_values steps
        in
        match after_the_steps with
        | Error why -> Error why
        | Ok env ->
          Ok
            (seen_in_the_compiler numbering ~threshold:4
               the_names_of_the_lines_of_values None env))
  in
  ( !order,
    match answer with
    | Ok answer -> answer
    | Error line -> Error ("a fatal error: " ^ general line) )

(* The twenty-four orders of the identifiers of four variables, least first, in
   the order of their texts. *)
let the_orders_of_four : string list =
  let rec orders (left : int list) : int list list =
    match left with
    | [] -> [[]]
    | _ :: _ ->
      List.concat_map
        (fun i ->
          List.map
            (fun rest -> i :: rest)
            (orders (List.filter (fun j -> j <> i) left)))
        left
  in
  List.sort String.compare
    (List.map
       (fun order ->
         String.concat " < " (List.map (Printf.sprintf "v%d") order))
       (orders [1; 2; 3; 4]))

(* Whether what is seen is an environment that is bottom; no answer where no
   environment is seen. *)
let seen_is_bottom (seen : seen) : bool option =
  match seen with
  | Seen { answer = _; env_is_bottom; of_names = _; of_others = _ } ->
    Some env_is_bottom
  | Nothing_seen _ | Bottom_seen -> None

let the_lines_of_t7_14_and_t7_15 () =
  let runs = 480 in
  Format.printf
    "@.Two lines of equations on four values on which the model and the two \
     meets of the compiler do not answer alike (T7-14, T7-15), each run %d \
     times with new variables with each meet of the compiler: what is \
     answered, in the twenty-four orders of the identifiers@."
    runs;
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  (* For each order seen, the answers seen in it. *)
  let answers_by_order (implementation : implementation) steps :
      (string * (seen, string) result list) list =
    let same (answer : (seen, string) result) (answer' : (seen, string) result)
        =
      match answer, answer' with
      | Ok seen, Ok seen' -> same_seen ~threshold:4 seen seen'
      | Error why, Error why' -> String.equal why why'
      | Ok _, Error _ | Error _, Ok _ -> false
    in
    let found = ref [] in
    for _ = 1 to runs do
      let order, answer = run_a_line_of_values implementation steps in
      let order = Option.value order ~default:"the variables were not made" in
      let answers = Option.value (List.assoc_opt order !found) ~default:[] in
      if not (List.exists (same answer) answers)
      then found := (order, answer :: answers) :: List.remove_assoc order !found
    done;
    List.sort
      (fun (order, (_ : (seen, string) result list)) (order', _) ->
        String.compare order order')
      !found
  in
  (* Each of the twenty-four orders is seen, with one answer, which is an
     environment of which [holds]. *)
  let in_every_order by_order (holds : seen -> bool) : bool =
    List.equal String.equal
      (List.map
         (fun (order, (_ : (seen, string) result list)) -> order)
         by_order)
      the_orders_of_four
    && List.for_all
         (fun ((_ : string), answers) ->
           match answers with
           | [Ok seen] -> holds seen
           | [Error (_ : string)] | [] | _ :: _ :: _ -> false)
         by_order
  in
  let print_by_order by_order =
    List.iter
      (fun (order, answers) ->
        Format.printf "      @[<v>%s:%a@]@." order
          (fun ppf ->
            List.iter (fun (answer : (seen, string) result) ->
                Format.fprintf ppf "@ %a" print_lines
                  (match answer with
                  | Ok seen -> print_seen ~threshold:4 seen
                  | Error why -> "the compiler stops: " ^ why)))
          answers)
      by_order
  in
  let in_the_model choices steps : seen =
    match
      scenario_in_the_model (the_model choices) the_kinds_of_the_lines_of_values
        steps
    with
    | Error why -> Nothing_seen why
    | Ok env ->
      seen_in_the_model ~threshold:4 the_names_of_the_lines_of_values None env
  in
  let is_bottom expected (seen : seen) =
    Option.equal Bool.equal (seen_is_bottom seen) (Some expected)
  in
  let asked ~title (implementation : implementation) steps
      (holds : seen -> bool) =
    let by_order = answers_by_order implementation steps in
    let as_expected = in_every_order by_order holds in
    check
      (Printf.sprintf
         "  the %s meet: each of the twenty-four orders is seen, and in each %s"
         implementation.implementation title)
      as_expected;
    if not as_expected then print_by_order by_order;
    by_order
  in
  (* The line of T7-14. *)
  Format.printf
    "  T7-14: v1 is a variant whose blocks, of tag 0 or 1, hold v3, and then \
     one whose blocks hold a block of two fields that are v2, for tag 0, or of \
     one field that is v3, for tag 1; v2 is v1; v4 is a variant whose blocks, \
     of tag 0, hold v2, and then one whose blocks hold a block of one field \
     that is v2@.";
  let before = the_line_of_values ~whole:false Line_of_t7_14 in
  let whole = the_line_of_values Line_of_t7_14 in
  let model_before = in_the_model Of_the_compiler before in
  check "  before the last equation, the model is not bottom"
    (is_bottom false model_before);
  List.iter
    (fun (implementation : implementation) ->
      ignore
        (asked
           ~title:
             "the compiler sees what the model sees, before the last equation"
           implementation before (fun seen ->
             same_seen ~threshold:4 model_before seen)
          : (string * (seen, string) result list) list))
    [binary; n_way];
  List.iter
    (fun (choices, name) ->
      check
        (Printf.sprintf
           "  after the last equation, the model, with %s, is bottom" name)
        (is_bottom true (in_the_model choices whole)))
    [ Of_the_compiler, "the compiler's choices";
      To_be_proved, "the choices that are to be proved" ];
  let after =
    List.map
      (fun (implementation : implementation) ->
        asked ~title:"the compiler is not bottom, after the last equation"
          implementation whole (is_bottom false))
      [binary; n_way]
  in
  (match after with
  | [of_binary; of_n_way] ->
    check "  in each order the two meets see the same"
      (List.equal
         (fun (order, answers) (order', answers') ->
           String.equal order order'
           &&
           match answers, answers' with
           | [Ok seen], [Ok seen'] -> same_seen ~threshold:4 seen seen'
           | ( ([] | [Error _] | [Ok _] | _ :: _ :: _),
               ([] | [Error _] | [Ok _] | _ :: _ :: _) ) ->
             false)
         of_binary of_n_way);
    (* What the n-way meet sees, as distinct texts with the orders of each. *)
    let texts =
      List.fold_left
        (fun texts (order, answers) ->
          match answers with
          | [Ok seen] ->
            let text = print_seen ~threshold:4 seen in
            let orders = Option.value (List.assoc_opt text texts) ~default:[] in
            (text, orders @ [order]) :: List.remove_assoc text texts
          | [] | [Error _] | _ :: _ :: _ -> texts)
        [] of_n_way
    in
    let v1_before_v3 order =
      match
        ( List.find_index (String.equal "v1") (String.split_on_char ' ' order),
          List.find_index (String.equal "v3") (String.split_on_char ' ' order) )
      with
      | Some i, Some j -> Some (i < j)
      | None, (Some _ | None) | Some _, None -> None
    in
    check
      "  the n-way meet sees one of two things, according to whether the \
       identifier of v1 is less than that of v3"
      (match texts with
      | [(_, orders); (_, orders')] ->
        let all expected orders =
          List.for_all
            (fun order ->
              Option.equal Bool.equal (v1_before_v3 order) (Some expected))
            orders
        in
        (all true orders && all false orders')
        || (all false orders && all true orders')
      | [] | [_] | _ :: _ :: _ :: _ -> false);
    List.iter
      (fun (text, orders) ->
        Format.printf "    @[<v>in %d orders, of which %s:@ %a@]@."
          (List.length orders)
          (match orders with order :: _ -> order | [] -> "none")
          print_lines text)
      (List.sort
         (fun ((_ : string), orders) ((_ : string), orders') ->
           compare orders orders')
         texts)
  | [] | [_] | _ :: _ :: _ :: _ -> check "  two meets are run" false);
  (* The control. *)
  Format.printf
    "  a control, of which no value has the equations either: the last two \
     equations replaced by one that says of v2 itself that it is a block of \
     one field that is v2@.";
  let on_v2 = the_line_of_values Line_of_t7_14_on_v2 in
  let model_on_v2 = in_the_model Of_the_compiler on_v2 in
  check "  the model is not bottom" (is_bottom false model_on_v2);
  List.iter
    (fun (implementation : implementation) ->
      ignore
        (asked ~title:"the compiler sees what the model sees" implementation
           on_v2 (fun seen -> same_seen ~threshold:4 model_on_v2 seen)
          : (string * (seen, string) result list) list))
    [binary; n_way];
  (* The line of T7-15. *)
  Format.printf
    "  T7-15: the same with one case: v1 is a variant whose blocks, of tag 0, \
     hold v3, and then one whose blocks hold a block of two fields that are \
     v2@.";
  let before = the_line_of_values ~whole:false Line_of_t7_15 in
  let whole = the_line_of_values Line_of_t7_15 in
  let model_before = in_the_model Of_the_compiler before in
  let model_after = in_the_model Of_the_compiler whole in
  check "  before the last equation, the model is not bottom"
    (is_bottom false model_before);
  check
    "  after the last equation, the model, with the compiler's choices, sees \
     what it sees before it"
    (is_bottom false model_after
    && same_seen ~threshold:4 model_before model_after);
  check "  and the model with the choices that are to be proved sees the same"
    (same_seen ~threshold:4 model_after (in_the_model To_be_proved whole));
  List.iter
    (fun (implementation : implementation) ->
      ignore
        (asked
           ~title:
             "the compiler sees what the model sees, before the last equation"
           implementation before (fun seen ->
             same_seen ~threshold:4 model_before seen)
          : (string * (seen, string) result list) list))
    [binary; n_way];
  ignore
    (asked ~title:"the compiler is bottom, after the last equation" binary whole
       (is_bottom true)
      : (string * (seen, string) result list) list);
  ignore
    (asked
       ~title:"the compiler sees what the model sees, after the last equation"
       n_way whole (fun seen -> same_seen ~threshold:4 model_after seen)
      : (string * (seen, string) result list) list)

(* The environment of the tests, in the model: the variables and the symbols
   that [names.env] defines, of which nothing is known. *)
let env_of_model (names : names) : Model_env.t * MT.name list =
  let defined =
    List.map
      (fun (name, kind) ->
        ( Converter.name_to_model names.numbering name,
          Converter.kind_to_model kind ))
      names.defined
  in
  ( List.fold_left
      (fun env ((name : MT.name), kind) ->
        match name with
        | Name_var x ->
          Model_env.add_variable_definition env x kind Model_name_mode.normal
        | Name_symbol symbol -> Model_env.add_symbol_definition env symbol)
      Model_env.create defined,
    List.map fst defined )

(* Equations on variables of the tests, a variable having at most one, whose
   type mentions no variable but those numbered after it: no type is reached
   from itself. *)
let generate_equations (names : names) r : (MT.name * MT.flambda_type) list =
  let pool = names.pool in
  let config : G.config = { well_formed = true; depth = 1 } in
  let candidates =
    List.map (fun x -> x, MK.Value) pool.value_variables
    @ List.map
        (fun x -> x, MK.Naked_number Kind_naked_immediate)
        pool.naked_immediate_variables
  in
  List.filter_map
    (fun (x, kind) ->
      let rec attempt n : (MT.name * MT.flambda_type) option =
        if n = 0
        then None
        else
          let t = G.type_of_kind config r pool kind in
          if
            List.for_all
              (fun y -> N.compare_positive x y < 0)
              (MT.free_variables t)
          then Some (Name_var x, t)
          else attempt (n - 1)
      in
      if G.chance r ~percent:50 then attempt 5 else None)
    candidates

(* Equations that make aliases: a variable of kind value may be given an alias
   to a variable of that kind numbered after it, an alias to a symbol, or a type
   as above. *)
let generate_aliases (names : names) r : (MT.name * MT.flambda_type) list =
  let pool = names.pool in
  let config : G.config = { well_formed = true; depth = 1 } in
  List.filter_map
    (fun x : (MT.name * MT.flambda_type) option ->
      let later =
        List.filter (fun y -> N.compare_positive x y < 0) pool.value_variables
      in
      match G.below r 4, later with
      | 0, _ -> None
      | 1, _ :: _ ->
        Some (Name_var x, MT.alias_type_of Value (Var (G.pick r later)))
      | (1 | 2), _ ->
        Some
          (Name_var x, MT.alias_type_of Value (Symbol (G.pick r pool.symbols)))
      | _, _ ->
        let t = G.type_of_kind config r pool MK.Value in
        if
          List.for_all
            (fun y -> N.compare_positive x y < 0)
            (MT.free_variables t)
        then Some (Name_var x, t)
        else None)
    pool.value_variables

(* Two types of the shape of T4-9, which the generator of types does not make: a
   variant whose blocks, of two or three tags, each hold one variable, the same
   for all; and a variant whose blocks of those tags each hold a block whose
   fields repeat one variable, or a number. The meet of the fields of a case
   gives the first variable a type in the extension of the case, and the join of
   the cases makes variables where the fields repeat. Both may be the field of a
   block. *)
let generate_meet_of_cases (names : names) r : MT.flambda_type * MT.flambda_type
    =
  let pool = names.pool in
  let alias x = MT.alias_type_of Value (Var x) in
  let variant immediates cases : MT.flambda_type =
    MT.create_variant false immediates
      (Known
         (Row_like_for_blocks
            ( List.map
                (fun (tag, fields) :
                     (_ * MT.row_like_block_case MT.Or_unknown.t) ->
                  ( N.z_of_int tag,
                    Known
                      { case_maps_to = fields;
                        case_index =
                          { index_domain =
                              Known (N.nat_of_int (List.length fields));
                            index_shape = Scannable Value_only
                          };
                        case_env_extension = MT.empty_env_extension
                      } ))
                cases,
              Bottom,
              Heap )))
      No_extensions
  in
  let blocks =
    variant (Known (MT.bottom (Naked_number Kind_naked_immediate)))
  in
  let immediates () : MT.flambda_type MT.Or_unknown.t =
    if G.chance r ~percent:50
    then Unknown
    else Known (MT.bottom (Naked_number Kind_naked_immediate))
  in
  let tags = List.init (2 + G.below r 2) Fun.id in
  let held = G.pick r pool.value_variables in
  let field () =
    if G.chance r ~percent:20
    then MT.this_tagged_immediate (G.small_number r)
    else
      let repeated = G.pick r pool.value_variables in
      blocks [0, List.init (1 + G.below r 3) (fun _ -> alias repeated)]
  in
  let one =
    variant (immediates ()) (List.map (fun tag -> tag, [alias held]) tags)
  in
  let other =
    variant (immediates ()) (List.map (fun tag -> tag, [field ()]) tags)
  in
  if G.chance r ~percent:50
  then blocks [0, [one]], blocks [0, [other]]
  else one, other

(* Two types that are alike: a type and, on either side, one of the types that
   one step of the shrinker gives for it; or the type again, where the shrinker
   gives none, and 15 times in 100 where it gives some. Two types generated
   apart seldom have a part in common, and their meet or join then stops at the
   top. A step of the shrinker changes the text in one place: a part, which may
   be the whole type or its head, becomes the unknown type; or one thing is
   removed: a case, a field, a slot, a number of a set, a string of a set or a
   character of a string, an equation of an extension, the variable of a
   relation. That is no order between types, and nothing is claimed of how the
   two compare: with a case or a number removed a type most often says more,
   with a part unknown it says less, with a character removed from a string each
   of the two has values that the other has not, with a field removed the fields
   after it are in other places; the second need not keep the structure of the
   first. "Smaller", in the titles of the runs, is in that sense only. *)
let generate_alike (names : names) r (kind : MK.flambda_kind) :
    MT.flambda_type * MT.flambda_type =
  let config : G.config = { well_formed = true; depth = 3 } in
  let t = G.type_of_kind config r names.pool kind in
  let smaller =
    List.filter (fun t' -> Stdlib.( = ) (MT.kind t') kind) (G.shrink t)
  in
  let other =
    match smaller with
    | [] -> t
    | _ :: _ -> if G.chance r ~percent:15 then t else G.pick r smaller
  in
  if G.chance r ~percent:50 then t, other else other, t

(* Types that are alike in another way: each is one of the smaller types that
   the shrinker gives for one type, so that each may say something that the
   others do not. *)
let generate_siblings (names : names) r (kind : MK.flambda_kind) :
    MT.flambda_type * MT.flambda_type * MT.flambda_type =
  let config : G.config = { well_formed = true; depth = 3 } in
  let t = G.type_of_kind config r names.pool kind in
  let smaller =
    List.filter (fun t' -> Stdlib.( = ) (MT.kind t') kind) (G.shrink t)
  in
  let one () = match smaller with [] -> t | _ :: _ -> G.pick r smaller in
  let first = one () in
  let second = one () in
  let third = one () in
  first, second, third

(* Two variants with variables of relations. A variant may name the naked
   immediate that holds whether its value is an immediate and the one that holds
   the tag of its block, and a type that may be null the one that holds whether
   its value is null. Each of the two types names such variables, most often,
   among the two naked immediates of the tests, on immediates and blocks that
   are the same for the two half of the time: so that the meet has two variables
   for one relation, and, where equations before the meet say what a variable
   is, a relation to reduce. *)
let generate_meet_of_relations (names : names) r :
    MT.flambda_type * MT.flambda_type =
  let pool = names.pool in
  let config : G.config = { well_formed = true; depth = 1 } in
  let variable () =
    if G.chance r ~percent:75
    then Some (G.pick r pool.naked_immediate_variables)
    else None
  in
  let is_null () : MT.is_null =
    match G.below r 4 with
    | 0 -> Maybe_null None
    | 1 | 2 -> Maybe_null (Some (G.pick r pool.naked_immediate_variables))
    | _ -> Not_null
  in
  let body () :
      MT.flambda_type MT.Or_unknown.t * MT.row_like_for_blocks MT.Or_unknown.t =
    ( (if G.chance r ~percent:30
       then Unknown
       else
         Known
           (G.type_of_kind config r pool (Naked_number Kind_naked_immediate))),
      if G.chance r ~percent:25
      then Unknown
      else Known (G.row_like_for_blocks config r pool) )
  in
  let variant (immediates, blocks) : MT.flambda_type =
    let is_int = variable () in
    let get_tag = variable () in
    let head : MT.head_of_kind_value_non_null =
      Variant (is_int, immediates, get_tag, blocks, No_extensions, false)
    in
    Type_value (Ok (No_alias (Head_of_kind_value (Ok head, is_null ()))))
  in
  let first = body () in
  let second = if G.chance r ~percent:50 then first else body () in
  let one = variant first in
  let other = variant second in
  one, other

(* Two closure types of a set of closures with the two function slots of the
   tests and one or two of their value slots. Each type has one case, for one of
   the two function slots, whose index is the contents of the set, exactly or at
   least, and whose entry gives most of the slots a function type, a type of the
   closure and a type of the value. The second type has most often the contents,
   the slot and the entry of the first, each part of the entry kept or made
   again. *)
let generate_meet_of_closures (names : names) r :
    MT.flambda_type * MT.flambda_type =
  let pool = names.pool in
  let config : G.config = { well_formed = true; depth = 1 } in
  let contents () : MT.set_of_closures_contents =
    { contents_closures = pool.function_slots;
      contents_value_slots =
        (if G.chance r ~percent:50
         then pool.value_slots
         else [List.hd pool.value_slots])
    }
  in
  let component () : MT.flambda_type =
    match G.below r 4 with
    | 0 -> MT.unknown Value
    | 1 -> MT.alias_type_of Value (Var (G.pick r pool.value_variables))
    | _ -> G.type_of_kind config r pool Value
  in
  let function_type () : MT.function_type MT.Or_unknown.t =
    if G.chance r ~percent:25
    then Unknown
    else Known { function_type_code_id = G.pick r pool.code_ids }
  in
  let most make slots =
    List.filter_map
      (fun slot ->
        if G.chance r ~percent:80 then Some (slot, make ()) else None)
      slots
  in
  let entry (contents : MT.set_of_closures_contents) : MT.closures_entry =
    let function_types = most function_type contents.contents_closures in
    let closure_types = most component contents.contents_closures in
    let value_slot_types = most component contents.contents_value_slots in
    Closures_entry (function_types, closure_types, value_slot_types)
  in
  (* The entry again for contents that may have fewer value slots, each of its
     parts kept or made again. *)
  let again (contents : MT.set_of_closures_contents)
      (Closures_entry (function_types, closure_types, value_slot_types) :
        MT.closures_entry) : MT.closures_entry =
    let part make (slot, old) =
      slot, if G.chance r ~percent:65 then old else make ()
    in
    let function_types = List.map (part function_type) function_types in
    let closure_types = List.map (part component) closure_types in
    let value_slot_types =
      List.map (part component)
        (List.filter
           (fun (slot, (_ : MT.flambda_type)) ->
             List.mem slot contents.contents_value_slots)
           value_slot_types)
    in
    Closures_entry (function_types, closure_types, value_slot_types)
  in
  let closures slot contents entry : MT.flambda_type =
    let case : MT.row_like_closures_case =
      { case_maps_to = entry;
        case_index =
          { index_domain =
              (if G.chance r ~percent:50
               then Known contents
               else At_least contents);
            index_shape = ()
          };
        case_env_extension =
          (if G.chance r ~percent:85
           then MT.empty_env_extension
           else G.env_extension config r pool)
      }
    in
    MT.create_closures (G.alloc_mode config r)
      (Row_like_for_closures [slot, case])
  in
  let slot = G.pick r pool.function_slots in
  let first_contents = contents () in
  let first_entry = entry first_contents in
  let one = closures slot first_contents first_entry in
  let second_slot =
    if G.chance r ~percent:85 then slot else G.pick r pool.function_slots
  in
  let second_contents =
    if G.chance r ~percent:75 then first_contents else contents ()
  in
  let second_entry =
    if G.chance r ~percent:75
    then again second_contents first_entry
    else entry second_contents
  in
  let other = closures second_slot second_contents second_entry in
  one, other

type meet_in_an_environment =
  { equations : (MT.name * MT.flambda_type) list;
    met : MT.flambda_type * MT.flambda_type
  }

let print_meet_in_an_environment ppf { equations; met = m1, m2 } =
  Format.fprintf ppf "@[<v>%a%a@ and@ %a@]"
    (fun ppf ->
      List.iter (fun (name, t) ->
          Format.fprintf ppf "with %a : %a@ " Model_type.print_name name
            Model_type.print t))
    equations Model_type.print m1 Model_type.print m2

let shrink_meet_in_an_environment ({ equations; met = m1, m2 } as input) =
  List.mapi
    (fun i _ ->
      { input with equations = List.filteri (fun j _ -> i <> j) equations })
    equations
  @ List.map (fun m1 -> { input with met = m1, m2 }) (G.shrink m1)
  @ List.map (fun m2 -> { input with met = m1, m2 }) (G.shrink m2)

(* The meet of the model against the n-way meet of the compiler, on two types in
   the environment of the tests, to which equations are added first: by
   [add_equation] of the model, and by that of the compiler with its n-way
   meet. *)
let model_against_compiler (names : names) :
    meet_in_an_environment comparison list =
  let model = the_model Of_the_compiler in
  let model_env, defined = env_of_model names in
  (* The variables of the environment are the first of the numbering. *)
  let threshold =
    List.length
      (List.filter
         (fun (name : MT.name) ->
           match name with Name_var _ -> true | Name_symbol _ -> false)
         defined)
  in
  let in_the_model { equations; met = m1, m2 } =
    let in_the_model =
      let rec add env equations =
        match equations with
        | [] -> Some env
        | (name, t) :: equations ->
          Option.bind (model.add_equation_of_model env name t) (fun env ->
              add env equations)
      in
      match add model_env equations with
      | None -> Nothing_seen "no answer where an equation is added"
      | Some env -> (
        match model.meet_of_model env m1 m2 with
        | None -> Nothing_seen "no answer of the meet"
        | Some Bottom -> Bottom_seen
        | Some (Ok (t, env)) ->
          seen_in_the_model ~threshold defined (Some t) env)
    in
    in_the_model
  in
  let in_the_compiler (implementation : implementation)
      { equations; met = m1, m2 } =
    let in_the_compiler =
      match
        or_fatal_error (fun () ->
            let meet_expanded_head = implementation.meet_expanded_head in
            let env =
              List.fold_left
                (fun env (name, t) ->
                  ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
                      ME.add_equation env
                        (Converter.name_of_model names.numbering name)
                        (to_compiler names t) ~meet_expanded_head))
                names.env equations
            in
            match
              implementation.meet env (to_compiler names m1)
                (to_compiler names m2)
            with
            | Bottom -> Bottom_seen
            | Ok (t, env) ->
              seen_in_the_compiler names.numbering ~threshold defined (Some t)
                env)
      with
      | Ok seen -> seen
      | Error line -> Nothing_seen ("a fatal error: " ^ general line)
    in
    in_the_compiler
  in
  let both input = in_the_model input, in_the_compiler n_way input in
  let different in_the_model in_the_compiler =
    if
      same_seen ~threshold
        (up_to_the_alias_kept in_the_model)
        (up_to_the_alias_kept in_the_compiler)
    then
      Known_to_disagree
        ( "T7-10, the alias that a name keeps is to another member of its \
           class, and the canonical elements are the same",
          "what is seen is the same once the type of a name that is an alias \
           is read as the alias of its canonical element" )
    else
      Disagree
        (Format.asprintf "@[<v>the model:@ %a@ the compiler:@ %a@]" print_lines
           (print_seen ~threshold in_the_model)
           print_lines
           (print_seen ~threshold in_the_compiler))
  in
  let against_the_binary_meet ~label input =
    let in_the_model = in_the_model input in
    let in_the_compiler = in_the_compiler binary input in
    match in_the_model, in_the_compiler with
    | Nothing_seen _, Nothing_seen _ -> Not_run "neither answers"
    | (Nothing_seen _ | Bottom_seen | Seen _), _ ->
      if same_seen ~threshold in_the_model in_the_compiler
      then Agree
      else
        Known_to_disagree
          (label, "what is seen of the two answers is not the same")
  in
  let on_a_symbol { equations = _; met = m1, m2 } =
    List.exists
      (fun symbol ->
        let name : MT.name = Name_symbol symbol in
        has_an_extension_on name m1 || has_an_extension_on name m2)
      names.pool.symbols
  in
  let an_alias_of_a_symbol { equations; met = m1, m2 } =
    has_an_alias_of_a_symbol m1
    || has_an_alias_of_a_symbol m2
    || List.exists
         (fun ((_ : MT.name), t) -> has_an_alias_of_a_symbol t)
         equations
  in
  [ { name = "meet: the model and the n-way meet of the compiler";
      compare =
        (fun input ->
          let in_the_model, in_the_compiler = both input in
          match in_the_model, in_the_compiler with
          | Nothing_seen why, Nothing_seen why' ->
            Not_run
              (Printf.sprintf
                 "neither answers (the model: %s; the compiler: %s)" why why')
          | Nothing_seen why, (Bottom_seen | Seen _) ->
            Known_to_disagree
              ( "the model has no answer where the compiler answers (T4-10, or \
                 the fuel)",
                Format.asprintf "@[<v>the model: %s@ the compiler:@ %a@]" why
                  print_lines
                  (print_seen ~threshold in_the_compiler) )
          | (Bottom_seen | Seen _), _ ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else different in_the_model in_the_compiler)
    };
    { name = "of which those where the meet is not bottom";
      compare =
        (fun input ->
          match both input with
          | (Seen _ as in_the_model), (Seen _ as in_the_compiler) ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else different in_the_model in_the_compiler
          | (Nothing_seen _ | Bottom_seen | Seen _), _ -> Not_this_comparison)
    };
    { name = "of which those where a variable that was made is mentioned";
      compare =
        (fun input ->
          match both input with
          | ( (Seen { of_others = _ :: _; _ } as in_the_model),
              (Seen _ as in_the_compiler) )
          | ( (Seen _ as in_the_model),
              (Seen { of_others = _ :: _; _ } as in_the_compiler) ) ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else different in_the_model in_the_compiler
          | (Nothing_seen _ | Bottom_seen | Seen _), _ -> Not_this_comparison)
    };
    (* The binary meet is not what the model transcribes: where it answers
       otherwise the model is not wrong, and the comparison is shown to tell two
       answers apart. *)
    { name = "meet: the model and the binary meet of the compiler";
      compare =
        against_the_binary_meet
          ~label:
            "the binary meet, which the model does not transcribe, answers \
             otherwise"
    };
    (* The same comparison on the inputs where an extension of one of the two
       types has an equation on a symbol. The n-way meet, and the model with it,
       forgets such an equation where one case is left; the binary meet keeps it
       (T7-9). The inputs are told by their text, not by what the meets do: a
       difference counted here may have another cause. *)
    { name =
        "of which those where an extension of the types met has an equation on \
         a symbol";
      compare =
        (fun input ->
          if on_a_symbol input
          then
            against_the_binary_meet
              ~label:
                "an input selected for T7-9 by its text (an extension has an \
                 equation on a symbol; the cause is not checked), on which the \
                 binary meet answers otherwise"
              input
          else Not_this_comparison)
    };
    (* And on the inputs where one of the two types, or a type that an equation
       before the meet gives a name, is or has within it an alias of a symbol: a
       meet with what the alias is met with may make an equation on the symbol,
       which is forgotten in the same way. Told by the text again. *)
    { name =
        "of which those where a part of the types met, or of the equations \
         before, is an alias of a symbol";
      compare =
        (fun input ->
          if an_alias_of_a_symbol input
          then
            against_the_binary_meet
              ~label:
                "an input selected for T7-9 by its text (a part is an alias of \
                 a symbol; the cause is not checked), on which the binary meet \
                 answers otherwise"
              input
          else Not_this_comparison)
    } ]

(* ======================================================================== *)
(* The join of the model beside the compiler's                              *)
(* ======================================================================== *)

(* The transcription of the n-way join of environments (FlambdaJoinEnv.v), with
   the transcriptions of the join of types and of the meet tied in
   (FlambdaJoinMeet.v), is extracted, and is run here beside the n-way join of
   the compiler ([Join_env.cut_and_n_way_join_with_analysis]), with the choices
   that are the compiler's: on the inputs of the examples of the join, where it
   is the function that the statements are about that is run and not an answer
   copied from them, and on generated types.

   The inputs are those of [join_env], made in the model by the same calls: the
   source is the environment of the tests with the parameter defined; a use is
   the source at the next scope, to which are added the equations that the use
   knows and then the type of the parameter; the join cuts after the scope of
   the source. A use may bind variables of its own, after the cut, and the
   source may define variables for types only. The names that the variables
   made by the model's join must avoid are all the variables numbered for the
   tests.

   What is compared is what can be seen of the two targets, as for the meet:
   whether the target is marked bottom, the type and the canonical element of
   the parameter and of every name of the tests, and the type of every other
   variable that those types mention (the variables that the join made), with
   those variables renamed in the order in which they come. The analysis that
   the join answers with the target is not compared. *)
(* The uses of a continuation with one parameter of a kind, as for the chosen
   joins: the variables that each use binds, what it knows of the names of the
   tests, and the type of what it passes; and the variables that the source
   defines for types only. *)
type join_at_a_continuation =
  { kind_of_parameter : MK.flambda_kind;
    for_types_at_source : M.variable list;
    at_uses : use list
  }

(* The names of which the target of a join is asked, besides the names of the
   tests: the variables that the source defines for types only, the variables
   that the uses bind (the target may still mention one, and say what it is),
   each once, and the parameter. The variables that the uses bind are numbered
   with those of the tests, so they are not among the variables that the join
   made, and are not reached as those are. *)
let names_asked_of_a_join
    ({ kind_of_parameter = _; for_types_at_source; at_uses } :
      join_at_a_continuation) (parameter : M.variable) : MT.name list =
  let bound =
    List.fold_left
      (fun bound (x, _) -> if List.mem x bound then bound else bound @ [x])
      []
      (List.concat_map (fun { binds; _ } -> binds) at_uses)
  in
  List.map (fun x -> MT.Name_var x) (for_types_at_source @ bound @ [parameter])

(* The uses of a continuation with a parameter of kind value, of which most
   bind, after the cut, a naked immediate that holds a relation: whether a value
   is an immediate, the tag of a block, whether a value is null. The value is a
   variable of the tests, or a variable that the use binds too. The use knows
   the two equations that the function of the compiler which adds the relation
   adds, in its order: the naked immediate is the result of the relation on the
   value, and the value is a variant that names the naked immediate (or, for
   [Is_null], may be null and names it). Half of the time it then knows more of
   the value, by a generated type, so that the relation may be reduced. It
   passes the value, a block that holds it, or the tagging of the naked
   immediate. The other uses bind nothing and pass a variable of the tests or a
   generated type.

   The naked immediate is the same variable at every use that binds one. It and
   the other variable that a use binds are among the names asked of the two
   targets: whether a target defines one, what it says of it, and what variables
   of its own it makes for a relation are compared, and nothing of that is
   assumed here. *)
let generate_join_of_relations (names : names) ~(known : bool) r :
    join_at_a_continuation =
  let pool = names.pool in
  let shallow : G.config = { well_formed = true; depth = 1 } in
  let naked_immediate : MK.flambda_kind = Naked_number Kind_naked_immediate in
  let local = Converter.variable_to_model names.numbering names.local in
  let held = Converter.variable_to_model names.numbering names.for_types_only in
  let relation_holds (relation : MT.Relation.t) (scrutinee : M.variable) :
      (MT.name * MT.flambda_type) list =
    let variant is_int get_tag : MT.flambda_type =
      let head : MT.head_of_kind_value_non_null =
        Variant (is_int, Unknown, get_tag, Unknown, No_extensions, false)
      in
      Type_value (Ok (No_alias (Head_of_kind_value (Ok head, Not_null))))
    in
    let may_be_null : MT.flambda_type =
      let head : MT.head_of_kind_value =
        Head_of_kind_value (Unknown, Maybe_null (Some held))
      in
      Type_value (Ok (No_alias head))
    in
    let of_the_result, of_the_scrutinee =
      match relation with
      | Is_int ->
        MT.is_int_for_scrutinee (Name_var scrutinee), variant (Some held) None
      | Get_tag ->
        MT.get_tag_for_block (Name_var scrutinee), variant None (Some held)
      | Is_null -> MT.is_null_for_scrutinee (Name_var scrutinee), may_be_null
    in
    [Name_var held, of_the_result; Name_var scrutinee, of_the_scrutinee]
  in
  let use () : use =
    let learns = if known then generate_equations names r else [] in
    if G.chance r ~percent:75
    then
      let relation = G.relation r in
      let own = G.chance r ~percent:35 in
      let scrutinee = if own then local else G.pick r pool.value_variables in
      let alias = MT.alias_type_of Value (Var scrutinee) in
      let more : (MT.name * MT.flambda_type) list =
        if G.chance r ~percent:50
        then [Name_var scrutinee, G.type_of_kind shallow r pool Value]
        else []
      in
      { (* The types of what is bound say nothing: the equations are among what
           the use knows, so that they are added in the order of the compiler's
           function. *)
        binds =
          (held, MT.unknown naked_immediate)
          :: (if own then [local, MT.unknown Value] else []);
        learns = learns @ relation_holds relation scrutinee @ more;
        passes =
          (match G.below r 5 with
          | 0 ->
            MT.immutable_block false (N.z_of_int 0) (Scannable Value_only) Heap
              [alias]
          | 1 -> MT.tag_immediate (MT.alias_type_of naked_immediate (Var held))
          | _ -> alias)
      }
    else
      { binds = [];
        learns;
        passes =
          (if G.chance r ~percent:50
           then MT.alias_type_of Value (Var (G.pick r pool.value_variables))
           else G.type_of_kind shallow r pool Value)
      }
  in
  let first = use () in
  let second = use () in
  { kind_of_parameter = Value;
    for_types_at_source = [];
    at_uses =
      (if G.chance r ~percent:35
       then [first; second; use ()]
       else [first; second])
  }

(* What is read of the analysis that a join answers with its target, by the two
   functions that Simplify reads it with
   ([Join_env.Analysis.simple_refined_at_join] and
   [fold_variables_created_at_join]) and by their transcriptions. The uses are
   named by their positions. *)
type refined =
  | Not_refined
  | Invariant of M.simple
  (* At each use, the constant that the name is there, if it is one. *)
  | Refined of (int * M.const option) list
  (* The model's reader has no answer: a fatal error of the compiler's. *)
  | No_reading

type analysis_read =
  { (* What the join did to each name asked that the target defines, and to each
       variable that the join made and what is seen reaches. *)
    refined : (MT.name * refined) list;
    (* The variables created at the join, each with what may stand for it at
       each use and its kind. *)
    created : (M.variable * (int * M.simple) list option * MK.flambda_kind) list
  }

let by_use uses = List.sort (fun (i, _) (j, _) -> Int.compare i j) uses

(* The names of which the analysis is asked: those asked of the target, and the
   variables that the join made which what is seen reaches. *)
let names_asked_of_an_analysis (asked : MT.name list) (seen : seen) :
    MT.name list =
  asked
  @
  match seen with
  | Seen { of_others; _ } ->
    List.map
      (fun (x, (_ : MT.flambda_type option)) : MT.name -> Name_var x)
      of_others
  | Nothing_seen _ | Bottom_seen -> []

let analysis_of_the_model (asked : MT.name list) (seen : seen)
    (target : Model_env.t) (analysis : int Model_analysis.t) : analysis_read =
  let refined =
    List.filter_map
      (fun name ->
        if not (Model_env.mem None target name)
        then None
        else
          let refined =
            match
              Model_analysis.simple_refined_at_join_of analysis target
                (MT.simple_of_name name)
            with
            | Not_refined_at_join -> Not_refined
            | Invariant_in_all_uses simple -> Invariant simple
            | Variable_refined_at_these_uses (canonicals, (_ : MK.flambda_kind))
              -> (
              match
                Model_analysis.values_at_uses analysis.external_ids canonicals
              with
              | Some values ->
                Refined
                  (by_use
                     (List.map
                        (fun (use, (value : M.const MT.Or_unknown.t)) ->
                          ( use,
                            match value with
                            | Known const -> Some const
                            | Unknown -> None ))
                        values))
              | None -> No_reading)
          in
          Some (name, refined))
      (names_asked_of_an_analysis asked seen)
  in
  let created =
    List.map
      (fun ((x, simples), kind) ->
        ( x,
          Option.map by_use
            (Model_analysis.definitions_at_uses analysis.external_ids simples),
          kind ))
      (Model_analysis.variables_created_at_join analysis)
  in
  { refined; created }

let analysis_of_the_compiler (names : names) (asked : MT.name list)
    (seen : seen) (target : TE.t) (analysis : int N_way_join.Analysis.t) :
    analysis_read =
  let numbering = names.numbering in
  let refined =
    List.filter_map
      (fun name ->
        let of_the_compiler = Converter.name_of_model numbering name in
        if not (TE.mem target of_the_compiler)
        then None
        else
          let refined =
            match
              N_way_join.Analysis.simple_refined_at_join analysis target
                (Simple.name of_the_compiler)
            with
            | Not_refined_at_join -> Not_refined
            | Invariant_in_all_uses simple ->
              Invariant (Converter.simple_to_model numbering simple)
            | Variable_refined_at_these_uses at_uses ->
              Refined
                (by_use
                   (N_way_join.Analysis.Variable_refined_at_join
                    .fold_values_at_uses
                      (fun use (value : Reg_width_const.t Or_unknown.t) values
                         ->
                        ( use,
                          match value with
                          | Known const -> Some (Converter.const_to_model const)
                          | Unknown -> None )
                        :: values)
                      at_uses []))
          in
          Some (name, refined))
      (names_asked_of_an_analysis asked seen)
  in
  let created =
    N_way_join.Analysis.fold_variables_created_at_join analysis ~init:[]
      ~f:(fun name simples kind created ->
        let x =
          Name.pattern_match name ~var:(Converter.variable_to_model numbering)
            ~symbol:(fun (_ : Symbol.t) ->
              Misc.fatal_error "a symbol among the variables created at a join")
        in
        let definitions =
          N_way_join.Analysis.Simples_at_join.fold_definitions_at_uses
            (fun use (At_normal_mode simple) definitions ->
              (use, Converter.simple_to_model numbering simple) :: definitions)
            simples []
        in
        (x, Some (by_use definitions), Converter.kind_to_model kind) :: created)
  in
  { refined; created }

(* What is read of an analysis, as text, after what is seen of the target: the
   variables that the join made have the names that they have there. The
   variables created come in the order of those names, and those that what is
   seen does not reach after them, in the order of what is read of them. *)
let print_analysis ~threshold (seen : seen)
    ({ refined; created } : analysis_read) : string =
  let own = Hashtbl.create 4 in
  let (_ : string) = print_seen ~own ~threshold seen in
  let print_uses print ppf uses =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.pp_print_string ppf "; ")
      (fun ppf (use, x) -> Format.fprintf ppf "at use %d %a" use print x)
      ppf uses
  in
  let print_refined ppf (refined : refined) =
    match refined with
    | Not_refined -> Format.pp_print_string ppf "not refined at the join"
    | Invariant simple ->
      Format.fprintf ppf "the same at every use, %a" Model_type.print_simple
        simple
    | Refined values ->
      print_uses
        (fun ppf (value : M.const option) ->
          match value with
          | Some const -> Model_type.print_const ppf const
          | None -> Format.pp_print_string ppf "no constant")
        ppf values
    | No_reading -> Format.pp_print_string ppf "no answer"
  in
  let of_the_names =
    List.map
      (fun (name, refined) ->
        Format.asprintf "%a: %a" Model_type.print_name name print_refined
          refined)
      refined
  in
  let what_is_read (definitions, kind) =
    Format.asprintf "of kind %a, %a" Model_type.print_kind kind
      (fun ppf (definitions : (int * M.simple) list option) ->
        match definitions with
        | Some definitions -> print_uses Model_type.print_simple ppf definitions
        | None -> Format.pp_print_string ppf "no answer")
      definitions
  in
  let place x =
    Option.bind (N.int_of_positive x) (fun number ->
        if number <= threshold
        then Some number
        else
          Option.map (fun i -> threshold + 1 + i) (Hashtbl.find_opt own number))
  in
  let created =
    List.sort
      (fun (x, read) (x', read') ->
        match place x, place x' with
        | Some i, Some j -> Int.compare i j
        | Some _, None -> -1
        | None, Some _ -> 1
        | None, None -> String.compare read read')
      (List.map
         (fun (x, definitions, kind) -> x, what_is_read (definitions, kind))
         created)
  in
  let of_the_created =
    List.map
      (fun (x, read) ->
        Format.asprintf "created: %a %s" Model_type.print_variable x read)
      created
  in
  rename_variables_above ~own threshold
    (String.concat "\n" (of_the_names @ of_the_created))

(* The names that the variables which a join of the model makes are to avoid.
   The join that Simplify calls is given them (FlambdaJoinMeet.v, [model_join]);
   the compiler's new variables differ from every other by construction. They
   are the variables of the test and those that the environments joined know,
   which may have variables that a meet or a join before made: it is what the
   model's join that the meet calls avoids ([Join_env.cut_and_n_way_join]). *)
let names_for_a_join_to_avoid (of_the_test : M.variable list)
    ~(source : Model_env.t) ~(uses : Model_env.t list) : M.variable list =
  of_the_test
  @ Model_join_env.variables_of_env source
  @ List.concat_map Model_join_env.variables_of_env uses

let join_in_the_model ?(read = fun _ _ _ _ -> ()) (names : names)
    (model : model)
    ({ kind_of_parameter = kind; for_types_at_source; at_uses } as input :
      join_at_a_continuation) : seen =
  let model_env, defined = env_of_model names in
  let parameter = names.numbering.variables.number (parameter names kind) in
  let kind_of x =
    Converter.kind_to_model
      (Variable.kind (Converter.variable_of_model names.numbering x))
  in
  let define mode env x =
    Model_env.add_variable_definition env x (kind_of x) mode
  in
  let source =
    List.fold_left
      (define Model_name_mode.in_types)
      (Model_env.add_variable_definition model_env parameter kind
         Model_name_mode.normal)
      for_types_at_source
  in
  let use { binds; learns; passes } =
    List.fold_left
      (fun env (name, t) ->
        Option.bind env (fun env -> model.add_equation_of_model env name t))
      (Some
         (List.fold_left
            (fun env (x, _) -> define Model_name_mode.normal env x)
            (Model_env.increment_scope source)
            binds))
      (learns
      @ List.map (fun (x, t) -> MT.Name_var x, t) binds
      @ [MT.Name_var parameter, passes])
  in
  let uses = List.map use at_uses in
  if List.exists Option.is_none uses
  then Nothing_seen "no answer where an equation is added at a use"
  else
    let uses = List.filter_map Fun.id uses in
    let avoid =
      names_for_a_join_to_avoid
        (List.init names.variables_with_a_name (fun i ->
             N.positive_of_int (i + 1)))
        ~source ~uses
    in
    match
      model.join_of_model avoid
        { source;
          cut_after = Model_env.current_scope source;
          uses = List.mapi (fun i env -> i, env) uses
        }
    with
    | None -> Nothing_seen "no answer of the join"
    | Some (target, analysis) ->
      let asked = defined @ names_asked_of_a_join input parameter in
      let seen =
        seen_in_the_model ~threshold:names.variables_with_a_name asked None
          target
      in
      read asked seen target analysis;
      seen

(* The same in the compiler. The conversion of the inputs may refuse them. *)
let join_in_the_compiler ?(read = fun _ _ _ -> ()) (names : names)
    (implementation : implementation)
    ({ kind_of_parameter = kind; for_types_at_source; at_uses } as input :
      join_at_a_continuation) : seen =
  let (_ : Model_env.t), defined = env_of_model names in
  let variable = Converter.variable_of_model names.numbering in
  let known =
    List.map
      (fun { learns; _ } ->
        List.map
          (fun (name, t) ->
            Converter.name_of_model names.numbering name, to_compiler names t)
          learns)
      at_uses
  in
  let locals =
    List.map
      (fun { binds; _ } ->
        List.map (fun (x, t) -> variable x, to_compiler names t) binds)
      at_uses
  in
  let types =
    List.map (fun { passes; _ } -> to_compiler names passes) at_uses
  in
  let in_types = List.map variable for_types_at_source in
  match
    or_fatal_error (fun () ->
        match
          join_env names implementation ~known:(List.nth known)
            ~locals:(List.nth locals) ~in_types kind types
        with
        | env, parameter ->
          let asked =
            defined
            @ names_asked_of_a_join input
                (names.numbering.variables.number parameter)
          in
          let seen =
            seen_in_the_compiler names.numbering
              ~threshold:names.variables_with_a_name asked None env
          in
          read asked seen env;
          seen
        | exception Assert_failure (file, _, _) ->
          Nothing_seen ("an assertion of " ^ Filename.basename file ^ " fails"))
  with
  | Ok seen -> seen
  | Error line -> Nothing_seen ("a fatal error: " ^ general line)

(* What becomes of the reading of the analysis of a join that answered a target:
   what is read, or why a reader stopped. *)
type analysis_reading =
  | Read of analysis_read
  | Reader_stopped of string

(* A join in the model, and in the compiler with its n-way join, each with the
   reading of the analysis that the join answers, where it answers a target. The
   analysis of the compiler is read once the join has answered, apart from it: a
   reader that stops with a fatal error does not take the target with it.
   [reader] is the reader, which a control replaces. *)
let join_and_analysis_in_the_model (names : names) (model : model)
    (input : join_at_a_continuation) : seen * analysis_reading option =
  let of_the_model = ref None in
  let seen =
    join_in_the_model names model input ~read:(fun asked seen target analysis ->
        of_the_model
          := Some (Read (analysis_of_the_model asked seen target analysis)))
  in
  seen, !of_the_model

let join_and_analysis_in_the_compiler ?(reader = analysis_of_the_compiler)
    (names : names) (input : join_at_a_continuation) :
    seen * analysis_reading option =
  let analysis = ref None in
  let with_its_analysis : implementation =
    { n_way with
      join =
        (fun env use_envs ~params:_ ~cut_after ->
          let target, of_the_join =
            N_way_join.cut_and_n_way_join_with_analysis
              ~meet_expanded_head:N_way.meet_expanded_head
              ~n_way_join_type:N_way.n_way_join env ~cut_after
              (List.mapi (fun i use_env -> i, use_env) use_envs)
          in
          analysis := Some of_the_join;
          target)
    }
  in
  let answered = ref None in
  let seen =
    join_in_the_compiler names with_its_analysis input
      ~read:(fun asked seen target -> answered := Some (asked, seen, target))
  in
  let reading =
    match !answered, !analysis with
    | Some (asked, seen, target), Some analysis -> (
      match
        or_fatal_error (fun () -> reader names asked seen target analysis)
      with
      | Ok read -> Some (Read read)
      | Error line -> Some (Reader_stopped ("a fatal error: " ^ general line)))
    | Some _, None | None, (Some _ | None) -> None
  in
  seen, reading

let a_variable_is_created ({ created; refined = _ } : analysis_read) =
  match created with [] -> false | _ :: _ -> true

let a_name_is_refined ({ refined; created = _ } : analysis_read) =
  List.exists
    (fun ((_ : MT.name), (refined : refined)) ->
      match refined with
      | Refined _ -> true
      | Not_refined | Invariant _ | No_reading -> false)
    refined

let a_reader_has_no_answer ({ refined; created } : analysis_read) =
  List.exists
    (fun ((_ : MT.name), (refined : refined)) ->
      match refined with
      | No_reading -> true
      | Not_refined | Invariant _ | Refined _ -> false)
    refined
  || List.exists
       (fun ((_ : M.variable), definitions, (_ : MK.flambda_kind)) ->
         Option.is_none definitions)
       created

(* The comparison of the readings of the two analyses of one join, in the model
   and in the compiler, where both joins answer a target and what is seen of the
   two targets is the same; elsewhere it is not this comparison, and that of the
   targets has said what there is to say. With [only], the comparison is of the
   joins whose reading, on one side or the other, [only] accepts.

   Where the targets are the same, a reading that is missing on one side is a
   disagreement: the reader of the compiler stopped where the model's has an
   answer everywhere, or no analysis was read of a join that answered. Where the
   reader of the compiler stops and a reader of the model has no answer, neither
   reads, and the two reasons are shown. *)
let compare_analyses ~threshold ?only
    ((in_the_model : seen), (of_the_model : analysis_reading option))
    ((in_the_compiler : seen), (of_the_compiler : analysis_reading option)) :
    outcome =
  let whole = Option.is_none only in
  let wanted read = match only with None -> true | Some only -> only read in
  let one_sided what = if whole then Disagree what else Not_this_comparison in
  match in_the_model, in_the_compiler with
  | Seen _, Seen _ -> (
    if not (same_seen ~threshold in_the_model in_the_compiler)
    then Not_this_comparison
    else
      match of_the_model, of_the_compiler with
      | Some (Read of_the_model), Some (Read of_the_compiler) ->
        if not (wanted of_the_model || wanted of_the_compiler)
        then Not_this_comparison
        else
          let in_the_model =
            print_analysis ~threshold in_the_model of_the_model
          in
          let in_the_compiler =
            print_analysis ~threshold in_the_compiler of_the_compiler
          in
          if same_text in_the_model in_the_compiler
          then Agree
          else
            Disagree
              (Format.asprintf "@[<v>the model:@ %a@ the compiler:@ %a@]"
                 print_lines in_the_model print_lines in_the_compiler)
      | Some (Read of_the_model), Some (Reader_stopped why) ->
        if not whole
        then Not_this_comparison
        else if a_reader_has_no_answer of_the_model
        then
          Not_run
            (Printf.sprintf
               "neither reads the analysis (the model: no answer of a reader; \
                the compiler: %s)"
               why)
        else
          Disagree
            (Printf.sprintf
               "the two targets are the same, and the reader of the compiler's \
                analysis stops where the model's reads: %s"
               why)
      | Some (Reader_stopped why), (Some (Read _ | Reader_stopped _) | None) ->
        one_sided
          (Printf.sprintf
             "the two targets are the same, and the reader of the model's \
              analysis stops: %s"
             why)
      | Some (Read _), None | None, (Some (Read _ | Reader_stopped _) | None) ->
        one_sided
          "the two targets are the same, and no analysis was read of one of \
           the two joins")
  | (Nothing_seen _ | Bottom_seen | Seen _), _ -> Not_this_comparison

let joins_in_the_model (names : names) ~title
    (joins : (string * join_at_a_continuation) list) =
  Format.printf "@.%s@." title;
  let model = the_model Of_the_compiler in
  let threshold = names.variables_with_a_name in
  let same = ref 0 and neither = ref 0 and different = ref 0 in
  let not_run = ref 0 in
  List.iter
    (fun (name, input) ->
      match join_in_the_compiler names n_way input with
      | exception Converter.Not_in_the_compiler what ->
        incr not_run;
        Format.printf "  %s: not run, not in the compiler: %s@." name what
      | in_the_compiler -> (
        let in_the_model = join_in_the_model names model input in
        match in_the_model, in_the_compiler with
        | Nothing_seen why, Nothing_seen why' ->
          incr neither;
          Format.printf
            "  @[<v>%s: neither answers.@ The model: %s@ The compiler: %s@]@."
            name why why'
        | (Nothing_seen _ | Bottom_seen | Seen _), _ ->
          if same_seen ~threshold in_the_model in_the_compiler
          then (
            incr same;
            Format.printf "  %s: the same@." name)
          else (
            incr different;
            Format.printf
              "  @[<v>%s: DIFFERENT. The model:@ %a@ the compiler:@ %a@]@." name
              print_lines
              (print_seen ~threshold in_the_model)
              print_lines
              (print_seen ~threshold in_the_compiler))))
    joins;
  Format.printf
    "  Of %d: %d the same, %d where neither answers, %d different, %d not run@."
    (List.length joins) !same !neither !different !not_run

let examples_of_joins_in_the_model (names : names) =
  joins_in_the_model names
    ~title:
      "The join of the model beside the n-way join of the compiler, on the \
       inputs of the examples of the join"
    (List.map
       (fun { example; source = _; knowing; inputs; answer = _ } ->
         let learns =
           List.map
             (fun (x, t) : (MT.name * MT.flambda_type) -> Name_var x, t)
             knowing
         in
         ( example,
           { kind_of_parameter = MT.kind (List.hd inputs);
             for_types_at_source = [];
             at_uses =
               List.map (fun passes -> { binds = []; learns; passes }) inputs
           } ))
       (examples_of_joins names))

let chosen_joins_in_the_model (names : names) =
  joins_in_the_model names
    ~title:
      "The join of the model beside the n-way join of the compiler, on the \
       chosen joins"
    (List.map
       (fun { title; kind; in_types; known; uses; probes = _; recorded = _ } ->
         ( title,
           { kind_of_parameter = kind;
             for_types_at_source = in_types;
             at_uses =
               List.map
                 (fun (use : use) -> { use with learns = known @ use.learns })
                 uses
           } ))
       (the_chosen_joins names))

(* A control of what is asked of a join: two joins that differ only in what a
   variable bound by one use is known to be. The parameter is in both a block
   whose field is an alias of that variable, and every name of the tests is as
   it was: the two targets differ in the type of that variable alone, and are to
   be told apart. One of the two is also to be the same as itself. *)
let control_of_the_joins (names : names) =
  let z = N.z_of_int in
  let local = Converter.variable_to_model names.numbering names.local in
  let join numbers : join_at_a_continuation =
    { kind_of_parameter = Value;
      for_types_at_source = [];
      at_uses =
        [ { binds = []; learns = []; passes = MT.this_tagged_immediate (z 0) };
          { binds = [local, MT.these_tagged_immediates (List.map z numbers)];
            learns = [];
            passes =
              MT.immutable_block false (z 0) (Scannable Value_only) Heap
                [MT.alias_type_of Value (Var local)]
          } ]
    }
  in
  let threshold = names.variables_with_a_name in
  let model = the_model Of_the_compiler in
  let told_apart seen seen' = not (same_seen ~threshold seen seen') in
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: two joins that differ only in what a variable bound by one \
     use is known to be (4 or 5; 4 or 6)@.";
  let in_the_compiler numbers =
    join_in_the_compiler names n_way (join numbers)
  in
  let in_the_model numbers = join_in_the_model names model (join numbers) in
  check "the two targets of the compiler are told apart"
    (told_apart (in_the_compiler [4; 5]) (in_the_compiler [4; 6]));
  check "the two targets of the model are told apart"
    (told_apart (in_the_model [4; 5]) (in_the_model [4; 6]));
  check
    "the target of the model for the one and of the compiler for the other are \
     told apart"
    (told_apart (in_the_model [4; 5]) (in_the_compiler [4; 6]));
  check "the targets of the model and of the compiler for the one are the same"
    (same_seen ~threshold (in_the_model [4; 5]) (in_the_compiler [4; 5]))

(* A control of what is read of an analysis: two joins of two uses that pass a
   block of two fields, both the number 4 at one use and both the number 5 at
   the other, and the same with the two uses exchanged. The two targets are the
   same: a block whose two fields are one variable that the join made, which is
   4 or 5. The analyses are not: the variable is 4 at the first use in the one,
   and 5 in the other. *)
let control_of_the_analysis (names : names) =
  let z = N.z_of_int in
  let join numbers : join_at_a_continuation =
    { kind_of_parameter = Value;
      for_types_at_source = [];
      at_uses =
        List.map
          (fun number ->
            let field = MT.this_tagged_immediate (z number) in
            { binds = [];
              learns = [];
              passes =
                MT.immutable_block false (z 0) (Scannable Value_only) Heap
                  [field; field]
            })
          numbers
    }
  in
  let threshold = names.variables_with_a_name in
  let model = the_model Of_the_compiler in
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: two joins that differ only in which of two uses passes which \
     of two blocks (4 and 4, 5 and 5)@.";
  let in_the_compiler numbers =
    join_and_analysis_in_the_compiler names (join numbers)
  in
  let in_the_model numbers =
    join_and_analysis_in_the_model names model (join numbers)
  in
  let both_read ((_ : seen), read) ((_ : seen), read')
      ~(f : analysis_read -> analysis_read -> bool) =
    match read, read' with
    | Some (Read read), Some (Read read') -> f read read'
    | Some (Read _), (Some (Reader_stopped _) | None)
    | Some (Reader_stopped _), (Some (Read _ | Reader_stopped _) | None)
    | None, (Some (Read _ | Reader_stopped _) | None) ->
      false
  in
  let same_analysis (seen, read) (seen', read') =
    both_read (seen, read) (seen', read') ~f:(fun read read' ->
        same_text
          (print_analysis ~threshold seen read)
          (print_analysis ~threshold seen' read'))
  in
  let answers ((seen : seen), (_ : analysis_reading option)) =
    match seen with Seen _ -> true | Nothing_seen _ | Bottom_seen -> false
  in
  let is_a_disagreement (outcome : outcome) =
    match outcome with
    | Disagree _ -> true
    | Agree | Not_run _ | Known_to_disagree _ | Not_this_comparison -> false
  in
  let compiler = in_the_compiler [4; 5] in
  let compiler' = in_the_compiler [5; 4] in
  let of_the_model = in_the_model [4; 5] in
  let of_the_model' = in_the_model [5; 4] in
  check "the four joins answer a target"
    (List.for_all answers [compiler; compiler'; of_the_model; of_the_model']);
  check
    "the targets of the model and of the compiler are the same, for the one \
     and for the other"
    (same_seen ~threshold (fst of_the_model) (fst compiler)
    && same_seen ~threshold (fst of_the_model') (fst compiler'));
  check "the two targets of the compiler are the same"
    (same_seen ~threshold (fst compiler) (fst compiler'));
  check "a variable is created at the join and a name is refined at it"
    (both_read compiler of_the_model ~f:(fun read read' ->
         a_variable_is_created read && a_name_is_refined read
         && a_variable_is_created read'
         && a_name_is_refined read'));
  check "what is read of the two analyses of the compiler is told apart"
    (both_read compiler compiler' ~f:(fun _ _ -> true)
    && not (same_analysis compiler compiler'));
  check "what is read of the two analyses of the model is told apart"
    (both_read of_the_model of_the_model' ~f:(fun _ _ -> true)
    && not (same_analysis of_the_model of_the_model'));
  check
    "what is read of the analysis of the model for the one and of the compiler \
     for the other is told apart"
    (both_read of_the_model compiler' ~f:(fun _ _ -> true)
    && not (same_analysis of_the_model compiler'));
  check
    "what is read of the analyses of the model and of the compiler for the one \
     is the same"
    (same_analysis of_the_model compiler);
  (* A reader that stops, once the join has answered: the target is as it was,
     and the comparison of the analyses says so, where a reading of the model
     with no answer in it would make it one that neither reads. *)
  let stopped =
    join_and_analysis_in_the_compiler names
      (join [4; 5])
      ~reader:(fun _ _ _ _ _ ->
        Misc.fatal_error "a reader of the analysis that stops, for a control")
  in
  check
    "with a reader of the compiler's analysis that stops, the target of the \
     compiler is as it was"
    (answers stopped && same_seen ~threshold (fst stopped) (fst compiler));
  check "and the comparison of the two analyses is a disagreement"
    (is_a_disagreement (compare_analyses ~threshold of_the_model stopped));
  check
    "a reading of the model in which a reader has no answer, against what the \
     compiler reads, is a disagreement too"
    (is_a_disagreement
       (compare_analyses ~threshold
          ( fst of_the_model,
            Some
              (Read
                 { refined =
                     [ ( Name_var
                           (names.numbering.variables.number
                              (parameter names Value)),
                         No_reading ) ];
                   created = []
                 }) )
          compiler))

let print_join_at_a_continuation ppf
    ({ kind_of_parameter; for_types_at_source = _; at_uses } :
      join_at_a_continuation) =
  Format.fprintf ppf "@[<v>a parameter of kind %a%a@]" Model_type.print_kind
    kind_of_parameter
    (fun ppf ->
      List.iter (fun { binds; learns; passes } ->
          Format.fprintf ppf "@ a use that passes %a%a%a" Model_type.print
            passes
            (fun ppf ->
              List.iter (fun (x, t) ->
                  Format.fprintf ppf "@   and binds %a : %a"
                    Model_type.print_variable x Model_type.print t))
            binds
            (fun ppf ->
              List.iter (fun (name, t) ->
                  Format.fprintf ppf "@   and knows %a : %a"
                    Model_type.print_name name Model_type.print t))
            learns))
    at_uses

(* A use less, if more than two are left; an equation less; a smaller type of
   what is passed. *)
let shrink_join_at_a_continuation (input : join_at_a_continuation) =
  let with_use i (use : use) =
    { input with
      at_uses =
        List.mapi (fun j use' -> if i = j then use else use') input.at_uses
    }
  in
  (if List.length input.at_uses > 2
   then
     List.mapi
       (fun i _ ->
         let at_uses = List.filteri (fun j _ -> i <> j) input.at_uses in
         { input with at_uses })
       input.at_uses
   else [])
  @ List.concat
      (List.mapi
         (fun i (use : use) ->
           List.mapi
             (fun k _ ->
               with_use i
                 { use with
                   learns = List.filteri (fun l _ -> k <> l) use.learns
                 })
             use.learns
           @ List.map
               (fun passes -> with_use i { use with passes })
               (G.shrink use.passes))
         input.at_uses)

(* The join of the model against the n-way join of the compiler. *)
let model_join_against_compiler (names : names) :
    join_at_a_continuation comparison list =
  let model = the_model Of_the_compiler in
  let threshold = names.variables_with_a_name in
  let both input =
    join_in_the_model names model input, join_in_the_compiler names n_way input
  in
  let different in_the_model in_the_compiler =
    Disagree
      (Format.asprintf "@[<v>the model:@ %a@ the compiler:@ %a@]" print_lines
         (print_seen ~threshold in_the_model)
         print_lines
         (print_seen ~threshold in_the_compiler))
  in
  let analyses ?only input =
    compare_analyses ~threshold ?only
      (join_and_analysis_in_the_model names model input)
      (join_and_analysis_in_the_compiler names input)
  in
  [ { name = "join: the model and the n-way join of the compiler";
      compare =
        (fun input ->
          let in_the_model, in_the_compiler = both input in
          match in_the_model, in_the_compiler with
          | Nothing_seen why, Nothing_seen why' ->
            Not_run
              (Printf.sprintf
                 "neither answers (the model: %s; the compiler: %s)" why why')
          | Nothing_seen why, (Bottom_seen | Seen _) ->
            Known_to_disagree
              ( "the model has no answer where the compiler answers",
                Format.asprintf "@[<v>the model: %s@ the compiler:@ %a@]" why
                  print_lines
                  (print_seen ~threshold in_the_compiler) )
          | (Bottom_seen | Seen _), _ ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else different in_the_model in_the_compiler)
    };
    { name = "of which those where a variable that was made is mentioned";
      compare =
        (fun input ->
          match both input with
          | ( (Seen { of_others = _ :: _; _ } as in_the_model),
              (Seen _ as in_the_compiler) )
          | ( (Seen _ as in_the_model),
              (Seen { of_others = _ :: _; _ } as in_the_compiler) ) ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else different in_the_model in_the_compiler
          | (Nothing_seen _ | Bottom_seen | Seen _), _ -> Not_this_comparison)
    };
    (* The analysis that each join answers, read by the functions that Simplify
       reads it with and by their transcriptions, where what is seen of the two
       targets is the same; and the same again where a variable was created at
       the join, and where a name is refined at it. *)
    { name = "the analysis of the join, where the two targets are the same";
      compare = (fun input -> analyses input)
    };
    { name = "of which those where a variable was created at the join";
      compare = analyses ~only:a_variable_is_created
    };
    { name = "of which those where a name is refined at the join";
      compare = analyses ~only:a_name_is_refined
    };
    (* The binary join is not what the model transcribes (T7-3): the comparison
       is shown to tell two answers apart. *)
    { name = "join: the model and the binary join of the compiler";
      compare =
        (fun input ->
          let in_the_model = join_in_the_model names model input in
          let in_the_compiler = join_in_the_compiler names binary input in
          match in_the_model, in_the_compiler with
          | Nothing_seen _, Nothing_seen _ -> Not_run "neither answers"
          | (Nothing_seen _ | Bottom_seen | Seen _), _ ->
            if same_seen ~threshold in_the_model in_the_compiler
            then Agree
            else
              Known_to_disagree
                ( "T7-3, the binary join, which the model does not transcribe, \
                   answers otherwise",
                  "what is seen of the two answers is not the same" ))
    } ]

(* ======================================================================== *)
(* The comparison by text                                                   *)
(* ======================================================================== *)

(* An environment in which a variable of the tests is a block whose field is a
   variable that is not of the tests, which is one of two numbers, and in which
   the parameter is that variable of the tests. What the environment says of the
   variable that is not of the tests is reached from the type of the name only.
   Two such environments that differ in the numbers print differently when the
   names are among the roots, as for the shared parts; from the parameter alone
   they print alike, which is the limit of the comparisons of two joins that
   print the parameter only. *)
let reached_by_a_name_only (names : names) : (string * bool) list =
  let x =
    Converter.variable_of_model names.numbering
      (List.hd names.pool.value_variables)
  in
  let parameter = parameter names Value in
  let environment numbers =
    let e = Variable.create "e" K.value in
    let define env variable =
      TE.add_definition env
        (Bound_name.create_var
           (Bound_var.create variable Flambda_debug_uid.none Name_mode.normal))
        K.value
    in
    let meet_expanded_head = n_way.meet_expanded_head in
    let block_of_e =
      MTC.immutable_block ~machine_width ~is_unique:false
        (Converter.tag_of_model machine_width (N.z_of_int 0))
        ~shape:(Converter.block_shape_of_model (Scannable Value_only))
        (Converter.alloc_mode_of_model Heap)
        ~fields:[TG.alias_type_of K.value (Simple.var e)]
    in
    ME.use_meet_env ~meet_expanded_head
      (define (define names.env parameter) e)
      ~f:(fun env ->
        List.fold_left
          (fun env (variable, t) ->
            ME.add_equation env (Name.var variable) t ~meet_expanded_head)
          env
          [ ( e,
              to_compiler names
                (MT.these_tagged_immediates (List.map N.z_of_int numbers)) );
            x, block_of_e;
            parameter, TG.alias_type_of K.value (Simple.var x) ])
  in
  let with_names numbers =
    print_joined_with_names names (environment numbers) parameter
  in
  let parameter_only numbers =
    print_joined names (joined_of_env names (environment numbers) parameter)
  in
  [ ( "a variable that only a name of the tests reaches, of two numbers and of \
       two others, printed with the names",
      not (same_text (with_names [0; 1]) (with_names [2; 3])) );
    ( "the same, printed from the parameter alone (one text)",
      same_text (parameter_only [0; 1]) (parameter_only [2; 3]) ) ]

(* Where two answers of the compiler are compared as text, the variables that
   are not of the tests are renamed over the whole text. So two types that are
   aliases of one such variable print as two types that are aliases of another
   one do, and not as aliases of two different ones do: a renaming of each type
   on its own would print all three alike. *)
let comparison_by_text (names : names) =
  let own i = N.positive_of_int (names.variables_with_a_name + i) in
  let alias i = MT.alias_type_of Value (Var (own i)) in
  let text = print_with_own_variables names in
  let cases =
    [ ( "two aliases of one variable, and of another one",
        same_text (text [alias 1; alias 1]) (text [alias 2; alias 2]) );
      ( "two aliases of one variable, and aliases of two",
        let of_one = text [alias 1; alias 1] in
        let of_two = text [alias 2; alias 3] in
        not (same_text of_one of_two) );
      ( "what a join says of one variable of its own, and of two",
        not
          (same_text
             (print_joined names
                { joined = alias 1; own = [own 1, alias 1]; own_in_terms = [] })
             (print_joined names
                { joined = alias 2;
                  own = [own 2, alias 3; own 3, alias 3];
                  own_in_terms = []
                })) );
      ( "a type with a variable of its own, and with one whose number is \
         longer, so that the lines are broken elsewhere",
        let variant i =
          MT.create_variant false
            (Known (MT.this_naked_immediate (N.z_of_int 0)))
            (Known
               (MT.row_like_for_blocks_create_at_least (N.z_of_int 0)
                  (N.nat_of_int 1) (Scannable Value_only)
                  [alias i]
                  Heap))
            No_extensions
        in
        same_text (text [variant 1]) (text [variant 100000]) ) ]
    @ reached_by_a_name_only names
  in
  let wrong = List.filter (fun (_, as_expected) -> not as_expected) cases in
  Format.printf "@.The comparison by text (%d cases): %d as expected@."
    (List.length cases)
    (List.length cases - List.length wrong);
  List.iter (fun (what, _) -> Format.printf "  WRONG: %s@." what) wrong;
  failures := !failures + List.length wrong

(* ======================================================================== *)
(* What the model has not got of [Rec_info]                                 *)
(* ======================================================================== *)

(* The comparisons of one type leave out what the converter drops of [Rec_info]
   (the [rec_info] of a function type, the expression of a [Rec_info] head), and
   nothing else. These are cases made by hand, each with the outcome that a
   comparison is to have on it: - an alias to a variable of kind [Rec_info],
   which the model has, against a type of the model without it: the free
   variables differ, and that is a disagreement; - a function type whose
   [rec_info] mentions a variable: the free variables are not compared as they
   are, they agree once the variables of what was dropped are added, and the
   type comes back the same with the [rec_info] put back; - a [Rec_info] head,
   the successor of a variable: the same for the free variables, and the type
   does not come back; - two closures that differ in the last byte of a long
   string and in the [rec_info]: with the [rec_info] of the first put into the
   second, the compiler's two equalities tell the two apart, though they are
   printed alike (the printer of the compiler shows the start of a long string);
   - a type in which one code id has a [rec_info] in one function type and the
   unknown one in another: the [rec_info]s cannot be put back by code id, and
   the type is not compared. *)
let rec_info_cases (names : names) =
  let numbering = names.numbering in
  let depth_of_model = List.hd names.pool.rec_info_variables in
  let depth = Converter.variable_of_model numbering depth_of_model in
  let slot_of_model = List.hd names.pool.function_slots in
  let value_slot_of_model = List.hd names.pool.value_slots in
  let code_of_model = List.hd names.pool.code_ids in
  let code = Converter.code_id_of_model numbering code_of_model in
  let rec_info = TG.this_rec_info (Rec_info_expr.var depth) in
  (* A closure of the model whose value slot holds a string. *)
  let closure (bytes : string) : MT.flambda_type =
    MT.exactly_this_closure slot_of_model
      [ ( slot_of_model,
          MT.Or_unknown.Known
            ({ function_type_code_id = code_of_model } : MT.function_type) ) ]
      []
      [ ( value_slot_of_model,
          MT.this_immutable_string
            (List.map
               (fun c -> N.z_of_int (Char.code c))
               (List.of_seq (String.to_seq bytes))) ) ]
      Heap
  in
  let with_rec_info m =
    Converter.type_of_model_with_rec_infos numbering machine_width
      ~rec_infos:[code, rec_info]
      m
  in
  let comparisons =
    Type_comparisons.comparisons_of_one_type numbering
    @ comparisons_of_what_is_dropped numbering
    @ Type_comparisons.round_trip_from_the_compiler numbering names.env
  in
  let outcome name (both : both) =
    let comparison =
      List.find
        (fun { name = name'; compare = _ } -> String.equal name name')
        comparisons
    in
    match guarded comparison.compare both with
    | Agree -> "agree"
    | Not_run _ -> "not run"
    | Disagree _ -> "disagree"
    | Known_to_disagree _ -> "recorded"
    | Not_this_comparison -> "not this comparison"
  in
  let of_compiler compiler : both =
    { model = to_model names compiler; compiler }
  in
  let exact = "free_variables against free_names, without shadowed bindings" in
  let with_dropped =
    "free_variables, with the variables of what the converter drops, against \
     free_names"
  in
  let as_printed = "the round trip from the compiler, as printed" in
  let for_equal_types =
    "the round trip from the compiler, for the compiler's equality"
  in
  let for_equal = "the round trip from the compiler, for Type_grammar.equal" in
  let alias_without : both =
    { model = MT.unknown Rec_info;
      compiler = TG.alias_type_of K.rec_info (Simple.var depth)
    }
  in
  let function_type = of_compiler (with_rec_info (closure "abc")) in
  (* [this_rec_info] of a variable alone is an alias to it, which the model has:
     a head is an expression that is more than a variable. *)
  let head =
    of_compiler
      (TG.this_rec_info (Rec_info_expr.succ (Rec_info_expr.var depth)))
  in
  (* The second closure, with the [rec_info] of the first put into it. *)
  let first = with_rec_info (closure "aaaaaaaaaaaaX") in
  let second_with_the_rec_info_of_the_first =
    let (_ : MT.flambda_type), (dropped : Converter.dropped) =
      Converter.type_to_model_with_what_is_dropped numbering first
    in
    Converter.type_of_model_with_rec_infos numbering machine_width
      ~rec_infos:dropped.rec_infos (closure "aaaaaaaaaaaaY")
  in
  (* A closure whose function type has the [rec_info], and whose value slot
     holds a closure of the same code with the unknown one. *)
  let two_rec_infos_for_one_code =
    let slot = Converter.function_slot_of_model numbering slot_of_model in
    let value_slot =
      Converter.value_slot_of_model numbering value_slot_of_model
    in
    let closure rec_info value_slots =
      MTC.exactly_this_closure slot
        ~all_function_slots_in_set:
          (Function_slot.Map.singleton slot
             (Or_unknown.Known (TG.Function_type.create code ~rec_info)))
        ~all_closure_types_in_set:Function_slot.Map.empty
        ~all_value_slots_in_set:value_slots
        (Converter.alloc_mode_of_model Heap)
    in
    of_compiler
      (closure rec_info
         (Value_slot.Map.singleton value_slot
            (closure TG.any_rec_info Value_slot.Map.empty)))
  in
  let expect what expected actual = what, String.equal expected actual in
  let cases =
    [ expect "an alias to a variable that the model's type has not: exact"
        "disagree"
        (outcome exact alias_without);
      expect
        "an alias to a variable that the model's type has not: with what is \
         dropped"
        "disagree"
        (outcome with_dropped alias_without);
      expect "a function type with a rec_info of a variable: exact" "not run"
        (outcome exact function_type);
      expect
        "a function type with a rec_info of a variable: with what is dropped"
        "agree"
        (outcome with_dropped function_type);
      expect "a function type with a rec_info of a variable: as printed" "agree"
        (outcome as_printed function_type);
      expect
        "a function type with a rec_info of a variable: the compiler's equality"
        "agree"
        (outcome for_equal_types function_type);
      expect "a function type with a rec_info of a variable: Type_grammar.equal"
        "agree"
        (outcome for_equal function_type);
      expect "a Rec_info head that mentions a variable: exact" "not run"
        (outcome exact head);
      expect "a Rec_info head that mentions a variable: with what is dropped"
        "agree"
        (outcome with_dropped head);
      expect "a Rec_info head that mentions a variable: as printed" "not run"
        (outcome as_printed head);
      ( "two closures that differ in the end of a long string: printed alike",
        String.equal
          (print_compiler_type first)
          (print_compiler_type second_with_the_rec_info_of_the_first) );
      ( "two closures that differ in the end of a long string: not the same \
         for Type_grammar.equal",
        not (TG.equal first second_with_the_rec_info_of_the_first) );
      ( "two closures that differ in the end of a long string: not the same \
         for the compiler's equality",
        not
          (Equal.equal_type
             ~config:(Equal.create_config ~ignore_alloc_mode:false ())
             ~meet_expanded_head:binary.meet_expanded_head names.env first
             second_with_the_rec_info_of_the_first) );
      expect "one code id with a rec_info and with the unknown one: as printed"
        "not run"
        (outcome as_printed two_rec_infos_for_one_code);
      expect
        "one code id with a rec_info and with the unknown one: \
         Type_grammar.equal"
        "not run"
        (outcome for_equal two_rec_infos_for_one_code) ]
  in
  let wrong = List.filter (fun (_, as_expected) -> not as_expected) cases in
  Format.printf
    "@.What the model has not got of Rec_info (%d cases): %d as expected@."
    (List.length cases)
    (List.length cases - List.length wrong);
  List.iter (fun (what, _) -> Format.printf "  WRONG: %s@." what) wrong;
  failures := !failures + List.length wrong

(* ======================================================================== *)
(* The ranges of numbers                                                    *)
(* ======================================================================== *)

(* Numbers of the model at the ends of the range of each kind, and the numbers
   next to them outside. An integer of n bits has four (the greatest, the least,
   and the one beyond each), and so has a bit pattern of n bits, which is read
   as such an integer. Of each, [const_in_range] of the model
   (FlambdaPrimitiveSpec.v) says whether it is in range; the converter takes one
   that is to the compiler and back unchanged, and refuses one that is not. The
   numbers are made with the arithmetic of the model, not with the functions of
   model_numbers.ml that are under test.

   Then constants made in the compiler whose top bit is set (the floats that are
   minus zero, patterns of ones): the converter gives for each a constant that
   is in the range of the model, and takes it back to the same constant of the
   compiler. *)
let ranges_of_numbers () =
  let z = N.z_of_int in
  let ends (const : N.z -> M.const) ~bits =
    let limit = Z.pow (z 2) (z (bits - 1)) in
    [ const (Z.sub limit (z 1)), true;
      const limit, false;
      const (Z.opp limit), true;
      const (Z.sub (Z.opp limit) (z 1)), false ]
  in
  let cases : (M.const * bool) list =
    ends (fun i -> Naked_immediate i) ~bits:63
    @ ends (fun i -> Tagged_immediate i) ~bits:63
    @ ends (fun i -> Naked_int8 i) ~bits:8
    @ ends (fun i -> Naked_int16 i) ~bits:16
    @ ends (fun i -> Naked_int32 i) ~bits:32
    @ ends (fun i -> Naked_int64 i) ~bits:64
    @ ends (fun i -> Naked_nativeint i) ~bits:64
    @ ends (fun f -> Naked_float32 f) ~bits:32
    @ ends (fun f -> Naked_float f) ~bits:64
    @ ends (fun m -> Naked_mask m) ~bits:64
    @ ends (fun v -> Naked_vec128 v) ~bits:128
    @ ends (fun v -> Naked_vec256 v) ~bits:256
    @ ends (fun v -> Naked_vec512 v) ~bits:512
  in
  let wrong =
    List.filter
      (fun (const, in_range) ->
        (not (Bool.equal (Primitive_spec.const_in_range const) in_range))
        ||
        match Converter.const_of_model machine_width const with
        | converted ->
          let back = Converter.const_to_model converted in
          not (in_range && Stdlib.( = ) back const)
        | exception Converter.Not_in_the_compiler _ -> in_range)
      cases
  in
  let ones = -1L and top = Int64.min_int in
  let module Float32 = Numeric_types.Float32_by_bit_pattern in
  let module Float = Numeric_types.Float_by_bit_pattern in
  let module Vec128 = Vector_types.Vec128.Bit_pattern in
  let module Vec256 = Vector_types.Vec256.Bit_pattern in
  let module Vec512 = Vector_types.Vec512.Bit_pattern in
  let module Mask = Vector_types.Mask.Bit_pattern in
  let module RWC = Reg_width_const in
  (* A vector whose most significant word is [high], the others [low]. *)
  let vec128 ~low ~high =
    RWC.naked_vec128 (Vec128.of_bits { word0 = low; word1 = high })
  in
  let vec256 ~low ~high =
    RWC.naked_vec256
      (Vec256.of_bits { word0 = low; word1 = low; word2 = low; word3 = high })
  in
  let vec512 ~low ~high =
    RWC.naked_vec512
      (Vec512.of_bits
         { word0 = low;
           word1 = low;
           word2 = low;
           word3 = low;
           word4 = low;
           word5 = low;
           word6 = low;
           word7 = high
         })
  in
  let from_the_compiler : (string * RWC.t) list =
    [ ( "the float32 that is minus zero",
        RWC.naked_float32 (Float32.of_bits Int32.min_int) );
      "a float32 of ones", RWC.naked_float32 (Float32.of_bits (-1l));
      "the float that is minus zero", RWC.naked_float (Float.of_bits top);
      "a float of ones", RWC.naked_float (Float.of_bits ones);
      ( "a mask with its top bit only",
        RWC.naked_mask (Mask.of_bits { word0 = top }) );
      "a mask of ones", RWC.naked_mask (Mask.of_bits { word0 = ones });
      "a vec128 with its top bit only", vec128 ~low:0L ~high:top;
      "a vec128 of ones", vec128 ~low:ones ~high:ones;
      "a vec256 with its top bit only", vec256 ~low:0L ~high:top;
      "a vec256 of ones", vec256 ~low:ones ~high:ones;
      "a vec512 with its top bit only", vec512 ~low:0L ~high:top;
      "a vec512 of ones", vec512 ~low:ones ~high:ones ]
  in
  let wrong_from_the_compiler =
    List.filter
      (fun ((_ : string), const) ->
        let converted = Converter.const_to_model const in
        not
          (Primitive_spec.const_in_range converted
          &&
          match Converter.const_of_model machine_width converted with
          | back -> RWC.equal back const
          | exception Converter.Not_in_the_compiler _ -> false))
      from_the_compiler
  in
  Format.printf "@.The ranges of numbers (%d constants): %d as expected@."
    (List.length cases)
    (List.length cases - List.length wrong);
  List.iter
    (fun (const, in_range) ->
      Format.printf "  WRONG: %a is %s@." Model_type.print_const const
        (if in_range
         then
           "refused, changed or not in the range of the model, and is in range"
         else "accepted, or in the range of the model, and is out of range"))
    wrong;
  Format.printf
    "Constants of the compiler with the top bit set (%d): %d in the range of \
     the model and unchanged by the round trip@."
    (List.length from_the_compiler)
    (List.length from_the_compiler - List.length wrong_from_the_compiler);
  List.iter
    (fun (what, _) -> Format.printf "  WRONG: %s@." what)
    wrong_from_the_compiler;
  failures
    := !failures + List.length wrong + List.length wrong_from_the_compiler

(* ======================================================================== *)
(* The types of the symbols of a group of static constants                  *)
(* ======================================================================== *)

(* The environment that the model has in the body of a [Let_static]
   (FlambdaSimplifyStaticsSpec.v, [add_static_bindings], extracted) beside the
   one that Simplify leaves ([Simplify_static_const.simplify_static_consts]), on
   generated groups of one to three constants that are not closures: blocks of
   values, boxed numbers and strings. The environment before the group has the
   names of the tests and equations that make variables of kind value aliases.
   Simplify wants the environment as the library exports it, of which the tests
   can make no type but through the functions exported: that is why the
   equations are aliases only, which is all that the model's type of a constant
   reads of a variable. It is made by the exported functions, and read through
   the observer.

   The model gives a constant no type where it says that Simplify stops with a
   fatal error: a field of a block that is not of the kind value, a symbol that
   is not defined when the constant is reached (T3-11). [static_const_type_in],
   which is extracted too, says of each constant whether it has a type in the
   environment that the constants before it make. Simplify is run on each prefix
   of the group, which tells at which constant it stops. What is seen of the two
   environments after the constants before that one, or after the group, is
   compared in every case, whatever the constant at which Simplify stops: the
   type and the canonical element of every name of the tests and of every symbol
   that a group may bind.

   Two things are known to differ, both of a boxed number whose contents are a
   variable (T3-10). Where the variable holds the naked number, the model gives
   the type of the box and Simplify stops: that is recorded only if what is seen
   before that constant passes the comparison. Where it is of the kind value,
   the model gives no type and Simplify goes on: it gives the symbol the type
   that the environment has for the variable, so that what is seen is the same
   if nothing is known of the variable, and differs if the variable is an alias,
   which the symbol then is too. The comparison goes past such a constant, and a
   difference in what is seen is recorded, and is not a disagreement, only if it
   goes once the model is made to give the symbol of each such constant the type
   that its environment has for the variable.

   Not generated: closures and code (Simplify would simplify the code); the
   constants that the model has not (mutable blocks, blocks of a mixed shape,
   arrays of naked numbers or of floats); an immutable array of values, empty or
   not, which the model has and to which it gives no type, and which is not
   converted here ([static_const_of_model] refuses it); a tag that is not one of
   a block that is scanned (the compiler cannot build the constant); a variable
   that the environment does not define. Simplify chooses its meet by the flag
   of the join algorithm: it is run with the n-way one, which the model
   transcribes, and with the binary one, the default, for a comparison of the
   two. *)
module Model_statics = Flambda2_rocq_semantics.FlambdaSimplifyStaticsSpec
module DE = Flambda2_simplify.Downwards_env
module DA = Flambda2_simplify.Downwards_acc
module Static_const = Flambda2_terms.Static_const
module Static_const_group = Flambda2_terms.Flambda.Static_const_group
module Static_const_or_code = Flambda2_terms.Flambda.Static_const_or_code

let with_join_algorithm (algorithm : Oxcaml_flags.join_algorithm)
    (f : unit -> 'a) : 'a =
  let before = !Oxcaml_flags.Flambda2.join_algorithm in
  Oxcaml_flags.Flambda2.join_algorithm := Oxcaml_flags.Set algorithm;
  Fun.protect f ~finally:(fun () ->
      Oxcaml_flags.Flambda2.join_algorithm := before)

type static_group =
  { (* The aliases of the environment before the group: a variable of kind
       value, and what it is an alias of. *)
    before : (M.variable * M.simple) list;
    bindings : (M.symbol * M.static_const) list
  }

(* Three symbols that the environment of the tests does not define, for the
   groups to bind. They are numbered after the symbols of the tests. *)
let symbols_for_static_groups (names : names) : M.symbol list =
  let unit = Current_unit.get_cu_exn () in
  List.map
    (fun name ->
      names.numbering.symbols.number
        (Symbol.create unit (Linkage_name.of_string name)))
    ["a"; "b"; "c"]

let same_symbol (symbol : M.symbol) (symbol' : M.symbol) : bool =
  N.compare_positive symbol symbol' = 0

(* A variable of kind value may be an alias of a variable of that kind numbered
   after it, of a symbol of the tests or of a tagged constant.

   A group binds one to three of the symbols, in some order. A field of a block
   is mostly a variable of kind value, a tagged constant, a symbol of the tests
   or one of the three symbols, whether the group has bound it by then, binds it
   later or does not bind it; now and then it is a naked immediate. The contents
   of a boxed number are a constant, the variable that holds a naked float, or a
   variable of kind value. *)
let generate_static_group (names : names) (symbols : M.symbol list) r :
    static_group =
  let pool = names.pool in
  let before =
    List.filter_map
      (fun x : (M.variable * M.simple) option ->
        let later =
          List.filter (fun y -> N.compare_positive x y < 0) pool.value_variables
        in
        match G.below r 6, later with
        | (0 | 1 | 2), _ -> None
        | 3, _ :: _ -> Some (x, Var (G.pick r later))
        | (3 | 4), _ -> Some (x, Symbol (G.pick r pool.symbols))
        | _, _ -> Some (x, Const (Tagged_immediate (G.small_number r))))
      pool.value_variables
  in
  let rec in_some_order symbols =
    match symbols with
    | [] -> []
    | _ :: _ ->
      let symbol = G.pick r symbols in
      symbol
      :: in_some_order
           (List.filter (fun other -> not (same_symbol other symbol)) symbols)
  in
  let count = 1 + G.below r 3 in
  let bound = List.filteri (fun i _ -> i < count) (in_some_order symbols) in
  let field () : M.simple =
    match G.below r 20 with
    | 0 | 1 | 2 | 3 | 4 | 5 -> Var (G.pick r pool.value_variables)
    | 6 | 7 | 8 | 9 -> Const (Tagged_immediate (G.small_number r))
    | 10 | 11 | 12 -> Symbol (G.pick r pool.symbols)
    | 13 | 14 | 15 | 16 | 17 -> Symbol (G.pick r symbols)
    | 18 -> Var (G.pick r pool.naked_immediate_variables)
    | _ -> Const (Naked_immediate (G.small_number r))
  in
  let contents () : M.simple =
    match G.below r 6 with
    | 0 ->
      Const
        (Naked_float
           (N.z_of_int64 (Int64.bits_of_float (Float.of_int (G.below r 5)))))
    | 1 -> Const (Naked_int32 (G.small_number r))
    | 2 -> Const (Naked_int64 (G.small_number r))
    | 3 -> Const (Naked_nativeint (G.small_number r))
    | 4 -> Var (G.pick r pool.naked_float_variables)
    | _ -> Var (G.pick r pool.value_variables)
  in
  let const () : M.static_const =
    match G.below r 10 with
    | 0 | 1 | 2 -> Static_const_boxed_number (contents ())
    | 3 ->
      Static_const_immutable_string
        (G.list r ~max:3 (fun () -> N.z_of_int (97 + G.below r 3)))
    | _ ->
      Static_const_block
        (N.z_of_int (G.pick r [0; 0; 1; 2; 250]), G.list r ~max:3 field)
  in
  { before; bindings = List.map (fun symbol -> symbol, const ()) bound }

let print_static_const ppf (const : M.static_const) =
  let simples =
    Format.pp_print_list
      ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
      Model_type.print_simple
  in
  match const with
  | Static_const_block (tag, fields) ->
    Format.fprintf ppf "@[<hov 2>the block of tag %a of (%a)@]" N.print_z tag
      simples fields
  | Static_const_boxed_number contents ->
    Format.fprintf ppf "the boxed number of %a" Model_type.print_simple contents
  | Static_const_immutable_string bytes ->
    Format.fprintf ppf "@[<hov 2>the string of the bytes (%a)@]"
      (Format.pp_print_list
         ~pp_sep:(fun ppf () -> Format.fprintf ppf ",@ ")
         N.print_z)
      bytes
  | Static_const_immutable_value_array elements ->
    Format.fprintf ppf "@[<hov 2>the immutable array of (%a)@]" simples elements
  | Static_const_set_of_closures _ | Static_const_closure_of_set _ ->
    Format.pp_print_string ppf "a closure"

let print_static_group ppf { before; bindings } =
  Format.fprintf ppf "@[<v>%a%a@]"
    (fun ppf ->
      List.iter (fun (x, simple) ->
          Format.fprintf ppf "with %a an alias of %a@ "
            Model_type.print_variable x Model_type.print_simple simple))
    before
    (Format.pp_print_list ~pp_sep:Format.pp_print_cut
       (fun ppf (symbol, const) ->
         Format.fprintf ppf "%a = %a" Model_type.print_symbol symbol
           print_static_const const))
    bindings

(* Fewer aliases, fewer constants, fewer fields. *)
let shrink_static_group ({ before; bindings } as group) =
  let without i items = List.filteri (fun j _ -> i <> j) items in
  List.mapi (fun i _ -> { group with before = without i before }) before
  @ (match bindings with
    | [] | [_] -> []
    | _ :: _ :: _ ->
      List.mapi
        (fun i _ -> { group with bindings = without i bindings })
        bindings)
  @ List.concat
      (List.mapi
         (fun i ((symbol, const) : M.symbol * M.static_const) ->
           match const with
           | Static_const_block (tag, fields) ->
             List.mapi
               (fun j _ ->
                 { group with
                   bindings =
                     List.mapi
                       (fun i' binding ->
                         if i = i'
                         then
                           symbol, M.Static_const_block (tag, without j fields)
                         else binding)
                       bindings
                 })
               fields
           | Static_const_boxed_number _ | Static_const_immutable_string _
           | Static_const_immutable_value_array _
           | Static_const_set_of_closures _ | Static_const_closure_of_set _ ->
             [])
         bindings)

(* The constant of the compiler. The model's boxed number does not say which
   number it boxes where its contents are a variable: it is a boxed float
   here. *)
let static_const_of_model (names : names) (const : M.static_const) :
    Static_const.t =
  let not_in_the_compiler what = raise (Converter.Not_in_the_compiler what) in
  match const with
  | Static_const_block (tag, fields) ->
    let tag =
      match Option.bind (N.int_of_z tag) Tag.Scannable.create with
      | Some tag -> tag
      | None ->
        not_in_the_compiler "a tag that is not one of a block that is scanned"
    in
    Static_const.block tag Immutable Value_only
      (List.map
         (fun field ->
           Simple.With_debuginfo.create
             (Converter.simple_of_model names.numbering machine_width field)
             Debuginfo.none)
         fields)
  | Static_const_boxed_number (Const const) -> (
    match
      Reg_width_const.descr (Converter.const_of_model machine_width const)
    with
    | Naked_float32 f -> Static_const.boxed_float32 (Const f)
    | Naked_float f -> Static_const.boxed_float (Const f)
    | Naked_int32 i -> Static_const.boxed_int32 (Const i)
    | Naked_int64 i -> Static_const.boxed_int64 (Const i)
    | Naked_nativeint i -> Static_const.boxed_nativeint (Const i)
    | Naked_vec128 v -> Static_const.boxed_vec128 (Const v)
    | Naked_vec256 v -> Static_const.boxed_vec256 (Const v)
    | Naked_vec512 v -> Static_const.boxed_vec512 (Const v)
    | Naked_mask m -> Static_const.boxed_mask (Const m)
    | Naked_immediate _ | Tagged_immediate _ | Naked_int8 _ | Naked_int16 _
    | Null | Poison _ ->
      not_in_the_compiler "the box of a constant that has none")
  | Static_const_boxed_number (Var x) ->
    Static_const.boxed_float
      (Var (Converter.variable_of_model names.numbering x, Debuginfo.none))
  | Static_const_boxed_number (Symbol _) ->
    not_in_the_compiler "a boxed number whose contents are a symbol"
  | Static_const_immutable_string bytes ->
    Static_const.immutable_string
      (String.concat ""
         (List.map
            (fun byte ->
              match N.int_of_z byte with
              | Some byte when 0 <= byte && byte < 256 ->
                String.make 1 (Char.chr byte)
              | Some _ | None -> not_in_the_compiler "a byte out of range")
            bytes))
  | Static_const_immutable_value_array _ ->
    not_in_the_compiler "an immutable array, which is not generated"
  | Static_const_set_of_closures _ | Static_const_closure_of_set _ ->
    not_in_the_compiler "a static closure, which is not generated"

(* The environment of the tests with the aliases, made by the functions that the
   library exports: the equations are added with the meet that the flag of the
   join algorithm chooses. *)
let env_before_a_static_group (names : names)
    (before : (M.variable * M.simple) list) : Flambda2_types.Typing_env.t =
  let module Exported = Flambda2_types.Typing_env in
  let env =
    List.fold_left
      (fun env (name, kind) ->
        Exported.add_definition env
          (Name.pattern_match name
             ~var:(fun var ->
               Bound_name.create_var
                 (Bound_var.create var Flambda_debug_uid.none Name_mode.normal))
             ~symbol:Bound_name.create_symbol)
          kind)
      (Exported.create ~machine_width ~resolver:(fun _ -> None))
      names.defined
  in
  List.fold_left
    (fun env (x, simple) ->
      Exported.add_equation env
        (Name.var (Converter.variable_of_model names.numbering x))
        (Flambda2_types.alias_type_of K.value
           (Converter.simple_of_model names.numbering machine_width simple)))
    env before

(* The environment that Simplify leaves after the constants, from the typing
   environment given, as the modules of the library have it. The rest of
   Simplify's environment is that of the top level of a unit; no code is defined
   and none is simplified. *)
let static_group_in_the_compiler (names : names)
    (env : Flambda2_types.Typing_env.t)
    (bindings : (M.symbol * M.static_const) list) : TE.t =
  let denv =
    DE.create ~round:0 ~machine_width
      ~resolver:(fun _ -> None)
      ~get_imported_code:(fun () -> Flambda2_cmx.Exported_code.empty)
      ~propagating_float_consts:false
      ~unit_toplevel_exn_continuation:(Continuation.create ())
      ~unit_toplevel_return_continuation:
        (Continuation.create ~sort:Toplevel_return ())
      ~toplevel_my_alloc_region:(Variable.create "region" K.region)
  in
  let dacc =
    DA.create
      (DE.with_typing_env denv env)
      Code_id.Map.empty Flambda2_simplify.Continuation_uses_env.empty
  in
  let bound =
    Bound_static.create
      (List.map
         (fun ((symbol, _) : M.symbol * M.static_const) ->
           Bound_static.Pattern.block_like
             (Converter.symbol_of_model names.numbering symbol))
         bindings)
  in
  let group =
    Static_const_group.create
      (List.map
         (fun ((_, const) : M.symbol * M.static_const) ->
           Static_const_or_code.create_static_const
             (static_const_of_model names const))
         bindings)
  in
  let ( (_ : Bound_static.t),
        (_ : Flambda2_simplify.Rebuilt_static_const.Group.t),
        dacc ) =
    Flambda2_simplify.Simplify_static_const.simplify_static_consts dacc bound
      group
      ~simplify_function_body:(fun
          _
          _
          ~return_continuation:_
          ~return_arity:_
          ~exn_continuation:_
          ~loopify_state:_
          ~params:_
          ~implicit_params:_
        -> Misc.fatal_error "No code is to be simplified here")
  in
  let observed = ref None in
  TE.For_testing.set_observer
    (Some (fun env (_ : Continuation.t) -> observed := Some env));
  Flambda2_types.Typing_env.notify_test_observer (DA.typing_env dacc)
    ~at_handler:(Continuation.create ());
  TE.For_testing.set_observer None;
  match !observed with
  | Some env -> env
  | None -> Misc.fatal_error "The observer was not called"

(* Why the model gives a block no type, by its text and the symbols that are
   defined when it is reached. *)
type untyped =
  { (* A field is a symbol that is not defined then. *)
    names_a_symbol_not_defined : bool;
    (* A field is a naked immediate. *)
    has_a_field_not_of_kind_value : bool
  }

let why_untyped (names : names) ~(defined : M.symbol list)
    (const : M.static_const) : untyped =
  let is_defined symbol = List.exists (same_symbol symbol) defined in
  match const with
  | Static_const_block ((_ : N.z), fields) ->
    { names_a_symbol_not_defined =
        List.exists
          (fun (field : M.simple) ->
            match field with
            | Symbol symbol -> not (is_defined symbol)
            | Var _ | Const _ -> false)
          fields;
      has_a_field_not_of_kind_value =
        List.exists
          (fun (field : M.simple) ->
            match field with
            | Var x ->
              List.exists
                (fun y -> N.compare_positive x y = 0)
                names.pool.naked_immediate_variables
            | Const (Naked_immediate _) -> true
            | Const
                ( Tagged_immediate _ | Naked_float32 _ | Naked_float _
                | Naked_int8 _ | Naked_int16 _ | Naked_int32 _ | Naked_int64 _
                | Naked_nativeint _ | Naked_vec128 _ | Naked_vec256 _
                | Naked_vec512 _ | Naked_mask _ | Null | Poison _ )
            | Symbol _ ->
              false)
          fields
    }
  | Static_const_boxed_number _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ | Static_const_set_of_closures _
  | Static_const_closure_of_set _ ->
    { names_a_symbol_not_defined = false;
      has_a_field_not_of_kind_value = false
    }

(* What the two say of a group. *)
type static_group_verdict =
  | Neither_starts of string
  | One_does_not_start of string
  | Compared of
      { (* The position of the constant at which Simplify stops, if it does,
           what it says, and whether the model gives that constant a type. *)
        stops : (int * string * bool) option;
        (* The positions of the constants before that one, or of all, to which
           the model gives no type though Simplify goes on. *)
        passed_untyped : int list;
        (* What is seen after the constants before the one at which Simplify
           stops, or after the group. *)
        in_the_model : seen;
        in_the_compiler : seen;
        (* The same of the model made to do as Simplify does of a boxed number
           whose contents are a variable of kind value. *)
        in_the_model_as_simplify : seen
      }

(* A fatal error of Simplify without the escape sequences that colour it. *)
let without_colours (line : string) : string =
  let plain = Buffer.create (String.length line) in
  let rec go i ~in_sequence =
    if i < String.length line
    then
      let c = line.[i] in
      if in_sequence
      then go (i + 1) ~in_sequence:(not (Char.equal c 'm'))
      else if Char.equal c '\027'
      then go (i + 1) ~in_sequence:true
      else (
        Buffer.add_char plain c;
        go (i + 1) ~in_sequence:false)
  in
  go 0 ~in_sequence:false;
  Buffer.contents plain

(* What the comparisons of static groups share: the model run, its environment
   of the tests, and the names that are asked about, which are those of the
   tests and the symbols that a group may bind. *)
type setting_of_static_groups =
  { model : model;
    model_env : Model_env.t;
    threshold : int;
    asked : MT.name list
  }

let setting_of_static_groups (names : names) (symbols : M.symbol list) :
    setting_of_static_groups =
  let model_env, defined = env_of_model names in
  { model = the_model Of_the_compiler;
    model_env;
    (* The variables of the environment are the first of the numbering. *)
    threshold =
      List.length
        (List.filter
           (fun (name : MT.name) ->
             match name with Name_var _ -> true | Name_symbol _ -> false)
           defined);
    asked =
      defined @ List.map (fun symbol : MT.name -> Name_symbol symbol) symbols
  }

let first n items = List.filteri (fun i _ -> i < n) items

(* What the model says of a group. *)
type static_group_in_the_model =
  { (* Whether it gives the constant of a position a type, in the environment
       that it has after those before. *)
    typed : int -> bool;
    (* What is seen after the first constants. *)
    seen_after : int -> seen;
    (* The same where, of a boxed number whose contents are a variable of kind
       value, the model is made to do what Simplify is seen to do in place of
       what it does itself: the symbol is defined and given, by an equation, the
       type that the environment has for the variable (T3-10). The other
       constants are added one by one by the model's function. *)
    seen_after_as_simplify : int -> seen
  }

let static_group_in_the_model (names : names)
    ({ model; model_env; threshold; asked } : setting_of_static_groups)
    { before; bindings } : (static_group_in_the_model, string) result =
  let rec add env aliases =
    match aliases with
    | [] -> Some env
    | (x, simple) :: aliases ->
      Option.bind
        (model.add_equation_of_model env (Name_var x)
           (MT.alias_type_of Value simple))
        (fun env -> add env aliases)
  in
  match add model_env before with
  | None -> Error "no answer where an alias is added"
  | Some env ->
    let binding ((symbol, const) : M.symbol * M.static_const) : M.static_binding
        =
      Bind_symbol (symbol, const)
    in
    let after n =
      Model_statics.add_static_bindings env
        (List.map binding (first n bindings))
    in
    let after_as_simplify n : Model_env.t option =
      List.fold_left
        (fun env ((symbol, const) as bound : M.symbol * M.static_const) ->
          Option.bind env (fun env ->
              match const with
              | Static_const_boxed_number (Var x)
                when List.exists
                       (fun y -> N.compare_positive x y = 0)
                       names.pool.value_variables ->
                Option.bind (Model_env.find env (Name_var x) (Some Value))
                  (fun t ->
                    model.add_equation_of_model
                      (Model_env.add_symbol_definition env symbol)
                      (Name_symbol symbol) t)
              | Static_const_boxed_number (Var _ | Const _ | Symbol _)
              | Static_const_block _ | Static_const_immutable_string _
              | Static_const_immutable_value_array _
              | Static_const_set_of_closures _ | Static_const_closure_of_set _
                ->
                Some (Model_statics.add_static_bindings env [binding bound])))
        (Some env) (first n bindings)
    in
    Ok
      { typed =
          (fun i ->
            let (_ : M.symbol), const = List.nth bindings i in
            let its_type = Model_statics.static_const_type_in (after i) const in
            Option.is_some its_type);
        seen_after =
          (fun n -> seen_in_the_model ~threshold asked None (after n));
        seen_after_as_simplify =
          (fun n ->
            match after_as_simplify n with
            | Some env -> seen_in_the_model ~threshold asked None env
            | None -> Nothing_seen "no answer where the model does as Simplify")
      }

(* At which constant Simplify stops, and with what, and what is seen after the
   first constants. *)
let static_group_in_simplify (names : names)
    ({ threshold; asked; model = _; model_env = _ } : setting_of_static_groups)
    (algorithm : Oxcaml_flags.join_algorithm) { before; bindings } :
    ((int * string) option * (int -> seen), string) result =
  with_join_algorithm algorithm (fun () ->
      match
        or_fatal_error (fun () -> env_before_a_static_group names before)
      with
      | Error line ->
        Error
          ("a fatal error where an alias is added: "
          ^ general (without_colours line))
      | Ok env -> (
        let after =
          List.init
            (List.length bindings + 1)
            (fun n ->
              or_fatal_error (fun () ->
                  static_group_in_the_compiler names env (first n bindings)))
        in
        match after with
        | [] | Error _ :: _ -> Error "a fatal error on a group of no constant"
        | Ok _ :: _ ->
          Ok
            ( List.find_mapi
                (fun n (after : (TE.t, string) result) ->
                  match after with
                  | Ok _ -> None
                  | Error line -> Some (n - 1, general (without_colours line)))
                after,
              fun n ->
                match List.nth after n with
                | Ok env ->
                  seen_in_the_compiler names.numbering ~threshold asked None env
                | Error line ->
                  Nothing_seen ("a fatal error: " ^ without_colours line) )))

let verdict_of_a_static_group (names : names)
    (setting : setting_of_static_groups) (input : static_group) :
    static_group_verdict =
  match
    ( static_group_in_the_model names setting input,
      static_group_in_simplify names setting N_way input )
  with
  | Error why, Error why' ->
    Neither_starts (Printf.sprintf "the model: %s; the compiler: %s" why why')
  | Error why, Ok _ -> One_does_not_start ("the model: " ^ why)
  | Ok _, Error why -> One_does_not_start ("the compiler: " ^ why)
  | Ok { typed; seen_after; seen_after_as_simplify }, Ok (stops, of_compiler) ->
    let compared =
      match stops with
      | Some (j, (_ : string)) -> j
      | None -> List.length input.bindings
    in
    Compared
      { stops = Option.map (fun (j, line) -> j, line, typed j) stops;
        passed_untyped =
          List.filter (fun i -> not (typed i)) (List.init compared Fun.id);
        in_the_model = seen_after compared;
        in_the_compiler = of_compiler compared;
        in_the_model_as_simplify = seen_after_as_simplify compared
      }

let print_both_seen ~threshold ?(one = "the model") ?(other = "the compiler")
    seen seen' =
  Format.asprintf "@[<v>%s:@ %a@ %s:@ %a@]" one print_lines
    (print_seen ~threshold seen)
    other print_lines
    (print_seen ~threshold seen')

(* A boxed number whose contents are a variable that holds a naked number. *)
let boxes_a_naked_number (names : names) (const : M.static_const) =
  match const with
  | Static_const_boxed_number (Var x) ->
    List.exists
      (fun y -> N.compare_positive x y = 0)
      names.pool.naked_float_variables
  | Static_const_boxed_number (Const _ | Symbol _)
  | Static_const_block _ | Static_const_immutable_string _
  | Static_const_immutable_value_array _ | Static_const_set_of_closures _
  | Static_const_closure_of_set _ ->
    false

(* What a verdict comes to. What is seen after the constants before the one at
   which Simplify stops, or after the group, is compared first, whatever comes
   after: what the model itself sees and what the compiler sees are to be read;
   then they are to be the same, or the same once the model is made to do as
   Simplify of the boxed numbers of variables of kind value that were gone past,
   which is recorded (T3-10); anything else is a disagreement. Only then is the
   constant at which Simplify stops looked at: if the model gives it a type,
   that is recorded where it is a boxed number whose contents are a variable
   that holds a naked number (T3-10), and is a disagreement otherwise. *)
let outcome_of_a_static_group ?(brief = false) (names : names) ~threshold
    (input : static_group) (verdict : static_group_verdict) : outcome =
  match verdict with
  | Neither_starts why -> Not_run ("neither starts (" ^ why ^ ")")
  | One_does_not_start why ->
    Disagree ("one of the two does not start (" ^ why ^ ")")
  | Compared
      { stops;
        passed_untyped;
        in_the_model;
        in_the_compiler;
        in_the_model_as_simplify
      } ->
    let is_read (seen : seen) =
      match seen with Seen _ -> true | Nothing_seen _ | Bottom_seen -> false
    in
    let same_as_the_compiler seen =
      is_read seen && same_seen ~threshold seen in_the_compiler
    in
    let passed = match passed_untyped with [] -> false | _ :: _ -> true in
    let once =
      "what is seen is the same once the model gives the symbol of a boxed \
       number whose contents are a variable of kind value the type that its \
       environment has for the variable"
    in
    let stops_where_the_model_types j line ~before : outcome =
      let what =
        Printf.sprintf
          "Simplify stops at the constant of position %d (%s), which the model \
           gives a type%s"
          (j + 1) line before
      in
      if boxes_a_naked_number names (snd (List.nth input.bindings j))
      then
        Known_to_disagree
          ( "T3-10, Simplify stops at a boxed number whose contents are a \
             variable that holds the naked number, and the model gives it the \
             type of the box",
            what )
      else Disagree what
    in
    (* What the model itself sees and what the compiler sees are to be read,
       whatever else is. *)
    if not (is_read in_the_model && is_read in_the_compiler)
    then
      let reading (seen : seen) =
        match seen with
        | Seen _ -> "read"
        | Nothing_seen why -> "no answer: " ^ why
        | Bottom_seen -> "bottom"
      in
      Disagree
        (Printf.sprintf
           "what is seen is not read on both sides (the model: %s; the \
            compiler: %s)"
           (reading in_the_model) (reading in_the_compiler))
    else if same_as_the_compiler in_the_model
    then
      match stops with
      | None | Some ((_ : int), (_ : string), false) -> Agree
      | Some (j, line, true) -> stops_where_the_model_types j line ~before:""
    else if passed && same_as_the_compiler in_the_model_as_simplify
    then
      match stops with
      | None | Some ((_ : int), (_ : string), false) ->
        Known_to_disagree
          ( "T3-10, the model gives no type to a boxed number whose contents \
             are a variable of kind value, and Simplify gives its symbol the \
             type that the environment has for the variable",
            if brief
            then once
            else
              Format.asprintf "@[<v>%s; as they are:@ %a@]" once print_lines
                (print_both_seen ~threshold in_the_model in_the_compiler) )
      | Some (j, line, true) ->
        stops_where_the_model_types j line
          ~before:("; before that constant, " ^ once)
    else Disagree (print_both_seen ~threshold in_the_model in_the_compiler)

let static_groups (names : names) (symbols : M.symbol list) :
    static_group comparison list =
  let setting = setting_of_static_groups names symbols in
  let threshold = setting.threshold in
  let in_the_compiler = static_group_in_simplify names setting in
  let verdict = verdict_of_a_static_group names setting in
  let outcome ?brief input verdict =
    outcome_of_a_static_group ?brief names ~threshold input verdict
  in
  let defined_before (input : static_group) i : M.symbol list =
    names.pool.symbols @ List.map fst (first i input.bindings)
  in
  let why_untyped_at (input : static_group) i : untyped =
    why_untyped names ~defined:(defined_before input i)
      (snd (List.nth input.bindings i))
  in
  (* The groups in which Simplify goes past no constant to which the model gives
     no type. *)
  let nothing_passed (input : static_group) :
      ((int * string * bool) option * static_group_verdict) option =
    match verdict input with
    | Compared { stops; passed_untyped = []; _ } as verdict ->
      Some (stops, verdict)
    | Compared { passed_untyped = _ :: _; _ }
    | Neither_starts _ | One_does_not_start _ ->
      None
  in
  (* Of those, the groups in which Simplify stops at a constant to which the
     model gives no type. The reason for which it gives none is told by the text
     of the constant and by the symbols that the tests define and that the
     constants before it bind. *)
  let where_both_part ~(select : untyped -> bool) (input : static_group) :
      outcome =
    match nothing_passed input with
    | Some (Some (j, (_ : string), false), verdict)
      when select (why_untyped_at input j) ->
      outcome input verdict
    | Some ((Some _ | None), (_ : static_group_verdict)) | None ->
      Not_this_comparison
  in
  [ { name = "static constants: the model and Simplify";
      compare = (fun input -> outcome input (verdict input))
    };
    { name =
        "of which those where the model gives no type to a constant that \
         Simplify goes past: what is seen after the group, or after the \
         constants before the one at which Simplify stops";
      compare =
        (fun input ->
          match verdict input with
          | Compared
              { stops = Some ((_ : int), (_ : string), false) | None;
                passed_untyped = _ :: _;
                _
              } as verdict ->
            outcome ~brief:true input verdict
          | Compared _ | Neither_starts _ | One_does_not_start _ ->
            Not_this_comparison)
    };
    { name =
        "of which the others where Simplify stops at no constant: what is seen \
         after the group";
      compare =
        (fun input ->
          match nothing_passed input with
          | Some (None, verdict) -> outcome input verdict
          | Some (Some _, (_ : static_group_verdict)) | None ->
            Not_this_comparison)
    };
    { name = "of which those where the group has two or three constants";
      compare =
        (fun input ->
          match nothing_passed input, input.bindings with
          | Some (None, verdict), _ :: _ :: _ -> outcome input verdict
          | Some (None, (_ : static_group_verdict)), ([] | [_])
          | (Some (Some _, (_ : static_group_verdict)) | None), _ ->
            Not_this_comparison)
    };
    { name =
        "of which the others where Simplify stops at a constant to which the \
         model gives no type: what is seen after the constants before it";
      compare = where_both_part ~select:(fun (_ : untyped) -> true)
    };
    { name =
        "of which those where that constant names a symbol that is not defined \
         by then, and all its fields are of kind value (T3-11)";
      compare =
        where_both_part ~select:(fun why ->
            why.names_a_symbol_not_defined
            && not why.has_a_field_not_of_kind_value)
    };
    { name =
        "of which those where that constant has a field that is not of kind \
         value, and names no symbol that is not defined by then";
      compare =
        where_both_part ~select:(fun why ->
            why.has_a_field_not_of_kind_value
            && not why.names_a_symbol_not_defined)
    };
    (* The compiler's default is the binary meet. Nothing here is the model's:
       the comparison says whether what the tests above say of Simplify with the
       n-way meet holds of Simplify as it is run by default. *)
    { name =
        "static constants: Simplify with the binary meet, the compiler's \
         default, and with the n-way meet";
      compare =
        (fun input ->
          match in_the_compiler N_way input, in_the_compiler Binary input with
          | Error why, Error why' ->
            Not_run
              (Printf.sprintf "neither starts (n-way: %s; binary: %s)" why why')
          | Error why, Ok _ -> Disagree ("the n-way does not start: " ^ why)
          | Ok _, Error why -> Disagree ("the binary does not start: " ^ why)
          | Ok (stops, seen), Ok (stops', seen') -> (
            let same n =
              if same_seen ~threshold (seen n) (seen' n)
              then Agree
              else
                Disagree
                  (print_both_seen ~threshold ~one:"n-way" ~other:"binary"
                     (seen n) (seen' n))
            in
            match stops, stops' with
            | None, None -> same (List.length input.bindings)
            | Some (j, (_ : string)), Some (j', (_ : string)) when j = j' ->
              same j
            | (None | Some _), (None | Some _) ->
              Disagree "the two do not stop at the same constant"))
    } ]

(* A control of the comparison of static groups, on groups written by hand that
   bind the first of the symbols: what is seen tells apart two groups that
   differ in one thing, and the model and Simplify see the same of one group;
   and the two stop where the comments above say. *)
let control_of_the_static_groups (names : names) (symbols : M.symbol list) =
  let setting = setting_of_static_groups names symbols in
  let threshold = setting.threshold in
  let symbol, other_symbol =
    match symbols with
    | symbol :: other_symbol :: _ -> symbol, other_symbol
    | [] | [_] -> Misc.fatal_error "Two symbols are wanted"
  in
  let x, y =
    match names.pool.value_variables with
    | x :: y :: _ -> x, y
    | [] | [_] -> Misc.fatal_error "Two variables are wanted"
  in
  let naked_immediate = List.hd names.pool.naked_immediate_variables in
  let z = N.z_of_int in
  let block ?(tag = 0) fields : M.static_const =
    Static_const_block (z tag, fields)
  in
  let group ?(before = []) ?(binds = symbol) const : static_group =
    { before; bindings = [binds, const] }
  in
  (* What is seen after the group, where every constant is passed. *)
  let after sides (group : static_group) : seen =
    match sides group with
    | Ok ((_ : _ option), seen) -> seen (List.length group.bindings)
    | Error why -> Nothing_seen why
  in
  let in_the_model group =
    after
      (fun group ->
        Result.map
          (fun ({ seen_after; _ } : static_group_in_the_model) ->
            None, seen_after)
          (static_group_in_the_model names setting group))
      group
  in
  let in_simplify group =
    after
      (fun group ->
        match static_group_in_simplify names setting N_way group with
        | Ok (None, seen) -> Ok (None, seen)
        | Ok (Some ((_ : int), line), (_ : int -> seen)) ->
          Error ("Simplify stops: " ^ line)
        | Error why -> Error why)
      group
  in
  let both_see (seen : seen) (seen' : seen) =
    match seen, seen' with
    | Seen _, Seen _ -> true
    | (Nothing_seen _ | Bottom_seen | Seen _), _ -> false
  in
  let same group group' =
    let seen = in_the_model group in
    let seen' = in_simplify group' in
    both_see seen seen' && same_seen ~threshold seen seen'
  in
  let told_apart group group' =
    let seen = in_the_model group in
    let seen' = in_simplify group' in
    both_see seen seen' && not (same_seen ~threshold seen seen')
  in
  let the_model_types (group : static_group) =
    match static_group_in_the_model names setting group with
    | Ok { typed; _ } -> Some (typed 0)
    | Error _ -> None
  in
  let simplify_stops (group : static_group) =
    match static_group_in_simplify names setting N_way group with
    | Ok (Some (0, (_ : string)), (_ : int -> seen)) -> true
    | Ok ((Some _ | None), (_ : int -> seen)) | Error _ -> false
  in
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: groups of one static constant, the model on one and Simplify \
     on one that differs from it in one thing@.";
  let pair = group (block [Var x; Var y]) in
  check "a block of two variables: the model and Simplify see the same"
    (same pair pair);
  check "and the block of the two variables exchanged: told apart"
    (told_apart pair (group (block [Var y; Var x])));
  check "and the block of another tag: told apart"
    (told_apart pair (group (block ~tag:1 [Var x; Var y])));
  check "and the block with a third field: told apart"
    (told_apart pair (group (block [Var x; Var y; Var x])));
  check "and the same block bound to another symbol: told apart"
    (told_apart pair (group ~binds:other_symbol (block [Var x; Var y])));
  check
    "and the same block where the first variable is an alias of the second: \
     told apart"
    (told_apart pair (group ~before:[x, Var y] (block [Var x; Var y])));
  let aliased = group ~before:[x, Var y] (block [Var x; Var y]) in
  check "with that alias on both sides: the model and Simplify see the same"
    (same aliased aliased);
  let boxed i : static_group =
    group (Static_const_boxed_number (Const (Naked_int64 (z i))))
  in
  check "a boxed number: the model and Simplify see the same"
    (same (boxed 1) (boxed 1));
  check "and the boxed number of another constant: told apart"
    (told_apart (boxed 1) (boxed 2));
  let text bytes : static_group =
    group (Static_const_immutable_string (List.map z bytes))
  in
  check "a string: the model and Simplify see the same"
    (same (text [97; 98]) (text [97; 98]));
  check "and the string of other bytes: told apart"
    (told_apart (text [97; 98]) (text [97; 99]));
  let names_itself = group (block [Symbol symbol]) in
  check "a block that names its own symbol: the model gives it no type"
    (Option.equal Bool.equal (the_model_types names_itself) (Some false));
  check "and Simplify stops at it" (simplify_stops names_itself);
  let not_a_value = group (block [Var naked_immediate]) in
  check
    "a block with a field that is a naked immediate: the model gives it no type"
    (Option.equal Bool.equal (the_model_types not_a_value) (Some false));
  check "and Simplify stops at it" (simplify_stops not_a_value);
  check "the model gives the block of two variables a type"
    (Option.equal Bool.equal (the_model_types pair) (Some true));
  check "and Simplify does not stop at it" (not (simplify_stops pair));
  let boxed_variable ?before variable : static_group =
    group ?before (Static_const_boxed_number (Var variable))
  in
  let unknown = boxed_variable x in
  check
    "a boxed number whose contents are a variable of kind value: the model \
     gives it no type"
    (Option.equal Bool.equal (the_model_types unknown) (Some false));
  check "and Simplify does not stop at it" (not (simplify_stops unknown));
  check "where nothing is known of the variable, the two see the same"
    (same unknown unknown);
  (* Of two variables that are aliases, the one defined second has the alias
     type. *)
  let alias = boxed_variable ~before:[x, Var y] y in
  check "where the variable has an alias type, the two are told apart"
    (told_apart alias alias);
  check
    "and the model made to give the symbol the type that its environment has \
     for the variable sees the same as Simplify"
    (let seen =
       match static_group_in_the_model names setting alias with
       | Ok { seen_after_as_simplify; _ } -> seen_after_as_simplify 1
       | Error why -> Nothing_seen why
     in
     let seen' = in_simplify alias in
     both_see seen seen' && same_seen ~threshold seen seen');
  (* What a group comes to where Simplify stops at a later constant to which the
     model gives a type: what is seen before that constant counts first. In some
     of the cases one side of the verdict is put there by hand: the compiler's
     from another group, or the model's taken away. *)
  let float_variable = List.hd names.pool.naked_float_variables in
  let then_a_boxed_float ?(before = []) const : static_group =
    { before;
      bindings =
        [ symbol, const;
          other_symbol, Static_const_boxed_number (Var float_variable) ]
    }
  in
  let verdict = verdict_of_a_static_group names setting in
  let outcome group verdict =
    outcome_of_a_static_group names ~threshold group verdict
  in
  let recorded_under_t3_10 (outcome : outcome) =
    match outcome with
    | Known_to_disagree (label, (_ : string)) ->
      String.starts_with ~prefix:"T3-10, Simplify stops" label
    | Agree | Disagree _ | Not_run _ | Not_this_comparison -> false
  in
  let is_a_disagreement (outcome : outcome) =
    match outcome with
    | Disagree _ -> true
    | Agree | Known_to_disagree _ | Not_run _ | Not_this_comparison -> false
  in
  (* The verdict of a group with, for what the compiler sees before the stop,
     what it sees of another group. *)
  let with_the_compiler_on (other : static_group) (group : static_group) :
      static_group_verdict =
    match verdict group, verdict other with
    | Compared compared, Compared { in_the_compiler; _ } ->
      Compared { compared with in_the_compiler }
    | ( ((Compared _ | Neither_starts _ | One_does_not_start _) as unchanged),
        (Compared _ | Neither_starts _ | One_does_not_start _) ) ->
      unchanged
  in
  let stopped = then_a_boxed_float (block [Var x; Var y]) in
  check
    "a block, then a boxed number of the variable that holds a naked float: \
     Simplify stops at the second, to which the model gives a type, and goes \
     past the first"
    (match verdict stopped with
    | Compared { stops = Some (1, (_ : string), true); passed_untyped = []; _ }
      ->
      true
    | Compared _ | Neither_starts _ | One_does_not_start _ -> false);
  check "what is seen before the stop is the same: recorded under T3-10"
    (recorded_under_t3_10 (outcome stopped (verdict stopped)));
  check
    "with, for the compiler, what it sees of the block of another tag: a \
     disagreement"
    (is_a_disagreement
       (outcome stopped
          (with_the_compiler_on
             (then_a_boxed_float (block ~tag:1 [Var x; Var y]))
             stopped)));
  check
    "with, for the compiler, what it sees of the block of the two variables \
     exchanged: a disagreement"
    (is_a_disagreement
       (outcome stopped
          (with_the_compiler_on
             (then_a_boxed_float (block [Var y; Var x]))
             stopped)));
  check
    "with, for the compiler, what it sees where the first variable is an alias \
     of the second: a disagreement"
    (is_a_disagreement
       (outcome stopped
          (with_the_compiler_on
             (then_a_boxed_float ~before:[x, Var y] (block [Var x; Var y]))
             stopped)));
  let without_the_model : static_group_verdict =
    match verdict stopped with
    | Compared compared ->
      Compared
        { compared with
          in_the_model = Nothing_seen "taken away";
          in_the_model_as_simplify = Nothing_seen "taken away"
        }
    | (Neither_starts _ | One_does_not_start _) as unchanged -> unchanged
  in
  check "with nothing read of the model before the stop: a disagreement"
    (is_a_disagreement (outcome stopped without_the_model));
  let explained =
    then_a_boxed_float ~before:[x, Var y] (Static_const_boxed_number (Var y))
  in
  check
    "a boxed number of a variable of kind value that has an alias type, then \
     the boxed number of the naked float: what is seen before the stop is not \
     the same, and is once the model does as Simplify"
    (match verdict explained with
    | Compared
        { stops = Some (1, (_ : string), true);
          passed_untyped = [0];
          in_the_model;
          in_the_compiler;
          in_the_model_as_simplify
        } ->
      (not (same_seen ~threshold in_the_model in_the_compiler))
      && same_seen ~threshold in_the_model_as_simplify in_the_compiler
    | Compared _ | Neither_starts _ | One_does_not_start _ -> false);
  check "it is recorded under T3-10"
    (recorded_under_t3_10 (outcome explained (verdict explained)));
  check
    "with, for the compiler, what it sees of the same group without the alias: \
     a disagreement"
    (is_a_disagreement
       (outcome explained
          (with_the_compiler_on
             (then_a_boxed_float (Static_const_boxed_number (Var y)))
             explained)));
  (* The reading of the model itself is wanted even where the model made to do
     as Simplify sees what Simplify sees. *)
  let without_what_the_model_sees (taken : seen) (group : static_group) :
      static_group_verdict =
    match verdict group with
    | Compared compared -> Compared { compared with in_the_model = taken }
    | (Neither_starts _ | One_does_not_start _) as unchanged -> unchanged
  in
  let recorded_as_gone_past (outcome : outcome) =
    match outcome with
    | Known_to_disagree (label, (_ : string)) ->
      String.starts_with ~prefix:"T3-10, the model gives no type" label
    | Agree | Disagree _ | Not_run _ | Not_this_comparison -> false
  in
  check
    "with only what the model itself sees taken away, the model made to do as \
     Simplify seeing what Simplify sees: a disagreement"
    (is_a_disagreement
       (outcome explained
          (without_what_the_model_sees (Nothing_seen "taken away") explained)));
  check
    "the boxed number of that variable alone, which Simplify goes past: \
     recorded under T3-10"
    (recorded_as_gone_past (outcome alias (verdict alias)));
  check "with only what the model itself sees taken away: a disagreement"
    (is_a_disagreement
       (outcome alias
          (without_what_the_model_sees (Nothing_seen "taken away") alias)));
  check "with bottom in the place of what the model itself sees: a disagreement"
    (is_a_disagreement
       (outcome alias (without_what_the_model_sees Bottom_seen alias)))

(* ======================================================================== *)
(* Joins after joins                                                        *)
(* ======================================================================== *)

(* The joins above are each of environments made from the environment of the
   tests by definitions and equations. Here an environment is made by steps of
   which one may be a join, of environments made by steps from the environment
   before it: a use of a continuation may come after another continuation, and
   then knows the variables that the join at that one brought in or made.

   The one program is the three joins of the shape of T5-10, as the owner of the
   transcription of the join wrote them (FlambdaJoinMeetExamplesSpec.v,
   [Recovered]): the join at [kx] makes [t0] the tag of [x]; the join at [k0]
   brings [x] in from the one use that knows it; the join at [k2] makes a
   variable for a field that is [x] at one use, and records that [t0] is its
   tag, though on a run through the other use of [k0] the variable [x] has no
   value; and where [t0] is then 2 the environment is bottom.

   It is run in the compiler, with the n-way join, in the model with the
   compiler's test for recovering relations, and in the model with the repaired
   test ([Join_env.repaired_choices]), which the compiler has not. What is seen
   at four places is compared: after each join and where [t0] is 2. The model
   with the compiler's test is to see what the compiler sees. The model with the
   repaired test is to see the same after the first two joins, and not after the
   third (T5-10). *)
type joined_step =
  | Defines of int
      (** The variable of that number is defined, at the normal name mode. *)
  | Further_scope
  | Learns of int * MT.flambda_type  (** An equation on that variable. *)
  | Defines_as of int * MT.flambda_type  (** Defined, with an equation. *)
  | Joins of joined_step list list
      (** The join, after the scope of the environment so far, of the
          environments that each list of steps makes from it. *)
  | Looked_at of string  (** What is seen here is kept under that name. *)

(* What a side has for the steps: the model, or the compiler. *)
type 'env side_of_joins =
  { empty : 'env;
    defines : 'env -> int -> 'env;
    further_scope : 'env -> 'env;
    learns : 'env -> int -> MT.flambda_type -> 'env option;
    joins : source:'env -> 'env list -> 'env option;
    (* Whether the environment is marked as one that no run reaches. Nothing
       that is run depends on the answer: it is asked of each use of a join, for
       what is told of a run. *)
    is_bottom : 'env -> bool;
    looks : 'env -> seen
  }

(* The environment that the steps make, if every step answers, and what was
   looked at on the way, in order.

   A join is given every use, also one whose environment is bottom: the join of
   the compiler leaves out the environments that are bottom
   ([Join_env.cut_and_n_way_join_with_analysis]), and that of the model is to do
   the same. [every_use_is_bottom] is told where every use of a join is. *)
let run_joined_steps ?(every_use_is_bottom = fun () -> ())
    (side : 'env side_of_joins) (steps : joined_step list) :
    'env option * (string * seen) list =
  let looked_at = ref [] in
  let rec run env (steps : joined_step list) : 'env option =
    match steps with
    | [] -> Some env
    | Defines i :: steps -> run (side.defines env i) steps
    | Further_scope :: steps -> run (side.further_scope env) steps
    | Learns (i, t) :: steps ->
      Option.bind (side.learns env i t) (fun env -> run env steps)
    | Defines_as (i, t) :: rest -> run env (Defines i :: Learns (i, t) :: rest)
    | Joins uses :: steps ->
      let uses = List.map (run env) uses in
      if List.exists Option.is_none uses
      then None
      else
        let uses = List.filter_map Fun.id uses in
        if List.for_all Fun.id (List.map side.is_bottom uses)
        then every_use_is_bottom ();
        Option.bind (side.joins ~source:env uses) (fun env -> run env steps)
    | Looked_at name :: steps ->
      looked_at := (name, side.looks env) :: !looked_at;
      run env steps
  in
  let env = run side.empty steps in
  env, List.rev !looked_at

(* The names that a join of the model is told to avoid, beside the variables of
   the program and those that its source and its uses know: the variables of the
   targets of every join run before it, or nothing more. *)
type names_avoided =
  | Every_variable_made
  | What_the_environments_know

(* What the side of the model tells of a run: the variables that a meet of the
   model made, where an equation was added; those that a join of the model made;
   and that the environment of a use of a join is bottom. *)
type events_of_joins =
  { made_by_a_meet : M.variable list -> unit;
    made_by_a_join : M.variable list -> unit;
    bottom_use : unit -> unit
  }

let no_events : events_of_joins =
  { made_by_a_meet = ignore; made_by_a_join = ignore; bottom_use = ignore }

(* The two sides, for variables of the kinds given, numbered from 1. The
   variables that the joins make are numbered after them.

   A meet of the model makes a variable one greater than those that its
   environment knows, and takes no other names to avoid: two meets on two paths
   make the same variable, which a join after them takes for one, or has no
   answer for (T7-11, through the join that the meet calls). With
   [a_variable_for_each_use], the first environment of the model, and its
   environment at the start of each use, are given a variable of their own,
   which nothing mentions and the compiler has not: the first is numbered 1000,
   and each next one 1000 more. The uses of a join are run one after the other.
   So the variable of a use is above everything that was made before the use
   began, on any path, as long as nothing was made a thousand above the variable
   then given last; and what a meet or a join makes in a use is above the
   variable of that use. A use then makes no variable that was made before it
   began. It does not follow that what is made later is above what was made
   earlier: once a join of its own is done, a use may make a variable below
   those that the uses of that join made, which are out of scope by then. The
   comparisons run with the argument ask that no variable is made twice. *)
let sides_of_joins ?(avoid = Every_variable_made) ?(events = no_events)
    ?(a_variable_for_each_use = false) (kinds : MK.flambda_kind list) :
    (model -> Model_env.t side_of_joins) * TE.t side_of_joins =
  let numbering = Model_names.create () in
  let variables =
    List.map
      (fun kind ->
        let variable = Variable.create "v" (Converter.kind_of_model kind) in
        ignore (numbering.variables.number variable : N.positive);
        variable)
      kinds
  in
  let threshold = List.length kinds in
  let v = N.positive_of_int in
  let asked = List.mapi (fun i _ : MT.name -> Name_var (v (i + 1))) kinds in
  let variable i = List.nth variables (i - 1) in
  let kind i = List.nth kinds (i - 1) in
  let of_the_model (model : model) : Model_env.t side_of_joins =
    (* The variables of the targets of the joins run so far. A join of the model
       makes a variable one greater than those that it is told to avoid: told
       only of the variables that its environments know, two uses of one join
       that each have a join of their own would each be given the same new
       variable, which the join after them would take for one. The compiler's
       new variables differ from every other. So a join here avoids what every
       join before it made (T7-11), but where a control asks for the other. *)
    let made = ref [] in
    let own_variables = ref 0 in
    let with_a_variable_of_its_own env =
      if a_variable_for_each_use
      then (
        incr own_variables;
        Model_env.add_variable_definition env
          (v (1000 * !own_variables))
          Value Model_name_mode.in_types)
      else env
    in
    (* The variables of an environment that none of the others has. *)
    let new_in env (others : Model_env.t list) : M.variable list =
      let known = List.concat_map Model_join_env.variables_of_env others in
      List.filter
        (fun x -> not (List.exists (fun y -> N.compare_positive x y = 0) known))
        (Model_join_env.variables_of_env env)
    in
    { empty = with_a_variable_of_its_own Model_env.create;
      defines =
        (fun env i ->
          Model_env.add_variable_definition env (v i) (kind i)
            Model_name_mode.normal);
      further_scope =
        (fun env -> with_a_variable_of_its_own (Model_env.increment_scope env));
      learns =
        (fun env i t ->
          let after = model.add_equation_of_model env (Name_var (v i)) t in
          Option.iter
            (fun after -> events.made_by_a_meet (new_in after [env]))
            after;
          after);
      joins =
        (fun ~source uses ->
          Option.map
            (fun (target, _) ->
              events.made_by_a_join (new_in target (source :: uses));
              (match avoid with
              | Every_variable_made ->
                made := Model_join_env.variables_of_env target @ !made
              | What_the_environments_know -> ());
              target)
            (model.join_of_model
               (names_for_a_join_to_avoid
                  (List.mapi (fun i _ -> v (i + 1)) kinds @ !made)
                  ~source ~uses)
               { source;
                 cut_after = Model_env.current_scope source;
                 uses = List.mapi (fun i use -> i, use) uses
               }));
      is_bottom =
        (fun env ->
          let bottom = Model_env.is_bottom env in
          if bottom then events.bottom_use ();
          bottom);
      looks = (fun env -> seen_in_the_model ~threshold asked None env)
    }
  in
  let meet_expanded_head = N_way.meet_expanded_head in
  let of_the_compiler : TE.t side_of_joins =
    { empty = TE.create ~machine_width ~resolver:(fun _ -> None);
      defines =
        (fun env i ->
          TE.add_definition env
            (Bound_name.create_var
               (Bound_var.create (variable i) Flambda_debug_uid.none
                  Name_mode.normal))
            (Variable.kind (variable i)));
      further_scope = TE.increment_scope;
      learns =
        (fun env i t ->
          Some
            (ME.use_meet_env ~meet_expanded_head env ~f:(fun env ->
                 ME.add_equation env
                   (Name.var (variable i))
                   (Converter.type_of_model numbering machine_width t)
                   ~meet_expanded_head)));
      joins =
        (fun ~source uses ->
          Some
            (fst
               (N_way_join.cut_and_n_way_join_with_analysis ~meet_expanded_head
                  ~n_way_join_type:N_way.n_way_join source
                  ~cut_after:(TE.current_scope source)
                  (List.map (fun use -> (), use) uses))));
      is_bottom = TE.is_bottom;
      looks =
        (fun env -> seen_in_the_compiler numbering ~threshold asked None env)
    }
  in
  of_the_model, of_the_compiler

(* The program of T5-10. The variables, in the order of their numbers: [n], a
   value; [t0], a naked immediate; [x], [p], [s], and the blocks made on the way
   [b0], [b1], [pa], [pb], [z], [pd], values. *)
let joins_of_the_shape_of_t5_10 () =
  let n = 1 and t0 = 2 and x = 3 and p = 4 and s = 5 in
  let b0 = 6 and b1 = 7 and pa = 8 and pb = 9 and z = 10 and pd = 11 in
  let kinds : MK.flambda_kind list =
    List.init 11 (fun i : MK.flambda_kind ->
        if i + 1 = t0 then Naked_number Kind_naked_immediate else Value)
  in
  let alias i = MT.alias_type_of Value (Var (N.positive_of_int i)) in
  let block tag fields =
    MT.immutable_block false (N.z_of_int tag) (Scannable Value_only) Heap fields
  in
  let number i = MT.this_naked_immediate (N.z_of_int i) in
  let use_of_kx tag made : joined_step list =
    [ Further_scope;
      Learns (t0, number tag);
      Defines_as (made, block tag [alias n]);
      Learns (x, alias made) ]
  in
  let after_kx = "after the join at kx" in
  let after_k0 = "after the join at k0" in
  let after_k2 = "after the join at k2" in
  let where_two = "where t0 is 2 after the join at k2" in
  let steps : joined_step list =
    [ Defines n;
      Defines t0;
      Defines p;
      Joins
        [ [ Further_scope;
            Defines x;
            Joins [use_of_kx 0 b0; use_of_kx 1 b1];
            Looked_at after_kx;
            Defines_as (pa, block 0 [alias x]);
            Learns (p, alias pa) ];
          [ Further_scope;
            Defines_as (pb, block 1 [alias n]);
            Learns (p, alias pb) ] ];
      Looked_at after_k0;
      Defines s;
      Joins
        [ [Further_scope; Learns (s, alias p)];
          [ Further_scope;
            Learns (t0, number 0);
            Defines_as (z, block 0 [alias n]);
            Defines_as (pd, block 0 [alias z]);
            Learns (s, alias pd) ] ];
      Looked_at after_k2;
      Learns (t0, number 2);
      Looked_at where_two ]
  in
  let threshold = List.length kinds in
  let of_the_model, of_the_compiler = sides_of_joins kinds in
  let in_the_compiler =
    match
      or_fatal_error (fun () ->
          match run_joined_steps of_the_compiler steps with
          | looked_at -> Ok looked_at
          | exception Assert_failure (file, _, _) ->
            Error ("an assertion of " ^ Filename.basename file ^ " fails"))
    with
    | Ok (Ok ((_ : TE.t option), looked_at)) -> Ok looked_at
    | Ok (Error why) -> Error why
    | Error line -> Error ("a fatal error: " ^ general line)
  in
  let in_the_model choices =
    let (_ : Model_env.t option), looked_at =
      run_joined_steps (of_the_model (the_model choices)) steps
    in
    looked_at
  in
  let as_the_compiler = in_the_model Of_the_compiler in
  let with_the_repaired_test =
    in_the_model With_the_repaired_test_of_the_join
  in
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.Joins after joins: the three joins of the shape of T5-10, in the \
     compiler, in the model with the compiler's test for recovering relations \
     and in the model with the repaired test@.";
  match in_the_compiler with
  | Error why ->
    incr failures;
    Format.printf "  the compiler does not answer: %s@." why
  | Ok in_the_compiler ->
    let seen looked_at name : seen =
      match List.assoc_opt name looked_at with
      | Some seen -> seen
      | None -> Nothing_seen "the steps did not come to it"
    in
    let is_seen (seen : seen) =
      match seen with Seen _ -> true | Nothing_seen _ | Bottom_seen -> false
    in
    let same looked_at name =
      let of_model = seen looked_at name in
      let of_compiler = seen in_the_compiler name in
      is_seen of_model && is_seen of_compiler
      && same_seen ~threshold of_model of_compiler
    in
    let told_apart looked_at name =
      let of_model = seen looked_at name in
      let of_compiler = seen in_the_compiler name in
      is_seen of_model && is_seen of_compiler
      && not (same_seen ~threshold of_model of_compiler)
    in
    let is_bottom looked_at name =
      match seen looked_at name with
      | Seen { env_is_bottom; _ } -> Some env_is_bottom
      | Nothing_seen _ | Bottom_seen -> None
    in
    let is expected (answer : bool option) =
      Option.equal Bool.equal answer (Some expected)
    in
    List.iter
      (fun name ->
        check
          (name
         ^ ", the model with the compiler's test sees what the compiler sees")
          (same as_the_compiler name))
      [after_kx; after_k0; after_k2; where_two];
    check "the environment where t0 is 2 is bottom in the compiler"
      (is true (is_bottom in_the_compiler where_two));
    check "and in the model with the compiler's test"
      (is true (is_bottom as_the_compiler where_two));
    List.iter
      (fun name ->
        check
          (name
         ^ ", the model with the repaired test sees what the compiler sees")
          (same with_the_repaired_test name))
      [after_kx; after_k0];
    List.iter
      (fun name ->
        check
          (name
         ^ ", the model with the repaired test and the compiler are told apart \
            (T5-10)")
          (told_apart with_the_repaired_test name))
      [after_k2; where_two];
    check
      "the environment where t0 is 2 is not bottom in the model with the \
       repaired test"
      (is false (is_bottom with_the_repaired_test where_two));
    (* The type that an environment looked at has for [t0]. *)
    let of_t0 looked_at name : string =
      match seen looked_at name with
      | Seen { of_names; _ } -> (
        match
          List.find_opt
            (fun ((name : MT.name), _, _) ->
              MT.name_eqb name (Name_var (N.positive_of_int t0)))
            of_names
        with
        | Some (_, Some t, _) ->
          rename_variables_above threshold
            (Format.asprintf "%a" Model_type.print (Model_type.normalise t))
        | Some (_, None, _) | None -> "not defined")
      | Nothing_seen why -> "nothing is seen: " ^ why
      | Bottom_seen -> "bottom"
    in
    Format.printf "  the type of t0 after the join at k2, in the compiler: %s@."
      (of_t0 in_the_compiler after_k2);
    Format.printf "  in the model with the compiler's test: %s@."
      (of_t0 as_the_compiler after_k2);
    Format.printf "  in the model with the repaired test: %s@."
      (of_t0 with_the_repaired_test after_k2)

(* Generated programs of joins. Twelve values, numbered from 1, and three naked
   immediates after them. The first two values and the immediates are defined at
   the start and the third value after the first join; each of the other values
   is defined at most once, at a use, so that no variable is defined on two
   paths.

   A program is a join of two uses, a value defined after it, a join of two uses
   that give that value a type, and at times an equation on an immediate after
   that. A use of the first join may begin with a join of its own, at a
   continuation whose parameter it defines. A use learns a number for an
   immediate, makes blocks, whose tag is most often that number, and gives the
   values that it is to give a type an alias of a value in scope or a block. So
   the tag of a value is often an immediate at every use of a join, which is
   where a join records a relation; and a use of the second join knows the
   variables that the first brought in.

   With [meets_of_cases], a use may also give a value two variants, as the two
   types of the shape of T4-9 ([generate_meet_of_cases]), whose meet makes
   variables: the blocks of the first, of tag 0 or 1, each hold one value, and
   those of the second a block whose fields repeat one value, or a number. Six
   of the values are kept for that and defined at the start: three for what the
   blocks of a first variant hold, and three for what is repeated. The variants
   mention no other name, and no step gives any of the six a type: so a variant
   does not reach, through what it mentions, the value that it is given to, nor
   any value that a step gives a type. Without the argument, the programs are
   those that were generated before it was there. *)
let kinds_of_generated_joins : MK.flambda_kind list =
  List.init 15 (fun i : MK.flambda_kind ->
      if i < 12 then Value else Naked_number Kind_naked_immediate)

let generate_joined_steps ?(meets_of_cases = false) r : joined_step list =
  let immediates = [13; 14; 15] in
  let held = [7; 8; 9] and repeated = [10; 11; 12] in
  let unused =
    ref (if meets_of_cases then [4; 5; 6] else [4; 5; 6; 7; 8; 9; 10; 11; 12])
  in
  let fresh () =
    match !unused with
    | [] -> None
    | x :: rest ->
      unused := rest;
      Some x
  in
  let alias i = MT.alias_type_of Value (Var (N.positive_of_int i)) in
  let number i = MT.this_naked_immediate (N.z_of_int i) in
  let block tag fields =
    MT.immutable_block false (N.z_of_int tag) (Scannable Value_only) Heap fields
  in
  (* A use: the values in scope, and those that it is to give a type. *)
  let rec use ~depth (in_scope : int list) (targets : int list) :
      joined_step list =
    let in_scope = ref in_scope in
    let steps = ref [Further_scope] in
    let add step = steps := step :: !steps in
    (* A join of its own, at a continuation whose parameter it defines. *)
    (if depth > 0 && G.chance r ~percent:50
     then
       match fresh () with
       | None -> ()
       | Some parameter ->
         add (Defines parameter);
         let before = !in_scope in
         add
           (Joins
              [ use ~depth:(depth - 1) before [parameter];
                use ~depth:(depth - 1) before [parameter] ]);
         in_scope := parameter :: before);
    let tag =
      if G.chance r ~percent:70
      then (
        let tag = G.below r 2 in
        add (Learns (G.pick r immediates, number tag));
        tag)
      else G.below r 2
    in
    let field () =
      if G.chance r ~percent:70
      then alias (G.pick r !in_scope)
      else MT.this_tagged_immediate (G.small_number r)
    in
    let a_block () =
      block
        (if G.chance r ~percent:80 then tag else G.below r 2)
        (List.init (1 + G.below r 2) (fun _ -> field ()))
    in
    List.iter
      (fun target ->
        if meets_of_cases && G.chance r ~percent:50
        then (
          let held = alias (G.pick r held) in
          let field () =
            if G.chance r ~percent:20
            then MT.this_tagged_immediate (G.small_number r)
            else
              let repeated = alias (G.pick r repeated) in
              variant_of_blocks
                [0, List.init (1 + G.below r 3) (fun _ -> repeated)]
          in
          let first = field () in
          let second = field () in
          add (Learns (target, variant_of_blocks [0, [held]; 1, [held]]));
          add (Learns (target, variant_of_blocks [0, [first]; 1, [second]])))
        else
          match G.below r 4 with
          | 0 -> ()
          | 1 -> add (Learns (target, alias (G.pick r !in_scope)))
          | _ -> (
            match fresh () with
            | None -> add (Learns (target, a_block ()))
            | Some made ->
              add (Defines_as (made, a_block ()));
              in_scope := made :: !in_scope;
              add (Learns (target, alias made))))
      targets;
    List.rev !steps
  in
  let first = Joins [use ~depth:1 [1; 2] [1; 2]; use ~depth:1 [1; 2] [1; 2]] in
  let second = Joins [use ~depth:0 [1; 2; 3] [3]; use ~depth:0 [1; 2; 3] [3]] in
  List.map
    (fun i -> Defines i)
    ([1; 2] @ immediates @ if meets_of_cases then held @ repeated else [])
  @ [first; Looked_at "after the first join"; Defines 3; second]
  @ [Looked_at "after the second join"]
  @
  if G.chance r ~percent:50
  then
    [ Learns (G.pick r immediates, number (G.below r 3));
      Looked_at "after an equation on an immediate" ]
  else []

let rec print_joined_steps ppf (steps : joined_step list) =
  Format.fprintf ppf "@[<v>%a@]"
    (Format.pp_print_list ~pp_sep:Format.pp_print_cut
       (fun ppf (step : joined_step) ->
         let variable ppf i =
           Model_type.print_variable ppf (N.positive_of_int i)
         in
         match step with
         | Defines i -> Format.fprintf ppf "%a is defined" variable i
         | Further_scope -> Format.pp_print_string ppf "one scope further"
         | Learns (i, t) ->
           Format.fprintf ppf "@[<hov 2>%a :@ %a@]" variable i Model_type.print
             t
         | Defines_as (i, t) ->
           Format.fprintf ppf "@[<hov 2>%a is defined, and@ %a :@ %a@]" variable
             i variable i Model_type.print t
         | Joins uses ->
           Format.fprintf ppf "@[<v>the join of@ %a@]"
             (Format.pp_print_list ~pp_sep:Format.pp_print_cut (fun ppf use ->
                  Format.fprintf ppf "@[<v 2>a use:@ %a@]" print_joined_steps
                    use))
             uses
         | Looked_at name -> Format.fprintf ppf "(looked at: %s)" name))
    steps

(* One step less, anywhere but for the places looked at; a join with one use
   less. What is left may not define what it mentions: then neither side
   answers. *)
let rec shrink_joined_steps (steps : joined_step list) : joined_step list list =
  let without i = List.filteri (fun j _ -> i <> j) steps in
  let with_at i step =
    List.mapi (fun j other -> if i = j then step else other) steps
  in
  List.concat
    (List.mapi
       (fun i (step : joined_step) ->
         match step with
         | Looked_at _ -> []
         | Defines _ | Further_scope | Learns _ | Defines_as _ -> [without i]
         | Joins uses ->
           (match uses with
             | [] | [_] -> []
             | _ :: _ :: _ ->
               List.mapi
                 (fun k _ ->
                   with_at i (Joins (List.filteri (fun l _ -> k <> l) uses)))
                 uses)
           @ List.concat
               (List.mapi
                  (fun k use ->
                    List.map
                      (fun use ->
                        with_at i
                          (Joins
                             (List.mapi
                                (fun l other -> if k = l then use else other)
                                uses)))
                      (shrink_joined_steps use))
                  uses))
       steps)

(* What the two sides see at the places looked at, or why one does not
   answer. *)
type joins_run =
  | Neither_answers of string
  | Only_one_answers of string
  | Both_answer of (string * seen * seen) list

let joined_steps_in_the_compiler (of_the_compiler : TE.t side_of_joins)
    (steps : joined_step list) : ((string * seen) list, string) result =
  match
    or_fatal_error (fun () ->
        match run_joined_steps of_the_compiler steps with
        | (_ : TE.t option), looked_at -> Ok looked_at
        | exception Assert_failure (file, _, _) ->
          Error ("an assertion of " ^ Filename.basename file ^ " fails"))
  with
  | Ok (Ok looked_at) -> Ok looked_at
  | Ok (Error why) -> Error why
  | Error line -> Error ("a fatal error: " ^ general line)

let joined_steps_in_the_model ?every_use_is_bottom
    (of_the_model : Model_env.t side_of_joins) (steps : joined_step list) :
    ((string * seen) list, string) result =
  match run_joined_steps ?every_use_is_bottom of_the_model steps with
  | Some (_ : Model_env.t), looked_at -> Ok looked_at
  | None, (_ : (string * seen) list) -> Error "no answer of a step"

(* A program run on both sides. [every_use_is_bottom] is told where, in the
   model, every use of a join is. *)
let run_generated_joins ?avoid ?events ?a_variable_for_each_use
    ?every_use_is_bottom choices (steps : joined_step list) : joins_run =
  let of_the_model, of_the_compiler =
    sides_of_joins ?avoid ?events ?a_variable_for_each_use
      kinds_of_generated_joins
  in
  let of_the_model = of_the_model (the_model choices) in
  match
    ( joined_steps_in_the_model ?every_use_is_bottom of_the_model steps,
      joined_steps_in_the_compiler of_the_compiler steps )
  with
  | Error why, Error why' ->
    Neither_answers (Printf.sprintf "the model: %s; the compiler: %s" why why')
  | Error why, Ok _ -> Only_one_answers ("the model: " ^ why)
  | Ok _, Error why -> Only_one_answers ("the compiler: " ^ why)
  | Ok of_model, Ok of_compiler ->
    Both_answer
      (List.map2
         (fun (name, seen) ((_ : string), seen') -> name, seen, seen')
         of_model of_compiler)

let seen_is_read (seen : seen) =
  match seen with Seen _ -> true | Nothing_seen _ | Bottom_seen -> false

(* The first place at which the two do not see the same. A place at which one of
   the two is not read is one. *)
let first_difference_of_joins (places : (string * seen * seen) list) :
    (string * seen * seen) option =
  let threshold = List.length kinds_of_generated_joins in
  List.find_opt
    (fun ((_ : string), seen, seen') ->
      not
        (seen_is_read seen && seen_is_read seen'
        && same_seen ~threshold seen seen'))
    places

(* A difference is shown without the variables that are not in scope at the
   place, of which nothing is seen on either side. *)
let print_difference_of_joins (name, seen, seen') : string =
  let threshold = List.length kinds_of_generated_joins in
  let in_scope (seen : seen) : seen =
    match seen with
    | Nothing_seen _ | Bottom_seen -> seen
    | Seen { answer; env_is_bottom; of_names; of_others } ->
      Seen
        { answer;
          env_is_bottom;
          of_names =
            List.filter
              (fun ((_ : MT.name), t, canonical) ->
                Option.is_some t || Option.is_some canonical)
              of_names;
          of_others
        }
  in
  Format.asprintf "@[<v>%s, the model:@ %a@ the compiler:@ %a@]" name
    print_lines
    (print_seen ~threshold (in_scope seen))
    print_lines
    (print_seen ~threshold (in_scope seen'))

(* What a run of the model with the compiler's test comes to: the two are to see
   the same at every place looked at. *)
let outcome_of_generated_joins (run : joins_run) : outcome =
  match run with
  | Neither_answers why -> Not_run ("neither answers (" ^ why ^ ")")
  | Only_one_answers why -> Disagree ("only one answers (" ^ why ^ ")")
  | Both_answer places -> (
    match first_difference_of_joins places with
    | None -> Agree
    | Some difference -> Disagree (print_difference_of_joins difference))

(* What a run of the model with the repaired test comes to, given the run of the
   model with the compiler's test on the same program. The repaired test is not
   the compiler's. A difference is recorded (T5-10), and is not a disagreement,
   only where both sides are read at every place looked at and the model with
   the compiler's test sees what the compiler sees at every place: then the
   choice of the test is all that differs between the two runs of the model.
   What the difference is, is not looked into. A place that is not read is a
   disagreement. *)
let outcome_of_the_repaired_test ~(with_the_test_of_the_compiler : joins_run)
    (with_the_repaired_test : joins_run) : outcome =
  match with_the_repaired_test with
  | Neither_answers why -> Not_run ("neither answers (" ^ why ^ ")")
  | Only_one_answers why -> Disagree ("only one answers (" ^ why ^ ")")
  | Both_answer places -> (
    match
      List.find_opt
        (fun ((_ : string), seen, seen') ->
          not (seen_is_read seen && seen_is_read seen'))
        places
    with
    | Some unread ->
      Disagree
        ("what is seen is not read on both sides: "
        ^ print_difference_of_joins unread)
    | None -> (
      match first_difference_of_joins places with
      | None -> Agree
      | Some difference -> (
        match outcome_of_generated_joins with_the_test_of_the_compiler with
        | Agree ->
          Known_to_disagree
            ( "T5-10, the model with the repaired test for recovering \
               relations, which the compiler has not, sees otherwise than the \
               compiler where the model with the compiler's test sees what the \
               compiler sees",
              print_difference_of_joins difference )
        | Not_this_comparison | Not_run _ | Disagree _ | Known_to_disagree _ ->
          Disagree (print_difference_of_joins difference))))

let generated_joins_after_joins () : joined_step list comparison list =
  let with_the_test_of_the_compiler (steps : joined_step list) : outcome =
    outcome_of_generated_joins (run_generated_joins Of_the_compiler steps)
  in
  let a_use_after_a_join (steps : joined_step list) =
    List.exists
      (fun (step : joined_step) ->
        match step with
        | Joins uses ->
          List.exists
            (List.exists (fun (step : joined_step) ->
                 match step with
                 | Joins _ -> true
                 | Defines _ | Further_scope | Learns _ | Defines_as _
                 | Looked_at _ ->
                   false))
            uses
        | Defines _ | Further_scope | Learns _ | Defines_as _ | Looked_at _ ->
          false)
      steps
  in
  [ { name = "joins after joins: the model and the n-way join of the compiler";
      compare = with_the_test_of_the_compiler
    };
    { name =
        "of which those where a use of the first join has a join of its own";
      compare =
        (fun steps ->
          if a_use_after_a_join steps
          then with_the_test_of_the_compiler steps
          else Not_this_comparison)
    };
    { name =
        "joins after joins: the model with the repaired test for recovering \
         relations, and the compiler";
      compare =
        (fun steps ->
          outcome_of_the_repaired_test
            ~with_the_test_of_the_compiler:
              (run_generated_joins Of_the_compiler steps)
            (run_generated_joins With_the_repaired_test_of_the_join steps))
    } ]

(* A run with, at the place of a position, what the model sees or what the
   compiler sees replaced: for the controls. *)
let with_a_reading_replaced position ~of_the_model (taken : seen)
    (run : joins_run) : joins_run =
  match run with
  | Both_answer places ->
    Both_answer
      (List.mapi
         (fun i ((name, seen, seen') as place) ->
           if i <> position
           then place
           else if of_the_model
           then name, taken, seen'
           else name, seen, taken)
         places)
  | (Neither_answers _ | Only_one_answers _) as unchanged -> unchanged

(* Whether a run tells the model and the compiler apart, and at which place
   first. That is said only of a run in which both sides answer and every place
   looked at is read on both: a reading that fails is not a difference between
   the two. *)
type told_apart =
  | No_answer of string
  | A_place_is_not_read
  | Not_told_apart
  | Told_apart_first_at of string

let told_apart_where_read (run : joins_run) : told_apart =
  match run with
  | Neither_answers why | Only_one_answers why -> No_answer why
  | Both_answer places -> (
    if
      not
        (List.for_all
           (fun ((_ : string), seen, seen') ->
             seen_is_read seen && seen_is_read seen')
           places)
    then A_place_is_not_read
    else
      match first_difference_of_joins places with
      | Some (place, (_ : seen), (_ : seen)) -> Told_apart_first_at place
      | None -> Not_told_apart)

(* A control of the names that a join of the model is told to avoid (T7-11), on
   two of the generated programs in which two uses of the first join each have a
   join of their own. Told to avoid every variable made so far, the model sees
   what the compiler sees. Told only of the variables that its environments
   know, it does not: the two inner joins are each given the same new variable.
   That the two are told apart is said only where every place looked at is read
   on both sides ([told_apart_where_read]): with a reading of that run taken
   away, it is not said. That run is not one of the comparisons above: it shows
   what they leave out. *)
let control_of_the_names_to_avoid () =
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: two generated programs of joins after joins, with the joins \
     of the model told to avoid every variable made so far, and told only of \
     the variables that their environments know (T7-11)@.";
  List.iter
    (fun seed ->
      let steps = generate_joined_steps (Random.of_int seed) in
      let run avoid = run_generated_joins ~avoid Of_the_compiler steps in
      check
        (Printf.sprintf
           "seed %d, every variable made so far avoided: the model sees what \
            the compiler sees at every place looked at"
           seed)
        (match run Every_variable_made with
        | Both_answer places ->
          Option.is_none (first_difference_of_joins places)
        | Neither_answers _ | Only_one_answers _ -> false);
      let narrow = run What_the_environments_know in
      let told_apart =
        Printf.sprintf
          "seed %d, only the variables that the environments know avoided: \
           every place looked at is read on both sides, and the model and the \
           compiler are told apart"
          seed
      in
      (match told_apart_where_read narrow with
      | Told_apart_first_at place ->
        check (told_apart ^ ", first " ^ place) true
      | No_answer why -> check (told_apart ^ " (" ^ general why ^ ")") false
      | A_place_is_not_read ->
        check (told_apart ^ " (a place is not read)") false
      | Not_told_apart -> check (told_apart ^ " (the two see the same)") false);
      let not_read (run : joins_run) =
        match told_apart_where_read run with
        | A_place_is_not_read -> true
        | No_answer _ | Not_told_apart | Told_apart_first_at _ -> false
      in
      let last =
        match narrow with
        | Both_answer places -> List.length places - 1
        | Neither_answers _ | Only_one_answers _ -> 0
      in
      let without position ~of_the_model =
        with_a_reading_replaced position ~of_the_model
          (Nothing_seen "taken away") narrow
      in
      check
        "with nothing read of the model at the first place of that run: the \
         two are not said to be told apart"
        (not_read (without 0 ~of_the_model:true));
      check
        "with nothing read of the compiler at the last place of that run: the \
         two are not said to be told apart"
        (not_read (without last ~of_the_model:false)))
    [161196; 161457]

(* Joins after meets that make variables: the programs generated with
   [meets_of_cases]. *)

(* A run of such a program, with what the side of the model told of it. *)
type run_after_meets =
  { run : joins_run;
    a_meet_made_a_variable : bool;
    (* A variable was made twice, by two meets or joins of the model. *)
    a_variable_was_made_twice : bool;
    (* The environment of a use of a join was bottom. *)
    a_use_was_bottom : bool;
    every_use_of_a_join_was_bottom : bool
  }

let run_after_meets ~(a_variable_for_each_use : bool) (steps : joined_step list)
    : run_after_meets =
  let made = ref [] in
  let by_a_meet = ref false in
  let twice = ref false in
  let bottom = ref false in
  let every_use = ref false in
  let record (variables : M.variable list) =
    if
      List.exists
        (fun x -> List.exists (fun y -> N.compare_positive x y = 0) !made)
        variables
    then twice := true;
    made := variables @ !made
  in
  let events : events_of_joins =
    { made_by_a_meet =
        (fun variables ->
          if not (List.is_empty variables) then by_a_meet := true;
          record variables);
      made_by_a_join = record;
      bottom_use = (fun () -> bottom := true)
    }
  in
  let run =
    run_generated_joins ~events ~a_variable_for_each_use
      ~every_use_is_bottom:(fun () -> every_use := true)
      Of_the_compiler steps
  in
  { run;
    a_meet_made_a_variable = !by_a_meet;
    a_variable_was_made_twice = !twice;
    a_use_was_bottom = !bottom;
    every_use_of_a_join_was_bottom = !every_use
  }

(* What a run comes to where each use of the model has a variable of its own:
   the model is to see what the compiler sees at every place, and no variable is
   to be made twice. *)
let outcome_with_a_variable_for_each_use (run : run_after_meets) : outcome =
  if run.a_variable_was_made_twice
  then
    Disagree
      "a variable is made twice by the model, though each use has a variable \
       of its own"
  else outcome_of_generated_joins run.run

(* What a run comes to where the uses of the model have no variable of their
   own. A run in which the two do not see the same, or in which only one of the
   two answers, is recorded under T7-11 only where a variable was made twice in
   that run and the same program, run with a variable for each use, comes to an
   agreement, which asks that no variable be made twice there: it is a
   disagreement otherwise. A run in which neither answers is not run, as for the
   other comparisons of these programs, whatever was made in it. That the two
   that made the variable are on two paths is not looked into. *)
let outcome_without_a_variable_for_each_use
    ~(with_a_variable_for_each_use : outcome) (run : run_after_meets) : outcome
    =
  match outcome_of_generated_joins run.run with
  | Disagree what -> (
    match with_a_variable_for_each_use with
    | Agree when run.a_variable_was_made_twice ->
      Known_to_disagree
        ( "T7-11, a variable is made twice by the meets and the joins of the \
           model, and the model does not see what the compiler sees, or only \
           one of the two answers; with a variable of its own for each use \
           none is made twice and the model sees what the compiler sees",
          what )
    | Agree | Disagree _ | Not_run _ | Not_this_comparison | Known_to_disagree _
      ->
      Disagree what)
  | (Agree | Not_run _ | Not_this_comparison | Known_to_disagree _) as outcome
    ->
    outcome

(* The compiler alone, with its debug check for concrete types on canonical
   names only (T7-13): it is to stop on none of these programs. *)
let the_check_of_the_compiler_passes (steps : joined_step list) : outcome =
  let flag = Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals in
  let before = !flag in
  flag := true;
  Fun.protect
    ~finally:(fun () -> flag := before)
    (fun () ->
      let (_ : model -> Model_env.t side_of_joins), of_the_compiler =
        sides_of_joins kinds_of_generated_joins
      in
      match joined_steps_in_the_compiler of_the_compiler steps with
      | Ok (_ : (string * seen) list) -> Agree
      | Error why -> Disagree ("the compiler stops: " ^ why))

let generated_joins_after_meets () : joined_step list comparison list =
  (* The run of a program with a variable for each use, which five of the
     comparisons ask for, is kept for the program as it is given, not for one
     that is equal to it: each program is run so once. *)
  let kept : (joined_step list * run_after_meets) list ref = ref [] in
  let with_a_variable (steps : joined_step list) =
    match List.assq_opt steps !kept with
    | Some run -> run
    | None ->
      let run = run_after_meets ~a_variable_for_each_use:true steps in
      kept := (steps, run) :: !kept;
      run
  in
  let where (holds : run_after_meets -> bool) (steps : joined_step list) :
      outcome =
    let run = with_a_variable steps in
    if holds run
    then outcome_with_a_variable_for_each_use run
    else Not_this_comparison
  in
  [ { name =
        "joins after meets, each use of the model with a variable of its own: \
         the model and the compiler";
      compare = where (fun (_ : run_after_meets) -> true)
    };
    { name = "of which those where a meet of the model makes a variable";
      compare = where (fun run -> run.a_meet_made_a_variable)
    };
    { name = "of which those where a use of a join is bottom";
      compare = where (fun run -> run.a_use_was_bottom)
    };
    { name = "of which those where every use of a join is bottom";
      compare = where (fun run -> run.every_use_of_a_join_was_bottom)
    };
    { name =
        "the compiler alone, with its check for concrete types on canonical \
         names only";
      compare = the_check_of_the_compiler_passes
    };
    { name =
        "joins after meets, the uses of the model with no variable of their \
         own: the model and the compiler";
      compare =
        (fun steps ->
          outcome_without_a_variable_for_each_use
            ~with_a_variable_for_each_use:
              (outcome_with_a_variable_for_each_use (with_a_variable steps))
            (run_after_meets ~a_variable_for_each_use:false steps))
    } ]

(* A control of what a run without a variable for each use comes to
   ([outcome_without_a_variable_for_each_use]), on the first two of the
   generated programs on which the model then makes a variable twice and does
   not see what the compiler sees: the first on which it has no answer, and the
   first on which both answer. Each is recorded; it is a disagreement where no
   variable is said to be made twice, and where the run with a variable for each
   use is said not to agree; and it is not run where neither side is said to
   answer. *)
let control_of_the_variables_of_the_meets ~seed ~count =
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: two of the generated programs of joins after meets, the uses \
     of the model with and without a variable of their own (T7-11)@.";
  let is_recorded (outcome : outcome) =
    match outcome with
    | Known_to_disagree _ -> true
    | Agree | Disagree _ | Not_run _ | Not_this_comparison -> false
  in
  let agrees (outcome : outcome) =
    match outcome with
    | Agree -> true
    | Known_to_disagree _ | Disagree _ | Not_run _ | Not_this_comparison ->
      false
  in
  let is_a_disagreement (outcome : outcome) =
    match outcome with
    | Disagree _ -> true
    | Agree | Known_to_disagree _ | Not_run _ | Not_this_comparison -> false
  in
  let is_not_run (outcome : outcome) =
    match outcome with
    | Not_run _ -> true
    | Agree | Known_to_disagree _ | Disagree _ | Not_this_comparison -> false
  in
  (* The first program, by its seed, whose run without a variable for each use
     makes a variable twice, does not agree, and is of the sort asked. *)
  let first_of (sort : joins_run -> bool) :
      (int * joined_step list * run_after_meets) option =
    let rec from case_seed =
      if case_seed >= seed + count
      then None
      else
        let steps =
          generate_joined_steps ~meets_of_cases:true (Random.of_int case_seed)
        in
        let without = run_after_meets ~a_variable_for_each_use:false steps in
        if
          without.a_variable_was_made_twice
          && is_a_disagreement (outcome_of_generated_joins without.run)
          && sort without.run
        then Some (case_seed, steps, without)
        else from (case_seed + 1)
    in
    from seed
  in
  List.iter
    (fun (sort, what) ->
      match first_of sort with
      | None -> check ("there is a program on which " ^ what) false
      | Some (case_seed, steps, without) ->
        Format.printf
          "  the first program on which, with no variable for each use, a \
           variable is made twice and %s: seed %d@."
          what case_seed;
        let with_one =
          outcome_with_a_variable_for_each_use
            (run_after_meets ~a_variable_for_each_use:true steps)
        in
        check
          "with a variable for each use, the model sees what the compiler sees \
           at every place looked at, and no variable is made twice"
          (agrees with_one);
        check "without, the run is recorded under T7-11"
          (is_recorded
             (outcome_without_a_variable_for_each_use
                ~with_a_variable_for_each_use:with_one without));
        check "said to have made no variable twice, it is a disagreement"
          (is_a_disagreement
             (outcome_without_a_variable_for_each_use
                ~with_a_variable_for_each_use:with_one
                { without with a_variable_was_made_twice = false }));
        check
          "with the run that has a variable for each use said to disagree, it \
           is a disagreement"
          (is_a_disagreement
             (outcome_without_a_variable_for_each_use
                ~with_a_variable_for_each_use:(Disagree "said so") without));
        check "with neither side said to answer, it is not run"
          (is_not_run
             (outcome_without_a_variable_for_each_use
                ~with_a_variable_for_each_use:with_one
                { without with run = Neither_answers "said so" })))
    [ ( (fun (run : joins_run) ->
          match run with
          | Only_one_answers _ -> true
          | Neither_answers _ | Both_answer _ -> false),
        "only one side answers" );
      ( (fun (run : joins_run) ->
          match run with
          | Both_answer _ -> true
          | Neither_answers _ | Only_one_answers _ -> false),
        "both sides answer and do not see the same" ) ]

(* A control of the comparison that runs the compiler alone with its check
   ([the_check_of_the_compiler_passes]): the steps of the scenario of T7-13 with
   three fields ([the_meet_of_t7_13]), on which the check stops the compiler in
   every order of the identifiers. The meet of the scenario, of the alias of the
   variable 2 and a type, is written as an equation on the variable 2. *)
let control_of_the_check_of_the_compiler () =
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: the compiler alone with its check for concrete types on \
     canonical names only, on the steps of T7-13 with three fields@.";
  let { label = _; kinds = _; before; met = first, second; given_to = _ } =
    the_meet_of_t7_13 ~fields:3
  in
  let steps : joined_step option list =
    List.map
      (fun (step : step) : joined_step option ->
        match step with
        | Define i -> Some (Defines i)
        | Equation (i, t) -> Some (Learns (i, t))
        | Relation _ | Meet_of _ -> None)
      before
  in
  let the_meet : joined_step option =
    match MT.get_alias_opt first with
    | Some (Var x) ->
      Option.map (fun i -> Learns (i, second)) (N.int_of_positive x)
    | Some (Symbol _ | Const _) | None -> None
  in
  let steps = steps @ [the_meet] in
  check
    "the steps are those of the scenario, its meet written as an equation on \
     the variable that is met"
    (List.for_all Option.is_some steps);
  let steps = List.filter_map Fun.id steps in
  let flag = Oxcaml_flags.Flambda2.Debug.concrete_types_only_on_canonicals in
  let as_it_was = !flag in
  let (_ : model -> Model_env.t side_of_joins), of_the_compiler =
    sides_of_joins kinds_of_generated_joins
  in
  let without_the_check =
    flag := false;
    Fun.protect
      ~finally:(fun () -> flag := as_it_was)
      (fun () -> joined_steps_in_the_compiler of_the_compiler steps)
  in
  check "without the check, the compiler does not stop on them"
    (Result.is_ok without_the_check);
  check "with the check, it is said to stop, with the fatal error of the check"
    (match the_check_of_the_compiler_passes steps with
    | Disagree what ->
      String.starts_with
        ~prefix:
          "the compiler stops: a fatal error: Trying to add equation giving \
           concrete type on"
        what
    | Agree | Not_run _ | Not_this_comparison | Known_to_disagree _ -> false);
  check "and the check is as it was before" (Bool.equal !flag as_it_was)

(* A control of what a run with the repaired test comes to
   ([outcome_of_the_repaired_test]), on a generated program on which the
   repaired test changes what is seen and on one on which it does not: with one
   reading taken away, on either side, it is a disagreement, and is neither
   recorded nor counted as agreeing. *)
let control_of_the_repaired_test () =
  let check what holds =
    if holds
    then Format.printf "  %s: yes@." what
    else (
      incr failures;
      Format.printf "  %s: NO@." what)
  in
  Format.printf
    "@.A control: what a run of the model with the repaired test comes to, \
     with a reading taken away@.";
  let is_recorded (outcome : outcome) =
    match outcome with
    | Known_to_disagree _ -> true
    | Agree | Disagree _ | Not_run _ | Not_this_comparison -> false
  in
  let agrees (outcome : outcome) =
    match outcome with
    | Agree -> true
    | Known_to_disagree _ | Disagree _ | Not_run _ | Not_this_comparison ->
      false
  in
  let is_a_disagreement (outcome : outcome) =
    match outcome with
    | Disagree _ -> true
    | Agree | Known_to_disagree _ | Not_run _ | Not_this_comparison -> false
  in
  let with_at = with_a_reading_replaced in
  List.iter
    (fun (seed, changed) ->
      let steps = generate_joined_steps (Random.of_int seed) in
      let baseline = run_generated_joins Of_the_compiler steps in
      let repaired =
        run_generated_joins With_the_repaired_test_of_the_join steps
      in
      let outcome =
        outcome_of_the_repaired_test ~with_the_test_of_the_compiler:baseline
      in
      let nothing : seen = Nothing_seen "taken away" in
      check
        (Printf.sprintf
           "seed %d: the model with the compiler's test sees what the compiler \
            sees"
           seed)
        (agrees (outcome_of_generated_joins baseline));
      if changed
      then
        check "the run with the repaired test is recorded under T5-10"
          (is_recorded (outcome repaired))
      else
        check "the run with the repaired test agrees"
          (agrees (outcome repaired));
      check "with nothing read of the model at the first place: a disagreement"
        (is_a_disagreement
           (outcome (with_at 0 ~of_the_model:true nothing repaired)));
      check "with bottom for the model at the second place: a disagreement"
        (is_a_disagreement
           (outcome (with_at 1 ~of_the_model:true Bottom_seen repaired)));
      check
        "with nothing read of the compiler at the second place: a disagreement"
        (is_a_disagreement
           (outcome (with_at 1 ~of_the_model:false nothing repaired))))
    [160037, true; 160000, false]

(* ======================================================================== *)
(* The tests                                                                *)
(* ======================================================================== *)

let () =
  let comp_unit = Compilation_unit.of_string "Types_difftest" in
  Env.set_current_unit
    (Unit_info.make_dummy ~input_name:"types_difftest" comp_unit);
  let names = create_names () in
  let structural = Oxcaml_flags.Set Oxcaml_flags.Structural in
  Oxcaml_flags.Flambda2.join_sharing := structural;
  Format.printf
    "The types of the compiler against the Rocq model.@.Stack allocation: \
     %s.@.Joins: with -flambda2-join-sharing structural, but where the \
     compiler's default is named.@."
    (if Flambda2_ui.Flambda_features.stack_allocation_enabled ()
     then "enabled"
     else "disabled");
  ranges_of_numbers ();
  comparison_by_text names;
  rec_info_cases names;
  run ~title:"Raw types of the model" ~seed:1000 ~count:3000
    ~generate:(fun r ->
      G.flambda_type { well_formed = false; depth = 3 } r names.pool)
    ~prepare:(both names) ~shrink:G.shrink ~print:Model_type.print
    (comparisons_of_one_type names);
  run ~title:"Types of the model that the compiler's functions make" ~seed:5000
    ~count:3000
    ~generate:(fun r ->
      G.flambda_type { well_formed = true; depth = 3 } r names.pool)
    ~prepare:(both names) ~shrink:G.shrink ~print:Model_type.print
    (comparisons_of_one_type names
    @ round_trip_from_the_compiler names
    @ well_formed_types names);
  let pairs_for_equality ~title ~seed (config : G.config) =
    run ~title ~seed ~count:1500
      ~generate:(generate_pair_for_equality names config)
      ~prepare:(fun (m1, m2) ->
        ignore (to_compiler names m1 : TG.t);
        ignore (to_compiler names m2 : TG.t);
        m1, m2)
      ~shrink:(fun (m1, m2) ->
        List.map (fun m1 -> m1, m2) (G.shrink m1)
        @ List.map (fun m2 -> m1, m2) (G.shrink m2))
      ~print:(fun ppf (m1, m2) ->
        Format.fprintf ppf "@[<v>%a@ and@ %a@]" Model_type.print m1
          Model_type.print m2)
      (equality_of_types names)
  in
  pairs_for_equality
    ~title:
      "Two types of the model that the compiler's functions make, for the \
       equality of types"
    ~seed:12000
    { well_formed = true; depth = 3 };
  pairs_for_equality
    ~title:"Two raw types of the model, for the equality of types" ~seed:14000
    { well_formed = false; depth = 2 };
  run ~title:"The creation functions" ~seed:20000 ~count:10000
    ~generate:(generate_call names)
    ~prepare:(fun call -> call)
    ~shrink:(fun _ -> [])
    ~print:print_call (creation_functions names);
  run ~title:"A relation and a constant" ~seed:30000 ~count:600
    ~generate:(fun r ->
      let config : G.config = { well_formed = true; depth = 0 } in
      let kind = G.pick r G.all_kinds in
      ( G.relation r,
        match G.const_of_kind config r kind with
        | Some const -> const
        | None -> M.Poison kind ))
    ~prepare:(fun input -> input)
    ~shrink:(fun _ -> [])
    ~print:(fun ppf (relation, const) ->
      Format.fprintf ppf "%a of %a" Model_type.print_relation relation
        Model_type.print_const const)
    (relation_of_const names);
  run ~title:"Raw row-like structures of blocks" ~seed:31000 ~count:1000
    ~generate:(fun r ->
      G.row_like_for_blocks { well_formed = false; depth = 1 } r names.pool)
    ~prepare:(fun blocks ->
      ignore
        (Converter.row_like_for_blocks_of_model names.numbering machine_width
           blocks
          : TG.row_like_for_blocks);
      blocks)
    ~shrink:G.shrink_row_like_for_blocks
    ~print:(fun ppf blocks ->
      Model_type.print ppf (blocks_as_type_of_model blocks))
    (row_like_bottom names);
  run ~title:"Two types of kind value, for the two meets and the two joins"
    ~seed:9000 ~count:1000
    ~generate:(fun r ->
      let config : G.config = { well_formed = true; depth = 2 } in
      ( G.type_of_kind config r names.pool Value,
        G.type_of_kind config r names.pool Value ))
    ~prepare:(fun (m1, m2) ->
      ignore (to_compiler names m1 : TG.t);
      ignore (to_compiler names m2 : TG.t);
      m1, m2)
    ~shrink:(fun (m1, m2) ->
      List.map (fun m1 -> m1, m2) (G.shrink m1)
      @ List.map (fun m2 -> m1, m2) (G.shrink m2))
    ~print:(fun ppf (m1, m2) ->
      Format.fprintf ppf "@[<v>%a@ and@ %a@]" Model_type.print m1
        Model_type.print m2)
    (two_implementations names);
  let semantic_pairs ~title ~seed ~count kind =
    run ~title ~seed ~count
      ~generate:(fun r ->
        let config : G.config = { well_formed = true; depth = 2 } in
        let first = G.type_of_kind config r names.pool kind in
        let second = G.type_of_kind config r names.pool kind in
        { world_seed = Random.int r; first; second })
      ~prepare:(fun case ->
        ignore (to_compiler names case.first : TG.t);
        ignore (to_compiler names case.second : TG.t);
        case)
      ~shrink:shrink_semantic_case ~print:print_semantic_case
  in
  semantic_pairs
    ~title:
      "Two types of kind value and a world, for the meaning of meet and join"
    ~seed:40000 ~count:1500 MK.Value
    (semantic_comparisons names);
  semantic_pairs
    ~title:
      "Two types of naked immediates and a world, for the meaning of meet and \
       join"
    ~seed:50000 ~count:1500 (MK.Naked_number Kind_naked_immediate)
    (semantic_comparisons names);
  let other_kinds =
    List.filter
      (fun (kind : MK.flambda_kind) ->
        match kind with
        | Value | Naked_number Kind_naked_immediate -> false
        | Naked_number
            ( Kind_naked_float32 | Kind_naked_float | Kind_naked_int8
            | Kind_naked_int16 | Kind_naked_int32 | Kind_naked_int64
            | Kind_naked_nativeint | Kind_naked_vec128 | Kind_naked_vec256
            | Kind_naked_vec512 | Kind_naked_mask )
        | Region | Rec_info ->
          true)
      G.all_kinds
  in
  run
    ~title:
      "Two types of one of the other kinds and a world, for the meaning of \
       meet and join"
    ~seed:55000 ~count:600
    ~generate:(fun r ->
      let config : G.config = { well_formed = true; depth = 2 } in
      let kind = G.pick r other_kinds in
      let first = G.type_of_kind config r names.pool kind in
      let second = G.type_of_kind config r names.pool kind in
      { world_seed = Random.int r; first; second })
    ~prepare:(fun case ->
      ignore (to_compiler names case.first : TG.t);
      ignore (to_compiler names case.second : TG.t);
      case)
    ~shrink:shrink_semantic_case ~print:print_semantic_case
    (semantic_comparisons names);
  (* Chosen pairs, each in a few worlds. The first is T1-4: a closure whose own
     function has a known code, and a closure of which nothing is said of the
     code. *)
  let chosen =
    let slot =
      List.find
        (fun slot -> Stdlib.( = ) slot V.unnamed_function_slot)
        names.pool.function_slots
    in
    let closure function_types =
      MT.at_least_this_closure slot function_types [] [] MT.Alloc_mode.Heap
    in
    let code : MT.function_type =
      { function_type_code_id = List.hd names.pool.code_ids }
    in
    [closure [slot, MT.Or_unknown.Known code], closure []]
  in
  let next = ref 0 in
  run ~title:"Chosen pairs of types, for the meaning of meet and join"
    ~seed:60000
    ~count:(3 * List.length chosen)
    ~generate:(fun r ->
      let first, second = List.nth chosen (!next / 3) in
      incr next;
      { world_seed = Random.int r; first; second })
    ~prepare:(fun case -> case)
    ~shrink:(fun _ -> [])
    ~print:print_semantic_case
    (semantic_comparisons names);
  chosen_joins names (the_chosen_joins names);
  run_examples_of_joins names;
  run_examples_of_meets ();
  chosen_meets ();
  scenarios_in_the_model ();
  the_meet_of_t7_13_in_every_order ();
  the_line_of_t7_13_in_every_order ();
  the_lines_of_t7_14_and_t7_15 ();
  control_of_the_alias_kept ();
  let model_and_compiler ~title ~seed ~count ~equations:shape kind =
    run ~title ~seed ~count
      ~generate:(fun r ->
        let config : G.config = { well_formed = true; depth = 2 } in
        let aliases, equations =
          match shape with
          | `None | `Cases | `Alike | `Siblings | `Relations | `Closures ->
            false, []
          | `Types | `Alike_with_equations | `Siblings_with_equations
          | `Relations_with_equations | `Closures_with_equations ->
            false, generate_equations names r
          | `Aliases -> true, generate_aliases names r
        in
        (* With the aliases, one of the two types is often an alias too. *)
        let side () =
          if aliases && G.chance r ~percent:50
          then
            MT.alias_type_of Value (Var (G.pick r names.pool.value_variables))
          else G.type_of_kind config r names.pool kind
        in
        { equations;
          met =
            (match shape with
            | `Cases -> generate_meet_of_cases names r
            | `Alike | `Alike_with_equations -> generate_alike names r kind
            | `Siblings | `Siblings_with_equations ->
              let first, second, (_ : MT.flambda_type) =
                generate_siblings names r kind
              in
              first, second
            | `Relations | `Relations_with_equations ->
              generate_meet_of_relations names r
            | `Closures | `Closures_with_equations ->
              generate_meet_of_closures names r
            | `None | `Types | `Aliases -> side (), side ())
        })
      ~prepare:(fun ({ equations; met = m1, m2 } as input) ->
        List.iter (fun (_, t) -> ignore (to_compiler names t : TG.t)) equations;
        ignore (to_compiler names m1 : TG.t);
        ignore (to_compiler names m2 : TG.t);
        input)
      ~shrink:shrink_meet_in_an_environment ~print:print_meet_in_an_environment
      (model_against_compiler names)
  in
  model_and_compiler
    ~title:
      "Two types of kind value, for the meet of the model and the n-way meet \
       of the compiler"
    ~seed:90000 ~count:1000 ~equations:`None MK.Value;
  model_and_compiler
    ~title:
      "Two types of naked immediates, for the meet of the model and the n-way \
       meet of the compiler"
    ~seed:92000 ~count:500 ~equations:`None
    (MK.Naked_number Kind_naked_immediate);
  model_and_compiler
    ~title:
      "Equations on variables and two types of kind value, for the meet of the \
       model and the n-way meet of the compiler"
    ~seed:94000 ~count:1000 ~equations:`Types MK.Value;
  model_and_compiler
    ~title:
      "Equations on variables and two types of naked immediates, for the meet \
       of the model and the n-way meet of the compiler"
    ~seed:96000 ~count:500 ~equations:`Types
    (MK.Naked_number Kind_naked_immediate);
  model_and_compiler
    ~title:
      "Aliases between variables and symbols, and two types of kind value, for \
       the meet of the model and the n-way meet of the compiler"
    ~seed:98000 ~count:1000 ~equations:`Aliases MK.Value;
  model_and_compiler
    ~title:
      "Two variants whose cases hold a variable and blocks of a repeated \
       variable, for the meet of the model and the n-way meet of the compiler"
    ~seed:99000 ~count:500 ~equations:`Cases MK.Value;
  model_and_compiler
    ~title:
      "A type of kind value and a smaller type made from it, for the meet of \
       the model and the n-way meet of the compiler"
    ~seed:112000 ~count:1500 ~equations:`Alike MK.Value;
  model_and_compiler
    ~title:
      "Equations on variables, then a type of kind value and a smaller type \
       made from it, for the meet of the model and the n-way meet of the \
       compiler"
    ~seed:114000 ~count:1000 ~equations:`Alike_with_equations MK.Value;
  model_and_compiler
    ~title:
      "A type of naked immediates and a smaller type made from it, for the \
       meet of the model and the n-way meet of the compiler"
    ~seed:116000 ~count:500 ~equations:`Alike
    (MK.Naked_number Kind_naked_immediate);
  model_and_compiler
    ~title:
      "Two smaller types made from one type of kind value, for the meet of the \
       model and the n-way meet of the compiler"
    ~seed:122000 ~count:1000 ~equations:`Siblings MK.Value;
  model_and_compiler
    ~title:
      "Equations on variables, then two smaller types made from one type of \
       kind value, for the meet of the model and the n-way meet of the \
       compiler"
    ~seed:124000 ~count:1000 ~equations:`Siblings_with_equations MK.Value;
  model_and_compiler
    ~title:
      "Two variants with variables of relations, for the meet of the model and \
       the n-way meet of the compiler"
    ~seed:132000 ~count:1000 ~equations:`Relations MK.Value;
  model_and_compiler
    ~title:
      "Equations on variables, then two variants with variables of relations, \
       for the meet of the model and the n-way meet of the compiler"
    ~seed:134000 ~count:1000 ~equations:`Relations_with_equations MK.Value;
  model_and_compiler
    ~title:
      "Two closure types of one set of closures with two function slots, for \
       the meet of the model and the n-way meet of the compiler"
    ~seed:136000 ~count:1000 ~equations:`Closures MK.Value;
  model_and_compiler
    ~title:
      "Equations on variables, then two closure types of one set of closures \
       with two function slots, for the meet of the model and the n-way meet \
       of the compiler"
    ~seed:138000 ~count:1000 ~equations:`Closures_with_equations MK.Value;
  examples_of_joins_in_the_model names;
  chosen_joins_in_the_model names;
  control_of_the_joins names;
  control_of_the_analysis names;
  let prepare_join (input : join_at_a_continuation) =
    List.iter
      (fun { binds; learns; passes } ->
        let converts t = ignore (to_compiler names t : TG.t) in
        List.iter (fun (_, t) -> converts t) binds;
        List.iter (fun (_, t) -> converts t) learns;
        converts passes)
      input.at_uses;
    input
  in
  let model_join_and_compiler ~title ~seed ~count ~known ?(binds = false)
      ?(alike = false) ?(siblings = false) kind =
    let local = Converter.variable_to_model names.numbering names.local in
    run ~title ~seed ~count
      ~generate:(fun r ->
        let config : G.config = { well_formed = true; depth = 2 } in
        let shallow : G.config = { well_formed = true; depth = 1 } in
        (* The types passed, where they are to be alike: the two of
           [generate_alike] at the first two uses, and at a third use one
           further step of the shrinker from the first of the two, or that first
           type again where the shrinker gives none of its kind. The first of
           the two may already be the shrunk one. Nothing is generated for them
           otherwise, so that the other runs have the inputs that they had. *)
        let alike_types =
          if alike then Some (generate_alike names r kind) else None
        in
        let sibling_types =
          if siblings then Some (generate_siblings names r kind) else None
        in
        let passed i =
          match sibling_types, alike_types with
          | Some (first, second, third), _ ->
            if i = 0 then first else if i = 1 then second else third
          | None, None -> G.type_of_kind config r names.pool kind
          | None, Some (one, other) -> (
            if i = 0
            then one
            else if i = 1
            then other
            else
              match
                List.filter
                  (fun t -> Stdlib.( = ) (MT.kind t) kind)
                  (G.shrink one)
              with
              | [] -> one
              | _ :: _ as smaller -> G.pick r smaller)
        in
        { kind_of_parameter = kind;
          for_types_at_source = [];
          at_uses =
            List.init
              (2 + G.below r 2)
              (fun i ->
                let learns = if known then generate_equations names r else [] in
                (* A use may bind a variable of its own and pass it, or a block
                   that holds it. *)
                if binds && G.chance r ~percent:60
                then
                  let alias = MT.alias_type_of Value (Var local) in
                  { binds = [local, G.type_of_kind shallow r names.pool Value];
                    learns;
                    passes =
                      (if G.chance r ~percent:50
                       then alias
                       else
                         MT.immutable_block false (N.z_of_int 0)
                           (Scannable Value_only) Heap
                           [alias; G.type_of_kind shallow r names.pool Value])
                  }
                else { binds = []; learns; passes = passed i })
        })
      ~prepare:prepare_join ~shrink:shrink_join_at_a_continuation
      ~print:print_join_at_a_continuation
      (model_join_against_compiler names)
  in
  (* The same comparisons on inputs that a function makes whole. *)
  let model_join_and_compiler_on ~title ~seed ~count generate =
    run ~title ~seed ~count ~generate ~prepare:prepare_join
      ~shrink:shrink_join_at_a_continuation ~print:print_join_at_a_continuation
      (model_join_against_compiler names)
  in
  model_join_and_compiler
    ~title:
      "Two or three types of kind value, for the join of the model and the \
       n-way join of the compiler"
    ~seed:100000 ~count:1000 ~known:false MK.Value;
  model_join_and_compiler
    ~title:
      "Two or three types of naked immediates, for the join of the model and \
       the n-way join of the compiler"
    ~seed:102000 ~count:500 ~known:false (MK.Naked_number Kind_naked_immediate);
  model_join_and_compiler
    ~title:
      "Two or three uses that know equations on variables and pass a type of \
       kind value, for the join of the model and the n-way join of the \
       compiler"
    ~seed:104000 ~count:1000 ~known:true MK.Value;
  model_join_and_compiler
    ~title:
      "Two or three uses that know equations on variables and pass a type of \
       naked immediates, for the join of the model and the n-way join of the \
       compiler"
    ~seed:106000 ~count:500 ~known:true (MK.Naked_number Kind_naked_immediate);
  model_join_and_compiler
    ~title:
      "Two or three uses, of which some bind a variable and pass it or a block \
       that holds it, for the join of the model and the n-way join of the \
       compiler"
    ~seed:108000 ~count:1000 ~known:false ~binds:true MK.Value;
  model_join_and_compiler
    ~title:
      "The same with uses that know equations on variables, for the join of \
       the model and the n-way join of the compiler"
    ~seed:110000 ~count:1000 ~known:true ~binds:true MK.Value;
  model_join_and_compiler
    ~title:
      "Two or three uses that pass a type of kind value and smaller types made \
       from it, for the join of the model and the n-way join of the compiler"
    ~seed:118000 ~count:1500 ~known:false ~alike:true MK.Value;
  model_join_and_compiler
    ~title:
      "The same with uses that know equations on variables, for the join of \
       the model and the n-way join of the compiler"
    ~seed:120000 ~count:1000 ~known:true ~alike:true MK.Value;
  model_join_and_compiler
    ~title:
      "Two or three uses that each pass a smaller type made from one type of \
       kind value, for the join of the model and the n-way join of the \
       compiler"
    ~seed:128000 ~count:1000 ~known:false ~siblings:true MK.Value;
  model_join_and_compiler
    ~title:
      "The same with uses that know equations on variables, for the join of \
       the model and the n-way join of the compiler"
    ~seed:130000 ~count:1000 ~known:true ~siblings:true MK.Value;
  let closures_at_uses ~known r : join_at_a_continuation =
    let one, other = generate_meet_of_closures names r in
    let use passes =
      { binds = [];
        learns = (if known then generate_equations names r else []);
        passes
      }
    in
    let first = use one in
    let second = use other in
    { kind_of_parameter = Value;
      for_types_at_source = [];
      at_uses =
        (if G.chance r ~percent:30
         then
           let again, (_ : MT.flambda_type) =
             generate_meet_of_closures names r
           in
           [first; second; use again]
         else [first; second])
    }
  in
  model_join_and_compiler_on
    ~title:
      "Two or three uses that pass closure types of one set of closures with \
       two function slots, for the join of the model and the n-way join of the \
       compiler"
    ~seed:144000 ~count:1000
    (closures_at_uses ~known:false);
  model_join_and_compiler_on
    ~title:
      "The same with uses that know equations on variables, for the join of \
       the model and the n-way join of the compiler"
    ~seed:146000 ~count:1000
    (closures_at_uses ~known:true);
  model_join_and_compiler_on
    ~title:
      "Two or three uses, of which most bind a naked immediate that holds a \
       relation on a value and pass the value, a block that holds it or the \
       tagging of the naked immediate, for the join of the model and the n-way \
       join of the compiler"
    ~seed:140000 ~count:1000
    (generate_join_of_relations names ~known:false);
  model_join_and_compiler_on
    ~title:
      "The same with uses that know equations on variables, for the join of \
       the model and the n-way join of the compiler"
    ~seed:142000 ~count:1000
    (generate_join_of_relations names ~known:true);
  let with_itself ~title sharing =
    run ~title ~seed:70000 ~count:1500
      ~generate:(fun r ->
        G.type_of_kind { well_formed = true; depth = 3 } r names.pool MK.Value)
      ~prepare:(fun m ->
        ignore (to_compiler names m : TG.t);
        m)
      ~shrink:G.shrink ~print:Model_type.print
      (shared_and_not names sharing)
  in
  let with_a_common_part ~title sharing =
    run ~title ~seed:80000 ~count:1000
      ~generate:(fun r ->
        let part () =
          G.type_of_kind { well_formed = true; depth = 2 } r names.pool MK.Value
        in
        let common = part () in
        let left = part () in
        let right = part () in
        { common; left; right })
      ~prepare:(fun ({ common; left; right } as parts) ->
        ignore (to_compiler names common : TG.t);
        ignore (to_compiler names left : TG.t);
        ignore (to_compiler names right : TG.t);
        parts)
      ~shrink:(fun ({ common; left; right } as parts) ->
        List.map (fun common -> { parts with common }) (G.shrink common)
        @ List.map (fun left -> { parts with left }) (G.shrink left)
        @ List.map (fun right -> { parts with right }) (G.shrink right))
      ~print:(fun ppf { common; left; right } ->
        Format.fprintf ppf
          "@[<v>the common part@ %a@ the other part of the first@ %a@ the \
           other part of the second@ %a@]"
          Model_type.print common Model_type.print left Model_type.print right)
      (shared_parts names sharing)
  in
  with_itself ~title:"A type of kind value with itself, shared and not"
    Structural_sharing;
  with_a_common_part
    ~title:"Two types of kind value with a common part, shared and not"
    Structural_sharing;
  with_join_sharing Oxcaml_flags.Physical (fun () ->
      with_itself
        ~title:
          "A type of kind value with itself, shared and not, with the \
           compiler's default for the sharing of joins"
        Default_sharing;
      with_a_common_part
        ~title:
          "Two types of kind value with a common part, shared and not, with \
           the compiler's default for the sharing of joins"
        Default_sharing);
  joins_of_the_shape_of_t5_10 ();
  run
    ~title:
      "Programs of two joins of two uses, of which a use of the first may have \
       a join of its own, for the join of the model and the n-way join of the \
       compiler"
    ~seed:160000 ~count:1500
    ~generate:(fun r -> generate_joined_steps r)
    ~prepare:(fun steps -> steps)
    ~shrink:shrink_joined_steps ~print:print_joined_steps
    (generated_joins_after_joins ());
  control_of_the_names_to_avoid ();
  control_of_the_repaired_test ();
  run
    ~title:
      "Programs of two joins of two uses, in which a use may give a value two \
       variants whose meet makes variables, for the model and the compiler"
    ~seed:170000 ~count:1500
    ~generate:(generate_joined_steps ~meets_of_cases:true)
    ~prepare:(fun steps -> steps)
    ~shrink:shrink_joined_steps ~print:print_joined_steps
    (generated_joins_after_meets ());
  control_of_the_variables_of_the_meets ~seed:170000 ~count:1500;
  control_of_the_check_of_the_compiler ();
  let symbols_of_groups = symbols_for_static_groups names in
  run
    ~title:
      "Groups of one to three static constants that are not closures, for the \
       types that the model and Simplify give their symbols"
    ~seed:150000 ~count:2000
    ~generate:(generate_static_group names symbols_of_groups)
    ~prepare:(fun ({ before; bindings } as input) ->
      List.iter
        (fun ((_, simple) : M.variable * M.simple) ->
          ignore
            (Converter.simple_of_model names.numbering machine_width simple
              : Simple.t))
        before;
      List.iter
        (fun ((_, const) : M.symbol * M.static_const) ->
          ignore (static_const_of_model names const : Static_const.t))
        bindings;
      input)
    ~shrink:shrink_static_group ~print:print_static_group
    (static_groups names symbols_of_groups);
  control_of_the_static_groups names symbols_of_groups;
  Format.printf "@.%s@."
    (if !failures = 0
     then "No disagreement."
     else Printf.sprintf "%d DISAGREEMENTS." !failures);
  if !failures > 0 then exit 1
