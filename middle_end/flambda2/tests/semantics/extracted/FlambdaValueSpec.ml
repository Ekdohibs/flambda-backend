open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open OxCamlOption

type location = positive

type value =
  | Value_const of const
  | Value_closure of code_id * (value_slot * value) list
  | Value_closure_of_set of
      function_slot * (function_slot * code_id) list * (value_slot * value) list
  | Value_symbol of symbol
  | Value_block of coq_Z * value list
  | Value_immutable_array of value list
  | Value_boxed_number of const
  | Value_location of location
  | Value_partial_application of value * value list * arity

(** val kind_of_value : value -> flambda_kind **)

let kind_of_value = function Value_const c -> kind_of_const c | _ -> Value

type import_class =
  | Import_identity
  | Import_data
  | Import_unknown

type statically_allocated_value =
  | Static_closure of code_id * (value_slot * value) list
  | Static_closure_of_set of
      function_slot
      * (function_slot * symbol) list
      * (function_slot * code_id) list
      * (value_slot * value) list
  | Static_block of coq_Z * value list
  | Static_boxed_number of value
  | Static_string of coq_Z list
  | Static_immutable_array of value list
  | Static_imported of import_class

type static_env =
  { static_code : (code_id * code) list;
    static_values : (symbol * statically_allocated_value) list
  }

(** val empty_static_env : static_env **)

let empty_static_env = { static_code = []; static_values = [] }

type cell =
  | Cell_block of cell_mutability * coq_Z * value list
  | Cell_array of array_kind * value list

type heap = (location * cell) list

(** val empty_heap : heap **)

let empty_heap = []

type read_extent =
  | Read_field of nat
  | Read_whole

type read_record = (location * read_extent) list

(** val no_reads : read_record **)

let no_reads = []

(** val closure_of : static_env -> value -> (code_id * (value_slot * value)
    list) option **)

let closure_of statics = function
  | Value_closure (code0, value_slots) -> Some (code0, value_slots)
  | Value_closure_of_set (slot, function_decls, value_slots) -> (
    match lookup slot function_decls with
    | Some code0 -> Some (code0, value_slots)
    | None -> None)
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_closure (code0, value_slots) -> Some (code0, value_slots)
      | Static_closure_of_set (slot, _, function_decls, value_slots) -> (
        match lookup slot function_decls with
        | Some code0 -> Some (code0, value_slots)
        | None -> None)
      | _ -> None)
    | None -> None)
  | _ -> None

(** val unnamed_function_slot : function_slot **)

let unnamed_function_slot = Coq_xH

(** val closure_in_set : static_env -> value -> ((function_slot * (function_slot
    * code_id) list) * (value_slot * value) list) option **)

let closure_in_set statics = function
  | Value_closure (code0, value_slots) ->
    Some
      ( (unnamed_function_slot, (unnamed_function_slot, code0) :: []),
        value_slots )
  | Value_closure_of_set (slot, function_decls, value_slots) -> (
    match lookup slot function_decls with
    | Some _ -> Some ((slot, function_decls), value_slots)
    | None -> None)
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_closure (code0, value_slots) ->
        Some
          ( (unnamed_function_slot, (unnamed_function_slot, code0) :: []),
            value_slots )
      | Static_closure_of_set (slot, _, function_decls, value_slots) -> (
        match lookup slot function_decls with
        | Some _ -> Some ((slot, function_decls), value_slots)
        | None -> None)
      | _ -> None)
    | None -> None)
  | _ -> None

(** val project_value_slot : static_env -> function_slot -> value_slot -> value
    -> value option **)

let project_value_slot statics project_from slot v =
  match closure_in_set statics v with
  | Some p ->
    let p0, value_slots = p in
    let at_slot, _ = p0 in
    if Pos.eqb at_slot project_from then lookup slot value_slots else None
  | None -> None

(** val project_function_slot : static_env -> function_slot -> function_slot ->
    value -> value option **)

let project_function_slot statics move_from move_to v =
  let unnamed =
    if Pos.eqb move_from unnamed_function_slot
    then if Pos.eqb move_to unnamed_function_slot then Some v else None
    else None
  in
  match v with
  | Value_closure (_, _) -> unnamed
  | Value_closure_of_set (slot, function_decls, value_slots) ->
    if Pos.eqb slot move_from
    then
      match lookup slot function_decls with
      | Some _ -> (
        match lookup move_to function_decls with
        | Some _ ->
          Some (Value_closure_of_set (move_to, function_decls, value_slots))
        | None -> None)
      | None -> None
    else None
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_closure (_, _) -> unnamed
      | Static_closure_of_set (slot, closure_symbols, function_decls, _) ->
        if Pos.eqb slot move_from
        then
          match lookup slot function_decls with
          | Some _ -> (
            match lookup move_to function_decls with
            | Some _ -> (
              match lookup move_to closure_symbols with
              | Some s' -> Some (Value_symbol s')
              | None -> None)
            | None -> None)
          | None -> None
        else None
      | _ -> None)
    | None -> None)
  | _ -> None

(** val block_of : static_env -> value -> (coq_Z * value list) option **)

let block_of statics = function
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_block (tag, fields) -> Some (tag, fields)
      | _ -> None)
    | None -> None)
  | Value_block (tag, fields) -> Some (tag, fields)
  | _ -> None

(** val boxable_number_of_const : const -> boxable_number option **)

let boxable_number_of_const c = boxable_number_of_kind (kind_of_const c)

(** val boxable_contents : const -> const option **)

let boxable_contents c =
  match boxable_number_of_const c with Some _ -> Some c | None -> None

(** val boxed_number_of : static_env -> value -> const option **)

let boxed_number_of statics = function
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_boxed_number contents -> (
        match contents with Value_const c -> boxable_contents c | _ -> None)
      | _ -> None)
    | None -> None)
  | Value_boxed_number c -> boxable_contents c
  | _ -> None

(** val string_of : static_env -> value -> coq_Z list option **)

let string_of statics = function
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with Static_string bytes -> Some bytes | _ -> None)
    | None -> None)
  | _ -> None

(** val array_of : static_env -> value -> value list option **)

let array_of statics = function
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> (
      match s0 with
      | Static_immutable_array elements -> Some elements
      | _ -> None)
    | None -> None)
  | Value_immutable_array elements -> Some elements
  | _ -> None

(** val partial_application_of : value -> ((value * value list) * arity) option
    **)

let partial_application_of = function
  | Value_partial_application (callee, args, args_arity) ->
    Some ((callee, args), args_arity)
  | _ -> None

(** val is_allocated : static_env -> value -> bool **)

let is_allocated statics v =
  ((((is_some (closure_of statics v) || is_some (block_of statics v))
    || is_some (boxed_number_of statics v))
   || is_some (string_of statics v))
  || is_some (array_of statics v))
  || is_some (partial_application_of v)

(** val imported_of : static_env -> value -> (symbol * import_class) option **)

let imported_of statics = function
  | Value_symbol s -> (
    match lookup s statics.static_values with
    | Some s0 -> ( match s0 with Static_imported c -> Some (s, c) | _ -> None)
    | None -> None)
  | _ -> None

(** val is_imported_pointer : static_env -> value -> bool **)

let is_imported_pointer statics v =
  match imported_of statics v with
  | Some p -> (
    let _, i = p in
    match i with Import_unknown -> false | _ -> true)
  | None -> false

(** val identity_of : static_env -> value -> symbol option **)

let identity_of statics v =
  match imported_of statics v with
  | Some p -> (
    let s, i = p in
    match i with Import_identity -> Some s | _ -> None)
  | None -> None
