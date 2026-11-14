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

val kind_of_value : value -> flambda_kind

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

val empty_static_env : static_env

type cell =
  | Cell_block of cell_mutability * coq_Z * value list
  | Cell_array of array_kind * value list

type heap = (location * cell) list

val empty_heap : heap

type read_extent =
  | Read_field of nat
  | Read_whole

type read_record = (location * read_extent) list

val no_reads : read_record

val closure_of :
  static_env -> value -> (code_id * (value_slot * value) list) option

val unnamed_function_slot : function_slot

val closure_in_set :
  static_env ->
  value ->
  ((function_slot * (function_slot * code_id) list) * (value_slot * value) list)
  option

val project_value_slot :
  static_env -> function_slot -> value_slot -> value -> value option

val project_function_slot :
  static_env -> function_slot -> function_slot -> value -> value option

val block_of : static_env -> value -> (coq_Z * value list) option

val boxable_number_of_const : const -> boxable_number option

val boxable_contents : const -> const option

val boxed_number_of : static_env -> value -> const option

val string_of : static_env -> value -> coq_Z list option

val array_of : static_env -> value -> value list option

val partial_application_of : value -> ((value * value list) * arity) option

val is_allocated : static_env -> value -> bool

val imported_of : static_env -> value -> (symbol * import_class) option

val is_imported_pointer : static_env -> value -> bool

val identity_of : static_env -> value -> symbol option
