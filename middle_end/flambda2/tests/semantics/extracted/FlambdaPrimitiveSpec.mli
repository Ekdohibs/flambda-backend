open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaValueSpec
open List
open ListDef
open OxCamlOption

val in_range : coq_Z -> coq_Z -> bool

val wrap : coq_Z -> coq_Z -> coq_Z

val unsigned : coq_Z -> coq_Z -> coq_Z

val reading : coq_Z -> signedness -> coq_Z -> coq_Z

val int_width : standard_int -> coq_Z

val const_of_int : standard_int -> coq_Z -> const

val int_of_value : standard_int -> value -> coq_Z option

val is_tagged_immediate : value -> bool

val eval_int_arith :
  coq_Z -> binary_int_arith_op -> coq_Z -> coq_Z -> coq_Z option

val little_endian : coq_Z list -> coq_Z

val bytes_of : nat -> coq_Z -> coq_Z list

val swap_bytes : nat -> coq_Z -> coq_Z

val eval_unary_int_arith : standard_int -> unary_int_arith_op -> coq_Z -> coq_Z

val shift_bound : standard_int -> coq_Z

val eval_int_shift : coq_Z -> int_shift_op -> coq_Z -> coq_Z -> coq_Z

val eval_int_comparison : coq_Z -> int_comparison -> coq_Z -> coq_Z -> bool

val eval_int_comp : coq_Z -> int_comparison_behaviour -> coq_Z -> coq_Z -> coq_Z

val array_kind_of_load_kind : array_load_kind -> array_kind

val array_element_ok : array_kind -> value -> bool

val array_made : array_kind -> value list -> value list option

val array_length_of : value list -> value option

val array_index : value list -> value -> nat option

val array_loaded : array_load_kind -> value list -> value -> value option

val bound_if_symbol : static_env -> value -> bool

val naked_bool : bool -> value

val eval_is_int : static_env -> bool -> value -> bool option

val eval_is_null : static_env -> value -> bool option

val is_number_of_kind : boxable_number -> const -> bool

val known_is : coq_Z option -> coq_Z -> bool

val eval_block_load :
  static_env -> block_access_kind -> coq_Z -> value -> value option

val eval_reinterpret_64_bit_word :
  reinterpret_64_bit_word -> value -> value option

val eval_unary_primitive :
  static_env -> unary_primitive -> value -> value option

val byte_width_of_string_accessor_width : string_accessor_width -> nat

val eval_string_load :
  static_env -> string_accessor_width -> value -> value -> value option

val eval_phys_equal : static_env -> value -> value -> bool option

val eval_binary_primitive :
  static_env -> binary_primitive -> value -> value -> value option

val eval_variadic_primitive :
  static_env -> variadic_primitive -> value list -> value option

val eval_prim : static_env -> prim -> value list -> value option

val tag_in_range : coq_Z -> bool

val byte_in_range : coq_Z -> bool

val ocaml_int_in_range : coq_Z -> bool

val const_in_range : const -> bool

val known_in_range : (coq_Z -> bool) -> coq_Z option -> bool

val unary_primitive_in_range : unary_primitive -> bool

val binary_primitive_in_range : binary_primitive -> bool

val variadic_primitive_in_range : variadic_primitive -> bool

val access_in_range : block_access_kind -> coq_Z -> bool

val heap_primitive_in_range : heap_primitive -> bool

val prim_in_range : prim -> bool

val naked_immediate_kind : flambda_kind

val arg_kind_of_unary_primitive : unary_primitive -> flambda_kind

val result_kind_of_unary_primitive : unary_primitive -> flambda_kind

val args_kind_of_binary_primitive :
  binary_primitive -> flambda_kind * flambda_kind

val result_kind_of_binary_primitive : binary_primitive -> flambda_kind

val args_kind_of_variadic_primitive : variadic_primitive -> flambda_kind

val result_kind_of_variadic_primitive : variadic_primitive -> flambda_kind

val result_kind_of_heap_primitive : heap_primitive -> flambda_kind

val result_kind_of_region_primitive : region_primitive -> flambda_kind

val prim_result_kind : prim -> flambda_kind

val prim_args_well_kinded : prim -> flambda_kind list -> bool

val prim_eq_dec : prim -> prim -> bool

val prim_eqb : prim -> prim -> bool
