open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaValueSpec
open List
open ListDef
open OxCamlOption

val fresh_location : heap -> location

val load_field :
  block_access_kind -> coq_Z -> coq_Z -> value list -> value option

val store_field :
  block_access_kind ->
  coq_Z ->
  value ->
  coq_Z ->
  value list ->
  value list option

val is_location : value -> bool

val eval_phys_equal_location : static_env -> value -> value -> bool option

val unit_value : value

val exec_heap_primitive :
  static_env -> heap -> heap_primitive -> value list -> (value * heap) option

val load_allowed : mutability -> cell_mutability -> bool

val exec_unary_on_cell : unary_primitive -> location -> cell -> value option

val exec_binary_on_cell :
  binary_primitive -> location -> cell -> value -> value option

val argument_bound : static_env -> heap -> value -> bool

val exec_bound_prim :
  static_env -> heap -> prim -> value list -> (value * heap) option

val exec_prim :
  static_env -> heap -> prim -> value list -> (value * heap) option

val named_changes_heap : named -> bool

val locations_of_value : value -> location list

val values_of_static : statically_allocated_value -> value list
