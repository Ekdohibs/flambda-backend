open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaPrimitiveSpec
open FlambdaValueSpec
open List

val value_of_subkind : static_env -> subkind -> value -> bool

val block_of_shapes :
  static_env -> constructor_shapes -> coq_Z -> value list -> bool

val fields_of_subkinds : static_env -> field_subkinds -> value list -> bool

val value_at : static_env -> param_subkind -> value -> bool

val values_of_subkinds : static_env -> value list -> param_subkind list -> bool
