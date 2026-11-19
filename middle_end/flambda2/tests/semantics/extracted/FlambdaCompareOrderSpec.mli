open BinInt
open BinPos
open Bool
open Datatypes
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaModel
open FlambdaSubkindOrderSpec
open ListDef

val handler_metadata_says_at_least :
  handler_metadata -> handler_metadata -> bool

val code_metadata_says_at_least : code_metadata -> code_metadata -> bool

val expr_says_at_least : name_map -> name_map -> expr -> expr -> bool

val handler_says_at_least :
  name_map -> name_map -> continuation_handler -> continuation_handler -> bool

val binding_says_at_least : name_map -> static_binding -> static_binding -> bool

val code_says_at_least : code -> code -> bool
