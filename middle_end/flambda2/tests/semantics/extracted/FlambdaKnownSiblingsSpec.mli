open Datatypes
open FlambdaModel
open FlambdaNormaliseSpec

val inline_known_in_group :
  code_id list ->
  static_binding list ->
  static_binding list ->
  static_binding list

val inline_known_sibling_calls : code_id list -> expr -> expr
