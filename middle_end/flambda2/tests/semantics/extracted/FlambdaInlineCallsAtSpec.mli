open BinNums
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open ListDef

type calls_in_code = (code_id * path list) list

val calls_of : calls_in_code -> code_id -> path list

val inline_calls_at_paths :
  calls_in_code -> path list -> (code_id * code) list -> path -> expr -> expr

val handler_inline_calls_at_paths :
  calls_in_code ->
  path list ->
  (code_id * code) list ->
  path ->
  continuation_handler ->
  continuation_handler

val binding_inline_calls_at_paths :
  calls_in_code -> (code_id * code) list -> static_binding -> static_binding

val inline_calls_at : path list -> calls_in_code -> expr -> expr
