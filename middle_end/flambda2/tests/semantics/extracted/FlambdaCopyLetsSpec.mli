open BinNums
open Datatypes
open FlambdaCopyCodeSpec
open FlambdaHeapFreeSpec
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open List
open ListDef

val copy_named : (code_id * code_id) list -> named -> named

val copy_lets_expr :
  (code_id * code_id) list -> path list -> path -> expr -> expr

val copy_lets_handler :
  (code_id * code_id) list ->
  path list ->
  path ->
  continuation_handler ->
  continuation_handler

val copy_lets_binding :
  (code_id * code_id) list ->
  (code_id * path list) list ->
  static_binding ->
  static_binding

val copy_lets_in_code :
  (code_id * code_id) list -> (code_id * path list) list -> expr -> expr

val copy_lets_in_code_handler :
  (code_id * code_id) list ->
  (code_id * path list) list ->
  continuation_handler ->
  continuation_handler

type let_copy =
  ((code_id * code_id) list * symbol list)
  * (path list * (code_id * path list) list)

val copies_apart : (code_id * code_id) list -> bool

val copy_let_code : let_copy -> expr -> expr

val with_let_copies : let_copy list -> expr -> expr
