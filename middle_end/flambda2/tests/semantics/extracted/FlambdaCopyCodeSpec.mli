open Datatypes
open FlambdaMapSpec
open FlambdaModel
open List
open ListDef

val copy_of : (code_id * code_id) list -> code_id -> code_id

val copy_decls :
  (code_id * code_id) list ->
  (function_slot * code_id) list ->
  (function_slot * code_id) list

val copy_static_const : (code_id * code_id) list -> static_const -> static_const

val with_call_kind : call_kind -> apply_expr -> apply_expr

val names_copied : (code_id * code_id) list -> code_id list -> bool

val copy_call :
  (code_id * code_id) list -> symbol list -> apply_expr -> apply_expr

val copy_expr : (code_id * code_id) list -> symbol list -> expr -> expr

val copy_handler :
  (code_id * code_id) list ->
  symbol list ->
  continuation_handler ->
  continuation_handler

val copy_binding :
  (code_id * code_id) list ->
  symbol list ->
  static_binding ->
  static_binding list

val copy_code : (code_id * code_id) list -> symbol list -> code -> code

val binds_no_code : code_id list -> expr -> bool

val handler_binds_no_code : code_id list -> continuation_handler -> bool

val binding_binds_no_code : code_id list -> static_binding -> bool

val code_binds_no_code : code_id list -> code -> bool

val copy_ok : (code_id * code_id) list -> expr -> bool

val copy_static_code :
  (code_id * code_id) list -> symbol list -> expr -> expr option

val with_copies : ((code_id * code_id) list * symbol list) list -> expr -> expr
