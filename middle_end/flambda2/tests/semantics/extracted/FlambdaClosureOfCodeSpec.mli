open Datatypes
open FlambdaKnownSymbolsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaSetsOfCodeSpec
open FlambdaStaticsInScopeSpec
open FlambdaWellFormedSpec
open List
open ListDef

val set_with_code : set_made list -> code_id -> function_slot -> set_made option

val sibling_code :
  set_made list -> code_id -> function_slot -> function_slot -> code_id option

val slot_held :
  set_made list -> code_id -> function_slot -> value_slot -> simple option

type closure_codes = code_id finite_map

val closure_code_made :
  set_made list -> closure_codes -> named -> code_id option

val codes_under_let :
  set_made list -> closure_codes -> variable -> named -> closure_codes

val read_code_slot :
  set_made list -> symbol list -> closure_codes -> named -> named

val call_code_closure : closure_codes -> apply_expr -> apply_expr

val forget_code_vars : variable list -> closure_codes -> closure_codes

val know_in_body : set_made list -> symbol list -> closure_codes -> expr -> expr

val know_in_body_handler :
  set_made list ->
  symbol list ->
  closure_codes ->
  continuation_handler ->
  continuation_handler

val know_in_code : set_made list -> symbol list -> code_id -> code -> code

val know_code_in_binding :
  set_made list -> symbol list -> static_binding -> static_binding

val know_code_closures : set_made list -> symbol list -> expr -> expr

val know_code_in_handler :
  set_made list -> symbol list -> continuation_handler -> continuation_handler

val groups_whole : expr -> bool

val handler_groups_whole : continuation_handler -> bool

val binding_groups_whole : static_binding -> bool

val code_groups_whole : code -> bool

val know_own_closures : expr -> expr
