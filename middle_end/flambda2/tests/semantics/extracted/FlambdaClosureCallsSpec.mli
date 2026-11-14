open BinPos
open FlambdaMapSpec
open FlambdaModel
open ListDef

type made_closure = function_slot * (function_slot * code_id) list

type made_closures = made_closure finite_map

val closure_made : made_closures -> named -> made_closure option

val made_under_let : made_closures -> variable -> named -> made_closures

val unbind_params : variable list -> made_closures -> made_closures

val made_callee_code : made_closures -> simple -> code_id option

val call_made_closure : made_closures -> apply_expr -> apply_expr

val call_closures_knowing : made_closures -> expr -> expr

val call_closures_in_handler :
  made_closures -> continuation_handler -> continuation_handler

val call_made_closures : expr -> expr
