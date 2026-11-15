open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaTypesSpec
open List
open ListDef

val var_in_type : variable -> flambda_type -> bool

val drop_var : variable -> typing_env -> typing_env

val drop_vars : variable list -> typing_env -> typing_env

val canonical_value_slots :
  typing_env -> (value_slot * simple) list -> (value_slot * simple) list

val find_closure :
  typing_env -> variable -> (code_id * (value_slot * simple) list) option

val closure_of_simple :
  typing_env -> simple -> (code_id * (value_slot * simple) list) option

val canonical_project : typing_env -> value_slot -> simple -> named

val canonical_named : typing_env -> named -> named

val canonical_static_const : typing_env -> static_const -> static_const

val canonical_static_binding : typing_env -> static_binding -> static_binding

val canonical_apply_cont : typing_env -> apply_cont -> apply_cont

val canonical_call_kind : typing_env -> simple -> call_kind -> call_kind

val canonical_apply : typing_env -> apply_expr -> apply_expr

val typing_env_under_let : typing_env -> variable -> named -> typing_env

val follow_aliases : typing_env -> expr -> expr

val follow_aliases_in_handler :
  typing_env -> continuation_handler -> continuation_handler
