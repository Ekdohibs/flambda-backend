open BinNums
open BinPos
open Datatypes
open FlambdaHeapFreeSpec
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List

val closed_named : named -> bool

val lifted_out :
  (path * lift_to) list ->
  path ->
  named ->
  (static_binding list * symbol) option

type lets_in_code = (code_id * (path * lift_to) list) list

val lets_of : lets_in_code -> code_id -> (path * lift_to) list

val lifted_out_bindings :
  (path * lift_to) list -> path -> expr -> static_binding list

val handler_lifted_out_bindings :
  (path * lift_to) list -> path -> continuation_handler -> static_binding list

val same_binding : static_binding -> static_binding -> bool

val once_each :
  static_binding list -> static_binding list -> static_binding list

val lift_from_code :
  lets_in_code -> (path * lift_to) list -> path -> expr -> expr

val handler_lift_from_code :
  lets_in_code ->
  (path * lift_to) list ->
  path ->
  continuation_handler ->
  continuation_handler

val binding_lift_from_code :
  lets_in_code -> static_binding -> static_binding list

val lift_from_code_unit : unit_interface -> lets_in_code -> expr -> expr option
