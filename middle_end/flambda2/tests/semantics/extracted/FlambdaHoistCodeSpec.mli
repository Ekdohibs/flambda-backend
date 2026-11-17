open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List

val is_code : static_binding -> bool

val code_bindings : expr -> static_binding list

val handler_code_bindings : continuation_handler -> static_binding list

val without_code : expr -> expr

val handler_without_code : continuation_handler -> continuation_handler

val decls_bound : (code_id -> bool) -> (function_slot * code_id) list -> bool

val named_closures_bound : (code_id -> bool) -> named -> bool

val const_closures_bound : (code_id -> bool) -> static_const -> bool

val bound_with : code_id list -> (code_id -> bool) -> code_id -> bool

val closures_in_scope : (code_id -> bool) -> expr -> bool

val handler_closures_in_scope :
  (code_id -> bool) -> continuation_handler -> bool

val binding_closures_in_scope : (code_id -> bool) -> static_binding -> bool

val code_closures_in_scope : (code_id -> bool) -> code -> bool

val statics_at_unit_toplevel_of_unit : expr -> bool

val hoist_code : expr -> expr option
