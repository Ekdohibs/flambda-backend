open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open ListDef
open PeanoNat

val full_call_of_b : apply_expr -> code -> bool

val demote_full_call : (code_id * code) list -> apply_expr -> apply_expr

val demote_full_calls : (code_id * code) list -> expr -> expr

val demote_full_calls_in_handler :
  (code_id * code) list -> continuation_handler -> continuation_handler

val demote_full_calls_in_code : (code_id * code) list -> code -> code

val demote_full_in_stub_binding :
  (code_id * code) list -> static_binding -> static_binding

val demote_full_in_stubs : (code_id * code) list -> expr -> expr

val demote_full_in_stubs_of_handler :
  (code_id * code) list -> continuation_handler -> continuation_handler

val demote_full_stub_calls : expr -> expr
