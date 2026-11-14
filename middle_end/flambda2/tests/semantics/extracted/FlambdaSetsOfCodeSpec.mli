open BinPos
open Datatypes
open FlambdaKnownSlotsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaValueSpec
open List

type set_made = (function_slot * code_id) list * (value_slot * simple) list

val named_sets_made : named -> set_made list

val static_const_sets_made : static_const -> set_made list

val sets_made : expr -> set_made list

val handler_sets_made : continuation_handler -> set_made list

val binding_sets_made : static_binding -> set_made list

val code_sets_made : code -> set_made list

type set_fact =
  | Code_at of function_slot * code_id
  | Slot_holds of value_slot * simple

val has_code_at :
  code_id -> function_slot -> (function_slot * code_id) list -> bool

val set_made_has : set_fact -> set_made -> bool

val sets_made_have :
  code_id -> function_slot -> set_fact -> set_made list -> bool
