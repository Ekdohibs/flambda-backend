open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaSetsOfCodeSpec
open FlambdaValueSpec
open List
open ListDef

type closure_seen = function_slot * (function_slot * code_id) list

val seen_code : closure_seen -> code_id option

type closure_sight = closure_seen finite_map

val sight_made : closure_sight -> named -> closure_seen option

val sight_under_let : closure_sight -> variable -> named -> closure_sight

val sight_without : variable list -> closure_sight -> closure_sight

val simple_seen_code : closure_sight -> simple -> code_id option

type set_seen =
  (function_slot * code_id) list * (value_slot * code_id option) list

val slots_seen :
  closure_sight ->
  (value_slot * simple) list ->
  (value_slot * code_id option) list

val named_sets_seen : closure_sight -> named -> set_seen list

val static_const_sets_seen : closure_sight -> static_const -> set_seen list

val sets_seen : closure_sight -> expr -> set_seen list

val handler_sets_seen : closure_sight -> continuation_handler -> set_seen list

val binding_sets_seen : closure_sight -> static_binding -> set_seen list

val code_sets_seen : code -> set_seen list

val set_seen_gives : value_slot -> code_id -> set_seen -> bool

val sets_seen_give :
  code_id -> function_slot -> value_slot -> code_id -> set_seen list -> bool
