open BinPos
open Datatypes
open FlambdaClosureOfCodeSpec
open FlambdaKnownSlotsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSetsOfCodeSpec
open FlambdaSlotsOfCodeSpec
open FlambdaValueSpec
open List
open ListDef

type made_seen = (value_slot * simple option) list

type made_sight = made_seen finite_map

val slots_made : named -> made_seen option

val forget_simple : variable -> simple option -> simple option

val forget_in_made : variable -> made_seen -> made_seen

val made_without : variable -> made_sight -> made_sight

val made_without_all : variable list -> made_sight -> made_sight

val made_under_let : made_sight -> variable -> named -> made_sight

type slot_through_seen = simple * made_seen option

val simple_made_seen : made_sight -> simple -> made_seen option

type set_through =
  (function_slot * code_id) list * (value_slot * slot_through_seen) list

val slots_through :
  made_sight ->
  (value_slot * simple) list ->
  (value_slot * slot_through_seen) list

val named_sets_through : made_sight -> named -> set_through list

val static_const_sets_through : made_sight -> static_const -> set_through list

val sets_through : made_sight -> expr -> set_through list

val handler_sets_through :
  made_sight -> continuation_handler -> set_through list

val binding_sets_through : made_sight -> static_binding -> set_through list

val code_sets_through : code -> set_through list

val set_through_gives :
  value_slot -> value_slot -> value_slot -> set_through -> bool

val sets_through_give :
  code_id ->
  function_slot ->
  value_slot ->
  value_slot ->
  value_slot ->
  set_through list ->
  bool

val slot_through :
  set_through list ->
  code_id ->
  function_slot ->
  value_slot ->
  value_slot ->
  value_slot option

type projections = ((variable * function_slot) * value_slot) finite_map

val through_without : variable -> projections -> projections

val through_without_all : variable list -> projections -> projections

val through_under_let : projections -> variable -> named -> projections

val read_through :
  set_through list -> closure_codes -> projections -> named -> named

val know_through_in_body :
  set_made list ->
  set_seen list ->
  set_through list ->
  closure_codes ->
  projections ->
  expr ->
  expr

val know_through_in_handler :
  set_made list ->
  set_seen list ->
  set_through list ->
  closure_codes ->
  projections ->
  continuation_handler ->
  continuation_handler

val know_through_in_code :
  set_made list -> set_seen list -> set_through list -> code_id -> code -> code

val know_through_in_binding :
  set_made list ->
  set_seen list ->
  set_through list ->
  static_binding ->
  static_binding

val know_code_through :
  set_made list -> set_seen list -> set_through list -> expr -> expr

val know_code_through_in_handler :
  set_made list ->
  set_seen list ->
  set_through list ->
  continuation_handler ->
  continuation_handler

val read_slots_through : expr -> expr

val know_own_closures_through : expr -> expr
