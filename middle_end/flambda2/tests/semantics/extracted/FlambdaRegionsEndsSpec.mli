open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaRegionsCheckSpec
open FlambdaRegionsSpec
open FlambdaValueSpec
open List
open ListDef
open Nat

type ends_options = { eo_own_end : bool }

type ends_cont =
  | Ends_handler of variable list * flambda_kind list * handler_entry * bool
  | Ends_unreachable
  | Ends_return
  | Ends_raise

type ends_state =
  { es_open : variable list;
    es_tries : variable list;
    es_regions : variable list;
    es_stale : bool;
    es_conts : (continuation * ends_cont) list
  }

val regions_above : variable -> variable list -> variable list

val variables_eqb : variable list -> variable list -> bool

val oldest_of : variable list -> variable list -> bool

val ends_region_args :
  variable list ->
  variable list ->
  flambda_kind list ->
  simple list option ->
  (variable * variable) list option

val entry_at_jump :
  variable list ->
  variable list ->
  flambda_kind list ->
  handler_entry ->
  simple list option ->
  variable list option

val ends_jump : ends_state -> continuation -> simple list option -> bool

val ends_raise_to : ends_state -> continuation -> simple list -> bool

val ends_first : expr -> bool

val enters_stale : ends_options -> continuation_handler -> bool

val ends_cont_of :
  certificate ->
  ends_options ->
  continuation ->
  continuation_handler ->
  ends_cont

val ends_with_conts :
  ends_state -> (continuation * ends_cont) list -> ends_state

val ends_handler_state :
  ends_state ->
  variable list ->
  flambda_kind list ->
  handler_entry ->
  bool ->
  ends_state

val ends_code_state : continuation -> continuation -> ends_state

val ends_expr : certificate -> ends_options -> ends_state -> expr -> bool

val ends_code : certificate -> ends_options -> code -> bool

val ends_check :
  certificate ->
  ends_options ->
  continuation ->
  continuation ->
  static_env ->
  expr ->
  bool
