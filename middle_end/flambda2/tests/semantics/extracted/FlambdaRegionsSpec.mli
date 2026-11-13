open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaSemanticsSpec
open FlambdaValueSpec
open List
open ListDef
open Nat
open OxCamlOption

type mark = symbol

type mark_kind =
  | Plain_mark
  | Try_mark

type frame =
  { frame_mark : mark;
    frame_kind : mark_kind;
    frame_locals : (symbol * statically_allocated_value) list
  }

type region_counts =
  { count_other_region : nat;
    count_cut_by_end_region : nat;
    count_cut_by_end_try_region : nat;
    count_ends : nat;
    count_local_allocations : nat;
    count_no_frame : nat;
    count_local_uses : nat
  }

val no_counts : region_counts

type region_state =
  { rs_state : state;
    rs_frames : frame list;
    rs_base : symbol;
    rs_next : symbol;
    rs_counts : region_counts
  }

val live_locals : frame list -> (symbol * statically_allocated_value) list

val live_statics : region_state -> static_env

val drawn : region_state -> symbol -> bool

val value_usable : region_state -> value -> bool

val value_local : region_state -> value -> bool

val simple_usable : region_state -> simple -> bool

val simple_local : region_state -> simple -> bool

val simples_of_named : named -> simple list

val simples_of_static_const : static_const -> simple list

val simples_of_binding : static_binding -> simple list

val evaluated_simples : expr -> simple list

val is_allocation : named -> bool

val number_slots :
  (function_slot * code_id) list -> symbol -> (function_slot * symbol) list

val localise :
  symbol ->
  value ->
  (symbol * (symbol * statically_allocated_value) list) option

val advance : symbol -> nat -> symbol

val on_stack : mark -> frame list -> bool

val cut : mark -> frame list -> (frame list * frame list) option

val is_plain : frame -> bool

val newest_plain : frame list -> mark option

type marker =
  | No_marker
  | Marker of variable * simple option * expr
  | Bad_marker

val marker_of : variable -> expr -> marker

type region_fault =
  | Dead_value_used
  | Heap_holds_local
  | Region_not_open
  | Not_of_the_conversion
  | No_plain_step

type region_step_result =
  | Region_step of region_state
  | Region_exit of continuation * value list * region_state
  | Region_undefined of region_fault

val with_statics : state -> static_env -> state

val with_counts : region_state -> region_counts -> region_state

val plain_step : region_state -> region_step_result

val static_step : region_state -> region_step_result

val bind_let :
  region_state ->
  variable ->
  value ->
  expr ->
  static_env ->
  frame list ->
  symbol ->
  region_counts ->
  region_state

val bind_pair :
  region_state ->
  variable ->
  variable ->
  value ->
  expr ->
  static_env ->
  frame list ->
  symbol ->
  region_counts ->
  region_state

val begin_region :
  region_state ->
  variable ->
  flambda_kind ->
  mark_kind ->
  expr ->
  region_step_result

val unit_value : value

val end_region :
  region_state ->
  variable ->
  flambda_kind ->
  mark_kind ->
  simple ->
  expr ->
  region_step_result

val named_region : region_state -> simple option -> bool option

val local_allocation :
  region_state ->
  variable ->
  variable ->
  flambda_kind ->
  named ->
  simple option ->
  expr ->
  region_step_result

val count_uses : region_state -> region_state

val region_step : region_state -> region_step_result

val initial_region_state : expr -> static_env -> symbol -> region_state

val symbol_below : symbol -> symbol -> bool

val simple_below : symbol -> simple -> bool

val static_const_below : symbol -> static_const -> bool

val apply_cont_below : symbol -> apply_cont -> bool

val expr_below : symbol -> expr -> bool

val handler_below : symbol -> continuation_handler -> bool

val binding_below : symbol -> static_binding -> bool

val code_below : symbol -> code -> bool

val value_below : symbol -> value -> bool

val allocated_below : symbol -> statically_allocated_value -> bool

val statics_below : symbol -> static_env -> bool

val region_names_fresh : symbol -> expr -> static_env -> bool
