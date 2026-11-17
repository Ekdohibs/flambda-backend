open BinNums
open BinPos
open Datatypes
open FlambdaBindersSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveSpec
open List
open ListDef
open OxCamlList

val free_continuations : expr -> continuation list

val handler_free_continuations : continuation_handler -> continuation list

val continuations_used : expr -> continuation list

val handler_continuations_used : continuation_handler -> continuation list

val traps : continuation -> apply_cont -> bool

val raises_to : continuation -> expr -> bool

val handler_raises_to : continuation -> continuation_handler -> bool

val handler_at_unit_toplevel :
  continuation -> continuation -> continuation_handler -> expr -> bool

val statics_outside_loops_and_code : bool -> expr -> bool

val handler_statics_outside_loops_and_code :
  bool -> continuation_handler -> bool

val binding_statics_outside_loops_and_code : static_binding -> bool

val code_statics_outside_loops_and_code : code -> bool

val code_ids_of_binding : static_binding -> code_id list

val symbols_of_binding : static_binding -> symbol list

val static_names : (static_binding -> positive list) -> expr -> positive list

val handler_static_names :
  (static_binding -> positive list) -> continuation_handler -> positive list

val set_well_formed : function_slot -> (function_slot * code_id) list -> bool

val named_sets_well_formed : named -> bool

val binds_symbol_to : symbol -> static_const -> static_binding -> bool

val static_set_whole : static_binding list -> static_binding -> bool

val static_set_code_ids : static_binding -> code_id list

val binds_symbol_of_set : static_binding -> static_binding -> bool

val static_sets_code_disjoint : static_binding list -> bool

val sets_well_formed : expr -> bool

val handler_sets_well_formed : continuation_handler -> bool

val binding_sets_well_formed : static_binding -> bool

val code_sets_well_formed : code -> bool

val simple_in_range : simple -> bool

val value_slots_in_range : (value_slot * simple) list -> bool

val named_in_range : named -> bool

val static_const_in_range : static_const -> bool

val apply_cont_in_range : apply_cont -> bool

val apply_in_range : apply_expr -> bool

val arm_in_range : coq_Z * apply_cont -> bool

val numbers_in_range : expr -> bool

val handler_numbers_in_range : continuation_handler -> bool

val binding_numbers_in_range : static_binding -> bool

val code_numbers_in_range : code -> bool

val applies : continuation -> apply_cont -> bool

val applies_cont : continuation -> expr -> bool

val handler_applies_cont : continuation -> continuation_handler -> bool

val gives_extra_args : continuation -> expr -> bool

val handler_gives_extra_args : continuation -> continuation_handler -> bool

val traps_apart : continuation -> apply_cont -> bool

val installs_or_removes : continuation -> expr -> bool

val handler_installs_or_removes : continuation -> continuation_handler -> bool

val ways_out_well_formed : continuation -> expr -> bool

val takes_exception : continuation_handler -> bool

val exn_handlers_well_formed : expr -> bool

val handler_exn_handlers_well_formed : continuation_handler -> bool

val binding_exn_handlers_well_formed : static_binding -> bool

val code_exn_handlers_well_formed : code -> bool

val unit_exn_handlers_well_formed : continuation -> expr -> bool

type trap_stack = continuation list

val trap_stack_eqb : trap_stack -> trap_stack -> bool

type demands = (continuation * trap_stack) list

type entry =
  | Not_entered
  | Entered_with of trap_stack
  | Entered_with_two_stacks

val entry_of : continuation -> demands -> entry

val leave_scope : continuation list -> demands -> demands option

val apply_cont_demands : trap_stack -> apply_cont -> demands option

val apply_demands : trap_stack -> apply_expr -> demands option

val arms_demands : trap_stack -> (coq_Z * apply_cont) list -> demands option

val code_demands_met : continuation -> continuation -> demands -> bool

val iterate : nat -> ('a1 -> 'a1 option) -> 'a1 -> 'a1 option

val trap_demands : trap_stack -> expr -> demands option

val handler_trap_demands : trap_stack -> continuation_handler -> demands option

val binding_traps_well_formed : static_binding -> bool

val code_traps_well_formed : code -> bool

val traps_well_formed : continuation -> expr -> bool

type unit_interface =
  { unit_return : continuation;
    unit_exn : continuation
  }

val well_formed_unit : unit_interface -> expr -> bool
