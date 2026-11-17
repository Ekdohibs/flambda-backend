open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List
open ListDef

val simple_statics_in_scope : symbol list -> simple -> bool

val simples_statics_in_scope : symbol list -> simple list -> bool

val value_slots_statics_in_scope :
  symbol list -> (value_slot * simple) list -> bool

val named_statics_in_scope : symbol list -> code_id list -> named -> bool

val static_const_statics_in_scope :
  symbol list -> code_id list -> static_const -> bool

val apply_statics_in_scope : symbol list -> code_id list -> apply_expr -> bool

val apply_cont_statics_in_scope : symbol list -> apply_cont -> bool

val is_closure_const : static_const -> bool

val closure_symbols_of_binding : static_binding -> symbol list

val declared_code_ids_of_binding : static_binding -> code_id list

val group_consts_in_scope :
  symbol list -> symbol list -> code_id list -> static_binding list -> bool

type exit_names = positive list finite_map

val with_bound : positive list -> exit_names -> exit_names

val add_exit : continuation -> positive list -> exit_names -> exit_names

val meet_exits : exit_names -> exit_names -> exit_names

val remove_keys : continuation list -> exit_names -> exit_names

val group_entry : continuation list -> exit_names -> positive list option

val exits : (static_binding -> positive list) -> expr -> exit_names

val handler_exits :
  (static_binding -> positive list) -> continuation_handler -> exit_names

val handler_scope :
  (static_binding -> positive list) -> continuation -> expr -> positive list

val recursive_scope :
  (static_binding -> positive list) ->
  continuation list ->
  expr ->
  positive list

val expr_statics_in_scope : symbol list -> code_id list -> expr -> bool

val handler_statics_in_scope :
  symbol list -> code_id list -> continuation_handler -> bool

val binding_statics_in_scope :
  symbol list -> code_id list -> static_binding -> bool

val code_statics_in_scope : symbol list -> code_id list -> code -> bool

val statics_in_scope : symbol list -> expr -> bool

val static_names_apart : symbol list -> expr -> bool

val bound_continuations : expr -> continuation list

val handler_bound_continuations : continuation_handler -> continuation list

val binding_bound_continuations : static_binding -> continuation list

val code_bound_continuations : code -> continuation list

val interface_conts_not_rebound : unit_interface -> expr -> bool
