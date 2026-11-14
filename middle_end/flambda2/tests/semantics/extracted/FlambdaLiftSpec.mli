open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaMyClosureSpec
open FlambdaPrimitiveModel
open FlambdaWellFormedSpec
open List
open ListDef
open PeanoNat

type lift_to =
  | Lift_to_symbol of symbol
  | Lift_to_closure_symbols of (function_slot * symbol) list

type path = positive

val root : path

val rec_handlers_map :
  (path -> continuation_handler -> continuation_handler) ->
  path ->
  (continuation * continuation_handler) list ->
  (continuation * continuation_handler) list

val rec_handlers_forall :
  (path -> continuation_handler -> bool) ->
  path ->
  (continuation * continuation_handler) list ->
  bool

val rec_handlers_bindings :
  (path -> continuation_handler -> static_binding list) ->
  path ->
  (continuation * continuation_handler) list ->
  static_binding list

type lifting =
  { lifting_unit : unit_interface;
    lifted_lets : (path * lift_to) list;
    lifted_my_closures : (code_id * symbol) list;
    lifted_out_of_code : (code_id * (path * lift_to) list) list;
    lifted_to_head : (path * lift_to) list;
    lifted_around : (path * lift_to) list;
    groups_put : (path * path list) list;
    tables_put : (symbol * simple list) list;
    code_copied : ((code_id * code_id) list * symbol list) list;
    made_closures_called : bool;
    own_closures_known : bool
  }

val lift_named : lift_to -> named -> (static_binding list * symbol) option

val lift_expr : (path * lift_to) list -> path -> expr -> expr

val lift_handler :
  (path * lift_to) list -> path -> continuation_handler -> continuation_handler

val plain_simple : simple -> bool

val plain_or : variable list -> simple -> bool

type spoilt = (continuation * nat option) list

val spoils_at : continuation -> nat -> spoilt -> bool

val forget_conts : continuation list -> spoilt -> spoilt

val args_spoil : continuation -> variable list -> nat -> simple list -> spoilt

val apply_cont_spoils : variable list -> apply_cont -> spoilt

val params_known :
  continuation ->
  spoilt ->
  nat ->
  variable list ->
  variable list ->
  variable list

val spoils : variable list -> expr -> spoilt

val handler_spoils : variable list -> continuation_handler -> spoilt

val apply_cont_exits_plain_knowing :
  continuation list -> variable list -> apply_cont -> bool

val exits_plain_knowing : continuation list -> variable list -> expr -> bool

val handler_exits_plain_knowing :
  continuation list -> variable list -> continuation_handler -> bool

val binding_exits_plain_knowing : static_binding -> bool

val code_exits_plain_knowing : code -> bool

val lift_unit : lifting -> expr -> expr option
