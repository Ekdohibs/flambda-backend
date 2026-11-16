open BinInt
open BinNums
open BinPos
open Bool
open Datatypes
open FlambdaKindModel
open FlambdaPrimitiveModel
open List
open ListDef
open PeanoNat

type variable = positive

type continuation = positive

type symbol = positive

type code_id = positive

type value_slot = positive

type function_slot = positive

type const =
  | Naked_immediate of coq_Z
  | Tagged_immediate of coq_Z
  | Naked_float32 of coq_Z
  | Naked_float of coq_Z
  | Naked_int8 of coq_Z
  | Naked_int16 of coq_Z
  | Naked_int32 of coq_Z
  | Naked_int64 of coq_Z
  | Naked_nativeint of coq_Z
  | Naked_vec128 of coq_Z
  | Naked_vec256 of coq_Z
  | Naked_vec512 of coq_Z
  | Naked_mask of coq_Z
  | Null
  | Poison of flambda_kind

val kind_of_const : const -> flambda_kind

type arity_component =
  | Singleton of flambda_kind
  | Unboxed_product of arity_component list

type arity = arity_component list

val unarize_component : arity_component -> flambda_kind list

val unarize : arity -> flambda_kind list

val unarize_per_parameter : arity -> flambda_kind list list

val arity_of_kinds : flambda_kind list -> arity

val arity_component_eq_dec : arity_component -> arity_component -> bool

val arity_eqb : arity -> arity -> bool

val kinds_eqb : flambda_kind list -> flambda_kind list -> bool

val kinds_per_parameter_eqb :
  flambda_kind list list -> flambda_kind list list -> bool

val cardinal_unarized : arity -> nat

val partially_apply : arity -> nat -> arity

val kinds_agree_on_prefix :
  flambda_kind list list -> flambda_kind list list -> bool

val is_one_param_of_kind_value : arity -> bool

val partial_or_over_application : arity -> arity -> arity -> arity -> bool

type simple =
  | Var of variable
  | Symbol of symbol
  | Const of const

type named =
  | Simple of simple
  | Prim of prim * simple list
  | Set_of_closures of code_id * (value_slot * simple) list
  | Closure_of_set of
      function_slot
      * (function_slot * code_id) list
      * (value_slot * simple) list
  | Project_value_slot of value_slot * simple
  | Project_value_slot_from of function_slot * value_slot * simple
  | Project_function_slot of function_slot * function_slot * simple

type static_const =
  | Static_const_set_of_closures of code_id * (value_slot * simple) list
  | Static_const_closure_of_set of
      function_slot
      * (function_slot * symbol) list
      * (function_slot * code_id) list
      * (value_slot * simple) list
  | Static_const_block of coq_Z * simple list
  | Static_const_boxed_number of simple
  | Static_const_immutable_string of coq_Z list
  | Static_const_immutable_value_array of simple list

val simple_eq_dec : simple -> simple -> bool

val static_const_eq_dec : static_const -> static_const -> bool

type raise_kind =
  | Regular
  | Reraise
  | No_trace

type trap_action =
  | Push of continuation
  | Pop of continuation * raise_kind option

type apply_cont =
  { ac_continuation : continuation;
    ac_args : simple list;
    ac_trap_action : trap_action option
  }

val trap_action_handler : trap_action -> continuation

val apply_cont_continuations : apply_cont -> continuation list

val no_trap_action : apply_cont -> bool

val is_raise : apply_cont -> bool

type call_kind =
  | Direct of code_id
  | Indirect_unknown_arity
  | Indirect_known_arity of code_id list option

type apply_expr =
  { ap_callee : simple;
    ap_call_kind : call_kind;
    ap_args : simple list;
    ap_continuation : continuation;
    ap_exn_continuation : continuation;
    ap_exn_extra_args : (simple * flambda_kind) list;
    ap_args_arity : arity;
    ap_return_arity : arity
  }

val exn_extra_simples : apply_expr -> simple list

val no_exn_extra_args : apply_expr -> bool

type handler_metadata =
  { hm_params_kinds : flambda_kind list;
    hm_is_exn_handler : bool
  }

val make_handler_metadata : flambda_kind list -> handler_metadata

val make_exn_handler_metadata : flambda_kind list -> handler_metadata

val handler_metadata_eqb : handler_metadata -> handler_metadata -> bool

type code_metadata =
  { cm_params_arity : arity;
    cm_result_arity : arity;
    cm_is_tupled : bool;
    cm_is_stub : bool
  }

val make_code_metadata : arity -> arity -> code_metadata

val make_tupled_code_metadata : arity -> arity -> code_metadata

val stub_code_metadata : code_metadata -> code_metadata

val code_metadata_eqb : code_metadata -> code_metadata -> bool

type expr =
  | Let of variable * flambda_kind * named * expr
  | Let_static of static_binding list * expr
  | Let_cont_non_recursive of continuation * continuation_handler * expr
  | Let_cont_recursive of (continuation * continuation_handler) list * expr
  | Apply of apply_expr
  | Apply_cont of apply_cont
  | Switch of simple * (coq_Z * apply_cont) list
  | Invalid

and continuation_handler = Handler of variable list * handler_metadata * expr

and static_binding =
  | Bind_code of code_id * code
  | Bind_deleted_code of code_id
  | Bind_symbol of symbol * static_const

and code =
  | Code of
      variable list
      * variable
      * continuation
      * continuation
      * code_metadata
      * expr

val handler_kinds : continuation_handler -> flambda_kind list

val handler_is_exn_handler : continuation_handler -> bool

val code_params : code -> variable list

val code_my_closure : code -> variable

val code_return_continuation : code -> continuation

val code_exn_continuation : code -> continuation

val code_params_arity : code -> arity

val code_result_arity : code -> arity

val code_is_tupled : code -> bool

val code_is_stub : code -> bool

val code_params_kinds : code -> flambda_kind list

val code_body : code -> expr

val let_projections :
  variable -> function_slot -> (variable * function_slot) list -> expr -> expr

val let_closures :
  variable list ->
  (function_slot * code_id) list ->
  (value_slot * simple) list ->
  expr ->
  expr option

val bind_closures :
  (function_slot * symbol) list ->
  (function_slot * code_id) list ->
  (value_slot * simple) list ->
  static_binding list
