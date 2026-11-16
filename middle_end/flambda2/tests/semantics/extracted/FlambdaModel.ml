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

(** val kind_of_const : const -> flambda_kind **)

let kind_of_const = function
  | Naked_immediate _ -> Naked_number Kind_naked_immediate
  | Naked_float32 _ -> Naked_number Kind_naked_float32
  | Naked_float _ -> Naked_number Kind_naked_float
  | Naked_int8 _ -> Naked_number Kind_naked_int8
  | Naked_int16 _ -> Naked_number Kind_naked_int16
  | Naked_int32 _ -> Naked_number Kind_naked_int32
  | Naked_int64 _ -> Naked_number Kind_naked_int64
  | Naked_nativeint _ -> Naked_number Kind_naked_nativeint
  | Naked_vec128 _ -> Naked_number Kind_naked_vec128
  | Naked_vec256 _ -> Naked_number Kind_naked_vec256
  | Naked_vec512 _ -> Naked_number Kind_naked_vec512
  | Naked_mask _ -> Naked_number Kind_naked_mask
  | Poison kind -> kind
  | _ -> Value

type arity_component =
  | Singleton of flambda_kind
  | Unboxed_product of arity_component list

type arity = arity_component list

(** val unarize_component : arity_component -> flambda_kind list **)

let rec unarize_component = function
  | Singleton k -> k :: []
  | Unboxed_product components -> flat_map unarize_component components

(** val unarize : arity -> flambda_kind list **)

let unarize a = flat_map unarize_component a

(** val unarize_per_parameter : arity -> flambda_kind list list **)

let unarize_per_parameter a = map unarize_component a

(** val arity_of_kinds : flambda_kind list -> arity **)

let arity_of_kinds kinds = map (fun x -> Singleton x) kinds

(** val arity_component_eq_dec : arity_component -> arity_component -> bool **)

let rec arity_component_eq_dec c c' =
  match c with
  | Singleton k -> (
    match c' with
    | Singleton k0 -> kind_eq_dec k k0
    | Unboxed_product _ -> false)
  | Unboxed_product components -> (
    match c' with
    | Singleton _ -> false
    | Unboxed_product components0 ->
      list_eq_dec arity_component_eq_dec components components0)

(** val arity_eqb : arity -> arity -> bool **)

let arity_eqb a a' =
  if list_eq_dec arity_component_eq_dec a a' then true else false

(** val kinds_eqb : flambda_kind list -> flambda_kind list -> bool **)

let kinds_eqb kinds kinds' =
  if list_eq_dec kind_eq_dec kinds kinds' then true else false

(** val kinds_per_parameter_eqb : flambda_kind list list -> flambda_kind list
    list -> bool **)

let kinds_per_parameter_eqb kinds kinds' =
  if list_eq_dec (list_eq_dec kind_eq_dec) kinds kinds' then true else false

(** val cardinal_unarized : arity -> nat **)

let cardinal_unarized a = length (unarize a)

(** val partially_apply : arity -> nat -> arity **)

let partially_apply a provided = skipn provided a

(** val kinds_agree_on_prefix : flambda_kind list list -> flambda_kind list list
    -> bool **)

let rec kinds_agree_on_prefix params args =
  match params with
  | [] -> true
  | param :: params0 -> (
    match args with
    | [] -> true
    | arg :: args0 -> kinds_eqb param arg && kinds_agree_on_prefix params0 args0
    )

(** val is_one_param_of_kind_value : arity -> bool **)

let is_one_param_of_kind_value = function
  | [] -> false
  | a0 :: l -> (
    match a0 with
    | Singleton k -> (
      match k with
      | Value -> ( match l with [] -> true | _ :: _ -> false)
      | _ -> false)
    | Unboxed_product _ -> false)

(** val partial_or_over_application : arity -> arity -> arity -> arity -> bool **)

let partial_or_over_application args_arity return_arity params_arity
    result_arity =
  let provided = length args_arity in
  let params = length params_arity in
  kinds_agree_on_prefix
    (unarize_per_parameter params_arity)
    (unarize_per_parameter args_arity)
  &&
  if Nat.ltb params provided
  then is_one_param_of_kind_value result_arity
  else
    (Nat.ltb O provided && Nat.ltb provided params)
    && is_one_param_of_kind_value return_arity

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

(** val simple_eq_dec : simple -> simple -> bool **)

let simple_eq_dec s s' =
  match s with
  | Var x -> ( match s' with Var x0 -> Pos.eq_dec x x0 | _ -> false)
  | Symbol s0 -> ( match s' with Symbol s1 -> Pos.eq_dec s0 s1 | _ -> false)
  | Const c -> (
    match s' with
    | Const c0 -> (
      match c with
      | Naked_immediate i -> (
        match c0 with Naked_immediate i0 -> Z.eq_dec i i0 | _ -> false)
      | Tagged_immediate i -> (
        match c0 with Tagged_immediate i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_float32 bits -> (
        match c0 with Naked_float32 bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Naked_float bits -> (
        match c0 with Naked_float bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Naked_int8 i -> (
        match c0 with Naked_int8 i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_int16 i -> (
        match c0 with Naked_int16 i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_int32 i -> (
        match c0 with Naked_int32 i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_int64 i -> (
        match c0 with Naked_int64 i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_nativeint i -> (
        match c0 with Naked_nativeint i0 -> Z.eq_dec i i0 | _ -> false)
      | Naked_vec128 bits -> (
        match c0 with Naked_vec128 bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Naked_vec256 bits -> (
        match c0 with Naked_vec256 bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Naked_vec512 bits -> (
        match c0 with Naked_vec512 bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Naked_mask bits -> (
        match c0 with Naked_mask bits0 -> Z.eq_dec bits bits0 | _ -> false)
      | Null -> ( match c0 with Null -> true | _ -> false)
      | Poison kind -> (
        match c0 with
        | Poison kind0 -> (
          match kind with
          | Value -> ( match kind0 with Value -> true | _ -> false)
          | Naked_number k -> (
            match kind0 with
            | Naked_number k0 -> (
              match k with
              | Kind_naked_immediate -> (
                match k0 with Kind_naked_immediate -> true | _ -> false)
              | Kind_naked_float32 -> (
                match k0 with Kind_naked_float32 -> true | _ -> false)
              | Kind_naked_float -> (
                match k0 with Kind_naked_float -> true | _ -> false)
              | Kind_naked_int8 -> (
                match k0 with Kind_naked_int8 -> true | _ -> false)
              | Kind_naked_int16 -> (
                match k0 with Kind_naked_int16 -> true | _ -> false)
              | Kind_naked_int32 -> (
                match k0 with Kind_naked_int32 -> true | _ -> false)
              | Kind_naked_int64 -> (
                match k0 with Kind_naked_int64 -> true | _ -> false)
              | Kind_naked_nativeint -> (
                match k0 with Kind_naked_nativeint -> true | _ -> false)
              | Kind_naked_vec128 -> (
                match k0 with Kind_naked_vec128 -> true | _ -> false)
              | Kind_naked_vec256 -> (
                match k0 with Kind_naked_vec256 -> true | _ -> false)
              | Kind_naked_vec512 -> (
                match k0 with Kind_naked_vec512 -> true | _ -> false)
              | Kind_naked_mask -> (
                match k0 with Kind_naked_mask -> true | _ -> false))
            | _ -> false)
          | Region -> ( match kind0 with Region -> true | _ -> false)
          | Rec_info -> ( match kind0 with Rec_info -> true | _ -> false))
        | _ -> false))
    | _ -> false)

(** val static_const_eq_dec : static_const -> static_const -> bool **)

let static_const_eq_dec const0 const' =
  match const0 with
  | Static_const_set_of_closures (code0, value_slots) -> (
    match const' with
    | Static_const_set_of_closures (code1, value_slots0) ->
      if Pos.eq_dec code0 code1
      then
        list_eq_dec
          (fun x y ->
            let a, b = x in
            let v, s = y in
            if Pos.eq_dec a v then simple_eq_dec b s else false)
          value_slots value_slots0
      else false
    | _ -> false)
  | Static_const_closure_of_set
      (slot, closure_symbols, function_decls, value_slots) -> (
    match const' with
    | Static_const_closure_of_set
        (slot0, closure_symbols0, function_decls0, value_slots0) ->
      if Pos.eq_dec slot slot0
      then
        if
          list_eq_dec
            (fun x y ->
              let a, b = x in
              let f, s = y in
              if Pos.eq_dec a f then Pos.eq_dec b s else false)
            closure_symbols closure_symbols0
        then
          if
            list_eq_dec
              (fun x y ->
                let a, b = x in
                let f, c = y in
                if Pos.eq_dec a f then Pos.eq_dec b c else false)
              function_decls function_decls0
          then
            list_eq_dec
              (fun x y ->
                let a, b = x in
                let v, s = y in
                if Pos.eq_dec a v then simple_eq_dec b s else false)
              value_slots value_slots0
          else false
        else false
      else false
    | _ -> false)
  | Static_const_block (tag, fields) -> (
    match const' with
    | Static_const_block (tag0, fields0) ->
      if Z.eq_dec tag tag0
      then list_eq_dec simple_eq_dec fields fields0
      else false
    | _ -> false)
  | Static_const_boxed_number contents -> (
    match const' with
    | Static_const_boxed_number contents0 -> simple_eq_dec contents contents0
    | _ -> false)
  | Static_const_immutable_string bytes -> (
    match const' with
    | Static_const_immutable_string bytes0 -> list_eq_dec Z.eq_dec bytes bytes0
    | _ -> false)
  | Static_const_immutable_value_array elements -> (
    match const' with
    | Static_const_immutable_value_array elements0 ->
      list_eq_dec simple_eq_dec elements elements0
    | _ -> false)

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

(** val trap_action_handler : trap_action -> continuation **)

let trap_action_handler = function
  | Push exn_handler -> exn_handler
  | Pop (exn_handler, _) -> exn_handler

(** val apply_cont_continuations : apply_cont -> continuation list **)

let apply_cont_continuations ac =
  ac.ac_continuation
  ::
  (match ac.ac_trap_action with
  | Some action -> trap_action_handler action :: []
  | None -> [])

(** val no_trap_action : apply_cont -> bool **)

let no_trap_action ac =
  match ac.ac_trap_action with Some _ -> false | None -> true

(** val is_raise : apply_cont -> bool **)

let is_raise ac =
  match ac.ac_trap_action with
  | Some t -> (
    match t with
    | Push _ -> false
    | Pop (exn_handler, _) -> Pos.eqb ac.ac_continuation exn_handler)
  | None -> false

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

(** val exn_extra_simples : apply_expr -> simple list **)

let exn_extra_simples ap = map fst ap.ap_exn_extra_args

(** val no_exn_extra_args : apply_expr -> bool **)

let no_exn_extra_args ap =
  match ap.ap_exn_extra_args with [] -> true | _ :: _ -> false

type handler_metadata =
  { hm_params_kinds : flambda_kind list;
    hm_is_exn_handler : bool
  }

(** val make_handler_metadata : flambda_kind list -> handler_metadata **)

let make_handler_metadata params_kinds =
  { hm_params_kinds = params_kinds; hm_is_exn_handler = false }

(** val make_exn_handler_metadata : flambda_kind list -> handler_metadata **)

let make_exn_handler_metadata params_kinds =
  { hm_params_kinds = params_kinds; hm_is_exn_handler = true }

(** val handler_metadata_eqb : handler_metadata -> handler_metadata -> bool **)

let handler_metadata_eqb m m' =
  kinds_eqb m.hm_params_kinds m'.hm_params_kinds
  && eqb m.hm_is_exn_handler m'.hm_is_exn_handler

type code_metadata =
  { cm_params_arity : arity;
    cm_result_arity : arity;
    cm_is_tupled : bool;
    cm_is_stub : bool
  }

(** val make_code_metadata : arity -> arity -> code_metadata **)

let make_code_metadata params_arity result_arity =
  { cm_params_arity = params_arity;
    cm_result_arity = result_arity;
    cm_is_tupled = false;
    cm_is_stub = false
  }

(** val make_tupled_code_metadata : arity -> arity -> code_metadata **)

let make_tupled_code_metadata params_arity result_arity =
  { cm_params_arity = params_arity;
    cm_result_arity = result_arity;
    cm_is_tupled = true;
    cm_is_stub = false
  }

(** val stub_code_metadata : code_metadata -> code_metadata **)

let stub_code_metadata m =
  { cm_params_arity = m.cm_params_arity;
    cm_result_arity = m.cm_result_arity;
    cm_is_tupled = m.cm_is_tupled;
    cm_is_stub = true
  }

(** val code_metadata_eqb : code_metadata -> code_metadata -> bool **)

let code_metadata_eqb m m' =
  ((arity_eqb m.cm_params_arity m'.cm_params_arity
   && arity_eqb m.cm_result_arity m'.cm_result_arity)
  && eqb m.cm_is_tupled m'.cm_is_tupled)
  && eqb m.cm_is_stub m'.cm_is_stub

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

(** val handler_kinds : continuation_handler -> flambda_kind list **)

let handler_kinds = function
  | Handler (_, metadata, _) -> metadata.hm_params_kinds

(** val handler_is_exn_handler : continuation_handler -> bool **)

let handler_is_exn_handler = function
  | Handler (_, metadata, _) -> metadata.hm_is_exn_handler

(** val code_params : code -> variable list **)

let code_params = function Code (params, _, _, _, _, _) -> params

(** val code_my_closure : code -> variable **)

let code_my_closure = function Code (_, my_closure, _, _, _, _) -> my_closure

(** val code_return_continuation : code -> continuation **)

let code_return_continuation = function
  | Code (_, _, return_continuation, _, _, _) -> return_continuation

(** val code_exn_continuation : code -> continuation **)

let code_exn_continuation = function
  | Code (_, _, _, exn_continuation, _, _) -> exn_continuation

(** val code_params_arity : code -> arity **)

let code_params_arity = function
  | Code (_, _, _, _, metadata, _) -> metadata.cm_params_arity

(** val code_result_arity : code -> arity **)

let code_result_arity = function
  | Code (_, _, _, _, metadata, _) -> metadata.cm_result_arity

(** val code_is_tupled : code -> bool **)

let code_is_tupled = function
  | Code (_, _, _, _, metadata, _) -> metadata.cm_is_tupled

(** val code_is_stub : code -> bool **)

let code_is_stub = function
  | Code (_, _, _, _, metadata, _) -> metadata.cm_is_stub

(** val code_params_kinds : code -> flambda_kind list **)

let code_params_kinds c = unarize (code_params_arity c)

(** val code_body : code -> expr **)

let code_body = function Code (_, _, _, _, _, body) -> body

(** val let_projections : variable -> function_slot -> (variable *
    function_slot) list -> expr -> expr **)

let rec let_projections base base_slot bound body =
  match bound with
  | [] -> body
  | p :: bound0 ->
    let x, slot = p in
    Let
      ( x,
        Value,
        Project_function_slot (base_slot, slot, Var base),
        let_projections base base_slot bound0 body )

(** val let_closures : variable list -> (function_slot * code_id) list ->
    (value_slot * simple) list -> expr -> expr option **)

let let_closures bound_vars function_decls value_slots body =
  match bound_vars with
  | [] -> None
  | x :: bound_vars0 -> (
    match function_decls with
    | [] -> None
    | p :: other_decls ->
      let slot, _ = p in
      if Nat.eqb (length bound_vars0) (length other_decls)
      then
        Some
          (Let
             ( x,
               Value,
               Closure_of_set (slot, function_decls, value_slots),
               let_projections x slot
                 (combine bound_vars0 (map fst other_decls))
                 body ))
      else None)

(** val bind_closures : (function_slot * symbol) list -> (function_slot *
    code_id) list -> (value_slot * simple) list -> static_binding list **)

let bind_closures closure_symbols function_decls value_slots =
  map
    (fun pat ->
      let slot, s = pat in
      Bind_symbol
        ( s,
          Static_const_closure_of_set
            (slot, closure_symbols, function_decls, value_slots) ))
    closure_symbols
