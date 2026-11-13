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

type lift_to =
  | Lift_to_symbol of symbol
  | Lift_to_closure_symbols of (function_slot * symbol) list

type path = positive

(** val root : path **)

let root = Coq_xH

(** val rec_handlers_map : (path -> continuation_handler ->
    continuation_handler) -> path -> (continuation * continuation_handler) list
    -> (continuation * continuation_handler) list **)

let rec rec_handlers_map f q = function
  | [] -> []
  | p :: rest ->
    let k, h = p in
    (k, f (Coq_xO q) h) :: rec_handlers_map f (Coq_xI q) rest

(** val rec_handlers_forall : (path -> continuation_handler -> bool) -> path ->
    (continuation * continuation_handler) list -> bool **)

let rec rec_handlers_forall f q = function
  | [] -> true
  | p :: rest ->
    let _, h = p in
    f (Coq_xO q) h && rec_handlers_forall f (Coq_xI q) rest

(** val rec_handlers_bindings : (path -> continuation_handler -> static_binding
    list) -> path -> (continuation * continuation_handler) list ->
    static_binding list **)

let rec rec_handlers_bindings f q = function
  | [] -> []
  | p :: rest ->
    let _, h = p in
    app (f (Coq_xO q) h) (rec_handlers_bindings f (Coq_xI q) rest)

type lifting =
  { lifting_unit : unit_interface;
    lifted_lets : (path * lift_to) list;
    lifted_my_closures : (code_id * symbol) list;
    lifted_out_of_code : (code_id * (path * lift_to) list) list;
    lifted_to_head : (path * lift_to) list;
    lifted_around : (path * lift_to) list;
    groups_put : (path * path list) list;
    tables_put : (symbol * simple list) list;
    code_copied : ((code_id * code_id) list * symbol list) list
  }

(** val lift_named : lift_to -> named -> (static_binding list * symbol) option **)

let lift_named target d =
  match target with
  | Lift_to_symbol s -> (
    match d with
    | Prim (p, elements) -> (
      match p with
      | Unary p0 -> (
        match p0 with
        | Box_number _ -> (
          match elements with
          | [] -> None
          | contents :: l -> (
            match l with
            | [] ->
              Some (Bind_symbol (s, Static_const_boxed_number contents) :: [], s)
            | _ :: _ -> None))
        | _ -> None)
      | Variadic p0 -> (
        match p0 with
        | Make_block kind ->
          let (Block_kind_values tag) = kind in
          Some (Bind_symbol (s, Static_const_block (tag, elements)) :: [], s)
        | Make_immutable_array _ ->
          Some
            ( Bind_symbol (s, Static_const_immutable_value_array elements) :: [],
              s ))
      | _ -> None)
    | Set_of_closures (code0, value_slots) ->
      Some
        ( Bind_symbol (s, Static_const_set_of_closures (code0, value_slots))
          :: [],
          s )
    | _ -> None)
  | Lift_to_closure_symbols closure_symbols -> (
    match d with
    | Closure_of_set (slot, function_decls, value_slots) -> (
      match lookup slot closure_symbols with
      | Some s ->
        if
          list_eq_dec Pos.eq_dec (map fst closure_symbols)
            (map fst function_decls)
        then Some (bind_closures closure_symbols function_decls value_slots, s)
        else None
      | None -> None)
    | _ -> None)

(** val lift_expr : (path * lift_to) list -> path -> expr -> expr **)

let rec lift_expr lets at_path e =
  match e with
  | Let (x, kind, d, body) -> (
    let body0 = lift_expr lets (Coq_xO at_path) body in
    match lookup at_path lets with
    | Some target -> (
      match lift_named target d with
      | Some p ->
        let bindings, s = p in
        Let_static (bindings, Let (x, kind, Simple (Symbol s), body0))
      | None -> Let (x, kind, d, body0))
    | None -> Let (x, kind, d, body0))
  | Let_static (bindings, body) ->
    Let_static (bindings, lift_expr lets (Coq_xO at_path) body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        lift_handler lets (Coq_xI at_path) h,
        lift_expr lets (Coq_xO at_path) body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, lift_expr lets (Coq_xO at_path) body)
  | _ -> e

(** val lift_handler : (path * lift_to) list -> path -> continuation_handler ->
    continuation_handler **)

and lift_handler lets at_path = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, lift_expr lets at_path handler)

(** val plain_simple : simple -> bool **)

let plain_simple = function Var _ -> false | _ -> true

(** val apply_cont_exits_plain : continuation list -> apply_cont -> bool **)

let apply_cont_exits_plain scope ac =
  among ac.ac_continuation scope || forallb plain_simple ac.ac_args

(** val exits_plain : continuation list -> expr -> bool **)

let rec exits_plain scope = function
  | Let (_, _, _, body) -> exits_plain scope body
  | Let_static (bindings, body) ->
    forallb binding_exits_plain bindings && exits_plain scope body
  | Let_cont_non_recursive (k, h, body) ->
    handler_exits_plain scope h && exits_plain (k :: scope) body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_exits_plain (app (map fst handlers) scope) h)
      handlers
    && exits_plain (app (map fst handlers) scope) body
  | Apply ap ->
    among ap.ap_continuation scope
    && (among ap.ap_exn_continuation scope
       || forallb plain_simple (exn_extra_simples ap))
  | Apply_cont ac -> apply_cont_exits_plain scope ac
  | Switch (_, arms) ->
    forallb (fun arm -> apply_cont_exits_plain scope (snd arm)) arms
  | Invalid -> true

(** val handler_exits_plain : continuation list -> continuation_handler -> bool
    **)

and handler_exits_plain scope = function
  | Handler (_, _, handler) -> exits_plain scope handler

(** val binding_exits_plain : static_binding -> bool **)

and binding_exits_plain = function
  | Bind_code (_, c) -> code_exits_plain c
  | _ -> true

(** val code_exits_plain : code -> bool **)

and code_exits_plain = function
  | Code (_, _, ret, _, _, body) -> exits_plain (ret :: []) body

(** val lift_unit : lifting -> expr -> expr option **)

let lift_unit l e =
  let lifted =
    bind_my_closures l.lifted_my_closures (lift_expr l.lifted_lets root e)
  in
  if
    (exits_plain [] e && well_formed_unit l.lifting_unit lifted)
    && only_closures_of l.lifted_my_closures lifted
  then Some lifted
  else None
