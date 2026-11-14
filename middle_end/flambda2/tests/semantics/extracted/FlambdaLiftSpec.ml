open BinNums
open BinPos
open Datatypes
open FlambdaHeapFreeSpec
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
    code_copied : ((code_id * code_id) list * symbol list) list;
    made_closures_called : bool;
    own_closures_known : bool
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

(** val plain_or : variable list -> simple -> bool **)

let plain_or plain = function Var x -> among x plain | _ -> true

type spoilt = (continuation * nat option) list

(** val spoils_at : continuation -> nat -> spoilt -> bool **)

let spoils_at k i sp =
  existsb
    (fun pat ->
      let k', at_ = pat in
      Pos.eqb k k' && match at_ with Some j -> Nat.eqb i j | None -> true)
    sp

(** val forget_conts : continuation list -> spoilt -> spoilt **)

let forget_conts ks sp =
  filter
    (fun pat ->
      let k, _ = pat in
      negb (among k ks))
    sp

(** val args_spoil : continuation -> variable list -> nat -> simple list ->
    spoilt **)

let rec args_spoil k plain i = function
  | [] -> []
  | s :: args0 ->
    app
      (if plain_or plain s then [] else (k, Some i) :: [])
      (args_spoil k plain (S i) args0)

(** val apply_cont_spoils : variable list -> apply_cont -> spoilt **)

let apply_cont_spoils plain ac =
  args_spoil ac.ac_continuation plain O ac.ac_args

(** val params_known : continuation -> spoilt -> nat -> variable list ->
    variable list -> variable list **)

let rec params_known k sp i params known =
  match params with
  | [] -> known
  | x :: params0 ->
    params_known k sp (S i) params0
      (if spoils_at k i sp then without (x :: []) known else x :: known)

(** val spoils : variable list -> expr -> spoilt **)

let rec spoils plain = function
  | Let (x, _, _, body) -> spoils (without (x :: []) plain) body
  | Let_static (_, body) -> spoils plain body
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, _, handler)) = handler0 in
    let sp = spoils plain body in
    app
      (spoils (params_known k sp O params plain) handler)
      (forget_conts (k :: []) sp)
  | Let_cont_recursive (handlers, body) ->
    forget_conts (map fst handlers)
      (app
         (flat_map
            (fun pat ->
              let _, h = pat in
              handler_spoils plain h)
            handlers)
         (spoils plain body))
  | Apply ap ->
    (ap.ap_continuation, None)
    ::
    (if forallb (plain_or plain) (exn_extra_simples ap)
     then []
     else (ap.ap_exn_continuation, None) :: [])
  | Apply_cont ac -> apply_cont_spoils plain ac
  | Switch (_, arms) ->
    flat_map (fun arm -> apply_cont_spoils plain (snd arm)) arms
  | Invalid -> []

(** val handler_spoils : variable list -> continuation_handler -> spoilt **)

and handler_spoils plain = function
  | Handler (params, _, handler) -> spoils (without params plain) handler

(** val apply_cont_exits_plain_knowing : continuation list -> variable list ->
    apply_cont -> bool **)

let apply_cont_exits_plain_knowing scope plain ac =
  among ac.ac_continuation scope || forallb (plain_or plain) ac.ac_args

(** val exits_plain_knowing : continuation list -> variable list -> expr -> bool
    **)

let rec exits_plain_knowing scope plain = function
  | Let (x, _, _, body) ->
    exits_plain_knowing scope (without (x :: []) plain) body
  | Let_static (bindings, body) ->
    forallb binding_exits_plain_knowing bindings
    && exits_plain_knowing scope plain body
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, _, handler)) = handler0 in
    exits_plain_knowing scope
      (params_known k (spoils plain body) O params plain)
      handler
    && exits_plain_knowing (k :: scope) plain body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_exits_plain_knowing (app (map fst handlers) scope) plain h)
      handlers
    && exits_plain_knowing (app (map fst handlers) scope) plain body
  | Apply ap ->
    among ap.ap_continuation scope
    && (among ap.ap_exn_continuation scope
       || forallb (plain_or plain) (exn_extra_simples ap))
  | Apply_cont ac -> apply_cont_exits_plain_knowing scope plain ac
  | Switch (_, arms) ->
    forallb
      (fun arm -> apply_cont_exits_plain_knowing scope plain (snd arm))
      arms
  | Invalid -> true

(** val handler_exits_plain_knowing : continuation list -> variable list ->
    continuation_handler -> bool **)

and handler_exits_plain_knowing scope plain = function
  | Handler (params, _, handler) ->
    exits_plain_knowing scope (without params plain) handler

(** val binding_exits_plain_knowing : static_binding -> bool **)

and binding_exits_plain_knowing = function
  | Bind_code (_, c) -> code_exits_plain_knowing c
  | _ -> true

(** val code_exits_plain_knowing : code -> bool **)

and code_exits_plain_knowing = function
  | Code (_, _, ret, _, _, body) -> exits_plain_knowing (ret :: []) [] body

(** val lift_unit : lifting -> expr -> expr option **)

let lift_unit l e =
  let lifted =
    bind_my_closures l.lifted_my_closures (lift_expr l.lifted_lets root e)
  in
  if
    ((exits_plain_knowing [] [] e && expr_free_of_heap e)
    && well_formed_unit l.lifting_unit lifted)
    && only_closures_of l.lifted_my_closures lifted
  then Some lifted
  else None
