open BinNums
open BinPos
open Datatypes
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List

(** val closed_named : named -> bool **)

let closed_named = function
  | Prim (_, args) -> forallb plain_simple args
  | Set_of_closures (_, value_slots) ->
    forallb (fun slot -> plain_simple (snd slot)) value_slots
  | Closure_of_set (_, _, value_slots) ->
    forallb (fun slot -> plain_simple (snd slot)) value_slots
  | _ -> false

(** val lifted_out : (path * lift_to) list -> path -> named -> (static_binding
    list * symbol) option **)

let lifted_out lets at_path d =
  match lookup at_path lets with
  | Some target -> if closed_named d then lift_named target d else None
  | None -> None

type lets_in_code = (code_id * (path * lift_to) list) list

(** val lets_of : lets_in_code -> code_id -> (path * lift_to) list **)

let lets_of k id = match lookup id k with Some lets -> lets | None -> []

(** val lifted_out_bindings : (path * lift_to) list -> path -> expr ->
    static_binding list **)

let rec lifted_out_bindings lets at_path = function
  | Let (_, _, d, body) ->
    app
      (match lifted_out lets at_path d with
      | Some p ->
        let bindings, _ = p in
        bindings
      | None -> [])
      (lifted_out_bindings lets (Coq_xO at_path) body)
  | Let_static (_, body) -> lifted_out_bindings lets (Coq_xO at_path) body
  | Let_cont_non_recursive (_, h, body) ->
    app
      (lifted_out_bindings lets (Coq_xO at_path) body)
      (handler_lifted_out_bindings lets (Coq_xI at_path) h)
  | Let_cont_recursive (handlers, body) ->
    app
      (lifted_out_bindings lets (Coq_xO at_path) body)
      (rec_handlers_bindings
         (handler_lifted_out_bindings lets)
         (Coq_xI at_path) handlers)
  | _ -> []

(** val handler_lifted_out_bindings : (path * lift_to) list -> path ->
    continuation_handler -> static_binding list **)

and handler_lifted_out_bindings lets at_path = function
  | Handler (_, _, handler) -> lifted_out_bindings lets at_path handler

(** val same_binding : static_binding -> static_binding -> bool **)

let same_binding binding binding' =
  match binding with
  | Bind_symbol (s, const) -> (
    match binding' with
    | Bind_symbol (s', const') ->
      Pos.eqb s s' && if static_const_eq_dec const const' then true else false
    | _ -> false)
  | _ -> false

(** val once_each : static_binding list -> static_binding list -> static_binding
    list **)

let rec once_each seen = function
  | [] -> []
  | binding :: rest ->
    if existsb (same_binding binding) seen
    then once_each seen rest
    else binding :: once_each (binding :: seen) rest

(** val lift_from_code : lets_in_code -> (path * lift_to) list -> path -> expr
    -> expr **)

let rec lift_from_code k lets at_path e =
  match e with
  | Let (x, kind, d, body) -> (
    let body0 = lift_from_code k lets (Coq_xO at_path) body in
    match lifted_out lets at_path d with
    | Some p ->
      let _, s = p in
      Let (x, kind, Simple (Symbol s), body0)
    | None -> Let (x, kind, d, body0))
  | Let_static (bindings, body) ->
    Let_static
      ( flat_map (binding_lift_from_code k) bindings,
        lift_from_code k lets (Coq_xO at_path) body )
  | Let_cont_non_recursive (k0, h, body) ->
    Let_cont_non_recursive
      ( k0,
        handler_lift_from_code k lets (Coq_xI at_path) h,
        lift_from_code k lets (Coq_xO at_path) body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( rec_handlers_map
          (handler_lift_from_code k lets)
          (Coq_xI at_path) handlers,
        lift_from_code k lets (Coq_xO at_path) body )
  | _ -> e

(** val handler_lift_from_code : lets_in_code -> (path * lift_to) list -> path
    -> continuation_handler -> continuation_handler **)

and handler_lift_from_code k lets at_path = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, lift_from_code k lets at_path handler)

(** val binding_lift_from_code : lets_in_code -> static_binding ->
    static_binding list **)

and binding_lift_from_code k binding =
  match binding with
  | Bind_code (id, c) ->
    let (Code (params, my_closure, ret, exn, metadata, body)) = c in
    app
      (once_each [] (lifted_out_bindings (lets_of k id) root body))
      (Bind_code
         ( id,
           Code
             ( params,
               my_closure,
               ret,
               exn,
               metadata,
               lift_from_code k (lets_of k id) root body ) )
      :: [])
  | _ -> binding :: []

(** val lift_from_code_unit : unit_interface -> lets_in_code -> expr -> expr
    option **)

let lift_from_code_unit u k e =
  let lifted = lift_from_code k [] root e in
  if exits_plain_knowing [] [] e && well_formed_unit u lifted
  then Some lifted
  else None
