open BinNums
open Datatypes
open FlambdaCopyCodeSpec
open FlambdaHeapFreeSpec
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open List
open ListDef

(** val copy_named : (code_id * code_id) list -> named -> named **)

let copy_named ids d =
  match d with
  | Set_of_closures (code, value_slots) ->
    Set_of_closures (copy_of ids code, value_slots)
  | Closure_of_set (slot, function_decls, value_slots) ->
    Closure_of_set (slot, copy_decls ids function_decls, value_slots)
  | _ -> d

(** val copy_lets_expr : (code_id * code_id) list -> path list -> path -> expr
    -> expr **)

let rec copy_lets_expr ids lets at_path e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        (if among at_path lets then copy_named ids d else d),
        copy_lets_expr ids lets (Coq_xO at_path) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, copy_lets_expr ids lets (Coq_xO at_path) body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        copy_lets_handler ids lets (Coq_xI at_path) h,
        copy_lets_expr ids lets (Coq_xO at_path) body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( rec_handlers_map (copy_lets_handler ids lets) (Coq_xI at_path) handlers,
        copy_lets_expr ids lets (Coq_xO at_path) body )
  | _ -> e

(** val copy_lets_handler : (code_id * code_id) list -> path list -> path ->
    continuation_handler -> continuation_handler **)

and copy_lets_handler ids lets at_path = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, copy_lets_expr ids lets at_path handler)

(** val copy_lets_binding : (code_id * code_id) list -> (code_id * path list)
    list -> static_binding -> static_binding **)

let copy_lets_binding ids in_code binding =
  match binding with
  | Bind_code (id, c) -> (
    let (Code (params, my_closure, ret, exn, metadata, body)) = c in
    match lookup id in_code with
    | Some lets ->
      Bind_code
        ( id,
          Code
            ( params,
              my_closure,
              ret,
              exn,
              metadata,
              copy_lets_expr ids lets root body ) )
    | None -> binding)
  | _ -> binding

(** val copy_lets_in_code : (code_id * code_id) list -> (code_id * path list)
    list -> expr -> expr **)

let rec copy_lets_in_code ids in_code e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, copy_lets_in_code ids in_code body)
  | Let_static (bindings, body) ->
    Let_static
      ( map (copy_lets_binding ids in_code) bindings,
        copy_lets_in_code ids in_code body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        copy_lets_in_code_handler ids in_code h,
        copy_lets_in_code ids in_code body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, copy_lets_in_code_handler ids in_code h)
          handlers,
        copy_lets_in_code ids in_code body )
  | _ -> e

(** val copy_lets_in_code_handler : (code_id * code_id) list -> (code_id * path
    list) list -> continuation_handler -> continuation_handler **)

and copy_lets_in_code_handler ids in_code = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, copy_lets_in_code ids in_code handler)

type let_copy =
  ((code_id * code_id) list * symbol list)
  * (path list * (code_id * path list) list)

(** val copies_apart : (code_id * code_id) list -> bool **)

let copies_apart ids =
  forallb (fun id -> negb (among id (map snd ids))) (map fst ids)

(** val copy_let_code : let_copy -> expr -> expr **)

let copy_let_code copy e =
  let p, p0 = copy in
  let ids, symbols = p in
  let lets, in_code = p0 in
  if (exits_plain_knowing [] [] e && expr_free_of_heap e) && copies_apart ids
  then
    match copy_static_code ids symbols e with
    | Some e' -> copy_lets_in_code ids in_code (copy_lets_expr ids lets root e')
    | None -> e
  else e

(** val with_let_copies : let_copy list -> expr -> expr **)

let rec with_let_copies copies e =
  match copies with
  | [] -> e
  | copy :: copies0 -> with_let_copies copies0 (copy_let_code copy e)
