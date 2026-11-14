open BinNums
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open ListDef

type calls_in_code = (code_id * path list) list

(** val calls_of : calls_in_code -> code_id -> path list **)

let calls_of k id = match lookup id k with Some sites -> sites | None -> []

(** val inline_calls_at_paths : calls_in_code -> path list -> (code_id * code)
    list -> path -> expr -> expr **)

let rec inline_calls_at_paths k sites known at_path e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, inline_calls_at_paths k sites known (Coq_xO at_path) body)
  | Let_static (bindings, body) ->
    let bindings0 = map (binding_inline_calls_at_paths k known) bindings in
    Let_static
      ( bindings0,
        inline_calls_at_paths k sites (codes_of bindings0 known)
          (Coq_xO at_path) body )
  | Let_cont_non_recursive (k0, h, body) ->
    Let_cont_non_recursive
      ( k0,
        handler_inline_calls_at_paths k sites known (Coq_xI at_path) h,
        inline_calls_at_paths k sites known (Coq_xO at_path) body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( rec_handlers_map
          (handler_inline_calls_at_paths k sites known)
          (Coq_xI at_path) handlers,
        inline_calls_at_paths k sites known (Coq_xO at_path) body )
  | Apply ap -> if among at_path sites then inline_apply [] known ap else e
  | _ -> e

(** val handler_inline_calls_at_paths : calls_in_code -> path list -> (code_id *
    code) list -> path -> continuation_handler -> continuation_handler **)

and handler_inline_calls_at_paths k sites known at_path = function
  | Handler (params, metadata, handler) ->
    Handler
      (params, metadata, inline_calls_at_paths k sites known at_path handler)

(** val binding_inline_calls_at_paths : calls_in_code -> (code_id * code) list
    -> static_binding -> static_binding **)

and binding_inline_calls_at_paths k known binding =
  match binding with
  | Bind_code (id, c) ->
    let (Code (params, my_closure, ret, exn, metadata, body)) = c in
    Bind_code
      ( id,
        Code
          ( params,
            my_closure,
            ret,
            exn,
            metadata,
            inline_calls_at_paths k (calls_of k id) known root body ) )
  | _ -> binding

(** val inline_calls_at : path list -> calls_in_code -> expr -> expr **)

let inline_calls_at sites k e = inline_calls_at_paths k sites [] root e
