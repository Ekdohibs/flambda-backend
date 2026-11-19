open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open ListDef

(** val rewrite_code_stated_in_binding : (stated_in_scope -> expr -> expr) ->
    static_binding -> static_binding **)

let rewrite_code_stated_in_binding g binding =
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
            rewrite_code_body (g (code_stated params my_closure metadata)) body
          ) )
  | _ -> binding

(** val rewrite_code_stated : (stated_in_scope -> expr -> expr) -> expr -> expr
    **)

let rec rewrite_code_stated g e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, rewrite_code_stated g body)
  | Let_static (bindings, body) ->
    Let_static
      ( map (rewrite_code_stated_in_binding g) bindings,
        rewrite_code_stated g body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, rewrite_code_stated_in_handler g h, rewrite_code_stated g body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, rewrite_code_stated_in_handler g h)
          handlers,
        rewrite_code_stated g body )
  | _ -> e

(** val rewrite_code_stated_in_handler : (stated_in_scope -> expr -> expr) ->
    continuation_handler -> continuation_handler **)

and rewrite_code_stated_in_handler g = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, rewrite_code_stated g handler)
