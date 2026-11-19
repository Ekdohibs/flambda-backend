open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open ListDef

(** val rewrite_rec_handlers_stated : (stated_in_scope -> expr -> expr) ->
    stated_in_scope -> expr -> expr **)

let rec rewrite_rec_handlers_stated f s e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        d,
        rewrite_rec_handlers_stated f ((x, stated_of_named s d) :: s) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, rewrite_rec_handlers_stated f s body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        rewrite_rec_handlers_stated_in_handler f s h,
        rewrite_rec_handlers_stated f s body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, rewrite_rec_handler_stated f s h)
          handlers,
        rewrite_rec_handlers_stated f s body )
  | _ -> e

(** val rewrite_rec_handlers_stated_in_handler : (stated_in_scope -> expr ->
    expr) -> stated_in_scope -> continuation_handler -> continuation_handler **)

and rewrite_rec_handlers_stated_in_handler f s = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        rewrite_rec_handlers_stated f
          (params_stated params metadata.hm_params_subkinds s)
          handler )

(** val rewrite_rec_handler_stated : (stated_in_scope -> expr -> expr) ->
    stated_in_scope -> continuation_handler -> continuation_handler **)

and rewrite_rec_handler_stated f s = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        rewrite_code_body
          (f (params_stated params metadata.hm_params_subkinds s))
          (rewrite_rec_handlers_stated f
             (params_stated params metadata.hm_params_subkinds s)
             handler) )
