open FlambdaModel

(** val walk_spine : (expr -> expr) -> expr -> expr **)

let rec walk_spine step e =
  step
    (match e with
    | Let (x, kind, d, body) -> Let (x, kind, d, walk_spine step body)
    | Let_static (bindings, body) -> Let_static (bindings, walk_spine step body)
    | Let_cont_non_recursive (k, handler0, body) ->
      let (Handler (params, metadata, handler)) = handler0 in
      Let_cont_non_recursive
        ( k,
          Handler (params, metadata, walk_spine step handler),
          walk_spine step body )
    | Let_cont_recursive (handlers, body) ->
      Let_cont_recursive (handlers, walk_spine step body)
    | _ -> e)
