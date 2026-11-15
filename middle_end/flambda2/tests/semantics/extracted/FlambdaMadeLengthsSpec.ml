open Datatypes
open FlambdaCseSpec
open FlambdaModel

(** val lengths_knowing : computed -> expr -> expr **)

let rec lengths_knowing known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        (match made_length_named known d with Some d' -> d' | None -> d),
        lengths_knowing
          (app (array_made_under_let x d) (forget_computed x known))
          body )
  | Let_static (bindings, body) ->
    Let_static (bindings, lengths_knowing known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, lengths_in_handler known h, lengths_knowing known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, lengths_knowing known body)
  | _ -> e

(** val lengths_in_handler : computed -> continuation_handler ->
    continuation_handler **)

and lengths_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        lengths_knowing (forget_computed_vars params known) handler )

(** val fold_made_lengths : expr -> expr **)

let fold_made_lengths e = lengths_knowing [] e
