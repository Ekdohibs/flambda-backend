open Datatypes
open FlambdaHeapSpec
open FlambdaModel
open List

(** val expr_free_of_heap : expr -> bool **)

let rec expr_free_of_heap = function
  | Let (_, _, d, body) -> negb (named_changes_heap d) && expr_free_of_heap body
  | Let_static (bindings, body) ->
    forallb binding_free_of_heap bindings && expr_free_of_heap body
  | Let_cont_non_recursive (_, h, body) ->
    handler_free_of_heap h && expr_free_of_heap body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_free_of_heap h)
      handlers
    && expr_free_of_heap body
  | _ -> true

(** val handler_free_of_heap : continuation_handler -> bool **)

and handler_free_of_heap = function
  | Handler (_, _, handler) -> expr_free_of_heap handler

(** val binding_free_of_heap : static_binding -> bool **)

and binding_free_of_heap = function
  | Bind_code (_, c) -> code_free_of_heap c
  | _ -> true

(** val code_free_of_heap : code -> bool **)

and code_free_of_heap = function
  | Code (_, _, _, _, _, body) -> expr_free_of_heap body
