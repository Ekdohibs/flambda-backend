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

(** val calls_nothing : expr -> bool **)

let rec calls_nothing = function
  | Let (_, _, _, body) -> calls_nothing body
  | Let_static (_, body) -> calls_nothing body
  | Let_cont_non_recursive (_, h, body) ->
    handler_calls_nothing h && calls_nothing body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_calls_nothing h)
      handlers
    && calls_nothing body
  | Apply _ -> false
  | _ -> true

(** val handler_calls_nothing : continuation_handler -> bool **)

and handler_calls_nothing = function
  | Handler (_, _, handler) -> calls_nothing handler

(** val expr_free_of_heap_outside_code : expr -> bool **)

let rec expr_free_of_heap_outside_code = function
  | Let (_, _, d, body) ->
    negb (named_changes_heap d) && expr_free_of_heap_outside_code body
  | Let_static (_, body) -> expr_free_of_heap_outside_code body
  | Let_cont_non_recursive (_, h, body) ->
    handler_free_of_heap_outside_code h && expr_free_of_heap_outside_code body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_free_of_heap_outside_code h)
      handlers
    && expr_free_of_heap_outside_code body
  | _ -> true

(** val handler_free_of_heap_outside_code : continuation_handler -> bool **)

and handler_free_of_heap_outside_code = function
  | Handler (_, _, handler) -> expr_free_of_heap_outside_code handler

(** val free_of_heap_fine : expr -> bool **)

let free_of_heap_fine e =
  expr_free_of_heap e || (calls_nothing e && expr_free_of_heap_outside_code e)
