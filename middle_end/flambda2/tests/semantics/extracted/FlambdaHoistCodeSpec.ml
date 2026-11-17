open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List

(** val is_code : static_binding -> bool **)

let is_code = function Bind_code (_, _) -> true | _ -> false

(** val code_bindings : expr -> static_binding list **)

let rec code_bindings = function
  | Let (_, _, _, body) -> code_bindings body
  | Let_static (bindings, body) ->
    app (filter is_code bindings) (code_bindings body)
  | Let_cont_non_recursive (_, h, body) ->
    app (code_bindings body) (handler_code_bindings h)
  | Let_cont_recursive (_, body) -> code_bindings body
  | _ -> []

(** val handler_code_bindings : continuation_handler -> static_binding list **)

and handler_code_bindings = function
  | Handler (_, _, handler) -> code_bindings handler

(** val without_code : expr -> expr **)

let rec without_code e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, without_code body)
  | Let_static (bindings, body) ->
    Let_static
      ( filter (fun binding -> negb (is_code binding)) bindings,
        without_code body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, handler_without_code h, without_code body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, without_code body)
  | _ -> e

(** val handler_without_code : continuation_handler -> continuation_handler **)

and handler_without_code = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, without_code handler)

(** val decls_bound : (code_id -> bool) -> (function_slot * code_id) list ->
    bool **)

let decls_bound bound decls = forallb (fun decl -> bound (snd decl)) decls

(** val named_closures_bound : (code_id -> bool) -> named -> bool **)

let named_closures_bound bound = function
  | Set_of_closures (code0, _) -> bound code0
  | Closure_of_set (_, decls, _) -> decls_bound bound decls
  | _ -> true

(** val const_closures_bound : (code_id -> bool) -> static_const -> bool **)

let const_closures_bound bound = function
  | Static_const_set_of_closures (code0, _) -> bound code0
  | Static_const_closure_of_set (_, _, decls, _) -> decls_bound bound decls
  | _ -> true

(** val bound_with : code_id list -> (code_id -> bool) -> code_id -> bool **)

let bound_with ids bound id = among id ids || bound id

(** val closures_in_scope : (code_id -> bool) -> expr -> bool **)

let rec closures_in_scope bound = function
  | Let (_, _, d, body) ->
    named_closures_bound bound d && closures_in_scope bound body
  | Let_static (bindings, body) ->
    let bound0 = bound_with (flat_map code_ids_of_binding bindings) bound in
    forallb (binding_closures_in_scope bound0) bindings
    && closures_in_scope bound0 body
  | Let_cont_non_recursive (_, h, body) ->
    handler_closures_in_scope bound h && closures_in_scope bound body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_closures_in_scope bound h)
      handlers
    && closures_in_scope bound body
  | _ -> true

(** val handler_closures_in_scope : (code_id -> bool) -> continuation_handler ->
    bool **)

and handler_closures_in_scope bound = function
  | Handler (_, _, handler) -> closures_in_scope bound handler

(** val binding_closures_in_scope : (code_id -> bool) -> static_binding -> bool
    **)

and binding_closures_in_scope bound = function
  | Bind_code (_, c) -> code_closures_in_scope bound c
  | Bind_deleted_code _ -> true
  | Bind_symbol (_, const) -> const_closures_bound bound const

(** val code_closures_in_scope : (code_id -> bool) -> code -> bool **)

and code_closures_in_scope bound = function
  | Code (_, _, _, _, _, body) -> closures_in_scope bound body

(** val statics_at_unit_toplevel_of_unit : expr -> bool **)

let statics_at_unit_toplevel_of_unit e =
  existsb
    (fun unit_exn -> statics_at_unit_toplevel unit_exn true e)
    (free_continuations e)

(** val hoist_code : expr -> expr option **)

let hoist_code e =
  let codes = code_bindings e in
  if
    ((statics_at_unit_toplevel_of_unit e
     && distinct (static_names code_ids_of_binding e))
    && distinct (flat_map code_ids_of_binding codes))
    && closures_in_scope (fun _ -> false) e
  then Some (Let_static (codes, without_code e))
  else None
