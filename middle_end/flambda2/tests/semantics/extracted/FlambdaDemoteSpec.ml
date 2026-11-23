open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List
open ListDef

(** val called_with_exn : continuation -> expr -> bool **)

let rec called_with_exn k = function
  | Let (_, _, _, body) -> called_with_exn k body
  | Let_static (_, body) -> called_with_exn k body
  | Let_cont_non_recursive (_, h, body) ->
    handler_called_with_exn k h || called_with_exn k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_called_with_exn k h)
      handlers
    || called_with_exn k body
  | Apply ap -> Pos.eqb k ap.ap_exn_continuation
  | _ -> false

(** val handler_called_with_exn : continuation -> continuation_handler -> bool **)

and handler_called_with_exn k = function
  | Handler (_, _, handler) -> called_with_exn k handler

(** val binds_continuation : continuation -> expr -> bool **)

let rec binds_continuation k = function
  | Let (_, _, _, body) -> binds_continuation k body
  | Let_static (_, body) -> binds_continuation k body
  | Let_cont_non_recursive (k', h, body) ->
    (Pos.eqb k k' || handler_binds_continuation k h)
    || binds_continuation k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let k', h = pat in
        Pos.eqb k k' || handler_binds_continuation k h)
      handlers
    || binds_continuation k body
  | _ -> false

(** val handler_binds_continuation : continuation -> continuation_handler ->
    bool **)

and handler_binds_continuation k = function
  | Handler (_, _, handler) -> binds_continuation k handler

(** val demotable : continuation list -> continuation -> continuation_handler ->
    expr -> bool **)

let demotable scope k h body =
  ((handler_is_exn_handler h && negb (called_with_exn k body))
  && negb (among k scope))
  && negb (binds_continuation k body)

(** val strip_trap : continuation -> apply_cont -> apply_cont **)

let strip_trap k ac =
  if traps k ac
  then
    { ac_continuation = ac.ac_continuation;
      ac_args = ac.ac_args;
      ac_trap_action = None
    }
  else ac

(** val strip_traps : continuation -> expr -> expr **)

let rec strip_traps k = function
  | Let (x, kind, defining_expr, body) ->
    Let (x, kind, defining_expr, strip_traps k body)
  | Let_static (bindings, body) -> Let_static (bindings, strip_traps k body)
  | Let_cont_non_recursive (k', h, body) ->
    Let_cont_non_recursive (k', strip_traps_handler k h, strip_traps k body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map (fun kh -> fst kh, strip_traps_handler k (snd kh)) handlers,
        strip_traps k body )
  | Apply_cont ac -> Apply_cont (strip_trap k ac)
  | Switch (scrutinee, arms) ->
    Switch (scrutinee, map (fun arm -> fst arm, strip_trap k (snd arm)) arms)
  | x -> x

(** val strip_traps_handler : continuation -> continuation_handler ->
    continuation_handler **)

and strip_traps_handler k = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, strip_traps k handler)

(** val clear_exn_handler : continuation_handler -> continuation_handler **)

let clear_exn_handler = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        { hm_params_kinds = metadata.hm_params_kinds;
          hm_params_subkinds = metadata.hm_params_subkinds;
          hm_is_exn_handler = false
        },
        handler )

(** val demote_handlers : continuation list -> expr -> expr **)

let rec demote_handlers scope e =
  match e with
  | Let (x, kind, defining_expr, body) ->
    Let (x, kind, defining_expr, demote_handlers scope body)
  | Let_static (bindings, body) ->
    Let_static (map demote_handlers_binding bindings, demote_handlers scope body)
  | Let_cont_non_recursive (k, h, body) ->
    if demotable scope k h body
    then
      Let_cont_non_recursive
        ( k,
          clear_exn_handler (demote_handlers_handler scope h),
          strip_traps k (demote_handlers (k :: scope) body) )
    else
      Let_cont_non_recursive
        (k, demote_handlers_handler scope h, demote_handlers (k :: scope) body)
  | Let_cont_recursive (handlers, body) ->
    let inner = app (map fst handlers) scope in
    Let_cont_recursive
      ( map (fun kh -> fst kh, demote_handlers_handler inner (snd kh)) handlers,
        demote_handlers inner body )
  | _ -> e

(** val demote_handlers_handler : continuation list -> continuation_handler ->
    continuation_handler **)

and demote_handlers_handler scope = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, demote_handlers scope handler)

(** val demote_handlers_binding : static_binding -> static_binding **)

and demote_handlers_binding binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, demote_handlers_code c)
  | _ -> binding

(** val demote_handlers_code : code -> code **)

and demote_handlers_code = function
  | Code
      (params, my_closure, return_continuation, exn_continuation, metadata, body)
    ->
    Code
      ( params,
        my_closure,
        return_continuation,
        exn_continuation,
        metadata,
        demote_handlers [return_continuation; exn_continuation] body )
