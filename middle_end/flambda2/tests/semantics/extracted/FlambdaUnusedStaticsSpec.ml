open Datatypes
open FlambdaMapSpec
open FlambdaModel
open List
open ListDef

type dropped =
  { dropped_symbols : symbol list;
    dropped_code : code_id list
  }

(** val symbol_kept : dropped -> symbol -> bool **)

let symbol_kept d s = negb (among s d.dropped_symbols)

(** val code_kept : dropped -> code_id -> bool **)

let code_kept d id = negb (among id d.dropped_code)

(** val function_decls_avoid : dropped -> (function_slot * code_id) list -> bool
    **)

let function_decls_avoid d decls =
  forallb (fun decl -> code_kept d (snd decl)) decls

(** val closure_symbols_avoid : dropped -> (function_slot * symbol) list -> bool
    **)

let closure_symbols_avoid d symbols =
  forallb (fun sym -> symbol_kept d (snd sym)) symbols

(** val simple_avoids : dropped -> simple -> bool **)

let simple_avoids d = function Symbol sym -> symbol_kept d sym | _ -> true

(** val simples_avoid : dropped -> simple list -> bool **)

let simples_avoid d args = forallb (simple_avoids d) args

(** val value_slots_avoid : dropped -> (value_slot * simple) list -> bool **)

let value_slots_avoid d slots =
  forallb (fun slot -> simple_avoids d (snd slot)) slots

(** val named_avoids : dropped -> named -> bool **)

let named_avoids d = function
  | Simple s -> simple_avoids d s
  | Prim (_, args) -> simples_avoid d args
  | Set_of_closures (code0, slots) ->
    code_kept d code0 && value_slots_avoid d slots
  | Closure_of_set (_, decls, slots) ->
    function_decls_avoid d decls && value_slots_avoid d slots
  | Project_value_slot (_, closure) -> simple_avoids d closure
  | Project_value_slot_from (_, _, closure) -> simple_avoids d closure
  | Project_function_slot (_, _, closure) -> simple_avoids d closure

(** val static_const_avoids : dropped -> static_const -> bool **)

let static_const_avoids d = function
  | Static_const_set_of_closures (code0, slots) ->
    code_kept d code0 && value_slots_avoid d slots
  | Static_const_closure_of_set (_, symbols, decls, slots) ->
    (closure_symbols_avoid d symbols && function_decls_avoid d decls)
    && value_slots_avoid d slots
  | Static_const_block (_, fields) -> simples_avoid d fields
  | Static_const_boxed_number contents -> simple_avoids d contents
  | Static_const_immutable_string _ -> true
  | Static_const_immutable_value_array elements -> simples_avoid d elements

(** val apply_cont_avoids : dropped -> apply_cont -> bool **)

let apply_cont_avoids d ac = simples_avoid d ac.ac_args

(** val apply_avoids : dropped -> apply_expr -> bool **)

let apply_avoids d ap =
  ((simple_avoids d ap.ap_callee
   &&
   match ap.ap_call_kind with
   | Direct code0 -> code_kept d code0
   | Indirect_unknown_arity -> true
   | Indirect_known_arity code_ids0 -> (
     match code_ids0 with
     | Some code_ids -> forallb (code_kept d) code_ids
     | None -> true))
  && simples_avoid d ap.ap_args)
  && simples_avoid d (exn_extra_simples ap)

(** val expr_avoids : dropped -> expr -> bool **)

let rec expr_avoids d = function
  | Let (_, _, d0, body) -> named_avoids d d0 && expr_avoids d body
  | Let_static (bindings, body) ->
    forallb (binding_avoids d) bindings && expr_avoids d body
  | Let_cont_non_recursive (_, h, body) ->
    handler_avoids d h && expr_avoids d body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_avoids d h)
      handlers
    && expr_avoids d body
  | Apply ap -> apply_avoids d ap
  | Apply_cont ac -> apply_cont_avoids d ac
  | Switch (scrutinee, arms) ->
    simple_avoids d scrutinee
    && forallb (fun arm -> apply_cont_avoids d (snd arm)) arms
  | Invalid -> true

(** val handler_avoids : dropped -> continuation_handler -> bool **)

and handler_avoids d = function
  | Handler (_, _, handler) -> expr_avoids d handler

(** val binding_avoids : dropped -> static_binding -> bool **)

and binding_avoids d = function
  | Bind_code (id, c) -> if code_kept d id then code_avoids d c else true
  | Bind_deleted_code _ -> true
  | Bind_symbol (s, const) ->
    if symbol_kept d s then static_const_avoids d const else true

(** val code_avoids : dropped -> code -> bool **)

and code_avoids d = function Code (_, _, _, _, _, body) -> expr_avoids d body

(** val binding_kept : dropped -> static_binding -> bool **)

let binding_kept d = function
  | Bind_code (id, _) -> code_kept d id
  | Bind_deleted_code _ -> true
  | Bind_symbol (s, _) -> symbol_kept d s

(** val keep_map : ('a1 -> bool) -> ('a1 -> 'a1) -> 'a1 list -> 'a1 list **)

let rec keep_map keep f = function
  | [] -> []
  | a :: l0 -> if keep a then f a :: keep_map keep f l0 else keep_map keep f l0

(** val erase_expr : dropped -> expr -> expr **)

let rec erase_expr d e =
  match e with
  | Let (x, kind, d0, body) -> Let (x, kind, d0, erase_expr d body)
  | Let_static (bindings, body) ->
    Let_static
      (keep_map (binding_kept d) (erase_binding d) bindings, erase_expr d body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, erase_handler d h, erase_expr d body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, erase_handler d h)
          handlers,
        erase_expr d body )
  | _ -> e

(** val erase_handler : dropped -> continuation_handler -> continuation_handler
    **)

and erase_handler d = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, erase_expr d handler)

(** val erase_binding : dropped -> static_binding -> static_binding **)

and erase_binding d binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, erase_code d c)
  | _ -> binding

(** val erase_code : dropped -> code -> code **)

and erase_code d = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, erase_expr d body)

(** val drop_statics : dropped -> expr -> expr option **)

let drop_statics d e = if expr_avoids d e then Some (erase_expr d e) else None

(** val binds_only_kept : dropped -> expr -> bool **)

let rec binds_only_kept d = function
  | Let (_, _, _, body) -> binds_only_kept d body
  | Let_static (bindings, body) ->
    forallb (binding_binds_only_kept d) bindings && binds_only_kept d body
  | Let_cont_non_recursive (_, h, body) ->
    handler_binds_only_kept d h && binds_only_kept d body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_binds_only_kept d h)
      handlers
    && binds_only_kept d body
  | _ -> true

(** val handler_binds_only_kept : dropped -> continuation_handler -> bool **)

and handler_binds_only_kept d = function
  | Handler (_, _, handler) -> binds_only_kept d handler

(** val binding_binds_only_kept : dropped -> static_binding -> bool **)

and binding_binds_only_kept d = function
  | Bind_code (id, c) -> code_kept d id && code_binds_only_kept d c
  | Bind_deleted_code _ -> true
  | Bind_symbol (s, _) -> symbol_kept d s

(** val code_binds_only_kept : dropped -> code -> bool **)

and code_binds_only_kept d = function
  | Code (_, _, _, _, _, body) -> binds_only_kept d body
