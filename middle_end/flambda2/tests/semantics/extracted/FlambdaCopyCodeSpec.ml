open Datatypes
open FlambdaMapSpec
open FlambdaModel
open List
open ListDef

(** val copy_of : (code_id * code_id) list -> code_id -> code_id **)

let copy_of ids id = match lookup id ids with Some id' -> id' | None -> id

(** val copy_decls : (code_id * code_id) list -> (function_slot * code_id) list
    -> (function_slot * code_id) list **)

let copy_decls ids function_decls =
  map (fun decl -> fst decl, copy_of ids (snd decl)) function_decls

(** val copy_static_const : (code_id * code_id) list -> static_const ->
    static_const **)

let copy_static_const ids const =
  match const with
  | Static_const_set_of_closures (code0, value_slots) ->
    Static_const_set_of_closures (copy_of ids code0, value_slots)
  | Static_const_closure_of_set
      (slot, closure_symbols, function_decls, value_slots) ->
    Static_const_closure_of_set
      (slot, closure_symbols, copy_decls ids function_decls, value_slots)
  | _ -> const

(** val with_call_kind : call_kind -> apply_expr -> apply_expr **)

let with_call_kind kind ap =
  { ap_callee = ap.ap_callee;
    ap_call_kind = kind;
    ap_args = ap.ap_args;
    ap_continuation = ap.ap_continuation;
    ap_exn_continuation = ap.ap_exn_continuation;
    ap_exn_extra_args = ap.ap_exn_extra_args;
    ap_args_arity = ap.ap_args_arity;
    ap_return_arity = ap.ap_return_arity
  }

(** val names_copied : (code_id * code_id) list -> code_id list -> bool **)

let names_copied ids code_ids =
  existsb (fun id -> among id (map fst ids)) code_ids

(** val copy_call : (code_id * code_id) list -> symbol list -> apply_expr ->
    apply_expr **)

let copy_call ids symbols ap =
  match ap.ap_call_kind with
  | Direct id -> (
    match ap.ap_callee with
    | Symbol s ->
      if among s symbols
      then with_call_kind (Direct (copy_of ids id)) ap
      else ap
    | _ -> (
      match lookup id ids with
      | Some _ -> with_call_kind Indirect_unknown_arity ap
      | None -> ap))
  | Indirect_unknown_arity -> ap
  | Indirect_known_arity code_ids0 -> (
    match code_ids0 with
    | Some code_ids ->
      if names_copied ids code_ids
      then with_call_kind (Indirect_known_arity None) ap
      else ap
    | None -> ap)

(** val copy_expr : (code_id * code_id) list -> symbol list -> expr -> expr **)

let rec copy_expr ids symbols e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, copy_expr ids symbols body)
  | Let_static (bindings, body) ->
    Let_static
      (flat_map (copy_binding ids symbols) bindings, copy_expr ids symbols body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, copy_handler ids symbols h, copy_expr ids symbols body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, copy_handler ids symbols h)
          handlers,
        copy_expr ids symbols body )
  | Apply ap -> Apply (copy_call ids symbols ap)
  | _ -> e

(** val copy_handler : (code_id * code_id) list -> symbol list ->
    continuation_handler -> continuation_handler **)

and copy_handler ids symbols = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, copy_expr ids symbols handler)

(** val copy_binding : (code_id * code_id) list -> symbol list -> static_binding
    -> static_binding list **)

and copy_binding ids symbols binding =
  match binding with
  | Bind_code (id, c) ->
    let c' = copy_code ids symbols c in
    Bind_code (id, c')
    ::
    (match lookup id ids with
    | Some id' -> Bind_code (id', c') :: []
    | None -> [])
  | Bind_deleted_code _ -> binding :: []
  | Bind_symbol (s, const) ->
    Bind_symbol
      (s, if among s symbols then copy_static_const ids const else const)
    :: []

(** val copy_code : (code_id * code_id) list -> symbol list -> code -> code **)

and copy_code ids symbols = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, copy_expr ids symbols body)

(** val binds_no_code : code_id list -> expr -> bool **)

let rec binds_no_code new0 = function
  | Let (_, _, _, body) -> binds_no_code new0 body
  | Let_static (bindings, body) ->
    forallb (binding_binds_no_code new0) bindings && binds_no_code new0 body
  | Let_cont_non_recursive (_, h, body) ->
    handler_binds_no_code new0 h && binds_no_code new0 body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_binds_no_code new0 h)
      handlers
    && binds_no_code new0 body
  | _ -> true

(** val handler_binds_no_code : code_id list -> continuation_handler -> bool **)

and handler_binds_no_code new0 = function
  | Handler (_, _, handler) -> binds_no_code new0 handler

(** val binding_binds_no_code : code_id list -> static_binding -> bool **)

and binding_binds_no_code new0 = function
  | Bind_code (id, c) -> negb (among id new0) && code_binds_no_code new0 c
  | _ -> true

(** val code_binds_no_code : code_id list -> code -> bool **)

and code_binds_no_code new0 = function
  | Code (_, _, _, _, _, body) -> binds_no_code new0 body

(** val copy_ok : (code_id * code_id) list -> expr -> bool **)

let copy_ok ids e =
  (distinct (map fst ids) && distinct (map snd ids))
  && binds_no_code (map snd ids) e

(** val copy_static_code : (code_id * code_id) list -> symbol list -> expr ->
    expr option **)

let copy_static_code ids symbols e =
  if copy_ok ids e then Some (copy_expr ids symbols e) else None

(** val with_copies : ((code_id * code_id) list * symbol list) list -> expr ->
    expr **)

let rec with_copies copies e =
  match copies with
  | [] -> e
  | p :: copies0 ->
    let ids, symbols = p in
    with_copies copies0
      (match copy_static_code ids symbols e with Some e' -> e' | None -> e)
