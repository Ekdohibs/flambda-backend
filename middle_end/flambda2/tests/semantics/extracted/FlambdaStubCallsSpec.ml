open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open ListDef
open PeanoNat

(** val full_call_of_b : apply_expr -> code -> bool **)

let full_call_of_b ap c =
  ((negb (code_is_stub c) && negb (code_is_tupled c))
  && match ap.ap_args_arity with [] -> false | _ :: _ -> true)
  && Nat.eqb (length ap.ap_args_arity) (length (code_params_arity c))

(** val demote_full_call : (code_id * code) list -> apply_expr -> apply_expr **)

let demote_full_call known ap =
  match ap.ap_call_kind with
  | Direct called -> (
    match lookup called known with
    | Some c ->
      if full_call_of_b ap c
      then
        { ap_callee = ap.ap_callee;
          ap_call_kind = Indirect_known_arity None;
          ap_args = ap.ap_args;
          ap_continuation = ap.ap_continuation;
          ap_exn_continuation = ap.ap_exn_continuation;
          ap_exn_extra_args = ap.ap_exn_extra_args;
          ap_args_arity = ap.ap_args_arity;
          ap_return_arity = ap.ap_return_arity
        }
      else ap
    | None -> ap)
  | _ -> ap

(** val demote_full_calls : (code_id * code) list -> expr -> expr **)

let rec demote_full_calls known e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, demote_full_calls known body)
  | Let_static (bindings, body) ->
    Let_static (bindings, demote_full_calls known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, demote_full_calls_in_handler known h, demote_full_calls known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, demote_full_calls_in_handler known h)
          handlers,
        demote_full_calls known body )
  | Apply ap -> Apply (demote_full_call known ap)
  | _ -> e

(** val demote_full_calls_in_handler : (code_id * code) list ->
    continuation_handler -> continuation_handler **)

and demote_full_calls_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, demote_full_calls known handler)

(** val demote_full_calls_in_code : (code_id * code) list -> code -> code **)

let demote_full_calls_in_code known = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, demote_full_calls known body)

(** val demote_full_in_stub_binding : (code_id * code) list -> static_binding ->
    static_binding **)

let demote_full_in_stub_binding known binding =
  match binding with
  | Bind_code (id, c) ->
    if code_is_stub c
    then Bind_code (id, demote_full_calls_in_code known c)
    else binding
  | _ -> binding

(** val demote_full_in_stubs : (code_id * code) list -> expr -> expr **)

let rec demote_full_in_stubs known e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, demote_full_in_stubs known body)
  | Let_static (bindings, body) ->
    let known0 = codes_of bindings known in
    Let_static
      ( map (demote_full_in_stub_binding known0) bindings,
        demote_full_in_stubs known0 body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        demote_full_in_stubs_of_handler known h,
        demote_full_in_stubs known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, demote_full_in_stubs_of_handler known h)
          handlers,
        demote_full_in_stubs known body )
  | _ -> e

(** val demote_full_in_stubs_of_handler : (code_id * code) list ->
    continuation_handler -> continuation_handler **)

and demote_full_in_stubs_of_handler known = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, demote_full_in_stubs known handler)

(** val demote_full_stub_calls : expr -> expr **)

let demote_full_stub_calls e = demote_full_in_stubs [] e
