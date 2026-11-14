open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaUnusedContsSpec
open ListDef
open PeanoNat

(** val never_returns_b : apply_expr -> code -> bool **)

let never_returns_b ap c =
  ((negb
      (cont_free_b
         (code_return_continuation c)
         (remove_unused_conts (code_body c)))
   && negb (code_is_tupled c))
  && Nat.leb (length (code_params_arity c)) (length ap.ap_args_arity))
  && negb (Pos.eqb ap.ap_continuation ap.ap_exn_continuation)

(** val invalid_params : code -> nat -> variable list option **)

let invalid_params c count =
  let names = code_my_closure c :: code_params c in
  if Nat.leb count (length names) then Some (firstn count names) else None

(** val cut_return : (code_id * code) list -> apply_expr -> expr **)

let cut_return known ap =
  match ap.ap_call_kind with
  | Direct called -> (
    match lookup called known with
    | Some c -> (
      let kinds = unarize ap.ap_return_arity in
      match invalid_params c (length kinds) with
      | Some params ->
        if never_returns_b ap c
        then
          Let_cont_non_recursive
            ( ap.ap_continuation,
              Handler (params, make_handler_metadata kinds, Invalid),
              Apply ap )
        else Apply ap
      | None -> Apply ap)
    | None -> Apply ap)
  | Indirect -> Apply ap

(** val returns_to_invalid : continuation -> continuation_handler -> expr ->
    bool **)

let returns_to_invalid k h body =
  let (Handler (_, _, handler)) = h in
  match handler with
  | Invalid -> (
    match body with Apply ap -> Pos.eqb ap.ap_continuation k | _ -> false)
  | _ -> false

(** val cut_returns : (code_id * code) list -> expr -> expr **)

let rec cut_returns known e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, cut_returns known body)
  | Let_static (bindings, body) ->
    let bindings0 = map (cut_returns_in_binding known) bindings in
    Let_static (bindings0, cut_returns (codes_of bindings0 known) body)
  | Let_cont_non_recursive (k, h, body) ->
    if returns_to_invalid k h body
    then e
    else
      Let_cont_non_recursive
        (k, cut_returns_in_handler known h, cut_returns known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, cut_returns_in_handler known h)
          handlers,
        cut_returns known body )
  | Apply ap -> cut_return known ap
  | _ -> e

(** val cut_returns_in_handler : (code_id * code) list -> continuation_handler
    -> continuation_handler **)

and cut_returns_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, cut_returns known handler)

(** val cut_returns_in_binding : (code_id * code) list -> static_binding ->
    static_binding **)

and cut_returns_in_binding known binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, cut_returns_in_code known c)
  | _ -> binding

(** val cut_returns_in_code : (code_id * code) list -> code -> code **)

and cut_returns_in_code known = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, cut_returns known body)
