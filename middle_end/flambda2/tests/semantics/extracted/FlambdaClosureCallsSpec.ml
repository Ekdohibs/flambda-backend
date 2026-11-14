open BinPos
open FlambdaMapSpec
open FlambdaModel
open ListDef

type made_closure = function_slot * (function_slot * code_id) list

type made_closures = made_closure finite_map

(** val closure_made : made_closures -> named -> made_closure option **)

let closure_made known = function
  | Closure_of_set (slot, function_decls, _) -> Some (slot, function_decls)
  | Project_function_slot (move_from, move_to, closure0) -> (
    match closure0 with
    | Var closure -> (
      match lookup closure known with
      | Some m ->
        let slot, function_decls = m in
        if Pos.eqb slot move_from then Some (move_to, function_decls) else None
      | None -> None)
    | _ -> None)
  | _ -> None

(** val made_under_let : made_closures -> variable -> named -> made_closures **)

let made_under_let known x d =
  match closure_made known d with
  | Some closure -> (x, closure) :: unbind x known
  | None -> unbind x known

(** val unbind_params : variable list -> made_closures -> made_closures **)

let rec unbind_params params known =
  match params with
  | [] -> known
  | param :: params0 -> unbind_params params0 (unbind param known)

(** val made_callee_code : made_closures -> simple -> code_id option **)

let made_callee_code known = function
  | Var x -> (
    match lookup x known with
    | Some m ->
      let slot, function_decls = m in
      lookup slot function_decls
    | None -> None)
  | _ -> None

(** val call_made_closure : made_closures -> apply_expr -> apply_expr **)

let call_made_closure known ap =
  match ap.ap_call_kind with
  | Direct _ -> ap
  | Indirect -> (
    match made_callee_code known ap.ap_callee with
    | Some code ->
      { ap_callee = ap.ap_callee;
        ap_call_kind = Direct code;
        ap_args = ap.ap_args;
        ap_continuation = ap.ap_continuation;
        ap_exn_continuation = ap.ap_exn_continuation;
        ap_exn_extra_args = ap.ap_exn_extra_args;
        ap_args_arity = ap.ap_args_arity;
        ap_return_arity = ap.ap_return_arity
      }
    | None -> ap)

(** val call_closures_knowing : made_closures -> expr -> expr **)

let rec call_closures_knowing known e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, call_closures_knowing (made_under_let known x d) body)
  | Let_static (bindings, body) ->
    Let_static (bindings, call_closures_knowing known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, call_closures_in_handler known h, call_closures_knowing known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, call_closures_in_handler known h)
          handlers,
        call_closures_knowing known body )
  | Apply ap -> Apply (call_made_closure known ap)
  | _ -> e

(** val call_closures_in_handler : made_closures -> continuation_handler ->
    continuation_handler **)

and call_closures_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        call_closures_knowing (unbind_params params known) handler )

(** val call_made_closures : expr -> expr **)

let call_made_closures e = call_closures_knowing [] e
