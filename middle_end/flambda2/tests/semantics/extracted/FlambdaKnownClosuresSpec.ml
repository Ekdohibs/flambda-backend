open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaValueSpec
open List

type known_closures = (symbol * static_const) list

(** val static_closure_made : static_binding -> (symbol * static_const) list **)

let static_closure_made = function
  | Bind_symbol (s, const) -> (
    match const with
    | Static_const_set_of_closures (_, _) -> (s, const) :: []
    | Static_const_closure_of_set (_, _, _, _) -> (s, const) :: []
    | _ -> [])
  | _ -> []

(** val closures_under_statics : known_closures -> static_binding list ->
    known_closures **)

let closures_under_statics known bindings =
  app (flat_map static_closure_made bindings) known

(** val forget_slot_var : variable -> known_closures -> known_closures **)

let forget_slot_var x known =
  filter
    (fun entry ->
      negb (var_in_static_binding x (Bind_symbol (fst entry, snd entry))))
    known

(** val forget_slot_vars : variable list -> known_closures -> known_closures **)

let rec forget_slot_vars xs known =
  match xs with
  | [] -> known
  | x :: xs0 -> forget_slot_vars xs0 (forget_slot_var x known)

(** val static_closure_slots : static_const -> (function_slot * (value_slot *
    simple) list) option **)

let static_closure_slots = function
  | Static_const_set_of_closures (_, slots) ->
    Some (unnamed_function_slot, slots)
  | Static_const_closure_of_set (slot, _, _, slots) -> Some (slot, slots)
  | _ -> None

(** val slot_read : known_closures -> function_slot option -> value_slot ->
    simple -> simple option **)

let slot_read known project_from slot = function
  | Symbol s -> (
    match lookup s known with
    | Some const -> (
      match static_closure_slots const with
      | Some p -> (
        let at_slot, slots = p in
        match project_from with
        | Some from -> if Pos.eqb at_slot from then lookup slot slots else None
        | None -> lookup slot slots)
      | None -> None)
    | None -> None)
  | _ -> None

(** val read_slot_named : known_closures -> named -> named **)

let read_slot_named known d =
  match d with
  | Project_value_slot (slot, closure) -> (
    match slot_read known None slot closure with
    | Some contents -> Simple contents
    | None -> d)
  | Project_value_slot_from (project_from, slot, closure) -> (
    match slot_read known (Some project_from) slot closure with
    | Some contents -> Simple contents
    | None -> d)
  | _ -> d

(** val static_closure_code : static_const -> code_id option **)

let static_closure_code = function
  | Static_const_set_of_closures (code, _) -> Some code
  | Static_const_closure_of_set (slot, _, function_decls, _) ->
    lookup slot function_decls
  | _ -> None

(** val callee_code : known_closures -> simple -> code_id option **)

let callee_code known = function
  | Symbol s -> (
    match lookup s known with
    | Some const -> static_closure_code const
    | None -> None)
  | _ -> None

(** val call_known_closure : known_closures -> apply_expr -> apply_expr **)

let call_known_closure known ap =
  match ap.ap_call_kind with
  | Indirect_unknown_arity -> (
    match callee_code known ap.ap_callee with
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
  | _ -> ap

(** val use_closures_knowing : known_closures -> expr -> expr **)

let rec use_closures_knowing known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        read_slot_named known d,
        use_closures_knowing (forget_slot_var x known) body )
  | Let_static (bindings, body) ->
    Let_static
      ( bindings,
        use_closures_knowing (closures_under_statics known bindings) body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, use_closures_in_handler known h, use_closures_knowing known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, use_closures_knowing known body)
  | Apply ap -> Apply (call_known_closure known ap)
  | _ -> e

(** val use_closures_in_handler : known_closures -> continuation_handler ->
    continuation_handler **)

and use_closures_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        use_closures_knowing (forget_slot_vars params known) handler )

(** val use_known_closures : expr -> expr **)

let use_known_closures e = use_closures_knowing [] e
