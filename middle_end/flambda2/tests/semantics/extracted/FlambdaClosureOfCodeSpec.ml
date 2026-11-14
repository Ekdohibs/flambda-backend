open Datatypes
open FlambdaKnownSymbolsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaSetsOfCodeSpec
open FlambdaStaticsInScopeSpec
open FlambdaWellFormedSpec
open List
open ListDef

(** val set_with_code : set_made list -> code_id -> function_slot -> set_made
    option **)

let set_with_code sets c f = find (fun set -> has_code_at c f (fst set)) sets

(** val sibling_code : set_made list -> code_id -> function_slot ->
    function_slot -> code_id option **)

let sibling_code sets c f g =
  match set_with_code sets c f with
  | Some set -> (
    match lookup g (fst set) with
    | Some d ->
      if sets_made_have c f (Code_at (g, d)) sets then Some d else None
    | None -> None)
  | None -> None

(** val slot_held : set_made list -> code_id -> function_slot -> value_slot ->
    simple option **)

let slot_held sets c f slot =
  match set_with_code sets c f with
  | Some set -> (
    match lookup slot (snd set) with
    | Some a ->
      if closed_simple a && sets_made_have c f (Slot_holds (slot, a)) sets
      then Some a
      else None
    | None -> None)
  | None -> None

type closure_codes = code_id finite_map

(** val closure_code_made : set_made list -> closure_codes -> named -> code_id
    option **)

let closure_code_made sets known = function
  | Project_function_slot (move_from, move_to, closure) -> (
    match closure with
    | Var x -> (
      match lookup x known with
      | Some c -> sibling_code sets c move_from move_to
      | None -> None)
    | _ -> None)
  | _ -> None

(** val codes_under_let : set_made list -> closure_codes -> variable -> named ->
    closure_codes **)

let codes_under_let sets known x d =
  match closure_code_made sets known d with
  | Some c -> (x, c) :: unbind x known
  | None -> unbind x known

(** val read_code_slot : set_made list -> symbol list -> closure_codes -> named
    -> named **)

let read_code_slot sets syms known d =
  match d with
  | Project_value_slot_from (project_from, slot, closure) -> (
    match closure with
    | Var x -> (
      match lookup x known with
      | Some c -> (
        match slot_held sets c project_from slot with
        | Some a -> if simple_statics_in_scope syms a then Simple a else d
        | None -> d)
      | None -> d)
    | _ -> d)
  | _ -> d

(** val call_code_closure : closure_codes -> apply_expr -> apply_expr **)

let call_code_closure known ap =
  match ap.ap_call_kind with
  | Direct _ -> ap
  | Indirect -> (
    match ap.ap_callee with
    | Var x -> (
      match lookup x known with
      | Some c ->
        { ap_callee = ap.ap_callee;
          ap_call_kind = Direct c;
          ap_args = ap.ap_args;
          ap_continuation = ap.ap_continuation;
          ap_exn_continuation = ap.ap_exn_continuation;
          ap_exn_extra_args = ap.ap_exn_extra_args;
          ap_args_arity = ap.ap_args_arity;
          ap_return_arity = ap.ap_return_arity
        }
      | None -> ap)
    | _ -> ap)

(** val forget_code_vars : variable list -> closure_codes -> closure_codes **)

let rec forget_code_vars xs known =
  match xs with
  | [] -> known
  | x :: xs0 -> forget_code_vars xs0 (unbind x known)

(** val know_in_body : set_made list -> symbol list -> closure_codes -> expr ->
    expr **)

let rec know_in_body sets syms known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        read_code_slot sets syms known d,
        know_in_body sets syms (codes_under_let sets known x d) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, know_in_body sets syms known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        know_in_body_handler sets syms known h,
        know_in_body sets syms known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, know_in_body_handler sets syms known h)
          handlers,
        know_in_body sets syms known body )
  | Apply ap -> Apply (call_code_closure known ap)
  | _ -> e

(** val know_in_body_handler : set_made list -> symbol list -> closure_codes ->
    continuation_handler -> continuation_handler **)

and know_in_body_handler sets syms known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        know_in_body sets syms (forget_code_vars params known) handler )

(** val know_in_code : set_made list -> symbol list -> code_id -> code -> code **)

let know_in_code sets syms id c =
  match c with
  | Code (params, my_closure, ret, exn, metadata, body) ->
    if metadata.cm_is_stub || among my_closure params
    then c
    else
      Code
        ( params,
          my_closure,
          ret,
          exn,
          metadata,
          know_in_body sets syms ((my_closure, id) :: []) body )

(** val know_code_in_binding : set_made list -> symbol list -> static_binding ->
    static_binding **)

let know_code_in_binding sets syms binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, know_in_code sets syms id c)
  | _ -> binding

(** val know_code_closures : set_made list -> symbol list -> expr -> expr **)

let rec know_code_closures sets syms e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, know_code_closures sets syms body)
  | Let_static (bindings, body) ->
    let all = app (flat_map symbols_of_binding bindings) syms in
    Let_static
      ( map (know_code_in_binding sets all) bindings,
        know_code_closures sets all body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        know_code_in_handler sets
          (app (static_names symbols_of_binding body) syms)
          h,
        know_code_closures sets syms body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            ( k,
              know_code_in_handler sets
                (app (static_names symbols_of_binding body) syms)
                h ))
          handlers,
        know_code_closures sets syms body )
  | _ -> e

(** val know_code_in_handler : set_made list -> symbol list ->
    continuation_handler -> continuation_handler **)

and know_code_in_handler sets syms = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, know_code_closures sets syms handler)

(** val groups_whole : expr -> bool **)

let rec groups_whole = function
  | Let (_, _, _, body) -> groups_whole body
  | Let_static (bindings, body) ->
    (forallb (static_set_whole bindings) bindings
    && forallb binding_groups_whole bindings)
    && groups_whole body
  | Let_cont_non_recursive (_, h, body) ->
    handler_groups_whole h && groups_whole body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_groups_whole h)
      handlers
    && groups_whole body
  | _ -> true

(** val handler_groups_whole : continuation_handler -> bool **)

and handler_groups_whole = function
  | Handler (_, _, handler) -> groups_whole handler

(** val binding_groups_whole : static_binding -> bool **)

and binding_groups_whole = function
  | Bind_code (_, c) -> code_groups_whole c
  | _ -> true

(** val code_groups_whole : code -> bool **)

and code_groups_whole = function
  | Code (_, _, _, _, _, body) -> groups_whole body

(** val know_own_closures : expr -> expr **)

let know_own_closures e =
  if groups_whole e then know_code_closures (sets_made e) [] e else e
