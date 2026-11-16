open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaWellFormedSpec
open List
open ListDef

(** val simple_statics_in_scope : symbol list -> simple -> bool **)

let simple_statics_in_scope syms = function
  | Symbol sym -> among sym syms
  | _ -> true

(** val simples_statics_in_scope : symbol list -> simple list -> bool **)

let simples_statics_in_scope syms args =
  forallb (simple_statics_in_scope syms) args

(** val value_slots_statics_in_scope : symbol list -> (value_slot * simple) list
    -> bool **)

let value_slots_statics_in_scope syms value_slots =
  forallb (fun slot -> simple_statics_in_scope syms (snd slot)) value_slots

(** val named_statics_in_scope : symbol list -> code_id list -> named -> bool **)

let named_statics_in_scope syms ids = function
  | Simple s -> simple_statics_in_scope syms s
  | Prim (_, args) -> simples_statics_in_scope syms args
  | Set_of_closures (code0, value_slots) ->
    among code0 ids && value_slots_statics_in_scope syms value_slots
  | Closure_of_set (_, function_decls, value_slots) ->
    all_among (map snd function_decls) ids
    && value_slots_statics_in_scope syms value_slots
  | Project_value_slot (_, closure) -> simple_statics_in_scope syms closure
  | Project_value_slot_from (_, _, closure) ->
    simple_statics_in_scope syms closure
  | Project_function_slot (_, _, closure) ->
    simple_statics_in_scope syms closure

(** val static_const_statics_in_scope : symbol list -> code_id list ->
    static_const -> bool **)

let static_const_statics_in_scope syms ids = function
  | Static_const_set_of_closures (code0, value_slots) ->
    among code0 ids && value_slots_statics_in_scope syms value_slots
  | Static_const_closure_of_set (_, closure_symbols, function_decls, value_slots)
    ->
    (all_among (map snd closure_symbols) syms
    && all_among (map snd function_decls) ids)
    && value_slots_statics_in_scope syms value_slots
  | Static_const_block (_, fields) -> simples_statics_in_scope syms fields
  | Static_const_boxed_number contents -> simple_statics_in_scope syms contents
  | Static_const_immutable_string _ -> true
  | Static_const_immutable_value_array elements ->
    simples_statics_in_scope syms elements

(** val apply_statics_in_scope : symbol list -> code_id list -> apply_expr ->
    bool **)

let apply_statics_in_scope syms ids ap =
  ((simple_statics_in_scope syms ap.ap_callee
   &&
   match ap.ap_call_kind with
   | Direct code0 -> among code0 ids
   | Indirect_unknown_arity -> true
   | Indirect_known_arity code_ids0 -> (
     match code_ids0 with
     | Some code_ids -> all_among code_ids ids
     | None -> true))
  && simples_statics_in_scope syms ap.ap_args)
  && simples_statics_in_scope syms (exn_extra_simples ap)

(** val apply_cont_statics_in_scope : symbol list -> apply_cont -> bool **)

let apply_cont_statics_in_scope syms ac =
  simples_statics_in_scope syms ac.ac_args

(** val is_closure_const : static_const -> bool **)

let is_closure_const = function
  | Static_const_set_of_closures (_, _) -> true
  | Static_const_closure_of_set (_, _, _, _) -> true
  | _ -> false

(** val closure_symbols_of_binding : static_binding -> symbol list **)

let closure_symbols_of_binding = function
  | Bind_symbol (s, const) -> if is_closure_const const then s :: [] else []
  | _ -> []

(** val declared_code_ids_of_binding : static_binding -> code_id list **)

let declared_code_ids_of_binding = function
  | Bind_code (id, _) -> id :: []
  | Bind_deleted_code id -> id :: []
  | Bind_symbol (_, _) -> []

(** val group_consts_in_scope : symbol list -> symbol list -> code_id list ->
    static_binding list -> bool **)

let rec group_consts_in_scope before all ids = function
  | [] -> true
  | s0 :: bindings0 -> (
    match s0 with
    | Bind_symbol (s, const) ->
      if is_closure_const const
      then
        static_const_statics_in_scope all ids const
        && group_consts_in_scope before all ids bindings0
      else
        static_const_statics_in_scope before ids const
        && group_consts_in_scope (s :: before) all ids bindings0
    | _ -> group_consts_in_scope before all ids bindings0)

(** val expr_statics_in_scope : symbol list -> code_id list -> expr -> bool **)

let rec expr_statics_in_scope syms ids = function
  | Let (_, _, defining_expr, body) ->
    named_statics_in_scope syms ids defining_expr
    && expr_statics_in_scope syms ids body
  | Let_static (bindings, body) ->
    let all = app (flat_map symbols_of_binding bindings) syms in
    let ids' = app (flat_map declared_code_ids_of_binding bindings) ids in
    (group_consts_in_scope
       (app (flat_map closure_symbols_of_binding bindings) syms)
       all ids' bindings
    && forallb (binding_statics_in_scope all ids') bindings)
    && expr_statics_in_scope all ids' body
  | Let_cont_non_recursive (_, h, body) ->
    handler_statics_in_scope
      (app (static_names symbols_of_binding body) syms)
      (app (static_names declared_code_ids_of_binding body) ids)
      h
    && expr_statics_in_scope syms ids body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_statics_in_scope
          (app (static_names symbols_of_binding body) syms)
          (app (static_names declared_code_ids_of_binding body) ids)
          h)
      handlers
    && expr_statics_in_scope syms ids body
  | Apply ap -> apply_statics_in_scope syms ids ap
  | Apply_cont ac -> apply_cont_statics_in_scope syms ac
  | Switch (scrutinee, arms) ->
    simple_statics_in_scope syms scrutinee
    && forallb (fun arm -> apply_cont_statics_in_scope syms (snd arm)) arms
  | Invalid -> true

(** val handler_statics_in_scope : symbol list -> code_id list ->
    continuation_handler -> bool **)

and handler_statics_in_scope syms ids = function
  | Handler (_, _, handler) -> expr_statics_in_scope syms ids handler

(** val binding_statics_in_scope : symbol list -> code_id list -> static_binding
    -> bool **)

and binding_statics_in_scope syms ids = function
  | Bind_code (_, c) -> code_statics_in_scope syms ids c
  | _ -> true

(** val code_statics_in_scope : symbol list -> code_id list -> code -> bool **)

and code_statics_in_scope syms ids = function
  | Code (_, _, _, _, _, body) -> expr_statics_in_scope syms ids body

(** val statics_in_scope : symbol list -> expr -> bool **)

let statics_in_scope imports e = expr_statics_in_scope imports [] e

(** val static_names_apart : symbol list -> expr -> bool **)

let static_names_apart imports e =
  negb (existsb (fun s -> among s imports) (static_names symbols_of_binding e))
  && distinct (static_names declared_code_ids_of_binding e)

(** val bound_continuations : expr -> continuation list **)

let rec bound_continuations = function
  | Let (_, _, _, body) -> bound_continuations body
  | Let_static (bindings, body) ->
    app
      (flat_map binding_bound_continuations bindings)
      (bound_continuations body)
  | Let_cont_non_recursive (k, h, body) ->
    k :: app (handler_bound_continuations h) (bound_continuations body)
  | Let_cont_recursive (handlers, body) ->
    app (map fst handlers)
      (app
         (flat_map
            (fun pat ->
              let _, h = pat in
              handler_bound_continuations h)
            handlers)
         (bound_continuations body))
  | _ -> []

(** val handler_bound_continuations : continuation_handler -> continuation list
    **)

and handler_bound_continuations = function
  | Handler (_, _, handler) -> bound_continuations handler

(** val binding_bound_continuations : static_binding -> continuation list **)

and binding_bound_continuations = function
  | Bind_code (_, c) -> code_bound_continuations c
  | _ -> []

(** val code_bound_continuations : code -> continuation list **)

and code_bound_continuations = function
  | Code (_, _, return_continuation, exn_continuation, _, body) ->
    return_continuation :: exn_continuation :: bound_continuations body

(** val interface_conts_not_rebound : unit_interface -> expr -> bool **)

let interface_conts_not_rebound u e =
  negb (among u.unit_return (bound_continuations e))
  && negb (among u.unit_exn (bound_continuations e))
