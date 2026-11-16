open BinNums
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaSemanticsSpec
open FlambdaValueSpec
open List
open ListDef
open OxCamlOption

(** val closed_simple : simple -> bool **)

let closed_simple = function Var _ -> false | _ -> true

(** val closed_static_const : static_const -> bool **)

let closed_static_const = function
  | Static_const_set_of_closures (_, value_slots) ->
    forallb (fun slot -> closed_simple (snd slot)) value_slots
  | Static_const_closure_of_set (_, _, _, value_slots) ->
    forallb (fun slot -> closed_simple (snd slot)) value_slots
  | Static_const_block (_, fields) -> forallb closed_simple fields
  | Static_const_boxed_number contents -> closed_simple contents
  | Static_const_immutable_string _ -> true
  | Static_const_immutable_value_array elements ->
    forallb closed_simple elements

(** val known_of_binding : static_binding -> (symbol * static_const) list **)

let known_of_binding = function
  | Bind_symbol (s, const0) ->
    if closed_static_const const0
    then
      match eval_static_const [] const0 with
      | Some _ -> (s, const0) :: []
      | None -> []
    else []
  | _ -> []

(** val known_bindings : expr -> (symbol * static_const) list **)

let rec known_bindings = function
  | Let (_, _, _, body) -> known_bindings body
  | Let_static (bindings, body) ->
    app (flat_map known_of_binding bindings) (known_bindings body)
  | Let_cont_non_recursive (_, h, body) ->
    app (known_bindings_of_handler h) (known_bindings body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           known_bindings_of_handler h)
         handlers)
      (known_bindings body)
  | _ -> []

(** val known_bindings_of_handler : continuation_handler -> (symbol *
    static_const) list **)

and known_bindings_of_handler = function
  | Handler (_, _, handler) -> known_bindings handler

(** val known_value : (symbol * static_const) -> (symbol *
    statically_allocated_value) list **)

let known_value binding =
  match eval_static_const [] (snd binding) with
  | Some allocated -> (fst binding, allocated) :: []
  | None -> []

(** val known_statics : (symbol * static_const) list -> static_env **)

let known_statics known =
  { static_code = []; static_values = flat_map known_value known }

(** val unknown_field : const **)

let unknown_field = Naked_immediate Z0

(** val set_of_binding : static_binding -> (symbol * static_const) list **)

let set_of_binding = function
  | Bind_symbol (s, const0) -> (
    match const0 with
    | Static_const_set_of_closures (code0, _) ->
      (s, Static_const_set_of_closures (code0, [])) :: []
    | Static_const_closure_of_set (slot, symbols, decls, _) -> (
      let set = Static_const_closure_of_set (slot, symbols, decls, []) in
      match eval_static_const [] set with
      | Some _ -> (s, set) :: []
      | None -> [])
    | Static_const_block (tag, fields) ->
      ( s,
        Static_const_block
          ( tag,
            map
              (fun field ->
                if closed_simple field then field else Const unknown_field)
              fields ) )
      :: []
    | _ -> [])
  | _ -> []

(** val known_sets : expr -> (symbol * static_const) list **)

let rec known_sets = function
  | Let (_, _, _, body) -> known_sets body
  | Let_static (bindings, body) ->
    app (flat_map set_of_binding bindings) (known_sets body)
  | Let_cont_non_recursive (_, h, body) ->
    app (known_sets_of_handler h) (known_sets body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           known_sets_of_handler h)
         handlers)
      (known_sets body)
  | _ -> []

(** val known_sets_of_handler : continuation_handler -> (symbol * static_const)
    list **)

and known_sets_of_handler = function
  | Handler (_, _, handler) -> known_sets handler

(** val simple_known : static_env -> simple -> bool **)

let simple_known statics = function
  | Var _ -> false
  | Symbol sym -> is_some (lookup sym statics.static_values)
  | Const _ -> true

(** val reads_string : prim -> bool **)

let reads_string = function
  | Binary p0 -> (
    match p0 with String_or_bigstring_load _ -> true | _ -> false)
  | _ -> false

(** val named_known : static_env -> named -> bool **)

let named_known statics = function
  | Prim (p, args) ->
    (prim_may_fold p && negb (reads_string p))
    && forallb (simple_known statics) args
  | Project_value_slot (_, closure) -> simple_known statics closure
  | Project_value_slot_from (_, _, closure) -> simple_known statics closure
  | Project_function_slot (_, _, closure) -> simple_known statics closure
  | _ -> false

(** val fold_known_named : static_env -> named -> named **)

let fold_known_named statics d =
  if named_known statics d
  then
    match eval_named statics [] d with
    | Some v -> (
      match v with
      | Value_const c -> Simple (Const c)
      | Value_symbol s -> Simple (Symbol s)
      | _ -> d)
    | None -> d
  else d

(** val fold_function_slot : static_env -> named -> named **)

let fold_function_slot sets d =
  match d with
  | Project_function_slot (_, _, closure) -> (
    match closure with
    | Symbol _ -> (
      match eval_named sets [] d with
      | Some v -> ( match v with Value_symbol s -> Simple (Symbol s) | _ -> d)
      | None -> d)
    | _ -> d)
  | _ -> d

(** val fold_block_load : static_env -> named -> named **)

let fold_block_load sets d =
  match d with
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Block_load (_, mut, _) -> (
        match mut with
        | Mutable -> d
        | _ -> (
          match args with
          | [] -> d
          | s :: l -> (
            match s with
            | Symbol _ -> (
              match l with
              | [] -> (
                match eval_named sets [] d with
                | Some v -> (
                  match v with
                  | Value_const c -> Simple (Const c)
                  | Value_symbol s0 -> Simple (Symbol s0)
                  | _ -> d)
                | None -> d)
              | _ :: _ -> d)
            | _ -> d)))
      | _ -> d)
    | _ -> d)
  | _ -> d

(** val fold_known_apply : static_env -> apply_expr -> apply_expr **)

let fold_known_apply sets ap =
  match ap.ap_call_kind with
  | Direct _ -> ap
  | _ -> (
    match ap.ap_callee with
    | Symbol s -> (
      match closure_of sets (Value_symbol s) with
      | Some p ->
        let code0, _ = p in
        { ap_callee = ap.ap_callee;
          ap_call_kind = Direct code0;
          ap_args = ap.ap_args;
          ap_continuation = ap.ap_continuation;
          ap_exn_continuation = ap.ap_exn_continuation;
          ap_exn_extra_args = ap.ap_exn_extra_args;
          ap_args_arity = ap.ap_args_arity;
          ap_return_arity = ap.ap_return_arity
        }
      | None -> ap)
    | _ -> ap)

(** val fold_known : static_env -> static_env -> expr -> expr **)

let rec fold_known statics sets e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        fold_block_load sets
          (fold_function_slot sets (fold_known_named statics d)),
        fold_known statics sets body )
  | Let_static (bindings, body) ->
    Let_static
      ( map (fold_known_binding statics sets) bindings,
        fold_known statics sets body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, fold_known_handler statics sets h, fold_known statics sets body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, fold_known_handler statics sets h)
          handlers,
        fold_known statics sets body )
  | Apply ap -> Apply (fold_known_apply sets ap)
  | _ -> e

(** val fold_known_handler : static_env -> static_env -> continuation_handler ->
    continuation_handler **)

and fold_known_handler statics sets = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, fold_known statics sets handler)

(** val fold_known_binding : static_env -> static_env -> static_binding ->
    static_binding **)

and fold_known_binding statics sets binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, fold_known_code statics sets c)
  | _ -> binding

(** val fold_known_code : static_env -> static_env -> code -> code **)

and fold_known_code statics sets = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, fold_known statics sets body)

(** val binding_bound_as : (symbol * static_const) list -> (symbol *
    static_const) list -> static_binding -> bool **)

let binding_bound_as known sets binding =
  match binding with
  | Bind_symbol (s, const0) -> (
    (match lookup s known with
      | Some expected ->
        closed_static_const const0
        && if static_const_eq_dec const0 expected then true else false
      | None -> true)
    &&
    match lookup s sets with
    | Some expected -> (
      match set_of_binding binding with
      | [] -> false
      | p :: l -> (
        let _, set = p in
        match l with
        | [] -> if static_const_eq_dec set expected then true else false
        | _ :: _ -> false))
    | None -> true)
  | _ -> true

(** val bound_as : (symbol * static_const) list -> (symbol * static_const) list
    -> expr -> bool **)

let rec bound_as known sets = function
  | Let (_, _, _, body) -> bound_as known sets body
  | Let_static (bindings, body) ->
    (forallb (binding_bound_as known sets) bindings
    && forallb (code_of_binding_bound_as known sets) bindings)
    && bound_as known sets body
  | Let_cont_non_recursive (_, h, body) ->
    handler_bound_as known sets h && bound_as known sets body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_bound_as known sets h)
      handlers
    && bound_as known sets body
  | _ -> true

(** val handler_bound_as : (symbol * static_const) list -> (symbol *
    static_const) list -> continuation_handler -> bool **)

and handler_bound_as known sets = function
  | Handler (_, _, handler) -> bound_as known sets handler

(** val code_of_binding_bound_as : (symbol * static_const) list -> (symbol *
    static_const) list -> static_binding -> bool **)

and code_of_binding_bound_as known sets = function
  | Bind_code (_, c) -> code_bound_as known sets c
  | _ -> true

(** val code_bound_as : (symbol * static_const) list -> (symbol * static_const)
    list -> code -> bool **)

and code_bound_as known sets = function
  | Code (_, _, _, _, _, body) -> bound_as known sets body

(** val fold_known_unit : expr -> expr option **)

let fold_known_unit e =
  let known = known_bindings e in
  let sets = known_sets e in
  if bound_as known sets e
  then Some (fold_known (known_statics known) (known_statics sets) e)
  else None
