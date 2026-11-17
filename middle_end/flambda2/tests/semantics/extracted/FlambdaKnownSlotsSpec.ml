open BinPos
open Datatypes
open FlambdaKnownSymbolsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaStaticsInScopeSpec
open FlambdaWellFormedSpec
open List
open ListDef

type known_slots = (value_slot * simple) list

(** val same_simple : simple -> simple -> bool **)

let same_simple a a' = if simple_eq_dec a a' then true else false

(** val closed_slots : known_slots -> bool **)

let closed_slots known = forallb (fun entry -> closed_simple (snd entry)) known

(** val named_slots_made : named -> (value_slot * simple) list **)

let named_slots_made = function
  | Set_of_closures (_, value_slots) -> value_slots
  | Closure_of_set (_, _, value_slots) -> value_slots
  | _ -> []

(** val static_const_slots_made : static_const -> (value_slot * simple) list **)

let static_const_slots_made = function
  | Static_const_set_of_closures (_, value_slots) -> value_slots
  | Static_const_closure_of_set (_, _, _, value_slots) -> value_slots
  | _ -> []

(** val slots_made : expr -> (value_slot * simple) list **)

let rec slots_made = function
  | Let (_, _, d, body) -> app (named_slots_made d) (slots_made body)
  | Let_static (bindings, body) ->
    app (flat_map binding_slots_made bindings) (slots_made body)
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_slots_made h) (slots_made body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_slots_made h)
         handlers)
      (slots_made body)
  | _ -> []

(** val handler_slots_made : continuation_handler -> (value_slot * simple) list
    **)

and handler_slots_made = function
  | Handler (_, _, handler) -> slots_made handler

(** val binding_slots_made : static_binding -> (value_slot * simple) list **)

and binding_slots_made = function
  | Bind_code (_, c) -> code_slots_made c
  | Bind_deleted_code _ -> []
  | Bind_symbol (_, const) -> static_const_slots_made const

(** val code_slots_made : code -> (value_slot * simple) list **)

and code_slots_made = function Code (_, _, _, _, _, body) -> slots_made body

(** val slots_known : (value_slot * simple) list -> known_slots **)

let slots_known made =
  filter
    (fun pat ->
      let slot, a = pat in
      closed_simple a
      && forallb
           (fun pat0 ->
             let slot', a' = pat0 in
             negb (Pos.eqb slot' slot) || same_simple a' a)
           made)
    made

(** val value_slots_as : known_slots -> (value_slot * simple) list -> bool **)

let value_slots_as known value_slots =
  forallb
    (fun pat ->
      let slot, a = pat in
      match lookup slot known with Some a' -> same_simple a a' | None -> true)
    value_slots

(** val slots_as : known_slots -> expr -> bool **)

let rec slots_as known = function
  | Let (_, _, d, body) ->
    value_slots_as known (named_slots_made d) && slots_as known body
  | Let_static (bindings, body) ->
    forallb (binding_slots_as known) bindings && slots_as known body
  | Let_cont_non_recursive (_, h, body) ->
    handler_slots_as known h && slots_as known body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_slots_as known h)
      handlers
    && slots_as known body
  | _ -> true

(** val handler_slots_as : known_slots -> continuation_handler -> bool **)

and handler_slots_as known = function
  | Handler (_, _, handler) -> slots_as known handler

(** val binding_slots_as : known_slots -> static_binding -> bool **)

and binding_slots_as known = function
  | Bind_code (_, c) -> code_slots_as known c
  | Bind_deleted_code _ -> true
  | Bind_symbol (_, const) ->
    value_slots_as known (static_const_slots_made const)

(** val code_slots_as : known_slots -> code -> bool **)

and code_slots_as known = function
  | Code (_, _, _, _, _, body) -> slots_as known body

(** val read_slot : known_slots -> symbol list -> named -> named **)

let read_slot known syms d =
  match d with
  | Project_value_slot (slot, _) -> (
    match lookup slot known with
    | Some a -> if simple_statics_in_scope syms a then Simple a else d
    | None -> d)
  | Project_value_slot_from (_, slot, _) -> (
    match lookup slot known with
    | Some a -> if simple_statics_in_scope syms a then Simple a else d
    | None -> d)
  | _ -> d

(** val read_slots : known_slots -> symbol list -> expr -> expr **)

let rec read_slots known syms e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, read_slot known syms d, read_slots known syms body)
  | Let_static (bindings, body) ->
    let all = app (flat_map symbols_of_binding bindings) syms in
    Let_static
      (map (read_slots_in_binding known all) bindings, read_slots known all body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        read_slots_in_handler known
          (app (handler_scope symbols_of_binding k body) syms)
          h,
        read_slots known syms body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            ( k,
              read_slots_in_handler known
                (app
                   (recursive_scope symbols_of_binding (map fst handlers) body)
                   syms)
                h ))
          handlers,
        read_slots known syms body )
  | _ -> e

(** val read_slots_in_handler : known_slots -> symbol list ->
    continuation_handler -> continuation_handler **)

and read_slots_in_handler known syms = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, read_slots known syms handler)

(** val read_slots_in_binding : known_slots -> symbol list -> static_binding ->
    static_binding **)

and read_slots_in_binding known syms binding =
  match binding with
  | Bind_code (id, c) ->
    if code_is_stub c
    then binding
    else Bind_code (id, read_slots_in_code known syms c)
  | _ -> binding

(** val read_slots_in_code : known_slots -> symbol list -> code -> code **)

and read_slots_in_code known syms = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, read_slots known syms body)

(** val read_known_slots : expr -> expr **)

let read_known_slots e =
  let known = slots_known (slots_made e) in
  if closed_slots known && slots_as known e then read_slots known [] e else e

(** val read_known_slots_anywhere : expr -> expr **)

let read_known_slots_anywhere e =
  let known = slots_known (slots_made e) in
  if closed_slots known && slots_as known e
  then read_slots known (static_names symbols_of_binding e) e
  else e
