open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open List
open ListDef
open OxCamlOption

type static_place =
  | Place_slot of value_slot
  | Place_field of coq_Z

(** val same_place : static_place -> static_place -> bool **)

let same_place p p' =
  match p with
  | Place_slot slot -> (
    match p' with
    | Place_slot slot' -> Pos.eqb slot slot'
    | Place_field _ -> false)
  | Place_field field -> (
    match p' with
    | Place_slot _ -> false
    | Place_field field' -> Z.eqb field field')

(** val place_held : static_const -> static_place -> simple option **)

let place_held const = function
  | Place_slot slot -> (
    match const with
    | Static_const_set_of_closures (_, value_slots) -> lookup slot value_slots
    | Static_const_closure_of_set (_, _, _, value_slots) ->
      lookup slot value_slots
    | _ -> None)
  | Place_field field -> (
    match const with
    | Static_const_block (_, fields) ->
      if Z.leb Z0 field then nth_error fields (Z.to_nat field) else None
    | _ -> None)

(** val const_places : static_const -> static_place list **)

let const_places = function
  | Static_const_set_of_closures (_, value_slots) ->
    map (fun slot -> Place_slot (fst slot)) value_slots
  | Static_const_closure_of_set (_, _, _, value_slots) ->
    map (fun slot -> Place_slot (fst slot)) value_slots
  | Static_const_block (_, fields) ->
    map (fun n -> Place_field (Z.of_nat n)) (seq O (length fields))
  | _ -> []

type place_class = symbol * variable

(** val same_class : place_class -> place_class -> bool **)

let same_class c c' = Pos.eqb (fst c) (fst c') && Pos.eqb (snd c) (snd c')

type held_places = ((symbol * static_place) * place_class) list

(** val first_symbol : static_binding list -> symbol option **)

let rec first_symbol = function
  | [] -> None
  | s0 :: bindings0 -> (
    match s0 with Bind_symbol (s, _) -> Some s | _ -> first_symbol bindings0)

(** val place_entry : symbol -> symbol -> static_const -> static_place ->
    held_places **)

let place_entry first s const p =
  match place_held const p with
  | Some s0 -> ( match s0 with Var x -> ((s, p), (first, x)) :: [] | _ -> [])
  | None -> []

(** val binding_places : symbol -> static_binding -> held_places **)

let binding_places first = function
  | Bind_symbol (s, const) ->
    flat_map (place_entry first s const) (const_places const)
  | _ -> []

(** val group_places : static_binding list -> held_places **)

let group_places bindings =
  match first_symbol bindings with
  | Some first -> flat_map (binding_places first) bindings
  | None -> []

(** val places_held : expr -> held_places **)

let rec places_held = function
  | Let (_, _, _, body) -> places_held body
  | Let_static (bindings, body) ->
    app (group_places bindings) (places_held body)
  | Let_cont_non_recursive (_, h, body) ->
    app (places_held_of_handler h) (places_held body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           places_held_of_handler h)
         handlers)
      (places_held body)
  | _ -> []

(** val places_held_of_handler : continuation_handler -> held_places **)

and places_held_of_handler = function
  | Handler (_, _, handler) -> places_held handler

(** val group_const : symbol -> static_binding list -> static_const option **)

let rec group_const s = function
  | [] -> None
  | s0 :: bindings0 -> (
    match s0 with
    | Bind_symbol (s', const) ->
      if Pos.eqb s s' then Some const else group_const s bindings0
    | _ -> group_const s bindings0)

(** val entry_meets_group : static_binding list -> ((symbol * static_place) *
    place_class) -> bool **)

let entry_meets_group bindings = function
  | p, p0 -> (
    let s, _ = p in
    let first, _ = p0 in
    is_some (group_const s bindings)
    ||
    match first_symbol bindings with
    | Some first' -> Pos.eqb first first'
    | None -> false)

(** val entry_of_group : static_binding list -> ((symbol * static_place) *
    place_class) -> bool **)

let entry_of_group bindings = function
  | p0, p1 -> (
    let s, p = p0 in
    let first, x = p1 in
    match first_symbol bindings with
    | Some first' -> (
      match group_const s bindings with
      | Some const -> (
        Pos.eqb first first'
        &&
        match place_held const p with
        | Some s0 -> ( match s0 with Var x' -> Pos.eqb x x' | _ -> false)
        | None -> false)
      | None -> false)
    | None -> false)

(** val group_holds : held_places -> static_binding list -> bool **)

let group_holds held bindings =
  forallb
    (fun entry ->
      negb (entry_meets_group bindings entry) || entry_of_group bindings entry)
    held

(** val groups_hold : held_places -> expr -> bool **)

let rec groups_hold held = function
  | Let (_, _, _, body) -> groups_hold held body
  | Let_static (bindings, body) ->
    (group_holds held bindings
    && forallb (code_of_binding_groups_hold held) bindings)
    && groups_hold held body
  | Let_cont_non_recursive (_, h, body) ->
    handler_groups_hold held h && groups_hold held body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_groups_hold held h)
      handlers
    && groups_hold held body
  | _ -> true

(** val handler_groups_hold : held_places -> continuation_handler -> bool **)

and handler_groups_hold held = function
  | Handler (_, _, handler) -> groups_hold held handler

(** val code_of_binding_groups_hold : held_places -> static_binding -> bool **)

and code_of_binding_groups_hold held = function
  | Bind_code (_, c) -> code_groups_hold held c
  | _ -> true

(** val code_groups_hold : held_places -> code -> bool **)

and code_groups_hold held = function
  | Code (_, _, _, _, _, body) -> groups_hold held body

(** val read_of : named -> (symbol * static_place) option **)

let read_of = function
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Block_load (_, mut, field) -> (
        match mut with
        | Mutable -> None
        | _ -> (
          match args with
          | [] -> None
          | s0 :: l -> (
            match s0 with
            | Symbol s -> (
              match l with [] -> Some (s, Place_field field) | _ :: _ -> None)
            | _ -> None)))
      | _ -> None)
    | _ -> None)
  | Project_value_slot (slot, closure) -> (
    match closure with Symbol s -> Some (s, Place_slot slot) | _ -> None)
  | Project_value_slot_from (_, slot, closure) -> (
    match closure with Symbol s -> Some (s, Place_slot slot) | _ -> None)
  | _ -> None

(** val class_of : held_places -> symbol -> static_place -> place_class option **)

let rec class_of held s p =
  match held with
  | [] -> None
  | p0 :: held0 ->
    let p1, c = p0 in
    let s', p' = p1 in
    if Pos.eqb s s' && same_place p p' then Some c else class_of held0 s p

(** val read_class : held_places -> named -> place_class option **)

let read_class held d =
  match read_of d with
  | Some p0 ->
    let s, p = p0 in
    class_of held s p
  | None -> None

type reads_made = (place_class * variable) list

(** val forget_read : variable -> reads_made -> reads_made **)

let forget_read x made = filter (fun entry -> negb (Pos.eqb x (snd entry))) made

(** val forget_reads : variable list -> reads_made -> reads_made **)

let rec forget_reads xs made =
  match xs with [] -> made | x :: xs0 -> forget_reads xs0 (forget_read x made)

(** val read_before : reads_made -> place_class -> variable option **)

let rec read_before made c =
  match made with
  | [] -> None
  | p :: made0 ->
    let c', v = p in
    if same_class c c' then Some v else read_before made0 c

(** val share_reads : held_places -> reads_made -> expr -> expr **)

let rec share_reads held made e =
  match e with
  | Let (x, kind, d, body) -> (
    match read_class held d with
    | Some c -> (
      match read_before made c with
      | Some v ->
        Let (x, kind, Simple (Var v), share_reads held (forget_read x made) body)
      | None ->
        Let (x, kind, d, share_reads held ((c, x) :: forget_read x made) body))
    | None -> Let (x, kind, d, share_reads held (forget_read x made) body))
  | Let_static (bindings, body) ->
    Let_static
      (map (share_reads_in_binding held) bindings, share_reads held made body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, share_reads_in_handler held made h, share_reads held made body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, share_reads_in_handler held made h)
          handlers,
        share_reads held made body )
  | _ -> e

(** val share_reads_in_handler : held_places -> reads_made ->
    continuation_handler -> continuation_handler **)

and share_reads_in_handler held made = function
  | Handler (params, metadata, handler) ->
    Handler
      (params, metadata, share_reads held (forget_reads params made) handler)

(** val share_reads_in_binding : held_places -> static_binding -> static_binding
    **)

and share_reads_in_binding held binding =
  match binding with
  | Bind_code (id, c) ->
    if code_is_stub c
    then binding
    else Bind_code (id, share_reads_in_code held c)
  | _ -> binding

(** val share_reads_in_code : held_places -> code -> code **)

and share_reads_in_code held = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, share_reads held [] body)

(** val share_static_reads : expr -> expr **)

let share_static_reads e =
  let held = places_held e in
  if groups_hold held e then share_reads held [] e else e
