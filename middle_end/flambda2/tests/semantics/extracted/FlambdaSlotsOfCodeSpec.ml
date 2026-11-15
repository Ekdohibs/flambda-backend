open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaSetsOfCodeSpec
open FlambdaValueSpec
open List
open ListDef

type closure_seen = function_slot * (function_slot * code_id) list

(** val seen_code : closure_seen -> code_id option **)

let seen_code seen = lookup (fst seen) (snd seen)

type closure_sight = closure_seen finite_map

(** val sight_made : closure_sight -> named -> closure_seen option **)

let sight_made known = function
  | Set_of_closures (code0, _) ->
    Some (unnamed_function_slot, (unnamed_function_slot, code0) :: [])
  | Closure_of_set (slot, function_decls, _) -> Some (slot, function_decls)
  | Project_function_slot (_, move_to, closure0) -> (
    match closure0 with
    | Var closure -> (
      match lookup closure known with
      | Some c ->
        let _, function_decls = c in
        Some (move_to, function_decls)
      | None -> None)
    | _ -> None)
  | _ -> None

(** val sight_under_let : closure_sight -> variable -> named -> closure_sight **)

let sight_under_let known x d =
  match sight_made known d with
  | Some seen -> (x, seen) :: unbind x known
  | None -> unbind x known

(** val sight_without : variable list -> closure_sight -> closure_sight **)

let rec sight_without xs known =
  match xs with [] -> known | x :: xs0 -> sight_without xs0 (unbind x known)

(** val simple_seen_code : closure_sight -> simple -> code_id option **)

let simple_seen_code known = function
  | Var x -> (
    match lookup x known with Some seen -> seen_code seen | None -> None)
  | _ -> None

type set_seen =
  (function_slot * code_id) list * (value_slot * code_id option) list

(** val slots_seen : closure_sight -> (value_slot * simple) list -> (value_slot
    * code_id option) list **)

let slots_seen known value_slots =
  map
    (fun pat ->
      let slot, a = pat in
      slot, simple_seen_code known a)
    value_slots

(** val named_sets_seen : closure_sight -> named -> set_seen list **)

let named_sets_seen known = function
  | Set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], slots_seen known value_slots) :: []
  | Closure_of_set (_, function_decls, value_slots) ->
    (function_decls, slots_seen known value_slots) :: []
  | _ -> []

(** val static_const_sets_seen : closure_sight -> static_const -> set_seen list
    **)

let static_const_sets_seen known = function
  | Static_const_set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], slots_seen known value_slots) :: []
  | Static_const_closure_of_set (_, _, function_decls, value_slots) ->
    (function_decls, slots_seen known value_slots) :: []
  | _ -> []

(** val sets_seen : closure_sight -> expr -> set_seen list **)

let rec sets_seen known = function
  | Let (x, _, d, body) ->
    app (named_sets_seen known d) (sets_seen (sight_under_let known x d) body)
  | Let_static (bindings, body) ->
    app (flat_map (binding_sets_seen known) bindings) (sets_seen known body)
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_sets_seen known h) (sets_seen known body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_sets_seen known h)
         handlers)
      (sets_seen known body)
  | _ -> []

(** val handler_sets_seen : closure_sight -> continuation_handler -> set_seen
    list **)

and handler_sets_seen known = function
  | Handler (params, _, handler) ->
    sets_seen (sight_without params known) handler

(** val binding_sets_seen : closure_sight -> static_binding -> set_seen list **)

and binding_sets_seen known = function
  | Bind_code (_, c) -> code_sets_seen c
  | Bind_deleted_code _ -> []
  | Bind_symbol (_, const) -> static_const_sets_seen known const

(** val code_sets_seen : code -> set_seen list **)

and code_sets_seen = function Code (_, _, _, _, _, body) -> sets_seen [] body

(** val set_seen_gives : value_slot -> code_id -> set_seen -> bool **)

let set_seen_gives slot d set =
  match lookup slot (snd set) with
  | Some o -> ( match o with Some code0 -> Pos.eqb code0 d | None -> false)
  | None -> false

(** val sets_seen_give : code_id -> function_slot -> value_slot -> code_id ->
    set_seen list -> bool **)

let sets_seen_give c f slot d seen =
  forallb
    (fun set -> negb (has_code_at c f (fst set)) || set_seen_gives slot d set)
    seen
