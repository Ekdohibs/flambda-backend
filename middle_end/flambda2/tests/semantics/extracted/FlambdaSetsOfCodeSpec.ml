open BinPos
open Datatypes
open FlambdaKnownSlotsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaValueSpec
open List

type set_made = (function_slot * code_id) list * (value_slot * simple) list

(** val named_sets_made : named -> set_made list **)

let named_sets_made = function
  | Set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], value_slots) :: []
  | Closure_of_set (_, function_decls, value_slots) ->
    (function_decls, value_slots) :: []
  | _ -> []

(** val static_const_sets_made : static_const -> set_made list **)

let static_const_sets_made = function
  | Static_const_set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], value_slots) :: []
  | Static_const_closure_of_set (_, _, function_decls, value_slots) ->
    (function_decls, value_slots) :: []
  | _ -> []

(** val sets_made : expr -> set_made list **)

let rec sets_made = function
  | Let (_, _, d, body) -> app (named_sets_made d) (sets_made body)
  | Let_static (bindings, body) ->
    app (flat_map binding_sets_made bindings) (sets_made body)
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_sets_made h) (sets_made body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_sets_made h)
         handlers)
      (sets_made body)
  | _ -> []

(** val handler_sets_made : continuation_handler -> set_made list **)

and handler_sets_made = function Handler (_, _, handler) -> sets_made handler

(** val binding_sets_made : static_binding -> set_made list **)

and binding_sets_made = function
  | Bind_code (_, c) -> code_sets_made c
  | Bind_deleted_code _ -> []
  | Bind_symbol (_, const) -> static_const_sets_made const

(** val code_sets_made : code -> set_made list **)

and code_sets_made = function Code (_, _, _, _, _, body) -> sets_made body

type set_fact =
  | Code_at of function_slot * code_id
  | Slot_holds of value_slot * simple

(** val has_code_at : code_id -> function_slot -> (function_slot * code_id) list
    -> bool **)

let has_code_at c f function_decls =
  match lookup f function_decls with
  | Some code0 -> Pos.eqb code0 c
  | None -> false

(** val set_made_has : set_fact -> set_made -> bool **)

let set_made_has fact made =
  match fact with
  | Code_at (slot, code0) -> has_code_at code0 slot (fst made)
  | Slot_holds (slot, held) -> (
    match lookup slot (snd made) with
    | Some a -> same_simple a held
    | None -> false)

(** val sets_made_have : code_id -> function_slot -> set_fact -> set_made list
    -> bool **)

let sets_made_have c f fact made =
  forallb
    (fun set -> negb (has_code_at c f (fst set)) || set_made_has fact set)
    made
