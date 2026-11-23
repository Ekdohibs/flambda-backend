open BinPos
open Datatypes
open FlambdaClosureOfCodeSpec
open FlambdaKnownSlotsSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSetsOfCodeSpec
open FlambdaSlotsOfCodeSpec
open FlambdaValueSpec
open List
open ListDef

type made_seen = (value_slot * simple option) list

type made_sight = made_seen finite_map

(** val slots_made : named -> made_seen option **)

let slots_made = function
  | Set_of_closures (_, value_slots) ->
    Some (map_values (fun x -> Some x) value_slots)
  | Closure_of_set (_, _, value_slots) ->
    Some (map_values (fun x -> Some x) value_slots)
  | _ -> None

(** val forget_simple : variable -> simple option -> simple option **)

let forget_simple x = function
  | Some s -> if var_in_simple x s then None else Some s
  | None -> None

(** val forget_in_made : variable -> made_seen -> made_seen **)

let forget_in_made x made = map_values (forget_simple x) made

(** val made_without : variable -> made_sight -> made_sight **)

let made_without x known = map_values (forget_in_made x) (unbind x known)

(** val made_without_all : variable list -> made_sight -> made_sight **)

let rec made_without_all xs known =
  match xs with
  | [] -> known
  | x :: xs0 -> made_without_all xs0 (made_without x known)

(** val made_under_let : made_sight -> variable -> named -> made_sight **)

let made_under_let known x d =
  match slots_made d with
  | Some made -> (x, forget_in_made x made) :: made_without x known
  | None -> made_without x known

type slot_through_seen = simple * made_seen option

(** val simple_made_seen : made_sight -> simple -> made_seen option **)

let simple_made_seen known = function Var y -> lookup y known | _ -> None

type set_through =
  (function_slot * code_id) list * (value_slot * slot_through_seen) list

(** val slots_through : made_sight -> (value_slot * simple) list -> (value_slot
    * slot_through_seen) list **)

let slots_through known value_slots =
  map
    (fun pat ->
      let slot, s = pat in
      slot, (s, simple_made_seen known s))
    value_slots

(** val named_sets_through : made_sight -> named -> set_through list **)

let named_sets_through known = function
  | Set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], slots_through known value_slots)
    :: []
  | Closure_of_set (_, function_decls, value_slots) ->
    (function_decls, slots_through known value_slots) :: []
  | _ -> []

(** val static_const_sets_through : made_sight -> static_const -> set_through
    list **)

let static_const_sets_through known = function
  | Static_const_set_of_closures (code0, value_slots) ->
    ((unnamed_function_slot, code0) :: [], slots_through known value_slots)
    :: []
  | Static_const_closure_of_set (_, _, function_decls, value_slots) ->
    (function_decls, slots_through known value_slots) :: []
  | _ -> []

(** val sets_through : made_sight -> expr -> set_through list **)

let rec sets_through known = function
  | Let (x, _, d, body) ->
    app
      (named_sets_through known d)
      (sets_through (made_under_let known x d) body)
  | Let_static (bindings, body) ->
    app
      (flat_map (binding_sets_through known) bindings)
      (sets_through known body)
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_sets_through known h) (sets_through known body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_sets_through known h)
         handlers)
      (sets_through known body)
  | _ -> []

(** val handler_sets_through : made_sight -> continuation_handler -> set_through
    list **)

and handler_sets_through known = function
  | Handler (params, _, handler) ->
    sets_through (made_without_all params known) handler

(** val binding_sets_through : made_sight -> static_binding -> set_through list
    **)

and binding_sets_through known = function
  | Bind_code (_, c) -> code_sets_through c
  | Bind_deleted_code _ -> []
  | Bind_symbol (_, const) -> static_const_sets_through known const

(** val code_sets_through : code -> set_through list **)

and code_sets_through = function
  | Code (_, _, _, _, _, body) -> sets_through [] body

(** val set_through_gives : value_slot -> value_slot -> value_slot ->
    set_through -> bool **)

let set_through_gives a b d set =
  match lookup a (snd set) with
  | Some s -> (
    let _, o = s in
    match o with
    | Some made -> (
      match lookup b (snd set) with
      | Some s1 -> (
        let held, _ = s1 in
        match lookup d made with
        | Some o0 -> (
          match o0 with Some inner -> same_simple inner held | None -> false)
        | None -> false)
      | None -> false)
    | None -> false)
  | None -> false

(** val sets_through_give : code_id -> function_slot -> value_slot -> value_slot
    -> value_slot -> set_through list -> bool **)

let sets_through_give c f a b d seen =
  forallb
    (fun set -> negb (has_code_at c f (fst set)) || set_through_gives a b d set)
    seen

(** val slot_through : set_through list -> code_id -> function_slot ->
    value_slot -> value_slot -> value_slot option **)

let slot_through seen c f a d =
  match find (fun set -> has_code_at c f (fst set)) seen with
  | Some set ->
    option_map fst
      (find (fun entry -> sets_through_give c f a (fst entry) d seen) (snd set))
  | None -> None

type projections = ((variable * function_slot) * value_slot) finite_map

(** val through_without : variable -> projections -> projections **)

let through_without x through =
  filter
    (fun entry -> negb (Pos.eqb x (fst (fst (snd entry)))))
    (unbind x through)

(** val through_without_all : variable list -> projections -> projections **)

let rec through_without_all xs through =
  match xs with
  | [] -> through
  | x :: xs0 -> through_without_all xs0 (through_without x through)

(** val through_under_let : projections -> variable -> named -> projections **)

let through_under_let through x = function
  | Project_value_slot_from (f, a, closure) -> (
    match closure with
    | Var from ->
      if Pos.eqb from x
      then through_without x through
      else (x, ((from, f), a)) :: through_without x through
    | _ -> through_without x through)
  | _ -> through_without x through

(** val read_through : set_through list -> closure_codes -> projections -> named
    -> named **)

let read_through seen known through d =
  match d with
  | Project_value_slot_from (_, slot, closure) -> (
    match closure with
    | Var y -> (
      match lookup y through with
      | Some p -> (
        let p0, a = p in
        let x, f = p0 in
        match lookup x known with
        | Some c -> (
          match slot_through seen c f a slot with
          | Some b -> Project_value_slot_from (f, b, Var x)
          | None -> d)
        | None -> d)
      | None -> d)
    | _ -> d)
  | _ -> d

(** val know_through_in_body : set_made list -> set_seen list -> set_through
    list -> closure_codes -> projections -> expr -> expr **)

let rec know_through_in_body sets codes seen known through e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        read_through seen known through d,
        know_through_in_body sets codes seen
          (codes_under_read sets codes known x d)
          (through_under_let through x d)
          body )
  | Let_static (bindings, body) ->
    Let_static
      (bindings, know_through_in_body sets codes seen known through body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        know_through_in_handler sets codes seen known through h,
        know_through_in_body sets codes seen known through body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, know_through_in_handler sets codes seen known through h)
          handlers,
        know_through_in_body sets codes seen known through body )
  | _ -> e

(** val know_through_in_handler : set_made list -> set_seen list -> set_through
    list -> closure_codes -> projections -> continuation_handler ->
    continuation_handler **)

and know_through_in_handler sets codes seen known through = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        know_through_in_body sets codes seen
          (forget_code_vars params known)
          (through_without_all params through)
          handler )

(** val know_through_in_code : set_made list -> set_seen list -> set_through
    list -> code_id -> code -> code **)

let know_through_in_code sets codes seen id c =
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
          know_through_in_body sets codes seen ((my_closure, id) :: []) [] body
        )

(** val know_through_in_binding : set_made list -> set_seen list -> set_through
    list -> static_binding -> static_binding **)

let know_through_in_binding sets codes seen binding =
  match binding with
  | Bind_code (id, c) ->
    Bind_code (id, know_through_in_code sets codes seen id c)
  | _ -> binding

(** val know_code_through : set_made list -> set_seen list -> set_through list
    -> expr -> expr **)

let rec know_code_through sets codes seen e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, know_code_through sets codes seen body)
  | Let_static (bindings, body) ->
    Let_static
      ( map (know_through_in_binding sets codes seen) bindings,
        know_code_through sets codes seen body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        know_code_through_in_handler sets codes seen h,
        know_code_through sets codes seen body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, know_code_through_in_handler sets codes seen h)
          handlers,
        know_code_through sets codes seen body )
  | _ -> e

(** val know_code_through_in_handler : set_made list -> set_seen list ->
    set_through list -> continuation_handler -> continuation_handler **)

and know_code_through_in_handler sets codes seen = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, know_code_through sets codes seen handler)

(** val read_slots_through : expr -> expr **)

let read_slots_through e =
  if groups_whole e
  then know_code_through (sets_made e) (sets_seen [] e) (sets_through [] e) e
  else e

(** val know_own_closures_through : expr -> expr **)

let know_own_closures_through e = read_slots_through (know_own_closures e)
