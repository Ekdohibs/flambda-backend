open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaValueSpec
open List
open ListDef
open OxCamlOption

(** val fresh_location : heap -> location **)

let fresh_location h = Pos.succ (fold_right Pos.max Coq_xH (domain h))

(** val load_field : block_access_kind -> coq_Z -> coq_Z -> value list -> value
    option **)

let load_field kind field block_tag fields =
  let (Block_access_values (tag, size, field_kind)) = kind in
  if
    (known_is tag block_tag && known_is size (Z.of_nat (length fields)))
    && Z.leb Z0 field
  then
    match nth_error fields (Z.to_nat field) with
    | Some loaded ->
      if
        match field_kind with
        | Any_value -> kind_eqb (kind_of_value loaded) Value
        | Immediate -> is_tagged_immediate loaded
      then Some loaded
      else None
    | None -> None
  else None

(** val store_field : block_access_kind -> coq_Z -> value -> coq_Z -> value list
    -> value list option **)

let store_field kind field new0 block_tag fields =
  let (Block_access_values (tag, size, field_kind)) = kind in
  if
    ((known_is tag block_tag && known_is size (Z.of_nat (length fields)))
    && Z.leb Z0 field)
    &&
    match field_kind with
    | Any_value -> kind_eqb (kind_of_value new0) Value
    | Immediate -> is_tagged_immediate new0
  then replace_nth (Z.to_nat field) new0 fields
  else None

(** val is_location : value -> bool **)

let is_location = function Value_location _ -> true | _ -> false

(** val eval_phys_equal_location : static_env -> value -> value -> bool option **)

let eval_phys_equal_location statics v1 v2 =
  match v1 with
  | Value_location l1 -> (
    match v2 with
    | Value_location l2 -> Some (Pos.eqb l1 l2)
    | _ ->
      if
        (is_some (int_of_value Standard_tagged_immediate v2)
        || is_allocated statics v2)
        || is_imported_pointer statics v2
      then Some false
      else None)
  | _ -> (
    match v2 with
    | Value_location _ ->
      if
        (is_some (int_of_value Standard_tagged_immediate v1)
        || is_allocated statics v1)
        || is_imported_pointer statics v1
      then Some false
      else None
    | _ -> None)

(** val unit_value : value **)

let unit_value = Value_const (Tagged_immediate Z0)

(** val exec_heap_primitive : static_env -> heap -> heap_primitive -> value list
    -> (value * heap) option **)

let exec_heap_primitive _ h p vs =
  match p with
  | Make_block_with_identity (mut, tag) ->
    if forallb (fun v -> kind_eqb (kind_of_value v) Value) vs
    then
      let l = fresh_location h in
      Some (Value_location l, (l, Cell_block (mut, tag, vs)) :: h)
    else None
  | Block_set (kind, _, field) -> (
    match vs with
    | [] -> None
    | v :: l -> (
      match l with
      | [] -> None
      | new0 :: l0 -> (
        match l0 with
        | [] -> (
          match v with
          | Value_location l1 -> (
            match lookup l1 h with
            | Some c -> (
              match c with
              | Cell_block (mut, tag, fields) -> (
                match mut with
                | Cell_mutable -> (
                  match store_field kind field new0 tag fields with
                  | Some fields' ->
                    Some
                      ( unit_value,
                        (l1, Cell_block (Cell_mutable, tag, fields')) :: h )
                  | None -> None)
                | Cell_immutable_unique -> None)
              | Cell_array (_, _) -> None)
            | None -> None)
          | _ -> None)
        | _ :: _ -> None)))
  | Make_mutable_array kind -> (
    match vs with
    | [] -> None
    | _ :: _ -> (
      match array_made kind vs with
      | Some elements ->
        let l = fresh_location h in
        Some (Value_location l, (l, Cell_array (kind, elements)) :: h)
      | None -> None))
  | Array_set (_, set_kind) -> (
    match vs with
    | [] -> None
    | v :: l -> (
      match l with
      | [] -> None
      | index :: l0 -> (
        match l0 with
        | [] -> None
        | new0 :: l1 -> (
          match l1 with
          | [] -> (
            match v with
            | Value_location l2 -> (
              match lookup l2 h with
              | Some c -> (
                match c with
                | Cell_block (_, _, _) -> None
                | Cell_array (kind, elements) -> (
                  match array_stored set_kind kind elements index new0 with
                  | Some elements' ->
                    Some (unit_value, (l2, Cell_array (kind, elements')) :: h)
                  | None -> None))
              | None -> None)
            | _ -> None)
          | _ :: _ -> None))))

(** val load_allowed : mutability -> cell_mutability -> bool **)

let load_allowed load_mut mut =
  match load_mut with
  | Mutable -> true
  | _ -> (
    match mut with Cell_mutable -> false | Cell_immutable_unique -> true)

(** val exec_unary_on_cell : unary_primitive -> location -> cell -> value option
    **)

let exec_unary_on_cell p l = function
  | Cell_block (mut, tag, fields) -> (
    match p with
    | Block_load (kind, load_mut, field) ->
      if load_allowed load_mut mut
      then load_field kind field tag fields
      else None
    | Is_int _ -> Some (naked_bool false)
    | Is_null -> Some (naked_bool false)
    | Get_tag -> Some (Value_const (Naked_immediate tag))
    | Opaque_identity kind ->
      if kind_eqb kind Value then Some (Value_location l) else None
    | _ -> None)
  | Cell_array (_, elements) -> (
    match p with
    | Is_int variant_only ->
      if variant_only then None else Some (naked_bool false)
    | Is_null -> Some (naked_bool false)
    | Array_length _ -> array_length_of elements
    | Opaque_identity kind ->
      if kind_eqb kind Value then Some (Value_location l) else None
    | _ -> None)

(** val exec_binary_on_cell : binary_primitive -> location -> cell -> value ->
    value option **)

let exec_binary_on_cell p _ c v2 =
  match c with
  | Cell_block (_, _, _) -> None
  | Cell_array (_, elements) -> (
    match p with
    | Array_load (_, load_kind, load_mut) ->
      if load_allowed load_mut Cell_mutable
      then array_loaded load_kind elements v2
      else None
    | _ -> None)

(** val argument_bound : static_env -> heap -> value -> bool **)

let argument_bound statics h = function
  | Value_symbol s -> is_bound s statics.static_values
  | Value_location l -> is_bound l h
  | _ -> true

(** val exec_bound_prim : static_env -> heap -> prim -> value list -> (value *
    heap) option **)

let exec_bound_prim statics h p vs =
  match p with
  | Unary p0 -> (
    match vs with
    | [] -> (
      match eval_prim statics p vs with Some v -> Some (v, h) | None -> None)
    | v :: l0 -> (
      match v with
      | Value_location l -> (
        match l0 with
        | [] -> (
          match lookup l h with
          | Some c -> (
            match exec_unary_on_cell p0 l c with
            | Some v0 -> Some (v0, h)
            | None -> None)
          | None -> None)
        | _ :: _ -> (
          match eval_prim statics p vs with
          | Some v0 -> Some (v0, h)
          | None -> None))
      | _ -> (
        match eval_prim statics p vs with
        | Some v0 -> Some (v0, h)
        | None -> None)))
  | Binary p0 -> (
    match p0 with
    | Phys_equal op -> (
      match vs with
      | [] -> (
        match eval_prim statics p vs with Some v -> Some (v, h) | None -> None)
      | v1 :: l0 -> (
        match v1 with
        | Value_location _ -> (
          match l0 with
          | [] -> (
            match eval_prim statics p vs with
            | Some v -> Some (v, h)
            | None -> None)
          | v2 :: l1 -> (
            match l1 with
            | [] -> (
              if is_location v1 || is_location v2
              then
                match eval_phys_equal_location statics v1 v2 with
                | Some same ->
                  Some
                    ( naked_bool
                        (match op with
                        | Phys_eq -> same
                        | Phys_neq -> negb same),
                      h )
                | None -> None
              else
                match eval_prim statics p vs with
                | Some v -> Some (v, h)
                | None -> None)
            | _ :: _ -> (
              match eval_prim statics p vs with
              | Some v -> Some (v, h)
              | None -> None)))
        | _ -> (
          match l0 with
          | [] -> (
            match eval_prim statics p vs with
            | Some v -> Some (v, h)
            | None -> None)
          | v2 :: l -> (
            match l with
            | [] -> (
              if is_location v1 || is_location v2
              then
                match eval_phys_equal_location statics v1 v2 with
                | Some same ->
                  Some
                    ( naked_bool
                        (match op with
                        | Phys_eq -> same
                        | Phys_neq -> negb same),
                      h )
                | None -> None
              else
                match eval_prim statics p vs with
                | Some v -> Some (v, h)
                | None -> None)
            | _ :: _ -> (
              match eval_prim statics p vs with
              | Some v -> Some (v, h)
              | None -> None)))))
    | _ -> (
      match vs with
      | [] -> (
        match eval_prim statics p vs with Some v -> Some (v, h) | None -> None)
      | v :: l0 -> (
        match v with
        | Value_location l -> (
          match l0 with
          | [] -> (
            match eval_prim statics p vs with
            | Some v0 -> Some (v0, h)
            | None -> None)
          | v2 :: l1 -> (
            match l1 with
            | [] -> (
              match lookup l h with
              | Some c -> (
                match exec_binary_on_cell p0 l c v2 with
                | Some v0 -> Some (v0, h)
                | None -> None)
              | None -> None)
            | _ :: _ -> (
              match eval_prim statics p vs with
              | Some v0 -> Some (v0, h)
              | None -> None)))
        | _ -> (
          match eval_prim statics p vs with
          | Some v0 -> Some (v0, h)
          | None -> None))))
  | Heap p0 -> exec_heap_primitive statics h p0 vs
  | _ -> (
    match eval_prim statics p vs with Some v -> Some (v, h) | None -> None)

(** val exec_prim : static_env -> heap -> prim -> value list -> (value * heap)
    option **)

let exec_prim statics h p vs =
  if forallb (argument_bound statics h) vs
  then exec_bound_prim statics h p vs
  else None

(** val named_changes_heap : named -> bool **)

let named_changes_heap = function
  | Prim (p, _) -> ( match p with Heap _ -> true | _ -> false)
  | _ -> false

(** val locations_of_value : value -> location list **)

let rec locations_of_value = function
  | Value_closure (_, slots) ->
    flat_map (fun slot -> locations_of_value (snd slot)) slots
  | Value_closure_of_set (_, _, slots) ->
    flat_map (fun slot -> locations_of_value (snd slot)) slots
  | Value_block (_, fields) -> flat_map locations_of_value fields
  | Value_immutable_array fields -> flat_map locations_of_value fields
  | Value_location l -> l :: []
  | Value_partial_application (callee, args, _) ->
    app (locations_of_value callee) (flat_map locations_of_value args)
  | _ -> []

(** val values_of_static : statically_allocated_value -> value list **)

let values_of_static = function
  | Static_closure (_, slots) -> map snd slots
  | Static_closure_of_set (_, _, _, slots) -> map snd slots
  | Static_block (_, fields) -> fields
  | Static_boxed_number v -> v :: []
  | Static_immutable_array elements -> elements
  | _ -> []
