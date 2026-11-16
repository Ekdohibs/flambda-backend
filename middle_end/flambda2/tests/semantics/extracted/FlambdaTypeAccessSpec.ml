open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaTypeGrammar
open List
open PeanoNat

(** val row_like_for_blocks_all_tags : row_like_for_blocks -> coq_Z list
    Or_unknown.t **)

let row_like_for_blocks_all_tags = function
  | Row_like_for_blocks (known_tags, other_tags, _) -> (
    match other_tags with
    | Or_bottom.Ok _ -> Or_unknown.Unknown
    | Or_bottom.Bottom -> Or_unknown.Known (Keyed.domain Z.eqb known_tags))

(** val block_case_size_and_shape : row_like_block_case Or_unknown.t -> (nat *
    block_shape) option **)

let block_case_size_and_shape = function
  | Or_unknown.Known case0 -> (
    match case0.case_index.index_domain with
    | Row_like_index_domain.Known size ->
      Some (size, case0.case_index.index_shape)
    | Row_like_index_domain.At_least _ -> None)
  | Or_unknown.Unknown -> None

(** val row_like_for_blocks_all_tags_and_sizes : row_like_for_blocks -> (coq_Z,
    nat * block_shape) Keyed.finite_map Or_unknown.t **)

let row_like_for_blocks_all_tags_and_sizes = function
  | Row_like_for_blocks (known_tags, other_tags, _) -> (
    match other_tags with
    | Or_bottom.Ok _ -> Or_unknown.Unknown
    | Or_bottom.Bottom ->
      if
        Keyed.for_all_bindings_b Z.eqb
          (fun _ case ->
            match block_case_size_and_shape case with
            | Some _ -> true
            | None -> false)
          known_tags
      then
        Or_unknown.Known
          (Keyed.map_values
             (fun case ->
               match block_case_size_and_shape case with
               | Some size_and_shape -> size_and_shape
               | None -> O, Scannable Value_only)
             known_tags)
      else Or_unknown.Unknown)

(** val row_like_for_blocks_get_singleton : row_like_for_blocks -> ((((coq_Z *
    block_shape) * nat) * flambda_type list) * Alloc_mode.t) option **)

let row_like_for_blocks_get_singleton = function
  | Row_like_for_blocks (known_tags, other_tags, alloc_mode) -> (
    match other_tags with
    | Or_bottom.Ok _ -> None
    | Or_bottom.Bottom -> (
      match Keyed.get_singleton_binding Z.eqb known_tags with
      | Some p -> (
        let tag, t0 = p in
        match t0 with
        | Or_unknown.Known case -> (
          match case.case_index.index_domain with
          | Row_like_index_domain.Known size ->
            Some
              ( (((tag, case.case_index.index_shape), size), case.case_maps_to),
                alloc_mode )
          | Row_like_index_domain.At_least _ -> None)
        | Or_unknown.Unknown -> None)
      | None -> None))

(** val row_like_for_blocks_get_field : row_like_for_blocks -> nat ->
    flambda_type or_unknown_or_bottom **)

let row_like_for_blocks_get_field blocks index =
  match row_like_for_blocks_get_singleton blocks with
  | Some p -> (
    let p0, _ = p in
    let p1, maps_to = p0 in
    let _, size = p1 in
    if Nat.leb size index
    then Bottom
    else match nth_error maps_to index with Some ty -> Ok ty | None -> Unknown)
  | None -> Unknown

(** val closures_entry_find_function_type : closures_entry -> bool ->
    function_slot -> function_type or_unknown_or_bottom **)

let closures_entry_find_function_type entry exact slot =
  let (Closures_entry (function_types, _, _)) = entry in
  match lookup slot function_types with
  | Some t0 -> (
    match t0 with
    | Or_unknown.Known function_type0 -> Ok function_type0
    | Or_unknown.Unknown -> Unknown)
  | None -> if exact then Bottom else Unknown

type get_single_tag_result =
  | No_singleton
  | Exact_closure of function_slot * closures_entry
  | Incomplete_closure of function_slot * closures_entry

(** val row_like_for_closures_get_single_tag : row_like_for_closures ->
    get_single_tag_result option **)

let row_like_for_closures_get_single_tag = function
  | Row_like_for_closures known_closures -> (
    match Keyed.get_singleton_binding Pos.eqb known_closures with
    | Some p -> (
      let slot, case = p in
      match case.case_index.index_domain with
      | Row_like_index_domain.Known contents ->
        if among slot contents.contents_closures
        then Some (Exact_closure (slot, case.case_maps_to))
        else None
      | Row_like_index_domain.At_least contents ->
        if among slot contents.contents_closures
        then Some (Incomplete_closure (slot, case.case_maps_to))
        else Some No_singleton)
    | None -> Some No_singleton)

(** val row_like_for_closures_get_closure : row_like_for_closures ->
    function_slot -> flambda_type Or_unknown.t option **)

let row_like_for_closures_get_closure closures slot =
  match row_like_for_closures_get_single_tag closures with
  | Some g -> (
    match g with
    | No_singleton -> Some Or_unknown.Unknown
    | Exact_closure (_, entry) -> (
      let (Closures_entry (_, closure_types, _)) = entry in
      match lookup slot closure_types with
      | Some ty -> Some (Or_unknown.Known ty)
      | None -> Some Or_unknown.Unknown)
    | Incomplete_closure (_, entry) -> (
      let (Closures_entry (_, closure_types, _)) = entry in
      match lookup slot closure_types with
      | Some ty -> Some (Or_unknown.Known ty)
      | None -> Some Or_unknown.Unknown))
  | None -> None

(** val row_like_for_closures_get_env_var : row_like_for_closures -> value_slot
    -> flambda_type Or_unknown.t option **)

let row_like_for_closures_get_env_var closures slot =
  match row_like_for_closures_get_single_tag closures with
  | Some g -> (
    match g with
    | No_singleton -> Some Or_unknown.Unknown
    | Exact_closure (_, entry) -> (
      let (Closures_entry (_, _, value_slot_types)) = entry in
      match lookup slot value_slot_types with
      | Some ty -> Some (Or_unknown.Known ty)
      | None -> Some Or_unknown.Unknown)
    | Incomplete_closure (_, entry) -> (
      let (Closures_entry (_, _, value_slot_types)) = entry in
      match lookup slot value_slot_types with
      | Some ty -> Some (Or_unknown.Known ty)
      | None -> Some Or_unknown.Unknown))
  | None -> None
