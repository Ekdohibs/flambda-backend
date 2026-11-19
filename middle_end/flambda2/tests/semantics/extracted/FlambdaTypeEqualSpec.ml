open BinInt
open BinPos
open Bool
open Datatypes
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaTypeGrammar
open FlambdaTypeMeaningBoolSpec
open PeanoNat

(** val map_eqb : ('a1 -> 'a1 -> bool) -> ('a2 -> 'a2 -> bool) -> ('a1, 'a2)
    Keyed.finite_map -> ('a1, 'a2) Keyed.finite_map -> bool **)

let map_eqb key_eqb eqb0 bindings bindings' =
  Keyed.for_all_bindings_b key_eqb
    (fun key a ->
      match Keyed.lookup key_eqb key bindings' with
      | Some a' -> eqb0 a a'
      | None -> false)
    bindings
  && Keyed.for_all_bindings_b key_eqb
       (fun key _ -> Keyed.is_bound key_eqb key bindings)
       bindings'

(** val simple_eqb : simple -> simple -> bool **)

let simple_eqb s s' =
  match s with
  | Var x -> ( match s' with Var x' -> Pos.eqb x x' | _ -> false)
  | Symbol sym -> (
    match s' with Symbol sym' -> Pos.eqb sym sym' | _ -> false)
  | Const c -> ( match s' with Const c' -> const_eqb c c' | _ -> false)

(** val option_variable_eqb : variable option -> variable option -> bool **)

let option_variable_eqb x x' =
  match x with
  | Some x0 -> ( match x' with Some x'0 -> Pos.eqb x0 x'0 | None -> false)
  | None -> ( match x' with Some _ -> false | None -> true)

(** val alloc_mode_eqb : Alloc_mode.t -> Alloc_mode.t -> bool **)

let alloc_mode_eqb mode mode' =
  match mode with
  | Alloc_mode.Heap -> (
    match mode' with Alloc_mode.Heap -> true | _ -> false)
  | Alloc_mode.Local -> (
    match mode' with Alloc_mode.Local -> true | _ -> false)
  | Alloc_mode.Heap_or_local -> (
    match mode' with Alloc_mode.Heap_or_local -> true | _ -> false)

(** val relation_eqb : Relation.t -> Relation.t -> bool **)

let relation_eqb relation relation' =
  match relation with
  | Relation.Is_null -> (
    match relation' with Relation.Is_null -> true | _ -> false)
  | Relation.Is_int -> (
    match relation' with Relation.Is_int -> true | _ -> false)
  | Relation.Get_tag -> (
    match relation' with Relation.Get_tag -> true | _ -> false)

(** val naked_number_kind_eqb : naked_number_kind -> naked_number_kind -> bool **)

let naked_number_kind_eqb k k' = kind_eqb (Naked_number k) (Naked_number k')

(** val is_null_eqb : is_null -> is_null -> bool **)

let is_null_eqb is_null0 is_null' =
  match is_null0 with
  | Not_null -> (
    match is_null' with Not_null -> true | Maybe_null _ -> false)
  | Maybe_null x -> (
    match is_null' with
    | Not_null -> false
    | Maybe_null x' -> option_variable_eqb x x')

(** val block_shape_eqb : block_shape -> block_shape -> bool **)

let block_shape_eqb shape shape' =
  match shape with
  | Scannable shape0 -> (
    match shape0 with
    | Value_only -> (
      match shape' with
      | Scannable shape1 -> (
        match shape1 with Value_only -> true | Mixed_record (_, _) -> false)
      | Float_record -> false)
    | Mixed_record (value_prefix_size, flat_suffix) -> (
      match shape' with
      | Scannable shape1 -> (
        match shape1 with
        | Value_only -> false
        | Mixed_record (value_prefix_size', flat_suffix') ->
          Nat.eqb value_prefix_size value_prefix_size'
          && lists_eqb naked_number_kind_eqb flat_suffix flat_suffix')
      | Float_record -> false))
  | Float_record -> (
    match shape' with Scannable _ -> false | Float_record -> true)

(** val function_type_eqb : function_type -> function_type -> bool **)

let function_type_eqb function_type0 function_type' =
  Pos.eqb function_type0.function_type_code_id
    function_type'.function_type_code_id

(** val block_index_eqb : (nat, block_shape) row_like_index -> (nat,
    block_shape) row_like_index -> bool **)

let block_index_eqb index index' =
  (match index.index_domain with
    | Row_like_index_domain.Known size -> (
      match index'.index_domain with
      | Row_like_index_domain.Known size' -> Nat.eqb size size'
      | Row_like_index_domain.At_least _ -> false)
    | Row_like_index_domain.At_least size -> (
      match index'.index_domain with
      | Row_like_index_domain.Known _ -> false
      | Row_like_index_domain.At_least size' -> Nat.eqb size size'))
  && block_shape_eqb index.index_shape index'.index_shape

(** val closures_index_eqb : (set_of_closures_contents, unit) row_like_index ->
    (set_of_closures_contents, unit) row_like_index -> bool **)

let closures_index_eqb index index' =
  match index.index_domain with
  | Row_like_index_domain.Known contents -> (
    match index'.index_domain with
    | Row_like_index_domain.Known contents' ->
      Keyed.same_keys_b Pos.eqb contents.contents_closures
        contents'.contents_closures
      && Keyed.same_keys_b Pos.eqb contents.contents_value_slots
           contents'.contents_value_slots
    | Row_like_index_domain.At_least _ -> false)
  | Row_like_index_domain.At_least contents -> (
    match index'.index_domain with
    | Row_like_index_domain.Known _ -> false
    | Row_like_index_domain.At_least contents' ->
      Keyed.same_keys_b Pos.eqb contents.contents_closures
        contents'.contents_closures
      && Keyed.same_keys_b Pos.eqb contents.contents_value_slots
           contents'.contents_value_slots)

(** val or_unknown_eqb : ('a1 -> 'a1 -> bool) -> 'a1 Or_unknown.t -> 'a1
    Or_unknown.t -> bool **)

let or_unknown_eqb eqb0 a a' =
  match a with
  | Or_unknown.Known a0 -> (
    match a' with
    | Or_unknown.Known a'0 -> eqb0 a0 a'0
    | Or_unknown.Unknown -> false)
  | Or_unknown.Unknown -> (
    match a' with Or_unknown.Known _ -> false | Or_unknown.Unknown -> true)

(** val or_bottom_eqb : ('a1 -> 'a1 -> bool) -> 'a1 Or_bottom.t -> 'a1
    Or_bottom.t -> bool **)

let or_bottom_eqb eqb0 a a' =
  match a with
  | Or_bottom.Ok a0 -> (
    match a' with Or_bottom.Ok a'0 -> eqb0 a0 a'0 | Or_bottom.Bottom -> false)
  | Or_bottom.Bottom -> (
    match a' with Or_bottom.Ok _ -> false | Or_bottom.Bottom -> true)

(** val or_unknown_or_bottom_eqb : ('a1 -> 'a1 -> bool) -> 'a1
    or_unknown_or_bottom -> 'a1 or_unknown_or_bottom -> bool **)

let or_unknown_or_bottom_eqb eqb0 a a' =
  match a with
  | Unknown -> ( match a' with Unknown -> true | _ -> false)
  | Ok a0 -> ( match a' with Ok a'0 -> eqb0 a0 a'0 | _ -> false)
  | Bottom -> ( match a' with Bottom -> true | _ -> false)

(** val descr_eqb : ('a1 -> 'a1 -> bool) -> 'a1 type_descr -> 'a1 type_descr ->
    bool **)

let descr_eqb head_eqb ty ty' =
  or_unknown_or_bottom_eqb
    (fun descr descr' ->
      match descr with
      | No_alias head -> (
        match descr' with
        | No_alias head' -> head_eqb head head'
        | Equals _ -> false)
      | Equals s -> (
        match descr' with No_alias _ -> false | Equals s' -> simple_eqb s s'))
    ty ty'

(** val head_of_kind_naked_immediate_eqb : head_of_kind_naked_immediate ->
    head_of_kind_naked_immediate -> bool **)

let head_of_kind_naked_immediate_eqb head head' =
  or_unknown_eqb (Keyed.same_keys_b Z.eqb) head.naked_immediates
    head'.naked_immediates
  && Keyed.same_keys_b
       (fun relation relation' ->
         relation_eqb (fst relation) (fst relation')
         && name_eqb (snd relation) (snd relation'))
       head.inverse_relations head'.inverse_relations

(** val block_case_eqb : (flambda_type -> flambda_type -> bool) ->
    (env_extension -> env_extension -> bool) -> row_like_block_case ->
    row_like_block_case -> bool **)

let block_case_eqb type_eqb0 env_extension_eqb0 case case' =
  (block_index_eqb case.case_index case'.case_index
  && lists_eqb type_eqb0 case.case_maps_to case'.case_maps_to)
  && env_extension_eqb0 case.case_env_extension case'.case_env_extension

(** val closures_case_eqb : (closures_entry -> closures_entry -> bool) ->
    (env_extension -> env_extension -> bool) -> row_like_closures_case ->
    row_like_closures_case -> bool **)

let closures_case_eqb closures_entry_eqb0 env_extension_eqb0 case case' =
  (closures_index_eqb case.case_index case'.case_index
  && closures_entry_eqb0 case.case_maps_to case'.case_maps_to)
  && env_extension_eqb0 case.case_env_extension case'.case_env_extension

(** val variant_extensions_eqb : (env_extension -> env_extension -> bool) ->
    variant_extensions -> variant_extensions -> bool **)

let variant_extensions_eqb env_extension_eqb0 extensions extensions' =
  match extensions with
  | No_extensions -> (
    match extensions' with No_extensions -> true | Ext (_, _) -> false)
  | Ext (when_immediate, when_block) -> (
    match extensions' with
    | No_extensions -> false
    | Ext (when_immediate', when_block') ->
      env_extension_eqb0 when_immediate when_immediate'
      && env_extension_eqb0 when_block when_block')

(** val type_eqb : flambda_type -> flambda_type -> bool **)

let rec type_eqb t0 t' =
  match t0 with
  | Type_value ty -> (
    match t' with
    | Type_value ty' -> descr_eqb head_of_kind_value_eqb ty ty'
    | _ -> false)
  | Type_naked_immediate ty -> (
    match t' with
    | Type_naked_immediate ty' ->
      descr_eqb head_of_kind_naked_immediate_eqb ty ty'
    | _ -> false)
  | Type_naked_number (k, ty) -> (
    match t' with
    | Type_naked_number (k', ty') ->
      naked_number_kind_eqb k k' && descr_eqb (Keyed.same_keys_b Z.eqb) ty ty'
    | _ -> false)
  | Type_rec_info ty -> (
    match t' with
    | Type_rec_info ty' -> descr_eqb (fun _ _ -> true) ty ty'
    | _ -> false)
  | Type_region ty -> (
    match t' with
    | Type_region ty' -> descr_eqb (fun _ _ -> true) ty ty'
    | _ -> false)

(** val head_of_kind_value_eqb : head_of_kind_value -> head_of_kind_value ->
    bool **)

and head_of_kind_value_eqb head head' =
  let (Head_of_kind_value (non_null, is_null0)) = head in
  let (Head_of_kind_value (non_null', is_null')) = head' in
  or_unknown_or_bottom_eqb head_of_kind_value_non_null_eqb non_null non_null'
  && is_null_eqb is_null0 is_null'

(** val head_of_kind_value_non_null_eqb : head_of_kind_value_non_null ->
    head_of_kind_value_non_null -> bool **)

and head_of_kind_value_non_null_eqb head head' =
  match head with
  | Variant (is_int, immediates, get_tag, blocks, extensions, is_unique) -> (
    match head' with
    | Variant (is_int', immediates', get_tag', blocks', extensions', is_unique')
      ->
      ((((option_variable_eqb is_int is_int'
         && or_unknown_eqb type_eqb immediates immediates')
        && option_variable_eqb get_tag get_tag')
       && or_unknown_eqb row_like_for_blocks_eqb blocks blocks')
      && variant_extensions_eqb env_extension_eqb extensions extensions')
      && eqb is_unique is_unique'
    | _ -> false)
  | Mutable_block alloc_mode -> (
    match head' with
    | Mutable_block alloc_mode' -> alloc_mode_eqb alloc_mode alloc_mode'
    | _ -> false)
  | Boxed_number (k, contents, alloc_mode) -> (
    match head' with
    | Boxed_number (k', contents', alloc_mode') ->
      (boxable_number_eqb k k' && type_eqb contents contents')
      && alloc_mode_eqb alloc_mode alloc_mode'
    | _ -> false)
  | Closures (by_function_slot, alloc_mode) -> (
    match head' with
    | Closures (by_function_slot', alloc_mode') ->
      row_like_for_closures_eqb by_function_slot by_function_slot'
      && alloc_mode_eqb alloc_mode alloc_mode'
    | _ -> false)
  | String strs -> (
    match head' with
    | String strs' -> Keyed.same_keys_b (lists_eqb Z.eqb) strs strs'
    | _ -> false)

(** val row_like_for_blocks_eqb : row_like_for_blocks -> row_like_for_blocks ->
    bool **)

and row_like_for_blocks_eqb blocks blocks' =
  let (Row_like_for_blocks (known_tags, other_tags, alloc_mode)) = blocks in
  let (Row_like_for_blocks (known_tags', other_tags', alloc_mode')) = blocks' in
  (map_eqb Z.eqb
     (or_unknown_eqb (block_case_eqb type_eqb env_extension_eqb))
     known_tags known_tags'
  && or_bottom_eqb
       (block_case_eqb type_eqb env_extension_eqb)
       other_tags other_tags')
  && alloc_mode_eqb alloc_mode alloc_mode'

(** val row_like_for_closures_eqb : row_like_for_closures ->
    row_like_for_closures -> bool **)

and row_like_for_closures_eqb closures closures' =
  let (Row_like_for_closures known_closures) = closures in
  let (Row_like_for_closures known_closures') = closures' in
  map_eqb Pos.eqb
    (closures_case_eqb closures_entry_eqb env_extension_eqb)
    known_closures known_closures'

(** val closures_entry_eqb : closures_entry -> closures_entry -> bool **)

and closures_entry_eqb entry entry' =
  let (Closures_entry (function_types, closure_types, value_slot_types)) =
    entry
  in
  let (Closures_entry (function_types', closure_types', value_slot_types')) =
    entry'
  in
  (map_eqb Pos.eqb
     (or_unknown_eqb function_type_eqb)
     function_types function_types'
  && map_eqb Pos.eqb type_eqb closure_types closure_types')
  && map_eqb Pos.eqb type_eqb value_slot_types value_slot_types'

(** val env_extension_eqb : env_extension -> env_extension -> bool **)

and env_extension_eqb extension extension' =
  let (Env_extension equations) = extension in
  let (Env_extension equations') = extension' in
  map_eqb name_eqb type_eqb equations equations'
