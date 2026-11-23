let flat_suffix_element (kind : Flambda_kind.Naked_number_kind.t) :
    Flambda_kind.flat_suffix_element =
  match kind with
  | Naked_immediate -> Naked_immediate
  | Naked_float32 -> Naked_float32
  | Naked_float -> Naked_float
  | Naked_int8 -> Naked_int8
  | Naked_int16 -> Naked_int16
  | Naked_int32 -> Naked_int32
  | Naked_int64 -> Naked_int64
  | Naked_nativeint -> Naked_nativeint
  | Naked_vec128 -> Naked_vec128
  | Naked_vec256 -> Naked_vec256
  | Naked_vec512 -> Naked_vec512
  | Naked_mask -> Naked_mask

let scannable (kinds : Flambda_kind.t list) :
    (Flambda_kind.Scannable_block_shape.t, string) result =
  let rec value_prefix size (kinds : Flambda_kind.t list) =
    match kinds with
    | Value :: kinds -> value_prefix (size + 1) kinds
    | [] | (Naked_number _ | Region | Rec_info) :: _ -> size, kinds
  in
  let rec flat_suffix elements (kinds : Flambda_kind.t list) =
    match kinds with
    | [] -> Ok (List.rev elements)
    | Naked_number kind :: kinds ->
      flat_suffix (flat_suffix_element kind :: elements) kinds
    | Value :: _ -> Error "a field that is a value after one that is not"
    | (Region | Rec_info) :: _ ->
      Error "a field of a kind that a block has not got"
  in
  let value_prefix_size, kinds = value_prefix 0 kinds in
  match kinds with
  | [] -> Ok Value_only
  | _ :: _ ->
    Result.map
      (fun elements : Flambda_kind.Scannable_block_shape.t ->
        Mixed_record
          (Flambda_kind.Mixed_block_shape.from_prefix_size_and_suffix_elements
             value_prefix_size elements))
      (flat_suffix [] kinds)
