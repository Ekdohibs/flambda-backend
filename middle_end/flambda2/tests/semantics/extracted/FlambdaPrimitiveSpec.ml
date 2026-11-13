open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaValueSpec
open List
open ListDef
open OxCamlOption

(** val in_range : coq_Z -> coq_Z -> bool **)

let in_range width z =
  Z.leb (Z.opp (Z.pow (Zpos (Coq_xO Coq_xH)) (Z.sub width (Zpos Coq_xH)))) z
  && Z.ltb z (Z.pow (Zpos (Coq_xO Coq_xH)) (Z.sub width (Zpos Coq_xH)))

(** val wrap : coq_Z -> coq_Z -> coq_Z **)

let wrap width z =
  Z.sub
    (Z.modulo
       (Z.add z (Z.pow (Zpos (Coq_xO Coq_xH)) (Z.sub width (Zpos Coq_xH))))
       (Z.pow (Zpos (Coq_xO Coq_xH)) width))
    (Z.pow (Zpos (Coq_xO Coq_xH)) (Z.sub width (Zpos Coq_xH)))

(** val unsigned : coq_Z -> coq_Z -> coq_Z **)

let unsigned width z =
  if Z.ltb z Z0 then Z.add z (Z.pow (Zpos (Coq_xO Coq_xH)) width) else z

(** val reading : coq_Z -> signedness -> coq_Z -> coq_Z **)

let reading width s z =
  match s with Signed -> z | Unsigned -> unsigned width z

(** val int_width : standard_int -> coq_Z **)

let int_width = function
  | Standard_tagged_immediate ->
    Zpos (Coq_xI (Coq_xI (Coq_xI (Coq_xI (Coq_xI Coq_xH)))))
  | Standard_naked_immediate ->
    Zpos (Coq_xI (Coq_xI (Coq_xI (Coq_xI (Coq_xI Coq_xH)))))
  | Standard_naked_int8 -> Zpos (Coq_xO (Coq_xO (Coq_xO Coq_xH)))
  | Standard_naked_int16 -> Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))
  | Standard_naked_int32 ->
    Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))
  | _ -> Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))

(** val const_of_int : standard_int -> coq_Z -> const **)

let const_of_int kind i =
  match kind with
  | Standard_tagged_immediate -> Tagged_immediate i
  | Standard_naked_immediate -> Naked_immediate i
  | Standard_naked_int8 -> Naked_int8 i
  | Standard_naked_int16 -> Naked_int16 i
  | Standard_naked_int32 -> Naked_int32 i
  | Standard_naked_int64 -> Naked_int64 i
  | Standard_naked_nativeint -> Naked_nativeint i

(** val int_of_value : standard_int -> value -> coq_Z option **)

let int_of_value kind v =
  match kind with
  | Standard_tagged_immediate -> (
    match v with
    | Value_const c -> (
      match c with
      | Tagged_immediate i ->
        if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_immediate -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_immediate i ->
        if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_int8 -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_int8 i -> if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_int16 -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_int16 i -> if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_int32 -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_int32 i -> if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_int64 -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_int64 i -> if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)
  | Standard_naked_nativeint -> (
    match v with
    | Value_const c -> (
      match c with
      | Naked_nativeint i ->
        if in_range (int_width kind) i then Some i else None
      | _ -> None)
    | _ -> None)

(** val is_tagged_immediate : value -> bool **)

let is_tagged_immediate v = is_some (int_of_value Standard_tagged_immediate v)

(** val eval_int_arith : coq_Z -> binary_int_arith_op -> coq_Z -> coq_Z -> coq_Z
    option **)

let eval_int_arith width op a b =
  match op with
  | Add -> Some (wrap width (Z.add a b))
  | Sub -> Some (wrap width (Z.sub a b))
  | Mul -> Some (wrap width (Z.mul a b))
  | Div s ->
    if Z.eqb b Z0
    then None
    else Some (wrap width (Z.quot (reading width s a) (reading width s b)))
  | Mod s ->
    if Z.eqb b Z0
    then None
    else Some (wrap width (Z.rem (reading width s a) (reading width s b)))
  | And -> Some (wrap width (Z.coq_land a b))
  | Or -> Some (wrap width (Z.coq_lor a b))
  | Xor -> Some (wrap width (Z.coq_lxor a b))

(** val little_endian : coq_Z list -> coq_Z **)

let rec little_endian = function
  | [] -> Z0
  | byte :: rest ->
    Z.add
      (Z.modulo byte
         (Zpos
            (Coq_xO
               (Coq_xO
                  (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))))))
      (Z.mul
         (Zpos
            (Coq_xO
               (Coq_xO
                  (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))))
         (little_endian rest))

(** val bytes_of : nat -> coq_Z -> coq_Z list **)

let rec bytes_of count z =
  match count with
  | O -> []
  | S count0 ->
    Z.modulo z
      (Zpos
         (Coq_xO
            (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))))
    :: bytes_of count0
         (Z.div z
            (Zpos
               (Coq_xO
                  (Coq_xO
                     (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))))))

(** val swap_bytes : nat -> coq_Z -> coq_Z **)

let swap_bytes count z = little_endian (rev (bytes_of count z))

(** val eval_unary_int_arith : standard_int -> unary_int_arith_op -> coq_Z ->
    coq_Z **)

let eval_unary_int_arith kind op a =
  match op with
  | Not -> wrap (int_width kind) (Z.lnot a)
  | Swap_byte_endianness -> (
    match kind with
    | Standard_tagged_immediate -> swap_bytes (S (S O)) a
    | Standard_naked_immediate -> swap_bytes (S (S O)) a
    | Standard_naked_int8 -> wrap (Zpos (Coq_xO (Coq_xO (Coq_xO Coq_xH)))) a
    | Standard_naked_int16 ->
      wrap (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))
        (swap_bytes (S (S O)) a)
    | Standard_naked_int32 ->
      wrap (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))
        (swap_bytes (S (S (S (S O)))) a)
    | _ ->
      wrap (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
        (swap_bytes (S (S (S (S (S (S (S (S O)))))))) a))

(** val shift_bound : standard_int -> coq_Z **)

let shift_bound kind =
  match kind with
  | Standard_tagged_immediate ->
    Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))
  | _ -> int_width kind

(** val eval_int_shift : coq_Z -> int_shift_op -> coq_Z -> coq_Z -> coq_Z **)

let eval_int_shift width op a amount =
  match op with
  | Lsl -> wrap width (Z.shiftl a amount)
  | Lsr -> wrap width (Z.shiftr (unsigned width a) amount)
  | Asr -> wrap width (Z.shiftr a amount)

(** val eval_int_comparison : coq_Z -> int_comparison -> coq_Z -> coq_Z -> bool
    **)

let eval_int_comparison width c a b =
  match c with
  | Int_eq -> Z.eqb a b
  | Int_neq -> negb (Z.eqb a b)
  | Int_lt s -> Z.ltb (reading width s a) (reading width s b)
  | Int_gt s -> Z.ltb (reading width s b) (reading width s a)
  | Int_le s -> Z.leb (reading width s a) (reading width s b)
  | Int_ge s -> Z.leb (reading width s b) (reading width s a)

(** val eval_int_comp : coq_Z -> int_comparison_behaviour -> coq_Z -> coq_Z ->
    coq_Z **)

let eval_int_comp width behaviour a b =
  match behaviour with
  | Yielding_bool c ->
    if eval_int_comparison width c a b then Zpos Coq_xH else Z0
  | Yielding_int_like_compare_functions s -> (
    match Z.compare (reading width s a) (reading width s b) with
    | Eq -> Z0
    | Lt -> Zneg Coq_xH
    | Gt -> Zpos Coq_xH)

(** val array_kind_of_load_kind : array_load_kind -> array_kind **)

let array_kind_of_load_kind = function
  | Load_immediates -> Array_immediates
  | Load_gc_ignorable_values -> Array_gc_ignorable_values
  | Load_values -> Array_values

(** val array_element_ok : array_kind -> value -> bool **)

let array_element_ok kind v =
  match kind with
  | Array_immediates -> is_tagged_immediate v
  | _ -> kind_eqb (kind_of_value v) Value

(** val array_made : array_kind -> value list -> value list option **)

let array_made kind vs =
  if forallb (array_element_ok kind) vs then Some vs else None

(** val array_length_of : value list -> value option **)

let array_length_of elements =
  let length0 = Z.of_nat (length elements) in
  if in_range (int_width Standard_tagged_immediate) length0
  then Some (Value_const (Tagged_immediate length0))
  else None

(** val array_index : value list -> value -> nat option **)

let array_index elements index =
  match int_of_value Standard_tagged_immediate index with
  | Some i ->
    if Z.leb Z0 i && Z.ltb i (Z.of_nat (length elements))
    then Some (Z.to_nat i)
    else None
  | None -> None

(** val array_loaded : array_load_kind -> value list -> value -> value option **)

let array_loaded load_kind elements index =
  match array_index elements index with
  | Some i -> (
    match nth_error elements i with
    | Some v ->
      if array_element_ok (array_kind_of_load_kind load_kind) v
      then Some v
      else None
    | None -> None)
  | None -> None

(** val bound_if_symbol : static_env -> value -> bool **)

let bound_if_symbol statics = function
  | Value_symbol s -> is_some (lookup s statics.static_values)
  | _ -> true

(** val naked_bool : bool -> value **)

let naked_bool b = Value_const (Naked_immediate (if b then Zpos Coq_xH else Z0))

(** val eval_is_int : static_env -> bool -> value -> bool option **)

let eval_is_int statics variant_only v =
  if is_tagged_immediate v
  then Some true
  else if
    if variant_only
    then is_some (block_of statics v)
    else is_allocated statics v
  then Some false
  else None

(** val eval_is_null : static_env -> value -> bool option **)

let eval_is_null statics v =
  match v with
  | Value_const c -> (
    match c with
    | Null -> Some true
    | _ ->
      if is_tagged_immediate v || is_allocated statics v
      then Some false
      else None)
  | _ ->
    if is_tagged_immediate v || is_allocated statics v then Some false else None

(** val is_number_of_kind : boxable_number -> const -> bool **)

let is_number_of_kind kind c =
  match kind with
  | Boxable_naked_float32 -> (
    match c with Naked_float32 _ -> true | _ -> false)
  | Boxable_naked_float -> ( match c with Naked_float _ -> true | _ -> false)
  | Boxable_naked_int32 -> ( match c with Naked_int32 _ -> true | _ -> false)
  | Boxable_naked_int64 -> ( match c with Naked_int64 _ -> true | _ -> false)
  | Boxable_naked_nativeint -> (
    match c with Naked_nativeint _ -> true | _ -> false)
  | Boxable_naked_vec128 -> (
    match c with Naked_vec128 _ -> true | _ -> false)
  | Boxable_naked_vec256 -> (
    match c with Naked_vec256 _ -> true | _ -> false)
  | Boxable_naked_vec512 -> (
    match c with Naked_vec512 _ -> true | _ -> false)
  | Boxable_naked_mask -> ( match c with Naked_mask _ -> true | _ -> false)

(** val known_is : coq_Z option -> coq_Z -> bool **)

let known_is known actual =
  match known with Some expected -> Z.eqb expected actual | None -> true

(** val eval_block_load : static_env -> block_access_kind -> coq_Z -> value ->
    value option **)

let eval_block_load statics kind field v =
  let (Block_access_values (tag, size, field_kind)) = kind in
  match block_of statics v with
  | Some p ->
    let block_tag, fields = p in
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
  | None -> None

(** val eval_reinterpret_64_bit_word : reinterpret_64_bit_word -> value -> value
    option **)

let eval_reinterpret_64_bit_word reinterpret v =
  match reinterpret with
  | Tagged_int63_as_unboxed_int64 -> (
    match int_of_value Standard_tagged_immediate v with
    | Some i ->
      Some
        (Value_const
           (Naked_int64 (Z.add (Z.mul (Zpos (Coq_xO Coq_xH)) i) (Zpos Coq_xH))))
    | None -> None)
  | Unboxed_int64_as_tagged_int63 -> (
    match int_of_value Standard_naked_int64 v with
    | Some word ->
      Some (Value_const (Tagged_immediate (Z.div word (Zpos (Coq_xO Coq_xH)))))
    | None -> None)

(** val eval_unary_primitive : static_env -> unary_primitive -> value -> value
    option **)

let eval_unary_primitive statics p v =
  match p with
  | Block_load (kind, _, field) -> eval_block_load statics kind field v
  | Is_int variant_only ->
    option_map naked_bool (eval_is_int statics variant_only v)
  | Is_null -> option_map naked_bool (eval_is_null statics v)
  | Get_tag -> (
    match block_of statics v with
    | Some p0 ->
      let tag, _ = p0 in
      Some (Value_const (Naked_immediate tag))
    | None -> None)
  | Array_length _ -> (
    match array_of statics v with
    | Some elements -> array_length_of elements
    | None -> None)
  | String_length -> (
    match string_of statics v with
    | Some bytes ->
      let length0 = Z.of_nat (length bytes) in
      if in_range (int_width Standard_naked_immediate) length0
      then Some (Value_const (Naked_immediate length0))
      else None
    | None -> None)
  | Opaque_identity kind ->
    if kind_eqb (kind_of_value v) kind && bound_if_symbol statics v
    then Some v
    else None
  | Unary_int_arith (kind, op) -> (
    match int_of_value kind v with
    | Some a ->
      Some (Value_const (const_of_int kind (eval_unary_int_arith kind op a)))
    | None -> None)
  | Num_conv (src, dst) -> (
    match int_of_value src v with
    | Some i -> Some (Value_const (const_of_int dst (wrap (int_width dst) i)))
    | None -> None)
  | Boolean_not -> (
    match int_of_value Standard_tagged_immediate v with
    | Some i ->
      if Z.eqb i Z0
      then Some (Value_const (Tagged_immediate (Zpos Coq_xH)))
      else if Z.eqb i (Zpos Coq_xH)
      then Some (Value_const (Tagged_immediate Z0))
      else None
    | None -> None)
  | Unbox_number kind -> (
    match boxed_number_of statics v with
    | Some c -> if is_number_of_kind kind c then Some (Value_const c) else None
    | None -> None)
  | Box_number kind -> (
    match v with
    | Value_const c ->
      if is_number_of_kind kind c then Some (Value_boxed_number c) else None
    | _ -> None)
  | Untag_immediate -> (
    match int_of_value Standard_tagged_immediate v with
    | Some i -> Some (Value_const (Naked_immediate i))
    | None -> None)
  | Tag_immediate -> (
    match int_of_value Standard_naked_immediate v with
    | Some i -> Some (Value_const (Tagged_immediate i))
    | None -> None)
  | Reinterpret_64_bit_word reinterpret ->
    eval_reinterpret_64_bit_word reinterpret v

(** val byte_width_of_string_accessor_width : string_accessor_width -> nat **)

let byte_width_of_string_accessor_width = function
  | Eight -> S O
  | Eight_signed -> S O
  | Thirty_two -> S (S (S (S O)))
  | Sixty_four -> S (S (S (S (S (S (S (S O)))))))
  | _ -> S (S O)

(** val eval_string_load : static_env -> string_accessor_width -> value -> value
    -> value option **)

let eval_string_load statics width v1 v2 =
  match string_of statics v1 with
  | Some bytes -> (
    match int_of_value Standard_naked_nativeint v2 with
    | Some index ->
      let count = byte_width_of_string_accessor_width width in
      if
        Z.leb Z0 index
        && Z.leb (Z.add index (Z.of_nat count)) (Z.of_nat (length bytes))
      then
        let read =
          little_endian (firstn count (skipn (Z.to_nat index) bytes))
        in
        Some
          (Value_const
             (match width with
             | Eight_signed ->
               Naked_int8 (wrap (Zpos (Coq_xO (Coq_xO (Coq_xO Coq_xH)))) read)
             | Sixteen_signed ->
               Naked_int16
                 (wrap (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))) read)
             | Thirty_two ->
               Naked_int32
                 (wrap
                    (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))
                    read)
             | Sixty_four ->
               Naked_int64
                 (wrap
                    (Zpos
                       (Coq_xO
                          (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
                    read)
             | _ -> Naked_immediate read))
      else None
    | None -> None)
  | None -> None

(** val eval_phys_equal : static_env -> value -> value -> bool option **)

let eval_phys_equal statics v1 v2 =
  match int_of_value Standard_tagged_immediate v1 with
  | Some a -> (
    match int_of_value Standard_tagged_immediate v2 with
    | Some b -> Some (Z.eqb a b)
    | None ->
      if is_allocated statics v2 || is_imported_pointer statics v2
      then Some false
      else None)
  | None -> (
    match int_of_value Standard_tagged_immediate v2 with
    | Some _ ->
      if is_allocated statics v1 || is_imported_pointer statics v1
      then Some false
      else None
    | None -> (
      match identity_of statics v1 with
      | Some s1 -> (
        match identity_of statics v2 with
        | Some s2 -> Some (Pos.eqb s1 s2)
        | None ->
          if is_allocated statics v2 || is_imported_pointer statics v2
          then Some false
          else None)
      | None -> (
        match identity_of statics v2 with
        | Some _ ->
          if is_allocated statics v1 || is_imported_pointer statics v1
          then Some false
          else None
        | None -> None)))

(** val eval_binary_primitive : static_env -> binary_primitive -> value -> value
    -> value option **)

let eval_binary_primitive statics p v1 v2 =
  match p with
  | Array_load (_, load_kind, _) -> (
    match array_of statics v1 with
    | Some elements -> array_loaded load_kind elements v2
    | None -> None)
  | String_or_bigstring_load width -> eval_string_load statics width v1 v2
  | Phys_equal op -> (
    match eval_phys_equal statics v1 v2 with
    | Some same ->
      Some
        (naked_bool (match op with Phys_eq -> same | Phys_neq -> negb same))
    | None -> None)
  | Int_arith (kind, op) -> (
    match int_of_value kind v1 with
    | Some a -> (
      match int_of_value kind v2 with
      | Some b -> (
        match eval_int_arith (int_width kind) op a b with
        | Some result -> Some (Value_const (const_of_int kind result))
        | None -> None)
      | None -> None)
    | None -> None)
  | Int_shift (kind, op) -> (
    match int_of_value kind v1 with
    | Some a -> (
      match int_of_value Standard_naked_immediate v2 with
      | Some amount ->
        if Z.leb Z0 amount && Z.ltb amount (shift_bound kind)
        then
          Some
            (Value_const
               (const_of_int kind (eval_int_shift (int_width kind) op a amount)))
        else None
      | None -> None)
    | None -> None)
  | Int_comp (kind, behaviour) -> (
    match int_of_value kind v1 with
    | Some a -> (
      match int_of_value kind v2 with
      | Some b ->
        Some
          (Value_const
             (Naked_immediate (eval_int_comp (int_width kind) behaviour a b)))
      | None -> None)
    | None -> None)

(** val eval_variadic_primitive : static_env -> variadic_primitive -> value list
    -> value option **)

let eval_variadic_primitive statics p vs =
  match p with
  | Make_block kind ->
    let (Block_kind_values tag) = kind in
    if
      forallb (fun v -> kind_eqb (kind_of_value v) Value) vs
      && forallb (bound_if_symbol statics) vs
    then Some (Value_block (tag, vs))
    else None
  | Make_immutable_array kind -> (
    match array_made kind vs with
    | Some elements ->
      if forallb (bound_if_symbol statics) vs
      then Some (Value_immutable_array elements)
      else None
    | None -> None)

(** val eval_prim : static_env -> prim -> value list -> value option **)

let eval_prim statics p vs =
  match p with
  | Unary p0 -> (
    match vs with
    | [] -> None
    | v :: l -> (
      match l with [] -> eval_unary_primitive statics p0 v | _ :: _ -> None))
  | Binary p0 -> (
    match vs with
    | [] -> None
    | v1 :: l -> (
      match l with
      | [] -> None
      | v2 :: l0 -> (
        match l0 with
        | [] -> eval_binary_primitive statics p0 v1 v2
        | _ :: _ -> None)))
  | Variadic p0 -> eval_variadic_primitive statics p0 vs
  | _ -> None

(** val tag_in_range : coq_Z -> bool **)

let tag_in_range tag =
  Z.leb Z0 tag
  && Z.ltb tag
       (Zpos
          (Coq_xI (Coq_xI (Coq_xO (Coq_xI (Coq_xI (Coq_xI (Coq_xI Coq_xH))))))))

(** val byte_in_range : coq_Z -> bool **)

let byte_in_range byte =
  Z.leb Z0 byte
  && Z.ltb byte
       (Zpos
          (Coq_xO
             (Coq_xO
                (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))))

(** val ocaml_int_in_range : coq_Z -> bool **)

let ocaml_int_in_range i = in_range (int_width Standard_tagged_immediate) i

(** val const_in_range : const -> bool **)

let const_in_range = function
  | Naked_immediate i -> ocaml_int_in_range i
  | Tagged_immediate i -> ocaml_int_in_range i
  | Naked_float32 i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))) i
  | Naked_float i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
      i
  | Naked_int8 i -> in_range (Zpos (Coq_xO (Coq_xO (Coq_xO Coq_xH)))) i
  | Naked_int16 i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))) i
  | Naked_int32 i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))) i
  | Naked_int64 i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
      i
  | Naked_nativeint i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
      i
  | Naked_vec128 bits ->
    in_range
      (Zpos
         (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))))
      bits
  | Naked_vec256 bits ->
    in_range
      (Zpos
         (Coq_xO
            (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))))
      bits
  | Naked_vec512 bits ->
    in_range
      (Zpos
         (Coq_xO
            (Coq_xO
               (Coq_xO
                  (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH))))))))))
      bits
  | Naked_mask i ->
    in_range (Zpos (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO (Coq_xO Coq_xH)))))))
      i
  | _ -> true

(** val known_in_range : (coq_Z -> bool) -> coq_Z option -> bool **)

let known_in_range test = function Some z -> test z | None -> true

(** val unary_primitive_in_range : unary_primitive -> bool **)

let unary_primitive_in_range = function
  | Block_load (kind, _, field) ->
    let (Block_access_values (tag, size, _)) = kind in
    (known_in_range tag_in_range tag && known_in_range ocaml_int_in_range size)
    && ocaml_int_in_range field
  | _ -> true

(** val binary_primitive_in_range : binary_primitive -> bool **)

let binary_primitive_in_range _ = true

(** val variadic_primitive_in_range : variadic_primitive -> bool **)

let variadic_primitive_in_range = function
  | Make_block kind ->
    let (Block_kind_values tag) = kind in
    tag_in_range tag
  | Make_immutable_array _ -> true

(** val access_in_range : block_access_kind -> coq_Z -> bool **)

let access_in_range kind field =
  let (Block_access_values (tag, size, _)) = kind in
  (known_in_range tag_in_range tag && known_in_range ocaml_int_in_range size)
  && ocaml_int_in_range field

(** val heap_primitive_in_range : heap_primitive -> bool **)

let heap_primitive_in_range = function
  | Make_block_with_identity (_, tag) -> tag_in_range tag
  | Block_set (kind, _, field) -> access_in_range kind field
  | _ -> true

(** val prim_in_range : prim -> bool **)

let prim_in_range = function
  | Unary p0 -> unary_primitive_in_range p0
  | Binary p0 -> binary_primitive_in_range p0
  | Variadic p0 -> variadic_primitive_in_range p0
  | Heap p0 -> heap_primitive_in_range p0
  | Region_primitive _ -> true

(** val naked_immediate_kind : flambda_kind **)

let naked_immediate_kind = Naked_number Kind_naked_immediate

(** val arg_kind_of_unary_primitive : unary_primitive -> flambda_kind **)

let arg_kind_of_unary_primitive = function
  | Opaque_identity kind -> kind
  | Unary_int_arith (kind, _) -> kind_of_standard_int kind
  | Num_conv (src, _) -> kind_of_standard_int src
  | Box_number kind -> unboxed_kind kind
  | Tag_immediate -> naked_immediate_kind
  | Reinterpret_64_bit_word reinterpret -> (
    match reinterpret with
    | Tagged_int63_as_unboxed_int64 -> Value
    | Unboxed_int64_as_tagged_int63 -> Naked_number Kind_naked_int64)
  | _ -> Value

(** val result_kind_of_unary_primitive : unary_primitive -> flambda_kind **)

let result_kind_of_unary_primitive = function
  | Block_load (_, _, _) -> Value
  | Array_length _ -> Value
  | Opaque_identity kind -> kind
  | Unary_int_arith (kind, _) -> kind_of_standard_int kind
  | Num_conv (_, dst) -> kind_of_standard_int dst
  | Boolean_not -> Value
  | Unbox_number kind -> unboxed_kind kind
  | Box_number _ -> Value
  | Tag_immediate -> Value
  | Reinterpret_64_bit_word reinterpret -> (
    match reinterpret with
    | Tagged_int63_as_unboxed_int64 -> Naked_number Kind_naked_int64
    | Unboxed_int64_as_tagged_int63 -> Value)
  | _ -> naked_immediate_kind

(** val args_kind_of_binary_primitive : binary_primitive -> flambda_kind *
    flambda_kind **)

let args_kind_of_binary_primitive = function
  | String_or_bigstring_load _ -> Value, Naked_number Kind_naked_nativeint
  | Int_arith (kind, _) -> kind_of_standard_int kind, kind_of_standard_int kind
  | Int_shift (kind, _) -> kind_of_standard_int kind, naked_immediate_kind
  | Int_comp (kind, _) -> kind_of_standard_int kind, kind_of_standard_int kind
  | _ -> Value, Value

(** val result_kind_of_binary_primitive : binary_primitive -> flambda_kind **)

let result_kind_of_binary_primitive = function
  | Array_load (_, _, _) -> Value
  | String_or_bigstring_load width -> (
    match width with
    | Eight_signed -> Naked_number Kind_naked_int8
    | Sixteen_signed -> Naked_number Kind_naked_int16
    | Thirty_two -> Naked_number Kind_naked_int32
    | Sixty_four -> Naked_number Kind_naked_int64
    | _ -> naked_immediate_kind)
  | Int_arith (kind, _) -> kind_of_standard_int kind
  | Int_shift (kind, _) -> kind_of_standard_int kind
  | _ -> naked_immediate_kind

(** val args_kind_of_variadic_primitive : variadic_primitive -> flambda_kind **)

let args_kind_of_variadic_primitive _ = Value

(** val result_kind_of_variadic_primitive : variadic_primitive -> flambda_kind **)

let result_kind_of_variadic_primitive _ = Value

(** val result_kind_of_heap_primitive : heap_primitive -> flambda_kind **)

let result_kind_of_heap_primitive _ = Value

(** val result_kind_of_region_primitive : region_primitive -> flambda_kind **)

let result_kind_of_region_primitive = function
  | Begin_region -> Region
  | Begin_try_region -> Region
  | _ -> Value

(** val prim_result_kind : prim -> flambda_kind **)

let prim_result_kind = function
  | Unary p0 -> result_kind_of_unary_primitive p0
  | Binary p0 -> result_kind_of_binary_primitive p0
  | Variadic p0 -> result_kind_of_variadic_primitive p0
  | Heap p0 -> result_kind_of_heap_primitive p0
  | Region_primitive p0 -> result_kind_of_region_primitive p0

(** val prim_args_well_kinded : prim -> flambda_kind list -> bool **)

let prim_args_well_kinded p kinds =
  match p with
  | Unary p0 -> (
    match kinds with
    | [] -> false
    | kind :: l -> (
      match l with
      | [] -> kind_eqb kind (arg_kind_of_unary_primitive p0)
      | _ :: _ -> false))
  | Binary p0 -> (
    match kinds with
    | [] -> false
    | kind1 :: l -> (
      match l with
      | [] -> false
      | kind2 :: l0 -> (
        match l0 with
        | [] ->
          kind_eqb kind1 (fst (args_kind_of_binary_primitive p0))
          && kind_eqb kind2 (snd (args_kind_of_binary_primitive p0))
        | _ :: _ -> false)))
  | Variadic p0 ->
    forallb
      (fun kind -> kind_eqb kind (args_kind_of_variadic_primitive p0))
      kinds
  | _ -> false

(** val prim_eq_dec : prim -> prim -> bool **)

let prim_eq_dec p p' =
  match p with
  | Unary p0 -> (
    match p' with
    | Unary p1 -> (
      match p0 with
      | Block_load (kind, mut, field) -> (
        match p1 with
        | Block_load (kind0, mut0, field0) ->
          let (Block_access_values (tag, size, field_kind)) = kind in
          let (Block_access_values (tag0, size0, field_kind0)) = kind0 in
          if
            match tag with
            | Some a -> (
              match tag0 with
              | Some z -> (
                match a with
                | Z0 -> ( match z with Z0 -> true | _ -> false)
                | Zpos p2 -> (
                  match z with
                  | Zpos p3 ->
                    let rec f p4 x =
                      match p4 with
                      | Coq_xI p5 -> (
                        match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                      | Coq_xO p5 -> (
                        match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                      | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                    in
                    f p2 p3
                  | _ -> false)
                | Zneg p2 -> (
                  match z with
                  | Zneg p3 ->
                    let rec f p4 x =
                      match p4 with
                      | Coq_xI p5 -> (
                        match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                      | Coq_xO p5 -> (
                        match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                      | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                    in
                    f p2 p3
                  | _ -> false))
              | None -> false)
            | None -> ( match tag0 with Some _ -> false | None -> true)
          then
            if
              match size with
              | Some a -> (
                match size0 with
                | Some z -> (
                  match a with
                  | Z0 -> ( match z with Z0 -> true | _ -> false)
                  | Zpos p2 -> (
                    match z with
                    | Zpos p3 ->
                      let rec f p4 x =
                        match p4 with
                        | Coq_xI p5 -> (
                          match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                        | Coq_xO p5 -> (
                          match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                        | Coq_xH -> (
                          match x with Coq_xH -> true | _ -> false)
                      in
                      f p2 p3
                    | _ -> false)
                  | Zneg p2 -> (
                    match z with
                    | Zneg p3 ->
                      let rec f p4 x =
                        match p4 with
                        | Coq_xI p5 -> (
                          match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                        | Coq_xO p5 -> (
                          match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                        | Coq_xH -> (
                          match x with Coq_xH -> true | _ -> false)
                      in
                      f p2 p3
                    | _ -> false))
                | None -> false)
              | None -> ( match size0 with Some _ -> false | None -> true)
            then
              match field_kind with
              | Any_value -> (
                match field_kind0 with
                | Any_value -> (
                  match mut with
                  | Mutable -> (
                    match mut0 with
                    | Mutable -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false)
                  | Immutable -> (
                    match mut0 with
                    | Immutable -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with
                    | Immutable_unique -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false))
                | Immediate -> false)
              | Immediate -> (
                match field_kind0 with
                | Any_value -> false
                | Immediate -> (
                  match mut with
                  | Mutable -> (
                    match mut0 with
                    | Mutable -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false)
                  | Immutable -> (
                    match mut0 with
                    | Immutable -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with
                    | Immutable_unique -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | _ -> false)))
            else false
          else false
        | _ -> false)
      | Is_int variant_only -> (
        match p1 with
        | Is_int variant_only0 ->
          if variant_only
          then if variant_only0 then true else false
          else if variant_only0
          then false
          else true
        | _ -> false)
      | Is_null -> ( match p1 with Is_null -> true | _ -> false)
      | Get_tag -> ( match p1 with Get_tag -> true | _ -> false)
      | Array_length kind -> (
        match p1 with
        | Array_length kind0 -> (
          match kind with
          | Array_kind kind1 -> (
            match kind0 with
            | Array_kind kind2 -> (
              match kind1 with
              | Array_immediates -> (
                match kind2 with Array_immediates -> true | _ -> false)
              | Array_gc_ignorable_values -> (
                match kind2 with
                | Array_gc_ignorable_values -> true
                | _ -> false)
              | Array_values -> (
                match kind2 with Array_values -> true | _ -> false))
            | Float_array_opt_dynamic -> false)
          | Float_array_opt_dynamic -> (
            match kind0 with
            | Array_kind _ -> false
            | Float_array_opt_dynamic -> true))
        | _ -> false)
      | String_length -> ( match p1 with String_length -> true | _ -> false)
      | Opaque_identity kind -> (
        match p1 with
        | Opaque_identity kind0 -> (
          match kind with
          | Value -> ( match kind0 with Value -> true | _ -> false)
          | Naked_number k -> (
            match kind0 with
            | Naked_number k0 -> (
              match k with
              | Kind_naked_immediate -> (
                match k0 with Kind_naked_immediate -> true | _ -> false)
              | Kind_naked_float32 -> (
                match k0 with Kind_naked_float32 -> true | _ -> false)
              | Kind_naked_float -> (
                match k0 with Kind_naked_float -> true | _ -> false)
              | Kind_naked_int8 -> (
                match k0 with Kind_naked_int8 -> true | _ -> false)
              | Kind_naked_int16 -> (
                match k0 with Kind_naked_int16 -> true | _ -> false)
              | Kind_naked_int32 -> (
                match k0 with Kind_naked_int32 -> true | _ -> false)
              | Kind_naked_int64 -> (
                match k0 with Kind_naked_int64 -> true | _ -> false)
              | Kind_naked_nativeint -> (
                match k0 with Kind_naked_nativeint -> true | _ -> false)
              | Kind_naked_vec128 -> (
                match k0 with Kind_naked_vec128 -> true | _ -> false)
              | Kind_naked_vec256 -> (
                match k0 with Kind_naked_vec256 -> true | _ -> false)
              | Kind_naked_vec512 -> (
                match k0 with Kind_naked_vec512 -> true | _ -> false)
              | Kind_naked_mask -> (
                match k0 with Kind_naked_mask -> true | _ -> false))
            | _ -> false)
          | Region -> ( match kind0 with Region -> true | _ -> false)
          | Rec_info -> ( match kind0 with Rec_info -> true | _ -> false))
        | _ -> false)
      | Unary_int_arith (kind, op) -> (
        match p1 with
        | Unary_int_arith (kind0, op0) -> (
          match kind with
          | Standard_tagged_immediate -> (
            match kind0 with
            | Standard_tagged_immediate -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_immediate -> (
            match kind0 with
            | Standard_naked_immediate -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_int8 -> (
            match kind0 with
            | Standard_naked_int8 -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_int16 -> (
            match kind0 with
            | Standard_naked_int16 -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_int32 -> (
            match kind0 with
            | Standard_naked_int32 -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_int64 -> (
            match kind0 with
            | Standard_naked_int64 -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false)
          | Standard_naked_nativeint -> (
            match kind0 with
            | Standard_naked_nativeint -> (
              match op with
              | Not -> (
                match op0 with Not -> true | Swap_byte_endianness -> false)
              | Swap_byte_endianness -> (
                match op0 with Not -> false | Swap_byte_endianness -> true))
            | _ -> false))
        | _ -> false)
      | Num_conv (src, dst) -> (
        match p1 with
        | Num_conv (src0, dst0) -> (
          match src with
          | Standard_tagged_immediate -> (
            match src0 with
            | Standard_tagged_immediate -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_immediate -> (
            match src0 with
            | Standard_naked_immediate -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int8 -> (
            match src0 with
            | Standard_naked_int8 -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int16 -> (
            match src0 with
            | Standard_naked_int16 -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int32 -> (
            match src0 with
            | Standard_naked_int32 -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int64 -> (
            match src0 with
            | Standard_naked_int64 -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false)
          | Standard_naked_nativeint -> (
            match src0 with
            | Standard_naked_nativeint -> (
              match dst with
              | Standard_tagged_immediate -> (
                match dst0 with Standard_tagged_immediate -> true | _ -> false)
              | Standard_naked_immediate -> (
                match dst0 with Standard_naked_immediate -> true | _ -> false)
              | Standard_naked_int8 -> (
                match dst0 with Standard_naked_int8 -> true | _ -> false)
              | Standard_naked_int16 -> (
                match dst0 with Standard_naked_int16 -> true | _ -> false)
              | Standard_naked_int32 -> (
                match dst0 with Standard_naked_int32 -> true | _ -> false)
              | Standard_naked_int64 -> (
                match dst0 with Standard_naked_int64 -> true | _ -> false)
              | Standard_naked_nativeint -> (
                match dst0 with Standard_naked_nativeint -> true | _ -> false))
            | _ -> false))
        | _ -> false)
      | Boolean_not -> ( match p1 with Boolean_not -> true | _ -> false)
      | Unbox_number kind -> (
        match p1 with
        | Unbox_number kind0 -> (
          match kind with
          | Boxable_naked_float32 -> (
            match kind0 with Boxable_naked_float32 -> true | _ -> false)
          | Boxable_naked_float -> (
            match kind0 with Boxable_naked_float -> true | _ -> false)
          | Boxable_naked_int32 -> (
            match kind0 with Boxable_naked_int32 -> true | _ -> false)
          | Boxable_naked_int64 -> (
            match kind0 with Boxable_naked_int64 -> true | _ -> false)
          | Boxable_naked_nativeint -> (
            match kind0 with Boxable_naked_nativeint -> true | _ -> false)
          | Boxable_naked_vec128 -> (
            match kind0 with Boxable_naked_vec128 -> true | _ -> false)
          | Boxable_naked_vec256 -> (
            match kind0 with Boxable_naked_vec256 -> true | _ -> false)
          | Boxable_naked_vec512 -> (
            match kind0 with Boxable_naked_vec512 -> true | _ -> false)
          | Boxable_naked_mask -> (
            match kind0 with Boxable_naked_mask -> true | _ -> false))
        | _ -> false)
      | Box_number kind -> (
        match p1 with
        | Box_number kind0 -> (
          match kind with
          | Boxable_naked_float32 -> (
            match kind0 with Boxable_naked_float32 -> true | _ -> false)
          | Boxable_naked_float -> (
            match kind0 with Boxable_naked_float -> true | _ -> false)
          | Boxable_naked_int32 -> (
            match kind0 with Boxable_naked_int32 -> true | _ -> false)
          | Boxable_naked_int64 -> (
            match kind0 with Boxable_naked_int64 -> true | _ -> false)
          | Boxable_naked_nativeint -> (
            match kind0 with Boxable_naked_nativeint -> true | _ -> false)
          | Boxable_naked_vec128 -> (
            match kind0 with Boxable_naked_vec128 -> true | _ -> false)
          | Boxable_naked_vec256 -> (
            match kind0 with Boxable_naked_vec256 -> true | _ -> false)
          | Boxable_naked_vec512 -> (
            match kind0 with Boxable_naked_vec512 -> true | _ -> false)
          | Boxable_naked_mask -> (
            match kind0 with Boxable_naked_mask -> true | _ -> false))
        | _ -> false)
      | Untag_immediate -> (
        match p1 with Untag_immediate -> true | _ -> false)
      | Tag_immediate -> ( match p1 with Tag_immediate -> true | _ -> false)
      | Reinterpret_64_bit_word reinterpret -> (
        match p1 with
        | Reinterpret_64_bit_word reinterpret0 -> (
          match reinterpret with
          | Tagged_int63_as_unboxed_int64 -> (
            match reinterpret0 with
            | Tagged_int63_as_unboxed_int64 -> true
            | Unboxed_int64_as_tagged_int63 -> false)
          | Unboxed_int64_as_tagged_int63 -> (
            match reinterpret0 with
            | Tagged_int63_as_unboxed_int64 -> false
            | Unboxed_int64_as_tagged_int63 -> true))
        | _ -> false))
    | _ -> false)
  | Binary p0 -> (
    match p' with
    | Binary p1 -> (
      match p0 with
      | Array_load (kind, load_kind, mut) -> (
        match p1 with
        | Array_load (kind0, load_kind0, mut0) -> (
          match kind with
          | Array_immediates -> (
            match kind0 with
            | Array_immediates -> (
              match load_kind with
              | Load_immediates -> (
                match load_kind0 with
                | Load_immediates -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_gc_ignorable_values -> (
                match load_kind0 with
                | Load_gc_ignorable_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_values -> (
                match load_kind0 with
                | Load_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false))
            | _ -> false)
          | Array_gc_ignorable_values -> (
            match kind0 with
            | Array_gc_ignorable_values -> (
              match load_kind with
              | Load_immediates -> (
                match load_kind0 with
                | Load_immediates -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_gc_ignorable_values -> (
                match load_kind0 with
                | Load_gc_ignorable_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_values -> (
                match load_kind0 with
                | Load_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false))
            | _ -> false)
          | Array_values -> (
            match kind0 with
            | Array_values -> (
              match load_kind with
              | Load_immediates -> (
                match load_kind0 with
                | Load_immediates -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_gc_ignorable_values -> (
                match load_kind0 with
                | Load_gc_ignorable_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false)
              | Load_values -> (
                match load_kind0 with
                | Load_values -> (
                  match mut with
                  | Mutable -> ( match mut0 with Mutable -> true | _ -> false)
                  | Immutable -> (
                    match mut0 with Immutable -> true | _ -> false)
                  | Immutable_unique -> (
                    match mut0 with Immutable_unique -> true | _ -> false))
                | _ -> false))
            | _ -> false))
        | _ -> false)
      | String_or_bigstring_load width -> (
        match p1 with
        | String_or_bigstring_load width0 -> (
          match width with
          | Eight -> ( match width0 with Eight -> true | _ -> false)
          | Eight_signed -> (
            match width0 with Eight_signed -> true | _ -> false)
          | Sixteen -> ( match width0 with Sixteen -> true | _ -> false)
          | Sixteen_signed -> (
            match width0 with Sixteen_signed -> true | _ -> false)
          | Thirty_two -> ( match width0 with Thirty_two -> true | _ -> false)
          | Sixty_four -> ( match width0 with Sixty_four -> true | _ -> false))
        | _ -> false)
      | Phys_equal op -> (
        match p1 with
        | Phys_equal op0 -> (
          match op with
          | Phys_eq -> ( match op0 with Phys_eq -> true | Phys_neq -> false)
          | Phys_neq -> ( match op0 with Phys_eq -> false | Phys_neq -> true))
        | _ -> false)
      | Int_arith (kind, op) -> (
        match p1 with
        | Int_arith (kind0, op0) -> (
          match kind with
          | Standard_tagged_immediate -> (
            match kind0 with
            | Standard_tagged_immediate -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_immediate -> (
            match kind0 with
            | Standard_naked_immediate -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int8 -> (
            match kind0 with
            | Standard_naked_int8 -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int16 -> (
            match kind0 with
            | Standard_naked_int16 -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int32 -> (
            match kind0 with
            | Standard_naked_int32 -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int64 -> (
            match kind0 with
            | Standard_naked_int64 -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false)
          | Standard_naked_nativeint -> (
            match kind0 with
            | Standard_naked_nativeint -> (
              match op with
              | Add -> ( match op0 with Add -> true | _ -> false)
              | Sub -> ( match op0 with Sub -> true | _ -> false)
              | Mul -> ( match op0 with Mul -> true | _ -> false)
              | Div s -> (
                match op0 with
                | Div s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | Mod s -> (
                match op0 with
                | Mod s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))
                | _ -> false)
              | And -> ( match op0 with And -> true | _ -> false)
              | Or -> ( match op0 with Or -> true | _ -> false)
              | Xor -> ( match op0 with Xor -> true | _ -> false))
            | _ -> false))
        | _ -> false)
      | Int_shift (kind, op) -> (
        match p1 with
        | Int_shift (kind0, op0) -> (
          match kind with
          | Standard_tagged_immediate -> (
            match kind0 with
            | Standard_tagged_immediate -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_immediate -> (
            match kind0 with
            | Standard_naked_immediate -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int8 -> (
            match kind0 with
            | Standard_naked_int8 -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int16 -> (
            match kind0 with
            | Standard_naked_int16 -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int32 -> (
            match kind0 with
            | Standard_naked_int32 -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_int64 -> (
            match kind0 with
            | Standard_naked_int64 -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false)
          | Standard_naked_nativeint -> (
            match kind0 with
            | Standard_naked_nativeint -> (
              match op with
              | Lsl -> ( match op0 with Lsl -> true | _ -> false)
              | Lsr -> ( match op0 with Lsr -> true | _ -> false)
              | Asr -> ( match op0 with Asr -> true | _ -> false))
            | _ -> false))
        | _ -> false)
      | Int_comp (kind, behaviour) -> (
        match p1 with
        | Int_comp (kind0, behaviour0) -> (
          match kind with
          | Standard_tagged_immediate -> (
            match kind0 with
            | Standard_tagged_immediate -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_immediate -> (
            match kind0 with
            | Standard_naked_immediate -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_int8 -> (
            match kind0 with
            | Standard_naked_int8 -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_int16 -> (
            match kind0 with
            | Standard_naked_int16 -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_int32 -> (
            match kind0 with
            | Standard_naked_int32 -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_int64 -> (
            match kind0 with
            | Standard_naked_int64 -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false)
          | Standard_naked_nativeint -> (
            match kind0 with
            | Standard_naked_nativeint -> (
              match behaviour with
              | Yielding_bool c -> (
                match behaviour0 with
                | Yielding_bool c0 -> (
                  match c with
                  | Int_eq -> ( match c0 with Int_eq -> true | _ -> false)
                  | Int_neq -> ( match c0 with Int_neq -> true | _ -> false)
                  | Int_lt s -> (
                    match c0 with
                    | Int_lt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_gt s -> (
                    match c0 with
                    | Int_gt s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_le s -> (
                    match c0 with
                    | Int_le s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false)
                  | Int_ge s -> (
                    match c0 with
                    | Int_ge s0 -> (
                      match s with
                      | Signed -> (
                        match s0 with Signed -> true | Unsigned -> false)
                      | Unsigned -> (
                        match s0 with Signed -> false | Unsigned -> true))
                    | _ -> false))
                | Yielding_int_like_compare_functions _ -> false)
              | Yielding_int_like_compare_functions s -> (
                match behaviour0 with
                | Yielding_bool _ -> false
                | Yielding_int_like_compare_functions s0 -> (
                  match s with
                  | Signed -> (
                    match s0 with Signed -> true | Unsigned -> false)
                  | Unsigned -> (
                    match s0 with Signed -> false | Unsigned -> true))))
            | _ -> false))
        | _ -> false))
    | _ -> false)
  | Variadic p0 -> (
    match p' with
    | Variadic p1 -> (
      match p0 with
      | Make_block kind -> (
        match p1 with
        | Make_block kind0 -> (
          let (Block_kind_values tag) = kind in
          let (Block_kind_values tag0) = kind0 in
          match tag with
          | Z0 -> ( match tag0 with Z0 -> true | _ -> false)
          | Zpos p2 -> (
            match tag0 with
            | Zpos p3 ->
              let rec f p4 x =
                match p4 with
                | Coq_xI p5 -> (
                  match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                | Coq_xO p5 -> (
                  match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
              in
              f p2 p3
            | _ -> false)
          | Zneg p2 -> (
            match tag0 with
            | Zneg p3 ->
              let rec f p4 x =
                match p4 with
                | Coq_xI p5 -> (
                  match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                | Coq_xO p5 -> (
                  match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
              in
              f p2 p3
            | _ -> false))
        | Make_immutable_array _ -> false)
      | Make_immutable_array kind -> (
        match p1 with
        | Make_block _ -> false
        | Make_immutable_array kind0 -> (
          match kind with
          | Array_immediates -> (
            match kind0 with Array_immediates -> true | _ -> false)
          | Array_gc_ignorable_values -> (
            match kind0 with Array_gc_ignorable_values -> true | _ -> false)
          | Array_values -> (
            match kind0 with Array_values -> true | _ -> false))))
    | _ -> false)
  | Heap p0 -> (
    match p' with
    | Heap p1 -> (
      match p0 with
      | Make_block_with_identity (mut, tag) -> (
        match p1 with
        | Make_block_with_identity (mut0, tag0) -> (
          match mut with
          | Cell_mutable -> (
            match mut0 with
            | Cell_mutable -> (
              match tag with
              | Z0 -> ( match tag0 with Z0 -> true | _ -> false)
              | Zpos p2 -> (
                match tag0 with
                | Zpos p3 ->
                  let rec f p4 x =
                    match p4 with
                    | Coq_xI p5 -> (
                      match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                    | Coq_xO p5 -> (
                      match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                    | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                  in
                  f p2 p3
                | _ -> false)
              | Zneg p2 -> (
                match tag0 with
                | Zneg p3 ->
                  let rec f p4 x =
                    match p4 with
                    | Coq_xI p5 -> (
                      match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                    | Coq_xO p5 -> (
                      match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                    | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                  in
                  f p2 p3
                | _ -> false))
            | Cell_immutable_unique -> false)
          | Cell_immutable_unique -> (
            match mut0 with
            | Cell_mutable -> false
            | Cell_immutable_unique -> (
              match tag with
              | Z0 -> ( match tag0 with Z0 -> true | _ -> false)
              | Zpos p2 -> (
                match tag0 with
                | Zpos p3 ->
                  let rec f p4 x =
                    match p4 with
                    | Coq_xI p5 -> (
                      match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                    | Coq_xO p5 -> (
                      match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                    | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                  in
                  f p2 p3
                | _ -> false)
              | Zneg p2 -> (
                match tag0 with
                | Zneg p3 ->
                  let rec f p4 x =
                    match p4 with
                    | Coq_xI p5 -> (
                      match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                    | Coq_xO p5 -> (
                      match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                    | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                  in
                  f p2 p3
                | _ -> false))))
        | _ -> false)
      | Block_set (kind, init, field) -> (
        match p1 with
        | Block_set (kind0, init0, field0) ->
          let (Block_access_values (tag, size, field_kind)) = kind in
          let (Block_access_values (tag0, size0, field_kind0)) = kind0 in
          if
            match tag with
            | Some a -> (
              match tag0 with
              | Some z -> (
                match a with
                | Z0 -> ( match z with Z0 -> true | _ -> false)
                | Zpos p2 -> (
                  match z with
                  | Zpos p3 ->
                    let rec f p4 x =
                      match p4 with
                      | Coq_xI p5 -> (
                        match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                      | Coq_xO p5 -> (
                        match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                      | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                    in
                    f p2 p3
                  | _ -> false)
                | Zneg p2 -> (
                  match z with
                  | Zneg p3 ->
                    let rec f p4 x =
                      match p4 with
                      | Coq_xI p5 -> (
                        match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                      | Coq_xO p5 -> (
                        match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                      | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
                    in
                    f p2 p3
                  | _ -> false))
              | None -> false)
            | None -> ( match tag0 with Some _ -> false | None -> true)
          then
            if
              match size with
              | Some a -> (
                match size0 with
                | Some z -> (
                  match a with
                  | Z0 -> ( match z with Z0 -> true | _ -> false)
                  | Zpos p2 -> (
                    match z with
                    | Zpos p3 ->
                      let rec f p4 x =
                        match p4 with
                        | Coq_xI p5 -> (
                          match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                        | Coq_xO p5 -> (
                          match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                        | Coq_xH -> (
                          match x with Coq_xH -> true | _ -> false)
                      in
                      f p2 p3
                    | _ -> false)
                  | Zneg p2 -> (
                    match z with
                    | Zneg p3 ->
                      let rec f p4 x =
                        match p4 with
                        | Coq_xI p5 -> (
                          match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                        | Coq_xO p5 -> (
                          match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                        | Coq_xH -> (
                          match x with Coq_xH -> true | _ -> false)
                      in
                      f p2 p3
                    | _ -> false))
                | None -> false)
              | None -> ( match size0 with Some _ -> false | None -> true)
            then
              match field_kind with
              | Any_value -> (
                match field_kind0 with
                | Any_value -> (
                  match init with
                  | Initialization -> (
                    match init0 with
                    | Initialization -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | Assignment -> false)
                  | Assignment -> (
                    match init0 with
                    | Initialization -> false
                    | Assignment -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))))
                | Immediate -> false)
              | Immediate -> (
                match field_kind0 with
                | Any_value -> false
                | Immediate -> (
                  match init with
                  | Initialization -> (
                    match init0 with
                    | Initialization -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false))
                    | Assignment -> false)
                  | Assignment -> (
                    match init0 with
                    | Initialization -> false
                    | Assignment -> (
                      match field with
                      | Z0 -> ( match field0 with Z0 -> true | _ -> false)
                      | Zpos p2 -> (
                        match field0 with
                        | Zpos p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)
                      | Zneg p2 -> (
                        match field0 with
                        | Zneg p3 ->
                          let rec f p4 x =
                            match p4 with
                            | Coq_xI p5 -> (
                              match x with Coq_xI p6 -> f p5 p6 | _ -> false)
                            | Coq_xO p5 -> (
                              match x with Coq_xO p6 -> f p5 p6 | _ -> false)
                            | Coq_xH -> (
                              match x with Coq_xH -> true | _ -> false)
                          in
                          f p2 p3
                        | _ -> false)))))
            else false
          else false
        | _ -> false)
      | Make_mutable_array kind -> (
        match p1 with
        | Make_mutable_array kind0 -> (
          match kind with
          | Array_immediates -> (
            match kind0 with Array_immediates -> true | _ -> false)
          | Array_gc_ignorable_values -> (
            match kind0 with Array_gc_ignorable_values -> true | _ -> false)
          | Array_values -> (
            match kind0 with Array_values -> true | _ -> false))
        | _ -> false)
      | Array_set (kind, set_kind) -> (
        match p1 with
        | Array_set (kind0, set_kind0) -> (
          match kind with
          | Array_immediates -> (
            match kind0 with
            | Array_immediates -> (
              match set_kind with
              | Set_immediates -> (
                match set_kind0 with Set_immediates -> true | _ -> false)
              | Set_gc_ignorable_values -> (
                match set_kind0 with
                | Set_gc_ignorable_values -> true
                | _ -> false)
              | Set_values init -> (
                match set_kind0 with
                | Set_values init0 -> (
                  match init with
                  | Initialization -> (
                    match init0 with
                    | Initialization -> true
                    | Assignment -> false)
                  | Assignment -> (
                    match init0 with
                    | Initialization -> false
                    | Assignment -> true))
                | _ -> false))
            | _ -> false)
          | Array_gc_ignorable_values -> (
            match kind0 with
            | Array_gc_ignorable_values -> (
              match set_kind with
              | Set_immediates -> (
                match set_kind0 with Set_immediates -> true | _ -> false)
              | Set_gc_ignorable_values -> (
                match set_kind0 with
                | Set_gc_ignorable_values -> true
                | _ -> false)
              | Set_values init -> (
                match set_kind0 with
                | Set_values init0 -> (
                  match init with
                  | Initialization -> (
                    match init0 with
                    | Initialization -> true
                    | Assignment -> false)
                  | Assignment -> (
                    match init0 with
                    | Initialization -> false
                    | Assignment -> true))
                | _ -> false))
            | _ -> false)
          | Array_values -> (
            match kind0 with
            | Array_values -> (
              match set_kind with
              | Set_immediates -> (
                match set_kind0 with Set_immediates -> true | _ -> false)
              | Set_gc_ignorable_values -> (
                match set_kind0 with
                | Set_gc_ignorable_values -> true
                | _ -> false)
              | Set_values init -> (
                match set_kind0 with
                | Set_values init0 -> (
                  match init with
                  | Initialization -> (
                    match init0 with
                    | Initialization -> true
                    | Assignment -> false)
                  | Assignment -> (
                    match init0 with
                    | Initialization -> false
                    | Assignment -> true))
                | _ -> false))
            | _ -> false))
        | _ -> false))
    | _ -> false)
  | Region_primitive p0 -> (
    match p' with
    | Region_primitive p1 -> (
      match p0 with
      | Begin_region -> ( match p1 with Begin_region -> true | _ -> false)
      | Begin_try_region -> (
        match p1 with Begin_try_region -> true | _ -> false)
      | End_region -> ( match p1 with End_region -> true | _ -> false)
      | End_try_region -> ( match p1 with End_try_region -> true | _ -> false)
      | Local_allocation -> (
        match p1 with Local_allocation -> true | _ -> false))
    | _ -> false)

(** val prim_eqb : prim -> prim -> bool **)

let prim_eqb p p' = if prim_eq_dec p p' then true else false
