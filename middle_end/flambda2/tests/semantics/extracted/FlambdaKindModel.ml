open BinInt
open BinNums
open Datatypes
open List

type naked_number_kind =
  | Kind_naked_immediate
  | Kind_naked_float32
  | Kind_naked_float
  | Kind_naked_int8
  | Kind_naked_int16
  | Kind_naked_int32
  | Kind_naked_int64
  | Kind_naked_nativeint
  | Kind_naked_vec128
  | Kind_naked_vec256
  | Kind_naked_vec512
  | Kind_naked_mask

type flambda_kind =
  | Value
  | Naked_number of naked_number_kind
  | Region
  | Rec_info

(** val kind_eq_dec : flambda_kind -> flambda_kind -> bool **)

let kind_eq_dec k k' =
  match k with
  | Value -> ( match k' with Value -> true | _ -> false)
  | Naked_number k0 -> (
    match k' with
    | Naked_number k1 -> (
      match k0 with
      | Kind_naked_immediate -> (
        match k1 with Kind_naked_immediate -> true | _ -> false)
      | Kind_naked_float32 -> (
        match k1 with Kind_naked_float32 -> true | _ -> false)
      | Kind_naked_float -> (
        match k1 with Kind_naked_float -> true | _ -> false)
      | Kind_naked_int8 -> (
        match k1 with Kind_naked_int8 -> true | _ -> false)
      | Kind_naked_int16 -> (
        match k1 with Kind_naked_int16 -> true | _ -> false)
      | Kind_naked_int32 -> (
        match k1 with Kind_naked_int32 -> true | _ -> false)
      | Kind_naked_int64 -> (
        match k1 with Kind_naked_int64 -> true | _ -> false)
      | Kind_naked_nativeint -> (
        match k1 with Kind_naked_nativeint -> true | _ -> false)
      | Kind_naked_vec128 -> (
        match k1 with Kind_naked_vec128 -> true | _ -> false)
      | Kind_naked_vec256 -> (
        match k1 with Kind_naked_vec256 -> true | _ -> false)
      | Kind_naked_vec512 -> (
        match k1 with Kind_naked_vec512 -> true | _ -> false)
      | Kind_naked_mask -> (
        match k1 with Kind_naked_mask -> true | _ -> false))
    | _ -> false)
  | Region -> ( match k' with Region -> true | _ -> false)
  | Rec_info -> ( match k' with Rec_info -> true | _ -> false)

(** val kind_eqb : flambda_kind -> flambda_kind -> bool **)

let kind_eqb k k' = if kind_eq_dec k k' then true else false

type standard_int =
  | Standard_tagged_immediate
  | Standard_naked_immediate
  | Standard_naked_int8
  | Standard_naked_int16
  | Standard_naked_int32
  | Standard_naked_int64
  | Standard_naked_nativeint

(** val kind_of_standard_int : standard_int -> flambda_kind **)

let kind_of_standard_int = function
  | Standard_tagged_immediate -> Value
  | Standard_naked_immediate -> Naked_number Kind_naked_immediate
  | Standard_naked_int8 -> Naked_number Kind_naked_int8
  | Standard_naked_int16 -> Naked_number Kind_naked_int16
  | Standard_naked_int32 -> Naked_number Kind_naked_int32
  | Standard_naked_int64 -> Naked_number Kind_naked_int64
  | Standard_naked_nativeint -> Naked_number Kind_naked_nativeint

type boxable_number =
  | Boxable_naked_float32
  | Boxable_naked_float
  | Boxable_naked_int32
  | Boxable_naked_int64
  | Boxable_naked_nativeint
  | Boxable_naked_vec128
  | Boxable_naked_vec256
  | Boxable_naked_vec512
  | Boxable_naked_mask

(** val unboxed_kind : boxable_number -> flambda_kind **)

let unboxed_kind = function
  | Boxable_naked_float32 -> Naked_number Kind_naked_float32
  | Boxable_naked_float -> Naked_number Kind_naked_float
  | Boxable_naked_int32 -> Naked_number Kind_naked_int32
  | Boxable_naked_int64 -> Naked_number Kind_naked_int64
  | Boxable_naked_nativeint -> Naked_number Kind_naked_nativeint
  | Boxable_naked_vec128 -> Naked_number Kind_naked_vec128
  | Boxable_naked_vec256 -> Naked_number Kind_naked_vec256
  | Boxable_naked_vec512 -> Naked_number Kind_naked_vec512
  | Boxable_naked_mask -> Naked_number Kind_naked_mask

(** val boxable_number_of_kind : flambda_kind -> boxable_number option **)

let boxable_number_of_kind = function
  | Naked_number k0 -> (
    match k0 with
    | Kind_naked_float32 -> Some Boxable_naked_float32
    | Kind_naked_float -> Some Boxable_naked_float
    | Kind_naked_int32 -> Some Boxable_naked_int32
    | Kind_naked_int64 -> Some Boxable_naked_int64
    | Kind_naked_nativeint -> Some Boxable_naked_nativeint
    | Kind_naked_vec128 -> Some Boxable_naked_vec128
    | Kind_naked_vec256 -> Some Boxable_naked_vec256
    | Kind_naked_vec512 -> Some Boxable_naked_vec512
    | Kind_naked_mask -> Some Boxable_naked_mask
    | _ -> None)
  | _ -> None

(** val boxable_number_eq_dec : boxable_number -> boxable_number -> bool **)

let boxable_number_eq_dec n n' =
  match n with
  | Boxable_naked_float32 -> (
    match n' with Boxable_naked_float32 -> true | _ -> false)
  | Boxable_naked_float -> (
    match n' with Boxable_naked_float -> true | _ -> false)
  | Boxable_naked_int32 -> (
    match n' with Boxable_naked_int32 -> true | _ -> false)
  | Boxable_naked_int64 -> (
    match n' with Boxable_naked_int64 -> true | _ -> false)
  | Boxable_naked_nativeint -> (
    match n' with Boxable_naked_nativeint -> true | _ -> false)
  | Boxable_naked_vec128 -> (
    match n' with Boxable_naked_vec128 -> true | _ -> false)
  | Boxable_naked_vec256 -> (
    match n' with Boxable_naked_vec256 -> true | _ -> false)
  | Boxable_naked_vec512 -> (
    match n' with Boxable_naked_vec512 -> true | _ -> false)
  | Boxable_naked_mask -> (
    match n' with Boxable_naked_mask -> true | _ -> false)

(** val boxable_number_eqb : boxable_number -> boxable_number -> bool **)

let boxable_number_eqb n n' = if boxable_number_eq_dec n n' then true else false

type subkind =
  | Subkind_anything
  | Subkind_tagged_immediate
  | Subkind_boxed_number of boxable_number
  | Subkind_variant of coq_Z list * constructor_shapes

and constructor_shapes =
  | Shapes_nil
  | Shape_undetermined of coq_Z * constructor_shapes
  | Shape_determined of coq_Z * field_subkinds * constructor_shapes

and field_subkinds =
  | Fields_nil
  | Fields_cons of bool * subkind * field_subkinds

(** val subkind_eq_dec : subkind -> subkind -> bool **)

let rec subkind_eq_dec sk sk' =
  match sk with
  | Subkind_anything -> (
    match sk' with Subkind_anything -> true | _ -> false)
  | Subkind_tagged_immediate -> (
    match sk' with Subkind_tagged_immediate -> true | _ -> false)
  | Subkind_boxed_number n -> (
    match sk' with
    | Subkind_boxed_number n0 -> boxable_number_eq_dec n n0
    | _ -> false)
  | Subkind_variant (consts, non_consts) -> (
    match sk' with
    | Subkind_variant (consts0, non_consts0) ->
      if list_eq_dec Z.eq_dec consts consts0
      then constructor_shapes_eq_dec non_consts non_consts0
      else false
    | _ -> false)

(** val constructor_shapes_eq_dec : constructor_shapes -> constructor_shapes ->
    bool **)

and constructor_shapes_eq_dec shapes shapes' =
  match shapes with
  | Shapes_nil -> ( match shapes' with Shapes_nil -> true | _ -> false)
  | Shape_undetermined (tag, rest) -> (
    match shapes' with
    | Shape_undetermined (tag0, rest0) ->
      if Z.eq_dec tag tag0 then constructor_shapes_eq_dec rest rest0 else false
    | _ -> false)
  | Shape_determined (tag, fields, rest) -> (
    match shapes' with
    | Shape_determined (tag0, fields0, rest0) ->
      if Z.eq_dec tag tag0
      then
        if field_subkinds_eq_dec fields fields0
        then constructor_shapes_eq_dec rest rest0
        else false
      else false
    | _ -> false)

(** val field_subkinds_eq_dec : field_subkinds -> field_subkinds -> bool **)

and field_subkinds_eq_dec sks sks' =
  match sks with
  | Fields_nil -> (
    match sks' with Fields_nil -> true | Fields_cons (_, _, _) -> false)
  | Fields_cons (nullable, sk, rest) -> (
    match sks' with
    | Fields_nil -> false
    | Fields_cons (nullable0, sk0, rest0) ->
      if nullable
      then
        if nullable0
        then
          if subkind_eq_dec sk sk0
          then field_subkinds_eq_dec rest rest0
          else false
        else false
      else if nullable0
      then false
      else if subkind_eq_dec sk sk0
      then field_subkinds_eq_dec rest rest0
      else false)

type param_subkind = bool * subkind

(** val param_subkind_eq_dec : param_subkind -> param_subkind -> bool **)

let param_subkind_eq_dec ps ps' =
  let a, b = ps in
  let b0, s = ps' in
  if a
  then if b0 then subkind_eq_dec b s else false
  else if b0
  then false
  else subkind_eq_dec b s

(** val param_subkinds_eqb : param_subkind list -> param_subkind list -> bool **)

let param_subkinds_eqb sks sks' =
  if list_eq_dec param_subkind_eq_dec sks sks' then true else false

(** val any_value : param_subkind **)

let any_value = true, Subkind_anything

(** val says_nothing : bool -> subkind -> bool **)

let says_nothing nullable sk =
  nullable && match sk with Subkind_anything -> true | _ -> false

(** val subkinds_say_nothing : param_subkind list -> bool **)

let subkinds_say_nothing sks =
  forallb (fun ps -> says_nothing (fst ps) (snd ps)) sks
