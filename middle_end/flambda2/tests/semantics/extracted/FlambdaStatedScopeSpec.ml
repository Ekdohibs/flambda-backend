open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaSubkindBinderSpec
open FlambdaSubkindOrderSpec
open FlambdaSubkindSpec
open FlambdaValueSpec
open List

type stated_in_scope = (variable * param_subkind) list

(** val stated_of_var : stated_in_scope -> variable -> param_subkind **)

let stated_of_var s x =
  match lookup x s with Some ps -> ps | None -> any_value

(** val not_null : param_subkind **)

let not_null = false, Subkind_anything

(** val tagged : param_subkind **)

let tagged = false, Subkind_tagged_immediate

(** val stated_of_simple : stated_in_scope -> simple -> param_subkind **)

let stated_of_simple s = function
  | Var x -> stated_of_var s x
  | Symbol _ -> not_null
  | Const c -> (
    match c with
    | Tagged_immediate i ->
      if is_tagged_immediate (Value_const (Tagged_immediate i))
      then false, Subkind_variant (i :: [], Shapes_nil)
      else any_value
    | _ -> any_value)

(** val fields_stated : stated_in_scope -> simple list -> field_subkinds **)

let rec fields_stated s = function
  | [] -> Fields_nil
  | s0 :: args0 ->
    let ps = stated_of_simple s s0 in
    Fields_cons (fst ps, snd ps, fields_stated s args0)

(** val stated_of_field : param_subkind -> coq_Z -> param_subkind **)

let stated_of_field ps field =
  match snd ps with
  | Subkind_variant (_, non_consts) -> (
    match non_consts with
    | Shape_determined (_, fields, rest) -> (
      match rest with
      | Shapes_nil ->
        if Z.leb Z0 field
        then
          match field_subkind_at fields (Z.to_nat field) with
          | Some ps' -> ps'
          | None -> any_value
        else any_value
      | _ -> any_value)
    | _ -> any_value)
  | _ -> any_value

(** val stated_of_named : stated_in_scope -> named -> param_subkind **)

let stated_of_named s = function
  | Simple s0 -> stated_of_simple s s0
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Block_load (kind, _, field) -> (
        let (Block_access_values (_, _, field_kind)) = kind in
        let ps =
          match args with
          | [] -> any_value
          | s0 :: l -> (
            match s0 with
            | Var b -> (
              match l with
              | [] -> stated_of_field (stated_of_var s b) field
              | _ :: _ -> any_value)
            | _ -> any_value)
        in
        match field_kind with
        | Any_value -> ps
        | Immediate -> if param_says_at_least ps tagged then ps else tagged)
      | Array_length _ -> tagged
      | Opaque_identity _ -> (
        match args with
        | [] -> any_value
        | s0 :: l -> (
          match l with [] -> stated_of_simple s s0 | _ :: _ -> any_value))
      | Unary_int_arith (kind, _) -> (
        match kind with Standard_tagged_immediate -> tagged | _ -> any_value)
      | Num_conv (_, dst) -> (
        match dst with Standard_tagged_immediate -> tagged | _ -> any_value)
      | Boolean_not -> tagged
      | Box_number n -> false, Subkind_boxed_number n
      | Tag_immediate -> tagged
      | _ -> any_value)
    | Binary p0 -> (
      match p0 with
      | Array_load (_, load_kind, _) -> (
        match load_kind with Load_immediates -> tagged | _ -> any_value)
      | Int_arith (kind, _) -> (
        match kind with Standard_tagged_immediate -> tagged | _ -> any_value)
      | Int_shift (kind, _) -> (
        match kind with Standard_tagged_immediate -> tagged | _ -> any_value)
      | _ -> any_value)
    | Variadic p0 -> (
      match p0 with
      | Make_block kind ->
        let (Block_kind_values tag) = kind in
        ( false,
          Subkind_variant
            ([], Shape_determined (tag, fields_stated s args, Shapes_nil)) )
      | Make_immutable_array _ -> not_null)
    | Heap p0 -> (
      match p0 with
      | Make_block_with_identity (_, _) -> not_null
      | Make_mutable_array _ -> not_null
      | _ -> any_value)
    | Region_primitive _ -> any_value)
  | Project_value_slot (_, _) -> any_value
  | Project_value_slot_from (_, _, _) -> any_value
  | _ -> not_null

(** val params_stated : variable list -> param_subkind list -> stated_in_scope
    -> stated_in_scope **)

let rec params_stated params sks s =
  match params with
  | [] -> s
  | param :: params0 -> (
    match sks with
    | [] -> params_stated params0 [] ((param, any_value) :: s)
    | ps :: sks0 -> params_stated params0 sks0 ((param, ps) :: s))

(** val is_stated : stated_in_scope -> param_subkind -> simple -> bool **)

let is_stated s ps s0 =
  match s0 with
  | Const c -> value_at empty_static_env ps (Value_const c)
  | _ -> param_says_at_least (stated_of_simple s s0) ps

(** val are_stated : stated_in_scope -> simple list -> param_subkind list ->
    bool **)

let rec are_stated s args = function
  | [] -> true
  | ps :: sks0 -> (
    match args with
    | [] -> true
    | s0 :: args0 -> is_stated s ps s0 && are_stated s args0 sks0)

(** val code_stated : variable list -> variable -> code_metadata ->
    stated_in_scope **)

let code_stated params my_closure metadata =
  params_stated params metadata.cm_params_subkinds ((my_closure, not_null) :: [])

(** val is_stated_at : stated_in_scope -> param_subkind list -> nat -> simple ->
    bool **)

let is_stated_at s sks i s0 =
  match nth_error sks i with Some ps -> is_stated s ps s0 | None -> true
