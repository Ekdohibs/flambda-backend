open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaPrimitiveSpec
open FlambdaValueSpec
open List

(** val value_of_subkind : static_env -> subkind -> value -> bool **)

let rec value_of_subkind statics sk v =
  match sk with
  | Subkind_anything -> true
  | Subkind_tagged_immediate -> is_tagged_immediate v
  | Subkind_boxed_number n -> (
    match boxed_number_of statics v with
    | Some c -> (
      match boxable_number_of_const c with
      | Some n' -> boxable_number_eqb n n'
      | None -> false)
    | None -> false)
  | Subkind_variant (consts, non_consts) -> (
    match v with
    | Value_const c -> (
      match c with
      | Tagged_immediate i -> is_tagged_immediate v && existsb (Z.eqb i) consts
      | _ -> (
        match block_of statics v with
        | Some p ->
          let tag, fields = p in
          block_of_shapes statics non_consts tag fields
        | None -> false))
    | _ -> (
      match block_of statics v with
      | Some p ->
        let tag, fields = p in
        block_of_shapes statics non_consts tag fields
      | None -> false))

(** val block_of_shapes : static_env -> constructor_shapes -> coq_Z -> value
    list -> bool **)

and block_of_shapes statics shapes tag fields =
  match shapes with
  | Shapes_nil -> false
  | Shape_undetermined (tag', rest) ->
    if Z.eqb tag tag' then true else block_of_shapes statics rest tag fields
  | Shape_determined (tag', sks, rest) ->
    if Z.eqb tag tag'
    then fields_of_subkinds statics sks fields
    else block_of_shapes statics rest tag fields

(** val fields_of_subkinds : static_env -> field_subkinds -> value list -> bool
    **)

and fields_of_subkinds statics sks fields =
  match sks with
  | Fields_nil -> ( match fields with [] -> true | _ :: _ -> false)
  | Fields_cons (nullable, sk, sks0) -> (
    match fields with
    | [] -> false
    | v :: fields0 ->
      (kind_eqb (kind_of_value v) Value
      &&
      match v with
      | Value_const c -> (
        match c with
        | Null -> nullable
        | Poison _ -> says_nothing nullable sk
        | _ -> value_of_subkind statics sk v)
      | _ -> value_of_subkind statics sk v)
      && fields_of_subkinds statics sks0 fields0)

(** val value_at : static_env -> param_subkind -> value -> bool **)

let value_at statics ps v =
  match v with
  | Value_const c -> (
    match c with
    | Null -> fst ps
    | Poison _ -> says_nothing (fst ps) (snd ps)
    | _ -> value_of_subkind statics (snd ps) v)
  | _ -> value_of_subkind statics (snd ps) v

(** val values_of_subkinds : static_env -> value list -> param_subkind list ->
    bool **)

let rec values_of_subkinds statics vs = function
  | [] -> true
  | ps :: sks0 -> (
    match vs with
    | [] -> true
    | v :: vs0 -> value_at statics ps v && values_of_subkinds statics vs0 sks0)
