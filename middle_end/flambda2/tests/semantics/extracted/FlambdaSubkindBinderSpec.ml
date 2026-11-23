open Datatypes
open FlambdaKindModel

(** val field_subkind_at : field_subkinds -> nat -> param_subkind option **)

let rec field_subkind_at sks field =
  match sks with
  | Fields_nil -> None
  | Fields_cons (nullable, sk, rest) -> (
    match field with
    | O -> Some (nullable, sk)
    | S field0 -> field_subkind_at rest field0)
