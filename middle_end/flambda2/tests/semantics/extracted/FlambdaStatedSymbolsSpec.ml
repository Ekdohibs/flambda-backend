open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaSubkindOrderSpec

type symbols_stated = (symbol * param_subkind) list

(** val stated_of_symbol : symbols_stated -> symbol -> param_subkind **)

let stated_of_symbol y sym =
  match lookup sym y with Some ps -> ps | None -> not_null

(** val static_field_stated : symbols_stated -> simple -> param_subkind option **)

let static_field_stated y s =
  match s with
  | Var _ -> None
  | Symbol sym -> Some (stated_of_symbol y sym)
  | Const c -> (
    match c with
    | Tagged_immediate _ -> Some (stated_of_simple [] s)
    | _ -> None)

(** val static_fields_stated : symbols_stated -> simple list -> field_subkinds
    option **)

let rec static_fields_stated y = function
  | [] -> Some Fields_nil
  | s :: fields0 -> (
    match static_field_stated y s with
    | Some ps -> (
      match static_fields_stated y fields0 with
      | Some rest -> Some (Fields_cons (fst ps, snd ps, rest))
      | None -> None)
    | None -> None)

(** val stated_of_static_const : symbols_stated -> static_const -> param_subkind
    **)

let stated_of_static_const y = function
  | Static_const_block (tag, fields) -> (
    match static_fields_stated y fields with
    | Some fields0 ->
      false, Subkind_variant ([], Shape_determined (tag, fields0, Shapes_nil))
    | None -> not_null)
  | _ -> not_null

(** val statics_stated : static_binding list -> symbols_stated -> symbols_stated
    **)

let rec statics_stated bindings y =
  match bindings with
  | [] -> y
  | s :: bindings0 -> (
    match s with
    | Bind_symbol (sym, const) ->
      statics_stated bindings0 ((sym, stated_of_static_const y const) :: y)
    | _ -> statics_stated bindings0 y)

(** val is_stated_knowing : symbols_stated -> stated_in_scope -> param_subkind
    -> simple -> bool **)

let is_stated_knowing y s ps s0 =
  match s0 with
  | Symbol sym -> param_says_at_least (stated_of_symbol y sym) ps
  | _ -> is_stated s ps s0

(** val are_stated_knowing : symbols_stated -> stated_in_scope -> simple list ->
    param_subkind list -> bool **)

let rec are_stated_knowing y s args = function
  | [] -> true
  | ps :: sks0 -> (
    match args with
    | [] -> true
    | s0 :: args0 ->
      is_stated_knowing y s ps s0 && are_stated_knowing y s args0 sks0)
