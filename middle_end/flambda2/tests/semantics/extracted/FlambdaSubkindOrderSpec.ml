open BinInt
open BinNums
open Datatypes
open FlambdaKindModel
open List

(** val shape_of_tag : coq_Z -> constructor_shapes -> field_subkinds option
    option **)

let rec shape_of_tag tag = function
  | Shapes_nil -> None
  | Shape_undetermined (tag', rest) ->
    if Z.eqb tag tag' then Some None else shape_of_tag tag rest
  | Shape_determined (tag', sks, rest) ->
    if Z.eqb tag tag' then Some (Some sks) else shape_of_tag tag rest

(** val subkind_says_at_least : subkind -> subkind -> bool **)

let rec subkind_says_at_least a = function
  | Subkind_anything -> true
  | Subkind_tagged_immediate -> (
    match a with
    | Subkind_tagged_immediate -> true
    | Subkind_variant (_, non_consts) -> (
      match non_consts with Shapes_nil -> true | _ -> false)
    | _ -> false)
  | Subkind_boxed_number n' -> (
    match a with
    | Subkind_boxed_number n -> boxable_number_eqb n n'
    | _ -> false)
  | Subkind_variant (consts', shapes') -> (
    match a with
    | Subkind_variant (consts, shapes) ->
      forallb (fun i -> existsb (Z.eqb i) consts') consts
      && shapes_say_at_least [] shapes shapes'
    | _ -> false)

(** val shapes_say_at_least : coq_Z list -> constructor_shapes ->
    constructor_shapes -> bool **)

and shapes_say_at_least seen shapes shapes' =
  match shapes with
  | Shapes_nil -> true
  | Shape_undetermined (tag, rest) ->
    (existsb (Z.eqb tag) seen
    ||
    match shape_of_tag tag shapes' with
    | Some o -> ( match o with Some _ -> false | None -> true)
    | None -> false)
    && shapes_say_at_least (tag :: seen) rest shapes'
  | Shape_determined (tag, sks, rest) ->
    (existsb (Z.eqb tag) seen
    ||
    match shape_of_tag tag shapes' with
    | Some o -> (
      match o with Some sks' -> fields_say_at_least sks sks' | None -> true)
    | None -> false)
    && shapes_say_at_least (tag :: seen) rest shapes'

(** val fields_say_at_least : field_subkinds -> field_subkinds -> bool **)

and fields_say_at_least sks sks' =
  match sks with
  | Fields_nil -> (
    match sks' with Fields_nil -> true | Fields_cons (_, _, _) -> false)
  | Fields_cons (nullable, sk, rest) -> (
    match sks' with
    | Fields_nil -> false
    | Fields_cons (nullable', sk', rest') ->
      ((negb nullable || nullable') && subkind_says_at_least sk sk')
      && fields_say_at_least rest rest')

(** val param_says_at_least : param_subkind -> param_subkind -> bool **)

let param_says_at_least a b =
  (negb (fst a) || fst b) && subkind_says_at_least (snd a) (snd b)

(** val params_say_at_least : param_subkind list -> param_subkind list -> bool **)

let rec params_say_at_least a = function
  | [] -> true
  | pb :: b' -> (
    match a with
    | [] -> says_nothing (fst pb) (snd pb) && params_say_at_least [] b'
    | pa :: a' -> param_says_at_least pa pb && params_say_at_least a' b')
