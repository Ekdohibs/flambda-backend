open BinPos
open Datatypes
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open List

(** val shareable : prim -> bool **)

let shareable = function
  | Unary p0 -> (
    match p0 with
    | Block_load (_, mut, _) -> ( match mut with Mutable -> false | _ -> true)
    | Is_null -> false
    | Opaque_identity _ -> false
    | Reinterpret_64_bit_word _ -> false
    | _ -> true)
  | Binary p0 -> (
    match p0 with
    | Array_load (_, _, mut) -> ( match mut with Mutable -> false | _ -> true)
    | String_or_bigstring_load _ -> false
    | _ -> true)
  | _ -> false

type computed = (variable * (prim * simple list)) list

(** val forget_computed : variable -> computed -> computed **)

let forget_computed x known =
  filter
    (fun entry ->
      negb (Pos.eqb x (fst entry)) && negb (var_in_simples x (snd (snd entry))))
    known

(** val forget_computed_vars : variable list -> computed -> computed **)

let rec forget_computed_vars xs known =
  match xs with
  | [] -> known
  | x :: xs0 -> forget_computed_vars xs0 (forget_computed x known)

(** val tagged_again : variable -> prim -> simple list -> computed **)

let tagged_again x p args =
  match p with
  | Unary p0 -> (
    match p0 with
    | Untag_immediate -> (
      match args with
      | [] -> []
      | s :: l -> (
        match s with
        | Var t -> (
          match l with
          | [] -> (t, (Unary Tag_immediate, Var x :: [])) :: []
          | _ :: _ -> [])
        | _ -> []))
    | _ -> [])
  | _ -> []

(** val computed_under_let : computed -> variable -> named -> computed **)

let computed_under_let known x d =
  let known0 = forget_computed x known in
  match d with
  | Prim (p, args) ->
    if shareable p && negb (var_in_simples x args)
    then (x, (p, args)) :: app (tagged_again x p args) known0
    else known0
  | _ -> known0

(** val same_simples : simple list -> simple list -> bool **)

let same_simples args args' =
  if list_eq_dec simple_eq_dec args args' then true else false

(** val computed_by : computed -> prim -> simple list -> variable option **)

let rec computed_by known p args =
  match known with
  | [] -> None
  | p0 :: known0 ->
    let x, p1 = p0 in
    let p', args' = p1 in
    if prim_eqb p p' && same_simples args args'
    then Some x
    else computed_by known0 p args

(** val share_named : computed -> named -> named **)

let share_named known d =
  match d with
  | Prim (p, args) -> (
    match computed_by known p args with Some x -> Simple (Var x) | None -> d)
  | _ -> d

(** val share_primitives_knowing : computed -> expr -> expr **)

let rec share_primitives_knowing known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        share_named known d,
        share_primitives_knowing (computed_under_let known x d) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, share_primitives_knowing known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        share_primitives_in_handler known h,
        share_primitives_knowing known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, share_primitives_knowing known body)
  | _ -> e

(** val share_primitives_in_handler : computed -> continuation_handler ->
    continuation_handler **)

and share_primitives_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        share_primitives_knowing (forget_computed_vars params known) handler )

(** val share_known_primitives : expr -> expr **)

let share_known_primitives e = share_primitives_knowing [] e
