open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open List
open ListDef

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

(** val array_made_under_let : variable -> named -> computed **)

let array_made_under_let x = function
  | Prim (p, args) -> (
    match p with
    | Heap p0 -> (
      match p0 with
      | Make_mutable_array kind ->
        if var_in_simples x args
        then []
        else (x, (Heap (Make_mutable_array kind), args)) :: []
      | _ -> [])
    | _ -> [])
  | _ -> []

(** val computed_or_made_under_let : computed -> variable -> named -> computed **)

let computed_or_made_under_let known x d =
  app (array_made_under_let x d) (computed_under_let known x d)

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

(** val made_length : computed -> variable -> coq_Z option **)

let rec made_length known x =
  match known with
  | [] -> None
  | p :: known0 -> (
    let y, p0 = p in
    let p1, args = p0 in
    match p1 with
    | Heap p2 -> (
      match p2 with
      | Make_mutable_array _ ->
        if Pos.eqb x y
        then Some (Z.of_nat (length args))
        else made_length known0 x
      | _ -> made_length known0 x)
    | _ -> made_length known0 x)

(** val made_length_named : computed -> named -> named option **)

let made_length_named known = function
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Array_length _ -> (
        match args with
        | [] -> None
        | s :: l -> (
          match s with
          | Var x -> (
            match l with
            | [] -> (
              match made_length known x with
              | Some n ->
                if in_range (int_width Standard_tagged_immediate) n
                then Some (Simple (Const (Tagged_immediate n)))
                else None
              | None -> None)
            | _ :: _ -> None)
          | _ -> None))
      | _ -> None)
    | _ -> None)
  | _ -> None

(** val share_named : computed -> named -> named **)

let share_named known d =
  match made_length_named known d with
  | Some d' -> d'
  | None -> (
    match d with
    | Prim (p, args) ->
      if shareable p
      then
        match computed_by known p args with
        | Some x -> Simple (Var x)
        | None -> d
      else d
    | _ -> d)

(** val share_primitives_knowing : computed -> expr -> expr **)

let rec share_primitives_knowing known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        share_named known d,
        share_primitives_knowing (computed_or_made_under_let known x d) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, share_primitives_knowing known body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        share_primitives_in_handler known h,
        share_primitives_knowing known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, share_primitives_in_handler known h)
          handlers,
        share_primitives_knowing known body )
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
