open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaSubkindOrderSpec
open FlambdaUntagParamsSpec
open List
open ListDef

(** val states_tagged : param_subkind -> bool **)

let states_tagged ps = param_says_at_least ps (false, Subkind_tagged_immediate)

(** val tagged_params : variable list -> param_subkind list -> variable list **)

let rec tagged_params params sks =
  match params with
  | [] -> []
  | p :: params0 -> (
    match sks with
    | [] -> []
    | ps :: sks0 ->
      if states_tagged ps
      then p :: tagged_params params0 sks0
      else tagged_params params0 sks0)

(** val untags : variable -> flambda_kind -> named -> bool **)

let untags x kind d =
  kind_eqb kind untagged_kind
  && match untag_of d with Some p -> Pos.eqb p x | None -> false

(** val untaggings : variable -> expr -> variable list **)

let rec untaggings x = function
  | Let (y, kind, d, body) ->
    app
      (if untags x kind d then y :: [] else [])
      (if Pos.eqb y x then [] else untaggings x body)
  | Let_static (_, body) -> untaggings x body
  | Let_cont_non_recursive (_, h, body) ->
    app (untaggings_in_handler x h) (untaggings x body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           untaggings_in_handler x h)
         handlers)
      (untaggings x body)
  | _ -> []

(** val untaggings_in_handler : variable -> continuation_handler -> variable
    list **)

and untaggings_in_handler x = function
  | Handler (params, _, handler) ->
    if among x params then [] else untaggings x handler

(** val first_untagging : variable -> expr -> variable option **)

let first_untagging x e =
  match untaggings x e with [] -> None | u :: _ -> Some u

(** val drop_untagging : variable -> variable -> expr -> expr **)

let rec drop_untagging x u e =
  match e with
  | Let (y, kind, d, body) ->
    if Pos.eqb y u
    then if untags x kind d then drop_untagging x u body else e
    else if Pos.eqb y x
    then e
    else Let (y, kind, d, drop_untagging x u body)
  | Let_static (bindings, body) -> Let_static (bindings, drop_untagging x u body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, drop_untagging_in_handler x u h, drop_untagging x u body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, drop_untagging_in_handler x u h)
          handlers,
        drop_untagging x u body )
  | _ -> e

(** val drop_untagging_in_handler : variable -> variable -> continuation_handler
    -> continuation_handler **)

and drop_untagging_in_handler x u h =
  match h with
  | Handler (params, metadata, handler) ->
    if among x params || among u params
    then h
    else Handler (params, metadata, drop_untagging x u handler)

(** val untagging_at_head : variable list -> variable -> expr -> expr **)

let untagging_at_head params x handler =
  match first_untagging x handler with
  | Some u ->
    if negb (among u params) && negb (var_free_b u handler)
    then
      Let
        ( u,
          untagged_kind,
          Prim (Unary Untag_immediate, Var x :: []),
          drop_untagging x u handler )
    else handler
  | None -> handler

(** val returns_to : continuation -> expr -> bool **)

let rec returns_to k = function
  | Let (_, _, _, body) -> returns_to k body
  | Let_static (_, body) -> returns_to k body
  | Let_cont_non_recursive (k', h, body) ->
    returns_to_in_handler k h || (negb (Pos.eqb k' k) && returns_to k body)
  | Let_cont_recursive (handlers, body) ->
    negb (among k (map fst handlers))
    && (existsb
          (fun pat ->
            let _, h = pat in
            returns_to_in_handler k h)
          handlers
       || returns_to k body)
  | Apply ap -> Pos.eqb ap.ap_continuation k
  | _ -> false

(** val returns_to_in_handler : continuation -> continuation_handler -> bool **)

and returns_to_in_handler k = function
  | Handler (_, _, handler) -> returns_to k handler

(** val bind_untaggings : expr -> expr **)

let rec bind_untaggings e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, bind_untaggings body)
  | Let_static (bindings, body) -> Let_static (bindings, bind_untaggings body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, bind_untaggings_in_handler (returns_to k body) h, bind_untaggings body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, bind_untaggings body)
  | _ -> e

(** val bind_untaggings_in_handler : bool -> continuation_handler ->
    continuation_handler **)

and bind_untaggings_in_handler returned_to = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        if (returned_to && negb metadata.hm_is_exn_handler) && distinct params
        then
          fold_right (untagging_at_head params) (bind_untaggings handler)
            (tagged_params params metadata.hm_params_subkinds)
        else bind_untaggings handler )
