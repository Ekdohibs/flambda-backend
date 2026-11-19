open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open List
open ListDef

(** val keep : bool list -> 'a1 list -> 'a1 list **)

let rec keep mask l =
  match mask with
  | [] -> []
  | b :: mask0 -> (
    if b
    then match l with [] -> [] | a :: l0 -> a :: keep mask0 l0
    else match l with [] -> [] | _ :: l0 -> keep mask0 l0)

(** val keep_all_or : bool list option -> 'a1 list -> 'a1 list **)

let keep_all_or mask l =
  match mask with Some mask0 -> keep mask0 l | None -> l

type masks = (continuation * bool list option) list

(** val mask_of : masks -> continuation -> bool list option **)

let mask_of known k =
  match lookup k known with Some mask -> mask | None -> None

(** val drop_args : masks -> apply_cont -> apply_cont **)

let drop_args known ac =
  { ac_continuation = ac.ac_continuation;
    ac_args = keep_all_or (mask_of known ac.ac_continuation) ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val returned_to : continuation -> expr -> bool **)

let rec returned_to k = function
  | Let (_, _, _, body) -> returned_to k body
  | Let_static (_, body) -> returned_to k body
  | Let_cont_non_recursive (_, h, body) ->
    returned_to_in_handler k h || returned_to k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        returned_to_in_handler k h)
      handlers
    || returned_to k body
  | Apply ap -> Pos.eqb k ap.ap_continuation || Pos.eqb k ap.ap_exn_continuation
  | _ -> false

(** val returned_to_in_handler : continuation -> continuation_handler -> bool **)

and returned_to_in_handler k = function
  | Handler (_, _, handler) -> returned_to k handler

(** val used_params : bool -> variable list -> expr -> bool list option **)

let used_params is_exn_handler params handler =
  let mask = map (fun param -> var_free_b param handler) params in
  let mask0 =
    if is_exn_handler
    then match mask with [] -> mask | _ :: mask0 -> true :: mask0
    else mask
  in
  if forallb (fun used -> used) mask0 then None else Some mask0

(** val keep_kinds : bool list option -> handler_metadata -> handler_metadata **)

let keep_kinds mask metadata =
  match mask with
  | Some mask0 ->
    { hm_params_kinds = keep mask0 metadata.hm_params_kinds;
      hm_params_subkinds = keep mask0 metadata.hm_params_subkinds;
      hm_is_exn_handler = metadata.hm_is_exn_handler
    }
  | None -> metadata

(** val used_params_in_loop : continuation -> continuation_handler -> expr ->
    bool list option **)

let used_params_in_loop k h whole =
  let (Handler (params, metadata, handler)) = h in
  if returned_to k whole
  then None
  else used_params metadata.hm_is_exn_handler params handler

(** val drop_params : masks -> expr -> expr **)

let rec drop_params known e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, drop_params known body)
  | Let_static (bindings, body) -> Let_static (bindings, drop_params known body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let handler1 = drop_params known handler in
    let mask =
      if returned_to k body
      then None
      else used_params metadata.hm_is_exn_handler params handler1
    in
    Let_cont_non_recursive
      ( k,
        Handler (keep_all_or mask params, keep_kinds mask metadata, handler1),
        drop_params ((k, mask) :: known) body )
  | Let_cont_recursive (handlers, body) ->
    let known0 =
      app
        (map
           (fun pat ->
             let k, h = pat in
             k, used_params_in_loop k h e)
           handlers)
        known
    in
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, drop_params_in_loop known0 (used_params_in_loop k h e) h)
          handlers,
        drop_params known0 body )
  | Apply_cont ac -> Apply_cont (drop_args known ac)
  | Switch (scrutinee, arms) ->
    Switch (scrutinee, map (fun arm -> fst arm, drop_args known (snd arm)) arms)
  | _ -> e

(** val drop_params_in_loop : masks -> bool list option -> continuation_handler
    -> continuation_handler **)

and drop_params_in_loop known mask = function
  | Handler (params, metadata, handler) ->
    Handler
      ( keep_all_or mask params,
        keep_kinds mask metadata,
        drop_params known handler )

(** val drop_unused_params : expr -> expr **)

let drop_unused_params e = drop_params [] e
