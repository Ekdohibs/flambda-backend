open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaUnaliasParamsSpec
open List

(** val same_simple : simple -> simple -> bool **)

let same_simple s s' = if simple_eq_dec s s' then true else false

(** val is_plain : simple -> bool **)

let is_plain = function Var _ -> false | _ -> true

(** val apply_cont_gives : continuation -> nat -> simple -> apply_cont -> bool **)

let apply_cont_gives k i s ac =
  (match ac.ac_trap_action with
    | Some action -> negb (Pos.eqb (trap_action_handler action) k)
    | None -> true)
  &&
  if Pos.eqb ac.ac_continuation k
  then
    match nth_error ac.ac_args i with
    | Some s' -> same_simple s' s
    | None -> false
  else true

(** val all_give : continuation -> nat -> simple -> expr -> bool **)

let rec all_give k i s = function
  | Let (_, _, _, body) -> all_give k i s body
  | Let_static (_, body) -> all_give k i s body
  | Let_cont_non_recursive (k', h, body) ->
    (negb (Pos.eqb k' k) && handler_all_give k i s h) && all_give k i s body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k', h = pat in
        negb (Pos.eqb k' k) && handler_all_give k i s h)
      handlers
    && all_give k i s body
  | Apply ap ->
    negb (Pos.eqb ap.ap_continuation k)
    && negb (Pos.eqb ap.ap_exn_continuation k)
  | Apply_cont ac -> apply_cont_gives k i s ac
  | Switch (_, arms) ->
    forallb
      (fun pat ->
        let _, ac = pat in
        apply_cont_gives k i s ac)
      arms
  | Invalid -> true

(** val handler_all_give : continuation -> nat -> simple -> continuation_handler
    -> bool **)

and handler_all_give k i s = function
  | Handler (_, _, handler) -> all_give k i s handler

(** val given_by : continuation -> nat -> apply_cont -> simple option option **)

let given_by k i ac =
  if Pos.eqb ac.ac_continuation k then Some (nth_error ac.ac_args i) else None

(** val first_given : continuation -> nat -> expr -> simple option option **)

let rec first_given k i = function
  | Let (_, _, _, body) -> first_given k i body
  | Let_static (_, body) -> first_given k i body
  | Let_cont_non_recursive (_, h, body) -> (
    match handler_first_given k i h with
    | Some r -> Some r
    | None -> first_given k i body)
  | Let_cont_recursive (handlers, body) -> (
    match
      first_answer
        (fun pat ->
          let _, h = pat in
          handler_first_given k i h)
        handlers
    with
    | Some r -> Some r
    | None -> first_given k i body)
  | Apply_cont ac -> given_by k i ac
  | Switch (_, arms) ->
    first_answer
      (fun pat ->
        let _, ac = pat in
        given_by k i ac)
      arms
  | _ -> None

(** val handler_first_given : continuation -> nat -> continuation_handler ->
    simple option option **)

and handler_first_given k i = function
  | Handler (_, _, handler) -> first_given k i handler

(** val bind_const_param : nat -> continuation -> continuation_handler -> expr
    -> (continuation_handler * expr) option **)

let bind_const_param i k h body =
  let (Handler (params, metadata, handler)) = h in
  match nth_error params i with
  | Some p -> (
    match nth_error metadata.hm_params_kinds i with
    | Some kind -> (
      match first_given k i body with
      | Some o -> (
        match o with
        | Some s ->
          if
            ((is_plain s && negb metadata.hm_is_exn_handler)
            && all_give k i s body)
            && negb (among p (drop_nth i params))
          then
            Some
              ( Handler
                  ( drop_nth i params,
                    { hm_params_kinds = drop_nth i metadata.hm_params_kinds;
                      hm_is_exn_handler = false
                    },
                    Let (p, kind, Simple s, handler) ),
                drop_arg k i body )
          else None
        | None -> None)
      | None -> None)
    | None -> None)
  | None -> None

(** val bind_const_from : nat -> continuation -> continuation_handler -> expr ->
    continuation_handler * expr **)

let rec bind_const_from i k h body =
  match i with
  | O -> h, body
  | S i0 -> (
    match bind_const_param i0 k h body with
    | Some p ->
      let h', body' = p in
      bind_const_from i0 k h' body'
    | None -> bind_const_from i0 k h body)

(** val bind_const_params : expr -> expr **)

let rec bind_const_params e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, bind_const_params body)
  | Let_static (bindings, body) -> Let_static (bindings, bind_const_params body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let h = Handler (params, metadata, bind_const_params handler) in
    let body0 = bind_const_params body in
    let h0, body1 = bind_const_from (length params) k h body0 in
    Let_cont_non_recursive (k, h0, body1)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, bind_const_params body)
  | _ -> e
