open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaUnaliasParamsSpec
open List
open ListDef
open Nat

(** val trap_free_of : continuation -> apply_cont -> bool **)

let trap_free_of k ac =
  match ac.ac_trap_action with
  | Some action -> negb (Pos.eqb (trap_action_handler action) k)
  | None -> true

(** val only_applied_or_returned_to : continuation -> expr -> bool **)

let rec only_applied_or_returned_to k = function
  | Let (_, _, _, body) -> only_applied_or_returned_to k body
  | Let_static (_, body) -> only_applied_or_returned_to k body
  | Let_cont_non_recursive (k', h, body) ->
    (negb (Pos.eqb k' k) && handler_only_applied_or_returned_to k h)
    && only_applied_or_returned_to k body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k', h = pat in
        negb (Pos.eqb k' k) && handler_only_applied_or_returned_to k h)
      handlers
    && only_applied_or_returned_to k body
  | Apply ap -> negb (Pos.eqb ap.ap_exn_continuation k)
  | Apply_cont ac -> trap_free_of k ac
  | Switch (_, arms) ->
    forallb
      (fun pat ->
        let _, ac = pat in
        trap_free_of k ac)
      arms
  | Invalid -> true

(** val handler_only_applied_or_returned_to : continuation ->
    continuation_handler -> bool **)

and handler_only_applied_or_returned_to k = function
  | Handler (_, _, handler) -> only_applied_or_returned_to k handler

(** val one_if_same : continuation -> continuation -> nat **)

let one_if_same k k' = if Pos.eqb k' k then S O else O

(** val uses_of : continuation -> expr -> nat **)

let rec uses_of k = function
  | Let (_, _, _, body) -> uses_of k body
  | Let_static (_, body) -> uses_of k body
  | Let_cont_non_recursive (_, h, body) ->
    add (uses_in_handler k h) (uses_of k body)
  | Let_cont_recursive (handlers, body) ->
    add
      (list_sum
         (map
            (fun pat ->
              let _, h = pat in
              uses_in_handler k h)
            handlers))
      (uses_of k body)
  | Apply ap -> one_if_same k ap.ap_continuation
  | Apply_cont ac -> one_if_same k ac.ac_continuation
  | Switch (_, arms) ->
    list_sum
      (map
         (fun pat ->
           let _, ac = pat in
           one_if_same k ac.ac_continuation)
         arms)
  | Invalid -> O

(** val uses_in_handler : continuation -> continuation_handler -> nat **)

and uses_in_handler k = function Handler (_, _, handler) -> uses_of k handler

(** val wrapper : continuation -> nat -> variable list -> handler_metadata ->
    continuation_handler **)

let wrapper k i params metadata =
  Handler
    ( params,
      metadata,
      Apply_cont
        { ac_continuation = k;
          ac_args = map (fun x -> Var x) (drop_nth i params);
          ac_trap_action = None
        } )

(** val wrap_uses : continuation -> nat -> continuation_handler -> expr -> expr
    **)

let rec wrap_uses k i w e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, wrap_uses k i w body)
  | Let_static (bindings, body) -> Let_static (bindings, wrap_uses k i w body)
  | Let_cont_non_recursive (k', h, body) ->
    Let_cont_non_recursive
      (k', wrap_uses_in_handler k i w h, wrap_uses k i w body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k', h = pat in
            k', wrap_uses_in_handler k i w h)
          handlers,
        wrap_uses k i w body )
  | Apply ap ->
    if Pos.eqb ap.ap_continuation k
    then Let_cont_non_recursive (k, w, Apply ap)
    else e
  | Apply_cont ac -> Apply_cont (drop_arg_of k i ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let n, ac = pat in
            n, drop_arg_of k i ac)
          arms )
  | Invalid -> e

(** val wrap_uses_in_handler : continuation -> nat -> continuation_handler ->
    continuation_handler -> continuation_handler **)

and wrap_uses_in_handler k i w = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, wrap_uses k i w handler)

(** val wrap_return : nat -> continuation -> continuation_handler -> expr ->
    (continuation_handler * expr) option **)

let wrap_return i k h body =
  let (Handler (params, metadata, handler)) = h in
  match nth_error params i with
  | Some p ->
    if
      (((negb metadata.hm_is_exn_handler && distinct params)
       && negb (var_free_b p handler))
      && only_applied_or_returned_to k body)
      && PeanoNat.Nat.leb (S (S O)) (uses_of k body)
    then
      Some
        ( Handler
            ( drop_nth i params,
              { hm_params_kinds = drop_nth i metadata.hm_params_kinds;
                hm_params_subkinds = drop_nth i metadata.hm_params_subkinds;
                hm_is_exn_handler = false
              },
              handler ),
          wrap_uses k i (wrapper k i params metadata) body )
    else None
  | None -> None

(** val wrap_from : nat -> continuation -> continuation_handler -> expr ->
    continuation_handler * expr **)

let rec wrap_from i k h body =
  match i with
  | O -> h, body
  | S i0 -> (
    match wrap_return i0 k h body with
    | Some p -> p
    | None -> wrap_from i0 k h body)

(** val wrap_returns : expr -> expr **)

let rec wrap_returns e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, wrap_returns body)
  | Let_static (bindings, body) -> Let_static (bindings, wrap_returns body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let h = Handler (params, metadata, wrap_returns handler) in
    let body0 = wrap_returns body in
    let h0, body1 = wrap_from (length params) k h body0 in
    Let_cont_non_recursive (k, h0, body1)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, wrap_returns body)
  | _ -> e
