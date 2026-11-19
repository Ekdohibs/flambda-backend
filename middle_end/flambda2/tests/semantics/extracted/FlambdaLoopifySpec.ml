open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaUnusedContsSpec
open List
open ListDef
open PeanoNat

(** val is_self_tail_call : variable -> arity -> continuation -> continuation ->
    apply_expr -> bool **)

let is_self_tail_call my_closure params_arity ret exn ap =
  ((((match ap.ap_callee with Var x -> Pos.eqb x my_closure | _ -> false)
    && Nat.eqb (length ap.ap_args_arity) (length params_arity))
   && Pos.eqb ap.ap_continuation ret)
  && Pos.eqb ap.ap_exn_continuation exn)
  && no_exn_extra_args ap

(** val loop_calls : variable -> arity -> continuation -> continuation ->
    continuation -> expr -> expr **)

let rec loop_calls my_closure params_arity ret exn self e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, loop_calls my_closure params_arity ret exn self body)
  | Let_static (bindings, body) ->
    Let_static (bindings, loop_calls my_closure params_arity ret exn self body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        loop_calls_in_handler my_closure params_arity ret exn self h,
        loop_calls my_closure params_arity ret exn self body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, loop_calls_in_handler my_closure params_arity ret exn self h)
          handlers,
        loop_calls my_closure params_arity ret exn self body )
  | Apply ap ->
    if is_self_tail_call my_closure params_arity ret exn ap
    then
      Apply_cont
        { ac_continuation = self; ac_args = ap.ap_args; ac_trap_action = None }
    else e
  | _ -> e

(** val loop_calls_in_handler : variable -> arity -> continuation ->
    continuation -> continuation -> continuation_handler -> continuation_handler
    **)

and loop_calls_in_handler my_closure params_arity ret exn self = function
  | Handler (params, metadata, handler) ->
    Handler
      (params, metadata, loop_calls my_closure params_arity ret exn self handler)

(** val loop_safe : variable -> continuation -> continuation -> continuation ->
    expr -> bool **)

let rec loop_safe my_closure ret exn self = function
  | Let (x, _, _, body) ->
    negb (Pos.eqb x my_closure) && loop_safe my_closure ret exn self body
  | Let_static (_, body) -> loop_safe my_closure ret exn self body
  | Let_cont_non_recursive (k, h, body) ->
    (((negb (Pos.eqb k ret) && negb (Pos.eqb k exn)) && negb (Pos.eqb k self))
    && loop_safe_handler my_closure ret exn self h)
    && loop_safe my_closure ret exn self body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        ((negb (Pos.eqb k ret) && negb (Pos.eqb k exn)) && negb (Pos.eqb k self))
        && loop_safe_handler my_closure ret exn self h)
      handlers
    && loop_safe my_closure ret exn self body
  | Apply ap ->
    negb (Pos.eqb ap.ap_continuation self)
    && negb (Pos.eqb ap.ap_exn_continuation self)
  | Apply_cont ac -> negb (among self (apply_cont_continuations ac))
  | Switch (_, arms) ->
    forallb
      (fun arm -> negb (among self (apply_cont_continuations (snd arm))))
      arms
  | Invalid -> true

(** val loop_safe_handler : variable -> continuation -> continuation ->
    continuation -> continuation_handler -> bool **)

and loop_safe_handler my_closure ret exn self = function
  | Handler (params, _, handler) ->
    negb (among my_closure params) && loop_safe my_closure ret exn self handler

(** val loop_handler : continuation -> code -> continuation_handler **)

let loop_handler self c =
  Handler
    ( code_params c,
      { hm_params_kinds = code_params_kinds c;
        hm_params_subkinds = code_params_subkinds c;
        hm_is_exn_handler = false
      },
      loop_calls (code_my_closure c) (code_params_arity c)
        (code_return_continuation c)
        (code_exn_continuation c) self (code_body c) )

(** val looped_code : continuation -> code -> code **)

let looped_code self c =
  match c with
  | Code (params, my_closure, ret, exn, metadata, _) ->
    Code
      ( params,
        my_closure,
        ret,
        exn,
        metadata,
        Let_cont_recursive
          ( (self, loop_handler self c) :: [],
            Apply_cont
              { ac_continuation = self;
                ac_args = map (fun x -> Var x) params;
                ac_trap_action = None
              } ) )

(** val loopable : continuation -> code -> bool **)

let loopable self c =
  let looped =
    loop_calls (code_my_closure c) (code_params_arity c)
      (code_return_continuation c)
      (code_exn_continuation c) self (code_body c)
  in
  (((((negb (among (code_my_closure c) (code_params c))
      && distinct (code_params c))
     && negb (Pos.eqb self (code_return_continuation c)))
    && negb (Pos.eqb self (code_exn_continuation c)))
   && loop_safe (code_my_closure c)
        (code_return_continuation c)
        (code_exn_continuation c) self (code_body c))
  && cont_free_b self looped)
  && negb (code_is_tupled c)

(** val loopify_code : continuation -> code -> code **)

let loopify_code self c = if loopable self c then looped_code self c else c

type loopified = (code_id * continuation) list

(** val loopify_binding : loopified -> static_binding -> static_binding **)

let loopify_binding g binding =
  match binding with
  | Bind_code (id, c) -> (
    match lookup id g with
    | Some self -> Bind_code (id, loopify_code self c)
    | None -> binding)
  | _ -> binding

(** val loopify_unit : loopified -> expr -> expr **)

let rec loopify_unit g e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, loopify_unit g body)
  | Let_static (bindings, body) ->
    Let_static (map (loopify_binding g) bindings, loopify_unit g body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, loopify_unit g handler),
        loopify_unit g body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, loopify_unit g body)
  | _ -> e
