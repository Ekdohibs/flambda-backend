open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open PeanoNat

(** val tail_args_stated : stated_in_scope -> param_subkind list -> expr -> bool
    **)

let rec tail_args_stated s sks = function
  | Let (x, _, d, body0) ->
    tail_args_stated ((x, stated_of_named s d) :: s) sks body0
  | Let_static (_, body0) -> tail_args_stated s sks body0
  | Apply_cont ac -> are_stated s ac.ac_args sks
  | _ -> false

(** val inline_tail_conts_kinded : expr -> expr **)

let rec inline_tail_conts_kinded e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, inline_tail_conts_kinded body)
  | Let_static (bindings, body) ->
    Let_static (bindings, inline_tail_conts_kinded body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let handler' = inline_tail_conts_kinded handler in
    let body' = inline_tail_conts_kinded body in
    let kept =
      Let_cont_non_recursive (k, Handler (params, metadata, handler'), body')
    in
    if
      Nat.eqb (length params) (length metadata.hm_params_kinds)
      && tail_args_stated [] metadata.hm_params_subkinds body'
    then
      match inline_cont_at_tail k params metadata handler' body' with
      | Some inlined -> inlined
      | None -> kept
    else kept
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, inline_tail_conts_kinded body)
  | _ -> e
