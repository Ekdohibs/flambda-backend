open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaUnaliasParamsSpec
open List

(** val unalias_rec_param : stated_in_scope -> nat -> continuation ->
    continuation_handler -> expr -> (continuation_handler * expr) option **)

let unalias_rec_param s i k h body =
  let (Handler (params, metadata, handler)) = h in
  match nth_error params i with
  | Some p -> (
    match nth_error metadata.hm_params_kinds i with
    | Some kind -> (
      match first_given_var k i body with
      | Some o -> (
        match o with
        | Some x ->
          if
            ((((negb metadata.hm_is_exn_handler && all_give_var k i x true body)
              && all_give_var k i p true handler)
             && negb (among x (drop_nth i params)))
            && negb (among p (drop_nth i params)))
            && is_stated_at s metadata.hm_params_subkinds i (Var x)
          then
            Some
              ( Handler
                  ( drop_nth i params,
                    { hm_params_kinds = drop_nth i metadata.hm_params_kinds;
                      hm_params_subkinds =
                        drop_nth i metadata.hm_params_subkinds;
                      hm_is_exn_handler = false
                    },
                    Let (p, kind, Simple (Var x), drop_arg k i handler) ),
                drop_arg k i body )
          else None
        | None -> None)
      | None -> None)
    | None -> None)
  | None -> None

(** val unalias_rec_from : stated_in_scope -> nat -> continuation ->
    continuation_handler -> expr -> continuation_handler * expr **)

let rec unalias_rec_from s i k h body =
  match i with
  | O -> h, body
  | S i0 -> (
    match unalias_rec_param s i0 k h body with
    | Some p ->
      let h', body' = p in
      unalias_rec_from s i0 k h' body'
    | None -> unalias_rec_from s i0 k h body)

(** val unalias_rec_params_under : stated_in_scope -> expr -> expr **)

let rec unalias_rec_params_under s e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      (x, kind, d, unalias_rec_params_under ((x, stated_of_named s d) :: s) body)
  | Let_static (bindings, body) ->
    Let_static (bindings, unalias_rec_params_under s body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler
          ( params,
            metadata,
            unalias_rec_params_under
              (params_stated params metadata.hm_params_subkinds s)
              handler ),
        unalias_rec_params_under s body )
  | Let_cont_recursive (handlers, body) -> (
    let body0 = unalias_rec_params_under s body in
    match handlers with
    | [] -> Let_cont_recursive (handlers, body0)
    | p :: l -> (
      let k, c = p in
      let (Handler (params, metadata, handler)) = c in
      match l with
      | [] ->
        let h, body1 =
          unalias_rec_from s (length params) k
            (Handler (params, metadata, handler))
            body0
        in
        Let_cont_recursive ((k, h) :: [], body1)
      | _ :: _ -> Let_cont_recursive (handlers, body0)))
  | _ -> e
