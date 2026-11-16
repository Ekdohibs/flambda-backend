open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaSpineWalkSpec
open List
open ListDef

(** val drop_nth : nat -> 'a1 list -> 'a1 list **)

let drop_nth i l = app (firstn i l) (skipn (S i) l)

(** val first_answer : ('a1 -> 'a2 option) -> 'a1 list -> 'a2 option **)

let rec first_answer f = function
  | [] -> None
  | a :: l0 -> ( match f a with Some b -> Some b | None -> first_answer f l0)

(** val apply_cont_gives_var : continuation -> nat -> variable -> bool ->
    apply_cont -> bool **)

let apply_cont_gives_var k i x ok ac =
  (match ac.ac_trap_action with
    | Some action -> negb (Pos.eqb (trap_action_handler action) k)
    | None -> true)
  &&
  if Pos.eqb ac.ac_continuation k
  then
    match nth_error ac.ac_args i with
    | Some s -> ( match s with Var y -> ok && Pos.eqb x y | _ -> false)
    | None -> false
  else true

(** val all_give_var : continuation -> nat -> variable -> bool -> expr -> bool **)

let rec all_give_var k i x ok = function
  | Let (y, _, _, body) -> all_give_var k i x (ok && negb (Pos.eqb x y)) body
  | Let_static (_, body) -> all_give_var k i x ok body
  | Let_cont_non_recursive (k', h, body) ->
    (negb (Pos.eqb k' k) && handler_all_give_var k i x ok h)
    && all_give_var k i x ok body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k', h = pat in
        negb (Pos.eqb k' k) && handler_all_give_var k i x ok h)
      handlers
    && all_give_var k i x ok body
  | Apply ap ->
    negb (Pos.eqb ap.ap_continuation k)
    && negb (Pos.eqb ap.ap_exn_continuation k)
  | Apply_cont ac -> apply_cont_gives_var k i x ok ac
  | Switch (_, arms) ->
    forallb
      (fun pat ->
        let _, ac = pat in
        apply_cont_gives_var k i x ok ac)
      arms
  | Invalid -> true

(** val handler_all_give_var : continuation -> nat -> variable -> bool ->
    continuation_handler -> bool **)

and handler_all_give_var k i x ok = function
  | Handler (params, _, handler) ->
    all_give_var k i x (ok && negb (among x params)) handler

(** val given_var_by : continuation -> nat -> apply_cont -> variable option
    option **)

let given_var_by k i ac =
  if Pos.eqb ac.ac_continuation k
  then
    Some
      (match nth_error ac.ac_args i with
      | Some s -> ( match s with Var y -> Some y | _ -> None)
      | None -> None)
  else None

(** val first_given_var : continuation -> nat -> expr -> variable option option
    **)

let rec first_given_var k i = function
  | Let (_, _, _, body) -> first_given_var k i body
  | Let_static (_, body) -> first_given_var k i body
  | Let_cont_non_recursive (_, h, body) -> (
    match handler_first_given_var k i h with
    | Some r -> Some r
    | None -> first_given_var k i body)
  | Let_cont_recursive (handlers, body) -> (
    match
      first_answer
        (fun pat ->
          let _, h = pat in
          handler_first_given_var k i h)
        handlers
    with
    | Some r -> Some r
    | None -> first_given_var k i body)
  | Apply_cont ac -> given_var_by k i ac
  | Switch (_, arms) ->
    first_answer
      (fun pat ->
        let _, ac = pat in
        given_var_by k i ac)
      arms
  | _ -> None

(** val handler_first_given_var : continuation -> nat -> continuation_handler ->
    variable option option **)

and handler_first_given_var k i = function
  | Handler (_, _, handler) -> first_given_var k i handler

(** val drop_arg_of : continuation -> nat -> apply_cont -> apply_cont **)

let drop_arg_of k i ac =
  if Pos.eqb ac.ac_continuation k
  then
    { ac_continuation = ac.ac_continuation;
      ac_args = drop_nth i ac.ac_args;
      ac_trap_action = ac.ac_trap_action
    }
  else ac

(** val drop_arg : continuation -> nat -> expr -> expr **)

let rec drop_arg k i e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, drop_arg k i body)
  | Let_static (bindings, body) -> Let_static (bindings, drop_arg k i body)
  | Let_cont_non_recursive (k', h, body) ->
    Let_cont_non_recursive (k', drop_arg_in_handler k i h, drop_arg k i body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k', h = pat in
            k', drop_arg_in_handler k i h)
          handlers,
        drop_arg k i body )
  | Apply_cont ac -> Apply_cont (drop_arg_of k i ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let n, ac = pat in
            n, drop_arg_of k i ac)
          arms )
  | _ -> e

(** val drop_arg_in_handler : continuation -> nat -> continuation_handler ->
    continuation_handler **)

and drop_arg_in_handler k i = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, drop_arg k i handler)

(** val unalias_param : nat -> continuation -> continuation_handler -> expr ->
    (continuation_handler * expr) option **)

let unalias_param i k h body =
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
            ((negb metadata.hm_is_exn_handler && all_give_var k i x true body)
            && negb (among x params))
            && negb (among p (drop_nth i params))
          then
            Some
              ( Handler
                  ( drop_nth i params,
                    { hm_params_kinds = drop_nth i metadata.hm_params_kinds;
                      hm_is_exn_handler = false
                    },
                    Let (p, kind, Simple (Var x), handler) ),
                drop_arg k i body )
          else None
        | None -> None)
      | None -> None)
    | None -> None)
  | None -> None

(** val unalias_from : nat -> continuation -> continuation_handler -> expr ->
    continuation_handler * expr **)

let rec unalias_from i k h body =
  match i with
  | O -> h, body
  | S i0 -> (
    match unalias_param i0 k h body with
    | Some p ->
      let h', body' = p in
      unalias_from i0 k h' body'
    | None -> unalias_from i0 k h body)

(** val unalias_node : expr -> expr **)

let unalias_node e =
  match e with
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let h, body0 =
      unalias_from (length params) k (Handler (params, metadata, handler)) body
    in
    Let_cont_non_recursive (k, h, body0)
  | _ -> e

(** val unalias_params : expr -> expr **)

let unalias_params e = walk_spine unalias_node e
