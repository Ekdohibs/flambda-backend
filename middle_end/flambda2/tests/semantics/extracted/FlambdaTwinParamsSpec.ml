open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaUntagParamsSpec
open List
open ListDef
open Nat

(** val alias_of : named -> variable option **)

let alias_of = function
  | Simple s -> ( match s with Var p -> Some p | _ -> None)
  | _ -> None

type twinned = (variable * variable) list

(** val twin_simple : twinned -> simple -> simple **)

let twin_simple pairs s =
  match s with
  | Var x -> ( match lookup x pairs with Some p -> Var p | None -> s)
  | _ -> s

(** val twin_args : nat list -> twinned -> simple list -> simple list **)

let twin_args ts pairs args =
  app args (map (twin_simple pairs) (moved_from ts O args))

(** val twin_apply_cont : positions -> twinned -> apply_cont -> apply_cont **)

let twin_apply_cont active pairs ac =
  { ac_continuation = ac.ac_continuation;
    ac_args =
      twin_args (positions_of active ac.ac_continuation) pairs ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val twin_named : twinned -> named -> named **)

let twin_named pairs d =
  match untag_of d with
  | Some v -> (
    match lookup v pairs with
    | Some p -> Prim (Unary Untag_immediate, Var p :: [])
    | None -> d)
  | None -> d

(** val twin_names : variable list -> expr -> variable list **)

let rec twin_names pending e =
  match pending with
  | [] -> []
  | p :: rest -> (
    match e with
    | Let (x, _, d, body) -> (
      match alias_of d with
      | Some q ->
        if Pos.eqb q p
        then x :: twin_names rest body
        else twin_names pending body
      | None -> [])
    | _ -> [])

(** val renamed_from : nat list -> nat -> variable list -> variable list ->
    variable list **)

let rec renamed_from ts i params names =
  match params with
  | [] -> []
  | p :: params0 ->
    if at_position ts i
    then
      match names with
      | [] -> p :: renamed_from ts (S i) params0 []
      | x :: names0 -> x :: renamed_from ts (S i) params0 names0
    else p :: renamed_from ts (S i) params0 names

(** val twin_params_list : nat list -> variable list -> expr -> variable list **)

let twin_params_list ts params handler =
  app
    (renamed_from ts O params (twin_names (moved_from ts O params) handler))
    (moved_from ts O params)

(** val twin_metadata : nat list -> handler_metadata -> handler_metadata **)

let twin_metadata ts metadata =
  { hm_params_kinds =
      app metadata.hm_params_kinds (moved_from ts O metadata.hm_params_kinds);
    hm_params_subkinds =
      app
        (stated_of_params metadata)
        (map (fun _ -> any_value) (moved_from ts O (stated_of_params metadata)));
    hm_is_exn_handler = metadata.hm_is_exn_handler
  }

(** val twin_expr : positions -> positions -> twinned -> variable list -> expr
    -> expr **)

let rec twin_expr sites active pairs pending e =
  match e with
  | Let (x, kind, d, body) -> (
    match pending with
    | [] ->
      Let
        ( x,
          kind,
          twin_named pairs d,
          twin_expr sites active (tags_without (x :: []) pairs) [] body )
    | p :: rest -> (
      match alias_of d with
      | Some q ->
        if Pos.eqb q p
        then
          twin_expr sites active
            ((x, p) :: tags_without (x :: []) pairs)
            rest body
        else
          Let
            ( x,
              kind,
              d,
              twin_expr sites active (tags_without (x :: []) pairs) pending body
            )
      | None ->
        Let
          ( x,
            kind,
            twin_named pairs d,
            twin_expr sites active (tags_without (x :: []) pairs) [] body )))
  | Let_static (bindings, body) ->
    Let_static (bindings, twin_expr sites active pairs [] body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        twin_handler sites active pairs (positions_of sites k) h,
        twin_expr sites ((k, positions_of sites k) :: active) pairs [] body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            ( k,
              twin_handler sites
                (active_under sites handlers active)
                pairs (positions_of sites k) h ))
          handlers,
        twin_expr sites (active_under sites handlers active) pairs [] body )
  | Apply_cont ac -> Apply_cont (twin_apply_cont active pairs ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let i, ac = pat in
            i, twin_apply_cont active pairs ac)
          arms )
  | _ -> e

(** val twin_handler : positions -> positions -> twinned -> nat list ->
    continuation_handler -> continuation_handler **)

and twin_handler sites active pairs ts = function
  | Handler (params, metadata, handler) ->
    Handler
      ( twin_params_list ts params handler,
        twin_metadata ts metadata,
        twin_expr sites active
          (tags_without params pairs)
          (moved_from ts O params) handler )

(** val head_ok : variable list -> variable list -> expr -> bool **)

let rec head_ok avoid pending e =
  match pending with
  | [] -> true
  | p :: rest -> (
    match e with
    | Let (x, _, d, body) -> (
      match alias_of d with
      | Some q ->
        if Pos.eqb q p
        then head_ok avoid rest body
        else
          (negb (among x avoid) && negb (among q avoid))
          && head_ok avoid pending body
      | None -> false)
    | _ -> false)

(** val handler_twins_fit : nat list -> variable list -> expr -> bool **)

let handler_twins_fit ts params handler =
  distinct (twin_params_list ts params handler)
  && head_ok
       (app
          (twin_names (moved_from ts O params) handler)
          (moved_from ts O params))
       (moved_from ts O params) handler

(** val twin_ok : positions -> positions -> expr -> bool **)

let rec twin_ok sites active = function
  | Let (_, _, _, body) -> twin_ok sites active body
  | Let_static (_, body) -> twin_ok sites active body
  | Let_cont_non_recursive (k, h, body) ->
    twin_handler_ok sites active (positions_of sites k) h
    && twin_ok sites ((k, positions_of sites k) :: active) body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        twin_handler_ok sites
          (active_under sites handlers active)
          (positions_of sites k) h)
      handlers
    && twin_ok sites (active_under sites handlers active) body
  | Apply ap ->
    no_positions active ap.ap_continuation
    && no_positions active ap.ap_exn_continuation
  | _ -> true

(** val twin_handler_ok : positions -> positions -> nat list ->
    continuation_handler -> bool **)

and twin_handler_ok sites active ts = function
  | Handler (params, _, handler) ->
    handler_twins_fit ts params handler && twin_ok sites active handler

(** val twin_sites : positions -> expr -> positions **)

let rec twin_sites sites = function
  | Let (_, _, _, body) -> twin_sites sites body
  | Let_static (_, body) -> twin_sites sites body
  | Let_cont_non_recursive (k, h, body) ->
    app (handler_twin_sites sites k h) (twin_sites sites body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           handler_twin_sites sites k h)
         handlers)
      (twin_sites sites body)
  | _ -> []

(** val handler_twin_sites : positions -> continuation -> continuation_handler
    -> positions **)

and handler_twin_sites sites k = function
  | Handler (params, _, handler) ->
    ( k,
      seq (length params) (length (moved_from (positions_of sites k) O params))
    )
    :: twin_sites sites handler

(** val twin_untag_at : positions -> expr -> expr **)

let twin_untag_at sites e =
  if
    twin_ok sites [] e
    && untag_ok (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
  then untag_expr (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
  else e

(** val after_alias : variable -> expr -> expr option **)

let rec after_alias p = function
  | Let (_, _, d, body) -> (
    match alias_of d with
    | Some q -> if Pos.eqb q p then Some body else after_alias p body
    | None -> None)
  | _ -> None

(** val head_positions : nat -> variable list -> flambda_kind list -> expr ->
    nat list **)

let rec head_positions i params kinds e =
  match params with
  | [] -> []
  | p :: params0 -> (
    match kinds with
    | [] -> []
    | kind :: kinds0 -> (
      match if kind_eqb kind Value then after_alias p e else None with
      | Some rest -> i :: head_positions (S i) params0 kinds0 rest
      | None -> head_positions (S i) params0 kinds0 e))

(** val twin_candidates : expr -> positions **)

let rec twin_candidates = function
  | Let (_, _, _, body) -> twin_candidates body
  | Let_static (_, body) -> twin_candidates body
  | Let_cont_non_recursive (k, h, body) ->
    app (handler_twin_candidates k h) (twin_candidates body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           handler_twin_candidates k h)
         handlers)
      (twin_candidates body)
  | _ -> []

(** val handler_twin_candidates : continuation -> continuation_handler ->
    positions **)

and handler_twin_candidates k = function
  | Handler (params, metadata, handler) ->
    ( k,
      if metadata.hm_is_exn_handler
      then []
      else head_positions O params metadata.hm_params_kinds handler )
    :: twin_candidates handler

(** val twin_blame : positions -> positions -> expr -> (continuation * nat) list
    **)

let rec twin_blame sites active = function
  | Let (_, _, _, body) -> twin_blame sites active body
  | Let_static (_, body) -> twin_blame sites active body
  | Let_cont_non_recursive (k, h, body) ->
    app
      (twin_handler_blame sites active k h)
      (twin_blame sites ((k, positions_of sites k) :: active) body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           twin_handler_blame sites (active_under sites handlers active) k h)
         handlers)
      (twin_blame sites (active_under sites handlers active) body)
  | Apply ap ->
    app
      (all_of active ap.ap_continuation)
      (all_of active ap.ap_exn_continuation)
  | _ -> []

(** val twin_handler_blame : positions -> positions -> continuation ->
    continuation_handler -> (continuation * nat) list **)

and twin_handler_blame sites active k = function
  | Handler (params, _, handler) ->
    app
      (if handler_twins_fit (positions_of sites k) params handler
       then []
       else map (fun u -> k, u) (positions_of sites k))
      (twin_blame sites active handler)

(** val twin_origin : positions -> positions -> (continuation * nat) ->
    (continuation * nat) list **)

let twin_origin sites new0 = function
  | k, u -> (
    match positions_of new0 k with
    | [] -> []
    | n :: _ -> (
      match nth_error (positions_of sites k) (sub u n) with
      | Some i -> (k, i) :: []
      | None -> []))

(** val twin_settle : nat -> positions -> expr -> positions **)

let rec twin_settle fuel sites e =
  match fuel with
  | O -> sites
  | S fuel0 -> (
    match twin_blame sites [] e with
    | [] -> (
      match
        blame (twin_sites sites e) [] [] [] (twin_expr sites [] [] [] e)
      with
      | [] -> sites
      | p :: l ->
        twin_settle fuel0
          (prune
             (flat_map (twin_origin sites (twin_sites sites e)) (p :: l))
             sites)
          e)
    | p :: l -> twin_settle fuel0 (prune (p :: l) sites) e)

(** val twin_positions : expr -> positions **)

let twin_positions e =
  twin_settle (S (count_positions (twin_candidates e))) (twin_candidates e) e

(** val twin_params : expr -> expr **)

let twin_params e =
  match count_positions (twin_candidates e) with
  | O -> e
  | S _ -> twin_untag_at (twin_positions e) e
