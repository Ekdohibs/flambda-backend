open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open List
open ListDef
open PeanoNat

type positions = (continuation * nat list) list

(** val positions_of : positions -> continuation -> nat list **)

let positions_of sites k =
  match lookup k sites with Some us -> us | None -> []

(** val at_position : nat list -> nat -> bool **)

let at_position us i = existsb (Nat.eqb i) us

(** val kept_from : nat list -> nat -> 'a1 list -> 'a1 list **)

let rec kept_from us i = function
  | [] -> []
  | a :: l0 ->
    if at_position us i
    then kept_from us (S i) l0
    else a :: kept_from us (S i) l0

(** val moved_from : nat list -> nat -> 'a1 list -> 'a1 list **)

let rec moved_from us i = function
  | [] -> []
  | a :: l0 ->
    if at_position us i
    then a :: moved_from us (S i) l0
    else moved_from us (S i) l0

(** val untagged_kind : flambda_kind **)

let untagged_kind = Naked_number Kind_naked_immediate

type tags = (variable * variable) list

(** val tag_of : named -> variable option **)

let tag_of = function
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Tag_immediate -> (
        match args with
        | [] -> None
        | s :: l -> (
          match s with
          | Var n -> ( match l with [] -> Some n | _ :: _ -> None)
          | _ -> None))
      | _ -> None)
    | _ -> None)
  | _ -> None

(** val tags_without : variable list -> tags -> tags **)

let tags_without xs tagged =
  filter
    (fun pat ->
      let v, n = pat in
      negb (among v xs) && negb (among n xs))
    tagged

(** val tags_under_let : variable -> named -> tags -> tags **)

let tags_under_let x d tagged =
  match tag_of d with
  | Some n ->
    if Pos.eqb n x
    then tags_without (x :: []) tagged
    else (x, n) :: tags_without (x :: []) tagged
  | None -> tags_without (x :: []) tagged

(** val untag_simple : tags -> simple -> simple **)

let untag_simple tagged s =
  match s with
  | Var v -> ( match lookup v tagged with Some n -> Var n | None -> s)
  | Symbol _ -> s
  | Const c -> (
    match c with Tagged_immediate i -> Const (Naked_immediate i) | _ -> s)

(** val untag_args : nat list -> tags -> simple list -> simple list **)

let untag_args us tagged args =
  app (kept_from us O args)
    (rev (map (untag_simple tagged) (moved_from us O args)))

(** val untag_apply_cont : positions -> tags -> apply_cont -> apply_cont **)

let untag_apply_cont active tagged ac =
  { ac_continuation = ac.ac_continuation;
    ac_args =
      untag_args (positions_of active ac.ac_continuation) tagged ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val untag_of : named -> variable option **)

let untag_of = function
  | Prim (p0, args) -> (
    match p0 with
    | Unary p1 -> (
      match p1 with
      | Untag_immediate -> (
        match args with
        | [] -> None
        | s :: l -> (
          match s with
          | Var p -> ( match l with [] -> Some p | _ :: _ -> None)
          | _ -> None))
      | _ -> None)
    | _ -> None)
  | _ -> None

(** val untag_named : variable list -> named -> named **)

let untag_named untagged d =
  match untag_of d with
  | Some p -> if among p untagged then Simple (Var p) else d
  | None -> d

(** val untag_params_list : nat list -> variable list -> variable list **)

let untag_params_list us params =
  app (kept_from us O params) (rev (moved_from us O params))

(** val stated_of_params : handler_metadata -> param_subkind list **)

let stated_of_params metadata =
  firstn (length metadata.hm_params_kinds) metadata.hm_params_subkinds

(** val untag_metadata : nat list -> handler_metadata -> handler_metadata **)

let untag_metadata us metadata =
  { hm_params_kinds =
      app
        (kept_from us O metadata.hm_params_kinds)
        (map
           (fun _ -> untagged_kind)
           (rev (moved_from us O metadata.hm_params_kinds)));
    hm_params_subkinds =
      app
        (kept_from us O (stated_of_params metadata))
        (map
           (fun _ -> any_value)
           (rev (moved_from us O (stated_of_params metadata))));
    hm_is_exn_handler = metadata.hm_is_exn_handler
  }

(** val untagged_under : nat list -> variable list -> variable list -> variable
    list **)

let untagged_under us params untagged =
  app (moved_from us O params) (without params untagged)

(** val active_under : positions -> (continuation * continuation_handler) list
    -> positions -> positions **)

let active_under sites handlers active =
  app
    (map
       (fun pat ->
         let k, _ = pat in
         k, positions_of sites k)
       handlers)
    active

(** val untag_expr : positions -> positions -> variable list -> tags -> expr ->
    expr **)

let rec untag_expr sites active untagged tagged e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        untag_named untagged d,
        untag_expr sites active
          (without (x :: []) untagged)
          (tags_under_let x d tagged)
          body )
  | Let_static (bindings, body) ->
    Let_static (bindings, untag_expr sites active untagged tagged body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        untag_handler sites active untagged tagged (positions_of sites k) h,
        untag_expr sites
          ((k, positions_of sites k) :: active)
          untagged tagged body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            ( k,
              untag_handler sites
                (active_under sites handlers active)
                untagged tagged (positions_of sites k) h ))
          handlers,
        untag_expr sites
          (active_under sites handlers active)
          untagged tagged body )
  | Apply_cont ac -> Apply_cont (untag_apply_cont active tagged ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let i, ac = pat in
            i, untag_apply_cont active tagged ac)
          arms )
  | _ -> e

(** val untag_handler : positions -> positions -> variable list -> tags -> nat
    list -> continuation_handler -> continuation_handler **)

and untag_handler sites active untagged tagged us = function
  | Handler (params, metadata, handler) ->
    Handler
      ( untag_params_list us params,
        untag_metadata us metadata,
        untag_expr sites active
          (untagged_under us params untagged)
          (tags_without params tagged)
          handler )

(** val none_in : variable list -> (variable -> bool) -> bool **)

let none_in untagged mentions = forallb (fun p -> negb (mentions p)) untagged

(** val untagged_arg : variable list -> tags -> simple -> bool **)

let untagged_arg untagged tagged = function
  | Var x -> (
    among x untagged
    || match lookup x tagged with Some _ -> true | None -> false)
  | Symbol _ -> false
  | Const c -> ( match c with Tagged_immediate _ -> true | _ -> false)

(** val args_ok : variable list -> tags -> nat list -> simple list -> bool **)

let args_ok untagged tagged us args =
  none_in untagged (fun p -> var_in_simples p (kept_from us O args))
  && forallb (untagged_arg untagged tagged) (moved_from us O args)

(** val apply_cont_ok : positions -> variable list -> tags -> apply_cont -> bool
    **)

let apply_cont_ok active untagged tagged ac =
  args_ok untagged tagged (positions_of active ac.ac_continuation) ac.ac_args

(** val reads_untagged : variable list -> named -> bool **)

let reads_untagged untagged d =
  match untag_of d with Some p -> among p untagged | None -> false

(** val named_ok : variable list -> named -> bool **)

let named_ok untagged d =
  reads_untagged untagged d || none_in untagged (fun p -> var_in_named p d)

(** val no_positions : positions -> continuation -> bool **)

let no_positions active k =
  match positions_of active k with [] -> true | _ :: _ -> false

(** val handler_fits : nat list -> variable list -> handler_metadata -> bool **)

let handler_fits us params metadata =
  distinct params
  && match us with [] -> true | _ :: _ -> negb metadata.hm_is_exn_handler

(** val untag_ok : positions -> positions -> variable list -> tags -> expr ->
    bool **)

let rec untag_ok sites active untagged tagged = function
  | Let (x, _, d, body) ->
    named_ok untagged d
    && untag_ok sites active
         (without (x :: []) untagged)
         (tags_under_let x d tagged)
         body
  | Let_static (bindings, body) ->
    none_in untagged (fun p -> existsb (var_in_static_binding p) bindings)
    && untag_ok sites active untagged tagged body
  | Let_cont_non_recursive (k, h, body) ->
    handler_ok sites active untagged tagged (positions_of sites k) h
    && untag_ok sites ((k, positions_of sites k) :: active) untagged tagged body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        handler_ok sites
          (active_under sites handlers active)
          untagged tagged (positions_of sites k) h)
      handlers
    && untag_ok sites (active_under sites handlers active) untagged tagged body
  | Apply ap ->
    (none_in untagged (fun p ->
         (var_in_simple p ap.ap_callee || var_in_simples p ap.ap_args)
         || var_in_simples p (exn_extra_simples ap))
    && no_positions active ap.ap_continuation)
    && no_positions active ap.ap_exn_continuation
  | Apply_cont ac -> apply_cont_ok active untagged tagged ac
  | Switch (scrutinee, arms) ->
    none_in untagged (fun p -> var_in_simple p scrutinee)
    && forallb
         (fun pat ->
           let _, ac = pat in
           apply_cont_ok active untagged tagged ac)
         arms
  | Invalid -> true

(** val handler_ok : positions -> positions -> variable list -> tags -> nat list
    -> continuation_handler -> bool **)

and handler_ok sites active untagged tagged us = function
  | Handler (params, metadata, handler) ->
    handler_fits us params metadata
    && untag_ok sites active
         (untagged_under us params untagged)
         (tags_without params tagged)
         handler

(** val untag_at : positions -> expr -> expr **)

let untag_at sites e =
  if untag_ok sites [] [] [] e then untag_expr sites [] [] [] e else e

(** val value_positions : nat -> flambda_kind list -> nat list **)

let rec value_positions i = function
  | [] -> []
  | kind :: kinds0 ->
    if kind_eqb kind Value
    then i :: value_positions (S i) kinds0
    else value_positions (S i) kinds0

(** val candidates : expr -> positions **)

let rec candidates = function
  | Let (_, _, _, body) -> candidates body
  | Let_static (_, body) -> candidates body
  | Let_cont_non_recursive (k, h, body) ->
    app (handler_candidates k h) (candidates body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           handler_candidates k h)
         handlers)
      (candidates body)
  | _ -> []

(** val handler_candidates : continuation -> continuation_handler -> positions **)

and handler_candidates k = function
  | Handler (_, metadata, handler) ->
    (k, value_positions O metadata.hm_params_kinds) :: candidates handler

type origin_map = (variable * (continuation * nat)) list

(** val origins : origin_map -> (variable -> bool) -> (continuation * nat) list
    **)

let origins origin mentions =
  flat_map
    (fun pat ->
      let x, o = pat in
      if mentions x then o :: [] else [])
    origin

(** val all_of : positions -> continuation -> (continuation * nat) list **)

let all_of active k = map (fun u -> k, u) (positions_of active k)

(** val params_origins : continuation -> nat list -> nat -> variable list ->
    origin_map **)

let rec params_origins k us i = function
  | [] -> []
  | p :: params0 ->
    if at_position us i
    then (p, (k, i)) :: params_origins k us (S i) params0
    else params_origins k us (S i) params0

(** val origins_under : continuation -> nat list -> variable list -> origin_map
    -> origin_map **)

let origins_under k us params origin =
  app
    (params_origins k us O params)
    (filter
       (fun pat ->
         let x, _ = pat in
         negb (among x params))
       origin)

(** val args_blame : origin_map -> tags -> continuation -> nat list -> nat ->
    simple list -> (continuation * nat) list **)

let rec args_blame origin tagged k us i = function
  | [] -> []
  | a :: args0 ->
    app
      (if at_position us i
       then if untagged_arg (map fst origin) tagged a then [] else (k, i) :: []
       else origins origin (fun x -> var_in_simple x a))
      (args_blame origin tagged k us (S i) args0)

(** val apply_cont_blame : positions -> origin_map -> tags -> apply_cont ->
    (continuation * nat) list **)

let apply_cont_blame active origin tagged ac =
  args_blame origin tagged ac.ac_continuation
    (positions_of active ac.ac_continuation)
    O ac.ac_args

(** val blame : positions -> positions -> origin_map -> tags -> expr ->
    (continuation * nat) list **)

let rec blame sites active origin tagged = function
  | Let (x, _, d, body) ->
    app
      (if reads_untagged (map fst origin) d
       then []
       else origins origin (fun y -> var_in_named y d))
      (blame sites active (unbind x origin) (tags_under_let x d tagged) body)
  | Let_static (bindings, body) ->
    app
      (origins origin (fun y -> existsb (var_in_static_binding y) bindings))
      (blame sites active origin tagged body)
  | Let_cont_non_recursive (k, h, body) ->
    app
      (handler_blame sites active origin tagged k h)
      (blame sites ((k, positions_of sites k) :: active) origin tagged body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let k, h = pat in
           handler_blame sites
             (active_under sites handlers active)
             origin tagged k h)
         handlers)
      (blame sites (active_under sites handlers active) origin tagged body)
  | Apply ap ->
    app
      (origins origin (fun y ->
           (var_in_simple y ap.ap_callee || var_in_simples y ap.ap_args)
           || var_in_simples y (exn_extra_simples ap)))
      (app
         (all_of active ap.ap_continuation)
         (all_of active ap.ap_exn_continuation))
  | Apply_cont ac -> apply_cont_blame active origin tagged ac
  | Switch (scrutinee, arms) ->
    app
      (origins origin (fun y -> var_in_simple y scrutinee))
      (flat_map
         (fun pat ->
           let _, ac = pat in
           apply_cont_blame active origin tagged ac)
         arms)
  | Invalid -> []

(** val handler_blame : positions -> positions -> origin_map -> tags ->
    continuation -> continuation_handler -> (continuation * nat) list **)

and handler_blame sites active origin tagged k = function
  | Handler (params, metadata, handler) ->
    app
      (if handler_fits (positions_of sites k) params metadata
       then []
       else map (fun u -> k, u) (positions_of sites k))
      (blame sites active
         (origins_under k (positions_of sites k) params origin)
         (tags_without params tagged)
         handler)

(** val blamed : (continuation * nat) list -> continuation -> nat -> bool **)

let blamed refused k u =
  existsb
    (fun pat ->
      let k', u' = pat in
      Pos.eqb k' k && Nat.eqb u' u)
    refused

(** val prune : (continuation * nat) list -> positions -> positions **)

let prune refused sites =
  map
    (fun pat ->
      let k, us = pat in
      k, filter (fun u -> negb (blamed refused k u)) us)
    sites

(** val settle_positions : nat -> positions -> expr -> positions **)

let rec settle_positions fuel sites e =
  match fuel with
  | O -> sites
  | S fuel0 -> (
    match blame sites [] [] [] e with
    | [] -> sites
    | p :: l -> settle_positions fuel0 (prune (p :: l) sites) e)

(** val count_positions : positions -> nat **)

let count_positions sites =
  list_sum
    (map
       (fun pat ->
         let _, us = pat in
         length us)
       sites)

(** val untag_positions : expr -> positions **)

let untag_positions e =
  settle_positions (S (count_positions (candidates e))) (candidates e) e

(** val untag_params : expr -> expr **)

let untag_params e = untag_at (untag_positions e) e
