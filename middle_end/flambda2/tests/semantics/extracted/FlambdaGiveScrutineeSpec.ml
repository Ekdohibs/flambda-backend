open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaStatedScopeSpec
open FlambdaUnaliasParamsSpec
open FlambdaUnusedContsSpec
open List
open ListDef

(** val first_arm_of : continuation -> (coq_Z * apply_cont) list -> coq_Z option
    **)

let first_arm_of k arms =
  first_answer
    (fun pat ->
      let d, ac = pat in
      if Pos.eqb ac.ac_continuation k then Some d else None)
    arms

(** val only_arm : continuation -> coq_Z -> (coq_Z * apply_cont) list -> bool **)

let only_arm k c arms =
  forallb
    (fun pat ->
      let d, ac = pat in
      negb (Pos.eqb ac.ac_continuation k) || Z.eqb d c)
    arms

(** val arm_fact : variable -> continuation -> expr -> coq_Z option **)

let rec arm_fact x k = function
  | Let_cont_non_recursive (k', h, body0) ->
    if Pos.eqb k' k || cont_free_in_handler_b k h
    then None
    else arm_fact x k body0
  | Switch (scrutinee, arms) -> (
    match scrutinee with
    | Var y ->
      if Pos.eqb y x
      then
        match first_arm_of k arms with
        | Some c -> if only_arm k c arms then Some c else None
        | None -> None
      else None
    | _ -> None)
  | _ -> None

(** val arm_known : variable -> simple -> coq_Z -> coq_Z option -> coq_Z option
    **)

let arm_known x scrutinee d known =
  match scrutinee with
  | Var y -> if Pos.eqb y x then Some d else known
  | _ -> known

(** val known_under : variable -> variable list -> coq_Z option -> coq_Z option
    **)

let known_under x params known = if among x params then None else known

(** val give_nth : variable -> coq_Z -> nat -> simple list -> simple list **)

let rec give_nth x c i args =
  match i with
  | O -> (
    match args with
    | [] -> args
    | s :: rest -> (
      match s with
      | Const c0 -> (
        match c0 with
        | Naked_immediate c' -> if Z.eqb c' c then Var x :: rest else args
        | _ -> args)
      | _ -> args))
  | S i0 -> (
    match args with [] -> args | a :: rest -> a :: give_nth x c i0 rest)

(** val give_in_apply_cont : variable -> continuation -> nat -> coq_Z option ->
    apply_cont -> apply_cont **)

let give_in_apply_cont x k i known ac =
  match known with
  | Some c ->
    if Pos.eqb ac.ac_continuation k
    then
      { ac_continuation = ac.ac_continuation;
        ac_args = give_nth x c i ac.ac_args;
        ac_trap_action = ac.ac_trap_action
      }
    else ac
  | None -> ac

(** val give_scrutinee : variable -> continuation -> nat -> coq_Z option -> expr
    -> expr **)

let rec give_scrutinee x k i known e =
  match e with
  | Let (y, kind, d, body) ->
    Let
      ( y,
        kind,
        d,
        give_scrutinee x k i (if Pos.eqb y x then None else known) body )
  | Let_static (bindings, body) ->
    Let_static (bindings, give_scrutinee x k i known body)
  | Let_cont_non_recursive (k', h, body) ->
    Let_cont_non_recursive
      ( k',
        give_in_handler x k i
          (match arm_fact x k' body with Some c -> Some c | None -> known)
          h,
        give_scrutinee x k i known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k', h = pat in
            k', give_in_handler x k i known h)
          handlers,
        give_scrutinee x k i known body )
  | Apply_cont ac -> Apply_cont (give_in_apply_cont x k i known ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let d, ac = pat in
            d, give_in_apply_cont x k i (arm_known x scrutinee d known) ac)
          arms )
  | _ -> e

(** val give_in_handler : variable -> continuation -> nat -> coq_Z option ->
    continuation_handler -> continuation_handler **)

and give_in_handler x k i known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        give_scrutinee x k i (known_under x params known) handler )

(** val scrutinee_by : continuation -> nat -> variable list -> apply_cont ->
    variable list option **)

let scrutinee_by k i current ac =
  if Pos.eqb ac.ac_continuation k
  then
    Some
      (match nth_error ac.ac_args i with
      | Some s -> (
        match s with
        | Var y -> y :: []
        | Symbol _ -> []
        | Const c -> ( match c with Naked_immediate _ -> current | _ -> []))
      | None -> [])
  else None

(** val scrutinee_var : variable list -> simple -> variable list **)

let scrutinee_var current = function Var y -> y :: current | _ -> current

(** val scrutinee_of : variable list -> expr -> variable list **)

let rec scrutinee_of current = function
  | Let_cont_non_recursive (_, _, body) -> scrutinee_of current body
  | Switch (scrutinee, _) -> scrutinee_var current scrutinee
  | _ -> current

(** val first_scrutinee : continuation -> nat -> variable list -> expr ->
    variable list option **)

let rec first_scrutinee k i current = function
  | Let (_, _, _, body) -> first_scrutinee k i current body
  | Let_static (_, body) -> first_scrutinee k i current body
  | Let_cont_non_recursive (_, h, body) -> (
    match handler_first_scrutinee k i (scrutinee_of current body) h with
    | Some r -> Some r
    | None -> first_scrutinee k i current body)
  | Let_cont_recursive (handlers, body) -> (
    match
      first_answer
        (fun pat ->
          let _, h = pat in
          handler_first_scrutinee k i current h)
        handlers
    with
    | Some r -> Some r
    | None -> first_scrutinee k i current body)
  | Apply_cont ac -> scrutinee_by k i current ac
  | Switch (scrutinee, arms) ->
    first_answer
      (fun pat ->
        let _, ac = pat in
        scrutinee_by k i (scrutinee_var current scrutinee) ac)
      arms
  | _ -> None

(** val handler_first_scrutinee : continuation -> nat -> variable list ->
    continuation_handler -> variable list option **)

and handler_first_scrutinee k i current = function
  | Handler (_, _, handler) -> first_scrutinee k i current handler

(** val give_first : stated_in_scope -> variable list -> nat -> continuation ->
    continuation_handler -> expr -> expr **)

let rec give_first s xs i k h body =
  match xs with
  | [] -> body
  | x :: xs0 -> (
    let body' = give_scrutinee x k i None body in
    match unalias_param s i k h body' with
    | Some _ -> body'
    | None -> give_first s xs0 i k h body)

(** val give_at : stated_in_scope -> nat -> continuation -> continuation_handler
    -> expr -> expr **)

let give_at s i k h body =
  match first_scrutinee k i [] body with
  | Some xs -> give_first s xs i k h body
  | None -> body

(** val give_from : stated_in_scope -> nat -> continuation ->
    continuation_handler -> expr -> expr **)

let rec give_from s i k h body =
  match i with O -> body | S i0 -> give_from s i0 k h (give_at s i0 k h body)

(** val give_scrutinees_under : stated_in_scope -> expr -> expr **)

let rec give_scrutinees_under s e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, give_scrutinees_under ((x, stated_of_named s d) :: s) body)
  | Let_static (bindings, body) ->
    Let_static (bindings, give_scrutinees_under s body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let h =
      Handler
        ( params,
          metadata,
          give_scrutinees_under
            (params_stated params metadata.hm_params_subkinds s)
            handler )
    in
    Let_cont_non_recursive
      (k, h, give_from s (length params) k h (give_scrutinees_under s body))
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, give_scrutinees_under s body)
  | _ -> e
