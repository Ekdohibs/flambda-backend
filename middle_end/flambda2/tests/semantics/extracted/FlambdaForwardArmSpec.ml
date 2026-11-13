open BinNums
open BinPos
open Datatypes
open FlambdaCodeLetsSpec
open FlambdaForwardKindedSpec
open FlambdaForwardSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSemanticsSpec
open List
open ListDef
open PeanoNat

(** val position_of : variable -> variable list -> nat option **)

let rec position_of x = function
  | [] -> None
  | y :: params0 -> (
    if Pos.eqb x y
    then Some O
    else match position_of x params0 with Some i -> Some (S i) | None -> None)

type arm_entry =
  (((variable list * handler_metadata) * nat) * (coq_Z * apply_cont) list)
  * expr

type arm_forwards = (continuation * arm_entry option) list

(** val arm_entry_of : bool -> variable list -> handler_metadata -> expr -> expr
    -> arm_entry option **)

let arm_entry_of kinded params metadata handler body =
  match handler with
  | Switch (scrutinee, arms) -> (
    match scrutinee with
    | Var p -> (
      if
        kinded
        && negb (Nat.eqb (length params) (length metadata.hm_params_kinds))
      then None
      else
        match position_of p params with
        | Some i -> Some ((((params, metadata), i), arms), body)
        | None -> None)
    | _ -> None)
  | _ -> None

(** val arm_of_apply_cont : arm_forwards -> apply_cont -> ((variable list *
    flambda_kind list) * apply_cont) option **)

let arm_of_apply_cont f ac =
  match lookup ac.ac_continuation f with
  | Some o -> (
    match o with
    | Some a -> (
      let p, body = a in
      let p0, arms = p in
      let p1, i = p0 in
      let params, metadata = p1 in
      match nth_error ac.ac_args i with
      | Some s -> (
        match s with
        | Const c -> (
          match c with
          | Naked_immediate n -> (
            match find_arm n arms with
            | Some target -> (
              match
                forwarding ac.ac_continuation params metadata
                  (Apply_cont target) body
              with
              | Some p2 ->
                let params0, target0 = p2 in
                Some ((params0, metadata.hm_params_kinds), target0)
              | None -> None)
            | None -> None)
          | _ -> None)
        | _ -> None)
      | None -> None)
    | None -> None)
  | None -> None

(** val forward_to_arm_apply_cont : bool -> kinds_in_scope -> arm_forwards ->
    apply_cont -> apply_cont **)

let forward_to_arm_apply_cont kinded g f ac =
  match arm_of_apply_cont f ac with
  | Some p ->
    let p0, target = p in
    let params, kinds = p0 in
    if
      ((Nat.eqb (length ac.ac_args) (length params) && no_trap_action ac)
      && no_trap_action target)
      && (negb kinded || have_kinds g ac.ac_args kinds)
    then forwarded params ac.ac_args target
    else ac
  | None -> ac

(** val forward_to_arms_under : bool -> kinds_in_scope -> arm_forwards -> expr
    -> expr **)

let rec forward_to_arms_under kinded g f e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, forward_to_arms_under kinded ((x, kind) :: g) f body)
  | Let_static (bindings, body) ->
    Let_static (bindings, forward_to_arms_under kinded g f body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let handler1 =
      forward_to_arms_under kinded
        (params_in_scope params metadata.hm_params_kinds g)
        f handler
    in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, handler1),
        forward_to_arms_under kinded g
          ((k, arm_entry_of kinded params metadata handler1 body) :: f)
          body )
  | Let_cont_recursive (handlers, body) ->
    let f0 =
      app
        (map
           (fun pat ->
             let k, _ = pat in
             k, None)
           handlers)
        f
    in
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, forward_to_arms_in_handler kinded g f0 h)
          handlers,
        forward_to_arms_under kinded g f0 body )
  | Apply_cont ac -> Apply_cont (forward_to_arm_apply_cont kinded g f ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun arm -> fst arm, forward_to_arm_apply_cont kinded g f (snd arm))
          arms )
  | _ -> e

(** val forward_to_arms_in_handler : bool -> kinds_in_scope -> arm_forwards ->
    continuation_handler -> continuation_handler **)

and forward_to_arms_in_handler kinded g f = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        forward_to_arms_under kinded
          (params_in_scope params metadata.hm_params_kinds g)
          f handler )

(** val forward_to_arms : expr -> expr **)

let forward_to_arms e = forward_to_arms_under false [] [] e

(** val forward_to_arms_kinded : expr -> expr **)

let forward_to_arms_kinded e = forward_to_arms_under true [] [] e

(** val forward_to_arms_in_binding : static_binding -> static_binding **)

let forward_to_arms_in_binding binding =
  match binding with
  | Bind_code (id, c) ->
    let (Code (params, my_closure, ret, exn, metadata, body)) = c in
    Bind_code
      ( id,
        Code
          ( params,
            my_closure,
            ret,
            exn,
            metadata,
            forward_to_arms_under true
              (code_in_scope params my_closure metadata)
              [] body ) )
  | _ -> binding

(** val forward_to_arms_in_code : expr -> expr **)

let rec forward_to_arms_in_code e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, forward_to_arms_in_code body)
  | Let_static (bindings, body) ->
    Let_static
      (map forward_to_arms_in_binding bindings, forward_to_arms_in_code body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, forward_to_arms_in_code_handler h, forward_to_arms_in_code body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, forward_to_arms_in_code_handler h)
          handlers,
        forward_to_arms_in_code body )
  | _ -> e

(** val forward_to_arms_in_code_handler : continuation_handler ->
    continuation_handler **)

and forward_to_arms_in_code_handler = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, forward_to_arms_in_code handler)
