open Datatypes
open FlambdaCodeLetsSpec
open FlambdaForwardSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaStatedScopeSpec
open FlambdaStatedSymbolsSpec
open ListDef
open PeanoNat

type cont_kinds = (continuation * (flambda_kind list * param_subkind list)) list

(** val have_kinds : kinds_in_scope -> simple list -> flambda_kind list -> bool
    **)

let rec have_kinds g args kinds =
  match args with
  | [] -> ( match kinds with [] -> true | _ :: _ -> false)
  | s :: args0 -> (
    match kinds with
    | [] -> false
    | kind :: kinds0 -> (
      match kind_of_simple_in_scope g s with
      | Some kind' -> kind_eqb kind' kind && have_kinds g args0 kinds0
      | None -> false))

(** val kinded_entry : continuation -> variable list -> handler_metadata -> expr
    -> expr -> (variable list * apply_cont) option **)

let kinded_entry k params metadata handler body =
  if Nat.eqb (length params) (length metadata.hm_params_kinds)
  then forwarding k params metadata handler body
  else None

(** val forward_kinded_apply_cont : symbols_stated -> kinds_in_scope ->
    stated_in_scope -> cont_kinds -> forwards -> apply_cont -> apply_cont **)

let forward_kinded_apply_cont y g s k f ac =
  match forward_of f ac.ac_continuation with
  | Some p -> (
    let params, target = p in
    match lookup ac.ac_continuation k with
    | Some p0 ->
      let kinds, sks = p0 in
      if
        ((Nat.eqb (length ac.ac_args) (length params) && no_trap_action target)
        && have_kinds g ac.ac_args kinds)
        && are_stated_knowing y s ac.ac_args sks
      then forwarded_keeping ac.ac_trap_action params ac.ac_args target
      else ac
    | None -> ac)
  | None -> ac

(** val forward_kinded_arm : symbols_stated -> kinds_in_scope -> stated_in_scope
    -> cont_kinds -> forwards -> apply_cont -> apply_cont **)

let forward_kinded_arm y g s k f ac =
  match forward_of f ac.ac_continuation with
  | Some p -> (
    let l, target = p in
    match l with
    | [] -> (
      match lookup ac.ac_continuation k with
      | Some p0 -> (
        let l0, _ = p0 in
        match l0 with
        | [] -> (
          match ac.ac_args with
          | [] ->
            if no_trap_action ac
            then forwarded [] [] target
            else forward_kinded_apply_cont y g s k f ac
          | _ :: _ -> forward_kinded_apply_cont y g s k f ac)
        | _ :: _ -> forward_kinded_apply_cont y g s k f ac)
      | None -> forward_kinded_apply_cont y g s k f ac)
    | _ :: _ -> forward_kinded_apply_cont y g s k f ac)
  | None -> forward_kinded_apply_cont y g s k f ac

(** val forward_kinded_under : symbols_stated -> kinds_in_scope ->
    stated_in_scope -> cont_kinds -> forwards -> expr -> expr **)

let rec forward_kinded_under y g s k f e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        d,
        forward_kinded_under y ((x, kind) :: g)
          ((x, stated_of_named s d) :: s)
          k f body )
  | Let_static (bindings, body) ->
    Let_static
      (bindings, forward_kinded_under (statics_stated bindings y) g s k f body)
  | Let_cont_non_recursive (k0, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    let kinds = metadata.hm_params_kinds in
    let sks = metadata.hm_params_subkinds in
    let handler1 =
      forward_kinded_under y
        (params_in_scope params kinds g)
        (params_stated params sks s)
        k f handler
    in
    Let_cont_non_recursive
      ( k0,
        Handler (params, metadata, handler1),
        forward_kinded_under y g s
          ((k0, (kinds, sks)) :: k)
          ((k0, kinded_entry k0 params metadata handler1 body) :: f)
          body )
  | Let_cont_recursive (handlers, body) ->
    let f0 =
      app
        (map
           (fun pat ->
             let k0, _ = pat in
             k0, None)
           handlers)
        f
    in
    Let_cont_recursive
      ( map
          (fun pat ->
            let k0, h = pat in
            k0, forward_kinded_in_handler y g s k f0 h)
          handlers,
        forward_kinded_under y g s k f0 body )
  | Apply_cont ac -> Apply_cont (forward_kinded_apply_cont y g s k f ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map (fun arm -> fst arm, forward_kinded_arm y g s k f (snd arm)) arms )
  | _ -> e

(** val forward_kinded_in_handler : symbols_stated -> kinds_in_scope ->
    stated_in_scope -> cont_kinds -> forwards -> continuation_handler ->
    continuation_handler **)

and forward_kinded_in_handler y g s k f = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        forward_kinded_under y
          (params_in_scope params metadata.hm_params_kinds g)
          (params_stated params metadata.hm_params_subkinds s)
          k f handler )

(** val forward_kinded : expr -> expr **)

let forward_kinded e = forward_kinded_under [] [] [] [] [] e

(** val forward_kinded_in_binding : symbols_stated -> static_binding ->
    static_binding **)

let forward_kinded_in_binding y binding =
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
            forward_kinded_under y
              (code_in_scope params my_closure metadata)
              (code_stated params my_closure metadata)
              [] [] body ) )
  | _ -> binding

(** val forward_kinded_in_code_under : symbols_stated -> expr -> expr **)

let rec forward_kinded_in_code_under y e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, forward_kinded_in_code_under y body)
  | Let_static (bindings, body) ->
    Let_static
      ( map (forward_kinded_in_binding (statics_stated bindings y)) bindings,
        forward_kinded_in_code_under (statics_stated bindings y) body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        forward_kinded_in_code_handler y h,
        forward_kinded_in_code_under y body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, forward_kinded_in_code_handler y h)
          handlers,
        forward_kinded_in_code_under y body )
  | _ -> e

(** val forward_kinded_in_code_handler : symbols_stated -> continuation_handler
    -> continuation_handler **)

and forward_kinded_in_code_handler y = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, forward_kinded_in_code_under y handler)

(** val forward_kinded_in_code : expr -> expr **)

let forward_kinded_in_code e = forward_kinded_in_code_under [] e
