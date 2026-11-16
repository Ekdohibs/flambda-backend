open BinNums
open BinPos
open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaTypesSpec
open List
open ListDef

(** val var_in_type : variable -> flambda_type -> bool **)

let var_in_type x = function
  | Equals s -> var_in_simple x s
  | Closure_of (_, slots) -> var_in_value_slots x slots
  | _ -> false

(** val drop_var : variable -> typing_env -> typing_env **)

let drop_var x tenv =
  filter
    (fun pat ->
      let y, t = pat in
      negb (Pos.eqb x y) && negb (var_in_type x t))
    tenv

(** val drop_vars : variable list -> typing_env -> typing_env **)

let drop_vars xs tenv = fold_right drop_var tenv xs

(** val canonical_value_slots : typing_env -> (value_slot * simple) list ->
    (value_slot * simple) list **)

let canonical_value_slots tenv slots =
  map (fun slot -> fst slot, canonical_simple tenv (snd slot)) slots

(** val find_closure : typing_env -> variable -> (code_id * (value_slot *
    simple) list) option **)

let rec find_closure tenv x =
  match tenv with
  | [] -> None
  | p :: tenv0 -> (
    let y, f = p in
    match f with
    | Closure_of (code, slots) ->
      if Pos.eqb x y then Some (code, slots) else find_closure tenv0 x
    | _ -> find_closure tenv0 x)

(** val closure_of_simple : typing_env -> simple -> (code_id * (value_slot *
    simple) list) option **)

let closure_of_simple tenv = function Var x -> find_closure tenv x | _ -> None

(** val canonical_project : typing_env -> value_slot -> simple -> named **)

let canonical_project tenv slot closure =
  match closure_of_simple tenv closure with
  | Some p -> (
    let _, slots = p in
    match lookup slot slots with
    | Some contents -> Simple contents
    | None -> Project_value_slot (slot, closure))
  | None -> Project_value_slot (slot, closure)

(** val canonical_named : typing_env -> named -> named **)

let canonical_named tenv = function
  | Simple s -> Simple (canonical_simple tenv s)
  | Prim (p, args) -> Prim (p, map (canonical_simple tenv) args)
  | Set_of_closures (code, slots) ->
    Set_of_closures (code, canonical_value_slots tenv slots)
  | Closure_of_set (slot, decls, slots) ->
    Closure_of_set (slot, decls, canonical_value_slots tenv slots)
  | Project_value_slot (slot, closure) ->
    canonical_project tenv slot (canonical_simple tenv closure)
  | Project_value_slot_from (project_from, slot, closure) ->
    Project_value_slot_from (project_from, slot, canonical_simple tenv closure)
  | Project_function_slot (move_from, move_to, closure) ->
    Project_function_slot (move_from, move_to, canonical_simple tenv closure)

(** val canonical_static_const : typing_env -> static_const -> static_const **)

let canonical_static_const tenv = function
  | Static_const_set_of_closures (code, slots) ->
    Static_const_set_of_closures (code, canonical_value_slots tenv slots)
  | Static_const_closure_of_set (slot, symbols, decls, slots) ->
    Static_const_closure_of_set
      (slot, symbols, decls, canonical_value_slots tenv slots)
  | Static_const_block (tag, fields) ->
    Static_const_block (tag, map (canonical_simple tenv) fields)
  | Static_const_boxed_number contents ->
    Static_const_boxed_number (canonical_simple tenv contents)
  | Static_const_immutable_string bytes -> Static_const_immutable_string bytes
  | Static_const_immutable_value_array elements ->
    Static_const_immutable_value_array (map (canonical_simple tenv) elements)

(** val canonical_static_binding : typing_env -> static_binding ->
    static_binding **)

let canonical_static_binding tenv = function
  | Bind_symbol (s, const) -> Bind_symbol (s, canonical_static_const tenv const)
  | x -> x

(** val canonical_apply_cont : typing_env -> apply_cont -> apply_cont **)

let canonical_apply_cont tenv ac =
  { ac_continuation = ac.ac_continuation;
    ac_args = map (canonical_simple tenv) ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val canonical_call_kind : typing_env -> simple -> call_kind -> call_kind **)

let canonical_call_kind tenv callee kind =
  match kind with
  | Indirect_unknown_arity -> (
    match closure_of_simple tenv callee with
    | Some p ->
      let code, _ = p in
      Direct code
    | None -> kind)
  | _ -> kind

(** val canonical_apply : typing_env -> apply_expr -> apply_expr **)

let canonical_apply tenv ap =
  let callee = canonical_simple tenv ap.ap_callee in
  { ap_callee = callee;
    ap_call_kind = canonical_call_kind tenv callee ap.ap_call_kind;
    ap_args = map (canonical_simple tenv) ap.ap_args;
    ap_continuation = ap.ap_continuation;
    ap_exn_continuation = ap.ap_exn_continuation;
    ap_exn_extra_args =
      map
        (fun arg -> canonical_simple tenv (fst arg), snd arg)
        ap.ap_exn_extra_args;
    ap_args_arity = ap.ap_args_arity;
    ap_return_arity = ap.ap_return_arity
  }

(** val typing_env_under_let : typing_env -> variable -> named -> typing_env **)

let typing_env_under_let tenv x = function
  | Simple s ->
    if var_in_simple x s
    then drop_var x tenv
    else (x, Equals s) :: drop_var x tenv
  | Prim (p, _) -> (
    match p with
    | Heap p0 -> (
      match p0 with
      | Make_block_with_identity (_, _) -> drop_var x tenv
      | Make_mutable_array _ -> drop_var x tenv
      | _ -> (x, Equals (Const (Tagged_immediate Z0))) :: drop_var x tenv)
    | _ -> drop_var x tenv)
  | Set_of_closures (code, slots) ->
    if var_in_value_slots x slots
    then drop_var x tenv
    else (x, Closure_of (code, slots)) :: drop_var x tenv
  | _ -> drop_var x tenv

(** val follow_aliases : typing_env -> expr -> expr **)

let rec follow_aliases tenv = function
  | Let (x, kind, d, body) ->
    let d' = canonical_named tenv d in
    Let (x, kind, d', follow_aliases (typing_env_under_let tenv x d') body)
  | Let_static (bindings, body) ->
    Let_static
      (map (canonical_static_binding tenv) bindings, follow_aliases tenv body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, follow_aliases_in_handler tenv h, follow_aliases tenv body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, follow_aliases tenv body)
  | Apply ap -> Apply (canonical_apply tenv ap)
  | Apply_cont ac -> Apply_cont (canonical_apply_cont tenv ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( canonical_simple tenv scrutinee,
        map (fun arm -> fst arm, canonical_apply_cont tenv (snd arm)) arms )
  | Invalid -> Invalid

(** val follow_aliases_in_handler : typing_env -> continuation_handler ->
    continuation_handler **)

and follow_aliases_in_handler tenv = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, follow_aliases (drop_vars params tenv) handler)
