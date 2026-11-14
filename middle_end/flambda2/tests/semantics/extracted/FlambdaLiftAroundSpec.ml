open BinNums
open BinPos
open Bool
open Datatypes
open FlambdaBindersSpec
open FlambdaHeapFreeSpec
open FlambdaLiftSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaSemanticsSpec
open FlambdaWellFormedSpec
open List
open ListDef

type known_constants = (symbol * static_const) list

(** val constants_of_group : static_binding list -> known_constants **)

let constants_of_group bindings =
  flat_map
    (fun binding ->
      match binding with Bind_symbol (s, const) -> (s, const) :: [] | _ -> [])
    bindings

(** val constant_known : known_constants -> static_binding -> bool **)

let constant_known known = function
  | Bind_symbol (s, const) ->
    existsb
      (fun bound ->
        Pos.eqb s (fst bound)
        && if static_const_eq_dec const (snd bound) then true else false)
      known
  | _ -> false

(** val forget : variable list -> known_constants -> known_constants **)

let forget bound known =
  filter
    (fun constant ->
      negb
        (existsb
           (fun x -> among x bound)
           (free_variables_of_static_const (snd constant))))
    known

(** val alias_of : (path * lift_to) list -> path -> named -> (static_binding
    list * symbol) option **)

let alias_of lets at_path d =
  match lookup at_path lets with
  | Some target -> lift_named target d
  | None -> None

(** val group_at : (path * static_binding list) list -> path -> static_binding
    list **)

let group_at groups at_path =
  match lookup at_path groups with Some bindings -> bindings | None -> []

(** val lift_around : (path * lift_to) list -> (path * static_binding list) list
    -> path -> expr -> expr **)

let rec lift_around lets groups at_path e =
  match e with
  | Let (x, kind, d, body) -> (
    let body0 = lift_around lets groups (Coq_xO at_path) body in
    match alias_of lets at_path d with
    | Some p ->
      let _, s = p in
      Let (x, kind, Simple (Symbol s), body0)
    | None -> Let (x, kind, d, body0))
  | Let_static (bindings, body) ->
    Let_static (bindings, lift_around lets groups (Coq_xO at_path) body)
  | Let_cont_non_recursive (k, h, body) -> (
    let rest =
      Let_cont_non_recursive
        ( k,
          lift_around_handler lets groups (Coq_xI at_path) h,
          lift_around lets groups (Coq_xO at_path) body )
    in
    match group_at groups at_path with
    | [] -> rest
    | binding :: bindings -> Let_static (binding :: bindings, rest))
  | Let_cont_recursive (handlers, body) -> (
    let rest =
      Let_cont_recursive
        ( rec_handlers_map
            (lift_around_handler lets groups)
            (Coq_xI at_path) handlers,
          lift_around lets groups (Coq_xO at_path) body )
    in
    match group_at groups at_path with
    | [] -> rest
    | binding :: bindings -> Let_static (binding :: bindings, rest))
  | _ -> e

(** val lift_around_handler : (path * lift_to) list -> (path * static_binding
    list) list -> path -> continuation_handler -> continuation_handler **)

and lift_around_handler lets groups at_path = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, lift_around lets groups at_path handler)

(** val total_const : static_const -> bool **)

let total_const = function
  | Static_const_closure_of_set (slot, _, function_decls, _) -> (
    match lookup slot function_decls with Some _ -> true | None -> false)
  | _ -> true

(** val putable_group : static_binding list -> bool **)

let putable_group bindings =
  forallb
    (fun binding ->
      match binding with
      | Bind_symbol (_, const) -> total_const const
      | _ -> false)
    bindings

(** val group_in_scope : static_binding list -> expr -> bool **)

let group_in_scope bindings e =
  all_among
    (flat_map free_variables_of_binding bindings)
    (free_variables_of_expr e)

(** val around_ok : (path * lift_to) list -> (path * static_binding list) list
    -> known_constants -> path -> expr -> bool **)

let rec around_ok lets groups known at_path e =
  match e with
  | Let (x, _, d, body) ->
    (match alias_of lets at_path d with
      | Some p ->
        let bindings, _ = p in
        forallb (constant_known known) bindings
      | None -> true)
    && around_ok lets groups (forget (x :: []) known) (Coq_xO at_path) body
  | Let_static (bindings, body) ->
    around_ok lets groups
      (app (constants_of_group bindings) known)
      (Coq_xO at_path) body
  | Let_cont_non_recursive (_, h, body) ->
    let bindings = group_at groups at_path in
    let known0 = app (constants_of_group bindings) known in
    ((putable_group bindings && group_in_scope bindings e)
    && handler_around_ok lets groups known0 (Coq_xI at_path) h)
    && around_ok lets groups known0 (Coq_xO at_path) body
  | Let_cont_recursive (handlers, body) ->
    let bindings = group_at groups at_path in
    let known0 = app (constants_of_group bindings) known in
    (((putable_group bindings && group_in_scope bindings e)
     && eqb
          (handlers_bind_statics
             (rec_handlers_map
                (lift_around_handler lets groups)
                (Coq_xI at_path) handlers))
          (handlers_bind_statics handlers))
    && rec_handlers_forall
         (handler_around_ok lets groups known0)
         (Coq_xI at_path) handlers)
    && around_ok lets groups known0 (Coq_xO at_path) body
  | _ -> true

(** val handler_around_ok : (path * lift_to) list -> (path * static_binding
    list) list -> known_constants -> path -> continuation_handler -> bool **)

and handler_around_ok lets groups known at_path = function
  | Handler (params, _, handler) ->
    around_ok lets groups (forget params known) at_path handler

(** val bindings_of_lets : (path * lift_to) list -> path list -> path -> expr ->
    static_binding list **)

let rec bindings_of_lets lets wanted at_path = function
  | Let (_, _, d, body) ->
    app
      (if among at_path wanted
       then
         match alias_of lets at_path d with
         | Some p ->
           let bindings, _ = p in
           bindings
         | None -> []
       else [])
      (bindings_of_lets lets wanted (Coq_xO at_path) body)
  | Let_static (_, body) -> bindings_of_lets lets wanted (Coq_xO at_path) body
  | Let_cont_non_recursive (_, h, body) ->
    app
      (bindings_of_lets lets wanted (Coq_xO at_path) body)
      (handler_bindings_of_lets lets wanted (Coq_xI at_path) h)
  | Let_cont_recursive (handlers, body) ->
    app
      (bindings_of_lets lets wanted (Coq_xO at_path) body)
      (rec_handlers_bindings
         (handler_bindings_of_lets lets wanted)
         (Coq_xI at_path) handlers)
  | _ -> []

(** val handler_bindings_of_lets : (path * lift_to) list -> path list -> path ->
    continuation_handler -> static_binding list **)

and handler_bindings_of_lets lets wanted at_path = function
  | Handler (_, _, handler) -> bindings_of_lets lets wanted at_path handler

(** val groups_of : (path * lift_to) list -> (path * path list) list -> expr ->
    (path * static_binding list) list **)

let groups_of lets places e =
  map (fun place -> fst place, bindings_of_lets lets (snd place) root e) places

(** val lift_around_unit : unit_interface -> (path * lift_to) list -> (path *
    path list) list -> expr -> expr option **)

let lift_around_unit u lets places e =
  let groups = groups_of lets places e in
  let lifted = lift_around lets groups root e in
  if
    ((exits_plain_knowing [] [] e && expr_free_of_heap e)
    && around_ok lets groups [] root e)
    && well_formed_unit u lifted
  then Some lifted
  else None
