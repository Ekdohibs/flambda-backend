open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaHeapSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaValueSpec
open List
open ListDef
open PeanoNat

type env = (variable * value) list

type cont_value =
  | Closure of
      continuation_handler
      * env
      * (continuation * cont_value) list
      * (continuation * continuation_handler) list
  | Outer of continuation
  | Extra_args of cont_value * value list
  | Over_application of value list * arity * arity * cont_value * cont_value

type cont_env = (continuation * cont_value) list

(** val resolve : continuation -> cont_env -> cont_value **)

let resolve k conts = match lookup k conts with Some c -> c | None -> Outer k

(** val rec_closures : (continuation * continuation_handler) list -> env ->
    cont_env -> cont_env **)

let rec_closures handlers vars conts =
  map
    (fun pat ->
      let k, h = pat in
      k, Closure (h, vars, conts, handlers))
    handlers

(** val eval_simple : env -> simple -> value option **)

let eval_simple vars = function
  | Var x -> lookup x vars
  | Symbol sym -> Some (Value_symbol sym)
  | Const c -> Some (Value_const c)

(** val eval_simples : env -> simple list -> value list option **)

let rec eval_simples vars = function
  | [] -> Some []
  | s :: args0 -> (
    match eval_simple vars s with
    | Some v -> (
      match eval_simples vars args0 with
      | Some vs -> Some (v :: vs)
      | None -> None)
    | None -> None)

(** val eval_value_slots : env -> (value_slot * simple) list -> (value_slot *
    value) list option **)

let rec eval_value_slots vars = function
  | [] -> Some []
  | p :: value_slots0 -> (
    let slot, s = p in
    match eval_simple vars s with
    | Some v -> (
      match eval_value_slots vars value_slots0 with
      | Some vs -> Some ((slot, v) :: vs)
      | None -> None)
    | None -> None)

(** val eval_named : static_env -> env -> named -> value option **)

let eval_named statics vars = function
  | Simple s -> eval_simple vars s
  | Prim (p, args) -> (
    match eval_simples vars args with
    | Some vs -> eval_prim statics p vs
    | None -> None)
  | Set_of_closures (code0, value_slots) -> (
    match eval_value_slots vars value_slots with
    | Some vs -> Some (Value_closure (code0, vs))
    | None -> None)
  | Closure_of_set (slot, function_decls, value_slots) -> (
    match lookup slot function_decls with
    | Some _ -> (
      match eval_value_slots vars value_slots with
      | Some vs -> Some (Value_closure_of_set (slot, function_decls, vs))
      | None -> None)
    | None -> None)
  | Project_value_slot (slot, closure) -> (
    match eval_simple vars closure with
    | Some v -> (
      match closure_of statics v with
      | Some called -> lookup slot (snd called)
      | None -> None)
    | None -> None)
  | Project_value_slot_from (project_from, slot, closure) -> (
    match eval_simple vars closure with
    | Some v -> project_value_slot statics project_from slot v
    | None -> None)
  | Project_function_slot (move_from, move_to, closure) -> (
    match eval_simple vars closure with
    | Some v -> project_function_slot statics move_from move_to v
    | None -> None)

(** val exec_named : static_env -> heap -> read_record -> env -> named ->
    ((value * heap) * read_record) option **)

let exec_named statics hp rd vars defining_expr =
  match defining_expr with
  | Prim (p, args) -> (
    match eval_simples vars args with
    | Some vs -> (
      match exec_prim statics hp p vs with
      | Some p0 -> Some (p0, rd)
      | None -> None)
    | None -> None)
  | _ -> (
    match eval_named statics vars defining_expr with
    | Some v -> Some ((v, hp), rd)
    | None -> None)

(** val eval_static_const : env -> static_const -> statically_allocated_value
    option **)

let eval_static_const vars = function
  | Static_const_set_of_closures (code0, value_slots) -> (
    match eval_value_slots vars value_slots with
    | Some vs -> Some (Static_closure (code0, vs))
    | None -> None)
  | Static_const_closure_of_set
      (slot, closure_symbols, function_decls, value_slots) -> (
    match lookup slot function_decls with
    | Some _ -> (
      match eval_value_slots vars value_slots with
      | Some vs ->
        Some (Static_closure_of_set (slot, closure_symbols, function_decls, vs))
      | None -> None)
    | None -> None)
  | Static_const_block (tag, fields) -> (
    match eval_simples vars fields with
    | Some vs -> Some (Static_block (tag, vs))
    | None -> None)
  | Static_const_boxed_number contents -> (
    match eval_simple vars contents with
    | Some v -> Some (Static_boxed_number v)
    | None -> None)
  | Static_const_immutable_string bytes -> Some (Static_string bytes)
  | Static_const_immutable_value_array elements -> (
    match eval_simples vars elements with
    | Some vs -> Some (Static_immutable_array vs)
    | None -> None)

(** val eval_static_bindings : env -> static_binding list -> static_env ->
    static_env option **)

let rec eval_static_bindings vars bindings statics =
  match bindings with
  | [] -> Some statics
  | s0 :: bindings0 -> (
    match s0 with
    | Bind_code (id, c) -> (
      match lookup id statics.static_code with
      | Some _ -> None
      | None ->
        eval_static_bindings vars bindings0
          { static_code = (id, c) :: statics.static_code;
            static_values = statics.static_values
          })
    | Bind_deleted_code _ -> eval_static_bindings vars bindings0 statics
    | Bind_symbol (s, const) -> (
      match lookup s statics.static_values with
      | Some _ -> None
      | None -> (
        match eval_static_const vars const with
        | Some allocated ->
          eval_static_bindings vars bindings0
            { static_code = statics.static_code;
              static_values = (s, allocated) :: statics.static_values
            }
        | None -> None)))

(** val binding_binds : static_binding -> bool **)

let binding_binds = function Bind_deleted_code _ -> false | _ -> true

(** val binds_statics : expr -> bool **)

let rec binds_statics = function
  | Let (_, _, _, body) -> binds_statics body
  | Let_static (bindings, body) ->
    existsb binding_binds bindings || binds_statics body
  | Let_cont_non_recursive (_, h, body) ->
    handler_binds_statics h || binds_statics body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_binds_statics h)
      handlers
    || binds_statics body
  | _ -> false

(** val handler_binds_statics : continuation_handler -> bool **)

and handler_binds_statics = function
  | Handler (_, _, handler) -> binds_statics handler

(** val handlers_bind_statics : (continuation * continuation_handler) list ->
    bool **)

let handlers_bind_statics handlers =
  existsb
    (fun pat ->
      let _, h = pat in
      handler_binds_statics h)
    handlers

(** val bind_params : variable list -> value list -> env -> env option **)

let rec bind_params params vs vars =
  match params with
  | [] -> ( match vs with [] -> Some vars | _ :: _ -> None)
  | x :: params0 -> (
    match vs with
    | [] -> None
    | v :: vs0 -> bind_params params0 vs0 ((x, v) :: vars))

(** val find_arm : coq_Z -> (coq_Z * apply_cont) list -> apply_cont option **)

let rec find_arm i = function
  | [] -> None
  | p :: arms0 ->
    let discriminant, ac = p in
    if Z.eqb i discriminant then Some ac else find_arm i arms0

(** val naked_immediate_of_value : value -> coq_Z option **)

let naked_immediate_of_value = function
  | Value_const c -> ( match c with Naked_immediate i -> Some i | _ -> None)
  | _ -> None

type state =
  { state_expr : expr;
    state_vars : env;
    state_conts : cont_env;
    state_statics : static_env;
    state_heap : heap;
    state_reads : read_record
  }

type step_result =
  | Step of state
  | Exit of continuation * value list * heap * read_record
  | Undefined

(** val with_extra_args : cont_value -> value list -> cont_value **)

let with_extra_args c extra =
  match extra with [] -> c | _ :: _ -> Extra_args (c, extra)

(** val call_code : value -> code -> value list -> arity -> arity -> cont_value
    -> cont_value -> static_env -> heap -> read_record -> step_result **)

let call_code closure c vs args_arity return_arity return_to exn_to statics hp
    rd =
  if binds_statics (code_body c)
  then Undefined
  else if
    negb
      ((kinds_eqb (map kind_of_value vs) (unarize args_arity)
       && kinds_per_parameter_eqb
            (unarize_per_parameter args_arity)
            (unarize_per_parameter (code_params_arity c)))
      && arity_eqb return_arity (code_result_arity c))
  then Undefined
  else
    match
      bind_params (code_params c) vs ((code_my_closure c, closure) :: [])
    with
    | Some callee_vars ->
      Step
        { state_expr = code_body c;
          state_vars = callee_vars;
          state_conts =
            [ code_return_continuation c, return_to;
              code_exn_continuation c, exn_to ];
          state_statics = statics;
          state_heap = hp;
          state_reads = rd
        }
    | None -> Undefined

type call_shape =
  | Full_call of value * code * value list * arity
  | Over_call of value * code * value list * value list * arity * arity
  | Partial_call of value

(** val tuple_access : nat -> block_access_kind **)

let tuple_access size =
  Block_access_values (Some Z0, Some (Z.of_nat size), Any_value)

(** val tuple_fields_from : static_env -> nat -> value -> nat -> nat -> value
    list option **)

let rec tuple_fields_from statics size tuple field = function
  | O -> Some []
  | S count0 -> (
    match
      eval_prim statics
        (Unary (Block_load (tuple_access size, Immutable, Z.of_nat field)))
        (tuple :: [])
    with
    | Some v -> (
      match tuple_fields_from statics size tuple (S field) count0 with
      | Some vs -> Some (v :: vs)
      | None -> None)
    | None -> None)

(** val tuple_fields : static_env -> nat -> value -> value list option **)

let tuple_fields statics size tuple =
  tuple_fields_from statics size tuple O size

(** val detuple : static_env -> nat -> value list -> arity -> (value list *
    arity) option **)

let detuple statics size vs args_arity =
  match vs with
  | [] -> None
  | tuple :: rest -> (
    match args_arity with
    | [] -> None
    | first :: rest_arity -> (
      if
        negb
          (kinds_eqb (map kind_of_value vs) (unarize args_arity)
          && kinds_eqb (unarize_component first) (Value :: []))
      then None
      else
        match tuple_fields statics size tuple with
        | Some fields ->
          Some
            ( app fields rest,
              app (arity_of_kinds (repeat Value size)) rest_arity )
        | None -> None))

(** val arguments_for_code : static_env -> code -> value list -> arity -> (value
    list * arity) option **)

let arguments_for_code statics c vs args_arity =
  if code_is_tupled c
  then detuple statics (cardinal_unarized (code_params_arity c)) vs args_arity
  else Some (vs, args_arity)

(** val shape_of_code_call : value -> code -> value list -> arity -> arity ->
    call_shape option **)

let shape_of_code_call closure c vs args_arity return_arity =
  let provided = length args_arity in
  let params = length (code_params_arity c) in
  if Nat.eqb provided params
  then Some (Full_call (closure, c, vs, args_arity))
  else if
    negb
      (kinds_eqb (map kind_of_value vs) (unarize args_arity)
      && partial_or_over_application args_arity return_arity
           (code_params_arity c) (code_result_arity c))
  then None
  else if Nat.ltb params provided
  then
    let size = cardinal_unarized (code_params_arity c) in
    Some
      (Over_call
         ( closure,
           c,
           firstn size vs,
           skipn size vs,
           firstn params args_arity,
           partially_apply args_arity params ))
  else if code_is_tupled c
  then None
  else Some (Partial_call (Value_partial_application (closure, vs, args_arity)))

(** val code_ids_allow : code_id list option -> code_id -> bool **)

let code_ids_allow code_ids called =
  match code_ids with Some ids -> among called ids | None -> true

(** val known_arity_full : static_env -> code_id -> arity -> bool **)

let known_arity_full statics called args_arity =
  match args_arity with
  | [] -> false
  | _ :: _ -> (
    match lookup called statics.static_code with
    | Some c ->
      negb (code_is_tupled c)
      && Nat.eqb (length args_arity) (length (code_params_arity c))
    | None -> false)

(** val shape_of_closure_call : static_env -> value -> call_kind -> value list
    -> arity -> arity -> call_shape option **)

let shape_of_closure_call statics closure kind vs args_arity return_arity =
  match closure_of statics closure with
  | Some p ->
    let called, _ = p in
    let expected =
      match kind with
      | Direct named0 -> Pos.eqb called named0
      | Indirect_unknown_arity -> true
      | Indirect_known_arity code_ids ->
        code_ids_allow code_ids called
        && known_arity_full statics called args_arity
    in
    if expected
    then
      match lookup called statics.static_code with
      | Some c -> (
        match arguments_for_code statics c vs args_arity with
        | Some p0 ->
          let vs', args_arity' = p0 in
          shape_of_code_call closure c vs' args_arity' return_arity
        | None -> None)
      | None -> None
    else None
  | None -> None

(** val shape_of_call : static_env -> value -> call_kind -> value list -> arity
    -> arity -> call_shape option **)

let shape_of_call statics callee kind vs args_arity return_arity =
  match kind with
  | Indirect_unknown_arity -> (
    match partial_application_of callee with
    | Some p -> (
      let p0, applied_arity = p in
      let closure, applied = p0 in
      match args_arity with
      | [] -> None
      | _ :: _ ->
        if kinds_eqb (map kind_of_value vs) (unarize args_arity)
        then
          shape_of_closure_call statics closure Indirect_unknown_arity
            (app applied vs)
            (app applied_arity args_arity)
            return_arity
        else None)
    | None ->
      shape_of_closure_call statics callee kind vs args_arity return_arity)
  | _ -> shape_of_closure_call statics callee kind vs args_arity return_arity

(** val run_call : (value -> step_result) -> call_shape option -> arity ->
    cont_value -> cont_value -> static_env -> heap -> read_record -> step_result
    **)

let run_call return_partial shape return_arity return_to exn_to statics hp rd =
  match shape with
  | Some c0 -> (
    match c0 with
    | Full_call (closure, c, vs, args_arity) ->
      call_code closure c vs args_arity return_arity return_to exn_to statics hp
        rd
    | Over_call (closure, c, vs, remaining, args_arity, remaining_arity) ->
      call_code closure c vs args_arity (code_result_arity c)
        (Over_application
           (remaining, remaining_arity, return_arity, return_to, exn_to))
        exn_to statics hp rd
    | Partial_call partial -> return_partial partial)
  | None -> Undefined

(** val apply_cont_value : cont_value -> value list -> static_env -> heap ->
    read_record -> step_result **)

let rec apply_cont_value c vs statics hp rd =
  match c with
  | Closure (h, def_vars, def_conts, recursive) ->
    let (Handler (params, metadata, handler)) = h in
    if kinds_eqb (map kind_of_value vs) metadata.hm_params_kinds
    then
      match bind_params params vs def_vars with
      | Some handler_vars ->
        Step
          { state_expr = handler;
            state_vars = handler_vars;
            state_conts =
              app (rec_closures recursive def_vars def_conts) def_conts;
            state_statics = statics;
            state_heap = hp;
            state_reads = rd
          }
      | None -> Undefined
    else Undefined
  | Outer k -> Exit (k, vs, hp, rd)
  | Extra_args (c0, extra) -> apply_cont_value c0 (app vs extra) statics hp rd
  | Over_application
      (remaining, remaining_arity, return_arity, return_to, exn_to) -> (
    match vs with
    | [] -> Undefined
    | v :: l -> (
      match l with
      | [] ->
        run_call
          (fun partial ->
            apply_cont_value return_to (partial :: []) statics hp rd)
          (shape_of_call statics v Indirect_unknown_arity remaining
             remaining_arity return_arity)
          return_arity return_to exn_to statics hp rd
      | _ :: _ -> Undefined))

(** val apply_continuation : continuation -> value list -> cont_env ->
    static_env -> heap -> read_record -> step_result **)

let apply_continuation k vs conts statics hp rd =
  apply_cont_value (resolve k conts) vs statics hp rd

(** val call_function : value -> call_kind -> value list -> arity -> arity ->
    cont_value -> cont_value -> static_env -> heap -> read_record -> step_result
    **)

let call_function callee kind vs args_arity return_arity return_to exn_to
    statics hp rd =
  run_call
    (fun partial -> apply_cont_value return_to (partial :: []) statics hp rd)
    (shape_of_call statics callee kind vs args_arity return_arity)
    return_arity return_to exn_to statics hp rd

(** val step : state -> step_result **)

let step s =
  let { state_expr = state_expr0;
        state_vars = vars;
        state_conts = conts;
        state_statics = statics;
        state_heap = hp;
        state_reads = rd
      } =
    s
  in
  match state_expr0 with
  | Let (x, kind, defining_expr, body) -> (
    match exec_named statics hp rd vars defining_expr with
    | Some p ->
      let p0, rd' = p in
      let v, hp' = p0 in
      if kind_eqb (kind_of_value v) kind
      then
        Step
          { state_expr = body;
            state_vars = (x, v) :: vars;
            state_conts = conts;
            state_statics = statics;
            state_heap = hp';
            state_reads = rd'
          }
      else Undefined
    | None -> Undefined)
  | Let_static (bindings, body) -> (
    match eval_static_bindings vars bindings statics with
    | Some statics' ->
      Step
        { state_expr = body;
          state_vars = vars;
          state_conts = conts;
          state_statics = statics';
          state_heap = hp;
          state_reads = rd
        }
    | None -> Undefined)
  | Let_cont_non_recursive (k, h, body) ->
    Step
      { state_expr = body;
        state_vars = vars;
        state_conts = (k, Closure (h, vars, conts, [])) :: conts;
        state_statics = statics;
        state_heap = hp;
        state_reads = rd
      }
  | Let_cont_recursive (handlers, body) ->
    if handlers_bind_statics handlers
    then Undefined
    else
      Step
        { state_expr = body;
          state_vars = vars;
          state_conts = app (rec_closures handlers vars conts) conts;
          state_statics = statics;
          state_heap = hp;
          state_reads = rd
        }
  | Apply ap -> (
    match eval_simple vars ap.ap_callee with
    | Some callee -> (
      match eval_simples vars ap.ap_args with
      | Some vs -> (
        match eval_simples vars (exn_extra_simples ap) with
        | Some extra ->
          call_function callee ap.ap_call_kind vs ap.ap_args_arity
            ap.ap_return_arity
            (resolve ap.ap_continuation conts)
            (with_extra_args (resolve ap.ap_exn_continuation conts) extra)
            statics hp rd
        | None -> Undefined)
      | None -> Undefined)
    | None -> Undefined)
  | Apply_cont ac -> (
    match eval_simples vars ac.ac_args with
    | Some vs -> apply_continuation ac.ac_continuation vs conts statics hp rd
    | None -> Undefined)
  | Switch (scrutinee, arms) -> (
    match eval_simple vars scrutinee with
    | Some v -> (
      match naked_immediate_of_value v with
      | Some i -> (
        match find_arm i arms with
        | Some ac ->
          Step
            { state_expr = Apply_cont ac;
              state_vars = vars;
              state_conts = conts;
              state_statics = statics;
              state_heap = hp;
              state_reads = rd
            }
        | None -> Undefined)
      | None -> Undefined)
    | None -> Undefined)
  | Invalid -> Undefined

(** val run : nat -> state -> step_result **)

let rec run fuel s =
  match fuel with
  | O -> Step s
  | S fuel0 -> ( match step s with Step s' -> run fuel0 s' | x -> x)
