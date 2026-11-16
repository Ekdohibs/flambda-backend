open BinPos
open Datatypes
open FlambdaBindingTime
open FlambdaEnvAliases
open FlambdaEnvAliasesRenamingSpec
open FlambdaEnvAliasesSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaTypeGrammar
open FlambdaValueSpec
open List
open ListDef

module Known = struct
  type closure =
    | Of_code of code_id * (value_slot * simple) list
    | Of_set of
        function_slot
        * (function_slot * code_id) list
        * (value_slot * simple) list

  (** val closure_value_slots : closure -> (value_slot * simple) list **)

  let closure_value_slots = function
    | Of_code (_, value_slots) -> value_slots
    | Of_set (_, _, value_slots) -> value_slots

  (** val closure_code : closure -> code_id option **)

  let closure_code = function
    | Of_code (code, _) -> Some code
    | Of_set (slot, function_decls, _) -> lookup slot function_decls

  (** val closure_slot : closure -> function_slot **)

  let closure_slot = function
    | Of_code (_, _) -> unnamed_function_slot
    | Of_set (slot, _, _) -> slot

  type static_closure =
    { as_closure : closure;
      closure_symbols : (function_slot * symbol) list
    }

  (** val as_closure : static_closure -> closure **)

  let as_closure s = s.as_closure

  (** val closure_symbols : static_closure -> (function_slot * symbol) list **)

  let closure_symbols s = s.closure_symbols

  type t =
    { aliases : Aliases.t;
      binding_times :
        (name, unit * Binding_time.With_name_mode.t) Keyed.finite_map;
      next_binding_time : Binding_time.t;
      closures : (variable * closure) list;
      static_closures : (symbol * static_closure) list
    }

  (** val aliases : t -> Aliases.t **)

  let aliases t0 = t0.aliases

  (** val binding_times : t -> (name, unit * Binding_time.With_name_mode.t)
      Keyed.finite_map **)

  let binding_times t0 = t0.binding_times

  (** val next_binding_time : t -> Binding_time.t **)

  let next_binding_time t0 = t0.next_binding_time

  (** val closures : t -> (variable * closure) list **)

  let closures t0 = t0.closures

  (** val static_closures : t -> (symbol * static_closure) list **)

  let static_closures t0 = t0.static_closures

  (** val empty : t **)

  let empty =
    { aliases = Aliases.empty;
      binding_times = [];
      next_binding_time = Binding_time.earliest_var;
      closures = [];
      static_closures = []
    }

  (** val binding_time_resolver : name -> Binding_time.With_name_mode.t **)

  let binding_time_resolver = function
    | Name_var _ -> Binding_time.With_name_mode.imported_variables
    | Name_symbol _ -> Binding_time.With_name_mode.symbols

  (** val canonical_simple : t -> simple -> simple **)

  let canonical_simple known s = canonical_element_of known.aliases s

  (** val add_equal : (name, unit * Binding_time.With_name_mode.t)
      Keyed.finite_map -> Aliases.t -> simple -> simple -> Aliases.t **)

  let add_equal binding_times0 aliases0 e e' =
    let canonical_element1 = canonical_element_of aliases0 e in
    let canonical_element2 = canonical_element_of aliases0 e' in
    if simple_equal canonical_element1 canonical_element2
    then aliases0
    else
      match
        Aliases.add binding_time_resolver binding_times0 aliases0
          canonical_element1 canonical_element2
      with
      | Some t0 -> (
        match t0 with
        | Or_bottom.Ok result -> result.Aliases.add_result_t
        | Or_bottom.Bottom -> aliases0)
      | None -> aliases0

  (** val forget_in_aliases : (name, unit * Binding_time.With_name_mode.t)
      Keyed.finite_map -> Aliases.t -> variable -> Aliases.t **)

  let forget_in_aliases binding_times0 aliases0 x =
    if negb (among x (aliases_free_variables aliases0))
    then aliases0
    else
      let demoted_to_x =
       fun n ->
        simple_equal
          (Aliases.get_canonical_ignoring_name_mode aliases0 n)
          (Var x)
      in
      let demoted =
        filter
          (fun n -> negb (name_eqb n (Name_var x)))
          (Keyed.domain name_eqb aliases0.Aliases.canonical_elements)
      in
      let in_place_of_x = find demoted_to_x demoted in
      fold_left
        (fun rebuilt n ->
          if demoted_to_x n
          then
            match in_place_of_x with
            | Some n' ->
              add_equal binding_times0 rebuilt (simple_of_name n)
                (simple_of_name n')
            | None -> rebuilt
          else
            add_equal binding_times0 rebuilt (simple_of_name n)
              (Aliases.get_canonical_ignoring_name_mode aliases0 n))
        demoted Aliases.empty

  (** val forget : t -> variable -> t **)

  let forget known x =
    { aliases = forget_in_aliases known.binding_times known.aliases x;
      binding_times = known.binding_times;
      next_binding_time = known.next_binding_time;
      closures =
        filter
          (fun entry ->
            negb (Pos.eqb x (fst entry))
            && negb (var_in_value_slots x (closure_value_slots (snd entry))))
          known.closures;
      static_closures =
        filter
          (fun entry ->
            negb
              (var_in_value_slots x
                 (closure_value_slots (snd entry).as_closure)))
          known.static_closures
    }

  (** val bind : t -> variable -> t **)

  let bind known x =
    let known0 = forget known x in
    { aliases = known0.aliases;
      binding_times =
        ( Name_var x,
          ( (),
            { Binding_time.With_name_mode.binding_time =
                known0.next_binding_time;
              Binding_time.With_name_mode.name_mode = Name_mode.normal
            } ) )
        :: known0.binding_times;
      next_binding_time = Binding_time.succ known0.next_binding_time;
      closures = known0.closures;
      static_closures = known0.static_closures
    }

  (** val bind_all : t -> variable list -> t **)

  let bind_all known xs = fold_left bind xs known

  (** val with_aliases : t -> Aliases.t -> t **)

  let with_aliases known aliases0 =
    { aliases = aliases0;
      binding_times = known.binding_times;
      next_binding_time = known.next_binding_time;
      closures = known.closures;
      static_closures = known.static_closures
    }

  (** val add_closure : t -> variable -> closure -> t **)

  let add_closure known x c =
    { aliases = known.aliases;
      binding_times = known.binding_times;
      next_binding_time = known.next_binding_time;
      closures = (x, c) :: known.closures;
      static_closures = known.static_closures
    }

  (** val find_closure : t -> simple -> closure option **)

  let find_closure known = function
    | Var x -> lookup x known.closures
    | Symbol sym ->
      option_map (fun s0 -> s0.as_closure) (lookup sym known.static_closures)
    | Const _ -> None

  (** val under_let : t -> variable -> named -> t **)

  let under_let known x d =
    let known0 = bind known x in
    match d with
    | Simple s ->
      if var_in_simple x s
      then known0
      else
        with_aliases known0
          (add_equal known0.binding_times known0.aliases (Var x) s)
    | Set_of_closures (code, value_slots) ->
      if var_in_value_slots x value_slots
      then known0
      else add_closure known0 x (Of_code (code, value_slots))
    | Closure_of_set (slot, function_decls, value_slots) ->
      if var_in_value_slots x value_slots
      then known0
      else add_closure known0 x (Of_set (slot, function_decls, value_slots))
    | Project_function_slot (move_from, move_to, closure0) -> (
      match closure0 with
      | Var y -> (
        match lookup y known0.closures with
        | Some c -> (
          match c with
          | Of_code (_, _) -> known0
          | Of_set (slot, function_decls, value_slots) ->
            if Pos.eqb slot move_from
            then
              add_closure known0 x
                (Of_set (move_to, function_decls, value_slots))
            else known0)
        | None -> known0)
      | _ -> known0)
    | _ -> known0

  (** val static_closure_of_binding : static_binding -> (symbol *
      static_closure) option **)

  let static_closure_of_binding = function
    | Bind_symbol (s, const) -> (
      match const with
      | Static_const_set_of_closures (code, value_slots) ->
        Some
          (s, { as_closure = Of_code (code, value_slots); closure_symbols = [] })
      | Static_const_closure_of_set
          (slot, closure_symbols0, function_decls, value_slots) ->
        Some
          ( s,
            { as_closure = Of_set (slot, function_decls, value_slots);
              closure_symbols = closure_symbols0
            } )
      | _ -> None)
    | _ -> None

  (** val under_let_static : t -> static_binding list -> t **)

  let under_let_static known bindings =
    { aliases = known.aliases;
      binding_times = known.binding_times;
      next_binding_time = known.next_binding_time;
      closures = known.closures;
      static_closures =
        app
          (flat_map
             (fun binding ->
               match static_closure_of_binding binding with
               | Some entry -> entry :: []
               | None -> [])
             bindings)
          known.static_closures
    }
end

(** val canonical_value_slots : Known.t -> (value_slot * simple) list ->
    (value_slot * simple) list **)

let canonical_value_slots known slots =
  map (fun slot -> fst slot, Known.canonical_simple known (snd slot)) slots

(** val canonical_project : Known.t -> value_slot -> simple -> named **)

let canonical_project known slot closure0 =
  match Known.find_closure known closure0 with
  | Some c -> (
    match lookup slot (Known.closure_value_slots c) with
    | Some contents -> Simple contents
    | None -> Project_value_slot (slot, closure0))
  | None -> Project_value_slot (slot, closure0)

(** val canonical_project_from : Known.t -> function_slot -> value_slot ->
    simple -> named **)

let canonical_project_from known project_from slot closure0 =
  match Known.find_closure known closure0 with
  | Some c ->
    if Pos.eqb (Known.closure_slot c) project_from
    then
      match lookup slot (Known.closure_value_slots c) with
      | Some contents -> Simple contents
      | None -> Project_value_slot_from (project_from, slot, closure0)
    else Project_value_slot_from (project_from, slot, closure0)
  | None -> Project_value_slot_from (project_from, slot, closure0)

(** val canonical_project_function_slot : Known.t -> function_slot ->
    function_slot -> simple -> named **)

let canonical_project_function_slot known move_from move_to closure0 =
  match closure0 with
  | Symbol s -> (
    match lookup s known.Known.static_closures with
    | Some c -> (
      match c.Known.as_closure with
      | Known.Of_code (_, _) ->
        Project_function_slot (move_from, move_to, closure0)
      | Known.Of_set (slot, function_decls, _) ->
        if Pos.eqb slot move_from && is_bound move_to function_decls
        then
          match lookup move_to c.Known.closure_symbols with
          | Some s' -> Simple (Symbol s')
          | None -> Project_function_slot (move_from, move_to, closure0)
        else Project_function_slot (move_from, move_to, closure0))
    | None -> Project_function_slot (move_from, move_to, closure0))
  | _ -> Project_function_slot (move_from, move_to, closure0)

(** val canonical_named : Known.t -> named -> named **)

let canonical_named known = function
  | Simple s -> Simple (Known.canonical_simple known s)
  | Prim (p, args) -> Prim (p, map (Known.canonical_simple known) args)
  | Set_of_closures (code, slots) ->
    Set_of_closures (code, canonical_value_slots known slots)
  | Closure_of_set (slot, decls, slots) ->
    Closure_of_set (slot, decls, canonical_value_slots known slots)
  | Project_value_slot (slot, closure0) ->
    canonical_project known slot (Known.canonical_simple known closure0)
  | Project_value_slot_from (project_from, slot, closure0) ->
    canonical_project_from known project_from slot
      (Known.canonical_simple known closure0)
  | Project_function_slot (move_from, move_to, closure0) ->
    canonical_project_function_slot known move_from move_to
      (Known.canonical_simple known closure0)

(** val canonical_static_const : Known.t -> static_const -> static_const **)

let canonical_static_const known = function
  | Static_const_set_of_closures (code, slots) ->
    Static_const_set_of_closures (code, canonical_value_slots known slots)
  | Static_const_closure_of_set (slot, symbols0, decls, slots) ->
    Static_const_closure_of_set
      (slot, symbols0, decls, canonical_value_slots known slots)
  | Static_const_block (tag, fields) ->
    Static_const_block (tag, map (Known.canonical_simple known) fields)
  | Static_const_boxed_number contents ->
    Static_const_boxed_number (Known.canonical_simple known contents)
  | Static_const_immutable_string bytes -> Static_const_immutable_string bytes
  | Static_const_immutable_value_array elements ->
    Static_const_immutable_value_array
      (map (Known.canonical_simple known) elements)

(** val canonical_static_binding : Known.t -> static_binding -> static_binding **)

let canonical_static_binding known = function
  | Bind_symbol (s, const) -> Bind_symbol (s, canonical_static_const known const)
  | x -> x

(** val canonical_apply_cont : Known.t -> apply_cont -> apply_cont **)

let canonical_apply_cont known ac =
  { ac_continuation = ac.ac_continuation;
    ac_args = map (Known.canonical_simple known) ac.ac_args;
    ac_trap_action = ac.ac_trap_action
  }

(** val canonical_call_kind : Known.t -> simple -> call_kind -> call_kind **)

let canonical_call_kind known callee kind =
  match kind with
  | Indirect_unknown_arity -> (
    match Known.find_closure known callee with
    | Some c -> (
      match Known.closure_code c with
      | Some code -> Direct code
      | None -> Indirect_unknown_arity)
    | None -> Indirect_unknown_arity)
  | _ -> kind

(** val canonical_apply : Known.t -> apply_expr -> apply_expr **)

let canonical_apply known ap =
  let callee = Known.canonical_simple known ap.ap_callee in
  { ap_callee = callee;
    ap_call_kind = canonical_call_kind known callee ap.ap_call_kind;
    ap_args = map (Known.canonical_simple known) ap.ap_args;
    ap_continuation = ap.ap_continuation;
    ap_exn_continuation = ap.ap_exn_continuation;
    ap_exn_extra_args =
      map
        (fun arg -> Known.canonical_simple known (fst arg), snd arg)
        ap.ap_exn_extra_args;
    ap_args_arity = ap.ap_args_arity;
    ap_return_arity = ap.ap_return_arity
  }

(** val follow_aliases : Known.t -> expr -> expr **)

let rec follow_aliases known = function
  | Let (x, kind, d, body) ->
    let d' = canonical_named known d in
    Let (x, kind, d', follow_aliases (Known.under_let known x d') body)
  | Let_static (bindings, body) ->
    let bindings' = map (canonical_static_binding known) bindings in
    Let_static
      (bindings', follow_aliases (Known.under_let_static known bindings') body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, follow_aliases_in_handler known h, follow_aliases known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, follow_aliases known body)
  | Apply ap -> Apply (canonical_apply known ap)
  | Apply_cont ac -> Apply_cont (canonical_apply_cont known ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( Known.canonical_simple known scrutinee,
        map (fun arm -> fst arm, canonical_apply_cont known (snd arm)) arms )
  | Invalid -> Invalid

(** val follow_aliases_in_handler : Known.t -> continuation_handler ->
    continuation_handler **)

and follow_aliases_in_handler known = function
  | Handler (params, metadata, handler) ->
    Handler
      (params, metadata, follow_aliases (Known.bind_all known params) handler)
