open BinNums
open BinPos
open Datatypes
open FlambdaBindersSpec
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveSpec
open List
open ListDef
open OxCamlList

(** val free_continuations : expr -> continuation list **)

let rec free_continuations = function
  | Let (_, _, _, body) -> free_continuations body
  | Let_static (_, body) -> free_continuations body
  | Let_cont_non_recursive (k, h, body) ->
    app
      (handler_free_continuations h)
      (without (k :: []) (free_continuations body))
  | Let_cont_recursive (handlers, body) ->
    without (map fst handlers)
      (app
         (flat_map
            (fun pat ->
              let _, h = pat in
              handler_free_continuations h)
            handlers)
         (free_continuations body))
  | Apply ap -> [ap.ap_continuation; ap.ap_exn_continuation]
  | Apply_cont ac -> apply_cont_continuations ac
  | Switch (_, arms) ->
    flat_map (fun arm -> apply_cont_continuations (snd arm)) arms
  | Invalid -> []

(** val handler_free_continuations : continuation_handler -> continuation list **)

and handler_free_continuations = function
  | Handler (_, _, handler) -> free_continuations handler

(** val traps : continuation -> apply_cont -> bool **)

let traps k ac =
  match ac.ac_trap_action with
  | Some action -> Pos.eqb k (trap_action_handler action)
  | None -> false

(** val raises_to : continuation -> expr -> bool **)

let rec raises_to k = function
  | Let (_, _, _, body) -> raises_to k body
  | Let_static (_, body) -> raises_to k body
  | Let_cont_non_recursive (_, h, body) ->
    handler_raises_to k h || raises_to k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_raises_to k h)
      handlers
    || raises_to k body
  | Apply ap -> Pos.eqb k ap.ap_exn_continuation
  | Apply_cont ac -> traps k ac
  | Switch (_, arms) -> existsb (fun arm -> traps k (snd arm)) arms
  | Invalid -> false

(** val handler_raises_to : continuation -> continuation_handler -> bool **)

and handler_raises_to k = function
  | Handler (_, _, handler) -> raises_to k handler

(** val handler_at_unit_toplevel : continuation -> continuation ->
    continuation_handler -> expr -> bool **)

let handler_at_unit_toplevel unit_exn0 k h body =
  negb (handler_is_exn_handler h)
  && all_among (free_continuations body) [k; unit_exn0]

(** val statics_at_unit_toplevel : continuation -> bool -> expr -> bool **)

let rec statics_at_unit_toplevel unit_exn0 toplevel = function
  | Let (_, _, _, body) -> statics_at_unit_toplevel unit_exn0 toplevel body
  | Let_static (bindings, body) ->
    (toplevel && forallb (binding_statics_at_unit_toplevel unit_exn0) bindings)
    && statics_at_unit_toplevel unit_exn0 toplevel body
  | Let_cont_non_recursive (k, h, body) ->
    handler_statics_at_unit_toplevel unit_exn0
      (toplevel && handler_at_unit_toplevel unit_exn0 k h body)
      h
    && statics_at_unit_toplevel unit_exn0 toplevel body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_statics_at_unit_toplevel unit_exn0 false h)
      handlers
    && statics_at_unit_toplevel unit_exn0 toplevel body
  | _ -> true

(** val handler_statics_at_unit_toplevel : continuation -> bool ->
    continuation_handler -> bool **)

and handler_statics_at_unit_toplevel unit_exn0 toplevel = function
  | Handler (_, _, handler) ->
    statics_at_unit_toplevel unit_exn0 toplevel handler

(** val binding_statics_at_unit_toplevel : continuation -> static_binding ->
    bool **)

and binding_statics_at_unit_toplevel unit_exn0 = function
  | Bind_code (_, c) -> code_statics_at_unit_toplevel unit_exn0 c
  | _ -> true

(** val code_statics_at_unit_toplevel : continuation -> code -> bool **)

and code_statics_at_unit_toplevel unit_exn0 = function
  | Code (_, _, _, _, _, body) -> statics_at_unit_toplevel unit_exn0 false body

(** val code_ids_of_binding : static_binding -> code_id list **)

let code_ids_of_binding = function Bind_code (id, _) -> id :: [] | _ -> []

(** val symbols_of_binding : static_binding -> symbol list **)

let symbols_of_binding = function Bind_symbol (s, _) -> s :: [] | _ -> []

(** val static_names : (static_binding -> positive list) -> expr -> positive
    list **)

let rec static_names names_of = function
  | Let (_, _, _, body) -> static_names names_of body
  | Let_static (bindings, body) ->
    app (flat_map names_of bindings) (static_names names_of body)
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_static_names names_of h) (static_names names_of body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_static_names names_of h)
         handlers)
      (static_names names_of body)
  | _ -> []

(** val handler_static_names : (static_binding -> positive list) ->
    continuation_handler -> positive list **)

and handler_static_names names_of = function
  | Handler (_, _, handler) -> static_names names_of handler

(** val set_well_formed : function_slot -> (function_slot * code_id) list ->
    bool **)

let set_well_formed slot function_decls =
  (is_bound slot function_decls && distinct (map fst function_decls))
  && distinct (map snd function_decls)

(** val named_sets_well_formed : named -> bool **)

let named_sets_well_formed = function
  | Closure_of_set (slot, function_decls, _) ->
    set_well_formed slot function_decls
  | _ -> true

(** val binds_symbol_to : symbol -> static_const -> static_binding -> bool **)

let binds_symbol_to s const = function
  | Bind_symbol (s', const') ->
    Pos.eqb s s' && if static_const_eq_dec const const' then true else false
  | _ -> false

(** val static_set_whole : static_binding list -> static_binding -> bool **)

let static_set_whole bindings = function
  | Bind_symbol (s, const) -> (
    match const with
    | Static_const_closure_of_set
        (slot, closure_symbols, function_decls, value_slots) ->
      ((set_well_formed slot function_decls
       &&
       if
         list_eq_dec Pos.eq_dec (map fst closure_symbols)
           (map fst function_decls)
       then true
       else false)
      &&
      match lookup slot closure_symbols with
      | Some s' -> Pos.eqb s s'
      | None -> false)
      && forallb
           (fun pat ->
             let slot', s' = pat in
             existsb
               (binds_symbol_to s'
                  (Static_const_closure_of_set
                     (slot', closure_symbols, function_decls, value_slots)))
               bindings)
           closure_symbols
    | _ -> true)
  | _ -> true

(** val static_set_code_ids : static_binding -> code_id list **)

let static_set_code_ids = function
  | Bind_symbol (_, const) -> (
    match const with
    | Static_const_set_of_closures (code0, _) -> code0 :: []
    | Static_const_closure_of_set (_, _, function_decls, _) ->
      map snd function_decls
    | _ -> [])
  | _ -> []

(** val binds_symbol_of_set : static_binding -> static_binding -> bool **)

let binds_symbol_of_set binding binding' =
  match binding with
  | Bind_symbol (_, const) -> (
    match const with
    | Static_const_closure_of_set (_, closure_symbols, _, _) -> (
      match binding' with
      | Bind_symbol (s', _) -> among s' (map snd closure_symbols)
      | _ -> false)
    | _ -> false)
  | _ -> false

(** val static_sets_code_disjoint : static_binding list -> bool **)

let rec static_sets_code_disjoint = function
  | [] -> true
  | binding :: bindings0 ->
    forallb
      (fun binding' ->
        binds_symbol_of_set binding binding'
        ||
        match
          inter (static_set_code_ids binding) (static_set_code_ids binding')
        with
        | [] -> true
        | _ :: _ -> false)
      bindings0
    && static_sets_code_disjoint bindings0

(** val sets_well_formed : expr -> bool **)

let rec sets_well_formed = function
  | Let (_, _, d, body) -> named_sets_well_formed d && sets_well_formed body
  | Let_static (bindings, body) ->
    ((forallb (static_set_whole bindings) bindings
     && static_sets_code_disjoint bindings)
    && forallb binding_sets_well_formed bindings)
    && sets_well_formed body
  | Let_cont_non_recursive (_, h, body) ->
    handler_sets_well_formed h && sets_well_formed body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_sets_well_formed h)
      handlers
    && sets_well_formed body
  | _ -> true

(** val handler_sets_well_formed : continuation_handler -> bool **)

and handler_sets_well_formed = function
  | Handler (_, _, handler) -> sets_well_formed handler

(** val binding_sets_well_formed : static_binding -> bool **)

and binding_sets_well_formed = function
  | Bind_code (_, c) -> code_sets_well_formed c
  | _ -> true

(** val code_sets_well_formed : code -> bool **)

and code_sets_well_formed = function
  | Code (_, _, _, _, _, body) -> sets_well_formed body

(** val simple_in_range : simple -> bool **)

let simple_in_range = function Const c -> const_in_range c | _ -> true

(** val value_slots_in_range : (value_slot * simple) list -> bool **)

let value_slots_in_range value_slots =
  forallb (fun slot -> simple_in_range (snd slot)) value_slots

(** val named_in_range : named -> bool **)

let named_in_range = function
  | Simple s -> simple_in_range s
  | Prim (p, args) -> prim_in_range p && forallb simple_in_range args
  | Set_of_closures (_, value_slots) -> value_slots_in_range value_slots
  | Closure_of_set (_, _, value_slots) -> value_slots_in_range value_slots
  | Project_value_slot (_, closure) -> simple_in_range closure
  | Project_value_slot_from (_, _, closure) -> simple_in_range closure
  | Project_function_slot (_, _, closure) -> simple_in_range closure

(** val static_const_in_range : static_const -> bool **)

let static_const_in_range = function
  | Static_const_set_of_closures (_, value_slots) ->
    value_slots_in_range value_slots
  | Static_const_closure_of_set (_, _, _, value_slots) ->
    value_slots_in_range value_slots
  | Static_const_block (tag, fields) ->
    tag_in_range tag && forallb simple_in_range fields
  | Static_const_boxed_number contents -> simple_in_range contents
  | Static_const_immutable_string bytes -> forallb byte_in_range bytes
  | Static_const_immutable_value_array elements ->
    forallb simple_in_range elements

(** val apply_cont_in_range : apply_cont -> bool **)

let apply_cont_in_range ac = forallb simple_in_range ac.ac_args

(** val apply_in_range : apply_expr -> bool **)

let apply_in_range ap =
  (simple_in_range ap.ap_callee && forallb simple_in_range ap.ap_args)
  && forallb simple_in_range (exn_extra_simples ap)

(** val arm_in_range : (coq_Z * apply_cont) -> bool **)

let arm_in_range arm =
  ocaml_int_in_range (fst arm) && apply_cont_in_range (snd arm)

(** val numbers_in_range : expr -> bool **)

let rec numbers_in_range = function
  | Let (_, _, d, body) -> named_in_range d && numbers_in_range body
  | Let_static (bindings, body) ->
    forallb binding_numbers_in_range bindings && numbers_in_range body
  | Let_cont_non_recursive (_, h, body) ->
    handler_numbers_in_range h && numbers_in_range body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_numbers_in_range h)
      handlers
    && numbers_in_range body
  | Apply ap -> apply_in_range ap
  | Apply_cont ac -> apply_cont_in_range ac
  | Switch (scrutinee, arms) ->
    simple_in_range scrutinee && forallb arm_in_range arms
  | Invalid -> true

(** val handler_numbers_in_range : continuation_handler -> bool **)

and handler_numbers_in_range = function
  | Handler (_, _, handler) -> numbers_in_range handler

(** val binding_numbers_in_range : static_binding -> bool **)

and binding_numbers_in_range = function
  | Bind_code (_, c) -> code_numbers_in_range c
  | Bind_deleted_code _ -> true
  | Bind_symbol (_, const) -> static_const_in_range const

(** val code_numbers_in_range : code -> bool **)

and code_numbers_in_range = function
  | Code (_, _, _, _, _, body) -> numbers_in_range body

(** val applies : continuation -> apply_cont -> bool **)

let applies k ac = Pos.eqb k ac.ac_continuation && negb (is_raise ac)

(** val applies_cont : continuation -> expr -> bool **)

let rec applies_cont k = function
  | Let (_, _, _, body) -> applies_cont k body
  | Let_static (_, body) -> applies_cont k body
  | Let_cont_non_recursive (_, h, body) ->
    handler_applies_cont k h || applies_cont k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_applies_cont k h)
      handlers
    || applies_cont k body
  | Apply ap -> Pos.eqb k ap.ap_continuation
  | Apply_cont ac -> applies k ac
  | Switch (_, arms) -> existsb (fun arm -> applies k (snd arm)) arms
  | Invalid -> false

(** val handler_applies_cont : continuation -> continuation_handler -> bool **)

and handler_applies_cont k = function
  | Handler (_, _, handler) -> applies_cont k handler

(** val gives_extra_args : continuation -> expr -> bool **)

let rec gives_extra_args k = function
  | Let (_, _, _, body) -> gives_extra_args k body
  | Let_static (_, body) -> gives_extra_args k body
  | Let_cont_non_recursive (_, h, body) ->
    handler_gives_extra_args k h || gives_extra_args k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_gives_extra_args k h)
      handlers
    || gives_extra_args k body
  | Apply ap -> Pos.eqb k ap.ap_exn_continuation && negb (no_exn_extra_args ap)
  | _ -> false

(** val handler_gives_extra_args : continuation -> continuation_handler -> bool
    **)

and handler_gives_extra_args k = function
  | Handler (_, _, handler) -> gives_extra_args k handler

(** val traps_apart : continuation -> apply_cont -> bool **)

let traps_apart k ac = traps k ac && negb (is_raise ac)

(** val installs_or_removes : continuation -> expr -> bool **)

let rec installs_or_removes k = function
  | Let (_, _, _, body) -> installs_or_removes k body
  | Let_static (_, body) -> installs_or_removes k body
  | Let_cont_non_recursive (_, h, body) ->
    handler_installs_or_removes k h || installs_or_removes k body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        handler_installs_or_removes k h)
      handlers
    || installs_or_removes k body
  | Apply_cont ac -> traps_apart k ac
  | Switch (_, arms) -> existsb (fun arm -> traps_apart k (snd arm)) arms
  | _ -> false

(** val handler_installs_or_removes : continuation -> continuation_handler ->
    bool **)

and handler_installs_or_removes k = function
  | Handler (_, _, handler) -> installs_or_removes k handler

(** val ways_out_well_formed : continuation -> expr -> bool **)

let ways_out_well_formed exn e =
  forallb
    (fun k ->
      (if Pos.eqb k exn
       then negb (installs_or_removes k e)
       else negb (raises_to k e))
      && negb (gives_extra_args k e))
    (free_continuations e)

(** val takes_exception : continuation_handler -> bool **)

let takes_exception = function
  | Handler (params, _, _) -> (
    match params with [] -> false | _ :: _ -> true)

(** val exn_handlers_well_formed : expr -> bool **)

let rec exn_handlers_well_formed = function
  | Let (_, _, _, body) -> exn_handlers_well_formed body
  | Let_static (bindings, body) ->
    forallb binding_exn_handlers_well_formed bindings
    && exn_handlers_well_formed body
  | Let_cont_non_recursive (k, h, body) ->
    ((if handler_is_exn_handler h
      then negb (applies_cont k body) && takes_exception h
      else negb (raises_to k body))
    && handler_exn_handlers_well_formed h)
    && exn_handlers_well_formed body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        ((negb (handler_is_exn_handler h) && negb (raises_to k body))
        && negb
             (existsb
                (fun pat0 ->
                  let _, h' = pat0 in
                  handler_raises_to k h')
                handlers))
        && handler_exn_handlers_well_formed h)
      handlers
    && exn_handlers_well_formed body
  | _ -> true

(** val handler_exn_handlers_well_formed : continuation_handler -> bool **)

and handler_exn_handlers_well_formed = function
  | Handler (_, _, handler) -> exn_handlers_well_formed handler

(** val binding_exn_handlers_well_formed : static_binding -> bool **)

and binding_exn_handlers_well_formed = function
  | Bind_code (_, c) -> code_exn_handlers_well_formed c
  | _ -> true

(** val code_exn_handlers_well_formed : code -> bool **)

and code_exn_handlers_well_formed = function
  | Code (_, _, _, exn, _, body) ->
    (ways_out_well_formed exn body && negb (applies_cont exn body))
    && exn_handlers_well_formed body

(** val unit_exn_handlers_well_formed : continuation -> expr -> bool **)

let unit_exn_handlers_well_formed unit_exn0 e =
  (ways_out_well_formed unit_exn0 e && negb (applies_cont unit_exn0 e))
  && exn_handlers_well_formed e

type trap_stack = continuation list

(** val trap_stack_eqb : trap_stack -> trap_stack -> bool **)

let rec trap_stack_eqb ts ts' =
  match ts with
  | [] -> ( match ts' with [] -> true | _ :: _ -> false)
  | k :: ts0 -> (
    match ts' with
    | [] -> false
    | k' :: ts'0 -> Pos.eqb k k' && trap_stack_eqb ts0 ts'0)

type demands = (continuation * trap_stack) list

type entry =
  | Not_entered
  | Entered_with of trap_stack
  | Entered_with_two_stacks

(** val entry_of : continuation -> demands -> entry **)

let entry_of k ds =
  match map snd (filter (fun d -> Pos.eqb (fst d) k) ds) with
  | [] -> Not_entered
  | ts :: others ->
    if forallb (trap_stack_eqb ts) others
    then Entered_with ts
    else Entered_with_two_stacks

(** val leave_scope : continuation list -> demands -> demands option **)

let leave_scope ks ds =
  let outer = filter (fun d -> negb (among (fst d) ks)) ds in
  if forallb (fun d -> forallb (fun k -> negb (among k (snd d))) ks) outer
  then Some outer
  else None

(** val apply_cont_demands : trap_stack -> apply_cont -> demands option **)

let apply_cont_demands ts ac =
  match ac.ac_trap_action with
  | Some t -> (
    match t with
    | Push k -> Some [k, ts; ac.ac_continuation, k :: ts]
    | Pop (k, _) -> (
      match ts with
      | [] -> None
      | top :: rest ->
        if Pos.eqb top k then Some ((ac.ac_continuation, rest) :: []) else None)
    )
  | None -> Some ((ac.ac_continuation, ts) :: [])

(** val apply_demands : trap_stack -> apply_expr -> demands option **)

let apply_demands ts ap =
  match ts with
  | [] -> None
  | top :: rest ->
    if Pos.eqb top ap.ap_exn_continuation
    then Some [ap.ap_continuation, ts; ap.ap_exn_continuation, rest]
    else None

(** val arms_demands : trap_stack -> (coq_Z * apply_cont) list -> demands option
    **)

let rec arms_demands ts = function
  | [] -> Some []
  | p :: arms0 -> (
    let _, ac = p in
    match apply_cont_demands ts ac with
    | Some ds -> (
      match arms_demands ts arms0 with
      | Some ds' -> Some (app ds ds')
      | None -> None)
    | None -> None)

(** val code_demands_met : continuation -> continuation -> demands -> bool **)

let code_demands_met ret exn ds =
  forallb
    (fun d ->
      if Pos.eqb (fst d) exn
      then trap_stack_eqb (snd d) []
      else Pos.eqb (fst d) ret && trap_stack_eqb (snd d) (exn :: []))
    ds

(** val iterate : nat -> ('a1 -> 'a1 option) -> 'a1 -> 'a1 option **)

let rec iterate n f x =
  match n with
  | O -> Some x
  | S n0 -> ( match f x with Some x0 -> iterate n0 f x0 | None -> None)

(** val trap_demands : trap_stack -> expr -> demands option **)

let rec trap_demands ts = function
  | Let (_, _, _, body) -> trap_demands ts body
  | Let_static (bindings, body) ->
    if forallb binding_traps_well_formed bindings
    then trap_demands ts body
    else None
  | Let_cont_non_recursive (k, h, body) -> (
    if among k ts
    then None
    else
      match trap_demands ts body with
      | Some ds -> (
        match leave_scope (k :: []) ds with
        | Some outer -> (
          match entry_of k ds with
          | Not_entered -> Some outer
          | Entered_with entered -> (
            if among k entered
            then None
            else
              match handler_trap_demands entered h with
              | Some ds' -> Some (app outer ds')
              | None -> None)
          | Entered_with_two_stacks -> None)
        | None -> None)
      | None -> None)
  | Let_cont_recursive (handlers, body) -> (
    let ks = map fst handlers in
    if existsb (fun k -> among k ts) ks
    then None
    else
      match trap_demands ts body with
      | Some ds -> (
        let pass =
         fun state ->
          fold_left
            (fun state0 pat ->
              let k, h = pat in
              match state0 with
              | Some y -> (
                let checked, ds0 = y in
                if among k checked
                then Some (checked, ds0)
                else
                  match entry_of k ds0 with
                  | Not_entered -> Some (checked, ds0)
                  | Entered_with entered -> (
                    match handler_trap_demands entered h with
                    | Some ds' -> Some (k :: checked, app ds0 ds')
                    | None -> None)
                  | Entered_with_two_stacks -> None)
              | None -> None)
            (Some state) handlers
        in
        match iterate (length handlers) pass ([], ds) with
        | Some g ->
          let checked, ds0 = g in
          if
            forallb
              (fun k ->
                match entry_of k ds0 with
                | Not_entered -> true
                | Entered_with _ -> among k checked
                | Entered_with_two_stacks -> false)
              ks
          then leave_scope ks ds0
          else None
        | None -> None)
      | None -> None)
  | Apply ap -> apply_demands ts ap
  | Apply_cont ac -> apply_cont_demands ts ac
  | Switch (_, arms) -> arms_demands ts arms
  | Invalid -> Some []

(** val handler_trap_demands : trap_stack -> continuation_handler -> demands
    option **)

and handler_trap_demands ts = function
  | Handler (_, _, handler) -> trap_demands ts handler

(** val binding_traps_well_formed : static_binding -> bool **)

and binding_traps_well_formed = function
  | Bind_code (_, c) -> code_traps_well_formed c
  | _ -> true

(** val code_traps_well_formed : code -> bool **)

and code_traps_well_formed = function
  | Code (_, _, ret, exn, _, body) -> (
    match trap_demands (exn :: []) body with
    | Some ds -> code_demands_met ret exn ds
    | None -> false)

(** val traps_well_formed : continuation -> expr -> bool **)

let traps_well_formed unit_exn0 e =
  match trap_demands (unit_exn0 :: []) e with
  | Some ds ->
    forallb
      (fun d ->
        trap_stack_eqb (snd d)
          (if Pos.eqb (fst d) unit_exn0 then [] else unit_exn0 :: []))
      ds
  | None -> false

type unit_interface =
  { unit_return : continuation;
    unit_exn : continuation
  }

(** val well_formed_unit : unit_interface -> expr -> bool **)

let well_formed_unit u e =
  ((((((((statics_at_unit_toplevel u.unit_exn true e
         && distinct (static_names code_ids_of_binding e))
        && distinct (static_names symbols_of_binding e))
       && names_in_scope u.unit_return u.unit_exn e)
      && binders_distinct e)
     && code_closed e)
    && sets_well_formed e)
   && numbers_in_range e)
  && unit_exn_handlers_well_formed u.unit_exn e)
  && traps_well_formed u.unit_exn e
