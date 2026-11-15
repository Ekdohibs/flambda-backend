open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaRegionsCheckSpec
open FlambdaRegionsSpec
open FlambdaValueSpec
open List
open ListDef
open Nat

type ends_options = { eo_own_end : bool }

type ends_cont =
  | Ends_handler of variable list * flambda_kind list * handler_entry * bool
  | Ends_unreachable
  | Ends_return
  | Ends_raise

type ends_state =
  { es_open : variable list;
    es_tries : variable list;
    es_regions : variable list;
    es_stale : bool;
    es_conts : (continuation * ends_cont) list
  }

(** val regions_above : variable -> variable list -> variable list **)

let rec regions_above r = function
  | [] -> []
  | x :: open1 -> if Pos.eqb x r then [] else x :: regions_above r open1

(** val variables_eqb : variable list -> variable list -> bool **)

let rec variables_eqb xs ys =
  match xs with
  | [] -> ( match ys with [] -> true | _ :: _ -> false)
  | x :: xs0 -> (
    match ys with
    | [] -> false
    | y :: ys0 -> Pos.eqb x y && variables_eqb xs0 ys0)

(** val oldest_of : variable list -> variable list -> bool **)

let oldest_of xs ys = variables_eqb xs (skipn (sub (length ys) (length xs)) ys)

(** val ends_region_args : variable list -> variable list -> flambda_kind list
    -> simple list option -> (variable * variable) list option **)

let rec ends_region_args open0 params kinds args =
  match params with
  | [] -> (
    match kinds with
    | [] -> (
      match args with
      | Some l -> ( match l with [] -> Some [] | _ :: _ -> None)
      | None -> Some [])
    | _ :: _ -> None)
  | p :: params0 -> (
    match kinds with
    | [] -> None
    | kind :: kinds0 -> (
      let rest_args =
        match args with
        | Some l -> (
          match l with [] -> None | _ :: args0 -> Some (Some args0))
        | None -> Some None
      in
      match rest_args with
      | Some rest_args0 -> (
        match ends_region_args open0 params0 kinds0 rest_args0 with
        | Some rest ->
          if kind_eqb kind Region
          then
            match args with
            | Some l -> (
              match l with
              | [] -> None
              | s :: _ -> (
                match s with
                | Var r -> if among r open0 then Some ((p, r) :: rest) else None
                | _ -> None))
            | None -> None
          else Some rest
        | None -> None)
      | None -> None))

(** val entry_at_jump : variable list -> variable list -> flambda_kind list ->
    handler_entry -> simple list option -> variable list option **)

let entry_at_jump open0 params kinds entry args =
  match ends_region_args open0 params kinds args with
  | Some given -> Some (map (region_at_jump given) entry.he_open)
  | None -> None

(** val ends_jump : ends_state -> continuation -> simple list option -> bool **)

let ends_jump es k args =
  match lookup k es.es_conts with
  | Some e -> (
    match e with
    | Ends_handler (params, kinds, entry, stale) -> (
      match entry_at_jump es.es_open params kinds entry args with
      | Some named ->
        (stale && oldest_of named es.es_open)
        || (negb es.es_stale && in_order named es.es_open)
           && forallb (fun r -> among r es.es_tries || among r named) es.es_open
      | None -> false)
    | Ends_unreachable -> false
    | Ends_return ->
      negb es.es_stale && forallb (fun r -> among r es.es_tries) es.es_open
    | Ends_raise -> true)
  | None -> false

(** val ends_raise_to : ends_state -> continuation -> simple list -> bool **)

let ends_raise_to es k extra =
  match lookup k es.es_conts with
  | Some e -> (
    match e with
    | Ends_handler (params, kinds, entry, stale) -> (
      stale
      &&
      match
        entry_at_jump es.es_open params kinds entry
          (Some (unit_simple :: extra))
      with
      | Some named -> oldest_of named es.es_open
      | None -> false)
    | Ends_raise -> true
    | _ -> false)
  | None -> false

(** val ends_first : expr -> bool **)

let rec ends_first = function
  | Let (_, _, defining_expr, body) -> (
    match defining_expr with
    | Prim (p0, args) -> (
      match p0 with
      | Region_primitive p -> (
        match p with
        | End_region -> (
          match args with
          | [] -> false
          | s :: l -> (
            match s with
            | Var _ -> ( match l with [] -> true | _ :: _ -> false)
            | _ -> false))
        | End_try_region -> (
          match args with
          | [] -> false
          | s :: l -> (
            match s with
            | Var _ -> ( match l with [] -> true | _ :: _ -> false)
            | _ -> false))
        | _ -> false)
      | _ -> ends_first body)
    | _ -> ends_first body)
  | Let_static (_, body) -> ends_first body
  | _ -> false

(** val enters_stale : ends_options -> continuation_handler -> bool **)

let enters_stale opts = function
  | Handler (_, metadata, handler) ->
    metadata.hm_is_exn_handler || (negb opts.eo_own_end && ends_first handler)

(** val ends_cont_of : certificate -> ends_options -> continuation ->
    continuation_handler -> ends_cont **)

let ends_cont_of cert opts k h =
  match h with
  | Handler (params, metadata, _) -> (
    match lookup k cert.cert_handlers with
    | Some entry ->
      Ends_handler (params, metadata.hm_params_kinds, entry, enters_stale opts h)
    | None -> Ends_unreachable)

(** val ends_with_conts : ends_state -> (continuation * ends_cont) list ->
    ends_state **)

let ends_with_conts es conts =
  { es_open = es.es_open;
    es_tries = es.es_tries;
    es_regions = es.es_regions;
    es_stale = es.es_stale;
    es_conts = app conts es.es_conts
  }

(** val ends_handler_state : ends_state -> variable list -> flambda_kind list ->
    handler_entry -> bool -> ends_state **)

let ends_handler_state es params kinds entry stale =
  { es_open = entry.he_open;
    es_tries = es.es_tries;
    es_regions = app (region_params_of params kinds) es.es_regions;
    es_stale = stale;
    es_conts = es.es_conts
  }

(** val ends_code_state : continuation -> continuation -> ends_state **)

let ends_code_state return_continuation exn_continuation =
  { es_open = [];
    es_tries = [];
    es_regions = [];
    es_stale = false;
    es_conts = [return_continuation, Ends_return; exn_continuation, Ends_raise]
  }

(** val ends_expr : certificate -> ends_options -> ends_state -> expr -> bool **)

let ends_expr cert opts =
  let rec ends_expr0 es = function
    | Let (x, _, defining_expr, body) -> (
      match defining_expr with
      | Prim (p0, args) -> (
        match p0 with
        | Region_primitive p -> (
          negb (among x es.es_regions)
          &&
          match p with
          | Begin_region -> (
            match args with
            | [] ->
              negb es.es_stale
              && ends_expr0
                   { es_open = x :: es.es_open;
                     es_tries = es.es_tries;
                     es_regions = x :: es.es_regions;
                     es_stale = false;
                     es_conts = es.es_conts
                   }
                   body
            | _ :: _ -> false)
          | Begin_try_region -> (
            match args with
            | [] ->
              negb es.es_stale
              && ends_expr0
                   { es_open = x :: es.es_open;
                     es_tries = x :: es.es_tries;
                     es_regions = x :: es.es_regions;
                     es_stale = false;
                     es_conts = es.es_conts
                   }
                   body
            | _ :: _ -> false)
          | Local_allocation -> ends_expr0 es body
          | _ -> (
            match args with
            | [] -> false
            | s :: l -> (
              match s with
              | Var r -> (
                match l with
                | [] ->
                  (among r es.es_open
                  && (negb opts.eo_own_end
                     || forallb
                          (fun r' -> among r' es.es_tries)
                          (regions_above r es.es_open)))
                  && ends_expr0
                       { es_open = regions_below r es.es_open;
                         es_tries = es.es_tries;
                         es_regions = es.es_regions;
                         es_stale = false;
                         es_conts = es.es_conts
                       }
                       body
                | _ :: _ -> false)
              | _ -> false)))
        | _ -> negb (among x es.es_regions) && ends_expr0 es body)
      | _ -> negb (among x es.es_regions) && ends_expr0 es body)
    | Let_static (bindings, body) ->
      forallb ends_binding bindings && ends_expr0 es body
    | Let_cont_non_recursive (k, h, body) ->
      ends_handler es k h
      && ends_expr0
           (ends_with_conts es ((k, ends_cont_of cert opts k h) :: []))
           body
    | Let_cont_recursive (handlers, body) ->
      let es' =
        ends_with_conts es
          (map
             (fun pat ->
               let k, h = pat in
               k, ends_cont_of cert opts k h)
             handlers)
      in
      forallb
        (fun pat ->
          let k, h = pat in
          ends_handler es' k h)
        handlers
      && ends_expr0 es' body
    | Apply ap ->
      ends_jump es ap.ap_continuation None
      && ends_raise_to es ap.ap_exn_continuation (exn_extra_simples ap)
    | Apply_cont ac -> ends_jump es ac.ac_continuation (Some ac.ac_args)
    | Switch (_, arms) ->
      forallb
        (fun pat ->
          let _, ac = pat in
          ends_jump es ac.ac_continuation (Some ac.ac_args))
        arms
    | Invalid -> true
  and ends_handler es k h =
    match h with
    | Handler (params, metadata, handler) -> (
      match lookup k cert.cert_handlers with
      | Some entry ->
        let kinds = metadata.hm_params_kinds in
        (((distinct params
          && forallb (fun p -> negb (among p es.es_regions)) params)
         && distinct entry.he_open)
        && forallb
             (fun r ->
               among r (region_params_of params kinds) || among r es.es_open)
             entry.he_open)
        && ends_expr0
             (ends_handler_state es params kinds entry (enters_stale opts h))
             handler
      | None -> true)
  and ends_binding = function Bind_code (_, c) -> ends_code0 c | _ -> true
  and ends_code0 = function
    | Code (_, _, return_continuation, exn_continuation, _, body) ->
      negb (Pos.eqb return_continuation exn_continuation)
      && ends_expr0 (ends_code_state return_continuation exn_continuation) body
  in
  ends_expr0

(** val ends_code : certificate -> ends_options -> code -> bool **)

let ends_code cert opts =
  let rec ends_expr0 es = function
    | Let (x, _, defining_expr, body) -> (
      match defining_expr with
      | Prim (p0, args) -> (
        match p0 with
        | Region_primitive p -> (
          negb (among x es.es_regions)
          &&
          match p with
          | Begin_region -> (
            match args with
            | [] ->
              negb es.es_stale
              && ends_expr0
                   { es_open = x :: es.es_open;
                     es_tries = es.es_tries;
                     es_regions = x :: es.es_regions;
                     es_stale = false;
                     es_conts = es.es_conts
                   }
                   body
            | _ :: _ -> false)
          | Begin_try_region -> (
            match args with
            | [] ->
              negb es.es_stale
              && ends_expr0
                   { es_open = x :: es.es_open;
                     es_tries = x :: es.es_tries;
                     es_regions = x :: es.es_regions;
                     es_stale = false;
                     es_conts = es.es_conts
                   }
                   body
            | _ :: _ -> false)
          | Local_allocation -> ends_expr0 es body
          | _ -> (
            match args with
            | [] -> false
            | s :: l -> (
              match s with
              | Var r -> (
                match l with
                | [] ->
                  (among r es.es_open
                  && (negb opts.eo_own_end
                     || forallb
                          (fun r' -> among r' es.es_tries)
                          (regions_above r es.es_open)))
                  && ends_expr0
                       { es_open = regions_below r es.es_open;
                         es_tries = es.es_tries;
                         es_regions = es.es_regions;
                         es_stale = false;
                         es_conts = es.es_conts
                       }
                       body
                | _ :: _ -> false)
              | _ -> false)))
        | _ -> negb (among x es.es_regions) && ends_expr0 es body)
      | _ -> negb (among x es.es_regions) && ends_expr0 es body)
    | Let_static (bindings, body) ->
      forallb ends_binding bindings && ends_expr0 es body
    | Let_cont_non_recursive (k, h, body) ->
      ends_handler es k h
      && ends_expr0
           (ends_with_conts es ((k, ends_cont_of cert opts k h) :: []))
           body
    | Let_cont_recursive (handlers, body) ->
      let es' =
        ends_with_conts es
          (map
             (fun pat ->
               let k, h = pat in
               k, ends_cont_of cert opts k h)
             handlers)
      in
      forallb
        (fun pat ->
          let k, h = pat in
          ends_handler es' k h)
        handlers
      && ends_expr0 es' body
    | Apply ap ->
      ends_jump es ap.ap_continuation None
      && ends_raise_to es ap.ap_exn_continuation (exn_extra_simples ap)
    | Apply_cont ac -> ends_jump es ac.ac_continuation (Some ac.ac_args)
    | Switch (_, arms) ->
      forallb
        (fun pat ->
          let _, ac = pat in
          ends_jump es ac.ac_continuation (Some ac.ac_args))
        arms
    | Invalid -> true
  and ends_handler es k h =
    match h with
    | Handler (params, metadata, handler) -> (
      match lookup k cert.cert_handlers with
      | Some entry ->
        let kinds = metadata.hm_params_kinds in
        (((distinct params
          && forallb (fun p -> negb (among p es.es_regions)) params)
         && distinct entry.he_open)
        && forallb
             (fun r ->
               among r (region_params_of params kinds) || among r es.es_open)
             entry.he_open)
        && ends_expr0
             (ends_handler_state es params kinds entry (enters_stale opts h))
             handler
      | None -> true)
  and ends_binding = function Bind_code (_, c) -> ends_code0 c | _ -> true
  and ends_code0 = function
    | Code (_, _, return_continuation, exn_continuation, _, body) ->
      negb (Pos.eqb return_continuation exn_continuation)
      && ends_expr0 (ends_code_state return_continuation exn_continuation) body
  in
  ends_code0

(** val ends_check : certificate -> ends_options -> continuation -> continuation
    -> static_env -> expr -> bool **)

let ends_check cert opts ret exn statics e =
  (forallb
     (fun pat ->
       let _, c = pat in
       ends_code cert opts c)
     statics.static_code
  && negb (Pos.eqb ret exn))
  && ends_expr cert opts (ends_code_state ret exn) e
