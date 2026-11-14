open BinNums
open BinPos
open Datatypes
open FlambdaHeapSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaPrimitiveModel
open FlambdaSemanticsSpec
open FlambdaValueSpec
open List
open ListDef
open Nat
open OxCamlOption

type mark = symbol

type mark_kind =
  | Plain_mark
  | Try_mark

type frame =
  { frame_mark : mark;
    frame_kind : mark_kind;
    frame_locals : (symbol * statically_allocated_value) list
  }

type region_counts =
  { count_other_region : nat;
    count_cut_by_end_region : nat;
    count_cut_by_end_try_region : nat;
    count_ends : nat;
    count_local_allocations : nat;
    count_no_frame : nat;
    count_local_uses : nat
  }

(** val no_counts : region_counts **)

let no_counts =
  { count_other_region = O;
    count_cut_by_end_region = O;
    count_cut_by_end_try_region = O;
    count_ends = O;
    count_local_allocations = O;
    count_no_frame = O;
    count_local_uses = O
  }

type region_state =
  { rs_state : state;
    rs_frames : frame list;
    rs_base : symbol;
    rs_next : symbol;
    rs_counts : region_counts;
    rs_cells : (location * mark) list
  }

(** val live_locals : frame list -> (symbol * statically_allocated_value) list **)

let live_locals frames = flat_map (fun f -> f.frame_locals) frames

(** val live_statics : region_state -> static_env **)

let live_statics rs =
  let statics = rs.rs_state.state_statics in
  { static_code = statics.static_code;
    static_values = app (live_locals rs.rs_frames) statics.static_values
  }

(** val drawn : region_state -> symbol -> bool **)

let drawn rs s = Pos.leb rs.rs_base s

(** val on_stack : mark -> frame list -> bool **)

let on_stack m frames = existsb (fun f -> Pos.eqb f.frame_mark m) frames

(** val local_cell : region_state -> location -> mark option **)

let local_cell rs l =
  match lookup l rs.rs_cells with
  | Some m -> if on_stack m rs.rs_frames then Some m else None
  | None -> None

(** val cell_dead : region_state -> location -> bool **)

let cell_dead rs l =
  match lookup l rs.rs_cells with
  | Some m -> negb (on_stack m rs.rs_frames)
  | None -> false

(** val value_usable : region_state -> value -> bool **)

let value_usable rs = function
  | Value_symbol s ->
    negb (drawn rs s) || is_some (lookup s (live_statics rs).static_values)
  | Value_location l -> negb (cell_dead rs l)
  | _ -> true

(** val value_local : region_state -> value -> bool **)

let value_local rs = function
  | Value_symbol s -> is_some (lookup s (live_locals rs.rs_frames))
  | Value_location l -> is_some (local_cell rs l)
  | _ -> false

(** val simple_usable : region_state -> simple -> bool **)

let simple_usable rs s =
  match eval_simple rs.rs_state.state_vars s with
  | Some v -> value_usable rs v
  | None -> true

(** val simple_local : region_state -> simple -> bool **)

let simple_local rs s =
  match eval_simple rs.rs_state.state_vars s with
  | Some v -> value_local rs v
  | None -> false

(** val simples_of_named : named -> simple list **)

let simples_of_named = function
  | Simple s -> s :: []
  | Prim (_, args) -> args
  | Set_of_closures (_, value_slots) -> map snd value_slots
  | Closure_of_set (_, _, value_slots) -> map snd value_slots
  | Project_value_slot (_, closure) -> closure :: []
  | Project_value_slot_from (_, _, closure) -> closure :: []
  | Project_function_slot (_, _, closure) -> closure :: []

(** val simples_of_static_const : static_const -> simple list **)

let simples_of_static_const = function
  | Static_const_set_of_closures (_, value_slots) -> map snd value_slots
  | Static_const_closure_of_set (_, _, _, value_slots) -> map snd value_slots
  | Static_const_block (_, fields) -> fields
  | Static_const_boxed_number contents -> contents :: []
  | Static_const_immutable_string _ -> []
  | Static_const_immutable_value_array elements -> elements

(** val simples_of_binding : static_binding -> simple list **)

let simples_of_binding = function
  | Bind_symbol (_, const0) -> simples_of_static_const const0
  | _ -> []

(** val evaluated_simples : expr -> simple list **)

let evaluated_simples = function
  | Let (_, _, defining_expr, _) -> (
    match defining_expr with
    | Prim (p, _) -> (
      match p with
      | Region_primitive _ -> []
      | _ -> simples_of_named defining_expr)
    | _ -> simples_of_named defining_expr)
  | Let_static (bindings, _) -> flat_map simples_of_binding bindings
  | Apply ap -> ap.ap_callee :: app ap.ap_args (exn_extra_simples ap)
  | Apply_cont ac -> ac.ac_args
  | Switch (scrutinee, _) -> scrutinee :: []
  | _ -> []

(** val is_allocation : named -> bool **)

let is_allocation = function
  | Prim (p, _) -> (
    match p with
    | Unary p0 -> ( match p0 with Box_number _ -> true | _ -> false)
    | Variadic _ -> true
    | _ -> false)
  | Set_of_closures (_, _) -> true
  | Closure_of_set (_, _, _) -> true
  | _ -> false

(** val is_cell_allocation : named -> bool **)

let is_cell_allocation = function
  | Prim (p, _) -> (
    match p with
    | Heap p0 -> (
      match p0 with
      | Make_block_with_identity (_, _) -> true
      | Make_mutable_array _ -> true
      | _ -> false)
    | _ -> false)
  | _ -> false

(** val stored : named -> (simple * simple) option **)

let stored = function
  | Prim (p, args) -> (
    match p with
    | Heap p0 -> (
      match p0 with
      | Block_set (_, _, _) -> (
        match args with
        | [] -> None
        | cell :: l -> (
          match l with
          | [] -> None
          | new0 :: l0 -> (
            match l0 with [] -> Some (cell, new0) | _ :: _ -> None)))
      | Array_set (_, _) -> (
        match args with
        | [] -> None
        | cell :: l -> (
          match l with
          | [] -> None
          | _ :: l0 -> (
            match l0 with
            | [] -> None
            | new0 :: l1 -> (
              match l1 with [] -> Some (cell, new0) | _ :: _ -> None))))
      | _ -> None)
    | _ -> None)
  | _ -> None

(** val number_slots : (function_slot * code_id) list -> symbol ->
    (function_slot * symbol) list **)

let rec number_slots function_decls next =
  match function_decls with
  | [] -> []
  | p :: function_decls0 ->
    let slot, _ = p in
    (slot, next) :: number_slots function_decls0 (Pos.succ next)

(** val localise : symbol -> value -> (symbol * (symbol *
    statically_allocated_value) list) option **)

let localise next = function
  | Value_closure (code0, value_slots) ->
    Some (next, (next, Static_closure (code0, value_slots)) :: [])
  | Value_closure_of_set (slot, function_decls, value_slots) -> (
    let symbols = number_slots function_decls next in
    match lookup slot symbols with
    | Some s ->
      Some
        ( s,
          map
            (fun pat ->
              let slot', s' = pat in
              ( s',
                Static_closure_of_set
                  (slot', symbols, function_decls, value_slots) ))
            symbols )
    | None -> None)
  | Value_block (tag, fields) ->
    Some (next, (next, Static_block (tag, fields)) :: [])
  | Value_immutable_array elements ->
    Some (next, (next, Static_immutable_array elements) :: [])
  | Value_boxed_number c ->
    Some (next, (next, Static_boxed_number (Value_const c)) :: [])
  | _ -> None

(** val advance : symbol -> nat -> symbol **)

let advance next count = PeanoNat.Nat.iter count Pos.succ next

(** val cut : mark -> frame list -> (frame list * frame list) option **)

let rec cut m = function
  | [] -> None
  | f :: frames0 -> (
    if Pos.eqb f.frame_mark m
    then Some ([], frames0)
    else
      match cut m frames0 with
      | Some p ->
        let above, below = p in
        Some (f :: above, below)
      | None -> None)

(** val is_plain : frame -> bool **)

let is_plain f =
  match f.frame_kind with Plain_mark -> true | Try_mark -> false

(** val newest_plain : frame list -> mark option **)

let newest_plain frames =
  option_map (fun f -> f.frame_mark) (find is_plain frames)

type marker =
  | No_marker
  | Marker of variable * simple option * expr
  | Bad_marker

(** val marker_of : variable -> expr -> marker **)

let marker_of x = function
  | Let (y, kind, defining_expr, body') -> (
    match defining_expr with
    | Prim (p, args) -> (
      match p with
      | Region_primitive p0 -> (
        match p0 with
        | Local_allocation -> (
          match args with
          | [] -> Bad_marker
          | s :: region -> (
            match s with
            | Var x' ->
              if Pos.eqb x x' && kind_eqb kind Value
              then
                match region with
                | [] -> Marker (y, None, body')
                | r :: l -> (
                  match r with
                  | Var z -> (
                    match l with
                    | [] ->
                      if Pos.eqb z x
                      then Bad_marker
                      else Marker (y, Some (Var z), body')
                    | _ :: _ -> Bad_marker)
                  | Symbol _ -> (
                    match l with
                    | [] -> Marker (y, Some r, body')
                    | _ :: _ -> Bad_marker)
                  | Const _ -> (
                    match l with
                    | [] -> Marker (y, Some r, body')
                    | _ :: _ -> Bad_marker))
              else Bad_marker
            | _ -> Bad_marker))
        | _ -> No_marker)
      | _ -> No_marker)
    | _ -> No_marker)
  | _ -> No_marker

(** val value_open_mark : region_state -> value -> bool **)

let value_open_mark rs v =
  match v with
  | Value_symbol m -> negb (value_usable rs v) && on_stack m rs.rs_frames
  | _ -> false

(** val simple_open_mark : region_state -> simple -> bool **)

let simple_open_mark rs s =
  match eval_simple rs.rs_state.state_vars s with
  | Some v -> value_open_mark rs v
  | None -> false

(** val region_params : continuation_handler -> bool list **)

let region_params h = map (fun kind -> kind_eqb kind Region) (handler_kinds h)

(** val region_places : region_state -> bool list **)

let region_places rs =
  match rs.rs_state.state_expr with
  | Apply_cont ac -> (
    match resolve ac.ac_continuation rs.rs_state.state_conts with
    | Closure (h, _, _, _) -> region_params h
    | _ -> [])
  | _ -> []

(** val region_kind_of_value : region_state -> value -> flambda_kind **)

let region_kind_of_value rs v =
  if value_open_mark rs v then Region else kind_of_value v

type region_fault =
  | Dead_value_used
  | Heap_holds_local
  | Cell_outlives_value
  | Region_not_open
  | Region_used_as_value
  | Not_of_the_conversion
  | No_plain_step

type region_step_result =
  | Region_step of region_state
  | Region_exit of continuation * value list * region_state
  | Region_undefined of region_fault

(** val with_statics : state -> static_env -> state **)

let with_statics s statics =
  { state_expr = s.state_expr;
    state_vars = s.state_vars;
    state_conts = s.state_conts;
    state_statics = statics;
    state_heap = s.state_heap;
    state_reads = s.state_reads
  }

(** val with_counts : region_state -> region_counts -> region_state **)

let with_counts rs counts =
  { rs_state = rs.rs_state;
    rs_frames = rs.rs_frames;
    rs_base = rs.rs_base;
    rs_next = rs.rs_next;
    rs_counts = counts;
    rs_cells = rs.rs_cells
  }

(** val plain_step : region_state -> region_step_result **)

let plain_step rs =
  let s = rs.rs_state in
  match step (with_statics s (live_statics rs)) with
  | Step s' ->
    Region_step
      { rs_state = with_statics s' s.state_statics;
        rs_frames = rs.rs_frames;
        rs_base = rs.rs_base;
        rs_next = rs.rs_next;
        rs_counts = rs.rs_counts;
        rs_cells = rs.rs_cells
      }
  | Exit (k, vs, _, _) -> Region_exit (k, vs, rs)
  | Undefined -> Region_undefined No_plain_step

(** val static_step : region_state -> region_step_result **)

let static_step rs =
  match step rs.rs_state with
  | Step s' ->
    Region_step
      { rs_state = s';
        rs_frames = rs.rs_frames;
        rs_base = rs.rs_base;
        rs_next = rs.rs_next;
        rs_counts = rs.rs_counts;
        rs_cells = rs.rs_cells
      }
  | Exit (k, vs, _, _) -> Region_exit (k, vs, rs)
  | Undefined -> Region_undefined No_plain_step

(** val bind_let : region_state -> variable -> value -> expr -> static_env ->
    frame list -> symbol -> region_counts -> region_state **)

let bind_let rs x v body statics frames next counts =
  let s = rs.rs_state in
  { rs_state =
      { state_expr = body;
        state_vars = (x, v) :: s.state_vars;
        state_conts = s.state_conts;
        state_statics = statics;
        state_heap = s.state_heap;
        state_reads = s.state_reads
      };
    rs_frames = frames;
    rs_base = rs.rs_base;
    rs_next = next;
    rs_counts = counts;
    rs_cells = rs.rs_cells
  }

(** val bind_pair : region_state -> variable -> variable -> value -> expr ->
    static_env -> frame list -> symbol -> region_counts -> region_state **)

let bind_pair rs x y v body statics frames next counts =
  let s = rs.rs_state in
  { rs_state =
      { state_expr = body;
        state_vars = (y, v) :: (x, v) :: s.state_vars;
        state_conts = s.state_conts;
        state_statics = statics;
        state_heap = s.state_heap;
        state_reads = s.state_reads
      };
    rs_frames = frames;
    rs_base = rs.rs_base;
    rs_next = next;
    rs_counts = counts;
    rs_cells = rs.rs_cells
  }

(** val begin_region : region_state -> variable -> flambda_kind -> mark_kind ->
    expr -> region_step_result **)

let begin_region rs x kind mark_kind0 body =
  if kind_eqb kind Region
  then
    let m = rs.rs_next in
    Region_step
      (bind_let rs x (Value_symbol m) body rs.rs_state.state_statics
         ({ frame_mark = m; frame_kind = mark_kind0; frame_locals = [] }
         :: rs.rs_frames)
         (Pos.succ m) rs.rs_counts)
  else Region_undefined Not_of_the_conversion

(** val unit_value : value **)

let unit_value = Value_const (Tagged_immediate Z0)

(** val end_region : region_state -> variable -> flambda_kind -> mark_kind ->
    simple -> expr -> region_step_result **)

let end_region rs x kind ended_by region body =
  if kind_eqb kind Value
  then
    match eval_simple rs.rs_state.state_vars region with
    | Some v -> (
      match v with
      | Value_symbol m -> (
        match cut m rs.rs_frames with
        | Some p ->
          let above, below = p in
          let cut_away = length (filter is_plain above) in
          let counts = rs.rs_counts in
          let counts' =
            match ended_by with
            | Plain_mark ->
              { count_other_region = counts.count_other_region;
                count_cut_by_end_region =
                  add cut_away counts.count_cut_by_end_region;
                count_cut_by_end_try_region = counts.count_cut_by_end_try_region;
                count_ends = S counts.count_ends;
                count_local_allocations = counts.count_local_allocations;
                count_no_frame = counts.count_no_frame;
                count_local_uses = counts.count_local_uses
              }
            | Try_mark ->
              { count_other_region = counts.count_other_region;
                count_cut_by_end_region = counts.count_cut_by_end_region;
                count_cut_by_end_try_region =
                  add cut_away counts.count_cut_by_end_try_region;
                count_ends = S counts.count_ends;
                count_local_allocations = counts.count_local_allocations;
                count_no_frame = counts.count_no_frame;
                count_local_uses = counts.count_local_uses
              }
          in
          Region_step
            (bind_let rs x unit_value body rs.rs_state.state_statics below
               rs.rs_next counts')
        | None -> Region_undefined Region_not_open)
      | _ -> Region_undefined Region_not_open)
    | None -> Region_undefined Region_not_open
  else Region_undefined Not_of_the_conversion

(** val named_region : region_state -> simple option -> bool option **)

let named_region rs = function
  | Some r -> (
    match eval_simple rs.rs_state.state_vars r with
    | Some v -> (
      match v with
      | Value_symbol m ->
        if on_stack m rs.rs_frames
        then
          match newest_plain rs.rs_frames with
          | Some newest -> Some (negb (Pos.eqb m newest))
          | None -> Some true
        else None
      | _ -> None)
    | None -> None)
  | None -> Some false

(** val frames_from : mark -> frame list -> frame list option **)

let rec frames_from m = function
  | [] -> None
  | f :: below ->
    if Pos.eqb f.frame_mark m then Some (f :: below) else frames_from m below

(** val value_frame : region_state -> value -> mark option **)

let value_frame rs = function
  | Value_symbol s ->
    option_map
      (fun f -> f.frame_mark)
      (find (fun f -> is_some (lookup s f.frame_locals)) rs.rs_frames)
  | Value_location l -> local_cell rs l
  | _ -> None

(** val lives_as_long : region_state -> value -> mark -> bool **)

let lives_as_long rs v m =
  match value_frame rs v with
  | Some m' -> (
    match frames_from m rs.rs_frames with
    | Some frames -> on_stack m' frames
    | None -> false)
  | None -> true

(** val store_fault : region_state -> named -> region_fault option **)

let store_fault rs defining_expr =
  match stored defining_expr with
  | Some p -> (
    let cell, new0 = p in
    let vars = rs.rs_state.state_vars in
    match eval_simple vars cell with
    | Some v0 -> (
      match v0 with
      | Value_location l -> (
        match eval_simple vars new0 with
        | Some v -> (
          match local_cell rs l with
          | Some m ->
            if lives_as_long rs v m then None else Some Cell_outlives_value
          | None -> if value_local rs v then Some Heap_holds_local else None)
        | None -> None)
      | _ -> None)
    | None -> None)
  | None -> None

(** val local_cell_allocation : region_state -> variable -> variable -> named ->
    simple option -> expr -> region_step_result **)

let local_cell_allocation rs x y defining_expr region body =
  let s = rs.rs_state in
  match named_region rs region with
  | Some other -> (
    match
      exec_named (live_statics rs) s.state_heap s.state_reads s.state_vars
        defining_expr
    with
    | Some p -> (
      let p0, rd = p in
      let v, hp = p0 in
      match v with
      | Value_location l ->
        let counts = rs.rs_counts in
        let counts' =
          { count_other_region =
              add (if other then S O else O) counts.count_other_region;
            count_cut_by_end_region = counts.count_cut_by_end_region;
            count_cut_by_end_try_region = counts.count_cut_by_end_try_region;
            count_ends = counts.count_ends;
            count_local_allocations = S counts.count_local_allocations;
            count_no_frame =
              add
                (match rs.rs_frames with [] -> S O | _ :: _ -> O)
                counts.count_no_frame;
            count_local_uses = counts.count_local_uses
          }
        in
        Region_step
          { rs_state =
              { state_expr = body;
                state_vars =
                  (y, Value_location l) :: (x, Value_location l) :: s.state_vars;
                state_conts = s.state_conts;
                state_statics = s.state_statics;
                state_heap = hp;
                state_reads = rd
              };
            rs_frames = rs.rs_frames;
            rs_base = rs.rs_base;
            rs_next = rs.rs_next;
            rs_counts = counts';
            rs_cells =
              (match rs.rs_frames with
              | [] -> rs.rs_cells
              | f :: _ -> (l, f.frame_mark) :: rs.rs_cells)
          }
      | _ -> Region_undefined Not_of_the_conversion)
    | None -> Region_undefined No_plain_step)
  | None -> Region_undefined Region_not_open

(** val local_allocation : region_state -> variable -> variable -> flambda_kind
    -> named -> simple option -> expr -> region_step_result **)

let local_allocation rs x y kind defining_expr region body =
  let s = rs.rs_state in
  if is_cell_allocation defining_expr && kind_eqb kind Value
  then local_cell_allocation rs x y defining_expr region body
  else if is_allocation defining_expr && kind_eqb kind Value
  then
    match named_region rs region with
    | Some other -> (
      match eval_named (live_statics rs) s.state_vars defining_expr with
      | Some v -> (
        match localise rs.rs_next v with
        | Some p -> (
          let symbol0, bindings = p in
          let counts = rs.rs_counts in
          let counts' =
            { count_other_region =
                add (if other then S O else O) counts.count_other_region;
              count_cut_by_end_region = counts.count_cut_by_end_region;
              count_cut_by_end_try_region = counts.count_cut_by_end_try_region;
              count_ends = counts.count_ends;
              count_local_allocations = S counts.count_local_allocations;
              count_no_frame =
                add
                  (match rs.rs_frames with [] -> S O | _ :: _ -> O)
                  counts.count_no_frame;
              count_local_uses = counts.count_local_uses
            }
          in
          let next = advance rs.rs_next (length bindings) in
          match rs.rs_frames with
          | [] ->
            Region_step
              (bind_pair rs x y (Value_symbol symbol0) body
                 { static_code = s.state_statics.static_code;
                   static_values = app bindings s.state_statics.static_values
                 }
                 [] next counts')
          | f :: frames ->
            let { frame_mark = m;
                  frame_kind = mark_kind0;
                  frame_locals = locals
                } =
              f
            in
            Region_step
              (bind_pair rs x y (Value_symbol symbol0) body s.state_statics
                 ({ frame_mark = m;
                    frame_kind = mark_kind0;
                    frame_locals = app bindings locals
                  }
                 :: frames)
                 next counts'))
        | None -> Region_undefined Not_of_the_conversion)
      | None -> Region_undefined No_plain_step)
    | None -> Region_undefined Region_not_open
  else Region_undefined Not_of_the_conversion

(** val count_uses : region_state -> region_state **)

let count_uses rs =
  let counts = rs.rs_counts in
  let uses =
    length (filter (simple_local rs) (evaluated_simples rs.rs_state.state_expr))
  in
  with_counts rs
    { count_other_region = counts.count_other_region;
      count_cut_by_end_region = counts.count_cut_by_end_region;
      count_cut_by_end_try_region = counts.count_cut_by_end_try_region;
      count_ends = counts.count_ends;
      count_local_allocations = counts.count_local_allocations;
      count_no_frame = counts.count_no_frame;
      count_local_uses = add uses counts.count_local_uses
    }

(** val simples_fault : region_state -> simple list -> bool list -> region_fault
    option **)

let rec simples_fault rs simples places =
  match simples with
  | [] -> None
  | s :: simples0 ->
    let place = hd false places in
    if simple_usable rs s
    then simples_fault rs simples0 (tl places)
    else if simple_open_mark rs s
    then
      if place
      then simples_fault rs simples0 (tl places)
      else Some Region_used_as_value
    else Some (if place then Region_not_open else Dead_value_used)

(** val region_apply_cont : region_state -> apply_cont -> region_step_result **)

let region_apply_cont rs ac =
  let s = rs.rs_state in
  match resolve ac.ac_continuation s.state_conts with
  | Closure (h, def_vars, def_conts, recursive) -> (
    let (Handler (params, metadata, handler)) = h in
    match eval_simples s.state_vars ac.ac_args with
    | Some vs ->
      if kinds_eqb (map (region_kind_of_value rs) vs) metadata.hm_params_kinds
      then
        match bind_params params vs def_vars with
        | Some handler_vars ->
          Region_step
            { rs_state =
                { state_expr = handler;
                  state_vars = handler_vars;
                  state_conts =
                    app (rec_closures recursive def_vars def_conts) def_conts;
                  state_statics = s.state_statics;
                  state_heap = s.state_heap;
                  state_reads = s.state_reads
                };
              rs_frames = rs.rs_frames;
              rs_base = rs.rs_base;
              rs_next = rs.rs_next;
              rs_counts = rs.rs_counts;
              rs_cells = rs.rs_cells
            }
        | None -> Region_undefined No_plain_step
      else Region_undefined No_plain_step
    | None -> Region_undefined No_plain_step)
  | _ -> plain_step rs

(** val region_step : region_state -> region_step_result **)

let region_step rs =
  let e = rs.rs_state.state_expr in
  match simples_fault rs (evaluated_simples e) (region_places rs) with
  | Some fault -> Region_undefined fault
  | None -> (
    let rs0 = count_uses rs in
    match e with
    | Let (x, kind, defining_expr, body) -> (
      match defining_expr with
      | Prim (p0, args) -> (
        match p0 with
        | Region_primitive p -> (
          match p with
          | Begin_region -> (
            match args with
            | [] -> begin_region rs0 x kind Plain_mark body
            | _ :: _ -> Region_undefined Not_of_the_conversion)
          | Begin_try_region -> (
            match args with
            | [] -> begin_region rs0 x kind Try_mark body
            | _ :: _ -> Region_undefined Not_of_the_conversion)
          | End_region -> (
            match args with
            | [] -> Region_undefined Not_of_the_conversion
            | region :: l -> (
              match l with
              | [] -> end_region rs0 x kind Plain_mark region body
              | _ :: _ -> Region_undefined Not_of_the_conversion))
          | End_try_region -> (
            match args with
            | [] -> Region_undefined Not_of_the_conversion
            | region :: l -> (
              match l with
              | [] -> end_region rs0 x kind Try_mark region body
              | _ :: _ -> Region_undefined Not_of_the_conversion))
          | Local_allocation -> Region_undefined Not_of_the_conversion)
        | _ -> (
          match marker_of x body with
          | No_marker -> (
            if
              (is_allocation defining_expr || is_cell_allocation defining_expr)
              && existsb (simple_local rs0) (simples_of_named defining_expr)
            then Region_undefined Heap_holds_local
            else
              match store_fault rs0 defining_expr with
              | Some fault -> Region_undefined fault
              | None -> plain_step rs0)
          | Marker (y, region, body') ->
            local_allocation rs0 x y kind defining_expr region body'
          | Bad_marker -> Region_undefined Not_of_the_conversion))
      | _ -> (
        match marker_of x body with
        | No_marker -> (
          if
            (is_allocation defining_expr || is_cell_allocation defining_expr)
            && existsb (simple_local rs0) (simples_of_named defining_expr)
          then Region_undefined Heap_holds_local
          else
            match store_fault rs0 defining_expr with
            | Some fault -> Region_undefined fault
            | None -> plain_step rs0)
        | Marker (y, region, body') ->
          local_allocation rs0 x y kind defining_expr region body'
        | Bad_marker -> Region_undefined Not_of_the_conversion))
    | Let_static (_, _) ->
      if existsb (simple_local rs0) (evaluated_simples e)
      then Region_undefined Heap_holds_local
      else static_step rs0
    | Apply_cont ac ->
      if existsb (simple_open_mark rs0) ac.ac_args
      then region_apply_cont rs0 ac
      else plain_step rs0
    | _ -> plain_step rs0)

(** val initial_region_state : expr -> static_env -> symbol -> region_state **)

let initial_region_state e statics base =
  { rs_state =
      { state_expr = e;
        state_vars = [];
        state_conts = [];
        state_statics = statics;
        state_heap = empty_heap;
        state_reads = no_reads
      };
    rs_frames = [];
    rs_base = base;
    rs_next = base;
    rs_counts = no_counts;
    rs_cells = []
  }

(** val symbol_below : symbol -> symbol -> bool **)

let symbol_below base s = Pos.ltb s base

(** val simple_below : symbol -> simple -> bool **)

let simple_below base = function
  | Symbol sym -> symbol_below base sym
  | _ -> true

(** val static_const_below : symbol -> static_const -> bool **)

let static_const_below base const0 =
  forallb (simple_below base) (simples_of_static_const const0)
  &&
  match const0 with
  | Static_const_closure_of_set (_, closure_symbols, _, _) ->
    forallb
      (fun pat ->
        let _, s = pat in
        symbol_below base s)
      closure_symbols
  | _ -> true

(** val apply_cont_below : symbol -> apply_cont -> bool **)

let apply_cont_below base ac = forallb (simple_below base) ac.ac_args

(** val expr_below : symbol -> expr -> bool **)

let rec expr_below base = function
  | Let (_, _, defining_expr, body) ->
    forallb (simple_below base) (simples_of_named defining_expr)
    && expr_below base body
  | Let_static (bindings, body) ->
    forallb (binding_below base) bindings && expr_below base body
  | Let_cont_non_recursive (_, h, body) ->
    handler_below base h && expr_below base body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_below base h)
      handlers
    && expr_below base body
  | Apply ap ->
    forallb (simple_below base)
      (ap.ap_callee :: app ap.ap_args (exn_extra_simples ap))
  | Apply_cont ac -> apply_cont_below base ac
  | Switch (scrutinee, arms) ->
    simple_below base scrutinee
    && forallb
         (fun pat ->
           let _, ac = pat in
           apply_cont_below base ac)
         arms
  | Invalid -> true

(** val handler_below : symbol -> continuation_handler -> bool **)

and handler_below base = function
  | Handler (_, _, handler) -> expr_below base handler

(** val binding_below : symbol -> static_binding -> bool **)

and binding_below base = function
  | Bind_code (_, c) -> code_below base c
  | Bind_deleted_code _ -> true
  | Bind_symbol (s, const0) ->
    symbol_below base s && static_const_below base const0

(** val code_below : symbol -> code -> bool **)

and code_below base = function
  | Code (_, _, _, _, _, body) -> expr_below base body

(** val value_below : symbol -> value -> bool **)

let rec value_below base = function
  | Value_closure (_, value_slots) ->
    forallb
      (fun pat ->
        let _, v0 = pat in
        value_below base v0)
      value_slots
  | Value_closure_of_set (_, _, value_slots) ->
    forallb
      (fun pat ->
        let _, v0 = pat in
        value_below base v0)
      value_slots
  | Value_symbol s -> symbol_below base s
  | Value_block (_, fields) -> forallb (value_below base) fields
  | Value_immutable_array elements -> forallb (value_below base) elements
  | Value_partial_application (callee, args, _) ->
    value_below base callee && forallb (value_below base) args
  | _ -> true

(** val allocated_below : symbol -> statically_allocated_value -> bool **)

let allocated_below base = function
  | Static_closure (_, value_slots) ->
    forallb
      (fun pat ->
        let _, v = pat in
        value_below base v)
      value_slots
  | Static_closure_of_set (_, closure_symbols, _, value_slots) ->
    forallb
      (fun pat ->
        let _, s = pat in
        symbol_below base s)
      closure_symbols
    && forallb
         (fun pat ->
           let _, v = pat in
           value_below base v)
         value_slots
  | Static_block (_, fields) -> forallb (value_below base) fields
  | Static_boxed_number contents -> value_below base contents
  | Static_immutable_array elements -> forallb (value_below base) elements
  | _ -> true

(** val statics_below : symbol -> static_env -> bool **)

let statics_below base statics =
  forallb
    (fun pat ->
      let _, c = pat in
      code_below base c)
    statics.static_code
  && forallb
       (fun pat ->
         let s, allocated = pat in
         symbol_below base s && allocated_below base allocated)
       statics.static_values

(** val statics_hold_no_location : static_env -> bool **)

let statics_hold_no_location statics =
  forallb
    (fun pat ->
      let _, allocated = pat in
      forallb
        (fun v ->
          match locations_of_value v with [] -> true | _ :: _ -> false)
        (values_of_static allocated))
    statics.static_values

(** val region_names_fresh : symbol -> expr -> static_env -> bool **)

let region_names_fresh base e statics =
  (expr_below base e && statics_below base statics)
  && statics_hold_no_location statics

(** val unit_simple : simple **)

let unit_simple = Const (Tagged_immediate Z0)

(** val erase_region_primitive : variable -> region_primitive -> simple list ->
    named option **)

let erase_region_primitive x p args =
  match p with
  | End_region -> (
    match args with
    | [] -> Some (Prim (Region_primitive p, args))
    | _ :: l -> (
      match l with
      | [] -> Some (Simple unit_simple)
      | _ :: _ -> Some (Prim (Region_primitive p, args))))
  | End_try_region -> (
    match args with
    | [] -> Some (Prim (Region_primitive p, args))
    | _ :: l -> (
      match l with
      | [] -> Some (Prim (Unary (Opaque_identity Value), unit_simple :: []))
      | _ :: _ -> Some (Prim (Region_primitive p, args))))
  | Local_allocation -> (
    match args with
    | [] -> Some (Prim (Region_primitive p, args))
    | s :: _ -> (
      match s with
      | Var y -> if Pos.eqb x y then None else Some (Simple (Var y))
      | _ -> Some (Prim (Region_primitive p, args))))
  | _ -> (
    match args with
    | [] -> None
    | _ :: _ -> Some (Prim (Region_primitive p, args)))

(** val drop_regions : bool list -> 'a1 list -> 'a1 list **)

let rec drop_regions places l =
  match places with
  | [] -> l
  | place :: places0 -> (
    match l with
    | [] -> []
    | a :: l0 ->
      if place then drop_regions places0 l0 else a :: drop_regions places0 l0)

type region_conts = (continuation * bool list) list

(** val erase_apply_cont : region_conts -> apply_cont -> apply_cont **)

let erase_apply_cont conts ac =
  match lookup ac.ac_continuation conts with
  | Some places ->
    { ac_continuation = ac.ac_continuation;
      ac_args = drop_regions places ac.ac_args;
      ac_trap_action = ac.ac_trap_action
    }
  | None -> ac

(** val erase : region_conts -> expr -> expr **)

let rec erase conts e =
  match e with
  | Let (x, kind, defining_expr, body) -> (
    match defining_expr with
    | Prim (p0, args) -> (
      match p0 with
      | Region_primitive p -> (
        match erase_region_primitive x p args with
        | Some defining_expr0 -> Let (x, kind, defining_expr0, erase conts body)
        | None -> erase conts body)
      | _ -> Let (x, kind, defining_expr, erase conts body))
    | _ -> Let (x, kind, defining_expr, erase conts body))
  | Let_static (bindings, body) ->
    Let_static (map erase_binding bindings, erase conts body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, erase_handler conts h, erase ((k, region_params h) :: conts) body)
  | Let_cont_recursive (handlers, body) ->
    let conts' =
      app
        (map
           (fun pat ->
             let k, h = pat in
             k, region_params h)
           handlers)
        conts
    in
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, erase_handler conts' h)
          handlers,
        erase conts' body )
  | Apply_cont ac -> Apply_cont (erase_apply_cont conts ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let i, ac = pat in
            i, erase_apply_cont conts ac)
          arms )
  | _ -> e

(** val erase_handler : region_conts -> continuation_handler ->
    continuation_handler **)

and erase_handler conts = function
  | Handler (params, metadata, handler) ->
    let places =
      map (fun kind -> kind_eqb kind Region) metadata.hm_params_kinds
    in
    Handler
      ( drop_regions places params,
        { hm_params_kinds = drop_regions places metadata.hm_params_kinds;
          hm_is_exn_handler = metadata.hm_is_exn_handler
        },
        erase conts handler )

(** val erase_binding : static_binding -> static_binding **)

and erase_binding binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, erase_code c)
  | _ -> binding

(** val erase_code : code -> code **)

and erase_code = function
  | Code
      (params, my_closure, return_continuation, exn_continuation, metadata, body)
    ->
    Code
      ( params,
        my_closure,
        return_continuation,
        exn_continuation,
        metadata,
        erase [] body )

(** val takes_no_region : region_conts -> continuation -> bool **)

let takes_no_region conts k =
  match lookup k conts with
  | Some places -> negb (existsb (fun place -> place) places)
  | None -> true

(** val calls_return_no_region : region_conts -> expr -> bool **)

let rec calls_return_no_region conts = function
  | Let (_, _, _, body) -> calls_return_no_region conts body
  | Let_static (bindings, body) ->
    forallb binding_returns_no_region bindings
    && calls_return_no_region conts body
  | Let_cont_non_recursive (k, h, body) ->
    handler_returns_no_region conts h
    && calls_return_no_region ((k, region_params h) :: conts) body
  | Let_cont_recursive (handlers, body) ->
    let conts' =
      app
        (map
           (fun pat ->
             let k, h = pat in
             k, region_params h)
           handlers)
        conts
    in
    forallb
      (fun pat ->
        let _, h = pat in
        handler_returns_no_region conts' h)
      handlers
    && calls_return_no_region conts' body
  | Apply ap ->
    takes_no_region conts ap.ap_continuation
    && takes_no_region conts ap.ap_exn_continuation
  | _ -> true

(** val handler_returns_no_region : region_conts -> continuation_handler -> bool
    **)

and handler_returns_no_region conts = function
  | Handler (_, _, handler) -> calls_return_no_region conts handler

(** val binding_returns_no_region : static_binding -> bool **)

and binding_returns_no_region = function
  | Bind_code (_, c) -> code_returns_no_region c
  | _ -> true

(** val code_returns_no_region : code -> bool **)

and code_returns_no_region = function
  | Code (_, _, _, _, _, body) -> calls_return_no_region [] body

(** val statics_return_no_region : static_env -> bool **)

let statics_return_no_region statics =
  forallb
    (fun pat ->
      let _, c = pat in
      code_returns_no_region c)
    statics.static_code

(** val const_no_region_poison : const -> bool **)

let const_no_region_poison = function
  | Poison kind -> negb (kind_eqb kind Region)
  | _ -> true

(** val simple_no_region_poison : simple -> bool **)

let simple_no_region_poison = function
  | Const c -> const_no_region_poison c
  | _ -> true

(** val apply_cont_no_region_poison : apply_cont -> bool **)

let apply_cont_no_region_poison ac = forallb simple_no_region_poison ac.ac_args

(** val no_region_poison : expr -> bool **)

let rec no_region_poison = function
  | Let (_, _, defining_expr, body) ->
    forallb simple_no_region_poison (simples_of_named defining_expr)
    && no_region_poison body
  | Let_static (bindings, body) ->
    forallb binding_no_region_poison bindings && no_region_poison body
  | Let_cont_non_recursive (_, h, body) ->
    handler_no_region_poison h && no_region_poison body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_no_region_poison h)
      handlers
    && no_region_poison body
  | Apply ap ->
    forallb simple_no_region_poison
      (ap.ap_callee :: app ap.ap_args (exn_extra_simples ap))
  | Apply_cont ac -> apply_cont_no_region_poison ac
  | Switch (scrutinee, arms) ->
    simple_no_region_poison scrutinee
    && forallb
         (fun pat ->
           let _, ac = pat in
           apply_cont_no_region_poison ac)
         arms
  | Invalid -> true

(** val handler_no_region_poison : continuation_handler -> bool **)

and handler_no_region_poison = function
  | Handler (_, _, handler) -> no_region_poison handler

(** val binding_no_region_poison : static_binding -> bool **)

and binding_no_region_poison = function
  | Bind_code (_, c) -> code_no_region_poison c
  | Bind_deleted_code _ -> true
  | Bind_symbol (_, const0) ->
    forallb simple_no_region_poison (simples_of_static_const const0)

(** val code_no_region_poison : code -> bool **)

and code_no_region_poison = function
  | Code (_, _, _, _, _, body) -> no_region_poison body

(** val value_no_region_poison : value -> bool **)

let rec value_no_region_poison = function
  | Value_const c -> const_no_region_poison c
  | Value_closure (_, value_slots) ->
    forallb
      (fun pat ->
        let _, v0 = pat in
        value_no_region_poison v0)
      value_slots
  | Value_closure_of_set (_, _, value_slots) ->
    forallb
      (fun pat ->
        let _, v0 = pat in
        value_no_region_poison v0)
      value_slots
  | Value_block (_, fields) -> forallb value_no_region_poison fields
  | Value_immutable_array elements -> forallb value_no_region_poison elements
  | Value_boxed_number c -> const_no_region_poison c
  | Value_partial_application (callee, args, _) ->
    value_no_region_poison callee && forallb value_no_region_poison args
  | _ -> true

(** val allocated_no_region_poison : statically_allocated_value -> bool **)

let allocated_no_region_poison = function
  | Static_closure (_, value_slots) ->
    forallb
      (fun pat ->
        let _, v = pat in
        value_no_region_poison v)
      value_slots
  | Static_closure_of_set (_, _, _, value_slots) ->
    forallb
      (fun pat ->
        let _, v = pat in
        value_no_region_poison v)
      value_slots
  | Static_block (_, fields) -> forallb value_no_region_poison fields
  | Static_boxed_number contents -> value_no_region_poison contents
  | Static_immutable_array elements -> forallb value_no_region_poison elements
  | _ -> true

(** val statics_no_region_poison : static_env -> bool **)

let statics_no_region_poison statics =
  forallb
    (fun pat ->
      let _, c = pat in
      code_no_region_poison c)
    statics.static_code
  && forallb
       (fun pat ->
         let _, allocated = pat in
         allocated_no_region_poison allocated)
       statics.static_values

(** val erasable : expr -> static_env -> bool **)

let erasable e statics =
  ((calls_return_no_region [] e && statics_return_no_region statics)
  && no_region_poison e)
  && statics_no_region_poison statics
