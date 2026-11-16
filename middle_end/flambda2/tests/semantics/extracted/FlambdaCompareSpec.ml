open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaKindModel
open FlambdaModel
open FlambdaPrimitiveSpec
open List
open ListDef
open PeanoNat

type name_map = (positive * positive) list

(** val names_match : name_map -> positive -> positive -> bool **)

let rec names_match m a a' =
  match m with
  | [] -> Pos.eqb a a'
  | p :: m0 ->
    let b, b' = p in
    if Pos.eqb a b
    then Pos.eqb a' b'
    else if Pos.eqb a' b'
    then false
    else names_match m0 a a'

(** val bind_names : positive list -> positive list -> name_map -> name_map
    option **)

let rec bind_names xs xs' m =
  match xs with
  | [] -> ( match xs' with [] -> Some m | _ :: _ -> None)
  | x :: xs0 -> (
    match xs' with
    | [] -> None
    | x' :: xs'0 -> bind_names xs0 xs'0 ((x, x') :: m))

(** val combine_names : positive list -> positive list -> name_map -> name_map
    option **)

let combine_names xs xs' m =
  if Nat.eqb (length xs) (length xs')
  then Some (app (combine xs xs') m)
  else None

(** val const_eq_dec : const -> const -> bool **)

let const_eq_dec c c' =
  match c with
  | Naked_immediate i -> (
    match c' with Naked_immediate i0 -> Z.eq_dec i i0 | _ -> false)
  | Tagged_immediate i -> (
    match c' with Tagged_immediate i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_float32 bits -> (
    match c' with Naked_float32 bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Naked_float bits -> (
    match c' with Naked_float bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Naked_int8 i -> (
    match c' with Naked_int8 i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_int16 i -> (
    match c' with Naked_int16 i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_int32 i -> (
    match c' with Naked_int32 i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_int64 i -> (
    match c' with Naked_int64 i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_nativeint i -> (
    match c' with Naked_nativeint i0 -> Z.eq_dec i i0 | _ -> false)
  | Naked_vec128 bits -> (
    match c' with Naked_vec128 bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Naked_vec256 bits -> (
    match c' with Naked_vec256 bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Naked_vec512 bits -> (
    match c' with Naked_vec512 bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Naked_mask bits -> (
    match c' with Naked_mask bits0 -> Z.eq_dec bits bits0 | _ -> false)
  | Null -> ( match c' with Null -> true | _ -> false)
  | Poison kind -> (
    match c' with Poison kind0 -> kind_eq_dec kind kind0 | _ -> false)

(** val const_eqb : const -> const -> bool **)

let const_eqb c c' = if const_eq_dec c c' then true else false

(** val forall2b : ('a1 -> 'a2 -> bool) -> 'a1 list -> 'a2 list -> bool **)

let rec forall2b f l l' =
  match l with
  | [] -> ( match l' with [] -> true | _ :: _ -> false)
  | a :: l0 -> (
    match l' with [] -> false | b :: l'0 -> f a b && forall2b f l0 l'0)

(** val simple_eqb : name_map -> simple -> simple -> bool **)

let simple_eqb vm s s' =
  match s with
  | Var x -> ( match s' with Var x' -> names_match vm x x' | _ -> false)
  | Symbol sym -> (
    match s' with Symbol sym' -> Pos.eqb sym sym' | _ -> false)
  | Const c -> ( match s' with Const c' -> const_eqb c c' | _ -> false)

(** val simples_eqb : name_map -> simple list -> simple list -> bool **)

let simples_eqb vm args args' = forall2b (simple_eqb vm) args args'

(** val value_slots_eqb : name_map -> (value_slot * simple) list -> (value_slot
    * simple) list -> bool **)

let value_slots_eqb vm slots slots' =
  forall2b
    (fun slot slot' ->
      Pos.eqb (fst slot) (fst slot') && simple_eqb vm (snd slot) (snd slot'))
    slots slots'

(** val names_eqb : (positive * positive) list -> (positive * positive) list ->
    bool **)

let names_eqb l l' =
  forall2b
    (fun a a' -> Pos.eqb (fst a) (fst a') && Pos.eqb (snd a) (snd a'))
    l l'

(** val named_eqb : name_map -> named -> named -> bool **)

let named_eqb vm d d' =
  match d with
  | Simple s -> ( match d' with Simple s' -> simple_eqb vm s s' | _ -> false)
  | Prim (p, args) -> (
    match d' with
    | Prim (p', args') -> prim_eqb p p' && simples_eqb vm args args'
    | _ -> false)
  | Set_of_closures (code0, slots) -> (
    match d' with
    | Set_of_closures (code', slots') ->
      Pos.eqb code0 code' && value_slots_eqb vm slots slots'
    | _ -> false)
  | Closure_of_set (slot, decls, slots) -> (
    match d' with
    | Closure_of_set (slot', decls', slots') ->
      (Pos.eqb slot slot' && names_eqb decls decls')
      && value_slots_eqb vm slots slots'
    | _ -> false)
  | Project_value_slot (slot, closure) -> (
    match d' with
    | Project_value_slot (slot', closure') ->
      Pos.eqb slot slot' && simple_eqb vm closure closure'
    | _ -> false)
  | Project_value_slot_from (project_from, slot, closure) -> (
    match d' with
    | Project_value_slot_from (project_from', slot', closure') ->
      (Pos.eqb project_from project_from' && Pos.eqb slot slot')
      && simple_eqb vm closure closure'
    | _ -> false)
  | Project_function_slot (move_from, move_to, closure) -> (
    match d' with
    | Project_function_slot (move_from', move_to', closure') ->
      (Pos.eqb move_from move_from' && Pos.eqb move_to move_to')
      && simple_eqb vm closure closure'
    | _ -> false)

(** val static_const_eqb : name_map -> static_const -> static_const -> bool **)

let static_const_eqb vm const0 const' =
  match const0 with
  | Static_const_set_of_closures (code0, slots) -> (
    match const' with
    | Static_const_set_of_closures (code', slots') ->
      Pos.eqb code0 code' && value_slots_eqb vm slots slots'
    | _ -> false)
  | Static_const_closure_of_set (slot, symbols, decls, slots) -> (
    match const' with
    | Static_const_closure_of_set (slot', symbols', decls', slots') ->
      ((Pos.eqb slot slot' && names_eqb symbols symbols')
      && names_eqb decls decls')
      && value_slots_eqb vm slots slots'
    | _ -> false)
  | Static_const_block (tag, fields) -> (
    match const' with
    | Static_const_block (tag', fields') ->
      Z.eqb tag tag' && simples_eqb vm fields fields'
    | _ -> false)
  | Static_const_boxed_number contents -> (
    match const' with
    | Static_const_boxed_number contents' -> simple_eqb vm contents contents'
    | _ -> false)
  | Static_const_immutable_string bytes -> (
    match const' with
    | Static_const_immutable_string bytes' -> forall2b Z.eqb bytes bytes'
    | _ -> false)
  | Static_const_immutable_value_array elements -> (
    match const' with
    | Static_const_immutable_value_array elements' ->
      simples_eqb vm elements elements'
    | _ -> false)

(** val call_kind_eqb : call_kind -> call_kind -> bool **)

let call_kind_eqb kind kind' =
  match kind with
  | Direct code0 -> (
    match kind' with Direct code' -> Pos.eqb code0 code' | _ -> false)
  | Indirect_unknown_arity -> (
    match kind' with Indirect_unknown_arity -> true | _ -> false)
  | Indirect_known_arity code_ids0 -> (
    match code_ids0 with
    | Some code_ids -> (
      match kind' with
      | Indirect_known_arity code_ids1 -> (
        match code_ids1 with
        | Some code_ids' -> forall2b Pos.eqb code_ids code_ids'
        | None -> false)
      | _ -> false)
    | None -> (
      match kind' with
      | Indirect_known_arity code_ids -> (
        match code_ids with Some _ -> false | None -> true)
      | _ -> false))

(** val raise_kind_eqb : raise_kind option -> raise_kind option -> bool **)

let raise_kind_eqb kind kind' =
  match kind with
  | Some r -> (
    match r with
    | Regular -> (
      match kind' with
      | Some r0 -> ( match r0 with Regular -> true | _ -> false)
      | None -> false)
    | Reraise -> (
      match kind' with
      | Some r0 -> ( match r0 with Reraise -> true | _ -> false)
      | None -> false)
    | No_trace -> (
      match kind' with
      | Some r0 -> ( match r0 with No_trace -> true | _ -> false)
      | None -> false))
  | None -> ( match kind' with Some _ -> false | None -> true)

(** val trap_action_eqb : name_map -> trap_action option -> trap_action option
    -> bool **)

let trap_action_eqb km action action' =
  match action with
  | Some t -> (
    match t with
    | Push k -> (
      match action' with
      | Some t0 -> (
        match t0 with Push k' -> names_match km k k' | Pop (_, _) -> false)
      | None -> false)
    | Pop (k, kind) -> (
      match action' with
      | Some t0 -> (
        match t0 with
        | Push _ -> false
        | Pop (k', kind') -> names_match km k k' && raise_kind_eqb kind kind')
      | None -> false))
  | None -> ( match action' with Some _ -> false | None -> true)

(** val apply_cont_eqb : name_map -> name_map -> apply_cont -> apply_cont ->
    bool **)

let apply_cont_eqb vm km ac ac' =
  (names_match km ac.ac_continuation ac'.ac_continuation
  && simples_eqb vm ac.ac_args ac'.ac_args)
  && trap_action_eqb km ac.ac_trap_action ac'.ac_trap_action

(** val apply_eqb : name_map -> name_map -> apply_expr -> apply_expr -> bool **)

let apply_eqb vm km ap ap' =
  (((((((simple_eqb vm ap.ap_callee ap'.ap_callee
        && call_kind_eqb ap.ap_call_kind ap'.ap_call_kind)
       && simples_eqb vm ap.ap_args ap'.ap_args)
      && names_match km ap.ap_continuation ap'.ap_continuation)
     && names_match km ap.ap_exn_continuation ap'.ap_exn_continuation)
    && simples_eqb vm (exn_extra_simples ap) (exn_extra_simples ap'))
   && kinds_eqb (map snd ap.ap_exn_extra_args) (map snd ap'.ap_exn_extra_args))
  && arity_eqb ap.ap_args_arity ap'.ap_args_arity)
  && arity_eqb ap.ap_return_arity ap'.ap_return_arity

(** val expr_eqb : name_map -> name_map -> expr -> expr -> bool **)

let rec expr_eqb vm km e e' =
  match e with
  | Let (x, kind, d, body) -> (
    match e' with
    | Let (x', kind', d', body') ->
      (kind_eqb kind kind' && named_eqb vm d d')
      && expr_eqb ((x, x') :: vm) km body body'
    | _ -> false)
  | Let_static (bindings, body) -> (
    match e' with
    | Let_static (bindings', body') ->
      forall2b (binding_eqb vm) bindings bindings' && expr_eqb vm km body body'
    | _ -> false)
  | Let_cont_non_recursive (k, h, body) -> (
    match e' with
    | Let_cont_non_recursive (k', h', body') ->
      handler_eqb vm km h h' && expr_eqb vm ((k, k') :: km) body body'
    | _ -> false)
  | Let_cont_recursive (handlers, body) -> (
    match e' with
    | Let_cont_recursive (handlers', body') -> (
      match combine_names (map fst handlers) (map fst handlers') km with
      | Some km' ->
        forall2b
          (fun pat pat0 ->
            let _, h = pat in
            let _, h' = pat0 in
            handler_eqb vm km' h h')
          handlers handlers'
        && expr_eqb vm km' body body'
      | None -> false)
    | _ -> false)
  | Apply ap -> (
    match e' with Apply ap' -> apply_eqb vm km ap ap' | _ -> false)
  | Apply_cont ac -> (
    match e' with Apply_cont ac' -> apply_cont_eqb vm km ac ac' | _ -> false)
  | Switch (scrutinee, arms) -> (
    match e' with
    | Switch (scrutinee', arms') ->
      simple_eqb vm scrutinee scrutinee'
      && forall2b
           (fun arm arm' ->
             Z.eqb (fst arm) (fst arm')
             && apply_cont_eqb vm km (snd arm) (snd arm'))
           arms arms'
    | _ -> false)
  | Invalid -> ( match e' with Invalid -> true | _ -> false)

(** val handler_eqb : name_map -> name_map -> continuation_handler ->
    continuation_handler -> bool **)

and handler_eqb vm km h h' =
  let (Handler (params, metadata, handler)) = h in
  let (Handler (params', metadata', handler')) = h' in
  match bind_names params params' vm with
  | Some vm' ->
    handler_metadata_eqb metadata metadata' && expr_eqb vm' km handler handler'
  | None -> false

(** val binding_eqb : name_map -> static_binding -> static_binding -> bool **)

and binding_eqb vm binding binding' =
  match binding with
  | Bind_code (id, c) -> (
    match binding' with
    | Bind_code (id', c') -> Pos.eqb id id' && code_eqb c c'
    | _ -> false)
  | Bind_deleted_code id -> (
    match binding' with Bind_deleted_code id' -> Pos.eqb id id' | _ -> false)
  | Bind_symbol (s, const0) -> (
    match binding' with
    | Bind_symbol (s', const') ->
      Pos.eqb s s' && static_const_eqb vm const0 const'
    | _ -> false)

(** val code_eqb : code -> code -> bool **)

and code_eqb c c' =
  let (Code (params, my_closure, ret, exn, metadata, body)) = c in
  let (Code (params', my_closure', ret', exn', metadata', body')) = c' in
  match bind_names params params' ((my_closure, my_closure') :: []) with
  | Some vm ->
    code_metadata_eqb metadata metadata'
    && expr_eqb vm [ret, ret'; exn, exn'] body body'
  | None -> false
