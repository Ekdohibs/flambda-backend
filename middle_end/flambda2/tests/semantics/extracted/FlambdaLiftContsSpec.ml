open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaGiveScrutineeSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaSemanticsSpec
open FlambdaUnusedContsSpec
open List
open ListDef
open PeanoNat

(** val give_to : continuation -> variable list -> apply_cont -> apply_cont **)

let give_to j ps ac =
  if Pos.eqb ac.ac_continuation j
  then
    { ac_continuation = ac.ac_continuation;
      ac_args = app (map (fun x -> Var x) ps) ac.ac_args;
      ac_trap_action = ac.ac_trap_action
    }
  else ac

(** val give : continuation -> variable list -> expr -> expr **)

let rec give j ps e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, give j ps body)
  | Let_static (bindings, body) -> Let_static (bindings, give j ps body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, give_in_handler j ps h, give j ps body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, give_in_handler j ps h)
          handlers,
        give j ps body )
  | Apply_cont ac -> Apply_cont (give_to j ps ac)
  | Switch (scrutinee, arms) ->
    Switch
      ( scrutinee,
        map
          (fun pat ->
            let n, ac = pat in
            n, give_to j ps ac)
          arms )
  | _ -> e

(** val give_in_handler : continuation -> variable list -> continuation_handler
    -> continuation_handler **)

and give_in_handler j ps = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, give j ps handler)

(** val applies_only : continuation -> bool -> apply_cont -> bool **)

let applies_only j ok ac =
  (match ac.ac_trap_action with
    | Some action -> negb (Pos.eqb (trap_action_handler action) j)
    | None -> true)
  && (ok || negb (Pos.eqb ac.ac_continuation j))

(** val only_applied : continuation -> variable list -> bool -> expr -> bool **)

let rec only_applied j ps ok = function
  | Let (y, _, _, body) -> only_applied j ps (ok && negb (among y ps)) body
  | Let_static (_, body) -> only_applied j ps ok body
  | Let_cont_non_recursive (k, h, body) ->
    (negb (Pos.eqb k j) && handler_only_applied j ps ok h)
    && only_applied j ps ok body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k, h = pat in
        negb (Pos.eqb k j) && handler_only_applied j ps ok h)
      handlers
    && only_applied j ps ok body
  | Apply ap ->
    (match ps with [] -> true | _ :: _ -> negb (Pos.eqb ap.ap_continuation j))
    && negb (Pos.eqb ap.ap_exn_continuation j)
  | Apply_cont ac -> applies_only j ok ac
  | Switch (_, arms) ->
    forallb
      (fun pat ->
        let _, ac = pat in
        applies_only j ok ac)
      arms
  | Invalid -> true

(** val handler_only_applied : continuation -> variable list -> bool ->
    continuation_handler -> bool **)

and handler_only_applied j ps ok = function
  | Handler (params, _, handler) ->
    only_applied j ps
      (ok && negb (existsb (fun p -> among p params) ps))
      handler

(** val read_params : variable list -> flambda_kind list -> expr -> (variable *
    flambda_kind) list **)

let rec read_params params kinds e =
  match params with
  | [] -> []
  | p :: params0 -> (
    match kinds with
    | [] -> []
    | kind :: kinds0 ->
      if var_free_b p e
      then (p, kind) :: read_params params0 kinds0 e
      else read_params params0 kinds0 e)

(** val give_all : (continuation * variable list) list -> expr -> expr **)

let give_all moved e =
  fold_right
    (fun pat e0 ->
      let j, ps = pat in
      give j ps e0)
    e moved

(** val lift_spine : expr -> variable list -> flambda_kind list -> (continuation
    * variable list) list -> expr -> ((continuation * continuation_handler) list
    * expr) option **)

let rec lift_spine body params kinds moved handler =
  match handler with
  | Let_cont_non_recursive (j, handler0, rest) ->
    let (Handler (q, metadata, h)) = handler0 in
    let h0 = give_all moved h in
    let read = read_params params kinds h0 in
    let ps = map fst read in
    if
      (((negb metadata.hm_is_exn_handler && distinct (app params q))
       && negb (cont_free_b j body))
      && negb (cont_free_b j h0))
      && only_applied j ps true (give_all moved rest)
    then
      match lift_spine body params kinds ((j, ps) :: moved) rest with
      | Some p ->
        let others, switch = p in
        Some
          ( ( j,
              Handler
                ( app ps q,
                  { hm_params_kinds =
                      app (map snd read) metadata.hm_params_kinds;
                    hm_is_exn_handler = false
                  },
                  h0 ) )
            :: others,
            switch )
      | None -> None
    else None
  | Switch (scrutinee, _) -> (
    match scrutinee with
    | Var p ->
      if among p params then Some ([], give_all moved handler) else None
    | _ -> None)
  | _ -> None

(** val position : variable -> variable list -> nat option **)

let rec position p = function
  | [] -> None
  | p' :: params0 ->
    if Pos.eqb p p'
    then Some O
    else option_map (fun x -> S x) (position p params0)

(** val tested : variable list -> expr -> nat option **)

let rec tested params = function
  | Let_cont_non_recursive (_, _, rest) -> tested params rest
  | Switch (scrutinee, _) -> (
    match scrutinee with Var p -> position p params | _ -> None)
  | _ -> None

(** val gives_immediate : continuation -> nat -> apply_cont -> bool **)

let gives_immediate k i ac =
  (match ac.ac_trap_action with
    | Some action -> negb (Pos.eqb (trap_action_handler action) k)
    | None -> true)
  &&
  if Pos.eqb ac.ac_continuation k
  then
    match nth_error ac.ac_args i with
    | Some s -> (
      match s with
      | Const c -> ( match c with Naked_immediate _ -> true | _ -> false)
      | _ -> false)
    | None -> false
  else true

(** val all_give_immediate : continuation -> nat -> expr -> bool **)

let rec all_give_immediate k i = function
  | Let (_, _, _, body) -> all_give_immediate k i body
  | Let_static (_, body) -> all_give_immediate k i body
  | Let_cont_non_recursive (k', h, body) ->
    (negb (Pos.eqb k' k) && handler_all_give_immediate k i h)
    && all_give_immediate k i body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let k', h = pat in
        negb (Pos.eqb k' k) && handler_all_give_immediate k i h)
      handlers
    && all_give_immediate k i body
  | Apply ap ->
    negb (Pos.eqb ap.ap_continuation k)
    && negb (Pos.eqb ap.ap_exn_continuation k)
  | Apply_cont ac -> gives_immediate k i ac
  | Switch (_, arms) ->
    forallb
      (fun pat ->
        let _, ac = pat in
        gives_immediate k i ac)
      arms
  | Invalid -> true

(** val handler_all_give_immediate : continuation -> nat -> continuation_handler
    -> bool **)

and handler_all_give_immediate k i = function
  | Handler (_, _, handler) -> all_give_immediate k i handler

(** val tested_through : expr -> variable option **)

let rec tested_through = function
  | Let (_, _, _, body) -> tested_through body
  | Let_static (_, body) -> tested_through body
  | Let_cont_non_recursive (_, _, body) -> tested_through body
  | Switch (scrutinee, _) -> (
    match scrutinee with Var x -> Some x | _ -> None)
  | _ -> None

(** val arm_through : continuation -> expr -> coq_Z option **)

let rec arm_through k' = function
  | Let (_, _, _, body0) -> arm_through k' body0
  | Let_static (_, body0) -> arm_through k' body0
  | Let_cont_non_recursive (k'', h, body0) ->
    if Pos.eqb k'' k' || cont_free_in_handler_b k' h
    then None
    else arm_through k' body0
  | Switch (_, arms) -> (
    match first_arm_of k' arms with
    | Some c -> if only_arm k' c arms then Some c else None
    | None -> None)
  | _ -> None

(** val facts_in : continuation -> expr -> (variable * coq_Z) list -> (variable
    * coq_Z) list **)

let facts_in k' body known =
  match tested_through body with
  | Some x -> (
    match arm_through k' body with Some c -> (x, c) :: known | None -> known)
  | None -> known

(** val forgetting : variable list -> (variable * coq_Z) list -> (variable *
    coq_Z) list **)

let forgetting ys known =
  filter
    (fun pat ->
      let y, _ = pat in
      negb (among y ys))
    known

(** val in_arm : simple -> coq_Z -> (variable * coq_Z) list -> (variable *
    coq_Z) list **)

let in_arm scrutinee d known =
  match scrutinee with Var x -> (x, d) :: known | _ -> known

(** val keep_known : continuation -> nat -> (variable * coq_Z) list ->
    apply_cont -> variable list -> variable list **)

let keep_known k i known ac cands =
  if Pos.eqb ac.ac_continuation k
  then
    match nth_error ac.ac_args i with
    | Some s -> (
      match s with
      | Const c0 -> (
        match c0 with
        | Naked_immediate c ->
          filter
            (fun x ->
              existsb
                (fun pat ->
                  let y, d = pat in
                  Pos.eqb y x && Z.eqb d c)
                known)
            cands
        | _ -> [])
      | _ -> [])
    | None -> []
  else cands

(** val still_known : continuation -> nat -> (variable * coq_Z) list -> expr ->
    variable list -> variable list **)

let rec still_known k i known e cands =
  match e with
  | Let (y, _, _, body) ->
    still_known k i (forgetting (y :: []) known) body cands
  | Let_static (_, body) -> still_known k i known body cands
  | Let_cont_non_recursive (k', h, body) ->
    still_known k i known body
      (handler_still_known k i (facts_in k' body known) h cands)
  | Let_cont_recursive (handlers, body) ->
    still_known k i known body
      (fold_right
         (fun pat cands0 ->
           let _, h = pat in
           handler_still_known k i known h cands0)
         cands handlers)
  | Apply_cont ac -> keep_known k i known ac cands
  | Switch (scrutinee, arms) ->
    fold_right
      (fun pat cands0 ->
        let d, ac = pat in
        keep_known k i (in_arm scrutinee d known) ac cands0)
      cands arms
  | _ -> cands

(** val handler_still_known : continuation -> nat -> (variable * coq_Z) list ->
    continuation_handler -> variable list -> variable list **)

and handler_still_known k i known h cands =
  let (Handler (params, _, handler)) = h in
  still_known k i (forgetting params known) handler cands

(** val scrutinees : expr -> variable list **)

let rec scrutinees = function
  | Let (_, _, _, body) -> scrutinees body
  | Let_static (_, body) -> scrutinees body
  | Let_cont_non_recursive (_, h, body) ->
    app (handler_scrutinees h) (scrutinees body)
  | Let_cont_recursive (handlers, body) ->
    app
      (flat_map
         (fun pat ->
           let _, h = pat in
           handler_scrutinees h)
         handlers)
      (scrutinees body)
  | Switch (scrutinee, _) -> (
    match scrutinee with Var x -> x :: [] | _ -> [])
  | _ -> []

(** val handler_scrutinees : continuation_handler -> variable list **)

and handler_scrutinees = function
  | Handler (_, _, handler) -> scrutinees handler

(** val number_of_a_scrutinee : continuation -> nat -> expr -> bool **)

let number_of_a_scrutinee k i body =
  match still_known k i [] body (scrutinees body) with
  | [] -> false
  | _ :: _ -> true

(** val only_read_given : variable list -> simple list -> apply_cont -> bool **)

let rec only_read_given params args target =
  match params with
  | [] -> true
  | p :: params0 -> (
    match args with
    | [] -> true
    | a :: args0 ->
      (match a with
        | Var _ ->
          existsb
            (fun b -> match b with Var y -> Pos.eqb y p | _ -> false)
            target.ac_args
        | _ -> true)
      && only_read_given params0 args0 target)

(** val gives_what_is_read : continuation -> nat -> variable list -> (coq_Z *
    apply_cont) list -> apply_cont -> bool **)

let gives_what_is_read k i params arms ac =
  if Pos.eqb ac.ac_continuation k
  then
    match nth_error ac.ac_args i with
    | Some s -> (
      match s with
      | Const c -> (
        match c with
        | Naked_immediate n -> (
          match find_arm n arms with
          | Some target -> only_read_given params ac.ac_args target
          | None -> true)
        | _ -> true)
      | _ -> true)
    | None -> true
  else true

(** val all_give_read : continuation -> nat -> variable list -> (coq_Z *
    apply_cont) list -> expr -> bool **)

let rec all_give_read k i params arms = function
  | Let (_, _, _, body) -> all_give_read k i params arms body
  | Let_static (_, body) -> all_give_read k i params arms body
  | Let_cont_non_recursive (_, h, body) ->
    handler_all_give_read k i params arms h
    && all_give_read k i params arms body
  | Let_cont_recursive (handlers, body) ->
    forallb
      (fun pat ->
        let _, h = pat in
        handler_all_give_read k i params arms h)
      handlers
    && all_give_read k i params arms body
  | Apply_cont ac -> gives_what_is_read k i params arms ac
  | Switch (_, arms') ->
    forallb
      (fun pat ->
        let _, ac = pat in
        gives_what_is_read k i params arms ac)
      arms'
  | _ -> true

(** val handler_all_give_read : continuation -> nat -> variable list -> (coq_Z *
    apply_cont) list -> continuation_handler -> bool **)

and handler_all_give_read k i params arms = function
  | Handler (_, _, handler) -> all_give_read k i params arms handler

(** val read_where_given : continuation -> nat -> variable list -> expr -> expr
    -> bool **)

let read_where_given k i params switch body =
  match switch with
  | Switch (_, arms) -> all_give_read k i params arms body
  | _ -> true

(** val bind_lifted : (continuation * continuation_handler) list -> expr -> expr
    **)

let bind_lifted lifted e =
  fold_right
    (fun pat e0 ->
      let j, hj = pat in
      Let_cont_non_recursive (j, hj, e0))
    e lifted

(** val lift_conts : continuation -> continuation_handler -> expr -> expr option
    **)

let lift_conts k h body =
  let (Handler (params, metadata, handler)) = h in
  match tested params handler with
  | Some i ->
    if
      (((negb metadata.hm_is_exn_handler && distinct params)
       && Nat.eqb (length metadata.hm_params_kinds) (length params))
      && all_give_immediate k i body)
      && negb (number_of_a_scrutinee k i body)
    then
      match lift_spine body params metadata.hm_params_kinds [] handler with
      | Some p -> (
        let l, switch = p in
        match l with
        | [] -> None
        | j :: others ->
          if read_where_given k i params switch body
          then
            Some
              (bind_lifted (j :: others)
                 (Let_cont_non_recursive
                    (k, Handler (params, metadata, switch), body)))
          else None)
      | None -> None
    else None
  | None -> None

(** val lift_handler_conts : expr -> expr **)

let rec lift_handler_conts e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, lift_handler_conts body)
  | Let_static (bindings, body) -> Let_static (bindings, lift_handler_conts body)
  | Let_cont_non_recursive (k, handler0, body) -> (
    let (Handler (params, metadata, handler)) = handler0 in
    let h = Handler (params, metadata, lift_handler_conts handler) in
    let body0 = lift_handler_conts body in
    match lift_conts k h body0 with
    | Some lifted -> lifted
    | None -> Let_cont_non_recursive (k, h, body0))
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, lift_handler_conts body)
  | _ -> e
