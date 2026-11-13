open FlambdaCopyCodeSpec
open FlambdaLiftAroundSpec
open FlambdaLiftFromArmsSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaModel
open FlambdaMyClosureSpec
open FlambdaUnusedHeadStaticsSpec
open FlambdaWellFormedSpec

(** val lift_to_head_unit_observed : unit_interface -> (path * lift_to) list ->
    expr -> expr option **)

let lift_to_head_unit_observed u lets e =
  let lifted = lift_to_head lets e in
  if well_formed_unit u lifted then Some lifted else None

(** val lift_from_code_unit_observed : unit_interface -> lets_in_code -> expr ->
    expr option **)

let lift_from_code_unit_observed u k e =
  let lifted = lift_from_code k [] root e in
  if well_formed_unit u lifted then Some lifted else None

(** val lift_unit_observed : lifting -> expr -> expr option **)

let lift_unit_observed l e =
  let lifted =
    bind_my_closures l.lifted_my_closures (lift_expr l.lifted_lets root e)
  in
  if
    well_formed_unit l.lifting_unit lifted
    && only_closures_of l.lifted_my_closures lifted
  then Some lifted
  else None

(** val lift_around_unit_observed : unit_interface -> (path * lift_to) list ->
    (path * path list) list -> expr -> expr option **)

let lift_around_unit_observed u lets places e =
  let groups = groups_of lets places e in
  let lifted = lift_around lets groups root e in
  if around_ok lets groups [] root e && well_formed_unit u lifted
  then Some lifted
  else None

(** val to_head_observed : lifting -> expr -> expr **)

let to_head_observed l e =
  match lift_to_head_unit_observed l.lifting_unit l.lifted_to_head e with
  | Some e' -> e'
  | None -> e

(** val out_of_code_observed : lifting -> expr -> expr **)

let out_of_code_observed l e =
  match lift_from_code_unit_observed l.lifting_unit l.lifted_out_of_code e with
  | Some e' -> e'
  | None -> e

(** val lets_observed : lifting -> expr -> expr **)

let lets_observed l e =
  match lift_unit_observed l e with Some e' -> e' | None -> e

(** val around_observed : lifting -> expr -> expr **)

let around_observed l e =
  match
    lift_around_unit_observed l.lifting_unit l.lifted_around l.groups_put e
  with
  | Some e' -> e'
  | None -> e

(** val liftings_observed : lifting -> expr -> expr **)

let liftings_observed l e =
  around_observed l
    (lets_observed l (out_of_code_observed l (to_head_observed l e)))

(** val with_lifting_observed : lifting -> expr -> expr **)

let with_lifting_observed l e =
  with_copies l.code_copied
    (add_head_tables l.tables_put (liftings_observed l e))
