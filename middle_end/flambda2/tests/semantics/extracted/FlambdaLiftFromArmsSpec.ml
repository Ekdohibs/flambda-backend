open FlambdaHeapFreeSpec
open FlambdaLiftFromCodeSpec
open FlambdaLiftSpec
open FlambdaModel
open FlambdaWellFormedSpec

(** val lift_to_head : (path * lift_to) list -> expr -> expr **)

let lift_to_head lets e =
  let bindings = once_each [] (lifted_out_bindings lets root e) in
  match bindings with
  | [] -> e
  | _ :: _ -> Let_static (bindings, lift_from_code [] lets root e)

(** val lift_to_head_unit : unit_interface -> (path * lift_to) list -> expr ->
    expr option **)

let lift_to_head_unit u lets e =
  let lifted = lift_to_head lets e in
  if
    (exits_plain_knowing [] [] e && expr_free_of_heap e)
    && well_formed_unit u lifted
  then Some lifted
  else None
