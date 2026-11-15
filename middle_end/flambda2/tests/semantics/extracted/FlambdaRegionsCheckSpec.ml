open BinPos
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel

type lifetime =
  | Forever
  | Of_caller
  | Of_region of variable
  | Gone

(** val regions_below : variable -> variable list -> variable list **)

let rec regions_below r = function
  | [] -> []
  | x :: open1 -> if Pos.eqb x r then open1 else regions_below r open1

(** val in_order : variable list -> variable list -> bool **)

let rec in_order xs = function
  | [] -> ( match xs with [] -> true | _ :: _ -> false)
  | y :: ys' -> (
    match xs with
    | [] -> true
    | x :: xs' -> if Pos.eqb x y then in_order xs' ys' else in_order xs ys')

type handler_entry =
  { he_open : variable list;
    he_params : lifetime list
  }

type code_entry =
  { ce_params_local : bool list;
    ce_closure_local : bool;
    ce_returns_local : bool
  }

type certificate =
  { cert_handlers : (continuation * handler_entry) list;
    cert_code : (code_id * code_entry) list;
    cert_indirect_local : bool;
    cert_unknown_returns_local : bool;
    cert_partial_local : bool
  }

(** val region_at_jump : (variable * variable) list -> variable -> variable **)

let region_at_jump given r =
  match lookup r given with Some r' -> r' | None -> r

(** val region_params_of : variable list -> flambda_kind list -> variable list **)

let rec region_params_of params kinds =
  match params with
  | [] -> []
  | p :: params0 -> (
    match kinds with
    | [] -> []
    | kind :: kinds0 ->
      if kind_eqb kind Region
      then p :: region_params_of params0 kinds0
      else region_params_of params0 kinds0)
