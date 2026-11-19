open BinInt
open BinPos
open Bool
open Datatypes
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaModel
open FlambdaSubkindOrderSpec
open ListDef

(** val handler_metadata_says_at_least : handler_metadata -> handler_metadata ->
    bool **)

let handler_metadata_says_at_least m m' =
  (kinds_eqb m.hm_params_kinds m'.hm_params_kinds
  && params_say_at_least m.hm_params_subkinds m'.hm_params_subkinds)
  && eqb m.hm_is_exn_handler m'.hm_is_exn_handler

(** val code_metadata_says_at_least : code_metadata -> code_metadata -> bool **)

let code_metadata_says_at_least m m' =
  ((((arity_eqb m.cm_params_arity m'.cm_params_arity
     && params_say_at_least m.cm_params_subkinds m'.cm_params_subkinds)
    && arity_eqb m.cm_result_arity m'.cm_result_arity)
   && params_say_at_least m.cm_result_subkinds m'.cm_result_subkinds)
  && eqb m.cm_is_tupled m'.cm_is_tupled)
  && eqb m.cm_is_stub m'.cm_is_stub

(** val expr_says_at_least : name_map -> name_map -> expr -> expr -> bool **)

let rec expr_says_at_least vm km e e' =
  match e with
  | Let (x, kind, d, body) -> (
    match e' with
    | Let (x', kind', d', body') ->
      (kind_eqb kind kind' && named_eqb vm d d')
      && expr_says_at_least ((x, x') :: vm) km body body'
    | _ -> false)
  | Let_static (bindings, body) -> (
    match e' with
    | Let_static (bindings', body') ->
      forall2b (binding_says_at_least vm) bindings bindings'
      && expr_says_at_least vm km body body'
    | _ -> false)
  | Let_cont_non_recursive (k, h, body) -> (
    match e' with
    | Let_cont_non_recursive (k', h', body') ->
      handler_says_at_least vm km h h'
      && expr_says_at_least vm ((k, k') :: km) body body'
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
            handler_says_at_least vm km' h h')
          handlers handlers'
        && expr_says_at_least vm km' body body'
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

(** val handler_says_at_least : name_map -> name_map -> continuation_handler ->
    continuation_handler -> bool **)

and handler_says_at_least vm km h h' =
  let (Handler (params, metadata, handler)) = h in
  let (Handler (params', metadata', handler')) = h' in
  match bind_names params params' vm with
  | Some vm' ->
    handler_metadata_says_at_least metadata metadata'
    && expr_says_at_least vm' km handler handler'
  | None -> false

(** val binding_says_at_least : name_map -> static_binding -> static_binding ->
    bool **)

and binding_says_at_least vm binding binding' =
  match binding with
  | Bind_code (id, c) -> (
    match binding' with
    | Bind_code (id', c') -> Pos.eqb id id' && code_says_at_least c c'
    | _ -> false)
  | Bind_deleted_code id -> (
    match binding' with Bind_deleted_code id' -> Pos.eqb id id' | _ -> false)
  | Bind_symbol (s, const) -> (
    match binding' with
    | Bind_symbol (s', const') ->
      Pos.eqb s s' && static_const_eqb vm const const'
    | _ -> false)

(** val code_says_at_least : code -> code -> bool **)

and code_says_at_least c c' =
  let (Code (params, my_closure, ret, exn, metadata, body)) = c in
  let (Code (params', my_closure', ret', exn', metadata', body')) = c' in
  match bind_names params params' ((my_closure, my_closure') :: []) with
  | Some vm ->
    code_metadata_says_at_least metadata metadata'
    && expr_says_at_least vm [ret, ret'; exn, exn'] body body'
  | None -> false
