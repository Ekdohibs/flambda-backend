open BinNums
open Datatypes
open FlambdaBindersSpec
open FlambdaCompareSpec
open FlambdaFreshQuantifierSpec
open FlambdaModel
open FlambdaNominalSpec
open List
open ListDef
open Nat

(** val binders : expr -> nat **)

let rec binders = function
  | Let (_, _, _, body) -> S (binders body)
  | Let_static (bindings, body) ->
    add (list_sum (map binders_in_binding bindings)) (binders body)
  | Let_cont_non_recursive (_, h, body) ->
    add (binders_in_handler h) (S (binders body))
  | Let_cont_recursive (handlers, body) ->
    add
      (add (length handlers)
         (list_sum (map (fun kh -> binders_in_handler (snd kh)) handlers)))
      (binders body)
  | _ -> O

(** val binders_in_handler : continuation_handler -> nat **)

and binders_in_handler = function
  | Handler (params, _, handler) -> add (length params) (binders handler)

(** val binders_in_binding : static_binding -> nat **)

and binders_in_binding = function
  | Bind_code (_, c) -> binders_in_code c
  | _ -> O

(** val binders_in_code : code -> nat **)

and binders_in_code = function
  | Code (params, _, _, _, _, body) ->
    S (add (length params) (S (S (binders body))))

(** val bind_variable : renaming -> variable -> variable -> renaming **)

let bind_variable r x x' =
  { variable_permutation = (rename_variable r x, x') :: r.variable_permutation;
    continuation_permutation = r.continuation_permutation
  }

(** val bind_continuation : renaming -> continuation -> continuation -> renaming
    **)

let bind_continuation r k k' =
  { variable_permutation = r.variable_permutation;
    continuation_permutation =
      (rename_continuation r k, k') :: r.continuation_permutation
  }

(** val new_variable : renaming -> positive list -> variable -> variable **)

let new_variable r news x = hd (rename_variable r x) news

(** val new_continuation : renaming -> positive list -> continuation ->
    continuation **)

let new_continuation r news k = hd (rename_continuation r k) news

(** val bind_variables : renaming -> positive list -> variable list -> (positive
    list * variable list) * renaming **)

let rec bind_variables r news = function
  | [] -> (news, []), r
  | x :: xs0 ->
    let x' = new_variable r news x in
    let p, r' = bind_variables (bind_variable r x x') (tl news) xs0 in
    let rest, xs' = p in
    (rest, x' :: xs'), r'

(** val bind_continuations : renaming -> positive list -> continuation list ->
    (positive list * continuation list) * renaming **)

let rec bind_continuations r news = function
  | [] -> (news, []), r
  | k :: ks0 ->
    let k' = new_continuation r news k in
    let p, r' = bind_continuations (bind_continuation r k k') (tl news) ks0 in
    let rest, ks' = p in
    (rest, k' :: ks'), r'

(** val thread : (positive list -> 'a1 -> positive list * 'a2) -> positive list
    -> 'a1 list -> positive list * 'a2 list **)

let rec thread f news = function
  | [] -> news, []
  | a :: l0 ->
    let rest, b = f news a in
    let rest', bs = thread f rest l0 in
    rest', b :: bs

(** val apart : renaming -> positive list -> expr -> positive list * expr **)

let rec apart r news = function
  | Let (x, kind, d, body) ->
    let x' = new_variable r news x in
    let rest, body' = apart (bind_variable r x x') (tl news) body in
    rest, Let (x', kind, rename_named r d, body')
  | Let_static (bindings, body) ->
    let rest, bindings' = thread (apart_binding r) news bindings in
    let rest', body' = apart r rest body in
    rest', Let_static (bindings', body')
  | Let_cont_non_recursive (k, h, body) ->
    let rest, h' = apart_handler r news h in
    let k' = new_continuation r rest k in
    let rest', body' = apart (bind_continuation r k k') (tl rest) body in
    rest', Let_cont_non_recursive (k', h', body')
  | Let_cont_recursive (handlers, body) ->
    let p, r' = bind_continuations r news (map fst handlers) in
    let rest, ks' = p in
    let rest', hs' =
      thread (fun news0 kh -> apart_handler r' news0 (snd kh)) rest handlers
    in
    let rest'', body' = apart r' rest' body in
    rest'', Let_cont_recursive (combine ks' hs', body')
  | Apply ap -> news, Apply (rename_apply r ap)
  | Apply_cont ac -> news, Apply_cont (rename_apply_cont r ac)
  | Switch (scrutinee, arms) ->
    ( news,
      Switch
        ( rename_simple r scrutinee,
          map (fun arm -> fst arm, rename_apply_cont r (snd arm)) arms ) )
  | Invalid -> news, Invalid

(** val apart_handler : renaming -> positive list -> continuation_handler ->
    positive list * continuation_handler **)

and apart_handler r news = function
  | Handler (params, metadata, handler) ->
    let p, r' = bind_variables r news params in
    let rest, params' = p in
    let rest', handler' = apart r' rest handler in
    rest', Handler (params', metadata, handler')

(** val apart_binding : renaming -> positive list -> static_binding -> positive
    list * static_binding **)

and apart_binding r news binding =
  match binding with
  | Bind_code (id, c) ->
    let rest, c' = apart_code r news c in
    rest, Bind_code (id, c')
  | Bind_deleted_code _ -> news, binding
  | Bind_symbol (s, const) -> news, Bind_symbol (s, rename_static_const r const)

(** val apart_code : renaming -> positive list -> code -> positive list * code **)

and apart_code r news = function
  | Code
      (params, my_closure, return_continuation, exn_continuation, metadata, body)
    ->
    let my_closure' = new_variable r news my_closure in
    let r0 = bind_variable r my_closure my_closure' in
    let p, r1 = bind_variables r0 (tl news) params in
    let rest, params' = p in
    let return_continuation' = new_continuation r1 rest return_continuation in
    let r2 = bind_continuation r1 return_continuation return_continuation' in
    let exn_continuation' = new_continuation r2 (tl rest) exn_continuation in
    let r3 = bind_continuation r2 exn_continuation exn_continuation' in
    let rest', body' = apart r3 (tl (tl rest)) body in
    ( rest',
      Code
        ( params',
          my_closure',
          return_continuation',
          exn_continuation',
          metadata,
          body' ) )

(** val canonical_avoiding : continuation list -> expr -> expr **)

let canonical_avoiding avoid e =
  snd
    (apart identity_renaming
       (fresh_names (binders e)
          (app avoid
             (app (free_variables_of_expr e) (free_continuations_of_expr e))))
       e)

(** val rename_apart_avoiding : continuation list -> expr -> expr **)

let rename_apart_avoiding avoid e =
  let e' = canonical_avoiding avoid e in
  if expr_eqb [] [] e e' then e' else e
