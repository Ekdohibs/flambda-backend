open Datatypes
open FlambdaBlockLoadSpec
open FlambdaKindModel
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open List

(** val surely_value : simple -> bool **)

let surely_value = function
  | Var _ -> false
  | Symbol _ -> true
  | Const c -> ( match c with Tagged_immediate _ -> true | _ -> false)

(** val surely_immediate : simple -> bool **)

let surely_immediate = function
  | Const c -> (
    match c with
    | Tagged_immediate i -> in_range (int_width Standard_tagged_immediate) i
    | _ -> false)
  | _ -> false

(** val of_known_block_surely : unary_primitive -> known_block -> simple option
    **)

let of_known_block_surely p block =
  match p with
  | Block_load (_, _, _) -> (
    match of_known_block p block with
    | Some contents -> if surely_value contents then Some contents else None
    | None -> None)
  | Is_int _ -> of_known_block p block
  | Get_tag -> of_known_block p block
  | _ -> None

(** val load_known_named_surely : known_blocks -> named -> named **)

let load_known_named_surely known d =
  match d with
  | Prim (p, args) -> (
    match unary_applied p args with
    | Some p0 -> (
      let p1, block = p0 in
      match known_block_of known block with
      | Some known_block0 -> (
        match of_known_block_surely p1 known_block0 with
        | Some result -> Simple result
        | None -> d)
      | None -> d)
    | None -> d)
  | _ -> d

(** val load_blocks_knowing_surely : known_blocks -> expr -> expr **)

let rec load_blocks_knowing_surely known e =
  match e with
  | Let (x, kind, d, body) ->
    Let
      ( x,
        kind,
        load_known_named_surely known d,
        load_blocks_knowing_surely (known_under_let known x d) body )
  | Let_static (bindings, body) ->
    Let_static
      ( bindings,
        load_blocks_knowing_surely (known_under_statics known bindings) body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        load_blocks_in_handler_surely known h,
        load_blocks_knowing_surely known body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, load_blocks_knowing_surely known body)
  | _ -> e

(** val load_blocks_in_handler_surely : known_blocks -> continuation_handler ->
    continuation_handler **)

and load_blocks_in_handler_surely known = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        load_blocks_knowing_surely (forget_vars params known) handler )

(** val load_known_blocks_surely : expr -> expr **)

let load_known_blocks_surely e = load_blocks_knowing_surely no_blocks e

(** val allocation_of_constants : named -> bool **)

let allocation_of_constants = function
  | Prim (p, args) -> (
    match p with Variadic _ -> forallb surely_immediate args | _ -> false)
  | _ -> false

(** val remove_constant_allocations : expr -> expr **)

let rec remove_constant_allocations e =
  match e with
  | Let (x, kind, d, body) ->
    let body' = remove_constant_allocations body in
    if
      (kind_eqb kind Value && allocation_of_constants d)
      && negb (var_free_b x body')
    then body'
    else Let (x, kind, d, body')
  | Let_static (bindings, body) ->
    Let_static (bindings, remove_constant_allocations body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, remove_constant_allocations handler),
        remove_constant_allocations body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, remove_constant_allocations body)
  | _ -> e
