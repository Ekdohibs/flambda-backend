open BinNums
open FlambdaModel
open FlambdaPrimitiveModel

(** val same_answer : equality_comparison -> const **)

let same_answer op =
  Naked_immediate (match op with Phys_eq -> Zpos Coq_xH | Phys_neq -> Z0)

(** val different_answer : equality_comparison -> const **)

let different_answer op =
  Naked_immediate (match op with Phys_eq -> Z0 | Phys_neq -> Zpos Coq_xH)

(** val fold_same_comparison : named -> named **)

let fold_same_comparison d =
  match d with
  | Prim (p, args) -> (
    match p with
    | Binary p0 -> (
      match p0 with
      | Phys_equal op -> (
        match args with
        | [] -> d
        | a :: l -> (
          match l with
          | [] -> d
          | b :: l0 -> (
            match l0 with
            | [] -> (
              if simple_eq_dec a b
              then Simple (Const (same_answer op))
              else
                match a with
                | Symbol _ -> (
                  match b with
                  | Symbol _ -> Simple (Const (different_answer op))
                  | _ -> d)
                | _ -> d)
            | _ :: _ -> d)))
      | _ -> d)
    | _ -> d)
  | _ -> d

(** val fold_same_comparisons : expr -> expr **)

let rec fold_same_comparisons e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, fold_same_comparison d, fold_same_comparisons body)
  | Let_static (bindings, body) ->
    Let_static (bindings, fold_same_comparisons body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, fold_same_comparisons handler),
        fold_same_comparisons body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, fold_same_comparisons body)
  | _ -> e
