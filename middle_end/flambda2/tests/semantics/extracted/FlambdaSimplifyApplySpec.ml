open Datatypes
open FlambdaMapSpec
open FlambdaModel
open FlambdaProvers
open FlambdaSimplifyPrimitivesSpec
open FlambdaTypeGrammar
open FlambdaTypingEnv
open FlambdaValueSpec
open List
open ListDef

(** val with_call_kind : apply_expr -> call_kind -> apply_expr **)

let with_call_kind ap kind =
  { ap_callee = ap.ap_callee;
    ap_call_kind = kind;
    ap_args = ap.ap_args;
    ap_continuation = ap.ap_continuation;
    ap_exn_continuation = ap.ap_exn_continuation;
    ap_exn_extra_args = ap.ap_exn_extra_args;
    ap_args_arity = ap.ap_args_arity;
    ap_return_arity = ap.ap_return_arity
  }

(** val with_callee_and_args : apply_expr -> simple -> simple list -> apply_expr
    **)

let with_callee_and_args ap callee args =
  { ap_callee = callee;
    ap_call_kind = ap.ap_call_kind;
    ap_args = args;
    ap_continuation = ap.ap_continuation;
    ap_exn_continuation = ap.ap_exn_continuation;
    ap_exn_extra_args = ap.ap_exn_extra_args;
    ap_args_arity = ap.ap_args_arity;
    ap_return_arity = ap.ap_return_arity
  }

(** val direct_call_kind : Typing_env.t -> apply_expr -> call_kind **)

let direct_call_kind tenv ap =
  match ap.ap_call_kind with
  | Indirect_unknown_arity -> (
    match simplify_simple tenv ap.ap_callee with
    | Some p -> (
      let ty, _ = p in
      match Provers.prove_single_closures_entry tenv ty with
      | Some t0 -> (
        match t0 with
        | Proof_of_property.Proved a ->
          let _, function_type = a in
          Direct function_type.function_type_code_id
        | Proof_of_property.Unknown -> Indirect_unknown_arity)
      | None -> Indirect_unknown_arity)
    | None -> Indirect_unknown_arity)
  | x -> x

(** val canonical_apply : Typing_env.t -> apply_expr -> apply_expr **)

let canonical_apply tenv ap =
  match simplify_simple tenv ap.ap_callee with
  | Some p -> (
    let _, callee = p in
    match simplify_simples tenv ap.ap_args with
    | Some answers -> with_callee_and_args ap callee (map snd answers)
    | None -> ap)
  | None -> ap

(** val simplify_apply : Typing_env.t -> apply_expr -> expr **)

let simplify_apply tenv ap =
  Apply (canonical_apply tenv (with_call_kind ap (direct_call_kind tenv ap)))

(** val closure_of_set_type : function_slot -> (function_slot * code_id) list ->
    value_slot list -> flambda_type list -> flambda_type **)

let closure_of_set_type slot function_decls slots tys =
  exactly_this_closure slot
    (map_values
       (fun code -> Or_unknown.Known { function_type_code_id = code })
       function_decls)
    [] (combine slots tys) Alloc_mode.Heap_or_local

(** val set_of_closures_type : code_id -> value_slot list -> flambda_type list
    -> flambda_type **)

let set_of_closures_type code slots tys =
  closure_of_set_type unnamed_function_slot
    ((unnamed_function_slot, code) :: [])
    slots tys
