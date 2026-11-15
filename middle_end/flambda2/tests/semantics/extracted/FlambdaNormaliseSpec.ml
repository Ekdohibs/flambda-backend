open BinNums
open BinPos
open Bool
open Datatypes
open FlambdaApplySpec
open FlambdaCompareSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNominalSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaSemanticsSpec
open FlambdaTupledSpec
open FlambdaValueSpec
open List
open ListDef
open PeanoNat

(** val simple_in_scope : variable list -> simple -> bool **)

let simple_in_scope vs = function Var x -> among x vs | _ -> true

(** val simples_in_scope : variable list -> simple list -> bool **)

let simples_in_scope vs args = forallb (simple_in_scope vs) args

(** val named_in_scope : variable list -> named -> bool **)

let named_in_scope vs = function
  | Simple s -> simple_in_scope vs s
  | Prim (_, args) -> simples_in_scope vs args
  | Set_of_closures (_, slots) ->
    forallb (fun slot -> simple_in_scope vs (snd slot)) slots
  | Closure_of_set (_, _, slots) ->
    forallb (fun slot -> simple_in_scope vs (snd slot)) slots
  | Project_value_slot (_, closure) -> simple_in_scope vs closure
  | Project_value_slot_from (_, _, closure) -> simple_in_scope vs closure
  | Project_function_slot (_, _, closure) -> simple_in_scope vs closure

(** val static_binding_in_scope : variable list -> static_binding -> bool **)

let static_binding_in_scope vs = function
  | Bind_symbol (_, const0) -> (
    match const0 with
    | Static_const_set_of_closures (_, slots) ->
      forallb (fun slot -> simple_in_scope vs (snd slot)) slots
    | Static_const_closure_of_set (_, _, _, slots) ->
      forallb (fun slot -> simple_in_scope vs (snd slot)) slots
    | Static_const_block (_, fields) -> simples_in_scope vs fields
    | Static_const_boxed_number contents -> simple_in_scope vs contents
    | Static_const_immutable_string _ -> true
    | Static_const_immutable_value_array elements ->
      simples_in_scope vs elements)
  | _ -> true

(** val apply_cont_in_scope : variable list -> continuation list -> apply_cont
    -> bool **)

let apply_cont_in_scope vs ks ac =
  all_among (apply_cont_continuations ac) ks && simples_in_scope vs ac.ac_args

(** val expr_in_scope : variable list -> continuation list -> expr -> bool **)

let rec expr_in_scope vs ks = function
  | Let (x, _, d, body) ->
    named_in_scope vs d && expr_in_scope (x :: vs) ks body
  | Let_static (bindings, body) ->
    forallb (static_binding_in_scope vs) bindings && expr_in_scope vs ks body
  | Let_cont_non_recursive (k, h, body) ->
    handler_in_scope vs ks h && expr_in_scope vs (k :: ks) body
  | Let_cont_recursive (handlers, body) ->
    let ks' = app (map fst handlers) ks in
    forallb
      (fun pat ->
        let _, h = pat in
        handler_in_scope vs ks' h)
      handlers
    && expr_in_scope vs ks' body
  | Apply ap ->
    (((simple_in_scope vs ap.ap_callee && simples_in_scope vs ap.ap_args)
     && among ap.ap_continuation ks)
    && among ap.ap_exn_continuation ks)
    && simples_in_scope vs (exn_extra_simples ap)
  | Apply_cont ac -> apply_cont_in_scope vs ks ac
  | Switch (scrutinee, arms) ->
    simple_in_scope vs scrutinee
    && forallb (fun arm -> apply_cont_in_scope vs ks (snd arm)) arms
  | Invalid -> true

(** val handler_in_scope : variable list -> continuation list ->
    continuation_handler -> bool **)

and handler_in_scope vs ks = function
  | Handler (params, _, handler) -> expr_in_scope (app params vs) ks handler

(** val inline_renaming : continuation -> continuation -> continuation ->
    continuation -> renaming **)

let inline_renaming ret exn k k_exn =
  { variable_permutation = [];
    continuation_permutation = [swap ret k exn, k_exn; ret, k]
  }

(** val var_in_simples : variable -> simple list -> bool **)

let var_in_simples x args =
  existsb (fun s -> match s with Var y -> Pos.eqb x y | _ -> false) args

(** val inlinable_b : apply_expr -> code -> bool **)

let inlinable_b ap c =
  ((((((((negb (code_is_tupled c)
         && expr_in_scope
              (code_my_closure c :: code_params c)
              [code_return_continuation c; code_exn_continuation c]
              (code_body c))
        && negb (Pos.eqb (code_return_continuation c) (code_exn_continuation c))
        )
       && negb (Pos.eqb ap.ap_continuation ap.ap_exn_continuation))
      && no_exn_extra_args ap)
     && negb (var_in_simples (code_my_closure c) ap.ap_args))
    && forallb
         (fun param -> negb (var_in_simples param ap.ap_args))
         (code_params c))
   && Nat.eqb (length (code_params c)) (length ap.ap_args))
  && Nat.eqb (length ap.ap_args_arity) (length (code_params_arity c)))
  && negb (binds_statics (code_body c))

(** val tupled_inlinable_b : apply_expr -> code -> bool **)

let tupled_inlinable_b ap c =
  ((((((((((code_is_tupled c
           && expr_in_scope
                (code_my_closure c :: code_params c)
                [code_return_continuation c; code_exn_continuation c]
                (code_body c))
          && negb
               (Pos.eqb (code_return_continuation c) (code_exn_continuation c))
          )
         && negb (Pos.eqb ap.ap_continuation ap.ap_exn_continuation))
        && no_exn_extra_args ap)
       && negb (var_in_simples (code_my_closure c) ap.ap_args))
      && forallb
           (fun param -> negb (var_in_simples param ap.ap_args))
           (code_params c))
     && Nat.eqb (length ap.ap_args) (S O))
    && arity_eqb ap.ap_args_arity (Singleton Value :: []))
   && arity_eqb (code_params_arity c)
        (arity_of_kinds (repeat Value (length (code_params c)))))
  && negb (Nat.eqb (length (code_params c)) O))
  && negb (binds_statics (code_body c))

(** val over_applied_b : apply_expr -> code -> bool **)

let over_applied_b ap c =
  ((((negb (code_is_tupled c)
     && Nat.ltb (length (code_params_arity c)) (length ap.ap_args_arity))
    && Nat.leb (cardinal_unarized (code_params_arity c)) (length ap.ap_args))
   && negb (Pos.eqb (code_return_continuation c) ap.ap_exn_continuation))
  && negb
       (var_in_simples (code_my_closure c)
          (skipn (cardinal_unarized (code_params_arity c)) ap.ap_args)))
  && negb (var_in_simples (code_my_closure c) (exn_extra_simples ap))

(** val split_over_application : code_id -> code -> apply_expr -> expr **)

let split_over_application called c ap =
  let size = cardinal_unarized (code_params_arity c) in
  let params = length (code_params_arity c) in
  let k = code_return_continuation c in
  let p = code_my_closure c in
  Let_cont_non_recursive
    ( k,
      Handler
        ( p :: [],
          make_handler_metadata (Value :: []),
          Apply
            { ap_callee = Var p;
              ap_call_kind = Indirect;
              ap_args = skipn size ap.ap_args;
              ap_continuation = ap.ap_continuation;
              ap_exn_continuation = ap.ap_exn_continuation;
              ap_exn_extra_args = ap.ap_exn_extra_args;
              ap_args_arity = skipn params ap.ap_args_arity;
              ap_return_arity = ap.ap_return_arity
            } ),
      Apply
        { ap_callee = ap.ap_callee;
          ap_call_kind = Direct called;
          ap_args = firstn size ap.ap_args;
          ap_continuation = k;
          ap_exn_continuation = ap.ap_exn_continuation;
          ap_exn_extra_args = ap.ap_exn_extra_args;
          ap_args_arity = code_params_arity c;
          ap_return_arity = code_result_arity c
        } )

(** val codes_of : static_binding list -> (code_id * code) list -> (code_id *
    code) list **)

let rec codes_of bindings known =
  match bindings with
  | [] -> known
  | s :: bindings0 -> (
    match s with
    | Bind_code (id, c) -> codes_of bindings0 ((id, c) :: known)
    | _ -> codes_of bindings0 known)

(** val inline_apply : code_id list -> (code_id * code) list -> apply_expr ->
    expr **)

let inline_apply kept known ap =
  match ap.ap_call_kind with
  | Direct called -> (
    match lookup called known with
    | Some c ->
      if negb (among called kept) && inlinable_b ap c
      then
        inlined_body c
          (inline_renaming
             (code_return_continuation c)
             (code_exn_continuation c) ap.ap_continuation ap.ap_exn_continuation)
          ap
      else if negb (among called kept) && tupled_inlinable_b ap c
      then
        inlined_tupled_call c
          (inline_renaming
             (code_return_continuation c)
             (code_exn_continuation c) ap.ap_continuation ap.ap_exn_continuation)
          ap
      else if over_applied_b ap c
      then split_over_application called c ap
      else Apply ap
    | None -> Apply ap)
  | Indirect -> Apply ap

(** val inline_calls : code_id list -> (code_id * code) list -> expr -> expr **)

let rec inline_calls kept known e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, inline_calls kept known body)
  | Let_static (bindings, body) ->
    let bindings0 = map (inline_calls_in_binding kept known) bindings in
    Let_static (bindings0, inline_calls kept (codes_of bindings0 known) body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, inline_calls_in_handler kept known h, inline_calls kept known body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, inline_calls_in_handler kept known h)
          handlers,
        inline_calls kept known body )
  | Apply ap -> inline_apply kept known ap
  | _ -> e

(** val inline_calls_in_handler : code_id list -> (code_id * code) list ->
    continuation_handler -> continuation_handler **)

and inline_calls_in_handler kept known = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, inline_calls kept known handler)

(** val inline_calls_in_binding : code_id list -> (code_id * code) list ->
    static_binding -> static_binding **)

and inline_calls_in_binding kept known binding =
  match binding with
  | Bind_code (id, c) -> Bind_code (id, inline_calls_in_code kept known c)
  | _ -> binding

(** val inline_calls_in_code : code_id list -> (code_id * code) list -> code ->
    code **)

and inline_calls_in_code kept known = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code (params, my_closure, ret, exn, metadata, inline_calls kept known body)

(** val var_in_simple : variable -> simple -> bool **)

let var_in_simple x = function Var y -> Pos.eqb x y | _ -> false

(** val var_in_value_slots : variable -> (value_slot * simple) list -> bool **)

let var_in_value_slots x slots =
  existsb (fun slot -> var_in_simple x (snd slot)) slots

(** val var_in_named : variable -> named -> bool **)

let var_in_named x = function
  | Simple s -> var_in_simple x s
  | Prim (_, args) -> var_in_simples x args
  | Set_of_closures (_, slots) -> var_in_value_slots x slots
  | Closure_of_set (_, _, slots) -> var_in_value_slots x slots
  | Project_value_slot (_, closure) -> var_in_simple x closure
  | Project_value_slot_from (_, _, closure) -> var_in_simple x closure
  | Project_function_slot (_, _, closure) -> var_in_simple x closure

(** val var_in_static_binding : variable -> static_binding -> bool **)

let var_in_static_binding x = function
  | Bind_symbol (_, const0) -> (
    match const0 with
    | Static_const_set_of_closures (_, slots) -> var_in_value_slots x slots
    | Static_const_closure_of_set (_, _, _, slots) -> var_in_value_slots x slots
    | Static_const_block (_, fields) -> var_in_simples x fields
    | Static_const_boxed_number contents -> var_in_simple x contents
    | Static_const_immutable_string _ -> false
    | Static_const_immutable_value_array elements -> var_in_simples x elements)
  | _ -> false

(** val var_free_b : variable -> expr -> bool **)

let rec var_free_b x = function
  | Let (y, _, d, body) ->
    var_in_named x d || (negb (Pos.eqb x y) && var_free_b x body)
  | Let_static (bindings, body) ->
    existsb (var_in_static_binding x) bindings || var_free_b x body
  | Let_cont_non_recursive (_, h, body) ->
    var_free_in_handler_b x h || var_free_b x body
  | Let_cont_recursive (handlers, body) ->
    existsb
      (fun pat ->
        let _, h = pat in
        var_free_in_handler_b x h)
      handlers
    || var_free_b x body
  | Apply ap ->
    (var_in_simple x ap.ap_callee || var_in_simples x ap.ap_args)
    || var_in_simples x (exn_extra_simples ap)
  | Apply_cont ac -> var_in_simples x ac.ac_args
  | Switch (scrutinee, arms) ->
    var_in_simple x scrutinee
    || existsb (fun arm -> var_in_simples x (snd arm).ac_args) arms
  | Invalid -> false

(** val var_free_in_handler_b : variable -> continuation_handler -> bool **)

and var_free_in_handler_b x = function
  | Handler (params, _, handler) ->
    negb (among x params) && var_free_b x handler

(** val prim_may_fold : prim -> bool **)

let prim_may_fold = function
  | Unary p0 -> ( match p0 with Opaque_identity _ -> false | _ -> true)
  | Binary _ -> true
  | Variadic _ -> true
  | _ -> false

(** val named_may_go : named -> bool **)

let named_may_go = function Prim (p, _) -> prim_may_fold p | _ -> true

(** val remove_unused_lets : expr -> expr **)

let rec remove_unused_lets e =
  match e with
  | Let (x, kind, d, body) ->
    let body' = remove_unused_lets body in
    if var_free_b x body' || negb (named_may_go d)
    then Let (x, kind, d, body')
    else body'
  | Let_static (bindings, body) -> Let_static (bindings, remove_unused_lets body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      (k, remove_unused_lets_in_handler h, remove_unused_lets body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, remove_unused_lets body)
  | _ -> e

(** val remove_unused_lets_in_handler : continuation_handler ->
    continuation_handler **)

and remove_unused_lets_in_handler = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, remove_unused_lets handler)

(** val inline_cont_at_tail : continuation -> variable list -> handler_metadata
    -> expr -> expr -> expr option **)

let rec inline_cont_at_tail k params metadata handler = function
  | Let (x, kind, d, body0) ->
    if var_free_in_handler_b x (Handler (params, metadata, handler))
    then None
    else
      option_map
        (fun x0 -> Let (x, kind, d, x0))
        (inline_cont_at_tail k params metadata handler body0)
  | Let_static (bindings, body0) ->
    option_map
      (fun x -> Let_static (bindings, x))
      (inline_cont_at_tail k params metadata handler body0)
  | Apply_cont ac ->
    if
      ((Pos.eqb ac.ac_continuation k && no_trap_action ac)
      && Nat.eqb (length params) (length ac.ac_args))
      && forallb (fun param -> negb (var_in_simples param ac.ac_args)) params
    then Some (let_params params metadata.hm_params_kinds ac.ac_args handler)
    else None
  | _ -> None

(** val inline_tail_conts : expr -> expr **)

let rec inline_tail_conts e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, inline_tail_conts body)
  | Let_static (bindings, body) -> Let_static (bindings, inline_tail_conts body)
  | Let_cont_non_recursive (k, handler0, body) -> (
    let (Handler (params, metadata, handler)) = handler0 in
    let handler' = inline_tail_conts handler in
    let body' = inline_tail_conts body in
    match inline_cont_at_tail k params metadata handler' body' with
    | Some inlined -> inlined
    | None ->
      Let_cont_non_recursive (k, Handler (params, metadata, handler'), body'))
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, inline_tail_conts body)
  | _ -> e

(** val binds_something : static_binding -> bool **)

let binds_something = function Bind_deleted_code _ -> false | _ -> true

(** val let_static_if_any : static_binding list -> expr -> expr **)

let let_static_if_any bindings body =
  match bindings with [] -> body | _ :: _ -> Let_static (bindings, body)

(** val drop_deleted_code : expr -> expr **)

let rec drop_deleted_code e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, drop_deleted_code body)
  | Let_static (bindings, body) ->
    let_static_if_any (filter binds_something bindings) (drop_deleted_code body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, drop_deleted_code handler),
        drop_deleted_code body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, drop_deleted_code body)
  | _ -> e

(** val consts_of_simples : simple list -> const list option **)

let rec consts_of_simples = function
  | [] -> Some []
  | s :: args0 -> (
    match s with
    | Const c -> option_map (fun x -> c :: x) (consts_of_simples args0)
    | _ -> None)

(** val fold_named : named -> named **)

let fold_named d =
  match d with
  | Prim (p, args) -> (
    match p with
    | Unary p0 -> (
      match p0 with
      | Array_length _ ->
        Prim (Unary (Array_length (Array_kind Array_values)), args)
      | _ -> (
        match if prim_may_fold p then consts_of_simples args else None with
        | Some cs -> (
          match
            eval_prim empty_static_env p (map (fun x -> Value_const x) cs)
          with
          | Some v -> (
            match v with Value_const c -> Simple (Const c) | _ -> d)
          | None -> d)
        | None -> d))
    | Heap p0 -> (
      match p0 with
      | Array_set (_, set_kind) ->
        Prim (Heap (Array_set (array_kind_of_set_kind set_kind, set_kind)), args)
      | _ -> (
        match if prim_may_fold p then consts_of_simples args else None with
        | Some cs -> (
          match
            eval_prim empty_static_env p (map (fun x -> Value_const x) cs)
          with
          | Some v -> (
            match v with Value_const c -> Simple (Const c) | _ -> d)
          | None -> d)
        | None -> d))
    | _ -> (
      match if prim_may_fold p then consts_of_simples args else None with
      | Some cs -> (
        match
          eval_prim empty_static_env p (map (fun x -> Value_const x) cs)
        with
        | Some v -> ( match v with Value_const c -> Simple (Const c) | _ -> d)
        | None -> d)
      | None -> d))
  | _ -> d

(** val fold_switch : simple -> (coq_Z * apply_cont) list -> expr **)

let fold_switch scrutinee arms =
  match scrutinee with
  | Const c -> (
    match c with
    | Naked_immediate i -> (
      match find_arm i arms with
      | Some ac -> Apply_cont ac
      | None -> Switch (scrutinee, arms))
    | _ -> Switch (scrutinee, arms))
  | _ -> Switch (scrutinee, arms)

(** val fold_constants : expr -> expr **)

let rec fold_constants e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, fold_named d, fold_constants body)
  | Let_static (bindings, body) -> Let_static (bindings, fold_constants body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, fold_constants handler),
        fold_constants body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, fold_constants body)
  | Switch (scrutinee, arms) -> fold_switch scrutinee arms
  | _ -> e

(** val named_is_total : flambda_kind -> named -> bool **)

let named_is_total kind = function
  | Simple s -> (
    match s with
    | Var _ -> false
    | Symbol _ -> kind_eqb Value kind
    | Const c -> kind_eqb (kind_of_const c) kind)
  | _ -> false

(** val remove_unused_total_lets : expr -> expr **)

let rec remove_unused_total_lets e =
  match e with
  | Let (x, kind, d, body) ->
    let body' = remove_unused_total_lets body in
    if named_is_total kind d && negb (var_free_b x body')
    then body'
    else Let (x, kind, d, body')
  | Let_static (bindings, body) ->
    Let_static (bindings, remove_unused_total_lets body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, remove_unused_total_lets handler),
        remove_unused_total_lets body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, remove_unused_total_lets body)
  | _ -> e

type kinds_in_scope = (variable * flambda_kind) list

(** val kind_of_simple_in_scope : kinds_in_scope -> simple -> flambda_kind
    option **)

let kind_of_simple_in_scope kinds = function
  | Var x -> lookup x kinds
  | Symbol _ -> Some Value
  | Const c -> Some (kind_of_const c)

(** val named_defined : kinds_in_scope -> flambda_kind -> named -> bool **)

let named_defined kinds kind = function
  | Simple s -> (
    match kind_of_simple_in_scope kinds s with
    | Some found -> kind_eqb found kind
    | None -> false)
  | Set_of_closures (_, slots) ->
    kind_eqb Value kind
    && forallb (fun slot -> simple_in_scope (map fst kinds) (snd slot)) slots
  | Closure_of_set (slot, decls, slots) ->
    (kind_eqb Value kind && is_bound slot decls)
    && forallb (fun slot0 -> simple_in_scope (map fst kinds) (snd slot0)) slots
  | _ -> false

(** val params_in_scope : variable list -> flambda_kind list -> kinds_in_scope
    -> kinds_in_scope **)

let rec params_in_scope params params_kinds kinds =
  match params with
  | [] -> kinds
  | param :: params0 -> (
    match params_kinds with
    | [] -> kinds
    | kind :: params_kinds0 ->
      params_in_scope params0 params_kinds0 ((param, kind) :: kinds))

(** val remove_unused_defined_lets : kinds_in_scope -> expr -> expr **)

let rec remove_unused_defined_lets kinds e =
  match e with
  | Let (x, kind, d, body) ->
    let body' = remove_unused_defined_lets ((x, kind) :: kinds) body in
    if named_defined kinds kind d && negb (var_free_b x body')
    then body'
    else Let (x, kind, d, body')
  | Let_static (bindings, body) ->
    Let_static (bindings, remove_unused_defined_lets kinds body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler
          ( params,
            metadata,
            remove_unused_defined_lets
              (params_in_scope params metadata.hm_params_kinds kinds)
              handler ),
        remove_unused_defined_lets kinds body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, remove_unused_defined_lets kinds body)
  | _ -> e

(** val symbols_of : static_binding list -> (symbol * code_id option) list ->
    (symbol * code_id option) list **)

let rec symbols_of bindings known =
  match bindings with
  | [] -> known
  | s0 :: bindings0 -> (
    match s0 with
    | Bind_symbol (s, const0) -> (
      match const0 with
      | Static_const_set_of_closures (code0, _) ->
        symbols_of bindings0 ((s, Some code0) :: known)
      | Static_const_closure_of_set (slot, _, function_decls, _) ->
        symbols_of bindings0 ((s, lookup slot function_decls) :: known)
      | _ -> symbols_of bindings0 ((s, None) :: known))
    | _ -> symbols_of bindings0 known)

(** val call_agrees_b : apply_expr -> code -> bool **)

let call_agrees_b ap c =
  (kinds_per_parameter_eqb
     (unarize_per_parameter ap.ap_args_arity)
     (unarize_per_parameter (code_params_arity c))
  && arity_eqb ap.ap_return_arity (code_result_arity c))
  && Nat.eqb (length (code_params c)) (length (code_params_kinds c))

(** val inline_known_apply : code_id list -> (code_id * code) list -> (symbol *
    code_id option) list -> apply_expr -> expr **)

let inline_known_apply kept known_code known_symbols ap =
  match ap.ap_callee with
  | Symbol s -> (
    match ap.ap_call_kind with
    | Direct called -> (
      match lookup s known_symbols with
      | Some o -> (
        match o with
        | Some code0 -> (
          match lookup called known_code with
          | Some c ->
            if
              negb (among called kept)
              && (Pos.eqb code0 called && inlinable_b ap c)
              && call_agrees_b ap c
            then
              inlined_body c
                (inline_renaming
                   (code_return_continuation c)
                   (code_exn_continuation c) ap.ap_continuation
                   ap.ap_exn_continuation)
                ap
            else Apply ap
          | None -> Apply ap)
        | None -> Apply ap)
      | None -> Apply ap)
    | Indirect -> Apply ap)
  | _ -> Apply ap

(** val inline_known_calls : code_id list -> (code_id * code) list -> (symbol *
    code_id option) list -> expr -> expr **)

let rec inline_known_calls kept known_code known_symbols e =
  match e with
  | Let (x, kind, d, body) ->
    Let (x, kind, d, inline_known_calls kept known_code known_symbols body)
  | Let_static (bindings, body) ->
    let bindings0 =
      map (inline_known_calls_in_binding kept known_code known_symbols) bindings
    in
    Let_static
      ( bindings0,
        inline_known_calls kept
          (codes_of bindings0 known_code)
          (symbols_of bindings0 known_symbols)
          body )
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive
      ( k,
        inline_known_calls_in_handler kept known_code known_symbols h,
        inline_known_calls kept known_code known_symbols body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, inline_known_calls_in_handler kept known_code known_symbols h)
          handlers,
        inline_known_calls kept known_code known_symbols body )
  | Apply ap -> inline_known_apply kept known_code known_symbols ap
  | _ -> e

(** val inline_known_calls_in_handler : code_id list -> (code_id * code) list ->
    (symbol * code_id option) list -> continuation_handler ->
    continuation_handler **)

and inline_known_calls_in_handler kept known_code known_symbols = function
  | Handler (params, metadata, handler) ->
    Handler
      ( params,
        metadata,
        inline_known_calls kept known_code known_symbols handler )

(** val inline_known_calls_in_binding : code_id list -> (code_id * code) list ->
    (symbol * code_id option) list -> static_binding -> static_binding **)

and inline_known_calls_in_binding kept known_code known_symbols binding =
  match binding with
  | Bind_code (id, c) ->
    Bind_code (id, inline_known_calls_in_code kept known_code known_symbols c)
  | _ -> binding

(** val inline_known_calls_in_code : code_id list -> (code_id * code) list ->
    (symbol * code_id option) list -> code -> code **)

and inline_known_calls_in_code kept known_code known_symbols = function
  | Code (params, my_closure, ret, exn, metadata, body) ->
    Code
      ( params,
        my_closure,
        ret,
        exn,
        metadata,
        inline_known_calls kept known_code known_symbols body )

(** val binds_code : static_binding -> bool **)

let binds_code = function Bind_symbol (_, _) -> false | _ -> true

(** val code_first : static_binding list -> static_binding list **)

let code_first bindings =
  app
    (filter binds_code bindings)
    (filter (fun binding -> negb (binds_code binding)) bindings)

(** val merge_statics : expr -> expr **)

let rec merge_statics e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, merge_statics body)
  | Let_static (bindings, body) -> (
    match merge_statics body with
    | Let_static (bindings', body') ->
      Let_static (code_first (app bindings bindings'), body')
    | x -> Let_static (code_first bindings, x))
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      (k, Handler (params, metadata, merge_statics handler), merge_statics body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, merge_statics body)
  | _ -> e

(** val simple_eq_dec : simple -> simple -> bool **)

let simple_eq_dec s s' =
  match s with
  | Var x -> ( match s' with Var x0 -> Pos.eq_dec x x0 | _ -> false)
  | Symbol s0 -> ( match s' with Symbol s1 -> Pos.eq_dec s0 s1 | _ -> false)
  | Const c -> ( match s' with Const c0 -> const_eq_dec c c0 | _ -> false)

(** val trap_action_eq_dec : trap_action -> trap_action -> bool **)

let trap_action_eq_dec action action' =
  match action with
  | Push exn_handler -> (
    match action' with
    | Push exn_handler0 ->
      let rec f p x =
        match p with
        | Coq_xI p0 -> ( match x with Coq_xI p1 -> f p0 p1 | _ -> false)
        | Coq_xO p0 -> ( match x with Coq_xO p1 -> f p0 p1 | _ -> false)
        | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
      in
      f exn_handler exn_handler0
    | Pop (_, _) -> false)
  | Pop (exn_handler, kind) -> (
    match action' with
    | Push _ -> false
    | Pop (exn_handler0, kind0) ->
      if
        let rec f p x =
          match p with
          | Coq_xI p0 -> ( match x with Coq_xI p1 -> f p0 p1 | _ -> false)
          | Coq_xO p0 -> ( match x with Coq_xO p1 -> f p0 p1 | _ -> false)
          | Coq_xH -> ( match x with Coq_xH -> true | _ -> false)
        in
        f exn_handler exn_handler0
      then
        match kind with
        | Some a -> (
          match kind0 with
          | Some r -> (
            match a with
            | Regular -> ( match r with Regular -> true | _ -> false)
            | Reraise -> ( match r with Reraise -> true | _ -> false)
            | No_trace -> ( match r with No_trace -> true | _ -> false))
          | None -> false)
        | None -> ( match kind0 with Some _ -> false | None -> true)
      else false)

(** val apply_cont_eq_dec : apply_cont -> apply_cont -> bool **)

let apply_cont_eq_dec ac ac' =
  let { ac_continuation = ac_continuation0;
        ac_args = ac_args0;
        ac_trap_action = ac_trap_action0
      } =
    ac
  in
  let { ac_continuation = ac_continuation1;
        ac_args = ac_args1;
        ac_trap_action = ac_trap_action1
      } =
    ac'
  in
  if Pos.eq_dec ac_continuation0 ac_continuation1
  then
    if list_eq_dec simple_eq_dec ac_args0 ac_args1
    then
      match ac_trap_action0 with
      | Some a -> (
        match ac_trap_action1 with
        | Some t -> trap_action_eq_dec a t
        | None -> false)
      | None -> ( match ac_trap_action1 with Some _ -> false | None -> true)
    else false
  else false

(** val collapse_switch : simple -> (coq_Z * apply_cont) list -> expr **)

let collapse_switch scrutinee arms =
  match arms with
  | [] -> Switch (scrutinee, arms)
  | p :: others ->
    let _, ac = p in
    if
      forallb
        (fun arm -> if apply_cont_eq_dec (snd arm) ac then true else false)
        others
    then Apply_cont ac
    else Switch (scrutinee, arms)

(** val collapse_switches : expr -> expr **)

let rec collapse_switches e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, collapse_switches body)
  | Let_static (bindings, body) -> Let_static (bindings, collapse_switches body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    Let_cont_non_recursive
      ( k,
        Handler (params, metadata, collapse_switches handler),
        collapse_switches body )
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, collapse_switches body)
  | Switch (scrutinee, arms) -> collapse_switch scrutinee arms
  | _ -> e

(** val rewrite_code_body : (expr -> expr) -> expr -> expr **)

let rewrite_code_body f body =
  let body' = f body in
  if eqb (binds_statics body) (binds_statics body') then body' else body

(** val rewrite_code_in_binding : (expr -> expr) -> static_binding ->
    static_binding **)

let rewrite_code_in_binding f binding =
  match binding with
  | Bind_code (id, c) ->
    let (Code (params, my_closure, ret, exn, metadata, body)) = c in
    Bind_code
      ( id,
        Code (params, my_closure, ret, exn, metadata, rewrite_code_body f body)
      )
  | _ -> binding

(** val rewrite_code : (expr -> expr) -> expr -> expr **)

let rec rewrite_code f e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, rewrite_code f body)
  | Let_static (bindings, body) ->
    Let_static (map (rewrite_code_in_binding f) bindings, rewrite_code f body)
  | Let_cont_non_recursive (k, h, body) ->
    Let_cont_non_recursive (k, rewrite_code_in_handler f h, rewrite_code f body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive
      ( map
          (fun pat ->
            let k, h = pat in
            k, rewrite_code_in_handler f h)
          handlers,
        rewrite_code f body )
  | _ -> e

(** val rewrite_code_in_handler : (expr -> expr) -> continuation_handler ->
    continuation_handler **)

and rewrite_code_in_handler f = function
  | Handler (params, metadata, handler) ->
    Handler (params, metadata, rewrite_code f handler)

(** val thread_arm : continuation -> (coq_Z * apply_cont) list -> (coq_Z *
    apply_cont) -> coq_Z * apply_cont **)

let thread_arm k inner arm =
  match arm with
  | i, ac -> (
    match ac.ac_args with
    | [] ->
      if Pos.eqb ac.ac_continuation k && no_trap_action ac
      then
        match find_arm i inner with
        | Some ac' ->
          if among k (apply_cont_continuations ac') then arm else i, ac'
        | None -> arm
      else arm
    | _ :: _ -> arm)

(** val arms_apply : continuation -> (coq_Z * apply_cont) list -> bool **)

let arms_apply k arms =
  existsb (fun arm -> among k (apply_cont_continuations (snd arm))) arms

(** val thread_switch : continuation -> continuation_handler -> expr -> expr **)

let thread_switch k h body =
  let (Handler (params, metadata, handler)) = h in
  match params with
  | [] -> (
    match handler with
    | Switch (inner_scrutinee, inner) -> (
      match body with
      | Switch (scrutinee, arms) -> (
        match metadata.hm_params_kinds with
        | [] ->
          if simple_eq_dec scrutinee inner_scrutinee
          then
            let arms0 = map (thread_arm k inner) arms in
            if arms_apply k arms0
            then Let_cont_non_recursive (k, h, Switch (scrutinee, arms0))
            else Switch (scrutinee, arms0)
          else Let_cont_non_recursive (k, h, body)
        | _ :: _ -> Let_cont_non_recursive (k, h, body))
      | _ -> Let_cont_non_recursive (k, h, body))
    | _ -> Let_cont_non_recursive (k, h, body))
  | _ :: _ -> Let_cont_non_recursive (k, h, body)

(** val thread_switches : expr -> expr **)

let rec thread_switches e =
  match e with
  | Let (x, kind, d, body) -> Let (x, kind, d, thread_switches body)
  | Let_static (bindings, body) -> Let_static (bindings, thread_switches body)
  | Let_cont_non_recursive (k, handler0, body) ->
    let (Handler (params, metadata, handler)) = handler0 in
    thread_switch k
      (Handler (params, metadata, thread_switches handler))
      (thread_switches body)
  | Let_cont_recursive (handlers, body) ->
    Let_cont_recursive (handlers, thread_switches body)
  | _ -> e
