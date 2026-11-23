open BinNums
open BinPos
open Datatypes
open FlambdaBlockLoadSpec
open FlambdaCseSpec
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaUnboxSpec
open List
open ListDef
open PeanoNat

(** val reads_a_box : unary_primitive -> bool **)

let reads_a_box = function
  | Block_load (_, mut, _) -> ( match mut with Mutable -> false | _ -> true)
  | Is_int _ -> true
  | Get_tag -> true
  | Unbox_number _ -> true
  | Untag_immediate -> true
  | _ -> false

type facts =
  { facts_blocks : known_blocks;
    facts_numbers : known_numbers;
    facts_reads : computed
  }

(** val no_facts : facts **)

let no_facts =
  { facts_blocks = no_blocks; facts_numbers = no_numbers; facts_reads = [] }

(** val reads_under_let : computed -> variable -> named -> computed **)

let reads_under_let known x d =
  let known0 = forget_computed x known in
  match d with
  | Prim (p0, args) -> (
    match p0 with
    | Unary p -> (
      match args with
      | [] -> known0
      | s :: l -> (
        match s with
        | Var y -> (
          match l with
          | [] ->
            if reads_a_box p && negb (Pos.eqb x y)
            then (x, (Unary p, Var y :: [])) :: known0
            else known0
          | _ :: _ -> known0)
        | _ -> known0))
    | _ -> known0)
  | _ -> known0

(** val facts_under_let : facts -> variable -> named -> facts **)

let facts_under_let known x d =
  { facts_blocks = known_under_let known.facts_blocks x d;
    facts_numbers = numbers_under_let known.facts_numbers x d;
    facts_reads = reads_under_let known.facts_reads x d
  }

(** val facts_under_statics : facts -> static_binding list -> facts **)

let facts_under_statics known bindings =
  { facts_blocks = known_under_statics known.facts_blocks bindings;
    facts_numbers = numbers_under_statics known.facts_numbers bindings;
    facts_reads = known.facts_reads
  }

(** val facts_under_params : facts -> variable list -> facts **)

let facts_under_params known params =
  { facts_blocks = forget_vars params known.facts_blocks;
    facts_numbers = forget_wrapped_vars params known.facts_numbers;
    facts_reads = forget_computed_vars params known.facts_reads
  }

(** val projection_of : facts -> unary_primitive -> simple -> simple option **)

let projection_of known p arg =
  if reads_a_box p
  then
    let d = Prim (Unary p, arg :: []) in
    match fold_named d with
    | Simple s -> Some s
    | _ -> (
      match load_known_named known.facts_blocks d with
      | Simple s -> Some s
      | _ -> (
        match unbox_known_named known.facts_numbers d with
        | Simple s -> Some s
        | _ ->
          option_map
            (fun x -> Var x)
            (computed_by known.facts_reads (Unary p) (arg :: []))))
  else None

(** val extend_args : facts -> nat -> unary_primitive -> simple list -> simple
    list option **)

let extend_args known i p args =
  match nth_error args i with
  | Some arg -> (
    match projection_of known p arg with
    | Some projection -> Some (app args (projection :: []))
    | None -> None)
  | None -> None

(** val extend_apply_cont : facts -> continuation -> nat -> unary_primitive ->
    apply_cont -> apply_cont option **)

let extend_apply_cont known k i p ac =
  if Pos.eqb ac.ac_continuation k
  then
    match extend_args known i p ac.ac_args with
    | Some args ->
      Some
        { ac_continuation = ac.ac_continuation;
          ac_args = args;
          ac_trap_action = ac.ac_trap_action
        }
    | None -> None
  else Some ac

(** val extend_arms : facts -> continuation -> nat -> unary_primitive -> (coq_Z
    * apply_cont) list -> (coq_Z * apply_cont) list option **)

let rec extend_arms known k i p = function
  | [] -> Some []
  | p0 :: arms0 -> (
    let discriminant, ac = p0 in
    match extend_apply_cont known k i p ac with
    | Some ac0 -> (
      match extend_arms known k i p arms0 with
      | Some arms1 -> Some ((discriminant, ac0) :: arms1)
      | None -> None)
    | None -> None)

(** val all_some : 'a1 option list -> 'a1 list option **)

let rec all_some = function
  | [] -> Some []
  | a :: l0 -> (
    match a with
    | Some a0 -> (
      match all_some l0 with Some l1 -> Some (a0 :: l1) | None -> None)
    | None -> None)

(** val extend_applications : facts -> continuation -> nat -> unary_primitive ->
    expr -> expr option **)

let rec extend_applications known k i p e =
  match e with
  | Let (x, kind, d, body) -> (
    match extend_applications (facts_under_let known x d) k i p body with
    | Some body0 -> Some (Let (x, kind, d, body0))
    | None -> None)
  | Let_static (bindings, body) -> (
    match
      extend_applications (facts_under_statics known bindings) k i p body
    with
    | Some body0 -> Some (Let_static (bindings, body0))
    | None -> None)
  | Let_cont_non_recursive (k', h, body) -> (
    match extend_in_handler known k i p h with
    | Some h0 -> (
      if Pos.eqb k' k
      then Some (Let_cont_non_recursive (k', h0, body))
      else
        match extend_applications known k i p body with
        | Some body0 -> Some (Let_cont_non_recursive (k', h0, body0))
        | None -> None)
    | None -> None)
  | Let_cont_recursive (handlers, body) -> (
    if is_bound k handlers
    then Some e
    else
      match
        all_some
          (map
             (fun pat ->
               let k', h = pat in
               match extend_in_handler known k i p h with
               | Some h0 -> Some (k', h0)
               | None -> None)
             handlers)
      with
      | Some handlers0 -> (
        match extend_applications known k i p body with
        | Some body0 -> Some (Let_cont_recursive (handlers0, body0))
        | None -> None)
      | None -> None)
  | Apply ap ->
    if Pos.eqb ap.ap_continuation k || Pos.eqb ap.ap_exn_continuation k
    then None
    else Some e
  | Apply_cont ac -> (
    match extend_apply_cont known k i p ac with
    | Some ac0 -> Some (Apply_cont ac0)
    | None -> None)
  | Switch (scrutinee, arms) -> (
    match extend_arms known k i p arms with
    | Some arms0 -> Some (Switch (scrutinee, arms0))
    | None -> None)
  | Invalid -> Some e

(** val extend_in_handler : facts -> continuation -> nat -> unary_primitive ->
    continuation_handler -> continuation_handler option **)

and extend_in_handler known k i p = function
  | Handler (params, metadata, handler) -> (
    match
      extend_applications (facts_under_params known params) k i p handler
    with
    | Some handler0 -> Some (Handler (params, metadata, handler0))
    | None -> None)

(** val extend_apply_cont_by : (facts -> simple list -> simple list option) ->
    facts -> continuation -> apply_cont -> apply_cont option **)

let extend_apply_cont_by extend known k ac =
  if Pos.eqb ac.ac_continuation k
  then
    match extend known ac.ac_args with
    | Some args ->
      Some
        { ac_continuation = ac.ac_continuation;
          ac_args = args;
          ac_trap_action = ac.ac_trap_action
        }
    | None -> None
  else Some ac

(** val extend_arms_by : (facts -> simple list -> simple list option) -> facts
    -> continuation -> (coq_Z * apply_cont) list -> (coq_Z * apply_cont) list
    option **)

let rec extend_arms_by extend known k = function
  | [] -> Some []
  | p :: arms0 -> (
    let discriminant, ac = p in
    match extend_apply_cont_by extend known k ac with
    | Some ac0 -> (
      match extend_arms_by extend known k arms0 with
      | Some arms1 -> Some ((discriminant, ac0) :: arms1)
      | None -> None)
    | None -> None)

(** val extend_applications_by : (facts -> simple list -> simple list option) ->
    facts -> continuation -> expr -> expr option **)

let rec extend_applications_by extend known k e =
  match e with
  | Let (x, kind, d, body) -> (
    match extend_applications_by extend (facts_under_let known x d) k body with
    | Some body0 -> Some (Let (x, kind, d, body0))
    | None -> None)
  | Let_static (bindings, body) -> (
    match
      extend_applications_by extend (facts_under_statics known bindings) k body
    with
    | Some body0 -> Some (Let_static (bindings, body0))
    | None -> None)
  | Let_cont_non_recursive (k', h, body) -> (
    match extend_in_handler_by extend known k h with
    | Some h0 -> (
      if Pos.eqb k' k
      then Some (Let_cont_non_recursive (k', h0, body))
      else
        match extend_applications_by extend known k body with
        | Some body0 -> Some (Let_cont_non_recursive (k', h0, body0))
        | None -> None)
    | None -> None)
  | Let_cont_recursive (handlers, body) -> (
    if is_bound k handlers
    then Some e
    else
      match
        all_some
          (map
             (fun pat ->
               let k', h = pat in
               match extend_in_handler_by extend known k h with
               | Some h0 -> Some (k', h0)
               | None -> None)
             handlers)
      with
      | Some handlers0 -> (
        match extend_applications_by extend known k body with
        | Some body0 -> Some (Let_cont_recursive (handlers0, body0))
        | None -> None)
      | None -> None)
  | Apply ap ->
    if Pos.eqb ap.ap_continuation k || Pos.eqb ap.ap_exn_continuation k
    then None
    else Some e
  | Apply_cont ac -> (
    match extend_apply_cont_by extend known k ac with
    | Some ac0 -> Some (Apply_cont ac0)
    | None -> None)
  | Switch (scrutinee, arms) -> (
    match extend_arms_by extend known k arms with
    | Some arms0 -> Some (Switch (scrutinee, arms0))
    | None -> None)
  | Invalid -> Some e

(** val extend_in_handler_by : (facts -> simple list -> simple list option) ->
    facts -> continuation -> continuation_handler -> continuation_handler option
    **)

and extend_in_handler_by extend known k = function
  | Handler (params, metadata, handler) -> (
    match
      extend_applications_by extend (facts_under_params known params) k handler
    with
    | Some handler0 -> Some (Handler (params, metadata, handler0))
    | None -> None)

(** val position : variable -> variable list -> nat option **)

let rec position x = function
  | [] -> None
  | y :: xs0 ->
    if Pos.eqb x y then Some O else option_map (fun x0 -> S x0) (position x xs0)

(** val may_take : variable list -> handler_metadata -> variable -> bool **)

let may_take params metadata f =
  ((negb metadata.hm_is_exn_handler && distinct params)
  && Nat.eqb (length metadata.hm_params_kinds) (length params))
  && negb (among f params)

(** val with_one_more : handler_metadata -> flambda_kind -> handler_metadata **)

let with_one_more metadata kind =
  { hm_params_kinds = app metadata.hm_params_kinds (kind :: []);
    hm_params_subkinds =
      firstn (length metadata.hm_params_kinds) metadata.hm_params_subkinds;
    hm_is_exn_handler = false
  }
