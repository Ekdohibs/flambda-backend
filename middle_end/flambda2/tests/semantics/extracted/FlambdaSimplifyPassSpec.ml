open BinInt
open BinNums
open BinPos
open Datatypes
open FlambdaBindingTime
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaNormaliseSpec
open FlambdaPrimitiveModel
open FlambdaPrimitiveSpec
open FlambdaProvers
open FlambdaSemanticsSpec
open FlambdaSimplifyApplySpec
open FlambdaSimplifyArithSpec
open FlambdaSimplifyPrimitivesSpec
open FlambdaSimplifyProjectionsSpec
open FlambdaSimplifyValueSlotsSpec
open FlambdaTypeGrammar
open FlambdaTypingEnv
open FlambdaTypingEnvInvariantSpec
open FlambdaUnusedContsSpec
open List
open ListDef
open OxCamlOption
open PeanoNat

type outcome =
  | Becomes_invalid
  | Becomes_simple of simple
  | Stays_with_type of flambda_type
  | Stays_with_kind of flambda_kind
  | Not_known

(** val naked_bool_const : bool -> simple **)

let naked_bool_const b = Const (Naked_immediate (if b then Zpos Coq_xH else Z0))

(** val of_bool_shortcut : bool Meet_shortcut.t option -> outcome -> outcome **)

let of_bool_shortcut answer otherwise =
  match answer with
  | Some t0 -> (
    match t0 with
    | Meet_shortcut.Known_result b -> Becomes_simple (naked_bool_const b)
    | Meet_shortcut.Need_meet -> otherwise
    | Meet_shortcut.Invalid -> Becomes_invalid)
  | None -> otherwise

(** val of_simple_shortcut : simple Meet_shortcut.t option -> outcome -> outcome
    **)

let of_simple_shortcut answer otherwise =
  match answer with
  | Some t0 -> (
    match t0 with
    | Meet_shortcut.Known_result s -> Becomes_simple s
    | Meet_shortcut.Need_meet -> otherwise
    | Meet_shortcut.Invalid -> Becomes_invalid)
  | None -> otherwise

(** val simplify_unary : Typing_env.t -> unary_primitive -> simple ->
    flambda_type -> outcome **)

let simplify_unary tenv p arg ty =
  let unknown = Stays_with_kind (result_kind_of_unary_primitive p) in
  match p with
  | Block_load (_, mut, field) -> (
    match mut with
    | Mutable -> unknown
    | _ ->
      of_simple_shortcut
        (Provers.meet_block_field_simple tenv Name_mode.normal Value ty
           (Z.to_nat field))
        unknown)
  | Is_int variant_only -> (
    if variant_only
    then of_bool_shortcut (Provers.meet_is_int_variant_only tenv ty) unknown
    else
      match Provers.prove_is_int tenv ty with
      | Some t0 -> (
        match t0 with
        | Proof_of_property.Proved b -> Becomes_simple (naked_bool_const b)
        | Proof_of_property.Unknown -> unknown)
      | None -> unknown)
  | Is_null -> of_bool_shortcut (Provers.meet_is_null tenv ty) unknown
  | Get_tag -> (
    match Provers.prove_is_int tenv ty with
    | Some t0 -> (
      match t0 with
      | Proof_of_property.Proved a -> (
        if a
        then Becomes_invalid
        else
          match Provers.prove_get_tag tenv ty with
          | Some t1 -> (
            match t1 with
            | Proof_of_property.Proved a0 -> (
              match a0 with
              | [] -> unknown
              | tag :: l -> (
                match l with
                | [] -> Becomes_simple (Const (Naked_immediate tag))
                | _ :: _ -> unknown))
            | Proof_of_property.Unknown -> unknown)
          | None -> unknown)
      | Proof_of_property.Unknown -> (
        match Provers.prove_get_tag tenv ty with
        | Some t1 -> (
          match t1 with
          | Proof_of_property.Proved a -> (
            match a with
            | [] -> unknown
            | tag :: l -> (
              match l with
              | [] -> Becomes_simple (Const (Naked_immediate tag))
              | _ :: _ -> unknown))
          | Proof_of_property.Unknown -> unknown)
        | None -> unknown))
    | None -> (
      match Provers.prove_get_tag tenv ty with
      | Some t0 -> (
        match t0 with
        | Proof_of_property.Proved a -> (
          match a with
          | [] -> unknown
          | tag :: l -> (
            match l with
            | [] -> Becomes_simple (Const (Naked_immediate tag))
            | _ :: _ -> unknown))
        | Proof_of_property.Unknown -> unknown)
      | None -> unknown))
  | Unbox_number kind0 ->
    of_simple_shortcut
      (Provers.meet_boxed_number_containing_simple kind0 tenv Name_mode.normal
         (alias_type_of Value arg))
      unknown
  | Box_number kind0 -> Stays_with_type (box_number kind0 ty Alloc_mode.Heap)
  | Untag_immediate ->
    if types_well_formed_b tenv
    then
      of_simple_shortcut
        (Provers.meet_tagging_of_simple tenv Name_mode.normal
           (alias_type_of Value arg))
        unknown
    else unknown
  | Tag_immediate -> Stays_with_type (tag_immediate ty)
  | _ -> unknown

type one_side_outcome =
  | Exactly of coq_Z
  | The_other_side
  | No_result
  | Cannot_simplify

(** val symmetric_op_one_side_unknown : binary_int_arith_op -> coq_Z ->
    one_side_outcome **)

let symmetric_op_one_side_unknown op this_side =
  match op with
  | Add -> if Z.eqb this_side Z0 then The_other_side else Cannot_simplify
  | Mul ->
    if Z.eqb this_side Z0
    then Exactly Z0
    else if Z.eqb this_side (Zpos Coq_xH)
    then The_other_side
    else Cannot_simplify
  | And ->
    if Z.eqb this_side (Zneg Coq_xH)
    then The_other_side
    else if Z.eqb this_side Z0
    then Exactly Z0
    else Cannot_simplify
  | Or ->
    if Z.eqb this_side (Zneg Coq_xH)
    then Exactly (Zneg Coq_xH)
    else if Z.eqb this_side Z0
    then The_other_side
    else Cannot_simplify
  | Xor -> if Z.eqb this_side Z0 then The_other_side else Cannot_simplify
  | _ -> Cannot_simplify

(** val op_lhs_unknown : binary_int_arith_op -> coq_Z -> one_side_outcome **)

let op_lhs_unknown op rhs =
  match op with
  | Sub -> if Z.eqb rhs Z0 then The_other_side else Cannot_simplify
  | Div _ ->
    if Z.eqb rhs Z0
    then No_result
    else if Z.eqb rhs (Zpos Coq_xH)
    then The_other_side
    else Cannot_simplify
  | Mod s -> (
    if Z.eqb rhs Z0
    then No_result
    else if Z.eqb rhs (Zpos Coq_xH)
    then Exactly Z0
    else
      match s with
      | Signed ->
        if Z.eqb rhs (Zneg Coq_xH) then Exactly Z0 else Cannot_simplify
      | Unsigned -> Cannot_simplify)
  | _ -> symmetric_op_one_side_unknown op rhs

(** val op_rhs_unknown : binary_int_arith_op -> coq_Z -> one_side_outcome **)

let op_rhs_unknown op lhs =
  match op with
  | Sub -> Cannot_simplify
  | Div _ -> Cannot_simplify
  | Mod _ -> Cannot_simplify
  | _ -> symmetric_op_one_side_unknown op lhs

type possible_result =
  | Result_exactly of coq_Z
  | Result_other_side

(** val possible_result_eqb : possible_result -> possible_result -> bool **)

let possible_result_eqb r r' =
  match r with
  | Result_exactly i -> (
    match r' with Result_exactly i' -> Z.eqb i i' | Result_other_side -> false)
  | Result_other_side -> (
    match r' with Result_exactly _ -> false | Result_other_side -> true)

(** val one_side_results : (coq_Z -> one_side_outcome) -> coq_Z list ->
    possible_result list option **)

let rec one_side_results table = function
  | [] -> Some []
  | i :: nums0 -> (
    match one_side_results table nums0 with
    | Some results -> (
      match table i with
      | Exactly result -> Some (Result_exactly result :: results)
      | The_other_side -> Some (Result_other_side :: results)
      | No_result -> Some results
      | Cannot_simplify -> None)
    | None -> None)

(** val both_sides_results : binary_int_arith_op -> coq_Z list -> coq_Z list ->
    possible_result list **)

let both_sides_results op nums1 nums2 =
  flat_map
    (fun i1 ->
      flat_map
        (fun i2 ->
          match eval_int_arith tagged_width op i1 i2 with
          | Some result -> Result_exactly result :: []
          | None -> [])
        nums2)
    nums1

(** val exact_results : possible_result list -> coq_Z list **)

let rec exact_results = function
  | [] -> []
  | p :: results0 -> (
    match p with
    | Result_exactly result -> result :: exact_results results0
    | Result_other_side -> exact_results results0)

(** val check_possible_results : possible_result list -> simple -> outcome **)

let check_possible_results results other_side =
  match results with
  | [] -> Becomes_invalid
  | result :: rest ->
    if forallb (possible_result_eqb result) rest
    then
      match result with
      | Result_exactly i -> Becomes_simple (Const (Tagged_immediate i))
      | Result_other_side -> Becomes_simple other_side
    else if
      forallb
        (fun r ->
          match r with Result_exactly _ -> true | Result_other_side -> false)
        results
    then Stays_with_type (these_tagged_immediates (exact_results results))
    else Stays_with_type any_tagged_immediate

(** val max_num_possible_results : nat **)

let max_num_possible_results = S (S (S (S (S (S (S (S (S (S O)))))))))

(** val simplify_int_arith_tagged : Typing_env.t -> binary_int_arith_op ->
    simple -> flambda_type -> simple -> flambda_type -> outcome **)

let simplify_int_arith_tagged tenv op arg1 ty1 arg2 ty2 =
  let unknown = Stays_with_type any_tagged_immediate in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match Provers.meet_equals_tagged_immediates tenv ty1 with
    | Some t0 -> (
      match t0 with
      | Meet_shortcut.Known_result nums1 -> (
        match Provers.meet_equals_tagged_immediates tenv ty2 with
        | Some t1 -> (
          match t1 with
          | Meet_shortcut.Known_result nums2 ->
            if
              Nat.ltb max_num_possible_results (length nums1)
              || Nat.ltb max_num_possible_results (length nums2)
            then unknown
            else check_possible_results (both_sides_results op nums1 nums2) arg1
          | Meet_shortcut.Need_meet -> (
            match one_side_results (op_rhs_unknown op) nums1 with
            | Some results -> check_possible_results results arg2
            | None -> unknown)
          | Meet_shortcut.Invalid -> Becomes_invalid)
        | None -> (
          match one_side_results (op_rhs_unknown op) nums1 with
          | Some results -> check_possible_results results arg2
          | None -> unknown))
      | Meet_shortcut.Need_meet -> (
        match Provers.meet_equals_tagged_immediates tenv ty2 with
        | Some t1 -> (
          match t1 with
          | Meet_shortcut.Known_result nums2 -> (
            match one_side_results (op_lhs_unknown op) nums2 with
            | Some results -> check_possible_results results arg1
            | None -> unknown)
          | Meet_shortcut.Need_meet -> unknown
          | Meet_shortcut.Invalid -> Becomes_invalid)
        | None -> unknown)
      | Meet_shortcut.Invalid -> Becomes_invalid)
    | None -> (
      match Provers.meet_equals_tagged_immediates tenv ty2 with
      | Some t0 -> (
        match t0 with
        | Meet_shortcut.Known_result nums2 -> (
          match one_side_results (op_lhs_unknown op) nums2 with
          | Some results -> check_possible_results results arg1
          | None -> unknown)
        | Meet_shortcut.Need_meet -> unknown
        | Meet_shortcut.Invalid -> Becomes_invalid)
      | None -> unknown)
  else unknown

(** val comparison_unknown : int_comparison_behaviour -> outcome **)

let comparison_unknown = function
  | Yielding_bool _ -> Stays_with_type any_naked_bool
  | Yielding_int_like_compare_functions _ ->
    Stays_with_kind naked_immediate_kind

(** val simplify_int_comp_tagged : Typing_env.t -> int_comparison_behaviour ->
    flambda_type -> flambda_type -> outcome **)

let simplify_int_comp_tagged tenv behaviour ty1 ty2 =
  let unknown = comparison_unknown behaviour in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match Provers.meet_equals_tagged_immediates tenv ty1 with
    | Some t0 -> (
      match t0 with
      | Meet_shortcut.Known_result nums1 -> (
        match Provers.meet_equals_tagged_immediates tenv ty2 with
        | Some t1 -> (
          match t1 with
          | Meet_shortcut.Known_result nums2 -> (
            let results =
              flat_map
                (fun i1 ->
                  map
                    (fun i2 -> eval_int_comp tagged_width behaviour i1 i2)
                    nums2)
                nums1
            in
            match results with
            | [] -> unknown
            | result :: rest ->
              if forallb (Z.eqb result) rest
              then Becomes_simple (Const (Naked_immediate result))
              else unknown)
          | Meet_shortcut.Need_meet -> unknown
          | Meet_shortcut.Invalid -> Becomes_invalid)
        | None -> unknown)
      | Meet_shortcut.Need_meet -> (
        match Provers.meet_equals_tagged_immediates tenv ty2 with
        | Some t1 -> (
          match t1 with
          | Meet_shortcut.Invalid -> Becomes_invalid
          | _ -> unknown)
        | None -> unknown)
      | Meet_shortcut.Invalid -> Becomes_invalid)
    | None -> (
      match Provers.meet_equals_tagged_immediates tenv ty2 with
      | Some t0 -> (
        match t0 with Meet_shortcut.Invalid -> Becomes_invalid | _ -> unknown)
      | None -> unknown)
  else unknown

(** val simplify_phys_equal : Typing_env.t -> equality_comparison ->
    flambda_type -> flambda_type -> outcome **)

let simplify_phys_equal tenv op ty1 ty2 =
  let unknown = Stays_with_type any_naked_bool in
  if well_formed_input_b tenv ty1 && well_formed_input_b tenv ty2
  then
    match Provers.prove_physical_equality tenv ty1 ty2 with
    | Some t0 -> (
      match t0 with
      | Proof_of_property.Proved b ->
        Becomes_simple
          (naked_bool_const (match op with Phys_eq -> b | Phys_neq -> negb b))
      | Proof_of_property.Unknown -> unknown)
    | None -> unknown
  else unknown

(** val simplify_binary : Typing_env.t -> binary_primitive -> simple ->
    flambda_type -> simple -> flambda_type -> outcome **)

let simplify_binary tenv p arg1 ty1 arg2 ty2 =
  match p with
  | Phys_equal op -> simplify_phys_equal tenv op ty1 ty2
  | Int_arith (kind0, op) -> (
    match kind0 with
    | Standard_tagged_immediate ->
      simplify_int_arith_tagged tenv op arg1 ty1 arg2 ty2
    | _ -> Stays_with_kind (result_kind_of_binary_primitive p))
  | Int_comp (kind0, behaviour) -> (
    match kind0 with
    | Standard_tagged_immediate ->
      simplify_int_comp_tagged tenv behaviour ty1 ty2
    | _ -> comparison_unknown behaviour)
  | _ -> Stays_with_kind (result_kind_of_binary_primitive p)

(** val simplify_variadic : variadic_primitive -> flambda_type list -> outcome **)

let simplify_variadic p tys =
  match p with
  | Make_block kind0 ->
    let (Block_kind_values tag) = kind0 in
    if Z.ltb tag no_scan_tag
    then Stays_with_type (make_block_type tag tys)
    else Stays_with_kind Value
  | Make_immutable_array _ -> Stays_with_kind Value

(** val simplify_named : Typing_env.t -> named -> outcome **)

let simplify_named tenv = function
  | Simple s -> Becomes_simple s
  | Prim (p, args) -> (
    match simplify_simples tenv args with
    | Some answers ->
      let tys = map fst answers in
      if prim_args_well_kinded p (map kind tys)
      then
        match p with
        | Unary p0 -> (
          match args with
          | [] -> Not_known
          | arg :: l -> (
            match l with
            | [] -> (
              match tys with
              | [] -> Not_known
              | ty :: l0 -> (
                match l0 with
                | [] -> simplify_unary tenv p0 arg ty
                | _ :: _ -> Not_known))
            | _ :: _ -> Not_known))
        | Binary p0 -> (
          match args with
          | [] -> Not_known
          | arg1 :: l -> (
            match l with
            | [] -> Not_known
            | arg2 :: l0 -> (
              match l0 with
              | [] -> (
                match tys with
                | [] -> Not_known
                | ty1 :: l1 -> (
                  match l1 with
                  | [] -> Not_known
                  | ty2 :: l2 -> (
                    match l2 with
                    | [] -> simplify_binary tenv p0 arg1 ty1 arg2 ty2
                    | _ :: _ -> Not_known)))
              | _ :: _ -> Not_known)))
        | Variadic p0 -> simplify_variadic p0 tys
        | _ -> Not_known
      else Becomes_invalid
    | None -> Not_known)
  | Set_of_closures (code, value_slots) -> (
    match simplify_simples tenv (map snd value_slots) with
    | Some answers ->
      Stays_with_type
        (set_of_closures_type code (map fst value_slots) (map fst answers))
    | None -> Stays_with_kind Value)
  | Closure_of_set (slot, function_decls, value_slots) -> (
    match simplify_simples tenv (map snd value_slots) with
    | Some answers ->
      Stays_with_type
        (closure_of_set_type slot function_decls (map fst value_slots)
           (map fst answers))
    | None -> Stays_with_kind Value)
  | Project_function_slot (_, move_to, closure) -> (
    match simplify_simple tenv closure with
    | Some p ->
      let ty, _ = p in
      of_simple_shortcut
        (Provers.meet_project_function_slot_simple tenv Name_mode.normal ty
           move_to)
        (Stays_with_kind Value)
    | None -> Not_known)
  | _ -> Not_known

(** val canonical_named : Typing_env.t -> named -> named **)

let canonical_named tenv defining_expr =
  match defining_expr with
  | Prim (p, args) -> (
    match simplify_simples tenv args with
    | Some answers -> Prim (p, map snd answers)
    | None -> defining_expr)
  | _ -> defining_expr

type let_step =
  | Step_invalid
  | Step_let of named * Typing_env.t
  | Step_stop

(** val simple_in_terms_b : Typing_env.t -> simple -> bool **)

let simple_in_terms_b tenv = function
  | Var x -> Typing_env.mem (Some Name_mode.normal) tenv (Name_var x)
  | _ -> true

(** val simple_of_kind_b : Typing_env.t -> simple -> flambda_kind -> bool **)

let simple_of_kind_b tenv s kind0 =
  match s with
  | Var x -> is_some (Typing_env.find tenv (Name_var x) (Some kind0))
  | Symbol sym -> is_some (Typing_env.find tenv (Name_symbol sym) (Some kind0))
  | Const c -> kind_eqb (kind (type_for_const c)) kind0

(** val bind_simple : Typing_env.t -> variable -> simple -> let_step **)

let bind_simple tenv x s =
  match simplify_simple tenv s with
  | Some p ->
    let ty, canonical = p in
    let kind0 = kind ty in
    if simple_in_terms_b tenv canonical && simple_of_kind_b tenv canonical kind0
    then
      match
        Typing_env.add_alias
          (Typing_env.add_variable_definition tenv x kind0 Name_mode.normal)
          (Var x) canonical
      with
      | Some o -> (
        match o with
        | Ok result ->
          Step_let (Simple canonical, result.Typing_env.add_alias_result_t)
        | _ -> Step_stop)
      | None -> Step_stop
    else Step_stop
  | None -> Step_stop

(** val type_in_scope_b : Typing_env.t -> flambda_type -> bool **)

let type_in_scope_b tenv ty =
  forallb
    (fun y -> Typing_env.mem (Some Name_mode.normal) tenv (Name_var y))
    (free_variables ty)

(** val resolve_value_slot : Typing_env.t -> flambda_kind -> named -> named **)

let resolve_value_slot tenv declared defining_expr =
  match defining_expr with
  | Project_value_slot (slot, closure) -> (
    match simplify_simple tenv closure with
    | Some p -> (
      let ty, _ = p in
      match simplify_project_value_slot tenv declared slot ty with
      | Some s -> Simple s
      | None -> defining_expr)
    | None -> defining_expr)
  | Project_value_slot_from (_, slot, closure) -> (
    match simplify_simple tenv closure with
    | Some p -> (
      let ty, _ = p in
      match simplify_project_value_slot tenv declared slot ty with
      | Some s -> Simple s
      | None -> defining_expr)
    | None -> defining_expr)
  | _ -> defining_expr

(** val simplify_apply_cont : Typing_env.t -> apply_cont -> apply_cont **)

let simplify_apply_cont tenv ac =
  match simplify_simples tenv ac.ac_args with
  | Some answers ->
    { ac_continuation = ac.ac_continuation;
      ac_args = map snd answers;
      ac_trap_action = ac.ac_trap_action
    }
  | None -> ac

(** val simplify_switch : Typing_env.t -> simple -> (coq_Z * apply_cont) list ->
    expr **)

let simplify_switch tenv scrutinee arms =
  match simplify_simple tenv scrutinee with
  | Some p -> (
    let _, s = p in
    match s with
    | Const c -> (
      match c with
      | Naked_immediate i -> (
        match find_arm i arms with
        | Some ac -> Apply_cont (simplify_apply_cont tenv ac)
        | None -> Switch (scrutinee, arms))
      | _ -> Switch (scrutinee, arms))
    | _ -> Switch (scrutinee, arms))
  | None -> Switch (scrutinee, arms)

(** val arm_env : Typing_env.t -> simple -> coq_Z -> Typing_env.t **)

let arm_env tenv scrutinee d =
  match simplify_simple tenv scrutinee with
  | Some p -> (
    let _, s = p in
    match s with
    | Var c ->
      if
        simple_in_terms_b tenv (Var c)
        && simple_of_kind_b tenv (Var c) naked_immediate_kind
      then
        match Typing_env.add_alias tenv (Var c) (Const (Naked_immediate d)) with
        | Some o -> (
          match o with
          | Ok result -> result.Typing_env.add_alias_result_t
          | _ -> tenv)
        | None -> tenv
      else tenv
    | _ -> tenv)
  | None -> tenv

(** val not_among : variable list -> simple -> bool **)

let not_among xs = function Var x -> negb (among x xs) | _ -> true

(** val leaf_uses : Typing_env.t -> continuation -> expr -> (Typing_env.t *
    simple list) list option **)

let leaf_uses tenv k = function
  | Apply_cont ac ->
    Some (if Pos.eqb ac.ac_continuation k then (tenv, ac.ac_args) :: [] else [])
  | Switch (scrutinee, arms) ->
    Some
      (flat_map
         (fun arm ->
           if Pos.eqb (snd arm).ac_continuation k
           then (arm_env tenv scrutinee (fst arm), (snd arm).ac_args) :: []
           else [])
         arms)
  | _ -> None

(** val binders_not_free : variable list -> continuation list ->
    continuation_handler -> bool **)

let binders_not_free xs ks h =
  forallb (fun x -> negb (var_free_in_handler_b x h)) xs
  && forallb (fun k -> negb (cont_free_in_handler_b k h)) ks
