open BinInt
open BinNums
open Datatypes
open FlambdaBindingTime
open FlambdaKindModel
open FlambdaMapSpec
open FlambdaModel
open FlambdaProvers
open FlambdaTypeAccessSpec
open FlambdaTypeGrammar
open FlambdaTypingEnv
open List
open PeanoNat

module To_lift = struct
  type t =
    | Immutable_block of coq_Z * bool * scannable_block_shape * simple list
    | Boxed_number of boxable_number * coq_Z
end

module Reification_result = struct
  type t =
    | Lift of To_lift.t
    | Simple of simple
    | Cannot_reify
    | Invalid
end

module Reify = struct
  (** val reify_field : Typing_env.t -> (Alloc_mode.t -> variable -> bool
      option) -> Alloc_mode.t -> (flambda_type * flambda_kind) -> simple option
      option **)

  let reify_field tenv var_allowed0 alloc_mode field =
    match
      Provers.prove_equals_to_simple_of_kind tenv (fst field) (snd field)
    with
    | Some proof -> (
      match proof with
      | Proof_of_property.Proved simple0 -> (
        match simple0 with
        | Var var -> (
          match var_allowed0 alloc_mode var with
          | Some allowed -> if allowed then Some (Some simple0) else Some None
          | None -> None)
        | _ -> Some (Some simple0))
      | Proof_of_property.Unknown -> Some None)
    | None -> None

  (** val reify_fields : Typing_env.t -> (Alloc_mode.t -> variable -> bool
      option) -> Alloc_mode.t -> (flambda_type * flambda_kind) list -> simple
      list option **)

  let rec reify_fields tenv var_allowed0 alloc_mode = function
    | [] -> Some []
    | field :: fields0 -> (
      match reify_field tenv var_allowed0 alloc_mode field with
      | Some simple0 -> (
        match reify_fields tenv var_allowed0 alloc_mode fields0 with
        | Some simples -> (
          match simple0 with
          | Some simple1 -> Some (simple1 :: simples)
          | None -> Some simples)
        | None -> None)
      | None -> None)

  (** val try_to_reify_fields : Typing_env.t -> (Alloc_mode.t -> variable ->
      bool option) -> Alloc_mode.t -> (flambda_type * flambda_kind) list ->
      simple list option option **)

  let try_to_reify_fields tenv var_allowed0 alloc_mode
      field_types_and_expected_kinds =
    match
      reify_fields tenv var_allowed0 alloc_mode field_types_and_expected_kinds
    with
    | Some field_simples ->
      if Nat.eqb (length field_types_and_expected_kinds) (length field_simples)
      then Some (Some field_simples)
      else Some None
    | None -> None

  (** val var_allowed : Typing_env.t -> (variable -> bool) -> (variable -> bool)
      -> Typing_env.t -> Name_mode.t -> Alloc_mode.t -> variable -> bool option
      **)

  let var_allowed allowed_if_free_vars_defined_in var_is_defined_at_toplevel
      var_is_symbol_projection tenv min_name_mode alloc_mode var =
    if
      Typing_env.mem (Some min_name_mode) allowed_if_free_vars_defined_in
        (Name_var var)
    then
      if var_is_symbol_projection var
      then Some true
      else if var_is_defined_at_toplevel var
      then
        match alloc_mode with
        | Alloc_mode.Heap -> Some true
        | _ -> (
          match Provers.never_holds_locally_allocated_values tenv var with
          | Some proof -> (
            match proof with
            | Proof_of_property.Proved _ -> Some true
            | Proof_of_property.Unknown -> Some false)
          | None -> None)
      else Some false
    else Some false

  (** val unboxed_number_kind : boxable_number -> naked_number_kind **)

  let unboxed_number_kind = function
    | Boxable_naked_float32 -> Kind_naked_float32
    | Boxable_naked_float -> Kind_naked_float
    | Boxable_naked_int32 -> Kind_naked_int32
    | Boxable_naked_int64 -> Kind_naked_int64
    | Boxable_naked_nativeint -> Kind_naked_nativeint
    | Boxable_naked_vec128 -> Kind_naked_vec128
    | Boxable_naked_vec256 -> Kind_naked_vec256
    | Boxable_naked_vec512 -> Kind_naked_vec512
    | Boxable_naked_mask -> Kind_naked_mask

  (** val reify_expanded : (Alloc_mode.t -> variable -> bool option) ->
      Typing_env.t -> Reification_result.t -> flambda_type ->
      Reification_result.t option **)

  let reify_expanded var_allowed0 tenv try_canonical_simple t0 =
    match Provers.expand_head tenv t0 with
    | Some expanded -> (
      match expanded with
      | Oub_value head -> (
        match head with
        | Unknown -> Some try_canonical_simple
        | Ok a -> (
          let (Head_of_kind_value (non_null, is_null)) = a in
          match non_null with
          | Ok a0 -> (
            match a0 with
            | Variant (_, immediates, _, blocks, _, is_unique) -> (
              match is_null with
              | Not_null -> (
                match blocks with
                | Or_unknown.Known blocks0 -> (
                  match immediates with
                  | Or_unknown.Known imms -> (
                    match Provers.is_bottom tenv imms with
                    | Some imms_bottom ->
                      if imms_bottom
                      then
                        match row_like_for_blocks_get_singleton blocks0 with
                        | Some p -> (
                          let p0, alloc_mode = p in
                          let p1, field_types = p0 in
                          let p2, size = p1 in
                          let tag, shape = p2 in
                          if negb (Nat.eqb size (length field_types))
                          then None
                          else
                            let kinds =
                              field_kinds shape (length field_types)
                            in
                            if
                              negb (Nat.eqb (length kinds) (length field_types))
                            then None
                            else
                              let field_types_and_expected_kinds =
                                combine field_types kinds
                              in
                              match shape with
                              | Scannable shape0 -> (
                                if negb (Provers.tag_is_scannable tag)
                                then None
                                else
                                  match
                                    try_to_reify_fields tenv var_allowed0
                                      alloc_mode field_types_and_expected_kinds
                                  with
                                  | Some fields -> (
                                    match fields with
                                    | Some fields0 ->
                                      Some
                                        (Reification_result.Lift
                                           (To_lift.Immutable_block
                                              (tag, is_unique, shape0, fields0)))
                                    | None -> Some try_canonical_simple)
                                  | None -> None)
                              | Float_record -> Some try_canonical_simple)
                        | None -> Some try_canonical_simple
                      else if row_like_for_blocks_is_bottom blocks0
                      then
                        match Provers.meet_naked_immediates tenv imms with
                        | Some proof -> (
                          match proof with
                          | Meet_shortcut.Known_result imms0 -> (
                            match Keyed.get_singleton Z.eqb imms0 with
                            | Some imm ->
                              Some
                                (Reification_result.Simple
                                   (Const (Tagged_immediate imm)))
                            | None -> Some try_canonical_simple)
                          | Meet_shortcut.Need_meet -> Some try_canonical_simple
                          | Meet_shortcut.Invalid ->
                            Some Reification_result.Invalid)
                        | None -> None
                      else Some try_canonical_simple
                    | None -> None)
                  | Or_unknown.Unknown -> Some try_canonical_simple)
                | Or_unknown.Unknown -> Some try_canonical_simple)
              | Maybe_null _ -> Some try_canonical_simple)
            | Boxed_number (k, contents, _) -> (
              match is_null with
              | Not_null -> (
                match
                  Provers.meet_naked_number (unboxed_number_kind k) tenv
                    contents
                with
                | Some proof -> (
                  match proof with
                  | Meet_shortcut.Known_result ns -> (
                    match Keyed.get_singleton Z.eqb ns with
                    | Some n ->
                      Some
                        (Reification_result.Lift (To_lift.Boxed_number (k, n)))
                    | None -> Some try_canonical_simple)
                  | Meet_shortcut.Need_meet -> Some try_canonical_simple
                  | Meet_shortcut.Invalid -> Some Reification_result.Invalid)
                | None -> None)
              | Maybe_null _ -> Some try_canonical_simple)
            | _ -> Some try_canonical_simple)
          | _ -> Some try_canonical_simple)
        | Bottom -> Some Reification_result.Invalid)
      | Oub_naked_immediate head0 -> (
        match head0 with
        | Unknown -> Some try_canonical_simple
        | Ok head -> (
          match
            Provers.meet_naked_immediates tenv
              (Type_naked_immediate (Ok (No_alias head)))
          with
          | Some proof -> (
            match proof with
            | Meet_shortcut.Known_result imms -> (
              match Keyed.get_singleton Z.eqb imms with
              | Some i ->
                Some (Reification_result.Simple (Const (Naked_immediate i)))
              | None -> Some try_canonical_simple)
            | Meet_shortcut.Need_meet -> Some try_canonical_simple
            | Meet_shortcut.Invalid -> Some Reification_result.Invalid)
          | None -> None)
        | Bottom -> Some Reification_result.Invalid)
      | Oub_naked_number (k, head) -> (
        match head with
        | Unknown -> Some try_canonical_simple
        | Ok ns -> (
          match Keyed.get_singleton Z.eqb ns with
          | Some n ->
            Some (Reification_result.Simple (Const (const_of_naked_number k n)))
          | None -> Some try_canonical_simple)
        | Bottom -> Some Reification_result.Invalid)
      | Oub_rec_info head -> (
        match head with
        | Bottom -> Some Reification_result.Invalid
        | _ -> Some try_canonical_simple)
      | Oub_region head -> (
        match head with
        | Bottom -> Some Reification_result.Invalid
        | _ -> Some try_canonical_simple))
    | None -> None

  (** val reify : Typing_env.t -> (variable -> bool) -> (variable -> bool) ->
      Typing_env.t -> flambda_type -> Reification_result.t option **)

  let reify allowed_if_free_vars_defined_in var_is_defined_at_toplevel
      var_is_symbol_projection tenv t0 =
    let var_allowed0 =
      var_allowed allowed_if_free_vars_defined_in var_is_defined_at_toplevel
        var_is_symbol_projection tenv Name_mode.normal
    in
    match
      Typing_env.get_alias_then_canonical_simple_exn tenv
        (Some Name_mode.normal) None t0
    with
    | Some canonical_simple -> (
      match canonical_simple with
      | Some s -> (
        match s with
        | Var _ ->
          let try_canonical_simple =
            match canonical_simple with
            | Some canonical_simple0 ->
              Reification_result.Simple canonical_simple0
            | None -> Reification_result.Cannot_reify
          in
          reify_expanded var_allowed0 tenv try_canonical_simple t0
        | x -> Some (Reification_result.Simple x))
      | None ->
        let try_canonical_simple =
          match canonical_simple with
          | Some canonical_simple0 ->
            Reification_result.Simple canonical_simple0
          | None -> Reification_result.Cannot_reify
        in
        reify_expanded var_allowed0 tenv try_canonical_simple t0)
    | None -> None
end
