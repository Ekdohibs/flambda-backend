module ET = Expand_head.Expanded_type
module TE = Typing_env
module TG = Type_grammar
module MTC = More_type_creators
module TI = Target_ocaml_int
module ME = Meet_env
module K = Flambda_kind

type discriminant =
  | Tagged_immediate
  | Block of Tag.t option
  | Array
  | Closure

type accessor =
  | Untag_imm
  | Block_field of TI.t * K.t
  | Array_field of TI.t * K.t
  | Value_slot of Value_slot.t
  | Function_slot of Function_slot.t
  | Rec_info of Function_slot.t

module Accessor = struct
  module T0 = struct
    type t = accessor

    let print ppf accessor =
      let print_kind ppf kind =
        Format.fprintf ppf "@[<hov 1>(kind@ %a)@]" K.print kind
      in
      match accessor with
      | Untag_imm -> Format.fprintf ppf "untag_imm"
      | Block_field (index, kind) ->
        Format.fprintf ppf "@[<hov 1>(field@ %a@ %a)@]" TI.print index
          print_kind kind
      | Array_field (index, kind) ->
        Format.fprintf ppf "@[<hov 1>(array_get@ %a@ %a)@]" TI.print index
          print_kind kind
      | Value_slot value_slot ->
        Format.fprintf ppf "@[<hov 1>(value_slot@ %a)@]" Value_slot.print
          value_slot
      | Function_slot function_slot ->
        Format.fprintf ppf "@[<hov 1>(function_slot@ %a)@]" Function_slot.print
          function_slot
      | Rec_info function_slot ->
        Format.fprintf ppf "@[<hov 1>(rec_info@ %a)@]" Function_slot.print
          function_slot

    let equal accessor1 accessor2 =
      match accessor1, accessor2 with
      | Untag_imm, Untag_imm -> true
      | Block_field (index1, kind1), Block_field (index2, kind2) ->
        TI.equal index1 index2 && K.equal kind1 kind2
      | Array_field (index1, kind1), Array_field (index2, kind2) ->
        TI.equal index1 index2 && K.equal kind1 kind2
      | Value_slot slot1, Value_slot slot2 -> Value_slot.equal slot1 slot2
      | Function_slot slot1, Function_slot slot2 ->
        Function_slot.equal slot1 slot2
      | Rec_info slot1, Rec_info slot2 -> Function_slot.equal slot1 slot2
      | ( ( Untag_imm | Block_field _ | Array_field _ | Value_slot _
          | Function_slot _ | Rec_info _ ),
          _ ) ->
        false

    let compare accessor1 accessor2 =
      match accessor1, accessor2 with
      | Untag_imm, Untag_imm -> 0
      | Block_field (index1, kind1), Block_field (index2, kind2) ->
        let c = TI.compare index1 index2 in
        if c <> 0 then c else K.compare kind1 kind2
      | Array_field (index1, kind1), Array_field (index2, kind2) ->
        let c = TI.compare index1 index2 in
        if c <> 0 then c else K.compare kind1 kind2
      | Value_slot slot1, Value_slot slot2 -> Value_slot.compare slot1 slot2
      | Function_slot slot1, Function_slot slot2 ->
        Function_slot.compare slot1 slot2
      | Rec_info slot1, Rec_info slot2 -> Function_slot.compare slot1 slot2
      | ( Untag_imm,
          ( Block_field _ | Array_field _ | Value_slot _ | Function_slot _
          | Rec_info _ ) )
      | ( Block_field _,
          (Array_field _ | Value_slot _ | Function_slot _ | Rec_info _) )
      | Array_field _, (Value_slot _ | Function_slot _ | Rec_info _)
      | Value_slot _, (Function_slot _ | Rec_info _)
      | Function_slot _, Rec_info _ ->
        -1
      | ( ( Block_field _ | Array_field _ | Value_slot _ | Function_slot _
          | Rec_info _ ),
          _ ) ->
        1

    let hash accessor =
      match accessor with
      | Untag_imm -> Hashtbl.hash 0
      | Block_field (index, kind) -> Hashtbl.hash (0, TI.hash index, K.hash kind)
      | Array_field (index, kind) -> Hashtbl.hash (1, TI.hash index, K.hash kind)
      | Value_slot slot -> Hashtbl.hash (2, Value_slot.hash slot)
      | Function_slot slot -> Hashtbl.hash (3, Function_slot.hash slot)
      | Rec_info slot -> Hashtbl.hash (4, Function_slot.hash slot)
  end

  include T0
  include Container_types.Make (T0)
end

let unknown_accessor = function
  | Untag_imm -> TG.any_naked_immediate
  | Block_field (_, kind) | Array_field (_, kind) -> MTC.unknown kind
  | Value_slot value_slot -> MTC.unknown (Value_slot.kind value_slot)
  | Function_slot function_slot ->
    MTC.unknown (Function_slot.kind function_slot)
  | Rec_info _ -> MTC.unknown K.rec_info

let bottom_accessor accessor = MTC.bottom_like (unknown_accessor accessor)

let rec destructure_expanded_head discriminant accessor expanded =
  match ET.descr expanded with
  | Unknown -> unknown_accessor accessor
  | Bottom -> bottom_accessor accessor
  | Ok (Value head) -> destructure_head_of_kind_value discriminant accessor head
  | Ok
      ( Naked_immediate _ | Naked_float32 _ | Naked_float _ | Naked_int8 _
      | Naked_int16 _ | Naked_int32 _ | Naked_int64 _ | Naked_nativeint _
      | Naked_vec128 _ | Naked_vec256 _ | Naked_vec512 _ | Rec_info _ | Region _
        ) ->
    Misc.fatal_error "Cannot destructure non-value kinds"

and destructure_head_of_kind_value discriminant accessor head =
  let ({ non_null; is_null = _ } : TG.head_of_kind_value) = head in
  match non_null with
  | Unknown -> unknown_accessor accessor
  | Bottom -> bottom_accessor accessor
  | Ok head ->
    destructure_head_of_kind_value_non_null discriminant accessor head

and destructure_head_of_kind_value_non_null discriminant accessor head =
  match discriminant, accessor, (head : TG.head_of_kind_value_non_null) with
  | ( Tagged_immediate,
      Untag_imm,
      Variant { immediates; blocks = _; extensions = _; is_unique = _ } ) -> (
    match immediates with
    | Unknown -> unknown_accessor accessor
    | Known ty -> ty)
  | Block _, Block_field (_, _), Mutable_block _ -> unknown_accessor accessor
  | ( Block tag,
      Block_field (index, kind),
      Variant { blocks; immediates = _; extensions = _; is_unique = _ } ) -> (
    match blocks with
    | Unknown -> unknown_accessor accessor
    | Known row_like ->
      destructure_block_field_row_like_for_blocks tag index kind row_like)
  | Array, Array_field (index, kind), Array { contents; element_kind; _ } -> (
    match element_kind with
    | Bottom -> bottom_accessor accessor
    | Ok element_kind when not (K.equal kind (K.With_subkind.kind element_kind))
      ->
      bottom_accessor accessor
    | Unknown | Ok _ -> (
      match contents with
      | Unknown | Known Mutable -> unknown_accessor accessor
      | Known (Immutable { fields }) ->
        let index = TI.to_int index in
        if 0 <= index && index < Array.length fields
        then fields.(index)
        else bottom_accessor accessor))
  | ( Closure,
      Value_slot value_slot,
      Closures { by_function_slot; alloc_mode = _ } ) -> (
    match TG.Row_like_for_closures.get_env_var by_function_slot value_slot with
    | Unknown -> unknown_accessor accessor
    | Known ty -> ty)
  | ( Closure,
      Function_slot function_slot,
      Closures { by_function_slot; alloc_mode = _ } ) -> (
    match
      TG.Row_like_for_closures.get_closure by_function_slot function_slot
    with
    | Unknown -> unknown_accessor accessor
    | Known ty -> ty)
  | ( Closure,
      Rec_info function_slot,
      Closures { by_function_slot; alloc_mode = _ } ) -> (
    match TG.Row_like_for_closures.get_single_tag by_function_slot with
    | No_singleton -> unknown_accessor accessor
    | Exact_closure (_tag, maps_to) | Incomplete_closure (_tag, maps_to) -> (
      match
        TG.Closures_entry.find_function_type maps_to ~exact:false function_slot
      with
      | Bottom -> bottom_accessor accessor
      | Unknown -> unknown_accessor accessor
      | Ok function_type -> TG.Function_type.rec_info function_type))
  | ( (Tagged_immediate | Block _ | Array | Closure),
      ( Untag_imm | Block_field _ | Array_field _ | Value_slot _
      | Function_slot _ | Rec_info _ ),
      ( Variant _ | Mutable_block _ | Boxed_float32 _ | Boxed_float _
      | Boxed_int32 _ | Boxed_int64 _ | Boxed_nativeint _ | Boxed_vec128 _
      | Boxed_vec256 _ | Boxed_vec512 _ | Closures _ | String _ | Array _ ) ) ->
    bottom_accessor accessor

and destructure_block_field_row_like_for_blocks tag index kind row_like =
  let ({ known_tags; other_tags; alloc_mode = _ } : TG.row_like_for_blocks) =
    row_like
  in
  match tag with
  | Some tag -> (
    match Tag.Map.find_opt tag known_tags with
    | None -> (
      match other_tags with
      | Bottom -> MTC.bottom kind
      | Ok row_like_case ->
        destructure_block_field_row_like_block_case index kind row_like_case)
    | Some Unknown -> MTC.unknown kind
    | Some (Known row_like_case) ->
      destructure_block_field_row_like_block_case index kind row_like_case)
  | None -> (
    (* CR bclement: We could create a variable to represent the union of
       multiple fields, but it is not clear it would be that useful. *)
    match other_tags with
    | Ok row_like_case ->
      if Tag.Map.is_empty known_tags
      then destructure_block_field_row_like_block_case index kind row_like_case
      else MTC.unknown kind
    | Bottom -> (
      match Tag.Map.get_singleton known_tags with
      | Some (_, Unknown) | None -> MTC.unknown kind
      | Some (_, Known row_like_case) ->
        destructure_block_field_row_like_block_case index kind row_like_case))

and destructure_block_field_row_like_block_case index kind
    ({ maps_to; _ } : TG.row_like_block_case) =
  let index = TI.to_int index in
  if 0 <= index && index < Array.length maps_to
  then maps_to.(index)
  else MTC.unknown kind

module Var : sig
  type t

  module Map : Container_types.Map with type key = t

  val create : unit -> t
end = struct
  type t = int

  module Tree = Patricia_tree.Make (struct
    let print ppf n = Format.fprintf ppf "x%d" n
  end)

  module Map = Tree.Map

  let create =
    let cnt = ref 0 in
    fun () ->
      incr cnt;
      !cnt
end

type 'a pattern =
  | Any
  | Keep of (Var.t * 'a)
  | Unbox of discriminant * 'a pattern Accessor.Map.t

module Pattern : sig
  type 'a t = 'a pattern

  val any : 'a t

  val var : Var.t -> 'a -> 'a t

  val untag : 'a t -> 'a t

  type 'a block_field

  val block_field : TI.t -> K.t -> 'a t -> 'a block_field

  val block : ?tag:Tag.t -> 'a block_field list -> 'a t

  type 'a array_field

  val array_field : TI.t -> K.t -> 'a t -> 'a array_field

  val array : 'a array_field list -> 'a t

  type 'a closure_field

  val rec_info : Function_slot.t -> 'a t -> 'a closure_field

  val value_slot : Value_slot.t -> 'a t -> 'a closure_field

  val function_slot : Function_slot.t -> 'a t -> 'a closure_field

  val closure : 'a closure_field list -> 'a t
end = struct
  type 'a t = 'a pattern

  let any = Any

  let var var value = Keep (var, value)

  let unbox discriminant fields =
    Unbox
      ( discriminant,
        List.fold_left
          (Accessor.Map.disjoint_union ?eq:None ?print:None)
          Accessor.Map.empty fields )

  type 'a block_field = 'a pattern Accessor.Map.t

  type 'a array_field = 'a pattern Accessor.Map.t

  type 'a closure_field = 'a pattern Accessor.Map.t

  let accessor accessor t = Accessor.Map.singleton accessor t

  let untag var = unbox Tagged_immediate [accessor Untag_imm var]

  let block_field index kind t = accessor (Block_field (index, kind)) t

  let array_field index kind t = accessor (Array_field (index, kind)) t

  let rec_info function_slot t = accessor (Rec_info function_slot) t

  let value_slot value_slot t = accessor (Value_slot value_slot) t

  let function_slot function_slot t = accessor (Function_slot function_slot) t

  let block ?tag fields = unbox (Block tag) fields

  let array fields = unbox Array fields

  let closure fields = unbox Closure fields
end

let rec fold_destructuring ~f destructuring env ty acc =
  match destructuring with
  | Any -> acc
  | Keep id -> f id ty acc
  | Unbox (discriminant, accessors) ->
    let expanded = Expand_head.expand_head env ty in
    Accessor.Map.fold
      (fun accessor accessor_destructuring acc ->
        let accessor_ty =
          destructure_expanded_head discriminant accessor expanded
        in
        fold_destructuring ~f accessor_destructuring env accessor_ty acc)
      accessors acc

type 'a function_type =
  { code_id : Code_id.t;
    rec_info : 'a
  }

module Function_type = struct
  type 'a t = 'a function_type

  let create code_id ~rec_info = { code_id; rec_info }
end

type 'a expr =
  | Identity of 'a
  | Unknown of K.With_subkind.t
  | Tag_imm of 'a
  | Block of
      { is_unique : bool;
        tag : Tag.t;
        shape : K.Block_shape.t;
        alloc_mode : Alloc_mode.For_types.t;
        fields : 'a list
      }
  | Closure of
      { function_slot : Function_slot.t;
        all_function_slots_in_set :
          'a function_type Or_unknown.t Function_slot.Map.t;
        all_closure_types_in_set : 'a Function_slot.Map.t;
        all_value_slots_in_set : 'a Value_slot.Map.t;
        alloc_mode : Alloc_mode.For_types.t
      }

module Expr = struct
  type 'a t = 'a expr

  module Function_type = Function_type

  let var var = Identity var

  let unknown kind = Unknown (K.With_subkind.anything kind)

  let unknown_with_subkind kind = Unknown kind

  let tag_immediate naked = Tag_imm naked

  let immutable_block ~is_unique tag ~shape alloc_mode ~fields =
    Block { is_unique; tag; shape; alloc_mode; fields }

  let exactly_this_closure function_slot ~all_function_slots_in_set
      ~all_closure_types_in_set ~all_value_slots_in_set alloc_mode =
    Closure
      { function_slot;
        all_function_slots_in_set;
        all_closure_types_in_set;
        all_value_slots_in_set;
        alloc_mode
      }
end

type 'a rewrite =
  | Identity
  | Rewrite of 'a pattern * Var.t expr

module Rule = struct
  type 'a t = 'a rewrite

  let identity = Identity

  let rewrite pattern expr = Rewrite (pattern, expr)
end

module Make (X : sig
  type t

  module Map : Container_types.Map with type key = t

  val rewrite : t -> TE.t -> TG.t -> t rewrite

  val block_slot : ?tag:Tag.t -> t -> TI.t -> TE.t -> TG.t -> t

  val array_slot : t -> TI.t -> TE.t -> TG.t -> t

  type set_of_closures

  val set_of_closures :
    t -> Function_slot.t -> TE.t -> TG.closures_entry -> set_of_closures

  val rec_info :
    TE.t -> set_of_closures -> Function_slot.t -> Code_id.t -> TG.t -> t

  val value_slot : set_of_closures -> Value_slot.t -> TE.t -> TG.t -> t

  val function_slot : set_of_closures -> Function_slot.t -> TE.t -> TG.t -> t
end) =
struct
  open Or_unknown.Let_syntax

  type u =
    { aliases_of_names : (Name.t * K.t) X.Map.t Name.Map.t;
      names_to_process : (Name.t * X.t * K.t * Name.t) list
    }

  let empty = { aliases_of_names = Name.Map.empty; names_to_process = [] }

  let get_canonical_with ({ aliases_of_names; names_to_process } as u) _env
      canonical kind metadata =
    match Name.Map.find_opt canonical aliases_of_names with
    | None ->
      let aliases_of_names =
        Name.Map.add canonical
          (X.Map.singleton metadata (canonical, kind))
          aliases_of_names
      in
      let names_to_process =
        (canonical, metadata, kind, canonical) :: names_to_process
      in
      canonical, Coercion.id, { aliases_of_names; names_to_process }
    | Some aliases_of_name -> (
      match X.Map.find_opt metadata aliases_of_name with
      | Some (name_with_metadata, _kind) -> name_with_metadata, Coercion.id, u
      | None ->
        let name_as_string =
          Name.pattern_match canonical ~var:Variable.name
            ~symbol:Symbol.linkage_name_as_string
        in
        let var' = Variable.create name_as_string kind in
        let aliases_of_name =
          X.Map.add metadata (Name.var var', kind) aliases_of_name
        in
        let aliases_of_names =
          Name.Map.add canonical aliases_of_name aliases_of_names
        in
        let names_to_process =
          (canonical, metadata, kind, Name.var var') :: names_to_process
        in
        Name.var var', Coercion.id, { aliases_of_names; names_to_process })

  let rec rewrite_expanded_head env acc metadata expanded =
    let acc_ref = ref acc in
    let expanded_or_unknown : _ Or_unknown.t =
      match ET.descr expanded with
      | Unknown -> Unknown
      | Bottom -> Known (ET.bottom_like expanded)
      | Ok (Value ty) ->
        let ty_result, new_acc =
          rewrite_head_of_kind_value env !acc_ref metadata ty
        in
        acc_ref := new_acc;
        let>+ ty = ty_result in
        ET.create_value ty
      | Ok (Naked_immediate head) ->
        let>+ head = rewrite_head_of_kind_naked_immediate head in
        ET.create_naked_immediate head
      | Ok (Naked_float32 head) ->
        let>+ head = rewrite_head_of_kind_naked_float32 head in
        ET.create_naked_float32 head
      | Ok (Naked_float head) ->
        let>+ head = rewrite_head_of_kind_naked_float head in
        ET.create_naked_float head
      | Ok (Naked_int8 head) ->
        let>+ head = rewrite_head_of_kind_naked_int8 head in
        ET.create_naked_int8 head
      | Ok (Naked_int16 head) ->
        let>+ head = rewrite_head_of_kind_naked_int16 head in
        ET.create_naked_int16 head
      | Ok (Naked_int32 head) ->
        let>+ head = rewrite_head_of_kind_naked_int32 head in
        ET.create_naked_int32 head
      | Ok (Naked_int64 head) ->
        let>+ head = rewrite_head_of_kind_naked_int64 head in
        ET.create_naked_int64 head
      | Ok (Naked_nativeint head) ->
        let>+ head = rewrite_head_of_kind_naked_nativeint head in
        ET.create_naked_nativeint head
      | Ok (Naked_vec128 head) ->
        let>+ head = rewrite_head_of_kind_naked_vec128 head in
        ET.create_naked_vec128 head
      | Ok (Naked_vec256 head) ->
        let>+ head = rewrite_head_of_kind_naked_vec256 head in
        ET.create_naked_vec256 head
      | Ok (Naked_vec512 head) ->
        let>+ head = rewrite_head_of_kind_naked_vec512 head in
        ET.create_naked_vec512 head
      | Ok (Rec_info head) ->
        let>+ head = rewrite_head_of_kind_rec_info head in
        ET.create_rec_info head
      | Ok (Region head) ->
        let>+ head = rewrite_head_of_kind_region head in
        ET.create_region head
    in
    match expanded_or_unknown with
    | Known expanded -> expanded, !acc_ref
    | Unknown -> ET.unknown_like expanded, !acc_ref

  and match_pattern pattern env ty acc =
    fold_destructuring pattern env ty (Var.Map.empty, acc)
      ~f:(fun (var, field_metadata) field_ty (sigma, acc) ->
        let field_ty', acc =
          rewrite_arbitrary_type env acc field_metadata field_ty
        in
        Var.Map.add var field_ty' sigma, acc)

  and rewrite_concrete_type_of env acc name kind abs =
    let ty = TE.find env name (Some kind) in
    rewrite env acc abs ty

  and rewrite env acc abs ty =
    match X.rewrite abs env ty with
    | Identity ->
      let expanded = Expand_head.expand_head env ty in
      let expanded, acc = rewrite_expanded_head env acc abs expanded in
      ET.to_type expanded, acc
    | Rewrite (pattern, expr) -> (
      let sigma, acc = match_pattern pattern env ty acc in
      let subst var =
        match Var.Map.find_opt var sigma with
        | Some ty -> ty
        | None -> Misc.fatal_error "Not defined"
      in
      match expr with
      | Identity var -> subst var, acc
      | Unknown kind ->
        MTC.unknown_with_subkind ~machine_width:(TE.machine_width env) kind, acc
      | Tag_imm field -> TG.tag_immediate (subst field), acc
      | Block { is_unique; tag; shape; alloc_mode; fields } ->
        let fields = List.map subst fields in
        ( MTC.immutable_block ~machine_width:(TE.machine_width env) ~is_unique
            tag ~shape alloc_mode ~fields,
          acc )
      | Closure
          { function_slot;
            all_function_slots_in_set;
            all_closure_types_in_set;
            all_value_slots_in_set;
            alloc_mode
          } ->
        let all_function_slots_in_set =
          Function_slot.Map.map
            (Or_unknown.map ~f:(fun { code_id; rec_info } ->
                 TG.Function_type.create code_id ~rec_info:(subst rec_info)))
            all_function_slots_in_set
        in
        let all_closure_types_in_set =
          Function_slot.Map.map subst all_closure_types_in_set
        in
        let all_value_slots_in_set =
          Value_slot.Map.map subst all_value_slots_in_set
        in
        ( MTC.exactly_this_closure function_slot ~all_function_slots_in_set
            ~all_closure_types_in_set ~all_value_slots_in_set alloc_mode,
          acc ))

  and rewrite_arbitrary_type env acc metadata ty =
    match TG.get_alias_opt ty with
    | Some alias ->
      let canonical =
        TE.get_canonical_simple_exn ~min_name_mode:Name_mode.in_types env alias
      in
      let canonical_with_metadata, acc =
        Simple.pattern_match canonical
          ~const:(fun _ -> canonical, acc)
          ~name:(fun name ~coercion ->
            (* Do not rewrite the types of names coming from other compilation
               units, since we can't re-define them and it's hard to think of a
               situation where it would be useful anyways.

               Note that simply skipping them here means that we lose any more
               precise type that we would have for these variables in the
               current compilation unit, but this should be rare at top level
               (which is where this API is intended to be used). *)
            if not
                 (Compilation_unit.equal
                    (Name.compilation_unit name)
                    (Compilation_unit.get_current_exn ()))
            then canonical, acc
            else
              let canonical_name, coercion_to_name, acc =
                get_canonical_with acc env name (TG.kind ty) metadata
              in
              let coercion =
                Coercion.compose_exn coercion_to_name ~then_:coercion
              in
              let simple = Simple.name canonical_name in
              Simple.with_coercion simple coercion, acc)
      in
      TG.alias_type_of (TG.kind ty) canonical_with_metadata, acc
    | None ->
      let ty', acc = rewrite env acc metadata ty in
      let expanded = Expand_head.expand_head env ty' in
      let expanded, acc = rewrite_expanded_head env acc metadata expanded in
      ET.to_type expanded, acc

  and rewrite_head_of_kind_value env acc metadata head :
      TG.head_of_kind_value Or_unknown.t * _ =
    let ({ non_null; is_null } : TG.head_of_kind_value) = head in
    match non_null with
    | Unknown | Bottom -> Known head, acc
    | Ok non_null ->
      let non_null, acc =
        rewrite_head_of_kind_value_non_null env acc metadata non_null
      in
      Known { non_null = Ok non_null; is_null }, acc

  and rewrite_head_of_kind_value_non_null env acc metadata
      (head : TG.head_of_kind_value_non_null) =
    match head with
    | Variant { blocks; immediates; extensions = _; is_unique } ->
      let blocks, acc =
        match blocks with
        | Unknown -> blocks, acc
        | Known blocks ->
          let blocks, acc =
            rewrite_row_like_for_blocks env acc metadata blocks
          in
          Or_unknown.Known blocks, acc
      in
      let immediates, acc =
        match immediates with
        | Unknown -> immediates, acc
        | Known immediates ->
          let immediates, acc =
            rewrite_arbitrary_type env acc metadata immediates
          in
          Or_unknown.Known immediates, acc
      in
      (* Drop extensions because it's not clear what to do *)
      ( TG.Head_of_kind_value_non_null.create_variant ~is_unique ~blocks
          ~immediates ~extensions:No_extensions,
        acc )
    | Mutable_block { alloc_mode = _ } -> head, acc
    | Boxed_float32 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_float32 ty alloc_mode, acc
    | Boxed_float (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_float ty alloc_mode, acc
    | Boxed_int32 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_int32 ty alloc_mode, acc
    | Boxed_int64 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_int64 ty alloc_mode, acc
    | Boxed_nativeint (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_nativeint ty alloc_mode, acc
    | Boxed_vec128 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_vec128 ty alloc_mode, acc
    | Boxed_vec256 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_vec256 ty alloc_mode, acc
    | Boxed_vec512 (ty, alloc_mode) ->
      let ty, acc = rewrite_arbitrary_type env acc metadata ty in
      TG.Head_of_kind_value_non_null.create_boxed_vec512 ty alloc_mode, acc
    | Closures { by_function_slot; alloc_mode } ->
      let by_function_slot, acc =
        rewrite_row_like_for_closures env acc metadata by_function_slot
      in
      ( TG.Head_of_kind_value_non_null.create_closures by_function_slot
          alloc_mode,
        acc )
    | String _ -> head, acc
    | Array { element_kind; length; contents; alloc_mode } ->
      let length, acc = rewrite_arbitrary_type env acc metadata length in
      let contents, acc =
        match contents with
        | Known (Immutable { fields }) ->
          let fields, acc =
            rewrite_int_indexed_product ~slot:X.array_slot env acc metadata
              fields
          in
          Or_unknown.Known (TG.Immutable { fields }), acc
        | Unknown | Known Mutable -> contents, acc
      in
      ( TG.Head_of_kind_value_non_null.create_array_with_contents ~element_kind
          ~length contents alloc_mode,
        acc )

  and rewrite_head_of_kind_naked_immediate
      (head : TG.head_of_kind_naked_immediate) : _ Or_unknown.t =
    match head with
    | Naked_immediates _ -> Or_unknown.Known head
    | Is_int _ | Get_tag _ | Is_null _ ->
      (* CR bclement: replace with prove. *)
      Or_unknown.Unknown

  and rewrite_head_of_kind_naked_float32 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_float head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_int8 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_int16 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_int32 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_int64 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_nativeint head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_vec128 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_vec256 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_naked_vec512 head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_rec_info head : _ Or_unknown.t =
    Or_unknown.Known head

  and rewrite_head_of_kind_region () : _ Or_unknown.t = Or_unknown.Known ()

  and rewrite_row_like_for_blocks env acc metadata
      ({ known_tags; other_tags; alloc_mode } : TG.row_like_for_blocks) =
    let known_tags, acc =
      Tag.Map.fold
        (fun tag case (known_tags, acc) ->
          let case, acc =
            match (case : _ TG.row_like_case Or_unknown.t) with
            | Unknown -> case, acc
            | Known { maps_to; env_extension = _; index } ->
              let maps_to, acc =
                rewrite_int_indexed_product
                  ~slot:(fun t index env ty -> X.block_slot ~tag t index env ty)
                  env acc metadata maps_to
              in
              (* Drop env extension because it is not clear what to do. *)
              ( Or_unknown.Known
                  (TG.Row_like_case.create ~maps_to
                     ~env_extension:TG.Env_extension.empty ~index),
                acc )
          in
          Tag.Map.add tag case known_tags, acc)
        known_tags (Tag.Map.empty, acc)
    in
    let other_tags, acc =
      match (other_tags : _ TG.row_like_case Or_bottom.t) with
      | Bottom -> Or_bottom.Bottom, acc
      | Ok { maps_to; env_extension = _; index } ->
        let maps_to, acc =
          rewrite_int_indexed_product
            ~slot:(fun t index env ty -> X.block_slot t index env ty)
            env acc metadata maps_to
        in
        ( Or_bottom.Ok
            (TG.Row_like_case.create ~maps_to
               ~env_extension:TG.Env_extension.empty ~index),
          acc )
    in
    TG.Row_like_for_blocks.create_raw ~known_tags ~other_tags ~alloc_mode, acc

  and rewrite_row_like_for_closures env acc metadata
      ({ known_closures; other_closures } : TG.row_like_for_closures) =
    let known_closures, acc =
      Function_slot.Map.fold
        (fun function_slot
             ({ maps_to; env_extension = _; index } : _ TG.row_like_case)
             (known_closures, acc) ->
          let set_of_closures_metadata =
            X.set_of_closures metadata function_slot env maps_to
          in
          let maps_to, acc =
            rewrite_closures_entry ~this_function_slot:function_slot env acc
              set_of_closures_metadata maps_to
          in
          let row_like_case =
            TG.Row_like_case.create ~maps_to
              ~env_extension:TG.Env_extension.empty ~index
          in
          Function_slot.Map.add function_slot row_like_case known_closures, acc)
        known_closures
        (Function_slot.Map.empty, acc)
    in
    let other_closures, acc =
      match other_closures with
      | Bottom -> Or_bottom.Bottom, acc
      | Ok { maps_to = _; env_extension = _; index = _ } -> assert false
    in
    TG.Row_like_for_closures.create_raw ~known_closures ~other_closures, acc

  and rewrite_closures_entry ~this_function_slot:_ env acc metadata
      ({ function_types; closure_types; value_slot_types } : TG.closures_entry)
      =
    let function_types, acc =
      Function_slot.Map.fold
        (fun function_slot function_type (function_types, acc) ->
          let function_type, acc =
            match (function_type : _ Or_unknown.t) with
            | Unknown -> function_type, acc
            | Known function_type ->
              (* XXX: Code_of_closure field *)
              (* Path does not change for function types within the entry *)
              let function_type, acc =
                rewrite_function_type env acc metadata function_slot
                  function_type
              in
              Or_unknown.Known function_type, acc
          in
          Function_slot.Map.add function_slot function_type function_types, acc)
        function_types
        (Function_slot.Map.empty, acc)
    in
    let closure_types, acc =
      rewrite_function_slot_indexed_product env acc metadata closure_types
    in
    let value_slot_types, acc =
      rewrite_value_slot_indexed_product env acc metadata value_slot_types
    in
    ( TG.Closures_entry.create ~function_types ~closure_types ~value_slot_types,
      acc )

  and rewrite_function_slot_indexed_product env acc metadata
      ({ function_slot_components_by_index } : TG.function_slot_indexed_product)
      =
    let function_slot_components_by_index, acc =
      Function_slot.Map.fold
        (fun function_slot function_slot_ty
             (function_slot_components_by_index, acc) ->
          let function_slot_metadata =
            X.function_slot metadata function_slot env function_slot_ty
          in
          let function_slot_ty', acc =
            rewrite_arbitrary_type env acc function_slot_metadata
              function_slot_ty
          in
          ( Function_slot.Map.add function_slot function_slot_ty'
              function_slot_components_by_index,
            acc ))
        function_slot_components_by_index
        (Function_slot.Map.empty, acc)
    in
    ( TG.Product.Function_slot_indexed.create function_slot_components_by_index,
      acc )

  and rewrite_value_slot_indexed_product env acc metadata
      ({ value_slot_components_by_index } : TG.value_slot_indexed_product) =
    let value_slot_components_by_index, acc =
      Value_slot.Map.fold
        (fun value_slot value_slot_ty (value_slot_components_by_index, acc) ->
          let value_slot_metadata =
            X.value_slot metadata value_slot env value_slot_ty
          in
          let value_slot_ty', acc =
            rewrite_arbitrary_type env acc value_slot_metadata value_slot_ty
          in
          ( Value_slot.Map.add value_slot value_slot_ty'
              value_slot_components_by_index,
            acc ))
        value_slot_components_by_index
        (Value_slot.Map.empty, acc)
    in
    TG.Product.Value_slot_indexed.create value_slot_components_by_index, acc

  and rewrite_int_indexed_product ~slot env acc metadata fields =
    let (acc, _), fields =
      Array.fold_left_map
        (fun (acc, index) field_ty ->
          let field_metadata = slot metadata index env field_ty in
          let field_ty', acc =
            rewrite_arbitrary_type env acc field_metadata field_ty
          in
          (acc, TI.(add index (one (TE.machine_width env)))), field_ty')
        (acc, TI.of_int (TE.machine_width env) 0)
        fields
    in
    fields, acc

  and rewrite_function_type env acc metadata function_slot
      ({ code_id; rec_info } : TG.function_type) =
    let rec_info_metadata =
      X.rec_info env metadata function_slot code_id rec_info
    in
    let rec_info, acc =
      rewrite_arbitrary_type env acc rec_info_metadata rec_info
    in
    TG.Function_type.create code_id ~rec_info, acc

  let rewrite env symbol_abstraction live_vars =
    let base_env =
      TE.create ~resolver:(TE.resolver env)
        ~get_imported_names:(TE.get_imported_names env)
        ~machine_width:(TE.machine_width env)
    in
    let base_env, new_types, acc =
      Symbol.Set.fold
        (fun symbol (base_env, types, acc) ->
          let abs = symbol_abstraction symbol in
          let ty, acc =
            rewrite_concrete_type_of env acc (Name.symbol symbol) K.value abs
          in
          let bound_name = Bound_name.create_symbol symbol in
          let base_env = TE.add_definition base_env bound_name (TG.kind ty) in
          base_env, Name.Map.add (Name.symbol symbol) ty types, acc)
        (TE.defined_symbols env)
        (base_env, Name.Map.empty, empty)
    in
    let base_env, new_types, acc =
      Variable.Map.fold
        (fun var (abs, kind) (base_env, types, acc) ->
          let ty, acc =
            rewrite_concrete_type_of env acc (Name.var var) kind abs
          in
          let bound_name =
            Bound_name.create_var
              (Bound_var.create var Flambda_debug_uid.none Name_mode.normal)
          in
          let base_env = TE.add_definition base_env bound_name (TG.kind ty) in
          base_env, Name.Map.add (Name.var var) ty types, acc)
        live_vars (base_env, new_types, acc)
    in
    let rec loop { aliases_of_names; names_to_process } new_types =
      match names_to_process with
      | [] -> new_types, aliases_of_names
      | _ :: _ ->
        let new_types, acc =
          List.fold_left
            (fun (new_types, acc)
                 (name_before_rewrite, abs, kind, name_after_rewrite) ->
              let ty, acc =
                rewrite_concrete_type_of env acc name_before_rewrite kind abs
              in
              let new_types = Name.Map.add name_after_rewrite ty new_types in
              new_types, acc)
            (new_types, { aliases_of_names; names_to_process = [] })
            names_to_process
        in
        loop acc new_types
    in
    let new_types, aliases_of_names = loop acc new_types in
    let base_env =
      Name.Map.fold
        (fun _name aliases_of_name base_env ->
          X.Map.fold
            (fun _metadata (name_after_rewrite, kind) base_env ->
              Name.pattern_match name_after_rewrite
                ~symbol:(fun _ -> base_env)
                ~var:(fun var_after_rewrite ->
                  if Variable.Map.mem var_after_rewrite live_vars
                  then base_env
                  else
                    let bound_name =
                      Bound_name.create_var
                        (Bound_var.create var_after_rewrite
                           Flambda_debug_uid.none Name_mode.in_types)
                    in
                    TE.add_definition base_env bound_name kind))
            aliases_of_name base_env)
        aliases_of_names base_env
    in
    ME.use_meet_env base_env ~f:(fun env ->
        Name.Map.fold
          (fun name ty env ->
            ME.add_equation env name ty ~meet_type:(Meet.meet_type ()))
          new_types env)
end
