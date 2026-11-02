(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*                       Pierre Chambart, OCamlPro                        *)
(*           Mark Shinwell and Leo White, Jane Street Europe              *)
(*                                                                        *)
(*   Copyright 2019 OCamlPro SAS                                          *)
(*   Copyright 2019 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

module Id = Table_by_int_id.Id
module Flags = Int_ids.Flags

(* The occurrences of one key, for up to three roles (e.g. a continuation can
   occur normally, applied with traps, or in a trap action), packed in an
   immediate.

   For each role there is a presence bit, in bits [0, 3), since a key can be
   present without any occurrences after
   [downgrade_occurrences_at_strictly_greater_name_mode].

   For each role and name mode there is a saturating counter (0b00: zero, 0b01:
   one, 0b11: more than one). Bits [3, 9) hold the counters for [Normal], bits
   [9, 15) for [In_types] and bits [15, 21) for [Phantom], two bits per role.

   The presence bits come first so that the most common values are small, and
   marshalled compactly in .cmx files. *)
module Counts = struct
  type t = int

  let presence_mask = 0b111

  let low_bits = 0x15555 lsl 3

  let counters_mask = 0x3ffff lsl 3

  let[@inline] mode_shift (name_mode : Name_mode.t) =
    match name_mode with Normal -> 3 | In_types -> 9 | Phantom -> 15

  let[@inline] presence role = 1 lsl role

  let[@inline] counters_of_role role = (0b11 lsl (3 + (2 * role))) * 0x1041

  let[@inline] normal_counter role = 0b11 lsl (3 + (2 * role))

  let[@inline] role_mask role = counters_of_role role lor presence role

  let[@inline] one role name_mode =
    presence role lor (1 lsl (mode_shift name_mode + (2 * role)))

  (* Saturating addition of all the counters at once *)
  let[@inline] union t1 t2 = t1 lor t2 lor ((t1 land t2 land low_bits) lsl 1)

  let[@inline] increase t = t lor ((t land low_bits) lsl 1)

  let[@inline] mem t role = t land presence role <> 0

  let[@inline] mem_any t roles = t land roles <> 0

  let[@inline] counter t role name_mode =
    (t lsr (mode_shift name_mode + (2 * role))) land 0b11

  (* [bits] has either zero, one or two bits set per counter. *)
  let[@inline] num_occurrences bits : Num_occurrences.t =
    if bits = 0
    then Zero
    else if bits land (bits - 1) = 0
    then One
    else More_than_one

  let[@inline] count t role = num_occurrences (t land counters_of_role role)

  let[@inline] count_normal t role =
    num_occurrences (counter t role Name_mode.normal)

  let[@inline] greatest_name_mode t role : Name_mode.Or_absent.t =
    if counter t role Name_mode.normal <> 0
    then Name_mode.Or_absent.present Name_mode.normal
    else if counter t role Name_mode.in_types <> 0
    then Name_mode.Or_absent.present Name_mode.in_types
    else if counter t role Name_mode.phantom <> 0
    then Name_mode.Or_absent.present Name_mode.phantom
    else Name_mode.Or_absent.absent

  let downgrade t (name_mode : Name_mode.t) =
    match name_mode with
    | Normal -> t
    | In_types | Phantom ->
      let shift = mode_shift name_mode in
      let normal = t land (0x3f lsl 3) in
      let other = ((t lsr shift) land 0x3f) lsl 3 in
      t land presence_mask lor (union normal other lsl (shift - 3))

  let[@inline] mask_of_present_roles t =
    (if mem t 0 then role_mask 0 else 0)
    lor (if mem t 1 then role_mask 1 else 0)
    lor if mem t 2 then role_mask 2 else 0

  (* Removes the roles present in [t2] from [t1] *)
  let[@inline] diff t1 t2 =
    let t = t1 land lnot (mask_of_present_roles t2) in
    if t land presence_mask = 0 then None else Some t

  let[@inline] present_roles_included t1 t2 =
    t1 land presence_mask land lnot t2 = 0

  let[@inline] present_roles_intersect t1 t2 =
    t1 land t2 land presence_mask <> 0

  (* Merges the occurrences of role 1 into role 0 *)
  let promote_role_1 t =
    let role_1 = t land role_mask 1 in
    let role_0 =
      ((role_1 land counters_mask) lsr 2) lor ((role_1 land presence_mask) lsr 1)
    in
    union (t land lnot (role_mask 1)) role_0

  let print_counter ppf c =
    Format.pp_print_string ppf
      (match c with 0 -> "0" | 1 -> "1" | _ -> "many")

  let print ppf t =
    Format.fprintf ppf "@[<hov 1>(";
    for role = 0 to 2 do
      if mem t role
      then
        Format.fprintf ppf
          "@[<hov 1>(role %d@ normal %a@ in_types %a@ phantom %a)@]" role
          print_counter
          (counter t role Name_mode.normal)
          print_counter
          (counter t role Name_mode.in_types)
          print_counter
          (counter t role Name_mode.phantom)
    done;
    Format.fprintf ppf ")@]"
end

let print_key ppf key =
  let flags = Id.flags key in
  if flags = Flags.continuation
  then Continuation.print ppf (Obj.magic key : Continuation.t)
  else if flags = Flags.code_id
  then Code_id.print ppf (Obj.magic key : Code_id.t)
  else Name.print ppf (Obj.magic key : Name.t)

module M = Patricia_tree.Make (struct
  let print = print_key
end)

module Make_ops (Map : Container_types.Map) = struct
  let[@inline] add_counts map key counts =
    Map.update key
      (function
        | None -> Some counts
        | Some counts' -> Some (Counts.union counts counts'))
      map

  let[@inline] add map key role name_mode =
    add_counts map key (Counts.one role name_mode)

  let union map1 map2 =
    Map.union_total (fun _ c1 c2 -> Counts.union c1 c2) map1 map2

  let equal map1 map2 = Map.equal Int.equal map1 map2

  let diff map1 map2 =
    Map.diff_sharing (fun _ c1 c2 -> Counts.diff c1 c2) map1 map2

  let subset_domain map1 map2 =
    Map.for_all
      (fun key c1 ->
        match Map.find_or_null key map2 with
        | Null -> false
        | This c2 -> Counts.present_roles_included c1 c2)
      map1

  let inter_domain_is_non_empty map1 map2 =
    Map.inter_domain_is_non_empty map1 map2
    && Map.exists
         (fun key c1 ->
           match Map.find_or_null key map2 with
           | Null -> false
           | This c2 -> Counts.present_roles_intersect c1 c2)
         map1

  let downgrade map (name_mode : Name_mode.t) =
    match name_mode with
    | Normal -> map
    | In_types | Phantom ->
      Map.map_sharing (fun c -> Counts.downgrade c name_mode) map

  let increase_counts map = Map.map_sharing Counts.increase map

  let mem map key role =
    match Map.find_or_null key map with
    | Null -> false
    | This c -> Counts.mem c role

  let mem_any map key roles =
    match Map.find_or_null key map with
    | Null -> false
    | This c -> Counts.mem_any c roles

  let count map key role : Num_occurrences.t =
    match Map.find_or_null key map with
    | Null -> Zero
    | This c -> Counts.count c role

  let count_normal map key role : Num_occurrences.t =
    match Map.find_or_null key map with
    | Null -> Zero
    | This c -> Counts.count_normal c role

  let keys_with_any_role map roles =
    Map.fold
      (fun key c acc ->
        if Counts.mem_any c roles then Map.Set.add key acc else acc)
      map Map.Set.empty

  let keys_at_normal_mode map normal_counters acc =
    Map.fold
      (fun key c acc ->
        if c land normal_counters <> 0 then Map.Set.add key acc else acc)
      map acc
end

module Ids = Make_ops (M.Map)
module Function_slots = Make_ops (Function_slot.Map)
module Value_slots = Make_ops (Value_slot.Map)

let name_role = 0

let continuation_role = 0

let continuation_with_traps_role = 1

let continuation_in_trap_actions_role = 2

let code_id_role = 0

let newer_version_of_code_id_role = 1

let in_projections_role = 0

let in_declarations_role = 1

let[@inline] name_key (name : Name.t) = (name :> int)

let[@inline] var_key (var : Variable.t) = (var :> int)

let[@inline] symbol_key (symbol : Symbol.t) = (symbol :> int)

let[@inline] continuation_key (cont : Continuation.t) = (cont :> int)

let[@inline] code_id_key (code_id : Code_id.t) = (code_id :> int)

let[@inline] restrict flags ids =
  M.Map.restrict_to_top_bits ~num_bits:Id.flags_size_in_bits ~top_bits:flags ids

let () = assert (Flags.variable lsr 1 = Flags.symbol lsr 1)

(* The variables and symbols *)
let[@inline] names_part ids =
  M.Map.restrict_to_top_bits
    ~num_bits:(Id.flags_size_in_bits - 1)
    ~top_bits:(Flags.variable lsr 1) ids

let[@inline] disjoint_union ids1 ids2 =
  M.Map.union_total (fun _ c _ -> c) ids1 ids2

type t =
  { ids : Counts.t M.Map.t;
        (* Names, continuations and code IDs. The roles are:

           - for continuations, [continuation_role],
           [continuation_with_traps_role] and
           [continuation_in_trap_actions_role];

           - for code IDs, [code_id_role] and [newer_version_of_code_id_role]
           (for code IDs that occur in "newer version of" fields, e.g. in
           [Flambda_static.Static_part.code]). *)
    function_slots : Counts.t Function_slot.Map.t;
    value_slots : Counts.t Value_slot.Map.t
        (* The roles for slots are [in_projections_role] and
           [in_declarations_role]. *)
  }

let empty =
  { ids = M.Map.empty;
    function_slots = Function_slot.Map.empty;
    value_slots = Value_slot.Map.empty
  }

let is_empty { ids; function_slots; value_slots } =
  M.Map.is_empty ids
  && Function_slot.Map.is_empty function_slots
  && Value_slot.Map.is_empty value_slots

let[@inline] singleton_id key role name_mode =
  { empty with ids = M.Map.singleton key (Counts.one role name_mode) }

let singleton_continuation cont =
  singleton_id (continuation_key cont) continuation_role Name_mode.normal

let singleton_continuation_in_trap_action cont =
  singleton_id (continuation_key cont) continuation_in_trap_actions_role
    Name_mode.normal

let add_continuation t cont ~has_traps =
  let counts = Counts.one continuation_role Name_mode.normal in
  let counts =
    if has_traps
    then
      Counts.union counts
        (Counts.one continuation_with_traps_role Name_mode.normal)
    else counts
  in
  { t with ids = Ids.add_counts t.ids (continuation_key cont) counts }

let count_continuation t cont =
  Ids.count t.ids (continuation_key cont) continuation_role

let continuation_is_applied_with_traps t cont =
  Ids.mem t.ids (continuation_key cont) continuation_with_traps_role

let count_variable t var = Ids.count t.ids (var_key var) name_role

let count_variable_normal_mode t var =
  Ids.count_normal t.ids (var_key var) name_role

let singleton_variable var name_mode =
  singleton_id (var_key var) name_role name_mode

let add_variable t var name_mode =
  { t with ids = Ids.add t.ids (var_key var) name_role name_mode }

let add_symbol t sym name_mode =
  { t with ids = Ids.add t.ids (symbol_key sym) name_role name_mode }

let add_name t name name_mode =
  { t with ids = Ids.add t.ids (name_key name) name_role name_mode }

let add_function_slot_in_projection t function_slot name_mode =
  { t with
    function_slots =
      Function_slots.add t.function_slots function_slot in_projections_role
        name_mode
  }

let add_value_slot_in_projection t value_slot name_mode =
  { t with
    value_slots =
      Value_slots.add t.value_slots value_slot in_projections_role name_mode
  }

let add_function_slot_in_declaration t function_slot name_mode =
  { t with
    function_slots =
      Function_slots.add t.function_slots function_slot in_declarations_role
        name_mode
  }

let add_value_slot_in_declaration t value_slot name_mode =
  { t with
    value_slots =
      Value_slots.add t.value_slots value_slot in_declarations_role name_mode
  }

let in_types_in_both_roles =
  Counts.union
    (Counts.one in_projections_role Name_mode.in_types)
    (Counts.one in_declarations_role Name_mode.in_types)

let add_function_slot_in_types t function_slot =
  { t with
    function_slots =
      Function_slots.add_counts t.function_slots function_slot
        in_types_in_both_roles
  }

let add_value_slot_in_types t value_slot =
  { t with
    value_slots =
      Value_slots.add_counts t.value_slots value_slot in_types_in_both_roles
  }

let add_code_id t code_id name_mode =
  { t with ids = Ids.add t.ids (code_id_key code_id) code_id_role name_mode }

let singleton_code_id code_id name_mode =
  singleton_id (code_id_key code_id) code_id_role name_mode

let add_newer_version_of_code_id t code_id name_mode =
  { t with
    ids =
      Ids.add t.ids (code_id_key code_id) newer_version_of_code_id_role
        name_mode
  }

let singleton_symbol sym name_mode =
  singleton_id (symbol_key sym) name_role name_mode

let singleton_name name name_mode =
  singleton_id (name_key name) name_role name_mode

let create_variables vars name_mode =
  let counts = Counts.one name_role name_mode in
  let ids =
    Variable.Set.fold
      (fun var ids -> M.Map.add (var_key var) counts ids)
      vars M.Map.empty
  in
  { empty with ids }

let create_names names name_mode =
  let counts = Counts.one name_role name_mode in
  let ids =
    Name.Set.fold
      (fun name ids -> M.Map.add (name_key name) counts ids)
      names M.Map.empty
  in
  { empty with ids }

let diff_ids key c1 c2 =
  (* Note special case here: code IDs in [t2] are also removed from the "newer
     version of" code IDs in [t1]. *)
  let c2 =
    if Id.flags key = Flags.code_id && Counts.mem c2 code_id_role
    then
      Counts.union c2
        (Counts.one newer_version_of_code_id_role Name_mode.normal)
    else c2
  in
  Counts.diff c1 c2

let diff t1 ~without:t2 =
  { ids = M.Map.diff_sharing diff_ids t1.ids t2.ids;
    function_slots = Function_slots.diff t1.function_slots t2.function_slots;
    value_slots = Value_slots.diff t1.value_slots t2.value_slots
  }

let union t1 t2 =
  if is_empty t1
  then t2
  else if is_empty t2
  then t1
  else
    { ids = Ids.union t1.ids t2.ids;
      function_slots = Function_slots.union t1.function_slots t2.function_slots;
      value_slots = Value_slots.union t1.value_slots t2.value_slots
    }

let equal t1 t2 =
  Ids.equal t1.ids t2.ids
  && Function_slots.equal t1.function_slots t2.function_slots
  && Value_slots.equal t1.value_slots t2.value_slots

let no_variables t = M.Map.is_empty (restrict Flags.variable t.ids)

let continuation_roles =
  Counts.presence continuation_role
  lor Counts.presence continuation_in_trap_actions_role

let no_continuations t =
  (* Note: continuations with traps are included in continuations *)
  M.Map.for_all
    (fun _ c -> not (Counts.mem_any c continuation_roles))
    (restrict Flags.continuation t.ids)

let subset_domain t1 t2 =
  Ids.subset_domain t1.ids t2.ids
  && Function_slots.subset_domain t1.function_slots t2.function_slots
  && Value_slots.subset_domain t1.value_slots t2.value_slots

let inter_domain_is_non_empty t1 t2 =
  Ids.inter_domain_is_non_empty t1.ids t2.ids
  || Function_slots.inter_domain_is_non_empty t1.function_slots
       t2.function_slots
  || Value_slots.inter_domain_is_non_empty t1.value_slots t2.value_slots

let union_list ts = List.fold_left union empty ts

let function_slots_in_normal_projections t =
  Function_slots.keys_at_normal_mode t.function_slots
    (Counts.normal_counter in_projections_role)
    Function_slot.Set.empty

let all_function_slots_at_normal_mode t =
  Function_slots.keys_at_normal_mode t.function_slots
    (Counts.normal_counter in_projections_role
    lor Counts.normal_counter in_declarations_role)
    Function_slot.Set.empty

let value_slots_in_normal_projections t =
  Value_slots.keys_at_normal_mode t.value_slots
    (Counts.normal_counter in_projections_role)
    Value_slot.Set.empty

let all_value_slots_at_normal_mode t =
  Value_slots.keys_at_normal_mode t.value_slots
    (Counts.normal_counter in_projections_role
    lor Counts.normal_counter in_declarations_role)
    Value_slot.Set.empty

let symbols t : Symbol.Set.t =
  Obj.magic (M.Map.keys (restrict Flags.symbol t.ids))

let continuations_with_any_role t roles : Continuation.Set.t =
  Obj.magic (Ids.keys_with_any_role (restrict Flags.continuation t.ids) roles)

let continuations t =
  continuations_with_any_role t (Counts.presence continuation_role)

let continuations_with_traps t =
  continuations_with_any_role t (Counts.presence continuation_with_traps_role)

let continuations_including_in_trap_actions t =
  continuations_with_any_role t continuation_roles

let code_ids_with_any_role t roles : Code_id.Set.t =
  Obj.magic (Ids.keys_with_any_role (restrict Flags.code_id t.ids) roles)

let code_ids t = code_ids_with_any_role t (Counts.presence code_id_role)

let newer_version_of_code_ids t =
  code_ids_with_any_role t (Counts.presence newer_version_of_code_id_role)

let code_ids_and_newer_version_of_code_ids t : Code_id.Set.t =
  Obj.magic (M.Map.keys (restrict Flags.code_id t.ids))

let mem_name t name = M.Map.mem (name_key name) t.ids

let mem_var t var = M.Map.mem (var_key var) t.ids

let mem_symbol t symbol = M.Map.mem (symbol_key symbol) t.ids

let mem_code_id t code_id = Ids.mem t.ids (code_id_key code_id) code_id_role

let mem_continuation t cont =
  Ids.mem_any t.ids (continuation_key cont) continuation_roles

let value_slot_is_used_or_imported t value_slot =
  Value_slot.is_imported value_slot
  || Value_slots.mem t.value_slots value_slot in_projections_role

let remove_key t key =
  if M.Map.is_empty t.ids then t else { t with ids = M.Map.remove key t.ids }

let remove_var t ~var = remove_key t (var_key var)

let remove_var_opt t ~var =
  match var with None -> t | Some var -> remove_var t ~var

let remove_code_id_or_symbol t ~(code_id_or_symbol : Code_id_or_symbol.t) =
  remove_key t (code_id_or_symbol :> int)

let remove_continuation t ~continuation =
  remove_key t (continuation_key continuation)

let greatest_name_mode_var t var : Name_mode.Or_absent.t =
  match M.Map.find_or_null (var_key var) t.ids with
  | Null -> Name_mode.Or_absent.absent
  | This c -> Counts.greatest_name_mode c name_role

let downgrade_occurrences_at_strictly_greater_name_mode
    ({ ids; function_slots; value_slots } as t) (max_name_mode : Name_mode.t) =
  match max_name_mode with
  | Normal -> t
  | In_types | Phantom ->
    { ids = Ids.downgrade ids max_name_mode;
      function_slots = Function_slots.downgrade function_slots max_name_mode;
      value_slots = Value_slots.downgrade value_slots max_name_mode
    }

let with_only_variables t = { empty with ids = restrict Flags.variable t.ids }

let with_only_names t = { empty with ids = names_part t.ids }

let with_only_names_and_code_ids_promoting_newer_version_of t =
  let code_ids =
    M.Map.map_sharing Counts.promote_role_1 (restrict Flags.code_id t.ids)
  in
  { empty with ids = disjoint_union (names_part t.ids) code_ids }

let without_names_or_continuations t =
  { t with ids = restrict Flags.code_id t.ids }

let without_code_ids t =
  { t with
    ids = disjoint_union (names_part t.ids) (restrict Flags.continuation t.ids)
  }

let fold_names t ~init ~f =
  M.Map.fold
    (fun key _ acc -> f acc (Obj.magic key : Name.t))
    (names_part t.ids) init

let fold_variables t ~init ~f =
  M.Map.fold
    (fun key _ acc -> f acc (Obj.magic key : Variable.t))
    (restrict Flags.variable t.ids)
    init

let fold_continuations_with_role continuations role ~init ~f =
  M.Map.fold
    (fun key c acc ->
      if Counts.mem c role then f acc (Obj.magic key : Continuation.t) else acc)
    continuations init

let fold_continuations_including_in_trap_actions t ~init ~f =
  (* Note: continuations with traps are included in continuations *)
  let continuations = restrict Flags.continuation t.ids in
  let init =
    fold_continuations_with_role continuations continuation_role ~init ~f
  in
  fold_continuations_with_role continuations continuation_in_trap_actions_role
    ~init ~f

let fold_code_ids t ~init ~f =
  M.Map.fold
    (fun key c acc ->
      if Counts.mem c code_id_role
      then f acc (Obj.magic key : Code_id.t)
      else acc)
    (restrict Flags.code_id t.ids)
    init

let apply_renaming_to_key renaming key =
  let flags = Id.flags key in
  if flags = Flags.continuation
  then
    (Renaming.apply_continuation renaming (Obj.magic key : Continuation.t)
      : Continuation.t
      :> int)
  else if flags = Flags.code_id
  then
    (Renaming.apply_code_id renaming (Obj.magic key : Code_id.t)
      : Code_id.t
      :> int)
  else (Renaming.apply_name renaming (Obj.magic key : Name.t) : Name.t :> int)

let ids_affected_by_renaming ids renaming =
  not
    (M.Map.for_all (fun key _ -> apply_renaming_to_key renaming key = key) ids)

let apply_renaming t renaming =
  if Renaming.is_identity renaming || M.Map.is_empty t.ids
  then t
  else
    let rename part =
      if ids_affected_by_renaming part renaming
      then M.Map.map_keys (apply_renaming_to_key renaming) part
      else part
    in
    let names = names_part t.ids in
    let continuations = restrict Flags.continuation t.ids in
    let code_ids = restrict Flags.code_id t.ids in
    let names' = rename names in
    let continuations' = rename continuations in
    let code_ids' = rename code_ids in
    if
      names == names'
      && continuations == continuations'
      && code_ids == code_ids'
    then t
    else
      { t with
        ids = disjoint_union names' (disjoint_union continuations' code_ids')
      }

let affected_by_renaming t renaming = ids_affected_by_renaming t.ids renaming

let restrict_to_value_slots_and_function_slots t = { t with ids = M.Map.empty }

let ids_for_export t = Ids_for_export.of_int_set (M.Map.keys t.ids)

let increase_counts { ids; function_slots; value_slots } =
  { ids = Ids.increase_counts ids;
    function_slots = Function_slots.increase_counts function_slots;
    value_slots = Value_slots.increase_counts value_slots
  }

let print ppf ({ ids; function_slots; value_slots } as t) =
  if is_empty t
  then Format.fprintf ppf "no_occurrences"
  else
    Format.fprintf ppf
      "@[<hov 1>@[<hov 1>(ids %a)@]@ @[<hov 1>(function_slots %a)@]@ @[<hov \
       1>(value_slots %a)@]@]"
      (M.Map.print Counts.print) ids
      (Function_slot.Map.print Counts.print)
      function_slots
      (Value_slot.Map.print Counts.print)
      value_slots
