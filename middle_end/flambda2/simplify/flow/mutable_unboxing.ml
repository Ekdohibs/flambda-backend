(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*    Pierre Chambart and Guillaume Bury, OCamlPro                        *)
(*                                                                        *)
(*   Copyright 2021--2021 OCamlPro SAS                                    *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

module T = Flow_types
module EPA = Continuation_extra_params_and_args

let ref_to_var_debug = Sys.getenv_opt "RTV" <> None

(* Types *)

type non_escaping_block =
  { tag : Tag.t;
    fields_kinds : Flambda_kind.With_subkind.t list
  }

type extra_params = Bound_parameter.t list

type extra_args =
  Simple.t Numeric_types.Int.Map.t Apply_cont_rewrite_id.Map.t
  Continuation.Map.t

type t =
  { non_escaping_makeblocks : non_escaping_block Variable.Map.t;
    continuations_with_live_block : Variable.Set.t Continuation.Map.t;
    extra_params_and_args : (extra_params * extra_args) Continuation.Map.t;
    rewrites : Named_rewrite.t Named_rewrite_id.Map.t;
    required_regions : Name.Set.t
  }

(* Escaping analysis *)

let free_names_of_apply_cont_args
    (apply_cont_args :
      T.Cont_arg.t Numeric_types.Int.Map.t Apply_cont_rewrite_id.Map.t) :
    Variable.Set.t =
  Apply_cont_rewrite_id.Map.fold
    (fun _ args free ->
      Numeric_types.Int.Map.fold
        (fun _ (arg : T.Cont_arg.t) free ->
          match arg with
          | Function_result -> free
          | New_let_binding _ ->
            free (* Doesn't really matter, already escaping *)
          | Simple s -> (
            match Simple.must_be_var s with
            | None -> free
            | Some (var, _coer) -> Variable.Set.add var free))
        args free)
    apply_cont_args Variable.Set.empty

let escaping_by_alias ~(dom : Dominator_graph.alias_map)
    ~(dom_graph : Dominator_graph.t) =
  Variable.Map.fold
    (fun flow_to flow_from escaping ->
      let alias_to =
        match Variable.Map.find flow_to dom with
        | exception Not_found -> flow_to
        | v -> v
      in
      let new_escaping =
        Variable.Set.fold
          (fun flow_from escaping ->
            let alias_from =
              match Variable.Map.find flow_from dom with
              | exception Not_found -> flow_from
              | v -> v
            in
            let is_escaping = not (Variable.equal alias_to alias_from) in
            if is_escaping
            then Variable.Set.add alias_from escaping
            else escaping)
          flow_from Variable.Set.empty
      in
      Variable.Set.union escaping new_escaping)
    dom_graph.graph Variable.Set.empty

let names_escaping_from_mutable_prim (prim : T.Mutable_prim.t) =
  match prim with
  | Make_block { fields; _ } -> Simple.List.free_names fields
  | Block_set { value; _ } -> Simple.free_names value
  | Is_int _ | Get_tag _ | Block_load _ -> Name_occurrences.empty

let escaping_by_use_for_one_continuation ~required_names
    ~(dom : Dominator_graph.alias_map) (elt : T.Continuation_info.t) =
  let add_name_occurrences occurrences init =
    Name_occurrences.fold_variables occurrences ~init ~f:(fun escaping var ->
        let escaping =
          match Variable.Map.find var dom with
          | exception Not_found -> escaping
          | var -> Variable.Set.add var escaping
        in
        Variable.Set.add var escaping)
  in
  let escaping = add_name_occurrences elt.used_in_handler Variable.Set.empty in
  let escaping =
    Name.Map.fold
      (fun name deps escaping ->
        if Name.Set.mem name required_names
        then add_name_occurrences deps escaping
        else escaping)
      elt.bindings escaping
  in
  let escaping =
    Value_slot.Map.fold
      (fun _value_slot map escaping ->
        Name.Map.fold
          (fun _closure_name values_in_env escaping ->
            add_name_occurrences values_in_env escaping)
          map escaping)
      elt.value_slots escaping
  in
  let escaping =
    List.fold_left
      (fun escaping T.Mutable_let_prim.{ prim; _ } ->
        add_name_occurrences (names_escaping_from_mutable_prim prim) escaping)
      escaping elt.mutable_let_prims_rev
  in
  escaping

let escaping_by_use ~required_names ~(dom : Dominator_graph.alias_map)
    ~(source_info : T.Acc.t) =
  Continuation.Map.fold
    (fun _cont elt escaping ->
      Variable.Set.union
        (escaping_by_use_for_one_continuation ~required_names ~dom elt)
        escaping)
    source_info.map Variable.Set.empty

let escaping_by_return ~(dom : Dominator_graph.alias_map)
    ~(source_info : T.Acc.t) ~return_continuation ~exn_continuation =
  Continuation.Map.fold
    (fun _cont (elt : T.Continuation_info.t) escaping ->
      let add_escaping cont escaping =
        match Continuation.Map.find cont elt.apply_cont_args with
        | exception Not_found -> escaping
        | apply_cont_args ->
          Variable.Set.fold
            (fun var escaping ->
              let escaping =
                match Variable.Map.find var dom with
                | exception Not_found -> escaping
                | var -> Variable.Set.add var escaping
              in
              Variable.Set.add var escaping)
            (free_names_of_apply_cont_args apply_cont_args)
            escaping
      in
      let escaping = add_escaping return_continuation escaping in
      add_escaping exn_continuation escaping)
    source_info.map Variable.Set.empty

let escaping ~(dom : Dominator_graph.alias_map) ~(dom_graph : Dominator_graph.t)
    ~(source_info : T.Acc.t) ~return_continuation ~exn_continuation
    ~required_names =
  let escaping_by_alias = escaping_by_alias ~dom ~dom_graph in
  let escaping_by_use = escaping_by_use ~required_names ~dom ~source_info in
  let escaping_by_return =
    escaping_by_return ~dom ~source_info ~return_continuation ~exn_continuation
  in
  Variable.Set.union escaping_by_alias
    (Variable.Set.union escaping_by_return escaping_by_use)

(* *)

let non_escaping_makeblocks_and_required_regions ~escaping ~source_info =
  Continuation.Map.fold
    (fun _cont (elt : T.Continuation_info.t) (map, regions) ->
      List.fold_left
        (fun (map, regions) T.Mutable_let_prim.{ bound_var = var; prim; _ } ->
          match prim with
          | Block_load _ | Block_set _ | Is_int _ | Get_tag _ -> map, regions
          | Make_block { kind; alloc_mode; fields; _ } ->
            if Variable.Set.mem var escaping
            then
              let regions =
                match alloc_mode with
                | Heap -> regions
                | Local { region } -> Name.Set.add (Name.var region) regions
              in
              map, regions
            else
              let non_escaping_block =
                match kind with
                | Values (tag, fields_kinds) ->
                  { tag = Tag.Scannable.to_tag tag; fields_kinds }
                | Naked_floats ->
                  { tag = Tag.double_array_tag;
                    fields_kinds =
                      List.map
                        (fun _ -> Flambda_kind.With_subkind.naked_float)
                        fields
                  }
              in
              Variable.Map.add var non_escaping_block map, regions)
        (map, regions) elt.mutable_let_prims_rev)
    source_info.T.Acc.map
    (Variable.Map.empty, Name.Set.empty)

let prims_using_block ~non_escaping_blocks ~dom prim =
  match (prim : T.Mutable_prim.t) with
  | Make_block _ -> Variable.Set.empty
  | Is_int block
  | Get_tag block
  | Block_set { block; _ }
  | Block_load { block; _ } ->
    let block =
      match Variable.Map.find block dom with
      | exception Not_found -> block
      | block -> block
    in
    if Variable.Map.mem block non_escaping_blocks
    then Variable.Set.singleton block
    else Variable.Set.empty

let continuations_using_blocks ~non_escaping_blocks ~dom
    ~(source_info : T.Acc.t) =
  Continuation.Map.mapi
    (fun _cont (elt : T.Continuation_info.t) ->
      let used =
        List.fold_left
          (fun used_block T.Mutable_let_prim.{ prim; _ } ->
            Variable.Set.union used_block
              (prims_using_block ~non_escaping_blocks ~dom prim))
          Variable.Set.empty elt.mutable_let_prims_rev
      in
      if (not (Variable.Set.is_empty used)) && ref_to_var_debug
      then
        Format.printf "Cont using block %a %a@." Continuation.print _cont
          Variable.Set.print used;
      used)
    source_info.map

let continuations_defining_blocks ~non_escaping_blocks ~(source_info : T.Acc.t)
    =
  Continuation.Map.mapi
    (fun _cont (elt : T.Continuation_info.t) ->
      let defined =
        List.fold_left
          (fun defined_blocks T.Mutable_let_prim.{ bound_var = var; _ } ->
            if Variable.Map.mem var non_escaping_blocks
            then Variable.Set.add var defined_blocks
            else defined_blocks)
          Variable.Set.empty elt.mutable_let_prims_rev
      in
      if (not (Variable.Set.is_empty defined)) && ref_to_var_debug
      then
        Format.printf "Cont defining block %a %a@." Continuation.print _cont
          Variable.Set.print defined;
      defined)
    source_info.map

let continuations_with_live_block ~non_escaping_blocks ~dom ~source_info
    ~(control_flow_graph : Control_flow_graph.t) =
  let continuations_defining_blocks =
    continuations_defining_blocks ~non_escaping_blocks ~source_info
  in
  let continuations_using_blocks =
    continuations_using_blocks ~non_escaping_blocks ~dom ~source_info
  in
  let continuations_using_blocks_but_not_defining_them =
    Continuation.Map.merge
      (fun _cont defined used ->
        match defined, used with
        | None, None -> assert false (* *)
        | None, Some _used ->
          Misc.fatal_errorf
            "In Data_flow: incomplete map of continuation defining blocks"
        | Some _defined, None ->
          Misc.fatal_errorf
            "In Data_flow: incomplete map of continuation using blocks"
        | Some defined, Some used -> Some (Variable.Set.diff used defined))
      continuations_defining_blocks continuations_using_blocks
  in
  Control_flow_graph.fixpoint control_flow_graph
    ~init:continuations_using_blocks_but_not_defining_them
    ~f:(fun
         ~caller
         ~caller_set:old_using_blocks
         ~callee:_
         ~callee_set:used_blocks
       ->
      let defined_blocks =
        Continuation.Map.find caller continuations_defining_blocks
      in
      Variable.Set.diff
        (Variable.Set.union old_using_blocks used_blocks)
        defined_blocks)

let list_to_int_map l =
  let _, map =
    List.fold_left
      (fun (i, fields) elt ->
        let fields = Numeric_types.Int.Map.add i elt fields in
        i + 1, fields)
      (0, Numeric_types.Int.Map.empty)
      l
  in
  map

let int_map_to_list m = List.map snd (Numeric_types.Int.Map.bindings m)

module Fold_prims = struct
  type env =
    { bindings : Simple.t Numeric_types.Int.Map.t Variable.Map.t;
      (* non escaping / unboxed blocks fields values *)
      rewrites : Named_rewrite.t Named_rewrite_id.Map.t
    }

  let with_unboxed_block ~block ~dom ~env ~non_escaping_blocks ~f =
    let block =
      match Variable.Map.find block dom with
      | exception Not_found -> block
      | block -> block
    in
    match Variable.Map.find block non_escaping_blocks with
    | exception Not_found -> env
    | { tag; fields_kinds } -> f ~tag ~fields_kinds

  let with_unboxed_fields ~block ~dom ~env ~f =
    let block =
      match Variable.Map.find block dom with
      | exception Not_found -> block
      | block -> block
    in
    match Variable.Map.find block env.bindings with
    | exception Not_found -> env
    | fields -> f fields

  let apply_prim ~dom ~non_escaping_blocks env rewrite_id var
      (prim : T.Mutable_prim.t) =
    match prim with
    | Is_int block ->
      with_unboxed_fields ~block ~dom ~env ~f:(fun _fields ->
          (* We only consider for unboxing vluaes which are aliases to a single
             makeblock. In particular, for variants, this means that we only
             consider for unboxing variant values which are blocks. *)
          let bound_to = Simple.untagged_const_bool false in
          let rewrite =
            Named_rewrite.prim_rewrite
              (Named_rewrite.Prim_rewrite.replace_by_binding ~var ~bound_to)
          in
          { env with
            rewrites = Named_rewrite_id.Map.add rewrite_id rewrite env.rewrites
          })
    | Get_tag block ->
      with_unboxed_block ~block ~dom ~env ~non_escaping_blocks
        ~f:(fun ~tag ~fields_kinds:_ ->
          let bound_to =
            Simple.untagged_const_int (Tag.to_targetint_31_63 tag)
          in
          let rewrite =
            Named_rewrite.prim_rewrite
              (Named_rewrite.Prim_rewrite.replace_by_binding ~var ~bound_to)
          in
          { env with
            rewrites = Named_rewrite_id.Map.add rewrite_id rewrite env.rewrites
          })
    | Block_load { block; field; bak; _ } ->
      with_unboxed_fields ~block ~dom ~env ~f:(fun fields ->
          let rewrite =
            match Numeric_types.Int.Map.find field fields with
            | bound_to ->
              Named_rewrite.prim_rewrite
                (Named_rewrite.Prim_rewrite.replace_by_binding ~var ~bound_to)
            | exception Not_found ->
              let k =
                Flambda_primitive.Block_access_kind.element_kind_for_load bak
              in
              Named_rewrite.prim_rewrite (Named_rewrite.Prim_rewrite.invalid k)
          in
          { env with
            rewrites = Named_rewrite_id.Map.add rewrite_id rewrite env.rewrites
          })
    | Block_set { bak; block; field; value } ->
      with_unboxed_fields ~block ~dom ~env ~f:(fun fields ->
          if ref_to_var_debug
          then Format.printf "Remove Block set %a@." Variable.print var;
          let rewrite, fields =
            if Numeric_types.Int.Map.mem field fields
            then
              ( Named_rewrite.prim_rewrite Named_rewrite.Prim_rewrite.remove_prim,
                Numeric_types.Int.Map.add field value fields )
            else
              let k =
                Flambda_primitive.Block_access_kind.element_kind_for_load bak
              in
              ( Named_rewrite.prim_rewrite (Named_rewrite.Prim_rewrite.invalid k),
                fields )
          in
          { bindings = Variable.Map.add block fields env.bindings;
            rewrites = Named_rewrite_id.Map.add rewrite_id rewrite env.rewrites
          })
    | Make_block { fields; _ } ->
      if not (Variable.Map.mem var non_escaping_blocks)
      then env
      else
        let () =
          if ref_to_var_debug
          then Format.printf "Remove Makeblock %a@." Variable.print var
        in
        let rewrite =
          Named_rewrite.prim_rewrite Named_rewrite.Prim_rewrite.remove_prim
        in
        let fields = list_to_int_map fields in
        { bindings = Variable.Map.add var fields env.bindings;
          rewrites = Named_rewrite_id.Map.add rewrite_id rewrite env.rewrites
        }

  let init_env ~(non_escaping_blocks : non_escaping_block Variable.Map.t)
      ~(blocks_needed : Variable.Set.t) ~rewrites =
    let env, params =
      Variable.Set.fold
        (fun block_needed (env, params) ->
          let { tag = _; fields_kinds } =
            Variable.Map.find block_needed non_escaping_blocks
          in
          let block_params =
            List.mapi
              (fun i kind ->
                let name = Variable.unique_name block_needed in
                let var = Variable.create (Printf.sprintf "%s_%i" name i) in
                Bound_parameter.create var kind)
              fields_kinds
          in
          let env =
            let fields =
              list_to_int_map
                (List.map
                   (fun bp -> Simple.var (Bound_parameter.var bp))
                   block_params)
            in
            { env with
              bindings = Variable.Map.add block_needed fields env.bindings
            }
          in
          env, List.rev_append block_params params)
        blocks_needed
        ({ bindings = Variable.Map.empty; rewrites }, [])
    in
    env, List.rev params

  let append_int_map i1 i2 =
    if Numeric_types.Int.Map.is_empty i1
    then i2
    else
      let max, _ = Numeric_types.Int.Map.max_binding i1 in
      let shifted_i2 =
        Numeric_types.Int.Map.map_keys (fun i -> i + max + 1) i2
      in
      Numeric_types.Int.Map.disjoint_union i1 shifted_i2

  let compute_rewrites
      ~(non_escaping_blocks : non_escaping_block Variable.Map.t)
      ~continuations_with_live_block ~dom ~(source_info : T.Acc.t) =
    let rewrites = ref Named_rewrite_id.Map.empty in
    let extra_params_and_args =
      Continuation.Map.mapi
        (fun cont (blocks_needed : Variable.Set.t) ->
          let elt = Continuation.Map.find cont source_info.map in
          let env, extra_block_params =
            init_env ~non_escaping_blocks ~blocks_needed ~rewrites:!rewrites
          in
          let env =
            List.fold_left
              (fun env T.Mutable_let_prim.{ named_rewrite_id; bound_var; prim } ->
                apply_prim ~dom ~non_escaping_blocks env named_rewrite_id
                  bound_var prim)
              env
              (List.rev elt.mutable_let_prims_rev)
          in
          rewrites := env.rewrites;
          let blocks_params_to_add cont rewrites =
            Apply_cont_rewrite_id.Map.map
              (fun _args ->
                match
                  Continuation.Map.find cont continuations_with_live_block
                with
                | exception Not_found -> Numeric_types.Int.Map.empty
                | blocks_needed ->
                  let extra_args =
                    Variable.Set.fold
                      (fun block_needed extra_args ->
                        let args =
                          Variable.Map.find block_needed env.bindings
                        in
                        append_int_map extra_args args)
                      blocks_needed Numeric_types.Int.Map.empty
                  in
                  extra_args)
              rewrites
          in
          let new_apply_cont_args =
            Continuation.Map.mapi blocks_params_to_add elt.apply_cont_args
          in
          extra_block_params, new_apply_cont_args)
        continuations_with_live_block
    in
    extra_params_and_args, !rewrites
end

let create ~(dom : Dominator_graph.alias_map) ~(dom_graph : Dominator_graph.t)
    ~(source_info : T.Acc.t) ~(control_flow_graph : Control_flow_graph.t)
    ~required_names ~return_continuation ~exn_continuation : t =
  let escaping =
    escaping ~dom ~dom_graph ~source_info ~required_names ~return_continuation
      ~exn_continuation
  in
  let non_escaping_blocks, required_regions =
    non_escaping_makeblocks_and_required_regions ~escaping ~source_info
  in
  if (not (Variable.Map.is_empty non_escaping_blocks)) && ref_to_var_debug
  then
    Format.printf "Non escaping makeblocks %a@."
      (Variable.Map.print (fun ppf { tag; fields_kinds } ->
           Format.fprintf ppf "{%a}[%a]" Tag.print tag
             (Format.pp_print_list ~pp_sep:Format.pp_print_space
                Flambda_kind.With_subkind.print)
             fields_kinds))
      non_escaping_blocks;
  let continuations_with_live_block =
    continuations_with_live_block ~non_escaping_blocks ~dom ~source_info
      ~control_flow_graph
  in
  let toplevel_used =
    Continuation.Map.find source_info.dummy_toplevel_cont
      continuations_with_live_block
  in
  if not (Variable.Set.is_empty toplevel_used)
  then
    Misc.fatal_errorf
      "Toplevel continuation cannot have needed extra argument for block: %a@."
      Variable.Set.print toplevel_used;
  let extra_params_and_args, rewrites =
    Fold_prims.compute_rewrites ~dom ~source_info ~continuations_with_live_block
      ~non_escaping_blocks
  in
  { extra_params_and_args;
    non_escaping_makeblocks = non_escaping_blocks;
    continuations_with_live_block;
    rewrites;
    required_regions
  }

let pp_node { non_escaping_makeblocks = _; continuations_with_live_block; _ }
    ppf cont =
  match Continuation.Map.find cont continuations_with_live_block with
  | exception Not_found -> ()
  | live_blocks -> Format.fprintf ppf " %a" Variable.Set.print live_blocks

let add_to_extra_params_and_args result =
  let epa = Continuation.Map.empty in
  let extra_block_args :
      EPA.Extra_arg.t Apply_cont_rewrite_id.Map.t list Continuation.Map.t =
    Continuation.Map.fold
      (fun _caller_cont (_extra_params, extra_args) all_extra_args ->
        Continuation.Map.fold
          (fun callee_cont
               (caller_extra_args :
                 Simple.t Numeric_types.Int.Map.t Apply_cont_rewrite_id.Map.t)
               (all_extra_args :
                 EPA.Extra_arg.t Apply_cont_rewrite_id.Map.t list
                 Continuation.Map.t) :
               EPA.Extra_arg.t Apply_cont_rewrite_id.Map.t list
               Continuation.Map.t ->
            Continuation.Map.update callee_cont
              (fun previous_extra_args ->
                let previous_extra_args :
                    EPA.Extra_arg.t Apply_cont_rewrite_id.Map.t list =
                  match previous_extra_args with
                  | None -> (
                    match
                      Continuation.Map.find callee_cont
                        result.extra_params_and_args
                    with
                    | exception Not_found -> []
                    | extra_params, _ ->
                      List.map
                        (fun _ -> Apply_cont_rewrite_id.Map.empty)
                        extra_params)
                  | Some extra_args -> extra_args
                in
                let extra_args =
                  Apply_cont_rewrite_id.Map.fold
                    (fun rewrite_id (args : Simple.t Numeric_types.Int.Map.t)
                         (previous_extra_args :
                           EPA.Extra_arg.t Apply_cont_rewrite_id.Map.t list) ->
                      let args = int_map_to_list args in
                      List.map2
                        (fun arg previous_extra_args ->
                          Apply_cont_rewrite_id.Map.add rewrite_id
                            (EPA.Extra_arg.Already_in_scope arg)
                            previous_extra_args)
                        args previous_extra_args)
                    caller_extra_args previous_extra_args
                in
                Some extra_args)
              all_extra_args)
          extra_args all_extra_args)
      result.extra_params_and_args Continuation.Map.empty
  in
  let epa =
    Continuation.Map.fold
      (fun cont extra_args epa ->
        let extra_params =
          match Continuation.Map.find cont result.extra_params_and_args with
          | exception Not_found -> []
          | extra_params, _ -> extra_params
        in
        Continuation.Map.update cont
          (fun epa_for_cont ->
            let epa_for_cont =
              match epa_for_cont with None -> EPA.empty | Some epa -> epa
            in
            let epa_for_cont =
              List.fold_left2
                (fun epa_for_cont extra_param extra_args ->
                  EPA.add epa_for_cont ~extra_param ~extra_args)
                epa_for_cont extra_params extra_args
            in
            Some epa_for_cont)
          epa)
      extra_block_args epa
  in
  epa

let make_result result =
  let additionnal_epa = add_to_extra_params_and_args result in
  let let_rewrites = result.rewrites in
  ( T.Mutable_unboxing_result.{ additionnal_epa; let_rewrites },
    result.required_regions,
    Variable.Map.keys result.non_escaping_makeblocks )
