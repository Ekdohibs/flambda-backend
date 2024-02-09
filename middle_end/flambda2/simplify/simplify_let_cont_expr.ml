(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*   Guillaume Bury, Pierre Chambart and Nathanaëlle Courant, OCamlPro    *)
(*           Mark Shinwell and Leo White, Jane Street Europe              *)
(*                                                                        *)
(*   Copyright 2013--2020 OCamlPro SAS                                    *)
(*   Copyright 2014--2020 Jane Street Group LLC                           *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

open! Simplify_import

(* High-level view of the workflow for simplification of let cont:
 *
 * +--------------------+           +--------------------+
 * | simplify_let_cont  |           | rebuild_let_cont   |
 * +--------------------+           +--------------------+
 *           |                                 ^
 *           | DOWN(body)             UP(body) |
 *           v                                 |
 * +--------------------+           +--------------------+
 * | after_downwards_   |           | prepare_to_        |
 * | traversal_of_body  |           | rebuild_body       |
 * +--------------------+           +--------------------+
 *           |                                 ^
 *           | DOWN(handlers)     UP(handlers) |
 *           v                                 |
 * +--------------------+           +--------------------+
 * | after_downwards_   |           | prepare_to_        |
 * | traversal_of_body_ |           | rebuild_handlers   |
 * | and_handlers       |           |                    |
 * +--------------------+           +--------------------+
 *           |                                 ^
 *           |       DOWN_TO_UP(global)        |
 *           +---------------------------------+
 *)

(* For each stage, the information received by this stage is of type
   stage_data. *)

type one_recursive_handler = Lifted_cont.one_recursive_handler =
  { params : Bound_parameters.t;
    handler : Expr.t;
    is_cold : bool
  }

type non_recursive_handler = Lifted_cont.non_recursive_handler =
  { cont : Continuation.t;
    params : Bound_parameters.t;
    lifted_params : Lifted_cont_params.t;
    handler : Expr.t;
    is_exn_handler : bool;
    is_cold : bool;
  }

type original_handlers = Lifted_cont.original_handlers =
  | Recursive of
      { invariant_params : Bound_parameters.t;
        lifted_params : Lifted_cont_params.t;
        continuation_handlers : one_recursive_handler Continuation.Map.t
      }
  | Non_recursive of non_recursive_handler

type simplify_let_cont_data =
  { body : Expr.t;
    handlers : original_handlers
  }

type after_downwards_traversal_of_body_data =
  { denv_for_join : DE.t;
    prior_lifted_constants : LCS.t;
    handlers : original_handlers
  }

type expr_to_rebuild = (Rebuilt_expr.t * Upwards_acc.t) Simplify_common.rebuild

type handler_after_downwards_traversal =
  { params : Bound_parameters.t;
    rebuild_handler : expr_to_rebuild;
    is_exn_handler : bool;
    is_cold : bool;
    (* continuations_used is the set of which continuations from this block of
       mutually recursive continuations is used inside the handler *)
    continuations_used : Continuation.Set.t;
    unbox_decisions : Unbox_continuation_params.Decisions.t;
    extra_params_and_args_for_cse : EPA.t
  }

type after_downwards_traversal_of_body_and_handlers_lifted =
  { rebuild_body : expr_to_rebuild; }

type after_downwards_traversal_of_body_and_handlers_rebuild =
  { rebuild_body : expr_to_rebuild;
    cont_uses_env : CUE.t;
    (* total cont uses env in body + handlers (+ previous exprs) *)
    at_unit_toplevel : bool;
    lifted_params : Lifted_cont_params.t;
    invariant_params : Bound_parameters.t;
    invariant_extra_params_and_args : EPA.t;
    handlers : handler_after_downwards_traversal Continuation.Map.t
  }

type after_downwards_traversal_of_body_and_handlers_data =
  | Lifted_out of after_downwards_traversal_of_body_and_handlers_lifted
  | Rebuild of after_downwards_traversal_of_body_and_handlers_rebuild

type handler_to_rebuild =
  { params : Bound_parameters.t;
    rebuild_handler : expr_to_rebuild;
    is_exn_handler : bool;
    is_cold : bool;
    extra_params_and_args : EPA.t;
    (* Note: EPA.extra_params invariant_extra_params_and_args should always be
       equal to invariant_extra_params in stage4 *)
    invariant_extra_params_and_args : EPA.t;
    rewrite_ids : Apply_cont_rewrite_id.Set.t
  }

type handlers_to_rebuild_group =
  | Recursive of
      { rebuild_continuation_handlers : handler_to_rebuild Continuation.Map.t }
  | Non_recursive of
      { cont : Continuation.t;
        handler : handler_to_rebuild;
        is_single_inlinable_use : bool
      }

type prepare_to_rebuild_handlers_lifted =
  { rebuild_body : expr_to_rebuild; }

type prepare_to_rebuild_handlers_rebuild =
  { rebuild_body : expr_to_rebuild;
    at_unit_toplevel : bool;
    handlers_from_the_outside_to_the_inside : handlers_to_rebuild_group list;
    original_invariant_params : Bound_parameters.t;
    invariant_extra_params : Bound_parameters.t
  }

type prepare_to_rebuild_handlers_data =
  | Lifted_out of prepare_to_rebuild_handlers_lifted
  | Rebuild of prepare_to_rebuild_handlers_rebuild

type rebuilt_handler =
  { handler : Rebuilt_expr.Continuation_handler.t;
    handler_expr : Rebuilt_expr.t;
    name_occurrences_of_handler : Name_occurrences.t;
    cost_metrics_of_handler : Cost_metrics.t
  }

type rebuilt_handlers_group =
  | Recursive of
      { continuation_handlers : rebuilt_handler Continuation.Map.t;
        invariant_params : Bound_parameters.t
      }
  | Non_recursive of
      { cont : Continuation.t;
        handler : rebuilt_handler
      }

type prepare_to_rebuild_body_data =
  { rebuild_body : expr_to_rebuild;
    handlers_from_the_inside_to_the_outside : rebuilt_handlers_group list;
    name_occurrences_of_subsequent_exprs : Name_occurrences.t;
    cost_metrics_of_subsequent_exprs : Cost_metrics.t;
    uenv_of_subsequent_exprs : UE.t
  }

type rebuild_let_cont_data =
  { handlers_from_the_inside_to_the_outside : rebuilt_handlers_group list;
    name_occurrences_of_subsequent_exprs : Name_occurrences.t;
    cost_metrics_of_subsequent_exprs : Cost_metrics.t;
    uenv_of_subsequent_exprs : UE.t
  }

(* Helpers *)

let split_non_recursive_let_cont handler =
  let cont, body =
    Non_recursive_let_cont_handler.pattern_match handler
      ~f:(fun cont ~body -> cont, body)
  in
  let cont_handler = Non_recursive_let_cont_handler.handler handler in
  let is_exn_handler = CH.is_exn_handler cont_handler in
  let is_cold = CH.is_cold cont_handler in
  let params, handler =
    CH.pattern_match cont_handler ~f:(fun params ~handler ->
        params, handler)
  in
  body, { cont; params; handler; lifted_params = Lifted_cont_params.empty;
          is_exn_handler; is_cold; }

let split_recursive_let_cont handlers =
  let invariant_params, body, rec_handlers =
    Recursive_let_cont_handlers.pattern_match handlers
      ~f:(fun ~invariant_params ~body rec_handlers ->
          invariant_params, body, rec_handlers)
  in
  assert (not (Continuation_handlers.contains_exn_handler rec_handlers));
  let handlers = Continuation_handlers.to_map rec_handlers in
  let continuation_handlers =
    Continuation.Map.map
      (fun handler ->
         let is_cold = CH.is_cold handler in
         CH.pattern_match handler ~f:(fun params ~handler ->
             { params; handler; is_cold; }))
      handlers
  in
  body, invariant_params, continuation_handlers

let split_let_cont let_cont : _ * original_handlers =
  match (let_cont : Let_cont.t) with
  | Non_recursive { handler; _ } ->
    let body, non_rec_handler = split_non_recursive_let_cont handler in
    body, Non_recursive non_rec_handler
  | Recursive handlers ->
    let lifted_params = Lifted_cont_params.empty in
    let body, invariant_params, continuation_handlers = split_recursive_let_cont handlers in
    body, Recursive { invariant_params; lifted_params; continuation_handlers; }

(* Helpers for extra parameters of lifted continuations *)

let extend_lifted_continuation_uses lifted_cont_params callee_lifted_params uses =
  if debug () then
    Format.eprintf "@\n~~~ EXTEND ~~~@\ncallee: %a@\nlifted_cont_params: %a@\nuses: %a@\n@."
      Lifted_cont_params.print callee_lifted_params
      (Continuation.Map.print Lifted_cont_params.print) lifted_cont_params
      (Format.pp_print_list ~pp_sep:Format.pp_print_space One_continuation_use.print) uses
  ;
  if Lifted_cont_params.is_empty callee_lifted_params then
    Bound_parameters.empty, uses
  else begin
    let lifted_params = Lifted_cont_params.bound_parameters callee_lifted_params in
    let callee_lifted_param_kinds = List.map (fun bound_param ->
        K.With_subkind.kind (Bound_parameter.kind bound_param))
        (Bound_parameters.to_list lifted_params)
    in
    let uses =
      List.map (fun one_use ->
          let id = One_continuation_use.id one_use in
          let env_at_use = One_continuation_use.env_at_use one_use in
          let stack = DE.continuation_stack env_at_use in
          if debug () then
            Format.eprintf "''' USE %a '''@\nstack: %a@\n@."
              Apply_cont_rewrite_id.print id
              (Format.pp_print_list ~pp_sep:Format.pp_print_space Continuation.print) stack;
          let caller_stack_lifted_params = List.filter_map (fun c -> Continuation.Map.find_opt c lifted_cont_params) stack in
          let arg_types = One_continuation_use.arg_types one_use in
          let lifted_args = Lifted_cont_params.args ~callee_lifted_params ~caller_stack_lifted_params in
          let lifted_arg_types = List.map2 T.alias_type_of callee_lifted_param_kinds lifted_args in
          let use_kind = One_continuation_use.use_kind one_use in
          One_continuation_use.create use_kind ~env_at_use id
            ~arg_types:(List.append arg_types lifted_arg_types)
        ) uses
    in
    lifted_params, uses
  end

let add_extra_params_and_args_for_lifting cont_lifting_params callee_lifted_params uses epa =
  Lifted_cont_params.fold callee_lifted_params ~init:epa
    ~f:(fun lifted_param_id extra_param epa ->
        let extra_args =
          List.fold_left (fun map one_use ->
              let id = One_continuation_use.id one_use in
              let env_at_use = One_continuation_use.env_at_use one_use in
              let stack = DE.continuation_stack env_at_use in
              let caller_stack_lifted_params = List.filter_map (fun c -> Continuation.Map.find_opt c cont_lifting_params) stack in
              let arg = Lifted_cont_params.find_arg lifted_param_id caller_stack_lifted_params in
              Apply_cont_rewrite_id.Map.add id (EPA.Extra_arg.Already_in_scope arg)  map
            ) Apply_cont_rewrite_id.Map.empty uses
        in
        EPA.add epa ~extra_param ~extra_args
      )

let decide_param_usage_non_recursive ~free_names ~required_names
    ~removed_aliased ~exn_bucket param : Apply_cont_rewrite.used =
  (* The free_names computation is the reference here, because it records
     precisely what is actually used in the term being rebuilt. The required
     variables computed by the data_flow analysis can only be an over
     approximation of it here (given that some simplification/dead code
     elimination may have removed some uses on the way up). To make sure the
     data_flow analysis is correct (or rather than the pre-condition for its
     correctness are verified, i.e. that on the way down, the use constraints
     accumulated are an over-approximation of the actual use constraints), we
     check here that all actually-used variables were also marked as used by the
     data_flow analysis. *)
  let param_var = BP.var param in
  let is_used =
    match NO.count_variable_normal_mode free_names param_var with
    | Zero -> Option.equal Variable.equal exn_bucket (Some param_var)
    | One | More_than_one -> true
  in
  if is_used && not (Name.Set.mem (Name.var param_var) required_names)
  then
    Misc.fatal_errorf
      "The data_flow analysis marked the param %a@ as not required, but the \
       free_names indicate it is actually used:@ \n\
       free_names = %a" BP.print param NO.print free_names;
  if is_used && Variable.Set.mem param_var removed_aliased
  then
    Misc.fatal_errorf
      "The alias analysis marked the param %a@ as removed, but the free_names \
       indicate it is actually used:@ \n\
       free_names = %a" BP.print param NO.print free_names;
  if is_used then Used else Unused

let decide_param_usage_recursive ~required_names ~invariant_set ~removed_aliased
    param : Apply_cont_rewrite.used =
  if Name.Set.mem (BP.name param) required_names
  && not (Variable.Set.mem (BP.var param) removed_aliased)
  then
    if Bound_parameter.Set.mem param invariant_set
    then Used_as_invariant
    else Used
  else Unused

let extra_params_for_continuation_param_aliases cont uacc rewrite_ids =
  let Flow_types.Alias_result.{ continuation_parameters; aliases_kind; _ } =
    UA.continuation_param_aliases uacc
  in
  let required_extra_args =
    Continuation.Map.find cont continuation_parameters
  in
  Variable.Set.fold
    (fun var epa ->
       let extra_args =
         Apply_cont_rewrite_id.Map.of_set
           (fun _id -> EPA.Extra_arg.Already_in_scope (Simple.var var))
           rewrite_ids
       in
       let var_kind =
         Flambda_kind.With_subkind.create
           (Variable.Map.find var aliases_kind)
           Anything
       in
       EPA.add ~extra_param:(Bound_parameter.create var var_kind) ~extra_args epa)
    required_extra_args.extra_args_for_aliases EPA.empty

let add_extra_params_for_mutable_unboxing cont uacc extra_params_and_args =
  let Flow_types.Mutable_unboxing_result.{ additionnal_epa; _ } =
    UA.mutable_unboxing_result uacc
  in
  match Continuation.Map.find cont additionnal_epa with
  | exception Not_found -> extra_params_and_args
  | additionnal_epa ->
    EPA.concat ~inner:extra_params_and_args ~outer:additionnal_epa

type behaviour =
  | Invalid
  | Alias_for of Continuation.t
  | Unknown

let bound_parameters_equal b1 b2 =
  List.equal Bound_parameter.equal
    (Bound_parameters.to_list b1)
    (Bound_parameters.to_list b2)

let get_removed_aliased_params uacc cont =
  let param_aliases = UA.continuation_param_aliases uacc in
  let cont_params =
    Continuation.Map.find cont param_aliases.continuation_parameters
  in
  cont_params.removed_aliased_params_and_extra_params

let make_rewrite_for_recursive_continuation uacc ~cont
    ~original_invariant_params ~invariant_extra_params_and_args
    ~original_variant_params ~variant_extra_params_and_args ~rewrite_ids =
  (* Note: extra_params_and_args come from CSE & immutable unboxing *)
  let alias_epa =
    extra_params_for_continuation_param_aliases cont uacc rewrite_ids
  in
  let invariant_extra_params_and_args =
    EPA.concat ~inner:invariant_extra_params_and_args ~outer:alias_epa
  in
  let variant_extra_params_and_args =
    add_extra_params_for_mutable_unboxing cont uacc
      variant_extra_params_and_args
  in
  let required_names = UA.required_names uacc in
  let removed_aliased = get_removed_aliased_params uacc cont in
  let invariant_set =
    BP.Set.union
      (Bound_parameters.to_set original_invariant_params)
      (Bound_parameters.to_set
         (EPA.extra_params invariant_extra_params_and_args))
  in
  let decide_param_usage =
    decide_param_usage_recursive ~required_names ~invariant_set ~removed_aliased
  in
  let rewrite =
    Apply_cont_rewrite.create
      ~original_params:
        (Bound_parameters.append original_invariant_params
           original_variant_params)
      ~extra_params_and_args:
        (EPA.concat ~outer:invariant_extra_params_and_args
           ~inner:variant_extra_params_and_args)
      ~decide_param_usage
  in
  let invariant_params, variant_params =
    Apply_cont_rewrite.get_used_params rewrite
  in
  let params = Bound_parameters.append invariant_params variant_params in
  let uacc =
    UA.map_uenv uacc ~f:(fun uenv ->
        let uenv = UE.add_apply_cont_rewrite uenv cont rewrite in
        UE.add_non_inlinable_continuation uenv cont ~params ~handler:Unknown)
  in
  uacc

let rebuild_let_cont (data : rebuild_let_cont_data) ~after_rebuild body uacc =
  (* Here both the body and the handlers have been rebuilt. We only need to
     restore the cost metrics and name occurrences accumulators, rebuild all the
     let cont expressions, and call after_rebuild with the result and the new
     upwards accumulator. *)
  let name_occurrences_body = UA.name_occurrences uacc in
  let cost_metrics_of_body = UA.cost_metrics uacc in
  let rec rebuild_groups body name_occurrences_body cost_metrics_of_body uacc
      groups =
    match groups with
    | [] ->
      (* Everything has now been rebuilt.

         We correctly set the name occurrences and the cost metrics, and we
         restore the upwards environment of [uacc] so that out-of-scope
         continuation bindings do not end up in the accumulator. *)
      let uacc =
        UA.with_name_occurrences
          ~name_occurrences:
            (Name_occurrences.union name_occurrences_body
               data.name_occurrences_of_subsequent_exprs)
          uacc
      in
      let uacc =
        UA.with_cost_metrics
          (Cost_metrics.( + ) cost_metrics_of_body
             data.cost_metrics_of_subsequent_exprs)
          uacc
      in
      let uacc = UA.with_uenv uacc data.uenv_of_subsequent_exprs in
      after_rebuild body uacc
    | Non_recursive { cont; handler } :: groups ->
      let num_free_occurrences_of_cont_in_body =
        (* Note that this does not count uses in trap actions. If there are uses
           in trap actions, but [remove_let_cont_leaving_body] is [true] below,
           then this must be a case where the exception handler can be demoted
           to a normal handler. This will cause the trap actions to be
           erased. *)
        NO.count_continuation name_occurrences_body cont
      in
      let is_applied_with_traps =
        NO.continuation_is_applied_with_traps name_occurrences_body cont
      in
      let remove_let_cont_leaving_body =
        match num_free_occurrences_of_cont_in_body with
        | Zero -> true
        | One | More_than_one -> false
      in
      (* We are passing back over a binder, so remove the bound continuation
         from the free name information. Then compute the free names of the
         whole [Let_cont]. *)
      let name_occurrences_body =
        NO.remove_continuation name_occurrences_body ~continuation:cont
      in
      (* Having rebuilt both the body and handler, the [Let_cont] expression
         itself is rebuilt -- unless either the continuation had zero uses, in
         which case we're left with the body; or if the body is just an
         [Apply_cont] (with no trap action) of [cont], in which case we're left
         with the handler. *)
      let expr, name_occurrences, cost_metrics =
        if remove_let_cont_leaving_body
        then body, name_occurrences_body, cost_metrics_of_body
        else
          let remove_let_cont_leaving_handler =
            match RE.to_apply_cont body with
            | Some apply_cont -> (
                if not
                    (Continuation.equal cont
                       (Apply_cont.continuation apply_cont))
                then false
                else
                  match Apply_cont.args apply_cont with
                  | [] -> Option.is_none (Apply_cont.trap_action apply_cont)
                  | _ :: _ -> false)
            | None -> false
          in
          if remove_let_cont_leaving_handler
          then
            ( handler.handler_expr,
              handler.name_occurrences_of_handler,
              handler.cost_metrics_of_handler )
          else
            let name_occurrences =
              NO.union name_occurrences_body handler.name_occurrences_of_handler
            in
            let cost_metrics =
              Cost_metrics.( + ) cost_metrics_of_body
                (Cost_metrics.increase_due_to_let_cont_non_recursive
                   ~cost_metrics_of_handler:handler.cost_metrics_of_handler)
            in
            let expr =
              RE.create_non_recursive_let_cont'
                (UA.are_rebuilding_terms uacc)
                cont handler.handler ~body ~num_free_occurrences_of_cont_in_body
                ~is_applied_with_traps
            in
            expr, name_occurrences, cost_metrics
      in
      rebuild_groups expr name_occurrences cost_metrics uacc groups
    | Recursive { continuation_handlers; invariant_params } :: groups ->
      let rec_handlers =
        Continuation.Map.map
          (fun handler -> handler.handler)
          continuation_handlers
      in
      let expr =
        RE.create_recursive_let_cont
          (UA.are_rebuilding_terms uacc)
          ~invariant_params rec_handlers ~body
      in
      let name_occurrences =
        Continuation.Map.fold
          (fun _ handler name_occurrences ->
             NO.union name_occurrences
               (NO.increase_counts handler.name_occurrences_of_handler))
          continuation_handlers name_occurrences_body
      in
      let name_occurrences =
        Continuation.Map.fold
          (fun cont _ name_occurrences ->
             NO.remove_continuation name_occurrences ~continuation:cont)
          continuation_handlers name_occurrences
      in
      let cost_metrics_of_handlers =
        Continuation.Map.fold
          (fun _ handler cost_metrics ->
             Cost_metrics.( + ) cost_metrics handler.cost_metrics_of_handler)
          continuation_handlers Cost_metrics.zero
      in
      let cost_metrics =
        Cost_metrics.increase_due_to_let_cont_recursive
          ~cost_metrics_of_handlers
      in
      let cost_metrics = Cost_metrics.( + ) cost_metrics cost_metrics_of_body in
      rebuild_groups expr name_occurrences cost_metrics uacc groups
  in
  rebuild_groups body name_occurrences_body cost_metrics_of_body uacc
    data.handlers_from_the_inside_to_the_outside

let prepare_to_rebuild_body (data : prepare_to_rebuild_body_data) uacc
    ~after_rebuild =
  (* At this point all handlers have been rebuild and added to the upwards
     environment. All that we still need to do is to rebuild the body, and then
     rebuild the chain of let cont expressions once this is done. We reinit the
     name occurrences and cost metrics one last time to get precise information
     for those two in the body, we rebuild the body, and we pass on to the final
     stage for the reconstruction of the let cont expressions. *)
  let uacc = UA.clear_cost_metrics (UA.clear_name_occurrences uacc) in
  let rebuild_body = data.rebuild_body in
  let data : rebuild_let_cont_data =
    { name_occurrences_of_subsequent_exprs =
        data.name_occurrences_of_subsequent_exprs;
      cost_metrics_of_subsequent_exprs = data.cost_metrics_of_subsequent_exprs;
      uenv_of_subsequent_exprs = data.uenv_of_subsequent_exprs;
      handlers_from_the_inside_to_the_outside =
        data.handlers_from_the_inside_to_the_outside
    }
  in
  rebuild_body uacc ~after_rebuild:(rebuild_let_cont data ~after_rebuild)

let add_lets_around_handler cont at_unit_toplevel uacc handler =
  let Flow_types.Alias_result.{ continuation_parameters; _ } =
    UA.continuation_param_aliases uacc
  in
  let continuation_parameters =
    Continuation.Map.find cont continuation_parameters
  in
  let handler, uacc =
    Variable.Map.fold
      (fun var bound_to (handler, uacc) ->
         let bound_pattern =
           Bound_pattern.singleton (Bound_var.create var Name_mode.normal)
         in
         let named = Named.create_simple (Simple.var bound_to) in
         let handler, uacc =
           Expr_builder.create_let_binding uacc bound_pattern named
             ~free_names_of_defining_expr:
               (Name_occurrences.singleton_variable bound_to Name_mode.normal)
             ~cost_metrics_of_defining_expr:Cost_metrics.zero ~body:handler
         in
         handler, uacc)
      continuation_parameters.lets_to_introduce (handler, uacc)
  in
  let handler, uacc =
    (* We might need to place lifted constants now, as they could depend on
       continuation parameters. As such we must also compute the unused
       parameters after placing any constants! *)
    if not at_unit_toplevel
    then handler, uacc
    else
      let uacc, lifted_constants_from_body =
        UA.get_and_clear_lifted_constants uacc
      in
      EB.place_lifted_constants uacc
        ~lifted_constants_from_defining_expr:LCS.empty
        ~lifted_constants_from_body
        ~put_bindings_around_body:(fun uacc ~body -> body, uacc)
        ~body:handler
  in
  let free_names = UA.name_occurrences uacc in
  let cost_metrics = UA.cost_metrics uacc in
  handler, uacc, free_names, cost_metrics

let add_phantom_params_bindings uacc handler new_phantom_params =
  let new_phantom_param_bindings_outermost_first =
    List.map
      (fun param ->
         let var = BP.var param in
         let kind = K.With_subkind.kind (BP.kind param) in
         let var = Bound_var.create var Name_mode.phantom in
         let let_bound = Bound_pattern.singleton var in
         let prim = Flambda_primitive.(Nullary (Optimised_out kind)) in
         let named = Named.create_prim prim Debuginfo.none in
         let simplified_defining_expr = Simplified_named.create named in
         { Expr_builder.let_bound;
           simplified_defining_expr;
           original_defining_expr = Some named
         })
      (Bound_parameters.to_list new_phantom_params)
  in
  EB.make_new_let_bindings uacc ~body:handler
    ~bindings_outermost_first:new_phantom_param_bindings_outermost_first

let remove_params params free_names =
  ListLabels.fold_left (Bound_parameters.to_list params) ~init:free_names
    ~f:(fun free_names param -> NO.remove_var free_names ~var:(BP.var param))

let rebuild_single_non_recursive_handler ~at_unit_toplevel
    ~is_single_inlinable_use ~original_invariant_params cont
    (handler_to_rebuild : handler_to_rebuild) uacc k =
  (* Clear existing name occurrences & cost metrics *)
  let uacc = UA.clear_name_occurrences (UA.clear_cost_metrics uacc) in
  let { is_exn_handler;
        is_cold;
        rewrite_ids;
        params;
        rebuild_handler;
        extra_params_and_args;
        invariant_extra_params_and_args
      } =
    handler_to_rebuild
  in
  (* In case the continuation was previously recursive, we make sure not to
     forget the invariant original and extra params. *)
  let params = Bound_parameters.append original_invariant_params params in
  let extra_params_and_args =
    EPA.concat ~inner:invariant_extra_params_and_args
      ~outer:extra_params_and_args
  in
  rebuild_handler uacc ~after_rebuild:(fun handler uacc ->
      let handler, uacc, free_names, cost_metrics =
        add_lets_around_handler cont at_unit_toplevel uacc handler
      in
      let extra_params_and_args =
        EPA.concat ~inner:extra_params_and_args
          ~outer:
            (extra_params_for_continuation_param_aliases cont uacc rewrite_ids)
        |> add_extra_params_for_mutable_unboxing cont uacc
      in
      let exn_bucket =
        if is_exn_handler
        then
          Some (Bound_parameter.var (List.hd (Bound_parameters.to_list params)))
        else None
      in
      let removed_aliased = get_removed_aliased_params uacc cont in
      let decide_param_usage =
        decide_param_usage_non_recursive ~free_names
          ~required_names:(UA.required_names uacc) ~removed_aliased ~exn_bucket
      in
      let rewrite =
        Apply_cont_rewrite.create ~original_params:params ~extra_params_and_args
          ~decide_param_usage
      in
      let invariant_params, params =
        Apply_cont_rewrite.get_used_params rewrite
      in
      if not (Bound_parameters.is_empty invariant_params)
      then
        Misc.fatal_errorf "Non-recursive continuation has invariant params: %a"
          Apply_cont_rewrite.print rewrite;
      let new_phantom_params =
        Bound_parameters.filter
          (fun param -> NO.mem_var free_names (BP.var param))
          (Apply_cont_rewrite.get_unused_params rewrite)
      in
      let handler, uacc =
        add_phantom_params_bindings uacc handler new_phantom_params
      in
      let free_names = remove_params new_phantom_params free_names in
      let cont_handler =
        RE.Continuation_handler.create
          (UA.are_rebuilding_terms uacc)
          params ~handler ~free_names_of_handler:free_names ~is_exn_handler
          ~is_cold
      in
      let uacc =
        UA.map_uenv uacc ~f:(fun uenv ->
            UE.add_apply_cont_rewrite uenv cont rewrite)
      in
      let uenv = UA.uenv uacc in
      (* TODO move to its own function *)
      let uenv =
        (* CR : factor this out in a separate function ? *)
        if (* We must make the final decision now as to whether to inline this
              continuation or not; we can't wait until
              [Simplify_apply_cont.rebuild_apply_cont] because we need to decide
              sooner than that whether to keep the [Let_cont] (in order to keep
              free name sets correct). *)
          is_single_inlinable_use
        then (
          (* Note that [Continuation_uses] won't set [is_single_inlinable_use]
             if [cont] is an exception handler. *)
          assert (not is_exn_handler);
          (* We pass the parameters and the handler expression, rather than the
             [CH.t], to avoid re-opening the name abstraction. *)
          UE.add_linearly_used_inlinable_continuation uenv cont ~params ~handler
            ~free_names_of_handler:free_names
            ~cost_metrics_of_handler:cost_metrics)
        else
          let behaviour =
            (* CR-someday mshinwell: This could be replaced by a more
               sophisticated analysis, but for the moment we just use a simple
               syntactic check. *)
            if is_exn_handler
            then Unknown
            else
              match RE.to_apply_cont handler with
              | Some apply_cont -> (
                  match Apply_cont.trap_action apply_cont with
                  | Some _ -> Unknown
                  | None ->
                    let args = Apply_cont.args apply_cont in
                    let params = Bound_parameters.simples params in
                    if Misc.Stdlib.List.compare Simple.compare args params = 0
                    then Alias_for (Apply_cont.continuation apply_cont)
                    else Unknown)
              | None ->
                if RE.can_be_removed_as_invalid handler
                    (UA.are_rebuilding_terms uacc)
                then Invalid
                else Unknown
          in
          match behaviour with
          | Invalid ->
            let arity = Bound_parameters.arity params in
            UE.add_invalid_continuation uenv cont arity
          | Alias_for alias_for ->
            let arity = Bound_parameters.arity params in
            UE.add_continuation_alias uenv cont arity ~alias_for
          | Unknown ->
            UE.add_non_inlinable_continuation uenv cont ~params
              ~handler:(if is_cold then Unknown else Known handler)
      in
      let uacc = UA.with_uenv uacc uenv in
      (* The parameters are removed from the free name information as they are
         no longer in scope. *)
      let free_names = remove_params params free_names in
      let rebuilt_handler : rebuilt_handler =
        { handler = cont_handler;
          handler_expr = handler;
          name_occurrences_of_handler = free_names;
          cost_metrics_of_handler = cost_metrics
        }
      in
      k rebuilt_handler uacc)

let rebuild_single_recursive_handler cont
    (handler_to_rebuild : handler_to_rebuild) uacc k =
  (* Clear existing name occurrences & cost metrics *)
  let uacc = UA.clear_name_occurrences (UA.clear_cost_metrics uacc) in
  handler_to_rebuild.rebuild_handler uacc ~after_rebuild:(fun handler uacc ->
      let handler, uacc, free_names, cost_metrics =
        add_lets_around_handler cont false uacc handler
      in
      let rewrite =
        match UE.find_apply_cont_rewrite (UA.uenv uacc) cont with
        | None ->
          Misc.fatal_errorf
            "An [Apply_cont_rewrite] for the recursive continuation %a should \
             have already been added"
            Continuation.print cont
        | Some rewrite -> rewrite
      in
      let new_phantom_params =
        Bound_parameters.filter
          (fun param -> NO.mem_var free_names (BP.var param))
          (Apply_cont_rewrite.get_unused_params rewrite)
      in
      let handler, uacc =
        add_phantom_params_bindings uacc handler new_phantom_params
      in
      let free_names = remove_params new_phantom_params free_names in
      let invariant_params, variant_params =
        Apply_cont_rewrite.get_used_params rewrite
      in
      let cont_handler =
        RE.Continuation_handler.create
          (UA.are_rebuilding_terms uacc)
          variant_params ~handler ~free_names_of_handler:free_names
          ~is_exn_handler:false ~is_cold:handler_to_rebuild.is_cold
      in
      let free_names =
        remove_params invariant_params (remove_params variant_params free_names)
      in
      let rebuilt_handler : rebuilt_handler =
        { handler = cont_handler;
          handler_expr = handler;
          name_occurrences_of_handler = free_names;
          cost_metrics_of_handler = cost_metrics
        }
      in
      k invariant_params rebuilt_handler uacc)

let rec rebuild_continuation_handlers_loop ~rebuild_body
    ~name_occurrences_of_subsequent_exprs ~cost_metrics_of_subsequent_exprs
    ~uenv_of_subsequent_exprs ~at_unit_toplevel ~original_invariant_params
    ~invariant_extra_params uacc ~after_rebuild
    (groups_to_rebuild : handlers_to_rebuild_group list) rebuilt_groups =
  match groups_to_rebuild with
  | [] ->
    let data : prepare_to_rebuild_body_data =
      { rebuild_body;
        name_occurrences_of_subsequent_exprs;
        cost_metrics_of_subsequent_exprs;
        uenv_of_subsequent_exprs;
        handlers_from_the_inside_to_the_outside = rebuilt_groups
      }
    in
    prepare_to_rebuild_body data uacc ~after_rebuild
  | Non_recursive { cont; handler; is_single_inlinable_use }
    :: groups_to_rebuild ->
    rebuild_single_non_recursive_handler ~at_unit_toplevel
      ~original_invariant_params ~is_single_inlinable_use cont handler uacc
      (fun rebuilt_handler uacc ->
         rebuild_continuation_handlers_loop ~rebuild_body
           ~name_occurrences_of_subsequent_exprs
           ~cost_metrics_of_subsequent_exprs ~uenv_of_subsequent_exprs
           ~at_unit_toplevel ~original_invariant_params ~invariant_extra_params
           uacc ~after_rebuild groups_to_rebuild
           (Non_recursive { cont; handler = rebuilt_handler } :: rebuilt_groups))
  | Recursive { rebuild_continuation_handlers } :: groups_to_rebuild ->
    (* Common setup for recursive handlers: add rewrites; for now: always add
       params (ignore alias analysis) *)
    let uacc =
      Continuation.Map.fold
        (fun cont handler uacc ->
           make_rewrite_for_recursive_continuation uacc ~cont
             ~original_invariant_params ~original_variant_params:handler.params
             ~invariant_extra_params_and_args:
               handler.invariant_extra_params_and_args
             ~variant_extra_params_and_args:handler.extra_params_and_args
             ~rewrite_ids:handler.rewrite_ids)
        rebuild_continuation_handlers uacc
    in
    (* Rebuild all the handlers *)
    let rec loop uacc invariant_params remaining_handlers rebuilt_handlers k =
      match Continuation.Map.min_binding_opt remaining_handlers with
      | None -> k (Option.get invariant_params) rebuilt_handlers uacc
      | Some (cont, handler) ->
        let remaining_handlers =
          Continuation.Map.remove cont remaining_handlers
        in
        rebuild_single_recursive_handler cont handler uacc
          (fun cont_invariant_params rebuilt_handler uacc ->
             let invariant_params =
               match invariant_params with
               | None -> Some cont_invariant_params
               | Some invariant_params ->
                 if not
                     (bound_parameters_equal invariant_params
                        cont_invariant_params)
                 then Misc.fatal_errorf "TODO good error message"
                 else Some invariant_params
             in
             loop uacc invariant_params remaining_handlers
               (Continuation.Map.add cont rebuilt_handler rebuilt_handlers)
               k)
    in
    loop uacc None rebuild_continuation_handlers Continuation.Map.empty
      (fun invariant_params rebuilt_handlers uacc ->
         (* Add all rewrites and continue rebuilding *)
         rebuild_continuation_handlers_loop ~rebuild_body
           ~name_occurrences_of_subsequent_exprs
           ~cost_metrics_of_subsequent_exprs ~uenv_of_subsequent_exprs
           ~at_unit_toplevel ~original_invariant_params ~invariant_extra_params
           uacc ~after_rebuild groups_to_rebuild
           (Recursive
              { continuation_handlers = rebuilt_handlers; invariant_params }
            :: rebuilt_groups))

let prepare_to_rebuild_handlers (data : prepare_to_rebuild_handlers_data) uacc
    ~after_rebuild =
  match data with
  | Lifted_out { rebuild_body; } ->
    rebuild_body uacc ~after_rebuild
  | Rebuild data ->
    (* Here we just returned from the global [down_to_up], which is asking us to
       rebuild the let cont. The flow analyses have been done, and we start to
       rebuild the expressions. As with the downward pass, we loop over each
       defined handler to rebuild it. As the handlers have been sorted by
       strongly-connected components, we must process the handlers group-by-group,
       and rebuild the groups from the outside to the inside. For each of these
       groups, we need to perform different steps depending on whether it is a
       mutually-recursive group or a single non-recursive handler.

       For mutually-recursive groups, we need to add the rewrites to the
       environment, and then rebuild the handlers, as the handlers might use
       themselves, so the rewrites need to be ready at that point. We can only use
       the dataflow information to compute which parameters are used and which are
       not, since we won't be able to remove parameters after the handlers have
       been rebuilt, since we can't rewrite them inside themselves.

       For a single non-recursive handler, we first rebuild the handler. This
       allows us to get precise information on the parameters which are used,
       which are then used to add the rewrites to the environment.

       In both cases, we add the handlers to the environment after rebuilding
       them, so that the environment is ready when rebuilding the remaining
       handlers. We also reset the name occurrences and the cost metrics before
       rebuilding each handler, so that we know the name occurrences and cost
       metrics corresponding to each handler when rebuilding later. *)
    let name_occurrences_of_subsequent_exprs = UA.name_occurrences uacc in
    let cost_metrics_of_subsequent_exprs = UA.cost_metrics uacc in
    let uenv_of_subsequent_exprs = UA.uenv uacc in
    rebuild_continuation_handlers_loop ~rebuild_body:data.rebuild_body
      ~at_unit_toplevel:data.at_unit_toplevel
      ~original_invariant_params:data.original_invariant_params
      ~invariant_extra_params:data.invariant_extra_params
      ~name_occurrences_of_subsequent_exprs ~cost_metrics_of_subsequent_exprs
      ~uenv_of_subsequent_exprs uacc ~after_rebuild
      data.handlers_from_the_outside_to_the_inside []

let get_uses (data : after_downwards_traversal_of_body_and_handlers_rebuild) cont =
  match CUE.get_continuation_uses data.cont_uses_env cont with
  | None ->
    Misc.fatal_errorf
      "Uses of %a not found in \
       [after_downwards_traversal_of_body_and_handlers_data]@."
      Continuation.print cont
  | Some cont -> cont

let create_handler_to_rebuild cont_lifting_params
    (data : after_downwards_traversal_of_body_and_handlers_rebuild) cont
    (handler : handler_after_downwards_traversal) =
  (* See comment at the top of [after_downwards_travsersal_of_body_and_handlers]. *)
  let uses = get_uses data cont in
  let use_ids = Continuation_uses.get_use_ids uses in
  let invariant_extra_args =
    Apply_cont_rewrite_id.Map.of_set
      (fun rewrite_id ->
         match
           Apply_cont_rewrite_id.Map.find rewrite_id
             (EPA.extra_args data.invariant_extra_params_and_args)
         with
         | extra_args -> extra_args
         | exception Not_found ->
           List.map
             (fun param ->
                EPA.Extra_arg.Already_in_scope
                  (Simple.var (Bound_parameter.var param)))
             (Bound_parameters.to_list
                (EPA.extra_params data.invariant_extra_params_and_args)))
      use_ids
  in
  let invariant_epa =
    EPA.replace_extra_args data.invariant_extra_params_and_args
      invariant_extra_args
  in
  let invariant_epa =
    add_extra_params_and_args_for_lifting cont_lifting_params
      data.lifted_params (Continuation_uses.get_uses uses) invariant_epa
  in
  let extra_params_and_args =
    let arg_types_by_use_id = Continuation_uses.get_arg_types_by_use_id uses in
    let _, arg_types_by_use_id =
      Misc.Stdlib.List.split_at
        (List.length (Bound_parameters.to_list data.invariant_params))
        arg_types_by_use_id
    in
    Unbox_continuation_params.compute_extra_params_and_args
      handler.unbox_decisions ~arg_types_by_use_id
      handler.extra_params_and_args_for_cse
  in
  { params = handler.params;
    rebuild_handler = handler.rebuild_handler;
    is_exn_handler = handler.is_exn_handler;
    is_cold = handler.is_cold;
    extra_params_and_args;
    invariant_extra_params_and_args = invariant_epa;
    rewrite_ids = Continuation_uses.get_use_ids uses
  }

module SCC = Strongly_connected_components.Make (Continuation)

let sort_handlers data handlers =
  let handlers_graph =
    Continuation.Map.map
      (fun handler -> handler.continuations_used)
      data.handlers
  in
  let sorted_handlers_from_the_inside_to_the_outside =
    SCC.connected_components_sorted_from_roots_to_leaf handlers_graph
  in
  Array.fold_left
    (fun inner group ->
       let group : handlers_to_rebuild_group =
         match (group : SCC.component) with
         | Has_loop conts ->
           let rebuild_continuation_handlers =
             List.fold_left
               (fun group cont ->
                  Continuation.Map.add cont
                    (Continuation.Map.find cont handlers)
                    group)
               Continuation.Map.empty conts
           in
           Recursive { rebuild_continuation_handlers }
         | No_loop cont ->
           let handler = Continuation.Map.find cont handlers in
           let is_single_inlinable_use =
             match Continuation_uses.get_uses (get_uses data cont) with
             | [] | _ :: _ :: _ -> false
             | [use] -> (
                 match One_continuation_use.use_kind use with
                 | Inlinable -> not handler.is_cold
                 | Non_inlinable _ -> false)
           in
           Non_recursive { cont; handler; is_single_inlinable_use }
       in
       group :: inner)
    [] sorted_handlers_from_the_inside_to_the_outside

let rec after_downwards_traversal_of_body_and_handlers
    ~simplify_expr ~denv_for_join
    (data : after_downwards_traversal_of_body_and_handlers_data) ~down_to_up
    dacc =
  (* At this point we have done a downwards traversal on the body and all the
     handlers. *)
  match data with
  | Lifted_out { rebuild_body; } ->
    (* In this case, we want to lift the continuations being defined *)
    let data : prepare_to_rebuild_handlers_data = Lifted_out { rebuild_body; } in
    down_to_up dacc ~rebuild:(prepare_to_rebuild_handlers data)
  | Rebuild data ->
    let dacc, down_to_up =
      (* Format.eprintf "LIFTOUT %a@." DA.print dacc; *)
      let dacc, lifted_conts = DA.get_and_clear_lifted_continuations dacc in
      dacc, down_to_up_for_lifted_continuations
        ~simplify_expr ~denv_for_join lifted_conts ~down_to_up
    in
    (* Here, we finally have the "normal" path, when we want to actually rebuild
       the let-cont, so we need to call the global [down_to_up] function. First
       however, we have to take care of several things:

       - We need to compute the extra params and args related to unboxed
         parameters. This could not be done earlier, as we might not have seen every
         use of the continuations (particularly in the recursive case)
         (done in the [create_handler_to_rebuild] function).

       - We need to compute the extra params and args related to lifting of
         continuations. For the same reasons as unboxing of params, we need
         to wait until here to ensure that we have seen all uses of the continuation
         (done in the [create_handler_to_rebuild] function).

       - Now that we were able to compute the extra params and args, we add this
         information to the flow analysis, so that it can correctly compute all the
         information we need when going up.

       - We perform a strongly-connected components analysis of the continuations
         to be able to turn mutually-recursive continuations into several
         independant blocks of recursive or non-recursive continuations. In case one
         of those is non-recursive, we can check whether the continuation is
         inlinable if it is used a single time. *)
    let cont_lifting_params = DA.cont_lifting_params dacc in
    let handlers =
      Continuation.Map.mapi (create_handler_to_rebuild cont_lifting_params data) data.handlers
    in
    let dacc =
      DA.map_flow_acc dacc ~f:(fun flow_acc ->
          Continuation.Map.fold
            (fun cont handler flow_acc ->
               Flow.Acc.add_extra_params_and_args cont
                 (EPA.concat ~inner:handler.invariant_extra_params_and_args
                    ~outer:handler.extra_params_and_args)
                 flow_acc)
            handlers flow_acc)
    in
    let handlers_from_the_outside_to_the_inside = sort_handlers data handlers in
    let data : prepare_to_rebuild_handlers_data =
      Rebuild { rebuild_body = data.rebuild_body;
                handlers_from_the_outside_to_the_inside;
                at_unit_toplevel = data.at_unit_toplevel;
                original_invariant_params = data.invariant_params;
                invariant_extra_params =
                  EPA.extra_params data.invariant_extra_params_and_args
              }
    in
    down_to_up dacc ~rebuild:(prepare_to_rebuild_handlers data)

and prepare_dacc_for_handlers dacc ~env_at_fork ~params ~is_recursive
    ~consts_lifted_during_body continuation_sort is_exn_handler_cont uses
    ~lifted_params ~arg_types_by_use_id =
  (* In the recursive case, [params] are actually the invariant params, as we
     prepare a common dacc for all handlers. *)
  let join_result =
    (* We add to the continuation uses (and therefore to the handler env) the
       params added by the eventual lifting of continuations. However, we do not
       want those lifted params to be unboxed: it should not be beneficial (since
       the lifted params should already include unbxoed versions). *)
    let lifted_params, uses =
      extend_lifted_continuation_uses (DA.cont_lifting_params dacc) lifted_params uses
    in
    let params = Bound_parameters.append params lifted_params in
    if false && debug () then
      Format.eprintf "@\n___ JOIN ___@\n%a@\nenv_at_fork: %a@\n@[<v>%a@]@\n@."
        Bound_parameters.print params
        DE.print env_at_fork
        (Format.pp_print_list ~pp_sep:Format.pp_print_space One_continuation_use.print) uses;
    Join_points.compute_handler_env uses ~params ~is_recursive ~env_at_fork
      ~consts_lifted_during_body
  in
  let code_age_relation = TE.code_age_relation (DA.typing_env dacc) in
  let handler_env =
    DE.with_code_age_relation code_age_relation join_result.handler_env
    |> DE.bump_current_level_scope
  in
  let do_not_unbox () =
    Unbox_continuation_params.make_do_not_unbox_decisions params
  in
  let handler_env, unbox_decisions, is_exn_handler, dacc =
    match (continuation_sort : Continuation.Sort.t) with
    | Normal_or_exn when join_result.is_single_inlinable_use -> (
        match is_exn_handler_cont with
        | Some cont ->
          (* This should be prevented by [Simplify_apply_cont_expr]. *)
          Misc.fatal_errorf
            "Exception handlers should never be marked as [Inlinable]:@ %a"
            Continuation.print cont
        (* Don't try to unbox parameters of inlinable continuations, since the
           typing env still contains enough information to avoid re-reading the
           fields. *)
        | None ->
          let dacc = DA.with_are_lifting_conts dacc Are_lifting_conts.no_lifting in
          handler_env, do_not_unbox (), false, dacc)
    | Normal_or_exn | Define_root_symbol ->
      let dacc, is_exn_handler =
        match is_exn_handler_cont with
        | None -> dacc, false
        | Some cont ->
          if join_result.escapes
          then dacc, true
          else DA.demote_exn_handler dacc cont, false
      in
      if is_exn_handler
      then handler_env, do_not_unbox (), true, dacc
      else
        (* Unbox the parameters of the continuation if possible. Any such
           unboxing will induce a rewrite (or wrapper) on the application sites
           of the continuation. *)
        let param_types = TE.find_params (DE.typing_env handler_env) params in
        let handler_env, decisions =
          Unbox_continuation_params.make_decisions handler_env
            ~continuation_arg_types:(Non_recursive arg_types_by_use_id) params
            param_types
        in
        handler_env, decisions, false, dacc
    | Return | Toplevel_return -> (
        match is_exn_handler_cont with
        | Some cont ->
          (* This should be prevented by [Simplify_apply_cont_expr]. *)
          Misc.fatal_errorf
            "Exception handlers should never be marked as [Return] or \
             [Toplevel_return]:@ %a"
            Continuation.print cont
        | None -> handler_env, do_not_unbox (), false, dacc)
  in
  ( DA.with_denv dacc handler_env,
    unbox_decisions,
    is_exn_handler,
    join_result.extra_params_and_args )

and simplify_handler ~simplify_expr ~is_recursive ~is_exn_handler
    ~lifted_params ~invariant_params ~params ~all_extra_params cont dacc handler k =
  let all_params =
      (Bound_parameters.append invariant_params params) in
  let dacc = DA.map_denv dacc ~f:(DE.enter_continuation cont lifted_params) in
  let dacc = DA.map_denv dacc ~f:(fun denv ->
      List.fold_left (fun denv bp ->
          DE.add_variable_defined_in_current_continuation denv bp
        ) denv (Bound_parameters.to_list
                  (Bound_parameters.append all_params all_extra_params))
    )
  in
  if debug () then
    Format.eprintf "DEFINED: %a@\n@."
    Lifted_cont_params.print
     (DE.variables_defined_in_current_continuation (DA.denv dacc));
  let dacc = DA.with_continuation_uses_env dacc ~cont_uses_env:CUE.empty in
  let dacc =
    DA.map_flow_acc
      ~f:
        (Flow.Acc.enter_continuation cont ~recursive:is_recursive
           ~is_exn_handler all_params)
      dacc
  in
  simplify_expr dacc handler ~down_to_up:(fun dacc ~rebuild:rebuild_handler ->
      let dacc = DA.map_flow_acc ~f:(Flow.Acc.exit_continuation cont) dacc in
      let cont_uses_env_in_handler = DA.continuation_uses_env dacc in
      let cont_uses_env_in_handler =
        if is_recursive
        then CUE.mark_non_inlinable cont_uses_env_in_handler
        else cont_uses_env_in_handler
      in
      k dacc rebuild_handler cont_uses_env_in_handler)

and simplify_single_recursive_handler ~simplify_expr cont_uses_env_so_far
    ~lifted_params ~invariant_params ~invariant_epa
    consts_lifted_during_body all_handlers_set denv_to_reset
    dacc cont ({ params; handler; is_cold } : one_recursive_handler) k =
  (* Here we perform the downwards traversal on a single handler.

     We also make unboxing decisions at this step, which are necessary to
     correctly simplify the handler using the unboxed parameters, but we delay
     the unboxing extra_params_and_args to later, when we will have seen all
     uses (needed for the recursive continuation handlers). *)
  let handler_env =
    DE.add_parameters_with_unknown_types ~at_unit_toplevel:false denv_to_reset
      params
  in
  let handler_env = LCS.add_to_denv handler_env consts_lifted_during_body in
  let code_age_relation = TE.code_age_relation (DA.typing_env dacc) in
  let handler_env = DE.with_code_age_relation code_age_relation handler_env in
  let handler_env, unbox_decisions, dacc =
    (* Unbox the parameters of the continuation if possible. Any such unboxing
       will induce a rewrite (or wrapper) on the application sites of the
       continuation; that rewrite will be comptued later, when we compute all
       the extra args and params. *)
    let param_types = TE.find_params (DE.typing_env handler_env) params in
    let handler_env, decisions =
      Unbox_continuation_params.make_decisions handler_env
        ~continuation_arg_types:Recursive params param_types
    in
    handler_env, decisions, dacc
  in
  let dacc = DA.with_denv dacc handler_env in
  let all_extra_params =
    Bound_parameters.append
      (EPA.extra_params invariant_epa)
      (Unbox_continuation_params.compute_extra_params_in_unspecified_order unbox_decisions)
  in
  simplify_handler ~simplify_expr ~is_recursive:true ~is_exn_handler:false
    ~lifted_params ~params ~invariant_params ~all_extra_params cont dacc handler
    (fun dacc rebuild_handler cont_uses_env_in_handler ->
       let cont_uses_env_so_far =
         CUE.union cont_uses_env_so_far cont_uses_env_in_handler
       in
       let continuations_used =
         Continuation.Set.inter all_handlers_set
           (CUE.all_continuations_used cont_uses_env_in_handler)
       in
       k dacc
         { params;
           rebuild_handler;
           is_exn_handler = false;
           is_cold;
           continuations_used;
           unbox_decisions;
           extra_params_and_args_for_cse = EPA.empty
         }
         cont_uses_env_so_far)

and simplify_recursive_handlers ~rebuild_body ~lifted_params ~invariant_params ~invariant_epa
    ~continuation_handlers ~simplify_expr ~consts_lifted_during_body
    ~all_conts_set ~common_denv =
  let rec loop cont_uses_env_so_far reachable_handlers_to_simplify
      simplified_handlers_set simplified_handlers dacc k =
    (* This is the core loop to simplify all handlers defined by a recursive let
       cont. We loop over all handlers, each time taking the first handler that
       we have not yet processed and that has at least one use, until we have
       seen every handler that is reachable. *)
    (* CR-someday ncourant: this makes the order in which continuations are
       processed dependant on things like the name of the compilation unit
       (because it affects the order on [Continuation.t], and thus the element
       returned by [min_elt_opt]). However, recursive continuations are
       specified using a [Continuation.Map.t], whose order already depends on
       the name of the compilation unit. *)
    match Continuation.Set.min_elt_opt reachable_handlers_to_simplify with
    | None ->
      (* all remaining_handlers are unreachable *)
      let cont_uses_env =
        Continuation.Set.fold
          (fun cont cont_uses_env -> CUE.remove cont_uses_env cont)
          simplified_handlers_set cont_uses_env_so_far
      in
      let dacc = DA.with_continuation_uses_env dacc ~cont_uses_env in
      let data : after_downwards_traversal_of_body_and_handlers_rebuild = {
        rebuild_body;
        cont_uses_env = cont_uses_env_so_far;
        lifted_params;
        invariant_params;
        invariant_extra_params_and_args = invariant_epa;
        handlers = simplified_handlers;
        at_unit_toplevel = false;
      }
      in
      k dacc data
    | Some cont ->
      let reachable_handlers_to_simplify =
        Continuation.Set.remove cont reachable_handlers_to_simplify
      in
      let handler = Continuation.Map.find cont continuation_handlers in
      simplify_single_recursive_handler ~simplify_expr ~lifted_params ~invariant_params ~invariant_epa
        cont_uses_env_so_far consts_lifted_during_body all_conts_set common_denv
        dacc cont handler (fun dacc rebuild cont_uses_env_so_far ->
            let simplified_handlers_set =
              Continuation.Set.add cont simplified_handlers_set
            in
            let reachable_handlers_to_simplify =
              Continuation.Set.union reachable_handlers_to_simplify
                (Continuation.Set.diff rebuild.continuations_used
                   simplified_handlers_set)
            in
            let simplified_handlers =
              Continuation.Map.add cont rebuild simplified_handlers
            in
            loop cont_uses_env_so_far reachable_handlers_to_simplify
              simplified_handlers_set simplified_handlers dacc k)
  in
  loop

and simplify_handlers ~simplify_expr ~rebuild_body
    (data: after_downwards_traversal_of_body_data) dacc k =
  (* In this case we have decided not to lift the continuation being let-bound
     outside of its context. So the remaining thing to do is setup the dacc,
     and later decide whether to lift any continuation the is defined inside
     the continuation being currentl let-bound. *)
  let body_continuation_uses_env = DA.continuation_uses_env dacc in
  let denv = data.denv_for_join in
  let consts_lifted_during_body = DA.get_lifted_constants dacc in
  let dacc =
    DA.add_to_lifted_constant_accumulator dacc data.prior_lifted_constants
  in
  let previous_are_lifting_conts = DA.are_lifting_conts dacc in
  match data.handlers with
  | Non_recursive { cont; params; lifted_params; handler; is_exn_handler; is_cold } -> (
      if debug () then Format.eprintf "SIMPLIFY_HANDLERS %a@\n@." Continuation.print cont;
      let dacc = DA.with_are_lifting_conts dacc Are_lifting_conts.no_lifting in
      match
        Continuation_uses_env.get_continuation_uses body_continuation_uses_env
          cont
      with
      | None ->
        (* Continuation unused, no need to traverse its handler *)
        let data : after_downwards_traversal_of_body_and_handlers_rebuild =
          { rebuild_body;
            cont_uses_env = body_continuation_uses_env;
            lifted_params = Lifted_cont_params.empty;
            invariant_params = Bound_parameters.empty;
            invariant_extra_params_and_args = EPA.empty;
            handlers = Continuation.Map.empty;
            at_unit_toplevel = false;
          }
        in
        k dacc data
      | Some uses ->
        let dacc = DA.with_are_lifting_conts dacc
            (Are_lifting_conts.think_about_lifting_out_of cont uses)
        in
        (* let _uses = Continuation_uses.get_uses uses in
        Format.eprintf "USES = %a@." (Format.pp_print_list (fun ff use ->
            One_continuation_use.print ff use;
            DE.print ff (One_continuation_use.env_at_use use)
          )
           ) _uses; *)
        let at_unit_toplevel =
          (* We try to show that [handler] postdominates [body] (which is done by
             showing that [body] can only return through [cont]) and that if
             [body] raises any exceptions then it only does so to toplevel. If
             this can be shown and we are currently at the toplevel of a
             compilation unit, the handler for the environment can remain marked
             as toplevel (and suitable for "let symbol" bindings); otherwise, it
             cannot. *)
          DE.at_unit_toplevel denv && (not is_exn_handler)
          && Continuation.Set.subset
            (CUE.all_continuations_used body_continuation_uses_env)
            (Continuation.Set.of_list
               [cont; DE.unit_toplevel_exn_continuation denv])
        in
        let denv = DE.set_at_unit_toplevel_state denv at_unit_toplevel in
        let dacc, unbox_decisions, is_exn_handler, extra_params_and_args_for_cse =
          prepare_dacc_for_handlers dacc ~env_at_fork:denv ~params ~lifted_params
            ~consts_lifted_during_body ~is_recursive:false
            (Continuation.sort cont)
            (if is_exn_handler then Some cont else None)
            (Continuation_uses.get_uses uses)
            ~arg_types_by_use_id:(Continuation_uses.get_arg_types_by_use_id uses)
        in
        Format.eprintf "Decisions for %a:@\n%a@\ndacc:@\n%a@\n@."
          Continuation.print cont Unbox_continuation_params.Decisions.print unbox_decisions
          TE.print (DE.typing_env (DA.denv dacc));
        let all_extra_params =
          Bound_parameters.append
            (EPA.extra_params extra_params_and_args_for_cse)
            (Unbox_continuation_params.compute_extra_params_in_unspecified_order unbox_decisions)
        in
        simplify_handler ~simplify_expr ~is_recursive:false ~is_exn_handler
          ~lifted_params ~params ~all_extra_params cont dacc handler ~invariant_params:Bound_parameters.empty
          (fun dacc rebuild_handler cont_uses_env_in_handler ->
             let cont_uses_env_so_far =
               CUE.union body_continuation_uses_env cont_uses_env_in_handler
             in
             let continuations_used = Continuation.Set.empty in
             let rebuild =
               { params;
                 rebuild_handler;
                 is_exn_handler;
                 is_cold;
                 continuations_used;
                 unbox_decisions;
                 extra_params_and_args_for_cse
               }
             in
             let dacc =
               (* Update the dacc with the new cont_uses_env, which removes the
                  uses of [cont] since it leaves scope. *)
               let cont_uses_env = CUE.remove cont_uses_env_so_far cont in
               DA.with_continuation_uses_env dacc ~cont_uses_env
             in
             let dacc = DA.with_are_lifting_conts dacc previous_are_lifting_conts in
             let data : after_downwards_traversal_of_body_and_handlers_rebuild =
               { rebuild_body;
                 cont_uses_env = cont_uses_env_so_far;
                 lifted_params;
                 invariant_params = Bound_parameters.empty;
                 invariant_extra_params_and_args = EPA.empty;
                 handlers = Continuation.Map.singleton cont rebuild;
                 at_unit_toplevel;
               }
             in
             k dacc data
          ))
  | Recursive { continuation_handlers; invariant_params; lifted_params; } ->
    let dacc = DA.with_are_lifting_conts dacc Are_lifting_conts.no_lifting in
    let denv = DE.set_at_unit_toplevel_state denv false in
    let all_conts_set = Continuation.Map.keys continuation_handlers in
    let used_handlers_in_body =
      Continuation.Set.inter all_conts_set
        (CUE.all_continuations_used body_continuation_uses_env)
    in
    let all_uses =
      List.filter_map
        (CUE.get_continuation_uses body_continuation_uses_env)
        (Continuation.Set.elements all_conts_set)
    in
    let arity = Bound_parameters.arity invariant_params in
    let arg_types_by_use_id =
      Continuation_uses.get_arg_types_by_use_id_for_invariant_params arity
        all_uses
    in
    let dacc, unbox_decisions, is_exn_handler, extra_params_and_args_for_cse =
      prepare_dacc_for_handlers dacc ~env_at_fork:denv
        ~params:invariant_params ~lifted_params
        ~is_recursive:true ~consts_lifted_during_body (Normal_or_exn : Continuation.Sort.t) None
        (List.concat_map Continuation_uses.get_uses all_uses)
        ~arg_types_by_use_id
    in
    let invariant_epa =
      Unbox_continuation_params.compute_extra_params_and_args unbox_decisions
        ~arg_types_by_use_id extra_params_and_args_for_cse
    in
    let common_denv = DA.denv dacc in
    assert (not is_exn_handler);
    simplify_recursive_handlers ~rebuild_body ~lifted_params ~invariant_params ~invariant_epa
      ~continuation_handlers ~simplify_expr
      ~consts_lifted_during_body ~all_conts_set ~common_denv
      body_continuation_uses_env used_handlers_in_body Continuation.Set.empty
      Continuation.Map.empty dacc (fun dacc data -> let dacc = DA.with_are_lifting_conts dacc previous_are_lifting_conts in k dacc data)

and after_downwards_traversal_of_body ~simplify_expr ~down_to_up
    (data : after_downwards_traversal_of_body_data)
    dacc ~rebuild:rebuild_body =
  (* At this point, we have done the downwards traversal of the body, and
     we have two situations wrt to continuation lifting. *)
  let denv_for_join = data.denv_for_join in
  match DA.are_lifting_conts dacc with
  | Lifting_out_of { continuation = _; } ->
    (* In this case, we have decided to lift the continuation being bound out
         of its defining handler. Therefore we save the non-simplified version
         of the continuation in the dacc, so that we can simplify during the
         `down_to_up` later. *)
    let params_to_lift = DE.variables_defined_in_current_continuation (DA.denv dacc) in
    let handlers = Lifted_cont.add_params_to_lift data.handlers params_to_lift in
    let dacc = DA.add_lifted_continuation data.denv_for_join handlers dacc in
    (* Restore lifted constants in dacc *)
    let dacc = DA.add_to_lifted_constant_accumulator dacc data.prior_lifted_constants in
    if debug () then Format.eprintf "ADD %a@." Lifted_cont.print_original_handlers handlers;
    let data : after_downwards_traversal_of_body_and_handlers_data = Lifted_out { rebuild_body; } in
    after_downwards_traversal_of_body_and_handlers ~simplify_expr ~denv_for_join data ~down_to_up dacc
  | Not_lifting | Analyzing _ ->
    simplify_handlers ~simplify_expr data dacc ~rebuild_body (fun dacc data ->
        after_downwards_traversal_of_body_and_handlers ~simplify_expr ~denv_for_join (Rebuild data : after_downwards_traversal_of_body_and_handlers_data) ~down_to_up dacc
      )

and down_to_up_for_lifted_continuations ~simplify_expr ~denv_for_join lifted_conts ~down_to_up =
  match lifted_conts with
  | [] -> down_to_up
  | (denv, handlers) :: other_lifted_handlers ->
    (* At this point, we are lifting a continuation k' with handler [handlers], out of
       a continuation k, and:
       - [denv_for_join] is the denv just before the letk for k
       - [denv] is the denv just before the letk for k'

       And we need to decide which parts of denv to use to simplify the handlers of k'
       after there are lifted out from the handler of k. *)
    let actual_denv = DE.denv_for_lifted_continuation ~denv_for_join ~denv in
    if debug () then Format.eprintf "stacking downwards for %a@."
        Lifted_cont.print_original_handlers handlers;
    let data : after_downwards_traversal_of_body_data =
      { denv_for_join = actual_denv; prior_lifted_constants = LCS.empty; handlers; }
    in
    let down_to_up =
      after_downwards_traversal_of_body ~simplify_expr data ~down_to_up
    in
    down_to_up_for_lifted_continuations ~simplify_expr ~denv_for_join
      other_lifted_handlers ~down_to_up

let simplify_let_cont0 ~(simplify_expr : _ Simplify_common.expr_simplifier) dacc (data : simplify_let_cont_data)
    ~down_to_up =
  (* We begin to simplify a let cont by simplifying its body, so that we can see
     all external calls to the handler in the non-recursive case, and so that we
     can know the values of all invariant arguments in the recursive case. We
     reset the [continuation_uses_env] so we can have precise information on
     continuations called by the body, and we reset the lifted constants because
     we need to add them to the handler's denv. *)
  let dacc, prior_lifted_constants = DA.get_and_clear_lifted_constants dacc in
  let denv_before_body = DA.denv dacc in
  (* About scopes: supposing we are at scope 'n' before the let-cont, we will:

     - use scope 'n + 2' to inspect the body

     - use scope 'n + 1' to perform the join of continuation uses

     - bump the scope of the env after join to get an env at scope 'n + 2'

     - recursively for lifted continuations, whose join will take place at level
       'n + 1' (which is the level for denv_for_join) *)
  let denv_for_join = DE.increment_continuation_scope denv_before_body in
  let denv_for_body = DE.increment_continuation_scope denv_for_join in
  let dacc = DA.with_denv dacc denv_for_body in
  let body = data.body in
  let data : after_downwards_traversal_of_body_data =
    { denv_for_join; prior_lifted_constants; handlers = data.handlers; } in
  simplify_expr dacc body
    ~down_to_up:(after_downwards_traversal_of_body ~simplify_expr data ~down_to_up)

let simplify_let_cont ~simplify_expr dacc let_cont ~down_to_up =
  (* This is the entry point to simplify a let cont expression. The only thing
     it does is to match all handlers to break the name abstraction, and then
     call [simplify_let_cont_stage1]. *)
  let body, handlers = split_let_cont let_cont in
  simplify_let_cont0 ~simplify_expr dacc { body; handlers; }  ~down_to_up

let simplify_as_recursive_let_cont ~simplify_expr dacc (body, handlers)
    ~down_to_up =
  (* Loopify needs to simplify a recursive continuation, but knowing the unique
     id of the continuation and parameters being simplified, so this function
     allows to simplify a recursive let cont with the name abstraction already
     opened. *)
  let continuation_handlers =
    Continuation.Map.map
      (fun handler ->
         let is_cold = CH.is_cold handler in
         CH.pattern_match handler ~f:(fun params ~handler ->
             { params; handler; is_cold }))
      handlers
  in
  let data : simplify_let_cont_data =
    { body;
      handlers =
        Recursive
          { invariant_params = Bound_parameters.empty;
            lifted_params = Lifted_cont_params.empty;
            continuation_handlers }
    }
  in
  simplify_let_cont0 ~simplify_expr dacc data ~down_to_up
