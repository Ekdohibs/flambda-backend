open! Flambda

let num_occ_to_name_occ num_occ =
  Variable.Map.fold
    (fun v num acc ->
      match (num : Num_occurrences.t) with
      | Zero -> acc
      | One -> Name_occurrences.add_variable acc v Name_mode.normal
      | More_than_one ->
        Name_occurrences.add_variable
          (Name_occurrences.add_variable acc v Name_mode.normal)
          v Name_mode.normal)
    num_occ Name_occurrences.empty

let traverse_list l ~f ~down_to_up =
  let rec loop to_rebuild l =
    match l with
    | [] ->
      down_to_up (fun ~after_rebuild ->
          let rec do_rebuild rebuilt l =
            match l with
            | [] -> after_rebuild rebuilt
            | rebuild :: l ->
              rebuild ~after_rebuild:(fun x -> do_rebuild (x :: rebuilt) l)
          in
          do_rebuild [] to_rebuild)
    | x :: l -> f x ~down_to_up:(fun rebuild -> loop (rebuild :: to_rebuild) l)
  in
  loop [] l

let rec traverse_expr e ~down_to_up =
  match Expr.descr e with
  | Let let_expr -> traverse_let_expr let_expr ~down_to_up
  | Let_cont let_cont_expr -> traverse_let_cont let_cont_expr ~down_to_up
  | Apply apply_expr ->
    down_to_up (fun ~after_rebuild ->
        after_rebuild (Expr.create_apply apply_expr))
  | Apply_cont apply_cont_expr ->
    down_to_up (fun ~after_rebuild ->
        after_rebuild (Expr.create_apply_cont apply_cont_expr))
  | Switch switch_expr ->
    down_to_up (fun ~after_rebuild ->
        after_rebuild (Expr.create_switch switch_expr))
  | Invalid { message } ->
    down_to_up (fun ~after_rebuild ->
        after_rebuild (Expr.create_invalid (Invalid.Message message)))

and traverse_let_expr let_expr ~down_to_up =
  Let_expr.pattern_match' let_expr
    ~f:(fun p ~num_normal_occurrences_of_bound_vars ~body ->
      let free_names_of_body =
        Or_unknown.Known
          (num_occ_to_name_occ num_normal_occurrences_of_bound_vars)
      in
      traverse_named (Let_expr.defining_expr let_expr)
        ~down_to_up:(fun rebuild_named ->
          traverse_expr body ~down_to_up:(fun rebuild_body ->
              down_to_up (fun ~after_rebuild ->
                  rebuild_body ~after_rebuild:(fun body ->
                      rebuild_named ~after_rebuild:(fun named ->
                          after_rebuild
                            (Expr.create_let
                               (Let_expr.create p named ~body
                                  ~free_names_of_body))))))))

and traverse_named named ~down_to_up =
  match named with
  | Simple _ | Prim _ | Set_of_closures _ | Rec_info _ ->
    down_to_up (fun ~after_rebuild -> after_rebuild named)
  | Static_consts group ->
    traverse_list (Static_const_group.to_list group) ~f:traverse_static_const
      ~down_to_up:(fun rebuild_list ->
        down_to_up (fun ~after_rebuild ->
            rebuild_list ~after_rebuild:(fun l ->
                after_rebuild
                  (Named.create_static_consts (Static_const_group.create l)))))

and traverse_static_const sc ~down_to_up =
  match sc with
  | Deleted_code | Static_const _ ->
    down_to_up (fun ~after_rebuild -> after_rebuild sc)
  | Code code ->
    let params_and_body = Code.params_and_body code in
    Function_params_and_body.pattern_match params_and_body
      ~f:(fun
           ~return_continuation
           ~exn_continuation
           params
           ~body
           ~my_closure
           ~is_my_closure_used:_
           ~my_region
           ~my_depth
           ~free_names_of_body
         ->
        traverse_expr body ~down_to_up:(fun rebuild_body ->
            down_to_up (fun ~after_rebuild ->
                rebuild_body ~after_rebuild:(fun body ->
                    let params_and_body =
                      Function_params_and_body.create ~return_continuation
                        ~exn_continuation params ~body ~free_names_of_body
                        ~my_closure ~my_region ~my_depth
                    in
                    after_rebuild
                      (Static_const_or_code.create_code
                         (Code.create_with_metadata ~params_and_body
                            ~free_names_of_params_and_body:
                              (Code.free_names_of_params_and_body code)
                            ~code_metadata:(Code.code_metadata code)))))))

and traverse_let_cont let_cont_expr ~down_to_up =
  match let_cont_expr with
  | Non_recursive { handler; num_free_occurrences; is_applied_with_traps } ->
    Non_recursive_let_cont_handler.pattern_match handler ~f:(fun cont ~body ->
        let handler = Non_recursive_let_cont_handler.handler handler in
        traverse_expr body ~down_to_up:(fun rebuild_body ->
            traverse_continuation_handler handler
              ~down_to_up:(fun rebuild_handler ->
                down_to_up (fun ~after_rebuild ->
                    rebuild_handler ~after_rebuild:(fun handler ->
                        rebuild_body ~after_rebuild:(fun body ->
                            after_rebuild
                              (Let_cont_expr.create_non_recursive' ~cont handler
                                 ~body
                                 ~num_free_occurrences_of_cont_in_body:
                                   num_free_occurrences ~is_applied_with_traps)))))))
  | Recursive handlers ->
    Recursive_let_cont_handlers.pattern_match handlers
      ~f:(fun ~invariant_params ~body handlers ->
        let handlers =
          Continuation.Map.bindings (Continuation_handlers.to_map handlers)
        in
        traverse_expr body ~down_to_up:(fun rebuild_body ->
            traverse_list (List.map snd handlers)
              ~f:traverse_continuation_handler
              ~down_to_up:(fun rebuild_handlers ->
                down_to_up (fun ~after_rebuild ->
                    rebuild_handlers ~after_rebuild:(fun rebuilt_handlers ->
                        rebuild_body ~after_rebuild:(fun body ->
                            let handlers =
                              List.map2
                                (fun (cont, _) handler -> cont, handler)
                                handlers rebuilt_handlers
                            in
                            after_rebuild
                              (Let_cont_expr.create_recursive ~invariant_params
                                 (Continuation.Map.of_list handlers)
                                 ~body)))))))

and traverse_continuation_handler handler ~down_to_up =
  let is_exn_handler = Continuation_handler.is_exn_handler handler in
  let is_cold = Continuation_handler.is_cold handler in
  Continuation_handler.pattern_match' handler
    ~f:(fun params ~num_normal_occurrences_of_params ~handler ->
      traverse_expr handler ~down_to_up:(fun rebuild_handler ->
          down_to_up (fun ~after_rebuild ->
              rebuild_handler ~after_rebuild:(fun handler ->
                  let free_names_of_handler =
                    Or_unknown.Known
                      (num_occ_to_name_occ num_normal_occurrences_of_params)
                  in
                  after_rebuild
                    (Continuation_handler.create params ~handler
                       ~free_names_of_handler ~is_exn_handler ~is_cold)))))

let traverse_toplevel unit =
  let return_continuation = Flambda_unit.return_continuation unit in
  let exn_continuation = Flambda_unit.exn_continuation unit in
  let toplevel_my_region = Flambda_unit.toplevel_my_region unit in
  let body = Flambda_unit.body unit in
  let module_symbol = Flambda_unit.module_symbol unit in
  let used_value_slots = Flambda_unit.used_value_slots unit in
  let body =
    traverse_expr body ~down_to_up:(fun rebuild ->
        rebuild ~after_rebuild:(fun result -> result))
  in
  Flambda_unit.create ~return_continuation ~exn_continuation ~toplevel_my_region
    ~body ~module_symbol ~used_value_slots
