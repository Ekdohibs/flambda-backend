(**************************************************************************)
(*                                                                        *)
(*                                 OCaml                                  *)
(*                                                                        *)
(*           Nathanaëlle Courant, Pierre Chambart, OCamlPro               *)
(*                                                                        *)
(*   Copyright 2024 OCamlPro SAS                                          *)
(*   Copyright 2024 Jane Street Group LLC                                 *)
(*                                                                        *)
(*   All rights reserved.  This file is distributed under the terms of    *)
(*   the GNU Lesser General Public License version 2.1, with the          *)
(*   special exception on linking described in the file LICENSE.          *)
(*                                                                        *)
(**************************************************************************)

module Graph = Global_flow_graph

type code_dep =
  { arity : [`Complex] Flambda_arity.t;
    params : Variable.t list;
    my_closure : Variable.t;
    return : Variable.t list; (* Dummy variable representing return value *)
    exn : Variable.t (* Dummy variable representing exn return value *)
  }

type apply_dep =
  { apply_in_func : Code_id.t option;
    apply_code_id : Code_id.t;
    apply_params : Simple.t list;
    apply_closure : Simple.t option;
    apply_return : Variable.t list option;
    apply_exn : Variable.t
  }

type t =
  { mutable code : code_dep Code_id.Map.t;
    mutable apply_deps : apply_dep list;
    mutable set_of_closures_dep : (Name.t * Code_id.t) list;
    deps : Graph.graph;
    mutable kinds : Flambda_kind.t Name.Map.t
  }

let code_deps t = t.code

let create () =
  { code = Code_id.Map.empty;
    apply_deps = [];
    set_of_closures_dep = [];
    deps = Graph.create ();
    kinds = Name.Map.empty
  }

let kinds t = t.kinds

let kind name k t = t.kinds <- Name.Map.add name k t.kinds

let bound_parameter_kind (bp : Bound_parameter.t) t =
  let kind = Flambda_kind.With_subkind.kind (Bound_parameter.kind bp) in
  let name = Name.var (Bound_parameter.var bp) in
  t.kinds <- Name.Map.add name kind t.kinds

let alias_kind name simple t =
  let kind =
    Simple.pattern_match simple
      ~name:(fun name ~coercion:_ ->
        (* Symbols are always values and might not be in t.kinds *)
        if Name.is_symbol name
        then Flambda_kind.value
        else
          match Name.Map.find_opt name t.kinds with
          | Some k -> k
          | None -> Misc.fatal_errorf "Unbound name %a" Name.print name)
      ~const:(fun const ->
        match Int_ids.Const.descr const with
        | Naked_immediate _ -> Flambda_kind.naked_immediate
        | Tagged_immediate _ -> Flambda_kind.value
        | Naked_float _ -> Flambda_kind.naked_float
        | Naked_float32 _ -> Flambda_kind.naked_float32
        | Naked_int32 _ -> Flambda_kind.naked_int32
        | Naked_int64 _ -> Flambda_kind.naked_int64
        | Naked_nativeint _ -> Flambda_kind.naked_nativeint
        | Naked_vec128 _ -> Flambda_kind.naked_vec128)
  in
  t.kinds <- Name.Map.add name kind t.kinds

let add_code code_id dep t = t.code <- Code_id.Map.add code_id dep t.code

let find_code t code_id = Code_id.Map.find code_id t.code

let record_dep ~denv:_ name dep t =
  let name = Code_id_or_name.name name in
  Graph.add_dep t.deps name dep

let record_dep' ~denv:_ code_id_or_name dep t =
  Graph.add_dep t.deps code_id_or_name dep

let record_deps ~denv:_ code_id_or_name deps t =
  Graph.add_deps t.deps code_id_or_name deps

let cont_dep ~denv:_ pat dep t =
  Simple.pattern_match dep
    ~name:(fun name ~coercion:_ -> Graph.add_cont_dep t.deps pat name)
    ~const:(fun _ -> ())

let func_param_dep ~denv:_ param arg t =
  Graph.add_func_param t.deps
    ~param:(Bound_parameter.var param)
    ~arg:(Name.var arg)

let root v t = Graph.add_use t.deps (Code_id_or_name.var v)

let used ~(denv : Traverse_denv.t) dep t =
  Simple.pattern_match dep
    ~name:(fun name ~coercion:_ ->
      match denv.current_code_id with
      | None -> Graph.add_use t.deps (Code_id_or_name.name name)
      | Some code_id ->
        Graph.add_dep t.deps
          (Code_id_or_name.code_id code_id)
          (Graph.Dep.Use { target = Code_id_or_name.name name }))
    ~const:(fun _ -> ())

let used_code_id code_id t =
  Graph.add_use t.deps (Code_id_or_name.code_id code_id)

let called ~(denv : Traverse_denv.t) code_id t =
  match denv.current_code_id with
  | None -> Graph.add_called t.deps code_id
  | Some code_id2 ->
    Graph.add_dep t.deps
      (Code_id_or_name.code_id code_id2)
      (Graph.Dep.Use { target = Code_id_or_name.code_id code_id })

let add_apply apply t = t.apply_deps <- apply :: t.apply_deps

let add_set_of_closures_dep name code_id t =
  t.set_of_closures_dep <- (name, code_id) :: t.set_of_closures_dep

let record_set_of_closure_deps t =
  List.iter
    (fun (name, code_id) ->
      match find_code t code_id with
      | exception Not_found ->
        assert (
          not
            (Compilation_unit.is_current (Code_id.get_compilation_unit code_id)));
        (* The code comes from another compilation unit; so we don't know what
           happens once it is applied. As such, it must escape the whole
           block. *)
        Graph.add_dep t.deps
          (Code_id_or_name.name name)
          (Block
             { relation = Code_of_closure; target = Code_id_or_name.name name })
      | code_dep ->
        Graph.add_dep t.deps
          (Code_id_or_name.var code_dep.my_closure)
          (Graph.Dep.Alias { target = name });
        let num_params = Flambda_arity.num_params code_dep.arity in
        let acc = ref (Code_id_or_name.name name) in
        for i = 1 to num_params - 1 do
          let tmp_name =
            Code_id_or_name.var
              (Variable.create (Printf.sprintf "partial_apply_%i" i))
          in
          Graph.add_dep t.deps !acc
            (Block { relation = Apply (Normal 0); target = tmp_name });
          (* The code_id needs to stay alive even if the function is only
             partially applied, as the arity is needed at runtime in that
             case. *)
          Graph.add_dep t.deps !acc
            (Block
               { relation = Code_of_closure;
                 target = Code_id_or_name.code_id code_id
               });
          acc := tmp_name
        done;
        List.iteri
          (fun i v ->
            Graph.add_dep t.deps !acc
              (Block
                 { relation = Apply (Normal i); target = Code_id_or_name.var v }))
          code_dep.return;
        Graph.add_dep t.deps !acc
          (Block
             { relation = Apply Exn; target = Code_id_or_name.var code_dep.exn });
        Graph.add_dep t.deps !acc
          (Block
             { relation = Code_of_closure;
               target = Code_id_or_name.code_id code_id
             }))
    t.set_of_closures_dep

let deps t =
  List.iter
    (fun { apply_in_func;
           apply_code_id;
           apply_params;
           apply_closure;
           apply_return;
           apply_exn
         } ->
      let code_dep = find_code t apply_code_id in
      let add_cond_dep param name =
        let param = Name.var param in
        match apply_in_func with
        | None ->
          Graph.add_dep t.deps
            (Code_id_or_name.name param)
            (Graph.Dep.Alias { target = name })
        | Some code_id ->
          Graph.add_dep t.deps
            (Code_id_or_name.name param)
            (Graph.Dep.Alias_if_def { target = name; if_defined = code_id });
          Graph.add_dep t.deps
            (Code_id_or_name.code_id code_id)
            (Graph.Dep.Propagate { target = name; source = param })
      in
      List.iter2
        (fun param arg ->
          Simple.pattern_match arg
            ~name:(fun name ~coercion:_ -> add_cond_dep param name)
            ~const:(fun _ -> ()))
        code_dep.params apply_params;
      (match apply_closure with
      | None -> ()
      | Some apply_closure ->
        Simple.pattern_match apply_closure
          ~name:(fun name ~coercion:_ -> add_cond_dep code_dep.my_closure name)
          ~const:(fun _ -> ()));
      (match apply_return with
      | None -> ()
      | Some apply_return ->
        List.iter2
          (fun arg param -> add_cond_dep param (Name.var arg))
          code_dep.return apply_return);
      add_cond_dep apply_exn (Name.var code_dep.exn))
    t.apply_deps;
  record_set_of_closure_deps t;
  t.deps
