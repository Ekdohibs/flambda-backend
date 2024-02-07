module Deps = Cleanup_deps

type code_dep =
  { params : Variable.t list;
    my_closure : Variable.t;
    return : Variable.t list; (* Dummy variable representing return value *)
    exn : Variable.t (* Dummy variable representing exn return value *)
  }

let dep_graph_ppf =
  lazy
    (let filename = "dep.dot" in
     let ch = open_out filename in
     let ppf = Format.formatter_of_out_channel ch in
     (* Format.fprintf ppf "digraph g {@\n"; *)
     (* at_exit (fun () -> *)
     (*     Format.fprintf ppf "@\n}@."; *)
     (*     close_out ch); *)
     ppf)

let dot_count = ref ~-1

let print_graph ~print ~print_name ~lazy_ppf ~graph =
  match print_name with
  | None -> ()
  | Some print_name ->
    incr dot_count;
    let ppf = Lazy.force lazy_ppf in
    Format.fprintf ppf "digraph g {@\n";
    print ~ctx:!dot_count ~print_name ppf graph;
    Format.fprintf ppf "@\n}@."

module P = struct
  let node_id ~ctx ppf (variable : Code_id_or_name.t) =
    Format.fprintf ppf "node_%d_%d" ctx (variable :> int)

  let node ~ctx ~root ppf name =
    if root
    then
      Format.fprintf ppf "%a [shape=record label=\"%a\"];@\n" (node_id ~ctx)
        name Code_id_or_name.print name
    else
      Format.fprintf ppf "%a [label=\"%a\"];@\n" (node_id ~ctx) name
        Code_id_or_name.print name

  let dep_names (dep : Deps.Dep.t) =
    match dep with
    | Deps.Dep.Return_of_that_function n
    | Deps.Dep.Alias n
    | Deps.Dep.Use n
    | Deps.Dep.Field (_, n) ->
      [Code_id_or_name.name n]
    | Deps.Dep.Contains n | Deps.Dep.Block (_, n) -> [n]
    | Deps.Dep.Apply (n, c) ->
      [Code_id_or_name.name n; Code_id_or_name.code_id c]

  let all_names t =
    let names = Hashtbl.create 100 in
    Hashtbl.iter
      (fun name dep ->
        let dep_names =
          Deps.DepSet.fold (fun dep acc -> dep_names dep @ acc) dep []
        in
        List.iter (fun name -> Hashtbl.replace names name ()) (name :: dep_names))
      t.Deps.name_to_dep;
    names

  let nodes ~all_cdep ~ctx ppf t =
    Hashtbl.iter
      (fun name _ ->
        if Code_id_or_name.Set.mem name all_cdep
        then ()
        else
          let root =
            Code_id_or_name.pattern_match'
              ~code_id:(fun _ -> false)
              ~name:(fun name ->
                Hashtbl.mem t.Deps.used (Code_id_or_name.name name))
              name
          in
          node ~ctx ~root ppf name)
      (all_names t)

  let edge ~ctx ppf src (dst : Deps.Dep.t) =
    let color, deps =
      match dst with
      | Return_of_that_function name -> "purple", [Code_id_or_name.name name]
      | Alias name -> "black", [Code_id_or_name.name name]
      | Use name ->
        (* ignore name; *)
        (* "red", [] *)
        "red", [Code_id_or_name.name name]
      | Contains name -> "yellow", [name]
      | Field (_, name) -> "green", [Code_id_or_name.name name]
      | Block (_, name) -> "blue", [name]
      | Apply (name, _code) -> "pink", [Code_id_or_name.name name]
    in
    List.iter
      (fun dst ->
        Format.fprintf ppf "%a -> %a [color=\"%s\"];@\n" (node_id ~ctx) src
          (node_id ~ctx) dst color)
      deps

  let edges ~ctx ppf t =
    Hashtbl.iter
      (fun src dst_set ->
        Deps.DepSet.iter (fun dst -> edge ~ctx ppf src dst) dst_set)
      t.Deps.name_to_dep

  let code_deps ~ctx ~code_id ppf code_dep =
    node ~ctx ~root:false ppf (Code_id_or_name.code_id code_id);
    node ~ctx ~root:false ppf
      (Code_id_or_name.var code_dep.my_closure);
    List.iter
      (fun v -> node ~ctx ~root:false ppf (Code_id_or_name.var v))
      ((code_dep.exn :: code_dep.return) @ code_dep.params)

  let print_fundep ~all_cdep ~code_dep ~ctx (code_id : Code_id.t) ppf t =
    Format.fprintf ppf
      "subgraph cluster_%d_%d { label=\"%a\"@\n\
      \ subgraph cluster_%d_%d_intf { label=\"interface\"@\n\
      \ %a } @\n\
       %a@\n\
       @\n\
       }@\n"
      ctx
      (code_id :> int)
      Code_id.print code_id ctx
      (code_id :> int)
      (code_deps ~ctx ~code_id) code_dep (nodes ~all_cdep ~ctx) t

  let print_fundeps ~all_cdep ~code_dep ~ctx ppf fundeps =
    Hashtbl.iter
      (fun code_id fungraph ->
        print_fundep ~all_cdep
          ~code_dep:(Code_id.Map.find code_id code_dep)
          ~ctx code_id ppf fungraph)
      fundeps

  let all_edges ~ctx ppf (t : Deps.graph) =
    edges ~ctx ppf t.toplevel_graph;
    Hashtbl.iter (fun _ fungraph -> edges ~ctx ppf fungraph) t.function_graphs

  let print ~ctx ~print_name ppf
      ((code_dep, t) : code_dep Code_id.Map.t * Deps.graph) =
    let all_cdep =
      Code_id.Map.fold
        (fun code_id dep s ->
          List.fold_left
            (fun s x -> Code_id_or_name.Set.add x s)
            s
            (Code_id_or_name.code_id code_id
            :: List.map Code_id_or_name.var
                 ((dep.my_closure :: dep.exn :: dep.return) @ dep.params)))
        code_dep Code_id_or_name.Set.empty
    in
    Flambda_colours.without_colours ~f:(fun () ->
        Format.fprintf ppf
          "subgraph cluster_%d { label=\"%s\"@\n%a@\n%a@\n%a}@." ctx print_name
          (nodes ~all_cdep ~ctx) t.toplevel_graph
          (print_fundeps ~all_cdep ~code_dep ~ctx)
          t.function_graphs (all_edges ~ctx) t)
end

let print_dep dep =
  let print_name = Some "dep" in
  print_graph ~print_name ~lazy_ppf:dep_graph_ppf ~graph:dep ~print:P.print